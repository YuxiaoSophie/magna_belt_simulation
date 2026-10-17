#!/usr/bin/env python3
"""Checks of the learned cost-to-go term (``MppiParams.value``, ``task_common.latent_value``,
node ``v_now`` / ``value_H``; V12-39).

V1 params: default None; every realfirst ``mppi*.yaml`` loads with ``to_dict`` == the pre-change
   module's + ``value: None``; bad blocks refused; npz refused on another z_std / n_x / format;
   the cached solver keys on the block.
V2 forward: numpy ``LatentValue`` == torch ``value_forward`` (float64 to 1e-12, float32 to 1e-4 s)
   on random and recorded latents; == an independent numpy MLP from the npz arrays.
V3 cost: synthetic rollouts, float32 + float64: ``plan_cost`` on - off == weight V(z_H)
   (terminal) / weight sum_k V(z_k), k = 1..H (all_knots) from the numpy reference; weight 0 ==
   off bit for bit; an explicit ``vt`` == from params.
V4 solver: pre vs tree ``mppi_plan`` with the value off (cold + warm): plans and stats identical;
   weight 0 == off (plans, stats); on: the solver's ``plan_cost`` == ``rollout_cost`` of its plan,
   ``value_H`` == numpy V of the plan's last knot; with CUDA (``--cuda``): captured graph ==
   eager on the same noise.
V5 node: recorded C1 stream: off pre == tree (JSONL rows minus timing, plans, stdout); on:
   ``v_now`` == numpy V(z), ``value_H`` present, plans change; a value npz from another encoder
   is refused.

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_value_cost.py \\
        --pre-dir <dir with the pre-change mppi_lcs.py, mppi_controller_node.py> \\
        --value data/lcs/realfirst/cmp/value_cost/models/<tag>/value.npz
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import overshoot_offline as oo
import torch
from check_support_box import load_pair

from task_common import latent_value as lv
from task_common import mppi_costs
from task_common.latent_encoder import LATENT_STATE_CHANNEL, latent_state_message

C1 = REPO / "data/lcs/realfirst/cmp/C1_transport"
RUN = C1 / "iter1/collect"
DEPLOY = json.loads((C1 / "iter1/deploy.json").read_text())["deploy"]
WP = REPO / "data/lcs/realfirst/cmp/C1_transport/iter1/latent_waypoints.npz"
N_LAT = 30
LATER = (("stage_skip_ahead", False), ("stage_no_timeout_into", None), ("prior_mode", "demo"),
         ("bc_prior", None), ("goal_set", None), ("goal_set_tau", 0.0), ("stage_start", None),
         ("support_box", None), ("prior_sync", None), ("overshoot", None),
         ("latent_support", None), ("ee_support", None))


def base_doc(device: str, small: bool = False) -> dict:
    a = json.loads((RUN / "index.json").read_text())["args"]
    doc = yaml.safe_load(Path(a["mppi_config"]).read_text())
    doc.update(action_noise_frac=0.0, prior_alpha=0.0, prior_weight=0.0)
    if device == "cpu":
        doc["clearance"] = None
    if small:
        doc.update(K=256, n_iter=2)
    return doc


def recorded(n: int):
    idx = json.loads((RUN / "index.json").read_text())
    ep = idx["episodes"][0]
    rows = [json.loads(x) for x in (RUN / "controller" / Path(ep["file"]).with_suffix(
        ".solves.jsonl").name).read_text().splitlines()]
    with np.load(RUN / ep["file"], allow_pickle=True) as d:
        ut, z = np.asarray(d["utime"]), np.asarray(d["sim_latent"], np.float64)
        ef, eu = np.asarray(d["sim_ee_franka"]), np.asarray(d["sim_ee_ur"])
    k0 = int(np.flatnonzero(ut == rows[0]["utime"])[0])
    sl = slice(k0, min(len(ut), k0 + n))
    return ut[sl], z[sl], ef[sl], eu[sl], int(rows[0].get("idx", 20))


def ref_value(path, Z) -> np.ndarray:
    """Independent numpy MLP from the npz arrays."""
    with np.load(path) as f:
        x = (np.asarray(Z, np.float64) - f["z_mean"]) / f["z_std"]
        n = int(f["n_layers"])
        for i in range(n):
            x = x @ f[f"w{i}"] + f[f"b{i}"]
            if i + 1 < n:
                x = np.maximum(x, 0.0)
        return float(f["v_max"]) / (1.0 + np.exp(-x[..., 0]))


def variant(src, dst: Path, **over) -> str:
    with np.load(src) as f:
        d = {k: f[k] for k in f.files}
    d.update(over)
    np.savez(dst, **d)
    return str(dst)


def v1(pre, tree, value: str, tmp: Path) -> tuple[bool, str]:
    ok, notes, n, n_on = tree.MppiParams().value is None, [], 0, 0
    for y in sorted((REPO / "data/lcs/realfirst").rglob("mppi*.yaml")):
        doc = yaml.safe_load(y.read_text()) or {}
        if "value" in doc:
            n_on += tree.MppiParams.from_yaml(y).value is not None
            continue
        try:
            a = pre.MppiParams.from_yaml(y).to_dict()
        except ValueError:
            continue
        b = tree.MppiParams.from_yaml(y).to_dict()
        for k, v in LATER:
            if k in b and b[k] == v and k not in a:
                b.pop(k)
        n += 1
        if b.pop("value", "missing") is not None or a != b:
            ok = False
            notes.append(f"diff {y}")
    good = {"path": value, "weight": 1.0}
    bads = [{"weight": 1.0}, {"path": value}, {**good, "weight": -1.0},
            {**good, "weight": math.nan}, {**good, "weight": math.inf}, {**good, "where": "last"},
            {**good, "other": 1}, {**good, "path": 3}, "x"]
    refused = 0
    for bad in bads:
        try:
            tree.MppiParams(value=bad)
        except (ValueError, TypeError):
            refused += 1
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu")
    with np.load(value) as f:
        z_std, z_mean = f["z_std"], f["z_mean"]
    npz_bad = [variant(value, tmp / "zs.npz", z_std=z_std * 1.001),
               variant(value, tmp / "nx.npz", z_std=z_std[:-1], z_mean=z_mean[:-1]),
               variant(value, tmp / "fmt.npz", format="other")]
    npz_refused = 0
    for p in npz_bad:
        try:
            tree.MppiSolver(lcs, tree.MppiParams(**{**base_doc("cpu", True),
                                                    "value": {"path": p, "weight": 1.0}}),
                            use_graph=False)
        except ValueError:
            npz_refused += 1
    p1 = tree.MppiParams(**{**base_doc("cpu", True), "value": good})
    p2 = dataclasses.replace(p1, value={**good, "weight": 2.0})
    keyed = tree.solver_for(lcs, p1) is not tree.solver_for(lcs, p2) \
        and tree.solver_for(lcs, p1) is tree.solver_for(lcs, dataclasses.replace(p1))
    ok = ok and refused == len(bads) and npz_refused == len(npz_bad) and keyed and n > 0
    return ok, (f"{n} yamls: to_dict == pre + value None ({n_on} with value); {refused}/"
                f"{len(bads)} bad blocks, {npz_refused}/{len(npz_bad)} bad npz refused; solver "
                f"keyed on the block {keyed}" + (f"; FAIL {notes[:3]}" if notes else ""))


def v2(value: str) -> tuple[bool, str]:
    vf = lv.LatentValue.load(value)
    rng = np.random.default_rng(0)
    _, z, *_ = recorded(200)
    Zs = [z, vf.z_mean + vf.z_std * rng.normal(scale=3.0, size=(500, vf.n_x))]
    err64 = err32 = err_ref = 0.0
    for Z in Zs:
        a = vf(Z)
        err_ref = max(err_ref, float(np.abs(a - ref_value(value, Z)).max()))
        for dt, name in ((torch.float64, "64"), (torch.float32, "32")):
            t = lambda x, dt=dt: torch.as_tensor(np.asarray(x), dtype=dt)
            b = lv.value_forward(vf.torch_params(t), t(Z)).double().numpy()
            e = float(np.abs(a - b).max())
            if name == "64":
                err64 = max(err64, e)
            else:
                err32 = max(err32, e)
    ok = err64 < 1e-12 and err32 < 1e-4 and err_ref < 1e-12
    return ok, (f"numpy vs torch float64 {err64:.1e} s, float32 {err32:.1e} s, vs independent "
                f"numpy {err_ref:.1e} s ({len(Zs[0])} recorded + 500 random latents)")


def v3(tree, value: str) -> tuple[bool, str]:
    vf = lv.LatentValue.load(value)
    rng = np.random.default_rng(1)
    worst64 = worst32 = 0.0
    n, exact0, same_vt = 0, True, True
    for dtype in (torch.float64, torch.float32):
        lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu", dtype)
        for where in ("terminal", "all_knots"):
            for w in (0.0, 0.5, 3.0):
                prm = tree.MppiParams(**{**base_doc("cpu", True),
                                         "value": {"path": value, "weight": w, "where": where}})
                off = dataclasses.replace(prm, value=None)
                K, H = 64, prm.H
                Z = vf.z_mean + vf.z_std * rng.normal(size=(K, H + 1, lcs.n_x))
                P = np.tile(np.r_[0.37, 0.02, 0.08, 0.51, -0.03, 0.06], (K, H + 1, 1))
                R = np.zeros((K, H + 1, 2))
                U = rng.normal(scale=1e-3, size=(K, H, lcs.n_u))
                half = tree._box(lcs, prm).half
                tz, tP, tR, tU = (lcs.tensor(x) for x in (Z, P, R, U))
                goal = lcs.tensor(vf.z_mean)
                c_on, _, _ = tree.plan_cost(lcs, prm, tz, tP, tR, tU, goal, half, None)
                c_off, _, _ = tree.plan_cost(lcs, off, tz, tP, tR, tU, goal, half, None)
                vt = mppi_costs.value_tensors(lcs, prm.value)
                c_vt, _, _ = tree.plan_cost(lcs, prm, tz, tP, tR, tU, goal, half, None,
                                            terms=mppi_costs.CostTerms(value=vt))
                same_vt = same_vt and torch.equal(c_on, c_vt)
                Zr = tz.double().numpy()  # the dtype-rounded latents
                ref = w * (vf(Zr[:, -1]) if where == "terminal" else vf(Zr[:, 1:]).sum(1))
                if w == 0.0:
                    exact0 = exact0 and torch.equal(c_on, c_off)
                else:
                    e = float(np.abs((c_on - c_off).double().numpy() - ref).max()
                              / max(1.0, float(c_on.abs().max())))
                    if dtype == torch.float64:
                        worst64 = max(worst64, e)
                    else:
                        worst32 = max(worst32, e)
                n += 1
    ok = worst64 < 1e-12 and worst32 < 1e-5 and exact0 and same_vt
    return ok, (f"{n} cases (float64 / float32 x terminal / all_knots x weight 0 / 0.5 / 3): on "
                f"- off == weight x numpy V (error / max cost: float64 {worst64:.1e}, float32 "
                f"{worst32:.1e}); weight 0 == off bit for bit {exact0}; explicit vt == from "
                f"params {same_vt}")


def solve(mod, doc, z, p0, goal, device="cpu", use_graph=None, seed=3):
    lcs = mod.TorchLcs.from_deploy(DEPLOY, device)
    prm = mod.MppiParams(**doc)
    s = mod.MppiSolver(lcs, prm, use_graph=use_graph)
    gen = torch.Generator(device=device).manual_seed(seed)
    U1, i1 = s.solve(z[0], p0, None, goal, None, gen, mod.safety_limits(prm))
    U2, i2 = s.solve(z[1], p0, None, goal, U1, gen, mod.safety_limits(prm))
    info = {k: v for k, v in {**i1, **{f"w_{k}": v for k, v in i2.items()}}.items()
            if "solve_ms" not in k}
    return U1, U2, info, lcs, prm


def same(x, y, skip=()) -> bool:
    ok = np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
    for k in x[2]:
        if k in skip:
            continue
        va, vb = x[2][k], y[2].get(k)
        ok = ok and (np.array_equal(va, vb) if isinstance(va, np.ndarray) else
                     (va == vb or (isinstance(va, float) and math.isnan(va) and math.isnan(vb))))
    return ok


def v4(pre, tree, value: str, cuda: bool) -> tuple[bool, str]:
    _, z, ef, eu, _ = recorded(5)
    p0 = np.r_[ef[0, :3], eu[0, :3]]
    with np.load(WP) as f:
        goal = np.asarray(f["z_wp"][-1])
    doc = base_doc("cpu")
    off_pre, off_tree = solve(pre, doc, z, p0, goal), solve(tree, doc, z, p0, goal)
    w0 = solve(tree, {**doc, "value": {"path": value, "weight": 0.0}}, z, p0, goal)
    ok_off = same(off_pre, off_tree) and same(off_tree, w0) and "value_H" not in off_tree[2]
    msgs = [f"off pre == tree {same(off_pre, off_tree)}, weight 0 == off {same(off_tree, w0)}"]
    ok = ok_off
    vf = lv.LatentValue.load(value)
    for where, w in (("terminal", 2.0), ("all_knots", 0.2)):
        U1, _, info, lcs, prm = solve(tree, {**doc, "value": {"path": value, "weight": w,
                                                              "where": where}}, z, p0, goal)
        c, nb, _ = tree.rollout_cost(lcs, prm, z[0], p0, None, goal, U1)
        rel = abs(c - info["plan_cost"]) / max(abs(c), 1.0)
        dv = abs(info["value_H"] - float(vf(info["Z"][-1])))
        changed = not np.array_equal(U1, off_tree[0])
        ok = ok and rel < 1e-4 and nb == info["plan_violations"] and dv < 1e-9 and changed
        msgs.append(f"{where} w {w:g}: plan_cost == rollout_cost (rel {rel:.1e}), value_H == "
                    f"numpy ({dv:.1e}), plan changed {changed}")
    if cuda and torch.cuda.is_available():
        d = {**doc, "value": {"path": value, "weight": 2.0}, "K": 256}
        g = solve(tree, d, z, p0, goal, "cuda", True)
        e = solve(tree, {**d, "fused_rollout": True}, z, p0, goal, "cuda", False)
        du = float(np.abs(g[0] - e[0]).max())
        ok = ok and du < 1e-6 and abs(g[2]["value_H"] - e[2]["value_H"]) < 1e-5
        msgs.append(f"CUDA graph == eager (max |dU| {du:.1e})")
    return ok, "; ".join(msgs)


def stream(node_mod, doc: dict, n: int) -> dict:
    ut, z, ef, eu, offset = recorded(n)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, log = Path(tmp) / "mppi.yaml", Path(tmp) / "solves.jsonl"
        cfg.write_text(yaml.safe_dump(doc))
        argv = ["--lcm-url", "fake", "--deploy", DEPLOY, "--config", str(cfg),
                "--max-duration-s", "30", "--log", str(log), "--goal", f"waypoints:{WP}",
                "--stages", "all", "--goal-mode", "stage", "--demo-offset-frames", str(offset),
                "--demo-actions", "auto", "--device", "cpu"]
        args = node_mod.create_parser().parse_args(argv)
        out, plans = io.StringIO(), []
        with contextlib.redirect_stdout(out):
            node = node_mod.MppiControllerNode(args, oo.FakeLc())
            node.start()
            for k in range(len(ut)):
                msg = latent_state_message(int(ut[k]), ut[k] * 1e-6, z[k], ef[k], eu[k],
                                           np.zeros(40))
                n0 = node.n_solves
                node.lc.cb(LATENT_STATE_CHANNEL, msg.encode())
                node._process(*node.pending)
                node.pending = None
                if node.n_solves > n0:
                    plans.append(node.plan.copy())
            node.log.close()
        rows = [json.loads(x) for x in log.read_text().splitlines()]
    for r in rows:
        r.pop("solve_ms", None)
    stdout = "\n".join(ln for ln in out.getvalue().splitlines()
                       if not ln.startswith(("Module ", "Warp ", "   ")))
    return {"rows": rows, "plans": plans, "stdout": stdout, "z": z}


def v5(pre_node, tree_node, value: str, tmp: Path) -> tuple[bool, str]:
    doc = base_doc("cpu")
    a, b = stream(pre_node, doc, N_LAT), stream(tree_node, doc, N_LAT)
    same_off = (a["rows"] == b["rows"] and len(a["plans"]) == len(b["plans"])
                and all(np.array_equal(x, y) for x, y in zip(a["plans"], b["plans"]))
                and a["stdout"] == b["stdout"])
    vf = lv.LatentValue.load(value)
    on = stream(tree_node, {**doc, "value": {"path": value, "weight": 2.0}}, N_LAT)
    ut, z, *_ = recorded(N_LAT)
    zmap = {int(u): zz for u, zz in zip(ut, z)}
    dv = max(abs(r["v_now"] - float(vf(zmap[r["utime"]]))) for r in on["rows"])
    has_h = all("value_H" in r for r in on["rows"]) and all("v_now" not in r for r in b["rows"])
    changed = sum(not np.array_equal(x, y) for x, y in zip(b["plans"], on["plans"]))
    bad = variant(value, tmp / "enc.npz", encoder_sha256="0" * 64)
    refused = False
    try:
        stream(tree_node, {**doc, "value": {"path": bad, "weight": 2.0}}, 2)
    except ValueError as exc:
        refused = "another encoder" in str(exc)
    ok = same_off and dv < 1e-9 and has_h and changed > 0 and refused
    return ok, (f"off pre == tree {same_off} ({len(b['rows'])} solves: rows, plans, stdout); on: "
                f"v_now == numpy V(z) (max {dv:.1e}), value_H logged {has_h}, {changed}/"
                f"{len(on['plans'])} plans changed; other encoder refused "
                f"{refused}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--cuda", action="store_true", help="V4: also the captured graph (GPU)")
    ap.add_argument("--out", type=Path, default=None, help="json report")
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    pre_lcs, pre_node = load_pair(a.pre_dir, "pre")
    tree_lcs, tree_node = load_pair(None, "tree")
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, fn in (("V1", lambda: v1(pre_lcs, tree_lcs, a.value, tmp)),
                         ("V2", lambda: v2(a.value)), ("V3", lambda: v3(tree_lcs, a.value)),
                         ("V4", lambda: v4(pre_lcs, tree_lcs, a.value, a.cuda)),
                         ("V5", lambda: v5(pre_node, tree_node, a.value, tmp))):
            if a.only and name not in a.only:
                continue
            ok, msg = fn()
            res[name] = {"ok": bool(ok), "msg": msg}
            print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    n = sum(r["ok"] for r in res.values())
    print(f"{n}/{len(res)} PASS")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
