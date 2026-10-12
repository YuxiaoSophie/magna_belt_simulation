#!/usr/bin/env python3
"""Checks of the data-support box (``MppiParams.support_box``, node ``sbox_mm``,
``realfirst_loop.py`` ``support_box``).

B1 params: default None; every realfirst ``mppi*.yaml`` loads with ``to_dict`` == the pre-change
   module's plus ``support_box: None``; bad blocks refused; the cached solver keys on the box.
B2 cost: synthetic P inside / outside / both sides / knot 0 outside -> ``plan_cost`` (float64)
   on - off == weight sum_{knots 1..H, coords} (hinge mm)^2 (numpy); hard_margin_m: knots
   beyond it outside count in ``n_bad``, at it not; an explicit ``sbox`` tensor == from params.
B3 solver: pre vs tree ``mppi_plan`` with the box off (cold + warm, with / without a prior):
   plans and stats identical; a box containing every knot (on) == off exactly; the solver's
   ``plan_cost`` == ``rollout_cost`` of its plan with the box biting (one ``plan_cost`` serves
   the torch rollout and the fused Warp rollout, which returns Z / P / R only; on CUDA also the
   captured graph vs eager).
B4 node: recorded C1 stream with the box on: JSONL ``sbox_mm`` == [hinge of p0, max hinge of
   the plan's knots] (recomputed), absent with the box off; a box the measured P is 20 mm
   outside: no crash, the plan heads back (terminal hinge < now).
B5 loop: ``mppi_yaml`` bytes identical pre vs tree without ``support_box`` (C1, full2_gate);
   with it the doc == pre + the box, ``support_box.json`` lo / hi == an independent recompute
   from B_i's episodes; ``check_cfg`` refusals.

Clearance is off in B3 / B4 on the CPU (its torch path is minutes per solve there).

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_support_box.py \\
        --pre-dir <dir with the pre-change mppi_lcs.py, mppi_controller_node.py, realfirst_loop.py>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import importlib.util
import io
import json
import math
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import overshoot_offline as oo
import torch

from task_common.latent_encoder import LATENT_STATE_CHANNEL, latent_state_message

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C1 = REPO / "data/lcs/realfirst/cmp/C1_transport/iter1/collect"
F2G = REPO / "data/lcs/realfirst/cmp/full2_gate"
N_LAT = 30
SB_CFG = {"kinds": ["demo", "success"], "lo_pct": 0, "hi_pct": 100, "margin_mm": 15,
          "weight": 1.0, "hard_margin_mm": None}


def load_pair(d: Path | None, tag: str):
    """``(mppi_lcs, node)`` from ``d`` (flat: ``mppi_lcs.py`` + ``mppi_controller_node.py``)."""
    src = ({"lcs": REPO / "src/task_common/mppi_lcs.py",
            "node": REPO / "scripts/lcs/mppi_controller_node.py"} if d is None else
           {"lcs": d / "mppi_lcs.py", "node": d / "mppi_controller_node.py"})
    spec = importlib.util.spec_from_file_location(f"mppi_lcs_{tag}", src["lcs"])
    lcs_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lcs_mod
    spec.loader.exec_module(lcs_mod)
    sys.modules["task_common.mppi_lcs"] = lcs_mod
    spec = importlib.util.spec_from_file_location(f"mppi_node_{tag}", src["node"])
    node_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = node_mod
    spec.loader.exec_module(node_mod)
    return lcs_mod, node_mod


def load_loop(d: Path | None, tag: str):
    path = REPO / "scripts/lcs/realfirst_loop.py" if d is None else d / "realfirst_loop.py"
    mod = types.ModuleType(f"realfirst_loop_{tag}")
    mod.__file__ = str(REPO / "scripts/lcs/realfirst_loop.py")
    sys.modules[mod.__name__] = mod
    exec(compile(path.read_text(), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def c1_args() -> tuple[dict, dict]:
    idx = json.loads((C1 / "index.json").read_text())
    return idx["args"], idx["episodes"][0]


def base_doc(no_prior: bool = False) -> dict:
    a, _ = c1_args()
    doc = yaml.safe_load(Path(a["mppi_config"]).read_text())
    doc["action_noise_frac"] = 0.0
    if DEVICE == "cpu":
        doc["clearance"] = None
    if no_prior:
        doc.update(prior_alpha=0.0, prior_weight=0.0)
    return doc


def recorded(n: int):
    _, ep = c1_args()
    rows = [json.loads(x) for x in (C1 / "controller" / Path(ep["file"]).with_suffix(
        ".solves.jsonl").name).read_text().splitlines()]
    with np.load(C1 / ep["file"], allow_pickle=True) as d:
        ut, z = np.asarray(d["utime"]), np.asarray(d["sim_latent"], np.float64)
        ef, eu = np.asarray(d["sim_ee_franka"]), np.asarray(d["sim_ee_ur"])
    k0 = int(np.flatnonzero(ut == rows[0]["utime"])[0])
    sl = slice(k0, min(len(ut), k0 + n))
    return ut[sl], z[sl], ef[sl], eu[sl], int(rows[0].get("idx", 20))


def b1(pre, tree) -> tuple[bool, str]:
    ok, notes = tree.MppiParams().support_box is None, []
    yamls = sorted((REPO / "data/lcs/realfirst").rglob("mppi*.yaml"))
    n_on = 0
    for y in yamls:
        if {"stage_skip_ahead", "stage_no_timeout_into", "prior_mode", "bc_prior", "goal_set",
                "goal_set_tau", "stage_start", "value"} & set(yaml.safe_load(y.read_text()) or {}):
            continue  # schedule / policy-prior keys: tree only
        if "support_box" in (yaml.safe_load(y.read_text()) or {}):  # box on: tree only
            n_on += tree.MppiParams.from_yaml(y).support_box is not None
            continue
        a, b = pre.MppiParams.from_yaml(y).to_dict(), tree.MppiParams.from_yaml(y).to_dict()
        for k, v in (("stage_skip_ahead", False), ("stage_no_timeout_into", None),
                     ("prior_mode", "demo"), ("bc_prior", None),
                     ("goal_set", None), ("goal_set_tau", 0.0), ("stage_start", None),
                     ("value", None), ("latent_support", None), ("ee_support", None)):  # later
            if k in b and b[k] == v and k not in a:
                b.pop(k)
        if b.pop("support_box", "missing") is not None or a.pop("support_box", None) is not None \
                or a != b:
            ok = False
            notes.append(f"diff {y}")
    good = {"lo": [0.0] * 6, "hi": [1.0] * 6, "weight": 1.0}
    bad = [{"lo": [0.0] * 6, "hi": [1.0] * 6}, {**good, "lo": [0.0] * 5},
           {**good, "lo": [2.0] + [0.0] * 5}, {**good, "weight": -1.0},
           {**good, "hard_margin_m": -0.001}, {**good, "other": 1},
           {**good, "hi": [math.nan] + [1.0] * 5}, {**good, "hi": [math.inf] * 6}]
    refused = 0
    for sb in bad:
        try:
            tree.MppiParams(support_box=sb)
        except ValueError:
            refused += 1
    ok = ok and refused == len(bad)
    tree.MppiParams(support_box={**good, "hard_margin_m": None})
    tree.MppiParams(support_box={**good, "hard_margin_m": 0.005})
    lcs = tree.TorchLcs.from_deploy(c1_args()[0]["deploy"], DEVICE)
    p = tree.MppiParams(K=8, H=6, n_iter=1, cuda_graph=False)
    s0 = tree.solver_for(lcs, p)
    s1 = tree.solver_for(lcs, dataclasses.replace(p, support_box=good))
    s2 = tree.solver_for(lcs, dataclasses.replace(p, support_box={**good, "weight": 2.0}))
    keyed = s0 is not s1 and s1 is not s2 and s0.s_sbox is None \
        and torch.equal(s1.s_sbox, lcs.tensor(tree.support_limits(good)))
    return ok and keyed, (f"{len(yamls) - n_on} yamls: to_dict == pre + support_box None; {n_on} "
                          f"with the box load; {refused}/{len(bad)} bad blocks refused; solver "
                          f"keyed on the box {keyed}" + (f"; {notes[:3]}" if notes else ""))


def b2(tree) -> tuple[bool, str]:
    a, _ = c1_args()
    lcs = tree.TorchLcs.from_deploy(a["deploy"], "cpu", torch.float64)
    rng = np.random.default_rng(0)
    K, H = 6, 5
    lo = np.array([0.30, -0.05, 0.03, 0.49, -0.07, 0.02])
    hi = np.array([0.40, 0.15, 0.11, 0.54, 0.00, 0.12])
    mid = 0.5 * (lo + hi)
    P = np.repeat(mid[None, None], K, 0).repeat(H + 1, 1).copy()
    P[1, 3, 1] = hi[1] + 0.004  # Franka y 4 mm above
    P[2, 1:, 1] = hi[1] + np.linspace(0.001, 0.010, H)  # sweep out
    P[3, 2, 0], P[3, 4, 5] = lo[0] - 0.002, hi[5] + 0.0075  # two coords, both sides
    P[4, 0, 1] = hi[1] + 0.050  # knot 0 (the measured P) outside: not costed
    P[5, 1:, 4] = lo[4] - 0.003  # UR y 3 mm below at every knot
    Z = lcs.tensor(rng.normal(size=(K, H + 1, lcs.n_x)))
    U = lcs.tensor(rng.normal(size=(K, H, 12)) * 1e-3)
    R = lcs.tensor(np.zeros((K, H + 1, 2)))
    Pt = lcs.tensor(P)
    goal = lcs.tensor(rng.normal(size=lcs.n_x))
    half = lcs.tensor(np.full(12, 0.003))
    hinge = np.maximum(lo - P[:, 1:], 0.0) + np.maximum(P[:, 1:] - hi, 0.0)
    want = (hinge * 1e3) ** 2
    ok, notes = True, []
    p_off = tree.MppiParams(K=K, H=H)
    c_off, n_off, _ = tree.plan_cost(lcs, p_off, Z, Pt, R, U, goal, half, None)
    for w, hm in ((1.0, None), (0.37, None), (2.0, 0.005), (1.0, 0.0075)):
        sb = {"lo": lo.tolist(), "hi": hi.tolist(), "weight": w, "hard_margin_m": hm}
        p_on = dataclasses.replace(p_off, support_box=sb)
        c_on, n_on, _ = tree.plan_cost(lcs, p_on, Z, Pt, R, U, goal, half, None)
        c_ex, n_ex, _ = tree.plan_cost(lcs, p_on, Z, Pt, R, U, goal, half, None,
                                       sbox=lcs.tensor(tree.support_limits(sb)))
        extra_bad = 0.0 if hm is None else (hinge > hm).any(-1).sum(-1)
        exp = w * want.sum((1, 2)) + p_off.safety_penalty * extra_bad
        got = (c_on - c_off).numpy()
        if not np.allclose(got, exp, rtol=1e-9, atol=1e-7):
            ok = False
            notes.append(f"w {w} hm {hm}: {got} vs {exp}")
        if not np.array_equal((n_on - n_off).numpy(), np.asarray(extra_bad, float) + 0 * got):
            ok = False
            notes.append(f"n_bad w {w} hm {hm}: {(n_on - n_off).numpy()} vs {extra_bad}")
        if not (torch.equal(c_on, c_ex) and torch.equal(n_on, n_ex)):
            ok = False
            notes.append("explicit sbox != params")
    inside0 = (c_on - c_off).numpy()[[0, 4]]
    nb5 = int((hinge > 0.005).any(-1).sum())
    return ok and np.all(inside0 == 0.0) and nb5 == 4, (  # sweep 5.5 / 7.75 / 10, 7.5 mm
        f"6 synthetic plans x 4 (weight, hard margin): on - off == weight sum (hinge mm)^2 "
        f"(+ penalty x knots beyond the margin: {nb5} knots at 5 mm), knot 0 and inside cost 0, "
        f"explicit sbox == params; e.g. 4 mm at one knot -> {want[1].sum():.0f}, sweep 1..10 mm "
        f"-> {want[2].sum():.1f}" + (f"; FAIL {notes[:3]}" if notes else ""))


def b3(pre, tree) -> tuple[bool, str]:
    a, _ = c1_args()
    _, z, ef, eu, offset = recorded(5)
    p0 = np.r_[ef[0, :3], eu[0, :3]]
    q = np.stack([ef[0, 3:7], eu[0, 3:7]])
    from latent_waypoints import Waypoints

    wp = Waypoints.load(REPO / a["mppi_waypoints"])

    def solve(mod, doc, sb=None, seed=3):
        lcs = mod.TorchLcs.from_deploy(a["deploy"], DEVICE)
        prm = mod.MppiParams(**doc, **({} if sb is None else {"support_box": sb}))
        if prm.clearance is not None:
            prm = dataclasses.replace(prm, clearance={**prm.clearance,
                                                      "model": str(C1 / "clearance_model.npz")})
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        kw = {"q_now": q} if prm.clearance else {}
        ud = wp.u_seq(offset + 14, prm.H) if prm.prior_weight > 0 else None
        U1, i1 = mod.mppi_plan(lcs, prm, z[0], p0, None, wp.z_wp[3], None, gen, u_demo=ud, **kw)
        U2, i2 = mod.mppi_plan(lcs, prm, z[1], p0, None, wp.z_wp[3], U1, gen, u_demo=ud, **kw)
        info = {k: v for k, v in {**i1, **{f"w_{k}": v for k, v in i2.items()}}.items()
                if "solve_ms" not in k}
        return U1, U2, info, lcs, prm, ud

    def same(x, y):
        ok = np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
        for k in x[2]:
            va, vb = x[2][k], y[2][k]
            ok = ok and (np.array_equal(va, vb) if isinstance(va, np.ndarray) else
                         (va == vb or (isinstance(va, float) and math.isnan(va)
                                       and math.isnan(vb))))
        return ok

    ok, n = True, 0
    big = {"lo": (p0 - 10.0).tolist(), "hi": (p0 + 10.0).tolist(), "weight": 5.0,
           "hard_margin_m": 0.001}
    for doc in (base_doc(), base_doc(no_prior=True)):
        off_pre, off_tree = solve(pre, doc), solve(tree, doc)
        ok = ok and same(off_pre, off_tree) and same(off_tree, solve(tree, doc, big))
        n += 1
    # a box biting the plan: Franka y hi 5 mm below the measured y
    hi = (p0 + 0.2).copy()
    hi[1] = p0[1] - 0.005
    tight = {"lo": (p0 - 0.2).tolist(), "hi": hi.tolist(), "weight": 1.0, "hard_margin_m": None}
    U1, _, info, lcs, prm, ud = solve(tree, base_doc(), tight)
    c, nb, _ = tree.rollout_cost(lcs, prm, z[0], p0, None, wp.z_wp[3], U1, q_now=q, u_demo=ud)
    rel = abs(c - info["plan_cost"]) / max(abs(c), 1.0)
    P = info["P"]
    hinge_mm = (np.maximum(P[1:] - hi, 0) + np.maximum(np.asarray(tight["lo"]) - P[1:], 0)) * 1e3
    ok = ok and rel < 1e-4 and nb == info["plan_violations"] and hinge_mm.max() > 0.0
    graph = "n/a (CPU)"
    if DEVICE == "cuda":
        ge = []
        for use in (True, False):
            s = tree.MppiSolver(lcs, prm, use_graph=use)
            gen = torch.Generator(device=DEVICE).manual_seed(5)
            ge.append(s.solve(z[0], p0, None, wp.z_wp[3], None, gen, q_now=q, u_demo=ud)[0])
        gok = np.allclose(ge[0], ge[1], atol=1e-6)
        graph = f"graph == eager {gok}"
        ok = ok and gok
    return ok, (f"{n} configs x cold + warm: box off pre == tree, and a box containing every knot "
                f"== off (exact); box biting (F y hi 5 mm below P): solver plan_cost "
                f"{info['plan_cost']:.3f} == rollout_cost {c:.3f} (rel {rel:.1e}), plan max "
                f"hinge {hinge_mm.max():.2f} mm; {graph}")


def stream(node_mod, doc: dict, n: int = N_LAT) -> dict:
    a, _ = c1_args()
    ut, z, ef, eu, offset = recorded(n)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, log = Path(tmp) / "mppi.yaml", Path(tmp) / "solves.jsonl"
        cfg.write_text(yaml.safe_dump(doc))
        argv = ["--lcm-url", "fake", "--deploy", a["deploy"], "--config", str(cfg),
                "--max-duration-s", "30", "--log", str(log),
                "--goal", f"waypoints:{REPO / a['mppi_waypoints']}", "--stages", "all",
                "--goal-mode", "stage", "--demo-offset-frames", str(offset),
                "--demo-actions", "auto"]
        if doc.get("clearance"):
            argv += ["--clearance-model", str(C1 / "clearance_model.npz")]
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
    return {"rows": rows, "plans": plans, "stdout": out.getvalue()}


def b4(tree, tree_node) -> tuple[bool, str]:
    _, _, ef, eu, _ = recorded(1)
    p0 = np.r_[ef[0, :3], eu[0, :3]]
    sb = {"lo": (p0 - 0.05).tolist(), "hi": (p0 + 0.05).tolist(), "weight": 1.0,
          "hard_margin_m": None}
    lo, hi = np.asarray(sb["lo"]), np.asarray(sb["hi"])
    ok, notes = True, []
    off = stream(tree_node, base_doc())
    ok = ok and all("sbox_mm" not in r for r in off["rows"])
    on = stream(tree_node, {**base_doc(), "support_box": sb})
    for r, U in zip(on["rows"], on["plans"]):
        pp = np.asarray(r["p0"])
        P = pp + np.cumsum(np.r_[np.zeros((1, 6)), U[:, :6]], axis=0)
        h = lambda x: float(np.max(np.maximum(lo - x, 0) + np.maximum(x - hi, 0)) * 1e3)
        if abs(r["sbox_mm"][0] - h(pp)) > 1e-9 or abs(r["sbox_mm"][1] - h(P[1:])) > 1e-3:
            ok = False
            notes.append((r["t"], r["sbox_mm"], h(pp), h(P[1:])))
    # measured P already 20 mm outside (Franka y hi below it)
    hi2 = (p0 + 0.05).copy()
    hi2[1] = p0[1] - 0.020
    out = stream(tree_node, {**base_doc(), "support_box": {**sb, "hi": hi2.tolist(),
                                                           "lo": (p0 - 0.05).tolist()}}, 8)
    r0 = out["rows"][0]
    back = r0["sbox_mm"][1] < r0["sbox_mm"][0] and all(x["solve_ok"] == 1.0 for x in out["rows"])
    u0y = np.mean([x["u0"][1] for x in out["rows"]]) * 1e3
    ok = ok and back and u0y < 0.0
    return ok, (f"off: no sbox_mm in {len(off['rows'])} rows; on: {len(on['rows'])} rows sbox_mm "
                f"== recomputed; P 20 mm outside: no crash, plan max hinge "
                f"{r0['sbox_mm'][1]:.1f} < now {r0['sbox_mm'][0]:.1f} mm, u0 y mean {u0y:+.2f} mm"
                + (f"; FAIL {notes[:3]}" if notes else ""))


def b5(pre_loop, tree_loop) -> tuple[bool, str]:
    ok, notes, res = True, [], []
    for cell in ("C1_transport", "full2_gate"):
        cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
            (REPO / f"data/lcs/realfirst/cmp/configs/{cell}.yaml").read_text()))
        root = REPO / f"data/lcs/realfirst/cmp/{cell}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "goal.json").write_bytes((root / "iter1/goal.json").read_bytes())

            def ctx_for(c, tmp=tmp, root=root):
                return types.SimpleNamespace(
                    cfg=c, i=1, seed_buffer=REPO / c["seed_buffer_merged"],
                    it=lambda i=None: tmp if i is None or i == 1 else root / f"iter{i}")

            outs = {}
            with contextlib.redirect_stdout(io.StringIO()):
                for tag, mod, c in (("pre", pre_loop, cfg), ("tree", tree_loop, cfg),
                                    ("on", tree_loop, tree_loop.merge(cfg,
                                                                      {"support_box": SB_CFG}))):
                    p = tmp / f"{tag}.yaml"
                    mod.mppi_yaml(ctx_for(c), p, 0.2, 7)
                    outs[tag] = p.read_text()
            if outs["pre"] != outs["tree"]:
                ok = False
                notes.append(f"{cell} off bytes differ")
            dp, do = yaml.safe_load(outs["pre"]), yaml.safe_load(outs["on"])
            sbd = do.pop("support_box", None)
            sj = json.loads((tmp / "support_box.json").read_text())
            # independent recompute from B_1 = iter0/buffer.json
            ps = []
            for r in json.loads((root / "iter0/buffer.json").read_text())["rollouts"]:
                if r["kind"] == "demo" or r["label_human"]:
                    with np.load(r["episode"], allow_pickle=True) as d:
                        ps.append(np.c_[d["sim_ee_franka"][:, :3], d["sim_ee_ur"][:, :3]])
            pa = np.concatenate(ps)
            lo, hi = pa.min(0) - 0.015, pa.max(0) + 0.015
            good = (dp == do and sbd is not None and np.allclose(sbd["lo"], lo, atol=1e-12)
                    and np.allclose(sbd["hi"], hi, atol=1e-12) and sbd["weight"] == 1.0
                    and sbd["hard_margin_m"] is None and sj["lo"] == sbd["lo"]
                    and sj["n_frames"] == len(pa))
            if not good:
                ok = False
                notes.append(f"{cell} on doc / box")
            res.append(f"{cell}: off bytes == pre; on: doc == pre + box (Franka y "
                       f"{lo[1] * 1e3:.0f}..{hi[1] * 1e3:.0f} mm, {len(pa)} frames)")
    refused = 0
    cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
        (REPO / "data/lcs/realfirst/cmp/configs/C1_transport.yaml").read_text()))
    for over in ({"kinds": ["demos"]}, {"kinds": []}, {"lo_pct": 50, "hi_pct": 50},
                 {"margin_mm": -1}, {"weight": -1}, {"hard_margin_mm": -1}):
        try:
            tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg),
                                                {"support_box": {**SB_CFG, **over}}))
        except SystemExit:
            refused += 1
    tree_loop.check_cfg(tree_loop.merge(cfg, {"support_box": SB_CFG}))
    ok = ok and refused == 6
    return ok, "; ".join(res) + f"; check_cfg refusals {refused}/6" + (
        f"; FAIL {notes}" if notes else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="json report")
    a = ap.parse_args()
    pre_lcs, _ = load_pair(a.pre_dir, "pre")
    tree_lcs, tree_node = load_pair(None, "tree")
    pre_loop, tree_loop = load_loop(a.pre_dir, "pre"), load_loop(None, "tree")
    res = {}
    for name, fn in (("B1", lambda: b1(pre_lcs, tree_lcs)), ("B2", lambda: b2(tree_lcs)),
                     ("B3", lambda: b3(pre_lcs, tree_lcs)),
                     ("B4", lambda: b4(tree_lcs, tree_node)),
                     ("B5", lambda: b5(pre_loop, tree_loop))):
        ok, msg = fn()
        res[name] = {"ok": bool(ok), "msg": msg}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": DEVICE, **res}, indent=1) + "\n")
    n = sum(r["ok"] for r in res.values())
    print(f"{n}/{len(res)} PASS")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
