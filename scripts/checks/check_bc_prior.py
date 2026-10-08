#!/usr/bin/env python3
"""Checks of the MILES-BC policy prior (``train_bc_prior.py``, ``task_common.bc_prior``,
``MppiParams.prior_mode`` / ``bc_prior``, ``mppi_lcs.PolicyPrior``, node, loop ``demo_prior.mode``).

B1 params: defaults demo / None; bad mode, policy without bc_prior, bc_prior without policy, policy
   + prior_sync, policy + stage_no_timeout_into refused; the cached solver ignores both fields.
B2 node identity (prior_mode demo): pre-change node + solver vs the tree's on recorded streams
   (FULL iter0 ep4, C1 iter1 ep0, V12-20 UR z-30 ep0): plans, log rows minus timing, stdout.
B3 forward / rollout, per prior (``--bc-prior`` inputs z, ``--bc-prior-zp`` inputs zp or a legacy
   file without the key): numpy ``BcPrior.forward`` vs torch ``PolicyPrior.forward`` on 2000
   inputs; numpy closed-loop rollout (``LearnedLcs`` float64) vs ``PolicyPrior`` (torch float32);
   a z prior ignores P (forward and rollout bit-identical under changed P), a zp prior does not;
   on CUDA also the captured graph vs eager.
B4 node with policy (V12-20 UR z-30 ep0): every passed ``u_demo`` == the numpy rollout from the
   solve's (z, p0); logged ``prior_u0`` == its first row; a deploy with another encoder refused.
B5 training: frame selection rules (demo motion + hold, miles segments / npz key / walk / refusal);
   toy fit of a known map (inputs z and zp) reaches < 10 % of the mean-baseline MSE on held-out
   episodes; the V12h
   cache latents == the deploy's numpy encoder on demo frames.
B6 loop: ``mppi_yaml`` bytes identical pre vs tree without ``mode``; with ``mode: policy`` the doc ==
   pre + prior_mode / bc_prior; ``check_cfg`` refusals.

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_bc_prior.py \\
        --pre-dir <dir: pre-change src/task_common/mppi_lcs.py, scripts/lcs/mppi_controller_node.py>
        --pre-loop <dir: pre-change scripts/lcs/realfirst_loop.py> --bc-prior <bc_prior.npz> \\
        [--bc-prior-zp <bc_prior.npz with inputs zp>]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import importlib.util
import io
import json
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # cpu: clearance off
CMP = REPO / "data/lcs/realfirst/cmp"
FULL = CMP / "full/iter0/collect"
C1 = CMP / "C1_transport/iter1/collect"
SLANT = CMP / "ur_slant/ee_ur_z-30"
C1_0_DEPLOY = CMP / "C1_transport/iter0/deploy.json"
N_LAT = 120


def load_pair(d: Path | None, tag: str):
    """``(mppi_lcs, node)`` from ``d`` (repo layout) or the tree."""
    root = REPO if d is None else d
    spec = importlib.util.spec_from_file_location(f"mppi_lcs_{tag}",
                                                  root / "src/task_common/mppi_lcs.py")
    lcs_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lcs_mod
    spec.loader.exec_module(lcs_mod)
    sys.modules["task_common.mppi_lcs"] = lcs_mod
    spec = importlib.util.spec_from_file_location(
        f"mppi_node_{tag}", root / "scripts/lcs/mppi_controller_node.py")
    node_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = node_mod
    spec.loader.exec_module(node_mod)
    return lcs_mod, node_mod


def load_script(d: Path | None, name: str, tag: str):
    """``scripts/lcs/<name>.py`` from ``d``, run as if at the tree's path."""
    path = (REPO if d is None else d) / f"scripts/lcs/{name}.py"
    mod = types.ModuleType(f"{name}_{tag}")
    mod.__file__ = str(REPO / f"scripts/lcs/{name}.py")
    sys.modules[mod.__name__] = mod
    exec(compile(path.read_text(), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def episode(run: Path, i: int) -> tuple[dict, dict]:
    idx = json.loads((run / "index.json").read_text())
    return idx["args"], idx["episodes"][i]


def run_replay(node_mod, run: Path, i: int, frag: dict, n: int | None, spy=None) -> dict:
    a, ep = episode(run, i)
    if DEVICE == "cpu":
        a, frag = {**a, "mppi_clearance": False}, {**frag, "clearance": None}
    real = node_mod.mppi_plan
    if spy is not None:
        node_mod.mppi_plan = spy(real)
    try:
        r = oo.replay(node_mod, run, a, ep, frag, n)
    finally:
        node_mod.mppi_plan = real
    for row in r["rows"]:
        row.pop("solve_ms", None)
        row.pop("prior_ms", None)
    r["stdout_clean"] = "\n".join(ln for ln in r["stdout"].splitlines()
                                  if not ln.startswith(("Module ", "Warp ", "   ")))
    return r


def deploy_of(run: Path) -> str:
    return episode(run, 0)[0]["deploy"]


def b1(tree) -> tuple[bool, str]:
    p = tree.MppiParams()
    ok = p.prior_mode == "demo" and p.bc_prior is None
    refused = 0
    bads = ({"prior_mode": "bc"}, {"prior_mode": "policy"}, {"bc_prior": "x.npz"},
            {"prior_mode": "policy", "bc_prior": "x.npz", "prior_sync": "stage"},
            {"prior_mode": "policy", "bc_prior": "x.npz", "stage_no_timeout_into": "latch"},
            {"prior_mode": None})
    for bad in bads:
        try:
            tree.MppiParams(**bad)
        except ValueError:
            refused += 1
    good = tree.MppiParams(prior_mode="policy", bc_prior="x.npz")
    lcs = tree.TorchLcs.from_deploy(deploy_of(C1), DEVICE)
    q = tree.MppiParams(K=8, H=6, n_iter=1, cuda_graph=False)
    same = tree.solver_for(lcs, q) is tree.solver_for(lcs, dataclasses.replace(
        q, prior_mode="policy", bc_prior="x.npz"))
    ok = ok and refused == len(bads) and good.prior_mode == "policy" and same
    return ok, f"defaults demo/None; {refused}/{len(bads)} bad refused; solver shared {same}"


def b2(pre_node, tree_node) -> tuple[bool, str]:
    res = []
    for name, run, i in (("FULL i0 ep4", FULL, 4), ("C1 i1 ep0", C1, 0),
                         ("V12-20 z-30 ep0", SLANT, 0)):
        a = run_replay(pre_node, run, i, {}, N_LAT)
        b = run_replay(tree_node, run, i, {}, N_LAT)
        plans = len(a["plans"]) == len(b["plans"]) and all(
            ka == kb and np.array_equal(pa, pb) for (ka, pa), (kb, pb) in
            zip(a["plans"], b["plans"]))
        same = (plans, a["rows"] == b["rows"], a["stdout_clean"] == b["stdout_clean"])
        res.append((name, all(same), len(a["plans"]), same))
    return all(r[1] for r in res), "; ".join(
        f"{n}: {k} solves, plans/rows/stdout {s}" for n, _, k, s in res)


def numpy_rollout(bc, lcs_np, params, tlcs, z, p):
    lb, ub = params.bounds(tlcs)
    return bc.rollout(lcs_np, z, p, params.H, lb, ub, params.pgd_iters)[0]


def b3(tree, bc_path: Path, zp_path: Path | None) -> tuple[bool, str]:
    from task_common.bc_prior import BcPrior
    from task_common.latent_encoder import LearnedLcs

    dep = deploy_of(SLANT)
    tl = tree.TorchLcs.from_deploy(dep, "cpu")
    prm = tree.MppiParams.from_yaml(REPO / episode(SLANT, 0)[0]["mppi_config"])
    prm = dataclasses.replace(prm, clearance=None)
    lcs_np = LearnedLcs.load(dep)
    ok, msgs = True, []
    for path, want in ((bc_path, "z"), (zp_path, "zp")):
        if path is None:
            continue
        bc = BcPrior.load(path)
        pp = tree.PolicyPrior(path, tl, prm, dep)
        rng = np.random.default_rng(0)
        Z = bc.in_mean[:bc.n_x] + bc.in_std[:bc.n_x] * rng.normal(size=(2000, bc.n_x)) * 1.5
        P = np.r_[0.37, 0.02, 0.06, 0.52, -0.03, 0.05] + 0.03 * rng.normal(size=(2000, 6))
        P2 = P + 0.05 * rng.normal(size=P.shape)
        u_np = bc.forward(Z, P)
        with torch.no_grad():
            u_t = pp.forward(tl.tensor(Z), tl.tensor(P)).double().numpy()
            u_t2 = pp.forward(tl.tensor(Z), tl.tensor(P2)).double().numpy()
        fwd = float(np.max(np.abs(u_np - u_t) / bc.u_half))
        dp = float(np.max(np.abs(u_t2 - u_t) / bc.u_half))  # P sensitivity
        roll, roll_p = 0.0, 0.0
        for k in range(20):
            un = numpy_rollout(bc, lcs_np, prm, tl, Z[k], P[k])
            ut = pp(Z[k], P[k])
            roll = max(roll, float(np.max(np.abs(un - ut) / bc.u_half)))
            roll_p = max(roll_p, float(np.max(np.abs(pp(Z[k], P2[k]) - ut) / bc.u_half)))
        ind = dp == 0.0 and roll_p == 0.0 if want == "z" else dp > 1e-3
        ok = ok and bc.inputs == want and fwd < 1e-4 and roll < 1e-3 and ind  # float32 vs 64
        msg = (f"{want} ({path.parent.name}, inputs {bc.inputs}): forward max |du|/half "
               f"{fwd:.1e} (2000 inputs), rollout H {prm.H} {roll:.1e} (20 starts), P changed: "
               f"forward {dp:.1e} rollout {roll_p:.1e}")
        if torch.cuda.is_available():
            tc = tree.TorchLcs.from_deploy(dep, "cuda")
            g = tree.PolicyPrior(path, tc, prm, dep)
            e = tree.PolicyPrior(path, tc, prm, dep, use_graph=False)
            gd = max(float(np.max(np.abs(g(Z[k], P[k]) - e(Z[k], P[k])))) for k in range(20))
            ok = ok and g.graph is not None and gd < 1e-6
            msg += f", CUDA graph vs eager {gd:.1e}"
        msgs.append(msg)
    if not torch.cuda.is_available():
        msgs.append("CUDA graph path not run (no GPU)")
    return ok, "; ".join(msgs)


def b4(tree_node, tree, bc_path: Path) -> tuple[bool, str]:
    from task_common.bc_prior import BcPrior
    from task_common.latent_encoder import LearnedLcs

    calls = []

    def spy(real):
        def fn(lcs, params, z0, p0, rot0, goal, u_prev, gen, **kw):
            calls.append((np.array(z0), np.array(p0), None if kw.get("u_demo") is None
                          else kw["u_demo"].copy(), params))
            return real(lcs, params, z0, p0, rot0, goal, u_prev, gen, **kw)
        return fn

    frag = {"prior_mode": "policy", "bc_prior": str(bc_path)}
    r = run_replay(tree_node, SLANT, 0, frag, 60, spy=spy)
    rows = r["rows"]
    calls = calls[-len(rows):]
    dep = deploy_of(SLANT)
    bc, lcs_np = BcPrior.load(bc_path), LearnedLcs.load(dep)
    tl = tree.TorchLcs.from_deploy(dep, "cpu")
    worst, log_ok = 0.0, True
    for row, (z, p, ud, prm) in zip(rows, calls):
        un = numpy_rollout(bc, lcs_np, prm, tl, z, p)
        worst = max(worst, float(np.max(np.abs(un - ud) / bc.u_half)))
        log_ok = log_ok and np.allclose(row["prior_u0"], ud[0], rtol=0, atol=1e-12)
    other = json.loads(C1_0_DEPLOY.read_text())["deploy"]
    refused = 0
    try:
        tree.PolicyPrior(bc_path, tree.TorchLcs.from_deploy(other, "cpu"), tree.MppiParams(), other)
    except ValueError:
        refused = 1
    ok = len(rows) > 10 and worst < 1e-3 and log_ok and refused == 1
    return ok, (f"{len(rows)} solves: u_demo == numpy rollout (max |du|/half {worst:.1e}), "
                f"prior_u0 logged {log_ok}; other-encoder deploy refused {refused}/1")


def b5() -> tuple[bool, str]:
    tb = load_script(None, "train_bc_prior", "tree")
    ok, notes = True, []
    acts = np.zeros((30, 12))
    acts[5:12, 1] = 1.0
    fr = tb.demo_frames(acts, 3)
    ok &= fr.tolist() == list(range(5, 15))
    seg = np.array(["walk"] * 2 + ["dev"] * 3 + ["return"] * 3 + ["hold"] * 2)
    ok &= tb.miles_frames({"episode": "x"}, {"miles_segment": seg}, 10).tolist() == [0, 1, 5, 6,
                                                                                      7, 8, 9]
    row = {"episode": "x", "segments": {"dev": [0, 4], "return": [4, 8], "hold": [8, 12]}}
    ok &= tb.miles_frames(row, {}, 10).tolist() == list(range(4, 10))
    ok &= tb.miles_frames({"episode": "x", "segment": "walk"}, {}, 4).tolist() == [0, 1, 2, 3]
    refused = 0
    for r_, npz in (({"episode": "x"}, {}), ({"episode": "x"}, {"miles_segment": seg[:2].tolist()
                                                                 + ["deviate"]})):
        try:
            tb.miles_frames(r_, npz, 3)
        except SystemExit:
            refused += 1
    ok &= refused == 2
    # toy per inputs: u = tanh(W x) in the box (x = z or [z, p]), 12 episodes
    toy = {}
    for inputs in ("z", "zp"):
        rng = np.random.default_rng(1)
        n_x, lb, ub = 10, -np.full(12, 0.006), np.full(12, 0.006)
        W = rng.normal(size=(16, 12)) * 0.5
        eps = []
        for _ in range(12):
            Z = rng.normal(size=(80, n_x))
            P = 0.3 + 0.02 * rng.normal(size=(80, 6))
            X = np.concatenate([Z, (P - 0.3) / 0.02 * (inputs == "zp")], 1)
            eps.append({"Z": Z, "P": P, "U": 0.006 * np.tanh(X @ W), "frames": np.arange(80)})
        cli = argparse.Namespace(seed=0, val_episodes=2, hidden=[128, 128], lr=1e-3, wd=1e-4,
                                 epochs=300, batch=128, noise_in=0.0, inputs=inputs)
        mid, half = (ub + lb) / 2, (ub - lb) / 2
        _net, *_, (Xtr, Ytr, _, Yva), _, best, curve = tb.train(cli, eps, mid, half, lb, ub, n_x)
        base = float(np.mean((Yva - Ytr.mean(0)) ** 2))
        lva = curve[best][2]
        toy[inputs] = (lva, base, Xtr.shape[1])
        ok &= lva < 0.1 * base and Xtr.shape[1] == n_x + 6 * (inputs == "zp")
    # cache latents == the deploy's numpy encoder (V12h, demo frames)
    from latent_stage_tols import encode_frames

    from task_common.latent_encoder import LatentEncoder

    dj = json.loads((CMP / "C1_transport/iter1/deploy.json").read_text())
    look = tb.cache_lookup(Path(dj["cache"]), Path(dj["deploy"]))
    enc = LatentEncoder.load(dj["deploy"])
    ep = str(REPO / "data/lcs/realfirst/v1_1/seed/demo/episode_0000.npz")
    with np.load(ep, allow_pickle=True) as d:
        segs = list(d["pcd_belt_seg"])
    dz = 0.0
    for t in range(20, 60, 4):
        zc = look(ep, t)
        if zc is None:
            notes.append(f"cache miss {t}")
            continue
        dz = max(dz, float(np.abs(encode_frames(enc, segs, [t])[0][0] - zc).max()))
    ok &= dz < 1e-4 and not notes
    return bool(ok), (f"frame rules ok, {refused}/2 refusals; toy val mse / baseline (n_in): "
                      + ", ".join(f"{k} {v[0]:.4f} / {v[1]:.4f} ({v[2]})" for k, v in toy.items())
                      + f"; cache vs numpy encoder max |dz| {dz:.1e}"
                      + (f"; {notes}" if notes else ""))


def b6(pre_loop, tree_loop) -> tuple[bool, str]:
    cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
        (CMP / "configs/C1_transport.yaml").read_text()))
    ok, notes = True, []
    it = CMP / "C1_transport/iter1"
    pol = {"demo_prior": {"mode": "policy", "bc_prior": "data/x/bc_prior.npz"}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        outs = {}
        for tag, mod, c in (("pre", pre_loop, cfg), ("tree", tree_loop, cfg),
                            ("policy", tree_loop, tree_loop.merge(cfg, pol))):
            p = tmp / f"{tag}.yaml"
            mod.mppi_yaml(types.SimpleNamespace(cfg=c, it=lambda i=None: it), p, 0.2, 7)
            outs[tag] = p.read_text()
        if outs["pre"] != outs["tree"]:
            ok = False
            notes.append("off bytes differ")
        dp, ds = yaml.safe_load(outs["pre"]), yaml.safe_load(outs["policy"])
        added = {k: ds.pop(k, None) for k in ("prior_mode", "bc_prior")}
        if added != {"prior_mode": "policy", "bc_prior": str(REPO / "data/x/bc_prior.npz")} \
                or dp != ds:
            ok = False
            notes.append("policy doc")
        refused = 0
        for over in ({"demo_prior": {"mode": "bc"}}, {"demo_prior": {"mode": "policy"}},
                     {"demo_prior": {**pol["demo_prior"], "sync": "stage"}},
                     {**pol, "waypoints": {"no_timeout_into": "latch"}}):
            try:
                tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg), over))
            except SystemExit:
                refused += 1
        tree_loop.check_cfg(tree_loop.merge(cfg, pol))
        ok = ok and refused == 4
    return ok, (f"off: bytes identical; policy: doc == pre + prior_mode/bc_prior; check_cfg "
                f"refusals {refused}/4" + (f"; FAIL {notes}" if notes else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True)
    ap.add_argument("--pre-loop", type=Path, required=True)
    ap.add_argument("--bc-prior", type=Path, required=True, help="inputs z")
    ap.add_argument("--bc-prior-zp", type=Path, default=None, help="inputs zp (B3)")
    ap.add_argument("--only", default=None, help="e.g. B3,B4")
    ap.add_argument("--out", type=Path, default=None, help="json report")
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (a sim may be live)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bc = a.bc_prior.resolve()
    with contextlib.redirect_stdout(io.StringIO()):
        _, pre_node = load_pair(a.pre_dir, "pre")
        tree, tree_node = load_pair(None, "tree")
    pre_loop = load_script(a.pre_loop, "realfirst_loop", "pre")
    tree_loop = load_script(None, "realfirst_loop", "tree")
    res = {}
    for name, fn in (("B1", lambda: b1(tree)), ("B2", lambda: b2(pre_node, tree_node)),
                     ("B3", lambda: b3(tree, bc, a.bc_prior_zp and a.bc_prior_zp.resolve())), ("B4", lambda: b4(tree_node, tree, bc)),
                     ("B5", b5), ("B6", lambda: b6(pre_loop, tree_loop))):
        if a.only and name not in a.only.split(","):
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
