#!/usr/bin/env python3
"""Checks of the MPPI smoothing options (V12-104): ``noise_beta`` (AR(1) sampling noise),
``du_max`` (hard per-step input change) and the node's ``goal_blend_s`` (stage goal blending).

S1 params: defaults off; bad values refused; ``goal_blend_s`` does not key a separate solver.
S2 off identical: pre-change (``--pre-dir``) vs tree ``MppiSolver.solve`` (cold + warm, plans and
   every info field) on the stage-1 V45c config, the sim96 V49c config and the V50b ffg29 config.
S3 ``du_max`` (V50b ffg29): ``_rate_limit`` on random inputs keeps every knot in the box and
   within the limit of the knot before (knot 0: of the anchor; free without one); solved plans
   (cold, then 6 warm solves) obey it; graph == eager.
S4 ``noise_beta``: the solve's noise has lag-1 correlation ~beta along the horizon and unit
   variance; 0 leaves it white.
S5 ``goal_blend_s`` (node, synthetic waypoints, solves captured): off, the goal is the stage's;
   on, after a switch knot k's goal is on the segment from the previous goal at
   min(1, (t - t_switch + k dt) / blend); a switch inside a blend starts from the blended point
   (knot-1 goal step <= dt / blend x the goal gap, no jump); the reach test uses the stage goal.

    uv run --frozen python scripts/checks/check_smooth_mppi.py [--pre-dir <pre-change tree>]
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from check_mppi_feasible import _solves
from check_tilt_limit import (
    CFG_96,
    CFG_S1,
    DEVICE,
    V45C_DEP,
    V49C_DEP,
    _same,
    frame_state,
    load_pre,
    params_for,
)

from task_common import mppi_lcs as tree

ES = REPO / "data/lcs/realfirst/cmp/encoder_study"
V50B_DEP = ES / "sim/V50b/deploy_v18b/deploy.npz"
CFG_FFG = ES / "sim97_V49/configs/V50b_ffg29.yaml"
DU = {"trans_mm": 2.0, "rot_deg": 1.5}


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def _limits(du: dict) -> np.ndarray:
    return np.r_[np.full(6, du["trans_mm"] * 1e-3), np.full(6, np.radians(du["rot_deg"]))]


def s1() -> str:
    p = tree.MppiParams()
    _require(p.noise_beta == 0.0 and p.du_max is None and p.goal_blend_s is None, "defaults")
    refused = 0
    for bad in ({"noise_beta": 1.0}, {"noise_beta": -0.1}, {"du_max": {"trans_mm": 1.0}},
                {"du_max": {"trans_mm": 0.0, "rot_deg": 1.0}},
                {"du_max": {"trans_mm": 1.0, "rot_deg": float("inf")}},
                {"goal_blend_s": 0.0}, {"goal_blend_s": float("nan")}):
        try:
            tree.MppiParams(**bad)
        except ValueError:
            refused += 1
    _require(refused == 7, f"refused {refused}/7")
    lcs = tree.TorchLcs.from_deploy(V50B_DEP, DEVICE)
    prm = params_for(tree, CFG_FFG)
    same = tree.solver_for(lcs, prm) is tree.solver_for(
        lcs, dataclasses.replace(prm, goal_blend_s=1.0))
    other = tree.solver_for(lcs, prm) is not tree.solver_for(
        lcs, dataclasses.replace(prm, noise_beta=0.5))
    _require(same and other, f"solver cache: blend shares {same}, noise_beta separate {other}")
    return "defaults off; 7/7 bad refused; goal_blend_s shares the cached solver"


def s2(pre) -> str:
    n, names = 0, []
    for dep, cfg in ((V45C_DEP, CFG_S1), (V49C_DEP, CFG_96), (V50B_DEP, CFG_FFG)):
        lcs = tree.TorchLcs.from_deploy(dep, DEVICE)
        rng = np.random.default_rng(3)
        p0, rot0, q = frame_state(30)
        z0 = lcs.np["z_goal"] + rng.normal(size=lcs.n_x) * lcs.z_std_np * 0.5
        goal = lcs.stage_goals[-1]
        new = _solves(tree, lcs, params_for(tree, cfg), z0, p0, rot0, goal, q)
        if pre is not None:
            plcs = pre.TorchLcs.from_deploy(dep, DEVICE)
            old = _solves(pre, plcs, params_for(pre, cfg), z0, p0, rot0, goal, q)
            for (Ua, ia), (Ub, ib) in zip(old, new):
                _require(np.array_equal(Ua, Ub) and _same(ia, ib), f"{cfg.name}: pre != tree")
                n += 1
        names.append(f"{cfg.parent.parent.name}/{cfg.name}")
    msg = f"pre == tree on {n} solves" if pre is not None else "no --pre-dir: skipped"
    return f"{msg} (plans + every info field); {', '.join(names)}"


def s3() -> str:
    lcs = tree.TorchLcs.from_deploy(V50B_DEP, DEVICE)
    prm = dataclasses.replace(params_for(tree, CFG_FFG), du_max=DU)
    d = _limits(DU)
    sol = tree.MppiSolver(lcs, prm)
    lb, ub = (x.cpu().numpy() for x in (sol.box.lb, sol.box.ub))
    rng = np.random.default_rng(5)
    n_bad = 0
    for anchor_on in (1.0, 0.0):
        anchor = rng.uniform(lb, ub)
        sol.s_anchor.copy_(lcs.tensor(anchor))
        sol.s_anchor_on.fill_(anchor_on)
        U = rng.uniform(lb - 0.01, ub + 0.01, size=(256, prm.H, lcs.n_u))
        V = sol._rate_limit(lcs.tensor(U)).cpu().numpy()
        tol = 1e-6 * d.max()  # float32 solver
        n_bad += int(((V < lb - tol) | (V > ub + tol)).sum())
        n_bad += int((np.abs(np.diff(V, axis=1)) > d + tol).sum())
        if anchor_on:
            n_bad += int((np.abs(V[:, 0] - anchor) > d + tol).sum())
        else:
            _require(np.allclose(V[:, 0], np.clip(U[:, 0], lb, ub)), "knot 0 not free")
    _require(n_bad == 0, f"{n_bad} rate-limit / box violations")
    p0, rot0, q = frame_state(30)
    z0 = lcs.np["z_goal"] + np.random.default_rng(3).normal(size=lcs.n_x) * lcs.z_std_np * 2.0
    goal = lcs.stage_goals[-1]
    plans, worst = {}, 0.0
    for mode in (True, False):
        s = tree.MppiSolver(lcs, prm, use_graph=mode and DEVICE == "cuda")
        g = torch.Generator(device=lcs.device).manual_seed(7)
        prev, out = None, []
        for _ in range(7):
            U, _ = s.solve(z0, p0, rot0, goal, prev, g, None, q)
            steps = np.abs(np.diff(U, axis=0)) / d
            if prev is not None:
                steps = np.vstack([np.abs(U[0] - prev[0]) / d, steps])
            worst = max(worst, float(steps.max()))
            out.append(U)
            prev = U
        plans[mode] = out
    _require(worst <= 1.0 + 1e-5, f"plan step / limit max {worst:.4f}")
    if DEVICE == "cuda":
        _require(all(np.allclose(a, b, atol=1e-9) for a, b in zip(plans[True], plans[False])),
                 "graph != eager")
    return (f"_rate_limit: 512 random plans, 0 outside box / limit, knot 0 free without anchor; "
            f"7 solves: max step / limit {worst:.3f}; graph == eager")


def s4() -> str:
    lcs = tree.TorchLcs.from_deploy(V50B_DEP, DEVICE)
    p0, rot0, q = frame_state(30)
    goal = lcs.stage_goals[-1]
    out = []
    for beta in (0.0, 0.8):
        prm = dataclasses.replace(params_for(tree, CFG_FFG), noise_beta=beta)
        s = tree.MppiSolver(lcs, prm)
        s.solve(lcs.np["z_goal"], p0, rot0, goal, None,
                torch.Generator(device=lcs.device).manual_seed(1), None, q)
        e = s.s_eps.cpu().numpy()
        r1 = float(np.mean(e[:, :, 1:] * e[:, :, :-1]))
        sd = e.std(axis=(0, 1, 3))
        _require(abs(r1 - beta) < 0.02 and np.all(np.abs(sd - 1.0) < 0.03),
                 f"beta {beta}: lag-1 {r1:.3f}, std {sd.round(3)}")
        out.append(f"beta {beta}: lag-1 {r1:.3f}, std {sd.min():.3f}-{sd.max():.3f}")
    return "; ".join(out)


def s5() -> str:
    import check_waypoint_schedule as ws

    _, node_mod = ws.load_pair(None, "tree104")
    seen: list = []

    def fake_plan(lcs, params, z, p0, rot0, goal, u_prev, gen, **kw):
        seen.append(np.array(goal, dtype=np.float64))
        return np.zeros((params.H, lcs.n_u)), {"solve_ms": 1.0, "fallback": False}

    node_mod.mppi_plan = fake_plan
    res = {}
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        tmp = Path(tmp)
        lcs = node_mod.TorchLcs.from_deploy(ws.c1_deploy(), ws.DEVICE)
        g = ws.synth_wp(tmp / "wp.npz", lcs)
        for name, frag in (("off", {}), ("on", {"goal_blend_s": 1.0})):
            node = ws.make_node(node_mod, tmp / "wp.npz", frag, tmp, stub=False)
            rows = []  # (latent index, stage, stage_t0, goal or None)
            for i, z in enumerate([g[0]] * 6 + [g[1]] * 3 + [g[2]] * 20):
                n0 = len(seen)
                (st, _), = ws.feed(node, [z])
                rows.append((i, st, node.stage_t0, seen[-1] if len(seen) > n0 else None))
            res[name] = rows
        dt = node.dt
    st_off, st_on = ([r[1] for r in res[k]] for k in ("off", "on"))
    _require(st_off == st_on, f"reach schedule changed: {st_off} vs {st_on}")
    _require(all(r[3] is None or np.array_equal(r[3], g[r[1]]) for r in res["off"]),
             "off: goal != the stage goal")
    sol = [r for r in res["on"] if r[3] is not None]
    t = lambda r: 1.0 + r[0] * dt  # feed: utime 1 s + k x 75 ms
    k1 = [r[3][0] if r[3].ndim == 2 else r[3] for r in sol]
    gap = max(float(np.linalg.norm(g[k + 1] - g[k])) for k in range(max(st_on)))
    worst = max(float(np.linalg.norm(b - a)) / ((rb[0] - ra[0]) * dt * gap)
                for (ra, a), (rb, b) in zip(zip(sol, k1), zip(sol[1:], k1[1:])))
    _require(worst <= 1.0001, f"knot-1 goal step {worst:.3f} x (ticks x dt / blend x gap)")
    r = sol[0]  # first solve: stage 1, blending from g0
    a = np.clip(t(r) - r[2] + dt * np.arange(1, r[3].shape[0] + 1), 0.0, 1.0)
    _require(r[1] == 1 and np.allclose(r[3], g[0] + a[:, None] * (g[1] - g[0])),
             "blend rows after the first switch")
    fin = sol[-1]
    _require(t(fin) - fin[2] > 1.0 and np.allclose(fin[3], g[fin[1]]),
             "goal after the blend != stage goal")
    n_blend = sum(x.ndim == 2 for x in (r[3] for r in sol))
    return (f"off: the stage goal every solve; on (1 s): same stages {st_on[:10]}.., first rows "
            f"on the g0->g1 segment, {n_blend} blended solves incl. 2 switches inside a blend, "
            f"knot-1 goal step <= {worst:.3f} x the per-tick blend rate, stage goal after")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="pre-change tree (repo layout: src/task_common/mppi_{lcs,warp}.py)")
    ap.add_argument("--out", type=Path, default=None, help="json report")
    a = ap.parse_args()
    pre = load_pre(a.pre_dir) if a.pre_dir else None
    res = {}
    for name, fn in (("S1", s1), ("S2", lambda: s2(pre)), ("S3", s3), ("S4", s4), ("S5", s5)):
        try:
            ok, msg = True, fn()
        except AssertionError as exc:
            ok, msg = False, str(exc)
        res[name] = {"ok": ok, "msg": msg}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": DEVICE, **res}, indent=1) + "\n")
    n = sum(r["ok"] for r in res.values())
    print(f"{n}/{len(res)} PASS ({DEVICE})")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
