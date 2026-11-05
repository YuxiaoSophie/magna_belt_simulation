#!/usr/bin/env python3
"""Checks of the MPPI feasible-first constraint handling (``MppiParams.constraint_mode``) and the
executed-command guard (``MppiParams.exec_guard``, ``mppi_guards.guard_plan``), V12-97.

F1 params: defaults ``penalty`` / no guard; bad modes / guard blocks refused; the guard does not
   key a separate cached solver.
F2 ``violation_amount`` (float64) == a numpy reference (z bounds, envelope, tilt, clearance,
   support-box hard margin) on random rollouts, and > 0 exactly where ``plan_cost`` flags a knot.
F3 ``feasible_first_cost``: every feasible sample below every violator; violators ordered by the
   violation; a violation the inputs cannot change (same for all) -> the penalty-free cost order;
   no feasible sample -> finite, ordered by violation.
F4 off identical: pre-change (``--pre-dir``) vs tree ``MppiSolver.solve`` (cold + warm, plans and
   every info field) on the stage-1 V45c config and the sim96 V49c config; tree penalty == tree
   with ``exec_guard`` set (solver unchanged). CUDA: fused + graph (clearance on); CPU: torch.
F5 feasible_first solver (sim96 V49c, a 40 deg start, goal weights x1 and x1000): penalty plans
   stay over the tilt limit, feasible_first plans tilt back (fewer violating knots, smaller
   violation); ``viol_plan`` == ``violation_amount`` of the returned plan; graph == eager.
F6 guard (float64, random): guarded knots inside z / tilt / envelope (less margins) or no
   further out than the knot before; in-limit plans unchanged (bitwise); a tilt cut keeps the
   world-z (yaw) component; ``knots`` caps the guarded rows.
F7 ``stage_min_dwell`` (node, synthetic waypoints, check_waypoint_schedule helpers): off, a
   latent on the goal reaches at once; 1.0 x the demo segment (10 frames = 0.75 s) holds each
   stage until 0.75 s in it (one stdout line per stage held); 0 == off; bad values refused.

    uv run --frozen python scripts/checks/check_mppi_feasible.py [--pre-dir <pre-change tree>]
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
    q_tilted,
    scipy_chain,
    scipy_tilt,
)
from scipy.spatial.transform import Rotation

from task_common import mppi_lcs as tree
from task_common.mppi_guards import guard_plan


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def f1() -> str:
    p = tree.MppiParams()
    _require(p.constraint_mode == "penalty" and not p.feasible_first and p.exec_guard is None,
             "defaults")
    refused = 0
    for bad in ({"constraint_mode": "hard"}, {"constraint_mode": None},
                {"exec_guard": {"knots": 0}}, {"exec_guard": {"margin_deg": -1}},
                {"exec_guard": {"x": 1}}, {"exec_guard": {"knots": 1.5}}):
        try:
            tree.MppiParams(**bad)
        except (ValueError, TypeError):
            refused += 1
    _require(refused == 6, f"refused {refused}/6")
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, "cpu", torch.float64)
    p96 = tree.MppiParams.from_yaml(CFG_96)
    g = dataclasses.replace(p96, exec_guard={"knots": 2})
    _require(tree.solver_for(lcs, g) is tree.solver_for(lcs, p96), "guard keys a new solver")
    ff = dataclasses.replace(p96, constraint_mode="feasible_first")
    _require(tree.solver_for(lcs, ff) is not tree.solver_for(lcs, p96), "ff shares a solver")
    return f"defaults penalty / no guard; {refused}/6 bad refused; guard shares the cached solver"


def _random_rollout(lcs, prm, K=512, seed=0):
    rng = np.random.default_rng(seed)
    H = prm.H
    p0, rot0, q = frame_state(30)
    U = np.zeros((K, H, 12))
    U[:, :, :3] = rng.uniform(-1, 1, (K, H, 3)) * 0.007
    U[:, :, 6:9] = rng.uniform(-1, 1, (K, H, 3)) * np.radians(6.0)
    Ut = lcs.tensor(U)
    Z, P, R = lcs.rollout(lcs.np["z_goal"], p0, Ut, rot0, prm.pgd_iters,
                          **({"q_abs": q} if prm.tilt_on else {}))
    return U, Ut, Z, P, R, p0, rot0, q


def f2() -> str:
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, "cpu", torch.float64)
    base = dataclasses.replace(tree.MppiParams.from_yaml(CFG_96), clearance=None)
    sb = dict(base.support_box, hard_margin_m=0.004)
    prm = dataclasses.replace(base, support_box=sb, rot_envelope_deg={"franka": 60.0, "ur": 1.0})
    U, Ut, Z, P, R, _, _, _ = _random_rollout(lcs, prm)
    pz = P[:, 1:, 2].numpy()
    lim = tree.safety_limits(prm)
    lim[0] = np.quantile(pz, 0.3)  # make the z bound bind on a share of the knots
    lim[1] = np.quantile(pz, 0.95)
    limt = lcs.tensor(lim)
    # fake clearance distances: a share of the knots under the Franka threshold
    rng = np.random.default_rng(1)
    C = lcs.tensor(rng.uniform(-0.002, 0.01, size=(len(U), prm.H + 1, 2)))

    class T:  # the terms violation_amount reads
        clear = type("c", (), {"lim": lcs.tensor([0.003, 0.0])})()
        sbox = lcs.tensor(np.stack([sb["lo"], sb["hi"]]))

    v = tree.violation_amount(prm, P, R, limt, C, T).numpy()
    Pn, Rn, Cn = P.numpy()[:, 1:], R.numpy()[:, 1:], C.numpy()[:, 1:]
    relu = lambda x: np.maximum(x, 0.0)
    mm = (relu(lim[0] - Pn[..., 2]) + relu(Pn[..., 2] - lim[1]) + relu(lim[2] - Pn[..., 5])
          + relu(Pn[..., 5] - lim[3]) + relu(0.003 - Cn[..., 0]) + relu(0.0 - Cn[..., 1]))
    lo, hi = np.asarray(sb["lo"]), np.asarray(sb["hi"])
    mm = mm + relu(relu(lo - Pn) + relu(Pn - hi) - 0.004).sum(-1)
    deg = np.degrees(relu(Rn[..., 0] - lim[4]) + relu(Rn[..., 1] - lim[5])
                     + relu(Rn[..., 2] - lim[6]) + relu(Rn[..., 3] - lim[7]))
    ref = (mm * 1e3 + deg).sum(1)
    e = float(np.abs(v - ref).max())
    _require(e < 1e-9, f"violation_amount vs numpy {e:.2e}")
    # flags: plan_cost with the same inputs
    goal = lcs.tensor(lcs.stage_goals[-1])
    half = lcs.tensor(np.ones(12))

    class T2(T):
        pass

    T2.clear = type("c", (), {"lim": T.clear.lim, "w_soft": 0.0, "soft_margin": 0.0})()
    _, n_bad, _ = tree.plan_cost(lcs, prm, Z, P, R, Ut, goal, half, None, lim=limt, C=C,
                                 terms=T2)
    agree = np.array_equal(n_bad.numpy() > 0, v > 0)
    _require(agree and 0 < (v > 0).sum() < len(v), f"flags vs amount: {(v > 0).sum()} "
             f"{(n_bad.numpy() > 0).sum()}")
    return (f"{len(v)} random rollouts: == numpy max |d| {e:.1e}; > 0 on exactly the "
            f"{int((v > 0).sum())} flagged plans")


def f3() -> str:
    rng = np.random.default_rng(2)
    K, pen = 1000, 1e4
    raw = torch.as_tensor(rng.uniform(0, 1e5, K))
    n_bad = torch.as_tensor(rng.integers(0, 3, K).astype(np.float64)) * (rng.random(K) < 0.6)
    viol = torch.where(n_bad > 0, torch.as_tensor(rng.uniform(0.01, 50, K)), torch.zeros(K))
    eff = tree.feasible_first_cost(raw + pen * n_bad, n_bad, viol, pen)
    f = (n_bad == 0)
    _require(bool(eff[f].max() < eff[~f].min()), "a violator beats a feasible sample")
    vo, eo = viol[~f].numpy(), eff[~f].numpy()
    order_ok = np.mean(np.argsort(vo) == np.argsort(eo))
    feas_ok = np.array_equal(np.argsort(raw[f].numpy()), np.argsort(eff[f].numpy()))
    _require(feas_ok, "feasible order != raw cost order")
    # constant (uncontrollable) violation: the raw cost decides
    nb1 = torch.ones(K)
    eff1 = tree.feasible_first_cost(raw + pen, nb1, torch.full((K,), 3.0), pen)
    _require(np.array_equal(np.argsort(raw.numpy()), np.argsort(eff1.numpy()))
             and bool(torch.isfinite(eff1).all()), "constant violation: not the cost order")
    # none feasible, varying violation: violation decides up to FF_COST_EQUIV
    v2 = torch.as_tensor(rng.uniform(1, 50, K))
    eff2 = tree.feasible_first_cost(raw + pen, nb1, v2, pen)
    i = torch.argsort(v2)
    gap_ok = bool(((v2[i][1:] - v2[i][:-1] > tree.FF_COST_EQUIV)
                   <= (eff2[i][1:] > eff2[i][:-1])).all())
    _require(gap_ok and bool(torch.isfinite(eff2).all()), "none feasible: violation order")
    return (f"feasible max {float(eff[f].max()):.3g} < violator min {float(eff[~f].min()):.3g}; "
            f"violator order == violation order on {order_ok:.0%} (cost tie-break within "
            f"{tree.FF_COST_EQUIV}); constant violation -> cost order; none feasible -> "
            "violation order, finite")


def _solves(mod, lcs, prm, z0, p0, rot0, goal, q, use_graph=None):
    s = mod.MppiSolver(lcs, prm, use_graph=use_graph)
    g = torch.Generator(device=lcs.device).manual_seed(11)
    lim = mod.safety_limits(prm)
    U1, i1 = s.solve(z0, p0, rot0, goal, None, g, lim, q)
    U2, i2 = s.solve(z0 + 0.1 * lcs.z_std_np, p0, rot0, goal, U1, g, lim, q)
    return (U1, i1), (U2, i2)


def f4(pre) -> str:
    msgs, n = [], 0
    for dep, cfg in ((V45C_DEP, CFG_S1), (V49C_DEP, CFG_96)):
        lcs = tree.TorchLcs.from_deploy(dep, DEVICE)
        prm = params_for(tree, cfg)
        rng = np.random.default_rng(3)
        p0, rot0, q = frame_state(30)
        z0 = lcs.np["z_goal"] + rng.normal(size=lcs.n_x) * lcs.z_std_np * 0.5
        goal = lcs.stage_goals[-1]
        new = _solves(tree, lcs, prm, z0, p0, rot0, goal, q)
        if pre is not None and (cfg != CFG_96 or hasattr(pre.MppiParams(), "tilt_max_deg")):
            plcs = pre.TorchLcs.from_deploy(dep, DEVICE)
            old = _solves(pre, plcs, params_for(pre, cfg), z0, p0, rot0, goal, q)
            for (Ua, ia), (Ub, ib) in zip(old, new):
                _require(np.array_equal(Ua, Ub) and _same(ia, ib), f"{cfg.name}: pre != tree")
                n += 1
        gd = _solves(tree, lcs, dataclasses.replace(prm, exec_guard={}), z0, p0, rot0, goal, q)
        for (Ua, ia), (Ub, ib) in zip(new, gd):
            _require(np.array_equal(Ua, Ub) and _same(ia, ib), f"{cfg.name}: guard set != off")
        msgs.append(f"{cfg.parent.parent.name}/{cfg.name}")
    how = "fused + graph" if DEVICE == "cuda" else "torch, CPU"
    pre_msg = (f"pre == tree on {n} solves (plans + every info field)" if pre is not None
               else "no --pre-dir: pre vs tree skipped")
    return f"{pre_msg}; exec_guard set == off in the solver; {', '.join(msgs)}; {how}"


def f5() -> str:
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, DEVICE)
    base = params_for(tree, CFG_96)
    p0, rot0, q = frame_state(0)
    rng = np.random.default_rng(4)
    z0 = lcs.np["z_goal"] + rng.normal(size=lcs.n_x) * lcs.z_std_np * 2.0
    goal = lcs.stage_goals[-1]
    q40 = np.stack([q_tilted(40.0, (0.0, 1.0, 0.0)), q[1]])
    out = []
    for wq in (1.0, 1000.0):
        res = {}
        for mode in ("penalty", "feasible_first"):
            prm = dataclasses.replace(base, constraint_mode=mode, w_q=wq, w_T=base.w_T * wq)
            g = torch.Generator(device=lcs.device).manual_seed(2)
            U, info = tree.MppiSolver(lcs, prm).solve(z0, p0, rot0, goal, None, g, None, q40)
            knots = np.degrees(scipy_chain(q40[0], U[:, 6:9]))
            res[mode] = (U, info, knots)
        (_, ip, kp), (Uf, iff, kf) = res["penalty"], res["feasible_first"]
        _require(kf[1:].max() < 40.0 and kf[-1] < kp[-1] and iff["plan_violations"]
                 < ip["plan_violations"], f"x{wq:g}: ff knots {kf.round(1)} vs {kp.round(1)}")
        prm = dataclasses.replace(base, constraint_mode="feasible_first", w_q=wq,
                                  w_T=base.w_T * wq)
        kw = {"q_abs": q40}
        _, P, R = lcs.rollout(z0, p0, lcs.tensor(Uf)[None], rot0, prm.pgd_iters, **kw)
        terms = tree.load_cost_terms(lcs, prm)
        C = None if terms.clear is None else terms.clear.torch_path(P, lcs.tensor(Uf)[None],
                                                                    lcs.tensor(q40))
        v = float(tree.violation_amount(prm, P, R, lcs.tensor(tree.safety_limits(prm)), C,
                                        terms)[0])
        _require(abs(v - iff["viol_plan"]) <= 1e-3 * max(1.0, v), f"viol_plan {iff['viol_plan']}"
                 f" != {v}")
        out.append(f"goal x{wq:g}: penalty tilt {kp[1]:.1f}->{kp[-1]:.1f} deg "
                   f"({ip['plan_violations']} knots) vs feasible_first {kf[1]:.1f}->{kf[-1]:.1f} deg "
                   f"({iff['plan_violations']} knots, viol {iff['viol_plan']:.1f})")
    msg = "; ".join(out) + "; viol_plan == violation_amount"
    if DEVICE == "cuda":
        prm = dataclasses.replace(base, constraint_mode="feasible_first")
        res = []
        for use in (True, False):
            g = torch.Generator(device=lcs.device).manual_seed(2)
            res.append(tree.MppiSolver(lcs, prm, use_graph=use).solve(
                z0, p0, rot0, goal, None, g, None, q40)[0])
        _require(np.array_equal(*res), "graph != eager")
        msg += "; graph == eager"
    return msg


def _knots(p_now, q_now, q_start, U):
    """Per arm (z (H+1,), tilt deg (H+1,), envelope deg (H+1,)) of the knots of ``U``."""
    out = []
    for a in range(2):
        r = Rotation.from_quat(q_now[a][[1, 2, 3, 0]])
        r0i = Rotation.from_quat(q_start[a][[1, 2, 3, 0]]).inv()
        z, t, e = [p_now[3 * a + 2]], [r], [r * r0i]
        for u in U:
            z.append(z[-1] + u[3 * a + 2])
            r = Rotation.from_rotvec(u[6 + 3 * a:9 + 3 * a]) * r
            t.append(r)
            e.append(r * r0i)
        out.append((np.array(z), np.degrees(scipy_tilt(np.array([x.as_quat()[[3, 0, 1, 2]]
                                                                  for x in t]))),
                    np.degrees([x.magnitude() for x in e])))
    return out


def f6() -> str:
    rng = np.random.default_rng(6)
    lim = np.r_[0.03, 0.09, -np.inf, np.inf, np.radians([60.0, 1.0]), np.radians(32.0),
                np.inf]
    n_cut = n_same = n_out = 0
    worst = 0.0
    for trial in range(300):
        t0 = rng.uniform(5.0, 40.0)
        ax = rng.normal(size=3)
        ax[2] = 0.0
        q_now = np.stack([q_tilted(t0, ax / np.linalg.norm(ax)), [1.0, 0, 0, 0]])
        q_start = np.stack([q_tilted(rng.uniform(0, 30), (0, 1, 0)), [1.0, 0, 0, 0]])
        p_now = np.r_[0.4, 0.1, rng.uniform(0.025, 0.095), 0.5, -0.14, 0.11]
        U = np.zeros((7, 12))
        U[:, :3] = rng.uniform(-1, 1, (7, 3)) * 0.007
        U[:, 6:9] = rng.uniform(-1, 1, (7, 3)) * np.radians([3.0, 3.0, 8.0])
        if trial % 3 == 0:
            U *= 0.05
        cfg = {"margin_deg": 0.5, "margin_mm": 1.0}
        G, info = guard_plan(U, p_now, q_now, q_start, lim, cfg)
        (zb, tb, eb), _ = _knots(p_now, q_now, q_start, U[:6])
        (z, t, e), _ = _knots(p_now, q_now, q_start, G[:6])
        tl, el = 31.5, max(59.5, eb[0])
        zl, zh = 0.031, 0.089
        for k in range(1, 7):
            ok_t = t[k] <= max(tl, t[k - 1]) + 1e-6
            ok_z = (z[k] >= min(zl, z[k - 1]) - 1e-12) and (z[k] <= max(zh, z[k - 1]) + 1e-12)
            ok_e = e[k] <= max(59.5, e[k - 1]) + 1e-6
            if not (ok_t and ok_z and ok_e):
                n_out += 1
                worst = max(worst, t[k] - max(tl, t[k - 1]))
        inside = ((tb[1:] <= tl).all() and (eb[1:] <= el).all() and (zb[1:] >= zl).all()
                  and (zb[1:] <= zh).all())
        if inside:
            _require(np.array_equal(G, U) and info["first"] < 0, "in-limit plan changed")
            n_same += 1
        if info["n_rot"]:
            n_cut += 1
        _require(np.array_equal(G[6:], U[6:]), "rows past knots changed")
    _require(n_out == 0, f"{n_out} guarded knots outside (worst {worst:.3g} deg)")
    _require(n_cut > 20 and n_same > 20, f"coverage: {n_cut} cut, {n_same} untouched")
    # yaw kept: a pure tilt-increasing rotation + yaw from 31 deg
    q_now = np.stack([q_tilted(31.0, (1, 0, 0)), [1.0, 0, 0, 0]])
    U = np.zeros((7, 12))
    U[:, 6], U[:, 8] = np.radians(3.0), np.radians(5.0)
    G, info = guard_plan(U, np.r_[0.4, 0.1, 0.05, 0.5, -0.14, 0.11], q_now, q_now, lim,
                         {"knots": 3})
    (_, t, _), _ = _knots(np.r_[0.4, 0.1, 0.05, 0.5, -0.14, 0.11], q_now, q_now, G[:3])
    _require(np.allclose(G[:3, 8], np.radians(5.0)) and t.max() <= 32.0 + 1e-6
             and np.array_equal(G[3:], U[3:]), f"yaw / knots cap: {G[:4, 6:9]} {t}")
    return (f"300 random plans (margins 0.5 deg / 1 mm): 0 knots outside, {n_cut} rotation cuts, "
            f"{n_same} in-limit plans unchanged; tilt cut keeps yaw (5 deg/step), knots 3 caps "
            f"the rows; tilt from 31 deg -> max {t.max():.2f} deg")


def f7() -> str:
    import check_waypoint_schedule as ws

    _, node_mod = ws.load_pair(None, "tree97")
    out = {}
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as so:
        tmp = Path(tmp)
        lcs = node_mod.TorchLcs.from_deploy(ws.c1_deploy(), ws.DEVICE)
        g = ws.synth_wp(tmp / "wp.npz", lcs)
        for name, frag in (("off", {}), ("zero", {"stage_min_dwell": 0.0}),
                           ("one", {"stage_min_dwell": 1.0})):
            node = ws.make_node(node_mod, tmp / "wp.npz", frag, tmp)
            out[name] = [s for s, _ in ws.feed(node, [g[0]] * 14)] + \
                [s for s, _ in ws.feed(node, [g[1]] * 14)]
        held = so.getvalue().count("[mppi] stage min dwell")
    _require(out["off"][:2] == [1, 1] and out["off"] == out["zero"], f"off {out['off']}")
    one = out["one"]
    # stage 0 from the first latent: reached at 10 x 75 ms; stage 1 from that latent on
    _require(one[:10] == [0] * 10 and one[10:14] == [1] * 4 and one[14:20] == [1] * 6
             and one[20] == 2 and held == 2, f"dwell 1.0: {one}, held lines {held}")
    refused = 0
    for bad in (-0.1, float("nan"), float("inf")):
        try:
            tree.MppiParams(stage_min_dwell=bad)
        except ValueError:
            refused += 1
    _require(refused == 3, f"refused {refused}/3")
    return (f"off / 0: stage 1 at latent 0; dwell 1.0: stage 0 -> 1 at latent 10 (0.75 s), "
            f"stage 1 -> 2 at latent 20 (0.75 s after the switch), {held} held lines; 3/3 bad "
            "refused")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="pre-change tree (repo layout: src/task_common/mppi_{lcs,warp}.py)")
    ap.add_argument("--out", type=Path, default=None, help="json report")
    a = ap.parse_args()
    pre = load_pre(a.pre_dir) if a.pre_dir else None
    res = {}
    for name, fn in (("F1", f1), ("F2", f2), ("F3", f3), ("F4", lambda: f4(pre)), ("F5", f5),
                     ("F6", f6), ("F7", f7)):
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
