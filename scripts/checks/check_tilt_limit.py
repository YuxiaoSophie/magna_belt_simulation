#!/usr/bin/env python3
"""Checks of the MPPI tilt limit (``MppiParams.tilt_max_deg``) and the per-axis per-step rotation
box (``u_box_outer.rot_deg_xyz``), V12-96.

T1 params: defaults off (``tilt_on`` False, ``safety_limits`` the 6 pre-change entries, no
   ``q_now`` needed); ``{franka: inf, ur: None}`` is off too; bad keys / values refused; the sim96
   configs load with the tilt on and key a different cached solver than the sim95 ones.
T2 box: ``rot_deg_xyz`` -> rotation channels 6..11 = +-[x, y, z, x, y, z] deg intersected with
   the deploy box (both arms, UR then pinned by ``u_franka_only``); ``[r, r, r]`` == the scalar
   ``rot_deg`` exactly; absent -> the pre-change ``bounds`` (with ``--pre-dir``).
T3 maths (float64): ``quat_tilt_t`` == scipy acos(|R[:, 2] . e_z|) on random rotations (incl.
   exactly down / horizontal); ``tilt_angles`` == the scipy composition ``q_k = exp(w_k)
   q_{k-1}`` from the absolute ``q_abs``.
T4 flags (``plan_cost``, float64, V49c deploy): from a 20 deg start a 4.5 deg/step tilt about
   world x is flagged at exactly the knots past 32 deg; a 110 deg pure yaw about world z is not;
   random plans: added violating knots == the scipy count; off: none; ``rollout_cost`` refuses a
   missing ``q_now`` with the tilt on.
T5 Warp (CUDA): ``rollout_fused`` tilt == the torch rollout's (float32) on random plans; Z / P /
   envelope R with ``q_abs`` == without (bitwise).
T6 off identical: pre-change (``--pre-dir``, repo layout, with its own ``mppi_warp``) vs tree
   ``MppiSolver.solve`` on the stage-1 V45c config and the sim95 V49c config, cold + warm: plans
   and every info field identical; tree with a non-binding tilt limit (90 deg) == off plans.
   Without ``--pre-dir`` only the latter. CUDA: fused + graph (clearance on); CPU: torch, no
   clearance.
T7 solver, tilt on (sim96 V49c): ``tilt_now`` == scipy of ``q_now``, ``tilt_plan_max`` == the
   plan's recomputed knots, solver ``plan_cost`` == ``rollout_cost`` of its plan; graph == eager
   (CUDA); a start past the limit -> the plan tilts back.

    uv run --frozen python scripts/checks/check_tilt_limit.py [--pre-dir <pre-change tree>]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from scipy.spatial.transform import Rotation

from task_common import mppi_lcs as tree

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ES = REPO / "data/lcs/realfirst/cmp/encoder_study"
V49C_DEP = ES / "sim/V49c/deploy_v17c/deploy.npz"
V45C_DEP = ES / "sim/V45c/deploy_v17c/deploy.npz"
CFG_S1 = ES / "sim87_V45c_ur30/configs/V45c_ur30_H7.yaml"
CFG_95 = ES / "sim95_V49/configs/V49c_stage2.yaml"
CFG_96 = ES / "sim96_V49/configs/V49c_stage2.yaml"
EPISODE = ES / "sim95_V49/V49c_nominal/episode_V45c_nominal_0_0.npz"
Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz: pi about x, the hand's z axis pointing down


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def scipy_tilt(q_wxyz: np.ndarray) -> np.ndarray:
    m = Rotation.from_quat(np.asarray(q_wxyz)[..., [1, 2, 3, 0]]).as_matrix()
    return np.arccos(np.clip(np.abs(m[..., 2, 2]), 0.0, 1.0))


def scipy_chain(q0: np.ndarray, W: np.ndarray) -> np.ndarray:
    """``(H+1,)`` tilt of ``q_k = exp(w_k) q_{k-1}`` from ``q0`` (wxyz), ``W (H, 3)``."""
    r = Rotation.from_quat(q0[[1, 2, 3, 0]])
    out = [r]
    for w in W:
        r = Rotation.from_rotvec(w) * r
        out.append(r)
    return np.array([math.acos(min(1.0, abs(x.as_matrix()[2, 2]))) for x in out])


def frame_state(t: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(p0 (6,), rot0 (2, 3), q_now (2, 4))`` of a recorded stage-2 frame (start = frame 0)."""
    with np.load(EPISODE, allow_pickle=True) as d:
        f, u = np.asarray(d["sim_ee_franka"], np.float64), np.asarray(d["sim_ee_ur"], np.float64)
    q = np.stack([f[t, 3:7], u[t, 3:7]])
    rot0 = np.stack([(Rotation.from_quat(e[t, [4, 5, 6, 3]])
                      * Rotation.from_quat(e[0, [4, 5, 6, 3]]).inv()).as_rotvec() for e in (f, u)])
    return np.r_[f[t, :3], u[t, :3]], rot0, q


def q_tilted(deg: float, axis=(1.0, 0.0, 0.0)) -> np.ndarray:
    w = np.radians(deg) * np.asarray(axis, np.float64)
    return tree.qmul(tree.quat_exp(w), Q_DOWN)


def load_pre(d: Path):
    """Pre-change ``mppi_lcs`` from ``d`` (repo layout) on its own ``mppi_warp``."""
    mods = {}
    for name in ("mppi_warp", "mppi_lcs"):
        spec = importlib.util.spec_from_file_location(f"{name}_pre",
                                                      d / f"src/task_common/{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mods[name]
        spec.loader.exec_module(mods[name])
    mods["mppi_lcs"]._FusedRollout = mods["mppi_warp"]._FusedRollout
    return mods["mppi_lcs"]


def params_for(mod, path: Path):
    p = mod.MppiParams.from_yaml(path)
    if DEVICE == "cpu":  # the clearance torch path is minutes per solve on the CPU
        p = dataclasses.replace(p, clearance=None)
    return p


# --- checks -------------------------------------------------------------------------------------

def t1(pre) -> str:
    p = tree.MppiParams()
    _require(p.tilt_max_deg is None and not p.tilt_on and not p.needs_q_now, "defaults on")
    _require(tree.safety_limits(p).shape == (6,), "safety_limits off != 6 entries")
    off = tree.MppiParams(tilt_max_deg={"franka": math.inf, "ur": None})
    _require(not off.tilt_on and tree.safety_limits(off).shape == (6,), "inf / None not off")
    refused = 0
    for bad in ({"tilt_max_deg": {"franka": 30.0}}, {"tilt_max_deg": {"franka": 0.0, "ur": None}},
                {"tilt_max_deg": {"franka": 30.0, "ur": None, "x": 1}},
                {"u_box_outer": {"trans_m": 0.007, "rot_deg": 8.0, "rot_deg_xyz": [3.0, 3.0]}},
                {"u_box_outer": {"trans_m": 0.007, "rot_deg": 8.0, "rot_deg_xyz": [3, -1, 8]}}):
        try:
            tree.MppiParams(**bad)
        except ValueError:
            refused += 1
    _require(refused == 5, f"refused {refused}/5")
    p96, p95 = tree.MppiParams.from_yaml(CFG_96), tree.MppiParams.from_yaml(CFG_95)
    _require(p96.tilt_on and p96.needs_q_now and not p95.tilt_on, "sim96 / sim95 tilt flags")
    lim = tree.safety_limits(p96)
    _require(lim.shape == (8,) and np.isclose(np.degrees(lim[6]), 32.0) and np.isinf(lim[7]),
             f"sim96 limits {lim}")
    _require(dataclasses.replace(p96, tilt_max_deg=None, u_box_outer=p95.u_box_outer)
             .to_dict() == p95.to_dict(), "sim96 != sim95 + tilt + rot_deg_xyz")
    if pre is not None:
        a = pre.MppiParams.from_yaml(CFG_95).to_dict()
        later = ("tilt_max_deg", "constraint_mode", "exec_guard", "stage_min_dwell")  # V12-96/97
        _require(a == {k: v for k, v in p95.to_dict().items() if k in a or k not in later},
                 "sim95 to_dict != pre + later fields")
        _require(np.array_equal(pre.safety_limits(pre.MppiParams.from_yaml(CFG_95)),
                                tree.safety_limits(p95)), "sim95 safety_limits != pre")
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, "cpu", torch.float64)
    _require(tree.solver_for(lcs, p96) is not tree.solver_for(lcs, p95), "same cached solver")
    return (f"defaults off; inf/None off; {refused}/5 bad blocks refused; sim96 = sim95 + "
            f"tilt {p96.tilt_max_deg} + rot_deg_xyz {p96.u_box_outer['rot_deg_xyz']}, limits "
            f"{np.degrees(lim[6:]).round(1).tolist()} deg; separate cached solver")


def t2(pre) -> str:
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, "cpu", torch.float64)
    p95 = tree.MppiParams.from_yaml(CFG_95)
    p96 = tree.MppiParams.from_yaml(CFG_96)
    msgs = []
    for fo in (False, True):
        b = {"trans_m": 0.007, "rot_deg": 8.0, "rot_deg_xyz": [3.0, 3.0, 8.0]}
        p = dataclasses.replace(p96, u_franka_only=fo, u_box_outer=b)
        lb, ub = p.bounds(lcs)
        half = np.radians([3.0, 3.0, 8.0] * 2)
        dlb, dub = lcs.u_lb_np, lcs.u_ub_np
        want_lb, want_ub = np.maximum(dlb[6:], -half), np.minimum(dub[6:], half)
        if fo:
            want_lb[3:] = want_ub[3:] = 0.0
        _require(np.array_equal(lb[6:], want_lb) and np.array_equal(ub[6:], want_ub),
                 f"u_franka_only {fo}: rotation bounds {lb[6:]} / {ub[6:]}")
        _require(np.array_equal(lb[:6], p95.bounds(lcs)[0][:6]) if fo else True,
                 "translation bounds changed")
        msgs.append(f"u_franka_only {fo}: rot ub {np.degrees(ub[6:]).round(3).tolist()} deg")
    for r in (2.0, 8.0):
        sc = dataclasses.replace(p95, u_box_outer={"trans_m": 0.007, "rot_deg": r})
        xyz = dataclasses.replace(p95, u_box_outer={"trans_m": 0.007, "rot_deg": 99.0,
                                                    "rot_deg_xyz": [r, r, r]})
        _require(all(np.array_equal(a, b) for a, b in zip(sc.bounds(lcs), xyz.bounds(lcs))),
                 f"[{r}] x 3 != scalar {r}")
    if pre is not None:
        plcs = pre.TorchLcs.from_deploy(V49C_DEP, "cpu", torch.float64)
        for path in (CFG_95, CFG_S1):
            a, b = pre.MppiParams.from_yaml(path).bounds(plcs), \
                tree.MppiParams.from_yaml(path).bounds(lcs)
            _require(all(np.array_equal(x, y) for x, y in zip(a, b)), f"{path.name} != pre")
        msgs.append("absent: == pre (sim95 V49c, V45c stage 1)")
    same = all(np.array_equal(a, b) for a, b in zip(p95.bounds(lcs), p96.bounds(lcs)))
    return "; ".join(msgs) + (f"; [r, r, r] == scalar r; sim96 vs sim95 effective bounds on the "
                              f"V49c deploy identical: {same} (the deploy box is tighter)")


def t3() -> str:
    rng = np.random.default_rng(0)
    q = Rotation.random(2000, random_state=1).as_quat()[:, [3, 0, 1, 2]]
    q = np.vstack([q, Q_DOWN, [1.0, 0, 0, 0], q_tilted(90.0), q_tilted(90.0, (0, 1, 0)),
                   q_tilted(1e-6)])
    got = tree.quat_tilt_t(torch.as_tensor(q * 1.7)).numpy()  # any norm
    e1 = float(np.abs(got - scipy_tilt(q)).max())
    _require(e1 < 1e-7, f"quat_tilt_t vs scipy {e1:.2e} rad")
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, "cpu", torch.float64)
    K, H = 64, 7
    U = rng.normal(size=(K, H, 12)) * np.radians(6.0)
    qa = np.stack([q_tilted(17.0, (0.6, 0.8, 0.0)), Rotation.random(random_state=3).as_quat()[
        [3, 0, 1, 2]]])
    T = lcs.tilt_angles(torch.as_tensor(U), qa).numpy()
    e2 = max(float(np.abs(T[k, :, a] - scipy_chain(qa[a], U[k, :, sl])).max())
             for k in range(K) for a, sl in enumerate(tree.ROT_SLICES))
    _require(e2 < 1e-7, f"tilt_angles vs scipy chain {e2:.2e} rad")
    return (f"quat_tilt_t vs scipy max |d| {e1:.1e} rad ({len(q)} rotations incl. down / "
            f"horizontal); tilt_angles vs scipy chain {e2:.1e} rad ({K} plans x {H} knots x 2)")


def t4() -> str:
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, "cpu", torch.float64)
    base = dataclasses.replace(tree.MppiParams.from_yaml(CFG_96), clearance=None,
                               support_box=None)
    off = dataclasses.replace(base, tilt_max_deg=None)
    p0, _, _ = frame_state(0)
    goal = lcs.tensor(lcs.stage_goals[-1])
    z0 = lcs.np["z_goal"]
    lb, ub = base.bounds(lcs)
    half = lcs.tensor(np.where(ub > lb, (ub - lb) / 2.0, 1.0))

    def n_bad(prm, U, q):
        Ut = lcs.tensor(U)[None] if U.ndim == 2 else lcs.tensor(U)
        kw = {"q_abs": q} if prm.tilt_on else {}
        Z, P, R = lcs.rollout(z0, p0, Ut, np.zeros((2, 3)), prm.pgd_iters, **kw)
        return tree.plan_cost(lcs, prm, Z, P, R, Ut, goal, half, None)[1].numpy().astype(int)

    H = base.H
    q20 = np.stack([q_tilted(20.0), [1.0, 0, 0, 0]])
    Ux = np.zeros((H, 12))
    Ux[:, 6] = np.radians(4.5)
    want = int((np.degrees(scipy_chain(q20[0], Ux[:, 6:9]))[1:] > 32.0).sum())
    nx_on, nx_off = int(n_bad(base, Ux, q20)[0]), int(n_bad(off, Ux, q20)[0])
    _require(want == 5 and nx_on == want and nx_off == 0,
             f"tilt plan: on {nx_on} off {nx_off} want {want}")
    Uz = np.zeros((H, 12))
    Uz[:, 8] = np.radians(110.0) / H
    yaw = np.degrees(scipy_chain(q20[0], Uz[:, 6:9]))
    nz = int(n_bad(base, Uz, q20)[0])
    _require(nz == 0 and np.abs(yaw - 20.0).max() < 1e-9, f"yaw plan flagged {nz}, tilt {yaw}")
    rng = np.random.default_rng(5)
    K = 256
    Ur = np.zeros((K, H, 12))
    Ur[:, :, 6:9] = rng.uniform(-1.0, 1.0, size=(K, H, 3)) * np.radians(5.0)
    qr = np.stack([q_tilted(25.0, (0.0, 1.0, 0.0)), [1.0, 0, 0, 0]])
    env_off = {"franka": 1e9, "ur": 1e9}
    on_r = n_bad(dataclasses.replace(base, rot_envelope_deg=env_off), Ur, qr)
    off_r = n_bad(dataclasses.replace(off, rot_envelope_deg=env_off), Ur, qr)
    want_r = np.array([(np.degrees(scipy_chain(qr[0], u[:, 6:9]))[1:] > 32.0).sum() for u in Ur])
    _require(np.array_equal(on_r - off_r, want_r) and not off_r.any(),
             f"random plans: {int((on_r - off_r != want_r).sum())} mismatches")
    try:
        tree.rollout_cost(lcs, base, z0, p0, None, lcs.stage_goals[-1], Ux)
        refused = False
    except ValueError:
        refused = True
    _require(refused, "rollout_cost without q_now accepted with the tilt on")
    c_on = tree.rollout_cost(lcs, base, z0, p0, None, lcs.stage_goals[-1], Ux, q_now=q20)
    _require(c_on[1] == nx_on, f"rollout_cost n_bad {c_on[1]} != {nx_on}")
    return (f"4.5 deg/step about x from 20 deg: {nx_on} knots flagged (scipy {want}), off 0; "
            f"110 deg yaw: tilt stays {yaw.max():.1f} deg, 0 flagged; {K} random plans: added "
            f"knots == scipy count ({int(want_r.sum())} knots, {int((want_r > 0).sum())} plans); "
            "rollout_cost needs q_now")


def t5() -> str:
    if DEVICE != "cuda":
        return "skipped (no CUDA)"
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, "cuda")
    rng = np.random.default_rng(7)
    K, H = 4096, 7
    U = lcs.tensor(rng.uniform(-1.0, 1.0, size=(K, H, 12)) * np.r_[[0.007] * 6,
                                                                   [np.radians(8.0)] * 6])
    p0, rot0, q = frame_state(40)
    z0 = lcs.np["z_goal"]
    qa = np.stack([q[0], Rotation.random(random_state=9).as_quat()[[3, 0, 1, 2]]])
    Zt, _, Rt = lcs.rollout(z0, p0, U, rot0, 25, q_abs=qa)
    Zf, Pf, Rf = lcs.rollout_fused(z0, p0, U, rot0, 25, q_abs=qa)
    Zo, Po, Ro = lcs.rollout_fused(z0, p0, U, rot0, 25)
    e = float((Rt[..., 2:] - Rf[..., 2:]).abs().max())
    _require(Rt.shape == Rf.shape == (K, H + 1, 4) and e < 1e-4, f"torch vs warp tilt {e:.2e}")
    same = (torch.equal(Zf, Zo) and torch.equal(Pf, Po) and torch.equal(Rf[..., :2], Ro)
            and Ro.shape[-1] == 2)
    _require(same, "fused Z / P / R changed by q_abs")
    ez = float((Zt - Zf).abs().max())
    return (f"{K} plans: torch vs Warp tilt max |d| {e:.1e} rad (Z {ez:.1e}); fused Z / P / R "
            "with q_abs == without (bitwise)")


def _same(i1: dict, i2: dict) -> bool:
    if set(i1) != set(i2):
        return False
    for k, v in i1.items():
        if k == "solve_ms":
            continue
        a, b = np.asarray(v), np.asarray(i2[k])
        if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            return False
    return True


def _solves(mod, lcs, prm, z0, p0, rot0, goal, q, use_graph=None):
    s = mod.MppiSolver(lcs, prm, use_graph=use_graph)
    g = torch.Generator(device=lcs.device).manual_seed(11)
    lim = mod.safety_limits(prm)
    U1, i1 = s.solve(z0, p0, rot0, goal, None, g, lim, q)
    U2, i2 = s.solve(z0 + 0.1 * lcs.z_std_np, p0, rot0, goal, U1, g, lim, q)
    return (U1, i1), (U2, i2)


def t6(pre) -> str:
    msgs, n = [], 0
    for dep, cfg in ((V45C_DEP, CFG_S1), (V49C_DEP, CFG_95)):
        lcs = tree.TorchLcs.from_deploy(dep, DEVICE)
        prm = params_for(tree, cfg)
        rng = np.random.default_rng(3)
        p0, rot0, q = frame_state(30)
        z0 = lcs.np["z_goal"] + rng.normal(size=lcs.n_x) * lcs.z_std_np * 0.5
        goal = lcs.stage_goals[-1]
        new = _solves(tree, lcs, prm, z0, p0, rot0, goal, q)
        if pre is not None:
            plcs = pre.TorchLcs.from_deploy(dep, DEVICE)
            old = _solves(pre, plcs, params_for(pre, cfg), z0, p0, rot0, goal, q)
            for (Ua, ia), (Ub, ib) in zip(old, new):
                _require(np.array_equal(Ua, Ub) and _same(ia, ib), f"{cfg.name}: pre != tree")
                n += 1
        wide = dataclasses.replace(prm, tilt_max_deg={"franka": 90.0, "ur": 90.0})
        on = _solves(tree, lcs, wide, z0, p0, rot0, goal, q)
        for (Ua, _), (Ub, ib) in zip(new, on):
            _require(np.array_equal(Ua, Ub) and ib["frac_tilt_penalised"] == 0.0,
                     f"{cfg.name}: non-binding tilt != off")
        msgs.append(f"{cfg.parent.parent.name}/{cfg.name} (clearance "
                    f"{'on' if prm.clearance else 'off'})")
    how = "fused + graph" if DEVICE == "cuda" else "torch, CPU"
    pre_msg = (f"pre == tree on {n} solves (plans + every info field)" if pre is not None
               else "no --pre-dir: pre vs tree skipped")
    return f"{pre_msg}; tilt 90 deg (non-binding) == off; {', '.join(msgs)}; {how}"


def t7() -> str:
    lcs = tree.TorchLcs.from_deploy(V49C_DEP, DEVICE)
    prm = params_for(tree, CFG_96)
    rng = np.random.default_rng(4)
    p0, rot0, q = frame_state(0)
    z0 = lcs.np["z_goal"] + rng.normal(size=lcs.n_x) * lcs.z_std_np * 0.5
    goal = lcs.stage_goals[-1]
    g = torch.Generator(device=lcs.device).manual_seed(2)
    U, info = tree.MppiSolver(lcs, prm).solve(z0, p0, rot0, goal, None, g, None, q)
    now = float(np.degrees(scipy_tilt(q[0])))
    knots = np.degrees(scipy_chain(q[0], U[:, 6:9]))
    _require(abs(info["tilt_now_f_deg"] - now) < 1e-3, f"tilt_now {info['tilt_now_f_deg']} {now}")
    _require(abs(info["tilt_plan_max_f_deg"] - knots[1:].max()) < 1e-2, "tilt_plan_max")
    c, nb, _ = tree.rollout_cost(lcs, prm, z0, p0, rot0, goal, U, q_now=q)
    rel = abs(c - info["plan_cost"]) / max(abs(c), 1.0)
    _require(rel < 1e-4 and nb == info["plan_violations"], f"plan_cost rel {rel:.1e} n {nb}")
    _require(knots[1:].max() <= 32.0, f"plan tilts to {knots.max():.1f} from a {now:.1f} start")
    msg = (f"start tilt {now:.2f} deg (info {info['tilt_now_f_deg']:.2f}); plan max "
           f"{knots[1:].max():.2f} deg (info {info['tilt_plan_max_f_deg']:.2f}); plan_cost == "
           f"rollout_cost (rel {rel:.1e}), violations {nb}")
    if DEVICE == "cuda":
        res = []
        for use in (True, False):
            g = torch.Generator(device=lcs.device).manual_seed(2)
            res.append(tree.MppiSolver(lcs, prm, use_graph=use).solve(
                z0, p0, rot0, goal, None, g, None, q)[0])
        _require(np.array_equal(*res), "graph != eager")
        msg += "; graph == eager"
    q40 = np.stack([q_tilted(40.0, (0.0, 1.0, 0.0)), q[1]])
    g = torch.Generator(device=lcs.device).manual_seed(2)
    U4, i4 = tree.MppiSolver(lcs, prm).solve(z0, p0, rot0, goal, None, g, None, q40)
    k4 = np.degrees(scipy_chain(q40[0], U4[:, 6:9]))
    _require(k4[-1] < 40.0 - 5.0, f"40 deg start: plan tilt {k4.round(1).tolist()}")
    return msg + (f"; 40 deg start: plan tilt {k4[0]:.0f} -> {k4[-1]:.1f} deg, violations "
                  f"{i4['plan_violations']}, frac_tilt_penalised {i4['frac_tilt_penalised']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="pre-change tree (repo layout: src/task_common/mppi_{lcs,warp}.py)")
    ap.add_argument("--out", type=Path, default=None, help="json report")
    a = ap.parse_args()
    pre = load_pre(a.pre_dir) if a.pre_dir else None
    res = {}
    for name, fn in (("T1", lambda: t1(pre)), ("T2", lambda: t2(pre)), ("T3", t3), ("T4", t4),
                     ("T5", t5), ("T6", lambda: t6(pre)), ("T7", t7)):
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
