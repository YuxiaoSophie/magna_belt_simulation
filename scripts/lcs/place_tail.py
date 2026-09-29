"""Post-``place_3`` contact tail of ``collect_lcs_dataset.py --tail`` (opt-in).

After the ``place_3`` settle window the sim keeps running: one scripted smooth motion of both
arms (a family drawn per episode) starts from the commanded hold poses (the Franka's latch = its
measured pose at the latch, the UR's last IK target: no jump), runs ``U[min_s, max_s]`` and holds
0.5 s. As ``collect_motion_primitives.run_primitive``: Franka 7 knots on the scripted pose, UR
per-step IK + a 2-knot line pose(t) -> pose(t + dt), no excitation, per-step caps. Live guards:
grasp (a loss discards the episode), belt crop box (x/y + top), 2F-85 board clearance
(``UrExciteGuard``), rod stretch gain, Franka tip floor; any other guard ends the tail early with
the frames kept.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import collect_motion_primitives as cmp

from round_belt_task.arm_kinematics import UrTracking, ik, rot_axis_angle
from round_belt_task.commander import (
    FrankaCommand,
    UrCommand,
    UrLine,
    UrTarget,
    mat3_to_quat,
    pose_mat,
    saved_traj_message,
)
from task_common import lcs_dataset as lcs

FAMILIES = ("groove_slide", "press_deeper", "lift_repress", "partial_pullout",
            "tension_release", "random_contact")
GENTLE_FAMILIES = ("groove_slide", "tension_release", "lift_repress")
GENTLE_MUL = 0.3
GENTLE_LIFT_MAX_MM = 3.0
HOLD_S = 0.5
FRANKA_FLOOR_MM = 12.0
CLEAR_TOL_MM = 0.2  # the tail may not lower the 2F-85 clearance below min(min, start) - this
CROP_TOP_TOL_MM = 2.0
RANDOM_AMP = (10.0, 2.5)  # mm, deg: 0.5 x the primitives' default --amp-mm / --amp-deg
RANDOM_HOLD_S = 1.0
PHASE_HOLD = "hold:tail"
EZ = np.array([0.0, 0.0, 1.0])


def phase_label(family: str) -> str:
    return f"tail:{family}"


def draw_spec(cfg: dict, seed: int, i: int, band: bool, dt_s: float,
              amp_mul: float = 1.0) -> dict:
    """The episode's tail: family, duration, amplitudes; draws in a fixed order (RNG [seed, i,
    15]) so a retry at ``amp_mul`` 0.5 replays the same family and shape."""
    rng = np.random.default_rng([seed, i, 15])
    gentle = cfg["gentle"] == "on" or (cfg["gentle"] == "auto" and band)
    allowed = tuple(cfg["families"] or (GENTLE_FAMILIES if gentle else FAMILIES))
    bad = [f for f in allowed if f not in FAMILIES]
    if bad:
        raise ValueError(f"unknown tail families {bad} (have {FAMILIES})")
    u = rng.uniform(size=8)
    family = allowed[int(u[0] * len(allowed))]
    T = cfg["min_s"] + u[1] * (cfg["max_s"] - cfg["min_s"])
    T = dt_s * max(1, round(T / dt_s))
    mul = (GENTLE_MUL if gentle else 1.0) * amp_mul
    sign = 1.0 if u[2] < 0.5 else -1.0
    params = {
        "groove_slide": {"amp_deg": 10.0 * sign},
        "press_deeper": {"amp_mm": 3.0 + 5.0 * u[3]},
        "lift_repress": {"amp_mm": 5.0 + 7.0 * u[3], "bumps": 1 + int(u[4] < 0.5)},
        "partial_pullout": {"amp_mm": 8.0 + 7.0 * u[3]},
        "tension_release": {"amp_mm": 2.0 + 3.0 * u[3]},
        "random_contact": {"amp_mm": RANDOM_AMP[0], "amp_deg": RANDOM_AMP[1],
                           "hold_tau": 0.3 + 0.4 * u[5], "hold_s": RANDOM_HOLD_S,
                           "shape_seed": [seed, i, 15, 1]},
    }[family]
    drawn = dict(params)
    for k in ("amp_deg", "amp_mm"):
        if k in params:
            params[k] = params[k] * mul
    if gentle and family == "lift_repress":
        params["amp_mm"] = min(params["amp_mm"], GENTLE_LIFT_MAX_MM)
    return {"family": family, "T_s": float(T), "hold_s": HOLD_S, "gentle": bool(gentle),
            "amp_mul": float(mul), "params": params, "params_drawn": drawn,
            "allowed": list(allowed)}


def one_hold_warp(tau0: float, frac: float):
    """Monotone tau(s) with one plateau of ``frac`` (of T) at ``tau0``, eased on both sides."""
    s1 = tau0 * (1.0 - frac)
    s2 = s1 + frac

    def warp(s: float) -> float:
        if s <= s1:
            return float(tau0 * cmp.smoothstep(s / s1))
        if s <= s2:
            return float(tau0)
        return float(tau0 + (1.0 - tau0) * cmp.smoothstep((s - s2) / (1.0 - s2)))

    return warp, [{"tau": tau0, "s0": s1, "s1": s2}]


class TailMotion(cmp.Motion):
    """:class:`Motion` plus ``orbit`` components: rotation of the whole EE pose by ``amp *
    shape`` [rad] about the pulley axis through its origin."""

    def __init__(self, comps, X_f0, X_u0, t_start, T, pulley: dict | None, warp=None) -> None:
        super().__init__(comps, X_f0, X_u0, t_start, T, 1.0, warp)
        self.pulley = pulley

    def offset(self, arm: str, t: float, scale: dict | None = None) -> np.ndarray:
        tau = (t - self.t_start) / self.T
        out = np.zeros(6)
        if tau <= 0.0 or tau >= 1.0:
            return out
        if self.warp is not None:
            tau = self.warp(tau)
        ang = 0.0
        for c in self.comps:
            if c.arm != arm:
                continue
            s = 1.0 if scale is None else scale.get(c.group, 1.0)
            v = c.amp * s * self.value(c, tau)
            if c.kind == "orbit":
                ang += v
            elif c.kind == "trans":
                out[:3] += v * c.direction
            else:
                out[3:] += v * c.direction
        if ang:
            axis, centre = self.pulley["axis"], self.pulley["centre"]
            X0 = self.X0[arm]
            r = X0[:3, 3] - centre
            out[:3] += rot_axis_angle(axis, ang) @ r - r
            out[3:] += ang * (X0[:3, :3].T @ axis)  # world rotation as a tool rotvec
        return out


def components(spec: dict, X_f0, X_u0, pulley: dict, T: float) -> tuple[list, object, list]:
    """``(components, warp, holds)`` of ``spec`` at the base poses."""
    C = cmp.Component
    fam, par = spec["family"], spec["params"]
    g = X_u0[:3, 3] - X_f0[:3, 3]
    g[2] = 0.0
    G = g / np.linalg.norm(g)
    warp, holds = None, []
    if fam == "groove_slide":
        a = math.radians(par["amp_deg"])
        comps = [C("f_orbit", "franka", "orbit", pulley["axis"], cmp.osc(1), a, "all"),
                 C("u_orbit", "ur", "orbit", pulley["axis"], cmp.osc(1), a, "all")]
    elif fam == "press_deeper":  # per-arm groups: the board guard may stop the UR only
        a = par["amp_mm"] * 1e-3
        comps = [C("f_z", "franka", "trans", -EZ, cmp.bump(1), a, "franka"),
                 C("u_z", "ur", "trans", -EZ, cmp.bump(1), a, "ur")]
    elif fam == "lift_repress":
        a, b = par["amp_mm"] * 1e-3, cmp.bump(par["bumps"])
        comps = [C("f_z", "franka", "trans", EZ, b, a, "all"),
                 C("u_z", "ur", "trans", EZ, b, a, "all")]
    elif fam == "partial_pullout":  # rigid shift along the tangent, UR grasp away from the pulley
        away = G if float((X_u0[:3, 3] - pulley["centre"]) @ G) >= 0.0 else -G
        a = par["amp_mm"] * 1e-3
        comps = [C("f_pull", "franka", "trans", away, cmp.bump(1), a, "all"),
                 C("u_pull", "ur", "trans", away, cmp.bump(1), a, "all")]
    elif fam == "tension_release":
        a = par["amp_mm"] * 1e-3
        comps = [C("f_grasp", "franka", "trans", -G, cmp.osc(2), a, "all", 1),
                 C("u_grasp", "ur", "trans", G, cmp.osc(2), a, "all", 1)]
    elif fam == "random_contact":
        rng = np.random.default_rng(par["shape_seed"])
        a_m, a_r = par["amp_mm"] * 1e-3, math.radians(par["amp_deg"])
        comps = [c for arm in ("franka", "ur")
                 for c in cmp._random_arm(arm, rng, cmp.RANDOM_T_REF, a_m, a_r)]
        warp, holds = one_hold_warp(par["hold_tau"], min(0.5, par["hold_s"] / T))
    else:
        raise KeyError(fam)
    return comps, warp, holds


def _caps(cap: list[float]) -> tuple[np.ndarray, np.ndarray]:
    f_mm, f_mrad, u_mm, u_mrad = (x * 1e-3 for x in cap)
    ub = np.array([f_mm] * 3 + [u_mm] * 3 + [f_mrad] * 3 + [u_mrad] * 3)  # action_vector
    return -ub, ub


def run_tail(sim, ctx, col, sampler, state: dict, s0: int, q_ur, franka_cmdr, ur_cmdr,
             spec: dict, rods: list) -> dict:
    """Run the tail from the current step; appends frames to ``sampler`` and ``rods``.

    Returns the tail record (``grasp_lost`` True = discard the episode)."""
    args, n, params = ctx.args, ctx.n, ctx.params
    cfg = ctx.opts["tail_cfg"]
    dt_s = n * lcs.SIM_DT_S
    t0 = time.perf_counter()
    X_f0 = pose_mat(franka_cmdr.reached_pos, franka_cmdr.reached_quat)
    X_u0 = UrTracking.fk(q_ur)
    hand, byte = franka_cmdr.hand_mm, ur_cmdr.byte
    start_frame = len(sampler.frames) - 1
    t_start = sim.osc_time_s(sim.step_index + 1)
    T = spec["T_s"]
    comps, warp, holds = components(spec, X_f0, X_u0, ctx.pulley, T)
    motion = TailMotion(comps, X_f0, X_u0, t_start, T, ctx.pulley, warp)
    motion_n = round(T / dt_s)
    lb, ub = _caps(cfg["cap"])
    scale = cmp.plan_scales(motion, dt_s, motion_n + 1, lb, ub, 1.0)
    jaws = (byte,) if byte is not None else (None,)
    base_clear = min(ctx.gripper.predict(ctx.board, X_u0, j) for j in jaws)
    floor_c = min(args.min_clearance * 1e-3, base_clear) - CLEAR_TOL_MM * 1e-3
    groups = {arm: {c.group for c in comps if c.arm == arm} for arm in ("franka", "ur")}

    def shrink(arm: str, f: float) -> None:
        for gname in groups[arm]:
            scale[gname] = scale.get(gname, 1.0) * f

    def clearance_of(sc: dict) -> float:
        return cmp.plan_clearance(motion, sc, ctx.board, ctx.gripper, jaws, sim.frame_dt)

    if len(groups["ur"]) > 1:  # each UR group alone first: shrink only the offenders
        for gname in sorted(groups["ur"]):
            alone = {h: (scale.get(h, 1.0) if h == gname else 0.0) for h in scale}
            for _ in range(8):
                if clearance_of(alone) >= floor_c:
                    break
                alone[gname] *= 0.7
            scale[gname] = alone[gname] if clearance_of(alone) >= floor_c else 0.0
    clear = clearance_of(scale)
    for _ in range(8):  # UR groups only (rigid families share one group with the Franka)
        if clear >= floor_c:
            break
        shrink("ur", 0.7)
        clear = cmp.plan_clearance(motion, scale, ctx.board, ctx.gripper, jaws, sim.frame_dt)
    if clear < floor_c:
        shrink("ur", 0.0)
        clear = cmp.plan_clearance(motion, scale, ctx.board, ctx.gripper, jaws, sim.frame_dt)

    def tip_floor() -> float:
        return min(motion.pose("franka", t_start + k * dt_s, scale)[2, 3]
                   for k in range(motion_n + 1)) - ctx.plate_top_z

    tip = tip_floor()
    for _ in range(8):
        if tip * 1e3 >= FRANKA_FLOOR_MM:
            break
        shrink("franka", 0.7)
        tip = tip_floor()
    eff = {c.name: c.amp * scale.get(c.group, 1.0) * (1e3 if c.kind == "trans"
                                                      else 180.0 / math.pi) for c in comps}

    guard = col.UrExciteGuard(ctx.board, ctx.gripper, floor_c, 0.5)
    base_u = UrTarget(pos=X_u0[:3, 3].copy(), quat_wxyz=mat3_to_quat(X_u0[:3, :3]), byte=byte)
    u_coords = sim.arm_coords()[1]
    times_rel = np.arange(params.n_knots) * params.dt
    end_t = t_start + T
    k_end = state["tick"].k + (motion_n + round(HOLD_S / dt_s)) * n
    moving_label, hold_label = phase_label(spec["family"]), PHASE_HOLD
    pb0 = lcs.belt_points_ordered(sampler.frames[-1]["belt"])
    z_top_ref = max(float(cmp.CROP_HI[2]), float(pb0[:, 2].max()))
    rod0, n_rod0 = rods[-1], len(rods)
    live = {"tip_min": math.inf, "hook_s": 0.0}

    def hook(step, t, joint_q, body_q):
        h0 = time.perf_counter()
        ee_f = sim.franka_measured_pose7(joint_q)
        X_u = UrTracking.fk(joint_q[u_coords])
        live["tip_min"] = min(live["tip_min"], ee_f[2] - ctx.plate_top_z)
        moving = t < end_t
        ts = t + times_rel
        Xs = [motion.pose("franka", tk, scale) for tk in ts]
        kp = np.stack([X[:3, 3] for X in Xs])
        kq = np.stack([mat3_to_quat(X[:3, :3]) for X in Xs])
        cmd = FrankaCommand(knots_pos=kp, knots_quat=kq, times=ts, hold=not moving,
                            target_index=0, phase=moving_label if moving else hold_label,
                            hand_mm=hand)
        p0, q0 = ur_cmdr.to_tool0(motion.pose("ur", t, scale))
        if moving:
            p1, q1 = ur_cmdr.to_tool0(motion.pose("ur", t + params.dt, scale))
            line = UrLine(p0=p0, q0=q0, t0=float(t), p1=p1, q1=q1, t1=float(t) + params.dt)
        else:  # zero-span line: cmd_delta exactly 0
            line = UrLine(p0=p0, q0=q0, t0=0.0, p1=p0, q1=q0, t1=0.0)
        urc = UrCommand(line=line, regenerated=True, byte=byte, t=float(t),
                        X_tool0_tracking=ctx.x_tool0)
        state["tick"] = col.OscTick(step - s0, t, ee_f, X_u, cmd, urc, None, None, guard.scale)
        live["hook_s"] += time.perf_counter() - h0
        return saved_traj_message(round(t * 1e6), kp, kq, ts)

    lead_gain = sim.arm_kd / sim.arm_ke / sim.frame_dt
    scales, offs_f, offs_u = [], [], []
    stop, grasp_lost, first = None, False, True
    sim.commander_hook = hook
    try:
        while True:
            t1 = sim.osc_time_s(sim.step_index + 1)
            raw = motion.world_rotvec("ur", t1, scale)
            applied = guard.apply(t1, base_u, raw, jaws)
            X = col.UrExciteGuard.target(base_u, applied)
            q_new, err_p, err_r, _ = ik(UrTracking, pose_mat(X.pos, X.quat_wxyz), q_ur,
                                        pos_tol=col.UR_IK_POS_TOL, rot_tol=col.UR_IK_ROT_TOL)
            if err_p > col.UR_IK_POS_TOL or err_r > col.UR_IK_ROT_TOL:
                stop = f"ur ik miss at k {state['tick'].k + 1}"
                break
            lead = 0.0 if first else col.UR_VELOCITY_LEAD * lead_gain
            first = False
            sim.set_ur_target(q_new + lead * (q_new - q_ur))
            q_ur = q_new
            sim.set_grippers(hand, byte)
            sim.control_step()
            k = state["tick"].k
            if k % col.CLEARANCE_EVERY == 0 or k % n == 0:
                sampler.clearance(sim.state_0.body_q.numpy())
            if k % n != 0:
                continue
            sampler.sample_tick(state["tick"])
            rods.append(ctx.gauge.measure(sim.state_0.body_q.numpy())["stretch_pct"])
            scales.append(guard.scale)
            offs_f.append(motion.offset("franka", state["tick"].t, scale))
            offs_u.append(motion.offset("ur", state["tick"].t, scale))
            fr = sampler.frames[-1]
            pb = lcs.belt_points_ordered(fr["belt"])
            if not all(fr["grasp_ok"]):
                stop, grasp_lost = f"grasp lost at k {k} ({fr['grasp_ok'].tolist()})", True
                break
            if np.any(pb[:, :2] < cmp.CROP_LO[:2]) or np.any(pb[:, :2] > cmp.CROP_HI[:2]):
                stop = f"belt outside the crop x/y at k {k}"
                break
            if pb[:, 2].max() > z_top_ref + CROP_TOP_TOL_MM * 1e-3:
                stop = f"belt top {pb[:, 2].max():.4f} > {z_top_ref:.4f} + 2 mm at k {k}"
                break
            if rods[-1] - rod0 > cfg["stretch_cap_pct"]:
                stop = f"rod stretch gain {rods[-1] - rod0:.2f} % at k {k}"
                break
            if live["tip_min"] * 1e3 < FRANKA_FLOOR_MM:
                stop = f"franka tip {live['tip_min'] * 1e3:.1f} mm at k {k}"
                break
            if k >= k_end:
                break
    finally:
        sim.commander_hook = sim.make_hold_hook()
    frames = sampler.frames[start_frame + 1:]
    track = col.tracking_errors(sampler.frames[start_frame:])[1:]
    sc = np.asarray(scales) if scales else np.ones(1)
    return {
        "family": spec["family"], "gentle": spec["gentle"], "amp_mul": spec["amp_mul"],
        "params": spec["params"], "params_drawn": spec["params_drawn"],
        "allowed": spec["allowed"], "T_s": T, "hold_s": HOLD_S, "start_frame": start_frame,
        "frames": len(frames), "duration_s": len(frames) * dt_s,
        "stopped": stop, "grasp_lost": grasp_lost, "group_scale": scale,
        "amp_effective": eff, "unit": {c.name: "mm" if c.kind == "trans" else "deg"
                                       for c in comps},
        "holds": holds, "cap": cfg["cap"],
        "plan_min_2f85_clearance_mm": clear * 1e3, "start_2f85_clearance_mm": base_clear * 1e3,
        "clearance_floor_mm": floor_c * 1e3, "plan_franka_tip_floor_mm": tip * 1e3,
        "min_franka_tip_mm": live["tip_min"] * 1e3,
        "stretch_gain_max_pct": float(max(r - rod0 for r in rods[n_rod0 - 1:])),
        "ur_guard_scale_min": float(sc.min()), "ur_guard_scaled_frames": int((sc < 1.0).sum()),
        "tracking_rms_mm": [float(np.sqrt(np.mean(track[:, j] ** 2))) if len(track) else None
                            for j in range(2)],
        "wall_s": time.perf_counter() - t0, "hook_s": live["hook_s"],
        "_arrays": {"guard_scale": sc if scales else np.zeros(0),
                    "offset_franka": np.asarray(offs_f).reshape(-1, 6),
                    "offset_ur": np.asarray(offs_u).reshape(-1, 6)},
    }
