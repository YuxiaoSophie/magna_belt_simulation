#!/usr/bin/env python3
"""Scripted free-space motion primitives of both arms holding the belt (OOD one-step test).

Each episode restores ``pre_place_1_osc.npz`` under magna's OSC (the collector's OSC loop),
holds ``--pre-hold-s`` (u = 0), plays one smooth scripted EE motion of ``--motion-s`` and holds
``--post-hold-s``. The Franka gets 7 knots sampled on the scripted pose (knot j = pose(t + j dt)),
the UR a per-step IK target on it plus a 2-knot line pose(t) -> pose(t + dt), so ``cmd_delta`` =
pose(t + dt) - pose(t) exactly. Files are in the collector's format (``collect_lcs_dataset``'s
writer) plus ``sim_motion_*`` / ``sim_rod_stretch_pct`` extras.

Amplitudes are capped per primitive so the planned ``cmd_delta`` stays within
``--bound-frac`` of the v2 export's ``u_lb/u_ub`` (per dim, per sign) and the planned 2F-85 pose
keeps the board guard's clearance. Live: grasp, crop box, board clearance and rod stretch are
checked; a failing episode is retried at half amplitude.

Axes: ``long`` = the start belt's long axis (horizontal PCA, ~world y), ``grasp`` = horizontal
Franka -> UR grasp direction (~world x). Vertical motions only go down: the start belt is already
~3 mm above the crop box top (z 0.11 m).

Run:
    uv run python scripts/lcs/collect_motion_primitives.py \\
        --lcm-url 'udpm://239.255.76.131:7731?ttl=0' --record
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import collect_lcs_dataset as col
from loguru import logger
from make_flat_engaged_state import RodGauge

from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import UrTracking, ik, rot_axis_angle
from round_belt_task.commander import (
    CommanderParams,
    FrankaCommand,
    UrCommand,
    UrLine,
    UrLineCommander,
    UrTarget,
    mat3_to_quat,
    parse_saved_traj_message,
    pose_mat,
    saved_traj_message,
    x_tool0_tracking,
)
from round_belt_task.episode_io import (
    finish_recording,
    git_info,
    sha256_file,
    write_json,
)
from round_belt_task.outcome import DEFAULT_THRESHOLDS
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.osc_process import check_private_url

DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v2_20260925/"
              "deploy_v2_decoded_only/deploy.npz")
OUT_ROOT = REPO_ROOT / "data" / "lcs" / "motion_primitives"
CROP_LO = np.array([0.2, -0.25, 0.015])  # round_belt_scene.yaml cropped_point_cloud
CROP_HI = np.array([0.7, 0.25, 0.11])
FRANKA_MIN_TIP_MM = 15.0
PRIMITIVES = ("both_up_down", "both_fwd_back", "both_sideways", "opposite_fwd_back",
              "opposite_sideways", "opposite_up_down", "franka_only", "ur_only", "wrist_roll",
              "random_mix", "good_mix", "good_mix_tilt")
V1_PRIMITIVES = PRIMITIVES[:10]
EZ = np.array([0.0, 0.0, 1.0])
RANDOM_T_REF = 8.0  # random_mix shapes are drawn on this time base, then stretched to --motion-s
PULLEY_KEEPOUT = ((0.020, 0.015), (0.035, 0.015))  # small, large: (radius, height above) [m]


# ---- shapes (tau in [0, 1], peak |shape| = 1) ---------------------------------------------

def hann(tau):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.clip(tau, 0.0, 1.0))


def _norm(f):
    grid = np.linspace(0.0, 1.0, 4001)
    peak = float(np.abs(f(grid)).max())
    return lambda tau: f(np.asarray(tau, np.float64)) / peak


def osc(cycles: int = 1):
    """Two-sided: +, -, ... under a Hann envelope; 0 value and slope at both ends."""
    return _norm(lambda t: np.sin(2.0 * np.pi * cycles * t) * hann(t))


def bump(n: int = 1):
    """One-sided out-and-back, ``n`` times; in [0, 1]."""
    return _norm(lambda t: 0.5 - 0.5 * np.cos(2.0 * np.pi * n * np.clip(t, 0.0, 1.0)))


def slot(shape, i: int, n: int):
    """``shape`` squeezed into slot ``i`` of ``n``; 0 elsewhere."""
    def f(t):
        s = np.asarray(t, np.float64) * n - i
        return np.where((s >= 0.0) & (s <= 1.0), shape(np.clip(s, 0.0, 1.0)), 0.0)
    return f


def smooth_random(rng: np.random.Generator, tau_c: float, sigma: float, T: float,
                  n: int = 2001):
    """OU noise (correlation ``tau_c`` s) Gaussian-smoothed (``sigma`` s), peak 1."""
    dt = T / (n - 1)
    x = np.zeros(n + 400)
    a = math.exp(-dt / tau_c)
    for k in range(1, len(x)):
        x[k] = a * x[k - 1] + math.sqrt(1 - a * a) * rng.standard_normal()
    x = x[400:]
    w = np.exp(-0.5 * (np.arange(-4 * sigma, 4 * sigma + dt, dt) / sigma) ** 2)
    x = np.convolve(np.pad(x, len(w) // 2, mode="reflect"), w / w.sum(), mode="valid")[:n]
    grid = np.linspace(0.0, 1.0, n)
    x = x / np.abs(x).max()
    return lambda tau: np.interp(np.asarray(tau, np.float64), grid, x)


@dataclasses.dataclass
class Component:
    """``amp * dir * shape(tau)``: world translation [m] or tool-frame rotation vector [rad]."""

    name: str
    arm: str  # franka | ur
    kind: str  # trans | rot
    direction: np.ndarray
    shape: object
    amp: float
    group: str
    stretch_when: int = 0  # sign of the shape that pulls the grasps apart (0: neither)


def primitive_components(name: str, axes: dict, amp_m: float, amp_rad: float,
                         rng: np.random.Generator, T: float) -> list[Component]:
    L, G = axes["long"], axes["grasp"]
    C = Component
    o1, b1, b2 = osc(1), bump(1), bump(2)
    if name == "both_up_down":
        return [C("f_z", "franka", "trans", -EZ, b2, amp_m, "all"),
                C("u_z", "ur", "trans", -EZ, b2, amp_m, "all")]
    if name == "both_fwd_back":
        return [C("f_long", "franka", "trans", L, o1, amp_m, "all"),
                C("u_long", "ur", "trans", L, o1, amp_m, "all")]
    if name == "both_sideways":
        return [C("f_grasp", "franka", "trans", G, o1, amp_m, "all"),
                C("u_grasp", "ur", "trans", G, o1, amp_m, "all")]
    if name == "opposite_fwd_back":
        return [C("f_long", "franka", "trans", L, o1, amp_m, "all"),
                C("u_long", "ur", "trans", -L, o1, amp_m, "all")]
    if name == "opposite_sideways":  # Franka -grasp, UR +grasp: stretch then slack
        return [C("f_grasp", "franka", "trans", -G, o1, amp_m, "all", 1),
                C("u_grasp", "ur", "trans", G, o1, amp_m, "all", 1)]
    if name == "opposite_up_down":  # both <= 0; relative tilt +-amp
        s = _norm(lambda t: hann(t) * (1.0 + np.sin(4.0 * np.pi * t)))
        r = _norm(lambda t: hann(t) * (1.0 - np.sin(4.0 * np.pi * t)))
        return [C("f_z", "franka", "trans", -EZ, s, amp_m, "all"),
                C("u_z", "ur", "trans", -EZ, r, amp_m, "all")]
    if name in ("franka_only", "ur_only"):
        arm = name.split("_")[0]
        a = arm[0]
        return [C(f"{a}_grasp", arm, "trans", G, slot(o1, 0, 3), amp_m, "grasp",
                  -1 if arm == "franka" else 1),
                C(f"{a}_long", arm, "trans", L, slot(o1, 1, 3), amp_m, "long"),
                C(f"{a}_z", arm, "trans", -EZ, slot(b1, 2, 3), amp_m, "z")]
    if name == "wrist_roll":
        return [C("f_rz_tool", "franka", "rot", EZ, osc(2), amp_rad, "franka"),
                C("u_rz_tool", "ur", "rot", EZ, osc(2), amp_rad, "ur")]
    if name in ("good_mix", "good_mix_tilt"):
        # Axes v1 showed are captured: both arms together, vertical (down only) and long.
        ph = rng.uniform(0.0, 2.0 * np.pi, 6)
        tw = 2.0 * np.pi
        zs = _norm(lambda t: hann(t) * (1.0 + 0.55 * np.sin(tw * 1.3 * t + ph[0])
                                        + 0.45 * np.sin(tw * 2.17 * t + ph[1])))
        ls = _norm(lambda t: hann(t) * (np.sin(tw * 1.0 * t + ph[2])
                                        + 0.6 * np.sin(tw * 1.73 * t + ph[3])))
        a_z = amp_m if name == "good_mix" else 0.75 * amp_m
        out = [C("f_z", "franka", "trans", -EZ, zs, a_z, "z"),
               C("u_z", "ur", "trans", -EZ, zs, a_z, "z"),
               C("f_long", "franka", "trans", L, ls, amp_m, "long"),
               C("u_long", "ur", "trans", L, ls, amp_m, "long")]
        if name == "good_mix_tilt":  # small independent vertical offsets, down only
            df = _norm(lambda t: hann(t) * (1.0 + np.sin(tw * 2.9 * t + ph[4])))
            du = _norm(lambda t: hann(t) * (1.0 + np.sin(tw * 3.4 * t + ph[5])))
            out += [C("f_dz", "franka", "trans", -EZ, df, 0.25 * amp_m, "tilt"),
                    C("u_dz", "ur", "trans", -EZ, du, 0.25 * amp_m, "tilt")]
        return out
    if name == "random_mix":
        out = []
        for arm in ("franka", "ur"):
            for j, ax in enumerate(("x", "y", "z")):
                n = smooth_random(rng, 0.8, 0.25, T)
                d = np.eye(3)[j]
                sw = (-1 if arm == "franka" else 1) if ax == "x" else 0  # world x ~ grasp axis
                if ax == "z":  # down only
                    f = _norm(lambda t, n=n: hann(t) * (1.0 + np.tanh(1.5 * n(t))))
                    out.append(C(f"{arm[0]}_{ax}", arm, "trans", -d, f, amp_m, f"{arm}_{ax}"))
                else:
                    f = _norm(lambda t, n=n: hann(t) * n(t))
                    out.append(C(f"{arm[0]}_{ax}", arm, "trans", d, f, amp_m, f"{arm}_{ax}",
                                 sw))
            for j, ax in enumerate(("rx", "ry", "rz")):
                n = smooth_random(rng, 0.8, 0.25, T)
                f = _norm(lambda t, n=n: hann(t) * n(t))
                out.append(C(f"{arm[0]}_{ax}_tool", arm, "rot", np.eye(3)[j], f, amp_rad,
                             f"{arm}_rot"))
        return out
    raise KeyError(name)


class Motion:
    """Scripted poses of both arms: base poses + the components' offsets at time ``t``."""

    def __init__(self, comps: list[Component], X_f0: np.ndarray, X_u0: np.ndarray,
                 t_start: float, T: float, lobe: float = 1.0) -> None:
        self.comps, self.X0 = comps, {"franka": X_f0, "ur": X_u0}
        self.t_start, self.T, self.lobe = t_start, T, lobe

    def value(self, c: Component, tau: float) -> float:
        """``c``'s shape at ``tau``, its stretching lobe scaled by ``lobe``."""
        v = float(c.shape(tau))
        return v * self.lobe if c.stretch_when and v * c.stretch_when > 0.0 else v

    def offset(self, arm: str, t: float, scale: dict | None = None) -> np.ndarray:
        """``[dpos world, rotvec tool]`` at time ``t``."""
        tau = (t - self.t_start) / self.T
        out = np.zeros(6)
        if tau <= 0.0 or tau >= 1.0:
            return out
        for c in self.comps:
            if c.arm != arm:
                continue
            s = 1.0 if scale is None else scale.get(c.group, 1.0)
            v = c.amp * s * self.value(c, tau) * c.direction
            if c.kind == "trans":
                out[:3] += v
            else:
                out[3:] += v
        return out

    def pose(self, arm: str, t: float, scale: dict | None = None) -> np.ndarray:
        o = self.offset(arm, t, scale)
        X0 = self.X0[arm]
        X = X0.copy()
        X[:3, 3] += o[:3]
        ang = float(np.linalg.norm(o[3:]))
        if ang > 0.0:
            X[:3, :3] = X0[:3, :3] @ rot_axis_angle(o[3:], ang)
        return X

    def world_rotvec(self, arm: str, t: float, scale: dict | None = None) -> np.ndarray:
        """``[dpos, world rotvec]``: the ``UrExciteGuard`` offset convention."""
        o = self.offset(arm, t, scale)
        return np.concatenate([o[:3], self.X0[arm][:3, :3] @ o[3:]])


def pose7(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X[:3, 3], mat3_to_quat(X[:3, :3])])


def planned_actions(motion: Motion, dt: float, scale: dict | None, ticks: int) -> np.ndarray:
    """``(ticks, 12)`` cmd_delta at the sample ticks ``t_start + k dt``."""
    out = np.zeros((ticks, lcs.ACTION_DIM))
    for k in range(ticks):
        t = motion.t_start + k * dt
        out[k] = lcs.action_vector(pose7(motion.pose("franka", t, scale)),
                                   pose7(motion.pose("franka", t + dt, scale)),
                                   pose7(motion.pose("ur", t, scale)),
                                   pose7(motion.pose("ur", t + dt, scale)))
    return out


def bound_ratio(u: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """``u / ub`` for u > 0, ``u / lb`` for u < 0 (>= 0; 1 = at the bound)."""
    return np.where(u >= 0.0, u / ub, u / lb)


def plan_scales(motion: Motion, dt: float, ticks: int, lb, ub, frac: float) -> dict:
    """Per group: the largest scale <= 1 whose own actions stay within ``frac`` of the bounds;
    then one common factor if the groups together exceed it."""
    groups = sorted({c.group for c in motion.comps})
    scale = {}
    for g in groups:
        only = {h: (1.0 if h == g else 0.0) for h in groups}
        for _ in range(3):  # rotations are mildly nonlinear in the amplitude
            r = bound_ratio(planned_actions(motion, dt, {h: only[h] * scale.get(h, 1.0)
                                                         for h in groups}, ticks), lb, ub).max()
            s = scale.get(g, 1.0)
            if r <= frac * (1 + 1e-6):
                break
            scale[g] = min(1.0, s * frac / r)
        scale.setdefault(g, 1.0)
    for _ in range(4):
        r = bound_ratio(planned_actions(motion, dt, scale, ticks), lb, ub).max()
        if r <= frac * (1 + 1e-6):
            break
        scale = {g: s * frac / r for g, s in scale.items()}
    return scale


def plan_duration(motion: Motion, dt: float, lb, ub, frac: float, t_min: float,
                  t_max: float) -> float:
    """Shortest duration (multiple of ``dt``, clipped to [t_min, t_max]) at which the full
    amplitude keeps every action within ``frac`` of the bounds."""
    T = motion.T
    for _ in range(5):
        motion.T = T
        r = bound_ratio(planned_actions(motion, dt, None, round(T / dt) + 1), lb, ub).max()
        T_new = min(max(T * r / frac, t_min), t_max)
        T_new = min(dt * math.ceil(T_new / dt - 1e-9), dt * math.floor(t_max / dt + 1e-9))
        if abs(T_new - T) < 0.5 * dt:
            break
        T = T_new
    motion.T = T
    return T


def lobe_amplitudes(motion: Motion, scale: dict) -> dict:
    """Per component: planned peak (+) and trough (-) amplitude, mm or deg."""
    grid = np.linspace(0.0, 1.0, 2001)
    out = {}
    for c in motion.comps:
        v = np.array([motion.value(c, t) for t in grid]) * c.amp * scale.get(c.group, 1.0)
        k = 1e3 if c.kind == "trans" else 180.0 / math.pi
        out[c.name] = [float(v.max() * k), float(v.min() * k)]
    return out


def plan_clearance(motion: Motion, scale: dict, board, gripper, jaws, step_dt: float
                   ) -> float:
    """Min predicted 2F-85 clearance [m] along the planned UR poses (every control step)."""
    low = math.inf
    n = round(motion.T / step_dt)
    for k in range(0, n + 1, 3):
        X = motion.pose("ur", motion.t_start + k * step_dt, scale)
        low = min(low, min(gripper.predict(board, X, jaw) for jaw in jaws))
    return low


def axes_from_start(belt: np.ndarray, X_f: np.ndarray, X_u: np.ndarray) -> dict:
    xy = belt[:, :2] - belt[:, :2].mean(0)
    _, v = np.linalg.eigh(xy.T @ xy)
    long = np.array([*v[:, -1], 0.0])
    if long[1] < 0:
        long = -long
    g = X_u[:3, 3] - X_f[:3, 3]
    g[2] = 0.0
    return {"long": long, "grasp": g / np.linalg.norm(g)}


def crop_margins(pcd_belt: np.ndarray) -> dict:
    """Per-frame margins [mm] of the belt points to the crop box faces (< 0 = outside)."""
    lo = (pcd_belt - CROP_LO).min(axis=1) * 1e3
    hi = (CROP_HI - pcd_belt).min(axis=1) * 1e3
    return {"z_top_mm": hi[:, 2], "z_bottom_mm": lo[:, 2],
            "xy_mm": np.minimum(lo[:, :2], hi[:, :2]).min(axis=1),
            "belt_z_max_m": pcd_belt[..., 2].max(axis=1)}


# ---- one episode --------------------------------------------------------------------------

def run_primitive(sim, ctx, i: int, name: str, amp_mul: float, gauge: RodGauge, lb, ub,
                  rng_seed: int, lobe: float = 1.0) -> dict:
    args, n, params = ctx.args, ctx.n, ctx.params
    t0 = time.perf_counter()
    dt_s = n * lcs.SIM_DT_S
    fname = f"episode_{i:04d}"
    label = f"{fname}-{name}"
    row = {"file": f"{fname}.npz", "primitive": name, "amp_mul": amp_mul, "stretch_lobe": lobe}
    sim.restore(ctx.snap, settle_steps=0)
    grasp = sim.settle(round(args.osc_settle_s / sim.frame_dt))
    if grasp.held() != (True, True):
        return {**row, "status": "failed", "reason": "grasp lost after settle"}
    hand0, byte0 = sim.gripper_commands()
    q_ur = sim.arm_targets()[1]
    body_q = sim.state_0.body_q.numpy()
    rod0 = gauge.measure(body_q)["stretch_pct"]
    belt0 = lcs.belt_points_ordered(body_q[sim.info.belt_bodies, :3].astype(np.float64))
    z_top0 = float(belt0[:, 2].max())

    phase_labels = [f"move:{name}", f"hold:{name}", "done", col.PREHOLD_PHASE]
    pre_n = max(0, round(args.pre_hold_s / dt_s))
    post_n = round(args.post_hold_s / dt_s)
    s0 = sim.step_index + 1 + pre_n * n
    t_start = sim.osc_time_s(s0)
    # Bases: the Franka hold latch (no jump), the UR's commanded tracking pose.
    pos, quat, _ = parse_saved_traj_message(sim.commander_hook(
        sim.step_index, sim.osc_time_s(), sim.state_0.joint_q.numpy(), body_q))
    X_f0 = pose_mat(pos[0], quat[0])
    X_u0 = UrTracking.fk(q_ur)
    axes = axes_from_start(belt0, X_f0, X_u0)
    rng = np.random.default_rng([rng_seed, PRIMITIVES.index(name)])
    comps = primitive_components(name, axes, args.amp_mm * 1e-3 * amp_mul,
                                 math.radians(args.amp_deg) * amp_mul, rng, RANDOM_T_REF)
    auto = args.motion_s <= 0.0
    motion = Motion(comps, X_f0, X_u0, t_start, RANDOM_T_REF if auto else args.motion_s, lobe)
    if auto:
        plan_duration(motion, dt_s, lb, ub, args.bound_frac, args.motion_s_min,
                      args.motion_s_max)
    motion_s = motion.T
    motion_n = round(motion_s / dt_s)
    scale = plan_scales(motion, dt_s, motion_n + 1, lb, ub, args.bound_frac)
    jaws = (byte0,) if byte0 is not None else (None,)
    min_c = args.min_clearance * 1e-3
    for _ in range(8):  # board guard at plan time: shrink until the 2F-85 keeps min_c
        clear = plan_clearance(motion, scale, ctx.board, ctx.gripper, jaws, sim.frame_dt)
        if clear >= min_c:
            break
        scale = {g: 0.7 * s for g, s in scale.items()}
    tip_floor = min(motion.pose("franka", t_start + k * dt_s, scale)[2, 3]
                    for k in range(motion_n + 1)) - ctx.plate_top_z
    if tip_floor * 1e3 < FRANKA_MIN_TIP_MM:
        return {**row, "status": "failed", "reason": f"franka tip floor {tip_floor * 1e3:.1f} mm"}
    u_plan = planned_actions(motion, dt_s, scale, motion_n + 1)
    eff = {c.name: c.amp * scale.get(c.group, 1.0) * (1e3 if c.kind == "trans"
                                                      else 180.0 / math.pi) for c in comps}
    logger.info(f"[PRIM] {i} {name} x{amp_mul:g}: scales {({g: round(s, 3) for g, s in scale.items()})}"
                f", effective amp (mm|deg) {({k: round(v, 2) for k, v in eff.items()})}, plan "
                f"bound ratio max {bound_ratio(u_plan, lb, ub).max():.3f}, plan 2F-85 clearance "
                f"{clear * 1e3:.1f} mm, franka tip floor {tip_floor * 1e3:.1f} mm")

    guard = col.UrExciteGuard(ctx.board, ctx.gripper, min_c, 0.5)
    base_u = UrTarget(pos=X_u0[:3, 3].copy(), quat_wxyz=mat3_to_quat(X_u0[:3, :3]), byte=byte0)
    ur_cmdr = UrLineCommander([None], params, X_tool0_tracking=ctx.x_tool0, byte=byte0)
    u_coords = sim.arm_coords()[1]
    times_rel = np.arange(params.n_knots) * params.dt
    end_t = t_start + motion_s
    pulleys = body_q[np.asarray(sim.info.pulley_bodies), :3].astype(np.float64)
    state = {"tick": None, "hook_s": 0.0, "tip_min": math.inf, "ur_raw": np.zeros(6),
             "ur_base": base_u}

    def hook(step, t, joint_q, body_q):
        h0 = time.perf_counter()
        k = step - s0
        ee_f = sim.franka_measured_pose7(joint_q)
        X_u = UrTracking.fk(joint_q[u_coords])
        state["tip_min"] = min(state["tip_min"], ee_f[2] - ctx.plate_top_z)
        moving = t < end_t
        ts = t + times_rel
        Xs = [motion.pose("franka", tk, scale) for tk in ts]
        kp = np.stack([X[:3, 3] for X in Xs])
        kq = np.stack([mat3_to_quat(X[:3, :3]) for X in Xs])
        cmd = FrankaCommand(knots_pos=kp, knots_quat=kq, times=ts, hold=not moving,
                            target_index=0, phase=phase_labels[0 if moving else 1],
                            hand_mm=hand0)
        p0, q0 = ur_cmdr.to_tool0(motion.pose("ur", t, scale))
        if moving:
            p1, q1 = ur_cmdr.to_tool0(motion.pose("ur", t + params.dt, scale))
            line = UrLine(p0=p0, q0=q0, t0=float(t), p1=p1, q1=q1, t1=float(t) + params.dt)
        else:  # zero-span line: cmd_delta exactly 0
            line = UrLine(p0=p0, q0=q0, t0=0.0, p1=p0, q1=q0, t1=0.0)
        urc = UrCommand(line=line, regenerated=True, byte=byte0, t=float(t),
                        X_tool0_tracking=ctx.x_tool0)
        state["tick"] = col.OscTick(k, t, ee_f, X_u, cmd, urc, None, None, guard.scale)
        state["hook_s"] += time.perf_counter() - h0
        return saved_traj_message(round(t * 1e6), kp, kq, ts)

    if args.record:
        sim.start_recording(ctx.out / "recordings", label)
    sampler = col.OscEpisodeSampler(sim, ctx, phase_labels)
    if pre_n > 0:
        col._pre_hold(sim, ctx, sampler, state, pre_n * n, s0, q_ur, ur_cmdr, hand0, byte0)
    lead_gain = sim.arm_kd / sim.arm_ke / sim.frame_dt
    total = (motion_n + post_n) * n
    rods, scales, offs_f, offs_u = [], [], [], []
    fail = None
    first = True
    sim.commander_hook = hook
    try:
        while True:
            tick = state["tick"]
            if tick is not None:
                t1 = sim.osc_time_s(sim.step_index + 1)
                raw = motion.world_rotvec("ur", t1, scale)
                applied = guard.apply(t1, base_u, raw, jaws)
                X = col.UrExciteGuard.target(base_u, applied)
                X = pose_mat(X.pos, X.quat_wxyz)
                q_new, err_p, err_r, _ = ik(UrTracking, X, q_ur, pos_tol=col.UR_IK_POS_TOL,
                                                rot_tol=col.UR_IK_ROT_TOL)
                if err_p > col.UR_IK_POS_TOL or err_r > col.UR_IK_ROT_TOL:
                    fail = f"UR IK miss at k {tick.k + 1}"
                    break
                lead = 0.0 if first else col.UR_VELOCITY_LEAD * lead_gain
                first = False
                sim.set_ur_target(q_new + lead * (q_new - q_ur))
                q_ur = q_new
                sim.set_grippers(hand0, byte0)
            sim.control_step()
            tick = state["tick"]
            k = tick.k
            if k % col.CLEARANCE_EVERY == 0 or k % n == 0:
                sampler.clearance(sim.state_0.body_q.numpy())
            if k % n == 0:
                sampler.sample_tick(tick)
                bq = sim.state_0.body_q.numpy()
                rods.append(gauge.measure(bq)["stretch_pct"])
                scales.append(guard.scale)
                offs_f.append(motion.offset("franka", tick.t, scale))
                offs_u.append(motion.offset("ur", tick.t, scale))
                fr = sampler.frames[-1]
                if not all(fr["grasp_ok"]):
                    fail = f"grasp lost at k {k} ({fr['grasp_ok'].tolist()})"
                    break
                pb = lcs.belt_points_ordered(fr["belt"])
                if pb[:, 2].max() > z_top0 + args.crop_z_tol_mm * 1e-3:
                    fail = (f"belt z {pb[:, 2].max():.4f} > start {z_top0:.4f} + "
                            f"{args.crop_z_tol_mm:g} mm at k {k}")
                    break
                if np.any(pb < CROP_LO) or np.any(pb[:, :2] > CROP_HI[:2]):
                    fail = f"belt outside the crop box x/y/z-bottom at k {k}"
                    break
                near = [bool(np.any((np.linalg.norm(pb[:, :2] - pc[:2], axis=1) < r)
                                    & (pb[:, 2] < pc[2] + h)))
                        for pc, (r, h) in zip(pulleys, PULLEY_KEEPOUT, strict=True)]
                if any(near):
                    fail = f"belt near a pulley ({near}) at k {k}"
                    break
                if rods[-1] - rods[0] > args.stretch_cap_pct:
                    fail = f"rod stretch gain {rods[-1] - rods[0]:.2f} % at k {k}"
                    break
                if k >= total:
                    break
    finally:
        sim.commander_hook = sim.make_hold_hook()
    if fail is not None:
        finish_recording(sim, ctx.out, label, "episode failed")
        logger.warning(f"[PRIM] {i} {name} x{amp_mul:g} failed: {fail}")
        return {**row, "status": "failed", "reason": fail}
    recording = finish_recording(sim, ctx.out, label, "episode done")

    t3 = time.perf_counter()
    frames = sampler.frames
    pre = np.array([phase_labels[fr["phase"]] == col.PREHOLD_PHASE for fr in frames])
    u_all = np.stack([col.frame_action(fr, "cmd_delta") for fr in frames])
    if np.any(u_all[pre] != 0.0):
        raise RuntimeError("pre-hold cmd_delta not 0")
    label_out, metrics, min_clear_mm, contact = col._classify(sampler, ctx.thresholds,
                                                              ctx.tangent)
    ratio = bound_ratio(u_all, lb, ub)
    rmax = ratio.max(axis=1)
    moving = ~pre
    pb_all = np.stack([lcs.belt_points_ordered(fr["belt"]) for fr in frames])
    crop = crop_margins(pb_all)
    rod = np.asarray(rods)
    stats = {
        "frames": len(frames), "pre_hold_frames": int(pre.sum()),
        "u_bound_ratio_max": float(rmax.max()),
        "u_bound_ratio_max_per_dim": ratio.max(axis=0).round(4).tolist(),
        "u_near_bound_frac": {f">={q:g}": float((rmax[moving] >= q).mean())
                              for q in (0.5, 0.75, 0.9, 1.0)},
        "u_plan_max_err": float(np.abs(u_all[pre.sum():pre.sum() + len(u_plan)]
                                       - u_plan[:moving.sum()]).max()),
        "crop_margin_min_mm": {"z_top": float(crop["z_top_mm"].min()),
                               "z_top_start": float(crop["z_top_mm"][0]),
                               "z_bottom": float(crop["z_bottom_mm"].min()),
                               "xy": float(crop["xy_mm"].min())},
        "belt_z_max_m": float(crop["belt_z_max_m"].max()),
        "belt_z_max_start_m": float(crop["belt_z_max_m"][0]),
        "rod_stretch_pct_settle": float(rod0),
        "rod_stretch_pct_motion_start": float(rod[0]),
        "rod_stretch_gain_pct_max": float((rod - rod[0]).max()),  # vs the first motion frame
        "rod_stretch_gain_pct_min": float((rod - rod[0]).min()),
        "ur_guard_scale_min": float(min(scales)),
        "min_board_clearance_mm": float(min_clear_mm), "board_contact": bool(contact),
        "min_franka_tip_clearance_mm": float(state["tip_min"] * 1e3),
        "grasp_ok_all": bool(all(all(fr["grasp_ok"]) for fr in frames)),
        "outcome_label": label_out, "final_wrap_deg": float(metrics["wrap_deg"][-1]),
        "wrap_deg_max": float(np.max(metrics["wrap_deg"])),
    }
    track = col.tracking_errors(frames)
    stats["tracking_rms_mm"] = [float(np.sqrt(np.mean(track[moving][1:, j] ** 2)))
                                for j in range(2)]
    plan_meta = {"primitive": name, "components": [
        {"name": c.name, "arm": c.arm, "kind": c.kind, "direction": c.direction.tolist(),
         "group": c.group, "amp_requested": c.amp * (1e3 if c.kind == "trans"
                                                     else 180 / math.pi),
         "amp_effective": eff[c.name], "unit": "mm" if c.kind == "trans" else "deg"}
        for c in comps], "group_scale": scale, "axes": {k: v.tolist() for k, v in axes.items()},
        "motion_s": motion_s, "stretch_lobe": lobe, "lobe_amplitudes": lobe_amplitudes(
            motion, scale), "action_ood": args.bound_frac > 1.0, "pre_hold_s": args.pre_hold_s,
        "post_hold_s": args.post_hold_s, "bound_frac": args.bound_frac, "amp_mul": amp_mul,
        "plan_min_2f85_clearance_mm": clear * 1e3, "plan_franka_tip_floor_mm": tip_floor * 1e3}
    extras_meta = {
        "phase_labels": phase_labels, "intent": name, "outcome": label_out,
        "perturbation": {"intent": name}, "backend": "osc", "scenario": "motion_primitive",
        "motion_primitive": plan_meta,
        "ee_pose_source": "measured finger_tip / tracking frame (FK of the measured joints)",
        "action_source": "cmd_delta of the scripted pose: Franka knot 1 - knot 0 (knot j = "
                         "pose(t + j dt)); UR line pose(t) -> pose(t + dt)",
        "pre_hold_frames": int(pre.sum()), "osc": ctx.osc,
        "commander": dataclasses.asdict(params), "excitation": None,
        "sample_period_s": dt_s, "time_source": "OSC clock (FRANKA_STATE utime)",
        "pulley_pose_layout": lcs.POSE_LAYOUT, "start_step": s0,
        "start_state": str(ctx.start_state), "thresholds": dataclasses.asdict(ctx.thresholds),
        "clamp": None, "stats": stats,
    }
    osc_post = {"backend": np.array("osc"),
                "osc_utime_offset_us": np.int64(sim.bridge.utime_offset_us),
                "min_franka_tip_clearance_mm": np.float32(state["tip_min"] * 1e3),
                "primitive": np.array(name),
                "rod_stretch_pct": np.concatenate([np.full(pre.sum(), np.nan), rod]),
                "motion_offset_franka": np.concatenate([np.zeros((pre.sum(), 6)),
                                                        np.stack(offs_f)]),
                "motion_offset_ur": np.concatenate([np.zeros((pre.sum(), 6)), np.stack(offs_u)]),
                "ur_guard_scale": np.concatenate([np.ones(pre.sum()), np.asarray(scales)])}
    path = ctx.out / row["file"]
    writer, _ = col.build_osc_writer(frames, n, pcd=not args.no_pcd, definition="cmd_delta")
    timing = {"sim": t3 - t0}
    col._write_episode(i, pert.Perturbation(intent=name), path, writer, label_out, extras_meta,
                       metrics, min_clear_mm, contact, None, osc_post, args.no_pcd, n, timing,
                       t0, t3, f", {name}")
    return {**row, "status": "ok", "stats": stats, "plan": plan_meta, "recording": recording,
            "size_bytes": path.stat().st_size, "timing_s": timing}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lcm-url", required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--primitives", default=",".join(V1_PRIMITIVES),
                   help=f"comma list of {', '.join(PRIMITIVES)}")
    p.add_argument("--append", action="store_true",
                   help="add episodes to an existing --out (its index.json is extended)")
    p.add_argument("--amp-mm", type=float, default=20.0, help="max translation amplitude")
    p.add_argument("--amp-deg", type=float, default=5.0, help="max rotation amplitude")
    p.add_argument("--bound-frac", type=float, default=0.8,
                   help="planned |cmd_delta| <= this fraction of the export's u_lb/u_ub")
    p.add_argument("--motion-s", type=float, default=8.0,
                   help="motion duration; <= 0: per primitive, the shortest in [--motion-s-min, "
                        "--motion-s-max] at which the full amplitude keeps --bound-frac")
    p.add_argument("--motion-s-min", type=float, default=15.0)
    p.add_argument("--motion-s-max", type=float, default=25.0)
    p.add_argument("--retry-policy", choices=("halve", "lobe"), default="halve",
                   help="halve: halve the amplitude on any failure (v1); lobe: on a stretch "
                        "failure first halve the stretching lobe, else amplitude x0.7")
    p.add_argument("--pre-hold-s", type=float, default=col.PRE_HOLD_S)
    p.add_argument("--post-hold-s", type=float, default=0.5)
    p.add_argument("--stretch-cap-pct", type=float, default=2.0)
    p.add_argument("--crop-z-tol-mm", type=float, default=2.0,
                   help="belt top may exceed the start belt top by this much")
    p.add_argument("--retries", type=int, default=2, help="halve the amplitude on failure")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deploy", type=Path, default=DEPLOY)
    p.add_argument("--start-state", type=Path,
                   default=sim_snapshot.DEFAULT_START_STATE_DIR / "pre_place_1_osc.npz")
    p.add_argument("--min-clearance", type=float, default=col.clr.DEFAULT_MIN_CLEARANCE_MM)
    p.add_argument("--osc-settle-s", type=float, default=0.5)
    p.add_argument("--osc-timeout-s", type=float, default=5.0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--no-pcd", action="store_true")
    args = p.parse_args()
    col.configure_logging()
    check_private_url(args.lcm_url)
    names = [s.strip() for s in args.primitives.split(",") if s.strip()]
    bad = [s for s in names if s not in PRIMITIVES]
    if bad:
        p.error(f"unknown primitives {bad}")
    with np.load(args.deploy, allow_pickle=True) as d:
        lb, ub = d["u_lb"].astype(np.float64), d["u_ub"].astype(np.float64)
    out = args.out or OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    args.action_definition = "cmd_delta"
    n = col.sample_steps(lcs.SAMPLE_PERIOD_S)
    nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=col.FIRST, last=col.LAST)
    snap = sim_snapshot.load(args.start_state)
    index = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
             "git": git_info(), "kind": "motion_primitives", "backend": "osc",
             "start_state": {"path": str(args.start_state),
                             "sha256": sha256_file(args.start_state)},
             "deploy": str(args.deploy), "u_lb": lb.tolist(), "u_ub": ub.tolist(),
             "action_definition": "cmd_delta", "sample_period_s": n * lcs.SIM_DT_S,
             "sample_steps": n, "belt_sampling": lcs.BELT_SAMPLING,
             "crop_box": {"lo": CROP_LO.tolist(), "hi": CROP_HI.tolist()}, "episodes": []}
    index_path = out / "index.json"
    if args.append and index_path.exists():
        old = json.loads(index_path.read_text())
        index["episodes"] = old["episodes"]
        index["appended_runs"] = old.get("appended_runs", []) + [
            {"args": old["args"], "summary": old.get("summary")}]
    write_json(index_path, index)
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=args.osc_timeout_s)
    sim.args.record_state_every = col.RECORD_STATE_EVERY
    t_run = time.perf_counter()
    reason = "error"
    try:
        warm = sim.start_osc(out / ("osc_append.log" if args.append else "osc.log"))
        osc_info = {**sim.osc.describe(), "warm_up_s": warm}
        index["osc"] = osc_info
        sim.restore(snap)
        jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
        board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
        ctx = col.OscContext(args=args, out=out, n=n, thresholds=DEFAULT_THRESHOLDS,
                             tangent=pert.belt_tangent(nominal), snap=snap, nominal=nominal,
                             board=board, gripper=gripper, excite={"on": False}, scenario=None,
                             osc=osc_info, start_state=args.start_state,
                             params=CommanderParams(), x_tool0=x_tool0_tracking())
        gauge = RodGauge(sim)
        i = sum(r["status"] == "ok" for r in index["episodes"])
        for name in names:
            mul, lobe = 1.0, 1.0
            for attempt in range(args.retries + 1):
                row = run_primitive(sim, ctx, i, name, mul, gauge, lb, ub, args.seed, lobe)
                row["attempt"] = attempt
                index["episodes"].append(row)
                write_json(index_path, index)
                if row["status"] == "ok":
                    i += 1
                    break
                stretchy = name in ("opposite_sideways", "franka_only", "ur_only", "random_mix")
                if args.retry_policy == "halve":
                    mul *= 0.5
                elif "stretch" in row.get("reason", "") and stretchy and lobe > 0.2:
                    lobe *= 0.5
                else:
                    mul *= 0.7
        index["osc_log_errors"] = [e for e in col.OSC_LOG_ERRORS if e in sim.osc.log_text()]
        reason = "finished"
    finally:
        sim.close(reason)
    ok = [r for r in index["episodes"] if r["status"] == "ok"]
    index["summary"] = {"episodes_ok": len(ok), "failed_attempts": len(index["episodes"]) - len(ok),
                        "wall_s": time.perf_counter() - t_run}
    write_json(index_path, index)
    for r in ok:
        s = r["stats"]
        print(f"{r['file']} {r['primitive']:<18} x{r['amp_mul']:<5g} bound max "
              f"{s['u_bound_ratio_max']:.2f} crop z_top {s['crop_margin_min_mm']['z_top']:+.1f} mm"
              f" (start {s['crop_margin_min_mm']['z_top_start']:+.1f}) stretch gain "
              f"{s['rod_stretch_gain_pct_max']:+.3f} % clr {s['min_board_clearance_mm']:.1f} mm "
              f"wrap max {s['wrap_deg_max']:.0f}")
    return 0 if len({r["primitive"] for r in ok} & set(names)) == len(set(names)) else 1


if __name__ == "__main__":
    sys.exit(main())
