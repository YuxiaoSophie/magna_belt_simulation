#!/usr/bin/env python3
"""Demo-prefix replay: the robot replays the first part of a recorded demo before a controller
starts, so a rollout can begin where the demo left off (a start-state choice, not a controller).

Both arms track the demo's measured EE poses (``sim_ee_franka`` / ``sim_ee_ur``) RELATIVE to the
demo's pose at the replay's first frame, composed onto the arms' current poses (a scripted
reset's jitter is kept): orientation ``R_now0 R_demo0^T R_demo_k`` per arm, position deltas
rotated by the common start-pose yaw about the large-pulley axis (as ``jitter_pose7``). Demo
timing (one frame = ``lcs.SAMPLE_PERIOD_S``), piecewise linear / slerp between frames, then a
bounded hold at the end pose. Franka: 7-knot ``TARGET_CARTESIAN_POSE_TRAJECTORY`` per OSC tick
(``saved_traj_message``, dropped by ``ControllerBridge``); UR: IK of the path + the collector's
velocity lead. Aborts as ``scripted_reset``: grasp ``held()`` every sample, Franka tip floor,
2F-85 board clearance, UR IK miss. ``ok`` iff no abort, grasps held and the end error <= 3 mm /
1 deg per arm.

``frac`` in [0, 2]: 0..1 maps onto [motion start, latch], 1..2 onto [latch, first hold] (latch =
first frame after the last ``move:pre_place_2``, first hold = first ``hold:place_3``). The demo's
``prehold`` frames (the robot standing still) are skipped; frac 0 is a no-op.

Run (one sim session, a snapshot restore before each frac):
    uv run --frozen python scripts/lcs/demo_prefix.py \
        --episode data/lcs/realfirst/v1_1/seed/demo/episode_0000.npz --frac 0.5 1.0 1.5 \
        --start-state data/lcs/start_states/grasp_variants/set2/gv_00_osc.npz \
        --lcm-url 'udpm://239.255.76.191:7791?ttl=0' --out data/lcs/realfirst/cmp/smoke_prefix/cli
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_lcs_dataset as cli
import eval_learned_mpc as harness
import scripted_reset as sr
from latent_stage_tols import latch_frame
from loguru import logger

from round_belt_task.arm_kinematics import UrTracking, ik, rot_axis_angle
from round_belt_task.commander import (
    CommanderParams,
    mat3_to_quat,
    pose_mat,
    quat_to_mat3,
    saved_traj_message,
    slerp,
)
from round_belt_task.episode_io import git_info, pose7_mat, sha256_file, write_json
from round_belt_task.osc_bridge import OscTimeout
from round_belt_task.outcome import DEFAULT_THRESHOLDS
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot

FRAME_DT_S = lcs.SAMPLE_PERIOD_S
TOL_MM, TOL_DEG = 3.0, 1.0
SETTLE_TOL_MM, SETTLE_TOL_DEG = 1.0, 0.3  # hold until inside, so the controller starts at rest
SETTLE_MAX_S = 1.2
PHASE_LABEL = "prefix:demo"


@dataclasses.dataclass(frozen=True)
class PrefixSpec:
    episode: str
    frac: float


@dataclasses.dataclass
class PrefixReport:
    episode: str
    frac: float
    end_frame: int
    duration_s: float
    end_err_mm: list[float] | None
    end_err_deg: list[float] | None
    ok: bool
    reason: str | None
    start_frame: int = 0
    n_frames: int = 0  # samples recorded during the prefix (the caller's on_frame calls)
    steps: int = 0
    settle_s: float = 0.0
    grasp_held: list[bool] | None = None
    max_track_err_mm: list[float] | None = None
    tip_min_mm: float | None = None
    min_clearance_mm: float | None = None
    yaw_deg: float | None = None
    marks: dict | None = None
    node_offset_frames: int | None = None  # set by the harness when passed to the MPPI node

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --- demo --------------------------------------------------------------------------------------

def demo_marks(labels: list[str]) -> dict:
    """``move0`` (first ``move:`` frame), ``latch``, ``first_hold`` (first ``hold:place_3``)."""
    move0 = next((t for t, lab in enumerate(labels) if lab.startswith("move:")), None)
    if move0 is None or "hold:place_3" not in labels:
        raise ValueError("demo needs move:* and hold:place_3 phases")
    return {"move0": int(move0), "latch": int(latch_frame(labels)),
            "first_hold": int(labels.index("hold:place_3"))}


def load_demo(episode) -> dict:
    path = harness._resolve(REPO_ROOT, str(episode))
    with np.load(path, allow_pickle=True) as d:
        meta = json.loads(str(d["sim_meta"]))
        labels = [meta["phase_labels"][p] for p in np.asarray(d["sim_phase"], np.int64)]
        ee_f = np.asarray(d["sim_ee_franka"], np.float64)
        ee_u = np.asarray(d["sim_ee_ur"], np.float64)
        dt = float(np.median(np.diff(np.asarray(d["sim_time"], np.float64))))
    if abs(dt - FRAME_DT_S) > 1e-6:
        raise ValueError(f"{path}: frame period {dt} != {FRAME_DT_S}")
    return {"path": str(path), "ee_f": ee_f, "ee_u": ee_u, "labels": labels,
            **demo_marks(labels)}


def frac_frame(marks: dict, frac: float) -> int:
    """Demo frame of ``frac`` (0..1 -> [move0, latch], 1..2 -> [latch, first_hold]), half up."""
    frac = float(frac)
    if not 0.0 <= frac <= 2.0:
        raise ValueError(f"prefix frac {frac} outside [0, 2]")
    a, b = ((marks["move0"], marks["latch"]) if frac <= 1.0
            else (marks["latch"], marks["first_hold"]))
    x = a + (frac if frac <= 1.0 else frac - 1.0) * (b - a)
    return math.floor(x + 0.5)


def prefix_end_frame(episode, frac: float) -> int:
    """End frame of a ``frac`` prefix of ``episode`` (0 = no prefix)."""
    if frac <= 0:
        return 0
    marks = demo_marks(load_demo(episode)["labels"])
    end = frac_frame(marks, frac)
    return end if end > marks["move0"] else 0


class KnotPath:
    """Pose path through uniformly timed knots (``dt``), linear xyz / slerp, clamped."""

    def __init__(self, poses7: np.ndarray, dt: float) -> None:
        self.p = np.asarray(poses7, np.float64)
        self.dt = float(dt)
        self.end = (len(self.p) - 1) * self.dt

    def __call__(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        x = min(max(float(tau), 0.0), self.end) / self.dt
        i = min(int(x), len(self.p) - 2) if len(self.p) > 1 else 0
        s = x - i
        if len(self.p) == 1:
            return self.p[0, :3].copy(), self.p[0, 3:].copy()
        a, b = self.p[i], self.p[i + 1]
        q = slerp(a[3:], b[3:], s)
        return a[:3] + s * (b[:3] - a[:3]), q / np.linalg.norm(q)

    def pose7(self, tau: float) -> np.ndarray:
        return np.concatenate(self(tau))


def relative_targets(demo: dict, start: int, end: int, home_f7, home_u7, axis
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """Demo frames ``start..end`` re-anchored on the homes; returns ``(F (n, 7), UR (n, 7),
    common yaw deg)``."""
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    homes = [np.asarray(home_f7, np.float64), np.asarray(home_u7, np.float64)]
    tracks = [demo["ee_f"][start:end + 1], demo["ee_u"][start:end + 1]]
    yaw = float(np.mean([sr.yaw_about_deg(tr[0, 3:], h[3:], axis)
                         for tr, h in zip(tracks, homes)]))
    R_yaw = rot_axis_angle(axis, math.radians(yaw))
    out = []
    for tr, h in zip(tracks, homes):
        R_rel = quat_to_mat3(h[3:]) @ quat_to_mat3(tr[0, 3:]).T
        pos = h[:3] + (tr[:, :3] - tr[0, :3]) @ R_yaw.T
        quat = np.stack([mat3_to_quat(R_rel @ quat_to_mat3(q)) for q in tr[:, 3:]])
        out.append(np.hstack([pos, quat]))
    return out[0], out[1], yaw


def pose_err(meas7, target7) -> tuple[float, float]:
    """(mm, deg) between two xyz_wxyz poses."""
    m, t = np.asarray(meas7, np.float64), np.asarray(target7, np.float64)
    c = abs(float(m[3:] @ t[3:])) / (np.linalg.norm(m[3:]) * np.linalg.norm(t[3:]))
    ang = math.degrees(2 * math.acos(min(1.0, c)))
    return float(np.linalg.norm(m[:3] - t[:3]) * 1e3), ang


# --- replay ------------------------------------------------------------------------------------

def replay_prefix(sim, ctx, spec: PrefixSpec, home_f7, home_u7,
                  on_frame: Callable | None = None) -> PrefixReport:
    """Replay ``spec`` from the current state anchored on ``home_f7`` / ``home_u7`` (the arms'
    current measured poses). ``on_frame(k, cmd)`` at every sample tick before the step (``k`` =
    prefix step, ``cmd`` = the frame's command: ``knots``, ``knot_times``, ``knot0``, ``knot1``,
    ``ur_t``, ``ur_t1``, ``traj_utime``); the state after the last step is NOT sampled (the
    caller's next frame). ``ctx`` needs ``board``, ``gripper``, ``plate_top_z``. Leaves the
    Franka on a fresh hold hook and the UR at its last target.
    """
    demo = load_demo(spec.episode)
    marks = {k: demo[k] for k in ("move0", "latch", "first_hold")}
    start = marks["move0"]
    end = frac_frame(marks, spec.frac) if spec.frac > 0 else start
    if end <= start:
        return PrefixReport(episode=demo["path"], frac=float(spec.frac), end_frame=0,
                            duration_s=0.0, end_err_mm=[0.0, 0.0], end_err_deg=[0.0, 0.0],
                            ok=True, reason=None, start_frame=start, marks=marks)
    dt, n = sim.frame_dt, harness.SAMPLE_STEPS
    pulley = cli.pulley_frame(sim)
    tf, tu, yaw = relative_targets(demo, start, end, home_f7, home_u7, pulley["axis"])
    path_f, path_u = KnotPath(tf, FRAME_DT_S), KnotPath(tu, FRAME_DT_S)
    motion_steps = round(path_f.end / dt)
    max_steps = motion_steps + n * round(SETTLE_MAX_S / (n * dt))
    knot_t = CommanderParams().dt * np.arange(harness.N_KNOTS)
    u_coords = sim.arm_coords()[1]
    t0 = sim.osc_time_s(sim.step_index)

    def hook(step, t, joint_q, body_q):
        knots = [path_f(t - t0 + o) for o in knot_t]
        return saved_traj_message(round(t * 1e6), np.stack([k[0] for k in knots]),
                                  np.stack([k[1] for k in knots]), t + knot_t)

    plate_top = ctx.plate_top_z
    q_ur = sim.arm_targets()[1]
    lead_gain = sim.arm_kd / sim.arm_ke / dt
    tip_min = clear_min = math.inf
    last_clear = None
    board_neg = 0
    abort = None
    track_max = [0.0, 0.0]
    n_frames = 0
    k = 0
    w0 = time.perf_counter()
    sim.commander_hook = hook
    try:
        while True:
            if k % n == 0:
                t = sim.osc_time_s(sim.step_index)
                tau = t - t0
                joint_q = sim.state_0.joint_q.numpy()
                ee_f = sim.franka_measured_pose7(joint_q)
                ee_u = pose7_mat(UrTracking.fk(joint_q[u_coords]))
                ef = pose_err(ee_f, path_f.pose7(tau))
                eu = pose_err(ee_u, path_u.pose7(tau))
                track_max = [max(track_max[0], ef[0]), max(track_max[1], eu[0])]
                inside = max(ef[0], eu[0]) <= SETTLE_TOL_MM and max(ef[1], eu[1]) <= SETTLE_TOL_DEG
                if k >= motion_steps and (inside or k >= max_steps):
                    break
                grasp = sim.grasp_state()
                if not all(grasp.held()):
                    abort = f"grasp lost at prefix step {k} ({grasp.describe()})"
                    break
                if on_frame is not None:
                    knots = [path_f(tau + o) for o in knot_t]
                    kn = np.hstack([np.stack([x[0] for x in knots]),
                                    np.stack([x[1] for x in knots])])
                    on_frame(k, {"knots": kn, "knot_times": t + knot_t, "knot0": kn[0].copy(),
                                 "knot1": kn[1].copy(), "ur_t": path_u.pose7(tau),
                                 "ur_t1": path_u.pose7(tau + lcs.ACTION_KNOT_DT_S),
                                 "traj_utime": round(t * 1e6)})
                    n_frames += 1
            p, q = path_u(sim.osc_time_s(sim.step_index + 1) - t0)
            q_new, err_p, err_r, iters = ik(UrTracking, pose_mat(p, q), q_ur,
                                            pos_tol=cli.UR_IK_POS_TOL, rot_tol=cli.UR_IK_ROT_TOL)
            if err_p > cli.UR_IK_POS_TOL or err_r > cli.UR_IK_ROT_TOL:
                abort = (f"UR IK missed at prefix step {k} ({err_p * 1e3:.3f} mm, "
                         f"{np.degrees(err_r):.3f} deg after {iters} iterations)")
                break
            lead = cli.UR_VELOCITY_LEAD * lead_gain if k > 0 else 0.0
            sim.set_ur_target(q_new + lead * (q_new - q_ur))
            q_ur = q_new
            sim.control_step()
            k += 1
            joint_q = sim.state_0.joint_q.numpy()
            tip = float(sim.franka_measured_pose7(joint_q)[2] - plate_top)
            tip_min = min(tip_min, tip)
            if k % cli.CLEARANCE_EVERY == 0:
                last_clear = float(ctx.gripper.measure(
                    ctx.board, sim.state_0.body_q.numpy().astype(np.float64)))
                clear_min = min(clear_min, last_clear)
            board_neg = board_neg + 1 if last_clear is not None and last_clear < 0.0 else 0
            if tip * 1e3 < harness.TIP_FLOOR_MM:
                abort = f"franka finger_tip {tip * 1e3:.2f} mm above the plate at prefix step {k}"
            elif board_neg > harness.BOARD_NEGATIVE_STEPS:
                abort = f"2F-85 board clearance < 0 for {board_neg} steps at prefix step {k}"
            if abort is not None:
                break
    finally:
        sim.commander_hook = sim.make_hold_hook()
        if abort is not None:
            sim.set_ur_target(sim.state_0.joint_q.numpy()[u_coords])
    joint_q = sim.state_0.joint_q.numpy()
    ef = pose_err(sim.franka_measured_pose7(joint_q), tf[-1])
    eu = pose_err(pose7_mat(UrTracking.fk(joint_q[u_coords])), tu[-1])
    held = [bool(v) for v in sim.grasp_state().held()]
    err_mm, err_deg = [ef[0], eu[0]], [ef[1], eu[1]]
    reason = abort
    if reason is None and not all(held):
        reason = f"grasp not held at the prefix end ({sim.grasp_state().describe()})"
    if reason is None and (max(err_mm) > TOL_MM or max(err_deg) > TOL_DEG):
        reason = (f"end error F {err_mm[0]:.2f} mm / {err_deg[0]:.2f} deg, UR {err_mm[1]:.2f} mm "
                  f"/ {err_deg[1]:.2f} deg > {TOL_MM:g} mm / {TOL_DEG:g} deg")
    rep = PrefixReport(
        episode=demo["path"], frac=float(spec.frac), end_frame=int(end), duration_s=k * dt,
        end_err_mm=err_mm, end_err_deg=err_deg, ok=reason is None, reason=reason,
        start_frame=int(start), n_frames=n_frames, steps=k,
        settle_s=max(0, k - motion_steps) * dt, grasp_held=held, max_track_err_mm=track_max,
        tip_min_mm=tip_min * 1e3, min_clearance_mm=clear_min * 1e3, yaw_deg=yaw, marks=marks)
    if abort is not None:
        logger.warning(f"[PREFIX] abort: {abort}")
    logger.info(f"[PREFIX] {'ok' if rep.ok else 'FAILED'} frac {spec.frac:g} frames {start}..{end}"
                f" in {rep.duration_s:.2f} s sim (settle {rep.settle_s:.2f} s, "
                f"{time.perf_counter() - w0:.1f} s wall), end err F {err_mm[0]:.2f} mm "
                f"{err_deg[0]:.2f} deg / UR {err_mm[1]:.2f} mm {err_deg[1]:.2f} deg, yaw "
                f"{yaw:+.2f} deg" + (f": {reason}" if reason else ""))
    return rep


# --- CLI ---------------------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episode", type=Path, required=True, help="demo episode npz")
    p.add_argument("--frac", type=float, nargs="+", required=True, help="one replay per value")
    p.add_argument("--start-state", type=Path, required=True, help="snapshot npz restored first")
    p.add_argument("--lcm-url", required=True)
    p.add_argument("--out", type=Path, required=True, help="new dir (must not hold report.json)")
    p.add_argument("--osc-binary", type=Path, default=None)
    p.add_argument("--osc-timeout-s", type=float, default=5.0)
    return p


def run_cli(args: argparse.Namespace) -> dict:
    harness.check_private_url(args.lcm_url)
    out = Path(args.out)
    if (out / "report.json").exists():
        raise FileExistsError(f"{out / 'report.json'} exists: pick a new --out")
    out.mkdir(parents=True, exist_ok=True)
    report = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "git": git_info(), "episode_sha256": sha256_file(args.episode),
              "start_state_sha256": sha256_file(args.start_state),
              "tol": {"mm": TOL_MM, "deg": TOL_DEG}, "rows": []}
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    osc_binary, osc_root = harness.default_osc_binary()
    if args.osc_binary is not None:
        osc_binary = Path(args.osc_binary)
        osc_root = (harness.MAGNA_WORKTREE if harness.MAGNA_WORKTREE in osc_binary.parents
                    else harness.MAGNA_ROOT)
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=args.osc_timeout_s)
    snap = sim_snapshot.load(args.start_state)
    t_run = time.perf_counter()
    try:
        warm = sim.start_osc(out / "osc.log", binary=osc_binary, cwd=osc_root)
        report["osc"] = {**sim.osc.describe(), "warm_up_s": warm}
        sim.restore(snap)
        nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=cli.FIRST, last=cli.LAST)
        jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
        board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
        ctx = argparse.Namespace(board=board, gripper=gripper, thresholds=DEFAULT_THRESHOLDS)
        plate = board.plate
        ctx.plate_top_z = float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half)
        for frac in args.frac:
            grasp = sim.restore(snap)
            if grasp.held() != (True, True):
                report["rows"].append({"frac": frac, "ok": False,
                                       "reason": f"restore: {grasp.describe()}"})
                continue
            joint_q = sim.state_0.joint_q.numpy()
            home_f = sim.franka_measured_pose7(joint_q)
            home_u = pose7_mat(UrTracking.fk(joint_q[sim.arm_coords()[1]]))
            frames: list[int] = []
            try:
                rep = replay_prefix(sim, ctx, PrefixSpec(str(args.episode), frac), home_f, home_u,
                                    on_frame=lambda k, cmd, acc=frames: acc.append(k))
            except OscTimeout as exc:
                if not exc.process_alive:
                    raise
                sim.bridge.forget_pending()
                rep = PrefixReport(episode=str(args.episode), frac=frac, end_frame=-1,
                                   duration_s=0.0, end_err_mm=None, end_err_deg=None, ok=False,
                                   reason=f"osc timeout: {exc}")
            report["rows"].append({**rep.to_dict(), "home_franka": home_f.tolist(),
                                   "home_ur": home_u.tolist(), "sample_steps": frames})
            write_json(out / "report.json", report)
    finally:
        sim.close("finished")
    rows = [r for r in report["rows"] if "end_err_mm" in r and r["end_err_mm"] is not None]
    report["summary"] = {
        "n": len(report["rows"]), "ok": sum(bool(r.get("ok")) for r in report["rows"]),
        "max_end_err_mm": max((max(r["end_err_mm"]) for r in rows), default=None),
        "max_end_err_deg": max((max(r["end_err_deg"]) for r in rows), default=None)}
    report["wall_s"] = time.perf_counter() - t_run
    write_json(out / "report.json", report)
    logger.info(f"[PREFIX] {report['summary']} -> {out}")
    return report


def main() -> int:
    args = create_parser().parse_args()
    cli.configure_logging()
    run_cli(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
