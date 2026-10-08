#!/usr/bin/env python3
"""Scripted robot reset between closed-loop episodes (grippers stay closed on the belt).

Phase 0 (optional: ``pull_mm`` toward the Franka side, ``slack_mm`` toward the pulley axis) moves
both end effectors horizontally, phase 1 lifts both by ``lift_mm`` along world z, phase 2 moves
them to the episode-start poses
moved by one seeded rigid jitter (xy shift + yaw about the large-pulley axis, the same for both
arms so the belt stays taut), phase 3 holds; all min-jerk. The Franka follows 7-knot
``TARGET_CARTESIAN_POSE_TRAJECTORY`` messages on the path (``commander.saved_traj_message``, as
the collector), the UR the path by IK with the collector's velocity lead. The harness's aborts
stay active (Franka tip floor, 2F-85 board clearance, grasp loss, UR IK miss). The grasp check is
``GraspState.held()`` (width/status + tip gap). A failed reset falls back to a snapshot restore,
counted as a human reset. No reset frame is recorded.

The CLI resets from recorded end states (``final_*`` contact snapshots) toward each source
episode's frame-0 poses and writes ``report.json``.

Run:
    uv run --frozen python scripts/lcs/scripted_reset.py --snapshots data/lcs/contact/snapshots \
        --select working_set.json --n 10 --lcm-url 'udpm://239.255.76.161:7761?ttl=0' \
        --out data/lcs/realfirst/smoke/reset/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_lcs_dataset as cli
import eval_learned_mpc as harness
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
from round_belt_task.outcome import DEFAULT_THRESHOLDS, classify, frame_metrics
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot

RESET_MODES = ("snapshot", "scripted")
FREED_H_MM = 20.0
JITTER_TOL_MM, JITTER_TOL_DEG = 3.0, 1.0
FINAL_CLASSES = ("final_engaged", "final_over", "final_under", "final_slanted")
DEFAULT_SNAPSHOTS = REPO_ROOT / "data" / "lcs" / "contact" / "snapshots"
DEFAULT_LCM_URL = "udpm://239.255.76.161:7761?ttl=0"


@dataclasses.dataclass(frozen=True)
class ResetSpec:
    lift_mm: float = 40.0
    lift_s: float = 2.0
    return_s: float = 3.0
    settle_s: float = 1.0
    jitter_xy_mm: float = 8.0
    jitter_yaw_deg: float = 5.0
    pull_mm: float = 0.0  # phase 0: horizontal, from the UR EE toward the Franka EE
    slack_mm: float = 0.0  # phase 0: horizontal, from the EE midpoint toward the pulley axis
    pull_s: float = 1.0
    seed: int = 0


def parse_reset_spec(pairs) -> ResetSpec:
    """``KEY=VALUE`` overrides of :class:`ResetSpec`."""
    names = {f.name for f in dataclasses.fields(ResetSpec)}
    kw: dict = {}
    for pair in pairs or ():
        key, sep, value = str(pair).partition("=")
        if not sep or key not in names:
            raise ValueError(f"reset spec {pair!r}: want KEY=VALUE, KEY in {sorted(names)}")
        kw[key] = int(value) if key == "seed" else float(value)
    return ResetSpec(**kw)


@dataclasses.dataclass
class ResetReport:
    """``ok``: both grasps held after the reset (None: checked by the caller's settle)."""

    ok: bool | None
    kind: str
    reason: str | None
    grasp_held: list[bool] | None
    duration_s: float
    jitter: dict | None = None
    belt_freed_sim: bool | None = None
    tip_min_mm: float | None = None
    min_clearance_mm: float | None = None
    h_after_lift_mm: float | None = None
    label_after_sim: str | None = None
    realised: dict | None = None
    attempt: dict | None = None  # the failed scripted attempt behind a human reset

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def human(cls, reason: str, attempt: ResetReport | None = None) -> ResetReport:
        """A snapshot restore done by the caller (the harness's own restore + settle)."""
        return cls(ok=None, kind="human", reason=reason, grasp_held=None,
                   duration_s=0.0 if attempt is None else attempt.duration_s,
                   attempt=None if attempt is None else attempt.to_dict())


# --- geometry ---------------------------------------------------------------------------------

def draw_jitter(spec: ResetSpec, rng: np.random.Generator) -> dict:
    dx, dy = rng.uniform(-1.0, 1.0, 2) * spec.jitter_xy_mm
    yaw = rng.uniform(-1.0, 1.0) * spec.jitter_yaw_deg
    return {"dx": float(dx), "dy": float(dy), "yaw_deg": float(yaw)}


def jitter_pose7(pose7, jitter: dict, pulley: dict) -> np.ndarray:
    """``pose7`` (xyz_wxyz) rotated by ``yaw_deg`` about the pulley axis, then shifted in xy."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    R = rot_axis_angle(pulley["axis"], math.radians(jitter["yaw_deg"]))
    c = np.asarray(pulley["centre"], dtype=np.float64)
    pos = c + R @ (pose7[:3] - c) + np.array([jitter["dx"], jitter["dy"], 0.0]) * 1e-3
    return np.concatenate([pos, mat3_to_quat(R @ quat_to_mat3(pose7[3:]))])


def yaw_about_deg(quat_from, quat_to, axis) -> float:
    """Rotation ``R_to R_from^T`` projected on ``axis`` [deg]."""
    rv = lcs.delta_rotvec(np.asarray(quat_from, float), np.asarray(quat_to, float))
    return math.degrees(float(rv @ (np.asarray(axis, float) / np.linalg.norm(axis))))


class MinJerkPath:
    """Piecewise min-jerk pose path (xyz + wxyz slerp) from ``pose7`` at ``tau = 0``."""

    def __init__(self, pose7) -> None:
        pose7 = np.asarray(pose7, dtype=np.float64)
        self.t = [0.0]
        self.pos = [pose7[:3].copy()]
        self.quat = [pose7[3:] / np.linalg.norm(pose7[3:])]

    def add(self, duration: float, pos, quat=None) -> None:
        if duration <= 0.0:
            return
        q = self.quat[-1] if quat is None else np.asarray(quat, np.float64)
        self.t.append(self.t[-1] + float(duration))
        self.pos.append(np.asarray(pos, dtype=np.float64).copy())
        self.quat.append(q / np.linalg.norm(q))

    @property
    def end(self) -> float:
        return self.t[-1]

    def __call__(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        tau = min(max(float(tau), 0.0), self.t[-1])
        i = max(1, min(int(np.searchsorted(self.t, tau, side="right")), len(self.t) - 1))
        if len(self.t) == 1:
            return self.pos[0].copy(), self.quat[0].copy()
        s = (tau - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        m = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
        q = slerp(self.quat[i - 1], self.quat[i], m)
        return self.pos[i - 1] + m * (self.pos[i] - self.pos[i - 1]), q / np.linalg.norm(q)


# --- reset ------------------------------------------------------------------------------------

def _horizontal_unit(v) -> np.ndarray:
    v = np.array([v[0], v[1], 0.0], dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _belt_metrics(sim, th):
    body_q = sim.state_0.body_q.numpy()
    belt = body_q[sim.info.belt_bodies, :3].astype(np.float64)
    return frame_metrics(belt, body_q[int(sim.info.pulley_bodies[1])].astype(np.float64), th=th)


def scripted_reset(sim, ctx, home_f7, home_u7, spec: ResetSpec, rng: np.random.Generator,
                   on_frame: Callable | None = None) -> ResetReport:
    """Lift, return to the jittered home poses, settle, check the grasp (see the module doc).

    ``ctx`` needs ``board``, ``gripper``, ``plate_top_z`` (``EvalContext`` has them) and may
    carry ``thresholds``; ``on_frame(k, phase, sim)`` every sample period. Leaves the Franka
    on a fresh hold hook and the UR at its last target (its measured joints after an abort).
    """
    th = getattr(ctx, "thresholds", None) or DEFAULT_THRESHOLDS
    dt = sim.frame_dt
    knot_t = CommanderParams().dt * np.arange(harness.N_KNOTS)
    jitter = draw_jitter(spec, rng)
    pulley = cli.pulley_frame(sim)
    u_coords = sim.arm_coords()[1]
    joint_q = sim.state_0.joint_q.numpy()
    f0 = sim.franka_measured_pose7(joint_q)
    u0 = pose7_mat(UrTracking.fk(joint_q[u_coords]))
    goal_f, goal_u = jitter_pose7(home_f7, jitter, pulley), jitter_pose7(home_u7, jitter, pulley)
    pull = _horizontal_unit(f0[:3] - u0[:3]) * spec.pull_mm * 1e-3
    pull += _horizontal_unit(pulley["centre"] - 0.5 * (f0[:3] + u0[:3])) * spec.slack_mm * 1e-3
    up = np.array([0.0, 0.0, spec.lift_mm * 1e-3])
    paths = []
    for start, goal in ((f0, goal_f), (u0, goal_u)):
        path = MinJerkPath(start)
        path.add(spec.pull_s if np.any(pull) else 0.0, start[:3] + pull)
        path.add(spec.lift_s, start[:3] + pull + up)
        path.add(spec.return_s, goal[:3], goal[3:])
        paths.append(path)
    path_f, path_u = paths
    t_lift = path_f.end - spec.return_s
    total_s = path_f.end + spec.settle_s
    n_steps = round(total_s / dt)
    lift_step = round(t_lift / dt)

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
    fm_lift = None
    k = 0
    w0 = time.perf_counter()
    sim.commander_hook = hook
    try:
        while k < n_steps:
            p, q = path_u(sim.osc_time_s(sim.step_index + 1) - t0)
            q_new, err_p, err_r, iters = ik(UrTracking, pose_mat(p, q), q_ur,
                                            pos_tol=cli.UR_IK_POS_TOL, rot_tol=cli.UR_IK_ROT_TOL)
            if err_p > cli.UR_IK_POS_TOL or err_r > cli.UR_IK_ROT_TOL:
                abort = (f"UR IK missed at step {k} ({err_p * 1e3:.3f} mm, "
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
            if k == lift_step:
                fm_lift = _belt_metrics(sim, th)
            if tip * 1e3 < harness.TIP_FLOOR_MM:
                abort = f"franka finger_tip {tip * 1e3:.2f} mm above the plate at step {k}"
            elif board_neg > harness.BOARD_NEGATIVE_STEPS:
                abort = f"2F-85 board clearance < 0 for {board_neg} steps at step {k}"
            elif k % harness.SAMPLE_STEPS == 0:
                grasp = sim.grasp_state()
                if not all(grasp.held()):
                    abort = f"grasp lost at step {k} ({grasp.describe()})"
                if on_frame is not None:
                    tau = k * dt
                    phase = ("lift" if tau <= t_lift else "return" if tau <= path_f.end
                             else "settle")
                    on_frame(k, phase, sim)
            if abort is not None:
                break
    finally:
        sim.commander_hook = sim.make_hold_hook()
        if abort is not None:
            sim.set_ur_target(sim.state_0.joint_q.numpy()[u_coords])
    if abort is not None:
        logger.warning(f"[RESET] abort: {abort}")

    joint_q = sim.state_0.joint_q.numpy()
    ee_f = sim.franka_measured_pose7(joint_q)
    ee_u = pose7_mat(UrTracking.fk(joint_q[u_coords]))
    grasp = sim.grasp_state()
    held = [bool(v) for v in grasp.held()]
    ok = abort is None and all(held)
    reason = abort if abort is not None else (
        None if ok else f"grasp not held after settle ({grasp.describe()})")
    realised = {}
    for arm, ee, home, goal in (("franka", ee_f, home_f7, goal_f), ("ur", ee_u, home_u7, goal_u)):
        home = np.asarray(home, np.float64)
        realised[arm] = {
            "offset_mm": ((ee[:3] - home[:3]) * 1e3).tolist(),
            "drawn_offset_mm": ((goal[:3] - home[:3]) * 1e3).tolist(),
            "err_mm": float(np.linalg.norm(ee[:3] - goal[:3]) * 1e3),
            "yaw_deg": yaw_about_deg(home[3:], ee[3:], pulley["axis"]),
            "yaw_err_deg": yaw_about_deg(goal[3:], ee[3:], pulley["axis"]),
        }
    fm_end = _belt_metrics(sim, th)
    h_lift = None if fm_lift is None else float(fm_lift.h_median_mm)
    report = ResetReport(
        ok=ok, kind="scripted", reason=reason, grasp_held=held, duration_s=k * dt,
        jitter=jitter,
        belt_freed_sim=None if fm_lift is None else not (fm_lift.h_median_mm <= FREED_H_MM),
        tip_min_mm=tip_min * 1e3, min_clearance_mm=clear_min * 1e3, h_after_lift_mm=h_lift,
        label_after_sim=classify(fm_end, th), realised=realised)
    logger.info(f"[RESET] {'ok' if ok else 'FAILED'} in {k * dt:.2f} s sim "
                f"({time.perf_counter() - w0:.1f} s wall), jitter {jitter['dx']:+.1f}/"
                f"{jitter['dy']:+.1f} mm {jitter['yaw_deg']:+.2f} deg, h after lift "
                f"{h_lift if h_lift is None else round(h_lift, 1)} mm, err franka "
                f"{realised['franka']['err_mm']:.2f} / ur {realised['ur']['err_mm']:.2f} mm, "
                f"tip min {tip_min * 1e3:.1f} mm, clearance min {clear_min * 1e3:.1f} mm"
                + (f": {reason}" if reason else ""))
    return report


def fallback_restore(sim, snap) -> ResetReport:
    """Snapshot restore + the OSC settle: the operator's reset."""
    grasp = sim.restore(snap)
    held = [bool(v) for v in grasp.held()]
    return ResetReport(ok=all(held), kind="human", reason="snapshot restore", grasp_held=held,
                       duration_s=0.0)


# --- CLI --------------------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS,
                   help="contact snapshot dir with index.json")
    p.add_argument("--select", default=None,
                   help="working-set json (relative to --snapshots) whose final_* ids to use")
    p.add_argument("--variants", default=None, help="comma list of snapshot ids (overrides)")
    p.add_argument("--n", type=int, default=10, help="resets, round-robin over the labels")
    p.add_argument("--lcm-url", default=DEFAULT_LCM_URL)
    p.add_argument("--out", type=Path, required=True, help="new dir (must not hold report.json)")
    p.add_argument("--reset-spec", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--osc-binary", type=Path, default=None)
    p.add_argument("--osc-timeout-s", type=float, default=5.0)
    return p


def select_snapshots(snap_dir: Path, select: str | None, variants: str | None,
                     n: int) -> list[dict]:
    """Index rows of ``final_*`` snapshots, round-robin over their classes."""
    rows = json.loads((snap_dir / "index.json").read_text())["variants"]
    by_id = {r["id"]: r for r in rows}
    if variants:
        ids = [v.strip() for v in variants.split(",") if v.strip()]
        missing = [v for v in ids if v not in by_id]
        if missing:
            raise ValueError(f"{missing} not in {snap_dir / 'index.json'}")
        return [by_id[v] for v in ids][:n]
    if select:
        plan = json.loads((snap_dir / select).read_text())["plan"]
        ids = list(dict.fromkeys(p["id"] for p in plan if p["class"] in FINAL_CLASSES))
    else:
        ids = [r["id"] for r in rows if r["class"] in FINAL_CLASSES]
    queues = {c: [by_id[i] for i in ids if by_id[i]["class"] == c and by_id[i].get("held")]
              for c in FINAL_CLASSES}
    out: list[dict] = []
    while len(out) < n and any(queues.values()):
        for c in FINAL_CLASSES:
            if queues[c] and len(out) < n:
                out.append(queues[c].pop(0))
    return out


def source_home(row: dict) -> tuple[np.ndarray, np.ndarray, Path]:
    """Frame-0 measured EE poses of the snapshot's source episode and its start state file."""
    path = harness._resolve(REPO_ROOT, row["source_episode"]["file"])
    with np.load(path, allow_pickle=True) as d:
        meta = json.loads(str(d["sim_meta"]))
        return (d["sim_ee_franka"][0].astype(np.float64), d["sim_ee_ur"][0].astype(np.float64),
                harness._resolve(REPO_ROOT, meta["start_state"]))


def _summary(rows: list[dict]) -> dict:
    done = [r for r in rows if r["kind"] is not None]
    per: dict = {}
    for label in dict.fromkeys(r["label"] for r in done):
        rs = [r for r in done if r["label"] == label]
        freed = [r["belt_freed_sim"] for r in rs if r["belt_freed_sim"] is not None]
        per[label] = {"n": len(rs), "ok": sum(bool(r["ok"]) for r in rs),
                      "ok_rate": sum(bool(r["ok"]) for r in rs) / len(rs),
                      "freed": sum(freed), "freed_rate": sum(freed) / len(freed) if freed else None,
                      "labels_after": dict(Counter(r["label_after_sim"] for r in rs))}
    ok = [r for r in done if r["ok"]]
    err_mm = [r["realised"][a]["err_mm"] for r in ok for a in ("franka", "ur")]
    err_deg = [abs(r["realised"][a]["yaw_err_deg"]) for r in ok for a in ("franka", "ur")]
    tips = [r["tip_min_mm"] for r in done if r["tip_min_mm"] is not None]
    clears = [r["min_clearance_mm"] for r in done if r["min_clearance_mm"] is not None]
    return {
        "n": len(done), "ok": len(ok), "ok_rate": len(ok) / len(done) if done else None,
        "per_label": per,
        "jitter_accuracy": {"n": len(ok), "max_err_mm": max(err_mm) if err_mm else None,
                            "max_yaw_err_deg": max(err_deg) if err_deg else None,
                            "tol_mm": JITTER_TOL_MM, "tol_deg": JITTER_TOL_DEG,
                            "pass": bool(err_mm) and max(err_mm) <= JITTER_TOL_MM
                            and max(err_deg) <= JITTER_TOL_DEG},
        "safety": {"tip_min_mm": min(tips) if tips else None,
                   "tip_floor_mm": harness.TIP_FLOOR_MM,
                   "min_board_clearance_mm": min(clears) if clears else None,
                   "violations": sum(t < harness.TIP_FLOOR_MM for t in tips)
                   + sum(c < 0.0 for c in clears)},
        "reset_s": float(sum(r["duration_s"] for r in done)),
    }


def run_cli(args: argparse.Namespace) -> dict:
    harness.check_private_url(args.lcm_url)
    spec = parse_reset_spec(args.reset_spec)
    out = Path(args.out)
    if (out / "report.json").exists():
        raise FileExistsError(f"{out / 'report.json'} exists: pick a new --out")
    picks = select_snapshots(Path(args.snapshots), args.select, args.variants, args.n)
    if not picks:
        raise ValueError("no snapshots selected")
    out.mkdir(parents=True, exist_ok=True)
    report = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "git": git_info(), "spec": dataclasses.asdict(spec),
              "freed_rule": f"GT h_median > {FREED_H_MM:g} mm after the lift (sim only)",
              "rows": []}
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    osc_binary, osc_root = harness.default_osc_binary()
    if args.osc_binary is not None:
        osc_binary = Path(args.osc_binary)
        osc_root = (harness.MAGNA_WORKTREE if harness.MAGNA_WORKTREE in osc_binary.parents
                    else harness.MAGNA_ROOT)
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=args.osc_timeout_s)
    rng = np.random.default_rng(spec.seed)
    t_run = time.perf_counter()
    try:
        warm = sim.start_osc(out / "osc.log", binary=osc_binary, cwd=osc_root)
        report["osc"] = {**sim.osc.describe(), "warm_up_s": warm}
        sim.restore(sim_snapshot.load(harness.DEFAULT_START_STATE))
        nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=cli.FIRST, last=cli.LAST)
        jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
        board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
        ctx = argparse.Namespace(board=board, gripper=gripper, thresholds=DEFAULT_THRESHOLDS)
        plate = board.plate
        ctx.plate_top_z = float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half)
        for i, pick in enumerate(picks):
            path = harness._resolve(Path(args.snapshots), pick["file"])
            home_f, home_u, start_file = source_home(pick)
            row = {"i": i, "id": pick["id"], "label": pick["label"], "class": pick["class"],
                   "file": str(path), "sha256": sha256_file(path),
                   "source_episode": pick["source_episode"]["file"],
                   "home_franka": home_f.tolist(), "home_ur": home_u.tolist()}
            grasp = sim.restore(sim_snapshot.load(path))
            if grasp.held() != (True, True):
                logger.warning(f"[RESET] {pick['id']}: grasp not held after restore; skipped")
                report["rows"].append({**row, "kind": None, "status": "skipped",
                                       "reason": f"restore: {grasp.describe()}"})
                continue
            fm0 = _belt_metrics(sim, DEFAULT_THRESHOLDS)
            row["label_before_sim"] = classify(fm0, DEFAULT_THRESHOLDS)
            try:
                rep = scripted_reset(sim, ctx, home_f, home_u, spec, rng)
            except OscTimeout as exc:
                if not exc.process_alive:
                    raise
                sim.bridge.forget_pending()
                rep = ResetReport(ok=False, kind="scripted", reason=f"osc timeout: {exc}",
                                  grasp_held=None, duration_s=0.0)
            row.update(rep.to_dict())
            row["status"] = "ok"
            if not rep.ok:
                row["fallback"] = fallback_restore(sim, sim_snapshot.load(start_file)).to_dict()
            logger.info(f"[RESET] {i:2d} {pick['id']:<30} {pick['label']:<8} -> "
                        f"{'ok' if rep.ok else 'FAILED'}, freed {rep.belt_freed_sim}, after "
                        f"{rep.label_after_sim}")
            report["rows"].append(row)
            report["summary"] = _summary(report["rows"])
            write_json(out / "report.json", report)
    finally:
        sim.close("finished")
    report["summary"] = _summary(report["rows"])
    report["wall_s"] = time.perf_counter() - t_run
    write_json(out / "report.json", report)
    s = report["summary"]
    logger.info(f"[RESET] {s['ok']}/{s['n']} ok; per label "
                + ", ".join(f"{k} {v['ok']}/{v['n']} (freed {v['freed']})"
                            for k, v in s["per_label"].items())
                + f"; jitter max err {s['jitter_accuracy']['max_err_mm']} mm / "
                  f"{s['jitter_accuracy']['max_yaw_err_deg']} deg; safety {s['safety']} -> {out}")
    return report


def main() -> int:
    args = create_parser().parse_args()
    cli.configure_logging()
    run_cli(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
