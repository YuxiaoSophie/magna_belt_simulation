#!/usr/bin/env python3
"""Collect LCS episodes of the round-belt engagement segment ``pre_place_1 -> place_3``.

Each episode restores the ``pre_place_1`` start state (belt held by both arms), perturbs the
``pre_place_2`` / ``place_3`` waypoints per a sampled intent (``round_belt_task.perturbation``),
plays the motion in-process, samples every ``--sample-period`` with a camera point cloud, labels
the outcome (``round_belt_task.outcome``) and writes ``<out>/episode_XXXX.npz`` in the
lcs_learning format (``docs/lcs-dataset.md``) plus ``<out>/index.json``. No magna, no LCM.

Run:
    uv run python scripts/collect_lcs_dataset.py --episodes 40 --seed 0
    uv run python scripts/collect_lcs_dataset.py --episodes 4 --seed 1 --dry-run
    uv run python scripts/collect_lcs_dataset.py --episodes 4 --record --out data/lcs/smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from loguru import logger

from round_belt_task import clearance as clr
from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import pose7_from_mat
from round_belt_task.constants import ARM_TARGET_KD, ARM_TARGET_KE
from round_belt_task.motion import MotionError
from round_belt_task.outcome import (
    DEFAULT_THRESHOLDS,
    LABELS,
    OutcomeThresholds,
    classify_episode,
)
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot

FIRST, LAST = "pre_place_1", "place_3"
DEFAULT_START_STATE = sim_snapshot.DEFAULT_START_STATE_DIR / f"{FIRST}.npz"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "lcs"
RECORD_STATE_EVERY = 4
CLEARANCE_EVERY = 4  # 20 ms; the UR moves <= 1.6 mm between checks
PATH_CLAMP_ITERS = 3
START_PHASE = "start"


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stdout, level=level,
               format="<level>{level: <7}</level> | <level>{message}</level>")


def _csv(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default data/lcs/<YYYYmmdd-HHMMSS>-<label>)")
    p.add_argument("--label", default="lcs")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--intents", default=",".join(pert.INTENTS))
    p.add_argument("--weights", default="1,1,1,1")
    p.add_argument("--start-state", type=Path, default=DEFAULT_START_STATE)
    p.add_argument("--fresh-pick", action="store_true",
                   help="run the nominal pick per episode instead of restoring (slow)")
    p.add_argument("--record", action="store_true",
                   help="a RunRecorder run per episode under <out>/recordings/")
    p.add_argument("--settle-s", type=float, default=1.0)
    p.add_argument("--min-clearance", type=float, default=clr.DEFAULT_MIN_CLEARANCE_MM,
                   help="mm of 2F-85 -> board clearance the perturbed UR waypoints are clamped "
                        "to (0 disables the clamp; the clearance is measured either way)")
    p.add_argument("--sample-period", type=float, default=lcs.SAMPLE_PERIOD_S,
                   help="seconds, a multiple of the 5 ms control step")
    p.add_argument("--arm-ke", type=float, default=ARM_TARGET_KE)
    p.add_argument("--arm-kd", type=float, default=ARM_TARGET_KD)
    p.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML)
    p.add_argument("--no-pcd", action="store_true",
                   help="skip camera renders; files lack pcd and fail validation (speed tests)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the sampled perturbation table and exit")
    p.add_argument("--thresholds", nargs="*", default=[], metavar="KEY=VALUE",
                   help="OutcomeThresholds overrides")
    return p


def parse_thresholds(pairs: list[str]) -> OutcomeThresholds:
    fields = {f.name: f.type for f in dataclasses.fields(OutcomeThresholds)}
    kw = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in fields:
            raise ValueError(f"--thresholds {pair!r}: expected KEY=VALUE, KEY in {list(fields)}")
        kw[key] = int(value) if fields[key] in (int, "int") else float(value)
    return dataclasses.replace(DEFAULT_THRESHOLDS, **kw)


def sample_steps(period_s: float) -> int:
    n = round(period_s / lcs.SIM_DT_S)
    if n < 1 or abs(n * lcs.SIM_DT_S - period_s) > 1e-9:
        raise ValueError(f"--sample-period {period_s}: not a multiple of {lcs.SIM_DT_S} s")
    return n


def plan_episodes(args, ranges=None) -> list[pert.Perturbation]:
    """The per-episode perturbations, a pure function of ``--seed``/``--intents``/``--weights``."""
    ranges = pert.DEFAULT_RANGES if ranges is None else ranges
    intents = _csv(args.intents)
    weights = np.array([float(w) for w in _csv(args.weights)])
    if len(weights) != len(intents) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(f"--weights {args.weights!r} do not fit --intents {args.intents!r}")
    weights = weights / weights.sum()
    out = []
    for i in range(args.episodes):
        rng = np.random.default_rng([args.seed, i])
        intent = intents[int(rng.choice(len(intents), p=weights))]
        out.append(pert.sample(rng, intent, ranges))
    return out


def perturbation_table(perts: list[pert.Perturbation]) -> str:
    rows = ["  ep  intent    ur dx/dy/dz mm         ur tilt  franka dx/dy/dz mm     f tilt"]
    for i, p in enumerate(perts):
        u, f = p.ur_dpos_m * 1e3, p.franka_dpos_m * 1e3
        rows.append(f"{i:4d}  {p.intent:<8} {u[0]:6.2f} {u[1]:6.2f} {u[2]:7.2f}  "
                    f"{p.ur_tilt_deg:7.2f}  {f[0]:6.2f} {f[1]:6.2f} {f[2]:6.2f}  "
                    f"{p.franka_tilt_deg:6.2f}")
    return "\n".join(rows)


def confusion_table(rows: list[dict], intents: list[str]) -> str:
    counts = Counter((r["intent"], r["outcome"]) for r in rows if r.get("status") == "ok")
    skipped = Counter(r["intent"] for r in rows if r.get("status") != "ok")
    head = f"{'intent':<9}" + "".join(f"{lab:>9}" for lab in LABELS) + f"{'skipped':>9}"
    lines = [head]
    for intent in intents:
        lines.append(f"{intent:<9}" + "".join(f"{counts[(intent, lab)]:>9}" for lab in LABELS)
                     + f"{skipped[intent]:>9}")
    total = Counter(r["outcome"] for r in rows if r.get("status") == "ok")
    lines.append(f"{'total':<9}" + "".join(f"{total[lab]:>9}" for lab in LABELS)
                 + f"{sum(skipped.values()):>9}")
    return "\n".join(lines)


def clearance_table(rows: list[dict], intents: list[str]) -> str:
    """Per-intent 2F-85 -> board clearance (mm) and how much the guard had to lift."""
    head = (f"{'intent':<9}{'n':>4}{'min mm':>9}{'median mm':>11}{'max lift mm':>13}"
            f"{'contacts':>10}")
    lines = [head]
    for intent in [*intents, "total"]:
        got = rows if intent == "total" else [r for r in rows if r["intent"] == intent]
        if not got:
            continue
        low = [r["min_board_clearance_mm"] for r in got]
        lift = [r["clamp_lift_mm"] for r in got]
        lines.append(f"{intent:<9}{len(got):>4}{min(low):>9.2f}{float(np.median(low)):>11.2f}"
                     f"{max(lift):>13.2f}{sum(r['board_contact'] for r in got):>10}")
    return "\n".join(lines)


def _git() -> dict:
    def run(*cmd) -> str:
        return subprocess.run(["git", *cmd], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lcs.json_default) + "\n")
    os.replace(tmp, path)


class EpisodeSampler:
    """``on_step`` for :meth:`RoundBeltOfflineSimulation.play`: one frame every ``n`` steps."""

    def __init__(self, sim, n: int, pcd: bool, traj_phase: np.ndarray, phase_offset: int,
                 board, gripper, every: int = CLEARANCE_EVERY) -> None:
        self.sim, self.n, self.pcd, self.phase_offset = sim, n, pcd, phase_offset
        self.traj_phase = traj_phase
        self.frames: list[dict] = []
        self.render_s = 0.0
        self.sample_s = 0.0
        self.clearance_s = 0.0
        self.board, self.gripper, self.every = board, gripper, every
        self.min_clearance_m = float("inf")
        self.last_clearance_m: float | None = None
        self._large = int(sim.info.pulley_bodies[1])

    def clearance(self, body_q) -> float:
        t0 = time.perf_counter()
        value = self.gripper.measure(self.board, np.asarray(body_q, dtype=np.float64))
        self.min_clearance_m = min(self.min_clearance_m, value)
        self.last_clearance_m = value
        self.clearance_s += time.perf_counter() - t0
        return value

    def sample(self, step: int, cmd_f, cmd_u, phase: int, body_q=None) -> None:
        t0 = time.perf_counter()
        sim = self.sim
        body_q = sim.state_0.body_q.numpy() if body_q is None else body_q
        meas_f, meas_u = sim.ee_poses(body_q)
        q_f, q_u = sim.arm_positions()
        v_f, v_u = sim.arm_velocities()
        belt = body_q[sim.info.belt_bodies, :3].astype(np.float64)
        grasp = sim.grasp_state(body_q)
        frame = {
            "step": step, "time": sim.sim_time, "q_f": q_f, "q_u": q_u, "v_f": v_f, "v_u": v_u,
            "ee_f": pose7_from_mat(meas_f), "ee_u": pose7_from_mat(meas_u),
            "cmd_f": pose7_from_mat(cmd_f), "cmd_u": pose7_from_mat(cmd_u),
            "track_mm": np.array([np.linalg.norm(meas_f[:3, 3] - cmd_f[:3, 3]),
                                  np.linalg.norm(meas_u[:3, 3] - cmd_u[:3, 3])]) * 1e3,
            "belt": belt, "pulley_raw": body_q[self._large].astype(np.float64),
            "hand_mm": grasp.franka_width_mm, "robotiq": grasp.ur_status, "phase": phase,
            "grasp_ok": np.array(grasp.held()),
            "clearance_mm": (self.last_clearance_m if self.last_clearance_m is not None
                             else self.clearance(body_q)) * 1e3,
        }
        t1 = time.perf_counter()
        if self.pcd:
            xyz, rgb = sim.point_cloud()
            frame["pcd"], frame["pcd_rgb"] = xyz, rgb
        self.render_s += time.perf_counter() - t1
        self.sample_s += time.perf_counter() - t0
        self.frames.append(frame)

    def __call__(self, i: int, info: dict) -> None:
        sampling = (i + 1) % self.n == 0
        if sampling or i % self.every == 0:
            self.clearance(info["body_q"])
        if sampling:
            phase = int(self.traj_phase[i]) + self.phase_offset
            self.sample(i + 1, info["franka_cmd"], info["ur_cmd"], phase, info["body_q"])


def build_writer(frames: list[dict], n: int, pcd: bool) -> lcs.EpisodeWriter:
    writer = lcs.EpisodeWriter(sample_steps=n)
    T = len(frames)
    for t, fr in enumerate(frames):
        state = lcs.state_vector(fr["q_f"], fr["q_u"], fr["v_f"], fr["v_u"], fr["cmd_f"],
                                 fr["cmd_u"])
        if t + 1 < T:
            nxt = frames[t + 1]
            action = lcs.action_vector(fr["cmd_f"], nxt["cmd_f"], fr["cmd_u"], nxt["cmd_u"])
        else:
            action = np.zeros(lcs.ACTION_DIM)
        cloud = lcs.camera_points(fr["pcd"]) if pcd else np.zeros((1, 3), np.float32)
        writer.add_frame(fr["sim_step"], fr["time"], state, action, cloud,
                         lcs.belt_points_ordered(fr["belt"]), lcs.kinematic_points(),
                         pcd_rgb=fr.get("pcd_rgb") if pcd else None, extras={
                             "phase": np.int16(fr["phase"]), "ee_franka": fr["ee_f"],
                             "ee_ur": fr["ee_u"], "ee_cmd_franka": fr["cmd_f"],
                             "ee_cmd_ur": fr["cmd_u"], "belt_xyz": fr["belt"].astype(np.float32),
                             "pulley_large_pose": lcs.pose_from_xyz_xyzw(fr["pulley_raw"]),
                             "hand_mm": np.float32(fr["hand_mm"]),
                             "robotiq_byte": np.uint8(fr["robotiq"]),
                             "tracking_err_mm": fr["track_mm"], "grasp_ok": fr["grasp_ok"],
                             "episode_step": np.int64(fr["step"]),
                             "board_clearance_mm": np.float32(fr["clearance_mm"]),
                         })
    return writer


def _finish_recording(sim, out: Path, name: str, reason: str) -> str | None:
    recorder = sim.recorder
    if recorder is None:
        return None
    sim.close_recording(reason)
    sim.recorder = None
    target = out / "recordings" / name
    if target.exists():
        return str(recorder.path.relative_to(out))
    recorder.path.rename(target)
    return str(target.relative_to(out))


def collect(args: argparse.Namespace, ranges: dict | None = None) -> dict:
    """Run the collection; returns the ``index.json`` dict."""
    from round_belt_task.offline_simulation import RoundBeltOfflineSimulation

    ranges = pert.DEFAULT_RANGES if ranges is None else ranges
    n = sample_steps(args.sample_period)
    thresholds = parse_thresholds(args.thresholds)
    intents = _csv(args.intents)
    perts = plan_episodes(args, ranges)
    nominal = load_pre_mpc_segment(args.params, first=FIRST, last=LAST)
    tangent = pert.belt_tangent(nominal)
    if args.dry_run:
        print(perturbation_table(perts))
        return {"episodes": [p.to_dict() for p in perts]}

    out = args.out or DEFAULT_OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.label}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start_meta: dict = {"path": None if args.fresh_pick else str(args.start_state)}
    snap = None
    if not args.fresh_pick:
        snap = sim_snapshot.load(args.start_state)
        start_meta.update(sha256=_sha256(args.start_state), label=snap.meta.get("label"),
                          notes=snap.meta.get("notes"), created=snap.meta.get("created"),
                          git_commit=snap.meta.get("git_commit"),
                          step_index=snap.step_index, sim_time=snap.sim_time)
    index = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git": _git(), "start_state": start_meta, "sample_period_s": n * lcs.SIM_DT_S,
        "sample_steps": n, "thresholds": dataclasses.asdict(thresholds),
        "ranges": pert.ranges_to_dict(ranges), "labels": list(LABELS),
        "belt_tangent": tangent, "min_clearance_mm": args.min_clearance,
        "clearance_every": CLEARANCE_EVERY, "episodes": [],
    }
    index_path = out / "index.json"
    _write_json(index_path, index)

    sim = RoundBeltOfflineSimulation.build(arm_ke=args.arm_ke, arm_kd=args.arm_kd)
    sim.args.record_state_every = RECORD_STATE_EVERY
    if snap is not None:
        sim.restore(snap)  # the guard's geometry is read at the start pose, gripper closed
    jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
    board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
    logger.info(f"[LCS] clearance guard: min {args.min_clearance:g} mm, {len(gripper.local)} "
                f"2F-85 points on {len(gripper.shape_labels)} colliders (jaw bytes {jaws}) vs "
                f"plate + {len(board.pulleys)} pulleys")
    initial = sim_snapshot.capture(sim, "initial") if args.fresh_pick else None
    t_run = time.perf_counter()
    timings = []
    try:
        for i, p in enumerate(perts):
            row = run_episode(sim, i, p, args, out, n, thresholds, tangent, snap, initial,
                              nominal, board, gripper)
            index["episodes"].append(row)
            _write_json(index_path, index)
            if row["status"] == "ok":
                timings.append(row["timing_s"])
    finally:
        sim.close("finished")
    wall = time.perf_counter() - t_run
    rows = index["episodes"]
    ok = [r for r in rows if r["status"] == "ok"]
    print(confusion_table(rows, intents))
    print(clearance_table(ok, intents))
    per_min = 60.0 * len(ok) / wall if wall > 0 else 0.0
    contacts = [r["file"] for r in ok if r["board_contact"]]
    summary = {"episodes_ok": len(ok), "episodes_skipped": len(rows) - len(ok),
               "wall_s": wall, "episodes_per_min": per_min,
               "board_contact_files": contacts,
               "min_board_clearance_mm": min((r["min_board_clearance_mm"] for r in ok),
                                             default=None)}
    if contacts:
        logger.warning(f"[LCS] {len(contacts)} episode(s) touched the board and are flagged "
                       f"sim_board_contact / board_contact: {', '.join(contacts)}")
    if timings:
        summary["timing_mean_s"] = {k: float(np.mean([t[k] for t in timings]))
                                    for k in timings[0]}
    index["summary"] = summary
    _write_json(index_path, index)
    mean = summary.get("timing_mean_s", {})
    logger.info(f"[LCS] {len(ok)} ok / {len(rows) - len(ok)} skipped in {wall:.1f} s = "
                f"{per_min:.2f} episodes/min; mean per episode "
                + ", ".join(f"{k} {v:.2f}" for k, v in mean.items()) + f" s -> {out}")
    return index


def run_episode(sim, i: int, p: pert.Perturbation, args, out: Path, n: int, thresholds,
                tangent, snap, initial, nominal, board, gripper) -> dict:
    t0 = time.perf_counter()
    timing = {}
    name = f"episode_{i:04d}"
    if args.fresh_pick:
        sim.restore(initial)
        sim.nominal_pick(args.params, log_every_s=0)
    else:
        sim.restore(snap)
    timing["restore"] = time.perf_counter() - t0
    row = {"file": f"{name}.npz", "intent": p.intent, "perturbation": p.to_dict()}
    waypoints, clamp = clr.clamp_waypoints(nominal, p, tangent, board, gripper,
                                           min_clearance=args.min_clearance * 1e-3)
    row["clamp"] = clamp.to_dict()
    t1 = time.perf_counter()
    want = args.min_clearance * 1e-3
    try:
        traj = sim.plan(waypoints, settle_s=args.settle_s)
        for _ in range(PATH_CLAMP_ITERS):
            low = clr.path_clearance(board, gripper, traj.ur_4x4, traj.ur_gripper_byte)
            if clamp.path_before_mm is None:
                clamp.path_before_mm = low * 1e3
            if low >= want - clr.CLAMP_TOL_M:
                break
            clamp.path_lift_mm += (want - low) * 1e3
            waypoints = clr.lift_waypoints(waypoints, pert.PERTURBED_LABELS, want - low)
            traj = sim.plan(waypoints, settle_s=args.settle_s)
        clamp.path_after_mm = clr.path_clearance(board, gripper, traj.ur_4x4,
                                                 traj.ur_gripper_byte) * 1e3
    except MotionError as exc:
        logger.warning(f"[LCS] {i:4d} {p.intent:<8} skipped: {exc}")
        return {**row, "file": None, "status": "skipped", "reason": str(exc)}
    row["clamp"] = clamp.to_dict()
    timing["ik"] = time.perf_counter() - t1

    if args.record:
        sim.start_recording(out / "recordings", f"{name}-{p.intent}")
    sampler = EpisodeSampler(sim, n, pcd=not args.no_pcd, traj_phase=traj.phase, phase_offset=1,
                             board=board, gripper=gripper)
    start_step = sim.step_index
    cmd_f, cmd_u = sim.commanded_poses()
    sampler.sample(0, cmd_f, cmd_u, 0)
    first_sample_s, first_clearance_s = sampler.sample_s, sampler.clearance_s
    t2 = time.perf_counter()
    result = sim.play(traj, on_step=sampler, log_every_s=0)
    play_s = time.perf_counter() - t2
    timing["render"] = sampler.render_s
    timing["clearance"] = sampler.clearance_s - first_clearance_s
    timing["physics"] = play_s - (sampler.sample_s - first_sample_s) - timing["clearance"]
    recording = _finish_recording(sim, out, f"{name}-{p.intent}", "episode done")

    t3 = time.perf_counter()
    frames = sampler.frames
    for fr in frames:
        fr["sim_step"] = start_step + fr["step"]
    belt_T = np.stack([fr["belt"] for fr in frames])
    pulley_T = np.stack([fr["pulley_raw"] for fr in frames])
    label, metrics = classify_episode(belt_T, pulley_T, thresholds)
    track = result.summary()
    min_clear_mm = sampler.min_clearance_m * 1e3
    contact = min_clear_mm < 0.0
    phase_labels = [START_PHASE, *traj.labels]
    extras_meta = {
        "phase_labels": phase_labels, "intent": p.intent, "outcome": label,
        "perturbation": p.to_dict(), "ee_pose_source": "commanded Cartesian target",
        "action_source": "commanded Cartesian target at samples t and t+1",
        "pulley_pose_layout": lcs.POSE_LAYOUT, "start_step": start_step,
        "start_state": None if args.fresh_pick else str(args.start_state),
        "thresholds": dataclasses.asdict(thresholds), "clamp": clamp.to_dict(),
        "clearance_source": "2F-85 colliders vs the board plate box and the two pulleys, "
                            f"sampled every {CLEARANCE_EVERY} control steps",
    }
    post = {
        "intent": np.array(p.intent), "outcome": np.array(label),
        "perturbation": np.array(json.dumps(p.to_dict())),
        "wrap_deg": metrics["wrap_deg"], "h_median_mm": metrics["h_median_mm"],
        "min_board_clearance_mm": np.float32(min_clear_mm),
        "board_contact": np.array(contact),
        "clamp": np.array(json.dumps(clamp.to_dict())),
    }
    post = {lcs.EXTRA_PREFIX + k: v for k, v in post.items()}
    omit: tuple[str, ...] = ()
    if args.no_pcd:
        post["sim_no_pcd"] = np.array(True)
        omit = ("pcd", "pcd_rgb")
    path = out / row["file"]
    build_writer(frames, n, pcd=not args.no_pcd).write(path, label, extras_meta, arrays=post,
                                                        omit=omit)
    if not args.no_pcd:
        lcs.validate_episode(path, period_us=round(n * lcs.SIM_DT_S * 1e6))
    timing["write"] = time.perf_counter() - t3
    timing["total"] = time.perf_counter() - t0

    wrap, h_med = float(metrics["wrap_deg"][-1]), float(metrics["h_median_mm"][-1])
    log = logger.warning if contact else logger.info
    log(f"[LCS] {i:4d} {p.intent:<8} -> {label:<8} wrap {wrap:6.1f} deg, h median "
        f"{h_med:6.1f} mm, clearance {min_clear_mm:5.2f} mm"
        f"{' CONTACT' if contact else ''} (lift {clamp.max_lift_mm:.2f} mm, of which path "
        f"{clamp.path_lift_mm:.2f} mm), {timing['total']:.1f} s")
    return {**row, "outcome": label, "steps": len(traj), "frames": len(frames),
            "duration_s": len(traj) * lcs.SIM_DT_S,
            "tracking_rms_mm": [track["franka_mm_rms"], track["ur_mm_rms"]],
            "final_wrap_deg": wrap, "final_h_median_mm": h_med,
            "min_board_clearance_mm": min_clear_mm, "board_contact": contact,
            "clamp_lift_mm": clamp.max_lift_mm, "clamp_tilt_scale": clamp.tilt_scale,
            "grasp_ok_final": [bool(v) for v in frames[-1]["grasp_ok"]],
            "size_bytes": path.stat().st_size, "recording": recording,
            "timing_s": timing, "status": "ok"}


def main() -> int:
    args = create_parser().parse_args()
    configure_logging()
    collect(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
