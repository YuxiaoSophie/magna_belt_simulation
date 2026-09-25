#!/usr/bin/env python3
"""Evaluate magna's assembly controller (learned staged MPC or the waypoint baseline) in the sim.

Each episode restores a start state (the nominal ``pre_place_1_osc.npz`` or one of a grasp-varied
set), settles 0.5 s under the OSC hold, starts a fresh ``run_round_belt_assembly_controller``
(worktree binary, params yaml picks learned vs baseline) next to magna's Franka OSC on one private
LCM URL, and steps the sim in lock-step with the OSC: every 15 steps the harness samples a frame
(camera cloud, material belt points, 40-dim state) and, in learned mode, publishes
``LATENT_STATE`` from the in-process encoder; the UR is position-driven by IK along the
controller's UR line; grippers stay at the start-state commands. The episode ends on the
controller's end marker (+ ``--settle-s``), ``--max-episode-s`` (``timeout``) or a safety abort.
Episodes are classified (``outcome.classify_episode``), scored against the demo's belt/poses
(``demo_goals.npz``) and written in the dataset format plus ``index.json``.

Run:
    uv run python scripts/lcs/eval_learned_mpc.py --mode baseline --repeats 2
    uv run python scripts/lcs/eval_learned_mpc.py --mode learned --repeats 2
    uv run python scripts/lcs/eval_learned_mpc.py --mode learned \
        --start-states data/lcs/start_states/grasp_variants/set1 --variants gv_00,gv_03
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_lcs_dataset as cli
from loguru import logger

from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import UrTracking, ik
from round_belt_task.controller_bridge import ControllerBridge
from round_belt_task.episode_io import (
    finish_recording,
    git_info,
    pose7_mat,
    sha256_file,
    slant_row,
    write_json,
)
from round_belt_task.motion import MotionError
from round_belt_task.osc_bridge import OscTimeout
from round_belt_task.outcome import (
    LABELS,
    classify_episode,
    slant_episode,
)
from round_belt_task.waypoints import (
    MAGNA_PARAMS_SIM_YAML,
    load_pre_mpc_segment,
)
from task_common import belt_metrics as bm
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.latent_encoder import (
    DemoGoals,
    LatentEncoder,
    LearnedLcs,
    latent_state_message,
)
from task_common.magna_process import (
    CONTROLLER_MARKERS,
    CONTROLLER_STARTED,
    MAGNA_WORKTREE,
    AssemblyControllerProcess,
    check_private_url,
    parse_pre_mpc_line,
    parse_stage_line,
    sha256,
)
from task_common.osc_process import MAGNA_ROOT, OSC_BINARY

MODES = ("learned", "baseline")
DEFAULT_LCM_URL = "udpm://239.255.76.88:7688?ttl=0"
# Must match DEFAULT_PARAMS["learned"]'s lcs_file (learned_lcs_v2_flat_pp2.yaml).
DEFAULT_DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v2_20260925/"
                      "deploy_v2_flat_pp2/deploy.npz")
DEFAULT_DEMO_GOALS = REPO_ROOT / "data" / "lcs" / "demo_flat_pp2" / "demo_goals.npz"
DEFAULT_START_STATE = sim_snapshot.DEFAULT_START_STATE_DIR / "pre_place_1_osc.npz"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "lcs" / "mpc_eval"
DEFAULT_PARAMS = {
    "learned": "systems/parameters/round_belt_controller_params_learned_eval.yaml",
    "baseline": "systems/parameters/round_belt_controller_params_baseline_eval.yaml",
}
WORKTREE_OSC = MAGNA_WORKTREE / OSC_BINARY.relative_to(MAGNA_ROOT)
END_MARKERS = {"learned": ("MPC completed!", "Switching to Terminate"),
               "baseline": ("All pre-MPC targets completed!",)}
SAMPLE_STEPS = lcs.SAMPLE_STEPS
START_SETTLE_S = 0.5
CONTROLLER_START_TIMEOUT_S = 60.0
TIP_FLOOR_MM = 2.0
BOARD_NEGATIVE_STEPS = 40
PRED_ITERS = 25
N_KNOTS = 7
END_POLL_S = 0.05
UR_EXACT_DT_TOL_S = 1e-6  # line knot times are utime-quantised (1 us)
# Where the frame's command comes from (sim_cmd_source).
CMD_RESPONSE, CMD_IN_FORCE, CMD_NONE = 0, 1, 2


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=MODES, default="learned")
    p.add_argument("--params", default=None,
                   help="controller params yaml relative to --magna-root (default: the mode's "
                        "*_eval.yaml)")
    p.add_argument("--magna-root", type=Path, default=MAGNA_WORKTREE)
    p.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY, help="learned mode: deploy.npz")
    p.add_argument("--demo-goals", type=Path, default=DEFAULT_DEMO_GOALS)
    p.add_argument("--start-states", type=Path, default=DEFAULT_START_STATE,
                   help="a snapshot .npz or a grasp-variant set dir with index.json")
    p.add_argument("--variants", default=None, help="comma list of variant ids (set dir only)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=None,
                   help="shuffle the episode order with this seed (default: set order)")
    p.add_argument("--lcm-url", default=DEFAULT_LCM_URL)
    p.add_argument("--osc-binary", type=Path, default=None,
                   help="default: the worktree's OSC if built, else the main checkout's")
    p.add_argument("--osc-timeout-s", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=None,
                   help="default data/lcs/mpc_eval/<YYYYmmdd-HHMMSS>-<label>")
    p.add_argument("--label", default=None, help="default: the mode")
    p.add_argument("--max-episode-s", type=float, default=12.0)
    p.add_argument("--pace", type=float, default=1.0,
                   help="minimum wall seconds per simulated second (0 = free-running)")
    p.add_argument("--settle-s", type=float, default=0.5, help="after the end marker")
    p.add_argument("--record", action="store_true",
                   help="a RunRecorder run per episode under <out>/recordings/")
    p.add_argument("--no-pcd", action="store_true",
                   help="baseline only: skip camera renders (files fail validation)")
    p.add_argument("--thresholds", nargs="*", default=[], metavar="KEY=VALUE",
                   help="OutcomeThresholds overrides")
    p.add_argument("--dry-run", action="store_true", help="print the episode plan and exit")
    p.add_argument("--action-definition", choices=lcs.CMD_ACTION_DEFINITIONS,
                   default=lcs.DEFAULT_ACTION_DEFINITION,
                   help="recorded actions (and the latent_pred input): cmd_delta = knot1 - knot0 "
                        "of the plan in force / line(t+dt) - line(t) (fresh exact-dt line: line "
                        "end - latent ee_ur = u0); knot1_minus_measured = the pre-2026-09-25 "
                        "definition (matches models trained on it)")
    return p


# --- start states -----------------------------------------------------------------------------

@dataclasses.dataclass
class StartState:
    id: str
    file: Path
    commanded: dict | None = None
    measured: dict | None = None
    held: bool | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "file": str(self.file), "commanded": self.commanded,
                "measured": self.measured, "held": self.held}


def _resolve(base: Path, file: str) -> Path:
    path = Path(file)
    if path.is_absolute():
        return path
    for root in (base, REPO_ROOT):
        if (root / path).exists():
            return root / path
    return base / path


def load_start_states(path: Path, variants: str | None) -> tuple[list[StartState], dict]:
    """``(states, meta)`` from a snapshot ``.npz`` or a variant-set dir (``index.json``)."""
    path = Path(path)
    if path.is_file():
        if variants:
            raise ValueError("--variants needs a variant-set dir for --start-states")
        return [StartState(path.stem, path)], {"path": str(path), "kind": "snapshot",
                                               "sha256": sha256_file(path)}
    index_path = path / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"{path}: neither a snapshot .npz nor a dir with index.json")
    index = json.loads(index_path.read_text())
    rows = index.get("variants", [])
    by_id = {str(r["id"]): r for r in rows}
    wanted = ([v.strip() for v in variants.split(",") if v.strip()] if variants
              else [str(r["id"]) for r in rows])
    missing = [v for v in wanted if v not in by_id]
    if missing:
        raise ValueError(f"variants {missing} not in {index_path} ({sorted(by_id)})")
    states = []
    for vid in wanted:
        row = by_id[vid]
        if row.get("file") is None or row.get("held") is False:
            logger.warning(f"[EVAL] variant {vid} has no held start state; skipped")
            continue
        states.append(StartState(vid, _resolve(path, row["file"]), row.get("commanded"),
                                 row.get("measured"), row.get("held")))
    meta = {"path": str(path), "kind": "variant_set", "set": index.get("set"),
            "index_sha256": sha256_file(index_path), "variants": [s.id for s in states]}
    return states, meta


def plan_episodes(states: list[StartState], repeats: int, seed: int | None
                  ) -> list[tuple[StartState, int]]:
    plan = [(s, r) for s in states for r in range(repeats)]
    if seed is not None:
        order = np.random.default_rng(seed).permutation(len(plan))
        plan = [plan[i] for i in order]
    return plan


# --- metrics ----------------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def alignment(pcd_belt, ee_f, ee_u, demo: DemoGoals, stage: int) -> dict:
    """Belt/pose alignment of one frame to the demo's stage ``stage`` frame."""
    ref = demo.pcd_belt_stage[stage]
    belt = np.asarray(pcd_belt, dtype=np.float64)
    shift_mm, shift = bm.belt_best_shift_rmse_mm(belt, ref)
    f_mm, f_deg = bm.pose_error(ee_f, demo.ee_pose_franka[stage])
    u_mm, u_deg = bm.pose_error(ee_u, demo.ee_pose_ur[stage])
    return {"stage": int(stage), "stage_label": demo.stage_labels[stage],
            "belt_rmse_mm": bm.belt_rmse_mm(belt, ref),
            "belt_chamfer_mm": bm.belt_chamfer_mm(belt, ref),
            "belt_best_shift_rmse_mm": shift_mm, "belt_best_shift": int(shift),
            "franka_pose_err": [f_mm, f_deg], "ur_pose_err": [u_mm, u_deg]}


def _stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "median": None, "mean": None}
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean())}


def summarize(rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("outcome") is not None]
    engaged = sum(r["outcome"] == "engaged" for r in done)
    lo, hi = wilson_ci(engaged, len(done))
    out = {"episodes": len(rows), "classified": len(done),
           "skipped": sum(r["status"] == "skipped" for r in rows),
           "aborted": sum(r["status"] == "aborted" for r in rows),
           "timeouts": sum(r["status"] == "timeout" for r in rows),
           "engaged": engaged, "engaged_rate": engaged / len(done) if done else None,
           "engaged_ci95": [lo, hi] if done else None,
           "outcomes": dict(Counter(r["outcome"] for r in done)),
           "duration_s_mean": _stats(r["duration_s"] for r in done)["mean"]}
    final = [r["alignment"]["final"] for r in done if r.get("alignment")]
    out["alignment_final"] = {k: _stats(a[k] for a in final) for k in
                              ("belt_rmse_mm", "belt_chamfer_mm", "belt_best_shift_rmse_mm")}
    gd = [r["latent"]["goal_dist_final"] for r in done if r.get("latent")]
    gd = [g for g in gd if g is not None]
    if gd:
        arr = np.asarray(gd, dtype=np.float64)
        out["goal_dist_final"] = {"mean": np.nanmean(arr, axis=0).tolist() if np.isfinite(
            arr).any() else None, "median": np.nanmedian(arr, axis=0).tolist() if np.isfinite(
            arr).any() else None}
    ends = Counter()
    for r in done:
        for k, how in enumerate(r.get("stage_ends", [])):
            ends[f"stage{k}_{how}"] += 1
    out["stage_ends"] = dict(ends)
    return out


# --- episode ----------------------------------------------------------------------------------

@dataclasses.dataclass
class EvalContext:
    args: argparse.Namespace
    out: Path
    mode: str
    params_rel: Path
    thresholds: object
    tangent: np.ndarray
    board: object
    gripper: object
    encoder: LatentEncoder | None
    lcs_model: LearnedLcs | None
    demo: DemoGoals | None
    phase_labels: list[str]
    n_pre: int
    n_stages: int
    stage_labels: list[str]
    snaps: dict = dataclasses.field(default_factory=dict)
    controller_info: dict | None = None
    learned_mpc_meta: dict | None = None  # recording meta.json "learned_mpc" (sans start)

    @property
    def plate_top_z(self) -> float:
        plate = self.board.plate
        return float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half)


def _marker_events(lines: list[tuple[str, float]], mode: str) -> dict:
    """Parse ``(line, detected osc t)`` marker lines into times and stage ends."""
    ev = {"pre_reached": {}, "stages": {}, "all_pre_completed": None, "mpc_completed": None,
          "terminate": None, "no_pre_mpc": None, "end": None}
    for line, t_seen in lines:
        pre = parse_pre_mpc_line(line)
        stage = parse_stage_line(line)
        if pre is not None:
            ev["pre_reached"].setdefault(pre[0], pre[1])
        elif stage is not None:
            ev["stages"].setdefault(stage[0], {"how": stage[1], "t": stage[2], "dist": stage[3]})
        elif "All pre-MPC targets completed!" in line:
            ev["all_pre_completed"] = ev["all_pre_completed"] or t_seen
        elif "No pre-MPC targets found" in line:
            ev["no_pre_mpc"] = ev["no_pre_mpc"] or t_seen
        if "MPC completed!" in line:
            ev["mpc_completed"] = ev["mpc_completed"] or t_seen
        if "Switching to Terminate" in line:
            ev["terminate"] = ev["terminate"] or t_seen
        if ev["end"] is None and any(m in line for m in END_MARKERS[mode]):
            ev["end"] = t_seen
    return ev


def _phase_and_stage(ctx: EvalContext, ev: dict, t: float) -> tuple[int, int]:
    """(phase index into ``ctx.phase_labels``, stage count) at osc time ``t``."""
    n_pre, done = ctx.n_pre, len(ctx.phase_labels) - 1
    pre_done = sum(1 for tt in ev["pre_reached"].values() if tt <= t + 1e-9)
    stages_done = sum(1 for s in ev["stages"].values() if s["t"] <= t + 1e-9)
    stage = stages_done if ctx.mode == "learned" else pre_done
    all_pre = ev["all_pre_completed"]
    if n_pre and (all_pre is None or t < all_pre):
        return min(pre_done, n_pre - 1), stage
    if ctx.mode == "baseline":
        return done, stage
    end = ev["mpc_completed"]
    if end is not None and t >= end:
        return done, stage
    return n_pre + min(stages_done, max(ctx.n_stages - 1, 0)), stage


def _resolve_commands(frames: list[dict], bridge: ControllerBridge, dt_step: float) -> None:
    """Per frame: the controller's Franka plan answering this tick (else the plan in force)
    and the UR line published with it; ``knot0/knot1/knots/ur_t/ur_t1`` + ``cmd_source``."""
    trajs, lines = bridge.franka_trajs, bridge.ur_lines
    eps = 0.5 * dt_step
    for fr in frames:
        t = fr["time"]
        pick = next((tr for tr in trajs if t - eps <= tr.times[0] <= t + 2 * dt_step + eps),
                    None)
        source = CMD_RESPONSE
        if pick is None:
            older = [tr for tr in trajs if tr.times[0] < t - eps]
            pick = older[-1] if older else None
            source = CMD_IN_FORCE if pick is not None else CMD_NONE
        if pick is None:
            knots = np.repeat(fr["ee_f"][None, :], N_KNOTS, axis=0)
            times = t + lcs.ACTION_KNOT_DT_S * np.arange(N_KNOTS)
            fr["traj_utime"] = -1
        else:
            knots = np.hstack([pick.pos, pick.quat])
            times = pick.times
            fr["traj_utime"] = pick.utime
        n = len(knots)
        fr["n_knots"] = n
        if n < N_KNOTS:
            knots = np.vstack([knots, np.repeat(knots[-1:], N_KNOTS - n, 0)])
            times = np.concatenate([times, np.repeat(times[-1:], N_KNOTS - n)])
        fr["knots"], fr["knot_times"] = knots[:N_KNOTS].copy(), np.asarray(times[:N_KNOTS])
        fr["knot0"], fr["knot1"] = knots[0].copy(), knots[1].copy()
        fr["cmd_source"] = source
        line = None
        if pick is not None:
            line = next((m for m in lines if m.wall >= pick.wall), None)
            if line is None:
                earlier = [m for m in lines if m.wall < pick.wall]
                line = earlier[-1] if earlier else None
        else:
            earlier = [m for m in lines if m.wall <= fr["wall"]]
            line = earlier[-1] if earlier else None
        fr["ur_fresh_exact"] = False
        fr["ur_own_delta"] = np.full(6, np.nan)
        if line is None:
            fr["ur_t"] = fr["ur_t1"] = fr["ee_u"].copy()
            fr["ur_line"] = np.full(16, np.nan)
        else:
            ln = line.line
            fr["ur_line"] = np.concatenate([ln.p0, ln.q0, ln.p1, ln.q1, [ln.t0, ln.t1]])
            if _fresh_exact_dt(ln, t):
                # Line starts at the measured tool0 ~5 ms late; anchor at the latent pose => u0.
                end = pose7_mat(bridge.tracking_pose(ln, ln.t1))
                own = lcs.action_vector(end, end, pose7_mat(bridge.tracking_pose(ln, ln.t0)),
                                        end)
                fr["ur_t"], fr["ur_t1"] = fr["ee_u"].copy(), end
                fr["ur_fresh_exact"] = True
                fr["ur_own_delta"] = own[[3, 4, 5, 9, 10, 11]]
            else:
                fr["ur_t"] = pose7_mat(bridge.tracking_pose(ln, t))
                fr["ur_t1"] = pose7_mat(bridge.tracking_pose(ln, t + lcs.ACTION_KNOT_DT_S))


def _fresh_exact_dt(ln, t: float) -> bool:
    """A learned-MPC UR line of exactly one knot dt that starts inside this frame."""
    dt = lcs.ACTION_KNOT_DT_S
    return abs((ln.t1 - ln.t0) - dt) <= UR_EXACT_DT_TOL_S and t - 1e-9 <= ln.t0 < t + dt


def _gripper_at(bridge: ControllerBridge, wall: float) -> tuple[float, int]:
    hand = [h for h in bridge.hand_commands if h[0] <= wall]
    rq = [r for r in bridge.robotiq_commands if r[0] <= wall]
    return (hand[-1][2] if hand else math.nan), (rq[-1][1] if rq else -1)


def frame_action(fr: dict, definition: str) -> np.ndarray:
    return lcs.command_action(definition, fr["ee_f"], fr["ee_u"], fr["knot0"], fr["knot1"],
                              fr["ur_t"], fr["ur_t1"])


def build_writer(frames: list[dict], pcd: bool,
                 definition: str = lcs.DEFAULT_ACTION_DEFINITION
                 ) -> tuple[lcs.EpisodeWriter, np.ndarray]:
    writer = lcs.EpisodeWriter(sample_steps=SAMPLE_STEPS, action_definition=definition)
    track = cli.tracking_errors(frames)
    realised = lcs.realised_delta(np.stack([fr["prop"] for fr in frames]))
    for t, fr in enumerate(frames):
        action = frame_action(fr, definition)
        cloud = fr["cloud"] if pcd else np.zeros((1, 3), np.float32)
        writer.add_frame(fr["sim_step"], fr["time"], fr["prop"], action, cloud, fr["belt_pts"],
                         lcs.kinematic_points(), pcd_rgb=fr.get("pcd_rgb") if pcd else None,
                         extras={
                             "ee_franka": fr["ee_f"], "ee_ur": fr["ee_u"],
                             "cmd_knot0_franka": fr["knot0"], "cmd_knot1_franka": fr["knot1"],
                             "cmd_knots_franka": fr["knots"],
                             "cmd_knot_times_franka": fr["knot_times"],
                             "cmd_n_knots": np.int16(fr["n_knots"]),
                             "cmd_traj_utime": np.int64(fr["traj_utime"]),
                             "cmd_ur_t": fr["ur_t"], "cmd_ur_t1": fr["ur_t1"],
                             "cmd_ur_line": fr["ur_line"],
                             "cmd_ur_fresh_exact": np.int8(fr["ur_fresh_exact"]),
                             "ur_line_own_window_delta": fr["ur_own_delta"],
                             "cmd_source": np.int8(fr["cmd_source"]),
                             "cmd_ctrl_hand_mm": np.float32(fr["ctrl_hand_mm"]),
                             "cmd_ctrl_robotiq_byte": np.int16(fr["ctrl_robotiq"]),
                             "phase": np.int16(fr["phase"]), "stage": np.int16(fr["stage"]),
                             "latent": fr["latent"], "latent_pred": fr["latent_pred"],
                             "goal_dist": fr["goal_dist"],
                             "osc_utime": np.int64(fr["osc_utime"]),
                             "render_step": np.int64(fr["render_step"]),
                             "belt_xyz": fr["belt"].astype(np.float32),
                             "pulley_large_pose": lcs.pose_from_xyz_xyzw(fr["pulley_raw"]),
                             "hand_mm": np.float32(fr["hand_mm"]),
                             "robotiq_byte": np.uint8(fr["robotiq"]),
                             "grasp_ok": fr["grasp_ok"], "episode_step": np.int64(fr["step"]),
                             "board_clearance_mm": np.float32(fr["clearance_mm"]),
                             "franka_tip_clearance_mm": np.float32(fr["tip_clearance_mm"]),
                             "tracking_err_mm": track[t],
                             "action_knot1_minus_measured":
                                 frame_action(fr, "knot1_minus_measured"),
                             "realised_delta": realised[t],
                         })
    return writer, track


def run_episode(sim, ctx: EvalContext, state: StartState, repeat: int) -> dict:
    args, bridge, n = ctx.args, sim.bridge, SAMPLE_STEPS
    dt = sim.frame_dt
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    name = f"episode_{state.id}_{repeat}"
    row = {"file": f"{name}.npz", "start_state": state.to_dict(), "repeat": repeat,
           "mode": ctx.mode}
    snap = ctx.snaps.get(state.id)
    if snap is None:
        snap = ctx.snaps[state.id] = sim_snapshot.load(state.file)
    sim.restore(snap, settle_steps=0)
    timing["restore"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    grasp = sim.settle(round(START_SETTLE_S / dt))
    timing["settle"] = time.perf_counter() - t1
    if grasp.held() != (True, True):
        logger.warning(f"[EVAL] {name} skipped: grasp lost after settle ({grasp.describe()})")
        return {**row, "file": None, "status": "skipped", "outcome": None,
                "reason": f"grasp lost after settle ({grasp.describe()})"}

    sim.commander_hook = None  # the controller owns TARGET_CARTESIAN_POSE_TRAJECTORY now
    bridge.reset_controller_io()
    bridge.reset_stats()
    log_path = ctx.out / "controller" / f"{name}.log"
    ctrl = AssemblyControllerProcess()
    t2 = time.perf_counter()
    events_raw: list[tuple[str, float]] = []
    status, reason = "ok", None
    try:
        ctrl.start(args.lcm_url, ctx.params_rel, log_path, magna_root=args.magna_root)
        ctrl.wait_for(CONTROLLER_STARTED, CONTROLLER_START_TIMEOUT_S)
        if ctx.controller_info is None:
            ctx.controller_info = ctrl.describe()
        timing["controller_start"] = time.perf_counter() - t2
        if args.record:
            # Plan/debug times are OSC clock: osc_utime = step * control_dt_us + offset.
            sim.start_recording(ctx.out / "recordings", name, extra_meta={
                "learned_mpc": _recording_learned_meta(ctx, state),
                "osc_utime_offset_us": int(bridge.utime_offset_us)})
        sampler = cli.EpisodeSampler(sim, n, pcd=not args.no_pcd,
                                     traj_phase=np.zeros(0, np.int64), phase_offset=0,
                                     board=ctx.board, gripper=ctx.gripper)
        loop = _run_loop(sim, ctx, ctrl, sampler, events_raw)
        status, reason = loop["status"], loop["reason"]
        frames = loop["frames"]
        # Late answers to the last published state.
        end = time.perf_counter() + END_POLL_S
        while time.perf_counter() < end:
            bridge.poll()
            time.sleep(0.005)
        events_raw.extend((line, sim.osc_time_s()) for line in ctrl.read_new_lines()
                          if any(m in line for m in CONTROLLER_MARKERS))
    finally:
        ctrl_code = ctrl.stop()
        sim.commander_hook = sim.make_hold_hook()
    timing.update(loop["timing"])
    recording = finish_recording(sim, ctx.out, name, "episode done")

    t3 = time.perf_counter()
    ev = _marker_events(events_raw, ctx.mode)
    _resolve_commands(frames, bridge, dt)
    t_start = frames[0]["time"]
    for fr in frames:
        fr["phase"], fr["stage"] = _phase_and_stage(ctx, ev, fr["time"])
        fr["ctrl_hand_mm"], fr["ctrl_robotiq"] = _gripper_at(bridge, fr["wall"])
    latent_stats = _latent_diagnostics(ctx, frames)
    belt_T = np.stack([fr["belt"] for fr in frames])
    pulley_T = np.stack([fr["pulley_raw"] for fr in frames])
    label, metrics = classify_episode(belt_T, pulley_T, ctx.thresholds)
    metrics.update(slant_episode(belt_T, pulley_T, ctx.tangent, ctx.thresholds))
    min_clear_mm = sampler.min_clearance_m * 1e3
    tip_min_mm = loop["tip_min_m"] * 1e3
    align = None
    if ctx.demo is not None:
        first, last = frames[0], frames[-1]
        align = {"initial": alignment(first["belt_pts"], first["ee_f"], first["ee_u"], ctx.demo,
                                      0),
                 "final": alignment(last["belt_pts"], last["ee_f"], last["ee_u"], ctx.demo,
                                    ctx.demo.n_stages - 1)}
    ctrl_events = _events_dict(ev, t_start)
    stage_ends = [ev["stages"][k]["how"] for k in sorted(ev["stages"])]
    stage_times = {str(k): ev["stages"][k]["t"] - t_start for k in sorted(ev["stages"])}
    if ctx.mode == "baseline":
        stage_times = {str(k): v - t_start for k, v in sorted(ev["pre_reached"].items())}
    grip_diff = sorted({(round(h, 3), int(b)) for h, b in
                        ((fr["ctrl_hand_mm"], fr["ctrl_robotiq"]) for fr in frames)
                        if not math.isnan(h) or b >= 0})
    if grip_diff:
        logger.info(f"[EVAL] {name}: controller gripper commands (not applied): {grip_diff}")

    extras_meta = {
        "phase_labels": ctx.phase_labels, "mode": ctx.mode, "outcome": label, "status": status,
        "reason": reason, "start_state": state.to_dict(), "repeat": repeat, "backend": "osc",
        "controller": ctx.controller_info, "controller_events": ctrl_events,
        "ee_pose_source": "measured finger_tip / tracking frame (FK of the measured joints)",
        "action_source": (
            "knot 1 - knot 0 of the controller's TARGET_CARTESIAN_POSE_TRAJECTORY answering the "
            "sample tick (else the plan in force) / the controller's UR line at t + knot dt minus "
            "at t (tracking frame, world); a fresh exact-dt line starting in the frame "
            "(sim_cmd_ur_fresh_exact): its end minus the frame's ee_ur (= the latent's "
            "ee_pose_ur, so u0), own-window span in sim_ur_line_own_window_delta"
            if args.action_definition == "cmd_delta" else
            "knot 1 of the controller's TARGET_CARTESIAN_POSE_TRAJECTORY answering "
            "the sample tick (else the plan in force) / the controller's UR line at "
            "t + knot dt (tracking frame, world), minus the measured pose at t"),
        "cmd_source_codes": {"response": CMD_RESPONSE, "in_force": CMD_IN_FORCE,
                             "none": CMD_NONE},
        "latent_source": ("in-process LatentEncoder, published as LATENT_STATE"
                          if ctx.mode == "learned" else "none (baseline)"),
        "time_source": "OSC clock (FRANKA_STATE utime)", "sample_period_s": n * lcs.SIM_DT_S,
        "pulley_pose_layout": lcs.POSE_LAYOUT, "start_step": frames[0]["sim_step"],
        "thresholds": dataclasses.asdict(ctx.thresholds),
        "grippers": "held at the start-state commands; controller commands logged only",
    }
    post = {"outcome": np.array(label), "status": np.array(status), "mode": np.array(ctx.mode),
            "start_state": np.array(json.dumps(state.to_dict())),
            "wrap_deg": metrics["wrap_deg"], "h_median_mm": metrics["h_median_mm"],
            **{k: np.array(metrics[k]) for k in cli.SLANT_KEYS},
            "min_board_clearance_mm": np.float32(min_clear_mm),
            "board_contact": np.array(min_clear_mm < 0.0),
            "min_franka_tip_clearance_mm": np.float32(tip_min_mm),
            "alignment": np.array(json.dumps(align)),
            "backend": np.array("osc"),
            "osc_utime_offset_us": np.int64(bridge.utime_offset_us)}
    post = {lcs.EXTRA_PREFIX + k: v for k, v in post.items()}
    omit: tuple[str, ...] = ()
    if args.no_pcd:
        post["sim_no_pcd"] = np.array(True)
        omit = ("pcd", "pcd_rgb")
    path = ctx.out / row["file"]
    writer, track = build_writer(frames, pcd=not args.no_pcd, definition=args.action_definition)
    writer.write(path, label, extras_meta, arrays=post, omit=omit)
    if not args.no_pcd:
        lcs.validate_episode(path, period_us=round(n * lcs.SIM_DT_S * 1e6))
    timing["write"] = time.perf_counter() - t3
    timing["total"] = time.perf_counter() - t0

    moving = np.array([ctx.phase_labels[fr["phase"]] != "done" for fr in frames])
    moving[0] = False
    move = track[moving] if moving.any() else track[1:]
    steps = frames[-1]["step"]
    out_row = {
        **row, "outcome": label, "status": status, "reason": reason,
        "duration_s": steps * dt, "steps": steps, "frames": len(frames),
        "controller_events": ctrl_events, "stage_times": stage_times, "stage_ends": stage_ends,
        "latent": latent_stats, "alignment": align,
        "tracking_rms_mm": [float(np.sqrt(np.mean(move[:, j] ** 2))) for j in range(2)],
        "tracking_max_mm": [float(move[:, j].max()) for j in range(2)],
        "final_wrap_deg": float(metrics["wrap_deg"][-1]),
        "final_h_median_mm": float(metrics["h_median_mm"][-1]), **slant_row(metrics),
        "min_board_clearance_mm": min_clear_mm, "board_contact": min_clear_mm < 0.0,
        "min_franka_tip_clearance_mm": tip_min_mm,
        "grasp_ok_final": [bool(v) for v in frames[-1]["grasp_ok"]],
        "cmd_source_counts": dict(Counter(int(fr["cmd_source"]) for fr in frames)),
        "controller_gripper_commands": [list(g) for g in grip_diff],
        "timing_s": timing, "osc_stats": {**bridge.stats(), **bridge.controller_stats()},
        "controller_log": str(log_path.relative_to(ctx.out)), "controller_exit": ctrl_code,
        "recording": recording, "size_bytes": path.stat().st_size,
    }
    a = align["final"] if align else None
    logger.info(f"[EVAL] {name} {ctx.mode}: {label} ({status}{f': {reason}' if reason else ''})"
                f", {steps * dt:.2f} s, stages {stage_ends}, wrap "
                f"{out_row['final_wrap_deg']:.1f} deg"
                + (f", final belt rmse {a['belt_rmse_mm']:.1f} / chamfer "
                   f"{a['belt_chamfer_mm']:.1f} / shift {a['belt_best_shift_rmse_mm']:.1f} mm"
                   if a else "")
                + f", tracking {out_row['tracking_rms_mm'][0]:.2f}/"
                  f"{out_row['tracking_rms_mm'][1]:.2f} mm, {timing['total']:.1f} s")
    return out_row


def _events_dict(ev: dict, t_start: float) -> dict:
    out = {}
    for i, t in sorted(ev["pre_reached"].items()):
        out[f"pre_mpc_target_{i}_reached"] = t - t_start
    for k, s in sorted(ev["stages"].items()):
        out[f"stage_{k}_{s['how']}"] = s["t"] - t_start
        out[f"stage_{k}_dist"] = s["dist"]
    for key in ("no_pre_mpc", "all_pre_completed", "mpc_completed", "terminate"):
        if ev[key] is not None:
            out[key] = ev[key] - t_start
    return out


def _latent_diagnostics(ctx: EvalContext, frames: list[dict]) -> dict | None:
    """Fill ``latent_pred``/``goal_dist`` per frame; the row's latent summary."""
    n_goal = ctx.demo.n_stages if ctx.demo is not None else (
        ctx.lcs_model.stage_goals.shape[0] if ctx.lcs_model is not None else 1)
    nx = ctx.encoder.latent_dim if ctx.encoder is not None else 16
    prev = None
    errs = []
    for fr in frames:
        z = fr["latent"]
        fr["latent_pred"] = np.full(nx, np.nan)
        fr["goal_dist"] = np.full(n_goal, np.nan)
        if not np.isfinite(z).all():
            continue
        if ctx.demo is not None:
            fr["goal_dist"] = np.array([ctx.demo.dist(z, k) for k in range(n_goal)])
        elif ctx.lcs_model is not None:
            fr["goal_dist"] = np.array([ctx.lcs_model.whitened_dist(z, g)
                                        for g in ctx.lcs_model.stage_goals])
        if prev is not None and ctx.lcs_model is not None:
            u = frame_action(prev, ctx.args.action_definition)
            fr["latent_pred"] = ctx.lcs_model.step(prev["latent"], u, iters=PRED_ITERS)[0]
            errs.append(float(np.linalg.norm((fr["latent_pred"] - z) / ctx.lcs_model.z_std)))
        prev = fr
    if ctx.mode != "learned":
        return None
    gd = np.stack([fr["goal_dist"] for fr in frames])
    return {"pred_err_rms_whitened": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
            "goal_dist_first": gd[0].tolist(), "goal_dist_final": gd[-1].tolist(),
            "goal_dist_min": np.nanmin(gd, axis=0).tolist(),
            "goal_source": "demo_goals" if ctx.demo is not None else "deploy stage goals"}


def _run_loop(sim, ctx: EvalContext, ctrl, sampler, events_raw: list) -> dict:
    args, bridge, n = ctx.args, sim.bridge, SAMPLE_STEPS
    dt = sim.frame_dt
    u_coords = sim.arm_coords()[1]
    plate_top = ctx.plate_top_z
    q_ur = sim.arm_targets()[1]
    lead_gain = sim.arm_kd / sim.arm_ke / sim.frame_dt
    lead_key = None
    max_steps = round(args.max_episode_s / dt)
    settle_steps = round(args.settle_s / dt)
    min_wall = args.pace * dt
    frames: list[dict] = []
    s0 = sim.step_index
    k = 0
    end_at: int | None = None
    abort: str | None = None
    status, reason = "ok", None
    tip_min = math.inf
    board_neg = 0
    timing = {"render": 0.0, "encode": 0.0, "ik": 0.0, "step": 0.0, "pace_sleep": 0.0}
    wait0 = bridge.wait_s
    t_loop = time.perf_counter()
    while True:
        if k % n == 0:
            finished = abort is not None or (end_at is not None and k >= end_at + settle_steps)
            if abort is None and end_at is None and k >= max_steps:
                status, finished = "timeout", True
                reason = f"no end marker after {args.max_episode_s:g} s"
            frames.append(_sample(sim, ctx, sampler, k, s0, publish=not finished, timing=timing))
            if abort is None and not all(frames[-1]["grasp_ok"]):
                abort = f"grasp lost at step {k} ({sim.grasp_state().describe()})"
                status, reason, finished = "aborted", abort, True
            if finished:
                break
        w0 = time.perf_counter()
        sim.control_step()
        k += 1
        bridge.poll()
        timing["step"] += time.perf_counter() - w0
        joint_q = sim.state_0.joint_q.numpy()
        ee_f = sim.franka_measured_pose7(joint_q)
        tip = float(ee_f[2] - plate_top)
        tip_min = min(tip_min, tip)
        if k % cli.CLEARANCE_EVERY == 0:
            sampler.clearance(sim.state_0.body_q.numpy())
        clear = sampler.last_clearance_m
        board_neg = board_neg + 1 if clear is not None and clear < 0.0 else 0
        for line in ctrl.read_new_lines():
            if any(m in line for m in CONTROLLER_MARKERS):
                events_raw.append((line, sim.osc_time_s()))
                if end_at is None and any(m in line for m in END_MARKERS[ctx.mode]):
                    end_at = k
        if abort is None:
            if tip * 1e3 < TIP_FLOOR_MM:
                abort = f"franka finger_tip {tip * 1e3:.2f} mm above the plate at step {k}"
            elif board_neg > BOARD_NEGATIVE_STEPS:
                abort = f"2F-85 board clearance < 0 for {board_neg} steps at step {k}"
            elif not ctrl.alive():
                abort = f"controller exited ({ctrl.proc.returncode}) at step {k}"
            if abort is not None:
                status, reason = "aborted", abort
                logger.warning(f"[EVAL] abort: {abort}")
                _hold(sim, ctrl)
        line = bridge.latest_ur_line() if abort is None else None
        if line is not None:
            i0 = time.perf_counter()
            X = bridge.tracking_pose(line, sim.osc_time_s(sim.step_index + 1))
            q_new, err_p, err_r, iters = ik(UrTracking, X, q_ur, pos_tol=cli.UR_IK_POS_TOL,
                                            rot_tol=cli.UR_IK_ROT_TOL)
            if err_p > cli.UR_IK_POS_TOL or err_r > cli.UR_IK_ROT_TOL:
                abort = (f"UR IK missed at step {k} ({err_p * 1e3:.3f} mm, "
                         f"{np.degrees(err_r):.3f} deg after {iters} iterations)")
                status, reason = "aborted", abort
                logger.warning(f"[EVAL] abort: {abort}")
                _hold(sim, ctrl)
            else:
                # Velocity feed-forward as in the collector, never across a line change (the
                # controller republishes the same line every tick: compare its knot times).
                key = (line.t0, line.t1)
                lead = cli.UR_VELOCITY_LEAD * lead_gain if key == lead_key else 0.0
                sim.set_ur_target(q_new + lead * (q_new - q_ur))
                q_ur, lead_key = q_new, key
            timing["ik"] += time.perf_counter() - i0
        elif abort is not None:
            sim.set_ur_target(joint_q[u_coords])
        spent = time.perf_counter() - w0
        if spent < min_wall:
            time.sleep(min_wall - spent)
            timing["pace_sleep"] += min_wall - spent
    timing["loop"] = time.perf_counter() - t_loop
    timing["lcm_wait"] = bridge.wait_s - wait0
    return {"frames": frames, "status": status, "reason": reason, "tip_min_m": tip_min,
            "timing": timing}


def _hold(sim, ctrl) -> None:
    """Safety stop: controller off (own pid), Franka held at the measured pose."""
    ctrl.stop()
    sim.commander_hook = sim.make_hold_hook()


def _sample(sim, ctx: EvalContext, sampler, k: int, s0: int, publish: bool,
            timing: dict) -> dict:
    """Frame at the CURRENT state (step ``s0 + k``); learned mode sets its ``LATENT_STATE``."""
    bridge = sim.bridge
    body_q = sim.state_0.body_q.numpy()
    q_f, q_u = sim.arm_positions()
    v_f, v_u = sim.arm_velocities()
    ee_f = sim.franka_measured_pose7()
    ee_u = pose7_mat(UrTracking.fk(q_u))
    grasp = sim.grasp_state(body_q)
    utime = bridge.osc_utime(sim.step_index)
    t = utime * 1e-6
    belt = body_q[sim.info.belt_bodies, :3].astype(np.float64)
    sampler.clearance(body_q)
    frame = {
        "step": k, "sim_step": sim.step_index, "time": t, "osc_utime": utime,
        "wall": time.perf_counter(), "q_f": q_f, "q_u": q_u, "v_f": v_f, "v_u": v_u,
        "ee_f": ee_f, "ee_u": ee_u, "belt": belt,
        "pulley_raw": body_q[int(sim.info.pulley_bodies[1])].astype(np.float64),
        "hand_mm": grasp.franka_width_mm, "robotiq": grasp.ur_status,
        "grasp_ok": np.array(grasp.held()), "clearance_mm": sampler.last_clearance_m * 1e3,
        "tip_clearance_mm": (ee_f[2] - ctx.plate_top_z) * 1e3, "render_step": sim.step_index,
        "prop": lcs.state_vector(q_f, q_u, v_f, v_u, ee_f, ee_u),
        "belt_pts": lcs.belt_points_ordered(belt),
    }
    r0 = time.perf_counter()
    if not ctx.args.no_pcd:
        xyz, rgb = sim.point_cloud()
        frame["cloud"], frame["pcd_rgb"] = lcs.camera_points(xyz), rgb
    timing["render"] += time.perf_counter() - r0
    frame["latent"] = np.full(ctx.encoder.latent_dim if ctx.encoder else 16, np.nan)
    if ctx.mode == "learned":
        e0 = time.perf_counter()
        z = ctx.encoder.encode(frame["cloud"], frame["prop"], frame["belt_pts"])
        frame["latent"] = z
        if publish:
            bridge.set_latent(latent_state_message(utime, t, z, ee_f, ee_u, frame["prop"]))
        timing["encode"] += time.perf_counter() - e0
    return frame


# --- run --------------------------------------------------------------------------------------

def _file_ref(path: Path | None) -> tuple[str | None, str | None]:
    if path is None or not Path(path).is_file():
        return None, None
    path = Path(path).resolve()
    return str(path), sha256_file(path)


def learned_mpc_meta(args: argparse.Namespace, mode: str, params_rel: Path, params: dict,
                     demo: DemoGoals | None) -> dict:
    """Run-level part of a recording's ``meta.json`` ``learned_mpc`` block (baseline: nulls)."""
    keys = ("params_yaml", "lcs_yaml", "demo_traj_yaml", "deploy", "demo_goals", "demo_episode")
    out = {"mode": mode, **{k: None for k in keys}, **{f"{k}_sha256": None for k in keys},
           "action_definition": None, "debug_channel": None}
    if mode != "learned":
        return out
    block = params["learned_mpc"]
    root = Path(args.magna_root)
    demo_traj = (block.get("demo_traj") or {}).get("file")
    episode = None
    if demo is not None:
        episode = Path(demo.demo_episode)
        episode = episode if episode.is_absolute() else REPO_ROOT / episode
    files = {"params_yaml": root / params_rel, "lcs_yaml": root / block["lcs_file"],
             "demo_traj_yaml": root / demo_traj if demo_traj else None,
             "deploy": args.deploy, "demo_goals": args.demo_goals, "demo_episode": episode}
    for k, path in files.items():
        out[k], out[f"{k}_sha256"] = _file_ref(path)
    out["action_definition"] = args.action_definition
    out["debug_channel"] = block.get("debug_channel")
    return out


def _recording_learned_meta(ctx: EvalContext, state: StartState) -> dict:
    meta = dict(ctx.learned_mpc_meta or {"mode": ctx.mode})
    meta["start_state"] = ({"id": state.id, "file": str(state.file)}
                           if ctx.mode == "learned" else None)
    return meta


def _controller_yaml(root: Path, rel: Path) -> dict:
    return yaml.safe_load((root / rel).read_text()) or {}


def default_osc_binary() -> tuple[Path, Path]:
    if WORKTREE_OSC.exists():
        return WORKTREE_OSC, MAGNA_WORKTREE
    return OSC_BINARY, MAGNA_ROOT


def evaluate(args: argparse.Namespace) -> dict:
    """Run the evaluation; returns the ``index.json`` dict."""
    mode = args.mode
    check_private_url(args.lcm_url)
    if args.no_pcd and mode == "learned":
        raise ValueError("--no-pcd is baseline only (the encoder needs the cloud)")
    params_rel = Path(args.params or DEFAULT_PARAMS[mode])
    params = _controller_yaml(args.magna_root, params_rel)
    has_learned = params.get("learned_mpc") is not None
    if has_learned != (mode == "learned"):
        raise ValueError(f"--mode {mode} but {params_rel} "
                         f"{'has' if has_learned else 'lacks'} a learned_mpc block")
    pre_labels = [str(w.get("label", f"target{i}"))
                  for i, w in enumerate(params.get("pre_mpc_motion") or [])]
    states, start_meta = load_start_states(args.start_states, args.variants)
    plan = plan_episodes(states, args.repeats, args.seed)
    if args.dry_run:
        for i, (s, r) in enumerate(plan):
            print(f"{i:4d}  {s.id:<24} repeat {r}  {s.file}")
        return {"episodes": [{"start_state": s.to_dict(), "repeat": r} for s, r in plan]}

    encoder = lcs_model = None
    deploy_meta = None
    if mode == "learned":
        encoder, lcs_model = LatentEncoder.load(args.deploy), LearnedLcs.load(args.deploy)
        deploy_meta = {"path": str(args.deploy), "sha256": sha256_file(args.deploy),
                       "goal_source": lcs_model.goal_source,
                       "n_stage_goals": int(lcs_model.stage_goals.shape[0])}
    demo = None
    demo_meta = None
    if args.demo_goals is not None and Path(args.demo_goals).is_file():
        demo = DemoGoals.load(args.demo_goals)
        demo_meta = {"path": str(args.demo_goals), "sha256": sha256_file(args.demo_goals),
                     "stage_labels": demo.stage_labels, "demo_episode": demo.demo_episode}
    else:
        logger.warning(f"[EVAL] no demo goals at {args.demo_goals}: alignment metrics off, "
                       "goal_dist vs the deploy's stage goals")
    n_stages = 0
    stage_labels: list[str] = []
    if lcs_model is not None:
        n_stages = int(lcs_model.stage_goals.shape[0])
        stage_labels = (demo.stage_labels if demo is not None and demo.n_stages == n_stages
                        else [f"stage{k}" for k in range(n_stages)])
    phase_labels = ([f"pre:{lab}" for lab in pre_labels]
                    + [f"mpc:{lab}" for lab in stage_labels] + ["done"])

    label = args.label or mode
    out = Path(args.out or DEFAULT_OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}")
    out.mkdir(parents=True, exist_ok=True)
    thresholds = cli.parse_thresholds(args.thresholds)
    nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=cli.FIRST, last=cli.LAST)
    tangent = pert.belt_tangent(nominal)
    osc_binary, osc_root = default_osc_binary()
    if args.osc_binary is not None:
        osc_binary = Path(args.osc_binary)
        osc_root = MAGNA_WORKTREE if MAGNA_WORKTREE in osc_binary.parents else MAGNA_ROOT
    index = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git": git_info(), "mode": mode, "start_states": start_meta,
        "controller": {"mode": mode, "magna_root": str(args.magna_root),
                       "params": str(params_rel),
                       "params_sha256": sha256(args.magna_root / params_rel)},
        "deploy": deploy_meta, "demo_goals": demo_meta,
        "sample_period_s": SAMPLE_STEPS * lcs.SIM_DT_S, "sample_steps": SAMPLE_STEPS,
        "action_definition": args.action_definition, "belt_sampling": lcs.BELT_SAMPLING,
        "thresholds": dataclasses.asdict(thresholds), "labels": list(LABELS),
        "phase_labels": phase_labels, "belt_tangent": tangent, "episodes": [],
    }
    index_path = out / "index.json"
    write_json(index_path, index)

    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=args.osc_timeout_s,
                                       bridge_cls=ControllerBridge)
    sim.args.record_state_every = cli.RECORD_STATE_EVERY
    sim.bridge.record_targets = bool(args.record)
    t_run = time.perf_counter()
    rows = index["episodes"]
    try:
        warm = sim.start_osc(out / "osc.log", binary=osc_binary, cwd=osc_root)
        index["osc"] = {**sim.osc.describe(), "warm_up_s": warm, "timeout_s": args.osc_timeout_s}
        first = sim_snapshot.load(plan[0][0].file)
        sim.restore(first)  # clearance geometry is read at a start pose, gripper closed
        jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
        board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
        ctx = EvalContext(args=args, out=out, mode=mode, params_rel=params_rel,
                          thresholds=thresholds, tangent=tangent, board=board, gripper=gripper,
                          encoder=encoder, lcs_model=lcs_model, demo=demo,
                          phase_labels=phase_labels, n_pre=len(pre_labels), n_stages=n_stages,
                          stage_labels=stage_labels,
                          learned_mpc_meta=learned_mpc_meta(args, mode, params_rel, params, demo))
        for s, r in plan:
            try:
                row = run_episode(sim, ctx, s, r)
            except OscTimeout as exc:
                finish_recording(sim, out, f"episode_{s.id}_{r}", "osc timeout")
                if not exc.process_alive:
                    index["summary"] = {"aborted": "osc died", "error": str(exc)}
                    write_json(index_path, index)
                    raise
                logger.warning(f"[EVAL] {s.id}/{r} skipped: osc timeout ({exc})")
                sim.bridge.forget_pending()
                row = {"file": None, "start_state": s.to_dict(), "repeat": r, "mode": mode,
                       "status": "skipped", "outcome": None, "reason": f"osc timeout: {exc}"}
            except (MotionError, TimeoutError, RuntimeError) as exc:
                logger.warning(f"[EVAL] {s.id}/{r} skipped: {exc}")
                row = {"file": None, "start_state": s.to_dict(), "repeat": r, "mode": mode,
                       "status": "skipped", "outcome": None, "reason": str(exc)}
            rows.append(row)
            index["controller"].update(ctx.controller_info or {})
            index["summary"] = _summary(rows, time.perf_counter() - t_run)
            write_json(index_path, index)
    finally:
        sim.close("finished")
    index["summary"] = _summary(rows, time.perf_counter() - t_run)
    write_json(index_path, index)
    s = index["summary"]["overall"]
    ci = s["engaged_ci95"]
    logger.info(f"[EVAL] {mode}: {s['engaged']}/{s['classified']} engaged"
                + (f" (95% CI {ci[0]:.2f}-{ci[1]:.2f})" if ci else "")
                + f", outcomes {s['outcomes']}, aborted {s['aborted']}, timeouts "
                  f"{s['timeouts']}, skipped {s['skipped']} -> {out}")
    return index


def _summary(rows: list[dict], wall: float) -> dict:
    per = {}
    for sid in dict.fromkeys(r["start_state"]["id"] for r in rows):
        per[sid] = summarize([r for r in rows if r["start_state"]["id"] == sid])
    return {"overall": summarize(rows), "per_start_state": per, "wall_s": wall}


def main() -> int:
    args = create_parser().parse_args()
    cli.configure_logging()
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
