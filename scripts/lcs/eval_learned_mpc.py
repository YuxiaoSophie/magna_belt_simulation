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
(``demo_goals.npz``) and written in the dataset format plus ``index.json``. Opt-in ``--reset
scripted``: a held episode is followed by ``scripted_reset.py`` (lift, return to the start poses
with jitter) instead of the next restore; ``index.json`` then has ``reset_before`` rows and a
``budget`` block. Opt-in ``--controller mppi``: ``scripts/lcs/mppi_controller_node.py`` replaces
the C++ controller (same markers and channels; ``--params`` ignored); ``--mppi-clearance`` also
exports ``<out>/clearance_model.npz`` at the first start state and runs the node with it.

Run:
    uv run python scripts/lcs/eval_learned_mpc.py --mode baseline --repeats 2
    uv run python scripts/lcs/eval_learned_mpc.py --mode learned --repeats 2
    uv run python scripts/lcs/eval_learned_mpc.py --mode learned \
        --start-states data/lcs/start_states/grasp_variants/set1 --variants gv_00,gv_03
    uv run python scripts/lcs/eval_learned_mpc.py --mode learned --controller mppi --mppi-debug
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import zlib
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
    classify,
    classify_episode,
    frame_metrics,
    slant_episode,
)
from round_belt_task.waypoints import (
    MAGNA_PARAMS_SIM_YAML,
    load_pre_mpc_segment,
)
from task_common import belt_cloud as bc
from task_common import belt_metrics as bm
from task_common import clearance_model as cm
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.latent_encoder import (
    BeltCloudPointNetEncoder,
    DemoGoals,
    LatentEncoder,
    LearnedLcs,
    encoder_input,
    latent_state_message,
)
from task_common.lcm_contract import LcmChannels
from task_common.magna_process import (
    CONTROLLER_MARKERS,
    CONTROLLER_STARTED,
    MAGNA_WORKTREE,
    AssemblyControllerProcess,
    MppiControllerProcess,
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
RESET_ARG_DEFAULTS = {"reset": "snapshot", "reset_spec": []}
CLOUD_ARG_DEFAULTS = {"cloud_noise": "off", "belt_seg": False, "belt_seg_source": "camera"}
CONTROLLERS = ("magna", "mppi")
DEFAULT_MPPI_CONFIG = REPO_ROOT / "data" / "lcs" / "realfirst" / "v1" / "mppi.yaml"
MPPI_ARG_DEFAULTS = {"controller": "magna", "mppi_config": DEFAULT_MPPI_CONFIG,
                     "mppi_goal": None, "mppi_stages": "last", "mppi_max_duration_s": 30.0,
                     "mppi_debug": False}
# Dropped from index.args at default even with --controller mppi (pre-clearance index identity).
CLEARANCE_ARG_DEFAULTS = {"mppi_clearance": False, "mppi_clearance_ur_min_mm": cm.DEFAULT_UR_MIN_MM,
                          "mppi_clearance_franka_min_mm": cm.DEFAULT_FRANKA_MIN_MM}
# Demo transport (latent_waypoints.py goals, track mode, demo action prior): dropped from
# index.args at default.
TRANSPORT_ARG_DEFAULTS = {"mppi_waypoints": None, "mppi_goal_mode": "stage",
                          "mppi_demo_actions": None, "mppi_demo_offset_frames": 0}
# Grasp-adaptive reference demo (node --waypoint-bank): dropped from index.args at default.
BANK_ARG_DEFAULTS = {"mppi_waypoint_bank": None, "mppi_bank_knn": 1}
# Where the frame's command comes from (sim_cmd_source).
CMD_RESPONSE, CMD_IN_FORCE, CMD_NONE = 0, 1, 2
CMD_PREFIX = 3  # demo-prefix replay before the controller starts (demo_prefix.py)
PREFIX_ARG_DEFAULTS = {"prefix_demos": None, "prefix_frac": 0.0, "prefix_offset_to_node": None}


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
    p.add_argument("--early-stop", default=None, metavar="W_ENGAGED,W_FAIL,T_MIN",
                   help="end the episode once the per-frame outcome label has held W_ENGAGED s (engaged) or "
                        "W_FAIL s (any other label), not before T_MIN s (default: off)")
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
    p.add_argument("--cloud-noise", default=CLOUD_ARG_DEFAULTS["cloud_noise"],
                   metavar="PRESET|JSON",
                   help="the sensor cloud (encoder input + pcd/pcd_rgb) is the noised one: "
                        f"{sorted(bc.CLOUD_NOISE_PRESETS)} or a .json of CloudNoiseSpec fields "
                        "(RNG [crc32(start id), repeat, 11])")
    p.add_argument("--belt-seg", action="store_true", default=CLOUD_ARG_DEFAULTS["belt_seg"],
                   help="also store pcd_belt_seg (colour-segmented belt, known pulley frame)")
    p.add_argument("--belt-seg-source", choices=("camera", "gt"),
                   default=CLOUD_ARG_DEFAULTS["belt_seg_source"],
                   help="pcd_belt_seg (encoder input + record): camera = colour-segmented cloud; "
                        "gt = the 150 sim material points in the known pulley frame (SIM GT, "
                        "diagnostic only, V12-62)")
    p.add_argument("--reset", choices=("snapshot", "scripted"),
                   default=RESET_ARG_DEFAULTS["reset"],
                   help="between episodes: snapshot restore, or scripted_reset.py (lift, return "
                        "to the start poses + jitter) after a held episode of the same start, "
                        "snapshot fallback = human reset")
    p.add_argument("--reset-spec", nargs="*", default=RESET_ARG_DEFAULTS["reset_spec"],
                   metavar="KEY=VALUE", help="scripted reset: ResetSpec overrides")
    f = PREFIX_ARG_DEFAULTS
    p.add_argument("--prefix-demos", default=f["prefix_demos"], metavar="JSON|PATH",
                   help="demo-prefix replay: {'<set>/<variant>' (a snapshot: its stem): demo "
                        "episode npz} as JSON text or a .json file")
    p.add_argument("--prefix-frac", type=float, default=f["prefix_frac"],
                   help="before the controller starts, replay the start's demo up to this "
                        "fraction (0..1: motion start..pre_place_2 latch, 1..2: latch..first "
                        "hold:place_3; 0 = off); recorded with sim_cmd_source 3")
    p.add_argument("--prefix-offset-to-node", action=argparse.BooleanOptionalAction,
                   default=f["prefix_offset_to_node"],
                   help="mppi: --mppi-demo-offset-frames = the prefix end frame, per episode "
                        "(default: on with --mppi-goal waypoints or --mppi-demo-actions)")
    d = MPPI_ARG_DEFAULTS
    p.add_argument("--controller", choices=CONTROLLERS, default=d["controller"],
                   help="learned mode: magna's C++ assembly controller, or "
                        "scripts/lcs/mppi_controller_node.py (same markers/channels)")
    p.add_argument("--mppi-config", type=Path, default=d["mppi_config"],
                   help="mppi: MppiParams yaml")
    p.add_argument("--mppi-goal", choices=("deploy", "demo_goals", "waypoints"),
                   default=d["mppi_goal"],
                   help="mppi: latent goals from the deploy, --demo-goals (default "
                        "demo_goals when that file exists, else deploy) or --mppi-waypoints")
    p.add_argument("--mppi-stages", choices=("last", "all"), default=d["mppi_stages"],
                   help="mppi: the last goal only (one stage) or every goal as a stage")
    p.add_argument("--mppi-max-duration-s", type=float, default=d["mppi_max_duration_s"],
                   help="mppi: stage timeout (latent time)")
    p.add_argument("--mppi-debug", action="store_true", default=d["mppi_debug"],
                   help="mppi: the node publishes LEARNED_MPC_DEBUG (C++ layout)")
    c = CLEARANCE_ARG_DEFAULTS
    p.add_argument("--mppi-clearance", action="store_true", default=c["mppi_clearance"],
                   help="mppi: 2F-85 / Franka-hand clearance vs plate + pulleys in the MPPI cost "
                        "(model exported at the first start state)")
    p.add_argument("--mppi-clearance-ur-min-mm", type=float,
                   default=c["mppi_clearance_ur_min_mm"], help="mppi clearance: UR set minimum")
    p.add_argument("--mppi-clearance-franka-min-mm", type=float,
                   default=c["mppi_clearance_franka_min_mm"],
                   help="mppi clearance: Franka set minimum (seed demos' rule)")
    t = TRANSPORT_ARG_DEFAULTS
    p.add_argument("--mppi-waypoints", type=Path, default=t["mppi_waypoints"],
                   help="mppi: latent_waypoints.py artifact (with --mppi-goal waypoints)")
    p.add_argument("--mppi-goal-mode", choices=("stage", "track"), default=t["mppi_goal_mode"],
                   help="mppi: stage goals, or track the demo's per-frame latents (waypoints)")
    p.add_argument("--mppi-demo-actions", default=t["mppi_demo_actions"], metavar="auto|PATH",
                   help="mppi: demo action prior inputs (prior_alpha / prior_weight in the "
                        "config); auto = the waypoints artifact's")
    p.add_argument("--mppi-demo-offset-frames", type=int, default=t["mppi_demo_offset_frames"],
                   help="mppi: demo progress index at the node's first latent")
    b = BANK_ARG_DEFAULTS
    p.add_argument("--mppi-waypoint-bank", type=Path, default=b["mppi_waypoint_bank"],
                   metavar="BANK_JSON",
                   help="mppi (with --mppi-goal waypoints): per episode the node picks the bank "
                        "demo with the nearest initial latent (latent_waypoints.py --bank-out)")
    p.add_argument("--mppi-bank-knn", type=int, default=b["mppi_bank_knn"],
                   help="mppi bank: mean of the k nearest demos (1 = nearest only)")
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
    cloud: dict | None = None  # collect_lcs_dataset.cloud_options (+ "origins")
    cloud_rng: np.random.Generator | None = None
    reset: dict | None = None  # --reset scripted: {module, spec, rng, homes}
    mppi: dict | None = None  # --controller mppi: _mppi_options
    prefix: dict | None = None  # --prefix-frac > 0: _prefix_options

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
                         pcd_belt_seg=fr.get("pcd_belt_seg"), extras={
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
                             **cli.cloud_feature_extras(fr),
                         })
    return writer, track


def run_episode(sim, ctx: EvalContext, state: StartState, repeat: int,
                from_reset: dict | None = None) -> dict:
    """``from_reset``: the scripted reset's report; the episode starts from the current state."""
    args, bridge, n = ctx.args, sim.bridge, SAMPLE_STEPS
    dt = sim.frame_dt
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    name = f"episode_{state.id}_{repeat}"
    start = state.to_dict()
    if from_reset is not None:
        start.update(from_reset=True, jitter=from_reset["jitter"])
    row = {"file": f"{name}.npz", "start_state": start, "repeat": repeat,
           "mode": ctx.mode}
    if from_reset is None:
        snap = ctx.snaps.get(state.id)
        if snap is None:
            snap = ctx.snaps[state.id] = sim_snapshot.load(state.file)
        sim.restore(snap, settle_steps=0)
    timing["restore"] = time.perf_counter() - t0
    if ctx.cloud is not None:
        ctx.cloud_rng = np.random.default_rng([zlib.crc32(state.id.encode()), repeat,
                                               cli.CLOUD_NOISE_RNG])
    t1 = time.perf_counter()
    grasp = sim.settle(round(START_SETTLE_S / dt))
    timing["settle"] = time.perf_counter() - t1
    if grasp.held() != (True, True):
        logger.warning(f"[EVAL] {name} skipped: grasp lost after settle ({grasp.describe()})")
        return {**row, "file": None, "status": "skipped", "outcome": None,
                "reason": f"grasp lost after settle ({grasp.describe()})"}
    pre = None
    if ctx.prefix is not None:
        pre = _run_prefix(sim, ctx, state, name, timing)
        if not pre["report"]["ok"]:
            return {**row, "file": None, "status": "skipped", "outcome": None,
                    "reason": f"prefix: {pre['report']['reason']}", "prefix": pre["report"]}

    sim.commander_hook = None  # the controller owns TARGET_CARTESIAN_POSE_TRAJECTORY now
    bridge.reset_controller_io()
    bridge.reset_stats()
    log_path = ctx.out / "controller" / f"{name}.log"
    ctrl = AssemblyControllerProcess() if ctx.mppi is None else MppiControllerProcess()
    t2 = time.perf_counter()
    events_raw: list[tuple[str, float]] = []
    status, reason = "ok", None
    try:
        if ctx.mppi is None:
            ctrl.start(args.lcm_url, ctx.params_rel, log_path, magna_root=args.magna_root)
        else:
            off = (pre["report"]["node_offset_frames"]
                   if pre is not None and ctx.prefix["offset_to_node"] else None)
            _start_mppi(ctrl, ctx, log_path, demo_offset_frames=off)
        ctrl.wait_for(CONTROLLER_STARTED, CONTROLLER_START_TIMEOUT_S)
        if ctx.controller_info is None:
            ctx.controller_info = ctrl.describe()
        timing["controller_start"] = time.perf_counter() - t2
        if args.record and pre is None:
            _start_recording(sim, ctx, state, name)
        sampler = _sampler(sim, ctx) if pre is None else pre["sampler"]
        loop = _run_loop(sim, ctx, ctrl, sampler, events_raw)
        status, reason = loop["status"], loop["reason"]
        frames = loop["frames"]
        n_pre = pre_steps = 0
        if pre is not None:  # one frame sequence; duration_s / events stay controller-relative
            n_pre, pre_steps = len(pre["frames"]), pre["report"]["steps"]
            for fr in frames:
                fr["step"] += pre_steps
            frames = pre["frames"] + frames
        # Late answers to the last published state.
        end = time.perf_counter() + END_POLL_S
        while time.perf_counter() < end:
            bridge.poll()
            time.sleep(0.005)
        events_raw.extend((line, sim.osc_time_s()) for line in ctrl.read_new_lines()
                          if any(m in line for m in CONTROLLER_MARKERS))
    except BaseException:
        if pre is not None:  # the prefix opened the recording before the controller start
            finish_recording(sim, ctx.out, name, "controller error after the prefix")
        raise
    finally:
        ctrl_code = ctrl.stop()
        sim.commander_hook = sim.make_hold_hook()
    timing.update(loop["timing"])
    recording = finish_recording(sim, ctx.out, name, "episode done")

    t3 = time.perf_counter()
    ev = _marker_events(events_raw, ctx.mode)
    _resolve_commands(frames[n_pre:], bridge, dt)
    t_start = frames[n_pre]["time"]
    if ctx.reset is not None and from_reset is None:
        ctx.reset["homes"].setdefault(state.id, (frames[0]["ee_f"].copy(),
                                                 frames[0]["ee_u"].copy()))
    done = len(ctx.phase_labels) - 1
    for j, fr in enumerate(frames):
        fr["phase"], fr["stage"] = _phase_and_stage(ctx, ev, fr["time"])
        if pre is not None:  # phase 0 = the prefix label; "done" stays last
            fr["phase"] = 0 if j < n_pre else (done if fr["phase"] == done else fr["phase"] + 1)
        fr["ctrl_hand_mm"], fr["ctrl_robotiq"] = _gripper_at(bridge, fr["wall"])
    latent_stats = _latent_diagnostics(ctx, frames)
    belt_T = np.stack([fr["belt"] for fr in frames])
    pulley_T = np.stack([fr["pulley_raw"] for fr in frames])
    label, metrics = classify_episode(belt_T, pulley_T, ctx.thresholds)
    metrics.update(slant_episode(belt_T, pulley_T, ctx.tangent, ctx.thresholds))
    min_clear_mm = sampler.min_clearance_m * 1e3
    tip_min_mm = loop["tip_min_m"] * 1e3
    if pre is not None:
        min_clear_mm = min(min_clear_mm, pre["report"]["min_clearance_mm"])
        tip_min_mm = min(tip_min_mm, pre["report"]["tip_min_mm"])
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
        "reason": reason, "start_state": start, "repeat": repeat, "backend": "osc",
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
    if pre is not None:
        extras_meta["cmd_source_codes"]["prefix"] = CMD_PREFIX
        extras_meta["prefix"] = pre["report"]
    if ctx.cloud is not None:
        extras_meta["cloud_noise"] = {**ctx.cloud["noise"].to_dict(),
                                      "camera_origins": ctx.cloud["origins"].tolist()}
        if ctx.cloud["seg"] is not None:
            extras_meta["belt_seg"] = bc.belt_seg_meta(ctx.cloud["noise"], ctx.cloud["seg"],
                                                       "record", ctx.cloud["pose"])
            if ctx.args.belt_seg_source == "gt":
                extras_meta["belt_seg"]["points"] = "sim_gt_pcd_belt"
    post = {"outcome": np.array(label), "status": np.array(status), "mode": np.array(ctx.mode),
            "start_state": np.array(json.dumps(start)),
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
    steps = frames[-1]["step"] - pre_steps
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
    if ctx.mppi is not None:
        out_row["mppi"] = _mppi_solve_stats(ctx.out, log_path)
    if pre is not None:
        out_row["prefix"] = pre["report"]
        timing["prefix"] = pre["wall_s"]
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


def _start_recording(sim, ctx: EvalContext, state: StartState, name: str) -> None:
    # Plan/debug times are OSC clock: osc_utime = step * control_dt_us + offset.
    sim.start_recording(ctx.out / "recordings", name, extra_meta={
        "learned_mpc": _recording_learned_meta(ctx, state),
        "osc_utime_offset_us": int(sim.bridge.utime_offset_us)})


def _sampler(sim, ctx: EvalContext):
    return cli.EpisodeSampler(sim, SAMPLE_STEPS, pcd=not ctx.args.no_pcd,
                              traj_phase=np.zeros(0, np.int64), phase_offset=0,
                              board=ctx.board, gripper=ctx.gripper)


def _run_prefix(sim, ctx: EvalContext, state: StartState, name: str, timing: dict) -> dict:
    """Demo-prefix replay from the settled state (recording on): ``{report, frames, sampler,
    wall_s}``; frames carry the replay's own command (``sim_cmd_source`` 3)."""
    dp, spec = ctx.prefix["module"], ctx.prefix["demos"][state.id]
    w0 = time.perf_counter()
    if ctx.args.record:
        _start_recording(sim, ctx, state, name)
    sampler = _sampler(sim, ctx)
    frames: list[dict] = []
    s0 = sim.step_index

    def on_frame(k: int, cmd: dict) -> None:
        fr = _sample(sim, ctx, sampler, k, s0, publish=False, timing=timing.setdefault(
            "prefix_detail", {"render": 0.0, "encode": 0.0}))
        fr.update(cmd, n_knots=N_KNOTS, cmd_source=CMD_PREFIX, ur_line=np.full(16, np.nan),
                  ur_fresh_exact=False, ur_own_delta=np.full(6, np.nan))
        frames.append(fr)

    home_f = sim.franka_measured_pose7()
    home_u = pose7_mat(UrTracking.fk(sim.arm_positions()[1]))
    rep = dp.replay_prefix(sim, ctx, dp.PrefixSpec(spec["episode"], ctx.prefix["frac"]),
                           home_f, home_u, on_frame=on_frame).to_dict()
    rep.update(key=spec["key"], tol_mm=dp.TOL_MM, tol_deg=dp.TOL_DEG)
    if ctx.prefix["offset_to_node"]:
        rep["node_offset_frames"] = _node_offset(ctx, rep["end_frame"])
    if not rep["ok"]:
        finish_recording(sim, ctx.out, name, f"prefix failed: {rep['reason']}")
        logger.warning(f"[EVAL] {name} skipped: prefix failed ({rep['reason']})")
    return {"report": rep, "frames": frames, "sampler": sampler,
            "wall_s": time.perf_counter() - w0}


def _node_offset(ctx: EvalContext, end_frame: int) -> int:
    """The prefix end on the node's demo clock: the same frac on the waypoint artifact's demo
    (``--mppi-waypoints``, key ``demo_episode``) when there is one, else ``end_frame``."""
    wp = getattr(ctx.args, "mppi_waypoints", None)
    if wp and Path(wp).is_file():
        with np.load(wp, allow_pickle=True) as d:
            if "demo_episode" in d.files:
                return ctx.prefix["module"].prefix_end_frame(str(d["demo_episode"]),
                                                             ctx.prefix["frac"])
    return int(end_frame)


def _prefix_options(args: argparse.Namespace, states: list[StartState],
                    start_meta: dict) -> dict | None:
    """``--prefix-frac > 0``: ``{module, frac, demos (state id -> demo), offset_to_node}``;
    None when off (the default)."""
    frac = float(getattr(args, "prefix_frac", 0.0) or 0.0)
    offset = getattr(args, "prefix_offset_to_node", None)
    if not 0.0 <= frac <= 2.0:
        raise ValueError(f"--prefix-frac {frac} outside [0, 2]")
    if frac == 0.0:
        if offset:
            raise ValueError("--prefix-offset-to-node needs --prefix-frac > 0")
        return None
    if not getattr(args, "prefix_demos", None):
        raise ValueError("--prefix-frac > 0 needs --prefix-demos")
    import demo_prefix as dp

    src = Path(args.prefix_demos)
    demos = json.loads(src.read_text() if src.suffix == ".json" and src.is_file()
                       else args.prefix_demos)
    set_name = start_meta.get("set") if start_meta["kind"] == "variant_set" else None
    by_state = {}
    for s in states:
        key = f"{set_name}/{s.id}" if set_name else s.id
        if key not in demos:
            raise ValueError(f"--prefix-demos has no demo for start {key!r} ({sorted(demos)})")
        path = _resolve(REPO_ROOT, demos[key])
        if not path.is_file():
            raise FileNotFoundError(f"prefix demo {path}")
        d = dp.load_demo(path)
        marks = {k: d[k] for k in ("move0", "latch", "first_hold")}
        by_state[s.id] = {"key": key, "episode": str(path.resolve()), "sha256": sha256_file(path),
                          "end_frame": dp.frac_frame(marks, frac), **marks}
    auto = (getattr(args, "mppi_goal", None) == "waypoints"
            or bool(getattr(args, "mppi_demo_actions", None)))
    offset = auto if offset is None else bool(offset)
    if offset and getattr(args, "controller", "magna") != "mppi":
        raise ValueError("--prefix-offset-to-node needs --controller mppi")
    return {"module": dp, "frac": frac, "demos": by_state, "offset_to_node": offset}


def _prefix_index_block(prefix: dict) -> dict:
    dp = prefix["module"]
    return {"frac": prefix["frac"], "demos": prefix["demos"],
            "offset_to_node": prefix["offset_to_node"], "phase_label": dp.PHASE_LABEL,
            "cmd_source": CMD_PREFIX, "tol_mm": dp.TOL_MM, "tol_deg": dp.TOL_DEG,
            "settle": {"tol_mm": dp.SETTLE_TOL_MM, "tol_deg": dp.SETTLE_TOL_DEG,
                       "max_s": dp.SETTLE_MAX_S},
            "frac_rule": "0..1 -> [first move:* frame, pre_place_2 latch], 1..2 -> [latch, "
                         "first hold:place_3]; demo prehold skipped"}


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
    es = [float(x) for x in args.early_stop.split(",")] if args.early_stop else None
    es_lab, es_t0 = None, 0.0
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
            if es is not None and not finished and abort is None and end_at is None:
                lab = classify(frame_metrics(frames[-1]["belt"], frames[-1]["pulley_raw"], th=ctx.thresholds),
                               ctx.thresholds)
                t = k * dt
                if lab != es_lab:
                    es_lab, es_t0 = lab, t
                if t >= es[2] and t - es_t0 >= (es[0] if lab == "engaged" else es[1]):
                    reason, finished = f"early stop: {lab} held {t - es_t0:.1f} s at {t:.1f} s", True
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
            elif not ctrl.alive() and not (ctx.mppi is not None and end_at is not None):
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
        if ctx.cloud is not None:
            c = ctx.cloud
            out = bc.belt_seg_frame(xyz, rgb, c["noise"], c["seg"] or bc.DEFAULT_SEG,
                                    ctx.cloud_rng, c["origins"],
                                    bc.jaw_centres_from_poses(ee_f, ee_u), c["pose"])
            xyz, rgb = out["pcd"], out["pcd_rgb"]
            if c["seg"] is not None:
                frame["pcd_belt_seg"] = (out["pcd_belt_seg"] if ctx.args.belt_seg_source == "camera"
                                         else bc.to_pulley_frame(frame["belt_pts"], c["pose"])
                                         .astype(np.float32))
        frame["cloud"], frame["pcd_rgb"] = lcs.camera_points(xyz), rgb
    timing["render"] += time.perf_counter() - r0
    frame["latent"] = np.full(ctx.encoder.latent_dim if ctx.encoder else 16, np.nan)
    if ctx.mode == "learned":
        e0 = time.perf_counter()
        z = ctx.encoder.encode(encoder_input(ctx.encoder, frame), frame["prop"], frame["belt_pts"])
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
    if getattr(args, "controller", "magna") == "mppi":
        return _mppi_learned_meta(args, out, demo)
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


def _demo_episode_path(demo: DemoGoals | None) -> Path | None:
    if demo is None:
        return None
    episode = Path(demo.demo_episode)
    return episode if episode.is_absolute() else REPO_ROOT / episode


def _mppi_learned_meta(args: argparse.Namespace, out: dict, demo: DemoGoals | None) -> dict:
    """``learned_mpc`` meta of an mppi run: no C++ params/lcs yaml; ``demo_traj_yaml`` None
    (= staged run for the replay viewer)."""
    for k, path in (("deploy", args.deploy), ("demo_goals", args.demo_goals),
                    ("demo_episode", _demo_episode_path(demo)),
                    ("mppi_config", args.mppi_config)):
        out[k], out[f"{k}_sha256"] = _file_ref(path)
    out["action_definition"] = args.action_definition
    out["debug_channel"] = LcmChannels().learned_mpc_debug_channel if args.mppi_debug else None
    out["controller"] = "mppi"
    return out


def _mppi_options(args: argparse.Namespace, demo_exists: bool) -> dict | None:
    """``--controller mppi``: node options + the config; None for magna (mppi flags at default)."""
    clear = {k: getattr(args, k, v) for k, v in CLEARANCE_ARG_DEFAULTS.items()}
    if getattr(args, "controller", "magna") != "mppi":
        bad = [k for k, v in {**MPPI_ARG_DEFAULTS, **CLEARANCE_ARG_DEFAULTS,
                              **TRANSPORT_ARG_DEFAULTS, **BANK_ARG_DEFAULTS}.items()
               if getattr(args, k, v) != v]
        if bad:
            raise ValueError(f"--{', --'.join(k.replace('_', '-') for k in bad)}: only with "
                             "--controller mppi")
        return None
    if not clear["mppi_clearance"] and clear != CLEARANCE_ARG_DEFAULTS:
        raise ValueError("--mppi-clearance-*-min-mm: only with --mppi-clearance")
    if args.mode != "learned":
        raise ValueError("--controller mppi needs --mode learned")
    if args.params is not None:
        logger.warning(f"[EVAL] --params {args.params} ignored with --controller mppi")
    config = Path(args.mppi_config)
    if not config.is_file():
        raise FileNotFoundError(config)
    kind = args.mppi_goal or ("demo_goals" if demo_exists else "deploy")
    if kind == "demo_goals" and not demo_exists:
        raise ValueError(f"--mppi-goal demo_goals but no file at {args.demo_goals}")
    transport = _mppi_transport(args, kind)
    goal = ("deploy" if kind == "deploy" else f"waypoints:{transport['waypoints']}"
            if kind == "waypoints" else f"demo_goals:{Path(args.demo_goals).resolve()}")
    debug = LcmChannels().learned_mpc_debug_channel if args.mppi_debug else None
    out = {"config": config.resolve(), "config_sha256": sha256(config),
           "params": yaml.safe_load(config.read_text()) or {}, "goal_kind": kind,
           "goal": goal, "stages": args.mppi_stages,
           "max_duration_s": float(args.mppi_max_duration_s), "debug_channel": debug,
           "clearance": None, "transport": transport}
    bank = _mppi_bank(args, kind)
    if bank is not None:
        out["bank"] = bank
    return out


def _mppi_bank(args: argparse.Namespace, kind: str) -> dict | None:
    """``--mppi-waypoint-bank/-bank-knn``; None at default."""
    b = {k: getattr(args, k, v) for k, v in BANK_ARG_DEFAULTS.items()}
    if b == BANK_ARG_DEFAULTS:
        return None
    if b["mppi_waypoint_bank"] is None or kind != "waypoints":
        raise ValueError("--mppi-waypoint-bank PATH needs --mppi-goal waypoints (the base)")
    if int(b["mppi_bank_knn"]) < 1:
        raise ValueError("--mppi-bank-knn >= 1")
    path = Path(b["mppi_waypoint_bank"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    doc = json.loads(path.read_text())
    return {"path": str(path), "sha256": sha256(path), "knn": int(b["mppi_bank_knn"]),
            "n_entries": len(doc["entries"]), "deploy_sha256": doc["deploy_sha256"],
            "max_waypoints": max(len(e["frames"]) for e in doc["entries"])}


def _mppi_transport(args: argparse.Namespace, kind: str) -> dict | None:
    """``--mppi-waypoints/-goal-mode/-demo-actions/-demo-offset-frames``; None at default."""
    t = {k: getattr(args, k, v) for k, v in TRANSPORT_ARG_DEFAULTS.items()}
    if (kind == "waypoints") != (t["mppi_waypoints"] is not None):
        raise ValueError("--mppi-goal waypoints and --mppi-waypoints PATH go together")
    if t == TRANSPORT_ARG_DEFAULTS:
        return None
    if t["mppi_goal_mode"] == "track" and kind != "waypoints":
        raise ValueError("--mppi-goal-mode track needs --mppi-goal waypoints")
    if t["mppi_demo_actions"] == "auto" and kind != "waypoints":
        raise ValueError("--mppi-demo-actions auto needs --mppi-goal waypoints")
    out = {"waypoints": None, "waypoints_sha256": None, "goal_mode": t["mppi_goal_mode"],
           "demo_actions": t["mppi_demo_actions"], "demo_actions_sha256": None,
           "demo_offset_frames": int(t["mppi_demo_offset_frames"])}
    if t["mppi_waypoints"] is not None:
        wp = Path(t["mppi_waypoints"]).resolve()
        if not wp.is_file():
            raise FileNotFoundError(wp)
        out.update(waypoints=str(wp), waypoints_sha256=sha256(wp))
    if t["mppi_demo_actions"] not in (None, "auto"):
        da = Path(t["mppi_demo_actions"]).resolve()
        if not da.is_file():
            raise FileNotFoundError(da)
        out.update(demo_actions=str(da), demo_actions_sha256=sha256(da))
    return out


def _export_clearance(sim, args: argparse.Namespace, out: Path, start: Path) -> dict:
    """``--mppi-clearance``: ``<out>/clearance_model.npz`` at the current (first start) state."""
    model = cm.export_from_sim(sim, args.mppi_clearance_ur_min_mm * 1e-3,
                               args.mppi_clearance_franka_min_mm * 1e-3)
    model.source.update(start_state=str(Path(start).resolve()),
                        start_state_sha256=sha256_file(start))
    path = model.save(out / "clearance_model.npz").resolve()
    logger.info(f"[EVAL] clearance model {path}: UR {len(model.ur_local)} / Franka "
                f"{len(model.franka_local)} points ({model.source['franka']}), min "
                f"{model.ur_min_m * 1e3:g} / {model.franka_min_m * 1e3:g} mm")
    return {"path": str(path), "sha256": sha256_file(path), "ur_min_m": model.ur_min_m,
            "franka_min_m": model.franka_min_m,
            "n_points": {"ur": len(model.ur_local), "franka": len(model.franka_local)}}


def _start_mppi(ctrl: MppiControllerProcess, ctx: EvalContext, log_path: Path,
                demo_offset_frames: int | None = None) -> None:
    """Node offset = ``demo_offset_frames`` if given, else ``ctx.args.mppi_demo_offset_frames``
    read now (a prefix replay may set it per episode)."""
    m = ctx.mppi
    extra = ["--log", str(_mppi_solve_log(log_path).resolve())]
    if m["debug_channel"]:
        extra += ["--debug-channel", m["debug_channel"]]
    if m.get("clearance"):
        extra += ["--clearance-model", m["clearance"]["path"]]
    offset = int(getattr(ctx.args, "mppi_demo_offset_frames", 0) if demo_offset_frames is None
                 else demo_offset_frames)
    tr = m.get("transport")
    if tr is not None:
        extra += ["--goal-mode", tr["goal_mode"], "--demo-offset-frames", str(offset)]
        if tr["demo_actions"] is not None:
            extra += ["--demo-actions", tr["demo_actions"]]
    elif offset:
        extra += ["--demo-offset-frames", str(offset)]
    if m.get("bank"):
        extra += ["--waypoint-bank", m["bank"]["path"], "--bank-knn", str(m["bank"]["knn"])]
    ctrl.start(ctx.args.lcm_url, ctx.args.deploy, m["config"], log_path, goal=m["goal"],
               stages=m["stages"], max_duration_s=m["max_duration_s"], extra_args=extra)


def _mppi_solve_log(log_path: Path) -> Path:
    return log_path.with_suffix(".solves.jsonl")


def _mppi_solve_stats(out: Path, log_path: Path) -> dict:
    """Per-episode summary of the node's per-solve JSONL."""
    path = _mppi_solve_log(log_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()] if path.is_file() else []
    ms = np.array([r["solve_ms"] for r in rows if r.get("solve_ms") is not None
                   and np.isfinite(r["solve_ms"])], dtype=np.float64)
    pct = {f"p{q}": float(np.percentile(ms, q)) if ms.size else None for q in (50, 95)}
    stats = {"solve_log": str(path.relative_to(out)), "solves": len(rows),
             "solve_ms": {**pct, "max": float(ms.max()) if ms.size else None,
                          "mean": float(ms.mean()) if ms.size else None},
             "solve_failed": sum(r.get("solve_ok", 1.0) == 0.0 for r in rows),
             "fallbacks": sum(bool(r.get("fallback")) for r in rows),
             "plans_violating": sum(int(r.get("plan_violations") or 0) > 0 for r in rows)}
    clear = [r for r in rows if "clear_min_ur_mm" in r]
    if clear:
        stats["clear_min_mm"] = {"ur": min(r["clear_min_ur_mm"] for r in clear),
                                 "franka": min(r["clear_min_f_mm"] for r in clear)}
        stats["clear_violating_plans"] = sum(int(r["clear_violating"]) > 0 for r in clear)
    bank = next((r["bank"] for r in rows if "bank" in r), None)
    if bank is not None:
        stats["bank"] = bank
    return stats


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


def _reset_options(args: argparse.Namespace) -> dict | None:
    """``--reset scripted``: ``{module, spec, rng, homes}``; None for the snapshot default."""
    pairs = getattr(args, "reset_spec", None) or []
    if getattr(args, "reset", "snapshot") != "scripted":
        if pairs:
            raise ValueError("--reset-spec needs --reset scripted")
        return None
    import scripted_reset

    spec = scripted_reset.parse_reset_spec(pairs)
    return {"module": scripted_reset, "spec": spec, "rng": np.random.default_rng(spec.seed),
            "homes": {}}


def _reset_before(sim, ctx: EvalContext, state: StartState, rows: list[dict]
                  ) -> tuple[dict, dict | None]:
    """``(reset_before row, from_reset)``: scripted only after a held episode of the same start
    (its first-frame poses = home); else the episode's own snapshot restore (human)."""
    sr = ctx.reset["module"]
    prev = rows[-1] if rows else None
    home = ctx.reset["homes"].get(state.id)
    reason = None
    if prev is None:
        reason = "initial start state"
    elif prev["start_state"]["id"] != state.id:
        reason = "start state changed"
    elif prev.get("status") == "skipped" or prev.get("grasp_ok_final") != [True, True]:
        reason = "grasp not held at the end of the previous episode"
    elif home is None:
        reason = "no home poses for this start state"
    if reason is not None:
        return sr.ResetReport.human(reason).to_dict(), None
    try:
        rep = sr.scripted_reset(sim, ctx, home[0], home[1], ctx.reset["spec"], ctx.reset["rng"])
    except OscTimeout as exc:
        if not exc.process_alive:
            raise
        sim.bridge.forget_pending()
        rep = sr.ResetReport(ok=False, kind="scripted", reason=f"osc timeout: {exc}",
                             grasp_held=None, duration_s=0.0)
    if not rep.ok:
        return sr.ResetReport.human(f"scripted reset failed: {rep.reason}", rep).to_dict(), None
    return rep.to_dict(), rep.to_dict()


def _reset_budget(rows: list[dict]) -> dict:
    resets = [r["reset_before"] for r in rows if r.get("reset_before")]
    return {"resets_scripted": sum(x["kind"] == "scripted" for x in resets),
            "resets_human": sum(x["kind"] == "human" for x in resets),
            "reset_s": float(sum(x["duration_s"] for x in resets))}


def _run_budget(ctx: EvalContext, rows: list[dict]) -> dict:
    """``index.budget``: the scripted-reset counts; with a prefix also its time and failures
    and ``robot_s`` (episodes + 0.5 s settle each + prefixes + scripted resets)."""
    b = _reset_budget(rows) if ctx.reset is not None else {}
    if ctx.prefix is not None:
        pre = [r["prefix"] for r in rows if r.get("prefix")]
        b.update(prefix_s=float(sum(p["duration_s"] for p in pre)),
                 prefix_failed=sum(not p["ok"] for p in pre))
        b["robot_s"] = float(sum(r.get("duration_s", 0.0) + START_SETTLE_S for r in rows)
                             + b["prefix_s"] + b.get("reset_s", 0.0))
    return b


def evaluate(args: argparse.Namespace) -> dict:
    """Run the evaluation; returns the ``index.json`` dict."""
    mode = args.mode
    check_private_url(args.lcm_url)
    if args.no_pcd and mode == "learned":
        raise ValueError("--no-pcd is baseline only (the encoder needs the cloud)")
    cloud = cli.cloud_options(args)
    reset = _reset_options(args)
    mppi = _mppi_options(args, args.demo_goals is not None and Path(args.demo_goals).is_file())
    if mppi is None:
        params_rel = Path(args.params or DEFAULT_PARAMS[mode])
        params = _controller_yaml(args.magna_root, params_rel)
        has_learned = params.get("learned_mpc") is not None
        if has_learned != (mode == "learned"):
            raise ValueError(f"--mode {mode} but {params_rel} "
                             f"{'has' if has_learned else 'lacks'} a learned_mpc block")
    else:
        params_rel, params = None, {}
    pre_labels = [str(w.get("label", f"target{i}"))
                  for i, w in enumerate(params.get("pre_mpc_motion") or [])]
    states, start_meta = load_start_states(args.start_states, args.variants)
    plan = plan_episodes(states, args.repeats, args.seed)
    prefix = _prefix_options(args, states, start_meta)
    if args.dry_run:
        for i, (s, r) in enumerate(plan):
            print(f"{i:4d}  {s.id:<24} repeat {r}  {s.file}")
        return {"episodes": [{"start_state": s.to_dict(), "repeat": r} for s, r in plan]}

    encoder = lcs_model = None
    deploy_meta = None
    if mode == "learned":
        encoder, lcs_model = LatentEncoder.load(args.deploy), LearnedLcs.load(args.deploy)
        if isinstance(encoder, BeltCloudPointNetEncoder) and not args.belt_seg:
            raise ValueError(f"{args.deploy}: a belt_cloud_pointnet deploy encodes the segmented "
                             "belt cloud; add --belt-seg")
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
    if mppi is not None:
        n_goals = demo.n_stages if mppi["goal_kind"] == "demo_goals" else n_stages
        tr = mppi["transport"]
        if mppi["goal_kind"] == "waypoints":
            with np.load(tr["waypoints"], allow_pickle=False) as f:
                n_goals = len(f["frames"])
            if mppi.get("bank"):  # stage labels cover the longest bank entry
                n_goals = max(n_goals, mppi["bank"]["max_waypoints"])
        track = tr is not None and tr["goal_mode"] == "track"
        n_stages = 1 if mppi["stages"] == "last" or track else n_goals
        stage_labels = (["stage0"] if mppi["stages"] == "last" or track else
                        demo.stage_labels if demo is not None and demo.n_stages == n_goals
                        else [f"stage{k}" for k in range(n_goals)])
    phase_labels = ([f"pre:{lab}" for lab in pre_labels]
                    + [f"mpc:{lab}" for lab in stage_labels] + ["done"])
    if prefix is not None:
        phase_labels = [prefix["module"].PHASE_LABEL, *phase_labels]

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
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                 if not (k in CLOUD_ARG_DEFAULTS and v == CLOUD_ARG_DEFAULTS[k])},
        "git": git_info(), "mode": mode, "start_states": start_meta,
        "controller": ({"mode": mode, "magna_root": str(args.magna_root),
                        "params": str(params_rel),
                        "params_sha256": sha256(args.magna_root / params_rel)}
                       if mppi is None else _mppi_index_block(mppi)),
        "deploy": deploy_meta, "demo_goals": demo_meta,
        "sample_period_s": SAMPLE_STEPS * lcs.SIM_DT_S, "sample_steps": SAMPLE_STEPS,
        "action_definition": args.action_definition, "belt_sampling": lcs.BELT_SAMPLING,
        "thresholds": dataclasses.asdict(thresholds), "labels": list(LABELS),
        "phase_labels": phase_labels, "belt_tangent": tangent, "episodes": [],
    }
    if reset is None:
        for key in RESET_ARG_DEFAULTS:
            index["args"].pop(key, None)
    else:
        index["reset"] = {"mode": "scripted", "spec": dataclasses.asdict(reset["spec"])}
    if mppi is None:
        for key in MPPI_ARG_DEFAULTS:
            index["args"].pop(key, None)
    for key, value in {**CLEARANCE_ARG_DEFAULTS, **TRANSPORT_ARG_DEFAULTS,
                       **BANK_ARG_DEFAULTS}.items():
        if index["args"].get(key) == value:
            index["args"].pop(key)
    for key, value in PREFIX_ARG_DEFAULTS.items():
        if index["args"].get(key) == value:
            index["args"].pop(key)
    if prefix is not None:
        index["prefix"] = _prefix_index_block(prefix)
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
        if mppi is not None and args.mppi_clearance:
            mppi["clearance"] = _export_clearance(sim, args, out, plan[0][0].file)
            index["controller"]["clearance"] = mppi["clearance"]
            write_json(index_path, index)
        ctx = EvalContext(args=args, out=out, mode=mode, params_rel=params_rel,
                          thresholds=thresholds, tangent=tangent, board=board, gripper=gripper,
                          encoder=encoder, lcs_model=lcs_model, demo=demo,
                          phase_labels=phase_labels, n_pre=len(pre_labels), n_stages=n_stages,
                          stage_labels=stage_labels,
                          learned_mpc_meta=learned_mpc_meta(args, mode, params_rel, params, demo),
                          mppi=mppi)
        if cloud is not None:
            ctx.cloud = {**cloud, "origins": cli.sim_camera_origins(sim)}
            index["cloud_noise_spec"] = cloud["noise"].to_dict()
        ctx.reset = reset
        ctx.prefix = prefix
        for s, r in plan:
            reset_row = from_reset = None
            try:
                if ctx.reset is not None:
                    reset_row, from_reset = _reset_before(sim, ctx, s, rows)
                row = run_episode(sim, ctx, s, r, from_reset)
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
            if reset_row is not None:
                row["reset_before"] = reset_row
            rows.append(row)
            if ctx.reset is not None or ctx.prefix is not None:
                index["budget"] = _run_budget(ctx, rows)
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


def _mppi_index_block(m: dict) -> dict:
    return {"mode": "mppi", "node": str(MppiControllerProcess().node),
            "config": str(m["config"]), "config_sha256": m["config_sha256"],
            "mppi_params": m["params"], "goal": m["goal"], "stages": m["stages"],
            "max_duration_s": m["max_duration_s"], "debug_channel": m["debug_channel"],
            **({} if m.get("transport") is None else {"transport": m["transport"]}),
            **({"bank": m["bank"]} if m.get("bank") else {})}


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
