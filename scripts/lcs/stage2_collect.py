#!/usr/bin/env python3
"""Stage-2 data collector (V12-93): randomised scripted Franka wraps with out-and-back loops.

Per episode: restore a stage-2 start (``stage2_starts.py``: UR released and parked at place_9, the
Franka holding the belt), settle, and walk the D10L plan with parameters drawn from its success
envelope (``PARAM_RANGES``): transit over the small pulley (late yaw slerp, tilt cap) to the
place_10 hover, descent to place_11, wrap over target_1..3, arc to the end azimuth. DART noise:
zero-mean Gaussian offsets on the Franka command per knot; the recorded action is the noisy
``cmd_delta``. Out-and-back loops (min-jerk out / back / hold, miles_seed ``CyclePath``) only at
the hover, at target_2 (one) and at the arc end, in per-phase boxes (plan 4.3, 50 % of the
envelope) x a multiplier halved on every trip and persisted in ``budget.json``. Online trip
(real-available signals): Franka hand-width delta from the loop start, belt-cloud h / wrap at the
large pulley, belt-cloud point count at the small pulley (wrap / arc / end). A trip returns to the
plan pose; unless the signals recover the walk ends there. Then the scripted release (open, out,
lift) in its own phases. Sim GT (both pulleys) labels for curation and the report only.

Writes ``<out>/episodes/*.npz`` (one per episode; ``cmd_delta``, ``sim_cmd_source`` 5),
``buffer.json`` (one ``stage2_demo`` row per episode, loops as segments), ``budget.json``,
``index.json``, ``report.md``.

Run (stage-2 scene, one sim on the GPU):
    ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_stage2.yaml \\
      flock data/lcs/realfirst/cmp/v12_79/sim.lock uv run --frozen python scripts/lcs/stage2_collect.py \\
      --start-ids V45c_nominal_0,V48b_up+20_0 --out data/lcs/realfirst/cmp/v12_93/smoke
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fnmatch
import json
import math
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_lcs_dataset as cli
import eval_learned_mpc as harness
import miles_seed as ms
import phase2_probe as pp
from loguru import logger
from make_ee_offset_starts import min_jerk
from scipy.spatial.transform import Rotation
from scripted_reset import MinJerkPath
from stage2_starts import STAGE2_SCENE

from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import UrTracking, mat4
from round_belt_task.commander import CommanderParams, mat3_to_quat, quat_to_mat3, slerp
from round_belt_task.episode_io import finish_recording, git_info, pose7_mat, write_json
from round_belt_task.outcome import (
    DEFAULT_THRESHOLDS,
    SMALL_SEAT_MM,
    classify,
    classify_episode,
    frame_metrics,
    slant_episode,
    success_both,
)
from round_belt_task.waypoints import (
    MAGNA_PARAMS_SIM_YAML,
    X_W_BOARD_PLANNING,
    load_params,
    load_pre_mpc_segment,
)
from task_common import belt_cloud as bc
from task_common import belt_cloud_features as bcf
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.replay_metrics import _largest_arc_deg as largest_arc_deg
from task_common.replay_metrics import wrap_merge_deg

DEFAULT_STARTS = REPO_ROOT / "data" / "lcs" / "start_states" / "stage2"
DEFAULT_LCM_URL = "udpm://239.255.76.191:7795?ttl=0"
PHASES = ("transit", "hover", "descent", "wrap", "arc", "end", "release", "out", "lift", "done")
POST_PHASES = ("release", "out", "lift", "done")
SEGMENTS = (*ms.SEGMENTS, "post")
CMD_STAGE2 = 5  # sim_cmd_source of every stage-2 scripted frame (MILES 4)
KNOT_DT = CommanderParams().dt

# D10L plan in the small-pulley frame (r mm, azimuth deg, h mm over the seat plane)
TRANSIT_RAZ = ((70.0, 150.0), (35.0, 90.0), (36.0, -69.0))
ARC_R_MM, ARC_AZ_DEG, END_R_MM = 22.5, (85.0, 115.0), 25.0
ARC_H0_MM = -4.1  # arc height before the raise
OUT_R_MM, OUT_SPEED_MPS, LIFT_H_MM, OPEN_MM = 45.0, 0.02, 27.0, 40.0
MIN_SEG_S = 1.0
HOVER_HOLD_S, END_HOLD_S, RELEASE_S, POST_HOLD_S, DONE_S = 0.3, 1.0, 1.0, 0.3, 1.5
# Half-width ranges around NOMINAL: the full ranges gave 2/10 both-engaged vs nominal 4-7/10 (V12-93 A/B).
PARAM_RANGES = {
    "hover_h_mm": (27.5, 35.0),
    "slerp_f1": (0.075, 0.125), "slerp_f2": (0.175, 0.225), "slerp_f3": (0.375, 0.425),
    "tilt_cap_deg": (7.5, 10.0),
    "wrap_dh_mm": (-0.5, 1.0),
    "arc_raise_mm": (2.5, 3.5),
    "end_az_deg": (138.0, 144.0),
    "speed_mps": (0.0425, 0.055),
}
NOMINAL = {"hover_h_mm": 30.0, "slerp_f1": 0.1, "slerp_f2": 0.2, "slerp_f3": 0.4,
           "tilt_cap_deg": 10.0, "wrap_dh_mm": 0.0, "arc_raise_mm": 3.0, "end_az_deg": 141.0,
           "speed_mps": 0.05}
DART = {"transit": (1.0, 0.5), "descent": (1.0, 0.5), "wrap": (1.0, 0.5), "arc": (1.0, 0.5),
        "hover": (2.0, 1.0), "end": (2.0, 1.0)}  # sigma per knot: mm, deg
# Loop boxes at multiplier 1 = plan 4.3 (<= 50 % of the sim envelope); tilt about a horizontal axis.
BOXES = {"hover": {"mm": 8.0, "z_mm": 8.0, "yaw_deg": 3.0, "tilt_deg": 3.0, "away_mm": 8.0},
         "wrap": {"mm": 3.0, "z_mm": 3.0, "yaw_deg": 2.0, "tilt_deg": 0.0},
         "end": {"mm": 8.0, "z_mm": 5.0, "yaw_deg": 5.0, "tilt_deg": 5.0}}
LOOPS = {"hover": (2, 3), "wrap": (1, 1), "end": (3, 3)}  # loops per episode: min, max
LOOP_WP = {"hover": 3, "wrap": 6, "end": 10}
MIN_MULT = 0.125
SMALL_SPEC = bcf.CloudFeatureSpec(seat_mm=SMALL_SEAT_MM)


@dataclasses.dataclass(frozen=True)
class TripSpec:
    hand_lo_mm: float = -1.0
    hand_hi_mm: float = 2.0
    large_h_mm: float = 3.0
    large_wrap_drop_deg: float = 15.0
    small_drop_frac: float = 0.5
    small_min_pts: int = 10  # a thinner loop-start count is not compared
    small_phases: tuple[str, ...] = ("wrap", "arc", "end")


TRIP = TripSpec()


# --- geometry + plan --------------------------------------------------------------------------

def small_to_world(r_mm: float, az_deg: float, h_mm: float) -> np.ndarray:
    s = bc.known_small_pulley_pose7()
    a = math.radians(az_deg)
    return s[:3] + quat_to_mat3(s[3:]) @ (np.array([r_mm * math.cos(a), r_mm * math.sin(a),
                                                     h_mm]) * 1e-3)


def small_polar(xyz) -> tuple[float, float, float]:
    """``(r mm, azimuth deg, h mm)`` of a world point over the small pulley."""
    v = bc.to_pulley_frame(xyz, bc.known_small_pulley_pose7())[0] * 1e3
    return float(math.hypot(v[0], v[1])), float(math.degrees(math.atan2(v[1], v[0]))), float(v[2])


def tilt_deg(q) -> float:
    return math.degrees(math.acos(min(1.0, abs(quat_to_mat3(q)[2, 2]))))


def heading_deg(q) -> float:
    """World azimuth of the EE y (closing) axis."""
    R = quat_to_mat3(q)
    return math.degrees(math.atan2(R[1, 1], R[0, 1]))


def cap_tilt(q, cap_deg: float) -> np.ndarray:
    """The rotation from the level frame with the same closing-axis heading scaled so the EE z
    tilts ``cap_deg`` from vertical (the heading moves < 1 deg)."""
    q = np.asarray(q, np.float64) / np.linalg.norm(q)
    R = quat_to_mat3(q)
    tilt = tilt_deg(q)
    if tilt <= cap_deg:
        return q
    yaw = math.radians(heading_deg(q))
    yv = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    V = np.column_stack([np.cross(yv, [0.0, 0.0, 1.0]), yv, [0.0, 0.0, 1.0]])
    D = Rotation.from_matrix(V.T @ R).as_rotvec()
    return mat3_to_quat(V @ Rotation.from_rotvec(D * cap_deg / tilt).as_matrix())


def hw_poses() -> dict:
    """World Franka poses of the hardware waypoints on the scene board (``X_W_BOARD_PLANNING``)."""
    targets = dict(pp.franka_targets("scene")[0])
    prm = load_params()

    def on_board(e: dict) -> np.ndarray:
        return pose7_mat(X_W_BOARD_PLANNING @ mat4(quat_to_mat3(e["franka_orientation"]),
                                                   e["franka_position"]))

    p11 = next(e for e in prm["pre_mpc_motion"] if e.get("label") == "place_11")
    # the sim yaml's place_9 is the hardware place_10
    return {"P10": targets["place_9"], "P11": on_board(p11),
            "T": [targets[f"target_{i}"] for i in (1, 2, 3)],
            "PO1": on_board(prm["osc_target_poses_post_mpc"][0])}


def shift_h(p7, dh_mm: float) -> np.ndarray:
    r, az, h = small_polar(p7[:3])
    return np.r_[small_to_world(r, az, h + dh_mm), p7[3:]]


def draw_params(rng: np.random.Generator) -> dict:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in PARAM_RANGES.items()}


def build_plan(prm: dict, ee0, hw: dict) -> list[tuple[str, str, np.ndarray]]:
    """``(label, phase, pose7)`` w0..w10: transit w0-w3, descent w4, wrap w5-w7, arc w8-w10."""
    q10 = hw["P10"][3:]
    pos = [small_to_world(r, az, prm["hover_h_mm"]) for r, az in TRANSIT_RAZ] + [hw["P10"][:3]]
    fr = (prm["slerp_f1"], prm["slerp_f2"], prm["slerp_f3"], 1.0)
    out = [(f"w{i}", "transit", np.r_[p, cap_tilt(slerp(ee0[3:], q10, f), prm["tilt_cap_deg"])])
           for i, (p, f) in enumerate(zip(pos, fr, strict=True))]
    out.append(("w4", "descent", shift_h(hw["P11"], prm["wrap_dh_mm"])))
    out += [(f"w{5 + i}", "wrap", shift_h(t, prm["wrap_dh_mm"])) for i, t in enumerate(hw["T"])]
    h, q = ARC_H0_MM + prm["arc_raise_mm"], hw["PO1"][3:]
    out += [(f"w{8 + i}", "arc", np.r_[small_to_world(ARC_R_MM, az, h), q])
            for i, az in enumerate(ARC_AZ_DEG)]
    out.append(("w10", "arc", np.r_[small_to_world(END_R_MM, prm["end_az_deg"], h), q]))
    return out


def post_poses(prm: dict, hw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Out (radially to ``OUT_R_MM`` at the arc height) and lift targets after the opening."""
    q, az = hw["PO1"][3:], prm["end_az_deg"]
    return (np.r_[small_to_world(OUT_R_MM, az, ARC_H0_MM + prm["arc_raise_mm"]), q],
            np.r_[small_to_world(OUT_R_MM, az, LIFT_H_MM), q])


def release_targets(ee7) -> tuple[np.ndarray, np.ndarray]:
    """``post_poses`` from a pose (the MPC end): radially out to ``OUT_R_MM`` at its height, then
    up to ``LIFT_H_MM``, orientation kept; never inwards or down."""
    r, az, h = small_polar(ee7[:3])
    q, r = np.asarray(ee7[3:], np.float64), max(OUT_R_MM, r)
    return (np.r_[small_to_world(r, az, h), q],
            np.r_[small_to_world(r, az, max(LIFT_H_MM, h)), q])


def release_plan(ee7, speed: float = NOMINAL["speed_mps"]) -> list[tuple[str, list, float, float]]:
    """The collector's post phases from ``ee7``: ``(phase, targets, speed m/s, hold s)``; the hand
    opens to ``OPEN_MM`` before the first."""
    out7, lift7 = release_targets(ee7)
    return [("release", [], speed, RELEASE_S), ("out", [out7], OUT_SPEED_MPS, POST_HOLD_S),
            ("lift", [lift7], speed, POST_HOLD_S), ("done", [], speed, DONE_S)]


def seg_s(a, b, speed: float) -> float:
    return max(MIN_SEG_S, float(np.linalg.norm(np.asarray(b)[:3] - np.asarray(a)[:3])) / speed)


def walk_path(start7, targets, speed: float, hold_s: float = 0.0) -> MinJerkPath:
    path, prev = MinJerkPath(start7), np.asarray(start7, np.float64)
    for p in targets:
        path.add(seg_s(prev, p, speed), p[:3], p[3:])
        prev = p
    path.add(hold_s, prev[:3], prev[3:])
    return path


# --- command paths ----------------------------------------------------------------------------

class KnotNoise:
    """Zero-mean Gaussian Franka offsets (world mm, world rotvec deg) on a grid of <= ``dt`` over
    ``[0, end]``, zero at both ends so consecutive drives join on the plan; linear between."""

    def __init__(self, rng: np.random.Generator, end: float, sigma_mm: float, sigma_deg: float,
                 dt: float = KNOT_DT) -> None:
        n = max(1, math.ceil(end / dt - 1e-9))
        self.t = np.linspace(0.0, max(end, 1e-6), n + 1)
        self.v = rng.normal(size=(n + 1, 6)) * np.repeat([sigma_mm, sigma_deg], 3)
        self.v[[0, -1]] = 0.0

    def __call__(self, tau: float) -> np.ndarray:
        return np.array([np.interp(tau, self.t, c, left=0.0, right=0.0) for c in self.v.T])


class Fixed:
    """Constant pose path (the parked UR: its actions are exactly 0)."""

    def __init__(self, pose7) -> None:
        self.p, self.end = np.asarray(pose7, np.float64), 0.0

    def __call__(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        return self.p[:3].copy(), self.p[3:].copy()

    def pose7(self, tau: float) -> np.ndarray:
        return self.p.copy()


class Cmd:
    """One drive's Franka command: ``base`` + knot noise. ``trip(tau)`` before ``reroute_until``
    replaces the rest by a min-jerk return from the commanded pose to ``home`` over ``ret_s``."""

    def __init__(self, base, noise: KnotNoise | None = None, seg_of=None, home=None,
                 ret_s: float = 0.0, reroute_until: float = -1.0) -> None:
        self.base, self.noise, self.end = base, noise, base.end
        self.seg_of = seg_of or (lambda tau: "walk")
        self.home, self.ret_s, self.reroute_until = home, ret_s, reroute_until
        self.trip_t, self._from = None, None

    def __call__(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        if self.trip_t is not None and tau >= self.trip_t:
            s = min_jerk((tau - self.trip_t) / self.ret_s)
            a, b = self._from, self.home
            q = slerp(a[3:], b[3:], s)
            return a[:3] + s * (b[:3] - a[:3]), q / np.linalg.norm(q)
        p, q = self.base(tau)
        if self.noise is None:
            return p, q
        x = ms.offset_pose(np.r_[p, q], self.noise(tau))
        return x[:3], x[3:]

    def pose7(self, tau: float) -> np.ndarray:
        return np.concatenate(self(tau))

    def trip(self, tau: float) -> bool:
        """Reroute home (True) while still going out; later the path is already returning."""
        if self.trip_t is not None or self.home is None or tau >= self.reroute_until:
            return False
        self._from, self.trip_t = self.pose7(tau), tau
        return True

    def segment(self, tau: float) -> str:
        if self.trip_t is not None and tau >= self.trip_t:
            return "return" if tau < self.trip_t + self.ret_s else "hold"
        return self.seg_of(tau)


def command_at(cmd: Cmd, tau: float) -> dict:
    """The recorded command fields of ``cmd`` at ``tau`` (``miles_seed.drive``'s knots)."""
    kt = KNOT_DT * np.arange(harness.N_KNOTS)
    kn = np.stack([cmd.pose7(tau + o) for o in kt])
    return {"knots": kn, "knot0": kn[0].copy(), "knot1": kn[1].copy()}


def draw_loop(rng: np.random.Generator, box: dict, emphasis_frac: float,
              away=None) -> np.ndarray:
    """``[dx, dy, dz mm, rx, ry, rz deg]`` (world rotvec) in ``box``; half the draws one of x / y at
    the box edge; ``away`` (horizontal unit vector): component along it capped at ``away_mm``."""
    off = np.zeros(6)
    if rng.random() < emphasis_frac:
        off[int(rng.integers(2))] = box["mm"] * (1.0 if rng.random() < 0.5 else -1.0)
    else:
        off[:2] = rng.uniform(-box["mm"], box["mm"], 2)
        off[2] = rng.uniform(-box["z_mm"], box["z_mm"])
        off[5] = rng.uniform(-box["yaw_deg"], box["yaw_deg"])
        a, m = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(0.0, box["tilt_deg"])
        off[3:5] = m * math.cos(a), m * math.sin(a)
    if away is not None and "away_mm" in box:
        d = float(off[:3] @ away)
        if d > box["away_mm"]:
            off[:3] -= (d - box["away_mm"]) * np.asarray(away)
    return off


# --- signals + trips --------------------------------------------------------------------------

def small_cloud(seg_large, large7) -> dict:
    """Small-pulley features of a large-frame ``pcd_belt_seg``: band h / r (``cloud_features``),
    wrap with the small seat's merge gap, points within ``pp.SEG_NEAR_MM`` of the axis in plane."""
    small7 = bc.known_small_pulley_pose7()
    p = np.asarray(seg_large, np.float64).reshape(-1, 3)
    p = bc.to_pulley_frame(p @ quat_to_mat3(large7[3:]).T + large7[:3], small7)
    f = bcf.cloud_features(p, SMALL_SPEC)
    h, r, az = bcf.polar_mm(p)
    s = SMALL_SPEC
    arc = ((r >= s.seat_mm - s.band_inner_mm) & (r <= s.seat_mm + s.band_outer_mm)
           & (np.abs(h - s.h_bias_mm) <= s.arc_h_tol_mm) & (np.abs(r - s.seat_mm) <= s.arc_r_tol_mm))
    wrap = 0.0
    if f.n_band >= s.min_band_points and arc.any():
        bins = np.unique(np.floor(np.mod(az[arc], 360.0) / s.arc_bin_deg))
        wrap = float(largest_arc_deg((bins + 0.5) * s.arc_bin_deg, wrap_merge_deg(SMALL_SEAT_MM)))
    return {"h_mm": f.h_median_mm, "r_mm": f.r_median_mm, "wrap_deg": wrap, "n_band": f.n_band,
            "n_near": int((r <= pp.SEG_NEAR_MM).sum())}


def signals(fr: dict, large7) -> dict:
    seg = fr.get("pcd_belt_seg")
    seg = np.zeros((0, 3)) if seg is None else seg
    f = bcf.cloud_features(seg)
    return {"hand_mm": float(fr["hand_mm"]), "large_h_mm": f.h_median_mm,
            "large_wrap_deg": f.wrap_deg, "small": small_cloud(seg, large7)}


def trip_signal(ref: dict, cur: dict, phase: str, spec: TripSpec = TRIP) -> str | None:
    """First tripped signal of ``cur`` vs the loop-start ``ref`` (real-available only), else None."""
    d = cur["hand_mm"] - ref["hand_mm"]
    if not spec.hand_lo_mm <= d <= spec.hand_hi_mm:
        return "hand_width"
    h0, h = ref["large_h_mm"], cur["large_h_mm"]
    if math.isfinite(h0) and not (math.isfinite(h) and abs(h - h0) <= spec.large_h_mm):
        return "large_h"
    if ref["large_wrap_deg"] - cur["large_wrap_deg"] > spec.large_wrap_drop_deg:
        return "large_wrap"
    n0 = ref["small"]["n_near"]
    if (phase in spec.small_phases and n0 >= spec.small_min_pts
            and cur["small"]["n_near"] < spec.small_drop_frac * n0):
        return "small_count"
    return None


class Budget:
    """Per-phase loop-box multipliers, halved on a trip (floor ``MIN_MULT``), saved on change."""

    def __init__(self, path: Path, seed_from: Path | None = None) -> None:
        self.path, self.seeded_from = path, None if seed_from is None else str(seed_from)
        self.mult = {p: 1.0 for p in BOXES}
        self.history: list[dict] = []
        if seed_from is not None:
            self.mult.update(json.loads(Path(seed_from).read_text())["multipliers"])
        self.save()

    def box(self, phase: str) -> dict:
        return {k: v * self.mult[phase] for k, v in BOXES[phase].items()}

    def halve(self, phase: str, signal: str, episode: str) -> float:
        before = self.mult[phase]
        self.mult[phase] = max(MIN_MULT, 0.5 * before)
        self.history.append({"episode": episode, "phase": phase, "signal": signal,
                             "before": before, "after": self.mult[phase]})
        self.save()
        return self.mult[phase]

    def save(self) -> None:
        write_json(self.path, {
            "multipliers": self.mult, "boxes_at_1": BOXES, "min_mult": MIN_MULT,
            "rule": ("box = boxes_at_1 x multiplier; a trip at a phase halves its multiplier for "
                     "the rest of the episode and the following episodes"),
            "seeded_from": self.seeded_from, "history": self.history})


# --- collector --------------------------------------------------------------------------------

class EpisodeRec(ms.FileRec):
    """One episode's frames: ``FileRec`` + stage-2 phase, segment, waypoint and cloud signals."""

    def __init__(self, sim, ctx) -> None:
        super().__init__(sim, ctx, lambda tau: 0, lambda tau: -1)
        self.phase, self.wp, self.cmd, self.hand_cmd = 0, -1, None, math.nan

    def __call__(self, k: int, cmd: dict, tau: float) -> None:
        super().__call__(k, cmd, tau)
        fr = self.frames[-1]
        fr.update(cmd_source=CMD_STAGE2, phase=self.phase, segment=self.cmd.segment(tau),
                  waypoint=self.wp, demo_frame=self.wp, trip="", ctrl_hand_mm=self.hand_cmd)
        fr["signals"] = signals(fr, self.ctx.cloud["pose"])


@dataclasses.dataclass
class Episode:
    name: str
    rec: EpisodeRec
    ur: Fixed
    cur: np.ndarray
    clr: ms.Clearance | None = None
    loops: list = dataclasses.field(default_factory=list)
    walk_end: dict | None = None
    aborts: list = dataclasses.field(default_factory=list)
    robot_s: float = 0.0
    tip_min_mm: float = math.inf
    clear_min_mm: float = math.inf


def gt_labels(fr: dict, ctx) -> dict:
    fl = frame_metrics(fr["belt"], fr["pulley_raw"], th=ctx.thresholds)
    fs = frame_metrics(fr["belt"], fr["pulley_small_raw"], SMALL_SEAT_MM, ctx.small_thresholds)
    return {"large": classify(fl, ctx.thresholds), "small": classify(fs, ctx.small_thresholds),
            "both": success_both(fl, fs, ctx.thresholds, ctx.small_thresholds),
            "large_wrap_deg": fl.wrap_deg, "small_wrap_deg": fs.wrap_deg,
            "large_h_mm": fl.h_median_mm, "small_h_mm": fs.h_median_mm,
            "held": bool(fr["grasp_ok"][0])}


class Collector:
    def __init__(self, args, sim, ctx, hw: dict, budget: Budget) -> None:
        self.args, self.sim, self.ctx, self.hw, self.budget = args, sim, ctx, hw, budget
        self.dir = args.out / "episodes"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.episodes: list[dict] = []

    def _noise(self, phase: str, end: float, rng) -> KnotNoise | None:
        if self.args.no_dart or phase not in DART:
            return None
        return KnotNoise(rng, end, *DART[phase])

    def _drive(self, ep: Episode, cmd: Cmd, phase: str, wp: int, loop: dict | None = None,
               arms: str | None = "franka") -> dict:
        sim, n = self.sim, harness.SAMPLE_STEPS
        ep.rec.phase, ep.rec.wp, ep.rec.cmd = PHASES.index(phase), wp, cmd
        steps = n * math.ceil(round(cmd.end / sim.frame_dt) / n)  # whole sample periods

        def on_loop_frame(k, c, tau):
            ep.rec(k, c, tau)
            self._check_trip(ep, cmd, loop, phase, tau)

        s0 = sim.step_index
        rep = ms.drive(sim, self.ctx, cmd, ep.ur, steps, ep.rec if loop is None else on_loop_frame,
                       grasp_arms=arms)
        while (sim.step_index - s0) % n:  # an abort mid-period: keep the episode's sample grid
            sim.control_step()
        ep.robot_s += (sim.step_index - s0) * sim.frame_dt
        ep.tip_min_mm = min(ep.tip_min_mm, rep["tip_min_mm"])
        ep.clear_min_mm = min(ep.clear_min_mm, rep["min_clearance_mm"])
        if rep["abort"]:
            ep.aborts.append({"phase": phase, "waypoint": wp, "kind": rep["abort_kind"],
                              "abort": rep["abort"], "frame": len(ep.rec.frames)})
        return rep

    def _check_trip(self, ep: Episode, cmd: Cmd, loop: dict, phase: str, tau: float) -> None:
        frames = ep.rec.frames
        fr = frames[-1]
        if loop["ref"] is None:
            loop["ref"] = fr["signals"]
            return
        if loop["trip"] is not None:
            return
        why = trip_signal(loop["ref"], fr["signals"], phase)
        if why is None:
            return
        loop.update(trip=why, trip_frame=len(frames) - 1, trip_tau=tau,
                    rerouted=cmd.trip(tau), trip_signals=fr["signals"])
        fr["trip"] = why
        fr.update(command_at(cmd, tau), segment=cmd.segment(tau))

    def _walk(self, ep: Episode, targets, phase: str, wp: int, speed: float, rng,
              hold_s: float = 0.0, arms: str | None = "franka") -> bool:
        path = walk_path(ep.cur, targets, speed, hold_s)
        seg = "post" if phase in POST_PHASES else "walk"
        cmd = Cmd(path, self._noise(phase, path.end, rng), seg_of=lambda tau: seg)
        rep = self._drive(ep, cmd, phase, wp, arms=arms)
        if len(targets):
            ep.cur = np.asarray(targets[-1], np.float64)
        if rep["abort"] and ep.walk_end is None and phase not in POST_PHASES:
            ep.walk_end = {"waypoint": wp, "phase": phase, "why": f"abort:{rep['abort_kind']}"}
        return rep["abort"] is None

    def _loops(self, ep: Episode, phase: str, rng_l, rng_n) -> bool:
        """``phase``'s loops at the current plan pose; False = the walk ends here."""
        args, frames = self.args, ep.rec.frames
        wp, W = LOOP_WP[phase], ep.cur.copy()
        lo, hi = LOOPS[phase]
        away = W[:3] - bc.known_large_pulley_pose7()[:3]
        away[2] = 0.0
        away /= np.linalg.norm(away)
        for li in range(int(rng_l.integers(lo, hi + 1))):
            mult = self.budget.mult[phase]
            off = draw_loop(rng_l, self.budget.box(phase), args.emphasis_frac, away)
            scale, d0, d1 = ep.clr.scale("franka", W, off)
            cyc = ms.CyclePath(W, ms.offset_pose(W, off * scale), args.dev_s, args.ret_s,
                               args.hold_s)
            cmd = Cmd(cyc, self._noise(phase, cyc.end, rng_n),
                      seg_of=lambda tau, c=cyc: ms.SEGMENTS[c.segment(tau)], home=W,
                      ret_s=args.ret_s, reroute_until=cyc.t1)
            loop = {"phase": phase, "waypoint": wp, "loop": li, "mult": mult,
                    "offset_drawn": off.tolist(), "offset": (off * scale).tolist(),
                    "scale": scale, "clearance_model_mm": [d0, d1], "ref": None, "trip": None,
                    "trip_frame": None, "rerouted": False, "recovered": None}
            f0 = len(frames)
            rep = self._drive(ep, cmd, phase, wp, loop=loop)
            loop.update(span=[f0, len(frames)], abort=rep["abort"], abort_kind=rep["abort_kind"])
            if loop["trip"] is not None:
                self.budget.halve(phase, loop["trip"], ep.name)
                loop["recovered"] = trip_signal(loop["ref"], frames[-1]["signals"], phase) is None
            ep.loops.append(loop)
            logger.info(f"[S2] {ep.name} {phase} loop {li}: |d| "
                        f"{np.linalg.norm(off[:3] * scale):.1f} mm x{mult:g}"
                        + (f", TRIP {loop['trip']} (recovered {loop['recovered']})"
                           if loop["trip"] else "")
                        + (f", ABORT {rep['abort']}" if rep["abort"] else ""))
            if rep["abort"]:
                ep.walk_end = {"waypoint": wp, "phase": phase, "why": f"abort:{rep['abort_kind']}"}
                return False
            if loop["trip"] is not None and not loop["recovered"]:
                ep.walk_end = {"waypoint": wp, "phase": phase, "why": f"trip:{loop['trip']}"}
                return False
        return True

    def run(self, start: dict, rep: int, starts_dir: Path) -> dict:
        args, sim = self.args, self.sim
        name = f"{start['id']}_r{rep}"
        key = [args.seed, zlib.crc32(start["id"].encode()), rep]
        rng_p, rng_n, rng_l = (np.random.default_rng([*key, i]) for i in range(3))
        self.ctx.cloud_rng = np.random.default_rng([*key, cli.CLOUD_NOISE_RNG])
        prm = {**(dict(NOMINAL) if args.nominal else draw_params(rng_p)), **args.set}
        row = {"episode": name, "start": {"set": "stage2", "id": start["id"],
                                          "key": start["start"], "arm": start["arm"],
                                          "file": _rel(starts_dir / start["file"])},
               "repeat": rep, "params": prm, "file": None, "skipped": None}
        w0 = time.perf_counter()
        sim.restore(sim_snapshot.load(starts_dir / start["file"]), settle_steps=0)
        if args.record:
            sim.start_recording(args.out / "recordings_tmp", name)
        try:
            self._run(row, prm, name, rng_n, rng_l)
        finally:
            if args.record:
                (args.out / "recordings").mkdir(exist_ok=True)
                row["recording"] = finish_recording(sim, args.out, name, "episode done")
        row["wall_s"] = time.perf_counter() - w0
        self.episodes.append(row)
        return row

    def _run(self, row: dict, prm: dict, name: str, rng_n, rng_l) -> None:
        args, sim, ctx = self.args, self.sim, self.ctx
        grasp = sim.settle(round(args.settle_s / sim.frame_dt))
        if not grasp.held_for("franka"):
            row["skipped"] = f"grasp lost after the restore ({grasp.describe()})"
            return
        ee0 = sim.franka_measured_pose7()
        ep = Episode(name, EpisodeRec(sim, ctx),
                     Fixed(pose7_mat(UrTracking.fk(sim.arm_targets()[1]))), ee0.copy(),
                     ms.Clearance(sim, ctx.plate_top_z) if args.loops else None)
        ep.robot_s += args.settle_s
        plan = build_plan(prm, ee0, self.hw)
        poses = [p for _, _, p in plan]
        v = prm["speed_mps"]
        ok = self._walk(ep, poses[0:4], "transit", 3, v, rng_n)
        ok = ok and self._walk(ep, [], "hover", 3, v, rng_n, HOVER_HOLD_S)
        ok = ok and (not args.loops or self._loops(ep, "hover", rng_l, rng_n))
        ok = ok and self._walk(ep, poses[4:5], "descent", 4, v, rng_n)
        ok = ok and self._walk(ep, poses[5:7], "wrap", 6, v, rng_n)
        ok = ok and (not args.loops or self._loops(ep, "wrap", rng_l, rng_n))
        ok = ok and self._walk(ep, poses[7:8], "wrap", 7, v, rng_n)
        ok = ok and self._walk(ep, poses[8:11], "arc", 10, v, rng_n)
        ok = ok and self._walk(ep, [], "end", 10, v, rng_n, END_HOLD_S)
        ok = ok and (not args.loops or self._loops(ep, "end", rng_l, rng_n))
        frames = ep.rec.frames
        n_task = len(frames)
        mpc_end = gt_labels(frames[-1], ctx) if frames else None
        # scripted release: no grasp abort, no noise, its own phases (training excludes them)
        sim.set_grippers(OPEN_MM, None)
        ep.rec.hand_cmd = OPEN_MM
        out7, lift7 = post_poses(prm, self.hw)
        post_ok = self._walk(ep, [], "release", 10, v, rng_n, RELEASE_S, arms=None)
        post_ok = post_ok and self._walk(ep, [out7], "out", 11, OUT_SPEED_MPS, rng_n,
                                         POST_HOLD_S, arms=None)
        post_ok = post_ok and self._walk(ep, [lift7], "lift", 12, v, rng_n, POST_HOLD_S,
                                         arms=None)
        self._walk(ep, [], "done", 12, v, rng_n, DONE_S, arms=None)
        final = gt_labels(frames[-1], ctx)
        mpc_end = mpc_end or {**final, "held": False}  # no frame before the release
        held = mpc_end["held"]
        row.update(
            plan=[{"label": lab, "phase": ph, "pose": p.tolist(), "polar": small_polar(p[:3])}
                  for lab, ph, p in plan],
            post_targets={"out": out7.tolist(), "lift": lift7.tolist()},
            ee0=ee0.tolist(), loops=_json_loops(ep.loops), walk_end=ep.walk_end, aborts=ep.aborts,
            post_ok=post_ok, mpc_end=mpc_end, final=final, n_frames=len(frames),
            n_task_frames=n_task, post_from=n_task, robot_s=ep.robot_s,
            tip_min_mm=ep.tip_min_mm, clearance_min_mm=ep.clear_min_mm,
            extra_reset=bool(final["large"] != "engaged" or not held),
            label_human=bool(final["both"]))
        row["file"] = write_episode(ctx, self.dir / f"{name}.npz", frames, row)
        logger.info(f"[S2] {name}: {len(frames)} frames ({n_task} task), walk end "
                    f"{ep.walk_end}, MPC end {mpc_end['large']}/{mpc_end['small']}, final "
                    f"{final['large']}/{final['small']} both {final['both']}, "
                    f"{len(ep.loops)} loops, {sum(lp['trip'] is not None for lp in ep.loops)} trips")


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _json_loops(loops: list[dict]) -> list[dict]:
    return json.loads(json.dumps(loops, default=lcs.json_default))


# --- files ------------------------------------------------------------------------------------

def write_episode(ctx, path: Path, frames: list[dict], row: dict) -> str | None:
    """Write + validate the episode npz (``build_writer``, ``cmd_delta``); None below 2 frames."""
    if len(frames) < 2:
        return None
    belt_T = np.stack([fr["belt"] for fr in frames])
    pulley_T = np.stack([fr["pulley_raw"] for fr in frames])
    small_T = np.stack([fr["pulley_small_raw"] for fr in frames])
    _, m = classify_episode(belt_T, pulley_T, ctx.thresholds)
    m.update(slant_episode(belt_T, pulley_T, ctx.tangent, ctx.thresholds))
    _, m_s = classify_episode(belt_T, small_T, ctx.small_thresholds, seat_mm=SMALL_SEAT_MM)
    fin, end = row["final"], row["mpc_end"]
    label = harness._stream_label(ctx, fin["large"], fin["small"], fin["both"])
    c = ctx.cloud
    s2 = {k: row[k] for k in ("episode", "start", "repeat", "params", "plan", "post_targets",
                              "loops", "walk_end", "aborts", "mpc_end", "final", "post_from",
                              "extra_reset")}
    meta = {
        "phase_labels": list(PHASES), "segment_labels": list(SEGMENTS), "mode": "stage2_demo",
        "outcome": label, "status": "ok",
        "reason": None if row["walk_end"] is None else row["walk_end"]["why"],
        "start_state": row["start"], "backend": "osc",
        "ee_pose_source": "measured finger_tip / tracking frame (FK of the measured joints)",
        "action_source": ("knot 1 - knot 0 of the collector's 7-knot Franka "
                          "TARGET_CARTESIAN_POSE_TRAJECTORY on the scripted stage-2 path (DART "
                          "knot noise included) / the parked UR's constant pose (UR channels 0)"),
        "cmd_source_codes": {"stage2_scripted": CMD_STAGE2},
        "latent_source": "none (stage-2 collector)", "time_source": "OSC clock",
        "sample_period_s": harness.SAMPLE_STEPS * lcs.SIM_DT_S,
        "pulley_pose_layout": lcs.POSE_LAYOUT, "start_step": frames[0]["sim_step"],
        "thresholds": dataclasses.asdict(ctx.thresholds),
        "small_thresholds": dataclasses.asdict(ctx.small_thresholds),
        "grippers": (f"Franka at the start command until the release phase ({OPEN_MM:g} mm); "
                     "UR open and parked at place_9"),
        "trip_signals": {**dataclasses.asdict(TRIP), "force": "hardware only (no sim estimate)"},
        "cloud_noise": {**c["noise"].to_dict(), "camera_origins": c["origins"].tolist()},
        "belt_seg": bc.belt_seg_meta(c["noise"], c["seg"], "record", c["pose"]),
        "stage2": s2,
    }
    sm = [fr["signals"]["small"] for fr in frames]
    min_clear = min(row["clearance_min_mm"], min(fr["clearance_mm"] for fr in frames))
    post = {"outcome": np.array(label), "status": np.array("ok"), "mode": np.array("stage2_demo"),
            "start_state": np.array(json.dumps(row["start"])),
            "wrap_deg": m["wrap_deg"], "h_median_mm": m["h_median_mm"],
            **{k: np.array(m[k]) for k in cli.SLANT_KEYS},
            "wrap_small_deg": m_s["wrap_deg"], "h_small_median_mm": m_s["h_median_mm"],
            "outcome_large": np.array(fin["large"]), "outcome_small": np.array(fin["small"]),
            "success_both": np.array(fin["both"]),
            "mpc_end_outcome_large": np.array(end["large"]),
            "mpc_end_outcome_small": np.array(end["small"]),
            "mpc_end_success_both": np.array(end["both"]),
            "pulley_small_pose": np.stack([lcs.pose_from_xyz_xyzw(fr["pulley_small_raw"])
                                           for fr in frames]),
            "cloud_small_h_mm": np.array([s["h_mm"] for s in sm], np.float32),
            "cloud_small_r_mm": np.array([s["r_mm"] for s in sm], np.float32),
            "cloud_small_wrap_deg": np.array([s["wrap_deg"] for s in sm], np.float32),
            "cloud_small_n_band": np.array([s["n_band"] for s in sm], np.int32),
            "cloud_small_n_near": np.array([s["n_near"] for s in sm], np.int32),
            "stage2_segment": np.array([fr["segment"] for fr in frames]),
            "stage2_waypoint": np.array([fr["waypoint"] for fr in frames], np.int16),
            "stage2_trip": np.array([fr["trip"] for fr in frames]),
            "min_board_clearance_mm": np.float32(min_clear),
            "board_contact": np.array(min_clear < 0.0),
            "min_franka_tip_clearance_mm": np.float32(row["tip_min_mm"]),
            "backend": np.array("osc"),
            "osc_utime_offset_us": np.int64(ctx.sim.bridge.utime_offset_us),
            "stage2": np.array(json.dumps(s2, default=lcs.json_default))}
    post = {lcs.EXTRA_PREFIX + k: v for k, v in post.items()}
    writer, _ = harness.build_writer(frames, pcd=True, definition="cmd_delta")
    writer.write(path, label, json.loads(json.dumps(meta, default=lcs.json_default)), arrays=post)
    lcs.validate_episode(path, period_us=round(harness.SAMPLE_STEPS * lcs.SIM_DT_S * 1e6))
    return path.name


def segments_of(row: dict) -> list[dict]:
    """Contiguous spans: walk pieces, loops (with their trip) and the post-release tail."""
    out, i = [], 0
    for lp in row["loops"]:
        a, b = lp["span"]
        if a > i:
            out.append({"kind": "walk", "span": [i, a]})
        out.append({"kind": "loop", "span": [a, b], "phase": lp["phase"],
                    "waypoint": lp["waypoint"], "loop": lp["loop"], "trip": lp["trip"],
                    "recovered": lp["recovered"]})
        i = b
    if row["post_from"] > i:
        out.append({"kind": "walk", "span": [i, row["post_from"]]})
    out.append({"kind": "post", "span": [row["post_from"], row["n_frames"]]})
    return out


def buffer_rows(col: Collector) -> list[dict]:
    out = []
    for i, r in enumerate(e for e in col.episodes if e["file"]):
        fin = r["final"]
        out.append({
            "id": f"stage2_{col.args.out.name}_{i:04d}", "iter": -1, "kind": "stage2_demo",
            "status": "ok", "weight": 1.0, "episode": str((col.dir / r["file"]).resolve()),
            "label_human": r["label_human"],
            "label_human_detail": {"success": r["label_human"], "rule": "both",
                                   "source": "sim_ground_truth_final_frame (after the release)",
                                   "large": fin["large"], "small": fin["small"],
                                   "mpc_end": {k: r["mpc_end"][k] for k in ("large", "small",
                                                                            "both")}},
            "label_strict": r["label_human"], "cloud_success": None,
            "cloud_success_role": "monitor only",
            "start": {**r["start"], "from_reset": False, "jitter": None},
            "reset_before": "human", "n_frames": r["n_frames"], "n_mpc_frames": 0,
            "n_prehold_frames": 0, "n_task_frames": r["n_task_frames"], "post_from": r["post_from"],
            "outcome_gt": harness._stream_label(col.ctx, fin["large"], fin["small"], fin["both"]),
            "fallback_demo": False,
            "segment": "episode", "segments": segments_of(r),
            "stage2": {"params": r["params"], "walk_end": r["walk_end"],
                       "trips": [(lp["phase"], lp["trip"], lp["recovered"]) for lp in r["loops"]
                                 if lp["trip"]],
                       "extra_reset": r["extra_reset"]},
        })
    return out


# --- report -----------------------------------------------------------------------------------

def _pct(k: int, n: int) -> str:
    return f"{k}/{n}" + (f" ({100.0 * k / n:.0f} %)" if n else "")


def write_report(out: Path, args, col: Collector, meta: dict, loader: dict | None,
                 wall_s: float) -> None:
    eps = [e for e in col.episodes if e.get("file")]
    skipped = [e for e in col.episodes if not e.get("file")]
    rel = out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out
    frames = [e["n_frames"] for e in eps]
    L = [f"# Stage-2 collection `{rel}`", "",
         (f"Created {meta['created']}, git {meta['git'].get('commit', '?')[:10]}, seed "
          f"{args.seed}, LCM `{args.lcm_url}`, wall {wall_s / 60:.1f} min, robot "
          f"{sum(e.get('robot_s', 0.0) for e in eps):.0f} s sim; {len(eps)} episodes written, "
          f"{len(skipped)} skipped; loops {'on' if args.loops else 'off'}, DART "
          f"{'off' if args.no_dart else 'on'}, cloud noise `{args.cloud_noise}`."), "",
         ("Labels: sim GT (`frame_metrics` / `classify`, both pulleys). Final = after the "
          "scripted release (the episode label); MPC end = before the release. Trips use "
          "real-available signals only (hand width, belt cloud)."), "",
         "## Success by start group", "",
         ("| start | episodes | both (final) | both (MPC end) | large engaged (final) | Franka "
          "held (MPC end) | walk ended early |"), "|---|---|---|---|---|---|---|"]
    for key in dict.fromkeys(e["start"]["key"] for e in eps):
        g = [e for e in eps if e["start"]["key"] == key]
        L.append(f"| {key} | {len(g)} | {_pct(sum(e['final']['both'] for e in g), len(g))} | "
                 f"{_pct(sum(e['mpc_end']['both'] for e in g), len(g))} | "
                 f"{_pct(sum(e['final']['large'] == 'engaged' for e in g), len(g))} | "
                 f"{_pct(sum(e['mpc_end']['held'] for e in g), len(g))} | "
                 f"{sum(e['walk_end'] is not None for e in g)} |")
    n = len(eps)
    L.append(f"| **all** | {n} | {_pct(sum(e['final']['both'] for e in eps), n)} | "
             f"{_pct(sum(e['mpc_end']['both'] for e in eps), n)} | "
             f"{_pct(sum(e['final']['large'] == 'engaged' for e in eps), n)} | "
             f"{_pct(sum(e['mpc_end']['held'] for e in eps), n)} | "
             f"{sum(e['walk_end'] is not None for e in eps)} |")
    loops = [lp for e in eps for lp in e["loops"]]
    sigs = sorted({lp["trip"] for lp in loops if lp["trip"]})
    L += ["", "## Loops and trips by phase", "",
          "| phase | loops | trips | " + " | ".join(sigs) + (" | " if sigs else "")
          + "rerouted | recovered | walk ended | aborts |",
          "|---|---|---|" + "---|" * len(sigs) + "---|---|---|---|"]
    for ph in BOXES:
        g = [lp for lp in loops if lp["phase"] == ph]
        t = [lp for lp in g if lp["trip"]]
        c = Counter(lp["trip"] for lp in t)
        ended = sum(e["walk_end"] is not None and e["walk_end"]["phase"] == ph for e in eps)
        L.append(f"| {ph} | {len(g)} | {_pct(len(t), len(g))} | "
                 + "".join(f"{c[s]} | " for s in sigs)
                 + f"{sum(lp['rerouted'] for lp in t)} | {sum(bool(lp['recovered']) for lp in t)} | "
                 f"{ended} | {sum(lp['abort'] is not None for lp in g)} |")
    ends = Counter(e["walk_end"]["why"] for e in eps if e["walk_end"])
    extra = sum(e["extra_reset"] for e in eps)
    useful = n - extra
    L += ["", "Walk ends: " + (", ".join(f"{k} {v}" for k, v in ends.items()) or "none") + ".",
          "", "## Resets", "",
          (f"One human reset per episode ({n}). Extra resets = episodes ending with the belt off "
           f"the large pulley (final GT label not engaged) or out of the Franka fingers (GT grasp "
           f"lost before the release): {extra}; useful = the rest: {useful}; extra resets per "
           f"useful episode {extra / useful:.2f}." if useful else
           f"One human reset per episode ({n}); extra resets {extra}, no useful episode."),
          "", "## Frames", ""]
    if frames:
        task = [e["n_task_frames"] for e in eps]
        L.append(f"Per episode: {np.mean(frames):.0f} mean (min {min(frames)}, max {max(frames)})"
                 f"; task frames (before the release) {np.mean(task):.0f} mean; loop frames "
                 f"{np.mean([sum(lp['span'][1] - lp['span'][0] for lp in e['loops']) for e in eps]):.0f}"
                 f" mean; total {sum(frames)} ({sum(task)} task).")
    L += ["", "## Episodes", "",
          ("| episode | hover h | slerp f1/f2/f3 | tilt cap | wrap dh | raise | end az | speed | "
           "loops | trips | walk end | MPC end L/S | final L/S | both | frames (task) | wall s |"),
          "|" + "---|" * 16]
    for e in eps:
        p = e["params"]
        trips = ", ".join(f"{lp['phase']}:{lp['trip']}" for lp in e["loops"] if lp["trip"])
        L.append(f"| {e['episode']} | {p['hover_h_mm']:.1f} | {p['slerp_f1']:.2f}/"
                 f"{p['slerp_f2']:.2f}/{p['slerp_f3']:.2f} | {p['tilt_cap_deg']:.1f} | "
                 f"{p['wrap_dh_mm']:+.1f} | {p['arc_raise_mm']:+.1f} | {p['end_az_deg']:.0f} | "
                 f"{p['speed_mps']:.3f} | {len(e['loops'])} | {trips or '-'} | "
                 f"{e['walk_end']['why'] if e['walk_end'] else '-'} | "
                 f"{e['mpc_end']['large']}/{e['mpc_end']['small']} | "
                 f"{e['final']['large']}/{e['final']['small']} | {e['final']['both']} | "
                 f"{e['n_frames']} ({e['n_task_frames']}) | {e['wall_s']:.0f} |")
    for e in skipped:
        L.append(f"| {e['episode']} | skipped: {e.get('skipped') or e.get('error')} |"
                 + " |" * 14)
    L += ["", "## Budget", "", f"Multipliers {col.budget.mult} (boxes at 1: plan 4.3)."]
    L += [f"- {h['episode']}: {h['phase']} {h['signal']} {h['before']:g} -> {h['after']:g}"
          for h in col.budget.history] or ["- no halving"]
    if loader is not None:
        t, a = loader["test"], loader["load_all"]
        L += ["", "## Loader", "",
              (f"- `test_belt_seg_loader.py` on {t['file'] if t else '-'}: "
               f"{'PASS' if t and t['pass'] else 'FAIL' if t else 'not run'}"),
              (f"- `load_belt_seg_episode` on all {len(eps)} files: "
               f"{'PASS' if a['pass'] else 'FAIL'} ({a['files']} files, {a['pairs']} pairs)")]
    (out / "report.md").write_text("\n".join(L) + "\n")


# --- CLI --------------------------------------------------------------------------------------

def select_starts(index: dict, ids: str | None) -> list[dict]:
    """Variants whose id (or start key) matches a comma list of fnmatch patterns; all if None."""
    pats = [x.strip() for x in (ids or "").split(",") if x.strip()]
    return [v for v in index["variants"]
            if not pats or any(fnmatch.fnmatch(v["id"], p) or fnmatch.fnmatch(v["start"], p)
                               for p in pats)]


def _param(text: str) -> tuple[str, float]:
    k, v = text.split("=", 1)
    if k not in PARAM_RANGES:
        raise argparse.ArgumentTypeError(f"{k} not in {sorted(PARAM_RANGES)}")
    return k, float(v)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--starts", type=Path, default=DEFAULT_STARTS, help="stage-2 start-set dir")
    p.add_argument("--start-ids", default=None,
                   help="comma list of variant ids or start keys (fnmatch patterns); default all")
    p.add_argument("--episodes-per-start", type=int, default=1)
    p.add_argument("--out", type=Path, required=True, help="new dir")
    p.add_argument("--lcm-url", default=DEFAULT_LCM_URL)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--record", action="store_true", help="a viser recording per episode")
    p.add_argument("--no-loops", dest="loops", action="store_false")
    p.add_argument("--no-dart", action="store_true")
    p.add_argument("--nominal", action="store_true", help="D10L nominal parameters, no draws")
    p.add_argument("--set", type=_param, action="append", default=[], metavar="KEY=VALUE",
                   help=f"fix a plan parameter (envelope study), keys {sorted(PARAM_RANGES)}")
    # 2 s: the large pulley stays engaged 9/10 vs 6/10 at the harness's 0.5 s (V12-93 A/B)
    p.add_argument("--settle-s", type=float, default=2.0,
                   help="settle after the restore, before the walk")
    p.add_argument("--dev-s", type=float, default=0.6)
    p.add_argument("--ret-s", type=float, default=0.6)
    p.add_argument("--hold-s", type=float, default=0.2)
    p.add_argument("--emphasis-frac", type=float, default=0.5)
    p.add_argument("--budget-from", type=Path, default=None,
                   help="start from an earlier run's budget.json multipliers")
    p.add_argument("--cloud-noise", default="off", metavar="PRESET|JSON")
    # The stiffer stage-2 belt opens the sim hand to ~10 mm while still held (stage 1 uses 10).
    p.add_argument("--franka-width-max", type=float, default=20.0)
    p.add_argument("--osc-timeout-s", type=float, default=5.0)
    p.add_argument("--no-loader-check", dest="loader_check", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="print the schedule and plans, no sim")
    return p


def main() -> int:
    args = create_parser().parse_args()
    args.set = dict(args.set)
    cli.configure_logging()
    harness.check_private_url(args.lcm_url)
    import round_belt_task.offline_simulation as osim
    from round_belt_task.constants import SCENE_DIRECTIVES
    from round_belt_task.osc_bridge import OscTimeout

    # the stage-2 scene or a V12-106 belt-scan variant of it (round_belt_scene_wr*.yaml)
    if Path(SCENE_DIRECTIVES).name != STAGE2_SCENE and not Path(SCENE_DIRECTIVES).name.startswith(
            "round_belt_scene_wr"):
        raise SystemExit(f"scene {SCENE_DIRECTIVES}: set ROUND_BELT_SCENE to the {STAGE2_SCENE} "
                         "variant")
    osim.FRANKA_WIDTH_MM = (osim.FRANKA_WIDTH_MM[0], args.franka_width_max)
    starts_dir = (REPO_ROOT / args.starts).resolve()
    s_index = json.loads((starts_dir / "index.json").read_text())
    starts = select_starts(s_index, args.start_ids)
    if not starts:
        raise SystemExit(f"--start-ids {args.start_ids}: no variant in {starts_dir}")
    schedule = [(rep, s) for rep in range(args.episodes_per_start) for s in starts]
    hw = hw_poses()
    if args.dry_run:
        for rep, s in schedule:
            key = [args.seed, zlib.crc32(s["id"].encode()), rep]
            prm = {**(dict(NOMINAL) if args.nominal else draw_params(
                np.random.default_rng([*key, 0]))), **args.set}
            plan = build_plan(prm, np.asarray(s["measured"]["ee_franka"]), hw)
            print(f"{s['id']} r{rep}: " + ", ".join(f"{k} {v:.3f}" for k, v in prm.items()))
            print("   " + " ".join(f"{lab}({ph[0]}) " + "/".join(f"{x:.0f}" for x in
                                                              small_polar(p[:3]))
                                   for lab, ph, p in plan))
        return 0
    out = (REPO_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} exists and is not empty; pick a new dir")
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_info()}
    budget = Budget(out / "budget.json", args.budget_from)
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=cli.FIRST, last=cli.LAST)
    osc_binary, osc_root = harness.default_osc_binary()
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=args.osc_timeout_s)
    t_run = time.perf_counter()
    index = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
             **meta, "scene": str(SCENE_DIRECTIVES), "starts": str(starts_dir),
             "phase_labels": list(PHASES), "segment_labels": list(SEGMENTS),
             "cmd_source": CMD_STAGE2, "param_ranges": PARAM_RANGES, "dart": DART,
             "trip": dataclasses.asdict(TRIP), "loops": LOOPS, "loop_waypoints": LOOP_WP,
             "hw_poses": {k: (np.asarray(v).tolist()) for k, v in hw.items()}}
    col = None
    try:
        warm = sim.start_osc(out / "osc.log", binary=osc_binary, cwd=osc_root)
        index["osc"] = {**sim.osc.describe(), "warm_up_s": warm}
        sim.restore(sim_snapshot.load(starts_dir / starts[0]["file"]))
        live = sim.state_0.body_q.numpy()[int(sim.info.pulley_bodies[0])][:3].astype(float)
        err = float(np.linalg.norm(live - bc.known_small_pulley_pose7()[:3]) * 1e3)
        if err > 2.0:
            raise SystemExit(f"small pulley {err:.2f} mm from known_small_pulley_pose7")
        index["small_pulley_err_mm"] = err
        board, gripper = sim.clearance_geometry(jaw_bytes=(s_index["args"].get("ur_byte", 191),))
        plate = board.plate
        cloud = cli.cloud_options(argparse.Namespace(cloud_noise=args.cloud_noise, belt_seg=True,
                                                     no_pcd=False))
        ctx = argparse.Namespace(
            args=argparse.Namespace(no_pcd=False, belt_seg_source="camera"), mode="stage2",
            encoder=None, sim=sim, board=board, gripper=gripper, thresholds=DEFAULT_THRESHOLDS,
            small_thresholds=DEFAULT_THRESHOLDS, outcome_pulley="both",
            tangent=pert.belt_tangent(nominal), cloud_rng=None,
            cloud={**cloud, "origins": cli.sim_camera_origins(sim)},
            plate_top_z=float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half))
        col = Collector(args, sim, ctx, hw, budget)
        for i, (rep, s) in enumerate(schedule):
            logger.info(f"[S2] episode {i + 1}/{len(schedule)}: {s['id']} r{rep}")
            try:
                col.run(s, rep, starts_dir)
            except OscTimeout as exc:
                if not exc.process_alive:
                    raise
                sim.bridge.forget_pending()
                col.episodes.append({"episode": f"{s['id']}_r{rep}", "file": None,
                                     "error": f"osc timeout: {exc}"})
                logger.warning(f"[S2] {s['id']} r{rep}: osc timeout ({exc}); next episode")
            write_json(out / "index.json", json.loads(json.dumps(
                {**index, "episodes": col.episodes}, default=lcs.json_default)))
    finally:
        sim.close("finished")
        with contextlib.suppress(OSError):  # emptied by finish_recording
            (out / "recordings_tmp").rmdir()
    wall = time.perf_counter() - t_run
    rows = buffer_rows(col)
    write_json(out / "buffer.json", {"version": 2, "iter": -1, "n": len(rows),
                                     "source": "stage-2 scripted demos with loops (V12-93)",
                                     "rollouts": rows})
    files = [col.dir / e["file"] for e in col.episodes if e.get("file")]
    loader = ms.loader_check(files) if args.loader_check and files else None
    index.update(episodes=col.episodes, budget=json.loads(budget.path.read_text()),
                 loader=loader, wall_s=wall)
    write_json(out / "index.json", json.loads(json.dumps(index, default=lcs.json_default)))
    write_report(out, args, col, meta, loader, wall)
    logger.info(f"[S2] {len(files)} episodes, both {sum(e['final']['both'] for e in col.episodes if e.get('file'))}, "
                f"wall {wall / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
