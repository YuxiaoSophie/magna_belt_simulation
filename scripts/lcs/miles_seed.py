#!/usr/bin/env python3
"""MILES-style seed collector: deviate-and-return cycles around demo waypoints (V12-23/24).

After Papagiannis & Johns, "MILES" (CoRL 2024, arXiv 2410.19693); plan
``handoffs/PLAN-20261006-miles.md`` (package P-M1). Per reference demo: restore its start snapshot
(the one human reset), settle 0.5 s, walk the demo by replaying its recorded EE poses
(``demo_prefix.relative_targets`` anchored on the settled poses, 7-knot Franka trajectory + UR IK
as ``demo_prefix.replay_prefix``) from motion start to the last frame and, at each waypoint k,
run ``--cycles`` deviate-and-return cycles: min-jerk to a sampled 12-d EE offset (``--dev-s``),
straight min-jerk return to the waypoint pose (``--ret-s``), hold (``--hold-s``). Offsets: uniform
in a box (free k < latch, contact k >= latch), half the draws on one emphasised axis per moving
arm at the box edge (Franka x/y, UR x/y/z); modes rotate Franka / UR / both over the demo's
cycles; each arm's offset is scaled down until the frozen clearance model (CAD, real-available)
keeps >= min(2 mm, clearance at the waypoint) and the Franka tip >= 4 mm above the plate.

Validity after each return, real signals only: reach (EE within ``--reach-mm`` / ``--reach-deg``
of the waypoint target), grasp ``held()``, belt not disturbed (symmetric Chamfer of the
segmented belt cloud vs the demo's ``pcd_belt_seg`` at frame k <= tau; tau per phase = the
``--tau-pct`` percentile of the clean demos' same-grasp matched-frame Chamfer spread;
``--belt-ref arrival``: vs the cloud measured on arrival at the waypoint instead);
``--deploy``: whitened latent distance, report-only. Sim GT (label change or Chamfer of the GT
belt points vs the same reference > the same-percentile GT spread) is for scoring only. The first
invalid cycle stops the demo at waypoint R (frames kept); ``--on-disturb skip-waypoint``: a
disturbed cycle only ends that waypoint's cycles (recorded as skipped) and the walk continues;
``--diag-restore`` (sim-only, diagnostic) instead restores the waypoint's arrival snapshot.

Writes ``<out>/miles/episode_*.npz`` (one per cycle, one per walk segment; dataset format of
``eval_learned_mpc.build_writer``, ``cmd_delta`` actions, ``sim_cmd_source`` 4),
``buffer.json`` (rows ``kind: miles``), ``budget.json``, ``index.json``, ``miles_report.md``.

Run:
    uv run --frozen python scripts/lcs/miles_seed.py --demos gv_00 --waypoints 20,30,33,43,53 \\
        --lcm-url 'udpm://239.255.76.196:7796?ttl=0' --out data/lcs/realfirst/miles/smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import subprocess
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
import demo_prefix as dp
import eval_learned_mpc as harness
from loguru import logger
from make_ee_offset_starts import min_jerk
from scipy.spatial import cKDTree

from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import UrTracking, ik, rot_axis_angle
from round_belt_task.commander import (
    CommanderParams,
    mat3_to_quat,
    pose_mat,
    quat_to_mat3,
    saved_traj_message,
    slerp,
)
from round_belt_task.episode_io import (
    git_info,
    pose7_mat,
    sha256_file,
    slant_row,
    write_json,
)
from round_belt_task.osc_bridge import OscTimeout
from round_belt_task.outcome import (
    DEFAULT_THRESHOLDS,
    classify,
    classify_episode,
    frame_metrics,
    slant_episode,
)
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import belt_cloud as bc
from task_common import belt_metrics as bm
from task_common import clearance_model as cm
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot

GV = REPO_ROOT / "data" / "lcs" / "start_states" / "grasp_variants"
V11 = REPO_ROOT / "data" / "lcs" / "realfirst" / "v1_1"
CMP = REPO_ROOT / "data" / "lcs" / "realfirst" / "cmp"
# reference demos (plan E1/E2): name -> (episode, start set, start variant)
DEMOS = {"gv_00": (V11 / "seed" / "demo" / "episode_0000.npz", "set2", "gv_00"),
         "gv_01": (V11 / "seed" / "demo" / "episode_0002.npz", "set2", "gv_01"),
         "gv_02": (V11 / "seed_ext" / "demo" / "episode_0004.npz", "set2", "gv_02"),
         "s3_gv_02": (CMP / "demos_ext2" / "demo" / "episode_0004.npz", "set3", "gv_02"),
         "s3_gv_03": (CMP / "demos_ext2" / "demo" / "episode_0006.npz", "set3", "gv_03")}
TAU_BUFFERS = (V11 / "seed" / "buffer.json", V11 / "seed_ext" / "buffer.json")
LCS_ROOT = Path.home() / "git" / "lcs_learning"
PY_LCS = LCS_ROOT / ".venv" / "bin" / "python"
LOADER_TEST = LCS_ROOT / "outputs" / "sim_belt_v11_20261005" / "tests" / "test_belt_seg_loader.py"

MODES = ("franka", "ur", "both")
SEGMENTS = ("walk", "dev", "return", "hold")
PHASE_LABELS = [f"miles:{s}" for s in SEGMENTS]
CMD_MILES = 4  # sim_cmd_source of every MILES frame (harness codes 0-3)
EMPHASIS = {"franka": (0, 1), "ur": (0, 1, 2)}  # world axes oversampled per arm
DEV_CLEAR_MM = 2.0
TIP_MIN_MM = 4.0
SCALE_BISECT = 10
START_SETTLE_S = harness.START_SETTLE_S
LOADER_MIN_FRAMES = 7  # the deleted-keys test empties frame 5


# --- offsets ----------------------------------------------------------------------------------

def waypoint_frames(marks: dict, last: int, stride: int, stride_contact: int) -> list[int]:
    """Every ``stride`` frames from motion start; every ``stride_contact`` in
    [latch - 5, first_hold + 5]."""
    lo, hi = marks["latch"] - 5, marks["first_hold"] + 5
    out, k = [], marks["move0"]
    while k <= last:
        out.append(k)
        k += stride_contact if lo <= k < hi else stride
        if out[-1] < lo < k:
            k = lo
    return out


def motion_waypoint_frames(demo: dict, marks: dict, last: int, step_mm: float,
                           n_hold: int) -> list[int]:
    """One waypoint per ``step_mm`` of summed Franka + UR EE path from motion start, then
    ``n_hold`` evenly spaced frames over the static rest (motion end .. last)."""
    f = np.asarray(demo["ee_f"], np.float64)[:, :3]
    u = np.asarray(demo["ee_u"], np.float64)[:, :3]
    d = 1e3 * (np.linalg.norm(np.diff(f, axis=0), axis=1) + np.linalg.norm(np.diff(u, axis=0), axis=1))
    c = np.concatenate([[0.0], np.cumsum(d)])
    s = c - c[marks["move0"]]
    moving = np.flatnonzero(d > 0.05)
    end = int(moving[-1]) + 1 if len(moving) else marks["move0"]
    out = [marks["move0"]]
    for k in range(marks["move0"] + 1, end + 1):
        if s[k] - s[out[-1]] >= step_mm:
            out.append(k)
    if out[-1] != end:
        out.append(end)
    hold = np.linspace(end, last, n_hold + 1)[1:] if n_hold > 0 and last > end else []
    return sorted(set(out) | {int(np.rint(h)) for h in hold})


def draw_offset(rng: np.random.Generator, mode: str, box: tuple[float, float],
                emphasis_frac: float) -> dict:
    """``{arm: [dx, dy, dz mm, rx, ry, rz deg (world rotvec)]}`` for the moving arms."""
    mm, deg = box
    arms = ("franka", "ur") if mode == "both" else (mode,)
    single = bool(rng.random() < emphasis_frac)
    out = {"franka": np.zeros(6), "ur": np.zeros(6)}
    for arm in arms:
        if single:
            ax = int(rng.choice(EMPHASIS[arm]))
            out[arm][ax] = mm * (1.0 if rng.random() < 0.5 else -1.0)
        else:
            out[arm][:3] = rng.uniform(-mm, mm, 3)
            out[arm][3:] = rng.uniform(-deg, deg, 3)
    return {"franka": out["franka"], "ur": out["ur"], "single_axis": single}


def offset_pose(pose7, off6, s: float = 1.0) -> np.ndarray:
    """``pose7`` moved by ``s`` x the offset (world translation, world rotvec about the EE)."""
    p = np.asarray(pose7, np.float64)
    off = np.asarray(off6, np.float64) * s
    ang = math.radians(float(np.linalg.norm(off[3:])))
    R = quat_to_mat3(p[3:])
    if ang > 0.0:
        R = rot_axis_angle(off[3:] / np.linalg.norm(off[3:]), ang) @ R
    return np.concatenate([p[:3] + off[:3] * 1e-3, mat3_to_quat(R)])


class Clearance:
    """Frozen clearance model of the current grasp (``clearance_model.export_from_sim``)."""

    def __init__(self, sim, plate_top_z: float) -> None:
        self.model = cm.export_from_sim(sim, cm.DEFAULT_UR_MIN_MM * 1e-3,
                                        cm.DEFAULT_FRANKA_MIN_MM * 1e-3)
        self.board = self.model.board()
        self.points = self.model.kernel_points()
        self.plate_top_z = plate_top_z

    def ur_mm(self, pose7) -> float:
        return 1e3 * cm.set_distance(self.board, self.points[0], cm.pose_mat(pose7))

    def franka_mm(self, pose7) -> float:
        d = 1e3 * cm.set_distance(self.board, self.points[1], cm.pose_mat(pose7))
        return min(d, 1e3 * (float(pose7[2]) - self.plate_top_z) - TIP_MIN_MM + DEV_CLEAR_MM)

    def scale(self, arm: str, wp7, off6) -> tuple[float, float, float]:
        """``(scale, d at the waypoint mm, d at the deviated pose mm)``: largest s in [0, 1]
        (bisection) whose path points s/2 and s keep d >= min(DEV_CLEAR_MM, d(waypoint))."""
        f = self.ur_mm if arm == "ur" else self.franka_mm
        d0 = f(wp7)
        need = min(DEV_CLEAR_MM, d0)

        def ok(s):
            return all(f(offset_pose(wp7, off6, x)) >= need - 1e-6 for x in (0.5 * s, s))

        if not np.any(off6) or ok(1.0):
            return 1.0, d0, f(offset_pose(wp7, off6))
        lo, hi = 0.0, 1.0
        for _ in range(SCALE_BISECT):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo, d0, f(offset_pose(wp7, off6, lo))


# --- paths + drive ----------------------------------------------------------------------------

class CyclePath:
    """Min-jerk ``a -> b`` over ``dev_s``, min-jerk ``b -> a`` over ``ret_s``, hold ``a``."""

    def __init__(self, a7, b7, dev_s: float, ret_s: float, hold_s: float) -> None:
        self.a, self.b = np.asarray(a7, np.float64), np.asarray(b7, np.float64)
        self.t1, self.t2 = float(dev_s), float(dev_s + ret_s)
        self.end = self.t2 + float(hold_s)

    def segment(self, tau: float) -> int:
        return SEGMENTS.index("dev" if tau < self.t1 else "return" if tau < self.t2 else "hold")

    def __call__(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        if tau < self.t1:
            x, y, s = self.a, self.b, min_jerk(tau / self.t1)
        else:
            x, y, s = self.b, self.a, min_jerk((tau - self.t1) / (self.t2 - self.t1))
        q = slerp(x[3:], y[3:], s)
        return x[:3] + s * (y[:3] - x[:3]), q / np.linalg.norm(q)

    def pose7(self, tau: float) -> np.ndarray:
        return np.concatenate(self(tau))


def drive(sim, ctx, path_f, path_u, steps: int, on_frame=None) -> dict:
    """``demo_prefix.replay_prefix``'s step loop on given paths (same aborts: grasp ``held()``
    every sample, Franka tip floor, 2F-85 board clearance, UR IK miss); ``on_frame(k, cmd, tau)``
    at every sample tick before the step. Leaves the Franka on a fresh hold hook."""
    dt, n = sim.frame_dt, harness.SAMPLE_STEPS
    knot_t = CommanderParams().dt * np.arange(harness.N_KNOTS)
    u_coords = sim.arm_coords()[1]
    t0 = sim.osc_time_s(sim.step_index)

    def hook(step, t, joint_q, body_q):
        knots = [path_f(t - t0 + o) for o in knot_t]
        return saved_traj_message(round(t * 1e6), np.stack([k[0] for k in knots]),
                                  np.stack([k[1] for k in knots]), t + knot_t)

    q_ur = sim.arm_targets()[1]
    lead_gain = sim.arm_kd / sim.arm_ke / dt
    tip_min = clear_min = math.inf
    last_clear = None
    board_neg = 0
    abort = None
    kind = None
    k = 0
    sim.commander_hook = hook
    try:
        while k < steps:
            if k % n == 0:
                t = sim.osc_time_s(sim.step_index)
                tau = t - t0
                grasp = sim.grasp_state()
                if not all(grasp.held()):
                    abort, kind = f"grasp lost at step {k} ({grasp.describe()})", "grasp_lost"
                    break
                if on_frame is not None:
                    knots = [path_f(tau + o) for o in knot_t]
                    kn = np.hstack([np.stack([x[0] for x in knots]),
                                    np.stack([x[1] for x in knots])])
                    on_frame(k, {"knots": kn, "knot_times": t + knot_t, "knot0": kn[0].copy(),
                                 "knot1": kn[1].copy(), "ur_t": path_u.pose7(tau),
                                 "ur_t1": path_u.pose7(tau + lcs.ACTION_KNOT_DT_S),
                                 "traj_utime": round(t * 1e6)}, tau)
            p, q = path_u(sim.osc_time_s(sim.step_index + 1) - t0)
            q_new, err_p, err_r, _ = ik(UrTracking, pose_mat(p, q), q_ur,
                                            pos_tol=cli.UR_IK_POS_TOL, rot_tol=cli.UR_IK_ROT_TOL)
            if err_p > cli.UR_IK_POS_TOL or err_r > cli.UR_IK_ROT_TOL:
                abort, kind = (f"UR IK missed at step {k} ({err_p * 1e3:.3f} mm, "
                               f"{np.degrees(err_r):.3f} deg)"), "unreachable"
                break
            lead = cli.UR_VELOCITY_LEAD * lead_gain if k > 0 else 0.0
            sim.set_ur_target(q_new + lead * (q_new - q_ur))
            q_ur = q_new
            sim.control_step()
            k += 1
            joint_q = sim.state_0.joint_q.numpy()
            tip = float(sim.franka_measured_pose7(joint_q)[2] - ctx.plate_top_z)
            tip_min = min(tip_min, tip)
            if k % cli.CLEARANCE_EVERY == 0:
                last_clear = float(ctx.gripper.measure(
                    ctx.board, sim.state_0.body_q.numpy().astype(np.float64)))
                clear_min = min(clear_min, last_clear)
            board_neg = board_neg + 1 if last_clear is not None and last_clear < 0.0 else 0
            if tip * 1e3 < harness.TIP_FLOOR_MM:
                abort = f"franka finger_tip {tip * 1e3:.2f} mm above the plate at step {k}"
            elif board_neg > harness.BOARD_NEGATIVE_STEPS:
                abort = f"2F-85 board clearance < 0 for {board_neg} steps at step {k}"
            if abort is not None:
                kind = "unreachable"
                break
    finally:
        sim.commander_hook = sim.make_hold_hook()
        if abort is not None:
            sim.set_ur_target(sim.state_0.joint_q.numpy()[u_coords])
    return {"steps": k, "abort": abort, "abort_kind": kind, "tip_min_mm": tip_min * 1e3,
            "min_clearance_mm": clear_min * 1e3}


# --- frames + files ---------------------------------------------------------------------------

class FileRec:
    """Frames of one output file (``eval_learned_mpc._sample`` + the drive's command)."""

    def __init__(self, sim, ctx, seg_of, demo_frame_of) -> None:
        self.sim, self.ctx = sim, ctx
        self.sampler = harness._sampler(sim, ctx)
        self.seg_of, self.demo_frame_of = seg_of, demo_frame_of
        self.frames: list[dict] = []
        self.timing = {"render": 0.0, "encode": 0.0}

    def __call__(self, k: int, cmd: dict, tau: float) -> None:
        fr = harness._sample(self.sim, self.ctx, self.sampler, k, 0, publish=False,
                             timing=self.timing)
        nx = len(fr["latent"])
        fr.update(cmd, n_knots=harness.N_KNOTS, cmd_source=CMD_MILES,
                  ur_line=np.full(16, np.nan), ur_fresh_exact=False,
                  ur_own_delta=np.full(6, np.nan), ctrl_hand_mm=math.nan, ctrl_robotiq=-1,
                  stage=0, latent_pred=np.full(nx, np.nan), goal_dist=np.full(1, np.nan),
                  phase=self.seg_of(tau), demo_frame=self.demo_frame_of(tau))
        self.frames.append(fr)


def write_file(ctx, path: Path, frames: list[dict], drive_rep: dict, miles: dict) -> dict | None:
    """Write + validate one file; ``None`` (not written) below 2 frames."""
    if len(frames) < 2:
        return None
    belt_T = np.stack([fr["belt"] for fr in frames])
    pulley_T = np.stack([fr["pulley_raw"] for fr in frames])
    label, metrics = classify_episode(belt_T, pulley_T, ctx.thresholds)
    metrics.update(slant_episode(belt_T, pulley_T, ctx.tangent, ctx.thresholds))
    c = ctx.cloud
    meta = {
        "phase_labels": PHASE_LABELS, "mode": "miles", "outcome": label, "status": "ok",
        "reason": drive_rep["abort"], "start_state": miles["start"], "backend": "osc",
        "ee_pose_source": "measured finger_tip / tracking frame (FK of the measured joints)",
        "action_source": ("knot 1 - knot 0 of the collector's 7-knot Franka "
                          "TARGET_CARTESIAN_POSE_TRAJECTORY on the MILES path / the UR path at "
                          "t + knot dt minus at t (tracking frame, world)"),
        "cmd_source_codes": {"miles": CMD_MILES},
        "latent_source": "none (miles collector)", "time_source": "OSC clock",
        "sample_period_s": harness.SAMPLE_STEPS * lcs.SIM_DT_S,
        "pulley_pose_layout": lcs.POSE_LAYOUT, "start_step": frames[0]["sim_step"],
        "thresholds": dataclasses.asdict(ctx.thresholds),
        "grippers": "held at the start-state commands",
        "cloud_noise": {**c["noise"].to_dict(), "camera_origins": c["origins"].tolist()},
        "belt_seg": bc.belt_seg_meta(c["noise"], c["seg"], "record", c["pose"]),
        "miles": miles,
    }
    min_clear = min(drive_rep["min_clearance_mm"],
                    min(fr["clearance_mm"] for fr in frames))
    post = {"outcome": np.array(label), "status": np.array("ok"), "mode": np.array("miles"),
            "start_state": np.array(json.dumps(miles["start"])),
            "wrap_deg": metrics["wrap_deg"], "h_median_mm": metrics["h_median_mm"],
            **{k: np.array(metrics[k]) for k in cli.SLANT_KEYS},
            "min_board_clearance_mm": np.float32(min_clear),
            "board_contact": np.array(min_clear < 0.0),
            "min_franka_tip_clearance_mm": np.float32(drive_rep["tip_min_mm"]),
            "backend": np.array("osc"),
            "osc_utime_offset_us": np.int64(ctx.sim.bridge.utime_offset_us),
            "miles_demo_frame": np.array([fr["demo_frame"] for fr in frames], np.int16),
            "miles": np.array(json.dumps(miles, default=lcs.json_default))}
    post = {lcs.EXTRA_PREFIX + k: v for k, v in post.items()}
    seg = [SEGMENTS[fr["phase"]] for fr in frames]
    post["miles_segment"] = np.array(seg)  # train_bc_prior.py reads this unprefixed key
    writer, _ = harness.build_writer(frames, pcd=True, definition="cmd_delta")
    writer.write(path, label, meta, arrays=post)
    lcs.validate_episode(path, period_us=round(harness.SAMPLE_STEPS * lcs.SIM_DT_S * 1e6))
    spans = {}
    for i, name in enumerate(seg):
        spans.setdefault(name, [i, i + 1])[1] = i + 1
    return {"file": path.name, "n_frames": len(frames), "outcome_gt": label, "segments": spans,
            "final_wrap_deg": float(metrics["wrap_deg"][-1]),
            "final_h_median_mm": float(metrics["h_median_mm"][-1]), **slant_row(metrics),
            "size_bytes": path.stat().st_size}


# --- belt checks ------------------------------------------------------------------------------

def chamfer_mm(a, b) -> tuple[float, float]:
    """``(mean, p90)`` of the symmetric nearest-neighbour distances [mm] of two clouds."""
    a, b = np.asarray(a, np.float64).reshape(-1, 3), np.asarray(b, np.float64).reshape(-1, 3)
    if not len(a) or not len(b):
        return math.inf, math.inf
    da, db = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
    mean = 0.5 * (da.mean() + db.mean())
    return 1e3 * float(mean), 1e3 * float(max(np.percentile(da, 90), np.percentile(db, 90)))


def gt_rmse_mm(a, b) -> float:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return 1e3 * float(np.sqrt((d * d).sum(-1).mean()))


def _stats(x) -> dict:
    x = np.asarray(x, np.float64)
    return {"n": len(x), "p50": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": float(x.max())}


def demo_spread(buffers, pct: float) -> dict:
    """Matched-frame spread of the clean demos (``label_human`` True), same-grasp pairs only:
    Chamfer of ``pcd_belt_seg`` and GT belt-body RMSE per phase (free = [move0, latch),
    contact = [latch, end]); ``tau`` = the ``pct`` percentile."""
    rows = [r for b in buffers for r in json.loads(Path(b).read_text())["rollouts"]
            if r["kind"] == "demo" and r.get("label_human") is True]
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(f"{r['start']['set']}/{r['start']['id']}", []).append(r["episode"])
    data = {}
    for f in {e for g in groups.values() for e in g}:
        with np.load(f, allow_pickle=True) as d:
            data[f] = (d["pcd_belt_seg"], np.asarray(d["pcd_belt"], np.float64))
        data[f] += (dp.load_demo(f),)
    acc = {(m, ph): [] for m in ("cloud", "gt") for ph in ("free", "contact")}
    pairs = []
    for key, eps in sorted(groups.items()):
        for a, b in itertools.combinations(sorted(eps), 2):
            (sa, ga, ma), (sb, gb, mb) = data[a], data[b]
            if ma["move0"] != mb["move0"]:  # latches differ by <= 1 frame on reset repeats
                raise ValueError(f"{a} / {b}: motion starts differ")
            pairs.append([key, a, b])
            for t in range(ma["move0"], min(len(sa), len(sb))):
                ph = "free" if t < ma["latch"] else "contact"
                acc[("cloud", ph)].append(chamfer_mm(sa[t], sb[t])[0])
                acc[("gt", ph)].append(bm.belt_chamfer_mm(ga[t], gb[t]))
    out = {"pct": pct, "groups": groups, "pairs": len(pairs),
           "rule": (f"tau[phase] = p{pct:g} of the symmetric Chamfer (mean NN) of pcd_belt_seg "
                    "between every same-grasp pair of clean demos at matched demo frames "
                    "(frames move0..end; free = before the first demo's latch, contact = from it); "
                    "gt_tau likewise on the Chamfer of the 150 GT belt material points "
                    "(shape only: sliding along the belt is not a disturbance; report-only)")}
    for m in ("cloud", "gt"):
        out[m] = {ph: _stats(acc[(m, ph)]) for ph in ("free", "contact")}
    out["tau_mm"] = {ph: float(np.percentile(acc[("cloud", ph)], pct))
                     for ph in ("free", "contact")}
    out["gt_tau_mm"] = {ph: float(np.percentile(acc[("gt", ph)], pct))
                        for ph in ("free", "contact")}
    return out


# --- collector --------------------------------------------------------------------------------

def ee_now(sim) -> tuple[np.ndarray, np.ndarray]:
    return (sim.franka_measured_pose7(),
            pose7_mat(UrTracking.fk(sim.arm_positions()[1])))


class Collector:
    def __init__(self, args, sim, ctx, tau: dict, encoder=None, z_std=None) -> None:
        self.args, self.sim, self.ctx, self.tau = args, sim, ctx, tau
        self.encoder, self.z_std = encoder, z_std
        self.dir = Path(args.out) / "miles"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files: list[dict] = []  # one per written file (the buffer rows' source)
        self.cycles: list[dict] = []
        self.demos: list[dict] = []
        self.resets: list[dict] = []
        self.robot_s = 0.0
        self.n_cycles = 0  # mode rotation over every cycle of the run

    # one measurement of the current state vs the demo at frame k
    def measure(self, demo: dict, k: int, phase: str, ref: dict | None) -> dict:
        sim, ctx = self.sim, self.ctx
        fr = harness._sample(sim, ctx, harness._sampler(sim, ctx), 0, 0, publish=False,
                             timing={"render": 0.0, "encode": 0.0})
        ch, ch90 = chamfer_mm(fr["pcd_belt_seg"], demo["seg"][k])
        body_q = sim.state_0.body_q.numpy()
        pulley = body_q[int(sim.info.pulley_bodies[1])].astype(np.float64)
        fm = frame_metrics(fr["belt"], pulley, th=ctx.thresholds)
        fm_d = frame_metrics(demo["belt"][k], pulley, th=ctx.thresholds)
        gt = bm.belt_chamfer_mm(fr["belt_pts"], demo["pcd_belt"][k])
        lab, lab_d = classify(fm), classify(fm_d)
        out = {"ee_f": fr["ee_f"], "ee_u": fr["ee_u"], "seg": fr["pcd_belt_seg"],
               "grasp": [bool(v) for v in fr["grasp_ok"]],
               "cloud": {"chamfer_mm": ch, "chamfer_p90_mm": ch90,
                         "n_pts": len(fr["pcd_belt_seg"]), "tau_mm": self.tau[phase]},
               "gt": {"belt_chamfer_mm": gt, "tau_mm": self.args.gt_tau[phase],
                      "belt_rmse_mm": gt_rmse_mm(fr["belt"], demo["belt"][k]),
                      "disturbed": bool(gt > self.args.gt_tau[phase] or lab != lab_d),
                      "label": lab, "label_demo": lab_d, "h_median_mm": fm.h_median_mm,
                      "wrap_deg": fm.wrap_deg}}
        if ref is not None:
            out["cloud"]["chamfer_vs_arrival_mm"] = chamfer_mm(fr["pcd_belt_seg"], ref["seg"])[0]
            out["gt"]["belt_rmse_vs_arrival_mm"] = gt_rmse_mm(fr["belt"], ref["belt"])
            out["gt"]["belt_chamfer_vs_arrival_mm"] = bm.belt_chamfer_mm(fr["belt_pts"],
                                                                         ref["belt_pts"])
        if self.encoder is not None:
            from task_common.latent_encoder import encoder_input

            z = self.encoder.encode(encoder_input(self.encoder, fr), fr["prop"], fr["belt_pts"])
            out["latent_dist_w"] = float(np.linalg.norm((z - demo["z"][k]) / self.z_std))
        out["belt"], out["belt_pts"] = fr["belt"], fr["belt_pts"]
        return out

    def demo_latents(self, demo: dict, frames) -> dict:
        from task_common.latent_encoder import encoder_input

        out = {}
        for k in frames:
            fr = {"pcd_belt_seg": demo["seg"][k], "cloud": None}
            out[k] = self.encoder.encode(encoder_input(self.encoder, fr), demo["state"][k],
                                         demo["pcd_belt"][k])
        return out

    def _file(self, name: str, rec: FileRec, rep: dict, miles: dict) -> dict | None:
        row = write_file(self.ctx, self.dir / f"{name}.npz", rec.frames, rep, miles)
        if row is not None:
            row.update(miles=miles, render_s=rec.timing["render"])
            self.files.append(row)
        return row

    def run_demo(self, di: int, name: str) -> dict:
        if not self.args.record:
            return self._run_demo(di, name)
        self.sim.start_recording(self.args.out / "recordings", f"miles_{name}")
        try:
            return self._run_demo(di, name)
        finally:
            self.sim.close_recording(f"miles demo {name} done")

    def _run_demo(self, di: int, name: str) -> dict:
        args, sim, ctx = self.args, self.sim, self.ctx
        ep, set_name, vid = DEMOS[name]
        start_file = GV / set_name / f"{vid}_osc.npz"
        demo = dp.load_demo(ep)
        with np.load(ep, allow_pickle=True) as d:
            demo.update(seg=d["pcd_belt_seg"], belt=np.asarray(d["sim_belt_xyz"], np.float64),
                        state=np.asarray(d["state"]), pcd_belt=np.asarray(d["pcd_belt"]))
        marks = {k: demo[k] for k in ("move0", "latch", "first_hold")}
        last = len(demo["ee_f"]) - 1
        if args.waypoints:
            wps = sorted(set(args.waypoints))
        elif args.wp_motion_mm:
            wps = motion_waypoint_frames(demo, marks, last, args.wp_motion_mm, args.wp_hold)
        else:
            wps = waypoint_frames(marks, last, args.wp_stride, args.wp_stride_contact)
        start = {"set": set_name, "id": vid, "file": str(start_file.relative_to(REPO_ROOT))}
        drow = {"demo": name, "episode": str(ep), "episode_sha256": sha256_file(ep),
                "start": start, "marks": marks, "waypoints": wps, "R": None, "R_why": None,
                "cycles": 0, "valid": 0, "walk_s": 0.0, "cycle_s": 0.0, "skipped": []}
        self.demos.append(drow)
        if self.encoder is not None:
            demo["z"] = self.demo_latents(demo, wps)
        ctx.cloud_rng = np.random.default_rng([zlib.crc32(name.encode()), cli.CLOUD_NOISE_RNG])
        sim.restore(sim_snapshot.load(start_file), settle_steps=0)
        grasp = sim.settle(round(START_SETTLE_S / sim.frame_dt))
        self.robot_s += START_SETTLE_S
        self.resets.append({"demo": name, "kind": "human", "reason": "demo start snapshot",
                            "held": [bool(v) for v in grasp.held()]})
        if grasp.held() != (True, True):
            drow.update(R=wps[0], R_why=f"grasp lost after the restore ({grasp.describe()})")
            return drow
        home_f, home_u = ee_now(sim)
        tf, tu, yaw = dp.relative_targets(demo, marks["move0"], last, home_f, home_u,
                                          cli.pulley_frame(sim)["axis"])
        drow["yaw_deg"] = yaw
        clr = Clearance(sim, ctx.plate_top_z)
        cur = marks["move0"]
        for wi, k in enumerate(wps):
            j0, j1 = cur - marks["move0"], k - marks["move0"]
            phase = "free" if k < marks["latch"] else "contact"
            if k > cur:  # walk the demo cur -> k
                pf, pu = dp.KnotPath(tf[j0:j1 + 1], dp.FRAME_DT_S), dp.KnotPath(tu[j0:j1 + 1],
                                                                                 dp.FRAME_DT_S)
                c0 = cur
                rec = FileRec(sim, ctx, lambda tau: 0,
                              lambda tau, c0=c0: c0 + round(tau / dp.FRAME_DT_S))
                t0 = time.perf_counter()
                rep = drive(sim, ctx, pf, pu, round(pf.end / sim.frame_dt), rec)
                self.robot_s += rep["steps"] * sim.frame_dt
                drow["walk_s"] += rep["steps"] * sim.frame_dt
                miles = {"demo": name, "segment": "walk", "from_frame": cur, "to_frame": k,
                         "start": start, "abort": rep["abort"]}
                self._file(f"{name}_walk_{cur:03d}_{k:03d}", rec, rep, miles)
                logger.info(f"[MILES] {name} walk {cur} -> {k}: {len(rec.frames)} frames, "
                            f"{time.perf_counter() - t0:.1f} s wall"
                            + (f", ABORT {rep['abort']}" if rep["abort"] else ""))
                if rep["abort"]:
                    drow.update(R=k, R_why=f"walk: {rep['abort']}")
                    return drow
                cur = k
            W_f, W_u = tf[j1], tu[j1]
            arrival = self.measure(demo, k, phase, None)
            arr_err = [dp.pose_err(arrival["ee_f"], W_f), dp.pose_err(arrival["ee_u"], W_u)]
            snap = sim_snapshot.capture(sim, f"miles_{name}_{k}") if args.diag_restore else None
            box = args.box_free if phase == "free" else args.box_contact
            for c in range(args.cycles):
                mode = args.modes[self.n_cycles % len(args.modes)]
                self.n_cycles += 1
                rng = np.random.default_rng([args.seed, di, k, c])
                off = draw_offset(rng, mode, box, args.emphasis_frac)
                sc_f = clr.scale("franka", W_f, off["franka"])
                sc_u = clr.scale("ur", W_u, off["ur"])
                applied = {"franka": off["franka"] * sc_f[0], "ur": off["ur"] * sc_u[0]}
                D_f, D_u = offset_pose(W_f, applied["franka"]), offset_pose(W_u, applied["ur"])
                pf = CyclePath(W_f, D_f, args.dev_s, args.ret_s, args.hold_s)
                pu = CyclePath(W_u, D_u, args.dev_s, args.ret_s, args.hold_s)
                rec = FileRec(sim, ctx, lambda tau, p=pf: p.segment(tau), lambda tau, k=k: k)
                t0 = time.perf_counter()
                rep = drive(sim, ctx, pf, pu, round(pf.end / sim.frame_dt), rec)
                dt_robot = rep["steps"] * sim.frame_dt
                self.robot_s += dt_robot
                drow["cycle_s"] += dt_robot
                m = self.measure(demo, k, phase, arrival)
                e_f, e_u = dp.pose_err(m["ee_f"], W_f), dp.pose_err(m["ee_u"], W_u)
                reach_ok = (max(e_f[0], e_u[0]) <= args.reach_mm
                            and max(e_f[1], e_u[1]) <= args.reach_deg)
                grasp_ok = m["grasp"] == [True, True]
                g = m["gt"]
                g["disturbed_vs_demo"] = g["disturbed"]
                g["disturbed_vs_arrival"] = bool(
                    g["label"] != arrival["gt"]["label"]
                    or g["belt_chamfer_vs_arrival_mm"] > self.args.gt_tau[phase])
                on_arrival = args.belt_ref == "arrival"
                g["disturbed"] = g["disturbed_vs_arrival" if on_arrival else "disturbed_vs_demo"]
                check = m["cloud"]["chamfer_vs_arrival_mm" if on_arrival else "chamfer_mm"]
                belt_ok = check <= self.tau[phase]
                why = (rep["abort_kind"] if rep["abort"] else "grasp_lost" if not grasp_ok
                       else "unreachable" if not reach_ok else "disturbed" if not belt_ok
                       else None)
                cyc = {
                    "demo": name, "waypoint": k, "phase": phase, "cycle": c, "mode": mode,
                    "offset_drawn": {a: off[a].tolist() for a in ("franka", "ur")},
                    "offset": {a: applied[a].tolist() for a in ("franka", "ur")},
                    "single_axis": off["single_axis"], "box": list(box),
                    "scale": {"franka": sc_f[0], "ur": sc_u[0]},
                    "clearance_model_mm": {"franka": [sc_f[1], sc_f[2]], "ur": [sc_u[1], sc_u[2]]},
                    "abort": rep["abort"], "steps": rep["steps"], "robot_s": dt_robot,
                    "reach": {"err_mm": [e_f[0], e_u[0]], "err_deg": [e_f[1], e_u[1]],
                              "arrival_err_mm": [arr_err[0][0], arr_err[1][0]],
                              "vs_arrival_mm": [dp.pose_err(m["ee_f"], arrival["ee_f"])[0],
                                                dp.pose_err(m["ee_u"], arrival["ee_u"])[0]],
                              "ok": reach_ok},
                    "grasp": {"held": m["grasp"], "ok": grasp_ok},
                    "belt": {**m["cloud"], "check_mm": check, "ref": args.belt_ref, "ok": belt_ok,
                             "arrival_chamfer_mm": arrival["cloud"]["chamfer_mm"]},
                    "gt": {**m["gt"], "arrival_belt_chamfer_mm": arrival["gt"]["belt_chamfer_mm"],
                           "arrival_label": arrival["gt"]["label"]},
                    "valid": why is None, "why": why,
                }
                if "latent_dist_w" in m:
                    cyc["latent_dist_w"] = m["latent_dist_w"]
                    cyc["arrival_latent_dist_w"] = arrival.get("latent_dist_w")
                fname = f"{name}_wp{k:03d}_c{c}"
                row = self._file(fname, rec, rep, {
                    "demo": name, "segment": "cycle", "waypoint": k, "cycle": c, "mode": mode,
                    "offset": cyc["offset"], "valid": cyc["valid"], "why": why,
                    "tau_mm": self.tau[phase], "start": start})
                cyc["file"] = None if row is None else row["file"]
                cyc["wall_s"] = time.perf_counter() - t0
                self.cycles.append(cyc)
                drow["cycles"] += 1
                drow["valid"] += cyc["valid"]
                logger.info(
                    f"[MILES] {name} wp {k} c{c} {mode:<6} |dF| "
                    f"{np.linalg.norm(applied['franka'][:3]):4.1f} |dU| "
                    f"{np.linalg.norm(applied['ur'][:3]):4.1f} mm: reach {e_f[0]:.2f}/{e_u[0]:.2f}"
                    f" mm, chamfer vs {args.belt_ref} {check:.2f} (tau {self.tau[phase]:.2f}), "
                    f"GT {m['gt']['belt_chamfer_mm']:.2f} {m['gt']['label']} -> "
                    f"{'valid' if why is None else why.upper()}, {cyc['wall_s']:.1f} s wall")
                if why is not None:
                    skip = (why == "disturbed" and args.on_disturb == "skip-waypoint"
                            or why == "unreachable" and args.on_unreachable == "skip-waypoint")
                    if skip and snap is None:
                        drow["skipped"].append({"waypoint": k, "after_cycle": c, "why": why,
                                                "cycles_left": args.cycles - c - 1})
                        break  # the walk goes on from the disturbed state
                    if drow["R"] is None:
                        drow.update(R=k, R_why=why)
                    if snap is None:
                        return drow
                    sim.restore(snap, settle_steps=0)
                    self.resets.append({"demo": name, "kind": "human", "waypoint": k,
                                        "reason": f"diag restore after {why}"})
        return drow


# --- outputs ----------------------------------------------------------------------------------

def buffer_rows(col: Collector, R: dict) -> list[dict]:
    skipped = {d["demo"]: [x["waypoint"] for x in d["skipped"]] for d in col.demos}
    out = []
    for i, f in enumerate(col.files):
        m = f["miles"]
        out.append({
            "id": f"miles_{col.args.out.name}_{i:04d}", "iter": -1, "kind": "miles",
            "status": "ok", "weight": 1.0, "episode": str((col.dir / f["file"]).resolve()),
            "label_human": False,
            "label_human_detail": {"success": False, "rule": "n/a",
                                   "source": "MILES segment, not a task attempt"},
            "label_strict": False, "cloud_success": None, "cloud_success_role": "monitor only",
            "start": {**m["start"], "from_reset": False, "jitter": None},
            "reset_before": "human" if i == 0 or col.files[i - 1]["miles"]["demo"] != m["demo"]
            else "none",
            "n_frames": f["n_frames"], "n_mpc_frames": 0, "n_prehold_frames": 0,
            "outcome_gt": f["outcome_gt"], "fallback_demo": False, "segment": m["segment"], "segments": f["segments"],
            "miles": {"demo": m["demo"], "segment": m["segment"],
                      "waypoint": m.get("waypoint"), "from_frame": m.get("from_frame"),
                      "to_frame": m.get("to_frame"), "cycle": m.get("cycle"),
                      "mode": m.get("mode"), "offset": m.get("offset"),
                      "valid": m.get("valid"), "why": m.get("why"), "tau_mm": m.get("tau_mm"),
                      "R": R.get(m["demo"]), "skipped_waypoints": skipped.get(m["demo"])},
        })
    return out


def budget(col: Collector, wall_s: float) -> dict:
    return {"rollouts_miles_cycles": len(col.cycles),
            "rollouts_miles_walks": sum(f["miles"]["segment"] == "walk" for f in col.files),
            "rollouts_mpc": 0, "rollouts_eval": 0,
            "resets_human": len(col.resets),
            "resets_human_diag": sum("diag" in r["reason"] for r in col.resets),
            "resets_scripted": 0, "robot_s": col.robot_s,
            "frames": sum(f["n_frames"] for f in col.files), "files": len(col.files),
            "gpu_s": 0.0, "wall_s": wall_s,
            "notes": "robot_s = sim seconds under control (start settle + walks + cycles); "
                     "resets_human = demo start restores (+ diagnostic restores)"}


def loader_check(files: list[Path]) -> dict:
    """lcs_learning's deleted-keys loader test on the first file with >= 7 frames and
    ``load_belt_seg_episode`` on every file (CPU, the lcs_learning venv)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    out = {"test": None, "load_all": None}
    pick = next((f for f in files if _n_frames(f) >= LOADER_MIN_FRAMES), None)
    if pick is not None:
        r = subprocess.run([str(PY_LCS), str(LOADER_TEST), str(pick)], cwd=LCS_ROOT, env=env,
                           capture_output=True, text=True, check=False)
        out["test"] = {"file": pick.name, "rc": r.returncode,
                       "pass": r.returncode == 0 and r.stdout.rstrip().endswith("PASS"),
                       "tail": r.stdout.strip().splitlines()[-3:] + r.stderr.strip()
                       .splitlines()[-3:]}
    code = ("import sys\nsys.path.insert(0, sys.argv[1])\n"
            "from lcs_learning.dataset_round_belt_tuples import load_belt_seg_episode\n"
            "n = 0\nfor f in sys.argv[2:]:\n    ep = load_belt_seg_episode(f, 256)\n"
            "    n += len(ep['u']) - 1\nprint(len(sys.argv) - 2, n)\n")
    r = subprocess.run([str(PY_LCS), "-c", code, str(LCS_ROOT), *map(str, files)], env=env,
                       capture_output=True, text=True, check=False)
    ok = r.returncode == 0
    out["load_all"] = {"rc": r.returncode, "pass": ok,
                       "files": int(r.stdout.split()[0]) if ok else None,
                       "pairs": int(r.stdout.split()[1]) if ok else None,
                       "err": None if ok else r.stderr.strip().splitlines()[-3:]}
    return out


def _n_frames(path: Path) -> int:
    with np.load(path, allow_pickle=True) as d:
        return int(d["actions"].shape[0])


def _f(v, nd=2) -> str:
    return "-" if v is None else f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_report(out: Path, args, col: Collector, spread: dict, bud: dict, meta: dict,
                 loader: dict | None) -> None:
    cyc = col.cycles
    L = [f"# MILES seed `{out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}`",
         "", (f"Created {meta['created']}, git {meta['git'].get('commit', '?')[:10]}, seed "
             f"{args.seed}, LCM `{args.lcm_url}`, wall {bud['wall_s'] / 60:.1f} min, robot "
             f"{bud['robot_s']:.1f} s sim, {bud['files']} files / {bud['frames']} frames."), "",
         f"Cycles: {args.cycles} per waypoint, dev {args.dev_s:g} s (min-jerk) + return "
         f"{args.ret_s:g} s (straight line, min-jerk) + hold {args.hold_s:g} s; box free "
         f"+-{args.box_free[0]:g} mm / +-{args.box_free[1]:g} deg, contact +-"
         f"{args.box_contact[0]:g} mm / +-{args.box_contact[1]:g} deg; single-axis draws "
         f"{args.emphasis_frac:g} (Franka x/y, UR x/y/z at the edge); clearance scaling >= "
         f"min({DEV_CLEAR_MM:g} mm, d at the waypoint), Franka tip >= {TIP_MIN_MM:g} mm. Reach "
         f"{args.reach_mm:g} mm / {args.reach_deg:g} deg vs the waypoint target; belt check vs "
         f"the {'demo at k' if args.belt_ref == 'demo' else 'waypoint arrival cloud'}; on "
         f"disturbed: {args.on_disturb}."
         + (" `--diag-restore` on (sim-only)." if args.diag_restore else ""), "",
         "## Disturbance threshold", "", spread["rule"] + ".", "",
         "| measure | phase | n | p50 | p95 | max | tau |", "|---|---|---|---|---|---|---|"]
    for m, tk in (("cloud", "tau_mm"), ("gt", "gt_tau_mm")):
        for ph in ("free", "contact"):
            s = spread[m][ph]
            L.append(f"| {'Chamfer (check)' if m == 'cloud' else 'GT belt Chamfer (report)'} | "
                     f"{ph} | {s['n']} | {s['p50']:.2f} | {s['p95']:.2f} | {s['max']:.2f} | "
                     f"{spread[tk][ph]:.2f} |")
    if args.tau_mm is not None:
        L += ["", f"`--tau-mm {args.tau_mm:g}` overrides the check tau (both phases)."]
    L += ["", "## Demos", "", ("| demo | waypoints | cycles | valid | skipped waypoints | R | why "
                               "| walk s | cycle s |"), "|---|---|---|---|---|---|---|---|---|"]
    for d in col.demos:
        sk = ", ".join(f"{x['waypoint']} (c{x['after_cycle']})" for x in d["skipped"]) or "-"
        L.append(f"| {d['demo']} | {len(d['waypoints'])} ({d['waypoints'][0]}..."
                 f"{d['waypoints'][-1]}) | {d['cycles']} | {d['valid']} | {sk} | {d['R']} | "
                 f"{d['R_why'] or '-'} | {d['walk_s']:.1f} | {d['cycle_s']:.1f} |")
    L += ["", "## Per waypoint", "",
          ("| demo | k | phase | cycles | valid | disturbed | grasp lost | unreachable | GT "
          "disturbed | check max | GT Chamfer vs demo max | reach max F/U mm |"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (dn, k), grp in itertools.groupby(cyc, key=lambda c: (c["demo"], c["waypoint"])):
        g = list(grp)
        why = Counter(c["why"] for c in g)
        L.append(f"| {dn} | {k} | {g[0]['phase']} | {len(g)} | {why[None]} | "
                 f"{why['disturbed']} | {why['grasp_lost']} | {why['unreachable']} | "
                 f"{sum(c['gt']['disturbed'] for c in g)} | "
                 f"{max(c['belt']['check_mm'] for c in g):.2f} | "
                 f"{max(c['gt']['belt_chamfer_mm'] for c in g):.2f} | "
                 f"{max(c['reach']['err_mm'][0] for c in g):.2f}/"
                 f"{max(c['reach']['err_mm'][1] for c in g):.2f} |")
    tp = sum((not c["belt"]["ok"]) and c["gt"]["disturbed"] for c in cyc)
    fp = sum((not c["belt"]["ok"]) and not c["gt"]["disturbed"] for c in cyc)
    fn = sum(c["belt"]["ok"] and c["gt"]["disturbed"] for c in cyc)
    tn = sum(c["belt"]["ok"] and not c["gt"]["disturbed"] for c in cyc)
    ref = "the demo at k" if args.belt_ref == "demo" else "the waypoint arrival"
    L += ["", (f"## Belt check vs sim GT (every cycle; check = cloud Chamfer vs {ref} > tau; GT "
               f"disturbed = GT label differs from {ref} or GT belt Chamfer vs {ref} > gt_tau)"), "", "| | GT disturbed | GT not disturbed |", "|---|---|---|",
          f"| check disturbed | {tp} | {fp} |", f"| check not disturbed | {fn} | {tn} |", "",
          f"Agreement {tp + tn}/{len(cyc)}." if cyc else "No cycles.", "", "## Cycles", "",
          ("| demo | k | c | mode | 1-axis | dF mm | dU mm | rot F/U deg | scale F/U | reach F/U "
           "mm | check | cloud vs demo (arrival) | GT vs demo (arrival) | GT vs arrival | GT label "
           "(arrival / demo) | latent | valid | why | wall s | file |"), "|" + "---|" * 20]
    for c in cyc:
        o = c["offset"]
        L.append(
            f"| {c['demo']} | {c['waypoint']} | {c['cycle']} | {c['mode']} | "
            f"{'y' if c['single_axis'] else 'n'} | {np.round(o['franka'][:3], 1).tolist()} | "
            f"{np.round(o['ur'][:3], 1).tolist()} | {np.linalg.norm(o['franka'][3:]):.1f}/"
            f"{np.linalg.norm(o['ur'][3:]):.1f} | {c['scale']['franka']:.2f}/"
            f"{c['scale']['ur']:.2f} | {c['reach']['err_mm'][0]:.2f}/"
            f"{c['reach']['err_mm'][1]:.2f} | {c['belt']['check_mm']:.2f} | "
            f"{c['belt']['chamfer_mm']:.2f} ({c['belt']['arrival_chamfer_mm']:.2f}) | "
            f"{c['gt']['belt_chamfer_mm']:.2f} ({c['gt']['arrival_belt_chamfer_mm']:.2f}) | "
            f"{c['gt']['belt_chamfer_vs_arrival_mm']:.2f} | {c['gt']['label']} "
            f"({c['gt']['arrival_label']} / {c['gt']['label_demo']}) | "
            f"{_f(c.get('latent_dist_w'))} | {c['valid']} | {c['why'] or '-'} | "
            f"{c['wall_s']:.1f} | {c['file']} |")
    walls = [c["wall_s"] for c in cyc]
    if walls:
        L += ["", (f"Timing: {len(cyc)} cycles, wall {np.mean(walls):.1f} s mean / "
                  f"{max(walls):.1f} s max per cycle (robot "
                  f"{np.mean([c['robot_s'] for c in cyc]):.2f} s each).")]
    if loader is not None:
        t, a = loader["test"], loader["load_all"]
        L += ["", "## Loader", "",
              (f"- `test_belt_seg_loader.py` on {t['file'] if t else '-'}: "
              f"{'PASS' if t and t['pass'] else 'FAIL' if t else 'not run'}"),
              (f"- `load_belt_seg_episode` on all {len(col.files)} files: "
              f"{'PASS' if a['pass'] else 'FAIL'} ({a['files']} files, {a['pairs']} pairs)")]
    L += ["", "## Budget", "", "```json", json.dumps(bud, indent=1), "```", ""]
    (out / "miles_report.md").write_text("\n".join(L))


# --- CLI --------------------------------------------------------------------------------------

def _pair(text: str) -> tuple[float, float]:
    a, b = (float(x) for x in text.split(","))
    return a, b


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="new dir")
    p.add_argument("--lcm-url", required=True)
    p.add_argument("--demos", default=",".join(DEMOS), help=f"comma list of {sorted(DEMOS)}")
    p.add_argument("--waypoints", type=lambda s: [int(x) for x in s.split(",")], default=None,
                   help="explicit demo frames (default: the stride rule)")
    p.add_argument("--wp-stride", type=int, default=5)
    p.add_argument("--wp-stride-contact", type=int, default=3)
    p.add_argument("--wp-motion-mm", type=float, default=None,
                   help="motion-based spacing: a waypoint per this much summed EE path (mm)")
    p.add_argument("--wp-hold", type=int, default=4, help="--wp-motion-mm: waypoints in the rest")
    p.add_argument("--cycles", type=int, default=4)
    p.add_argument("--dev-s", type=float, default=0.6)
    p.add_argument("--ret-s", type=float, default=0.6)
    p.add_argument("--hold-s", type=float, default=0.2)
    p.add_argument("--box-free", type=_pair, default=(15.0, 4.0), metavar="MM,DEG")
    p.add_argument("--box-contact", type=_pair, default=(6.0, 2.0), metavar="MM,DEG")
    p.add_argument("--emphasis-frac", type=float, default=0.5)
    p.add_argument("--modes", type=lambda t: tuple(m for m in t.split(",") if m), default=MODES,
                   help=f"cycle modes in rotation, from {MODES}")
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--tau-mm", type=float, default=None, help="override the check tau")
    p.add_argument("--tau-pct", type=float, default=95.0)
    p.add_argument("--belt-ref", choices=("demo", "arrival"), default="demo",
                   help="check reference: the demo's cloud at k (plan) or the cloud measured "
                        "on arrival at the waypoint before its cycles")
    p.add_argument("--reach-mm", type=float, default=2.0)
    p.add_argument("--reach-deg", type=float, default=1.0)
    p.add_argument("--deploy", type=Path, default=None, help="latent distance (report-only)")
    p.add_argument("--record", action="store_true",
                   help="a replay_viewer recording per demo under <out>/recordings/")
    p.add_argument("--on-unreachable", choices=("stop", "skip-waypoint"), default="stop",
                   help="skip-waypoint: a return that misses the reach tolerance ends that "
                        "waypoint's cycles; the walk re-targets the demo (grasp loss still stops)")
    p.add_argument("--on-disturb", choices=("stop", "skip-waypoint"), default="stop",
                   help="skip-waypoint: a disturbed cycle ends that waypoint's cycles and the "
                        "walk continues; stop only on grasp loss / unreachable")
    p.add_argument("--diag-restore", action="store_true",
                   help="sim-only diagnostic: restore the waypoint after an invalid cycle")
    p.add_argument("--osc-timeout-s", type=float, default=5.0)
    p.add_argument("--no-loader-check", dest="loader_check", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = create_parser().parse_args()
    cli.configure_logging()
    harness.check_private_url(args.lcm_url)
    names = [x.strip() for x in args.demos.split(",") if x.strip()]
    if unknown := [x for x in names if x not in DEMOS]:
        raise SystemExit(f"--demos {unknown}: not in {sorted(DEMOS)}")
    out = Path(args.out).resolve()
    args.out = out
    t_spread = time.perf_counter()
    spread = demo_spread(TAU_BUFFERS, args.tau_pct)
    tau = ({ph: float(args.tau_mm) for ph in ("free", "contact")} if args.tau_mm is not None
           else spread["tau_mm"])
    args.gt_tau = spread["gt_tau_mm"]
    logger.info(f"[MILES] tau {tau} mm (p{args.tau_pct:g} of {spread['pairs']} same-grasp demo "
                f"pairs, {time.perf_counter() - t_spread:.1f} s), GT tau {args.gt_tau}")
    for di, name in enumerate(names):
        demo = dp.load_demo(DEMOS[name][0])
        marks = {k: demo[k] for k in ("move0", "latch", "first_hold")}
        last = len(demo["ee_f"]) - 1
        wps = args.waypoints or (
            motion_waypoint_frames(demo, marks, last, args.wp_motion_mm, args.wp_hold)
            if args.wp_motion_mm else
            waypoint_frames(marks, last, args.wp_stride, args.wp_stride_contact))
        print(f"{name}: {DEMOS[name][0].relative_to(REPO_ROOT)} marks {marks}, {len(wps)} "
              f"waypoints {wps}, {len(wps) * args.cycles} cycles, ~"
              f"{len(wps) * args.cycles * (args.dev_s + args.ret_s + args.hold_s):.0f} s robot")
    if args.dry_run:
        for di, name in enumerate(names):
            rng = np.random.default_rng([args.seed, di, 20, 0])
            d = draw_offset(rng, args.modes[0], args.box_free, args.emphasis_frac)
            print(f"  {name} first draw (free, {args.modes[0]}): franka "
                  f"{np.round(d['franka'], 2).tolist()} ur {np.round(d['ur'], 2).tolist()}")
        return 0
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} exists and is not empty; pick a new dir")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_info()}
    encoder = z_std = None
    if args.deploy is not None:
        from task_common.latent_encoder import LatentEncoder, LearnedLcs

        encoder, z_std = LatentEncoder.load(args.deploy), LearnedLcs.load(args.deploy).z_std
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=cli.FIRST, last=cli.LAST)
    osc_binary, osc_root = harness.default_osc_binary()
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=args.osc_timeout_s)
    t_run = time.perf_counter()
    col = None
    index = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
             **meta, "tau_mm": tau, "spread": spread, "phase_labels": PHASE_LABELS,
             "cmd_source": CMD_MILES}
    try:
        warm = sim.start_osc(out / "osc.log", binary=osc_binary, cwd=osc_root)
        index["osc"] = {**sim.osc.describe(), "warm_up_s": warm}
        sim.restore(sim_snapshot.load(GV / DEMOS[names[0]][1] / f"{DEMOS[names[0]][2]}_osc.npz"))
        jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
        board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
        plate = board.plate
        cloud = cli.cloud_options(argparse.Namespace(cloud_noise="off", belt_seg=True,
                                                     no_pcd=False))
        ctx = argparse.Namespace(
            args=argparse.Namespace(no_pcd=False), mode="miles", encoder=None, sim=sim,
            board=board, gripper=gripper, thresholds=DEFAULT_THRESHOLDS,
            tangent=pert.belt_tangent(nominal), cloud_rng=None,
            cloud={**cloud, "origins": cli.sim_camera_origins(sim)},
            plate_top_z=float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half))
        col = Collector(args, sim, ctx, tau, encoder, z_std)
        for di, name in enumerate(names):
            try:
                col.run_demo(di, name)
            except OscTimeout as exc:
                if not exc.process_alive:
                    raise
                sim.bridge.forget_pending()
                d = col.demos[-1]
                d.update(R=d["R"] or -1, R_why=f"osc timeout: {exc}")
                logger.warning(f"[MILES] {name}: osc timeout ({exc}); next demo")
            write_json(out / "index.json", {**index, "demos": col.demos, "cycles": col.cycles})
    finally:
        sim.close("finished")
    wall = time.perf_counter() - t_run
    R = {d["demo"]: d["R"] for d in col.demos}
    rows = buffer_rows(col, R)
    bud = budget(col, wall)
    write_json(out / "buffer.json", {"version": 2, "iter": -1, "n": len(rows),
                                     "source": "MILES seed (deviate-and-return cycles)",
                                     "rollouts": rows})
    write_json(out / "budget.json", bud)
    loader = loader_check([col.dir / f["file"] for f in col.files]) if (
        args.loader_check and col.files) else None
    index.update(demos=col.demos, cycles=col.cycles, files=col.files, resets=col.resets,
                 budget=bud, loader=loader, wall_s=wall)
    write_json(out / "index.json", json.loads(json.dumps(index, default=lcs.json_default)))
    write_report(out, args, col, spread, bud, meta, loader)
    logger.info(f"[MILES] {len(col.cycles)} cycles ({sum(c['valid'] for c in col.cycles)} "
                f"valid), R {R}, skipped "
                f"{ {d['demo']: [x['waypoint'] for x in d['skipped']] for d in col.demos} }, robot {bud['robot_s']:.1f} s, {bud['frames']} frames, wall "
                f"{wall / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
