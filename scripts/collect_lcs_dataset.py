#!/usr/bin/env python3
"""Collect LCS episodes of the round-belt engagement segment ``pre_place_1 -> place_3``.

Each episode restores the ``pre_place_1`` start state (belt held by both arms), perturbs the
``pre_place_2`` / ``place_3`` waypoints per a sampled intent (``round_belt_task.perturbation``),
plays the motion, samples every ``--sample-period`` (default 0.075 s, the MPC knot spacing) with a
camera point cloud, labels the outcome (``round_belt_task.outcome``) and writes
``<out>/episode_XXXX.npz`` in the lcs_learning format (``docs/lcs-dataset.md``) plus
``<out>/index.json``.

``--backend osc`` (default): the Franka is torque-driven by magna's Cartesian OSC (child process
on the private ``--lcm-url``, lock-step), commanded by an emulation of magna's waypoint generator
(``round_belt_task.commander``) from the MEASURED pose every control step; the UR follows magna's
2-knot line by per-step IK. ``state`` holds the measured poses, ``action`` (``--action-definition``)
= ``cmd_delta`` (default): the commanded per-dt displacement, Franka knot 1 - knot 0 of the command
published at the sample tick, UR line(t + dt) - line(t) of the line in force; or
``knot1_minus_measured``: knot 1 / line(t + dt) minus the measured pose. Both, the raw commands
(``sim_cmd_*``) and the realised ``measured_{t+1} - measured_t`` are always stored. Each episode
starts with ``--pre-hold-s`` of hold frames (``sim_episode_step < 0``, phase ``prehold``, u = 0).
Optional OU excitation of Franka knots 1..6 and of the UR target pose (``--excite-ur-*``); the
UR offset is scaled down per sample so the excited target keeps the guard's min clearance.
The board clearance guard models only the 2F-85 (UR); the Franka is bounded by the excitation
cap + z-floor and its measured finger_tip-to-plate clearance is recorded.

``--backend position``: both arms position-driven along a precomputed joint trajectory, ``state``
= commanded targets, ``action`` = cmd(t+1) - cmd(t). No magna, no LCM.

Run:
    uv run python scripts/collect_lcs_dataset.py --episodes 40 --seed 0
    uv run python scripts/collect_lcs_dataset.py --episodes 4 --seed 1 --dry-run
    uv run python scripts/collect_lcs_dataset.py --episodes 4 --record --out data/lcs/smoke
    uv run python scripts/collect_lcs_dataset.py --backend position --episodes 4
    uv run python scripts/collect_lcs_dataset.py --episodes 24 \\
        --start-states data/lcs/start_states/grasp_variants/set2 --variants gv_01,gv_02
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from loguru import logger

from round_belt_task import clearance as clr
from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import (
    FrankaTip,
    UrTracking,
    ik,
    mat3_to_quat_xyzw,
    pose7_from_mat,
    quat_xyzw_to_mat3,
    rot_axis_angle,
)
from round_belt_task.commander import (
    EXCITE_MODES,
    EXCITE_RAMP_S,
    CommanderParams,
    FrankaCommand,
    FrankaTarget,
    FrankaWaypointCommander,
    OuExcitation,
    UrCommand,
    UrLine,
    UrLineCommander,
    UrTarget,
    angular_distance,
    draw_excite,
    mat3_to_quat,
    parse_saved_traj_message,
    pose_mat,
    quat_axis_angle,
    quat_to_mat3,
    saved_traj_message,
    targets_from_waypoints,
    x_tool0_tracking,
)
from round_belt_task.constants import ARM_TARGET_KD, ARM_TARGET_KE
from round_belt_task.episode_io import finish_recording as _finish_recording
from round_belt_task.episode_io import git_info as _git
from round_belt_task.episode_io import pose7_mat as _pose7_mat
from round_belt_task.episode_io import sha256_file as _sha256
from round_belt_task.episode_io import slant_row as _slant_row
from round_belt_task.episode_io import write_json as _write_json
from round_belt_task.motion import MotionError, build_cartesian_trajectory
from round_belt_task.osc_bridge import OscTimeout
from round_belt_task.outcome import (
    DEFAULT_THRESHOLDS,
    LABELS,
    OutcomeThresholds,
    classify,
    classify_episode,
    frame_metrics,
    slant_episode,
    slant_metrics,
)
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.osc_process import check_private_url

FIRST, LAST = "pre_place_1", "place_3"
BACKENDS = ("position", "osc")
DEFAULT_START_STATES = {"position": sim_snapshot.DEFAULT_START_STATE_DIR / f"{FIRST}.npz",
                        "osc": sim_snapshot.DEFAULT_START_STATE_DIR / f"{FIRST}_osc.npz"}
DEFAULT_START_STATE = DEFAULT_START_STATES["position"]
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "lcs"
DEFAULT_LCM_URL = "udpm://239.255.76.83:7683?ttl=0"
RECORD_STATE_EVERY = 4
CLEARANCE_EVERY = 4  # 20 ms; the UR moves <= 1.6 mm between checks
PATH_CLAMP_ITERS = 3
START_PHASE = "start"
SCENARIOS = ("pure_translation", "nominal")
PURE_TRANSLATION_M = np.array([-0.03, 0.0, 0.01])
PURE_TRANSLATION_DWELL_S = 0.5
UR_IK_POS_TOL, UR_IK_ROT_TOL = 1e-4, 7.5e-4
UR_VELOCITY_LEAD = 1.0
OSC_LOG_ERRORS = ("resetting", "Exception caught")
OSC_ACTION_DEFINITION = lcs.DEFAULT_ACTION_DEFINITION
EXCITE_DEFAULTS = {"ou": (1.5, 1.0), "white": (4.0, 2.0)}  # (pos mm, rot deg)
EXCITE_UR_DEFAULTS = (2.0, 1.0)  # ou only: per-axis std (mm, deg)
UR_SCALE_BISECT = 12
PRE_HOLD_S = 1.5
PREHOLD_PHASE = "prehold"
# Opt-in approach / place_3 / tail modes (flags off = the paths above, unchanged).
APPROACH_KEYS = ("yaw_deg", "elev_mm", "offset_mm", "tilt_deg")
PLACE3_MODES = ("intents", "continuous", "engaged_band", "fixed")
PLACE3_ARM_KEYS = ("depth_mm", "normal_mm", "tangent_mm", "roll_deg", "yaw_deg")
WIDE_BOX = {"depth_mm": (-6.0, 20.0), "normal_mm": (-10.0, 10.0), "tangent_mm": (-10.0, 10.0),
            "roll_deg": (-10.0, 10.0), "yaw_deg": (-15.0, 15.0)}
BAND_YAW_DEG = 2.0
FRANKA_FLOOR_MM = 12.0
IN_CONTACT_MIN_NEIGHBOUR = 3
APPROACH_RNG, PLACE3_RNG, TAIL_RNG = 13, 14, 15
NEW_ARG_DEFAULTS = {"approach": None, "approach_range": None, "place3_mode": "intents",
                    "engaged_share": 0.40, "place3_box": None, "place3_box_ur": None,
                    "place3_values": None,
                    "tail": None, "tail_families": None, "tail_gentle": "auto",
                    "tail_cap": "3,25,3,25", "tail_stretch_cap_pct": 2.0, "row_tag": None}


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
    p.add_argument("--intents", default=",".join(pert.INTENTS),
                   help=f"comma list of {', '.join(pert.ALL_INTENTS)}")
    p.add_argument("--weights", default=None,
                   help="one weight per --intents entry (default uniform)")
    p.add_argument("--backend", choices=BACKENDS, default="osc",
                   help="osc: Franka on magna's OSC (lock-step); position: in-process PD only")
    p.add_argument("--lcm-url", default=DEFAULT_LCM_URL,
                   help="private LCM URL of the OSC (never magna's shared group)")
    p.add_argument("--osc-timeout-s", type=float, default=5.0)
    p.add_argument("--osc-settle-s", type=float, default=0.5,
                   help="hold under the OSC after each restore")
    p.add_argument("--osc-log", type=Path, default=None, help="default <out>/osc.log")
    p.add_argument("--start-state", type=Path, default=None,
                   help=f"default {DEFAULT_START_STATES['osc'].name} (osc) / "
                        f"{DEFAULT_START_STATES['position'].name} (position)")
    p.add_argument("--start-states", type=Path, default=None,
                   help="snapshot .npz or a grasp-variant set dir (index.json); episodes "
                        "round-robin over its held states (osc only, excludes --start-state)")
    p.add_argument("--variants", default=None,
                   help="comma list of variant ids of --start-states (default: all held)")
    p.add_argument("--fresh-pick", action="store_true",
                   help="position backend only: run the nominal pick per episode (slow)")
    p.add_argument("--record", action="store_true",
                   help="a RunRecorder run per episode under <out>/recordings/")
    p.add_argument("--settle-s", type=float, default=1.0)
    p.add_argument("--min-clearance", type=float, default=clr.DEFAULT_MIN_CLEARANCE_MM,
                   help="mm of 2F-85 -> board clearance the perturbed UR waypoints are clamped "
                        "to (0 disables the clamp; the clearance is measured either way)")
    p.add_argument("--sample-period", type=float, default=lcs.SAMPLE_PERIOD_S,
                   help="seconds, a multiple of the 5 ms control step (0.1 = magna's logs)")
    p.add_argument("--excite-mode", choices=EXCITE_MODES, default="ou",
                   help="osc: ou = smooth Ornstein-Uhlenbeck offset; white = a fresh draw per "
                        "sample (the pre-OU data)")
    p.add_argument("--excite-pos-mm", type=float, default=None,
                   help="osc: Franka knot 1..6 offset; ou: per-axis stationary std (default 1.5); "
                        "white: ball radius (default 4); 0 with --excite-rot-deg 0 = off")
    p.add_argument("--excite-rot-deg", type=float, default=None,
                   help="ou: per-axis rotation-vector std (default 1); white: max angle "
                        "(default 2)")
    p.add_argument("--excite-tau-s", type=float, default=0.4,
                   help="ou: correlation time")
    p.add_argument("--excite-cap-factor", type=float, default=2.0,
                   help="knot step capped to factor * speed * dt")
    p.add_argument("--excite-down-mm", type=float, default=2.0,
                   help="largest downward excitation offset")
    p.add_argument("--excite-ur-pos-mm", type=float, default=EXCITE_UR_DEFAULTS[0],
                   help="osc + ou: UR target offset per-axis std (z clipped >= 0); 0 with "
                        "--excite-ur-rot-deg 0 = off")
    p.add_argument("--excite-ur-rot-deg", type=float, default=EXCITE_UR_DEFAULTS[1],
                   help="osc + ou: UR target rotation-vector per-axis std")
    p.add_argument("--action-definition", choices=lcs.CMD_ACTION_DEFINITIONS,
                   default=lcs.DEFAULT_ACTION_DEFINITION, help="osc: the recorded actions")
    p.add_argument("--pre-hold-s", type=float, default=PRE_HOLD_S,
                   help="osc: hold frames sampled after the settle, before the first move")
    p.add_argument("--max-episode-s", type=float, default=20.0,
                   help="osc: skip an episode whose targets are not all reached by then")
    p.add_argument("--scenario", choices=SCENARIOS, default=None,
                   help="osc: pure_translation = one Franka target 30 mm -x, 10 mm +z, UR still; "
                        "nominal = the unperturbed waypoints (intents ignored, excitation off)")
    p.add_argument("--hold-ur-gripper", action="store_true",
                   help="osc: keep the start-state UR byte (waypoint bytes logged, not applied)")
    p.add_argument("--nominal-ur-dz-mm", default=None, metavar="A[:B]",
                   help="nominal scenario: UR z offset (mm) of pre_place_2 (A) and place_3 (B = A)")
    p.add_argument("--nominal-franka-offset-mm", default=None, metavar="X,Y,Z[:X,Y,Z]",
                   help="nominal scenario: Franka world offset (mm) of pre_place_2 and place_3")
    p.add_argument("--arm-ke", type=float, default=ARM_TARGET_KE)
    p.add_argument("--arm-kd", type=float, default=ARM_TARGET_KD)
    p.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML)
    p.add_argument("--no-pcd", action="store_true",
                   help="skip camera renders; files lack pcd and fail validation (speed tests)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the sampled perturbation table and exit")
    p.add_argument("--thresholds", nargs="*", default=[], metavar="KEY=VALUE",
                   help="OutcomeThresholds overrides")
    d = NEW_ARG_DEFAULTS
    p.add_argument("--approach", default=d["approach"], metavar="YAW,ELEV,OFFSET,TILT|start",
                   help="osc: rigid approach transform of pre_place_2/place_3 (deg, mm, mm, deg) "
                        "about the large pulley axis; 'start' = each start state's index entry")
    p.add_argument("--approach-range", default=d["approach_range"],
                   metavar="YLO:YHI,ELO:EHI,OLO:OHI,TLO:THI",
                   help="osc: per-episode approach ~ U[range] (RNG [seed, i, 13])")
    p.add_argument("--place3-mode", choices=PLACE3_MODES, default=d["place3_mode"],
                   help="intents: --intents (today); continuous: engaged band with "
                        "--engaged-share, else the wide box per arm; engaged_band; fixed: "
                        "--place3-values")
    p.add_argument("--engaged-share", type=float, default=d["engaged_share"])
    p.add_argument("--place3-box", default=d["place3_box"],
                   metavar="DLO:DHI,NORMAL,TANGENT,ROLL,YAW",
                   help="wide box (mm, mm, mm, deg, deg; symmetric except depth)")
    p.add_argument("--place3-box-ur", default=d["place3_box_ur"],
                   metavar="DLO:DHI,NORMAL,TANGENT,ROLL,YAW",
                   help="UR wide box (default --place3-box)")
    p.add_argument("--place3-values", default=d["place3_values"], metavar="F:U[;F:U]",
                   help="fixed mode: per arm depth,normal,tangent,roll,yaw; episode i uses "
                        "entry i %% n")
    p.add_argument("--tail", default=d["tail"], metavar="MIN_S,MAX_S",
                   help="osc: post-place_3 scripted contact tail of U[MIN_S, MAX_S] + 0.5 s hold")
    p.add_argument("--tail-families", default=d["tail_families"],
                   help="comma list of tail families to draw from (default all / gentle set)")
    p.add_argument("--tail-gentle", choices=("auto", "on", "off"), default=d["tail_gentle"],
                   help="auto: engaged-band episodes get gentle tails")
    p.add_argument("--tail-cap", default=d["tail_cap"], metavar="F_MM,F_MRAD,U_MM,U_MRAD",
                   help="per-step cmd_delta caps of the tail")
    p.add_argument("--tail-stretch-cap-pct", type=float, default=d["tail_stretch_cap_pct"])
    p.add_argument("--row-tag", default=d["row_tag"], help="recorded as row['tag']")
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


def parse_ur_dz(text: str | None) -> dict[str, float] | None:
    """``"A[:B]"`` -> ``{pre_place_2: A, place_3: B}`` in mm (B defaults to A)."""
    if text is None:
        return None
    parts = [float(v) for v in str(text).split(":")]
    if len(parts) not in (1, 2):
        raise ValueError(f"--nominal-ur-dz-mm {text!r}: want A or A:B")
    return dict(zip(pert.PERTURBED_LABELS, (parts[0], parts[-1])))


def parse_franka_offset(text: str | None) -> dict[str, list[float]] | None:
    """``"x,y,z[:x,y,z]"`` -> ``{pre_place_2: [x,y,z], place_3: [...]}`` in mm (B defaults to A)."""
    if text is None:
        return None
    parts = [[float(v) for v in part.split(",")] for part in str(text).split(":")]
    if len(parts) not in (1, 2) or any(len(v) != 3 for v in parts):
        raise ValueError(f"--nominal-franka-offset-mm {text!r}: want x,y,z or x,y,z:x,y,z")
    return dict(zip(pert.PERTURBED_LABELS, (parts[0], parts[-1])))


def hold_options(args, scenario: str | None) -> dict:
    """The opt-in UR options; ``{}`` when both are off (outputs unchanged)."""
    out = {}
    if getattr(args, "hold_ur_gripper", False):
        out["hold_ur_gripper"] = True
    dz = parse_ur_dz(getattr(args, "nominal_ur_dz_mm", None))
    if dz is not None:
        if scenario != "nominal":
            raise ValueError("--nominal-ur-dz-mm needs --scenario nominal")
        out["nominal_ur_dz_mm"] = dz
    off = parse_franka_offset(getattr(args, "nominal_franka_offset_mm", None))
    if off is not None:
        if scenario != "nominal":
            raise ValueError("--nominal-franka-offset-mm needs --scenario nominal")
        out["nominal_franka_offset_mm"] = off
    return out


def override_waypoints(waypoints: list, opts: dict) -> tuple[list, dict | None]:
    """UR z / Franka offsets and/or stripped UR bytes; ``(waypoints, stripped bytes or None)``."""
    dz = opts.get("nominal_ur_dz_mm") or {}
    off_f = opts.get("nominal_franka_offset_mm") or {}
    hold = opts.get("hold_ur_gripper", False)
    out, stripped = [], {}
    for w in waypoints:
        if w.label in dz and w.ur_pos is not None:
            w = dataclasses.replace(w, ur_pos=np.asarray(w.ur_pos, float)
                                    + np.array([0.0, 0.0, dz[w.label] * 1e-3]))
        if w.label in off_f:
            w = dataclasses.replace(w, franka_pos=np.asarray(w.franka_pos, float)
                                    + np.asarray(off_f[w.label], float) * 1e-3)
        if hold and w.ur_gripper_byte is not None:
            stripped[w.label] = int(w.ur_gripper_byte)
            w = dataclasses.replace(w, ur_gripper_byte=None)
        out.append(w)
    return out, (stripped if hold else None)


def _floats(text: str, n: int, flag: str) -> list[float]:
    v = [float(x) for x in str(text).split(",")]
    if len(v) != n:
        raise ValueError(f"{flag} {text!r}: want {n} comma-separated numbers")
    return v


def _spans(text: str, flag: str) -> list[tuple[float, float]]:
    out = []
    for part in str(text).split(","):
        lo, _, hi = part.partition(":")
        lo, hi = float(lo), float(hi if hi else lo)
        if hi < lo:
            raise ValueError(f"{flag} {text!r}: {part!r} has HI < LO")
        out.append((lo, hi))
    return out


def new_options(args) -> dict:
    """The opt-in approach / place_3 / tail options; ``{}`` when all are off."""
    out = {}
    approach = getattr(args, "approach", None)
    a_range = getattr(args, "approach_range", None)
    if approach is not None and a_range is not None:
        raise ValueError("--approach excludes --approach-range")
    if approach == "start":
        out["approach_cfg"] = {"source": "start_state"}
    elif approach is not None:
        out["approach_cfg"] = {"source": "fixed", **dict(zip(
            APPROACH_KEYS, _floats(approach, 4, "--approach"), strict=True))}
    elif a_range is not None:
        spans = _spans(a_range, "--approach-range")
        if len(spans) != 4:
            raise ValueError("--approach-range wants 4 LO:HI spans")
        out["approach_cfg"] = {"source": "range", "range": dict(zip(APPROACH_KEYS, spans))}
    mode = getattr(args, "place3_mode", "intents")
    if mode != "intents":
        box = parse_box(getattr(args, "place3_box", None), dict(WIDE_BOX), "--place3-box")
        box_ur = parse_box(getattr(args, "place3_box_ur", None), dict(box), "--place3-box-ur")
        cfg = {"mode": mode, "engaged_share": float(args.engaged_share),
               "wide_box": {k: list(v) for k, v in box.items()},
               "band": {"ranges": "OSC_RANGES['engaged']", "yaw_deg": BAND_YAW_DEG,
                        "depth": "UR dz on pre_place_2 and place_3 (as the engaged intent)"},
               "franka_floor_mm": FRANKA_FLOOR_MM, "intents_ignored": True}
        if box_ur != box:
            cfg["wide_box_ur"] = {k: list(v) for k, v in box_ur.items()}
        if mode == "fixed":
            if not getattr(args, "place3_values", None):
                raise ValueError("--place3-mode fixed needs --place3-values")
            cfg["values"] = [parse_place3_values(v) for v in args.place3_values.split(";")]
        out["place3_cfg"] = cfg
    elif getattr(args, "place3_values", None):
        raise ValueError("--place3-values needs --place3-mode fixed")
    if getattr(args, "tail", None) is not None:
        lo, hi = _floats(args.tail, 2, "--tail")
        if not 0.0 < lo <= hi:
            raise ValueError("--tail wants 0 < MIN_S <= MAX_S")
        fams = _csv(args.tail_families) if getattr(args, "tail_families", None) else None
        out["tail_cfg"] = {"min_s": lo, "max_s": hi, "families": fams,
                           "gentle": args.tail_gentle,
                           "cap": _floats(args.tail_cap, 4, "--tail-cap"),
                           "stretch_cap_pct": float(args.tail_stretch_cap_pct)}
    if getattr(args, "row_tag", None):
        out["row_tag"] = args.row_tag
    return out


def parse_box(text: str | None, box: dict, flag: str) -> dict:
    """``DLO:DHI,NORMAL,TANGENT,ROLL,YAW`` over ``box``; a single value X means +-X."""
    if not text:
        return box
    spans = _spans(text, flag)
    if len(spans) != 5:
        raise ValueError(f"{flag} wants DLO:DHI,NORMAL,TANGENT,ROLL,YAW")
    box = dict(box, depth_mm=spans[0])
    for key, (lo, hi) in zip(PLACE3_ARM_KEYS[1:], spans[1:], strict=True):
        box[key] = (-hi, hi) if lo == hi else (lo, hi)
    return box


def parse_place3_values(text: str) -> dict:
    """``"d,n,t,roll,yaw:d,n,t,roll,yaw"`` (Franka : UR) -> per-arm dict."""
    parts = str(text).split(":")
    if len(parts) != 2:
        raise ValueError(f"--place3-values {text!r}: want FRANKA:UR")
    return {arm: dict(zip(PLACE3_ARM_KEYS, _floats(v, 5, "--place3-values"), strict=True))
            for arm, v in zip(("franka", "ur"), parts, strict=True)}


def pulley_frame(sim) -> dict:
    """Large pulley origin + axis (body +z) from the current state."""
    q = sim.state_0.body_q.numpy()[int(sim.info.pulley_bodies[1])].astype(np.float64)
    R = quat_xyzw_to_mat3(q[3:7])
    return {"centre": q[:3].copy(), "axis": R[:, 2] / np.linalg.norm(R[:, 2])}


def approach_frame(a: dict, pulley: dict, ref_f: np.ndarray, ref_u: np.ndarray) -> dict:
    """Rotation about the pulley axis, the rotated tangent / horizontal normal, the tilt."""
    R = rot_axis_angle(pulley["axis"], math.radians(a["yaw_deg"]))
    t = R @ (np.asarray(ref_u, float) - np.asarray(ref_f, float))
    t[2] = 0.0
    t /= np.linalg.norm(t)
    n = np.cross([0.0, 0.0, 1.0], t)
    return {"R": R, "tangent": t, "normal": n, "R_tilt": rot_axis_angle(t, math.radians(
        a["tilt_deg"])), "centre": np.asarray(pulley["centre"], float)}


def approach_pose(X: np.ndarray, a: dict, fr: dict, elev: bool) -> np.ndarray:
    """``X`` (4x4) through the approach transform; ``elev`` adds ``elev_mm`` to z."""
    out = np.eye(4)
    c = fr["centre"]
    out[:3, 3] = (c + fr["R"] @ (X[:3, 3] - c) + fr["normal"] * a["offset_mm"] * 1e-3
                  + np.array([0.0, 0.0, a["elev_mm"] * 1e-3 if elev else 0.0]))
    out[:3, :3] = fr["R_tilt"] @ fr["R"] @ X[:3, :3]
    return out


def approach_waypoints(waypoints: list, a: dict | None, pulley: dict | None) -> list:
    """Both arms' ``pre_place_1/2`` / ``place_3`` moved rigidly; ``elev`` on all but place_3."""
    if a is None:
        return waypoints
    ref = next(w for w in waypoints if w.label == pert.TANGENT_LABEL)
    fr = approach_frame(a, pulley, ref.franka_pos, ref.ur_pos)
    out = []
    for w in waypoints:
        if w.label not in (FIRST, *pert.PERTURBED_LABELS):
            out.append(w)
            continue
        elev = w.label != LAST
        Xf = approach_pose(w.franka_mat(), a, fr, elev)
        kw = {"franka_pos": Xf[:3, 3], "franka_quat_xyzw": mat3_to_quat_xyzw(Xf[:3, :3])}
        if w.ur_pos is not None:
            Xu = approach_pose(w.ur_mat(), a, fr, elev)
            kw.update(ur_pos=Xu[:3, 3], ur_quat_xyzw=mat3_to_quat_xyzw(Xu[:3, :3]))
        out.append(dataclasses.replace(w, **kw))
    return out


def draw_approach(cfg: dict, seed: int, i: int, start: dict | None) -> dict | None:
    if cfg is None:
        return None
    if cfg["source"] == "fixed":
        return {k: cfg[k] for k in APPROACH_KEYS} | {"source": "fixed"}
    if cfg["source"] == "start_state":
        if start is None:
            raise ValueError("--approach start: the start state's index row has no 'approach'")
        return {k: float(start[k]) for k in APPROACH_KEYS} | {"source": "start_state"}
    rng = np.random.default_rng([seed, i, APPROACH_RNG])
    return {k: float(rng.uniform(*cfg["range"][k])) for k in APPROACH_KEYS} | {"source": "range"}


def draw_place3(cfg: dict, seed: int, i: int, ranges: dict) -> dict:
    """The episode's place_3 sample: ``{mode, perturbation | per-arm values}`` (world-free)."""
    rng = np.random.default_rng([seed, i, PLACE3_RNG])
    mode = cfg["mode"]
    if mode == "fixed":
        vals = cfg["values"][i % len(cfg["values"])]
        return {"mode": "fixed", **{arm: dict(v) for arm, v in vals.items()}}
    band = mode == "engaged_band" or rng.random() < cfg["engaged_share"]
    if band:
        p = pert.sample(rng, "engaged", ranges)
        yaw = rng.uniform(-BAND_YAW_DEG, BAND_YAW_DEG, 2)
        p = dataclasses.replace(p, intent="engaged_band", franka_yaw_deg=float(yaw[0]),
                                ur_yaw_deg=float(yaw[1]))
        return {"mode": "engaged_band", "perturbation": p.to_dict()}
    boxes = {"franka": cfg["wide_box"], "ur": cfg.get("wide_box_ur", cfg["wide_box"])}
    return {"mode": "wide", **{arm: {k: float(rng.uniform(*boxes[arm][k]))
                                     for k in PLACE3_ARM_KEYS} for arm in ("franka", "ur")}}


def place3_perturbation(sample: dict, tangent: np.ndarray
                        ) -> tuple[pert.Perturbation, dict[str, float]]:
    """``(Perturbation, place_3-only depth per arm [mm])`` of a :func:`draw_place3` sample."""
    if sample["mode"] == "engaged_band":
        d = sample["perturbation"]
        return pert.Perturbation(
            intent="engaged_band", ur_dpos_m=np.array(d["ur_dpos_m"]),
            ur_tilt_deg=d["ur_tilt_deg"], franka_dpos_m=np.array(d["franka_dpos_m"]),
            franka_tilt_deg=d["franka_tilt_deg"], ur_yaw_deg=d["ur_yaw_deg"],
            franka_yaw_deg=d["franka_yaw_deg"]), {}
    t = np.asarray(tangent, float)
    n = np.cross([0.0, 0.0, 1.0], t)

    def dpos(v):
        return (v["normal_mm"] * n + v["tangent_mm"] * t) * 1e-3

    f, u = sample["franka"], sample["ur"]
    p = pert.Perturbation(intent=sample["mode"], ur_dpos_m=dpos(u), ur_tilt_deg=u["roll_deg"],
                          franka_dpos_m=dpos(f), franka_tilt_deg=f["roll_deg"],
                          ur_yaw_deg=u["yaw_deg"], franka_yaw_deg=f["yaw_deg"])
    return p, {"franka": f["depth_mm"], "ur": u["depth_mm"]}


def depth_waypoints(waypoints: list, depth_mm: dict) -> list:
    """``place_3`` z per arm: depth > 0 above the nominal, < 0 pressing past it."""
    if not depth_mm:
        return waypoints
    out = []
    for w in waypoints:
        if w.label == LAST:
            dz = np.array([0.0, 0.0, 1e-3])
            w = dataclasses.replace(w, franka_pos=np.asarray(w.franka_pos, float)
                                    + depth_mm["franka"] * dz,
                                    ur_pos=None if w.ur_pos is None
                                    else np.asarray(w.ur_pos, float) + depth_mm["ur"] * dz)
        out.append(w)
    return out


def start_approaches(path: Path | None) -> dict:
    """``{variant id: approach dict}`` of a start-state set's ``index.json`` (if any)."""
    if path is None or not (Path(path) / "index.json").is_file():
        return {}
    index = json.loads((Path(path) / "index.json").read_text())
    return {str(r["id"]): r["approach"] for r in index.get("variants", []) if r.get("approach")}


def sample_steps(period_s: float) -> int:
    n = round(period_s / lcs.SIM_DT_S)
    if n < 1 or abs(n * lcs.SIM_DT_S - period_s) > 1e-9:
        raise ValueError(f"--sample-period {period_s}: not a multiple of {lcs.SIM_DT_S} s")
    return n


def plan_episodes(args, ranges=None, scenario: str | None = None) -> list[pert.Perturbation]:
    """The per-episode perturbations, a pure function of ``--seed``/``--intents``/``--weights``."""
    if (scenario or getattr(args, "scenario", None)) == "nominal":
        return [pert.Perturbation(intent="nominal") for _ in range(args.episodes)]
    ranges = pert.DEFAULT_RANGES if ranges is None else ranges
    intents = _csv(args.intents)
    weights = (np.ones(len(intents)) if args.weights is None
               else np.array([float(w) for w in _csv(args.weights)]))
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
    w = max(8, *(len(p.intent) for p in perts))
    rows = [f"  ep  {'intent':<{w}} ur dx/dy/dz mm         ur tilt  franka dx/dy/dz mm     f tilt"]
    for i, p in enumerate(perts):
        u, f = p.ur_dpos_m * 1e3, p.franka_dpos_m * 1e3
        rows.append(f"{i:4d}  {p.intent:<{w}} {u[0]:6.2f} {u[1]:6.2f} {u[2]:7.2f}  "
                    f"{p.ur_tilt_deg:7.2f}  {f[0]:6.2f} {f[1]:6.2f} {f[2]:6.2f}  "
                    f"{p.franka_tilt_deg:6.2f}")
    return "\n".join(rows)


def _intent_width(intents: list[str]) -> int:
    return max(9, *(len(i) + 1 for i in intents))


def confusion_table(rows: list[dict], intents: list[str]) -> str:
    counts = Counter((r["intent"], r["outcome"]) for r in rows if r.get("status") == "ok")
    skipped = Counter(r["intent"] for r in rows if r.get("status") != "ok")
    w = _intent_width(intents)
    head = f"{'intent':<{w}}" + "".join(f"{lab:>9}" for lab in LABELS) + f"{'skipped':>9}"
    lines = [head]
    for intent in intents:
        lines.append(f"{intent:<{w}}" + "".join(f"{counts[(intent, lab)]:>9}" for lab in LABELS)
                     + f"{skipped[intent]:>9}")
    total = Counter(r["outcome"] for r in rows if r.get("status") == "ok")
    lines.append(f"{'total':<{w}}" + "".join(f"{total[lab]:>9}" for lab in LABELS)
                 + f"{sum(skipped.values()):>9}")
    return "\n".join(lines)


def slant_table(rows: list[dict], intents: list[str]) -> str:
    """Per intent: slanted count, final slant_deg median [min, max] and slant_dir counts."""
    w = _intent_width(intents)
    lines = [f"{'intent':<{w}}{'slanted':>8}{'deg med [min, max]':>22}  slant_dir"]
    for intent in intents:
        got = [r for r in rows if r["intent"] == intent and r["outcome"] == "slanted"]
        deg = [r["final_slant_deg"] for r in got if r["final_slant_deg"] is not None]
        span = (f"{np.median(deg):6.1f} [{min(deg):5.1f}, {max(deg):5.1f}]" if deg else "-")
        dirs = Counter(r["final_slant_dir"] for r in got)
        lines.append(f"{intent:<{w}}{len(got):>8}{span:>22}  "
                     + " ".join(f"{k} {v}" for k, v in dirs.most_common()))
    return "\n".join(lines)


def clearance_table(rows: list[dict], intents: list[str]) -> str:
    """Per-intent 2F-85 -> board clearance (mm) and how much the guard had to lift."""
    w = _intent_width(intents)
    head = (f"{'intent':<{w}}{'n':>4}{'min mm':>9}{'median mm':>11}{'max lift mm':>13}"
            f"{'contacts':>10}")
    lines = [head]
    for intent in [*intents, "total"]:
        got = rows if intent == "total" else [r for r in rows if r["intent"] == intent]
        if not got:
            continue
        low = [r["min_board_clearance_mm"] for r in got]
        lift = [r["clamp_lift_mm"] for r in got]
        lines.append(f"{intent:<{w}}{len(got):>4}{min(low):>9.2f}{float(np.median(low)):>11.2f}"
                     f"{max(lift):>13.2f}{sum(r['board_contact'] for r in got):>10}")
    return "\n".join(lines)


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


def resolve_start_state(args) -> Path:
    return Path(args.start_state) if args.start_state else DEFAULT_START_STATES[args.backend]


def _resolve_file(base: Path, file: str) -> Path:
    path = Path(file)
    if path.is_absolute():
        return path
    for root in (base, REPO_ROOT):
        if (root / path).exists():
            return root / path
    return base / path


def load_start_set(path: Path, variants: str | None) -> tuple[list[tuple[str, Path]], dict]:
    """``([(id, file)], meta)`` from a snapshot ``.npz`` or a variant-set dir (``index.json``)."""
    path = Path(path)
    if path.is_file():
        if variants:
            raise ValueError("--variants needs a variant-set dir for --start-states")
        return [(path.stem, path)], {"kind": "snapshot", "path": str(path)}
    index_path = path / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"{path}: neither a snapshot .npz nor a dir with index.json")
    index = json.loads(index_path.read_text())
    by_id = {str(r["id"]): r for r in index.get("variants", [])}
    wanted = _csv(variants) if variants else list(by_id)
    missing = [v for v in wanted if v not in by_id]
    if missing:
        raise ValueError(f"variants {missing} not in {index_path} ({sorted(by_id)})")
    states = []
    for vid in wanted:
        row = by_id[vid]
        if row.get("file") is None or row.get("held") is False:
            logger.warning(f"[LCS] variant {vid} has no held start state; skipped")
            continue
        states.append((vid, _resolve_file(path, row["file"])))
    if not states:
        raise ValueError(f"{path}: no held start state among {wanted}")
    meta = {"kind": "variant_set", "path": str(path), "set": index.get("set"),
            "index_sha256": _sha256(index_path), "variants": [v for v, _ in states],
            "assignment": "round-robin: episode i uses variants[i % len(variants)]"}
    return states, meta


def excitation_config(args, scenario: str | None) -> dict:
    mode = getattr(args, "excite_mode", "white")
    pos_mm, rot_deg = EXCITE_DEFAULTS[mode]
    pos_mm = pos_mm if args.excite_pos_mm is None else args.excite_pos_mm
    rot_deg = rot_deg if args.excite_rot_deg is None else args.excite_rot_deg
    on = scenario is None and (pos_mm > 0.0 or rot_deg > 0.0)
    cfg = {"on": on, "mode": mode, "pos_mm": pos_mm, "rot_deg": rot_deg,
           "cap_factor": args.excite_cap_factor, "down_mm": args.excite_down_mm}
    if mode == "ou":
        p = CommanderParams()
        cfg.update(tau_s=args.excite_tau_s, ramp_s=EXCITE_RAMP_S,
                   fade_dist_mm=(p.lin_speed * EXCITE_RAMP_S + p.pos_tol) * 1e3,
                   pos_meaning="per-axis stationary std", rot_meaning="per-axis rotvec std")
    else:
        cfg.update(pos_meaning="ball radius", rot_meaning="max angle")
    ur_pos = getattr(args, "excite_ur_pos_mm", 0.0)
    ur_rot = getattr(args, "excite_ur_rot_deg", 0.0)
    cfg["ur"] = {
        "on": scenario is None and mode == "ou" and (ur_pos > 0.0 or ur_rot > 0.0),
        "pos_mm": ur_pos, "rot_deg": ur_rot, "z_floor_mm": 0.0, "rng_stream": 11,
        "target": "UR tracking-frame target fed to UrLineCommander.tick (regenerates the line "
                  "every sample)",
        "guard": "per sample, the offset is scaled so the excited target keeps >= "
                 "min_clearance (clearance.required_lift on the offset pose); the scale falls "
                 "at once and recovers at 1/ramp_s per second",
        "recover_s": EXCITE_RAMP_S,
    }
    return cfg


def collect(args: argparse.Namespace, ranges: dict | None = None,
            scenario: str | None = None) -> dict:
    """Run the collection; returns the ``index.json`` dict."""
    backend = getattr(args, "backend", "position")
    scenario = scenario or getattr(args, "scenario", None)
    if backend not in BACKENDS:
        raise ValueError(f"--backend {backend!r} not in {BACKENDS}")
    if backend == "osc" and args.fresh_pick:
        raise ValueError("--fresh-pick is position-backend only")
    if scenario is not None and (backend != "osc" or scenario not in SCENARIOS):
        raise ValueError(f"--scenario {scenario!r} needs --backend osc and one of {SCENARIOS}")
    if backend == "osc":
        check_private_url(args.lcm_url)
    ranges = pert.ranges_for_backend(backend) if ranges is None else ranges
    n = sample_steps(args.sample_period)
    thresholds = parse_thresholds(args.thresholds)
    intents = ["nominal"] if scenario == "nominal" else _csv(args.intents)
    new_opts = new_options(args)
    if new_opts and backend != "osc":
        raise ValueError("--approach* / --place3-* / --tail / --row-tag need --backend osc")
    p3_cfg = new_opts.get("place3_cfg")
    if p3_cfg is not None:
        if scenario is not None:
            raise ValueError("--place3-mode needs no --scenario")
        logger.info(f"[LCS] --place3-mode {p3_cfg['mode']}: --intents ignored")
        intents = ["engaged_band", "wide"] if p3_cfg["mode"] == "continuous" else (
            ["engaged_band"] if p3_cfg["mode"] == "engaged_band" else ["fixed"])
        perts = [pert.Perturbation(intent="place3") for _ in range(args.episodes)]
    else:
        perts = plan_episodes(args, ranges, scenario)
    nominal = load_pre_mpc_segment(args.params, first=FIRST, last=LAST)
    tangent = pert.belt_tangent(nominal)
    if args.dry_run:
        if p3_cfg is not None:
            for i in range(args.episodes):
                print(i, json.dumps(draw_place3(p3_cfg, args.seed, i, ranges)))
            return {"episodes": []}
        print(perturbation_table(perts))
        return {"episodes": [p.to_dict() for p in perts]}

    out = args.out or DEFAULT_OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.label}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start_states = getattr(args, "start_states", None)
    set_states: list[tuple[str, Path, object]] = []
    if start_states is not None:
        if backend != "osc" or args.start_state is not None:
            raise ValueError("--start-states needs --backend osc and no --start-state")
        states, set_meta = load_start_set(start_states, getattr(args, "variants", None))
        set_states = [(vid, f, sim_snapshot.load(f)) for vid, f in states]
    elif getattr(args, "variants", None):
        raise ValueError("--variants needs --start-states")
    start_state = set_states[0][1] if set_states else resolve_start_state(args)
    start_meta: dict = {"path": None if args.fresh_pick else str(start_state)}
    snap = None
    if not args.fresh_pick:
        snap = set_states[0][2] if set_states else sim_snapshot.load(start_state)
        start_meta.update(sha256=_sha256(start_state), label=snap.meta.get("label"),
                          notes=snap.meta.get("notes"), created=snap.meta.get("created"),
                          git_commit=snap.meta.get("git_commit"),
                          step_index=snap.step_index, sim_time=snap.sim_time)
    if set_states:
        start_meta = {**set_meta, "states": [{"id": v, "file": str(f), "sha256": _sha256(f)}
                                             for v, f, _ in set_states]}
    excite = excitation_config(args, scenario) if backend == "osc" else None
    opts = hold_options(args, scenario)
    if opts and backend != "osc":
        raise ValueError("--hold-ur-gripper / --nominal-ur-dz-mm need --backend osc")
    opts.update(new_opts)
    # New flags at their defaults stay out of index.args (flags-off index unchanged).
    arg_items = {k: v for k, v in vars(args).items()
                 if not (k in NEW_ARG_DEFAULTS and v == NEW_ARG_DEFAULTS[k])}
    index = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in arg_items.items()},
        "git": _git(), "backend": backend, "scenario": scenario, "start_state": start_meta,
        "sample_period_s": n * lcs.SIM_DT_S,
        "sample_steps": n, "thresholds": dataclasses.asdict(thresholds),
        "action_definition": (getattr(args, "action_definition", OSC_ACTION_DEFINITION)
                              if backend == "osc" else None),
        "pre_hold_s": getattr(args, "pre_hold_s", 0.0) if backend == "osc" else None,
        "belt_sampling": lcs.BELT_SAMPLING,
        "ranges": pert.ranges_to_dict(ranges), "labels": list(LABELS),
        "belt_tangent": tangent, "min_clearance_mm": args.min_clearance,
        "clearance_every": CLEARANCE_EVERY, "excitation": excite, **opts, "episodes": [],
    }
    index_path = out / "index.json"
    _write_json(index_path, index)

    osc_info = None
    if backend == "osc":
        from round_belt_task.osc_simulation import RoundBeltOscSimulation

        sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=args.osc_timeout_s,
                                           arm_ke=args.arm_ke, arm_kd=args.arm_kd)
    else:
        from round_belt_task.offline_simulation import RoundBeltOfflineSimulation

        sim = RoundBeltOfflineSimulation.build(arm_ke=args.arm_ke, arm_kd=args.arm_kd)
    sim.args.record_state_every = RECORD_STATE_EVERY
    t_run = time.perf_counter()
    timings = []
    try:
        if backend == "osc":
            warm = sim.start_osc(args.osc_log or out / "osc.log")
            osc_info = {**sim.osc.describe(), "warm_up_s": warm, "settle_s": args.osc_settle_s,
                        "timeout_s": args.osc_timeout_s}
            index["osc"] = osc_info
            _write_json(index_path, index)
        if snap is not None:
            sim.restore(snap)  # the guard's geometry is read at the start pose, gripper closed
        jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
        board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
        logger.info(f"[LCS] clearance guard: min {args.min_clearance:g} mm, {len(gripper.local)} "
                    f"2F-85 points on {len(gripper.shape_labels)} colliders (jaw bytes {jaws}) vs "
                    f"plate + {len(board.pulleys)} pulleys")
        initial = sim_snapshot.capture(sim, "initial") if args.fresh_pick else None
        ctx = None
        if backend == "osc":
            ctx = OscContext(args=args, out=out, n=n, thresholds=thresholds, tangent=tangent,
                             snap=snap, nominal=nominal, board=board, gripper=gripper,
                             excite=excite, scenario=scenario, osc=osc_info,
                             start_state=start_state, opts=opts)
            if new_opts:
                ctx.pulley = pulley_frame(sim)
                index["pulley_large"] = {k: v.tolist() for k, v in ctx.pulley.items()}
                index["board_colliders"] = {"plate": clr.BOARD_PLATE_SHAPE,
                                            "pulleys": len(board.pulleys)}
                _write_json(index_path, index)
            if "tail_cfg" in new_opts:
                ctx.gauge = _rod_gauge(sim)
        approaches = start_approaches(start_states) if new_opts.get("approach_cfg") else {}
        for i, p in enumerate(perts):
            if ctx is None:
                row = run_episode(sim, i, p, args, out, n, thresholds, tangent, snap, initial,
                                  nominal, board, gripper)
            else:
                variant = None
                if set_states:
                    variant, ctx.start_state, ctx.snap = set_states[i % len(set_states)]
                if new_opts:
                    ctx.ep = plan_new_episode(ctx, new_opts, i, variant, approaches, ranges)
                try:
                    row = run_osc_episode(sim, ctx, i, p)
                    if row.get("reason_key") == "tail grasp lost":
                        logger.warning(f"[LCS] {i:4d} tail grasp lost: retry at amplitude x0.5")
                        ctx.ep["tail"] = {**ctx.ep["tail"], "amp_mul": 0.5, "retry": 1}
                        first = row
                        row = run_osc_episode(sim, ctx, i, p)
                        row["tail_retry"] = {"first_reason": first.get("reason")}
                except OscTimeout as exc:
                    _finish_recording(sim, out, f"episode_{i:04d}-{p.intent}", "osc timeout")
                    if not exc.process_alive:
                        logger.error(f"[LCS] {i:4d} OSC died: {exc}")
                        index["summary"] = {"aborted": "osc died", "error": str(exc),
                                            "episodes_done": len(index["episodes"])}
                        _write_json(index_path, index)
                        raise
                    logger.warning(f"[LCS] {i:4d} {p.intent:<8} skipped: osc timeout ({exc})")
                    sim.bridge.forget_pending()
                    row = {"file": None, "intent": p.intent, "perturbation": p.to_dict(),
                           "backend": backend, "status": "skipped", "reason": "osc timeout",
                           "error": str(exc)}
                if variant is not None:
                    row["start_variant"] = variant
                if new_opts.get("row_tag"):
                    row["tag"] = new_opts["row_tag"]
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
    if ok:
        print(slant_table(ok, intents))
        print(clearance_table(ok, intents))
    per_min = 60.0 * len(ok) / wall if wall > 0 else 0.0
    contacts = [r["file"] for r in ok if r["board_contact"]]
    summary = {"episodes_ok": len(ok), "episodes_skipped": len(rows) - len(ok),
               "wall_s": wall, "episodes_per_min": per_min,
               "board_contact_files": contacts,
               "min_board_clearance_mm": min((r["min_board_clearance_mm"] for r in ok),
                                             default=None),
               "skipped_by_reason": dict(Counter(r.get("reason_key", r.get("reason"))
                                                 for r in rows if r["status"] != "ok"))}
    if backend == "osc":
        summary["osc"] = osc_info
        summary["osc_log_errors"] = [e for e in OSC_LOG_ERRORS if e in sim.osc.log_text()]
        summary["min_franka_tip_clearance_mm"] = min(
            (r["min_franka_tip_clearance_mm"] for r in ok), default=None)
        summary["excite_clip"] = {k: sum(r["excite_clip"][k] for r in ok)
                                  for k in ("active_steps", "floor_steps", "cap_steps")}
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


def _classify(sampler, thresholds, tangent) -> tuple[str, dict, float, bool]:
    """``(outcome, metrics + slant, min board clearance [mm], board contact)`` of the frames."""
    belt_T = np.stack([fr["belt"] for fr in sampler.frames])
    pulley_T = np.stack([fr["pulley_raw"] for fr in sampler.frames])
    label, metrics = classify_episode(belt_T, pulley_T, thresholds)
    metrics.update(slant_episode(belt_T, pulley_T, tangent, thresholds))
    min_clear_mm = sampler.min_clearance_m * 1e3
    return label, metrics, min_clear_mm, min_clear_mm < 0.0


SLANT_KEYS = ("slant_deg", "slant_axis_deg", "slant_dir", "slant_deg_t", "slant_axis_deg_t")


def _write_episode(i: int, p: pert.Perturbation, path: Path, writer, label: str,
                   extras_meta: dict, metrics: dict, min_clear_mm: float, contact: bool,
                   clamp_dict: dict | None, extra_post: dict, no_pcd: bool, n: int,
                   timing: dict, t0: float, t3: float, log_detail: str) -> tuple[float, float]:
    """Write + validate one episode, fill ``timing`` write/total, log it; ``(wrap, h median)``."""
    post = {
        "intent": np.array(p.intent), "outcome": np.array(label),
        "perturbation": np.array(json.dumps(p.to_dict())),
        "wrap_deg": metrics["wrap_deg"], "h_median_mm": metrics["h_median_mm"],
        **{k: np.array(metrics[k]) for k in SLANT_KEYS},
        "min_board_clearance_mm": np.float32(min_clear_mm),
        "board_contact": np.array(contact),
        "clamp": np.array(json.dumps(clamp_dict)),
        **extra_post,
    }
    post = {lcs.EXTRA_PREFIX + k: v for k, v in post.items()}
    omit: tuple[str, ...] = ()
    if no_pcd:
        post["sim_no_pcd"] = np.array(True)
        omit = ("pcd", "pcd_rgb")
    writer.write(path, label, extras_meta, arrays=post, omit=omit)
    if not no_pcd:
        lcs.validate_episode(path, period_us=round(n * lcs.SIM_DT_S * 1e6))
    timing["write"] = time.perf_counter() - t3
    timing["total"] = time.perf_counter() - t0

    wrap, h_med = float(metrics["wrap_deg"][-1]), float(metrics["h_median_mm"][-1])
    log = logger.warning if contact else logger.info
    log(f"[LCS] {i:4d} {p.intent:<8} -> {label:<8} wrap {wrap:6.1f} deg, h median "
        f"{h_med:6.1f} mm, slant {float(metrics['slant_deg']):5.1f} deg {metrics['slant_dir']}, "
        f"clearance {min_clear_mm:5.2f} mm{' CONTACT' if contact else ''}"
        f"{log_detail}, {timing['total']:.1f} s")
    return wrap, h_med


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
    label, metrics, min_clear_mm, contact = _classify(sampler, thresholds, tangent)
    track = result.summary()
    phase_labels = [START_PHASE, *traj.labels]
    extras_meta = {
        "phase_labels": phase_labels, "intent": p.intent, "outcome": label,
        "perturbation": p.to_dict(), "ee_pose_source": "commanded Cartesian target",
        "action_source": "commanded Cartesian target at samples t and t+1",
        "pulley_pose_layout": lcs.POSE_LAYOUT, "start_step": start_step,
        "start_state": None if args.fresh_pick else str(resolve_start_state(args)),
        "backend": "position", "sample_period_s": n * lcs.SIM_DT_S,
        "thresholds": dataclasses.asdict(thresholds), "clamp": clamp.to_dict(),
        "clearance_source": "2F-85 colliders vs the board plate box and the two pulleys, "
                            f"sampled every {CLEARANCE_EVERY} control steps",
    }
    path = out / row["file"]
    wrap, h_med = _write_episode(
        i, p, path, build_writer(frames, n, pcd=not args.no_pcd), label, extras_meta, metrics,
        min_clear_mm, contact, clamp.to_dict(), {}, args.no_pcd, n, timing, t0, t3,
        f" (lift {clamp.max_lift_mm:.2f} mm, of which path {clamp.path_lift_mm:.2f} mm)")
    return {**row, "backend": "position", "outcome": label, "steps": len(traj),
            "frames": len(frames),
            "duration_s": len(traj) * lcs.SIM_DT_S,
            "tracking_rms_mm": [track["franka_mm_rms"], track["ur_mm_rms"]],
            "final_wrap_deg": wrap, "final_h_median_mm": h_med, **_slant_row(metrics),
            "min_board_clearance_mm": min_clear_mm, "board_contact": contact,
            "clamp_lift_mm": clamp.max_lift_mm, "clamp_tilt_scale": clamp.tilt_scale,
            "grasp_ok_final": [bool(v) for v in frames[-1]["grasp_ok"]],
            "size_bytes": path.stat().st_size, "recording": recording,
            "timing_s": timing, "status": "ok"}


@dataclasses.dataclass
class OscContext:
    """Per-run inputs of :func:`run_osc_episode`."""

    args: argparse.Namespace
    out: Path
    n: int
    thresholds: OutcomeThresholds
    tangent: np.ndarray
    snap: object
    nominal: list
    board: clr.Board
    gripper: clr.Gripper
    excite: dict
    scenario: str | None
    osc: dict
    start_state: Path
    opts: dict = dataclasses.field(default_factory=dict)
    params: CommanderParams = dataclasses.field(default_factory=CommanderParams)
    x_tool0: np.ndarray = dataclasses.field(default_factory=x_tool0_tracking)
    ep: dict = dataclasses.field(default_factory=dict)  # opt-in per-episode plan
    pulley: dict | None = None
    gauge: object = None

    @property
    def plate_top_z(self) -> float:
        plate = self.board.plate
        return float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half)


@dataclasses.dataclass(slots=True)
class OscTick:
    """The hook's view of one control step: measured poses + the command published."""

    k: int
    t: float
    ee_f: np.ndarray
    X_u: np.ndarray
    cmd: object
    ur: object
    ur_raw: np.ndarray | None = None
    ur_applied: np.ndarray | None = None
    ur_scale: float = 1.0


class OscEpisodeSampler(EpisodeSampler):
    """Frames for the osc backend: measured state + the command published at the sample tick."""

    def __init__(self, sim, ctx: OscContext, phase_labels: list[str]) -> None:
        super().__init__(sim, ctx.n, pcd=not ctx.args.no_pcd, traj_phase=np.zeros(0, np.int64),
                         phase_offset=0, board=ctx.board, gripper=ctx.gripper)
        self.ctx = ctx
        self.phase_index = {label: i for i, label in enumerate(phase_labels)}
        self.on_frame = None  # opt-in: (k, frame, FrameMetrics, sim) per sampled frame

    def sample_tick(self, tick: OscTick) -> None:
        t0 = time.perf_counter()
        sim, ctx, cmd = self.sim, self.ctx, tick.cmd
        body_q = sim.state_0.body_q.numpy()
        q_f, q_u = sim.arm_positions()
        v_f, v_u = sim.arm_velocities()
        grasp = sim.grasp_state(body_q)
        knots = np.hstack([cmd.knots_pos, cmd.knots_quat])
        if len(knots) < ctx.params.n_knots:
            knots = np.vstack([knots, np.repeat(knots[-1:], ctx.params.n_knots - len(knots), 0)])
        ex = cmd.excite_applied
        frame = {
            "step": tick.k, "sim_step": sim.step_index, "time": tick.t, "q_f": q_f, "q_u": q_u,
            "v_f": v_f, "v_u": v_u, "ee_f": tick.ee_f, "ee_u": _pose7_mat(tick.X_u),
            "knot0": knots[0].copy(), "knot1": knots[1].copy(), "knots": knots,
            "ur_t": _pose7_mat(tick.ur.pose_at(tick.t)),
            "ur_t1": _pose7_mat(tick.ur.pose_at(tick.t + ctx.params.dt)),
            "hold": bool(cmd.hold), "phase": self.phase_index[cmd.phase],
            "excite_dpos": np.zeros(3) if ex is None else np.asarray(ex.dpos, np.float64),
            "excite_rotvec": np.zeros(3) if ex is None else np.asarray(ex.axis) * ex.angle,
            "excite_ur_raw": np.zeros(6) if tick.ur_raw is None else tick.ur_raw.copy(),
            "excite_ur": np.zeros(6) if tick.ur_applied is None else tick.ur_applied.copy(),
            "ur_excite_scale": float(tick.ur_scale),
            "osc_utime": round(tick.t * 1e6),
            "belt": body_q[sim.info.belt_bodies, :3].astype(np.float64),
            "pulley_raw": body_q[self._large].astype(np.float64),
            "hand_mm": grasp.franka_width_mm, "robotiq": grasp.ur_status,
            "grasp_ok": np.array(grasp.held()),
            "clearance_mm": (self.last_clearance_m if self.last_clearance_m is not None
                             else self.clearance(body_q)) * 1e3,
            "tip_clearance_mm": (tick.ee_f[2] - ctx.plate_top_z) * 1e3,
        }
        t1 = time.perf_counter()
        if self.pcd:
            frame["pcd"], frame["pcd_rgb"] = sim.point_cloud()
        frame["render_step"] = sim.step_index
        self.render_s += time.perf_counter() - t1
        self.sample_s += time.perf_counter() - t0
        self.frames.append(frame)
        if self.on_frame is not None:
            fm = frame_metrics(frame["belt"], frame["pulley_raw"], th=ctx.thresholds)
            self.on_frame(tick.k, frame, fm, sim)


def tracking_errors(frames: list[dict]) -> np.ndarray:
    """``(T, 2)`` mm, 0 at t=0: Franka ``|ee_t - knot1_{t-1}|``, UR ``|ee_u_t - line_{t-1}(t)|``.

    Equals ``|ee_t - (ee_{t-1} + u_{t-1})|`` (position part) under ``knot1_minus_measured``
    (``sim_action_knot1_minus_measured``).
    """
    out = np.zeros((len(frames), 2))
    for t in range(1, len(frames)):
        prev, fr = frames[t - 1], frames[t]
        out[t, 0] = np.linalg.norm(fr["ee_f"][:3] - prev["knot1"][:3]) * 1e3
        out[t, 1] = np.linalg.norm(fr["ee_u"][:3] - prev["ur_t1"][:3]) * 1e3
    return out


def frame_action(fr: dict, definition: str) -> np.ndarray:
    return lcs.command_action(definition, fr["ee_f"], fr["ee_u"], fr["knot0"], fr["knot1"],
                              fr["ur_t"], fr["ur_t1"])


def build_osc_writer(frames: list[dict], n: int, pcd: bool,
                     definition: str = OSC_ACTION_DEFINITION
                     ) -> tuple[lcs.EpisodeWriter, np.ndarray]:
    writer = lcs.EpisodeWriter(sample_steps=n, action_definition=definition)
    track = tracking_errors(frames)
    states = [lcs.state_vector(fr["q_f"], fr["q_u"], fr["v_f"], fr["v_u"], fr["ee_f"], fr["ee_u"])
              for fr in frames]
    realised = lcs.realised_delta(np.stack(states))
    for t, fr in enumerate(frames):
        state = states[t]
        # Every row, the last included: the command published at this tick (no zero padding).
        action = frame_action(fr, definition)
        cloud = lcs.camera_points(fr["pcd"]) if pcd else np.zeros((1, 3), np.float32)
        writer.add_frame(fr["sim_step"], fr["time"], state, action, cloud,
                         lcs.belt_points_ordered(fr["belt"]), lcs.kinematic_points(),
                         pcd_rgb=fr.get("pcd_rgb") if pcd else None, extras={
                             "ee_franka": fr["ee_f"], "ee_ur": fr["ee_u"],
                             "cmd_knot0_franka": fr["knot0"], "cmd_knot1_franka": fr["knot1"],
                             "cmd_knots_franka": fr["knots"], "cmd_ur_t": fr["ur_t"],
                             "cmd_ur_t1": fr["ur_t1"], "cmd_hold": np.array(fr["hold"]),
                             "phase": np.int16(fr["phase"]),
                             "excite_dpos_m": fr["excite_dpos"],
                             "excite_rotvec": fr["excite_rotvec"],
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
                             "excite_ur_dpos_m": fr["excite_ur"][:3],
                             "excite_ur_rotvec": fr["excite_ur"][3:],
                             "excite_ur_raw": fr["excite_ur_raw"],
                             "ur_excite_scale": np.float64(fr["ur_excite_scale"]),
                         })
    return writer, track


class UrExciteGuard:
    """Scales the UR target offset so the excited target keeps ``min_clearance`` (smoothly).

    The required scale is the largest ``s`` in [0, 1] whose offset pose needs no
    :func:`clearance.required_lift`; the applied scale falls to it at once and recovers at
    ``1 / recover_s`` per second, so a binding guard never snaps the offset back.
    """

    def __init__(self, board, gripper, min_clearance_m: float, recover_s: float) -> None:
        self.board, self.gripper, self.min_c = board, gripper, float(min_clearance_m)
        self.rate = 1.0 / recover_s if recover_s > 0.0 else math.inf
        self.scale, self.t = 1.0, None
        self._raw: np.ndarray | None = None

    @staticmethod
    def target(base: UrTarget, offset: np.ndarray) -> UrTarget:
        """``base`` moved by ``offset = (dpos, rotvec)``, rotation applied on the left (world)."""
        angle = float(np.linalg.norm(offset[3:]))
        R = quat_to_mat3(base.quat_wxyz)
        if angle > 0.0:
            R = quat_to_mat3(quat_axis_angle(offset[3:], angle)) @ R
        return UrTarget(pos=base.pos + offset[:3], quat_wxyz=mat3_to_quat(R), byte=base.byte)

    def _clear(self, base: UrTarget, offset: np.ndarray, jaws) -> bool:
        tgt = self.target(base, offset)
        X = pose_mat(tgt.pos, tgt.quat_wxyz)
        return all(clr.required_lift(self.board, self.gripper, X, self.min_c, jaw)
                   <= clr.CLAMP_TOL_M for jaw in jaws)

    def required(self, base: UrTarget, raw: np.ndarray, jaws) -> float:
        if self.min_c <= 0.0 or not np.any(raw) or self._clear(base, raw, jaws):
            return 1.0
        if not self._clear(base, 0.0 * raw, jaws):
            return 0.0
        lo, hi = 0.0, 1.0
        for _ in range(UR_SCALE_BISECT):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self._clear(base, mid * raw, jaws) else (lo, mid)
        return lo

    def apply(self, t: float, base: UrTarget, raw: np.ndarray, jaws) -> np.ndarray:
        """The applied offset for this tick (re-evaluated only when ``raw`` changes)."""
        if self._raw is None or not np.array_equal(raw, self._raw):
            rise = math.inf if self.t is None else self.rate * (t - self.t)
            self.scale = min(self.required(base, raw, jaws), self.scale + rise)
            self.t, self._raw = t, raw.copy()
        return self.scale * raw


def _scenario_targets(sim) -> tuple[list[FrankaTarget], list[UrTarget | None]]:
    q_f, q_u = sim.arm_positions()
    X_f, X_u = FrankaTip.fk(q_f), UrTracking.fk(q_u)
    franka = FrankaTarget(label="pure_translation", pos=X_f[:3, 3] + PURE_TRANSLATION_M,
                          quat_wxyz=mat3_to_quat(X_f[:3, :3]), hand_mm=None,
                          dwell_s=PURE_TRANSLATION_DWELL_S)
    return [franka], [UrTarget(pos=X_u[:3, 3].copy(), quat_wxyz=mat3_to_quat(X_u[:3, :3]),
                               byte=None)]


def _rod_gauge(sim):
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "lcs"))
    from make_flat_engaged_state import RodGauge

    return RodGauge(sim)


def plan_new_episode(ctx: OscContext, opts: dict, i: int, variant: str | None,
                     approaches: dict, ranges: dict) -> dict:
    """Episode ``i``'s approach / place_3 sample / tail request (the opt-in modes)."""
    seed = ctx.args.seed
    ep = {"approach": draw_approach(opts.get("approach_cfg"), seed, i, approaches.get(variant))}
    if opts.get("place3_cfg"):
        ep["place3"] = draw_place3(opts["place3_cfg"], seed, i, ranges)
    if opts.get("tail_cfg"):
        ep["tail"] = {"amp_mul": 1.0, "retry": 0}
    return ep


def franka_floor(waypoints: list, plate_top_z: float) -> tuple[list, float]:
    """``place_3`` Franka tip raised to plate + FRANKA_FLOOR_MM; ``(waypoints, lift mm)``."""
    z_min = plate_top_z + FRANKA_FLOOR_MM * 1e-3
    out, lift = [], 0.0
    for w in waypoints:
        if w.label == LAST and w.franka_pos[2] < z_min:
            lift = z_min - float(w.franka_pos[2])
            w = dataclasses.replace(w, franka_pos=np.asarray(w.franka_pos, float)
                                    + np.array([0.0, 0.0, lift]))
        out.append(w)
    return out, lift * 1e3


def _guard(sim, ctx: OscContext, p: pert.Perturbation, plan: dict | None = None):
    """Perturbed, clamped waypoints + the path check on straight lines from the measured poses.

    ``plan`` (opt-in modes): ``{nominal, tangent, depth}`` -- the approach-transformed waypoints,
    their tangent and the place_3-only depths; adds the Franka floor (``clamp.franka_lift_mm``).
    """
    args = ctx.args
    base = ctx.nominal if plan is None else plan["nominal"]
    nominal, _ = override_waypoints(base, ctx.opts) if ctx.opts else (base, None)
    tangent = ctx.tangent if plan is None else plan["tangent"]
    if plan is not None:
        nominal = depth_waypoints(nominal, plan["depth"])
    waypoints, clamp = clr.clamp_waypoints(nominal, p, tangent, ctx.board, ctx.gripper,
                                           min_clearance=args.min_clearance * 1e-3)
    if plan is not None:
        waypoints, clamp.franka_lift_mm = franka_floor(waypoints, ctx.plate_top_z)
    q_f, q_u = sim.arm_positions()
    meas_f, meas_u = FrankaTip.fk(q_f), UrTracking.fk(q_u)
    franka_mm, ur_byte = sim.gripper_commands()
    want = args.min_clearance * 1e-3

    def lines(wps):
        return build_cartesian_trajectory(wps, meas_f, meas_u, dt=sim.frame_dt, settle_s=0.0,
                                          start_franka_mm=franka_mm, start_ur_byte=ur_byte)

    traj = lines(waypoints)
    for _ in range(PATH_CLAMP_ITERS):
        low = clr.path_clearance(ctx.board, ctx.gripper, traj.ur_4x4, traj.ur_gripper_byte)
        if clamp.path_before_mm is None:
            clamp.path_before_mm = low * 1e3
        if low >= want - clr.CLAMP_TOL_M:
            break
        clamp.path_lift_mm += (want - low) * 1e3
        waypoints = clr.lift_waypoints(waypoints, pert.PERTURBED_LABELS, want - low)
        traj = lines(waypoints)
    clamp.path_after_mm = clr.path_clearance(ctx.board, ctx.gripper, traj.ur_4x4,
                                             traj.ur_gripper_byte) * 1e3
    return waypoints, clamp


def _skip(row: dict, i: int, reason: str, key: str) -> dict:
    logger.warning(f"[LCS] {i:4d} {row['intent']:<8} skipped: {reason}")
    return {**row, "file": None, "status": "skipped", "reason": reason, "reason_key": key}


def run_osc_episode(sim, ctx: OscContext, i: int, p: pert.Perturbation,
                    on_frame=None) -> dict:
    """``on_frame(k, frame, FrameMetrics, sim)``: opt-in, called at every sampled frame
    (between control steps)."""
    args, n, params = ctx.args, ctx.n, ctx.params
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    name = f"episode_{i:04d}"
    ep, plan, tangent = ctx.ep, None, ctx.tangent
    if ep:
        nominal_ep = approach_waypoints(ctx.nominal, ep.get("approach"), ctx.pulley)
        if ep.get("approach") is not None:
            tangent = pert.belt_tangent(nominal_ep)
        depth = {}
        if ep.get("place3"):
            p, depth = place3_perturbation(ep["place3"], tangent)
        plan = {"nominal": nominal_ep, "tangent": tangent, "depth": depth}
    row = {"file": f"{name}.npz", "intent": p.intent, "perturbation": p.to_dict(),
           "backend": "osc", "scenario": ctx.scenario, "excite": bool(ctx.excite["on"])}
    if plan is not None:
        row.update(approach=ep.get("approach"), tangent=tangent.tolist(),
                   place3_sample=None if not ep.get("place3")
                   else {**ep["place3"], "depth_mm": depth})
    opts_meta = dict(ctx.opts)
    if ctx.opts.get("hold_ur_gripper"):
        opts_meta["ur_gripper_bytes_not_applied"] = override_waypoints(ctx.nominal, ctx.opts)[1]
    row.update(opts_meta)
    sim.restore(ctx.snap, settle_steps=0)
    timing["restore"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    grasp = sim.settle(round(args.osc_settle_s / sim.frame_dt))
    timing["settle"] = time.perf_counter() - t1
    if grasp.held() != (True, True):
        return _skip(row, i, f"grasp lost after settle ({grasp.describe()})", "grasp lost")

    t2 = time.perf_counter()
    if ctx.scenario == "pure_translation":
        f_targets, u_targets = _scenario_targets(sim)
        clamp = None
        row["clamp"] = None
    else:
        waypoints, clamp = _guard(sim, ctx, p, plan)
        row["clamp"] = _clamp_dict(clamp)
        f_targets, u_targets = targets_from_waypoints([w for w in waypoints if w.label != FIRST])
    timing["guard"] = time.perf_counter() - t2
    last = len(f_targets) - 1
    phase_labels = [f"{kind}:{t.label}" for t in f_targets for kind in ("move", "hold")]
    phase_labels += ["done", PREHOLD_PHASE]
    tail_spec = None
    if plan is not None and ep.get("tail") is not None:
        place_tail = _place_tail()
        band = (ep.get("place3") or {}).get("mode") == "engaged_band"
        tail_spec = place_tail.draw_spec(ctx.opts["tail_cfg"], args.seed, i, band,
                                         n * lcs.SIM_DT_S, ep["tail"]["amp_mul"])
        phase_labels += [place_tail.phase_label(tail_spec["family"]), place_tail.PHASE_HOLD]

    hand0, byte0 = sim.gripper_commands()
    franka = FrankaWaypointCommander(f_targets, params, hand_mm=hand0)
    ur = UrLineCommander(u_targets, params, X_tool0_tracking=ctx.x_tool0, byte=byte0)
    ex_rng = np.random.default_rng([args.seed, i, 7])
    ex_cfg = ctx.excite
    ex_on = ex_cfg["on"]
    ex_args = (ex_cfg["pos_mm"] * 1e-3, np.radians(ex_cfg["rot_deg"]), ex_cfg["cap_factor"],
               ex_cfg["down_mm"] * 1e-3)
    ou = None
    if ex_on and ex_cfg["mode"] == "ou":
        ou = OuExcitation(ex_rng, ex_args[0], ex_args[1], ex_cfg["tau_s"], n * lcs.SIM_DT_S,
                          ex_args[2], ex_args[3], ramp_s=ex_cfg["ramp_s"])
    fade_m = ex_cfg.get("fade_dist_mm", 0.0) * 1e-3
    ur_cfg = ex_cfg.get("ur") or {"on": False}
    ou_ur = guard = None
    if ur_cfg["on"]:
        ou_ur = OuExcitation(np.random.default_rng([args.seed, i, ur_cfg["rng_stream"]]),
                             ur_cfg["pos_mm"] * 1e-3, np.radians(ur_cfg["rot_deg"]),
                             ex_cfg["tau_s"], n * lcs.SIM_DT_S, 1.0, 0.0,
                             ramp_s=ex_cfg["ramp_s"])
        guard = UrExciteGuard(ctx.board, ctx.gripper, args.min_clearance * 1e-3,
                              ur_cfg["recover_s"])
    clip = {"active_steps": 0, "floor_steps": 0, "cap_steps": 0}
    u_coords = sim.arm_coords()[1]
    plate_top = ctx.plate_top_z
    pre_n = max(0, round(getattr(args, "pre_hold_s", 0.0) / (n * lcs.SIM_DT_S)))
    # The hook runs after control_step advanced the counter; step 0 = first commander tick.
    s0 = sim.step_index + 1 + pre_n * n
    t_start = sim.osc_time_s(s0)
    state = {"tick": None, "excite": None, "hook_s": 0.0, "tip_min": float("inf"),
             "ur_raw": np.zeros(6), "ur_base": None}
    latched_at: dict[str, float] = {}
    latch_err: dict[str, float] = {}

    def hook(step, t, joint_q, body_q):
        h0 = time.perf_counter()
        k = step - s0
        ee_f = sim.franka_measured_pose7(joint_q)
        pos_f, quat_f = ee_f[:3], ee_f[3:]
        X_u = UrTracking.fk(joint_q[u_coords])
        state["tip_min"] = min(state["tip_min"], pos_f[2] - plate_top)
        u_target = u_targets[franka.index]
        if u_target is None:
            e_pos = e_ori = 0.0
        else:
            e_pos = float(np.linalg.norm(X_u[:3, 3] - u_target.pos))
            e_ori = angular_distance(mat3_to_quat(X_u[:3, :3]), u_target.quat_wxyz)
        # Excite knots 1..6 except in the last target's hold/dwell/settle (outcome window).
        quiet = franka.index == last and (
            franka.latched or franka.is_reached(pos_f, quat_f, e_pos, e_ori))
        # Fade so the offset is 0 by the place_3 reach: a held offset would shift the latch.
        fade = quiet or (franka.index == last and float(
            np.linalg.norm(pos_f - f_targets[last].pos)) <= fade_m)
        if ou is not None:
            if fade:
                state["excite"] = ou.fade(t)
            elif k % n == 0:
                state["excite"] = ou.step() if k > 0 else ou.excite()
        elif ex_on and k % n == 0:
            state["excite"] = draw_excite(ex_rng, *ex_args)
        excite = state["excite"] if ou is not None or not quiet else None
        f_target = f_targets[franka.index]
        state["reach"] = (float(np.linalg.norm(pos_f - f_target.pos)) * 1e3,
                          np.degrees(angular_distance(quat_f, f_target.quat_wxyz)),
                          e_pos * 1e3, np.degrees(e_ori))
        cmd = franka.tick(t, pos_f, quat_f, e_pos, e_ori, excite)
        if excite is not None:
            got = cmd.excite_applied
            want = excite.dpos.copy()
            want[2] = max(want[2], -excite.down_m)
            clip["active_steps"] += 1
            clip["floor_steps"] += int(want[2] != excite.dpos[2])
            clip["cap_steps"] += int(not np.array_equal(got.dpos, want)
                                     or got.angle != excite.angle)
        if cmd.latched_index is not None:
            label = f_targets[cmd.latched_index].label
            latched_at[label] = t - t_start
            latch_err[label] = float(np.linalg.norm(pos_f - f_targets[cmd.latched_index].pos)
                                     ) * 1e3
        ur.on_latch(cmd.latched_index)
        ur_target = u_targets[cmd.target_index]
        state["ur_base"] = ur_target if ur_target is not None else state["ur_base"]
        applied, scale = None, 1.0
        if ou_ur is not None and state["ur_base"] is not None:
            ex_ur = None
            if fade:
                ex_ur = ou_ur.fade(t)
            elif k % n == 0:
                ex_ur = ou_ur.step() if k > 0 else ou_ur.excite()
            if fade or k % n == 0:
                raw = np.zeros(6) if ex_ur is None else np.concatenate(
                    [ex_ur.dpos, np.asarray(ex_ur.axis) * ex_ur.angle])
                raw[2] = max(raw[2], 0.0)  # never below the nominal target
                state["ur_raw"] = raw
            base = state["ur_base"]
            jaws = tuple(dict.fromkeys(b for b in (ur.byte, base.byte) if b is not None)) \
                or (None,)
            applied = guard.apply(t, base, state["ur_raw"], jaws)
            scale = guard.scale
            ur_target = UrExciteGuard.target(base, applied)
        ur_cmd = ur.tick(t, X_u, ur_target)
        state["tick"] = OscTick(k, t, ee_f, X_u, cmd, ur_cmd, state["ur_raw"].copy(),
                                applied, scale)
        msg = saved_traj_message(round(t * 1e6), cmd.knots_pos, cmd.knots_quat, cmd.times)
        state["hook_s"] += time.perf_counter() - h0
        return msg

    if args.record:
        sim.start_recording(ctx.out / "recordings", f"{name}-{p.intent}")
    sampler = OscEpisodeSampler(sim, ctx, phase_labels)
    sampler.on_frame = on_frame
    q_ur = sim.arm_targets()[1]
    if pre_n > 0:
        _pre_hold(sim, ctx, sampler, state, pre_n * n, s0, q_ur, ur, hand0, byte0)
    lead_gain = sim.arm_kd / sim.arm_ke / sim.frame_dt
    lead_line = None
    settle_steps = round(args.settle_s / sim.frame_dt)
    max_steps = round(args.max_episode_s / sim.frame_dt)
    ik_s = step_s = 0.0
    wait0 = sim.bridge.wait_s
    done_at = None
    rods = [] if ctx.gauge is not None else None
    tail = None
    sim.commander_hook = hook
    try:
        while True:
            tick = state["tick"]
            if tick is not None:
                i0 = time.perf_counter()
                # The UR target this physics step drives towards: the line at the step's end.
                X = tick.ur.pose_at(sim.osc_time_s(sim.step_index + 1))
                q_new, err_p, err_r, iters = ik(UrTracking, X, q_ur, pos_tol=UR_IK_POS_TOL,
                                                rot_tol=UR_IK_ROT_TOL)
                if err_p > UR_IK_POS_TOL or err_r > UR_IK_ROT_TOL:
                    raise MotionError(f"step {tick.k + 1} ({tick.cmd.phase}), ur: IK missed "
                                      f"({err_p * 1e3:.3f} mm, {np.degrees(err_r):.3f} deg "
                                      f"after {iters} iterations)")
                # Velocity feed-forward (as in play()), not across a regenerated line's jump.
                lead = UR_VELOCITY_LEAD * lead_gain if tick.ur.line is lead_line else 0.0
                sim.set_ur_target(q_new + lead * (q_new - q_ur))
                q_ur, lead_line = q_new, tick.ur.line
                sim.set_grippers(tick.cmd.hand_mm, tick.ur.byte)
                ik_s += time.perf_counter() - i0
            c0 = time.perf_counter()
            sim.control_step()
            step_s += time.perf_counter() - c0
            tick = state["tick"]
            k = tick.k
            if k % CLEARANCE_EVERY == 0 or k % n == 0:
                sampler.clearance(sim.state_0.body_q.numpy())
            if k % n == 0:
                sampler.sample_tick(tick)
                if rods is not None:
                    rods.append(ctx.gauge.measure(sim.state_0.body_q.numpy())["stretch_pct"])
            if done_at is None and tick.cmd.phase == "done":
                done_at = k
            if done_at is not None and k >= done_at + settle_steps and k % n == 0:
                break
            if done_at is None and k >= max_steps:
                raise TimeoutError(f"timeout: {tick.cmd.phase} after {args.max_episode_s:g} s "
                                   "(franka {:.1f} mm {:.1f} deg, ur {:.1f} mm {:.1f} deg)"
                                   .format(*state["reach"]))
        if tail_spec is not None:
            tail = _place_tail().run_tail(sim, ctx, sys.modules[__name__], sampler, state, s0,
                                          q_ur, franka, ur, tail_spec, rods)
    except (MotionError, TimeoutError) as exc:
        _finish_recording(sim, ctx.out, f"{name}-{p.intent}", "episode skipped")
        key = "timeout" if isinstance(exc, TimeoutError) else "ik miss"
        return _skip({**row, "latched_at_s": latched_at}, i, str(exc), key)
    finally:
        sim.commander_hook = sim.make_hold_hook()
    lcm_wait = sim.bridge.wait_s - wait0
    timing["render"] = sampler.render_s
    timing["clearance"] = sampler.clearance_s
    timing["commander"] = state["hook_s"] + ik_s
    timing["lcm_wait"] = lcm_wait
    timing["physics"] = step_s - lcm_wait - state["hook_s"]
    if tail is not None:
        timing["tail"] = tail["wall_s"]
        if tail["grasp_lost"]:
            _finish_recording(sim, ctx.out, f"{name}-{p.intent}", "episode skipped")
            return _skip({**row, "tail": _tail_row(tail)}, i, f"tail: {tail['stopped']}",
                         "tail grasp lost")
    recording = _finish_recording(sim, ctx.out, f"{name}-{p.intent}", "episode done")

    t3 = time.perf_counter()
    frames = sampler.frames
    definition = getattr(args, "action_definition", OSC_ACTION_DEFINITION)
    pre = np.array([phase_labels[fr["phase"]] == PREHOLD_PHASE for fr in frames])
    if pre.any():
        u_pre = np.stack([frame_action(fr, "cmd_delta") for fr, m in zip(frames, pre) if m])
        if np.any(u_pre != 0.0):
            bad = np.argwhere(u_pre != 0.0)
            raise RuntimeError(f"pre-hold cmd_delta not 0 ({np.abs(u_pre).max():.2e} at "
                               f"[row, dim] {bad[:4].tolist()})")
    label, metrics, min_clear_mm, contact = _classify(sampler, ctx.thresholds, tangent)
    tip_min_mm = state["tip_min"] * 1e3
    if tail is not None:
        tip_min_mm = min(tip_min_mm, tail["min_franka_tip_mm"])
    clamp_dict = None if clamp is None else _clamp_dict(clamp)
    ur_scale = np.array([fr["ur_excite_scale"] for fr in frames])
    ur_active = np.array([bool(np.any(fr["excite_ur_raw"])) for fr in frames])
    ur_stats = {"frames_active": int(ur_active.sum()),
                "frames_scaled": int((ur_active & (ur_scale < 1.0)).sum()),
                "scale_mean_active": float(ur_scale[ur_active].mean()) if ur_active.any()
                else None,
                "scale_min": float(ur_scale.min())}
    extras_meta = {
        "phase_labels": phase_labels, "intent": p.intent, "outcome": label,
        "perturbation": p.to_dict(), "backend": "osc", "scenario": ctx.scenario,
        "ee_pose_source": "measured finger_tip / tracking frame (FK of the measured joints)",
        "action_source": (
            "knot 1 - knot 0 of the TARGET_CARTESIAN_POSE_TRAJECTORY published at the sample tick "
            "(Franka) / UR line(t + knot dt) - line(t) of the line in force (tracking frame)"
            if definition == "cmd_delta" else
            "knot 1 of the TARGET_CARTESIAN_POSE_TRAJECTORY published at the sample "
            "tick (Franka) / UR line at t + knot dt (tracking frame), minus the "
            "measured pose at t; knot 0 / line(t) kept as sim_cmd_*"),
        "pre_hold_frames": int(pre.sum()), "ur_excite": ur_stats,
        "osc": ctx.osc, "commander": dataclasses.asdict(params), "excitation": ctx.excite,
        "excite_clip": clip, "sample_period_s": n * lcs.SIM_DT_S,
        "time_source": "OSC clock (FRANKA_STATE utime)",
        "pulley_pose_layout": lcs.POSE_LAYOUT, "start_step": s0,
        "start_state": str(ctx.start_state),
        "thresholds": dataclasses.asdict(ctx.thresholds), "clamp": clamp_dict,
        "clearance_source": "2F-85 colliders vs the board plate box and the two pulleys, "
                            f"sampled every {CLEARANCE_EVERY} control steps",
        "franka_tip_clearance_source": "finger_tip z - board plate top z, every control step",
        **opts_meta,
    }
    osc_post = {"backend": np.array("osc"),
                "osc_utime_offset_us": np.int64(sim.bridge.utime_offset_us),
                "min_franka_tip_clearance_mm": np.float32(tip_min_mm)}
    new_row = {}
    if plan is not None:
        new_meta, new_post, new_row = _new_mode_outputs(ctx, frames, metrics, tangent, ep, row,
                                                        tail, rods, int(pre.sum()))
        extras_meta.update(new_meta)
        osc_post.update(new_post)
    path = ctx.out / row["file"]
    writer, track = build_osc_writer(frames, n, pcd=not args.no_pcd, definition=definition)
    steps = frames[-1]["step"] + 1
    detail = (f", tip {tip_min_mm:5.1f} mm, latched "
              + ", ".join(f"{k} {v:.2f} s ({latch_err[k]:.1f} mm)" for k, v in latched_at.items())
              + f", {steps} steps")
    wrap, h_med = _write_episode(i, p, path, writer, label, extras_meta, metrics, min_clear_mm,
                                 contact, clamp_dict, osc_post, args.no_pcd, n, timing, t0, t3,
                                 detail)
    moving = np.array([phase_labels[fr["phase"]].startswith("move:") for fr in frames])
    moving[0] = False  # frame 0 has no previous knot 1
    move = track[moving] if moving.any() else track[1:]
    return {**row, "outcome": label, "steps": steps, "frames": len(frames),
            "duration_s": steps * lcs.SIM_DT_S, "latched_at_s": latched_at,
            "final_target_err_mm": latch_err,
            "tracking_rms_mm": [float(np.sqrt(np.mean(move[:, j] ** 2))) for j in range(2)],
            "tracking_max_mm": [float(move[:, j].max()) for j in range(2)],
            "final_wrap_deg": wrap, "final_h_median_mm": h_med, **_slant_row(metrics),
            "min_board_clearance_mm": min_clear_mm, "board_contact": contact,
            "min_franka_tip_clearance_mm": tip_min_mm, "excite_clip": clip,
            "pre_hold_frames": int(pre.sum()), "ur_excite": ur_stats,
            "clamp_lift_mm": 0.0 if clamp is None else clamp.max_lift_mm,
            "clamp_tilt_scale": 1.0 if clamp is None else clamp.tilt_scale,
            "grasp_ok_final": [bool(v) for v in frames[-1]["grasp_ok"]],
            "osc_stats": sim.bridge.stats(),
            "size_bytes": path.stat().st_size, "recording": recording,
            "timing_s": timing, **new_row, "status": "ok"}


def _place_tail():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "lcs"))
    import place_tail

    return place_tail


def _clamp_dict(clamp) -> dict:
    out = clamp.to_dict()
    if hasattr(clamp, "franka_lift_mm"):
        out["franka_lift_mm"] = clamp.franka_lift_mm
    return out


def _tail_row(tail: dict | None) -> dict | None:
    return None if tail is None else {k: v for k, v in tail.items() if k != "_arrays"}


def _new_mode_outputs(ctx: OscContext, frames: list, metrics: dict, tangent, ep: dict,
                      row: dict, tail: dict | None, rods: list | None, n_pre: int
                      ) -> tuple[dict, dict, dict]:
    """``(sim_meta, npz extras, row)`` additions of the opt-in modes: the place_3 label, the
    per-frame contact metrics, the approach / place_3 sample / tail records."""
    th = ctx.thresholds
    p3 = len(frames) - 1 if tail is None else tail["start_frame"]
    fm3 = frame_metrics(frames[p3]["belt"], frames[p3]["pulley_raw"], th=th)
    label3 = classify(fm3, th)
    slant3 = slant_metrics(frames[p3]["belt"], frames[p3]["pulley_raw"], tangent, th=th)
    nn = np.asarray(metrics["n_neighbour"])
    in_contact = nn >= IN_CONTACT_MIN_NEIGHBOUR
    grasp = np.stack([fr["grasp_ok"] for fr in frames])
    tail_rec = _tail_row(tail)
    if tail_rec is not None:
        tail_contact = in_contact[p3 + 1:]
        tail_rec["contact_frame_frac"] = float(tail_contact.mean()) if tail_contact.size else None
    p3_rec = {"outcome": label3, "frame": p3, "wrap_deg": fm3.wrap_deg,
              "h_median_mm": fm3.h_median_mm, "n_neighbour": fm3.n_neighbour,
              "slant_deg": slant3.slant_deg, "slant_dir": slant3.slant_dir}
    meta = {"approach": ep.get("approach"), "place3_sample": row.get("place3_sample"),
            "tail": tail_rec, "outcome_place3": label3, "place3_frame": p3,
            "place3_metrics": p3_rec, "tangent_episode": np.asarray(tangent).tolist(),
            "pulley_large": {k: np.asarray(v).tolist() for k, v in (ctx.pulley or {}).items()},
            "in_contact_rule": f"n_neighbour >= {IN_CONTACT_MIN_NEIGHBOUR}",
            "label_rule": "outcome = classifier at the last frame; outcome_place3 = classifier "
                          "at place3_frame (end of the place_3 settle window)"}
    post = {"n_neighbour": nn.astype(np.int32), "in_contact": in_contact,
            "outcome_place3": np.array(label3), "place3_frame": np.int64(p3),
            "approach": np.array(json.dumps(ep.get("approach"))),
            "place3_sample": np.array(json.dumps(row.get("place3_sample"))),
            "tail": np.array(json.dumps(tail_rec, default=float))}
    out_row = {"outcome_place3": label3, "place3_frame": p3, "place3": p3_rec,
               "grasp_ok_all": [bool(v) for v in grasp.all(axis=0)],
               "contact_frame_frac": float(in_contact.mean())}
    if rods is not None:
        rod = np.concatenate([np.full(n_pre, np.nan), np.asarray(rods, float)])
        post["rod_stretch_pct"] = rod
        out_row["rod_stretch_pct_max"] = float(np.nanmax(rod))
        out_row["rod_stretch_gain_pct_max"] = float(np.nanmax(rod) - rod[n_pre])
    if tail is not None:
        a = tail["_arrays"]
        n_before = p3 + 1
        post["ur_guard_scale"] = np.concatenate([np.ones(n_before), a["guard_scale"]])
        post["motion_offset_franka"] = np.concatenate([np.zeros((n_before, 6)),
                                                       a["offset_franka"]])
        post["motion_offset_ur"] = np.concatenate([np.zeros((n_before, 6)), a["offset_ur"]])
        out_row["tail"] = tail_rec
    return meta, post, out_row


def _pre_hold(sim, ctx: OscContext, sampler, state: dict, steps: int, s0: int, q_ur,
              ur: UrLineCommander, hand_mm, byte) -> None:
    """``steps`` control steps under the settle's hold (Franka latched, UR target unchanged),
    sampled like the episode; ``k = step - s0 < 0``. Knots and the UR line are constant, so the
    ``cmd_delta`` of these frames is exactly 0."""
    hold = sim.commander_hook
    n, u_coords = ctx.n, sim.arm_coords()[1]
    p_u, q_u = ur.to_tool0(UrTracking.fk(q_ur))
    line = UrLine(p0=p_u, q0=q_u, t0=0.0, p1=p_u, q1=q_u, t1=0.0)

    def pre_hook(step, t, joint_q, body_q):
        msg = hold(step, t, joint_q, body_q)
        pos, quat, times = parse_saved_traj_message(msg)
        cmd = FrankaCommand(knots_pos=pos, knots_quat=quat, times=times, hold=True,
                            target_index=0, phase=PREHOLD_PHASE, hand_mm=hand_mm)
        urc = UrCommand(line=line, regenerated=False, byte=byte, t=float(t),
                        X_tool0_tracking=ctx.x_tool0)
        state["tick"] = OscTick(step - s0, t, sim.franka_measured_pose7(joint_q),
                                UrTracking.fk(joint_q[u_coords]), cmd, urc)
        return msg

    sim.commander_hook = pre_hook
    try:
        for _ in range(steps):
            sim.control_step()
            k = state["tick"].k
            if k % CLEARANCE_EVERY == 0 or k % n == 0:
                sampler.clearance(sim.state_0.body_q.numpy())
            if k % n == 0:
                sampler.sample_tick(state["tick"])
    finally:
        sim.commander_hook = hold
        state["tick"] = None


def main() -> int:
    args = create_parser().parse_args()
    configure_logging()
    collect(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
