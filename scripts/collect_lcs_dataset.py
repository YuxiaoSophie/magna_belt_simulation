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
2-knot line by per-step IK. ``state`` holds the measured poses, ``action`` = the commanded pose one
knot dt ahead minus the measured pose at the sample tick: knot 1 of the published command (Franka)
and the UR line at t + dt (``action_definition`` ``knot1_minus_measured``); knot 0 / the line at t
are kept as ``sim_cmd_*``. Optional bounded excitation of Franka knots 1..6.
The board clearance guard models only the 2F-85 (UR); the Franka is bounded by the excitation
cap + z-floor and its measured finger_tip-to-plate clearance is recorded.

``--backend position``: both arms position-driven along a precomputed joint trajectory, ``state``
= commanded targets, ``action`` = cmd(t+1) - cmd(t). No magna, no LCM.

Run:
    uv run python scripts/collect_lcs_dataset.py --episodes 40 --seed 0
    uv run python scripts/collect_lcs_dataset.py --episodes 4 --seed 1 --dry-run
    uv run python scripts/collect_lcs_dataset.py --episodes 4 --record --out data/lcs/smoke
    uv run python scripts/collect_lcs_dataset.py --backend position --episodes 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
from round_belt_task.arm_kinematics import FrankaTip, UrTracking, ik, pose7_from_mat
from round_belt_task.commander import (
    EXCITE_MODES,
    EXCITE_RAMP_S,
    CommanderParams,
    FrankaTarget,
    FrankaWaypointCommander,
    OuExcitation,
    UrLineCommander,
    UrTarget,
    angular_distance,
    draw_excite,
    mat3_to_quat,
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
    classify_episode,
    slant_episode,
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
OSC_ACTION_DEFINITION = "knot1_minus_measured"
EXCITE_DEFAULTS = {"ou": (1.5, 1.0), "white": (4.0, 2.0)}  # (pos mm, rot deg)


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
    p.add_argument("--max-episode-s", type=float, default=20.0,
                   help="osc: skip an episode whose targets are not all reached by then")
    p.add_argument("--scenario", choices=SCENARIOS, default=None,
                   help="osc: pure_translation = one Franka target 30 mm -x, 10 mm +z, UR still; "
                        "nominal = the unperturbed waypoints (intents ignored, excitation off)")
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
    perts = plan_episodes(args, ranges, scenario)
    nominal = load_pre_mpc_segment(args.params, first=FIRST, last=LAST)
    tangent = pert.belt_tangent(nominal)
    if args.dry_run:
        print(perturbation_table(perts))
        return {"episodes": [p.to_dict() for p in perts]}

    out = args.out or DEFAULT_OUT_ROOT / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.label}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start_state = resolve_start_state(args)
    start_meta: dict = {"path": None if args.fresh_pick else str(start_state)}
    snap = None
    if not args.fresh_pick:
        snap = sim_snapshot.load(start_state)
        start_meta.update(sha256=_sha256(start_state), label=snap.meta.get("label"),
                          notes=snap.meta.get("notes"), created=snap.meta.get("created"),
                          git_commit=snap.meta.get("git_commit"),
                          step_index=snap.step_index, sim_time=snap.sim_time)
    excite = excitation_config(args, scenario) if backend == "osc" else None
    index = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git": _git(), "backend": backend, "scenario": scenario, "start_state": start_meta,
        "sample_period_s": n * lcs.SIM_DT_S,
        "sample_steps": n, "thresholds": dataclasses.asdict(thresholds),
        "action_definition": OSC_ACTION_DEFINITION if backend == "osc" else None,
        "belt_sampling": lcs.BELT_SAMPLING,
        "ranges": pert.ranges_to_dict(ranges), "labels": list(LABELS),
        "belt_tangent": tangent, "min_clearance_mm": args.min_clearance,
        "clearance_every": CLEARANCE_EVERY, "excitation": excite, "episodes": [],
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
                             start_state=start_state)
        for i, p in enumerate(perts):
            if ctx is None:
                row = run_episode(sim, i, p, args, out, n, thresholds, tangent, snap, initial,
                                  nominal, board, gripper)
            else:
                try:
                    row = run_osc_episode(sim, ctx, i, p)
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
    params: CommanderParams = dataclasses.field(default_factory=CommanderParams)
    x_tool0: np.ndarray = dataclasses.field(default_factory=x_tool0_tracking)

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


class OscEpisodeSampler(EpisodeSampler):
    """Frames for the osc backend: measured state + the command published at the sample tick."""

    def __init__(self, sim, ctx: OscContext, phase_labels: list[str]) -> None:
        super().__init__(sim, ctx.n, pcd=not ctx.args.no_pcd, traj_phase=np.zeros(0, np.int64),
                         phase_offset=0, board=ctx.board, gripper=ctx.gripper)
        self.ctx = ctx
        self.phase_index = {label: i for i, label in enumerate(phase_labels)}

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


def tracking_errors(frames: list[dict]) -> np.ndarray:
    """``(T, 2)`` mm, 0 at t=0: Franka ``|ee_t - knot1_{t-1}|``, UR ``|ee_u_t - line_{t-1}(t)|``.

    Equals ``|ee_t - (ee_{t-1} + u_{t-1})|`` (position part) under ``knot1_minus_measured``.
    """
    out = np.zeros((len(frames), 2))
    for t in range(1, len(frames)):
        prev, fr = frames[t - 1], frames[t]
        out[t, 0] = np.linalg.norm(fr["ee_f"][:3] - prev["knot1"][:3]) * 1e3
        out[t, 1] = np.linalg.norm(fr["ee_u"][:3] - prev["ur_t1"][:3]) * 1e3
    return out


def build_osc_writer(frames: list[dict], n: int, pcd: bool
                     ) -> tuple[lcs.EpisodeWriter, np.ndarray]:
    writer = lcs.EpisodeWriter(sample_steps=n, action_definition=OSC_ACTION_DEFINITION)
    track = tracking_errors(frames)
    for t, fr in enumerate(frames):
        state = lcs.state_vector(fr["q_f"], fr["q_u"], fr["v_f"], fr["v_u"], fr["ee_f"],
                                 fr["ee_u"])
        # Every row, the last included: the command published at this tick (no zero padding).
        action = lcs.action_vector(fr["ee_f"], fr["knot1"], fr["ee_u"], fr["ur_t1"])
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
                         })
    return writer, track


def _scenario_targets(sim) -> tuple[list[FrankaTarget], list[UrTarget | None]]:
    q_f, q_u = sim.arm_positions()
    X_f, X_u = FrankaTip.fk(q_f), UrTracking.fk(q_u)
    franka = FrankaTarget(label="pure_translation", pos=X_f[:3, 3] + PURE_TRANSLATION_M,
                          quat_wxyz=mat3_to_quat(X_f[:3, :3]), hand_mm=None,
                          dwell_s=PURE_TRANSLATION_DWELL_S)
    return [franka], [UrTarget(pos=X_u[:3, 3].copy(), quat_wxyz=mat3_to_quat(X_u[:3, :3]),
                               byte=None)]


def _guard(sim, ctx: OscContext, p: pert.Perturbation):
    """Perturbed, clamped waypoints + the path check on straight lines from the measured poses."""
    args = ctx.args
    waypoints, clamp = clr.clamp_waypoints(ctx.nominal, p, ctx.tangent, ctx.board, ctx.gripper,
                                           min_clearance=args.min_clearance * 1e-3)
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


def run_osc_episode(sim, ctx: OscContext, i: int, p: pert.Perturbation) -> dict:
    args, n, params = ctx.args, ctx.n, ctx.params
    t0 = time.perf_counter()
    timing: dict[str, float] = {}
    name = f"episode_{i:04d}"
    row = {"file": f"{name}.npz", "intent": p.intent, "perturbation": p.to_dict(),
           "backend": "osc", "scenario": ctx.scenario, "excite": bool(ctx.excite["on"])}
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
        waypoints, clamp = _guard(sim, ctx, p)
        row["clamp"] = clamp.to_dict()
        f_targets, u_targets = targets_from_waypoints([w for w in waypoints if w.label != FIRST])
    timing["guard"] = time.perf_counter() - t2
    last = len(f_targets) - 1
    phase_labels = [f"{kind}:{t.label}" for t in f_targets for kind in ("move", "hold")]
    phase_labels.append("done")

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
    clip = {"active_steps": 0, "floor_steps": 0, "cap_steps": 0}
    u_coords = sim.arm_coords()[1]
    plate_top = ctx.plate_top_z
    s0 = sim.step_index + 1  # the hook runs after control_step advanced the counter
    t_start = sim.osc_time_s(s0)
    state = {"tick": None, "excite": None, "hook_s": 0.0, "tip_min": float("inf")}
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
        if ou is not None:
            # Fade so the offset is 0 by the place_3 reach: a held offset would shift the latch.
            fade = quiet or (franka.index == last and float(
                np.linalg.norm(pos_f - f_targets[last].pos)) <= fade_m)
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
        ur_cmd = ur.tick(t, X_u, u_targets[cmd.target_index])
        state["tick"] = OscTick(k, t, ee_f, X_u, cmd, ur_cmd)
        msg = saved_traj_message(round(t * 1e6), cmd.knots_pos, cmd.knots_quat, cmd.times)
        state["hook_s"] += time.perf_counter() - h0
        return msg

    if args.record:
        sim.start_recording(ctx.out / "recordings", f"{name}-{p.intent}")
    sampler = OscEpisodeSampler(sim, ctx, phase_labels)
    q_ur = sim.arm_targets()[1]
    lead_gain = sim.arm_kd / sim.arm_ke / sim.frame_dt
    lead_line = None
    settle_steps = round(args.settle_s / sim.frame_dt)
    max_steps = round(args.max_episode_s / sim.frame_dt)
    ik_s = step_s = 0.0
    wait0 = sim.bridge.wait_s
    done_at = None
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
            if done_at is None and tick.cmd.phase == "done":
                done_at = k
            if done_at is not None and k >= done_at + settle_steps and k % n == 0:
                break
            if done_at is None and k >= max_steps:
                raise TimeoutError(f"timeout: {tick.cmd.phase} after {args.max_episode_s:g} s "
                                   "(franka {:.1f} mm {:.1f} deg, ur {:.1f} mm {:.1f} deg)"
                                   .format(*state["reach"]))
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
    recording = _finish_recording(sim, ctx.out, f"{name}-{p.intent}", "episode done")

    t3 = time.perf_counter()
    frames = sampler.frames
    label, metrics, min_clear_mm, contact = _classify(sampler, ctx.thresholds, ctx.tangent)
    tip_min_mm = state["tip_min"] * 1e3
    clamp_dict = None if clamp is None else clamp.to_dict()
    extras_meta = {
        "phase_labels": phase_labels, "intent": p.intent, "outcome": label,
        "perturbation": p.to_dict(), "backend": "osc", "scenario": ctx.scenario,
        "ee_pose_source": "measured finger_tip / tracking frame (FK of the measured joints)",
        "action_source": "knot 1 of the TARGET_CARTESIAN_POSE_TRAJECTORY published at the sample "
                         "tick (Franka) / UR line at t + knot dt (tracking frame), minus the "
                         "measured pose at t; knot 0 / line(t) kept as sim_cmd_*",
        "osc": ctx.osc, "commander": dataclasses.asdict(params), "excitation": ctx.excite,
        "excite_clip": clip, "sample_period_s": n * lcs.SIM_DT_S,
        "time_source": "OSC clock (FRANKA_STATE utime)",
        "pulley_pose_layout": lcs.POSE_LAYOUT, "start_step": s0,
        "start_state": str(ctx.start_state),
        "thresholds": dataclasses.asdict(ctx.thresholds), "clamp": clamp_dict,
        "clearance_source": "2F-85 colliders vs the board plate box and the two pulleys, "
                            f"sampled every {CLEARANCE_EVERY} control steps",
        "franka_tip_clearance_source": "finger_tip z - board plate top z, every control step",
    }
    osc_post = {"backend": np.array("osc"),
                "osc_utime_offset_us": np.int64(sim.bridge.utime_offset_us),
                "min_franka_tip_clearance_mm": np.float32(tip_min_mm)}
    path = ctx.out / row["file"]
    writer, track = build_osc_writer(frames, n, pcd=not args.no_pcd)
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
            "clamp_lift_mm": 0.0 if clamp is None else clamp.max_lift_mm,
            "clamp_tilt_scale": 1.0 if clamp is None else clamp.tilt_scale,
            "grasp_ok_final": [bool(v) for v in frames[-1]["grasp_ok"]],
            "osc_stats": sim.bridge.stats(),
            "size_bytes": path.stat().st_size, "recording": recording,
            "timing_s": timing, "status": "ok"}


def main() -> int:
    args = create_parser().parse_args()
    configure_logging()
    collect(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
