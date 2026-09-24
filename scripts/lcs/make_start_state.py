#!/usr/bin/env python3
"""Run the nominal in-process pick and save the ``pre_place_1`` start-state snapshot.

``--backend position`` (default) builds :class:`RoundBeltOfflineSimulation` from the scene
defaults, plays magna's waypoints ``[current pose] -> pre_pick_0 -> ... -> pre_place_1`` plus a
hold, checks that both grippers hold the belt off the holder, and saves the snapshot every LCS
episode restores. No magna, no LCM.

``--backend osc`` loads that snapshot into :class:`RoundBeltOscSimulation`, settles it 2 s under
magna's OSC hold (the OSC runs as a child process on ``--lcm-url``, a private group) and saves
``pre_place_1_osc.npz``.

Run:
    uv run python scripts/lcs/make_start_state.py
    uv run python scripts/lcs/make_start_state.py --record data/lcs/recordings
    uv run python scripts/lcs/make_start_state.py --backend osc
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from loguru import logger

from round_belt_task.arm_kinematics import rot_angle
from round_belt_task.constants import ARM_TARGET_KD, ARM_TARGET_KE
from round_belt_task.offline_simulation import (
    PICK_FIRST,
    PICK_HOLD_S,
    PICK_LAST,
    RoundBeltOfflineSimulation,
)
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML
from task_common import sim_snapshot

RECORD_LABEL = "nominal_pick"
DEFAULT_OUT = sim_snapshot.DEFAULT_START_STATE_DIR / f"{PICK_LAST}.npz"
DEFAULT_RECORD_DIR = REPO_ROOT / "data" / "lcs" / "recordings"
DEFAULT_OSC_OUT = sim_snapshot.DEFAULT_START_STATE_DIR / f"{PICK_LAST}_osc.npz"
DEFAULT_OSC_URL = "udpm://239.255.76.83:7683?ttl=0"
OSC_SETTLE_S = 2.0
OSC_LOG_ERRORS = ("resetting", "Exception caught")


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stdout, level=level,
               format="<level>{level: <7}</level> | <level>{message}</level>")


def pose_error(measured, target) -> tuple[float, float]:
    """(mm, deg) between two 4x4 poses."""
    return (float(((measured[:3, 3] - target[:3, 3]) ** 2).sum() ** 0.5) * 1e3,
            math.degrees(rot_angle(measured, target)))


def make_osc_start_state(args: argparse.Namespace) -> int:
    """``pre_place_1.npz`` settled under magna's OSC hold -> ``pre_place_1_osc.npz``."""
    from round_belt_task.arm_kinematics import FrankaTip
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    out = args.out or DEFAULT_OSC_OUT
    snap = sim_snapshot.load(args.start)
    log_path = Path(tempfile.mkdtemp(prefix="make_start_state_osc_")) / "osc.log"
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, arm_ke=args.arm_ke,
                                       arm_kd=args.arm_kd)
    reason = "error"
    try:
        warm = sim.start_osc(log_path)
        logger.info(f"[OSC] {sim.osc.describe()} warm-up {warm:.2f} s, log {log_path}")
        sim.restore(snap, settle_steps=0)
        before = FrankaTip.fk(sim.arm_positions()[0])
        steps = round(OSC_SETTLE_S / sim.frame_dt)
        grasp = sim.settle(steps)
        after = FrankaTip.fk(sim.arm_positions()[0])
        drift_mm, drift_deg = pose_error(after, before)
        stats = sim.bridge.stats()
        logger.info(f"[OSC] settle {steps} steps: finger_tip drift {drift_mm:.3f} mm "
                    f"{drift_deg:.3f} deg, belt min z {grasp.belt_min_z:.4f}, held "
                    f"{grasp.held()}; replies {stats['replies']} stale "
                    f"{stats['stale_replies']} republished {stats['republished']}, wait mean/max "
                    f"{stats['wait_mean_ms']:.3f}/{stats['wait_max_ms']:.3f} ms")
        logger.info(f"[OSC] grasp: {grasp.describe()}")
        sim.stop_osc()
        bad = [e for e in OSC_LOG_ERRORS if e in sim.osc.log_text()]
        failures = grasp.failures()
        if not all(grasp.held()):
            failures.append(f"held {grasp.held()}")
        if bad:
            reason = "osc_log"
            print(f"OSC LOG ERROR: {bad} in {log_path}")
            return 1
        if failures:
            reason = "grasp_failed"
            print(f"GRASP FAILED: {'; '.join(failures)} (drift {drift_mm:.3f} mm "
                  f"{drift_deg:.3f} deg; {grasp.describe()})")
            return 1
        offset = sim.bridge.utime_offset_us
        snap_out = sim_snapshot.capture(sim, PICK_LAST, notes=(
            f"{PICK_LAST} snapshot settled {OSC_SETTLE_S:g} s under magna OSC hold; utime offset "
            f"{offset}; drift {drift_mm:.3f} mm {drift_deg:.3f} deg; {grasp.describe()}"))
        path = sim_snapshot.save(snap_out, out)
        logger.success(f"[OSC] saved {path} (step {snap_out.step_index}, "
                       f"t {snap_out.sim_time:.2f} s)")
        reason = "finished"
        return 0
    finally:
        sim.close(reason)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("position", "osc"), default="position")
    parser.add_argument("--lcm-url", default=DEFAULT_OSC_URL,
                        help="private LCM URL for the OSC backend (never magna's shared group)")
    parser.add_argument("--start", type=Path, default=DEFAULT_OUT,
                        help="osc backend: the position-backend snapshot to settle")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"snapshot .npz path (default {DEFAULT_OUT.name} / "
                             f"{DEFAULT_OSC_OUT.name})")
    parser.add_argument("--record", nargs="?", const=str(DEFAULT_RECORD_DIR), default=None,
                        metavar="DIR", help=f"also record the run (default {DEFAULT_RECORD_DIR})")
    parser.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML,
                        help="magna round_belt_controller_params_sim.yaml (read-only)")
    parser.add_argument("--arm-ke", type=float, default=ARM_TARGET_KE)
    parser.add_argument("--arm-kd", type=float, default=ARM_TARGET_KD)
    parser.add_argument("--no-velocity-lead", action="store_false", dest="velocity_lead",
                        help="plain position targets (no kd/ke * qdot feed-forward)")
    parser.add_argument("--hold", type=float, default=PICK_HOLD_S, help="hold at the end [s]")
    args = parser.parse_args()
    configure_logging()
    if args.backend == "osc":
        return make_osc_start_state(args)
    args.out = args.out or DEFAULT_OUT

    sim = RoundBeltOfflineSimulation.build(arm_ke=args.arm_ke, arm_kd=args.arm_kd,
                                           velocity_lead=args.velocity_lead)
    reason = "error"
    try:
        if args.record is not None:
            sim.start_recording(Path(args.record), RECORD_LABEL)
        waypoints, traj, result = sim.nominal_pick(args.params, args.hold)
        segments = ", ".join(f"{s.end.label} {s.duration_s:.2f}+{s.hold_s:.2f}"
                             for s in traj.segments)
        logger.info(f"[PICK] segments (move+hold s): {segments}; total {traj.t[-1]:.2f} s, "
                    f"{len(traj)} steps in {result.wall_s:.1f} s wall")
        logger.info(f"[PICK] IK {traj.stats}")
        logger.info(f"[PICK] tracking {result.describe()}")
        franka, ur = sim.ee_poses()
        target = waypoints[-1]
        (fe_mm, fe_deg), (ue_mm, ue_deg) = (pose_error(franka, target.franka_mat()),
                                            pose_error(ur, target.ur_mat()))
        logger.info(f"[PICK] at {PICK_LAST}: franka {fe_mm:.2f} mm {fe_deg:.2f} deg, ur "
                    f"{ue_mm:.2f} mm {ue_deg:.2f} deg from the waypoint")
        grasp = sim.grasp_state()
        failures = grasp.failures()
        logger.info(f"[PICK] grasp: {grasp.describe()}")
        if failures:
            reason = "grasp_failed"
            print(f"GRASP FAILED: {'; '.join(failures)}")
            return 1
        snap = sim_snapshot.capture(sim, PICK_LAST, notes=(
            f"nominal in-process pick {PICK_FIRST}->{PICK_LAST}, arm ke {args.arm_ke:g} kd "
            f"{args.arm_kd:g}, velocity lead {args.velocity_lead}; {grasp.describe()}"))
        path = sim_snapshot.save(snap, args.out)
        logger.success(f"[PICK] saved {path} (step {snap.step_index}, t {snap.sim_time:.2f} s)")
        reason = "finished"
        return 0
    finally:
        sim.close(reason)


if __name__ == "__main__":
    sys.exit(main())
