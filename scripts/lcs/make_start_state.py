#!/usr/bin/env python3
"""Run the nominal in-process pick and save the ``pre_place_1`` start-state snapshot.

Builds :class:`RoundBeltOfflineSimulation` from the scene defaults, plays magna's waypoints
``[current pose] -> pre_pick_0 -> pre_pick_1 -> pick -> post_pick -> pre_place_1`` plus a hold,
checks that both grippers hold the belt off the holder, and saves the snapshot every LCS episode
restores. No magna, no LCM.

Run:
    uv run python scripts/lcs/make_start_state.py
    uv run python scripts/lcs/make_start_state.py --record data/lcs/recordings
"""

from __future__ import annotations

import argparse
import math
import sys
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


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stdout, level=level,
               format="<level>{level: <7}</level> | <level>{message}</level>")


def pose_error(measured, target) -> tuple[float, float]:
    """(mm, deg) between two 4x4 poses."""
    return (float(((measured[:3, 3] - target[:3, 3]) ** 2).sum() ** 0.5) * 1e3,
            math.degrees(rot_angle(measured, target)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="snapshot .npz path")
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
