from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import round_belt_command as command


class RecordingCommand(command.PickAndPlaceExample):
    def __init__(self):
        self.sim_time = 0.0
        self.gripper_open_values = [0.0, 0.0]
        self.gripper_closed_values = [1.0, 1.0]
        self.position = None
        self.gripper = None
        self.strength = None

    def set_grasp_strength(self, strength):
        self.strength = float(strength)

    def set_task_target(self, position, rotation, gripper_values):
        self.position = tuple(float(v) for v in position)
        self.gripper = np.asarray(gripper_values, dtype=np.float32)


class TestRoundBeltCommand(unittest.TestCase):
    def setUp(self):
        self.example = RecordingCommand()

    def solve_at(self, sim_time):
        self.example.sim_time = sim_time
        self.example.solve_gripper_targets()
        return self.example

    def test_small_pulley_is_seated_before_forward_motion(self):
        small_seat_start = sum(
            (
                command.START_HOLD,
                command.DESCEND_TIME,
                command.CLOSE_TIME,
                command.GRASP_SETTLE,
                command.LIFT_TIME,
                command.MOVE_TO_SMALL_TIME,
                command.LOWER_TO_SMALL_TIME,
                command.LOOP_SETTLE_TIME,
                command.SMALL_PRESS_TIME,
                command.LOWER_SMALL_INTO_GROOVE_TIME,
            )
        )
        result = self.solve_at(small_seat_start + 0.5 * command.SMALL_SEAT_TIME)

        np.testing.assert_allclose(result.position, command.BELT_SMALL_SEAT_DOWN)
        np.testing.assert_allclose(result.gripper, [1.0, 1.0])
        self.assertEqual(result.strength, 1.0)

    def test_forward_motion_stays_in_groove_plane(self):
        forward_start = sum(
            (
                command.START_HOLD,
                command.DESCEND_TIME,
                command.CLOSE_TIME,
                command.GRASP_SETTLE,
                command.LIFT_TIME,
                command.MOVE_TO_SMALL_TIME,
                command.LOWER_TO_SMALL_TIME,
                command.LOOP_SETTLE_TIME,
                command.SMALL_PRESS_TIME,
                command.LOWER_SMALL_INTO_GROOVE_TIME,
                command.SMALL_SEAT_TIME,
            )
        )
        result = self.solve_at(
            forward_start + 0.5 * command.MOVE_SMALL_TO_LARGE_TIME
        )

        expected_midpoint = 0.5 * (
            np.asarray(command.BELT_SMALL_SEAT_DOWN)
            + np.asarray(command.BELT_LARGE_PLACE_DOWN)
        )
        np.testing.assert_allclose(result.position, expected_midpoint)
        self.assertAlmostEqual(
            result.position[2],
            command.BELT_SMALL_SEAT_DOWN[2],
        )
        self.assertEqual(result.strength, 1.0)

    def test_final_pose_releases_above_large_pulley(self):
        result = self.solve_at(command.MOTION_END_TIME + 0.1)

        np.testing.assert_allclose(result.position, command.RETREAT_OVER_LARGE)
        np.testing.assert_allclose(result.gripper, [0.0, 0.0])
        self.assertEqual(result.strength, 0.0)
        self.assertGreater(
            command.BELT_LARGE_PLACE_DOWN[0],
            command.BELT_SMALL_SEAT_DOWN[0],
        )
        self.assertGreaterEqual(
            command.DEFAULT_NUM_FRAMES,
            int(command.MOTION_END_TIME * 60.0),
        )


if __name__ == "__main__":
    unittest.main()
