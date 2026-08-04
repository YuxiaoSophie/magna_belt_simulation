"""Pick the cloth belt, hook its opposite side on the small pulley, then place the held side on the large pulley."""
from __future__ import annotations

import numpy as np
import newton
import newton.examples

from timing_belt import (
    BELT_APPROACH_POS,
    BELT_GRASP_POS,
    BELT_OUTER_MAJOR_DIAMETER,
    BELT_PLACE_DOWN,
    BELT_RADIUS,
    GRIPPER_DOWN_QUAT,
    SMALL_PULLEY_SHEAVE_RADIUS,
    SMALL_PULLEY_WORLD_CENTER,
    Example,
    smoothstep,
)

START_HOLD = 0.50
DESCEND_TIME = 2.00
CLOSE_TIME = 1.25
GRASP_SETTLE = 0.80
LIFT_TIME = 1.75
TRAVERSE_TO_SMALL_HOOK_TIME = 2.75
LOWER_SMALL_HOOK_TIME = 2.25
SMALL_HOOK_SETTLE = 1.50
REACH_EDGE_TIME = 1.50
LOWER_TO_LARGE_TIME = 1.75
LARGE_SETTLE = 1.25
OPEN_TIME = 1.00
RETREAT_TIME = 1.50

CARRY_CLEARANCE = 0.25
CARRY_Z = BELT_GRASP_POS[2] + CARRY_CLEARANCE
CARRY_POS = (BELT_GRASP_POS[0], BELT_GRASP_POS[1], CARRY_Z)

SMALL_OUTER_RIM_X = SMALL_PULLEY_WORLD_CENTER[0] - SMALL_PULLEY_SHEAVE_RADIUS - BELT_RADIUS
SMALL_HOOK_Z = SMALL_PULLEY_WORLD_CENTER[2] + BELT_RADIUS
SMALL_FREE_SIDE_HOOK_POS = (
    SMALL_OUTER_RIM_X,
    SMALL_PULLEY_WORLD_CENTER[1],
    SMALL_HOOK_Z,
)
SMALL_HOOK_HELD_DOWN = (
    SMALL_FREE_SIDE_HOOK_POS[0] + BELT_OUTER_MAJOR_DIAMETER,
    SMALL_FREE_SIDE_HOOK_POS[1],
    SMALL_FREE_SIDE_HOOK_POS[2],
)
CARRY_OVER_SMALL_HOOK = (
    SMALL_HOOK_HELD_DOWN[0],
    SMALL_HOOK_HELD_DOWN[1],
    CARRY_Z,
)

LARGE_STRETCH_EXTRA_X = 0.035
LARGE_APPROACH_Z_OFFSET = 0.015
BELT_PLACE_LARGE_HOOK = (
    BELT_PLACE_DOWN[0] + LARGE_STRETCH_EXTRA_X,
    BELT_PLACE_DOWN[1],
    BELT_PLACE_DOWN[2] + LARGE_APPROACH_Z_OFFSET,
)
LARGE_MID_CLEARANCE = 0.06
LARGE_PRE_EDGE = (
    BELT_PLACE_DOWN[0],
    BELT_PLACE_LARGE_HOOK[1],
    BELT_PLACE_LARGE_HOOK[2] + LARGE_MID_CLEARANCE,
)
LARGE_OVER_EDGE = (
    BELT_PLACE_LARGE_HOOK[0],
    BELT_PLACE_LARGE_HOOK[1],
    BELT_PLACE_LARGE_HOOK[2] + LARGE_MID_CLEARANCE,
)
CARRY_OVER_LARGE = (
    BELT_PLACE_LARGE_HOOK[0],
    BELT_PLACE_LARGE_HOOK[1],
    CARRY_Z,
)


def _lerp3(a, b, alpha: float) -> tuple[float, float, float]:
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    p = (1.0 - alpha) * a_np + alpha * b_np
    return float(p[0]), float(p[1]), float(p[2])


class PickAndPlaceExample(Example):
    def solve_gripper_targets(self):
        t = float(self.sim_time)
        open_q = np.asarray(self.gripper_open_values, dtype=np.float32)
        closed_q = np.asarray(self.gripper_closed_values, dtype=np.float32)

        t0 = START_HOLD
        t1 = t0 + DESCEND_TIME
        t2 = t1 + CLOSE_TIME
        t3 = t2 + GRASP_SETTLE
        t4 = t3 + LIFT_TIME
        t5 = t4 + TRAVERSE_TO_SMALL_HOOK_TIME
        t6 = t5 + LOWER_SMALL_HOOK_TIME
        t7 = t6 + SMALL_HOOK_SETTLE
        t8b = t7 + REACH_EDGE_TIME
        t8 = t8b + LOWER_TO_LARGE_TIME
        t9 = t8 + LARGE_SETTLE
        t10 = t9 + OPEN_TIME
        t11 = t10 + RETREAT_TIME

        if t < t0:
            tcp_pos, grip_q = BELT_APPROACH_POS, open_q
        elif t < t1:
            a = float(smoothstep((t - t0) / DESCEND_TIME))
            tcp_pos, grip_q = _lerp3(BELT_APPROACH_POS, BELT_GRASP_POS, a), open_q
        elif t < t2:
            a = float(smoothstep((t - t1) / CLOSE_TIME))
            tcp_pos, grip_q = BELT_GRASP_POS, open_q + a * (closed_q - open_q)
        elif t < t3:
            tcp_pos, grip_q = BELT_GRASP_POS, closed_q
        elif t < t4:
            a = float(smoothstep((t - t3) / LIFT_TIME))
            tcp_pos, grip_q = _lerp3(BELT_GRASP_POS, CARRY_POS, a), closed_q
        elif t < t5:
            a = float(smoothstep((t - t4) / TRAVERSE_TO_SMALL_HOOK_TIME))
            tcp_pos, grip_q = _lerp3(CARRY_POS, CARRY_OVER_SMALL_HOOK, a), closed_q
        elif t < t6:
            a = float(smoothstep((t - t5) / LOWER_SMALL_HOOK_TIME))
            tcp_pos, grip_q = _lerp3(CARRY_OVER_SMALL_HOOK, LARGE_PRE_EDGE, a), closed_q
        elif t < t7:
            tcp_pos, grip_q = LARGE_PRE_EDGE, closed_q
        elif t < t8b:
            a = float(smoothstep((t - t7) / REACH_EDGE_TIME))
            tcp_pos, grip_q = _lerp3(LARGE_PRE_EDGE, LARGE_OVER_EDGE, a), closed_q
        elif t < t8:
            a = float(smoothstep((t - t8b) / LOWER_TO_LARGE_TIME))
            tcp_pos, grip_q = _lerp3(LARGE_OVER_EDGE, BELT_PLACE_LARGE_HOOK, a), closed_q
        elif t < t9:
            tcp_pos, grip_q = BELT_PLACE_LARGE_HOOK, closed_q
        elif t < t10:
            a = float(smoothstep((t - t9) / OPEN_TIME))
            tcp_pos, grip_q = BELT_PLACE_LARGE_HOOK, closed_q + a * (open_q - closed_q)
        elif t < t11:
            a = float(smoothstep((t - t10) / RETREAT_TIME))
            tcp_pos, grip_q = _lerp3(BELT_PLACE_LARGE_HOOK, CARRY_OVER_LARGE, a), open_q
        else:
            tcp_pos, grip_q = CARRY_OVER_LARGE, open_q

        self.set_task_target(tcp_pos, GRIPPER_DOWN_QUAT, grip_q)


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    viewer._pause = False
    newton.examples.run(PickAndPlaceExample(viewer, args), args)
