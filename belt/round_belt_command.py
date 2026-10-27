"""
Pick the belt from the table, hook the OPPOSITE side of the loop around the
small pulley, then place the GRASPED side onto the large pulley.
"""

from __future__ import annotations

import numpy as np

import newton
import newton.examples

from round_belt import (
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

# Motion timing (s)
START_HOLD = 0.50
DESCEND_TIME = 2.00
CLOSE_TIME = 1.25
GRASP_SETTLE = 0.60
LIFT_TIME = 1.50

# Move the held +X vertex to the location that places the free -X vertex over
# the small pulley's outer rim.
TRAVERSE_TO_SMALL_HOOK_TIME = 2.50

# Lower the whole loop. The opposite side catches the small pulley.
LOWER_SMALL_HOOK_TIME = 2.00
SMALL_HOOK_SETTLE = 1.20

# Transfer of the held side onto the large pulley.
REACH_EDGE_TIME = 1.25
LOWER_TO_LARGE_TIME = 1.50
LARGE_SETTLE = 1.00

OPEN_TIME = 1.00
RETREAT_TIME = 1.50

# Carry geometry
CARRY_CLEARANCE = 0.25
CARRY_Z = BELT_GRASP_POS[2] + CARRY_CLEARANCE

CARRY_POS = (
    BELT_GRASP_POS[0],
    BELT_GRASP_POS[1],
    CARRY_Z,
)

# Small-pulley hook geometry
SMALL_OUTER_RIM_X = (
    SMALL_PULLEY_WORLD_CENTER[0]
    - SMALL_PULLEY_SHEAVE_RADIUS
    - BELT_RADIUS
)

SMALL_HOOK_Z = SMALL_PULLEY_WORLD_CENTER[2] + BELT_RADIUS

# Desired position of the belt's opposite/free -X vertex when hooking.
SMALL_FREE_SIDE_HOOK_POS = (
    SMALL_OUTER_RIM_X,
    SMALL_PULLEY_WORLD_CENTER[1],
    SMALL_HOOK_Z,
)

# The gripper still holds the belt's +X vertex. The +X and -X belt vertices are
# separated by the belt's full major diameter. Thus, when the free -X side is
# at the small pulley, the held side must be this far to +X.
SMALL_HOOK_HELD_DOWN = (
    SMALL_FREE_SIDE_HOOK_POS[0] + BELT_OUTER_MAJOR_DIAMETER,
    SMALL_FREE_SIDE_HOOK_POS[1],
    SMALL_FREE_SIDE_HOOK_POS[2],
)

# Same X/Y as SMALL_HOOK_HELD_DOWN but at carrying height. The robot travels
# here first and then lowers vertically. This keeps the gripper away from the
# small pulley.
CARRY_OVER_SMALL_HOOK = (
    SMALL_HOOK_HELD_DOWN[0],
    SMALL_HOOK_HELD_DOWN[1],
    CARRY_Z,
)

# Large-pulley placement geometry
# Push the held +X vertex this far past the large pulley's outer rim (+X) so the
# belt drapes over the edge and drops into the groove instead of only touching
# the top. 
LARGE_STRETCH_EXTRA_X = 0.035  
LARGE_APPROACH_Z_OFFSET = 0.015

# Final descend target: over the outer edge of the large pulley.
BELT_PLACE_LARGE_HOOK = (
    BELT_PLACE_DOWN[0] + LARGE_STRETCH_EXTRA_X,
    BELT_PLACE_DOWN[1],
    BELT_PLACE_DOWN[2] + LARGE_APPROACH_Z_OFFSET,
)

# Height of the mid-air "stop" while the held side is carried over to the large
# pulley.
LARGE_MID_CLEARANCE = 0.06

# (a) Mid-air stop: above the large pulley's NEAR (-X) rim, not yet reached out.
LARGE_PRE_EDGE = (
    BELT_PLACE_DOWN[0],
    BELT_PLACE_LARGE_HOOK[1],
    BELT_PLACE_LARGE_HOOK[2] + LARGE_MID_CLEARANCE,
)

# (b) Reach out over the OUTER (+X) edge, still in mid-air at the same height.
LARGE_OVER_EDGE = (
    BELT_PLACE_LARGE_HOOK[0],
    BELT_PLACE_LARGE_HOOK[1],
    BELT_PLACE_LARGE_HOOK[2] + LARGE_MID_CLEARANCE,
)

# Final high retreat location above the large pulley.
CARRY_OVER_LARGE = (
    BELT_PLACE_LARGE_HOOK[0],
    BELT_PLACE_LARGE_HOOK[1],
    CARRY_Z,
)


def _lerp3(a, b, alpha: float) -> tuple[float, float, float]:
    """Smooth XYZ interpolation helper."""

    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    p = (1.0 - alpha) * a_np + alpha * b_np

    return float(p[0]), float(p[1]), float(p[2])


class PickAndPlaceExample(Example):
    """Hook the free belt side on the small pulley, then seat the held side."""

    def solve_gripper_targets(self):
        t = float(self.sim_time)

        open_q = np.asarray(self.gripper_open_values, dtype=np.float32)
        closed_q = np.asarray(self.gripper_closed_values, dtype=np.float32)

        # Phase boundaries
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

        # 1) Begin open above the original +X grasp vertex
        if t < t0:
            tcp_pos = BELT_APPROACH_POS
            grip_q = open_q

        # 2) Descend vertically onto the original +X grasp vertex
        elif t < t1:
            alpha = float(smoothstep((t - t0) / DESCEND_TIME))
            tcp_pos = _lerp3(BELT_APPROACH_POS, BELT_GRASP_POS, alpha)
            grip_q = open_q

        # 3) Close without changing the grasp location
        elif t < t2:
            alpha = float(smoothstep((t - t1) / CLOSE_TIME))
            tcp_pos = BELT_GRASP_POS
            grip_q = open_q + alpha * (closed_q - open_q)

        # 4) Let the physical pinch settle
        elif t < t3:
            tcp_pos = BELT_GRASP_POS
            grip_q = closed_q

        # 5) Lift the held side straight upward by 0.30 m
        elif t < t4:
            alpha = float(smoothstep((t - t3) / LIFT_TIME))
            tcp_pos = _lerp3(BELT_GRASP_POS, CARRY_POS, alpha)
            grip_q = closed_q

        # 6) Move the HELD side above its small-hook target
        elif t < t5:
            alpha = float(
                smoothstep((t - t4) / TRAVERSE_TO_SMALL_HOOK_TIME)
            )
            tcp_pos = _lerp3(CARRY_POS, CARRY_OVER_SMALL_HOOK, alpha)
            grip_q = closed_q

        # 7) Lower the loop while moving directly to the existing mid-air stop.
        elif t < t6:
            alpha = float(smoothstep((t - t5) / LOWER_SMALL_HOOK_TIME))
            tcp_pos = _lerp3(
                CARRY_OVER_SMALL_HOOK,
                LARGE_PRE_EDGE,
                alpha,
            )
            grip_q = closed_q

        # 8) Hold at the mid-air stop while the opposite side finishes seating
        #    around the small-pulley groove.
        elif t < t7:
            tcp_pos = LARGE_PRE_EDGE
            grip_q = closed_q

        # 9a) Stretch directly outward from the mid-air stop so the TCP hovers
        #     over the outer edge of the large pulley.
        elif t < t8b:
            alpha = float(smoothstep((t - t7) / REACH_EDGE_TIME))
            tcp_pos = _lerp3(LARGE_PRE_EDGE, LARGE_OVER_EDGE, alpha)
            grip_q = closed_q

        # 9b) Descend down so the belt drapes over the edge into the groove.
        elif t < t8:
            alpha = float(smoothstep((t - t8b) / LOWER_TO_LARGE_TIME))
            tcp_pos = _lerp3(LARGE_OVER_EDGE, BELT_PLACE_LARGE_HOOK, alpha)
            grip_q = closed_q

        # 10) Let both sides settle while the gripper remains closed.
        elif t < t9:
            tcp_pos = BELT_PLACE_LARGE_HOOK
            grip_q = closed_q

        # 11) Release the held side onto the large pulley
        elif t < t10:
            alpha = float(smoothstep((t - t9) / OPEN_TIME))
            tcp_pos = BELT_PLACE_LARGE_HOOK
            grip_q = closed_q + alpha * (open_q - closed_q)

        # 12) Retreat vertically above the large pulley
        elif t < t11:
            alpha = float(smoothstep((t - t10) / RETREAT_TIME))
            tcp_pos = _lerp3(BELT_PLACE_LARGE_HOOK, CARRY_OVER_LARGE, alpha)
            grip_q = open_q

        else:
            tcp_pos = CARRY_OVER_LARGE
            grip_q = open_q

        self.set_task_target(
            tcp_pos,
            GRIPPER_DOWN_QUAT,
            grip_q,
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    viewer._pause = False
    newton.examples.run(PickAndPlaceExample(viewer, args), args)