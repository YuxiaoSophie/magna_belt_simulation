"""Waypoint -> per-control-step Cartesian targets -> joint targets for both arms.

:func:`build_cartesian_trajectory` stops at every waypoint (magna stops at each of
``pre_pick_0..place_3``): both arms arrive together after ``max(1.875 |dp| / v, 1.875 theta /
w, min_seg_s)`` on a minimum-jerk profile (1.875 = its peak / mean speed), position lerp,
orientation slerp; each waypoint then holds ``max(dwell_s, dwell_pad_s if a gripper command
changed)``. :func:`solve_joint_trajectory` warm-starts :func:`arm_kinematics.ik` step by step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from round_belt_task.arm_kinematics import (
    FrankaTip,
    UrdfChain,
    UrTracking,
    ik,
    mat4,
    rot_angle,
    rot_axis_angle,
    rotvec,
)
from round_belt_task.waypoints import Waypoint

MIN_JERK_PEAK = 1.875
MAX_JOINT_STEP = 0.05
NO_FRANKA_CMD = np.nan
NO_UR_CMD = -1


class MotionError(RuntimeError):
    """IK or joint continuity failure at a named step/arm."""


@dataclass
class Segment:
    """A move from ``start`` (franka 4x4, ur 4x4) to ``end`` over ``duration_s``."""

    start: tuple[np.ndarray, np.ndarray]
    end: Waypoint
    duration_s: float
    hold_s: float = 0.0
    first_step: int = 0


@dataclass
class CartesianTrajectory:
    t: np.ndarray
    franka_4x4: np.ndarray
    ur_4x4: np.ndarray
    phase: np.ndarray
    labels: list[str]
    franka_gripper_mm: np.ndarray  # NaN = no command yet
    ur_gripper_byte: np.ndarray  # -1 = no command yet
    segments: list[Segment] = field(default_factory=list)
    dt: float = 0.005

    def __len__(self) -> int:
        return len(self.t)

    def phase_label(self, i: int) -> str:
        return self.labels[int(self.phase[i])]

    def move_mask(self) -> np.ndarray:
        moves = np.array([label.startswith("move") for label in self.labels])
        return moves[self.phase]


@dataclass
class JointTrajectory:
    cart: CartesianTrajectory
    q_franka: np.ndarray
    q_ur: np.ndarray
    stats: dict

    def __len__(self) -> int:
        return len(self.cart)

    def __getattr__(self, name: str):
        # Everything else (t, phase, labels, gripper commands, poses) lives on the cart.
        if name == "cart":
            raise AttributeError(name)
        return getattr(self.cart, name)


def min_jerk(s: np.ndarray) -> np.ndarray:
    return s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)


def _interp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """Lerp of the positions, slerp of the rotations of two 4x4 poses."""
    rel = rotvec(a[:3, :3].T @ b[:3, :3])
    angle = float(np.linalg.norm(rel))
    rot = a[:3, :3] if angle < 1e-12 else a[:3, :3] @ rot_axis_angle(rel, s * angle)
    return mat4(rot, (1.0 - s) * a[:3, 3] + s * b[:3, 3])


def _duration(a: np.ndarray, b: np.ndarray, lin_speed: float, ang_speed: float) -> float:
    return max(MIN_JERK_PEAK * float(np.linalg.norm(b[:3, 3] - a[:3, 3])) / lin_speed,
               MIN_JERK_PEAK * rot_angle(a, b) / ang_speed)


def build_cartesian_trajectory(
    waypoints: Sequence[Waypoint], start_franka_4x4: np.ndarray, start_ur_4x4: np.ndarray, *,
    dt: float = 0.005, lin_speed: float = 0.08, ang_speed: float = 0.5,
    min_seg_s: float = 0.25, dwell_pad_s: float = 1.0, settle_s: float = 1.0,
    start_franka_mm: float | None = None, start_ur_byte: int | None = None,
) -> CartesianTrajectory:
    """Stop-at-every-waypoint trajectory from the start poses; see the module docstring."""
    labels: list[str] = []
    rows: dict[str, list] = {k: [] for k in ("franka", "ur", "phase", "mm", "byte")}
    franka_cmd = NO_FRANKA_CMD if start_franka_mm is None else float(start_franka_mm)
    ur_cmd = NO_UR_CMD if start_ur_byte is None else int(start_ur_byte)
    franka, ur = np.asarray(start_franka_4x4, float), np.asarray(start_ur_4x4, float)
    segments = []

    def emit(label: str, f: np.ndarray, u: np.ndarray) -> None:
        if not labels or labels[-1] != label:
            labels.append(label)
        rows["franka"].append(f)
        rows["ur"].append(u)
        rows["phase"].append(len(labels) - 1)
        rows["mm"].append(franka_cmd)
        rows["byte"].append(ur_cmd)

    for wp_ in waypoints:
        f_end = wp_.franka_mat()
        u_end = ur if wp_.ur_pos is None else wp_.ur_mat()
        duration = max(_duration(franka, f_end, lin_speed, ang_speed),
                       _duration(ur, u_end, lin_speed, ang_speed), min_seg_s)
        n = max(1, math.ceil(duration / dt - 1e-9))
        seg = Segment(start=(franka, ur), end=wp_, duration_s=n * dt,
                      first_step=len(rows["phase"]))
        s_values = min_jerk(np.arange(1, n + 1) / n)
        for s in s_values[:-1]:
            emit(f"move:{wp_.label}", _interp(franka, f_end, float(s)),
                 _interp(ur, u_end, float(s)))
        # The waypoint's commands take effect on the step it is reached.
        changed = False
        if wp_.franka_gripper_mm is not None:
            changed |= franka_cmd != wp_.franka_gripper_mm
            franka_cmd = float(wp_.franka_gripper_mm)
        if wp_.ur_gripper_byte is not None:
            changed |= ur_cmd != wp_.ur_gripper_byte
            ur_cmd = int(wp_.ur_gripper_byte)
        emit(f"move:{wp_.label}", f_end, u_end)
        franka, ur = f_end, u_end
        hold = max(wp_.dwell_s, dwell_pad_s if changed else 0.0)
        seg.hold_s = round(hold / dt) * dt
        segments.append(seg)
        for _ in range(round(hold / dt)):
            emit(f"dwell:{wp_.label}", franka, ur)
    for _ in range(round(settle_s / dt)):
        emit("settle", franka, ur)

    count = len(rows["phase"])
    return CartesianTrajectory(
        t=(np.arange(count) + 1) * dt, franka_4x4=np.asarray(rows["franka"]),
        ur_4x4=np.asarray(rows["ur"]), phase=np.asarray(rows["phase"], dtype=np.int64),
        labels=labels, franka_gripper_mm=np.asarray(rows["mm"], dtype=np.float64),
        ur_gripper_byte=np.asarray(rows["byte"], dtype=np.int64), segments=segments, dt=dt,
    )


def _solve_arm(chain: UrdfChain, poses: np.ndarray, q0, arm: str,
               cart: CartesianTrajectory) -> tuple[np.ndarray, dict]:
    q = np.asarray(q0, dtype=np.float64)
    out = np.empty((len(poses), chain.n))
    iters = np.empty(len(poses), dtype=np.int64)
    err_p = np.empty(len(poses))
    err_r = np.empty(len(poses))
    pos_tol, rot_tol = 1e-4, 1e-3
    for i, target in enumerate(poses):
        q_new, err_p[i], err_r[i], iters[i] = ik(chain, target, q, pos_tol=pos_tol,
                                                 rot_tol=rot_tol)
        where = f"step {i} ({cart.phase_label(i)}), {arm}"
        if err_p[i] > pos_tol or err_r[i] > rot_tol:
            raise MotionError(f"{where}: IK missed ({err_p[i] * 1e3:.3f} mm, "
                              f"{math.degrees(err_r[i]):.3f} deg after {iters[i]} iterations)")
        jump = float(np.abs(q_new - q).max())
        if jump > MAX_JOINT_STEP:
            raise MotionError(f"{where}: joint jump {jump:.4f} rad > {MAX_JOINT_STEP} rad")
        out[i] = q = q_new
    jumps = np.abs(np.diff(np.vstack([np.asarray(q0, float)[None], out]), axis=0)).max(axis=1)
    return out, {"iters_mean": float(iters.mean()), "iters_max": int(iters.max()),
                 "err_pos_max_m": float(err_p.max()), "err_rot_max_rad": float(err_r.max()),
                 "jump_max_rad": float(jumps.max())}


def solve_joint_trajectory(cart: CartesianTrajectory, q0_franka, q0_ur) -> JointTrajectory:
    """Per-step IK for both arms; raises :class:`MotionError` naming the step and arm."""
    q_franka, franka_stats = _solve_arm(FrankaTip, cart.franka_4x4, q0_franka, "franka", cart)
    q_ur, ur_stats = _solve_arm(UrTracking, cart.ur_4x4, q0_ur, "ur", cart)
    return JointTrajectory(cart=cart, q_franka=q_franka, q_ur=q_ur,
                           stats={"franka": franka_stats, "ur": ur_stats})
