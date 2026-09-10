"""Seeding the finalized model with the Drake default joint state.

``scene.py`` constructs geometry on a ``ModelBuilder``; this is the next phase --
:func:`apply_default_joint_state` writes joint values, position gains and effort limits
onto an already-finalized :class:`newton.Model`, and records the resolved indices in
``info.joint_config`` for the simulation to reuse.  Different object, different phase,
hence a separate module.  ``JointConfig`` itself lives in ``scene.py`` because it is the
type of a ``SceneInfo`` field.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger

import newton
from newton import JointTargetMode

import round_belt
from round_belt_task.constants import (
    ARM_TARGET_KD, ARM_TARGET_KE, FINGER_TARGET_KD, FINGER_TARGET_KE,
    GRIPPER_OPEN_MARGIN, PANDA_DEFAULT_Q, PANDA_FINGER_DEFAULT_Q, PANDA_FINGER_LABELS,
    PANDA_JOINT_LABELS, UR10_DEFAULT_Q, UR10_JOINT_LABELS,
)
from round_belt_task.scene import SceneInfo
from utils.labels import joint_index


def _index_layout(length: int, n_coords: int, n_dofs: int, what: str) -> str:
    """Whether a per-joint array of ``length`` entries is coord- or dof-indexed.

    Newton's gain/target arrays are not consistently in one space and the two differ here
    (223 coords vs 221 dofs), so the layout is detected, not assumed.
    """
    if length == n_coords:
        return "coord"
    if length == n_dofs:
        return "dof"
    raise RuntimeError(
        f"{what} length {length} matches neither joint_coord_count ({n_coords}) "
        f"nor joint_dof_count ({n_dofs})."
    )


def apply_default_joint_state(model: newton.Model, info: SceneInfo) -> None:
    """Seed ``model.joint_q`` with the Drake defaults and set position gains.

    Mirrors ``round_belt._configure_robot_joints`` but resolves every index by joint label
    rather than by ordinal position.  The resolved indices land in ``info.joint_config``
    for ``RoundBeltTaskSimulation._seed_control_targets`` / ``.test_final``.
    """
    as_numpy = round_belt.as_numpy
    joint_labels = list(model.joint_label)
    q_start = as_numpy(model.joint_q_start)
    qd_start = as_numpy(model.joint_qd_start)

    def coord_dof(label: str) -> tuple[int, int]:
        j = joint_index(joint_labels, label)
        return int(q_start[j]), int(qd_start[j])

    def coord_dof_lists(labels: Sequence[str]) -> tuple[list[int], list[int]]:
        pairs = [coord_dof(label) for label in labels]
        return [c for c, _ in pairs], [d for _, d in pairs]

    arm_labels = PANDA_JOINT_LABELS + UR10_JOINT_LABELS
    arm_defaults = list(PANDA_DEFAULT_Q) + list(UR10_DEFAULT_Q)
    arm_coord, arm_dof = coord_dof_lists(arm_labels)
    finger_coord, finger_dof = coord_dof_lists(PANDA_FINGER_LABELS)

    gripper_joint_start = info.gripper_joints[0] if info.gripper_joints else 0
    gripper_joint_end = (info.gripper_joints[-1] + 1) if info.gripper_joints else 0
    gripper_coord, gripper_dof = round_belt.find_gripper_driver_indices(
        model, gripper_joint_start, end_joint_index=gripper_joint_end
    )
    if not gripper_coord:
        raise RuntimeError("no 2f85 driver joints found in the gripper joint range")

    lower_np = as_numpy(model.joint_limit_lower)
    upper_np = as_numpy(model.joint_limit_upper)
    gripper_open_values = []
    for d in gripper_dof:
        lo, hi = float(lower_np[d]), float(upper_np[d])
        gripper_open_values.append(lo + GRIPPER_OPEN_MARGIN if np.isfinite(lo) and hi > lo else 0.0)

    joint_q = model.joint_q.numpy().copy()
    for c, v in zip(arm_coord, arm_defaults):
        joint_q[c] = v
    for c, v in zip(finger_coord, PANDA_FINGER_DEFAULT_Q):
        joint_q[c] = v
    for c, v in zip(gripper_coord, gripper_open_values):
        joint_q[c] = v
    model.joint_q.assign(joint_q)
    model.joint_qd.zero_()

    n_coords = int(model.joint_coord_count)
    n_dofs = int(model.joint_dof_count)
    mode_np = as_numpy(model.joint_target_mode).copy()
    target_len = len(mode_np)

    robot_joint_end = info.robot_joints[-1] + 1
    layout = _index_layout(target_len, n_coords, n_dofs, "model.joint_target_mode")
    if layout == "coord":
        arm_t, finger_t, grip_t = arm_coord, finger_coord, gripper_coord
        robot_end = int(q_start[robot_joint_end]) if robot_joint_end < len(q_start) else n_coords
    else:
        arm_t, finger_t, grip_t = arm_dof, finger_dof, gripper_dof
        robot_end = int(qd_start[robot_joint_end]) if robot_joint_end < len(qd_start) else n_dofs

    ke_np = as_numpy(model.joint_target_ke).copy()
    kd_np = as_numpy(model.joint_target_kd).copy()
    ke_np[:robot_end] = 0.0
    kd_np[:robot_end] = 0.0
    mode_np[:robot_end] = int(JointTargetMode.NONE)

    for indices, ke, kd in (
        (arm_t, ARM_TARGET_KE, ARM_TARGET_KD),
        (finger_t, FINGER_TARGET_KE, FINGER_TARGET_KD),
        (grip_t, round_belt.GRIPPER_DRIVE_KE, round_belt.GRIPPER_DRIVE_KD),
    ):
        ke_np[indices] = ke
        kd_np[indices] = kd
        mode_np[indices] = int(JointTargetMode.POSITION)

    model.joint_target_ke.assign(ke_np)
    model.joint_target_kd.assign(kd_np)
    model.joint_target_mode.assign(mode_np)

    effort_np = as_numpy(model.joint_effort_limit).copy()
    for d in gripper_dof:
        effort_np[d] = round_belt.GRIPPER_EFFORT_LIMIT
    model.joint_effort_limit.assign(effort_np)

    info.joint_config.update(
        gains_layout=layout,
        arm_labels=arm_labels,
        arm_defaults=arm_defaults,
        arm_coord_indices=arm_coord,
        finger_coord_indices=finger_coord,
        gripper_open_values=gripper_open_values,
        arm_target_indices=list(arm_t),
        finger_target_indices=list(finger_t),
        gripper_target_indices=list(grip_t),
    )
    logger.info(
        f"Gain-array layout: {layout}-space (len={target_len}); "
        f"13 arm joints at ke={ARM_TARGET_KE}/kd={ARM_TARGET_KD}; "
        f"gripper open targets {['%.4f' % v for v in gripper_open_values]}"
    )
