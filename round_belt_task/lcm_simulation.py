"""The round-belt LCM simulation: :class:`LcmBeltTaskSimulation` on this scene, plus the belt
trigger of Drake's ``magna_simulation.cc``."""

from __future__ import annotations

import argparse

import newton
import numpy as np
import warp as wp
from loguru import logger

from round_belt_task.constants import (
    BELT_CENTER,
    BELT_RADIUS,
    BELT_TRIGGER_BODY,
    BELT_TRIGGER_GRASP_DEPTH,
    BELT_TRIGGER_NEAREST_RADIUS,
    BELT_TRIGGER_POINT,
    BELT_TRIGGER_TOLERANCE,
    LCM_GRIPPER_DRIVE_DAMPING,
    LCM_GRIPPER_DRIVE_EFFORT_LIMIT,
    LCM_GRIPPER_DRIVE_KD,
    LCM_GRIPPER_DRIVE_KE,
    LCM_GRIPPER_DRIVE_STOP,
    LCM_HAND_DRIVE_ARMATURE,
    LCM_HAND_DRIVE_EFFORT_LIMIT,
    LCM_HAND_DRIVE_KD,
    LCM_HAND_DRIVE_KE,
    LCM_HAND_DRIVE_LIMIT_KD,
    LCM_HAND_DRIVE_LIMIT_KE,
    LCM_HAND_DRIVE_MAX_SPEED,
    LCM_HAND_DRIVE_STALE_TIMEOUT,
    LCM_SOLVER_SUBSTEPS,
    LCM_SOLVER_VBD_ITERATIONS,
    LCM_UR_GRIPPER_TIP_Z,
    PANDA_ARM_URDF,
    PANDA_FINGER_LABELS,
    PANDA_JOINT_LABELS,
    TABLE_TOP_Z,
    UR10_JOINT_LABELS,
    UR10_WRIST3_LABEL,
    X_USDWRIST3_URDFWRIST3,
)
from round_belt_task.joint_state import apply_default_joint_state
from round_belt_task.scene import SceneInfo, build_scene
from task_common.lcm_contract import HandDrive, franka_hand_spec, franka_spec, ur10_spec
from task_common.lcm_simulation import LcmBeltTaskSimulation, reflected_rotor_inertia
from utils.labels import body_index


def _fmt(v: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(x):.5f}" for x in v) + ")"


class RoundBeltLcmSimulation(LcmBeltTaskSimulation):
    """Drake round-belt scene over LCM; places the belt once the Franka reaches the trigger."""

    belt_center = BELT_CENTER
    table_top_z = TABLE_TOP_Z
    belt_radius = BELT_RADIUS
    solver_substeps = LCM_SOLVER_SUBSTEPS
    solver_vbd_iterations = LCM_SOLVER_VBD_ITERATIONS
    gripper_drive_ke = LCM_GRIPPER_DRIVE_KE
    gripper_drive_kd = LCM_GRIPPER_DRIVE_KD
    gripper_drive_stop = LCM_GRIPPER_DRIVE_STOP
    gripper_drive_effort_limit = LCM_GRIPPER_DRIVE_EFFORT_LIMIT
    gripper_drive_damping = LCM_GRIPPER_DRIVE_DAMPING
    hand_drive = HandDrive(
        ke=LCM_HAND_DRIVE_KE, kd=LCM_HAND_DRIVE_KD, effort_limit=LCM_HAND_DRIVE_EFFORT_LIMIT,
        stale_timeout=LCM_HAND_DRIVE_STALE_TIMEOUT, armature=LCM_HAND_DRIVE_ARMATURE,
        max_speed=LCM_HAND_DRIVE_MAX_SPEED, limit_ke=LCM_HAND_DRIVE_LIMIT_KE,
        limit_kd=LCM_HAND_DRIVE_LIMIT_KD,
    )

    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace) -> None:
        self.belt_trigger_point = np.asarray(BELT_TRIGGER_POINT, dtype=np.float64)
        self.belt_trigger_tolerance = BELT_TRIGGER_TOLERANCE
        self.grasp_depth = BELT_TRIGGER_GRASP_DEPTH
        self.belt_anchor_body = -1
        super().__init__(viewer, args)
        logger.info(
            f"[BELT] anchor target = finger_tip - {BELT_TRIGGER_GRASP_DEPTH:g} m along the hand z"
        )
        body_labels = list(self.model.body_label)
        self._finger_tip_body = body_index(body_labels, BELT_TRIGGER_BODY)
        self._ur_wrist_body = body_index(body_labels, UR10_WRIST3_LABEL)
        self._ur_tip_in_wrist = wp.transform_point(
            X_USDWRIST3_URDFWRIST3, wp.vec3(0.0, 0.0, LCM_UR_GRIPPER_TIP_Z)
        )
        self._belt_bodies = np.asarray(self.info.belt_bodies, dtype=np.int64)

    def _build_scene(self, builder: newton.ModelBuilder) -> SceneInfo:
        info = build_scene(builder)
        self._couple_hand_fingers(builder)
        return info

    def _apply_task_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        apply_default_joint_state(model, info)

    def _robot_specs(self):
        return [
            franka_spec(PANDA_JOINT_LABELS),
            franka_hand_spec(PANDA_FINGER_LABELS),
            ur10_spec(UR10_JOINT_LABELS),
        ]

    def _reflected_inertia(self) -> dict[str, float]:
        prefix = PANDA_JOINT_LABELS[0].rsplit("/", 1)[0]
        return {
            f"{prefix}/{name}": value
            for name, value in reflected_rotor_inertia(PANDA_ARM_URDF).items()
        }

    def _grasp_tip_body(self) -> int:
        return self._finger_tip_body

    def _grasp_ur_tip_point(self, body_q: np.ndarray) -> np.ndarray:
        row = [float(v) for v in body_q[self._ur_wrist_body]]
        X_W_wrist = wp.transform(wp.vec3(*row[:3]), wp.quat(*row[3:]))
        return np.asarray(wp.transform_point(X_W_wrist, self._ur_tip_in_wrist), dtype=np.float64)

    def _after_control_step(self, body_q: np.ndarray) -> None:
        if self.belt_placed:
            return
        finger_tip = body_q[self._finger_tip_body, :3].astype(np.float64)
        if np.linalg.norm(finger_tip - self.belt_trigger_point) >= self.belt_trigger_tolerance:
            return
        # Rigid translation, not a re-fit: the FEM shape is stretched and would spring back.
        current = body_q[self._belt_bodies, :3].astype(np.float64)
        distance = np.linalg.norm(current - finger_tip, axis=1)
        nearest = int(np.argmin(distance))
        if distance[nearest] < BELT_TRIGGER_NEAREST_RADIUS:
            anchor, rule = nearest, "nearest"
        else:
            anchor, rule = int(np.argmax(current[:, 1])), "+Y end"
        hand_z = wp.quat_rotate(wp.quat(*[float(v) for v in body_q[self._hand_body, 3:7]]),
                                wp.vec3(0.0, 0.0, 1.0))
        target = finger_tip - self.grasp_depth * np.asarray(hand_z, dtype=np.float64)
        translation = target - current[anchor]
        placed = current + translation
        self.reset_body_poses(self._belt_bodies, placed)
        self.belt_placed = True
        self.belt_anchor_body = int(self._belt_bodies[anchor])
        logger.success(
            f"[BELT] placed at step {self.step_index} (sim t={self.sim_time:.3f} s): "
            f"finger_tip={_fmt(finger_tip)}, target={_fmt(target)}, "
            f"anchor body {self.belt_anchor_body} ({rule}), "
            f"translation={_fmt(translation)}, anchor now={_fmt(placed[anchor])}"
        )
