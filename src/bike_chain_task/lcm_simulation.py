"""The bike-chain LCM simulation: :class:`RoundBeltLcmSimulation`'s robot I/O on the chain scene,
everything in one ``SolverMuJoCo`` (no belt trigger)."""

from __future__ import annotations

import dataclasses

import newton
import numpy as np

from bike_chain_task.constants import LCM_SIM_PARAMS, SCENE_DIRECTIVES
from bike_chain_task.joint_state import apply_default_joint_state
from bike_chain_task.simulation import _BikeChainTask
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from task_common.recording import file_digest
from task_common.scene import SceneInfo


class _BikeChainLcmTask(_BikeChainTask):
    """The LCM-side overrides shared by the LCM, offline and OSC bike-chain sims."""

    belt_radius = 0.005  # half the ~10 mm plate stack; LCM tube mesh only
    # Real grasp forces (see _BikeChainTask.finger_*/gripper_*): replaces the round belt's.
    hand_drive = dataclasses.replace(
        RoundBeltLcmSimulation.hand_drive, ke=_BikeChainTask.finger_drive_ke,
        kd=_BikeChainTask.finger_drive_kd, effort_limit=_BikeChainTask.finger_effort_limit)
    gripper_drive_ke = _BikeChainTask.gripper_drive_ke
    gripper_drive_effort_limit = _BikeChainTask.gripper_effort_limit
    # 200 Hz control: 4 substeps = 1.25 ms, at or below the spike's 1.67 ms (60 fps x 10).
    solver_substeps = 4

    def _build_scene(self, builder: newton.ModelBuilder) -> SceneInfo:
        info = super()._build_scene(builder)
        self._couple_hand_fingers(builder)
        return info

    def _apply_task_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        apply_default_joint_state(model, info)

    def _recording_meta(self) -> dict:
        return {
            "scene_directives": file_digest(SCENE_DIRECTIVES),
            "lcm_sim_params": file_digest(LCM_SIM_PARAMS),
            "finger_tip_body": int(self._finger_tip_body),
            "ur_wrist_body": int(self._ur_wrist_body),
            "ur_tip_in_wrist": [float(v) for v in self._ur_tip_in_wrist],
        }

    def _after_control_step(self, body_q: np.ndarray) -> None:
        pass  # no belt teleport trigger


class BikeChainLcmSimulation(_BikeChainLcmTask, RoundBeltLcmSimulation):
    """The bike-chain scene over LCM (magna controllers)."""
