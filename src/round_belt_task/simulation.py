"""The round-belt simulation: :class:`task_common.simulation.BeltTaskSimulation` on this scene."""

from __future__ import annotations

import newton

from round_belt_task.constants import BELT_CENTER, TABLE_TOP_Z
from round_belt_task.joint_state import apply_default_joint_state
from round_belt_task.scene import SceneInfo, build_scene
from task_common.simulation import BeltTaskSimulation


class RoundBeltTaskSimulation(BeltTaskSimulation):
    """Drake round-belt scene held at its default configuration."""

    belt_center = BELT_CENTER
    table_top_z = TABLE_TOP_Z

    def _build_scene(self, builder: newton.ModelBuilder) -> SceneInfo:
        return build_scene(builder)

    def _apply_default_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        apply_default_joint_state(model, info)
