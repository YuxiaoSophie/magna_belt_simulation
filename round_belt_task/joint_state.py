"""The default joint state, bound to the round-belt arm/finger labels and Drake defaults."""

from __future__ import annotations

import newton

from round_belt_task.constants import (
    PANDA_DEFAULT_Q,
    PANDA_FINGER_DEFAULT_Q,
    PANDA_FINGER_LABELS,
    PANDA_JOINT_LABELS,
    UR10_DEFAULT_Q,
    UR10_JOINT_LABELS,
)
from task_common import joint_state
from task_common.scene import SceneInfo


def apply_default_joint_state(model: newton.Model, info: SceneInfo) -> None:
    """Seed ``model.joint_q`` with the Drake defaults and set position gains."""
    joint_state.apply_default_joint_state(
        model, info,
        arm_labels=PANDA_JOINT_LABELS + UR10_JOINT_LABELS,
        arm_defaults=list(PANDA_DEFAULT_Q) + list(UR10_DEFAULT_Q),
        finger_labels=PANDA_FINGER_LABELS, finger_defaults=PANDA_FINGER_DEFAULT_Q,
    )
