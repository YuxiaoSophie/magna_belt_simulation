"""The default joint state, bound to the round-belt arm/finger labels and Drake defaults."""

from __future__ import annotations

from pathlib import Path

import newton
from loguru import logger

from round_belt_task.constants import (
    PANDA_DEFAULT_Q,
    PANDA_FINGER_DEFAULT_Q,
    PANDA_FINGER_LABELS,
    PANDA_JOINT_LABELS,
    UR10_DEFAULT_Q,
    UR10_JOINT_LABELS,
)
from task_common import joint_state
from task_common.initial_state import load_initial_state
from task_common.scene import SceneInfo

# magna initial-state key -> the joint labels it seeds, in order.
INITIAL_STATE_KEYS = {
    "q_init_franka": PANDA_JOINT_LABELS,
    "q_init_franka_hand": PANDA_FINGER_LABELS,
    "q_init_ur": UR10_JOINT_LABELS,
}


def apply_default_joint_state(
    model: newton.Model, info: SceneInfo, initial_state: str | Path | None = None
) -> None:
    """Seed ``model.joint_q`` with the Drake defaults and set position gains.

    ``initial_state`` is a magna ``*_initial_state.yaml`` whose ``q_init_*`` lists replace the
    scene's ``default_joint_positions`` for this run (see ``src/task_common/initial_state.py``).
    """
    panda_q = list(PANDA_DEFAULT_Q)
    finger_q = list(PANDA_FINGER_DEFAULT_Q)
    ur_q = list(UR10_DEFAULT_Q)
    if initial_state is not None:
        lengths = {key: len(labels) for key, labels in INITIAL_STATE_KEYS.items()}
        state = load_initial_state(initial_state, lengths)
        panda_q = state["q_init_franka"]
        finger_q = state["q_init_franka_hand"]
        ur_q = state["q_init_ur"]
        logger.info(
            f"[INIT] initial state {Path(initial_state)}: "
            f"q_init_franka={_fmt(panda_q)}, q_init_franka_hand={_fmt(finger_q)}, "
            f"q_init_ur={_fmt(ur_q)}"
        )
        for label, default, value in zip(
            PANDA_JOINT_LABELS + PANDA_FINGER_LABELS + UR10_JOINT_LABELS,
            list(PANDA_DEFAULT_Q) + list(PANDA_FINGER_DEFAULT_Q) + list(UR10_DEFAULT_Q),
            panda_q + finger_q + ur_q,
        ):
            if abs(value - default) > 1e-12:
                logger.info(f"[INIT]   {label}: {default:+.6f} -> {value:+.6f} rad")

    joint_state.apply_default_joint_state(
        model, info,
        arm_labels=PANDA_JOINT_LABELS + UR10_JOINT_LABELS,
        arm_defaults=panda_q + ur_q,
        finger_labels=PANDA_FINGER_LABELS, finger_defaults=finger_q,
    )


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"
