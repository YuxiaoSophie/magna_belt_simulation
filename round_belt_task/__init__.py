"""Newton port of the Drake round-belt scene, split by concern.

    constants.py    every Drake-transcribed number: asset paths, welds, joint defaults
    scene.py        assembly of the geometry into a ModelBuilder
    joint_state.py  seeding the finalized Model with the Drake default joint state
    simulation.py   RoundBeltTaskSimulation: solver, stepping, CUDA graph, diagnostics

``round_belt_task_simulation.py`` at the repo root is the entry point.  Assets and
provenance are documented in ``assets/README.md`` and ``README.md``.  This module is the
public surface: everything below is what importers -- notably
``scripts/check_round_belt_task_poses.py`` -- read.
"""

from round_belt_task.constants import (
    ALOHA_FINGER_COLOR, ALOHA_FINGER_OFFSET_X, ALOHA_FINGER_OFFSET_Z, ARM_TARGET_KD,
    ARM_TARGET_KE, ASSETS_DIR, BELT_CENTER, BELT_COLOR, BELT_NUM_ELEMENTS, BELT_RADIUS,
    BELT_SEMI_AXIS_X, BELT_SEMI_AXIS_Y, BOARD_COLOR, BOARD_PANEL_MIN_SPAN, BOARD_URDF,
    COMMON_ASSETS_DIR, FINGER_TARGET_KD, FINGER_TARGET_KE, GRIPPER_OPEN_MARGIN,
    HOLDER_URDF, LARGE_PULLEY_COLOR, MJCF_BASE_MOUNT_OFFSET_Z, PANDA_ARM_URDF,
    PANDA_DEFAULT_Q, PANDA_FINGER_DEFAULT_Q, PANDA_FINGER_LABELS, PANDA_HAND_URDF,
    PANDA_JOINT_LABELS, PULLEY_MOUNT_COLOR, REPO_ROOT, ROBOTIQ_FINGER_DIR, ROBOTIQ_MJCF,
    ROUND_BELT_ASSETS_DIR, SMALL_PULLEY_COLOR, SMALL_PULLEY_MOUNT_LOCAL_XY,
    SMALL_PULLEY_MOUNT_RADIUS, TABLE_TOP_Z, TABLE_URDF, TABLE_VISUAL_LABEL,
    TABLETOP_COLLISION_THICKNESS, UR10_BASE_LABEL, UR10_DEFAULT_Q, UR10_JOINT_LABELS,
    UR10_URDF, UR10_USD_ASSET, UR10_USD_RELPATH, UR10_WRIST3_D6, UR10_WRIST3_LABEL,
    X_LINK8_HAND, X_USDWRIST3_GRIPPER, X_USDWRIST3_URDFWRIST3, X_W_BOARD, X_W_HOLDER,
    X_W_PANDA, X_W_TABLE, X_W_UR10, X_WRIST3_GRIPPER,
)

from round_belt_task.joint_state import apply_default_joint_state
from round_belt_task.scene import JointConfig, SceneInfo, build_scene, make_builder

# Re-exported on purpose: this is how importers resolve a body/joint by label.
from utils.labels import body_index, joint_index
