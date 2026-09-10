"""Round-belt scene constants: a typed view over the scene directives YAML.

Every scene number -- welds, frames, joint defaults, colours, belt geometry -- is authored
once, next to its Drake provenance comment, in ``assets/round_belt_task/round_belt_scene.yaml``
(+ ``assets/common/directives/ur10_2f85.yaml``); that file also documents the world frame.
This module parses it with :func:`utils.directives.parse_directives` -- pure: no builder, no
asset download -- and re-exports the values under the names ``scene.py``, ``joint_state.py``,
``simulation.py`` and ``scripts/check_round_belt_task_poses.py`` read.  What stays Python is
not scene data: path roots, joint-label conventions, controller gains, the URDF kinematic
reference, and ``UR10_WRIST3_D6`` (documentation, asserted against the YAML below).
"""

from __future__ import annotations

from pathlib import Path

import warp as wp

from utils.directives import Directive, parse_directives

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
COMMON_ASSETS_DIR = ASSETS_DIR / "common"
ROUND_BELT_ASSETS_DIR = ASSETS_DIR / "round_belt_task"
SCENE_DIRECTIVES = ROUND_BELT_ASSETS_DIR / "round_belt_scene.yaml"

_D = parse_directives(SCENE_DIRECTIVES)


def _source(directive: Directive) -> Path:
    """The yaml that authored ``directive``; its relative paths resolve against it."""
    return next(source for entry, source in _D.entries if entry is directive)


def _model_file(name: str) -> Path:
    model = _D.model(name)
    return _D.resolve_path(model.file, _source(model))


# ---- label conventions (Python-only) ----------------------------------------------------
# The scene.urdf visual whose AABB defines the floor height.  Looked up by exact label so
# dict ordering can never pick another one.
TABLE_VISUAL_LABEL = "table/scene/visual0"
PANDA_JOINT_LABELS = [f"panda_arm/panda_joint{i}" for i in range(1, 8)]
PANDA_FINGER_LABELS = ["panda_hand/panda_finger_joint1", "panda_hand/panda_finger_joint2"]
# USD joint labels are full prim paths ("/ur10/<parent link>/<joint>"); the six names and
# their order are identical to the Drake URDF's, hence ur.dmd.yaml's defaults apply.
UR10_JOINT_LABELS = [
    "/ur10/base_link/shoulder_pan_joint",
    "/ur10/shoulder_link/shoulder_lift_joint",
    "/ur10/upper_arm_link/elbow_joint",
    "/ur10/forearm_link/wrist_1_joint",
    "/ur10/wrist_1_link/wrist_2_joint",
    "/ur10/wrist_2_link/wrist_3_joint",
]
UR10_BASE_LABEL = "/ur10/base_link"
UR10_WRIST3_LABEL = "/ur10/wrist_3_link"

# ---- asset paths ------------------------------------------------------------------------
TABLE_URDF = _model_file("table")
BOARD_URDF = _model_file("board")
HOLDER_URDF = _model_file("belt_chain_holder")
PANDA_ARM_URDF = _model_file("panda_arm")
PANDA_HAND_URDF = _model_file("panda_hand")
# MJCF; Newton has no SDF importer, so this substitutes for the Drake Robotiq SDF.
ROBOTIQ_MJCF = _model_file("robotiq_2f85")
# The scene builds the NVIDIA USD UR10 (see ur10_2f85.yaml), not this URDF.  The URDF stays
# the *kinematic* source of truth -- scripts/check_round_belt_task_poses.py walks it with an
# independent numpy FK and asserts the USD arm lands in the same place.  Measured USD-vs-URDF
# frame differences: see assets/README.md "UR10" and the ur10_wrist_3_link_drake frame.
UR10_URDF = COMMON_ASSETS_DIR / "ur10" / "ur10.urdf"

# ---- welds and frames -------------------------------------------------------------------
X_W_TABLE = _D.weld("table::scene").X_PC.to_transform()
X_W_BOARD = _D.weld("board::board").X_PC.to_transform()
X_W_HOLDER = _D.weld("belt_chain_holder::belt_chain_holder_first_half").X_PC.to_transform()
X_W_PANDA = _D.weld("panda_arm::panda_link0").X_PC.to_transform()
X_LINK8_HAND = _D.weld("panda_hand::panda_hand").X_PC.to_transform()
X_W_UR10 = _D.weld("ur10::base_link").X_PC.to_transform()
# USD wrist_3_link -> Drake/URDF wrist_3_link, i.e. T(0, d6, 0) . Rx(-90 deg).
X_USDWRIST3_URDFWRIST3 = _D.frame("ur10_wrist_3_link_drake").X_PF.to_transform()
# Drake/URDF wrist_3_link -> MJCF gripper root: Rz(+90 deg), as ur.dmd.yaml.
X_WRIST3_GRIPPER = _D.weld("robotiq_2f85").X_PC.to_transform()
# Gripper weld expressed on the USD wrist_3 body, so the MJCF root lands at exactly the
# same world pose it had on the Drake URDF's wrist_3_link:
#     X(usd_wrist3) . X_USDWRIST3_GRIPPER == X(urdf_wrist3) . Rz(+90 deg)
X_USDWRIST3_GRIPPER = X_USDWRIST3_URDFWRIST3 * X_WRIST3_GRIPPER

# Documentation constant, and the one scene number deliberately duplicated in Python: the
# ``wrist_3_joint`` origin offset d6 that the ur10_wrist_3_link_drake frame encodes.
UR10_WRIST3_D6 = 0.0922
if abs(float(wp.transform_get_translation(X_USDWRIST3_URDFWRIST3)[1]) - UR10_WRIST3_D6) >= 1e-6:
    raise AssertionError(
        f"ur10_wrist_3_link_drake translation {X_USDWRIST3_URDFWRIST3} does not carry "
        f"d6 = UR10_WRIST3_D6 = {UR10_WRIST3_D6} on +Y"
    )


# ---- joint defaults ---------------------------------------------------------------------
def _joint_defaults(model: str, labels: list[str]) -> list[float]:
    """``model``'s ``default_joint_positions``, ordered by the leaf names of ``labels``."""
    positions = _D.model(model).default_joint_positions
    leaves = [label.rsplit("/", 1)[-1] for label in labels]
    if sorted(positions) != sorted(leaves) or any(len(v) != 1 for v in positions.values()):
        raise ValueError(
            f"{SCENE_DIRECTIVES}: {model} default_joint_positions {positions} must give "
            f"exactly one value for each of {leaves}"
        )
    return [positions[leaf][0] for leaf in leaves]


PANDA_DEFAULT_Q = _joint_defaults("panda_arm", PANDA_JOINT_LABELS)
PANDA_FINGER_DEFAULT_Q = _joint_defaults("panda_hand", PANDA_FINGER_LABELS)
UR10_DEFAULT_Q = _joint_defaults("ur10", UR10_JOINT_LABELS)

# ---- custom-directive parameters --------------------------------------------------------
_TABLETOP = _D.custom("add_tabletop_collision").params
_GROUND = _D.custom("add_ground_plane").params
_TABLE_REFS = (_TABLETOP.get("table_visual"), _GROUND.get("height_from_aabb_min_z_of"))
if any(ref != TABLE_VISUAL_LABEL for ref in _TABLE_REFS):
    raise ValueError(
        f"{SCENE_DIRECTIVES}: tabletop and ground must both reference {TABLE_VISUAL_LABEL!r}, "
        f"got {_TABLE_REFS}"
    )
TABLE_TOP_Z = float(_TABLETOP["top_z"])
TABLETOP_COLLISION_THICKNESS = float(_TABLETOP["thickness"])

_BELT = _D.custom("add_rod_ellipse", "flexible_ellipse_cable").params
BELT_CENTER = tuple(float(v) for v in _BELT["center"])
BELT_SEMI_AXIS_X, BELT_SEMI_AXIS_Y = (float(v) for v in _BELT["semi_axes"])
BELT_RADIUS = float(_BELT["radius"])
BELT_NUM_ELEMENTS = int(_BELT["num_elements"])
BELT_COLOR = tuple(float(v) for v in _BELT["color"])

# ---- board colours (the board add_model's color / link_colors / component_colors) -------
_BOARD = _D.model("board")
_MOUNT = _BOARD.component_colors[0]
BOARD_COLOR = _BOARD.color
SMALL_PULLEY_COLOR = _BOARD.link_colors["small_round_pulley"]
LARGE_PULLEY_COLOR = _BOARD.link_colors["large_round_pulley"]
PULLEY_MOUNT_COLOR = _MOUNT.color
SMALL_PULLEY_MOUNT_LOCAL_XY = _MOUNT.near_local_xy
SMALL_PULLEY_MOUNT_RADIUS = _MOUNT.radius
BOARD_PANEL_MIN_SPAN = _MOUNT.max_span

# ---- controller gains (physics tuning, not scene data) ----------------------------------
ARM_TARGET_KE = 700.0
ARM_TARGET_KD = 110.0
FINGER_TARGET_KE = 100.0
FINGER_TARGET_KD = 10.0
GRIPPER_OPEN_MARGIN = 0.005
