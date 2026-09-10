"""Drake-transcribed constants for the round-belt scene: asset paths, welds, joint
defaults, colours.

Poses and joint defaults come from, and only from, ``round-belt-scene.dmd.yaml``
(welds), ``ur.dmd.yaml`` (UR10 + Robotiq), ``round_belt.sdf`` (belt) and
``round_belt_simulation_params.yaml`` (Franka start pose), all under ``magna/``.
World frame == the Drake world frame verbatim: origin at ``panda_link0``, Z up, table
top z = -0.02858, floor z = -0.81852 (the table visual's own AABB min z).  This is
deliberately NOT ``round_belt.py``'s ``TABLE_TOP_Z = 0.72``; do not "fix" it.
"""

from __future__ import annotations

import math
from pathlib import Path

import warp as wp

# Helpers/constants only; round_belt has a ``__main__`` guard, so this starts nothing.
import round_belt

# Importing ``utils.transforms`` runs its import-time RPY-convention proof once.
from utils.transforms import drake_xform

REPO_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = REPO_ROOT / "assets"
COMMON_ASSETS_DIR = ASSETS_DIR / "common"
ROUND_BELT_ASSETS_DIR = ASSETS_DIR / "round_belt_task"
TABLE_URDF = COMMON_ASSETS_DIR / "scene.urdf"
BOARD_URDF = ROUND_BELT_ASSETS_DIR / "round_belt_task_board.urdf"
HOLDER_URDF = COMMON_ASSETS_DIR / "belt_chain_holder" / "belt_chain_holder.urdf"
PANDA_ARM_URDF = COMMON_ASSETS_DIR / "franka" / "urdf" / "panda_arm.urdf"
PANDA_HAND_URDF = COMMON_ASSETS_DIR / "franka" / "urdf" / "panda_hand_with_long_fingers.urdf"
# The scene builds the NVIDIA USD UR10, not this URDF: the Drake glTFs carry no images,
# so that arm renders flat.  The URDF stays the *kinematic* source of truth --
# scripts/check_round_belt_task_poses.py walks it with an independent numpy FK and asserts
# the USD arm lands in the same place.  Measured USD-vs-URDF frame differences: see
# assets/README.md "UR10" and X_USDWRIST3_URDFWRIST3 below.
UR10_URDF = COMMON_ASSETS_DIR / "ur10" / "ur10.urdf"
UR10_USD_ASSET = "universal_robots_ur10"
UR10_USD_RELPATH = ("usd", "ur10_instanceable.usda")
# MJCF; Newton has no SDF importer, so this substitutes for the Drake Robotiq SDF.
ROBOTIQ_MJCF = REPO_ROOT / "2f85.xml"

# The scene.urdf visual whose AABB defines the floor height.  Looked up by exact
# label so dict ordering can never pick another one.
TABLE_VISUAL_LABEL = "table/scene/visual0"

# add_weld: world -> table::scene   (no X_PC => identity)
X_W_TABLE = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

# add_weld: world -> nist_board::board
X_W_BOARD = drake_xform(
    (0.64483928, -0.19718233, 0.01076393),
    (-3.32822058e-01, -6.87450103e-02, 8.95207485e01),
)

# add_weld: world -> belt_holder::belt_chain_holder_first_half
X_W_HOLDER = drake_xform((0.4736603358808432, 0.3520562100563749, -0.02858), (0.0, 0.0, 90.0))

# add_weld: world -> panda::panda_link0   (no X_PC => identity; this IS the world origin)
X_W_PANDA = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

# add_weld: panda::panda_link8 -> panda_hand::panda_hand
X_LINK8_HAND = drake_xform((0.0, 0.0, 0.0), (0.0, 0.0, -45.0))

# add_weld: world -> ur10::base_link  (base_link is the URDF root, so this IS the
# root xform)
X_W_UR10 = drake_xform(
    (1.33235648, -0.15491427, 0.03076191),
    (-0.35720724, 0.72781324, 179.78723941),
)

# ur.dmd.yaml welds robotiq_85::robotiq_85_base_link to ur10::wrist_3_link with yaw
# +90 deg.  2f85.xml is substituted for the Drake Robotiq SDF, so the weld is re-derived
# for the MJCF root frame:
#   * Drake SDF: the fingers translate along +/-X of robotiq_85_base_link and the
#     fingertips lie along +Z of wrist_3_link (ur.dmd's finger_tip_frame is
#     wrist_3_link + (0, 0, 0.194)), so the +90 deg yaw closes the pads on +/-Y.
#   * 2f85.xml: the root child body ``base`` carries quat="1 0 0 -1" = Rz(-90 deg)
#     w.r.t. the MJCF root and the pads are offset along ``base`` +/-Y, so in MJCF-root
#     coordinates they sit along +/-X and the gripper extends along the root's +Z.
#   => MJCF root = wrist_3_link . Rz(+90 deg), zero translation, putting the pads back
#      on +/-Y of wrist_3_link, matching Drake.
# NOT round_belt.py's Ry(90 deg): that compensates the Newton USD UR10's tool frame,
# which this scene does not use.
X_WRIST3_GRIPPER = wp.transform(
    wp.vec3(0.0, 0.0, 0.0), round_belt.quat_from_rpy(0.0, 0.0, 0.5 * math.pi)
)

# The NVIDIA USD UR10's ``wrist_3_link`` body frame is not the ROS/Drake URDF's frame
# for the same link.  Measured (USD at identity, both at UR10_DEFAULT_Q):
#     inv(X_usd_wrist3) . X_urdf_wrist3 = T(0, d6, 0) . Rx(-90 deg)
# with d6 = 0.0922 m, exactly the ``wrist_3_joint`` origin translation in
# assets/ur10/ur10.urdf:259: the USD keeps its wrist_3 frame at the joint axis while the
# URDF pushes it out by the link length, and the two differ by a quarter turn about X.
# Every other UR10 link differs by a similar constant; base_link is the one link where
# they agree exactly, which is why X_W_UR10 needs no correction.
UR10_WRIST3_D6 = 0.0922
X_USDWRIST3_URDFWRIST3 = wp.transform(
    wp.vec3(0.0, UR10_WRIST3_D6, 0.0), round_belt.quat_from_rpy(-0.5 * math.pi, 0.0, 0.0)
)

# Gripper weld expressed on the USD wrist_3 body, so the MJCF root lands at
# exactly the same world pose it had on the Drake URDF's wrist_3_link:
#     X(usd_wrist3) . X_USDWRIST3_GRIPPER == X(urdf_wrist3) . Rz(+90 deg)
X_USDWRIST3_GRIPPER = X_USDWRIST3_URDFWRIST3 * X_WRIST3_GRIPPER

# round_belt.sdf pose "x y 0  0 0 1.57079" applied to the X-major reference ellipse
# (major 0.248 along X, minor 0.168 along Y): the +90 deg yaw turns the major axis onto
# world +Y, hence semi-axis X = 0.168/2, semi-axis Y = 0.248/2.
BELT_CENTER = (0.4736603358808432, 0.3520562100563749, 0.0)
BELT_SEMI_AXIS_X = 0.084
BELT_SEMI_AXIS_Y = 0.124
BELT_RADIUS = round_belt.BELT_RADIUS  # 0.0033
BELT_NUM_ELEMENTS = round_belt.BELT_NUM_ELEMENTS  # 48
# Drake's belt orange (round_belt.sdf: <diffuse>1.0 0.5 0.0 1</diffuse>).
BELT_COLOR = (1.0, 0.5, 0.0)

# Colours for parts whose assets author none of their own: the pulley OBJs carry no MTL
# data and the board URDF paints every pulley sub-mesh one flat colour, so the small
# pulley and its mounting plate came out identically white.
#
# These follow round_belt.py:640-695, NOT round_belt_task_board.sdf: that SDF paints
# every pulley half black (<diffuse>0 0 0 1</diffuse>) and leaves the board -- which the
# mounting plate is part of -- light grey, i.e. the inverse of the real hardware.
# round_belt.py has it right: white pulley (small halves 0.80) on a black mount
# (bracket/bearing/bolt 0.10, large pulley 0.10).
SMALL_PULLEY_COLOR = (0.80, 0.80, 0.80)
LARGE_PULLEY_COLOR = (0.10, 0.10, 0.10)
PULLEY_MOUNT_COLOR = (0.10, 0.10, 0.10)
BOARD_COLOR = (0.80, 0.80, 0.80)
# The small pulley's mounting plate is not its own file here: it is baked into
# round_belt_task_board.obj, one material spanning 18 disconnected solids.  The plate and
# its four corner bolts sit at this board-local XY (where round_belt.py places its
# separate bracket mesh); splitting the mesh lets just those parts go black.
SMALL_PULLEY_MOUNT_LOCAL_XY = (0.3504, 0.1964)
SMALL_PULLEY_MOUNT_RADIUS = 0.05
BOARD_PANEL_MIN_SPAN = 0.2  # the 384 x 384 mm panel: never recolour it as if it were the plate

# Drake's Robotiq fingers (external/robotiq-driver+/models/robotiq_arg85_parallel_grippers.sdf):
# left/right_finger links at (+/-0.047285310862444, 0, 0.1148045193817614) in the gripper
# base frame; 2f85.xml's root body carries a 7 mm +Z offset to be undone to reach it
# (``<body name="base_mount" pos="0 0 0.007">``).
ROBOTIQ_FINGER_DIR = COMMON_ASSETS_DIR / "robotiq_2f85" / "fingers"
ALOHA_FINGER_OFFSET_X = 0.047285310862444
ALOHA_FINGER_OFFSET_Z = 0.1148045193817614
MJCF_BASE_MOUNT_OFFSET_Z = 0.007
# Drake's SDF paints these fingers orange (<diffuse>1 0.5 0 1</diffuse>); black is a
# deliberate deviation, matching the 2f85 body they mount on and the SDF's own
# robotiq_85_base_link (<diffuse>0.1 0.1 0.1 1</diffuse>).
ALOHA_FINGER_COLOR = (0.10, 0.10, 0.10)

# The belt-holder weld height; also the Drake table-top height.
TABLE_TOP_Z = -0.02858

# Safety collision box under the belt; not part of the Drake scene.
TABLETOP_COLLISION_THICKNESS = 0.04

# Franka start pose == what the Drake round-belt SIM actually seeds, not the "ready"
# pose from franka.dmd.yaml:
#   magna/systems/parameters/round_belt_simulation_params.yaml:9-10
#       q_init_franka      = [1.71717, 1.20002, -1.4548, -2.25837, 1.259265, 1.94719, 0.331838]
#       q_init_franka_hand = [-0.004, 0.004]
#   magna/systems/simulation/magna_simulation.cc:187-189 -- plant.SetPositions(...,
#       franka_index, sim_params.q_init_franka) and the hand equivalent.
# round-belt-scene.dmd.yaml adds panda_arm.urdf with NO ``default_joint_positions`` and
# never loads franka.dmd.yaml, so that file's ready pose is not what this scene starts
# in.  The finger values are asymmetric because panda_hand_with_long_fingers.urdf limits
# panda_finger_joint1 to [-0.045, 0.0] and panda_finger_joint2 to [0.0, 0.045];
# -0.004/+0.004 is a 4 mm-per-jaw opening inside both limits (no clamping needed).
PANDA_DEFAULT_Q = [1.71717, 1.20002, -1.4548, -2.25837, 1.259265, 1.94719, 0.331838]
PANDA_JOINT_LABELS = [f"panda_arm/panda_joint{i}" for i in range(1, 8)]
PANDA_FINGER_LABELS = ["panda_hand/panda_finger_joint1", "panda_hand/panda_finger_joint2"]
PANDA_FINGER_DEFAULT_Q = [-0.004, 0.004]

# ur.dmd.yaml default_joint_positions
UR10_DEFAULT_Q = [0.0, -1.57079632679, 1.57079632679, -1.57079632679, -1.57079632679, 0.0]
# USD joint labels are full prim paths ("/ur10/<parent link>/<joint>"); the six names and
# their order are identical to the Drake URDF's, hence UR10_DEFAULT_Q applies.
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

ARM_TARGET_KE = 700.0
ARM_TARGET_KD = 110.0
FINGER_TARGET_KE = 100.0
FINGER_TARGET_KD = 10.0
GRIPPER_OPEN_MARGIN = 0.005
