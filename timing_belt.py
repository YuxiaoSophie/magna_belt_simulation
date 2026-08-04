"""
Round-belt scene -- SolverCoupledProxy version.

Coupled MuJoCo (robot) + VBD (deformable belt + free-spinning pulleys + static
world).  

TUNING SUMMARY (what changed vs. the original, and why)
Goal 1 - grip the belt firmly so it never slips out of the gripper while it is
         carried and dragged from one pulley to the other:
   * GRIPPER_CONTACT_MU raised 2.0 -> 4.0
   * GRIPPER_CONTACT_KE raised 1e4 -> 2e4 (and the pad ke/kd are now RE-APPLIED
     after the global shape_material fill_, which previously reset them)
   * ROBOTIQ_GRIPPER_SAFE_CLOSE_FRACTION raised 0.90 -> 0.93
   * arm + gripper PD gains stiffened in _configure_robot_joints()

Goal 2 - once the belt is on a sheave it must not slip through the groove / off
         the bottom of the pulleys:
   * PULLEY_SHEAVE_MU raised 1.0 -> 2.5
   * PULLEY_CONTACT_KE raised 1e5 -> 3e5, PULLEY_CONTACT_KD added
   * PULLEY_GROOVE_HALF_WIDTH narrowed (1.55 -> 1.35 * belt_r) and
     PULLEY_FLANGE_EXTRA_RADIUS raised (3.2 -> 4.0 * belt_r): taller, snugger
     flanges that physically retain the belt in Z
   * VBD_RIGID_CONTACT_K_START raised 1e3 -> 3e3 (stiffer belt<->pulley contact)

The place MOTION (small pulley first, then roll forward onto the large pulley)
lives in the command script. The small-pulley target poses it needs are
exported here as BELT_PLACE_SMALL_ABOVE / BELT_PLACE_SMALL_DOWN.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_BOARD_URDF_DIR = SCRIPT_DIR / "task_board_urdf"

ROUND_BELT_XACRO = SCRIPT_DIR / "round_belt.urdf.xacro"
if not ROUND_BELT_XACRO.exists():
    ROUND_BELT_XACRO = TASK_BOARD_URDF_DIR / "round_belt.urdf.xacro"

TABLE_OBJ = TASK_BOARD_URDF_DIR / "common" / "table" / "table.obj"
BOARD_MESH = TASK_BOARD_URDF_DIR / "common" / "task_board_just_board.glb"

SMALL_DIR = TASK_BOARD_URDF_DIR / "round_belt_task" / "round_belt_task_board" / "small_round_pulley"
LARGE_DIR = TASK_BOARD_URDF_DIR / "round_belt_task" / "round_belt_task_board" / "large_round_pulley"
SMALL_BRACKET_MESH = SMALL_DIR / "slide_tensioner_bracket.gltf"
SMALL_BEARING_MESH = SMALL_DIR / "slide_tensioner_bearing.gltf"
SMALL_BOLT_MESH = SMALL_DIR / "slide_tensioner_bolt.gltf"
SMALL_HALF_MESH = SMALL_DIR / "small_round_pulley_half.obj"
LARGE_HALF_MESH = LARGE_DIR / "large_round_pulley_half.obj"

TABLE_LENGTH_X = 1.20
TABLE_WIDTH_Y = 0.70
TABLE_BOTTOM_Z = 0.0
TABLE_HEIGHT = 0.72
TABLE_TOP_Z = TABLE_BOTTOM_Z + TABLE_HEIGHT
TABLE_THICKNESS = 0.04

BOARD_SIZE = 0.384
BOARD_THICKNESS = 0.010
BOARD_ROOT_X = 0.150
BOARD_ROOT_Y = -0.192
BOARD_ROOT_Z = TABLE_TOP_Z + BOARD_THICKNESS
BOARD_YAW = math.pi  # rotate the complete task board 180 degrees about its center

# Cloth timing-belt geometry and material (SI units).
BELT_OUTER_MAJOR_DIAMETER = 0.248
BELT_OUTER_MINOR_DIAMETER = 0.168
BELT_STRIP_WIDTH = 0.050
BELT_CIRCUMFERENCE_CELLS = 128
BELT_WIDTH_CELLS = 10
BELT_PARTICLE_RADIUS = 0.0010
BELT_GROUND_EPSILON = 0.0005
BELT_CLOTH_DENSITY = 10.0

# Ordinary membrane stiffness.
BELT_TRI_KE = 1.0e3
BELT_TRI_KA = 1.0e3
BELT_TRI_KD = 1.0e2

# Low ordinary bending lets the loop curve around the pulley axes.
BELT_BASE_EDGE_KE = 1.0e1
BELT_BASE_EDGE_KD = 0.0

# High anisotropic rib stiffness prevents the 50-mm strip from folding across
# its width, while still allowing circumferential wrapping.
BELT_RIB_BENDING_KE = 2.0e4
BELT_RIB_BENDING_KD = 2.0e1
BELT_RIB_SPRING_KE = 2.0e4
BELT_RIB_SPRING_KD = 2.0e1

BELT_CENTER_X = -0.320
BELT_CENTER_Y = 0.000
BELT_BOTTOM_Z = TABLE_TOP_Z + BELT_PARTICLE_RADIUS + BELT_GROUND_EPSILON
BELT_CENTER_Z = BELT_BOTTOM_Z + 0.5 * BELT_STRIP_WIDTH

# Compatibility name used by the existing motion script.  For a strip belt,
# this is the contact offset from a pulley sheave to the strip mid-surface.
BELT_RADIUS = BELT_PARTICLE_RADIUS

# UR10 + Robotiq 2F-85 gripper placement.
UR10_ARM_DOFS = 6
UR10_STAND_RADIUS = 0.08
UR10_STAND_HEIGHT = TABLE_TOP_Z + 0.15
UR10_EDGE_CLEARANCE = 0.15  # gap between the table edge and the robot stand

UR10_BASE_X = BELT_CENTER_X
UR10_BASE_Y = 0.5 * TABLE_WIDTH_Y + UR10_EDGE_CLEARANCE + UR10_STAND_RADIUS
UR10_BASE_Z = UR10_STAND_HEIGHT
UR10_BASE_YAW = -math.pi / 2.0  # face -Y, i.e. toward the belt/board

UR10_ARM_HOME_POSE = [0.0, -1.35, 1.75, -1.95, -1.57, 0.0]
UR10_GRIPPER_CLOSED_FALLBACK = 0.8

# Task-space grasp geometry
BELT_GRASP_POS = (
    BELT_CENTER_X + 0.5 * BELT_OUTER_MAJOR_DIAMETER,
    BELT_CENTER_Y,
    BELT_CENTER_Z,
)
BELT_APPROACH_CLEARANCE = 0.16
BELT_APPROACH_POS = (
    BELT_GRASP_POS[0],
    BELT_GRASP_POS[1],
    BELT_GRASP_POS[2] + BELT_APPROACH_CLEARANCE,
)

GRIPPER_TABLE_YAW = math.pi / 2.0
GRIPPER_DOWN_QUAT = (
    math.cos(0.5 * GRIPPER_TABLE_YAW),
    math.sin(0.5 * GRIPPER_TABLE_YAW),
    0.0,
    0.0,
)

GRIPPER_TCP_LOCAL_OFFSET = (0.0, 0.0, 0.145)

IK_INIT_ITERS = 96
IK_TRACK_ITERS = 24
IK_LAMBDA_INITIAL = 0.05

# Contact material.
CABLE_CONTACT_KE = 1.0e4
CABLE_CONTACT_KD = 1.0e-5 * CABLE_CONTACT_KE
CABLE_CONTACT_MU = 1.0

# Stiffer + damped pulley contact so the belt cannot penetrate the
# sheave and squeeze out the bottom of the groove.
PULLEY_CONTACT_KE = 3.0e5                        
PULLEY_CONTACT_KD = 1.0e-5 * PULLEY_CONTACT_KE   

# Firmer, higher-friction finger pads so the belt does not slip out of
# the gripper while it is carried and dragged over both pulleys.
GRIPPER_CONTACT_KE = 2.0e4                        
GRIPPER_CONTACT_KD = 1.0e-5 * GRIPPER_CONTACT_KE
GRIPPER_CONTACT_MU = 4.0                        

# Pulley parameters.
PULLEY_DENSITY = 1000.0
PULLEY_SHEAVE_MU = 2.5           
PULLEY_FLANGE_MU = 0.0           
PULLEY_ARMATURE = 1.0e-5
PULLEY_JOINT_FRICTION = 2.0e-4   
PULLEY_AXIS = (0.0, 0.0, 1.0)    

SMALL_PULLEY_SHEAVE_RADIUS = 0.015
LARGE_PULLEY_SHEAVE_RADIUS = 0.035

# Narrower groove + taller flanges
# The original round-belt groove was only a few millimetres wide.  The cloth
# strip is 50 mm wide along Z, so its collision sheave must span that width.
PULLEY_GROOVE_HALF_WIDTH = 0.5 * BELT_STRIP_WIDTH + 2.0 * BELT_PARTICLE_RADIUS
PULLEY_FLANGE_HALF_THICKNESS = 1.5 * BELT_PARTICLE_RADIUS
PULLEY_FLANGE_EXTRA_RADIUS = 8.0 * BELT_PARTICLE_RADIUS

# Close the collision gap between each pulley and the board.
PULLEY_BOARD_GAP = 0.0005  # 0.5 mm numerical clearance above the board

PULLEY_SHOW_COLLISION_SHAPES = False

GRIPPER_PROXY_PAD_BODIES = ("left_pad", "right_pad")
GRIPPER_PROXY_FALLBACK_BODIES = ("left_follower", "right_follower")
GRIPPER_PAD_KEYWORDS = ("pad",)

ROBOTIQ_GRIPPER_SAFE_CLOSE_FRACTION = 0.93

# Proxy-coupling / solver settings.
PROXY_ITERATIONS = 1
PROXY_MASS_SCALE = 1.0
PROXY_COUPLING_MODE = "lagged"
VBD_ITERATIONS = 20
VBD_RIGID_AVBD_BETA = 1.0e2
VBD_RIGID_CONTACT_K_START = 3.0e3
VBD_RIGID_CONTACT_BUFFER_SIZE = 256
MUJOCO_ITERATIONS = 50
MUJOCO_LS_ITERATIONS = 20


# Geometry
def quat_from_rpy(roll: float, pitch: float, yaw: float) -> wp.quat:
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return wp.quat(qx, qy, qz, qw)


def tf(xyz, rpy=(0.0, 0.0, 0.0)) -> wp.transform:
    return wp.transform(
        wp.vec3(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        quat_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])),
    )


def board_world(local_xyz):
    return (
        BOARD_ROOT_X + BOARD_SIZE - float(local_xyz[0]),
        BOARD_ROOT_Y + BOARD_SIZE - float(local_xyz[1]),
        BOARD_ROOT_Z + float(local_xyz[2]),
    )


def board_rpy(rpy=(0.0, 0.0, 0.0)):
    return (float(rpy[0]), float(rpy[1]), float(rpy[2]) + BOARD_YAW)


# Large-pulley placement geometry (final destination of the grasped belt vertex)
LARGE_PULLEY_CENTER_LOCAL = (0.140, 0.196, 0.0248)
LARGE_PULLEY_WORLD_CENTER = board_world(LARGE_PULLEY_CENTER_LOCAL)

LARGE_PULLEY_PLACE_EDGE_OFFSET = (LARGE_PULLEY_SHEAVE_RADIUS + BELT_RADIUS, 0.0, 0.0)
_place_x = LARGE_PULLEY_WORLD_CENTER[0] + LARGE_PULLEY_PLACE_EDGE_OFFSET[0]
_place_y = LARGE_PULLEY_WORLD_CENTER[1] + LARGE_PULLEY_PLACE_EDGE_OFFSET[1]

BELT_PLACE_ABOVE = (_place_x, _place_y, LARGE_PULLEY_WORLD_CENTER[2] + BELT_APPROACH_CLEARANCE)
BELT_PLACE_DOWN  = (_place_x, _place_y, LARGE_PULLEY_WORLD_CENTER[2] + BELT_RADIUS)

# Small-pulley placement geometry (the pulley the belt seats on FIRST).
# SMALL_PULLEY_CENTER_LOCAL matches the pose used in add_pulleys_from_xacro_poses.
SMALL_PULLEY_CENTER_LOCAL = (0.3504 - 0.01200845, 0.1964 - 0.0004, 0.0248)
SMALL_PULLEY_WORLD_CENTER = board_world(SMALL_PULLEY_CENTER_LOCAL)

SMALL_PULLEY_PLACE_EDGE_OFFSET = (SMALL_PULLEY_SHEAVE_RADIUS + BELT_RADIUS, 0.0, 0.0)
_small_place_x = SMALL_PULLEY_WORLD_CENTER[0] + SMALL_PULLEY_PLACE_EDGE_OFFSET[0]
_small_place_y = SMALL_PULLEY_WORLD_CENTER[1] + SMALL_PULLEY_PLACE_EDGE_OFFSET[1]

BELT_PLACE_SMALL_ABOVE = (_small_place_x, _small_place_y, SMALL_PULLEY_WORLD_CENTER[2] + BELT_APPROACH_CLEARANCE)
BELT_PLACE_SMALL_DOWN  = (_small_place_x, _small_place_y, SMALL_PULLEY_WORLD_CENTER[2] + BELT_RADIUS)

BELT_PLACE_LARGE_ABOVE = BELT_PLACE_ABOVE
BELT_PLACE_LARGE_DOWN = BELT_PLACE_DOWN


def make_visual_cfg() -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=0.0, has_shape_collision=False, has_particle_collision=False,
        collision_group=0, is_visible=True,
    )


def make_robust_table_collision_cfg(visible=True) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=0.0, ke=CABLE_CONTACT_KE, kd=CABLE_CONTACT_KD, mu=CABLE_CONTACT_MU,
        has_shape_collision=True, has_particle_collision=True,
        collision_group=1, is_visible=visible,
    )


def make_pulley_shape_cfg(mu: float, visible: bool) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=PULLEY_DENSITY, ke=PULLEY_CONTACT_KE, kd=PULLEY_CONTACT_KD, mu=mu,
        has_shape_collision=True, has_particle_collision=True,
        collision_group=1, is_visible=visible,
    )


def _dim_color(color, scale: float):
    return tuple(max(0.0, min(1.0, float(c) * scale)) for c in color)


def add_visual_mesh(builder, path, xyz, rpy=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
                    color=(0.8, 0.8, 0.8), label="visual_mesh", body: int = -1) -> bool:
    if not path.exists():
        print(f"[missing mesh] {path}")
        return False
    try:
        mesh = newton.Mesh.create_from_file(str(path), compute_inertia=False, is_solid=False)
        builder.add_shape_mesh(
            body=body, xform=tf(xyz, rpy), mesh=mesh,
            scale=wp.vec3(float(scale[0]), float(scale[1]), float(scale[2])),
            cfg=make_visual_cfg(),
            color=wp.vec3(float(color[0]), float(color[1]), float(color[2])),
            label=label,
        )
        return True
    except Exception as e:
        print(f"[failed mesh] {path}: {e}")
        return False


def read_obj_as_triangle_mesh(path: Path):
    vertices = []; indices = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                face = []
                for token in parts[1:]:
                    idx_str = token.split("/")[0]
                    idx = int(idx_str)
                    idx = len(vertices) + idx if idx < 0 else idx - 1
                    face.append(idx)
                for i in range(1, len(face) - 1):
                    indices.extend([face[0], face[i], face[i + 1]])
    if len(vertices) == 0 or len(indices) == 0:
        raise RuntimeError(f"No usable vertices/faces in {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)


def fit_table_vertices(vertices: np.ndarray) -> np.ndarray:
    v = vertices.astype(np.float32).copy()
    v_min = v.min(axis=0); v_max = v.max(axis=0)
    dims = np.maximum(v_max - v_min, 1.0e-8)
    center_xy = 0.5 * (v_min[:2] + v_max[:2])
    scale_xy = min(TABLE_LENGTH_X / dims[0], TABLE_WIDTH_Y / dims[1])
    v[:, 0] = (v[:, 0] - center_xy[0]) * scale_xy
    v[:, 1] = (v[:, 1] - center_xy[1]) * scale_xy
    v[:, 2] = (v[:, 2] - v_min[2]) * (TABLE_HEIGHT / dims[2]) + TABLE_BOTTOM_Z
    return v


def add_table(builder: newton.ModelBuilder) -> None:
    table_mesh_loaded = False
    if TABLE_OBJ.exists():
        try:
            vertices, indices = read_obj_as_triangle_mesh(TABLE_OBJ)
            vertices = fit_table_vertices(vertices)
            mesh = newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False)
            builder.add_shape_mesh(
                body=-1, xform=tf((0.0, 0.0, 0.0)), mesh=mesh,
                scale=wp.vec3(1.0, 1.0, 1.0), cfg=make_visual_cfg(),
                color=wp.vec3(0.55, 0.35, 0.14), label="common_table_obj_visual",
            )
            table_mesh_loaded = True
        except Exception as e:
            print(f"[TABLE] failed to load {TABLE_OBJ}: {e}")

    builder.add_shape_box(
        body=-1, xform=tf((0.0, 0.0, TABLE_TOP_Z - 0.5 * TABLE_THICKNESS)),
        hx=0.5 * TABLE_LENGTH_X, hy=0.5 * TABLE_WIDTH_Y, hz=0.5 * TABLE_THICKNESS,
        cfg=make_robust_table_collision_cfg(visible=not table_mesh_loaded),
        color=wp.vec3(0.55, 0.35, 0.14), label="tabletop_collision",
    )

    if not table_mesh_loaded:
        leg_cfg = make_visual_cfg()
        leg_hx, leg_hy = 0.025, 0.025
        leg_height = TABLE_TOP_Z - TABLE_THICKNESS - TABLE_BOTTOM_Z
        leg_hz = 0.5 * leg_height
        leg_center_z = TABLE_BOTTOM_Z + leg_hz
        x_edge = 0.5 * TABLE_LENGTH_X - 0.08
        y_edge = 0.5 * TABLE_WIDTH_Y - 0.08
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                builder.add_shape_box(
                    body=-1, xform=tf((sx * x_edge, sy * y_edge, leg_center_z)),
                    hx=leg_hx, hy=leg_hy, hz=leg_hz, cfg=leg_cfg,
                    color=wp.vec3(0.35, 0.20, 0.08), label="fallback_table_leg",
                )


def add_board(builder: newton.ModelBuilder) -> None:
    board_mesh_ok = add_visual_mesh(
        builder, BOARD_MESH, xyz=board_world((0.0, 0.0, 0.0)),
        rpy=board_rpy(), scale=(1.0, 1.0, 1.0),
        color=(0.80, 0.80, 0.80), label="task_board_mesh_visual",
    )
    builder.add_shape_box(
        body=-1, xform=tf(board_world((0.192, 0.192, -0.005)), board_rpy()),
        hx=0.5 * BOARD_SIZE, hy=0.5 * BOARD_SIZE, hz=0.5 * BOARD_THICKNESS,
        cfg=make_robust_table_collision_cfg(visible=not board_mesh_ok),
        color=wp.vec3(0.82, 0.82, 0.82), label="board_collision_exact_xacro",
    )


def add_dynamic_pulley(builder, center, sheave_radius, color, label,
                       collision_visible=PULLEY_SHOW_COLLISION_SHAPES) -> dict[str, Any]:
    """Free-spinning pulley: link + revolute axle about +Z + grooved sheave
    sandwiched between two frictionless flanges (as in Newton's XY-table example)."""
    body = builder.add_link(xform=tf(center), label=f"{label}_body")
    joint = builder.add_joint_revolute(
        parent=-1, child=body, axis=wp.vec3(*PULLEY_AXIS),
        parent_xform=tf(center),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        armature=PULLEY_ARMATURE, friction=PULLEY_JOINT_FRICTION,
        label=f"{label}_free_axle",
    )
    builder.add_articulation([joint], label=f"{label}_articulation")

    flange_radius = sheave_radius + PULLEY_FLANGE_EXTRA_RADIUS
    flange_z = PULLEY_GROOVE_HALF_WIDTH + PULLEY_FLANGE_HALF_THICKNESS
    flange_color = _dim_color(color, 0.68)

    sheave = builder.add_shape_cylinder(
        body=body, xform=tf((0.0, 0.0, 0.0)), radius=sheave_radius,
        half_height=PULLEY_GROOVE_HALF_WIDTH,
        cfg=make_pulley_shape_cfg(PULLEY_SHEAVE_MU, collision_visible),
        color=wp.vec3(*[float(c) for c in color]), label=f"{label}_sheave",
    )
    flanges = []
    for suffix, z in (("flange_neg", -flange_z), ("flange_pos", flange_z)):
        flanges.append(builder.add_shape_cylinder(
            body=body, xform=tf((0.0, 0.0, z)), radius=flange_radius,
            half_height=PULLEY_FLANGE_HALF_THICKNESS,
            cfg=make_pulley_shape_cfg(PULLEY_FLANGE_MU, collision_visible),
            color=wp.vec3(*[float(c) for c in flange_color]), label=f"{label}_{suffix}",
        ))

    # The original collision model left a large empty space below the lower
    # flange because the axle centre is about 24.8 mm above the board.  A belt
    # segment could therefore pass under the pulley even though the visual mesh
    # looked mounted to the board.  Fill that space with a rotationally symmetric
    # pedestal attached to the same free-spinning body.
    lower_flange_bottom_local = -flange_z - PULLEY_FLANGE_HALF_THICKNESS
    board_top_local = BOARD_ROOT_Z - float(center[2]) + PULLEY_BOARD_GAP
    pedestal_half_height = 0.5 * max(lower_flange_bottom_local - board_top_local, 1.0e-4)
    pedestal_center_z = board_top_local + pedestal_half_height

    pedestal = builder.add_shape_cylinder(
        body=body,
        xform=tf((0.0, 0.0, pedestal_center_z)),
        radius=flange_radius,
        half_height=pedestal_half_height,
        cfg=make_pulley_shape_cfg(PULLEY_FLANGE_MU, collision_visible),
        color=wp.vec3(*[float(c) for c in flange_color]),
        label=f"{label}_board_gap_guard",
    )

    return {
        "body": body,
        "joint": joint,
        "sheave": sheave,
        "flanges": flanges,
        "guard": pedestal,
    }


def add_pulleys_from_xacro_poses(builder: newton.ModelBuilder) -> dict[str, Any]:
    visual_cfg = make_visual_cfg()

    small_bracket_xyz = board_world((0.3504, 0.1964, 0.0))
    bracket_ok = add_visual_mesh(
        builder, SMALL_BRACKET_MESH, xyz=small_bracket_xyz,
        rpy=board_rpy((math.pi / 2.0, 0.0, math.pi / 2.0)), scale=(1.0, 1.0, 1.0),
        color=(0.10, 0.10, 0.10), label="small_round_pulley_bracket_visual_exact_xacro",
    )
    if not bracket_ok:
        builder.add_shape_box(
            body=-1, xform=tf(board_world((0.3504, 0.1964, 0.010)), board_rpy()),
            hx=0.018, hy=0.020, hz=0.006, cfg=visual_cfg,
            color=wp.vec3(0.10, 0.10, 0.10), label="small_bracket_fallback_visual",
        )

    small_center_local = (0.3504 - 0.01200845, 0.1964 - 0.0004, 0.0248)
    small_center_xyz = board_world(small_center_local)
    add_visual_mesh(
        builder, SMALL_BEARING_MESH,
        xyz=board_world((small_center_local[0], small_center_local[1], small_center_local[2] + 0.0035)),
        rpy=board_rpy((0.0, math.pi / 2.0, 0.0)), scale=(1.0, 1.0, 1.0),
        color=(0.10, 0.10, 0.10), label="small_bearing_visual_exact_xacro",
    )
    add_visual_mesh(
        builder, SMALL_BOLT_MESH, xyz=small_center_xyz,
        rpy=board_rpy((0.0, math.pi / 2.0, 0.0)), scale=(1.0, 1.0, 1.0),
        color=(0.10, 0.10, 0.10), label="small_bolt_visual_exact_xacro",
    )

    small = add_dynamic_pulley(builder, center=small_center_xyz,
                               sheave_radius=SMALL_PULLEY_SHEAVE_RADIUS,
                               color=(0.80, 0.80, 0.80), label="small_round_pulley")
    small_half_1_ok = add_visual_mesh(builder, SMALL_HALF_MESH, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001), color=(0.80, 0.80, 0.80),
        label="small_pulley_first_half_exact_xacro", body=small["body"])
    small_half_2_ok = add_visual_mesh(builder, SMALL_HALF_MESH, xyz=(0.0, 0.0, 0.0), rpy=(math.pi, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001), color=(0.80, 0.80, 0.80),
        label="small_pulley_second_half_exact_xacro", body=small["body"])
    if not (small_half_1_ok and small_half_2_ok):
        print("[WARNING] Small pulley half mesh missing; the collision sheave is the only visual.")

    large_center_xyz = board_world(LARGE_PULLEY_CENTER_LOCAL)
    large = add_dynamic_pulley(builder, center=large_center_xyz,
                               sheave_radius=LARGE_PULLEY_SHEAVE_RADIUS,
                               color=(0.10, 0.10, 0.10), label="large_round_pulley")
    large_half_1_ok = add_visual_mesh(builder, LARGE_HALF_MESH, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001), color=(0.10, 0.10, 0.10),
        label="large_pulley_first_half_exact_xacro", body=large["body"])
    large_half_2_ok = add_visual_mesh(builder, LARGE_HALF_MESH, xyz=(0.0, 0.0, 0.0), rpy=(math.pi, 0.0, 0.0),
        scale=(0.001, 0.001, 0.001), color=(0.10, 0.10, 0.10),
        label="large_pulley_second_half_exact_xacro", body=large["body"])
    if not (large_half_1_ok and large_half_2_ok):
        print("[WARNING] Large pulley half mesh missing; the collision sheave is the only visual.")

    bodies = [small["body"], large["body"]]
    joints = [small["joint"], large["joint"]]
    sheave_shapes = [small["sheave"], large["sheave"]]
    flange_shapes = [*small["flanges"], *large["flanges"]]
    guard_shapes = [small["guard"], large["guard"]]
    print(f"[INFO] Added {len(bodies)} free-spinning pulleys (sheave radii "
          f"{SMALL_PULLEY_SHEAVE_RADIUS:.4f} / {LARGE_PULLEY_SHEAVE_RADIUS:.4f} m).")
    return {
        "bodies": bodies,
        "joints": joints,
        "sheave_shapes": sheave_shapes,
        "flange_shapes": flange_shapes,
        "guard_shapes": guard_shapes,
        "shapes": sheave_shapes + flange_shapes + guard_shapes,
    }


def build_elliptical_cloth_belt_mesh(
    center_x: float,
    center_y: float,
    bottom_z: float,
    major_diameter: float = BELT_OUTER_MAJOR_DIAMETER,
    minor_diameter: float = BELT_OUTER_MINOR_DIAMETER,
    strip_width: float = BELT_STRIP_WIDTH,
    circumference_cells: int = BELT_CIRCUMFERENCE_CELLS,
    width_cells: int = BELT_WIDTH_CELLS,
):
    """Create the seamless vertical cloth-strip loop used by the task scene."""
    a = 0.5 * major_diameter
    b = 0.5 * minor_diameter
    row_size = width_cells + 1
    vertices = []
    indices = []

    for i in range(circumference_cells):
        theta = 2.0 * math.pi * i / circumference_cells
        x = center_x + a * math.cos(theta)
        y = center_y + b * math.sin(theta)
        for j in range(row_size):
            z = bottom_z + strip_width * j / width_cells
            vertices.append(wp.vec3(x, y, z))

    def vid(i: int, j: int) -> int:
        return (i % circumference_cells) * row_size + j

    for i in range(circumference_cells):
        ni = (i + 1) % circumference_cells
        for j in range(width_cells):
            v00, v10 = vid(i, j), vid(ni, j)
            v11, v01 = vid(ni, j + 1), vid(i, j + 1)
            indices.extend((v00, v10, v01, v10, v11, v01))
    return vertices, indices


def _cloth_particle_id(particle_start: int, i: int, j: int) -> int:
    return particle_start + (i % BELT_CIRCUMFERENCE_CELLS) * (BELT_WIDTH_CELLS + 1) + j


def make_cloth_cross_sections_stiff(builder, particle_start: int, edge_start: int, edge_end: int) -> int:
    """Raise bending stiffness only for hinges that fold a width-wise rib."""
    row_size = BELT_WIDTH_CELLS + 1
    particle_end = particle_start + BELT_CIRCUMFERENCE_CELLS * row_size
    count = 0

    def decode(pid: int):
        return divmod(pid - particle_start, row_size)

    for edge_id in range(edge_start, edge_end):
        opposite_0, opposite_1, hinge_0, hinge_1 = builder.edge_indices[edge_id]
        if opposite_0 == -1 or opposite_1 == -1:
            continue
        if not (particle_start <= hinge_0 < particle_end and particle_start <= hinge_1 < particle_end):
            continue
        circ_0, width_0 = decode(hinge_0)
        circ_1, width_1 = decode(hinge_1)
        delta = (circ_1 - circ_0) % BELT_CIRCUMFERENCE_CELLS
        if width_0 == width_1 and delta in (1, BELT_CIRCUMFERENCE_CELLS - 1):
            builder.edge_bending_properties[edge_id] = (
                float(BELT_RIB_BENDING_KE), float(BELT_RIB_BENDING_KD))
            count += 1
    return count


def add_cloth_cross_section_springs(builder, particle_start: int) -> int:
    """Keep each vertical cross-section straight and at its original width."""
    before = builder.spring_count
    for i in range(BELT_CIRCUMFERENCE_CELLS):
        bottom = _cloth_particle_id(particle_start, i, 0)
        top = _cloth_particle_id(particle_start, i, BELT_WIDTH_CELLS)
        builder.add_spring(bottom, top, BELT_RIB_SPRING_KE, BELT_RIB_SPRING_KD, 0.0)
        for j in range(1, BELT_WIDTH_CELLS):
            p = _cloth_particle_id(particle_start, i, j)
            builder.add_spring(bottom, p, BELT_RIB_SPRING_KE, BELT_RIB_SPRING_KD, 0.0)
            builder.add_spring(p, top, BELT_RIB_SPRING_KE, BELT_RIB_SPRING_KD, 0.0)
    return builder.spring_count - before


def as_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@wp.kernel
def _set_single_task_target(target_positions: wp.array[wp.vec3], target_rotations: wp.array[wp.vec4],
                            pos: wp.vec3, rot: wp.vec4):
    target_positions[0] = pos
    target_rotations[0] = rot


@wp.kernel
def _scatter_ik_arm_targets(control_target_q: wp.array[float], ik_joint_q: wp.array2d[float],
                            ik_coord_indices: wp.array[int], control_target_indices: wp.array[int]):
    i = wp.tid()
    control_target_q[control_target_indices[i]] = ik_joint_q[0, ik_coord_indices[i]]


@wp.kernel
def _scatter_gripper_targets(control_target_q: wp.array[float], control_target_indices: wp.array[int],
                             desired_values: wp.array[float]):
    i = wp.tid()
    control_target_q[control_target_indices[i]] = desired_values[i]


def _shape_label_lower(builder, shape_index: int) -> str:
    for attr in ("shape_label", "shape_key"):
        arr = getattr(builder, attr, None)
        if arr is not None and shape_index < len(arr) and arr[shape_index]:
            return str(arr[shape_index]).lower()
    return ""


def _body_label_lower(builder, body_index: int) -> str:
    for attr in ("body_label", "body_key"):
        arr = getattr(builder, attr, None)
        if arr is not None and body_index < len(arr) and arr[body_index]:
            return str(arr[body_index]).lower()
    return ""


def _apply_gripper_pad_contact_material(builder, first_shape_index, end_shape_index,
                                        ke=GRIPPER_CONTACT_KE, kd=GRIPPER_CONTACT_KD,
                                        mu=GRIPPER_CONTACT_MU) -> int:
    n = 0
    end = min(end_shape_index, builder.shape_count)
    for i in range(first_shape_index, end):
        if not any(kw in _shape_label_lower(builder, i) for kw in GRIPPER_PAD_KEYWORDS):
            continue
        for attr, val in (("shape_material_ke", ke), ("shape_material_kd", kd), ("shape_material_mu", mu)):
            arr = getattr(builder, attr, None)
            if arr is not None and i < len(arr):
                arr[i] = val
        n += 1
    return n


def _select_gripper_proxy_bodies(builder, first_body_index, end_body_index) -> list[int]:
    body_info = []
    for b in range(first_body_index, end_body_index):
        leaf = _body_label_lower(builder, b).replace("\\", "/").rsplit("/", 1)[-1]
        body_info.append((b, leaf))

    selected = [b for b, leaf in body_info if leaf in GRIPPER_PROXY_PAD_BODIES]
    if selected:
        print(f"[INFO] Gripper proxy bodies: exact pad bodies only: "
              f"{[(b, leaf) for b, leaf in body_info if b in selected]}")
        return selected

    selected = [b for b, leaf in body_info if leaf in GRIPPER_PROXY_FALLBACK_BODIES]
    if selected:
        print("[WARNING] Exact pad body labels not found; using follower bodies: "
              f"{[(b, leaf) for b, leaf in body_info if b in selected]}")
        return selected

    selected = [b for b, leaf in body_info if "pad" in leaf and "silicone" not in leaf]
    if selected:
        print("[WARNING] Using generic non-silicone pad-labelled proxy bodies: "
              f"{[(b, leaf) for b, leaf in body_info if b in selected]}")
        return selected

    selected = list(range(first_body_index, end_body_index))
    print(f"[WARNING] Could not locate pad/follower bodies; exposing all gripper bodies ({len(selected)}).")
    return selected


def _find_gripper_tcp_body(builder, first_body_index, end_body_index) -> int:
    exact = []; fallback = []
    for b in range(first_body_index, end_body_index):
        leaf = _body_label_lower(builder, b).replace("\\", "/").rsplit("/", 1)[-1]
        if leaf == "base":
            exact.append(b)
        elif "base" in leaf and "mount" not in leaf:
            fallback.append(b)
    if exact:
        body = exact[0]
    elif fallback:
        body = fallback[0]
    elif first_body_index + 1 < end_body_index:
        body = first_body_index + 1
    else:
        body = first_body_index
    print(f"[INFO] IK TCP body: {body} label={_body_label_lower(builder, body)!r}")
    return body


def find_n_actuated_indices(model, start_joint_index, n, end_joint_index=None):
    joint_labels = getattr(model, "joint_label", [])
    q_start = as_numpy(model.joint_q_start)
    qd_start = as_numpy(model.joint_qd_start)
    dof_dim = as_numpy(model.joint_dof_dim)
    end = len(joint_labels) if end_joint_index is None else min(end_joint_index, len(joint_labels))
    coord_indices = []; dof_indices = []
    for j in range(start_joint_index, end):
        dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
        if dofs <= 0:
            continue
        q0 = int(q_start[j]); qd0 = int(qd_start[j])
        for k in range(dofs):
            coord_indices.append(q0 + k); dof_indices.append(qd0 + k)
            if len(coord_indices) >= n:
                return coord_indices, dof_indices
    return coord_indices, dof_indices


def find_gripper_driver_indices(model, start_joint_index, end_joint_index=None):
    joint_labels = getattr(model, "joint_label", [])
    q_start = as_numpy(model.joint_q_start)
    qd_start = as_numpy(model.joint_qd_start)
    dof_dim = as_numpy(model.joint_dof_dim)
    end = len(joint_labels) if end_joint_index is None else min(end_joint_index, len(joint_labels))
    pairs = []
    for j in range(start_joint_index, end):
        lname = joint_labels[j].lower()
        dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
        if dofs <= 0:
            continue
        if "driver_joint" in lname:
            pairs.append((int(q_start[j]), int(qd_start[j])))
    pairs = sorted(set(pairs))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _add_ur10_and_gripper_assets(builder: newton.ModelBuilder, body_offset: int) -> dict[str, Any]:
    base_pos = wp.vec3(UR10_BASE_X, UR10_BASE_Y, UR10_BASE_Z)
    base_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), UR10_BASE_YAW)
    base_tf = wp.transform(base_pos, base_rot)

    asset_path = newton.utils.download_asset("universal_robots_ur10")
    asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")

    arm_shape_start = builder.shape_count
    builder.add_usd(asset_file, xform=base_tf, collapse_fixed_joints=False,
                    enable_self_collisions=False, hide_collision_shapes=True)
    arm_shape_end = builder.shape_count

    tool_body_idx = body_offset + 7
    gripper_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
    gripper_offset_tf = wp.transform(wp.vec3(0.0, 0.0, 0.0), gripper_rot)

    gripper_body_start = builder.body_count
    gripper_shape_start = builder.shape_count
    mjcf_kwargs = dict(parent_body=tool_body_idx, xform=gripper_offset_tf)
    try:
        builder.add_mjcf("2f85.xml", enable_self_collisions=False, **mjcf_kwargs)
    except TypeError:
        print("[INFO] add_mjcf has no enable_self_collisions kwarg; continuing without it.")
        builder.add_mjcf("2f85.xml", **mjcf_kwargs)
    gripper_body_end = builder.body_count
    gripper_shape_end = builder.shape_count

    return {
        "tool_body_idx": tool_body_idx,
        "arm_shape_start": arm_shape_start, "arm_shape_end": arm_shape_end,
        "gripper_body_start": gripper_body_start, "gripper_body_end": gripper_body_end,
        "gripper_shape_start": gripper_shape_start, "gripper_shape_end": gripper_shape_end,
    }


class Example:
    """Base scene. Arm starts directly above the belt grasp vertex, gripper OPEN."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame_id = 0
        self.debug_belt_positions = True
        self.debug_every_n_frames = 60

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.rigid_gap = 0.01
        SolverMuJoCo.register_custom_attributes(builder)
        try:
            SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        except TypeError:
            SolverVBD.register_custom_attributes(builder)

        builder.default_shape_cfg.ke = CABLE_CONTACT_KE
        builder.default_shape_cfg.kd = CABLE_CONTACT_KD
        builder.default_shape_cfg.mu = CABLE_CONTACT_MU

        robot_body_start = builder.body_count
        robot_joint_start = builder.joint_count
        robot_shape_start = builder.shape_count
        asset_info = _add_ur10_and_gripper_assets(builder, robot_body_start)
        robot_body_end = builder.body_count
        robot_joint_end = builder.joint_count
        robot_shape_end = builder.shape_count

        self.robot_bodies = list(range(robot_body_start, robot_body_end))
        self.robot_joints = list(range(robot_joint_start, robot_joint_end))
        self.robot_shapes = list(range(robot_shape_start, robot_shape_end))
        self._robot_joint_count = robot_joint_end
        self._robot_tool_body_idx = asset_info["tool_body_idx"]

        self.gripper_pad_shapes = [
            i for i in range(asset_info["gripper_shape_start"], asset_info["gripper_shape_end"])
            if any(kw in _shape_label_lower(builder, i) for kw in GRIPPER_PAD_KEYWORDS)
        ]
        n_pad = _apply_gripper_pad_contact_material(
            builder, asset_info["gripper_shape_start"], asset_info["gripper_shape_end"])
        print(f"[INFO] Applied gripper contact material to {n_pad} pad shapes.")

        self.gripper_proxy_bodies = _select_gripper_proxy_bodies(
            builder, asset_info["gripper_body_start"], asset_info["gripper_body_end"])
        self._robot_tcp_body_idx = _find_gripper_tcp_body(
            builder, asset_info["gripper_body_start"], asset_info["gripper_body_end"])
        self.robot_proxy_bodies = list(self.gripper_proxy_bodies)
        print(f"[INFO] Proxy coupling exposes {len(self.robot_proxy_bodies)} Robotiq gripper bodies only.")

        try:
            gravcomp = builder.custom_attributes["mujoco:gravcomp"]
            if gravcomp.values is None:
                gravcomp.values = {}
            for b in self.robot_bodies:
                gravcomp.values[b] = 1.0
        except (KeyError, AttributeError):
            print("[INFO] mujoco:gravcomp attribute not available; skipping gravity compensation.")

        builder.add_shape_cylinder(
            -1, xform=wp.transform(wp.vec3(UR10_BASE_X, UR10_BASE_Y, UR10_BASE_Z / 2.0)),
            half_height=UR10_BASE_Z / 2.0, radius=UR10_STAND_RADIUS, cfg=make_visual_cfg())

        add_table(builder)
        add_board(builder)

        pulley_info = add_pulleys_from_xacro_poses(builder)
        self.pulley_bodies = list(pulley_info["bodies"])
        self.pulley_joints = list(pulley_info["joints"])
        self.pulley_sheave_shapes = list(pulley_info["sheave_shapes"])
        self.pulley_flange_shapes = list(pulley_info["flange_shapes"])
        self.pulley_guard_shapes = list(pulley_info["guard_shapes"])
        self.pulley_shapes = list(pulley_info["shapes"])

        # Add the newest cloth belt to the VBD part of the coupled model.
        belt_particle_start = builder.particle_count
        belt_edge_start = builder.edge_count
        belt_shape_start = builder.shape_count

        cloth_vertices, cloth_indices = build_elliptical_cloth_belt_mesh(
            BELT_CENTER_X, BELT_CENTER_Y, BELT_BOTTOM_Z)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=cloth_vertices,
            indices=cloth_indices,
            density=BELT_CLOTH_DENSITY,
            tri_ke=BELT_TRI_KE,
            tri_ka=BELT_TRI_KA,
            tri_kd=BELT_TRI_KD,
            edge_ke=BELT_BASE_EDGE_KE,
            edge_kd=BELT_BASE_EDGE_KD,
            add_springs=False,
            particle_radius=BELT_PARTICLE_RADIUS,
        )
        belt_edge_end = builder.edge_count
        self.belt_particles = list(range(belt_particle_start, builder.particle_count))
        self.belt_bodies = []
        self.belt_joints = []
        self.belt_shapes = list(range(belt_shape_start, builder.shape_count))

        stiff_hinges = make_cloth_cross_sections_stiff(
            builder, belt_particle_start, belt_edge_start, belt_edge_end)
        rib_springs = add_cloth_cross_section_springs(builder, belt_particle_start)
        print(f"[INFO] Cloth belt: {len(self.belt_particles)} particles, "
              f"{stiff_hinges} stiff rib hinges, {rib_springs} rib springs.")

        builder.add_ground_plane()

        self.vbd_shapes = [s for s in range(builder.shape_count) if s >= robot_shape_end]
        self.vbd_bodies = sorted(self.pulley_bodies)
        self.vbd_joints = sorted(self.pulley_joints)

        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device

        # Global reset of material arrays...
        self.model.shape_material_ke.fill_(CABLE_CONTACT_KE)
        self.model.shape_material_kd.fill_(CABLE_CONTACT_KD)
        self.model.shape_material_mu.fill_(CABLE_CONTACT_MU)

        # Re-apply per-shape friction.
        mu_np = self.model.shape_material_mu.numpy().copy()
        if self.gripper_pad_shapes:
            mu_np[np.asarray(self.gripper_pad_shapes, dtype=np.int32)] = GRIPPER_CONTACT_MU
        if self.pulley_sheave_shapes:
            mu_np[np.asarray(self.pulley_sheave_shapes, dtype=np.int32)] = PULLEY_SHEAVE_MU
        if self.pulley_flange_shapes:
            mu_np[np.asarray(self.pulley_flange_shapes, dtype=np.int32)] = PULLEY_FLANGE_MU
        if self.pulley_guard_shapes:
            mu_np[np.asarray(self.pulley_guard_shapes, dtype=np.int32)] = PULLEY_FLANGE_MU
        self.model.shape_material_mu.assign(mu_np)

        # Re-apply per-shape STIFFNESS too.
        ke_np = self.model.shape_material_ke.numpy().copy()
        kd_np = self.model.shape_material_kd.numpy().copy()
        if self.gripper_pad_shapes:
            pad_idx = np.asarray(self.gripper_pad_shapes, dtype=np.int32)
            ke_np[pad_idx] = GRIPPER_CONTACT_KE
            kd_np[pad_idx] = GRIPPER_CONTACT_KD
        pulley_shape_idx = np.asarray(
            self.pulley_sheave_shapes + self.pulley_flange_shapes + self.pulley_guard_shapes,
            dtype=np.int32,
        )
        if pulley_shape_idx.size:
            ke_np[pulley_shape_idx] = PULLEY_CONTACT_KE
            kd_np[pulley_shape_idx] = PULLEY_CONTACT_KD
        self.model.shape_material_ke.assign(ke_np)
        self.model.shape_material_kd.assign(kd_np)

        rinfo = self._configure_robot_joints()
        self.gripper_open_values = rinfo["gripper_open_values"]
        self.gripper_closed_values = rinfo["gripper_closed_values"]
        print(f"[INFO] Robot arm coord indices:      {rinfo['arm_coord_indices']}")
        print(f"[INFO] Gripper driver coord indices: {rinfo['gripper_coord_indices']}")
        print(f"[INFO] Gripper open targets:   {self.gripper_open_values}")
        print(f"[INFO] Gripper closed targets: {self.gripper_closed_values}")

        self.control = self.model.control()
        self.arm_coord_indices = list(rinfo["arm_coord_indices"])
        self.arm_dof_indices = list(rinfo["arm_dof_indices"])
        self.gripper_coord_indices = list(rinfo["gripper_coord_indices"])
        self.gripper_dof_indices = list(rinfo["gripper_dof_indices"])

        self._build_ik()
        self._initialize_robot_at_approach()

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v, solver="newton", integrator="implicitfast", cone="elliptic",
                        iterations=MUJOCO_ITERATIONS, ls_iterations=MUJOCO_LS_ITERATIONS,
                        use_mujoco_contacts=False, njmax=256, nconmax=128),
                    bodies=self.robot_bodies, joints=self.robot_joints, shapes=self.robot_shapes),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v, iterations=VBD_ITERATIONS, rigid_avbd_beta=VBD_RIGID_AVBD_BETA,
                        rigid_contact_k_start=VBD_RIGID_CONTACT_K_START, rigid_contact_history=False,
                        rigid_body_contact_buffer_size=VBD_RIGID_CONTACT_BUFFER_SIZE,
                        particle_enable_self_contact=True,
                        particle_self_contact_radius=BELT_PARTICLE_RADIUS,
                        particle_self_contact_margin=1.5 * BELT_PARTICLE_RADIUS),
                    bodies=self.vbd_bodies, joints=self.vbd_joints, shapes=self.vbd_shapes),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(
                    source="mjc", destination="vbd", bodies=self.robot_proxy_bodies,
                    mass_scale=PROXY_MASS_SCALE, mode=PROXY_COUPLING_MODE,
                    collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                        model, broad_phase="explicit"),
                    collide_interval=1)],
                iterations=PROXY_ITERATIONS)) 

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Cloth contact is particle<->shape, so do not use the old rod-only
        # shape-pair filter.  The coupled proxy separately creates gripper-pad
        # contacts against the VBD particles.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, broad_phase="explicit")
        self.contacts = self.collision_pipeline.contacts()
        if hasattr(self.solver, "prepare_contacts"):
            self.solver.prepare_contacts(self.contacts)

        self._main_view_layer = None
        self._proxy_contact_layer = None
        if hasattr(self.viewer, "activate"):
            self._main_view_layer = "coupled_scene"
            self._proxy_contact_layer = "gripper_belt_contacts"
            self.viewer.activate(self._main_view_layer)

        self.viewer.set_model(self.model)
        newton.examples.configure_coupled_view(self, self.args)
        self.viewer.show_contacts = False

        self.proxy_contacts = (self.solver.get_proxy_contacts("mjc", "vbd")
                               if hasattr(self.solver, "get_proxy_contacts") else None)
        if self.proxy_contacts is not None and self._proxy_contact_layer is not None:
            self.viewer.activate(self._proxy_contact_layer)
            self.viewer.set_model(self.solver.view("vbd"))
            self.viewer.show_visual = False
            self.viewer.show_collision = False
            self.viewer.show_static = False
            self.viewer.show_contacts = False
            self.viewer.activate(self._main_view_layer)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        ctrl_len = len(as_numpy(self.control.joint_target_q))
        n_coords = int(self.model.joint_coord_count)
        n_dofs = int(self.model.joint_dof_count)
        joint_q_np = self.model.joint_q.numpy().astype(np.float32)

        if ctrl_len == n_coords:
            targetq_layout = "coord"
            self.arm_targetq_indices = list(self.arm_coord_indices)
            self.gripper_targetq_indices = list(self.gripper_coord_indices)
            base_targets = joint_q_np.copy()
        elif ctrl_len == n_dofs:
            targetq_layout = "dof"
            self.arm_targetq_indices = list(self.arm_dof_indices)
            self.gripper_targetq_indices = list(self.gripper_dof_indices)
            base_targets = np.zeros(ctrl_len, dtype=np.float32)
            for target_idx, coord_idx in zip(self.arm_targetq_indices, self.arm_coord_indices):
                base_targets[target_idx] = joint_q_np[coord_idx]
            for target_idx, open_val in zip(self.gripper_targetq_indices, self.gripper_open_values):
                base_targets[target_idx] = open_val
        else:
            raise RuntimeError(
                f"control.joint_target_q length {ctrl_len} matches neither "
                f"joint_coord_count ({n_coords}) nor joint_dof_count ({n_dofs}).")

        self.targetq_layout = targetq_layout
        print(f"[INFO] control.joint_target_q layout: {targetq_layout}-space "
              f"(len={ctrl_len}, n_coords={n_coords}, n_dofs={n_dofs}; "
              f"gain arrays are {rinfo['gains_layout']}-space)")

        self.n_robot_targets = ctrl_len
        self.joint_target_q_view = self.control.joint_target_q.reshape((1, self.n_robot_targets))
        self.base_targets_np = base_targets
        self.joint_target_q_view.assign(self.base_targets_np.reshape(1, -1))

        self._ik_arm_coord_indices_wp = wp.array(
            np.asarray(self.ik_arm_coord_indices, dtype=np.int32), dtype=int, device=self.device)
        self._arm_targetq_indices_wp = wp.array(
            np.asarray(self.arm_targetq_indices, dtype=np.int32), dtype=int, device=self.device)
        self._gripper_targetq_indices_wp = wp.array(
            np.asarray(self.gripper_targetq_indices, dtype=np.int32), dtype=int, device=self.device)
        self._desired_gripper_values_np = np.asarray(self.gripper_open_values, dtype=np.float32)
        self._desired_gripper_values_wp = wp.array(
            self._desired_gripper_values_np, dtype=float, device=self.device)

        self.set_task_target(BELT_APPROACH_POS, GRIPPER_DOWN_QUAT, self.gripper_open_values)

        if hasattr(self.viewer, "set_picking_linear_only_bodies"):
            self.viewer.set_picking_linear_only_bodies(self.belt_bodies)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.30, -1.30, 1.30), -22.0, -38.0)

    # IK
    def _build_ik(self) -> None:
        ik_builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        ik_info = _add_ur10_and_gripper_assets(ik_builder, body_offset=0)
        ik_tcp_body = _find_gripper_tcp_body(ik_builder, ik_info["gripper_body_start"], ik_info["gripper_body_end"])

        self.ik_model = ik_builder.finalize(device=self.device)
        self.ik_n_coords = int(self.ik_model.joint_coord_count)

        self.ik_arm_coord_indices, _ = find_n_actuated_indices(self.ik_model, 0, UR10_ARM_DOFS)
        if len(self.ik_arm_coord_indices) != len(self.arm_coord_indices):
            raise RuntimeError("IK/main arm coordinate mismatch: "
                               f"ik={self.ik_arm_coord_indices}, main={self.arm_coord_indices}")

        ik_q_seed = self.ik_model.joint_q.numpy().astype(np.float32).copy()
        main_q = self.model.joint_q.numpy().astype(np.float32)
        for ik_idx, main_idx in zip(self.ik_arm_coord_indices, self.arm_coord_indices):
            ik_q_seed[ik_idx] = main_q[main_idx]
        ik_grip_coord, _ = find_gripper_driver_indices(self.ik_model, 0)
        for ik_idx, open_val in zip(ik_grip_coord, self.gripper_open_values):
            ik_q_seed[ik_idx] = open_val

        self.ik_joint_q = wp.array(ik_q_seed.reshape(1, -1), dtype=float, device=self.device)
        self.ik_target_positions = wp.array([wp.vec3(*BELT_APPROACH_POS)], dtype=wp.vec3, device=self.device)
        self.ik_target_rotations = wp.array([wp.vec4(*GRIPPER_DOWN_QUAT)], dtype=wp.vec4, device=self.device)

        self.ik_pos_obj = ik.IKObjectivePosition(
            link_index=ik_tcp_body, link_offset=wp.vec3(*GRIPPER_TCP_LOCAL_OFFSET),
            target_positions=self.ik_target_positions)
        self.ik_rot_obj = ik.IKObjectiveRotation(
            link_index=ik_tcp_body, link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ik_target_rotations)
        joint_limit_lower = wp.clone(self.ik_model.joint_limit_lower)
        joint_limit_upper = wp.clone(self.ik_model.joint_limit_upper)
        self.ik_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=joint_limit_lower, joint_limit_upper=joint_limit_upper, weight=10.0)

        self.ik_solver = ik.IKSolver(
            model=self.ik_model, n_problems=1,
            objectives=[self.ik_pos_obj, self.ik_rot_obj, self.ik_limit_obj],
            lambda_initial=IK_LAMBDA_INITIAL, jacobian_mode=ik.IKJacobianType.ANALYTIC)

    def _assign_ik_target_arrays(self, position, rotation) -> None:
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        rot = np.asarray(rotation, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(rot))
        if not np.isfinite(pos).all() or not np.isfinite(rot).all() or norm < 1.0e-8:
            raise ValueError(f"Invalid IK target: position={position}, rotation={rotation}")
        rot = rot / norm
        wp.launch(_set_single_task_target, dim=1,
                  inputs=[self.ik_target_positions, self.ik_target_rotations,
                          wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                          wp.vec4(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))],
                  device=self.device)

    def set_task_target(self, position, rotation=GRIPPER_DOWN_QUAT, gripper_values=None) -> None:
        self._assign_ik_target_arrays(position, rotation)
        if gripper_values is not None:
            values = np.asarray(gripper_values, dtype=np.float32).reshape(-1)
            if values.size != len(self.gripper_targetq_indices):
                raise ValueError(f"Expected {len(self.gripper_targetq_indices)} gripper values, got {values.size}")
            self._desired_gripper_values_np = values.copy()
            self._desired_gripper_values_wp.assign(values)

    def _initialize_robot_at_approach(self) -> None:
        self._assign_ik_target_arrays(BELT_APPROACH_POS, GRIPPER_DOWN_QUAT)
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=IK_INIT_ITERS)
        solved = self.ik_joint_q.numpy().reshape(-1)
        if not np.isfinite(solved).all():
            raise RuntimeError("Initial UR10 IK produced NaN/Inf")
        main_q = self.model.joint_q.numpy().copy()
        for main_idx, ik_idx in zip(self.arm_coord_indices, self.ik_arm_coord_indices):
            main_q[main_idx] = solved[ik_idx]
        for main_idx, open_val in zip(self.gripper_coord_indices, self.gripper_open_values):
            main_q[main_idx] = open_val
        self.model.joint_q.assign(main_q)
        self.model.joint_qd.zero_()
        ik_q = self.ik_joint_q.numpy().reshape(-1)
        for ik_idx, main_idx in zip(self.ik_arm_coord_indices, self.arm_coord_indices):
            ik_q[ik_idx] = main_q[main_idx]
        self.ik_joint_q.assign(ik_q.reshape(1, -1))
        print("[INFO] t=0 TCP target above belt vertex: "
              f"approach={BELT_APPROACH_POS}, grasp={BELT_GRASP_POS}, open gripper={self.gripper_open_values}")

    def _solve_and_write_robot_targets(self) -> None:
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=IK_TRACK_ITERS)
        if len(self.arm_targetq_indices) > 0:
            wp.launch(_scatter_ik_arm_targets, dim=len(self.arm_targetq_indices),
                      inputs=[self.control.joint_target_q, self.ik_joint_q,
                              self._ik_arm_coord_indices_wp, self._arm_targetq_indices_wp],
                      device=self.device)
        if len(self.gripper_targetq_indices) > 0:
            wp.launch(_scatter_gripper_targets, dim=len(self.gripper_targetq_indices),
                      inputs=[self.control.joint_target_q, self._gripper_targetq_indices_wp,
                              self._desired_gripper_values_wp],
                      device=self.device)

    # Joint config
    def _configure_robot_joints(self) -> dict[str, Any]:
        model = self.model
        arm_coord_indices, arm_dof_indices = find_n_actuated_indices(
            model, 0, UR10_ARM_DOFS, end_joint_index=self._robot_joint_count)
        gripper_coord_indices, gripper_dof_indices = find_gripper_driver_indices(
            model, 0, end_joint_index=self._robot_joint_count)

        if len(arm_coord_indices) < UR10_ARM_DOFS:
            print(f"[WARNING] Could not find all 6 UR10 arm joints (found {len(arm_coord_indices)}).")
        if len(gripper_coord_indices) == 0:
            print("[WARNING] No gripper driver joint found; gripper will stay static.")

        lower_np = as_numpy(model.joint_limit_lower)
        upper_np = as_numpy(model.joint_limit_upper)
        limit_margin = 0.005
        open_values, closed_values = [], []
        for dof_idx in gripper_dof_indices:
            lo = float(lower_np[dof_idx]); hi = float(upper_np[dof_idx])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                open_q = lo + limit_margin
                safe_close_q = lo + ROBOTIQ_GRIPPER_SAFE_CLOSE_FRACTION * (hi - lo)
                close_q = min(max(safe_close_q, open_q), hi - limit_margin)
                open_values.append(open_q)
                closed_values.append(close_q)
            else:
                open_values.append(0.0)
                closed_values.append(UR10_GRIPPER_CLOSED_FALLBACK * ROBOTIQ_GRIPPER_SAFE_CLOSE_FRACTION)

        joint_q_np = model.joint_q.numpy().copy()
        for coord_idx, val in zip(arm_coord_indices, UR10_ARM_HOME_POSE):
            joint_q_np[coord_idx] = val
        for coord_idx, open_val in zip(gripper_coord_indices, open_values):
            joint_q_np[coord_idx] = open_val
        model.joint_q.assign(joint_q_np)
        model.joint_qd.zero_()

        n_coords = int(model.joint_coord_count)
        n_dofs = int(model.joint_dof_count)
        q_start = as_numpy(model.joint_q_start)
        qd_start = as_numpy(model.joint_qd_start)
        robot_coord_end = int(q_start[self._robot_joint_count]) if self._robot_joint_count < len(q_start) else n_coords
        robot_dof_end = int(qd_start[self._robot_joint_count]) if self._robot_joint_count < len(qd_start) else n_dofs

        mode_np = as_numpy(model.joint_target_mode).copy()
        target_len = len(mode_np)
        if target_len == n_coords:
            arm_t, grip_t = arm_coord_indices, gripper_coord_indices
            robot_end = robot_coord_end; layout = "coord"
        elif target_len == n_dofs:
            arm_t, grip_t = arm_dof_indices, gripper_dof_indices
            robot_end = robot_dof_end; layout = "dof"
        else:
            raise RuntimeError(f"joint_target_mode length {target_len} matches neither "
                               f"n_coords ({n_coords}) nor n_dofs ({n_dofs}).")
        print(f"[INFO] Gain-array layout: {layout}-space (len={target_len})")

        ke_np = as_numpy(model.joint_target_ke).copy()
        kd_np = as_numpy(model.joint_target_kd).copy()
        ke_np[:robot_end] = 0.0
        kd_np[:robot_end] = 0.0
        mode_np[:robot_end] = int(JointTargetMode.NONE)

        # Stiffer arm gains so the TCP still tracks its target while the
        # belt is dragged from the small pulley onto the large one.
        for idx in arm_t:
            ke_np[idx] = 700.0
            kd_np[idx] = 110.0   
            mode_np[idx] = int(JointTargetMode.POSITION)
        # Stiffer finger gains so the closed grip is held firmly and the
        # belt cannot creep out of the pads under load.
        for idx in grip_t:
            ke_np[idx] = 260.0 
            kd_np[idx] = 45.0 
            mode_np[idx] = int(JointTargetMode.POSITION)

        model.joint_target_ke.assign(ke_np)
        model.joint_target_kd.assign(kd_np)
        model.joint_target_mode.assign(mode_np)

        return {"gains_layout": layout,
                "arm_coord_indices": arm_coord_indices, "arm_dof_indices": arm_dof_indices,
                "gripper_coord_indices": gripper_coord_indices, "gripper_dof_indices": gripper_dof_indices,
                "gripper_open_values": open_values, "gripper_closed_values": closed_values}

    # Collision pair filter
    def _belt_world_shape_pairs(self) -> wp.array:
        belt_shapes = set(self.belt_shapes)
        vbd_shapes = set(self.vbd_shapes)
        static_vbd_shapes = vbd_shapes - belt_shapes
        pairs = []
        n_pulley_pairs = 0
        pulley_shapes = set(self.pulley_shapes)
        for a, b in self.model.shape_contact_pairs.numpy():
            a = int(a); b = int(b)
            a_belt = a in belt_shapes; b_belt = b in belt_shapes
            if (a_belt ^ b_belt) and ((a in static_vbd_shapes) or (b in static_vbd_shapes)):
                pairs.append((a, b))
                if a in pulley_shapes or b in pulley_shapes:
                    n_pulley_pairs += 1
        if not pairs:
            raise RuntimeError("No belt-world contact pairs were generated")
        print(f"[INFO] Main collision pipeline: {len(pairs)} belt<->static-world shape pairs "
              f"({n_pulley_pairs} of them belt<->pulley; belt self-contact disabled).")
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.model.device)

    # Control hook
    def solve_gripper_targets(self):
        self.set_task_target(BELT_APPROACH_POS, GRIPPER_DOWN_QUAT, self.gripper_open_values)

    # Sim loop
    def simulate(self):
        self.solve_gripper_targets()
        self._solve_and_write_robot_targets()
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_id += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self._main_view_layer is not None:
            self.viewer.activate(self._main_view_layer)
        show_contacts = self.viewer.show_contacts
        newton.examples.log_coupled_view(self, self.contacts)

        if self.proxy_contacts is not None and self._proxy_contact_layer is not None:
            output_valid = getattr(self.solver, "entry_output_state_valid", None)
            sync_entry_states = getattr(self.solver, "sync_entry_states", None)
            if callable(output_valid) and callable(sync_entry_states) and not output_valid():
                sync_entry_states(self.state_0)
            self.viewer.activate(self._proxy_contact_layer)
            self.viewer.show_contacts = show_contacts
            self.viewer.log_contacts(self.proxy_contacts, self.solver.entry_state("vbd"))
            self.viewer.activate(self._main_view_layer)

        if (self.debug_belt_positions and self.state_0.particle_q is not None
                and self.belt_particles and self.frame_id % self.debug_every_n_frames == 0):
            particle_q = self.state_0.particle_q.numpy()
            belt_xyz = particle_q[np.asarray(self.belt_particles, dtype=np.int32), :3]
            print("[BELT DEBUG] min xyz =", belt_xyz.min(axis=0), "max xyz =", belt_xyz.max(axis=0))
            if len(self.pulley_bodies) > 0 and self.state_0.joint_q is not None:
                q_start = as_numpy(self.model.joint_q_start)
                jq = self.state_0.joint_q.numpy()
                angles = [float(jq[int(q_start[j])]) for j in self.pulley_joints]
                print("[PULLEY DEBUG] axle angles [rad] =", angles)

        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        return parser

    def test_final(self):
        if self.state_0.body_q is not None:
            body_q = self.state_0.body_q.numpy()
            assert np.isfinite(body_q).all(), "Non-finite body transforms"
            belt_xyz = body_q[np.asarray(self.belt_bodies, dtype=np.int32), :3]
            belt_min_z = float(np.min(belt_xyz[:, 2]))
            assert belt_min_z > TABLE_TOP_Z - 0.05, (
                f"Belt fell too far below table: min_z={belt_min_z:.4f}, table_top={TABLE_TOP_Z:.4f}")
            if len(self.pulley_bodies) > 0:
                pulley_xyz = body_q[np.asarray(self.pulley_bodies, dtype=np.int32), :3]
                assert np.all(pulley_xyz[:, 2] > TABLE_TOP_Z - 0.01), (
                    f"A pulley fell off its axle: min_z={float(pulley_xyz[:, 2].min()):.4f}")


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    viewer._pause = False
    newton.examples.run(Example(viewer, args), args)