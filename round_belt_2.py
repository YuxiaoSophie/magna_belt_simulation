"""
Round-belt FULL scene (UR10 + Robotiq 2F-85 + deformable belt + pulleys +
board + table) driven by the SpaceMouse teleop pipeline as the standalone
gripper demo.

Run it together with the producer:
    # terminal A (this file): builds the scene, seeds the shared buffer at the
    #                         live gripper TCP, then chases the target every frame
    # terminal B: reads the SpaceMouse, integrates velocity -> target pose,
    #             constantly overwrites the shared buffer

The producer waits for this file to seed the buffer, so start this one first.
"""

from __future__ import annotations

import os
import math
import mmap
import time
import json
import atexit
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode
from newton.viewer import ViewerFile
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

# Belt dimension
BELT_OUTER_MAJOR_DIAMETER = 0.248
BELT_OUTER_MINOR_DIAMETER = 0.168
BELT_TUBE_DIAMETER = 0.0066
BELT_RADIUS = BELT_TUBE_DIAMETER * 0.5
BELT_CENTER_X = -0.320
BELT_CENTER_Y = 0.000
BELT_CENTER_Z = TABLE_TOP_Z + BELT_RADIUS
BELT_NUM_ELEMENTS = 48  # maximum 87
TOTAL_BELT_MASS = 0.022  # 22 grams

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

# Gripper-pad contact used by the mjc -> vbd proxy coupling.
GRIPPER_CONTACT_KE = 2.0e4
GRIPPER_CONTACT_KD = 20.0
GRIPPER_CONTACT_MU = 8.0

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
PULLEY_GROOVE_HALF_WIDTH = 1.35 * BELT_RADIUS
PULLEY_FLANGE_HALF_THICKNESS = 0.6 * BELT_RADIUS
PULLEY_FLANGE_EXTRA_RADIUS = 4.0 * BELT_RADIUS

# Close the collision gap between each pulley and the board.
PULLEY_BOARD_GAP = 0.0005  # 0.5 mm numerical clearance above the board

PULLEY_SHOW_COLLISION_SHAPES = False

GRIPPER_PROXY_PAD_BODIES = ("left_pad", "right_pad")
GRIPPER_PROXY_FALLBACK_BODIES = ("left_follower", "right_follower")
GRIPPER_PAD_KEYWORDS = ("pad",)

ROBOTIQ_GRIPPER_SAFE_CLOSE_FRACTION = 0.93

# Gripper grasp-safety settings.
GRIPPER_DRIVE_KE = 180.0
GRIPPER_DRIVE_KD = 80.0
GRIPPER_EFFORT_LIMIT = 1.0

# Anti-crush grasp latch.
GRIPPER_STALL_MIN_FRACTION = 0.60
GRIPPER_STALL_ERROR_FRACTION = 0.010
GRIPPER_STALL_SPEED_FRACTION_PER_SEC = 0.15
# Require a real persistent stall instead of latching on one noisy frame.
GRIPPER_STALL_FRAMES = 3

# Contact-critical grasp transport settings.
GRASP_CONTACT_SAFE_FRACTION = 0.45
GRASP_CONTACT_SAFE_UNCAPTURED = True

GRASPED_MAX_ARM_SPEED = 0.75
GRIPPER_HOLD_PRELOAD_FRACTION = 0.004
GRIPPER_RELEASE_HYSTERESIS = 0.020

# Proxy-coupling / solver settings.
PROXY_ITERATIONS = 1
# Make the robot pad proxies dynamically dominant over the 22 g belt.
PROXY_MASS_SCALE = 10.0
PROXY_COUPLING_MODE = "lagged"
VBD_ITERATIONS = 20
VBD_RIGID_AVBD_BETA = 1.0e2
VBD_RIGID_CONTACT_K_START = 3.0e3
VBD_RIGID_CONTACT_BUFFER_SIZE = 256
MUJOCO_ITERATIONS = 50  # restored from stable pre-CUDA version; important for gripper/proxy convergence
MUJOCO_LS_ITERATIONS = 20  # restored from stable pre-CUDA version

# Shared memory target buffer (identical transport to the standalone demo).
SHARED_PATH_DEFAULT = "/tmp/sm_teleop_target.bin"


# Shared-memory target buffer
class SharedTarget:
    N = 10
    SIZE = N * 8

    def __init__(self, path=SHARED_PATH_DEFAULT):
        self.path = path
        if (not os.path.exists(path)) or os.path.getsize(path) != self.SIZE:
            with open(path, "wb") as f:
                f.write(b"\x00" * self.SIZE)
        self.f = open(path, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), self.SIZE)
        self.arr = np.ndarray((self.N,), dtype=np.float64, buffer=self.mm)
        self._last = None

    def write(self, pos, quat, grip, ready=1.0):
        seq = float(self.arr[0])
        self.arr[0] = seq + 1.0
        self.arr[1:4] = pos
        self.arr[4:8] = quat
        self.arr[8] = float(grip)
        self.arr[9] = float(ready)
        self.arr[0] = seq + 2.0

    def read(self):
        for _ in range(16):
            s1 = float(self.arr[0])
            if int(s1) & 1:
                continue
            pos = np.array(self.arr[1:4], dtype=np.float64)
            quat = np.array(self.arr[4:8], dtype=np.float64)
            grip = float(self.arr[8])
            ready = float(self.arr[9])
            if float(self.arr[0]) == s1:
                self._last = (pos, quat, grip, ready)
                return pos, quat, grip, ready
        if self._last is not None:
            return self._last
        return (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 0.0, 0.0)

    def is_ready(self):
        return float(self.arr[9]) >= 0.5

    def close(self):
        try:
            self.mm.close()
            self.f.close()
        except Exception:
            pass


# quaternion / vector helpers
def _norm4(q):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _quat_conj(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quat_to_rotvec(q):
    """Shortest rotation vector for an xyzw quaternion."""
    q = _norm4(q)
    if q[3] < 0.0:
        q = -q
    v = q[:3]
    s = float(np.linalg.norm(v))
    if s < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(s, float(q[3]))
    return v * (angle / s)


def _rotate_vec(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], dtype=np.float64)


def _v3(a):
    return wp.vec3(float(a[0]), float(a[1]), float(a[2]))


def _v4(a):
    return wp.vec4(float(a[0]), float(a[1]), float(a[2]), float(a[3]))


def parse_vec3(text, default=(0.0, 0.0, 0.0)):
    if text is None or text.strip() == "":
        return np.array(default, dtype=np.float64)
    parts = [float(v.strip()) for v in text.split(",") if v.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 floats, got {text!r}")
    return np.array(parts, dtype=np.float64)


def _unit_sphere_wire(n_lat=2, seg=28):
    """Unit-radius wireframe sphere as (starts, ends) numpy arrays (M,3)."""
    starts, ends = [], []

    def ring(pts):
        for k in range(len(pts) - 1):
            starts.append(pts[k])
            ends.append(pts[k + 1])

    for pl in ("xy", "yz", "xz"):
        pts = []
        for k in range(seg + 1):
            a = 2.0 * math.pi * k / seg
            if pl == "xy":
                pts.append([math.cos(a), math.sin(a), 0.0])
            elif pl == "yz":
                pts.append([0.0, math.cos(a), math.sin(a)])
            else:
                pts.append([math.cos(a), 0.0, math.sin(a)])
        ring(pts)
    for i in range(1, n_lat + 1):
        z = i / (n_lat + 1)
        for zz in (z, -z):
            r = math.sqrt(max(0.0, 1.0 - zz * zz))
            pts = [[r * math.cos(2.0 * math.pi * k / seg),
                    r * math.sin(2.0 * math.pi * k / seg), zz] for k in range(seg + 1)]
            ring(pts)
    return np.array(starts, dtype=np.float64), np.array(ends, dtype=np.float64)


# Geometry helpers 
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
BELT_PLACE_DOWN = (_place_x, _place_y, LARGE_PULLEY_WORLD_CENTER[2] + BELT_RADIUS)

# Small-pulley placement geometry (the pulley the belt seats on FIRST).
SMALL_PULLEY_CENTER_LOCAL = (0.3504 - 0.01200845, 0.1964 - 0.0004, 0.0248)
SMALL_PULLEY_WORLD_CENTER = board_world(SMALL_PULLEY_CENTER_LOCAL)

SMALL_PULLEY_PLACE_EDGE_OFFSET = (SMALL_PULLEY_SHEAVE_RADIUS + BELT_RADIUS, 0.0, 0.0)
_small_place_x = SMALL_PULLEY_WORLD_CENTER[0] + SMALL_PULLEY_PLACE_EDGE_OFFSET[0]
_small_place_y = SMALL_PULLEY_WORLD_CENTER[1] + SMALL_PULLEY_PLACE_EDGE_OFFSET[1]

BELT_PLACE_SMALL_ABOVE = (_small_place_x, _small_place_y, SMALL_PULLEY_WORLD_CENTER[2] + BELT_APPROACH_CLEARANCE)
BELT_PLACE_SMALL_DOWN = (_small_place_x, _small_place_y, SMALL_PULLEY_WORLD_CENTER[2] + BELT_RADIUS)

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


def create_ellipse_cable_geometry(pos: wp.vec3, num_elements=48, twisting_angle=0.0):
    num_points = num_elements + 1
    points = []
    a = 0.248 / 2.0
    b = 0.168 / 2.0
    for i in range(num_points):
        theta = 2.0 * np.pi * i / num_elements
        points.append(pos + wp.vec3(a * np.cos(theta), b * np.sin(theta), 0.0))
    edge_q = newton.utils.create_parallel_transport_cable_quaternions(points, twist_total=float(twisting_angle))
    return points, edge_q


def _estimate_belt_density() -> float:
    a = 0.5 * BELT_OUTER_MAJOR_DIAMETER
    b = 0.5 * BELT_OUTER_MINOR_DIAMETER
    h = ((a - b) ** 2) / ((a + b) ** 2)
    length = math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h)))
    volume = math.pi * BELT_RADIUS * BELT_RADIUS * length
    return float(TOTAL_BELT_MASS / max(volume, 1.0e-12))


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


# Dataset recording / replay support only
class TeleopEpisodeRecorder:
    """Passively save the initial state, actions, and Newton state snapshots."""

    def __init__(self, root_dir: Path, model, state, control, example):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.episode_dir = Path(root_dir).expanduser() / f"episode_{stamp}"
        suffix = 1
        while self.episode_dir.exists():
            self.episode_dir = Path(root_dir).expanduser() / f"episode_{stamp}_{suffix:02d}"
            suffix += 1
        self.episode_dir.mkdir(parents=True, exist_ok=False)

        self.actions_file = (self.episode_dir / "actions.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self.state_file = self.episode_dir / "newton_states.bin"
        self.initial_file = self.episode_dir / "initial_state.npz"

        # Save the exact initial simulation arrays plus the current control/teleop state.
        initial = {}
        for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
            value = getattr(state, name, None)
            if value is not None:
                initial[name] = value.numpy().copy()
        initial["control_joint_target_q"] = control.joint_target_q.numpy().copy()
        initial["target_pos"] = np.asarray(example._target_pos, dtype=np.float64).copy()
        initial["target_xyzw"] = np.asarray(example._target_xyzw, dtype=np.float64).copy()
        initial["grip_fraction"] = np.asarray([example._grip_fraction], dtype=np.float64)
        initial["grip_requested_fraction"] = np.asarray([example._grip_requested_fraction], dtype=np.float64)
        initial["arm_cmd"] = np.asarray(example._arm_cmd, dtype=np.float64).copy()
        np.savez_compressed(self.initial_file, **initial)

        metadata = {
            "fps": int(example.fps),
            "frame_dt": float(example.frame_dt),
            "sim_substeps": int(example.sim_substeps),
            "sim_dt": float(example.sim_dt),
            "targetq_layout": str(example.targetq_layout),
            "arm_target_indices": [int(i) for i in example.arm_target_indices],
            "gripper_target_indices": [int(i) for i in example.gripper_target_indices],
        }
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        # Newton-native recording. 
        self.viewer_file = ViewerFile(str(self.state_file), auto_save=True, save_interval=1)
        self.viewer_file.set_model(model)
        self.viewer_file.record(state)
        self.viewer_file.save_recording(verbose=False)
        self.closed = False
        atexit.register(self.close)

        print(f"[RECORD] recording episode to: {self.episode_dir}")
        print(f"[RECORD] Newton state file: {self.state_file}")

    def record_frame(self, frame_id, sim_time, target_pos, target_xyzw,
                     grip_fraction, grip_requested_fraction, joint_target_q, state):
        row = {
            "frame": int(frame_id),
            "sim_time": float(sim_time),
            "target_pos": np.asarray(target_pos, dtype=np.float64).tolist(),
            "target_xyzw": np.asarray(target_xyzw, dtype=np.float64).tolist(),
            "grip_fraction": float(grip_fraction),
            "grip_requested_fraction": float(grip_requested_fraction),
            "joint_target_q": np.asarray(joint_target_q, dtype=np.float64).reshape(-1).tolist(),
        }
        self.actions_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.viewer_file.record(state)

    def close(self):
        if getattr(self, "closed", True):
            return
        self.closed = True
        try:
            self.actions_file.flush()
            self.actions_file.close()
        finally:
            try:
                self.viewer_file.save_recording(verbose=False)
            finally:
                self.viewer_file.close()
        print(f"[RECORD] saved episode: {self.episode_dir}")



class Example:
    """Full round-belt scene, but the arm TCP now chases the SpaceMouse target
    written into shared memory -- using the exact control/IK logic of the
    standalone teleop demo. Arm starts directly above the belt grasp vertex,
    gripper OPEN, and the shared buffer is seeded there so the target begins
    exactly on the gripper tip."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.frame_id = 0
        
        # Real-time diagnostics
        # NOTE: wall_start_time is only a placeholder here; it is re-stamped on the
        # first real step() so construction cost (asset download + finalize + init
        # IK) does not bias the real-time factor low.
        self.wall_start_time = time.perf_counter()
        self.last_timing_print = self.wall_start_time
        self._last_sim_time = self.sim_time
        self._last_wall_time = self.wall_start_time

        # Optional synchronized step profiler. Profiling is opt-in because the
        # required CUDA synchronization perturbs throughput.
        self.profile_step = bool(getattr(args, "profile_step", False))
        self._prof_frames = 0
        self._prof_ik_s = 0.0
        self._prof_phys_s = 0.0
        self._prof_collide_s = 0.0
        self._prof_solve_s = 0.0
        
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

        belt_body_start = builder.body_count
        belt_joint_start = builder.joint_count
        belt_shape_start = builder.shape_count

        start_pos = wp.vec3(BELT_CENTER_X, BELT_CENTER_Y, BELT_CENTER_Z)
        cable_points, cable_edge_q = create_ellipse_cable_geometry(
            pos=start_pos, num_elements=BELT_NUM_ELEMENTS, twisting_angle=0.0)

        belt_cfg = newton.ModelBuilder.ShapeConfig(
            density=_estimate_belt_density(), ke=CABLE_CONTACT_KE, kd=CABLE_CONTACT_KD,
            mu=CABLE_CONTACT_MU, margin=0.0, gap=0.01)
        rod_bodies, _rod_joints = builder.add_rod(
            positions=cable_points, quaternions=cable_edge_q, radius=BELT_RADIUS, cfg=belt_cfg,
            stretch_stiffness=2.0e4, stretch_damping=1.0e-1,
            bend_stiffness=1.5e-1, bend_damping=1.0e-1,
            closed=True, body_frame_origin="com", label="flexible_ellipse_cable")
        self.belt_bodies = list(rod_bodies)
        self.belt_joints = list(range(belt_joint_start, builder.joint_count))
        self.belt_shapes = list(range(belt_shape_start, builder.shape_count))
        assert self.belt_bodies == list(range(belt_body_start, builder.body_count))

        builder.add_ground_plane()

        self.vbd_shapes = [s for s in range(builder.shape_count) if s >= robot_shape_end]
        self.vbd_bodies = sorted(self.pulley_bodies + self.belt_bodies)
        self.vbd_joints = sorted(self.pulley_joints + self.belt_joints)

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

        # Re-apply per-shape stiffness too.
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
                        rigid_body_contact_buffer_size=VBD_RIGID_CONTACT_BUFFER_SIZE),
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

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, broad_phase="explicit", shape_pairs_filtered=self._belt_world_shape_pairs())
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

        # TELEOP STATE 
        self.tip_offset = np.asarray(GRIPPER_TCP_LOCAL_OFFSET, dtype=np.float64)
        self.link_index = int(self._robot_tcp_body_idx)
        self.g_open = list(self.gripper_open_values)
        self.g_closed = list(self.gripper_closed_values)
        self.arm_target_indices = list(self.arm_targetq_indices)
        self.gripper_target_indices = list(self.gripper_targetq_indices)

        # Seed the shared target at the ACTUAL current TCP pose + OPEN gripper.
        bq = self.state_0.body_q.numpy()
        base = bq[self.link_index]
        base_pos = np.array(base[0:3], dtype=np.float64)
        base_quat = _norm4(np.array(base[3:7], dtype=np.float64))
        tip0 = base_pos + _rotate_vec(base_quat, self.tip_offset)

        self._target_pos = tip0.copy()
        self._target_xyzw = base_quat.copy()
        self._grip_fraction = 0.0 # applied 0=open, 1=closed
        self._grip_requested_fraction = 0.0
        self._grip_hold_fraction = None
        self._grip_stall_frames = 0
        self._grasp_stabilize_frames_remaining = 0
        self._grip_actual_fraction = 0.0
        self._grip_actual_speed = 0.0

        # arm command we follow (used for slew limiting + posture bias).
        main_q = self.model.joint_q.numpy()
        self._arm_cmd = np.array(
            [float(main_q[ci]) for ci in self.arm_coord_indices], dtype=np.float64
        )

        self.shared = SharedTarget(self.args.buffer)
        self.shared.write(self._target_pos, self._target_xyzw, 0.0, ready=1.0)
        print(f"[INFO] seeded shared target buffer: {self.args.buffer}")

        # Frame + sphere visuals (verbatim from the standalone demo).
        self.axis_len = float(self.args.axis_len)
        self.tip_radius = float(self.args.tip_radius)
        self._axis_bright = wp.array(
            [wp.vec3(1.0, 0.15, 0.15), wp.vec3(0.15, 1.0, 0.15), wp.vec3(0.2, 0.4, 1.0)],
            dtype=wp.vec3, device=self.device)
        self._axis_dim = wp.array(
            [wp.vec3(0.55, 0.1, 0.1), wp.vec3(0.1, 0.55, 0.1), wp.vec3(0.12, 0.2, 0.6)],
            dtype=wp.vec3, device=self.device)
        self._sph_s, self._sph_e = _unit_sphere_wire(n_lat=2, seg=int(self.args.sphere_seg))
        sc = parse_vec3(self.args.sphere_color, default=(0.3, 0.85, 0.95))
        self._sph_colors = wp.array([_v3(sc)] * len(self._sph_s), dtype=wp.vec3, device=self.device)
        self._last_dbg = 0.0

        if hasattr(self.viewer, "set_picking_linear_only_bodies"):
            self.viewer.set_picking_linear_only_bodies(self.belt_bodies)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.30, -1.30, 1.30), -22.0, -38.0)

        # Optional dataset I/O only. 
        self.episode_recorder = None
        self._replay_actions = None
        self._replay_index = 0
        self._replay_done = False

        if getattr(self.args, "replay_episode", None):
            self._load_replay_episode(Path(self.args.replay_episode))
        elif getattr(self.args, "record_episode", False):
            self.episode_recorder = TeleopEpisodeRecorder(
                Path(self.args.record_dir), self.model, self.state_0, self.control, self
            )

        # CUDA graph: capture only the repeated coupled-physics work.
        self.physics_graph = None
        self.use_cuda_graph = bool(getattr(self.args, "cuda_graph", True)) and self.device.is_cuda
        if self.profile_step and self.use_cuda_graph:
            self.use_cuda_graph = False
            print("[PROFILE] --profile-step enabled: detailed collide/solve timing uses "
                  "the uncaptured physics loop; solver/contact settings are unchanged.")
        if self.use_cuda_graph:
            self._capture_physics_graph()
        else:
            print(f"[CUDA GRAPH] disabled (device={self.device})")

    def _simulate_physics(self) -> None:
        """One full rendered frame of the coupled physics.
        """
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)

            # Keep the exact original collision call/path and filtered pair set.
            self.model.collide(
                self.state_0, self.contacts, collision_pipeline=self.collision_pipeline
            )

            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            newton.eval_ik(
                self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _simulate_physics_profiled(self) -> tuple[float, float, float]:
        """Run unchanged physics with synchronized timing for --profile-step."""
        collide_s = 0.0
        solve_s = 0.0
        wp.synchronize_device(self.device)
        phys_start = time.perf_counter()

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)

            wp.synchronize_device(self.device)
            t0 = time.perf_counter()
            self.model.collide(
                self.state_0, self.contacts, collision_pipeline=self.collision_pipeline
            )
            wp.synchronize_device(self.device)
            collide_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            newton.eval_ik(
                self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd
            )
            wp.synchronize_device(self.device)
            solve_s += time.perf_counter() - t0

            self.state_0, self.state_1 = self.state_1, self.state_0

        phys_s = time.perf_counter() - phys_start
        return phys_s, collide_s, solve_s

    def _capture_physics_graph(self) -> None:
        """Capture the fixed GPU physics launch sequence once, then replay it."""
        saved_state_0 = self.state_0
        saved_state_1 = self.state_1
        try:
            wp.synchronize_device(self.device)
            with wp.ScopedDevice(self.device):
                with wp.ScopedCapture() as capture:
                    self._simulate_physics()
            if capture.graph is None:
                raise RuntimeError("Warp returned no CUDA graph")
            self.physics_graph = capture.graph
            print(
                f"[CUDA GRAPH] captured coupled physics: "
                f"{self.sim_substeps} substeps/frame, dt={self.sim_dt:.9f} s; "
                "contact/solver settings unchanged"
            )
        except Exception as exc:
            # Safe fallback: preserving the original simulation is more important
            # than forcing graph mode on a driver/solver combination that cannot
            # be captured.
            self.physics_graph = None
            self.use_cuda_graph = False
            print(f"[CUDA GRAPH] capture failed; using original uncaptured loop: {exc}")
        finally:
            self.state_0 = saved_state_0
            self.state_1 = saved_state_1

    def _grasp_contact_critical(self) -> bool:
        """Use the stable uncaptured path only while the pinch is being formed.
        """
        forming_grasp = (
            self._grip_hold_fraction is None
            and (
                self._grip_fraction >= GRASP_CONTACT_SAFE_FRACTION
                or self._grip_requested_fraction >= GRASP_CONTACT_SAFE_FRACTION
            )
        )
        settling_grasp = self._grasp_stabilize_frames_remaining > 0
        return forming_grasp or settling_grasp

    def _launch_physics(self) -> None:
        # Hybrid fast/stable execution:
        #   1) free motion                      -> CUDA graph
        #   2) fingers entering belt contact    -> uncaptured stable physics
        #   3) first few frames after latch     -> uncaptured stable physics
        #   4) established grasp / transport    -> CUDA graph again
        if GRASP_CONTACT_SAFE_UNCAPTURED and self._grasp_contact_critical():
            self._simulate_physics()
            if self._grip_hold_fraction is not None and self._grasp_stabilize_frames_remaining > 0:
                self._grasp_stabilize_frames_remaining -= 1
                if self._grasp_stabilize_frames_remaining == 0:
                    print("[GRASP] stabilized; resuming CUDA graph for fast transport.")
            return

        if self.physics_graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.physics_graph)
        else:
            self._simulate_physics()

    def _load_replay_episode(self, episode_dir: Path) -> None:
        episode_dir = Path(episode_dir).expanduser()
        initial_path = episode_dir / "initial_state.npz"
        actions_path = episode_dir / "actions.jsonl"
        metadata_path = episode_dir / "metadata.json"

        if not initial_path.exists():
            raise FileNotFoundError(f"Replay episode missing {initial_path}")
        if not actions_path.exists():
            raise FileNotFoundError(f"Replay episode missing {actions_path}")

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key, current in (("fps", self.fps), ("sim_substeps", self.sim_substeps)):
                if key in metadata and int(metadata[key]) != int(current):
                    raise RuntimeError(
                        f"Replay metadata mismatch for {key}: recorded={metadata[key]}, current={current}"
                    )
            if "sim_dt" in metadata and not np.isclose(float(metadata["sim_dt"]), float(self.sim_dt)):
                raise RuntimeError(
                    f"Replay metadata mismatch for sim_dt: recorded={metadata['sim_dt']}, current={self.sim_dt}"
                )

        data = np.load(initial_path, allow_pickle=False)
        for state in (self.state_0, self.state_1):
            for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
                if name in data.files:
                    value = getattr(state, name, None)
                    if value is not None:
                        value.assign(data[name])

        if "control_joint_target_q" in data.files:
            self.control.joint_target_q.assign(data["control_joint_target_q"])
        if "target_pos" in data.files:
            self._target_pos = np.asarray(data["target_pos"], dtype=np.float64).copy()
        if "target_xyzw" in data.files:
            self._target_xyzw = _norm4(np.asarray(data["target_xyzw"], dtype=np.float64))
        if "grip_fraction" in data.files:
            self._grip_fraction = float(np.asarray(data["grip_fraction"]).reshape(-1)[0])
        if "grip_requested_fraction" in data.files:
            self._grip_requested_fraction = float(np.asarray(data["grip_requested_fraction"]).reshape(-1)[0])
        else:
            self._grip_requested_fraction = self._grip_fraction
        if "arm_cmd" in data.files:
            self._arm_cmd = np.asarray(data["arm_cmd"], dtype=np.float64).copy()

        # Reset anti-crush latch internals to their normal t=0 values.
        self._grip_hold_fraction = None
        self._grip_stall_frames = 0
        self._grasp_stabilize_frames_remaining = 0
        self._grip_actual_fraction = 0.0
        self._grip_actual_speed = 0.0

        actions = []
        with actions_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "joint_target_q" not in row:
                    raise RuntimeError(f"{actions_path}:{lineno} has no joint_target_q")
                actions.append(row)
        if not actions:
            raise RuntimeError(f"Replay episode has no actions: {actions_path}")

        expected = self.control.joint_target_q.numpy().size
        got = len(actions[0]["joint_target_q"])
        if got != expected:
            raise RuntimeError(
                f"Replay action size mismatch: recorded joint_target_q has {got}, current control has {expected}"
            )

        self._replay_actions = actions
        self._replay_index = 0
        self._replay_done = False
        self.sim_time = 0.0
        self.frame_id = 0
        print(f"[REPLAY] loaded {len(actions)} actions from: {episode_dir}")
        if not (episode_dir / "newton_states.bin").exists():
            print("[REPLAY] newton_states.bin is absent; replaying from initial_state.npz + actions.jsonl.")

    def _step_replay(self) -> None:
        if self._replay_done:
            return
        if self._replay_index >= len(self._replay_actions):
            self._replay_done = True
            print(f"[REPLAY] finished {len(self._replay_actions)} frames.")
            if hasattr(self.viewer, "_pause"):
                self.viewer._pause = True
            return

        row = self._replay_actions[self._replay_index]

        if "target_pos" in row:
            self._target_pos = np.asarray(row["target_pos"], dtype=np.float64)
        if "target_xyzw" in row:
            self._target_xyzw = _norm4(np.asarray(row["target_xyzw"], dtype=np.float64))
        if "grip_fraction" in row:
            self._grip_fraction = float(row["grip_fraction"])
        if "grip_requested_fraction" in row:
            self._grip_requested_fraction = float(row["grip_requested_fraction"])
        else:
            self._grip_requested_fraction = self._grip_fraction

        # Replay the exact robot action generated during recording.
        self.control.joint_target_q.assign(
            np.asarray(row["joint_target_q"], dtype=np.float32)
        )
        
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self._replay_index += 1
        self.sim_time += self.frame_dt
        self.frame_id += 1

    # IK model
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

        # teleop aliases + TCP body + null-space FK workspace
        self.ik_tcp_body = int(ik_tcp_body)
        self.pos_obj = self.ik_pos_obj
        self.rot_obj = self.ik_rot_obj
        self.lim_obj = self.ik_limit_obj

        ns_seed = ik_q_seed.copy()
        self._ns_joint_q = wp.array(ns_seed, dtype=float, device=self.device)
        self._ns_joint_qd = wp.zeros(int(self.ik_model.joint_dof_count), dtype=float, device=self.device)
        self._ns_state = self.ik_model.state()

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

    def _initialize_robot_at_approach(self) -> None:
        # One high-iteration solve at t=0 (same role as the demo's init solve):
        # the arm settles above the belt vertex so the seeded target lands on it.
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

    # Null-space posture bias + slew-limited target write 
    @staticmethod
    def _nearest_equivalent_angle(angle, reference):
        """Return angle + 2*pi*k that is closest to reference."""
        return reference + ((angle - reference + np.pi) % (2.0 * np.pi) - np.pi)

    def _ik_tcp_pose_for_arm(self, arm_q):
        """Evaluate the IK model TCP pose for one 6-joint arm configuration."""
        q_all = self._ns_joint_q.numpy()

        for ik_qi, qj in zip(self.ik_arm_coord_indices, arm_q):
            q_all[ik_qi] = float(qj)

        self._ns_joint_q.assign(q_all)
        self._ns_joint_qd.zero_()

        newton.eval_fk(
            self.ik_model,
            self._ns_joint_q,
            self._ns_joint_qd,
            self._ns_state,
        )

        body = self._ns_state.body_q.numpy()[self.ik_tcp_body]
        base_pos = np.asarray(body[0:3], dtype=np.float64)
        base_quat = _norm4(np.asarray(body[3:7], dtype=np.float64))

        tcp_pos = base_pos + _rotate_vec(base_quat, self.tip_offset)
        return tcp_pos, base_quat

    def _numeric_task_jacobian(self, arm_q):
        """
        Numerical 6x6 TCP Jacobian around arm_q.
        rows 0:3 -> TCP translation
        rows 3:6 -> TCP world-frame rotation vector
        Used only for the secondary null-space projector; Newton's own analytic
        Jacobian still performs the primary IK solve.
        """
        arm_q = np.asarray(arm_q, dtype=np.float64)
        eps = float(self.args.nullspace_eps)

        p0, r0 = self._ik_tcp_pose_for_arm(arm_q)
        J = np.zeros((6, len(arm_q)), dtype=np.float64)

        for j in range(len(arm_q)):
            q1 = arm_q.copy()
            q1[j] += eps

            p1, r1 = self._ik_tcp_pose_for_arm(q1)

            J[0:3, j] = (p1 - p0) / eps

            dq = _norm4(_quat_mul(r1, _quat_conj(r0)))
            J[3:6, j] = _quat_to_rotvec(dq) / eps

        return J

    def _nullspace_posture_seed(self, solved_q):
        """
        Build a posture-biased IK seed without changing the final TCP command.
            1. Newton IK solves the primary TCP task.
            2. Compute the local TCP Jacobian J.
            3. Find the TRUE numerical null space of J with SVD.
            4. Project a "stay near previous posture" correction into it.
            5. Use that corrected posture only as the seed for a second IK solve.
            6. The second IK solve restores the exact TCP position/orientation.
        """
        q_ik = np.array(
            [float(solved_q[i]) for i in self.ik_arm_coord_indices],
            dtype=np.float64,
        )

        for j in range(len(q_ik)):
            q_ik[j] = self._nearest_equivalent_angle(q_ik[j], self._arm_cmd[j])

        J = self._numeric_task_jacobian(q_ik)

        U, S, Vt = np.linalg.svd(J, full_matrices=True)

        sigma_max = float(S[0]) if len(S) else 0.0
        tol = max(
            float(self.args.nullspace_svd_abs_tol),
            float(self.args.nullspace_svd_rel_tol) * sigma_max,
        )

        rank = int(np.sum(S > tol))

        if rank >= len(q_ik):
            return q_ik

        V = Vt.T
        Z = V[:, rank:]
        N = Z @ Z.T

        posture_error = self._arm_cmd - q_ik

        dq_null = float(self.args.nullspace_gain) * (N @ posture_error)

        max_null_step = float(self.args.max_nullspace_step)
        dq_null = np.clip(dq_null, -max_null_step, max_null_step)

        return q_ik + dq_null

    def _measure_actual_gripper(self):
        """Return actual Robotiq closure fraction and closure speed from the solved state."""
        if not self.gripper_coord_indices:
            return 0.0, 0.0

        jq = self.state_0.joint_q.numpy()
        jqd = self.state_0.joint_qd.numpy()
        fractions = []
        speeds = []

        for coord_idx, dof_idx, q_open, q_closed in zip(
                self.gripper_coord_indices, self.gripper_dof_indices, self.g_open, self.g_closed):
            span = float(q_closed) - float(q_open)
            if abs(span) < 1.0e-9:
                continue
            fractions.append((float(jq[coord_idx]) - float(q_open)) / span)
            speeds.append(float(jqd[dof_idx]) / span)

        if not fractions:
            return 0.0, 0.0

        frac = float(np.clip(np.median(fractions), 0.0, 1.0))
        speed = float(np.median(speeds)) if speeds else 0.0
        return frac, speed

    def _update_gripper_antcrush(self):
        """Convert the raw SpaceMouse close request into a load-limited grasp command.
        """
        requested = float(np.clip(self._grip_requested_fraction, 0.0, 1.0))
        actual, actual_speed = self._measure_actual_gripper()
        self._grip_actual_fraction = actual
        self._grip_actual_speed = actual_speed

        # If a grasp has already been latched, keep only the small preload.
        # The user must command OPEN below the latch point to release it.
        if self._grip_hold_fraction is not None:
            if requested < self._grip_hold_fraction - GRIPPER_RELEASE_HYSTERESIS:
                print(f"[GRASP] release latch: requested={requested:.3f}, actual={actual:.3f}")
                self._grip_hold_fraction = None
                self._grip_stall_frames = 0
                self._grasp_stabilize_frames_remaining = 0
                self._grip_fraction = requested
            else:
                self._grip_fraction = min(requested, self._grip_hold_fraction)
            return

        # Detect a closing stall.
        error = requested - actual
        closing_request = requested > self._grip_fraction + 1.0e-5
        stalled = (
            closing_request
            and actual >= GRIPPER_STALL_MIN_FRACTION
            and error >= GRIPPER_STALL_ERROR_FRACTION
            and actual_speed <= GRIPPER_STALL_SPEED_FRACTION_PER_SEC
        )

        if stalled:
            self._grip_stall_frames += 1
        else:
            self._grip_stall_frames = 0

        if self._grip_stall_frames >= GRIPPER_STALL_FRAMES:
            hold = min(requested, actual + GRIPPER_HOLD_PRELOAD_FRACTION)
            self._grip_hold_fraction = float(np.clip(hold, 0.0, 1.0))
            self._grip_fraction = self._grip_hold_fraction
            self._grasp_stabilize_frames_remaining = GRASP_STABILIZE_FRAMES
            print(
                f"[GRASP] anti-crush latch: requested={requested:.3f}, actual={actual:.3f}, "
                f"hold={self._grip_hold_fraction:.3f}, speed={actual_speed:.3f}/s"
            )
        else:
            self._grip_fraction = requested

    def _write_control_targets(self, solved_q):
        """Write the final primary-task IK solution to the robot (slew-limited)."""
        targets = self.control.joint_target_q.numpy().copy()

        desired = np.array(
            [float(solved_q[i]) for i in self.ik_arm_coord_indices],
            dtype=np.float64,
        )

        for j in range(len(desired)):
            desired[j] = self._nearest_equivalent_angle(desired[j], self._arm_cmd[j])

        # Once the belt is actually latched, avoid injecting a sudden tangential
        # impulse/torque through one pad that can roll a round belt out of the pinch.
        arm_speed_limit = float(self.args.max_arm_speed)
        if self._grip_hold_fraction is not None:
            arm_speed_limit = min(arm_speed_limit, GRASPED_MAX_ARM_SPEED)
        max_step = arm_speed_limit * self.frame_dt

        delta = np.clip(desired - self._arm_cmd, -max_step, max_step)

        self._arm_cmd = self._arm_cmd + delta

        for ti, cmd in zip(self.arm_target_indices, self._arm_cmd):
            targets[ti] = float(cmd)

        frac = float(np.clip(self._grip_fraction, 0.0, 1.0))
        grip_vals = (
            np.asarray(self.g_open, dtype=np.float64)
            + frac * (
                np.asarray(self.g_closed, dtype=np.float64)
                - np.asarray(self.g_open, dtype=np.float64)
            )
        )

        for ti, val in zip(self.gripper_target_indices, grip_vals):
            targets[ti] = float(val)

        self.control.joint_target_q.assign(targets)

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

        for idx in arm_t:
            ke_np[idx] = 700.0
            kd_np[idx] = 110.0
            mode_np[idx] = int(JointTargetMode.POSITION)
        for idx in grip_t:
            ke_np[idx] = GRIPPER_DRIVE_KE
            kd_np[idx] = GRIPPER_DRIVE_KD
            mode_np[idx] = int(JointTargetMode.POSITION)

        model.joint_target_ke.assign(ke_np)
        model.joint_target_kd.assign(kd_np)
        model.joint_target_mode.assign(mode_np)

        # Force/effort safety for grasping the deformable belt.
        # The SpaceMouse may continue commanding frac=1.0, but the actuator is
        # not allowed to build unbounded squeeze effort against the trapped belt.
        effort_np = as_numpy(model.joint_effort_limit).copy()
        for dof_idx in gripper_dof_indices:
            effort_np[dof_idx] = GRIPPER_EFFORT_LIMIT
        model.joint_effort_limit.assign(effort_np)
        print(f"[INFO] Gripper compliant drive: ke={GRIPPER_DRIVE_KE}, kd={GRIPPER_DRIVE_KD}, "
              f"effort_limit={GRIPPER_EFFORT_LIMIT}")

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

    # Sim loop: teleop control + coupled physics stepping
    def step(self):
        if self._replay_actions is not None:
            self._step_replay()
            return

        # Charge the real-time clock from the first real frame, not from
        # construction (asset download + finalize + init IK would bias RTF low).
        if self.frame_id == 0:
            self.wall_start_time = time.perf_counter()
            self.last_timing_print = self.wall_start_time
            self._last_sim_time = self.sim_time
            self._last_wall_time = self.wall_start_time

        # 1) Read the SpaceMouse target. Neutral input leaves the pose unchanged.
        pos, quat, grip, ready = self.shared.read()
        if ready >= 0.5:
            pos = np.asarray(pos, dtype=np.float64)
            quat = _norm4(quat)
            if np.isfinite(pos).all() and np.isfinite(quat).all():
                self._target_pos = pos.copy()
                self._target_xyzw = quat.copy()
            self._grip_requested_fraction = float(np.clip(grip, 0.0, 1.0))

        # Convert the raw SpaceMouse command into an anti-crush applied command.
        self._update_gripper_antcrush()

        # 2) Solve the PRIMARY TCP objective.
        if self.profile_step:
            wp.synchronize_device(self.device)
            _ik_t0 = time.perf_counter()

        self.pos_obj.set_target_position(0, _v3(self._target_pos))
        self.rot_obj.set_target_rotation(0, _v4(self._target_xyzw))

        ik_seed = self.ik_joint_q.numpy()
        for j, ik_qi in enumerate(self.ik_arm_coord_indices):
            ik_seed[0, ik_qi] = float(self._arm_cmd[j])
        self.ik_joint_q.assign(ik_seed)

        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=IK_TRACK_ITERS)

        solved_1 = self.ik_joint_q.numpy().reshape(-1)

        if np.isfinite(solved_1).all():
            q_posture_seed = self._nullspace_posture_seed(solved_1)

            ik_seed_2 = self.ik_joint_q.numpy()
            for j, ik_qi in enumerate(self.ik_arm_coord_indices):
                ik_seed_2[0, ik_qi] = float(q_posture_seed[j])
            self.ik_joint_q.assign(ik_seed_2)

            self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=IK_TRACK_ITERS)

            solved_2 = self.ik_joint_q.numpy().reshape(-1)
            if np.isfinite(solved_2).all():
                self._write_control_targets(solved_2)

        if self.profile_step:
            wp.synchronize_device(self.device)
            self._prof_ik_s += time.perf_counter() - _ik_t0

        # 3) Coupled MuJoCo(robot) + VBD(belt) stepping (full-scene physics).
        if self.profile_step:
            _phys_s, _collide_s, _solve_s = self._simulate_physics_profiled()
            self._prof_phys_s += _phys_s
            self._prof_collide_s += _collide_s
            self._prof_solve_s += _solve_s
            self._prof_frames += 1
        else:
            self._launch_physics()

        # Passive recording hook
        if self.episode_recorder is not None:
            self.episode_recorder.record_frame(
                self.frame_id,
                self.sim_time,
                self._target_pos,
                self._target_xyzw,
                self._grip_fraction,
                self._grip_requested_fraction,
                self.control.joint_target_q.numpy().copy(),
                self.state_0,
            )

        self.sim_time += self.frame_dt
        self.frame_id += 1
        
        wall_now = time.perf_counter()
        wall_time = wall_now - self.wall_start_time

        if wall_now - self.last_timing_print >= 1.0:
            # Windowed RTF (current capability) vs lifetime-average RTF.
            # RTF(now) = sim advanced / wall elapsed over just the last window.
            # RTF(avg) = same ratio since the first real frame.
            # behind    = how far the sim clock trails the wall clock (grows if slow).
            window_sim = self.sim_time - self._last_sim_time
            window_wall = wall_now - self._last_wall_time
            rtf_now = window_sim / window_wall if window_wall > 1.0e-9 else 0.0
            rtf_avg = self.sim_time / wall_time if wall_time > 1.0e-9 else 0.0

            self._last_sim_time = self.sim_time
            self._last_wall_time = wall_now
            self.last_timing_print = wall_now

            behind = wall_time - self.sim_time

            if self.profile_step and self._prof_frames > 0:
                _n = float(self._prof_frames)
                ik_ms = 1000.0 * self._prof_ik_s / _n
                phys_ms = 1000.0 * self._prof_phys_s / _n
                collide_ms = 1000.0 * self._prof_collide_s / _n
                solve_ms = 1000.0 * self._prof_solve_s / _n
                profile_suffix = (
                    f" | ik={ik_ms:5.1f} | phys={phys_ms:6.1f} "
                    f"(collide={collide_ms:5.1f} solve={solve_ms:6.1f}) ms/frame"
                )
            else:
                profile_suffix = ""

            print(
                f"[TIME] "
                f"sim={self.sim_time:8.3f} s | "
                f"real={wall_time:8.3f} s | "
                f"behind={behind:+6.2f} s | "
                f"RTF(now)={rtf_now:6.3f}x | "
                f"RTF(avg)={rtf_avg:6.3f}x"
                f"{profile_suffix}"
            )

            if self.profile_step:
                self._prof_frames = 0
                self._prof_ik_s = 0.0
                self._prof_phys_s = 0.0
                self._prof_collide_s = 0.0
                self._prof_solve_s = 0.0

    # Frame + sphere logging (verbatim demo helpers)
    def _log_frame(self, name, pos, quat, length, colors):
        ex = _rotate_vec(quat, np.array([1.0, 0.0, 0.0]))
        ey = _rotate_vec(quat, np.array([0.0, 1.0, 0.0]))
        ez = _rotate_vec(quat, np.array([0.0, 0.0, 1.0]))
        starts = wp.array([_v3(pos), _v3(pos), _v3(pos)], dtype=wp.vec3, device=self.device)
        ends = wp.array([_v3(pos + ex * length), _v3(pos + ey * length), _v3(pos + ez * length)],
                        dtype=wp.vec3, device=self.device)
        self.viewer.log_lines(name, starts, ends, colors)

    def _log_sphere(self, name, center, radius):
        s = wp.array(center + radius * self._sph_s, dtype=wp.vec3, device=self.device)
        e = wp.array(center + radius * self._sph_e, dtype=wp.vec3, device=self.device)
        self.viewer.log_lines(name, s, e, self._sph_colors)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self._main_view_layer is not None:
            self.viewer.activate(self._main_view_layer)
        show_contacts = self.viewer.show_contacts
        newton.examples.log_coupled_view(self, self.contacts)

        # teleop visuals: live gripper TIP frame + commanded target frame/sphere
        bq = self.state_0.body_q.numpy()[self.link_index]
        base_pos = np.array(bq[0:3], dtype=np.float64)
        base_quat = _norm4(np.array(bq[3:7], dtype=np.float64))
        tip = base_pos + _rotate_vec(base_quat, self.tip_offset)
        self._log_frame("/tip_current", tip, base_quat, self.axis_len, self._axis_bright)
        self._log_frame("/tip_target_axes", self._target_pos, self._target_xyzw,
                        self.axis_len, self._axis_dim)
        self._log_sphere("/tip_target_sphere", self._target_pos, self.tip_radius)

        now = time.perf_counter()
        if now - self._last_dbg >= 0.5:
            self._last_dbg = now
            err = float(np.linalg.norm(self._target_pos - tip))
            latch = "ON" if self._grip_hold_fraction is not None else "off"
            # print(f"[track] target-tip error = {err*1000:6.1f} mm  "
            #       f"grip_req={self._grip_requested_fraction:0.2f} "
            #       f"grip_cmd={self._grip_fraction:0.2f} "
            #       f"grip_actual={self._grip_actual_fraction:0.2f} latch={latch}")

        if (self.debug_belt_positions and self.state_0.body_q is not None
                and len(self.belt_bodies) > 0 and self.frame_id % self.debug_every_n_frames == 0):
            body_q = self.state_0.body_q.numpy()
            belt_xyz = body_q[np.asarray(self.belt_bodies, dtype=np.int32), :3]
            # print("[BELT DEBUG] min xyz =", belt_xyz.min(axis=0), "max xyz =", belt_xyz.max(axis=0))
            if len(self.pulley_bodies) > 0 and self.state_0.joint_q is not None:
                q_start = as_numpy(self.model.joint_q_start)
                jq = self.state_0.joint_q.numpy()
                angles = [float(jq[int(q_start[j])]) for j in self.pulley_joints]
                # print("[PULLEY DEBUG] axle angles [rad] =", angles)

        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)

        # shared-memory target buffer
        parser.add_argument("--buffer", type=str, default=SHARED_PATH_DEFAULT)

        # Dataset recording/replay only; these do not change normal simulation settings.
        parser.add_argument("--record-episode", action="store_true",
                            help="record initial state, per-frame controls, and Newton states")
        parser.add_argument("--record-dir", type=str, default="recordings",
                            help="directory in which recording episodes are created")
        parser.add_argument("--replay-episode", type=str, default=None,
                            help="replay an episode directory using initial_state.npz + actions.jsonl")

        # Performance only: keep the stable physics settings above; CUDA graph only reduces repeated GPU launch overhead.
        parser.add_argument("--no-cuda-graph", action="store_false", dest="cuda_graph",
                            default=True,
                            help="disable CUDA graph capture for A/B stability testing; physics parameters stay identical")
        parser.add_argument("--profile-step", action="store_true",
                            help=("print synchronized per-frame IK / physics / collide / solve timing; "
                                  "detailed profiling uses the uncaptured physics loop so collide and solve "
                                  "can be measured separately"))

        # teleop target visuals
        parser.add_argument("--axis-len", type=float, default=0.08)
        parser.add_argument("--tip-radius", type=float, default=0.06)
        parser.add_argument("--sphere-seg", type=int, default=28)
        parser.add_argument("--sphere-color", type=str, default="0.3,0.85,0.95")

        # IK slew limit + null-space posture bias
        parser.add_argument("--max-arm-speed", type=float, default=1.5,
                            help="maximum commanded UR10 joint speed in rad/s")
        parser.add_argument("--nullspace-gain", type=float, default=0.35,
                            help="strength of previous-posture preference in the task null space")
        parser.add_argument("--nullspace-svd-rel-tol", type=float, default=1.0e-3,
                            help="relative SVD threshold for detecting a true null-space direction")
        parser.add_argument("--nullspace-svd-abs-tol", type=float, default=1.0e-5,
                            help="absolute SVD threshold for detecting a true null-space direction")
        parser.add_argument("--max-nullspace-step", type=float, default=0.02,
                            help="maximum null-space seed correction per joint in rad")
        parser.add_argument("--nullspace-eps", type=float, default=1.0e-4,
                            help="joint perturbation in rad used for numerical TCP Jacobian")
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

    def __del__(self):
        try:
            if getattr(self, "episode_recorder", None) is not None:
                self.episode_recorder.close()
        except Exception:
            pass
        try:
            self.shared.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    viewer._pause = False
    newton.examples.run(Example(viewer, args), args)