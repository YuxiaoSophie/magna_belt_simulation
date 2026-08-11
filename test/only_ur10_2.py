"""
Newton UR10 + 2f85 sim that tracks the shared target.
  * SEEDS the shared memory buffer with the gripper's current TIP pose,
  * every frame READS the target from the buffer and runs IK to move the arm to it,
  * draws two things so you can verify motion:
        - GRIPPER TIP frame : bright RGB axes at the real grasp point
                              (midpoint of the two finger pads),
        - TARGET frame      : dim RGB axes + a transparent wireframe SPHERE
                              at the commanded pose. Initially it sits exactly
                              on the gripper; as you move the SpaceMouse the
                              sphere moves and the gripper chases it.
"""

from __future__ import annotations

import os
import math
import mmap
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode
from newton.solvers import SolverMuJoCo

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MJCF_PATH = PROJECT_DIR / "2f85.xml"

# Match the working round_belt controller exactly.
UR10_ARM_DOFS = 6
GRIPPER_TCP_LOCAL_OFFSET = (0.0, 0.0, 0.145)
IK_INIT_ITERS = 96
IK_TRACK_ITERS = 24
IK_LAMBDA_INITIAL = 0.05
ROBOTIQ_GRIPPER_SAFE_CLOSE_FRACTION = 0.93

# Shared memory target buffer 
SHARED_PATH_DEFAULT = "/tmp/sm_teleop_target.bin"


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
        
        
# helpers
def as_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def parse_indices(text):
    if text is None or text.strip() == "":
        return []
    return [int(v.strip()) for v in text.split(",") if v.strip() != ""]


def parse_vec3(text, default=(0.0, 0.0, 0.0)):
    if text is None or text.strip() == "":
        return np.array(default, dtype=np.float64)
    parts = [float(v.strip()) for v in text.split(",") if v.strip() != ""]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 floats, got {text!r}")
    return np.array(parts, dtype=np.float64)


def body_labels(model):
    return list(getattr(model, "body_label", []))


def find_body_contains(model, key, start=0):
    if not key:
        return -1
    key = key.lower()
    for i, label in enumerate(body_labels(model)):
        if i >= start and key in label.lower():
            return i
    return -1


def find_gripper_driver_coords(model, arm_dofs):
    joint_labels = getattr(model, "joint_label", [])
    q_start = as_numpy(model.joint_q_start)
    dof_dim = as_numpy(model.joint_dof_dim)
    coords = []
    for j, name in enumerate(joint_labels):
        if j < arm_dofs:
            continue
        dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
        if dofs <= 0:
            continue
        if "driver_joint" in name.lower():
            coords.append(int(q_start[j]))
    return sorted(set(coords))


def print_model_tree(model):
    print("\n========== BODY LABELS ==========")
    for i, name in enumerate(body_labels(model)):
        print(f"body {i:3d}: {name}")
    print("\n========== JOINT LABELS / COORD INDICES ==========")
    joint_labels = getattr(model, "joint_label", [])
    q_start = as_numpy(model.joint_q_start)
    dof_dim = as_numpy(model.joint_dof_dim)
    for j, name in enumerate(joint_labels):
        dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
        q0 = int(q_start[j])
        print(f"joint {j:3d}: q={list(range(q0, q0 + dofs))}  name={name}")
    print("=================================\n")


def _norm4(q):
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _quat_conj(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)


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


# Example
class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.arm_dofs = 6
        self.axis_len = float(args.axis_len)
        self.tip_radius = float(args.tip_radius)

        # scene: UR10 + 2f85 + column + ground
        builder = newton.ModelBuilder()
        # Use MuJoCo dynamics so the imported 2F-85 linkage/equality constraints
        # are actually solved.
        SolverMuJoCo.register_custom_attributes(builder)
        height = 1.2
        base_tf = wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity())
        tool_body_idx = 7

        # Track the robot's shape + body range so we can (a) FILTER the collision
        # pipeline (drop intra-robot pairs) and (b) enable gravity compensation
        # on the robot bodies. Both tricks are copied from round_belt_sim.
        robot_shape_start = builder.shape_count
        robot_body_start = builder.body_count

        if args.asset_file:
            if args.asset_type == "usd":
                builder.add_usd(args.asset_file, xform=base_tf,
                                collapse_fixed_joints=False,
                                enable_self_collisions=False, hide_collision_shapes=False)
            else:
                builder.add_urdf(args.asset_file, xform=base_tf, floating=False,
                                 enable_self_collisions=False, parse_visuals_as_colliders=False)
        else:
            asset_path = newton.utils.download_asset("universal_robots_ur10")
            builder.add_usd(str(asset_path / "usd" / "ur10_instanceable.usda"),
                            xform=base_tf, collapse_fixed_joints=False,
                            enable_self_collisions=False, hide_collision_shapes=True)
            print(f"[INFO] Attaching 2f85.xml to body {tool_body_idx} "
                  f"(self-collisions OFF).")
            grot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
            mjcf_kwargs = dict(parent_body=tool_body_idx,
                               xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), grot))
            gripper_body_start = builder.body_count
            try:
                builder.add_mjcf(str(MJCF_PATH), enable_self_collisions=False, **mjcf_kwargs)
            except TypeError:
                print("[INFO] add_mjcf has no enable_self_collisions kwarg; "
                      "continuing without it (pipeline filter still applies).")
                builder.add_mjcf(str(MJCF_PATH), **mjcf_kwargs)
            gripper_body_end = builder.body_count

        robot_shape_end = builder.shape_count
        robot_body_end = builder.body_count
        if args.asset_file:
            gripper_body_start = robot_body_start
            gripper_body_end = robot_body_end

        builder.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0.0, 0.0, height / 2.0)),
                                   half_height=height / 2.0, radius=0.08)
        builder.add_ground_plane()

        try:
            gravcomp = builder.custom_attributes["mujoco:gravcomp"]
            if gravcomp.values is None:
                gravcomp.values = {}
            for b in range(robot_body_start, robot_body_end):
                gravcomp.values[b] = 1.0
            print(f"[INFO] Enabled mujoco:gravcomp=1.0 on robot bodies "
                  f"{robot_body_start}..{robot_body_end - 1}.")
        except (KeyError, AttributeError):
            print("[INFO] mujoco:gravcomp attribute not available; skipping "
                  "gravity compensation (arm may sag under gravity).")

        self.model = builder.finalize()
        if args.print_model:
            print_model_tree(self.model)
        self.n_coords = self.model.joint_coord_count
        labels = body_labels(self.model)

        self.robot_shape_start = int(robot_shape_start)
        self.robot_shape_end = int(robot_shape_end)

        # Gripper driver coords + continuous presets
        gargs = args.gripper_indices
        if not gargs and not args.asset_file:
            drv = find_gripper_driver_coords(self.model, self.arm_dofs)
            if drv:
                gargs = ",".join(map(str, drv))
        self.gripper_coord_indices = parse_indices(gargs)

        # Find the corresponding DOF indices too.  Model joint limits / gains are
        # DOF-aligned on the Newton version used by the coupled reference scene.
        q_start = as_numpy(self.model.joint_q_start)
        qd_start = as_numpy(self.model.joint_qd_start)
        dof_dim = as_numpy(self.model.joint_dof_dim)
        self.gripper_dof_indices = []
        for j, name in enumerate(getattr(self.model, "joint_label", [])):
            dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
            if dofs <= 0 or "driver_joint" not in name.lower():
                continue
            qi = int(q_start[j])
            if qi in self.gripper_coord_indices:
                self.gripper_dof_indices.append(int(qd_start[j]))

        lower = as_numpy(self.model.joint_limit_lower)
        upper = as_numpy(self.model.joint_limit_upper)
        m = 0.005
        self.g_open, self.g_closed = [], []
        for dof_idx in self.gripper_dof_indices:
            lo, hi = float(lower[dof_idx]), float(upper[dof_idx])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                open_q = lo + m
                safe_close_q = lo + ROBOTIQ_GRIPPER_SAFE_CLOSE_FRACTION * (hi - lo)
                close_q = min(max(safe_close_q, open_q), hi - m)
                self.g_open.append(open_q)
                self.g_closed.append(close_q)
            else:
                self.g_open.append(0.0)
                self.g_closed.append(float(args.gripper_closed))
        print(f"[INFO] Gripper driver coords: {self.gripper_coord_indices}")
        print(f"[INFO] Gripper driver dofs:   {self.gripper_dof_indices}")
        print(f"[INFO] Gripper open/closed:   {self.g_open} -> {self.g_closed}")
        
        self.link_index = -1
        for b in range(gripper_body_start, gripper_body_end):
            leaf = labels[b].replace("\\", "/").rsplit("/", 1)[-1].lower()
            if leaf == "base":
                self.link_index = b
                break
        if self.link_index < 0:
            for b in range(gripper_body_start, gripper_body_end):
                leaf = labels[b].replace("\\", "/").rsplit("/", 1)[-1].lower()
                if "base" in leaf and "mount" not in leaf:
                    self.link_index = b
                    break
        if self.link_index < 0:
            self.link_index = gripper_body_start
        self.tip_offset = np.asarray(GRIPPER_TCP_LOCAL_OFFSET, dtype=np.float64)
        print(f"[INFO] IK TCP body {self.link_index} "
              f"({labels[self.link_index] if 0 <= self.link_index < len(labels) else '?'})")
        print(f"[INFO] TCP local offset = {GRIPPER_TCP_LOCAL_OFFSET}")

        # home pose + open gripper
        q0 = np.zeros(self.n_coords, dtype=np.float32)
        q0[0:6] = [0.0, -1.35, 1.75, -1.95, -1.57, 0.0]
        for qi, ov in zip(self.gripper_coord_indices, self.g_open):
            q0[qi] = ov
        self.model.joint_q.assign(q0)
        self.model.joint_qd.zero_()

        # Configure PD position targets for the 6 UR10 joints and the 2F-85
        # driver joint(s).
        self.control = self.model.control()
        n_dofs = int(self.model.joint_dof_count)
        qd_start = as_numpy(self.model.joint_qd_start)
        dof_dim = as_numpy(self.model.joint_dof_dim)

        self.arm_coord_indices = []
        self.arm_dof_indices = []
        for j, name in enumerate(getattr(self.model, "joint_label", [])):
            dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
            if dofs <= 0:
                continue
            self.arm_coord_indices.append(int(as_numpy(self.model.joint_q_start)[j]))
            self.arm_dof_indices.append(int(qd_start[j]))
            if len(self.arm_coord_indices) >= self.arm_dofs:
                break

        mode_np = as_numpy(self.model.joint_target_mode).copy()
        ke_np = as_numpy(self.model.joint_target_ke).copy()
        kd_np = as_numpy(self.model.joint_target_kd).copy()
        mode_np[:] = int(JointTargetMode.NONE)
        ke_np[:] = 0.0
        kd_np[:] = 0.0

        # Lower gains + explicit command rate limiting give visibly smooth motion.
        for di in self.arm_dof_indices:
            mode_np[di] = int(JointTargetMode.POSITION)
            ke_np[di] = float(args.arm_ke)
            kd_np[di] = float(args.arm_kd)
        for di in self.gripper_dof_indices:
            mode_np[di] = int(JointTargetMode.POSITION)
            ke_np[di] = float(args.gripper_ke)
            kd_np[di] = float(args.gripper_kd)

        self.model.joint_target_mode.assign(mode_np)
        self.model.joint_target_ke.assign(ke_np)
        self.model.joint_target_kd.assign(kd_np)

        # Determine whether Control.joint_target_q is coord-space or DOF-space.
        ctrl_len = len(as_numpy(self.control.joint_target_q))
        if ctrl_len == int(self.model.joint_coord_count):
            self.target_layout = "coord"
            self.arm_target_indices = list(self.arm_coord_indices)
            self.gripper_target_indices = list(self.gripper_coord_indices)
            base_targets = self.model.joint_q.numpy().astype(np.float32).copy()
        elif ctrl_len == n_dofs:
            self.target_layout = "dof"
            self.arm_target_indices = list(self.arm_dof_indices)
            self.gripper_target_indices = list(self.gripper_dof_indices)
            base_targets = np.zeros(ctrl_len, dtype=np.float32)
            qnow = self.model.joint_q.numpy()
            for ti, qi in zip(self.arm_target_indices, self.arm_coord_indices):
                base_targets[ti] = qnow[qi]
            for ti, ov in zip(self.gripper_target_indices, self.g_open):
                base_targets[ti] = ov
        else:
            raise RuntimeError(
                f"control.joint_target_q length {ctrl_len} matches neither "
                f"joint_coord_count={self.model.joint_coord_count} nor joint_dof_count={n_dofs}"
            )
        self.control.joint_target_q.assign(base_targets)
        self._arm_cmd = np.array(
            [base_targets[i] for i in self.arm_target_indices], dtype=np.float64
        )

        # Dynamic state + MuJoCo solver.
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, broad_phase="explicit",
            shape_pairs_filtered=self._robot_external_shape_pairs(),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.dyn_solver = SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=int(args.mj_iters),
            ls_iterations=int(args.mj_ls_iters),
            use_mujoco_contacts=False,
            njmax=int(args.mj_njmax),
            nconmax=int(args.mj_nconmax),
        )
        self.sim_substeps = int(args.sim_substeps)
        self.sim_dt = self.frame_dt / max(1, self.sim_substeps)

        bq = self.state_0.body_q.numpy()
        base = bq[self.link_index]
        base_pos = np.array(base[0:3], dtype=np.float64)
        base_quat = _norm4(np.array(base[3:7], dtype=np.float64))
        tip0 = base_pos + _rotate_vec(base_quat, self.tip_offset)

        self._target_pos = tip0.copy()
        self._target_xyzw = base_quat.copy()
        self._grip_fraction = 0.0  # 0=open, 1=closed

        self._build_ik_model(base_tf, tool_body_idx)

        # Seed shared memory at the ACTUAL current TCP and OPEN gripper.
        self.shared = SharedTarget(args.buffer)
        self.shared.write(self._target_pos, self._target_xyzw, 0.0, ready=1.0)
        print(f"[INFO] seeded shared target buffer: {args.buffer}")

        # Frame + sphere visuals
        self._axis_bright = wp.array(
            [wp.vec3(1.0, 0.15, 0.15), wp.vec3(0.15, 1.0, 0.15), wp.vec3(0.2, 0.4, 1.0)],
            dtype=wp.vec3, device=self.device)
        self._axis_dim = wp.array(
            [wp.vec3(0.55, 0.1, 0.1), wp.vec3(0.1, 0.55, 0.1), wp.vec3(0.12, 0.2, 0.6)],
            dtype=wp.vec3, device=self.device)
        self._sph_s, self._sph_e = _unit_sphere_wire(n_lat=2, seg=int(args.sphere_seg))
        sc = parse_vec3(args.sphere_color, default=(0.3, 0.85, 0.95))
        self._sph_colors = wp.array([_v3(sc)] * len(self._sph_s), dtype=wp.vec3, device=self.device)

        self._last_dbg = 0.0

        self.viewer.set_model(self.model)
        try:
            self.viewer.set_camera(pos=wp.vec3(1.9, -1.9, 2.1), pitch=-22.0, yaw=45.0)
        except Exception:
            pass

    # Keep robot<->world pairs, drop robot<->robot
    def _robot_external_shape_pairs(self) -> wp.array:
        """Filtered shape-pair list for the main collision pipeline.

        Mirrors round_belt_sim's _belt_world_shape_pairs: we take the model's
        candidate contact pairs and remove every pair where BOTH shapes belong
        to the robot (arm + gripper). This eliminates the 2F-85 self-collision
        flood while still letting the arm collide with the support column and
        ground plane.
        """
        robot_shapes = set(range(self.robot_shape_start, self.robot_shape_end))
        pairs = []
        n_dropped = 0
        for a, b in self.model.shape_contact_pairs.numpy():
            a = int(a); b = int(b)
            if a in robot_shapes and b in robot_shapes:
                n_dropped += 1
                continue  # intra-robot (gripper linkage / arm self) -> drop
            pairs.append((a, b))
        arr = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        print(f"[INFO] Collision filter: kept {len(pairs)} robot<->world pairs, "
              f"dropped {n_dropped} intra-robot pairs "
              f"(robot shapes {self.robot_shape_start}..{self.robot_shape_end - 1}).")
        return wp.array(arr, dtype=wp.vec2i, device=self.model.device)

    # Separate IK model
    def _build_ik_model(self, base_tf, tool_body_idx):
        ik_builder = newton.ModelBuilder()
        asset_path = newton.utils.download_asset("universal_robots_ur10")
        ik_builder.add_usd(
            str(asset_path / "usd" / "ur10_instanceable.usda"),
            xform=base_tf, collapse_fixed_joints=False,
            enable_self_collisions=False, hide_collision_shapes=True,
        )
        grot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
        gstart = ik_builder.body_count
        kwargs = dict(
            parent_body=tool_body_idx,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), grot),
        )
        try:
            ik_builder.add_mjcf(str(MJCF_PATH), enable_self_collisions=False, **kwargs)
        except TypeError:
            ik_builder.add_mjcf(str(MJCF_PATH), **kwargs)
        gend = ik_builder.body_count

        # Locate the SAME Robotiq base link in the IK model.
        ik_labels = list(getattr(ik_builder, "body_label", []))
        ik_tcp_body = -1
        for b in range(gstart, gend):
            leaf = ik_labels[b].replace("\\", "/").rsplit("/", 1)[-1].lower()
            if leaf == "base":
                ik_tcp_body = b
                break
        if ik_tcp_body < 0:
            for b in range(gstart, gend):
                leaf = ik_labels[b].replace("\\", "/").rsplit("/", 1)[-1].lower()
                if "base" in leaf and "mount" not in leaf:
                    ik_tcp_body = b
                    break
        if ik_tcp_body < 0:
            ik_tcp_body = gstart

        self.ik_model = ik_builder.finalize(device=self.device)
        self.ik_n_coords = int(self.ik_model.joint_coord_count)

        # First six actuated coordinates are the UR10, same as round_belt.
        q_start = as_numpy(self.ik_model.joint_q_start)
        dof_dim = as_numpy(self.ik_model.joint_dof_dim)
        self.ik_arm_coord_indices = []
        for j in range(len(getattr(self.ik_model, "joint_label", []))):
            dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
            if dofs <= 0:
                continue
            q0 = int(q_start[j])
            for k in range(dofs):
                self.ik_arm_coord_indices.append(q0 + k)
                if len(self.ik_arm_coord_indices) >= UR10_ARM_DOFS:
                    break
            if len(self.ik_arm_coord_indices) >= UR10_ARM_DOFS:
                break
        if len(self.ik_arm_coord_indices) != len(self.arm_coord_indices):
            raise RuntimeError(
                f"IK/main arm coordinate mismatch: "
                f"ik={self.ik_arm_coord_indices}, main={self.arm_coord_indices}"
            )

        ik_seed = self.ik_model.joint_q.numpy().astype(np.float32).copy()
        main_q = self.model.joint_q.numpy().astype(np.float32)
        for ik_idx, main_idx in zip(self.ik_arm_coord_indices, self.arm_coord_indices):
            ik_seed[ik_idx] = main_q[main_idx]

        self.ik_joint_q = wp.array(
            ik_seed.reshape((1, self.ik_n_coords)), dtype=float, device=self.device
        )
        self.ik_target_positions = wp.array(
            [_v3(self._target_pos)], dtype=wp.vec3, device=self.device
        )
        self.ik_target_rotations = wp.array(
            [_v4(self._target_xyzw)], dtype=wp.vec4, device=self.device
        )
        self.pos_obj = ik.IKObjectivePosition(
            link_index=ik_tcp_body,
            link_offset=_v3(GRIPPER_TCP_LOCAL_OFFSET),
            target_positions=self.ik_target_positions,
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=ik_tcp_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ik_target_rotations,
        )
        self.lim_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.clone(self.ik_model.joint_limit_lower),
            joint_limit_upper=wp.clone(self.ik_model.joint_limit_upper),
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model, n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.lim_obj],
            lambda_initial=IK_LAMBDA_INITIAL,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        # One high-iteration solve at t=0. Since the target is the current TCP,
        # this should return the current joint branch rather than jumping elsewhere.
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=IK_INIT_ITERS)

    def _write_control_targets(self, solved_q):
        """Write IK + gripper targets exactly like round_belt (no extra rate limiter)."""
        targets = self.control.joint_target_q.numpy().copy()

        for ti, ik_qi in zip(self.arm_target_indices, self.ik_arm_coord_indices):
            targets[ti] = float(solved_q[ik_qi])

        frac = float(np.clip(self._grip_fraction, 0.0, 1.0))
        grip_vals = (
            np.asarray(self.g_open, dtype=np.float64)
            + frac * (np.asarray(self.g_closed, dtype=np.float64)
                      - np.asarray(self.g_open, dtype=np.float64))
        )
        for ti, val in zip(self.gripper_target_indices, grip_vals):
            targets[ti] = float(val)

        self.control.joint_target_q.assign(targets)

    def step(self):
        # 1) Read SpaceMouse target. Neutral input simply leaves this pose unchanged.
        pos, quat, grip, ready = self.shared.read()
        if ready >= 0.5:
            pos = np.asarray(pos, dtype=np.float64)
            quat = _norm4(quat)
            if np.isfinite(pos).all() and np.isfinite(quat).all():
                self._target_pos = pos.copy()
                self._target_xyzw = quat.copy()
            self._grip_fraction = float(np.clip(grip, 0.0, 1.0))

        # 2) Solve the TCP objective.
        self.pos_obj.set_target_position(0, _v3(self._target_pos))
        self.rot_obj.set_target_rotation(0, _v4(self._target_xyzw))
        self.ik_solver.step(
            self.ik_joint_q, self.ik_joint_q, iterations=IK_TRACK_ITERS
        )
        solved = self.ik_joint_q.numpy().reshape(-1)
        if np.isfinite(solved).all():
            self._write_control_targets(solved)

        # 3) Dynamic MuJoCo tracking, then synchronize joint state exactly as in
        #    the working round_belt simulation.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.dyn_solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            newton.eval_ik(
                self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt

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
        self.viewer.log_state(self.state_0)

        # actual gripper TIP frame
        bq = self.state_0.body_q.numpy()[self.link_index]
        base_pos = np.array(bq[0:3], dtype=np.float64)
        base_quat = _norm4(np.array(bq[3:7], dtype=np.float64))
        tip = base_pos + _rotate_vec(base_quat, self.tip_offset)
        self._log_frame("/tip_current", tip, base_quat, self.axis_len, self._axis_bright)

        # target frame + transparent sphere
        self._log_frame("/tip_target_axes", self._target_pos, self._target_xyzw,
                        self.axis_len, self._axis_dim)
        self._log_sphere("/tip_target_sphere", self._target_pos, self.tip_radius)

        # numeric tracking check
        now = time.perf_counter()
        if now - self._last_dbg >= 0.5:
            self._last_dbg = now
            err = float(np.linalg.norm(self._target_pos - tip))
            print(f"[track] target-tip error = {err*1000:6.1f} mm  "
                  f"grip={self._grip_fraction:0.2f}")

        self.viewer.end_frame()

    def test_final(self):
        pass

    def __del__(self):
        try:
            self.shared.close()
        except Exception:
            pass

    @staticmethod
    def create_parser():
        p = newton.examples.create_parser()
        p.add_argument("--buffer", type=str, default=SHARED_PATH_DEFAULT)
        p.add_argument("--asset-file", type=str, default="")
        p.add_argument("--asset-type", type=str, default="usd", choices=["usd", "urdf"])
        p.add_argument("--axis-len", type=float, default=0.08)
        p.add_argument("--tip-radius", type=float, default=0.06)
        p.add_argument("--sphere-seg", type=int, default=28)
        p.add_argument("--sphere-color", type=str, default="0.3,0.85,0.95")
        p.add_argument("--gripper-indices", type=str, default="")
        p.add_argument("--gripper-closed", type=float, default=0.8)
        p.add_argument("--arm-ke", type=float, default=700.0)   # match round_belt
        p.add_argument("--arm-kd", type=float, default=110.0)   # match round_belt
        p.add_argument("--gripper-ke", type=float, default=260.0)
        p.add_argument("--gripper-kd", type=float, default=45.0)
        p.add_argument("--sim-substeps", type=int, default=8)
        p.add_argument("--mj-iters", type=int, default=50)
        p.add_argument("--mj-ls-iters", type=int, default=20)
        p.add_argument("--mj-njmax", type=int, default=2048,
                       help="MuJoCo max constraints (raise if you ever see njmax warnings)")
        p.add_argument("--mj-nconmax", type=int, default=2048,
                       help="MuJoCo max contacts (raise if you ever see nconmax warnings)")
        p.add_argument("--print-model", action="store_true")
        return p


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    viewer._pause = False
    newton.examples.run(Example(viewer, args), args)