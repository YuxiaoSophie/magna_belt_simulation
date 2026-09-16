"""The runnable simulation: solver, stepping and the startup diagnostics.

:class:`BeltTaskSimulation` finalizes a task's scene, partitions it into the repo's
MuJoCo (robots) + VBD (belt) proxy-coupled solver, and holds both arms at their Drake
default joint angles (see ``joint_state.py``) so it can be inspected -- no teleop / IK /
recording / ADMM.
Every solver parameter comes from ``round_belt.py``; only the partition is scene-specific.
The fixed 10-substep frame is captured into a CUDA graph for speed only;
``--no-cuda-graph`` runs identical physics.  Diagnostics go through loguru, whose sink is
installed by the entry script's ``__main__`` alone.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import warp as wp
from loguru import logger

import newton
import newton.examples
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

import round_belt
from task_common.cameras import RgbdCameras
from task_common.joint_state import index_layout
from task_common.point_cloud import CroppedPointCloud
from task_common.scene import (
    UR10_BASE_LABEL, UR10_WRIST3_LABEL, JointConfig, SceneInfo, make_builder,
)
from utils.labels import body_index, body_label_endswith
from utils.transforms import rpy_deg_from_quat


class BeltTaskSimulation:
    """A two-arm belt scene held at its default configuration; subclasses supply the hooks."""

    belt_center: tuple[float, float, float]
    table_top_z: float
    frame_rate: int = 60

    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace) -> None:
        self.viewer = viewer
        self.args = args

        self.fps = self.frame_rate
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # Odd counts leave the newest state in state_1 after a graph replay (ping-pong swap).
        self.sim_substeps = int(getattr(args, "substeps", 10))
        if self.sim_substeps < 2 or self.sim_substeps % 2 != 0:
            raise ValueError(f"substeps must be even and >= 2, got {self.sim_substeps}")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.vbd_iterations = int(getattr(args, "vbd_iterations", round_belt.VBD_ITERATIONS))
        self.frame_id = 0

        builder = make_builder()
        self.info = info = self._build_scene(builder)

        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device

        # The VBD entry owns everything that is not a robot shape (belt + statics).
        robot_shape_set = set(info.robot_shapes)
        self.vbd_shapes = [s for s in range(self.model.shape_count) if s not in robot_shape_set]
        self.vbd_bodies = sorted(info.belt_bodies)
        self.vbd_joints = sorted(info.belt_joints)

        self._apply_contact_materials(info)

        self._apply_default_joint_state(self.model, info)
        self.control = self.model.control()
        self._seed_control_targets(info.joint_config)
        self.solver = self._build_solver(info)
        proxy_labels = [str(self.model.body_label[b]) for b in info.proxy_bodies]
        logger.info(f"[PROXY] mjc->vbd bodies: {proxy_labels}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, broad_phase="explicit",
            shape_pairs_filtered=self._belt_world_shape_pairs())
        self.contacts = self.collision_pipeline.contacts()
        if hasattr(self.solver, "prepare_contacts"):
            self.solver.prepare_contacts(self.contacts)

        self.viewer.set_model(self.model)
        newton.examples.configure_coupled_view(self, self.args)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        body_labels = list(self.model.body_label)
        self._gripper_root_body = body_label_endswith(body_labels, "/base_mount")
        self._hand_body = body_index(body_labels, "panda_hand/panda_hand")
        body_q = self.state_0.body_q.numpy()
        self._initial_hand_pos = np.array(body_q[self._hand_body][:3], dtype=np.float64)
        self._initial_gripper_pos = np.array(body_q[self._gripper_root_body][:3], dtype=np.float64)
        self._print_pose_table()

        self.cameras: RgbdCameras | None = None
        self.point_clouds: dict[str, CroppedPointCloud] = {}
        self.cameras_updated = False
        self._next_camera_time = 0.0
        if info.cameras and getattr(self.args, "cameras", True):
            self.cameras = RgbdCameras(self.model, info.cameras)
            self.point_clouds = {
                spec.name: CroppedPointCloud(self.cameras, spec) for spec in info.point_clouds
            }
            names = [spec.name for spec in info.cameras]
            logger.info(f"[CAMERAS] {names} at {1.0 / self.cameras.period:g} fps")

        # Performance only: the frame is a fixed sim_substeps sequence of the same kernels,
        # so it is captured once (after the first uncaptured frame, see _step_physics) and
        # replayed.  No solver setting, substep count or dt changes.
        self.physics_graph = None
        self.use_cuda_graph = bool(getattr(self.args, "cuda_graph", True)) and self.device.is_cuda
        if not self.use_cuda_graph:
            logger.info(f"[CUDA GRAPH] disabled (device={self.device})")

    def _build_scene(self, builder: newton.ModelBuilder) -> SceneInfo:
        raise NotImplementedError

    def _apply_default_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        raise NotImplementedError

    def _apply_contact_materials(self, info: SceneInfo) -> None:
        """Global cable material for every shape (as round_belt.py), then the proxy override."""
        self.model.shape_material_ke.fill_(round_belt.CABLE_CONTACT_KE)
        self.model.shape_material_kd.fill_(round_belt.CABLE_CONTACT_KD)
        self.model.shape_material_mu.fill_(round_belt.CABLE_CONTACT_MU)
        if not info.contact_override_shapes:
            return
        pad_idx = np.asarray(info.contact_override_shapes, dtype=np.int32)
        for array, value in (
            (self.model.shape_material_mu, round_belt.GRIPPER_CONTACT_MU),
            (self.model.shape_material_ke, round_belt.GRIPPER_CONTACT_KE),
            (self.model.shape_material_kd, round_belt.GRIPPER_CONTACT_KD),
        ):
            values = array.numpy().copy()
            values[pad_idx] = value
            array.assign(values)

    def _build_solver(self, info: SceneInfo) -> SolverCoupledProxy:
        """MuJoCo (robots) + VBD (belt) with the pads and Franka fingers proxied into VBD."""
        return SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v, solver="newton", integrator="implicitfast", cone="elliptic",
                        iterations=round_belt.MUJOCO_ITERATIONS,
                        ls_iterations=round_belt.MUJOCO_LS_ITERATIONS,
                        use_mujoco_contacts=False, njmax=256, nconmax=128),
                    bodies=list(info.robot_bodies), joints=list(info.robot_joints),
                    shapes=list(info.robot_shapes)),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v, iterations=self.vbd_iterations,
                        rigid_avbd_beta=round_belt.VBD_RIGID_AVBD_BETA,
                        rigid_contact_k_start=round_belt.VBD_RIGID_CONTACT_K_START,
                        rigid_contact_history=True,
                        rigid_body_contact_buffer_size=round_belt.VBD_RIGID_CONTACT_BUFFER_SIZE),
                    bodies=self.vbd_bodies, joints=self.vbd_joints, shapes=self.vbd_shapes),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(
                    source="mjc", destination="vbd", bodies=list(info.proxy_bodies),
                    mass_scale=round_belt.PROXY_MASS_SCALE, mode=round_belt.PROXY_COUPLING_MODE,
                    # Sticky matching keeps friction anchors across steps: static grip.
                    collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                        model, broad_phase="explicit", contact_matching="sticky"),
                    collide_interval=1)],
                iterations=round_belt.PROXY_ITERATIONS),
        )

    def _seed_control_targets(self, cfg: JointConfig) -> None:
        as_numpy = round_belt.as_numpy
        ctrl_len = len(as_numpy(self.control.joint_target_q))
        n_coords = int(self.model.joint_coord_count)
        n_dofs = int(self.model.joint_dof_count)
        joint_q_np = self.model.joint_q.numpy().astype(np.float32)

        layout = index_layout(ctrl_len, n_coords, n_dofs, "control.joint_target_q")
        if layout == "coord":
            base = joint_q_np.copy()
        else:
            base = np.zeros(ctrl_len, dtype=np.float32)
            for t, c in zip(cfg["arm_target_indices"], cfg["arm_coord_indices"]):
                base[t] = joint_q_np[c]
            for t, c in zip(cfg["finger_target_indices"], cfg["finger_coord_indices"]):
                base[t] = joint_q_np[c]
            for t, v in zip(cfg["gripper_target_indices"], cfg["gripper_open_values"]):
                base[t] = v
        self.control.joint_target_q.reshape((1, ctrl_len)).assign(base.reshape(1, -1))
        logger.info(
            f"control.joint_target_q layout: {layout}-space (len={ctrl_len}, "
            f"n_coords={n_coords}, n_dofs={n_dofs}; gains are {cfg['gains_layout']}-space)"
        )

    def _belt_world_shape_pairs(self) -> wp.array:
        """Belt <-> static/ground pairs only (belt self-contact and the robot are
        handled by their own solver entries)."""
        belt_shapes = set(self.info.belt_shapes)
        static_vbd_shapes = set(self.vbd_shapes) - belt_shapes
        pairs = []
        for a, b in self.model.shape_contact_pairs.numpy():
            a, b = int(a), int(b)
            a_belt, b_belt = a in belt_shapes, b in belt_shapes
            if (a_belt ^ b_belt) and ((a in static_vbd_shapes) or (b in static_vbd_shapes)):
                pairs.append((a, b))
        if not pairs:
            raise RuntimeError("No belt contact pairs were generated")
        logger.debug(
            f"Collision pipeline: {len(pairs)} belt<->world pairs (belt self-contact disabled)."
        )
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.model.device)

    def _print_pose_table(self) -> None:
        body_labels = list(self.model.body_label)
        body_q = self.state_0.body_q.numpy()
        rows = [
            ("panda_arm/panda_link0", body_index(body_labels, "panda_arm/panda_link0")),
            ("panda_arm/panda_link8", body_index(body_labels, "panda_arm/panda_link8")),
            ("panda_hand/panda_hand", self._hand_body),
            (UR10_BASE_LABEL, body_index(body_labels, UR10_BASE_LABEL)),
            (UR10_WRIST3_LABEL, body_index(body_labels, UR10_WRIST3_LABEL)),
            (f"{body_labels[self._gripper_root_body]} (2f85 root)", self._gripper_root_body),
        ]
        for i, pad in enumerate(self.info.gripper_pad_bodies):
            rows.append((f"{body_labels[pad]} (pad {i})", pad))

        # ONE multi-line record: loguru prefixes a record once, so the columns stay aligned.
        lines = [
            "[POSE TABLE] world poses after eval_fk at the default configuration",
            f"{'body':<44}{'x':>9}{'y':>9}{'z':>9}   "
            f"{'qx':>8}{'qy':>8}{'qz':>8}{'qw':>8}   "
            f"{'roll':>9}{'pitch':>9}{'yaw':>9}   (deg)",
        ]
        for label, idx in rows:
            t = body_q[idx]
            rpy = rpy_deg_from_quat(t[3:7])
            lines.append(
                f"{label:<44}{t[0]:>9.5f}{t[1]:>9.5f}{t[2]:>9.5f}   "
                f"{t[3]:>8.4f}{t[4]:>8.4f}{t[5]:>8.4f}{t[6]:>8.4f}   "
                f"{rpy[0]:>9.3f}{rpy[1]:>9.3f}{rpy[2]:>9.3f}"
            )
        logger.info("\n".join(lines))

    def _simulate_physics(self) -> None:
        """One frame of physics: sim_substeps of collide + coupled solve + IK.

        This is the graph-captured body; it must contain GPU work only (no host
        readback, no allocation, no logging) so wp.capture_launch can replay it.
        """
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(
                self.state_0, self.contacts, collision_pipeline=self.collision_pipeline
            )
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture_physics_graph(self) -> None:
        """Capture the fixed sim_substeps-substep frame once; fall back if it fails.

        Mirrors round_belt.py:1604-1648.  Capture records the launches without executing
        them, so the simulation state is untouched -- but the Python-level state_0/state_1
        swap does run, hence the explicit restore.
        """
        saved_state_0, saved_state_1 = self.state_0, self.state_1
        try:
            wp.synchronize_device(self.device)
            with wp.ScopedDevice(self.device):
                with wp.ScopedCapture() as capture:
                    self._simulate_physics()
            if capture.graph is None:
                raise RuntimeError("Warp returned no CUDA graph")
            self.physics_graph = capture.graph
            logger.success(
                f"[CUDA GRAPH] captured coupled proxy physics: "
                f"{self.sim_substeps} substeps/frame, dt={self.sim_dt:.9f} s"
            )
        except Exception as exc:  # noqa: BLE001 - any capture failure must degrade, not crash
            self.physics_graph = None
            self.use_cuda_graph = False
            logger.warning(f"[CUDA GRAPH] capture failed; running uncaptured physics: {exc}")
        finally:
            # sim_substeps is even so the swap restores the original ordering anyway,
            # but restore explicitly so capture cannot perturb it.
            self.state_0, self.state_1 = saved_state_0, saved_state_1

    def reset_body_poses(self, bodies: Sequence[int], positions: np.ndarray) -> None:
        """Teleport ``bodies`` to world ``positions`` (orientation kept, velocities zeroed).

        ``solver.reset(flags=0)`` keeps the authored pose and only rebaselines solver history;
        it may reallocate entry caches, so the CUDA graph is dropped and re-captured after the
        next (uncaptured) frame.
        """
        idx = np.asarray(bodies, dtype=np.int64)
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        body_q[idx, :3] = np.asarray(positions, dtype=np.float64).reshape(len(idx), 3)
        body_qd[idx] = 0.0
        self.state_0.body_q.assign(body_q)
        self.state_0.body_qd.assign(body_qd)

        q_start = self.model.joint_q_start.numpy()
        robot_end = self.info.robot_joints[-1] + 1
        n_coords = int(self.model.joint_coord_count)
        coord_end = int(q_start[robot_end]) if robot_end < len(q_start) else n_coords
        robot_coords = slice(int(q_start[self.info.robot_joints[0]]), coord_end)
        joint_q_before = self.state_0.joint_q.numpy()[robot_coords]
        self.solver.reset(self.state_0, flags=0)
        moved = float(np.abs(self.state_0.joint_q.numpy()[robot_coords] - joint_q_before).max())
        assert moved < 1e-6, f"solver.reset moved the robots' joint_q by {moved:.3g}"
        self.physics_graph = None

    def _step_physics(self) -> None:
        """Replay the captured frame, or run it uncaptured and then capture it."""
        if self.physics_graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.physics_graph)
            return
        self._simulate_physics()
        # VBD contact history must be allocated by a real frame before capture.
        if self.use_cuda_graph:
            self._capture_physics_graph()

    def step(self) -> None:
        self._step_physics()
        self.sim_time += self.frame_dt
        self.frame_id += 1
        self._update_cameras()

    def _update_cameras(self) -> None:
        """Render the cameras once per camera period."""
        self.cameras_updated = False
        # Half a frame of slack so float drift in sim_time never skips a period.
        if self.cameras is None or self.sim_time + 0.5 * self.frame_dt < self._next_camera_time:
            return
        self.cameras.update(self.state_0)
        self.cameras_updated = True
        self._next_camera_time += self.cameras.period

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        if self.cameras_updated and not isinstance(self.viewer, newton.viewer.ViewerNull):
            self.cameras.log(self.viewer)
            for cloud in self.point_clouds.values():
                cloud.log(self.viewer)
        self.viewer.end_frame()

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        # Performance only: physics is identical either way; the graph just removes
        # repeated GPU launch overhead.
        parser.add_argument(
            "--no-cuda-graph", action="store_false", dest="cuda_graph", default=True,
            help="disable CUDA graph capture (A/B testing); solver settings are unchanged",
        )
        parser.add_argument(
            "--substeps", type=int, default=10,
            help="physics substeps per frame; must be even (CUDA graph state ping-pong)",
        )
        parser.add_argument(
            "--vbd-iterations", type=int, default=round_belt.VBD_ITERATIONS,
            help="VBD solver iterations per substep",
        )
        parser.add_argument(
            "--no-cameras", action="store_false", dest="cameras", default=True,
            help="do not render the scene's RGBD cameras",
        )
        # The scene's colliders are created with is_visible=False (round_belt.py's
        # make_robust_table_collision_cfg), so ViewerBase._shape_visible only draws them
        # when show_collision is set.  The GL viewer has a "Show Collision" checkbox,
        # viser has none -- hence the flag.
        parser.add_argument(
            "--show-collision", action="store_true", default=False,
            help="draw collision geometry as well as visuals (works in every viewer)",
        )
        return parser

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(body_q).all(), "Non-finite body transforms"

        belt_xyz = body_q[np.asarray(self.info.belt_bodies, dtype=np.int32), :3]
        belt_min_z = float(belt_xyz[:, 2].min())
        assert belt_min_z > self.table_top_z - 0.01, (
            f"Belt fell below the holder: min_z={belt_min_z:.5f} (limit "
            f"{self.table_top_z - 0.01:.5f})"
        )
        belt_centroid = belt_xyz[:, :2].mean(axis=0)
        drift = float(np.linalg.norm(belt_centroid - np.asarray(self.belt_center[:2])))
        assert drift < 0.02, f"Belt XY centroid drifted {drift * 1000:.1f} mm from BELT_CENTER"

        cfg = self.info.joint_config
        joint_q = self.state_0.joint_q.numpy()
        for label, coord, default in zip(
            cfg["arm_labels"], cfg["arm_coord_indices"], cfg["arm_defaults"]
        ):
            err = abs(float(joint_q[coord]) - float(default))
            assert err < 0.05, f"{label} drifted {err:.4f} rad from its default"

        for what, body, initial in (
            ("panda_hand", self._hand_body, self._initial_hand_pos),
            ("2f85 root", self._gripper_root_body, self._initial_gripper_pos),
        ):
            drift = float(np.linalg.norm(np.array(body_q[body][:3]) - initial))
            assert drift < 0.005, f"{what} moved {drift * 1000:.2f} mm"
        logger.success("[TEST] test_final passed.")
