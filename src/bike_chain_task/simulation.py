"""The bike-chain simulation: chain, sprockets, robots and statics in ONE ``SolverMuJoCo``.

:class:`_BikeChainTask` swaps :class:`task_common.simulation.BeltTaskSimulation`'s coupled
MuJoCo + VBD solver for :func:`bike_chain_task.solver.make_mujoco_solver` (MuJoCo's own contacts,
no proxies) through the base's hooks; the LCM / offline / OSC subclasses reuse it as a mixin.
"""

from __future__ import annotations

import math
import time
from typing import ClassVar

import mujoco
import newton
import newton.examples
import numpy as np
import warp as wp
from loguru import logger

from bike_chain_task.constants import (
    CHAIN_CENTER,
    HOLDER_TIER_Z,
    PANDA_FINGER_LABELS,
    TABLE_TOP_Z,
)
from bike_chain_task.directives import CHAIN_CONTACT_MU
from bike_chain_task.joint_state import apply_default_joint_state
from bike_chain_task.scene import SceneInfo, build_scene
from bike_chain_task.solver import make_mujoco_solver
from task_common.joint_state import find_gripper_driver_indices
from task_common.replay_metrics import quat_multiply, quat_rotate
from task_common.simulation import BeltTaskSimulation
from utils.labels import joint_index

# V13-03 smoke bounds (handoffs/packages/PKG-20261022-bike-chain-sim.md).
MAX_GAP_P95_MM = 0.2
MIN_Z_ABOVE_TIER_MM = -2.0
MAX_ARM_DRIFT_RAD = 1e-3
MAX_ROBOT_PENETRATION_MM = 1.0
MIN_RTF = 0.59 * 0.8  # spike S6 (full scene, magna-exact chain, aligned board) - 20 %


class _BikeChainTask:
    """Solver hooks, chain pose reset and rest metrics shared by every bike-chain sim class."""

    belt_center = CHAIN_CENTER
    table_top_z = TABLE_TOP_Z
    # V13-02b spike settings (make_mujoco_solver defaults; data/bike_chain/v13_02b/spike.md).
    solver_kwargs: ClassVar[dict] = {"iterations": 30, "ls_iterations": 10, "nconmax": 2048,
                                     "njmax": 8192, "impratio": 1.0}
    # Opt-in pad contact (condim 4/6 + torsional [m] / rolling [m] friction for the pads' area).
    # V13-02b: condim 4 or 6 did not stop the lifted ring tilting (44-50 deg), so default off.
    pad_condim: ClassVar[int] = 3
    pad_mu_torsional: ClassVar[float] = 0.01
    pad_mu_rolling: ClassVar[float] = 1.0e-4
    # MuJoCo friction is soft (creeps under steady load): dmax nearer 1 stiffens the pad rows;
    # solmix >> 1 makes the pad's solref/solimp win the pair mix.
    pad_solimp: ClassVar[tuple] = (0.9, 0.95, 0.001, 0.5, 2.0)
    pad_solmix: ClassVar[float] = 1000.0
    # Pads at 0.02 s sank 2.2 mm into a link at 16 N, so the drives never reached their force
    # limits; 0.004 s (>= 2 dt at the OSC's 1.25 ms) is ~25x stiffer. solmix makes it win the mix.
    pad_solref: ClassVar[tuple] = (0.004, 1.0)
    # Real grasp forces (V13-05): the round belt's drives squeezed a 7 mm link with ~3.5 N
    # (Franka) / ~13 N (2F-85) and the lifted ring slipped out. Franka hand ~70 N continuous:
    # force-limited closure (the demo's 3.6 mm command on a ~7.2 mm link demands ~45 N/finger,
    # clamped at 35). 2F-85 (20-235 N): the driver torque limit sets the squeeze (~6.5 N per N m).
    finger_drive_ke: ClassVar[float] = 25000.0  # N/m per finger
    finger_drive_kd: ClassVar[float] = 160.0
    finger_effort_limit: ClassVar[float] = 35.0  # N per finger -> ~70 N squeeze
    gripper_drive_ke: ClassVar[float] = 1000.0  # N m/rad on the 2F-85 driver joints
    gripper_effort_limit: ClassVar[float] = 15.0  # N m -> ~100 N squeeze

    def _build_scene(self, builder: newton.ModelBuilder) -> SceneInfo:
        return build_scene(builder)

    def _build_solver(self, info: SceneInfo):
        solver = make_mujoco_solver(
            self.model, solref_overrides=((info.contact_override_shapes, self.pad_solref),),
            **self.solver_kwargs)
        mj = solver.mj_model
        to_newton = solver.mjc_body_to_newton.numpy()[0]
        chain = set(info.belt_bodies)
        connect = [k for k in range(mj.neq) if mj.eq_type[k] == mujoco.mjtEq.mjEQ_CONNECT
                   and {int(to_newton[mj.eq_obj1id[k]]), int(to_newton[mj.eq_obj2id[k]])} <= chain]
        logger.info(f"[MUJOCO] chain bodies {len(info.belt_bodies)}, sprocket bodies "
                    f"{len(info.pulley_bodies)}, neq {mj.neq} (chain closure CONNECT "
                    f"{len(connect)}), nv {mj.nv}, nu {mj.nu}, ngeom {mj.ngeom}; "
                    f"{self.solver_kwargs}")
        return solver

    def _apply_contact_materials(self, info: SceneInfo) -> None:
        """The base cable/pad materials, then steel friction back on the chain and sprockets."""
        chain_mu = self.model.shape_material_mu.numpy()[info.belt_shapes]
        super()._apply_contact_materials(info)
        # MuJoCo pairs take max(mu): at the cable's 1.0 on either side links stick to the teeth.
        mu = self.model.shape_material_mu.numpy()
        mu[info.belt_shapes] = chain_mu
        mu[info.pulley_shapes] = CHAIN_CONTACT_MU
        self.model.shape_material_mu.assign(mu)
        pads = info.contact_override_shapes  # 2F-85 pads + Franka fingers; MuJoCo pairs use max
        for array, value in ((self.model.mujoco.condim, self.pad_condim),
                             (self.model.mujoco.geom_solimp, self.pad_solimp),
                             (self.model.mujoco.geom_solmix, self.pad_solmix),
                             (self.model.shape_material_mu_torsional, self.pad_mu_torsional),
                             (self.model.shape_material_mu_rolling, self.pad_mu_rolling)):
            host = array.numpy()
            host[pads] = value
            array.assign(host)
        logger.info(f"[CONTACT] pads/fingers {pads}: condim {self.pad_condim}, torsional "
                    f"{self.pad_mu_torsional:g} m, rolling {self.pad_mu_rolling:g} m, solimp "
                    f"{self.pad_solimp}, solmix {self.pad_solmix:g}")

    def _apply_pulley_damping(self, info: SceneInfo) -> None:
        """No-op: MuJoCo applies ``joint_damping`` natively; the VBD ke-0 drive would double it."""

    def _pin_pulley_joint_stiffness(self, info: SceneInfo) -> None:
        """No-op: no VBD entry."""

    def _pin_belt_stretch_stiffness(self, info: SceneInfo) -> None:
        """No-op: no VBD entry."""

    def _simulate_physics(self) -> None:
        """The base frame without ``model.collide``: MuJoCo collides (``contacts`` is ignored)."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset_chain_pose(self, xy_offset=(0.0, 0.0), yaw_rad: float = 0.0) -> None:
        """Rotate the chain by ``yaw_rad`` about ``belt_center`` (world z), then shift it by
        ``xy_offset``; chain velocities zeroed.  Goes through the free joint's ``joint_q``
        (MuJoCo derives ``body_q``), never ``reset_body_poses``."""
        free = self.info.belt_joints[0]
        q0 = int(self.model.joint_q_start.numpy()[free])
        d0 = int(self.model.joint_qd_start.numpy()[free])
        d1 = int(self.model.joint_qd_start.numpy()[self.info.belt_joints[-1] + 1]) if (
            self.info.belt_joints[-1] + 1 < self.model.joint_count) else self.model.joint_dof_count
        rot = np.array([0.0, 0.0, math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad)])
        c = np.array([*self.belt_center[:2], 0.0])
        joint_q, joint_qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()
        p, q = joint_q[q0:q0 + 3].astype(np.float64), joint_q[q0 + 3:q0 + 7].astype(np.float64)
        joint_q[q0:q0 + 3] = quat_rotate(rot, p - c) + c + [*xy_offset, 0.0]
        joint_q[q0 + 3:q0 + 7] = quat_multiply(rot, q)
        joint_qd[d0:d1] = 0.0
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.solver.reset(self.state_0, flags=0)
        self.physics_graph = None

    def belt_positions(self) -> np.ndarray:
        """Chain link origins (pin k of link k), world xyz."""
        return self.state_0.body_q.numpy()[self.info.belt_bodies, :3].astype(np.float64)

    def chain_rest_metrics(self) -> dict:
        """NaN, pin gaps (incl. the closing pin), lowest link origin above the holder tier, arm
        drift from the scene defaults, deepest robot contact [mm]."""
        model, body_q = self.model, self.state_0.body_q.numpy().astype(np.float64)
        jt = model.joint_type.numpy()
        pins = [j for j in self.info.belt_joints if jt[j] == int(newton.JointType.REVOLUTE)]
        xp, xc = model.joint_X_p.numpy()[pins], model.joint_X_c.numpy()[pins]
        bp = body_q[model.joint_parent.numpy()[pins]]
        bc = body_q[model.joint_child.numpy()[pins]]
        gaps = np.linalg.norm(
            bp[:, :3] + quat_rotate(bp[:, 3:7], xp[:, :3])
            - bc[:, :3] - quat_rotate(bc[:, 3:7], xc[:, :3]), axis=1) * 1e3
        z = body_q[self.info.belt_bodies, 2]
        cfg = self.info.joint_config
        jq = self.state_0.joint_q.numpy()
        drift = np.abs(jq[cfg["arm_coord_indices"]] - np.asarray(cfg["arm_defaults"]))
        d = self.solver.mjw_data
        n = int(d.nacon.numpy()[0])
        pen = 0.0
        if n:
            shape = self.solver.mjc_geom_to_newton_shape.numpy()[0][d.contact.geom.numpy()[:n]]
            robot = np.isin(model.shape_body.numpy()[shape], self.info.robot_bodies)
            hit = robot[:, 0] ^ robot[:, 1]
            if hit.any():
                pen = max(0.0, -float(d.contact.dist.numpy()[:n][hit].min()) * 1e3)
        support = TABLE_TOP_Z if HOLDER_TIER_Z is None else HOLDER_TIER_Z
        return {"nan": not np.isfinite(body_q).all(), "gap_p95_mm": float(np.percentile(gaps, 95)),
                "gap_max_mm": float(gaps.max()), "closure_gap_mm": float(gaps[-1]),
                "min_z_above_tier_mm": float((z.min() - support) * 1e3),
                "arm_drift_rad": float(drift.max()), "robot_penetration_mm": pen, "nacon": n}


class BikeChainTaskSimulation(_BikeChainTask, BeltTaskSimulation):
    """The bike-chain scene held at its default configuration (viewer / headless)."""

    def __init__(self, viewer: newton.viewer.ViewerBase, args) -> None:
        self._physics_wall = 0.0
        self._timed_frames = 0
        super().__init__(viewer, args)

    def _apply_default_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        apply_default_joint_state(model, info)
        cfg = info.joint_config
        qd = model.joint_qd_start.numpy()
        labels = list(model.joint_label)
        fingers = [int(qd[joint_index(labels, n)]) for n in PANDA_FINGER_LABELS]
        drivers = find_gripper_driver_indices(model, info.gripper_joints[0],
                                              info.gripper_joints[-1] + 1)[1]
        for array, idx, value in (
                (model.joint_target_ke, cfg["finger_target_indices"], self.finger_drive_ke),
                (model.joint_target_kd, cfg["finger_target_indices"], self.finger_drive_kd),
                (model.joint_target_ke, cfg["gripper_target_indices"], self.gripper_drive_ke),
                (model.joint_effort_limit, fingers, self.finger_effort_limit),
                (model.joint_effort_limit, drivers, self.gripper_effort_limit)):
            host = array.numpy()
            host[list(idx)] = value
            array.assign(host)

    def _step_physics(self) -> None:
        t0 = time.perf_counter()
        warm = self.frame_id == 0 or (self.use_cuda_graph and self.physics_graph is None)
        super()._step_physics()
        wp.synchronize_device(self.device)
        if not warm:  # uncaptured warm-up + capture frames are not timed
            self._physics_wall += time.perf_counter() - t0
            self._timed_frames += 1

    def rtf(self) -> float:
        """Physics real-time factor over the captured frames."""
        return self._timed_frames * self.frame_dt / max(self._physics_wall, 1e-9)

    def test_final(self) -> None:
        m = self.chain_rest_metrics()
        rtf = self.rtf()
        logger.info(
            f"[TEST] t {self.sim_time:.2f} s: chain gap p95 {m['gap_p95_mm']:.4f} mm max "
            f"{m['gap_max_mm']:.4f} mm (closure {m['closure_gap_mm']:.4f}), min z above tier "
            f"{m['min_z_above_tier_mm']:+.3f} mm, arm drift {m['arm_drift_rad']:.2e} rad, robot "
            f"penetration {m['robot_penetration_mm']:.3f} mm, contacts {m['nacon']}, physics RTF "
            f"{rtf:.2f} over {self._timed_frames} frames")
        assert not m["nan"], "non-finite body transforms"
        assert m["gap_p95_mm"] < MAX_GAP_P95_MM, f"pin gap p95 {m['gap_p95_mm']:.3f} mm"
        assert m["min_z_above_tier_mm"] > MIN_Z_ABOVE_TIER_MM, "chain sank below the holder tier"
        assert m["arm_drift_rad"] < MAX_ARM_DRIFT_RAD, f"arm drift {m['arm_drift_rad']:.2e} rad"
        assert m["robot_penetration_mm"] <= MAX_ROBOT_PENETRATION_MM, "robot-world penetration"
        assert rtf >= MIN_RTF, f"physics RTF {rtf:.2f} < {MIN_RTF:.2f}"
        logger.success("[TEST] test_final passed.")
