"""The round-belt sim with the Franka torque-driven by magna's real Cartesian OSC, in lock-step.

:class:`RoundBeltOscSimulation` keeps the offline sim's in-process grippers and position-driven
UR, but leaves the Franka in torque mode: every control step publishes ``FRANKA_STATE`` (after
the tick's ``TARGET_CARTESIAN_POSE_TRAJECTORY`` from :attr:`commander_hook`) and the next step
blocks until the OSC's ``FRANKA_INPUT`` for exactly that state (:class:`OscBridge`). The OSC runs
as a child process on a private LCM URL (:class:`OscProcess`), started once per run.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import newton
import numpy as np
from loguru import logger
from newton import JointTargetMode

from round_belt_task.arm_kinematics import FrankaTip, model_pose_franka_tip
from round_belt_task.commander import CommanderParams, mat3_to_quat, saved_traj_message
from round_belt_task.constants import ARM_TARGET_KD, ARM_TARGET_KE
from round_belt_task.offline_simulation import GraspState, RoundBeltOfflineSimulation
from round_belt_task.osc_bridge import OscBridge
from task_common import sim_snapshot
from task_common.joint_state import index_layout
from task_common.osc_process import (
    DEFAULT_ARGS,
    MAGNA_ROOT,
    OSC_BINARY,
    OscProcess,
    check_private_url,
)
from task_common.scene import SceneInfo

OSC_SETTLE_STEPS = 100  # 0.5 s
DEFAULT_OSC_TIMEOUT_S = 5.0
FK_MATCH_TOL_M = 1e-6

# (step_index, osc_time_s, joint_q, body_q) -> lcmt_timestamped_saved_traj | None (keep the last)
CommanderHook = Callable[[int, float, np.ndarray, np.ndarray], object]


class RoundBeltOscSimulation(RoundBeltOfflineSimulation):
    """Franka on magna's OSC (lock-step, private URL); UR position-driven; grippers in-process."""

    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace, *,
                 lcm_url: str, osc_timeout_s: float = DEFAULT_OSC_TIMEOUT_S,
                 arm_ke: float = ARM_TARGET_KE, arm_kd: float = ARM_TARGET_KD,
                 bridge_cls: type[OscBridge] = OscBridge) -> None:
        self.osc_url = check_private_url(lcm_url)
        self._bridge_cls = bridge_cls
        self.osc_timeout_s = float(osc_timeout_s)
        self.osc: OscProcess | None = None
        self._closed = False
        super().__init__(viewer, args, arm_ke=arm_ke, arm_kd=arm_kd, velocity_lead=False)
        self.lcm_url = args.lcm_url = self.osc_url
        self.commander_hook: CommanderHook | None = self.make_hold_hook()
        self._check_fk()
        logger.info(f"[OSC] Franka torque-driven by magna's OSC over {self.osc_url} (lock-step, "
                    f"timeout {self.osc_timeout_s:g} s); UR position ke {self.arm_ke:g} kd "
                    f"{self.arm_kd:g}")

    def _make_bridge(self) -> OscBridge:
        return self._bridge_cls(self.osc_url, self.channels, timeout_s=self.osc_timeout_s,
                                alive_fn=lambda: self.osc is not None and self.osc.alive())

    def _apply_default_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        # Skip the offline override (both arms position-driven): torque mode on both arms first.
        super(RoundBeltOfflineSimulation, self)._apply_default_joint_state(model, info)
        ur = self._io_by_spec_name("ur10")
        layout = index_layout(model.joint_target_ke.shape[0], int(model.joint_coord_count),
                              int(model.joint_dof_count), "model.joint_target_ke")
        slots = ur.coords if layout == "coord" else ur.dofs
        arm = {int(i) for i in info.joint_config["arm_target_indices"]}
        if not {int(i) for i in slots} <= arm:
            raise RuntimeError(f"UR target slots {slots.tolist()} not in arm_target_indices")
        for array, value in ((model.joint_target_ke, self.arm_ke),
                             (model.joint_target_kd, self.arm_kd),
                             (model.joint_target_mode, int(JointTargetMode.POSITION))):
            values = array.numpy().copy()
            values[slots] = value
            array.assign(values)

    def _check_fk(self) -> None:
        q_franka, _ = self.arm_positions()
        fk = FrankaTip.fk(q_franka)
        body = model_pose_franka_tip(self.state_0.body_q.numpy(), self._finger_tip_body)
        err = float(np.linalg.norm(fk[:3, 3] - body[:3, 3]))
        if err > FK_MATCH_TOL_M:
            raise RuntimeError(f"FrankaTip.fk is {err * 1e3:.4f} mm off the Newton finger_tip "
                               f"body (tol {FK_MATCH_TOL_M * 1e3:g} mm)")

    # --- OSC process ------------------------------------------------------------------------------

    def start_osc(self, log_path: Path, warm_up_timeout_s: float = 60.0,
                  osc_args: tuple[str, ...] = DEFAULT_ARGS, binary: Path = OSC_BINARY,
                  cwd: Path = MAGNA_ROOT) -> float:
        """Launch the OSC on this sim's private URL and wait for its first reply; warm-up [s]."""
        if self.osc is not None and self.osc.alive():
            raise RuntimeError(f"OSC already running (pid {self.osc.pid})")
        if self.osc is not None:
            self.osc.stop()
        self.osc = OscProcess()
        self.osc.start(self.osc_url, Path(log_path), cwd=cwd, binary=binary, extra_args=osc_args)
        joint_q, joint_qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()
        # A fresh sim's published positions are zeros until its first control step.
        for io in self._robot_io:
            io.positions, io.velocities = joint_q[io.coords], joint_qd[io.dofs]
        warm = self.bridge.warm_up(lambda: self._publish_state(joint_q, joint_qd),
                                   timeout_s=warm_up_timeout_s)
        info = self.osc.describe()
        logger.info(f"[OSC] handshake: url {info['lcm_url']} pid {info['pid']} binary sha256 "
                    f"{info['binary_sha256'][:12]} warm-up {warm:.2f} s")
        return warm

    def stop_osc(self) -> int | None:
        return None if self.osc is None else self.osc.stop()

    def close(self, reason: str = "closed") -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stop_osc()
        finally:
            super().close(reason)

    # --- commands ---------------------------------------------------------------------------------

    def osc_time_s(self, step: int | None = None) -> float:
        return self.bridge.osc_time_s(self.step_index if step is None else step)

    def set_ur_target(self, q_ur) -> None:
        host = self._target_q_host.reshape(-1)
        host[self._ur_slots] = q_ur
        self.control.joint_target_q.assign(self._target_q_host)

    def set_arm_targets(self, q_franka, q_ur) -> None:
        """UR part only: the Franka is torque-driven by the OSC and ``q_franka`` is ignored."""
        self.set_ur_target(q_ur)

    def make_hold_hook(self, params: CommanderParams | None = None) -> CommanderHook:
        """magna's 2-knot hold at the measured ``finger_tip`` pose latched on its first call."""
        dt = (params or CommanderParams()).dt
        latched: list[np.ndarray] = []

        def hold_at_measured(step, t, joint_q, body_q):
            if not latched:
                pose = self.franka_measured_pose7(joint_q)
                latched.extend((pose[:3], pose[3:]))
            pos, quat = latched
            return saved_traj_message(round(t * 1e6), np.stack([pos, pos]),
                                      np.stack([quat, quat]), np.array([t, t + dt]))

        return hold_at_measured

    def _publish_state(self, joint_q: np.ndarray, joint_qd: np.ndarray) -> None:
        if self.commander_hook is not None:
            t = self.bridge.osc_time_s(self.step_index)
            msg = self.commander_hook(self.step_index, t, joint_q, self.state_0.body_q.numpy())
            if msg is not None:
                self.bridge.set_trajectory(msg)
        super()._publish_state(joint_q, joint_qd)

    # --- restore / settle -------------------------------------------------------------------------

    def restore(self, snap, settle_steps: int = OSC_SETTLE_STEPS) -> GraspState:
        """Snapshot restore -> OSC clock rebase -> hold at the restored pose -> settle."""
        sim_snapshot.restore(self, snap)
        self.bridge.rebase(self.step_index)
        self.commander_hook = self.make_hold_hook()
        self.after_restore()
        return self.settle(settle_steps)

    def settle(self, steps: int = OSC_SETTLE_STEPS) -> GraspState:
        for _ in range(int(steps)):
            self.control_step()
        return self.grasp_state()

    # --- measurements -----------------------------------------------------------------------------

    def franka_measured_pose7(self, joint_q: np.ndarray | None = None) -> np.ndarray:
        """``xyz`` + ``wxyz`` (``w >= 0``) of ``finger_tip`` = FK of the measured joints (what the
        OSC computes from ``FRANKA_STATE``); ``joint_q`` defaults to ``state_0``'s."""
        if joint_q is None:
            joint_q = self.state_0.joint_q.numpy()
        X = FrankaTip.fk(joint_q[self.arm_coords()[0]])
        return np.concatenate([X[:3, 3], mat3_to_quat(X[:3, :3])])

    def franka_efforts(self) -> np.ndarray:
        """The Franka joint efforts applied this step (the OSC's last reply)."""
        return self._io_by_spec_name("franka").efforts.copy()

    # --- construction -----------------------------------------------------------------------------

    @classmethod
    def build(cls, args: argparse.Namespace | None = None, *, lcm_url: str,
              **kw) -> RoundBeltOscSimulation:
        """Headless instance (null viewer, not realtime, no cameras); ``lcm_url`` is required.

        Constructor options: ``osc_timeout_s``, ``arm_ke``, ``arm_kd``, ``bridge_cls``; other
        ``kw`` override parsed arguments.
        """
        check_private_url(lcm_url)
        if args is None:
            args = cls.create_parser().parse_args(
                ["--viewer", "null", "--no-realtime", "--no-cameras"])
        options = {k: kw.pop(k) for k in ("osc_timeout_s", "arm_ke", "arm_kd", "bridge_cls")
                   if k in kw}
        for key, value in kw.items():
            setattr(args, key, value)
        args.lcm_url = lcm_url
        viewer = newton.viewer.ViewerNull(num_frames=getattr(args, "num_frames", 100))
        t0 = time.perf_counter()
        sim = cls(viewer, args, lcm_url=lcm_url, **options)
        logger.debug(f"[OSC] built in {time.perf_counter() - t0:.1f} s")
        return sim
