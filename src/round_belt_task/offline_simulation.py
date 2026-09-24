"""The round-belt sim driven in-process: no magna, no LCM socket, no belt teleport.

:class:`RoundBeltOfflineSimulation` swaps the LCM bridge for :class:`OfflineBridge` (zero arm
torques, gripper commands set from Python), puts the arm joints under a position drive and plays
a :class:`motion.JointTrajectory` step by step. The Drake belt trigger is disabled: the belt is
picked physically by closing both grippers.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
from loguru import logger
from newton import JointTargetMode

from round_belt_task import clearance
from round_belt_task.arm_kinematics import (
    FrankaTip,
    UrTracking,
    model_pose_franka_tip,
    model_pose_ur_tracking,
    rot_angle,
)
from round_belt_task.constants import ARM_TARGET_KD, ARM_TARGET_KE
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from round_belt_task.motion import (
    JointTrajectory,
    build_cartesian_trajectory,
    solve_joint_trajectory,
)
from round_belt_task.waypoints import (
    MAGNA_PARAMS_SIM_YAML,
    Waypoint,
    load_pre_mpc_segment,
)
from task_common import sim_snapshot
from task_common.cameras import RgbdCameras
from task_common.joint_state import index_layout
from task_common.point_cloud import CroppedPointCloud
from task_common.scene import SceneInfo

OFFLINE_URL = "offline"
HOLDER_RIM_TOP_Z = -0.01858
LIFT_CLEARANCE = 0.03
GRASP_GAP_MM = 15.0
FRANKA_WIDTH_MM = (1.0, 10.0)
UR_CLOSED_BYTE = 255
ROBOTIQ_HOLD_STATUS = 200
PICK_FIRST, PICK_LAST = "pre_pick_0", "pre_place_1"
PICK_HOLD_S = 1.0
JAW_SETTLE_STEPS = 100  # 0.5 s: the Robotiq drive ramps at 0.1 m/s


class OfflineBridge:
    """Socket-free stand-in for :class:`LcmBridge` (same read API)."""

    url = OFFLINE_URL

    def __init__(self) -> None:
        self.sim_time = 0.0
        self.subscribed: list[str] = []
        self._robotiq: tuple[int, int, int] | None = None
        self._hand_mm: float | None = None

    def set_robotiq(self, byte: int, speed: int = 255, force: int = 0) -> None:
        self._robotiq = (int(byte), int(speed), int(force))

    def set_hand_mm(self, mm: float) -> None:
        self._hand_mm = float(mm)

    def drain(self, sim_time: float) -> int:
        self.sim_time = sim_time
        return 0

    def latest_efforts(self, spec) -> tuple[np.ndarray, float]:
        return np.zeros(len(spec.effort_names)), 0.0

    def latest_robotiq(self) -> tuple[int, int, int] | None:
        return self._robotiq

    def latest_hand_command(self) -> tuple[int, float, float] | None:
        if self._hand_mm is None:
            return None
        # Stamped with the current sim time so the stale check never trips.
        return round(self.sim_time * 1e6), self._hand_mm, 0.0

    def publish(self, channel: str, msg: object) -> None:
        pass

    def take_targets(self) -> list:
        return []


@dataclass
class PlayResult:
    """Per-step tracking errors (commanded vs measured after the step) and move-phase stats."""

    franka_mm: np.ndarray
    franka_deg: np.ndarray
    ur_mm: np.ndarray
    ur_deg: np.ndarray
    move: np.ndarray
    steps: int
    wall_s: float

    def summary(self) -> dict:
        out = {}
        m = self.move if self.move.any() else np.ones_like(self.move)
        for name in ("franka_mm", "franka_deg", "ur_mm", "ur_deg"):
            v = getattr(self, name)[m]
            out[f"{name}_rms"] = float(np.sqrt(np.mean(v * v)))
            out[f"{name}_max"] = float(v.max())
        return out

    def describe(self) -> str:
        s = self.summary()
        return (f"franka {s['franka_mm_rms']:.2f}/{s['franka_mm_max']:.2f} mm "
                f"{s['franka_deg_rms']:.2f}/{s['franka_deg_max']:.2f} deg, ur "
                f"{s['ur_mm_rms']:.2f}/{s['ur_mm_max']:.2f} mm "
                f"{s['ur_deg_rms']:.2f}/{s['ur_deg_max']:.2f} deg (rms/max over moves)")


@dataclass
class GraspState:
    """The ``[GRASP]`` log's numbers, plus the belt's lowest point."""

    franka_width_mm: float
    franka_tip_gap_mm: float
    ur_byte: int | None
    ur_status: int
    ur_tip_gap_mm: float
    ur_pad_gaps_mm: tuple[float, ...]
    belt_min_z: float

    def failures(self, min_belt_z: float | None = HOLDER_RIM_TOP_Z + LIFT_CLEARANCE
                 ) -> list[str]:
        """Why this is not a two-handed hold (empty = held)."""
        out = []
        lo, hi = FRANKA_WIDTH_MM
        if not lo <= self.franka_width_mm <= hi:
            out.append(f"franka width {self.franka_width_mm:.1f} mm not in [{lo:g}, {hi:g}]")
        if self.franka_tip_gap_mm > GRASP_GAP_MM:
            out.append(f"belt->finger_tip {self.franka_tip_gap_mm:.1f} mm > {GRASP_GAP_MM:g}")
        if self.ur_byte != UR_CLOSED_BYTE:
            out.append(f"robotiq byte {self.ur_byte} != {UR_CLOSED_BYTE}")
        # The fingertip point, not the pad origins: the finger colliders reach ~74 mm past them.
        if self.ur_tip_gap_mm > GRASP_GAP_MM:
            out.append(f"belt->2f85 tip {self.ur_tip_gap_mm:.1f} mm > {GRASP_GAP_MM:g}")
        if min_belt_z is not None and self.belt_min_z <= min_belt_z:
            out.append(f"belt min z {self.belt_min_z:.4f} <= {min_belt_z:.4f}")
        return out

    def held(self) -> tuple[bool, bool]:
        """(Franka, UR) each holding the belt: width/status in range and tip within the gap."""
        lo, hi = FRANKA_WIDTH_MM
        return (lo <= self.franka_width_mm <= hi and self.franka_tip_gap_mm <= GRASP_GAP_MM,
                self.ur_status >= ROBOTIQ_HOLD_STATUS and self.ur_tip_gap_mm <= GRASP_GAP_MM)

    def describe(self) -> str:
        pads = "/".join(f"{g:.1f}" for g in self.ur_pad_gaps_mm)
        return (f"hand width {self.franka_width_mm:.1f} mm, belt->finger_tip "
                f"{self.franka_tip_gap_mm:.1f} mm, robotiq byte {self.ur_byte} (status "
                f"{self.ur_status}), belt->2f85 tip {self.ur_tip_gap_mm:.1f} mm, pads {pads} mm, "
                f"belt min z {self.belt_min_z:.4f}")


class RoundBeltOfflineSimulation(RoundBeltLcmSimulation):
    """In-process round-belt sim: position-driven arms, stub bridge, no belt trigger."""

    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace, *,
                 arm_ke: float = ARM_TARGET_KE, arm_kd: float = ARM_TARGET_KD,
                 velocity_lead: bool = True) -> None:
        self.arm_ke = float(arm_ke)
        self.arm_kd = float(arm_kd)
        self.velocity_lead = bool(velocity_lead)
        args.cameras = False  # cameras are rendered on demand, never per step
        args.lcm_url = OFFLINE_URL
        super().__init__(viewer, args)
        self._rgbd: RgbdCameras | None = None
        self._clouds: dict[str, CroppedPointCloud] = {}
        logger.info(f"[OFFLINE] arms position-driven ke {self.arm_ke:g} kd {self.arm_kd:g}, "
                    f"velocity lead {self.velocity_lead}, belt trigger off")

    def _make_bridge(self) -> OfflineBridge:
        return OfflineBridge()

    def _after_control_step(self, body_q: np.ndarray) -> None:
        pass  # no belt teleport trigger: the pick is physical

    def _apply_default_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        super()._apply_default_joint_state(model, info)
        slots = np.asarray(info.joint_config["arm_target_indices"], dtype=np.int64)
        for array, value in ((model.joint_target_ke, self.arm_ke),
                             (model.joint_target_kd, self.arm_kd),
                             (model.joint_target_mode, int(JointTargetMode.POSITION))):
            values = array.numpy().copy()
            values[slots] = value
            array.assign(values)

    def _seed_control_targets(self, cfg) -> None:
        super()._seed_control_targets(cfg)
        layout = index_layout(self._target_q_host.size, int(self.model.joint_coord_count),
                              int(self.model.joint_dof_count), "control.joint_target_q")
        franka, ur = self._io_by_spec_name("franka"), self._io_by_spec_name("ur10")
        self._franka_slots = franka.coords if layout == "coord" else franka.dofs
        self._ur_slots = ur.coords if layout == "coord" else ur.dofs

    # --- commands ---------------------------------------------------------------------------

    def set_arm_targets(self, q_franka, q_ur) -> None:
        host = self._target_q_host.reshape(-1)
        host[self._franka_slots] = q_franka
        host[self._ur_slots] = q_ur
        self.control.joint_target_q.assign(self._target_q_host)

    def arm_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """The arm position targets currently in ``control`` (float64 copies)."""
        host = self.control.joint_target_q.numpy().reshape(-1).astype(np.float64)
        return host[self._franka_slots].copy(), host[self._ur_slots].copy()

    def arm_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """``joint_q`` indices of the Franka and UR arm joints."""
        return self._io_by_spec_name("franka").coords, self._io_by_spec_name("ur10").coords

    def arm_positions(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.state_0.joint_q.numpy().astype(np.float64)
        return (q[self._io_by_spec_name("franka").coords].copy(),
                q[self._io_by_spec_name("ur10").coords].copy())

    def arm_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        qd = self.state_0.joint_qd.numpy().astype(np.float64)
        return (qd[self._io_by_spec_name("franka").dofs].copy(),
                qd[self._io_by_spec_name("ur10").dofs].copy())

    def set_grippers(self, franka_mm: float | None, ur_byte: int | None) -> None:
        if franka_mm is not None and not math.isnan(franka_mm):
            self.bridge.set_hand_mm(franka_mm)
        if ur_byte is not None and ur_byte >= 0:
            self.bridge.set_robotiq(ur_byte)

    def gripper_commands(self) -> tuple[float | None, int | None]:
        """The commands in force (hand mm, Robotiq byte), None if never set."""
        return self.bridge._hand_mm, self.robotiq_position_byte

    def restore(self, snap) -> None:
        """``sim_snapshot.restore`` plus the stub bridge's commands from the snapshot."""
        sim_snapshot.restore(self, snap)
        self.after_restore()

    def after_restore(self) -> None:
        """Re-seed the stub bridge and the Robotiq status from a restored snapshot."""
        self.bridge.set_hand_mm(self.hand_target_mm)
        if self.robotiq_position_byte is not None:
            self.bridge.set_robotiq(self.robotiq_position_byte)
        # A restore leaves robotiq_status_byte stale; the offline publish only refreshes it.
        self._publish_state(self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy())

    # --- measurements -------------------------------------------------------------------------

    def ee_poses(self, body_q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Measured (Franka finger_tip, UR tracking frame) world 4x4 poses."""
        body_q = self.state_0.body_q.numpy() if body_q is None else body_q
        return (model_pose_franka_tip(body_q, self._finger_tip_body),
                model_pose_ur_tracking(body_q, self._ur_wrist_body))

    def grasp_state(self, body_q: np.ndarray | None = None) -> GraspState:
        body_q = self.state_0.body_q.numpy() if body_q is None else body_q
        belt = body_q[self.info.belt_bodies, :3].astype(np.float64)

        def gap_mm(point) -> float:
            return float(np.linalg.norm(belt - np.asarray(point, float), axis=1).min()) * 1e3

        q = self.state_0.joint_q.numpy()
        hand = q[self._io_by_spec_name("franka_hand").coords]
        return GraspState(
            franka_width_mm=float(-hand[0] + hand[1]) * 1e3,
            franka_tip_gap_mm=gap_mm(body_q[self._finger_tip_body, :3]),
            ur_byte=self.robotiq_position_byte, ur_status=int(self.robotiq_status_byte),
            ur_tip_gap_mm=gap_mm(self._grasp_ur_tip_point(body_q)),
            ur_pad_gaps_mm=tuple(gap_mm(body_q[b, :3]) for b in self.info.gripper_pad_bodies),
            belt_min_z=float(belt[:, 2].min()),
        )

    def clearance_geometry(self, jaw_bytes: tuple[int, ...] = ()
                           ) -> tuple[clearance.Board, clearance.Gripper]:
        """``(board colliders, 2F-85 colliders)`` for :mod:`round_belt_task.clearance`.

        Every byte in ``jaw_bytes`` is driven to rest (arm held still) and added to the gripper's
        frozen point set, then the state is restored; the jaws open at ``place_3`` and the
        closed-jaw geometry alone under-predicts the clearance there by several mm.
        """
        body_q = self.state_0.body_q.numpy().astype(np.float64)
        X_track = model_pose_ur_tracking(body_q, self._ur_wrist_body)
        poses = [body_q]
        if jaw_bytes:
            snap = sim_snapshot.capture(self, "clearance-geometry")
            for byte in jaw_bytes:
                self.set_grippers(None, int(byte))
                for _ in range(JAW_SETTLE_STEPS):
                    self.control_step()
                poses.append(self.state_0.body_q.numpy().astype(np.float64))
            self.restore(snap)
        return (clearance.board_obstacles(self.model, body_q),
                clearance.gripper_geometry(self.model, poses, X_track,
                                           jaw_keys=tuple(int(b) for b in jaw_bytes)))

    def belt_positions(self) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.info.belt_bodies, :3].astype(np.float64)

    def commanded_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """FK of the current arm targets: where the last trajectory left the commands."""
        q_franka, q_ur = self.arm_targets()
        return FrankaTip.fk(q_franka), UrTracking.fk(q_ur)

    @property
    def rgbd_cameras(self) -> RgbdCameras:
        if self._rgbd is None:
            self._rgbd = RgbdCameras(self.model, self.info.cameras)
            self._clouds = {spec.name: CroppedPointCloud(self._rgbd, spec)
                            for spec in self.info.point_clouds}
        return self._rgbd

    def point_cloud(self, name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Render the cameras now; ``(xyz, rgb)`` of point cloud ``name`` (default: first)."""
        cameras = self.rgbd_cameras
        cameras.update(self.state_0)
        cloud = self._clouds[name] if name is not None else next(iter(self._clouds.values()))
        return cloud.compute()

    # --- playback -----------------------------------------------------------------------------

    def play(self, traj: JointTrajectory, on_step: Callable | None = None,
             log_every_s: float = 1.0) -> PlayResult:
        """Step through ``traj``; ``on_step(i, info)`` after every control step."""
        n = len(traj)
        errs = np.zeros((4, n))
        lead_f = lead_u = np.zeros(1)
        if self.velocity_lead:
            # Feed-forward velocity: the PD's kd term otherwise lags a moving target.
            gain = self.arm_kd / self.arm_ke / traj.dt
            lead_f = gain * np.diff(traj.q_franka, axis=0, prepend=traj.q_franka[:1])
            lead_u = gain * np.diff(traj.q_ur, axis=0, prepend=traj.q_ur[:1])
        log_every = max(1, round(log_every_s / self.frame_dt)) if log_every_s > 0 else 0
        t0 = time.perf_counter()
        for i in range(n):
            q_f, q_u = traj.q_franka[i], traj.q_ur[i]
            if self.velocity_lead:
                q_f, q_u = q_f + lead_f[i], q_u + lead_u[i]
            self.set_arm_targets(q_f, q_u)
            mm, byte = float(traj.franka_gripper_mm[i]), int(traj.ur_gripper_byte[i])
            self.set_grippers(mm, byte)
            self.control_step()
            body_q = self.state_0.body_q.numpy()
            franka, ur = self.ee_poses(body_q)
            cmd_f, cmd_u = traj.franka_4x4[i], traj.ur_4x4[i]
            errs[:, i] = (np.linalg.norm(franka[:3, 3] - cmd_f[:3, 3]) * 1e3,
                          math.degrees(rot_angle(franka, cmd_f)),
                          np.linalg.norm(ur[:3, 3] - cmd_u[:3, 3]) * 1e3,
                          math.degrees(rot_angle(ur, cmd_u)))
            if on_step is not None:
                on_step(i, {"franka_cmd": cmd_f, "ur_cmd": cmd_u, "franka": franka, "ur": ur,
                            "phase": traj.phase_label(i), "franka_gripper_mm": mm,
                            "ur_gripper_byte": byte, "body_q": body_q})
            if log_every and (i + 1) % log_every == 0:
                logger.info(f"[PLAY] step {i + 1}/{n} {traj.phase_label(i)}: franka "
                            f"{errs[0, i]:.2f} mm {errs[1, i]:.2f} deg, ur {errs[2, i]:.2f} mm "
                            f"{errs[3, i]:.2f} deg")
        return PlayResult(franka_mm=errs[0], franka_deg=errs[1], ur_mm=errs[2], ur_deg=errs[3],
                          move=traj.move_mask(), steps=n, wall_s=time.perf_counter() - t0)

    def plan(self, waypoints: list[Waypoint], **kw) -> JointTrajectory:
        """Joint trajectory from the current commands (poses, grippers) through ``waypoints``;
        ``kw`` goes to :func:`motion.build_cartesian_trajectory`."""
        start_franka, start_ur = self.commanded_poses()
        franka_mm, ur_byte = self.gripper_commands()
        cart = build_cartesian_trajectory(waypoints, start_franka, start_ur, dt=self.frame_dt,
                                          start_franka_mm=franka_mm, start_ur_byte=ur_byte, **kw)
        return solve_joint_trajectory(cart, *self.arm_targets())

    def nominal_pick(self, params: Path = MAGNA_PARAMS_SIM_YAML, hold_s: float = PICK_HOLD_S,
                     log_every_s: float = 1.0) -> tuple[list[Waypoint], JointTrajectory,
                                                        PlayResult]:
        """Plan and play ``PICK_FIRST..PICK_LAST`` + ``hold_s`` from the current commands."""
        waypoints = load_pre_mpc_segment(params, first=PICK_FIRST, last=PICK_LAST)
        traj = self.plan(waypoints, settle_s=hold_s)
        return waypoints, traj, self.play(traj, log_every_s=log_every_s)

    # --- construction ---------------------------------------------------------------------------

    @classmethod
    def build(cls, args: argparse.Namespace | None = None, **kw) -> RoundBeltOfflineSimulation:
        """A headless instance without argparse: null viewer, not realtime, no cameras.

        ``kw`` keys that are constructor options (``arm_ke``, ``arm_kd``, ``velocity_lead``) go
        to the constructor; the rest override parsed arguments (e.g. ``initial_state``).
        """
        if args is None:
            args = cls.create_parser().parse_args(
                ["--viewer", "null", "--no-realtime", "--no-cameras"])
        options = {k: kw.pop(k) for k in ("arm_ke", "arm_kd", "velocity_lead") if k in kw}
        for key, value in kw.items():
            setattr(args, key, value)
        viewer = newton.viewer.ViewerNull(num_frames=getattr(args, "num_frames", 100))
        return cls(viewer, args, **options)

    def start_recording(self, root: Path, label: str) -> None:
        self.args.record_label = label
        self._start_recording(Path(root))

    def close(self, reason: str = "closed") -> None:
        self.close_recording(reason)
        self.viewer.close()
