"""Real-time LCM simulation of a belt task that speaks the Drake ``magna_simulation`` contract.

:class:`LcmBeltTaskSimulation` runs :class:`BeltTaskSimulation` at a 5 ms control step in
torque mode: the arm joints take ``control.joint_f`` from LCM (MuJoCo gravity compensation
holds them), while the 2F-85 drivers and (with ``hand_drive``) the Panda fingers keep a
position drive and follow ``ROBOTIQ_COMMAND`` / ``PANDA_HAND_COMMAND``.  Every step publishes
the six state messages, plus the pulley state when ``pulley_state_object_name`` is set.
Subclasses supply the scene hooks, ``_robot_specs``, ``_reflected_inertia`` and
``_after_control_step``.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import warp as wp
from loguru import logger
from newton import JointTargetMode

from task_common.belt_mesh_lcm import deformable_mesh_msg, tube_mesh
from task_common.joint_state import find_gripper_driver_indices, index_layout
from task_common.lcm_bridge import LcmBridge
from task_common.lcm_contract import (
    BELT_MESH_COLOR,
    BELT_MESH_NAME,
    CONTROL_DT,
    DEFAULT_LCM_URL,
    ROBOTIQ_EFFORT_NAMES,
    ROBOTIQ_POSITION_NAMES,
    ROBOTIQ_PRISMATIC_RANGE,
    ROBOTIQ_VELOCITY_NAMES,
    HandDrive,
    LcmChannels,
    RobotIoSpec,
    hand_targets,
    object_state_msg,
    robot_output_msg,
    robotiq_status_msg,
    schunk_status_msg,
)
from task_common.recording import (
    DEFAULT_CHUNK_STEPS,
    DEFAULT_RECORDINGS_DIR,
    DEFAULT_STATE_EVERY,
    RunRecorder,
    git_state,
)
from task_common.scene import SceneInfo
from task_common.simulation import BeltTaskSimulation
from utils.labels import joint_index

STATS_PERIOD = 5.0
RESYNC_LAG = 0.25
SPIN_MARGIN = 0.0005
DRAKE_XMLNS = "{http://drake.mit.edu}"
BELT_MESH_PUBLISH_EVERY = 10  # 10 steps @ 5 ms = 20 Hz
GRASP_LOG_PERIOD = 1.0


def reflected_rotor_inertia(urdf: Path) -> dict[str, float]:
    """Drake's reflected actuator inertia ``rotor_inertia * gear_ratio**2`` per URDF joint."""
    inertia = {}
    for transmission in ET.parse(urdf).getroot().iter("transmission"):
        joint, actuator = transmission.find("joint"), transmission.find("actuator")
        if joint is None or actuator is None:
            continue
        rotor = actuator.find(f"{DRAKE_XMLNS}rotor_inertia")
        gear = actuator.find(f"{DRAKE_XMLNS}gear_ratio")
        if rotor is None:
            continue
        ratio = float(gear.get("value")) if gear is not None else 1.0
        inertia[joint.get("name")] = float(rotor.get("value")) * ratio**2
    return inertia


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not >= 1")
    return value


@dataclass
class _RobotIo:
    """A spec resolved against the model, plus this step's values."""

    spec: RobotIoSpec
    output_channel: str
    coords: np.ndarray
    dofs: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray
    stale: bool = False
    seen: bool = False


class LcmBeltTaskSimulation(BeltTaskSimulation):
    """A belt task driven over LCM by the magna controllers, one control step at a time."""

    frame_rate = round(1.0 / CONTROL_DT)
    solver_substeps: int = 2
    solver_vbd_iterations: int = 10
    gripper_drive_ke: float | None = None
    gripper_drive_kd: float | None = None
    gripper_drive_stop: float | None = None
    gripper_drive_effort_limit: float | None = None
    gripper_drive_damping: float | None = None
    gripper_drive_width_calibration: Sequence[tuple[float, float]] | None = None
    belt_radius: float | None = None
    hand_drive: HandDrive | None = None
    pulley_state_object_name: str | None = None

    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace) -> None:
        self.lcm_url = str(getattr(args, "lcm_url", DEFAULT_LCM_URL))
        channels_file = getattr(args, "lcm_channels_file", None)
        self.channels = LcmChannels.from_yaml(channels_file) if channels_file else LcmChannels()
        self.robot_specs = list(self._robot_specs())
        self.step_index = 0
        self.belt_placed = False
        self.robotiq_position_byte: int | None = None
        self.robotiq_force_byte = 0
        self.robotiq_status_byte = 0
        self.hand_target_mm = 0.0
        self._hand_stale = False
        self._hand_goal_logged: tuple[float, float] | None = None
        self.robotiq_status_speed_byte = 0
        self.robotiq_opening = 0.0
        self.recorder: RunRecorder | None = None
        self._event_step = 0
        self._last_compute_ms = 0.0
        self.publish_belt_mesh = bool(getattr(args, "publish_belt_mesh", False))
        if self.publish_belt_mesh and self.belt_radius is None:
            raise ValueError("--publish-belt-mesh requires the task to set belt_radius")

        super().__init__(viewer, args)

        self._utime_per_step = round(1e6 * self.frame_dt)
        self._grasp_log_every = max(1, round(GRASP_LOG_PERIOD / self.frame_dt))
        self._io_by_name = {io.spec.name: io for io in self._robot_io}
        self._resolve_pulleys()
        self._resolve_task_bodies()
        self._reset_stats(time.perf_counter())
        self._resyncs = 0
        record_dir = getattr(args, "record", None)
        self.bridge = self._make_bridge()
        self._log_startup()
        if record_dir is not None:
            self._start_recording(Path(record_dir))

    def _robot_specs(self) -> Sequence[RobotIoSpec]:
        raise NotImplementedError

    def _make_bridge(self) -> LcmBridge:
        """The command/state transport; subclasses may swap in a socket-free stand-in."""
        record_targets = getattr(self.args, "record", None) is not None
        return LcmBridge(self.lcm_url, self.channels, self.robot_specs,
                         record_targets=record_targets)

    def _resolve_pulleys(self) -> None:
        """Coords/dofs and dairlib names of ``info.pulley_joints``, and their initial centres."""
        joints = list(self.info.pulley_joints)
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        leaves = [str(self.model.joint_label[j]).rsplit("/", 1)[-1] for j in joints]
        self._pulley_coords = np.asarray([int(q_start[j]) for j in joints], dtype=np.int64)
        self._pulley_dofs = np.asarray([int(qd_start[j]) for j in joints], dtype=np.int64)
        self._pulley_position_names = leaves
        self._pulley_velocity_names = [f"{leaf}dot" for leaf in leaves]
        bodies = np.asarray(self.info.pulley_bodies, dtype=np.int64)
        self._pulley_bodies = bodies
        self._pulley_initial_centres = self.state_0.body_q.numpy()[bodies, :3].astype(np.float64)

    def _resolve_task_bodies(self) -> None:
        """Task body/point lookups on the finalized model, before the recorder starts."""

    def _recording_meta(self) -> dict:
        """Task-specific ``meta.json`` keys (scene files, trigger bodies and points)."""
        return {}

    def recorder_meta(self) -> dict:
        """The ``meta.json`` content :class:`RunRecorder` is built with."""
        model = self.model
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        signal_joints = sorted(self.info.robot_joints + self.info.pulley_joints)
        signal_coords = [c for j in signal_joints for c in range(q_start[j], q_start[j + 1])]
        signal_dofs = [d for j in signal_joints for d in range(qd_start[j], qd_start[j + 1])]
        commit, dirty = git_state()
        initial_state = getattr(self.args, "initial_state", None)

        def ints(values) -> list[int]:
            return [int(v) for v in values]

        robot_io = {}
        for io in self._robot_io:
            spec = io.spec
            key = spec.input_channel_key
            robot_io[spec.name] = {
                "input_channel": None if key is None else getattr(self.channels, key),
                "output_channel": io.output_channel, "coords": ints(io.coords),
                "dofs": ints(io.dofs), "position_names": list(spec.position_names),
                "velocity_names": list(spec.velocity_names),
                "effort_names": list(spec.effort_names),
            }
        return {
            "argv": list(sys.argv), "git_commit": commit, "git_dirty": dirty,
            "control_dt": float(self.frame_dt), "lcm_url": self.lcm_url,
            "channels": dataclasses.asdict(self.channels),
            "scene_directives": None, "lcm_sim_params": None,
            "initial_state": None if initial_state is None else str(initial_state),
            "body_labels": [str(label) for label in model.body_label],
            "body_q_layout": "xyz_xyzw",
            "joint_labels": [str(label) for label in model.joint_label],
            "joint_q_start": ints(q_start), "joint_qd_start": ints(qd_start),
            "joint_coord_count": int(model.joint_coord_count),
            "joint_dof_count": int(model.joint_dof_count),
            "signal_coords": ints(signal_coords), "signal_dofs": ints(signal_dofs),
            "belt_bodies": ints(self.info.belt_bodies),
            "pulley_bodies": ints(self.info.pulley_bodies),
            "pulley_joints": ints(self.info.pulley_joints),
            "pulley_coords": ints(self._pulley_coords), "pulley_dofs": ints(self._pulley_dofs),
            "gripper_pad_bodies": ints(self.info.gripper_pad_bodies),
            "franka_finger_bodies": ints(self.info.franka_finger_bodies),
            "driver_coords": ints(self._driver_coords), "driver_dofs": ints(self._driver_dofs),
            "hand_body": int(self._hand_body),
            "robot_io": robot_io,
            "effort_layout": [f"{io.spec.name}/{name}" for io in self._robot_io
                              for name in io.spec.effort_names],
            **self._recording_meta(),
        }

    def _start_recording(self, root: Path) -> None:
        state_every = int(getattr(self.args, "record_state_every", DEFAULT_STATE_EVERY))
        chunk_steps = int(getattr(self.args, "record_chunk_steps", DEFAULT_CHUNK_STEPS))
        label = getattr(self.args, "record_label", None) or "round_belt"
        meta = self.recorder_meta()
        self.recorder = recorder = RunRecorder(
            root, label, meta, state_every=state_every, chunk_steps=chunk_steps
        )
        mb_per_min = recorder.bytes_per_step * self.frame_rate * 60.0 / 1e6
        logger.info(
            f"[RECORD] {recorder.path} state every {state_every} steps "
            f"({self.frame_rate / state_every:g} Hz), chunk {chunk_steps} steps, "
            f"~{mb_per_min:.1f} MB/min (B {len(meta['body_labels'])}, "
            f"C' {len(meta['signal_coords'])}, D' {len(meta['signal_dofs'])}, "
            f"{recorder.bytes_per_step:.0f} B/step)"
        )

    def _event(self, kind: str, /, **data) -> None:
        if self.recorder is not None:
            step = self._event_step
            self.recorder.event(step, step * self.frame_dt, kind, **data)

    def close_recording(self, reason: str = "closed") -> None:
        """End the recording (idempotent): ``run_end`` event, final ``meta.json``, size log."""
        recorder = self.recorder
        if recorder is None or recorder.closed:
            return
        self._event_step = self.step_index
        self._event("run_end", reason=reason, step=self.step_index)
        recorder.close(finished=True, reason=reason)
        meta = recorder.meta
        logger.info(
            f"[RECORD] closed {recorder.path}: {meta['num_steps']} steps, "
            f"{sum(c['state_frames'] for c in meta['chunks'])} frames, "
            f"{recorder.size_bytes() / 1e6:.1f} MB (B {len(meta['body_labels'])}, "
            f"C' {len(meta['signal_coords'])}, D' {len(meta['signal_dofs'])}), reason {reason}"
        )

    def _record_step(self, t_start: float, handled: int, joint_q: np.ndarray,
                     joint_qd: np.ndarray, body_q: np.ndarray) -> None:
        recorder = self.recorder
        step, sim_time = self.step_index, self.sim_time
        for channel, msg in self.bridge.take_targets():
            recorder.target(step, sim_time, channel, self._target_payload(channel, msg))
        franka = self._io_by_name.get("franka")
        recorder.step(
            step=step, sim_time=sim_time, wall_time=t_start, joint_q=joint_q, joint_qd=joint_qd,
            efforts=[io.efforts for io in self._robot_io], hand_target_mm=self.hand_target_mm,
            hand_stale=self._hand_stale, franka_stale=franka is not None and franka.stale,
            robotiq_cmd=self.bridge.latest_robotiq(),
            robotiq_status=(self.robotiq_status_byte, self.robotiq_status_speed_byte,
                            self.robotiq_force_byte),
            robotiq_opening=self.robotiq_opening, lcm_rx=handled,
            compute_ms=self._last_compute_ms, body_q=body_q,
        )

    def _target_payload(self, channel: str, msg) -> dict:
        if channel == self.channels.ur_target_spatial_pose_channel:
            return {"utime": int(msg.utime), "position": [float(v) for v in msg.position],
                    "orientation_wxyz": [float(v) for v in msg.orientation]}
        blocks = {}
        for block in msg.saved_traj.trajectories:
            blocks[block.trajectory_name] = {
                "t": [float(v) for v in block.time_vec],
                "data": [[float(v) for v in row] for row in block.datapoints],
                "datatypes": [str(v) for v in block.datatypes],
            }
        return {"utime": int(msg.utime), "blocks": blocks}

    def _apply_task_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        raise NotImplementedError

    def _reflected_inertia(self) -> Mapping[str, float]:
        """Joint label -> armature [kg m^2]; Drake adds it to the mass matrix, Newton does not."""
        return {}

    def _after_control_step(self, body_q: np.ndarray) -> None:
        """Called with the host ``state_0.body_q`` after every step's publishes."""

    def _grasp_tip_body(self) -> int | None:
        """Body whose distance to the belt the ``[GRASP]`` log reports (None = skip it)."""
        return None

    def _grasp_ur_tip_point(self, body_q: np.ndarray) -> np.ndarray | None:
        """World 2F-85 fingertip point the ``[GRASP]`` log measures from (None = skip it)."""
        return None

    def _apply_default_joint_state(self, model: newton.Model, info: SceneInfo) -> None:
        """The task's defaults, then torque mode (no PD) on the arms, and on the Franka fingers
        unless ``hand_drive`` gives them a position drive."""
        self._apply_task_joint_state(model, info)
        cfg = info.joint_config
        joint_labels = list(model.joint_label)
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        drive = self.hand_drive
        finger_targets = np.asarray(cfg["finger_target_indices"], dtype=np.int64)
        torque_targets = np.asarray(cfg["arm_target_indices"], dtype=np.int64)
        if drive is None:
            torque_targets = np.concatenate([torque_targets, finger_targets])
        gripper_targets = np.asarray(cfg["gripper_target_indices"], dtype=np.int64)
        for array, targets, value in (
            (model.joint_target_ke, torque_targets, 0.0),
            (model.joint_target_kd, torque_targets, 0.0),
            (model.joint_target_mode, torque_targets, int(JointTargetMode.NONE)),
            (model.joint_target_ke, gripper_targets, self.gripper_drive_ke),
            (model.joint_target_kd, gripper_targets, self.gripper_drive_kd),
            (model.joint_target_ke, finger_targets, None if drive is None else drive.ke),
            (model.joint_target_kd, finger_targets, None if drive is None else drive.kd),
            (model.joint_target_mode, finger_targets,
             None if drive is None else int(JointTargetMode.POSITION)),
        ):
            if value is None:
                continue
            values = array.numpy().copy()
            values[targets] = value
            array.assign(values)

        # Without it the 5 ms-lagged Franka torque loop (even the stale damping) diverges.
        self._armature = dict(self._reflected_inertia())
        if drive is not None:
            hand_spec = next(s for s in self.robot_specs if s.name == "franka_hand")
            self._armature.update({label: drive.armature for label in hand_spec.joint_labels})
        armature = model.joint_armature.numpy().copy()
        for label, value in self._armature.items():
            armature[int(qd_start[joint_index(joint_labels, label)])] = value
        model.joint_armature.assign(armature)

        self._robot_io = []
        for spec in self.robot_specs:
            joints = [joint_index(joint_labels, label) for label in spec.joint_labels]
            n = len(joints)
            self._robot_io.append(_RobotIo(
                spec=spec, output_channel=getattr(self.channels, spec.output_channel_key),
                coords=np.asarray([int(q_start[j]) for j in joints], dtype=np.int64),
                dofs=np.asarray([int(qd_start[j]) for j in joints], dtype=np.int64),
                positions=np.zeros(n), velocities=np.zeros(n), efforts=np.zeros(n),
            ))

        self._finger_dofs = self._io_by_spec_name("franka_hand").dofs
        effort_limit = model.joint_effort_limit.numpy().copy()
        self._imported_finger_effort_limit = effort_limit[self._finger_dofs].copy()
        if drive is not None:
            # Before _build_solver: MuJoCo bakes forcerange into the model.
            effort_limit[self._finger_dofs] = drive.effort_limit
            model.joint_effort_limit.assign(effort_limit)
            for array, value in ((model.joint_limit_ke, drive.limit_ke),
                                 (model.joint_limit_kd, drive.limit_kd)):
                values = array.numpy().copy()
                values[self._finger_dofs] = value
                array.assign(values)

        start, end = info.gripper_joints[0], info.gripper_joints[-1] + 1
        coords, dofs = find_gripper_driver_indices(model, start, end_joint_index=end)
        if len(coords) != len(cfg["gripper_open_values"]):
            raise RuntimeError(f"2f85 drivers {coords} do not match the gripper open values")
        self._driver_labels = [
            joint_labels[j] for j in range(start, end) if int(q_start[j]) in coords
            and "driver_joint" in joint_labels[j].lower()
        ]
        self._driver_coords = np.asarray(coords, dtype=np.int64)
        self._driver_dofs = np.asarray(dofs, dtype=np.int64)
        self._driver_open = np.asarray(cfg["gripper_open_values"], dtype=np.float64)
        self._driver_span = model.joint_limit_upper.numpy()[self._driver_dofs] - self._driver_open
        # A mechanical stop, not a target clamp: the span (status fraction, grip force) is kept.
        # Damping is passive so it still acts while the PD torque is clamped on a stalled grip.
        for array, value in ((model.joint_limit_upper, self.gripper_drive_stop),
                             (model.joint_effort_limit, self.gripper_drive_effort_limit),
                             (model.joint_damping, self.gripper_drive_damping)):
            if value is None:
                continue
            values = array.numpy().copy()
            values[self._driver_dofs] = value
            array.assign(values)
        self._build_robotiq_width_map(model)

    def _build_robotiq_width_map(self, model: newton.Model) -> None:
        """Byte <-> driver angle through the measured jaw width, per the 2F-85 manual.

        The POSITION REQUEST register is quasi-linear in jaw WIDTH (0x00 open, 0xFF closed,
        0.4 mm per count of the 85 mm stroke), while the four-bar driver angle is not, so a
        byte commands ``open_gap * (1 - byte / 255)`` of pad gap and the calibration
        ``[driver angle, free-air pad gap]`` inverts that.  Without a calibration the two-point
        default reproduces the old linear-in-angle map.
        """
        cal = self.gripper_drive_width_calibration
        if cal is None:
            angles = self._driver_open[0] + np.array([0.0, self._driver_span[0]])
            fractions = np.array([0.0, 1.0])
        else:
            angles, gaps = (np.asarray(v, dtype=np.float64) for v in zip(*cal))
            if not (np.all(np.diff(angles) > 0.0) and np.all(np.diff(gaps) < 0.0)):
                raise RuntimeError(f"gripper_drive.width_calibration {cal} must rise in angle "
                                   "and fall in gap")
            if not np.allclose(angles[0], self._driver_open):
                raise RuntimeError(f"width_calibration opens at {angles[0]} rad but the drivers "
                                   f"open at {self._driver_open.tolist()}")
            stop = float(model.joint_limit_upper.numpy()[self._driver_dofs].max())
            if angles[-1] < stop - 1e-6:
                raise RuntimeError(f"width_calibration ends at {angles[-1]} rad, short of the "
                                   f"driver's upper limit {stop} (gripper_drive.stop)")
            fractions = 1.0 - gaps / gaps[0]
        self._robotiq_cal_angles = angles
        self._robotiq_cal_fractions = fractions
        self._robotiq_cal_slopes = np.gradient(fractions, angles)
        self._robotiq_open_gap = None if cal is None else float(cal[0][1])
        targets = np.interp(np.arange(256) / 255.0, fractions, angles) - angles[0]
        self._robotiq_targets = self._driver_open + targets[:, None]
        # 0xFF keeps the full-close target: the overdrive past the stop is what makes the grip.
        self._robotiq_targets[255] = self._driver_open + self._driver_span

    def _couple_hand_fingers(self, builder: newton.ModelBuilder) -> None:
        """With ``hand_drive``: ``panda_finger_joint2 = -panda_finger_joint1``, before finalize."""
        if self.hand_drive is None:
            return
        # As the real hand: two independent saturated fingers slide sideways under belt loads.
        hand_spec = next(s for s in self.robot_specs if s.name == "franka_hand")
        labels = list(builder.joint_label)
        first, second = (joint_index(labels, label) for label in hand_spec.joint_labels)
        builder.set_joint_mimic(second, first, (0.0, -1.0))

    def _io_by_spec_name(self, name: str) -> _RobotIo:
        return next(io for io in self._robot_io if io.spec.name == name)

    def _seed_control_targets(self, cfg) -> None:
        super()._seed_control_targets(cfg)
        n_dofs = int(self.model.joint_dof_count)
        # Allocated before graph capture: the replay reads this exact buffer every step.
        if self.control.joint_f is None:
            self.control.joint_f = wp.zeros(n_dofs, dtype=wp.float32, device=self.model.device)
        self.control.joint_f.zero_()
        self._joint_f_host = np.zeros(n_dofs, dtype=np.float32)

        self._target_q_host = self.control.joint_target_q.numpy().copy()
        layout = index_layout(
            self._target_q_host.size, int(self.model.joint_coord_count), n_dofs,
            "control.joint_target_q",
        )
        self._driver_target_slots = self._driver_coords if layout == "coord" else self._driver_dofs
        hand = self._io_by_spec_name("franka_hand")
        self._finger_target_slots = hand.coords if layout == "coord" else hand.dofs
        self._hand_target_q = np.asarray(hand_targets(self.hand_target_mm))
        self._hand_goal_q = self._hand_target_q.copy()
        self._hand_ramp_q = self._hand_target_q.copy()
        if self.hand_drive is not None:
            # Closed from step 1, like ShunkCommandToTrajectory's default command.
            self._target_q_host.reshape(-1)[self._finger_target_slots] = self._hand_target_q
            self.control.joint_target_q.assign(self._target_q_host)

    def _log_startup(self) -> None:
        realtime = getattr(self.args, "realtime", True)
        lines = [
            (f"[LCM] {self.lcm_url}: control dt {self.frame_dt * 1e3:g} ms, "
             f"{self.sim_substeps} substeps (dt {self.sim_dt * 1e3:g} ms), "
             f"vbd_iterations={self.vbd_iterations}, realtime={realtime}, "
             f"initial state {getattr(self.args, 'initial_state', None) or '(scene defaults)'}"),
            (f"{'robot':<12}{'lcm name':<24}{'newton joint':<42}{'coord':>6}{'dof':>6}"
             f"{'armature':>10}"),
        ]
        for io in self._robot_io:
            for name, label, coord, dof in zip(
                io.spec.position_names, io.spec.joint_labels, io.coords, io.dofs
            ):
                lines.append(
                    f"{io.spec.name:<12}{name:<24}{label:<42}{coord:>6}{dof:>6}"
                    f"{self._armature.get(label, 0.0):>10.4f}"
                )
        for i, label in enumerate(self._driver_labels):
            lines.append(
                f"{'robotiq':<12}{'(both: driver mean)':<24}{label.rsplit('/', 1)[-1]:<42}"
                f"{self._driver_coords[i]:>6}{self._driver_dofs[i]:>6}{'':>10}   target slot "
                f"{self._driver_target_slots[i]}, open {self._driver_open[i]:.4f}, "
                f"closed {self._driver_open[i] + self._driver_span[i]:.4f}, "
                f"stop {self.model.joint_limit_upper.numpy()[self._driver_dofs[i]]:.4f}, "
                f"ke {self.model.joint_target_ke.numpy()[self._driver_dofs[i]]:g} "
                f"kd {self.model.joint_target_kd.numpy()[self._driver_dofs[i]]:g}, "
                f"effort {self.model.joint_effort_limit.numpy()[self._driver_dofs[i]]:g} "
                f"damping {self.model.joint_damping.numpy()[self._driver_dofs[i]]:g}"
            )
        if self._robotiq_open_gap is None:
            lines.append("robotiq: byte -> driver angle is linear (no width calibration)")
        else:
            samples = " ".join(f"{b}:{self._robotiq_targets[b, 0]:.4f}"
                               for b in (0, 63, 127, 191, 255))
            lines.append(f"robotiq: byte -> jaw width is linear over a "
                         f"{self._robotiq_open_gap * 1e3:.2f} mm open gap "
                         f"({len(self._robotiq_cal_angles)}-point width_calibration); "
                         f"byte -> target [rad] {samples}")
        c = self.channels
        drive = self.hand_drive
        for io in self._robot_io:
            if io.spec.input_channel_key is None:
                inp = c.franka_hand_input_channel if drive is not None else "(no input)"
                rule = "position drive, efforts = drive law" if drive is not None else "zero torque"
                lines.append(f"{io.spec.name}: {inp} -> {io.output_channel}; {rule}")
                continue
            inp = getattr(self.channels, io.spec.input_channel_key)
            if io.spec.stale_damping is None:
                rule = "no input -> zero torque"
            else:
                rule = (
                    f"input older than {io.spec.stale_timeout:g} s sim time -> "
                    f"tau = -{np.asarray(io.spec.stale_damping).tolist()} * v"
                )
            lines.append(f"{io.spec.name}: {inp} -> {io.output_channel}; {rule}")
        if drive is not None:
            imported = "/".join(f"{v:g}" for v in self._imported_finger_effort_limit)
            lines.append(
                f"hand drive: {c.franka_hand_input_channel} target_position_mm p -> fingers "
                f"(-p, +p) / 2000 m, ke {drive.ke:g} kd {drive.kd:g}, effort limit "
                f"{drive.effort_limit:g} N (imported {imported}), slew {drive.max_speed:g} m/s, "
                f"limit ke {drive.limit_ke:g} kd {drive.limit_kd:g}, default 0 mm (closed), "
                f"> {drive.stale_timeout:g} s stale -> 0 mm, force ignored"
            )
        lines.append(
            f"franka_hand status: {c.franka_hand_state_channel}; robotiq: "
            f"{c.robotiq_command_channel} -> {c.robotiq_status_channel}, "
            f"{c.robotiq_robot_output_channel}"
        )
        lines.append(f"subscribed: {', '.join(self.bridge.subscribed)}")
        if self.pulley_state_object_name is not None:
            joints = ", ".join(
                f"{name} (coord {coord}, dof {dof})" for name, coord, dof in zip(
                    self._pulley_position_names, self._pulley_coords, self._pulley_dofs
                )
            )
            lines.append(
                f"pulleys: {c.round_belt_pulley_state_channel} object_name "
                f"{self.pulley_state_object_name!r}: {joints or 'none'}"
            )
        logger.info("\n".join(lines))

    def control_step(self) -> None:
        """One control step: drain inputs, apply torques/targets, step physics, publish."""
        t_start = time.perf_counter()
        handled = self.bridge.drain(self.sim_time)
        self._event_step = self.step_index + 1

        joint_f = self._joint_f_host
        for io in self._robot_io:
            if io.spec.input_channel_key is None:
                continue
            efforts, age = self.bridge.latest_efforts(io.spec)
            stale = io.spec.stale_damping is not None and age > io.spec.stale_timeout
            if stale:
                io.efforts = -io.spec.stale_damping * io.velocities
            else:
                io.efforts = efforts if efforts is not None else np.zeros(len(io.dofs))
            if efforts is not None and not io.seen:
                io.seen = True
                self._event("input_first", robot=io.spec.name,
                            channel=getattr(self.channels, io.spec.input_channel_key))
            if stale != io.stale:
                io.stale = stale
                logger.info(f"[LCM] {io.spec.name} input: {'stale-damping' if stale else 'fresh'}")
                self._event("input_stale", robot=io.spec.name, stale=stale)
            joint_f[io.dofs] = io.efforts
        self.control.joint_f.assign(joint_f)
        targets_changed = self._apply_robotiq_command()
        targets_changed |= self._apply_hand_command()
        if targets_changed:
            self.control.joint_target_q.assign(self._target_q_host)

        self._step_physics()
        self.step_index += 1
        self.frame_id = self.step_index
        self.sim_time = self.step_index * self.frame_dt
        self._update_cameras()

        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        body_q = self.state_0.body_q.numpy()
        if not (np.isfinite(joint_q).all() and np.isfinite(joint_qd).all()
                and np.isfinite(body_q).all()):
            raise RuntimeError(f"non-finite simulation state after step {self.step_index}")
        for io in self._robot_io:
            io.positions = joint_q[io.coords]
            io.velocities = joint_qd[io.dofs]
        if self.hand_drive is not None:
            self._update_hand_efforts()
        self._publish_state(joint_q, joint_qd)
        if self.publish_belt_mesh and self.step_index % BELT_MESH_PUBLISH_EVERY == 0:
            self._publish_belt_mesh(body_q)
        self._after_control_step(body_q)
        if self.step_index % self._grasp_log_every == 0:
            self._log_grasp(body_q, joint_q)
        if self.recorder is not None:
            self._record_step(t_start, handled, joint_q, joint_qd, body_q)

        compute = time.perf_counter() - t_start
        self._last_compute_ms = compute * 1e3
        self._stats_steps += 1
        self._stats_compute_sum += compute
        self._stats_compute_max = max(self._stats_compute_max, compute)

    def _apply_robotiq_command(self) -> bool:
        """Write the newest Robotiq target into the host target buffer; True if it changed."""
        command = self.bridge.latest_robotiq()
        if command is None:
            return False
        position, speed, self.robotiq_force_byte = command
        if position == self.robotiq_position_byte:
            return False
        self.robotiq_position_byte = position
        self._event("robotiq_command", position=position, speed=speed,
                    force=self.robotiq_force_byte)
        self._target_q_host.reshape(-1)[self._driver_target_slots] = self._robotiq_targets[position]
        return True

    def _apply_hand_command(self) -> bool:
        """``ShunkCommandToTrajectory`` goal, slewed into the host targets; True if written."""
        drive = self.hand_drive
        if drive is None:
            return False
        command = self.bridge.latest_hand_command()
        if command is not None:
            utime, target_mm, force = command
            # magna republishes the same goal every tick: log only goal changes.
            if (target_mm, force) != self._hand_goal_logged:
                self._hand_goal_logged = (target_mm, force)
                self._event("hand_command", utime=utime, target_mm=target_mm, force=force)
            lag = self.sim_time - utime / 1e6
            stale = abs(lag) > drive.stale_timeout
            if stale != self._hand_stale:
                self._hand_stale = stale
                self._event("hand_stale", stale=stale, lag_s=lag)
                channel = self.channels.franka_hand_input_channel
                if stale:
                    logger.warning(f"[LCM] {channel} stale by {lag:.3f} s: closing")
                else:
                    logger.info(f"[LCM] {channel} fresh again")
            self.hand_target_mm = 0.0 if stale else target_mm
            self._hand_goal_q = np.asarray(hand_targets(self.hand_target_mm))

        step = drive.max_speed * self.frame_dt
        delta = np.clip(self._hand_goal_q - self._hand_ramp_q, -step, step)
        self._hand_ramp_q = self._hand_ramp_q + delta
        # Velocity lead: the overdamped drive would otherwise trail the ramp by kd/ke * speed.
        target = self._hand_ramp_q + drive.kd / drive.ke * delta / self.frame_dt
        goal = self._hand_goal_q
        target = np.where(delta >= 0.0, np.minimum(target, goal), np.maximum(target, goal))
        if np.array_equal(target, self._hand_target_q):
            return False
        self._hand_target_q = target
        self._target_q_host.reshape(-1)[self._finger_target_slots] = target
        return True

    def hand_ramp_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(target, goal, ramp)`` finger targets of the hand slew, for ``sim_snapshot``."""
        return self._hand_target_q.copy(), self._hand_goal_q.copy(), self._hand_ramp_q.copy()

    def sync_after_restore(self, hand_target_q, hand_goal_q, hand_ramp_q) -> None:
        """Re-derive the host mirrors from a restored ``state_0``/``control`` (sim_snapshot)."""
        self._hand_target_q = np.array(hand_target_q, dtype=np.float64)
        self._hand_goal_q = np.array(hand_goal_q, dtype=np.float64)
        self._hand_ramp_q = np.array(hand_ramp_q, dtype=np.float64)
        self._target_q_host = self.control.joint_target_q.numpy().copy()
        self._joint_f_host = self.control.joint_f.numpy().copy()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        for io in self._robot_io:
            io.positions = joint_q[io.coords]
            io.velocities = joint_qd[io.dofs]
        if self.hand_drive is not None:
            self._update_hand_efforts()
        if self.cameras is not None:
            self._next_camera_time = self.sim_time
        self._reset_stats(time.perf_counter())

    def _update_hand_efforts(self) -> None:
        drive = self.hand_drive
        hand = self._io_by_name["franka_hand"]
        force = drive.ke * (self._hand_target_q - hand.positions) - drive.kd * hand.velocities
        hand.efforts = np.clip(force, -drive.effort_limit, drive.effort_limit)

    def _publish_state(self, joint_q: np.ndarray, joint_qd: np.ndarray) -> None:
        utime = self.step_index * self._utime_per_step
        publish = self.bridge.publish
        for io in self._robot_io:
            spec = io.spec
            publish(io.output_channel, robot_output_msg(
                utime, spec.position_names, io.positions, spec.velocity_names, io.velocities,
                spec.effort_names, io.efforts,
            ))
        hand = self._io_by_name["franka_hand"]
        publish(self.channels.franka_hand_state_channel,
                schunk_status_msg(utime, hand.positions, hand.velocities))

        # Reported on the same width scale the command byte uses, so a reached command echoes.
        angle = float(np.mean(joint_q[self._driver_coords]))
        cal_angles, cal_fractions = self._robotiq_cal_angles, self._robotiq_cal_fractions
        fraction = float(np.clip(np.interp(angle, cal_angles, cal_fractions), 0.0, 1.0))
        speed = float(np.interp(angle, cal_angles, self._robotiq_cal_slopes)
                      * np.mean(joint_qd[self._driver_dofs])) * ROBOTIQ_PRISMATIC_RANGE
        opening = fraction * ROBOTIQ_PRISMATIC_RANGE
        publish(self.channels.robotiq_robot_output_channel, robot_output_msg(
            utime, ROBOTIQ_POSITION_NAMES, (opening, opening), ROBOTIQ_VELOCITY_NAMES,
            (speed, speed), ROBOTIQ_EFFORT_NAMES, (0.0, 0.0),
        ))
        status = robotiq_status_msg(utime, fraction, speed, self.robotiq_force_byte)
        self.robotiq_status_byte = int(status.position)
        self.robotiq_status_speed_byte = int(status.speed)
        self.robotiq_opening = opening
        publish(self.channels.robotiq_status_channel, status)

        if self.pulley_state_object_name is not None:
            publish(self.channels.round_belt_pulley_state_channel, object_state_msg(
                utime, self.pulley_state_object_name,
                self._pulley_position_names, joint_q[self._pulley_coords],
                self._pulley_velocity_names, joint_qd[self._pulley_dofs],
            ))

    def _publish_belt_mesh(self, body_q: np.ndarray) -> None:
        centres = body_q[self.info.belt_bodies, :3].astype(np.float64)
        vertices, triangles = tube_mesh(centres, self.belt_radius)
        msg = deformable_mesh_msg(vertices, triangles, BELT_MESH_NAME, BELT_MESH_COLOR)
        self.bridge.publish(self.channels.deformable_geometry_channel, msg)

    def _log_grasp(self, body_q: np.ndarray, joint_q: np.ndarray) -> None:
        belt = body_q[self.info.belt_bodies, :3].astype(np.float64)

        def point_gap_mm(point: np.ndarray | None) -> str:
            if point is None:
                return "n/a"
            return f"{np.linalg.norm(belt - point, axis=1).min() * 1e3:.1f}"

        def gap_mm(body: int | None) -> str:
            return point_gap_mm(None if body is None else body_q[body, :3])

        hand = self._io_by_name["franka_hand"]
        width = float(-hand.positions[0] + hand.positions[1]) * 1e3
        pads = "/".join(gap_mm(b) for b in self.info.gripper_pad_bodies)
        tip = point_gap_mm(self._grasp_ur_tip_point(body_q))
        command = "none" if self.robotiq_position_byte is None else self.robotiq_position_byte
        pulleys = " / ".join(f"{np.degrees(q):.1f}" for q in joint_q[self._pulley_coords])
        logger.info(
            f"[GRASP] t {self.sim_time:.1f} s: hand width {width:.1f} mm, belt->finger_tip "
            f"{gap_mm(self._grasp_tip_body())} mm, belt->2f85 pads {pads} mm, belt->2f85 tip "
            f"{tip} mm, robotiq byte {command} (status {self.robotiq_status_byte}), "
            f"placed {self.belt_placed}, pulleys {pulleys or 'n/a'} deg"
        )

    def _log_belt_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        belt = body_q[self.info.belt_bodies, :3].astype(np.float64)
        centroid = ", ".join(f"{v:.4f}" for v in belt.mean(axis=0))
        centres = body_q[self._pulley_bodies, :3].astype(np.float64)
        drift = np.linalg.norm(centres - self._pulley_initial_centres, axis=1)
        logger.info(
            f"[BELT] final centroid ({centroid}), min z {belt[:, 2].min():.4f}, "
            f"placed {self.belt_placed}, step {self.step_index}, pulleys drift "
            f"{'/'.join(f'{1e3 * d:.3f}' for d in drift) or 'n/a'} mm"
        )

    def _reset_stats(self, now: float) -> None:
        self._stats_t0 = now
        self._stats_step0 = self.step_index
        self._stats_steps = 0
        self._stats_compute_sum = 0.0
        self._stats_compute_max = 0.0

    def _log_stats(self, now: float) -> None:
        wall = now - self._stats_t0
        if self._stats_steps == 0 or wall <= 0.0:
            return
        rate = self._stats_steps / wall
        franka = self._io_by_name.get("franka")
        franka_state = "n/a" if franka is None else ("stale-damping" if franka.stale else "fresh")
        robotiq = "none" if self.robotiq_position_byte is None else self.robotiq_position_byte
        logger.info(
            f"[STATS] step {self.step_index}: {rate:.1f} steps/s, compute "
            f"mean {self._stats_compute_sum / self._stats_steps * 1e3:.2f} ms "
            f"max {self._stats_compute_max * 1e3:.2f} ms, realtime {rate * self.frame_dt:.2f}x, "
            f"franka={franka_state}, robotiq_target={robotiq}, belt_placed={self.belt_placed}, "
            f"resyncs={self._resyncs}"
        )
        self._reset_stats(now)

    def run(self, num_steps: int = 0, realtime: bool = True) -> None:
        """Step until ``num_steps`` (0 = until the viewer closes or Ctrl-C), paced on wall time."""
        render = not isinstance(self.viewer, newton.viewer.ViewerNull)
        render_every = max(1, int(getattr(self.args, "render_every", 20)))
        first_step = self.step_index
        self._event_step = self.step_index
        self._event("run_start", num_steps=num_steps, realtime=realtime, render_every=render_every)
        reason = "num_steps"
        try:
            if self.use_cuda_graph and self.physics_graph is None:
                self.control_step()  # captures the CUDA graph: kept out of the pacing and stats
            t0 = time.perf_counter()
            self._reset_stats(t0)
            try:
                while num_steps <= 0 or self.step_index - first_step < num_steps:
                    if render and not self.viewer.is_running():
                        reason = "viewer_closed"
                        break
                    self.control_step()
                    if render and self.step_index % render_every == 0:
                        self.render()
                    now = time.perf_counter()
                    if realtime:
                        deadline = t0 + (self.step_index - first_step) * self.frame_dt
                        if now - deadline > RESYNC_LAG:
                            if self._resyncs == 0:
                                logger.warning(
                                    f"[LCM] {now - deadline:.3f} s behind real time at step "
                                    f"{self.step_index}; resyncing (no catch-up burst)"
                                )
                            self._resyncs += 1
                            self._event("resync", lag_s=now - deadline, step=self.step_index)
                            t0 += now - deadline
                        else:
                            if deadline - now > SPIN_MARGIN:
                                time.sleep(deadline - now - SPIN_MARGIN)
                            while time.perf_counter() < deadline:
                                pass
                            now = time.perf_counter()
                    if now - self._stats_t0 >= STATS_PERIOD:
                        self._log_stats(now)
            except KeyboardInterrupt:
                reason = "interrupted"
                logger.info(f"[LCM] interrupted at step {self.step_index}")
            self._log_stats(time.perf_counter())
            self._log_belt_final()
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            reason = "interrupted" if interrupted else f"error: {type(exc).__name__}"
            raise
        finally:
            try:
                self.close_recording(reason)
            except Exception as exc:  # noqa: BLE001 - never mask the run's own outcome
                logger.opt(exception=exc).error(f"[RECORD] closing the recording failed: {exc!r}")

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        parser = super().create_parser()
        parser.add_argument("--lcm-url", default=DEFAULT_LCM_URL, help="LCM provider URL")
        parser.add_argument(
            "--lcm-channels-file", default=None,
            help="lcm_channels.yaml whose keys override the default channel names",
        )
        parser.add_argument(
            "--no-realtime", action="store_false", dest="realtime",
            help="step as fast as possible instead of pacing to wall-clock time",
        )
        parser.add_argument(
            "--num-steps", type=int, default=0, help="control steps to run (0 = until stopped)"
        )
        parser.add_argument(
            "--render-every", type=int, default=20, help="render every N control steps"
        )
        parser.add_argument("--cameras", action="store_true", help="render the RGBD cameras")
        parser.add_argument(
            "--publish-belt-mesh", action="store_true",
            help="publish a 20 Hz belt tube mesh on DRAKE_VIEWER_DEFORMABLE",
        )
        parser.add_argument(
            "--initial-state", default=None,
            help="magna *_initial_state.yaml whose q_init_* lists replace the scene's "
                 "default joint positions (default: the scene defaults)",
        )
        parser.add_argument(
            "--record", nargs="?", const=str(DEFAULT_RECORDINGS_DIR), default=None,
            metavar="DIR", help="record the run under DIR/<timestamp>-<label>/ (default off)",
        )
        parser.add_argument(
            "--record-label", default=None, metavar="LABEL",
            help="run directory label (default round_belt)",
        )
        parser.add_argument(
            "--record-state-every", type=_positive_int, default=DEFAULT_STATE_EVERY, metavar="N",
            help="record body poses every N control steps (1 = every step)",
        )
        parser.add_argument(
            "--record-chunk-steps", type=_positive_int, default=DEFAULT_CHUNK_STEPS, metavar="N",
            help="control steps per recording chunk file",
        )
        # Newton's own --realtime (benchmark priority) shares the dest; this default wins.
        parser.set_defaults(
            viewer="null", realtime=True, cameras=False, publish_belt_mesh=False,
            substeps=cls.solver_substeps, vbd_iterations=cls.solver_vbd_iterations,
        )
        return parser
