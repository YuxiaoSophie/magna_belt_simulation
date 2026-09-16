"""The Drake ``magna_simulation`` LCM contract: channels, per-robot name lists, messages.

Pure Python over the generated ``dairlib`` / ``drake`` / ``robotiq`` types (no Newton), so
peers and check scripts can import it cheaply.  Name lists, channel keys and the Robotiq /
Franka conventions follow magna ``systems/simulation`` and ``lcm_channels.yaml``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml

from dairlib import lcmt_robot_input, lcmt_robot_output
from drake import lcmt_schunk_wsg_status
from robotiq import lcmt_robotiq_status

DEFAULT_LCM_URL = "udpm://239.255.76.67:7667?ttl=0"
CONTROL_DT = 0.005

ROBOTIQ_PRISMATIC_RANGE = 0.04588
ROBOTIQ_STATUS_SPEED_SCALE = 2.0
ROBOTIQ_POSITION_NAMES = ("left_finger_joint", "right_finger_joint")
ROBOTIQ_VELOCITY_NAMES = ("left_finger_jointdot", "right_finger_jointdot")
# Drake publishes the Robotiq effort names without a suffix.
ROBOTIQ_EFFORT_NAMES = ("left_finger_joint", "right_finger_joint")

BELT_MESH_NAME = "round_belt::round_belt"
BELT_MESH_COLOR = (1.0, 0.5, 0.0, 1.0)

FRANKA_STALE_DAMPING = np.array([37.5, 50.0, 37.5, 25.0, 5.0, 3.75, 2.5])
FRANKA_STALE_TIMEOUT = 0.1
# ShunkCommandToTrajectory: width [mm] -> per-finger target [m].
HAND_WIDTH_TO_FINGER = 1.0 / 2000.0

_UR_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


@dataclass(frozen=True)
class LcmChannels:
    """Channel names keyed like magna ``systems/parameters/lcm_channels.yaml``."""

    franka_state_channel: str = "FRANKA_STATE"
    franka_input_channel: str = "FRANKA_INPUT"
    franka_hand_robot_output_channel: str = "FRANKA_HAND_ROBOT_OUTPUT"
    franka_hand_state_channel: str = "PANDA_HAND_STATUS"
    franka_hand_robot_input_channel: str = "FRANKA_HAND_ROBOT_INPUT"
    franka_hand_input_channel: str = "PANDA_HAND_COMMAND"
    ur_state_channel_sim: str = "UR_STATE_SIM"
    ur_input_channel_sim: str = "UR_INPUT_SIM"
    robotiq_robot_output_channel: str = "ROBOTIQ_ROBOT_OUTPUT"
    robotiq_status_channel: str = "ROBOTIQ_STATUS"
    robotiq_command_channel: str = "ROBOTIQ_COMMAND"
    deformable_geometry_channel: str = "DRAKE_VIEWER_DEFORMABLE"

    @classmethod
    def from_yaml(cls, path: str | Path) -> LcmChannels:
        """Channels from an ``lcm_channels.yaml``; keys it does not set keep their defaults."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{key: str(value) for key, value in data.items() if key in known})


@dataclass(frozen=True)
class RobotIoSpec:
    """One robot's input/output channels and the Newton joints behind its name lists."""

    name: str
    input_channel_key: str | None
    output_channel_key: str
    joint_labels: tuple[str, ...]
    position_names: tuple[str, ...]
    velocity_names: tuple[str, ...]
    effort_names: tuple[str, ...]
    stale_damping: np.ndarray | None = None
    stale_timeout: float = math.inf

    def __post_init__(self) -> None:
        n = len(self.joint_labels)
        lengths = (len(self.position_names), len(self.velocity_names), len(self.effort_names))
        if any(length != n for length in lengths):
            raise ValueError(f"{self.name}: {n} joint labels but name list lengths {lengths}")
        if self.stale_damping is not None and len(self.stale_damping) != n:
            raise ValueError(f"{self.name}: stale_damping needs {n} gains")


@dataclass(frozen=True)
class HandDrive:
    """The in-sim Panda finger position drive and its ``PANDA_HAND_COMMAND`` stale rule."""

    ke: float
    kd: float
    effort_limit: float
    stale_timeout: float
    armature: float
    max_speed: float
    limit_ke: float
    limit_kd: float


def hand_targets(target_position_mm: float) -> tuple[float, float]:
    """``(panda_finger_joint1, panda_finger_joint2)`` targets; not clamped to the joint range."""
    p = float(target_position_mm) * HAND_WIDTH_TO_FINGER
    return -p, p


def franka_spec(joint_labels: Sequence[str]) -> RobotIoSpec:
    """``FRANKA_INPUT`` -> ``FRANKA_STATE`` over the 7 Newton arm joints, in order."""
    return RobotIoSpec(
        name="franka", input_channel_key="franka_input_channel",
        output_channel_key="franka_state_channel", joint_labels=tuple(joint_labels),
        position_names=tuple(f"panda_joint{i}" for i in range(1, 8)),
        velocity_names=tuple(f"panda_joint{i}dot" for i in range(1, 8)),
        effort_names=tuple(f"panda_motor{i}" for i in range(1, 8)),
        stale_damping=FRANKA_STALE_DAMPING, stale_timeout=FRANKA_STALE_TIMEOUT,
    )


def franka_hand_spec(joint_labels: Sequence[str]) -> RobotIoSpec:
    """``FRANKA_HAND_ROBOT_OUTPUT`` over the 2 finger joints (driven from the hand command)."""
    return RobotIoSpec(
        name="franka_hand", input_channel_key=None,
        output_channel_key="franka_hand_robot_output_channel", joint_labels=tuple(joint_labels),
        position_names=("panda_finger_joint1", "panda_finger_joint2"),
        velocity_names=("panda_finger_joint1dot", "panda_finger_joint2dot"),
        effort_names=("panda_finger_motor1", "panda_finger_motor2"),
    )


def ur10_spec(joint_labels: Sequence[str]) -> RobotIoSpec:
    """``UR_INPUT_SIM`` -> ``UR_STATE_SIM`` over the 6 Newton UR10 joints, in order."""
    return RobotIoSpec(
        name="ur10", input_channel_key="ur_input_channel_sim",
        output_channel_key="ur_state_channel_sim", joint_labels=tuple(joint_labels),
        position_names=_UR_JOINT_NAMES,
        velocity_names=tuple(f"{name}dot" for name in _UR_JOINT_NAMES),
        effort_names=tuple(f"{name}_actuator" for name in _UR_JOINT_NAMES),
    )


def robot_output_msg(
    utime: int,
    position_names: Sequence[str], positions: Sequence[float],
    velocity_names: Sequence[str], velocities: Sequence[float],
    effort_names: Sequence[str], efforts: Sequence[float],
) -> lcmt_robot_output:
    msg = lcmt_robot_output()
    msg.utime = int(utime)
    msg.num_positions = len(position_names)
    msg.num_velocities = len(velocity_names)
    msg.num_efforts = len(effort_names)
    msg.position_names = list(position_names)
    msg.position = [float(v) for v in positions]
    msg.velocity_names = list(velocity_names)
    msg.velocity = [float(v) for v in velocities]
    msg.effort_names = list(effort_names)
    msg.effort = [float(v) for v in efforts]
    msg.imu_accel = [0.0, 0.0, 0.0]
    return msg


def schunk_status_msg(
    utime: int, finger_q: Sequence[float], finger_qd: Sequence[float]
) -> lcmt_schunk_wsg_status:
    """``PANDA_HAND_STATUS``: opening = ``-q1 + q2`` in mm, force always 0 (as Drake)."""
    msg = lcmt_schunk_wsg_status()
    msg.utime = int(utime)
    msg.actual_position_mm = float(-finger_q[0] + finger_q[1]) * 1000.0
    msg.actual_speed_mm_per_s = float(-finger_qd[0] + finger_qd[1]) * 1000.0
    msg.actual_force = 0.0
    return msg


def robotiq_status_msg(
    utime: int, fraction: float, mapped_speed: float, force: int
) -> lcmt_robotiq_status:
    """``ROBOTIQ_STATUS`` from the opening ``fraction`` (0 open, 1 closed) and its speed [m/s]."""
    msg = lcmt_robotiq_status()
    msg.utime = int(utime)
    msg.activation_status = True
    msg.gripper_mode = True
    msg.goto_status = True
    msg.position = round(min(max(float(fraction), 0.0), 1.0) * 255)
    speed = min(max(abs(float(mapped_speed)) / ROBOTIQ_STATUS_SPEED_SCALE, 0.0), 1.0)
    msg.speed = round(speed * 255)
    msg.force = int(force) & 0xFF
    return msg


def efforts_by_name(msg: lcmt_robot_input) -> dict[str, float]:
    count = min(int(msg.num_efforts), len(msg.effort_names), len(msg.efforts))
    return {str(msg.effort_names[i]): float(msg.efforts[i]) for i in range(count)}
