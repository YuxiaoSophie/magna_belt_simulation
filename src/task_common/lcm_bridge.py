"""Non-blocking LCM I/O for the simulation: latest-message inputs, immediate publishes.

The simulation drains the socket once per control step with the current SIM time, so input
ages are measured on the simulation clock (Drake's ``FrankaInputSimulator`` rule).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import lcm
import numpy as np
from loguru import logger

from dairlib import lcmt_robot_input
from drake import lcmt_schunk_wsg_command
from robotiq import lcmt_robotiq_command
from task_common.lcm_contract import LcmChannels, RobotIoSpec, efforts_by_name

MAX_MESSAGES_PER_DRAIN = 64


@dataclass
class _Latest:
    payload: object = None
    sim_time: float = -math.inf
    efforts: np.ndarray | None = None


class LcmBridge:
    """One ``lcm.LCM`` handle: the robot-input, Robotiq- and hand-command subscriptions."""

    def __init__(self, url: str, channels: LcmChannels, specs: Sequence[RobotIoSpec]) -> None:
        self.url = url
        self.channels = channels
        self.lc = lcm.LCM(url)
        self.sim_time = 0.0
        self._latest: dict[str, _Latest] = {}
        self._warned: set[tuple[str, str]] = set()
        for spec in specs:
            if spec.input_channel_key is None:
                continue
            channel = getattr(channels, spec.input_channel_key)
            self._latest[channel] = _Latest()
            self.lc.subscribe(channel, self._on_robot_input)
        self._latest[channels.robotiq_command_channel] = _Latest()
        self.lc.subscribe(channels.robotiq_command_channel, self._on_robotiq_command)
        self._latest[channels.franka_hand_input_channel] = _Latest()
        self.lc.subscribe(channels.franka_hand_input_channel, self._on_hand_command)

    def _warn_once(self, channel: str, key: str, text: str) -> None:
        if (channel, key) not in self._warned:
            self._warned.add((channel, key))
            logger.warning(f"[LCM] {channel}: {text}")

    def _on_robot_input(self, channel: str, data: bytes) -> None:
        try:
            efforts = efforts_by_name(lcmt_robot_input.decode(data))
        except ValueError as exc:
            self._warn_once(channel, "<decode>", f"undecodable lcmt_robot_input ({exc})")
            return
        for name, value in efforts.items():
            if not math.isfinite(value):
                self._warn_once(channel, "<non-finite>", f"non-finite effort {name}; using 0")
                efforts[name] = 0.0
        latest = self._latest[channel]
        latest.payload, latest.sim_time, latest.efforts = efforts, self.sim_time, None

    def _on_robotiq_command(self, channel: str, data: bytes) -> None:
        try:
            msg = lcmt_robotiq_command.decode(data)
        except ValueError as exc:
            self._warn_once(channel, "<decode>", f"undecodable lcmt_robotiq_command ({exc})")
            return
        latest = self._latest[channel]
        latest.payload = (int(msg.position), int(msg.speed), int(msg.force))
        latest.sim_time = self.sim_time

    def _on_hand_command(self, channel: str, data: bytes) -> None:
        try:
            msg = lcmt_schunk_wsg_command.decode(data)
        except ValueError as exc:
            self._warn_once(channel, "<decode>", f"undecodable lcmt_schunk_wsg_command ({exc})")
            return
        if not math.isfinite(msg.target_position_mm):
            self._warn_once(channel, "<non-finite>", "non-finite target_position_mm; ignored")
            return
        latest = self._latest[channel]
        latest.payload = (int(msg.utime), float(msg.target_position_mm), float(msg.force))
        latest.sim_time = self.sim_time

    def drain(self, sim_time: float) -> int:
        """Dispatch up to ``MAX_MESSAGES_PER_DRAIN`` pending messages; returns how many."""
        self.sim_time = sim_time
        handled = 0
        while handled < MAX_MESSAGES_PER_DRAIN and self.lc.handle_timeout(0) > 0:
            handled += 1
        return handled

    def publish(self, channel: str, msg: object) -> None:
        self.lc.publish(channel, msg.encode())

    def latest_efforts(self, spec: RobotIoSpec) -> tuple[np.ndarray | None, float]:
        """The newest efforts ordered as ``spec.effort_names`` and their sim-time age [s]."""
        channel = getattr(self.channels, spec.input_channel_key)
        latest = self._latest[channel]
        if latest.payload is None:
            return None, math.inf
        if latest.efforts is None:
            for name in latest.payload:
                if name not in spec.effort_names:
                    self._warn_once(channel, name, f"ignoring unknown effort name {name!r}")
            latest.efforts = np.array(
                [latest.payload.get(name, 0.0) for name in spec.effort_names], dtype=np.float64
            )
        return latest.efforts, self.sim_time - latest.sim_time

    def latest_robotiq(self) -> tuple[int, int, int] | None:
        """``(position, speed, force)`` bytes of the newest Robotiq command, if any."""
        return self._latest[self.channels.robotiq_command_channel].payload

    def latest_hand_command(self) -> tuple[int, float, float] | None:
        """``(utime_us, target_position_mm, force)`` of the newest hand command, if any."""
        return self._latest[self.channels.franka_hand_input_channel].payload
