"""Lock-step LCM bridge to magna's Franka OSC; grippers stay in-process (:class:`OfflineBridge`).

Per control step: ``publish(FRANKA_STATE)`` (the tick's trajectory first) sets ``pending``; the
next ``drain`` blocks until the ``FRANKA_INPUT`` answering that state arrives. The OSC clock is
``step * 5000 + offset_us`` µs; :meth:`OscBridge.rebase` keeps it strictly increasing across
snapshot restores so magna's ``LcmDrivenLoop`` never resets.
"""

from __future__ import annotations

import math
import time

import lcm
import numpy as np
from loguru import logger

from dairlib import lcmt_robot_input
from round_belt_task.offline_simulation import OfflineBridge
from task_common.lcm_contract import CONTROL_DT, LcmChannels, efforts_by_name
from task_common.osc_process import check_private_url

UTIME_PER_STEP = round(CONTROL_DT * 1e6)
# RobotCommandSender echoes the context time cast back to µs: +-1 µs float rounding.
UTIME_MATCH_US = 2
HANDLE_MS = 50
WARM_UP_REPUBLISH_S = 0.1
WARM_UP_QUIET_S = 0.3


class OscTimeout(RuntimeError):
    def __init__(self, step: int, pending_utime: int, process_alive: bool | None) -> None:
        super().__init__(f"no FRANKA_INPUT for step {step} (utime {pending_utime}); "
                         f"OSC process alive: {process_alive}")
        self.step = step
        self.pending_utime = pending_utime
        self.process_alive = process_alive


class OscBridge(OfflineBridge):
    """``OfflineBridge`` + a socket for ``FRANKA_STATE``/trajectory out, ``FRANKA_INPUT`` in."""

    def __init__(self, url: str, channels: LcmChannels | None = None, timeout_s: float = 5.0,
                 alive_fn=None) -> None:
        super().__init__()
        self.url = check_private_url(url)
        self.channels = channels or LcmChannels()
        self.timeout_s = float(timeout_s)
        self.alive_fn = alive_fn  # () -> bool | None, reported in OscTimeout
        self.lc = lcm.LCM(self.url)
        self._input_channel = self.channels.franka_input_channel
        self._state_channel = self.channels.franka_state_channel
        self._traj_channel = self.channels.tracking_trajectory_actor_channel
        self.lc.subscribe(self._input_channel, self._on_input)
        self.subscribed = [self._input_channel]
        # Starts one step in: utime and the hold's time_vec[0] must be > 0.
        self.utime_offset_us = UTIME_PER_STEP
        self.last_sent_utime: int | None = None
        self.sent_utimes: list[int] = []
        self.pending: int | None = None
        self._pending_step = -1
        self._pending_bytes: bytes | None = None
        self._trajectory = None
        self._received: list[tuple[int, dict]] = []
        self._efforts: dict[str, float] | None = None
        self._efforts_array: np.ndarray | None = None
        self.last_traj = None
        self.reset_stats()

    def reset_stats(self) -> None:
        self.wait_s = 0.0
        self.wait_max_s = 0.0
        self.replies = 0
        self.stale_replies = 0
        self.republished = 0
        self.waits = 0

    def stats(self) -> dict:
        return {"replies": self.replies, "stale_replies": self.stale_replies,
                "republished": self.republished, "wait_s": self.wait_s,
                "wait_mean_ms": 1e3 * self.wait_s / max(1, self.waits),
                "wait_max_ms": 1e3 * self.wait_max_s}

    # --- clock ----------------------------------------------------------------------------------

    def osc_utime(self, step: int) -> int:
        return int(step) * UTIME_PER_STEP + self.utime_offset_us

    def osc_time_s(self, step: int) -> float:
        return self.osc_utime(step) * 1e-6

    def rebase(self, step_index: int) -> None:
        """After a restore: the next published utime (step ``step_index``) = last sent + 1 step."""
        if self.pending is not None:
            self.drain(self.sim_time)  # consume the in-flight answer: no stale reply later
        if self.last_sent_utime is None:
            return
        self.utime_offset_us = self.last_sent_utime + UTIME_PER_STEP - step_index * UTIME_PER_STEP
        self._trajectory = None

    def forget_pending(self) -> None:
        """Drop a publish whose reply was lost (after :class:`OscTimeout`): rebase won't wait."""
        self.pending = None

    # --- I/O --------------------------------------------------------------------------------------

    def set_trajectory(self, msg) -> None:
        self._trajectory = msg

    def _on_input(self, channel: str, data: bytes) -> None:
        try:
            msg = lcmt_robot_input.decode(data)
        except ValueError as exc:
            logger.warning(f"[OSC] undecodable {channel} ({exc})")
            return
        self._received.append((int(msg.utime), efforts_by_name(msg)))

    def publish(self, channel: str, msg: object) -> None:
        if channel == self._state_channel:
            utime = self.osc_utime(round(msg.utime / UTIME_PER_STEP))
            msg.utime = utime
            data = msg.encode()
            # Trajectory before state: the OSC evaluates it exactly at its start (the
            # deterministic best case of magna's publish race).
            if self._trajectory is not None:
                self.lc.publish(self._traj_channel, self._trajectory.encode())
                self.last_traj, self._trajectory = self._trajectory, None
            self.lc.publish(channel, data)
            self.pending, self._pending_bytes = utime, data
            self._pending_step = round((utime - self.utime_offset_us) / UTIME_PER_STEP)
            if utime != self.last_sent_utime:  # warm-up republishes repeat the utime
                self.sent_utimes.append(utime)
            self.last_sent_utime = utime
        elif channel == self._traj_channel:
            self.lc.publish(channel, msg.encode())
        # every other channel (hand, UR, Robotiq, pulley state) has no consumer: dropped

    def _handle_all(self, timeout_ms: int) -> None:
        if self.lc.handle_timeout(timeout_ms) > 0:
            while self.lc.handle_timeout(0) > 0:
                pass

    def _take_match(self) -> dict | None:
        match = None
        for utime, efforts in self._received:
            if abs(utime - self.pending) <= UTIME_MATCH_US:
                match = efforts
            else:
                self.stale_replies += 1
        self._received.clear()
        return match

    def drain(self, sim_time: float) -> int:
        self.sim_time = sim_time
        if self.pending is None:
            return 0
        t0 = time.perf_counter()
        republished = False
        deadline = t0 + self.timeout_s
        while True:
            self._handle_all(HANDLE_MS)
            efforts = self._take_match()
            if efforts is not None:
                break
            if time.perf_counter() >= deadline:
                if republished:
                    alive = None if self.alive_fn is None else bool(self.alive_fn())
                    raise OscTimeout(self._pending_step, self.pending, alive)
                # UDP drop insurance: resends only FRANKA_STATE, not the trajectory.
                self.lc.publish(self._state_channel, self._pending_bytes)
                self.republished += 1
                republished = True
                deadline = time.perf_counter() + self.timeout_s
        for name, value in efforts.items():
            if not math.isfinite(value):
                efforts[name] = 0.0
        waited = time.perf_counter() - t0
        self.wait_s += waited
        self.wait_max_s = max(self.wait_max_s, waited)
        self.waits += 1
        self.replies += 1
        self._efforts, self._efforts_array = efforts, None
        self.pending = None
        return 1

    def latest_efforts(self, spec) -> tuple[np.ndarray | None, float]:
        if spec.input_channel_key != "franka_input_channel":
            return np.zeros(len(spec.effort_names)), 0.0
        if self._efforts is None:
            return None, math.inf
        if self._efforts_array is None:
            self._efforts_array = np.array(
                [self._efforts.get(name, 0.0) for name in spec.effort_names], dtype=np.float64)
        return self._efforts_array, 0.0  # the reply answers the state just published: age 0

    def warm_up(self, publish_state_fn, timeout_s: float = 60.0) -> float:
        """Republish the current state every 100 ms until the OSC answers; the reply is discarded.

        ``publish_state_fn()`` publishes FRANKA_STATE at the CURRENT step (same utime each call).
        Leaves one fresh publish pending for the next :meth:`drain`. Returns the warm-up time
        [s]; raises :class:`OscTimeout` after ``timeout_s``.
        """
        t0 = time.perf_counter()
        self._received.clear()
        while True:
            publish_state_fn()
            end = time.perf_counter() + WARM_UP_REPUBLISH_S
            while time.perf_counter() < end:
                self._handle_all(round(WARM_UP_REPUBLISH_S * 1e3 / 4))
                if any(abs(u - self.pending) <= UTIME_MATCH_US for u, _ in self._received):
                    waited = time.perf_counter() - t0
                    # Swallow late answers to the earlier republishes (same utime).
                    self._handle_all(round(WARM_UP_QUIET_S * 1e3))
                    self._received.clear()
                    # One more publish (same utime) so the next step drains a real reply.
                    publish_state_fn()
                    return waited
            if time.perf_counter() - t0 > timeout_s:
                alive = None if self.alive_fn is None else bool(self.alive_fn())
                raise OscTimeout(self._pending_step, self.pending, alive)
