"""Lock-step OSC bridge that also serves magna's assembly controller on the same private URL.

Per control step, in this order: a pending ``LATENT_STATE`` (:meth:`ControllerBridge.set_latent`),
the sim's UR state (utime rewritten to the OSC clock, like ``FRANKA_STATE``) re-published on the
controller's ``ur_state_channel`` (``UR_STATE``; it does not read ``UR_STATE_SIM``), then
``FRANKA_STATE`` (and a harness trajectory only if :meth:`set_trajectory` was called). The
controller's outputs (Franka trajectory, UR line, hand/Robotiq commands) are recorded, never
applied here.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from dairlib import lcmt_timestamped_saved_traj
from drake import lcmt_schunk_wsg_command
from robotiq import lcmt_robotiq_command
from round_belt_task.arm_kinematics import urdf_fixed_transform, wp_transform_to_mat4
from round_belt_task.commander import (
    METADATA_NAME,
    UrLine,
    mat3_to_quat,
    parse_saved_traj_message,
    pose_mat,
    x_tool0_tracking,
)
from round_belt_task.constants import UR10_URDF, X_W_UR10
from round_belt_task.osc_bridge import UTIME_PER_STEP, OscBridge
from task_common.latent_encoder import LATENT_STATE_CHANNEL
from task_common.lcm_contract import LcmChannels

UR_BASE_JOINT = "base_link-base_fixed_joint"


def ur_base_world() -> np.ndarray:
    """``X_W_base``: magna's UR ``base`` frame (``base_link`` * RotZ(pi)) in the world."""
    return wp_transform_to_mat4(X_W_UR10) @ urdf_fixed_transform(UR10_URDF, UR_BASE_JOINT)


@dataclass(frozen=True)
class FrankaTraj:
    wall: float
    utime: int
    pos: np.ndarray
    quat: np.ndarray
    times: np.ndarray


@dataclass(frozen=True)
class UrLineMsg:
    wall: float
    utime: int
    line: UrLine  # tool0 in the WORLD frame
    base_pos: np.ndarray  # (2, 3) the raw knots, tool0 in the UR base frame
    base_quat: np.ndarray  # (2, 4)


class ControllerBridge(OscBridge):
    """:class:`OscBridge` + UR state/``LATENT_STATE`` out, controller commands in."""

    def __init__(self, url: str, channels: LcmChannels | None = None, timeout_s: float = 5.0,
                 alive_fn=None, ur_state_out_channel: str | None = None) -> None:
        super().__init__(url, channels, timeout_s=timeout_s, alive_fn=alive_fn)
        c = self.channels
        self._ur_state_channel = c.ur_state_channel_sim
        # The controller subscribes the hardware name, not the sim's.
        self.ur_state_out_channel = ur_state_out_channel or c.ur_state_channel
        self._ur_traj_channel = c.ur_tracking_trajectory_actor_channel
        self.latent_channel = LATENT_STATE_CHANNEL
        self.X_W_base = ur_base_world()
        self.X_tool0_tracking = x_tool0_tracking()
        for channel, handler in ((self._traj_channel, self._on_franka_traj),
                                 (self._ur_traj_channel, self._on_ur_traj),
                                 (c.franka_hand_input_channel, self._on_hand),
                                 (c.robotiq_command_channel, self._on_robotiq)):
            self.lc.subscribe(channel, handler)
            self.subscribed.append(channel)
        self._held_state = None
        self._latent = None
        self.reset_controller_io()

    def reset_controller_io(self) -> None:
        """Forget everything received/sent for the controller (call per episode)."""
        self.franka_trajs: list[FrankaTraj] = []
        self.ur_lines: list[UrLineMsg] = []
        self.hand_commands: list[tuple[float, int, float, float]] = []
        self.robotiq_commands: list[tuple[float, int, int, int]] = []
        self.sent_latents: list[tuple[int, float, float]] = []  # (utime, t, wall)
        self._latent = None
        self.ur_sent: dict[int, np.ndarray] = {}
        self.pair_mismatch = 0
        self.undecodable = 0
        self.empty_trajs = 0

    # --- out ---------------------------------------------------------------------------------

    def set_latent(self, msg) -> None:
        """Publish ``msg`` on ``LATENT_STATE`` right before the next state pair."""
        self._latent = msg

    def publish(self, channel: str, msg: object) -> None:
        if channel == self._state_channel:
            self._held_state = msg  # sent after the UR state of the same step
            return
        if channel == self._ur_state_channel:
            utime = self.osc_utime(round(msg.utime / UTIME_PER_STEP))
            held = self._held_state
            if held is None or held.utime != msg.utime:
                self.pair_mismatch += 1
            msg.utime = utime
            self._send_latent()
            self.lc.publish(self.ur_state_out_channel, msg.encode())
            self.ur_sent[utime] = np.asarray(msg.position, dtype=np.float64)
            self._flush_state()
            return
        super().publish(channel, msg)

    def _send_latent(self) -> None:
        if self._latent is None:
            return
        self.lc.publish(self.latent_channel, self._latent.encode())
        blocks = self._latent.saved_traj.trajectories
        self.sent_latents.append((int(self._latent.utime), float(blocks[0].time_vec[0]),
                                  time.perf_counter()))
        self._latent = None

    def _flush_state(self) -> None:
        if self._held_state is not None:
            held, self._held_state = self._held_state, None
            super().publish(self._state_channel, held)

    def drain(self, sim_time: float) -> int:
        self._send_latent()
        self._flush_state()  # a state published without a UR state still goes out
        return super().drain(sim_time)

    def poll(self) -> None:
        """Handle whatever has arrived, without blocking (OSC replies stay queued)."""
        self._handle_all(0)

    # --- in ----------------------------------------------------------------------------------

    def _decode_traj(self, data: bytes):
        try:
            msg = lcmt_timestamped_saved_traj.decode(data)
            if msg.saved_traj.metadata.name == METADATA_NAME:
                return None  # our own hold trajectory looping back
            if not msg.saved_traj.trajectories:
                self.empty_trajs += 1  # the controller's output before its first plan
                return None
            return msg, *parse_saved_traj_message(msg)
        except (ValueError, KeyError) as exc:
            self.undecodable += 1
            logger.warning(f"[CTRL] undecodable trajectory ({exc})")
            return None

    def _on_franka_traj(self, channel: str, data: bytes) -> None:
        got = self._decode_traj(data)
        if got is not None:
            msg, pos, quat, times = got
            self.franka_trajs.append(FrankaTraj(time.perf_counter(), int(msg.utime), pos, quat,
                                                times))

    def _on_ur_traj(self, channel: str, data: bytes) -> None:
        got = self._decode_traj(data)
        if got is None:
            return
        msg, pos, quat, times = got
        if len(times) != 2:
            self.undecodable += 1
            return
        world = [self.X_W_base @ pose_mat(p, q) for p, q in zip(pos, quat, strict=True)]
        line = UrLine(p0=world[0][:3, 3].copy(), q0=mat3_to_quat(world[0][:3, :3]),
                      t0=float(times[0]), p1=world[1][:3, 3].copy(),
                      q1=mat3_to_quat(world[1][:3, :3]), t1=float(times[1]))
        self.ur_lines.append(UrLineMsg(time.perf_counter(), int(msg.utime), line, pos, quat))

    def _on_hand(self, channel: str, data: bytes) -> None:
        try:
            msg = lcmt_schunk_wsg_command.decode(data)
        except ValueError:
            self.undecodable += 1
            return
        self.hand_commands.append((time.perf_counter(), int(msg.utime),
                                   float(msg.target_position_mm), float(msg.force)))

    def _on_robotiq(self, channel: str, data: bytes) -> None:
        try:
            msg = lcmt_robotiq_command.decode(data)
        except ValueError:
            self.undecodable += 1
            return
        self.robotiq_commands.append((time.perf_counter(), int(msg.position), int(msg.speed),
                                     int(msg.force)))

    def latest_ur_line(self) -> UrLine | None:
        return self.ur_lines[-1].line if self.ur_lines else None

    def gripper_commands(self) -> tuple[float | None, int | None]:
        """The controller's latest (hand mm, Robotiq byte), None if never received."""
        return (self.hand_commands[-1][2] if self.hand_commands else None,
                self.robotiq_commands[-1][1] if self.robotiq_commands else None)

    def tracking_pose(self, line: UrLine, t: float) -> np.ndarray:
        """UR tracking frame (world 4x4) on ``line`` at ``clamp(t, t0, t1)``."""
        p, q = line.sample(t)
        return pose_mat(p, q) @ self.X_tool0_tracking

    def traj_after_latent_ms(self) -> list[float]:
        """Per sent latent: wall ms until the first Franka plan with ``time_vec[0] >= t``."""
        out = []
        trajs = self.franka_trajs
        for _, t, wall in self.sent_latents:
            got = next((tr.wall for tr in trajs
                        if tr.wall >= wall and tr.times[0] >= t - 0.5 * UTIME_PER_STEP * 1e-6),
                       None)
            if got is not None:
                out.append((got - wall) * 1e3)
        return out

    def controller_stats(self) -> dict:
        lat = self.traj_after_latent_ms()
        return {"franka_trajs": len(self.franka_trajs), "ur_lines": len(self.ur_lines),
                "hand_commands": len(self.hand_commands),
                "robotiq_commands": len(self.robotiq_commands),
                "latents_sent": len(self.sent_latents), "pair_mismatch": self.pair_mismatch,
                "undecodable": self.undecodable, "empty_trajs": self.empty_trajs,
                "traj_after_latent_ms": {
                    "n": len(lat), "mean": float(np.mean(lat)) if lat else math.nan,
                    "max": float(np.max(lat)) if lat else math.nan}}
