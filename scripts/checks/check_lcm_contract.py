#!/usr/bin/env python3
"""Headless, in-process check that the Newton LCM sim speaks Drake's ``magna_simulation``
LCM contract: message names/layouts, utime, torque sign and units, gravity compensation
under zero torque, the Franka stale-input damping, the Panda hand command drive, the Robotiq
command -> status round trip, the belt trigger + Franka grasp and lift, the belt tube mesh
message and the task-board pulley state.

Builds ``RoundBeltLcmSimulation`` exactly as ``scripts/round_belt_lcm_simulation.py`` does (but
non-realtime, null viewer, cameras off) and drives it with ``sim.control_step()`` only. A
second ``lcm.LCM`` handle on the same private URL plays the role of the magna controllers:
it records every published state message and publishes the robot/Robotiq commands each check
needs. Checks run in order and share the sim's running state (later checks build on earlier
ones), so they are not independent fixtures.

Run:
    uv run python scripts/checks/check_lcm_contract.py
    uv run python scripts/checks/check_lcm_contract.py --device cpu
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
# The task packages live under src/; make them importable regardless of CWD.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import lcm
import newton.examples

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from dairlib import lcmt_object_state, lcmt_robot_input, lcmt_robot_output
from drake import lcmt_schunk_wsg_command, lcmt_schunk_wsg_status
from robotiq import lcmt_robotiq_command, lcmt_robotiq_status
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from task_common.belt_mesh_lcm import (
    DEFORMABLE_LINK_NAME,
    deformable_mesh_msg,
    tube_mesh,
)
from task_common.lcm_contract import BELT_MESH_COLOR, BELT_MESH_NAME, LcmChannels
from utils.labels import body_index

sys.path.insert(0, str(Path(__file__).parent))
from lcm_peer_utils import (
    HAND_CLOSED_WIDTH,
    StatePeer,
    finger_tip_ik,
    grasp_sequence,
    pd_step,
)

# A private multicast group so this check never disturbs a running magna stack, and never
# collides with the parallel belt-mesh worker's own scratch group.
PRIVATE_LCM_URL = "udpm://239.255.76.68:7668?ttl=0"
LAYOUTS_PATH = REPO_ROOT / "scripts/checks/data/drake_lcm_layouts.json"
MAGNA_LCM_CHANNELS_YAML = Path("/home/hienbui/git/magna/systems/parameters/lcm_channels.yaml")

CONTROL_DT_UTIME = 5000  # 5 ms control step (src/task_common/lcm_contract.py CONTROL_DT)
DRAIN_TIMEOUT_MS = 20  # generous vs. the ~0.1 ms observed loopback delivery latency
LAYOUT_WAIT_S = 0.5  # bounded wait for check 1's six same-step packets, all under load
LAYOUT_POLL_MS = 10
# Franka stale-input rule (magna_simulation.cc's FrankaInputSimulator), transcribed.
FRANKA_STALE_DAMPING = np.array([37.5, 50.0, 37.5, 25.0, 5.0, 3.75, 2.5])
FRANKA_STALE_TIMEOUT = 0.1
ROBOTIQ_PRISMATIC_RANGE = 0.04588
# 20 mm/finger @ hand_drive.max_speed 0.1 m/s = 0.2 s slew + drive settle; measured 0.380-0.385 s.
HAND_MOVE_TIME_LIMIT = 0.5
HAND_MOVE_HOLD_STEPS = 150  # 0.75 s: clears the limit with margin so settle_time is never clipped
LIFT_HEIGHT = 0.05
LIFT_TIME = 1.0
PULLEY_STATE_STEPS = 50
PULLEY_OBJECT_NAME = "nist_board"
PULLEY_POSITION_NAMES = ["small_round_pulley_joint", "large_round_pulley_joint"]
PULLEY_VELOCITY_NAMES = ["small_round_pulley_jointdot", "large_round_pulley_jointdot"]
PULLEY_JOINT_LABELS = ["board/small_round_pulley_joint", "board/large_round_pulley_joint"]

# Robot-output channels: (drake_lcm_layouts.json key, LcmChannels attribute).
ROBOT_OUTPUT_CHANNELS = [
    ("FRANKA_STATE", "franka_state_channel"),
    ("FRANKA_HAND_ROBOT_OUTPUT", "franka_hand_robot_output_channel"),
    ("UR_STATE_SIM", "ur_state_channel_sim"),
    ("ROBOTIQ_ROBOT_OUTPUT", "robotiq_robot_output_channel"),
]
# Decoder per output channel attribute the peer subscribes to.
DECODERS = {
    "franka_state_channel": lcmt_robot_output,
    "franka_hand_robot_output_channel": lcmt_robot_output,
    "ur_state_channel_sim": lcmt_robot_output,
    "robotiq_robot_output_channel": lcmt_robot_output,
    "franka_hand_state_channel": lcmt_schunk_wsg_status,
    "robotiq_status_channel": lcmt_robotiq_status,
    "round_belt_pulley_state_channel": lcmt_object_state,
}

# Check 11's own synthetic belt-mesh fixture.
BELT_MESH_N_BODIES = 48
BELT_MESH_SIDES = 8
BELT_MESH_RADIUS = 0.015
BELT_MESH_EXPECTED_V = BELT_MESH_N_BODIES * BELT_MESH_SIDES
BELT_MESH_EXPECTED_T = BELT_MESH_N_BODIES * BELT_MESH_SIDES * 2
BELT_MESH_EXPECTED_NUM_FLOAT_DATA = 2 + 3 * BELT_MESH_EXPECTED_V + 3 * BELT_MESH_EXPECTED_T


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


class Skipped(Exception):
    """Raised by a check to signal [SKIP] rather than [FAIL] (an optional reference is absent)."""


class Recorder:
    """A peer ``lcm.LCM`` handle: records every message on the sim's seven output channels."""

    def __init__(self, channels: LcmChannels) -> None:
        self.peer = lcm.LCM(PRIVATE_LCM_URL)
        self.messages: dict[str, list] = {}
        for key, decoder in DECODERS.items():
            channel = getattr(channels, key)
            self.messages[channel] = []
            self.peer.subscribe(channel, self._handler(channel, decoder))

    def _handler(self, channel: str, decoder):
        def on_message(_channel: str, data: bytes) -> None:
            self.messages[channel].append(decoder.decode(data))

        return on_message

    def drain(self) -> None:
        # blocking wait for the first message avoids racing loopback delivery.
        if self.peer.handle_timeout(DRAIN_TIMEOUT_MS) <= 0:
            return
        while self.peer.handle_timeout(0) > 0:
            pass

    def publish(self, channel: str, msg: object) -> None:
        self.peer.publish(channel, msg.encode())

    def wait_for_each(self, channels, timeout_s: float = LAYOUT_WAIT_S,
                       poll_ms: int = LAYOUT_POLL_MS) -> None:
        """Bounded wait until every channel in ``channels`` has >= 1 recorded message.

        Avoids the "ROBOTIQ_STATUS: 0 messages" flake from drain()'s single-shot wait.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and any(
            len(self.messages[ch]) == 0 for ch in channels
        ):
            self.peer.handle_timeout(poll_ms)
        while self.peer.handle_timeout(0) > 0:
            pass


def _step(sim: RoundBeltLcmSimulation, recorder: Recorder, n: int = 1) -> None:
    for _ in range(n):
        sim.control_step()
        recorder.drain()


def _publish_effort(
    recorder: Recorder, channel: str, effort_names, efforts, utime: int
) -> None:
    msg = lcmt_robot_input()
    msg.utime = int(utime)
    msg.effort_names = list(effort_names)
    msg.efforts = [float(v) for v in efforts]
    msg.num_efforts = len(msg.efforts)
    recorder.publish(channel, msg)


# ----------------------------------------------------------------------------
# Checks, registered in order by @check(title). Each takes the shared Ctx and steps the
# running sim onward -- they are a pipeline, not independent fixtures.
# ----------------------------------------------------------------------------

CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


@check("1. Layouts")
def check_layouts(ctx: SimpleNamespace) -> None:
    sim, recorder, layouts = ctx.sim, ctx.recorder, ctx.layouts
    _step(sim, recorder, 1)
    utime_expected = sim.step_index * CONTROL_DT_UTIME

    expected_channels = [getattr(sim.channels, attr) for _, attr in ROBOT_OUTPUT_CHANNELS]
    expected_channels += [
        sim.channels.franka_hand_state_channel, sim.channels.robotiq_status_channel
    ]
    recorder.wait_for_each(expected_channels)

    for key, chan_attr in ROBOT_OUTPUT_CHANNELS:
        channel = getattr(sim.channels, chan_attr)
        msgs = recorder.messages[channel]
        _require(len(msgs) == 1, f"{channel}: {len(msgs)} messages after one step, expected 1")
        msg, want = msgs[0], layouts[key]
        _require(list(msg.position_names) == want["position_names"],
                  f"{channel}: position_names {list(msg.position_names)} != "
                  f"{want['position_names']}")
        _require(list(msg.velocity_names) == want["velocity_names"],
                  f"{channel}: velocity_names {list(msg.velocity_names)} != "
                  f"{want['velocity_names']}")
        _require(list(msg.effort_names) == want["effort_names"],
                  f"{channel}: effort_names {list(msg.effort_names)} != {want['effort_names']}")
        _require(msg.num_positions == len(want["position_names"]),
                  f"{channel}: num_positions {msg.num_positions} != {len(want['position_names'])}")
        _require(msg.num_velocities == len(want["velocity_names"]),
                  f"{channel}: num_velocities {msg.num_velocities} != "
                  f"{len(want['velocity_names'])}")
        _require(msg.num_efforts == len(want["effort_names"]),
                  f"{channel}: num_efforts {msg.num_efforts} != {len(want['effort_names'])}")
        _require(list(msg.imu_accel) == [0.0, 0.0, 0.0],
                  f"{channel}: imu_accel {list(msg.imu_accel)} != [0, 0, 0]")
        _require(msg.utime == utime_expected, f"{channel}: utime {msg.utime} != {utime_expected}")

    hand_channel = sim.channels.franka_hand_state_channel
    hand_msgs = recorder.messages[hand_channel]
    _require(len(hand_msgs) == 1, f"{hand_channel}: {len(hand_msgs)} messages, expected 1")
    hand = hand_msgs[0]
    _require(hand.utime == utime_expected, f"{hand_channel}: utime {hand.utime} != "
              f"{utime_expected}")
    _require(hand.actual_force == 0.0, f"{hand_channel}: actual_force {hand.actual_force} != 0")

    status_channel = sim.channels.robotiq_status_channel
    status_msgs = recorder.messages[status_channel]
    _require(len(status_msgs) == 1, f"{status_channel}: {len(status_msgs)} messages, expected 1")
    status = status_msgs[0]
    _require(status.utime == utime_expected, f"{status_channel}: utime {status.utime} != "
              f"{utime_expected}")
    for name, value in (("position", status.position), ("speed", status.speed),
                        ("force", status.force)):
        _require(0 <= value <= 255, f"{status_channel}: {name} {value} out of 0..255")
    _require(status.activation_status and status.gripper_mode and status.goto_status,
              f"{status_channel}: activation/gripper_mode/goto_status = "
              f"{status.activation_status}/{status.gripper_mode}/{status.goto_status}, "
              "want all True")


@check("2. utime monotonic")
def check_utime_monotonic(ctx: SimpleNamespace) -> None:
    channel = ctx.sim.channels.franka_state_channel
    before = len(ctx.recorder.messages[channel])
    _step(ctx.sim, ctx.recorder, 50)
    window = ctx.recorder.messages[channel][before - 1:]
    diffs = {window[i + 1].utime - window[i].utime for i in range(len(window) - 1)}
    _require(diffs == {CONTROL_DT_UTIME}, f"{channel}: utime deltas {sorted(diffs)}, "
              f"expected all {CONTROL_DT_UTIME}")


@check("3. Gravity hold")
def check_gravity_hold(ctx: SimpleNamespace) -> None:
    sim, recorder = ctx.sim, ctx.recorder
    franka_ch = sim.channels.franka_state_channel
    ur_ch = sim.channels.ur_state_channel_sim
    hand_ch = sim.channels.franka_hand_state_channel
    franka_before = len(recorder.messages[franka_ch])
    ur_before = len(recorder.messages[ur_ch])
    hand_before = len(recorder.messages[hand_ch])
    _step(sim, recorder, 400)

    franka_all = recorder.messages[franka_ch]
    franka_window = franka_all[franka_before:]
    baseline = np.array(franka_window[0].position)
    for msg in franka_window:
        pos, vel, eff = np.array(msg.position), np.array(msg.velocity), np.array(msg.effort)
        _require(np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(eff).all(),
                  f"{franka_ch}: non-finite value at utime {msg.utime}")
        drift = float(np.max(np.abs(pos - baseline)))
        _require(drift <= 0.02, f"{franka_ch}: position drifted {drift:.4f} rad from its "
                  "first value")
    for i in range(franka_before, len(franka_all)):
        prev, cur = franka_all[i - 1], franka_all[i]
        damping = -FRANKA_STALE_DAMPING * np.array(prev.velocity)
        err = float(np.max(np.abs(np.array(cur.effort) - damping)))
        _require(err <= 1e-6, f"{franka_ch}: effort {list(cur.effort)} != -K*v_prev "
                  f"{damping.tolist()} (err {err:.2e})")

    ur_window = recorder.messages[ur_ch][ur_before:]
    baseline_ur = np.array(ur_window[0].position)
    for msg in ur_window:
        pos, vel = np.array(msg.position), np.array(msg.velocity)
        _require(np.isfinite(pos).all() and np.isfinite(vel).all(),
                  f"{ur_ch}: non-finite value at utime {msg.utime}")
        drift = float(np.max(np.abs(pos - baseline_ur)))
        _require(drift <= 0.02, f"{ur_ch}: position drifted {drift:.4f} rad from its first value")

    hand_window = recorder.messages[hand_ch][hand_before:]
    for msg in hand_window:
        _require(math.isfinite(msg.actual_position_mm), f"{hand_ch}: non-finite "
                  "actual_position_mm")
    # No PANDA_HAND_COMMAND yet: the default command closes the hand.
    width = hand_window[-1].actual_position_mm
    _require(abs(width) <= 1.0, f"{hand_ch}: actual_position_mm {width:.3f} not within 1 mm "
              "of 0.0 (closed by the default command)")


@check("4. Torque sign, Franka")
def check_torque_franka(ctx: SimpleNamespace) -> None:
    sim, recorder, layouts = ctx.sim, ctx.recorder, ctx.layouts
    channel = sim.channels.franka_state_channel
    input_channel = sim.channels.franka_input_channel
    effort_names = layouts["FRANKA_STATE"]["effort_names"]
    baseline_pos0 = float(recorder.messages[channel][-1].position[0])

    efforts = [0.0] * len(effort_names)
    efforts[0] = 1.0
    for _ in range(40):
        _publish_effort(recorder, input_channel, effort_names, efforts,
                         sim.step_index * CONTROL_DT_UTIME)
        _step(sim, recorder, 1)

    last = recorder.messages[channel][-1]
    v0 = float(last.velocity[0])
    _require(0.02 < v0 < 5.0, f"{channel}: panda_joint1dot {v0:.4f}, expected in (0.02, 5.0)")
    _require(float(last.position[0]) > baseline_pos0,
              f"{channel}: panda_joint1 {last.position[0]:.5f} did not increase from "
              f"{baseline_pos0:.5f}")
    _require(float(last.effort[0]) == 1.0, f"{channel}: panda_motor1 effort {last.effort[0]} "
              "!= 1.0")
    ctx.franka_burst_velocity = v0

    # Explicit zero so check 5's stale-timeout clock starts from a clean edge.
    _publish_effort(recorder, input_channel, effort_names, [0.0] * len(effort_names),
                     sim.step_index * CONTROL_DT_UTIME)
    _step(sim, recorder, 1)


@check("5. Stale damping")
def check_stale_damping(ctx: SimpleNamespace) -> None:
    sim, recorder = ctx.sim, ctx.recorder
    channel = sim.channels.franka_state_channel
    start_log = len(ctx.log_lines)
    v_at_stop = abs(ctx.franka_burst_velocity)

    _step(sim, recorder, 100)

    v_now = abs(float(recorder.messages[channel][-1].velocity[0]))
    _require(v_now < 0.1 * v_at_stop, f"{channel}: |panda_joint1dot| {v_now:.4f} not < 10% of "
              f"the stop value {v_at_stop:.4f}")
    switched = any("stale-damping" in line for line in ctx.log_lines[start_log:])
    _require(switched, "no '[LCM] franka input: stale-damping' log line seen after input "
              "stopped")


@check("6. Torque sign, UR")
def check_torque_ur(ctx: SimpleNamespace) -> None:
    sim, recorder, layouts = ctx.sim, ctx.recorder, ctx.layouts
    channel = sim.channels.ur_state_channel_sim
    input_channel = sim.channels.ur_input_channel_sim
    effort_names = layouts["UR_STATE_SIM"]["effort_names"]
    baseline_pos0 = float(recorder.messages[channel][-1].position[0])

    efforts = [0.0] * len(effort_names)
    efforts[0] = 5.0
    for _ in range(40):
        _publish_effort(recorder, input_channel, effort_names, efforts,
                         sim.step_index * CONTROL_DT_UTIME)
        _step(sim, recorder, 1)

    last = recorder.messages[channel][-1]
    v0 = float(last.velocity[0])
    _require(v0 > 0.02, f"{channel}: shoulder_pan_jointdot {v0:.4f}, expected > 0.02")
    _require(float(last.position[0]) > baseline_pos0,
              f"{channel}: shoulder_pan_joint {last.position[0]:.5f} did not increase from "
              f"{baseline_pos0:.5f}")
    _require(float(last.effort[0]) == 5.0, f"{channel}: shoulder_pan_joint_actuator effort "
              f"{last.effort[0]} != 5.0")

    # UR has no stale-input rule: send an explicit zero, stopping publishes alone won't do it.
    _publish_effort(recorder, input_channel, effort_names, [0.0] * len(effort_names),
                     sim.step_index * CONTROL_DT_UTIME)
    _step(sim, recorder, 1)
    v_at_stop = abs(float(recorder.messages[channel][-1].velocity[0]))
    _step(sim, recorder, 100)
    v_now = abs(float(recorder.messages[channel][-1].velocity[0]))
    _require(math.isfinite(v_now), f"{channel}: shoulder_pan_jointdot not finite after coast")
    _require(v_now <= v_at_stop + 0.01, f"{channel}: |shoulder_pan_jointdot| {v_now:.4f} grew "
              f"past its value at the zero command {v_at_stop:.4f} + 0.01")


@check("7. Hand command")
def check_hand_command(ctx: SimpleNamespace) -> None:
    sim, recorder = ctx.sim, ctx.recorder
    status_channel = sim.channels.franka_hand_state_channel
    output_channel = sim.channels.franka_hand_robot_output_channel
    command_channel = sim.channels.franka_hand_input_channel
    effort_limit = sim.hand_drive.effort_limit
    start_log = len(ctx.log_lines)

    def hold(width_mm: float, steps: int, utime_offset: int = 0) -> list:
        """Command ``width_mm`` every step; the status messages of those steps, in order."""
        first_utime = (sim.step_index + 1) * CONTROL_DT_UTIME
        for _ in range(steps):
            msg = lcmt_schunk_wsg_command()
            msg.utime = sim.step_index * CONTROL_DT_UTIME + utime_offset
            msg.target_position_mm, msg.force = width_mm, 0.0
            recorder.publish(command_channel, msg)
            _step(sim, recorder, 1)
        recorder.wait_for_each([status_channel, output_channel])
        return [m for m in recorder.messages[status_channel] if m.utime >= first_utime]

    def settle_time(window: list, width_mm: float) -> float:
        """Sim time from the command until the width stays within 1 mm."""
        last_out = window[0].utime - CONTROL_DT_UTIME
        for msg in window:
            if abs(msg.actual_position_mm - width_mm) > 1.0:
                last_out = msg.utime
        return (last_out + 2 * CONTROL_DT_UTIME - window[0].utime) * 1e-6

    def last_width(window: list) -> float:
        return float(window[-1].actual_position_mm)

    opened = hold(40.0, HAND_MOVE_HOLD_STEPS)
    peak = max(m.actual_position_mm for m in opened)
    _require(abs(last_width(opened) - 40.0) <= 1.0, f"{status_channel}: width "
              f"{last_width(opened):.2f} mm after {HAND_MOVE_HOLD_STEPS} steps at 40 mm, "
              "expected 40 +- 1")
    _require(peak <= 46.0, f"{status_channel}: width overshoot to {peak:.2f} mm (> 46)")
    open_time = settle_time(opened, 40.0)

    closed = hold(0.0, HAND_MOVE_HOLD_STEPS)
    _require(last_width(closed) <= 1.0, f"{status_channel}: width {last_width(closed):.2f} mm "
              f"after {HAND_MOVE_HOLD_STEPS} steps at 0 mm, expected <= 1")
    close_time = settle_time(closed, 0.0)
    for what, t in (("0 -> 40 mm", open_time), ("40 -> 0 mm", close_time)):
        _require(t <= HAND_MOVE_TIME_LIMIT, f"{what} took {t:.3f} s (> {HAND_MOVE_TIME_LIMIT})")

    squeezed = hold(-100.0, 40)
    _require(last_width(squeezed) <= 1.0, f"{status_channel}: width "
              f"{last_width(squeezed):.2f} mm at -100 mm, expected <= 1")
    effort = np.array(recorder.messages[output_channel][-1].effort)
    want = np.array([effort_limit, -effort_limit])
    _require(np.allclose(effort, want, rtol=0.0, atol=1e-6), f"{output_channel}: effort "
              f"{effort.tolist()} at -100 mm, expected the saturated {want.tolist()}")

    stale = hold(40.0, 60, utime_offset=-2_000_000)
    # 2 mm, not 1: the finger armature rebounds ~1.3 mm out of the -100 mm squeeze.
    _require(max(m.actual_position_mm for m in stale) <= 2.0, f"{status_channel}: width "
              f"reached {max(m.actual_position_mm for m in stale):.2f} mm under a 2 s stale "
              "40 mm command, expected <= 2 (closed)")
    _require(any("stale" in line and command_channel in line
                 for line in ctx.log_lines[start_log:]),
              f"no '[LCM] {command_channel} stale' warning logged")

    fresh = hold(40.0, HAND_MOVE_HOLD_STEPS)
    _require(abs(last_width(fresh) - 40.0) <= 1.0, f"{status_channel}: width "
              f"{last_width(fresh):.2f} mm after a fresh 40 mm command, expected 40 +- 1")
    hold(0.0, 60)
    print(f"[INFO] 7. open 0 -> 40 mm {open_time:.3f} s (peak {peak:.2f} mm), close 40 -> 0 mm "
          f"{close_time:.3f} s, squeeze width {last_width(squeezed):.2f} mm")


@check("8. Robotiq round trip")
def check_robotiq_round_trip(ctx: SimpleNamespace) -> None:
    sim, recorder = ctx.sim, ctx.recorder
    command_channel = sim.channels.robotiq_command_channel
    status_channel = sim.channels.robotiq_status_channel
    output_channel = sim.channels.robotiq_robot_output_channel

    def burst(position: int, speed: int, force: int, steps: int = 200, period: int = 10) -> None:
        for i in range(steps):
            if i % period == 0:
                msg = lcmt_robotiq_command()
                msg.utime = sim.step_index * CONTROL_DT_UTIME
                msg.position, msg.speed, msg.force = position, speed, force
                recorder.publish(command_channel, msg)
            _step(sim, recorder, 1)

    burst(255, 255, 0)
    status = recorder.messages[status_channel][-1]
    _require(status.position >= 200, f"{status_channel}: position {status.position} < 200 "
              "after close")
    output = recorder.messages[output_channel][-1]
    # Driver settles ~0.815-0.82 against the gripper_drive.stop mechanical limit; 0.8 was a
    # knife-edge (measured as low as 0.8085), so use 0.7 for margin without weakening the check.
    threshold = 0.7 * ROBOTIQ_PRISMATIC_RANGE
    _require(float(output.position[0]) >= threshold, f"{output_channel}: position[0] "
              f"{output.position[0]:.5f} < {threshold:.5f}")
    _require(output.position[0] == output.position[1], f"{output_channel}: position "
              f"{list(output.position)} left != right")
    _require(list(output.effort) == [0.0, 0.0], f"{output_channel}: effort "
              f"{list(output.effort)} != [0, 0]")

    burst(0, 255, 0)
    status = recorder.messages[status_channel][-1]
    _require(status.position <= 20, f"{status_channel}: position {status.position} > 20 "
              "after open")

    burst(128, 255, 0)
    status = recorder.messages[status_channel][-1]
    _require(100 <= status.position <= 156, f"{status_channel}: position {status.position} "
              "not in [100, 156] at half-command")


@check("9. Belt trigger + Franka grasp")
def check_belt_grasp(ctx: SimpleNamespace) -> None:
    sim, recorder = ctx.sim, ctx.recorder
    tip_body = body_index(list(sim.model.body_label), "panda_hand/finger_tip")
    peer = StatePeer(PRIVATE_LCM_URL, sim.channels)
    start_log = len(ctx.log_lines)
    q_t, anchor = grasp_sequence(sim, peer)
    _require(sim.belt_placed, "sim.belt_placed is not True after the grasp sequence")

    placed = [line for line in ctx.log_lines[start_log:] if line.startswith("[BELT] placed")]
    _require(len(placed) == 1, f"{len(placed)} '[BELT] placed' log lines, expected 1")
    tip_at, target_at, anchor_at = (
        np.array([float(v) for v in re.search(rf"{key}=\(([^)]*)\)", placed[0])[1].split(",")])
        for key in ("finger_tip", "target", "anchor now")
    )
    place_err = float(np.linalg.norm(anchor_at - target_at))
    depth_err = abs(float(np.linalg.norm(target_at - tip_at)) - sim.grasp_depth)
    _require(place_err <= 0.002 and depth_err <= 0.0005, f"anchor body {anchor} placed "
              f"{place_err * 1e3:.2f} mm from its target, target depth error "
              f"{depth_err * 1e3:.2f} mm")

    # Flush the backlog the grasp left in the recorder, then 0.1 s of fresh status.
    recorder.drain()
    for messages in recorder.messages.values():
        messages.clear()
    for _ in range(round(0.1 / sim.frame_dt)):
        pd_step(sim, peer, q_t, HAND_CLOSED_WIDTH)
        recorder.drain()
    status = recorder.messages[sim.channels.franka_hand_state_channel]
    # Median: at 2 substeps the finger-rod contact kicks single samples by several mm.
    width_mm = float(np.median([msg.actual_position_mm for msg in status]))
    _require(4.1 <= width_mm <= 9.1, f"median hand width {width_mm:.2f} mm after close, "
              "expected the 6.6 mm rod +- 2.5 mm")

    belt = np.asarray(sim.info.belt_bodies, dtype=np.int64)
    sample_every = round(0.1 / sim.frame_dt)
    max_dist, min_belt_z, widths = 0.0, math.inf, []
    for i in range(1, round(3.0 / sim.frame_dt) + 1):
        pd_step(sim, peer, q_t, HAND_CLOSED_WIDTH)
        if i * sim.frame_dt <= 1.0 and peer.hand is not None:
            widths.append(float(peer.hand.position[1] - peer.hand.position[0]))
        if i % sample_every:
            continue
        body_q = sim.state_0.body_q.numpy()
        _require(np.isfinite(body_q).all() and np.isfinite(sim.state_0.joint_q.numpy()).all(),
                  f"non-finite state {i * sim.frame_dt:.1f} s into the hold")
        tip, held = body_q[tip_body, :3], body_q[anchor, :3]
        dist = float(np.linalg.norm(held - tip))
        max_dist, min_belt_z = max(max_dist, dist), min(min_belt_z, float(body_q[belt, 2].min()))
        _require(dist <= 0.010, f"anchor body {dist * 1e3:.1f} mm from finger_tip "
                  f"{i * sim.frame_dt:.1f} s into the hold")
        _require(held[2] >= tip[2] - 0.010, f"anchor z {held[2]:.4f} below finger_tip z "
                  f"{tip[2]:.4f} - 0.010 at {i * sim.frame_dt:.1f} s")
    _require(min_belt_z > sim.table_top_z - 0.01, f"belt min z {min_belt_z:.4f} below the "
              f"table top {sim.table_top_z:.4f} - 0.01")

    body_q = sim.state_0.body_q.numpy()
    tip_0, anchor_0 = body_q[tip_body, :3].astype(np.float64), body_q[anchor, :3].copy()
    q_lift = finger_tip_ik(sim, tip_0 + np.array([0.0, 0.0, LIFT_HEIGHT]))
    n = round(LIFT_TIME / sim.frame_dt)
    for i in range(1, n + 1):
        s = 0.5 * (1.0 - math.cos(math.pi * i / n))
        pd_step(sim, peer, q_t + (q_lift - q_t) * s, HAND_CLOSED_WIDTH)
    body_q = sim.state_0.body_q.numpy()
    rise = float(body_q[anchor, 2] - anchor_0[2])
    lift_dist = float(np.linalg.norm(body_q[anchor, :3] - body_q[tip_body, :3]))
    _require(rise >= 0.04, f"anchor rose {rise * 1e3:.1f} mm in the lift, expected >= 40 mm")
    _require(lift_dist <= 0.010, f"anchor body {lift_dist * 1e3:.1f} mm from finger_tip "
              "after the lift")
    recorder.drain()
    print(f"[INFO] 9. width {width_mm:.2f} mm (1 s p2p {np.ptp(widths) * 1e3:.3f} mm), max "
          f"anchor-tip {max_dist * 1e3:.2f} mm, belt min z {min_belt_z:.4f}, placement error "
          f"{place_err * 1e3:.3f} mm; lift: anchor rose {rise * 1e3:.1f} mm, anchor-tip "
          f"{lift_dist * 1e3:.2f} mm")


@check("10. Channel override")
def check_channel_override(ctx: SimpleNamespace) -> None:
    if not MAGNA_LCM_CHANNELS_YAML.is_file():
        raise Skipped(f"{MAGNA_LCM_CHANNELS_YAML} not found")
    n_fields = len(fields(LcmChannels))
    _require(n_fields == 16, f"LcmChannels has {n_fields} fields, expected 16")
    overridden = LcmChannels.from_yaml(MAGNA_LCM_CHANNELS_YAML)
    _require(overridden == LcmChannels(), f"LcmChannels.from_yaml(magna) = {overridden} != "
              f"defaults {LcmChannels()}")


@check("11. Belt mesh")
def check_belt_mesh(ctx: SimpleNamespace) -> None:
    """Pure-numpy: tube_mesh() geometry + deformable_mesh_msg() wire layout (no sim step)."""
    angle = 2.0 * np.pi * np.arange(BELT_MESH_N_BODIES) / BELT_MESH_N_BODIES
    centres = np.stack(
        [0.3 * np.cos(angle), 0.3 * np.sin(angle), np.zeros(BELT_MESH_N_BODIES)], axis=1
    )
    vertices, triangles = tube_mesh(centres, BELT_MESH_RADIUS, sides=BELT_MESH_SIDES)
    _require(vertices.shape == (BELT_MESH_EXPECTED_V, 3), f"vertices shape {vertices.shape}")
    _require(triangles.shape == (BELT_MESH_EXPECTED_T, 3), f"triangles shape {triangles.shape}")
    _require(vertices.dtype == np.float32, f"vertices dtype {vertices.dtype}")
    _require(triangles.dtype == np.int32, f"triangles dtype {triangles.dtype}")
    ring_centres = np.repeat(centres, BELT_MESH_SIDES, axis=0)
    dist = np.linalg.norm(vertices.astype(np.float64) - ring_centres, axis=1)
    radius_err = float(np.max(np.abs(dist - BELT_MESH_RADIUS)))
    _require(radius_err <= 1e-6, f"tube_mesh: max radius error {radius_err:.2e}")
    _require(triangles.min() >= 0 and triangles.max() < BELT_MESH_EXPECTED_V,
              f"triangle index out of range [0, {BELT_MESH_EXPECTED_V})")

    msg = deformable_mesh_msg(vertices, triangles, BELT_MESH_NAME, BELT_MESH_COLOR)
    _require(msg.name == DEFORMABLE_LINK_NAME, f"name {msg.name!r}")
    _require(msg.robot_num == 0, f"robot_num {msg.robot_num}")
    _require(msg.num_geom == 1, f"num_geom {msg.num_geom}")
    geom = msg.geom[0]
    _require(geom.type == 4, f"geom.type {geom.type}")
    _require(list(geom.position) == [0.0, 0.0, 0.0], f"geom.position {list(geom.position)}")
    _require(list(geom.quaternion) == [1.0, 0.0, 0.0, 0.0],
              f"geom.quaternion {list(geom.quaternion)}")
    _require(list(geom.color) == [1.0, 0.5, 0.0, 1.0], f"geom.color {list(geom.color)}")
    _require(geom.string_data == "round_belt::round_belt", f"string_data {geom.string_data!r}")
    _require(geom.num_float_data == BELT_MESH_EXPECTED_NUM_FLOAT_DATA,
              f"num_float_data {geom.num_float_data} != {BELT_MESH_EXPECTED_NUM_FLOAT_DATA}")
    _require(len(geom.float_data) == BELT_MESH_EXPECTED_NUM_FLOAT_DATA,
              f"len(float_data) {len(geom.float_data)} != {BELT_MESH_EXPECTED_NUM_FLOAT_DATA}")
    _require(geom.float_data[0] == float(vertices.shape[0]),
              f"float_data[0] {geom.float_data[0]} != {vertices.shape[0]}")
    _require(geom.float_data[1] == float(triangles.shape[0]),
              f"float_data[1] {geom.float_data[1]} != {triangles.shape[0]}")


@check("12. Pulley state")
def check_pulley_state(ctx: SimpleNamespace) -> None:
    sim, recorder = ctx.sim, ctx.recorder
    channel = sim.channels.round_belt_pulley_state_channel
    labels = [str(label) for label in sim.model.joint_label]
    joints = [labels.index(label) for label in PULLEY_JOINT_LABELS]
    coords = [int(sim.model.joint_q_start.numpy()[j]) for j in joints]
    dofs = [int(sim.model.joint_qd_start.numpy()[j]) for j in joints]
    recorder.drain()
    recorder.messages[channel].clear()
    for _ in range(PULLEY_STATE_STEPS):
        before = len(recorder.messages[channel])
        sim.control_step()
        deadline = time.monotonic() + LAYOUT_WAIT_S
        while len(recorder.messages[channel]) == before and time.monotonic() < deadline:
            recorder.peer.handle_timeout(LAYOUT_POLL_MS)
        msgs = recorder.messages[channel][before:]
        _require(len(msgs) == 1, f"{channel}: {len(msgs)} messages after step {sim.step_index}, "
                 "expected 1")
        msg = msgs[0]
        _require(msg.utime == sim.step_index * CONTROL_DT_UTIME,
                 f"{channel}: utime {msg.utime} != {sim.step_index * CONTROL_DT_UTIME}")
        _require(msg.object_name == PULLEY_OBJECT_NAME,
                 f"{channel}: object_name {msg.object_name!r} != {PULLEY_OBJECT_NAME!r}")
        _require(msg.num_positions == 2 and msg.num_velocities == 2,
                 f"{channel}: num_positions/num_velocities {msg.num_positions}/"
                 f"{msg.num_velocities}, expected 2/2")
        _require(list(msg.position_names) == PULLEY_POSITION_NAMES,
                 f"{channel}: position_names {list(msg.position_names)}")
        _require(list(msg.velocity_names) == PULLEY_VELOCITY_NAMES,
                 f"{channel}: velocity_names {list(msg.velocity_names)}")
        position, velocity = np.array(msg.position), np.array(msg.velocity)
        _require(np.isfinite(position).all() and np.isfinite(velocity).all(),
                 f"{channel}: non-finite values {position.tolist()} / {velocity.tolist()}")
        want_q = sim.state_0.joint_q.numpy()[coords].astype(np.float64)
        want_qd = sim.state_0.joint_qd.numpy()[dofs].astype(np.float64)
        _require(np.allclose(position, want_q, atol=1e-9, rtol=0.0)
                 and np.allclose(velocity, want_qd, atol=1e-9, rtol=0.0),
                 f"{channel}: {position.tolist()} / {velocity.tolist()} != state joint_q "
                 f"{want_q.tolist()} / joint_qd {want_qd.tolist()}")
    last = recorder.messages[channel][-1]
    print(f"[INFO] 12. {PULLEY_STATE_STEPS} messages; last q {np.round(last.position, 5).tolist()} "
          f"rad, qd {np.round(last.velocity, 5).tolist()} rad/s")


def main() -> int:
    layouts = json.loads(LAYOUTS_PATH.read_text())
    log_lines: list[str] = []
    logger.add(log_lines.append, level="INFO", format="{message}")

    # Build exactly as scripts/round_belt_lcm_simulation.py does, but headless / non-realtime.
    parser = RoundBeltLcmSimulation.create_parser()
    parser.set_defaults(viewer="null", realtime=False, cameras=False, lcm_url=PRIVATE_LCM_URL)
    viewer, args = newton.examples.init(parser)
    sim = RoundBeltLcmSimulation(viewer, args)
    recorder = Recorder(sim.channels)
    ctx = SimpleNamespace(sim=sim, recorder=recorder, layouts=layouts, log_lines=log_lines)

    exit_code = 0
    for name, fn in CHECKS:
        try:
            fn(ctx)
        except Skipped as exc:
            print(f"[SKIP] {name}: {exc}")
            continue
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            exit_code = 1
            break
        print(f"[PASS] {name}")

    viewer.close()
    if exit_code == 0:
        print("ALL LCM CONTRACT CHECKS PASSED")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
