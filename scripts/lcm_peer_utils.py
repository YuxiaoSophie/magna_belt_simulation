"""In-script LCM peer tooling for the round-belt LCM simulation checks and benches.

A finger-tip IK on the sim's own model, a state subscriber, a joint-space PD command for the
Franka arm, the ``PANDA_HAND_COMMAND`` hand command, and the grasp sequence that drives the
Franka onto the belt trigger and closes the fingers.  Import from a script under ``scripts/``
with ``sys.path.insert(0, str(Path(__file__).parent))``.
"""

from __future__ import annotations

import lcm
import newton
import numpy as np
import warp as wp

from dairlib import lcmt_robot_input, lcmt_robot_output
from drake import lcmt_schunk_wsg_command
from round_belt_task.constants import BELT_TRIGGER_BODY
from task_common.lcm_contract import LcmChannels
from utils.labels import body_index

FRANKA_KP = (300.0, 300.0, 300.0, 300.0, 100.0, 100.0, 50.0)
FRANKA_KD = (30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0)
FRANKA_TAU_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
HAND_OPEN_WIDTH = 0.040
# 2.7 mm clearance per side on the 6.6 mm rod: the close catches the placed belt before it sags.
HAND_APPROACH_WIDTH = 0.012
# The assembly controller's closed command (-0.05 * 2000 mm): a saturated squeeze.
HAND_CLOSED_WIDTH = -0.100
MAX_MESSAGES_PER_DRAIN = 256
DRAIN_WAIT_MS = 20


def finger_tip_ik(sim, target_xyz, tol: float = 0.002, iters: int = 100) -> np.ndarray:
    """Franka arm joint angles putting ``panda_hand/finger_tip`` at ``target_xyz`` (DLS IK)."""
    model = sim.model
    coords = np.asarray(sim.info.joint_config["arm_coord_indices"][:7], dtype=np.int64)
    tip_body = body_index(list(model.body_label), BELT_TRIGGER_BODY)
    scratch = model.state()
    joint_q_host = sim.state_0.joint_q.numpy().copy()
    joint_q = wp.array(joint_q_host, dtype=wp.float32, device=model.device)
    joint_qd = wp.zeros(int(model.joint_dof_count), dtype=wp.float32, device=model.device)
    target = np.asarray(target_xyz, dtype=np.float64)
    eps, damping, gain = 1e-4, 0.01, 0.5

    def tip(q: np.ndarray) -> np.ndarray:
        joint_q_host[coords] = q
        joint_q.assign(joint_q_host)
        newton.eval_fk(model, joint_q, joint_qd, scratch)
        return scratch.body_q.numpy()[tip_body, :3].astype(np.float64)

    q = joint_q_host[coords].astype(np.float64)
    for _ in range(iters):
        p = tip(q)
        error = target - p
        if np.linalg.norm(error) <= 0.1 * tol:
            break
        jac = np.empty((3, 7))
        for i in range(7):
            dq = q.copy()
            dq[i] += eps
            jac[:, i] = (tip(dq) - p) / eps
        step = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), error)
        q = q + gain * step
    final = float(np.linalg.norm(target - tip(q)))
    if final > tol:
        raise RuntimeError(f"finger_tip IK: final error {final * 1e3:.2f} mm > {tol * 1e3:g} mm")
    return q


class StatePeer:
    """Latest ``FRANKA_STATE`` and ``FRANKA_HAND_ROBOT_OUTPUT`` messages on ``url``."""

    def __init__(self, url: str, channels: LcmChannels) -> None:
        self.lc = lcm.LCM(url)
        self.channels = channels
        self.latest: dict[str, lcmt_robot_output] = {}
        self._echoes = self._pending = 0
        for channel in (channels.franka_state_channel, channels.franka_hand_robot_output_channel):
            self.lc.subscribe(channel, self._on_message)
        for channel in (channels.franka_input_channel, channels.franka_hand_input_channel):
            self.lc.subscribe(channel, self._on_echo)

    def _on_message(self, channel: str, data: bytes) -> None:
        self.latest[channel] = lcmt_robot_output.decode(data)

    def _on_echo(self, channel: str, data: bytes) -> None:
        self._echoes += 1

    def drain(self) -> None:
        # A zero timeout races the loopback delivery of the step just published (stale PD).
        if self.lc.handle_timeout(DRAIN_WAIT_MS) <= 0:
            return
        for _ in range(MAX_MESSAGES_PER_DRAIN):
            if self.lc.handle_timeout(0) <= 0:
                break

    @property
    def franka(self) -> lcmt_robot_output | None:
        return self.latest.get(self.channels.franka_state_channel)

    @property
    def hand(self) -> lcmt_robot_output | None:
        return self.latest.get(self.channels.franka_hand_robot_output_channel)

    def positions(self, msg: lcmt_robot_output) -> dict[str, float]:
        return dict(zip(msg.position_names, msg.position))

    def velocities(self, msg: lcmt_robot_output) -> dict[str, float]:
        return dict(zip(msg.velocity_names, msg.velocity))

    def publish(self, channel: str, msg: object) -> None:
        self.lc.publish(channel, msg.encode())
        self._pending += 1

    def wait_delivered(self) -> None:
        """Block until the loopback echo of every published command arrived (bounded)."""
        # Otherwise the sim's non-blocking drain misses some commands: a jittery 5-10 ms lag.
        for _ in range(MAX_MESSAGES_PER_DRAIN):
            if self._echoes >= self._pending or self.lc.handle_timeout(DRAIN_WAIT_MS) <= 0:
                break
        self._echoes = self._pending = 0


def _input_msg(names, efforts, utime: int) -> lcmt_robot_input:
    msg = lcmt_robot_input()
    msg.utime = int(utime)
    msg.effort_names = list(names)
    msg.efforts = [float(v) for v in efforts]
    msg.num_efforts = len(msg.efforts)
    return msg


def franka_pd_msg(
    state: lcmt_robot_output, q_target, kp=FRANKA_KP, kd=FRANKA_KD, utime: int = 0
) -> lcmt_robot_input:
    """Joint PD torques for the Franka arm (the sim adds gravity), for ``franka_input_channel``."""
    q, qd = np.asarray(state.position[:7]), np.asarray(state.velocity[:7])
    tau = np.asarray(kp) * (np.asarray(q_target) - q) - np.asarray(kd) * qd
    tau = np.clip(tau, -FRANKA_TAU_LIMIT, FRANKA_TAU_LIMIT)
    return _input_msg(state.effort_names[:7], tau, utime)


def hand_command_msg(width_m: float, utime: int) -> lcmt_schunk_wsg_command:
    """``width_m`` [m] as a hand command, for ``franka_hand_input_channel``."""
    msg = lcmt_schunk_wsg_command()
    msg.utime = int(utime)
    msg.target_position_mm = float(width_m) * 1000.0
    msg.force = 0.0
    return msg


def pd_step(sim, peer: StatePeer, q_target, width: float) -> None:
    """Drain, publish the Franka PD and the hand command, then one ``sim.control_step()``."""
    peer.drain()
    utime = sim.step_index * round(1e6 * sim.frame_dt)
    if peer.franka is not None and q_target is not None:
        peer.publish(sim.channels.franka_input_channel,
                     franka_pd_msg(peer.franka, q_target, utime=utime))
    peer.publish(sim.channels.franka_hand_input_channel, hand_command_msg(width, utime))
    peer.wait_delivered()
    sim.control_step()


def _steps(sim, seconds: float) -> int:
    return max(1, round(seconds / sim.frame_dt))


def grasp_sequence(
    sim, peer: StatePeer, *, open_width: float = HAND_APPROACH_WIDTH, approach_s: float = 2.0,
    settle_s: float = 0.5, close_s: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Open the hand, move the finger tip onto the belt trigger, close; ``(q_t, anchor_body)``.

    The hand closes as soon as the belt is placed: left open, the belt sags out of the fingers.
    """
    coords = np.asarray(sim.info.joint_config["arm_coord_indices"][:7], dtype=np.int64)
    tip_body = body_index(list(sim.model.body_label), BELT_TRIGGER_BODY)
    q_start = sim.state_0.joint_q.numpy()[coords].astype(np.float64)
    for _ in range(_steps(sim, settle_s)):
        pd_step(sim, peer, q_start, open_width)

    q_t = finger_tip_ik(sim, sim.belt_trigger_point)
    q_0 = sim.state_0.joint_q.numpy()[coords].astype(np.float64)
    n = _steps(sim, approach_s)
    for i in range(1, n + 1):
        width = HAND_CLOSED_WIDTH if sim.belt_placed else open_width
        pd_step(sim, peer, q_0 + (q_t - q_0) * (i / n), width)
    for _ in range(_steps(sim, settle_s)):
        pd_step(sim, peer, q_t, HAND_CLOSED_WIDTH if sim.belt_placed else open_width)

    tip = sim.state_0.body_q.numpy()[tip_body, :3].astype(np.float64)
    gap = float(np.linalg.norm(tip - sim.belt_trigger_point))
    if gap >= sim.belt_trigger_tolerance:
        raise RuntimeError(f"finger_tip {gap * 1e3:.2f} mm from the trigger point after approach")
    if not sim.belt_placed:
        raise RuntimeError("belt trigger did not fire during the approach")

    for _ in range(_steps(sim, close_s)):
        pd_step(sim, peer, q_t, HAND_CLOSED_WIDTH)
    return q_t, sim.belt_anchor_body
