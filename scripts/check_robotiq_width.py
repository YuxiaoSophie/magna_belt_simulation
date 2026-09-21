#!/usr/bin/env python3
"""Headless check that a ``ROBOTIQ_COMMAND`` byte opens the 2F-85 jaw to the width it means.

Robotiq's 2F-85 POSITION REQUEST register is quasi-linear in jaw WIDTH -- 0x00 open,
0xFF closed, "Opening / count: 0.4 mm" over the 85 mm stroke -- so the byte must map to a pad
gap, not to the four-bar driver angle (which spreads 87 mm/rad open against 121 mm/rad at the
stop).  ``gripper_drive.width_calibration`` in round_belt_lcm_sim.yaml inverts that for our
ALOHA fingertips; this check measures the jaw and holds the mapping to it:

    T0  the byte -> driver target table: byte 0 at the open value, 0xFF at the full-close
        target (the overdrive that makes the grip), monotone in between
    T1  free air, a byte sweep: the minimum distance between the two pad collision meshes is
        within 1 mm of ``open_gap * (1 - byte / 255)``, endpoints included
    T2  the published ``ROBOTIQ_STATUS`` position echoes the reached command byte

Run:
    uv run python scripts/check_robotiq_width.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # This script lives under scripts/; make the repo root importable regardless of CWD.
    sys.path.insert(0, str(REPO_ROOT))

import lcm
import newton.examples

from robotiq import lcmt_robotiq_command, lcmt_robotiq_status
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation

# A private multicast group: this check never disturbs a running magna stack.
PRIVATE_LCM_URL = "udpm://239.255.76.68:7669?ttl=0"
CONTROL_DT_UTIME = 5000  # 5 ms control step (task_common/lcm_contract.py CONTROL_DT)
SWEEP_BYTES = (0, 32, 63, 96, 127, 160, 191, 203, 223, 255)
WIDTH_TOL_MM = 1.0
SETTLE_STEPS = 300  # 1.5 s; a full stroke takes under 1 s
COMMAND_PERIOD = 10
SETTLED_SPEED = 0.05  # rad/s
STATUS_TOL = 3  # counts


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _rotate(quat: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Rotate ``points`` by the (x, y, z, w) quaternion, without pulling in scipy."""
    axis, w = quat[:3], quat[3]
    t = 2.0 * np.cross(np.broadcast_to(axis, points.shape), points)
    return points + w * t + np.cross(np.broadcast_to(axis, t.shape), t)


class Rig:
    """The LCM sim, a peer that speaks ``ROBOTIQ_COMMAND``, and the pad-gap measurement."""

    def __init__(self) -> None:
        parser = RoundBeltLcmSimulation.create_parser()
        parser.set_defaults(viewer="null", cameras=False, realtime=False,
                            lcm_url=PRIVATE_LCM_URL)
        viewer, args = newton.examples.init(parser)
        self.viewer = viewer
        self.sim = sim = RoundBeltLcmSimulation(viewer, args)
        model = sim.model
        shapes = list(sim.info.gripper_pad_shapes)
        _require(len(shapes) == 2, f"expected 2 gripper pad shapes, got {shapes}")
        self.pad_bodies = [int(model.shape_body.numpy()[s]) for s in shapes]
        transforms = model.shape_transform.numpy()
        scales = model.shape_scale.numpy()
        self.pad_vertices = []
        for shape in shapes:
            vertices = np.asarray(model.shape_source[shape].vertices, dtype=np.float64)
            local = transforms[shape].astype(np.float64)
            self.pad_vertices.append(_rotate(local[3:7], vertices * scales[shape]) + local[:3])
        self.peer = lcm.LCM(PRIVATE_LCM_URL)
        self.status: lcmt_robotiq_status | None = None
        self.samples: list[tuple[int, float, int]] = []
        self.peer.subscribe(sim.channels.robotiq_status_channel, self._on_status)

    def _on_status(self, _channel: str, data: bytes) -> None:
        self.status = lcmt_robotiq_status.decode(data)

    def gap_mm(self) -> float:
        """Minimum distance between the two pad collision meshes [mm]."""
        body_q = self.sim.state_0.body_q.numpy().astype(np.float64)
        world = []
        for body, vertices in zip(self.pad_bodies, self.pad_vertices):
            pose = body_q[body]
            world.append(_rotate(pose[3:7], vertices) + pose[:3])
        best = np.inf
        for chunk in np.array_split(world[0], max(1, len(world[0]) // 256)):
            best = min(best, np.linalg.norm(chunk[:, None, :] - world[1][None, :, :],
                                            axis=2).min())
        return float(best) * 1e3

    def command(self, position: int, steps: int = SETTLE_STEPS) -> None:
        """Publish ``position`` at the controller's rate for ``steps`` sim steps."""
        for i in range(steps):
            if i % COMMAND_PERIOD == 0:
                msg = lcmt_robotiq_command()
                msg.utime = self.sim.step_index * CONTROL_DT_UTIME
                msg.position, msg.speed, msg.force = position, 255, 0
                self.peer.publish(self.sim.channels.robotiq_command_channel, msg.encode())
            self.sim.control_step()
            while self.peer.handle_timeout(0) > 0:
                pass

    def driver(self) -> tuple[np.ndarray, np.ndarray]:
        sim = self.sim
        return (sim.state_0.joint_q.numpy()[sim._driver_coords].astype(np.float64),
                sim.state_0.joint_qd.numpy()[sim._driver_dofs].astype(np.float64))


def check_map(rig: Rig) -> str:
    sim = rig.sim
    targets = sim._robotiq_targets
    open_q, span = sim._driver_open, sim._driver_span
    _require(targets.shape == (256, len(open_q)), f"target table shape {targets.shape}")
    _require(np.allclose(targets[0], open_q, rtol=0.0, atol=1e-12),
             f"byte 0 targets {targets[0].tolist()}, expected the open values {open_q.tolist()}")
    _require(np.allclose(targets[255], open_q + span, rtol=0.0, atol=1e-12),
             f"byte 255 targets {targets[255].tolist()}, expected the full-close "
             f"{(open_q + span).tolist()} (the overdrive past the stop is the grip)")
    _require(np.all(np.diff(targets, axis=0) >= 0.0), "byte -> target is not monotone")
    gap = sim._robotiq_open_gap
    _require(gap is not None, "no gripper_drive.width_calibration: the byte -> width map is off")
    angles = sim._robotiq_cal_angles
    stop = float(sim.model.joint_limit_upper.numpy()[sim._driver_dofs].max())
    _require(angles[0] == open_q[0] and angles[-1] >= stop - 1e-6,
             f"calibration spans {angles[0]}..{angles[-1]} rad, not the open {open_q[0]} to the "
             f"stop {stop}")
    return (f"open gap {gap * 1e3:.2f} mm, {len(angles)} calibration points "
            f"{angles[0]:.3f}..{angles[-1]:.3f} rad, byte 63/127/191 -> "
            f"{targets[63, 0]:.4f}/{targets[127, 0]:.4f}/{targets[191, 0]:.4f} rad")


def check_sweep(rig: Rig) -> str:
    sim = rig.sim
    open_gap = sim._robotiq_open_gap * 1e3
    rows, worst = [], 0.0
    for position in SWEEP_BYTES:
        rig.command(position)
        q, qd = rig.driver()
        gap = rig.gap_mm()
        _require(rig.status is not None, "no ROBOTIQ_STATUS published")
        rig.samples.append((position, gap, int(rig.status.position)))
        want = open_gap * (1.0 - position / 255.0)
        error = gap - want
        worst = max(worst, abs(error))
        rows.append(f"{position}:{gap:.2f}")
        _require(np.abs(qd).max() < SETTLED_SPEED, f"byte {position}: driver still at "
                 f"{np.abs(qd).max():.3f} rad/s after {SETTLE_STEPS} steps")
        _require(abs(error) <= WIDTH_TOL_MM, f"byte {position}: pad gap {gap:.2f} mm at driver "
                 f"{q[0]:.4f} rad, expected {want:.2f} mm +- {WIDTH_TOL_MM:g} "
                 f"(error {error:+.2f} mm)")
    return (f"max |error| {worst:.2f} mm over {len(SWEEP_BYTES)} bytes; byte:gap[mm] "
            f"{' '.join(rows)}")


def check_status_echo(rig: Rig) -> str:
    """A reached command must read back as itself: the status byte is on the same width scale."""
    for position, _gap, got in rig.samples:
        _require(abs(got - position) <= STATUS_TOL, f"byte {position}: ROBOTIQ_STATUS position "
                 f"{got}, expected {position} +- {STATUS_TOL}")
    return " ".join(f"{position}->{got}" for position, _gap, got in rig.samples)


CHECKS = [
    ("T0 byte -> driver target map", check_map),
    ("T1 free-air width sweep", check_sweep),
    ("T2 status echo", check_status_echo),
]


def main() -> int:
    rig = Rig()
    exit_code = 0
    for name, fn in CHECKS:
        try:
            detail = fn(rig)
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            exit_code = 1
            break
        print(f"[PASS] {name}: {detail}")
    rig.viewer.close()
    if exit_code == 0:
        print("ALL ROBOTIQ WIDTH CHECKS PASSED")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
