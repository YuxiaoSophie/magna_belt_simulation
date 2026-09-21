#!/usr/bin/env python3
"""Headless check that the round-belt task-board pulleys are damped free VBD axles.

Builds the position-PD ``RoundBeltTaskSimulation`` (null viewer, cameras off, default substeps
and iterations) and drives the pulley dofs through ``control.joint_f`` only:

    T0  build: 2 pulley bodies/joints on world revolute joints, axis = board normal, centres at
        the board weld composed with the URDF joint origins (transcribed here)
    T1  2 s gravity hold with zero torque: no rotation, no axle drift or sag
    T2  2e-3 N m on the small pulley for 0.5 s reaches torque / damping, then spins down with
        the time constant Izz / damping once released
    T3  the same for the large pulley

Run:
    uv run python scripts/checks/check_pulleys.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[2]
# The task packages live under src/; make them importable regardless of CWD.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import newton.examples

from round_belt_task.simulation import RoundBeltTaskSimulation

# Transcribed from round_belt_scene.yaml (board weld) and round_belt_task_board.urdf (joints).
BOARD_XYZ = (0.64483928, -0.19718233, 0.01076393)
BOARD_RPY_DEG = (-3.32822058e-01, -6.87450103e-02, 8.95207485e01)
PULLEYS = (
    ("small", "board/small_round_pulley_joint", (0.355, 0.196, 0.0248)),
    ("large", "board/large_round_pulley_joint", (0.140, 0.196, 0.0248)),
)
PULLEY_IZZ = 6.0454911e-05  # kg m^2
PULLEY_DAMPING = 0.001  # N m s/rad

POSE_ATOL = 1e-6
HOLD_TIME = 2.0
HOLD_MAX_ANGLE = 0.005
MAX_DRIFT = 0.0005
SPIN_TORQUE = 2e-3
SPIN_TIME = 0.5
SPEED_TOL = 0.15
ANGLE_TOL = 0.2
OTHER_MAX_ANGLE = 0.01
COAST_TIME = 0.5
COAST_MAX_SPEED_FRACTION = 0.05
NO_EFFECT_ANGLE = 0.01


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _rot_rpy(rpy_rad) -> np.ndarray:
    r, p, y = rpy_rad
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


R_W_BOARD = _rot_rpy([math.radians(a) for a in BOARD_RPY_DEG])


class Rig:
    """The sim plus per-pulley index lookups and a frame-stepping recorder."""

    def __init__(self) -> None:
        parser = RoundBeltTaskSimulation.create_parser()
        parser.set_defaults(viewer="null", cameras=False)
        viewer, args = newton.examples.init(parser)
        self.viewer = viewer
        self.sim = sim = RoundBeltTaskSimulation(viewer, args)
        model = sim.model
        labels = [str(label) for label in model.joint_label]
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        self.joints = [labels.index(label) for _, label, _ in PULLEYS]
        self.coords = [int(q_start[j]) for j in self.joints]
        self.dofs = [int(qd_start[j]) for j in self.joints]
        self.bodies = [int(model.joint_child.numpy()[j]) for j in self.joints]
        # Allocated before the first (uncaptured) frame: the CUDA graph replays this buffer.
        if sim.control.joint_f is None:
            sim.control.joint_f = wp.zeros(
                int(model.joint_dof_count), dtype=wp.float32, device=model.device
            )
        self.joint_f = np.zeros(int(model.joint_dof_count), dtype=np.float32)
        self.apply_torque(None, 0.0)
        self.centres0 = self.centres()

    def apply_torque(self, pulley: int | None, torque: float) -> None:
        self.joint_f[:] = 0.0
        if pulley is not None:
            self.joint_f[self.dofs[pulley]] = torque
        self.sim.control.joint_f.assign(self.joint_f)

    def centres(self) -> np.ndarray:
        return self.sim.state_0.body_q.numpy()[self.bodies, :3].astype(np.float64)

    def q(self) -> np.ndarray:
        return self.sim.state_0.joint_q.numpy()[self.coords].astype(np.float64)

    def qd(self) -> np.ndarray:
        return self.sim.state_0.joint_qd.numpy()[self.dofs].astype(np.float64)

    def run(self, seconds: float) -> SimpleNamespace:
        """Step ``seconds``; max centre translation and z sag [m] per pulley over the window."""
        drift, sag = np.zeros(len(PULLEYS)), np.zeros(len(PULLEYS))
        for _ in range(round(seconds / self.sim.frame_dt)):
            self.sim.step()
            centres = self.centres()
            _require(np.isfinite(centres).all() and np.isfinite(self.q()).all(),
                     f"non-finite pulley state at t {self.sim.sim_time:.3f} s")
            drift = np.maximum(drift, np.linalg.norm(centres - self.centres0, axis=1))
            sag = np.maximum(sag, self.centres0[:, 2] - centres[:, 2])
        return SimpleNamespace(drift=drift, sag=sag)


def check_build(rig: Rig) -> str:
    sim, model = rig.sim, rig.sim.model
    _require(len(sim.info.pulley_bodies) == 2 and len(sim.info.pulley_joints) == 2,
             f"info.pulley_bodies {sim.info.pulley_bodies} / pulley_joints "
             f"{sim.info.pulley_joints}, expected 2 each")
    _require(sorted(rig.joints) == sorted(sim.info.pulley_joints),
             f"pulley joint labels resolve to {rig.joints}, info has {sim.info.pulley_joints}")
    parents = model.joint_parent.numpy()
    joint_x_p = model.joint_X_p.numpy()
    axes = model.joint_axis.numpy()
    body_q = sim.state_0.body_q.numpy()
    board_z = R_W_BOARD @ np.array([0.0, 0.0, 1.0])
    for i, (name, _label, centre) in enumerate(PULLEYS):
        joint, body = rig.joints[i], rig.bodies[i]
        _require(int(parents[joint]) == -1, f"{name}: joint parent {parents[joint]}, expected -1")
        _require(body in sim.vbd_bodies and joint in sim.vbd_joints,
                 f"{name}: body {body} / joint {joint} not owned by the VBD entry")
        damping = float(model.joint_damping.numpy()[rig.dofs[i]])
        _require(np.isclose(damping, PULLEY_DAMPING),
                 f"{name}: joint damping {damping}, expected {PULLEY_DAMPING}")
        X_p = wp.transform(*[float(v) for v in joint_x_p[joint]])
        axis = np.array(wp.quat_rotate(wp.transform_get_rotation(X_p),
                                       wp.vec3(*[float(v) for v in axes[rig.dofs[i]]])))
        # The board is welded 0.33 deg off level, so the axle is the board normal, not world z.
        _require(np.allclose(axis, board_z, atol=POSE_ATOL, rtol=0.0),
                 f"{name}: world axis {axis.tolist()} != board normal {board_z.tolist()}")
        want = np.asarray(BOARD_XYZ) + R_W_BOARD @ np.asarray(centre)
        got = body_q[body, :3].astype(np.float64)
        _require(np.allclose(got, want, atol=POSE_ATOL, rtol=0.0),
                 f"{name}: body at {got.tolist()}, expected {want.tolist()} "
                 f"(err {np.abs(got - want).max():.2e})")
    return f"bodies {rig.bodies}, joints {rig.joints}, board normal {np.round(board_z, 6).tolist()}"


def check_hold(rig: Rig) -> str:
    rig.apply_torque(None, 0.0)
    window = rig.run(HOLD_TIME)
    q = rig.q()
    _require(np.all(np.abs(q) < HOLD_MAX_ANGLE), f"joint_q {q.tolist()} after a {HOLD_TIME:g} s "
             f"zero-torque hold (limit {HOLD_MAX_ANGLE})")
    _require(np.all(window.drift < MAX_DRIFT) and np.all(window.sag < MAX_DRIFT),
             f"centre drift {(window.drift * 1e3).tolist()} mm / sag "
             f"{(window.sag * 1e3).tolist()} mm (limit {MAX_DRIFT * 1e3:g} mm)")
    return (f"q {q.tolist()} rad, drift {np.round(window.drift * 1e3, 4).tolist()} mm, "
            f"sag {np.round(window.sag * 1e3, 4).tolist()} mm")


def _spin(rig: Rig, pulley: int) -> str:
    other = 1 - pulley
    name = PULLEYS[pulley][0]
    tau = PULLEY_IZZ / PULLEY_DAMPING
    speed = SPIN_TORQUE / PULLEY_DAMPING
    q0 = rig.q()
    rig.apply_torque(pulley, SPIN_TORQUE)
    window = rig.run(SPIN_TIME)
    q1, qd1 = rig.q(), rig.qd()
    angle = q1[pulley] - q0[pulley]
    if abs(angle) < NO_EFFECT_ANGLE:
        raise AssertionError(f"{name}: joint_f {SPIN_TORQUE:g} N m moved the pulley only "
                             f"{angle:.2e} rad; joint_f does not reach the VBD joint")
    want_angle = speed * (SPIN_TIME - tau * (1.0 - math.exp(-SPIN_TIME / tau)))
    _require(abs(qd1[pulley] - speed) <= SPEED_TOL * speed,
             f"{name}: joint_qd {qd1[pulley]:.4f} rad/s under {SPIN_TORQUE:g} N m, expected "
             f"torque / damping = {speed:g} +- {SPEED_TOL:.0%}")
    _require(abs(angle - want_angle) <= ANGLE_TOL * want_angle,
             f"{name}: turned {angle:.4f} rad in {SPIN_TIME:g} s, expected {want_angle:.4f} "
             f"+- {ANGLE_TOL:.0%}")
    _require(window.drift[pulley] < MAX_DRIFT, f"{name}: centre moved "
             f"{window.drift[pulley] * 1e3:.4f} mm while driven (limit {MAX_DRIFT * 1e3:g} mm)")
    other_change = abs(q1[other] - q0[other])
    _require(other_change < OTHER_MAX_ANGLE, f"{PULLEYS[other][0]}: turned {other_change:.4f} "
             f"rad while only {name} was driven")

    rig.apply_torque(None, 0.0)
    coast = rig.run(COAST_TIME)
    q2, qd2 = rig.q(), rig.qd()
    coast_angle = q2[pulley] - q1[pulley]
    want_coast = qd1[pulley] * tau * (1.0 - math.exp(-COAST_TIME / tau))
    _require(abs(qd2[pulley]) < COAST_MAX_SPEED_FRACTION * qd1[pulley],
             f"{name}: still at {qd2[pulley]:.4f} rad/s {COAST_TIME:g} s after release "
             f"(damping time constant {tau:.3f} s)")
    _require(abs(coast_angle - want_coast) <= ANGLE_TOL * want_coast,
             f"{name}: coasted {coast_angle:.4f} rad, expected {want_coast:.4f} +- {ANGLE_TOL:.0%}")
    _require(coast.drift[pulley] < MAX_DRIFT, f"{name}: centre moved "
             f"{coast.drift[pulley] * 1e3:.4f} mm while coasting")
    return (f"qd {qd1[pulley]:.4f} rad/s (torque/damping {speed:g}), angle {angle:.4f} rad "
            f"(expected {want_angle:.4f}); coast {coast_angle:.4f} rad (expected "
            f"{want_coast:.4f}), qd after {COAST_TIME:g} s {qd2[pulley]:.2e} rad/s; drift "
            f"driven/coast {window.drift[pulley] * 1e3:.4f}/{coast.drift[pulley] * 1e3:.4f} mm, "
            f"other q {other_change:.2e} rad")


CHECKS = [
    ("T0 build, axes and centres", check_build),
    ("T1 gravity hold", check_hold),
    ("T2 small pulley spin", lambda rig: _spin(rig, 0)),
    ("T3 large pulley spin", lambda rig: _spin(rig, 1)),
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
        print("ALL PULLEY CHECKS PASSED")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
