#!/usr/bin/env python3
"""Headless behaviour checks of the timing belt: MuJoCo D6 chain (``belt.py``) or VBD strip (``belt_strip.py``).

Isolated scenes (no robots) asserting the belt spec S0-S5 (chain also S4b), then the
performance numbers [PERF] of the S5 scene with CUDA graph capture. The strip runs at
SOFT_STRIP_OPERATING_POINT (HARD bars pass/fail) or, with --sweep, the characterisation table.

Run:
    uv run python scripts/check_timing_belt_behaviour.py --model strip
    uv run python scripts/check_timing_belt_behaviour.py --model strip --sweep
    uv run python scripts/check_timing_belt_behaviour.py --model chain --num-elements 68 --device cuda:0
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import newton
from newton.solvers import SolverMuJoCo, SolverVBD

import round_belt
from timing_belt_task.belt import (
    BeltMaterial,
    BeltSection,
    add_timing_belt_chain,
    belt_joint_count,
    ellipse_points,
    two_pulley_path,
    two_pulley_path_length,
)
from timing_belt_task.belt_strip import (
    POISSON,
    SOFT_STRIP_CALIBRATED_EDGE_KE,
    SOFT_STRIP_OPERATING_POINT,
    StripMaterial,
    StripSection,
    add_timing_belt_strip,
    strip_centreline,
    strip_row_vectors,
)

SECTION = BeltSection(width=0.025, thickness=0.0022, density=1027.0)
DAMPING = 1.0e-3
ARMATURE = 2.0e-6
ARMATURE_ALT = 1.0e-5
MUJOCO_NJMAX, MUJOCO_NCONMAX = 768, 384
GRAVITY = 9.81
W_PER_LENGTH = SECTION.width * SECTION.thickness * SECTION.density * GRAVITY
PITCH_LENGTH = 0.675
FPS, SUBSTEPS = 60, 10
FRAME_DT = 1.0 / FPS
SETTLE_SPEED, SETTLE_TIME, MIN_SETTLE_TIME = 1.0e-3, 6.0, 0.5
HALF_THICK = 0.5 * SECTION.thickness

# S5 geometry (board-local, Z up): (xy, body r, body z, plate r, bottom z, top z), h = 0.027 / 0.0025.
PULLEYS = {
    "small": ((0.348, 0.058), 0.02505, 0.025, 0.0275, 0.01025, 0.03975, (0.0115, 0.0385)),
    "large": ((0.134, 0.058), 0.047325, 0.027, 0.0495, 0.01225, 0.04175, (0.0135, 0.0405)),
}
BELT_Z = 0.026

ARGS = argparse.Namespace(num_elements=135, ei=2.0e-4, gj=2.0e-4, armature=ARMATURE)
RESULTS: dict[str, float] = {}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _in_band(name: str, value: float, expected: float, rel: float) -> str:
    lo, hi = expected * (1 - rel), expected * (1 + rel)
    _require(lo <= value <= hi, f"{name} = {value:.6g} outside [{lo:.6g}, {hi:.6g}]")
    return f"{name} {value:.6g} in [{lo:.6g}, {hi:.6g}]"


# ----------------------------------------------------------------------------
# Scene plumbing
# ----------------------------------------------------------------------------


def make_builder(gravity: float) -> newton.ModelBuilder:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -gravity))
    builder.rigid_gap = 0.001
    SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.ke = round_belt.CABLE_CONTACT_KE
    builder.default_shape_cfg.kd = round_belt.CABLE_CONTACT_KD
    builder.default_shape_cfg.mu = round_belt.CABLE_CONTACT_MU
    builder.default_shape_cfg.gap = 0.001
    return builder


def material(ei: float | None = None, gj: float | None = None) -> BeltMaterial:
    return BeltMaterial(ARGS.ei if ei is None else ei, ARGS.gj if gj is None else gj,
                        DAMPING, DAMPING, ARGS.armature)


def add_belt(builder, points, *, up, closed, mat=None, root_kinematic=False):
    bodies, joints = add_timing_belt_chain(
        builder, points, up=up, section=SECTION, material=mat or material(), closed=closed,
        cfg=builder.default_shape_cfg, label="belt", color=(0.1, 0.1, 0.1),
        root_kinematic=root_kinematic)
    n_seg = len(points) - 1
    _require(len(joints) == belt_joint_count(n_seg, closed),
             f"len(joints)={len(joints)} != belt_joint_count={belt_joint_count(n_seg, closed)}")
    _require(joints == list(range(joints[0], joints[0] + len(joints))), "joints not contiguous")
    return bodies, joints


def straight_points(num: int, seg: float) -> list[wp.vec3]:
    return [wp.vec3(i * seg, 0.0, 0.0) for i in range(num + 1)]


class BeltSim:
    """SolverMuJoCo stepping with Newton contacts (belt-vs-static pairs only), as task_common/simulation.py."""

    def __init__(self, builder, bodies, joints, *, contacts: bool = False, use_graph: bool = True):
        self.model = model = builder.finalize()
        self.bodies, self.joints = bodies, joints
        self.control = model.control()
        self.state_0, self.state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        self.root_start = self.state_0.body_q.numpy()[bodies[0], :3].astype(np.float64)
        self.solver = SolverMuJoCo(
            model, solver="newton", integrator="implicitfast", cone="elliptic",
            iterations=round_belt.MUJOCO_ITERATIONS, ls_iterations=round_belt.MUJOCO_LS_ITERATIONS,
            use_mujoco_contacts=False, njmax=MUJOCO_NJMAX, nconmax=MUJOCO_NCONMAX)
        self.sim_dt = FRAME_DT / SUBSTEPS
        self.pipeline = newton.CollisionPipeline(
            model, broad_phase="explicit", shape_pairs_filtered=self._belt_static_pairs(contacts))
        self.contacts = self.pipeline.contacts()
        self.wrench = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        self.half_len = self._segment_half_lengths()
        self.graph = None
        if use_graph and model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    def _belt_static_pairs(self, contacts: bool) -> wp.array:
        body = self.model.shape_body.numpy()
        belt = set(self.bodies)
        pairs = [(int(a), int(b)) for a, b in self.model.shape_contact_pairs.numpy()
                 if (body[a] in belt) != (body[b] in belt) and -1 in (body[a], body[b])] if contacts else []
        _require(bool(pairs) or not contacts, "no belt<->static contact pairs")
        return wp.array(np.asarray(pairs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i,
                        device=self.model.device)

    def _segment_half_lengths(self) -> np.ndarray:
        shape_body = self.model.shape_body.numpy()
        scale = self.model.shape_scale.numpy()
        half = np.zeros(self.model.body_count)
        for s, b in enumerate(shape_body):
            if b >= 0:
                half[b] = scale[s][2]
        return half[self.bodies]

    def set_wrench(self, body: int, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0)) -> None:
        w = np.zeros((self.model.body_count, 6), dtype=np.float32)
        w[body, :3], w[body, 3:] = force, torque
        self.wrench.assign(w)

    def _simulate(self) -> None:
        for _ in range(SUBSTEPS):
            self.state_0.clear_forces()
            wp.copy(self.state_0.body_f, self.wrench)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()

    def body_q(self) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.bodies].astype(np.float64)

    def assert_structure(self, closed: bool) -> None:
        """D6 joints are 2 angular DOFs without drives; exactly one CONNECT per closed loop."""
        jtype = self.model.joint_type.numpy()
        dims = self.model.joint_dof_dim.numpy()
        mode = self.model.joint_target_mode.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        _require(jtype[self.joints[0]] == newton.JointType.FREE, "root joint is not FREE")
        for j in self.joints[1:]:
            _require(jtype[j] == newton.JointType.D6, f"joint {j} is not D6")
            _require(dims[j][0] == 0 and dims[j][1] == 2, f"joint {j} dof dims {dims[j]} != (0, 2)")
            _require(np.all(mode[qd_start[j]:qd_start[j] + 2] == int(newton.JointTargetMode.NONE)),
                     f"joint {j} has a drive")
        n_eq = int(self.model.mujoco.equality_constraint_count)
        _require(n_eq == (1 if closed else 0), f"{n_eq} equality constraints (closed={closed})")

    def root_drift(self) -> float:
        return float(np.linalg.norm(self.body_q()[0, :3] - self.root_start))


def rot(q) -> np.ndarray:
    return np.array(wp.quat_to_matrix(wp.quat(*[float(v) for v in q])), dtype=np.float64).reshape(3, 3)


def local_point(q_row, p_local) -> np.ndarray:
    return np.asarray(q_row[:3], dtype=np.float64) + rot(q_row[3:7]) @ np.asarray(p_local, float)


def anchors(sim: BeltSim, bq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start, end) centreline nodes of every segment."""
    start = np.array([local_point(q, (0, 0, -h)) for q, h in zip(bq, sim.half_len)])
    end = np.array([local_point(q, (0, 0, h)) for q, h in zip(bq, sim.half_len)])
    return start, end


def connect_gap(sim: BeltSim, bq: np.ndarray) -> float:
    start, end = anchors(sim, bq)
    return float(np.linalg.norm(end[-1] - start[0]))


def tangents(bq: np.ndarray) -> np.ndarray:
    return np.array([rot(q[3:7])[:, 2] for q in bq])


def max_hinge_angle(bq: np.ndarray, closed: bool) -> float:
    t = tangents(bq)
    nxt = np.roll(t, -1, axis=0) if closed else t[1:]
    cur = t if closed else t[:-1]
    return float(np.max(np.arccos(np.clip(np.sum(cur * nxt, axis=1), -1.0, 1.0))))


def run_until_settled(sim: BeltSim, probe: Callable[[np.ndarray], np.ndarray]) -> tuple[float, float]:
    """Step until probe-point speed < SETTLE_SPEED (after MIN_SETTLE_TIME) or SETTLE_TIME."""
    prev = probe(sim.body_q())
    frames = round(SETTLE_TIME * FPS)
    speed = math.inf
    for f in range(1, frames + 1):
        sim.step()
        cur = probe(sim.body_q())
        speed = float(np.linalg.norm(cur - prev)) / FRAME_DT
        prev = cur
        if f * FRAME_DT >= MIN_SETTLE_TIME and speed < SETTLE_SPEED:
            break
    return f * FRAME_DT, speed


def tip_corner_probe(sim: BeltSim) -> Callable[[np.ndarray], np.ndarray]:
    h = sim.half_len[-1]
    return lambda bq: local_point(bq[-1], (0.5 * SECTION.width, HALF_THICK, h))


def run_frames(sim: BeltSim, seconds: float) -> None:
    for _ in range(round(seconds * FPS)):
        sim.step()


# ----------------------------------------------------------------------------
# Check registry: groups so a later package can append a "full-scene" group.
# ----------------------------------------------------------------------------

CHECKS: dict[str, list[tuple[str, Callable[[], str]]]] = {"isolated": [], "full-scene": []}


def check(title: str, group: str = "isolated") -> Callable[[Callable[[], str]], Callable[[], str]]:
    def register(fn: Callable[[], str]) -> Callable[[], str]:
        CHECKS[group].append((title, fn))
        return fn
    return register


def s0_preamble(sim: BeltSim) -> str:
    """Build-time checks of the ref/springref convention before any dynamics."""
    m, bodies = sim.model, sim.bodies
    fk_err = float(np.max(np.linalg.norm(
        sim.state_0.body_q.numpy()[bodies, :3] - m.body_q.numpy()[bodies, :3], axis=1)))
    _require(fk_err < 1e-5, f"eval_fk(joint_q) position error {fk_err:.3e} >= 1e-5 m")
    joint_q, q_start = m.joint_q.numpy(), m.joint_q_start.numpy()
    mj_q_start = sim.solver.mj_q_start.numpy()
    qpos0 = sim.solver.mjw_model.qpos0.numpy()[0]
    qpos_spring = sim.solver.mjw_model.qpos_spring.numpy()[0]
    qpos = sim.solver.mjw_data.qpos.numpy()[0]
    theta = np.array([joint_q[q_start[j]:q_start[j] + 2] for j in sim.joints[1:]])
    mj = np.array([mj_q_start[j] for j in sim.joints[1:]])
    twist = float(np.max(np.abs(theta[:, 1])))
    _require(twist < 1e-6, f"authored twist {twist:.3e} rad != 0")
    errs = {}
    for name, arr, scale in (("qpos0", qpos0, 1.0), ("qpos_spring", qpos_spring, 1.0), ("qpos", qpos, 2.0)):
        errs[name] = float(np.max(np.abs(np.stack((arr[mj], arr[mj + 1]), axis=1) - scale * theta)))
        _require(errs[name] < 1e-6, f"{name} - {scale:g}*theta max {errs[name]:.3e} rad >= 1e-6")
    eq = sim.solver.mjw_model.eq_data.numpy()[0, 0]
    bq0 = sim.state_0.body_q.numpy()[bodies].astype(np.float64)
    gap0 = float(np.linalg.norm(local_point(bq0[-1], eq[0:3]) - local_point(bq0[0], eq[3:6])))
    _require(gap0 < 1e-5, f"frame-0 CONNECT gap {gap0:.3e} m >= 1e-5")
    return (f"preamble fk err {fk_err:.1e} m, qpos0/qpos_spring/qpos-2θ err "
            f"{errs['qpos0']:.1e}/{errs['qpos_spring']:.1e}/{errs['qpos']:.1e} rad, "
            f"frame-0 CONNECT gap {gap0:.1e} m < 1e-5")


@check("S0 loop rest is straight")
def check_s0_loop_rest() -> str:
    n = ARGS.num_elements
    builder = make_builder(0.0)
    pts = ellipse_points((0.0, 0.0, 0.0), (0.08597, 0.12691), n)
    bodies, joints = add_belt(builder, pts, up=(0.0, 0.0, 1.0), closed=True)
    sim = BeltSim(builder, bodies, joints)
    sim.assert_structure(closed=True)
    pre = s0_preamble(sim)

    run_frames(sim, 3.0)
    bq = sim.body_q()
    _require(bool(np.all(np.isfinite(bq))), "non-finite body_q")
    p = bq[:, :3]
    ext = p.max(axis=0) - p.min(axis=0)
    aspect = float(ext[0] / ext[1])
    _require(0.95 <= aspect <= 1.05, f"XY AABB aspect {aspect:.4f} outside [0.95, 1.05]")
    radius = float(np.mean(np.linalg.norm(p[:, :2] - p[:, :2].mean(axis=0), axis=1)))
    r_msg = _in_band("mean radius", radius, PITCH_LENGTH / (2 * math.pi), 0.05)
    z_max = float(np.max(np.abs(p[:, 2])))
    _require(z_max < 1e-3, f"max |z| {z_max:.3e} >= 1 mm")
    start, _ = anchors(sim, bq)
    perim = float(np.sum(np.linalg.norm(np.roll(start, -1, axis=0) - start, axis=1)))
    p_msg = _in_band("perimeter", perim, PITCH_LENGTH, 0.003)
    gap = connect_gap(sim, bq)
    _require(gap <= 1e-3, f"CONNECT gap {gap * 1e3:.4f} mm > 1 mm")
    return (f"N={n}: {pre}; aspect {aspect:.4f} in [0.95, 1.05]; {r_msg}; "
            f"max|z| {z_max * 1e3:.3f} mm < 1; {p_msg}; CONNECT gap {gap * 1e3:.4f} mm <= 1")


@check("S1 inextensible")
def check_s1_inextensible() -> str:
    n, seg = 20, 0.005
    builder = make_builder(0.0)
    bodies, joints = add_belt(builder, straight_points(n, seg), up=(0.0, 0.0, 1.0), closed=False,
                              root_kinematic=True)
    sim = BeltSim(builder, bodies, joints)
    sim.assert_structure(closed=False)
    sim.set_wrench(bodies[-1], force=(10.0, 0.0, 0.0))
    t, speed = run_until_settled(sim, tip_corner_probe(sim))
    bq = sim.body_q()
    _require(bool(np.all(np.isfinite(bq))), "non-finite body_q")
    start, end = anchors(sim, bq)
    elong = float(end[-1, 0] - n * seg)
    gap = float(np.max(np.linalg.norm(end[:-1] - start[1:], axis=1)))
    drift = sim.root_drift()
    _require(drift < 1e-6, f"kinematic root moved {drift:.3e} m")
    _require(elong <= 2e-4, f"elongation {elong * 1e3:.4f} mm > 0.2 mm")
    _require(gap <= 5e-5, f"max anchor gap {gap * 1e3:.4f} mm > 0.05 mm")
    return (f"elongation {elong * 1e3:.4f} mm <= 0.2; max anchor gap {gap * 1e3:.4f} mm <= 0.05; "
            f"root drift {drift:.1e} m (t={t:.2f} s, tip speed {speed * 1e3:.3f} mm/s)")


def cantilever_droop(up, ei: float) -> tuple[float, float, float]:
    """(droop of the centreline tip node, droop of the tip COM, settle time) for 10 x 5 mm."""
    n, seg = 10, 0.005
    builder = make_builder(GRAVITY)
    bodies, joints = add_belt(builder, straight_points(n, seg), up=up, closed=False,
                              mat=material(ei=ei), root_kinematic=True)
    sim = BeltSim(builder, bodies, joints)
    sim.assert_structure(closed=False)
    t, _ = run_until_settled(sim, tip_corner_probe(sim))
    bq = sim.body_q()
    _require(bool(np.all(np.isfinite(bq))), "non-finite body_q")
    _require(sim.root_drift() < 1e-6, f"kinematic root moved {sim.root_drift():.3e} m")
    root_start, _ = anchors(sim, bq[:1])
    t0 = tangents(bq[:1])[0]

    def droop(point: np.ndarray) -> float:
        v = point - root_start[0]
        return float(-(v - np.dot(v, t0) * t0)[2])

    _, end = anchors(sim, bq)
    return droop(end[-1]), droop(bq[-1, :3]), t


@check("S2 flat droop, EI tunable")
def check_s2_flat_droop() -> str:
    length = 0.05
    ei1, ei2 = ARGS.ei, 2.0 * ARGS.ei
    d1, c1, t1 = cantilever_droop((0.0, 1.0, 0.0), ei1)
    d2, c2, t2 = cantilever_droop((0.0, 1.0, 0.0), ei2)
    RESULTS["droop_flat"], RESULTS["droop_flat2"] = d1, d2
    m1 = _in_band(f"droop(EI={ei1:g}) [mm]", d1 * 1e3, W_PER_LENGTH * length**4 / (8 * ei1) * 1e3, 0.25)
    m2 = _in_band(f"droop(EI={ei2:g}) [mm]", d2 * 1e3, W_PER_LENGTH * length**4 / (8 * ei2) * 1e3, 0.25)
    ratio = d1 / d2
    _require(1.7 <= ratio <= 2.3, f"droop ratio {ratio:.3f} outside [1.7, 2.3]")
    return (f"{m1}; {m2}; ratio {ratio:.3f} in [1.7, 2.3] (tip COM droop {c1 * 1e3:.3f} / "
            f"{c2 * 1e3:.3f} mm; settled at {t1:.2f} / {t2:.2f} s)")


@check("S3 lateral stiffness")
def check_s3_lateral() -> str:
    _require("droop_flat" in RESULTS, "S3 needs S2's flat droop")
    d, _, t = cantilever_droop((0.0, 0.0, 1.0), ARGS.ei)
    bar = min(1e-4, RESULTS["droop_flat"] / 20.0)
    _require(d <= bar, f"lateral droop {d * 1e3:.5f} mm > {bar * 1e3:.5f} mm")
    return f"lateral droop {d * 1e3:.5f} mm <= {bar * 1e3:.5f} mm (settled at {t:.2f} s)"


def twist_angle(gj: float) -> tuple[float, float]:
    n, seg, torque = 20, 0.005, 1e-4
    builder = make_builder(0.0)
    bodies, joints = add_belt(builder, straight_points(n, seg), up=(0.0, 0.0, 1.0), closed=False,
                              mat=material(gj=gj), root_kinematic=True)
    sim = BeltSim(builder, bodies, joints)
    sim.assert_structure(closed=False)
    sim.set_wrench(bodies[-1], torque=(torque, 0.0, 0.0))
    t, _ = run_until_settled(sim, tip_corner_probe(sim))
    bq = sim.body_q()
    _require(bool(np.all(np.isfinite(bq))), "non-finite body_q")
    _require(sim.root_drift() < 1e-6, f"kinematic root moved {sim.root_drift():.3e} m")
    r_rel = rot(bq[0, 3:7]).T @ rot(bq[-1, 3:7])
    return math.atan2(r_rel[1, 0], r_rel[0, 0]), t


@check("S4 twist compliance")
def check_s4_twist() -> str:
    length, torque = 0.1, 1e-4
    gj1, gj2 = ARGS.gj, 2.0 * ARGS.gj
    a1, t1 = twist_angle(gj1)
    a2, t2 = twist_angle(gj2)
    RESULTS["twist"], RESULTS["twist2"] = a1, a2
    m1 = _in_band(f"twist(GJ={gj1:g}) [rad]", a1, torque * length / gj1, 0.25)
    m2 = _in_band(f"twist(GJ={gj2:g}) [rad]", a2, torque * length / gj2, 0.25)
    return f"{m1}; {m2} (settled at {t1:.2f} / {t2:.2f} s)"


@check("S4b armature insensitivity")
def check_s4b_armature() -> str:
    _require(all(k in RESULTS for k in ("droop_flat", "twist")), "S4b needs S2 and S4")
    base = ARGS.armature
    ARGS.armature = ARMATURE_ALT
    try:
        alt = {"droop_flat": cantilever_droop((0.0, 1.0, 0.0), ARGS.ei)[0],
               "droop_flat2": cantilever_droop((0.0, 1.0, 0.0), 2.0 * ARGS.ei)[0],
               "twist": twist_angle(ARGS.gj)[0], "twist2": twist_angle(2.0 * ARGS.gj)[0]}
    finally:
        ARGS.armature = base
    rel = {k: abs(v / RESULTS[k] - 1.0) for k, v in alt.items()}
    msg = ", ".join(f"{k} {RESULTS[k]:.5g}->{alt[k]:.5g} ({rel[k] * 100:.2f} %)" for k in alt)
    _require(max(rel.values()) <= 0.02, f"armature {base:g}->{ARMATURE_ALT:g}: {msg} (bar 2 %)")
    return f"armature {base:g}->{ARMATURE_ALT:g}: {msg}; all <= 2 %"


def build_pulley_scene(n: int, use_graph: bool = True) -> BeltSim:
    builder = make_builder(GRAVITY)
    cfg = builder.default_shape_cfg
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.192, 0.192, -0.005), wp.quat_identity()),
                          hx=0.192, hy=0.192, hz=0.005, cfg=cfg, label="board")
    for name, ((x, y), r, z, r_plate, z_bot, z_top, _) in PULLEYS.items():
        for label, radius, zc, half_h in ((f"{name}_body", r, z, 0.0135),
                                          (f"{name}_bottom", r_plate, z_bot, 0.00125),
                                          (f"{name}_top", r_plate, z_top, 0.00125)):
            builder.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(x, y, zc), wp.quat_identity()),
                                       radius=radius, half_height=half_h, cfg=cfg, label=label)

    (cs, rs), (cl, rl) = [(np.asarray(PULLEYS[k][0]), PULLEYS[k][1] + HALF_THICK)
                          for k in ("small", "large")]
    away = (cl - cs) / np.linalg.norm(cl - cs)
    lo, hi = 0.0, 0.05
    for _ in range(100):  # bisection on the virtual large-pulley offset
        mid = 0.5 * (lo + hi)
        if two_pulley_path_length(cs, rs, cl + mid * away, rl) < PITCH_LENGTH:
            lo = mid
        else:
            hi = mid
    RESULTS["virtual_offset"] = lo
    pts = two_pulley_path(cs, rs, cl + lo * away, rl, n, BELT_Z)
    bodies, joints = add_belt(builder, pts, up=(0.0, 0.0, 1.0), closed=True)
    return BeltSim(builder, bodies, joints, contacts=True, use_graph=use_graph)


def check_pulley_wrap(n: int) -> str:
    """Evaluates every S5 bar before failing so the report carries all measured values."""
    sim = build_pulley_scene(n)
    sim.assert_structure(closed=True)
    run_frames(sim, 3.0 - FRAME_DT)
    prev = sim.body_q()
    sim.step()
    bq = sim.body_q()
    _require(bool(np.all(np.isfinite(bq))), "(i) non-finite body_q")
    p = bq[:, :3]
    msgs = [f"N={n}, virtual offset {RESULTS['virtual_offset'] * 1e3:.2f} mm"]
    failed: list[str] = []

    def bar(ok: bool, text: str) -> None:
        msgs.append(text if ok else f"FAILED {text}")
        if not ok:
            failed.append(text)

    for name, ((x, y), r, *_rest, z_band) in PULLEYS.items():
        d = np.hypot(p[:, 0] - x, p[:, 1] - y)
        near = d < r + 0.010
        if not np.any(near):
            bar(False, f"(ii/iv/v) no belt body within r + 10 mm of the {name} pulley")
            continue
        inner = float(np.min(d[near] - HALF_THICK - r))
        bar(inner >= -5e-4, f"(ii) {name} min inner gap {inner * 1e3:.3f} mm >= -0.5")
        zs = p[near, 2]
        bar(bool(np.all((zs >= z_band[0]) & (zs <= z_band[1]))),
            f"(iv) {name} z [{zs.min():.4f}, {zs.max():.4f}] in {z_band}")
        closest = float(np.min(d - r - HALF_THICK))
        bar(closest <= 0.002, f"(v) {name} closest centreline {closest * 1e3:.3f} mm <= 2")
    fold_bar = 2.2 * (PITCH_LENGTH / n) / 0.02505
    fold = max_hinge_angle(bq, closed=True)
    bar(fold <= fold_bar, f"(iii) max hinge {fold:.4f} rad <= {fold_bar:.4f}")
    speed = float(np.mean(np.linalg.norm(p - prev[:, :3], axis=1))) / FRAME_DT
    bar(speed < 5e-3, f"(vi) mean speed {speed * 1e3:.3f} mm/s < 5")
    start, _ = anchors(sim, bq)
    perim = float(np.sum(np.linalg.norm(np.roll(start, -1, axis=0) - start, axis=1)))
    bar(abs(perim / PITCH_LENGTH - 1.0) <= 0.003,
        f"(vii) perimeter {perim:.6f} in [{PITCH_LENGTH * 0.997:.5f}, {PITCH_LENGTH * 1.003:.5f}]")
    gap = connect_gap(sim, bq)
    bar(gap <= 1e-3, f"(viii) CONNECT gap {gap * 1e3:.4f} mm <= 1")
    _require(not failed, "; ".join(msgs))
    return "; ".join(msgs)


@check("S5 pulley wrap")
def check_s5_pulley_wrap() -> str:
    return check_pulley_wrap(ARGS.num_elements)


def measure_perf(n: int) -> float:
    sim = build_pulley_scene(n, use_graph=True)
    _require(sim.graph is not None, "CUDA graph capture unavailable (perf needs a CUDA device)")
    for _ in range(200):
        sim.step()
    wp.synchronize_device(sim.model.device)
    t200 = time.perf_counter()
    for _ in range(200):
        sim.step()
    wp.synchronize_device(sim.model.device)
    return (time.perf_counter() - t200) / 200 * 1e3


def run_perf() -> None:
    """[PERF] ms/frame of the S5 scene (N = 135 and 68); N = 68 also reruns S5 first."""
    if ARGS.num_elements != 68:
        print(f"[PASS] S5 pulley wrap (N=68 rerun): {check_pulley_wrap(68)}")
    ms = {n: measure_perf(n) for n in (135, 68)}
    for n in (135, 68):
        bar = " (bar <= 15)" if n == 135 else ""
        print(f"[PERF] S5 scene N={n}: {ms[n]:.2f} ms/frame{bar}")
    _require(ms[135] <= 15.0, f"N=135 {ms[135]:.2f} ms/frame > 15")


# ----------------------------------------------------------------------------
# VBD cloth strip (--model strip)
# ----------------------------------------------------------------------------

STRIP_SECTION = StripSection(width=0.025, thickness=0.0022, area_density=0.0565 / 0.025)
STRIP_N_W = 5
STRIP_ITERATIONS, STRIP_RETRY_ITERATIONS = 20, 40
STRIP_CONTACT_GAP = 0.001
RIB_SPRING_KE, RIB_SPRING_KD = 2.0e4, 20.0
PACKAGE_DAMPING = (0.1, 1.0e-2)  # (tri_kd, edge_kd) kd/ke ratios
KNOB_DAMPING = (0.01, 1.0e-3)  # package damping x0.1
CORD_KD_RATIO = 1.0e-3  # rib-spring kd/ke
SWEEP_TRI_KE = (1.0e2, 3.0e2, 1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5)
SWEEP_CORD_KE = (1.0e3, 1.0e4, 1.0e5)
PROBE_RIB_SPRING_KE = 1.0e3
CAL_START, CAL_MAX_ITER, CAL_TOL = (4.8, 0.48), 6, 0.10
S3_BAR, S3_STRICT, S5_PEN_BAR, S5_TILT_BAR, PERF_BAR = 1.0e-3, 1.0e-4, 5.0e-4, 10.0, 15.0
SWEEP_RESULTS = REPO_ROOT / "handoffs" / "research" / "soft-strip-sweep.md"
NAN = math.nan


@dataclass(frozen=True)
class StripProfile:
    section: StripSection = STRIP_SECTION
    n_w: int | None = None  # None keeps --width-cells
    iterations: int = STRIP_ITERATIONS
    substeps: int = SUBSTEPS
    self_contact: tuple[tuple[str, float], ...] = (("particle_self_contact_margin", 1.1e-3),
                                                   ("particle_self_contact_gap", 1.1e-3))


SOFT_PROFILE = StripProfile()
# timing_belt.py's own belt and solver settings.
CONTROL_PROFILE = StripProfile(section=StripSection(width=0.05, thickness=0.002, area_density=10.0), n_w=10,
                               substeps=20, self_contact=(("particle_self_contact_radius", 1.0e-3),
                                                          ("particle_self_contact_margin", 1.5e-3)))
CONTROL_MATERIAL = StripMaterial(tri_ke=1.0e3, tri_kd=0.1, edge_ke_flat=10.0, edge_kd=1.0e-3, rib_ratio=2.0e3,
                                 rib_spring_ke=2.0e4, rib_spring_kd=20.0, zero_rest_angles=False,
                                 geometric_hinges=False)
STRIP_ARGS = argparse.Namespace(n_w=STRIP_N_W, profile=SOFT_PROFILE)


@contextmanager
def strip_profile(profile: StripProfile):
    saved = STRIP_ARGS.profile, STRIP_ARGS.n_w
    STRIP_ARGS.profile, STRIP_ARGS.n_w = profile, profile.n_w or STRIP_ARGS.n_w
    try:
        yield
    finally:
        STRIP_ARGS.profile, STRIP_ARGS.n_w = saved


def sec() -> StripSection:
    return STRIP_ARGS.profile.section


def half_thick() -> float:
    return 0.5 * sec().thickness


def soft_material(tri_ke: float, cord_ke: float = 0.0, edge_ke_flat: float | None = None,
                  damping: tuple[float, float] = KNOB_DAMPING) -> StripMaterial:
    return StripMaterial(tri_ke=tri_ke, edge_ke_flat=edge_ke_flat, tri_kd=damping[0], edge_kd=damping[1],
                         rib_spring_ke=RIB_SPRING_KE, rib_spring_kd=RIB_SPRING_KD, cord_ke=cord_ke,
                         cord_kd=CORD_KD_RATIO * cord_ke)


def operating_material(tri_ke: float | None = None, cord_ke: float | None = None,
                       edge_ke_flat: float | None = None) -> StripMaterial:
    """SOFT_STRIP_OPERATING_POINT with overrides; edge_ke_flat None means uncalibrated for that tri_ke."""
    op = SOFT_STRIP_OPERATING_POINT
    tri_ke = op.tri_ke if tri_ke is None else tri_ke
    cord_ke = op.cord_ke if cord_ke is None else cord_ke
    if edge_ke_flat is None:
        edge_ke_flat = op.edge_ke_flat if tri_ke == op.tri_ke else SOFT_STRIP_CALIBRATED_EDGE_KE.get(tri_ke)
    return replace(op, tri_ke=tri_ke, cord_ke=cord_ke, cord_kd=CORD_KD_RATIO * cord_ke, edge_ke_flat=edge_ke_flat)


def droop_target(ei: float | None = None) -> float:
    return sec().area_density * sec().width * GRAVITY * 0.05**4 / (8 * (ARGS.ei if ei is None else ei))


def edge_ke_nominal() -> float:
    """Across-width hinge ke of the discrete-shell formula, 3 (EI / w) / a."""
    return 3.0 * ARGS.ei / sec().width / strip_cell()


def predicted_ei_lat(mat: StripMaterial) -> float:
    w = sec().width
    return 2.0 * mat.tri_ke * (1.0 + POISSON) * w**3 / 12.0 + 2.0 * mat.cord_ke * strip_cell() * w**2 / 4.0


def make_strip_builder(gravity: float) -> newton.ModelBuilder:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -gravity))
    builder.default_shape_cfg.ke = round_belt.CABLE_CONTACT_KE
    builder.default_shape_cfg.kd = round_belt.CABLE_CONTACT_KD
    builder.default_shape_cfg.mu = round_belt.CABLE_CONTACT_MU
    builder.default_shape_cfg.has_particle_collision = True
    return builder


def add_strip(builder, points, *, up, closed, mat: StripMaterial, n_w: int | None = None) -> dict:
    return add_timing_belt_strip(builder, points, up=up, section=sec(), material=mat, n_w=n_w or STRIP_ARGS.n_w,
                                 closed=closed, color=(0.1, 0.1, 0.1), label="belt")


class StripSim:
    """SolverVBD stepping of the strip; particle-shape contacts only when the scene has shapes."""

    def __init__(self, builder, info: dict, *, pinned_rows=(), contacts: bool = False, use_graph: bool = True,
                 iterations: int | None = None):
        prof = STRIP_ARGS.profile
        self.info, self.rows = info, info["rows"]
        p0 = info["particles"].start
        for r in pinned_rows:
            for k in range(self.rows[1]):
                builder.particle_mass[p0 + r * self.rows[1] + k] = 0.0
        builder.color()
        self.model = model = builder.finalize()
        model.soft_contact_ke = round_belt.CABLE_CONTACT_KE
        model.soft_contact_kd = round_belt.CABLE_CONTACT_KD
        model.soft_contact_mu = round_belt.CABLE_CONTACT_MU
        self.control = model.control()
        self.state_0, self.state_1 = model.state(), model.state()
        self.solver = SolverVBD(model, iterations=iterations or prof.iterations, particle_enable_self_contact=True,
                                **dict(prof.self_contact))
        self.substeps = prof.substeps
        self.sim_dt = FRAME_DT / self.substeps
        self.pipeline = (newton.CollisionPipeline(model, broad_phase="explicit", soft_contact_gap=STRIP_CONTACT_GAP)
                         if contacts else None)
        self.contacts = self.pipeline.contacts() if self.pipeline else None
        self.force = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.use_graph = use_graph and model.device.is_cuda
        self.graph = None
        self.q0 = self.particle_q()

    def set_row_forces(self, row: int, forces) -> None:
        f = np.zeros((self.model.particle_count, 3), dtype=np.float32)
        s = self.info["particles"].start + row * self.rows[1]
        f[s:s + self.rows[1]] = forces
        self.force.assign(f)

    def _simulate(self) -> None:
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            wp.copy(self.state_0.particle_f, self.force)
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
            return
        self._simulate()
        if self.use_graph:  # VBD sizes its contact buffers on the first uncaptured step
            wp.synchronize_device(self.model.device)
            with wp.ScopedCapture() as capture:
                self._simulate()
            self.graph = capture.graph

    def particle_q(self) -> np.ndarray:
        return self.state_0.particle_q.numpy()[self.info["particles"]].astype(np.float64)


def strip_run_until_settled(sim: StripSim) -> dict:
    """Step until mean particle speed < SETTLE_SPEED (after MIN_SETTLE_TIME) or SETTLE_TIME.

    Unsettled: v = mean speed over the last 0.5 s, v3 = over 2.5-3 s.
    """
    prev = sim.particle_q()
    speeds: list[float] = []
    for f in range(1, round(SETTLE_TIME * FPS) + 1):
        sim.step()
        cur = sim.particle_q()
        if not np.all(np.isfinite(cur)):
            return {"t": f * FRAME_DT, "ok": False, "v": NAN, "v3": NAN}
        speeds.append(float(np.mean(np.linalg.norm(cur - prev, axis=1))) / FRAME_DT)
        prev = cur
        if f * FRAME_DT >= MIN_SETTLE_TIME and speeds[-1] < SETTLE_SPEED:
            return {"t": f * FRAME_DT, "ok": True, "v": speeds[-1], "v3": NAN}
    win, mid = round(0.5 * FPS), round(3.0 * FPS)
    return {"t": SETTLE_TIME, "ok": False, "v": float(np.mean(speeds[-win:])), "v3": float(np.mean(speeds[mid - win:mid]))}


def _settle_keys(prefix: str, s: dict) -> dict:
    return {f"{prefix}_settled": s["ok"], f"{prefix}_t": s["t"], f"{prefix}_v": s["v"], f"{prefix}_v3": s["v3"]}


def strip_cell() -> float:
    return PITCH_LENGTH / ARGS.num_elements


def clamped_strip(length: float, up, gravity: float, mat: StripMaterial) -> StripSim:
    """Open strip along +X; rows 0 and 1 pinned (one pinned row is a hinge), free length beyond row 1 (x = 0)."""
    k = max(1, round(length / strip_cell()))
    a = length / k
    builder = make_strip_builder(gravity)
    pts = [wp.vec3((i - 1) * a, 0.0, 0.0) for i in range(k + 2)]
    info = add_strip(builder, pts, up=up, closed=False, mat=mat)
    return StripSim(builder, info, pinned_rows=(0, 1))


def row_angles_from_vertical(q: np.ndarray, rows) -> np.ndarray:
    v = strip_row_vectors(q, rows)
    return np.degrees(np.arccos(np.clip(np.abs(v[:, 2]) / np.linalg.norm(v, axis=1), -1.0, 1.0)))


def closed_perimeter(c: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)))


def _finite(q: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(q)))


def strip_s0(mat: StripMaterial, iterations: int | None = None) -> dict:
    """Zero-g loop released from an ellipse."""
    builder = make_strip_builder(0.0)
    info = add_strip(builder, ellipse_points((0.0, 0.0, 0.0), (0.08597, 0.12691), ARGS.num_elements),
                     up=(0.0, 0.0, 1.0), closed=True, mat=mat)
    sim = StripSim(builder, info, iterations=iterations)
    rest = float(np.max(np.abs(sim.model.edge_rest_angle.numpy()[info["edges"]])))
    mass = float(np.sum(sim.model.particle_mass.numpy()[info["particles"]]))
    out = {"s0_rest_angle": rest, "s0_mass": mass, "s0_hinges": info["hinges"], "s0_springs": info["springs"],
           **_settle_keys("s0", strip_run_until_settled(sim))}
    q = sim.particle_q()
    if _finite(q):
        c = strip_centreline(q, sim.rows)
        ext = c.max(axis=0) - c.min(axis=0)
        out["s0_aspect"] = float(ext[0] / ext[1])
        out["s0_radius"] = float(np.mean(np.linalg.norm(c[:, :2] - c[:, :2].mean(axis=0), axis=1)))
    return out


def strip_s1(mat: StripMaterial) -> dict:
    """10 N on the tip row of a 100 mm strip, zero g."""
    sim = clamped_strip(0.1, (0.0, 0.0, 1.0), 0.0, mat)
    sim.set_row_forces(sim.rows[0] - 1, (10.0 / sim.rows[1], 0.0, 0.0))
    s = strip_run_until_settled(sim)
    q = sim.particle_q()
    finite = _finite(q)
    elong = float(strip_centreline(q, sim.rows)[-1, 0] - strip_centreline(sim.q0, sim.rows)[-1, 0]) if finite else NAN
    return {"s1_finite": finite, "s1_settled": s["ok"], "s1_elong": elong}


def strip_hang(mat: StripMaterial) -> dict:
    """H: closed band in XZ, width along Y, top row pinned, 3 s under gravity."""
    n = ARGS.num_elements
    r = PITCH_LENGTH / (2 * n * math.sin(math.pi / n))
    pts = [wp.vec3(r * math.sin(2 * math.pi * i / n), 0.0, r * math.cos(2 * math.pi * i / n)) for i in range(n)]
    builder = make_strip_builder(GRAVITY)
    info = add_strip(builder, pts, up=(0.0, 1.0, 0.0), closed=True, mat=mat)
    sim = StripSim(builder, info, pinned_rows=(0,))
    run_frames(sim, 3.0)
    q = sim.particle_q()
    if not _finite(q):
        return {"h_finite": False, "h_perim": NAN, "h_cell": NAN}
    shape = (sim.rows[0], sim.rows[1], 3)
    cur, rest = q.reshape(shape), sim.q0.reshape(shape)
    seg = np.linalg.norm(np.roll(cur, -1, axis=0) - cur, axis=2)
    seg0 = np.linalg.norm(np.roll(rest, -1, axis=0) - rest, axis=2)
    perim = closed_perimeter(strip_centreline(q, sim.rows)) / closed_perimeter(strip_centreline(sim.q0, sim.rows))
    return {"h_finite": True, "h_perim": 100.0 * (perim - 1.0), "h_cell": 100.0 * float(np.max(seg / seg0 - 1.0))}


def strip_flat_droop(mat: StripMaterial) -> dict:
    sim = clamped_strip(0.05, (0.0, 1.0, 0.0), GRAVITY, mat)
    s = strip_run_until_settled(sim)
    q = sim.particle_q()
    c = strip_centreline(q, sim.rows)
    s["d"] = float(-(c[-1, 2] - c[1, 2])) if _finite(q) else NAN
    return s


def calibrate_flat(mat: StripMaterial) -> dict:
    """Secant on log(droop) vs log(edge_ke_flat) until droop(EI) is within CAL_TOL of the analytic value."""
    target = droop_target()
    hist: list[tuple[float, float, float, bool]] = []

    def run(ke: float) -> float:
        s = strip_flat_droop(replace(mat, edge_ke_flat=ke))
        hist.append((ke, s["d"], s["t"], s["ok"]))
        return math.log(s["d"] / target) if math.isfinite(s["d"]) and s["d"] > 0 else NAN

    def hit() -> bool:
        return any(ok and abs(d / target - 1) <= CAL_TOL for _, d, _, ok in hist)

    pts = [(math.log(ke), run(ke)) for ke in CAL_START]
    for _ in range(CAL_MAX_ITER):
        if hit():
            break
        good = [p for p in pts if math.isfinite(p[1])]
        bad = [p[0] for p in pts if not math.isfinite(p[1])]
        if len(good) >= 2:
            (x0, e0), (x1, e1) = good[-2], good[-1]
            x2 = x1 + 1.0 if e1 == e0 else x1 - e1 * (x1 - x0) / (e1 - e0)
            x2 = min(max(x2, x1 - math.log(100.0)), x1 + math.log(100.0))
        else:
            x2 = (good[-1][0] if good else pts[-1][0]) + math.log(3.0)
        if bad and x2 >= min(bad):  # NaN on the stiff side: bisect towards the stiffest finite point below it
            below = [p[0] for p in good if p[0] < min(bad)]
            x2 = 0.5 * (max(below) + min(bad)) if below else min(bad) - math.log(3.0)
        pts.append((x2, run(math.exp(x2))))
    hits = [h for h in hist if h[3] and abs(h[1] / target - 1) <= CAL_TOL]
    pool = hits or [h for h in hist if math.isfinite(h[1])] or hist
    ke = min(pool, key=lambda h: abs(h[1] / target - 1) if math.isfinite(h[1]) else math.inf)[0]
    return {"edge_ke_flat": ke, "c_bend": ke / edge_ke_nominal(), "calibrated": hit(), "cal_evals": len(hist),
            "cal_hist": hist}


def strip_s2(mat: StripMaterial) -> dict:
    s1 = strip_flat_droop(mat)
    s2 = strip_flat_droop(replace(mat, edge_ke_flat=2.0 * mat.edge_ke_flat))
    ratio = s1["d"] / s2["d"] if math.isfinite(s2["d"]) and s2["d"] else NAN
    return {"s2_droop": s1["d"], "s2_ratio": ratio, **_settle_keys("s2", s1)}


def strip_s3(mat: StripMaterial) -> dict:
    sim = clamped_strip(0.05, (0.0, 0.0, 1.0), GRAVITY, mat)
    s = strip_run_until_settled(sim)
    q = sim.particle_q()
    c = strip_centreline(q, sim.rows)
    return {"s3_droop": float(-(c[-1, 2] - c[1, 2])) if _finite(q) else NAN, "s3_settled": s["ok"],
            "s3_ei_lat": predicted_ei_lat(mat)}


def strip_s4(mat: StripMaterial) -> dict:
    length, torque = 0.1, 1e-4
    sim = clamped_strip(length, (0.0, 0.0, 1.0), 0.0, mat)
    z = (np.arange(sim.rows[1]) / (sim.rows[1] - 1) - 0.5) * sec().width
    f = np.zeros((sim.rows[1], 3))
    f[:, 1] = -torque * z / np.sum(z * z)  # linear, zero net force, moment exactly `torque` about +X
    sim.set_row_forces(sim.rows[0] - 1, f)
    s = strip_run_until_settled(sim)
    q = sim.particle_q()
    if not _finite(q):
        return {"s4_twist": NAN, "s4_gj_ei": NAN, "s4_settled": False}
    v = strip_row_vectors(q, sim.rows)
    r0, r1 = v[0, 1:], v[-1, 1:]
    phi = math.atan2(r0[0] * r1[1] - r0[1] * r1[0], float(np.dot(r0, r1)))
    return {"s4_twist": phi, "s4_gj_ei": torque * length / (phi * ARGS.ei) if phi else math.inf, "s4_settled": s["ok"]}


def build_strip_pulley_scene(n_l: int, n_w: int, use_graph: bool = True, mat: StripMaterial | None = None) -> StripSim:
    builder = make_strip_builder(GRAVITY)
    cfg = builder.default_shape_cfg
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.192, 0.192, -0.005), wp.quat_identity()),
                          hx=0.192, hy=0.192, hz=0.005, cfg=cfg, label="board")
    for name, ((x, y), r, z, r_plate, z_bot, z_top, _) in PULLEYS.items():
        for label, radius, zc, half_h in ((f"{name}_body", r, z, 0.0135),
                                          (f"{name}_bottom", r_plate, z_bot, 0.00125),
                                          (f"{name}_top", r_plate, z_top, 0.00125)):
            builder.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(x, y, zc), wp.quat_identity()),
                                       radius=radius, half_height=half_h, cfg=cfg, label=label)
    (cs, rs), (cl, rl) = [(np.asarray(PULLEYS[k][0]), PULLEYS[k][1] + half_thick()) for k in ("small", "large")]
    away = (cl - cs) / np.linalg.norm(cl - cs)
    lo, hi = 0.0, 0.05
    for _ in range(100):  # bisection on the virtual large-pulley offset
        mid = 0.5 * (lo + hi)
        if two_pulley_path_length(cs, rs, cl + mid * away, rl) < PITCH_LENGTH:
            lo = mid
        else:
            hi = mid
    RESULTS["virtual_offset"] = lo
    pts = two_pulley_path(cs, rs, cl + lo * away, rl, n_l, BELT_Z)
    info = add_strip(builder, pts, up=(0.0, 0.0, 1.0), closed=True, mat=mat or operating_material(), n_w=n_w)
    return StripSim(builder, info, contacts=True, use_graph=use_graph)


def strip_s5(mat: StripMaterial) -> dict:
    sim = build_strip_pulley_scene(ARGS.num_elements, STRIP_ARGS.n_w, mat=mat)
    out = {"s5_pen": NAN, "s5_in_gap": False, "s5_tilt": NAN, "s5_perim": NAN,
           **_settle_keys("s5", strip_run_until_settled(sim))}
    q = sim.particle_q()
    if not _finite(q):
        return out
    c = strip_centreline(q, sim.rows)
    q_rows = q.reshape(sim.rows[0], sim.rows[1], 3)
    pen, in_gap = -math.inf, True
    for (x, y), r, *_rest, z_band in PULLEYS.values():
        d = np.hypot(q[:, 0] - x, q[:, 1] - y)
        near = d < r + 0.010
        near_rows = np.hypot(c[:, 0] - x, c[:, 1] - y) < r + 0.010
        if not np.any(near) or not np.any(near_rows):
            in_gap = False
            continue
        pen = max(pen, float(np.max(r + half_thick() - d[near])))
        zs = q_rows[near_rows, :, 2]
        in_gap &= bool(np.all((zs >= z_band[0]) & (zs <= z_band[1])))
    perim = closed_perimeter(c) / closed_perimeter(strip_centreline(sim.q0, sim.rows))
    out.update(s5_pen=pen, s5_in_gap=in_gap, s5_tilt=float(np.max(row_angles_from_vertical(q, sim.rows))),
               s5_perim=100.0 * (perim - 1.0))
    return out


def other_gpu_compute() -> list[str]:
    """Compute processes on the GPU other than this one (empty if nvidia-smi is unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip() and ln.split(",")[0].strip() != str(os.getpid())]


def measure_strip_perf(mat: StripMaterial) -> dict:
    busy = other_gpu_compute()
    if busy:
        return {"perf": NAN, "perf_note": f"GPU busy: {'; '.join(busy)}"}
    sim = build_strip_pulley_scene(ARGS.num_elements, STRIP_ARGS.n_w, use_graph=True, mat=mat)
    _require(sim.use_graph, "CUDA graph capture unavailable (perf needs a CUDA device)")
    for _ in range(200):
        sim.step()
    wp.synchronize_device(sim.model.device)
    t200 = time.perf_counter()
    for _ in range(200):
        sim.step()
    wp.synchronize_device(sim.model.device)
    return {"perf": (time.perf_counter() - t200) / 200 * 1e3}


def measure_point(mat: StripMaterial, *, label: str, calibrate: bool, perf: bool, profile: StripProfile = SOFT_PROFILE,
                  probe: bool = False, skip: tuple[str, ...] = ()) -> dict:
    """One sweep row; a stage that raises is recorded in row['errors'] and leaves its values missing."""
    row: dict = {"label": label, "tri_ke": mat.tri_ke, "cord_ke": mat.cord_ke, "damping": (mat.tri_kd, mat.edge_kd),
                 "probe": probe, "errors": [], "skipped": skip}

    def stage(name: str, fn: Callable[[], dict]) -> None:
        if name in skip:
            return
        try:
            row.update(fn())
        except Exception as exc:  # noqa: BLE001 - per-point failures are rows, not crashes
            row["errors"].append(f"{name}: {type(exc).__name__}: {exc}")

    t0 = time.perf_counter()
    with strip_profile(profile):
        if calibrate:
            stage("cal", lambda: calibrate_flat(mat))
        else:
            row.update(edge_ke_flat=mat.edge_ke_flat, c_bend=mat.edge_ke_flat / edge_ke_nominal(), calibrated=None)
        mat = replace(mat, edge_ke_flat=row.get("edge_ke_flat", CAL_START[0]))
        for name, fn in (("S0", strip_s0), ("S1", strip_s1), ("H", strip_hang), ("S2", strip_s2),
                         ("S3", strip_s3), ("S4", strip_s4), ("S5", strip_s5)):
            stage(name, lambda fn=fn: fn(mat))
        if perf:
            stage("perf", lambda: measure_strip_perf(mat))
        row["target"] = droop_target()
    row["wall"] = time.perf_counter() - t0
    row["hard"] = hard_bars(row)
    return row


def hard_bars(row: dict) -> dict[str, bool]:
    g = row.get
    d, ratio = g("s2_droop", NAN), g("s2_ratio", NAN)
    return {
        "stability": bool(g("s0_settled") and g("s2_settled") and g("s5_settled") and g("s1_finite")
                          and g("h_finite")),
        "S2": bool(abs(d / row["target"] - 1) <= CAL_TOL and 1.7 <= ratio <= 2.3),
        "S3": bool(g("s3_droop", NAN) <= S3_BAR),
        "S5": bool(g("s5_pen", NAN) <= S5_PEN_BAR and g("s5_in_gap") and g("s5_tilt", NAN) <= S5_TILT_BAR
                   and g("s5_settled")),
        "perf": bool(g("perf", NAN) <= PERF_BAR),
    }


def _f(v, fmt: str = ".3g") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return "NaN" if isinstance(v, float) and math.isnan(v) else format(v, fmt)


def _settle(row: dict, key: str) -> str:
    """Settle time [s], or 'v<speed at 6 s> (<speed at 3 s>)' in mm/s when unsettled."""
    if f"{key}_settled" not in row:
        return "-"
    if row[f"{key}_settled"]:
        return f"{row[f'{key}_t']:.2f}s"
    return f"v{_f(row[f'{key}_v'] * 1e3)} ({_f(row[f'{key}_v3'] * 1e3)})"


TABLE_HEAD = ("| row | tri_ke | cord_ke | edge_ke_flat | c_bend | cal | settle S0 / S2 / S5 | S1 elong [mm] | "
              "H perim [%] | H max cell [%] | S5 perim [%] | S2 droop [mm] | ratio | S3 [mm] | EI_lat pred [N m2] | "
              "S4 twist [rad] | GJ/EI | S5 pen [mm] | in gap | tilt [deg] | ms/frame | HARD | failing / notes |")
TABLE_COLS = 23


def table_row(row: dict) -> str:
    if "hard" not in row:
        return f"| {row['label']} | {_f(row['tri_ke'])} | {_f(row['cord_ke'])} |" + " - |" * 19 + f" {row.get('note', '')} |"
    g = row.get
    fails = [k for k, ok in row["hard"].items() if not ok]
    cal = {True: "yes", False: "uncal.", None: "fixed"}[g("calibrated")]
    notes = ("probe; " if row["probe"] else "") + (", ".join(fails) or "all pass")
    extra = [n for n in (g("note"), g("perf_note")) if n] + [e.split(":")[0] + " error" for e in row["errors"]]
    extra += [f"{s} skipped" for s in row["skipped"]]
    if extra:
        notes += "; " + "; ".join(extra)
    cells = [
        row["label"], _f(row["tri_ke"]), _f(row["cord_ke"]), _f(g("edge_ke_flat")), _f(g("c_bend")), cal,
        f"{_settle(row, 's0')} / {_settle(row, 's2')} / {_settle(row, 's5')}",
        _f(g("s1_elong", NAN) * 1e3), _f(g("h_perim")), _f(g("h_cell")), _f(g("s5_perim")),
        _f(g("s2_droop", NAN) * 1e3), _f(g("s2_ratio")), _f(g("s3_droop", NAN) * 1e3), _f(g("s3_ei_lat")),
        _f(g("s4_twist")), _f(g("s4_gj_ei")), _f(g("s5_pen", NAN) * 1e3), _f(g("s5_in_gap")), _f(g("s5_tilt")),
        _f(g("perf"), ".2f"), "PASS" if not fails else "fail", notes,
    ]
    return "| " + " | ".join(cells) + " |"


def _damping_name(row: dict) -> str:
    return "x1" if tuple(row["damping"]) == PACKAGE_DAMPING else "x0.1"


def recommend(rows: list[dict]) -> str:
    done = [r for r in rows if "hard" in r and not r["probe"]]
    qual = [r for r in done if all(r["hard"].values())]
    if qual:
        best = max(qual, key=lambda r: (r["tri_ke"], -r["cord_ke"], _damping_name(r) == "x1"))
        return (f"[RECOMMEND] tri_ke={best['tri_ke']:g} cord_ke={best['cord_ke']:g} "
                f"edge_ke_flat={best['edge_ke_flat']:.4g} damping={_damping_name(best)}")
    if not done:
        return "[RECOMMEND] none (no point measured)"
    close = min(done, key=lambda r: (sum(not v for v in r["hard"].values()), -r["tri_ke"], r["cord_ke"]))
    fails = ", ".join(k for k, ok in close["hard"].items() if not ok)
    return (f"[RECOMMEND] none; closest tri_ke={close['tri_ke']:g} cord_ke={close['cord_ke']:g} "
            f"edge_ke_flat={close.get('edge_ke_flat', NAN):.4g} damping={_damping_name(close)} fails {fails}")


def _best_membranes(grid: list[dict]) -> list[dict]:
    """Stiffest S0+S2-settling point and the grid point below it (same damping); else the two closest points."""
    def score(r):
        return (sum(not v for v in r["hard"].values()), -r["tri_ke"])
    settling = [r for r in grid if r.get("s0_settled") and r.get("s2_settled")]
    if not settling:
        return sorted(grid, key=score)[:2]
    top = max(settling, key=lambda r: (r["tri_ke"], -score(r)[0]))
    below = [r for r in grid if r["damping"] == top["damping"] and r["tri_ke"] < top["tri_ke"]]
    return [top] + ([max(below, key=lambda r: r["tri_ke"])] if below else [])


def run_sweep(perf: bool) -> int:
    busy = other_gpu_compute()
    print(f"[GPU] other compute processes at start: {'; '.join(busy) or 'none'}", flush=True)
    rows: list[dict] = []
    ceilings: dict[str, float | None] = {}

    def add(row: dict) -> dict:
        rows.append(row)
        print(table_row(row), flush=True)
        return row

    for name, damping in (("x1", PACKAGE_DAMPING), ("x0.1", KNOB_DAMPING)):
        settled_below, ceilings[name] = False, None
        for tri_ke in SWEEP_TRI_KE:
            row = measure_point(soft_material(tri_ke, damping=damping), label=f"grid {name}", calibrate=True, perf=perf)
            if row.get("s0_settled"):
                settled_below = True
            elif settled_below and ceilings[name] is None:
                ceilings[name] = tri_ke
                try:
                    retry = strip_s0(soft_material(tri_ke, 0.0, row.get("edge_ke_flat", CAL_START[0]), damping),
                                     iterations=STRIP_RETRY_ITERATIONS)
                    row["note"] = (f"ceiling; S0 retry at {STRIP_RETRY_ITERATIONS} it: "
                                   f"{_settle({'s0_settled': retry['s0_settled'], **retry}, 's0')}")
                except Exception as exc:  # noqa: BLE001
                    row["note"] = f"ceiling; S0 retry error {type(exc).__name__}"
            add(row)
    grid = [r for r in rows if "hard" in r]
    best = _best_membranes(grid)
    for base in best:
        for cord_ke in SWEEP_CORD_KE:
            add(measure_point(soft_material(base["tri_ke"], cord_ke, base["edge_ke_flat"], base["damping"]),
                              label=f"cord {_damping_name(base)}", calibrate=False, perf=perf))

    top = best[0]
    wide = ("S5", "perf")  # a 50 mm band does not fit the 27 mm plate gap
    heavy = StripProfile(section=replace(STRIP_SECTION, area_density=10.0))
    light_ctrl = replace(CONTROL_PROFILE, section=replace(CONTROL_PROFILE.section, area_density=STRIP_SECTION.area_density))
    narrow_ctrl = replace(light_ctrl, section=replace(light_ctrl.section, width=STRIP_SECTION.width), n_w=STRIP_N_W)
    probes = (
        ("control", CONTROL_MATERIAL, CONTROL_PROFILE, wide),
        ("control rest 0", replace(CONTROL_MATERIAL, zero_rest_angles=True), CONTROL_PROFILE, wide),
        ("control 2.26 kg/m2", CONTROL_MATERIAL, light_ctrl, wide),
        ("control 25 mm 2.26 kg/m2", CONTROL_MATERIAL, narrow_ctrl, ()),
        (f"rib springs {PROBE_RIB_SPRING_KE:g}", replace(soft_material(SWEEP_TRI_KE[0], 0.0, None, top["damping"]),
                                                          rib_spring_ke=PROBE_RIB_SPRING_KE,
                                                          rib_spring_kd=PROBE_RIB_SPRING_KE * 1e-3), SOFT_PROFILE, ()),
        ("best @ 10 kg/m2", soft_material(top["tri_ke"], 0.0, top["edge_ke_flat"], top["damping"]), heavy, ()),
    )
    for label, mat, profile, skip in probes:
        add(measure_point(mat, label=label, calibrate=mat.edge_ke_flat is None, perf=perf and "perf" not in skip,
                          profile=profile, probe=True, skip=skip))

    rec = recommend(rows)
    busy_end = other_gpu_compute()
    lines = [TABLE_HEAD, "|" + "---|" * TABLE_COLS] + [table_row(r) for r in rows]
    print("\n" + "\n".join(lines) + "\n")
    print(rec)
    _write_sweep(rows, lines, rec, busy, busy_end, ceilings)
    return 0


def _write_sweep(rows, lines, rec, busy, busy_end, ceilings) -> None:
    cal = []
    for r in rows:
        if r.get("cal_hist"):
            steps = ", ".join(f"{ke:.4g}->{_f(d * 1e3)} mm{'' if ok else ' (unsettled)'}"
                              for ke, d, _, ok in r["cal_hist"])
            cal.append(f"- {r['label']} tri_ke {r['tri_ke']:g}: edge_ke_flat {r['edge_ke_flat']:.4g} "
                       f"(c_bend {r['c_bend']:.3g}), {'calibrated' if r['calibrated'] else 'uncalibratable'}; "
                       f"secant {steps}")
    errors = [f"- {r['label']} tri_ke {r['tri_ke']:g} cord_ke {r['cord_ke']:g}: {e}"
              for r in rows for e in r.get("errors", [])]
    ceil = ", ".join(f"damping {k}: {'none' if v is None else f'{v:g}'}" for k, v in ceilings.items())
    text = "\n".join([
        "# Soft VBD strip sweep (PKG-20260910-belt-model-spike-soft-strip)",
        "",
        f"Generated by `scripts/check_timing_belt_behaviour.py --model strip --sweep` on {time.strftime('%Y-%m-%d %H:%M')}.",
        "",
        (f"Setup: {ARGS.num_elements} x {STRIP_N_W} cells, 2.26 kg/m^2, particle radius 1.1 mm, SolverVBD "
         f"{STRIP_ITERATIONS} it, self-contact margin/gap 1.1 mm, dt 1/600, 10 substeps/frame; tri_ka = tri_ke, "
         f"rib hinges 100 x edge_ke_flat, rib springs {RIB_SPRING_KE:g}/{RIB_SPRING_KD:g}, cord kd "
         f"{CORD_KD_RATIO:g} cord_ke. Damping x1 = package (tri_kd {PACKAGE_DAMPING[0]:g} tri_ke, edge_kd "
         f"{PACKAGE_DAMPING[1]:g} edge_ke); x0.1 = the package's lowest damping knob. Target droop(EI={ARGS.ei:g}) "
         f"{droop_target() * 1e3:.4g} mm; nominal edge_ke_flat {edge_ke_nominal():.3g} (c_bend 1)."),
        (f"HARD: stability (S0/S2/S5 settle < 1 mm/s in 6 s, S1/H finite); S2 droop +-10 %, ratio [1.7, 2.3]; "
         f"S3 <= {S3_BAR * 1e3:g} mm (strict {S3_STRICT * 1e3:g} mm reported); S5 pen <= {S5_PEN_BAR * 1e3:g} mm, "
         f"rows in plate gap, tilt <= {S5_TILT_BAR:g} deg, settled; perf <= {PERF_BAR:g} ms/frame."),
        ("Settle cells: time to mean particle speed < 1 mm/s, or when unsettled `v<mean speed over 5.5-6 s> "
         "(<mean speed over 2.5-3 s>)` in mm/s."),
        ("Probe rows (not candidates): `control` = timing_belt.py's belt (50 mm, 10 kg/m^2, r 1 mm, tri_ke 1e3, "
         "tri_kd 1e2, hinges 10 / ribs 2e4, rib springs 2e4, mesh rest angles, 20 substeps, legacy self-contact "
         "radius 1 mm / margin 1.5 mm); its S2 compares against the 2e-4 target at its own weight; S5/perf skipped "
         "for 50 mm bands."),
        f"Ceiling membrane (first tri_ke above a settling one that does not settle in S0): {ceil}.",
        f"GPU compute processes other than the sweep: start {'; '.join(busy) or 'none'}; end {'; '.join(busy_end) or 'none'}.",
        "",
        *lines,
        "",
        rec,
        "",
        "## Bending calibration",
        *cal,
        *(["", "## Stage errors", *errors] if errors else []),
        "",
    ])
    SWEEP_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_RESULTS.write_text(text)
    print(f"[SWEEP] wrote {SWEEP_RESULTS}")


def run_strip_point(args) -> int:
    mat = operating_material(args.tri_ke, args.cord_ke, args.edge_ke_flat)
    print(f"[POINT] tri_ke={mat.tri_ke:g} cord_ke={mat.cord_ke:g} edge_ke_flat="
          f"{'calibrate' if mat.edge_ke_flat is None else format(mat.edge_ke_flat, '.4g')} "
          f"tri_kd/edge_kd ratios {mat.tri_kd:g}/{mat.edge_kd:g}", flush=True)
    row = measure_point(mat, label="point", calibrate=mat.edge_ke_flat is None, perf=not args.no_perf)
    g = row.get
    target = row["target"]
    print(f"[INFO] S0 preamble: rest angle max {_f(g('s0_rest_angle'))}, mass {_f(g('s0_mass'), '.5f')} kg, "
          f"hinges {g('s0_hinges')}, springs {g('s0_springs')}")
    if row.get("cal_hist"):
        print(f"[INFO] calibration: edge_ke_flat {row['edge_ke_flat']:.4g} (c_bend {row['c_bend']:.3g}), "
              f"{'calibrated' if row['calibrated'] else 'uncalibratable'} in {row['cal_evals']} droop runs")
    bars = {
        "stability": (f"settle S0/S2/S5 {_settle(row, 's0')} / {_settle(row, 's2')} / {_settle(row, 's5')}; "
                      f"S1 finite {_f(g('s1_finite'))}; H finite {_f(g('h_finite'))}"),
        "S2": (f"droop {_f(g('s2_droop', NAN) * 1e3, '.4g')} mm in {target * 0.9e3:.4g}..{target * 1.1e3:.4g}; "
               f"ratio {_f(g('s2_ratio'), '.3f')} in [1.7, 2.3]; edge_ke_flat {_f(g('edge_ke_flat'), '.4g')} "
               f"(c_bend {_f(g('c_bend'))})"),
        "S3": (f"edgewise droop {_f(g('s3_droop', NAN) * 1e3, '.4g')} mm <= {S3_BAR * 1e3:g} "
               f"(strict {S3_STRICT * 1e3:g}: {'pass' if g('s3_droop', NAN) <= S3_STRICT else 'fail'}); "
               f"predicted EI_lat {_f(g('s3_ei_lat'))} N m^2"),
        "S5": (f"penetration {_f(g('s5_pen', NAN) * 1e3, '.3f')} mm <= {S5_PEN_BAR * 1e3:g}; rows in plate gap "
               f"{_f(g('s5_in_gap'))}; max row tilt {_f(g('s5_tilt'), '.2f')} deg <= {S5_TILT_BAR:g}; settled "
               f"{_f(g('s5_settled'))}"),
        "perf": f"{_f(g('perf'), '.2f')} ms/frame <= {PERF_BAR:g}" + (f" ({g('perf_note')})" if g("perf_note") else ""),
    }
    if args.no_perf:
        row["hard"].pop("perf")
        bars.pop("perf")
    for name, text in bars.items():
        print(f"[{'PASS' if row['hard'][name] else 'FAIL'}] {name}: {text}")
    print(f"[REPORT] S0 aspect {_f(g('s0_aspect'), '.4f')}, radius {_f(g('s0_radius', NAN) * 1e3, '.2f')} mm; "
          f"S1 elongation {_f(g('s1_elong', NAN) * 1e3, '.4g')} mm; H perimeter {_f(g('h_perim'))} %, max cell "
          f"{_f(g('h_cell'))} %; S5 perimeter {_f(g('s5_perim'))} %; S4 twist {_f(g('s4_twist'), '.4g')} rad "
          f"(GJ/EI {_f(g('s4_gj_ei'))})")
    for e in row["errors"]:
        print(f"[ERROR] {e}", file=sys.stderr)
    if all(row["hard"].values()):
        print("ALL STRIP HARD BARS PASSED")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Warp device (e.g. 'cuda:0', 'cpu').")
    parser.add_argument("--model", choices=("chain", "strip"), default="chain", help="belt model to check")
    parser.add_argument("--num-elements", type=int, default=135, help="segments (chain) / cells along (strip)")
    parser.add_argument("--width-cells", type=int, default=STRIP_N_W, help="strip cells across the width")
    parser.add_argument("--ei", type=float, default=2.0e-4, help="flat bend rigidity [N m^2]")
    parser.add_argument("--gj", type=float, default=2.0e-4, help="twist rigidity [N m^2]")
    parser.add_argument("--sweep", action="store_true", help="strip: run the membrane/cord sweep table")
    parser.add_argument("--tri-ke", type=float, default=None, help="strip: membrane tri_ke = tri_ka override")
    parser.add_argument("--cord-ke", type=float, default=None, help="strip: edge-cord spring ke override [N/m]")
    parser.add_argument("--edge-ke-flat", type=float, default=None, help="strip: across-width hinge ke override")
    parser.add_argument("--full-scene", action="store_true",
                        help="reserved for the full-scene checks (not implemented yet)")
    parser.add_argument("--no-perf", action="store_true", help="skip the [PERF] measurements")
    args = parser.parse_args()
    if args.full_scene:
        parser.error("--full-scene is reserved and not implemented yet")
    if args.model != "strip" and (args.sweep or args.tri_ke or args.cord_ke or args.edge_ke_flat):
        parser.error("--sweep/--tri-ke/--cord-ke/--edge-ke-flat need --model strip")
    ARGS.num_elements, ARGS.ei, ARGS.gj = args.num_elements, args.ei, args.gj
    STRIP_ARGS.n_w = args.width_cells
    if args.device:
        wp.set_device(args.device)

    if args.model == "strip":
        return run_sweep(perf=not args.no_perf) if args.sweep else run_strip_point(args)

    for name, fn in CHECKS["isolated"]:
        try:
            msg = fn()
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            return 1
        print(f"[PASS] {name}: {msg}", flush=True)
    if not args.no_perf and ARGS.num_elements == 135:
        try:
            run_perf()
        except AssertionError as exc:
            print(f"[FAIL] P performance: {exc}", file=sys.stderr)
            return 1

    print("ALL BELT BEHAVIOUR CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
