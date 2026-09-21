"""Timing belt as a SolverMuJoCo chain of rigid boxes: D6 hinges (flat bend + twist), loop closed by CONNECT."""

from __future__ import annotations

import math
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton.solvers import SolverMuJoCo


@dataclass(frozen=True)
class BeltSection:
    width: float
    thickness: float
    density: float


@dataclass(frozen=True)
class BeltMaterial:
    bend_rigidity: float
    twist_rigidity: float
    bend_damping: float
    twist_damping: float
    armature: float = 2.0e-6


def belt_joint_count(num_segments: int, closed: bool) -> int:
    del closed  # the loop is closed by an equality, not a joint
    return 1 + (num_segments - 1)


def _frames(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-segment lengths and rotation matrices (cols: width X, Y = Z x X, tangent Z)."""
    seg = points[1:] - points[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError("add_timing_belt_chain: zero-length segment")
    rots = np.empty((len(seg), 3, 3))
    for i, (d, L) in enumerate(zip(seg, lengths)):
        z = d / L
        x = up - np.dot(up, z) * z
        n = np.linalg.norm(x)
        if n < 1e-6:
            raise ValueError(f"add_timing_belt_chain: segment {i} is parallel to up")
        x /= n
        rots[i] = np.column_stack((x, np.cross(z, x), z))
    return lengths, rots


def _xz_angles(r_parent: np.ndarray, r_child: np.ndarray) -> tuple[float, float]:
    """(bend about X, twist about Z) with R_rel = Rx(q0) Rz(q1), the D6 FK composition."""
    r = r_parent.T @ r_child
    return math.atan2(-r[1, 2], r[2, 2]), math.atan2(-r[0, 1], r[0, 0])


def add_timing_belt_chain(
    builder: newton.ModelBuilder,
    points,
    *,
    up,
    section: BeltSection,
    material: BeltMaterial,
    closed: bool,
    cfg: newton.ModelBuilder.ShapeConfig,
    label: str,
    color=None,
    root_kinematic: bool = False,
) -> tuple[list[int], list[int]]:
    """Needs ``SolverMuJoCo.register_custom_attributes(builder)`` first; returns (bodies, joints)."""
    pts = np.asarray([[float(c) for c in p] for p in points], dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("add_timing_belt_chain: need at least 2 segments")
    if closed and np.linalg.norm(pts[-1] - pts[0]) > 1e-9:
        raise ValueError("add_timing_belt_chain: closed=True needs points[-1] == points[0]")
    up_np = np.asarray([float(c) for c in up], dtype=np.float64)
    lengths, rots = _frames(pts, up_np / np.linalg.norm(up_np))
    n_seg = len(lengths)

    shape_cfg = cfg.copy()
    shape_cfg.density = section.density
    hx, hy = 0.5 * section.width, 0.5 * section.thickness

    bodies: list[int] = []
    for i in range(n_seg):
        mid = 0.5 * (pts[i] + pts[i + 1])
        q = wp.quat_from_matrix(wp.mat33(*rots[i].flatten().tolist()))
        body = builder.add_link(xform=wp.transform(wp.vec3(*mid.tolist()), q), label=f"{label}_seg_{i}",
                                is_kinematic=root_kinematic and i == 0)
        builder.add_shape_box(body, hx=hx, hy=hy, hz=0.5 * lengths[i], cfg=shape_cfg, color=color)
        bodies.append(body)

    dof = newton.ModelBuilder.JointDofConfig
    joints = [builder.add_joint_free(bodies[0], label=f"{label}_root")]
    for i in range(n_seg - 1):
        l_p, l_c = float(lengths[i]), float(lengths[i + 1])
        l_dual = 0.5 * (l_p + l_c)
        theta = list(_xz_angles(rots[i], rots[i + 1]))
        j = builder.add_joint_d6(
            bodies[i], bodies[i + 1],
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * l_p), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * l_c), wp.quat_identity()),
            linear_axes=[],
            angular_axes=[
                dof(axis=newton.Axis.X, damping=material.bend_damping, armature=material.armature),
                dof(axis=newton.Axis.Z, damping=material.twist_damping, armature=material.armature),
            ],
            label=f"{label}_joint_{i}", collision_filter_parent=True,
            # qpos = joint_q + ref: ref = θ makes the CONNECT reference pose the loop; springref = θ is joint_q = 0 (straight).
            custom_attributes={
                "mujoco:dof_passive_stiffness": [material.bend_rigidity / l_dual,
                                                 material.twist_rigidity / l_dual],
                "mujoco:dof_ref": theta,
                "mujoco:dof_springref": theta,
            },
        )
        qs = builder.joint_q_start[j]
        builder.joint_q[qs], builder.joint_q[qs + 1] = theta
        joints.append(j)
    builder.add_articulation(list(joints), label=f"{label}_articulation")
    if closed:
        _add_equality_constraint(builder, SolverMuJoCo.EqType.CONNECT, body1=bodies[-1], body2=bodies[0],
                                 anchor=(0.0, 0.0, 0.5 * float(lengths[-1])), label=f"{label}_closure")
    return bodies, joints


def _resample_closed(dense: np.ndarray, num_segments: int) -> list[wp.vec3]:
    """Equal-arc-length resampling of a closed dense polyline (dense[-1] == dense[0])."""
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
    targets = np.linspace(0.0, s[-1], num_segments + 1)
    out = np.column_stack([np.interp(targets, s, dense[:, k]) for k in range(3)])
    out[-1] = out[0]
    return [wp.vec3(*p.tolist()) for p in out]


def ellipse_points(center, semi_axes, num_segments: int) -> list[wp.vec3]:
    c = np.asarray([float(v) for v in center])
    t = np.linspace(0.0, 2.0 * math.pi, 200_001)
    dense = np.column_stack((c[0] + semi_axes[0] * np.cos(t), c[1] + semi_axes[1] * np.sin(t),
                             np.full_like(t, c[2])))
    dense[-1] = dense[0]
    return _resample_closed(dense, num_segments)


def two_pulley_path_length(c_small, r_small: float, c_large, r_large: float) -> float:
    d = float(np.linalg.norm(np.asarray(c_large[:2], float) - np.asarray(c_small[:2], float)))
    beta = math.asin((r_large - r_small) / d)
    return r_small * (math.pi - 2 * beta) + r_large * (math.pi + 2 * beta) + 2 * d * math.cos(beta)


def two_pulley_path(c_small, r_small: float, c_large, r_large: float, num_segments: int,
                    z: float) -> list[wp.vec3]:
    """Taut open-belt path (CCW) around two circles of the given centreline radii at height z."""
    cs, cl = np.asarray(c_small[:2], float), np.asarray(c_large[:2], float)
    d = float(np.linalg.norm(cl - cs))
    e = (cl - cs) / d
    e_perp = np.array([-e[1], e[0]])
    beta = math.asin((r_large - r_small) / d)

    def on(center, radius, theta):
        return center + radius * (np.cos(theta)[:, None] * e + np.sin(theta)[:, None] * e_perp)

    a = 0.5 * math.pi + beta
    m = 20_000
    large = on(cl, r_large, np.linspace(-a, a, m))
    small = on(cs, r_small, np.linspace(a, 2.0 * math.pi - a, m))
    dense2 = np.vstack((large, small, large[:1]))
    dense = np.column_stack((dense2, np.full(len(dense2), float(z))))
    return _resample_closed(dense, num_segments)
