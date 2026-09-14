"""Timing belt as a VBD cloth band: membrane, per-class hinges, rib and edge-cord springs, straight flat rest."""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

AXIAL_STIFFNESS = 2.1e5  # EA [N]
POISSON = 0.3


@dataclass(frozen=True)
class StripSection:
    width: float
    thickness: float
    area_density: float


@dataclass(frozen=True)
class StripMaterial:
    membrane_scale: float = 1.0
    bend_rigidity: float = 2.0e-4
    rib_ratio: float = 100.0
    bend_calibration: float = 1.0
    tri_kd: float = 0.1  # kd/ke ratio
    edge_kd: float = 1.0e-2  # kd/ke ratio
    tri_ke: float | None = None  # tri_ke = tri_ka; overrides membrane_scale
    edge_ke_flat: float | None = None  # across-width hinge ke; overrides bend_rigidity * bend_calibration
    rib_spring_ke: float = 0.0
    rib_spring_kd: float = 0.0
    cord_ke: float = 0.0
    cord_kd: float = 0.0
    zero_rest_angles: bool = True  # False keeps add_cloth_mesh's rest angles (the initial shape is the rest shape)
    geometric_hinges: bool = True  # False: across/diagonal = edge_ke_flat, along = rib_ratio x edge_ke_flat, as timing_belt.py


# Previous spike's stiff-membrane defaults.
STIFF_STRIP_MATERIAL = StripMaterial(membrane_scale=1.0 / 40.0, tri_kd=1.0e-8, edge_kd=1.0e-4)

SOFT_STRIP_CALIBRATED_EDGE_KE: dict[float, float] = {}

SOFT_STRIP_OPERATING_POINT = StripMaterial(
    tri_ke=1.0e3, edge_ke_flat=4.8, rib_spring_ke=2.0e4, rib_spring_kd=20.0)


def _points(centreline_points, closed: bool) -> np.ndarray:
    pts = np.asarray([[float(c) for c in p] for p in centreline_points], dtype=np.float64)
    if closed and len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) < 1e-9:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("band_mesh: need at least 3 centreline points")
    return pts


def band_mesh(centreline_points, up, width: float, n_w: int, closed: bool) -> tuple[list[wp.vec3], list[int]]:
    """Rows across the width, row i = centreline point i; particle (i, j) at index i * (n_w + 1) + j."""
    pts = _points(centreline_points, closed)
    n = len(pts)
    if closed:
        tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    else:
        tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    up_np = np.asarray([float(c) for c in up], dtype=np.float64)
    ups = up_np[None, :] - np.sum(tang * up_np, axis=1, keepdims=True) * tang
    norms = np.linalg.norm(ups, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("band_mesh: centreline tangent parallel to up")
    ups /= norms
    offsets = (np.arange(n_w + 1) / n_w - 0.5) * width
    verts = pts[:, None, :] + offsets[None, :, None] * ups[:, None, :]
    vertices = [wp.vec3(*v) for v in verts.reshape(-1, 3).tolist()]

    row = n_w + 1
    indices: list[int] = []
    for i in range(n if closed else n - 1):
        ni = (i + 1) % n
        for j in range(n_w):
            v00, v10, v11, v01 = i * row + j, ni * row + j, ni * row + j + 1, i * row + j + 1
            indices.extend((v00, v10, v01, v10, v11, v01))
    return vertices, indices


def _tri_area(p: np.ndarray, a: int, b: int, c: int) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(p[b] - p[a], p[c] - p[a])))


def add_timing_belt_strip(
    builder: newton.ModelBuilder,
    centreline_points,
    *,
    up,
    section: StripSection,
    material: StripMaterial,
    n_w: int,
    closed: bool,
    color=None,
    label: str | None = None,
    particle_radius: float | None = None,
) -> dict:
    vertices, indices = band_mesh(centreline_points, up, section.width, n_w, closed)
    n_rows = len(vertices) // (n_w + 1)
    if material.tri_ke is None:
        et = AXIAL_STIFFNESS / section.width * material.membrane_scale
        mu = et / (2.0 * (1.0 + POISSON))
        lam = et * POISSON / (1.0 - POISSON**2)
    else:
        mu = lam = material.tri_ke
    if material.edge_ke_flat is None:
        d_flat = material.bend_rigidity / section.width * material.bend_calibration
    else:
        pts = _points(centreline_points, closed)
        seg = np.linalg.norm(np.diff(np.vstack((pts, pts[:1])) if closed else pts, axis=0), axis=1)
        d_flat = material.edge_ke_flat * float(np.mean(seg)) / 3.0  # across ke = 3 D / a
    d_rib = material.rib_ratio * d_flat

    p0, t0, e0 = builder.particle_count, builder.tri_count, builder.edge_count
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices, indices=indices, density=section.area_density,
        tri_ke=mu, tri_ka=lam, tri_kd=material.tri_kd * mu, edge_ke=0.0, edge_kd=0.0,
        particle_radius=0.5 * section.thickness if particle_radius is None else particle_radius,
        color=color, label=label)
    p1, t1, e1 = builder.particle_count, builder.tri_count, builder.edge_count
    if p1 - p0 != len(vertices):
        raise RuntimeError("add_timing_belt_strip: add_cloth_mesh changed the vertex count")

    pos = np.asarray([[float(c) for c in v] for v in vertices])
    counts = {"across": 0, "along": 0, "diagonal": 0, "boundary": 0}
    for e in range(e0, e1):
        o0, o1, h0, h1 = builder.edge_indices[e]
        if material.zero_rest_angles:
            builder.edge_rest_angle[e] = 0.0
        if o0 == -1 or o1 == -1:
            builder.edge_bending_properties[e] = (0.0, 0.0)
            counts["boundary"] += 1
            continue
        (i0, j0), (i1, j1) = divmod(h0 - p0, n_w + 1), divmod(h1 - p0, n_w + 1)
        if i0 == i1:
            cls, d = "across", d_flat
        elif j0 == j1:
            cls, d = "along", d_rib
        else:
            cls, d = "diagonal", d_flat
        counts[cls] += 1
        if material.geometric_hinges:
            e_len = float(np.linalg.norm(pos[h1 - p0] - pos[h0 - p0]))
            area = (_tri_area(pos, o0 - p0, h0 - p0, h1 - p0) + _tri_area(pos, o1 - p0, h0 - p0, h1 - p0))
            ke = 3.0 * d * e_len / area
        else:
            ke = material.edge_ke_flat * (material.rib_ratio if cls == "along" else 1.0)
        builder.edge_bending_properties[e] = (ke, material.edge_kd * ke)

    s0 = builder.spring_count
    row = n_w + 1
    if material.rib_spring_ke > 0.0:
        for i in range(n_rows):
            bottom, top = p0 + i * row, p0 + i * row + n_w
            builder.add_spring(bottom, top, material.rib_spring_ke, material.rib_spring_kd, 0.0)
            for j in range(1, n_w):
                builder.add_spring(bottom, bottom + j, material.rib_spring_ke, material.rib_spring_kd, 0.0)
                builder.add_spring(bottom + j, top, material.rib_spring_ke, material.rib_spring_kd, 0.0)
    s1 = builder.spring_count
    if material.cord_ke > 0.0:
        for i in range(n_rows if closed else n_rows - 1):
            ni = (i + 1) % n_rows
            for j in (0, n_w):
                builder.add_spring(p0 + i * row + j, p0 + ni * row + j, material.cord_ke, material.cord_kd, 0.0)
    return {
        "particles": range(p0, p1),
        "triangles": range(t0, t1),
        "edges": range(e0, e1),
        "rows": (n_rows, n_w + 1),
        "hinges": counts,
        "springs": {"rib": s1 - s0, "cord": builder.spring_count - s1},
    }


def strip_centreline(particle_q, rows) -> np.ndarray:
    q = np.asarray(particle_q, dtype=np.float64)
    return q.reshape(rows[0], rows[1], 3).mean(axis=1)


def strip_row_vectors(particle_q, rows) -> np.ndarray:
    """Last minus first particle of each row."""
    q = np.asarray(particle_q, dtype=np.float64).reshape(rows[0], rows[1], 3)
    return q[:, -1] - q[:, 0]
