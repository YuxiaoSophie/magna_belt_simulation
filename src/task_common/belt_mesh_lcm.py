"""A tube mesh around a closed loop of belt-body centres, wrapped as a Drake viewer message.

Vectorised with numpy so it stays cheap to call every publish (see
``scripts/checks/check_lcm_contract.py`` check 11 for the ``float_data`` layout:
``V | T | vertices | triangles``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drake import lcmt_viewer_geometry_data, lcmt_viewer_link_data

# Drake's fixed link name for a scene's lone deformable-geometry entry.
DEFORMABLE_LINK_NAME = "deformable_geometries"


def tube_mesh(
    centres: np.ndarray, radius: float, sides: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """A closed tube (ring of ``sides`` vertices per centre) around a closed loop of centres.

    Returns ``(vertices, triangles)``: vertices ``(len(centres) * sides, 3)`` float32,
    triangles ``(len(centres) * sides * 2, 3)`` int32 (two per quad between rings i, i+1).
    """
    centres = np.asarray(centres, dtype=np.float64)
    n = centres.shape[0]
    tangent = np.roll(centres, -1, axis=0) - np.roll(centres, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    axis = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    # Avoid a near-parallel tangent/axis (degenerate cross product) at the loop's poles.
    axis[np.abs(np.sum(tangent * axis, axis=1)) > 0.9] = (1.0, 0.0, 0.0)
    u = np.cross(tangent, axis)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(tangent, u)

    angle = 2.0 * np.pi * np.arange(sides) / sides
    ring = radius * (
        np.cos(angle)[None, :, None] * u[:, None, :] + np.sin(angle)[None, :, None] * v[:, None, :]
    )
    vertices = (centres[:, None, :] + ring).reshape(n * sides, 3).astype(np.float32)

    i, k = np.meshgrid(np.arange(n), np.arange(sides), indexing="ij")
    i2, k2 = (i + 1) % n, (k + 1) % sides
    a, b, c, d = sides * i + k, sides * i2 + k, sides * i2 + k2, sides * i + k2
    triangles = np.stack([a, b, c, a, c, d], axis=-1).reshape(n * sides, 2, 3)
    triangles = triangles.reshape(n * sides * 2, 3).astype(np.int32)
    return vertices, triangles


def deformable_mesh_msg(
    vertices: np.ndarray, triangles: np.ndarray, name: str, color: Sequence[float]
) -> lcmt_viewer_link_data:
    """A one-geometry ``lcmt_viewer_link_data`` carrying the raw triangle mesh in-line."""
    v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    t = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    geom = lcmt_viewer_geometry_data()
    geom.type = lcmt_viewer_geometry_data.MESH
    geom.position = [0.0, 0.0, 0.0]
    geom.quaternion = [1.0, 0.0, 0.0, 0.0]
    geom.color = [float(c) for c in color]
    geom.string_data = name
    geom.float_data = [
        float(v.shape[0]), float(t.shape[0]), *v.ravel().tolist(), *t.ravel().tolist()
    ]
    geom.num_float_data = len(geom.float_data)

    msg = lcmt_viewer_link_data()
    msg.name = DEFORMABLE_LINK_NAME
    msg.robot_num = 0
    msg.geom = [geom]
    msg.num_geom = 1
    return msg
