#!/usr/bin/env python3
"""Export the round-belt scene's visible geometry as one STL per part, for CAD import.

Builds the scene with ``round_belt_task.build_scene``, poses it at the simulation's initial
joint state (``apply_default_joint_state``: q_init arms, open gripper), and writes each part's visible shapes as an STL (meters) in the part's own
frame, plus ``scene.json`` with every part's world pose. Parts are the robot and pulley bodies,
the static models grouped by ``<model>/<link>`` label, and the belt merged into one part at its
rest shape. Feed the output to magna's ``python/push_scene_to_onshape.py``.

Run:
    uv run python scripts/export_scene_meshes.py --out /tmp/round_belt_scene
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import trimesh
import warp as wp

import newton

import round_belt_task
from round_belt_task.joint_state import apply_default_joint_state


def tf_mat4(tf) -> np.ndarray:
    """(px py pz qx qy qz qw) -> 4x4."""
    row = [float(v) for v in tf]
    m = np.eye(4)
    m[:3, :3] = np.array(wp.quat_to_matrix(wp.quat(*row[3:7])), dtype=np.float64).reshape(3, 3)
    m[:3, 3] = row[:3]
    return m


def shape_mesh(geo_type: int, scale, source) -> trimesh.Trimesh | None:
    """A shape's geometry in its own frame, or None for shapes with no CAD meaning (planes)."""
    sx, sy, sz = (float(s) for s in scale)
    if geo_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
        vertices = np.asarray(source.vertices, dtype=np.float64) * [sx, sy, sz]
        return trimesh.Trimesh(vertices, np.asarray(source.indices).reshape(-1, 3), process=False)
    if geo_type == newton.GeoType.BOX:
        return trimesh.creation.box(extents=[2 * sx, 2 * sy, 2 * sz])
    if geo_type == newton.GeoType.SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=sx)
    if geo_type == newton.GeoType.CAPSULE:
        return trimesh.creation.capsule(height=2 * sy, radius=sx, count=[24, 12])
    if geo_type == newton.GeoType.CYLINDER:
        return trimesh.creation.cylinder(radius=sx, height=2 * sy, sections=48)
    if geo_type == newton.GeoType.ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3)
        mesh.apply_scale([sx, sy, sz])
        return mesh
    if geo_type == newton.GeoType.PLANE:
        return None
    raise ValueError(f"Unsupported shape type {newton.GeoType(geo_type).name}")


def body_part_name(label: str) -> str:
    """'robotiq_2f85/worldbody/base_mount/base' -> 'robotiq_2f85::base' (model::body)."""
    parts = [p for p in label.split("/") if p]
    return f"{parts[0]}::{parts[-1]}"


def static_part_name(label: str) -> str:
    """'table/scene/visual0' -> 'table::scene' (model::link)."""
    return "::".join([p for p in label.split("/") if p][:2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    args = parser.parse_args()

    builder = round_belt_task.make_builder()
    info = round_belt_task.build_scene(builder)
    model = builder.finalize(device="cpu")
    apply_default_joint_state(model, info)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()
    belt_bodies = set(int(b) for b in info.belt_bodies)

    # part name -> (X_WP, [mesh in part frame])
    parts: dict[str, tuple[np.ndarray, list[trimesh.Trimesh]]] = {}
    shapes_per_part: dict[str, int] = defaultdict(int)
    for s in range(builder.shape_count):
        if not builder.shape_flags[s] & newton.ShapeFlags.VISIBLE:
            continue
        mesh = shape_mesh(builder.shape_type[s], builder.shape_scale[s], builder.shape_source[s])
        if mesh is None:
            continue
        body = builder.shape_body[s]
        X_BS = tf_mat4(builder.shape_transform[s])
        if body in belt_bodies:
            # One rigid part for the whole belt, at its rest shape, in the world frame.
            name, X_WP, X_PS = "round_belt", np.eye(4), tf_mat4(body_q[body]) @ X_BS
        elif body >= 0:
            name, X_WP, X_PS = body_part_name(builder.body_label[body]), tf_mat4(body_q[body]), X_BS
        else:
            # Static shapes are posed in the world; give each model/link group its own part.
            name, X_WP, X_PS = static_part_name(builder.shape_label[s]), np.eye(4), X_BS
        mesh.apply_transform(X_PS)
        X_WP_seen, meshes = parts.setdefault(name, (X_WP, []))
        if not np.allclose(X_WP_seen, X_WP):
            raise ValueError(f"Two bodies map to part {name!r}")
        meshes.append(mesh)
        shapes_per_part[name] += 1

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (X_WP, meshes) in parts.items():
        stl = args.out / f"{re.sub(r'[^A-Za-z0-9_.-]+', '__', name)}.stl"
        trimesh.util.concatenate(meshes).export(stl)
        manifest.append({"name": name, "X_WB": X_WP.tolist(), "stl": str(stl.resolve())})
        print(f"{name}: {shapes_per_part[name]} shapes")
    (args.out / "scene.json").write_text(json.dumps({"bodies": manifest}, indent=2))
    print(f"Wrote {len(manifest)} parts to {args.out}")


if __name__ == "__main__":
    main()
