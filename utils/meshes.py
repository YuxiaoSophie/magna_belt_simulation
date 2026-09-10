"""Mesh loading and geometry helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import warp as wp

import newton

_MESH_CACHE: dict[str, list[newton.Mesh]] = {}


def _load_meshes_per_material(key: str, *, split_components: bool = False) -> list[newton.Mesh]:
    """Load a mesh file as one ``newton.Mesh`` per material, keeping each material's look.

    Both loaders Newton ships collapse a multi-material asset: ``Mesh.create_from_file``
    drops material data, and ``load_meshes_from_file`` uses ``trimesh.load(force="mesh")``,
    which concatenates sub-meshes and re-bakes one texture atlas -- for ``table.gltf``
    (7 sub-meshes, one textured) that atlas came out mostly black, rendering the table
    blue. Loading the scene and dumping it keeps materials and node transforms.
    """
    import trimesh

    loaded = trimesh.load(key, process=False)
    geoms = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
    if split_components:
        # One material, many disconnected solids (the panel, each pulley bracket, every
        # bolt). Splitting lets them be coloured independently.
        split: list = []
        for geom in geoms:
            try:
                split.extend(geom.split(only_watertight=False))
            except Exception:
                split.append(geom)
        geoms = split or geoms

    meshes: list[newton.Mesh] = []
    for geom in geoms:
        if not hasattr(geom, "faces") or len(geom.faces) == 0:
            continue
        visual = getattr(geom, "visual", None)
        material = getattr(visual, "material", None)

        texture = None
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
        uvs = getattr(visual, "uv", None)
        if image is not None and uvs is not None and len(uvs) == len(geom.vertices):
            texture = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            uvs = None

        color = None
        if texture is None:
            factor = getattr(material, "baseColorFactor", None)
            if factor is None:
                factor = getattr(material, "main_color", None)
            if factor is not None:
                rgb = np.asarray(factor, dtype=np.float64)[:3]
                # trimesh reports PBR factors as 0-255; newton wants 0-1.
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                color = tuple(float(c) for c in rgb)

        meshes.append(
            newton.Mesh(
                np.asarray(geom.vertices, dtype=np.float32),
                np.asarray(geom.faces, dtype=np.int32).flatten(),
                uvs=None if uvs is None else np.asarray(uvs, dtype=np.float32),
                compute_inertia=False,
                is_solid=False,
                color=color,
                texture=texture,
            )
        )
    if not meshes:
        raise RuntimeError("no drawable geometry")
    return meshes


def _load_merged_mesh(path: Path) -> list[newton.Mesh]:
    return [newton.Mesh.create_from_file(str(path), compute_inertia=False, is_solid=False)]


def load_meshes(
    path: Path, *, per_material: bool = True, split_components: bool = False
) -> list[newton.Mesh]:
    """Load a mesh file, keeping any material texture the asset authored (cached).

    ``newton.Mesh.create_from_file`` discards material data, so the table and Franka
    mount plate came out untextured; the per-material path uses ``load_meshes_from_file``,
    the same loader Newton's own ``add_urdf`` runs. For ``table.gltf``, ``wooden_plate.gltf`` and
    ``base_plate.obj`` both loaders return identical vertices (verified: matching AABBs),
    so the per-material path moves no geometry. Falls back to the merged loader if the
    per-material load fails; ``per_material=False`` is the collider path.
    """
    key = f"{path}|mat={per_material}|split={split_components}"
    meshes = _MESH_CACHE.get(key)
    if meshes is None:
        if not per_material:
            meshes = _load_merged_mesh(path)
        else:
            try:
                meshes = _load_meshes_per_material(str(path), split_components=split_components)
            except Exception as exc:  # noqa: BLE001 - appearance must never break the build
                print(f"[MESH] {path.name}: per-material load failed ({exc}); using merged mesh")
                meshes = _load_merged_mesh(path)
            if not meshes:
                raise RuntimeError(f"no meshes loaded from {path}")
        _MESH_CACHE[key] = meshes
    return meshes


def fix_inverted_mesh_winding(
    builder: newton.ModelBuilder, shape_start: int, shape_end: int
) -> list[str]:
    """Flip meshes that wind inward so they stop rendering see-through; returns their labels.

    ``finger_holder.obj`` (Franka hand) is authored inside-out: NEGATIVE signed volume
    (-2.498e-05, divergence theorem ``V = sum(dot(v0, cross(v1, v2))) / 6``), so back-face
    culling hides it. Same asset upstream in Drake, so not a copy artefact. Visual-only:
    the finger's *collision* mesh (``long_finger.obj``) is wound correctly.
    """
    flipped: list[str] = []
    for shape in range(shape_start, shape_end):
        mesh = builder.shape_source[shape]
        if mesh is None:
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        indices = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
        if len(indices) == 0 or len(vertices) == 0:
            continue
        v0, v1, v2 = vertices[indices[:, 0]], vertices[indices[:, 1]], vertices[indices[:, 2]]
        volume = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
        # Only act on a clearly inverted solid; open shells sit near zero and must be left be.
        if volume >= -1e-9:
            continue
        normals = getattr(mesh, "normals", None)
        builder.shape_source[shape] = newton.Mesh(
            vertices.astype(np.float32),
            indices[:, ::-1].astype(np.int32).flatten(),
            normals=None if normals is None else (-np.asarray(normals, dtype=np.float32)),
            uvs=getattr(mesh, "uvs", None),
            compute_inertia=False,
            is_solid=False,
            color=getattr(mesh, "color", None),
            texture=getattr(mesh, "texture", None),
        )
        flipped.append(builder.shape_label[shape] or f"shape_{shape}")
    return flipped


def neutralize_textured_shape_colors(
    builder: newton.ModelBuilder, shape_start: int, shape_end: int
) -> int:
    """Render textured meshes that carry no base colour untinted; returns how many.

    Newton's glTF path has no equivalent of its COLLADA material parser, so a
    texture-only material leaves ``Mesh.color = None`` (``utils/mesh.py:1526-1550``) and
    ModelBuilder's debug-palette colour is multiplied into the texture by the GL shader
    (``albedo = ObjectColor^2.2 * texture^2.2``). ``mesh.py:1404-1405`` guards exactly
    this in the sibling branch; Newton's own Franka escapes it by shipping COLLADA.
    """
    fixed = 0
    for shape in range(shape_start, shape_end):
        mesh = builder.shape_source[shape]
        if mesh is None or getattr(mesh, "texture", None) is None:
            continue
        if getattr(mesh, "color", None) is not None:
            continue  # the asset authored a base colour; respect it
        builder.shape_color[shape] = (1.0, 1.0, 1.0)
        fixed += 1
    return fixed


def _mesh_vertices(mesh: newton.Mesh) -> np.ndarray:
    verts = getattr(mesh, "vertices", None)
    if verts is None:
        raise AttributeError("newton.Mesh has no 'vertices' attribute")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3)


def _transform_points(xform: wp.transform, pts: np.ndarray) -> np.ndarray:
    p = np.array([float(v) for v in wp.transform_get_translation(xform)], dtype=np.float64)
    rot = np.array(wp.quat_to_matrix(wp.transform_get_rotation(xform)), dtype=np.float64)
    return pts @ rot.reshape(3, 3).T + p


def mesh_world_aabb(
    mesh: newton.Mesh, xform: wp.transform, scale: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """World-space (lower, upper) corners of a scaled, transformed mesh."""
    world = _transform_points(xform, _mesh_vertices(mesh) * np.asarray(scale, dtype=np.float64))
    return world.min(axis=0), world.max(axis=0)
