"""Runtime workarounds for the Newton viewers.

Each patch fixes a Newton/viser bug in the installed viewer at runtime rather than
by editing the submodule.
"""

from __future__ import annotations

import os.path

import numpy as np

import newton


def _is_viser_viewer(viewer: object) -> bool:
    """True when the viewer is a ViewerViser (import guarded: viser is optional)."""
    try:
        from newton.viewer import ViewerViser
    except Exception:
        return False
    return isinstance(viewer, ViewerViser)


def _batch_name(names: list[str]) -> str:
    """Name a multi-shape render batch after what its member labels share."""
    # Longest common "/"-separated prefix.
    parts = names[0].split("/")
    for other in names[1:]:
        other_parts = other.split("/")
        keep = 0
        while keep < min(len(parts), len(other_parts)) and parts[keep] == other_parts[keep]:
            keep += 1
        parts = parts[:keep]
    stem = "/".join(parts)
    if not stem and any("/" in n for n in names):
        # Hierarchical labels sharing no first segment: one batch spanning two model
        # roots (Newton instances identical collision primitives across the panda_arm
        # and panda_hand URDFs). A common character prefix would invent a "panda"
        # parent that is not a real node, so name it for what it spans.
        roots = sorted({n.split("/")[0] for n in names if n})
        stem = "+".join(roots) if len(roots) <= 3 else f"{roots[0]}+{len(roots) - 1}_more"
    if not stem:
        # Flat labels ("flexible_ellipse_cable_edge_capsule_7"): fall back to the longest
        # common CHARACTER prefix. Trailing digits belong to the per-element index, and a
        # shared leading digit (capsule_21 / capsule_26 -> "..._2") would otherwise split
        # one part across several sibling nodes.
        stem = os.path.commonprefix(names).rstrip("_-/")
        stem = stem.rstrip("0123456789").rstrip("_-/")
    # Name the child by the leaf segments' common prefix (visual0/visual1 -> "visual"),
    # so a batch reads as ".../collision_x2" rather than a bare ".../x2"; fall back to
    # their common suffix, which is what separates mirrored pairs (right_driver_geom_0_-
    # visual / left_driver_geom_0_visual share no prefix but a long suffix).
    leaves = [n.split("/")[-1] for n in names]
    leaf = os.path.commonprefix(leaves).rstrip("_-").rstrip("0123456789").rstrip("_-")
    if not leaf:
        shared = os.path.commonprefix([x[::-1] for x in leaves])[::-1]
        # The shared suffix can start mid-word -- "right_coupler..." and "left_coupler..."
        # share "t_coupler...", the t being the tail of righT/lefT. Only keep it if it
        # begins on a separator in every member; otherwise drop the partial leading word.
        on_boundary = all(len(x) == len(shared) or x[-len(shared) - 1] in "_-/" for x in leaves)
        if shared and not on_boundary:
            cuts = [i for i in (shared.find("_"), shared.find("-")) if i >= 0]
            shared = shared[min(cuts) + 1 :] if cuts else ""
        leaf = shared.lstrip("_-")
    # Hang the count off as a child segment rather than gluing it to the stem, so every
    # batch of one part nests under a single tree node: one collapsible
    # "flexible_ellipse_cable_edge_capsule" holding the belt's 18 capsule-length batches.
    stem = stem or "shapes"
    if leaf and not stem.endswith(leaf):
        return f"{stem}/{leaf}_x{len(names)}"
    return f"{stem}/x{len(names)}"


def patch_viewer_shape_names() -> bool:
    """Name the viewer's shape batches after ``Model.shape_label`` instead of an index.

    ``ViewerBase._populate_shapes`` names each instanced batch ``/model/shapes/shape_<n>``
    from its creation order (``viewer.py:2531``) and never consults the model's real
    labels, so every node in viser's Scene tree reads as an anonymous ``shape_N``. Each
    batch does record its member shapes (``ShapeInstances.model_shapes``), so they can be
    renamed afterwards; labels contain ``/``, which viser reads as tree hierarchy.
    """
    try:
        from newton._src.viewer.viewer import ViewerBase
    except Exception:
        return False

    if getattr(ViewerBase, "_shape_names_patched", False):
        return True

    def _sanitize(text: str) -> str:
        cleaned = "".join(c if (c.isalnum() or c in "_-/+") else "_" for c in text)
        return "/".join(part for part in cleaned.split("/") if part) or "shape"

    original = ViewerBase._populate_shapes

    visible_flag = int(newton.ShapeFlags.VISIBLE)
    collide_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)

    def _populate_shapes(self) -> None:
        original(self)
        labels = getattr(getattr(self, "model", None), "shape_label", None)
        if not labels:
            return
        used: set[str] = set()
        for batch in self._shape_instances.values():
            indices = [i for i in getattr(batch, "model_shapes", []) if 0 <= i < len(labels)]
            if not indices:
                continue
            names = [str(labels[i]) for i in indices]
            name = _sanitize(names[0] if len(names) == 1 else _batch_name(names))
            # Split the tree into /visual and /collision so the two can be collapsed and
            # inspected independently, classifying each batch by its flags.
            flags = int(getattr(batch, "flags", 0))
            group = "visual" if flags & visible_flag else ("collision" if flags & collide_flag else "other")
            name = f"{group}/{name}"
            unique, suffix = name, 2
            while unique in used:
                unique = f"{name}_{suffix}"
                suffix += 1
            used.add(unique)
            # Put /visual and /collision at the tree root for viser, where the
            # "/model/shapes" level is pure noise. The GL viewer keeps it: it finds its
            # stale batches with startswith("/model/shapes/") when rebuilding opacity
            # groups (viewer_gl.py:923), and dropping the prefix would leak them.
            root = "" if _is_viser_viewer(self) else "/model/shapes"
            batch.name = self._qualify(f"{root}/{unique}")

    ViewerBase._populate_shapes = _populate_shapes
    ViewerBase._shape_names_patched = True
    return True


def patch_viser_texture_material() -> bool:
    """Stop the viser viewer from dimming textured meshes to 40% brightness.

    ``ViewerViser._build_trimesh_mesh`` passes ``image=`` to ``TextureVisuals``, so trimesh
    attaches a ``SimpleMaterial`` (default diffuse 0.4 grey) and its GLB exporter writes
    ``baseColorFactor = [0.4, 0.4, 0.4, 1.0]``; glTF shades ``baseColor = baseColorFactor *
    baseColorTexture``, so the Franka arrives dark. An explicit ``PBRMaterial`` exports
    ``[1, 1, 1, 1]``. Viser-side sibling of ``neutralize_textured_shape_colors``.
    """
    try:
        from newton.viewer import ViewerViser
    except Exception:
        return False

    if getattr(ViewerViser, "_texture_material_patched", False):
        return True

    @staticmethod
    def _build_trimesh_mesh(points, indices, uvs, texture):
        try:
            import trimesh
            from PIL import Image
            from trimesh.visual.material import PBRMaterial
            from trimesh.visual.texture import TextureVisuals
        except Exception:
            return None
        if len(uvs) != len(points):
            return None
        mesh = trimesh.Trimesh(vertices=points, faces=indices.astype(np.int64), process=False)
        image = texture if isinstance(texture, Image.Image) else Image.fromarray(texture)
        mesh.visual = TextureVisuals(
            uv=uvs,
            material=PBRMaterial(
                baseColorTexture=image,
                baseColorFactor=[255, 255, 255, 255],  # do not modulate the texture
                metallicFactor=0.0,
                roughnessFactor=0.8,
            ),
        )
        return mesh

    ViewerViser._build_trimesh_mesh = _build_trimesh_mesh

    # Textured meshes reach the browser as GLB via add_batched_meshes_trimesh, whose
    # BatchedGlbHandle has no batched_opacities (viser _scene_api.py). Newton pushes an
    # opacity array every frame regardless and warns once per handle, a dozen lines of
    # noise for a fully opaque scene. Drop the array when it asks for nothing -- every
    # value is 1.0 -- and leave the warning intact when opacity is real, since then
    # viser genuinely cannot honour it.
    original_log_instances = ViewerViser.log_instances

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False, opacities=None):
        if opacities is not None:
            try:
                mesh_data = self._meshes.get(self._qualify(mesh), {})
                if mesh_data.get("trimesh") is not None:
                    values = opacities.numpy() if hasattr(opacities, "numpy") else np.asarray(opacities)
                    if values.size == 0 or float(np.min(values)) >= 1.0 - 1e-6:
                        opacities = None
            except Exception:
                pass  # never let an appearance tweak break rendering
        return original_log_instances(
            self, name, mesh, xforms, scales, colors, materials, hidden=hidden, opacities=opacities
        )

    ViewerViser.log_instances = log_instances
    ViewerViser._texture_material_patched = True
    return True
