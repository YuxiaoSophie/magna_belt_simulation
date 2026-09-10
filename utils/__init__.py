"""Task-agnostic Newton plumbing shared by the scene scripts.

Drake/Warp transform conventions, label lookups, mesh loading, static-URDF import
and the Newton viewer workarounds. Nothing here knows about a specific scene, and
nothing here may import a scene module.
"""

from __future__ import annotations

from utils.labels import (
    body_index,
    body_label_endswith,
    hide_shapes,
    joint_index,
    label_shapes_by_body,
)
from utils.meshes import (
    fix_inverted_mesh_winding,
    load_meshes,
    mesh_world_aabb,
    neutralize_textured_shape_colors,
)
from utils.transforms import drake_xform, rpy_deg_from_quat
from utils.urdf import add_urdf_as_static_shapes
from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material

__all__ = [
    "add_urdf_as_static_shapes",
    "body_index",
    "body_label_endswith",
    "drake_xform",
    "fix_inverted_mesh_winding",
    "hide_shapes",
    "joint_index",
    "label_shapes_by_body",
    "load_meshes",
    "mesh_world_aabb",
    "neutralize_textured_shape_colors",
    "patch_viewer_shape_names",
    "patch_viser_texture_material",
    "rpy_deg_from_quat",
]
