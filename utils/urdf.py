"""Static-shape URDF import."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

import newton


from utils.meshes import load_meshes, mesh_world_aabb
from utils.transforms import quat_from_rpy

Color = tuple[float, float, float]
ColorFn = Callable[[newton.Mesh], Color | None]
_IDENTITY = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
_JOINTED_TYPES = ("continuous", "revolute")


@dataclass
class StaticUrdf:
    """What :func:`add_urdf_as_static_shapes` added: shapes by label, plus any jointed links."""

    labels: dict[str, int] = field(default_factory=dict)
    bodies: list[int] = field(default_factory=list)
    joints: list[int] = field(default_factory=list)
    kept_visual_shapes: list[int] = field(default_factory=list)


def _parse_vec(text: str | None, default: Color) -> Color:
    if text is None:
        return default
    parts = [float(v) for v in text.replace(",", " ").split()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) != 3:
        raise ValueError(f"expected 3 numbers, got {text!r}")
    return (parts[0], parts[1], parts[2])


def _origin_xform(elem: ET.Element | None) -> wp.transform:
    """URDF <origin xyz rpy> -> wp.transform (rpy is the same Rz.Ry.Rx as Drake)."""
    origin = None if elem is None else elem.find("origin")
    if origin is None:
        return wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    xyz = _parse_vec(origin.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _parse_vec(origin.get("rpy"), (0.0, 0.0, 0.0))
    return wp.transform(wp.vec3(*xyz), quat_from_rpy(*rpy))


def _material_color(elem: ET.Element, fallback: Color) -> Color:
    material = elem.find("material")
    if material is not None:
        color = material.find("color")
        if color is not None and color.get("rgba"):
            rgba = [float(v) for v in color.get("rgba").split()]
            if len(rgba) >= 3:
                return (rgba[0], rgba[1], rgba[2])
    return fallback


def _inertial(link: ET.Element) -> tuple[float, wp.vec3, wp.mat33]:
    """``(mass, com, inertia about the com in the link frame)`` of a URDF ``<inertial>``."""
    inertial = link.find("inertial")
    if inertial is None:
        return 0.0, wp.vec3(0.0, 0.0, 0.0), wp.mat33(np.zeros((3, 3), dtype=np.float32))
    mass_elem = inertial.find("mass")
    mass = 0.0 if mass_elem is None else float(mass_elem.get("value", 0.0))
    X = _origin_xform(inertial)
    tensor = np.zeros((3, 3), dtype=np.float64)
    elem = inertial.find("inertia")
    if elem is not None:
        keys = (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))
        tensor = np.array([[float(elem.get(key, 0.0)) for key in row] for row in keys])
    R = np.array(wp.quat_to_matrix(wp.transform_get_rotation(X)), dtype=np.float64).reshape(3, 3)
    inertia = wp.mat33(*(R @ tensor @ R.T).reshape(-1).astype(np.float32))
    return mass, wp.transform_get_translation(X), inertia


def add_urdf_as_static_shapes(
    builder: newton.ModelBuilder,
    urdf_path: Path,
    X_world_root: wp.transform,
    *,
    label_prefix: str,
    visual_cfg: newton.ModelBuilder.ShapeConfig,
    collision_cfg: newton.ModelBuilder.ShapeConfig,
    visual_color: Color | None = None,
    collect_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    split_components: bool = False,
    color_fn: ColorFn | None = None,
    keep_visual_material: Sequence[str] = (),
) -> StaticUrdf:
    """Add every <visual>/<collision> of a fixed-only URDF as world shapes (body -1).

    These have to be plain world shapes rather than an ``add_urdf`` articulation so the
    VBD entry of the coupled solver can own them (exactly what ``round_belt.add_table``
    / ``add_board`` do), and so no zero-mass articulation ends up inside the MuJoCo
    entry. A link on a continuous/revolute joint to the root becomes a body on a world
    revolute joint (as ``round_belt.add_dynamic_pulley``) carrying that link's shapes.
    ``collect_aabbs`` optionally receives each mesh shape's world-space AABB (at q = 0),
    keyed by shape label.
    """
    urdf_path = Path(urdf_path)
    root = ET.parse(str(urdf_path)).getroot()
    urdf_dir = urdf_path.parent

    links = {link.get("name"): link for link in root.findall("link")}
    joints = list(root.findall("joint"))

    children = {joint.find("child").get("link") for joint in joints}
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f"{urdf_path.name}: expected exactly one root link, got {roots}")

    jointed: dict[str, ET.Element] = {}
    for joint in joints:
        jtype = joint.get("type")
        if jtype == "fixed":
            continue
        if jtype not in _JOINTED_TYPES or joint.find("parent").get("link") != roots[0]:
            raise ValueError(
                f"{urdf_path.name}: joint {joint.get('name')!r} has type {jtype!r}; "
                "add_urdf_as_static_shapes only supports fixed joints and "
                f"{'/'.join(_JOINTED_TYPES)} joints whose parent is the root link"
            )
        jointed[joint.find("child").get("link")] = joint

    # Chain the joint origins to get X_root_link (at q = 0) for every link.
    X_root_link = {roots[0]: wp.transform(_IDENTITY)}
    remaining = list(joints)
    while remaining:
        progressed = False
        for joint in list(remaining):
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent in X_root_link:
                X_root_link[child] = X_root_link[parent] * _origin_xform(joint)
                remaining.remove(joint)
                progressed = True
        if not progressed:
            raise ValueError(
                f"{urdf_path.name}: disconnected links {[j.get('name') for j in remaining]}"
            )
    for joint in joints:
        if joint.find("parent").get("link") in jointed:
            raise ValueError(
                f"{urdf_path.name}: joint {joint.get('name')!r} hangs off jointed link "
                f"{joint.find('parent').get('link')!r}; only root-parented joints are supported"
            )

    default_color = visual_color if visual_color is not None else (0.8, 0.8, 0.8)
    kept = set(keep_visual_material)
    result = StaticUrdf()
    shapes_by_link: dict[str, list[int]] = {}

    for link_name, link in links.items():
        X_wl = X_world_root * X_root_link[link_name]
        body, X_wb = -1, wp.transform(_IDENTITY)
        if link_name in jointed:
            joint = jointed[link_name]
            mass, com, inertia = _inertial(link)
            if mass <= 0.0:
                raise ValueError(
                    f"{urdf_path.name}: jointed link {link_name!r} needs a positive-mass <inertial>"
                )
            body = builder.add_link(
                xform=X_wl, com=com, inertia=inertia, mass=mass,
                label=f"{label_prefix}/{link_name}",
            )
            axis_elem = joint.find("axis")
            dynamics = joint.find("dynamics")
            limit = joint.find("limit")
            limits = {}
            if joint.get("type") == "revolute" and limit is not None:
                limits = {
                    key: float(limit.get(attr))
                    for key, attr in (("limit_lower", "lower"), ("limit_upper", "upper"))
                    if limit.get(attr) is not None
                }
            joint_index = builder.add_joint_revolute(
                parent=-1, child=body, parent_xform=X_wl, child_xform=wp.transform(_IDENTITY),
                axis=wp.vec3(*_parse_vec(
                    None if axis_elem is None else axis_elem.get("xyz"), (1.0, 0.0, 0.0)
                )),
                damping=0.0 if dynamics is None else float(dynamics.get("damping", 0.0)),
                label=f"{label_prefix}/{joint.get('name')}",
                **limits,
            )
            builder.add_articulation(
                [joint_index], label=f"{label_prefix}/{joint.get('name')}_articulation"
            )
            result.bodies.append(body)
            result.joints.append(joint_index)
            X_wb = X_wl
        link_shapes = shapes_by_link.setdefault(link_name, [])
        for kind, cfg in (("visual", visual_cfg), ("collision", collision_cfg)):
            is_visual = kind == "visual"
            for i, elem in enumerate(link.findall(kind)):
                geometry = elem.find("geometry")
                if geometry is None:
                    continue
                label = f"{label_prefix}/{link_name}/{kind}{i}"
                first = builder.shape_count
                X_elem = _origin_xform(elem)
                shape = _add_urdf_geometry(
                    builder,
                    geometry,
                    urdf_dir,
                    X_elem if body >= 0 else X_wl * X_elem,
                    cfg,
                    _material_color(elem, default_color) if is_visual else default_color,
                    label,
                    collect_aabbs,
                    body=body,
                    X_world_body=X_wb,
                    # Material/component splitting exists purely so parts can be coloured
                    # separately. Colliders must stay exactly as they were: splitting them
                    # yields degenerate pieces (small_round_pulley_half.obj's 27 mm cap is
                    # zero-thickness) and changes contact behaviour for no benefit.
                    per_material=is_visual,
                    split_components=split_components and is_visual,
                    color_fn=color_fn if is_visual else None,
                )
                added = list(range(first, builder.shape_count))
                link_shapes.extend(added)
                if is_visual and elem.get("name") in kept:
                    result.kept_visual_shapes.extend(added)
                if shape is not None:
                    result.labels[label] = shape

    # Jointed links sit in the root's geometry by design: filter them against the model's links.
    fixed_shapes = [s for name, ss in shapes_by_link.items() if name not in jointed for s in ss]
    jointed_links = list(jointed)
    for i, name in enumerate(jointed_links):
        others = fixed_shapes + [
            s for other in jointed_links[i + 1:] for s in shapes_by_link.get(other, [])
        ]
        for a in shapes_by_link.get(name, []):
            for b in others:
                builder.add_shape_collision_filter_pair(a, b)

    return result


def _mesh_color(mesh: newton.Mesh, color_fn: ColorFn | None, fallback: wp.vec3) -> wp.vec3:
    """Colour priority: caller override, white for a textured mesh (nothing may modulate
    a texture), the colour the asset authored, then the caller's ``visual_color``.

    Authored colours matter: table.gltf gives its frame white (1,1,1) and its base grey
    (0.85), which one ``visual_color`` would otherwise paint over in tabletop brown.
    """
    override = None if color_fn is None else color_fn(mesh)
    if override is not None:
        return wp.vec3(*(float(c) for c in override))
    if getattr(mesh, "texture", None) is not None:
        return wp.vec3(1.0, 1.0, 1.0)
    authored = getattr(mesh, "color", None)
    if authored is not None:
        return wp.vec3(*(float(c) for c in authored))
    return fallback


def _add_urdf_geometry(
    builder: newton.ModelBuilder,
    geometry: ET.Element,
    urdf_dir: Path,
    xform: wp.transform,
    cfg: newton.ModelBuilder.ShapeConfig,
    color: Sequence[float],
    label: str,
    collect_aabbs: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    *,
    body: int = -1,
    X_world_body: wp.transform = _IDENTITY,
    per_material: bool = True,
    split_components: bool = False,
    color_fn: ColorFn | None = None,
) -> int | None:
    wp_color = wp.vec3(float(color[0]), float(color[1]), float(color[2]))

    mesh_elem = geometry.find("mesh")
    if mesh_elem is not None:
        mesh_path = (urdf_dir / mesh_elem.get("filename")).resolve()
        if not mesh_path.exists():
            raise FileNotFoundError(f"{label}: missing mesh {mesh_path}")
        scale = _parse_vec(mesh_elem.get("scale"), (1.0, 1.0, 1.0))
        meshes = load_meshes(
            mesh_path, per_material=per_material, split_components=split_components
        )
        if collect_aabbs is not None:
            # Union over every material sub-mesh, so the AABB (and the floor height
            # derived from it) describes the whole visual, not just its first material.
            X_world_shape = X_world_body * xform
            lowers, uppers = zip(*(mesh_world_aabb(m, X_world_shape, scale) for m in meshes))
            collect_aabbs[label] = (
                np.min(np.stack(lowers), axis=0),
                np.max(np.stack(uppers), axis=0),
            )
        shapes = [
            builder.add_shape_mesh(
                body=body,
                xform=xform,
                mesh=mesh,
                scale=wp.vec3(*scale),
                cfg=cfg,
                color=_mesh_color(mesh, color_fn, wp_color),
                label=label if i == 0 else f"{label}_mat{i}",
            )
            for i, mesh in enumerate(meshes)
        ]
        return shapes[0] if shapes else None

    box_elem = geometry.find("box")
    if box_elem is not None:
        sx, sy, sz = _parse_vec(box_elem.get("size"), (1.0, 1.0, 1.0))
        return builder.add_shape_box(
            body=body, xform=xform, hx=0.5 * sx, hy=0.5 * sy, hz=0.5 * sz,
            cfg=cfg, color=wp_color, label=label,
        )

    sphere_elem = geometry.find("sphere")
    if sphere_elem is not None:
        return builder.add_shape_sphere(
            body=body, xform=xform, radius=float(sphere_elem.get("radius", 1.0)),
            cfg=cfg, color=wp_color, label=label,
        )

    cyl_elem = geometry.find("cylinder")
    if cyl_elem is not None:
        return builder.add_shape_cylinder(
            body=body, xform=xform, radius=float(cyl_elem.get("radius", 1.0)),
            half_height=0.5 * float(cyl_elem.get("length", 1.0)),
            cfg=cfg, color=wp_color, label=label,
        )

    raise ValueError(f"{label}: unsupported <geometry> child {[c.tag for c in geometry]}")
