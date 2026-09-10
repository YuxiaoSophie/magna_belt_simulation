"""Round-belt extension directives: the scene steps Drake's model directives cannot say.

Registered via ``load_directives(..., directives=EXTENSION_DIRECTIVES)`` and placed in order in
the YAML.  Each is a :data:`utils.directives.DirectiveFn` that validates its keys strictly
(unknown/missing -> ``ValueError``), coerces numbers with :func:`utils.directives.as_float`,
adds to ``ctx.builder`` and records its output in ``ctx.scene.extras[params["name"]]``.  (The
board's pulley-mount colour is the loader's ``component_colors``, not a directive.)  These
live here, not in ``utils/``, because they use ``round_belt.py``'s belt/contact constants and
the 2f85's labels.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from loguru import logger

import newton
import newton.utils

# Helpers/constants only; round_belt has a ``__main__`` guard, so this starts nothing.
import round_belt
from utils.directives import DirectiveContext, DirectiveFn, Vec3, as_float, as_floats, as_vec3
from utils.labels import hide_shapes
from utils.meshes import load_meshes

_REQUIRED: Any = object()
_IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


_Num = Callable[..., float]


def _params(
    kind: str, params: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[dict[str, Any], str, _Num]:
    """``params`` merged over ``spec``'s defaults (``_REQUIRED`` = no default), a ``where``
    prefix for errors, and ``num(key, default=None)``: the value as a float (YAML 1.1 rule),
    or ``default`` when it is ``None``.  Unknown and missing keys raise ``ValueError``."""
    where = f"{kind} {params.get('name', spec.get('name', ''))!r}"
    unknown = sorted(str(key) for key in params if key not in spec)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(spec)}")
    merged = {**spec, **params}
    missing = sorted(key for key, value in merged.items() if value is _REQUIRED)
    if missing:
        raise ValueError(f"{where}: missing required key(s) {missing}")

    def num(key: str, default: float | None = None) -> float:
        value = merged[key]
        if value is None and default is not None:
            return default
        return as_float(value, where, key)

    return merged, where, num


def add_tabletop_collision(ctx: DirectiveContext, params: Mapping[str, Any]) -> None:
    """Safety floor under the belt; NOT part of the Drake scene.

    The Drake table is visual-only, so without this the belt drops through it before it
    settles on the holder.  Adds one static box spanning the XY footprint of the
    ``table_visual`` shape's AABB, top face at ``top_z``, with ``ctx.collision_cfg``.

    Params: ``name`` (default ``"tabletop_collision"``; also the shape label),
    ``table_visual`` (a shape label in ``ctx.scene.aabbs``), ``top_z``, ``thickness``,
    ``color`` (rgb).  Stores ``extras[name] = {"shape": int, "aabb": (lo, hi)}``.
    """
    p, where, num = _params("add_tabletop_collision", params, {
        "name": "tabletop_collision", "table_visual": _REQUIRED, "top_z": _REQUIRED,
        "thickness": _REQUIRED, "color": _REQUIRED,
    })
    name, label = str(p["name"]), str(p["table_visual"])
    top_z, thickness = num("top_z"), num("thickness")
    color = as_vec3(p["color"], where, "color")
    table_aabb = ctx.scene.aabbs.get(label)
    if table_aabb is None:
        prefix = label.split("/", 1)[0] + "/"
        candidates = sorted(key for key in ctx.scene.aabbs if key.startswith(prefix))
        raise RuntimeError(
            f"{where}: table visual AABB {label!r} not found in the collected AABBs; "
            f"scene.urdf's visual labels changed. Available table labels: {candidates}"
        )
    center = 0.5 * (table_aabb[0] + table_aabb[1])
    half = 0.5 * (table_aabb[1] - table_aabb[0])
    shape = ctx.builder.add_shape_box(
        body=-1,
        xform=wp.transform(
            wp.vec3(float(center[0]), float(center[1]), top_z - 0.5 * thickness),
            wp.quat_identity(),
        ),
        hx=float(half[0]), hy=float(half[1]), hz=0.5 * thickness,
        cfg=ctx.collision_cfg, color=wp.vec3(*color), label=name,
    )
    ctx.scene.extras[name] = {"shape": int(shape), "aabb": (table_aabb[0], table_aabb[1])}


def add_ground_plane(ctx: DirectiveContext, params: Mapping[str, Any]) -> None:
    """``builder.add_ground_plane(height=h)``.

    Params: ``name`` (default ``"ground"``) and exactly one of ``height`` or
    ``height_from_aabb_min_z_of`` (a shape label in ``ctx.scene.aabbs``; ``h`` is its AABB
    min z).  Stores ``extras[name] = {"shape": int, "height": float}``.
    """
    p, where, num = _params("add_ground_plane", params, {
        "name": "ground", "height": None, "height_from_aabb_min_z_of": None,
    })
    name = str(p["name"])
    if (p["height"] is None) == (p["height_from_aabb_min_z_of"] is None):
        raise ValueError(f"{where}: give exactly one of 'height' or 'height_from_aabb_min_z_of'")
    if p["height"] is not None:
        height = num("height")
    else:
        label = str(p["height_from_aabb_min_z_of"])
        aabb = ctx.scene.aabbs.get(label)
        if aabb is None:
            raise KeyError(f"{where}: no collected AABB {label!r}; have {sorted(ctx.scene.aabbs)}")
        height = float(aabb[0][2])
    shape = ctx.builder.add_ground_plane(height=height)
    ctx.scene.extras[name] = {"shape": int(shape), "height": height}


_ROD_SPEC: dict[str, Any] = {
    "name": _REQUIRED, "center": _REQUIRED, "semi_axes": _REQUIRED, "radius": _REQUIRED,
    "num_elements": _REQUIRED, "color": _REQUIRED, "stretch_stiffness": _REQUIRED,
    "stretch_damping": _REQUIRED, "bend_stiffness": _REQUIRED, "bend_damping": _REQUIRED,
    "twist_total": 0.0, "closed": True, "body_frame_origin": "com", "margin": 0.0,
    "gap": 0.001, "density": None, "ke": None, "kd": None, "mu": None,
}


def add_rod_ellipse(ctx: DirectiveContext, params: Mapping[str, Any]) -> None:
    """A VBD rod (cable) on an axis-aligned ellipse in a horizontal plane.

    ``num_elements + 1`` points at ``center + (a cos t, b sin t, 0)``, parallel-transport
    edge quaternions, then ``builder.add_rod``.

    Params: ``name`` (the rod label), ``center`` (3), ``semi_axes`` (``[a, b]`` along world
    X, Y), ``radius``, ``num_elements`` (int), ``color``, ``stretch_stiffness``,
    ``stretch_damping``, ``bend_stiffness``, ``bend_damping``; optional ``twist_total``
    (0.0), ``closed`` (true), ``body_frame_origin`` ("com"), ``margin`` (0.0), ``gap``
    (0.001), ``density`` (``round_belt._estimate_belt_density()``) and ``ke``/``kd``/``mu``
    (``round_belt.CABLE_CONTACT_*``).  Stores
    ``extras[name] = {"bodies": [...], "joints": [...], "shapes": [...]}``.
    """
    p, where, num = _params("add_rod_ellipse", params, _ROD_SPEC)
    builder, name = ctx.builder, str(p["name"])
    count = p["num_elements"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        raise ValueError(f"{where}: num_elements must be an integer >= 3, got {count!r}")
    if not isinstance(p["closed"], bool):
        raise ValueError(f"{where}: closed must be true or false, got {p['closed']!r}")
    center = np.asarray(as_vec3(p["center"], where, "center"), dtype=np.float64)
    semi_x, semi_y = as_floats(p["semi_axes"], where, "semi_axes", 2)
    body_start, joint_start, shape_start = (
        builder.body_count, builder.joint_count, builder.shape_count
    )
    points = []
    for i in range(count + 1):
        theta = 2.0 * np.pi * i / count
        points.append(wp.vec3(
            float(center[0] + semi_x * np.cos(theta)),
            float(center[1] + semi_y * np.sin(theta)),
            float(center[2]),
        ))
    edge_q = newton.utils.create_parallel_transport_cable_quaternions(
        points, twist_total=num("twist_total")
    )
    bodies, _joints = builder.add_rod(
        positions=points, quaternions=edge_q, radius=num("radius"),
        cfg=newton.ModelBuilder.ShapeConfig(
            density=num("density", round_belt._estimate_belt_density()),
            ke=num("ke", round_belt.CABLE_CONTACT_KE), kd=num("kd", round_belt.CABLE_CONTACT_KD),
            mu=num("mu", round_belt.CABLE_CONTACT_MU), margin=num("margin"), gap=num("gap")),
        stretch_stiffness=num("stretch_stiffness"), stretch_damping=num("stretch_damping"),
        bend_stiffness=num("bend_stiffness"), bend_damping=num("bend_damping"),
        closed=p["closed"], body_frame_origin=str(p["body_frame_origin"]),
        label=name, color=as_vec3(p["color"], where, "color"),
    )
    if list(bodies) != list(range(body_start, builder.body_count)):
        raise RuntimeError(
            f"{where}: add_rod bodies {list(bodies)} are not the contiguous range "
            f"[{body_start}, {builder.body_count})"
        )
    ctx.scene.extras[name] = {
        "bodies": list(bodies),
        "joints": list(range(joint_start, builder.joint_count)),
        "shapes": list(range(shape_start, builder.shape_count)),
    }


def _add_finger_meshes(
    ctx: DirectiveContext, gripper: str, root_body: int, pad_bodies: Sequence[int],
    finger_dir: Path, offset_x: float, offset_z: float, base_mount_offset_z: float,
    color: Vec3,
) -> list[int]:
    """Attach the ALOHA-style Robotiq finger meshes to the gripper's pad bodies.

    The fingers are placed at the Drake SDF's ``left_finger``/``right_finger`` offsets
    (``offset_x``/``offset_z``, the right one yawed by pi) in the Drake gripper base frame --
    which is where the MJCF is welded, less the MJCF root's own ``base_mount_offset_z``.
    Each finger rides whichever pad body it is nearest so it tracks the gripper as the pads
    close; the 2f85 linkage rotates its pads where Drake's fingers translate, so the match
    is exact only at the configuration used here.  That is the point: these are the visible,
    contacting geometry, not a re-articulated gripper.  Returns the collider shapes, i.e.
    what GRIPPER_CONTACT_MU/KE/KD applies to.
    """
    builder = ctx.builder
    identity = wp.quat_identity()
    X_W_root = wp.transform(*builder.body_q[root_body])
    # Undo the MJCF's own base_mount offset to land on the Drake gripper base frame.
    X_W_base = X_W_root * wp.transform(wp.vec3(0.0, 0.0, -base_mount_offset_z), identity)

    yaw_pi = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)
    placements = [("left_finger", +offset_x, identity), ("right_finger", -offset_x, yaw_pi)]

    shapes: list[int] = []
    remaining = list(pad_bodies)
    for mesh_name, finger_x, rotation in placements:
        X_W_finger = X_W_base * wp.transform(wp.vec3(float(finger_x), 0.0, offset_z), rotation)
        finger_position = np.array([X_W_finger.p[0], X_W_finger.p[1], X_W_finger.p[2]])
        # Pair each finger with the pad it actually sits on rather than assuming an order.
        pad = min(
            remaining,
            key=lambda body: float(
                np.linalg.norm(np.array(builder.body_q[body][:3]) - finger_position)
            ),
        )
        remaining.remove(pad)
        X_pad_finger = wp.transform_inverse(wp.transform(*builder.body_q[pad])) * X_W_finger
        mesh = load_meshes(finger_dir / f"{mesh_name}.obj", per_material=False)[0]
        # The Drake SDF declares a <visual> and a <collision> from the same mesh, so mirror
        # that: one visible non-colliding shape plus one hidden collider.  One shape
        # carrying both flags is physically equivalent but collapses into a single viewer
        # batch, leaving no collision entry for --show-collision to toggle.
        builder.add_shape_mesh(
            body=pad, xform=X_pad_finger, mesh=mesh, cfg=ctx.visual_cfg,
            color=wp.vec3(*color), label=f"{gripper}/aloha_finger/{mesh_name}/visual",
        )
        shapes.append(
            builder.add_shape_mesh(
                body=pad, xform=X_pad_finger, mesh=mesh, cfg=ctx.collision_cfg,
                color=wp.vec3(*color), label=f"{gripper}/aloha_finger/{mesh_name}/collision",
            )
        )
    return shapes


def _body_poses_populated(builder: newton.ModelBuilder, bodies: Sequence[int]) -> bool:
    """False when every body sits at identity in ``builder.body_q`` -- what ``add_urdf``
    leaves behind, since it does no build-time forward kinematics (MJCF/USD do)."""
    return any(
        not np.allclose(
            [float(v) for v in builder.body_q[b]], _IDENTITY_POSE, atol=1.0e-9, rtol=0.0
        )
        for b in bodies
    )


def add_aloha_fingers(ctx: DirectiveContext, params: Mapping[str, Any]) -> None:
    """Replace a 2f85's pad geometry with the Drake ALOHA-style finger meshes.

    Selects the gripper's two pad bodies (``round_belt._select_gripper_proxy_bodies``),
    swaps their colliders for ``round_belt``'s two flat planes, attaches the finger meshes
    (visual + hidden collider each), then hides the 2f85 pad visuals and those planes: the
    fingers carry collision, so the gripper keeps its belt contact geometry.  Must directly
    follow the gripper's ``add_model``/``add_weld``; afterwards the gripper's
    ``shape_end`` is extended over the new shapes so they count as gripper shapes.  The
    fingers are placed from the gripper's build-time ``builder.body_q``, so the gripper must
    come from an importer that populates it (MJCF/USD); a URDF gripper raises ``ValueError``.

    Params: ``name`` (default ``"aloha_fingers"``), ``gripper`` (model name),
    ``finger_dir`` (dir with ``left_finger.obj``/``right_finger.obj``), ``offset_x``,
    ``offset_z``, ``base_mount_offset_z``, ``color``.  Stores
    ``extras[name] = {"pad_bodies": [...], "pad_shapes": [<finger collider shapes>]}``.
    """
    p, where, num = _params("add_aloha_fingers", params, {
        "name": "aloha_fingers", "gripper": _REQUIRED, "finger_dir": _REQUIRED,
        "offset_x": _REQUIRED, "offset_z": _REQUIRED, "base_mount_offset_z": _REQUIRED,
        "color": _REQUIRED,
    })
    builder, name, gripper = ctx.builder, str(p["name"]), str(p["gripper"])
    record = ctx.scene.models.get(gripper)
    if record is None:
        raise ValueError(
            f"{where}: gripper {gripper!r} is not a loaded model (have {list(ctx.scene.models)})"
        )
    if builder.shape_count != record.shape_end:
        raise ValueError(
            f"{where}: must directly follow {gripper!r}'s add_model/add_weld (the builder has "
            f"{builder.shape_count} shapes but the gripper's range ends at {record.shape_end}), "
            "or the finger shapes fall outside the gripper's shape range"
        )
    if not _body_poses_populated(builder, record.bodies):
        raise ValueError(
            f"{where}: every body of {gripper!r} is at identity in builder.body_q, so its "
            "importer did no build-time forward kinematics (add_urdf does not; add_mjcf and "
            "add_usd do). The fingers are placed from those poses and would land at the world "
            "origin; use an MJCF or USD gripper"
        )
    finger_dir = ctx.resolve_path(str(p["finger_dir"]))

    pad_bodies = round_belt._select_gripper_proxy_bodies(
        builder, record.body_start, record.body_end
    )
    if len(pad_bodies) != 2:
        raise RuntimeError(
            f"Two-plane gripper contact requires exactly 2 pad bodies; got {pad_bodies}"
        )
    round_belt._replace_proxy_pad_colliders_with_two_planes(builder, pad_bodies)

    finger_shapes = _add_finger_meshes(
        ctx, gripper, record.body_start, pad_bodies, finger_dir, num("offset_x"),
        num("offset_z"), num("base_mount_offset_z"), as_vec3(p["color"], where, "color"),
    )
    dropped_pad_visuals = hide_shapes(
        builder, lambda label: label.endswith("pad_geom_0_visual") or "silicone_pad_geom" in label
    )
    dropped_planes = hide_shapes(builder, lambda label: label.startswith("simple_belt_contact"))
    logger.info(
        f"ALOHA fingers: added {len(finger_shapes)} finger meshes; "
        f"dropped {dropped_pad_visuals} 2f85 pad visuals and {dropped_planes} "
        f"simple_belt_contact planes."
    )
    record.shape_end = builder.shape_count
    ctx.scene.extras[name] = {
        "pad_bodies": [int(b) for b in pad_bodies],
        "pad_shapes": [int(s) for s in finger_shapes],
    }


EXTENSION_DIRECTIVES: dict[str, DirectiveFn] = {
    "add_tabletop_collision": add_tabletop_collision,
    "add_ground_plane": add_ground_plane,
    "add_rod_ellipse": add_rod_ellipse,
    "add_aloha_fingers": add_aloha_fingers,
}
