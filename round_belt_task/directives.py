"""Round-belt extension directives: the scene steps Drake's model directives cannot say.

Registered via ``load_directives(..., directives=EXTENSION_DIRECTIVES)`` and placed in order in
the YAML.  Each is a :data:`utils.directives.DirectiveFn` that validates its keys strictly
(unknown/missing -> ``ValueError``), coerces numbers with :func:`utils.directives.as_float`,
adds to ``ctx.builder`` and records its output in ``ctx.scene.extras[params["name"]]``.  (The
board's pulley-mount colour is the loader's ``component_colors``, not a directive.)  These
live here, not in ``utils/``, because they use ``round_belt.py``'s belt/contact constants.
(The 2f85's ALOHA fingers are not a directive: they are geoms baked into ``2f85.xml``.)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.utils

# Helpers/constants only; round_belt has a ``__main__`` guard, so this starts nothing.
import round_belt
from utils.directives import DirectiveContext, DirectiveFn, as_float, as_floats, as_vec3

_REQUIRED: Any = object()


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


EXTENSION_DIRECTIVES: dict[str, DirectiveFn] = {
    "add_tabletop_collision": add_tabletop_collision,
    "add_ground_plane": add_ground_plane,
    "add_rod_ellipse": add_rod_ellipse,
}
