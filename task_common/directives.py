"""Extension directives every belt task shares, and the ``params`` key-validation helper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import warp as wp

from utils.directives import DirectiveContext, DirectiveFn, as_float, as_vec3

REQUIRED: Any = object()


_Num = Callable[..., float]


def params(
    kind: str, params: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[dict[str, Any], str, _Num]:
    """``params`` merged over ``spec``'s defaults (``REQUIRED`` = no default), a ``where``
    prefix for errors, and ``num(key, default=None)``: the value as a float (YAML 1.1 rule),
    or ``default`` when it is ``None``.  Unknown and missing keys raise ``ValueError``."""
    where = f"{kind} {params.get('name', spec.get('name', ''))!r}"
    unknown = sorted(str(key) for key in params if key not in spec)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(spec)}")
    merged = {**spec, **params}
    missing = sorted(key for key, value in merged.items() if value is REQUIRED)
    if missing:
        raise ValueError(f"{where}: missing required key(s) {missing}")

    def num(key: str, default: float | None = None) -> float:
        value = merged[key]
        if value is None and default is not None:
            return default
        return as_float(value, where, key)

    return merged, where, num


# The directives' own ``params`` argument shadows the helper.
_params = params


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
        "name": "tabletop_collision", "table_visual": REQUIRED, "top_z": REQUIRED,
        "thickness": REQUIRED, "color": REQUIRED,
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


COMMON_DIRECTIVES: dict[str, DirectiveFn] = {
    "add_tabletop_collision": add_tabletop_collision,
    "add_ground_plane": add_ground_plane,
}
