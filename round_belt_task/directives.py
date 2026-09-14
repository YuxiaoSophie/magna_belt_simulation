"""Round-belt extension directives: the scene steps Drake's model directives cannot say.

Registered via ``load_directives(..., directives=EXTENSION_DIRECTIVES)`` and placed in order in
the YAML.  Each is a :data:`utils.directives.DirectiveFn` that validates its keys strictly
(unknown/missing -> ``ValueError``), coerces numbers with :func:`utils.directives.as_float`,
adds to ``ctx.builder`` and records its output in ``ctx.scene.extras[params["name"]]``.  (The
board's pulley-mount colour is the loader's ``component_colors``, not a directive.)  The
tabletop and ground directives are shared, from :mod:`task_common.directives`.
(The 2f85's ALOHA fingers are not a directive: they are geoms baked into ``2f85.xml``.)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.utils

# Helpers/constants only; round_belt has a ``__main__`` guard, so this starts nothing.
import round_belt
from task_common.directives import COMMON_DIRECTIVES, REQUIRED
from task_common.directives import params as _params
from utils.directives import DirectiveContext, DirectiveFn, as_floats, as_vec3


_ROD_SPEC: dict[str, Any] = {
    "name": REQUIRED, "center": REQUIRED, "semi_axes": REQUIRED, "radius": REQUIRED,
    "num_elements": REQUIRED, "color": REQUIRED, "stretch_stiffness": REQUIRED,
    "stretch_damping": REQUIRED, "bend_stiffness": REQUIRED, "bend_damping": REQUIRED,
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
    **COMMON_DIRECTIVES, "add_rod_ellipse": add_rod_ellipse,
}
