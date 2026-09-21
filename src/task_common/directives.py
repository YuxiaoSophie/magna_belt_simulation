"""Extension directives every belt task shares, and the ``params`` key-validation helper."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import warp as wp

from task_common.cameras import CameraSpec
from task_common.point_cloud import PointCloudSpec
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


def add_rgbd_camera(ctx: DirectiveContext, params: Mapping[str, Any]) -> None:
    """A world-fixed pinhole RGBD camera, with Drake's ``CameraConfig`` defaults.

    ``base_frame`` is the optical frame (+Z forward, +Y down): ``world``, an ``add_frame`` name
    or a static model's weld child.  Focal length is ``focal_x``/``focal_y`` [px] or
    ``fov_y_deg``.  Adds nothing to the builder.  Stores ``extras[name] = CameraSpec``.
    """
    p, where, num = _params("add_rgbd_camera", params, {
        "name": REQUIRED, "base_frame": REQUIRED, "width": 640, "height": 480, "fps": 20.0,
        "fov_y_deg": None, "focal_x": None, "focal_y": None, "center_x": None,
        "center_y": None, "z_near": 0.1, "z_far": 5.0,
    })
    name, ref = str(p["name"]), str(p["base_frame"])
    width, height = int(num("width")), int(num("height"))
    focal_given = [p[key] is not None for key in ("focal_x", "focal_y")]
    if focal_given == [False, False]:
        fov_y = math.radians(num("fov_y_deg", 45.0))
        focal_x = focal_y = 0.5 * height / math.tan(0.5 * fov_y)
    elif focal_given == [True, True] and p["fov_y_deg"] is None:
        focal_x, focal_y = num("focal_x"), num("focal_y")
    else:
        raise ValueError(f"{where}: give both 'focal_x' and 'focal_y', or 'fov_y_deg'")
    center_x = num("center_x", 0.5 * (width - 1))
    center_y = num("center_y", 0.5 * (height - 1))

    model_name = ref.split("::", 1)[0]
    record = ctx.scene.models.get(model_name)
    if "::" in ref and record is not None and record.directive.kind == "static":
        # A static model has no bodies; its weld child sits at the weld pose.
        try:
            ctx.scene.directives.weld(ref)
        except KeyError as error:
            raise ValueError(f"{where}: {ref!r} is not {model_name!r}'s weld child") from error
        X_WC = wp.transform(record.xform)
    else:
        body, X_WC = ctx.scene.frame_transform(ref)
        if body != -1:
            raise ValueError(f"{where}: base_frame {ref!r} is not world-fixed (body {body})")
    ctx.scene.extras[name] = CameraSpec(
        name=name, X_WC=X_WC, width=width, height=height, focal_x=focal_x, focal_y=focal_y,
        center_x=center_x, center_y=center_y, z_near=num("z_near"), z_far=num("z_far"),
        fps=num("fps"),
    )


def add_cropped_point_cloud(ctx: DirectiveContext, params: Mapping[str, Any]) -> None:
    """Merged world cloud of ``cameras``, cropped to an AABB and voxel-downsampled.

    ``cameras`` names earlier ``add_rgbd_camera`` entries; ``voxel_size`` 0 disables the
    downsample.  Adds nothing to the builder.  Stores ``extras[name] = PointCloudSpec``.
    """
    p, where, num = _params("add_cropped_point_cloud", params, {
        "name": REQUIRED, "cameras": REQUIRED, "crop_lower_xyz": REQUIRED,
        "crop_upper_xyz": REQUIRED, "voxel_size": 0.0,
    })
    cameras = p["cameras"]
    if isinstance(cameras, str) or not cameras:
        raise ValueError(f"{where}: 'cameras' must be a non-empty list of camera names")
    cameras = tuple(str(camera) for camera in cameras)
    unknown = [c for c in cameras if not isinstance(ctx.scene.extras.get(c), CameraSpec)]
    if unknown:
        raise ValueError(f"{where}: {unknown} are not add_rgbd_camera entries defined above it")
    lower = as_vec3(p["crop_lower_xyz"], where, "crop_lower_xyz")
    upper = as_vec3(p["crop_upper_xyz"], where, "crop_upper_xyz")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError(f"{where}: crop_lower_xyz {lower} exceeds crop_upper_xyz {upper}")
    voxel_size = num("voxel_size")
    if voxel_size < 0.0:
        raise ValueError(f"{where}: voxel_size must be >= 0, got {voxel_size}")
    name = str(p["name"])
    ctx.scene.extras[name] = PointCloudSpec(
        name=name, cameras=cameras, crop_lower_xyz=lower, crop_upper_xyz=upper,
        voxel_size=voxel_size,
    )


COMMON_DIRECTIVES: dict[str, DirectiveFn] = {
    "add_tabletop_collision": add_tabletop_collision,
    "add_ground_plane": add_ground_plane,
    "add_rgbd_camera": add_rgbd_camera,
    "add_cropped_point_cloud": add_cropped_point_cloud,
}
