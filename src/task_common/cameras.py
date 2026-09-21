"""The scene's RGBD cameras, rendered with Newton's ``SensorTiledCamera``.

Images follow Drake's ``RgbdSensor``: row 0 is the top, depth is along the optical axis [m],
``0`` below ``z_near`` and ``inf`` beyond ``z_far`` or on a miss.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera


@dataclass(frozen=True)
class CameraSpec:
    """A world-fixed pinhole RGBD camera; the subset of Drake's ``CameraConfig`` we use."""

    name: str
    X_WC: wp.transform  # world <- OpenCV optical frame (+Z forward, +Y down)
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float
    z_near: float
    z_far: float
    fps: float


# Newton's camera frame is OpenGL (-Z forward, +Y up): a half turn about X from OpenCV's.
X_CV_GL = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(1.0, 0.0, 0.0, 0.0))


def _pad_rgb_textures(model: newton.Model) -> None:
    """SensorTiledCamera fails on 3-channel textures (Franka, table); make them opaque RGBA."""
    for shape in model.shape_source:
        texture = getattr(shape, "texture", None)
        if isinstance(texture, np.ndarray) and texture.ndim == 3 and texture.shape[2] == 3:
            alpha = np.full(texture.shape[:2] + (1,), 255, dtype=texture.dtype)
            shape.texture = np.concatenate([texture, alpha], axis=2)


class RgbdCameras:
    """All cameras of a scene, rendered in one sensor update."""

    def __init__(self, model: newton.Model, specs: Sequence[CameraSpec]) -> None:
        sizes = {(spec.width, spec.height) for spec in specs}
        if len(sizes) != 1:
            raise ValueError(f"cameras must share one resolution; got {sorted(sizes)}")
        width, height = sizes.pop()
        count = len(specs)

        self.model = model
        self.specs = list(specs)
        self.period = 1.0 / max(spec.fps for spec in specs)

        _pad_rgb_textures(model)
        config = SensorTiledCamera.RenderConfig(enable_shadows=True, enable_textures=True)
        self.sensor = SensorTiledCamera(model, default_render_config=config)
        utils = self.sensor.utils

        self.rays = wp.empty((count, height, width, 2), dtype=wp.vec3f, device=model.device)
        for index, spec in enumerate(specs):
            # Newton puts pixel (0, 0)'s centre at (0.5, 0.5); Drake and OpenCV at (0, 0).
            utils.compute_camera_rays_pinhole_opencv(
                width, height, spec.focal_x, spec.focal_y,
                spec.center_x + 0.5, spec.center_y + 0.5,
                out_rays=self.rays, camera_index=index,
            )
        self.transforms = wp.array(
            [[spec.X_WC * X_CV_GL] * model.world_count for spec in specs],
            dtype=wp.transformf, device=model.device,
        )
        self.color = utils.create_color_image_output(width, height, count)
        self.depth = utils.create_forward_depth_image_output(width, height, count)

    def update(self, state: newton.State) -> None:
        self.model.bvh_refit_shapes(state)
        self.model.bvh_refit_particles(state)
        self.sensor.update(
            state, self.transforms, self.rays,
            color_image=self.color, forward_depth_image=self.depth,
        )

    def rgbd(self, camera: int, world: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``(rgb uint8 HxWx3, depth float32 HxW)`` from the last :meth:`update`."""
        spec = self.specs[camera]
        rgba = self.color.numpy()[world, camera].view(np.uint8)
        rgb = rgba.reshape(spec.height, spec.width, 4)[..., :3].copy()
        depth = self.depth.numpy()[world, camera].copy()
        depth[(depth <= 0.0) | (depth > spec.z_far)] = np.inf
        depth[depth < spec.z_near] = 0.0
        return rgb, depth

    def log(self, viewer: newton.viewer.ViewerBase) -> None:
        utils = self.sensor.utils
        viewer.log_image("rgb", utils.to_rgba_from_color(self.color))
        viewer.log_image("depth", utils.to_rgba_from_depth(self.depth))
