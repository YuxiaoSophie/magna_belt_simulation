"""Merged, cropped, voxel-downsampled world point cloud from the scene's RGBD cameras.

Port of magna's ``AddRoundBeltCroppedPointCloudLcm``: Drake's ``DepthImageToPointCloud`` per
camera, then ``Concatenate``, ``Crop`` and ``VoxelizedDownSample``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from task_common.cameras import RgbdCameras


@dataclass(frozen=True)
class PointCloudSpec:
    name: str
    cameras: tuple[str, ...]
    crop_lower_xyz: tuple[float, float, float]
    crop_upper_xyz: tuple[float, float, float]
    voxel_size: float  # 0 disables the downsample


@wp.kernel(enable_backward=False)
def _depth_to_cropped_world_points(
    depth: wp.array4d[wp.float32],
    world: int,
    cameras: wp.array[wp.int32],
    focal: wp.array[wp.vec2f],
    center: wp.array[wp.vec2f],
    z_range: wp.array[wp.vec2f],
    X_WC: wp.array[wp.transformf],
    lower: wp.vec3f,
    upper: wp.vec3f,
    points: wp.array3d[wp.vec3f],
    keep: wp.array3d[wp.uint8],
):
    i, v, u = wp.tid()
    camera = cameras[i]
    z = depth[world, camera, v, u]  # a miss is 0, below z_near
    keep[i, v, u] = wp.uint8(0)
    if z < z_range[camera][0] or z > z_range[camera][1]:
        return
    p_C = wp.vec3f(
        (float(u) - center[camera][0]) * z / focal[camera][0],
        (float(v) - center[camera][1]) * z / focal[camera][1],
        z,
    )
    p_W = wp.transform_point(X_WC[camera], p_C)
    for k in range(3):
        if p_W[k] < lower[k] or p_W[k] > upper[k]:
            return
    points[i, v, u] = p_W
    keep[i, v, u] = wp.uint8(1)


def voxelized_down_sample(
    xyz: np.ndarray, rgb: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Drake's ``VoxelizedDownSample``: per world-aligned voxel, the centroid and the mean
    colour truncated to ``uint8``."""
    if len(xyz) == 0:
        return xyz, rgb
    keys = np.floor(xyz.astype(np.float64) / voxel_size).astype(np.int64)
    _, voxel, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    voxel = voxel.reshape(-1)

    def mean(values: np.ndarray) -> np.ndarray:
        sums = [np.bincount(voxel, weights=values[:, k].astype(np.float64)) for k in range(3)]
        return np.stack(sums, axis=1) / counts[:, None]

    return mean(xyz).astype(np.float32), mean(rgb).astype(np.uint8)


class CroppedPointCloud:
    """One :class:`PointCloudSpec`, evaluated on the images of an :class:`RgbdCameras`."""

    def __init__(self, cameras: RgbdCameras, spec: PointCloudSpec) -> None:
        names = [camera.name for camera in cameras.specs]
        missing = sorted(set(spec.cameras) - set(names))
        if missing:
            raise ValueError(f"point cloud {spec.name!r}: unknown cameras {missing}; have {names}")
        self.cameras = cameras
        self.spec = spec
        self.indices = [names.index(name) for name in spec.cameras]

        specs, device = cameras.specs, cameras.model.device
        shape = (len(self.indices), specs[0].height, specs[0].width)

        def array(values, dtype):
            return wp.array(values, dtype=dtype, device=device)

        self._inputs = [
            array(self.indices, wp.int32),
            array([(s.focal_x, s.focal_y) for s in specs], wp.vec2f),
            array([(s.center_x, s.center_y) for s in specs], wp.vec2f),
            array([(s.z_near, s.z_far) for s in specs], wp.vec2f),
            array([s.X_WC for s in specs], wp.transformf),
            wp.vec3f(*spec.crop_lower_xyz),
            wp.vec3f(*spec.crop_upper_xyz),
        ]
        self._points = wp.zeros(shape, dtype=wp.vec3f, device=device)
        self._keep = wp.zeros(shape, dtype=wp.uint8, device=device)

    def compute(self, world: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``(xyz float32 Nx3 [m, world], rgb uint8 Nx3)`` from the last camera update."""
        wp.launch(
            _depth_to_cropped_world_points, dim=self._points.shape,
            inputs=[self.cameras.depth, world, *self._inputs, self._points, self._keep],
            device=self._points.device,
        )
        keep = self._keep.numpy().astype(bool)
        xyz = self._points.numpy()[keep]
        color = self.cameras.color.numpy()[world, self.indices]
        rgb = color.view(np.uint8).reshape(color.shape + (4,))[keep][:, :3]
        if self.spec.voxel_size > 0.0:
            return voxelized_down_sample(xyz, rgb, self.spec.voxel_size)
        return xyz, rgb

    def log(self, viewer: newton.viewer.ViewerBase) -> None:
        xyz, rgb = self.compute()
        device = self._points.device
        viewer.log_points(
            self.spec.name, wp.array(xyz, dtype=wp.vec3, device=device),
            radii=0.5 * max(self.spec.voxel_size, 0.002),
            colors=wp.array(rgb / 255.0, dtype=wp.vec3, device=device),
        )
