#!/usr/bin/env python3
"""Headless, solver-free check of the round-belt scene's ZED cameras and cropped point cloud.

Asserts the intrinsics against magna's ``depth_camera_intrinsics``, the extrinsics against an
independent numpy rpy of the YAML welds, and the projection chain by back-projecting every
depth pixel: the most populated world height must be the table top.  The cropped cloud must be
non-empty, inside its crop box and one point per voxel.

Run:
    uv run python scripts/checks/check_scene_cameras.py
    uv run python scripts/checks/check_scene_cameras.py --out /tmp/zed   # also save images and cloud
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[2]
# The task packages live under src/; make them importable regardless of CWD.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import newton
from PIL import Image

from round_belt_task.constants import SCENE_DIRECTIVES
from round_belt_task.joint_state import apply_default_joint_state
from round_belt_task.scene import build_scene, make_builder
from task_common.cameras import CameraSpec, RgbdCameras
from task_common.point_cloud import CroppedPointCloud, PointCloudSpec
from utils.directives import DirectiveFile, parse_directives

# magna's round_belt_simulation_params.yaml
DRAKE_INTRINSICS = {
    "width": 640, "height": 480, "focal_x": 579.4112549695427, "focal_y": 579.4112549695427,
    "center_x": 319.5, "center_y": 239.5,
}
CAMERAS = ["zed_camera_franka_side", "zed_camera_ur_side"]


def rpy_deg_to_mat3(rpy_deg) -> np.ndarray:
    r, p, y = (math.radians(float(a)) for a in rpy_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def check_camera(
    spec: CameraSpec, rgb: np.ndarray, depth: np.ndarray, table_top_z: float,
    directives: DirectiveFile,
) -> list[str]:
    failures = []
    for key, want in DRAKE_INTRINSICS.items():
        if not math.isclose(float(getattr(spec, key)), want, rel_tol=0.0, abs_tol=1e-9):
            failures.append(f"{spec.name}.{key} = {getattr(spec, key)}, magna has {want}")

    pose = directives.weld(f"{spec.name}::zed_camera").X_PC
    R_WC = rpy_deg_to_mat3(pose.rpy_deg)
    p_WC = np.asarray(pose.translation, dtype=np.float64)
    R_got = np.array(wp.quat_to_matrix(spec.X_WC.q), dtype=np.float64).reshape(3, 3)
    if not (np.allclose(R_got, R_WC, atol=1e-6) and np.allclose(spec.X_WC.p, p_WC, atol=1e-6)):
        failures.append(f"{spec.name}: X_WC {spec.X_WC} != weld {pose}")

    valid = np.isfinite(depth) & (depth > 0.0)
    v, u = np.nonzero(valid)
    z = depth[valid].astype(np.float64)
    x = (u - spec.center_x) * z / spec.focal_x
    y = (v - spec.center_y) * z / spec.focal_y
    world_z = (np.stack([x, y, z], axis=1) @ R_WC.T + p_WC)[:, 2]

    counts, edges = np.histogram(world_z[world_z < 0.2], bins=np.arange(-0.2, 0.2, 0.002))
    mode_z = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
    print(f"{spec.name}: {valid.mean():.1%} valid pixels, most common world z {mode_z:+.4f} m, "
          f"table top {table_top_z:+.4f} m")
    if abs(mode_z - table_top_z) > 0.003:
        failures.append(f"{spec.name}: dominant plane at z={mode_z:+.4f}, not the table top")
    if not rgb.any():
        failures.append(f"{spec.name}: rgb image is all black")
    return failures


def check_point_cloud(xyz: np.ndarray, spec: PointCloudSpec) -> list[str]:
    lower, upper = np.asarray(spec.crop_lower_xyz), np.asarray(spec.crop_upper_xyz)
    print(f"{spec.name}: {len(xyz)} points from {list(spec.cameras)}")
    if len(xyz) == 0:
        return [f"{spec.name}: empty"]
    if not ((xyz >= lower - 1e-6).all() and (xyz <= upper + 1e-6).all()):
        return [f"{spec.name}: points outside the crop box"]
    if spec.voxel_size > 0.0:
        voxels = len(np.unique(np.floor(xyz / spec.voxel_size).astype(np.int64), axis=0))
        if voxels != len(xyz):
            return [f"{spec.name}: {len(xyz)} points in {voxels} voxels"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.device:
        wp.set_device(args.device)

    builder = make_builder()
    info = build_scene(builder)
    builder.color()
    model = builder.finalize()
    apply_default_joint_state(model, info)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    assert [spec.name for spec in info.cameras] == CAMERAS, info.cameras
    cameras = RgbdCameras(model, info.cameras)
    cameras.update(state)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    directives = parse_directives(SCENE_DIRECTIVES)
    table_top_z = float(info.table_visual_aabb[1][2])
    failures = []
    for index, spec in enumerate(info.cameras):
        rgb, depth = cameras.rgbd(index)
        failures += check_camera(spec, rgb, depth, table_top_z, directives)
        if args.out is not None:
            Image.fromarray(rgb).save(args.out / f"{spec.name}_rgb.png")
            near_bright = 1.0 - np.clip((depth - 0.3) / 1.2, 0.0, 1.0)  # 0.3 m white, 1.5 m black
            depth_png = Image.fromarray((255 * near_bright).astype(np.uint8))
            depth_png.save(args.out / f"{spec.name}_depth.png")
            np.save(args.out / f"{spec.name}_depth.npy", depth)

    for spec in info.point_clouds:
        xyz, rgb = CroppedPointCloud(cameras, spec).compute()
        failures += check_point_cloud(xyz, spec)
        if args.out is not None:
            np.savez(args.out / f"{spec.name}.npz", xyz=xyz, rgb=rgb)

    if failures:
        raise SystemExit("FAILED:\n  " + "\n  ".join(failures))
    print("[OK] cameras match magna's ZED intrinsics and extrinsics; point clouds are sane")


if __name__ == "__main__":
    main()
