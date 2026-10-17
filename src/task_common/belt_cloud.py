"""Real-like cropped cloud: depth/dropout/boundary noise, colour belt segmentation, pulley frame.

numpy + yaml only (no Newton/warp), so lcs_learning-side tooling and checks can import it.

Scene geometry is read from the same files the sim builds from:
- known large-pulley pose = ``round_belt_scene.yaml`` ``add_weld board::board`` ``X_PC``
  composed with ``round_belt_task_board.urdf`` joint ``large_round_pulley_joint`` ``origin``
  (joint angle 0);
- camera origins = the ``add_weld`` translations of the ``add_cropped_point_cloud`` cameras'
  ``<camera>::zed_camera`` (``common/directives/zed_cameras.yaml``; the weld child is the
  optical frame, so ``CameraSpec.X_WC`` translation == this);
- crop box = ``add_cropped_point_cloud`` ``crop_lower_xyz`` / ``crop_upper_xyz``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
# as round_belt_task.constants.SCENE_DIRECTIVES (ROUND_BELT_SCENE variant)
SCENE_YAML = (REPO_ROOT / os.environ["ROUND_BELT_SCENE"] if os.environ.get("ROUND_BELT_SCENE")
              else REPO_ROOT / "assets" / "round_belt_task" / "round_belt_scene.yaml")
BOARD_WELD_CHILD = "board::board"
PULLEY_JOINT = "large_round_pulley_joint"
POINT_CLOUD_NAME = "cropped_point_cloud"
CAMERA_FRAME = "zed_camera"

# Large-pulley section (data/lcs/diag/20261001-temporal-latent/scripts/pulley_sdf.py).
FLANGE_R_MM = 51.0
GROOVE_R_MM = 47.0
RIM_H_MM = 8.0
SEG_VERSION = 1
SEG_FRAME = "pulley_large"


# --- scene geometry ---------------------------------------------------------------------------

class _Loader(yaml.SafeLoader):
    pass


def _tag(loader: yaml.SafeLoader, suffix: str, node: yaml.Node):
    if isinstance(node, yaml.MappingNode):
        return {"!" + suffix: loader.construct_mapping(node, deep=True)}
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_Loader.add_multi_constructor("!", _tag)


def _directives(path: Path) -> list[tuple[str, dict]]:
    """``(kind, params)`` of ``path`` with ``add_directives`` files inlined."""
    doc = yaml.load(path.read_text(), Loader=_Loader) or {}
    out = []
    for entry in doc.get("directives", []):
        (kind, params), = entry.items()
        if kind == "add_directives":
            out += _directives((path.parent / params["file"]).resolve())
        else:
            out.append((kind, params))
    return out


def rpy_matrix(rpy_rad) -> np.ndarray:
    """Drake / URDF ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""
    r, p, y = (float(a) for a in rpy_rad)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), \
        math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat_wxyz_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_quat_wxyz(R) -> np.ndarray:
    """Unit quaternion ``wxyz`` with ``w >= 0``."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * math.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q) / np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def _weld(directives, child: str) -> tuple[np.ndarray, np.ndarray]:
    for kind, params in directives:
        if kind == "add_weld" and params.get("child") == child:
            x = params.get("X_PC") or {}
            rot = (x.get("rotation") or {}).get("!Rpy", {"deg": [0.0, 0.0, 0.0]})
            return (np.asarray(x.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
                    rpy_matrix(np.radians(rot["deg"])))
    raise KeyError(f"{SCENE_YAML.name}: no add_weld with child {child!r}")


def _cloud_params(directives) -> dict:
    for kind, params in directives:
        if kind == "add_cropped_point_cloud" and params.get("name") == POINT_CLOUD_NAME:
            return params
    raise KeyError(f"{SCENE_YAML.name}: no add_cropped_point_cloud {POINT_CLOUD_NAME!r}")


@functools.lru_cache(maxsize=1)
def _scene() -> dict:
    d = _directives(SCENE_YAML)
    board_file = next(p["file"] for k, p in d if k == "add_model" and p["name"] == "board")
    joints = ET.parse(SCENE_YAML.parent / board_file).getroot().iter("joint")
    origin = next(j for j in joints if j.get("name") == PULLEY_JOINT).find("origin")
    p_bj = np.array([float(v) for v in origin.get("xyz").split()])
    R_bj = rpy_matrix([float(v) for v in (origin.get("rpy") or "0 0 0").split()])
    p_wb, R_wb = _weld(d, BOARD_WELD_CHILD)
    cloud = _cloud_params(d)
    cams = np.stack([_weld(d, f"{c}::{CAMERA_FRAME}")[0] for c in cloud["cameras"]])
    return {"pulley": np.concatenate([p_wb + R_wb @ p_bj, matrix_quat_wxyz(R_wb @ R_bj)]),
            "cameras": cams,
            "crop": (np.asarray(cloud["crop_lower_xyz"], dtype=np.float64),
                     np.asarray(cloud["crop_upper_xyz"], dtype=np.float64))}


def known_large_pulley_pose7() -> np.ndarray:
    """World pose ``xyz_wxyz`` of the large pulley body at joint angle 0 (module docstring)."""
    return _scene()["pulley"].copy()


def scene_camera_origins() -> np.ndarray:
    """``(C, 3)`` world origins of the cropped cloud's cameras, in the cloud's camera order."""
    return _scene()["cameras"].copy()


def scene_crop_box() -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _scene()["crop"]
    return lo.copy(), hi.copy()


def to_pulley_frame(xyz_world, pose7_wxyz) -> np.ndarray:
    """``(N, 3)`` float64 ``R^T (p - c)``: x/y in the groove plane, z = h along the pulley +Z
    (``outcome.belt_in_pulley_frame``'s axis/azimuth convention)."""
    pose = np.asarray(pose7_wxyz, dtype=np.float64)
    p = np.asarray(xyz_world, dtype=np.float64).reshape(-1, 3)
    return (p - pose[:3]) @ quat_wxyz_matrix(pose[3:7])


def pulley_pose_mismatch(pose7_known, sim_pose7_wxyz, include_spin: bool = False
                         ) -> tuple[float, float]:
    """``(|dp| mm, deg)`` between two ``xyz_wxyz`` poses; ``deg`` = the angle between the two
    pulley axes (+Z). The spin about the axis is the free pulley joint's angle (up to ~8 deg in
    approach runs), not a calibration error, and h / r are invariant to it;
    ``include_spin=True`` returns the full relative rotation angle instead."""
    a = np.asarray(pose7_known, dtype=np.float64)
    b = np.asarray(sim_pose7_wxyz, dtype=np.float64)
    mm = float(np.linalg.norm(a[:3] - b[:3]) * 1e3)
    if include_spin:
        qa, qb = a[3:7] / np.linalg.norm(a[3:7]), b[3:7] / np.linalg.norm(b[3:7])
        return mm, 2.0 * math.degrees(math.acos(min(1.0, abs(float(qa @ qb)))))
    za, zb = quat_wxyz_matrix(a[3:7])[:, 2], quat_wxyz_matrix(b[3:7])[:, 2]
    return mm, math.degrees(math.atan2(np.linalg.norm(np.cross(za, zb)), float(za @ zb)))


# --- noise model ------------------------------------------------------------------------------

@dataclass(frozen=True)
class CloudNoiseSpec:
    """Sensor noise on the cropped voxel cloud (defaults = preset ``real_like``)."""

    depth_sigma_mm: float = 1.5
    depth_sigma_per_m: float = 2.0  # mm per metre of range, along the camera ray
    dropout_p: float = 0.15
    boundary_mm: float = 8.0
    boundary_dropout_p: float = 0.5
    boundary_flip_p: float = 0.2
    outlier_p: float = 0.01
    rgb_jitter: int = 8
    seed: int = 0

    def is_off(self) -> bool:
        return all(getattr(self, f.name) == 0 for f in dataclasses.fields(self)
                   if f.name != "seed")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_REAL = CloudNoiseSpec()
CLOUD_NOISE_PRESETS: dict[str, CloudNoiseSpec] = {
    "off": CloudNoiseSpec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
    "mild": CloudNoiseSpec(0.75, 1.0, 0.075, 4.0, 0.25, 0.1, 0.005, 4),
    "real_like": _REAL,
    "harsh": dataclasses.replace(_REAL, depth_sigma_mm=3.0, depth_sigma_per_m=4.0,
                                 dropout_p=0.30, boundary_mm=12.0),
}


def cloud_noise_spec(name_or_json) -> CloudNoiseSpec:
    """A preset name, a ``.json`` file or an inline JSON object (fields over ``real_like``)."""
    if isinstance(name_or_json, CloudNoiseSpec):
        return name_or_json
    text = str(name_or_json).strip()
    if text in CLOUD_NOISE_PRESETS:
        return CLOUD_NOISE_PRESETS[text]
    if text.startswith("{"):
        values = json.loads(text)
    elif text.endswith(".json") and Path(text).is_file():
        values = json.loads(Path(text).read_text())
    else:
        raise ValueError(f"cloud noise {text!r}: not a preset {sorted(CLOUD_NOISE_PRESETS)}, "
                         "an existing .json file or a JSON object")
    names = {f.name for f in dataclasses.fields(CloudNoiseSpec)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"cloud noise: unknown field(s) {unknown}; allowed {sorted(names)}")
    return dataclasses.replace(_REAL, **values)


def boundary_mask(xyz, jaw_centres, pulley_pose7, boundary_mm: float, rim: bool = True
                  ) -> np.ndarray:
    """Points within ``boundary_mm`` of a jaw centre or of the pulley rim
    (``|r - FLANGE_R_MM| <= boundary_mm`` and ``|h| <= RIM_H_MM``)."""
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    b = boundary_mm * 1e-3
    out = np.zeros(len(p), dtype=bool)
    for c in np.asarray(jaw_centres, dtype=np.float64).reshape(-1, 3):
        out |= np.linalg.norm(p - c, axis=1) <= b
    if not rim:
        return out
    local = to_pulley_frame(p, pulley_pose7) * 1e3
    r = np.linalg.norm(local[:, :2], axis=1)
    return out | ((np.abs(r - FLANGE_R_MM) <= boundary_mm) & (np.abs(local[:, 2]) <= RIM_H_MM))


def _flip_colours(xyz, rgb, belt, flip, rng, radius_m: float) -> np.ndarray:
    """Swap each flipped point's colour with a random other-class point within ``radius_m``;
    the indices that were flipped."""
    done = []
    for i in np.flatnonzero(flip):
        other = np.flatnonzero(belt != belt[i])
        if not len(other):
            continue
        near = other[np.linalg.norm(xyz[other] - xyz[i], axis=1) <= radius_m]
        if not len(near):
            continue
        j = near[rng.integers(len(near))]
        rgb[[i, j]] = rgb[[j, i]]
        belt[[i, j]] = belt[[j, i]]
        done.append(i)
    return np.asarray(done, dtype=np.int64)


def apply_cloud_noise(xyz, rgb, spec: CloudNoiseSpec, rng: np.random.Generator,
                      camera_origins, jaw_centres, pulley_pose7_known, belt_mask=None,
                      stats: dict | None = None):
    """``(xyz float32, rgb uint8)`` as the sensor would report them (``spec.is_off()``: the
    inputs, same objects). Per point: Gaussian range noise along the ray from the nearest camera
    origin (sigma = ``depth_sigma_mm + depth_sigma_per_m * range``), uniform dropout, extra
    dropout and belt <-> non-belt colour swaps inside :func:`boundary_mask`, ``outlier_p``
    points moved uniformly into the crop box, integer RGB jitter. Deterministic given ``rng``
    (fixed draw order). ``stats`` (optional) receives the per-call counts."""
    if spec.is_off():
        return xyz, rgb
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    colours = np.array(rgb, dtype=np.uint8).reshape(-1, 3)
    n = len(p)
    eps = rng.standard_normal(n)
    u_drop, u_bdrop, u_flip, u_out = (rng.random(n) for _ in range(4))
    out_xyz = rng.random((n, 3))
    jitter = rng.integers(-spec.rgb_jitter, spec.rgb_jitter + 1, size=(n, 3))

    belt = (segment_belt(p, colours) if belt_mask is None
            else np.asarray(belt_mask, dtype=bool).copy())
    edge = boundary_mask(p, jaw_centres, pulley_pose7_known, spec.boundary_mm)
    flips = _flip_colours(p, colours, belt, edge & (u_flip < spec.boundary_flip_p), rng,
                          spec.boundary_mm * 1e-3)

    origins = np.asarray(camera_origins, dtype=np.float64).reshape(-1, 3)
    ray = p[:, None, :] - origins[None]
    rng_m = np.linalg.norm(ray, axis=2)
    k = np.argmin(rng_m, axis=1)
    dist = rng_m[np.arange(n), k]
    unit = ray[np.arange(n), k] / np.maximum(dist, 1e-9)[:, None]
    sigma_m = (spec.depth_sigma_mm + spec.depth_sigma_per_m * dist) * 1e-3
    noisy = p + (eps * sigma_m)[:, None] * unit

    lo, hi = scene_crop_box()
    outlier = u_out < spec.outlier_p
    noisy[outlier] = lo + out_xyz[outlier] * (hi - lo)
    colours = np.clip(colours.astype(np.int64) + jitter, 0, 255).astype(np.uint8)

    keep = ~((u_drop < spec.dropout_p) | (edge & (u_bdrop < spec.boundary_dropout_p)))
    if not keep.any() and n:
        keep[0] = True  # camera_points() needs >= 1 point
    if stats is not None:
        moved = ~outlier & keep
        jaw = boundary_mask(p[flips], jaw_centres, pulley_pose7_known, spec.boundary_mm,
                            rim=False)
        stats.update(n_in=n, n_out=int(keep.sum()), n_boundary=int(edge.sum()),
                     n_flips=len(flips), n_flips_jaw=int(jaw.sum()),
                     n_outliers=int((outlier & keep).sum()),
                     mean_disp_mm=float(np.mean(np.abs(eps * sigma_m)[moved]) * 1e3)
                     if moved.any() else float("nan"))
    return noisy[keep].astype(np.float32), colours[keep]


# --- segmentation -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BeltSegSpec:
    """Orange-belt colour rule on uint8 RGB.

    Observed (approach/v1/train ep 0-4 + contact/v1/train ep 0-4, 4 frames each, clean): points
    within 6 mm of a belt body are R 47..125, G/R 0.43..0.51, B 0..4 (orange shaded by the
    scene's light); everything farther than 15 mm is neutral grey (R == G, B >= R): the black
    large pulley / plate ~(2..12, 2..12, 3..13), the grey board ~(77..97, 77..97, 82..103), the
    grippers dark grey. Rule: ``R >= r_min``, ``R - B >= r_minus_b_min`` and
    ``g_ratio_lo <= (G - B) / (R - B) <= g_ratio_hi``; frames with fewer than ``min_points``
    such points return an empty mask (belt not seen).
    """

    r_min: int = 40
    r_minus_b_min: int = 30
    g_ratio_lo: float = 0.2
    g_ratio_hi: float = 0.75
    min_points: int = 20

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_SEG = BeltSegSpec()


def segment_belt(xyz, rgb, spec: BeltSegSpec = DEFAULT_SEG) -> np.ndarray:
    """``(N,)`` bool belt mask of the cloud's colours (``xyz`` only fixes ``N``)."""
    c = np.asarray(rgb, dtype=np.int64).reshape(-1, 3)
    if len(c) != len(np.asarray(xyz).reshape(-1, 3)):
        raise ValueError(f"segment_belt: {len(c)} colours != {len(xyz)} points")
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    span = r - b
    ratio = (g - b) / np.maximum(span, 1)
    mask = ((r >= spec.r_min) & (span >= spec.r_minus_b_min)
            & (ratio >= spec.g_ratio_lo) & (ratio <= spec.g_ratio_hi))
    if mask.sum() < spec.min_points:
        mask[:] = False
    return mask


def resample_deterministic(xyz, n: int) -> tuple[np.ndarray, bool]:
    """``((n, 3) float32, empty)``: lexicographic (x, y, z) sort, then evenly spaced indices
    (M >= n), cyclic repetition (0 < M < n) or zeros (M == 0, ``empty``)."""
    p = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros((n, 3), dtype=np.float32), True
    p = p[np.lexsort((p[:, 2], p[:, 1], p[:, 0]))]
    idx = (np.floor(np.linspace(0, len(p) - 1, n)).astype(np.int64) if len(p) >= n
           else np.arange(n) % len(p))
    return p[idx], False


def belt_seg_frame(xyz, rgb, noise_spec: CloudNoiseSpec, seg_spec: BeltSegSpec,
                   rng: np.random.Generator, camera_origins, jaw_centres, pose7_known,
                   stats: dict | None = None) -> dict:
    """``{pcd, pcd_rgb}`` = the noised cloud (the sensor), ``pcd_belt_seg`` = its belt points
    in the known pulley frame ``(M, 3)`` float32, ``n_pts`` = M."""
    belt = segment_belt(xyz, rgb, seg_spec)
    pcd, pcd_rgb = apply_cloud_noise(xyz, rgb, noise_spec, rng, camera_origins, jaw_centres,
                                     pose7_known, belt_mask=belt, stats=stats)
    mask = segment_belt(pcd, pcd_rgb, seg_spec)
    seg = to_pulley_frame(np.asarray(pcd)[mask], pose7_known).astype(np.float32)
    return {"pcd": pcd, "pcd_rgb": pcd_rgb, "pcd_belt_seg": seg, "n_pts": len(seg)}


def belt_seg_meta(noise_spec: CloudNoiseSpec, seg_spec: BeltSegSpec, source: str,
                  pose7_known=None) -> dict:
    """The ``sim_meta["belt_seg"]`` block (``docs/lcs-dataset.md``)."""
    if source not in ("record", "offline", "rerender"):
        raise ValueError(f"belt_seg source {source!r} not in ('record', 'offline', 'rerender')")
    pose = known_large_pulley_pose7() if pose7_known is None else np.asarray(pose7_known)
    return {"version": SEG_VERSION, "frame": SEG_FRAME,
            "pulley_pose7_known": [float(v) for v in pose], "noise_spec": noise_spec.to_dict(),
            "seg_spec": seg_spec.to_dict(), "source": source, "scene": scene_info()}


def scene_info() -> dict:
    """Scene file of the cloud (camera resolution, voxel size) + its point-cloud settings."""
    pc = _cloud_params(_directives(SCENE_YAML))
    return {"path": str(SCENE_YAML.relative_to(REPO_ROOT)) if SCENE_YAML.is_relative_to(REPO_ROOT) else str(SCENE_YAML),
            "voxel_size": float(pc.get("voxel_size", 0.0))}


def jaw_centres_from_poses(ee_franka7, ee_ur7) -> np.ndarray:
    """``(2, 3)`` jaw centres = the EE positions (Franka ``finger_tip``, UR tracking frame)."""
    return np.stack([np.asarray(ee_franka7, dtype=np.float64)[:3],
                     np.asarray(ee_ur7, dtype=np.float64)[:3]])
