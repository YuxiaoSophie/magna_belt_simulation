"""Task features of the segmented belt cloud in the known large-pulley frame.

Real-computable stand-ins for the sim-only outcome metrics (``round_belt_task.outcome``): height
``h`` of the belt over the groove seat plane, radial offset ``r`` and the wrap arc, estimated
from ``pcd_belt_seg`` (``task_common.belt_cloud``), plus a cloud-based success detector.

The cloud sees the tube SURFACE (mostly from above and outside), not the belt centreline, so the
raw band median sits ``h_bias_mm`` above the ground-truth ``h``; points under the jaws are
missing, so the wrap under-reads by up to the jaw footprint. numpy only (no Newton/warp).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from task_common.replay_metrics import PULLEY_SEAT_RADIUS_MM
from task_common.replay_metrics import _largest_arc_deg as largest_arc_deg

LARGE_SEAT_MM = PULLEY_SEAT_RADIUS_MM[1]
RULES = ("strict", "engaged")
DEFAULT_RULE = "engaged"  # user decision Q1 (2026-10-05)
EXTRA_KEYS = ("cloud_h_mm", "cloud_wrap_deg", "cloud_r_mm", "cloud_n_pts", "cloud_success")


@dataclass(frozen=True)
class CloudFeatureSpec:
    seat_mm: float = LARGE_SEAT_MM
    band_outer_mm: float = 12.0
    band_inner_mm: float = 10.0
    tube_radius_mm: float = 3.3
    h_bias_mm: float = 1.1  # G1 2026-10-05: median raw - GT h, 80 GT-engaged frames, `off`
    arc_bin_deg: float = 10.0
    arc_h_tol_mm: float = 4.0
    arc_r_tol_mm: float = 5.0
    min_band_points: int = 10

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_SPEC = CloudFeatureSpec()


@dataclass(frozen=True)
class CloudFeatures:
    """``h_*`` are bias-corrected (mm over the seat plane); NaN when ``n_band`` is too small."""

    h_median_mm: float
    h_p10_mm: float
    h_p90_mm: float
    r_median_mm: float
    wrap_deg: float
    n_pts: int
    n_band: int
    slant_mm: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CloudSuccessThresholds:
    # G1 2026-10-05: best per-frame F1 on `real_like` (perception_gate.py sweep, 480 frames)
    strict_wrap_deg: float = 85.0
    strict_h_mm: float = 2.5
    engaged_wrap_deg: float = 65.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_THRESHOLDS = CloudSuccessThresholds()


def polar_mm(xyz_pulley) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h_mm, r_mm, azimuth_deg)`` of pulley-frame points in metres (azimuth from +X)."""
    p = np.asarray(xyz_pulley, dtype=np.float64).reshape(-1, 3) * 1e3
    return p[:, 2], np.hypot(p[:, 0], p[:, 1]), np.degrees(np.arctan2(p[:, 1], p[:, 0]))


def binned_arc_deg(azimuth_deg, bin_deg: float) -> float:
    """Largest contiguous arc (``replay_metrics._largest_arc_deg``) over occupied bin centres."""
    a = np.asarray(azimuth_deg, dtype=np.float64)
    if a.size == 0:
        return 0.0
    n_bins = round(360.0 / bin_deg)
    idx = np.unique(np.floor(np.mod(a, 360.0) / bin_deg).astype(np.int64) % n_bins)
    return float(largest_arc_deg((idx + 0.5) * bin_deg))


def cloud_features(xyz_pulley, spec: CloudFeatureSpec = DEFAULT_SPEC) -> CloudFeatures:
    """Features of the ``(M, 3)`` belt cloud in the pulley frame, METRES (``pcd_belt_seg``)."""
    h, r, az = polar_mm(xyz_pulley)
    band = (r >= spec.seat_mm - spec.band_inner_mm) & (r <= spec.seat_mm + spec.band_outer_mm)
    n_band = int(band.sum())
    if n_band < spec.min_band_points:
        nan = float("nan")
        return CloudFeatures(nan, nan, nan, nan, 0.0, len(h), n_band, nan)
    hb = h[band] - spec.h_bias_mm
    p10, med, p90 = (float(v) for v in np.percentile(hb, [10.0, 50.0, 90.0]))
    arc = (band & (np.abs(h - spec.h_bias_mm) <= spec.arc_h_tol_mm)
           & (np.abs(r - spec.seat_mm) <= spec.arc_r_tol_mm))
    return CloudFeatures(h_median_mm=med, h_p10_mm=p10, h_p90_mm=p90,
                         r_median_mm=float(np.median(r[band])),
                         wrap_deg=binned_arc_deg(az[arc], spec.arc_bin_deg), n_pts=len(h),
                         n_band=n_band, slant_mm=p90 - p10)


def cloud_success(f: CloudFeatures, rule: str = DEFAULT_RULE,
                  th: CloudSuccessThresholds = DEFAULT_THRESHOLDS) -> bool:
    """strict: wrap >= ``strict_wrap_deg`` and ``|h| <= strict_h_mm``; engaged: wrap >=
    ``engaged_wrap_deg``. NaN features -> False."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r} not in {RULES}")
    if not (math.isfinite(f.wrap_deg) and math.isfinite(f.h_median_mm)):
        return False
    if rule == "engaged":
        return f.wrap_deg >= th.engaged_wrap_deg
    return f.wrap_deg >= th.strict_wrap_deg and abs(f.h_median_mm) <= th.strict_h_mm


def episode_cloud_success(features_T, rule: str = DEFAULT_RULE, last_n: int = 13,
                          th: CloudSuccessThresholds = DEFAULT_THRESHOLDS) -> bool:
    """Strict majority of the last ``last_n`` frames (13 ~ 1 s at 75 ms)."""
    tail = list(features_T)[-max(1, last_n):]
    if not tail:
        return False
    return 2 * sum(cloud_success(f, rule, th) for f in tail) > len(tail)


def cloud_extras(xyz_pulley, spec: CloudFeatureSpec = DEFAULT_SPEC,
                 th: CloudSuccessThresholds = DEFAULT_THRESHOLDS) -> dict:
    """Per-frame dataset extras (stored as ``sim_cloud_*``); success = the default rule."""
    f = cloud_features(xyz_pulley, spec)
    return {"cloud_h_mm": np.float32(f.h_median_mm), "cloud_wrap_deg": np.float32(f.wrap_deg),
            "cloud_r_mm": np.float32(f.r_median_mm), "cloud_n_pts": np.int32(f.n_pts),
            "cloud_success": np.bool_(cloud_success(f, DEFAULT_RULE, th))}
