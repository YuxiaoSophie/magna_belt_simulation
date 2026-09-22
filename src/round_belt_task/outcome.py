"""Belt-engagement outcome of the large pulley from belt body positions and the pulley pose.

The pulley frame matches ``replay_metrics._pulley_wrap_deg``: axis = the body's +Z (up on the
board), ``h`` = height along the axis from the body origin (groove seat plane at ``h = 0``),
``r`` = radial distance, azimuth measured from the body's +X.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from task_common.replay_metrics import PULLEY_SEAT_RADIUS_MM, quat_rotate
from task_common.replay_metrics import _largest_arc_deg as largest_arc_deg

LABELS = ("engaged", "over", "under", "slanted", "outside", "other")
LARGE_SEAT_MM = PULLEY_SEAT_RADIUS_MM[1]
EPISODE_METRIC_KEYS = ("wrap_deg", "h_median_mm", "h_min_mm", "h_max_mm", "n_neighbour")


@dataclass(frozen=True)
class OutcomeThresholds:
    neighbour_radial_mm: float = 12.0
    neighbour_inner_mm: float = 10.0
    in_groove_axial_mm: float = 4.0
    in_groove_radial_mm: float = 5.0
    engaged_arc_deg: float = 60.0
    partial_arc_deg: float = 15.0
    over_under_h_mm: float = 5.0
    slant_spread_mm: float = 8.0
    outside_min_bodies: int = 2


DEFAULT_THRESHOLDS = OutcomeThresholds()


@dataclass
class FrameMetrics:
    """Neighbourhood N = bodies within the radial band around the seat; h/r stats are over N."""

    wrap_deg: float
    n_neighbour: int
    h_median_mm: float
    h_min_mm: float
    h_max_mm: float
    r_median_mm: float
    seated_bodies: int


def belt_in_pulley_frame(belt_xyz: np.ndarray, pulley_pose7: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h_mm, r_mm, azimuth_deg)`` of each belt body; ``pulley_pose7`` = xyz + quat xyzw."""
    pose = np.asarray(pulley_pose7, dtype=np.float64)
    centre, q = pose[:3], pose[3:7] / np.linalg.norm(pose[3:7])
    axis = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
    x_dir = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    y_dir = np.cross(axis, x_dir)
    d = np.asarray(belt_xyz, dtype=np.float64) - centre
    h = d @ axis
    radial = d - h[:, None] * axis
    azimuth = np.degrees(np.arctan2(radial @ y_dir, radial @ x_dir))
    return h * 1e3, np.linalg.norm(radial, axis=1) * 1e3, azimuth


def frame_metrics(belt_xyz: np.ndarray, pulley_pose7: np.ndarray, seat_mm: float = LARGE_SEAT_MM,
                  th: OutcomeThresholds = DEFAULT_THRESHOLDS) -> FrameMetrics:
    h, r, azimuth = belt_in_pulley_frame(belt_xyz, pulley_pose7)
    near = (r <= seat_mm + th.neighbour_radial_mm) & (r >= seat_mm - th.neighbour_inner_mm)
    seated = ((np.abs(h) <= th.in_groove_axial_mm)
              & (np.abs(r - seat_mm) <= th.in_groove_radial_mm))
    if near.any():
        hn = h[near]
        h_med, h_min, h_max = float(np.median(hn)), float(hn.min()), float(hn.max())
        r_med = float(np.median(r[near]))
    else:
        h_med = h_min = h_max = r_med = float("nan")
    return FrameMetrics(wrap_deg=largest_arc_deg(azimuth[seated]),
                        n_neighbour=int(near.sum()), h_median_mm=h_med, h_min_mm=h_min,
                        h_max_mm=h_max, r_median_mm=r_med, seated_bodies=int(seated.sum()))


def classify(fm: FrameMetrics, th: OutcomeThresholds = DEFAULT_THRESHOLDS) -> str:
    if fm.n_neighbour < th.outside_min_bodies:
        return "outside"
    if fm.wrap_deg >= th.engaged_arc_deg:
        return "engaged"
    if (th.partial_arc_deg <= fm.wrap_deg
            or (fm.h_max_mm - fm.h_min_mm > th.slant_spread_mm and fm.seated_bodies >= 1)):
        return "slanted"
    if fm.h_median_mm > th.over_under_h_mm:
        return "over"
    if fm.h_median_mm < -th.over_under_h_mm:
        return "under"
    return "other"


def classify_episode(belt_xyz_T: np.ndarray, pulley_pose7_T: np.ndarray,
                     th: OutcomeThresholds = DEFAULT_THRESHOLDS, last_n: int = 1,
                     seat_mm: float = LARGE_SEAT_MM) -> tuple[str, dict[str, np.ndarray]]:
    """Label = majority of the last ``last_n`` frames (ties go to the latest); per-frame metrics."""
    frames = [frame_metrics(b, p, seat_mm, th) for b, p in zip(belt_xyz_T, pulley_pose7_T,
                                                              strict=True)]
    if not frames:
        raise ValueError("classify_episode needs at least one frame")
    rows = [asdict(fm) for fm in frames]
    metrics = {k: np.array([row[k] for row in rows],
                           dtype=np.int32 if k == "n_neighbour" else np.float64)
               for k in EPISODE_METRIC_KEYS}
    tail = [classify(fm, th) for fm in frames[-max(1, last_n):]]
    counts = Counter(tail)
    best = max(counts.values())
    label = next(lab for lab in reversed(tail) if counts[lab] == best)
    return label, metrics
