"""Chain-on-sprocket outcome from roller positions and a sprocket body pose.

Frame as ``round_belt_task.outcome``: axis = the sprocket body's +Z, azimuth from its +X (tooth 0),
``h`` = height above the tooth (rim mid) plane. Geometry is read from the aligned board URDF.
Inputs: roller centres (64 x 3, world), from perception or ``rollers_from_inner_bodies``.
"""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from round_belt_task.outcome import (
    FrameMetrics,
    OutcomeThresholds,
    belt_in_pulley_frame,
    classify,
    frame_metrics,
    largest_arc_deg,
)
from task_common.replay_metrics import quat_rotate

BOARD_URDF = (Path(__file__).resolve().parents[2] / "assets" / "bike_chain_task"
              / "bike_chain_task_board_aligned.urdf")
SPROCKETS = ("small", "large", "tensioner")
PITCH_MM = 9.525
ROLLER_Z_MM = 3.5  # roller centre above the inner link's frame (plate 1.25 + half of the 4.5 gap)
CHAIN_THRESHOLDS = OutcomeThresholds(
    neighbour_radial_mm=8.0, neighbour_inner_mm=6.0, in_groove_axial_mm=3.0,
    in_groove_radial_mm=2.5, engaged_arc_deg=90.0, partial_arc_deg=20.0, over_under_h_mm=4.0,
    slant_spread_mm=6.0, outside_min_bodies=2)
MESHED_MIN = 0.8


def load_geometry(urdf: Path = BOARD_URDF) -> dict[str, dict]:
    """Per sprocket: ``teeth``, ``pitch_radius_mm``, ``tooth_z_mm`` (body frame), ``az0_deg``."""
    out = {}
    for link in ET.parse(urdf).getroot().iter("link"):
        name = link.get("name", "").removesuffix("_sprocket")
        if name not in SPROCKETS:
            continue
        rim = next(c for c in link.iter("collision") if c.get("name", "").endswith("_rim"))
        teeth = [c for c in link.iter("collision") if "_tooth_" in c.get("name", "")]
        t0 = [float(v) for v in next(c for c in teeth if c.get("name", "").endswith("_tooth_0"))
              .find("origin").get("xyz").split()]
        n = len(teeth)
        out[name] = {"teeth": n, "pitch_radius_mm": PITCH_MM / (2 * math.sin(math.pi / n)),
                     "tooth_z_mm": 1e3 * float(rim.find("origin").get("xyz").split()[2]),
                     "az0_deg": math.degrees(math.atan2(t0[1], t0[0]))}
    return out


GEOM = load_geometry()


def rollers_from_inner_bodies(inner_pose7: np.ndarray) -> np.ndarray:
    """Ground truth: the 32 inner-link body poses (xyz + quat xyzw) -> 64 roller centres."""
    p = np.asarray(inner_pose7, dtype=np.float64)
    off = np.array([[0.0, y, ROLLER_Z_MM * 1e-3] for y in (0.0, PITCH_MM * 1e-3)])
    pts = [p[:, :3] + quat_rotate(p[:, 3:7], o) for o in off]
    return np.stack(pts, axis=1).reshape(-1, 3)


def _seated(chain_xyz, pose7, which):
    g = GEOM[which]
    pose = np.array(pose7, dtype=np.float64)
    q = pose[3:7] / np.linalg.norm(pose[3:7])
    pose[:3] += quat_rotate(q, np.array([0.0, 0.0, g["tooth_z_mm"] * 1e-3]))  # h=0 at tooth plane
    return pose, g


def sprocket_metrics(chain_xyz: np.ndarray, sprocket_pose7: np.ndarray, which: str
                     ) -> FrameMetrics:
    pose, g = _seated(chain_xyz, sprocket_pose7, which)
    th, seat = CHAIN_THRESHOLDS, g["pitch_radius_mm"]
    fm = frame_metrics(chain_xyz, pose, seat, th)
    h, r, az = belt_in_pulley_frame(chain_xyz, pose)
    ok = (np.abs(h) <= th.in_groove_axial_mm) & (np.abs(r - seat) <= th.in_groove_radial_mm)
    merge = math.degrees(1.6 * PITCH_MM / seat)  # round-belt merge gap assumes its body spacing
    return dataclasses.replace(fm, wrap_deg=largest_arc_deg(az[ok], merge))


def meshed_fraction(chain_xyz: np.ndarray, sprocket_pose7: np.ndarray, which: str,
                    mesh_tol_deg: float = 4.0) -> float:
    """Fraction of seated rollers within ``mesh_tol_deg`` of a gap centre; nan if none seated."""
    pose, g = _seated(chain_xyz, sprocket_pose7, which)
    th, seat = CHAIN_THRESHOLDS, g["pitch_radius_mm"]
    h, r, az = belt_in_pulley_frame(chain_xyz, pose)
    ok = (np.abs(h) <= th.in_groove_axial_mm) & (np.abs(r - seat) <= th.in_groove_radial_mm)
    if not ok.any():
        return float("nan")
    tp = 360.0 / g["teeth"]
    off = np.abs((az[ok] - g["az0_deg"]) % tp - 0.5 * tp)
    return float((off <= mesh_tol_deg).mean())


def classify_sprocket(chain_xyz: np.ndarray, sprocket_pose7: np.ndarray, which: str
                      ) -> tuple[str, float]:
    """``(label, meshed_fraction)``; an arc-engaged chain with meshed < 0.8 or an h spread over the
    slant threshold is labelled slanted."""
    fm = sprocket_metrics(chain_xyz, sprocket_pose7, which)
    frac = meshed_fraction(chain_xyz, sprocket_pose7, which)
    label = classify(fm, CHAIN_THRESHOLDS)
    spread = fm.h_max_mm - fm.h_min_mm > CHAIN_THRESHOLDS.slant_spread_mm
    if label == "engaged" and (spread or not frac >= MESHED_MIN):
        label = "slanted"
    return label, frac


def success_small(chain_xyz: np.ndarray, small_pose7: np.ndarray) -> bool:
    return classify_sprocket(chain_xyz, small_pose7, "small")[0] == "engaged"


def success_both(chain_xyz: np.ndarray, small_pose7: np.ndarray, large_pose7: np.ndarray) -> bool:
    return success_small(chain_xyz, small_pose7) and (
        classify_sprocket(chain_xyz, large_pose7, "large")[0] == "engaged")


def classify_sprocket_episode(chain_xyz_T: np.ndarray, pose7_T: np.ndarray, which: str,
                              last_n: int = 1) -> str:
    """Majority label of the last ``last_n`` frames (ties go to the latest)."""
    tail = [classify_sprocket(c, p, which)[0]
            for c, p in zip(chain_xyz_T, pose7_T, strict=True)][-max(1, last_n):]
    counts = Counter(tail)
    return next(lab for lab in reversed(tail) if counts[lab] == max(counts.values()))
