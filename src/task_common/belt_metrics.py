"""Belt-alignment metrics between two material-ordered belts (``(150, 3)`` metres each).

Index-wise RMSE compares the same material points, so it only means "same shape" for the same
grasp; the chamfer and best-cyclic-shift metrics stay meaningful when the loop is rotated in
material index.
"""

from __future__ import annotations

import math

import numpy as np

from task_common.lcs_dataset import BELT_POINTS


def _belt(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (BELT_POINTS, 3):
        raise ValueError(f"{name} shape {arr.shape} != ({BELT_POINTS}, 3)")
    return arr


def belt_rmse_mm(a, b) -> float:
    """Index-wise point RMSE in mm."""
    a, b = _belt(a, "a"), _belt(b, "b")
    return 1e3 * math.sqrt(float(np.mean(np.sum((a - b) ** 2, axis=1))))


def belt_chamfer_mm(a, b) -> float:
    """Symmetric mean nearest-neighbour distance in mm."""
    a, b = _belt(a, "a"), _belt(b, "b")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return 1e3 * 0.5 * float(d.min(axis=1).mean() + d.min(axis=0).mean())


def belt_best_shift_rmse_mm(a, b) -> tuple[float, int]:
    """``(rmse_mm, shift)``: min over cyclic shifts, ``b ~ np.roll(a, shift, axis=0)``."""
    a, b = _belt(a, "a"), _belt(b, "b")
    errs = [float(np.mean(np.sum((a - np.roll(b, -s, axis=0)) ** 2, axis=1)))
            for s in range(BELT_POINTS)]
    s = int(np.argmin(errs))
    return 1e3 * math.sqrt(errs[s]), s


def pose_error(pose7_a, pose7_b) -> tuple[float, float]:
    """``(mm, deg)`` between two ``xyz_wxyz`` poses."""
    a = np.asarray(pose7_a, dtype=np.float64)
    b = np.asarray(pose7_b, dtype=np.float64)
    if a.shape != (7,) or b.shape != (7,):
        raise ValueError(f"pose shapes {a.shape} / {b.shape} != (7,)")
    qa, qb = a[3:] / np.linalg.norm(a[3:]), b[3:] / np.linalg.norm(b[3:])
    dot = min(abs(float(np.dot(qa, qb))), 1.0)
    return 1e3 * float(np.linalg.norm(a[:3] - b[:3])), math.degrees(2.0 * math.acos(dot))
