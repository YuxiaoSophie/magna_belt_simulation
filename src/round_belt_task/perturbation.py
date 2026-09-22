"""Class-conditioned perturbations of the ``pre_place_2`` / ``place_3`` waypoints.

Each episode samples an ``intent`` (the outcome it aims for) and, per arm, a world-frame position
offset plus a tilt about the belt tangent (the horizontal Franka-tip -> UR-tip direction at the
nominal ``pre_place_2``). The UR carries the class-specific offset; the Franka gets shared jitter.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from round_belt_task.arm_kinematics import (
    mat3_to_quat_xyzw,
    quat_xyzw_to_mat3,
    rot_axis_angle,
)
from round_belt_task.waypoints import Waypoint

INTENTS = ("engaged", "over", "under", "slanted")
PERTURBED_LABELS = ("pre_place_2", "place_3")
TANGENT_LABEL = "pre_place_2"


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))


@dataclass(frozen=True)
class ClassRanges:
    """Sampling ranges of one intent (mm / deg); the tilt sign is flipped at random if asked."""

    ur_dz_mm: Range
    ur_dxy_mm: Range = Range(-3.0, 3.0)
    ur_tilt_deg: Range = Range(-2.0, 2.0)
    ur_tilt_sign_random: bool = False
    franka_dxyz_mm: Range = Range(-2.0, 2.0)
    franka_tilt_deg: Range = Range(-3.0, 3.0)


# Tuned on the sim (RUN-STATE PKG-20260922-lcs-collector): engaged only for UR dz ~1..5 mm, and a
# 4-6 deg tangent tilt already slants the loop (>= 6 deg acts like a large dz). ``under`` is
# tilt-driven, not dz-driven: a negative dz drags the 2F-85 fingers through the board plate
# (RUN-STATE PKG-20260922-board-clearance), a +10..18 deg tilt puts the loop under the groove with
# the gripper 3-4 mm clear.
DEFAULT_RANGES: dict[str, ClassRanges] = {
    "engaged": ClassRanges(ur_dz_mm=Range(1.5, 5.0), ur_tilt_deg=Range(-2.0, 1.5)),
    "over": ClassRanges(ur_dz_mm=Range(11.0, 20.0), ur_tilt_deg=Range(-3.0, 1.0)),
    "under": ClassRanges(ur_dz_mm=Range(2.0, 4.0), ur_tilt_deg=Range(10.0, 18.0)),
    "slanted": ClassRanges(ur_dz_mm=Range(2.0, 4.0), ur_tilt_deg=Range(3.5, 5.5),
                           ur_tilt_sign_random=True),
}


@dataclass(frozen=True)
class Perturbation:
    intent: str
    ur_dpos_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ur_tilt_deg: float = 0.0
    franka_dpos_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    franka_tilt_deg: float = 0.0

    def to_dict(self) -> dict:
        return {"intent": self.intent, "ur_dpos_m": [float(v) for v in self.ur_dpos_m],
                "ur_tilt_deg": float(self.ur_tilt_deg),
                "franka_dpos_m": [float(v) for v in self.franka_dpos_m],
                "franka_tilt_deg": float(self.franka_tilt_deg)}


def ranges_to_dict(ranges: dict[str, ClassRanges] = DEFAULT_RANGES) -> dict:
    return {intent: asdict(r) for intent, r in ranges.items()}


def sample(rng: np.random.Generator, intent: str,
           ranges: dict[str, ClassRanges] = DEFAULT_RANGES) -> Perturbation:
    """One perturbation of ``intent``; draws in a fixed order so a seed reproduces it."""
    if intent not in ranges:
        raise KeyError(f"intent {intent!r} not in {tuple(ranges)}")
    r = ranges[intent]
    dz = r.ur_dz_mm.sample(rng)
    dx, dy = r.ur_dxy_mm.sample(rng), r.ur_dxy_mm.sample(rng)
    tilt = r.ur_tilt_deg.sample(rng)
    if r.ur_tilt_sign_random and rng.random() < 0.5:
        tilt = -tilt
    franka = np.array([r.franka_dxyz_mm.sample(rng) for _ in range(3)])
    franka_tilt = r.franka_tilt_deg.sample(rng)
    return Perturbation(intent=intent, ur_dpos_m=np.array([dx, dy, dz]) * 1e-3,
                        ur_tilt_deg=tilt, franka_dpos_m=franka * 1e-3,
                        franka_tilt_deg=franka_tilt)


def scale_tilt(p: Perturbation, scale: float) -> Perturbation:
    """``p`` with both tilts scaled (the clearance guard's last resort; 1.0 returns ``p``)."""
    if scale == 1.0:
        return p
    return replace(p, ur_tilt_deg=p.ur_tilt_deg * scale,
                   franka_tilt_deg=p.franka_tilt_deg * scale)


def belt_tangent(waypoints: list[Waypoint]) -> np.ndarray:
    """Unit horizontal Franka -> UR direction at the nominal ``pre_place_2``."""
    wp_ = next((w for w in waypoints if w.label == TANGENT_LABEL), None)
    if wp_ is None or wp_.ur_pos is None:
        raise ValueError(f"no {TANGENT_LABEL} waypoint with a UR pose")
    d = np.asarray(wp_.ur_pos, float) - np.asarray(wp_.franka_pos, float)
    d[2] = 0.0
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        raise ValueError(f"{TANGENT_LABEL}: Franka and UR tips coincide horizontally")
    return d / n


def _moved(pos, quat_xyzw, dpos, tilt_deg: float, tangent: np.ndarray):
    rot = rot_axis_angle(tangent, math.radians(tilt_deg)) @ quat_xyzw_to_mat3(quat_xyzw)
    return np.asarray(pos, float) + np.asarray(dpos, float), mat3_to_quat_xyzw(rot)


def apply(waypoints: list[Waypoint], p: Perturbation, tangent: np.ndarray) -> list[Waypoint]:
    """Copies of ``waypoints`` with ``pre_place_2`` / ``place_3`` moved and tilted per arm."""
    out = []
    for w in waypoints:
        if w.label not in PERTURBED_LABELS:
            out.append(w)
            continue
        f_pos, f_quat = _moved(w.franka_pos, w.franka_quat_xyzw, p.franka_dpos_m,
                               p.franka_tilt_deg, tangent)
        u_pos, u_quat = w.ur_pos, w.ur_quat_xyzw
        if u_pos is not None:
            u_pos, u_quat = _moved(u_pos, u_quat, p.ur_dpos_m, p.ur_tilt_deg, tangent)
        out.append(replace(w, franka_pos=f_pos, franka_quat_xyzw=f_quat, ur_pos=u_pos,
                           ur_quat_xyzw=u_quat))
    return out
