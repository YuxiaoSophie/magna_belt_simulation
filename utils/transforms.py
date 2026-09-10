"""Drake/Warp transform conventions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import warp as wp

def quat_from_rpy(roll: float, pitch: float, yaw: float) -> wp.quat:
    """Roll-pitch-yaw (rad) to a Warp (x, y, z, w) quaternion.

    Copied verbatim from ``round_belt.quat_from_rpy`` so ``utils/`` need not import that scene
    script (2,703 lines, pulling in every solver) for nine lines of arithmetic.
    """
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return wp.quat(qx, qy, qz, qw)

# Drake's ``!Rpy { deg: [r, p, y] }`` and URDF's ``<origin rpy>`` are the same
# convention: R = Rz(yaw) . Ry(pitch) . Rx(roll)  (extrinsic X-Y-Z).
# ``quat_from_rpy`` computes exactly that and returns a Warp (x, y, z, w)
# quaternion. Nothing here may use ``wp.quat_rpy`` or any other library RPY helper
# without first proving it against the matrix below.


def _assert_rpy_convention() -> None:
    """Import-time proof that quat_from_rpy(r, p, y) == Rz(y) . Ry(p) . Rx(r).

    Uses the (deliberately ugly) board rotation from the Drake scene so the
    check exercises all three non-trivial angles at once.
    """
    r, p, y = (math.radians(a) for a in (-0.332822058, -0.0687450103, 89.5207485))

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    expected = rz @ ry @ rx

    q = quat_from_rpy(r, p, y)
    got = np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)

    # atol=1e-6, rtol=0: float32 quats land ~1.3e-7 off the float64 reference, so 1e-9
    # would be a false precision claim; a swapped Rx/Rz order still gives 7e-3. rtol=0
    # so the tolerance is the stated absolute one, not np.allclose's default rtol=1e-5.
    if not np.allclose(got, expected, atol=1.0e-6, rtol=0.0):
        raise AssertionError(
            "Drake rpy -> Warp quaternion conversion is wrong.\n"
            f"quat (x,y,z,w) = {tuple(float(v) for v in q)}\n"
            f"quat_to_matrix =\n{got}\nRz@Ry@Rx =\n{expected}\n"
            f"max abs error = {np.abs(got - expected).max():.3e}"
        )

    # Cheap sanity anchors: pure single-axis rotations.
    c, s = math.cos(0.37), math.sin(0.37)
    references = {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64),
    }
    for idx, (axis, reference) in enumerate(references.items()):
        angles = [0.0, 0.0, 0.0]
        angles[idx] = 0.37
        q1 = quat_from_rpy(*angles)
        m1 = np.array(wp.quat_to_matrix(q1), dtype=np.float64).reshape(3, 3)
        if not np.allclose(m1, reference, atol=1.0e-6, rtol=0.0):  # float32 quats; see above
            raise AssertionError(f"quat_from_rpy single-axis {axis} rotation mismatch")


_assert_rpy_convention()


def drake_xform(xyz: Sequence[float], rpy_deg: Sequence[float]) -> wp.transform:
    """Build a Warp transform from a Drake ``translation`` + ``!Rpy { deg: ... }``."""
    return wp.transform(
        wp.vec3(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        quat_from_rpy(*(math.radians(float(a)) for a in rpy_deg)),
    )


def rpy_deg_from_quat(q: Sequence[float]) -> tuple[float, float, float]:
    """Inverse of quat_from_rpy: extract Rz.Ry.Rx angles in degrees."""
    quat = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    m = np.array(wp.quat_to_matrix(quat), dtype=np.float64).reshape(3, 3)
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(m[2, 0]) < 0.999999:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll = math.atan2(-m[1, 2], m[1, 1])
        yaw = 0.0
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))
