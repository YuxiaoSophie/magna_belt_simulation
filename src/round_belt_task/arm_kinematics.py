"""Numpy forward kinematics, geometric Jacobians and damped least-squares IK for the two arms.

:class:`UrdfChain` walks a URDF's parent/child graph once (joint origins, axes, limits);
:data:`FrankaTip` is ``panda_link0 -> panda_link8`` * ``X_LINK8_HAND`` * ``finger_tip`` (magna's
Franka pose frame) and :data:`UrTracking` is ``X_W_UR10`` * ``base_link -> wrist_3_link`` *
magna's ``tracking_frame`` (``run_round_belt_assembly_controller.cc``). Both are world frame;
``panda_link0`` is welded at world identity. Rotations use the URDF/Drake RPY ``Rz.Ry.Rx``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from round_belt_task.constants import (
    LCM_UR_GRIPPER_TIP_Z,
    PANDA_ARM_URDF,
    PANDA_HAND_URDF,
    UR10_URDF,
    X_LINK8_HAND,
    X_USDWRIST3_URDFWRIST3,
    X_W_PANDA,
    X_W_UR10,
)
from task_common.lcs_dataset import POSE_LAYOUT


def rpy_to_mat3(rpy) -> np.ndarray:
    """``Rz(yaw) . Ry(pitch) . Rx(roll)``."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), \
        math.cos(y), math.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def mat4(rot: np.ndarray | None = None, pos=None) -> np.ndarray:
    m = np.eye(4)
    if rot is not None:
        m[:3, :3] = rot
    if pos is not None:
        m[:3, 3] = np.asarray(pos, dtype=np.float64)
    return m


def rot_axis_angle(axis, angle: float) -> np.ndarray:
    """Rodrigues: rotation of ``angle`` about the unit ``axis``."""
    x, y, z = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def quat_xyzw_to_mat3(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat3_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """Unit quaternion ``[x, y, z, w]`` with ``w >= 0``."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def rotvec(m: np.ndarray) -> np.ndarray:
    """Axis * angle of the rotation matrix ``m`` (angle in ``[0, pi]``)."""
    q = mat3_to_quat_xyzw(m)
    s = float(np.linalg.norm(q[:3]))
    if s < 1e-12:
        return np.zeros(3)
    return q[:3] / s * 2.0 * math.atan2(s, q[3])


def rot_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle [rad] between the rotation parts of two 3x3/4x4 matrices."""
    return float(np.linalg.norm(rotvec(a[:3, :3] @ b[:3, :3].T)))


def wp_transform_to_mat4(t) -> np.ndarray:
    """A warp transform or 7-vector ``[xyz, qx, qy, qz, qw]`` as a 4x4."""
    v = [float(x) for x in t]
    return mat4(quat_xyzw_to_mat3(v[3:7]), v[:3])


def pose7_from_mat(m: np.ndarray, layout: str = POSE_LAYOUT) -> np.ndarray:
    """``[x, y, z, quat]`` with the quaternion in ``layout`` (``xyz_wxyz`` or ``xyz_xyzw``)."""
    q = mat3_to_quat_xyzw(m[:3, :3])
    if layout == "xyz_wxyz":
        q = np.array([q[3], q[0], q[1], q[2]])
    elif layout != "xyz_xyzw":
        raise ValueError(f"unknown pose layout {layout!r}")
    return np.concatenate([np.asarray(m[:3, 3], dtype=np.float64), q])


def _vec3(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.asarray(default if text is None else [float(v) for v in text.split()], float)


def _origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    if origin is None:
        return np.eye(4)
    return mat4(rpy_to_mat3(_vec3(origin.get("rpy"))), _vec3(origin.get("xyz")))


def urdf_fixed_transform(urdf: Path, joint_name: str) -> np.ndarray:
    """The origin of one (fixed) joint of ``urdf``."""
    for joint in ET.parse(urdf).getroot().findall("joint"):
        if joint.get("name") == joint_name:
            return _origin(joint)
    raise KeyError(f"{urdf}: no joint {joint_name!r}")


@dataclass
class _Joint:
    name: str
    origin: np.ndarray  # parent-link -> joint frame (fixed joints folded in)
    axis: np.ndarray
    lower: float
    upper: float


class UrdfChain:
    """``X_pre * (root_link -> tip_link) * X_post`` over the revolute joints of a URDF path."""

    def __init__(self, urdf: Path, root_link: str, tip_link: str,
                 X_pre: np.ndarray | None = None, X_post: np.ndarray | None = None) -> None:
        root = ET.parse(Path(urdf)).getroot()
        child_to_joint = {j.find("child").get("link"): j for j in root.findall("joint")}
        path = []
        link = tip_link
        while link != root_link:
            joint = child_to_joint.get(link)
            if joint is None:
                raise KeyError(f"{urdf}: no path {root_link!r} -> {tip_link!r} (at {link!r})")
            path.append(joint)
            link = joint.find("parent").get("link")
        path.reverse()
        self.joints: list[_Joint] = []
        pending = np.eye(4) if X_pre is None else np.asarray(X_pre, dtype=np.float64)
        for joint in path:
            pending = pending @ _origin(joint)
            kind = joint.get("type")
            if kind == "fixed":
                continue
            if kind not in ("revolute", "continuous"):
                raise ValueError(f"{urdf}: unsupported joint type {kind!r}")
            axis = _vec3(joint.find("axis").get("xyz") if joint.find("axis") is not None
                         else None, (1.0, 0.0, 0.0))
            limit = joint.find("limit")
            bounded = limit is not None and kind != "continuous"
            lower = float(limit.get("lower")) if bounded else -math.inf
            upper = float(limit.get("upper")) if bounded else math.inf
            self.joints.append(_Joint(joint.get("name"), pending, axis / np.linalg.norm(axis),
                                      lower, upper))
            pending = np.eye(4)
        self.X_post = pending @ (np.eye(4) if X_post is None else np.asarray(X_post, float))
        self.names = [j.name for j in self.joints]
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])
        self.n = len(self.joints)

    def frames(self, q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(origins (n,3), world axes (n,3), tip 4x4)``."""
        m = np.eye(4)
        origins = np.empty((self.n, 3))
        axes = np.empty((self.n, 3))
        for i, (joint, qi) in enumerate(zip(self.joints, np.asarray(q, dtype=np.float64))):
            m = m @ joint.origin
            origins[i] = m[:3, 3]
            axes[i] = m[:3, :3] @ joint.axis
            m = m @ mat4(rot_axis_angle(joint.axis, qi))
        return origins, axes, m @ self.X_post

    def fk(self, q) -> np.ndarray:
        return self.frames(q)[2]

    def jacobian(self, q) -> np.ndarray:
        """Geometric ``(6, n)`` Jacobian of the tip, world frame, linear rows first."""
        origins, axes, tip = self.frames(q)
        jac = np.empty((6, self.n))
        jac[:3] = np.cross(axes, tip[:3, 3] - origins).T
        jac[3:] = axes.T
        return jac


def ik(chain: UrdfChain, target: np.ndarray, q0, *, lam: float = 0.02, max_iters: int = 50,
       step_cap: float = 0.2, pos_tol: float = 1e-4, rot_tol: float = 1e-3
       ) -> tuple[np.ndarray, float, float, int]:
    """Damped least squares; ``(q, pos err [m], rot err [rad], iterations)``."""
    q = np.clip(np.asarray(q0, dtype=np.float64).copy(), chain.lower, chain.upper)
    target = np.asarray(target, dtype=np.float64)
    eye = np.eye(6) * lam * lam
    iters = 0
    while True:
        tip = chain.fk(q)
        e = np.concatenate([target[:3, 3] - tip[:3, 3], rotvec(target[:3, :3] @ tip[:3, :3].T)])
        err_pos, err_rot = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        if (err_pos <= pos_tol and err_rot <= rot_tol) or iters >= max_iters:
            return q, err_pos, err_rot, iters
        jac = chain.jacobian(q)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + eye, e)
        peak = float(np.abs(dq).max())
        if peak > step_cap:
            dq *= step_cap / peak
        q = np.clip(q + dq, chain.lower, chain.upper)
        iters += 1


X_HAND_TIP = urdf_fixed_transform(PANDA_HAND_URDF, "finger_tip_joint")
X_WRIST3_TRACKING = mat4(rpy_to_mat3((math.pi, 0.0, math.pi / 2)), (0.0, 0.0, LCM_UR_GRIPPER_TIP_Z))

FrankaTip = UrdfChain(PANDA_ARM_URDF, "panda_link0", "panda_link8",
                       X_pre=wp_transform_to_mat4(X_W_PANDA),
                       X_post=wp_transform_to_mat4(X_LINK8_HAND) @ X_HAND_TIP)
UrTracking = UrdfChain(UR10_URDF, "base_link", "wrist_3_link",
                        X_pre=wp_transform_to_mat4(X_W_UR10), X_post=X_WRIST3_TRACKING)
X_USDWRIST3_TRACKING = wp_transform_to_mat4(X_USDWRIST3_URDFWRIST3) @ X_WRIST3_TRACKING


def model_pose_franka_tip(body_q: np.ndarray, tip_body: int) -> np.ndarray:
    """Measured ``finger_tip`` 4x4 from a ``state.body_q`` host array."""
    return wp_transform_to_mat4(body_q[tip_body])


def model_pose_ur_tracking(body_q: np.ndarray, wrist_body: int) -> np.ndarray:
    """Measured UR tracking frame 4x4 from the USD ``wrist_3_link`` row of ``body_q``."""
    return wp_transform_to_mat4(body_q[wrist_body]) @ X_USDWRIST3_TRACKING
