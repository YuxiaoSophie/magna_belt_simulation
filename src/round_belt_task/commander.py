"""Socket-free emulation of magna's pre-MPC waypoint commander (``assembly_controller.cc``).

:class:`FrankaWaypointCommander` reproduces the reach/latch/dwell/advance state machine and the
7-knot ``TARGET_CARTESIAN_POSE_TRAJECTORY`` built from the MEASURED pose each tick;
:class:`UrLineCommander` reproduces ``HandleURRobot``'s 2-knot tool0 line and its regeneration rule.
Quaternions are ``wxyz`` throughout. Additions: an optional bounded excitation of Franka knots
1..n-1 (:class:`Excite`; white per-sample draws or a smooth :class:`OuExcitation`). One
difference: gripper commands take over on latch, also for dwell-0 targets; magna only sets them
in the hold branch, which a dwell-0 target never reaches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from dairlib import (
    lcmt_metadata,
    lcmt_saved_traj,
    lcmt_timestamped_saved_traj,
    lcmt_trajectory_block,
)
from task_common.lcs_dataset import delta_rotvec

BLOCK_NAMES = ("end_effector_position_target", "end_effector_orientation_target",
               "end_effector_force_target")
METADATA_NAME = "lcs_collector"
UR10_TOOL0_JOINTS = ("wrist_3-flange", "flange-tool0")


# --- quaternion helpers (wxyz, Eigen semantics) -------------------------------------------------

def _qn(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _qmul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def _qconj(q) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def angular_distance(q0, q1) -> float:
    """Eigen ``angularDistance``: ``2 atan2(|vec(q1 q0*)|, |w|)`` in ``[0, pi]``."""
    d = _qmul(_qn(q1), _qconj(_qn(q0)))
    return 2.0 * math.atan2(float(np.linalg.norm(d[1:])), abs(float(d[0])))


def slerp(q0, q1, s: float) -> np.ndarray:
    """Eigen ``QuaternionBase::slerp`` (shortest path, linear near-parallel fallback)."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    abs_d = abs(d)
    if abs_d >= 1.0 - np.finfo(np.float64).eps:
        s0, s1 = 1.0 - s, s
    else:
        theta = math.acos(abs_d)
        sin_theta = math.sin(theta)
        s0, s1 = math.sin((1.0 - s) * theta) / sin_theta, math.sin(s * theta) / sin_theta
    if d < 0.0:
        s1 = -s1
    return s0 * q0 + s1 * q1


def quat_axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(0.5 * angle)], math.sin(0.5 * angle) * a])


def quat_to_mat3(q) -> np.ndarray:
    w, x, y, z = _qn(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat3_to_quat(m) -> np.ndarray:
    """Unit ``wxyz`` with ``w >= 0``."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    q = _qn(q)
    return -q if q[0] < 0.0 else q


def pose_mat(pos, quat_wxyz) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_mat3(quat_wxyz)
    m[:3, 3] = np.asarray(pos, dtype=np.float64)
    return m


def _inv(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


# --- targets and params -------------------------------------------------------------------------

@dataclass(frozen=True)
class FrankaTarget:
    label: str
    pos: np.ndarray
    quat_wxyz: np.ndarray
    hand_mm: float | None
    dwell_s: float


@dataclass(frozen=True)
class UrTarget:
    pos: np.ndarray
    quat_wxyz: np.ndarray
    byte: int | None


def _xyzw_to_wxyz(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([w, x, y, z])


def targets_from_waypoints(waypoints) -> tuple[list[FrankaTarget], list[UrTarget | None]]:
    """``Waypoint`` list (world frame, xyzw) -> Franka / UR targets (wxyz)."""
    franka, ur = [], []
    for wp in waypoints:
        franka.append(FrankaTarget(
            label=str(wp.label), pos=np.asarray(wp.franka_pos, dtype=np.float64).copy(),
            quat_wxyz=_xyzw_to_wxyz(wp.franka_quat_xyzw), hand_mm=wp.franka_gripper_mm,
            dwell_s=float(wp.dwell_s)))
        if wp.ur_pos is None:
            ur.append(None)
        else:
            ur.append(UrTarget(pos=np.asarray(wp.ur_pos, dtype=np.float64).copy(),
                               quat_wxyz=_xyzw_to_wxyz(wp.ur_quat_xyzw),
                               byte=wp.ur_gripper_byte))
    return franka, ur


@dataclass(frozen=True)
class CommanderParams:
    """magna defaults: C3 ``dt``/``N`` and ``round_belt_controller_params_sim.yaml:195-200``."""

    dt: float = 0.075
    n_knots: int = 7
    lin_speed: float = 0.08
    ang_speed: float = 0.5
    pos_tol: float = 0.0055
    ori_tol: float = 0.15
    ur_lin_speed: float = 0.08
    ur_ang_speed: float = 0.5
    ur_min_duration: float = 0.5  # kUrControlDt, assembly_controller.cc:25
    ur_regen_end_pos: float = 1e-4
    ur_regen_end_ori: float = 1e-3
    ur_at_target_pos: float = 5e-3
    ur_at_target_ori: float = 0.1
    ur_deviation_pos: float = 2.5e-2
    ur_deviation_ori: float = 0.1


# --- excitation ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Excite:
    """World-frame offset of Franka knots 1..n-1: ``dpos`` and a left rotation ``(axis, angle)``."""

    dpos: np.ndarray
    axis: np.ndarray
    angle: float
    cap_factor: float
    down_m: float


def draw_excite(rng: np.random.Generator, pos_radius_m: float, rot_max_rad: float,
                cap_factor: float, down_m: float) -> Excite:
    """``dpos`` uniform in the ball, uniform random axis, ``angle ~ U(0, rot_max)``."""
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    dpos = direction * pos_radius_m * float(rng.uniform()) ** (1.0 / 3.0)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Excite(dpos=dpos, axis=axis, angle=float(rng.uniform(0.0, rot_max_rad)),
                  cap_factor=float(cap_factor), down_m=float(down_m))


EXCITE_MODES = ("ou", "white")
EXCITE_RAMP_S = 0.3


def excite_from_rotvec(dpos, rotvec, cap_factor: float, down_m: float) -> Excite:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    axis = rotvec / angle if angle > 0.0 else np.array([1.0, 0.0, 0.0])
    return Excite(dpos=np.asarray(dpos, dtype=np.float64).copy(), axis=axis, angle=angle,
                  cap_factor=float(cap_factor), down_m=float(down_m))


class OuExcitation:
    """Ornstein-Uhlenbeck offset ``x = (dpos, rotvec)``, starting at 0, stepped once per sample.

    ``x <- a x + sqrt(1 - a^2) sigma xi`` with ``a = exp(-dt / tau)``: each component's stationary
    std is ``sigma`` (``pos_std_m`` / ``rot_std_rad``) for any ``dt``. :meth:`fade` switches it off
    by a linear ramp of the held offset to 0 over ``ramp_s`` (no new noise, no jump).
    """

    def __init__(self, rng: np.random.Generator, pos_std_m: float, rot_std_rad: float,
                 tau_s: float, dt: float, cap_factor: float, down_m: float,
                 ramp_s: float = EXCITE_RAMP_S) -> None:
        if tau_s <= 0.0 or dt <= 0.0:
            raise ValueError(f"tau_s {tau_s} and dt {dt} must be > 0")
        self.rng = rng
        self.a = math.exp(-dt / tau_s)
        self.sigma = np.array([pos_std_m] * 3 + [rot_std_rad] * 3, dtype=np.float64)
        self.cap_factor, self.down_m, self.ramp_s = float(cap_factor), float(down_m), ramp_s
        self.x = np.zeros(6)
        self.t_off: float | None = None

    def step(self) -> Excite:
        noise = self.rng.normal(size=6)
        self.x = self.a * self.x + math.sqrt(1.0 - self.a * self.a) * self.sigma * noise
        return self.excite()

    def excite(self, scale: float = 1.0) -> Excite:
        return excite_from_rotvec(scale * self.x[:3], scale * self.x[3:], self.cap_factor,
                                  self.down_m)

    def fade(self, t: float) -> Excite | None:
        """The held offset ramped to 0 from the first call's ``t``; None once the ramp is over."""
        if self.t_off is None:
            self.t_off = float(t)
        s = 1.0 - (t - self.t_off) / self.ramp_s if self.ramp_s > 0.0 else 0.0
        return self.excite(s) if s > 1e-9 else None  # float slack at t_off + ramp_s


def _step_scale(a: np.ndarray, d: np.ndarray, limit: float) -> float:
    """Largest ``s`` in ``[0, 1]`` with ``|a + s d| <= limit`` (0 if ``|a|`` already exceeds)."""
    if np.linalg.norm(a + d) <= limit:
        return 1.0
    aa, ad, dd = float(a @ a), float(a @ d), float(d @ d)
    if aa > limit * limit or dd == 0.0:
        return 0.0
    return min(1.0, max(0.0, (-ad + math.sqrt(ad * ad - dd * (aa - limit * limit))) / dd))


def _apply_excite(pos: np.ndarray, quat: np.ndarray, ex: Excite, params: CommanderParams
                  ) -> tuple[np.ndarray, np.ndarray, Excite]:
    # z floor on the shared offset first, then scale: scaling towards 0 keeps the floor.
    dpos = np.asarray(ex.dpos, dtype=np.float64).copy()
    dpos[2] = max(dpos[2], -ex.down_m)
    dpos *= _step_scale(pos[1] - pos[0], dpos, params.lin_speed * params.dt * ex.cap_factor)

    ang_cap = params.ang_speed * params.dt * ex.cap_factor

    def step_angle(s: float) -> float:
        return angular_distance(quat[0], _qmul(quat_axis_angle(ex.axis, s * ex.angle), quat[1]))

    if ex.angle == 0.0 or step_angle(1.0) <= ang_cap:
        s_rot = 1.0
    elif step_angle(0.0) > ang_cap:
        s_rot = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if step_angle(mid) <= ang_cap else (lo, mid)
        s_rot = lo
    angle = s_rot * ex.angle
    q_ex = quat_axis_angle(ex.axis, angle)

    pos, quat = pos.copy(), quat.copy()
    pos[1:] += dpos
    for i in range(1, len(quat)):
        quat[i] = _qn(_qmul(q_ex, quat[i]))
    return pos, quat, replace(ex, dpos=dpos, angle=angle)


# --- Franka -------------------------------------------------------------------------------------

@dataclass
class FrankaCommand:
    knots_pos: np.ndarray
    knots_quat: np.ndarray
    times: np.ndarray
    hold: bool
    target_index: int
    phase: str
    hand_mm: float | None
    excite_applied: Excite | None = None
    latched_index: int | None = None  # target latched THIS tick (UR byte wiring)


class FrankaWaypointCommander:
    """``assembly_controller.cc`` pre-MPC waypoint phase for the Franka ``finger_tip``."""

    def __init__(self, targets: list[FrankaTarget], params: CommanderParams | None = None,
                 hand_mm: float | None = None) -> None:
        if not targets:
            raise ValueError("no targets")
        self.targets = list(targets)
        self.params = params or CommanderParams()
        self.index = 0
        self.latched = False
        self.reached_pos: np.ndarray | None = None
        self.reached_quat: np.ndarray | None = None
        self.reached_time = -1.0
        self.hand_mm = hand_mm

    def is_reached(self, meas_pos, meas_quat_wxyz, ur_pos_err: float, ur_ori_err: float) -> bool:
        """``IsOSCTargetReached`` (:960-1079); pass UR errors 0 when the target has no UR pose."""
        p, target = self.params, self.targets[self.index]
        dot = min(1.0, abs(float(np.dot(_qn(meas_quat_wxyz), _qn(target.quat_wxyz)))))
        return (float(np.linalg.norm(np.asarray(meas_pos) - target.pos)) < p.pos_tol
                and ur_pos_err < p.pos_tol
                and 2.0 * math.acos(dot) < p.ori_tol
                and ur_ori_err < p.ori_tol)

    def tick(self, t: float, meas_pos, meas_quat_wxyz, ur_pos_err: float, ur_ori_err: float,
             excite: Excite | None = None) -> FrankaCommand:
        p = self.params
        meas_pos = np.asarray(meas_pos, dtype=np.float64)
        meas_quat = _qn(meas_quat_wxyz)
        last = len(self.targets) - 1
        latched_index = None

        # magna tick order (:734-770): reach -> advance, then generate (:884-888) for the new index.
        if not self.latched and self.is_reached(meas_pos, meas_quat, ur_pos_err, ur_ori_err):
            self.latched = True
            self.reached_pos, self.reached_quat = meas_pos.copy(), meas_quat.copy()
            self.reached_time = float(t)
            latched_index = self.index
            if self.targets[self.index].hand_mm is not None:
                self.hand_mm = self.targets[self.index].hand_mm
        dwell_done = False
        if self.latched:
            dwell = self.targets[self.index].dwell_s
            dwell_done = dwell <= 0.0 or t - self.reached_time >= dwell  # :458-489
            if dwell_done and self.index < last:
                self.index += 1
                self.latched = False
                self.reached_time = -1.0

        target = self.targets[self.index]
        if self.latched:
            pos = np.stack([self.reached_pos, self.reached_pos])
            quat = np.stack([self.reached_quat, self.reached_quat])
            times = np.array([t, t + p.dt])
            phase = "done" if dwell_done else f"hold:{target.label}"
        else:
            pos, quat, times = self._move_knots(t, meas_pos, meas_quat, target)
            phase = f"move:{target.label}"

        applied = None
        if excite is not None:
            pos, quat, applied = _apply_excite(pos, quat, excite, p)
        return FrankaCommand(knots_pos=pos, knots_quat=quat, times=times, hold=self.latched,
                             target_index=self.index, phase=phase, hand_mm=self.hand_mm,
                             excite_applied=applied, latched_index=latched_index)

    def _move_knots(self, t: float, meas_pos: np.ndarray, meas_quat: np.ndarray,
                    target: FrankaTarget) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``GenerateMoveToTargetTrajectory`` non-hold branch (:1205-1385), linear only."""
        p = self.params
        n = p.n_knots
        i = np.arange(n, dtype=np.float64)
        times = t + i * p.dt
        d = target.pos - meas_pos
        dist = float(np.linalg.norm(d))
        if dist < p.pos_tol:
            pos = np.tile(target.pos, (n, 1))
        else:
            pos = meas_pos + np.minimum(i * p.lin_speed * p.dt, dist)[:, None] * (d / dist)
        q_t = _qn(target.quat_wxyz)
        if float(np.dot(meas_quat, q_t)) < 0.0:
            q_t = -q_t
        ang = angular_distance(meas_quat, q_t)
        t_ang = ang / p.ang_speed
        quat = np.empty((n, 4))
        for k in range(n):
            s = min(k * p.dt / t_ang, 1.0) if ang > 1e-12 else 1.0
            quat[k] = _qn(slerp(meas_quat, q_t, s))
        return pos, quat, times


def knot_action(cmd: FrankaCommand) -> tuple[np.ndarray, np.ndarray]:
    """``(knot1 - knot0, delta_rotvec(q0, q1))``."""
    return (cmd.knots_pos[1] - cmd.knots_pos[0],
            delta_rotvec(cmd.knots_quat[0], cmd.knots_quat[1]))


# --- UR -----------------------------------------------------------------------------------------

def x_tool0_tracking(urdf: Path | None = None) -> np.ndarray:
    """``X_wrist3_tool0^-1 * X_wrist3_tracking`` (magna's ``ur_tracking_frame_wrt_flange_``)."""
    from round_belt_task.arm_kinematics import X_WRIST3_TRACKING, urdf_fixed_transform
    from round_belt_task.constants import UR10_URDF

    urdf = UR10_URDF if urdf is None else Path(urdf)
    x_wrist3_tool0 = np.eye(4)
    for joint in UR10_TOOL0_JOINTS:
        x_wrist3_tool0 = x_wrist3_tool0 @ urdf_fixed_transform(urdf, joint)
    return _inv(x_wrist3_tool0) @ X_WRIST3_TRACKING


@dataclass(frozen=True)
class UrLine:
    """``HandleURRobot`` 2-knot line on ``tool0`` (world frame)."""

    p0: np.ndarray
    q0: np.ndarray
    t0: float
    p1: np.ndarray
    q1: np.ndarray
    t1: float

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Position FOH + slerp at ``clamp(t, t0, t1)`` (``ur_cartesian_pose_generator.cc``)."""
        span = self.t1 - self.t0
        s = 1.0 if span <= 0.0 else (min(max(t, self.t0), self.t1) - self.t0) / span
        return (1.0 - s) * self.p0 + s * self.p1, _qn(slerp(self.q0, self.q1, s))


@dataclass
class UrCommand:
    line: UrLine
    regenerated: bool
    byte: int | None
    t: float
    X_tool0_tracking: np.ndarray = field(repr=False)

    def pose_at(self, t2: float) -> np.ndarray:
        """Tracking frame in world on the line at ``clamp(t2, t0, t1)``."""
        p, q = self.line.sample(t2)
        return pose_mat(p, q) @ self.X_tool0_tracking

    @property
    def pose_4x4(self) -> np.ndarray:
        return self.pose_at(self.t)


class UrLineCommander:
    """``HandleURRobot`` + ``ShouldUpdateURTrajectory`` (:1390-1661) for the UR tracking frame."""

    def __init__(self, targets: list[UrTarget | None], params: CommanderParams | None = None,
                 X_tool0_tracking: np.ndarray | None = None, byte: int | None = None) -> None:
        self.targets = list(targets)
        self.params = params or CommanderParams()
        self.X_tool0_tracking = (x_tool0_tracking() if X_tool0_tracking is None
                                 else np.asarray(X_tool0_tracking, dtype=np.float64))
        self._X_tracking_tool0 = _inv(self.X_tool0_tracking)
        self.line: UrLine | None = None
        self.target_tool0: tuple[np.ndarray, np.ndarray] | None = None
        self.byte = byte

    def on_latch(self, index: int | None) -> None:
        """Franka latched ``targets[index]``: its UR byte (if any) takes over (:1420-1422)."""
        if index is None or self.targets[index] is None:
            return
        if self.targets[index].byte is not None:
            self.byte = self.targets[index].byte

    def to_tool0(self, X_tracking: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(X_tracking, dtype=np.float64) @ self._X_tracking_tool0
        return m[:3, 3].copy(), mat3_to_quat(m[:3, :3])

    def should_update(self, t: float, cur_p, cur_q, tgt_p, tgt_q) -> bool:
        """``ShouldUpdateURTrajectory`` verbatim (:1535-1661)."""
        p, line = self.params, self.line
        if line is None:
            return True
        if (np.linalg.norm(tgt_p - line.p1) > p.ur_regen_end_pos
                or angular_distance(line.q1, tgt_q) > p.ur_regen_end_ori):
            return True
        if (np.linalg.norm(cur_p - tgt_p) < p.ur_at_target_pos
                and angular_distance(tgt_q, cur_q) < p.ur_at_target_ori):
            return False
        if t <= line.t0 or t >= line.t1:
            return True
        lp, lq = line.sample(t)
        return bool(np.linalg.norm(cur_p - lp) > p.ur_deviation_pos
                    or angular_distance(lq, cur_q) > p.ur_deviation_ori)

    def tick(self, t: float, meas_tracking_4x4: np.ndarray, target: UrTarget | None
             ) -> UrCommand:
        p = self.params
        cur_p, cur_q = self.to_tool0(meas_tracking_4x4)
        if target is not None:
            self.target_tool0 = self.to_tool0(pose_mat(target.pos, target.quat_wxyz))
        # None keeps the previous target; with none yet, magna targets the current pose.
        tgt_p, tgt_q = self.target_tool0 if self.target_tool0 is not None else (cur_p, cur_q)
        regenerated = self.should_update(t, cur_p, cur_q, tgt_p, tgt_q)
        if regenerated:
            duration = max(float(np.linalg.norm(tgt_p - cur_p)) / p.ur_lin_speed,
                           angular_distance(cur_q, tgt_q) / p.ur_ang_speed, p.ur_min_duration)
            self.line = UrLine(p0=cur_p, q0=cur_q, t0=float(t), p1=tgt_p.copy(),
                               q1=tgt_q.copy(), t1=float(t) + duration)
        return UrCommand(line=self.line, regenerated=regenerated, byte=self.byte, t=float(t),
                         X_tool0_tracking=self.X_tool0_tracking)


def line_action(ur: UrCommand, t: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Tracking-frame ``pose_at(t + dt) - pose_at(t)`` as ``(dxyz, drotvec)``."""
    a, b = ur.pose_at(t), ur.pose_at(t + dt)
    return b[:3, 3] - a[:3, 3], delta_rotvec(mat3_to_quat(a[:3, :3]), mat3_to_quat(b[:3, :3]))


# --- LCM message --------------------------------------------------------------------------------

def _block(name: str, data: np.ndarray, times: np.ndarray) -> lcmt_trajectory_block:
    block = lcmt_trajectory_block()
    block.trajectory_name = name
    block.num_points = int(data.shape[1])
    block.num_datatypes = int(data.shape[0])
    block.time_vec = [float(v) for v in times]
    block.datapoints = [[float(v) for v in row] for row in data]
    block.datatypes = ["double"] * int(data.shape[0])
    return block


def saved_traj_message(t_utime: int, knots_pos, knots_quat_wxyz, times
                       ) -> lcmt_timestamped_saved_traj:
    """magna's ``AddEETrajectoriesToLcm`` layout (:1107-1146), zero force block."""
    pos = np.asarray(knots_pos, dtype=np.float64)
    quat = np.asarray(knots_quat_wxyz, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    if pos.shape != (n, 3) or quat.shape != (n, 4):
        raise ValueError(f"knot shapes {pos.shape} / {quat.shape} do not match {n} times")
    # Receivers ignore utime <= 1e-3; the mode switch drops a trajectory starting exactly at 0.
    if int(t_utime) <= 0 or times[0] == 0.0:
        raise ValueError(f"utime {t_utime} must be > 0 and time_vec[0] {times[0]} != 0")
    meta = lcmt_metadata()
    meta.git_dirty_flag = False
    meta.datetime = ""
    meta.name = METADATA_NAME
    meta.description = ""
    meta.git_commit_hash = ""
    blocks = [_block(BLOCK_NAMES[0], pos.T, times), _block(BLOCK_NAMES[1], quat.T, times),
              _block(BLOCK_NAMES[2], np.zeros((3, n)), times)]
    traj = lcmt_saved_traj()
    traj.metadata = meta
    traj.num_trajectories = len(blocks)
    traj.trajectories = blocks
    traj.trajectory_names = list(BLOCK_NAMES)
    msg = lcmt_timestamped_saved_traj()
    msg.utime = int(t_utime)
    msg.saved_traj = traj
    return msg


def parse_saved_traj_message(msg: lcmt_timestamped_saved_traj
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(knots_pos (n,3), knots_quat_wxyz (n,4), times (n,))``."""
    blocks = {b.trajectory_name: b for b in msg.saved_traj.trajectories}
    pos_block, quat_block = blocks[BLOCK_NAMES[0]], blocks[BLOCK_NAMES[1]]
    return (np.asarray(pos_block.datapoints, dtype=np.float64).T,
            np.asarray(quat_block.datapoints, dtype=np.float64).T,
            np.asarray(pos_block.time_vec, dtype=np.float64))
