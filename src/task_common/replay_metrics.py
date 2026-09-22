"""Metrics, derived events and target poses computed from a loaded :class:`Recording`.

The grasp gaps and hold rules match ``LcmBeltTaskSimulation._log_grasp`` and
``scripts/debug/summarize_e2e_logs.py``.
"""

from __future__ import annotations

import itertools
import weakref
from dataclasses import dataclass

import numpy as np
from loguru import logger

from task_common.recording import Event, Recording

HOLD_GAP_MM = 15.0
FRANKA_HOLD_WIDTH_MM = 10.0
FRANKA_MIN_WIDTH_MM = 1.0
UR_HOLD_BYTE = 200
UR_RELEASE_BYTE = 100
HOLD_DEBOUNCE_S = 0.2
TURNING_DEG = 3.0
TURNING_WINDOW_S = 1.0
PULLEY_NAMES = ("small", "large")
# Belt seat radius on the small / large pulley (docs/lcm-simulation.md §8).
PULLEY_SEAT_RADIUS_MM = (17.38, 50.73)
GROOVE_AXIAL_MM = 6.0
GROOVE_RADIAL_MM = 5.0
WRAP_MERGE_DEG = 30.0
TARGET_CLOCK_TOLERANCE_S = 0.5
FRANKA_BASE_BODY = "panda_arm/panda_link0"
UR_BASE_BODY = "/ur10/base_link"
DERIVED_KINDS = ("franka_hold_start", "franka_hold_end", "ur_hold_start", "ur_hold_end",
                 "ur_release", "ur_partial_release", "pulley_turning_start",
                 "pulley_turning_end")

# UR ``base`` = ``base_link`` rotated by pi about z (ur10.urdf ``base_link-base_fixed_joint``).
_Q_BASE_LINK_BASE = np.array([0.0, 0.0, 1.0, 0.0])


@dataclass
class Metrics:
    """Per-frame arrays (m,) or (m, k) at ``frame_time``; timing arrays (n,) at ``row_time``."""

    frame_step: np.ndarray
    frame_time: np.ndarray
    hand_width_mm: np.ndarray
    belt_tip_gap_mm: np.ndarray
    belt_pad_gap_mm: np.ndarray
    belt_ur_tip_gap_mm: np.ndarray
    robotiq_cmd_byte: np.ndarray
    belt_loop_mm: np.ndarray
    belt_loop_change_pct: np.ndarray
    pulley_angle_deg: np.ndarray
    pulley_wrap_deg: np.ndarray
    row_time: np.ndarray
    compute_ms: np.ndarray
    realtime_factor: np.ndarray


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` (..., 3) by xyzw quaternions ``q`` (..., 4), broadcasting."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _min_gap_mm(belt: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Min over belt bodies of the distance to ``points`` (m, 3), in mm."""
    return np.linalg.norm(belt - points[:, None, :], axis=2).min(axis=1) * 1e3


def _largest_arc_deg(azimuth_deg: np.ndarray) -> float:
    """Angular extent of the largest run of azimuths whose neighbour gaps are <= the merge gap."""
    if azimuth_deg.size == 0:
        return 0.0
    a = np.sort(np.mod(azimuth_deg, 360.0))
    gaps = np.append(np.diff(a), a[0] + 360.0 - a[-1])
    breaks = np.flatnonzero(gaps > WRAP_MERGE_DEG)
    if breaks.size == 0:
        return 360.0
    # Start right after a break so every run is contiguous in the rolled order.
    gaps = np.roll(gaps, -(int(breaks[0]) + 1))
    best = run = 0.0
    for g in gaps:
        if g > WRAP_MERGE_DEG:
            best, run = max(best, run), 0.0
        else:
            run += g
    return float(max(best, run))


def _pulley_wrap_deg(rec: Recording, belt: np.ndarray) -> np.ndarray:
    frames = belt.shape[0]
    wrap = np.zeros((frames, len(PULLEY_NAMES)))
    for k, body in enumerate(rec.meta["pulley_bodies"][:len(PULLEY_NAMES)]):
        q = rec.body_q[:, body, 3:7].astype(np.float64)
        centre = rec.body_q[:, body, :3].astype(np.float64)
        axis = quat_rotate(q, np.array([0.0, 0.0, 1.0]))
        x_dir = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        y_dir = np.cross(axis, x_dir)
        d = belt - centre[:, None, :]
        h = np.einsum("fbi,fi->fb", d, axis)
        radial = d - h[..., None] * axis[:, None, :]
        r_mm = np.linalg.norm(radial, axis=2) * 1e3
        in_groove = ((np.abs(h) * 1e3 <= GROOVE_AXIAL_MM)
                     & (np.abs(r_mm - PULLEY_SEAT_RADIUS_MM[k]) <= GROOVE_RADIAL_MM))
        azimuth = np.degrees(np.arctan2(np.einsum("fbi,fi->fb", radial, y_dir),
                                        np.einsum("fbi,fi->fb", radial, x_dir)))
        for f in np.flatnonzero(in_groove.any(axis=1)):
            wrap[f, k] = _largest_arc_deg(azimuth[f, in_groove[f]])
    return wrap


def compute_metrics(rec: Recording) -> Metrics:
    """All per-frame metrics of ``rec`` (vectorised over frames), NaN where undefined."""
    meta = rec.meta
    m = rec.frame_count
    if len(rec.step):
        rows = np.clip(rec.state_step - int(rec.step[0]), 0, len(rec.step) - 1)
    else:
        rows = np.zeros(m, dtype=np.int64)
    body_q = rec.body_q
    belt = body_q[:, meta["belt_bodies"], :3].astype(np.float64)

    def gap_to_body(body) -> np.ndarray:
        if body is None or belt.shape[1] == 0:
            return np.full(m, np.nan)
        return _min_gap_mm(belt, body_q[:, int(body), :3].astype(np.float64))

    pads = meta.get("gripper_pad_bodies") or []
    pad_gap = np.full((m, 2), np.nan)
    for k, body in enumerate(pads[:2]):
        pad_gap[:, k] = gap_to_body(body)

    wrist, tip_in_wrist = meta.get("ur_wrist_body"), meta.get("ur_tip_in_wrist")
    if wrist is None or tip_in_wrist is None or belt.shape[1] == 0:
        ur_tip_gap = np.full(m, np.nan)
    else:
        wq = body_q[:, int(wrist)].astype(np.float64)
        tip = wq[:, :3] + quat_rotate(wq[:, 3:7], np.asarray(tip_in_wrist, dtype=np.float64))
        ur_tip_gap = _min_gap_mm(belt, tip)

    if belt.shape[1]:
        loop = (np.linalg.norm(np.diff(belt, axis=1), axis=2).sum(axis=1)
                + np.linalg.norm(belt[:, 0] - belt[:, -1], axis=1)) * 1e3
        loop_pct = (loop / loop[0] - 1.0) * 100.0
    else:
        loop = loop_pct = np.full(m, np.nan)

    angle = np.full((m, 2), np.nan)
    for k, coord in enumerate(meta.get("pulley_coords", [])[:2]):
        q = np.unwrap(rec.signals["joint_q"][:, rec.coord_column(coord)].astype(np.float64))
        angle[:, k] = np.degrees(q[rows] - q[rows[0]])

    # Shift back one row so compute_ms[i] is the time of the step that produced row i.
    compute = np.full(len(rec.step), np.nan)
    compute[:-1] = rec.compute_ms[1:]
    rtf = np.full(len(rec.step), np.nan)
    if len(rec.step) > 1:
        with np.errstate(divide="ignore"):
            rtf[1:] = rec.control_dt / np.diff(rec.wall_time)
    return Metrics(
        frame_step=rec.state_step.astype(np.int64),
        frame_time=rec.state_step * rec.control_dt,
        hand_width_mm=rec.hand_width_mm()[rows].astype(np.float64),
        belt_tip_gap_mm=gap_to_body(meta.get("finger_tip_body")),
        belt_pad_gap_mm=pad_gap,
        belt_ur_tip_gap_mm=ur_tip_gap,
        robotiq_cmd_byte=rec.signals["robotiq_cmd"][rows, 0].astype(np.uint8),
        belt_loop_mm=loop,
        belt_loop_change_pct=loop_pct,
        pulley_angle_deg=angle,
        pulley_wrap_deg=(_pulley_wrap_deg(rec, belt) if belt.shape[1]
                         else np.zeros((m, 2))),
        row_time=rec.step * rec.control_dt,
        compute_ms=compute,
        realtime_factor=rtf,
    )


def _debounced(flags: np.ndarray, n: int) -> list[tuple[int, int]]:
    """``(start, end)`` frame spans where ``flags`` holds, ignoring runs shorter than ``n``."""
    spans, state, start = [], False, 0
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(flags.astype(np.int8))) + 1,
                             [len(flags)]))
    for a, b in itertools.pairwise(bounds):
        value = bool(flags[a])
        if value != state and b - a >= n:
            if value:
                start = int(a)
            else:
                spans.append((start, int(a)))
            state = value
    if state:
        spans.append((start, len(flags)))
    return spans


def derive_events(rec: Recording, metrics: Metrics) -> list[Event]:
    """Hold / release / pulley-turning events from the summarizer's rules, sorted by step."""
    steps, times = metrics.frame_step, metrics.frame_time
    m = len(steps)
    if m == 0:
        return []
    frame_dt = float(rec.meta.get("state_every", 1)) * rec.control_dt
    debounce = max(1, round(HOLD_DEBOUNCE_S / frame_dt))
    events: list[Event] = []

    def add(frame: int, kind: str, **data) -> None:
        frame = min(frame, m - 1)
        events.append(Event(int(steps[frame]), float(times[frame]), kind,
                            {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in data.items()}))

    width, tip = metrics.hand_width_mm, metrics.belt_tip_gap_mm
    with np.errstate(invalid="ignore"):
        franka = ((tip <= HOLD_GAP_MM) & (width >= FRANKA_MIN_WIDTH_MM)
                  & (width <= FRANKA_HOLD_WIDTH_MM))
        byte = metrics.robotiq_cmd_byte.astype(np.int64)
        ur = (metrics.belt_ur_tip_gap_mm <= HOLD_GAP_MM) & (byte >= UR_HOLD_BYTE)
    for start, end in _debounced(franka, debounce):
        add(start, "franka_hold_start", width_mm=float(width[start]), gap_mm=float(tip[start]))
        if end < m:
            add(end, "franka_hold_end", width_mm=float(width[end]), gap_mm=float(tip[end]),
                held_s=float(times[end] - times[start]))
    ur_spans = _debounced(ur, debounce)
    ur_gap = metrics.belt_ur_tip_gap_mm
    for start, end in ur_spans:
        add(start, "ur_hold_start", byte=int(byte[start]), gap_mm=float(ur_gap[start]))
        if end < m:
            add(end, "ur_hold_end", byte=int(byte[end]), gap_mm=float(ur_gap[end]),
                held_s=float(times[end] - times[start]))
    held = np.zeros(m, dtype=bool)
    for start, end in ur_spans:
        held[start:end] = True
    for f in np.flatnonzero(np.diff(byte) != 0) + 1:
        if held[f - 1] and byte[f] < UR_HOLD_BYTE:
            kind = "ur_release" if byte[f] < UR_RELEASE_BYTE else "ur_partial_release"
            add(int(f), kind, byte=int(byte[f]), previous=int(byte[f - 1]),
                gap_mm=float(ur_gap[f]))

    window = max(1, round(TURNING_WINDOW_S / frame_dt))
    for k, name in enumerate(PULLEY_NAMES):
        angle = metrics.pulley_angle_deg[:, k]
        if m <= window or np.isnan(angle).all():
            continue
        delta = np.zeros(m)
        delta[window:] = angle[window:] - angle[:-window]
        turning = np.abs(delta) >= TURNING_DEG
        edges = np.flatnonzero(np.diff(turning.astype(np.int8)))
        start = None
        for f in edges + 1:
            if turning[f]:
                start = int(f)
                add(start, "pulley_turning_start", pulley=name,
                    delta_deg=float(delta[f]), angle_deg=float(angle[f]))
            else:
                begin = start if start is not None else 0
                add(int(f), "pulley_turning_end", pulley=name, angle_deg=float(angle[f]),
                    turned_deg=float(angle[f] - angle[max(begin - window, 0)]))
                start = None
    events.sort(key=lambda e: (e.step, DERIVED_KINDS.index(e.kind)))
    return events


class _TargetTrack:
    """Messages of one channel, ready for ``step`` lookups."""

    def __init__(self, rec: Recording, channel: str) -> None:
        msgs = [t for t in rec.targets if t.channel == channel]
        self.steps = np.array([t.step for t in msgs], dtype=np.int64)
        self.parsed = [self._parse(t) for t in msgs]
        self.warned = False

    @staticmethod
    def _parse(msg):
        payload = msg.payload
        if "blocks" not in payload:
            return ("pose", np.asarray(payload["position"], dtype=np.float64),
                    np.asarray(payload["orientation_wxyz"], dtype=np.float64), msg)
        blocks = payload["blocks"]
        pos = blocks.get("end_effector_position_target")
        rot = blocks.get("end_effector_orientation_target")
        if pos is None or rot is None or not pos["t"]:
            return None
        quats = np.asarray(rot["data"], dtype=np.float64).T.copy()
        # Same hemisphere as the previous knot, so interpolation takes the short way round.
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i - 1]) < 0.0:
                quats[i] = -quats[i]
        return ("traj", np.asarray(pos["t"], dtype=np.float64),
                np.asarray(pos["data"], dtype=np.float64).T, np.asarray(rot["t"], np.float64),
                quats, msg)


_tracks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _track(rec: Recording, channel: str) -> _TargetTrack:
    per_rec = _tracks.setdefault(rec, {})
    if channel not in per_rec:
        per_rec[channel] = _TargetTrack(rec, channel)
    return per_rec[channel]


def _interp_rows(t: float, knots_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.array([np.interp(t, knots_t, values[:, i]) for i in range(values.shape[1])])


def target_pose_at(rec: Recording, channel: str, step: int
                   ) -> tuple[np.ndarray, np.ndarray] | None:
    """``(position, quat_wxyz)`` of the latest ``channel`` message with ``msg.step <= step``,
    in the frame the controller published it in (see :func:`target_world_pose`)."""
    track = _track(rec, channel)
    i = int(np.searchsorted(track.steps, step, side="right")) - 1
    if i < 0 or track.parsed[i] is None:
        return None
    parsed = track.parsed[i]
    if parsed[0] == "pose":
        return parsed[1].copy(), parsed[2].copy()
    _, pos_t, pos, rot_t, quats, msg = parsed
    t = step * rec.control_dt
    if abs(pos_t[0] - msg.sim_time) > TARGET_CLOCK_TOLERANCE_S:
        if not track.warned:
            track.warned = True
            logger.warning(f"[REPLAY] {channel}: trajectory t[0] {pos_t[0]:.3f} s is "
                           f"{pos_t[0] - msg.sim_time:+.3f} s from its arrival sim time; "
                           "holding the first knot")
        t = pos_t[0]
    position = _interp_rows(t, pos_t, pos)
    quat = _interp_rows(t, rot_t, quats)
    norm = float(np.linalg.norm(quat))
    return position, (quat / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0, 0.0]))


def target_channels(rec: Recording) -> dict[str, str]:
    """Triad name -> channel of the three recorded target streams."""
    channels = rec.meta.get("channels", {})
    return {
        "Franka EE target (traj)": channels.get("tracking_trajectory_actor_channel",
                                                "TARGET_CARTESIAN_POSE_TRAJECTORY"),
        "UR EE target (traj)": channels.get("ur_tracking_trajectory_actor_channel",
                                            "UR_TARGET_CARTESIAN_POSE_TRAJECTORY"),
        "UR target (spatial pose)": channels.get("ur_target_spatial_pose_channel",
                                                 "UR_TARGET_SPATIAL_POSE"),
    }


def target_world_pose(rec: Recording, channel: str, step: int, frame: int
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """:func:`target_pose_at` in world: the Franka trajectory is in ``panda_link0``, the UR
    trajectory is ``tool0`` in the UR ``base`` frame, the spatial pose is already world."""
    pose = target_pose_at(rec, channel, step)
    if pose is None:
        return None
    channels = rec.meta.get("channels", {})
    labels = rec.meta["body_labels"]
    if channel == channels.get("tracking_trajectory_actor_channel"):
        base, q_offset = FRANKA_BASE_BODY, None
    elif channel == channels.get("ur_tracking_trajectory_actor_channel"):
        base, q_offset = UR_BASE_BODY, _Q_BASE_LINK_BASE
    else:
        return pose
    if base not in labels:
        return pose
    row = rec.body_q[frame, labels.index(base)].astype(np.float64)
    q_base = row[3:7] if q_offset is None else quat_multiply(row[3:7], q_offset)
    position = row[:3] + quat_rotate(q_base, pose[0])
    quat = quat_multiply(q_base, wxyz_to_xyzw(pose[1]))
    return position, xyzw_to_wxyz(quat / np.linalg.norm(quat))
