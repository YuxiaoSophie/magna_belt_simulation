"""The ``.npz`` episode format consumed by ``lcs_learning`` (``RoundBeltTupleDataset``).

Mirrors magna's ``python/collect_round_belt_dataset.py``; see ``docs/lcs-dataset.md``. numpy-only
so ``validate_episode`` runs without the sim.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from loguru import logger

SIM_DT_S = 0.005
# Camera at 20 fps + the collector's 13.3 Hz rate limit -> every other cloud (docs §5).
SAMPLE_PERIOD_S = 0.1
SAMPLE_STEPS = round(SAMPLE_PERIOD_S / SIM_DT_S)
assert abs(SAMPLE_STEPS * SIM_DT_S - SAMPLE_PERIOD_S) < 1e-12, "period not a whole step count"
SAMPLE_PERIOD_US = round(SAMPLE_PERIOD_S * 1e6)
# C3 knot spacing the collector's action deltas span (informational).
ACTION_KNOT_DT_S = 0.075

STATE_DIM = 40
ACTION_DIM = 12
BELT_POINTS = 150
BELT_BODIES = 48
PCD_POINTS = None  # ragged: the collector stores the cropped cloud as received
POSE_LAYOUT = "xyz_wxyz"
POINT_CLOUD_SOURCE = "camera_plus_belt"
LOADER_SOURCES = ("belt", "camera", "camera_plus_belt", "belt_plus_static")

CLOUD_KEYS = ("pcd", "pcd_belt", "pcd_kinematic")
KEYS_REQUIRED = ("pcd", "pcd_belt", "pcd_kinematic", "state", "actions", "utime")
KEYS_OPTIONAL = (
    "pcd_rgb", "trajectory_label", "point_cloud_source", "record_kinematic_points",
    "kinematic_model_names", "kinematic_body_name_patterns", "kinematic_sampled_body_names",
)
LEGACY_CLOUD_KEYS = ("pcd_kept", "pcd_removed")
EXTRA_PREFIX = "sim_"
_RESERVED_EXTRAS = ("step", "time", "meta")

# The collector's argparse defaults, written verbatim for parity.
KINEMATIC_MODEL_NAMES = ("panda_hand", "robotiq_85", "nist_board")
KINEMATIC_BODY_NAME_PATTERNS = (
    "panda_leftfinger", "panda_rightfinger", "left_finger", "right_finger", "board",
    "small_round_pulley", "large_round_pulley", "pulley_bases_and_screw",
)


def _as_vec(x, n: int, name: str) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.shape != (n,):
        raise ValueError(f"{name}: expected {n} values, got shape {np.shape(x)}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name}: non-finite values")
    return v


def _as_xyz(x, name: str, min_points: int = 0) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name}: expected (N, 3), got {a.shape}")
    if a.shape[0] < min_points:
        raise ValueError(f"{name}: {a.shape[0]} points < {min_points}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: non-finite points")
    return a


def pose_from_xyz_xyzw(tf) -> np.ndarray:
    """Newton ``transform`` ``[x, y, z, qx, qy, qz, qw]`` -> ``[x, y, z, qw, qx, qy, qz]``."""
    t = _as_vec(tf, 7, "transform")
    return np.concatenate([t[:3], t[6:7], t[3:6]])


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def delta_rotvec(quat0_wxyz, quat1_wxyz) -> np.ndarray:
    """Rotation vector of ``R1 * R0^T`` (world-frame delta), angle in ``[0, pi]``."""
    q0 = _as_vec(quat0_wxyz, 4, "quat0")
    q1 = _as_vec(quat1_wxyz, 4, "quat1")
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    q = _quat_mul_wxyz(q1, q0 * np.array([1.0, -1.0, -1.0, -1.0]))
    if q[0] < 0.0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return 2.0 * q[1:]
    return q[1:] / s * (2.0 * math.atan2(s, q[0]))


def state_vector(q_franka, q_ur, v_franka, v_ur, ee_franka_pose7, ee_ur_pose7) -> np.ndarray:
    """``[q_franka(7), q_ur(6), v_franka(7), v_ur(6), ee_franka(7), ee_ur(7)]``, float64.

    Poses are ``xyz_wxyz`` in the world frame (Franka ``finger_tip``, UR tracking frame).
    """
    return np.concatenate([
        _as_vec(q_franka, 7, "q_franka"), _as_vec(q_ur, 6, "q_ur"),
        _as_vec(v_franka, 7, "v_franka"), _as_vec(v_ur, 6, "v_ur"),
        _as_vec(ee_franka_pose7, 7, "ee_franka_pose7"), _as_vec(ee_ur_pose7, 7, "ee_ur_pose7"),
    ])


def action_vector(pose_franka_t, pose_franka_t1, pose_ur_t, pose_ur_t1) -> np.ndarray:
    """``[dxyz_franka, dxyz_ur, drotvec_franka, drotvec_ur]``, float64, world frame.

    The collector's knot delta: ``dxyz = p1 - p0`` and ``drotvec = rotvec(R1 * R0^T)``
    (``_parse_trajectory_pose_and_delta`` / ``_delta_orientation_rotvec``). Poses are
    ``xyz_wxyz``; the caller picks which two commanded poses to difference.
    """
    f0 = _as_vec(pose_franka_t, 7, "pose_franka_t")
    f1 = _as_vec(pose_franka_t1, 7, "pose_franka_t1")
    u0 = _as_vec(pose_ur_t, 7, "pose_ur_t")
    u1 = _as_vec(pose_ur_t1, 7, "pose_ur_t1")
    return np.concatenate([
        f1[:3] - f0[:3], u1[:3] - u0[:3],
        delta_rotvec(f0[3:], f1[3:]), delta_rotvec(u0[3:], u1[3:]),
    ])


def belt_points_ordered(belt_xyz, n_points: int = BELT_POINTS) -> np.ndarray:
    """Arc-length resampling of the closed body polyline (index order, back to body 0).

    Point 0 is body 0 and the points walk in body-index order; returns ``(n_points, 3)`` float32.
    """
    p = np.asarray(belt_xyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 3:
        raise ValueError(f"belt_xyz: expected (M>=3, 3), got {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("belt_xyz: non-finite points")
    loop = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0.0:
        raise ValueError("belt_xyz: zero-length loop")
    s = np.arange(n_points) * (cum[-1] / n_points)
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    frac = np.divide(s - cum[i], seg[i], out=np.zeros_like(s), where=seg[i] > 0.0)
    return (loop[i] + frac[:, None] * (loop[i + 1] - loop[i])).astype(np.float32)


def camera_points(xyz) -> np.ndarray:
    """The collector's ``pcd``: the cropped cloud as received (float32, ragged ``(N, 3)``)."""
    return _as_xyz(xyz, "pcd", min_points=1).copy()


def kinematic_points(xyz=None) -> np.ndarray:
    """The collector's ``pcd_kinematic``: empty ``(0, 3)`` unless kinematic points are recorded."""
    if xyz is None:
        return np.zeros((0, 3), dtype=np.float32)
    return _as_xyz(xyz, "pcd_kinematic").copy()


def _ragged(frames: list) -> np.ndarray:
    # np.array(list, dtype=object) collapses to N-D when all frames share a shape.
    out = np.empty(len(frames), dtype=object)
    for i, f in enumerate(frames):
        out[i] = f
    return out


def _stack_or_ragged(frames: list, dtype) -> np.ndarray:
    if len({f.shape for f in frames}) == 1:
        return np.stack(frames).astype(dtype)
    return _ragged(frames)


def json_default(o):
    """``json.dumps(default=...)`` for numpy scalars/arrays and paths."""
    if isinstance(o, np.generic | np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


class EpisodeWriter:
    """Accumulates sampled frames and writes one episode ``.npz`` (``docs/lcs-dataset.md``)."""

    def __init__(self, sample_steps: int = SAMPLE_STEPS) -> None:
        self.sample_steps = int(sample_steps)
        self._steps: list[int] = []
        self._times: list[float] = []
        self._state: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._pcd: list[np.ndarray] = []
        self._rgb: list[np.ndarray] = []
        self._belt: list[np.ndarray] = []
        self._kin: list[np.ndarray] = []
        self._extras: dict[str, list] = {}

    def __len__(self) -> int:
        return len(self._steps)

    def add_frame(self, step: int, sim_time: float, state, action, pcd, pcd_belt,
                  pcd_kinematic=None, extras: dict | None = None, pcd_rgb=None) -> None:
        step = int(step)
        if self._steps and step - self._steps[-1] != self.sample_steps:
            raise ValueError(f"step {step}: expected {self._steps[-1] + self.sample_steps} "
                             f"(sample_steps={self.sample_steps} after {self._steps[-1]})")
        state = _as_vec(state, STATE_DIM, "state")
        action = _as_vec(action, ACTION_DIM, "action")
        pcd = camera_points(pcd)
        belt = _as_xyz(pcd_belt, "pcd_belt")
        if belt.shape != (BELT_POINTS, 3):
            raise ValueError(f"pcd_belt: expected ({BELT_POINTS}, 3), got {belt.shape}")
        kin = kinematic_points(pcd_kinematic)
        if pcd_rgb is None:
            rgb = np.zeros(pcd.shape, dtype=np.uint8)
        else:
            rgb = np.asarray(pcd_rgb, dtype=np.uint8)
            if rgb.shape != pcd.shape:
                raise ValueError(f"pcd_rgb: shape {rgb.shape} != pcd {pcd.shape}")
        extras = dict(extras or {})
        for name in extras:
            if name in _RESERVED_EXTRAS or name.startswith(EXTRA_PREFIX) or not name.isidentifier():
                raise ValueError(f"extra {name!r}: reserved or invalid (written as sim_<name>)")
        if self._steps and set(extras) != set(self._extras):
            raise ValueError(f"extras keys {sorted(extras)} != first frame {sorted(self._extras)}")
        if not self._steps:
            self._extras = {name: [] for name in extras}
        for name, value in extras.items():
            self._extras[name].append(np.asarray(value))
        self._steps.append(step)
        self._times.append(float(sim_time))
        self._state.append(state)
        self._actions.append(action)
        self._pcd.append(pcd)
        self._rgb.append(rgb)
        self._belt.append(belt)
        self._kin.append(kin)

    def arrays(self, trajectory_label: str, extras_meta: dict | None = None) -> dict:
        """The exact ``{key: array}`` payload :meth:`write` saves."""
        if not self._steps:
            raise ValueError("no frames added")
        meta = dict(extras_meta or {})
        meta["lcs_format"] = {
            "sample_period_s": self.sample_steps * SIM_DT_S, "sample_steps": self.sample_steps,
            "pose_layout": POSE_LAYOUT, "belt_points": BELT_POINTS,
            "action_knot_dt_s": ACTION_KNOT_DT_S,
        }
        kin = _stack_or_ragged(self._kin, np.float32)
        out = {
            "pcd": _ragged(self._pcd),
            "pcd_rgb": _ragged(self._rgb),
            "pcd_belt": np.stack(self._belt).astype(np.float32),
            "pcd_kinematic": kin,
            "kinematic_model_names": np.array(KINEMATIC_MODEL_NAMES),
            "kinematic_body_name_patterns": np.array(KINEMATIC_BODY_NAME_PATTERNS),
            "kinematic_sampled_body_names": np.array([]),
            "point_cloud_source": np.array(POINT_CLOUD_SOURCE),
            "record_kinematic_points": np.array(any(k.shape[0] > 0 for k in self._kin)),
            "state": np.stack(self._state).astype(np.float64),
            "actions": np.stack(self._actions).astype(np.float64),
            "utime": np.array([round(t * 1e6) for t in self._times], dtype=np.int64),
            "trajectory_label": np.array(str(trajectory_label)),
            "sim_step": np.array(self._steps, dtype=np.int64),
            "sim_time": np.array(self._times, dtype=np.float64),
            "sim_meta": np.array(json.dumps(meta, sort_keys=True, default=json_default)),
        }
        for name, values in self._extras.items():
            try:
                out[EXTRA_PREFIX + name] = np.stack(values)
            except ValueError:
                out[EXTRA_PREFIX + name] = _ragged(values)
        return out

    def write(self, path, trajectory_label: str, extras_meta: dict | None = None,
              arrays: dict | None = None, omit: tuple[str, ...] = ()) -> Path:
        """Write ``np.savez_compressed`` atomically; returns the path.

        ``arrays`` adds (or overrides) top-level keys; ``omit`` drops payload keys.
        """
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError(f"{path}: must end in .npz")
        if len(self) < 2:
            logger.warning(f"{path.name}: {len(self)} frame(s); lcs_learning skips files < 2")
        payload = self.arrays(trajectory_label, extras_meta)
        for key in omit:
            del payload[key]
        payload.update(arrays or {})
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **payload)
        os.replace(tmp, path)
        return path


def _fail(path: Path, key: str, message: str) -> None:
    raise ValueError(f"{path.name}: {key}: {message}")


def validate_episode(path, legacy: bool = False, period_us: int | None = SAMPLE_PERIOD_US) -> dict:
    """Re-implements ``RoundBeltTupleDataset``'s checks for every loader point-cloud source.

    ``legacy=True`` accepts ``pcd_kept``/``pcd_removed`` files (no ``pcd_belt``).
    ``period_us=None`` only requires a constant ``utime`` spacing. Raises ``ValueError`` naming
    the file and key; returns a summary dict.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        files = set(data.files)
        if "pcd_belt" in files:
            cloud_keys = CLOUD_KEYS
            required = KEYS_REQUIRED
        elif legacy:
            cloud_keys = LEGACY_CLOUD_KEYS
            required = ("state", "actions", *LEGACY_CLOUD_KEYS)
        else:
            _fail(path, "pcd_belt", "missing (pass legacy=True for pcd_kept/pcd_removed files)")
        for key in required:
            if key not in files:
                _fail(path, key, "missing")

        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"])
        if state.ndim != 2 or state.shape[1] != STATE_DIM:
            _fail(path, "state", f"expected (T, {STATE_DIM}), got {state.shape}")
        T = state.shape[0]
        if T < 2:
            _fail(path, "state", f"T={T} < 2 frames")
        if actions.shape != (T, ACTION_DIM):
            _fail(path, "actions", f"expected ({T}, {ACTION_DIM}), got {actions.shape}")
        for key, arr in (("state", state), ("actions", actions)):
            if not np.all(np.isfinite(arr.astype(np.float64))):
                _fail(path, key, "non-finite values")

        summary: dict = {"file": str(path), "T": T, "shapes": {}, "points": {}}
        for key in cloud_keys:
            arr = data[key]
            if len(arr) != T:
                _fail(path, key, f"{len(arr)} frames != state {T}")
            counts = []
            for i in range(T):
                try:
                    xyz = np.asarray(arr[i], dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    _fail(path, key, f"[{i}] not numeric: {exc}")
                if xyz.ndim != 2 or xyz.shape[1] != 3:
                    _fail(path, key, f"[{i}] expected (N, 3), got {xyz.shape}")
                if not np.all(np.isfinite(xyz)):
                    _fail(path, key, f"[{i}] non-finite points")
                counts.append(xyz.shape[0])
            # Empty kinematic frames are fine: belt_plus_static still has the 150 belt points.
            if key != "pcd_kinematic" and min(counts) < 1:
                _fail(path, key, f"frame {int(np.argmin(counts))} has no points")
            if key == "pcd_belt" and set(counts) != {BELT_POINTS}:
                _fail(path, key,
                      f"expected {BELT_POINTS} points per frame, got {sorted(set(counts))}")
            summary["points"][key] = (min(counts), max(counts))
        if legacy and "pcd_belt" not in files and min(summary["points"]["pcd_kept"]) < 1:
            _fail(path, "pcd_kept", "empty reconstruction target")

        if "utime" in files:
            utime = np.asarray(data["utime"])
            if utime.shape != (T,) or not np.issubdtype(utime.dtype, np.integer):
                _fail(path, "utime", f"expected ({T},) int, got {utime.shape} {utime.dtype}")
            du = np.diff(utime.astype(np.int64))
            if np.any(du <= 0):
                _fail(path, "utime", "not strictly increasing")
            if np.any(du != du[0]):
                _fail(path, "utime", f"spacing not constant: {sorted(set(du.tolist()))[:5]}")
            if period_us is not None and du[0] != period_us:
                _fail(path, "utime", f"spacing {int(du[0])} us != period {period_us} us")
            summary["period_us"] = int(du[0])
        if "trajectory_label" in files:
            label = data["trajectory_label"]
            if label.shape != () or label.dtype.kind not in "UO":
                _fail(path, "trajectory_label", f"expected a 0-d string, got {label.shape}")
            summary["trajectory_label"] = str(label.item())
        summary["sources"] = LOADER_SOURCES if "pcd_belt" in files else ("legacy",)
        summary["shapes"] = {k: (tuple(data[k].shape), str(data[k].dtype)) for k in data.files}
    return summary
