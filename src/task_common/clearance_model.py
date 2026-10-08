"""Frozen gripper / Franka-hand point sets vs the board's colliders, for the MPPI planner.

A :class:`ClearanceModel` is ``round_belt_task.clearance`` geometry exported once from a sim
state: the 2F-85 colliders frozen in the UR ``tracking_frame`` (``Gripper.points(None)``, the
as-held set), the Franka hand's mesh colliders frozen in ``finger_tip``, the plate box and the
revolved pulleys (all world frame), plus the two clearance thresholds. :func:`distance_numpy`
is the reference (``Board.min_distance`` on the transformed sets, i.e. ``Gripper.predict``);
``mppi_lcs`` runs the same maths batched on the GPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from round_belt_task import clearance

FRAMES = {"ur": "ur_tracking_frame", "franka": "franka_finger_tip"}
POSE_LAYOUT = "xyz_wxyz"
DEFAULT_UR_MIN_MM = clearance.DEFAULT_MIN_CLEARANCE_MM
# data/lcs/realfirst/v1_1/clearance/franka_threshold.json (seed demos' seated frames, rounded).
DEFAULT_FRANKA_MIN_MM = 2.0
MAX_KERNEL_POINTS = None  # full set: a lowest-1024 cap errs 7.8 mm at 40 deg (C10)
PRUNE_POINTS = 1024
SEATED_FRAMES = 13


@dataclass
class ClearanceModel:
    ur_local: np.ndarray  # (N, 3) in the UR tracking frame
    franka_local: np.ndarray  # (M, 3) in the Franka finger_tip frame
    plate: dict  # rot (3, 3), pos (3,), half (3,)
    pulleys: list[dict]  # rot (3, 3), pos (3,), profile (P, 2)
    ur_min_m: float
    franka_min_m: float
    source: dict = field(default_factory=dict)

    def board(self) -> clearance.Board:
        return clearance.Board(
            plate=clearance.Box(**{k: np.asarray(v, float) for k, v in self.plate.items()}),
            pulleys=tuple(clearance.Revolved(**{k: np.asarray(v, float) for k, v in p.items()})
                          for p in self.pulleys))

    def kernel_points(self, max_points: int | None = MAX_KERNEL_POINTS
                      ) -> tuple[np.ndarray, np.ndarray]:
        """``(ur, franka)`` sets for the planner, lowest (local z) first: the ``max_points``
        lowest of a larger set, else (or with None) the full set."""
        return tuple(_lowest(p, max_points) for p in (self.ur_local, self.franka_local))

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"ur_local": self.ur_local, "franka_local": self.franka_local,
                  "plate_rot": self.plate["rot"], "plate_pos": self.plate["pos"],
                  "plate_half": self.plate["half"], "n_pulleys": np.int64(len(self.pulleys)),
                  "ur_min_m": np.float64(self.ur_min_m),
                  "franka_min_m": np.float64(self.franka_min_m),
                  "source_json": np.asarray(json.dumps(self.source, default=float))}
        for i, p in enumerate(self.pulleys):
            arrays.update({f"pulley{i}_{k}": np.asarray(v, np.float64) for k, v in p.items()})
        np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
        return path

    @classmethod
    def load(cls, path) -> ClearanceModel:
        with np.load(Path(path), allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        n = int(d["n_pulleys"])
        return cls(ur_local=d["ur_local"].astype(np.float64),
                   franka_local=d["franka_local"].astype(np.float64),
                   plate={k: d[f"plate_{k}"].astype(np.float64) for k in ("rot", "pos", "half")},
                   pulleys=[{k: d[f"pulley{i}_{k}"].astype(np.float64)
                             for k in ("rot", "pos", "profile")} for i in range(n)],
                   ur_min_m=float(d["ur_min_m"]), franka_min_m=float(d["franka_min_m"]),
                   source=json.loads(str(d["source_json"])))

    def check_frames(self) -> None:
        """Raise unless the sets live in the LATENT_STATE ee frames (UR tracking, finger_tip)."""
        frames = self.source.get("frames")
        if frames != FRAMES or self.source.get("pose_layout") != POSE_LAYOUT:
            raise ValueError(f"clearance model frames {frames} / layout "
                             f"{self.source.get('pose_layout')} != {FRAMES} / {POSE_LAYOUT}")


def _lowest(points: np.ndarray, n: int | None) -> np.ndarray:
    return points[np.argsort(points[:, 2], kind="stable")[:n]]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_mat(pose7) -> np.ndarray:
    """4x4 of ``xyz`` + ``wxyz``."""
    p = np.asarray(pose7, dtype=np.float64)
    w, x, y, z = p[3:7]
    X = np.eye(4)
    X[:3, :3] = clearance._quat_mat([x, y, z, w])  # xyzw
    X[:3, 3] = p[:3]
    return X


def distance_numpy(model: ClearanceModel, X_W_track, X_W_tip, board: clearance.Board | None = None,
                   points: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[float, float]:
    """``(d_ur, d_f)`` [m]: ``Board.min_distance`` of each set placed at its frame (4x4).

    ``points`` overrides the sets (e.g. :meth:`ClearanceModel.kernel_points`).
    """
    board = model.board() if board is None else board
    ur, fr = (model.ur_local, model.franka_local) if points is None else points
    return set_distance(board, ur, X_W_track), set_distance(board, fr, X_W_tip)


def set_distance(board: clearance.Board, local: np.ndarray, X) -> float:
    """``Board.min_distance`` of one frozen set placed at ``X`` (4x4)."""
    X = np.asarray(X, dtype=np.float64)
    return board.min_distance(local @ X[:3, :3].T + X[:3, 3])


def export_from_sim(sim, ur_min_m: float, franka_min_m: float) -> ClearanceModel:
    """The model at the sim's current state (gripper as held)."""
    from round_belt_task.arm_kinematics import FrankaTip, UrTracking

    board, gripper = sim.clearance_geometry()
    body_q = sim.state_0.body_q.numpy().astype(np.float64)
    X_tip, X_track = sim.ee_poses(body_q)
    q_f, q_u = sim.arm_positions()
    # LATENT_STATE carries URDF FK of the measured joints; the sets are frozen at the model's.
    fk_err = {"franka": float(np.abs(FrankaTip.fk(q_f) - X_tip).max()),
              "ur": float(np.abs(UrTracking.fk(q_u) - X_track).max())}
    source = {"frames": FRAMES, "pose_layout": POSE_LAYOUT, "fk_vs_model_max_abs": fk_err,
              "ur": f"2F-85 colliders {list(gripper.shape_labels)} (as held, prune "
                    f"{clearance.PRUNE_MARGIN} m)",
              "X_W_track": X_track.tolist(), "X_W_tip": X_tip.tolist()}
    try:
        hand = clearance.franka_hand_geometry(sim.model, body_q, X_tip)
        franka, source["franka"] = hand.local, "mesh"
        source["franka_shapes"] = list(hand.shape_labels)
    except RuntimeError as exc:
        franka, source["franka"] = np.zeros((1, 3)), "tip_only"
        source["franka_note"] = str(exc)
    return ClearanceModel(
        ur_local=np.asarray(gripper.points(None), np.float64), franka_local=franka,
        plate={"rot": board.plate.rot, "pos": board.plate.pos, "half": board.plate.half},
        pulleys=[{"rot": p.rot, "pos": p.pos, "profile": p.profile} for p in board.pulleys],
        ur_min_m=float(ur_min_m), franka_min_m=float(franka_min_m), source=source)


def franka_threshold(model: ClearanceModel, demo_files, frames: int = SEATED_FRAMES) -> dict:
    """``franka_min_m = min(0.002, 0.5 x min d_f)`` (>= 0) over the demos' last ``frames``
    frames at the recorded ``sim_ee_franka`` poses."""
    board = model.board()
    per = {}
    for path in demo_files:
        with np.load(path, allow_pickle=True) as f:
            ee = f["sim_ee_franka"][-frames:].astype(np.float64)
        per[str(path)] = [set_distance(board, model.franka_local, pose_mat(p)) for p in ee]
    d_min = min(min(v) for v in per.values())
    value = max(0.0, min(0.002, 0.5 * d_min))
    return {"rule": "franka_min_m = max(0, min(0.002, 0.5 x min d_f)) over the demos' last "
                    f"{frames} frames", "min_d_f_m": d_min, "franka_min_m": value,
            "franka_min_mm_rounded": round(value * 1e3, 1),
            "per_demo_min_mm": {k: min(v) * 1e3 for k, v in per.items()},
            "per_demo_d_f_mm": {k: [x * 1e3 for x in v] for k, v in per.items()}}
