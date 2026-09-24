"""Torch-free learned latent encoder, learned-LCS step and the ``LATENT_STATE`` LCM message.

``LatentEncoder`` reproduces ``lcs_learning``'s ``PointNetObsAutoEncoder.encode`` from an exported
``deploy.npz`` (BatchNorm folded, float32 inside); ``LearnedLcs`` is the trained PGD forward step
(float64); ``DemoGoals`` loads the harness's ``demo_goals.npz``. numpy + LCM types only.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from dairlib import (
    lcmt_metadata,
    lcmt_saved_traj,
    lcmt_timestamped_saved_traj,
    lcmt_trajectory_block,
)
from task_common.lcs_dataset import BELT_BODIES, BELT_POINTS, STATE_DIM

LATENT_STATE_CHANNEL = "LATENT_STATE"
LATENT_METADATA_NAME = "latent_encoder"
LATENT_BLOCK_NAMES = ("latent", "ee_pose_franka", "ee_pose_ur", "proprio")
_ENC = "encoder__"


def resize_points_ordered(xyz, n_points: int) -> np.ndarray:
    """``RoundBeltTupleDataset._resize_points_ordered`` on a float32 cast: linspace or tiling."""
    arr = np.asarray(xyz, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {arr.shape}")
    n = arr.shape[0]
    if n == 0:
        raise ValueError("cannot resize an empty point cloud")
    if n >= n_points:
        idx = np.linspace(0, n - 1, n_points, dtype=np.int64)
    else:
        idx = np.tile(np.arange(n, dtype=np.int64), int(np.ceil(n_points / n)))[:n_points]
    return arr[idx]


def _load_npz(path) -> dict:
    with np.load(Path(path), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _scalar(x):
    return np.asarray(x).item()


def _fold_bn(w: dict, conv: str, bn: str, eps: float) -> tuple[np.ndarray, np.ndarray]:
    weight = w[f"{_ENC}pc_ae__{conv}__weight"][:, :, 0]
    bias = w[f"{_ENC}pc_ae__{conv}__bias"]
    gamma, beta = w[f"{_ENC}pc_ae__{bn}__weight"], w[f"{_ENC}pc_ae__{bn}__bias"]
    mean, var = w[f"{_ENC}pc_ae__{bn}__running_mean"], w[f"{_ENC}pc_ae__{bn}__running_var"]
    scale = gamma / np.sqrt(var + eps)
    # Transposed so the per-point layer is ``x @ W``.
    return ((weight * scale[:, None]).T.astype(np.float32),
            (scale * (bias - mean) + beta).astype(np.float32))


class LatentEncoder:
    """PointNet global feature + fusion MLP: ``(pc_raw, prop40, belt150) -> z``."""

    def __init__(self, weights: dict) -> None:
        self.num_points = int(_scalar(weights["encoder_num_points"]))
        self.belt_num_points = int(_scalar(weights["encoder_belt_num_points"]))
        self.proprio_dim = int(_scalar(weights["encoder_proprio_dim"]))
        self.point_feat_dim = int(_scalar(weights["encoder_point_feat_dim"]))
        eps = float(_scalar(weights["encoder_bn_epsilon"]))
        self._layers = [_fold_bn(weights, f"conv{i}", f"bn{i}", eps) for i in (1, 2, 3)]
        self._w0 = weights[f"{_ENC}fuse_mlp__0__weight"].T.astype(np.float32)
        self._b0 = weights[f"{_ENC}fuse_mlp__0__bias"].astype(np.float32)
        self._w2 = weights[f"{_ENC}fuse_mlp__2__weight"].T.astype(np.float32)
        self._b2 = weights[f"{_ENC}fuse_mlp__2__bias"].astype(np.float32)
        self.latent_dim = int(self._b2.shape[0])
        fused = self.point_feat_dim + self.proprio_dim + 3 * self.belt_num_points
        if self._w0.shape[0] != fused or self._layers[2][1].shape[0] != self.point_feat_dim:
            raise ValueError(f"encoder weights do not match the dims (fused {fused})")

    @classmethod
    def load(cls, deploy_npz) -> LatentEncoder:
        return cls(_load_npz(deploy_npz))

    def preprocess(self, pc_raw) -> np.ndarray:
        return resize_points_ordered(pc_raw, self.num_points)

    def pc_global(self, pc_resized) -> np.ndarray:
        """``(..., num_points, 3)`` resized cloud(s) -> ``(..., point_feat_dim)`` float32."""
        x = np.asarray(pc_resized, dtype=np.float32)
        (w1, b1), (w2, b2), (w3, b3) = self._layers
        x = np.maximum(x @ w1 + b1, 0.0)
        x = np.maximum(x @ w2 + b2, 0.0)
        return np.max(x @ w3 + b3, axis=-2)

    def _inputs(self, prop, belt) -> tuple[np.ndarray, np.ndarray]:
        prop = np.asarray(prop, dtype=np.float32)
        if prop.shape != (self.proprio_dim,):
            raise ValueError(f"proprio shape {prop.shape} != ({self.proprio_dim},)")
        belt = resize_points_ordered(belt, self.belt_num_points)
        return prop, belt.reshape(-1)

    def _fuse(self, g: np.ndarray, prop: np.ndarray, belt: np.ndarray) -> np.ndarray:
        fused = np.concatenate([g, prop, belt], axis=-1)
        return np.maximum(fused @ self._w0 + self._b0, 0.0) @ self._w2 + self._b2

    def encode(self, pc_raw, prop40, belt150) -> np.ndarray:
        """``z (latent_dim,)`` float64 (computed in float32)."""
        prop, belt = self._inputs(prop40, belt150)
        g = self.pc_global(self.preprocess(pc_raw))
        return self._fuse(g, prop, belt).astype(np.float64)

    def encode_batch(self, pc_list, props, belts) -> np.ndarray:
        """``(B, latent_dim)`` float64 for ``B`` raw clouds (ragged ok)."""
        if not (len(pc_list) == len(props) == len(belts)):
            raise ValueError("pc_list, props and belts must have the same length")
        pcs = np.stack([self.preprocess(pc) for pc in pc_list])
        ins = [self._inputs(p, b) for p, b in zip(props, belts, strict=True)]
        prop = np.stack([p for p, _ in ins])
        belt = np.stack([b for _, b in ins])
        return self._fuse(self.pc_global(pcs), prop, belt).astype(np.float64)


class LearnedLcs:
    """The exported learned LCS: ``z+ = A z + B u + D lam + d``, ``lam`` from Nesterov PGD."""

    def __init__(self, d: dict) -> None:
        for k in ("A", "B", "D", "d", "E", "F", "H", "c", "G", "J"):
            setattr(self, k, np.asarray(d[k], dtype=np.float64))
        self.stiffness = float(_scalar(d["stiffness"]))
        self.n_x, self.n_u, self.n_lam = (int(_scalar(d[k])) for k in ("n_x", "n_u", "n_lam"))
        self.dt = float(_scalar(d["dt"]))
        self.z_goal = np.asarray(d["z_goal"], dtype=np.float64)
        self.z_std = np.asarray(d["z_std"], dtype=np.float64)
        self.u_lb = np.asarray(d["u_lb"], dtype=np.float64)
        self.u_ub = np.asarray(d["u_ub"], dtype=np.float64)
        self.goal_tol = float(_scalar(d["goal_tol_whitened"]))
        self.goal_source = str(_scalar(d["goal_source"]))
        self.z_goal_mean_engaged = np.asarray(d["z_goal_mean_engaged"], dtype=np.float64)
        if "z_goal_stage1" in d and "z_goal_stage2" in d:
            self.stage_goals = np.stack([d["z_goal_stage1"], d["z_goal_stage2"]]).astype(
                np.float64)
        else:
            self.stage_goals = self.z_goal[None, :].copy()
        self.goal_frames = (np.asarray(d["goal_frames"], dtype=np.int64)
                            if "goal_frames" in d else None)
        self._check_shapes()
        self.F_pgd = self.G @ self.G.T + self.stiffness * np.eye(self.n_lam)
        self._step_size = 1.0 / max(float(np.linalg.eigvalsh(self.F_pgd).max()), 1e-6)

    def _check_shapes(self) -> None:
        nx, nu, nl = self.n_x, self.n_u, self.n_lam
        want = {"A": (nx, nx), "B": (nx, nu), "D": (nx, nl), "d": (nx,), "E": (nl, nx),
                "F": (nl, nl), "H": (nl, nu), "c": (nl,), "G": (nl, nl), "J": (nl, nl),
                "z_goal": (nx,), "z_std": (nx,), "u_lb": (nu,), "u_ub": (nu,),
                "z_goal_mean_engaged": (nx,)}
        for k, shape in want.items():
            if getattr(self, k).shape != shape:
                raise ValueError(f"{k} shape {getattr(self, k).shape} != {shape}")
        if self.stage_goals.shape[1] != nx:
            raise ValueError(f"stage_goals shape {self.stage_goals.shape}")
        if np.any(self.z_std <= 0.0):
            raise ValueError("z_std must be > 0")

    @classmethod
    def load(cls, deploy_npz) -> LearnedLcs:
        return cls(_load_npz(deploy_npz))

    def solve_lambda(self, z, u, iters: int = 25, tol: float = 0.0) -> np.ndarray:
        """``lcs_model.py`` PGD; ``tol > 0`` stops when ``|lam - lam_old| < tol``."""
        q = self.E @ z + self.H @ u + self.c
        lam = np.zeros(self.n_lam)
        y, t = lam, 1.0
        for _ in range(int(iters)):
            lam_old = lam
            lam = np.maximum(y - self._step_size * (self.F_pgd @ y + q), 0.0)
            if tol > 0.0 and np.linalg.norm(lam - lam_old) < tol:
                break
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            y = lam + ((t - 1.0) / t_next) * (lam - lam_old)
            t = t_next
        return lam

    def step(self, z, u, iters: int = 25, tol: float = 0.0
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(z_next, lam, slack)``; defaults = the trained fixed 25-iteration solver."""
        z = np.asarray(z, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        lam = self.solve_lambda(z, u, iters, tol)
        z_next = self.A @ z + self.B @ u + self.D @ lam + self.d
        slack = self.F @ lam + self.E @ z + self.H @ u + self.c
        return z_next, lam, slack

    def rollout(self, z0, U, iters: int = 25, tol: float = 0.0
                ) -> tuple[np.ndarray, np.ndarray]:
        """``(Z (T+1, n_x), Lam (T, n_lam))`` for inputs ``U (T, n_u)``."""
        U = np.asarray(U, dtype=np.float64).reshape(-1, self.n_u)
        Z = [np.asarray(z0, dtype=np.float64)]
        lams = []
        for u in U:
            z_next, lam, _ = self.step(Z[-1], u, iters, tol)
            Z.append(z_next)
            lams.append(lam)
        return np.stack(Z), np.asarray(lams).reshape(len(U), self.n_lam)

    def whitened_dist(self, z, goal=None) -> float:
        goal = self.z_goal if goal is None else np.asarray(goal, dtype=np.float64)
        return float(np.linalg.norm((np.asarray(z, dtype=np.float64) - goal) / self.z_std))

    def clip_u(self, u) -> np.ndarray:
        return np.clip(np.asarray(u, dtype=np.float64), self.u_lb, self.u_ub)


class DemoGoals:
    """The harness's ``demo_goals.npz`` (per-stage latent goals from one demonstration)."""

    _STR_KEYS = ("demo_episode", "demo_sha256", "deploy_sha256", "created", "notes")

    def __init__(self, d: dict) -> None:
        n = int(_scalar(d["n_stages"]))
        if n < 1:
            raise ValueError(f"n_stages {n} < 1")
        self.n_stages = n
        self.z_std = np.asarray(d["z_std"], dtype=np.float64)
        if self.z_std.ndim != 1 or np.any(self.z_std <= 0.0):
            raise ValueError(f"z_std must be a positive vector, got shape {self.z_std.shape}")
        nx = self.z_std.shape[0]
        n_t = np.asarray(d["z"]).shape[0]
        want = {"stage_labels": (n,), "stage_frames": (n,), "z_goals": (n, nx),
                "goal_tols": (n,), "max_durations_s": (n,), "ee_pose_franka": (n, 7),
                "ee_pose_ur": (n, 7), "z": (n_t, nx), "goal_dist": (n_t, n),
                "pcd_belt_stage": (n, BELT_POINTS, 3), "belt_xyz_stage": (n, BELT_BODIES, 3),
                "pulley_pose_stage": (n, 7), "state_stage": (n, STATE_DIM)}
        for k, shape in want.items():
            arr = np.asarray(d[k])
            if arr.shape != shape:
                raise ValueError(f"demo goals: {k} shape {arr.shape} != {shape}")
        self.stage_labels = [str(s) for s in d["stage_labels"]]
        self.stage_frames = np.asarray(d["stage_frames"], dtype=np.int64)
        for k in want:
            if k not in ("stage_labels", "stage_frames"):
                setattr(self, k, np.asarray(d[k], dtype=np.float64))
        self.first_hold_place_3_frame = int(_scalar(d["first_hold_place_3_frame"]))
        for k in self._STR_KEYS:
            setattr(self, k, str(_scalar(d[k])))

    @classmethod
    def load(cls, npz_path) -> DemoGoals:
        return cls(_load_npz(npz_path))

    def stage_goal(self, k: int) -> np.ndarray:
        return self.z_goals[k].copy()

    def dist(self, z, k: int) -> float:
        """Whitened distance to stage ``k`` with the demo file's ``z_std``."""
        diff = np.asarray(z, dtype=np.float64) - self.z_goals[k]
        return float(np.linalg.norm(diff / self.z_std))


def _column_block(name: str, values: np.ndarray, t: float) -> lcmt_trajectory_block:
    block = lcmt_trajectory_block()
    block.trajectory_name = name
    block.num_points = 1
    block.num_datatypes = int(values.shape[0])
    block.time_vec = [float(t)]
    block.datapoints = [[float(v)] for v in values]
    block.datatypes = ["double"] * int(values.shape[0])
    return block


def latent_state_message(utime: int, t: float, z, ee_pose_franka7, ee_pose_ur7, prop40
                         ) -> lcmt_timestamped_saved_traj:
    """``LATENT_STATE``: one single-column block per name in ``LATENT_BLOCK_NAMES``."""
    if int(utime) <= 0:
        raise ValueError(f"utime {utime} must be > 0")
    values = [np.asarray(v, dtype=np.float64).reshape(-1)
              for v in (z, ee_pose_franka7, ee_pose_ur7, prop40)]
    for name, v, n in zip(LATENT_BLOCK_NAMES[1:], values[1:], (7, 7, STATE_DIM), strict=True):
        if v.shape != (n,):
            raise ValueError(f"{name} shape {v.shape} != ({n},)")
    meta = lcmt_metadata()
    meta.git_dirty_flag = False
    meta.datetime = ""
    meta.name = LATENT_METADATA_NAME
    meta.description = ""
    meta.git_commit_hash = ""
    traj = lcmt_saved_traj()
    traj.metadata = meta
    traj.trajectories = [_column_block(n, v, t)
                         for n, v in zip(LATENT_BLOCK_NAMES, values, strict=True)]
    traj.num_trajectories = len(traj.trajectories)
    traj.trajectory_names = list(LATENT_BLOCK_NAMES)
    msg = lcmt_timestamped_saved_traj()
    msg.utime = int(utime)
    msg.saved_traj = traj
    return msg


def parse_latent_state_message(msg: lcmt_timestamped_saved_traj
                               ) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray]:
    """``(utime, t, z, ee_pose_franka7, ee_pose_ur7, prop40)``."""
    if int(msg.utime) <= 0:
        raise ValueError(f"utime {msg.utime} must be > 0")
    blocks = {b.trajectory_name: b for b in msg.saved_traj.trajectories}
    missing = [n for n in LATENT_BLOCK_NAMES if n not in blocks]
    if missing:
        raise ValueError(f"LATENT_STATE message lacks blocks {missing}")
    cols = [np.asarray(blocks[n].datapoints, dtype=np.float64)[:, 0] for n in LATENT_BLOCK_NAMES]
    t = float(blocks[LATENT_BLOCK_NAMES[0]].time_vec[0])
    return (int(msg.utime), t, *cols)
