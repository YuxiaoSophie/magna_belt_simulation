"""Torch-free learned latent encoder, learned-LCS step and the ``LATENT_STATE`` LCM message.

``LatentEncoder`` reproduces ``lcs_learning``'s ``PointNetObsAutoEncoder.encode`` from an exported
``deploy.npz`` (BatchNorm folded, float32 inside); ``BeltMlpEncoder`` and
``BeltCloudPointNetEncoder`` are opt-in ``encoder_type`` variants; ``LearnedLcs`` is the trained
PGD forward step (float64); ``DemoGoals`` loads the harness's ``demo_goals.npz``; ``LatentDecoder``
maps a latent back to belt points (``decoder.npz``, exporter ``--decoder-out``). numpy + LCM types
only, except ``BeltCloudPointMAEEncoder`` (``encoder_type`` belt_cloud_pointmae), which imports
torch (``task_common.pointmae_encoder``) when such a deploy is loaded.
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
from task_common.belt_cloud import SEG_FRAME, resample_deterministic
from task_common.lcs_dataset import BELT_BODIES, BELT_POINTS, STATE_DIM

LATENT_STATE_CHANNEL = "LATENT_STATE"
LATENT_METADATA_NAME = "latent_encoder"
LATENT_BLOCK_NAMES = ("latent", "ee_pose_franka", "ee_pose_ur", "proprio")
_ENC = "encoder__"
_DEC = "decoder__"
BELT_CLOUD_TYPE = "belt_cloud_pointnet"
POINTMAE_TYPE = "belt_cloud_pointmae"


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


def _fold_bn(w: dict, conv: str, bn: str, eps: float, dtype=np.float32
             ) -> tuple[np.ndarray, np.ndarray]:
    weight = w[f"{_ENC}pc_ae__{conv}__weight"][:, :, 0]
    bias = w[f"{_ENC}pc_ae__{conv}__bias"]
    gamma, beta = w[f"{_ENC}pc_ae__{bn}__weight"], w[f"{_ENC}pc_ae__{bn}__bias"]
    mean, var = w[f"{_ENC}pc_ae__{bn}__running_mean"], w[f"{_ENC}pc_ae__{bn}__running_var"]
    scale = gamma / np.sqrt(var + eps)
    # Transposed so the per-point layer is ``x @ W``.
    return ((weight * scale[:, None]).T.astype(dtype),
            (scale * (bias - mean) + beta).astype(dtype))


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
    def load(cls, deploy_npz, device=None
             ) -> LatentEncoder | BeltMlpEncoder | BeltCloudPointNetEncoder:
        """``device``: torch device of a belt_cloud_pointmae encoder (default cuda if
        available, else cpu); ignored by the numpy encoders."""
        weights = _load_npz(deploy_npz)
        if "encoder_type" in weights and str(_scalar(weights["encoder_type"])) == "belt_mlp":
            return BeltMlpEncoder(weights)
        if "encoder_type" in weights and str(_scalar(weights["encoder_type"])) == BELT_CLOUD_TYPE:
            return BeltCloudPointNetEncoder(weights)
        if "encoder_type" in weights and str(_scalar(weights["encoder_type"])) == POINTMAE_TYPE:
            return BeltCloudPointMAEEncoder(weights, device)
        return cls(weights)

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


_erf = np.frompyfunc(math.erf, 1, 1)


class BeltMlpEncoder:
    """Belt points only (no cloud, no proprio): ``(belt150 - pulley_xyz)`` flattened, normalised,
    then [Linear, LayerNorm, GELU] x 3 + Linear (``deploy.npz`` with ``encoder_type`` belt_mlp)."""

    def __init__(self, weights: dict) -> None:
        self.belt_num_points = int(_scalar(weights["encoder_belt_num_points"]))
        self.num_points = self.belt_num_points
        self.proprio_dim = STATE_DIM
        self._origin = weights["encoder_origin_xyz"].astype(np.float32)
        self._mu = weights["encoder_in_mean"].astype(np.float32)
        self._sd = weights["encoder_in_std"].astype(np.float32)
        self._eps = float(_scalar(weights["encoder_ln_epsilon"]))
        n = int(_scalar(weights["encoder_n_linear"]))
        self._lin = [(weights[f"{_ENC}lin{i}__weight"].T.astype(np.float32),
                      weights[f"{_ENC}lin{i}__bias"].astype(np.float32)) for i in range(n)]
        self._ln = [(weights[f"{_ENC}ln{i}__weight"].astype(np.float32),
                     weights[f"{_ENC}ln{i}__bias"].astype(np.float32)) for i in range(n - 1)]
        self.latent_dim = int(self._lin[-1][1].shape[0])
        if self._lin[0][0].shape[0] != 3 * self.belt_num_points:
            raise ValueError("belt_mlp input width != 3 x belt points")

    def preprocess(self, pc_raw) -> np.ndarray:
        return np.asarray(pc_raw, dtype=np.float32)

    def _mlp(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(self._lin):
            x = x @ w + b
            if i < len(self._ln):
                g, beta = self._ln[i]
                mu = x.mean(-1, keepdims=True)
                var = x.var(-1, keepdims=True)
                x = (x - mu) / np.sqrt(var + self._eps) * g + beta
                x = 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)).astype(np.float32))  # exact GELU
        return x

    def _belt(self, belt) -> np.ndarray:
        b = resize_points_ordered(belt, self.belt_num_points) - self._origin
        return (b.reshape(-1) - self._mu) / self._sd

    def encode(self, pc_raw, prop40, belt150) -> np.ndarray:
        return self._mlp(self._belt(belt150)[None])[0].astype(np.float64)

    def encode_batch(self, pc_list, props, belts) -> np.ndarray:
        return self._mlp(np.stack([self._belt(b) for b in belts])).astype(np.float64)


class BeltCloudPointNetEncoder:
    """Belt cloud only (``encoder_type`` belt_cloud_pointnet): the colour-segmented belt cloud in
    the known pulley frame (any M) -> ``belt_cloud.resample_deterministic`` -> PointNet (folded
    BN) + fusion MLP, float64. ``prop40``/``belt150`` are ignored; ``last_empty`` flags M == 0."""

    def __init__(self, weights: dict) -> None:
        self.num_points = int(_scalar(weights["n_points"]))
        self.frame = str(_scalar(weights["frame"]))
        self.proprio_dim = int(_scalar(weights["encoder_proprio_dim"]))
        self.belt_num_points = int(_scalar(weights["encoder_belt_num_points"]))
        self.point_feat_dim = int(_scalar(weights["encoder_point_feat_dim"]))
        if self.frame != SEG_FRAME:
            raise ValueError(f"{BELT_CLOUD_TYPE}: frame {self.frame!r} != {SEG_FRAME!r}")
        if self.proprio_dim or self.belt_num_points:
            raise ValueError(f"{BELT_CLOUD_TYPE}: proprio/belt inputs must be 0")
        if int(_scalar(weights["encoder_num_points"])) != self.num_points:
            raise ValueError(f"{BELT_CLOUD_TYPE}: encoder_num_points != n_points")
        eps = float(_scalar(weights["encoder_bn_epsilon"]))
        self._layers = [_fold_bn(weights, f"conv{i}", f"bn{i}", eps, np.float64)
                        for i in (1, 2, 3)]
        self._w0 = weights[f"{_ENC}fuse_mlp__0__weight"].T.astype(np.float64)
        self._b0 = weights[f"{_ENC}fuse_mlp__0__bias"].astype(np.float64)
        self._w2 = weights[f"{_ENC}fuse_mlp__2__weight"].T.astype(np.float64)
        self._b2 = weights[f"{_ENC}fuse_mlp__2__bias"].astype(np.float64)
        self.latent_dim = int(self._b2.shape[0])
        if (self._w0.shape[0] != self.point_feat_dim
                or self._layers[2][1].shape[0] != self.point_feat_dim):
            raise ValueError(f"encoder weights do not match point_feat_dim {self.point_feat_dim}")
        self.n_points = self.num_points
        self.last_empty = False

    def preprocess(self, pc_raw) -> np.ndarray:
        """``(num_points, 3)`` float32 (zeros for an empty cloud)."""
        return resample_deterministic(pc_raw, self.num_points)[0]

    def pc_global(self, pc_resized) -> np.ndarray:
        """``(..., num_points, 3)`` resampled cloud(s) -> ``(..., point_feat_dim)`` float64."""
        x = np.asarray(pc_resized, dtype=np.float64)
        (w1, b1), (w2, b2), (w3, b3) = self._layers
        x = np.maximum(x @ w1 + b1, 0.0)
        x = np.maximum(x @ w2 + b2, 0.0)
        return np.max(x @ w3 + b3, axis=-2)

    def _fuse(self, g: np.ndarray) -> np.ndarray:
        return np.maximum(g @ self._w0 + self._b0, 0.0) @ self._w2 + self._b2

    def encode(self, pc_raw, prop40=None, belt150=None) -> np.ndarray:
        """``z (latent_dim,)`` float64 of a pulley-frame belt cloud ``(M, 3)``."""
        pc, self.last_empty = resample_deterministic(pc_raw, self.num_points)
        return self._fuse(self.pc_global(pc))

    def encode_batch(self, pc_list, props=None, belts=None) -> np.ndarray:
        """``(B, latent_dim)`` float64; ``last_empty`` becomes a ``(B,)`` bool array."""
        pcs, empty = zip(*(resample_deterministic(pc, self.num_points) for pc in pc_list))
        self.last_empty = np.asarray(empty, dtype=bool)
        return self._fuse(self.pc_global(np.stack(pcs)))


class BeltCloudPointMAEEncoder(BeltCloudPointNetEncoder):
    """``encoder_type`` belt_cloud_pointmae: the same belt-cloud input and interface as
    ``BeltCloudPointNetEncoder`` (a subclass, so every belt-cloud code path takes it), encoded by
    the pretrained Point-MAE backbone + MLP head in torch (``pointmae_encoder``). ``pc_global``
    is the head input (max+mean tokens, centroid, scale; ``point_feat_dim`` 772). Not invariant to
    the order of the resampled points (FPS starts at index 0); ``preprocess`` sorts them."""

    encoder_type = POINTMAE_TYPE

    def __init__(self, weights: dict, device=None) -> None:
        from task_common.pointmae_encoder import PointMAETorchEncoder

        self.num_points = int(_scalar(weights["n_points"]))
        self.frame = str(_scalar(weights["frame"]))
        self.proprio_dim = int(_scalar(weights["encoder_proprio_dim"]))
        self.belt_num_points = int(_scalar(weights["encoder_belt_num_points"]))
        if self.frame != SEG_FRAME:
            raise ValueError(f"{POINTMAE_TYPE}: frame {self.frame!r} != {SEG_FRAME!r}")
        if self.proprio_dim or self.belt_num_points:
            raise ValueError(f"{POINTMAE_TYPE}: proprio/belt inputs must be 0")
        if int(_scalar(weights["encoder_num_points"])) != self.num_points:
            raise ValueError(f"{POINTMAE_TYPE}: encoder_num_points != n_points")
        self._net = PointMAETorchEncoder(weights, device)
        self.device = self._net.device
        self.point_feat_dim = self._net.feat_dim
        if int(_scalar(weights["encoder_point_feat_dim"])) != self.point_feat_dim:
            raise ValueError(f"{POINTMAE_TYPE}: encoder_point_feat_dim != {self.point_feat_dim}")
        self.latent_dim = self._net.latent_dim
        self.n_points = self.num_points
        self.last_empty = False

    def pc_global(self, pc_resized) -> np.ndarray:
        """``(..., num_points, 3)`` resampled cloud(s) -> ``(..., 772)`` float64."""
        x = np.asarray(pc_resized, dtype=np.float32)
        return self._net.features(x.reshape(-1, self.num_points, 3)).reshape(*x.shape[:-2], -1)

    def _fuse(self, g: np.ndarray) -> np.ndarray:
        g = np.asarray(g)
        return self._net.head(g.reshape(-1, g.shape[-1])).reshape(*g.shape[:-1], -1)


def encoder_input(encoder, frame: dict) -> np.ndarray:
    """The cloud ``encoder.encode`` takes: ``frame["pcd_belt_seg"]`` (pulley-frame segmented belt)
    for ``BeltCloudPointNetEncoder``, else ``frame["cloud"]`` (camera cloud)."""
    if isinstance(encoder, BeltCloudPointNetEncoder):
        seg = frame.get("pcd_belt_seg")
        if seg is None:
            raise ValueError(f"{BELT_CLOUD_TYPE} encoder needs the segmented belt cloud "
                             "(pcd_belt_seg, --belt-seg)")
        return seg
    return frame["cloud"]


class LatentDecoder:
    """``dec_mlp`` + ``dec_pc`` of the checkpoint: ``z (16,) -> belt points (num_points, 3)``,
    float32: 150 ordered world-frame points, or for ``belt_cloud_pointnet`` 256 unordered
    points in the known pulley frame."""

    def __init__(self, weights: dict) -> None:
        self._layers = [
            (weights[f"{_DEC}{mlp}__{i}__weight"].T.astype(np.float32),
             weights[f"{_DEC}{mlp}__{i}__bias"].astype(np.float32))
            for mlp in ("dec_mlp", "dec_pc") for i in (0, 2)]
        self.num_points = int(_scalar(weights["reconstruction_num_points"]))
        self.latent_dim = int(self._layers[0][0].shape[0])
        self.checkpoint_sha256 = str(_scalar(weights["checkpoint_sha256"]))
        self.frame = str(_scalar(weights["frame"])) if "frame" in weights else "world"
        if self._layers[-1][1].shape[0] != 3 * self.num_points:
            raise ValueError(f"dec_pc output {self._layers[-1][1].shape[0]} != 3 x "
                             f"{self.num_points} points")

    @classmethod
    def load(cls, decoder_npz) -> LatentDecoder:
        return cls(_load_npz(decoder_npz))

    def decode_batch(self, z) -> np.ndarray:
        """``(B, latent_dim)`` -> ``(B, num_points, 3)`` float32."""
        x = np.asarray(z, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.latent_dim:
            raise ValueError(f"z shape {x.shape}, expected (B, {self.latent_dim})")
        # ReLU after every layer except the last (dec_mlp ends in a ReLU).
        for i, (w, b) in enumerate(self._layers):
            x = x @ w + b
            if i < len(self._layers) - 1:
                x = np.maximum(x, 0.0)
        return x.reshape(-1, self.num_points, 3)

    def decode(self, z) -> np.ndarray:
        return self.decode_batch(np.asarray(z).reshape(1, -1))[0]


class LearnedLcs:
    """The exported learned LCS: ``z+ = A z + B u + D lam + d``, ``lam`` from Nesterov PGD.

    With a state head (``bz__*`` keys) ``B = B(z)`` (and ``d = d(z)`` if ``bz_delta_d``);
    ``rollout(freeze=True)`` holds ``B(z0)`` over the horizon, as the controller does per solve.
    """

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
        self.has_head = "bz__w1" in d
        if self.has_head:
            self._load_head(d)
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

    def _load_head(self, d: dict) -> None:
        self.bz_hidden = int(_scalar(d["bz__hidden"]))
        self.bz_act = str(_scalar(d["bz__act"]))
        self.bz_delta_d = bool(_scalar(d["bz__delta_d"]))
        for k in ("w1", "b1", "w2", "b2"):
            setattr(self, f"bz_{k}", np.asarray(d[f"bz__{k}"], dtype=np.float64))
        h, nx, nu = self.bz_hidden, self.n_x, self.n_u
        m = nx * nu + (nx if self.bz_delta_d else 0)
        want = {"w1": (h, nx), "b1": (h,), "w2": (m, h), "b2": (m,)}
        for k, shape in want.items():
            if getattr(self, f"bz_{k}").shape != shape:
                raise ValueError(f"bz__{k} shape {getattr(self, f'bz_{k}').shape} != {shape}")
        if self.bz_act != "tanh":
            raise ValueError(f"bz__act {self.bz_act!r} unsupported")

    @classmethod
    def load(cls, deploy_npz) -> LearnedLcs:
        return cls(_load_npz(deploy_npz))

    def head_out(self, z) -> np.ndarray:
        """``W2 tanh(W1 z + b1) + b2``: ``(M,)`` per z, ``(K, M)`` for ``(K, n_x)``."""
        z = np.asarray(z, dtype=np.float64)
        return np.tanh(z @ self.bz_w1.T + self.bz_b1) @ self.bz_w2.T + self.bz_b2

    def B_at(self, z) -> np.ndarray:
        if not self.has_head:
            return self.B
        o = self.head_out(z)
        return self.B + o[: self.n_x * self.n_u].reshape(self.n_x, self.n_u)

    def d_at(self, z) -> np.ndarray:
        if not (self.has_head and self.bz_delta_d):
            return self.d
        return self.d + self.head_out(z)[self.n_x * self.n_u:]

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

    def step(self, z, u, iters: int = 25, tol: float = 0.0, B=None, d=None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(z_next, lam, slack)``; defaults = the trained fixed 25-iteration solver.

        ``B``/``d`` default to ``B_at(z)``/``d_at(z)`` (``B``/``d`` without a head).
        """
        z = np.asarray(z, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        lam = self.solve_lambda(z, u, iters, tol)
        if self.has_head:
            B = self.B_at(z) if B is None else B
            d = self.d_at(z) if d is None else d
        else:
            B = self.B if B is None else B
            d = self.d if d is None else d
        z_next = self.A @ z + B @ u + self.D @ lam + d
        slack = self.F @ lam + self.E @ z + self.H @ u + self.c
        return z_next, lam, slack

    def rollout(self, z0, U, iters: int = 25, tol: float = 0.0, freeze: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
        """``(Z (T+1, n_x), Lam (T, n_lam))`` for inputs ``U (T, n_u)``.

        ``freeze``: ``B(z0)``/``d(z0)`` for every step (deploy); else re-evaluated per step.
        """
        U = np.asarray(U, dtype=np.float64).reshape(-1, self.n_u)
        Z = [np.asarray(z0, dtype=np.float64)]
        B = d = None
        if freeze and self.has_head:
            B, d = self.B_at(Z[0]), self.d_at(Z[0])
        lams = []
        for u in U:
            z_next, lam, _ = self.step(Z[-1], u, iters, tol, B=B, d=d)
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
