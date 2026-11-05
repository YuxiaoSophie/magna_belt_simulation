"""Torch-free learned latent encoder, learned-LCS step and the ``LATENT_STATE`` LCM message.

``LatentEncoder`` reproduces ``lcs_learning``'s ``PointNetObsAutoEncoder.encode`` from an exported
``deploy.npz`` (BatchNorm folded, float32 inside; also ``encoder_type`` scene_belt_prop_pointnet,
the belt-cloud trainer's V4a-input model); ``BeltMlpEncoder`` and
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
from task_common.belt_cloud import SEG_FRAME, anchor_g, resample_deterministic
from task_common.lcs_dataset import BELT_BODIES, BELT_POINTS, STATE_DIM

LATENT_STATE_CHANNEL = "LATENT_STATE"
LATENT_METADATA_NAME = "latent_encoder"
LATENT_BLOCK_NAMES = ("latent", "ee_pose_franka", "ee_pose_ur", "proprio")
_ENC = "encoder__"
_DEC = "decoder__"
BELT_CLOUD_TYPE = "belt_cloud_pointnet"
POINTMAE_TYPE = "belt_cloud_pointmae"
SCENE_BELT_PROP_TYPE = "scene_belt_prop_pointnet"  # V4a's inputs, trained by the belt-cloud recipe
GRIP_TYPE = "belt_cloud_pointnet_grip"  # V12-131: + 4 gripper features (encoder_grip_dim)
# gripper features: lcs_learning.belt_cloud_ae's definition (kept equal; check_grip_encoder)
GRIP_DIM = 4  # [franka gap / 20 mm, robotiq position / 255, franka bit, ur bit]
FRANKA_HELD_MM = (3.0, 20.0)  # measured finger gap of fingers stopped on the belt
FRANKA_CLOSED_MM = 20.0  # hand command below this = closing
UR_CLOSED_BYTE = 230  # Robotiq command >= this = closing (the UR exit opens to 191)
UR_FULL_BYTE = 254  # Robotiq position >= this = closed on nothing
FRANKA_U, UR_U = (0, 1, 2, 6, 7, 8), (3, 4, 5, 9, 10, 11)  # each arm's 6 u channels


def grip_features(gap_mm, ur_pos, franka_cmd_closed, ur_cmd_closed) -> np.ndarray:
    """(T, 4) float32 (or (4,) for scalars): measured finger gap / 20 mm, measured Robotiq
    position / 255, Franka grasp bit (closing command, gap stopped in FRANKA_HELD_MM), UR grasp bit
    (closing command, position stopped short of full closure)."""
    one = np.ndim(gap_mm) == 0
    gap = np.asarray(gap_mm, np.float64).reshape(-1)
    pos = np.asarray(ur_pos, np.float64).reshape(-1)
    fb = (np.asarray(franka_cmd_closed, bool).reshape(-1) & (gap >= FRANKA_HELD_MM[0])
          & (gap <= FRANKA_HELD_MM[1]))
    ub = np.asarray(ur_cmd_closed, bool).reshape(-1) & (pos >= UR_CLOSED_BYTE) & (pos < UR_FULL_BYTE)
    g = np.stack([gap / 20.0, pos / 255.0, fb, ub], axis=1).astype(np.float32)
    return g[0] if one else g


def recorded_grip_features(hand_mm, robotiq_pos, cmd_source=None) -> np.ndarray:
    """``grip_features`` of a recorded episode (``sim_hand_mm`` / ``sim_robotiq_byte``, both
    measured); the unlogged commands as lcs_learning reconstructs them (Franka closing except
    release frames, cmd source 6; Robotiq closing while its position is >= UR_CLOSED_BYTE)."""
    pos = np.asarray(robotiq_pos, np.float64).reshape(-1)
    f_cmd = np.ones(len(pos), bool) if cmd_source is None else np.asarray(cmd_source) != 6
    return grip_features(np.asarray(hand_mm, np.float64).reshape(-1), pos, f_cmd,
                         pos >= UR_CLOSED_BYTE)


# V12-131 seg_ctx: z = [z_belt (belt cloud), z_ctx (scene cloud, EE pose / twist, grip)], board
# frame; lcs_learning.ctx_features' definition (kept equal; check_ctx_encoder)
CTX_TYPE = "belt_cloud_pointnet_ctx"
CTXV_TYPE = "belt_cloud_pointnet_ctxv"  # V12-133: z_ctx = (ctx numbers - shift) * gain, no scene
RV_TYPE = "scene_rv_dp3"  # V12-133 V61e: z = (W_s f_scene + b) + W_r f_robot (lcs_learning SceneRvAE)
CTX_KIN_DIM = 30  # per arm: xyz, rot6d (R_B columns 0, 1), V_SCALE * v, W_SCALE * w
CTX_V_SCALE, CTX_W_SCALE = 20.0, 2.0


def ctx_frames(X_WB, pulley_pose7) -> dict:
    """``X_WB``, ``X_BW``, ``X_WP`` (known large pulley, xyz wxyz), ``X_BP`` (4x4 float64)."""
    X_WB = np.asarray(X_WB, np.float64)
    X_BW = np.linalg.inv(X_WB)
    X_WP = np.eye(4)
    X_WP[:3, :3], X_WP[:3, 3] = _quat_wxyz_mat(pulley_pose7[3:]), pulley_pose7[:3]
    return {"X_WB": X_WB, "X_BW": X_BW, "X_WP": X_WP, "X_BP": X_BW @ X_WP,
            "pulley_pose7": np.asarray(pulley_pose7, np.float64)}


def _quat_wxyz_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def board_points(X, pts) -> np.ndarray:
    """``(N, 3)`` float32 of ``X`` (4x4) applied to ``pts`` (lcs_learning does the same)."""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 3).astype(np.float64)
    return (p @ X[:3, :3].T + X[:3, 3]).astype(np.float32)


def ctx_kinematics(prop40, X_BW) -> np.ndarray:
    """(30,) EE pose (xyz, rot6d) and twist (J(q) q_dot of the measured joint velocities) of
    both arms in the board frame, from the 40-d state (magna's FK)."""
    from round_belt_task.arm_kinematics import FrankaTip, UrTracking

    s = np.asarray(prop40, np.float64)[:STATE_DIM]
    R_BW, t_BW = X_BW[:3, :3], X_BW[:3, 3]
    parts = []
    for ch, q, qd, ee in ((FrankaTip, s[:7], s[13:20], s[26:33]),
                          (UrTracking, s[7:13], s[20:26], s[33:40])):
        R = R_BW @ _quat_wxyz_mat(ee[3:])
        tw = ch.jacobian(q) @ qd
        parts += [R_BW @ ee[:3] + t_BW, R[:, 0], R[:, 1], CTX_V_SCALE * (R_BW @ tw[:3]),
                  CTX_W_SCALE * (R_BW @ tw[3:])]
    return np.concatenate(parts).astype(np.float32)


def grip_gate_vector(grip) -> np.ndarray:
    """(12,) per-channel input gate of a grip vector: each arm's 6 channels times its bit."""
    g = np.ones(12)
    g[list(FRANKA_U)] = float(grip[2])
    g[list(UR_U)] = float(grip[3])
    return g


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


def _fold_bn(w: dict, conv: str, bn: str, eps: float, dtype=np.float32, net: str = "pc_ae"
             ) -> tuple[np.ndarray, np.ndarray]:
    weight = w[f"{_ENC}{net}__{conv}__weight"][:, :, 0]
    bias = w[f"{_ENC}{net}__{conv}__bias"]
    gamma, beta = w[f"{_ENC}{net}__{bn}__weight"], w[f"{_ENC}{net}__{bn}__bias"]
    mean, var = w[f"{_ENC}{net}__{bn}__running_mean"], w[f"{_ENC}{net}__{bn}__running_var"]
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
        if "encoder_type" in weights and str(_scalar(weights["encoder_type"])) in (
                BELT_CLOUD_TYPE, GRIP_TYPE):
            return BeltCloudPointNetEncoder(weights)
        if "encoder_type" in weights and str(_scalar(weights["encoder_type"])) in (CTX_TYPE,
                                                                                    CTXV_TYPE):
            return BeltCtxEncoder(weights)
        if "encoder_type" in weights and str(_scalar(weights["encoder_type"])) == RV_TYPE:
            return SceneRvEncoder(weights)
        if "encoder_type" in weights and str(_scalar(weights["encoder_type"])) == POINTMAE_TYPE:
            return BeltCloudPointMAEEncoder(weights, device)
        if "encoder_type" in weights and str(_scalar(weights["encoder_type"])) == (
                SCENE_BELT_PROP_TYPE):
            return LatentEncoder(weights)
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

    _fuse_grip = True  # BeltCtxEncoder: the grip goes to z_ctx, not the belt fusion

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
        # V12-131: gripper features after the PointNet feature (0 = cloud only)
        self.grip_dim = int(_scalar(weights.get("encoder_grip_dim", 0)))
        eps = float(_scalar(weights["encoder_bn_epsilon"]))
        self._layers = [_fold_bn(weights, f"conv{i}", f"bn{i}", eps, np.float64)
                        for i in (1, 2, 3)]
        self._w0 = weights[f"{_ENC}fuse_mlp__0__weight"].T.astype(np.float64)
        self._b0 = weights[f"{_ENC}fuse_mlp__0__bias"].astype(np.float64)
        self._w2 = weights[f"{_ENC}fuse_mlp__2__weight"].T.astype(np.float64)
        self._b2 = weights[f"{_ENC}fuse_mlp__2__bias"].astype(np.float64)
        self.latent_dim = int(self._b2.shape[0])
        if (self._w0.shape[0] != self.point_feat_dim + self.grip_dim * self._fuse_grip
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

    def _fuse(self, g: np.ndarray, grip=None) -> np.ndarray:
        if self.grip_dim and self._fuse_grip:
            if grip is None:
                raise ValueError(f"{GRIP_TYPE} encoder needs the gripper features (grip=)")
            g = np.concatenate([g, np.broadcast_to(np.asarray(grip, np.float64),
                                                   (*g.shape[:-1], self.grip_dim))], axis=-1)
        return np.maximum(g @ self._w0 + self._b0, 0.0) @ self._w2 + self._b2

    def encode(self, pc_raw, prop40=None, belt150=None, grip=None) -> np.ndarray:
        """``z (latent_dim,)`` float64 of a pulley-frame belt cloud ``(M, 3)`` (+ ``grip`` (4,)
        when ``grip_dim``)."""
        pc, self.last_empty = resample_deterministic(pc_raw, self.num_points)
        return self._fuse(self.pc_global(pc), grip)

    def encode_batch(self, pc_list, props=None, belts=None, grips=None) -> np.ndarray:
        """``(B, latent_dim)`` float64; ``last_empty`` becomes a ``(B,)`` bool array."""
        pcs, empty = zip(*(resample_deterministic(pc, self.num_points) for pc in pc_list))
        self.last_empty = np.asarray(empty, dtype=bool)
        return self._fuse(self.pc_global(np.stack(pcs)), grips)


class BeltCtxEncoder(BeltCloudPointNetEncoder):
    """``encoder_type`` belt_cloud_pointnet_ctx (V12-131): ``z = [z_belt, z_ctx]``. z_belt = the
    belt_cloud_pointnet encoder on the belt_seg cloud moved to the board frame (``ctx_X_BP``);
    z_ctx = MLP([PointNet(scene camera cloud, board frame (``ctx_X_BW``), ``scene_points``),
    ``ctx_kinematics``, grip]). The deploy's frames are the trained board pose; ``set_board``
    swaps in a runtime ``X_WB`` (refused when it differs: a new board pose needs a re-export)."""

    _fuse_grip = False

    def __init__(self, weights: dict) -> None:
        super().__init__(weights)
        self.belt_dim = self.latent_dim
        self.ctx_dim = int(_scalar(weights["encoder_ctx_dim"]))
        self.scene_points = int(_scalar(weights["encoder_scene_points"]))
        self.ctx_affine = None  # ctxv: (shift, gain) of the 34 ctx numbers
        # V61c / V61h: a subset of the 34 numbers (no EE velocities); absent = all 34
        self.ctx_keep = (np.asarray(weights["encoder_ctx_keep"], np.int64)
                         if "encoder_ctx_keep" in weights else None)
        n = CTX_KIN_DIM + GRIP_DIM if self.ctx_keep is None else len(self.ctx_keep)
        if str(_scalar(weights["encoder_type"])) == CTXV_TYPE:
            self.ctx_affine = (np.asarray(weights["encoder_ctx_shift"], np.float64),
                               np.asarray(weights["encoder_ctx_gain"], np.float64))
            if (self.grip_dim != GRIP_DIM or self.scene_points or self.ctx_dim != n
                    or self.ctx_affine[0].shape != (self.ctx_dim,)):
                raise ValueError(f"{CTXV_TYPE}: ctx keys do not match the 34 ctx numbers")
        else:
            eps = float(_scalar(weights["encoder_bn_epsilon"]))
            self._ctx_layers = [_fold_bn(weights, f"conv{i}", f"bn{i}", eps, np.float64,
                                         "ctx_pc") for i in (1, 2, 3)]
            self._cw = [(weights[f"{_ENC}ctx_mlp__{i}__weight"].T.astype(np.float64),
                         weights[f"{_ENC}ctx_mlp__{i}__bias"].astype(np.float64)) for i in (0, 2)]
            if (self.grip_dim != GRIP_DIM or self._cw[1][1].shape[0] != self.ctx_dim
                    or self._cw[0][0].shape[0] != self._ctx_layers[2][1].shape[0] + n):
                raise ValueError(f"{CTX_TYPE}: ctx weights do not match the ctx inputs")
        self.frames = {k: np.asarray(weights[f"ctx_{k}"], np.float64)
                       for k in ("X_WB", "X_BW", "X_BP")}
        self.latent_dim = self.belt_dim + self.ctx_dim
        # belt points [xyz, this x along-belt distance from the Franka tip]; 0 = xyz only
        self.anchor_g = float(_scalar(weights.get("encoder_anchor_g", 0.0)))
        self.anchor_fold = float(_scalar(weights.get("encoder_anchor_fold", 0.0)))  # 0 = no fold

    def set_board(self, X_WB) -> None:
        # shortcut: refuses a board pose other than the trained one; re-export for a new board
        if not np.allclose(np.asarray(X_WB, np.float64), self.frames["X_WB"], rtol=0, atol=1e-9):
            raise ValueError(f"{CTX_TYPE}: runtime board pose != the deploy's ctx_X_WB")

    def _ctx(self, scene, prop40, grip) -> np.ndarray:
        if self.ctx_affine is not None:  # ctxv: the scaled ctx numbers (float32 as trained)
            if prop40 is None or grip is None:
                raise ValueError(f"{CTXV_TYPE} encoder needs prop40 and grip=")
            vec = np.concatenate([ctx_kinematics(prop40, self.frames["X_BW"]),
                                  np.asarray(grip, np.float32).reshape(GRIP_DIM)])
            if self.ctx_keep is not None:
                vec = vec[self.ctx_keep]
            return (vec.astype(np.float64) - self.ctx_affine[0]) * self.ctx_affine[1]
        if scene is None or prop40 is None or grip is None:
            raise ValueError(f"{CTX_TYPE} encoder needs scene=, prop40 and grip=")
        x, _ = resample_deterministic(board_points(self.frames["X_BW"], scene), self.scene_points)
        x = np.asarray(x, np.float64)
        (w1, b1), (w2, b2), (w3, b3) = self._ctx_layers
        x = np.maximum(x @ w1 + b1, 0.0)
        x = np.maximum(x @ w2 + b2, 0.0)
        vec = np.concatenate([ctx_kinematics(prop40, self.frames["X_BW"]).astype(np.float64),
                              np.asarray(grip, np.float64).reshape(GRIP_DIM)])
        if self.ctx_keep is not None:
            vec = vec[self.ctx_keep]
        h = np.concatenate([np.max(x @ w3 + b3, axis=-2), vec])
        (w0, b0), (w2_, b2_) = self._cw
        return np.maximum(h @ w0 + b0, 0.0) @ w2_ + b2_

    def preprocess(self, pc_raw) -> np.ndarray:
        """The pulley-frame belt cloud moved to the board frame, then resampled."""
        return resample_deterministic(board_points(self.frames["X_BP"], pc_raw),
                                      self.num_points)[0]

    def encode_belt(self, pc_raw, prop40=None) -> np.ndarray:
        """z_belt of a pulley-frame belt cloud (``prop40``: the anchor's Franka tip)."""
        pc, self.last_empty = resample_deterministic(
            board_points(self.frames["X_BP"], pc_raw), self.num_points)
        if self.anchor_g > 0.0:
            if prop40 is None:
                raise ValueError(f"{CTX_TYPE} anchor encoder needs prop40")
            tip = ctx_kinematics(prop40, self.frames["X_BW"]).astype(np.float32)[:3]
            g = (np.zeros(len(pc), np.float32) if self.last_empty
                 else np.float32(self.anchor_g) * anchor_g(pc, tip, self.anchor_fold))
            pc = np.concatenate([pc, g[:, None]], axis=1)
        return self._fuse(self.pc_global(pc))

    def encode(self, pc_raw, prop40=None, belt150=None, grip=None, scene=None) -> np.ndarray:
        zb = self.encode_belt(pc_raw, prop40)
        empty = self.last_empty
        z = np.concatenate([zb, self._ctx(scene, prop40, grip)])
        self.last_empty = empty
        return z

    def encode_batch(self, pc_list, props=None, belts=None, grips=None, scenes=None
                     ) -> np.ndarray:
        n = len(pc_list)
        props, grips, scenes = ([None] * n if v is None else v for v in (props, grips, scenes))
        out, empty = [], []
        for pc, p, g, sc in zip(pc_list, props, grips, scenes, strict=True):
            out.append(self.encode(pc, p, None, g, sc))
            empty.append(self.last_empty)
        self.last_empty = np.asarray(empty, dtype=bool)
        return np.stack(out)


def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps) * w + b


class SceneRvEncoder(BeltCtxEncoder):
    """``encoder_type`` scene_rv_dp3 (V61e): DP3-style scene branch on the board-frame scene camera
    cloud (``scene_points``) and robot vector rv (``encoder_rv_idx`` of the 34 ctx numbers), one
    linear fusion: ``z = scene_part(scene) + robot_part(prop40, grip)`` with ``scene_part = W_s
    f_scene + b`` and ``robot_part = W_r f_robot``. Goals keep only their scene part (the node
    adds the current robot part), so ``z - goal`` does not depend on rv. The belt_seg cloud
    (``encoder_input``) only sets ``last_empty``; float64."""

    skip_cloud_checks = True  # check_latent_encoder E1 / E10 test the belt-cloud resample path

    def __init__(self, weights: dict) -> None:  # no PointNet / fuse_mlp: not the parent's init
        self.num_points = int(_scalar(weights["n_points"]))
        self.frame = str(_scalar(weights["frame"]))
        self.scene_points = int(_scalar(weights["encoder_scene_points"]))
        self.grip_dim = int(_scalar(weights["encoder_grip_dim"]))
        self.proprio_dim = self.belt_num_points = self.ctx_dim = 0
        self.ctx_affine, self.ctx_keep = None, None
        self.frames = {k: np.asarray(weights[f"ctx_{k}"], np.float64) for k in ("X_WB", "X_BW", "X_BP")}
        self.w = {k[len(_ENC):]: np.asarray(v, np.float64) for k, v in weights.items()
                  if k.startswith(_ENC) and not k.startswith(f"{_ENC}pm_")}
        self.rv_idx = np.asarray(weights["encoder_rv_idx"], np.int64)
        self.rv_shift = np.asarray(weights["encoder_rv_shift"], np.float64)
        self.rv_gain = np.asarray(weights["encoder_rv_gain"], np.float64)
        # "belt" (V61f): the point branch reads the belt_seg cloud instead of the scene cloud
        self.rv_cloud = str(_scalar(weights["encoder_rv_cloud"])) if "encoder_rv_cloud" in weights else "scene"
        self.point_feat_dim = int(self.w["scene_proj__0__weight"].shape[0])
        self.latent_dim = self.belt_dim = int(self.w["fuse__bias"].shape[0])
        if self.frame != SEG_FRAME or self.grip_dim != GRIP_DIM or self.rv_idx.shape != self.rv_shift.shape:
            raise ValueError(f"{RV_TYPE}: frame / grip / rv keys do not match")
        self.n_points = self.num_points
        self.last_empty = False

    def pc_global(self, scene_resized) -> np.ndarray:
        """f_scene ``(..., 64)`` of resampled board-frame scene cloud(s)."""
        x, w = np.asarray(scene_resized, np.float64), self.w
        for i in (0, 3, 6):
            x = x @ w[f"scene_mlp__{i}__weight"].T + w[f"scene_mlp__{i}__bias"]
            x = np.maximum(_ln(x, w[f"scene_mlp__{i + 1}__weight"], w[f"scene_mlp__{i + 1}__bias"]), 0.0)
        f = x.max(axis=-2) @ w["scene_proj__0__weight"].T + w["scene_proj__0__bias"]
        return _ln(f, w["scene_proj__1__weight"], w["scene_proj__1__bias"])

    def scene_part(self, scene) -> np.ndarray:
        """``W_s f_scene + b`` of a raw world-frame scene cloud, or for rv_cloud belt the
        pulley-frame belt_seg cloud (empty -> zeros cloud)."""
        x, empty = self._resample(scene)
        self.last_empty = bool(empty)
        W = self.w["fuse__weight"]
        return self.pc_global(x) @ W[:, :self.point_feat_dim].T + self.w["fuse__bias"]

    def robot_part(self, prop40, grip) -> np.ndarray:
        """``W_r f_robot`` of the current state (40) and grip features (4)."""
        if prop40 is None or grip is None:
            raise ValueError(f"{RV_TYPE} encoder needs prop40 and grip=")
        vec = np.concatenate([ctx_kinematics(prop40, self.frames["X_BW"]),
                              np.asarray(grip, np.float32).reshape(GRIP_DIM)])
        rv = (vec.astype(np.float64)[self.rv_idx] - self.rv_shift) * self.rv_gain
        w = self.w
        h = np.maximum(rv @ w["rv_mlp__0__weight"].T + w["rv_mlp__0__bias"], 0.0)
        f = h @ w["rv_mlp__2__weight"].T + w["rv_mlp__2__bias"]
        return f @ w["fuse__weight"][:, self.point_feat_dim:].T

    def encode(self, pc_raw, prop40=None, belt150=None, grip=None, scene=None) -> np.ndarray:
        if self.rv_cloud == "belt":
            scene = pc_raw
        elif scene is None:
            raise ValueError(f"{RV_TYPE} encoder needs scene= (the camera cloud)")
        zs = self.scene_part(scene)
        return zs + self.robot_part(prop40, grip)

    def _resample(self, cloud):
        if self.rv_cloud == "belt":
            return resample_deterministic(board_points(self.frames["X_BP"], cloud), self.num_points)
        return resample_deterministic(board_points(self.frames["X_BW"], cloud), self.scene_points)

    def preprocess(self, pc_raw) -> np.ndarray:
        return self._resample(pc_raw)[0]


def goal_frame_latents(encoder, npz, frames) -> tuple[np.ndarray, np.ndarray]:
    """``encode_episode_frames``, except a SceneRvEncoder's goal latents keep only their scene part
    (V61e rule: the node adds the CURRENT robot part, so distances to them are W_s df_scene)."""
    if not isinstance(encoder, SceneRvEncoder):
        return encode_episode_frames(encoder, npz, frames)
    zs, empty = [], []
    for t in frames:
        zs.append(encoder.scene_part(npz["pcd_belt_seg" if encoder.rv_cloud == "belt" else "pcd"][t]))
        empty.append(encoder.last_empty)
    return np.stack(zs), np.asarray(empty, bool)


class BeltCloudPointMAEEncoder(BeltCloudPointNetEncoder):
    """``encoder_type`` belt_cloud_pointmae: the same belt-cloud input and interface as
    ``BeltCloudPointNetEncoder`` (a subclass, so every belt-cloud code path takes it), encoded by
    the pretrained Point-MAE backbone + MLP head in torch (``pointmae_encoder``). ``pc_global``
    is the head input (max+mean tokens, centroid, scale; ``point_feat_dim`` 772). Not invariant to
    the order of the resampled points (FPS starts at index 0); ``preprocess`` sorts them."""

    encoder_type = POINTMAE_TYPE
    grip_dim = 0

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

    def _fuse(self, g: np.ndarray, grip=None) -> np.ndarray:
        g = np.asarray(g)
        return self._net.head(g.reshape(-1, g.shape[-1])).reshape(*g.shape[:-1], -1)


def episode_input_keys(encoder) -> tuple[str, ...]:
    """The episode npz keys ``encode_episode_frames`` reads for ``encoder`` (grip encoders also
    read ``sim_cmd_source`` when present: release frames)."""
    if isinstance(encoder, BeltCtxEncoder):
        return ("pcd_belt_seg",) + (("pcd",) if encoder.scene_points else ()) + (
            "state", "sim_hand_mm", "sim_robotiq_byte")
    if isinstance(encoder, BeltCloudPointNetEncoder):
        if getattr(encoder, "grip_dim", 0):
            return ("pcd_belt_seg", "sim_hand_mm", "sim_robotiq_byte")
        return ("pcd_belt_seg",)
    if isinstance(encoder, LatentEncoder):
        return ("pcd", "state", "pcd_belt")
    raise TypeError(f"no episode inputs for {type(encoder).__name__}")


def encode_episode_frames(encoder, npz, frames) -> tuple[np.ndarray, np.ndarray]:
    """``(len(frames), n_x)`` per-frame ``encode`` latents (as the node) and ``(len(frames),)``
    empty flags of a recorded episode (``npz``: an open npz or a dict with
    ``episode_input_keys``): ``pcd_belt_seg`` for the belt-cloud encoders, else
    ``(pcd, state, pcd_belt)``."""
    frames = list(frames)
    if isinstance(encoder, BeltCloudPointNetEncoder):
        seg = npz["pcd_belt_seg"]
        grip = (recorded_grip_features(npz["sim_hand_mm"], npz["sim_robotiq_byte"],
                                       npz.get("sim_cmd_source"))
                if getattr(encoder, "grip_dim", 0) else None)
        zs, empty = [], []
        for t in frames:
            if isinstance(encoder, BeltCtxEncoder):
                zs.append(encoder.encode(np.asarray(seg[t], dtype=np.float32), npz["state"][t],
                                         grip=grip[t],
                                         scene=npz["pcd"][t] if encoder.scene_points else None))
            elif grip is None:
                zs.append(encoder.encode(np.asarray(seg[t], dtype=np.float32)))
            else:
                zs.append(encoder.encode(np.asarray(seg[t], dtype=np.float32), grip=grip[t]))
            empty.append(bool(encoder.last_empty))
        return np.stack(zs), np.asarray(empty, dtype=bool)
    pcd, state, belt = (npz[k] for k in episode_input_keys(encoder))
    z = np.stack([encoder.encode(pcd[t], state[t], belt[t]) for t in frames])
    return z, np.zeros(len(frames), dtype=bool)


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
        # V12-131 seg_ctx: z = [z_belt, z_ctx] (decoder_z_dim), decoded from z_belt only
        self.z_dim = int(_scalar(weights.get("decoder_z_dim", self.latent_dim)))
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
        if x.ndim != 2 or x.shape[1] != self.z_dim:
            raise ValueError(f"z shape {x.shape}, expected (B, {self.z_dim})")
        x = x[:, :self.latent_dim]
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
        # proprio: the 40-d state, + GRIP_DIM gripper features with a grip encoder (V12-131)
        if v.shape != (n,) and not (name == "proprio" and v.shape == (STATE_DIM + GRIP_DIM,)):
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
