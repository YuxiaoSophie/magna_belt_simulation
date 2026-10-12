"""Torch ``belt_cloud_pointmae`` encoder of a deploy export (Point-MAE backbone + MLP head).

Minimal copy of ``lcs_learning/pointmae_backbone.py`` (same state-dict names, same op order):
the pretrained backbone comes from ``encoder_pm_weights`` (sha256 checked), the head and any
fine-tuned blocks from the deploy's ``encoder__*`` arrays. float32 on the chosen device, TF32
off inside ``features`` so CUDA matches the exporter's reference z.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CKPT_PREFIX = "module.MAE_encoder."
TRANS_DIM = 384
NUM_HEADS = 6


@functools.cache
def _sha256(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path) -> str:
    st = Path(path).stat()
    return _sha256(str(Path(path).resolve()), st.st_size, st.st_mtime_ns)


def _fps(xyz: torch.Tensor, g: int) -> torch.Tensor:
    b, n, _ = xyz.shape
    idx = torch.zeros(b, g, dtype=torch.long, device=xyz.device)
    dist = torch.full((b, n), float("inf"), device=xyz.device, dtype=xyz.dtype)
    far = torch.zeros(b, dtype=torch.long, device=xyz.device)
    ar = torch.arange(b, device=xyz.device)
    for i in range(g):
        idx[:, i] = far
        dist = torch.minimum(dist, (xyz - xyz[ar, far][:, None, :]).pow(2).sum(-1))
        far = dist.argmax(dim=1)
    return idx


def _group(xyz: torch.Tensor, g: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    ar = torch.arange(xyz.shape[0], device=xyz.device)[:, None]
    center = xyz[ar, _fps(xyz, g)]
    knn = torch.cdist(center, xyz).topk(m, dim=-1, largest=False).indices
    return xyz[ar[:, :, None], knn] - center[:, :, None, :], center


def _normalize(xyz: torch.Tensor, mode: str, fixed_scale: float):
    if mode == "per_cloud":
        c = xyz.mean(dim=1)
        s = (xyz - c[:, None]).norm(dim=-1).amax(dim=1).clamp(min=1e-6)
    else:
        c = xyz.new_zeros(xyz.shape[0], 3)
        s = xyz.new_full((xyz.shape[0],), fixed_scale)
    return (xyz - c[:, None]) / s[:, None, None], c, s


class _Tokenizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(), nn.Conv1d(512, TRANS_DIM, 1))

    @staticmethod
    def _pw(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(conv.weight[:, :, 0], x) + conv.bias[:, None]

    def forward(self, nb: torch.Tensor) -> torch.Tensor:
        b, g, m, _ = nb.shape
        c1, bn1, _, c2 = self.first_conv
        c3, bn2, _, c4 = self.second_conv
        f = self._pw(c2, F.relu(bn1(self._pw(c1, nb.reshape(b * g, m, 3).transpose(1, 2)))))
        f = torch.cat([f.amax(dim=2, keepdim=True).expand(-1, -1, m), f], dim=1)
        return self._pw(c4, F.relu(bn2(self._pw(c3, f)))).amax(dim=2).reshape(b, g, TRANS_DIM)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(TRANS_DIM, 4 * TRANS_DIM)
        self.fc2 = nn.Linear(4 * TRANS_DIM, TRANS_DIM)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(TRANS_DIM, 3 * TRANS_DIM, bias=False)
        self.proj = nn.Linear(TRANS_DIM, TRANS_DIM)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, NUM_HEADS, c // NUM_HEADS).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(TRANS_DIM)
        self.attn = _Attention()
        self.norm2 = nn.LayerNorm(TRANS_DIM)
        self.mlp = _Mlp()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _Blocks(nn.Module):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(depth))


class _Backbone(nn.Module):
    def __init__(self, g: int, m: int, norm: str, fixed_scale: float, depth: int) -> None:
        super().__init__()
        self.g, self.m, self.norm_mode, self.fixed_scale = g, m, norm, fixed_scale
        self.encoder = _Tokenizer()
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, TRANS_DIM))
        self.blocks = _Blocks(depth)
        self.norm = nn.LayerNorm(TRANS_DIM)

    def forward(self, pc):
        xyz, c, s = _normalize(pc, self.norm_mode, self.fixed_scale)
        nb, center = _group(xyz, self.g, self.m)
        x = self.encoder(nb)
        pos = self.pos_embed(center)
        for blk in self.blocks.blocks:
            x = blk(x + pos)
        x = self.norm(x)
        return torch.cat([x.amax(1), x.mean(1)], -1), c, s


class _Encoder(nn.Module):
    def __init__(self, g, m, norm, fixed_scale, depth, hidden, latent_dim, pos_unit) -> None:
        super().__init__()
        self.backbone = _Backbone(g, m, norm, fixed_scale, depth)
        self.head = nn.Sequential(
            nn.Linear(2 * TRANS_DIM + 4, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim))
        self.pos_unit = pos_unit

    def features(self, pc):
        gl, c, s = self.backbone(pc)
        return torch.cat([gl, c / self.pos_unit, s[:, None] / self.pos_unit], dim=-1)


@contextlib.contextmanager
def _no_tf32():
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class PointMAETorchEncoder:
    """``features(pcs (B, N, 3)) -> (B, 772)`` and ``head(g) -> (B, latent_dim)``, float64 out."""

    def __init__(self, weights: dict, device: str | torch.device | None = None) -> None:
        def sc(k):
            return np.asarray(weights[k]).item()

        path = Path(str(sc("encoder_pm_weights")))
        sha = str(sc("encoder_pm_weights_sha256"))
        if not path.is_file():
            raise FileNotFoundError(f"Point-MAE backbone {path} missing (sha256 {sha})")
        if file_sha256(path) != sha:
            raise ValueError(f"{path}: sha256 != the deploy's {sha}")
        state = {k[len("encoder__"):].replace("__", "."): v for k, v in weights.items()
                 if k.startswith("encoder__")}
        latent_dim = int(state["head.4.bias"].shape[0])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        net = _Encoder(int(sc("encoder_pm_num_group")), int(sc("encoder_pm_group_size")),
                       str(sc("encoder_pm_norm")), float(sc("encoder_pm_fixed_scale")),
                       int(sc("encoder_pm_depth")), int(sc("encoder_pm_head_hidden")),
                       latent_dim, float(sc("encoder_pm_pos_unit")))
        sd = torch.load(path, map_location="cpu", weights_only=False)["base_model"]
        full = {f"backbone.{k[len(CKPT_PREFIX):]}": v for k, v in sd.items()
                if k.startswith(CKPT_PREFIX)}
        full = {k: v for k, v in full.items() if k in net.state_dict()}
        full.update({k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in state.items()})
        net.load_state_dict(full, strict=True)
        self.net = net.to(self.device).eval().requires_grad_(False)
        self.latent_dim = latent_dim
        self.feat_dim = 2 * TRANS_DIM + 4

    @torch.no_grad()
    def features(self, pcs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(pcs, dtype=np.float32)).to(self.device)
        with _no_tf32():
            return self.net.features(x).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def head(self, g: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(g, dtype=np.float32)).to(self.device)
        with _no_tf32():
            return self.net.head(x).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def encode(self, pcs: np.ndarray) -> np.ndarray:
        """(B, N, 3) -> (B, latent_dim) without the float64 round trip of features + head."""
        x = torch.from_numpy(np.ascontiguousarray(pcs, dtype=np.float32)).to(self.device)
        with _no_tf32():
            return self.net.head(self.net.features(x)).cpu().numpy().astype(np.float64)
