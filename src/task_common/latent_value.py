"""Learned cost-to-go ``V(z)`` on a deploy's latent (``scripts/lcs/train_value.py``, V12-39).

An MLP on ``x = (z - z_mean) / z_std`` (``z_std`` = the deploy's whitening, checked on load
against the solver's), ReLU hidden layers, ``V = v_max * sigmoid(y)`` in seconds: the discounted
time to the human-labelled seated state (``v_max`` = never). Pure numpy weights in an npz; the
torch twin for MPPI is ``LatentValue.torch_params`` + ``value_forward`` (static tensors, CUDA-graph
safe). ``encoder_sha256`` = ``bc_prior.encoder_fingerprint`` of the deploy whose encoder made the
training latents.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "latent_value_v1"
WHERE = ("terminal", "all_knots")
VALUE_KEYS = frozenset({"path", "weight", "where"})
Z_STD_RTOL = 1e-6


class LatentValue:
    """``value.npz`` weights; ``__call__`` float64 ``V (...)`` of ``Z (..., n_x)``."""

    def __init__(self, d: dict) -> None:
        if str(np.asarray(d["format"]).item()) != FORMAT:
            raise ValueError(f"latent value format {d['format']!r} != {FORMAT}")
        n = int(np.asarray(d["n_layers"]).item())
        self.layers = [(np.asarray(d[f"w{i}"], dtype=np.float64),
                        np.asarray(d[f"b{i}"], dtype=np.float64)) for i in range(n)]
        self.z_mean = np.asarray(d["z_mean"], dtype=np.float64)
        self.z_std = np.asarray(d["z_std"], dtype=np.float64)
        self.v_max = float(np.asarray(d["v_max"]).item())
        self.n_x = int(self.z_mean.shape[0])
        self.encoder_sha256 = str(np.asarray(d.get("encoder_sha256", "")).item())
        self.meta = json.loads(str(np.asarray(d["meta"]).item())) if "meta" in d else {}
        if self.z_std.shape != (self.n_x,) or np.any(self.z_std <= 0.0) \
                or not np.isfinite(self.v_max) or self.v_max <= 0.0:
            raise ValueError("latent value: z_std shape / > 0, v_max finite > 0")
        if self.layers[0][0].shape[0] != self.n_x or self.layers[-1][1].shape != (1,):
            raise ValueError(f"latent value layer shapes do not match {self.n_x} -> 1")

    @classmethod
    def load(cls, path) -> LatentValue:
        with np.load(Path(path), allow_pickle=False) as f:
            return cls({k: f[k] for k in f.files})

    def check_z_std(self, z_std) -> None:
        """Refuse a deploy whose whitening differs (another latent space)."""
        z_std = np.asarray(z_std, dtype=np.float64)
        if z_std.shape != self.z_std.shape or not np.allclose(z_std, self.z_std, rtol=Z_STD_RTOL,
                                                              atol=0.0):
            raise ValueError("latent value z_std != the deploy's z_std (trained on another "
                             "latent space)")

    def __call__(self, Z) -> np.ndarray:
        x = (np.asarray(Z, dtype=np.float64) - self.z_mean) / self.z_std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i + 1 < len(self.layers):
                x = np.maximum(x, 0.0)
        return self.v_max / (1.0 + np.exp(-x[..., 0]))

    def torch_params(self, tensor) -> tuple:
        """Static tensors for ``value_forward``; ``tensor`` = ``TorchLcs.tensor``."""
        return (tensor(self.z_mean), tensor(1.0 / self.z_std),
                [(tensor(w), tensor(b)) for w, b in self.layers], self.v_max)


def value_forward(vt: tuple, Z):
    """Torch ``V (...)`` of ``Z (..., n_x)`` with ``vt = LatentValue.torch_params``."""
    import torch

    mean, inv_std, layers, v_max = vt
    x = (Z - mean) * inv_std
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i + 1 < len(layers):
            x = torch.relu(x)
    return v_max * torch.sigmoid(x[..., 0])


def check_value(v: dict) -> None:
    """``MppiParams.value`` keys and ranges (the npz is checked when a solver loads it)."""
    if not isinstance(v, dict) or not set(v) <= VALUE_KEYS or not {"path", "weight"} <= set(v):
        raise ValueError(f"value {v!r}: need path, weight, within {sorted(VALUE_KEYS)}")
    if not isinstance(v["path"], (str, Path)):
        raise TypeError(f"value.path {v['path']!r} must be an npz path")
    w = float(v["weight"])
    if not (np.isfinite(w) and w >= 0.0):
        raise ValueError(f"value.weight {v['weight']} must be finite >= 0")
    if v.get("where", "terminal") not in WHERE:
        raise ValueError(f"value.where {v.get('where')!r}: one of {WHERE}")


def save(path, layers, z_mean, z_std, v_max: float, encoder_sha256: str, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrs = {f"w{i}": np.asarray(w, np.float64) for i, (w, _) in enumerate(layers)}
    arrs.update({f"b{i}": np.asarray(b, np.float64) for i, (_, b) in enumerate(layers)})
    np.savez(path, format=FORMAT, n_layers=len(layers), z_mean=np.asarray(z_mean, np.float64),
             z_std=np.asarray(z_std, np.float64), v_max=float(v_max),
             encoder_sha256=encoder_sha256, meta=json.dumps(meta), **arrs)
