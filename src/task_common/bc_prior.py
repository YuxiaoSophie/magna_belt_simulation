"""MILES-BC state-conditioned action prior for MPPI (``scripts/lcs/train_bc_prior.py``).

An MLP ``pi -> u`` (12-d cmd_delta) as pure numpy weights in ``bc_prior.npz``; ``inputs`` "z"
(default: ``pi(z)``, no EE positions, as the no-EE-target rule) or "zp" (``pi(z, p_franka, p_ur)``,
``p`` = world EE positions, the node's p0; files without the key). ``x = (in - in_mean) / in_std``,
ReLU hidden layers, ``u = u_mid + u_half * y`` (the deploy box's centre / half-width; zero-width
channels -> 1).
``encoder_sha256`` = ``encoder_fingerprint`` of the deploy whose encoder made the latents; the
torch rollout used by the node is ``mppi_lcs.PolicyPrior``. numpy only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from task_common.latent_encoder import LearnedLcs

N_P = 6
FORMAT = "bc_prior_v1"
INPUTS = ("z", "zp")


def encoder_fingerprint(deploy_npz) -> str:
    """sha256 over the deploy's ``encoder*`` arrays (name, dtype, shape, bytes; sorted)."""
    h = hashlib.sha256()
    with np.load(Path(deploy_npz), allow_pickle=False) as d:
        for k in sorted(k for k in d.files if k.startswith("encoder")):
            a = np.ascontiguousarray(d[k])
            h.update(f"{k}|{a.dtype}|{a.shape}".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def box_scale(u_lb, u_ub) -> tuple[np.ndarray, np.ndarray]:
    """``(u_mid, u_half)`` of the box; zero-width channels get half 1 (as MPPI's cost scale)."""
    lb, ub = np.asarray(u_lb, dtype=np.float64), np.asarray(u_ub, dtype=np.float64)
    half = (ub - lb) / 2.0
    return (ub + lb) / 2.0, np.where(half > 0.0, half, 1.0)


class BcPrior:
    """``bc_prior.npz`` weights; ``forward`` float64."""

    def __init__(self, d: dict) -> None:
        if str(np.asarray(d["format"]).item()) != FORMAT:
            raise ValueError(f"bc_prior format {d['format']!r} != {FORMAT}")
        n = int(np.asarray(d["n_layers"]).item())
        self.layers = [(np.asarray(d[f"w{i}"], dtype=np.float64),
                        np.asarray(d[f"b{i}"], dtype=np.float64)) for i in range(n)]
        self.in_mean = np.asarray(d["in_mean"], dtype=np.float64)
        self.in_std = np.asarray(d["in_std"], dtype=np.float64)
        self.u_mid = np.asarray(d["u_mid"], dtype=np.float64)
        self.u_half = np.asarray(d["u_half"], dtype=np.float64)
        self.n_x = int(np.asarray(d["n_x"]).item())
        self.n_u = int(self.u_mid.shape[0])
        self.inputs = str(np.asarray(d["inputs"]).item()) if "inputs" in d else "zp"
        if self.inputs not in INPUTS:
            raise ValueError(f"bc_prior inputs {self.inputs!r}: one of {INPUTS}")
        n_in = self.n_x + (N_P if self.inputs == "zp" else 0)
        self.encoder_sha256 = str(np.asarray(d["encoder_sha256"]).item())
        if self.in_mean.shape != (n_in,) or self.in_std.shape != self.in_mean.shape \
                or np.any(self.in_std <= 0.0):
            raise ValueError(f"bc_prior in_mean/in_std shape {self.in_mean.shape} != ({n_in},) "
                             "or std <= 0")
        if self.layers[0][0].shape[0] != n_in or self.layers[-1][1].shape != (self.n_u,):
            raise ValueError(f"bc_prior layer shapes do not match {n_in} -> n_u")

    @classmethod
    def load(cls, path) -> BcPrior:
        with np.load(Path(path), allow_pickle=False) as f:
            return cls({k: f[k] for k in f.files})

    def check_deploy(self, deploy_npz) -> None:
        """Refuse a deploy whose encoder is not the one the prior was trained on."""
        fp = encoder_fingerprint(deploy_npz)
        if fp != self.encoder_sha256:
            raise ValueError(f"bc_prior was trained on encoder {self.encoder_sha256[:12]}, deploy "
                             f"{deploy_npz} has {fp[:12]} (retrain on this deploy's latents)")

    def forward(self, z, p) -> np.ndarray:
        """``u`` (unclipped) for ``z (..., n_x)``, ``p (..., 6)`` (unused with inputs "z")."""
        x = np.asarray(z, np.float64)
        if self.inputs == "zp":
            x = np.concatenate([x, np.asarray(p, np.float64)], -1)
        x = (x - self.in_mean) / self.in_std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i + 1 < len(self.layers):
                x = np.maximum(x, 0.0)
        return self.u_mid + self.u_half * x

    def rollout(self, lcs: LearnedLcs, z0, p0, H: int, lb, ub, iters: int = 25
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closed loop through the LCS: ``(U (H, n_u), Z (H+1, n_x), P (H+1, 6))``, u clipped to
        ``[lb, ub]``, ``P' = P + u[:6]`` (fixed-B deploys)."""
        z, p = np.asarray(z0, np.float64), np.asarray(p0, np.float64)
        U, Z, P = [], [z], [p]
        for _ in range(int(H)):
            u = np.clip(self.forward(z, p), lb, ub)
            z = lcs.step(z, u, iters)[0]
            p = p + u[:N_P]
            U.append(u)
            Z.append(z)
            P.append(p)
        return np.stack(U), np.stack(Z), np.stack(P)
