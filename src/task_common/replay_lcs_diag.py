"""LCS diagnostics over recorded MPPI solves (the replay ``Learned MPC`` panel's
``lcs_accuracy`` / ``contact_forces`` layers).

Accuracy: from each solve's encoded latent ``z0_i`` (``x_sol`` knot 0) the deploy's LCS is rolled
``k`` steps with the executed actions (``u_sol[:, 0]`` of solves ``i..i+k-1``; the node records
``U_exec``, the noise-perturbed plan) and compared with ``z0_{i+k}``: whitened latent distance and
the Chamfer RMS (mm) between the two decoded belts. A gap in the solve times (not one control
period apart) makes every window across it NaN. Contact forces: ``lam`` at
``(z0_i, u_sol[:, 0]_i)`` and at the plan's last action knot ``N-1`` (``x_sol``/``u_sol`` rows).
The LCS is ``TorchLcs`` (CPU, float64) with the deploy's ``lcs_forward_solver``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

from task_common.mppi_lcs import N_P, TorchLcs

if TYPE_CHECKING:
    from task_common.latent_encoder import LatentDecoder

FORWARD_SOLVERS = ("pgd_symmetric", "full_lcp")
DEFAULT_PGD_ITERS = 25  # MppiParams.pgd_iters
DEFAULT_FULL_LCP_ITERS = 200
ACTIVE_TOL = 1e-6
CHAMFER_CHUNK = 32


def chamfer_rms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(B, P, 3)``, ``(B, Q, 3)`` -> ``(B,)`` sqrt of the mean squared nearest-neighbour
    distance, averaged over both directions (same units as the points)."""
    out = np.empty(len(a))
    for s in range(0, len(a), CHAMFER_CHUNK):
        x = np.asarray(a[s:s + CHAMFER_CHUNK], dtype=np.float64)
        y = np.asarray(b[s:s + CHAMFER_CHUNK], dtype=np.float64)
        d2 = np.maximum((x * x).sum(-1)[:, :, None] + (y * y).sum(-1)[:, None]
                        - 2.0 * x @ y.transpose(0, 2, 1), 0.0)
        out[s:s + CHAMFER_CHUNK] = np.sqrt(0.5 * (d2.min(2).mean(1) + d2.min(1).mean(1)))
    return out


def mppi_pgd_iters(config: str | Path | None) -> tuple[int, str]:
    """``pgd_iters`` of the run's MPPI config, else the ``MppiParams`` default."""
    if config and Path(config).is_file():
        try:
            cfg = yaml.safe_load(Path(config).read_text()) or {}
            if "pgd_iters" in cfg:
                return int(cfg["pgd_iters"]), "MPPI config"
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            pass
    return DEFAULT_PGD_ITERS, "MppiParams default"


class DiagLcs:
    """The deploy's LCS step on CPU float64: ``TorchLcs`` (Nesterov PGD on ``G G^T + s I``,
    as MPPI) or, for ``lcs_forward_solver = full_lcp``, the exact LCP of the full ``F`` by FISTA
    on the proxy-slack box QP (``lcs_learning.lcs_model_optimized.solve_full_lcp_proxy_qp``)."""

    def __init__(self, path: Path, pgd_iters: int = DEFAULT_PGD_ITERS) -> None:
        with np.load(Path(path), allow_pickle=False) as f:
            d = {k: f[k] for k in f.files if not k.startswith("encoder")}
        self.lcs = TorchLcs(d, "cpu", torch.float64)
        self.solver = str(d["lcs_forward_solver"]) if "lcs_forward_solver" in d else \
            "pgd_symmetric"
        if self.solver not in FORWARD_SOLVERS:
            raise ValueError(f"lcs_forward_solver {self.solver!r} not in {FORWARD_SOLVERS}")
        if self.solver == "full_lcp":
            self.iters = (int(np.asarray(d["lcs_forward_iters"]).item())
                          if "lcs_forward_iters" in d else DEFAULT_FULL_LCP_ITERS)
            f = self.lcs.F
            self._gamma = 0.5 * torch.linalg.eigvalsh(f + f.T).min()
            nf = torch.linalg.matrix_norm(f, 2)
            g = self._gamma
            self._step = 1.0 / ((1.0 + nf * nf)
                                * torch.maximum(1.0 / g, 0.5 * (g + torch.sqrt(g * g + 4.0))))
        else:
            self.iters = int(pgd_iters)
        self.n_x, self.n_lam = self.lcs.n_x, self.lcs.n_lam
        self.z_std = self.lcs.z_std_np

    def _q(self, Z: torch.Tensor, U: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        q = Z @ self.lcs.E.T + U @ self.lcs.H.T + self.lcs.c
        return q + P @ self.lcs.E_p.T if self.lcs.has_p else q

    def _full_lcp(self, q: torch.Tensor) -> torch.Tensor:
        f, g, step = self.lcs.F, self._gamma, self._step
        lam = torch.zeros_like(q)
        y = lam
        t = torch.ones(q.shape[0], 1, dtype=q.dtype)
        for _ in range(self.iters):
            w = y @ f.T + q
            phi = torch.clamp(w - g * y, min=0.0)
            lam_new = torch.clamp(y - step * (phi + (w - phi) @ f / g), min=0.0)
            t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
            restart = ((y - lam_new) * (lam_new - lam)).sum(1, keepdim=True) > 0.0
            y = torch.where(restart, lam_new, lam_new + (t - 1.0) / t_new * (lam_new - lam))
            t = torch.where(restart, torch.ones_like(t), t_new)
            lam = lam_new
        return lam

    def lam(self, Z, U, P) -> np.ndarray:
        Z, U, P = (self.lcs.tensor(np.atleast_2d(x)) for x in (Z, U, P))
        with torch.no_grad():
            if self.solver == "full_lcp":
                return self._full_lcp(self._q(Z, U, P)).numpy()
            return self.lcs.solve_lambda_batched(Z, U, P, self.iters).numpy()

    def step(self, Z, U, P) -> np.ndarray:
        Z, U, P = (self.lcs.tensor(np.atleast_2d(x)) for x in (Z, U, P))
        with torch.no_grad():
            if self.solver != "full_lcp":
                return self.lcs.step(Z, U, P, self.iters)[0].numpy()
            zn = Z @ self.lcs.A.T + U @ self.lcs.B.T + self._full_lcp(self._q(Z, U, P)) \
                @ self.lcs.D.T + self.lcs.d
            return (zn + P @ self.lcs.A_p.T if self.lcs.has_p else zn).numpy()


@dataclass
class DiagResult:
    """Per-solve series (NaN where undefined), ``k`` = (1, N)."""

    time_s: np.ndarray
    horizons: tuple[int, ...]
    latent_err: dict[int, np.ndarray]
    belt_err_mm: dict[int, np.ndarray] | None
    lam0: np.ndarray  # (S, n_lam) at (z0_i, u0_i)
    lam_last: np.ndarray  # (S, n_lam) at the plan's last action knot
    period_s: float
    gaps: int
    compute_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


def consecutive(t0: np.ndarray, period: float) -> np.ndarray:
    """``(S-1,)``: solve ``i+1`` is one control period after solve ``i``."""
    return np.abs(np.diff(t0) - period) < 0.5 * period


def compute(lcs: DiagLcs, time_s: np.ndarray, x_sols: list[np.ndarray],
            u_sols: list[np.ndarray], t_knots: list[np.ndarray],
            decoder: LatentDecoder | None) -> DiagResult:
    """All series for one run; ``x_sols`` (n_x+6, N+1), ``u_sols`` (12, N), ``t_knots`` (N+1,)."""
    n_x = lcs.n_x
    if x_sols[0].shape[0] != n_x + N_P:
        raise ValueError(f"x_sol has {x_sols[0].shape[0] - N_P} latents, deploy n_x {n_x}")
    Z0 = np.stack([x[:n_x, 0] for x in x_sols])
    P0 = np.stack([x[n_x:, 0] for x in x_sols])
    U0 = np.stack([u[:, 0] for u in u_sols])
    n = u_sols[0].shape[1]
    t0 = np.array([t[0] for t in t_knots])
    period = float(np.median([t[1] - t[0] for t in t_knots]))
    link = consecutive(t0, period)
    S = len(Z0)
    lam0 = lcs.lam(Z0, U0, P0)
    lam_last = lcs.lam(np.stack([x[:n_x, n - 1] for x in x_sols]),
                       np.stack([u[:, n - 1] for u in u_sols]),
                       np.stack([x[n_x:, n - 1] for x in x_sols]))
    horizons = tuple(sorted({1, n}))
    latent: dict[int, np.ndarray] = {}
    belt: dict[int, np.ndarray] | None = None if decoder is None else {}
    broken = np.concatenate([[0], np.cumsum(~link)])  # breaks among links 0..j-1
    for k in horizons:
        lat = np.full(S, np.nan)
        idx = np.arange(max(S - k, 0))
        idx = idx[broken[idx + k] == broken[idx]]
        if len(idx):
            z, p = Z0[idx], P0[idx]
            for j in range(k):
                u = U0[idx + j]
                z = lcs.step(z, u, p)
                p = p + u[:, :N_P]
            lat[idx] = np.linalg.norm((z - Z0[idx + k]) / lcs.z_std, axis=1)
            if belt is not None:
                b = np.full(S, np.nan)
                b[idx] = 1e3 * chamfer_rms(decoder.decode_batch(z),
                                           decoder.decode_batch(Z0[idx + k]))
                belt[k] = b
        elif belt is not None:
            belt[k] = np.full(S, np.nan)
        latent[k] = lat
    return DiagResult(time_s=np.asarray(time_s, dtype=np.float64), horizons=horizons,
                      latent_err=latent, belt_err_mm=belt, lam0=lam0, lam_last=lam_last,
                      period_s=period, gaps=int((~link).sum()))


def active_set(lam: np.ndarray, tol: float = ACTIVE_TOL) -> list[int]:
    """1-based indices with ``lam > tol``."""
    return [int(i) + 1 for i in np.flatnonzero(np.asarray(lam) > tol)]


def stats(values: np.ndarray) -> tuple[float, float]:
    """``(median, p90)`` of the finite values (NaN if none)."""
    v = np.asarray(values)[np.isfinite(values)]
    return (float(np.median(v)), float(np.percentile(v, 90))) if v.size else (np.nan, np.nan)
