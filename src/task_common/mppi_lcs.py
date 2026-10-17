"""MPPI on the exported latent LCS, batched on the GPU.

Model (``TorchLcs``): ``z' = A z + B u + D lam + d`` with ``lam`` from a fixed-iteration Nesterov
PGD on ``F_pgd = G G^T + stiffness I`` (the training forward); the EE positions
``P = [p_franka, p_ur]`` are carried as ``P' = P + u[:6]`` and only enter the cost as bounds.

One solve (``mppi_plan`` -> ``MppiSolver``):
1. warm start: the previous plan shifted by one step (zeros on the first solve);
2. sample K input sequences around the mean (clipped to the input box; sample 0 = the mean);
3. roll every sample out through the LCS (one Warp kernel, ``mppi_warp``);
4. cost each rollout (``plan_cost``): latent goal distance + inputs + smoothness, plus the
   safety knots (z bounds, rotation envelope) at ``safety_penalty`` each;
5. weight the samples ``exp(-cost / temperature)``, the temperature set so the effective sample
   size is ``ess_frac`` K, and move the mean to the weighted average;
6. repeat 2-5 ``n_iter`` times; return the mean plan, or the best sample when the mean
   violates more safety knots than it.
On CUDA the whole solve is one captured CUDA graph.

Optional cost terms live in ``mppi_costs``; the overshoot guard and the BC policy prior in
``mppi_guards``.
"""

import dataclasses
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from task_common.latent_value import check_value, value_forward
from task_common.mppi_costs import (
    CLEARANCE_KEYS,
    N_P,
    CostTerms,
    _qmul_t,
    _quat_exp_t,
    check_ee_support,
    check_latent_support,
    check_support_box,
    clearance_bad,
    ee_support_cost,
    goal_set_dist,
    latent_support_cost,
    load_cost_terms,
    nn_dist_np,
    softmin_sq,
    support_hinge,
)
from task_common.mppi_guards import check_overshoot
from task_common.mppi_warp import _FusedRollout

ROT_SLICES = (slice(6, 9), slice(9, 12))
ROT_CHANNELS = tuple(range(6, 12))
EPS = 1e-12
PRIOR_SYNC = (None, "stage")
PRIOR_MODES = ("demo", "policy")


def _scalar(x):
    return np.asarray(x).item()


# --- the LCS ------------------------------------------------------------------------------------

class TorchLcs:
    """The deploy's LCS as torch tensors (``dtype`` on ``device``), stepped for K states at once."""

    def __init__(self, d: dict, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float32) -> None:
        if "bz__w1" in d:
            raise ValueError("deploy has a state-dependent B(z) head (bz__*); the MPPI node "
                             "supports fixed-B deploys only (v1)")
        self.device, self.dtype = torch.device(device), dtype
        self.n_x, self.n_u, self.n_lam = (int(_scalar(d[k])) for k in ("n_x", "n_u", "n_lam"))
        self.dt = float(_scalar(d["dt"]))
        self.N = int(_scalar(d["N"])) if "N" in d else 7
        G = np.asarray(d["G"], dtype=np.float64)
        f_pgd = G @ G.T + float(_scalar(d["stiffness"])) * np.eye(self.n_lam)
        J = np.asarray(d["J"], dtype=np.float64)
        f_np = np.asarray(d["F"], dtype=np.float64) if "F" in d else f_pgd + J - J.T
        self.step_size = 1.0 / max(float(np.linalg.eigvalsh(f_pgd).max()), 1e-6)
        self.np = {k: np.asarray(d[k], dtype=np.float64)
                   for k in ("A", "B", "D", "d", "E", "H", "c", "z_std", "u_lb", "u_ub",
                             "z_goal")}
        self.np["F_pgd"], self.np["F"] = f_pgd, f_np
        self.np["A_p"] = (np.asarray(d["A_p"], dtype=np.float64) if "A_p" in d
                          else np.zeros((self.n_x, N_P)))
        self.np["E_p"] = (np.asarray(d["E_p"], dtype=np.float64) if "E_p" in d
                          else np.zeros((self.n_lam, N_P)))
        self.has_p = "A_p" in d or "E_p" in d
        if "z_goal_stage1" in d and "z_goal_stage2" in d:
            self.stage_goals = np.stack([d["z_goal_stage1"], d["z_goal_stage2"]]).astype(
                np.float64)
        else:
            self.stage_goals = self.np["z_goal"][None, :].copy()
        self.goal_tol = (float(_scalar(d["goal_tol_whitened"])) if "goal_tol_whitened" in d
                         else -1.0)
        self._check_shapes()
        t = {k: torch.as_tensor(v, dtype=dtype, device=self.device) for k, v in self.np.items()}
        for k, v in t.items():
            setattr(self, k, v)
        # lam <- max(y - s (F y + q), 0) == max(y M - s q, 0) with M = (I - s F)^T.
        self._M = torch.as_tensor((np.eye(self.n_lam) - self.step_size * f_pgd).T, dtype=dtype,
                                  device=self.device)
        self._sE, self._sH = self.step_size * self.E, self.step_size * self.H
        self._sc, self._sEp = self.step_size * self.c, self.step_size * self.E_p
        self.z_std_np = self.np["z_std"]
        self.u_lb_np, self.u_ub_np = self.np["u_lb"], self.np["u_ub"]
        self._fused = None

    def _check_shapes(self) -> None:
        nx, nu, nl = self.n_x, self.n_u, self.n_lam
        want = {"A": (nx, nx), "B": (nx, nu), "D": (nx, nl), "d": (nx,), "E": (nl, nx),
                "F": (nl, nl), "H": (nl, nu), "c": (nl,), "z_std": (nx,), "u_lb": (nu,),
                "u_ub": (nu,), "z_goal": (nx,), "A_p": (nx, N_P), "E_p": (nl, N_P)}
        for k, shape in want.items():
            if self.np[k].shape != shape:
                raise ValueError(f"{k} shape {self.np[k].shape} != {shape}")
        if nu != 12:
            raise ValueError(f"n_u {nu} != 12 (u = [dp_franka, dp_ur, drot_franka, drot_ur])")
        if np.any(self.np["z_std"] <= 0.0):
            raise ValueError("z_std must be > 0")

    @classmethod
    def from_deploy(cls, path, device: str | torch.device = "cuda",
                    dtype: torch.dtype = torch.float32) -> "TorchLcs":
        with np.load(Path(path), allow_pickle=False) as f:
            d = {k: f[k] for k in f.files if not k.startswith("encoder")}
        return cls(d, device, dtype)

    def tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)

    def nesterov_betas(self, iters: int) -> list[float]:
        out, t = [], 1.0
        for _ in range(int(iters)):
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            out.append((t - 1.0) / t_next)
            t = t_next
        return out

    def solve_lambda_batched(self, Z: torch.Tensor, U: torch.Tensor,
                             P: torch.Tensor | None = None, iters: int = 25,
                             betas: list[float] | None = None) -> torch.Tensor:
        """``(K, n_lam)``: ``LearnedLcs.solve_lambda`` (``tol`` 0) per row."""
        sq = torch.addmm(self._sc, Z, self._sE.T)
        sq = torch.addmm(sq, U, self._sH.T)
        if P is not None and self.has_p:
            sq = torch.addmm(sq, P, self._sEp.T)
        neg_sq = -sq
        lam = torch.zeros_like(sq)
        y = lam
        for beta in (self.nesterov_betas(iters) if betas is None else betas):
            lam_old = lam
            lam = torch.addmm(neg_sq, y, self._M).clamp_(min=0.0)
            y = torch.lerp(lam_old, lam, 1.0 + beta)
        return lam

    def step(self, Z: torch.Tensor, U: torch.Tensor, P: torch.Tensor | None = None,
             iters: int = 25, betas: list[float] | None = None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Z_next (K, n_x), lam (K, n_lam))``."""
        lam = self.solve_lambda_batched(Z, U, P, iters, betas)
        zn = torch.addmm(self.d, Z, self.A.T)
        zn = torch.addmm(zn, U, self.B.T)
        zn = torch.addmm(zn, lam, self.D.T)
        if P is not None and self.has_p:
            zn = torch.addmm(zn, P, self.A_p.T)
        return zn, lam

    def rollout(self, z0, p0, U: torch.Tensor, rot0=None, iters: int = 25
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``U (K, H, n_u)`` -> ``Z (K, H+1, n_x)``, ``P (K, H+1, 6)``, ``R (K, H+1, 2)``.

        ``R`` = per-arm geodesic angle away from the episode start: ``R_k = exp(w_k) R_{k-1}``,
        ``R_0 = exp(rot0[arm])`` (``rot0`` (2, 3) world rotation vectors of the measured
        orientation relative to the start), angle ``|log R_k|``.
        """
        K, H, _ = U.shape
        z = self.tensor(z0).reshape(-1, self.n_x).expand(K, -1).contiguous()
        p = self.tensor(p0).reshape(-1, N_P).expand(K, -1).contiguous()
        betas = self.nesterov_betas(iters)
        Z, P = [z], [p]
        for k in range(H):
            u = U[:, k]
            z, _ = self.step(z, u, p, iters, betas)
            p = p + u[:, :N_P]
            Z.append(z)
            P.append(p)
        return torch.stack(Z, dim=1), torch.stack(P, dim=1), self.rotation_angles(U, rot0)

    def rotation_angles(self, U: torch.Tensor, rot0=None) -> torch.Tensor:
        """``(K, H+1, 2)`` composed geodesic angles per arm (see ``rollout``)."""
        K, H, _ = U.shape
        r0 = self.tensor(np.zeros((2, 3)) if rot0 is None else rot0).reshape(2, 3)
        out = []
        for a, sl in enumerate(ROT_SLICES):
            q = _quat_exp_t(r0[a]).expand(K, 4)
            angles = [_quat_angle_t(q)]
            for k in range(H):
                q = _qmul_t(_quat_exp_t(U[:, k, sl]), q)
                angles.append(_quat_angle_t(q))
            out.append(torch.stack(angles, dim=1))
        return torch.stack(out, dim=-1)

    def rollout_fused(self, z0, p0, U: torch.Tensor, rot0=None, iters: int = 25
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``rollout`` as one Warp kernel (``mppi_warp``): same maths, float32, CUDA only."""
        if self._fused is None:
            self._fused = _FusedRollout(self)
        return self._fused(z0, p0, U, rot0, iters)

    def whitened_dist(self, z, goal) -> float:
        """``LearnedLcs.whitened_dist`` (float64)."""
        diff = np.asarray(z, dtype=np.float64) - np.asarray(goal, dtype=np.float64)
        return float(np.linalg.norm(diff / self.z_std_np))


def _quat_angle_t(q: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.atan2(q[..., 1:].norm(dim=-1), q[..., 0].abs())


# --- parameters ---------------------------------------------------------------------------------

@dataclass
class MppiParams:
    """MPPI knobs (plan 2026-10-05 §4); ``from_yaml`` reads a flat mapping of these fields."""

    K: int = 1024
    H: int = 16
    n_iter: int = 2
    temperature: float = 1.0  # x median(J - min J); used only when ess_frac is None
    ess_frac: float | None = 0.05  # adaptive temperature: ESS = 1 / sum w^2 = ess_frac K
    sigma_frac: float = 0.5
    w_q: float = 1.0
    w_T: float = 5.0
    w_r: float = 0.3
    w_s: float = 0.3
    safety_penalty: float = 1e4
    pgd_iters: int = 25
    fused_rollout: bool = True  # CUDA: one Warp kernel per rollout (else torch ops)
    cuda_graph: bool = True  # CUDA: replay the whole solve as one captured graph
    seed: int = 0
    u_lb: list[float] | None = None  # None: the deploy's
    u_ub: list[float] | None = None
    # Symmetric outer box intersected with u_lb/u_ub: {trans_m, rot_deg} per step (None: off).
    u_box_outer: dict | None = None
    z_min_franka: float = -math.inf
    z_min_ur: float = -math.inf
    z_max_franka: float = math.inf
    z_max_ur: float = math.inf
    # Ceiling relative to the episode's first latent (the node folds it into z_max_*).
    z_ceiling_above_start_m: float = math.inf
    rot_envelope_deg: dict = field(default_factory=lambda: {"franka": 20.0, "ur": 20.0})
    action_noise_frac: float = 0.0
    u_translations_only: bool = False
    stage_goal_tols: list[float] | None = None
    stage_max_durations_s: list[float] | None = None
    # Consecutive latents inside the tol to complete a stage before the last (1 = v1 rule).
    stage_reach_streak: int = 1
    # Node: advance to the furthest later stage whose tol holds the latent (never back).
    stage_skip_ahead: bool = False
    # Node: this stage (int = node stage index; "latch" = the one after the waypoints' latch) is
    # entered only by reaching the previous stage's tol; its timeout keeps it active, and the
    # demo prior is zero from that waypoint's frame on until then. None = off.
    stage_no_timeout_into: int | str | None = None
    # Node, first latent: "progress" = start at the nearest pre-latch stage, after dropping the
    # start-state waypoints (goal demo's first latent within their tol) the latent is outside.
    stage_start: str | None = None
    # {model: clearance_model npz, ur_min_m, franka_min_m (default: the npz's), w_soft 0,
    #  soft_margin_m 0.012}; None = off.
    clearance: dict | None = None
    # Demo action prior (a sampling prior on inputs, not a target): mean init = (1 - alpha) warm
    # start + alpha U_demo; cost += prior_weight sum(((U - U_demo) / half)^2). 0 / 0 = off.
    prior_alpha: float = 0.0
    prior_weight: float = 0.0
    # Prior demo index (node): None = frames since start + offset; "stage" = in stage k from
    # max(waypoint k-1's frame, offset) at the demo rate, clamped at waypoint k's frame.
    prior_sync: str | None = None
    # Prior sequence (node): "demo" = the demo's actions at the prior index; "policy" = the BC
    # prior ``bc_prior`` (npz, train_bc_prior.py) rolled through the LCS (``PolicyPrior``).
    prior_mode: str = "demo"
    bc_prior: str | None = None
    # Seated-goal-distance overshoot guard (``OvershootGuard``, ``check_overshoot``); None = off.
    overshoot: dict | None = None
    # Data-support box on P = [p_franka, p_ur] (m, the rollout's P): {lo [6], hi [6], weight,
    # hard_margin_m (None)}; cost += weight sum_knots,coords (hinge mm)^2, knots beyond
    # hard_margin_m outside count as violating. None = off.
    support_box: dict | None = None
    # Seated goal set (goal_set.py npz, ``z_goals`` (K, n_x)) in place of the goal in the track /
    # terminal cost: per knot ``softmin_sq`` over the set of the whitened squared distance (tau
    # 0: the min). The node uses it in the last stage only (cost and "reached"). None = off.
    goal_set: str | None = None
    goal_set_tau: float = 0.0
    # Learned cost-to-go (train_value.py npz): {path, weight, where: terminal | all_knots};
    # cost += weight V(z_H) or weight sum_k V(z_k), k = 1..H. None = off.
    value: dict | None = None
    # Latent data support: {file (npz, z_t (N, n_x) training latents), radius 1.5, weight 1,
    # max_points 3000 (stride subsample), where all | terminal}; cost += weight sum_k
    # relu(d_k - radius)^2, d_k = whitened distance of z_k to the nearest training latent,
    # k = 1..H (all) or H (terminal). None = off.
    latent_support: dict | None = None
    # EE data support: {file (npz, ee_pos (N, 6) [Franka xyz, UR xyz] m in the pulley frame
    # R^T (x - t)), pulley_pose7 (xyz + wxyz of that frame), radius_mm 35, weight 1, max_points
    # 3000, where all | terminal}; cost += weight sum_k relu((d_k - radius_mm) / 10 mm)^2,
    # d_k = 6-D Euclidean distance (mm) of P_k to the nearest training EE position. None = off.
    ee_support: dict | None = None

    def __post_init__(self) -> None:
        if self.K < 2 or self.H < 1 or self.n_iter < 1 or self.pgd_iters < 1:
            raise ValueError(f"K {self.K}, H {self.H}, n_iter {self.n_iter}, pgd_iters "
                             f"{self.pgd_iters} out of range")
        if self.temperature <= 0.0 or self.sigma_frac < 0.0 or self.action_noise_frac < 0.0:
            raise ValueError("temperature must be > 0, sigma_frac/action_noise_frac >= 0")
        if int(self.stage_reach_streak) != self.stage_reach_streak or self.stage_reach_streak < 1:
            raise ValueError(f"stage_reach_streak {self.stage_reach_streak} must be an int >= 1")
        if self.stage_skip_ahead is not True and self.stage_skip_ahead is not False:
            raise ValueError(f"stage_skip_ahead {self.stage_skip_ahead!r} must be a bool")
        nt = self.stage_no_timeout_into
        if nt is not None and nt != "latch" and (isinstance(nt, bool) or not isinstance(nt, int)
                                                 or nt < 1):
            raise ValueError(f"stage_no_timeout_into {nt!r}: an int >= 1, 'latch' or None")
        if self.stage_start not in (None, "progress"):
            raise ValueError(f"stage_start {self.stage_start!r}: 'progress' or None")
        if self.ess_frac is not None and not 0.0 < self.ess_frac <= 1.0:
            raise ValueError(f"ess_frac {self.ess_frac} outside (0, 1]")
        if set(self.rot_envelope_deg) != {"franka", "ur"}:
            raise ValueError(f"rot_envelope_deg keys {set(self.rot_envelope_deg)} != franka, ur")
        if self.clearance is not None and ("model" not in self.clearance
                                           or not set(self.clearance) <= CLEARANCE_KEYS):
            raise ValueError(f"clearance keys {sorted(self.clearance)}: need model, within "
                             f"{sorted(CLEARANCE_KEYS)}")
        if not 0.0 <= self.prior_alpha <= 1.0 or self.prior_weight < 0.0:
            raise ValueError(f"prior_alpha {self.prior_alpha} outside [0, 1] or prior_weight "
                             f"{self.prior_weight} < 0")
        if self.prior_sync not in PRIOR_SYNC:
            raise ValueError(f"prior_sync {self.prior_sync!r}: one of {PRIOR_SYNC}")
        if self.prior_mode not in PRIOR_MODES:
            raise ValueError(f"prior_mode {self.prior_mode!r}: one of {PRIOR_MODES}")
        if (self.prior_mode == "policy") != (self.bc_prior is not None):
            raise ValueError("prior_mode policy needs bc_prior (a bc_prior.npz path), and "
                             "bc_prior needs prior_mode policy")
        if self.prior_mode == "policy" and (self.prior_sync is not None
                                            or self.stage_no_timeout_into is not None):
            raise ValueError("prior_mode policy: prior_sync / stage_no_timeout_into act on the "
                             "demo's frame index, not on a policy prior; turn them off")
        if self.overshoot is not None:
            check_overshoot(self.overshoot)
        if self.support_box is not None:
            check_support_box(self.support_box)
        if not (math.isfinite(float(self.goal_set_tau)) and float(self.goal_set_tau) >= 0.0) \
                or (self.goal_set is not None and not isinstance(self.goal_set, (str, Path))):
            raise ValueError(f"goal_set {self.goal_set!r} (npz path or None) / goal_set_tau "
                             f"{self.goal_set_tau} (finite >= 0)")
        if self.value is not None:
            check_value(self.value)
        if self.latent_support is not None:
            check_latent_support(self.latent_support)
        if self.ee_support is not None:
            check_ee_support(self.ee_support)

    @classmethod
    def from_yaml(cls, path) -> "MppiParams":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown MPPI keys {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def bounds(self, lcs: TorchLcs) -> tuple[np.ndarray, np.ndarray]:
        """Effective ``(u_lb, u_ub)`` float64: override or deploy, outer box, rotations off."""
        lb = np.asarray(lcs.u_lb_np if self.u_lb is None else self.u_lb, dtype=np.float64).copy()
        ub = np.asarray(lcs.u_ub_np if self.u_ub is None else self.u_ub, dtype=np.float64).copy()
        if lb.shape != (lcs.n_u,) or ub.shape != (lcs.n_u,) or np.any(lb > ub):
            raise ValueError(f"bounds {lb} / {ub} invalid for n_u {lcs.n_u}")
        if self.u_box_outer is not None:
            half = np.r_[np.full(6, float(self.u_box_outer["trans_m"])),
                         np.full(6, math.radians(float(self.u_box_outer["rot_deg"])))]
            lb, ub = np.maximum(lb, -half), np.minimum(ub, half)
            if np.any(lb > ub):
                raise ValueError("u_box_outer does not intersect the input bounds")
        if self.u_translations_only:
            lb[list(ROT_CHANNELS)] = ub[list(ROT_CHANNELS)] = 0.0
        return lb, ub

    def rot_envelope_rad(self) -> np.ndarray:
        return np.radians([float(self.rot_envelope_deg["franka"]),
                           float(self.rot_envelope_deg["ur"])])


@dataclass
class _Box:
    lb: torch.Tensor
    ub: torch.Tensor
    half: torch.Tensor  # cost scale (zero-width channels -> 1)
    sigma: torch.Tensor


def _box(lcs: TorchLcs, params: MppiParams) -> _Box:
    lb, ub = params.bounds(lcs)
    width = (ub - lb) / 2.0
    half = np.where(width > 0.0, width, 1.0)
    return _Box(lcs.tensor(lb), lcs.tensor(ub), lcs.tensor(half),
                lcs.tensor(params.sigma_frac * width))


def safety_limits(params: MppiParams) -> np.ndarray:
    """``[z_min_f, z_max_f, z_min_ur, z_max_ur, env_f, env_ur]`` (m, rad)."""
    return np.r_[params.z_min_franka, params.z_max_franka, params.z_min_ur, params.z_max_ur,
                 params.rot_envelope_rad()]


# --- cost ---------------------------------------------------------------------------------------
# Each helper adds its terms to ``cost`` in a fixed order (the solve must stay bit-identical).

def plan_cost(lcs: TorchLcs, params: MppiParams, Z: torch.Tensor, P: torch.Tensor,
              R: torch.Tensor, U: torch.Tensor, z_goal: torch.Tensor, half: torch.Tensor,
              u_anchor: torch.Tensor | None, lim: torch.Tensor | None = None,
              anchor_on: torch.Tensor | None = None, C: torch.Tensor | None = None,
              u_demo: torch.Tensor | None = None, prior_on: torch.Tensor | None = None,
              terms: CostTerms | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(cost (K,), violating knots (K,), envelope violated (K,) bool)`` of rolled-out plans
    ``Z / P / R`` of inputs ``U`` (knots 1..H).

    ``z_goal`` ``(n_x,)`` or one row per knot ``(H, n_x)``. ``lim``: ``safety_limits`` as a tensor
    (default: from ``params``). ``anchor_on`` / ``prior_on`` (0/1 tensors) gate the anchor and
    prior terms without a branch (CUDA graphs). ``C``: the rollout's ``Clearance`` distances.
    ``terms``: the loaded optional terms (default: ``load_cost_terms``).
    """
    if lim is None:
        lim = lcs.tensor(safety_limits(params))
    if terms is None:
        terms = load_cost_terms(lcs, params, with_clearance=C is not None)
    cost = _goal_cost(lcs, params, Z, z_goal, terms)
    cost = _add_learned_costs(cost, lcs, params, Z, P, terms)
    cost = _add_input_costs(cost, params, U, half, u_anchor, anchor_on, u_demo, prior_on)
    env = (R[:, 1:, 0] > lim[4]) | (R[:, 1:, 1] > lim[5])  # rotation envelope
    pz_f, pz_u = P[:, 1:, 2], P[:, 1:, 5]
    bad = (pz_f < lim[0]) | (pz_f > lim[1]) | (pz_u < lim[2]) | (pz_u > lim[3]) | env
    if C is not None:
        clr = terms.clear
        bad = bad | clearance_bad(C, clr.lim)
        if clr.w_soft > 0.0:
            cost = cost + clr.w_soft * (clr.soft_margin - C[:, 1:]).clamp_min(0.0).square().sum(
                (1, 2))
    sb = params.support_box
    if sb is not None:
        hinge = support_hinge(P[:, 1:], terms.sbox)
        cost = cost + float(sb["weight"]) * (hinge * 1e3).square().sum((1, 2))
        if sb.get("hard_margin_m") is not None:
            bad = bad | (hinge > float(sb["hard_margin_m"])).any(-1)
    n_bad = bad.sum(1).to(cost.dtype)
    return cost + params.safety_penalty * n_bad, n_bad, env.any(1)


def _goal_cost(lcs, params, Z, z_goal, terms) -> torch.Tensor:
    """``w_q`` x summed + ``w_T`` x terminal whitened squared goal distance (or goal-set soft-min)."""
    w = 1.0 / lcs.z_std
    if params.goal_set is None:
        dz = (Z[:, 1:] - z_goal) * w
        track = (dz * dz).sum(-1)
    else:
        dz = (Z[:, 1:, None] - terms.gset) * w
        track = softmin_sq((dz * dz).sum(-1), float(params.goal_set_tau))
    return params.w_q * track.sum(1) + params.w_T * track[:, -1]


def _add_learned_costs(cost, lcs, params, Z, P, terms) -> torch.Tensor:
    """+ learned value, latent support, EE support (each when on)."""
    vv = params.value
    if vv is not None:
        if vv.get("where", "terminal") == "terminal":
            cost = cost + float(vv["weight"]) * value_forward(terms.value, Z[:, -1])
        else:
            cost = cost + float(vv["weight"]) * value_forward(terms.value, Z[:, 1:]).sum(1)
    if params.latent_support is not None:
        cost = cost + latent_support_cost(lcs, params.latent_support, Z, terms.lsup)
    if params.ee_support is not None:
        cost = cost + ee_support_cost(params.ee_support, P, terms.esup)
    return cost


def _add_input_costs(cost, params, U, half, u_anchor, anchor_on, u_demo, prior_on
                     ) -> torch.Tensor:
    """+ input size, smoothness (incl. vs the last sent input), demo prior; scaled by ``half``."""
    un = U / half
    cost = cost + params.w_r * (un * un).sum((1, 2))
    if params.w_s > 0.0:
        du = torch.diff(U, dim=1) / half
        cost = cost + params.w_s * (du * du).sum((1, 2))
        if u_anchor is not None:
            d0 = (U[:, 0] - u_anchor) / half
            term = params.w_s * (d0 * d0).sum(-1)
            cost = cost + (term if anchor_on is None else anchor_on * term)
    if params.prior_weight > 0.0 and u_demo is not None:
        dd = (U - u_demo) / half
        term = params.prior_weight * (dd * dd).sum((1, 2))
        cost = cost + (term if prior_on is None else prior_on * term)
    return cost


# --- the solver ---------------------------------------------------------------------------------

def warm_start(u_prev_plan, H: int, n_u: int) -> np.ndarray:
    """Previous plan shifted by one, last input repeated (zeros without one)."""
    if u_prev_plan is None:
        return np.zeros((H, n_u))
    prev = np.asarray(u_prev_plan, dtype=np.float64).reshape(-1, n_u)
    shifted = np.concatenate([prev[1:], prev[-1:]], axis=0)
    if len(shifted) >= H:
        return shifted[:H].copy()
    return np.concatenate([shifted, np.repeat(shifted[-1:], H - len(shifted), axis=0)])


def _pick(x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    """``x[i]`` for a 0-d device index without a host sync (graph-safe)."""
    return x.index_select(0, i.reshape(1))[0]


ESS_GRID = np.logspace(-5.0, 3.0, 161)  # temperature candidates x median(J - min J)
# The solve's scalar stats, packed after the plan / Z / P (``plan_violations`` int, ``fallback``
# bool on the host).
STAT_NAMES = ("cost_min", "cost_mean", "plan_cost", "plan_violations", "weight_entropy",
              "frac_penalised", "fallback", "ess", "temperature_abs", "frac_env_penalised")
N_STATS = len(STAT_NAMES)
# With clearance: C (H+1, 2) of the plan, then these (m on the device, mm in the info).
CLEAR_STAT_NAMES = ("frac_clear_penalised", "clear_min_ur_mm", "clear_min_f_mm",
                    "clear_violating", "clear_now_ur_mm", "clear_now_f_mm")
N_CLEAR_STATS = len(CLEAR_STAT_NAMES)


class MppiSolver:
    """One MPPI solve = ``n_iter`` passes + the mean plan's evaluation, on static ``s_*`` buffers.

    On CUDA the whole solve is one captured CUDA graph (fixed shapes; noise drawn into a static
    buffer before the replay); limits/anchor are buffer inputs so one graph serves every call.
    Its outputs are packed into ``self.out`` (one device -> host copy, see ``_unpack``).
    """

    def __init__(self, lcs: TorchLcs, params: MppiParams, use_graph: bool | None = None
                 ) -> None:
        self.lcs, self.p = lcs, params
        self.box = _box(lcs, params)
        K, H, nu, nx = params.K, params.H, lcs.n_u, lcs.n_x
        t = lcs.tensor
        self.s_z0, self.s_p0, self.s_rot0 = t(np.zeros(nx)), t(np.zeros(N_P)), t(np.zeros((2, 3)))
        self.s_goal, self.s_mean0 = t(np.zeros((H, nx))), t(np.zeros((H, nu)))
        self.s_udemo, self.s_prior_on = t(np.zeros((H, nu))), t(np.zeros(()))
        self.s_anchor, self.s_anchor_on = t(np.zeros(nu)), t(np.zeros(()))
        self.s_lim = t(safety_limits(params))
        self.s_eps = t(np.zeros((params.n_iter, K, H, nu)))
        self.betas = lcs.nesterov_betas(params.pgd_iters)
        self.grid = t(ESS_GRID)
        cuda = lcs.device.type == "cuda"
        self.use_graph = cuda and params.cuda_graph if use_graph is None else use_graph
        self.roll = lcs.rollout_fused if cuda and params.fused_rollout else lcs.rollout
        self.terms = load_cost_terms(lcs, params)
        self.clear = self.terms.clear
        self.clear_fused = cuda and params.fused_rollout
        if self.clear is not None:
            self.s_q0 = t(np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)))
        self.graph = None
        if self.use_graph:
            self._capture()

    def _weights(self, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(w (K,), ESS, temperature)``: softmax of ``-cost / temperature``."""
        c = cost - cost.min()
        scale = c.median() + 1e-9
        if self.p.ess_frac is None:
            temp = self.p.temperature * scale
            w = torch.exp(-c / temp)
            w = w / w.sum()
        else:
            # Smallest temperature on the grid whose ESS = 1 / sum w^2 reaches ess_frac K.
            temps = self.grid * scale
            W = torch.exp(-c[None] / temps[:, None])
            W = W / W.sum(1, keepdim=True)
            ess = 1.0 / (W * W).sum(1)
            idx = torch.argmax((ess >= self.p.ess_frac * self.p.K).to(torch.int32))
            w, temp = _pick(W, idx), _pick(temps, idx)
        return w, 1.0 / (w * w).sum(), temp

    def _cost(self, Z, P, R, U, C):
        b = self.box
        return plan_cost(self.lcs, self.p, Z, P, R, U, self.s_goal, b.half, self.s_anchor,
                         lim=self.s_lim, anchor_on=self.s_anchor_on, C=C, u_demo=self.s_udemo,
                         prior_on=self.s_prior_on, terms=self.terms)

    def _body(self) -> None:
        """The captured solve: sample, roll out, cost, reweight x ``n_iter``; then the mean plan."""
        p, box = self.p, self.box
        mean = torch.maximum(torch.minimum(self.s_mean0, box.ub), box.lb)
        for i in range(p.n_iter):
            U = torch.maximum(torch.minimum(mean + box.sigma * self.s_eps[i], box.ub), box.lb)
            U[0] = mean
            Z, P, R = self.roll(self.s_z0, self.s_p0, U, self.s_rot0, p.pgd_iters)
            C = self._clearance(P, U)
            cost, n_bad, env = self._cost(Z, P, R, U, C)
            w, ess, temp = self._weights(cost)
            mean = (w[:, None, None] * U).sum(0)
        Zm, Pm, Rm = self.roll(self.s_z0, self.s_p0, mean[None], self.s_rot0, p.pgd_iters)
        Cm = self._clearance(Pm, mean[None])
        cost_m, bad_m, _ = self._cost(Zm, Pm, Rm, mean[None], Cm)
        # fall back to the last pass's best sample when it violates fewer knots than the mean
        best = torch.argmin(cost)
        fb = (bad_m[0] > 0) & (_pick(n_bad, best) < bad_m[0])
        plan = torch.where(fb, _pick(U, best), mean)
        Zp = torch.where(fb, _pick(Z, best), Zm[0])
        Pp = torch.where(fb, _pick(P, best), Pm[0])
        stats = torch.stack([  # STAT_NAMES
            cost.min(), cost.mean(), torch.where(fb, _pick(cost, best), cost_m[0]),
            torch.where(fb, _pick(n_bad, best), bad_m[0]),
            -(w * torch.log(w.clamp_min(EPS))).sum(), (n_bad > 0).to(cost.dtype).mean(),
            fb.to(cost.dtype), ess, temp, env.to(cost.dtype).mean()])
        parts = [plan.reshape(-1), Zp.reshape(-1), Pp.reshape(-1), stats]
        if self.clear is not None:
            Cp = torch.where(fb, _pick(C, best), Cm[0])
            parts += [Cp.reshape(-1), torch.stack([  # CLEAR_STAT_NAMES
                clearance_bad(C, self.clear.lim).any(1).to(cost.dtype).mean(),
                Cp[1:, 0].min(), Cp[1:, 1].min(),
                clearance_bad(Cp[None], self.clear.lim).sum().to(cost.dtype),
                Cm[0, 0, 0], Cm[0, 0, 1]])]
        self.out = torch.cat(parts)

    def _clearance(self, P: torch.Tensor, U: torch.Tensor) -> torch.Tensor | None:
        if self.clear is None:
            return None
        return self.clear(P, U, self.s_q0, self.clear_fused)

    def _capture(self) -> None:
        s = torch.cuda.Stream(device=self.lcs.device)
        s.wait_stream(torch.cuda.current_stream(self.lcs.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream(self.lcs.device).wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

    def _unpack(self, out: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
        """``self.out`` on the host -> ``(plan (H, n_u), Z (H+1, n_x), P (H+1, 6), stats,
        clearance stats)``."""
        H, nu, nx = self.p.H, self.lcs.n_u, self.lcs.n_x
        sizes = [H * nu, (H + 1) * nx, (H + 1) * N_P, N_STATS]
        if self.clear is not None:
            sizes += [(H + 1) * 2, N_CLEAR_STATS]
        parts = np.split(out, np.cumsum(sizes)[:-1])
        st = parts[3]
        stats = dict(zip(STAT_NAMES, st))
        stats["plan_violations"], stats["fallback"] = int(st[3]), bool(st[6])
        if self.clear is not None:
            cs = parts[5]
            clear = {"frac_clear_penalised": cs[0], "clear_min_ur_mm": cs[1] * 1e3,
                     "clear_min_f_mm": cs[2] * 1e3, "clear_violating": int(cs[3]),
                     "clear_now_ur_mm": cs[4] * 1e3, "clear_now_f_mm": cs[5] * 1e3,
                     "C": parts[4].reshape(H + 1, 2)}
        else:
            clear = {}
        return (parts[0].reshape(H, nu), parts[1].reshape(H + 1, nx),
                parts[2].reshape(H + 1, N_P), stats, clear)

    def solve(self, z0, p0, rot0, z_goal, u_prev_plan, gen: torch.Generator,
              limits: np.ndarray | None = None, q_now=None, u_demo=None
              ) -> tuple[np.ndarray, dict]:
        """``q_now`` (2, 4) ``wxyz``: the current absolute Franka / UR orientations (needed
        with ``clearance``). ``z_goal`` ``(n_x,)`` (every knot) or ``(H, n_x)`` (knots 1..H).
        ``u_demo`` ``(H, n_u)``: demo actions for the prior (None: prior off this solve)."""
        t0 = time.perf_counter()
        lcs, p = self.lcs, self.p
        H, nu, nx = p.H, lcs.n_u, lcs.n_x
        # inputs -> the graph's static buffers
        self.s_z0.copy_(lcs.tensor(z0))
        self.s_p0.copy_(lcs.tensor(p0))
        self.s_rot0.copy_(lcs.tensor(np.zeros((2, 3)) if rot0 is None else rot0).reshape(2, 3))
        goal = goal_rows(z_goal, H, nx)
        self.s_goal.copy_(lcs.tensor(goal))
        mean0 = warm_start(u_prev_plan, H, nu)
        self.s_prior_on.fill_(0.0 if u_demo is None else 1.0)
        if u_demo is not None:
            ud = np.asarray(u_demo, dtype=np.float64).reshape(H, nu)
            if p.prior_alpha > 0.0:
                mean0 = (1.0 - p.prior_alpha) * mean0 + p.prior_alpha * ud
            self.s_udemo.copy_(lcs.tensor(ud))
        self.s_mean0.copy_(lcs.tensor(mean0))
        self.s_anchor_on.fill_(0.0 if u_prev_plan is None else 1.0)
        if u_prev_plan is not None:
            self.s_anchor.copy_(lcs.tensor(np.asarray(u_prev_plan)[0]))
        self.s_lim.copy_(lcs.tensor(safety_limits(p) if limits is None else limits))
        if self.clear is not None:
            if q_now is None:
                raise ValueError("clearance needs q_now (the current wxyz orientations)")
            q = np.asarray(q_now, dtype=np.float64).reshape(2, 4)
            self.s_q0.copy_(lcs.tensor(q / np.linalg.norm(q, axis=1, keepdims=True)))
        self.s_eps.normal_(generator=gen)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._body()
        plan, Zp, Pp, stats, clear = self._unpack(self.out.double().cpu().numpy())
        T = self.terms
        info = {**stats,
                "goal_dist_H": (lcs.whitened_dist(Zp[-1], goal[-1]) if T.gset_np is None else
                                goal_set_dist(Zp[-1], T.gset_np, lcs.z_std_np, p.goal_set_tau)),
                "Z": Zp, "P": Pp, **clear, "solve_ms": (time.perf_counter() - t0) * 1e3}
        if T.vfn is not None:
            info["value_H"] = float(T.vfn(Zp[-1]))
        if T.lsup_np is not None:
            d = nn_dist_np(Zp / lcs.z_std_np, T.lsup_np)
            info["support_d0"], info["support_dmax"] = float(d[0]), float(d[1:].max())
        if T.esup_np is not None:
            d = nn_dist_np(Pp * 1e3, T.esup_np)
            info["ee_support_d0"], info["ee_support_dmax"] = float(d[0]), float(d[1:].max())
        return plan, info


def goal_rows(z_goal, H: int, n_x: int) -> np.ndarray:
    """``(H, n_x)`` float64: a single goal repeated per knot, or a per-knot sequence as is."""
    g = np.asarray(z_goal, dtype=np.float64)
    if g.shape == (n_x,):
        return np.ascontiguousarray(np.broadcast_to(g, (H, n_x)))
    if g.shape != (H, n_x):
        raise ValueError(f"z_goal shape {g.shape}: expected ({n_x},) or ({H}, {n_x})")
    return g.copy()


# --- entry points -------------------------------------------------------------------------------

_SOLVERS: dict = {}


def solver_for(lcs: TorchLcs, params: MppiParams) -> MppiSolver:
    """Cached solver per (lcs, every field but the safety limits/seed/stage lists)."""
    skip = {"z_min_franka", "z_min_ur", "z_max_franka", "z_max_ur", "z_ceiling_above_start_m",
            "rot_envelope_deg", "seed", "action_noise_frac", "stage_goal_tols",
            "stage_max_durations_s", "prior_sync", "stage_skip_ahead", "stage_no_timeout_into",
            "prior_mode", "bc_prior", "stage_start"}
    key = (id(lcs), repr({k: v for k, v in params.to_dict().items() if k not in skip}))
    if key not in _SOLVERS:
        _SOLVERS[key] = MppiSolver(lcs, params)
    return _SOLVERS[key]


def mppi_plan(lcs: TorchLcs, params: MppiParams, z0, p0, rot0, z_goal, u_prev_plan,
              gen: torch.Generator, q_now=None, u_demo=None) -> tuple[np.ndarray, dict]:
    """``(U (H, n_u) float64, info)``; ``u_prev_plan`` warm-starts and anchors smoothness.

    ``q_now`` (2, 4) ``wxyz``: the current orientations, needed with ``params.clearance``.
    ``z_goal`` ``(n_x,)`` or ``(H, n_x)``; ``u_demo`` ``(H, n_u)``: the demo action prior's inputs.
    """
    return solver_for(lcs, params).solve(z0, p0, rot0, z_goal, u_prev_plan, gen,
                                         safety_limits(params), q_now, u_demo)


def rollout_cost(lcs: TorchLcs, params: MppiParams, z0, p0, rot0, z_goal, U,
                 u_anchor=None, q_now=None, u_demo=None) -> tuple[float, int, np.ndarray]:
    """``(cost, violating knots, Z (H+1, n_x))`` of one plan ``U (H, n_u)`` under ``params``."""
    box = _box(lcs, params)
    Ut = lcs.tensor(np.asarray(U, dtype=np.float64))[None]
    Z, P, R = lcs.rollout(z0, p0, Ut, rot0, params.pgd_iters)
    anchor = None if u_anchor is None else lcs.tensor(u_anchor)
    terms = load_cost_terms(lcs, params)
    if terms.clear is not None and q_now is None:
        raise ValueError("clearance needs q_now (the current wxyz orientations)")
    C = None if terms.clear is None else terms.clear.torch_path(P, Ut, lcs.tensor(q_now))
    ud = None if u_demo is None else lcs.tensor(np.asarray(u_demo, dtype=np.float64))
    cost, n_bad, _ = plan_cost(lcs, params, Z, P, R, Ut, lcs.tensor(z_goal), box.half, anchor,
                               C=C, u_demo=ud, terms=terms)
    return float(cost[0]), int(n_bad[0]), Z[0].double().cpu().numpy()


# --- poses (wxyz, numpy) ------------------------------------------------------------------------

def qmul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_exp(rotvec) -> np.ndarray:
    """Unit ``wxyz`` of a rotation vector."""
    r = np.asarray(rotvec, dtype=np.float64)
    a = float(np.linalg.norm(r))
    if a < 1e-12:
        q = np.r_[1.0, 0.5 * r]
        return q / np.linalg.norm(q)
    return np.r_[math.cos(0.5 * a), math.sin(0.5 * a) / a * r]


def quat_angle(q0, q1) -> float:
    """Angle of ``q1 q0^-1`` in ``[0, pi]``."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    d = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    return 2.0 * math.acos(min(1.0, d))


def franka_knots(pose7, U, n_knots: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative knots from the latent pose: ``p_i = p_{i-1} + u[0:3]``,
    ``q_i = exp(u[6:9]) q_{i-1}``; ``(pos (n, 3), quat (n, 4))``."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if len(U) < n_knots - 1:
        raise ValueError(f"{len(U)} inputs < {n_knots - 1} needed for {n_knots} knots")
    pos, quat = [pose7[:3].copy()], [pose7[3:7].copy()]
    for i in range(1, n_knots):
        pos.append(pos[-1] + U[i - 1, 0:3])
        q = qmul(quat_exp(U[i - 1, 6:9]), quat[-1])
        quat.append(q / np.linalg.norm(q))
    return np.stack(pos), np.stack(quat)
