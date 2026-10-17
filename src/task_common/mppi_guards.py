"""Node-side MPPI helpers that run around the solve, not inside it.

- ``OvershootGuard`` (``MppiParams.overshoot``): fades the demo prior / changes the weights /
  holds near the seated goal, from the latent distance to it.
- ``PolicyPrior`` (``prior_mode: policy``): a BC policy rolled through the LCS as the prior.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch

from task_common.mppi_costs import N_P

OVERSHOOT_KEYS = frozenset({"seat_tol", "fade", "fade_last_stage", "hold", "near"})
HOLD_KEYS = frozenset({"enter_k", "streak", "goal", "w_q", "w_T", "w_r", "w_s", "exit_k", "cold",
                       "stall"})
STALL_KEYS = frozenset({"latents", "eps", "k"})
HOLD_GOALS = ("seat", "entry", "best")
NEAR_KEYS = frozenset({"k", "w_T"})
W_KEYS = ("w_q", "w_T", "w_r", "w_s")


def check_overshoot(ov: dict) -> None:
    """``MppiParams.overshoot`` keys and ranges."""
    if not set(ov) <= OVERSHOOT_KEYS or "seat_tol" not in ov:
        raise ValueError(f"overshoot keys {sorted(ov)}: need seat_tol, within "
                         f"{sorted(OVERSHOOT_KEYS)}")
    if not float(ov["seat_tol"]) > 0.0:
        raise ValueError(f"overshoot.seat_tol {ov['seat_tol']} must be > 0")
    f = ov.get("fade")
    if f is not None and (len(f) != 2 or not 0.0 <= float(f[0]) < float(f[1])):
        raise ValueError(f"overshoot.fade {f}: [k1, k2] with 0 <= k1 < k2")
    h = ov.get("hold")
    if h is not None:
        if not set(h) <= HOLD_KEYS:
            raise ValueError(f"overshoot.hold keys {sorted(h)} not within {sorted(HOLD_KEYS)}")
        if h.get("goal", "seat") not in HOLD_GOALS:
            raise ValueError(f"overshoot.hold.goal {h.get('goal')!r}: one of {HOLD_GOALS}")
        st = h.get("streak", 1)
        if int(st) != st or st < 1 or not float(h.get("enter_k", 1.0)) > 0.0:
            raise ValueError(f"overshoot.hold streak {st} (int >= 1) / enter_k > 0")
        ex = h.get("exit_k")
        if ex is not None and not float(ex) > float(h.get("enter_k", 1.0)):
            raise ValueError(f"overshoot.hold.exit_k {ex} must exceed enter_k")
        if any(not float(h.get(k, 1.0)) > 0.0 for k in W_KEYS):
            raise ValueError("overshoot.hold weight scales must be > 0")
        sl = h.get("stall")
        if sl is not None and (not set(sl) <= STALL_KEYS or int(sl.get("latents", 0)) < 1
                               or float(sl.get("eps", 0.0)) < 0.0
                               or not float(sl.get("k", 1.0)) > 0.0):
            raise ValueError(f"overshoot.hold.stall {sl}: {{latents >= 1, eps >= 0, k > 0}}")
    n = ov.get("near")
    if n is not None and (not set(n) <= NEAR_KEYS or not float(n.get("k", 0.0)) > 0.0
                          or not float(n.get("w_T", 1.0)) > 0.0):
        raise ValueError(f"overshoot.near {n}: {{k > 0, w_T scale > 0}}")


def scaled_weights(params, scales: dict):
    """``params`` with ``w_q / w_T / w_r / w_s`` multiplied by ``scales`` (missing = 1)."""
    return dataclasses.replace(params, **{k: getattr(params, k) * float(scales.get(k, 1.0))
                                          for k in W_KEYS})


@dataclass
class GuardState:
    mode: str  # normal | near | hold
    prior_scale: float  # demo prior actions x this (0 = a zero-action prior)
    goal: np.ndarray | None  # hold: the goal replacing the stage goal (None: keep it)
    cold: bool  # hold entry: solve without the warm start / anchor
    entered: bool  # hold entered on this latent


class OvershootGuard:
    """Seated-goal-distance logic of ``MppiParams.overshoot`` (no EE terms; d = whitened latent
    distance to the seated goal). ``update`` once per solve -> ``GuardState``; ``params_for``
    gives the solve's weights. Hold enters after ``streak`` latents under ``enter_k x tol`` or,
    with ``stall``, in the last stage with min d under ``k x tol`` once d has not dropped ``eps``
    below its last anchor for ``latents`` solves; it latches (``exit_k`` None) or leaves above
    ``exit_k x tol``. Goal ``best`` = the episode's minimum-d latent so far.
    """

    def __init__(self, ov: dict) -> None:
        check_overshoot(ov)
        self.ov, self.tol = ov, float(ov["seat_tol"])
        self.hold_cfg = ov.get("hold")
        self.streak, self.hold, self.since_best = 0, False, 0
        self.z_entry = self.z_best = None
        self.d_best = self.d_anchor = math.inf

    def fade(self, d: float, last_stage: bool) -> float:
        if last_stage and self.ov.get("fade_last_stage"):
            return 0.0
        f = self.ov.get("fade")
        if f is None:
            return 1.0
        lo, hi = float(f[0]) * self.tol, float(f[1]) * self.tol
        return float(min(max((d - lo) / (hi - lo), 0.0), 1.0))

    def update(self, z, d: float, last_stage: bool) -> GuardState:
        h, entered = self.hold_cfg, False
        sl = (h or {}).get("stall")
        if d < self.d_anchor - (float(sl.get("eps", 0.0)) if sl else 0.0):
            self.since_best, self.d_anchor = 0, d
        else:
            self.since_best += 1
        if d < self.d_best:
            self.z_best, self.d_best = np.array(z), d
        if h is not None:
            if self.hold and h.get("exit_k") is not None and d > float(h["exit_k"]) * self.tol:
                self.hold, self.streak = False, 0
            if not self.hold:
                self.streak = self.streak + 1 if d < float(h.get("enter_k", 1.0)) * self.tol \
                    else 0
                stalled = (sl is not None and last_stage
                           and self.since_best >= int(sl["latents"])
                           and self.d_best < float(sl.get("k", 1.0)) * self.tol)
                if self.streak >= int(h.get("streak", 1)) or stalled:
                    self.hold, entered = True, True
                    self.z_entry = np.array(z)
        if self.hold:
            g = h.get("goal", "seat")
            goal = None if g == "seat" else self.z_entry if g == "entry" else self.z_best
            return GuardState("hold", 0.0, goal, entered and bool(h.get("cold", False)), entered)
        n = self.ov.get("near")
        mode = "near" if n is not None and d < float(n["k"]) * self.tol else "normal"
        return GuardState(mode, self.fade(d, last_stage), None, False, False)

    def params_for(self, params, mode: str):
        if mode == "hold":
            return scaled_weights(params, self.hold_cfg)
        if mode == "near":
            return scaled_weights(params, {"w_T": self.ov["near"].get("w_T", 1.0)})
        return params

    def variants(self, params) -> list:
        """The non-normal weight sets this guard can use (node warm-up)."""
        out = [self.params_for(params, "hold")] if self.hold_cfg is not None else []
        return out + ([self.params_for(params, "near")] if self.ov.get("near") else [])


class PolicyPrior:
    """``prior_mode: policy``: the BC prior ``pi`` (``bc_prior.BcPrior`` weights) rolled through
    the LCS from the current ``(z, P)``: ``u_k = clip(pi(z_k[, P_k]))`` to ``params.bounds`` (P
    an input only for a "zp" prior), ``z_{k+1}`` = LCS step, ``P_{k+1} = P_k + u_k[:6]``, k < H.
    Runs before the solve, as its own CUDA graph on CUDA (the solver's graph is unchanged)."""

    def __init__(self, path, lcs, params, deploy=None, use_graph: bool | None = None) -> None:
        from task_common.bc_prior import BcPrior

        bc = BcPrior.load(path)
        if bc.n_x != lcs.n_x or bc.n_u != lcs.n_u:
            raise ValueError(f"bc_prior n_x/n_u {bc.n_x}/{bc.n_u} != deploy {lcs.n_x}/{lcs.n_u}")
        if deploy is not None:
            bc.check_deploy(deploy)
        t = lcs.tensor
        self.lcs, self.H, self.iters = lcs, params.H, params.pgd_iters
        self.layers = [(t(w), t(b)) for w, b in bc.layers]
        self.use_p = bc.inputs == "zp"
        self.in_mean, self.in_std = t(bc.in_mean), t(bc.in_std)
        self.u_mid, self.u_half = t(bc.u_mid), t(bc.u_half)
        lb, ub = params.bounds(lcs)
        self.lb, self.ub = t(lb), t(ub)
        self.betas = lcs.nesterov_betas(self.iters)
        self.s_z, self.s_p = t(np.zeros(lcs.n_x)), t(np.zeros(N_P))
        self.graph = None
        cuda = lcs.device.type == "cuda"
        if (cuda and params.cuda_graph) if use_graph is None else use_graph:
            self._capture()

    def forward(self, Z: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """``u`` (unclipped) for ``Z (K, n_x)``, ``P (K, 6)`` (unused for a "z" prior)."""
        x = ((torch.cat([Z, P], -1) if self.use_p else Z) - self.in_mean) / self.in_std
        for i, (w, b) in enumerate(self.layers):
            x = torch.addmm(b, x, w)
            if i + 1 < len(self.layers):
                x = torch.relu(x)
        return self.u_mid + self.u_half * x

    def _body(self) -> None:
        z, p, out = self.s_z[None], self.s_p[None], []
        for _ in range(self.H):
            u = torch.maximum(torch.minimum(self.forward(z, p), self.ub), self.lb)
            z, _ = self.lcs.step(z, u, p, self.iters, self.betas)
            p = p + u[:, :N_P]
            out.append(u[0])
        self.out = torch.stack(out)

    def _capture(self) -> None:
        dev = self.lcs.device
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                self._body()
        torch.cuda.current_stream(dev).wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()

    def __call__(self, z0, p0) -> np.ndarray:
        """``(H, n_u)`` float64 prior sequence from ``z0 (n_x,)``, ``p0 (6,)``."""
        self.s_z.copy_(self.lcs.tensor(z0).reshape(-1))
        self.s_p.copy_(self.lcs.tensor(p0).reshape(-1))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._body()
        return self.out.double().cpu().numpy()
