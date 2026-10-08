"""MPPI on the exported latent LCS (torch, batched on the GPU).

``TorchLcs`` is ``latent_encoder.LearnedLcs`` (``z' = A z + B u + D lam + d``, ``lam`` from the
fixed-iteration Nesterov PGD on ``F_pgd = G G^T + stiffness I``) for K states at once, with the
EE positions ``p = [p_franka, p_ur]`` carried as ``p' = p + u[0:3] / u[3:6]`` and optional
``A_p``/``E_p`` couplings. ``mppi_plan`` samples input sequences around a warm-started mean and
costs latent goal distance + inputs + smoothness; EE positions only enter as safety bounds and
the optional data-support box (``MppiParams.support_box``, a bound, not a target).
``MppiParams.goal_set``: the goal distance is a soft-min over a set of encoded seated latents.
``MppiParams.value``: + weight x a learned cost-to-go ``V(z)`` (``latent_value``) at the terminal
knot or summed over knots 1..H.
"""

import dataclasses
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from task_common.latent_value import LatentValue, check_value, value_forward

N_P = 6  # [p_franka (3), p_ur (3)]
POS_SLICES = (slice(0, 3), slice(3, 6))
ROT_SLICES = (slice(6, 9), slice(9, 12))
ROT_CHANNELS = tuple(range(6, 12))
EPS = 1e-12
CLEARANCE_CHUNKS = 1  # threads per (sample, knot, set); more measured slower
CLEARANCE_CLUSTER_M = 0.01  # voxel edge of the point clusters (kernel culling)
CLEARANCE_KEYS = frozenset({"model", "ur_min_m", "franka_min_m", "w_soft", "soft_margin_m"})
OVERSHOOT_KEYS = frozenset({"seat_tol", "fade", "fade_last_stage", "hold", "near"})
PRIOR_SYNC = (None, "stage")
PRIOR_MODES = ("demo", "policy")
HOLD_KEYS = frozenset({"enter_k", "streak", "goal", "w_q", "w_T", "w_r", "w_s", "exit_k", "cold",
                       "stall"})
STALL_KEYS = frozenset({"latents", "eps", "k"})
HOLD_GOALS = ("seat", "entry", "best")
NEAR_KEYS = frozenset({"k", "w_T"})
SUPPORT_BOX_KEYS = frozenset({"lo", "hi", "weight", "hard_margin_m"})
W_KEYS = ("w_q", "w_T", "w_r", "w_s")


def _scalar(x):
    return np.asarray(x).item()


class TorchLcs:
    """The deploy's LCS as torch tensors (``dtype`` on ``device``)."""

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
        """``rollout`` as one Warp kernel (a thread per sample runs all H steps and the PGD);
        same maths, float32, on torch's current stream (CUDA-graph capturable)."""
        if self._fused is None:
            self._fused = _FusedRollout(self)
        return self._fused(z0, p0, U, rot0, iters)

    def whitened_dist(self, z, goal) -> float:
        """``LearnedLcs.whitened_dist`` (float64)."""
        diff = np.asarray(z, dtype=np.float64) - np.asarray(goal, dtype=np.float64)
        return float(np.linalg.norm(diff / self.z_std_np))


def _quat_exp_t(r: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` rotation vectors -> ``(..., 4)`` unit quaternions ``wxyz``."""
    a = r.norm(dim=-1, keepdim=True)
    s = torch.where(a > 1e-4, torch.sin(0.5 * a) / a.clamp_min(1e-30), 0.5 - a * a / 48.0)
    return torch.cat([torch.cos(0.5 * a), s * r], dim=-1)


def _qmul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat([aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)], dim=-1)


def _quat_angle_t(q: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.atan2(q[..., 1:].norm(dim=-1), q[..., 0].abs())


def _wp_funcs():
    import warp as wp

    @wp.func
    def quat_exp(r: wp.vec3):
        a = wp.length(r)
        s = 0.5 - a * a / 48.0
        if a > 1.0e-4:
            s = wp.sin(0.5 * a) / a
        return wp.quat(s * r[0], s * r[1], s * r[2], wp.cos(0.5 * a))

    @wp.func
    def quat_angle(q: wp.quat):
        return 2.0 * wp.atan2(wp.length(wp.vec3(q[0], q[1], q[2])), wp.abs(q[3]))

    return quat_exp, quat_angle


def _fused_kernel(nx: int, nl: int, nu: int):
    import warp as wp

    _wp_quat_exp, _wp_quat_angle = _wp_funcs()

    f32 = wp.float32
    vx, vl = wp.types.vector(nx, f32), wp.types.vector(nl, f32)
    vu, v6 = wp.types.vector(nu, f32), wp.types.vector(N_P, f32)
    mxx, mxu = wp.types.matrix((nx, nx), f32), wp.types.matrix((nx, nu), f32)
    mxl, mx6 = wp.types.matrix((nx, nl), f32), wp.types.matrix((nx, N_P), f32)
    mlx, mlu = wp.types.matrix((nl, nx), f32), wp.types.matrix((nl, nu), f32)
    mll, ml6 = wp.types.matrix((nl, nl), f32), wp.types.matrix((nl, N_P), f32)

    @wp.kernel(module="unique", enable_backward=False)
    def rollout(z0: wp.array(dtype=vx), p0: wp.array(dtype=v6), r0: wp.array(dtype=wp.vec3),
                U: wp.array2d(dtype=vu),
                A: wp.array(dtype=mxx), B: wp.array(dtype=mxu), D: wp.array(dtype=mxl),
                d: wp.array(dtype=vx), Ap: wp.array(dtype=mx6), sE: wp.array(dtype=mlx),
                sH: wp.array(dtype=mlu), sc: wp.array(dtype=vl), sEp: wp.array(dtype=ml6),
                Ms: wp.array(dtype=mll), betas: wp.array(dtype=f32), n_pgd: int, H: int,
                Z: wp.array2d(dtype=vx), P: wp.array2d(dtype=v6), R: wp.array2d(dtype=wp.vec2)):
        k = wp.tid()
        z = z0[0]
        p = p0[0]
        qf = _wp_quat_exp(r0[0])
        qu = _wp_quat_exp(r0[1])
        Z[k, 0] = z
        P[k, 0] = p
        R[k, 0] = wp.vec2(_wp_quat_angle(qf), _wp_quat_angle(qu))
        zero = vl()
        for h in range(H):
            u = U[k, h]
            nsq = -(sE[0] @ z + sH[0] @ u + sc[0] + sEp[0] @ p)
            lam = vl()
            y = vl()
            for i in range(n_pgd):
                lam_old = lam
                # lam = max(y - s (F y + q), 0) with Ms = I - s F
                lam = wp.max(nsq + Ms[0] @ y, zero)
                y = lam_old + (1.0 + betas[i]) * (lam - lam_old)
            z = A[0] @ z + B[0] @ u + D[0] @ lam + d[0] + Ap[0] @ p
            p = p + v6(u[0], u[1], u[2], u[3], u[4], u[5])
            qf = _wp_quat_exp(wp.vec3(u[6], u[7], u[8])) * qf
            qu = _wp_quat_exp(wp.vec3(u[9], u[10], u[11])) * qu
            Z[k, h + 1] = z
            P[k, h + 1] = p
            R[k, h + 1] = wp.vec2(_wp_quat_angle(qf), _wp_quat_angle(qu))

    types = {"vx": vx, "v6": v6, "vu": vu, "mxx": mxx, "mxu": mxu, "mxl": mxl, "mx6": mx6,
             "mlx": mlx, "mlu": mlu, "vl": vl, "ml6": ml6, "mll": mll}
    return rollout, types


def _clearance_kernel():
    import warp as wp

    _wp_quat_exp, _ = _wp_funcs()
    f32 = wp.float32
    vu, v6 = wp.types.vector(12, f32), wp.types.vector(N_P, f32)

    @wp.func
    def poly_dist(r: f32, z: f32, prof: wp.array2d(dtype=wp.vec2), j: int):
        # padded vertices give zero-length edges (t = 0)
        n = prof.shape[1]
        d2 = f32(1.0e30)
        inside = wp.int32(0)
        for i in range(n):
            A = prof[j, i]
            B = prof[j, (i + 1) % n]
            abx = B[0] - A[0]
            abz = B[1] - A[1]
            apx = r - A[0]
            apz = z - A[1]
            den = abx * abx + abz * abz
            t = f32(0.0)
            if den > f32(0.0):
                t = wp.clamp((apx * abx + apz * abz) / den, f32(0.0), f32(1.0))
            dx = apx - t * abx
            dz = apz - t * abz
            d2 = wp.min(d2, dx * dx + dz * dz)
            if (A[1] > z and B[1] <= z) or (A[1] <= z and B[1] > z):  # noqa: SIM102
                if r < A[0] + (z - A[1]) / abz * abx:
                    inside = 1 - inside
        d = wp.sqrt(d2)
        if inside == 1:
            d = -d
        return d

    @wp.func
    def box_sd(x: wp.vec3, h: wp.vec3):
        q = wp.vec3(wp.abs(x[0]) - h[0], wp.abs(x[1]) - h[1], wp.abs(x[2]) - h[2])
        out = wp.length(wp.vec3(wp.max(q[0], f32(0.0)), wp.max(q[1], f32(0.0)),
                                wp.max(q[2], f32(0.0))))
        return out + wp.min(wp.max(q[0], wp.max(q[1], q[2])), f32(0.0))

    @wp.func
    def cyl_bound(x: wp.vec3, lim: wp.vec2):
        a = wp.sqrt(x[0] * x[0] + x[1] * x[1]) - lim[0]
        b = wp.abs(x[2]) - lim[1]
        pa = wp.max(a, f32(0.0))
        pb = wp.max(b, f32(0.0))
        return wp.sqrt(pa * pa + pb * pb) + wp.min(wp.max(a, b), f32(0.0))

    @wp.func
    def set_dist(pts: wp.array(dtype=wp.vec3), cl: wp.array(dtype=wp.vec2i),
                 cc: wp.array(dtype=wp.vec4), c0: int, stride: int, Rq: wp.mat33, t: wp.vec3,
                 plate_R: wp.array(dtype=wp.mat33), plate_p: wp.array(dtype=wp.vec3),
                 plate_h: wp.array(dtype=wp.vec3), pul_R: wp.array(dtype=wp.mat33),
                 pul_p: wp.array(dtype=wp.vec3), pul_b: wp.array(dtype=wp.vec2),
                 prof: wp.array2d(dtype=wp.vec2)):
        # Board.min_distance; a cluster is skipped when centre bound - radius >= best.
        RbT = wp.transpose(plate_R[0])
        Mb = RbT * Rq
        bb = RbT * (t - plate_p[0])
        h = plate_h[0]
        best = f32(1.0e30)
        for ci in range(c0, cl.shape[0], stride):
            c = cc[ci]
            if box_sd(Mb * wp.vec3(c[0], c[1], c[2]) + bb, h) - c[3] < best:
                for i in range(cl[ci][0], cl[ci][1]):
                    best = wp.min(best, box_sd(Mb * pts[i] + bb, h))
        for j in range(pul_R.shape[0]):
            RjT = wp.transpose(pul_R[j])
            Mj = RjT * Rq
            bj = RjT * (t - pul_p[j])
            lim = pul_b[j]
            for ci in range(c0, cl.shape[0], stride):
                c = cc[ci]
                if cyl_bound(Mj * wp.vec3(c[0], c[1], c[2]) + bj, lim) - c[3] < best:
                    for i in range(cl[ci][0], cl[ci][1]):
                        x = Mj * pts[i] + bj
                        if cyl_bound(x, lim) < best:
                            best = wp.min(best, poly_dist(wp.sqrt(x[0] * x[0] + x[1] * x[1]),
                                                          x[2], prof, j))
        return best

    @wp.kernel(module="unique", enable_backward=False)
    def clear(P: wp.array2d(dtype=v6), U: wp.array2d(dtype=vu), q0: wp.array2d(dtype=wp.quat),
              pts_ur: wp.array(dtype=wp.vec3), cl_ur: wp.array(dtype=wp.vec2i),
              cc_ur: wp.array(dtype=wp.vec4), pts_f: wp.array(dtype=wp.vec3),
              cl_f: wp.array(dtype=wp.vec2i), cc_f: wp.array(dtype=wp.vec4),
              plate_R: wp.array(dtype=wp.mat33), plate_p: wp.array(dtype=wp.vec3),
              plate_h: wp.array(dtype=wp.vec3), pul_R: wp.array(dtype=wp.mat33),
              pul_p: wp.array(dtype=wp.vec3), pul_b: wp.array(dtype=wp.vec2),
              prof: wp.array2d(dtype=wp.vec2), n_chunks: int, C: wp.array3d(dtype=f32)):
        k, h, a, c = wp.tid()
        arm = 1 - a  # C: UR first; P / U / q0: Franka first
        q = q0[k % q0.shape[0], arm]
        for j in range(h):
            u = U[k, j]
            q = _wp_quat_exp(wp.vec3(u[6 + 3 * arm], u[7 + 3 * arm], u[8 + 3 * arm])) * q
        p = P[k, h]
        t = wp.vec3(p[3 * arm], p[3 * arm + 1], p[3 * arm + 2])
        Rq = wp.quat_to_matrix(q)
        d = f32(0.0)
        if a == 0:
            d = set_dist(pts_ur, cl_ur, cc_ur, c, n_chunks, Rq, t, plate_R, plate_p, plate_h,
                         pul_R, pul_p, pul_b, prof)
        else:
            d = set_dist(pts_f, cl_f, cc_f, c, n_chunks, Rq, t, plate_R, plate_p, plate_h,
                         pul_R, pul_p, pul_b, prof)
        wp.atomic_min(C, k, h, a, d)

    return clear, vu, v6


class Clearance:
    """``MppiParams.clearance``: frozen UR / Franka sets vs plate + pulleys per knot.

    ``__call__(P, U, q0)`` -> ``C (K, H+1, 2)`` min signed distance of the UR (0) / Franka (1)
    set at each knot: point = ``R(q_k) p_local + P[k, h]`` with ``q_k = exp(w_h) .. exp(w_1)
    q0`` (``q0`` (2, 4) or (K, 2, 4) absolute ``wxyz``); Warp kernel or torch (same maths).
    """

    def __init__(self, cfg: dict, lcs: TorchLcs, max_points: int | None = None) -> None:
        """``max_points``: cap of ``ClearanceModel.kernel_points`` (None: every point)."""
        from task_common.clearance_model import ClearanceModel

        self.lcs = lcs
        self.model = ClearanceModel.load(cfg["model"])
        self.model.check_frames()
        self.ur_min = float(cfg.get("ur_min_m", self.model.ur_min_m))
        self.franka_min = float(cfg.get("franka_min_m", self.model.franka_min_m))
        self.w_soft = float(cfg.get("w_soft", 0.0))
        self.soft_margin = float(cfg.get("soft_margin_m", 0.012))
        self.points = self.model.kernel_points(max_points)
        t = lcs.tensor
        self.lim = t(np.array([self.ur_min, self.franka_min]))
        self.pts = [t(p) for p in self.points]
        m = self.model
        self.plate = [t(m.plate[k]) for k in ("rot", "pos", "half")]
        self.pulleys = [[t(p[k]) for k in ("rot", "pos", "profile")] for p in m.pulleys]
        self._wp = None
        self.n_chunks, self.cluster_m = CLEARANCE_CHUNKS, CLEARANCE_CLUSTER_M

    def __call__(self, P: torch.Tensor, U: torch.Tensor, q0: torch.Tensor,
                 fused: bool = True) -> torch.Tensor:
        if fused and self.lcs.device.type == "cuda":
            return self._fused(P, U, q0)
        return self.torch_path(P, U, q0)

    def _arrays(self):
        import warp as wp

        kernel, vu, v6 = _clearance_kernel()
        dev = str(self.lcs.device) if self.lcs.device.index is not None else "cuda:0"
        m = self.model
        n_prof = max(len(p["profile"]) for p in m.pulleys)
        prof = np.stack([np.concatenate([p["profile"], np.repeat(p["profile"][-1:], n_prof
                                                                 - len(p["profile"]), 0)])
                         for p in m.pulleys]).astype(np.float32)
        bounds = [[p["profile"][:, 0].max(), np.abs(p["profile"][:, 1]).max()] for p in m.pulleys]

        def arr(x, dtype):
            return wp.array(np.asarray(x, np.float32), dtype=dtype, device=dev)

        sets = []
        for pts in self.points:
            order, spans, centres = point_clusters(pts, self.cluster_m)
            sets += [arr(pts[order], wp.vec3), wp.array(spans, dtype=wp.vec2i, device=dev),
                     arr(centres, wp.vec4)]
        geo = [*sets, arr([m.plate["rot"]], wp.mat33), arr([m.plate["pos"]], wp.vec3),
               arr([m.plate["half"]], wp.vec3), arr([p["rot"] for p in m.pulleys], wp.mat33),
               arr([p["pos"] for p in m.pulleys], wp.vec3), arr(bounds, wp.vec2),
               wp.array(prof, dtype=wp.vec2, device=dev)]
        return wp, kernel, vu, v6, dev, geo

    def _fused(self, P, U, q0):
        if self._wp is None:
            self._wp = self._arrays()
        wp, kernel, vu, v6, dev, geo = self._wp
        K, H1 = P.shape[0], P.shape[1]
        P, U = P.contiguous(), U.contiguous()
        q = q0.reshape(-1, 2, 4).roll(-1, dims=-1).contiguous()  # wxyz -> warp xyzw
        C = torch.full((K, H1, 2), 1.0e30, dtype=torch.float32, device=P.device)
        wp.launch(kernel, dim=(K, H1, 2, self.n_chunks), device=dev,
                  stream=wp.stream_from_torch(torch.cuda.current_stream(P.device)),
                  inputs=[wp.from_torch(P, dtype=v6), wp.from_torch(U, dtype=vu),
                          wp.from_torch(q, dtype=wp.quat), *geo, self.n_chunks],
                  outputs=[wp.from_torch(C)])
        return C

    def torch_path(self, P: torch.Tensor, U: torch.Tensor, q0: torch.Tensor,
                   chunk: int = 64) -> torch.Tensor:
        K, H1 = P.shape[0], P.shape[1]
        q = q0.reshape(-1, 2, 4).expand(K, 2, 4) if q0.numel() == 8 else q0.reshape(K, 2, 4)
        Q = [q]
        for j in range(H1 - 1):
            e = torch.stack([_quat_exp_t(U[:, j, 6:9]), _quat_exp_t(U[:, j, 9:12])], dim=1)
            q = _qmul_t(e, q)
            Q.append(q)
        Rm = _quat_mat_t(torch.stack(Q, dim=1))  # (K, H1, 2, 3, 3)
        t = P.reshape(K, H1, 2, 3)
        out = []
        for arm, pts in ((1, self.pts[0]), (0, self.pts[1])):
            Ra, ta = Rm[:, :, arm].reshape(-1, 3, 3), t[:, :, arm].reshape(-1, 3)
            res = [self._board_min(pts @ Ra[s:s + chunk].transpose(1, 2) + ta[s:s + chunk, None])
                   for s in range(0, Ra.shape[0], chunk)]
            out.append(torch.cat(res).reshape(K, H1))
        return torch.stack(out, dim=-1)

    def _board_min(self, W: torch.Tensor) -> torch.Tensor:
        """``(B, n, 3)`` world points -> ``(B,)`` min signed distance (no bound skip)."""
        rot, pos, half = self.plate
        q = ((W - pos) @ rot).abs() - half
        d = q.clamp_min(0.0).norm(dim=-1) + q.amax(-1).clamp_max(0.0)
        for rot, pos, prof in self.pulleys:
            loc = (W - pos) @ rot
            d = torch.minimum(d, _polygon_distance_t(loc[..., :2].norm(dim=-1), loc[..., 2], prof))
        return d.amin(-1)


def point_clusters(points: np.ndarray, cell: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(order, spans (n_cl, 2) int32, centres (n_cl, 4) xyz + radius)``: voxel clusters of
    ``points``, lowest cluster first; ``points[order]`` is contiguous per cluster."""
    keys = np.floor(np.asarray(points) / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    groups = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    groups.sort(key=lambda g: (float(points[g, 2].min()), int(g[0])))
    order = np.concatenate(groups)
    ends = np.cumsum([len(g) for g in groups])
    spans = np.stack([ends - [len(g) for g in groups], ends], 1).astype(np.int32)
    centres = []
    for g in groups:
        c = 0.5 * (points[g].min(0) + points[g].max(0))
        centres.append(np.r_[c, np.linalg.norm(points[g] - c, axis=1).max() * (1 + 1e-5) + 1e-7])
    return order, spans, np.asarray(centres)


def _quat_mat_t(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` unit ``wxyz`` -> ``(..., 3, 3)``."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        dim=-1).reshape(*q.shape[:-1], 3, 3)


def _polygon_distance_t(r: torch.Tensor, z: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
    """``clearance._polygon_distance`` on ``(r, z)`` (any shape) vs a closed ``(E, 2)`` polygon."""
    a, b = poly, poly.roll(-1, dims=0)
    ab = b - a
    apx, apz = r[..., None] - a[:, 0], z[..., None] - a[:, 1]
    t = ((apx * ab[:, 0] + apz * ab[:, 1]) / (ab * ab).sum(-1)).clamp(0.0, 1.0)
    d = torch.hypot(apx - t * ab[:, 0], apz - t * ab[:, 1]).amin(-1)
    zc = z[..., None]
    straddles = (a[:, 1] > zc) != (b[:, 1] > zc)
    den = torch.where(ab[:, 1] == 0.0, torch.ones_like(ab[:, 1]), ab[:, 1])
    x_at = a[:, 0] + (zc - a[:, 1]) / den * ab[:, 0]
    inside = (straddles & (r[..., None] < x_at)).sum(-1) % 2 == 1
    return torch.where(inside, -d, d)


class _FusedRollout:
    def __init__(self, lcs: TorchLcs) -> None:
        import warp as wp

        self.wp, self.lcs = wp, lcs
        self.kernel, ty = _fused_kernel(lcs.n_x, lcs.n_lam, lcs.n_u)
        self.ty = ty
        dev = str(lcs.device) if lcs.device.index is not None else "cuda:0"
        self.dev = dev
        s = lcs.step_size
        f_pgd = lcs.np["F_pgd"]

        def arr(x, t):
            return wp.array([np.asarray(x, dtype=np.float32)], dtype=t, device=dev)

        n = lcs.np
        self.mats = [arr(n["A"], ty["mxx"]), arr(n["B"], ty["mxu"]), arr(n["D"], ty["mxl"]),
                     arr(n["d"], ty["vx"]), arr(n["A_p"], ty["mx6"]), arr(s * n["E"], ty["mlx"]),
                     arr(s * n["H"], ty["mlu"]), arr(s * n["c"], ty["vl"]),
                     arr(s * n["E_p"], ty["ml6"]),
                     arr(np.eye(lcs.n_lam) - s * f_pgd, ty["mll"])]
        self._betas: dict[int, object] = {}

    def __call__(self, z0, p0, U, rot0, iters):
        wp, lcs = self.wp, self.lcs
        K, H, _ = U.shape
        if int(iters) not in self._betas:
            self._betas[int(iters)] = wp.array(np.asarray(lcs.nesterov_betas(iters), np.float32),
                                               dtype=wp.float32, device=self.dev)
        z0 = lcs.tensor(z0).reshape(1, lcs.n_x).contiguous()
        p0 = lcs.tensor(p0).reshape(1, N_P).contiguous()
        U = U.contiguous()
        r0 = lcs.tensor(np.zeros((2, 3)) if rot0 is None else rot0).reshape(2, 3).contiguous()
        Z = torch.empty((K, H + 1, lcs.n_x), dtype=torch.float32, device=lcs.device)
        P = torch.empty((K, H + 1, N_P), dtype=torch.float32, device=lcs.device)
        R = torch.empty((K, H + 1, 2), dtype=torch.float32, device=lcs.device)
        ty = self.ty
        wp.launch(self.kernel, dim=K, device=self.dev,
                  stream=wp.stream_from_torch(torch.cuda.current_stream(lcs.device)),
                  inputs=[wp.from_torch(z0, dtype=ty["vx"]), wp.from_torch(p0, dtype=ty["v6"]),
                          wp.from_torch(r0, dtype=wp.vec3), wp.from_torch(U, dtype=ty["vu"]),
                          *self.mats, self._betas[int(iters)], int(iters), int(H)],
                  outputs=[wp.from_torch(Z, dtype=ty["vx"]), wp.from_torch(P, dtype=ty["v6"]),
                           wp.from_torch(R, dtype=wp.vec2)])
        return Z, P, R


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


def check_support_box(sb: dict) -> None:
    """``MppiParams.support_box`` keys and ranges."""
    if not set(sb) <= SUPPORT_BOX_KEYS or not {"lo", "hi", "weight"} <= set(sb):
        raise ValueError(f"support_box keys {sorted(sb)}: need lo, hi, weight, within "
                         f"{sorted(SUPPORT_BOX_KEYS)}")
    lo, hi = np.asarray(sb["lo"], dtype=np.float64), np.asarray(sb["hi"], dtype=np.float64)
    if lo.shape != (N_P,) or hi.shape != (N_P,) or not np.all(np.isfinite(np.r_[lo, hi])) \
            or np.any(lo > hi):
        raise ValueError(f"support_box lo {sb['lo']} / hi {sb['hi']}: 6 finite, lo <= hi")
    if not float(sb["weight"]) >= 0.0:
        raise ValueError(f"support_box.weight {sb['weight']} must be >= 0")
    m = sb.get("hard_margin_m")
    if m is not None and not float(m) >= 0.0:
        raise ValueError(f"support_box.hard_margin_m {m} must be >= 0 or None")


def support_hinge(P: torch.Tensor, sbox: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` metres outside ``sbox`` = ``[lo, hi]`` ``(2, 6)`` per coordinate (0 inside)."""
    return (sbox[0] - P).clamp_min(0.0) + (P - sbox[1]).clamp_min(0.0)


def load_goal_set(path, n_x: int | None = None) -> np.ndarray:
    """``(K, n_x)`` float64 ``z_goals`` of a ``goal_set.py`` npz."""
    with np.load(Path(path), allow_pickle=False) as f:
        G = np.asarray(f["z_goals"], dtype=np.float64)
    if G.ndim != 2 or len(G) < 1 or (n_x is not None and G.shape[1] != n_x) \
            or not np.all(np.isfinite(G)):
        raise ValueError(f"goal set {path}: z_goals {G.shape} must be finite (K >= 1, {n_x})")
    return G


def load_value(v: dict, n_x: int, z_std) -> LatentValue:
    """``MppiParams.value``'s network, refused unless trained on this latent (n_x, z_std)."""
    vf = LatentValue.load(v["path"])
    if vf.n_x != n_x:
        raise ValueError(f"value {v['path']}: n_x {vf.n_x} != deploy {n_x}")
    vf.check_z_std(z_std)
    return vf


def value_tensors(lcs: "TorchLcs", v: dict) -> tuple:
    return load_value(v, lcs.n_x, lcs.z_std_np).torch_params(lcs.tensor)


def softmin_sq(D: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft-min over the last dim: ``m - tau log mean exp(-(D - m) / tau)``, ``m = min D``; in
    [min, mean], == min for tau 0 or one column, == D for identical columns."""
    m = D.amin(-1)
    if tau <= 0.0 or D.shape[-1] == 1:
        return m
    return m - tau * torch.log(torch.exp(-(D - m.unsqueeze(-1)) / tau).mean(-1))


def goal_set_dist(z, G: np.ndarray, z_std: np.ndarray, tau: float = 0.0) -> float:
    """Whitened distance of ``z`` to the set ``G`` (sqrt of ``softmin_sq`` of d^2, float64)."""
    d2 = np.sum(((np.asarray(z, np.float64) - G) / z_std) ** 2, axis=-1)
    return float(np.sqrt(softmin_sq(torch.as_tensor(d2, dtype=torch.float64), tau)))


def scaled_weights(params: "MppiParams", scales: dict) -> "MppiParams":
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

    def params_for(self, params: "MppiParams", mode: str) -> "MppiParams":
        if mode == "hold":
            return scaled_weights(params, self.hold_cfg)
        if mode == "near":
            return scaled_weights(params, {"w_T": self.ov["near"].get("w_T", 1.0)})
        return params

    def variants(self, params: "MppiParams") -> list["MppiParams"]:
        """The non-normal weight sets this guard can use (node warm-up)."""
        out = [self.params_for(params, "hold")] if self.hold_cfg is not None else []
        return out + ([self.params_for(params, "near")] if self.ov.get("near") else [])


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


def plan_cost(lcs: TorchLcs, params: MppiParams, Z: torch.Tensor, P: torch.Tensor,
              R: torch.Tensor, U: torch.Tensor, z_goal: torch.Tensor, half: torch.Tensor,
              u_anchor: torch.Tensor | None, lim: torch.Tensor | None = None,
              anchor_on: torch.Tensor | None = None, clr: Clearance | None = None,
              C: torch.Tensor | None = None, u_demo: torch.Tensor | None = None,
              prior_on: torch.Tensor | None = None, sbox: torch.Tensor | None = None,
              gset: torch.Tensor | None = None, vt: tuple | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cost (K,), violating knots (K,), envelope violated (K,) bool)`` of rolled-out plans
    (knots 1..H).

    ``lim`` = ``safety_limits`` as a tensor (default: from ``params``); ``anchor_on`` (0/1)
    gates the ``u_anchor`` smoothness term without a branch (CUDA graphs). ``C`` = the
    ``Clearance`` distances of the rollout (knots under ``clr``'s thresholds count as violating;
    ``clr.w_soft`` x hinge^2 below ``soft_margin_m``). ``z_goal`` ``(n_x,)`` or one row per
    knot ``(H, n_x)``. ``u_demo`` ``(H, n_u)``: the demo action prior (``params.prior_weight``),
    gated by ``prior_on`` (0/1) like the anchor. ``sbox`` ``(2, 6)``: ``params.support_box``'s
    ``[lo, hi]`` as a tensor (default: from ``params``). ``gset`` ``(K_g, n_x)``: with
    ``params.goal_set`` the set replacing ``z_goal`` (default: loaded from ``params``). ``vt``:
    ``params.value``'s ``value_tensors`` (default: loaded from ``params``).
    """
    if lim is None:
        lim = lcs.tensor(safety_limits(params))
    w = 1.0 / lcs.z_std
    if params.goal_set is None:
        dz = (Z[:, 1:] - z_goal) * w
        track = (dz * dz).sum(-1)
    else:
        if gset is None:
            gset = lcs.tensor(load_goal_set(params.goal_set, lcs.n_x))
        dz = (Z[:, 1:, None] - gset) * w
        track = softmin_sq((dz * dz).sum(-1), float(params.goal_set_tau))
    cost = params.w_q * track.sum(1) + params.w_T * track[:, -1]
    vv = params.value
    if vv is not None:
        if vt is None:
            vt = value_tensors(lcs, vv)
        if vv.get("where", "terminal") == "terminal":
            cost = cost + float(vv["weight"]) * value_forward(vt, Z[:, -1])
        else:
            cost = cost + float(vv["weight"]) * value_forward(vt, Z[:, 1:]).sum(1)
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
    pz_f, pz_u = P[:, 1:, 2], P[:, 1:, 5]
    env = (R[:, 1:, 0] > lim[4]) | (R[:, 1:, 1] > lim[5])
    bad = (pz_f < lim[0]) | (pz_f > lim[1]) | (pz_u < lim[2]) | (pz_u > lim[3]) | env
    if C is not None:
        bad = bad | clearance_bad(C, clr.lim)
        if clr.w_soft > 0.0:
            cost = cost + clr.w_soft * (clr.soft_margin - C[:, 1:]).clamp_min(0.0).square().sum(
                (1, 2))
    sb = params.support_box
    if sb is not None:
        if sbox is None:
            sbox = lcs.tensor(support_limits(sb))
        hinge = support_hinge(P[:, 1:], sbox)
        cost = cost + float(sb["weight"]) * (hinge * 1e3).square().sum((1, 2))
        if sb.get("hard_margin_m") is not None:
            bad = bad | (hinge > float(sb["hard_margin_m"])).any(-1)
    n_bad = bad.sum(1).to(cost.dtype)
    return cost + params.safety_penalty * n_bad, n_bad, env.any(1)


def support_limits(sb: dict) -> np.ndarray:
    """``(2, 6)`` float64 ``[lo, hi]`` of ``MppiParams.support_box``."""
    return np.stack([np.asarray(sb[k], dtype=np.float64) for k in ("lo", "hi")])


def clearance_bad(C: torch.Tensor, clim: torch.Tensor) -> torch.Tensor:
    """``(K, H)`` knots 1..H with the UR or the Franka set under its threshold."""
    return (C[:, 1:, 0] < clim[0]) | (C[:, 1:, 1] < clim[1])


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
N_STATS = 10  # cost_min/mean, plan cost/viol, entropy, frac_pen, fallback, ess, temp, frac_env
# clearance on: frac_clear_penalised, plan min UR / Franka (knots 1..H), plan violating knots,
# knot-0 UR / Franka (the current poses)
N_CLEAR_STATS = 6


class MppiSolver:
    """One MPPI solve = ``n_iter`` passes + the mean plan's evaluation, on static buffers.

    On CUDA the whole solve is one captured CUDA graph (fixed shapes; noise drawn into a static
    buffer before the replay); limits/anchor are buffer inputs so one graph serves every call.
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
        self.clear = clearance_for(lcs, params)
        self.clear_fused = cuda and params.fused_rollout
        self.s_sbox = (None if params.support_box is None
                       else t(support_limits(params.support_box)))
        self.g_set = (None if params.goal_set is None
                      else load_goal_set(params.goal_set, nx))
        self.s_gset = None if self.g_set is None else t(self.g_set)
        self.vfn = None if params.value is None else load_value(params.value, nx, lcs.z_std_np)
        self.s_value = None if self.vfn is None else self.vfn.torch_params(t)
        if self.clear is not None:
            self.s_q0 = t(np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)))
        self.graph = None
        if self.use_graph:
            self._capture()

    def _weights(self, cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _body(self) -> None:
        lcs, p, box = self.lcs, self.p, self.box
        mean = torch.maximum(torch.minimum(self.s_mean0, box.ub), box.lb)
        for i in range(p.n_iter):
            U = torch.maximum(torch.minimum(mean + box.sigma * self.s_eps[i], box.ub), box.lb)
            U[0] = mean
            Z, P, R = self.roll(self.s_z0, self.s_p0, U, self.s_rot0, p.pgd_iters)
            C = self._clearance(P, U)
            cost, n_bad, env = plan_cost(lcs, p, Z, P, R, U, self.s_goal, box.half, self.s_anchor,
                                    self.s_lim, self.s_anchor_on, self.clear, C, self.s_udemo,
                                    self.s_prior_on, self.s_sbox, self.s_gset, self.s_value)
            w, ess, temp = self._weights(cost)
            mean = (w[:, None, None] * U).sum(0)
        Zm, Pm, Rm = self.roll(self.s_z0, self.s_p0, mean[None], self.s_rot0, p.pgd_iters)
        Cm = self._clearance(Pm, mean[None])
        cost_m, bad_m, _ = plan_cost(lcs, p, Zm, Pm, Rm, mean[None], self.s_goal, box.half,
                                  self.s_anchor, self.s_lim, self.s_anchor_on, self.clear, Cm,
                                  self.s_udemo, self.s_prior_on, self.s_sbox, self.s_gset,
                                  self.s_value)
        best = torch.argmin(cost)
        fb = (bad_m[0] > 0) & (_pick(n_bad, best) < bad_m[0])
        plan = torch.where(fb, _pick(U, best), mean)
        Zp = torch.where(fb, _pick(Z, best), Zm[0])
        Pp = torch.where(fb, _pick(P, best), Pm[0])
        stats = torch.stack([
            cost.min(), cost.mean(), torch.where(fb, _pick(cost, best), cost_m[0]),
            torch.where(fb, _pick(n_bad, best), bad_m[0]),
            -(w * torch.log(w.clamp_min(EPS))).sum(), (n_bad > 0).to(cost.dtype).mean(),
            fb.to(cost.dtype), ess, temp, env.to(cost.dtype).mean()])
        parts = [plan.reshape(-1), Zp.reshape(-1), Pp.reshape(-1), stats]
        if self.clear is not None:
            Cp = torch.where(fb, _pick(C, best), Cm[0])
            parts += [Cp.reshape(-1), torch.stack([
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

    def solve(self, z0, p0, rot0, z_goal, u_prev_plan, gen: torch.Generator,
              limits: np.ndarray | None = None, q_now=None, u_demo=None
              ) -> tuple[np.ndarray, dict]:
        """``q_now`` (2, 4) ``wxyz``: the current absolute Franka / UR orientations (needed
        with ``clearance``). ``z_goal`` ``(n_x,)`` (every knot) or ``(H, n_x)`` (knots 1..H).
        ``u_demo`` ``(H, n_u)``: demo actions for the prior (None: prior off this solve)."""
        t0 = time.perf_counter()
        lcs, p = self.lcs, self.p
        H, nu, nx = p.H, lcs.n_u, lcs.n_x
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
        out = self.out.double().cpu().numpy()
        n_plan, n_z = H * nu, (H + 1) * nx
        plan = out[:n_plan].reshape(H, nu)
        Zp = out[n_plan:n_plan + n_z].reshape(H + 1, nx)
        Pp = out[n_plan + n_z:n_plan + n_z + (H + 1) * N_P].reshape(H + 1, N_P)
        extra = {}
        if self.clear is not None:
            n_c = (H + 1) * 2
            cs = out[-N_CLEAR_STATS:]
            out = out[:-N_CLEAR_STATS]
            extra = {"frac_clear_penalised": cs[0], "clear_min_ur_mm": cs[1] * 1e3,
                     "clear_min_f_mm": cs[2] * 1e3, "clear_violating": int(cs[3]),
                     "clear_now_ur_mm": cs[4] * 1e3, "clear_now_f_mm": cs[5] * 1e3,
                     "C": out[-n_c:].reshape(H + 1, 2)}
            out = out[:-n_c]
        st = out[-N_STATS:]
        info = {"cost_min": st[0], "cost_mean": st[1], "plan_cost": st[2],
                "plan_violations": int(st[3]), "weight_entropy": st[4], "frac_penalised": st[5],
                "fallback": bool(st[6]), "ess": st[7], "temperature_abs": st[8],
                "frac_env_penalised": st[9],
                "goal_dist_H": (lcs.whitened_dist(Zp[-1], goal[-1]) if self.g_set is None else
                                goal_set_dist(Zp[-1], self.g_set, lcs.z_std_np, p.goal_set_tau)),
                "Z": Zp, "P": Pp,
                **extra, "solve_ms": (time.perf_counter() - t0) * 1e3}
        if self.vfn is not None:
            info["value_H"] = float(self.vfn(Zp[-1]))
        return plan, info


def goal_rows(z_goal, H: int, n_x: int) -> np.ndarray:
    """``(H, n_x)`` float64: a single goal repeated per knot, or a per-knot sequence as is."""
    g = np.asarray(z_goal, dtype=np.float64)
    if g.shape == (n_x,):
        return np.ascontiguousarray(np.broadcast_to(g, (H, n_x)))
    if g.shape != (H, n_x):
        raise ValueError(f"z_goal shape {g.shape}: expected ({n_x},) or ({H}, {n_x})")
    return g.copy()


_SOLVERS: dict = {}
_CLEARANCE: dict = {}


def clearance_for(lcs: TorchLcs, params: MppiParams) -> Clearance | None:
    if params.clearance is None:
        return None
    key = (id(lcs), repr(sorted(params.clearance.items())))
    if key not in _CLEARANCE:
        _CLEARANCE[key] = Clearance(params.clearance, lcs)
    return _CLEARANCE[key]


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

    Sample 0 of every pass is the current mean. The returned plan is the weighted mean, unless
    it violates more safety knots than the best sample (rejection). ``q_now`` (2, 4) ``wxyz``:
    the current orientations, needed with ``params.clearance``. ``z_goal`` ``(n_x,)`` or
    ``(H, n_x)``; ``u_demo`` ``(H, n_u)`` = the demo action prior's inputs (``prior_*``).
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
    clr = clearance_for(lcs, params)
    if clr is not None and q_now is None:
        raise ValueError("clearance needs q_now (the current wxyz orientations)")
    C = None if clr is None else clr.torch_path(P, Ut, lcs.tensor(q_now))
    ud = None if u_demo is None else lcs.tensor(np.asarray(u_demo, dtype=np.float64))
    cost, n_bad, _ = plan_cost(lcs, params, Z, P, R, Ut, lcs.tensor(z_goal), box.half, anchor,
                               clr=clr, C=C, u_demo=ud)
    return float(cost[0]), int(n_bad[0]), Z[0].double().cpu().numpy()


class PolicyPrior:
    """``prior_mode: policy``: the BC prior ``pi`` (``bc_prior.BcPrior`` weights) rolled through
    the LCS from the current ``(z, P)``: ``u_k = clip(pi(z_k[, P_k]))`` to ``params.bounds`` (P
    an input only for a "zp" prior),
    ``z_{k+1}`` = LCS step, ``P_{k+1} = P_k + u_k[:6]``, k < H. Runs before the solve, as its own
    CUDA graph on CUDA (the solver's graph is unchanged)."""

    def __init__(self, path, lcs: TorchLcs, params: MppiParams, deploy=None,
                 use_graph: bool | None = None) -> None:
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
