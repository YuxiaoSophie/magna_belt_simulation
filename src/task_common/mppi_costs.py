"""Optional MPPI cost terms, each an ``MppiParams`` field that is off (None) by default.

- ``clearance``: gripper point sets vs the plate + pulleys (``Clearance``); knots under the
  thresholds violate, ``w_soft`` x hinge^2 below ``soft_margin_m``.
- ``support_box``: data-support box on the EE positions ``P`` (a bound, not a target).
- ``latent_support`` / ``ee_support``: hinge^2 of the distance to the nearest training latent /
  training EE position beyond a radius.
- ``goal_set``: soft-min goal distance over a set of seated latents.
- ``value``: a learned cost-to-go ``V(z)`` (``latent_value``).

``load_cost_terms`` loads whatever is on once per solver into a ``CostTerms``.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from task_common.latent_value import LatentValue

N_P = 6  # P = [p_franka (3), p_ur (3)]
POS_SLICES = (slice(0, 3), slice(3, 6))
CLEARANCE_CHUNKS = 1  # threads per (sample, knot, set); more measured slower
CLEARANCE_CLUSTER_M = 0.01  # voxel edge of the point clusters (kernel culling)
CLEARANCE_KEYS = frozenset({"model", "ur_min_m", "franka_min_m", "w_soft", "soft_margin_m"})
SUPPORT_BOX_KEYS = frozenset({"lo", "hi", "weight", "hard_margin_m"})
LATENT_SUPPORT_KEYS = frozenset({"file", "radius", "weight", "max_points", "where"})
LATENT_SUPPORT_WHERE = ("all", "terminal")
LATENT_SUPPORT_DEFAULTS = {"radius": 1.5, "weight": 1.0, "max_points": 3000, "where": "all"}
LATENT_SUPPORT_CHUNK_ELEMS = 1 << 23  # query rows x training points per cdist chunk (32 MB f32)
EE_SUPPORT_KEYS = frozenset({"file", "pulley_pose7", "radius_mm", "weight", "max_points",
                             "where"})
EE_SUPPORT_DEFAULTS = {"radius_mm": 35.0, "weight": 1.0, "max_points": 3000, "where": "all"}
EE_SUPPORT_SCALE_MM = 10.0


# --- torch quaternions (wxyz) -------------------------------------------------------------------

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


def _quat_mat_t(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` unit ``wxyz`` -> ``(..., 3, 3)``."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        dim=-1).reshape(*q.shape[:-1], 3, 3)


# --- all terms, loaded once per solver ----------------------------------------------------------

@dataclass
class CostTerms:
    """The optional terms of one ``MppiParams`` on the solver's device (None = off)."""

    clear: "Clearance | None" = None
    sbox: torch.Tensor | None = None  # (2, 6) support box [lo, hi], m
    gset: torch.Tensor | None = None  # (K_g, n_x) goal set
    value: tuple | None = None  # LatentValue.torch_params
    lsup: torch.Tensor | None = None  # (M, n_x) training latents, whitened
    esup: torch.Tensor | None = None  # (M, 6) training EE positions, mm
    # host copies for the solve's info (float64)
    gset_np: np.ndarray | None = None
    vfn: LatentValue | None = None
    lsup_np: np.ndarray | None = None
    esup_np: np.ndarray | None = None


def load_cost_terms(lcs, params, with_clearance: bool = True) -> CostTerms:
    """The terms ``params`` turns on, for ``lcs`` (a ``TorchLcs``)."""
    t, nx = lcs.tensor, lcs.n_x
    T = CostTerms()
    if with_clearance:
        T.clear = clearance_for(lcs, params)
    if params.support_box is not None:
        T.sbox = t(support_limits(params.support_box))
    if params.goal_set is not None:
        T.gset_np = load_goal_set(params.goal_set, nx)
        T.gset = t(T.gset_np)
    if params.value is not None:
        T.vfn = load_value(params.value, nx, lcs.z_std_np)
        T.value = T.vfn.torch_params(t)
    if params.latent_support is not None:
        T.lsup_np = load_latent_support(params.latent_support, nx) / lcs.z_std_np
        T.lsup = t(T.lsup_np)
    if params.ee_support is not None:
        T.esup_np = load_ee_support(params.ee_support) * 1e3
        T.esup = t(T.esup_np)
    return T


# --- clearance ----------------------------------------------------------------------------------

class Clearance:
    """``MppiParams.clearance``: frozen UR / Franka sets vs plate + pulleys per knot.

    ``__call__(P, U, q0)`` -> ``C (K, H+1, 2)`` min signed distance of the UR (0) / Franka (1)
    set at each knot: point = ``R(q_k) p_local + P[k, h]`` with ``q_k = exp(w_h) .. exp(w_1)
    q0`` (``q0`` (2, 4) or (K, 2, 4) absolute ``wxyz``); Warp kernel or torch (same maths).
    """

    def __init__(self, cfg: dict, lcs, max_points: int | None = None) -> None:
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

        from task_common.mppi_warp import _clearance_kernel

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


_CLEARANCE: dict = {}


def clearance_for(lcs, params) -> Clearance | None:
    """Cached ``Clearance`` per (lcs, clearance config); None when off."""
    if params.clearance is None:
        return None
    key = (id(lcs), repr(sorted(params.clearance.items())))
    if key not in _CLEARANCE:
        _CLEARANCE[key] = Clearance(params.clearance, lcs)
    return _CLEARANCE[key]


def clearance_bad(C: torch.Tensor, clim: torch.Tensor) -> torch.Tensor:
    """``(K, H)`` knots 1..H with the UR or the Franka set under its threshold."""
    return (C[:, 1:, 0] < clim[0]) | (C[:, 1:, 1] < clim[1])


# --- support box --------------------------------------------------------------------------------

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


def support_limits(sb: dict) -> np.ndarray:
    """``(2, 6)`` float64 ``[lo, hi]`` of ``MppiParams.support_box``."""
    return np.stack([np.asarray(sb[k], dtype=np.float64) for k in ("lo", "hi")])


def support_hinge(P: torch.Tensor, sbox: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` metres outside ``sbox`` = ``[lo, hi]`` ``(2, 6)`` per coordinate (0 inside)."""
    return (sbox[0] - P).clamp_min(0.0) + (P - sbox[1]).clamp_min(0.0)


# --- latent / EE data support -------------------------------------------------------------------

def check_latent_support(ls: dict) -> None:
    """``MppiParams.latent_support`` keys and ranges (the npz is checked when loaded)."""
    if not isinstance(ls, dict) or not set(ls) <= LATENT_SUPPORT_KEYS or "file" not in ls:
        raise ValueError(f"latent_support {ls!r}: need file, within "
                         f"{sorted(LATENT_SUPPORT_KEYS)}")
    if not isinstance(ls["file"], (str, Path)):
        raise TypeError(f"latent_support.file {ls['file']!r} must be an npz path")
    c = {**LATENT_SUPPORT_DEFAULTS, **ls}
    for k in ("radius", "weight"):
        if not (math.isfinite(float(c[k])) and float(c[k]) >= 0.0):
            raise ValueError(f"latent_support.{k} {c[k]} must be finite >= 0")
    m = c["max_points"]
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError(f"latent_support.max_points {m} must be an int >= 1")
    if c["where"] not in LATENT_SUPPORT_WHERE:
        raise ValueError(f"latent_support.where {c['where']!r}: one of {LATENT_SUPPORT_WHERE}")


def load_latent_support(ls: dict, n_x: int | None = None) -> np.ndarray:
    """``(M, n_x)`` float64 training latents of ``MppiParams.latent_support`` (un-whitened),
    every ``ceil(N / max_points)``-th row of the npz's ``z_t``."""
    with np.load(Path(ls["file"]), allow_pickle=False) as f:
        Z = np.asarray(f["z_t"], dtype=np.float64)
    if Z.ndim != 2 or len(Z) < 1 or (n_x is not None and Z.shape[1] != n_x) \
            or not np.all(np.isfinite(Z)):
        raise ValueError(f"latent support {ls['file']}: z_t {Z.shape} must be finite (N >= 1, "
                         f"{n_x})")
    m = int(ls.get("max_points", LATENT_SUPPORT_DEFAULTS["max_points"]))
    return Z[::-(-len(Z) // m)].copy()


def latent_support_cost(lcs, ls: dict, Z: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """``(K,)`` ``weight sum_k relu(d_k - radius)^2`` of rollouts ``Z (K, H+1, n_x)`` against
    the whitened training latents ``S``."""
    c = {**LATENT_SUPPORT_DEFAULTS, **ls}
    Zk = Z[:, -1:] if c["where"] == "terminal" else Z[:, 1:]
    d = nn_dist(Zk / lcs.z_std, S)
    return float(c["weight"]) * (d - float(c["radius"])).clamp_min(0.0).square().sum(1)


def check_ee_support(es: dict) -> None:
    """``MppiParams.ee_support`` keys and ranges (the npz is checked when loaded)."""
    if not isinstance(es, dict) or not set(es) <= EE_SUPPORT_KEYS \
            or not {"file", "pulley_pose7"} <= set(es):
        raise ValueError(f"ee_support {es!r}: need file, pulley_pose7, within "
                         f"{sorted(EE_SUPPORT_KEYS)}")
    if not isinstance(es["file"], (str, Path)):
        raise TypeError(f"ee_support.file {es['file']!r} must be an npz path")
    pose = np.asarray(es["pulley_pose7"], dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)) or np.linalg.norm(pose[3:]) < 1e-6:
        raise ValueError(f"ee_support.pulley_pose7 {es['pulley_pose7']}: 7 finite, xyz + wxyz")
    c = {**EE_SUPPORT_DEFAULTS, **es}
    for k in ("radius_mm", "weight"):
        if not (math.isfinite(float(c[k])) and float(c[k]) >= 0.0):
            raise ValueError(f"ee_support.{k} {c[k]} must be finite >= 0")
    m = c["max_points"]
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError(f"ee_support.max_points {m} must be an int >= 1")
    if c["where"] not in LATENT_SUPPORT_WHERE:
        raise ValueError(f"ee_support.where {c['where']!r}: one of {LATENT_SUPPORT_WHERE}")


def load_ee_support(es: dict) -> np.ndarray:
    """``(M, 6)`` float64 training EE positions of ``MppiParams.ee_support`` in ``P``'s frame
    (m): ``R q + t`` per arm of every ``ceil(N / max_points)``-th ``ee_pos`` row."""
    with np.load(Path(es["file"]), allow_pickle=False) as f:
        E = np.asarray(f["ee_pos"], dtype=np.float64)
    if E.ndim != 2 or len(E) < 1 or E.shape[1] != N_P or not np.all(np.isfinite(E)):
        raise ValueError(f"ee support {es['file']}: ee_pos {E.shape} must be finite (N >= 1, 6)")
    E = E[::-(-len(E) // int(es.get("max_points", EE_SUPPORT_DEFAULTS["max_points"])))]
    pose = np.asarray(es["pulley_pose7"], dtype=np.float64)
    R = _quat_mat_t(torch.as_tensor(pose[3:] / np.linalg.norm(pose[3:]))).numpy()
    return np.concatenate([E[:, sl] @ R.T + pose[:3] for sl in POS_SLICES], axis=1)


def ee_support_cost(es: dict, P: torch.Tensor, S_mm: torch.Tensor) -> torch.Tensor:
    """``(K,)`` ``weight sum_k relu((d_k - radius_mm) / EE_SUPPORT_SCALE_MM)^2`` of rollouts
    ``P (K, H+1, 6)`` (m) against the training EE positions ``S_mm`` (mm, ``P``'s frame)."""
    c = {**EE_SUPPORT_DEFAULTS, **es}
    Pk = P[:, -1:] if c["where"] == "terminal" else P[:, 1:]
    d = nn_dist(Pk * 1e3, S_mm)
    return float(c["weight"]) * ((d - float(c["radius_mm"])) / EE_SUPPORT_SCALE_MM).clamp_min(
        0.0).square().sum(1)


def nn_dist(Q: torch.Tensor, S: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
    """``(...)`` Euclidean distance of each row of ``Q (..., n)`` to its nearest row of
    ``S (M, n)``, over chunks of ``chunk`` query rows (default: bounded by
    ``LATENT_SUPPORT_CHUNK_ELEMS``)."""
    q = Q.reshape(-1, Q.shape[-1])
    if chunk is None:
        chunk = max(1, LATENT_SUPPORT_CHUNK_ELEMS // S.shape[0])
    # centred matmul expansion: cdist's direct mode cost ~750 ms per GPU solve (TF32 is off)
    c = S.mean(0)
    s = S - c
    s2 = (s * s).sum(-1)
    d = [((qc * qc).sum(-1, keepdim=True) + s2 - 2.0 * (qc @ s.T)).amin(-1).clamp_min(0.0).sqrt()
         for qc in (q[i:i + chunk] - c for i in range(0, q.shape[0], chunk))]
    return (d[0] if len(d) == 1 else torch.cat(d)).reshape(Q.shape[:-1])


def nn_dist_np(Q, S: np.ndarray) -> np.ndarray:
    """``(n,)`` float64 Euclidean distance of each row of ``Q (n, d)`` to its nearest row of
    ``S (M, d)``."""
    q = np.asarray(Q, dtype=np.float64).reshape(-1, S.shape[1])
    return np.sqrt(((q[:, None] - S[None]) ** 2).sum(-1).min(1))


# --- goal set -----------------------------------------------------------------------------------

def load_goal_set(path, n_x: int | None = None) -> np.ndarray:
    """``(K, n_x)`` float64 ``z_goals`` of a ``goal_set.py`` npz."""
    with np.load(Path(path), allow_pickle=False) as f:
        G = np.asarray(f["z_goals"], dtype=np.float64)
    if G.ndim != 2 or len(G) < 1 or (n_x is not None and G.shape[1] != n_x) \
            or not np.all(np.isfinite(G)):
        raise ValueError(f"goal set {path}: z_goals {G.shape} must be finite (K >= 1, {n_x})")
    return G


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


# --- learned value ------------------------------------------------------------------------------

def load_value(v: dict, n_x: int, z_std) -> LatentValue:
    """``MppiParams.value``'s network, refused unless trained on this latent (n_x, z_std)."""
    vf = LatentValue.load(v["path"])
    if vf.n_x != n_x:
        raise ValueError(f"value {v['path']}: n_x {vf.n_x} != deploy {n_x}")
    vf.check_z_std(z_std)
    return vf


def value_tensors(lcs, v: dict) -> tuple:
    return load_value(v, lcs.n_x, lcs.z_std_np).torch_params(lcs.tensor)
