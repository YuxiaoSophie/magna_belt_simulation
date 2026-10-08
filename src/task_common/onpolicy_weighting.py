"""Task-driven frame weights for the J&P loop's on-policy refits (V12-36; opt-in).

Tube distance of a latent z: r(z) = min_j ||(z - z_traj[j]) / z_std|| / tol_j over the goal
demo's per-frame latents ``z_traj`` (latent_waypoints.npz), tol_j = the tol of the waypoint
nearest to demo frame j among those with one (``latent_waypoints.Waypoints.tol_at``). A rollout
goes off track at the first frame that starts ``streak`` consecutive frames with r > ``tol_k``;
frames from there + ``after_s`` onward get ``off_track_weight``; failed rollouts (``label_human``
False) are scaled by ``fail_weight``. Output ``frame_weights`` = [[start, stop, w], ...] over the
episode's frames, weight-1 segments omitted (read by lcs_learning ``refit_from_buffer.py``).
Non-finite latents never count as off track.
"""

from __future__ import annotations

import numpy as np

OFF_TRACK_DEFAULTS = {"tol_k": 3.0, "after_s": 1.0, "streak": 1}
DEFAULTS = {"off_track": OFF_TRACK_DEFAULTS, "off_track_weight": 0.2, "fail_weight": 1.0}


def rule_of(cfg: dict) -> dict:
    """``cfg`` (the loop's ``onpolicy_weighting``) with the defaults filled in."""
    return {**DEFAULTS, **cfg, "off_track": {**OFF_TRACK_DEFAULTS, **(cfg.get("off_track") or {})}}


def check_rule(cfg) -> str | None:
    """None if valid, else the reason."""
    if not isinstance(cfg, dict) or not set(cfg) <= set(DEFAULTS):
        return f"keys within {sorted(DEFAULTS)}"
    ot = cfg.get("off_track") or {}
    if not isinstance(ot, dict) or not set(ot) <= set(OFF_TRACK_DEFAULTS):
        return f"off_track keys within {sorted(OFF_TRACK_DEFAULTS)}"
    r = rule_of(cfg)
    o = r["off_track"]

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)

    if not (num(o["tol_k"]) and o["tol_k"] > 0 and num(o["after_s"]) and o["after_s"] >= 0
            and isinstance(o["streak"], int) and not isinstance(o["streak"], bool)
            and o["streak"] >= 1):
        return "off_track: tol_k > 0, after_s >= 0, int streak >= 1"
    if not (num(r["off_track_weight"]) and r["off_track_weight"] >= 0
            and num(r["fail_weight"]) and r["fail_weight"] > 0):
        return "off_track_weight >= 0, fail_weight > 0"
    return None


def frame_tols(frames, tol, T: int) -> np.ndarray:
    """(T,) tol of the waypoint nearest to each demo frame among those with tol >= 0."""
    frames, tol = np.asarray(frames, np.int64), np.asarray(tol, np.float64)
    ok = np.flatnonzero(tol >= 0.0)
    if not len(ok):
        raise ValueError("no waypoint with a tolerance")
    j = np.argmin(np.abs(frames[ok][None, :] - np.arange(T)[:, None]), axis=1)
    return tol[ok[j]]


def tube_ratio(z, z_traj, z_std, tol_frames) -> np.ndarray:
    """(n,) min over demo frames j of whitened ||z - z_traj[j]|| / tol_j (NaN rows -> NaN)."""
    z = np.asarray(z, np.float64)
    zt = np.asarray(z_traj, np.float64)
    zs = np.asarray(z_std, np.float64)
    tf = np.asarray(tol_frames, np.float64)
    out = np.full(len(z), np.nan)
    for i in range(0, len(z), 256):
        d = np.linalg.norm((z[i:i + 256, None, :] - zt[None]) / zs, axis=-1)
        out[i:i + 256] = (d / tf).min(axis=1)
    return out


def first_off_track(ratio, tol_k: float, streak: int = 1) -> int | None:
    """First frame starting ``streak`` consecutive frames with ratio > tol_k."""
    run = 0
    for t, r in enumerate(np.asarray(ratio, np.float64)):
        run = run + 1 if np.isfinite(r) and r > tol_k else 0
        if run >= streak:
            return t - streak + 1
    return None


def segments(n: int, start: int | None, off_w: float, scale: float = 1.0) -> list[list]:
    """[[start, stop, w], ...]: ``scale`` before ``start``, ``scale x off_w`` from it (None = no
    off-track part); weight-1 segments omitted."""
    cut = n if start is None else min(max(int(start), 0), n)
    segs = [[0, cut, float(scale)], [cut, n, float(scale * off_w)]]
    return [s for s in segs if s[1] > s[0] and s[2] != 1.0]


def expand(frame_weights, n: int) -> np.ndarray:
    w = np.ones(n)
    for a, b, x in frame_weights or []:
        w[a:b] = x
    return w


def effective_frames(row: dict, n: int) -> float:
    """Frame mass of an episode under its ``frame_weights`` (n without the field)."""
    return float(expand(row.get("frame_weights"), n).sum()) if row.get("frame_weights") else n


def episode_weights(z, wp: dict, dt: float, success: bool, rule: dict) -> tuple[list, dict]:
    """``(frame_weights, info)`` of one rollout: ``z`` (n, n_x) latents per frame, ``wp`` the
    latent_waypoints arrays (z_traj, z_std, frames, tol), ``dt`` s per frame."""
    r = rule_of(rule)
    o = r["off_track"]
    n = len(z)
    tf = frame_tols(wp["frames"], wp["tol"], len(wp["z_traj"]))
    ratio = tube_ratio(z, wp["z_traj"], wp["z_std"], tf)
    first = first_off_track(ratio, float(o["tol_k"]), int(o["streak"]))
    after = round(float(o["after_s"]) / dt)
    start = None if first is None or first + after >= n else first + after
    scale = 1.0 if success else float(r["fail_weight"])
    fw = segments(n, start, float(r["off_track_weight"]), scale)
    fin = ratio[np.isfinite(ratio)]
    info = {"n_frames": n, "first_off_track": first, "weighted_from": start,
            "n_down": 0 if start is None else n - start, "after_frames": after,
            "ratio_max": float(fin.max()) if len(fin) else None,
            "ratio_median": float(np.median(fin)) if len(fin) else None,
            "n_nonfinite": int(n - len(fin)), "scale": scale}
    return fw, info
