"""Per-channel input trust region from buffer actions (PLAN-20261005-realfirst-v1 §4, S5).

``compute(actions (N, 12), k, floor_frac, outer_lb, outer_ub)``: box = mean +- max(k sigma,
floor_frac x outer half-width) per channel, clipped to the outer (safety) box; a channel whose
clipped width falls under 2 x floor is re-centred inside the outer box (V8 ``trust_region()``).
Units: m / rad per 75 ms step (``cmd_delta`` actions); the table adds mm / deg display values.
``with_min_box`` (opt-in) widens a box to contain a zero-centred minimum box.
"""

from __future__ import annotations

import numpy as np

CHANNELS = ("f_x", "f_y", "f_z", "ur_x", "ur_y", "ur_z", "fr_x", "fr_y", "fr_z", "urr_x",
            "urr_y", "urr_z")
DISPLAY_SCALE = np.array([1e3] * 6 + [np.degrees(1.0)] * 6)
DISPLAY_UNIT = ["mm"] * 6 + ["deg"] * 6


def outer_box(trans_m: float, rot_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-step safety box (translations first, then rotations)."""
    half = np.array([trans_m] * 6 + [np.radians(rot_deg)] * 6, dtype=np.float64)
    return -half, half


def with_min_box(lb, ub, outer_lb, outer_ub, frac: float) -> dict:
    """Planner box (v1.1, opt-in): hull of [lb, ub] and the zero-centred [-frac h, +frac h]
    (h = outer half-width), intersected with the outer box, so every channel can reverse."""
    lb, ub = np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)
    lo, hi = np.asarray(outer_lb, dtype=np.float64), np.asarray(outer_ub, dtype=np.float64)
    m = frac * (hi - lo) / 2
    plb, pub = np.maximum(np.minimum(lb, -m), lo), np.minimum(np.maximum(ub, m), hi)
    assert np.all(plb < pub)
    return {"u_lb": plb.tolist(), "u_ub": pub.tolist(), "min_frac": float(frac),
            "min_half": m.tolist(), "widened": ((plb < lb) | (pub > ub)).tolist(),
            "reversible": ((plb < 0) & (pub > 0)).tolist(),
            "disp": [{"channel": c, "unit": u, "lb": float(a * s), "ub": float(b * s)}
                     for c, u, a, b, s in zip(CHANNELS, DISPLAY_UNIT, plb, pub, DISPLAY_SCALE)]}


def _gm(x: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.maximum(x, 1e-12)))))


def compute(actions, k: float, floor_frac: float, outer_lb, outer_ub) -> dict:
    """``{u_lb, u_ub, channels, ratios, n_floor_hits, n_clip_hits, ...}`` (JSON-ready)."""
    u = np.asarray(actions, dtype=np.float64)
    lo, hi = np.asarray(outer_lb, dtype=np.float64), np.asarray(outer_ub, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != len(lo) or len(u) < 2:
        raise ValueError(f"actions {u.shape}: want (N >= 2, {len(lo)})")
    if not np.all(lo < hi):
        raise ValueError("outer box needs outer_lb < outer_ub")
    half = (hi - lo) / 2
    mean, std = u.mean(0), u.std(0)
    floor = floor_frac * half
    delta = np.maximum(k * std, floor)
    lb_raw, ub_raw = mean - delta, mean + delta
    lb, ub = np.maximum(lb_raw, lo), np.minimum(ub_raw, hi)
    w = 2 * floor  # keeps a positive width when the mean sits near / outside the outer box
    recenter = ub - lb < w
    lb = np.where(recenter, np.clip(np.minimum(lb, hi - w), lo, None), lb)
    ub = np.where(recenter, np.clip(np.maximum(ub, lo + w), None, hi), ub)
    assert np.all(lb < ub)
    ch = []
    for j, name in enumerate(CHANNELS[:len(lo)]):
        sc = DISPLAY_SCALE[j]
        ch.append({
            "channel": name, "unit": DISPLAY_UNIT[j], "mean": mean[j], "std": std[j],
            "delta": delta[j], "lb_raw": lb_raw[j], "ub_raw": ub_raw[j], "lb": lb[j],
            "ub": ub[j], "outer_lb": lo[j], "outer_ub": hi[j],
            "floor_hit": bool(k * std[j] < floor[j]), "clip_lb_hit": bool(lb_raw[j] < lo[j]),
            "clip_ub_hit": bool(ub_raw[j] > hi[j]), "recenter_hit": bool(recenter[j]),
            "delta_over_half": delta[j] / half[j],
            "box_over_outer": (ub[j] - lb[j]) / (2 * half[j]),
            "disp": {n: float(v * sc) for n, v in (("mean", mean[j]), ("std", std[j]),
                                                   ("delta", delta[j]), ("lb", lb[j]),
                                                   ("ub", ub[j]))}})
    r, b = delta / half, (ub - lb) / (2 * half)
    return {
        "k": float(k), "floor_frac": float(floor_frac), "n_frames": len(u),
        "units": "SI: m / rad per 75 ms step; disp: mm / deg",
        "u_lb": lb.tolist(), "u_ub": ub.tolist(), "outer_lb": lo.tolist(), "outer_ub": hi.tolist(),
        "channels": ch,
        "ratios": {"delta_trans_gm": _gm(r[:6]), "delta_rot_gm": _gm(r[6:]),
                   "box_trans_gm": _gm(b[:6]), "box_rot_gm": _gm(b[6:]),
                   "min_delta_over_half": float(r.min()),
                   "min_channel": CHANNELS[int(np.argmin(r))]},
        "n_floor_hits": int(sum(c["floor_hit"] for c in ch)),
        "n_clip_hits": int(sum(c["clip_lb_hit"] or c["clip_ub_hit"] for c in ch)),
        "all_at_floor": bool(all(c["floor_hit"] for c in ch)),
    }
