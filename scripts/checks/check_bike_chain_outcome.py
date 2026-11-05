"""Synthetic CPU check of ``bike_chain_task.outcome`` (no sim).

Run:
    uv run --frozen python scripts/checks/check_bike_chain_outcome.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bike_chain_task.outcome import (
    GEOM,
    classify_sprocket,
    classify_sprocket_episode,
    rollers_from_inner_bodies,
    sprocket_metrics,
    success_both,
)
from task_common.replay_metrics import quat_rotate


def pose(centre=(0.0, 0.0, 0.0), yaw_deg=0.0):
    h = math.radians(yaw_deg) / 2
    return np.array([*centre, 0.0, 0.0, math.sin(h), math.cos(h)])


def arc(which, p, n=10, shift_gaps=0.5, dz_mm=0.0, ramp_mm=0.0, start_deg=0.0):
    """n rollers on the pitch circle one tooth pitch apart, at gap centres (+ shift), world frame."""
    g = GEOM[which]
    tp = 360.0 / g["teeth"]
    a = np.radians(g["az0_deg"] + start_deg + tp * (np.arange(n) + shift_gaps))
    z = 1e-3 * (g["tooth_z_mm"] + dz_mm + ramp_mm * (np.arange(n) / (n - 1) - 0.5))
    r = 1e-3 * g["pitch_radius_mm"]
    local = np.stack([r * np.cos(a), r * np.sin(a), z], axis=1)
    return p[:3] + quat_rotate(p[3:7], local)


def main() -> int:
    ok_all, p0 = True, pose()

    def rep(name, ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}")

    ring = 0.097 * np.stack([np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False)),
                             np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False)),
                             np.zeros(64)], axis=1) + [0.3, 0.3, 0.0]
    lab, _ = classify_sprocket(ring, p0, "small")
    rep("O1", lab == "outside", lab)

    c = arc("small", p0)
    lab, fr = classify_sprocket(c, p0, "small")
    wrap = sprocket_metrics(c, p0, "small").wrap_deg
    rep("O2", lab == "engaged" and fr == 1.0, f"{lab} arc={wrap:.0f} meshed={fr:.2f}")

    lab, fr = classify_sprocket(arc("small", p0, shift_gaps=0.0), p0, "small")
    rep("O3", lab == "slanted" and fr == 0.0, f"{lab} meshed={fr:.2f}")

    up, _ = classify_sprocket(arc("small", p0, dz_mm=8), p0, "small")
    dn, _ = classify_sprocket(arc("small", p0, dz_mm=-8), p0, "small")
    rep("O4", up == "over" and dn == "under", f"+8mm={up} -8mm={dn}")

    lab, _ = classify_sprocket(arc("small", p0, ramp_mm=10), p0, "small")
    rep("O5", lab == "slanted", lab)

    pr = pose((0.05, -0.02, 0.01), 37.0)
    lab, fr = classify_sprocket(arc("small", pr), pr, "small")
    rep("O6", lab == "engaged" and fr == 1.0, f"{lab} meshed={fr:.2f}")

    pl = pose((0.18, 0.0, 0.0))
    both = np.vstack([arc("small", p0, start_deg=90), arc("large", pl, n=20, start_deg=-100)])
    only = np.vstack([arc("small", p0, start_deg=90), ring])
    t1, t2 = success_both(both, p0, pl), success_both(only, p0, pl)
    rep("O7", t1 and not t2, f"both={t1} small_only={t2}")

    seq = [arc("small", p0, dz_mm=8), ring, arc("small", p0, shift_gaps=0.0), c, c, c]
    lab = classify_sprocket_episode(seq[1:], [p0] * 5, "small", last_n=3)
    rep("O8", lab == "engaged", lab)

    # Ground truth: inner bodies placed with pins on the small pitch circle -> roller centres.
    inner = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    rl = rollers_from_inner_bodies(inner)
    rep("O9", np.allclose(rl, [[0, 0, 0.0035], [0, 0.009525, 0.0035]]), "roller offsets")

    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
