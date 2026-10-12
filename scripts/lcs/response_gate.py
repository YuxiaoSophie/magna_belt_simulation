#!/usr/bin/env python3
"""V12-44 response-direction gate (report-only): median cosine between the one-step whitened
latent response to a 1 mm Franka +-x / +-y input and the whitened direction to the seated goal,
over recorded closed-loop states (every ``stride``-th frame from the MPC start, non-empty cloud).
Pass: F+x and F+y medians <= ``max_cos``. Same computation as
data/lcs/realfirst/cmp/encoder_study/sim44/diag/gate.py (CPU, float64).

    uv run --frozen python scripts/lcs/response_gate.py --deploy <deploy.npz> --episodes <npz>...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

CHANNELS = {"F+x": (0, 1.0), "F-x": (0, -1.0), "F+y": (1, 1.0), "F-y": (1, -1.0)}
GATED = ("F+x", "F+y")


def states(episodes, stride: int = 20) -> tuple[list, np.ndarray, int]:
    """Clouds and [p_franka, p_ur] of every ``stride``-th frame from the first MPC frame
    (finite stage-0 goal distance, stage >= 0); episodes without one are skipped."""
    clouds, P, n_ep = [], [], 0
    for ep in episodes:
        with np.load(ep, allow_pickle=True) as z:
            st, gd = np.asarray(z["sim_stage"]), np.asarray(z["sim_goal_dist"])
            on = np.flatnonzero(np.isfinite(gd[:, 0]) & (st >= 0))
            if not len(on):
                continue
            seg = z["pcd_belt_seg"]
            F, U = np.asarray(z["sim_ee_franka"])[:, :3], np.asarray(z["sim_ee_ur"])[:, :3]
            n_ep += 1
            for i in range(int(on[0]), len(F), stride):
                if len(seg[i]):
                    clouds.append(seg[i])
                    P.append(np.r_[F[i], U[i]])
    return clouds, np.asarray(P, np.float64).reshape(-1, 6), n_ep


def compute(deploy, episodes, stride: int = 20, max_cos: float = 0.3) -> dict:
    import torch

    from task_common.latent_encoder import LatentEncoder
    from task_common.mppi_lcs import TorchLcs

    clouds, P, n_ep = states(episodes, stride)
    out = {"deploy": str(deploy), "stride": stride, "max_cos": max_cos, "n_episodes": n_ep,
           "n_states": len(clouds)}
    if not clouds:
        return {**out, "pass": None, "channels": {}}
    enc = LatentEncoder.load(deploy)
    lcs = TorchLcs.from_deploy(deploy, "cpu", torch.float64)
    Z = torch.as_tensor(np.concatenate([enc.encode_batch(clouds[i:i + 256])
                                        for i in range(0, len(clouds), 256)]), dtype=torch.float64)
    Pt = torch.as_tensor(P, dtype=torch.float64)
    z_std = torch.as_tensor(lcs.z_std_np, dtype=torch.float64)
    seat = torch.as_tensor(lcs.stage_goals[-1], dtype=torch.float64)
    u0 = torch.zeros(len(Z), lcs.n_u, dtype=torch.float64)
    with torch.no_grad():
        base = lcs.step(Z, u0, Pt)[0]
        g = (seat - Z) / z_std
        ch = {}
        for name, (k, s) in CHANNELS.items():
            u = u0.clone()
            u[:, k] = 1e-3 * s
            r = (lcs.step(Z, u, Pt)[0] - base) / z_std
            c = torch.nn.functional.cosine_similarity(r, g, dim=-1).numpy()
            ch[name] = {"cos_median": float(np.median(c)), "frac_pos": float((c > 0).mean())}
    return {**out, "channels": ch,
            "pass": bool(all(ch[k]["cos_median"] <= max_cos for k in GATED))}


def line(res: dict) -> str:
    if not res.get("channels"):
        return f"no states ({res['n_episodes']} episodes)"
    return (("PASS" if res["pass"] else "FAIL") + f" ({res['n_states']} states): " + ", ".join(
        f"{k} {v['cos_median']:+.2f} / {v['frac_pos']:.0%}" for k, v in res["channels"].items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deploy", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-cos", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = compute(a.deploy, a.episodes, a.stride, a.max_cos)
    print(line(res))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
