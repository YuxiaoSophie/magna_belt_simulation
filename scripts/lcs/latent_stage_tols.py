#!/usr/bin/env python3
"""Stage-1 ("pre_place_2 reached") latent tolerance from the seed demos' own spread.

Encodes (numpy twin, ``pcd_belt_seg``) each demo's pre_place_2 latch frame (first frame after
the last ``move:pre_place_2``, ``realfirst_loop.goal_frames``) and the ``--window`` frames after
it; ``tol_1`` = max(p``--percentile`` of their whitened distance to ``z_goal_stage1``,
``--floor``). Same on the last ``--n-last`` frames vs ``z_goal_stage2`` -> ``tol_2`` (report
only: stage 2 ends on its timeout). The node fires "reached" after ``streak`` consecutive
latents with ``0 <= dist < tol`` (``MppiParams.stage_reach_streak``).

    uv run --frozen python scripts/lcs/latent_stage_tols.py --deploy <deploy.npz> \\
        --demos data/lcs/realfirst/v1_1/seed/demo/*.npz \\
        data/lcs/realfirst/v1_1/seed_ext/demo/*.npz \\
        --out data/lcs/realfirst/v1_1/stage_rule/stage_tols.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from task_common.latent_encoder import LatentEncoder, LearnedLcs

STREAK = 3
N_LAST = 13
SEG_KEY = "pcd_belt_seg"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latch_frame(labels: list[str]) -> int:
    """First frame after the last ``move:pre_place_2`` (``realfirst_loop.goal_frames``)."""
    mv = [t for t, lab in enumerate(labels) if lab == "move:pre_place_2"]
    if not mv or mv[-1] + 1 >= len(labels):
        raise ValueError("no frame after move:pre_place_2")
    return mv[-1] + 1


def load_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        if SEG_KEY not in d.files:
            raise SystemExit(f"{path}: no {SEG_KEY} (record with --belt-seg)")
        meta = json.loads(str(d["sim_meta"]))
        phase = np.asarray(d["sim_phase"], dtype=np.int64)
        out = {"seg": list(d[SEG_KEY]), "ee_f": np.asarray(d["sim_ee_franka"], np.float64),
               "ee_u": np.asarray(d["sim_ee_ur"], np.float64),
               "stage": np.asarray(d["sim_stage"]) if "sim_stage" in d.files else None,
               "outcome": str(d["sim_outcome"]) if "sim_outcome" in d.files else None,
               "goal_dist": (np.asarray(d["sim_goal_dist"], np.float64)
                             if "sim_goal_dist" in d.files else None)}
    out["labels"] = [meta["phase_labels"][p] for p in phase] if "phase_labels" in meta else None
    return out


def encode_frames(enc, seg: list, frames) -> tuple[np.ndarray, int]:
    """``(len(frames), n_x)`` latents (per-frame ``encode``, as the node) and the empty count."""
    zs, n_empty = [], 0
    for t in frames:
        zs.append(enc.encode(np.asarray(seg[t], dtype=np.float32)))
        n_empty += bool(enc.last_empty)
    return np.stack(zs), n_empty


def dists(lcs: LearnedLcs, Z: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.array([lcs.whitened_dist(z, goal) for z in Z])


def streak_fire(dist, tol: float, streak: int, start: int = 0) -> int | None:
    """First index ``>= start`` ending ``streak`` consecutive ``0 <= dist < tol`` (node rule)."""
    run = 0
    for t in range(start, len(dist)):
        run = run + 1 if 0.0 <= dist[t] < tol else 0
        if run >= streak:
            return t
    return None


def stage_goals(deploy: Path) -> tuple[LearnedLcs, np.ndarray, np.ndarray]:
    lcs = LearnedLcs.load(deploy)
    with np.load(deploy, allow_pickle=True) as d:
        g1, g2 = (np.asarray(d[k], np.float64) for k in ("z_goal_stage1", "z_goal_stage2"))
    return lcs, g1, g2


def derive(deploy: Path, demos: list[Path], window: int, floor: float, pct: float,
           n_last: int = N_LAST, streak: int = STREAK) -> dict:
    lcs, g1, g2 = stage_goals(deploy)
    enc = LatentEncoder.load(deploy)
    rows, d1_all, d2_all = [], [], []
    for path in demos:
        ep = load_episode(path)
        if ep["labels"] is None:
            raise SystemExit(f"{path}: sim_meta has no phase_labels")
        i = latch_frame(ep["labels"])
        n = len(ep["seg"])
        w = list(range(i, min(i + window + 1, n)))
        last = list(range(max(0, n - n_last), n))
        Z1, e1 = encode_frames(enc, ep["seg"], w)
        Z2, e2 = encode_frames(enc, ep["seg"], last)
        d1, d2 = dists(lcs, Z1, g1), dists(lcs, Z2, g2)
        d1_all += d1.tolist()
        d2_all += d2.tolist()
        rows.append({"episode": str(path), "latch": i, "n_frames": n, "outcome": ep["outcome"],
                     "d_latch": float(d1[0]), "d_window_min": float(d1.min()),
                     "d_window": np.round(d1, 4).tolist(), "d2_last": np.round(d2, 4).tolist(),
                     "n_empty": e1 + e2})
    p1, p2 = float(np.percentile(d1_all, pct)), float(np.percentile(d2_all, pct))
    tol = [max(p1, floor), max(p2, floor)]
    for r in rows:
        r["frames_inside_tol1"] = int(np.sum(np.asarray(r["d_window"]) < tol[0]))
        r["latch_inside_tol1"] = bool(r["d_latch"] < tol[0])
        r["streak_in_window"] = streak_fire(r["d_window"], tol[0], streak) is not None
    n_in = sum(r["frames_inside_tol1"] for r in rows)
    return {"tol": tol, "streak": streak, "percentile": pct, "floor": floor, "window": window,
            "n_last": n_last, "p_stage1": p1, "p_stage2": p2,
            "stage_goal_dist": float(lcs.whitened_dist(g2, g1)),
            "frac_window_frames_inside_tol1": n_in / len(d1_all),
            "n_latch_inside_tol1": sum(r["latch_inside_tol1"] for r in rows),
            "n_streak_in_window": sum(r["streak_in_window"] for r in rows),
            "n_demos": len(rows), "per_demo": rows, "deploy": str(deploy.resolve()),
            "deploy_sha": sha256(deploy), "script": "scripts/lcs/latent_stage_tols.py"}


def table(doc: dict) -> str:
    t1 = doc["tol"][0]
    head = (f"tol_1 {t1:.3f} (p{doc['percentile']:g} {doc['p_stage1']:.3f}, floor "
            f"{doc['floor']:g}), tol_2 {doc['tol'][1]:.3f} (report only); stage goals "
            f"{doc['stage_goal_dist']:.2f} apart; window frames inside tol_1 "
            f"{doc['frac_window_frames_inside_tol1']:.0%}, latch inside "
            f"{doc['n_latch_inside_tol1']}/{doc['n_demos']}, streak {doc['streak']} inside "
            f"the window {doc['n_streak_in_window']}/{doc['n_demos']}")
    lines = [head,
             "| demo | outcome | latch | d latch | min window | inside tol_1 | d2 last p50 |",
             "|---|---|---|---|---|---|---|"]
    for r in doc["per_demo"]:
        name = "/".join(Path(r["episode"]).parts[-3:])
        lines.append(f"| {name} | {r['outcome']} | {r['latch']} | {r['d_latch']:.3f} | "
                     f"{r['d_window_min']:.3f} | {r['frames_inside_tol1']}/"
                     f"{len(r['d_window'])} | {np.median(r['d2_last']):.3f} |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deploy", type=Path, required=True)
    p.add_argument("--demos", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True, help="stage_tols.json")
    p.add_argument("--window", type=int, default=6, help="frames after the latch")
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--percentile", type=float, default=90.0)
    a = p.parse_args()
    doc = derive(a.deploy, a.demos, a.window, a.floor, a.percentile)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    text = table(doc)
    a.out.with_suffix(".md").write_text(text + "\n")
    print(text)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
