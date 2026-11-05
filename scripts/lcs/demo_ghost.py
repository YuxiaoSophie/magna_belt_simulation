#!/usr/bin/env python3
"""Align an end-to-end episode to its demos by belt shape (DTW) and write a demo ghost.

Stage-1 frames (``mpc:*`` phases) align to ``--demo1``, stage-2 frames (``s2:*``) to
``--demo2``, or both to the matching phases of one end-to-end ``--demo`` run (no jump at the
stage switch); UR-exit and post frames get no ghost. Prints how far the run is from the demo
(belt RMSE and EE distance along the alignment, durations, EE path lengths) and writes
``<out-dir>/<episode stem without episode_>.npz`` for ``record_replay_video.py --ghost``.

Run:
    uv run --frozen python scripts/lcs/demo_ghost.py --episode E.npz --demo1 D1.npz \
        --demo2 D2.npz --out-dir ghosts
    uv run --frozen python scripts/lcs/demo_ghost.py --episode E.npz --demo RUN.npz --out-dir g
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def dtw_path(cost: np.ndarray) -> list[tuple[int, int]]:
    """Endpoint-anchored DTW over a (n, m) cost; the (i, j) pairs from (0, 0) to (n-1, m-1)."""
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return path[::-1]


def belt_rmse_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, P, 3) x (m, P, 3) -> (n, m) per-point RMSE, mm (same body order)."""
    d = a[:, None] - b[None]
    return 1e3 * np.sqrt((d ** 2).sum(-1).mean(-1))


def stage_rows(d: dict, prefix: str) -> np.ndarray:
    """Frames of an end-to-end episode whose phase label starts with ``prefix``."""
    labels = json.loads(str(d["sim_meta"]))["phase_labels"]
    phase = np.array([labels[p] for p in d["sim_phase"]])
    return np.flatnonzero(np.char.startswith(phase, prefix) & (phase != "s2:done"))


def path_len_mm(xyz: np.ndarray) -> float:
    return float(1e3 * np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def align(ep: dict, rows: np.ndarray, demo: dict, label: str) -> tuple[np.ndarray, dict]:
    """Demo frame per episode row (median of its DTW matches) and the stage's numbers."""
    cost = belt_rmse_mm(ep["sim_belt_xyz"][rows].astype(np.float64),
                        demo["sim_belt_xyz"].astype(np.float64))
    match: dict[int, list[int]] = {}
    for i, j in dtw_path(cost):
        match.setdefault(i, []).append(j)
    idx = np.array([int(np.median(match[i])) for i in range(len(rows))])
    belt = cost[np.arange(len(rows)), idx]
    out = {"stage": label, "frames": [len(rows), len(demo["sim_time"])],
           "duration_s": [float(np.ptp(ep["sim_time"][rows])), float(np.ptp(demo["sim_time"]))],
           "belt_rmse_mm": {"start": belt[0], "mean": belt.mean(), "max": belt.max(),
                            "end": belt[-1]},
           "nearest_demo_belt_mm_mean": float(cost.min(axis=1).mean())}
    arms = ("franka", "ur") if label == "stage1" else ("franka",)
    for arm in arms:
        e, d = ep[f"sim_ee_{arm}"][rows, :3], demo[f"sim_ee_{arm}"][idx, :3]
        dist = 1e3 * np.linalg.norm(e - d, axis=1)
        out[f"ee_{arm}_mm"] = {"start": dist[0], "mean": dist.mean(), "max": dist.max(),
                               "end": dist[-1]}
        out[f"path_{arm}_mm"] = [path_len_mm(e), path_len_mm(demo[f"sim_ee_{arm}"][:, :3])]
    return idx, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--demo1", type=Path, help="stage-1 demo episode")
    parser.add_argument("--demo2", type=Path, help="stage-2 demo episode")
    parser.add_argument("--demo", type=Path, help="one end-to-end run for both stages")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.demo) == bool(args.demo1 and args.demo2):
        parser.error("give --demo, or both --demo1 and --demo2")

    ep = dict(np.load(args.episode, allow_pickle=True))
    n = len(ep["sim_phase"])
    ghost = {k: np.zeros((n, *ep[k].shape[1:]), np.float32)
             for k in ("sim_belt_xyz", "sim_ee_franka", "sim_ee_ur")}
    shown, demo_frame, report = np.zeros(n, bool), np.full(n, -1), []
    for label, prefix, path in (("stage1", "mpc:", args.demo or args.demo1),
                                ("stage2", "s2:", args.demo or args.demo2)):
        rows = stage_rows(ep, prefix)
        demo = dict(np.load(path, allow_pickle=True))
        if args.demo:
            keep = stage_rows(demo, prefix)
            demo = {k: demo[k][keep] for k in ("sim_time", "sim_belt_xyz", "sim_ee_franka",
                                               "sim_ee_ur")}
        idx, stats = align(ep, rows, demo, label)
        for k, arr in ghost.items():
            arr[rows] = demo[k][idx]
        shown[rows], demo_frame[rows] = True, idx
        report.append(stats | {"demo": str(path)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.episode.stem}.npz"
    np.savez(out, utime=ep["utime"], shown=shown, demo_frame=demo_frame,
             belt=ghost["sim_belt_xyz"], ee_franka=ghost["sim_ee_franka"],
             ee_ur=ghost["sim_ee_ur"], ee_franka_run=ep["sim_ee_franka"].astype(np.float32),
             ee_ur_run=ep["sim_ee_ur"].astype(np.float32), report=json.dumps(report))
    print(json.dumps(report, indent=1, default=lambda x: round(float(x), 1)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
