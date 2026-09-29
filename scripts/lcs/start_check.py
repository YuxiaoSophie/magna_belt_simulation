#!/usr/bin/env python3
"""Check each episode of a ``--start-states`` run starts from its own variant.

Prints the frame-0 belt RMSE of every ok episode to every variant of the run's set; exits 1 if
an episode is not closest to its own variant or its own RMSE exceeds ``--tol-mm``.

Run:
    uv run --frozen python scripts/lcs/start_check.py RUN [--tol-mm 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from task_common import lcs_dataset as lcs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--tol-mm", type=float, default=2.0)
    args = ap.parse_args()
    idx = json.loads((args.run / "index.json").read_text())
    states = idx["start_state"].get("states")
    if not states:
        print("no start-state set in index.json")
        return 1
    refs = {}
    for s in states:
        d = np.load(s["file"], allow_pickle=True)
        labels = json.loads(str(d["meta_json"]))["body_labels"]
        bi = [i for i, lab in enumerate(labels)
              if lab.startswith("flexible_ellipse_cable_edge_body_")]
        refs[s["id"]] = lcs.belt_points_ordered(d["body_q"][bi, :3].astype(np.float64))
    bad = 0
    for r in idx["episodes"]:
        if r["status"] != "ok":
            continue
        e = np.load(args.run / r["file"], allow_pickle=True)
        p0 = e["pcd_belt"][0].astype(np.float64)
        rm = {v: float(np.sqrt(((p0 - ref) ** 2).sum(1).mean()) * 1e3) for v, ref in refs.items()}
        own = r["start_variant"]
        good = min(rm, key=rm.get) == own and rm[own] <= args.tol_mm
        bad += not good
        meta = json.loads(str(e["sim_meta"]))
        print(f"{r['file']} {r['primitive']:<18} {own} {Path(meta['start_state']).name} own "
              f"{rm[own]:.2f} mm, others min "
              f"{min((v for k, v in rm.items() if k != own), default=float('nan')):.2f} mm"
              f"{'' if good else '  BAD'}")
    print(f"{'PASS' if not bad else 'FAIL'}: {bad} bad of "
          f"{sum(r['status'] == 'ok' for r in idx['episodes'])}")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
