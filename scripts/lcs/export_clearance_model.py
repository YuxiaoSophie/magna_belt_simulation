#!/usr/bin/env python3
"""Export the MPPI clearance model (``task_common.clearance_model``) from a sim snapshot.

Restores ``--start-state`` in an offline sim (no LCM), freezes the 2F-85 set in the UR tracking
frame and the Franka hand's mesh colliders in ``finger_tip``, and writes the board's plate +
pulleys with the thresholds. ``--franka-threshold DEMO.npz ...`` also derives the Franka
threshold from the demos' seated frames (``clearance_model.franka_threshold``) into
``--threshold-out``.

    uv run --frozen python scripts/lcs/export_clearance_model.py \\
        --start-state data/lcs/start_states/pre_place_1_osc.npz \\
        --out data/lcs/realfirst/v1_1/clearance/model_pp1.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from task_common import clearance_model as cm
from task_common import sim_snapshot


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-state", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ur-min-mm", type=float, default=cm.DEFAULT_UR_MIN_MM)
    p.add_argument("--franka-min-mm", type=float, default=cm.DEFAULT_FRANKA_MIN_MM)
    p.add_argument("--franka-threshold", type=Path, nargs="*", default=[], metavar="DEMO.npz",
                   help="derive the Franka threshold from these demos (report; --franka-min-mm "
                        "still sets the model's value)")
    p.add_argument("--threshold-out", type=Path, default=None)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"{args.out} exists (new files only)")

    from round_belt_task.offline_simulation import RoundBeltOfflineSimulation

    sim = RoundBeltOfflineSimulation.build()
    try:
        sim.restore(sim_snapshot.load(args.start_state))
        model = cm.export_from_sim(sim, args.ur_min_mm * 1e-3, args.franka_min_mm * 1e-3)
    finally:
        sim.close("clearance export")
    model.source.update(start_state=str(args.start_state.resolve()),
                        start_state_sha256=cm.sha256(args.start_state))
    X_track, X_tip = (np.asarray(model.source[k]) for k in ("X_W_track", "X_W_tip"))
    d_ur, d_f = cm.distance_numpy(model, X_track, X_tip)
    model.source["clearance_at_export_mm"] = {"ur": d_ur * 1e3, "franka": d_f * 1e3}
    path = model.save(args.out)
    print(f"[clearance] {path}: UR {len(model.ur_local)} points, Franka "
          f"{len(model.franka_local)} ({model.source['franka']}), {len(model.pulleys)} pulleys; "
          f"thresholds UR {model.ur_min_m * 1e3:g} / Franka {model.franka_min_m * 1e3:g} mm; at "
          f"the start UR {d_ur * 1e3:.2f} / Franka {d_f * 1e3:.2f} mm; FK vs model "
          f"{model.source['fk_vs_model_max_abs']}", flush=True)
    if args.franka_threshold:
        res = cm.franka_threshold(model, args.franka_threshold)
        res["model"] = str(path)
        out = args.threshold_out or path.with_name("franka_threshold.json")
        out.write_text(json.dumps(res, indent=1))
        print(f"[clearance] Franka threshold {res['franka_min_m'] * 1e3:.3f} mm (min d_f "
              f"{res['min_d_f_m'] * 1e3:.2f} mm over {len(args.franka_threshold)} demos) -> {out}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
