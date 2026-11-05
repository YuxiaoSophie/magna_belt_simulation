"""One grasp-variant start state for the demo GUI: ``make_grasp_variants`` phase A (position pick)
then phase B (OSC settle on ``--lcm-url``) -> ``<out>/<id>_osc.npz`` + a row appended to
``<out>/index.json`` (``run_set``'s row shape). Exit 1 if not held (the row's ``reason`` says why).

usage: make_start.py --out <s>/starts --id <id> --franka-slide-mm F --franka-roll-deg F
                     --ur-slide-mm F --ur-roll-deg F [--franka-pivot-deg F] [--ur-pivot-deg F]
                     [--franka-depth-mm F] [--ur-depth-mm F]
                     --lcm-url <private url> [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "lcs"))
import make_grasp_variants as mgv

OPT_IN = ("franka_pivot_deg", "ur_pivot_deg", "franka_depth_mm", "ur_depth_mm")


def append_row(out: Path, row: dict) -> None:
    p = out / "index.json"
    idx = json.loads(p.read_text()) if p.is_file() else {
        "set": "demo_gui", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": str(mgv.MAGNA_PARAMS_SIM_YAML), "nominal_start": str(mgv.DEFAULT_REFERENCE),
        "variants": []}
    idx["variants"].append(row)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=2) + "\n")
    tmp.replace(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id", required=True)
    for k in (*mgv.KNOBS, *OPT_IN):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=0.0)
    ap.add_argument("--lcm-url", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args()
    v = mgv.Variant(a.id, **{k: getattr(a, k) for k in (*mgv.KNOBS, *OPT_IN)})
    if a.dry_run:
        print(json.dumps({"variant": dataclasses.asdict(v), "lcm_url": a.lcm_url,
                          "params": str(mgv.MAGNA_PARAMS_SIM_YAML),
                          "reference": str(mgv.DEFAULT_REFERENCE),
                          "file": str(a.out / f"{a.id}_osc.npz")}, indent=1))
        return 0
    mgv.configure_logging()
    a.out.mkdir(parents=True, exist_ok=True)
    snaps, failed = mgv.phase_a([v], mgv.MAGNA_PARAMS_SIM_YAML)
    row = mgv.phase_b([v], snaps, failed, a.lcm_url, mgv.DEFAULT_REFERENCE, a.out)[v.id]
    append_row(a.out, row)
    mgv.print_table([row])
    return 0 if row["held"] else 1


if __name__ == "__main__":
    sys.exit(main())
