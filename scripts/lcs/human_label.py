#!/usr/bin/env python3
"""Emulated end-of-episode human success label (one bit per episode) from sim ground truth.

Stands in for the operator who looks at the belt after a rollout and says "seated" or not. This
is the ONLY place sim ground truth (``sim_belt_xyz`` + ``sim_pulley_large_pose`` through
``outcome.frame_metrics``) becomes a training-visible quantity, and only as this single bit per
episode from its final frames (plan §2: ``label_human``); nothing per frame leaves this module.

Rules (majority = more than half of the last ``last_n`` frames, 13 ~ 1 s at 75 ms):
    engaged  ``outcome.classify == "engaged"`` (wrap >= 60 deg in the groove); the default
             (user decision Q1, 2026-10-05)
    strict   wrap >= 90 deg and |h_median| <= 1 mm

    uv run --frozen python scripts/lcs/human_label.py \\
        --episodes data/lcs/approach/v1_heldout/episode_00*.npz \\
        --out data/lcs/realfirst/v1/perception_gate/labels_heldout.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from round_belt_task import outcome as oc

RULES = ("engaged", "strict")
DEFAULT_RULE = "engaged"
LAST_N = 13
SOURCE = "sim_ground_truth_final_frames"
STRICT_WRAP_DEG, STRICT_H_MM = 90.0, 1.0


def frame_passes(belt_xyz, pose_xyz_wxyz, rule: str) -> tuple[bool, oc.FrameMetrics]:
    """One frame under ``rule`` (``sim_pulley_large_pose`` is xyz_wxyz; outcome wants xyzw)."""
    p = np.asarray(pose_xyz_wxyz, dtype=np.float64)
    fm = oc.frame_metrics(np.asarray(belt_xyz, dtype=np.float64),
                          np.concatenate([p[:3], p[4:7], p[3:4]]))
    if rule == "engaged":
        return oc.classify(fm) == "engaged", fm
    return bool(fm.wrap_deg >= STRICT_WRAP_DEG and abs(fm.h_median_mm) <= STRICT_H_MM), fm


def emulated_human_label(episode_npz, rule: str = DEFAULT_RULE, last_n: int = LAST_N) -> dict:
    """``{success, rule, source, wrap_deg, h_mm, frames_used, n_pass, last_n}``; ``wrap_deg`` /
    ``h_mm`` = medians over the frames used. ``episode_npz``: a path or a loaded mapping."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r} not in {RULES}")
    if isinstance(episode_npz, (str, Path)):
        with np.load(episode_npz, allow_pickle=True) as d:
            belt, pose = d["sim_belt_xyz"], d["sim_pulley_large_pose"]
    else:
        belt, pose = episode_npz["sim_belt_xyz"], episode_npz["sim_pulley_large_pose"]
    T = len(belt)
    if T == 0:
        raise ValueError("emulated_human_label: empty episode")
    passes, wraps, hs = [], [], []
    for t in range(max(0, T - max(1, last_n)), T):
        ok, fm = frame_passes(belt[t], pose[t], rule)
        passes.append(ok)
        wraps.append(fm.wrap_deg)
        hs.append(fm.h_median_mm)
    h = np.asarray(hs, dtype=np.float64)
    return {"success": bool(2 * sum(passes) > len(passes)), "rule": rule, "source": SOURCE,
            "wrap_deg": float(np.median(wraps)),
            "h_mm": float(np.median(h[np.isfinite(h)])) if np.isfinite(h).any() else None,
            "frames_used": len(passes), "n_pass": int(sum(passes)), "last_n": last_n}


def _reference(path: Path) -> dict:
    """Recorded episode outcome and the final-frame strict test (for the agreement check)."""
    with np.load(path, allow_pickle=True) as d:
        outcome = str(d["sim_outcome"]) if "sim_outcome" in d.files else None
        wrap = float(d["sim_wrap_deg"][-1]) if "sim_wrap_deg" in d.files else None
        h = float(d["sim_h_median_mm"][-1]) if "sim_h_median_mm" in d.files else None
    index_outcome = None
    index = path.parent / "index.json"
    if index.is_file():
        rows = json.loads(index.read_text()).get("episodes", [])
        index_outcome = next((r.get("outcome") for r in rows if r.get("file") == path.name),
                             None)
    final_strict = (None if wrap is None or h is None
                    else bool(wrap >= STRICT_WRAP_DEG and abs(h) <= STRICT_H_MM))
    return {"sim_outcome": outcome, "index_outcome": index_outcome, "final_wrap_deg": wrap,
            "final_h_mm": h, "final_strict": final_strict}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes", type=Path, nargs="+", required=True)
    p.add_argument("--rule", choices=RULES, default=DEFAULT_RULE)
    p.add_argument("--last-n", type=int, default=LAST_N)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"--out {args.out} exists; refusing to overwrite")

    rows = []
    for path in sorted(args.episodes):
        labels = {r: emulated_human_label(path, r, args.last_n) for r in RULES}
        ref = _reference(path)
        rows.append({"episode": str(path), **labels[args.rule], "labels": labels,
                     "reference": ref,
                     "agree_engaged": (None if ref["sim_outcome"] is None else
                                       labels["engaged"]["success"]
                                       == (ref["sim_outcome"] == "engaged")),
                     "agree_strict": (None if ref["final_strict"] is None else
                                      labels["strict"]["success"] == ref["final_strict"])})
    summary = {}
    for rule, ref_name in (("engaged", "sim_outcome == engaged"),
                           ("strict", "final frame wrap >= 90 and |h| <= 1")):
        known = [r for r in rows if r[f"agree_{rule}"] is not None]
        summary[rule] = {"reference": ref_name, "agree": sum(r[f"agree_{rule}"] for r in known),
                         "n": len(known), "successes": sum(r["labels"][rule]["success"]
                                                           for r in rows),
                         "disagree": [{"episode": r["episode"],
                                       "label": r["labels"][rule]["success"],
                                       "n_pass": r["labels"][rule]["n_pass"],
                                       **r["reference"]}
                                      for r in known if not r[f"agree_{rule}"]]}
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": args.rule,
           "last_n": args.last_n, "source": SOURCE, "script": "scripts/lcs/human_label.py",
           "summary": summary, "episodes": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for rule, s in summary.items():
        print(f"[LABEL] {rule}: {s['successes']}/{len(rows)} success; agrees with "
              f"{s['reference']} on {s['agree']}/{s['n']}")
        for dis in s["disagree"]:
            print(f"  differs: {dis}")
    print(f"[LABEL] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
