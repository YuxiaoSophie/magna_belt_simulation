#!/usr/bin/env python3
"""Backfill material-sampled ``pcd_belt`` and the slant metrics into collected LCS episodes.

For every ``ok`` row of each ``index.json`` under the given dirs: recompute ``pcd_belt`` from the
stored ``sim_belt_xyz`` with ``lcs_dataset.belt_points_ordered`` (material sampling), set
``sim_meta.lcs_format.belt_sampling = "material"``, add ``sim_slant_*`` from ``sim_belt_xyz`` +
``sim_pulley_large_pose`` and the run's ``belt_tangent``, and add ``final_slant_*`` to the
index row. Every other array is kept bit-identical (checked after each write). Files already
marked ``material`` are skipped. Writes are atomic (temp file + rename).

Run:
    uv run python scripts/lcs/rewrite_belt_points.py data/lcs/<run> --dry-run
    uv run python scripts/lcs/rewrite_belt_points.py data/lcs/<run>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from round_belt_task.outcome import OutcomeThresholds, classify_episode, slant_episode
from task_common import lcs_dataset as lcs

SLANT_NPZ = ("slant_deg", "slant_axis_deg", "slant_dir", "slant_deg_t", "slant_axis_deg_t")
REWRITTEN = ("pcd_belt", "sim_meta", *(lcs.EXTRA_PREFIX + k for k in SLANT_NPZ))


def material_coord(bodies: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``i + t`` (body units) of each point's nearest projection on the closed body loop."""
    loop = np.vstack([bodies, bodies[:1]]).astype(np.float64)
    seg = np.diff(loop, axis=0)
    ln2 = np.sum(seg * seg, axis=1)
    pts = np.asarray(pts, dtype=np.float64)
    best_s, best_d = np.zeros(len(pts)), np.full(len(pts), np.inf)
    for i in range(len(seg)):
        t = np.clip(((pts - loop[i]) @ seg[i]) / ln2[i], 0.0, 1.0)
        d = np.linalg.norm(pts - (loop[i] + t[:, None] * seg[i]), axis=1)
        better = d < best_d
        best_d[better], best_s[better] = d[better], i + t[better]
    return best_s


def material_drift(belt_T: np.ndarray, pts_T: np.ndarray) -> float:
    """Max over frames/points of the material-coordinate change since frame 0 (body units)."""
    m = belt_T.shape[1]
    s0 = material_coord(belt_T[0], pts_T[0])
    out = 0.0
    for b, p in zip(belt_T, pts_T, strict=True):
        ds = (material_coord(b, p) - s0 + m / 2) % m - m / 2
        out = max(out, float(np.abs(ds).max()))
    return out


def wxyz_to_xyzw(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64)
    return np.concatenate([pose7[..., :3], pose7[..., 4:7], pose7[..., 3:4]], axis=-1)


def same_array(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == object:
        return all(same_array(np.asarray(x), np.asarray(y))
                   for x, y in zip(a.ravel(), b.ravel(), strict=True))
    return a.tobytes() == b.tobytes()


def _write_npz(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lcs.json_default) + "\n")
    os.replace(tmp, path)


def rewrite_episode(path: Path, tangent, thresholds: OutcomeThresholds, dry_run: bool) -> dict:
    with np.load(path, allow_pickle=True) as data:
        orig = {k: data[k] for k in data.files}
    meta = json.loads(str(orig["sim_meta"]))
    done = meta.get("lcs_format", {}).get("belt_sampling") == "material"
    belt = orig["sim_belt_xyz"].astype(np.float64)
    pulley = wxyz_to_xyzw(orig["sim_pulley_large_pose"])
    label = str(orig["trajectory_label"])
    stats = {"file": path, "label": label, "skipped": done}
    if done:
        stats["slant"] = {k: orig[lcs.EXTRA_PREFIX + k] for k in SLANT_NPZ}
        return stats

    relabel, _ = classify_episode(belt, pulley, thresholds)
    new_belt = np.stack([lcs.belt_points_ordered(b) for b in belt])
    slant = slant_episode(belt, pulley, np.asarray(tangent, dtype=np.float64), thresholds)
    stats.update(
        relabel=relabel, slant=slant,
        drift_before=material_drift(belt, orig["pcd_belt"]),
        drift_after=material_drift(belt, new_belt),
        max_disp_mm=float(np.linalg.norm(new_belt.astype(np.float64)
                                         - orig["pcd_belt"].astype(np.float64), axis=2).max()
                          * 1e3))
    if dry_run:
        return stats

    meta.setdefault("lcs_format", {})["belt_sampling"] = "material"
    payload = dict(orig)
    payload["pcd_belt"] = new_belt
    payload["sim_meta"] = np.array(json.dumps(meta, sort_keys=True, default=lcs.json_default))
    for k in SLANT_NPZ:
        payload[lcs.EXTRA_PREFIX + k] = np.array(slant[k])
    _write_npz(path, payload)

    with np.load(path, allow_pickle=True) as data:
        back = {k: data[k] for k in data.files}
    bad = [k for k in orig if k not in REWRITTEN and not same_array(orig[k], back[k])]
    if bad or not np.array_equal(back["pcd_belt"], new_belt):
        raise RuntimeError(f"{path}: arrays changed on rewrite: {bad}")
    old_meta = json.loads(str(orig["sim_meta"]))
    new_meta = json.loads(str(back["sim_meta"]))
    new_meta["lcs_format"].pop("belt_sampling")
    if new_meta != old_meta:
        raise RuntimeError(f"{path}: sim_meta changed beyond belt_sampling")
    summary = lcs.validate_episode(path)
    if summary["belt_sampling"] != "material":
        raise RuntimeError(f"{path}: belt_sampling not recorded")
    return stats


def _finite_or_none(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


def rewrite_run(run_dir: Path, dry_run: bool) -> list[dict]:
    index_path = run_dir / "index.json"
    text = index_path.read_text()
    index = json.loads(text)
    if json.dumps(index, indent=1, default=lcs.json_default) + "\n" != text:
        raise RuntimeError(f"{index_path}: does not round-trip; refusing to rewrite it")
    thresholds = OutcomeThresholds(**index["thresholds"])
    tangent = index["belt_tangent"]
    out = []
    for row in index["episodes"]:
        if row.get("status") != "ok" or not row.get("file"):
            continue
        stats = rewrite_episode(run_dir / row["file"], tangent, thresholds, dry_run)
        s = stats["slant"]
        row["final_slant_deg"] = _finite_or_none(s["slant_deg"])
        row["final_slant_axis_deg"] = _finite_or_none(s["slant_axis_deg"])
        row["final_slant_dir"] = str(s["slant_dir"])
        out.append(stats)
    if not dry_run:
        _write_json(index_path, index)
    return out


def _find_runs(roots: list[Path]) -> list[Path]:
    runs = []
    for root in roots:
        runs += sorted(p.parent for p in root.rglob("index.json"))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+", type=Path, help="run dirs (searched for index.json)")
    parser.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    args = parser.parse_args()

    runs = _find_runs(args.dirs)
    if not runs:
        print("no index.json found", file=sys.stderr)
        return 1
    rows = []
    for run in runs:
        got = rewrite_run(run, args.dry_run)
        n_skip = sum(r["skipped"] for r in got)
        print(f"{run}: {len(got)} episodes, {n_skip} already material"
              f"{' (dry run)' if args.dry_run else ''}")
        rows += got

    per: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per[r["label"]].append(r)
    done = [r for r in rows if not r["skipped"]]
    if done:
        print(f"{'label':<9}{'n':>4}  drift before max/median  after max  |new-old| max mm")
        for label, rs in sorted(per.items()):
            rs = [r for r in rs if not r["skipped"]]
            if not rs:
                continue
            before = np.array([r["drift_before"] for r in rs])
            after = np.array([r["drift_after"] for r in rs])
            disp = max(r["max_disp_mm"] for r in rs)
            print(f"{label:<9}{len(rs):>4}  {before.max():10.3f} / {np.median(before):6.3f}"
                  f"  {after.max():9.2e}  {disp:8.3f}")
        mismatch = [r for r in done if r["relabel"] != r["label"]]
        print(f"re-classified from stored arrays: {len(done) - len(mismatch)}/{len(done)} "
              f"match trajectory_label"
              + "".join(f"\n  {r['file']}: {r['label']} -> {r['relabel']}" for r in mismatch))
    print(f"{'label':<9}{'n':>4}  slant_deg median [p5, p95] max   slant_dir counts")
    for label, rs in sorted(per.items()):
        sd = np.array([float(r["slant"]["slant_deg"]) for r in rs])
        ok = sd[np.isfinite(sd)]
        dirs = defaultdict(int)
        for r in rs:
            dirs[str(r["slant"]["slant_dir"])] += 1
        stats = (f"{np.median(ok):6.2f} [{np.percentile(ok, 5):5.2f}, "
                 f"{np.percentile(ok, 95):5.2f}] {ok.max():6.2f}" if len(ok) else "   n/a")
        print(f"{label:<9}{len(rs):>4}  {stats}   {dict(sorted(dirs.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
