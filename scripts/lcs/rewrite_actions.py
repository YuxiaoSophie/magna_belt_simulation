#!/usr/bin/env python3
"""Copy OSC-backend LCS episodes with ``actions`` recomputed under another action definition.

Recomputes ``actions`` from the recorded commands (``sim_cmd_knot0/1_franka``, ``sim_cmd_ur_t/t1``)
and the measured poses in ``state``, (re)writes the extras ``sim_action_knot1_minus_measured`` and
``sim_realised_delta``, and sets ``sim_meta.lcs_format.action_definition``; every other key is
copied unchanged and verified byte-identical after the write. ``--src`` is an episode ``.npz``
or a run dir (walked recursively; ``recordings/`` is skipped, each ``index.json`` is copied with
``action_definition`` updated, ``osc.log`` copied). Never writes in place.

Run:
    uv run python scripts/lcs/rewrite_actions.py --definition cmd_delta \
        --src data/lcs/20260923-210508-ep300-ou --out data/lcs/20260925-ep300-ou-cmd_delta
    uv run python scripts/lcs/rewrite_actions.py --src data/lcs/demo/demo_episode.npz \
        --out data/lcs/demo/demo_episode_cmd_delta.npz --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from task_common import lcs_dataset as lcs

CMD_KEYS = ("sim_cmd_knot0_franka", "sim_cmd_knot1_franka", "sim_cmd_ur_t", "sim_cmd_ur_t1")
REWRITTEN = ("actions", "sim_meta", "sim_action_knot1_minus_measured", "sim_realised_delta")
SKIP_DIRS = ("recordings",)
COPY_FILES = ("osc.log",)
ACTION_SOURCES = {
    "cmd_delta": "knot 1 - knot 0 of the Franka command published at t / UR line(t + knot dt) - "
                 "line(t) of the line in force (tracking frame)",
    "knot1_minus_measured": "knot 1 of the Franka command published at t / UR line at t + knot "
                            "dt (tracking frame), minus the measured pose at t",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def same_array(a: np.ndarray, b: np.ndarray) -> bool:
    """Byte-identical (dtype, shape, contents; object arrays element by element)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype != object:
        return a.tobytes() == b.tobytes()
    return all(same_array(np.asarray(x), np.asarray(y)) for x, y in zip(a.flat, b.flat))


def rewrite_arrays(data: dict, definition: str, provenance: dict) -> tuple[dict, dict]:
    """``(new payload, stats)``: ``data`` with the four ``REWRITTEN`` keys replaced/added."""
    missing = [k for k in (*CMD_KEYS, "state", "actions", "sim_meta") if k not in data]
    if missing:
        raise ValueError(f"missing {missing} (not an OSC-backend episode?)")
    state = np.asarray(data["state"], dtype=np.float64)
    cmd = [np.asarray(data[k], dtype=np.float64) for k in CMD_KEYS]
    meta = json.loads(str(data["sim_meta"]))
    old_def = meta.get("lcs_format", {}).get("action_definition")
    actions = lcs.command_actions(definition, state, *cmd)
    k1m = lcs.command_actions("knot1_minus_measured", state, *cmd)
    stats = {"T": len(state), "source_definition": old_def}
    if old_def in lcs.CMD_ACTION_DEFINITIONS:
        stats["source_actions_err"] = float(np.abs(
            lcs.command_actions(old_def, state, *cmd) - data["actions"]).max())
    meta.setdefault("lcs_format", {})["action_definition"] = definition
    meta["action_source"] = ACTION_SOURCES[definition]
    meta.setdefault("action_rewrites", []).append({**provenance, "from": old_def,
                                                   "to": definition})
    out = dict(data)
    out["actions"] = actions
    out["sim_meta"] = np.array(json.dumps(meta, sort_keys=True, default=lcs.json_default))
    out["sim_action_knot1_minus_measured"] = k1m
    out["sim_realised_delta"] = lcs.realised_delta(state)
    return out, stats


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def rewrite_file(src: Path, dst: Path, definition: str, dry_run: bool = False) -> dict:
    data = _load(src)
    prov = {"tool": "scripts/lcs/rewrite_actions.py", "src": str(src), "src_sha256": sha256(src)}
    payload, stats = rewrite_arrays(data, definition, prov)
    stats.update(src=str(src), dst=str(dst))
    if dry_run:
        return stats
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, dst)
    got = _load(dst)
    extra = sorted(set(got) - set(data) - set(REWRITTEN))
    lost = sorted(set(data) - set(got))
    if extra or lost:
        raise RuntimeError(f"{dst}: keys added {extra} / lost {lost}")
    diff = [k for k in data if k not in REWRITTEN and not same_array(data[k], got[k])]
    if diff:
        raise RuntimeError(f"{dst}: non-action keys changed: {diff}")
    summary = lcs.validate_episode(dst, period_us=None)
    if summary["action_definition"] != definition:
        raise RuntimeError(f"{dst}: action_definition {summary['action_definition']!r}")
    stats.update(keys_identical=len(data) - len(set(REWRITTEN) & set(data)),
                 validated=True, dst_sha256=sha256(dst))
    return stats


def plan(src: Path, out: Path) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """``(npz pairs, other file pairs)``; ``src``/``out`` are both files or both dirs."""
    if src.is_file():
        return [(src, out)], []
    npz, other = [], []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
            continue
        if path.suffix == ".npz":
            npz.append((path, out / rel))
        elif path.name == "index.json" or path.name in COPY_FILES:
            other.append((path, out / rel))
    return npz, other


def _check_paths(src: Path, out: Path) -> None:
    src, out = src.resolve(), out.resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if out == src or src in out.parents or out in src.parents:
        raise ValueError(f"--out {out} overlaps --src {src}: never in place")
    if src.is_file() != (out.suffix == ".npz"):
        raise ValueError("--src file needs an --out .npz; --src dir needs an --out dir")
    if out.exists() and (out.is_file() or any(out.iterdir())):
        raise FileExistsError(f"{out} exists and is not empty")


def _index(src: Path, dst: Path, definition: str, rows: list[dict]) -> None:
    index = json.loads(src.read_text())
    index["action_definition_source"] = index.get("action_definition")
    index["action_definition"] = definition
    index["action_rewrite"] = {"tool": "scripts/lcs/rewrite_actions.py", "src_dir": str(src.parent),
                               "src_index_sha256": sha256(src),
                               "recordings": f"not copied, see {src.parent / 'recordings'}"}
    by_src = {Path(r["src"]).name: r for r in rows if Path(r["src"]).parent == src.parent}
    for row in index.get("episodes", []):
        got = by_src.get(row.get("file") or "")
        if got is not None:
            row["size_bytes"] = Path(got["dst"]).stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(index, indent=2, default=lcs.json_default))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--definition", choices=lcs.CMD_ACTION_DEFINITIONS,
                   default=lcs.DEFAULT_ACTION_DEFINITION)
    p.add_argument("--src", type=Path, required=True, help="episode .npz or run dir")
    p.add_argument("--out", type=Path, required=True, help="new .npz or new dir (never in place)")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="recompute + report, write nothing")
    args = p.parse_args()
    _check_paths(args.src, args.out)
    npz, other = plan(args.src, args.out)
    if not npz:
        raise SystemExit(f"no .npz under {args.src}")
    print(f"{len(npz)} episode(s), {len(other)} other file(s): {args.src} -> {args.out}"
          f"{' (dry run)' if args.dry_run else ''}")
    work = [(s, d, args.definition, args.dry_run) for s, d in npz]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(rewrite_file, *zip(*work)))
    if not args.dry_run:
        for s, d in other:
            if s.name == "index.json":
                _index(s, d, args.definition, rows)
            else:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)
    err = [r["source_actions_err"] for r in rows if "source_actions_err" in r]
    print(f"source actions reproduced by their own definition: max err "
          f"{max(err) if err else float('nan'):.2e} over {len(err)} file(s)")
    if not args.dry_run:
        print(f"{len(rows)} written + validated, non-action keys byte-identical "
              f"({rows[0]['keys_identical']} keys in the first file)")
    u, r = [], []
    for s, d in npz:
        data = _load(d if not args.dry_run else s)
        if args.dry_run:
            data, _ = rewrite_arrays(data, args.definition, {})
        u.append(data["actions"])
        r.append(data["sim_realised_delta"])
    print(f"causality (realised vs {args.definition}):")
    print(lcs.causality_table(lcs.causality_slopes(np.vstack(u), np.vstack(r))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
