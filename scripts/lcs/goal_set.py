#!/usr/bin/env python3
"""Seated goal set for MPPI's last stage (``MppiParams.goal_set``): the encoded final belt cloud
(``pcd_belt_seg`` of the last frame, numpy-twin ``encode`` as the encoder node and
``latent_waypoints.py``'s seated waypoint) of every seated (``label_human``; ``label_strict``
with ``--strict``) buffer row of ``--kinds`` (every demo with ``--all-demos``) plus, with
``--include-successes``, every seated MPC row. Belt observations only; no EE poses. Rows are deduplicated by episode
path, in buffer order. Deterministic: rerun -> the same ``z_goals``.

    uv run --frozen python scripts/lcs/goal_set.py --deploy <deploy.npz> \\
        --buffer data/lcs/realfirst/cmp/v12_18/seed_buffer_merged.json --kinds demo \\
        --out <goal_set.npz>
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

from task_common.latent_encoder import LatentEncoder

SEG_KEY = "pcd_belt_seg"
VERSION = 1


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def final_cloud(episode) -> np.ndarray:
    with np.load(episode, allow_pickle=True) as d:
        if SEG_KEY not in d.files:
            raise SystemExit(f"{episode}: no {SEG_KEY} (record with --belt-seg)")
        return np.asarray(d[SEG_KEY][-1], dtype=np.float32)


def final_latent(enc, episode) -> np.ndarray:
    """``(n_x,)`` the node's latent of the episode's last frame."""
    z = enc.encode(final_cloud(episode))
    if getattr(enc, "last_empty", False):
        raise SystemExit(f"{episode}: empty final belt cloud")
    return z


def select_rows(rows: list[dict], kinds, include_successes: bool, all_demos: bool,
                strict: bool = False) -> list[dict]:
    """Seated rows (``label_human``, ``label_strict`` with ``strict``; demos all with
    ``all_demos``) of ``kinds`` + seated MPC rows."""
    key = "label_strict" if strict else "label_human"
    out, seen = [], set()
    for r in rows:
        take = r["kind"] in kinds and (bool(r.get(key)) or (r["kind"] == "demo" and all_demos))
        take = take or (include_successes and r["kind"] == "mpc" and bool(r.get(key)))
        if take and str(r["episode"]) not in seen:
            seen.add(str(r["episode"]))
            out.append(r)
    return out


def build(deploy, rows: list[dict], kinds=("demo",), include_successes: bool = False,
          all_demos: bool = False, strict: bool = False) -> dict:
    sel = select_rows(rows, list(kinds), include_successes, all_demos, strict)
    if not sel:
        raise SystemExit(f"goal set: no rows of {list(kinds)} (successes {include_successes})")
    enc = LatentEncoder.load(deploy)
    Z = np.stack([final_latent(enc, r["episode"]) for r in sel])
    with np.load(deploy, allow_pickle=False) as d:
        z_std = np.asarray(d["z_std"], dtype=np.float64)
    if Z.shape[1] != z_std.shape[0]:
        raise SystemExit(f"goal set dim {Z.shape[1]} != deploy n_x {z_std.shape[0]}")
    return {"z_goals": Z, "episodes": [str(r["episode"]) for r in sel],
            "kinds": [r["kind"] for r in sel],
            "label_human": [bool(r.get("label_human")) for r in sel], "z_std": z_std,
            "deploy": str(deploy), "deploy_sha256": sha256(deploy),
            "rule": {"kinds": list(kinds), "include_successes": bool(include_successes),
                     "all_demos": bool(all_demos), "strict": bool(strict)}}


def save(path, gs: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, z_goals=gs["z_goals"], z_std=gs["z_std"], version=VERSION,
             episodes=np.asarray(gs["episodes"]), kinds=np.asarray(gs["kinds"]),
             label_human=np.asarray(gs["label_human"]), deploy=gs["deploy"],
             deploy_sha256=gs["deploy_sha256"], rule=json.dumps(gs["rule"]))


def pairwise(gs: dict) -> np.ndarray:
    """``(K, K)`` whitened distances between the set's latents."""
    Z = gs["z_goals"] / gs["z_std"]
    return np.linalg.norm(Z[:, None] - Z[None], axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--buffer", type=Path, nargs="+", required=True, help="buffer.json manifests")
    ap.add_argument("--kinds", nargs="+", default=["demo"])
    ap.add_argument("--include-successes", action="store_true",
                    help="also the MPC rows with label_human")
    ap.add_argument("--all-demos", action="store_true",
                    help="demo rows whatever label_human (default: seated demos only)")
    ap.add_argument("--strict", action="store_true",
                    help="seated = label_strict (wrap >= 90 deg, |h| <= 1 mm) instead of label_human")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [r for b in a.buffer for r in json.loads(b.read_text())["rollouts"]]
    gs = build(a.deploy, rows, a.kinds, a.include_successes, a.all_demos, a.strict)
    save(a.out, gs)
    D = pairwise(gs)
    off = D[np.triu_indices(len(D), 1)]
    n = {k: gs["kinds"].count(k) for k in dict.fromkeys(gs["kinds"])}
    print(f"goal set {a.out}: {len(gs['episodes'])} latents {n}"
          + (f", pairwise whitened distance median {np.median(off):.2f} max {off.max():.2f}"
             if len(off) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
