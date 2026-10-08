#!/usr/bin/env python3
"""Noised + belt-segmented copies of recorded episodes (``pcd_belt_seg``, source ``offline``).

For every ``episode_*.npz`` under each ``--in`` dir, writes ``OUT/<dir relative to data/lcs>/``
with ``pcd``/``pcd_rgb`` replaced by the noised cloud (``task_common.belt_cloud``), the new key
``pcd_belt_seg``, its per-frame ``sim_cloud_*`` features (``belt_cloud_features.cloud_extras``)
and ``sim_meta.belt_seg`` (``source = offline``); every other key is copied as is.
``index.json`` is copied with a ``belt_seg_offline`` block. Jaw centres = ``sim_ee_franka`` /
``sim_ee_ur``; camera origins = the scene directives. RNG per file:
``[seed, crc32(relative path), 11]``. Never writes into an input dir; refuses an existing --out.

    uv run --frozen python scripts/lcs/belt_seg_offline.py --in data/lcs/approach/v1_heldout \\
        --out data/lcs/realfirst/seg_offline/real_like --cloud-noise real_like --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from task_common import belt_cloud as bc
from task_common import belt_cloud_features as bcf
from task_common import lcs_dataset as lcs

DATA_ROOT = REPO_ROOT / "data" / "lcs"
CLOUD_NOISE_RNG = 11


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def relative_dir(path: Path) -> Path:
    """``path`` relative to ``data/lcs`` (else to the repo root, else its name)."""
    path = path.resolve()
    for root in (DATA_ROOT.resolve(), REPO_ROOT.resolve()):
        if path.is_relative_to(root):
            return path.relative_to(root)
    return Path(path.name)


def convert(src: Path, dst: Path, rel: str, noise: bc.CloudNoiseSpec, seg: bc.BeltSegSpec,
            seed: int, origins: np.ndarray, pose: np.ndarray) -> dict:
    with np.load(src, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    for key in ("pcd", "pcd_rgb", "sim_ee_franka", "sim_ee_ur"):
        if key not in arrays:
            return {"file": rel, "status": "skipped", "reason": f"no {key}"}
    rng = np.random.default_rng([seed, zlib.crc32(rel.encode()), CLOUD_NOISE_RNG])
    T = len(arrays["pcd"])
    pcd, rgb, belt_seg = (np.empty(T, dtype=object) for _ in range(3))
    feats: dict[str, list] = {k: [] for k in bcf.EXTRA_KEYS}
    for t in range(T):
        jaws = bc.jaw_centres_from_poses(arrays["sim_ee_franka"][t], arrays["sim_ee_ur"][t])
        out = bc.belt_seg_frame(np.asarray(arrays["pcd"][t], dtype=np.float32),
                                np.asarray(arrays["pcd_rgb"][t], dtype=np.uint8), noise, seg,
                                rng, origins, jaws, pose)
        pcd[t], rgb[t], belt_seg[t] = (lcs.camera_points(out["pcd"]), out["pcd_rgb"],
                                       out["pcd_belt_seg"])
        for k, v in bcf.cloud_extras(out["pcd_belt_seg"]).items():
            feats[k].append(v)
    meta = json.loads(str(arrays["sim_meta"])) if "sim_meta" in arrays else {}
    meta["belt_seg"] = bc.belt_seg_meta(noise, seg, "offline", pose)
    meta["cloud_noise"] = {**noise.to_dict(), "camera_origins": origins.tolist()}
    meta["belt_seg_offline"] = {"source": str(src), "source_sha256": _sha256(src),
                                "rng": [seed, f"crc32({rel})", CLOUD_NOISE_RNG]}
    arrays.update({lcs.EXTRA_PREFIX + k: np.stack(v) for k, v in feats.items()})
    arrays.update(pcd=pcd, pcd_rgb=rgb, pcd_belt_seg=belt_seg,
                  sim_meta=np.array(json.dumps(meta, sort_keys=True, default=lcs.json_default)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, dst)
    lcs.validate_episode(dst, period_us=None)
    m = np.array([len(s) for s in belt_seg])
    return {"file": rel, "status": "ok", "frames": T, "seg_min": int(m.min()),
            "seg_median": float(np.median(m)), "frac_ge_min": float(np.mean(m >= seg.min_points))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in", dest="inputs", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True, help="new root (must not exist)")
    p.add_argument("--cloud-noise", default="real_like", metavar="PRESET|JSON")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"--out {args.out} exists; refusing to overwrite")
    inputs = [d.resolve() for d in args.inputs]
    for d in inputs:
        if not d.is_dir():
            raise SystemExit(f"--in {d} is not a directory")
        if out.is_relative_to(d):
            raise SystemExit(f"--out {out} is inside the input dir {d}")
    noise, seg = bc.cloud_noise_spec(args.cloud_noise), bc.BeltSegSpec()
    origins, pose = bc.scene_camera_origins(), bc.known_large_pulley_pose7()
    for d in inputs:
        files = sorted(d.rglob("episode_*.npz"))
        rows = []
        for src in files:
            rel = relative_dir(src.parent) / src.name
            dst = out / rel
            row = convert(src, dst, str(rel), noise, seg, args.seed, origins, pose)
            rows.append(row)
            print(f"[SEG] {rel}: " + ", ".join(f"{k} {v}" for k, v in row.items()
                                                if k != "file"), flush=True)
        block = {"inputs": str(d), "cloud_noise": args.cloud_noise,
                 "noise_spec": noise.to_dict(), "seg_spec": seg.to_dict(), "seed": args.seed,
                 "pulley_pose7_known": pose.tolist(), "camera_origins": origins.tolist(),
                 "git": _git(), "script": "scripts/lcs/belt_seg_offline.py", "episodes": rows}
        for index_src in sorted({f.parent / "index.json" for f in files}):
            if not index_src.is_file():
                continue
            index = json.loads(index_src.read_text())
            index["belt_seg_offline"] = {**block, "source_index_sha256": _sha256(index_src)}
            dst = out / relative_dir(index_src.parent) / "index.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(index, indent=2, default=lcs.json_default) + "\n")
        ok = [r for r in rows if r["status"] == "ok"]
        print(f"[SEG] {d}: {len(ok)}/{len(rows)} written -> {out / relative_dir(d)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
