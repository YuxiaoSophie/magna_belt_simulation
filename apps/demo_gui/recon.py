"""Decoder reconstruction vs the observed belt cloud of a recorded episode frame (numpy, CPU).

GT = the decoder's training target: ``pcd_belt_seg`` (pulley frame) for the belt-cloud encoders,
else ``pcd_belt``, resampled to the decoder's point count. ``episode`` runs in a subprocess with
CUDA hidden: the encoder imports warp, and a CUDA context in the app would make ``gpu_free.sh``
hold every GPU job of the app.
"""

from __future__ import annotations

import functools
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from apps.demo_gui.config import REPO
from task_common import latent_encoder as le
from task_common.belt_cloud import resample_deterministic


@functools.lru_cache(maxsize=4)
def _model(deploy: str, decoder: str):
    return le.LatentEncoder.load(deploy, device="cpu"), le.LatentDecoder.load(decoder)


def chamfer_mm(a, b) -> float:
    """sqrt of the mean squared nearest-neighbour distance, both directions averaged (mm)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
    return float(1e3 * np.sqrt(0.5 * (d2.min(1).mean() + d2.min(0).mean())))


def _frames(deploy, decoder, npz, frames) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enc, dec = _model(str(deploy), str(decoder))
    belt_cloud = isinstance(enc, le.BeltCloudPointNetEncoder)
    keys = le.episode_input_keys(enc) + ("sim_cmd_source", "pcd_belt")
    with np.load(npz, allow_pickle=True) as z:
        d = {k: z[k] for k in keys if k in z.files}
    z_, _ = le.encode_episode_frames(enc, d, frames)
    pred = dec.decode_batch(z_)
    gt = np.stack([resample_deterministic(d["pcd_belt_seg" if belt_cloud else "pcd_belt"][i],
                                          dec.num_points)[0] for i in frames])
    return gt, pred, np.array([chamfer_mm(g, p) for g, p in zip(gt, pred, strict=True)])


def predict_frame(deploy, decoder, npz, i: int) -> tuple[np.ndarray, np.ndarray, float]:
    """``(gt_xyz, pred_xyz, chamfer_mm)`` of frame ``i``, both ``(decoder points, 3)``."""
    gt, pred, ch = _frames(deploy, decoder, npz, [int(i)])
    return gt[0], pred[0], float(ch[0])


def episode(deploy, decoder, npz, cache_dir) -> dict:
    """Every frame of ``npz`` (``gt``, ``pred``, ``chamfer_mm``), computed once per model and
    episode in a CUDA-free subprocess, cached under ``cache_dir``."""
    key = hashlib.sha1(f"{deploy}|{decoder}|{npz}".encode()).hexdigest()[:16]
    out = Path(cache_dir) / f"recon_{key}.npz"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.run([sys.executable, "-m", "apps.demo_gui.recon", str(deploy),
                            str(decoder), str(npz), str(out)], check=False, cwd=REPO,
                           capture_output=True, text=True, timeout=600,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        if p.returncode:
            raise RuntimeError(p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.returncode)
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


def main(deploy, decoder, npz, out) -> None:
    with np.load(npz, allow_pickle=True) as z:
        n = len(z["actions"])
    gt, pred, ch = _frames(deploy, decoder, npz, range(n))
    tmp = Path(out).with_suffix(".tmp.npz")
    np.savez(tmp, gt=gt, pred=pred, chamfer_mm=ch)
    os.replace(tmp, out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
