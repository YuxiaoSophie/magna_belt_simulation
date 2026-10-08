#!/usr/bin/env python3
"""MILES-BC prior (E6, V12-28): MLP ``pi(z) -> u`` (12-d cmd_delta; ``--inputs zp``: ``pi(z,
p_franka, p_ur)``, opt-in, pulls toward the demos' EE path) on a buffer's anchor data, exported as
``bc_prior.npz`` (``task_common.bc_prior``) for ``prior_mode: policy``.

Rows of ``--kinds`` from ``--buffer`` manifests (deduplicated by episode):
- ``demo``: frames from the first non-zero action to the last one + ``--max-hold-frames`` of the
  final hold (the hold would otherwise dominate the loss); ``--success-only``: label_human rows.
- ``miles``: return / hold / walk frames only, deviation frames excluded, from the collector's
  per-frame npz key ``miles_segment`` or the row's ``segments`` ({name: [start, stop)}) or
  ``segment: walk`` (whole file); a miles row with none of these is refused.
Latents: the deploy's numpy encoder per frame (as the encoder node; empty clouds dropped), or
``--cache`` (``encode_latents_cache.py`` output of the same checkpoint). ``p`` (zp only) = world
EE positions ``sim_ee_*[:, :3]`` (the node's p0). Targets = ``actions`` clipped to the deploy box,
normalised ``(u - u_mid) / u_half``. Validation: ``--val-episodes`` held-out episodes; the export
is the best-validation epoch. Writes ``--out`` and ``<out>.json`` (data, losses, per-channel MAE).

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/lcs/train_bc_prior.py \\
        --buffer data/lcs/realfirst/cmp/C1_transport/iter1/buffer.json \\
        --deploy <deploy.npz> --kinds demo --out <dir>/bc_prior.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from latent_stage_tols import encode_frames

from task_common.bc_prior import (
    FORMAT,
    INPUTS,
    N_P,
    BcPrior,
    box_scale,
    encoder_fingerprint,
)
from task_common.latent_encoder import LatentEncoder

KINDS = ("demo", "miles")
MILES_KEEP = ("walk", "return", "hold")
MILES_DROP = ("dev",)


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def demo_frames(actions: np.ndarray, max_hold: int) -> np.ndarray:
    moving = np.flatnonzero(np.abs(actions).sum(1) > 0.0)
    if not len(moving):
        return np.zeros(0, np.int64)
    return np.arange(moving[0], min(len(actions), moving[-1] + 1 + max_hold))


def miles_frames(row: dict, npz: dict, T: int) -> np.ndarray:
    """Return / hold / walk frames of a MILES cycle file (deviation excluded)."""
    if "miles_segment" in npz:
        seg = np.asarray(npz["miles_segment"]).astype(str)
        bad = set(seg) - set(MILES_KEEP) - set(MILES_DROP)
        if bad:
            raise SystemExit(f"{row['episode']}: unknown miles_segment values {sorted(bad)}")
        return np.flatnonzero(np.isin(seg, MILES_KEEP))
    if isinstance(row.get("segments"), dict):
        keep = [np.arange(int(a), min(int(b), T)) for k, (a, b) in row["segments"].items()
                if k in MILES_KEEP]
        return np.unique(np.concatenate(keep)) if keep else np.zeros(0, np.int64)
    if row.get("segment") == "walk":
        return np.arange(T)
    raise SystemExit(f"miles row {row.get('id')} ({row['episode']}): no miles_segment npz key, "
                     "'segments' or 'segment: walk' to exclude the deviation frames")


def select_rows(buffers: list[Path], kinds: list[str], success_only: bool) -> list[dict]:
    rows, seen = [], set()
    for b in buffers:
        for r in json.loads(b.read_text())["rollouts"]:
            if r["kind"] not in kinds or r["episode"] in seen:
                continue
            if success_only and r["kind"] == "demo" and not r.get("label_human"):
                continue
            seen.add(r["episode"])
            rows.append(r)
    return rows


def cache_lookup(cache: Path, deploy: Path):
    """``(file, step) -> z`` from an ``encode_latents_cache.py`` cache of the deploy's checkpoint."""
    with np.load(cache, allow_pickle=False) as c, np.load(deploy, allow_pickle=False) as d:
        if str(c["checkpoint_sha256"]) != str(d["checkpoint_sha256"]):
            raise SystemExit(f"{cache}: checkpoint != the deploy's")
        files = [str(f) for f in c["files"]]
        z, fi, st = np.asarray(c["z_t"], np.float64), c["file_idx"], c["step"]
    table = {(files[int(i)], int(s)): k for k, (i, s) in enumerate(zip(fi, st))}
    return lambda f, t: z[table[(f, t)]] if (f, t) in table else None


def load_data(cli, rows: list[dict]) -> tuple[list[dict], dict]:
    enc = LatentEncoder.load(cli.deploy)
    look = cache_lookup(cli.cache, cli.deploy) if cli.cache else None
    eps, stats = [], {"n_empty": 0, "n_cache_miss": 0}
    for r in rows:
        path = resolve(r["episode"])
        with np.load(path, allow_pickle=True) as f:
            npz = {k: f[k] for k in ("actions", "sim_ee_franka", "sim_ee_ur", "pcd_belt_seg",
                                     "miles_segment") if k in f.files}
        u = np.asarray(npz["actions"], np.float64)
        frames = (demo_frames(u, cli.max_hold_frames) if r["kind"] == "demo"
                  else miles_frames(r, npz, len(u)))
        if look is not None:
            zs = [look(r["episode"], int(t)) for t in frames]
            keep = np.array([z is not None for z in zs], bool)
            stats["n_cache_miss"] += int((~keep).sum())
            frames = frames[keep]
            Z = np.stack([z for z in zs if z is not None]) if keep.any() else np.zeros((0, 1))
        else:
            seg = list(npz["pcd_belt_seg"])
            Z, keep = [], []
            for t in frames:
                Z.append(encode_frames(enc, seg, [t])[0][0])
                keep.append(not bool(enc.last_empty))
            keep = np.asarray(keep, bool)
            stats["n_empty"] += int((~keep).sum())
            frames, Z = frames[keep], np.asarray(Z)[keep] if len(Z) else np.zeros((0, 1))
        if not len(frames):
            continue
        P = np.concatenate([np.asarray(npz["sim_ee_franka"], np.float64)[frames, :3],
                            np.asarray(npz["sim_ee_ur"], np.float64)[frames, :3]], axis=1)
        eps.append({"id": r.get("id"), "kind": r["kind"], "episode": r["episode"],
                    "frames": frames, "Z": Z, "P": P, "U": u[frames]})
    return eps, stats


def mlp(n_in: int, hidden: list[int], n_out: int) -> torch.nn.Sequential:
    layers, w = [], n_in
    for h in hidden:
        layers += [torch.nn.Linear(w, h), torch.nn.ReLU()]
        w = h
    return torch.nn.Sequential(*layers, torch.nn.Linear(w, n_out))


def export(net: torch.nn.Sequential, out: Path, meta: dict) -> None:
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    w = {f"w{i}": m.weight.detach().double().numpy().T.copy() for i, m in enumerate(lin)}
    b = {f"b{i}": m.bias.detach().double().numpy().copy() for i, m in enumerate(lin)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, format=np.array(FORMAT), n_layers=np.array(len(lin)), **w, **b, **meta)


def train(cli, eps: list[dict], u_mid, u_half, lb, ub, n_x: int) -> tuple:
    rng = np.random.default_rng(cli.seed)
    torch.manual_seed(cli.seed)
    n_val = cli.val_episodes if len(eps) > cli.val_episodes + 1 else 0
    val_ids = set(rng.choice(len(eps), n_val, replace=False).tolist()) if n_val else set()

    n_in = n_x + (N_P if cli.inputs == "zp" else 0)

    def stack(ids):
        if not ids:
            return np.zeros((0, n_in)), np.zeros((0, len(u_mid)))
        X = np.concatenate([np.concatenate([eps[i]["Z"], eps[i]["P"]], 1)[:, :n_in]
                            for i in ids])
        U = np.concatenate([np.clip(eps[i]["U"], lb, ub) for i in ids])
        return X, (U - u_mid) / u_half

    tr = [i for i in range(len(eps)) if i not in val_ids]
    Xtr, Ytr = stack(tr)
    Xva, Yva = stack(sorted(val_ids))
    in_mean, in_std = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    xtr, ytr = t((Xtr - in_mean) / in_std), t(Ytr)
    xva, yva = t((Xva - in_mean) / in_std), t(Yva)
    net = mlp(n_in, cli.hidden, len(u_mid))
    opt = torch.optim.AdamW(net.parameters(), lr=cli.lr, weight_decay=cli.wd)
    best, curve = (np.inf, None, -1), []
    for ep in range(cli.epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for k in range(0, len(xtr), cli.batch):
            idx = perm[k:k + cli.batch]
            x = xtr[idx]
            if cli.noise_in > 0.0:
                x = x + cli.noise_in * torch.randn_like(x)
            loss = torch.mean((net(x) - ytr[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ltr = float(torch.mean((net(xtr) - ytr) ** 2))
            lva = float(torch.mean((net(xva) - yva) ** 2)) if len(xva) else ltr
        curve.append((ep, ltr, lva))
        if lva < best[0]:
            best = (lva, {k: v.clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1])
    return net, in_mean, in_std, (Xtr, Ytr, Xva, Yva), sorted(val_ids), best[2], curve


def mae_mm(net, in_mean, in_std, X, Y, u_half) -> list[float]:
    if not len(X):
        return []
    with torch.no_grad():
        y = net(torch.as_tensor((X - in_mean) / in_std, dtype=torch.float32)).double().numpy()
    err = np.abs(y - Y).mean(0) * u_half
    return [round(float(e) * (1e3 if c < 6 else 180.0 / np.pi), 4) for c, e in enumerate(err)]


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--buffer", type=Path, action="append", required=True,
                   help="buffer.json manifest (repeatable; rows deduplicated by episode)")
    p.add_argument("--deploy", type=Path, required=True, help="deploy.npz (encoder + box)")
    p.add_argument("--out", type=Path, required=True, help="bc_prior.npz")
    p.add_argument("--kinds", default="demo,miles", help="row kinds (demo, miles)")
    p.add_argument("--inputs", choices=INPUTS, default="z",
                   help="z: pi(z) (default); zp: pi(z, p_franka, p_ur) (absolute EE positions)")
    p.add_argument("--success-only", action="store_true", help="demo rows with label_human only")
    p.add_argument("--max-hold-frames", type=int, default=20,
                   help="demo: frames kept after the last non-zero action")
    p.add_argument("--cache", type=Path, default=None,
                   help="encode_latents_cache.py npz of the deploy's checkpoint (else encode)")
    p.add_argument("--hidden", type=lambda s: [int(x) for x in s.split(",")], default=[128, 128])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--noise-in", type=float, default=0.0,
                   help="Gaussian noise on the normalised inputs during training")
    p.add_argument("--val-episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> int:
    cli = create_parser().parse_args()
    kinds = [k.strip() for k in cli.kinds.split(",") if k.strip()]
    if not kinds or set(kinds) - set(KINDS):
        raise SystemExit(f"--kinds {cli.kinds!r}: subset of {KINDS}")
    t0 = time.perf_counter()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    rows = select_rows(cli.buffer, kinds, cli.success_only)
    eps, stats = load_data(cli, rows)
    if not eps:
        raise SystemExit("no training frames")
    with np.load(cli.deploy, allow_pickle=False) as d:
        lb, ub = np.asarray(d["u_lb"], np.float64), np.asarray(d["u_ub"], np.float64)
        n_x = int(d["n_x"])
    u_mid, u_half = box_scale(lb, ub)
    net, in_mean, in_std, (Xtr, Ytr, Xva, Yva), val, best_ep, curve = train(
        cli, eps, u_mid, u_half, lb, ub, n_x)
    fp = encoder_fingerprint(cli.deploy)
    export(net, cli.out, {"in_mean": in_mean, "in_std": in_std, "u_mid": u_mid,
                          "u_half": u_half, "n_x": np.array(n_x), "encoder_sha256": np.array(fp),
                          "deploy": np.array(str(cli.deploy.resolve())),
                          "kinds": np.array(",".join(kinds)), "inputs": np.array(cli.inputs)})
    bc = BcPrior.load(cli.out)
    with torch.no_grad():
        y_net = net(torch.as_tensor((Xtr - in_mean) / in_std, dtype=torch.float32)).double()
    P_tr = Xtr[:, n_x:] if cli.inputs == "zp" else np.zeros((len(Xtr), N_P))
    u_np = bc.forward(Xtr[:, :n_x], P_tr)
    np_vs_torch = float(np.abs(u_np - (u_mid + u_half * y_net.numpy())).max())
    base = float(np.mean((Ytr - Ytr.mean(0)) ** 2))
    _, ltr, lva = curve[best_ep]
    by_kind = {k: sum(len(e["frames"]) for e in eps if e["kind"] == k) for k in kinds}
    rep = {"out": str(cli.out), "deploy": str(cli.deploy), "encoder_sha256": fp,
           "inputs": cli.inputs,
           "buffers": [str(b) for b in cli.buffer], "kinds": kinds,
           "success_only": cli.success_only, "max_hold_frames": cli.max_hold_frames,
           "cache": None if cli.cache is None else str(cli.cache), **stats,
           "n_episodes": len(eps), "frames_by_kind": by_kind, "n_train": len(Xtr),
           "n_val": len(Xva), "val_episodes": [eps[i]["episode"] for i in val],
           "hidden": cli.hidden, "epochs": cli.epochs, "best_epoch": best_ep,
           "mse_train": ltr, "mse_val": lva, "mse_train_mean_baseline": base,
           "mae_val_mm_deg": mae_mm(net, in_mean, in_std, Xva, Yva, u_half),
           "mae_train_mm_deg": mae_mm(net, in_mean, in_std, Xtr, Ytr, u_half),
           "numpy_vs_torch_max_abs_u": np_vs_torch, "wall_s": time.perf_counter() - t0,
           "curve_every_20": [c for c in curve if c[0] % 20 == 0]}
    cli.out.with_suffix(".json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[bc] inputs {cli.inputs}, {len(eps)} episodes {by_kind} -> {len(Xtr)} train / {len(Xva)} val frames; "
          f"best epoch {best_ep}: mse train {ltr:.4f} val {lva:.4f} (mean baseline {base:.4f}); "
          f"val MAE mm/deg {rep['mae_val_mm_deg']}; numpy vs torch {np_vs_torch:.1e}; "
          f"wrote {cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
