#!/usr/bin/env python3
"""One-step prediction of the v2 learned LCS on the motion-primitive episodes vs the test split.

Per episode (``collect_motion_primitives.py`` output) and for the held-out test split: the
one-step belt RMSE ``decode(LCS(encode(obs_k), u_k))`` vs ``belt_{k+1}`` (exact LCP, as
``prediction_video``), the no-motion baseline ``belt_k``, the reconstruction
``decode(encode(obs_{k+1}))``, and the dynamics-only error ``pred`` vs that reconstruction. OOD:
whitened latent distance (deploy ``z_std``) of each frame to its nearest training frame.
Writes ``<run>/eval.json`` (or ``--out``). Runs from other writers (no ``plan`` / ``primitive``
in the index rows) are keyed by file stem and skip the amplitude table.

Run:
    uv run python scripts/lcs/eval_motion_primitives.py --run data/lcs/motion_primitives/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from task_common import prediction_video as pv
from task_common.latent_encoder import LearnedLcs

CACHE = REPO_ROOT / "data" / "lcs" / "motion_primitives" / "_cache"
REFERENCE = {"model": 0.577, "nomotion": 1.31, "recon": 0.393}  # evaluate_v2, test split
KEYS = ("model", "recon", "nomotion", "pred_vs_recon", "nomotion_decoded")


def _sha8(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]


def train_latents(model: pv.OneStepModel, split: Path) -> tuple[np.ndarray, list]:
    """Latents of every train-split frame (cached per deploy)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"train_latents_{_sha8(model.deploy)}_{_sha8(split)}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as d:
            return d["z"], json.loads(str(d["ids"]))
    files = json.loads(split.read_text())["train"]
    zs, ids = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            zs.append(model.enc.encode_batch(list(d["pcd"]), d["state"], d["pcd_belt"]))
        ids += [[f, t] for t in range(len(zs[-1]))]
    Z = np.concatenate(zs)
    np.savez(cache, z=Z, ids=np.asarray(json.dumps(ids)))
    return Z, ids


def nearest(Z: np.ndarray, z: np.ndarray, z_std: np.ndarray, chunk: int = 512
            ) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``z``: whitened distance to the nearest row of ``Z`` and its index."""
    Zw = Z / z_std
    d_out, i_out = [], []
    for a in range(0, len(z), chunk):
        q = z[a:a + chunk] / z_std
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ Zw.T + (Zw ** 2).sum(1)[None]
        i = d2.argmin(1)
        d_out.append(np.sqrt(np.maximum(d2[np.arange(len(q)), i], 0.0)))
        i_out.append(i)
    return np.concatenate(d_out), np.concatenate(i_out)


def evaluate(model: pv.OneStepModel, path: Path) -> dict:
    ep = pv.Episode.load(path)
    rows = [pv.one_step(model, ep, k) for k in range(ep.frames - 1)]
    pred = np.stack([r["pred"] for r in rows])
    recon = np.stack([r["recon_next"] for r in rows])
    true, cur = ep.belt[1:], ep.belt[:-1]
    z = np.stack([r["z"] for r in rows])
    z_last = model.encode(ep.pcd[-1], ep.prop[-1], ep.belt[-1])
    recon_cur = model.decode(z)
    step = ep.data["sim_episode_step"] if "sim_episode_step" in ep.data else np.zeros(ep.frames)
    return {"ep": ep, "rmse": {"model": pv.rmse_mm(pred, true), "recon": pv.rmse_mm(recon, true),
                               "nomotion": pv.rmse_mm(cur, true),
                               "pred_vs_recon": pv.rmse_mm(pred, recon),
                               "nomotion_decoded": pv.rmse_mm(recon_cur, recon)},
            "dev_from_start": pv.rmse_mm(true, ep.belt[:1]),
            "gain": {"recon": gain(recon, true), "pred": gain(pred, true)},
            "z": np.vstack([z, z_last]), "moving": np.asarray(step[:-1]) >= 0,
            "u": ep.u[:-1], "time_s": ep.dt * np.arange(ep.frames - 1)}


def gain(est: np.ndarray, true: np.ndarray) -> float:
    """Share of the true belt displacement from tuple 0 that ``est`` reproduces (LS slope)."""
    d_t = np.asarray(true, np.float64) - true[0]
    d_e = np.asarray(est, np.float64) - est[0]
    return float((d_e * d_t).sum() / (d_t * d_t).sum())


def summarize(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "max": float(x.max()), "p95": float(np.percentile(x, 95))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--deploy", type=Path, default=pv.DEPLOY)
    p.add_argument("--decoder", type=Path, default=pv.DECODER)
    p.add_argument("--split", type=Path, default=pv.SPLIT)
    p.add_argument("--skip-test", action="store_true", help="no test-split recompute")
    p.add_argument("--out", type=Path, default=None, help="result json (default <run>/eval.json)")
    args = p.parse_args()
    t0 = time.perf_counter()
    model = pv.OneStepModel(args.deploy, args.decoder, "lcp")
    z_std = LearnedLcs.load(args.deploy).z_std
    index = json.loads((args.run / "index.json").read_text())
    eps = [r for r in index["episodes"] if r["status"] == "ok"]
    Z, ids = train_latents(model, args.split)
    print(f"train latents {Z.shape} ({time.perf_counter() - t0:.0f} s)", flush=True)

    ref = None
    if not args.skip_test:
        acc = {k: [] for k in KEYS}
        nn_test = []
        for f in json.loads(args.split.read_text())["test"]:
            r = evaluate(model, Path(f))
            for k in KEYS:
                acc[k].append(r["rmse"][k])
            nn_test.append(nearest(Z, r["z"], z_std)[0])
        nn_all = np.concatenate(nn_test)
        ref = {"episodes": len(acc["model"]), "tuples": int(sum(len(a) for a in acc["model"])),
               **{k: summarize(np.concatenate(v)) for k, v in acc.items()},
               "dyn_skill": float(1.0 - np.concatenate(acc["pred_vs_recon"]).mean()
                                  / np.concatenate(acc["nomotion_decoded"]).mean()),
               "nn_train_dist": {**summarize(nn_all), "p50": float(np.median(nn_all)),
                                 "p99": float(np.percentile(nn_all, 99))},
               "quoted": REFERENCE}
        print(f"test split: model {ref['model']['mean']:.3f} recon {ref['recon']['mean']:.3f} "
              f"no-motion {ref['nomotion']['mean']:.3f} mm; NN dist p50 "
              f"{ref['nn_train_dist']['p50']:.3f} p95 {ref['nn_train_dist']['p95']:.3f} "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)

    out = {"run": str(args.run), "deploy": str(args.deploy), "decoder": str(args.decoder),
           "solver": "lcp (exact, = evaluate_v2's qpOASES)", "split": str(args.split),
           "n_train_frames": len(Z), "reference_test_split": ref,
           "definitions": {
               "model": "rmse(decode(LCS(encode(obs_k), u_k)), belt_{k+1}) per tuple, mm",
               "recon": "rmse(decode(encode(obs_{k+1})), belt_{k+1})",
               "nomotion": "rmse(belt_k, belt_{k+1})",
               "pred_vs_recon": "rmse(pred, decode(encode(obs_{k+1}))): the latent dynamics' "
                                "error alone",
               "nomotion_decoded": "rmse(decode(encode(obs_k)), decode(encode(obs_{k+1}))): "
                                   "the no-motion baseline in decoded space",
               "dyn_skill": "1 - mean(pred_vs_recon) / mean(nomotion_decoded) over motion "
                            "tuples (1 = perfect latent step, 0 = no better than no motion)",
               "displacement_gain": "LS slope of (est_k - est_0) on (belt_{k+1} - belt_1) "
                                    "over all tuples: 1 = the displacement from the start is "
                                    "reproduced, 0 = the estimate stays at the start shape",
               "recon_vs_dev_corr": "corr over tuples of the recon error with the true belt's "
                                    "rmse to the episode's first frame",
               "all": "every tuple (as the test split numbers); motion: tuples with "
                      "sim_episode_step >= 0 (after the pre-hold)",
               "nn_train_dist": "whitened (deploy z_std) latent distance of each frame to its "
                                "nearest train-split frame"},
           "primitives": {}}
    for row in eps:
        name = row.get("primitive") or Path(row["file"]).stem
        if name in out["primitives"]:
            name = f"{name}#{Path(row['file']).stem}"
        r = evaluate(model, args.run / row["file"])
        mv = r["moving"]
        nn, nn_i = nearest(Z, r["z"], z_std)
        entry = {"file": row["file"], "frames": r["ep"].frames,
                 "all": {k: summarize(r["rmse"][k]) for k in KEYS},
                 "motion": {k: summarize(r["rmse"][k][mv]) for k in KEYS},
                 "dyn_skill": float(1.0 - r["rmse"]["pred_vs_recon"][mv].mean()
                                    / r["rmse"]["nomotion_decoded"][mv].mean()),
                 "belt_dev_from_start_mm": summarize(r["dev_from_start"]),
                 "displacement_gain": r["gain"],
                 "recon_vs_dev_corr": float(np.corrcoef(r["rmse"]["recon"],
                                                        r["dev_from_start"])[0, 1]),
                 "nn_train_dist": {"all": summarize(nn), "motion": summarize(nn[:-1][mv]),
                                   "nearest_at_max": ids[int(nn_i[int(nn.argmax())])]},
                 "series": {"time_s": r["time_s"].round(4).tolist(),
                            "moving": mv.tolist(),
                            **{f"rmse_{k}_mm": r["rmse"][k].round(5).tolist() for k in KEYS},
                            "belt_dev_from_start_mm": r["dev_from_start"].round(4).tolist(),
                            "nn_train_dist": nn.round(4).tolist()},
                 "collection": {k: row["stats"][k] for k in (
                     "u_bound_ratio_max", "u_bound_ratio_max_per_dim", "u_near_bound_frac",
                     "crop_margin_min_mm", "belt_z_max_m", "rod_stretch_gain_pct_max",
                     "min_board_clearance_mm", "grasp_ok_all", "wrap_deg_max")
                     if k in row.get("stats", {})},
                 "effective_amplitude": {c["name"]: [round(c["amp_effective"], 3), c["unit"]]
                                         for c in row.get("plan", {}).get("components", [])}}
        plan = row.get("plan", {})
        entry.update(motion_s=plan.get("motion_s"), action_ood=bool(plan.get("action_ood")),
                     bound_frac=plan.get("bound_frac"), amp_mul=row.get("amp_mul"),
                     stretch_lobe=row.get("stretch_lobe", 1.0),
                     lobe_amplitudes=plan.get("lobe_amplitudes"),
                     model_below_nomotion_frac={
                         "all": float((r["rmse"]["model"] < r["rmse"]["nomotion"]).mean()),
                         "motion": float((r["rmse"]["model"][mv]
                                          < r["rmse"]["nomotion"][mv]).mean())})
        if ref is not None:
            entry["nn_frac_above_test_p95"] = float((nn[:-1][mv] > ref["nn_train_dist"]["p95"])
                                                    .mean())
            entry["nn_frac_above_test_p99"] = float((nn[:-1][mv] > ref["nn_train_dist"]["p99"])
                                                    .mean())
        out["primitives"][name] = entry
        m = entry["all"]
        print(f"{name:<18} model {m['model']['mean']:.3f}/{m['model']['max']:.3f} "
              f"recon {m['recon']['mean']:.3f} no-motion {m['nomotion']['mean']:.3f} dyn "
              f"{m['pred_vs_recon']['mean']:.3f} | motion model "
              f"{entry['motion']['model']['mean']:.3f} | NN dist mean "
              f"{entry['nn_train_dist']['motion']['mean']:.3f} max "
              f"{entry['nn_train_dist']['motion']['max']:.3f} | dyn skill "
              f"{entry['dyn_skill']:.2f}, belt dev max {entry['belt_dev_from_start_mm']['max']:.1f}"
              f" mm, corr(recon, dev) {entry['recon_vs_dev_corr']:.2f}, displacement gain "
              f"recon {r['gain']['recon']:.2f} pred {r['gain']['pred']:.2f}", flush=True)
    out["table_md"] = table(out)
    has_plan = all("plan" in row for row in eps)
    out["amplitude_table_md"] = amp_table(out, index) if has_plan else None
    dst = args.out or args.run / "eval.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(out["table_md"])
    if has_plan:
        print(out["amplitude_table_md"])
    print(f"wrote {dst} ({time.perf_counter() - t0:.0f} s)")
    return 0


def table(out: dict) -> str:
    ref = out["reference_test_split"]
    head = ("| set | T s | model mean / max | recon mean | no-motion mean | dyn-only (pred vs "
            "recon) | motion-only model / recon / no-motion | dyn skill | disp. gain recon / pred "
            "| model < no-motion (motion tuples) | NN-train dist mean / max (motion) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    if ref is not None:
        lines.append(f"| test split ({ref['tuples']} tuples; quoted 0.577 / 0.393 / 1.31) | - | "
                     f"{ref['model']['mean']:.3f} / {ref['model']['max']:.3f} | "
                     f"{ref['recon']['mean']:.3f} | {ref['nomotion']['mean']:.3f} | "
                     f"{ref['pred_vs_recon']['mean']:.3f} | - | {ref['dyn_skill']:.2f} | - | - | "
                     f"{ref['nn_train_dist']['mean']:.3f} / p95 {ref['nn_train_dist']['p95']:.3f}"
                     f" / p99 {ref['nn_train_dist']['p99']:.3f} |")
    for name, e in out["primitives"].items():
        a, m, nn = e["all"], e["motion"], e["nn_train_dist"]["motion"]
        label = name + (f" (ACTION-OOD, <= {e['bound_frac']:g}x bounds)" if e["action_ood"]
                        else "")
        g = e["displacement_gain"]
        T = "-" if e.get("motion_s") is None else f"{e['motion_s']:.1f}"
        lines.append(f"| {label} | {T} | {a['model']['mean']:.3f} / {a['model']['max']:.3f} | "
                     f"{a['recon']['mean']:.3f} | {a['nomotion']['mean']:.3f} | "
                     f"{a['pred_vs_recon']['mean']:.3f} | {m['model']['mean']:.3f} / "
                     f"{m['recon']['mean']:.3f} / {m['nomotion']['mean']:.3f} | "
                     f"{e['dyn_skill']:.2f} | {g['recon']:.2f} / {g['pred']:.2f} | "
                     f"{100 * e['model_below_nomotion_frac']['motion']:.1f} % | "
                     f"{nn['mean']:.3f} / {nn['max']:.3f} |")
    return head + "\n".join(lines) + "\n"


def amp_table(out: dict, index: dict) -> str:
    """Achieved (planned) peak + / - amplitude per component, and the scaling that bound."""
    rows = ["| primitive | T s | amplitude +peak / -peak (mm or deg) | amp x | stretch lobe |",
            "|---|---|---|---|---|"]
    for name, e in out["primitives"].items():
        la = e.get("lobe_amplitudes") or {}
        amp = ", ".join(f"{k} {v[0]:+.1f}/{v[1]:+.1f}" for k, v in la.items())
        T = "-" if e.get("motion_s") is None else f"{e['motion_s']:.1f}"
        rows.append(f"| {name} | {T} | {amp} | {e.get('amp_mul', 1.0):g} | "
                    f"{e.get('stretch_lobe', 1.0):g} |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    sys.exit(main())
