#!/usr/bin/env python3
"""Offline encoder + latent-LCS evaluation for the encoder study (V12-42). CPU, no sim.

Models: ``--model NAME=PATH`` as in ``seed_compare.py`` (deploy.npz with decoder.npz beside it,
a loop iteration dir, or a checkpoint .pt with ``--export-buffer``). Eval sets: ``--eval-sets``
(``encoder_tiers.py``'s eval_sets.json: E_off = V12-25's 87 off-demo rollouts, E_hs = the
no-prior hard-suite rollouts of V12-37/39/40). The sets are fixed: an episode in a model's
training buffer is reported, not dropped.
Per set (``seed_compare`` functions): recon / one-step / H-step Chamfer RMS (mm) overall, by
group, phase and demo-tube distance (20 seed demos, as V12-41); +y and null-direction probes from
the final latents of engaged / over episodes. On E_off only:
- action response: from ``--ar-starts`` sampled E_off frames, each of the 24 signed unit actions
  (12 dims x +-, ``--ar-scale`` fixed 3 mm / 2 deg, or std = the per-dim action std of
  ``--ar-std-buffer``) held for 1 and 5 LCS steps; response = (z_k(u) - z_k(0)) / z_std. Reported:
  mean response norm per dim, the min pairwise distance of the 24 mean responses (means from one
  half of the episodes), separability = fraction of the other half's responses nearest to their
  own action's mean (Delta-JEPA Fig. 6);
- action decodability: ridge from Dz_k = (z_{t+k} - z_t) / z_std (k = 1, 5) to the executed
  actions summed over t..t+k-1, episodes split in half (alpha by an inner episode split); R^2 =
  mean over the arm's 6 dims, overall and by phase.
``--behaviour`` adds seed_compare's MPPI behaviour metrics. Writes ``<out>/report.{md,json}``.

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/lcs/encoder_eval.py \\
        --model V12h=data/lcs/realfirst/cmp/C1_transport/iter1 \\
        --out data/lcs/realfirst/cmp/encoder_study/baseline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")  # uncapped BLAS threads made the numpy encoder ~10x slower

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_compare import (
    CMP,
    DTYPE,
    PHASES,
    PROBES,
    REPO,
    Model,
    behaviour,
    demo_points,
    f,
    jload,
    load_episode,
    prediction,
    probe_states,
    probes,
    rel,
    resolve_model,
    split_table,
)

ARMS = {"Franka": [0, 1, 2, 6, 7, 8], "UR": [3, 4, 5, 9, 10, 11]}
DIMS = ("f_x", "f_y", "f_z", "u_x", "u_y", "u_z", "f_rx", "f_ry", "f_rz", "u_rx", "u_ry", "u_rz")
FIXED_SCALE = np.array([3e-3] * 6 + [math.radians(2.0)] * 6)
ALPHAS = 10.0 ** np.arange(-4, 3)
STUDY = "data/lcs/realfirst/cmp/encoder_study"


# ---------------------------------------------------------------- data


def load_set(rows: list[dict], a, demo_p, rng) -> list[dict]:
    if a.max_episodes and len(rows) > a.max_episodes:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), a.max_episodes, replace=False))]
    eps = []
    for r in rows:
        e = load_episode(r, a.horizon, a.frame_stride, demo_p)
        e["group"] = r["group"]
        e["empty"] = np.array([len(s) == 0 for s in e["seg"]], bool)
        eps.append(e)
    return eps


def episode_split(eps: list[dict], seed: int) -> np.ndarray:
    """(n_eps,) True = fit half; halves drawn per group."""
    rng = np.random.default_rng(seed)
    fit = np.zeros(len(eps), bool)
    for g in dict.fromkeys(e["group"] for e in eps):
        ids = np.flatnonzero([e["group"] == g for e in eps])
        fit[rng.permutation(ids)[: (len(ids) + 1) // 2]] = True
    return fit


def action_scale(a) -> np.ndarray:
    if a.ar_scale == "fixed":
        return FIXED_SCALE
    us = []
    for r in jload(REPO / a.ar_std_buffer)["rollouts"]:
        with np.load(r["episode"], allow_pickle=True) as d:
            us.append(np.asarray(d["actions"], np.float64))
    return np.concatenate(us).std(axis=0)


# ---------------------------------------------------------------- action response


def roll(model: Model, z, p, u, steps: int) -> torch.Tensor:
    for _ in range(steps):
        z = model.step(z, u, p)
        p = p + u[:, :6]
    return z


def action_response(model: Model, eps, zs, starts, fit, scale, steps_list) -> dict:
    z0 = torch.stack([zs[i][t] for i, t in starts])
    p0 = torch.as_tensor(np.stack([eps[i]["P"][t] for i, t in starts]), dtype=DTYPE)
    is_fit = np.array([fit[i] for i, _ in starts])
    n, nu = len(z0), model.lcs.n_u
    acts = [(d, s) for d in range(nu) for s in (1.0, -1.0)]
    U = torch.zeros(len(acts), nu, dtype=DTYPE)
    for j, (d, s) in enumerate(acts):
        U[j, d] = s * scale[d]
    out = {}
    for k in steps_list:
        base = roll(model, z0, p0, torch.zeros(n, nu, dtype=DTYPE), k)
        zz = roll(model, z0.repeat(len(acts), 1), p0.repeat(len(acts), 1),
                  U.repeat_interleave(n, 0), k)
        resp = ((zz.view(len(acts), n, -1) - base) / model.z_std).numpy()
        norms = np.linalg.norm(resp, axis=-1)
        means = resp[:, is_fit].mean(axis=1)
        dm = np.linalg.norm(means[:, None] - means[None], axis=-1)
        dm[np.diag_indices(len(acts))] = np.inf
        i, j = np.unravel_index(np.argmin(dm), dm.shape)
        held = resp[:, ~is_fit]
        nearest = np.linalg.norm(held[:, :, None] - means[None, None], axis=-1).argmin(-1)
        hit = nearest == np.arange(len(acts))[:, None]
        name = [f"{DIMS[d]}{'+' if s > 0 else '-'}" for d, s in acts]
        out[f"k{k}"] = {
            "norm_per_dim": {DIMS[d]: float(norms[2 * d:2 * d + 2].mean()) for d in range(nu)},
            "norm_median": float(np.median(norms)),
            "min_pair_dist": float(dm[i, j]), "min_pair": [name[i], name[j]],
            "min_pair_rel": float(dm[i, j] / np.median(norms)),
            "separability": float(hit.mean()),
            "separability_per_action": {name[q]: float(hit[q].mean()) for q in range(len(acts))},
            "n_fit": int(is_fit.sum()), "n_held": int((~is_fit).sum())}
    return out


# ---------------------------------------------------------------- decodability


def ridge_fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + alpha * len(X) * np.eye(X.shape[1]), Xs.T @ (Y - ym))
    return mu, sd, W, ym


def ridge_pred(fit, X):
    mu, sd, W, ym = fit
    return ((X - mu) / sd) @ W + ym


def r2_dims(Y, P) -> np.ndarray:
    sst = ((Y - Y.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ((Y - P) ** 2).sum(0) / np.where(sst > 0, sst, np.nan)
    return np.where(sst > 1e-18 * len(Y), r2, np.nan)


def arm_r2(r2: np.ndarray) -> dict:
    return {arm: (float(np.nanmean(r2[ix])) if np.isfinite(r2[ix]).any() else None)
            for arm, ix in ARMS.items()}


def decodability(eps, zs, z_std, fit, k: int, seed: int) -> dict:
    X, Y, ph, ep = [], [], [], []
    zs_np = [z.numpy() / z_std for z in zs]
    for i, e in enumerate(eps):
        T = len(e["seg"])
        t = np.arange(T - k)
        t = t[~e["empty"][t] & ~e["empty"][t + k]]
        if not len(t):
            continue
        X.append(zs_np[i][t + k] - zs_np[i][t])
        cs = np.vstack([np.zeros(12), np.cumsum(e["u"], axis=0)])
        Y.append(cs[t + k] - cs[t])
        ph.append(e["phase"][t])
        ep.append(np.full(len(t), i))
    X, Y, ph, ep = (np.concatenate(v) for v in (X, Y, ph, ep))
    tr = fit[ep]
    tr_eps = np.flatnonzero(fit)
    inner = np.random.default_rng(seed).permutation(tr_eps)[: max(1, len(tr_eps) // 5)]
    iv = np.isin(ep, inner)
    a_tr, a_va = tr & ~iv, tr & iv
    scores = [np.nanmean(r2_dims(Y[a_va], ridge_pred(ridge_fit(X[a_tr], Y[a_tr], al), X[a_va])))
              for al in ALPHAS]
    alpha = float(ALPHAS[int(np.nanargmax(scores))])
    model = ridge_fit(X[tr], Y[tr], alpha)
    te = ~tr
    P = ridge_pred(model, X[te])
    r2 = r2_dims(Y[te], P)
    out = {"alpha": alpha, "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "all": arm_r2(r2), "per_dim": {d: (None if math.isnan(v) else float(v))
                                          for d, v in zip(DIMS, r2, strict=True)}}
    for p in PHASES:
        m = ph[te] == p
        r = arm_r2(r2_dims(Y[te][m], P[m])) if m.sum() > 20 else {"Franka": None, "UR": None}
        out[f"phase {p}"] = r | {"n": int(m.sum())}
    return out


# ---------------------------------------------------------------- report


def report_md(doc: dict) -> str:
    names = [m["name"] for m in doc["models"]]
    md = ["# Encoder evaluation (`encoder_eval.py`)", "",
          "Models: " + "; ".join(f"{m['name']} = `{rel(m['deploy'])}`"
                                  + (f" ({m['model_id']})" if m.get("model_id") else "")
                                  for m in doc["models"]) + ".",
          "Sets: " + "; ".join(f"{k} {v['episodes']} episodes ({v['groups']}), "
                               f"{v['starts']} start frames" for k, v in doc["sets"].items())
          + f". Every {doc['args']['frame_stride']} frames, H {doc['args']['horizon']}.", ""]
    ov = [f"{m['name']} {k} {n}" for m in doc["models"]
          for k, n in m.get("train_overlap", {}).items() if n]
    md += [f"Training-buffer overlap: {', '.join(ov) if ov else 'none'}.", ""]
    for s in doc["sets"]:
        pr = doc["prediction"][s]
        md += [f"## {s}: prediction (Chamfer RMS mm: one-step / H-step / recon)", "",
               "| split | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for k in pr[names[0]]:
            md.append(f"| {k} | {pr[names[0]][k]['n']} | " + " | ".join(
                f"{f(pr[nm][k]['one'])} / {f(pr[nm][k]['hk'])} / {f(pr[nm][k]['rec'])}"
                for nm in names) + " |")
        md += ["", (f"## {s}: probes ({doc['args']['probe_steps']} steps x "
                    f"{doc['args']['probe_mm']:g} mm; dd = change of the whitened seated-goal "
                    "distance minus the zero-input change; worse = dd > 0)"), "",
               "| model | states | n | d0 | zero dd | " + " | ".join(
                   f"{p} dd (worse)" for p in PROBES) + " |",
               "|---|---|---|---|---|" + "---|" * len(PROBES)]
        for nm in names:
            for grp, r in doc["probes"][s][nm].items():
                md.append(f"| {nm} | {grp} | {r['n']} | {f(r['d0_median'])} | "
                          f"{f(r['zero_dd_median'])} | " + " | ".join(
                              f"{f(r[p]['dd_median'], '+.2f')} ({f(r[p]['frac_worse'], '.0%')})"
                              for p in PROBES) + " |")
        md.append("")
    ar = doc.get("action_response")
    if ar:
        sc = doc["ar_scale"]
        md += [(f"## E_off: action response (24 signed unit actions, {doc['args']['ar_scale']} "
                f"scale: " + ", ".join(f"{d} {v * (1e3 if i < 6 else 180 / math.pi):.2f}"
                                       for i, (d, v) in enumerate(zip(DIMS, sc, strict=True)))
                + " mm / deg; whitened response norm per dim, min pairwise distance of the 24 "
                "mean responses, separability = held-out nearest-own-mean)"), "",
               ("| model | k | " + " | ".join(DIMS)
                + " | min pair (actions) | min pair / median norm | separability |"),
               "|---|---|" + "---|" * len(DIMS) + "---|---|---|"]
        for nm in names:
            for k, r in ar[nm].items():
                md.append(f"| {nm} | {k[1:]} | " + " | ".join(
                    f(r["norm_per_dim"][d], ".3f") for d in DIMS)
                    + f" | {f(r['min_pair_dist'], '.4f')} ({' / '.join(r['min_pair'])}) | "
                    f"{f(r['min_pair_rel'], '.3f')} | {f(r['separability'], '.1%')} |")
        md.append("")
    dc = doc.get("decodability")
    if dc:
        md += [("## E_off: linear action decodability (ridge Dz_k -> summed actions; held-out "
                "episodes; R^2 = mean over the arm's 6 dims, Franka / UR)"), "",
               "| model | k | alpha | all | " + " | ".join(PHASES) + " |",
               "|---|---|---|---|" + "---|" * len(PHASES)]
        for nm in names:
            for k, r in dc[nm].items():
                md.append(f"| {nm} | {k[1:]} | {r['alpha']:g} | {f(r['all']['Franka'])} / "
                          f"{f(r['all']['UR'])} | " + " | ".join(
                              f"{f(r[f'phase {p}']['Franka'])} / {f(r[f'phase {p}']['UR'])} "
                              f"(n {r[f'phase {p}']['n']})" for p in PHASES) + " |")
        md.append("")
    if doc.get("behaviour"):
        b = doc["behaviour"]
        md += [f"## Behaviour (seed_compare; `{b['mppi_config']}`, stride {b['stride']})", "",
               "| model | cos approach | cos descent | cos hold | hold push mm/step |",
               "|---|---|---|---|---|"]
        for nm in names:
            r = b["models"][nm]
            md.append(f"| {nm} | {f(r['cos_median']['approach'], '.3f')} | "
                      f"{f(r['cos_median']['descent'], '.3f')} | "
                      f"{f(r['cos_median']['hold'], '.3f')} | {f(r['hold_push_mm'], '+.2f')} |")
        md.append("")
    md += ["## Runtime (s)", "", "| model | " + " | ".join(doc["timing"][names[0]]) + " |",
           "|---|" + "---|" * len(doc["timing"][names[0]])]
    for nm in names:
        md.append(f"| {nm} | " + " | ".join(f"{v:.0f}" for v in doc["timing"][nm].values())
                  + " |")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--eval-sets", default=f"{STUDY}/tiers/eval_sets.json")
    ap.add_argument("--sets", default="E_off,E_hs")
    ap.add_argument("--train-buffer", action="append", default=[], help="NAME=buffer.json")
    ap.add_argument("--export-buffer", action="append", default=[],
                    help="NAME=buffer.json of a checkpoint model (export data files)")
    ap.add_argument("--export-box", default=f"{CMP}/v12_18/iter0/box_0.json")
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--max-episodes", type=int, default=None, help="per set (seeded pick)")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--probe-steps", type=int, default=16)
    ap.add_argument("--probe-mm", type=float, default=3.0)
    ap.add_argument("--ar-starts", type=int, default=1000)
    ap.add_argument("--ar-steps", default="1,5")
    ap.add_argument("--ar-scale", choices=("fixed", "std"), default="fixed")
    ap.add_argument("--ar-std-buffer", default=f"{STUDY}/tiers/T1/buffer.json")
    ap.add_argument("--dec-k", default="1,5")
    ap.add_argument("--behaviour", action="store_true")
    ap.add_argument("--bg-stride", type=int, default=8)
    ap.add_argument("--bg-demos", type=int, default=None)
    ap.add_argument("--mppi-config", default=f"{CMP}/noprior/configs/mppi_noprior.yaml")
    ap.add_argument("--keep-clearance", action="store_true")
    ap.add_argument("--u-box-json", default=None)
    ap.add_argument("--wp-min-frame", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    out = (REPO / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = [resolve_model(s, a, out) for s in a.model]
    if len({m["name"] for m in models}) != len(models):
        raise SystemExit("model names must be unique")
    es = jload(REPO / a.eval_sets)["sets"]
    sets = [s for s in a.sets.split(",") if s]
    for m in models:
        tb = ({r["episode"] for r in jload(m["train_buffer"])["rollouts"]}
              if m["train_buffer"] else set())
        m["train_overlap"] = {s: sum(r["episode"] in tb for r in es[s]) for s in sets}
    rng = np.random.default_rng(a.seed)
    demo_p = demo_points()
    scale = action_scale(a)
    doc = {"args": vars(a), "models": models, "sets": {}, "prediction": {}, "probes": {},
           "ar_scale": scale.tolist(), "timing": {m["name"]: {} for m in models}}
    mods = {m["name"]: Model(m) for m in models}
    for s in sets:
        t1 = time.perf_counter()
        eps = load_set(es[s], a, demo_p, rng)
        doc["sets"][s] = {"episodes": len(eps), "starts": int(sum(len(e["starts"]) for e in eps)),
                          "groups": ", ".join(f"{g} {sum(e['group'] == g for e in eps)}"
                                              for g in dict.fromkeys(e["group"] for e in eps)),
                          "load_s": time.perf_counter() - t1}
        print(f"{s}: {doc['sets'][s]}", flush=True)
        states = probe_states(eps)
        doc["prediction"][s], doc["probes"][s] = {}, {}
        if s == "E_off":
            fit = episode_split(eps, a.seed)
            pool = [(i, t) for i, e in enumerate(eps) for t in np.flatnonzero(~e["empty"])]
            pick = np.random.default_rng(a.seed).choice(len(pool), min(a.ar_starts, len(pool)),
                                                        replace=False)
            starts = [pool[q] for q in sorted(pick)]
            doc["action_response"], doc["decodability"] = {}, {}
        for nm, mod in mods.items():
            tm = time.perf_counter()
            doc["prediction"][s][nm] = split_table(prediction(mod, eps, a.horizon))
            doc["probes"][s][nm] = probes(mod, states, a.probe_steps, a.probe_mm)
            doc["timing"][nm][f"{s} prediction + probes"] = time.perf_counter() - tm
            al = doc["prediction"][s][nm]["all"]
            print(f"  {nm} {s}: one-step {f(al['one'])} / H{a.horizon} {f(al['hk'])} / recon "
                  f"{f(al['rec'])} mm ({time.perf_counter() - tm:.0f} s)", flush=True)
            if s != "E_off":
                continue
            tm = time.perf_counter()
            zs = [mod.encode(e["seg"]) for e in eps]
            doc["action_response"][nm] = action_response(
                mod, eps, zs, starts, fit, scale, [int(k) for k in a.ar_steps.split(",")])
            doc["decodability"][nm] = {f"k{k}": decodability(eps, zs, mod.lcs.z_std_np, fit,
                                                             int(k), a.seed)
                                       for k in a.dec_k.split(",")}
            doc["timing"][nm]["E_off action probes"] = time.perf_counter() - tm
            print(f"  {nm} action: " + ", ".join(
                f"{k} sep {r['separability']:.2f} minpair {r['min_pair_dist']:.3f}"
                for k, r in doc["action_response"][nm].items()) + "; decode " + ", ".join(
                f"{k} {r['all']}" for k, r in doc["decodability"][nm].items()), flush=True)
        del eps
    if a.behaviour:
        doc["behaviour"] = behaviour(models, a, out)
    doc["wall_s"] = time.perf_counter() - t0
    (out / "report.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    md = report_md(doc)
    (out / "report.md").write_text(md)
    print(md)
    print(f"wrote {rel(out / 'report.md')} ({doc['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
