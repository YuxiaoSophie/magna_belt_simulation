#!/usr/bin/env python3
"""Offline model gates M1-M3 for exported latent-LCS deploys on recorded demos (no sim).

M1 demo-action open-loop rollout: z_0 = encode(frame t) (numpy twin on ``pcd_belt_seg``), the
demo's own ``actions`` through ``TorchLcs.rollout`` (p0 from proprio, rot0 zeros) in windows of
H from every 4th frame and over the full episode from frame 0; whitened error
``||z_pred - z_enc|| / sqrt(n_x)`` at k = H vs the no-motion baseline ``||z_enc(t) -
z_enc(t+H)||``; fraction of windows whose predicted seated-goal distance at k = H is within
+-25 % of the encoded truth's. ``active`` = windows starting before the first ``done`` frame.
M2 latent monotonicity: 5-frame moving median of the whitened distance of z_enc(t) to the
pre_place_2 goal (frames 0..latch), the seated goal (latch..first ``hold:place_3``) and the next
waypoint z_enc(10 (floor(t / 10) + 1)) (frames 0..first hold; dense-waypoint potential).
M3 policy screen: ``mppi_plan`` (seated goal, the yaml's K / n_iter, H swept, clearance on with
the frame's wxyz, box = deploy u_lb/u_ub or --u-box-json, capped by the outer box (node
``MppiParams.bounds``), cold start, ceilings / rot0 from the episode
start as the node) at every 4th frame before ``done``; cosine of u_0 translations (6-d) vs the
demo's mean action over the next 5 frames, by phase (approach < latch <= descent < first hold
<= hold); frames with a zero demo mean have no cosine (``n`` counts cosines only).

    uv run --frozen python scripts/lcs/model_gates.py --deploy <deploy.npz> [--deploy ...] \\
        --demos <npz ...> --mppi-config data/lcs/realfirst/v1_1/mppi.yaml --out <dir> \\
        [--horizons 8,16,24] [--pair WARM,FRESH] [--u-box-json LABEL=box.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from latent_stage_tols import latch_frame

from task_common.latent_encoder import (
    LatentDecoder,
    LatentEncoder,
    encoder_input,
)
from task_common.lcs_dataset import delta_rotvec
from task_common.mppi_lcs import MppiParams, TorchLcs, mppi_plan

STRIDE = 4
MED_WIN = 5
WP_STEP = 10
NEXT_U = 5
RATIO_TOL = 0.25
COS_GT = 0.5
PHASES = ("approach", "descent", "hold")
REFIT_KEYS = ("one_step_chamfer_mm", "open_loop7_chamfer_mm", "rho_A", "B_fro_w", "drift_u0_w",
              "separation_auc", "recon_chamfer_mm")
SCREEN_KEYS = ("deploy", "H", "phase", "n", "cos_median", "frac_cos_gt_0.5", "urz_sign_agree")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.2e}" if abs(x) >= 1e4 else f"{x:.{nd}f}"


def med(x) -> float | None:
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


# ---------------------------------------------------------------- demos


def load_demo(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        meta = json.loads(str(d["sim_meta"]))
        labels = [meta["phase_labels"][p] for p in np.asarray(d["sim_phase"], dtype=np.int64)]
        demo = {"path": str(path), "seg": list(d["pcd_belt_seg"]),
                "u": np.asarray(d["actions"], dtype=np.float64),
                "ee_f": np.asarray(d["sim_ee_franka"], dtype=np.float64),
                "ee_u": np.asarray(d["sim_ee_ur"], dtype=np.float64),
                "outcome": str(d["sim_outcome"]) if "sim_outcome" in d.files else None}
    demo["labels"] = labels
    demo["T"] = len(labels)
    demo["latch"] = latch_frame(labels)
    demo["hold0"] = labels.index("hold:place_3")
    demo["done0"] = labels.index("done") if "done" in labels else demo["T"]
    demo["P"] = np.c_[demo["ee_f"][:, :3], demo["ee_u"][:, :3]]
    demo["move0"] = int(np.flatnonzero(np.abs(demo["u"]).max(1) > 0.0)[0])
    return demo


def demo_name(path: str) -> str:
    p = Path(path)
    return f"{p.parent.parent.name}/{p.stem.replace('episode_', '')}"


def encode_demo(enc, demo: dict) -> np.ndarray:
    """Per-frame ``encode`` (the node's path, bit-exact)."""
    return np.stack([enc.encode(encoder_input(enc, {"pcd_belt_seg": s})) for s in demo["seg"]])


def phase_of(demo: dict, t: int) -> str:
    return "approach" if t < demo["latch"] else "descent" if t < demo["hold0"] else "hold"


# ---------------------------------------------------------------- M1


def wdist(a, b, z_std) -> np.ndarray:
    return np.linalg.norm((np.asarray(a) - np.asarray(b)) / z_std, axis=-1)


def rollout_np(lcs: TorchLcs, z0, p0, U) -> np.ndarray:
    """``(K, H+1, n_x)`` float64 of ``U (K, H, n_u)`` from per-window ``z0`` / ``p0``."""
    Z, _, _ = lcs.rollout(lcs.tensor(z0), lcs.tensor(p0), lcs.tensor(U), None)
    return Z.double().cpu().numpy()


def m1_demo(lcs: TorchLcs, Z: np.ndarray, demo: dict, H: int, goal) -> dict:
    T, z_std, sq = len(Z), lcs.z_std_np, math.sqrt(lcs.n_x)
    starts = np.arange(0, T - H, STRIDE)
    U = demo["u"][starts[:, None] + np.arange(H)]
    zH = rollout_np(lcs, Z[starts], demo["P"][starts], U)[:, -1]
    truth = Z[starts + H]
    err = wdist(zH, truth, z_std) / sq
    base = wdist(Z[starts], truth, z_std) / sq
    dp, dt = wdist(zH, goal, z_std), wdist(truth, goal, z_std)
    within = np.abs(dp - dt) <= RATIO_TOL * dt
    act = starts < demo["done0"]
    return {"n": len(starts), "n_active": int(act.sum()),
            "err_med": float(np.median(err)), "base_med": float(np.median(base)),
            "goal_within_frac": float(within.mean()),
            "err_med_active": float(np.median(err[act])),
            "base_med_active": float(np.median(base[act])),
            "goal_within_frac_active": float(within[act].mean())}


def m1_full(lcs: TorchLcs, Z: np.ndarray, demo: dict) -> dict:
    T, z_std, sq = len(Z), lcs.z_std_np, math.sqrt(lcs.n_x)
    Zp = rollout_np(lcs, Z[:1], demo["P"][:1], demo["u"][None, :T - 1])[0]
    e = wdist(Zp, Z, z_std) / sq
    d0 = min(demo["done0"], T - 1)
    return {"err_final": float(e[-1]), "err_done0": float(e[d0]), "done0": d0,
            "base_final": float(wdist(Z[0], Z[-1], z_std) / sq)}


def chamfer_rms_mm(dec: LatentDecoder, enc, Zpred: np.ndarray, segs: list) -> float:
    """``refit_from_buffer.evaluate`` one-step scale: sqrt(mean symmetric Chamfer^2) in mm of
    the decoded prediction vs the resampled recorded cloud."""
    acc = []
    for i in range(0, len(Zpred), 64):
        a = dec.decode_batch(Zpred[i:i + 64]).astype(np.float64)
        b = np.stack([enc.preprocess(s) for s in segs[i:i + 64]]).astype(np.float64)
        d2 = ((a[:, :, None] - b[:, None]) ** 2).sum(-1)
        acc.append(0.5 * (d2.min(2).mean(1) + d2.min(1).mean(1)))
    return 1e3 * float(np.sqrt(np.concatenate(acc).mean()))


def m1_sanity(lcs: TorchLcs, enc, dec, Zs: list, demos: list) -> dict:
    """H = 1 on every frame pair: whitened error and (with a decoder) decoded Chamfer RMS."""
    errs, preds, segs = [], [], []
    for Z, demo in zip(Zs, demos):
        t = np.arange(len(Z) - 1)
        z1 = rollout_np(lcs, Z[t], demo["P"][t], demo["u"][t, None])[:, -1]
        errs.append(wdist(z1, Z[t + 1], lcs.z_std_np) / math.sqrt(lcs.n_x))
        preds.append(z1)
        segs += [demo["seg"][k + 1] for k in t]
    e = np.concatenate(errs)
    out = {"n": len(e), "err_med": float(np.median(e)), "err_mean": float(e.mean())}
    if dec is not None:
        out["chamfer_rms_mm"] = chamfer_rms_mm(dec, enc, np.concatenate(preds), segs)
    return out


# ---------------------------------------------------------------- M2


def movmed(x: np.ndarray, w: int = MED_WIN) -> np.ndarray:
    """Centred moving median, edge-padded."""
    if len(x) == 0:
        return x
    h = w // 2
    xp = np.pad(x, h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, w), axis=1)


def mono(s: np.ndarray, first: int = 0) -> tuple[int, int, float]:
    """(non-increasing steps, steps, max rise above the running minimum) over ``s[first:]``."""
    s = s[max(first, 0):]
    if len(s) < 2:
        return 0, 0, 0.0
    return (int(np.sum(np.diff(s) <= 0.0)), len(s) - 1,
            float(np.max(s - np.minimum.accumulate(s))))


def m2_demo(Z: np.ndarray, demo: dict, g1, g2, z_std) -> dict:
    """``*_move``: the same from the first moving frame (the static prehold cloud jitters)."""
    mv = demo["move0"]
    d1 = movmed(wdist(Z, g1, z_std))[:demo["latch"] + 1]
    d2 = movmed(wdist(Z, g2, z_std))[demo["latch"]:demo["hold0"] + 1]
    out = {"demo": demo_name(demo["path"]), "latch": demo["latch"], "hold0": demo["hold0"],
           "move0": mv}
    for key, d, first in (("pp2", d1, mv), ("seated", d2, mv - demo["latch"])):
        k, n, u = mono(d)
        km, nm, um = mono(d, first)
        out.update({f"{key}_frac": k / n, f"{key}_uphill": u, f"{key}_frac_move": km / nm,
                    f"{key}_uphill_move": um})
    out.update(pp2_d0=float(d1[0]), pp2_dlatch=float(d1[-1]), seated_dlatch=float(d2[0]),
               seated_dhold=float(d2[-1]))
    acc = np.zeros((2, 2), np.int64)
    up = [0.0, 0.0]
    for a in range(0, demo["hold0"] + 1, WP_STEP):
        b = min(a + WP_STEP, demo["hold0"] + 1)
        s = movmed(wdist(Z[a:b], Z[min(a + WP_STEP, len(Z) - 1)], z_std))
        for j, first in enumerate((0, mv - a)):
            k, n, u = mono(s, first)
            acc[j] += (k, n)
            up[j] = max(up[j], u)
    out.update(wp_frac=acc[0, 0] / acc[0, 1], wp_uphill=up[0],
               wp_frac_move=acc[1, 0] / acc[1, 1], wp_uphill_move=up[1])
    return out


# ---------------------------------------------------------------- M3


def demo_next_mean(u: np.ndarray, t: int, n: int = NEXT_U) -> np.ndarray:
    """Mean of the demo's translation actions ``u[t:t+n, :6]``."""
    return np.asarray(u[t:t + n, :6], dtype=np.float64).mean(0)


def cosine(a, b, eps: float = 1e-12) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return None if na < eps or nb < eps else float(np.dot(a, b) / (na * nb))


def screen_rows(deploy: str, H: int, u0: np.ndarray, ud: np.ndarray, phases: list[str],
                extra: dict | None = None) -> list[dict]:
    """Shared C14 / M3 metric: per phase ``{deploy, H, phase, n, cos_median, frac_cos_gt_0.5,
    urz_sign_agree}`` from MPPI ``u0 (N, >=6)`` and the demo means ``ud (N, 6)``; ``n`` counts
    the frames with a defined cosine. ``extra``: per-frame arrays averaged into the row."""
    u0, ud, ph = np.asarray(u0)[:, :6], np.asarray(ud)[:, :6], np.asarray(phases)
    rows = []
    for phase in (*PHASES, "all"):
        m = np.ones(len(ph), bool) if phase == "all" else ph == phase
        cs = [cosine(a, b) for a, b in zip(u0[m], ud[m])]
        ok = np.array([c is not None for c in cs], bool)
        c = np.array([x for x in cs if x is not None])
        sel_u, sel_d = u0[m][ok], ud[m][ok]
        zmask = np.abs(sel_d[:, 5]) > 1e-12
        row = {"deploy": deploy, "H": int(H), "phase": phase, "n": int(ok.sum()),
               "cos_median": float(np.median(c)) if len(c) else None,
               "frac_cos_gt_0.5": float(np.mean(c > COS_GT)) if len(c) else None,
               "urz_sign_agree": float(np.mean(np.sign(sel_u[zmask, 5])
                                               == np.sign(sel_d[zmask, 5])))
               if zmask.any() else None,
               "n_solved": int(m.sum()),
               "u0_mean_mm": (1e3 * sel_u.mean(0)).round(3).tolist() if len(c) else None,
               "demo_mean_mm": (1e3 * sel_d.mean(0)).round(3).tolist() if len(c) else None}
        for k, v in (extra or {}).items():
            v = np.asarray(v, dtype=np.float64)[m]
            row[k] = float(v.mean()) if len(v) else None
        rows.append(row)
    return rows


def m3_deploy(label: str, lcs: TorchLcs, Zs: list, demos: list, base: MppiParams, H: int,
              goal, seed: int, box: dict | None = None) -> tuple[list[dict], dict]:
    """``box`` {u_lb, u_ub}: replaces the deploy's input bounds (then the outer box, as the
    node with a deploy exported with ``--u-bounds-json``)."""
    over = {} if box is None else {"u_lb": list(box["u_lb"]), "u_ub": list(box["u_ub"])}
    params = dataclasses.replace(base, H=H, action_noise_frac=0.0, **over)
    gen = torch.Generator(device=lcs.device).manual_seed(seed)
    c = params.z_ceiling_above_start_m
    u0s, uds, phs, viol0, fb, ms, u0n = [], [], [], [], [], [], []
    for Z, demo in zip(Zs, demos):
        ps, qs = demo["P"][0], (demo["ee_f"][0, 3:7], demo["ee_u"][0, 3:7])
        prm = dataclasses.replace(params, z_max_franka=min(params.z_max_franka, ps[2] + c),
                                  z_max_ur=min(params.z_max_ur, ps[5] + c))
        for t in range(0, demo["done0"], STRIDE):
            q = np.stack([demo["ee_f"][t, 3:7], demo["ee_u"][t, 3:7]])
            rot0 = np.stack([delta_rotvec(qs[0], q[0]), delta_rotvec(qs[1], q[1])])
            kw = {"q_now": q} if prm.clearance is not None else {}
            U, info = mppi_plan(lcs, prm, Z[t], demo["P"][t], rot0, goal, None, gen, **kw)
            u0s.append(U[0])
            uds.append(demo_next_mean(demo["u"], t))
            phs.append(phase_of(demo, t))
            viol0.append(float(info["plan_violations"] == 0))
            fb.append(float(info["fallback"]))
            ms.append(info["solve_ms"])
            u0n.append(1e3 * float(np.linalg.norm(U[0, :6])))
    rows = screen_rows(label, H, np.asarray(u0s), np.asarray(uds), phs,
                       {"plan_viol0_frac": viol0, "fallback_frac": fb, "u0_trans_mm": u0n})
    lb, ub = params.bounds(lcs)
    meta = {"n_solves": len(u0s), "plan_viol0_frac": float(np.mean(viol0)),
            "solve_ms_p50": float(np.percentile(ms[1:] or ms, 50)),
            "solve_ms_p95": float(np.percentile(ms[1:] or ms, 95)),
            "box_ur_z_mm": [1e3 * lb[5], 1e3 * ub[5]], "box_f_z_mm": [1e3 * lb[2], 1e3 * ub[2]]}
    return rows, meta


# ---------------------------------------------------------------- deploys


def parse_deploy(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    p = Path(spec)
    d = p.parent.name
    label = (("V" + d[len("deploy_v"):]) if d.startswith("deploy_v") else
             p.parent.parent.name if d == "deploy" else d)
    return label, p


def refit_report_for(deploy: Path) -> Path | None:
    """``<tag>/deploy/deploy.npz`` -> ``<tag>/refit_report.json``; loop deploys
    ``iter<i>/deploy_v11x`` -> ``iter<i>/refit/*/refit_report.json``."""
    root = deploy.parent.parent
    if (root / "refit_report.json").is_file():
        return root / "refit_report.json"
    hits = sorted(root.glob("refit/*/refit_report.json"))
    return hits[0] if len(hits) == 1 else None


def deploy_info(label: str, path: Path, lcs: TorchLcs) -> dict:
    rp = refit_report_for(path)
    info = {"label": label, "deploy": str(path), "sha256": sha256(path), "n_x": lcs.n_x,
            "refit_report": str(rp) if rp else None}
    if rp:
        rep = json.loads(rp.read_text())
        info["refit"] = {k: rep["metrics"].get(k) for k in REFIT_KEYS}
        info["refit"].update(epochs=rep.get("epochs"), init=rep.get("init_checkpoint"),
                             decision=rep.get("decision"), buffer=rep.get("buffer"),
                             heldout=[demo_name(x) for x in rep["split"]["heldout"]])
    return info


# ---------------------------------------------------------------- report


def reading(doc: dict, pairs: list[tuple[str, str]]) -> dict:
    """The six-line reading; recommendation rules from the package (the first (warm, fresh)
    pair decides, the others are reported)."""
    M1, M2, M3 = doc["M1"], doc["M2"], doc["M3"]
    hs = doc["horizons"]
    h16 = 16 if 16 in hs else hs[len(hs) // 2]

    def m1(lab, key="err_med", H=h16):
        return M1[lab]["windows"][str(H)][key]

    def m3(lab, H=h16, phase="approach"):
        r = [x for x in M3["rows"] if x["deploy"] == lab and x["H"] == H and x["phase"] == phase]
        return r[0]["cos_median"] if r else None

    labs = list(M1)
    out = {}
    e = {lab: m1(lab) for lab in labs}
    b = {lab: m1(lab, "base_med") for lab in labs}
    ea = {lab: m1(lab, "err_med_active") for lab in labs}
    ba = {lab: m1(lab, "base_med_active") for lab in labs}
    out["i_reproduce"] = (
        f"M1 H{h16} whitened error / no-motion baseline (all windows; active windows): "
        + "; ".join(f"{lab} {e[lab]:.3f}/{b[lab]:.3f} ({ea[lab]:.3f}/{ba[lab]:.3f})"
                    for lab in labs)
        + f". Models beating the baseline on active windows: "
        f"{sum(ea[lab] < ba[lab] for lab in labs)}/{len(labs)}.")
    med2 = {lab: M2[lab]["median"] for lab in labs}
    out["ii_potential"] = (
        "M2 median monotone fraction, strict over 0..latch / latch..hold / waypoints "
        "(moving frames only), max uphill: "
        + "; ".join(f"{lab} {m['pp2_frac']:.2f}/{m['seated_frac']:.2f}/{m['wp_frac']:.2f} "
                    f"({m['pp2_frac_move']:.2f}/{m['seated_frac_move']:.2f}/"
                    f"{m['wp_frac_move']:.2f}), uphill {m['pp2_uphill']:.2f}/"
                    f"{m['seated_uphill']:.2f}/{m['wp_uphill']:.2f}"
                    for lab, m in med2.items()))
    out["iii_policy"] = (
        f"M3 approach cos_median at H{h16} (seated goal; frac > 0.5; UR-z sign agree): "
        + "; ".join(
            f"{r['deploy']} {fnum(r['cos_median'], 2)} ({fnum(r['frac_cos_gt_0.5'], 2)}; "
            f"{fnum(r['urz_sign_agree'], 2)})" for r in M3["rows"]
            if r["H"] == h16 and r["phase"] == "approach"))
    pairs = [(w, f) for w, f in pairs if w in M1 and f in M1]
    lines = []
    for j, (warm, fresh) in enumerate(pairs):
        ew, ef = e[warm], e[fresh]
        cw, cf = m3(warm), m3(fresh)
        wins = bool(ef <= 0.8 * ew and cf is not None and cw is not None and cf >= cw)
        lines.append(
            f"{fresh} vs {warm}: M1 H{h16} {ef:.3f} vs {ew:.3f} (ratio {ef / ew:.2f}, need "
            f"<= 0.80; active windows {ea[fresh]:.3f} vs {ea[warm]:.3f}); M3 approach cos "
            f"{fnum(cf, 2)} vs {fnum(cw, 2)} (need >=) -> fresh "
            f"{'wins' if wins else 'does not win'}")
        if j == 0:
            out["fresh_each_iter"] = wins
            out["rule_pair"] = [warm, fresh]
    if lines:
        out["iv_fresh_vs_warm"] = "; ".join(lines) + " (first pair decides)."
    deltas = {}
    for H in hs:
        if H == h16:
            continue
        d = [m3(lab, H) - m3(lab) for lab in labs
             if m3(lab, H) is not None and m3(lab) is not None]
        deltas[H] = med(d)
    best = max(deltas, key=lambda h: deltas[h] if deltas[h] is not None else -9, default=None)
    change = best is not None and deltas[best] is not None and deltas[best] >= 0.15
    out["v_horizon"] = (
        f"M3 approach cos_median, median over deploys of (H - H{h16}): "
        + "; ".join(f"H{h} {fnum(v, 3)}" for h, v in deltas.items())
        + "; M1 error by H: " + "; ".join(
            f"{lab} " + "/".join(f"{m1(lab, 'err_med_active', h):.2f}" for h in hs)
            for lab in labs) + " (active windows, H " + "/".join(map(str, hs)) + ").")
    out["H"] = int(best) if change else h16
    out["vi_recommendation"] = (
        f"refit.fresh_each_iter = {str(out.get('fresh_each_iter', 'n/a')).lower()} (rule: "
        f"fresh M1 H{h16} <= 0.8 x warm AND fresh M3 approach cos >= warm); H = {out['H']} "
        f"(rule: change only if approach cos improves >= 0.15 over H{h16}).")
    out["horizon_deltas"] = {str(h): v for h, v in deltas.items()}
    return out


def m3_get(doc: dict, lab: str, H: int, phase: str, key: str):
    r = [x for x in doc["M3"]["rows"] if x["deploy"] == lab and x["H"] == H
         and x["phase"] == phase]
    return r[0][key] if r else None


def summary_table(doc: dict) -> list[str]:
    """Refit metrics side by side with M1-M3 (H 16 or the middle horizon)."""
    hs = doc["horizons"]
    h = 16 if 16 in hs else hs[len(hs) // 2]
    L = [(f"| deploy | one-step mm | h7 mm | rho_A | B_fro_w | drift | AUC | M1 H{h} err all "
          f"/ active | M2 pp2 / seated / wp (moving) | M3 H{h} cos approach / descent | M3 "
          f"H{h} UR-z agree approach / descent |"), "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, d in doc["deploys"].items():
        r, w = d.get("refit", {}), doc["M1"][lab]["windows"][str(h)]
        m2 = doc["M2"][lab]["median"]
        L.append(f"| {lab} | {fnum(r.get('one_step_chamfer_mm'), 2)} | "
                 f"{fnum(r.get('open_loop7_chamfer_mm'), 2)} | {fnum(r.get('rho_A'), 4)} | "
                 f"{fnum(r.get('B_fro_w'), 1)} | {fnum(r.get('drift_u0_w'), 3)} | "
                 f"{fnum(r.get('separation_auc'), 2)} | {w['err_med']:.3f} / "
                 f"{w['err_med_active']:.3f} | {m2['pp2_frac_move']:.2f} / "
                 f"{m2['seated_frac']:.2f} / {m2['wp_frac_move']:.2f} | "
                 + " / ".join(fnum(m3_get(doc, lab, h, ph, "cos_median"), 3)
                              for ph in ("approach", "descent")) + " | "
                 + " / ".join(fnum(m3_get(doc, lab, h, ph, "urz_sign_agree"), 2)
                              for ph in ("approach", "descent")) + " |")
    return L


def m3_metric_table(doc: dict, key: str, nd: int) -> list[str]:
    hs, ph = doc["horizons"], ("approach", "descent")
    L = ["| deploy | " + " | ".join(f"H{h} {p}" for h in hs for p in ph) + " |",
         "|---|" + "---|" * (len(hs) * len(ph))]
    for lab in doc["deploys"]:
        L.append(f"| {lab} | " + " | ".join(fnum(m3_get(doc, lab, h, p, key), nd)
                                            for h in hs for p in ph) + " |")
    return L


def md_report(doc: dict) -> str:
    hs = doc["horizons"]
    L = [f"# Offline model gates (W1-A, {doc['created']})", "",
         (f"Script `scripts/lcs/model_gates.py`; {len(doc['demos'])} clean demos; MPPI config "
         f"`{doc['mppi_config']}` (K {doc['mppi']['K']}, n_iter {doc['mppi']['n_iter']}, "
         f"clearance {'on' if doc['mppi']['clearance'] else 'off'}); stride {STRIDE}; wall "
         f"{doc['wall_s']:.0f} s. Whitened = divided by the deploy's z_std; M1 errors "
         "/ sqrt(n_x)."),
         "", "## Deploys", "",
         ("| deploy | sha | one-step mm | h7 mm | rho_A | B_fro_w | drift | AUC | recon mm | "
         "epochs | M3 box | box UR z mm |"), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, d in doc["deploys"].items():
        r = d.get("refit", {})
        bx = doc["M3"]["meta"].get(f"{lab}@{hs[0]}", {}).get("box_ur_z_mm", [None, None])
        L.append(f"| {lab} | {d['sha256'][:8]} | {fnum(r.get('one_step_chamfer_mm'), 2)} | "
                 f"{fnum(r.get('open_loop7_chamfer_mm'), 2)} | {fnum(r.get('rho_A'), 4)} | "
                 f"{fnum(r.get('B_fro_w'), 1)} | {fnum(r.get('drift_u0_w'), 3)} | "
                 f"{fnum(r.get('separation_auc'), 2)} | {fnum(r.get('recon_chamfer_mm'), 2)} | "
                 f"{r.get('epochs', '-')} | {Path(d.get('u_box_json') or 'deploy').name} | "
                 f"[{fnum(bx[0], 2)}, {fnum(bx[1], 2)}] |")
    L += ["", ("Refit metrics are each model's own held-out split (the M4 refits share one split; "
          "the v1.1 rows do not)."), ""]
    if doc["M3"]["rows"]:
        L += ["## Summary (refit metrics side by side with M1-M3)", "", *summary_table(doc), ""]
    L += [
          "## M1 demo-action open-loop rollout", "",
          ("Median over demos of the per-demo median over windows. err = model, base = "
          "no-motion; goal +-25 % = fraction of windows whose predicted seated-goal distance "
          "at k = H is within 25 % of the encoded truth's."), "",
          "### All windows (every 4th start, t + H <= T - 1)", "",
          "| deploy | " + " | ".join(f"H{h} err / base | H{h} goal +-25 %" for h in hs)
          + " | full: final | full: first done |",
          "|---|" + "---|---|" * len(hs) + "---|---|"]
    for lab, m in doc["M1"].items():
        w = m["windows"]
        L.append(f"| {lab} | " + " | ".join(
            f"{w[str(h)]['err_med']:.3f} / {w[str(h)]['base_med']:.3f} | "
            f"{w[str(h)]['goal_within_frac']:.2f}" for h in hs)
            + f" | {fnum(m['full']['err_final'])} (base {fnum(m['full']['base_final'])}) | "
            f"{fnum(m['full']['err_done0'])} |")
    L += ["", "### Active windows (start before the first `done` frame)", "",
          "| deploy | " + " | ".join(f"H{h} err / base | H{h} goal +-25 %" for h in hs) + " |",
          "|---|" + "---|---|" * len(hs)]
    for lab, m in doc["M1"].items():
        w = m["windows"]
        L.append(f"| {lab} | " + " | ".join(
            f"{w[str(h)]['err_med_active']:.3f} / {w[str(h)]['base_med_active']:.3f} | "
            f"{w[str(h)]['goal_within_frac_active']:.2f}" for h in hs) + " |")
    L += ["", "### Sanity: H 1 on every frame pair (reported, not gated)", "",
          ("| deploy | n | whitened err median | decoded Chamfer RMS mm | refit one-step mm "
          "(held-out) |"), "|---|---|---|---|---|"]
    for lab, m in doc["M1"].items():
        s = m["sanity_h1"]
        r = doc["deploys"][lab].get("refit", {})
        L.append(f"| {lab} | {s['n']} | {s['err_med']:.4f} | {fnum(s.get('chamfer_rms_mm'), 2)} "
                 f"| {fnum(r.get('one_step_chamfer_mm'), 2)} |")
    L += ["", "## M2 latent monotonicity along the demos", "",
          ("Fraction of non-increasing smoothed steps (5-frame moving median) and max uphill "
          "(rise above the running minimum, whitened). pp2 = `z_goal_stage1` over frames "
          "0..latch; seated = `z_goal_stage2` over latch..first `hold:place_3`; wp = the next "
          "waypoint z_enc(10 (floor(t / 10) + 1)) over 0..first hold. (moving) = from the "
          "first frame with a non-zero demo action (the static prehold cloud's segmentation "
          "jitter, M 394-404 points, moves d by ~0.03 frame to frame)."), "",
          "### Median over demos", "",
          ("| deploy | pp2 frac | pp2 frac (moving) | pp2 uphill | seated frac | seated uphill "
           "| wp frac | wp frac (moving) | wp uphill | wp uphill (moving) |"),
          "|---|---|---|---|---|---|---|---|---|---|"]
    keys = ("pp2_frac", "pp2_frac_move", "pp2_uphill", "seated_frac", "seated_uphill",
            "wp_frac", "wp_frac_move", "wp_uphill", "wp_uphill_move")
    for lab, m in doc["M2"].items():
        L.append(f"| {lab} | " + " | ".join(f"{m['median'][k]:.2f}" for k in keys) + " |")
    for lab, m in doc["M2"].items():
        L += ["", f"### {lab} per demo", "",
              ("| demo | latch | hold | pp2 frac (moving) | pp2 uphill | d0 -> d latch | "
               "seated frac | seated uphill | d latch -> d hold | wp frac (moving) | "
               "wp uphill |"),
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in m["per_demo"]:
            L.append(f"| {r['demo']} | {r['latch']} | {r['hold0']} | {r['pp2_frac']:.2f} "
                     f"({r['pp2_frac_move']:.2f}) | {r['pp2_uphill']:.2f} | "
                     f"{r['pp2_d0']:.2f} -> {r['pp2_dlatch']:.2f} | "
                     f"{r['seated_frac']:.2f} | {r['seated_uphill']:.2f} | "
                     f"{r['seated_dlatch']:.2f} -> {r['seated_dhold']:.2f} | {r['wp_frac']:.2f}"
                     f" ({r['wp_frac_move']:.2f}) | {r['wp_uphill']:.2f} |")
    L += ["", "## M3 policy screen on demo frames (seated goal, cold start)", "",
          ("cos = cosine of MPPI u_0 translations (F xyz, UR xyz) vs the demo's mean action over "
          "the next 5 frames; `n` = frames with a defined cosine (hold frames: the demo is "
          "still, so none); viol0 = fraction of solves with plan_violations == 0; |u0| mm = "
          "mean translation norm of u_0."), ""]
    for key, nd, name in (("cos_median", 3, "cos_median"), ("frac_cos_gt_0.5", 2,
                                                            "frac cos > 0.5"),
                          ("urz_sign_agree", 2, "UR-z sign agreement")):
        L += [f"### {name}", "", *m3_metric_table(doc, key, nd), ""]
    L += ["### Per-channel means at H16 (mm/step: F x, y, z, UR x, y, z; frames with a cosine)",
          "", "| deploy | phase | MPPI u_0 mean | demo next-5 mean |", "|---|---|---|---|"]
    for r in doc["M3"]["rows"]:
        if r["H"] == 16 and r["phase"] in ("approach", "descent") and r["u0_mean_mm"]:
            L.append(f"| {r['deploy']} | {r['phase']} | "
                     + ", ".join(f"{v:+.2f}" for v in r["u0_mean_mm"]) + " | "
                     + ", ".join(f"{v:+.2f}" for v in r["demo_mean_mm"]) + " |")
    L += ["", "### All rows", "",
          ("| deploy | H | phase | n | n solved | cos median | frac cos > 0.5 | UR-z sign agree | "
          "viol0 | fallback | mean |u0| mm |"), "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["M3"]["rows"]:
        L.append(f"| {r['deploy']} | {r['H']} | {r['phase']} | {r['n']} | {r['n_solved']} | "
                 f"{fnum(r['cos_median'], 3)} | {fnum(r['frac_cos_gt_0.5'], 2)} | "
                 f"{fnum(r['urz_sign_agree'], 2)} | {fnum(r['plan_viol0_frac'], 2)} | "
                 f"{fnum(r['fallback_frac'], 2)} | {fnum(r['u0_trans_mm'], 2)} |")
    L += ["", "| deploy @ H | solves | viol0 | solve p50 / p95 ms | box F z mm | box UR z mm |",
          "|---|---|---|---|---|---|"]
    for k, m in doc["M3"]["meta"].items():
        L.append(f"| {k} | {m['n_solves']} | {m['plan_viol0_frac']:.3f} | "
                 f"{m['solve_ms_p50']:.1f} / {m['solve_ms_p95']:.1f} | "
                 f"[{m['box_f_z_mm'][0]:.2f}, {m['box_f_z_mm'][1]:.2f}] | "
                 f"[{m['box_ur_z_mm'][0]:.2f}, {m['box_ur_z_mm'][1]:.2f}] |")
    rd = doc.get("reading") or {}
    L += ["", "## Reading", ""]
    for k in ("i_reproduce", "ii_potential", "iii_policy", "iv_fresh_vs_warm", "v_horizon",
              "vi_recommendation"):
        if k in rd:
            L.append(f"- ({k.split('_')[0]}) {rd[k]}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deploy", action="append", required=True,
                   help="deploy.npz or LABEL=deploy.npz (repeatable)")
    p.add_argument("--demos", type=Path, nargs="+", required=True)
    p.add_argument("--mppi-config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--horizons", default="8,16,24")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0, help="MPPI noise generator seed")
    p.add_argument("--pair", action="append", default=[],
                   help="WARM,FRESH deploy labels compared by the M4 rule (repeatable; the "
                        "first decides)")
    p.add_argument("--u-box-json", action="append", default=[],
                   help="LABEL=box.json: M3 input bounds {u_lb, u_ub} for that deploy label "
                        "instead of the deploy's (repeatable)")
    p.add_argument("--skip-m3", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    hs = [int(h) for h in a.horizons.split(",")]
    boxes = dict(x.split("=", 1) for x in a.u_box_json)
    demos = [load_demo(d) for d in a.demos]
    base = MppiParams.from_yaml(a.mppi_config)
    doc = {"created": time.strftime("%Y-%m-%d %H:%M"), "script": "scripts/lcs/model_gates.py",
           "argv": sys.argv[1:], "mppi_config": str(a.mppi_config), "horizons": hs,
           "mppi": {"K": base.K, "n_iter": base.n_iter, "clearance": base.clearance,
                    "u_box_outer": base.u_box_outer, "seed": a.seed},
           "demos": [{"episode": d["path"], "name": demo_name(d["path"]), "T": d["T"],
                      "latch": d["latch"], "hold0": d["hold0"], "done0": d["done0"],
                      "outcome": d["outcome"]} for d in demos],
           "deploys": {}, "M1": {}, "M2": {}, "M3": {"rows": [], "meta": {}}}
    for spec in a.deploy:
        label, path = parse_deploy(spec)
        lcs = TorchLcs.from_deploy(path, a.device)
        enc = LatentEncoder.load(path)
        dpath = path.parent / "decoder.npz"
        dec = LatentDecoder.load(dpath) if dpath.is_file() else None
        with np.load(path, allow_pickle=True) as d:
            g1, g2 = (np.asarray(d[k], np.float64) for k in ("z_goal_stage1", "z_goal_stage2"))
        doc["deploys"][label] = deploy_info(label, path, lcs)
        box = json.loads(Path(boxes[label]).read_text()) if label in boxes else None
        doc["deploys"][label]["u_box_json"] = boxes.get(label)
        te = time.time()
        Zs = [encode_demo(enc, d) for d in demos]
        print(f"[{label}] encoded {sum(len(z) for z in Zs)} frames in {time.time() - te:.1f} s",
              flush=True)
        z_std = lcs.z_std_np
        win = {}
        for H in sorted(set(hs) | {1}):
            per = [m1_demo(lcs, Z, d, H, g2) for Z, d in zip(Zs, demos)]
            win[str(H)] = {k: med([x[k] for x in per]) for k in per[0] if k not in ("n",
                                                                                   "n_active")}
            win[str(H)].update(n_windows=sum(x["n"] for x in per),
                               n_active=sum(x["n_active"] for x in per), per_demo=per)
        full = [m1_full(lcs, Z, d) for Z, d in zip(Zs, demos)]
        doc["M1"][label] = {
            "windows": win, "full": {k: med([x[k] for x in full]) for k in
                                     ("err_final", "err_done0", "base_final")},
            "full_per_demo": full, "sanity_h1": m1_sanity(lcs, enc, dec, Zs, demos)}
        per2 = [m2_demo(Z, d, g1, g2, z_std) for Z, d in zip(Zs, demos)]
        doc["M2"][label] = {"per_demo": per2, "median": {
            k: med([r[k] for r in per2]) for k in per2[0] if k not in ("demo", "latch", "hold0",
                                                                       "move0")}}
        print(f"[{label}] M1 H{hs[len(hs) // 2]} err {win[str(hs[len(hs) // 2])]['err_med']:.3f}"
              f" base {win[str(hs[len(hs) // 2])]['base_med']:.3f}; M2 median "
              f"{doc['M2'][label]['median']['pp2_frac']:.2f}/"
              f"{doc['M2'][label]['median']['seated_frac']:.2f}/"
              f"{doc['M2'][label]['median']['wp_frac']:.2f}", flush=True)
        if a.skip_m3:
            continue
        for H in hs:
            ts = time.time()
            rows, meta = m3_deploy(label, lcs, Zs, demos, base, H, g2, a.seed, box)
            doc["M3"]["rows"] += rows
            doc["M3"]["meta"][f"{label}@{H}"] = meta
            ap = next(r for r in rows if r["phase"] == "approach")
            print(f"[{label}] M3 H{H}: {meta['n_solves']} solves {time.time() - ts:.1f} s, "
                  f"approach cos {fnum(ap['cos_median'], 3)} (n {ap['n']}), viol0 "
                  f"{meta['plan_viol0_frac']:.3f}", flush=True)
    doc["wall_s"] = time.time() - t0
    if not a.skip_m3:
        doc["reading"] = reading(doc, [tuple(x.split(",", 1)) for x in a.pair])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "model_gates.json").write_text(json.dumps(doc, indent=1) + "\n")
    (a.out / "model_gates.md").write_text(md_report(doc))
    print(f"wrote {a.out / 'model_gates.json'} ({doc['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
