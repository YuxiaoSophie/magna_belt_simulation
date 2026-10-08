#!/usr/bin/env python3
"""Transport screen: does MPPI's first action agree with the demo on recorded demo frames?

On every ``STRIDE``-th frame (before ``done``) of each clean demo: the node's per-frame latent
(numpy-twin ``encode`` of ``pcd_belt_seg``), measured EE positions, ``rot0`` / ceiling relative
to the demo's frame 0 and ``q_now`` = the frame's quaternions (clearance on), cold start; one
``mppi_plan`` per config. Configs differ only in the goal and the demo action prior:
``baseline`` (deploy stage goal by phase: stage 1 before the latch, else seated), ``wp`` (the
first waypoint whose goal-demo frame is after the demo's own time index), ``track`` (per-knot
``z_traj[t+1 .. t+H]``), ``+prior`` (``u_traj[t .. t+H-1]`` as the sampling prior, alpha /
weight), and a secondary sweep of ``track+prior`` over H and alpha. Metric = W1-A's M3
(``model_gates.screen_rows``: cosine of u_0 translations vs the demo's mean action over the
next 5 frames, per phase) + ``frac_u0_near_demo`` (all 12 channels of u_0 within 10 % of the
half-box of the goal demo's box-clipped action at the same index: the open-loop replay guard)
+ the zero-violation plan rate. Routing rule (package W1-B): route to sim if a new config beats
baseline by >= 0.2 approach cos_median; ``track`` over ``wp`` only if its best primary config
beats ``wp``'s best by >= 0.1 there.

    uv run --frozen python scripts/lcs/transport_screen.py --deploy <deploy.npz> \\
        --config data/lcs/realfirst/v1_1/mppi.yaml --waypoints <latent_waypoints.npz> \\
        --out data/lcs/realfirst/cmp/offline/transport_screen
"""

from __future__ import annotations

import argparse
import dataclasses
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
from latent_stage_tols import sha256
from latent_waypoints import Waypoints
from model_gates import (
    STRIDE,
    demo_next_mean,
    encode_demo,
    load_demo,
    phase_of,
    screen_rows,
)

from task_common.latent_encoder import LatentEncoder
from task_common.lcs_dataset import delta_rotvec
from task_common.mppi_lcs import MppiParams, TorchLcs, mppi_plan

V1_1 = REPO_ROOT / "data" / "lcs" / "realfirst" / "v1_1"
DEMOS = tuple(sorted((V1_1 / "seed" / "demo").glob("episode_*.npz"))
              + sorted((V1_1 / "seed_ext" / "demo").glob("episode_*.npz")))
NEAR_FRAC = 0.1
ROUTE_MARGIN, TRACK_MARGIN = 0.2, 0.1
PRIMARY = ("baseline", "wp", "track", "wp+prior", "track+prior")
FAMILY = {"wp": ("wp", "wp+prior"), "track": ("track", "track+prior")}


@dataclasses.dataclass
class Config:
    name: str
    goal: str  # baseline | wp | track
    alpha: float = 0.0
    weight: float = 0.0
    H: int | None = None
    secondary: bool = False


def configs(alpha: float, weight: float) -> list[Config]:
    return [Config("baseline", "baseline"), Config("wp", "wp"), Config("track", "track"),
            Config("track+prior", "track", alpha, weight),
            Config("wp+prior", "wp", alpha, weight),
            Config("track+prior H8", "track", alpha, weight, 8, True),
            Config("track+prior H24", "track", alpha, weight, 24, True),
            Config("track+prior a0.25", "track", 0.25, weight, None, True),
            Config("track+prior a0.75", "track", 0.75, weight, None, True)]


def active_waypoint(wp: Waypoints, t: int) -> int:
    later = np.flatnonzero(wp.frames > t)
    return int(later[0]) if len(later) else len(wp.frames) - 1


def run_config(cfg: Config, base: MppiParams, lcs: TorchLcs, wp: Waypoints, demos, Zs,
               g1, g2, lb, ub, half, seed: int) -> tuple[list[dict], dict]:
    prm0 = dataclasses.replace(base, action_noise_frac=0.0, prior_alpha=cfg.alpha,
                               prior_weight=cfg.weight, **({"H": cfg.H} if cfg.H else {}))
    H = prm0.H
    gen = torch.Generator(device=lcs.device).manual_seed(seed)
    c = prm0.z_ceiling_above_start_m
    u0s, uds, phs, near, near6, viol0, ms, names, frames = [], [], [], [], [], [], [], [], []
    prior = cfg.weight > 0.0 or cfg.alpha > 0.0
    for Z, demo in zip(Zs, demos, strict=True):
        ps, qs = demo["P"][0], (demo["ee_f"][0, 3:7], demo["ee_u"][0, 3:7])
        prm = dataclasses.replace(prm0, z_max_franka=min(prm0.z_max_franka, ps[2] + c),
                                  z_max_ur=min(prm0.z_max_ur, ps[5] + c))
        for t in range(0, demo["done0"], STRIDE):
            q = np.stack([demo["ee_f"][t, 3:7], demo["ee_u"][t, 3:7]])
            rot0 = np.stack([delta_rotvec(qs[0], q[0]), delta_rotvec(qs[1], q[1])])
            if cfg.goal == "baseline":
                goal = g1 if t < demo["latch"] else g2
            elif cfg.goal == "wp":
                goal = wp.z_wp[active_waypoint(wp, t)]
            else:
                goal = wp.goal_seq(t, H)
            kw = {"q_now": q} if prm.clearance is not None else {}
            if prior:
                kw["u_demo"] = wp.u_seq(t, H)
            U, info = mppi_plan(lcs, prm, Z[t], demo["P"][t], rot0, goal, None, gen, **kw)
            ref = np.clip(wp.u_seq(t, 1)[0], lb, ub)
            u0s.append(U[0])
            uds.append(demo_next_mean(demo["u"], t))
            phs.append(phase_of(demo, t))
            close = np.abs(U[0] - ref) <= NEAR_FRAC * half
            near.append(float(np.all(close)))
            near6.append(float(np.all(close[:6])))
            viol0.append(float(info["plan_violations"] == 0))
            ms.append(info["solve_ms"])
            names.append(Path(demo["path"]).parent.parent.name + "/" + Path(demo["path"]).stem)
            frames.append(t)
    rows = screen_rows("V11d", H, np.asarray(u0s), np.asarray(uds), phs,
                       {"frac_u0_near_demo": near, "frac_u0_trans_near_demo": near6,
                        "plan_viol0_frac": viol0})
    sp = speed_ratio(np.asarray(u0s), np.asarray(uds))
    for r in rows:
        m = np.ones(len(phs), bool) if r["phase"] == "all" else np.asarray(phs) == r["phase"]
        v = sp[m][np.isfinite(sp[m])]
        r.update(config=cfg.name, alpha=cfg.alpha, weight=cfg.weight, secondary=cfg.secondary,
                 speed_ratio_median=float(np.median(v)) if len(v) else None)
    per_demo = {}
    for name in dict.fromkeys(names):
        m = np.array([n == name for n in names])
        sub = screen_rows("V11d", H, np.asarray(u0s)[m], np.asarray(uds)[m],
                          list(np.asarray(phs)[m]))
        per_demo[name] = {r["phase"]: {"n": r["n"], "cos_median": r["cos_median"]}
                          for r in sub if r["phase"] in ("approach", "descent")}
    meta = {"n_solves": len(u0s), "solve_ms_p50": float(np.percentile(ms[1:], 50)),
            "solve_ms_p95": float(np.percentile(ms[1:], 95)), "per_demo": per_demo,
            "frames": [[n, f] for n, f in zip(names, frames, strict=True)],
            "u0": np.asarray(u0s).round(7).tolist()}
    return rows, meta


def speed_ratio(u0: np.ndarray, ud: np.ndarray) -> np.ndarray:
    """``|u_0 xyz| / |demo mean xyz|`` per frame (nan where the demo holds)."""
    nd = np.linalg.norm(ud[:, :6], axis=1)
    out = np.full(len(nd), np.nan)
    ok = nd > 1e-12
    out[ok] = np.linalg.norm(u0[ok, :6], axis=1) / nd[ok]
    return out


def row_of(rows: list[dict], config: str, phase: str) -> dict:
    return next(r for r in rows if r["config"] == config and r["phase"] == phase)


def verdict(rows: list[dict]) -> dict:
    cos = {c: row_of(rows, c, "approach")["cos_median"] for c in PRIMARY}
    best = max((c for c in PRIMARY if c != "baseline"), key=lambda c: cos[c])
    route = cos[best] - cos["baseline"] >= ROUTE_MARGIN
    fam = {m: max(FAMILY[m], key=lambda c: cos[c]) for m in FAMILY}
    track_wins = cos[fam["track"]] - cos[fam["wp"]] >= TRACK_MARGIN
    mode = "track" if track_wins else "stage"
    best_family = fam["track"] if track_wins else fam["wp"]
    prior_helps = cos[f"{'track' if track_wins else 'wp'}+prior"] > cos[
        "track" if track_wins else "wp"]
    near = row_of(rows, f"{'track' if track_wins else 'wp'}+prior", "all")["frac_u0_near_demo"]
    line = (f"Verdict: best new config `{best}` approach cos_median {cos[best]:.3f} vs baseline "
            f"{cos['baseline']:.3f} (delta {cos[best] - cos['baseline']:+.3f}, rule >= "
            f"{ROUTE_MARGIN}) -> {'ROUTE to sim' if route else 'do NOT route to sim'}; track "
            f"family best `{fam['track']}` {cos[fam['track']]:.3f} vs wp family best "
            f"`{fam['wp']}` {cos[fam['wp']]:.3f} (delta "
            f"{cos[fam['track']] - cos[fam['wp']]:+.3f}, rule >= {TRACK_MARGIN}) -> goal mode "
            f"**{mode}** ({'track' if track_wins else 'waypoints, stage mode'}); prior "
            f"{'helps' if prior_helps else 'does not help'} on approach in that mode, "
            f"frac_u0_near_demo {near:.2f} (guard < 0.5)")
    return {"route_to_sim": bool(route), "best_config": best, "goal_mode": mode,
            "best_in_mode": best_family, "prior_helps": bool(prior_helps),
            "approach_cos_median": cos, "line": line}


def fmt(x, nd=3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def markdown(doc: dict) -> str:
    rows = doc["rows"]
    v = doc["verdict"]
    lines = ["# Transport screen (W1-B): MPPI u_0 vs the demo on recorded demo frames", "",
             v["line"], "",
             (f"- deploy `{doc['deploy']}` (V11d), config `{doc['config']}` (clearance on, cold "
              f"start, action noise 0), waypoints `{doc['waypoints']}` ({doc['n_waypoints']} "
              f"waypoints at frames {doc['waypoint_frames']}, tols {doc['waypoint_tols']})"),
             (f"- {doc['n_demos']} clean demos, every {STRIDE}th frame before `done`; phases "
              "approach = before the latch, descent = latch .. first hold, hold = the "
              "hold:place_3 frames (demo action 0 there: no cosine, n 0)"),
             ("- cos = cosine of u_0 translations (F xyz, UR xyz) vs the demo's mean action over "
              "the next 5 frames (W1-A M3, `model_gates.screen_rows`); urz = UR-z sign "
              "agreement; near = u_0 within 10 % of the half-box of the goal demo's (box-"
              "clipped) action at the same index on all 12 channels (open-loop replay guard, "
              "want < 0.5; near xyz = the 6 translation channels only); ok = plans with 0 "
              "violating knots"), "",
             ("- speed = median |u_0 xyz| / |demo mean xyz| (approach / descent)"), "",
             ("| config | H | alpha / w | approach cos (n) | > 0.5 | urz | descent cos (n) | "
              "> 0.5 | urz | speed | hold n | near (all) | near xyz | ok (all) | solve p50 / "
              "p95 ms |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, meta in doc["meta"].items():
        a, d = row_of(rows, name, "approach"), row_of(rows, name, "descent")
        h, al = row_of(rows, name, "hold"), row_of(rows, name, "all")
        lines.append(
            f"| {name} | {a['H']} | {a['alpha']:g} / {a['weight']:g} | {fmt(a['cos_median'])} "
            f"({a['n']}) | {fmt(a['frac_cos_gt_0.5'], 2)} | {fmt(a['urz_sign_agree'], 2)} | "
            f"{fmt(d['cos_median'])} ({d['n']}) | {fmt(d['frac_cos_gt_0.5'], 2)} | "
            f"{fmt(d['urz_sign_agree'], 2)} | {fmt(a['speed_ratio_median'], 2)} / "
            f"{fmt(d['speed_ratio_median'], 2)} | {h['n_solved']} | "
            f"{fmt(al['frac_u0_near_demo'], 2)} | {fmt(al['frac_u0_trans_near_demo'], 2)} | "
            f"{fmt(al['plan_viol0_frac'], 2)} | "
            f"{meta['solve_ms_p50']:.1f} / {meta['solve_ms_p95']:.1f} |")
    if doc.get("seeds"):
        lines += ["", "## Seed robustness (approach cos_median; verdict on the first seed)", "",
                  "| seed | " + " | ".join(PRIMARY) + " | best - baseline | track - wp |",
                  "|---" * (len(PRIMARY) + 3) + "|"]
        for sd in doc["seeds"]:
            c = sd["approach_cos_median"]
            best = max(c[k] for k in PRIMARY if k != "baseline")
            fam = {m: max(c[k] for k in FAMILY[m]) for m in FAMILY}
            lines.append(f"| {sd['seed']} | " + " | ".join(fmt(c[k]) for k in PRIMARY)
                         + f" | {best - c['baseline']:+.3f} | {fam['track'] - fam['wp']:+.3f} |")
    lines += ["", "## Approach cos_median per demo (primary configs)", "",
              "| demo | " + " | ".join(PRIMARY) + " |", "|---" * (len(PRIMARY) + 1) + "|"]
    for demo in doc["meta"]["baseline"]["per_demo"]:
        cells = [fmt(doc["meta"][c]["per_demo"][demo]["approach"]["cos_median"], 2)
                 for c in PRIMARY]
        lines.append(f"| {demo} | " + " | ".join(cells) + " |")
    lines += ["", "Mean u_0 / demo mean (mm, F xyz UR xyz) on approach frames:", ""]
    for name in doc["meta"]:
        a = row_of(rows, name, "approach")
        lines.append(f"- {name}: u_0 {a['u0_mean_mm']} vs demo {a['demo_mean_mm']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deploy", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--clearance-model", type=Path, default=None,
                   help="replaces the config's clearance model (thresholds kept)")
    p.add_argument("--waypoints", type=Path, required=True)
    p.add_argument("--demos", type=Path, nargs="+", default=list(DEMOS))
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--weight", type=float, default=0.1)
    p.add_argument("--seeds", type=int, nargs="+", default=[0],
                   help="the verdict uses the first; the others are a robustness table")
    p.add_argument("--out", type=Path, required=True, help="<out>.md / <out>.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    t0 = time.perf_counter()
    base = MppiParams.from_yaml(a.config)
    if a.clearance_model is not None:
        base = dataclasses.replace(base, clearance={**(base.clearance or {}),
                                                    "model": str(a.clearance_model)})
    lcs = TorchLcs.from_deploy(a.deploy, a.device)
    wp = Waypoints.load(a.waypoints)
    enc = LatentEncoder.load(a.deploy)
    demos = [load_demo(d) for d in a.demos]
    Zs = [encode_demo(enc, d) for d in demos]
    g1, g2 = lcs.stage_goals[0], lcs.stage_goals[-1]
    lb, ub = base.bounds(lcs)
    half = np.where((ub - lb) / 2.0 > 0.0, (ub - lb) / 2.0, 1.0)
    rows, meta, seeds = [], {}, []
    for k, seed in enumerate(a.seeds):
        rows_s = []
        for cfg in configs(a.alpha, a.weight):
            if k and cfg.secondary:
                continue
            r, m = run_config(cfg, base, lcs, wp, demos, Zs, g1, g2, lb, ub, half, seed)
            rows_s += r
            ap = row_of(r, cfg.name, "approach")
            print(f"seed {seed} {cfg.name}: approach cos {fmt(ap['cos_median'])} (n {ap['n']}), "
                  f"near {fmt(row_of(r, cfg.name, 'all')['frac_u0_near_demo'], 2)}, p95 "
                  f"{m['solve_ms_p95']:.1f} ms", flush=True)
            if not k:
                meta[cfg.name] = m
        if not k:
            rows = rows_s
        seeds.append({"seed": seed, "approach_cos_median": {
            c: row_of(rows_s, c, "approach")["cos_median"] for c in PRIMARY}})
    doc = {"deploy": str(a.deploy), "config": str(a.config), "waypoints": str(a.waypoints),
           "waypoints_sha256": sha256(a.waypoints), "deploy_sha256": sha256(a.deploy),
           "n_demos": len(demos),
           "demos": [str(d) for d in a.demos], "n_waypoints": len(wp.frames),
           "waypoint_frames": wp.frames.tolist(), "waypoint_tols": np.round(wp.tol, 3).tolist(),
           "stride": STRIDE, "seed": a.seeds[0], "seeds": seeds,
           "box": [lb.tolist(), ub.tolist()],
           "near_frac": NEAR_FRAC, "rows": rows, "verdict": verdict(rows), "meta": meta,
           "wall_s": time.perf_counter() - t0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(doc, indent=1, default=float))
    a.out.with_suffix(".md").write_text(markdown(doc))
    print(doc["verdict"]["line"])
    print(f"wrote {a.out.with_suffix('.md')} ({doc['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
