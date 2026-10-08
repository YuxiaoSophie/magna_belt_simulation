#!/usr/bin/env python3
"""Behavioural refit gate: does MPPI with a candidate model still give demo-like first actions?

Both deploys (candidate, reference = the previous accepted model) on the same frames, same
MPPI yaml (``action_noise_frac`` 0, cold start, clearance with the frame's wxyz, ceilings / rot0
from the episode start as the node), same generator seed. Frames: every ``--stride``-th frame
of each demo before ``done`` (goal = the active waypoint by the demo's frame index, prior =
``u_traj[t:t+H]`` when the yaml has a prior, ``--no-prior`` drops it; without waypoints the
deploy's stage goal by phase) and every ``--stride``-th solve of each MPC success (the logged
``goal_index`` / ``idx`` of the node's solves.jsonl). Metric = ``model_gates.screen_rows``
(cosine of u_0 translations vs the next 5 recorded actions) per phase (approach, descent,
hold, success); plus u_0 Franka y on demo hold frames (mm/step, the overshoot signature) and
the candidate-vs-reference u_0 cosine. Reject iff a gated phase's cos_median drops by more than
``--max-drop`` (phases with fewer than ``--min-n`` cosines are not gated) or, if set, the hold
push exceeds ``--max-hold-push-mm`` or the reference's by ``--max-hold-push-increase-mm``.
``success`` is report-only by default: those rollouts are the reference's own, so their
cosine favours it. Opt-in pre-latch rule (demo frames from the motion start, the first ``move:``
label, to the latch): reject if the candidate's mean u_0 Franka +y there exceeds the reference's
by ``--max-prelatch-push-increase-mm`` or its mean UR descent (-u_0 z) falls short of the
reference's by ``--max-prelatch-descent-drop-mm`` (the sweep-before-seating signature).

    uv run --frozen python scripts/lcs/behaviour_gate.py --cand <deploy.npz> --ref <deploy.npz>
        [--cand-waypoints NPZ --ref-waypoints NPZ] --mppi-config <yaml> --demos <npz ...> \\
        [--successes <episode.npz ...>] --out <gate.json>
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
from latent_waypoints import Waypoints, move0_frame
from model_gates import cosine, demo_next_mean, load_demo, phase_of, screen_rows
from transport_screen import active_waypoint

from task_common.latent_encoder import LatentEncoder, encoder_input
from task_common.lcs_dataset import delta_rotvec
from task_common.mppi_lcs import MppiParams, TorchLcs, mppi_plan

GATED = ("approach", "descent")


def load_success(path: Path, stride: int) -> dict:
    """An MPC rollout + its node solve log (frames of every ``stride``-th solve)."""
    sl = path.parent / "controller" / path.with_suffix(".solves.jsonl").name
    rows = [json.loads(x) for x in sl.read_text().splitlines() if x.strip()]
    with np.load(path, allow_pickle=True) as d:
        ut = np.asarray(d["utime"])
        ep = {"path": str(path), "seg": list(d["pcd_belt_seg"]),
              "u": np.asarray(d["actions"], np.float64),
              "ee_f": np.asarray(d["sim_ee_franka"], np.float64),
              "ee_u": np.asarray(d["sim_ee_ur"], np.float64)}
    pos = {int(u): k for k, u in enumerate(ut)}
    picks = [(pos[r["utime"]], r) for r in rows[::stride] if r["utime"] in pos]
    ep["start"] = pos.get(rows[0]["utime"], 0)
    ep["picks"] = [(k, int(r.get("goal_index", r.get("stage", 0))), r.get("idx"))
                   for k, r in picks]
    return ep


def frame_list(demos: list[dict], succ: list[dict], stride: int) -> list[dict]:
    """Model-independent frame specs (the latent is encoded per model)."""
    out = []
    for j, d in enumerate(demos):
        m0 = move0_frame(d["labels"])
        for t in range(0, d["done0"], stride):
            out.append({"src": ("demo", j), "t": t, "start": 0, "phase": phase_of(d, t),
                        "clock": t, "goal_index": None, "ud": demo_next_mean(d["u"], t),
                        "prelatch": m0 <= t < d["latch"]})
    for j, s in enumerate(succ):
        for t, gi, idx in s["picks"]:
            out.append({"src": ("succ", j), "t": t, "start": s["start"], "phase": "success",
                        "clock": idx, "goal_index": gi, "ud": demo_next_mean(s["u"], t)})
    return out


def run_model(deploy: Path, wp_path: Path | None, base: MppiParams, demos, succ, frames,
              seed: int, device: str, use_prior: bool = True) -> tuple[np.ndarray, dict]:
    lcs = TorchLcs.from_deploy(deploy, device)
    enc = LatentEncoder.load(deploy)
    wp = Waypoints.load(wp_path) if wp_path else None
    params = dataclasses.replace(base, action_noise_frac=0.0)
    prior = use_prior and wp is not None and (params.prior_weight > 0.0
                                              or params.prior_alpha > 0.0)
    eps = {("demo", j): d for j, d in enumerate(demos)}
    eps.update({("succ", j): s for j, s in enumerate(succ)})
    cache: dict = {}
    gen = torch.Generator(device=lcs.device).manual_seed(seed)
    c = params.z_ceiling_above_start_m
    u0s, ms = [], []
    for f in frames:
        ep, t, s0 = eps[f["src"]], f["t"], f["start"]
        key = (f["src"], t)
        if key not in cache:
            cache[key] = enc.encode(encoder_input(enc, {"pcd_belt_seg": ep["seg"][t]}))
        p_t = np.r_[ep["ee_f"][t, :3], ep["ee_u"][t, :3]]
        p_s = np.r_[ep["ee_f"][s0, :3], ep["ee_u"][s0, :3]]
        prm = dataclasses.replace(params, z_max_franka=min(params.z_max_franka, p_s[2] + c),
                                  z_max_ur=min(params.z_max_ur, p_s[5] + c))
        q = np.stack([ep["ee_f"][t, 3:7], ep["ee_u"][t, 3:7]])
        rot0 = np.stack([delta_rotvec(ep["ee_f"][s0, 3:7], q[0]),
                         delta_rotvec(ep["ee_u"][s0, 3:7], q[1])])
        if f["goal_index"] is None:
            goal = (wp.z_wp[active_waypoint(wp, t)] if wp is not None else
                    lcs.stage_goals[0] if f["phase"] == "approach" else lcs.stage_goals[-1])
        else:
            gz = wp.z_wp if wp is not None else lcs.stage_goals
            goal = gz[min(f["goal_index"], len(gz) - 1)]
        kw = {"q_now": q} if prm.clearance is not None else {}
        if prior and f["clock"] is not None:
            kw["u_demo"] = wp.u_seq(int(f["clock"]), prm.H)
        U, info = mppi_plan(lcs, prm, cache[key], p_t, rot0, goal, None, gen, **kw)
        u0s.append(U[0])
        ms.append(info["solve_ms"])
    return np.asarray(u0s), {"solve_ms_p95": float(np.percentile(ms[1:] or ms, 95)),
                             "prior": bool(prior), "waypoints": str(wp_path) if wp else None}


def phase_rows(label: str, H: int, u0: np.ndarray, frames: list[dict]) -> dict:
    ud = np.asarray([f["ud"] for f in frames])
    ph = [f["phase"] for f in frames]
    dm = np.array([p != "success" for p in ph])
    out = {}
    if dm.any():
        rows = screen_rows(label, H, u0[dm], ud[dm], list(np.asarray(ph)[dm]),
                           {"u0_fy_mm": 1e3 * u0[dm, 1]})
        out.update({r["phase"]: r for r in rows if r["phase"] != "all"})
    if (~dm).any():
        r = screen_rows(label, H, u0[~dm], ud[~dm], ["approach"] * int((~dm).sum()),
                        {"u0_fy_mm": 1e3 * u0[~dm, 1]})[-1]
        out["success"] = {**r, "phase": "success"}
    return out


def prelatch_stats(u0: np.ndarray, frames: list[dict]) -> dict:
    """Mean u_0 Franka y and UR descent (-z), mm/step, on the demos' pre-latch motion frames."""
    m = np.array([bool(f.get("prelatch")) for f in frames])
    u = np.asarray(u0)[m] if m.any() else np.zeros((0, 6))
    return {"n": int(m.sum()), "fy_mm": float(1e3 * u[:, 1].mean()) if len(u) else None,
            "ur_down_mm": float(-1e3 * u[:, 5].mean()) if len(u) else None}


def decide(rows_c: dict, rows_r: dict, max_drop: float, min_n: int, phases,
           max_hold_push: float | None, max_hold_increase: float | None = None,
           prelatch: dict | None = None, max_pre_push: float | None = None,
           max_pre_drop: float | None = None) -> dict:
    reasons, deltas = [], {}
    for ph in phases:
        c, r = rows_c.get(ph), rows_r.get(ph)
        if not c or not r or c["cos_median"] is None or r["cos_median"] is None:
            continue
        d = c["cos_median"] - r["cos_median"]
        deltas[ph] = d
        if min(c["n"], r["n"]) >= min_n and d < -max_drop:
            reasons.append(f"{ph} cos_median {c['cos_median']:.3f} < ref "
                           f"{r['cos_median']:.3f} - {max_drop:g}")
    push = (rows_c.get("hold") or {}).get("u0_fy_mm")
    ref = (rows_r.get("hold") or {}).get("u0_fy_mm")
    if max_hold_push is not None and push is not None and push > max_hold_push:
        reasons.append(f"hold-frame Franka u0 y {push:+.2f} mm/step > {max_hold_push:g}")
    if (max_hold_increase is not None and push is not None and ref is not None
            and push > ref + max_hold_increase):
        reasons.append(f"hold-frame Franka u0 y {push:+.2f} > ref {ref:+.2f} + "
                       f"{max_hold_increase:g} mm/step")
    out = {}
    if max_pre_push is not None or max_pre_drop is not None:
        c, r = prelatch["cand"], prelatch["ref"]
        if max_pre_push is not None and c["fy_mm"] is not None \
                and c["fy_mm"] > r["fy_mm"] + max_pre_push:
            reasons.append(f"pre-latch Franka u0 y {c['fy_mm']:+.2f} > ref {r['fy_mm']:+.2f} + "
                           f"{max_pre_push:g} mm/step")
        if max_pre_drop is not None and c["ur_down_mm"] is not None \
                and c["ur_down_mm"] < r["ur_down_mm"] - max_pre_drop:
            reasons.append(f"pre-latch UR u0 -z {c['ur_down_mm']:+.2f} < ref "
                           f"{r['ur_down_mm']:+.2f} - {max_pre_drop:g} mm/step")
        out["prelatch_mm"] = prelatch
    return {"accept": not reasons, "reasons": reasons, "deltas": deltas,
            "hold_push_mm": {"cand": push, "ref": ref}, **out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cand", type=Path, required=True, help="candidate deploy.npz")
    ap.add_argument("--ref", type=Path, required=True, help="reference (accepted) deploy.npz")
    ap.add_argument("--cand-waypoints", type=Path, default=None)
    ap.add_argument("--ref-waypoints", type=Path, default=None)
    ap.add_argument("--mppi-config", type=Path, required=True)
    ap.add_argument("--clearance-model", type=Path, default=None,
                    help="replaces the config's clearance model (thresholds kept)")
    ap.add_argument("--demos", type=Path, nargs="+", required=True)
    ap.add_argument("--successes", type=Path, nargs="*", default=[])
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-drop", type=float, default=0.1)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--phases", default=",".join(GATED))
    ap.add_argument("--max-hold-push-mm", type=float, default=None)
    ap.add_argument("--max-hold-push-increase-mm", type=float, default=None)
    ap.add_argument("--max-prelatch-push-increase-mm", type=float, default=None,
                    help="reject: pre-latch Franka u0 +y > ref's + this (mm/step)")
    ap.add_argument("--max-prelatch-descent-drop-mm", type=float, default=None,
                    help="reject: pre-latch UR u0 -z < ref's - this (mm/step)")
    ap.add_argument("--no-prior", action="store_true",
                    help="screen without the yaml's demo prior (the model + goal policy only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if (a.cand_waypoints is None) != (a.ref_waypoints is None):
        raise SystemExit("--cand-waypoints and --ref-waypoints go together")
    t0 = time.perf_counter()
    base = MppiParams.from_yaml(a.mppi_config)
    if a.clearance_model is not None:
        base = dataclasses.replace(base, clearance={**(base.clearance or {}),
                                                    "model": str(a.clearance_model)})
    demos = [load_demo(d) for d in a.demos]
    succ = [load_success(s, a.stride) for s in a.successes]
    frames = frame_list(demos, succ, a.stride)
    u_c, m_c = run_model(a.cand, a.cand_waypoints, base, demos, succ, frames, a.seed, a.device,
                         not a.no_prior)
    u_r, m_r = run_model(a.ref, a.ref_waypoints, base, demos, succ, frames, a.seed, a.device,
                         not a.no_prior)
    rows_c = phase_rows("cand", base.H, u_c, frames)
    rows_r = phase_rows("ref", base.H, u_r, frames)
    cross = {}
    for ph in sorted({f["phase"] for f in frames}):
        m = [f["phase"] == ph for f in frames]
        cs = [cosine(x[:6], y[:6]) for x, y, k in zip(u_c, u_r, m) if k]
        cs = [x for x in cs if x is not None]
        cross[ph] = float(np.median(cs)) if cs else None
    phases = [p for p in a.phases.split(",") if p]
    pre_rule = {k: v for k, v in (("max_prelatch_push_increase_mm",
                                   a.max_prelatch_push_increase_mm),
                                  ("max_prelatch_descent_drop_mm",
                                   a.max_prelatch_descent_drop_mm)) if v is not None}
    pre = ({"cand": prelatch_stats(u_c, frames), "ref": prelatch_stats(u_r, frames)}
           if pre_rule else None)
    dec = decide(rows_c, rows_r, a.max_drop, a.min_n, phases, a.max_hold_push_mm,
                 a.max_hold_push_increase_mm, pre, a.max_prelatch_push_increase_mm,
                 a.max_prelatch_descent_drop_mm)
    doc = {"cand": str(a.cand), "ref": str(a.ref), "cand_waypoints": str(a.cand_waypoints),
           "ref_waypoints": str(a.ref_waypoints), "mppi_config": str(a.mppi_config),
           "n_demos": len(demos), "n_successes": len(succ), "n_frames": len(frames),
           "stride": a.stride, "seed": a.seed, "prior": not a.no_prior,
           "rule": {"max_drop": a.max_drop, "min_n": a.min_n, "phases": phases,
                    "max_hold_push_mm": a.max_hold_push_mm,
                    "max_hold_push_increase_mm": a.max_hold_push_increase_mm, **pre_rule},
           "rows": {"cand": rows_c, "ref": rows_r}, "cross_cos_median": cross,
           "meta": {"cand": m_c, "ref": m_r}, "decision": dec,
           "wall_s": time.perf_counter() - t0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, default=float) + "\n")

    def cm(rows, ph):
        r = rows.get(ph) or {}
        return "-" if r.get("cos_median") is None else f"{r['cos_median']:.3f} (n {r['n']})"

    print(" | ".join(f"{ph} cand {cm(rows_c, ph)} ref {cm(rows_r, ph)}"
                     for ph in ("approach", "descent", "success") if ph in rows_c))
    hp = dec["hold_push_mm"]
    if pre is not None:
        print(f"pre-latch u0 Franka y / UR -z (n {pre['cand']['n']}) cand "
              f"{pre['cand']['fy_mm']:+.2f} / {pre['cand']['ur_down_mm']:+.2f}, ref "
              f"{pre['ref']['fy_mm']:+.2f} / {pre['ref']['ur_down_mm']:+.2f} mm/step")
    print(f"hold u0 y cand {hp['cand'] if hp['cand'] is None else round(hp['cand'], 2)} / ref "
          f"{hp['ref'] if hp['ref'] is None else round(hp['ref'], 2)} mm; cross cos "
          f"{ {k: None if v is None else round(v, 3) for k, v in cross.items()} }; "
          f"{'ACCEPT' if dec['accept'] else 'REJECT: ' + '; '.join(dec['reasons'])} "
          f"({doc['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
