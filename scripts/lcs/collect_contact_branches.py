#!/usr/bin/env python3
"""Contact-rich rollouts branched from sim snapshots taken at contact events of insertions.

Phases:
  ``snap``    run insertion source episodes with ``collect_lcs_dataset.collect``
              (``run_osc_episode`` wrapped in-process with an ``on_frame`` callback);
              snapshot ``first_contact`` (first frame with n_neighbour >= 3 and h_min <= 10 mm,
              or wrap > 0), ``partial`` (first frame after it with 15 <= wrap < 60 deg) and
              ``final_<label>``; each snapshot is
              validated right away (restore + 0.5 s settle + grasp check).
  ``select``  balanced working set of the validated snapshots (offline).
  ``branch``  per (snapshot, family): restore + 0.5 s settle, 1.0 s pre-hold (u = 0), a scripted
              4-8 s rollout of both arms, 0.5 s hold; the training format of the collector.
  ``report``  ``qc.json`` per branch run + the rebalance report (contact frames of insertion v2,
              approach set incl. tails, branch set).

Run (detached, private URLs only):
    uv run --frozen python scripts/lcs/collect_contact_branches.py snap \\
        --lcm-url 'udpm://239.255.76.139:7739?ttl=0' --src nominal --episodes 20 --seed 501
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import collect_lcs_dataset as col
import collect_motion_primitives as cmp
import place_tail as ptl
from loguru import logger

from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import UrTracking, ik, rotvec
from round_belt_task.commander import (
    CommanderParams,
    FrankaCommand,
    UrCommand,
    UrLine,
    UrLineCommander,
    UrTarget,
    mat3_to_quat,
    parse_saved_traj_message,
    pose_mat,
    saved_traj_message,
    x_tool0_tracking,
)
from round_belt_task.episode_io import (
    finish_recording,
    git_info,
    sha256_file,
    write_json,
)
from round_belt_task.outcome import DEFAULT_THRESHOLDS, classify, frame_metrics
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.osc_process import check_private_url

DATA = REPO_ROOT / "data" / "lcs"
CONTACT = DATA / "contact"
DEMO = DATA / "demo_flat" / "demo_episode.npz"
DEMO_FRAME = 59
FC_MIN_NEIGHBOUR, FC_H_MIN_MM = 3, 10.0
FAIL_LABELS = ("over", "under", "slanted")
CLASSES = ("first_contact", "partial", "final_engaged", "final_over", "final_under",
           "final_slanted")
FAMILIES = ("groove_slide", "pullout_reseat", "rim_press", "top_cross", "recover",
            "random_contact")
MAX_AMP = {"groove_slide": 15.0, "pullout_reseat": 25.0, "rim_press": 10.0, "rim_tilt": 6.0,
           "top_lift": 10.0, "top_cross": 15.0, "random_mm": 0.6 * 20.0, "random_deg": 0.6 * 5.0}
AMP_RANGE = (0.4, 1.0)
T_RANGE = (4.0, 8.0)
SETTLE_S, PRE_HOLD_S, HOLD_S = 0.5, 1.0, 0.5
CAP = (3.0, 25.0, 3.0, 25.0)
STRETCH_CAP_PCT = 2.0
FRANKA_FLOOR_MM, FLOOR_MARGIN_MM = 15.0, 0.5
CAP_MARGIN = 1.15  # min-jerk stage duration margin over the cap-limited minimum
MINJERK_PEAK = 1.875
RETRY_MUL = 0.7
WS_SIZES = {"first_contact": 10, "partial": 10, "final_engaged": 5, "failure": 15}
EZ = np.array([0.0, 0.0, 1.0])
URL_SNAP, URL_BRANCH = "7739", "7740"


def snap_class(event: str, label: str) -> str:
    return event if event in ("first_contact", "partial") else f"final_{label}"


def fm_dict(fm) -> dict:
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in dataclasses.asdict(fm).items()}


# ---- snap -----------------------------------------------------------------------------------

class Snapper:
    """``on_frame`` of one source episode: captures the first_contact / partial snapshots."""

    def __init__(self) -> None:
        self.j = -1
        self.events: dict[str, dict] = {}
        self.last_sim_step = None

    def __call__(self, k, frame, fm, sim) -> None:
        self.j += 1
        self.last_sim_step = int(sim.step_index)
        fc = self.events.get("first_contact")
        hit = ((fm.n_neighbour >= FC_MIN_NEIGHBOUR and fm.h_min_mm <= FC_H_MIN_MM)
               or fm.wrap_deg > 0.0)
        if fc is None and hit:
            self._take("first_contact", k, fm, sim)
        elif (fc is not None and "partial" not in self.events
              and DEFAULT_THRESHOLDS.partial_arc_deg <= fm.wrap_deg
              < DEFAULT_THRESHOLDS.engaged_arc_deg):
            self._take("partial", k, fm, sim)

    def _take(self, event, k, fm, sim) -> None:
        self.events[event] = {"snap": sim_snapshot.capture(sim, event), "frame": self.j, "k": k,
                              "sim_step": int(sim.step_index), "metrics": fm_dict(fm),
                              "label": classify(fm)}


def run_snap(args) -> int:
    check_private_url(args.lcm_url)
    snap_dir = Path(args.snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    src_out = Path(args.source_out)
    cli = ["--out", str(src_out), "--episodes", str(args.episodes), "--seed", str(args.seed),
           "--lcm-url", args.lcm_url, "--pre-hold-s", "0", "--no-pcd", "--hold-ur-gripper",
           "--excite-pos-mm", "0", "--excite-rot-deg", "0", "--excite-ur-pos-mm", "0",
           "--excite-ur-rot-deg", "0", "--label", f"contact-src-{args.src}"]
    if args.start_states:
        cli += ["--start-states", str(args.start_states)]
        if args.variants:
            cli += ["--variants", args.variants]
    cargs = col.create_parser().parse_args(cli)
    index_path = snap_dir / "index.json"
    index = (json.loads(index_path.read_text()) if index_path.exists() else
             {"set": snap_dir.name, "kind": "contact_snapshots",
              "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "variants": [], "sources": []})
    index["rules"] = {
        "first_contact": f"first frame with (n_neighbour >= {FC_MIN_NEIGHBOUR} and h_min_mm <= "
                         f"{FC_H_MIN_MM:g}) or wrap_deg > 0",
        "partial": "first frame after first_contact with partial_arc_deg <= wrap < "
                   "engaged_arc_deg (15 <= wrap < 60)",
        "final": "the last frame; label = the episode's classified outcome",
        "validation": "restore + 0.5 s settle under the hold-at-measured hook + grasp check"}
    index["sources"].append({"src": args.src, "source_out": str(src_out), "cli": cli})
    orig = col.run_osc_episode
    settle_steps = [0]

    def wrapped(sim, ctx, i, p):
        sn = Snapper()
        row = orig(sim, ctx, i, p, on_frame=sn)
        if row.get("status") != "ok":
            return row
        if int(sim.step_index) != sn.last_sim_step:
            raise RuntimeError(f"sim advanced after the last frame ({sim.step_index} != "
                               f"{sn.last_sim_step})")
        body_q = sim.state_0.body_q.numpy()
        fm = frame_metrics(body_q[sim.info.belt_bodies, :3].astype(np.float64),
                           body_q[int(sim.info.pulley_bodies[1])].astype(np.float64))
        sn.events[f"final_{row['outcome']}"] = {
            "snap": sim_snapshot.capture(sim, "final"), "frame": sn.j, "k": None,
            "sim_step": int(sim.step_index), "metrics": fm_dict(fm), "label": row["outcome"]}
        variant = Path(str(ctx.start_state)).stem.removesuffix("_osc")
        settle_steps[0] = round(SETTLE_S / sim.frame_dt)
        for event, ev in sn.events.items():
            sid = f"{args.src}_e{i:04d}_{event}"
            ev["snap"].meta["notes"] = (f"contact snapshot {event} of {args.src} episode {i} "
                                        f"({p.intent} -> {row['outcome']}), frame {ev['frame']}")
            path = sim_snapshot.save(ev["snap"], snap_dir / f"{sid}_osc.npz")
            grasp = sim.restore(sim_snapshot.load(path), settle_steps=settle_steps[0])
            bq = sim.state_0.body_q.numpy()
            fm2 = frame_metrics(bq[sim.info.belt_bodies, :3].astype(np.float64),
                                bq[int(sim.info.pulley_bodies[1])].astype(np.float64))
            held = grasp.held() == (True, True)
            ev_name = event if not event.startswith("final_") else "final"
            rec = {"id": sid, "file": path.name, "held": bool(held), "event": ev_name,
                   "class": snap_class(ev_name, ev["label"]), "label": ev["label"],
                   "metrics": ev["metrics"], "frame": ev["frame"], "k": ev["k"],
                   "sim_step": ev["sim_step"], "intent": p.intent,
                   "source_episode": {"src": args.src, "episode": i, "file": str(
                       src_out / row["file"]), "start_variant": variant, "seed": args.seed,
                       "outcome": row["outcome"]},
                   "recheck_0p5s": {"held": list(grasp.held()), "metrics": fm_dict(fm2),
                                    "label": classify(fm2), "grasp": grasp.describe()},
                   "sha256": sha256_file(path)}
            index["variants"] = [r for r in index["variants"] if r["id"] != sid] + [rec]
            logger.info(f"[SNAP] {sid}: {ev['label']} wrap {ev['metrics']['wrap_deg']:.0f} h_med "
                        f"{ev['metrics']['h_median_mm']} -> after 0.5 s {classify(fm2)} wrap "
                        f"{fm2.wrap_deg:.0f}, held {held}")
        write_json(index_path, index)
        return row

    col.run_osc_episode = wrapped
    try:
        col.collect(cargs)
    finally:
        col.run_osc_episode = orig
    write_json(index_path, index)
    counts = Counter(r["class"] for r in index["variants"] if r["held"])
    logger.info(f"[SNAP] {snap_dir}: validated per class {dict(counts)}")
    return 0


# ---- select ---------------------------------------------------------------------------------

def _interleave(rows: list[dict]) -> list[dict]:
    """Round-robin over sources (then start variants) in episode order."""
    by = defaultdict(list)
    def key(r):
        return r["source_episode"]["src"], r["source_episode"]["episode"]

    for r in sorted(rows, key=key):
        by[(r["source_episode"]["src"], r["source_episode"]["start_variant"])].append(r)
    out, keys = [], sorted(by)
    while any(by[k] for k in keys):
        for k in keys:
            if by[k]:
                out.append(by[k].pop(0))
    return out


def run_select(args) -> int:
    index = json.loads((Path(args.snap_dir) / "index.json").read_text())
    ok = [r for r in index["variants"] if r["held"]]
    by_class = {c: _interleave([r for r in ok if r["class"] == c]) for c in CLASSES}
    if args.all:
        chosen = [r for c in CLASSES for r in by_class[c]]
    else:
        chosen = []
        for c in ("first_contact", "partial", "final_engaged"):
            chosen += by_class[c][:WS_SIZES[c]]
        pools = [list(by_class[f"final_{lb}"]) for lb in FAIL_LABELS]
        fail = []
        while len(fail) < WS_SIZES["failure"] and any(pools):
            for pool in pools:
                if pool and len(fail) < WS_SIZES["failure"]:
                    fail.append(pool.pop(0))
        chosen += fail
    plan = assign(chosen, heldout=args.all)
    out = {"snap_dir": str(args.snap_dir), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "available": {c: len(v) for c, v in by_class.items()},
           "chosen": {c: sum(r["class"] == c for r in chosen) for c in CLASSES},
           "families": dict(Counter(f for _, f in plan)),
           "plan": [{"id": r["id"], "class": r["class"], "family": f} for r, f in plan]}
    write_json(Path(args.out), out)
    print(json.dumps({k: out[k] for k in ("available", "chosen", "families")}, indent=1))
    return 0


def assign(chosen: list[dict], heldout: bool) -> list[tuple[dict, str]]:
    """Train: 2 families per snapshot (the approved cycling); held-out: 1."""
    plan, seen = [], Counter()
    for r in chosen:
        c = r["class"]
        j = seen[c]
        seen[c] += 1
        fail = c in (f"final_{lb}" for lb in FAIL_LABELS)
        if heldout:
            plan.append((r, "recover" if fail else "random_contact"))
            continue
        if c == "first_contact":
            a, b = ("rim_press", "top_cross") if j % 2 == 0 else ("top_cross", "rim_press")
            fams = [a, b if j < 8 else "random_contact"]
        elif c == "partial":
            fams = (["pullout_reseat", "random_contact"] if j % 2 == 0
                    else ["groove_slide", "pullout_reseat"])
        elif c == "final_engaged":
            fams = ["groove_slide", "random_contact"]
        elif fail:
            fams = ["recover", "random_contact"]
        else:
            continue
        plan += [(r, f) for f in fams]
    return plan


# ---- branch: motions ------------------------------------------------------------------------

def ramp(a: float, b: float):
    """Min-jerk step 0 -> 1 over tau in [a, b]."""
    def f(t):
        x = np.clip((np.asarray(t, np.float64) - a) / (b - a), 0.0, 1.0)
        return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
    return f


def holds_warp(rng: np.random.Generator, T: float, n: int, lo: float = 0.8, hi: float = 2.0):
    """Monotone tau(s) with ``n`` plateaus of ``lo..hi`` s (<= 0.5 T total), eased between."""
    d = rng.uniform(lo, hi, n)
    if d.sum() > 0.5 * T:
        d *= 0.5 * T / d.sum()
    taus = np.sort(0.15 + 0.7 * (np.arange(n) + rng.uniform(0.2, 0.8, n)) / n)
    p = d / T
    m = 1.0 - p.sum()
    s_k, t_k, flat = [0.0], [0.0], []
    for tau, pi in zip(taus, p, strict=True):
        s_k.append(s_k[-1] + (tau - t_k[-1]) * m)
        t_k.append(tau)
        flat.append(len(s_k) - 1)
        s_k.append(s_k[-1] + pi)
        t_k.append(tau)
    s_k.append(1.0)
    t_k.append(1.0)
    s_k, t_k = np.array(s_k), np.array(t_k)

    def warp(s: float) -> float:
        j = min(int(np.searchsorted(s_k, s, side="right")) - 1, len(s_k) - 2)
        if j in flat:
            return float(t_k[j])
        x = (s - s_k[j]) / (s_k[j + 1] - s_k[j])
        return float(t_k[j] + (t_k[j + 1] - t_k[j]) * cmp.smoothstep(x))

    return warp, [{"tau": float(taus[i]), "s0": float(s_k[k]), "s1": float(s_k[k + 1])}
                  for i, k in enumerate(flat)]


class BranchMotion(ptl.TailMotion):
    """``TailMotion``; with ``hold_end`` the offset stays at its tau -> 1 value after T."""

    def __init__(self, *a, hold_end: bool = False, **kw) -> None:
        super().__init__(*a, **kw)
        self.hold_end = hold_end

    def offset(self, arm: str, t: float, scale: dict | None = None) -> np.ndarray:
        if self.hold_end:
            t = min(t, self.t_start + self.T * (1.0 - 1e-9))
        return super().offset(arm, t, scale)


def demo_poses() -> dict:
    with np.load(DEMO, allow_pickle=True) as d:
        out = {}
        for arm, key in (("franka", "sim_ee_franka"), ("ur", "sim_ee_ur")):
            v = d[key][DEMO_FRAME].astype(np.float64)
            out[arm] = pose_mat(v[:3], v[3:])
    return out


def draw_family(fam: str, key: list[int]) -> dict:
    rng = np.random.default_rng(key)
    u = rng.uniform(size=8)
    T = T_RANGE[0] + u[0] * (T_RANGE[1] - T_RANGE[0])
    m = AMP_RANGE[0] + u[1] * (AMP_RANGE[1] - AMP_RANGE[0])
    sign = 1.0 if u[2] < 0.5 else -1.0
    par = {"T_s": T, "amp_mul": m}
    if fam == "groove_slide":
        par["amp_deg"] = sign * m * MAX_AMP["groove_slide"]
    elif fam == "pullout_reseat":
        par["amp_mm"] = m * MAX_AMP["pullout_reseat"]
    elif fam == "rim_press":
        par.update(amp_mm=m * MAX_AMP["rim_press"], tilt_deg=sign * m * MAX_AMP["rim_tilt"])
    elif fam == "top_cross":
        par.update(lift_mm=m * MAX_AMP["top_lift"], cross_mm=sign * m * MAX_AMP["top_cross"])
    elif fam == "recover":
        par = {"T_s": T, "amp_mul": None, "lift_mm": 10.0 + 10.0 * u[3],
               "fraction": 0.5 + 0.5 * u[4], "dz_mm": -2.0 + 6.0 * u[5],
               "descend_s": 2.0 + u[6]}
    elif fam == "random_contact":
        par.update(amp_mm=m * MAX_AMP["random_mm"], amp_deg=m * MAX_AMP["random_deg"],
                   holds=1 + int(u[3] < 0.5), shape_seed=[*key, 1])
    else:
        raise KeyError(fam)
    return par


def min_stage_s(dist: float, cap: float, dt_s: float) -> float:
    """Min-jerk duration keeping the peak step within ``cap`` (same units per step)."""
    return MINJERK_PEAK * abs(dist) * dt_s / cap * CAP_MARGIN


def recover_components(par: dict, X0: dict, demo: dict, dt_s: float, plate_top: float,
                       floor_mm: float) -> tuple[list, float, dict]:
    """Lift, move a fraction toward the demo seated poses, descend to demo z + dz, hold."""
    C = cmp.Component
    cap_m, cap_r = CAP[0] * 1e-3, CAP[1] * 1e-3
    lift = par["lift_mm"] * 1e-3
    geo = {}
    for arm in ("franka", "ur"):
        X, D = X0[arm], demo[arm]
        dxy = par["fraction"] * (D[:3, 3] - X[:3, 3])
        dxy[2] = 0.0
        r = par["fraction"] * rotvec(X[:3, :3].T @ D[:3, :3])
        z_end = D[2, 3] + par["dz_mm"] * 1e-3
        if arm == "franka":
            z_end = max(z_end, plate_top + floor_mm * 1e-3)
        geo[arm] = {"dxy": dxy, "rot": r, "desc": (X[2, 3] + lift) - z_end}
    t_lift = max(1.0, min_stage_s(lift, cap_m, dt_s))
    t_move = max(1.0, *(min_stage_s(np.linalg.norm(g["dxy"]), cap_m, dt_s) for g in geo.values()),
                 *(min_stage_s(np.linalg.norm(g["rot"]), cap_r, dt_s) for g in geo.values()))
    t_desc = max(par["descend_s"], *(min_stage_s(g["desc"], cap_m, dt_s) for g in geo.values()))
    t_end = 0.3
    T = max(par["T_s"], t_lift + t_move + t_desc + t_end)
    t_move = T - t_lift - t_desc - t_end
    a1, a2, a3 = t_lift / T, (t_lift + t_move) / T, (t_lift + t_move + t_desc) / T
    comps = []
    for arm, g in geo.items():
        a = arm[0]
        comps.append(C(f"{a}_lift", arm, "trans", EZ, ramp(0.0, a1), lift, arm))
        n = float(np.linalg.norm(g["dxy"]))
        if n > 1e-6:
            comps.append(C(f"{a}_move", arm, "trans", g["dxy"] / n, ramp(a1, a2), n, arm))
        ang = float(np.linalg.norm(g["rot"]))
        if ang > 1e-6:
            comps.append(C(f"{a}_rot", arm, "rot", g["rot"] / ang, ramp(a1, a2), ang, arm))
        comps.append(C(f"{a}_desc", arm, "trans", -EZ, ramp(a2, a3), g["desc"], arm))
    stages = {"lift_s": t_lift, "move_s": t_move, "descend_s": t_desc, "end_s": t_end,
              "geometry": {arm: {"dxy_mm": (g["dxy"] * 1e3).tolist(),
                                 "rot_deg": np.degrees(np.linalg.norm(g["rot"])).item(),
                                 "descend_mm": g["desc"] * 1e3} for arm, g in geo.items()}}
    return comps, T, stages


def family_components(fam: str, par: dict, X0: dict, pulley: dict, dt_s: float,
                      plate_top: float, floor_mm: float, demo: dict):
    """``(components, T, warp, holds, hold_end, extra)``."""
    C = cmp.Component
    X_f0, X_u0 = X0["franka"], X0["ur"]
    g = X_u0[:3, 3] - X_f0[:3, 3]
    g[2] = 0.0
    G = g / np.linalg.norm(g)
    T = dt_s * max(1, round(par["T_s"] / dt_s))
    warp, holds, extra = None, [], {}
    if fam == "groove_slide":
        a = math.radians(par["amp_deg"])
        comps = [C("f_orbit", "franka", "orbit", pulley["axis"], cmp.osc(1), a, "all"),
                 C("u_orbit", "ur", "orbit", pulley["axis"], cmp.osc(1), a, "all")]
    elif fam == "pullout_reseat":
        away = G if float((X_u0[:3, 3] - pulley["centre"]) @ G) >= 0.0 else -G
        a = par["amp_mm"] * 1e-3
        comps = [C("f_pull", "franka", "trans", away, cmp.bump(1), a, "all"),
                 C("u_pull", "ur", "trans", away, cmp.bump(1), a, "all")]
    elif fam == "rim_press":
        a, tl = par["amp_mm"] * 1e-3, math.radians(par["tilt_deg"])
        comps = [C("f_z", "franka", "trans", -EZ, cmp.bump(1), a, "franka"),
                 C("u_z", "ur", "trans", -EZ, cmp.bump(1), a, "ur"),
                 C("f_tilt", "franka", "rot", X_f0[:3, :3].T @ G, cmp.osc(1), tl, "tilt"),
                 C("u_tilt", "ur", "rot", X_u0[:3, :3].T @ G, cmp.osc(1), tl, "tilt")]
    elif fam == "top_cross":
        up, cr = par["lift_mm"] * 1e-3, par["cross_mm"] * 1e-3
        w = cmp.window(cmp.osc(1), 0.2, 0.8)
        comps = [C("f_lift", "franka", "trans", EZ, cmp.plateau(0.2), up, "lift"),
                 C("u_lift", "ur", "trans", EZ, cmp.plateau(0.2), up, "lift"),
                 C("f_cross", "franka", "trans", G, w, cr, "cross"),
                 C("u_cross", "ur", "trans", G, w, cr, "cross")]
    elif fam == "recover":
        comps, T, extra = recover_components(par, X0, demo, dt_s, plate_top, floor_mm)
        T = dt_s * math.ceil(T / dt_s - 1e-9)
        return comps, T, None, [], True, extra
    elif fam == "random_contact":
        rng = np.random.default_rng(par["shape_seed"])
        a_m, a_r = par["amp_mm"] * 1e-3, math.radians(par["amp_deg"])
        comps = [c for arm in ("franka", "ur")
                 for c in cmp._random_arm(arm, rng, cmp.RANDOM_T_REF, a_m, a_r)]
        warp, holds = holds_warp(rng, T, par["holds"])
    else:
        raise KeyError(fam)
    return comps, T, warp, holds, False, extra


# ---- branch: one rollout --------------------------------------------------------------------

def rollout(sim, ctx, i: int, snap_row: dict, snap, fam: str, par: dict, post_mul: float,
            gauge, demo: dict) -> dict:
    args, n, params = ctx.args, ctx.n, ctx.params
    t0 = time.perf_counter()
    dt_s = n * lcs.SIM_DT_S
    fname = f"episode_{i:04d}"
    label = f"{fname}-{fam}"
    row = {"file": f"{fname}.npz", "family": fam, "snapshot": snap_row["id"],
           "event": snap_row["event"], "class": snap_row["class"],
           "source_label": snap_row["label"], "params": par, "retry_mul": post_mul}
    sim.restore(snap, settle_steps=0)
    grasp = sim.settle(round(SETTLE_S / sim.frame_dt))
    if grasp.held() != (True, True):
        return {**row, "status": "failed", "reason": "grasp lost after settle",
                "reason_key": "grasp_settle"}
    hand0, byte0 = sim.gripper_commands()
    q_ur = sim.arm_targets()[1]
    body_q = sim.state_0.body_q.numpy()
    fm_start = frame_metrics(body_q[sim.info.belt_bodies, :3].astype(np.float64),
                             body_q[int(sim.info.pulley_bodies[1])].astype(np.float64))
    phase_labels = [f"move:{fam}", f"hold:{fam}", "done", col.PREHOLD_PHASE]
    pre_n = round(PRE_HOLD_S / dt_s)
    post_n = round(HOLD_S / dt_s)
    s0 = sim.step_index + 1 + pre_n * n
    t_start = sim.osc_time_s(s0)
    pos, quat, _ = parse_saved_traj_message(sim.commander_hook(
        sim.step_index, sim.osc_time_s(), sim.state_0.joint_q.numpy(), body_q))
    X0 = {"franka": pose_mat(pos[0], quat[0]), "ur": UrTracking.fk(q_ur)}
    tip0_mm = (X0["franka"][2, 3] - ctx.plate_top_z) * 1e3
    floor_mm = min(FRANKA_FLOOR_MM, tip0_mm - FLOOR_MARGIN_MM)
    pulley = col.pulley_frame(sim)
    comps, T, warp, holds, hold_end, extra = family_components(
        fam, par, X0, pulley, dt_s, ctx.plate_top_z, floor_mm, demo)
    motion = BranchMotion(comps, X0["franka"], X0["ur"], t_start, T, pulley, warp,
                          hold_end=hold_end)
    motion_n = round(T / dt_s)
    lb, ub = ptl._caps(list(CAP))
    scale = cmp.plan_scales(motion, dt_s, motion_n + 1, lb, ub, 1.0)
    scale = {k: v * post_mul for k, v in scale.items()}
    jaws = (byte0,) if byte0 is not None else (None,)
    base_clear = min(ctx.gripper.predict(ctx.board, X0["ur"], j) for j in jaws)
    floor_c = min(args.min_clearance * 1e-3, base_clear) - ptl.CLEAR_TOL_MM * 1e-3
    groups = {arm: {c.group for c in comps if c.arm == arm} for arm in ("franka", "ur")}

    def clearance_of(sc):
        return cmp.plan_clearance(motion, sc, ctx.board, ctx.gripper, jaws, sim.frame_dt)

    def shrink(arm, f):
        for gname in groups[arm]:
            scale[gname] = scale.get(gname, 1.0) * f

    ur_only = groups["ur"] - groups["franka"]
    if len(groups["ur"]) > 1:  # each UR-only group alone first: shrink only the offenders
        for gname in sorted(ur_only):
            alone = {h: (scale.get(h, 1.0) if h == gname else 0.0) for h in scale}
            for _ in range(8):
                if clearance_of(alone) >= floor_c:
                    break
                alone[gname] *= 0.7
            scale[gname] = alone[gname] if clearance_of(alone) >= floor_c else 0.0
    clear = clearance_of(scale)
    for _ in range(8):
        if clear >= floor_c:
            break
        shrink("ur", 0.7)
        clear = clearance_of(scale)
    if clear < floor_c:
        shrink("ur", 0.0)
        clear = clearance_of(scale)

    def tip_floor():
        return min(motion.pose("franka", t_start + k * dt_s, scale)[2, 3]
                   for k in range(motion_n + 1)) - ctx.plate_top_z

    tip = tip_floor()
    for _ in range(8):
        if tip * 1e3 >= floor_mm:
            break
        shrink("franka", 0.7)
        tip = tip_floor()
    eff = {c.name: c.amp * scale.get(c.group, 1.0) * (1e3 if c.kind == "trans"
                                                      else 180.0 / math.pi) for c in comps}
    u_plan = cmp.planned_actions(motion, dt_s, scale, motion_n + 1)
    ratio_plan = float(cmp.bound_ratio(u_plan, lb, ub).max())

    guard = col.UrExciteGuard(ctx.board, ctx.gripper, floor_c, 0.5)
    base_u = UrTarget(pos=X0["ur"][:3, 3].copy(), quat_wxyz=mat3_to_quat(X0["ur"][:3, :3]),
                      byte=byte0)
    ur_cmdr = UrLineCommander([None], params, X_tool0_tracking=ctx.x_tool0, byte=byte0)
    u_coords = sim.arm_coords()[1]
    times_rel = np.arange(params.n_knots) * params.dt
    end_t = t_start + T
    state = {"tick": None, "hook_s": 0.0, "tip_min": math.inf}

    def hook(step, t, joint_q, bq):
        h0 = time.perf_counter()
        ee_f = sim.franka_measured_pose7(joint_q)
        X_u = UrTracking.fk(joint_q[u_coords])
        state["tip_min"] = min(state["tip_min"], ee_f[2] - ctx.plate_top_z)
        moving = t < end_t
        ts = t + times_rel
        Xs = [motion.pose("franka", tk, scale) for tk in ts]
        kp = np.stack([X[:3, 3] for X in Xs])
        kq = np.stack([mat3_to_quat(X[:3, :3]) for X in Xs])
        cmd = FrankaCommand(knots_pos=kp, knots_quat=kq, times=ts, hold=not moving,
                            target_index=0, phase=phase_labels[0 if moving else 1],
                            hand_mm=hand0)
        p0, q0 = ur_cmdr.to_tool0(motion.pose("ur", t, scale))
        if moving:
            p1, q1 = ur_cmdr.to_tool0(motion.pose("ur", t + params.dt, scale))
            line = UrLine(p0=p0, q0=q0, t0=float(t), p1=p1, q1=q1, t1=float(t) + params.dt)
        else:  # zero-span line: cmd_delta exactly 0
            line = UrLine(p0=p0, q0=q0, t0=0.0, p1=p0, q1=q0, t1=0.0)
        urc = UrCommand(line=line, regenerated=True, byte=byte0, t=float(t),
                        X_tool0_tracking=ctx.x_tool0)
        state["tick"] = col.OscTick(step - s0, t, ee_f, X_u, cmd, urc, None, None, guard.scale)
        state["hook_s"] += time.perf_counter() - h0
        return saved_traj_message(round(t * 1e6), kp, kq, ts)

    if args.record:
        sim.start_recording(ctx.out / "recordings", label)
    sampler = col.OscEpisodeSampler(sim, ctx, phase_labels)
    col._pre_hold(sim, ctx, sampler, state, pre_n * n, s0, q_ur, ur_cmdr, hand0, byte0)
    pb0 = lcs.belt_points_ordered(sampler.frames[-1]["belt"])
    z_top_ref = max(float(cmp.CROP_HI[2]), float(pb0[:, 2].max()))
    lead_gain = sim.arm_kd / sim.arm_ke / sim.frame_dt
    total = (motion_n + post_n) * n
    rods, scales, offs_f, offs_u = [], [], [], []
    fail = key = None
    first = True
    sim.commander_hook = hook
    try:
        while True:
            t1 = sim.osc_time_s(sim.step_index + 1)
            raw = motion.world_rotvec("ur", t1, scale)
            applied = guard.apply(t1, base_u, raw, jaws)
            X = col.UrExciteGuard.target(base_u, applied)
            q_new, err_p, err_r, _ = ik(UrTracking, pose_mat(X.pos, X.quat_wxyz), q_ur,
                                        pos_tol=col.UR_IK_POS_TOL, rot_tol=col.UR_IK_ROT_TOL)
            if err_p > col.UR_IK_POS_TOL or err_r > col.UR_IK_ROT_TOL:
                fail, key = "UR IK miss", "ik"
                break
            lead = 0.0 if first else col.UR_VELOCITY_LEAD * lead_gain
            first = False
            sim.set_ur_target(q_new + lead * (q_new - q_ur))
            q_ur = q_new
            sim.set_grippers(hand0, byte0)
            sim.control_step()
            k = state["tick"].k
            if k % col.CLEARANCE_EVERY == 0 or k % n == 0:
                sampler.clearance(sim.state_0.body_q.numpy())
            if sampler.min_clearance_m < 0.0:
                fail, key = f"board contact at k {k}", "board_contact"
                break
            if k % n != 0:
                continue
            sampler.sample_tick(state["tick"])
            rods.append(gauge.measure(sim.state_0.body_q.numpy())["stretch_pct"])
            scales.append(guard.scale)
            offs_f.append(motion.offset("franka", state["tick"].t, scale))
            offs_u.append(motion.offset("ur", state["tick"].t, scale))
            fr = sampler.frames[-1]
            pb = lcs.belt_points_ordered(fr["belt"])
            if not all(fr["grasp_ok"]):
                fail, key = f"grasp lost at k {k} ({fr['grasp_ok'].tolist()})", "grasp_lost"
                break
            if np.any(pb[:, :2] < cmp.CROP_LO[:2]) or np.any(pb[:, :2] > cmp.CROP_HI[:2]):
                fail, key = f"belt outside the crop x/y at k {k}", "crop_xy"
                break
            if pb[:, 2].max() > z_top_ref + ptl.CROP_TOP_TOL_MM * 1e-3:
                fail, key = (f"belt top {pb[:, 2].max():.4f} > {z_top_ref:.4f} + 2 mm at k {k}",
                             "crop_top")
                break
            if rods[-1] - rods[0] > STRETCH_CAP_PCT:
                fail, key = f"rod stretch gain {rods[-1] - rods[0]:.2f} % at k {k}", "stretch"
                break
            if state["tip_min"] * 1e3 < floor_mm:
                fail, key = f"franka tip {state['tip_min'] * 1e3:.1f} mm at k {k}", "tip_floor"
                break
            if k >= total:
                break
    finally:
        sim.commander_hook = sim.make_hold_hook()
    if fail is not None:
        finish_recording(sim, ctx.out, label, "episode failed")
        logger.warning(f"[BRANCH] {i} {snap_row['id']} {fam} x{post_mul:g} failed: {fail}")
        return {**row, "status": "failed", "reason": fail, "reason_key": key}
    recording = finish_recording(sim, ctx.out, label, "episode done")

    t3 = time.perf_counter()
    frames = sampler.frames
    pre = np.array([phase_labels[fr["phase"]] == col.PREHOLD_PHASE for fr in frames])
    hold = np.array([phase_labels[fr["phase"]] == phase_labels[1] for fr in frames])
    u_all = np.stack([col.frame_action(fr, "cmd_delta") for fr in frames])
    if np.any(u_all[pre] != 0.0) or np.any(u_all[hold] != 0.0):
        raise RuntimeError("hold cmd_delta not 0")
    label_out, metrics, min_clear_mm, contact = col._classify(sampler, ctx.thresholds,
                                                              ctx.tangent)
    nn = np.asarray(metrics["n_neighbour"])
    in_contact = nn >= col.IN_CONTACT_MIN_NEIGHBOUR
    rod = np.asarray(rods)
    n_pre = int(pre.sum())
    track = col.tracking_errors(frames)
    moving = ~pre & ~hold
    stats = {
        "frames": len(frames), "pre_hold_frames": n_pre, "hold_frames": int(hold.sum()),
        "contact_frame_frac": float(in_contact.mean()),
        "wrap_deg_min": float(np.min(metrics["wrap_deg"])),
        "wrap_deg_max": float(np.max(metrics["wrap_deg"])),
        "h_median_mm_min": float(np.nanmin(metrics["h_median_mm"])) if np.isfinite(
            metrics["h_median_mm"]).any() else None,
        "h_median_mm_max": float(np.nanmax(metrics["h_median_mm"])) if np.isfinite(
            metrics["h_median_mm"]).any() else None,
        "u_bound_ratio_max": float(cmp.bound_ratio(u_all, lb, ub).max()),
        "u_plan_bound_ratio_max": ratio_plan,
        "rod_stretch_gain_pct_max": float((rod - rod[0]).max()),
        "ur_guard_scale_min": float(min(scales)),
        "min_board_clearance_mm": float(min_clear_mm), "board_contact": bool(contact),
        "min_franka_tip_clearance_mm": float(state["tip_min"] * 1e3),
        "franka_floor_mm": floor_mm, "grasp_ok_all": bool(all(all(f["grasp_ok"]) for f in frames)),
        "tracking_rms_mm": [float(np.sqrt(np.mean(track[moving][1:, j] ** 2))) for j in range(2)],
        "outcome_start": snap_row["label"], "outcome_settled": classify(fm_start),
        "outcome": label_out, "final_wrap_deg": float(metrics["wrap_deg"][-1]),
    }
    branch = {"snapshot": snap_row["id"], "snapshot_file": snap_row["file"],
              "event": snap_row["event"], "class": snap_row["class"],
              "source_label": snap_row["label"], "source_episode": snap_row["source_episode"],
              "family": fam, "params": par, "caps": {"franka_mm": CAP[0], "franka_mrad": CAP[1],
                                                     "ur_mm": CAP[2], "ur_mrad": CAP[3]},
              "retry_mul": post_mul, "T_s": T, "group_scale": scale, "amp_effective": eff,
              "unit": {c.name: "mm" if c.kind == "trans" else "deg" for c in comps},
              "holds": [{**h, "t0_s": h["s0"] * T, "t1_s": h["s1"] * T} for h in holds],
              "hold_end": hold_end, "extra": extra, "settle_s": SETTLE_S,
              "pre_hold_s": PRE_HOLD_S, "hold_s": HOLD_S,
              "plan_min_2f85_clearance_mm": clear * 1e3, "clearance_floor_mm": floor_c * 1e3,
              "plan_franka_tip_floor_mm": tip * 1e3, "franka_floor_mm": floor_mm,
              "crop_top_ref_m": z_top_ref, "metrics_settled": fm_dict(fm_start)}
    extras_meta = {
        "phase_labels": phase_labels, "intent": fam, "outcome": label_out,
        "perturbation": {"intent": fam}, "backend": "osc", "scenario": "contact_branch",
        "branch": branch,
        "ee_pose_source": "measured finger_tip / tracking frame (FK of the measured joints)",
        "action_source": "cmd_delta of the scripted pose: Franka knot 1 - knot 0 (knot j = "
                         "pose(t + j dt)); UR line pose(t) -> pose(t + dt)",
        "pre_hold_frames": n_pre, "osc": ctx.osc, "commander": dataclasses.asdict(params),
        "excitation": None, "sample_period_s": dt_s,
        "time_source": "OSC clock (FRANKA_STATE utime)", "pulley_pose_layout": lcs.POSE_LAYOUT,
        "start_step": s0, "start_state": str(ctx.start_state),
        "thresholds": dataclasses.asdict(ctx.thresholds), "clamp": None, "stats": stats,
        "in_contact_rule": f"n_neighbour >= {col.IN_CONTACT_MIN_NEIGHBOUR}",
    }
    post = {"backend": np.array("osc"),
            "osc_utime_offset_us": np.int64(sim.bridge.utime_offset_us),
            "min_franka_tip_clearance_mm": np.float32(state["tip_min"] * 1e3),
            "family": np.array(fam), "outcome_start": np.array(snap_row["label"]),
            "n_neighbour": nn.astype(np.int32), "in_contact": in_contact,
            "branch": np.array(json.dumps(branch, default=float)),
            "rod_stretch_pct": np.concatenate([np.full(n_pre, np.nan), rod]),
            "motion_offset_franka": np.concatenate([np.zeros((n_pre, 6)), np.stack(offs_f)]),
            "motion_offset_ur": np.concatenate([np.zeros((n_pre, 6)), np.stack(offs_u)]),
            "ur_guard_scale": np.concatenate([np.ones(n_pre), np.asarray(scales)])}
    path = ctx.out / row["file"]
    writer, _ = col.build_osc_writer(frames, n, pcd=not args.no_pcd, definition="cmd_delta")
    timing = {"sim": t3 - t0}
    col._write_episode(i, pert.Perturbation(intent=fam), path, writer, label_out, extras_meta,
                       metrics, min_clear_mm, contact, None, post, args.no_pcd, n, timing, t0,
                       t3, f", {snap_row['id']} {snap_row['label']} -> {fam}")
    return {**row, "status": "ok", "outcome": label_out, "outcome_start": snap_row["label"],
            "stats": stats, "branch": branch, "recording": recording,
            "size_bytes": path.stat().st_size, "timing_s": timing}


def run_branch(args) -> int:
    check_private_url(args.lcm_url)
    snap_dir = Path(args.snap_dir)
    sindex = json.loads((snap_dir / "index.json").read_text())
    by_id = {r["id"]: r for r in sindex["variants"]}
    ws = json.loads(Path(args.working_set).read_text())
    plan = [(by_id[p["id"]], p["family"]) for p in ws["plan"]]
    if args.families:
        keep = set(args.families.split(","))
        plan = [x for x in plan if x[1] in keep]
    if args.limit:
        plan = plan[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.action_definition = "cmd_delta"
    n = col.sample_steps(lcs.SAMPLE_PERIOD_S)
    nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=col.FIRST, last=col.LAST)
    index_path = out / "index.json"
    index = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                      if k != "func"},
             "git": git_info(), "kind": "contact_branch", "backend": "osc",
             "snapshots": {"dir": str(snap_dir), "index_sha256": sha256_file(snap_dir /
                                                                             "index.json")},
             "working_set": str(args.working_set), "action_definition": "cmd_delta",
             "sample_period_s": n * lcs.SIM_DT_S, "sample_steps": n,
             "belt_sampling": lcs.BELT_SAMPLING, "caps": list(CAP),
             "crop_box": {"lo": cmp.CROP_LO.tolist(), "hi": cmp.CROP_HI.tolist()},
             "demo": {"file": str(DEMO), "frame": DEMO_FRAME}, "episodes": []}
    if args.append and index_path.exists():
        old = json.loads(index_path.read_text())
        index["episodes"] = old["episodes"]
        index["appended_runs"] = old.get("appended_runs", []) + [
            {"args": old["args"], "summary": old.get("summary")}]
    write_json(index_path, index)
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=5.0)
    sim.args.record_state_every = col.RECORD_STATE_EVERY
    t_run = time.perf_counter()
    reason = "error"
    demo = demo_poses()
    try:
        warm = sim.start_osc(out / ("osc_append.log" if args.append else "osc.log"))
        osc_info = {**sim.osc.describe(), "warm_up_s": warm}
        index["osc"] = osc_info
        first = sim_snapshot.load(snap_dir / plan[0][0]["file"])
        sim.restore(first)
        jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
        board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
        ctx = col.OscContext(args=args, out=out, n=n, thresholds=DEFAULT_THRESHOLDS,
                             tangent=pert.belt_tangent(nominal), snap=first, nominal=nominal,
                             board=board, gripper=gripper, excite={"on": False}, scenario=None,
                             osc=osc_info, start_state=snap_dir / plan[0][0]["file"],
                             params=CommanderParams(), x_tool0=x_tool0_tracking())
        gauge = cmp.RodGauge(sim)
        i = sum(r["status"] == "ok" for r in index["episodes"])
        fam_seen = Counter(r["family"] for r in index["episodes"] if r.get("attempt", 0) == 0)
        for srow, fam in plan:
            snap = sim_snapshot.load(snap_dir / srow["file"])
            ctx.snap, ctx.start_state = snap, snap_dir / srow["file"]
            key = [args.seed, FAMILIES.index(fam), fam_seen[fam]]
            fam_seen[fam] += 1
            par = draw_family(fam, key)
            for attempt, mul in enumerate((1.0, RETRY_MUL)):
                row = rollout(sim, ctx, i, srow, snap, fam, par, mul, gauge, demo)
                row["attempt"] = attempt
                row["rng_key"] = key
                index["episodes"].append(row)
                write_json(index_path, index)
                if row["status"] == "ok":
                    i += 1
                    break
                if row.get("reason_key") == "grasp_settle":
                    break
        index["osc_log_errors"] = [e for e in col.OSC_LOG_ERRORS if e in sim.osc.log_text()]
        reason = "finished"
    finally:
        sim.close(reason)
    ok = [r for r in index["episodes"] if r["status"] == "ok"]
    index["summary"] = {"episodes_ok": len(ok),
                        "failed_attempts": len(index["episodes"]) - len(ok),
                        "families_ok": dict(Counter(r["family"] for r in ok)),
                        "wall_s": time.perf_counter() - t_run}
    write_json(index_path, index)
    logger.info(f"[BRANCH] {out}: {index['summary']}")
    return 0


# ---- report ---------------------------------------------------------------------------------

def _load(run: Path, row: dict) -> dict:
    with np.load(run / row["file"], allow_pickle=True) as d:
        meta = json.loads(str(d["sim_meta"]))
        labels = meta["phase_labels"]
        ph = np.array([labels[k] for k in d["sim_phase"]])
        return {"phase": ph, "u": d["actions"].astype(np.float64),
                "r": d["sim_realised_delta"].astype(np.float64),
                "wrap": d["sim_wrap_deg"].astype(np.float64),
                "h": d["sim_h_median_mm"].astype(np.float64),
                "in_contact": d["sim_in_contact"].astype(bool),
                "grasp": d["sim_grasp_ok"], "rod": d["sim_rod_stretch_pct"]}


def run_report(args) -> int:
    runs = [Path(r) for r in args.runs]
    for run in runs:
        index = json.loads((run / "index.json").read_text())
        rows = index["episodes"]
        ok = [r for r in rows if r["status"] == "ok"]
        arrs = [_load(run, r) for r in ok]
        qc = {"run": str(run), "episodes_ok": len(ok),
              "attempts_failed": len(rows) - len(ok),
              "failed_reasons": dict(Counter(r.get("reason_key") for r in rows
                                             if r["status"] != "ok"))}
        # Final failures: (snapshot, family) with no ok attempt.
        done = {(r["snapshot"], r["family"]) for r in ok}
        lost = {(r["snapshot"], r["family"]): r.get("reason_key") for r in rows
                if r["status"] != "ok" and (r["snapshot"], r["family"]) not in done}
        qc["rollouts_failed"] = {f"{s}|{f}": k for (s, f), k in lost.items()}
        qc["grasp_losses"] = sum(k in ("grasp_lost", "grasp_settle") for k in lost.values())
        qc["grasp_loss_attempts"] = sum(r.get("reason_key") in ("grasp_lost", "grasp_settle")
                                        for r in rows)
        qc["board_contacts"] = sum(r.get("reason_key") == "board_contact" for r in rows)
        cells = defaultdict(list)
        for r, a in zip(ok, arrs, strict=True):
            cells[(r["family"], r["class"])].append((r, a))
        table = {}
        for (fam, cls), items in sorted(cells.items()):
            fr = sum(len(a["phase"]) for _, a in items)
            cf = sum(int(a["in_contact"].sum()) for _, a in items)
            table[f"{fam}|{cls}"] = {
                "ok": len(items), "frames": fr, "contact_frame_frac": cf / fr,
                "wrap_range_per_ep": [[round(float(a["wrap"].min()), 1),
                                       round(float(a["wrap"].max()), 1)] for _, a in items],
                "h_range_mm": [float(np.nanmin([np.nanmin(a["h"]) for _, a in items])),
                               float(np.nanmax([np.nanmax(a["h"]) for _, a in items]))],
                "transitions": dict(Counter(f"{r['outcome_start']}->{r['outcome']}"
                                            for r, _ in items))}
        qc["family_event"] = table
        qc["family_ok"] = dict(Counter(r["family"] for r in ok))
        wraps = np.concatenate([a["wrap"] for a in arrs]) if arrs else np.zeros(0)
        edges = np.arange(0.0, 360.0 + 15.0, 15.0)
        hist, _ = np.histogram(wraps, edges)
        qc["wrap_hist_15deg"] = {f"{int(lo)}-{int(lo + 15)}": int(c)
                                 for lo, c in zip(edges[:-1], hist, strict=True) if c}
        tot = max(1, len(wraps))
        qc["wrap_frac_30deg_bins"] = {f"{lo}-{lo + 30}": float(((wraps >= lo)
                                                                & (wraps < lo + 30)).sum() / tot)
                                      for lo in range(0, 150, 30)}
        qc["wrap_max"] = float(wraps.max()) if len(wraps) else None
        all_f = sum(len(a["phase"]) for a in arrs)
        qc["frames"] = all_f
        qc["contact_frame_frac"] = (sum(int(a["in_contact"].sum()) for a in arrs) / all_f
                                    if all_f else None)
        trans = Counter((r["outcome_start"], r["outcome"]) for r in ok)
        qc["transitions"] = {f"{a}->{b}": c for (a, b), c in sorted(trans.items())}
        rec = defaultdict(lambda: [0, 0])
        for r in ok:
            if r["outcome_start"] in FAIL_LABELS and r["class"].startswith("final_"):
                rec[r["family"]][0] += r["outcome"] == "engaged"
                rec[r["family"]][1] += 1
        qc["recovery_success"] = {f: {"engaged": e, "n": m, "frac": e / m if m else None}
                                  for f, (e, m) in rec.items()}
        qc["stretch_gain_max_pct"] = max((r["stats"]["rod_stretch_gain_pct_max"] for r in ok),
                                         default=None)
        qc["grasp_ok_all_frames"] = all(r["stats"]["grasp_ok_all"] for r in ok)
        qc["min_board_clearance_mm"] = min((r["stats"]["min_board_clearance_mm"] for r in ok),
                                           default=None)
        qc["min_franka_tip_mm"] = min((r["stats"]["min_franka_tip_clearance_mm"] for r in ok),
                                      default=None)
        qc["u_bound_ratio_max"] = max((r["stats"]["u_bound_ratio_max"] for r in ok),
                                      default=None)
        hold_bad = sum(int(np.any(a["u"][(a["phase"] == col.PREHOLD_PHASE)
                                         | np.char.startswith(a["phase"].astype(str), "hold:")]
                                  != 0.0)) for a in arrs)
        qc["hold_rows"] = {"rows": int(sum(((a["phase"] == col.PREHOLD_PHASE)
                                            | np.char.startswith(a["phase"].astype(str),
                                                                 "hold:")).sum()
                                           for a in arrs)),
                           "episodes_with_nonzero_u": hold_bad}
        mv = [np.char.startswith(a["phase"].astype(str), "move:") for a in arrs]
        if arrs:
            sl = lcs.causality_slopes(np.vstack([a["u"][m] for a, m in zip(arrs, mv)]),
                                      np.vstack([a["r"][m] for a, m in zip(arrs, mv)]))
            qc["causality_motion_rows"] = {
                "rows": sl[0]["rows"], "gate": [0.8, 1.1],
                "slopes": {x["dim"]: round(x["slope"], 4) for x in sl},
                "u_std": {x["dim"]: round(x["u_std"] * 1e3, 4) for x in sl},
                "pass": bool(all(0.8 <= x["slope"] <= 1.1 for x in sl))}
        write_json(run / "qc.json", qc)
        print(json.dumps({k: v for k, v in qc.items() if k not in ("family_event",)},
                         indent=1, default=float))
    if args.rebalance:
        rb = rebalance(runs)
        write_json(Path(args.rebalance), rb)
        print(json.dumps(rb["summary"], indent=1))
    return 0


def _contact(files: list, tail: bool = False) -> dict:
    """Frames, ``n_neighbour >= 3`` frames and strict (``and h_min <= 10 mm``) frames; with
    ``tail`` also those after ``sim_place3_frame``."""
    out = Counter()
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            B, P = d["sim_belt_xyz"], d["sim_pulley_large_pose"]
            p3 = int(d["sim_place3_frame"]) if tail else None
        P = np.concatenate([P[:, :3], P[:, 4:7], P[:, 3:4]], 1)  # xyz_wxyz -> xyz_xyzw
        fms = [frame_metrics(b, q) for b, q in zip(B, P, strict=True)]
        nn = np.array([m.n_neighbour >= col.IN_CONTACT_MIN_NEIGHBOUR for m in fms])
        st = nn & np.array([m.h_min_mm <= FC_H_MIN_MM or m.wrap_deg > 0.0 for m in fms])
        out["frames"] += len(nn)
        out["contact"] += int(nn.sum())
        out["contact_strict"] += int(st.sum())
        if tail:
            out["tail_frames"] += len(nn) - p3 - 1
            out["tail_contact"] += int(nn[p3 + 1:].sum())
            out["tail_contact_strict"] += int(st[p3 + 1:].sum())
    return {"files": len(files), **out}


def rebalance(branch_runs: list[Path]) -> dict:
    split = json.loads((DATA / "v2" / "split.json").read_text())
    out = {"rule": f"contact = n_neighbour >= {col.IN_CONTACT_MIN_NEIGHBOUR} (radial band only, "
                   "the sim_in_contact rule); contact_strict = contact and (h_min_mm <= "
                   f"{FC_H_MIN_MM:g} or wrap > 0) (the first_contact rule)",
           "unit": "frames", "sets": {}}
    for part in ("train", "val", "test"):
        out["sets"][f"v2_insertion_{part}"] = _contact(split[part])
    for name, pats in (("approach_train", ["v1/train", "v1/train_topup_*"]),
                       ("approach_heldout", ["v1_heldout", "v1_heldout_topup_*"])):
        files = sorted(f for p in pats for f in (DATA / "approach").glob(f"{p}/episode_*.npz"))
        out["sets"][name] = _contact(files, tail=True)
    for name in ("v1", "v1_heldout"):
        files = sorted((DATA / "free_space" / name).glob("*/episode_*.npz"))
        out["sets"][f"free_space_{'train' if name == 'v1' else 'heldout'}"] = _contact(files)
    for run in branch_runs:
        out["sets"][f"branch_{run.name}"] = _contact(sorted(run.glob("episode_*.npz")))
    s = out["sets"]
    for v in s.values():
        for k in ("contact", "contact_strict", "tail_contact", "tail_contact_strict"):
            if k in v:
                base = v["tail_frames"] if k.startswith("tail") else v["frames"]
                v[f"{k}_share"] = v[k] / base if base else None
    train = ["v2_insertion_train", "approach_train", "free_space_train"] + [
        k for k in s if k.startswith("branch_") and "heldout" not in k]

    def union(keys):
        fr = sum(s[k]["frames"] for k in keys)
        c = sum(s[k]["contact"] for k in keys)
        cs = sum(s[k]["contact_strict"] for k in keys)
        return {"sets": keys, "frames": fr, "contact": c, "share": c / fr,
                "contact_strict": cs, "share_strict": cs / fr}

    out["summary"] = {
        "union_train": union(train),
        "union_train_without_free_space": union([k for k in train if k != "free_space_train"]),
        "per_set_share": {k: [round(v["contact_share"], 4), round(v["contact_strict_share"], 4)]
                          for k, v in s.items()},
        "target": 0.35}
    out["summary"]["target_met"] = out["summary"]["union_train"]["share"] >= 0.35
    out["summary"]["target_met_strict"] = out["summary"]["union_train"]["share_strict"] >= 0.35
    return out


# ---- cli ------------------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="phase", required=True)
    s = sub.add_parser("snap")
    s.add_argument("--lcm-url", required=True)
    s.add_argument("--src", required=True, help="source name in snapshot ids")
    s.add_argument("--episodes", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--start-states", type=Path, default=None)
    s.add_argument("--variants", default=None)
    s.add_argument("--snap-dir", type=Path, default=CONTACT / "snapshots")
    s.add_argument("--source-out", type=Path, required=True)
    s.set_defaults(func=run_snap)
    s = sub.add_parser("select")
    s.add_argument("--snap-dir", type=Path, default=CONTACT / "snapshots")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--all", action="store_true", help="every validated snapshot, 1 rollout")
    s.set_defaults(func=run_select)
    s = sub.add_parser("branch")
    s.add_argument("--lcm-url", required=True)
    s.add_argument("--snap-dir", type=Path, default=CONTACT / "snapshots")
    s.add_argument("--working-set", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--append", action="store_true")
    s.add_argument("--families", default=None, help="only these families of the plan")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--no-pcd", action="store_true")
    s.add_argument("--record", action="store_true")
    s.add_argument("--min-clearance", type=float, default=col.clr.DEFAULT_MIN_CLEARANCE_MM)
    s.set_defaults(func=run_branch)
    s = sub.add_parser("report")
    s.add_argument("runs", nargs="+")
    s.add_argument("--rebalance", type=Path, default=None)
    s.set_defaults(func=run_report)
    args = p.parse_args()
    col.configure_logging()
    if getattr(args, "lcm_url", None):
        want = URL_SNAP if args.phase == "snap" else URL_BRANCH
        if f":{want}" not in args.lcm_url:
            logger.warning(f"[CONTACT] {args.phase} expects URL port {want}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
