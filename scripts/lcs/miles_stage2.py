#!/usr/bin/env python3
"""Stage-2 MILES (V12-125): deviate-and-return branches off whole-run stage-2 demos, one scene.

The opt-in stage-2 path of ``miles_seed.py`` (its helpers, file format and buffer rows; stage 1 is
untouched). Per demo (``--demo NAME=EPISODE,STATE``: a whole-run chain episode and its
``eval_learned_mpc.py --stage2-start-snapshot`` state; scripted cmd_source 5 or SpaceMouse 9):
restore the stage-2 start, settle ``--settle-s``, replay the demo's commanded Franka knot-0 path
(frames ``sim_chain_from[1]`` .. ``sim_post_from``) with the parked UR fixed, snapshotting every
``--wp-every`` frames. Then per waypoint, ``--branches`` branches: restore the waypoint, the demo
path continues underneath a Franka-only min-jerk offset (``miles_seed.draw_offset``, box cycled
over ``--boxes``, clearance-scaled) 0 -> D over ``--dev-s``, D -> 0 over ``--ret-s``, then the plain
demo path for ``--tail-s``. A branch ends early on a drive abort (grasp lost, tip floor, board) or
``--peel-tail-s`` after the large pulley is off (v12_109's large_off rule). Snapshot restores stay
in the one scene (data collection only).

Writes ``<out>/miles/<name>_walk.npz`` (the clean replay, fidelity check) and
``<name>_wp<k>_c<c>.npz`` (``miles_seed.write_file``: cmd_source 4, segments walk / dev / return /
hold = the post-return tail), ``buffer.json`` (kind miles, ``miles.stage`` 2), ``index.json``.

Run (``@file``: one ``--demo=...`` per line):
    uv run --frozen python scripts/lcs/miles_stage2.py --out DIR --lcm-url URL @demos.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_lcs_dataset as cli
import demo_prefix as dp
import eval_learned_mpc as harness
import miles_seed as ms
import stage2_collect as s2
from loguru import logger
from make_ee_offset_starts import min_jerk

from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import UrTracking
from round_belt_task.episode_io import git_info, pose7_mat, sha256_file, write_json
from round_belt_task.outcome import DEFAULT_THRESHOLDS, classify, frame_metrics
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot

OFF_WRAP_DEG, OFF_MARGIN = 30.0, 7  # v12_109 cut_of: large off = wrap < 30 deg for 8 frames
NEAR_WRAP_DEG = DEFAULT_THRESHOLDS.engaged_arc_deg
STAGE2_CMD = (s2.CMD_STAGE2, harness.CMD_TELEOP)
UR_BYTE = 191  # parked, open (the chain's --ur-exit-byte)


class _Stop(Exception):
    pass


def load_demo(spec: str) -> dict:
    name, rest = spec.split("=", 1)
    ep, state = (REPO_ROOT / Path(v.strip()) for v in rest.split(","))
    with np.load(ep, allow_pickle=True) as z:
        cs = np.asarray(z["sim_cmd_source"])
        c1 = int(z["sim_chain_from"][1])
        pf = int(z["sim_post_from"]) if "sim_post_from" in z.files else len(cs)
        if not (c1 < pf and np.isin(cs[c1:pf], STAGE2_CMD).all()):
            raise ValueError(f"{ep}: frames {c1}..{pf} are not one stage-2 segment ({set(cs[c1:pf])})")
        d = {"name": name, "episode": ep, "state": state, "from": c1, "to": pf,
             "knot0": np.asarray(z["sim_cmd_knot0_franka"][c1:pf], np.float64),
             "belt": np.asarray(z["sim_belt_xyz"][c1:pf], np.float64),
             "cmd_source": int(cs[c1]),
             "success_both": bool(z["sim_success_both"]) if "sim_success_both" in z.files else None,
             "start": json.loads(str(z["sim_start_state"]))}
    if not state.is_file():
        raise FileNotFoundError(state)
    d["path"] = dp.KnotPath(d["knot0"], dp.FRAME_DT_S)
    return d


class Shifted:
    """``base`` from ``t0``; plus a Franka offset 0 -> ``off`` (min-jerk, ``dev_s``) -> 0 (``ret_s``)."""

    def __init__(self, base, t0: float, off6=None, dev_s: float = 0.0, ret_s: float = 0.0) -> None:
        self.base, self.t0 = base, float(t0)
        self.off = None if off6 is None else np.asarray(off6, np.float64)
        self.t1, self.t2 = float(dev_s), float(dev_s + ret_s)

    def gain(self, tau: float) -> float:
        if self.off is None or tau >= self.t2:
            return 0.0
        return min_jerk(tau / self.t1) if tau < self.t1 else 1.0 - min_jerk((tau - self.t1) / (self.t2 - self.t1))

    def segment(self, tau: float) -> int:
        return ms.SEGMENTS.index("walk" if self.off is None else "dev" if tau < self.t1
                                 else "return" if tau < self.t2 else "hold")

    def pose7(self, tau: float) -> np.ndarray:
        p = self.base.pose7(self.t0 + tau)
        s = self.gain(tau)
        return ms.offset_pose(p, self.off, s) if s > 0.0 else p

    def __call__(self, tau: float):
        p = self.pose7(tau)
        return p[:3], p[3:]


class Watch:
    """Per-frame large-pulley wrap; raises ``_Stop`` ``peel_tail`` frames after large_off."""

    def __init__(self, rec, peel_tail: int | None, before=None) -> None:
        self.rec, self.peel_tail, self.before, self.wrap, self.run = rec, peel_tail, before, [], 0

    def __call__(self, k, cmd, tau) -> None:
        if self.before is not None:
            self.before(tau)
        self.rec(k, cmd, tau)
        fr = self.rec.frames[-1]
        w = float(frame_metrics(fr["belt"], fr["pulley_raw"], th=DEFAULT_THRESHOLDS).wrap_deg)
        self.wrap.append(w)
        self.run = self.run + 1 if w < OFF_WRAP_DEG else 0
        if self.peel_tail is not None and self.run >= OFF_MARGIN + 1 + self.peel_tail:
            raise _Stop

    def large_off(self) -> int | None:
        """v12_109's index (first off frame + 1 + margin), or None."""
        w = np.asarray(self.wrap) < OFF_WRAP_DEG
        for t in range(len(w)):
            if w[t:t + OFF_MARGIN + 1].all() and len(w[t:t + OFF_MARGIN + 1]) == OFF_MARGIN + 1:
                return t + 1 + OFF_MARGIN
        return None


def drive(sim, ctx, path, ur, steps: int, watch, grasp_arms: str | None = "franka") -> dict:
    """``miles_seed.drive`` (Franka grasp only); a ``_Stop`` ends it early (no abort)."""
    try:
        return ms.drive(sim, ctx, path, ur, steps, watch, grasp_arms=grasp_arms)
    except _Stop:
        fr = watch.rec.frames
        return {"steps": (len(fr) - 1) * harness.SAMPLE_STEPS, "abort": None, "abort_kind": None,
                "tip_min_mm": min(f["tip_clearance_mm"] for f in fr), "min_clearance_mm": math.inf,
                "stopped": "peel"}


def end_labels(sim) -> dict:
    body_q = sim.state_0.body_q.numpy()
    belt = body_q[sim.info.belt_bodies, :3].astype(np.float64)
    fl = frame_metrics(belt, body_q[int(sim.info.pulley_bodies[1])].astype(np.float64))
    fs = frame_metrics(belt, body_q[int(sim.info.pulley_bodies[0])].astype(np.float64),
                       harness.SMALL_SEAT_MM)
    return {"large": classify(fl), "small": classify(fs), "large_wrap_deg": float(fl.wrap_deg),
            "held_franka": bool(sim.grasp_state(body_q).held_for("franka"))}


def outcome(rep: dict, watch: Watch) -> str:
    """peel (large_off), near_peel (large wrap < engaged arc, not off), grasp_lost / abort, engaged."""
    if watch.large_off() is not None:
        return "peel"
    if rep["abort_kind"] == "grasp_lost":
        return "grasp_lost"
    if rep["abort"]:
        return "abort"
    return "near_peel" if min(watch.wrap) < NEAR_WRAP_DEG else "engaged"


def run_demo(args, sim, ctx, di: int, demo: dict, col: dict) -> None:
    name, n = demo["name"], len(demo["knot0"])
    if args.only and not any(o.startswith(f"{name}:") for o in args.only):
        return
    start = {"set": "stage2_demo", "id": name, "file": str(demo["state"].relative_to(REPO_ROOT)),
             "episode": str(demo["episode"].relative_to(REPO_ROOT)), "frame": demo["from"]}
    ctx.cloud_rng = np.random.default_rng([di, cli.CLOUD_NOISE_RNG])
    sim.restore(sim_snapshot.load(demo["state"]), settle_steps=0)
    grasp = sim.settle(round(args.settle_s / sim.frame_dt))
    drow = {"demo": name, "episode": start["episode"], "episode_sha256": sha256_file(demo["episode"]),
            "state": start["file"], "frames": [demo["from"], demo["to"]], "cmd_source": demo["cmd_source"],
            "success_both": demo["success_both"], "start": demo["start"], "held": grasp.held_for("franka")}
    col["demos"].append(drow)
    if not drow["held"]:
        drow["why"] = "grasp lost after the restore"
        return
    ur = s2.Fixed(pose7_mat(UrTracking.fk(sim.arm_targets()[1])))
    wps = list(range(args.wp_first, n - 1, args.wp_every))
    t0 = time.perf_counter()
    for attempt in range(args.walk_tries):
        if attempt:  # the OSC loop is not bit-repeatable: a fresh replay from the same start (V12-147)
            sim.restore(sim_snapshot.load(demo["state"]), settle_steps=0)
            sim.settle(round(args.settle_s / sim.frame_dt))
        snaps = {}
        rec = ms.FileRec(sim, ctx, lambda tau: 0, lambda tau: round(tau / dp.FRAME_DT_S))

        def snap(tau, snaps=snaps):
            j = round(tau / dp.FRAME_DT_S)
            if j in wps and j not in snaps:
                snaps[j] = sim_snapshot.capture(sim, f"miles2_{name}_{j}")

        watch = Watch(rec, None, snap)
        rep = ms.drive(sim, ctx, Shifted(demo["path"], 0.0), ur, (n - 1) * harness.SAMPLE_STEPS, watch,
                       grasp_arms="franka")
        m = len(rec.frames)
        belt_rmse = [ms.gt_rmse_mm(rec.frames[j]["belt"], demo["belt"][j]) for j in range(m)]
        off_demo = args.walk_max_rmse_mm is not None and max(belt_rmse) > args.walk_max_rmse_mm
        if args.walk_only:  # reproducibility check: every try, no files
            drow.setdefault("walk_rmse_max_mm", []).append(float(max(belt_rmse)))
            logger.info(f"[MILES2] {name} walk-only try {attempt + 1}: belt RMSE vs demo max {max(belt_rmse):.1f} mm "
                        f"({'off' if off_demo else 'on'} the demo)")
            if attempt + 1 < args.walk_tries:
                continue
            n_on = sum(x <= args.walk_max_rmse_mm for x in drow["walk_rmse_max_mm"])
            logger.info(f"[MILES2] {name} WALK_PASSES {n_on}/{args.walk_tries}")
            return
        if not off_demo:
            break
        logger.warning(f"[MILES2] {name} walk try {attempt + 1}/{args.walk_tries}: belt RMSE vs demo max "
                       f"{max(belt_rmse):.1f} > {args.walk_max_rmse_mm:g} mm")
        if attempt + 1 < args.walk_tries:  # the last try is counted below
            col["robot_s"] += args.settle_s + rep["steps"] * sim.frame_dt
    drow["walk_tries"] = attempt + 1
    lab = end_labels(sim)
    miles = {"demo": name, "stage": 2, "segment": "walk", "from_frame": 0, "to_frame": m - 1,
             "start": start, "abort": rep["abort"], "end": lab}
    drow.update(walk={"frames": m, "abort": rep["abort"], "end": lab, "wall_s": time.perf_counter() - t0,
                      "belt_rmse_vs_demo_mm": {"p50": float(np.median(belt_rmse)), "max": float(max(belt_rmse)),
                                               "end": belt_rmse[-1]},
                      "min_large_wrap_deg": float(min(watch.wrap))},
                waypoints=sorted(snaps), branches=0)
    col["robot_s"] += args.settle_s + rep["steps"] * sim.frame_dt
    logger.info(f"[MILES2] {name} walk: {m} frames, abort {rep['abort']}, end {lab}, belt RMSE vs demo "
                f"p50 {np.median(belt_rmse):.2f} max {max(belt_rmse):.2f} mm")
    if off_demo:
        drow["why"] = f"walk off the demo in {args.walk_tries} tries: no branches"
        logger.warning(f"[MILES2] {name}: {drow['why']}")
        return
    # a walk off the demo writes no file: it is not the demo's path
    col["files"].append({**(ms.write_file(ctx, args.out / "miles" / f"{name}_walk.npz", rec.frames,
                                          rep, miles) or {}), "miles": miles})
    sim.restore(snaps[wps[0]], settle_steps=0)
    clr = ms.Clearance(sim, ctx.plate_top_z)
    peel_tail = round(args.peel_tail_s / dp.FRAME_DT_S)
    for k in sorted(snaps):
        W_f = demo["path"].pose7(k * dp.FRAME_DT_S)
        for c in range(args.branches):
            if args.only and f"{name}:{k}:{c}" not in args.only:
                continue
            box = args.boxes[c % len(args.boxes)]
            rng = np.random.default_rng([args.seed, di, k, c])
            off = ms.draw_offset(rng, "franka", box, args.emphasis_frac)
            sc = clr.scale("franka", W_f, off["franka"])
            applied = off["franka"] * sc[0]
            path = Shifted(demo["path"], k * dp.FRAME_DT_S, applied, args.dev_s, args.ret_s)
            dur = args.dev_s + args.ret_s + args.tail_s
            sim.restore(snaps[k], settle_steps=0)
            rec = ms.FileRec(sim, ctx, path.segment, lambda tau, k=k: k + round(tau / dp.FRAME_DT_S))
            watch = Watch(rec, peel_tail)
            t1 = time.perf_counter()
            rep = drive(sim, ctx, path, ur, harness.SAMPLE_STEPS * round(dur / dp.FRAME_DT_S), watch,
                        grasp_arms=None if args.no_grasp_abort else "franka")
            col["robot_s"] += rep["steps"] * sim.frame_dt
            why = outcome(rep, watch)
            lab = end_labels(sim)
            br = {"demo": name, "stage": 2, "segment": "cycle", "waypoint": k, "cycle": c, "mode": "franka",
                  "box": list(box), "offset_drawn": off["franka"].tolist(), "offset": applied.tolist(),
                  "single_axis": off["single_axis"], "scale": sc[0], "clearance_model_mm": [sc[1], sc[2]],
                  "abort": rep["abort"], "abort_kind": rep["abort_kind"], "stopped": rep.get("stopped"),
                  "outcome": why, "large_off": watch.large_off(), "min_large_wrap_deg": float(min(watch.wrap)),
                  "end": lab, "frames": len(rec.frames), "start": start, "valid": why == "engaged",
                  "why": None if why == "engaged" else why,
                  "dev_s": args.dev_s, "ret_s": args.ret_s, "tail_s": args.tail_s}
            row = ms.write_file(ctx, args.out / "miles" / f"{name}_wp{k:03d}_c{c}.npz", rec.frames, rep, br)
            br["file"] = None if row is None else row["file"]
            br["wall_s"] = time.perf_counter() - t1
            if row is not None:
                col["files"].append({**row, "miles": br})
            col["branches"].append(br)
            drow["branches"] += 1
            logger.info(f"[MILES2] {name} wp {k} c{c} box {box} |dF| {np.linalg.norm(applied[:3]):4.1f} mm "
                        f"{np.linalg.norm(applied[3:]):4.1f} deg (scale {sc[0]:.2f}): {why}, min wrap "
                        f"{br['min_large_wrap_deg']:.0f}, off {br['large_off']}, {len(rec.frames)} frames, "
                        f"{br['wall_s']:.1f} s wall")


def run_demo_continuous(args, sim, ctx, di: int, demo: dict, col: dict) -> None:
    """``--continuous``: stage-1 style, no snapshot branching. One walk of the demo path; at each waypoint
    the path pauses for ``--branches`` cycles (``miles_seed.CyclePath``: 0 -> D -> 0, hold ``--tail-s``) with
    the same seeded offsets. Near-peel or a segmented-cloud Chamfer vs the arrival cloud > ``--tau-mm`` skips
    the waypoint's other cycles (the walk goes on); peel, grasp lost, an abort, or an arrival belt RMSE vs
    the demo > ``--walk-max-rmse-mm`` is a human reset to the stage-2 start (the next waypoint)."""
    name, n, dt = demo["name"], len(demo["knot0"]), dp.FRAME_DT_S
    start = {"set": "stage2_demo", "id": name, "file": str(demo["state"].relative_to(REPO_ROOT)),
             "episode": str(demo["episode"].relative_to(REPO_ROOT)), "frame": demo["from"]}
    ctx.cloud_rng = np.random.default_rng([di, cli.CLOUD_NOISE_RNG])
    drow = {"demo": name, "episode": start["episode"], "episode_sha256": sha256_file(demo["episode"]),
            "state": start["file"], "frames": [demo["from"], demo["to"]], "cmd_source": demo["cmd_source"],
            "success_both": demo["success_both"], "start": demo["start"], "resets": [], "skipped": [],
            "branches": 0, "why": None}
    col["demos"].append(drow)
    wps = list(range(args.wp_first, n - 1, args.wp_every))
    drow["waypoints"] = wps
    peel_tail = round(args.peel_tail_s / dt)
    clr = ur = cur = None
    fresh = False  # walking from a reset: a failed walk then ends the demo (no retry loop)

    def sample():
        return harness._sample(sim, ctx, harness._sampler(sim, ctx), 0, 0, publish=False,
                               timing={"render": 0.0, "encode": 0.0})

    for k in wps:
        if cur is None:
            sim.restore(sim_snapshot.load(demo["state"]), settle_steps=0)
            held = sim.settle(round(args.settle_s / sim.frame_dt)).held_for("franka")
            col["robot_s"] += args.settle_s
            drow["resets"].append({"before_waypoint": k, "why": drow.pop("_why", "demo start")})
            if not held:
                drow["why"] = "grasp lost after the restore"
                return
            ur = s2.Fixed(pose7_mat(UrTracking.fk(sim.arm_targets()[1])))
            clr = clr or ms.Clearance(sim, ctx.plate_top_z)
            cur, fresh = 0, True
        if k > cur:  # walk the demo cur -> k
            rec = ms.FileRec(sim, ctx, lambda tau: 0, lambda tau, c0=cur: c0 + round(tau / dt))
            watch = Watch(rec, peel_tail)
            rep = drive(sim, ctx, Shifted(demo["path"], cur * dt), ur, harness.SAMPLE_STEPS * (k - cur), watch)
            col["robot_s"] += rep["steps"] * sim.frame_dt
            why = outcome(rep, watch)
            if not fresh or len(drow["resets"]) == 1:  # a re-walk from a reset repeats the demo prefix
                miles = {"demo": name, "stage": 2, "segment": "walk", "from_frame": cur, "to_frame": k,
                         "start": start, "abort": rep["abort"], "outcome": why}
                row = ms.write_file(ctx, args.out / "miles" / f"{name}_walk_{cur:03d}_{k:03d}.npz", rec.frames, rep,
                                    miles)
                if row is not None:
                    col["files"].append({**row, "miles": miles})
            rmse = ms.gt_rmse_mm(rec.frames[-1]["belt"], demo["belt"][k])
            logger.info(f"[MILES2] {name} walk {cur} -> {k}: {why}, arrival belt RMSE vs demo {rmse:.1f} mm")
            if why in ("peel", "grasp_lost", "abort") or rmse > args.walk_max_rmse_mm:
                if fresh:
                    drow["why"] = f"walk {cur} -> {k} from a reset: {why}, RMSE {rmse:.1f} mm"
                    logger.warning(f"[MILES2] {name}: {drow['why']}; demo ends")
                    return
                drow["_why"], cur = f"walk to {k}: {why}, RMSE {rmse:.1f} mm", None
                continue
            cur = k
        fresh = False
        W_f = demo["path"].pose7(k * dt)
        arrival = sample()
        for c in range(args.branches):
            box = args.boxes[c % len(args.boxes)]
            rng = np.random.default_rng([args.seed, di, k, c])
            off = ms.draw_offset(rng, "franka", box, args.emphasis_frac)
            sc = clr.scale("franka", W_f, off["franka"])
            applied = off["franka"] * sc[0]
            path = ms.CyclePath(W_f, ms.offset_pose(W_f, applied), args.dev_s, args.ret_s, args.tail_s)
            rec = ms.FileRec(sim, ctx, path.segment, lambda tau, k=k: k)
            watch = Watch(rec, peel_tail)
            rep = drive(sim, ctx, path, ur, round(path.end / sim.frame_dt), watch)
            col["robot_s"] += rep["steps"] * sim.frame_dt
            why = outcome(rep, watch)
            ch = ms.chamfer_mm(rec.frames[-1]["pcd_belt_seg"], arrival["pcd_belt_seg"])[0]
            if why == "engaged" and ch > args.tau_mm:
                why = "disturbed"
            br = {"demo": name, "stage": 2, "segment": "cycle", "waypoint": k, "cycle": c, "mode": "franka",
                  "box": list(box), "offset_drawn": off["franka"].tolist(), "offset": applied.tolist(),
                  "single_axis": off["single_axis"], "scale": sc[0], "clearance_model_mm": [sc[1], sc[2]],
                  "abort": rep["abort"], "abort_kind": rep["abort_kind"], "stopped": rep.get("stopped"),
                  "outcome": why, "large_off": watch.large_off(), "min_large_wrap_deg": float(min(watch.wrap)),
                  "chamfer_vs_arrival_mm": ch, "end": end_labels(sim), "frames": len(rec.frames), "start": start,
                  "valid": why == "engaged", "why": None if why == "engaged" else why, "continuous": True,
                  "dev_s": args.dev_s, "ret_s": args.ret_s, "tail_s": args.tail_s}
            row = ms.write_file(ctx, args.out / "miles" / f"{name}_wp{k:03d}_c{c}.npz", rec.frames, rep, br)
            br["file"] = None if row is None else row["file"]
            if row is not None:
                col["files"].append({**row, "miles": br})
            col["branches"].append(br)
            drow["branches"] += 1
            logger.info(f"[MILES2] {name} wp {k} c{c} box {box} |dF| {np.linalg.norm(applied[:3]):4.1f} mm "
                        f"{np.linalg.norm(applied[3:]):4.1f} deg (scale {sc[0]:.2f}): {why}, chamfer vs arrival "
                        f"{ch:.2f} mm, min wrap {br['min_large_wrap_deg']:.0f}")
            if why in ("peel", "grasp_lost", "abort"):
                drow["_why"], cur = f"wp {k} c{c}: {why}", None
                break
            if why != "engaged":  # near_peel / disturbed: the walk goes on from here
                drow["skipped"].append({"waypoint": k, "after_cycle": c, "why": why})
                break
    logger.info(f"[MILES2] {name}: {drow['branches']} cycles, {len(drow['resets'])} resets, "
                f"{len(drow['skipped'])} skipped waypoints")


def buffer_rows(args, col: dict) -> list[dict]:
    out = []
    for i, f in enumerate(col["files"]):
        m = f["miles"]
        out.append({
            "id": f"miles2_{args.out.name}_{i:04d}", "iter": -1, "kind": "miles", "status": "ok",
            "weight": 1.0, "episode": str((args.out / "miles" / f["file"]).resolve()),
            "label_human": False,
            "label_human_detail": {"success": False, "rule": "n/a",
                                   "source": "stage-2 MILES segment, not a task attempt"},
            "label_strict": False, "start": {**m["start"], "from_reset": False, "jitter": None},
            "n_frames": f["n_frames"], "n_mpc_frames": 0, "outcome_gt": f["outcome_gt"],
            "segment": m["segment"], "segments": f["segments"],
            "miles": {k: m.get(k) for k in ("demo", "stage", "segment", "waypoint", "cycle", "offset",
                                            "box", "outcome", "large_off", "min_large_wrap_deg", "valid",
                                            "why")}})
    return out


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], fromfile_prefix_chars="@")
    p.add_argument("--out", type=Path, required=True, help="new dir")
    p.add_argument("--lcm-url", required=True)
    p.add_argument("--demo", action="append", required=True, metavar="NAME=EPISODE,STATE",
                   help="a whole-run chain episode + its stage-2 start state (repeat; @file)")
    p.add_argument("--settle-s", type=float, default=0.5, help="hold after the stage-2 start restore")
    p.add_argument("--wp-first", type=int, default=0, help="first waypoint (stage-2 frame)")
    p.add_argument("--wp-every", type=int, default=5, help="waypoint spacing (frames)")
    p.add_argument("--branches", type=int, default=6, help="branches per waypoint")
    p.add_argument("--boxes", type=lambda t: [ms._pair(b) for b in t.split(";")],
                   default=[(10.0, 4.0), (25.0, 8.0)], metavar="MM,DEG;MM,DEG",
                   help="offset boxes, cycled over a waypoint's branches")
    p.add_argument("--emphasis-frac", type=float, default=0.0)
    p.add_argument("--dev-s", type=float, default=0.6)
    p.add_argument("--ret-s", type=float, default=0.6)
    p.add_argument("--tail-s", type=float, default=1.5, help="plain demo path after the return")
    p.add_argument("--peel-tail-s", type=float, default=1.0, help="stop this long after large_off")
    p.add_argument("--franka-width-max", type=float, default=20.0,
                   help="held() Franka hand-width limit mm (the chain suite's stage-2 value)")
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--cloud-noise", default="off", metavar="PRESET|JSON")
    p.add_argument("--osc-timeout-s", type=float, default=5.0)
    p.add_argument("--franka-grip-n", type=float, default=None, help="Franka squeeze force N (default the yaml's)")
    p.add_argument("--walk-max-rmse-mm", type=float, default=None,
                   help="replay the walk again while its belt RMSE vs the demo exceeds this anywhere; "
                        "a demo never under it gets no branches (default: no check)")
    p.add_argument("--walk-tries", type=int, default=1, help="--walk-max-rmse-mm: replays per demo")
    p.add_argument("--walk-only", action="store_true",
                   help="with --walk-max-rmse-mm: replay the walk --walk-tries times, log WALK_PASSES, no branches")
    p.add_argument("--only", action="append", default=None, metavar="NAME:WP:C",
                   help="run only these branches (repeat); other demos are skipped (their index keeps the seeds)")
    p.add_argument("--no-grasp-abort", action="store_true",
                   help="branches keep running after the Franka loses the belt (diagnostic: view the loss)")
    p.add_argument("--continuous", action="store_true",
                   help="stage-1 style cycles on one walk (no snapshot branching; --tail-s = the hold after the "
                        "return); needs --walk-max-rmse-mm and --tau-mm")
    p.add_argument("--tau-mm", type=float, default=None,
                   help="--continuous: segmented-cloud Chamfer vs the arrival cloud above which a cycle is disturbed")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = create_parser().parse_args()
    if args.walk_only and args.walk_max_rmse_mm is None:
        raise SystemExit("--walk-only needs --walk-max-rmse-mm")
    if args.continuous and (args.walk_max_rmse_mm is None or args.tau_mm is None or args.walk_only or args.only
                            or args.no_grasp_abort):
        raise SystemExit("--continuous needs --walk-max-rmse-mm and --tau-mm, without --walk-only/--only/--no-grasp-abort")
    cli.configure_logging()
    harness.check_private_url(args.lcm_url)
    demos = [load_demo(s) for s in args.demo]
    if len({d["name"] for d in demos}) != len(demos):
        raise SystemExit("--demo: duplicate names")
    dur = args.dev_s + args.ret_s + args.tail_s
    for d in demos:
        nw = len(range(args.wp_first, len(d["knot0"]) - 1, args.wp_every))
        print(f"{d['name']}: {d['episode'].relative_to(REPO_ROOT)} frames {d['from']}..{d['to']} "
              f"(cmd {d['cmd_source']}, both {d['success_both']}), {nw} waypoints x {args.branches} = "
              f"{nw * args.branches} branches, <= {nw * args.branches * dur:.0f} s robot")
    if args.dry_run:
        return 0
    out = args.out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} exists and is not empty; pick a new dir")
    (out / "miles").mkdir(parents=True, exist_ok=True)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_info()}
    import round_belt_task.offline_simulation as osim
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    osim.FRANKA_WIDTH_MM = (osim.FRANKA_WIDTH_MM[0], float(args.franka_width_max))

    nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=cli.FIRST, last=cli.LAST)
    osc_binary, osc_root = harness.default_osc_binary()
    grip_kw = {} if args.franka_grip_n is None else {"franka_grip_n": args.franka_grip_n}
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url, osc_timeout_s=args.osc_timeout_s, **grip_kw)
    col = {"demos": [], "branches": [], "files": [], "robot_s": 0.0}
    index = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
             **meta, "phase_labels": ms.PHASE_LABELS, "cmd_source": ms.CMD_MILES,
             "rules": {"large_off": f"wrap < {OFF_WRAP_DEG} deg for {OFF_MARGIN + 1} frames (v12_109)",
                       "near_peel": f"min large wrap < {NEAR_WRAP_DEG} deg, not off",
                       "hold segment": "the plain demo path after the return (tail)"}}
    t_run = time.perf_counter()
    try:
        warm = sim.start_osc(out / "osc.log", binary=osc_binary, cwd=osc_root)
        index["osc"] = {**sim.osc.describe(), "warm_up_s": warm}
        sim.restore(sim_snapshot.load(demos[0]["state"]))
        board, gripper = sim.clearance_geometry(jaw_bytes=(UR_BYTE,))
        plate = board.plate
        cloud = cli.cloud_options(argparse.Namespace(cloud_noise=args.cloud_noise, belt_seg=True,
                                                     no_pcd=False))
        ctx = argparse.Namespace(
            args=argparse.Namespace(no_pcd=False, belt_seg_source="camera"), mode="miles",
            encoder=None, sim=sim, board=board, gripper=gripper, thresholds=DEFAULT_THRESHOLDS,
            tangent=pert.belt_tangent(nominal), cloud_rng=None,
            cloud={**cloud, "origins": cli.sim_camera_origins(sim)},
            plate_top_z=float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half))
        for di, d in enumerate(demos):
            (run_demo_continuous if args.continuous else run_demo)(args, sim, ctx, di, d, col)
            if args.continuous:  # a crash keeps the finished demos' rows
                write_json(out / "buffer.json", {"version": 2, "iter": -1, "n": len(col["files"]), "partial": True,
                                                 "source": "stage-2 MILES (miles_stage2.py)",
                                                 "rollouts": buffer_rows(args, col)})
            write_json(out / "index.json", json.loads(json.dumps(
                {**index, "demos": col["demos"], "branches": col["branches"]}, default=lcs.json_default)))
    finally:
        sim.close("finished")
    wall = time.perf_counter() - t_run
    rows = buffer_rows(args, col)
    write_json(out / "buffer.json", {"version": 2, "iter": -1, "n": len(rows),
                                     "source": "stage-2 MILES (miles_stage2.py)", "rollouts": rows})
    by = {d["demo"]: dict(Counter(b["outcome"] for b in col["branches"] if b["demo"] == d["demo"]))
          for d in col["demos"]}
    bud = {"branches": len(col["branches"]), "files": len(col["files"]),
           "frames": sum(f["n_frames"] for f in col["files"]), "robot_s": col["robot_s"],
           "wall_s": wall, "outcomes": by}
    write_json(out / "index.json", json.loads(json.dumps(
        {**index, "demos": col["demos"], "branches": col["branches"], "budget": bud},
        default=lcs.json_default)))
    logger.info(f"[MILES2] {bud['branches']} branches, {bud['frames']} frames, robot "
                f"{bud['robot_s']:.0f} s, wall {wall / 60:.1f} min, outcomes {by} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
