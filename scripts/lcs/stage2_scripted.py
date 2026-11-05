"""Scripted stage-2 walk of the whole-run chain (``eval_learned_mpc.py --stage2-scripted``).

After the scripted UR exit, on the live state (no restore): ``stage2_collect``'s D10L walk without
loops (transit, hover, descent, wrap, arc, end), parameters nominal or drawn from its
``PARAM_RANGES``, DART knot noise at ``--scripted-dart`` x its ``DART`` sigmas. Frames continue
the episode grid with ``sim_cmd_source`` 5 (the collector's) and phase ``s2:<phase>``; a drive
abort (grasp lost, tip floor, board, UR IK) ends the walk and skips the release.
"""

from __future__ import annotations

import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_learned_mpc as harness
import miles_seed as ms
import stage2_collect as s2
from loguru import logger
from scipy.spatial.transform import Rotation

from round_belt_task.arm_kinematics import UrTracking
from round_belt_task.episode_io import pose7_mat
from round_belt_task.outcome import SMALL_SEAT_MM, frame_metrics
from task_common import sim_snapshot
from task_common.replay_metrics import FRANKA_BASE_BODY
from utils.labels import body_index

PHASES = ("transit", "hover", "descent", "wrap", "arc", "end")
UR_MODES = {"hold": "UR keeps its grasp, stationary at its stage-1 end target",
            "feed": "UR keeps its grasp; min-jerk FEED_MM toward the small-pulley axis (horizontal) "
                    "over the descent, held through wrap / arc, back over the end hold"}
FEED_MM = 15.0
WP_KEYS = ("dx", "dy", "dz", "rx", "ry")  # per-waypoint offset: world mm, hand-local deg
SEARCH_BOX = (15.0, 15.0, 15.0, 10.0, 10.0)
# --scripted-set keys beyond stage2_collect.NOMINAL (V12-147)
EXTRA_SET = ("w0_dh_mm", "walk_daz_deg", "walk_rx_deg", "walk_ry_deg", "pull_base_mm", "search_n",
             "search_seed", "search_span_w", *(f"w{i}_{k}" for i in range(11) for k in WP_KEYS))
FEED_OUT, FEED_BACK = ("descent",), ("end",)


class Ramp:
    """Min-jerk position ramp ``a`` -> ``b`` (pose7, ``a``'s orientation) over ``t`` s."""

    def __init__(self, a, b, t: float) -> None:
        self.a, self.b, self.end = np.asarray(a, np.float64), np.asarray(b, np.float64), float(t)

    def pose7(self, tau: float) -> np.ndarray:
        s = min(max(tau / self.end, 0.0), 1.0) if self.end > 0 else 1.0
        s = s**3 * (10.0 - 15.0 * s + 6.0 * s * s)
        return np.r_[self.a[:3] + s * (self.b[:3] - self.a[:3]), self.a[3:]]

    def __call__(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        p = self.pose7(tau)
        return p[:3], p[3:]


def feed_target(sim, u0) -> np.ndarray:
    """``u0`` moved FEED_MM horizontally toward the small-pulley axis."""
    q = sim.state_0.body_q.numpy()
    small, large = (np.asarray(q[int(sim.info.pulley_bodies[i])][:3], np.float64) for i in (0, 1))
    d = small[:2] - large[:2]
    return np.r_[u0[:2] + FEED_MM * 1e-3 * d / np.linalg.norm(d), u0[2:]]


def legs(poses: list) -> list[tuple[str, list, float]]:
    """``Collector._run``'s walk calls without loops: ``(phase, targets, hold s)``."""
    return [("transit", poses[0:4], 0.0), ("hover", [], s2.HOVER_HOLD_S),
            ("descent", poses[4:5], 0.0), ("wrap", poses[5:7], 0.0), ("wrap", poses[7:8], 0.0),
            ("arc", poses[8:11], 0.0), ("end", [], s2.END_HOLD_S)]


def offset(p, o) -> np.ndarray:
    """pose7 ``p`` moved by WP_KEYS offsets ``o``."""
    q = Rotation.from_quat(np.r_[p[4:], p[3]]) * Rotation.from_euler("xy", o[3:5], degrees=True)
    return np.r_[p[:3] + np.asarray(o[:3]) * 1e-3, np.roll(q.as_quat(), 1)]


def leg_cost(sim, phase: str, rep: dict, span_w: float = 1.0) -> dict:
    """Search score at a leg's end (lower better): span_w x large-groove height span (mm), 0.5 per deg of
    large wrap short of 140, of small wrap short of 150 from the wrap leg on; abort / lost grasp 1e3."""
    q = sim.state_0.body_q.numpy().astype(np.float64)
    belt, (ks, kl) = q[sim.info.belt_bodies, :3], (int(b) for b in sim.info.pulley_bodies[:2])
    fl, fs = frame_metrics(belt, q[kl]), frame_metrics(belt, q[ks], SMALL_SEAT_MM)
    span = fl.h_max_mm - fl.h_min_mm if fl.n_neighbour else 50.0
    held = bool(sim.grasp_state().held_for("franka"))
    c = (1e3 * (bool(rep["abort"]) or not held) + span_w * span + 0.5 * max(0.0, 140.0 - fl.wrap_deg)
         + (0.5 * max(0.0, 150.0 - fs.wrap_deg) if phase in ("wrap", "arc") else 0.0))
    return {"cost": float(c), "span_mm": float(span), "wrap_large": float(fl.wrap_deg),
            "wrap_small": float(fs.wrap_deg), "held": held, "abort": rep["abort"]}


def run(sim, ctx, sampler, last: dict, timing: dict, args, name: str,
        ur_mode: str | None = None) -> dict:
    """The walk from the current state (on the episode grid of ``last``):
    ``{frames, report, done, tip_min_m}``; ``done`` False (an abort) -> no release.
    ``ur_mode`` hold / feed (UR_MODES): the UR still grasps the belt; None = parked."""
    n, dt, w0 = harness.SAMPLE_STEPS, sim.frame_dt, time.perf_counter()
    s0 = last["sim_step"] - last["step"]
    key = [int(args.scripted_seed), zlib.crc32(name.encode())]
    rng_p, rng_n = (np.random.default_rng([*key, i]) for i in range(2))
    prm = dict(s2.NOMINAL) if args.scripted_params == "nominal" else s2.draw_params(rng_p)
    prm.update({k: float(v) for k, _, v in
                (kv.partition("=") for kv in getattr(args, "scripted_set", None) or [])})
    ee0 = sim.franka_measured_pose7()
    plan = s2.build_plan({k: v for k, v in prm.items() if k not in EXTRA_SET}, ee0, s2.hw_poses())
    if "w0_dh_mm" in prm:  # first transit leg at the start height: no lift next to the large pulley
        lb, ph, p = plan[0]
        r, az = s2.TRANSIT_RAZ[0]
        plan[0] = (lb, ph, np.r_[s2.small_to_world(r, az, s2.small_polar(ee0[:3])[2]
                                                   + prm["w0_dh_mm"]), p[3:]])
    if "walk_daz_deg" in prm:  # whole walk turned about the small-pulley axis: grasp slid on the belt
        daz = prm["walk_daz_deg"]
        rot = Rotation.from_rotvec(math.radians(daz)
                                   * s2.quat_to_mat3(s2.bc.known_small_pulley_pose7()[3:])[:, 2])

        def turn(p):
            r, az, h = s2.small_polar(p[:3])
            q = (rot * Rotation.from_quat(np.r_[p[4:], p[3]])).as_quat()  # wxyz <-> xyzw
            return np.r_[s2.small_to_world(r, az + daz, h), q[3], q[:3]]

        plan = [(lb, ph, turn(p)) for lb, ph, p in plan]
    rl = Rotation.from_euler("xy", [prm.get("walk_rx_deg", 0.0), prm.get("walk_ry_deg", 0.0)],
                             degrees=True)
    if rl.magnitude() > 0:  # hand roll on every walk pose (hand-local x then y): grasp roll on the belt
        plan = [(lb, ph, np.r_[p[:3], np.roll((Rotation.from_quat(np.r_[p[4:], p[3]]) * rl).as_quat(), 1)])
                for lb, ph, p in plan]
    plan = [(lb, ph, offset(p, [prm.get(f"{lb}_{k}", 0.0) for k in WP_KEYS])) for lb, ph, p in plan]
    n_search = int(prm.get("search_n", 0))
    if n_search and any(k.startswith("w") and k[1].isdigit() and k.split("_")[1] in WP_KEYS
                        for k in prm):
        raise ValueError("search_n: starts from no per-waypoint offsets")
    if n_search and ur_mode:
        raise ValueError("search_n: parked UR only")
    rng_s, search = np.random.default_rng(int(prm.get("search_seed", 0))), []
    ur = s2.Fixed(pose7_mat(UrTracking.fk(sim.arm_targets()[1])))  # parked UR: u = 0
    u0 = ur.pose7(0.0)
    u_feed = feed_target(sim, u0) if ur_mode == "feed" else u0
    det = timing.setdefault("scripted_detail", {"render": 0.0, "encode": 0.0})
    frames: list[dict] = []
    done, cur, tip_min, rows = [], ee0, math.inf, []
    abort = None
    pre = []
    if prm.get("pull_base_mm"):  # user (V12-147): first step toward the Franka base, belt kept on the large
        base = np.asarray(sim.state_0.body_q.numpy()[body_index(list(sim.model.body_label),
                                                                FRANKA_BASE_BODY)][:3], np.float64)
        d = base - ee0[:3]
        d[2] = 0.0
        pull = np.r_[ee0[:3] + prm["pull_base_mm"] * 1e-3 * d / np.linalg.norm(d), ee0[3:]]
        plan.insert(0, ("pull", "transit", pull))
        pre = [("transit", [pull], 0.0)]
    labs = [(ph, [None] * len(t), h) for ph, t, h in pre] + legs([lb for lb, *_ in plan[len(pre):]])
    for (phase, targets, hold), (_, tl, _) in zip(pre + legs([p for *_, p in plan[len(pre):]]), labs,
                                                   strict=True):
        if n_search and targets and tl[0] is not None:  # V12-147 sampling demonstrator: best of n
            snap, tried = sim_snapshot.capture(sim, f"search:{phase}"), []
            rec, sim.recorder = sim.recorder, None  # rollouts stay out of --record

            def roll(tj, snap=snap, cur=cur, hold=hold, phase=phase, ur=ur):
                sim.restore(snap, settle_steps=0)
                # the arc is scored after the end hold: the state the release starts from
                path = s2.walk_path(cur, tj, prm["speed_mps"], hold + s2.END_HOLD_S * (phase == "arc"))
                rep = ms.drive(sim, ctx, s2.Cmd(path, None), ur,
                               n * math.ceil(round(path.end / dt) / n), None, grasp_arms="franka")
                return leg_cost(sim, phase, rep, prm.get("search_span_w", 1.0))

            for j in range(n_search):  # sample 0 = the plan as is
                o = np.zeros((len(targets), 5)) if j == 0 else rng_s.uniform(
                    -1.0, 1.0, (len(targets), 5)) * SEARCH_BOX
                tj = [offset(p, oi) for p, oi in zip(targets, o, strict=True)]
                tried.append({**roll(tj), "o": o, "targets": tj})
                logger.info(f"[S2] search {phase} {j}: " + " ".join(
                    f"{k} {v:.1f}" for k, v in tried[-1].items() if isinstance(v, float)))
            # the OSC loop is not bit-repeatable: top 3 re-run twice, best worst case wins
            for r in sorted(tried, key=lambda r: r["cost"])[:3]:
                r["costs_rerun"] = [roll(r["targets"])["cost"] for _ in range(2)]
                r["cost_worst"] = max(r["cost"], *r["costs_rerun"])
                logger.info(f"[S2] search {phase} rerun {r['cost']:.1f} -> {r['costs_rerun']}")
            best = min((r for r in tried if "cost_worst" in r), key=lambda r: r["cost_worst"])
            sim.restore(snap, settle_steps=0)
            sim.bridge.take_targets()
            sim.recorder = rec
            targets = best["targets"]
            for lb, oi in zip(tl, best["o"], strict=True):
                prm.update({f"{lb}_{k}": round(float(v), 2) for k, v in zip(WP_KEYS, oi, strict=True)})
            search.append({"phase": phase, "labels": list(tl), "best_cost": best["cost_worst"],
                           "costs": [r["cost"] for r in tried]})
        path = s2.walk_path(cur, targets, prm["speed_mps"], hold)
        sig = s2.DART.get(phase)
        noise = (s2.KnotNoise(rng_n, path.end, *(args.scripted_dart * np.asarray(sig)))
                 if args.scripted_dart > 0 and sig else None)
        cmd = s2.Cmd(path, noise)
        k_phase = ctx.phase_labels.index(f"s2:{phase}")
        if ur_mode == "feed":
            ur = (Ramp(u0, u_feed, path.end) if phase in FEED_OUT
                  else Ramp(u_feed, u0, path.end) if phase in FEED_BACK
                  else s2.Fixed(u_feed if phase in ("wrap", "arc") else u0))

        def on_frame(k: int, c: dict, tau: float, phase=phase, k_phase=k_phase) -> None:
            fr = harness._sample(sim, ctx, sampler, sim.step_index - s0, s0, publish=False,
                                 timing=det)
            fr.update(c, n_knots=harness.N_KNOTS, cmd_source=s2.CMD_STAGE2,
                      ur_line=np.full(16, np.nan), ur_fresh_exact=False,
                      ur_own_delta=np.full(6, np.nan), post_phase=phase, phase=k_phase, stage=0,
                      ctrl_hand_mm=math.nan, ctrl_robotiq=-1,
                      latent_pred=np.full(len(fr["latent"]), np.nan),
                      goal_dist=np.full(1, np.nan))
            frames.append(fr)

        st = sim.step_index
        rep = ms.drive(sim, ctx, cmd, ur, n * math.ceil(round(path.end / dt) / n), on_frame,
                       grasp_arms="franka")
        while (sim.step_index - s0) % n:  # an abort mid-period: keep the sample grid
            sim.control_step()
        tip_min = min(tip_min, rep["tip_min_mm"] * 1e-3)
        if search and search[-1]["phase"] == phase and "replay_cost" not in search[-1]:
            search[-1]["replay_cost"] = leg_cost(sim, phase, rep)["cost"]  # restore determinism
        rows.append({"phase": phase, "duration_s": (sim.step_index - st) * dt,
                     "abort": rep["abort"], "abort_kind": rep["abort_kind"],
                     "tip_min_mm": rep["tip_min_mm"], "min_clearance_mm": rep["min_clearance_mm"]})
        cur = np.asarray(targets[-1], np.float64) if targets else cur
        if rep["abort"]:
            abort = f"{phase}: {rep['abort_kind']}: {rep['abort']}"
            break
        done.append(phase)
    lab = harness._label_row(ctx, frames[-1]) if frames else {}
    if n_search:
        chosen = [f"{k}={v:g}" for k, v in prm.items() if k[0] == "w" and k[1].isdigit()
                  and k.split("_")[1] in WP_KEYS]
        logger.info(f"[S2] search legs {search}")
        logger.info("[S2] SEARCH_CHOSEN " + " ".join(chosen))
    report = {"status": "ok" if abort is None else "aborted", "reason": abort,
              "duration_s": (sim.step_index - (last["sim_step"] if not frames
                                                else frames[0]["sim_step"])) * dt,
              "label_end": lab.get("label"),
              "scripted": {"params": prm, "params_mode": args.scripted_params,
                           "dart_scale": float(args.scripted_dart),
                           "seed": key, "legs": rows, **({"search": search} if search else {}), "phases_done": done,
                           "plan": [{"label": lb, "phase": ph, "pose": p.tolist()}
                                    for lb, ph, p in plan], "ee0": ee0.tolist(),
                           "frames": len(frames), "large_label_end": lab.get("large"),
                           **({"ur_mode": ur_mode, "ur_feed_mm": FEED_MM if ur_mode == "feed"
                               else 0.0, "held_ur": bool(sim.grasp_state().held()[1])}
                              if ur_mode else {}),
                           "small_label_end": lab.get("small"), "held_franka": lab.get("held")}}
    timing["scripted"] = time.perf_counter() - w0
    logger.info(f"[S2] {name} scripted ({args.scripted_params}, dart x{args.scripted_dart:g}): "
                f"{'done' if abort is None else abort}, {len(frames)} frames, large "
                f"{lab.get('large')} small {lab.get('small')}")
    return {"frames": frames, "report": report, "done": abort is None, "tip_min_m": tip_min}
