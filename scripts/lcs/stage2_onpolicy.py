#!/usr/bin/env python3
"""Stage-2 on-policy collection (V12-99): the learned MPC with MILES-style reversible excursions.

Runs ``eval_learned_mpc.py`` (stage-2 suite flags: ``--controller mppi --post-mpc stage2_release
--grasp-arms franka --outcome-pulley both --belt-seg``) with three hooks installed in-process (the
harness file is not changed):

- Excursions around MPC states: every ``--ex-every-s`` of MPC time the node is paused (SIGSTOP of
  our own child, after it answered the last latent), the Franka runs a min-jerk out / back / hold
  cycle (``miles_seed.CyclePath``) from the measured pose, then a fresh latent goes out and the node
  resumes (SIGCONT). Boxes per region of the hand over the small pulley (``REGIONS``), scaled by
  the clearance model, x a per-region multiplier halved on every trip (``budget.json``).
- Online trips (real-available signals only: Franka hand width, belt-cloud features at the large
  pulley, belt-cloud point count at the small pulley). In an excursion vs its start
  (``EX_TRIP``): return home; not recovered -> safe hold. On every MPC frame (``MPC_TRIP``, opt-out
  ``--no-mpc-trip``): safe hold. Safe hold = node stopped, Franka held, then the scripted release.
- DART: noise on the executed MPC command = ``action_noise_frac`` in the MPPI yaml (node feature).

Excursion frames carry ``sim_cmd_source`` 7 (``CMD_EXCURSION``) and their own command; per-frame
``sim_onpolicy_segment`` (mpc / ex:dev / ex:return / ex:hold / post), ``sim_onpolicy_excursion``,
``sim_onpolicy_trip``, ``sim_cloud_small_n_near``. Writes the harness outputs plus
``buffer.json`` (kind ``stage2_onpolicy``, ``post_from``, segments), ``budget.json``, ``report.md``.
Sim GT (both pulleys) only for labels / reset counts.

Run (stage-2 scene, under the sim lock), e.g.:
    ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_stage2.yaml flock \\
      data/lcs/realfirst/cmp/v12_79/sim.lock uv run --frozen python scripts/lcs/stage2_onpolicy.py \\
      --ex-every-s 3 -- <eval_learned_mpc.py stage-2 flags> --out <new dir>
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import signal
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_lcs_dataset as cli
import eval_learned_mpc as harness
import miles_seed as ms
import stage2_collect as s2
from loguru import logger

from round_belt_task.arm_kinematics import UrTracking
from round_belt_task.episode_io import pose7_mat, write_json
from task_common import belt_cloud as bc
from task_common import lcs_dataset as lcs

CMD_EXCURSION = 7  # harness 0-3, MILES 4, stage-2 scripted 5, post-MPC release 6
SEGMENTS = ("mpc", "ex:dev", "ex:return", "ex:hold", "post")
KIND = "stage2_onpolicy"
# Hand over the small pulley: near = the wrap zone (r, h of the measured finger_tip, mm).
NEAR_R_MM, NEAR_H_MM = 45.0, 12.0
# Boxes at multiplier 1 (stage2_collect.draw_loop keys; away = cap along "away from the large axis").
REGIONS = {"far": {"mm": 8.0, "z_mm": 6.0, "yaw_deg": 4.0, "tilt_deg": 3.0, "away_mm": 4.0},
           "near": {"mm": 4.0, "z_mm": 3.0, "yaw_deg": 3.0, "tilt_deg": 2.0, "away_mm": 2.0}}
MIN_MULT = 0.125
ANSWER_WAIT_S = 0.4  # wall: wait for the node's plan (+ one 50 ms republish) before pausing


@dataclasses.dataclass(frozen=True)
class ExTrip:
    """Excursion trip vs the excursion's first frame (stage2_collect.TripSpec semantics)."""
    hand_lo_mm: float = -2.0
    hand_hi_mm: float = 3.0
    large_h_mm: float = 3.0
    large_wrap_drop_deg: float = 40.0  # the cloud wrap jitters +-30 deg frame to frame
    small_drop_frac: float = 0.5
    small_min_pts: int = 10
    small_phases: tuple[str, ...] = ("near",)


@dataclasses.dataclass(frozen=True)
class MpcTrip:
    """Belt-loss onset on one frame (and the previous): V12-95/96 MPC episodes, 88/98 GT grasp /
    large-off events caught at or before the event, 6/22 event-free episodes tripped."""
    hand_lo_mm: float = 4.0
    hand_hi_mm: float = 17.5
    hand_rate_mm: float = 2.5  # one 75 ms frame
    large_h_mm: float = 3.5
    large_wrap_min_deg: float = 60.0
    large_none_frames: int = 2


EX_TRIP, MPC_TRIP = ExTrip(), MpcTrip()


# --- pure helpers (check_stage2_onpolicy) -----------------------------------------------------

def region_of(ee7) -> str:
    r, _, h = s2.small_polar(np.asarray(ee7)[:3])
    return "near" if r < NEAR_R_MM and h < NEAR_H_MM else "far"


def away_dir(ee7) -> np.ndarray:
    """Horizontal unit vector from the large pulley axis to the hand."""
    v = np.asarray(ee7, np.float64)[:3] - bc.known_large_pulley_pose7()[:3]
    v[2] = 0.0
    return v / max(np.linalg.norm(v), 1e-9)


def slow_enough(prev_ee, ee, max_mm: float) -> bool:
    """Hand moved <= ``max_mm`` since the last sample (measured): no excursion mid-transit."""
    return prev_ee is not None and 1e3 * float(np.linalg.norm(ee[:3] - prev_ee[:3])) <= max_mm


def ex_trip(ref: dict, cur: dict, region: str, spec: ExTrip = EX_TRIP) -> str | None:
    return s2.trip_signal(ref, cur, region, spec)


def mpc_trip(prev: dict | None, cur: dict, none_run: int, spec: MpcTrip = MPC_TRIP) -> str | None:
    """``none_run``: consecutive frames (incl. ``cur``) without a large-pulley cloud band."""
    h = cur["hand_mm"]
    if not spec.hand_lo_mm <= h <= spec.hand_hi_mm:
        return "hand_width"
    if prev is not None and abs(h - prev["hand_mm"]) > spec.hand_rate_mm:
        return "hand_rate"
    lh, lw = cur["large_h_mm"], cur["large_wrap_deg"]
    if math.isfinite(lh) and abs(lh) > spec.large_h_mm:
        return "large_h"
    if math.isfinite(lh) and lw < spec.large_wrap_min_deg:
        return "large_wrap"
    if none_run >= spec.large_none_frames:
        return "large_none"
    return None


class Budget:
    """Per-region box multipliers, halved on a trip (floor ``MIN_MULT``), saved on change."""

    def __init__(self, path: Path, seed_from: Path | None = None) -> None:
        self.path, self.seeded_from = path, None if seed_from is None else str(seed_from)
        self.mult = {r: 1.0 for r in REGIONS}
        self.history: list[dict] = []
        if seed_from is not None:
            self.mult.update(json.loads(Path(seed_from).read_text())["multipliers"])
        self.save()

    def box(self, region: str) -> dict:
        return {k: v * self.mult[region] for k, v in REGIONS[region].items()}

    def halve(self, region: str, signal_: str, episode: str) -> float:
        before = self.mult[region]
        self.mult[region] = max(MIN_MULT, 0.5 * before)
        self.history.append({"episode": episode, "region": region, "signal": signal_,
                             "before": before, "after": self.mult[region]})
        self.save()
        return self.mult[region]

    def save(self) -> None:
        write_json(self.path, {
            "multipliers": self.mult, "boxes_at_1": REGIONS, "min_mult": MIN_MULT,
            "rule": "box = boxes_at_1 x multiplier; an excursion trip halves its region's",
            "seeded_from": self.seeded_from, "history": self.history})


def merge_frames(loop_frames: list, pending: list[tuple[int, list, dict]]) -> list:
    """Insert each excursion's frames before loop frame ``idx``; sets ``rec["span"]``."""
    by: dict[int, list] = {}
    for idx, frames, rec in pending:
        by.setdefault(idx, []).append((frames, rec))
    out: list = []
    for i in range(len(loop_frames) + 1):
        for frames, rec in by.pop(i, []):
            rec["span"] = [len(out), len(out) + len(frames)]
            out += frames
        if i < len(loop_frames):
            out.append(loop_frames[i])
    for frames, rec in (x for v in by.values() for x in v):  # never: a sample follows each
        rec["span"] = [len(out), len(out) + len(frames)]
        out += frames
    return out


def segment_of(fr: dict) -> str:
    if "post_phase" in fr:
        return "post"
    return f"ex:{fr['onpolicy_seg']}" if "onpolicy_seg" in fr else "mpc"


def segments_of(seg: list[str], ex_id: list[int]) -> list[dict]:
    """Contiguous spans of mpc / excursion (one per id) / post frames."""
    out: list[dict] = []
    for i, (s, e) in enumerate(zip(seg, ex_id, strict=True)):
        kind = "post" if s == "post" else "excursion" if s.startswith("ex:") else "mpc"
        if out and out[-1]["kind"] == kind and out[-1].get("excursion", -1) == e:
            out[-1]["span"][1] = i + 1
        else:
            out.append({"kind": kind, "span": [i, i + 1], **({"excursion": e} if e >= 0 else {})})
    return out


def extra_reset(row: dict) -> bool:
    """Real-robot manual reset: belt off the large pulley after the release or out of the
    fingers at the MPC end (sim GT)."""
    pm = row["post_mpc"]
    return bool(pm["final"]["large"] != "engaged" or not pm["mpc_end"]["held"])


# --- episode state + hooks --------------------------------------------------------------------

@dataclasses.dataclass
class Ep:
    name: str
    rng: np.random.Generator
    ctrl: object = None
    clr: ms.Clearance | None = None
    in_loop: bool = False
    n_loop: int = 0
    next_t: float = 0.0
    paused: bool = False
    trip: dict | None = None
    prev_sig: dict | None = None
    prev_ee: np.ndarray | None = None
    none_run: int = 0
    pending: list = dataclasses.field(default_factory=list)
    excursions: list = dataclasses.field(default_factory=list)
    mult0: dict = dataclasses.field(default_factory=dict)
    tip_min_mm: float = math.inf

    def summary(self) -> dict:
        ex = [{k: v for k, v in r.items() if k not in ("ref", "trip_signals")}
              for r in self.excursions]
        return {"excursions": json.loads(json.dumps(ex, default=lcs.json_default)),
                "trip": json.loads(json.dumps(self.trip, default=lcs.json_default)),
                "mult_at_start": self.mult0, "n_excursions": len(ex),
                "n_ex_trips": sum(r["trip"] is not None for r in ex)}


class OnPolicy:
    def __init__(self) -> None:
        self.cfg: argparse.Namespace | None = None
        self.budget: Budget | None = None
        self.ep: Ep | None = None
        self.orig: dict = {}


OP = OnPolicy()
HOOKED = ("run_episode", "_run_loop", "_sample", "_resolve_commands", "build_writer")


def install(cfg: argparse.Namespace, budget: Budget) -> None:
    OP.cfg, OP.budget = cfg, budget
    OP.orig = {k: getattr(harness, k) for k in HOOKED}
    harness.run_episode, harness._run_loop = _run_episode, _run_loop
    harness._sample, harness._resolve_commands = _sample, _resolve_commands
    harness.build_writer = _build_writer


def uninstall() -> None:
    for k, f in OP.orig.items():
        setattr(harness, k, f)
    OP.orig = {}


def _run_episode(sim, ctx, state, repeat, from_reset=None):
    cfg = OP.cfg
    name = f"episode_{state.id}_{repeat}"
    rng = np.random.default_rng([cfg.ex_seed, zlib.crc32(state.id.encode()), repeat])
    OP.ep = Ep(name, rng, mult0=dict(OP.budget.mult))
    try:
        row = OP.orig["run_episode"](sim, ctx, state, repeat, from_reset)
    finally:
        ep, OP.ep = OP.ep, None
        if ep is not None and ep.paused:
            _resume(ep)
    if row.get("file"):
        row["onpolicy"] = ep.summary()
    return row


def _run_loop(sim, ctx, ctrl, sampler, events_raw):
    ep, cfg = OP.ep, OP.cfg
    ep.ctrl, ep.in_loop, ep.next_t = ctrl, True, cfg.ex_first_s
    if cfg.excursions:
        ep.clr = ms.Clearance(sim, ctx.plate_top_z)
    try:
        loop = OP.orig["_run_loop"](sim, ctx, ctrl, sampler, events_raw)
    finally:
        ep.in_loop = False
        if ep.paused:
            _resume(ep)
    loop["frames"] = merge_frames(loop["frames"], ep.pending)
    loop["tip_min_m"] = min(loop["tip_min_m"], ep.tip_min_mm * 1e-3)
    if ep.trip is not None:
        loop["status"], loop["reason"] = "aborted", f"trip:{ep.trip['why']} at {ep.trip['t']:.2f} s"
    return loop


def _resolve_commands(frames, bridge, dt_step):
    OP.orig["_resolve_commands"]([fr for fr in frames if "onpolicy_seg" not in fr], bridge, dt_step)


def _answers(bridge, utime: int) -> int:
    """Plan messages (first + republishes) on the bridge answering latents >= ``utime``."""
    n = 0
    for tr in reversed(bridge.franka_trajs):
        if tr.utime < utime:
            break
        n += 1
    return n


def _pause(ep: Ep, bridge) -> bool:
    """SIGSTOP right after the node republished its plan for the last sent latent (it then
    waits for LCM, so after SIGCONT it reads the fresh latent first); wall-bounded. False = no
    answer (finished or stuck): not paused."""
    want, t0 = bridge.sent_latents[-1][0], time.perf_counter()
    while _answers(bridge, want) < 2 and time.perf_counter() - t0 < ANSWER_WAIT_S:
        bridge.poll()
        time.sleep(0.001)
    if not _answers(bridge, want):
        return False
    os.kill(ep.ctrl.proc.pid, signal.SIGSTOP)
    ep.paused = True
    return True


def hold_until_answer(sim, utime: int):
    """Commander hook: hold the measured pose until a node plan answering ``utime`` arrives,
    then hand the Franka back (a stale plan republished on wake is overridden every step: the
    OSC extrapolates an expired plan)."""
    hold = sim.make_hold_hook()

    def hook(step, t, joint_q, body_q):
        if _answers(sim.bridge, utime):
            sim.commander_hook = None
            return None
        return hold(step, t, joint_q, body_q)

    return hook


def _resume(ep: Ep) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(ep.ctrl.proc.pid, signal.SIGCONT)
    ep.paused = False


def _signals(ep: Ep, fr: dict, large7) -> dict:
    sig = s2.signals(fr, large7)
    fr["onpolicy_sig"] = sig
    return sig


def _sample(sim, ctx, sampler, k, s0, publish, timing):
    ep, cfg = OP.ep, OP.cfg
    if ep is None or not ep.in_loop:
        return OP.orig["_sample"](sim, ctx, sampler, k, s0, publish, timing)
    idx = ep.n_loop
    ep.n_loop += 1
    t_mpc = k * sim.frame_dt
    if (publish and cfg.excursions and ep.trip is None and t_mpc >= ep.next_t
            and len(ep.excursions) < cfg.ex_max and ep.ctrl.alive() and sim.bridge.sent_latents
            and slow_enough(ep.prev_ee, sim.franka_measured_pose7(), cfg.ex_max_speed_mm)):
        if _pause(ep, sim.bridge):
            _excursion(sim, ctx, sampler, k, idx, t_mpc, timing)
        else:
            ep.next_t = t_mpc + harness.SAMPLE_STEPS * sim.frame_dt
    fr = OP.orig["_sample"](sim, ctx, sampler, k, s0, publish and ep.trip is None, timing)
    if ep.paused:
        if ep.trip is None:
            sim.bridge._send_latent()  # in the socket before the node wakes
            sim.commander_hook = hold_until_answer(sim, sim.bridge.sent_latents[-1][0])
        _resume(ep)
        if ep.trip is not None:
            ep.ctrl.stop()  # safe hold: the loop aborts on "controller exited" and holds
    sig = _signals(ep, fr, ctx.cloud["pose"])
    ep.none_run = 0 if math.isfinite(sig["large_h_mm"]) else ep.none_run + 1
    if cfg.mpc_trip and ep.trip is None and publish and ep.ctrl.alive():
        why = mpc_trip(ep.prev_sig, sig, ep.none_run)
        if why is not None:
            ep.trip = {"why": f"mpc:{why}", "t": t_mpc, "frame_loop": idx, "signals": sig}
            fr["onpolicy_trip"] = why
            logger.warning(f"[ONPOL] {ep.name}: MPC trip {why} at {t_mpc:.2f} s -> safe hold")
            ep.ctrl.stop()
    ep.prev_sig, ep.prev_ee = sig, fr["ee_f"]
    return fr


def _excursion(sim, ctx, sampler, k: int, idx: int, t_mpc: float, timing: dict) -> None:
    ep, cfg, n = OP.ep, OP.cfg, harness.SAMPLE_STEPS
    W = sim.franka_measured_pose7()
    region = region_of(W)
    mult = OP.budget.mult[region]
    off = s2.draw_loop(ep.rng, OP.budget.box(region), cfg.emphasis_frac, away_dir(W))
    if s2.tilt_deg(W[3:]) + float(np.linalg.norm(off[3:5])) > cfg.tilt_max_deg:
        off[3:5] = 0.0  # keep under the MPC's tilt limit
    scale, d0, d1 = ep.clr.scale("franka", W, off)
    cyc = ms.CyclePath(W, ms.offset_pose(W, off * scale), cfg.dev_s, cfg.ret_s, cfg.hold_s)
    cmd = s2.Cmd(cyc, None, seg_of=lambda tau, c=cyc: ms.SEGMENTS[c.segment(tau)], home=W,
                 ret_s=cfg.ret_s, reroute_until=cyc.t1)
    ex_id = len(ep.excursions)
    rec = {"id": ex_id, "t_mpc": t_mpc, "loop_frame": idx, "region": region, "mult": mult,
           "polar": s2.small_polar(W[:3]), "offset_drawn": off.tolist(),
           "offset": (off * scale).tolist(), "scale": scale, "clearance_model_mm": [d0, d1],
           "ref": None, "trip": None, "trip_frame": None, "rerouted": False, "recovered": None}
    frames: list[dict] = []
    ur = s2.Fixed(pose7_mat(UrTracking.fk(sim.arm_targets()[1])))
    nan16, nan6 = np.full(16, np.nan), np.full(6, np.nan)
    state = {"prev": ep.prev_sig, "none": ep.none_run}

    def on_frame(kk, c, tau):
        fr = OP.orig["_sample"](sim, ctx, sampler, k, 0, False, timing)
        fr.update(c, n_knots=harness.N_KNOTS, cmd_source=CMD_EXCURSION, ur_line=nan16,
                  ur_fresh_exact=False, ur_own_delta=nan6, onpolicy_ex=ex_id,
                  onpolicy_seg=cmd.segment(tau))
        sig = _signals(ep, fr, ctx.cloud["pose"])
        state["none"] = 0 if math.isfinite(sig["large_h_mm"]) else state["none"] + 1
        frames.append(fr)
        if rec["ref"] is None:
            rec["ref"] = sig
        elif rec["trip"] is None:
            why = ex_trip(rec["ref"], sig, region) or (
                mpc_trip(state["prev"], sig, state["none"]) if cfg.mpc_trip else None)
            if why is not None:
                rec.update(trip=why, trip_frame=len(frames) - 1, trip_tau=tau,
                           rerouted=cmd.trip(tau), trip_signals=sig)
                fr["onpolicy_trip"] = why
                fr.update(s2.command_at(cmd, tau), onpolicy_seg=cmd.segment(tau))
        state["prev"] = sig

    st = sim.step_index
    steps = n * math.ceil(round(cmd.end / sim.frame_dt) / n)
    rep = ms.drive(sim, ctx, cmd, ur, steps, on_frame, grasp_arms="franka")
    while (sim.step_index - st) % n:
        sim.control_step()
    ep.tip_min_mm = min(ep.tip_min_mm, rep["tip_min_mm"])
    rec.update(abort=rep["abort"], abort_kind=rep["abort_kind"], n_frames=len(frames),
               duration_s=(sim.step_index - st) * sim.frame_dt,
               min_clearance_mm=rep["min_clearance_mm"], tip_min_mm=rep["tip_min_mm"])
    if rec["trip"] is not None:
        OP.budget.halve(region, rec["trip"], ep.name)
        rec["recovered"] = ex_trip(rec["ref"], frames[-1]["onpolicy_sig"], region) is None
    if rep["abort"]:
        ep.trip = {"why": f"excursion_abort:{rep['abort_kind']}", "t": t_mpc, "frame_loop": idx}
    elif rec["trip"] is not None and not rec["recovered"]:
        ep.trip = {"why": f"excursion:{rec['trip']}", "t": t_mpc, "frame_loop": idx}
    ep.prev_sig, ep.none_run = frames[-1]["onpolicy_sig"], state["none"]
    ep.pending.append((idx, frames, rec))
    ep.excursions.append(rec)
    ep.next_t = t_mpc + cfg.ex_every_s
    logger.info(f"[ONPOL] {ep.name} ex {ex_id} @ {t_mpc:.1f} s {region} x{mult:g}: |d| "
                f"{np.linalg.norm(off[:3] * scale):.1f} mm, {len(frames)} frames"
                + (f", TRIP {rec['trip']} (recovered {rec['recovered']})" if rec["trip"] else "")
                + (f", ABORT {rep['abort']}" if rep["abort"] else ""))


def _build_writer(frames, pcd, definition=lcs.DEFAULT_ACTION_DEFINITION):
    writer, track = OP.orig["build_writer"](frames, pcd, definition)
    ep = OP.ep
    if ep is None:
        return writer, track
    seg = [segment_of(fr) for fr in frames]
    ex = [int(fr.get("onpolicy_ex", -1)) for fr in frames]
    small = [s2.small_cloud(fr["pcd_belt_seg"], bc.known_large_pulley_pose7())["n_near"]
             if "onpolicy_sig" not in fr else fr["onpolicy_sig"]["small"]["n_near"]
             for fr in frames]
    arrays = {"onpolicy_segment": np.array(seg), "onpolicy_excursion": np.array(ex, np.int16),
              "onpolicy_trip": np.array([fr.get("onpolicy_trip", "") for fr in frames]),
              "cloud_small_n_near": np.array(small, np.int32)}
    arrays = {lcs.EXTRA_PREFIX + k: v for k, v in arrays.items()}
    write = writer.write

    def write_onpolicy(path, label, meta, arrays_=None, omit=()):
        meta = {**meta, "cmd_source_codes": {**meta["cmd_source_codes"],
                                             "onpolicy_excursion": CMD_EXCURSION},
                "onpolicy": {**ep.summary(), "segment_labels": list(SEGMENTS),
                             "config": _cfg_dict(OP.cfg)}}
        return write(path, label, meta, arrays={**(arrays_ or {}), **arrays}, omit=omit)

    writer.write = lambda path, label, meta, arrays=None, omit=(): write_onpolicy(
        path, label, meta, arrays, omit)
    return writer, track


def _cfg_dict(cfg) -> dict:
    return {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()},
            "regions": REGIONS, "near": {"r_mm": NEAR_R_MM, "h_mm": NEAR_H_MM},
            "ex_trip": dataclasses.asdict(EX_TRIP), "mpc_trip": dataclasses.asdict(MPC_TRIP)}


# --- outputs ----------------------------------------------------------------------------------

def buffer_rows(index: dict, out: Path) -> list[dict]:
    rows = []
    for r in (e for e in index["episodes"] if e.get("file") and "post_mpc" in e):
        path = (out / r["file"]).resolve()
        with np.load(path, allow_pickle=True) as d:
            seg = [str(s) for s in d["sim_onpolicy_segment"]]
            ex = d["sim_onpolicy_excursion"].tolist()
            src = d["sim_cmd_source"]
        pm, op = r["post_mpc"], r["onpolicy"]
        fin = pm["final"]
        n_task = int(pm["from_frame"])
        rows.append({
            "id": f"{KIND}_{out.name}_{len(rows):04d}", "iter": -1, "kind": KIND, "status": "ok",
            "weight": 1.0, "episode": str(path), "label_human": bool(fin["both"]),
            "label_human_detail": {"success": bool(fin["both"]), "rule": "both",
                                   "source": "sim_ground_truth_final_frame (after the release)",
                                   "large": fin["large"], "small": fin["small"],
                                   "mpc_end": {k: pm["mpc_end"][k] for k in ("large", "small",
                                                                             "both", "held")}},
            "label_strict": bool(fin["both"]), "cloud_success": None,
            "cloud_success_role": "monitor only",
            "start": start_row(r["start_state"]),
            "reset_before": "human", "n_frames": len(seg),
            "n_mpc_frames": int(np.isin(src[:n_task], (0, 1)).sum()),
            "n_excursion_frames": int((src[:n_task] == CMD_EXCURSION).sum()),
            "n_task_frames": n_task, "post_from": n_task, "outcome_gt": r["outcome"],
            "segment": "episode", "segments": segments_of(seg, ex),
            "controller": {"deploy": str(index["args"].get("deploy")),
                           "config": index["controller"]["config"],
                           "action_noise_frac": index["controller"]["mppi_params"].get(
                               "action_noise_frac", 0.0)},
            "onpolicy": {"n_excursions": op["n_excursions"], "n_ex_trips": op["n_ex_trips"],
                         "trip": None if op["trip"] is None else op["trip"]["why"],
                         "status": r["status"], "reason": r["reason"],
                         "extra_reset": extra_reset(r)},
        })
    return rows


def start_row(st: dict) -> dict:
    sid = st["id"]
    arm, rest = sid.split("_", 1)
    f = Path(st["file"])
    return {"set": "stage2", "id": sid, "key": rest.rsplit("_", 1)[0], "arm": arm,
            "file": str(f.relative_to(REPO_ROOT)) if f.is_relative_to(REPO_ROOT) else str(f),
            "from_reset": False, "jitter": None}


def write_report(out: Path, index: dict, rows: list[dict], budget: Budget, wall_s: float) -> None:
    eps = [e for e in index["episodes"] if e.get("file") and "post_mpc" in e]
    n = len(eps)
    both = sum(e["post_mpc"]["final"]["both"] for e in eps)
    resets = sum(extra_reset(e) for e in eps)
    ex = [x for e in eps for x in e["onpolicy"]["excursions"]]
    trips = Counter(x["trip"] for x in ex if x["trip"])
    holds = Counter(e["onpolicy"]["trip"]["why"] for e in eps if e["onpolicy"]["trip"])
    frames = sum(r["n_frames"] for r in rows)
    task = sum(r["n_task_frames"] for r in rows)
    exf = sum(r["n_excursion_frames"] for r in rows)
    L = [f"# Stage-2 on-policy collection `{out.relative_to(REPO_ROOT)}`", "",
         (f"{n} episodes ({len(index['episodes']) - n} skipped), wall {wall_s / 60:.1f} min; "
          f"deploy `{index['args'].get('deploy')}`, config `{index['controller']['config']}` "
          f"(action_noise_frac {index['controller']['mppi_params'].get('action_noise_frac')})."),
         "", f"- both engaged after the release (sim GT): {both}/{n}",
         (f"- extra resets (large off after the release or Franka not holding at the MPC end): "
         f"{resets}/{n}; useful (no extra reset) {n - resets}"),
         f"- frames {frames} (task {task}, of which excursion {exf})",
         (f"- excursions {len(ex)}: trips {dict(trips)}, recovered "
         f"{sum(bool(x['recovered']) for x in ex if x['trip'])}, aborts "
         f"{sum(x['abort'] is not None for x in ex)}"),
         f"- safe holds: {dict(holds) or 'none'}",
         f"- budget multipliers {budget.mult} (history {len(budget.history)})", "",
         ("| episode | status | reason | excursions (trips) | hold | MPC end L/S held | final L/S | "
         "both | reset | frames (task) |"), "|" + "---|" * 10]
    for e, r in zip(eps, rows, strict=True):
        pm, op = e["post_mpc"], e["onpolicy"]
        L.append(f"| {e['file']} | {e['status']} | {(e['reason'] or '-')[:60]} | "
                 f"{op['n_excursions']} ({op['n_ex_trips']}) | "
                 f"{op['trip']['why'] if op['trip'] else '-'} | {pm['mpc_end']['large']}/"
                 f"{pm['mpc_end']['small']} {pm['mpc_end']['held']} | {pm['final']['large']}/"
                 f"{pm['final']['small']} | {pm['final']['both']} | {extra_reset(e)} | "
                 f"{r['n_frames']} ({r['n_task_frames']}) |")
    (out / "report.md").write_text("\n".join(L) + "\n")


# --- CLI --------------------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                epilog="Arguments after -- go to eval_learned_mpc.py.")
    p.add_argument("--no-excursions", dest="excursions", action="store_false")
    p.add_argument("--no-mpc-trip", dest="mpc_trip", action="store_false")
    p.add_argument("--ex-first-s", type=float, default=1.5, help="MPC time of the first one")
    p.add_argument("--ex-every-s", type=float, default=3.0, help="MPC time between starts")
    p.add_argument("--ex-max", type=int, default=6, help="per episode")
    # V12-99 b01: belt lost after 4/11 excursions started > 3 mm/frame, 0/39 below
    p.add_argument("--ex-max-speed-mm", type=float, default=3.0,
                   help="start only if the hand moved <= this over the last 75 ms frame")
    p.add_argument("--dev-s", type=float, default=0.5)
    p.add_argument("--ret-s", type=float, default=0.5)
    p.add_argument("--hold-s", type=float, default=0.2)
    p.add_argument("--emphasis-frac", type=float, default=0.5)
    p.add_argument("--tilt-max-deg", type=float, default=30.0,
                   help="no tilt in the draw when the hand + draw would exceed it")
    p.add_argument("--ex-seed", type=int, default=20261020)
    p.add_argument("--budget-from", type=Path, default=None,
                   help="start from an earlier run's budget.json multipliers")
    p.add_argument("--no-loader-check", dest="loader_check", action="store_false")
    return p


def harness_args(argv: list[str]) -> argparse.Namespace:
    a = harness.create_parser().parse_args(argv)
    need = {"controller": "mppi", "post_mpc": "stage2_release", "grasp_arms": "franka",
            "outcome_pulley": "both", "belt_seg": True, "mode": "learned"}
    bad = {k: getattr(a, k) for k, v in need.items() if getattr(a, k) != v}
    if bad or a.out is None:
        raise SystemExit(f"harness flags {bad} (need {need}) and --out are required")
    return a


def main() -> int:
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    cfg = create_parser().parse_args(argv[:cut])
    hargs = harness_args(argv[cut + 1:])
    out = (REPO_ROOT / hargs.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} exists and is not empty; pick a new dir")
    out.mkdir(parents=True, exist_ok=True)
    hargs.out = out
    cli.configure_logging()
    budget = Budget(out / "budget.json", cfg.budget_from)
    install(cfg, budget)
    t0 = time.perf_counter()
    try:
        index = harness.evaluate(hargs)
    finally:
        uninstall()
    wall = time.perf_counter() - t0
    rows = buffer_rows(index, out)
    write_json(out / "buffer.json", {"version": 2, "iter": -1, "n": len(rows), "kind": KIND,
                                     "source": "stage-2 on-policy MPC + excursions (V12-99)",
                                     "config": _cfg_dict(cfg), "rollouts": rows})
    if cfg.loader_check and rows:
        index["loader"] = ms.loader_check([Path(r["episode"]) for r in rows])
        write_json(out / "index.json", index)
    write_report(out, index, rows, budget, wall)
    logger.info(f"[ONPOL] {len(rows)} episodes -> {out / 'buffer.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
