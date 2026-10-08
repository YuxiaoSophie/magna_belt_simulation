#!/usr/bin/env python3
"""Seed buffer of the real-first loop (V11): UR-held demos + OU excitation rollouts.

Runs the collector's OSC machinery in-process (``collect_lcs_dataset.collect`` with
``run_osc_episode`` wrapped, as ``collect_contact_branches.py``) over one planned chain of
episodes, every one recorded with the cloud noise preset + ``--belt-seg``:

    demo        ``--scenario nominal --hold-ur-gripper``, no excitation (the teleop stand-in)
    excitation  the same nominal waypoints + OU excitation of both arms (the random policy)

Episodes are grouped by start state; within a group they are chained through
``scripted_reset.scripted_reset`` toward the group's first-frame EE poses (both grasps held at
the end of the previous episode), else the start snapshot is restored (a human reset). Each
episode is labelled by the emulated human (``human_label.py``, engaged rule); the cloud success
detector is logged as a monitor only. Writes ``<out>/{demo,excite}/episode_*.npz``,
``buffer.json`` (plan §6), ``budget.json`` (plan §3), ``seed_report.md`` and ``index.json``.

Run (detached, private URL):
    uv run --frozen python scripts/lcs/realfirst_seed.py --out data/lcs/realfirst/v1/seed \\
        --lcm-url 'udpm://239.255.76.163:7763?ttl=0' --cloud-noise real_like --seed 20261005 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
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

import collect_lcs_dataset as col
import human_label as hl
import scripted_reset as sr
from loguru import logger

from round_belt_task import perturbation as pert
from round_belt_task.episode_io import git_info, sha256_file, write_json
from round_belt_task.osc_bridge import OscTimeout
from task_common import belt_cloud as bc
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.magna_process import check_private_url

STARTS = REPO_ROOT / "data" / "lcs" / "start_states"
GV = STARTS / "grasp_variants"
# set2 gv_00 = the zero-offset (nominal) grasp of the collection pool; pre_place_1_osc is held out
DEMO_STARTS = (("set2", "gv_00"), ("set2", "gv_01"), ("set2", "gv_04"))
EXCITE_STARTS = (("set2", "gv_01"), ("set2", "gv_04"))
EXCITE_POOL = "set3"
HELDOUT = (STARTS / "pre_place_1_osc.npz", GV / "set1" / "gv_01_osc.npz",
           GV / "set1" / "gv_05_osc.npz", STARTS / "ur_depth" / "ud+3_r0_osc.npz",
           STARTS / "ur_depth" / "ud+6_r0_osc.npz")
KIND_DIR = {"demo": "demo", "excitation": "excite"}
BOX_MM, BOX_DEG, ENVELOPE_DEG = 6.0, 2.0, 20.0
# cap of knot1 - knot0 = 6 mm / 2.148 deg x factor: 0.93 keeps Franka steps inside the box
DEFAULT_CAP_FACTOR = 0.93
DEFAULT_RESET_SPEC = ("slack_mm=10",)
# fallback demo (reference §5.3): lower corner of OSC_RANGES["engaged"]
FALLBACK_UR_DZ_MM, FALLBACK_UR_TILT_DEG = 1.5, 2.5
LAST_N = hl.LAST_N
MIN_DEMO_SUCCESS = 2


@dataclasses.dataclass
class Slot:
    i: int
    kind: str
    set: str
    vid: str
    file: Path
    fallback: bool = False

    @property
    def key(self) -> str:
        return f"{self.set}/{self.vid}"

    def perturbation(self) -> pert.Perturbation:
        if not self.fallback:
            return pert.Perturbation(intent="nominal")
        return pert.Perturbation(intent="engaged",
                                 ur_dpos_m=np.array([0.0, 0.0, FALLBACK_UR_DZ_MM * 1e-3]),
                                 ur_tilt_deg=FALLBACK_UR_TILT_DEG)


def variant_file(set_name: str, vid: str) -> Path:
    index = json.loads((GV / set_name / "index.json").read_text())
    row = next((r for r in index["variants"] if str(r["id"]) == vid), None)
    if row is None or row.get("file") is None or row.get("held") is False:
        raise ValueError(f"{set_name}/{vid}: no held start state")
    return GV / set_name / row["file"]


def pick_pool_variant(seed: int) -> str:
    """One variant of ``EXCITE_POOL`` (not gv_00, the nominal grasp) drawn from ``seed``."""
    index = json.loads((GV / EXCITE_POOL / "index.json").read_text())
    ids = sorted(str(r["id"]) for r in index["variants"]
                 if r.get("held") is not False and r.get("file") and str(r["id"]) != "gv_00")
    return ids[int(np.random.default_rng([seed, 3]).integers(len(ids)))]


def parse_starts(text: str) -> tuple[tuple[str, str], ...]:
    """``set2/gv_00,set2/gv_01`` -> ``(("set2", "gv_00"), ("set2", "gv_01"))``."""
    out = tuple(tuple(x.strip().split("/")) for x in text.split(",") if x.strip())
    if not out or any(len(x) != 2 for x in out):
        raise ValueError(f"--demo-starts {text!r}: want SET/VARIANT[,SET/VARIANT...]")
    return out


def plan_slots(n_demo: int, n_excite: int, seed: int,
               demo_starts: tuple[tuple[str, str], ...] = DEMO_STARTS) -> list[Slot]:
    """Episodes grouped by start (demos first): each group change is a human reset."""
    if n_demo % len(demo_starts):
        raise ValueError(f"--demos {n_demo}: want a multiple of {len(demo_starts)}")
    ex_starts = [*EXCITE_STARTS, (EXCITE_POOL, pick_pool_variant(seed))]
    base, rem = divmod(n_excite, len(ex_starts))
    n_ex = {s: base + (j < rem) for j, s in enumerate(ex_starts)}
    groups = list(dict.fromkeys([*demo_starts, *ex_starts]))
    slots: list[Slot] = []
    for s in groups:
        n_d = n_demo // len(demo_starts) if s in demo_starts else 0
        if n_d + n_ex.get(s, 0) == 0:
            continue
        f = variant_file(*s)
        for kind, n in (("demo", n_d),
                        ("excitation", n_ex.get(s, 0))):
            slots += [Slot(len(slots) + j, kind, s[0], s[1], f) for j in range(n)]
    for j, s in enumerate(slots):
        s.i = j
    return slots


def check_not_heldout(slots: list[Slot]) -> None:
    bad = {sha256_file(p): p for p in HELDOUT if p.is_file()}
    for s in slots:
        if s.file.resolve() in {p.resolve() for p in HELDOUT} or sha256_file(s.file) in bad:
            raise ValueError(f"slot {s.i} start {s.file} is a held-out eval cell")


# --- per-episode measures -------------------------------------------------------------------

def _angle_deg(q0, q1) -> float:
    return math.degrees(float(np.linalg.norm(lcs.delta_rotvec(np.asarray(q0, np.float64),
                                                              np.asarray(q1, np.float64)))))


def episode_measures(path: Path, noise_name: str, n: int) -> dict:
    """Validation, label (human emulation + cloud monitor), box / envelope stats of one file."""
    summary = lcs.validate_episode(path, period_us=round(n * lcs.SIM_DT_S * 1e6))
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        meta = json.loads(str(d["sim_meta"]))
        T = int(d["actions"].shape[0])
        acts = d["actions"].astype(np.float64)
        seg = d["pcd_belt_seg"] if "pcd_belt_seg" in files else None
        cloud_ok = d["sim_cloud_success"] if "sim_cloud_success" in files else None
        ee = {"franka": d["sim_ee_franka"].astype(np.float64),
              "ur": d["sim_ee_ur"].astype(np.float64)}
        phase = d["sim_phase"] if "sim_phase" in files else None
        home = (ee["franka"][0].copy(), ee["ur"][0].copy())
        label = hl.emulated_human_label(d, "engaged")
        strict = hl.emulated_human_label(d, "strict")
        outcome = str(d["sim_outcome"]) if "sim_outcome" in files else None
    cloud_keys = [k for k in (lcs.EXTRA_PREFIX + c for c in
                              ("cloud_h_mm", "cloud_wrap_deg", "cloud_r_mm", "cloud_n_pts",
                               "cloud_success")) if k in files]
    bs = meta.get("belt_seg") or {}
    noise_ok = bs.get("noise_spec") == bc.cloud_noise_spec(noise_name).to_dict()
    tail = None if cloud_ok is None else np.asarray(cloud_ok, bool)[-LAST_N:]
    trans = np.abs(acts[:, :6]) * 1e3
    rot = np.degrees(np.abs(acts[:, 6:]))
    over = (trans > BOX_MM + 1e-6).any(1) | (rot > BOX_DEG + 1e-6).any(1)
    env = {arm: max(_angle_deg(v[0, 3:], v[t, 3:]) for t in range(len(v)))
           for arm, v in ee.items()}
    pre = 0
    if phase is not None:
        labels = meta.get("phase_labels") or []
        pre = int(sum(labels[int(p)] == col.PREHOLD_PHASE for p in phase)) if labels else 0
    return {
        "valid": True, "validate": {k: v for k, v in summary.items()
                                    if isinstance(v, (int, float, str, bool))},
        "n_frames": T, "n_prehold_frames": pre,
        "has_pcd_belt_seg": seg is not None, "cloud_keys": len(cloud_keys),
        "seg_m_min": None if seg is None else int(min(len(s) for s in seg)),
        "seg_m_median": None if seg is None else float(np.median([len(s) for s in seg])),
        "belt_seg_noise_real_like": bool(noise_ok), "belt_seg_source": bs.get("source"),
        "label_human": label, "label_strict": strict, "outcome_gt": outcome,
        "cloud_success": None if tail is None else bool(2 * int(tail.sum()) > len(tail)),
        "cloud_success_frames": None if tail is None else int(tail.sum()),
        "box": {"frames_outside": int(over.sum()),
                "trans_max_mm": [round(float(v), 3) for v in trans.max(0)],
                "rot_max_deg": [round(float(v), 3) for v in rot.max(0)]},
        "envelope_max_deg": {k: round(v, 2) for k, v in env.items()},
        "home": home,
    }


# --- the chain (replaces col.run_osc_episode during collect) --------------------------------

class Chain:
    def __init__(self, args, n: int, spec: sr.ResetSpec, excite_cfg: dict) -> None:
        self.args, self.n, self.spec, self.excite_cfg = args, n, spec, excite_cfg
        self.out = Path(args.out)
        self.rng = np.random.default_rng(spec.seed)
        self.homes: dict[str, tuple] = {}
        self.snaps: dict[str, object] = {}
        self.slots: list[Slot] = []
        self.last: tuple[Slot, dict] | None = None
        self.records: list[dict] = []  # one per ok episode (the buffer rows)
        self.attempts: list[dict] = []  # every rollout incl. skipped ones
        self.resets: list[dict] = []  # every reset before an attempt

    def start_pass(self, slots: list[Slot]) -> None:
        self.slots, self.last = slots, None

    def _reset_before(self, sim, ctx, slot: Slot) -> tuple[dict, bool]:
        """``(reset row, chained)``; mirrors ``eval_learned_mpc._reset_before``."""
        reason = None
        if self.last is None:
            reason = "initial start state"
        elif self.last[0].key != slot.key:
            reason = "start state changed"
        elif (self.last[1].get("status") != "ok"
              or self.last[1].get("grasp_ok_final") != [True, True]):
            reason = "grasp not held at the end of the previous episode"
        elif slot.key not in self.homes:
            reason = "no home poses for this start state"
        if reason is not None:
            return sr.ResetReport.human(reason).to_dict(), False
        home_f, home_u = self.homes[slot.key]
        try:
            rep = sr.scripted_reset(sim, ctx, home_f, home_u, self.spec, self.rng)
        except OscTimeout as exc:
            if not exc.process_alive:
                raise
            sim.bridge.forget_pending()
            rep = sr.ResetReport(ok=False, kind="scripted", reason=f"osc timeout: {exc}",
                                 grasp_held=None, duration_s=0.0)
        if not rep.ok:
            return sr.ResetReport.human(f"scripted reset failed: {rep.reason}", rep).to_dict(), \
                False
        return rep.to_dict(), True

    def _attempt(self, sim, ctx, slot: Slot, chained: bool) -> tuple[dict, float]:
        """One ``run_osc_episode``; the restore is skipped when chained. ``(row, robot_s)``."""
        p = slot.perturbation()
        mark = {"step": sim.step_index}
        cls_restore = type(sim).restore

        def restore(snap, *a, **kw):
            out = None
            if not chained:
                out = cls_restore(sim, snap, *a, **kw)
            mark["step"] = sim.step_index
            return out

        sim.restore = restore
        try:
            row = ORIG_RUN(sim, ctx, slot.i, p)
        except OscTimeout as exc:
            col._finish_recording(sim, ctx.out, f"episode_{slot.i:04d}-{p.intent}",
                                  "osc timeout")
            if not exc.process_alive:
                raise
            sim.bridge.forget_pending()
            row = {"file": None, "intent": p.intent, "status": "skipped",
                   "reason": "osc timeout", "error": str(exc)}
        finally:
            sim.__dict__.pop("restore", None)
        return row, (sim.step_index - mark["step"]) * sim.frame_dt

    def __call__(self, sim, ctx, i: int, p) -> dict:
        slot = self.slots[i]
        snap = self.snaps.get(slot.key)
        if snap is None:
            snap = self.snaps[slot.key] = sim_snapshot.load(slot.file)
        ctx.out = self.out / KIND_DIR[slot.kind]
        ctx.out.mkdir(parents=True, exist_ok=True)
        ctx.snap, ctx.start_state = snap, slot.file
        ctx.scenario = "nominal" if slot.kind == "demo" else None
        ctx.excite = self.excite_cfg[slot.kind]
        reset, chained = self._reset_before(sim, ctx, slot)
        row, retried = None, None
        for attempt in range(2):
            self.resets.append({"before": slot.i, "try": attempt, **reset})
            row, robot_s = self._attempt(sim, ctx, slot, chained)
            self.attempts.append({"id": slot.i, "kind": slot.kind, "attempt": attempt,
                                  "status": row.get("status"), "reason": row.get("reason"),
                                  "robot_s": robot_s})
            if row.get("status") == "ok":
                break
            logger.warning(f"[SEED] {slot.i:3d} {slot.kind} skipped ({row.get('reason')}); "
                           + ("retry from the snapshot" if attempt == 0 else "giving up"))
            retried = row.get("reason")
            reset = sr.ResetReport.human(f"retry after a skipped episode: {retried}").to_dict()
            chained = False
        self.last = (slot, row)
        row.update(kind=slot.kind, start_set=slot.set, start_variant=slot.vid,
                   seed_id=slot.i, reset_before=reset["kind"], retried=retried,
                   fallback_demo=slot.fallback)
        if row.get("status") != "ok":
            return row
        path = ctx.out / row["file"]
        m = episode_measures(path, self.args.cloud_noise, self.n)
        if not chained and slot.key not in self.homes:
            self.homes[slot.key] = m["home"]
        lab = m["label_human"]
        row.update(path=str(path.relative_to(self.out)), label_human=lab["success"],
                   cloud_success=m["cloud_success"])
        self.records.append({"slot": slot, "path": path, "reset": reset, "measures": m,
                             "row": {k: v for k, v in row.items() if k != "osc_stats"}})
        logger.info(f"[SEED] {slot.i:3d} {slot.kind:<10} {slot.key:<10} reset {reset['kind']:<8}"
                    f" GT {m['outcome_gt']:<8} human {lab['success']!s:<5} (wrap "
                    f"{lab['wrap_deg']:.0f}) cloud {m['cloud_success']} | box out "
                    f"{m['box']['frames_outside']}/{m['n_frames']}, env "
                    f"{m['envelope_max_deg']}")
        return row


ORIG_RUN = col.run_osc_episode


def collect_args(args, slots: list[Slot], out: Path) -> argparse.Namespace:
    cli = ["--out", str(out), "--episodes", str(len(slots)), "--seed", str(args.seed),
           "--lcm-url", args.lcm_url, "--backend", "osc", "--scenario", "nominal",
           "--hold-ur-gripper", "--start-state", str(slots[0].file),
           "--settle-s", str(args.hold_s), "--max-episode-s", str(args.max_episode_s),
           "--cloud-noise", args.cloud_noise, "--belt-seg", "--label", "realfirst-seed",
           "--excite-pos-mm", str(col.EXCITE_DEFAULTS["ou"][0]),
           "--excite-rot-deg", str(col.EXCITE_DEFAULTS["ou"][1]),
           "--excite-ur-pos-mm", str(col.EXCITE_UR_DEFAULTS[0]),
           "--excite-ur-rot-deg", str(col.EXCITE_UR_DEFAULTS[1]),
           "--excite-cap-factor", str(args.excite_cap_factor)]
    if args.record:
        cli.append("--record")
    return col.create_parser().parse_args(cli)


def run_pass(chain: Chain, args, slots: list[Slot], index_dir: Path) -> dict:
    cargs = collect_args(args, slots, index_dir)
    excite = col.excitation_config(cargs, None)
    if not (excite["on"] and excite["ur"]["on"]):
        raise RuntimeError(f"excitation config not on: {excite}")
    chain.excite_cfg = {"demo": col.excitation_config(cargs, "nominal"), "excitation": excite}
    chain.start_pass(slots)
    col.run_osc_episode = chain
    try:
        return col.collect(cargs, scenario="nominal")
    finally:
        col.run_osc_episode = ORIG_RUN


# --- outputs --------------------------------------------------------------------------------

def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    return v


def buffer_rows(chain: Chain, id_prefix: str = "") -> list[dict]:
    out = []
    for r in sorted(chain.records, key=lambda r: r["slot"].i):
        s, m, rs = r["slot"], r["measures"], r["reset"]
        lab = m["label_human"]
        out.append({
            "id": f"{id_prefix}{s.kind}_{s.i:04d}" if id_prefix else f"seed_{s.i:04d}",
            "iter": -1, "kind": s.kind, "status": "ok", "weight": 1.0,
            "episode": str(r["path"].resolve()),
            "label_human": bool(lab["success"]),
            "label_human_detail": {k: lab[k] for k in ("success", "rule", "source", "wrap_deg",
                                                       "h_mm", "n_pass", "last_n")},
            "label_strict": bool(m["label_strict"]["success"]),
            "cloud_success": m["cloud_success"], "cloud_success_role": "monitor only",
            "start": {"set": s.set, "id": s.vid, "file": str(s.file.relative_to(REPO_ROOT)),
                      "from_reset": rs["kind"] == "scripted", "jitter": rs.get("jitter")},
            "reset_before": rs["kind"], "n_frames": m["n_frames"], "n_mpc_frames": 0,
            "n_prehold_frames": m["n_prehold_frames"], "outcome_gt": m["outcome_gt"],
            "fallback_demo": s.fallback,
        })
    return out


def budget(chain: Chain, wall_s: float) -> dict:
    ok = [a for a in chain.attempts if a["status"] == "ok"]
    resets = chain.resets
    between = [x for x in resets if x["reason"] != "initial start state"]
    rob_ep = float(sum(a["robot_s"] for a in chain.attempts))
    rob_reset = float(sum(x["duration_s"] for x in resets))
    return {
        "rollouts_demo": sum(a["kind"] == "demo" for a in ok),
        "rollouts_excitation": sum(a["kind"] == "excitation" for a in ok),
        "rollouts_mpc": 0, "rollouts_eval": 0,
        "rollouts_skipped": len(chain.attempts) - len(ok),
        "resets_scripted": sum(x["kind"] == "scripted" for x in resets),
        "resets_human": sum(x["kind"] == "human" for x in resets),
        "resets_between_episodes": len(between),
        "resets_between_scripted": sum(x["kind"] == "scripted" for x in between),
        "resets_scripted_failed": sum(x["kind"] == "human" and x.get("attempt") is not None
                                      for x in resets),
        "robot_s": rob_ep + rob_reset,
        "robot_s_detail": {"episodes_incl_settle_prehold": rob_ep, "resets": rob_reset},
        "labels_human": len(chain.records), "gpu_s": 0.0, "wall_s": wall_s,
        "notes": "resets_human counts the initial restore and every snapshot restore (start "
                 "change, failed scripted reset, retry); robot_s = sim seconds under control",
    }


def acceptance(chain: Chain, rows: list[dict], bud: dict, n_planned: int,
               min_scripted: int = 15) -> dict:
    ms = [r["measures"] for r in chain.records]
    demos_ok = sum(r["kind"] == "demo" and r["label_human"] for r in rows)
    files_ok = all(m["valid"] and m["has_pcd_belt_seg"] and m["cloud_keys"] == 5
                   and m["belt_seg_noise_real_like"] for m in ms)
    consistent = (bud["rollouts_demo"] + bud["rollouts_excitation"] == len(rows)
                  and bud["labels_human"] == len(rows)
                  and bud["resets_scripted"] + bud["resets_human"] == len(chain.attempts)
                  and sum(r["reset_before"] == "scripted" for r in rows)
                  <= bud["resets_scripted"])
    return {
        "episodes": {"want": n_planned, "got": len(rows), "pass": len(rows) == n_planned},
        "files_valid_seg_cloud_noise": {"pass": bool(files_ok) and len(ms) == n_planned},
        "demo_successes": {"want": f">= {MIN_DEMO_SUCCESS}", "got": demos_ok,
                           "pass": demos_ok >= MIN_DEMO_SUCCESS},
        "buffer_budget_consistent": {"pass": bool(consistent)},
        "resets_scripted_between": {"want": f">= {min_scripted} of "
                                            f"{bud['resets_between_episodes']}",
                                    "got": f"{bud['resets_between_scripted']} of "
                                           f"{bud['resets_between_episodes']}",
                                    "pass": bud["resets_between_scripted"] >= min_scripted},
    }


def _fmt(v, nd=1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_report(out: Path, args, chain: Chain, rows: list[dict], bud: dict, acc: dict,
                 meta: dict) -> None:
    recs = sorted(chain.records, key=lambda r: r["slot"].i)
    ex_f, ex_u = col.EXCITE_DEFAULTS["ou"], col.EXCITE_UR_DEFAULTS
    head = (f"Created {meta['created']}, git {meta['git'].get('commit', '?')[:10]}, seed "
            f"{args.seed}, LCM `{args.lcm_url}`, wall {bud['wall_s'] / 60:.1f} min, robot "
            f"{bud['robot_s']:.0f} s sim.")
    rec = (f"Recording: `--cloud-noise {args.cloud_noise} --belt-seg`, `--hold-ur-gripper`, "
           f"hold {args.hold_s:g} s after place_3, max {args.max_episode_s:g} s to done; reset "
           f"spec `{' '.join(args.reset_spec)}` (jitter seed {chain.spec.seed}); excitation OU "
           f"{ex_f[0]} mm / {ex_f[1]} deg (Franka), {ex_u[0]} mm / {ex_u[1]} deg (UR), cap "
           f"factor {args.excite_cap_factor}. Label = emulated human, engaged rule, majority of "
           f"the last {LAST_N} frames (`sim_ground_truth_final_frames`); cloud success = "
           "monitor only.")
    L = [f"# Seed buffer `{out.relative_to(REPO_ROOT)}` (V11 real-first, e1)", "", head, "",
         rec, "", "## Acceptance", "", "| criterion | want | got | pass |", "|---|---|---|---|"]
    for k, v in acc.items():
        L.append(f"| {k} | {v.get('want', '-')} | {v.get('got', '-')} | "
                 f"{'PASS' if v['pass'] else 'FAIL'} |")
    L += ["", "## Outcomes per kind", "",
          "| kind | n | human (engaged) | strict | cloud (monitor) | GT outcomes |",
          "|---|---|---|---|---|---|"]
    for kind in ("demo", "excitation"):
        rs = [r for r in rows if r["kind"] == kind]
        gt = ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["outcome_gt"] for r in rs).items()))
        L.append(f"| {kind} | {len(rs)} | {sum(r['label_human'] for r in rs)} | "
                 f"{sum(r['label_strict'] for r in rs)} | "
                 f"{sum(bool(r['cloud_success']) for r in rs)} | {gt} |")
    tp = sum(r["label_human"] and bool(r["cloud_success"]) for r in rows)
    tn = sum((not r["label_human"]) and not r["cloud_success"] for r in rows)
    fp = sum((not r["label_human"]) and bool(r["cloud_success"]) for r in rows)
    fn = sum(r["label_human"] and not r["cloud_success"] for r in rows)
    rate = f" ({(tp + tn) / len(rows):.0%})" if rows else ""
    L += ["", "## Label agreement (human vs cloud monitor, episode level)", "",
          "| | cloud success | cloud fail |", "|---|---|---|",
          f"| human success | {tp} | {fn} |", f"| human fail | {fp} | {tn} |", "",
          f"Agreement {tp + tn}/{len(rows)}{rate}.", "", "## Episodes", "",
          ("| id | kind | start | reset | GT | human | strict | cloud | wrap | h mm | frames "
           "| seg M min | box out | env F/U deg |"),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        s, m = r["slot"], r["measures"]
        lab, env = m["label_human"], m["envelope_max_deg"]
        L.append(f"| {s.i} | {s.kind} | {s.key} | {r['reset']['kind']} | {m['outcome_gt']} | "
                 f"{lab['success']} | {m['label_strict']['success']} | {m['cloud_success']} | "
                 f"{_fmt(lab['wrap_deg'], 0)} | {_fmt(lab['h_mm'])} | {m['n_frames']} | "
                 f"{m['seg_m_min']} | {m['box']['frames_outside']} | "
                 f"{env['franka']:.1f}/{env['ur']:.1f} |")
    allm = [r["measures"] for r in recs]
    n_fr = sum(m["n_frames"] for m in allm)
    n_out = sum(m["box"]["frames_outside"] for m in allm)
    env_over = sum(max(m["envelope_max_deg"].values()) > ENVELOPE_DEG for m in allm)
    L += ["", (f"Action box (+-{BOX_MM:g} mm / +-{BOX_DEG:g} deg per channel per step): "
               f"{n_out}/{n_fr} frames outside; orientation envelope > {ENVELOPE_DEG:g} deg "
               f"from frame 0: {env_over}/{len(allm)} episodes."), "", "## Resets", "",
          ("| before | try | kind | ok | reason | s | jitter dx/dy mm, yaw deg | freed (sim) "
           "| err F/U mm |"), "|---|---|---|---|---|---|---|---|---|"]
    for x in chain.resets:
        j, rl = x.get("jitter"), x.get("realised") or {}
        err = "-" if not rl else f"{rl['franka']['err_mm']:.2f}/{rl['ur']['err_mm']:.2f}"
        jit = "-" if not j else f"{j['dx']:+.1f}/{j['dy']:+.1f}, {j['yaw_deg']:+.2f}"
        L.append(f"| {x['before']} | {x['try']} | {x['kind']} | {x.get('ok')} | "
                 f"{x.get('reason') or '-'} | {x['duration_s']:.1f} | {jit} | "
                 f"{x.get('belt_freed_sim')} | {err} |")
    L += ["", "## Deviations", "", *(f"- {d}" for d in DEVIATIONS), "", "## Budget", "",
          "```json", json.dumps(bud, indent=1), "```", ""]
    (out / "seed_report.md").write_text("\n".join(L))


DEVIATIONS = [
    ("demo 'nominal' grasp = set2 gv_00 (zero commanded offsets, collection pool): "
     "pre_place_1_osc is a held-out eval cell"),
    ("excitation starts = the demo starts set2 gv_01 / gv_04 + one seeded set3 variant "
     "(<= 5 start groups so >= 15 of 19 resets can be scripted; plan §4 'same start states')"),
    (f"--excite-cap-factor {DEFAULT_CAP_FACTOR} (v2: 2.0) keeps Franka knot steps inside the "
     "+-6 mm / +-2 deg box; amplitudes are the v2 defaults"),
    ("buffer label_human is a bool (lcs_learning refit_from_buffer tests `is True`); the "
     "emulated-label record is label_human_detail"),
]


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True, help="new dir")
    p.add_argument("--lcm-url", required=True)
    p.add_argument("--cloud-noise", default="real_like")
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--demos", type=int, default=6)
    p.add_argument("--excite", type=int, default=14)
    p.add_argument("--demo-starts", default=None, metavar="SET/VARIANT,...",
                   help="demo start states (default: set2/gv_00,set2/gv_01,set2/gv_04)")
    p.add_argument("--id-prefix", default="",
                   help="buffer ids <prefix><kind>_<i> (default: seed_<i>)")
    p.add_argument("--reset-spec", nargs="*", default=list(DEFAULT_RESET_SPEC),
                   metavar="KEY=VALUE")
    p.add_argument("--hold-s", type=float, default=10.0, help="hold after place_3 (>= 10 s)")
    p.add_argument("--max-episode-s", type=float, default=30.0)
    p.add_argument("--excite-cap-factor", type=float, default=DEFAULT_CAP_FACTOR)
    p.add_argument("--no-record", dest="record", action="store_false",
                   help="skip the viewer recordings")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = create_parser().parse_args()
    col.configure_logging()
    check_private_url(args.lcm_url)
    bc.cloud_noise_spec(args.cloud_noise)
    out = Path(args.out).resolve()
    args.out = out
    spec = sr.parse_reset_spec([*args.reset_spec, f"seed={args.seed}"]
                               if not any(s.startswith("seed=") for s in args.reset_spec)
                               else args.reset_spec)
    demo_starts = parse_starts(args.demo_starts) if args.demo_starts else DEMO_STARTS
    slots = plan_slots(args.demos, args.excite, args.seed, demo_starts)
    check_not_heldout(slots)
    starts = sorted({s.file for s in slots})
    sha_before = {str(f): sha256_file(f) for f in starts}
    groups = len({s.key for s in slots})
    print(f"plan: {len(slots)} episodes in {groups} start groups (max scripted resets "
          f"{len(slots) - groups} of {len(slots) - 1}); reset spec {dataclasses.asdict(spec)}")
    for s in slots:
        print(f"  {s.i:3d}  {s.kind:<10} {s.key:<10} {s.file.relative_to(REPO_ROOT)}")
    if args.dry_run:
        cargs = collect_args(args, slots, out)
        print("excitation:", json.dumps(col.excitation_config(cargs, None)))
        print("collect args:", json.dumps({k: str(v) for k, v in vars(cargs).items()}))
        return 0
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} exists and is not empty; pick a new dir")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": git_info()}
    t0 = time.perf_counter()
    n = col.sample_steps(lcs.SAMPLE_PERIOD_S)
    chain = Chain(args, n, spec, {})
    passes = [{"name": "main", "index": str(out / "index.json")}]
    run_pass(chain, args, slots, out)
    demo_ok = sum(r["slot"].kind == "demo" and r["measures"]["label_human"]["success"]
                  for r in chain.records)
    if demo_ok < MIN_DEMO_SUCCESS:
        logger.warning(f"[SEED] {demo_ok} demo successes < {MIN_DEMO_SUCCESS}: rerunning the "
                       f"demo slots with the fallback offset (UR dz {FALLBACK_UR_DZ_MM} mm, tilt "
                       f"{FALLBACK_UR_TILT_DEG} deg)")
        fb = [dataclasses.replace(s, i=len(slots) + j, fallback=True)
              for j, s in enumerate(x for x in slots if x.kind == "demo")]
        run_pass(chain, args, fb, out / "fallback")
        passes.append({"name": "fallback_demos", "index": str(out / "fallback" / "index.json")})
    wall = time.perf_counter() - t0
    rows = buffer_rows(chain, args.id_prefix)
    bud = budget(chain, wall)
    n_planned = len(slots) + (len(passes) - 1) * args.demos
    # e1's 15 (of 19), capped by the same-start transitions of smaller plans
    acc = acceptance(chain, rows, bud, n_planned, min(15, len(slots) - groups))
    sha_after = {str(f): sha256_file(f) for f in starts}
    acc["sources_untouched"] = {"pass": sha_before == sha_after}
    write_json(out / "buffer.json", {"version": 2, "iter": -1, "max": 50, "n": len(rows),
                                     "source": "seed (e1): UR-held demos + OU excitation",
                                     "rollouts": rows})
    write_json(out / "budget.json", bud)
    seed_meta = {**meta, "args": {k: _jsonable(v) for k, v in vars(args).items()},
                 "reset_spec": dataclasses.asdict(spec), "passes": passes,
                 "plan": [{"i": s.i, "kind": s.kind, "start": s.key,
                           "file": str(s.file.relative_to(REPO_ROOT))} for s in slots],
                 "excitation": chain.excite_cfg.get("excitation"),
                 "resets": _jsonable(chain.resets), "attempts": chain.attempts,
                 "acceptance": acc, "deviations": DEVIATIONS, "start_sha256": sha_after,
                 "box": {"mm": BOX_MM, "deg": BOX_DEG, "envelope_deg": ENVELOPE_DEG}}
    index = json.loads((out / "index.json").read_text())
    index["seed"] = seed_meta
    write_json(out / "index.json", _jsonable(index))
    write_report(out, args, chain, rows, bud, acc, meta)
    logger.info("[SEED] acceptance " + ", ".join(f"{k} {'PASS' if v['pass'] else 'FAIL'}"
                                                 for k, v in acc.items()) + f" -> {out}")
    if demo_ok < MIN_DEMO_SUCCESS and sum(r["kind"] == "demo" and r["label_human"]
                                          for r in rows) < MIN_DEMO_SUCCESS:
        logger.error("[SEED] STOP: fewer than 2 demo successes after the fallback")
        return 2
    return 0 if all(v["pass"] for v in acc.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
