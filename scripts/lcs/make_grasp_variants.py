#!/usr/bin/env python3
"""Build grasp-varied ``pre_place_1`` start states: each gripper holds the belt elsewhere.

A variant ``(franka_slide_mm, franka_roll_deg, ur_slide_mm, ur_roll_deg)`` moves each gripper's
``pick`` pose rigidly: slide = along the rest belt's tangent at the rest body nearest the nominal
grasp point (``finger_tip`` / the UR tracking origin = the 2F-85 tip point), roll = about that
tangent through the grasp point. The UR's ``pre_pick_1`` gets the same UR offset; ``post_pick``
and ``pre_place_1`` stay nominal, so every variant ends at the nominal ``pre_place_1`` poses.

Phase A (position backend, one sim): restore the fresh state, plan + play the modified pick,
require a two-handed hold. Phase B (magna's OSC on ``--lcm-url``, one sim): restore, settle 2 s
under the OSC hold, require ``held() == (True, True)``, measure the grasp offsets against
``pre_place_1_osc.npz`` and save ``<root>/<set>/gv_<id>_osc.npz`` + ``index.json``.

``--probe`` runs single-knob sweeps instead and writes only ``<root>/<set>/probe.json``; its
feasible box is the default ``--box`` of a later set run.

Run:
    uv run python scripts/lcs/make_grasp_variants.py --probe
    uv run python scripts/lcs/make_grasp_variants.py --set set1 --count 12 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from loguru import logger

from round_belt_task.arm_kinematics import (
    mat3_to_quat_xyzw,
    model_pose_franka_tip,
    model_pose_ur_tracking,
    quat_xyzw_to_mat3,
)
from round_belt_task.offline_simulation import (
    PICK_FIRST,
    PICK_HOLD_S,
    PICK_LAST,
    OfflineBridge,
    RoundBeltOfflineSimulation,
)
from round_belt_task.waypoints import (
    MAGNA_PARAMS_SIM_YAML,
    Waypoint,
    load_pre_mpc_segment,
)
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot

DEFAULT_URL = "udpm://239.255.76.92:7692?ttl=0"
DEFAULT_ROOT = sim_snapshot.DEFAULT_START_STATE_DIR / "grasp_variants"
DEFAULT_REFERENCE = sim_snapshot.DEFAULT_START_STATE_DIR / f"{PICK_LAST}_osc.npz"
OSC_SETTLE_S = 2.0
OSC_LOG_ERRORS = ("resetting", "Exception caught")
KNOBS = ("franka_slide_mm", "franka_roll_deg", "ur_slide_mm", "ur_roll_deg")
PROBE_MAGNITUDES = {"franka_slide_mm": (5, 10, 15, 20), "franka_roll_deg": (5, 10, 15),
                    "ur_slide_mm": (5, 10, 15, 20), "ur_roll_deg": (5, 10, 15)}


@dataclass
class Variant:
    id: str
    franka_slide_mm: float = 0.0
    franka_roll_deg: float = 0.0
    ur_slide_mm: float = 0.0
    ur_roll_deg: float = 0.0

    def commanded(self) -> dict:
        return {k: float(getattr(self, k)) for k in KNOBS}


def configure_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stdout, level=level,
               format="<level>{level: <7}</level> | <level>{message}</level>")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- geometry -------------------------------------------------------------------------------------


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def grasp_offset(point: np.ndarray, tangent: np.ndarray, slide_mm: float,
                 roll_deg: float) -> np.ndarray:
    """World 4x4: roll about ``tangent`` through ``point``, then slide along it."""
    R = rotation_about(tangent, math.radians(roll_deg))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = point + slide_mm * 1e-3 * tangent - R @ point
    return T


def rest_tangent(rest: np.ndarray, point: np.ndarray) -> tuple[int, np.ndarray]:
    """(nearest rest body, unit tangent there in increasing-index direction)."""
    i = int(np.argmin(np.linalg.norm(rest - point, axis=1)))
    n = len(rest)
    t = rest[(i + 1) % n] - rest[(i - 1) % n]
    return i, t / np.linalg.norm(t)


def apply_pose(T: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray, mat: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    X = T @ mat
    return X[:3, 3].copy(), mat3_to_quat_xyzw(X[:3, :3])


def variant_waypoints(nominal: list[Waypoint], frames: dict, v: Variant) -> list[Waypoint]:
    """``nominal`` with the variant's offsets on ``pick`` (both arms) and ``pre_pick_1`` (UR)."""
    T_f = grasp_offset(frames["franka"]["point"], frames["franka"]["tangent"],
                       v.franka_slide_mm, v.franka_roll_deg)
    T_u = grasp_offset(frames["ur"]["point"], frames["ur"]["tangent"],
                       v.ur_slide_mm, v.ur_roll_deg)
    out = copy.deepcopy(nominal)
    for w in out:
        if w.label == "pick":
            w.franka_pos, w.franka_quat_xyzw = apply_pose(T_f, w.franka_pos, w.franka_quat_xyzw,
                                                          w.franka_mat())
        if w.label in ("pre_pick_1", "pick"):
            w.ur_pos, w.ur_quat_xyzw = apply_pose(T_u, w.ur_pos, w.ur_quat_xyzw, w.ur_mat())
    return out


# --- measurement ----------------------------------------------------------------------------------


def _rest_arc() -> tuple[np.ndarray, np.ndarray, float]:
    rest = lcs.rest_belt_bodies()
    seg = np.linalg.norm(np.diff(np.vstack([rest, rest[:1]]), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])[:-1], seg, float(seg.sum())


REST_CUM, REST_SEG, REST_PERIMETER = _rest_arc()


def material_coord(belt: np.ndarray, point: np.ndarray) -> float:
    """``i + f`` (body units) of the point's nearest projection on the closed body loop."""
    a = belt
    b = np.roll(belt, -1, axis=0)
    ab = b - a
    f = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    d = np.linalg.norm(a + f[:, None] * ab - point, axis=1)
    i = int(np.argmin(d))
    return i + float(f[i])


def material_arc_mm(s: float) -> float:
    i = math.floor(s) % len(REST_CUM)
    return float(REST_CUM[i] + (s - math.floor(s)) * REST_SEG[i]) * 1e3


def wrap(x: float, period: float) -> float:
    return (x + period / 2) % period - period / 2


def twist_deg(R: np.ndarray, axis: np.ndarray) -> float:
    """Twist angle of rotation ``R`` about unit ``axis`` (swing-twist split)."""
    w = math.sqrt(max(0.0, 1.0 + float(np.trace(R)))) / 2.0
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 4.0
    return math.degrees(2.0 * math.atan2(float(v @ axis), w))


def measure(sim, body_q: np.ndarray) -> dict:
    """Per gripper: nearest belt body, material coordinate/arc and the gripper's rotation in
    that body's frame (body z = belt tangent); belt z range."""
    body_q = np.asarray(body_q, dtype=np.float64)
    belt_ids = sim.info.belt_bodies
    belt = body_q[belt_ids, :3]
    poses = {"franka": model_pose_franka_tip(body_q, sim._finger_tip_body),
             "ur": model_pose_ur_tracking(body_q, sim._ur_wrist_body)}
    out = {"belt_min_z": float(belt[:, 2].min()), "belt_max_z": float(belt[:, 2].max())}
    for name, X in poses.items():
        nearest = int(np.argmin(np.linalg.norm(belt - X[:3, 3], axis=1)))
        s = material_coord(belt, X[:3, 3])
        R_body = quat_xyzw_to_mat3(body_q[belt_ids[nearest], 3:7])
        out[name] = {"nearest_body": nearest, "material_s": s, "arc_mm": material_arc_mm(s),
                     "gripper_in_body": (R_body.T @ X[:3, :3]).tolist()}
    return out


def offsets(m: dict, ref: dict) -> dict:
    """``m`` minus the reference: material offsets wrapped around the loop; roll = the gripper's
    extra rotation about the belt tangent relative to the grasped belt body."""
    n = len(REST_CUM)
    out = {}
    for name in ("franka", "ur"):
        a, b = m[name], ref[name]
        R = np.asarray(b["gripper_in_body"]) @ np.asarray(a["gripper_in_body"]).T
        out[name] = {"body_offset": int(wrap(a["nearest_body"] - b["nearest_body"], n)),
                     "slide_mm": wrap(a["arc_mm"] - b["arc_mm"], REST_PERIMETER * 1e3),
                     "roll_deg": twist_deg(R.T, np.array([0.0, 0.0, 1.0]))}
    return out


def measured_record(sim, body_q, ref: dict, grasp) -> dict:
    m = measure(sim, body_q)
    rec = {"offsets": offsets(m, ref), "absolute": m, "grasp": grasp.describe(),
           "franka_width_mm": grasp.franka_width_mm, "franka_tip_gap_mm": grasp.franka_tip_gap_mm,
           "ur_status": grasp.ur_status, "ur_tip_gap_mm": grasp.ur_tip_gap_mm,
           "ur_pad_gaps_mm": list(grasp.ur_pad_gaps_mm),
           "belt_min_z": m["belt_min_z"], "belt_max_z": m["belt_max_z"]}
    return rec


def short(rec: dict | None) -> str:
    if rec is None:
        return "-"
    o = rec["offsets"]
    return (f"franka {o['franka']['slide_mm']:+6.1f} mm {o['franka']['roll_deg']:+5.1f} deg, "
            f"ur {o['ur']['slide_mm']:+6.1f} mm {o['ur']['roll_deg']:+5.1f} deg")


# --- phases ---------------------------------------------------------------------------------------


def phase_a(variants: list[Variant], params: Path) -> tuple[dict, dict]:
    """Position-backend picks: ``({id: snapshot}, {id: failure reason})``."""
    sim = RoundBeltOfflineSimulation.build()
    snaps, failed = {}, {}
    try:
        initial = sim_snapshot.capture(sim, "initial")
        rest = sim.belt_positions()
        nominal = load_pre_mpc_segment(params, first=PICK_FIRST, last=PICK_LAST)
        pick = next(w for w in nominal if w.label == "pick")
        frames = {}
        for name, point in (("franka", pick.franka_pos), ("ur", pick.ur_pos)):
            i, t = rest_tangent(rest, np.asarray(point, float))
            frames[name] = {"point": np.asarray(point, float), "tangent": t, "rest_body": i}
            logger.info(f"[A] {name} grasp frame: rest body {i}, tangent {np.round(t, 4)}")
        for k, v in enumerate(variants):
            if k:
                sim.restore(initial)
                sim.bridge = OfflineBridge()  # a fresh sim has no gripper command yet
            t0 = time.perf_counter()
            traj = sim.plan(variant_waypoints(nominal, frames, v), settle_s=PICK_HOLD_S)
            sim.play(traj, log_every_s=0)
            grasp = sim.grasp_state()
            fails = grasp.failures()
            took = time.perf_counter() - t0
            if fails:
                failed[v.id] = "position: " + "; ".join(fails)
                logger.warning(f"[A] {v.id} {v.commanded()} FAILED in {took:.0f} s: "
                               f"{failed[v.id]}")
                continue
            snaps[v.id] = sim_snapshot.capture(sim, PICK_LAST, notes=(
                f"grasp variant {v.id} {json.dumps(v.commanded())}; position pick "
                f"{PICK_FIRST}->{PICK_LAST}; {grasp.describe()}"))
            logger.info(f"[A] {v.id} held in {took:.0f} s: {grasp.describe()}")
    finally:
        sim.close("finished")
    return snaps, failed


def phase_b(variants: list[Variant], snaps: dict, failed: dict, lcm_url: str,
            reference: Path, out_dir: Path | None) -> dict:
    """OSC settle + measure; saves ``gv_<id>_osc.npz`` if ``out_dir``. ``{id: row}``."""
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    rows = {v.id: {"id": v.id, "commanded": v.commanded(), "held": False,
                   "reason": failed.get(v.id), "measured": None, "file": None, "notes": None}
            for v in variants}
    if not snaps:
        return rows
    log_path = Path(tempfile.mkdtemp(prefix="make_grasp_variants_")) / "osc.log"
    sim = RoundBeltOscSimulation.build(lcm_url=lcm_url)
    reason = "error"
    try:
        ref = measure(sim, sim_snapshot.load(reference).body_q)
        warm = sim.start_osc(log_path)
        logger.info(f"[B] OSC {sim.osc.describe()} warm-up {warm:.2f} s, log {log_path}")
        steps = round(OSC_SETTLE_S / sim.frame_dt)
        for v in variants:
            if v.id not in snaps:
                continue
            row = rows[v.id]
            sim.restore(snaps[v.id], settle_steps=0)
            grasp = sim.settle(steps)
            fails = grasp.failures()
            if not all(grasp.held()):
                fails.append(f"held {grasp.held()}")
            bad = [e for e in OSC_LOG_ERRORS if e in sim.osc.log_text()]
            if bad:
                raise RuntimeError(f"OSC log has {bad} ({log_path})")
            row["measured"] = measured_record(sim, sim.state_0.body_q.numpy(), ref, grasp)
            if fails:
                row["reason"] = "osc: " + "; ".join(fails)
                logger.warning(f"[B] {v.id} FAILED: {row['reason']}")
                continue
            row["held"] = True
            offset = sim.bridge.utime_offset_us
            row["notes"] = (
                f"grasp variant {v.id} commanded {json.dumps(v.commanded())}; {PICK_LAST} settled "
                f"{OSC_SETTLE_S:g} s under magna OSC hold; utime offset {offset}; measured "
                f"{json.dumps(row['measured']['offsets'])}; {grasp.describe()}")
            if out_dir is not None:
                snap = sim_snapshot.capture(sim, PICK_LAST, notes=row["notes"])
                path = sim_snapshot.save(snap, out_dir / f"{v.id}_osc.npz")
                row["file"] = path.name
            logger.info(f"[B] {v.id} held: {short(row['measured'])}")
        reason = "finished"
    finally:
        sim.close(reason)
    return rows


# --- modes ----------------------------------------------------------------------------------------


def probe_variants() -> list[Variant]:
    out = [Variant("nominal")]
    for knob, mags in PROBE_MAGNITUDES.items():
        for m in mags:
            for sign in (1, -1):
                out.append(Variant(f"{knob}{sign * m:+d}", **{knob: float(sign * m)}))
    return out


def feasible_box(rows: list[dict]) -> dict:
    """Per knob ``[lo, hi]``: the largest magnitude each way with every smaller one held."""
    box = {}
    for knob, mags in PROBE_MAGNITUDES.items():
        held = {r["commanded"][knob]: r["held"] for r in rows
                if r["id"].startswith(knob)}
        edge = []
        for sign in (-1, 1):
            best = 0.0
            for m in sorted(mags):
                if not held.get(float(sign * m), False):
                    break
                best = float(sign * m)
            edge.append(best)
        box[knob] = edge
    return box


def parse_box(text: str) -> dict:
    parts = text.split()
    if len(parts) != len(KNOBS):
        raise ValueError(f"--box needs {len(KNOBS)} entries ({' '.join(KNOBS)}), got {text!r}")
    box = {}
    for knob, p in zip(KNOBS, parts, strict=True):
        lo, hi = (float(x) for x in p.split(":")) if ":" in p else (-abs(float(p)), abs(float(p)))
        if lo > hi:
            raise ValueError(f"--box {knob}: {lo} > {hi}")
        box[knob] = [lo, hi]
    return box


def draw_variants(count: int, seed: int, box: dict) -> list[Variant]:
    out = [Variant("gv_00")]
    for k in range(1, count + 1):
        rng = np.random.default_rng([seed, k])
        out.append(Variant(f"gv_{k:02d}", **{knob: round(float(rng.uniform(*box[knob])), 2)
                                             for knob in KNOBS}))
    return out


def print_table(rows: list[dict]) -> None:
    print(f"{'id':<22} {'held':<5} measured offsets / reason")
    for r in rows:
        tail = short(r["measured"]) if r["held"] else (r["reason"] or "")
        if not r["held"] and r["measured"] is not None:
            tail = f"{short(r['measured'])} | {r['reason']}"
        print(f"{r['id']:<22} {r['held']!s:<5} {tail}")


def run_probe(args: argparse.Namespace, set_dir: Path) -> int:
    variants = probe_variants()
    snaps, failed = phase_a(variants, args.params)
    rows = list(phase_b(variants, snaps, failed, args.lcm_url, args.reference, None).values())
    box = feasible_box(rows)
    print_table(rows)
    print(f"feasible box: {json.dumps(box)}")
    set_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "set": args.set,
               "params": str(args.params), "params_sha256": sha256(args.params),
               "reference": str(args.reference), "reference_sha256": sha256(args.reference),
               "magnitudes": PROBE_MAGNITUDES, "box": box, "rows": rows}
    path = set_dir / "probe.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.success(f"[PROBE] wrote {path}")
    return 0


def run_set(args: argparse.Namespace, set_dir: Path) -> int:
    probe_path = set_dir / "probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    if args.box is not None:
        box = parse_box(args.box)
    elif probe is not None:
        box = probe["box"]
    else:
        print(f"no --box and no {probe_path}: run --probe first or pass --box")
        return 2
    variants = draw_variants(args.count, args.seed, box)
    logger.info(f"[SET] {args.set}: {len(variants)} variants, box {json.dumps(box)}")
    set_dir.mkdir(parents=True, exist_ok=True)
    snaps, failed = phase_a(variants, args.params)
    rows = list(phase_b(variants, snaps, failed, args.lcm_url, args.reference, set_dir).values())
    index = {"set": args.set, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "seed": args.seed, "count": args.count, "box": box,
             "box_source": "--box" if args.box is not None else str(probe_path),
             "params": str(args.params), "params_sha256": sha256(args.params),
             "nominal_start": str(args.reference),
             "nominal_start_sha256": sha256(args.reference),
             "roll_metric": "twist about the belt tangent (body z) of the gripper rotation in the "
                            "nearest belt body frame, minus the reference",
             "ur_grasp_point": "UrTracking origin (== the 2F-85 tip point of grasp_state)",
             "variants": rows,
             "probe": None if probe is None else {
                 "file": probe_path.name, "box": probe["box"],
                 "rows": [{"id": r["id"], "held": r["held"], "reason": r["reason"],
                           "offsets": None if r["measured"] is None
                           else r["measured"]["offsets"]} for r in probe["rows"]]}}
    (set_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print_table(rows)
    held = sum(r["held"] for r in rows)
    logger.success(f"[SET] {set_dir}: {held}/{len(rows)} held")
    return 0 if rows[0]["held"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="store_true", help="single-knob feasibility sweep")
    parser.add_argument("--set", default="set1")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="parent of the set dir")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--box", default=None,
                        help="'f_slide f_roll u_slide u_roll' half-widths or lo:hi (default: "
                             "<set>/probe.json)")
    parser.add_argument("--lcm-url", default=DEFAULT_URL,
                        help="private LCM URL for the OSC (never magna's shared group)")
    parser.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help="nominal start state the offsets are measured against")
    args = parser.parse_args()
    configure_logging()
    set_dir = args.root / args.set
    return run_probe(args, set_dir) if args.probe else run_set(args, set_dir)


if __name__ == "__main__":
    sys.exit(main())
