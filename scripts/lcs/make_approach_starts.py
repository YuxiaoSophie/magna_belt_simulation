#!/usr/bin/env python3
"""Build approach start states: ``pre_place_1`` moved rigidly through an approach transform.

Each setting ``(yaw_deg, elev_mm, offset_mm, tilt_deg)`` restores ``--reference`` under magna's
OSC (``--lcm-url``) and moves BOTH arms together (min-jerk ``--move-s``) through the collector's
approach transform (``collect_lcs_dataset.approach_pose``: yaw about the large pulley axis,
offset along the rotated horizontal normal, z + elev, tilt about the rotated tangent). The
transform parameters are interpolated (not the positions), so the grasps stay rigid. Then it
holds ``--hold-s``, settles ``--settle-s`` under the hold-at-measured hook, checks the grasp every
10 steps and saves ``<out>/<id>_osc.npz`` + ``index.json`` (grasp-variant format + ``approach``),
then re-checks as the eval starts (restore, 0.5 s settle).

Run:
    uv run --frozen python scripts/lcs/make_approach_starts.py \\
        --lcm-url 'udpm://239.255.76.137:7737?ttl=0' --out data/lcs/start_states/approach \\
        --settings 'a01=20,15,0,0;a02=-20,0,10,8'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import collect_lcs_dataset as col
import make_ee_offset_starts as eo
import make_grasp_variants as gv
from loguru import logger

from round_belt_task.arm_kinematics import UrTracking, ik, rotvec
from round_belt_task.commander import mat3_to_quat, saved_traj_message
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import sim_snapshot

DEFAULT_OUT = sim_snapshot.DEFAULT_START_STATE_DIR / "approach"


def parse_settings(text: str) -> list[tuple[str, dict]]:
    """``"id=yaw,elev,offset,tilt;..."`` -> ``[(id, approach dict)]``."""
    out = []
    for item in (t.strip() for t in text.split(";") if t.strip()):
        vid, sep, vals = item.partition("=")
        if not sep:
            raise ValueError(f"--settings item {item!r}: want id=yaw,elev,offset,tilt")
        out.append((vid.strip(), dict(zip(col.APPROACH_KEYS,
                                          col._floats(vals, 4, "--settings"), strict=True))))
    return out


def scaled(a: dict, s: float) -> dict:
    return {k: a[k] * s for k in col.APPROACH_KEYS}


def pose_at(X0: np.ndarray, a: dict, s: float, pulley: dict, ref) -> np.ndarray:
    """``X0`` through ``s`` x the approach (the path keeps the grasps rigid)."""
    b = scaled(a, s)
    fr = col.approach_frame(b, pulley, ref.franka_pos, ref.ur_pos)
    return col.approach_pose(X0, b, fr, elev=True)


def horizontal_yaw_deg(v0: np.ndarray, v1: np.ndarray) -> float:
    return math.degrees(math.atan2(v0[0] * v1[1] - v0[1] * v1[0], v0[0] * v1[0] + v0[1] * v1[1]))


def achieved(X_f, X_u, X_f0, X_u0, X_f1, X_u1, fr) -> dict:
    """Achieved vs commanded: yaw of the Franka -> UR vector, per-arm offset / z / orientation."""
    v0, v1 = (X_u0[:3, 3] - X_f0[:3, 3]), (X_u[:3, 3] - X_f[:3, 3])
    out = {"yaw_deg": horizontal_yaw_deg(v0, v1)}
    for arm, X, X0, X1 in (("franka", X_f, X_f0, X_f1), ("ur", X_u, X_u0, X_u1)):
        d = X[:3, 3] - X1[:3, 3]
        out[arm] = {"pos_err_mm": float(np.linalg.norm(d) * 1e3),
                    "normal_err_mm": float(d @ fr["normal"] * 1e3),
                    "tangent_err_mm": float(d @ fr["tangent"] * 1e3),
                    "z_err_mm": float(d[2] * 1e3),
                    "rot_err_deg": float(np.degrees(np.linalg.norm(
                        rotvec(X[:3, :3] @ X1[:3, :3].T)))),
                    "moved_mm": float(np.linalg.norm(X[:3, 3] - X0[:3, 3]) * 1e3)}
    return out


def build_one(sim, snap, vid: str, a: dict, pulley: dict, ref, ref_belt, ref_meas, args) -> dict:
    dt = sim.frame_dt
    sim.restore(snap, settle_steps=0)
    X_f0 = sim.ee_poses()[0]
    q0 = sim.arm_targets()[1]
    X_u0 = UrTracking.fk(q0)  # commanded tracking pose: the UR is position-driven
    X_f1 = pose_at(X_f0, a, 1.0, pulley, ref)
    X_u1 = pose_at(X_u0, a, 1.0, pulley, ref)
    fr = col.approach_frame(a, pulley, ref.franka_pos, ref.ur_pos)
    row = {"id": vid, "approach": dict(a), "held": False, "reason": None, "file": None,
           "commanded": {"approach": dict(a), "franka_xyz": X_f1[:3, 3].tolist(),
                         "ur_xyz": X_u1[:3, 3].tolist()}}
    move_steps, hold_steps = round(args.move_s / dt), round(args.hold_s / dt)
    start: list[float] = []
    knot_dt = 0.075  # magna's 2-knot spacing

    def s_at(t: float) -> float:
        return eo.min_jerk((t - start[0]) / args.move_s)

    def franka_hook(step, t, joint_q, body_q):
        if not start:
            start.append(t)
        Xa, Xb = (pose_at(X_f0, a, s_at(tk), pulley, ref) for tk in (t, t + knot_dt))
        return saved_traj_message(round(t * 1e6), np.stack([Xa[:3, 3], Xb[:3, 3]]),
                                  np.stack([mat3_to_quat(Xa[:3, :3]), mat3_to_quat(Xb[:3, :3])]),
                                  np.array([t, t + knot_dt]))

    q_prev = [q0]

    def on_step(k):
        X = pose_at(X_u0, a, eo.min_jerk((k + 1) / move_steps), pulley, ref)
        q, ep, er, _ = ik(UrTracking, X, q_prev[0], pos_tol=1e-6, rot_tol=1e-5)
        if ep > 1e-5 or er > 1e-4:
            raise RuntimeError(f"UR IK err {ep:.2e} m {er:.2e} rad")
        q_prev[0] = q
        sim.set_ur_target(q)

    sim.commander_hook = franka_hook
    try:
        ok, why, _ = eo.run_steps(sim, move_steps, on_step)
    except RuntimeError as exc:
        ok, why = False, str(exc)
    if ok:
        ok, why, _ = eo.run_steps(sim, hold_steps)
    grasp = None
    if ok:
        sim.commander_hook = sim.make_hold_hook()  # the eval's hold: latch the measured pose
        ok, why, grasp = eo.run_steps(sim, round(args.settle_s / dt))
    bad = [e for e in gv.OSC_LOG_ERRORS if e in sim.osc.log_text()]
    if bad:
        raise RuntimeError(f"OSC log has {bad}")
    X_f, X_u = sim.ee_poses()
    row["achieved"] = achieved(X_f, X_u, X_f0, X_u0, X_f1, X_u1, fr)
    belt = sim.belt_positions()
    row["belt_rmse_vs_nominal_mm"] = eo.belt_rmse_mm(sim, ref_belt)
    moved = np.stack([pose_at(_trans(p), a, 1.0, pulley, ref)[:3, 3] for p in ref_belt])
    row["belt_rmse_vs_transformed_mm"] = float(
        np.sqrt(((belt - moved) ** 2).sum(axis=1).mean()) * 1e3)
    if not ok:
        row["reason"] = why
        return row
    row["held"] = True
    row["measured"] = gv.measured_record(sim, sim.state_0.body_q.numpy(), ref_meas, grasp)
    ac = row["achieved"]
    row["notes"] = (f"approach {vid}: yaw {a['yaw_deg']:g} deg, elev {a['elev_mm']:g} mm, offset "
                    f"{a['offset_mm']:g} mm, tilt {a['tilt_deg']:g} deg from "
                    f"{args.reference.name}; both arms min-jerk {args.move_s:g} s, hold "
                    f"{args.hold_s:g} s, settle {args.settle_s:g} s under magna OSC hold; "
                    f"achieved yaw {ac['yaw_deg']:.2f} deg; {grasp.describe()}")
    path = sim_snapshot.save(sim_snapshot.capture(sim, gv.PICK_LAST, notes=row["notes"]),
                             args.out / f"{vid}_osc.npz")
    row["file"] = path.name
    grasp2 = sim.restore(sim_snapshot.load(path), settle_steps=round(0.5 / dt))
    X_f2, X_u2 = sim.ee_poses()
    row["recheck_0p5s"] = {"held": list(grasp2.held()),
                           "achieved": achieved(X_f2, X_u2, X_f0, X_u0, X_f1, X_u1, fr),
                           "belt_rmse_vs_nominal_mm": eo.belt_rmse_mm(sim, ref_belt)}
    return row


def _trans(p) -> np.ndarray:
    X = np.eye(4)
    X[:3, 3] = p
    return X


def summary(row: dict) -> str:
    ac = row.get("achieved")
    if ac is None:
        return row.get("reason") or "-"
    return (f"yaw {ac['yaw_deg']:+6.2f} deg, err F/U normal {ac['franka']['normal_err_mm']:+5.2f}"
            f"/{ac['ur']['normal_err_mm']:+5.2f} mm z {ac['franka']['z_err_mm']:+5.2f}/"
            f"{ac['ur']['z_err_mm']:+5.2f} mm rot {ac['franka']['rot_err_deg']:.2f}/"
            f"{ac['ur']['rot_err_deg']:.2f} deg, belt rmse vs transformed "
            f"{row['belt_rmse_vs_transformed_mm']:.2f} mm"
            + ("" if row["held"] else f" | {row['reason']}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--settings", required=True, help="id=yaw,elev,offset,tilt;...")
    parser.add_argument("--reference", type=Path, default=gv.DEFAULT_REFERENCE)
    parser.add_argument("--lcm-url", required=True, help="private LCM URL for the OSC")
    parser.add_argument("--move-s", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML)
    args = parser.parse_args()
    gv.configure_logging()
    col.check_private_url(args.lcm_url)
    settings = parse_settings(args.settings)
    args.out.mkdir(parents=True, exist_ok=True)
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    nominal = load_pre_mpc_segment(args.params, first=col.FIRST, last=col.LAST)
    ref = next(w for w in nominal if w.label == "pre_place_2")
    snap = sim_snapshot.load(args.reference)
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url)
    rows, reason, t0 = [], "error", time.perf_counter()
    try:
        ref_meas = gv.measure(sim, snap.body_q)
        ref_belt = np.asarray(snap.body_q)[sim.info.belt_bodies, :3].astype(np.float64)
        warm = sim.start_osc(args.out / "osc.log")
        logger.info(f"OSC warm-up {warm:.2f} s")
        sim.restore(snap, settle_steps=0)
        pulley = col.pulley_frame(sim)
        for vid, a in settings:
            row = build_one(sim, snap, vid, a, pulley, ref, ref_belt, ref_meas, args)
            logger.info(f"{vid} {a}: held {row['held']}: {summary(row)}")
            rows.append(row)
        reason = "finished"
    finally:
        sim.close(reason)
    index = {"set": args.out.name, "kind": "approach",
             "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "nominal_start": str(args.reference),
             "nominal_start_sha256": gv.sha256(args.reference),
             "move_s": args.move_s, "hold_s": args.hold_s, "settle_s": args.settle_s,
             "pulley_large": {k: v.tolist() for k, v in pulley.items()},
             "transform": "collect_lcs_dataset.approach_pose (elev on); tangent / normal from "
                          "the nominal pre_place_2 rotated by yaw",
             "achieved_metric": "measured EE poses (Franka finger_tip / UR tracking) vs the "
                                "commanded transformed poses; yaw = horizontal Franka -> UR "
                                "vector rotation",
             "wall_s": time.perf_counter() - t0, "variants": rows}
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    for r in rows:
        print(f"{r['id']:<6} {json.dumps(r['approach'])} held {r['held']!s:<5} {summary(r)}")
    return 0 if all(r["held"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
