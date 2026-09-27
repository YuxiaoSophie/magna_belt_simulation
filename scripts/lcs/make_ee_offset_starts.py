#!/usr/bin/env python3
"""Build EE-offset ``pre_place_1`` start states: one arm moved a few mm / deg from nominal.

Each variant restores ``--reference`` under magna's OSC (``--lcm-url``), moves one arm smoothly
(min-jerk, ``--move-s``) to the offset pose, holds that target ``--hold-s``, then settles
``--settle-s`` under the standard hold-at-measured hook (as the grasp variants' phase B) and
saves ``<out>/<id>_osc.npz`` + ``index.json``. The Franka is moved via the OSC's Cartesian
target; the UR via its joint position target (IK of the offset tracking pose). The grasp must
hold at every check during the move and settle; otherwise the next magnitude in ``VARIANTS`` is
tried.

Run:
    uv run python scripts/lcs/make_ee_offset_starts.py --lcm-url 'udpm://239.255.76.118:7718?ttl=0'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import make_grasp_variants as gv
from loguru import logger

from round_belt_task.arm_kinematics import (
    UrTracking,
    ik,
    rot_axis_angle,
    rotvec,
)
from round_belt_task.commander import (
    mat3_to_quat,
    saved_traj_message,
    slerp,
)
from task_common import sim_snapshot

DEFAULT_OUT = sim_snapshot.DEFAULT_START_STATE_DIR / "ee_offsets"
CHECK_EVERY = 10  # steps between grasp checks during the move/settle

# id: (arm, kind, world axis, magnitudes tried in order; mm or deg); rot_tool_z = about the tip z.
VARIANTS = {
    "ee_f_x+8": ("franka", "trans", (1.0, 0.0, 0.0), (8.0, 5.0)),
    "ee_f_z+8": ("franka", "trans", (0.0, 0.0, 1.0), (8.0, 5.0)),
    "ee_ur_y-8": ("ur", "trans", (0.0, -1.0, 0.0), (8.0, 5.0)),
    "ee_f_rot5": ("franka", "rot_tool_z", None, (5.0, 3.0)),
}


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def target_pose(X0: np.ndarray, kind: str, axis, mag: float) -> np.ndarray:
    X = X0.copy()
    if kind == "trans":
        X[:3, 3] += np.asarray(axis, float) * mag * 1e-3
    else:
        X[:3, :3] = X0[:3, :3] @ rot_axis_angle((0.0, 0.0, 1.0), math.radians(mag))
    return X


def pose_delta(X: np.ndarray, X0: np.ndarray) -> dict:
    """World position delta [mm] and rotation delta (world rotvec + about the tool z) [deg]."""
    R = X0[:3, :3].T @ X[:3, :3]
    return {"dpos_mm": ((X[:3, 3] - X0[:3, 3]) * 1e3).tolist(),
            "dpos_norm_mm": float(np.linalg.norm(X[:3, 3] - X0[:3, 3]) * 1e3),
            "drot_world_deg": np.degrees(rotvec(X[:3, :3] @ X0[:3, :3].T)).tolist(),
            "drot_angle_deg": float(np.degrees(np.linalg.norm(rotvec(R)))),
            "drot_tool_z_deg": gv.twist_deg(R, np.array([0.0, 0.0, 1.0]))}


def franka_move_hook(sim, X0: np.ndarray, X1: np.ndarray, move_s: float):
    """Min-jerk Cartesian 2-knot target from X0 to X1 (starting at the first call), then hold."""
    dt = sim.frame_dt
    p0, p1 = X0[:3, 3], X1[:3, 3]
    q0, q1 = mat3_to_quat(X0[:3, :3]), mat3_to_quat(X1[:3, :3])
    start: list[float] = []

    def at(t):
        s = min_jerk((t - start[0]) / move_s)
        return p0 + s * (p1 - p0), slerp(q0, q1, s)

    def hook(step, t, joint_q, body_q):
        if not start:
            start.append(t)
        pa, qa = at(t)
        pb, qb = at(t + dt)
        return saved_traj_message(round(t * 1e6), np.stack([pa, pb]), np.stack([qa, qb]),
                                  np.array([t, t + dt]))

    return hook


def run_steps(sim, steps: int, on_step=None) -> tuple[bool, str | None, object]:
    """Step ``steps`` control steps; ``(held throughout, first failure, last grasp)``."""
    grasp = None
    for k in range(steps):
        if on_step is not None:
            on_step(k)
        sim.control_step()
        if (k + 1) % CHECK_EVERY == 0 or k + 1 == steps:
            grasp = sim.grasp_state()
            if not all(grasp.held()):
                return False, f"step {k + 1}: held {grasp.held()} ({grasp.describe()})", grasp
    return True, None, grasp


def belt_rmse_mm(sim, ref_belt: np.ndarray) -> float:
    d = sim.belt_positions() - ref_belt
    return float(np.sqrt((d * d).sum(axis=1).mean()) * 1e3)


def build_one(sim, snap, ref_poses, ref_belt, ref_meas, vid, spec, mag, args) -> dict:
    arm, kind, axis, _ = spec
    dt = sim.frame_dt
    sim.restore(snap, settle_steps=0)
    X_f0 = sim.ee_poses()[0]
    row = {"id": vid, "arm": arm, "kind": kind, "axis": axis, "magnitude": mag,
           "commanded": {"arm": arm, "kind": kind, "axis": axis, "magnitude": mag,
                         "unit": "mm" if kind == "trans" else "deg"},
           "held": False, "reason": None, "file": None}
    move_steps, hold_steps = round(args.move_s / dt), round(args.hold_s / dt)
    on_step = None
    if arm == "franka":
        sim.commander_hook = franka_move_hook(sim, X_f0, target_pose(X_f0, kind, axis, mag),
                                              args.move_s)
    else:
        # Offset the commanded tracking pose (not the measured one): the UR is position-driven.
        q_start = sim.arm_targets()[1]
        Xc0 = UrTracking.fk(q_start)
        Xc1 = target_pose(Xc0, kind, axis, mag)
        q_prev = [q_start]

        def on_step(k):
            s = min_jerk((k + 1) / move_steps)
            Xs = Xc0.copy()
            Xs[:3, 3] = Xc0[:3, 3] + s * (Xc1[:3, 3] - Xc0[:3, 3])
            q, ep, er, _ = ik(UrTracking, Xs, q_prev[0], pos_tol=1e-6, rot_tol=1e-5)
            if ep > 1e-5 or er > 1e-4:
                raise RuntimeError(f"UR IK err {ep:.2e} m {er:.2e} rad")
            q_prev[0] = q
            sim.set_ur_target(q)
    ok, why, _ = run_steps(sim, move_steps, on_step)
    if ok:
        ok, why, _ = run_steps(sim, hold_steps)
    if ok:
        sim.commander_hook = sim.make_hold_hook()  # the eval's hold: latch the measured pose
        ok, why, grasp = run_steps(sim, round(args.settle_s / dt))
    bad = [e for e in gv.OSC_LOG_ERRORS if e in sim.osc.log_text()]
    if bad:
        raise RuntimeError(f"OSC log has {bad}")
    X_f, X_u = sim.ee_poses()
    row["achieved"] = {"franka": pose_delta(X_f, ref_poses[0]),
                       "ur": pose_delta(X_u, ref_poses[1])}
    row["belt_rmse_vs_nominal_mm"] = belt_rmse_mm(sim, ref_belt)
    if not ok:
        row["reason"] = why
        return row
    row["held"] = True
    row["measured"] = gv.measured_record(sim, sim.state_0.body_q.numpy(), ref_meas, grasp)
    a = row["achieved"][arm]
    row["notes"] = (f"EE offset {vid}: {arm} {kind} {mag:g} {row['commanded']['unit']} "
                    f"from {args.reference.name}; move {args.move_s:g} s min-jerk, hold "
                    f"{args.hold_s:g} s, settle {args.settle_s:g} s under magna OSC hold; "
                    f"achieved {arm} dpos {np.round(a['dpos_mm'], 2).tolist()} mm, drot "
                    f"{a['drot_angle_deg']:.2f} deg; {grasp.describe()}")
    path = sim_snapshot.save(sim_snapshot.capture(sim, gv.PICK_LAST, notes=row["notes"]),
                             args.out / f"{vid}_osc.npz")
    row["file"] = path.name
    # Re-check as the eval starts: restore, hold at measured, settle its 0.5 s.
    grasp2 = sim.restore(sim_snapshot.load(path), settle_steps=round(0.5 / dt))
    X_f, X_u = sim.ee_poses()
    row["recheck_0p5s"] = {"held": list(grasp2.held()),
                           "franka": pose_delta(X_f, ref_poses[0]),
                           "ur": pose_delta(X_u, ref_poses[1]),
                           "belt_rmse_vs_nominal_mm": belt_rmse_mm(sim, ref_belt)}
    return row


def summary(row: dict) -> str:
    if row.get("achieved") is None:
        return row.get("reason") or "-"
    a = row["achieved"][row["arm"]]
    return (f"dpos {np.round(a['dpos_mm'], 2).tolist()} mm, drot {a['drot_angle_deg']:.2f} deg "
            f"(tool z {a['drot_tool_z_deg']:+.2f}), belt rmse {row['belt_rmse_vs_nominal_mm']:.2f}"
            f" mm" + ("" if row["held"] else f" | {row['reason']}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--reference", type=Path, default=gv.DEFAULT_REFERENCE)
    parser.add_argument("--lcm-url", required=True, help="private LCM URL for the OSC")
    parser.add_argument("--move-s", type=float, default=1.5)
    parser.add_argument("--hold-s", type=float, default=0.5)
    parser.add_argument("--settle-s", type=float, default=gv.OSC_SETTLE_S)
    args = parser.parse_args()
    gv.configure_logging()
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    snap = sim_snapshot.load(args.reference)
    log_path = Path(tempfile.mkdtemp(prefix="make_ee_offset_starts_")) / "osc.log"
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url)
    rows, reason = [], "error"
    try:
        ref_meas = gv.measure(sim, snap.body_q)
        ref_poses = sim.ee_poses(np.asarray(snap.body_q))
        ref_belt = np.asarray(snap.body_q)[sim.info.belt_bodies, :3].astype(np.float64)
        warm = sim.start_osc(log_path)
        logger.info(f"OSC warm-up {warm:.2f} s, log {log_path}")
        # Control: the nominal state through the same hold/settle, no move.
        sim.restore(snap, settle_steps=0)
        ok, why, _ = run_steps(sim, round((args.move_s + args.hold_s + args.settle_s)
                                          / sim.frame_dt))
        X_f, X_u = sim.ee_poses()
        control = {"held": ok, "reason": why, "franka": pose_delta(X_f, ref_poses[0]),
                   "ur": pose_delta(X_u, ref_poses[1]),
                   "belt_rmse_vs_nominal_mm": belt_rmse_mm(sim, ref_belt)}
        logger.info(f"control (no move): held {ok}, franka "
                    f"{control['franka']['dpos_norm_mm']:.2f} mm, belt rmse "
                    f"{control['belt_rmse_vs_nominal_mm']:.2f} mm")
        for vid in wanted:
            spec = VARIANTS[vid]
            tried = []
            for mag in spec[3]:
                row = build_one(sim, snap, ref_poses, ref_belt, ref_meas, vid, spec, mag, args)
                tried.append({k: row.get(k) for k in ("magnitude", "held", "reason",
                                                      "belt_rmse_vs_nominal_mm")})
                logger.info(f"{vid} @ {mag:g}: held {row['held']}: {summary(row)}")
                if row["held"]:
                    break
            row["tried"] = tried
            rows.append(row)
        reason = "finished"
    finally:
        sim.close(reason)
    index = {"set": args.out.name, "kind": "ee_offsets",
             "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "nominal_start": str(args.reference),
             "nominal_start_sha256": gv.sha256(args.reference),
             "move_s": args.move_s, "hold_s": args.hold_s, "settle_s": args.settle_s,
             "offset_metric": "measured EE pose (Franka finger_tip / UR tracking frame) minus "
                              "the nominal snapshot's; dpos world mm; drot_tool_z = twist of "
                              "R0^T R about the tool z",
             "control_no_move": control, "variants": rows}
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    for r in rows:
        print(f"{r['id']:<10} {r['magnitude']:>4g} held {r['held']!s:<5} {summary(r)}")
    return 0 if all(r["held"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
