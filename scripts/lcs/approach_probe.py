#!/usr/bin/env python3
"""Offline feasibility screen of the approach grid and the place_3 wide-box corners.

Restores ``pre_place_1_osc.npz`` under magna's OSC (``--lcm-url``; only for the board / 2F-85
geometry and the start poses), then per setting, with no physics:

* approach grid (yaw x elev x offset x tilt): UR IK reach of the transformed start pose,
  ``pre_place_2`` and ``place_3`` (seed chain, err <= the collector's IK tolerances), Franka proxy
  (tip >= plate + 12 mm, inside the crop x/y box) and the pre-guard straight-line path clearance
  of the 2F-85 (UR jaw held) >= ``--path-min-mm``;
* place_3 corners at yaw 0, per arm (the checks are per-arm separable, so 2^5 + 2^5 cover every
  combination): UR IK + pre-guard path clearance + the place_3 clamp lift; Franka proxy.

Run:
    uv run --frozen python scripts/lcs/approach_probe.py \\
        --lcm-url 'udpm://239.255.76.137:7737?ttl=0' --out data/lcs/approach/probe/screen.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
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
import collect_motion_primitives as cmp
from loguru import logger

from round_belt_task import clearance as clr
from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import UrTracking, ik
from round_belt_task.motion import build_cartesian_trajectory
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import sim_snapshot

GRID = {"yaw_deg": list(range(-30, 31, 10)), "elev_mm": [0, 15, 30], "offset_mm": [-10, 0, 10],
        "tilt_deg": [-8, 0, 8]}
CORNERS = {"depth_mm": (-6.0, 20.0), "normal_mm": (-10.0, 10.0), "tangent_mm": (-10.0, 10.0),
           "roll_deg": (-10.0, 10.0), "yaw_deg": (-15.0, 15.0)}


def ur_reach(poses: list[np.ndarray], q0: np.ndarray) -> tuple[float, float]:
    """Max IK residual (m, rad) along a seed chain."""
    q, worst_p, worst_r = q0, 0.0, 0.0
    for X in poses:
        q, ep, er, _ = ik(UrTracking, X, q, pos_tol=col.UR_IK_POS_TOL, rot_tol=col.UR_IK_ROT_TOL)
        worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
    return worst_p, worst_r


def franka_proxy(poses: list[np.ndarray], plate_top: float) -> dict:
    tip = min(X[2, 3] for X in poses) - plate_top
    xy = np.stack([X[:2, 3] for X in poses])
    inside = bool(np.all(xy >= cmp.CROP_LO[:2]) and np.all(xy <= cmp.CROP_HI[:2]))
    return {"tip_min_mm": tip * 1e3, "inside_crop_xy": inside}


def path_min(waypoints, X_f, X_u, board, gripper, byte, dt) -> float:
    traj = build_cartesian_trajectory(waypoints, X_f, X_u, dt=dt, settle_s=0.0,
                                      start_ur_byte=byte)
    return clr.path_clearance(board, gripper, traj.ur_4x4, traj.ur_gripper_byte) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lcm-url", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--path-min-mm", type=float, default=2.0)
    ap.add_argument("--min-clearance", type=float, default=clr.DEFAULT_MIN_CLEARANCE_MM)
    args = ap.parse_args()
    col.configure_logging()
    col.check_private_url(args.lcm_url)
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first=col.FIRST, last=col.LAST)
    held, _ = col.override_waypoints(nominal, {"hold_ur_gripper": True})
    snap = sim_snapshot.load(sim_snapshot.DEFAULT_START_STATE_DIR / "pre_place_1_osc.npz")
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url)
    t0 = time.perf_counter()
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        sim.start_osc(args.out.parent / "screen_osc.log")
        sim.restore(snap)
        jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
        board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
        pulley = col.pulley_frame(sim)
        X_f0 = sim.ee_poses()[0]
        q0 = sim.arm_targets()[1]
        X_u0 = UrTracking.fk(q0)
        byte = sim.gripper_commands()[1]
        labels = list(sim.model.shape_label)
        board_shapes = [lb for lb in labels if lb and lb.startswith("board/")]
    finally:
        sim.close("finished")
    plate = board.plate
    plate_top = float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half)
    ref = next(w for w in nominal if w.label == pert.TANGENT_LABEL)
    dt = 0.005
    rows = []
    for vals in itertools.product(*GRID.values()):
        a = {k: float(v) for k, v in zip(GRID, vals, strict=True)}
        wps = col.approach_waypoints(held, a, pulley)
        fr = col.approach_frame(a, pulley, ref.franka_pos, ref.ur_pos)
        Xf, Xu = (col.approach_pose(X, a, fr, elev=True) for X in (X_f0, X_u0))
        ep, er = ur_reach([Xu] + [w.ur_mat() for w in wps[1:]], q0)
        fp = franka_proxy([Xf] + [w.franka_mat() for w in wps[1:]], plate_top)
        pm = path_min(wps, Xf, Xu, board, gripper, byte, dt)
        ok = (ep <= col.UR_IK_POS_TOL and er <= col.UR_IK_ROT_TOL
              and fp["tip_min_mm"] >= col.FRANKA_FLOOR_MM and fp["inside_crop_xy"]
              and pm >= args.path_min_mm)
        rows.append({"approach": a, "ur_ik_err_mm": ep * 1e3, "ur_ik_err_deg": np.degrees(er),
                     **fp, "path_min_mm": pm, "ok": bool(ok)})
    tangent = pert.belt_tangent(held)
    corners = {"ur": [], "franka": []}
    for vals in itertools.product(*CORNERS.values()):
        v = dict(zip(CORNERS, vals, strict=True))
        sample = {"mode": "fixed", "franka": v, "ur": v}
        p, depth = col.place3_perturbation(sample, tangent)
        p_u = dataclasses.replace(p, franka_dpos_m=np.zeros(3), franka_tilt_deg=0.0,
                                  franka_yaw_deg=0.0)
        wps = col.depth_waypoints(held, {"franka": 0.0, "ur": depth["ur"]})
        moved = pert.apply(wps, p_u, tangent)
        ep, er = ur_reach([X_u0] + [w.ur_mat() for w in moved[1:]], q0)
        pm = path_min(moved, X_f0, X_u0, board, gripper, byte, dt)
        _, rep = clr.clamp_waypoints(wps, p_u, tangent, board, gripper,
                                     min_clearance=args.min_clearance * 1e-3)
        corners["ur"].append({"values": v, "ur_ik_err_mm": ep * 1e3, "path_min_mm": pm,
                              "clamp_lift_mm": rep.max_lift_mm, "tilt_scale": rep.tilt_scale,
                              "ok_ik": bool(ep <= col.UR_IK_POS_TOL),
                              "ok_path": bool(pm >= args.path_min_mm)})
        p_f = dataclasses.replace(p, ur_dpos_m=np.zeros(3), ur_tilt_deg=0.0, ur_yaw_deg=0.0)
        wps = col.depth_waypoints(held, {"franka": depth["franka"], "ur": 0.0})
        moved = pert.apply(wps, p_f, tangent)
        fp = franka_proxy([X_f0] + [w.franka_mat() for w in moved[1:]], plate_top)
        corners["franka"].append({"values": v, **fp, "ok": bool(
            fp["tip_min_mm"] >= col.FRANKA_FLOOR_MM and fp["inside_crop_xy"])})
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "grid": GRID, "corners": CORNERS,
           "pulley_large": {k: v.tolist() for k, v in pulley.items()},
           "plate_top_z": plate_top, "board_collider_shapes": board_shapes,
           "board_colliders_used": {"plate": clr.BOARD_PLATE_SHAPE, "pulleys": len(board.pulleys)},
           "path_min_mm": args.path_min_mm, "approach": rows, "place3_corners": corners,
           "wall_s": time.perf_counter() - t0}
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    n_ok = sum(r["ok"] for r in rows)
    logger.info(f"approach grid: {n_ok}/{len(rows)} pass")
    for key, values in GRID.items():
        line = ", ".join(f"{v}: {sum(r['ok'] for r in rows if r['approach'][key] == v)}/"
                         f"{sum(r['approach'][key] == v for r in rows)}" for v in values)
        logger.info(f"  {key}: {line}")
    for arm in ("ur", "franka"):
        rs = corners[arm]
        okk = "ok_path" if arm == "ur" else "ok"
        logger.info(f"place_3 corners {arm}: {sum(r[okk] for r in rs)}/{len(rs)} pass ({okk})")
        for key, (lo, hi) in CORNERS.items():
            logger.info(f"  {key}: {lo:g}: {sum(r[okk] for r in rs if r['values'][key] == lo)}"
                        f"/16, {hi:g}: {sum(r[okk] for r in rs if r['values'][key] == hi)}/16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
