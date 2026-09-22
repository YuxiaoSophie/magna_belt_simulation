#!/usr/bin/env python3
"""Headless check of the in-process waypoint motion (no magna, no LCM).

M0 waypoints from magna's yaml (world frame, gripper commands); M1 numpy FK vs the model's bodies;
M2 IK round trips and the nominal joint trajectory; M3 the nominal pick holds the belt and is
snapshotted; M4 restore + ``pre_place_1 -> place_3`` tracking, hold and pulley reach; M5 two
restored replays agree. Checks share one sim and run in order.

Run:
    uv run python scripts/checks/check_inproc_motion.py
    uv run python scripts/checks/check_inproc_motion.py --keep --record data/lcs/recordings
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# The task packages live under src/; make them importable regardless of CWD.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from loguru import logger

from round_belt_task import arm_kinematics as ak
from round_belt_task.constants import ARM_TARGET_KD, ARM_TARGET_KE
from round_belt_task.motion import MotionError
from round_belt_task.offline_simulation import PICK_LAST, RoundBeltOfflineSimulation
from round_belt_task.outcome import belt_in_pulley_frame
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import sim_snapshot

LABELS = ["pre_pick_0", "pre_pick_1", "pick", "post_pick", "pre_place_1", "pre_place_2",
          "place_3"]
# magna round-belt-scene.dmd.yaml board weld, typed here independently of the scene.
PLANNING_BOARD_XYZ = (0.64483928, -0.19718233, 0.01076393)
PLANNING_BOARD_RPY_DEG = (-3.32822058e-01, -6.87450103e-02, 8.95207485e01)
# magna yaml pre_place_1 franka_position (board frame).
PRE_PLACE_1_BOARD = (0.197403906678, 0.269402781323, 0.0814911961127)
FK_POS_TOL, FK_ROT_TOL = 1e-4, 1e-3
M2_JUMP_TOL = 0.02
TRACK_RMS_MM, TRACK_MAX_MM, TRACK_RMS_DEG = 5.0, 10.0, 2.0
PULLEY_REACH_MM, PULLEY_REACH_BODIES = 65.0, 6
# check_sim_snapshot.py S1 belt noise (RUN-STATE, PKG-20260922-sim-snapshot: 0.0166 / 0.0308).
S1_BELT_NOISE_MM = 0.0308
DETERMINISM_MM = max(2.0, 3.0 * S1_BELT_NOISE_MM)
RUNTIME_BUDGET_S = 150.0


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _pose_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), ak.rot_angle(a, b)


@check("M0 waypoints")
def check_m0(ctx: SimpleNamespace) -> str:
    wps = load_pre_mpc_segment(ctx.params, first=LABELS[0], last=LABELS[-1])
    ctx.waypoints = {w.label: w for w in wps}
    _require([w.label for w in wps] == LABELS, f"labels {[w.label for w in wps]} != {LABELS}")
    rot = ak.rpy_to_mat3(np.radians(PLANNING_BOARD_RPY_DEG))
    want = np.asarray(PLANNING_BOARD_XYZ) + rot @ np.asarray(PRE_PLACE_1_BOARD)
    got = ctx.waypoints["pre_place_1"].franka_pos
    _require(np.abs(got - want).max() <= 1e-9, f"pre_place_1 franka {got} != {want}")
    # The live waypoints use the controller's board pose instead (yaw 1.57079, not pi/2).
    start = load_pre_mpc_segment(ctx.params, first="start", last="start")[0]
    yaw = 1.57079
    bx, by, bz = 0.67206688, -0.19727797, 0.01063954
    px, py, pz = 0.53404437402613, 0.2799276084562198, 0.061741647730498186
    want_start = np.array([bx + math.cos(yaw) * px - math.sin(yaw) * py,
                           by + math.sin(yaw) * px + math.cos(yaw) * py, bz + pz])
    _require(np.abs(start.franka_pos - want_start).max() <= 1e-9,
             f"start franka {start.franka_pos} != {want_start}")
    w = ctx.waypoints
    expected = [("pre_pick_0", "franka_gripper_mm", 40.0), ("pre_pick_1", "ur_gripper_byte", 63),
                ("pick", "franka_gripper_mm", 0.0), ("pick", "ur_gripper_byte", 255),
                ("place_3", "ur_gripper_byte", 191), ("pick", "dwell_s", 2.0),
                ("pre_pick_0", "ur_gripper_byte", None)]
    for label, key, value in expected:
        _require(getattr(w[label], key) == value,
                 f"{label}.{key} {getattr(w[label], key)} != {value}")
    return (f"7 waypoints, pre_place_1 franka ({', '.join(f'{v:.6f}' for v in got)}); "
            f"commands 40 mm / 63 / 0 mm + 255 / 191, pick dwell 2.0 s")


@check("M1 FK vs model")
def check_m1(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    _require(not hasattr(sim.bridge, "lc") and sim.bridge.url == "offline",
             f"bridge {type(sim.bridge).__name__} url {sim.bridge.url!r} is not offline")
    q_franka, q_ur = sim.arm_positions()
    franka, ur = sim.ee_poses()
    ef = _pose_err(ak.FrankaTip.fk(q_franka), franka)
    eu = _pose_err(ak.UrTracking.fk(q_ur), ur)
    ctx.m1 = (ef, eu)
    for name, (p, r) in (("franka", ef), ("ur", eu)):
        _require(p <= FK_POS_TOL and r <= FK_ROT_TOL,
                 f"{name} FK off by {p:.3g} m / {r:.3g} rad")
    return (f"franka {ef[0] * 1e3:.4f} mm {ef[1]:.2e} rad, ur {eu[0] * 1e3:.4f} mm "
            f"{eu[1]:.2e} rad")


@check("M2 IK")
def check_m2(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    worst = [0.0, 0.0]
    q = dict(zip(("franka", "ur"), sim.arm_targets()))
    for label in LABELS:
        wp_ = ctx.waypoints[label]
        for arm, chain, target in (("franka", ak.FrankaTip, wp_.franka_mat()),
                                   ("ur", ak.UrTracking, wp_.ur_mat())):
            q[arm] = ak.ik(chain, target, q[arm], max_iters=200)[0]
            p, r = _pose_err(chain.fk(q[arm]), target)
            _require(p <= FK_POS_TOL and r <= FK_ROT_TOL,
                     f"{label} {arm}: fk(ik) off by {p:.3g} m / {r:.3g} rad")
            worst = [max(worst[0], p), max(worst[1], r)]
    try:
        ctx.pick_traj = traj = sim.plan([ctx.waypoints[k] for k in LABELS[:5]])
    except MotionError as exc:
        raise AssertionError(f"nominal trajectory: {exc}") from exc
    jump = max(traj.stats["franka"]["jump_max_rad"], traj.stats["ur"]["jump_max_rad"])
    _require(jump <= M2_JUMP_TOL, f"max per-step joint jump {jump:.4f} rad > {M2_JUMP_TOL}")
    s = traj.stats
    return (f"14 round trips <= {worst[0] * 1e3:.3f} mm / {worst[1]:.1e} rad; nominal "
            f"{len(traj)} steps, jump {jump:.4f} rad, iters mean/max franka "
            f"{s['franka']['iters_mean']:.2f}/{s['franka']['iters_max']} ur "
            f"{s['ur']['iters_mean']:.2f}/{s['ur']['iters_max']}")


@check("M3 nominal pick")
def check_m3(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    if ctx.record:
        sim.start_recording(Path(ctx.record), "check_inproc_motion")
    ctx.pick_result = sim.play(ctx.pick_traj, log_every_s=0)
    sim.close_recording("pick done")  # restores below rewind the step counter
    grasp = sim.grasp_state()
    failures = grasp.failures()
    _require(not failures, f"grasp at {PICK_LAST}: {'; '.join(failures)} ({grasp.describe()})")
    ctx.snap = sim_snapshot.capture(sim, PICK_LAST, notes="check_inproc_motion M3")
    sim_snapshot.save(ctx.snap, ctx.tmp_root / f"{PICK_LAST}.npz")
    return f"{grasp.describe()}; tracking {ctx.pick_result.describe()}"


def _play_place(ctx: SimpleNamespace) -> SimpleNamespace:
    sim = ctx.sim
    sim.restore(ctx.snap)
    traj = sim.plan([ctx.waypoints["pre_place_2"], ctx.waypoints["place_3"]])
    reach_pre_place_2 = traj.segments[0].first_step + round(
        traj.segments[0].duration_s / traj.dt) - 1
    held = {}

    def on_step(i, info):
        if i == reach_pre_place_2:
            held["grasp"] = sim.grasp_state()

    result = sim.play(traj, on_step=on_step, log_every_s=0)
    return SimpleNamespace(traj=traj, result=result, grasp=held["grasp"],
                           belt=sim.belt_positions(), body_q=sim.state_0.body_q.numpy().copy())


@check("M4 place from the snapshot")
def check_m4(ctx: SimpleNamespace) -> str:
    run = ctx.m4 = _play_place(ctx)
    s = run.result.summary()
    for arm in ("franka", "ur"):
        _require(s[f"{arm}_mm_rms"] <= TRACK_RMS_MM and s[f"{arm}_mm_max"] <= TRACK_MAX_MM
                 and s[f"{arm}_deg_rms"] <= TRACK_RMS_DEG,
                 f"{arm} tracking {run.result.describe()}")
    failures = run.grasp.failures(min_belt_z=None)
    _require(not failures, f"not held at pre_place_2: {'; '.join(failures)}")
    labels = list(ctx.sim.model.body_label)
    pulley = next(b for b in ctx.sim.info.pulley_bodies if "large_round_pulley" in labels[b])
    _, r_mm, _ = belt_in_pulley_frame(run.belt, run.body_q[pulley])
    near = int((r_mm <= PULLEY_REACH_MM).sum())
    _require(near >= PULLEY_REACH_BODIES,
             f"{near} belt bodies within {PULLEY_REACH_MM:g} mm of the large pulley axis")
    segs = ", ".join(f"{g.end.label} {g.duration_s:.2f}+{g.hold_s:.2f} s"
                     for g in run.traj.segments)
    return (f"{segs}, total {run.traj.t[-1]:.2f} s; tracking {run.result.describe()}; "
            f"pre_place_2 {run.grasp.describe()}; {near} bodies within "
            f"{PULLEY_REACH_MM:g} mm of the pulley axis")


@check("M5 determinism")
def check_m5(ctx: SimpleNamespace) -> str:
    run = _play_place(ctx)
    diff = float(np.linalg.norm(run.belt - ctx.m4.belt, axis=1).max()) * 1e3
    _require(diff <= DETERMINISM_MM, f"final belt differs by {diff:.4f} mm > "
             f"{DETERMINISM_MM:g} mm")
    return f"final belt max diff {diff:.4f} mm (tol {DETERMINISM_MM:g} mm)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the temp snapshot dir")
    parser.add_argument("--record", default=None, metavar="DIR",
                        help="record the M3 pick under DIR (label check_inproc_motion)")
    parser.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML)
    parser.add_argument("--arm-ke", type=float, default=ARM_TARGET_KE)
    parser.add_argument("--arm-kd", type=float, default=ARM_TARGET_KD)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stdout, level="WARNING", format="{level: <7} | {message}")

    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_inproc_motion_"))
    ctx = SimpleNamespace(tmp_root=tmp_root, params=args.params, record=args.record)
    exit_code = 0
    sim = None
    try:
        ctx.sim = sim = RoundBeltOfflineSimulation.build(arm_ke=args.arm_ke, arm_kd=args.arm_kd)
        for name, fn in CHECKS:
            try:
                detail = fn(ctx)
            except AssertionError as exc:
                print(f"[FAIL] {name}: {exc}", file=sys.stderr)
                exit_code = 1
                break
            except Exception as exc:  # noqa: BLE001 - report, then still clean up
                print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
                traceback.print_exc()
                exit_code = 1
                break
            print(f"[PASS] {name}: {detail}", flush=True)
    finally:
        if sim is not None:
            sim.close("finished" if exit_code == 0 else "failed")
        if args.keep:
            print(f"[INFO] kept {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        if runtime > RUNTIME_BUDGET_S:
            print(f"[WARN] runtime {runtime:.1f} s > {RUNTIME_BUDGET_S:g} s budget")
        print(f"ALL IN-PROCESS MOTION CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
