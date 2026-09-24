#!/usr/bin/env python3
"""Headless check of the lock-step OSC backend (magna's Franka OSC as a child process).

K0 launch on a private LCM URL + warm-up handshake; K1 200 lock-step control steps under the
hold; K2 two snapshot restores (monotonic OSC clock, grasp, determinism); K3 a 30 mm move from
the waypoint commander; K4 the timeout path (OSC stopped) and a restart; K5 shutdown leaves no
process behind. Checks share one sim and run in order.

Run:
    uv run python scripts/checks/check_osc_backend.py
    uv run python scripts/checks/check_osc_backend.py --keep
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
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
from round_belt_task.commander import (
    FrankaTarget,
    FrankaWaypointCommander,
    mat3_to_quat,
    parse_saved_traj_message,
    saved_traj_message,
)
from round_belt_task.osc_bridge import UTIME_MATCH_US, OscTimeout
from round_belt_task.osc_simulation import OSC_SETTLE_STEPS, RoundBeltOscSimulation
from task_common import sim_snapshot

# A private multicast group so this check never disturbs a running magna stack.
PRIVATE_LCM_URL = "udpm://239.255.76.84:7684?ttl=0"
PRIVATE_PORT, SHARED_PORT = "7684", "7667"
START_STATE = sim_snapshot.DEFAULT_START_STATE_DIR / "pre_place_1.npz"
# info level: the LcmDrivenLoop's "<diagram> started" line is logged at info.
OSC_ARGS = ("--input_mode=1", "--osc_debug_level=info")
OSC_LOG_ERRORS = ("resetting", "Exception caught")
WARM_UP_TIMEOUT_S = 60.0
K1_STEPS, K1_WAIT_MEAN_MS = 200, 2.0
HOLD_DRIFT_MM, HOLD_DRIFT_DEG = 2.0, 0.5
ARM_TOL_RAD = 2e-3
# check_sim_snapshot.py S1 belt noise (RUN-STATE, PKG-20260922-sim-snapshot: 0.0166 / 0.0308).
DETERMINISM_MM = max(2.0, 3.0 * 0.0308)
K3_MOVE_M, K3_S, K3_TICK_STEPS = 0.030, 1.5, 15
K3_ARRIVE_MM, K3_UR_MM = 5.5, 1.0
K4_TIMEOUT_S, K4_STEPS = 1.0, 10
RUNTIME_BUDGET_S = 240.0


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _gpu_apps() -> set[int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {int(v) for v in out.split() if v.strip().isdigit()}


def _udp_ports(pid: int) -> list[str]:
    """Local UDP ``addr:port`` strings bound by ``pid`` (``ss -ulnp``)."""
    out = subprocess.run(["ss", "-ulnp"], capture_output=True, text=True, check=True).stdout
    return [line.split()[3] for line in out.splitlines() if f"pid={pid}," in line]


def _pose_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 1e3, math.degrees(ak.rot_angle(a, b))


def _osc_log_errors(ctx: SimpleNamespace) -> list[str]:
    return [e for e in OSC_LOG_ERRORS if e in ctx.sim.osc.log_text()]


def _steps(ctx: SimpleNamespace, n: int) -> None:
    """``n`` control steps; each must drain exactly one matched reply."""
    bridge = ctx.sim.bridge
    for i in range(n):
        before = bridge.replies
        ctx.sim.control_step()
        _require(bridge.replies == before + 1, f"step {i}: {bridge.replies - before} replies")


@check("K0 launch")
def check_k0(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    warm = sim.start_osc(ctx.tmp_root / "osc.log", warm_up_timeout_s=WARM_UP_TIMEOUT_S,
                         osc_args=OSC_ARGS)
    ctx.pids = [sim.osc.pid]
    info = sim.osc.describe()
    print(f"[INFO] OSC {info}")
    ports = _udp_ports(sim.osc.pid)
    _require(any(p.endswith(f":{PRIVATE_PORT}") for p in ports),
             f"OSC pid {sim.osc.pid} not bound to {PRIVATE_PORT}: {ports}")
    _require(not any(p.endswith(f":{SHARED_PORT}") for p in ports),
             f"OSC pid {sim.osc.pid} bound to the shared port: {ports}")
    text = sim.osc.log_text()
    _require("started" in text, "OSC log has no 'started'")
    _require(not _osc_log_errors(ctx), f"OSC log has {_osc_log_errors(ctx)}")
    ctx.warm_up_s = warm
    return f"pid {sim.osc.pid} warm-up {warm:.2f} s, UDP {', '.join(ports)}"


@check("K1 lock-step hold")
def check_k1(ctx: SimpleNamespace) -> str:
    sim, bridge = ctx.sim, ctx.sim.bridge
    bridge.reset_stats()
    start = ak.FrankaTip.fk(sim.arm_positions()[0])
    efforts = []
    t0 = t1 = time.perf_counter()
    for i in range(K1_STEPS):
        if i == 1:
            t1 = time.perf_counter()  # step 0 captures the CUDA graph
        before = bridge.replies
        pending = bridge.pending
        sim.control_step()
        _require(pending is not None and bridge.replies == before + 1,
                 f"step {i}: pending {pending}, {bridge.replies - before} replies")
        efforts.append(sim.franka_efforts())
    wall = time.perf_counter() - t1
    s = bridge.stats()
    _require(s["stale_replies"] == 0 and s["republished"] == 0, f"stats {s}")
    _require(s["wait_mean_ms"] < K1_WAIT_MEAN_MS, f"mean wait {s['wait_mean_ms']:.3f} ms")
    efforts = np.asarray(efforts)
    _require(np.isfinite(efforts).all() and np.abs(efforts).max() > 0.0,
             "efforts zero or non-finite")
    mm, deg = _pose_err(ak.FrankaTip.fk(sim.arm_positions()[0]), start)
    _require(mm < HOLD_DRIFT_MM and deg < HOLD_DRIFT_DEG, f"drift {mm:.3f} mm {deg:.3f} deg")
    ctx.k1_rate = (K1_STEPS - 1) / wall
    peak = np.abs(efforts).max(axis=1)
    return (f"{K1_STEPS} steps matched (|d| <= {UTIME_MATCH_US} us), stale 0, republished 0, "
            f"wait mean/max {s['wait_mean_ms']:.3f}/{s['wait_max_ms']:.3f} ms, "
            f"{ctx.k1_rate:.0f} steps/s (first step {t1 - t0:.2f} s); |effort| max "
            f"{peak.max():.2f} Nm (step {int(peak.argmax())}), median {np.median(peak):.3f} Nm; "
            f"drift {mm:.4f} mm {deg:.4f} deg")


@check("K2 restore + settle")
def check_k2(ctx: SimpleNamespace) -> str:
    sim, bridge = ctx.sim, ctx.sim.bridge
    snap = sim_snapshot.load(START_STATE)
    runs = []
    for k in range(2):
        before = bridge.last_sent_utime
        grasp = sim.restore(snap)
        first = bridge.sent_utimes[bridge.sent_utimes.index(before) + 1]
        _require(first > before, f"restore {k}: utime {first} after {before}")
        _require(grasp.held() == (True, True), f"restore {k}: held {grasp.held()} "
                 f"({grasp.describe()})")
        runs.append((sim.arm_positions()[0], sim.belt_positions(), grasp, first - before))
    sent = np.asarray(bridge.sent_utimes)
    _require(bool(np.all(np.diff(sent) > 0)), "published utimes not strictly increasing")
    _require(not _osc_log_errors(ctx), f"OSC log has {_osc_log_errors(ctx)}")
    dq = float(np.abs(runs[0][0] - runs[1][0]).max())
    belt = float(np.linalg.norm(runs[0][1] - runs[1][1], axis=1).max()) * 1e3
    _require(dq <= ARM_TOL_RAD, f"franka joints differ by {dq:.2e} rad > {ARM_TOL_RAD}")
    _require(belt <= DETERMINISM_MM, f"belt differs by {belt:.4f} mm > {DETERMINISM_MM:g}")
    s = bridge.stats()
    return (f"2x restore + {OSC_SETTLE_STEPS} steps: utime steps across restore "
            f"{runs[0][3]}/{runs[1][3]} us, {len(sent)} utimes strictly increasing, offset "
            f"{bridge.utime_offset_us} us; held (True, True); franka dq {dq:.2e} rad, belt "
            f"{belt:.4f} mm; {runs[1][2].describe()}; stale {s['stale_replies']}")


@check("K3 move under OSC")
def check_k3(ctx: SimpleNamespace) -> str:
    sim, bridge = ctx.sim, ctx.sim.bridge
    start = ak.FrankaTip.fk(sim.arm_positions()[0])
    ur_start = sim.ee_poses()[1]
    quat = mat3_to_quat(start[:3, :3])
    goal = start[:3, 3] + np.array([-K3_MOVE_M, 0.0, 0.0])
    commander = FrankaWaypointCommander([FrankaTarget(label="k3", pos=goal, quat_wxyz=quat,
                                                      hand_mm=None, dwell_s=0.0)])
    first_step = sim.step_index + 1  # the hook runs after the step counter advanced
    errs: list[float] = []
    prev: list = []

    def hook(step, t, joint_q, body_q):
        if (step - first_step) % K3_TICK_STEPS:
            return None
        meas = ak.FrankaTip.fk(joint_q[sim.arm_coords()[0]])
        if prev:
            # The previous tick planned knot 1 for exactly this tick.
            errs.append(float(np.linalg.norm(meas[:3, 3] - prev[0][1])) * 1e3)
        cmd = commander.tick(t, meas[:3, 3], mat3_to_quat(meas[:3, :3]), 0.0, 0.0)
        prev[:] = [cmd.knots_pos, cmd.phase]
        return saved_traj_message(round(t * 1e6), cmd.knots_pos, cmd.knots_quat, cmd.times)

    sim.commander_hook = hook
    _steps(ctx, round(K3_S / sim.frame_dt))
    msg_pos, _, _ = parse_saved_traj_message(bridge.last_traj)
    end = ak.FrankaTip.fk(sim.arm_positions()[0])
    arrive = float(np.linalg.norm(end[:3, 3] - goal)) * 1e3
    moved = float(np.linalg.norm(end[:3, 3] - start[:3, 3])) * 1e3
    ur_mm, _ = _pose_err(sim.ee_poses()[1], ur_start)
    grasp = sim.grasp_state()
    _require(arrive < K3_ARRIVE_MM, f"franka {arrive:.2f} mm from the goal")
    _require(ur_mm < K3_UR_MM, f"UR moved {ur_mm:.3f} mm")
    _require(grasp.held() == (True, True), f"held {grasp.held()} ({grasp.describe()})")
    e = np.asarray(errs)
    rms = float(np.sqrt((e * e).mean()))
    sim.commander_hook = sim.make_hold_hook()
    return (f"moved {moved:.2f} mm, {arrive:.2f} mm from the goal (phase {prev[1]}, last "
            f"knots {len(msg_pos)}); tracking vs previous knot 1 rms/max {rms:.2f}/"
            f"{e.max():.2f} mm over {len(e)} ticks; UR {ur_mm:.3f} mm; held (True, True)")


@check("K4 timeout + restart")
def check_k4(ctx: SimpleNamespace) -> str:
    sim, bridge = ctx.sim, ctx.sim.bridge
    old_pid = sim.osc.pid
    bridge.drain(sim.sim_time)  # the answer to the last publish is already in flight
    sim.stop_osc()
    bridge.timeout_s = K4_TIMEOUT_S
    sim.control_step()  # nothing pending: steps, then publishes a state no one answers
    t0 = time.perf_counter()
    try:
        sim.control_step()
    except OscTimeout as exc:
        raised, elapsed = exc, time.perf_counter() - t0
    else:
        raise AssertionError("control_step did not raise OscTimeout with the OSC stopped")
    _require(raised.process_alive is False, f"process_alive {raised.process_alive}")
    _require(elapsed >= K4_TIMEOUT_S, f"raised after {elapsed:.2f} s < {K4_TIMEOUT_S}")
    warm = sim.start_osc(ctx.tmp_root / "osc_restart.log", osc_args=OSC_ARGS)
    ctx.pids.append(sim.osc.pid)
    bridge.reset_stats()
    _steps(ctx, K4_STEPS)
    s = bridge.stats()
    _require(s["stale_replies"] == 0, f"stats {s}")
    _require(not _osc_log_errors(ctx), f"OSC log has {_osc_log_errors(ctx)}")
    bridge.timeout_s = sim.osc_timeout_s
    return (f"pid {old_pid} stopped -> OscTimeout after {elapsed:.2f} s (one republish, alive "
            f"False); restart pid {sim.osc.pid} warm-up {warm:.2f} s, {K4_STEPS} steps matched")


@check("K5 shutdown")
def check_k5(ctx: SimpleNamespace) -> str:
    ctx.sim.close("finished")
    for pid in ctx.pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f"OSC pid {pid} still alive")
    left = subprocess.run(["pgrep", "-f", PRIVATE_PORT], capture_output=True, text=True,
                          check=False).stdout
    _require(not left.strip(), f"processes on {PRIVATE_PORT}: {left.split()}")
    gpu = _gpu_apps()
    if gpu is None or ctx.gpu_baseline is None:
        return f"pids {ctx.pids} gone, pgrep empty (nvidia-smi unavailable)"
    extra = gpu - ctx.gpu_baseline - {os.getpid()}
    _require(not (extra & set(ctx.pids)), f"OSC pids on the GPU: {extra}")
    note = "" if not extra else f" (foreign GPU apps appeared: {sorted(extra)})"
    return f"pids {ctx.pids} gone, pgrep -f {PRIVATE_PORT} empty, GPU apps at baseline{note}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the OSC logs")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stdout, level="WARNING", format="{level: <7} | {message}")

    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_osc_backend_"))
    ctx = SimpleNamespace(tmp_root=tmp_root, gpu_baseline=_gpu_apps(), pids=[])
    exit_code = 0
    sim = None
    try:
        ctx.sim = sim = RoundBeltOscSimulation.build(lcm_url=PRIVATE_LCM_URL)
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
        if args.keep or exit_code:
            print(f"[INFO] kept {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        if runtime > RUNTIME_BUDGET_S:
            print(f"[WARN] runtime {runtime:.1f} s > {RUNTIME_BUDGET_S:g} s budget")
        print(f"ALL OSC BACKEND CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
