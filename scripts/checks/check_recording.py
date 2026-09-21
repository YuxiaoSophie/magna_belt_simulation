#!/usr/bin/env python3
"""Headless check that ``--record`` writes a complete, loadable recording at bounded overhead.

Drives ``RoundBeltLcmSimulation --record`` (private LCM group) with ``control_step()`` through a
scripted 600-step run: hand commands (one goal republished, then changed), two Robotiq commands
and a belt trigger. Then checks the files, ``Recording.load``, events, signals, per-step
overhead, unfinished/missing-chunk loads and run-dir collisions. Checks share the sim's state.

Run:
    uv run python scripts/checks/check_recording.py
    uv run python scripts/checks/check_recording.py --keep
"""

from __future__ import annotations

import json
import os
import re
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

import lcm
import newton.examples

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from robotiq import lcmt_robotiq_command
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from task_common.recording import Recording, RunRecorder

sys.path.insert(0, str(Path(__file__).parent))
from lcm_peer_utils import StatePeer, hand_command_msg

# A private multicast group so this check never disturbs a running magna stack.
PRIVATE_LCM_URL = "udpm://239.255.76.72:7672?ttl=0"
RUN_DIR_RE = re.compile(r"\d{8}-\d{6}-check")
RECORD_CHUNK_STEPS = 250
RECORD_STATE_EVERY = 4
TOTAL_STEPS = 600
PERF_STEPS = 1000
PERF_OVERHEAD_BUDGET_MS = 0.25  # loose: the GPU may be shared with other work
PERF_GPU_BUSY_MS = 6.0
# Recording.signals: every per-row array except step/sim_time.
ROW_SIGNAL_DTYPES = {
    "wall_time": np.float64, "compute_ms": np.float32, "hand_target_mm": np.float32,
    "hand_stale": np.bool_, "franka_stale": np.bool_, "robotiq_cmd": np.uint8,
    "robotiq_cmd_valid": np.bool_, "robotiq_status": np.uint8, "robotiq_opening": np.float32,
    "lcm_rx": np.int32, "joint_q": np.float32, "joint_qd": np.float32, "efforts": np.float32,
}
# Timing-dependent: nothing feeds the arm inputs, so these fire from the first step on.
IGNORED_EVENT_KINDS = {"input_first", "input_stale", "hand_stale"}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _expected_signal_coords(sim: RoundBeltLcmSimulation) -> list[int]:
    q_start = sim.model.joint_q_start.numpy()
    joints = sorted(sim.info.robot_joints + sim.info.pulley_joints)
    return [int(c) for j in joints for c in range(q_start[j], q_start[j + 1])]


def _run_plain(sim: RoundBeltLcmSimulation, n: int) -> None:
    for _ in range(n):
        sim.control_step()
        _require(not sim.recorder.failed, f"sim.recorder.failed True at step {sim.step_index}")


def _publish_hand(peer: StatePeer, sim: RoundBeltLcmSimulation, width_m: float) -> int:
    utime = round(sim.step_index * 1e6 * sim.frame_dt)
    peer.publish(sim.channels.franka_hand_input_channel, hand_command_msg(width_m, utime))
    peer.wait_delivered()
    return utime


def _publish_robotiq(lc: lcm.LCM, sim: RoundBeltLcmSimulation, position: int) -> None:
    cmd = lcmt_robotiq_command()
    cmd.utime = round(sim.step_index * 1e6 * sim.frame_dt)
    cmd.position, cmd.speed, cmd.force = position, 255, 0
    lc.publish(sim.channels.robotiq_command_channel, cmd.encode())
    time.sleep(0.020)  # no echo channel for Robotiq: a fixed delay stands in for wait_delivered


@check("R0 build")
def check_r0(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    recorder = sim.recorder
    _require(recorder is not None, "sim.recorder is None after building with --record")
    run_dir = recorder.path
    ctx.run_dir = run_dir
    _require(run_dir.is_dir(), f"{run_dir}: run dir does not exist")
    _require(RUN_DIR_RE.fullmatch(run_dir.name) is not None,
             f"run dir name {run_dir.name!r} does not match ^\\d{{8}}-\\d{{6}}-check$")
    meta = json.loads((run_dir / "meta.json").read_text())
    _require(meta["schema"] == 1, f"meta schema {meta['schema']} != 1")
    _require(meta["finished"] is False, f"meta finished {meta['finished']} before any steps")
    n_labels, n_model = len(meta["body_labels"]), len(sim.model.body_label)
    _require(n_labels == n_model, f"meta body_labels {n_labels} != model body_label {n_model}")
    expected_coords = _expected_signal_coords(sim)
    _require(meta["signal_coords"] == expected_coords,
             "meta signal_coords != coords of info.robot_joints + info.pulley_joints")
    return (f"{run_dir.name}, body_labels {n_labels}, "
            f"signal_coords {len(expected_coords)}")


@check("R1 scripted run (600 steps)")
def check_r1(ctx: SimpleNamespace) -> str:
    sim, peer, robotiq_lc = ctx.sim, ctx.peer, ctx.robotiq_lc
    _run_plain(sim, 49)  # steps 1-49
    ctx.hand_command_utime = _publish_hand(peer, sim, 0.040)
    _run_plain(sim, 10)  # steps 50-59
    for _ in range(3):  # the same goal with a fresh utime, as magna republishes every tick
        _publish_hand(peer, sim, 0.040)
        _run_plain(sim, 10)  # steps 60-89
    _publish_hand(peer, sim, 0.030)
    _run_plain(sim, 10)  # steps 90-99, a changed goal at 90
    _publish_robotiq(robotiq_lc, sim, 255)
    _run_plain(sim, 101)  # steps 100-200

    # Set the trigger to the CURRENT finger tip: the next control_step() places the belt.
    finger_tip = sim.state_0.body_q.numpy()[sim._finger_tip_body, :3].astype(np.float64).copy()
    sim.belt_trigger_point = finger_tip
    sim.control_step()  # step 201: places the belt
    _require(sim.step_index == 201, f"step_index {sim.step_index} != 201 after the trigger step")
    _require(sim.belt_placed, "sim.belt_placed is not True after step 201")
    _require(not sim.recorder.failed, "sim.recorder.failed True at step 201")

    _run_plain(sim, 98)  # steps 202-299
    _publish_robotiq(robotiq_lc, sim, 0)
    _run_plain(sim, 100)  # steps 300-399
    sim.control_step()  # step 400 (400 % 4 == 0: a state frame)
    _require(sim.step_index == 400, f"step_index {sim.step_index} != 400")
    ctx.ref_body_q = sim.state_0.body_q.numpy().copy()
    ctx.ref_joint_q = sim.state_0.joint_q.numpy().copy()

    _run_plain(sim, 200)  # steps 401-600
    _require(sim.step_index == TOTAL_STEPS,
             f"step_index {sim.step_index} != {TOTAL_STEPS} after the scripted run")

    sim.close_recording("check")
    return "hand goals at steps 50 (+3 republishes) and 90, belt placed at 201, ref at 400"


@check("R2 files")
def check_r2(ctx: SimpleNamespace) -> str:
    run_dir = ctx.run_dir
    meta = json.loads((run_dir / "meta.json").read_text())
    _require(meta["finished"] is True, f"meta finished {meta['finished']} != True")
    _require(meta["reason"] == "check", f"meta reason {meta['reason']!r} != 'check'")
    _require(meta["num_steps"] == TOTAL_STEPS, f"meta num_steps {meta['num_steps']} != "
             f"{TOTAL_STEPS}")
    _require(meta["last_step"] == TOTAL_STEPS, f"meta last_step {meta['last_step']} != "
             f"{TOTAL_STEPS}")
    chunks = meta["chunks"]
    _require(len(chunks) == 3, f"{len(chunks)} chunks, expected 3")
    for chunk, want_steps in zip(chunks, (250, 250, 100)):
        _require(chunk["steps"] == want_steps, f"{chunk['file']}: steps {chunk['steps']} != "
                 f"{want_steps}")
        _require((run_dir / chunk["file"]).is_file(), f"{chunk['file']} does not exist")
    _require((run_dir / "events.jsonl").is_file(), "events.jsonl missing")
    _require((run_dir / "targets.jsonl").is_file(), "targets.jsonl missing")
    return "3 chunks (250/250/100 rows), finished, reason 'check'"


@check("R3 load")
def check_r3(ctx: SimpleNamespace) -> str:
    rec = Recording.load(ctx.run_dir)
    ctx.rec = rec
    _require(np.array_equal(rec.step, np.arange(1, TOTAL_STEPS + 1)),
             "rec.step != arange(1, 601)")
    _require(np.allclose(rec.sim_time, rec.step * ctx.sim.frame_dt, atol=1e-9),
             "rec.sim_time != step * control_dt")
    want_state_step = np.arange(RECORD_STATE_EVERY, TOTAL_STEPS + 1, RECORD_STATE_EVERY)
    _require(np.array_equal(rec.state_step, want_state_step),
             f"rec.state_step != arange(4, 601, 4) ({len(rec.state_step)} frames)")
    n_bodies = len(rec.meta["body_labels"])
    _require(rec.body_q.shape == (150, n_bodies, 7),
             f"rec.body_q.shape {rec.body_q.shape} != (150, {n_bodies}, 7)")
    for name, arr in {"body_q": rec.body_q, **rec.signals}.items():
        _require(np.isfinite(arr).all(), f"rec signal {name!r}: non-finite values")
    for name, dtype in ROW_SIGNAL_DTYPES.items():
        _require(name in rec.signals, f"rec.signals missing {name!r}")
        arr = rec.signals[name]
        _require(arr.dtype == dtype, f"rec.signals[{name!r}].dtype {arr.dtype} != {dtype}")
        _require(len(arr) == TOTAL_STEPS, f"rec.signals[{name!r}] leading length {len(arr)} != "
                 f"{TOTAL_STEPS}")

    frame = rec.frame_at_step(400)
    _require(frame == 99, f"rec.frame_at_step(400) {frame} != 99")
    _require(np.array_equal(rec.body_q[frame], ctx.ref_body_q), "rec.body_q[99] != ref_body_q")
    coords = rec.meta["signal_coords"]
    want_joint_q = ctx.ref_joint_q[coords].astype(np.float32)
    _require(np.array_equal(rec.signals["joint_q"][399], want_joint_q),
             "rec.signals['joint_q'][399] != ref_joint_q[signal_coords]")
    hand_coords = rec.meta["robot_io"]["franka_hand"]["coords"]
    q1, q2 = (ctx.ref_joint_q[c] for c in hand_coords)
    want_width = (-q1 + q2) * 1000.0
    got_width = float(rec.hand_width_mm()[399])
    _require(abs(got_width - want_width) <= 1e-6, f"hand_width_mm[399] {got_width} != "
             f"{want_width}")
    return f"150 frames, body_q {rec.body_q.shape}, step-400 body_q/joint_q/hand_width exact"


@check("R4 events")
def check_r4(ctx: SimpleNamespace) -> str:
    rec = ctx.rec
    all_kinds = [e.kind for e in rec.events]
    _require("run_start" not in all_kinds, "run_start event present; the check never calls "
             "sim.run()")
    _require("resync" not in all_kinds, "unexpected resync event")
    kept = [e for e in rec.events if e.kind not in IGNORED_EVENT_KINDS]
    # One hand_command per goal: three republishes of 40 mm add nothing, 30 mm adds one.
    want_order = ["hand_command", "hand_command", "robotiq_command", "belt_placed",
                  "robotiq_command", "run_end"]
    _require([e.kind for e in kept] == want_order,
             f"event kinds (filtered) {[e.kind for e in kept]} != {want_order}")

    hand_ev, hand_changed, robotiq_close, belt_ev, robotiq_open, run_end = kept
    _require(hand_ev.step in (50, 51), f"hand_command at step {hand_ev.step}, expected 50 or 51")
    _require(hand_ev.data.get("target_mm") == 40.0, "hand_command target_mm "
             f"{hand_ev.data.get('target_mm')} != 40.0")
    _require(hand_ev.data.get("utime") == ctx.hand_command_utime,
             f"hand_command utime {hand_ev.data.get('utime')} != the first message's "
             f"{ctx.hand_command_utime}")
    _require(hand_changed.step in (90, 91),
             f"changed hand_command at step {hand_changed.step}, expected 90 or 91")
    _require(hand_changed.data.get("target_mm") == 30.0, "changed hand_command target_mm "
             f"{hand_changed.data.get('target_mm')} != 30.0")

    _require(100 <= robotiq_close.step <= 105, f"robotiq_command(255) at step "
             f"{robotiq_close.step}, expected 100-105")
    _require(robotiq_close.data.get("position") == 255, "robotiq_command position "
             f"{robotiq_close.data.get('position')} != 255")

    _require(belt_ev.step == 201, f"belt_placed at step {belt_ev.step} != 201")
    _require(belt_ev.data.get("anchor_body") in rec.meta["belt_bodies"],
             f"belt_placed anchor_body {belt_ev.data.get('anchor_body')} not in "
             "meta['belt_bodies']")
    translation = belt_ev.data.get("translation")
    _require(isinstance(translation, list) and len(translation) == 3,
             f"belt_placed translation {translation!r} is not a 3-vector")

    _require(300 <= robotiq_open.step <= 305, f"robotiq_command(0) at step "
             f"{robotiq_open.step}, expected 300-305")
    _require(robotiq_open.data.get("position") == 0, "robotiq_command position "
             f"{robotiq_open.data.get('position')} != 0")

    _require(run_end.data.get("reason") == "check", "run_end reason "
             f"{run_end.data.get('reason')!r} != 'check'")

    robotiq_col0 = rec.signals["robotiq_cmd"][:, 0]
    valid = rec.signals["robotiq_cmd_valid"]
    close_row, open_row = robotiq_close.step - 1, robotiq_open.step - 1
    _require(not valid[:close_row].any(), "robotiq_cmd_valid True before the first robotiq "
             "event")
    _require(np.all(robotiq_col0[close_row:open_row] == 255), "robotiq_cmd[:, 0] != 255 "
             "between the close and open commands")
    _require(np.all(robotiq_col0[open_row:] == 0), "robotiq_cmd[:, 0] != 0 after the open "
             "command")
    return (f"hand_command@{hand_ev.step} (3 republishes dropped) hand_command(30)@"
            f"{hand_changed.step} robotiq(255)@{robotiq_close.step} belt_placed@201 "
            f"robotiq(0)@{robotiq_open.step} run_end")


@check("R5 list_runs")
def check_r5(ctx: SimpleNamespace) -> str:
    runs = Recording.list_runs(ctx.tmp_root)
    _require(runs == [ctx.run_dir], f"Recording.list_runs({ctx.tmp_root}) = {runs}, expected "
             f"[{ctx.run_dir}]")
    return f"[{ctx.run_dir.name}]"


def _step_times_ms(sim: RoundBeltLcmSimulation, n: int) -> np.ndarray:
    times = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        sim.control_step()
        times[i] = (time.perf_counter() - t0) * 1e3
    return times


@check("R6 overhead")
def check_r6(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    ctx.off_path_before = set(os.listdir(ctx.tmp_root))
    sim.recorder = None
    times_a = _step_times_ms(sim, PERF_STEPS)
    ctx.off_path_after = set(os.listdir(ctx.tmp_root))
    mean_a, max_a = float(times_a.mean()), float(times_a.max())
    if mean_a > PERF_GPU_BUSY_MS:
        print(f"[WARN] GPU busy, overhead numbers unreliable (mean_A {mean_a:.3f} ms)")

    perf_root = Path(tempfile.mkdtemp(prefix="check_recording_perf_"))
    ctx.perf_root = perf_root
    meta = sim.recorder_meta()
    sim.recorder = RunRecorder(perf_root, "perf", meta, state_every=RECORD_STATE_EVERY,
                               chunk_steps=2000)
    times_b = _step_times_ms(sim, PERF_STEPS)
    sim.close_recording("perf")
    mean_b, max_b = float(times_b.mean()), float(times_b.max())

    delta = mean_b - mean_a
    _require(delta <= PERF_OVERHEAD_BUDGET_MS, f"recording overhead {delta:.4f} ms > "
             f"{PERF_OVERHEAD_BUDGET_MS} ms budget (mean_A {mean_a:.4f}, mean_B {mean_b:.4f})")
    return (f"A off mean {mean_a:.4f}/max {max_a:.4f} ms; B mean {mean_b:.4f}/max {max_b:.4f} "
            f"ms; delta {delta:.4f} ms")


@check("R7 off-path")
def check_r7(ctx: SimpleNamespace) -> str:
    _require(ctx.off_path_after == ctx.off_path_before, f"new entries under {ctx.tmp_root} "
             f"appeared while sim.recorder was None: "
             f"{ctx.off_path_after - ctx.off_path_before}")
    return f"{ctx.tmp_root} unchanged ({len(ctx.off_path_before)} entries)"


def _load_error(path: Path) -> str | None:
    """The ``ValueError`` message of ``Recording.load(path)``, None if it loads."""
    try:
        Recording.load(path)
    except ValueError as exc:
        return str(exc)
    return None


@check("R8 unfinished load + missing chunk")
def check_r8(ctx: SimpleNamespace) -> str:
    scratch = ctx.tmp_root / "r8"
    scratch.mkdir()
    unfinished = scratch / "unfinished"
    shutil.copytree(ctx.run_dir, unfinished)
    meta_path = unfinished / "meta.json"
    meta = json.loads(meta_path.read_text())
    # A snapshot taken after the first chunk, as a killed run or a writer timeout leaves it.
    meta.update(finished=False, chunks=meta["chunks"][:1])
    meta_path.write_text(json.dumps(meta))
    rec = Recording.load(unfinished)
    _require(np.array_equal(rec.step, np.arange(1, TOTAL_STEPS + 1)),
             f"unfinished load: {len(rec.step)} rows, expected the 3 chunks' {TOTAL_STEPS}")

    (unfinished / "chunks" / "chunk_00001.npz").unlink()
    message = _load_error(unfinished)
    _require(message is not None and "expected 251" in message,
             f"unfinished load with chunk 1 gone: ValueError naming the gap expected, got "
             f"{message!r}")

    missing = scratch / "missing"
    shutil.copytree(ctx.run_dir, missing)
    (missing / "chunks" / "chunk_00001.npz").unlink()
    message = _load_error(missing)
    _require(message is not None and "chunk_00001.npz" in message and "missing" in message,
             f"finished load with a listed chunk gone: missing-chunk ValueError expected, got "
             f"{message!r}")
    return (f"unfinished copy: 3 chunks found by glob ({len(rec.step)} rows); gap and missing "
            "chunk raise ValueError")


@check("R9 run-dir collision")
def check_r9(ctx: SimpleNamespace) -> str:
    scratch = ctx.tmp_root / "r9"
    first = RunRecorder._make_run_dir(scratch, "20260101-000000-check")
    second = RunRecorder._make_run_dir(scratch, "20260101-000000-check")
    _require(second.name == first.name + "-2", f"second run dir {second.name} != "
             f"{first.name}-2")
    _require((second / "chunks").is_dir(), f"{second}/chunks missing")
    return f"{first.name}, then {second.name}"


def main() -> int:
    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_recording_"))

    parser = RoundBeltLcmSimulation.create_parser()
    parser.add_argument("--keep", action="store_true",
                        help="keep the temp recording dir(s) instead of deleting them")
    parser.set_defaults(viewer="null", realtime=False, cameras=False, lcm_url=PRIVATE_LCM_URL,
                        record=str(tmp_root), record_label="check",
                        record_chunk_steps=RECORD_CHUNK_STEPS,
                        record_state_every=RECORD_STATE_EVERY)
    exit_code = 0
    ctx = SimpleNamespace(tmp_root=tmp_root)
    viewer = args = None
    try:
        viewer, args = newton.examples.init(parser)
        sim = RoundBeltLcmSimulation(viewer, args)
        ctx.sim, ctx.peer = sim, StatePeer(PRIVATE_LCM_URL, sim.channels)
        ctx.robotiq_lc = lcm.LCM(PRIVATE_LCM_URL)
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
            print(f"[PASS] {name}: {detail}")
    finally:
        if viewer is not None:
            viewer.close()
        perf_root = getattr(ctx, "perf_root", None)
        if getattr(args, "keep", False):
            print(f"[INFO] kept {tmp_root}")
            if perf_root is not None:
                print(f"[INFO] kept {perf_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
            if perf_root is not None:
                shutil.rmtree(perf_root, ignore_errors=True)

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        print(f"ALL RECORDING CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
