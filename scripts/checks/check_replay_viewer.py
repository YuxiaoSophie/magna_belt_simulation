#!/usr/bin/env python3
"""Headless check of :class:`task_common.replay_app.ReplayApp` over two recorded runs.

Records ``check_recording.py``'s scripted 600-step run and a 100-step run on a private LCM
group, then drives one ``ReplayApp`` (no browser) through run selection, seeking, playback,
metrics, events, plots, triads, synthetic target poses and looped-playback rates.

Run:
    uv run python scripts/checks/check_replay_viewer.py
    uv run python scripts/checks/check_replay_viewer.py --keep
"""

from __future__ import annotations

import json
import random
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
import newton
import newton.examples

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from robotiq import lcmt_robotiq_command
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from round_belt_task.scene import build_scene
from task_common.recording import (
    DEFAULT_CHUNK_STEPS,
    DEFAULT_STATE_EVERY,
    Recording,
    RunRecorder,
)
from task_common.replay_app import ReplayApp
from task_common.replay_metrics import (
    UR_BASE_BODY,
    quat_multiply,
    quat_rotate,
    target_channels,
    target_pose_at,
    target_world_pose,
)
from task_common.replay_panels import BLANK_TRIAD, MAX_PLOT_POINTS, PLOT_PERIOD_S
from task_common.scene import make_builder
from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material

sys.path.insert(0, str(Path(__file__).parent))
from lcm_peer_utils import StatePeer, hand_command_msg

# A private multicast group so this check never disturbs a running magna stack.
PRIVATE_LCM_URL = "udpm://239.255.76.75:7675?ttl=0"
REPLAY_PORT = 18085
TOTAL_STEPS = 600
SECOND_RUN_STEPS = 100
SEEK_MEAN_LIMIT_MS = 50.0
SEEK_MAX_LIMIT_MS = 250.0
GPU_BUSY_COMPUTE_MS = 6.0
RANDOM_SEEK_COUNT = 100
RANDOM_SEED = 0
PLAYBACK_S = 5.0
TICK_HZ = 50.0


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def build_model() -> newton.Model:
    builder = make_builder()
    build_scene(builder)
    return builder.finalize()


def _run_plain(sim: RoundBeltLcmSimulation, n: int) -> None:
    for _ in range(n):
        sim.control_step()
        if sim.recorder is not None and sim.recorder.failed:
            raise RuntimeError(f"sim.recorder.failed True at step {sim.step_index}")


def _publish_robotiq(lc: lcm.LCM, sim: RoundBeltLcmSimulation, position: int) -> None:
    cmd = lcmt_robotiq_command()
    cmd.utime = round(sim.step_index * 1e6 * sim.frame_dt)
    cmd.position, cmd.speed, cmd.force = position, 255, 0
    lc.publish(sim.channels.robotiq_command_channel, cmd.encode())
    time.sleep(0.020)  # no echo channel for Robotiq: a fixed delay stands in for wait_delivered


def _record_two_runs(sim: RoundBeltLcmSimulation, peer: StatePeer, robotiq_lc: lcm.LCM,
                     tmp_root: Path) -> tuple[str, str]:
    """The scripted run (check_recording.py's R1 recipe), then a 100-step 'second' run."""
    _run_plain(sim, 49)  # steps 1-49
    utime = round(sim.step_index * 1e6 * sim.frame_dt)
    peer.publish(sim.channels.franka_hand_input_channel, hand_command_msg(0.040, utime))
    peer.wait_delivered()
    sim.control_step()  # step 50
    _run_plain(sim, 49)  # steps 51-99

    _publish_robotiq(robotiq_lc, sim, 255)
    sim.control_step()  # step 100
    _run_plain(sim, 100)  # steps 101-200

    # Set the trigger to the CURRENT finger tip: the next control_step() places the belt.
    finger_tip = sim.state_0.body_q.numpy()[sim._finger_tip_body, :3].astype(np.float64).copy()
    sim.belt_trigger_point = finger_tip
    sim.control_step()  # step 201: places the belt
    if sim.step_index != 201 or not sim.belt_placed:
        raise RuntimeError(f"belt not placed at step 201 (step_index={sim.step_index}, "
                           f"belt_placed={sim.belt_placed})")
    _run_plain(sim, 98)  # steps 202-299

    _publish_robotiq(robotiq_lc, sim, 0)
    sim.control_step()  # step 300
    _run_plain(sim, TOTAL_STEPS - 300)  # steps 301-600
    if sim.step_index != TOTAL_STEPS:
        raise RuntimeError(f"step_index {sim.step_index} != {TOTAL_STEPS} after the scripted run")

    scripted_name = sim.recorder.path.name
    sim.close_recording("check")

    time.sleep(1.1)  # run dirs are named to the second: keep "second" the newest
    meta = sim.recorder_meta()
    sim.recorder = RunRecorder(tmp_root, "second", meta, state_every=DEFAULT_STATE_EVERY,
                               chunk_steps=DEFAULT_CHUNK_STEPS)
    second_name = sim.recorder.path.name
    _run_plain(sim, SECOND_RUN_STEPS)
    sim.close_recording("second")
    return scripted_name, second_name


def _frame_after_step(rec, step: int) -> int:
    """First state frame whose step is strictly after ``step``."""
    return int(np.searchsorted(rec.state_step, step, side="right"))


@check("V0 runs")
def check_v0(ctx: SimpleNamespace) -> str:
    app = ctx.app
    names = app.runs()
    _require(set(names) == {ctx.scripted_name, ctx.second_name},
             f"runs() {names} != the two recorded dirs")
    _require(names[0] == ctx.second_name, f"runs()[0] {names[0]} != newest {ctx.second_name}")
    _require(app.current_run == ctx.second_name,
             f"current_run {app.current_run} != newest {ctx.second_name}")
    app.select_run(ctx.scripted_name)
    _require(app.current_run == ctx.scripted_name, "select_run(scripted) did not switch")
    _require(app.frame_count == 150, f"frame_count {app.frame_count} != 150")
    ctx.rec = app.recording
    return f"runs {names}, newest {names[0]}, scripted frame_count {app.frame_count}"


@check("V1 seek")
def check_v1(ctx: SimpleNamespace) -> str:
    app, rec = ctx.app, ctx.rec
    for k in (0, 74, 149):
        app.seek(k)
        _require(app.current_frame == k, f"current_frame {app.current_frame} != {k}")
        want_step = int(rec.state_step[k])
        _require(app.current_step == want_step,
                 f"current_step {app.current_step} != rec.state_step[{k}]={want_step}")
        got = app.state.body_q.numpy()
        _require(np.array_equal(got, rec.body_q[k]), f"state.body_q at frame {k} != rec.body_q")
    app.seek(0)
    app.step_frames(-1)
    _require(app.current_frame == 0, f"step_frames(-1) at 0 -> {app.current_frame} != 0")
    app.seek(145)
    app.step_frames(10)
    _require(app.current_frame == 149, f"step_frames(+10) from 145 -> {app.current_frame} != 149")
    return "k=0,74,149 exact body_q; step_frames clamps at both ends"


@check("V2 playback")
def check_v2(ctx: SimpleNamespace) -> str:
    app, rec = ctx.app, ctx.rec
    app.loop = False
    app.seek(0)
    t0 = app.current_time  # frame 0's own time (state_step[0] * control_dt), not 0.0
    app.set_speed(4.0)
    app.set_playing(True)
    app.tick(0.5)
    want = rec.frame_at_time(t0 + 4.0 * 0.5)
    _require(app.current_frame == want,
             f"current_frame {app.current_frame} != frame_at_time(t0+2.0)={want}")
    for _ in range(10):
        if not app.playing:
            break
        app.tick(0.5)
    _require(not app.playing, "playing still True after ticking to the end (loop=False)")
    _require(app.current_frame == 149, f"current_frame {app.current_frame} != 149 at the end")

    app.loop = True
    app.seek(0)
    app.set_playing(True)
    prev, wrapped = app.current_frame, False
    for _ in range(10):
        app.tick(0.5)
        if app.current_frame < prev:
            wrapped = True
            break
        prev = app.current_frame
    _require(wrapped, "loop=True: frame index never dropped (no wrap detected)")
    _require(app.current_frame < 149, f"post-wrap current_frame {app.current_frame} !< 149")
    _require(app.playing, "playing turned off while looping")
    app.set_playing(False)
    app.loop = False
    return f"tick(0.5)@speed4 -> frame {want}; stops at 149; loops to {app.current_frame}"


@check("V3 metrics")
def check_v3(ctx: SimpleNamespace) -> str:
    app, rec = ctx.app, ctx.rec
    m = app.metrics
    _require(m.hand_width_mm.shape == (150,), f"hand_width_mm.shape {m.hand_width_mm.shape}")
    _require(m.belt_pad_gap_mm.shape == (150, 2),
             f"belt_pad_gap_mm.shape {m.belt_pad_gap_mm.shape}")
    _require(m.pulley_angle_deg.shape == (150, 2),
             f"pulley_angle_deg.shape {m.pulley_angle_deg.shape}")
    _require(m.pulley_wrap_deg.shape == (150, 2),
             f"pulley_wrap_deg.shape {m.pulley_wrap_deg.shape}")
    for name in ("hand_width_mm", "belt_tip_gap_mm", "belt_ur_tip_gap_mm", "belt_loop_mm",
                 "belt_loop_change_pct", "pulley_angle_deg", "pulley_wrap_deg"):
        arr = getattr(m, name)
        _require(np.isfinite(arr).all(), f"m.{name} has non-finite values")

    max_width = float(m.hand_width_mm.max())
    _require(max_width >= 38.0, f"hand_width_mm max {max_width} < 38")

    frame_after_105 = _frame_after_step(rec, 105)
    byte_after_105 = int(m.robotiq_cmd_byte[frame_after_105])
    _require(byte_after_105 == 255, f"robotiq_cmd_byte after step 105 = {byte_after_105} != 255")
    byte_last = int(m.robotiq_cmd_byte[-1])
    _require(byte_last == 0, f"robotiq_cmd_byte at the last frame = {byte_last} != 0")

    frame_after_201 = _frame_after_step(rec, 201)
    min_gap_after = float(m.belt_tip_gap_mm[frame_after_201:].min())
    _require(min_gap_after < 15.0, f"belt_tip_gap_mm min after belt_placed {min_gap_after} >= 15")

    _require(bool(np.all(m.pulley_wrap_deg == 0.0)), "pulley_wrap_deg not all 0")
    max_loop_pct = float(np.abs(m.belt_loop_change_pct).max())
    _require(max_loop_pct < 5.0, f"|belt_loop_change_pct| max {max_loop_pct} >= 5")

    # The dropped belt lands on the large pulley's face and turns it: only the small one stays.
    small_max = float(np.abs(m.pulley_angle_deg[:, 0]).max())
    _require(small_max <= 1.0, f"small pulley angle max abs {small_max} > 1 deg")
    coords = rec.meta["pulley_coords"][:2]
    rows = np.clip(rec.state_step - int(rec.step[0]), 0, len(rec.step) - 1)
    for k, coord in enumerate(coords):
        q = np.unwrap(rec.signals["joint_q"][:, rec.coord_column(coord)].astype(np.float64))
        want = np.degrees(q[rows] - q[rows[0]])
        _require(np.allclose(m.pulley_angle_deg[:, k], want, atol=1e-6),
                 f"pulley_angle_deg[:, {k}] does not match unwrap(joint_q)")

    return (f"max width {max_width:.1f} mm, min tip gap after placement {min_gap_after:.1f} mm, "
            f"max |loop change| {max_loop_pct:.2f}%, small pulley max {small_max:.3f} deg")


@check("V4 events")
def check_v4(ctx: SimpleNamespace) -> str:
    app, rec = ctx.app, ctx.rec
    events = app.all_events
    kinds = [e.kind for e in events]
    for want in ("hand_command", "belt_placed", "run_end"):
        _require(want in kinds, f"{want!r} missing from all_events")
    _require(kinds.count("robotiq_command") == 2,
             f"robotiq_command count {kinds.count('robotiq_command')} != 2")
    steps = [e.step for e in events]
    _require(steps == sorted(steps), "all_events not in step order")
    idx = app.event_index()
    _require(len(idx) == len(events),
             f"event_index() length {len(idx)} != all_events {len(events)}")

    belt_i = next(i for i, e in enumerate(events) if e.kind == "belt_placed")
    belt_event = events[belt_i]
    _require(belt_event.step == 201, f"belt_placed step {belt_event.step} != 201")
    app.jump_to_event(belt_i)
    # frame_at_step rounds down: the jump lands up to one state_every before the event.
    want_frame = rec.frame_at_step(belt_event.step)
    _require(app.current_frame == want_frame,
             f"jump_to_event(belt_placed) frame {app.current_frame} != "
             f"frame_at_step(201)={want_frame}")
    state_every = rec.meta["state_every"]
    _require(belt_event.step - state_every < app.current_step <= belt_event.step,
             f"current_step {app.current_step} not within one state_every of event step "
             f"{belt_event.step}")
    return (f"{len(events)} events, kinds {sorted(set(kinds))}, "
            f"jump_to_event(belt_placed) -> step {app.current_step}")


@check("V5 plots")
def check_v5(ctx: SimpleNamespace) -> str:
    app, rec = ctx.app, ctx.rec
    app.set_plot_window(4.0)
    app.seek(rec.frame_at_time(1.5))
    t = app.current_time
    checked = 0
    for name, handle in app.plot_handles().items():
        if not handle.visible:
            continue
        data = handle.data
        x = np.asarray(data[0])
        _require(x.min() >= t - 2.0 - 1e-6, f"{name}: x.min() {x.min()} < window start")
        _require(x.max() <= t + 2.0 + 1e-6, f"{name}: x.max() {x.max()} > window end")
        _require(len(x) <= MAX_PLOT_POINTS, f"{name}: {len(x)} points > {MAX_PLOT_POINTS}")
        lengths = {len(np.asarray(s)) for s in data}
        _require(lengths == {len(x)}, f"{name}: series lengths differ {lengths}")
        now = np.asarray(data[-1])
        finite = int(np.isfinite(now).sum())
        _require(finite == 1, f"{name}: now series has {finite} finite samples, expected 1")
        checked += 1
    _require(checked > 0, "no visible plot handles found")
    # Franka efforts are 200 Hz: > MAX_PLOT_POINTS rows in the window, so min-max decimated.
    efforts = app.plot_handles()["Franka efforts"].data
    m = app.metrics
    inside = (m.row_time >= efforts[0][0]) & (m.row_time <= efforts[0][-1])
    column = next(i for i, n in enumerate(rec.meta["effort_layout"]) if n.startswith("franka/"))
    raw = rec.signals["efforts"][:, column].astype(np.float64)[inside]
    _require(len(efforts[0]) < int(inside.sum()), "Franka efforts window was not decimated")
    _require(np.nanmax(efforts[1]) == np.nanmax(raw) and np.nanmin(efforts[1]) == np.nanmin(raw),
             "decimated Franka efforts lost the window's min/max")
    return (f"{checked} visible charts within +-2 s of t={t:.3f}; Franka efforts "
            f"{int(inside.sum())} rows -> {len(efforts[0])} points keeping min/max")


@check("V6 triads")
def check_v6(ctx: SimpleNamespace) -> str:
    app, rec = ctx.app, ctx.rec
    panel = app._triads_panel
    _require(not app.triads(), f"triads shown by default: {sorted(app.triads())}")
    _require(panel.picker.options[0] == BLANK_TRIAD
             and set(rec.meta["body_labels"]) <= set(panel.picker.options),
             "the Add triad dropdown does not list every body")
    name = "panda_hand/panda_hand"
    panel._pick(app, name)  # what a dropdown selection runs
    _require(panel.picker.value == BLANK_TRIAD, "dropdown not reset after a pick")
    _require(name in panel.remove_buttons, f"no X button for {name!r}")
    triads = app.triads()
    _require(name in triads, f"{name!r} not in triads() after set_triad(..., True)")
    handle = triads[name]
    hand_body = rec.body_index(name)
    k = app.current_frame
    row = rec.body_q[k, hand_body].astype(np.float64)
    _require(np.allclose(handle.position, row[:3], atol=1e-6), f"{name} position != rec.body_q")
    want_wxyz = (row[6], row[3], row[4], row[5])
    _require(np.allclose(handle.wxyz, want_wxyz, atol=1e-6), f"{name} wxyz != reordered quat")

    for extra in ("panda_hand/finger_tip", "/ur10/wrist_3_link"):
        app.set_triad(extra, True)
    app.set_axes_length(0.1)
    lengths = {n: h.axes_length for n, h in app.triads().items()}
    _require(len(lengths) == 3 and all(v == 0.1 for v in lengths.values()),
             f"axes_length not 0.1 on the 3 shown triads: {lengths}")
    app.set_triad("/ur10/wrist_3_link", False)  # what its X button runs
    _require("/ur10/wrist_3_link" not in app.triads()
             and "/ur10/wrist_3_link" not in panel.remove_buttons, "X did not remove the triad")

    target_names = list(target_channels(rec))
    _require(not set(target_names) & set(panel.picker.options),
             "unrecorded targets offered in the dropdown")
    for tname in target_names:
        app.set_triad(tname, True)
    still_absent = [t for t in target_names if t in app.triads()]
    _require(not still_absent, f"target triads present with no recorded targets: {still_absent}")
    for shown in list(app.triads()):
        app.set_triad(shown, False)
    return (f"none by default; dropdown pick shows {name} (position/wxyz exact) and resets; "
            f"axes_length propagates to {len(lengths)} triads; X removes; "
            f"{len(target_names)} unrecorded targets not offered and stay absent")


@check("V7 refusal")
def check_v7(ctx: SimpleNamespace) -> str:
    app = ctx.app
    before_run = app.current_run
    bad_name = ctx.scripted_name + "-bogus"
    bad_dir = app.recordings_root / bad_name
    shutil.copytree(app.recordings_root / ctx.scripted_name, bad_dir)
    meta_path = bad_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["body_labels"][0] = "bogus"
    meta_path.write_text(json.dumps(meta))

    raised = None
    try:
        app.select_run(bad_name)
    except RuntimeError as exc:
        raised = exc
    _require(raised is not None, "select_run on the bogus-label copy did not raise RuntimeError")
    _require("bogus" in str(raised), f"RuntimeError message does not mention 'bogus': {raised}")
    _require(app.current_run == before_run, f"current_run changed to {app.current_run}")
    return f"select_run({bad_name}) raised RuntimeError mentioning 'bogus'; current_run unchanged"


@check("V8 timing")
def check_v8(ctx: SimpleNamespace) -> str:
    app, rec = ctx.app, ctx.rec
    rng = random.Random(RANDOM_SEED)
    frames = [rng.randrange(rec.frame_count) for _ in range(RANDOM_SEEK_COUNT)]
    times = np.empty(len(frames))
    for i, f in enumerate(frames):
        t0 = time.perf_counter()
        app.seek(f)
        times[i] = (time.perf_counter() - t0) * 1e3
    mean_ms, max_ms = float(times.mean()), float(times.max())
    compute_mean = float(np.mean(rec.compute_ms))
    if compute_mean > GPU_BUSY_COMPUTE_MS:
        print(f"[WARN] GPU busy, seek timing unreliable (recording compute mean "
              f"{compute_mean:.3f} ms)")
    _require(mean_ms <= SEEK_MEAN_LIMIT_MS, f"seek mean {mean_ms:.2f} ms > {SEEK_MEAN_LIMIT_MS}")
    _require(max_ms <= SEEK_MAX_LIMIT_MS, f"seek max {max_ms:.2f} ms > {SEEK_MAX_LIMIT_MS}")
    return (f"{RANDOM_SEEK_COUNT} random seeks: mean {mean_ms:.2f} ms, max {max_ms:.2f} ms "
            f"(recording compute mean {compute_mean:.3f} ms)")


def _append_synthetic_targets(run_dir: Path) -> dict:
    """One UR spatial pose and one UR trajectory at the run's first step, as the sim writes."""
    meta = json.loads((run_dir / "meta.json").read_text())
    channels = meta["channels"]
    step = int(Recording.load(run_dir).step[0])
    sim_time = step * float(meta["control_dt"])
    utime = round(sim_time * 1e6)
    pose = {"utime": utime, "position": [0.4, -0.2, 0.3],
            "orientation_wxyz": [0.0, 1.0, 0.0, 0.0]}
    traj = {"utime": utime, "blocks": {
        "end_effector_position_target": {
            "t": [sim_time, sim_time + 2.0], "data": [[0.5, 0.7], [0.1, 0.1], [0.4, 0.2]],
            "datatypes": ["double"] * 3},
        "end_effector_orientation_target": {
            "t": [sim_time, sim_time + 2.0],
            "data": [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "datatypes": ["double"] * 4},
    }}
    spatial = channels["ur_target_spatial_pose_channel"]
    trajectory = channels["ur_tracking_trajectory_actor_channel"]
    lines = [json.dumps({"step": step, "sim_time": sim_time, "channel": channel,
                         "payload": payload}) + "\n"
             for channel, payload in ((spatial, pose), (trajectory, traj))]
    with open(run_dir / "targets.jsonl", "a") as f:
        f.writelines(lines)
    return {"UR target (spatial pose)": spatial, "UR EE target (traj)": trajectory,
            "pose": pose}


class _FailingHook:
    """Raises from ``on_run_loaded`` for one run, to exercise the GUI's hook-error guard."""

    def __init__(self, run_name: str) -> None:
        self.run_name = run_name

    def build_gui(self, app: ReplayApp) -> None:
        pass

    def on_run_loaded(self, app: ReplayApp, rec) -> None:
        if rec.path.name == self.run_name:
            raise ValueError("synthetic hook failure")

    def on_frame(self, app: ReplayApp, frame_index: int, step: int, sim_time: float) -> None:
        pass


@check("V9 target triads + run switch")
def check_v9(ctx: SimpleNamespace) -> str:
    app = ctx.app
    root = app.recordings_root
    info = None
    for name in (ctx.scripted_name, ctx.second_name):
        info = _append_synthetic_targets(root / name)
    app.select_run(ctx.scripted_name)  # reload with the targets
    rec = app.recording
    _require(len(rec.targets) == 2, f"scripted run has {len(rec.targets)} targets, expected 2")
    options = app._triads_panel.picker.options
    for tname in ("UR target (spatial pose)", "UR EE target (traj)"):
        _require(tname in options, f"{tname!r} not offered with a recorded target")
        app.set_triad(tname, True)
        _require(tname in app.triads(), f"{tname!r} not shown after picking it")
    _require("Franka EE target (traj)" not in options,
             "Franka target offered with no Franka target recorded")

    app.seek(rec.frame_count // 2)
    frame, step = app.current_frame, app.current_step
    spatial = app.triads()["UR target (spatial pose)"]
    _require(np.allclose(spatial.position, info["pose"]["position"], atol=1e-9),
             "spatial-pose triad position != recorded payload")
    _require(np.allclose(spatial.wxyz, info["pose"]["orientation_wxyz"], atol=1e-9),
             "spatial-pose triad wxyz != recorded payload")
    world = target_world_pose(rec, info["UR EE target (traj)"], step, frame)
    raw = target_pose_at(rec, info["UR EE target (traj)"], step)
    _require(world is not None and raw is not None, "UR trajectory pose is None mid-run")
    base = rec.body_q[frame, rec.body_index(UR_BASE_BODY)].astype(np.float64)
    q_base = quat_multiply(base[3:7], np.array([0.0, 0.0, 1.0, 0.0]))  # base_link * RotZ(pi)
    want = base[:3] + quat_rotate(q_base, raw[0])
    _require(np.allclose(world[0], want, atol=1e-9), "UR trajectory world position != "
             "base_link * RotZ(pi) * raw")
    traj = app.triads()["UR EE target (traj)"]
    _require(np.allclose(traj.position, world[0], atol=1e-6) and
             np.allclose(traj.wxyz, world[1], atol=1e-6),
             "UR trajectory triad != target_world_pose")

    # Switch from past the end of the shorter run: frame must reset, not index out of range.
    app.seek(rec.frame_count - 1)
    app.select_run(ctx.second_name)
    _require(app.current_run == ctx.second_name, "select_run(second) did not switch")
    _require(app.current_frame == 0, f"current_frame {app.current_frame} != 0 after the switch")
    _require("UR target (spatial pose)" in app.triads(), "spatial triad lost after the switch")

    # A hook bug during a GUI run switch is logged and the previous run stays loaded.
    hook = _FailingHook(ctx.scripted_name)
    app.add_hook(hook)
    try:
        app.pending["run"] = ctx.scripted_name
        app.tick(0.0)
        _require(app.current_run == ctx.second_name,
                 f"current_run {app.current_run} after a failing hook, expected the previous")
        _require(app.recording.path.name == ctx.second_name, "recording not restored")
        app.seek(app.frame_count - 1)
    finally:
        app.hooks.remove(hook)
    return (f"target triads match payload/target_world_pose at frame {frame}; switch from frame "
            f"{rec.frame_count - 1} to the {app.frame_count}-frame run resets to 0; hook error "
            "keeps the previous run")


@check("V10 playback rates")
def check_v10(ctx: SimpleNamespace) -> str:
    app = ctx.app
    app.select_run(ctx.scripted_name)
    visible = sum(1 for h in app.plot_handles().values() if h.visible)
    app.loop = True
    app.set_speed(1.0)
    app.seek(0)
    renders0, (data0, scales0) = app.render_count, app.plot_send_counts()
    app.set_playing(True)
    wraps, prev, start = 0, app.current_frame, time.perf_counter()
    while (elapsed := time.perf_counter() - start) < PLAYBACK_S:
        tick_start = time.perf_counter()
        app.tick()
        wraps += app.current_frame < prev
        prev = app.current_frame
        time.sleep(max(1.0 / TICK_HZ - (time.perf_counter() - tick_start), 0.0))
    playing = app.playing
    app.set_playing(False)
    app.loop = False
    fps = (app.render_count - renders0) / elapsed
    data, scales = (n - n0 for n, n0 in zip(app.plot_send_counts(), (data0, scales0),
                                             strict=True))
    per_chart = data / elapsed / visible
    _require(playing and wraps >= 1, f"looped playback: playing={playing}, {wraps} wraps")
    _require(0.6 * app.render_fps <= fps <= 1.1 * app.render_fps,
             f"{fps:.1f} redraws/s outside [0.6, 1.1] x render_fps {app.render_fps:g}")
    _require(per_chart <= 1.0 / PLOT_PERIOD_S + 1.0,
             f"{per_chart:.1f} chart updates/s per chart > {1.0 / PLOT_PERIOD_S + 1.0:g}")
    _require(scales == 0, f"{scales} scale resends during playback (each rebuilds a chart)")
    return (f"{PLAYBACK_S:g} s looped at 1x: {fps:.1f} redraws/s (render_fps "
            f"{app.render_fps:g}), {wraps} wrap(s), {per_chart:.1f} updates/s per chart, "
            "0 scale resends")


def main() -> int:
    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_replay_"))

    parser = RoundBeltLcmSimulation.create_parser()
    parser.add_argument("--keep", action="store_true",
                        help="keep the temp recordings root instead of deleting it")
    parser.set_defaults(viewer="null", realtime=False, cameras=False, lcm_url=PRIVATE_LCM_URL,
                        record=str(tmp_root), record_label="scripted",
                        record_chunk_steps=DEFAULT_CHUNK_STEPS,
                        record_state_every=DEFAULT_STATE_EVERY)
    viewer, args = newton.examples.init(parser)
    sim = RoundBeltLcmSimulation(viewer, args)
    peer = StatePeer(PRIVATE_LCM_URL, sim.channels)
    robotiq_lc = lcm.LCM(PRIVATE_LCM_URL)

    exit_code = 0
    app = None
    try:
        scripted_name, second_name = _record_two_runs(sim, peer, robotiq_lc, tmp_root)
    except Exception as exc:  # noqa: BLE001 - a setup failure must still clean up and report
        print(f"[FAIL] setup: {exc!r}", file=sys.stderr)
        viewer.close()
        shutil.rmtree(tmp_root, ignore_errors=True)
        return 1
    viewer.close()  # release the sim's model/viewer before the replay app builds its own

    ctx = SimpleNamespace(tmp_root=tmp_root, scripted_name=scripted_name, second_name=second_name)
    try:
        patch_viewer_shape_names()
        patch_viser_texture_material()
        app = ReplayApp(tmp_root, build_model, port=REPLAY_PORT)
        print(f"[INFO] replay app on port {app.server.get_port()}")
        ctx.app = app
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
        if app is not None:
            app.close()
        if args.keep:
            print(f"[INFO] kept {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        print(f"ALL REPLAY CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
