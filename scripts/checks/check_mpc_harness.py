#!/usr/bin/env python3
"""Live check of the MPC evaluation harness (``scripts/lcs/eval_learned_mpc.py``).

H0 OSC + assembly controller launch on a private URL; H1 a baseline episode (markers, UR line
frame, tracking, state/utime layout); H2 a learned episode (``LATENT_STATE`` cadence and bit-exact
latents, plans answering each latent inside the LCS input bounds); H3 two grasp variants
(``[SKIP]`` without the set); H4 alignment metrics on the demo itself (``[SKIP]`` without it);
H5 dataset compatibility; H6 hygiene. H1-H3 run the harness as a subprocess while this process
listens on the same private URL.

Run:
    uv run python scripts/checks/check_mpc_harness.py
    uv run python scripts/checks/check_mpc_harness.py --keep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "checks",
           REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_learned_mpc as harness
import lcm
from check_lcs_collector import (
    LCS_LEARNING_ROOT,
    LCS_LEARNING_VENV,
    LOADER_SHAPES,
    Skip,
)

from dairlib import lcmt_robot_output, lcmt_timestamped_saved_traj
from round_belt_task.arm_kinematics import UrTracking
from round_belt_task.commander import (
    METADATA_NAME,
    parse_saved_traj_message,
    pose_mat,
    x_tool0_tracking,
)
from round_belt_task.controller_bridge import ur_base_world
from round_belt_task.outcome import LABELS
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.latent_encoder import (
    LATENT_STATE_CHANNEL,
    DemoGoals,
    LatentEncoder,
    LearnedLcs,
    parse_latent_state_message,
)
from task_common.lcm_contract import LcmChannels
from task_common.magna_process import (
    CONTROLLER_STARTED,
    MAGNA_WORKTREE,
    AssemblyControllerProcess,
)
from task_common.osc_process import OscProcess

PRIVATE_LCM_URL = "udpm://239.255.76.89:7689?ttl=0"
PRIVATE_PORT, SHARED_PORT = "7689", "7667"
MAIN_MAGNA = Path("/home/hienbui/git/magna")
VARIANT_SET = sim_snapshot.DEFAULT_START_STATE_DIR / "grasp_variants" / "set1"
DEMO_EPISODE = REPO_ROOT / "data" / "lcs" / "demo" / "demo_episode.npz"
OSC_ARGS = ("--input_mode=1", "--osc_debug_level=info")
TICK_S = lcs.SIM_DT_S
UR_KNOT0_TOL_MM = 0.5
TRACK_FRANKA_MM, TRACK_UR_MM = 3.0, 4.0
KNOT0_TOL = 1e-6
BOUND_TOL, UR_BOUND_TOL = 1e-9, 1e-6
ACTION_TOL = 1e-9
RUN_TIMEOUT_S = 300.0
H2_MAX_EPISODE_S = 3.0
RUNTIME_BUDGET_S = 480.0
FRANKA_TRAJ = "TARGET_CARTESIAN_POSE_TRAJECTORY"
UR_TRAJ = "UR_TARGET_CARTESIAN_POSE_TRAJECTORY"
FRANKA_STATE = "FRANKA_STATE"
CONTROLLER_UR_STATE = LcmChannels().ur_state_channel
CAPTURE_CHANNELS = (LATENT_STATE_CHANNEL, FRANKA_STATE, CONTROLLER_UR_STATE, FRANKA_TRAJ,
                    UR_TRAJ)


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


def _udp_lines() -> list[str]:
    return subprocess.run(["ss", "-ulnp"], capture_output=True, text=True,
                          check=True).stdout.splitlines()


def _ports_of(pid: int) -> list[str]:
    return [line.split()[3] for line in _udp_lines() if f"pid={pid}," in line]


def _pids_on(port: str) -> set[int]:
    out = set()
    for line in _udp_lines():
        if line.split()[3:4] and line.split()[3].endswith(f":{port}"):
            out.update(int(p.split(",")[0]) for p in line.split("pid=")[1:])
    return out


def _tree_digest(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _git_porcelain(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                          text=True, check=False).stdout


class Capture:
    """Records raw messages on the private URL from a background thread."""

    def __init__(self, url: str, channels=CAPTURE_CHANNELS) -> None:
        self.lc = lcm.LCM(url)
        self.msgs: list[tuple[str, bytes]] = []
        for ch in channels:
            self.lc.subscribe(ch, self._on)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _on(self, channel: str, data: bytes) -> None:
        self.msgs.append((channel, data))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.lc.handle_timeout(50)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _run_harness(ctx: SimpleNamespace, name: str, extra: list[str]) -> SimpleNamespace:
    out = ctx.tmp_root / name
    argv = [sys.executable, str(REPO_ROOT / "scripts" / "lcs" / "eval_learned_mpc.py"),
            "--lcm-url", ctx.lcm_url, "--out", str(out), *extra]
    t0 = time.perf_counter()
    with Capture(ctx.lcm_url) as cap, open(ctx.tmp_root / f"{name}.log", "w") as log:
        proc = subprocess.run(argv, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                              timeout=RUN_TIMEOUT_S, check=False)
    wall = time.perf_counter() - t0
    text = (ctx.tmp_root / f"{name}.log").read_text(errors="replace")
    _require(proc.returncode == 0, f"{name}: harness exit {proc.returncode}: {text[-1500:]}")
    index = json.loads((out / "index.json").read_text())
    return SimpleNamespace(out=out, index=index, rows=index["episodes"], msgs=cap.msgs, wall=wall)


def _decode_trajs(msgs, channel: str) -> list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    """``(order, utime, pos, quat, times)`` of the non-empty trajectories on ``channel``."""
    out = []
    for i, (ch, data) in enumerate(msgs):
        if ch != channel:
            continue
        msg = lcmt_timestamped_saved_traj.decode(data)
        if not msg.saved_traj.trajectories or msg.saved_traj.metadata.name == METADATA_NAME:
            continue  # the controller's pre-plan output / the harness's own settle hold
        try:
            pos, quat, times = parse_saved_traj_message(msg)
        except KeyError:
            continue
        out.append((i, int(msg.utime), pos, quat, times))
    return out


def _robot_states(msgs, channel: str) -> dict[int, np.ndarray]:
    return {int(m.utime): np.asarray(m.position, dtype=np.float64)
            for m in (lcmt_robot_output.decode(d) for ch, d in msgs if ch == channel)}


def _quat_close(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(min(np.abs(a - b).max(), np.abs(a + b).max()))


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


# --- checks -----------------------------------------------------------------------------------

@check("H0 launch")
def check_h0(ctx: SimpleNamespace) -> str:
    osc_bin, osc_root = harness.default_osc_binary()
    osc, ctrl = OscProcess(), AssemblyControllerProcess()
    try:
        osc.start(ctx.lcm_url, ctx.tmp_root / "h0_osc.log", cwd=osc_root, binary=osc_bin,
                  extra_args=OSC_ARGS)
        ctrl.start(ctx.lcm_url, harness.DEFAULT_PARAMS["baseline"], ctx.tmp_root / "h0_ctrl.log",
                   magna_root=MAGNA_WORKTREE)
        ctx.pids += [osc.pid, ctrl.pid]
        started = ctrl.wait_for(CONTROLLER_STARTED, harness.CONTROLLER_START_TIMEOUT_S)
        _require(osc.alive(), f"OSC exited: {osc.log_text()[-500:]}")
        time.sleep(0.5)
        ports = {pid: _ports_of(pid) for pid in (osc.pid, ctrl.pid)}
        for pid, got in ports.items():
            _require(any(p.endswith(f":{PRIVATE_PORT}") for p in got),
                     f"pid {pid} not bound to {PRIVATE_PORT}: {got}")
            _require(not any(p.endswith(f":{SHARED_PORT}") for p in got),
                     f"pid {pid} bound to the shared port: {got}")
        on_port = _pids_on(PRIVATE_PORT)
        _require(on_port <= {osc.pid, ctrl.pid, os.getpid()},
                 f"foreign pids on {PRIVATE_PORT}: {on_port - {osc.pid, ctrl.pid}}")
        argv = " ".join(ctrl.argv)
        _require(f"--lcm_url={ctx.lcm_url}" in argv and f"--local_lcm_url={ctx.lcm_url}" in argv,
                 f"controller argv {argv}")
    finally:
        codes = (ctrl.stop(), osc.stop())
    return (f"OSC pid {ctx.pids[-2]} + controller pid {ctx.pids[-1]} (started in {started:.2f} s)"
            f"; UDP {ports}; only ours on :{PRIVATE_PORT}; stopped by pid, exit codes {codes}")


@check("H1 baseline episode")
def check_h1(ctx: SimpleNamespace) -> str:
    run = _run_harness(ctx, "h1", ["--mode", "baseline"])
    ctx.h1 = run
    _require(len(run.rows) == 1, f"{len(run.rows)} rows")
    row = run.rows[0]
    _require(row["status"] == "ok" and row["outcome"] in LABELS, f"row {row.get('status')} "
             f"{row.get('outcome')} {row.get('reason')}")
    ev = row["controller_events"]
    order = [ev.get(f"pre_mpc_target_{i}_reached") for i in range(3)] + [
        ev.get("all_pre_completed")]
    _require(all(v is not None for v in order) and order == sorted(order),
             f"markers out of order / missing: {ev}")
    path = run.out / row["file"]
    lcs.validate_episode(path, period_us=lcs.SAMPLE_PERIOD_US)
    d = _load(path)
    _require(np.array_equal(d["state"][:, 26:33], d["sim_ee_franka"]),
             "state[:, 26:33] != sim_ee_franka")
    # UR line knot 0 at each regeneration vs FK of the UR state it was built from.
    ur_states = _robot_states(run.msgs, CONTROLLER_UR_STATE)
    X_W_base, X_t0_tr = ur_base_world(), x_tool0_tracking()
    errs, seen = [], set()
    for _, _, pos, quat, times in _decode_trajs(run.msgs, UR_TRAJ):
        key = round(times[0] * 1e6)
        if key in seen or key not in ur_states:
            continue
        seen.add(key)
        X = X_W_base @ pose_mat(pos[0], quat[0]) @ X_t0_tr
        errs.append(float(np.linalg.norm(X[:3, 3] - UrTracking.fk(ur_states[key])[:3, 3])) * 1e3)
    _require(errs, "no UR line matched a UR state utime")
    _require(max(errs) <= UR_KNOT0_TOL_MM, f"UR knot 0 vs FK max {max(errs):.3f} mm")
    track = row["tracking_rms_mm"]
    _require(track[0] <= TRACK_FRANKA_MM and track[1] <= TRACK_UR_MM,
             f"tracking rms {track[0]:.2f} / {track[1]:.2f} mm")
    franka_u = [int(lcmt_robot_output.decode(b).utime) for ch, b in run.msgs if ch == FRANKA_STATE]
    fset, uset = set(franka_u), set(ur_states)
    lo, hi = min(fset), max(fset)
    unpaired = {u for u in fset ^ uset if lo < u < hi}
    _require(not unpaired, f"{len(unpaired)} FRANKA_STATE/UR state utimes unpaired")
    ctx.h1_file = path
    return (f"{row['outcome']} in {row['duration_s']:.2f} s; markers pre-MPC 0/1/2 at "
            f"{order[0]:.3f}/{order[1]:.3f}/{order[2]:.3f} s, all completed {order[3]:.3f} s; "
            f"validates at {lcs.SAMPLE_PERIOD_US} us; UR knot 0 vs FK max {max(errs):.4f} mm "
            f"over {len(errs)} lines; tracking rms Franka {track[0]:.2f} / UR {track[1]:.2f} mm;"
            f" state ee == frames; {len(fset)} state utimes paired; final belt rmse "
            f"{row['alignment']['final']['belt_rmse_mm']:.2f} mm (harness {run.wall:.0f} s)")


@check("H2 latent path")
def check_h2(ctx: SimpleNamespace) -> str:
    deploy = harness.DEFAULT_DEPLOY
    if not deploy.is_file():
        deploy = harness.DEPLOY_ROOT / "deploy" / "deploy.npz"
    # Short on purpose: the latent path, not the policy's outcome, is under test here.
    run = _run_harness(ctx, "h2", ["--mode", "learned", "--deploy", str(deploy),
                                   "--max-episode-s", str(H2_MAX_EPISODE_S)])
    ctx.h2 = run
    row = run.rows[0]
    _require(row["status"] in ("ok", "timeout"), f"row {row['status']} {row.get('reason')}")
    path = run.out / row["file"]
    lcs.validate_episode(path, period_us=lcs.SAMPLE_PERIOD_US)
    d = _load(path)
    ctx.h2_file = path
    lat_msgs = [(i, parse_latent_state_message(lcmt_timestamped_saved_traj.decode(b)))
                for i, (ch, b) in enumerate(run.msgs) if ch == LATENT_STATE_CHANNEL]
    utimes = [m[1][0] for m in lat_msgs]
    frame_u = d["sim_osc_utime"].astype(np.int64)
    _require(utimes == frame_u[:-1].tolist(),
             f"LATENT_STATE utimes {utimes[:3]}.. ({len(utimes)}) != frame utimes[:-1] "
             f"{frame_u[:3].tolist()}.. ({len(frame_u) - 1})")
    _require(bool(np.all(np.diff(utimes) > 0)), "LATENT_STATE utimes not strictly increasing")
    enc = LatentEncoder.load(deploy)
    model = LearnedLcs.load(deploy)
    worst_z = 0.0
    for t, (_, (_, _, z, _, _, _)) in enumerate(lat_msgs):
        mine = enc.encode(d["pcd"][t], d["state"][t], d["pcd_belt"][t])
        _require(np.array_equal(mine, z), f"frame {t}: z differs by {np.abs(mine - z).max():.2e}")
        worst_z = max(worst_z, float(np.abs(d["sim_latent"][t] - z).max()))
    _require(worst_z == 0.0, f"sim_latent vs published z {worst_z:.2e}")
    # Each latent's answering plan.
    plans = _decode_trajs(run.msgs, FRANKA_TRAJ)
    ur_lines = _decode_trajs(run.msgs, UR_TRAJ)
    t0 = float(d["sim_time"][0])
    marks = [t0 + v for k, v in row["controller_events"].items()
             if k.startswith("stage_") and not k.endswith("_dist")]
    end_t = row["controller_events"].get("mpc_completed")
    end_t = math.inf if end_t is None else t0 + end_t
    X_W_base, X_t0_tr = ur_base_world(), x_tool0_tracking()
    matched, consumed, knot0_err, ur_err = 0, 0, 0.0, 0.0
    lags = []
    lb, ub = model.u_lb, model.u_ub
    for j, (order, (_, t, _, ee_f, ee_u, _)) in enumerate(lat_msgs):
        nxt = lat_msgs[j + 1][0] if j + 1 < len(lat_msgs) else len(run.msgs)
        plan = next((p for p in plans if order < p[0] < nxt and p[4][0] >= t - 0.5 * TICK_S),
                    None)
        if plan is None:
            near = any(abs(m - t) <= 2 * TICK_S + 1e-9 for m in marks) or t >= end_t - 1e-9
            _require(near, f"latent t={t:.3f}: no plan before the next latent")
            consumed += 1
            continue
        _, _, pos, quat, times = plan
        lags.append(times[0] - t)
        _require(times[0] - t <= 2 * TICK_S + 1e-9, f"latent t={t:.3f}: plan at {times[0]:.3f}")
        knot0_err = max(knot0_err, float(np.abs(pos[0] - ee_f[:3]).max()),
                        _quat_close(quat[0], ee_f[3:]))
        dp = np.diff(pos, axis=0)
        dr = np.array([lcs.delta_rotvec(quat[i], quat[i + 1]) for i in range(len(quat) - 1)])
        _require(np.all(dp >= lb[0:3] - BOUND_TOL) and np.all(dp <= ub[0:3] + BOUND_TOL),
                 f"latent t={t:.3f}: position deltas outside [u_lb, u_ub]")
        _require(np.all(dr >= lb[6:9] - BOUND_TOL) and np.all(dr <= ub[6:9] + BOUND_TOL),
                 f"latent t={t:.3f}: rotation deltas outside [u_lb, u_ub]")
        line = next((u for u in ur_lines if u[0] > plan[0] and u[4][0] >= t - 0.5 * TICK_S),
                    None)
        if line is not None:
            X1 = X_W_base @ pose_mat(line[2][1], line[3][1]) @ X_t0_tr
            du = X1[:3, 3] - ee_u[:3]
            q1 = harness.pose7_mat(X1)[3:]
            drot = lcs.delta_rotvec(ee_u[3:], q1)
            _require(np.all(du >= lb[3:6] - UR_BOUND_TOL) and np.all(du <= ub[3:6] + UR_BOUND_TOL)
                     and np.all(drot >= lb[9:12] - UR_BOUND_TOL)
                     and np.all(drot <= ub[9:12] + UR_BOUND_TOL),
                     f"latent t={t:.3f}: UR knot 1 - latent pose {du}, {drot} outside the bounds")
            ur_err = max(ur_err, float(np.abs(du).max()))
        matched += 1
    _require(matched > 0, "no latent was answered by a plan")
    _require(knot0_err <= KNOT0_TOL, f"knot 0 vs the latent's ee_pose_franka {knot0_err:.2e}")
    _require(np.isfinite(d["sim_goal_dist"]).all(), "sim_goal_dist not finite")
    _require(bool(np.all(np.diff(d["sim_stage"]) >= 0)), "sim_stage decreases")
    ended = end_t < math.inf or row["status"] == "timeout"
    _require(ended, f"episode did not end on a stage marker / max duration: {row['status']}")
    lat = row["osc_stats"]["traj_after_latent_ms"]
    lt = row["latent"]
    return (f"{len(utimes)} latents == frame utimes, z bit-exact; {matched} plans answer their "
            f"latent (t0 - t in [{min(lags) * 1e3:.0f}, {max(lags) * 1e3:.0f}] ms), {consumed} "
            f"consumed at a stage switch/end; knot 0 err {knot0_err:.1e}; deltas within "
            f"[u_lb, u_ub]; UR |u| max {ur_err * 1e3:.2f} mm; traj_after_latent mean/max "
            f"{lat['mean']:.1f}/{lat['max']:.1f} ms; pred err rms "
            f"{lt['pred_err_rms_whitened']:.2f}; stages {row['stage_ends']} at "
            f"{row['stage_times']} s; goal_dist final {np.round(lt['goal_dist_final'], 2)}; "
            f"{row['outcome']} ({row['status']}); final belt rmse "
            f"{row['alignment']['final']['belt_rmse_mm']:.1f} mm")


@check("H3 start-state set")
def check_h3(ctx: SimpleNamespace) -> str:
    index_path = VARIANT_SET / "index.json"
    if not index_path.is_file():
        raise Skip(f"{index_path} not found (grasp variants not built yet)")
    if not harness.DEFAULT_DEMO_GOALS.is_file():
        raise Skip(f"{harness.DEFAULT_DEMO_GOALS} not found")
    rows = json.loads(index_path.read_text())["variants"]
    by_id = {r["id"]: r for r in rows}
    _require("gv_00" in by_id, f"no gv_00 in {index_path}")

    def size(r) -> float:
        m = r.get("commanded") or {}
        return float(sum(abs(float(v)) for v in m.values() if isinstance(v, int | float)))

    other = max((r for r in rows if r["id"] != "gv_00" and r.get("held") is not False
                 and r.get("file")), key=size, default=None)
    _require(other is not None, "no held non-zero variant")
    run = _run_harness(ctx, "h3", ["--mode", "baseline", "--start-states", str(VARIANT_SET),
                                   "--variants", f"gv_00,{other['id']}"])
    got = {r["start_state"]["id"]: r for r in run.rows}
    _require(set(got) == {"gv_00", other["id"]}, f"rows {sorted(got)}")
    for vid, r in got.items():
        _require(r["status"] != "skipped", f"{vid} skipped: {r.get('reason')}")
        _require(r["start_state"]["measured"] == by_id[vid].get("measured"),
                 f"{vid}: measured offsets not carried")
    a0, a1 = (got[v]["alignment"]["initial"] for v in ("gv_00", other["id"]))
    _require(a1["belt_best_shift"] != 0 or a1["belt_rmse_mm"] > a0["belt_rmse_mm"],
             f"initial alignment does not differ: {a0} vs {a1}")
    return (f"gv_00 vs {other['id']} (commanded {other.get('commanded')}): initial belt rmse "
            f"{a0['belt_rmse_mm']:.2f} vs {a1['belt_rmse_mm']:.2f} mm, best shift "
            f"{a0['belt_best_shift']} vs {a1['belt_best_shift']} "
            f"({a1['belt_best_shift_rmse_mm']:.2f} mm); outcomes "
            f"{got['gv_00']['outcome']} / {got[other['id']]['outcome']}")


@check("H4 alignment metrics")
def check_h4(ctx: SimpleNamespace) -> str:
    if not (DEMO_EPISODE.is_file() and harness.DEFAULT_DEMO_GOALS.is_file()):
        raise Skip(f"{DEMO_EPISODE} / {harness.DEFAULT_DEMO_GOALS} not found")
    demo = DemoGoals.load(harness.DEFAULT_DEMO_GOALS)
    d = _load(DEMO_EPISODE)
    worst = {}
    for frame, stage in ((-1, demo.n_stages - 1), (0, 0)):
        a = harness.alignment(d["pcd_belt"][frame], d["sim_ee_franka"][frame],
                              d["sim_ee_ur"][frame], demo, stage)
        for k in ("belt_rmse_mm", "belt_chamfer_mm", "belt_best_shift_rmse_mm"):
            _require(a[k] == 0.0, f"demo frame {frame} vs stage {stage}: {k} {a[k]}")
        _require(a["belt_best_shift"] == 0, f"shift {a['belt_best_shift']}")
        mm = max(a["franka_pose_err"][0], a["ur_pose_err"][0])
        deg = max(a["franka_pose_err"][1], a["ur_pose_err"][1])
        _require(mm <= 1e-9 and deg <= 1e-5, f"demo frame {frame}: pose err {mm} mm {deg} deg")
        worst[stage] = (mm, deg)
    finite = 0
    for run in (getattr(ctx, "h1", None), getattr(ctx, "h2", None)):
        for r in [] if run is None else run.rows:
            for part in ("initial", "final"):
                a = r["alignment"][part]
                vals = [a["belt_rmse_mm"], a["belt_chamfer_mm"], a["belt_best_shift_rmse_mm"],
                        *a["franka_pose_err"], *a["ur_pose_err"]]
                _require(all(np.isfinite(vals)), f"{r['file']} {part}: {vals}")
                finite += 1
    return (f"demo last frame vs {demo.stage_labels[-1]} and frame 0 vs {demo.stage_labels[0]}: "
            f"rmse/chamfer/best-shift 0, shift 0, pose err <= {max(v[0] for v in worst.values())}"
            f" mm / {max(v[1] for v in worst.values()):.1e} deg; {finite} run alignments finite")


@check("H5 dataset compatibility")
def check_h5(ctx: SimpleNamespace) -> str:
    files = [p for p in (getattr(ctx, "h1_file", None), getattr(ctx, "h2_file", None)) if p]
    _require(files, "no run files (H1/H2 failed)")
    worst = 0.0
    for path in files:
        d = _load(path)
        T = d["state"].shape[0]
        want = np.stack([lcs.action_vector(d["sim_ee_franka"][t], d["sim_cmd_knot1_franka"][t],
                                           d["sim_ee_ur"][t], d["sim_cmd_ur_t1"][t])
                         for t in range(T)])
        err = float(np.abs(want - d["actions"]).max())
        _require(err <= ACTION_TOL, f"{path.name}: actions off by {err:.2e}")
        worst = max(worst, err)
        _require(d["sim_latent"].shape == (T, 16), f"sim_latent {d['sim_latent'].shape}")
        meta = json.loads(str(d["sim_meta"]))
        _require(meta["lcs_format"].get("action_definition") == "knot1_minus_measured",
                 f"action_definition {meta['lcs_format']}")
    loader = "skipped (no lcs_learning venv)"
    if LCS_LEARNING_VENV.is_file():
        code = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(LCS_LEARNING_ROOT)!r})\n"
            "from lcs_learning.dataset_round_belt_tuples import RoundBeltTupleDataset\n"
            f"ds = RoundBeltTupleDataset(data_file={[str(p) for p in files]!r}, num_points=1800, "
            "proprio_dim=40, control_dim=12, point_cloud_source='camera_plus_belt', "
            "belt_num_points=150)\n"
            f"keys = {list(LOADER_SHAPES)!r}\n"
            "print(json.dumps({'len': len(ds), 'shapes': {k: list(ds[0][k].shape) "
            "for k in keys}}))\n")
        proc = subprocess.run([str(LCS_LEARNING_VENV), "-c", code], capture_output=True,
                              text=True, timeout=180, check=False)
        _require(proc.returncode == 0, f"lcs_learning loader: {proc.stderr.strip()[-800:]}")
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        want_len = sum(_load(p)["state"].shape[0] - 1 for p in files)
        _require(out["len"] == want_len, f"loader len {out['len']} != {want_len}")
        _require(out["shapes"] == LOADER_SHAPES, f"loader shapes {out['shapes']}")
        loader = f"lcs_learning loader len {out['len']}, shapes OK"
    return (f"{len(files)} files: actions == knot1/line(t+dt) - measured (max {worst:.1e}), "
            f"sim_latent (T, 16); {loader}")


@check("H6 hygiene")
def check_h6(ctx: SimpleNamespace) -> str:
    for pid in ctx.pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f"pid {pid} still alive")
    left = subprocess.run(["pgrep", "-f", PRIVATE_PORT], capture_output=True, text=True,
                          check=False).stdout.split()
    _require(not left, f"processes on {PRIVATE_PORT}: {left}")
    _require(_git_porcelain(MAIN_MAGNA) == ctx.main_porcelain, "magna main checkout changed")
    now = _tree_digest(sim_snapshot.DEFAULT_START_STATE_DIR)
    changed = [k for k, v in ctx.start_states.items() if now.get(k) != v]
    _require(not changed, f"start states changed: {changed}")
    gpu = _gpu_apps()
    note = "nvidia-smi unavailable"
    if gpu is not None and ctx.gpu_baseline is not None:
        extra = gpu - ctx.gpu_baseline - {os.getpid()}
        _require(not (extra & set(ctx.pids)), f"our pids on the GPU: {extra & set(ctx.pids)}")
        note = "GPU apps at baseline" + (f" (foreign appeared: {sorted(extra)})" if extra else "")
    return (f"H0 pids {ctx.pids} gone, pgrep -f {PRIVATE_PORT} empty, {note}; magna main "
            f"status unchanged; {len(ctx.start_states)} start-state files unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the temp dir (runs + logs)")
    parser.add_argument("--lcm-url", default=PRIVATE_LCM_URL)
    args = parser.parse_args()
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="WARNING", format="{level: <7} | {message}")

    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_mpc_harness_"))
    ctx = SimpleNamespace(tmp_root=tmp_root, lcm_url=args.lcm_url, gpu_baseline=_gpu_apps(),
                          pids=[], main_porcelain=_git_porcelain(MAIN_MAGNA),
                          start_states=_tree_digest(sim_snapshot.DEFAULT_START_STATE_DIR))
    exit_code = 0
    for name, fn in CHECKS:
        try:
            detail = fn(ctx)
        except Skip as exc:
            print(f"[SKIP] {name}: {exc}", flush=True)
            continue
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            exit_code = 1
            if name.startswith("H6"):
                break
            continue
        except Exception as exc:  # noqa: BLE001 - report, then still clean up
            print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
            traceback.print_exc()
            exit_code = 1
            continue
        print(f"[PASS] {name}: {detail}", flush=True)

    if args.keep or exit_code:
        print(f"[INFO] kept {tmp_root}")
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)
    runtime = time.perf_counter() - t0
    if exit_code == 0:
        if runtime > RUNTIME_BUDGET_S:
            print(f"[WARN] runtime {runtime:.1f} s > {RUNTIME_BUDGET_S:g} s budget")
        print(f"ALL MPC HARNESS CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
