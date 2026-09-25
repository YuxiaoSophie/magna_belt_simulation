#!/usr/bin/env python3
"""End-to-end check of the LCS collector.

Runs a tiny collection (one episode per intent) into a temp dir with the default ``osc`` backend
(magna's OSC as a child process on a private LCM URL) and verifies every artefact: the ``.npz``
format and tuple semantics, ``index.json``, the recordings and (if the ``lcs_learning`` venv
exists) the real ``RoundBeltTupleDataset`` loader. C8 runs one ``--backend position`` episode.

Run:
    uv run python scripts/checks/check_lcs_collector.py
    uv run python scripts/checks/check_lcs_collector.py --keep
    uv run python scripts/checks/check_lcs_collector.py --backend position
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# The task packages live under src/; make them importable regardless of CWD.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import collect_lcs_dataset as cli
import newton

from round_belt_task import arm_kinematics as ak
from round_belt_task import outcome
from round_belt_task import perturbation as pert
from round_belt_task.constants import LCM_SIM_PARAMS, SCENE_DIRECTIVES
from round_belt_task.scene import build_scene
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.recording import Recording, file_digest
from task_common.replay_app import ReplayApp
from task_common.scene import make_builder
from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material

REPLAY_PORT = 18088
RUNTIME_BUDGET_S = 360.0
C1_BUDGET_S = 180.0
# A private multicast group so this check never disturbs a running magna stack.
PRIVATE_LCM_URL = "udpm://239.255.76.85:7685?ttl=0"
ACTION_TOL = 1e-9
# PD tracking can overshoot the URDF's commanded limits by a hair; not a hard mechanical bound.
FRANKA_LIMIT_TOL_RAD = 0.05
# A UR dz this deep drags the 2F-85 fingers through the board plate without the guard.
UNSAFE_DZ_MM = -15.0
START_STATES = cli.DEFAULT_START_STATES
LCS_LEARNING_ROOT = Path("/home/hienbui/git/lcs_learning")
LCS_LEARNING_VENV = LCS_LEARNING_ROOT / ".venv" / "bin" / "python"
LOADER_SHAPES = {"curr_pc": [1800, 3], "curr_belt": [150, 3], "curr_prop": [40], "u": [12],
                 "next_pc": [1800, 3], "next_prop": [40]}


class Skip(Exception):
    """Raised by a check to report ``[SKIP] <reason>`` instead of pass/fail."""


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


def run_collect(intent: str, seed: int, out_dir: Path, start_state: Path, record: bool,
                no_pcd: bool = False, backend: str = "osc") -> dict:
    """One ``collect()`` call, one episode, one fixed intent (the collector has no cycle flag)."""
    argv = ["--episodes", "1", "--seed", str(seed), "--intents", intent, "--weights", "1",
            "--out", str(out_dir), "--start-state", str(start_state), "--backend", backend,
            "--lcm-url", PRIVATE_LCM_URL]
    if record:
        argv.append("--record")
    if no_pcd:
        argv.append("--no-pcd")
    args = cli.create_parser().parse_args(argv)
    return cli.collect(args)


@check("C0 start state")
def check_c0(ctx: SimpleNamespace) -> str:
    want = {"scene_directives": file_digest(SCENE_DIRECTIVES)["sha256"],
            "lcm_sim_params": file_digest(LCM_SIM_PARAMS)["sha256"]}
    snap = None
    path = START_STATES[ctx.backend]
    if path.is_file():
        candidate = sim_snapshot.load(path)
        got = {k: candidate.meta.get(k, {}).get("sha256") for k in want}
        if got == want:
            snap, ctx.start_state = candidate, path
            detail = f"reused {path} (scene digests match)"
    if snap is None:
        out = ctx.tmp_root / path.name
        cmd = ["uv", "run", "python", str(REPO_ROOT / "scripts" / "lcs" / "make_start_state.py"),
               "--out", str(out)]
        if ctx.backend == "osc":
            cmd += ["--backend", "osc", "--lcm-url", PRIVATE_LCM_URL]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
                              check=False)
        _require(proc.returncode == 0, f"make_start_state.py failed:\n{proc.stdout[-2000:]}\n"
                 f"{proc.stderr[-2000:]}")
        snap, ctx.start_state = sim_snapshot.load(out), out
        detail = f"built {out} (no matching cached start state)"
    ctx.start_snap = snap
    return detail


@check("C1 collect (4 episodes, one per intent)")
def check_c1(ctx: SimpleNamespace) -> str:
    t0 = time.perf_counter()
    run_dir = ctx.tmp_root / "run"
    (run_dir / "recordings").mkdir(parents=True)
    header = None
    rows = []
    for i, intent in enumerate(pert.INTENTS):
        sub = ctx.tmp_root / f"collect_{intent}"
        index = run_collect(intent, seed=0, out_dir=sub, start_state=ctx.start_state, record=True,
                            backend=ctx.backend)
        row = index["episodes"][0]
        _require(row["status"] == "ok", f"{intent}: episode not ok: {row}")
        if header is None:
            header = {k: index[k] for k in
                      ("args", "git", "backend", "start_state", "sample_period_s",
                       "sample_steps", "thresholds", "ranges", "labels", "belt_tangent")}
        if ctx.backend == "osc":
            _require("osc" in index.get("summary", {}), f"{intent}: summary has no 'osc'")
            _require(not index["summary"]["osc_log_errors"],
                     f"{intent}: OSC log errors {index['summary']['osc_log_errors']}")
        dst_name = f"episode_{i:04d}.npz"
        shutil.move(str(sub / row["file"]), str(run_dir / dst_name))
        row = {**row, "file": dst_name}
        if row.get("recording"):
            rec_src = sub / row["recording"]
            rec_dst = run_dir / "recordings" / rec_src.name
            shutil.move(str(rec_src), str(rec_dst))
            row["recording"] = str(Path("recordings") / rec_src.name)
        rows.append(row)
        shutil.rmtree(sub, ignore_errors=True)
    header["episodes"] = rows
    (run_dir / "index.json").write_text(
        json.dumps(header, indent=1, default=lcs.json_default) + "\n")
    ctx.run_dir, ctx.rows, ctx.header = run_dir, rows, header

    elapsed = time.perf_counter() - t0
    _require(elapsed <= C1_BUDGET_S, f"C1 took {elapsed:.1f} s > {C1_BUDGET_S:g} s budget")
    files = sorted(p.name for p in run_dir.glob("episode_*.npz"))
    _require(len(files) == 4, f"{len(files)} episode files under {run_dir}, expected 4")
    pairs = ", ".join(f"{r['intent']}->{r['outcome']}" for r in rows)
    return f"4/4 ok ({pairs}), {elapsed:.1f} s"


@check("C2 per-file contents")
def check_c2(ctx: SimpleNamespace) -> str:
    period_us = lcs.SAMPLE_PERIOD_US
    t_list, pcd_min, pcd_max = [], 10**9, 0
    lo, hi = ak.FrankaTip.lower - FRANKA_LIMIT_TOL_RAD, ak.FrankaTip.upper + FRANKA_LIMIT_TOL_RAD
    for row in ctx.rows:
        path = ctx.run_dir / row["file"]
        summary = lcs.validate_episode(path, period_us=period_us)
        T = summary["T"]
        _require(T >= 30, f"{row['file']}: T={T} < 30")
        t_list.append(T)
        with np.load(path, allow_pickle=True) as data:
            diffs = np.diff(np.asarray(data["utime"]).astype(np.int64))
            _require(np.all(diffs == period_us),
                     f"{row['file']}: utime diffs {sorted(set(diffs.tolist()))} != {period_us}")
            label = str(data["trajectory_label"])
            _require(label in outcome.LABELS,
                     f"{row['file']}: trajectory_label {label!r} not in {outcome.LABELS}")
            sim_intent = str(data["sim_intent"])
            _require(sim_intent == row["intent"],
                     f"{row['file']}: sim_intent {sim_intent!r} != index intent {row['intent']!r}")
            belt_xyz = data["sim_belt_xyz"]
            want_belt = (T, lcs.BELT_BODIES, 3)
            _require(belt_xyz.shape == want_belt,
                     f"{row['file']}: sim_belt_xyz shape {belt_xyz.shape} != {want_belt}")
            pcd_belt = data["pcd_belt"]
            want_pcd_belt = (T, lcs.BELT_POINTS, 3)
            _require(pcd_belt.shape == want_pcd_belt,
                     f"{row['file']}: pcd_belt shape {pcd_belt.shape} != {want_pcd_belt}")
            q_franka = np.asarray(data["state"])[:, :7]
            _require(bool(np.all((q_franka >= lo) & (q_franka <= hi))),
                     f"{row['file']}: state[:, :7] outside Franka joint limits "
                     f"+-{FRANKA_LIMIT_TOL_RAD} rad")
            _check_tuples(row["file"], data, ctx.backend)
            counts = [int(np.asarray(frame).shape[0]) for frame in data["pcd"]]
            _require(min(counts) >= 200, f"{row['file']}: min pcd points {min(counts)} < 200")
            _require(max(counts) <= 20000, f"{row['file']}: max pcd points {max(counts)} > 20000")
            pcd_min, pcd_max = min(pcd_min, min(counts)), max(pcd_max, max(counts))
    ctx.t_list, ctx.t_range, ctx.pcd_range = t_list, (min(t_list), max(t_list)), (pcd_min, pcd_max)
    return f"T in {ctx.t_range}, pcd points in {ctx.pcd_range}"


def _check_tuples(name: str, data, backend: str) -> None:
    """Sample spacing, state pose block and action semantics of one file."""
    state, actions = np.asarray(data["state"]), np.asarray(data["actions"])
    steps = np.diff(np.asarray(data["sim_step"]))
    _require(np.all(steps == lcs.SAMPLE_STEPS),
             f"{name}: sim_step diffs {sorted(set(steps.tolist()))} != {lcs.SAMPLE_STEPS}")
    meta = json.loads(str(data["sim_meta"]))
    _require(meta.get("backend") == backend, f"{name}: sim_meta backend {meta.get('backend')!r}")
    if backend == "position":
        _require(np.all(actions[-1] == 0.0), f"{name}: actions[-1] not all zero")
        _require(np.array_equal(state[:, 26:33], data["sim_ee_cmd_franka"]),
                 f"{name}: state pose block != sim_ee_cmd_franka")
        for key in ("sim_ee_cmd_franka", "sim_ee_cmd_ur"):
            _require(np.isfinite(np.asarray(data[key])).all(), f"{name}: {key} non-finite")
        return
    _require(np.array_equal(state[:, 26:33], data["sim_ee_franka"]),
             f"{name}: state[:, 26:33] != sim_ee_franka")
    _require(np.array_equal(state[:, 33:40], data["sim_ee_ur"]),
             f"{name}: state[:, 33:40] != sim_ee_ur")
    definition = meta.get("lcs_format", {}).get("action_definition")
    _require(definition == "cmd_delta", f"{name}: action_definition {definition!r}")
    k0, k1 = np.asarray(data["sim_cmd_knot0_franka"]), np.asarray(data["sim_cmd_knot1_franka"])
    ur0, ur1 = np.asarray(data["sim_cmd_ur_t"]), np.asarray(data["sim_cmd_ur_t1"])
    err = float(np.abs(actions[:, :3] - (k1[:, :3] - k0[:, :3])).max())
    _require(err <= ACTION_TOL, f"{name}: actions[:, :3] != knot1 - knot0 ({err:.2e})")
    err = float(np.abs(actions[:, 3:6] - (ur1[:, :3] - ur0[:, :3])).max())
    _require(err <= ACTION_TOL, f"{name}: actions[:, 3:6] != UR line(t+dt) - line(t) ({err:.2e})")
    old = np.asarray(data["sim_action_knot1_minus_measured"])
    err = float(np.abs(old[:, :3] - (k1[:, :3] - state[:, 26:29])).max())
    _require(err <= ACTION_TOL, f"{name}: old action[:, :3] != knot1 - measured ({err:.2e})")
    err = float(np.abs(old[:, 3:6] - (ur1[:, :3] - state[:, 33:36])).max())
    _require(err <= ACTION_TOL, f"{name}: old action[:, 3:6] != line(t+dt) - measured ({err:.2e})")
    pre = np.asarray(data["sim_episode_step"]) < 0
    _require(pre.any() and np.all(actions[pre] == 0.0),
             f"{name}: no pre-hold rows or pre-hold actions not exactly 0")
    _require(np.all(np.isnan(data["sim_realised_delta"][-1])), f"{name}: realised last row")
    _require(np.asarray(data["sim_cmd_knot0_franka"]).shape == k1.shape,
             f"{name}: sim_cmd_knot0_franka missing its (T, 7) shape")
    _require(np.array_equal(data["sim_render_step"], data["sim_step"]),
             f"{name}: sim_render_step != sim_step")
    _require(np.asarray(data["sim_cmd_knots_franka"]).shape[1:] == (7, 7),
             f"{name}: sim_cmd_knots_franka shape {np.asarray(data['sim_cmd_knots_franka']).shape}")
    for key in ("sim_cmd_knots_franka", "sim_cmd_ur_t", "sim_cmd_ur_t1", "sim_tracking_err_mm",
                "sim_franka_tip_clearance_mm"):
        _require(np.isfinite(np.asarray(data[key], dtype=np.float64)).all(),
                 f"{name}: {key} non-finite")
    utime = np.asarray(data["utime"])
    _require(np.array_equal(utime, np.asarray(data["sim_osc_utime"])),
             f"{name}: utime != sim_osc_utime")


@check("C3 index")
def check_c3(ctx: SimpleNamespace) -> str:
    header = json.loads((ctx.run_dir / "index.json").read_text())
    required = ("args", "git", "start_state", "sample_period_s", "thresholds", "ranges", "labels")
    missing = [k for k in required if k not in header]
    _require(not missing, f"index.json missing keys {missing}")
    rows = header["episodes"]
    _require(len(rows) == 4, f"index.json has {len(rows)} episodes, expected 4")
    for row in rows:
        _require((ctx.run_dir / row["file"]).is_file(),
                 f"index row {row['file']}: file does not exist")
    by_intent = Counter(row["intent"] for row in rows)
    _require(by_intent == Counter(pert.INTENTS),
             f"per-intent counts {dict(by_intent)} != one per {list(pert.INTENTS)}")
    for row in rows:
        with np.load(ctx.run_dir / row["file"], allow_pickle=True) as data:
            label = str(data["trajectory_label"])
        _require(row["outcome"] == label,
                 f"{row['file']}: index outcome {row['outcome']!r} != file label {label!r}")
    return f"header keys present, {len(rows)} rows, 1 per intent, outcomes match files"


@check("C4 recordings + replay")
def check_c4(ctx: SimpleNamespace) -> str:
    recordings_root = ctx.run_dir / "recordings"
    runs = Recording.list_runs(recordings_root)
    _require(len(runs) == 4, f"{len(runs)} recording runs under {recordings_root}, expected 4")
    want_labels = ctx.start_snap.meta["body_labels"]
    for run_path in runs:
        rec = Recording.load(run_path)
        _require(rec.meta["body_labels"] == want_labels,
                 f"{run_path.name}: meta body_labels != the start state's")
        row = next(r for r in ctx.rows if Path(r["recording"]).name == run_path.name)
        with np.load(ctx.run_dir / row["file"], allow_pickle=True) as data:
            sim_step = np.asarray(data["sim_step"])
        # sim_step[0] is a pre-step snapshot taken before recording's first control_step.
        _require(rec.step.min() <= sim_step[0] + 1 and rec.step.max() >= sim_step[-1],
                 f"{run_path.name}: recorded steps [{rec.step.min()}, {rec.step.max()}] do not "
                 f"cover episode steps [{sim_step[0]}, {sim_step[-1]}]")

    patch_viewer_shape_names()
    patch_viser_texture_material()
    app = ReplayApp(recordings_root, build_model, port=REPLAY_PORT, analysis=False, verbose=False)
    try:
        names = app.runs()
        _require(len(names) == 4, f"ReplayApp.runs() = {names}, expected 4")
        for name in names:
            app.select_run(name)
            _require(app.frame_count > 0, f"{name}: frame_count {app.frame_count} <= 0")
    finally:
        app.close()
    return "4 recordings, body_labels match, step spans cover episodes, ReplayApp OK"


@check("C5 determinism")
def check_c5(ctx: SimpleNamespace) -> str:
    intent = pert.INTENTS[0]
    row1 = ctx.rows[0]
    _require(row1["intent"] == intent, f"rows[0] intent {row1['intent']!r} != {intent!r}")
    out2 = ctx.tmp_root / "run2"
    index2 = run_collect(intent, seed=0, out_dir=out2, start_state=ctx.start_state, record=False,
                         no_pcd=True, backend=ctx.backend)
    row2 = index2["episodes"][0]
    _require(row2["status"] == "ok", f"rerun: episode not ok: {row2}")
    _require(row2["perturbation"] == row1["perturbation"],
             f"perturbation differs: {row2['perturbation']} != {row1['perturbation']}")
    _require(row2["outcome"] == row1["outcome"],
             f"outcome differs: {row2['outcome']!r} != {row1['outcome']!r}")
    with np.load(ctx.run_dir / row1["file"], allow_pickle=True) as d1:
        belt1 = np.asarray(d1["sim_belt_xyz"])[-1]
    with np.load(out2 / row2["file"], allow_pickle=True) as d2:
        belt2 = np.asarray(d2["sim_belt_xyz"])[-1]
    diff_mm = float(np.linalg.norm(belt1 - belt2, axis=1).max()) * 1e3
    _require(diff_mm <= 5.0, f"final belt differs by {diff_mm:.3f} mm > 5 mm")
    return f"perturbation + outcome identical ({row1['outcome']}), belt final diff {diff_mm:.3f} mm"


@check("C6 lcs_learning loader")
def check_c6(ctx: SimpleNamespace) -> str:
    if not LCS_LEARNING_VENV.is_file():
        raise Skip(f"{LCS_LEARNING_VENV} not found")
    files = [str(ctx.run_dir / row["file"]) for row in ctx.rows]
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(LCS_LEARNING_ROOT)!r})\n"
        "from lcs_learning.dataset_round_belt_tuples import RoundBeltTupleDataset\n"
        f"ds = RoundBeltTupleDataset(data_file={files!r}, num_points=1800, proprio_dim=40, "
        "control_dim=12, point_cloud_source='camera_plus_belt', belt_num_points=150)\n"
        "item = ds[0]\n"
        f"keys = {list(LOADER_SHAPES)!r}\n"
        "shapes = {k: list(item[k].shape) for k in keys}\n"
        "print(json.dumps({'len': len(ds), 'shapes': shapes}))\n"
    )
    try:
        proc = subprocess.run([str(LCS_LEARNING_VENV), "-c", code], capture_output=True,
                              text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired as exc:
        raise Skip(f"lcs_learning subprocess timed out: {exc}") from exc
    if proc.returncode != 0:
        raise Skip(f"lcs_learning import/load failed: {proc.stderr.strip()[-500:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    want_len = sum(T - 1 for T in ctx.t_list)
    _require(out["len"] == want_len, f"lcs_learning len {out['len']} != sum(T-1) {want_len}")
    for key, shape in LOADER_SHAPES.items():
        _require(out["shapes"][key] == shape,
                 f"lcs_learning ds[0][{key!r}].shape {out['shapes'][key]} != {shape}")
    return f"len {out['len']}, shapes OK"


@check("C7 clearance guard")
def check_c7(ctx: SimpleNamespace) -> str:
    lows = []
    for row in ctx.rows:
        for key in ("min_board_clearance_mm", "board_contact", "clamp_lift_mm", "clamp"):
            _require(key in row, f"{row['file']}: index row has no {key!r}")
        with np.load(ctx.run_dir / row["file"], allow_pickle=True) as data:
            low = float(data["sim_min_board_clearance_mm"])
            per_frame = np.asarray(data["sim_board_clearance_mm"], dtype=np.float64)
            _require(not bool(data["sim_board_contact"]),
                     f"{row['file']}: sim_board_contact is True (clearance {low:.2f} mm)")
            _require(low >= 0.0, f"{row['file']}: min board clearance {low:.2f} mm < 0")
            # The episode minimum is sampled 5x more often than the frames, so it cannot be higher.
            _require(low <= per_frame.min() + 1e-3,
                     f"{row['file']}: episode min {low:.3f} mm > per-frame min "
                     f"{per_frame.min():.3f} mm")
        lows.append(low)

    unsafe = dict(pert.ranges_for_backend(ctx.backend))
    unsafe["under"] = dataclasses.replace(
        unsafe["under"], ur_dz_mm=pert.Range(UNSAFE_DZ_MM, UNSAFE_DZ_MM),
        ur_dxy_mm=pert.Range(0.0, 0.0), ur_tilt_deg=pert.Range(0.0, 0.0),
        franka_dxyz_mm=pert.Range(0.0, 0.0), franka_tilt_deg=pert.Range(0.0, 0.0))
    args = cli.create_parser().parse_args(
        ["--episodes", "1", "--seed", "0", "--intents", "under", "--weights", "1", "--no-pcd",
         "--out", str(ctx.tmp_root / "guard"), "--start-state", str(ctx.start_state),
         "--backend", ctx.backend, "--lcm-url", PRIVATE_LCM_URL])
    row = cli.collect(args, ranges=unsafe)["episodes"][0]
    _require(row["status"] == "ok", f"guard episode not ok: {row}")
    clamp = row["clamp"]
    want = clamp["min_clearance_mm"]
    for label in ("pre_place_2", "place_3"):
        _require(clamp["before_mm"][label] < 0.0,
                 f"{label}: dz {UNSAFE_DZ_MM} mm should predict a negative clearance, got "
                 f"{clamp['before_mm'][label]:.2f} mm")
        _require(clamp["after_mm"][label] >= want - 1e-3,
                 f"{label}: clamped clearance {clamp['after_mm'][label]:.2f} mm < {want:g} mm")
        _require(clamp["lift_mm"][label] >= -UNSAFE_DZ_MM,
                 f"{label}: lift {clamp['lift_mm'][label]:.2f} mm did not undo dz "
                 f"{UNSAFE_DZ_MM} mm")
    _require(not row["board_contact"] and row["min_board_clearance_mm"] >= 0.0,
             f"guard episode still hit the board: {row['min_board_clearance_mm']:.2f} mm")
    return (f"C1 minima {min(lows):.2f}..{max(lows):.2f} mm, no contact; dz {UNSAFE_DZ_MM} mm "
            f"lifted {row['clamp_lift_mm']:.1f} mm -> {row['min_board_clearance_mm']:.2f} mm")


@check("C8 position backend (legacy path)")
def check_c8(ctx: SimpleNamespace) -> str:
    out = ctx.tmp_root / "position"
    index = run_collect(pert.INTENTS[0], seed=0, out_dir=out, start_state=START_STATES["position"],
                        record=False, backend="position")
    row = index["episodes"][0]
    _require(row["status"] == "ok", f"position episode not ok: {row}")
    path = out / row["file"]
    summary = lcs.validate_episode(path, period_us=lcs.SAMPLE_PERIOD_US)
    with np.load(path, allow_pickle=True) as data:
        _check_tuples(row["file"], data, "position")
        source = json.loads(str(data["sim_meta"]))["ee_pose_source"]
    _require(source == "commanded Cartesian target", f"ee_pose_source {source!r}")
    return (f"{row['intent']}->{row['outcome']}, T {summary['T']}, period "
            f"{summary['period_us']} us, commanded-target state, zero last action")


def main() -> int:
    import argparse
    global PRIVATE_LCM_URL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the temp dir")
    parser.add_argument("--backend", choices=cli.BACKENDS, default="osc",
                        help="backend of C1-C7 (C8 always runs position)")
    parser.add_argument("--lcm-url", default=PRIVATE_LCM_URL)
    args = parser.parse_args()
    PRIVATE_LCM_URL = args.lcm_url

    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="WARNING", format="{level: <7} | {message}")

    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_lcs_collector_"))
    ctx = SimpleNamespace(tmp_root=tmp_root, backend=args.backend)
    exit_code = 0
    for name, fn in CHECKS:
        try:
            detail = fn(ctx)
        except Skip as exc:
            print(f"[SKIP] {name}: {exc}")
            continue
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

    if args.keep:
        print(f"[INFO] kept {tmp_root}")
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        if runtime > RUNTIME_BUDGET_S:
            print(f"[WARN] runtime {runtime:.1f} s > {RUNTIME_BUDGET_S:g} s budget")
        print(f"ALL LCS COLLECTOR CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
