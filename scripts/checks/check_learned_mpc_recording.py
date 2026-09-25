#!/usr/bin/env python3
"""Live check that recorded learned-MPC episodes carry the controller's own per-solve plan.

Runs the harness (``scripts/lcs/eval_learned_mpc.py --record``) twice on a private URL: a learned
episode with ``learned_mpc.debug_channel`` set and a baseline episode, while this process
listens on the same URL. R0 ``targets.jsonl`` channels and one ``LEARNED_MPC_DEBUG`` message per
MPC-window latent (same utimes, same count on the wire); R1 the message matches what was
executed (``u_sol[:, 0]`` == the published Franka knot 1 - knot 0 / UR line end - latent pose,
``x_sol`` column 0 == the latent, the augmentation, the demo references, ``k0``); R2
``meta.json`` ``learned_mpc`` (sha256s); R3 decoded planned belts (``[SKIP]`` without
``decoder.npz``); R4 ``Recording.load`` + ``replay_metrics``; R5 hygiene.

Run:
    uv run python scripts/checks/check_learned_mpc_recording.py [--keep]
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "checks",
           REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lcm
from private_url import pgrep_others, url_port

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from dairlib import lcmt_timestamped_saved_traj
from round_belt_task.commander import pose_mat, x_tool0_tracking
from round_belt_task.controller_bridge import ur_base_world
from task_common import lcs_dataset as lcs
from task_common import replay_metrics as rm
from task_common.latent_encoder import LatentDecoder
from task_common.lcm_contract import LcmChannels
from task_common.magna_process import CONTROLLER_REL, MAGNA_WORKTREE
from task_common.recording import Recording

PRIVATE_LCM_URL = "udpm://239.255.76.105:7705?ttl=0"
MAIN_MAGNA = Path("/home/hienbui/git/magna")
DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v2_20260925/"
              "deploy_v2_decoded_only/deploy.npz")
# R1 checks demo_traj references, which only this archived config has.
PARAMS = Path("systems/parameters/learned_archive/2026-09-24/learned_tuning/"
              "round_belt_controller_params_learned_eval_v2_honor_wr0p3_wp0p03_debug.yaml")
DEMO_GOALS = REPO_ROOT / "data" / "lcs" / "demo" / "v2" / "demo_goals.npz"
START_SET = REPO_ROOT / "data" / "lcs" / "start_states" / "grasp_variants" / "set1"
VARIANT = "gv_00"
MAX_EPISODE_S = 4.0
RUN_TIMEOUT_S = 300.0
CH = LcmChannels()
FRANKA_TRAJ = CH.tracking_trajectory_actor_channel
UR_TRAJ = CH.ur_tracking_trajectory_actor_channel
DEBUG = CH.learned_mpc_debug_channel
SCALARS = ("k0", "stage", "dist_ref", "dist_final", "solve_ms", "t_mpc_start", "ref_mode",
           "n_x", "N", "solve_ok")
META_FILES = ("params_yaml", "lcs_yaml", "demo_traj_yaml", "deploy", "demo_goals",
              "demo_episode")
# x_sol is C3's QP solution: x_0 and the EE augmentation hold to OSQP's tolerance, not 1e-9.
X0_LATENT_TOL = 1e-3
X0_POS_TOL = 1e-6  # m
AUG_TOL = 1e-4  # m
KNOT_TOL = 1e-12  # m, the Franka knots are built from u_sol
KNOT_ROT_TOL = 1e-9  # rad
UR_END_TOL = 1e-6  # m, python vs C++ UR base / tracking-frame transforms
REF_TOL = 1e-12
DIST_TOL = 1e-9
MAX_ARRIVAL_LAG_S = 0.5
WORKSPACE_BOX = np.array([[0.1, -0.5, -0.1], [0.9, 0.5, 0.4]])  # m, world


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


class Skip(Exception):
    pass


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


class Counter:
    """Counts messages per channel on the private URL from a background thread."""

    def __init__(self, url: str) -> None:
        self.lc = lcm.LCM(url)
        self.counts: dict[str, int] = {}
        self.debug_utimes: list[int] = []
        self.lc.subscribe(".*", self._on)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _on(self, channel: str, data: bytes) -> None:
        self.counts[channel] = self.counts.get(channel, 0) + 1
        if channel == DEBUG:
            self.debug_utimes.append(int(lcmt_timestamped_saved_traj.decode(data).utime))

    def _run(self) -> None:
        while not self._stop.is_set():
            self.lc.handle_timeout(50)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_porcelain(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                          text=True, check=False).stdout


def _run_harness(ctx: SimpleNamespace, name: str, extra: list[str]) -> SimpleNamespace:
    out = ctx.tmp_root / name
    argv = [sys.executable, str(REPO_ROOT / "scripts" / "lcs" / "eval_learned_mpc.py"),
            "--lcm-url", ctx.lcm_url, "--out", str(out), "--record", "--pace", "1",
            "--max-episode-s", str(MAX_EPISODE_S), "--start-states", str(START_SET),
            "--variants", VARIANT, "--demo-goals", str(DEMO_GOALS), *extra]
    with Counter(ctx.lcm_url) as wire, open(ctx.tmp_root / f"{name}.log", "w") as log:
        proc = subprocess.run(argv, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                              timeout=RUN_TIMEOUT_S, check=False)
    text = (ctx.tmp_root / f"{name}.log").read_text(errors="replace")
    _require(proc.returncode == 0, f"{name}: harness exit {proc.returncode}: {text[-1500:]}")
    index = json.loads((out / "index.json").read_text())
    row = index["episodes"][0]
    _require(row.get("recording"), f"{name}: no recording in {row}")
    rec_dir = out / row["recording"]
    targets = [json.loads(line) for line in (rec_dir / "targets.jsonl").read_text().splitlines()]
    with np.load(out / row["file"], allow_pickle=True) as d:
        ep = {k: d[k] for k in ("sim_osc_utime", "sim_latent", "sim_ee_franka", "sim_ee_ur")}
    return SimpleNamespace(out=out, row=row, rec_dir=rec_dir, targets=targets, ep=ep,
                           meta=json.loads((rec_dir / "meta.json").read_text()),
                           wire=dict(wire.counts), wire_debug=list(wire.debug_utimes))


def _blocks(payload: dict) -> dict[str, np.ndarray]:
    return {k: np.asarray(v["data"], dtype=np.float64) for k, v in payload["blocks"].items()}


def _times(payload: dict, block: str) -> np.ndarray:
    return np.asarray(payload["blocks"][block]["t"], dtype=np.float64)


def _debug(run: SimpleNamespace) -> list[dict]:
    return [t["payload"] for t in run.targets if t["channel"] == DEBUG]


# --- checks -----------------------------------------------------------------------------------

@check("R0 channels + one debug message per latent")
def check_r0(ctx: SimpleNamespace) -> str:
    for path in (MAGNA_WORKTREE / CONTROLLER_REL, MAGNA_WORKTREE / PARAMS, DEPLOY, DEMO_GOALS,
                 START_SET):
        if not path.exists():
            raise Skip(f"{path} absent")
    left = pgrep_others(ctx.port)
    _require(not left, f"processes already on {ctx.port}: {left}")
    ctx.learned = run = _run_harness(ctx, "learned", [
        "--mode", "learned", "--params", str(PARAMS), "--deploy", str(DEPLOY)])
    ctx.baseline = base = _run_harness(ctx, "baseline", ["--mode", "baseline", "--no-pcd"])
    chans = {t["channel"] for t in run.targets}
    _require({FRANKA_TRAJ, UR_TRAJ, DEBUG} <= chans, f"learned targets.jsonl channels {chans}")
    b_chans = {t["channel"] for t in base.targets}
    _require({FRANKA_TRAJ, UR_TRAJ} <= b_chans and DEBUG not in b_chans,
             f"baseline targets.jsonl channels {b_chans}")
    _require(base.wire.get(DEBUG, 0) == 0, f"baseline put {base.wire.get(DEBUG)} {DEBUG} on the wire")
    dbg = _debug(run)
    utimes = [int(p["utime"]) for p in dbg]
    _require(len(set(utimes)) == len(utimes), "duplicate debug utimes")
    _require(utimes == sorted(utimes), "debug utimes out of order")
    lat = [int(u) for u in run.ep["sim_osc_utime"]]
    window = [u for u in lat if utimes[0] <= u <= utimes[-1]]
    _require(window == utimes, f"{len(utimes)} debug messages vs {len(window)} latents in the "
             f"window; missing {sorted(set(window) - set(utimes))[:5]}, extra "
             f"{sorted(set(utimes) - set(window))[:5]}")
    _require(run.wire_debug == utimes, f"wire {len(run.wire_debug)} vs recorded {len(utimes)} "
             f"{DEBUG} messages")
    n_plan = sum(t["channel"] == FRANKA_TRAJ for t in run.targets)
    return (f"learned: {len(utimes)} {DEBUG} == latents {lat.index(utimes[0])}.."
            f"{lat.index(utimes[-1])} of {len(lat)} (utimes equal, == wire count), "
            f"{n_plan} {FRANKA_TRAJ}; baseline: {sorted(b_chans)}, no {DEBUG} on the wire")


@check("R1 debug message == the executed plan")
def check_r1(ctx: SimpleNamespace) -> str:
    run = ctx.learned
    lcs_yaml = yaml.safe_load(Path(run.meta["learned_mpc"]["lcs_yaml"]).read_text())
    demo = yaml.safe_load(Path(run.meta["learned_mpc"]["demo_traj_yaml"]).read_text())
    z_std = np.asarray(lcs_yaml["z_std"], dtype=np.float64)
    demo_z = np.asarray(demo["z"], dtype=np.float64)
    demo_f = np.asarray(demo["ee_pose_franka"], dtype=np.float64)[:, :3]
    demo_u = np.asarray(demo["ee_pose_ur"], dtype=np.float64)[:, :3]
    goal = demo_z[-1]
    nz, n_frames = z_std.shape[0], demo_z.shape[0]
    lat = {int(u): i for i, u in enumerate(run.ep["sim_osc_utime"])}
    plans = {}
    lines = {}
    for t in run.targets:
        b = t["payload"].get("blocks", {}).get("end_effector_position_target")
        if not b or not b["t"]:
            continue
        key = round(b["t"][0] * 1e6)
        if t["channel"] == FRANKA_TRAJ:
            plans.setdefault(key, t["payload"])
        elif t["channel"] == UR_TRAJ and len(b["t"]) == 2:
            lines.setdefault(key, t["payload"])
    X_W_base, X_t0_tr = ur_base_world(), x_tool0_tracking()
    err = dict.fromkeys(("x0_z", "x0_p", "aug", "knot", "knot_rot", "ur_end", "ref", "dist"), 0.0)
    k0_prev, n_ur, dt = -1, 0, None
    for p in _debug(run):
        B = _blocks(p)
        names = p["blocks"]["scalars"]["datatypes"]
        _require(tuple(names) == SCALARS, f"scalars rows {names}")
        s = dict(zip(SCALARS, B["scalars"][:, 0], strict=True))
        n_x, N = int(s["n_x"]), int(s["N"])
        shapes = {k: B[k].shape for k in B}
        want = {"x_sol": (n_x, N + 1), "u_sol": (12, N), "z_ref": (nz, N + 1),
                "p_ref": (6, N + 1), "scalars": (len(SCALARS), 1)}
        _require(shapes == want, f"block shapes {shapes} != {want}")
        _require(all(np.isfinite(v).all() for v in B.values()), "non-finite debug values")
        _require(s["solve_ok"] == 1.0 and s["ref_mode"] == 1.0 and n_x == nz + 6,
                 f"scalars {s}")
        T = _times(p, "x_sol")
        dt = float(T[1] - T[0])
        _require(np.allclose(np.diff(T), dt, atol=1e-9) and np.array_equal(_times(p, "u_sol"),
                 T[:N]) and _times(p, "scalars")[0] == T[0], "knot times")
        j = lat[int(p["utime"])]
        z, ef, eu = (run.ep[k][j] for k in ("sim_latent", "sim_ee_franka", "sim_ee_ur"))
        X, U = B["x_sol"], B["u_sol"]
        err["x0_z"] = max(err["x0_z"], float(np.abs(X[:nz, 0] - z).max()))
        err["x0_p"] = max(err["x0_p"], float(np.abs(X[nz:, 0] - np.r_[ef[:3], eu[:3]]).max()))
        err["aug"] = max(err["aug"], float(np.abs(np.diff(X[nz:], axis=1) - U[:6]).max()))
        k0 = int(s["k0"])
        _require(k0 >= k0_prev, f"k0 {k0} < previous {k0_prev} (progress mode)")
        k0_prev = k0
        ks = np.minimum(k0 + np.arange(N + 1), n_frames - 1)
        err["ref"] = max(err["ref"], float(np.abs(B["z_ref"] - demo_z[ks].T).max()),
                         float(np.abs(B["p_ref"] - np.r_[demo_f[ks].T, demo_u[ks].T]).max()))
        d_ref = float(np.linalg.norm((z - B["z_ref"][:, 0]) / z_std))
        d_fin = float(np.linalg.norm((z - goal) / z_std))
        err["dist"] = max(err["dist"], abs(d_ref - s["dist_ref"]), abs(d_fin - s["dist_final"]))
        key = round(T[0] * 1e6)
        plan = plans.get(key)
        _require(plan is not None, f"no {FRANKA_TRAJ} with t0 {T[0]:.4f}")
        P = _blocks(plan)
        pos, quat = P["end_effector_position_target"], P["end_effector_orientation_target"]
        _require(np.array_equal(pos[:, 0], ef[:3]), "Franka knot 0 != the latent's position")
        err["knot"] = max(err["knot"], float(np.abs(pos[:, 1] - pos[:, 0] - U[0:3, 0]).max()))
        drot = lcs.delta_rotvec(quat[:, 0], quat[:, 1])
        err["knot_rot"] = max(err["knot_rot"], float(np.abs(drot - U[6:9, 0]).max()))
        line = lines.get(key)
        if line is not None:
            L = _blocks(line)
            lt = _times(line, "end_effector_position_target")
            _require(abs(lt[1] - lt[0] - dt) <= 1e-6, f"UR line span {lt[1] - lt[0]} != dt")
            X1 = X_W_base @ pose_mat(L["end_effector_position_target"][:, 1],
                                     L["end_effector_orientation_target"][:, 1]) @ X_t0_tr
            err["ur_end"] = max(err["ur_end"], float(np.abs(X1[:3, 3] - eu[:3] - U[3:6, 0]).max()))
            n_ur += 1
    n = len(_debug(run))
    _require(n_ur == n, f"UR lines matched {n_ur} of {n} solves")
    tols = {"x0_z": X0_LATENT_TOL, "x0_p": X0_POS_TOL, "aug": AUG_TOL, "knot": KNOT_TOL,
            "knot_rot": KNOT_ROT_TOL, "ur_end": UR_END_TOL, "ref": REF_TOL, "dist": DIST_TOL}
    bad = {k: f"{err[k]:.2e} > {tols[k]:.0e}" for k in tols if not err[k] <= tols[k]}
    _require(not bad, f"exactness: {bad}")
    ctx.dt = dt
    return (f"{n} solves (N {N}, n_x {n_x}, dt {dt:g}): u0 vs Franka knot1-knot0 "
            f"{err['knot']:.1e} m / rot {err['knot_rot']:.1e} rad, vs UR line end - latent "
            f"{err['ur_end']:.1e} m; x_sol[:, 0] vs latent {err['x0_z']:.1e} / EE "
            f"{err['x0_p']:.1e} m, augmentation {err['aug']:.1e} m (QP tolerance); z_ref/p_ref "
            f"== demo[k0..k0+N] {err['ref']:.1e}; dist_ref/final {err['dist']:.1e}; k0 "
            f"non-decreasing to {k0_prev}")


@check("R2 meta.json learned_mpc")
def check_r2(ctx: SimpleNamespace) -> str:
    m = ctx.learned.meta["learned_mpc"]
    want = {"mode", "action_definition", "debug_channel", "start_state", *META_FILES,
            *(f"{k}_sha256" for k in META_FILES)}
    _require(want <= set(m), f"missing keys {want - set(m)}")
    _require(m["mode"] == "learned" and m["debug_channel"] == DEBUG
             and m["action_definition"] == "cmd_delta", f"learned_mpc {m}")
    _require(m["start_state"]["id"] == VARIANT and Path(m["start_state"]["file"]).name
             == f"{VARIANT}_osc.npz", f"start_state {m['start_state']}")
    for k in META_FILES:
        _require(m[k] is not None and _sha256(Path(m[k])) == m[f"{k}_sha256"],
                 f"{k}: {m[k]} sha256 mismatch")
    _require(Path(m["deploy"]) == DEPLOY.resolve()
             and Path(m["params_yaml"]) == (MAGNA_WORKTREE / PARAMS).resolve(), "paths")
    b = ctx.baseline.meta["learned_mpc"]
    _require(b["mode"] == "baseline" and all(v is None for k, v in b.items() if k != "mode"),
             f"baseline learned_mpc {b}")
    old = {"schema", "argv", "channels", "body_labels", "control_dt", "signal_coords"}
    _require(old <= set(ctx.learned.meta), "existing meta keys missing")
    return (f"{len(want)} keys, {len(META_FILES)} sha256s match their files; baseline "
            f"mode-only (rest null); existing keys intact")


@check("R3 decoded planned belts")
def check_r3(ctx: SimpleNamespace) -> str:
    path = DEPLOY.parent / "decoder.npz"
    if not path.is_file():
        raise Skip(f"{path} absent")
    dec = LatentDecoder.load(path)
    with np.load(DEPLOY) as d:
        _require(dec.checkpoint_sha256 == str(d["checkpoint_sha256"]), "decoder checkpoint")
    zs = np.concatenate([_blocks(p)["x_sol"][:dec.latent_dim].T for p in _debug(ctx.learned)])
    belts = dec.decode_batch(zs)
    _require(belts.shape == (len(zs), dec.num_points, 3) and np.isfinite(belts).all(),
             f"decoded {belts.shape}")
    lo, hi = belts.reshape(-1, 3).min(0), belts.reshape(-1, 3).max(0)
    _require(np.all(lo >= WORKSPACE_BOX[0]) and np.all(hi <= WORKSPACE_BOX[1]),
             f"decoded belt box {lo} .. {hi} outside {WORKSPACE_BOX.tolist()}")
    return (f"{len(zs)} planned latents -> ({dec.num_points}, 3) finite belts in "
            f"[{', '.join(f'{v:.3f}' for v in lo)}] .. [{', '.join(f'{v:.3f}' for v in hi)}] m")


@check("R4 Recording.load + replay_metrics")
def check_r4(ctx: SimpleNamespace) -> str:
    notes = []
    for name in ("learned", "baseline"):
        rec = Recording.load(getattr(ctx, name).rec_dir)
        metrics = rm.compute_metrics(rec)
        rm.derive_events(rec, metrics)
        _require(len(rec.targets) == len(getattr(ctx, name).targets), f"{name}: targets")
        off = int(rec.meta["osc_utime_offset_us"]) * 1e-6
        lag = [t.payload["blocks"]["x_sol"]["t"][0] - (t.step * rec.control_dt + off)
               for t in rec.targets if t.channel == DEBUG]
        # Arrives after its tick: lock-step waits for the OSC, not the solve.
        _require(all(-MAX_ARRIVAL_LAG_S <= v <= 1e-9 for v in lag),
                 f"{name}: debug t0 vs step * dt + osc offset {min(lag, default=0):.4f}..")
        last = rec.frame_count - 1
        step = int(rec.state_step[last])
        got = {label: rm.target_world_pose(rec, ch, step, last)
               for label, ch in rm.target_channels(rec).items()}
        for label in ("Franka EE target (traj)", "UR EE target (traj)"):
            _require(got[label] is not None and np.isfinite(got[label][0]).all(),
                     f"{name}: {label} at the last frame {got[label]}")
        notes.append(f"{name} {len(rec.targets)} targets"
                     + (f" (debug t0 - osc(step) {min(lag):+.3f}..{max(lag):+.3f} s)"
                        if lag else ""))
    return "; ".join(notes) + "; both plan triads resolve at the last frame"


@check("R5 hygiene")
def check_r5(ctx: SimpleNamespace) -> str:
    time.sleep(0.5)
    left = pgrep_others(ctx.port)
    _require(not left, f"processes on {ctx.port}: {left}")
    _require(_git_porcelain(MAIN_MAGNA) == ctx.main_porcelain, "magna main checkout changed")
    _require(_sha256(DEPLOY) == ctx.deploy_sha, "deploy.npz changed")
    return f"pgrep -f {ctx.port} empty; magna main status unchanged; deploy.npz unchanged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the temp dir (runs + logs)")
    parser.add_argument("--lcm-url", default=PRIVATE_LCM_URL)
    args = parser.parse_args()
    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_learned_mpc_recording_"))
    ctx = SimpleNamespace(tmp_root=tmp_root, lcm_url=args.lcm_url, port=url_port(args.lcm_url),
                          main_porcelain=_git_porcelain(MAIN_MAGNA),
                          deploy_sha=_sha256(DEPLOY) if DEPLOY.is_file() else None)
    exit_code = 0
    for name, fn in CHECKS:
        try:
            detail = fn(ctx)
        except Skip as exc:
            print(f"[SKIP] {name}: {exc}", flush=True)
            if name.startswith("R0"):
                break
            continue
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            exit_code = 1
            if name.startswith("R0") and not hasattr(ctx, "baseline"):
                break
            continue
        except Exception as exc:  # noqa: BLE001 - report, then still clean up
            print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
            traceback.print_exc()
            exit_code = 1
            if not hasattr(ctx, "baseline"):
                break
            continue
        print(f"[PASS] {name}: {detail}", flush=True)
    if args.keep or exit_code:
        print(f"[INFO] kept {tmp_root}")
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)
    if exit_code == 0:
        print(f"ALL LEARNED MPC RECORDING CHECKS PASSED ({time.perf_counter() - t0:.1f} s)")
    return exit_code


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
