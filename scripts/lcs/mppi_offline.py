#!/usr/bin/env python3
"""Offline checks, C3 bank comparison, policy screen and a live smoke for the MPPI node.

``--check``: batched torch step/rollout vs ``LearnedLcs`` (float32 vs float64), the ``A_p``/
``E_p`` path, the Franka knot chain vs an independent reference, the UR line round trip through
``ControllerBridge._on_ur_traj``, the debug message through the recorder/replay parsers.
``--bank``: on recorded ``LEARNED_MPC_DEBUG`` solves of a run with the same deploy, C3's
``u_sol`` vs an MPPI plan from the same ``x_0`` under one cost (C3's Q/R, horizon N, w_s 0, the
deploy's bounds). ``--policy-screen``: MPPI ``u_0`` Franka-z / UR-z signs on those states.
``--smoke``: LCM sim + ``latent_encoder_node.py`` + ``mppi_controller_node.py`` on a private URL;
``--override`` keys in ``NODE_KEYS`` (``goal=waypoints:PATH``, ``goal_mode=track``, ...) become
node flags, the rest ``MppiParams`` fields.

    uv run --frozen python scripts/lcs/mppi_offline.py --check --deploy <deploy.npz>
    uv run --frozen python scripts/lcs/mppi_offline.py --bank --policy-screen --deploy <...> \\
        --recordings data/lcs/mpc_eval/20260925-002801-stages --out data/lcs/realfirst/v1/mppi_bank
    uv run --frozen python scripts/lcs/mppi_offline.py --smoke --deploy <...> \\
        --lcm-url 'udpm://239.255.76.162:7762?ttl=0' --out data/lcs/realfirst/v1/mppi_smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "lcs", REPO_ROOT / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lcm
import mppi_controller_node as node_mod
import torch
from private_url import pgrep_others, url_port
from scipy.spatial.transform import Rotation

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from dairlib import lcmt_timestamped_saved_traj
from round_belt_task.commander import (
    METADATA_NAME,
    parse_saved_traj_message,
    pose_mat,
    x_tool0_tracking,
)
from round_belt_task.controller_bridge import ControllerBridge, ur_base_world
from task_common import clearance_model as cm
from task_common.latent_encoder import (
    LATENT_STATE_CHANNEL,
    LearnedLcs,
    parse_latent_state_message,
)
from task_common.lcm_contract import LcmChannels
from task_common.lcm_simulation import LcmBeltTaskSimulation
from task_common.lcs_dataset import delta_rotvec
from task_common.magna_process import (
    CONTROLLER_MARKERS,
    MAGNA_WORKTREE,
    check_private_url,
    parse_stage_line,
)
from task_common.mppi_lcs import (
    Clearance,
    MppiParams,
    MppiSolver,
    TorchLcs,
    mppi_plan,
    rollout_cost,
)
from task_common.recording import TargetMsg
from task_common.replay_learned_mpc import LearnedMpcPanel

DEFAULT_DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v2_20260925/"
                      "deploy_v2_flat_pp2/deploy.npz")
DEFAULT_START_STATE = REPO_ROOT / "data" / "lcs" / "start_states" / "pre_place_1_osc.npz"
DEFAULT_CONFIG = REPO_ROOT / "data" / "lcs" / "realfirst" / "v1" / "mppi.yaml"
V1 = REPO_ROOT / "data" / "lcs" / "realfirst" / "v1"
SEED_EPISODES = (V1 / "seed" / "demo", V1 / "seed" / "excite")
BOARD_ABORT_INDEXES = ("iter*/collect/index.json", "iter*/eval/*/index.json")
BOARD_ABORT = "2F-85 board clearance < 0"
N_CLEAR_POSES = 2000
CLEAR_POS_MM, CLEAR_ROT_DEG = 40.0, 40.0
CLEAR_FULL_TOL_M, CLEAR_PRUNED_TOL_M, CLEAR_TORCH_TOL_M = 0.5e-3, 1.0e-3, 1e-4
C11_MEDIAN_TOL_MM = 1.5
C11_LEAD_STEPS = 60  # 4 frames x 15 steps = 0.3 s
CH = LcmChannels()
FRANKA_CH, UR_CH = CH.tracking_trajectory_actor_channel, CH.ur_tracking_trajectory_actor_channel
DEBUG_CH = CH.learned_mpc_debug_channel
N_CHECK = 200
N_PLANS = 20
KNOT_TOL = 1e-12
ROT_TOL = 1e-9
UR_TOL = 1e-6
SPAN_TOL = 1e-6
BOUND_TOL = 1e-9
FZ, UZ = 2, 5
OVER_H_MM = (3.0, 35.0)  # lever_gate.py "above": stage-2 frames with h 3..35 mm
STARTED_MARKER = CONTROLLER_MARKERS[0]
# The procman executors of docs/lcm-simulation.md (worktree build), private URL each.
EXECUTORS = {"ur_control_simulation": "simulation/ur_control_simulation",
             "ur_trajectory_controller": "controllers/ur_cartesian_trajectory_controller",
             "franka_osc": "controllers/franka_cartesian_osc_controller"}
EXECUTOR_ARGS = {"ur_control_simulation": [], "ur_trajectory_controller": ["--ur_input_mode=1"],
                 "franka_osc": ["--input_mode=1", "--osc_debug_level=warn"]}
SMOKE_MARKERS = (CONTROLLER_MARKERS[0], CONTROLLER_MARKERS[1], "[learned-mpc] stage 0 ",
                 "MPC completed!", "Switching to Terminate")
# --override keys that are node flags (--goal, --goal-mode, ...), not MppiParams fields.
NODE_KEYS = ("goal", "goal_mode", "demo_actions", "demo_offset_frames", "track_hold_factor",
             "track_max_hold_s")
V1_1 = REPO_ROOT / "data" / "lcs" / "realfirst" / "v1_1"
CLEAN_DEMOS = tuple(sorted((V1_1 / "seed" / "demo").glob("episode_*.npz"))
                    + sorted((V1_1 / "seed_ext" / "demo").glob("episode_*.npz")))
GOAL_DEMO = V1_1 / "seed" / "demo" / "episode_0000.npz"
IDENTITY_DIR = REPO_ROOT / "data" / "lcs" / "realfirst" / "cmp" / "offline" / "identity"
IDENTITY_SEEDS = (0, 1, 2)
PRIOR_TOL_HALF = 1e-3


def _overrides(cli) -> dict:
    over = {}
    for item in getattr(cli, "override", []) or []:
        key, _, value = item.partition("=")
        over[key] = yaml.safe_load(value)
    return over


def node_overrides(cli) -> dict:
    """``--override`` items that are node flags (``NODE_KEYS``)."""
    return {k: v for k, v in _overrides(cli).items() if k in NODE_KEYS}


def load_params(cli) -> MppiParams:
    """``--config`` (or defaults) with the ``MppiParams`` ``--override KEY=VALUE`` applied."""
    params = MppiParams.from_yaml(cli.config) if cli.config else MppiParams()
    over = {k: v for k, v in _overrides(cli).items() if k not in NODE_KEYS}
    return dataclasses.replace(params, **over) if over else params


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _payload(channel: str, msg) -> dict:
    """The recorder's JSON form of a message (``_target_payload`` of the LCM sim)."""
    return LcmBeltTaskSimulation._target_payload(SimpleNamespace(channels=CH), channel, msg)


def _blocks(payload: dict) -> dict[str, np.ndarray]:
    return {k: np.asarray(v["data"], dtype=np.float64) for k, v in payload["blocks"].items()}


# --- shared verification of the wire contract ---------------------------------------------------

class UrDecoder:
    """``ControllerBridge._on_ur_traj`` / ``_decode_traj`` on a stub (no sockets)."""

    def __init__(self) -> None:
        self.stub = SimpleNamespace(X_W_base=ur_base_world(), ur_lines=[], undecodable=0,
                                    empty_trajs=0, record_targets=False)
        self.stub._queue_target = lambda ch, msg: None
        self.stub._decode_traj = lambda data, ch=None: ControllerBridge._decode_traj(
            self.stub, data, ch)

    def decode(self, data: bytes):
        n = len(self.stub.ur_lines)
        ControllerBridge._on_ur_traj(self.stub, UR_CH, data)
        return self.stub.ur_lines[-1].line if len(self.stub.ur_lines) > n else None


def executed_u0(franka_msg, ur_line, latent_ur7, X_t0_tr) -> np.ndarray:
    """``u_0`` as the receivers see it: Franka knot1 - knot0, UR line end - latent pose."""
    pos, quat, _ = parse_saved_traj_message(franka_msg)
    X1 = pose_mat(ur_line.p1, ur_line.q1) @ X_t0_tr
    q1 = Rotation.from_matrix(X1[:3, :3]).as_quat(scalar_first=True)
    return np.r_[pos[1] - pos[0], X1[:3, 3] - latent_ur7[:3], delta_rotvec(quat[0], quat[1]),
                 delta_rotvec(latent_ur7[3:7], q1)]


def verify_debug(dbg_msg, franka_msg, ur_line, latent, X_t0_tr) -> dict:
    """Recorder payload -> ``LearnedMpcPanel._index`` + the R1 exactness rules; max errors."""
    _, t, z, ee_f, ee_u = latent
    p_dbg = _payload(DEBUG_CH, dbg_msg)
    p_fr = _payload(FRANKA_CH, franka_msg)
    names = p_dbg["blocks"]["scalars"]["datatypes"]
    _require(tuple(names) == node_mod.SCALARS, f"scalars rows {names}")
    B = _blocks(p_dbg)
    s = dict(zip(names, B["scalars"][:, 0], strict=True))
    n_x, N = int(s["n_x"]), int(s["N"])
    want = {"x_sol": (n_x, N + 1), "u_sol": (12, N), "z_ref": (n_x - 6, N + 1),
            "p_ref": (6, N + 1), "scalars": (len(names), 1)}
    _require({k: B[k].shape for k in B} == want, f"shapes {({k: B[k].shape for k in B})}")
    T = np.asarray(p_dbg["blocks"]["x_sol"]["t"])
    _require(T[0] == t and np.array_equal(np.asarray(p_dbg["blocks"]["u_sol"]["t"]), T[:N])
             and p_dbg["blocks"]["scalars"]["t"][0] == T[0], "debug knot times")
    rec = SimpleNamespace(frame_at_step=lambda step: 0, meta={"body_labels": []},
                          targets=[TargetMsg(0, 0.0, FRANKA_CH, p_fr)])
    panel = LearnedMpcPanel()
    panel._index(rec, [TargetMsg(0, 0.0, DEBUG_CH, p_dbg)])
    solve = panel.solves[0]
    _require(solve.franka_knots is not None and solve.n == N, "panel did not pair the plan")
    U = B["u_sol"]
    u0 = executed_u0(franka_msg, ur_line, ee_u, X_t0_tr)
    pos, _, _ = parse_saved_traj_message(franka_msg)
    nz = n_x - 6
    return {"x0": float(max(np.abs(B["x_sol"][:nz, 0] - z).max(),
                            np.abs(B["x_sol"][nz:, 0] - np.r_[ee_f[:3], ee_u[:3]]).max())),
            "aug": float(np.abs(np.diff(B["x_sol"][nz:], axis=1) - U[:6]).max()),
            "knot": float(np.abs(u0[0:3] - U[0:3, 0]).max()),
            "knot_rot": float(np.abs(u0[6:9] - U[6:9, 0]).max()),
            "ur_end": float(max(np.abs(u0[3:6] - U[3:6, 0]).max(),
                                np.abs(u0[9:12] - U[9:12, 0]).max())),
            "panel_knots": float(np.abs(solve.franka_knots - pos).max())}


# --- --check ------------------------------------------------------------------------------------

def composed_angles_ref(U: np.ndarray, rot0: np.ndarray) -> np.ndarray:
    """scipy reference: per arm ``R_k = exp(w_k) R_{k-1}``, ``R_0 = exp(rot0)``, ``|log R_k|``."""
    out = np.zeros((U.shape[0], U.shape[1] + 1, 2))
    for i, plan in enumerate(U):
        for a, sl in enumerate((slice(6, 9), slice(9, 12))):
            Rm = Rotation.from_rotvec(rot0[a])
            out[i, 0, a] = Rm.magnitude()
            for k, u in enumerate(plan):
                Rm = Rotation.from_rotvec(u[sl]) * Rm
                out[i, k + 1, a] = Rm.magnitude()
    return out


class _FakeLc:
    def __init__(self) -> None:
        self.cb, self.debug = None, []

    def subscribe(self, _ch, cb) -> None:
        self.cb = cb

    def publish(self, ch, data) -> None:
        if ch == DEBUG_CH:
            self.debug.append(lcmt_timestamped_saved_traj.decode(data))


def _debug_stages(cli, stages: str, durations, n_latents: int) -> tuple[list[int], object]:
    """Node on a fake LCM fed ``n_latents`` latents 75 ms apart: debug ``stage`` rows."""
    import contextlib
    import io
    import tempfile

    from task_common.latent_encoder import latent_state_message

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "mppi.yaml"
        prm = dataclasses.replace(load_params(cli), stage_max_durations_s=durations)
        cfg.write_text(yaml.safe_dump(prm.to_dict()))
        args = node_mod.create_parser().parse_args(
            ["--lcm-url", "fake", "--deploy", str(cli.deploy), "--config", str(cfg), "--stages",
             stages, "--max-duration-s", "30", "--debug-channel", DEBUG_CH,
             "--device", cli.device])
        lc = _FakeLc()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            node = node_mod.MppiControllerNode(args, lc)
            node.start()
            ef, eu = np.r_[0.45, 0.05, 0.09, 1, 0, 0, 0], np.r_[0.55, -0.1, 0.1, 0, 1, 0, 0]
            for k in range(n_latents):
                utime = 10_000_000 + k * 75_000
                msg = latent_state_message(utime, utime * 1e-6, node.stages[-1].goal + 0.2, ef,
                                           eu, np.zeros(40))
                lc.cb(LATENT_STATE_CHANNEL, msg.encode())
                node._process(*node.pending)
                node.pending = None
    rows = []
    for m in lc.debug:
        b = {x.trajectory_name: x for x in m.saved_traj.trajectories}["scalars"]
        rows.append(int(dict(zip(b.datatypes, np.asarray(b.datapoints)[:, 0]))["stage"]))
    return rows, (node, out.getvalue())


def check_stage_rows(cli) -> tuple[bool, str]:
    n_goals = len(TorchLcs.from_deploy(cli.deploy, "cpu").stage_goals)
    last, _ = _debug_stages(cli, "last", None, 10)
    both, (node, text) = _debug_stages(cli, "all", [4.0, 30.0], 70)
    first_1 = both.index(1) if 1 in both else -1
    line = next((ln for ln in text.splitlines() if "[learned-mpc] stage 0 timeout" in ln), "")
    t_switch = float(line.split("t=")[1].split()[0]) - 10.0 if line else math.nan
    ok = (set(last) == {n_goals - 1} and n_goals == 2 and first_1 > 0
          and set(both[:first_1]) == {0} and set(both[first_1:]) == {1}
          and abs(t_switch - 4.0) < 0.075 + 1e-9 and node.stage == 1 and not node.done)
    return ok, (f"--stages last: {len(last)} debug msgs, stage row {sorted(set(last))} (= index "
                f"of the deploy's last of {n_goals} goals); --stages all, durations [4, 30]: "
                f"{first_1} msgs stage 0 then {len(both) - first_1} stage 1, stage 0 timeout at "
                f"{t_switch:.3f} s after the first latent")


class _BytesLc(_FakeLc):
    def __init__(self) -> None:
        super().__init__()
        self.raw: list[tuple[str, bytes]] = []

    def publish(self, ch, data) -> None:
        super().publish(ch, data)
        self.raw.append((ch, data))


def _masked(ch: str, data: bytes) -> bytes:
    """Published bytes with the debug message's run-varying ``solve_ms`` zeroed."""
    if ch != DEBUG_CH:
        return data
    m = lcmt_timestamped_saved_traj.decode(data)
    for b in m.saved_traj.trajectories:
        if b.trajectory_name == "scalars":
            b.datapoints[list(b.datatypes).index("solve_ms")] = [0.0]
    return m.encode()


def _stream(cli, overrides: dict, z_of, n_latents: int, node_module=None,
            node_flags: dict | None = None) -> dict:
    """Node (``node_module``, default this tree's) on a fake LCM; latent ``k`` = ``z_of(k,
    node)``, 75 ms apart from t = 10 s. Published bytes (debug ``solve_ms`` masked), stdout,
    log rows minus timing, the latent index of each stage marker; ``node_flags`` = extra node
    flags (``NODE_KEYS`` names)."""
    import contextlib
    import io
    import tempfile

    from task_common.latent_encoder import latent_state_message

    mod = node_module or node_mod
    with tempfile.TemporaryDirectory() as tmp:
        cfg, log = Path(tmp) / "mppi.yaml", Path(tmp) / "solves.jsonl"
        prm = load_params(cli).to_dict()
        prm.update(overrides)
        cfg.write_text(yaml.safe_dump(prm))
        flags = [x for k, v in (node_flags or {}).items()
                 for x in (f"--{k.replace('_', '-')}", str(v))]
        args = mod.create_parser().parse_args(
            ["--lcm-url", "fake", "--deploy", str(cli.deploy), "--config", str(cfg), "--stages",
             "all", "--max-duration-s", "30", "--debug-channel", DEBUG_CH, "--log", str(log),
             "--device", cli.device, *flags])
        lc = _BytesLc()
        out = io.StringIO()
        markers = []
        with contextlib.redirect_stdout(out):
            node = mod.MppiControllerNode(args, lc)
            node.start()
            ef, eu = np.r_[0.45, 0.05, 0.09, 1, 0, 0, 0], np.r_[0.55, -0.1, 0.1, 0, 1, 0, 0]
            for k in range(n_latents):
                if node.done:
                    break
                utime = 10_000_000 + k * 75_000
                n_lines = out.getvalue().count("[learned-mpc] stage")
                msg = latent_state_message(utime, utime * 1e-6, z_of(k, node), ef, eu,
                                           np.zeros(40))
                lc.cb(LATENT_STATE_CHANNEL, msg.encode())
                node._process(*node.pending)
                node.pending = None
                if out.getvalue().count("[learned-mpc] stage") > n_lines:
                    markers.append(k)
            node.log.close()
        rows = [json.loads(x) for x in log.read_text().splitlines()]
    for r in rows:
        r.pop("solve_ms", None)
    stage_rows = []
    for m in lc.debug:
        b = {x.trajectory_name: x for x in m.saved_traj.trajectories}["scalars"]
        stage_rows.append(int(dict(zip(b.datatypes, np.asarray(b.datapoints)[:, 0]))["stage"]))
    return {"raw": [(ch, _masked(ch, b)) for ch, b in lc.raw], "stdout": out.getvalue(),
            "rows": rows, "markers": markers, "node": node, "stage_rows": stage_rows}


def check_reach_streak(cli, T: TorchLcs, L) -> tuple[bool, str]:
    """C12: dist < tol from latent 20 -> stage-0 marker at 20 + streak - 1; a one-latent
    excursion resets the streak; the last stage keeps the single-latent rule; streak 1 = C9."""
    rng = np.random.default_rng(12)
    vecs = [L.z_goal + rng.normal(size=L.n_x) * L.z_std for _ in range(3)]
    goals = [T.stage_goals[0], T.stage_goals[-1], L.z_goal]
    wd = max(abs(T.whitened_dist(v, g) - L.whitened_dist(v, g)) for v, g in zip(vecs, goals))
    unit = rng.normal(size=L.n_x)
    unit /= np.linalg.norm(unit)
    tol, far, near = 1.0, 2.0, 0.5

    def at(goal_k: int, r: float) -> np.ndarray:
        return T.stage_goals[goal_k] + r * unit * T.z_std_np  # whitened distance exactly r

    def drop(excursion: int | None, last_at: int | None):
        def z_of(k, node):
            if node.stage == 1:
                return at(1, near if last_at is not None and k == last_at else far)
            return at(0, near if k >= 20 and k != excursion else far)
        return z_of

    over = {"stage_goal_tols": [tol, tol], "stage_max_durations_s": [10.0, 30.0]}
    s3 = _stream(cli, {**over, "stage_reach_streak": 3}, drop(None, 30), 40)
    s3x = _stream(cli, {**over, "stage_reach_streak": 3}, drop(21, None), 30)
    s1 = _stream(cli, {**over, "stage_reach_streak": 1}, drop(None, None), 24)
    c9 = {"stage_goal_tols": None, "stage_max_durations_s": [4.0, 30.0]}

    def far_z(k, node):  # C9's stream
        return T.stage_goals[-1] + 0.2

    c9_1 = _stream(cli, {**c9, "stage_reach_streak": 1}, far_z, 70)
    c9_cfg = _stream(cli, c9, far_z, 70)
    reached = [ln for ln in s3["stdout"].splitlines() if "[learned-mpc] stage" in ln]
    streak_rows = [r.get("reach_streak") for r in s3["rows"] if r["stage"] == 0][-3:]
    c9_same = (c9_1["raw"] == c9_cfg["raw"] and c9_1["stdout"] == c9_cfg["stdout"]
               and [{k: v for k, v in r.items() if k != "reach_streak"} for r in c9_cfg["rows"]]
               == c9_1["rows"])
    ok = (wd <= 1e-12 and s3["markers"] == [22, 30] and len(reached) == 2
          and "stage 0 reached" in reached[0] and "stage 1 reached" in reached[1]
          and s3["node"].done and streak_rows == [0, 1, 2] and s3x["markers"] == [24]
          and s1["markers"] == [20] and "stage 0 reached" in s1["stdout"] and c9_same
          and "reach_streak" not in c9_1["rows"][0] and len(c9_1["raw"]) > 0)
    return ok, (f"whitened_dist torch vs LearnedLcs on 3 vectors {wd:.1e}; dist < tol from "
                f"latent 20: streak 3 -> stage markers at latents {s3['markers']} (stage 1 = "
                f"last: one latent inside at 30 completes), log reach_streak before the switch "
                f"{streak_rows}; excursion at 21 -> {s3x['markers']}; streak 1 -> "
                f"{s1['markers']}; C9 stream streak 1 vs the config's "
                f"{load_params(cli).stage_reach_streak}: {len(c9_1['raw'])} published msgs, "
                f"stdout, log rows identical {c9_same}")


def run_check(cli) -> int:
    rng = np.random.default_rng(0)
    L = LearnedLcs.load(cli.deploy)
    T = TorchLcs.from_deploy(cli.deploy, cli.device)
    results = []

    def report(name: str, ok: bool, detail: str) -> None:
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    Z = L.z_goal + rng.normal(size=(N_CHECK, L.n_x)) * L.z_std * 2.0
    U = rng.uniform(L.u_lb, L.u_ub, size=(N_CHECK, L.n_u))
    ref = [L.step(z, u) for z, u in zip(Z, U, strict=True)]
    zn, lam = T.step(T.tensor(Z), T.tensor(U))
    zn, lam = zn.double().cpu().numpy(), lam.double().cpu().numpy()
    err = np.abs(zn - np.stack([r[0] for r in ref])).max(1) / np.maximum(1.0, np.abs(Z).max(1))
    lam_err = np.abs(lam - np.stack([r[1] for r in ref])).max()
    report("C1 step", bool(err.max() <= 1e-5),
           f"{N_CHECK} random (z, u): max |dz| / max(1, |z|) {err.max():.2e} (tol 1e-5), "
           f"lambda {lam_err:.2e}")

    H = 16
    Us = rng.uniform(L.u_lb, L.u_ub, size=(N_PLANS, H, L.n_u))
    z0 = Z[0]
    p0 = np.r_[0.45, 0.05, 0.08, 0.55, -0.1, 0.06]
    rot0 = rng.normal(size=(2, 3)) * 0.1
    Zr, Pr, Rr = (x.double().cpu().numpy() for x in T.rollout(z0, p0, T.tensor(Us), rot0))
    refs = np.stack([L.rollout(z0, u)[0] for u in Us])
    r_err = (np.abs(Zr - refs).max((1, 2)) / np.maximum(1.0, np.abs(refs).max((1, 2)))).max()
    p_ref = p0 + np.concatenate([np.zeros((N_PLANS, 1, 6)), np.cumsum(Us[..., :6], 1)], 1)
    R_ref = composed_angles_ref(Us, rot0)
    p_err, R_err = np.abs(Pr - p_ref).max(), np.abs(Rr - R_ref).max()
    report("C2 rollout", bool(r_err <= 1e-5 and p_err <= 1e-6 and R_err <= 1e-6),
           f"{N_PLANS} plans x H {H}: Z {r_err:.2e} (scaled, tol 1e-5), P {p_err:.1e} m, "
           f"R {R_err:.1e} rad")

    with np.load(cli.deploy) as f:
        d = {k: f[k] for k in f.files if not k.startswith("encoder")}
    d["A_p"] = rng.normal(size=(L.n_x, 6)) * 0.5
    d["E_p"] = rng.normal(size=(L.n_lam, 6)) * 0.5
    Tp = TorchLcs(d, cli.device)
    Pp = rng.normal(size=(N_CHECK, 6)) * 0.3
    zp, _ = Tp.step(Tp.tensor(Z), Tp.tensor(U), Tp.tensor(Pp))
    ref_p = []
    for z, u, p in zip(Z, U, Pp, strict=True):
        Lp = LearnedLcs.load(cli.deploy)
        Lp.c, Lp.d = L.c + d["E_p"] @ p, L.d + d["A_p"] @ p
        ref_p.append(Lp.step(z, u)[0])
    ap_err = (np.abs(zp.double().cpu().numpy() - np.stack(ref_p)).max(1)
              / np.maximum(1.0, np.abs(Z).max(1))).max()
    report("C3 A_p/E_p", bool(ap_err <= 1e-5),
           f"synthetic couplings vs LearnedLcs with c + E_p p, d + A_p p: {ap_err:.2e}")

    k_err = r_err2 = 0.0
    t_ok = name_ok = True
    for Up in Us:
        Up = Up.copy()
        Up[:, 6:12] = rng.uniform(-0.05, 0.05, size=(H, 6))
        q = Rotation.random(random_state=int(rng.integers(1 << 30))).as_quat(scalar_first=True)
        pose7 = np.r_[rng.uniform(-0.5, 0.5, 3), q]
        utime = int(rng.integers(1, 10**9))
        msg = node_mod.franka_message(utime, utime * 1e-6, T.dt, pose7, Up)
        dec = lcmt_timestamped_saved_traj.decode(msg.encode())
        pos, quat, times = parse_saved_traj_message(dec)
        Rm, p = Rotation.from_quat(q, scalar_first=True), pose7[:3].copy()
        for i in range(node_mod.N_KNOTS):
            if i:
                p = p + Up[i - 1, :3]
                Rm = Rotation.from_rotvec(Up[i - 1, 6:9]) * Rm
            qi = Rm.as_quat(scalar_first=True)
            k_err = max(k_err, float(np.abs(pos[i] - p).max()))
            r_err2 = max(r_err2, float(min(np.abs(quat[i] - qi).max(), np.abs(quat[i] + qi).max())))
        t_ok &= bool(len(times) == 7 and times[0] == utime * 1e-6
                     and np.allclose(np.diff(times), T.dt, atol=1e-12))
        name_ok &= dec.saved_traj.metadata.name == node_mod.METADATA_NAME
    report("C4 Franka knot chain", k_err <= KNOT_TOL and r_err2 <= ROT_TOL and t_ok and name_ok,
           f"{N_PLANS} plans: pos {k_err:.1e} m, quat {r_err2:.1e} vs scipy exp(u) q_(i-1); "
           f"7 knots at t + i dt; metadata {node_mod.METADATA_NAME!r}")

    ur = UrDecoder()
    X_t0_tr = x_tool0_tracking()
    X_base_W = node_mod._rigid_inv(ur_base_world())
    X_tr_t0 = node_mod._rigid_inv(X_t0_tr)
    u_err = 0.0
    span_ok = True
    for _ in range(N_PLANS):
        q = Rotation.random(random_state=int(rng.integers(1 << 30))).as_quat(scalar_first=True)
        pose7 = np.r_[rng.uniform([0.3, -0.4, 0.0], [0.9, 0.4, 0.3]), q]
        u0 = np.r_[rng.uniform(-0.006, 0.006, 6), rng.uniform(-0.035, 0.035, 6)]
        utime = int(rng.integers(1, 10**9))
        msg = node_mod.ur_message(utime, utime * 1e-6, T.dt, pose7, u0, X_base_W, X_tr_t0)
        line = ur.decode(msg.encode())
        _require(line is not None, "bridge dropped the UR line")
        W0, W1 = node_mod.ur_line_world(pose7, u0)
        X0 = pose_mat(line.p0, line.q0) @ X_t0_tr
        X1 = pose_mat(line.p1, line.q1) @ X_t0_tr
        u_err = max(u_err, float(np.abs(X0 - W0).max()), float(np.abs(X1 - W1).max()))
        span_ok &= abs(line.t1 - line.t0 - T.dt) <= SPAN_TOL and line.t0 == utime * 1e-6
    msg.saved_traj.metadata.name = METADATA_NAME
    dropped = ur.decode(msg.encode()) is None
    report("C5 UR line round trip", u_err <= UR_TOL and span_ok and dropped,
           f"{N_PLANS} lines world -> base -> _on_ur_traj -> world: {u_err:.1e} (4x4 entries); "
           f"span dt; a {METADATA_NAME!r}-named copy is dropped: {dropped}")

    params = load_params(cli)
    gen = torch.Generator(device=T.device).manual_seed(0)
    pose_f = np.r_[0.45, 0.05, 0.08, 1.0, 0.0, 0.0, 0.0]
    pose_u = np.r_[0.55, -0.1, 0.06, 0.0, 1.0, 0.0, 0.0]
    p0 = np.r_[pose_f[:3], pose_u[:3]]
    q_kw = {"q_now": np.stack([pose_f[3:7], pose_u[3:7]])} if params.clearance else {}
    Up, info = mppi_plan(T, params, z0, p0, None, L.z_goal, None, gen, **q_kw)
    lb, ub = params.bounds(T)
    in_box = bool(np.all(Up >= lb - BOUND_TOL) and np.all(Up <= ub + BOUND_TOL))
    utime = 123_450_000
    t = utime * 1e-6
    fr = node_mod.franka_message(utime, t, T.dt, pose_f, Up)
    urm = node_mod.ur_message(utime, t, T.dt, pose_u, Up[0], X_base_W, X_tr_t0)
    Zd, Pd, _ = T.rollout(z0, p0, T.tensor(Up[:7])[None])
    scal = {"k0": -1.0, "stage": 0.0, "dist_ref": 1.0, "dist_final": 1.0,
            "solve_ms": info["solve_ms"], "t_mpc_start": t, "ref_mode": -1.0, "n_x": T.n_x + 6.0,
            "N": 7.0, "solve_ok": 1.0}
    dbg = node_mod.debug_message(utime, t, T.dt, Zd[0].double().cpu().numpy(),
                                 Pd[0].double().cpu().numpy(), Up, L.z_goal, p0, scal)
    errs = verify_debug(lcmt_timestamped_saved_traj.decode(dbg.encode()), fr,
                        ur.decode(urm.encode()), (utime, t, z0, pose_f, pose_u), X_t0_tr)
    ok = (in_box and errs["x0"] <= 1e-6 and errs["aug"] <= 1e-6 and errs["knot"] <= KNOT_TOL
          and errs["knot_rot"] <= ROT_TOL and errs["ur_end"] <= UR_TOL
          and errs["panel_knots"] == 0.0)
    report("C6 MPPI plan + debug message", ok,
           f"plan in the box {in_box} (solve {info['solve_ms']:.1f} ms, K {params.K} H "
           f"{params.H}); recorder payload -> LearnedMpcPanel._index ok; "
           + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    if T.device.type == "cuda":
        Zf, Pf, Rf = (x.double().cpu().numpy() for x in T.rollout_fused(z0, p0, T.tensor(Us), rot0))
        f_err = (np.abs(Zf - refs).max((1, 2)) / np.maximum(1.0, np.abs(refs).max((1, 2)))).max()
        fp_err, fR_err = np.abs(Pf - p_ref).max(), np.abs(Rf - R_ref).max()
        zf_p = Tp.rollout_fused(Z[0], Pp[0], Tp.tensor(Us[:, :1]))[0][:, 1].double().cpu().numpy()
        zt_p = Tp.rollout(Z[0], Pp[0], Tp.tensor(Us[:, :1]))[0][:, 1].double().cpu().numpy()
        fap_err = np.abs(zf_p - zt_p).max() / max(1.0, np.abs(zt_p).max())
        plans = []
        for graph in (False, True):
            prm = dataclasses.replace(params, cuda_graph=graph)
            g = torch.Generator(device=T.device).manual_seed(7)
            plans.append(MppiSolver(T, prm).solve(z0, p0, None, L.z_goal, None, g,
                                                  **q_kw)[0])
        g_err = np.abs(plans[0] - plans[1]).max()
        report("C7 fused rollout + CUDA graph",
               bool(f_err <= 1e-5 and fp_err <= 1e-6 and fR_err <= 1e-6 and fap_err <= 1e-5
                    and g_err == 0.0),
               f"Warp kernel vs LearnedLcs.rollout {f_err:.2e} (scaled, tol 1e-5), P "
               f"{fp_err:.1e} m, R {fR_err:.1e}; A_p/E_p path vs torch {fap_err:.1e}; graph "
               f"replay vs eager plan (same seed) {g_err:.1e}")
    Ur = rng.uniform(-0.05, 0.05, size=(N_CHECK, H, L.n_u))
    r0 = rng.normal(size=(2, 3)) * 0.4
    ref = composed_angles_ref(Ur, r0)
    errs = {"torch": np.abs(T.rotation_angles(T.tensor(Ur), r0).double().cpu().numpy()
                            - ref).max()}
    if T.device.type == "cuda":
        errs["fused"] = np.abs(T.rollout_fused(z0, p0, T.tensor(Ur), r0)[2].double().cpu().numpy()
                               - ref).max()
    report("C8 composed rotation angle", all(v <= 1e-6 for v in errs.values()),
           f"{N_CHECK} random sequences x H {H} (|w| <= 0.05 rad/axis, start offset ~0.7 rad) "
           f"vs scipy: " + ", ".join(f"{k} {v:.1e} rad" for k, v in errs.items())
           + f" (tol 1e-6); max angle {ref.max():.2f} rad")
    report("C9 debug stage = goal index", *check_stage_rows(cli))
    report("C12 stage reach streak", *check_reach_streak(cli, T, L))
    report("C13 transport identity", *check_transport_identity(cli, T, L))
    report("C14 demo prior + waypoints + track", *check_transport(cli, T, L))
    if cli.clearance_model is None:
        print("[SKIP] C0/C10/C11 clearance: no --clearance-model", flush=True)
    else:
        lines = []
        report("C0 clearance inert", *check_clearance_inert(cli, T, L, z0, p0, rot0))
        report("C10 clearance kernel", *check_clearance_kernel(cli, T, rng, lines))
        report("C11 e2 board aborts", *check_clearance_aborts(cli, T, lines))
        if cli.validation_md is not None:
            cli.validation_md.parent.mkdir(parents=True, exist_ok=True)
            cli.validation_md.write_text("\n".join(lines) + "\n")
            print(f"[clearance] wrote {cli.validation_md}", flush=True)
    n_pass = sum(results)
    print(f"{'ALL MPPI CHECKS PASSED' if all(results) else 'MPPI CHECKS FAILED'} "
          f"({n_pass}/{len(results)})", flush=True)
    return 0 if all(results) else 1


# --- transport (C13, C14) ----------------------------------------------------------------------

def _load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _yaml_params(path: Path, mod, cli) -> object:
    """``mod.MppiParams`` of a yaml; a missing clearance npz -> ``--clearance-model`` / off."""
    raw = yaml.safe_load(path.read_text()) or {}
    clr = raw.get("clearance")
    if clr is not None and not (REPO_ROOT / clr["model"]).is_file():
        raw["clearance"] = ({**clr, "model": str(Path(cli.clearance_model).resolve())}
                            if cli.clearance_model is not None else None)
    return mod.MppiParams(**raw)


def _nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _same_info(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k, va in a.items():
        vb = b[k]
        if k == "solve_ms":
            continue
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not np.array_equal(np.asarray(va), np.asarray(vb)):
                return False
        elif not (va == vb or (_nan(va) and _nan(vb))):
            return False
    return True


def check_transport_identity(cli, T: TorchLcs, L) -> tuple[bool, str]:
    """C13: every ``mppi*.yaml`` under realfirst/ loads with the prior off; the pre-package
    solver vs this one on each distinct config (3 seeds, cold + warm-started solve): plans and
    stats identical; ``(H, n_x)`` goal of H identical rows == the single goal (0.0); the
    pre-package node vs this node at default flags (C9 stream): published bytes, stdout, log
    rows identical."""
    yamls = sorted((REPO_ROOT / "data" / "lcs" / "realfirst").rglob("mppi*.yaml"))
    loaded = [MppiParams.from_yaml(y) for y in yamls]
    off = all(p.prior_alpha == 0.0 and p.prior_weight == 0.0 for p in loaded)
    ref = IDENTITY_DIR / "mppi_lcs_pre.py"
    ref_node = IDENTITY_DIR / "mppi_controller_node_pre.py"
    rng = np.random.default_rng(13)
    z0 = L.z_goal + rng.normal(size=L.n_x) * L.z_std
    p0 = np.r_[0.45, 0.05, 0.09, 0.55, -0.1, 0.1]
    rot0 = rng.normal(size=(2, 3)) * 0.05
    q = (_quat_now(cli) if cli.clearance_model is not None
         else np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
    n_cfg = n_solves = 0
    plan_diff, bad = 0.0, []
    if ref.is_file():
        old = _load_module(ref, "mppi_lcs_pre")
        T_old = old.TorchLcs.from_deploy(cli.deploy, cli.device)
        seen = set()
        for y in yamls:
            p_new, p_old = _yaml_params(y, sys.modules["task_common.mppi_lcs"], cli), None
            key = repr(p_new.to_dict())
            if key in seen:
                continue
            seen.add(key)
            p_old = _yaml_params(y, old, cli)
            n_cfg += 1
            s_new, s_old = MppiSolver(T, p_new), old.MppiSolver(T_old, p_old)
            for seed in IDENTITY_SEEDS:
                prev = []
                for mod_solver in (s_new, s_old):
                    g = torch.Generator(device=T.device).manual_seed(seed)
                    U1, i1 = mod_solver.solve(z0, p0, rot0, T.np["z_goal"], None, g, None, q)
                    U2, i2 = mod_solver.solve(z0, p0, rot0, T.np["z_goal"], U1, g, None, q)
                    prev.append((U1, i1, U2, i2))
                (a1, ai1, a2, ai2), (b1, bi1, b2, bi2) = prev
                n_solves += 2
                plan_diff = max(plan_diff, float(np.abs(a1 - b1).max()),
                                float(np.abs(a2 - b2).max()))
                if not (np.array_equal(a1, b1) and np.array_equal(a2, b2)
                        and _same_info(ai1, bi1) and _same_info(ai2, bi2)):
                    bad.append(f"{y.relative_to(REPO_ROOT)} seed {seed}")
    params = load_params(cli)
    q_kw = {"q_now": q} if params.clearance else {}
    rows = []
    for goal in (T.np["z_goal"], np.tile(T.np["z_goal"], (params.H, 1))):
        g = torch.Generator(device=T.device).manual_seed(3)
        rows.append(MppiSolver(T, params).solve(z0, p0, rot0, goal, None, g, **q_kw))
    track_diff = float(np.abs(rows[0][0] - rows[1][0]).max())
    track_same = track_diff == 0.0 and _same_info(rows[0][1], rows[1][1])
    node_same = None
    if ref_node.is_file():
        old_node = _load_module(ref_node, "mppi_controller_node_pre")
        c9 = {"stage_goal_tols": None, "stage_max_durations_s": [4.0, 30.0]}

        def far_z(k, node):
            return T.stage_goals[-1] + 0.2

        a = _stream(cli, c9, far_z, 70)
        b = _stream(cli, c9, far_z, 70, node_module=old_node)
        node_same = a["raw"] == b["raw"] and a["stdout"] == b["stdout"] and a["rows"] == b["rows"]
    ok = (off and ref.is_file() and n_cfg > 0 and not bad and track_same
          and node_same is True)
    return ok, (f"{len(yamls)} realfirst mppi*.yaml load, prior off {off}; pre-package solver "
                f"({ref.name}) vs tree on {n_cfg} distinct configs x {len(IDENTITY_SEEDS)} "
                f"seeds ({n_solves} solves, cold + warm): max plan diff {plan_diff:.1e}, stats "
                f"identical {not bad}{' ' + str(bad[:3]) if bad else ''}; (H, n_x) goal of H "
                f"identical rows vs the single goal: plan diff {track_diff:.1e}, info identical "
                f"{track_same}; pre-package node vs tree (C9 stream, default flags): bytes, "
                f"stdout, log rows identical {node_same}")


def _is_belt_cloud(deploy: Path) -> bool:
    with np.load(deploy, allow_pickle=False) as f:
        return "encoder_type" in f.files and str(f["encoder_type"]) == "belt_cloud_pointnet"


def check_transport(cli, T: TorchLcs, L) -> tuple[bool, str]:
    """C14: (a) the demo prior (alpha 1, weight 1e6) returns the demo sequence (within
    ``PRIOR_TOL_HALF`` x half at sigma_frac 1e-4; the default-sigma deviation is reported:
    MPPI weights are invariant to the cost scale), the prior term vs numpy, graph == eager with
    the prior on, both 0 == no u_demo; on belt-cloud deploys also (b) the waypoints artifact
    re-encodes, (c) ``--goal waypoints: --stages all`` stage tols == the artifact's and the
    latch waypoint's tol == ``latent_stage_tols.py`` tol_1 for the same goal vector (window 6,
    stride on the latch), (d) ``track`` on a fake LCM: idx +1 per latent, the hold capped at
    ``--track-max-hold-s``, debug stage row 0."""
    import tempfile

    params = load_params(cli)
    lb, ub = params.bounds(T)
    half = np.where((ub - lb) / 2.0 > 0.0, (ub - lb) / 2.0, 1.0)
    rng = np.random.default_rng(14)
    H = params.H
    z0 = L.z_goal + rng.normal(size=L.n_x) * L.z_std * 0.5
    p0 = np.r_[0.45, 0.05, 0.09, 0.55, -0.1, 0.1]
    q = (_quat_now(cli) if cli.clearance_model is not None
         else np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]))
    q_kw = {"q_now": q} if params.clearance else {}
    u_demo = np.clip(rng.normal(size=(H, L.n_u)) * half * 0.3, lb, ub)
    goal = T.np["z_goal"]

    def plan(prm, ud, seed=4, graph=None):
        prm = prm if graph is None else dataclasses.replace(prm, cuda_graph=graph)
        g = torch.Generator(device=T.device).manual_seed(seed)
        return MppiSolver(T, prm).solve(z0, p0, None, goal, None, g, None, **q_kw, u_demo=ud)

    strong = dataclasses.replace(params, prior_alpha=1.0, prior_weight=1e6, sigma_frac=1e-4)
    e_strong = float((np.abs(plan(strong, u_demo)[0] - u_demo) / half).max())
    lit = dataclasses.replace(params, prior_alpha=1.0, prior_weight=1e6)
    e_lit = float((np.abs(plan(lit, u_demo)[0] - u_demo) / half).max())
    e_none = float((np.abs(plan(params, None)[0] - u_demo) / half).max())
    zero_same = np.array_equal(plan(params, u_demo)[0], plan(params, None)[0])
    on = dataclasses.replace(params, prior_alpha=0.5, prior_weight=0.1)
    g_err = (float(np.abs(plan(on, u_demo, graph=False)[0] - plan(on, u_demo, graph=True)[0])
                   .max()) if T.device.type == "cuda" else 0.0)
    U = np.clip(rng.normal(size=(H, L.n_u)) * half * 0.5, lb, ub)
    c_on = rollout_cost(T, on, z0, p0, None, goal, U, **q_kw, u_demo=u_demo)[0]
    c_off = rollout_cost(T, on, z0, p0, None, goal, U, **q_kw)[0]
    term = 0.1 * float((((U - u_demo) / half) ** 2).sum())
    t_err = abs((c_on - c_off) - term) / max(1.0, abs(c_on))  # float32 cost
    ok = e_strong <= PRIOR_TOL_HALF and zero_same and g_err == 0.0 and t_err <= 1e-5
    detail = (f"(a) alpha 1, weight 1e6, sigma_frac 1e-4: max |U - U_demo| / half {e_strong:.1e} "
              f"(tol {PRIOR_TOL_HALF:g}); default sigma {e_lit:.2f} (no prior {e_none:.2f}); "
              f"prior term vs numpy {t_err:.1e}; graph == eager with the prior on {g_err:.1e}; "
              f"u_demo given with alpha 0 / weight 0 == no u_demo {zero_same}")
    if not _is_belt_cloud(cli.deploy):
        return ok, detail + "; (b-d) SKIP: not a belt-cloud deploy (no waypoints artifact)"
    import latent_stage_tols as lst
    import latent_waypoints as lw

    from task_common.latent_encoder import LatentEncoder

    with tempfile.TemporaryDirectory() as tmp:
        arts = {}
        for tag, stride, window in (("default", 10, 3), ("latch", 11, 6)):
            doc = lw.derive(Path(cli.deploy), list(CLEAN_DEMOS), GOAL_DEMO, stride, window,
                            90.0, 0.5)
            arts[tag] = (doc, Path(tmp) / f"{tag}.npz")
            lw.save_npz(arts[tag][1], doc)
        doc, path = arts["default"]
        enc = LatentEncoder.load(cli.deploy)
        ep = lst.load_episode(GOAL_DEMO)
        z_re = np.stack([enc.encode(np.asarray(ep["seg"][t], np.float32))
                         for t in range(len(ep["seg"]))])
        enc_err = float(np.abs(z_re - doc["z_traj"]).max() / np.abs(z_re).max())
        wp_err = float(np.abs(doc["z_wp"] - z_re[doc["frames"]]).max())
        args = node_mod.create_parser().parse_args(
            ["--lcm-url", "fake", "--deploy", str(cli.deploy), "--goal", f"waypoints:{path}",
             "--stages", "all", "--device", cli.device])
        st = node_mod.build_stages(args, params, T, node_mod.load_waypoints(args))
        st_ok = ([s.tol for s in st[:-1]] == doc["tol"][:-1].tolist()
                 and [s.index for s in st] == list(range(len(st))) and st[-1].tol == -1.0)
        ldoc = arts["latch"][0]
        k = ldoc["frames"].tolist().index(int(ldoc["latch"]))
        g_lcs, g1, g2 = lst.stage_goals(Path(cli.deploy))
        base = lst.derive(Path(cli.deploy), list(CLEAN_DEMOS), 6, 0.5, 90.0)
        orig = lst.stage_goals
        lst.stage_goals = lambda dep: (g_lcs, ldoc["z_wp"][k], g2)
        try:
            same = lst.derive(Path(cli.deploy), list(CLEAN_DEMOS), 6, 0.5, 90.0)
        finally:
            lst.stage_goals = orig
        tol_err = abs(float(ldoc["tol"][k]) - same["tol"][0])
        tol_vs_deploy = abs(float(ldoc["tol"][k]) - base["tol"][0])
        tr = check_track_stream(cli, T, path)
    ok = (ok and len(doc["frames"]) >= 6 and enc_err <= 1e-6 and wp_err == 0.0 and st_ok
          and tol_err <= 1e-9 and tr[0])
    return ok, (detail + f"; (b) {len(doc['frames'])} waypoints at {doc['frames'].tolist()}, "
                f"re-encode {enc_err:.1e} x max|z|, z_wp == z_traj[frames] {wp_err:.1e}; (c) "
                f"stage tols == artifact {st_ok}; latch tol (stride 11, window 6) "
                f"{float(ldoc['tol'][k]):.6f} vs latent_stage_tols tol_1 with the same goal "
                f"{same['tol'][0]:.6f} (diff {tol_err:.1e}), vs the deploy's z_goal_stage1 "
                f"{base['tol'][0]:.6f} (diff {tol_vs_deploy:.1e}, goal vectors "
                f"{T.whitened_dist(ldoc['z_wp'][k], g1):.1e} apart); (d) {tr[1]}")


def check_track_stream(cli, T: TorchLcs, wp_path: Path) -> tuple[bool, str]:
    """Track mode on a fake LCM: on the demo latents idx advances 1 per 75 ms latent; far off,
    the hold caps at ``track_max_hold_s`` and idx then advances again; debug stage row 0."""
    wp = node_mod.Waypoints.load(wp_path)
    over = {"goal": f"waypoints:{wp_path}", "goal_mode": "track", "demo_actions": "auto"}
    prior = {"prior_alpha": 0.5, "prior_weight": 0.1}
    on = _stream(cli, prior, lambda k, node: wp.z_traj[min(k, wp.T - 1)], 30, node_flags=over)
    far = _stream(cli, prior, lambda k, node: wp.z_traj[0] + 50.0 * wp.tol.max() / np.sqrt(
        T.n_x) * T.z_std_np, 80, node_flags={**over, "track_max_hold_s": 2.0})
    idx_on = [r["idx"] for r in on["rows"]]
    idx_far = [r["idx"] for r in far["rows"]]
    held = [r["held_s"] for r in far["rows"]]
    stages = set(on["stage_rows"]) | set(far["stage_rows"])
    n_flat = sum(1 for a, b in itertools.pairwise(idx_far) if a == b)
    ok = (idx_on == list(range(len(idx_on))) and max(held) <= 2.0 + 1e-9
          and abs(max(held) - 2.0) < 0.08 and idx_far[-1] > idx_far[0]
          and all(b >= a for a, b in itertools.pairwise(idx_far)) and stages == {0}
          and all({"idx", "held_s", "d_idx"} <= set(r) for r in on["rows"]))
    return ok, (f"track on demo latents: idx {idx_on[:4]}..{idx_on[-1]} over {len(idx_on)} "
                f"latents; far off: held_s max {max(held):.3f} (cap 2.0), idx flat on {n_flat} "
                f"latents then {idx_far[-1]}; debug stage rows {sorted(stages)}")


# --- clearance (C0, C10, C11) ------------------------------------------------------------------

def _clear_cfg(cli, **kw) -> dict:
    return {"model": str(Path(cli.clearance_model).resolve()), **kw}


def _clear_cfg_config(cli) -> dict:
    """``--clearance-model`` with the config's thresholds when it has a clearance block (as the
    node merges them), else the npz's."""
    cfg = load_params(cli).clearance or {}
    return _clear_cfg(cli, **{k: cfg[k] for k in ("ur_min_m", "franka_min_m") if k in cfg})


def _quat_now(cli) -> np.ndarray:
    m = cm.ClearanceModel.load(cli.clearance_model)
    return np.stack([Rotation.from_matrix(np.asarray(m.source[k])[:3, :3]).as_quat(
        scalar_first=True) for k in ("X_W_tip", "X_W_track")])


def check_clearance_inert(cli, T: TorchLcs, L, z0, p0, rot0) -> tuple[bool, str]:
    """No ``clearance`` key -> off; on with -inf thresholds and w_soft 0 -> the same plan."""
    params = load_params(cli)
    raw = (yaml.safe_load(Path(cli.config).read_text()) if cli.config else {}) or {}
    raw.pop("clearance", None)  # a config with the block: inert-default checked without it
    off_default = MppiParams(**raw).clearance is None
    q = _quat_now(cli)
    plans = []
    for clear in (None, _clear_cfg(cli, ur_min_m=-math.inf, franka_min_m=-math.inf)):
        prm = dataclasses.replace(params, clearance=clear)
        g = torch.Generator(device=T.device).manual_seed(11)
        plans.append(MppiSolver(T, prm).solve(z0, p0, rot0, L.z_goal, None, g, None, q)[0])
    diff = float(np.abs(plans[0] - plans[1]).max())
    return off_default and diff == 0.0, (
        f"{Path(cli.config).name} without a clearance key -> MppiParams.clearance None "
        f"({off_default}); clearance on with -inf thresholds, w_soft 0: plan diff {diff:.1e} "
        f"vs off (same seed)")


def _seed_frames() -> tuple[np.ndarray, np.ndarray, list[Path]]:
    files = sorted(f for d in SEED_EPISODES for f in Path(d).glob("episode_*.npz"))
    ef, eu = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            ef.append(d["sim_ee_franka"].astype(np.float64))
            eu.append(d["sim_ee_ur"].astype(np.float64))
    return np.concatenate(ef), np.concatenate(eu), files


def _frame_clearance(clr: Clearance, ee_f, ee_u) -> np.ndarray:
    """``(B, 2)`` [UR, Franka] kernel distances at pose7 (xyz + wxyz) pairs."""
    t = clr.lcs.tensor
    B = len(ee_f)
    P = t(np.repeat(np.c_[ee_f[:, :3], ee_u[:, :3]][:, None], 2, axis=1))
    q = t(np.stack([ee_f[:, 3:7], ee_u[:, 3:7]], axis=1))
    C = clr(P, t(np.zeros((B, 1, 12))), q, True)
    return C[:, 0].double().cpu().numpy()


def check_clearance_kernel(cli, T: TorchLcs, rng, lines: list[str]) -> tuple[bool, str]:
    model = cm.ClearanceModel.load(cli.clearance_model)
    board = model.board()
    ef, eu, files = _seed_frames()
    idx = rng.choice(len(ef), N_CLEAR_POSES)
    poses = []
    for ee in (ef[idx], eu[idx]):
        pos = ee[:, :3] + rng.uniform(-CLEAR_POS_MM, CLEAR_POS_MM, (N_CLEAR_POSES, 3)) * 1e-3
        axis = rng.normal(size=(N_CLEAR_POSES, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        ang = np.radians(rng.uniform(0.0, CLEAR_ROT_DEG, N_CLEAR_POSES))
        rot = Rotation.from_rotvec(axis * ang[:, None]) * Rotation.from_quat(ee[:, 3:7],
                                                                            scalar_first=True)
        poses.append(np.c_[pos, rot.as_quat(scalar_first=True)])
    pf, pu = poses
    grip = cm.clearance.Gripper(bodies=(), shape_points=(), shape_labels=(), local=model.ur_local)
    ref = np.array([[grip.predict(board, cm.pose_mat(u)),
                     cm.set_distance(board, model.franka_local, cm.pose_mat(f))]
                    for f, u in zip(pf, pu, strict=True)])
    full = Clearance(_clear_cfg(cli), T)  # deployed: every point
    pruned = Clearance(_clear_cfg(cli), T, max_points=cm.PRUNE_POINTS)
    e_full = np.abs(_frame_clearance(full, pf, pu) - ref).max(0)
    e_pr = np.abs(_frame_clearance(pruned, pf, pu) - ref).max(0)
    # Rollouts: torch path vs the Warp kernel on the same P / U / start orientations.
    params = load_params(cli)
    lb, ub = params.bounds(T)
    n_roll, H = 64, params.H
    U = rng.uniform(lb, ub, size=(n_roll, H, 12))
    start = rng.integers(len(ef), size=n_roll)
    p0 = np.c_[ef[start, :3], eu[start, :3]]
    P = np.concatenate([p0[:, None], p0[:, None] + np.cumsum(U[..., :6], 1)], 1)
    q0 = np.stack([ef[start, 3:7], eu[start, 3:7]], 1)
    Pt, Ut, qt = T.tensor(P), T.tensor(U), T.tensor(q0)
    e_torch = float((full.torch_path(Pt, Ut, qt) - full(Pt, Ut, qt, True)).abs().max())
    # Graph replay vs eager with clearance on.
    prm = dataclasses.replace(params, clearance=_clear_cfg(cli))
    q_now = _quat_now(cli)
    m_src = model.source
    pn = np.r_[np.asarray(m_src["X_W_tip"])[:3, 3], np.asarray(m_src["X_W_track"])[:3, 3]]
    out = []
    for graph in (False, True):
        g = torch.Generator(device=T.device).manual_seed(5)
        out.append(MppiSolver(T, dataclasses.replace(prm, cuda_graph=graph)).solve(
            T.np["z_goal"] + 0.5, pn, None, T.np["z_goal"], None, g, None, q_now))
    g_err = float(np.abs(out[0][0] - out[1][0]).max())
    c_err = float(np.abs(out[0][1]["C"] - out[1][1]["C"]).max())
    ok = (bool(np.all(e_full <= CLEAR_FULL_TOL_M)) and e_torch <= CLEAR_TORCH_TOL_M
          and g_err == 0.0 and c_err == 0.0)
    n_ur, n_f = (len(p) for p in full.points)
    pr_ur, pr_f = (len(p) for p in pruned.points)
    lines += [
        "# Clearance model validation (mppi_offline.py --check C10/C11)", "",
        (f"- model `{cli.clearance_model}` ({model.source.get('start_state')}): UR "
         f"{len(model.ur_local)} points (kernel {n_ur}), Franka {len(model.franka_local)} "
         f"({model.source.get('franka')}, kernel {n_f}); thresholds UR "
         f"{model.ur_min_m * 1e3:g} / Franka {model.franka_min_m * 1e3:g} mm"), "",
        "## C10 kernel vs reference", "",
        (f"{N_CLEAR_POSES} poses = recorded frames of {len(files)} seed episodes + U(+-"
         f"{CLEAR_POS_MM:g}) mm, x random rotation <= {CLEAR_ROT_DEG:g} deg; reference UR = "
         "`Gripper.predict` (full set), Franka = `Board.min_distance` of the full hand set."), "",
        "| | UR max abs diff mm | Franka max abs diff mm | tol mm |", "|---|---|---|---|",
        (f"| kernel, full set (deployed) | {e_full[0] * 1e3:.4f} | {e_full[1] * 1e3:.4f} | "
         f"{CLEAR_FULL_TOL_M * 1e3:g} |"),
        (f"| kernel, {cm.PRUNE_POINTS} lowest per set (not deployed) | {e_pr[0] * 1e3:.4f} | "
         f"{e_pr[1] * 1e3:.4f} | {CLEAR_PRUNED_TOL_M * 1e3:g} (report) |"), "",
        (f"- reference range UR {ref[:, 0].min() * 1e3:.1f}..{ref[:, 0].max() * 1e3:.1f} mm, "
         f"Franka {ref[:, 1].min() * 1e3:.1f}..{ref[:, 1].max() * 1e3:.1f} mm"),
        (f"- torch path vs Warp kernel ({n_roll} plans x H {H} from seed frames, inputs in the "
         f"box): {e_torch:.1e} m (tol {CLEAR_TORCH_TOL_M:g}); graph replay vs eager plan "
         f"{g_err:.1e}, C {c_err:.1e}"), ""]
    return ok, (f"{N_CLEAR_POSES} poses: deployed full set (UR {n_ur}, Franka {n_f}) vs "
                f"reference UR {e_full[0] * 1e3:.4f} / Franka {e_full[1] * 1e3:.4f} mm (tol "
                f"{CLEAR_FULL_TOL_M * 1e3:g}); {cm.PRUNE_POINTS}-lowest pruning (UR {pr_ur}, "
                f"Franka {pr_f}; not deployed) {e_pr[0] * 1e3:.2f} / {e_pr[1] * 1e3:.2f} mm; "
                f"torch vs fused {e_torch:.1e} m; graph replay vs eager plan {g_err:.1e}, C "
                f"{c_err:.1e}")


def board_abort_rows() -> list[tuple[Path, dict, int]]:
    """``(episode npz, index row, abort step)`` of every e2 board-clearance abort."""
    out = []
    for pattern in BOARD_ABORT_INDEXES:
        for idx in sorted(V1.glob(pattern)):
            for row in json.loads(idx.read_text())["episodes"]:
                reason = row.get("reason") or ""
                if row.get("status") == "aborted" and reason.startswith(BOARD_ABORT):
                    out.append((idx.parent / row["file"], row,
                                int(reason.rsplit("at step", 1)[1].split()[0])))
    return out


def check_clearance_aborts(cli, T: TorchLcs, lines: list[str]) -> tuple[bool, str]:
    clr = Clearance(_clear_cfg_config(cli), T)
    model = cm.ClearanceModel.load(cli.clearance_model)
    board = model.board()
    rows = board_abort_rows()
    diffs, table, leads, num_err = [], [], [], 0.0
    for path, row, abort in rows:
        with np.load(path, allow_pickle=True) as d:
            ef, eu = d["sim_ee_franka"].astype(np.float64), d["sim_ee_ur"].astype(np.float64)
            step = d["sim_episode_step"].astype(np.int64)
            meas = (d["sim_board_clearance_mm"].astype(np.float64)
                    if "sim_board_clearance_mm" in d.files else None)
        pred = _frame_clearance(clr, ef, eu)[:, 0] * 1e3
        num = np.array([cm.set_distance(board, model.ur_local, cm.pose_mat(u)) for u in eu]) * 1e3
        num_err = max(num_err, float(np.abs(num - pred).max()))
        ref = meas if meas is not None else np.full(len(pred), row["min_board_clearance_mm"])
        diff = np.abs(pred - ref) if meas is not None else np.abs(pred.min() - ref[:1])
        diffs.append(diff)
        fire = np.flatnonzero(pred < clr.ur_min * 1e3)
        lead = abort - int(step[fire[0]]) if len(fire) else -1
        leads.append(lead)
        table.append(f"| {path.relative_to(V1)} | {abort} | {len(pred)} | "
                     f"{np.median(diff):.2f} / {diff.max():.2f} | {pred.min():.2f} / "
                     f"{np.min(ref):.2f} | {lead * lcs_dt():.3f} s ({lead} steps) |")
    med = float(np.median(np.concatenate(diffs))) if diffs else math.nan
    ep_med = [float(np.median(x)) for x in diffs]
    ef_seed, eu_seed, files = _seed_frames()
    seed_min = float(_frame_clearance(clr, ef_seed, eu_seed)[:, 0].min() * 1e3)
    seated = []
    for f in sorted(SEED_EPISODES[0].glob("episode_*.npz")):
        with np.load(f, allow_pickle=True) as d:
            seated.append(_frame_clearance(clr, d["sim_ee_franka"][-13:].astype(np.float64),
                                           d["sim_ee_ur"][-13:].astype(np.float64))[:, 0] * 1e3)
    seated = np.concatenate(seated)
    ok_a = len(rows) >= 16 and med <= C11_MEDIAN_TOL_MM
    ok_b = all(v >= C11_LEAD_STEPS for v in leads)
    ok_c = seed_min >= 0.0
    lines += ["## C11 e2 board-abort episodes (kernel, deployed point set)", "",
              (f"(a) predicted `d_ur` at the recorded `sim_ee_ur` vs the recorded per-frame "
               f"`sim_board_clearance_mm` (`Gripper.measure`): median |diff| {med:.2f} mm over "
               f"{sum(len(x) for x in diffs)} frames of {len(rows)} episodes (tol "
               f"{C11_MEDIAN_TOL_MM:g}); per-episode medians {min(ep_med):.2f}.."
               f"{max(ep_med):.2f} mm; kernel (deployed set) vs numpy full set on these frames "
               f"{num_err:.4f} mm"),
              (f"(b) first frame with predicted `d_ur < {clr.ur_min * 1e3:g} mm` before the "
               f"abort step: min lead {min(leads) * lcs_dt():.3f} s (needs >= "
               f"{C11_LEAD_STEPS * lcs_dt():.2f} s on every episode)"),
              (f"(c) {len(files)} seed episodes: min predicted `d_ur` {seed_min:.2f} mm "
               f"(must be >= 0); the demos' last 13 frames (seated): {seated.min():.2f}.."
               f"{seated.max():.2f} mm, {np.mean(seated < clr.ur_min * 1e3):.0%} under the UR "
               f"threshold {clr.ur_min * 1e3:g} mm"), "",
              ("| episode | abort step | frames | abs diff median / max mm | min pred / "
               "measured mm | lead (pred < min) |"), "|---|---|---|---|---|---|", *table]
    return ok_a and ok_b and ok_c, (
        f"(a) {len(rows)} board-abort episodes, median |pred - measured| {med:.2f} mm (tol "
        f"{C11_MEDIAN_TOL_MM:g}); (b) predicted d_ur < {clr.ur_min * 1e3:g} mm fires "
        f"{min(leads) * lcs_dt():.3f}..{max(leads) * lcs_dt():.3f} s before the abort (>= "
        f"{C11_LEAD_STEPS * lcs_dt():.2f} s on {sum(v >= C11_LEAD_STEPS for v in leads)}/"
        f"{len(leads)}); (c) {len(files)} seed episodes min predicted d_ur {seed_min:.2f} mm")


def lcs_dt() -> float:
    from task_common.lcs_dataset import SIM_DT_S

    return SIM_DT_S


# --- --bank / --policy-screen -------------------------------------------------------------------

def _rec_dirs(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += [f.parent for f in sorted(Path(p).rglob("targets.jsonl"))]
    return out


def collect_states(cli) -> tuple[list[dict], list[str]]:
    """Fixed-goal (ref_mode -1) solves with ``solve_ok`` of recordings made with ``--deploy``."""
    deploy = Path(cli.deploy).resolve()
    pool, notes = [], []
    for rec in _rec_dirs(cli.recordings):
        meta = json.loads((rec / "meta.json").read_text())
        lm = meta.get("learned_mpc") or {}
        if not lm.get("deploy") or Path(lm["deploy"]).resolve() != deploy:
            notes.append(f"skip {rec}: deploy {lm.get('deploy')}")
            continue
        ep_path = rec.parent.parent / f"{rec.name}.npz"
        ep = None
        if ep_path.is_file():
            with np.load(ep_path, allow_pickle=True) as f:
                ep = {k: f[k] for k in ("sim_osc_utime", "sim_latent", "sim_ee_franka",
                                        "sim_ee_ur", "sim_h_median_mm", "sim_stage")}
            row_of = {int(u): i for i, u in enumerate(ep["sim_osc_utime"])}
        with open(rec / "targets.jsonl") as f:
            for line in f:
                if DEBUG_CH not in line:
                    continue
                item = json.loads(line)
                if item["channel"] != DEBUG_CH:
                    continue
                p = item["payload"]
                B = _blocks(p)
                s = dict(zip(p["blocks"]["scalars"]["datatypes"], B["scalars"][:, 0],
                             strict=True))
                if s["solve_ok"] != 1.0 or s["ref_mode"] != -1.0:
                    continue
                st = {"rec": str(rec), "utime": int(p["utime"]), "stage": int(s["stage"]),
                      "z0": B["x_sol"][:16, 0], "p0": B["x_sol"][16:22, 0],
                      "goal": B["z_ref"][:, 0], "u_c3": B["u_sol"].T, "rot0": np.zeros((2, 3)),
                      "p_start": None, "h_mm": math.nan, "source": "x_sol", "q_now": None}
                if ep is not None and st["utime"] in row_of:
                    i = row_of[st["utime"]]
                    ef, eu = ep["sim_ee_franka"], ep["sim_ee_ur"]
                    st.update(z0=ep["sim_latent"][i], p0=np.r_[ef[i, :3], eu[i, :3]],
                              rot0=np.stack([delta_rotvec(ef[0, 3:7], ef[i, 3:7]),
                                             delta_rotvec(eu[0, 3:7], eu[i, 3:7])]),
                              p_start=np.r_[ef[0, :3], eu[0, :3]],
                              h_mm=float(ep["sim_h_median_mm"][i]), source="episode",
                              q_now=np.stack([ef[i, 3:7], eu[i, 3:7]]))
                pool.append(st)
    return pool, notes


def _with_ceiling(params: MppiParams, st: dict) -> MppiParams:
    if st["p_start"] is None or not math.isfinite(params.z_ceiling_above_start_m):
        return params
    c = params.z_ceiling_above_start_m
    return dataclasses.replace(params, z_max_franka=min(params.z_max_franka, st["p_start"][2] + c),
                               z_max_ur=min(params.z_max_ur, st["p_start"][5] + c))


def run_bank(cli) -> int:
    lcs = TorchLcs.from_deploy(cli.deploy, cli.device)
    live = load_params(cli)
    pool, notes = collect_states(cli)
    _require(len(pool) >= cli.n, f"only {len(pool)} fixed-goal solves found ({notes[:3]})")
    rng = np.random.default_rng(cli.seed)
    states = [pool[i] for i in sorted(rng.choice(len(pool), cli.n, replace=False))]
    # Like for like with C3: its horizon, Q/R, bounds and floors (no ceiling/envelope), no
    # smoothness, cold start.
    bank = dataclasses.replace(live, H=lcs.N, w_s=0.0, u_lb=None, u_ub=None, u_box_outer=None,
                               u_translations_only=False, z_ceiling_above_start_m=math.inf,
                               rot_envelope_deg={"franka": math.inf, "ur": math.inf})
    c3_obj = dataclasses.replace(bank, w_T=0.0)
    gen = torch.Generator(device=lcs.device).manual_seed(cli.seed)
    lb, ub = bank.bounds(lcs)
    rows = []
    for st in states:
        row = {"rec": Path(st["rec"]).name, "utime": st["utime"], "stage": st["stage"],
               "h_mm": st["h_mm"], "source": st["source"]}
        for tag, prm in (("", bank), ("c3obj_", c3_obj)):
            prm = _with_ceiling(prm, st)
            args = (st["z0"], st["p0"], st["rot0"], st["goal"])
            q = {"q_now": st["q_now"]} if prm.clearance else {}
            c3_cost, c3_bad, _ = rollout_cost(lcs, prm, *args, st["u_c3"], **q)
            U, info = mppi_plan(lcs, prm, *args, None, gen, **q)
            m_cost, m_bad, _ = rollout_cost(lcs, prm, *args, U, **q)
            row.update({f"{tag}c3_cost": c3_cost, f"{tag}c3_viol": c3_bad,
                        f"{tag}mppi_cost": m_cost, f"{tag}mppi_viol": m_bad,
                        f"{tag}ratio": m_cost / c3_cost, f"{tag}solve_ms": info["solve_ms"]})
            if prm.clearance:
                row.update({f"{tag}{k}": info[k] for k in (
                    "frac_clear_penalised", "clear_min_ur_mm", "clear_min_f_mm",
                    "clear_now_ur_mm")})
            if not tag:
                row["mppi_u0"] = U[0].tolist()
        row["c3_u0"] = st["u_c3"][0].tolist()
        row["c3_out_of_box"] = int(np.sum((st["u_c3"] < lb - 1e-9) | (st["u_c3"] > ub + 1e-9)))
        # Live config with this state as the episode start (envelope/ceiling about it).
        Ul, info = mppi_plan(lcs, _with_ceiling(live, {**st, "p_start": st["p0"]}), st["z0"],
                             st["p0"], None, st["goal"], None, gen,
                             **({"q_now": st["q_now"]} if live.clearance else {}))
        row.update(envelope_samples(lcs, live, st["rot0"], gen))
        row.update(live_u0=Ul[0].tolist(), live_solve_ms=info["solve_ms"],
                   live_frac_env=info["frac_env_penalised"],
                   live_goal_dist_H=info["goal_dist_H"],
                   dist0=lcs.whitened_dist(st["z0"], st["goal"]))
        rows.append(row)
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = bank_summary(rows)
    if cli.policy_screen:
        summary["policy_screen"] = policy_screen(rows)
    doc = {"deploy": str(Path(cli.deploy).resolve()), "config": str(cli.config),
           "recordings": [str(p) for p in cli.recordings], "n_pool": len(pool), "seed": cli.seed,
           "overrides": cli.override, "bank_params": bank.to_dict(),
           "live_params": live.to_dict(), "summary": summary,
           "notes": notes, "rows": rows}
    (out / "bank.json").write_text(json.dumps(doc, indent=1, default=float))
    (out / "bank.md").write_text(bank_markdown(doc))
    print((out / "bank.md").read_text(), flush=True)
    return 0


def envelope_samples(lcs: TorchLcs, params: MppiParams, rot0: np.ndarray,
                     gen: torch.Generator) -> dict:
    """Fraction of K first-pass samples (zero mean, the live box and sigma) outside the
    envelope, with ``rot0`` = the state's measured offset from its episode start: the old
    accumulator (|rot0| + sum of per-step |w|) vs the composed geodesic angle."""
    lb, ub = params.bounds(lcs)
    sigma = lcs.tensor(params.sigma_frac * (ub - lb) / 2.0)
    eps = torch.randn((params.K, params.H, lcs.n_u), generator=gen, device=lcs.device)
    U = torch.maximum(torch.minimum(sigma * eps, lcs.tensor(ub)), lcs.tensor(lb))
    env = lcs.tensor(params.rot_envelope_rad())
    new = lcs.rotation_angles(U, rot0)[:, 1:]
    steps = torch.stack([U[..., 6:9].norm(dim=-1), U[..., 9:12].norm(dim=-1)], -1)
    old = lcs.tensor(np.linalg.norm(rot0, axis=1)) + torch.cumsum(steps, 1)
    frac = {k: float((v > env).any(-1).any(-1).double().mean()) for k, v in
            (("env_old", old), ("env_new", new))}
    return {**frac, "rot0_deg": np.degrees(np.linalg.norm(rot0, axis=1)).tolist()}


def _stats(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    return {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def bank_summary(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for tag in ("", "c3obj_"):
        r = np.array([row[f"{tag}ratio"] for row in rows])
        out[f"{tag}frac_mppi_le_c3"] = float(np.mean(r <= 1.0))
        out[f"{tag}median_ratio"] = float(np.median(r))
        feas = np.array([row[f"{tag}ratio"] for row in rows if row[f"{tag}c3_viol"] == 0])
        out[f"{tag}n_c3_feasible"] = len(feas)
        out[f"{tag}frac_mppi_le_c3_feasible"] = float(np.mean(feas <= 1.0)) if len(feas) else None
        out[f"{tag}median_ratio_feasible"] = float(np.median(feas)) if len(feas) else None
        out[f"{tag}c3_plans_violating"] = int(sum(row[f"{tag}c3_viol"] > 0 for row in rows))
        out[f"{tag}mppi_plans_violating"] = int(sum(row[f"{tag}mppi_viol"] > 0 for row in rows))
        out[f"{tag}solve_ms"] = _stats([row[f"{tag}solve_ms"] for row in rows])
    out["live_solve_ms"] = _stats([row["live_solve_ms"] for row in rows])
    for k in ("env_old", "env_new", "live_frac_env"):
        out[f"{k}_mean"] = float(np.mean([row[k] for row in rows]))
    out["c3_inputs_out_of_box"] = int(sum(row["c3_out_of_box"] for row in rows))
    if "frac_clear_penalised" in rows[0]:
        out["frac_clear_penalised_mean"] = float(np.mean([r["frac_clear_penalised"]
                                                          for r in rows]))
        out["clear_now_ur_mm"] = _stats([r["clear_now_ur_mm"] for r in rows])
        out["clear_min_ur_mm_plan"] = _stats([r["clear_min_ur_mm"] for r in rows])
        out["clear_min_ur_mm_plan"]["min"] = float(min(r["clear_min_ur_mm"] for r in rows))
    return out


def policy_screen(rows: list[dict]) -> dict:
    over = [r for r in rows if r["stage"] == 1 and OVER_H_MM[0] <= r["h_mm"] <= OVER_H_MM[1]]
    out = {"n_all": len(rows), "n_riding_over": len(over),
           "riding_over_rule": f"stage 1 and {OVER_H_MM[0]:g} <= h_median_mm <= {OVER_H_MM[1]:g}"}
    for key in ("live_u0", "mppi_u0", "c3_u0"):
        for name, sel in (("all", rows), ("over", over)):
            if not sel:
                continue
            U = np.array([r[key] for r in sel])
            out[f"{key}_{name}"] = {
                "both_down": float(np.mean((U[:, FZ] < 0) & (U[:, UZ] < 0))),
                "both_up": float(np.mean((U[:, FZ] > 0) & (U[:, UZ] > 0))),
                "fz_mean_mm": float(U[:, FZ].mean() * 1e3),
                "uz_mean_mm": float(U[:, UZ].mean() * 1e3)}
    return out


def bank_markdown(doc: dict) -> str:
    s, bp = doc["summary"], doc["bank_params"]
    lines = [
        "# MPPI vs C3 on recorded V2 solve states", "",
        (f"- deploy `{doc['deploy']}`; {s['n']} states sampled (seed {doc['seed']}) from "
         f"{doc['n_pool']} fixed-goal solves in {', '.join(doc['recordings'])}"),
        ("- cost = MPPI cost at C3's horizon N 7, C3's Q (w_q 1 / z_std^2) and R (w_r 0.3 / "
         "u_half^2), w_s 0, the deploy's bounds, C3's floors as penalties (no ceiling/envelope: "
         "C3 has none); both plans "
         "rolled out through the same LCS from the same x_0; MPPI cold-started (no warm start)"),
        (f"- MPPI: K {bp['K']}, n_iter {bp['n_iter']}, "
         + (f"ESS-adaptive temperature (ess_frac {bp['ess_frac']})" if bp["ess_frac"]
            else f"median-normalised temperature {bp['temperature']}")
         + f", fused rollout {bp['fused_rollout']}, CUDA graph {bp['cuda_graph']}; overrides "
         f"{doc.get('overrides') or 'none'}"),
        "", ("| cost | MPPI <= C3 | median MPPI/C3 | same, C3-feasible states | C3 plans "
             "violating | MPPI plans violating | MPPI solve p50 / p95 ms |"),
        "|---|---|---|---|---|---|---|"]
    for tag, label in (("", "w_T 5 (MPPI cost)"), ("c3obj_", "w_T 0 (C3 objective)")):
        st = s[f"{tag}solve_ms"]
        lines.append(f"| {label} | {s[f'{tag}frac_mppi_le_c3']:.0%} | "
                     f"{s[f'{tag}median_ratio']:.3f} | "
                     f"{s[f'{tag}frac_mppi_le_c3_feasible'] or 0:.0%} / "
                     f"{s[f'{tag}median_ratio_feasible'] or 0:.3f} (n "
                     f"{s[f'{tag}n_c3_feasible']}) | {s[f'{tag}c3_plans_violating']} | "
                     f"{s[f'{tag}mppi_plans_violating']} | {st['p50']:.1f} / {st['p95']:.1f} |")
    lv = s["live_solve_ms"]
    lines += ["", (f"- live config (K {doc['live_params']['K']}, H {doc['live_params']['H']}, "
                   f"{doc['live_params']['n_iter']} iterations) solve p50 {lv['p50']:.1f} / p95 "
                   f"{lv['p95']:.1f} / max {lv['max']:.1f} ms (offline, no sim of ours running)"),
              f"- C3 inputs outside the deploy box: {s['c3_inputs_out_of_box']}",
              (f"- rotation envelope, live config samples (zero-mean first pass, start offset = "
               f"the state's from its episode): penalised {s['env_old_mean']:.1%} with the old "
               f"summed-norm accumulator vs {s['env_new_mean']:.1%} composed; last MPPI pass "
               f"(state as start) {s['live_frac_env_mean']:.1%}")]
    if "frac_clear_penalised_mean" in s:
        cn, cp = s["clear_now_ur_mm"], s["clear_min_ur_mm_plan"]
        lines += [(f"- clearance {bp['clearance']}: last-pass samples penalised "
                   f"{s['frac_clear_penalised_mean']:.1%} (mean over states); UR clearance at the "
                   f"states p50 {cn['p50']:.1f} mm; MPPI plan min (knots 1..N) p50 "
                   f"{cp['p50']:.1f} / min {cp['min']:.1f} mm; C3 / MPPI plans violating above "
                   "include clearance knots")]
    ps = s.get("policy_screen")
    if ps:
        lines += ["", "## Policy screen (u_0 Franka-z / UR-z)", "",
                  (f"riding over = {ps['riding_over_rule']} (sim GT, report only): "
                   f"{ps['n_riding_over']} of {ps['n_all']} states"), "",
                  "| plan | states | both down | both up | Fz mean mm | Uz mean mm |",
                  "|---|---|---|---|---|---|"]
        for key, label in (("live_u0", "MPPI live config"), ("mppi_u0", "MPPI bank (H 7)"),
                           ("c3_u0", "C3 recorded")):
            for name in ("over", "all"):
                v = ps.get(f"{key}_{name}")
                if v:
                    lines.append(f"| {label} | {name} | {v['both_down']:.0%} | "
                                 f"{v['both_up']:.0%} | {v['fz_mean_mm']:+.2f} | "
                                 f"{v['uz_mean_mm']:+.2f} |")
    return "\n".join(lines) + "\n"


# --- --smoke ------------------------------------------------------------------------------------

class Listener:
    """Every LATENT_STATE / plan / UR line / debug message on the URL, in a thread."""

    def __init__(self, url: str) -> None:
        self.lc = lcm.LCM(url)
        self.latents: dict[int, tuple] = {}
        self.franka: list = []
        self.ur: list = []
        self.debug: list = []
        for ch in (LATENT_STATE_CHANNEL, FRANKA_CH, UR_CH, DEBUG_CH):
            self.lc.subscribe(ch, self._on)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _on(self, ch: str, data: bytes) -> None:
        msg = lcmt_timestamped_saved_traj.decode(data)
        if ch == LATENT_STATE_CHANNEL:
            utime, t, z, ee_f, ee_u, _ = parse_latent_state_message(msg)
            self.latents[utime] = (utime, t, z, ee_f, ee_u)
        elif ch == FRANKA_CH:
            self.franka.append((msg, data))
        elif ch == UR_CH:
            self.ur.append((msg, data))
        else:
            self.debug.append(msg)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.lc.handle_timeout(50)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _spawn(argv: list[str], log: Path, cwd: Path = REPO_ROOT) -> subprocess.Popen:
    f = open(log, "w")  # noqa: SIM115
    return subprocess.Popen(argv, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})


def _stop(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print(f"[smoke] {name} pid {proc.pid} ignored SIGINT; killing", flush=True)
            proc.kill()
            proc.wait(timeout=10)


def run_smoke(cli) -> int:
    """Closed loop on a private URL: magna's executors (Franka OSC, UR trajectory controller +
    UR control sim, worktree binaries) + the LCM sim restored to ``--start-state`` + the encoder
    node + the MPPI node."""
    url = check_private_url(cli.lcm_url)
    port = url_port(url)
    _require(not pgrep_others(port), f"processes already on {port}: {pgrep_others(port)}")
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=False)
    py = sys.executable
    bins = {name: MAGNA_WORKTREE / "bazel-bin" / "systems" / rel
            for name, rel in EXECUTORS.items()}
    missing = [str(b) for b in bins.values() if not b.is_file()]
    _require(not missing, f"executor binaries missing: {missing}")
    lis = Listener(url)
    procs: list[tuple[subprocess.Popen, str]] = []
    rc = None
    try:
        for name, b in bins.items():
            procs.append((_spawn([str(b), f"--lcm_url={url}", *EXECUTOR_ARGS[name]],
                                 out / f"{name}.log", cwd=MAGNA_WORKTREE), name))
        time.sleep(2.0)
        procs.append((_spawn([py, "scripts/lcs/mppi_offline.py", "--smoke-sim", "--lcm-url", url,
                              "--start-state", str(cli.start_state)], out / "sim.log"), "sim"))
        enc_extra = (["--belt-seg"] if cli.belt_seg else []) + (
            ["--cloud-noise", cli.cloud_noise] if cli.cloud_noise is not None else [])
        procs.append((_spawn([py, "scripts/lcs/latent_encoder_node.py", "--lcm-url", url,
                              "--deploy", str(cli.deploy), "--ur-state-channel", "UR_STATE_SIM",
                              "--state-match", "exact", *enc_extra], out / "encoder.log"),
                      "encoder"))
        t_end = time.perf_counter() + cli.start_timeout_s
        while len(lis.latents) < 3 and time.perf_counter() < t_end:
            dead = [n for pr, n in procs if pr.poll() is not None]
            _require(not dead, f"{dead} exited early")
            time.sleep(0.5)
        _require(len(lis.latents) >= 3, f"no latents within {cli.start_timeout_s} s")
        print(f"[smoke] latents flowing ({len(lis.latents)}); starting the MPPI node", flush=True)
        n_lat0 = len(lis.latents)
        cfg = cli.config
        if cli.override:
            cfg = out / "mppi_smoke.yaml"
            cfg.write_text(yaml.safe_dump(load_params(cli).to_dict()))
        node_argv = [py, "scripts/lcs/mppi_controller_node.py", "--lcm-url", url, "--deploy",
                     str(cli.deploy), "--config", str(cfg), "--max-duration-s",
                     str(cli.duration_s), "--stages", cli.stages, "--debug-channel", DEBUG_CH,
                     "--log", str(out / "node.jsonl")]
        if cli.clearance_model is not None:
            node_argv += ["--clearance-model", str(Path(cli.clearance_model).resolve())]
        node_argv += smoke_node_flags(cli)
        node = _spawn(node_argv, out / "node.log")
        procs.append((node, "node"))
        gpu = GpuSampler()
        rc = node.wait(timeout=cli.duration_s * 20 + 120)
        gpu.stop()
        time.sleep(0.3)
    finally:
        for proc, name in reversed(procs):
            _stop(proc, name)
        lis.stop()
    return smoke_report(cli, out, lis, rc, n_lat0, node_argv, gpu.summary())


def smoke_node_flags(cli) -> list[str]:
    """Node flags from ``--override`` ``NODE_KEYS``; the prior on a waypoints goal without
    ``demo_actions`` -> ``--demo-actions auto``."""
    over = node_overrides(cli)
    prm = load_params(cli)
    if ("demo_actions" not in over and str(over.get("goal", "")).startswith("waypoints:")
            and (prm.prior_alpha > 0.0 or prm.prior_weight > 0.0)):
        over["demo_actions"] = "auto"
    return [x for k, v in over.items() for x in (f"--{k.replace('_', '-')}", str(v))]


def smoke_stage_plan(cli, lcs: TorchLcs) -> tuple[list[int], int]:
    """``(expected debug stage rows in order, number of stage markers)`` of a smoke run."""
    over = node_overrides(cli)
    goal = str(over.get("goal", "deploy"))
    if over.get("goal_mode") == "track":
        return [0], 1
    if goal.startswith("waypoints:") and cli.stages == "all":
        n = len(node_mod.Waypoints.load(goal.split(":", 1)[1]).frames)
        return list(range(n)), n
    n_goals = len(lcs.stage_goals)
    if goal.startswith("waypoints:"):
        n_goals = len(node_mod.Waypoints.load(goal.split(":", 1)[1]).frames)
    return ([n_goals - 1], 1) if cli.stages == "last" else (list(range(n_goals)), n_goals)


class GpuSampler:
    """``nvidia-smi`` GPU utilisation every 0.5 s while the node runs (report only)."""

    def __init__(self) -> None:
        self.util: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, timeout=5, check=True).stdout
                self.util.append(float(out.split()[0]))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                return

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=6.0)

    def summary(self) -> dict | None:
        v = np.asarray(self.util)
        return {"n": len(v), "mean": float(v.mean()), "p50": float(np.median(v)),
                "max": float(v.max())} if len(v) else None


def run_smoke_sim(cli) -> int:
    """The LCM sim (cameras, cloud, belt state) restored to ``--start-state``, until SIGINT.

    Publishes a hold at the restored poses (metadata ``smoke_hold``) until the MPPI node's first
    Franka plan, so the executors hold the start instead of letting the arms sag.
    """
    import newton

    from round_belt_task.arm_kinematics import FrankaTip, UrTracking
    from round_belt_task.episode_io import pose7_mat
    from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
    from task_common import sim_snapshot

    url = check_private_url(cli.lcm_url)
    args = RoundBeltLcmSimulation.create_parser().parse_args(
        ["--viewer", "null", "--cameras", "--publish-point-cloud", "--publish-belt-state",
         "--lcm-url", url])
    sim = RoundBeltLcmSimulation(newton.viewer.ViewerNull(num_frames=100), args)
    sim_snapshot.restore(sim, sim_snapshot.load(cli.start_state))
    ee_f = pose7_mat(FrankaTip.fk(sim._io_by_name["franka"].positions))
    ee_u = pose7_mat(UrTracking.fk(sim._io_by_name["ur10"].positions))
    print(f"[smoke-sim] restored {cli.start_state} at step {sim.step_index}: Franka tip "
          f"{np.round(ee_f[:3], 4).tolist()}, UR tracking {np.round(ee_u[:3], 4).tolist()}",
          flush=True)
    stop = threading.Event()
    threading.Thread(target=_hold_until_node, args=(url, sim, ee_f, ee_u, stop),
                     daemon=True).start()
    try:
        sim.run(0, True)
    finally:
        stop.set()
    return 0


def _hold_until_node(url: str, sim, ee_f, ee_u, stop: threading.Event) -> None:
    lc = lcm.LCM(url)
    seen = threading.Event()

    def on_plan(_ch, data):
        if (lcmt_timestamped_saved_traj.decode(data).saved_traj.metadata.name
                == node_mod.METADATA_NAME):
            seen.set()

    lc.subscribe(FRANKA_CH, on_plan)
    X_base_W = node_mod._rigid_inv(ur_base_world())
    X_tr_t0 = node_mod._rigid_inv(x_tool0_tracking())
    zero = np.zeros((node_mod.N_KNOTS, 12))
    n = 0
    while not (stop.is_set() or seen.is_set()):
        utime = max(1, int(sim.step_index) * 5000)
        t, dt = utime * 1e-6, 0.075
        for ch, msg in ((FRANKA_CH, node_mod.franka_message(utime, t, dt, ee_f, zero)),
                        (UR_CH, node_mod.ur_message(utime, t, dt, ee_u, zero[0], X_base_W,
                                                    X_tr_t0))):
            msg.saved_traj.metadata.name = "smoke_hold"
            lc.publish(ch, msg.encode())
        n += 1
        lc.handle_timeout(50)
    print(f"[smoke-sim] hold released after {n} publishes (node plan seen {seen.is_set()})",
          flush=True)


def smoke_report(cli, out: Path, lis: Listener, rc, n_lat0: int, node_argv,
                 gpu: dict | None = None) -> int:
    lcs = TorchLcs.from_deploy(cli.deploy, "cpu")
    lb, ub = load_params(cli).bounds(lcs)
    X_t0_tr = x_tool0_tracking()
    ur = UrDecoder()
    mine_f = [(m, d) for m, d in lis.franka if m.saved_traj.metadata.name == node_mod.METADATA_NAME]
    mine_u = [(m, d) for m, d in lis.ur if m.saved_traj.metadata.name == node_mod.METADATA_NAME]
    bad_f = [m.utime for m, _ in mine_f
             if len(parse_saved_traj_message(m)[2]) != 7
             or parse_saved_traj_message(m)[2][0] != m.utime * 1e-6
             or m.utime not in lis.latents]
    lines = {}
    bad_u = []
    for m, d in mine_u:
        line = ur.decode(d)
        if line is None or abs(line.t1 - line.t0 - lcs.dt) > SPAN_TOL or m.utime not in lis.latents:
            bad_u.append(m.utime)
        elif m.utime not in lines:
            lines[m.utime] = line
    plans = {}
    for m, _ in mine_f:
        plans.setdefault(m.utime, m)
    u0s, out_box = [], 0
    for utime, m in plans.items():
        if utime in lines and utime in lis.latents:
            u0 = executed_u0(m, lines[utime], lis.latents[utime][4], X_t0_tr)
            u0s.append(u0)
            out_box += int(np.any(u0 < lb - 1e-7) or np.any(u0 > ub + 1e-7))
    dbg_err: dict[str, float] = {}
    dbg_fail = []
    for msg in lis.debug:
        try:
            e = verify_debug(msg, plans[msg.utime], lines[msg.utime], lis.latents[msg.utime],
                             X_t0_tr)
            for k, v in e.items():
                dbg_err[k] = max(dbg_err.get(k, 0.0), v)
        except (AssertionError, KeyError) as exc:
            dbg_fail.append(f"{msg.utime}: {exc!r}")
    text = (out / "node.log").read_text(errors="replace").splitlines()
    want_rows, n_marks = smoke_stage_plan(cli, lcs)
    want = (*SMOKE_MARKERS[:2], *(f"[learned-mpc] stage {k} " for k in range(n_marks)),
            *SMOKE_MARKERS[3:])
    marks = [next((i for i, ln in enumerate(text) if m in ln), -1) for m in want]
    markers_ok = all(i >= 0 for i in marks) and marks == sorted(marks)
    rows = []
    for m in lis.debug:
        b = {x.trajectory_name: x for x in m.saved_traj.trajectories}["scalars"]
        rows.append(int(dict(zip(b.datatypes, np.asarray(b.datapoints)[:, 0]))["stage"]))
    runs = [[v, sum(1 for _ in g)] for v, g in itertools.groupby(rows)]
    stages_ok = [v for v, _ in runs] == want_rows
    solve = [json.loads(ln) for ln in (out / "node.jsonl").read_text().splitlines()]
    ms = np.array([r["solve_ms"] for r in solve if r["solve_ms"] == r["solve_ms"]])
    rep = {
        "node_argv": node_argv, "exit_code": rc, "latents_seen": len(lis.latents),
        "latents_before_node": n_lat0, "franka_msgs": len(mine_f), "ur_msgs": len(mine_u),
        "foreign_franka": len(lis.franka) - len(mine_f), "foreign_ur": len(lis.ur) - len(mine_u),
        "distinct_plans": len(plans), "bad_franka": bad_f[:10], "n_bad_franka": len(bad_f),
        "bad_ur": bad_u[:10], "n_bad_ur": len(bad_u), "u0_checked": len(u0s),
        "u0_out_of_box": out_box, "debug_msgs": len(lis.debug), "debug_fail": dbg_fail[:5],
        "debug_max_err": dbg_err, "markers_line_idx": dict(zip(want, marks, strict=True)),
        "stages": cli.stages, "debug_stage_runs": runs, "debug_stages_ok": stages_ok,
        "markers_in_order": markers_ok, "solves": len(solve),
        "solve_ms": _stats(ms) if len(ms) else None,
        "fallbacks": int(sum(bool(r.get("fallback")) for r in solve)),
        "plans_violating": int(sum(r.get("plan_violations", 0) > 0 for r in solve)),
        "frac_penalised_mean": float(np.mean([r.get("frac_penalised", 0.0) for r in solve])),
        "ess_mean": float(np.mean([r.get("ess", np.nan) for r in solve])),
        "p0_first": solve[0]["p0"] if solve else None,
        "p0_z_min": (np.min([r["p0"][2] for r in solve]), np.min([r["p0"][5] for r in solve]))
        if solve else None,
        "floors": [load_params(cli).z_min_franka, load_params(cli).z_min_ur],
        "first_latent_z": ([lis.latents[min(lis.latents)][3][2], lis.latents[min(lis.latents)][4][2]]
                           if lis.latents else None),
        "latent_z_min": ([min(v[3][2] for v in lis.latents.values()),
                          min(v[4][2] for v in lis.latents.values())] if lis.latents else None),
        "u_box": [lb.tolist(), ub.tolist()], "gpu_util_pct": gpu,
        "stage_markers": [list(st) for st in map(parse_stage_line, text) if st is not None],
        "stage0_dist_trace": [[round(r["t"] - solve[0]["t"], 3), round(r["dist"], 4)]
                              + ([r["reach_streak"]] if "reach_streak" in r else [])
                              for r in solve if r["stage"] == 0],
    }
    clear_ok = True
    if cli.clearance_model is not None:
        ok_rows = [r for r in solve if r.get("solve_ok", 1.0) == 1.0]
        ur = np.array([r["clear_min_ur_mm"] for r in ok_rows])
        ur_min = Clearance(_clear_cfg_config(cli), lcs).ur_min * 1e3
        clear_ok = len(ur) == len(ok_rows) > 0 and bool(np.all(ur >= ur_min))
        rep["clearance"] = {
            "model": str(cli.clearance_model), "ur_min_mm": ur_min,
            "plan_min_ur_mm": _stats(ur) | {"min": float(ur.min())} if len(ur) else None,
            "plan_min_f_mm_min": float(min(r["clear_min_f_mm"] for r in ok_rows)),
            "now_ur_mm_min": float(min(r["clear_now_ur_mm"] for r in ok_rows)),
            "plans_under_ur_min": int(np.sum(ur < ur_min)),
            "plans_violating_clearance": int(sum(r["clear_violating"] > 0 for r in ok_rows)),
            "frac_clear_penalised_mean": float(np.mean([r["frac_clear_penalised"]
                                                        for r in ok_rows])),
            "pass": clear_ok}
    ok = (rc == 0 and len(mine_f) >= 100 and len(mine_u) >= 100 and not bad_f and not bad_u
          and out_box == 0 and len(u0s) > 0 and markers_ok and not dbg_fail
          and len(lis.debug) > 0 and stages_ok and clear_ok)
    if node_overrides(cli).get("goal_mode") == "track" or "idx" in (solve[0] if solve else {}):
        rep["progress"] = smoke_progress(cli, solve, lcs.dt)
        ok = ok and rep["progress"]["pass"]
    rep["node_flags"] = smoke_node_flags(cli)
    rep["pass"] = ok
    (out / "smoke.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float), flush=True)
    print(f"[{'PASS' if ok else 'FAIL'}] smoke", flush=True)
    return 0 if ok else 1


def smoke_progress(cli, solve: list[dict], dt: float) -> dict:
    """Progress index of the node log: non-decreasing, advancing, holds within the budget."""
    over = node_overrides(cli)
    cap = float(over.get("track_max_hold_s", 3.0))
    idx = [r["idx"] for r in solve if "idx" in r]
    held = [r.get("held_s", 0.0) for r in solve if "idx" in r]
    d = [r["d_idx"] for r in solve if r.get("d_idx") is not None]
    span_s = solve[-1]["t"] - solve[0]["t"] if len(solve) > 1 else 0.0
    ok = (len(idx) == len(solve) > 1 and all(b >= a for a, b in itertools.pairwise(idx))
          and idx[-1] > idx[0] and max(held) <= cap + 1e-9)
    return {"n": len(idx), "idx_first": idx[0] if idx else None,
            "idx_last": idx[-1] if idx else None, "span_s": span_s,
            "frames_elapsed": span_s / dt,
            "held_s_max": max(held) if held else None, "hold_cap_s": cap,
            "d_idx": _stats(d) if d else None, "pass": ok}


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true")
    p.add_argument("--bank", action="store_true")
    p.add_argument("--policy-screen", action="store_true", help="with --bank")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-sim", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--stages", choices=("last", "all"), default="last", help="--smoke: node")
    p.add_argument("--start-state", type=Path, default=DEFAULT_START_STATE,
                   help="--smoke: sim snapshot the sim starts from (the harness default)")
    p.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--recordings", type=Path, nargs="+", default=[])
    p.add_argument("--n", type=int, default=50, help="bank states")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="MppiParams field over the config (yaml value), repeatable")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--lcm-url", default=None, help="--smoke: private URL")
    p.add_argument("--duration-s", type=float, default=10.0, help="--smoke: node stage length")
    p.add_argument("--start-timeout-s", type=float, default=300.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--clearance-model", type=Path, default=None,
                   help="--check: C0/C10/C11 on this clearance_model npz; --smoke: the node's "
                        "--clearance-model")
    p.add_argument("--belt-seg", action="store_true",
                   help="--smoke: encoder node --belt-seg (belt-cloud deploys)")
    p.add_argument("--cloud-noise", default=None, metavar="PRESET|JSON",
                   help="--smoke: encoder node --cloud-noise (its default if unset)")
    p.add_argument("--validation-md", type=Path, default=None,
                   help="--check: write the C10/C11 tables here")
    return p


def main() -> int:
    cli = create_parser().parse_args()
    if cli.smoke_sim:
        return run_smoke_sim(cli)
    if not (cli.check or cli.bank or cli.policy_screen or cli.smoke):
        create_parser().error("choose --check, --bank [--policy-screen] or --smoke")
    rc = 0
    if cli.check:
        rc |= run_check(cli)
    if cli.bank or cli.policy_screen:
        if not cli.recordings or cli.out is None:
            create_parser().error("--bank/--policy-screen need --recordings and --out")
        cli.policy_screen = True if not cli.bank else cli.policy_screen
        rc |= run_bank(cli)
    if cli.smoke:
        if not cli.lcm_url or cli.out is None:
            create_parser().error("--smoke needs --lcm-url and --out")
        rc |= run_smoke(cli)
    return rc


if __name__ == "__main__":
    sys.exit(main())
