#!/usr/bin/env python3
"""MPPI controller node on the learned latent LCS: ``LATENT_STATE`` in, the C++ learned
controller's two trajectories out (Franka 7-knot plan, UR 2-knot line over exactly dt).

Same stdout markers as the C++ controller (``magna_process.CONTROLLER_MARKERS``); replans on
every new latent utime, republishes the plan in force every 50 ms; optional
``LEARNED_MPC_DEBUG`` in the C++ layout. Metadata name ``mppi_controller`` (the bridge drops
``lcs_collector`` as its own loop-back). Reads only ``LATENT_STATE``.

Opt-in transport along a demo (``latent_waypoints.py`` artifact): ``--goal waypoints:PATH``
(``--stages all``: one stage per waypoint), ``--goal-mode track`` (per-knot goals = the demo's
per-frame latents from a time-indexed progress index with a bounded hold), ``--demo-actions``
(the ``prior_alpha`` / ``prior_weight`` demo action prior). Off by default.
Config ``prior_sync: stage`` (stage mode, off by default): the prior's demo index follows the
stage (``prior_sync_index``) instead of the wall clock.
Config ``prior_mode: policy`` + ``bc_prior: PATH`` (off by default): the prior sequence is the BC
policy rolled through the LCS from the current state (``mppi_lcs.PolicyPrior``) instead of the
demo's actions; JSONL ``prior_u0`` / ``prior_ms``.
``--waypoint-bank BANK.json`` (``latent_waypoints.py --bank-out``): at the first latent the
reference demo (waypoints, tols, action prior with ``--demo-actions auto``) becomes the bank
entry whose initial latent is nearest; ``--goal waypoints:PATH`` is the base until then.
Config ``overshoot`` (``mppi_lcs.OvershootGuard``, off by default): on the latent distance to
the seated (last) goal, fade the demo prior's actions to zero, then hold (prior off, scaled
weights, optional entry / best-latent goal); JSONL ``d_seat`` / ``prior_scale`` / ``mode``.
Config ``support_box`` (``MppiParams.support_box``, off by default): soft data-support box on the
EE positions; JSONL ``sbox_mm`` = [max mm outside now, max mm outside over the plan's knots].
Config ``goal_set`` / ``goal_set_tau`` (``MppiParams.goal_set``, off by default; stage mode): in
the last stage the cost and the "reached" distance (JSONL ``dist``) use the soft-min over the set
of seated latents instead of the stage goal; JSONL ``d_single`` = the distance to the stage goal.
Config ``stage_skip_ahead`` / ``stage_no_timeout_into`` (off by default): jump to the furthest
later stage whose tol holds the latent; a gated stage is entered only on reaching the previous
one, and until then the demo prior is zero from the gate's waypoint frame on (stdout ``[mppi]
skip ahead`` / ``timeout blocked``, JSONL ``blocked``).
Config ``stage_start: progress`` (off by default; stage mode, a latch waypoint): at the first
latent the stage index becomes ``progress_start`` over the pre-latch stages (none: no-op; below a
``stage_no_timeout_into`` gate's previous one; stdout ``[mppi] start stage``, JSONL ``start`` on
the first solve).
Config ``value`` (``MppiParams.value``, off by default): + weight x the learned cost-to-go V(z)
(``train_value.py``; refused unless trained on this deploy's encoder / z_std) in every stage;
JSONL ``v_now`` = V(current latent), ``value_H`` = V(plan's last knot).

    uv run --frozen python scripts/lcs/mppi_controller_node.py \\
        --lcm-url 'udpm://239.255.76.162:7762?ttl=0' --config data/lcs/realfirst/v1/mppi.yaml \\
        --deploy <deploy.npz> [--goal demo_goals:<demo_goals.npz> --stages all]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lcm
import torch
from latent_waypoints import WaypointBank, Waypoints, load_actions

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from dairlib import (
    lcmt_metadata,
    lcmt_saved_traj,
    lcmt_timestamped_saved_traj,
    lcmt_trajectory_block,
)
from round_belt_task.commander import (
    mat3_to_quat,
    pose_mat,
    saved_traj_message,
    x_tool0_tracking,
)
from round_belt_task.controller_bridge import ur_base_world
from task_common.bc_prior import encoder_fingerprint
from task_common.latent_encoder import (
    LATENT_STATE_CHANNEL,
    DemoGoals,
    parse_latent_state_message,
)
from task_common.lcm_contract import LcmChannels
from task_common.lcs_dataset import delta_rotvec
from task_common.magna_process import check_private_url
from task_common.mppi_lcs import (
    MppiParams,
    OvershootGuard,
    PolicyPrior,
    TorchLcs,
    clearance_for,
    franka_knots,
    goal_set_dist,
    load_goal_set,
    load_value,
    mppi_plan,
    qmul,
    quat_exp,
    support_limits,
)

METADATA_NAME = "mppi_controller"
N_KNOTS = 7
REPUBLISH_S = 0.05
HOLD_S = 1.0
FEATURES_BLOCK = "belt_features"
FEATURE_SUCCESS_FRAMES = 3
SCALARS = ("k0", "stage", "dist_ref", "dist_final", "solve_ms", "t_mpc_start", "ref_mode",
           "n_x", "N", "solve_ok")
CH = LcmChannels()


def _rigid_inv(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def _rename(msg: lcmt_timestamped_saved_traj) -> lcmt_timestamped_saved_traj:
    msg.saved_traj.metadata.name = METADATA_NAME
    return msg


def franka_message(utime: int, t: float, dt: float, pose7, U) -> lcmt_timestamped_saved_traj:
    pos, quat = franka_knots(pose7, U, N_KNOTS)
    return _rename(saved_traj_message(utime, pos, quat, t + dt * np.arange(N_KNOTS)))


def ur_line_world(pose7, u0) -> tuple[np.ndarray, np.ndarray]:
    """World tracking-frame line ends: the latent's UR pose and it + ``u0[3:6]`` /
    ``exp(u0[9:12])``; two 4x4."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    q1 = qmul(quat_exp(np.asarray(u0)[9:12]), pose7[3:7])
    return (pose_mat(pose7[:3], pose7[3:7]),
            pose_mat(pose7[:3] + np.asarray(u0)[3:6], q1 / np.linalg.norm(q1)))


def ur_message(utime: int, t: float, dt: float, pose7, u0, X_base_W: np.ndarray,
               X_tracking_tool0: np.ndarray) -> lcmt_timestamped_saved_traj:
    """Inverse of ``ControllerBridge._on_ur_traj``: world tracking -> tool0 -> UR base."""
    base = [X_base_W @ X @ X_tracking_tool0 for X in ur_line_world(pose7, u0)]
    pos = np.stack([m[:3, 3] for m in base])
    quat = np.stack([mat3_to_quat(m[:3, :3]) for m in base])
    return _rename(saved_traj_message(utime, pos, quat, np.array([t, t + dt])))


def _block(name: str, data: np.ndarray, times, datatypes=None):
    b = lcmt_trajectory_block()
    b.trajectory_name = name
    b.num_points = int(data.shape[1])
    b.num_datatypes = int(data.shape[0])
    b.time_vec = [float(v) for v in times]
    b.datapoints = [[float(v) for v in row] for row in data]
    b.datatypes = list(datatypes) if datatypes else ["double"] * int(data.shape[0])
    return b


def debug_message(utime: int, t: float, dt: float, Z, P, U, z_ref, p_now, scalars: dict
                  ) -> lcmt_timestamped_saved_traj:
    """``LEARNED_MPC_DEBUG`` in the C++ layout: x_sol (n_x+6, N+1), u_sol (12, N), z_ref
    (one goal repeated, or per-knot rows ``(>= N+1, n_x)``), p_ref (current positions),
    scalars (10, 1)."""
    n = N_KNOTS
    T = t + dt * np.arange(n + 1)
    x_sol = np.concatenate([np.asarray(Z)[: n + 1].T, np.asarray(P)[: n + 1].T])
    z_ref = np.asarray(z_ref)
    z_ref = (z_ref[: n + 1].T if z_ref.ndim == 2
             else np.repeat(z_ref[:, None], n + 1, axis=1))
    blocks = [
        _block("x_sol", x_sol, T),
        _block("u_sol", np.asarray(U)[:n].T, T[:n]),
        _block("z_ref", z_ref, T),
        _block("p_ref", np.repeat(np.asarray(p_now)[:, None], n + 1, axis=1), T),
        _block("scalars", np.array([[float(scalars[k])] for k in SCALARS]), T[:1], SCALARS),
    ]
    meta = lcmt_metadata()
    meta.git_dirty_flag = False
    meta.datetime = ""
    meta.name = METADATA_NAME
    meta.description = ""
    meta.git_commit_hash = ""
    traj = lcmt_saved_traj()
    traj.metadata = meta
    traj.trajectories = blocks
    traj.num_trajectories = len(blocks)
    traj.trajectory_names = [b.trajectory_name for b in blocks]
    msg = lcmt_timestamped_saved_traj()
    msg.utime = int(utime)
    msg.saved_traj = traj
    return msg


def belt_features(msg) -> np.ndarray | None:
    for b in msg.saved_traj.trajectories:
        if b.trajectory_name == FEATURES_BLOCK:
            return np.asarray(b.datapoints, dtype=np.float64)[:, 0]
    return None


@dataclasses.dataclass
class Stage:
    goal: np.ndarray
    tol: float
    max_duration_s: float
    index: int  # the goal's stage index in the deploy / demo_goals (the debug "stage" row)


def prior_sync_index(frames, s: int, offset: int, elapsed_s: float, dt: float) -> int:
    """Prior index in stage ``s`` (``frames[s]`` = its waypoint's demo frame): from
    ``max(frames[s-1], offset)`` (0 before stage 0) at one frame per ``dt``, clamped at
    ``frames[s]`` (never below the start)."""
    lo = max(int(offset), int(frames[s - 1]) if s > 0 else 0)
    return min(lo + round(elapsed_s / dt), max(lo, int(frames[s])))


def progress_start(d_now, d_start, tols, n: int) -> int:
    """Initial stage among stages ``0..n-1`` (before the latch): drop the start-state ones
    (``d_start[k] < tols[k]``: the goal demo's first latent is within their tol) whose tol does
    not hold the latent (``d_now[k] >= tols[k]``), then the nearest by ``d_now`` (ties: the
    earliest); ``n`` when none is left."""
    keep = [k for k in range(n) if not d_start[k] < tols[k] <= d_now[k]]
    return min(keep, key=lambda k: (d_now[k], k)) if keep else n


def load_waypoints(cli) -> Waypoints | None:
    if not cli.goal.startswith("waypoints:"):
        return None
    return Waypoints.load(cli.goal.split(":", 1)[1])


def build_stages(cli, params: MppiParams, lcs: TorchLcs, wp: Waypoints | None = None
                 ) -> list[Stage]:
    if cli.goal == "deploy":
        goals = lcs.stage_goals
    elif cli.goal.startswith("demo_goals:"):
        goals = DemoGoals.load(cli.goal.split(":", 1)[1]).z_goals
    elif wp is not None:
        goals = wp.z_wp
    else:
        raise ValueError(f"--goal {cli.goal!r}: expected deploy, demo_goals:PATH or "
                         "waypoints:PATH")
    if goals.shape[1] != lcs.n_x:
        raise ValueError(f"goal dim {goals.shape[1]} != n_x {lcs.n_x}")
    if wp is not None and (cli.stages == "all" or cli.goal_mode == "track"):
        last = min(float(wp.max_duration_s[-1]), cli.max_duration_s)
        if cli.goal_mode == "track":  # one stage: the seated goal, row 0 in the debug channel
            return [Stage(goals[-1].copy(), cli.goal_tol, last, 0)]
        return [Stage(g.copy(), float(wp.tol[k]) if k + 1 < len(goals) else cli.goal_tol,
                      float(wp.max_duration_s[k]) if k + 1 < len(goals) else last, k)
                for k, g in enumerate(goals)]
    if cli.stages == "last":
        return [Stage(goals[-1].copy(), cli.goal_tol, cli.max_duration_s, len(goals) - 1)]

    def pick(values, k, default):
        if values is None or k >= len(values) or values[k] is None or values[k] < 0:
            return default
        return float(values[k])

    return [Stage(g.copy(), pick(params.stage_goal_tols, k, cli.goal_tol),
                  pick(params.stage_max_durations_s, k, cli.max_duration_s), k)
            for k, g in enumerate(goals)]


class MppiControllerNode:
    def __init__(self, cli: argparse.Namespace, lc: lcm.LCM | None = None) -> None:
        self.cli = cli
        self.params = MppiParams.from_yaml(cli.config) if cli.config else MppiParams()
        if cli.clearance_model is not None:
            self.params = dataclasses.replace(self.params, clearance={
                **(self.params.clearance or {}), "model": str(cli.clearance_model.resolve())})
        if self.params.H < N_KNOTS - 1:
            raise ValueError(f"H {self.params.H} < {N_KNOTS - 1} (the 7-knot Franka plan)")
        self.lcs = TorchLcs.from_deploy(cli.deploy, cli.device)
        self.dt = self.lcs.dt
        self.wp = load_waypoints(cli)
        if cli.goal_mode == "track" and self.wp is None:
            raise ValueError("--goal-mode track needs --goal waypoints:PATH")
        if self.wp is not None and abs(self.wp.dt - self.dt) > 1e-9:
            raise ValueError(f"waypoints dt {self.wp.dt} != deploy dt {self.dt}")
        self.u_demo = self._demo_actions(cli)
        self.bc = (PolicyPrior(self.params.bc_prior, self.lcs, self.params, cli.deploy)
                   if self.params.prior_mode == "policy" else None)
        self.stages = build_stages(cli, self.params, self.lcs, self.wp)
        if self.params.goal_set is not None and cli.goal_mode != "stage":
            raise ValueError("goal_set needs --goal-mode stage (it replaces the last stage's goal)")
        self.gset = (None if self.params.goal_set is None
                     else load_goal_set(self.params.goal_set, self.lcs.n_x))
        self.vfn = (None if self.params.value is None
                    else load_value(self.params.value, self.lcs.n_x, self.lcs.z_std_np))
        if self.vfn is not None and self.vfn.encoder_sha256 \
                and self.vfn.encoder_sha256 != encoder_fingerprint(cli.deploy):
            raise ValueError(f"value {self.params.value['path']} was trained on another encoder")
        if self.params.prior_sync == "stage" and (self.wp is None or cli.goal_mode != "stage"):
            raise ValueError("prior_sync stage needs --goal waypoints:PATH and --goal-mode stage")
        self.demo_offset = cli.demo_offset_frames
        self.bank, self.bank_info = self._load_bank(cli)
        self.lb, self.ub = self.params.bounds(self.lcs)
        self.sigma = self.params.sigma_frac * (self.ub - self.lb) / 2.0
        self.gen = torch.Generator(device=self.lcs.device).manual_seed(int(self.params.seed))
        self.rng = np.random.default_rng(int(self.params.seed))
        self.X_base_W = _rigid_inv(ur_base_world())
        self.X_tracking_tool0 = _rigid_inv(x_tool0_tracking())
        self.lc = lc or lcm.LCM(cli.lcm_url)
        self.log = open(cli.log, "a") if cli.log else None  # noqa: SIM115
        self.stage = 0
        self.stage_t0: float | None = None
        self.t_mpc_start: float | None = None
        self.start_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.last_utime: int | None = None
        self.plan: np.ndarray | None = None
        self.out: list[tuple[str, bytes]] = []
        self.feature_streak = 0
        self.reach_streak = 0
        self.features_warned = False
        self.done = False
        self.last_latent = None
        self.pending = None
        self.n_solves = self.n_published = self.n_errors = self.n_skipped = 0
        self.solve_ms: list[float] = []
        self._last_pub = time.perf_counter()
        self.clear_reported = False
        self.held_s = 0.0
        self.prev_t: float | None = None
        self.idx: int | None = None
        ov = self.params.overshoot
        self.guard = OvershootGuard(ov) if ov is not None else None
        self.gate_stage = self._gate_stage()
        self.blocked = False
        self.start_info: dict | None = None
        if self.params.stage_start is not None and (self.wp is None
                                                    or cli.goal_mode != "stage"):
            raise ValueError("stage_start progress needs --goal waypoints:PATH --goal-mode stage")
        self._warmup()

    def _load_bank(self, cli) -> tuple[WaypointBank | None, dict | None]:
        if getattr(cli, "waypoint_bank", None) is None:
            return None, None
        if self.wp is None:
            raise ValueError("--waypoint-bank needs --goal waypoints:PATH (the base)")
        bank = WaypointBank(cli.waypoint_bank)
        for w in bank.entries:
            if abs(w.dt - self.dt) > 1e-9 or w.z_traj.shape[1] != self.lcs.n_x:
                raise ValueError(f"bank entry {w.demo_episode}: dt / n_x mismatch")
        self.bank_base = bank.base_marks(self.wp) if cli.demo_offset_frames > 0 else None
        return bank, {}

    def _select_demo(self, z) -> None:
        """Bank: reference demo = nearest initial latent; rebuilds the stages and prior."""
        wp, off, info = self.bank.select(z, self.lcs.z_std_np, self.cli.demo_offset_frames,
                                         self.bank_base, self.cli.bank_knn)
        self.wp, self.demo_offset = wp, off
        if self.cli.demo_actions == "auto":
            self.u_demo = wp.u_traj
        self.stages = build_stages(self.cli, self.params, self.lcs, wp)
        self.gate_stage = self._gate_stage()
        self.bank_info = {**info, "n_stages": len(self.stages)}
        s = info["second"]
        print(f"[mppi] bank: entry {info['entry']} {info['demo']} dist {info['dist']:.3f} "
              f"(key frame {info['key_frame']}, offset {off}"
              + (f"; next {s['entry']} {s['dist']:.3f}" if s else "")
              + (f"; knn {[k['entry'] for k in info['knn']]}" if "knn" in info else "")
              + f"), {len(self.stages)} stage(s)", flush=True)

    def _latch_stage(self) -> int | None:
        """Node stage index of the latch waypoint (stage mode); None without one."""
        if self.wp is None or self.cli.goal_mode != "stage":
            return None
        k = [s for s, st in enumerate(self.stages)
             if int(self.wp.frames[st.index]) == self.wp.latch]
        return k[0] if k else None

    def _start_stage(self, t: float, z) -> None:
        """``stage_start: progress`` at the first latent (after a bank pick)."""
        n = self._latch_stage() or 0  # no latch stage (--stages last): no-op
        if self.gate_stage is not None:  # a gated stage is still entered only by reaching
            n = min(n, self.gate_stage - 1)
        st = self.stages[:n]
        d_now = [self.lcs.whitened_dist(z, s.goal) for s in st]
        d_start = [self.lcs.whitened_dist(self.wp.z_traj[0], s.goal) for s in st]
        k = progress_start(d_now, d_start, [s.tol for s in st], n)
        dropped = [j for j in range(n) if d_start[j] < st[j].tol <= d_now[j]]
        self.start_info = {"stage": k, "dropped": dropped, "d_now": d_now, "d_start": d_start}
        print(f"[mppi] start stage {k} (frame {int(self.wp.frames[self.stages[k].index])}) "
              f"t={t:.3f} dist={self.lcs.whitened_dist(z, self.stages[k].goal):.4f}; dropped "
              f"start-state stages {dropped}", flush=True)
        self.stage = k

    def _gate_stage(self) -> int | None:
        """Node stage index of ``stage_no_timeout_into`` ("latch": after the latch waypoint);
        None with one stage (no entry to gate)."""
        nt = self.params.stage_no_timeout_into
        if nt is None or len(self.stages) < 2:
            return None
        if nt == "latch":
            k = [s for s, st in enumerate(self.stages) if self.wp is not None
                 and int(self.wp.frames[st.index]) == self.wp.latch]
            if not k:
                raise ValueError("stage_no_timeout_into latch needs --goal waypoints:PATH "
                                 "--stages all (a latch waypoint)")
            nt = k[0] + 1
        if not 1 <= nt < len(self.stages):
            raise ValueError(f"stage_no_timeout_into {nt} not in 1..{len(self.stages) - 1}")
        return int(nt)

    def _demo_actions(self, cli) -> np.ndarray | None:
        p = self.params
        if cli.demo_actions is None:
            if (p.prior_alpha > 0.0 or p.prior_weight > 0.0) and p.prior_mode == "demo":
                raise ValueError("prior_alpha / prior_weight need --demo-actions auto|PATH")
            return None
        if cli.demo_actions == "auto":
            if self.wp is None:
                raise ValueError("--demo-actions auto needs --goal waypoints:PATH")
            u = self.wp.u_traj
        else:
            u = load_actions(cli.demo_actions)
        if u.ndim != 2 or u.shape[1] != self.lcs.n_u:
            raise ValueError(f"demo actions shape {u.shape}: expected (T, {self.lcs.n_u})")
        return u

    def _warmup(self) -> None:
        z = self.stages[-1].goal
        kw = {"q_now": np.tile([1.0, 0.0, 0.0, 0.0], (2, 1))} if self.params.clearance else {}
        mppi_plan(self.lcs, self.params, z, np.zeros(6), None, z, None,
                  torch.Generator(device=self.lcs.device).manual_seed(0), **kw)
        prms = self.guard.variants(self.params) if self.guard else []
        if self.gset is not None:  # earlier stages solve without the set
            prms += [dataclasses.replace(q, goal_set=None) for q in [self.params, *prms]]
        for prm in prms:
            mppi_plan(self.lcs, prm, z, np.zeros(6), None, z, None,
                      torch.Generator(device=self.lcs.device).manual_seed(0), **kw)

    def start(self) -> None:
        self.lc.subscribe(LATENT_STATE_CHANNEL, self._on_latent)
        print("Assembly controller started", flush=True)
        print("No pre-MPC targets found. Initial phase set to MPC.", flush=True)

    # --- per latent ----------------------------------------------------------------------------

    def _on_latent(self, _ch: str, data: bytes) -> None:
        try:
            msg = lcmt_timestamped_saved_traj.decode(data)
            parsed = parse_latent_state_message(msg)
        except (ValueError, KeyError) as exc:
            self.n_errors += 1
            print(f"[mppi] latent dropped: {exc}", flush=True)
            return
        newest = self.pending[1][0] if self.pending else self.last_utime
        if newest is None or parsed[0] > newest:
            self.n_skipped += self.pending is not None
            self.pending = (msg, parsed)

    def _process(self, msg, parsed) -> None:
        """Replan on the newest latent only (the C++ reads the latest value on its port)."""
        utime, _, z, ee_f, ee_u, _prop = parsed
        self.last_utime = utime
        t = utime * 1e-6
        self.last_latent = (utime, t, ee_f, ee_u)
        if self.t_mpc_start is None:
            self.t_mpc_start = self.stage_t0 = t
            self.start_pose = (np.r_[ee_f[:3], ee_u[:3]], np.stack([ee_f[3:7], ee_u[3:7]]))
            if self.bank is not None:
                self._select_demo(z)
            if self.params.stage_start is not None:
                self._start_stage(t, z)
        if self.params.stage_skip_ahead:
            self._skip_ahead(t, z)
        stage = self.stages[self.stage]
        dist = self._stage_dist(z, self.stage)
        if self._stage_over(t, dist, msg):  # this latent is consumed without a solve
            return
        self._solve(utime, t, z, ee_f, ee_u, stage, dist)

    def _skip_ahead(self, t: float, z) -> None:
        """To the furthest later stage (tol >= 0) whose tol holds ``z``; below the gated stage
        only up to the one before it (the gate is passed by reaching, A3)."""
        hi = len(self.stages) - 1
        if self.gate_stage is not None and self.stage < self.gate_stage:
            hi = self.gate_stage - 1
        best = None
        for j in range(self.stage + 1, hi + 1):
            st = self.stages[j]
            d = self._stage_dist(z, j)
            if 0.0 <= d < st.tol:
                best = (j, d)
        if best is None:
            return
        print(f"[mppi] skip ahead: stage {self.stage} -> {best[0]} t={t:.3f} dist={best[1]:.4f}",
              flush=True)
        self.stage, self.stage_t0 = best[0], t
        self.feature_streak = self.reach_streak = 0
        self.blocked = False

    def _stage_dist(self, z, j: int) -> float:
        """Whitened distance to stage ``j``'s goal; the goal set's for the last stage."""
        if self.gset is not None and j == len(self.stages) - 1:
            return goal_set_dist(z, self.gset, self.lcs.z_std_np, self.params.goal_set_tau)
        return self.lcs.whitened_dist(z, self.stages[j].goal)

    def _stage_over(self, t: float, dist: float, msg) -> bool:
        stage = self.stages[self.stage]
        reached = 0.0 <= dist < stage.tol
        k = self.params.stage_reach_streak
        if k > 1 and self.stage + 1 < len(self.stages):
            self.reach_streak = self.reach_streak + 1 if reached else 0
            reached = self.reach_streak >= k
        if self.cli.complete_on_features:
            feats = belt_features(msg)
            if feats is None:
                if not self.features_warned:
                    print(f"[mppi] no {FEATURES_BLOCK} block: feature completion off", flush=True)
                    self.features_warned = True
            else:
                self.feature_streak = self.feature_streak + 1 if feats[4] > 0.5 else 0
                reached = reached or self.feature_streak >= FEATURE_SUCCESS_FRAMES
        timeout = t - self.stage_t0 >= stage.max_duration_s
        if not (reached or timeout):
            return False
        if not reached and self.stage + 1 == self.gate_stage:  # timeout cannot enter the gate
            if not self.blocked:
                self.blocked = True
                print(f"[mppi] stage {self.stage} timeout blocked (stage {self.gate_stage} only "
                      f"on reach) t={t:.3f} dist={dist:.4f}", flush=True)
            return False
        print(f"[learned-mpc] stage {self.stage} {'reached' if reached else 'timeout'} "
              f"t={t:.3f} dist={dist:.4f}", flush=True)
        if self.stage + 1 < len(self.stages):
            self.stage += 1
            self.stage_t0 = t
            self.feature_streak = self.reach_streak = 0
            self.blocked = False
            return True
        print("MPC completed!", flush=True)
        print("Switching to Terminate", flush=True)
        self.done = True
        return True

    def _rot0(self, ee_f, ee_u) -> np.ndarray:
        """``(2, 3)`` world rotation vectors of the current orientations from the start."""
        q0 = self.start_pose[1]
        return np.stack([delta_rotvec(q0[0], ee_f[3:7]), delta_rotvec(q0[1], ee_u[3:7])])

    def _q_now(self, rot0) -> np.ndarray:
        """``(2, 4)`` absolute ``wxyz``: ``exp(rot0) q_start`` per arm."""
        return np.stack([qmul(quat_exp(rot0[a]), self.start_pose[1][a]) for a in range(2)])

    def _params_now(self) -> MppiParams:
        p = self.params
        if not math.isfinite(p.z_ceiling_above_start_m):
            return p
        p0 = self.start_pose[0]
        return dataclasses.replace(
            p, z_max_franka=min(p.z_max_franka, p0[2] + p.z_ceiling_above_start_m),
            z_max_ur=min(p.z_max_ur, p0[5] + p.z_ceiling_above_start_m))

    def _progress(self, t: float, z) -> tuple[int, float | None]:
        """Demo progress index: frames since ``t_mpc_start`` + ``--demo-offset-frames``; in
        ``track`` mode held while the state is far from the current demo latent, for at most
        ``--track-max-hold-s`` in total (then time-indexed again); ``prior_sync`` stage:
        ``prior_sync_index`` on the current stage. ``(idx, d_idx)``."""
        cli = self.cli
        T = self.wp.T if self.wp is not None else len(self.u_demo)
        step = 0.0 if self.prev_t is None else max(0.0, t - self.prev_t)
        self.prev_t = t

        def at() -> int:
            if self.params.prior_sync == "stage":
                k = prior_sync_index([self.wp.frames[st.index] for st in self.stages],
                                     self.stage, self.demo_offset, t - self.stage_t0, self.dt)
                return int(min(max(k, 0), T - 1))
            k = self.demo_offset + round((t - self.t_mpc_start - self.held_s) / self.dt)
            return int(min(max(k, 0), T - 1))

        d_idx = None
        if cli.goal_mode == "track":
            if self.idx is not None and at() > self.idx:
                d_prev = self.lcs.whitened_dist(z, self.wp.z_traj[self.idx])
                if (d_prev > cli.track_hold_factor * self.wp.tol_at(self.idx)
                        and self.held_s < cli.track_max_hold_s):
                    self.held_s = min(self.held_s + step, cli.track_max_hold_s)
            self.idx = at()
            d_idx = self.lcs.whitened_dist(z, self.wp.z_traj[self.idx])
        else:
            self.idx = at()
        return self.idx, d_idx

    def _solve(self, utime: int, t: float, z, ee_f, ee_u, stage: Stage, dist: float) -> None:
        p0 = np.r_[ee_f[:3], ee_u[:3]]
        rot0 = self._rot0(ee_f, ee_u)
        params = self._params_now()
        solve_ok = 1.0
        kw = {"q_now": self._q_now(rot0)} if params.clearance else {}
        goal, z_ref, dist_ref, extra = stage.goal, stage.goal, dist, {}
        if self.cli.goal_mode == "track" or self.u_demo is not None:
            idx, d_idx = self._progress(t, z)
            extra["idx"] = idx
            if self.u_demo is not None:
                u = np.zeros((params.H, self.lcs.n_u))
                rows = self.u_demo[idx:idx + params.H]
                u[:len(rows)] = rows
                if self.gate_stage is not None and self.wp is not None \
                        and self.stage < self.gate_stage:  # no prior past the gate's waypoint
                    u[max(0, int(self.wp.frames[self.stages[self.gate_stage - 1].index])
                          - idx):] = 0.0
                kw["u_demo"] = u
            if self.cli.goal_mode == "track":
                goal = self.wp.goal_seq(idx, params.H)
                z_ref = np.vstack([self.wp.z_traj[idx], goal])
                dist_ref = d_idx
                extra.update(held_s=self.held_s, d_idx=d_idx)
        if self.bc is not None:
            tp = time.perf_counter()
            kw["u_demo"] = self.bc(z, p0)
            extra.update(prior_u0=kw["u_demo"][0].tolist(),
                         prior_ms=(time.perf_counter() - tp) * 1e3)
        u_prev = self.plan
        if self.guard is not None:
            d_seat = self.lcs.whitened_dist(z, self.stages[-1].goal)
            gs = self.guard.update(z, d_seat, self.stage == len(self.stages) - 1)
            params = self.guard.params_for(params, gs.mode)
            if "u_demo" in kw:
                kw["u_demo"] = kw["u_demo"] * gs.prior_scale
            if gs.mode == "hold":
                kw.pop("u_demo", None)
                goal = self.stages[-1].goal if gs.goal is None else gs.goal
                z_ref = goal
            if gs.cold:
                u_prev = None
            if gs.entered:
                print(f"[mppi] overshoot hold: t={t:.3f} d_seat={d_seat:.4f} (tol "
                      f"{self.guard.tol:.3f})", flush=True)
            extra.update(d_seat=d_seat, prior_scale=gs.prior_scale, mode=gs.mode)
        if self.gset is not None:
            if self.stage == len(self.stages) - 1 and goal is stage.goal:
                extra["d_single"] = self.lcs.whitened_dist(z, stage.goal)
            else:  # earlier stages / a hold goal of the guard: no set
                params = dataclasses.replace(params, goal_set=None)
        try:
            U, info = mppi_plan(self.lcs, params, z, p0, rot0, goal, u_prev, self.gen,
                                **kw)
            if not np.all(np.isfinite(U)):
                raise FloatingPointError("non-finite plan")
        except (RuntimeError, FloatingPointError, ValueError) as exc:
            print(f"[mppi] solve failed ({exc}); holding", flush=True)
            U = np.zeros((params.H, self.lcs.n_u))
            info = {"solve_ms": math.nan, "fallback": False}
            solve_ok = 0.0
        if params.clearance and solve_ok and not self.clear_reported:
            self.clear_reported = True
            print(f"[mppi] clearance at the first latent: UR {info['clear_now_ur_mm']:.1f} mm, "
                  f"Franka {info['clear_now_f_mm']:.1f} mm", flush=True)
        U_exec = U.copy()
        if params.action_noise_frac > 0.0 and solve_ok:
            noise = self.rng.normal(size=U.shape[1]) * params.action_noise_frac * self.sigma
            U_exec[0] = np.clip(U[0] + noise, self.lb, self.ub)
        self.plan = U_exec
        self.n_solves += 1
        self.solve_ms.append(info["solve_ms"])
        self.out = [(CH.tracking_trajectory_actor_channel,
                     franka_message(utime, t, self.dt, ee_f, U_exec).encode()),
                    (CH.ur_tracking_trajectory_actor_channel,
                     ur_message(utime, t, self.dt, ee_u, U_exec[0], self.X_base_W,
                                self.X_tracking_tool0).encode())]
        self._publish()
        if self.cli.debug_channel:
            roll = (self.lcs.rollout_fused if self.lcs.device.type == "cuda"
                    and params.fused_rollout else self.lcs.rollout)
            Zd, Pd, _ = roll(z, p0, self.lcs.tensor(U_exec[:N_KNOTS])[None], rot0,
                             params.pgd_iters)
            scal = {"k0": -1.0, "stage": float(stage.index), "dist_ref": dist_ref,
                    "dist_final": self.lcs.whitened_dist(z, self.stages[-1].goal),
                    "solve_ms": info["solve_ms"], "t_mpc_start": self.t_mpc_start,
                    "ref_mode": -1.0, "n_x": float(self.lcs.n_x + 6), "N": float(N_KNOTS),
                    "solve_ok": solve_ok}
            dbg = debug_message(utime, t, self.dt, Zd[0].double().cpu().numpy(),
                                Pd[0].double().cpu().numpy(), U_exec, z_ref, p0, scal)
            self.lc.publish(self.cli.debug_channel, dbg.encode())
        if self.log:
            row = {"utime": utime, "t": t, "stage": self.stage, "goal_index": stage.index,
                   "dist": dist,
                   "u0": U_exec[0].tolist(), "p0": p0.tolist(), "rot0": rot0.tolist(),
                   "z_max": [params.z_max_franka, params.z_max_ur], "solve_ok": solve_ok,
                   **{k: v for k, v in info.items() if k not in ("Z", "P", "C")}}
            if self.params.stage_reach_streak > 1:
                row["reach_streak"] = self.reach_streak
            if self.gate_stage is not None:
                row["blocked"] = self.blocked
            row.update(extra)
            if self.bank_info and self.n_solves == 1:
                row["bank"] = self.bank_info
            if self.start_info is not None and self.n_solves == 1:
                row["start"] = self.start_info
            if params.support_box is not None:
                row["sbox_mm"] = self._sbox_mm(p0, info.get("P"))
            if self.vfn is not None:
                row["v_now"] = float(self.vfn(z))
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()

    def _sbox_mm(self, p0, P) -> list[float]:
        """[max mm outside the support box now, over the plan's knots 1..H] (nan: no plan)."""
        lo, hi = support_limits(self.params.support_box)

        def out(x):
            x = np.asarray(x, dtype=np.float64)
            return float(np.max(np.maximum(lo - x, 0.0) + np.maximum(x - hi, 0.0)) * 1e3)

        return [out(p0), math.nan if P is None else out(np.asarray(P)[1:])]

    def _publish(self) -> None:
        for channel, data in self.out:
            self.lc.publish(channel, data)
        self.n_published += 1
        self._last_pub = time.perf_counter()

    def hold(self) -> None:
        """Zero-input plan at the last latent's poses, republished for ``HOLD_S``."""
        if self.last_latent is not None:
            utime, t, ee_f, ee_u = self.last_latent
            U = np.zeros((N_KNOTS, self.lcs.n_u))
            self.out = [(CH.tracking_trajectory_actor_channel,
                         franka_message(utime, t, self.dt, ee_f, U).encode()),
                        (CH.ur_tracking_trajectory_actor_channel,
                         ur_message(utime, t, self.dt, ee_u, U[0], self.X_base_W,
                                    self.X_tracking_tool0).encode())]
        end = time.perf_counter() + HOLD_S
        while self.out and time.perf_counter() < end:
            self._publish()
            time.sleep(REPUBLISH_S)

    def spin(self, stop) -> None:
        self._last_pub = time.perf_counter()
        while not stop() and not self.done:
            wait = max(0.0, self._last_pub + REPUBLISH_S - time.perf_counter())
            try:
                n = self.lc.handle_timeout(max(1, int(wait * 1e3)))
                while n > 0:
                    n = self.lc.handle_timeout(0)
            except OSError:
                if not stop():
                    raise
            if self.pending is not None:
                msg, parsed = self.pending
                self.pending = None
                self._process(msg, parsed)
            if self.out and not self.done and time.perf_counter() - self._last_pub >= REPUBLISH_S:
                self._publish()

    def summary(self) -> str:
        ms = np.asarray([v for v in self.solve_ms if math.isfinite(v)])
        timing = (f"solve mean {ms.mean():.1f} p95 {np.percentile(ms, 95):.1f} max "
                  f"{ms.max():.1f} ms" if len(ms) else "no solves")
        return (f"[mppi] {self.n_solves} solves, {self.n_published} publishes, "
                f"{self.n_skipped} stale latents skipped, {self.n_errors} errors, {timing}")


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lcm-url", required=True, help="private LCM URL")
    p.add_argument("--deploy", type=Path, required=True, help="deploy.npz (fixed-B LCS)")
    p.add_argument("--config", type=Path, default=None, help="mppi.yaml (MppiParams fields)")
    p.add_argument("--goal", default="deploy",
                   help="deploy | demo_goals:PATH | waypoints:PATH (latent_waypoints.py)")
    p.add_argument("--goal-mode", choices=("stage", "track"), default="stage",
                   help="track (waypoints goal): per-knot goals = the demo's per-frame latents "
                        "from the progress index; one stage (the seated goal)")
    p.add_argument("--demo-offset-frames", type=int, default=0,
                   help="progress index at the first latent (e.g. a replayed prefix's end)")
    p.add_argument("--track-hold-factor", type=float, default=2.0,
                   help="track: hold the index while dist(z, z_traj[idx]) > this x tol")
    p.add_argument("--track-max-hold-s", type=float, default=3.0,
                   help="track: total hold budget; then the index advances with time")
    p.add_argument("--demo-actions", default=None, metavar="auto|PATH",
                   help="demo action prior inputs (MppiParams prior_*): auto = the waypoints "
                        "artifact's u_traj, PATH = an npz with u_traj or actions")
    p.add_argument("--waypoint-bank", type=Path, default=None, metavar="BANK_JSON",
                   help="latent_waypoints.py --bank-out: per episode, the entry with the "
                        "nearest initial latent replaces the --goal waypoints (+ auto prior)")
    p.add_argument("--bank-knn", type=int, default=1,
                   help="bank: mean of the k nearest entries (1 = nearest only)")
    p.add_argument("--stages", choices=("last", "all"), default="last")
    p.add_argument("--max-duration-s", type=float, default=30.0)
    p.add_argument("--goal-tol", type=float, default=-1.0,
                   help="whitened; < 0 never completes on latent distance")
    p.add_argument("--complete-on-features", action="store_true",
                   help=f"complete after {FEATURE_SUCCESS_FRAMES} consecutive cloud-success "
                        f"frames (needs the {FEATURES_BLOCK} block)")
    p.add_argument("--clearance-model", type=Path, default=None,
                   help="clearance_model npz: MppiParams.clearance on (thresholds from the "
                        "config's clearance block, else the npz)")
    p.add_argument("--debug-channel", default=None, help="e.g. LEARNED_MPC_DEBUG (off if unset)")
    p.add_argument("--log", type=Path, default=None, help="per-solve JSONL")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> int:
    cli = create_parser().parse_args()
    check_private_url(cli.lcm_url)
    node = MppiControllerNode(cli)
    stopping: list[bool] = []
    signal.signal(signal.SIGINT, lambda *_: stopping.append(True))
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    p = node.params
    print(f"[mppi] {cli.lcm_url}: deploy {cli.deploy}, {len(node.stages)} stage(s), K {p.K} "
          f"H {p.H} iters {p.n_iter}, u box {np.round(node.lb, 5).tolist()} .. "
          f"{np.round(node.ub, 5).tolist()}, device {cli.device}", flush=True)
    if node.wp is not None or node.u_demo is not None or node.bc is not None:
        w = node.wp
        print(f"[mppi] goal {cli.goal.split(':', 1)[0]} mode {cli.goal_mode}"
              + (f", {len(w.frames)} waypoints at frames {w.frames.tolist()} tols "
                 f"{np.round(w.tol, 3).tolist()}" if w is not None else "")
              + f", demo offset {cli.demo_offset_frames} frames, prior alpha {p.prior_alpha:g} "
              f"weight {p.prior_weight:g} "
              f"({'on' if node.u_demo is not None or node.bc is not None else 'off'})"
              + (f", sync {p.prior_sync}" if p.prior_sync else "")
              + (f", policy prior {p.bc_prior}" if p.prior_mode == "policy" else "")
              + (", skip ahead" if p.stage_skip_ahead else "")
              + (f", start {p.stage_start}" if p.stage_start else "")
              + (f", no timeout into stage {node.gate_stage}" if node.gate_stage else "")
              + (f", bank {cli.waypoint_bank} ({len(node.bank.entries)} entries, knn "
                 f"{cli.bank_knn})" if node.bank is not None else ""), flush=True)
    if p.support_box:
        sb = p.support_box
        print(f"[mppi] support box (mm) lo {np.round(np.asarray(sb['lo']) * 1e3, 1).tolist()} hi "
              f"{np.round(np.asarray(sb['hi']) * 1e3, 1).tolist()}, weight {sb['weight']:g}, "
              f"hard margin {sb.get('hard_margin_m')}", flush=True)
    if node.gset is not None:
        print(f"[mppi] goal set {p.goal_set}: {len(node.gset)} seated latents, tau "
              f"{p.goal_set_tau:g} (last stage)", flush=True)
    if node.vfn is not None:
        print(f"[mppi] value {p.value['path']}: weight {float(p.value['weight']):g}, "
              f"{p.value.get('where', 'terminal')}, V_max {node.vfn.v_max:g} s", flush=True)
    if p.clearance:
        c = clearance_for(node.lcs, p)
        print(f"[mppi] clearance {p.clearance['model']}: UR {len(c.points[0])} of "
              f"{len(c.model.ur_local)} points, Franka {len(c.points[1])} of "
              f"{len(c.model.franka_local)} ({c.model.source.get('franka')}); min UR "
              f"{c.ur_min * 1e3:g} / Franka {c.franka_min * 1e3:g} mm", flush=True)
    node.start()
    node.spin(lambda: bool(stopping))
    if node.done:
        node.hold()
    print(node.summary(), flush=True)
    if node.log:
        node.log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
