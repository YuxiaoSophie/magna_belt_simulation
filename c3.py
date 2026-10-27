"""
Round-belt FULL scene (UR10 + Robotiq 2F-85 + deformable belt + pulleys +
board + table) where the gripper TCP is driven by a C3 (Consensus
Complementarity Control) MPC that only sees the recorded end-effector waypoints.

The scene, solvers, gripper anti-crush latch, IK and hybrid MuJoCo + VBD physics
all come from belt/round_belt.py (used as the baseline Example class); only the
target source (SpaceMouse -> C3 on recorded waypoints) and the belt error report
are new here.

Pipeline per frame:
    recorded waypoint (TCP pose + grip command) of frame i
        -> C3 MPC on a task-space LCS of the TCP (contact with table / board)
        -> planned next TCP position -> IK -> UR10 joint targets
        -> same hybrid MuJoCo(robot) + VBD(belt) coupled physics
    the resulting belt state is then compared against the belt state stored in
    the recorded episode (ground truth) and the error is printed.

Run (from the repo root, like round_belt.py):
    python belt/round_belt_c3_tracking.py --data-dir data/simple

Only the end-effector waypoints are used as control input; the recorded
joint_target_q and belt states are NEVER fed to the controller (the belt states
are only read for the error report and the orange reference belt in the viewer).
"""

from __future__ import annotations

import sys
import math
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples

# Make `from belt import round_belt` work when launched as `python belt/<this file>`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from belt import round_belt
from belt.round_belt import (
    BOARD_ROOT_X, BOARD_ROOT_Y, BOARD_ROOT_Z, BOARD_SIZE,
    TABLE_LENGTH_X, TABLE_WIDTH_Y, TABLE_TOP_Z, TABLE_BOTTOM_Z,
    IK_TRACK_ITERS, TELEOP_MAX_TARGET_DISTANCE,
    _norm4, _rotate_vec, _v3, _v4,
)

# Waypoint-following safety: same 3 cm IK-target lead clamp as the teleop pipeline.
TRACK_MAX_TARGET_DISTANCE = TELEOP_MAX_TARGET_DISTANCE # m, max IK target lead from actual TCP

# Recorded episode (waypoints + ground-truth belt states).
DATA_DIR_DEFAULT = "data/simple"

# C3 (Consensus Complementarity Control) MPC settings.
C3_HORIZON = 10 # MPC steps (one step = one control frame)
C3_Q_POS = 1.0e4 # TCP position tracking weight
C3_Q_VEL = 1.0e1 # TCP velocity tracking weight
C3_Q_TERMINAL_SCALE = 5.0 # terminal cost = scale * running cost
C3_R_ACC = 1.0e-2 # commanded acceleration weight
C3_FRICTION_MU = 0.5 # TCP <-> table/board friction in the LCS model
C3_FRICTION_DIRS = 4 # friction pyramid edges
C3_ADMM_ITERATIONS = 10
C3_RHO_INITIAL = 1.0
C3_RHO_SCALE = 1.6 # rho grows every ADMM iteration (standard C3 schedule)
C3_G_LAMBDA = 1.0 # ADMM consensus weight on contact impulses
C3_G_ETA = 1.0 # ADMM consensus weight on complementarity slacks
C3_COMMAND_STEP = 1 # which planned state is sent to IK as the TCP target
# The TCP point (0.145 m past the Robotiq base) sits slightly below the physical
# finger tips, so the recorded TCP can legally go ~5 mm under the table top.
C3_SURFACE_OFFSET = -0.006 # m, contact plane for the TCP = surface_z + offset


# Recorded episode loading (waypoints + ground-truth Newton states)
def _decode_recorded_array(entry, cache: dict) -> np.ndarray:
    """Decode one array written by ViewerFile (CBOR) into numpy, without warp."""
    data = entry.get("data", entry) if entry.get("__type__", "").startswith("warp.array") else entry
    kind = str(data.get("__type__", ""))
    if kind.endswith("_ref") or str(entry.get("__type__", "")).endswith("_ref"):
        idx = int(data.get("cache_index", entry.get("cache_index")))
        return cache[idx]
    dtype = np.dtype(data["dtype"])
    shape = tuple(int(s) for s in data["shape"])
    if "binary_data" in data:
        arr = np.frombuffer(data["binary_data"], dtype=dtype).reshape(shape, order=data.get("order", "C"))
    else:
        arr = np.asarray(data["data"], dtype=dtype).reshape(shape)
    for idx in (data.get("cache_index"), entry.get("cache_index")):
        if idx is not None:
            cache[int(idx)] = arr
    return arr


def load_recorded_states(state_file: Path, names=("body_q", "body_qd")) -> dict[str, Any]:
    """Read newton_states.bin -> {name: (n_states, ...) array, 'body_label': [...]}."""
    try:
        import cbor2
    except ImportError as e:
        raise ImportError("cbor2 is required to read newton_states.bin (pip install cbor2)") from e

    with Path(state_file).open("rb") as f:
        raw = cbor2.load(f)
    root = raw["items"] if isinstance(raw, dict) and "items" in raw else raw
    states = root["states"]
    states = states["items"] if isinstance(states, dict) and "items" in states else states

    cache: dict[int, np.ndarray] = {}
    out: dict[str, Any] = {}
    for name in names:
        frames = []
        for s in states:
            items = s["items"] if "items" in s else s
            if name not in items:
                break
            frames.append(np.array(_decode_recorded_array(items[name], cache), dtype=np.float64))
        if len(frames) == len(states):
            out[name] = np.stack(frames, axis=0)

    labels = []
    try:
        for it in root["model"]["attributes"]["body_label"]["items"]:
            labels.append(str(it.get("value", "")))
    except (KeyError, TypeError):
        pass
    out["body_label"] = labels
    out["n_states"] = len(states)
    return out


def load_recorded_episode(episode_dir: Path) -> dict[str, Any]:
    episode_dir = Path(episode_dir).expanduser()
    actions_path = episode_dir / "actions.jsonl"
    state_path = episode_dir / "newton_states.bin"
    initial_path = episode_dir / "initial_state.npz"
    metadata_path = episode_dir / "metadata.json"

    for p in (actions_path, state_path):
        if not p.exists():
            raise FileNotFoundError(f"Recorded episode missing {p}")

    actions = []
    with actions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                actions.append(json.loads(line))
    if not actions:
        raise RuntimeError(f"No recorded actions in {actions_path}")

    t0 = time.perf_counter()
    states = load_recorded_states(state_path)
    if "body_q" not in states:
        raise RuntimeError(f"{state_path} has no body_q snapshots")
    print(f"[DATA] loaded {states['n_states']} recorded Newton states in "
          f"{time.perf_counter() - t0:.1f} s from {state_path}")

    # State 0 is recorded at construction, state i+1 after action i.
    n_frames = min(len(actions), states["n_states"] - 1)
    if n_frames != len(actions):
        print(f"[DATA] WARNING: {len(actions)} actions but {states['n_states']} states; "
              f"using the first {n_frames} frames.")

    waypoint_pos = np.asarray([a["target_pos"] for a in actions[:n_frames]], dtype=np.float64)
    waypoint_xyzw = np.asarray([_norm4(a["target_xyzw"]) for a in actions[:n_frames]], dtype=np.float64)
    grip_key = "grip_requested_fraction" if "grip_requested_fraction" in actions[0] else "grip_fraction"
    waypoint_grip = np.asarray([float(a[grip_key]) for a in actions[:n_frames]], dtype=np.float64)

    return {
        "dir": episode_dir,
        "n_frames": int(n_frames),
        "waypoint_pos": waypoint_pos,
        "waypoint_xyzw": waypoint_xyzw,
        "waypoint_grip": waypoint_grip,
        "true_body_q": states["body_q"][: n_frames + 1],
        "true_body_qd": states.get("body_qd", None),
        "body_label": states["body_label"],
        "initial_path": initial_path if initial_path.exists() else None,
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {},
    }


def belt_state_error(sim_bq, true_bq, sim_bqd=None, true_bqd=None) -> dict[str, float]:
    """Belt state error: per-segment position / orientation, centroid, velocity."""
    dp = np.linalg.norm(sim_bq[:, :3] - true_bq[:, :3], axis=1)
    q_dot = np.abs(np.sum(sim_bq[:, 3:7] * true_bq[:, 3:7], axis=1)) / np.maximum(
        np.linalg.norm(sim_bq[:, 3:7], axis=1) * np.linalg.norm(true_bq[:, 3:7], axis=1), 1.0e-12)
    ang = 2.0 * np.arccos(np.clip(q_dot, 0.0, 1.0))
    err = {
        "pos_mean": float(dp.mean()),
        "pos_rms": float(np.sqrt(np.mean(dp * dp))),
        "pos_max": float(dp.max()),
        "centroid": float(np.linalg.norm(sim_bq[:, :3].mean(axis=0) - true_bq[:, :3].mean(axis=0))),
        "rot_mean_deg": float(np.degrees(ang.mean())),
        "vel_mean": float("nan"),
    }
    if sim_bqd is not None and true_bqd is not None:
        err["vel_mean"] = float(np.linalg.norm(sim_bqd[:, :3] - true_bqd[:, :3], axis=1).mean())
    return err


# Surface the TCP point can touch (raised task board, table top, else floor).
def surface_height_under(x: float, y: float) -> float:
    if (BOARD_ROOT_X <= x <= BOARD_ROOT_X + BOARD_SIZE
            and BOARD_ROOT_Y <= y <= BOARD_ROOT_Y + BOARD_SIZE):
        return BOARD_ROOT_Z
    if abs(x) <= 0.5 * TABLE_LENGTH_X and abs(y) <= 0.5 * TABLE_WIDTH_Y:
        return TABLE_TOP_Z
    return TABLE_BOTTOM_Z


# C3 controller
class C3Controller:
    """C3 MPC on a task-space LCS of the gripper TCP.

    LCS, one step = one control frame dt:
        x = [p, v]   TCP position / velocity (world)
        u = a        commanded TCP acceleration
        lam          Anitescu friction-pyramid impulses (per unit mass) of the
                     TCP against the surface below it
        x+ = A x + B u + D lam
        0 <= lam  _|_  eta = E x + F lam + H u + c >= 0

    The complementarity slack eta is an explicit decision variable (C3+ form),
    so every ADMM projection is an element-wise closed form instead of a MIQP:
        z-step : equality-constrained QP (dynamics + slack definition), whose
                 KKT matrices are constant and pre-inverted per rho value
        delta  : project each (lam_j, eta_j) pair onto {lam*eta = 0, lam, eta >= 0}
        w      : scaled dual update, rho increases every iteration
    """

    def __init__(self, dt, horizon=C3_HORIZON, q_pos=C3_Q_POS, q_vel=C3_Q_VEL,
                 q_terminal_scale=C3_Q_TERMINAL_SCALE, r_acc=C3_R_ACC, mu=C3_FRICTION_MU,
                 n_friction_dirs=C3_FRICTION_DIRS, admm_iterations=C3_ADMM_ITERATIONS,
                 rho_initial=C3_RHO_INITIAL, rho_scale=C3_RHO_SCALE,
                 g_lambda=C3_G_LAMBDA, g_eta=C3_G_ETA):
        self.dt = float(dt)
        self.N = N = int(horizon)
        self.m = m = int(n_friction_dirs)
        self.nx, self.nu = 6, 3
        self.ns = ns = self.nx + self.nu + 2 * m  # [x, u, lam, eta] per step
        self.nz = nz = N * ns + self.nx  # + terminal state
        dt = self.dt

        # Friction pyramid edges: n + mu * d_j.
        normal = np.array([0.0, 0.0, 1.0])
        dirs = [np.array([math.cos(2.0 * math.pi * j / m), math.sin(2.0 * math.pi * j / m), 0.0])
                for j in range(m)]
        Cc = np.stack([normal + mu * d for d in dirs], axis=1)  # 3 x m
        I3 = np.eye(3)

        self.A = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
        self.B = np.vstack([dt * dt * I3, dt * I3])
        self.D = np.vstack([dt * Cc, Cc])
        # eta_j = c_j . v+ + (p_z - z_surface) / dt
        self.E = np.zeros((m, 6))
        self.E[:, 2] = 1.0 / dt
        self.E[:, 3:6] = Cc.T
        self.F = Cc.T @ Cc
        self.H = dt * Cc.T

        # Index helpers.
        self._x = lambda k: slice(k * ns, k * ns + 6)
        self._u = lambda k: slice(k * ns + 6, k * ns + 9)
        self._l = lambda k: slice(k * ns + 9, k * ns + 9 + m)
        self._e = lambda k: slice(k * ns + 9 + m, k * ns + 9 + 2 * m)
        self.lam_idx = np.concatenate([np.arange(k * ns + 9, k * ns + 9 + m) for k in range(N)])
        self.eta_idx = self.lam_idx + m

        # Quadratic cost z^T P z + q^T z.
        eps = 1.0e-6
        self.Q = np.diag([q_pos] * 3 + [q_vel] * 3)
        self.Q_N = q_terminal_scale * self.Q
        P = np.zeros((nz, nz))
        for k in range(N):
            P[self._x(k), self._x(k)] = self.Q if k > 0 else eps * np.eye(6)
            P[self._u(k), self._u(k)] = r_acc * I3
            P[self._l(k), self._l(k)] = eps * np.eye(m)
            P[self._e(k), self._e(k)] = eps * np.eye(m)
        P[self._x(N), self._x(N)] = self.Q_N
        self.P = P

        # Equality constraints: x0, dynamics, slack definition.
        self.n_eq = n_eq = 6 + N * (6 + m)
        Aeq = np.zeros((n_eq, nz))
        Aeq[0:6, self._x(0)] = np.eye(6)
        self._comp_rows = []
        r = 6
        for k in range(N):
            Aeq[r:r + 6, self._x(k + 1)] = np.eye(6)
            Aeq[r:r + 6, self._x(k)] = -self.A
            Aeq[r:r + 6, self._u(k)] = -self.B
            Aeq[r:r + 6, self._l(k)] = -self.D
            r += 6
            Aeq[r:r + m, self._e(k)] = np.eye(m)
            Aeq[r:r + m, self._x(k)] = -self.E
            Aeq[r:r + m, self._l(k)] = -self.F
            Aeq[r:r + m, self._u(k)] = -self.H
            self._comp_rows.append(slice(r, r + m))
            r += m
        self.Aeq = Aeq

        # ADMM consensus weights G (only the complementarity pairs are projected).
        g = np.zeros(nz)
        g[self.lam_idx] = g_lambda
        g[self.eta_idx] = g_eta
        self.g = g
        self.g_lambda = float(g_lambda)
        self.g_eta = float(g_eta)

        # Constant KKT matrices -> pre-invert once per rho of the schedule.
        self.rhos = [float(rho_initial) * float(rho_scale) ** i for i in range(int(admm_iterations))]
        self.kkt_inv = []
        for rho in self.rhos:
            K = np.zeros((nz + n_eq, nz + n_eq))
            K[:nz, :nz] = 2.0 * P + 2.0 * rho * np.diag(g)
            K[:nz, nz:] = Aeq.T
            K[nz:, :nz] = Aeq
            self.kkt_inv.append(np.linalg.inv(K))

        self._delta = np.zeros(nz)
        self.last_info: dict[str, float] = {}
        print(f"[C3] LCS TCP model: horizon={N}, dt={dt:.4f} s, {m} friction-pyramid contacts, "
              f"{len(self.rhos)} ADMM iterations, QP size={nz} vars / {n_eq} eq")

    def _shift_warm_start(self):
        ns, N = self.ns, self.N
        d = self._delta
        shifted = d.copy()
        shifted[: (N - 1) * ns] = d[ns: N * ns]
        shifted[(N - 1) * ns: N * ns] = d[(N - 1) * ns: N * ns]
        self._delta = shifted

    def _project(self, v):
        """Closed-form projection of every (lam_j, eta_j) onto the complementarity set."""
        delta = v.copy()
        a = v[self.lam_idx]
        b = v[self.eta_idx]
        lam_a = np.maximum(a, 0.0)
        eta_b = np.maximum(b, 0.0)
        cost_a = self.g_lambda * (lam_a - a) ** 2 + self.g_eta * b * b  # contact mode
        cost_b = self.g_lambda * a * a + self.g_eta * (eta_b - b) ** 2  # separation mode
        use_a = cost_a <= cost_b
        delta[self.lam_idx] = np.where(use_a, lam_a, 0.0)
        delta[self.eta_idx] = np.where(use_a, 0.0, eta_b)
        return delta

    def solve(self, x0, p_ref, v_ref, z_contact):
        """
        x0        : (6,)  measured TCP [p, v]
        p_ref     : (N+1, 3) reference TCP positions (row 0 unused)
        v_ref     : (N+1, 3) reference TCP velocities (row 0 unused)
        z_contact : (N,)  contact plane height for every step
        returns   : dict with planned x (N+1, 6), u (N, 3), lam (N, m)
        """
        N, nz = self.N, self.nz
        q = np.zeros(nz)
        for k in range(1, N + 1):
            Qk = self.Q if k < N else self.Q_N
            xr = np.concatenate([p_ref[k], v_ref[k]])
            q[self._x(k)] = -2.0 * Qk @ xr

        b = np.zeros(self.n_eq)
        b[0:6] = x0
        for k, rows in enumerate(self._comp_rows):
            b[rows] = -float(z_contact[k]) / self.dt

        self._shift_warm_start()
        delta = self._delta
        w = np.zeros(nz)
        z = delta
        for rho, kinv in zip(self.rhos, self.kkt_inv):
            rhs = np.concatenate([-q + 2.0 * rho * self.g * (delta - w), b])
            z = (kinv @ rhs)[:nz]
            delta = self._project(z + w)
            w = w + z - delta
        self._delta = delta

        x_plan = np.stack([z[self._x(k)] for k in range(N + 1)], axis=0)
        u_plan = np.stack([z[self._u(k)] for k in range(N)], axis=0)
        lam_plan = np.stack([z[self._l(k)] for k in range(N)], axis=0)
        lam = z[self.lam_idx]
        eta = z[self.eta_idx]
        self.last_info = {
            "comp_residual": float(np.max(np.abs(np.minimum(lam, eta)))) if lam.size else 0.0,
            "consensus_residual": float(np.max(np.abs(z - delta))),
            "min_gap": float(np.min(x_plan[1:, 2] - np.asarray(z_contact, dtype=np.float64))),
            "contact_impulse": float(np.max(np.sum(np.maximum(lam_plan, 0.0), axis=1))),
        }
        return {"x": x_plan, "u": u_plan, "lam": lam_plan}



class Example(round_belt.Example):
    """round_belt.Example (identical scene / solvers / gripper logic / physics),
    but the arm TCP follows the RECORDED end-effector waypoints through a C3 MPC
    instead of the SpaceMouse, and every frame the simulated belt is compared
    against the recorded belt."""

    def __init__(self, viewer, args):
        # Baseline scene build, solvers, pipelines, IK, teleop state, CUDA graphs.
        super().__init__(viewer, args)

        # Recorded episode: waypoints are the ONLY control input, belt states are
        # only used as ground truth for the error report.
        self.episode = load_recorded_episode(Path(self.args.data_dir))
        self._check_episode_metadata(self.episode["metadata"])

        # Start from the exact recorded initial state (same belt / robot pose).
        if self.args.init_from_data:
            self._load_initial_state_from_episode()

        # Ground-truth consistency checks.
        true_bq = self.episode["true_body_q"]
        if true_bq.shape[1] != int(self.model.body_count):
            raise RuntimeError(
                f"Recorded episode has {true_bq.shape[1]} bodies, this scene has "
                f"{int(self.model.body_count)}; the scene setup does not match the recording.")
        self._belt_idx = np.asarray(self.belt_bodies, dtype=np.int32)
        self._true_belt_qd = None
        if self.episode["true_body_qd"] is not None:
            self._true_belt_qd = self.episode["true_body_qd"][:, self._belt_idx, :]
        self._true_belt_q = true_bq[:, self._belt_idx, :]
        e0 = belt_state_error(self.state_0.body_q.numpy()[self._belt_idx].astype(np.float64),
                              self._true_belt_q[0])
        print(f"[DATA] t=0 belt error vs recording: mean={1e3 * e0['pos_mean']:.3f} mm, "
              f"max={1e3 * e0['pos_max']:.3f} mm")

        # Waypoints (end-effector TCP pose + grip command per frame).
        self.n_frames = int(self.episode["n_frames"])
        if self.args.waypoint_source == "actual":
            # TCP actually reached in the recording (state i+1 after action i).
            tq = true_bq[1: self.n_frames + 1, self.link_index, :]
            self.wp_pos = np.asarray(
                [tq[i, :3] + _rotate_vec(_norm4(tq[i, 3:7]), self.tip_offset) for i in range(len(tq))])
            self.wp_xyzw = np.asarray([_norm4(tq[i, 3:7]) for i in range(len(tq))])
        else:
            # Commanded TCP target stored in actions.jsonl.
            self.wp_pos = self.episode["waypoint_pos"]
            self.wp_xyzw = self.episode["waypoint_xyzw"]
        self.wp_grip = np.clip(self.episode["waypoint_grip"], 0.0, 1.0)
        print(f"[DATA] {self.n_frames} end-effector waypoints ({self.args.waypoint_source}), "
              f"duration {self.n_frames * self.frame_dt:.2f} s")

        tcp0, quat0 = self._measure_tcp()
        self._target_pos = tcp0.copy()
        self._target_xyzw = quat0.copy()
        self._tcp_prev = tcp0.copy()

        # C3 controller on the TCP task-space LCS.
        self.c3 = C3Controller(
            dt=self.frame_dt,
            horizon=self.args.c3_horizon,
            q_pos=self.args.c3_q_pos,
            q_vel=self.args.c3_q_vel,
            r_acc=self.args.c3_r_acc,
            mu=self.args.c3_mu,
            admm_iterations=self.args.c3_admm_iterations,
            rho_initial=self.args.c3_rho,
            rho_scale=self.args.c3_rho_scale,
        )
        self.c3_command_step = int(np.clip(self.args.c3_command_step, 1, self.c3.N))
        self._c3_plan = None
        self._c3_time_s = 0.0

        # Error bookkeeping (printed only).
        self._done = False
        self._err_rows = []

        # Extra visuals: recorded belt (orange) + C3 plan (magenta).
        n_belt = len(self.belt_bodies)
        self._true_belt_colors = wp.array([wp.vec3(1.0, 0.55, 0.0)] * n_belt, dtype=wp.vec3, device=self.device)
        self._plan_colors = wp.array([wp.vec3(1.0, 0.2, 0.9)] * self.c3.N, dtype=wp.vec3, device=self.device)

    # Recorded episode helpers
    def _check_episode_metadata(self, metadata) -> None:
        for key, current in (("fps", self.fps), ("sim_substeps", self.sim_substeps)):
            if key in metadata and int(metadata[key]) != int(current):
                raise RuntimeError(
                    f"Episode metadata mismatch for {key}: recorded={metadata[key]}, current={current}")
        if "sim_dt" in metadata and not np.isclose(float(metadata["sim_dt"]), float(self.sim_dt)):
            raise RuntimeError(
                f"Episode metadata mismatch for sim_dt: recorded={metadata['sim_dt']}, current={self.sim_dt}")

    def _load_initial_state_from_episode(self) -> None:
        """Copy initial_state.npz into both states (robot, pulleys and belt at t=0)."""
        path = self.episode["initial_path"]
        if path is None:
            print("[DATA] initial_state.npz missing; starting from the default scene pose.")
            return
        data = np.load(path, allow_pickle=False)
        for state in (self.state_0, self.state_1):
            for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
                if name in data.files:
                    value = getattr(state, name, None)
                    if value is not None and value.shape[0] == data[name].shape[0]:
                        value.assign(data[name])
        if "control_joint_target_q" in data.files:
            self.control.joint_target_q.assign(data["control_joint_target_q"])
        if "arm_cmd" in data.files:
            self._arm_cmd = np.asarray(data["arm_cmd"], dtype=np.float64).copy()
            ik_q = self.ik_joint_q.numpy()
            for j, ik_qi in enumerate(self.ik_arm_coord_indices):
                ik_q[0, ik_qi] = float(self._arm_cmd[j])
            self.ik_joint_q.assign(ik_q)
        if "grip_requested_fraction" in data.files:
            self._grip_requested_fraction = float(np.asarray(data["grip_requested_fraction"]).reshape(-1)[0])
        if "grip_fraction" in data.files:
            self._grip_fraction = float(np.asarray(data["grip_fraction"]).reshape(-1)[0])
        print(f"[DATA] initial state loaded from {path}")

    def _measure_tcp(self):
        """Current physical TCP position + orientation from the solved Newton state."""
        bq = self.state_0.body_q.numpy()[self.link_index]
        base_pos = np.asarray(bq[0:3], dtype=np.float64)
        base_quat = _norm4(np.asarray(bq[3:7], dtype=np.float64))
        return base_pos + _rotate_vec(base_quat, self.tip_offset), base_quat

    def _c3_reference(self, frame):
        """Horizon reference from the recorded waypoints: x_k <- waypoint[frame + k - 1]."""
        N = self.c3.N
        last = self.n_frames - 1
        idx = [min(max(frame + k - 1, 0), last) for k in range(N + 1)]
        p_ref = self.wp_pos[idx]
        v_ref = np.asarray(
            [(self.wp_pos[j] - self.wp_pos[max(j - 1, 0)]) / self.frame_dt for j in idx])
        return p_ref, v_ref

    # Sim loop: C3 waypoint tracking + coupled physics stepping
    def step(self):
        if self._done:
            return
        if self.frame_id >= self.n_frames:
            self._finish()
            return

        # Charge the real-time clock from the first real frame.
        if self.frame_id == 0:
            self.wall_start_time = time.perf_counter()
            self.last_timing_print = self.wall_start_time
            self._last_sim_time = self.sim_time
            self._last_wall_time = self.wall_start_time

        i = self.frame_id

        # Measured TCP state for the C3 LCS (velocity by finite difference).
        current_tcp, _ = self._measure_tcp()
        tcp_vel = (current_tcp - self._tcp_prev) / self.frame_dt if i > 0 else np.zeros(3)
        self._tcp_prev = current_tcp.copy()
        x0 = np.concatenate([current_tcp, tcp_vel])

        # C3 MPC toward the recorded waypoints of the next horizon.
        p_ref, v_ref = self._c3_reference(i)
        z_contact = np.asarray([
            surface_height_under(*(current_tcp[:2] if k == 0 else p_ref[k, :2])) + self.args.c3_surface_offset
            for k in range(self.c3.N)
        ])
        t0 = time.perf_counter()
        self._c3_plan = self.c3.solve(x0, p_ref, v_ref, z_contact)
        self._c3_time_s += time.perf_counter() - t0

        # The planned TCP position becomes the IK target (lead-limited like teleop).
        candidate = np.asarray(self._c3_plan["x"][self.c3_command_step, :3], dtype=np.float64)
        lead = candidate - current_tcp
        lead_norm = float(np.linalg.norm(lead))
        if lead_norm > TRACK_MAX_TARGET_DISTANCE and lead_norm > 1.0e-12:
            candidate = current_tcp + lead * (TRACK_MAX_TARGET_DISTANCE / lead_norm)
        if np.isfinite(candidate).all():
            self._target_pos = candidate
        # The TCP LCS is translational; orientation follows the waypoint directly.
        self._target_xyzw = _norm4(self.wp_xyzw[i])

        # Grip command of the waypoint -> same anti-crush latch as teleop.
        self._grip_requested_fraction = float(self.wp_grip[i])
        self._update_gripper_antcrush()
        self._update_gripper_release_material()

        # IK for the C3 TCP target (warm-started from the current arm command).
        self.pos_obj.set_target_position(0, _v3(self._target_pos))
        self.rot_obj.set_target_rotation(0, _v4(self._target_xyzw))

        ik_seed = self.ik_joint_q.numpy()
        for j, ik_qi in enumerate(self.ik_arm_coord_indices):
            ik_seed[0, ik_qi] = float(self._arm_cmd[j])
        self.ik_joint_q.assign(ik_seed)
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=2 * IK_TRACK_ITERS)

        solved = self.ik_joint_q.numpy().reshape(-1)
        if np.isfinite(solved).all():
            self._write_control_targets(solved)

        # Coupled MuJoCo(robot) + VBD(belt) stepping (full-scene physics).
        self._launch_physics()

        self.sim_time += self.frame_dt
        self.frame_id += 1

        # Belt error vs the recorded (true) belt state of the same frame.
        self._record_error(i)

        wall_now = time.perf_counter()
        wall_time = wall_now - self.wall_start_time
        if wall_now - self.last_timing_print >= 1.0:
            window_sim = self.sim_time - self._last_sim_time
            window_wall = wall_now - self._last_wall_time
            rtf_now = window_sim / window_wall if window_wall > 1.0e-9 else 0.0
            rtf_avg = self.sim_time / wall_time if wall_time > 1.0e-9 else 0.0
            self._last_sim_time = self.sim_time
            self._last_wall_time = wall_now
            self.last_timing_print = wall_now
            print(
                f"[TIME] sim={self.sim_time:8.3f} s | real={wall_time:8.3f} s | "
                f"RTF(now)={rtf_now:6.3f}x | RTF(avg)={rtf_avg:6.3f}x | "
                f"C3={1e3 * self._c3_time_s / max(self.frame_id, 1):5.2f} ms/frame"
            )

        if self.frame_id >= self.n_frames:
            self._finish()

    def _record_error(self, action_index: int) -> None:
        truth_index = action_index + 1  # state recorded right after this action
        sim_bq = self.state_0.body_q.numpy()[self._belt_idx].astype(np.float64)
        sim_bqd = self.state_0.body_qd.numpy()[self._belt_idx].astype(np.float64)
        true_bqd = self._true_belt_qd[truth_index] if self._true_belt_qd is not None else None
        err = belt_state_error(sim_bq, self._true_belt_q[truth_index], sim_bqd, true_bqd)

        tcp, _ = self._measure_tcp()
        ee_err = float(np.linalg.norm(tcp - self.wp_pos[action_index]))
        row = {
            "belt_pos_mean_mm": 1e3 * err["pos_mean"],
            "belt_pos_rms_mm": 1e3 * err["pos_rms"],
            "belt_pos_max_mm": 1e3 * err["pos_max"],
            "belt_centroid_mm": 1e3 * err["centroid"],
            "belt_rot_mean_deg": err["rot_mean_deg"],
            "belt_vel_mean_mps": err["vel_mean"],
            "ee_waypoint_err_mm": 1e3 * ee_err,
        }
        self._err_rows.append(row)

        every = int(self.args.error_print_every)
        if every > 0 and (action_index % every == 0 or action_index == self.n_frames - 1):
            latch = "ON" if self._grip_hold_fraction is not None else "off"
            print(
                f"[BELT ERR] frame {action_index:5d}/{self.n_frames} | "
                f"pos mean={row['belt_pos_mean_mm']:7.2f} rms={row['belt_pos_rms_mm']:7.2f} "
                f"max={row['belt_pos_max_mm']:7.2f} mm | centroid={row['belt_centroid_mm']:7.2f} mm | "
                f"rot={row['belt_rot_mean_deg']:6.2f} deg | ee-wp={row['ee_waypoint_err_mm']:5.1f} mm | "
                f"grip={self._grip_requested_fraction:0.2f} latch={latch}"
            )

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        if not self._err_rows:
            print("[RESULT] no frames were simulated.")
            return

        last = self._err_rows[-1]
        print("")
        print("=" * 96)
        print(f"[RESULT] C3 waypoint tracking vs recorded belt ({len(self._err_rows)} frames, "
              f"waypoints={self.args.waypoint_source})")
        for k, label in (("belt_pos_mean_mm", "belt segment pos error  [mm] "),
                         ("belt_pos_max_mm", "belt segment max error  [mm] "),
                         ("belt_centroid_mm", "belt centroid error     [mm] "),
                         ("belt_rot_mean_deg", "belt segment rot error [deg] "),
                         ("belt_vel_mean_mps", "belt segment vel error [m/s] "),
                         ("ee_waypoint_err_mm", "TCP vs waypoint error   [mm] ")):
            v = np.asarray([r[k] for r in self._err_rows], dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                print(f"  {label}: mean={v.mean():8.3f}  max={v.max():8.3f}  final={last[k]:8.3f}")
        print("=" * 96)

        if hasattr(self.viewer, "_pause"):
            self.viewer._pause = True

    def _log_polyline(self, name, points, colors, closed=False):
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 2:
            return
        a = pts if closed else pts[:-1]
        b = np.roll(pts, -1, axis=0) if closed else pts[1:]
        starts = wp.array([_v3(p) for p in a], dtype=wp.vec3, device=self.device)
        ends = wp.array([_v3(p) for p in b], dtype=wp.vec3, device=self.device)
        self.viewer.log_lines(name, starts, ends, colors)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self._main_view_layer is not None:
            self.viewer.activate(self._main_view_layer)
        newton.examples.log_coupled_view(self, self.contacts)

        # live gripper TIP frame + C3 TCP target frame/sphere
        tip, base_quat = self._measure_tcp()
        self._log_frame("/tip_current", tip, base_quat, self.axis_len, self._axis_bright)
        self._log_frame("/tip_target_axes", self._target_pos, self._target_xyzw,
                        self.axis_len, self._axis_dim)
        self._log_sphere("/tip_target_sphere", self._target_pos, self.tip_radius)

        # C3 planned TCP path (magenta) and recorded belt centerline (orange).
        if self._c3_plan is not None:
            self._log_polyline("/c3_plan", self._c3_plan["x"][:, :3], self._plan_colors)
        if self.args.show_true_belt:
            k = min(self.frame_id, len(self._true_belt_q) - 1)
            self._log_polyline("/true_belt", self._true_belt_q[k, :, :3], self._true_belt_colors, closed=True)

        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        # Baseline round_belt options (cuda graph, visuals, max arm speed, ...).
        parser = round_belt.Example.create_parser()

        # recorded episode
        parser.add_argument("--data-dir", type=str, default=DATA_DIR_DEFAULT,
                            help="episode folder with actions.jsonl, newton_states.bin, initial_state.npz")
        parser.add_argument("--waypoint-source", choices=("target", "actual"), default="target",
                            help="'target': recorded commanded TCP target; "
                                 "'actual': TCP pose actually reached in the recording")
        parser.add_argument("--no-init-from-data", action="store_false", dest="init_from_data",
                            default=True, help="start from the default scene pose instead of initial_state.npz")
        parser.add_argument("--error-print-every", type=int, default=30,
                            help="print the belt error every N frames (0 = only the final summary)")
        parser.add_argument("--hide-true-belt", action="store_false", dest="show_true_belt",
                            default=True, help="do not draw the recorded belt centerline")

        # C3 controller
        parser.add_argument("--c3-horizon", type=int, default=C3_HORIZON)
        parser.add_argument("--c3-q-pos", type=float, default=C3_Q_POS)
        parser.add_argument("--c3-q-vel", type=float, default=C3_Q_VEL)
        parser.add_argument("--c3-r-acc", type=float, default=C3_R_ACC)
        parser.add_argument("--c3-mu", type=float, default=C3_FRICTION_MU)
        parser.add_argument("--c3-admm-iterations", type=int, default=C3_ADMM_ITERATIONS)
        parser.add_argument("--c3-rho", type=float, default=C3_RHO_INITIAL)
        parser.add_argument("--c3-rho-scale", type=float, default=C3_RHO_SCALE)
        parser.add_argument("--c3-command-step", type=int, default=C3_COMMAND_STEP,
                            help="planned state index sent to IK as the TCP target")
        parser.add_argument("--c3-surface-offset", type=float, default=C3_SURFACE_OFFSET,
                            help="TCP contact plane = table/board top + offset [m]")
        return parser

    def test_final(self):
        super().test_final()
        self._finish()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    viewer._pause = False
    newton.examples.run(Example(viewer, args), args)
