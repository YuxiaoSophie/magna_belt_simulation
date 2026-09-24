#!/usr/bin/env python3
"""Tuple verification for the OSC-backend LCS dataset: timing, action alignment
(``knot1_minus_measured``), tracking error, belt-point identity, the ``lcs_learning`` loader and
the OU excitation's smoothness against white excitation.

Collects 2 episodes (``--intents engaged,over``) plus 1 ``--scenario pure_translation`` episode
into a temp dir and asserts that every ``(x_t, u_t, x_{t+1})`` tuple is what MPC will query; T7
re-collects the 2 episodes with ``--excite-mode white`` for the jerk comparison.

Run:
    uv run python scripts/checks/check_lcs_tuples.py
    uv run python scripts/checks/check_lcs_tuples.py --keep
"""

from __future__ import annotations

import argparse
import json
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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "scripts" / "checks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "checks"))

import collect_lcs_dataset as cli
from check_lcs_collector import (
    LCS_LEARNING_ROOT,
    LCS_LEARNING_VENV,
    LOADER_SHAPES,
    Skip,
    _require,
)

from round_belt_task.commander import CommanderParams
from task_common import lcs_dataset as lcs

# A private multicast group so this check never disturbs a running magna stack.
PRIVATE_LCM_URL = "udpm://239.255.76.86:7686?ttl=0"
PRIVATE_PORT = "7686"
RUNTIME_BUDGET_S = 300.0
PARAMS = CommanderParams()

SIM_TIME_TOL = 1e-9
ACTION_DEFINITION = cli.OSC_ACTION_DEFINITION
ACTION_TOL = 1e-9  # dxyz/drotvec identity
FREE_MOVE_TOL = 0.0  # knot0 is the measured pose bit-for-bit
IDENTITY_TOL_M = 1e-9  # ee_{t+1} - (ee_t + u_t) == ee_{t+1} - knot1_t
# knot0 == measured pose only far from target (RUN-STATE trap); near/at target (move, waiting on
# the UR) or during a hold, knot0 is the target/latched pose, within the commander's own reach
# tolerance of the measured pose.
KNOT0_POS_TOL_M = PARAMS.pos_tol + 1e-6
KNOT0_ORI_TOL_RAD = PARAMS.ori_tol + 1e-6
T3_POS_TOL = 1e-6
T3_ROT_TOL = 1e-6
T3_ROT_BOUND_TOL = 1e-9
T3_HOLD_TOL = 1e-12
T3_MIN_MOVE_ROWS = 5
TRACKING_DIAG_TOL_MM = 1e-6
FRANKA_P95_MM = 5.0
FRANKA_MAX_MM = 15.0
UR_P95_MM = 4.0  # true lag behind the line now; the old residual cancelled it
FRANKA_MAX_MIN_M = 1e-6  # a tracking error, not an identity
# float32 pcd_belt/sim_belt_xyz: half an ulp at ~0.5 m over an 11 mm segment is ~3e-6.
BELT_DRIFT_TOL_BODY_UNITS = 1e-5
T5_MIN_STRETCH = 0.05  # some segment length changes >= 5 % over the episode
T5_MIN_OLD_DRIFT = 0.1  # arc-length sampling would drift >= 0.1 body units here
BELT_POINT0_TOL_M = 1e-6
T6_TIMEOUT_S = 120.0
T6_ACTION_TOL = 1e-6
T6_STATE_TOL = 1e-5  # float32 loader cast
JERK_RATIO_MAX = 0.5

RUN_ARGV = ["--backend", "osc", "--episodes", "2", "--seed", "3",
            "--intents", "engaged,over", "--weights", "1,1"]
PURE_ARGV = ["--scenario", "pure_translation", "--episodes", "1", "--seed", "3"]

CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _delta_rotvec_batch(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    return np.array([lcs.delta_rotvec(a, b) for a, b in zip(q0, q1)])


def _gpu_apps() -> set[int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {int(v) for v in out.split() if v.strip().isdigit()}


def _material_coord(belt_xyz_m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``i + t`` (body units) of each point's nearest projection on the closed body loop."""
    loop = np.vstack([belt_xyz_m, belt_xyz_m[:1]]).astype(np.float64)
    seg = np.diff(loop, axis=0)
    ln2 = np.sum(seg * seg, axis=1)
    pts = np.asarray(pts, dtype=np.float64)
    best_s, best_d = np.zeros(len(pts)), np.full(len(pts), np.inf)
    for i in range(len(seg)):
        t = np.clip(((pts - loop[i]) @ seg[i]) / ln2[i], 0.0, 1.0)
        dist = np.linalg.norm(pts - (loop[i] + t[:, None] * seg[i]), axis=1)
        better = dist < best_d
        best_d[better], best_s[better] = dist[better], i + t[better]
    return best_s


def _wrap(ds: np.ndarray, m: int) -> np.ndarray:
    return (ds + m / 2) % m - m / 2


def _arc_length_points(belt_xyz_m: np.ndarray, n: int = lcs.BELT_POINTS) -> np.ndarray:
    """The old equal-current-arc-length resampling (reference for the drift it caused)."""
    loop = np.vstack([belt_xyz_m, belt_xyz_m[:1]]).astype(np.float64)
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(n) * (cum[-1] / n)
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    frac = (s - cum[i]) / seg[i]
    return loop[i] + frac[:, None] * (loop[i + 1] - loop[i])


@check("T0 collect")
def check_t0(ctx: SimpleNamespace) -> str:
    run_dir = ctx.tmp_root / "run"
    args = cli.create_parser().parse_args(
        [*RUN_ARGV, "--lcm-url", ctx.lcm_url, "--out", str(run_dir)])
    index = cli.collect(args)
    rows = index["episodes"]
    _require(len(rows) == 2, f"run: {len(rows)} episodes, expected 2")
    for row in rows:
        _require(row["status"] == "ok", f"run {row['intent']}: not ok: {row}")

    pure_dir = ctx.tmp_root / "pure"
    args2 = cli.create_parser().parse_args(
        [*PURE_ARGV, "--lcm-url", ctx.lcm_url, "--out", str(pure_dir)])
    index2 = cli.collect(args2)
    rows2 = index2["episodes"]
    _require(len(rows2) == 1, f"pure: {len(rows2)} episodes, expected 1")
    _require(rows2[0]["status"] == "ok", f"pure: not ok: {rows2[0]}")

    order = [("run0", run_dir / rows[0]["file"]), ("run1", run_dir / rows[1]["file"]),
             ("pure", pure_dir / rows2[0]["file"])]
    for name, path in order:
        summary = lcs.validate_episode(path, period_us=lcs.SAMPLE_PERIOD_US)
        _require(summary["period_us"] == lcs.SAMPLE_PERIOD_US,
                 f"{name}: period {summary['period_us']} us != {lcs.SAMPLE_PERIOD_US}")
    ctx.order = order
    ctx.data = {name: _load(path) for name, path in order}
    ctx.t_list = {name: ctx.data[name]["state"].shape[0] for name, _ in order}
    ctx.run_log = run_dir / "osc.log"
    names = ", ".join(n for n, _ in order)
    return f"3 files ok ({names}), period {lcs.SAMPLE_PERIOD_US} us"


@check("T1 timing")
def check_t1(ctx: SimpleNamespace) -> str:
    prev_last = None
    for name, _ in ctx.order:
        d = ctx.data[name]
        step = np.asarray(d["sim_step"])
        _require(bool(np.all(np.diff(step) == lcs.SAMPLE_STEPS)),
                 f"{name}: sim_step diffs != {lcs.SAMPLE_STEPS}")
        _require(len(set(step.tolist())) == len(step), f"{name}: duplicate sim_step")
        utime = np.asarray(d["utime"]).astype(np.int64)
        _require(bool(np.all(np.diff(utime) == lcs.SAMPLE_PERIOD_US)),
                 f"{name}: utime diffs != {lcs.SAMPLE_PERIOD_US}")
        sim_time = np.asarray(d["sim_time"])
        _require(bool(np.all(np.abs(np.diff(sim_time) - lcs.SAMPLE_PERIOD_S) < SIM_TIME_TOL)),
                 f"{name}: sim_time diffs off {lcs.SAMPLE_PERIOD_S}")
        _require(np.array_equal(d["sim_render_step"], d["sim_step"]),
                 f"{name}: sim_render_step != sim_step")
        osc_utime = np.asarray(d["sim_osc_utime"]).astype(np.int64)
        _require(bool(np.all(np.diff(osc_utime) == lcs.SAMPLE_PERIOD_US)),
                 f"{name}: sim_osc_utime diffs != {lcs.SAMPLE_PERIOD_US}")
        if name == "run0":
            prev_last = int(osc_utime[-1])
        elif name == "run1":
            _require(int(osc_utime[0]) > prev_last,
                     f"run1 first osc_utime {osc_utime[0]} <= run0 last {prev_last}")
    return (f"sim_step/utime/sim_osc_utime spaced {lcs.SAMPLE_STEPS} steps / "
            f"{lcs.SAMPLE_PERIOD_US} us; OSC clock monotonic across run0->run1")


def _free_move(d: dict) -> np.ndarray:
    """Rows where knot 0 is the measured pose (far from the target, not holding)."""
    k0, state = np.asarray(d["sim_cmd_knot0_franka"]), d["state"]
    return (~np.asarray(d["sim_cmd_hold"])
            & np.all(np.abs(k0[:, :3] - state[:, 26:29]) <= FREE_MOVE_TOL, axis=1))


@check("T2 action alignment")
def check_t2(ctx: SimpleNamespace) -> str:
    free_n = hold_n = reached_n = 0
    hold_max = reached_max = 0.0
    for name, _ in ctx.order:
        d = ctx.data[name]
        state, actions = d["state"], d["actions"]
        meta = json.loads(str(d["sim_meta"]))
        definition = meta["lcs_format"].get("action_definition")
        _require(definition == ACTION_DEFINITION, f"{name}: action_definition {definition!r}")
        k0, k1 = d["sim_cmd_knot0_franka"], d["sim_cmd_knot1_franka"]
        u1 = d["sim_cmd_ur_t1"]

        err = float(np.abs(actions[:, 0:3] - (k1[:, :3] - state[:, 26:29])).max())
        _require(err <= ACTION_TOL, f"{name}: dxyz_f != knot1 - measured ({err:.2e})")
        rot_err = float(np.abs(actions[:, 6:9]
                               - _delta_rotvec_batch(state[:, 29:33], k1[:, 3:])).max())
        _require(rot_err <= ACTION_TOL, f"{name}: drotvec_f != knot1 - measured ({rot_err:.2e})")
        err = float(np.abs(actions[:, 3:6] - (u1[:, :3] - state[:, 33:36])).max())
        _require(err <= ACTION_TOL, f"{name}: dxyz_ur != line(t+dt) - measured ({err:.2e})")
        rot_err = float(np.abs(actions[:, 9:12]
                               - _delta_rotvec_batch(state[:, 36:40], u1[:, 3:])).max())
        _require(rot_err <= ACTION_TOL, f"{name}: drotvec_ur != line(t+dt) - measured "
                                        f"({rot_err:.2e})")

        # knot0 vs measured: equal on free-move rows, within the reach tolerance otherwise.
        pos_err = float(np.linalg.norm(state[:, 26:29] - k0[:, :3], axis=1).max())
        _require(pos_err <= KNOT0_POS_TOL_M,
                 f"{name}: measured pos vs knot0 off by {pos_err * 1e3:.3f} mm "
                 f"> {KNOT0_POS_TOL_M * 1e3:g} mm")
        ori_err = float(np.linalg.norm(_delta_rotvec_batch(state[:, 29:33], k0[:, 3:]),
                                        axis=1).max())
        _require(ori_err <= KNOT0_ORI_TOL_RAD,
                 f"{name}: measured ori vs knot0 off by {ori_err:.4f} rad > {KNOT0_ORI_TOL_RAD:g}")
        free = _free_move(d)
        if free.any():
            err = float(np.abs(actions[free, 0:3] - (k1[free, :3] - k0[free, :3])).max())
            _require(err <= ACTION_TOL, f"{name}: free-move dxyz_f != knot1 - knot0 ({err:.2e})")
            rot_err = float(np.abs(actions[free, 6:9]
                                   - _delta_rotvec_batch(k0[free, 3:], k1[free, 3:])).max())
            _require(rot_err <= ACTION_TOL,
                     f"{name}: free-move drotvec_f != knot1 - knot0 ({rot_err:.2e})")
        free_n += int(free.sum())

        cap_factor = meta["excitation"]["cap_factor"]
        cap = PARAMS.lin_speed * PARAMS.dt * cap_factor + KNOT0_POS_TOL_M
        _require(bool(np.all(np.linalg.norm(actions[:, 0:3], axis=1) <= cap)),
                 f"{name}: |dxyz_f| exceeds cap + reach tol {cap * 1e3:.2f} mm")

        # Quiet hold rows: knot1 == knot0 == latched pose, so u = latched - measured.
        hold = np.asarray(d["sim_cmd_hold"])
        no_excite = (np.abs(d["sim_excite_dpos_m"]).max(axis=1) == 0.0) & \
                    (np.abs(d["sim_excite_rotvec"]).max(axis=1) == 0.0)
        quiet = hold & no_excite
        if quiet.any():
            err = float(np.abs(actions[quiet, 0:3]
                               - (k0[quiet, :3] - state[quiet, 26:29])).max())
            _require(err <= ACTION_TOL, f"{name}: quiet hold dxyz_f != latched - measured")
            hold_max = max(hold_max, float(np.linalg.norm(actions[quiet, 0:3], axis=1).max()))
            hold_n += int(quiet.sum())
        # Reached-while-moving rows (all knots = target): u = target - measured.
        knots = np.asarray(d["sim_cmd_knots_franka"])
        reached = ~hold & no_excite & np.all(np.abs(knots - knots[:, :1]) == 0.0, axis=(1, 2))
        if reached.any():
            err = float(np.abs(actions[reached, 0:3]
                               - (knots[reached, -1, :3] - state[reached, 26:29])).max())
            _require(err <= ACTION_TOL, f"{name}: reached dxyz_f != target - measured")
            reached_max = max(reached_max,
                              float(np.linalg.norm(actions[reached, 0:3], axis=1).max()))
            reached_n += int(reached.sum())
    _require(free_n > 0, "no free-move rows found")
    _require(hold_n > 0 and hold_max > 0.0, "no quiet hold row with a non-zero Franka action")
    return (f"u == knot1/line(t+dt) - measured (both arms); {free_n} free-move rows == knot1 - "
            f"knot0; {hold_n} quiet hold rows == latched - measured (max "
            f"{hold_max * 1e3:.2f} mm); {reached_n} reached rows == target - measured (max "
            f"{reached_max * 1e3:.2f} mm)")


@check("T3 closed form (pure_translation)")
def check_t3(ctx: SimpleNamespace) -> str:
    d = ctx.data["pure"]
    meta = json.loads(str(d["sim_meta"]))
    labels = meta["phase_labels"]
    phase = np.asarray(d["sim_phase"])
    state, actions = d["state"], d["actions"]
    k0 = np.asarray(d["sim_cmd_knot0_franka"])

    target = meta.get("target")
    if target is not None:
        target_pos, target_quat = np.asarray(target[:3]), np.asarray(target[3:])
        note = "sim_meta['target']"
    else:
        first_move = next(t for t in range(len(phase)) if labels[int(phase[t])].startswith("move"))
        knots0 = np.asarray(d["sim_cmd_knots_franka"])[first_move]
        target_pos, target_quat = knots0[-1, :3], knots0[-1, 3:]
        note = f"sim_cmd_knots_franka[{first_move}][-1] (sim_meta has no 'target')"

    move_n = hold_n = 0
    hold_max = 0.0
    latched = None
    for t in range(len(phase)):
        label = labels[int(phase[t])]
        ee_pos, ee_quat = state[t, 26:29], state[t, 29:33]
        if label.startswith("move"):
            d_vec = target_pos - ee_pos
            dist = float(np.linalg.norm(d_vec))
            if dist < 1e-9:
                continue
            # Also covers dist < pos_tol (all knots = target): u = target - measured.
            expected = min(PARAMS.lin_speed * PARAMS.dt, dist) * (d_vec / dist)
            err = float(np.abs(actions[t, 0:3] - expected).max())
            _require(err <= T3_POS_TOL, f"pure t={t}: dxyz_f off by {err:.2e} m")

            rv = lcs.delta_rotvec(ee_quat, target_quat)
            ang = float(np.linalg.norm(rv))
            bound = PARAMS.ang_speed * PARAMS.dt + T3_ROT_BOUND_TOL
            _require(float(np.linalg.norm(actions[t, 6:9])) <= bound,
                     f"pure t={t}: |drotvec_f| exceeds {bound:g} rad")
            if ang > 1e-9:
                s = min(PARAMS.dt / (ang / PARAMS.ang_speed), 1.0)
                expected_rot = rv / ang * (ang * s)
            else:
                expected_rot = np.zeros(3)
            err_rot = float(np.abs(actions[t, 6:9] - expected_rot).max())
            _require(err_rot <= T3_ROT_TOL, f"pure t={t}: drotvec_f off by {err_rot:.2e} rad")
            move_n += 1
        elif label.startswith("hold") or label == "done":
            # One latched pose for the whole hold, within the reach tolerance of the target.
            if latched is None:
                latched = k0[t].copy()
                off = float(np.linalg.norm(latched[:3] - target_pos))
                _require(off < PARAMS.pos_tol, f"pure t={t}: latched {off * 1e3:.2f} mm from "
                                               f"target >= {PARAMS.pos_tol * 1e3:g} mm")
            _require(np.array_equal(k0[t], latched), f"pure t={t}: latched pose changed")
            err = float(np.abs(actions[t, 0:3] - (latched[:3] - ee_pos)).max())
            _require(err <= T3_HOLD_TOL, f"pure t={t} ({label}): dxyz_f {err:.2e} from "
                                         "latched - measured")
            err = float(np.abs(actions[t, 6:9] - lcs.delta_rotvec(ee_quat, latched[3:])).max())
            _require(err <= T3_HOLD_TOL, f"pure t={t} ({label}): drotvec_f {err:.2e} from "
                                         "latched - measured")
            hold_max = max(hold_max, float(np.linalg.norm(actions[t, 0:3])))
            hold_n += 1
    _require(move_n >= T3_MIN_MOVE_ROWS,
             f"only {move_n} move rows tested, need >= {T3_MIN_MOVE_ROWS} (scenario too short)")
    _require(hold_n > 0 and hold_max > 0.0, "no hold row with a non-zero Franka action")
    return (f"{move_n} move rows closed form, {hold_n} hold/done rows == latched - measured "
            f"(max {hold_max * 1e3:.3f} mm), target from {note}")


@check("T4 tracking error")
def check_t4(ctx: SimpleNamespace) -> str:
    f_mm, u_mm, rot_deg, f_m = [], [], [], []
    ident = diag_max = 0.0
    for name, _ in ctx.order:
        d = ctx.data[name]
        state, actions = d["state"], d["actions"]
        ee_f, q_f, ee_u = state[:, 26:29], state[:, 29:33], state[:, 33:36]
        k1 = np.asarray(d["sim_cmd_knot1_franka"])
        u1 = np.asarray(d["sim_cmd_ur_t1"])
        e_f = ee_f[1:] - (ee_f[:-1] + actions[:-1, 0:3])
        e_u = ee_u[1:] - (ee_u[:-1] + actions[:-1, 3:6])
        # Under knot1_minus_measured the tuple residual IS the tracking error.
        # Consistency guard only: implied by T2's action == knot1 - measured.
        for arm, e, ref in (("franka", e_f, ee_f[1:] - k1[:-1, :3]),
                            ("ur", e_u, ee_u[1:] - u1[:-1, :3])):
            res = float(np.abs(e - ref).max())
            _require(res <= IDENTITY_TOL_M, f"{name} {arm}: residual != ee_(t+1) - cmd_t "
                                            f"({res:.2e} m)")
            ident = max(ident, res)
        sim_track = np.asarray(d["sim_tracking_err_mm"])
        for j, e in enumerate((e_f, e_u)):
            diag = float(np.abs(sim_track[1:, j] - np.linalg.norm(e, axis=1) * 1e3).max())
            _require(diag <= TRACKING_DIAG_TOL_MM,
                     f"{name}: sim_tracking_err_mm[:, {j}] off by {diag:.2e} mm")
            diag_max = max(diag_max, diag)
        rot_err = np.linalg.norm(_delta_rotvec_batch(k1[:-1, 3:], q_f[1:]), axis=1)

        meta = json.loads(str(d["sim_meta"]))
        labels = meta["phase_labels"]
        phase = np.asarray(d["sim_phase"])
        move = np.array([labels[int(p)].startswith("move") for p in phase[:-1]])
        if not move.any():
            continue
        f_mm.extend((np.linalg.norm(e_f[move], axis=1) * 1e3).tolist())
        u_mm.extend((np.linalg.norm(e_u[move], axis=1) * 1e3).tolist())
        rot_deg.extend(np.degrees(rot_err[move]).tolist())
        f_m.extend(np.linalg.norm(e_f[move], axis=1).tolist())

    f_mm, u_mm, rot_deg = np.array(f_mm), np.array(u_mm), np.array(rot_deg)
    _require(len(f_mm) > 0, "no move-phase rows found across files")
    p95_f, max_f = float(np.percentile(f_mm, 95)), float(f_mm.max())
    p95_u = float(np.percentile(u_mm, 95))
    _require(p95_f <= FRANKA_P95_MM, f"franka p95 {p95_f:.2f} mm > {FRANKA_P95_MM:g}")
    _require(max_f <= FRANKA_MAX_MM, f"franka max {max_f:.2f} mm > {FRANKA_MAX_MM:g}")
    _require(p95_u <= UR_P95_MM, f"ur p95 {p95_u:.2f} mm > {UR_P95_MM:g}")
    _require(max(f_m) > FRANKA_MAX_MIN_M,
             f"franka max tracking error {max(f_m):.2e} m looks like an identity, not tracking")
    return (f"identity residual {ident:.1e} m, sim_tracking_err_mm diff {diag_max:.1e} mm; "
            f"move rows: franka mean {f_mm.mean():.2f} / median {float(np.median(f_mm)):.2f} / "
            f"p95 {p95_f:.2f} / max {max_f:.2f} mm; ur mean {u_mm.mean():.2f} / p95 {p95_u:.2f} "
            f"mm; rot mean {rot_deg.mean():.4f} / max {rot_deg.max():.4f} deg")


@check("T5 belt-point identity (material)")
def check_t5(ctx: SimpleNamespace) -> str:
    i_k, f_k = lcs.material_table()
    s_k = i_k + f_k
    max_drift, fixtures = 0.0, []
    for name, _ in ctx.order:
        d = ctx.data[name]
        sampling = json.loads(str(d["sim_meta"]))["lcs_format"].get("belt_sampling")
        _require(sampling == "material", f"{name}: belt_sampling {sampling!r}")
        belt_xyz = np.asarray(d["sim_belt_xyz"], dtype=np.float64)
        pcd_belt = np.asarray(d["pcd_belt"])
        m = belt_xyz.shape[1]
        seg = np.linalg.norm(np.diff(np.concatenate([belt_xyz, belt_xyz[:, :1]], 1), axis=1),
                             axis=2)
        stretch = float(np.abs(seg / seg[:1] - 1.0).max())
        old0 = _material_coord(belt_xyz[0], _arc_length_points(belt_xyz[0]))
        drift = old_drift = 0.0
        for t in range(belt_xyz.shape[0]):
            p0_err = float(np.linalg.norm(pcd_belt[t, 0] - belt_xyz[t, 0]))
            _require(p0_err <= BELT_POINT0_TOL_M,
                     f"{name} frame {t}: point 0 {p0_err:.2e} m from body 0")
            ds = _wrap(_material_coord(belt_xyz[t], pcd_belt[t]) - s_k, m)
            drift = max(drift, float(np.abs(ds).max()))
            old = _wrap(_material_coord(belt_xyz[t], _arc_length_points(belt_xyz[t])) - old0, m)
            old_drift = max(old_drift, float(np.abs(old).max()))
        _require(drift <= BELT_DRIFT_TOL_BODY_UNITS,
                 f"{name}: material drift {drift:.2e} body units > {BELT_DRIFT_TOL_BODY_UNITS:g}")
        max_drift = max(max_drift, drift)
        wrap = float(np.max(d["sim_wrap_deg"]))
        if stretch >= T5_MIN_STRETCH and wrap > 0.0 and old_drift >= T5_MIN_OLD_DRIFT:
            fixtures.append(f"{name} (stretch {stretch:.0%}, wrap {wrap:.0f} deg, "
                            f"arc-length drift {old_drift:.2f})")
    _require(bool(fixtures), "no episode stretches and wraps the pulley enough to test drift")
    return (f"max material drift {max_drift:.1e} body units (<= {BELT_DRIFT_TOL_BODY_UNITS:g}); "
            f"stretching fixtures: {', '.join(fixtures)}")


@check("T6 lcs_learning loader")
def check_t6(ctx: SimpleNamespace) -> str:
    if not LCS_LEARNING_VENV.is_file():
        raise Skip(f"{LCS_LEARNING_VENV} not found")
    files = [str(path) for _, path in ctx.order]
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(LCS_LEARNING_ROOT)!r})\n"
        "from lcs_learning.dataset_round_belt_tuples import RoundBeltTupleDataset\n"
        f"ds = RoundBeltTupleDataset(data_file={files!r}, num_points=1800, proprio_dim=40, "
        "control_dim=12, point_cloud_source='camera_plus_belt', belt_num_points=150)\n"
        "item = ds[0]\n"
        f"keys = {list(LOADER_SHAPES)!r}\n"
        "shapes = {k: list(item[k].shape) for k in keys}\n"
        "out = {'len': len(ds), 'shapes': shapes, 'u': item['u'].tolist(),\n"
        "       'next_prop': item['next_prop'].tolist()}\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run([str(LCS_LEARNING_VENV), "-c", code], capture_output=True, text=True,
                          timeout=T6_TIMEOUT_S, check=False)
    _require(proc.returncode == 0, f"lcs_learning failed: {proc.stderr.strip()[-800:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    want_len = sum(ctx.t_list[name] - 1 for name, _ in ctx.order)
    _require(out["len"] == want_len, f"loader len {out['len']} != sum(T-1) {want_len}")
    for key, shape in LOADER_SHAPES.items():
        _require(out["shapes"][key] == shape,
                 f"ds[0][{key!r}].shape {out['shapes'][key]} != {shape}")
    first_name, _ = ctx.order[0]
    d0 = ctx.data[first_name]
    u_err = float(np.abs(np.asarray(out["u"]) - d0["actions"][0]).max())
    _require(u_err <= T6_ACTION_TOL, f"ds[0]['u'] off by {u_err:.2e} from {first_name} actions[0]")
    np_err = float(np.abs(np.asarray(out["next_prop"]) - d0["state"][1]).max())
    _require(np_err <= T6_STATE_TOL, f"ds[0]['next_prop'] off by {np_err:.2e} from state[1]")
    return f"len {out['len']}, shapes OK, u/next_prop match {first_name}"


def _jerk_rms_mm(datas: list[dict]) -> float:
    """RMS of ``|u_{t+1} - 2 u_t + u_{t-1}|`` (Franka dxyz) over three consecutive move rows."""
    d2 = []
    for d in datas:
        labels = json.loads(str(d["sim_meta"]))["phase_labels"]
        move = np.array([labels[int(p)].startswith("move") for p in d["sim_phase"]])
        u = np.asarray(d["actions"])[:, 0:3]
        m3 = move[2:] & move[1:-1] & move[:-2]
        d2.append((u[2:] - 2.0 * u[1:-1] + u[:-2])[m3])
    d2 = np.vstack(d2)
    _require(len(d2) > 0, "no three consecutive move rows")
    return float(np.sqrt(np.mean(np.sum(d2 ** 2, axis=1)))) * 1e3


@check("T7 excitation smoothness")
def check_t7(ctx: SimpleNamespace) -> str:
    ou = [ctx.data["run0"], ctx.data["run1"]]
    for d in ou:
        mode = json.loads(str(d["sim_meta"]))["excitation"]["mode"]
        _require(mode == "ou", f"run: excitation mode {mode!r}, expected the ou default")
        # The ramp ends the excitation before the settle: the last rows are unexcited.
        tail = np.abs(np.asarray(d["sim_excite_dpos_m"])[-3:]).max()
        _require(tail == 0.0, f"ou excitation still on at the episode end ({tail:.2e} m)")
    white_dir = ctx.tmp_root / "white"
    args = cli.create_parser().parse_args(
        [*RUN_ARGV, "--excite-mode", "white", "--lcm-url", ctx.lcm_url, "--out", str(white_dir)])
    rows = cli.collect(args)["episodes"]
    _require(all(r["status"] == "ok" for r in rows), f"white run: {rows}")
    white = [_load(white_dir / r["file"]) for r in rows]
    j_ou, j_white = _jerk_rms_mm(ou), _jerk_rms_mm(white)
    _require(j_ou < JERK_RATIO_MAX * j_white,
             f"ou jerk proxy {j_ou:.2f} mm >= {JERK_RATIO_MAX:g} x white {j_white:.2f} mm")
    return (f"Franka action 2nd-difference RMS on move rows: ou {j_ou:.2f} mm vs white "
            f"{j_white:.2f} mm ({j_ou / j_white:.0%}); ou offset 0 at the episode end")


@check("T8 hygiene")
def check_t8(ctx: SimpleNamespace) -> str:
    left = subprocess.run(["pgrep", "-f", PRIVATE_PORT], capture_output=True, text=True,
                          check=False).stdout
    _require(not left.strip(), f"processes with {PRIVATE_PORT} in cmdline: pids {left.split()}")
    log_text = ctx.run_log.read_text() if ctx.run_log.is_file() else ""
    errors = [e for e in cli.OSC_LOG_ERRORS if e in log_text]
    _require(not errors, f"{ctx.run_log}: log has {errors}")
    gpu = _gpu_apps()
    if gpu is not None and ctx.gpu_baseline is not None:
        extra = gpu - ctx.gpu_baseline - {os.getpid()}
        _require(not extra, f"foreign GPU compute apps appeared: {sorted(extra)}")
    return f"pgrep -f {PRIVATE_PORT} empty, osc.log clean, GPU apps at baseline"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the temp dir")
    parser.add_argument("--lcm-url", default=PRIVATE_LCM_URL)
    args = parser.parse_args()

    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="WARNING", format="{level: <7} | {message}")

    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_lcs_tuples_"))
    ctx = SimpleNamespace(tmp_root=tmp_root, lcm_url=args.lcm_url, gpu_baseline=_gpu_apps())
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
        print(f"ALL LCS TUPLE CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
