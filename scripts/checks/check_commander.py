#!/usr/bin/env python3
"""Headless check of the emulated magna waypoint commander (``round_belt_task.commander``).

No sim, no LCM socket. N0 Franka position knots, N1 orientation knots, N2 reach/latch/dwell state
machine, N3 UR line + regeneration + tool0 frame, N4 bounded excitation, N5 the LCM message (and,
if a recorded magna run has target messages, its layout), N6 OU vs white excitation statistics.

Run:
    uv run python scripts/checks/check_commander.py
"""

from __future__ import annotations

import glob
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from round_belt_task import commander as cm

TARGET_CHANNEL = "TARGET_CARTESIAN_POSE_TRAJECTORY"
RECORDING_GLOBS = ("data/recordings/*/targets.jsonl", "recordings/*/targets.jsonl")
MAGNA_UR10_GLOBS = (
    "/home/hienbui/git/magna/bazel-bin/external/drake-ur-driver+/models/ur10.urdf",
    "/home/hienbui/.cache/bazel/_bazel_hienbui/*/external/drake-ur-driver+/models/ur10.urdf",
)
P = cm.CommanderParams()
MEAS_POS = np.array([0.5, 0.1, 0.3])
MEAS_QUAT = cm.quat_axis_angle([0.3, -0.2, 1.0], 2.5)  # wxyz


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _close(a, b, tol: float, what: str) -> None:
    err = float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, np.float64))))
    _require(err <= tol, f"{what}: max err {err:.3e} > {tol:.0e}")


def _qz(angle: float) -> np.ndarray:
    return cm.quat_axis_angle([0.0, 0.0, 1.0], angle)


def _target(label: str, pos, quat=MEAS_QUAT, hand_mm=None, dwell_s=0.0) -> cm.FrankaTarget:
    return cm.FrankaTarget(label, np.asarray(pos, dtype=np.float64), np.asarray(quat, float),
                           hand_mm, dwell_s)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


@check("N0")
def n0_position_knots() -> str:
    t = 10.0
    cmd = cm.FrankaWaypointCommander([_target("a", MEAS_POS + [0.06, 0, 0])]).tick(
        t, MEAS_POS, MEAS_QUAT, 0.0, 0.0)
    i = np.arange(7)
    _close(cmd.knots_pos, MEAS_POS + np.outer(i * 0.006, [1, 0, 0]), 1e-12, "60 mm knots")
    _close(cmd.times, t + 0.075 * i, 1e-12, "times")
    _require(cmd.phase == "move:a" and not cmd.hold, f"phase {cmd.phase}")

    # 5 mm away but the UR is off: not reached, all 7 knots snap to the target.
    near = MEAS_POS + [0.0, 0.005, 0.0]
    cmd = cm.FrankaWaypointCommander([_target("b", near)]).tick(t, MEAS_POS, MEAS_QUAT, 0.01, 0)
    _close(cmd.knots_pos, np.tile(near, (7, 1)), 1e-15, "5 mm snap")
    _require(len(cmd.times) == 7 and not cmd.hold, "5 mm target must stay a 7-knot move")

    fc = cm.FrankaWaypointCommander([_target("c", MEAS_POS + [0.0, 0.0, 0.004])])
    cmd = fc.tick(t, MEAS_POS, MEAS_QUAT, 0.010, 0.0)
    _require(not fc.latched and cmd.phase == "move:c", "4 mm target with UR 10 mm off latched")
    cmd = fc.tick(t + 0.075, MEAS_POS, MEAS_QUAT, 0.0, 0.0)
    _require(fc.latched and cmd.phase == "done", f"4 mm target, UR on target: {cmd.phase}")
    return "60 mm -> 6 mm/knot, 7 knots at 0.075 s; 5 mm snaps; UR 10 mm off blocks the reach"


@check("N1")
def n1_orientation_knots() -> str:
    q_t = cm._qmul(_qz(0.3), MEAS_QUAT)
    far = MEAS_POS + [0.1, 0.0, 0.0]
    cmd = cm.FrankaWaypointCommander([_target("r", far, q_t)]).tick(1.0, MEAS_POS, MEAS_QUAT,
                                                                     0.0, 0.0)
    angles = 0.3 * np.minimum(0.075 * np.arange(7) / 0.6, 1.0)
    _close(angles[:2], [0.0, 0.0375], 1e-15, "expected angles")
    got = [cm.angular_distance(MEAS_QUAT, q) for q in cmd.knots_quat]
    _close(got, angles, 1e-9, "slerp angles")
    expect = np.array([cm._qmul(_qz(a), MEAS_QUAT) for a in angles])
    _close(cmd.knots_quat, expect, 1e-9, "knots = Rz(s*0.3) q_meas")

    cmd_neg = cm.FrankaWaypointCommander([_target("r", far, -q_t)]).tick(1.0, MEAS_POS,
                                                                         MEAS_QUAT, 0.0, 0.0)
    _close(cmd_neg.knots_quat, cmd.knots_quat, 1e-12, "antipodal target")
    _require(all(np.dot(q, MEAS_QUAT) > 0 for q in cmd_neg.knots_quat), "sign flip not applied")
    return "0.3 rad about z: s_i = min(0.075 i/0.6, 1), angles 0, 0.0375, ...; -q handled"


@check("N2")
def n2_state_machine() -> str:
    a, b, c = MEAS_POS, MEAS_POS + [0.05, 0, 0], MEAS_POS + [0.05, 0.05, 0]
    fc = cm.FrankaWaypointCommander([_target("A", a, hand_mm=40.0),
                                     _target("B", b, hand_mm=0.0, dwell_s=1.0),
                                     _target("C", c, dwell_s=0.5)])
    cmd = fc.tick(1.0, a, MEAS_QUAT, 0.0, 0.0)
    _require(cmd.phase == "move:B" and cmd.target_index == 1 and cmd.latched_index == 0,
             f"dwell 0 must advance on the reach tick: {cmd.phase} {cmd.latched_index}")
    _close(cmd.knots_pos[1] - cmd.knots_pos[0], [0.006, 0, 0], 1e-12, "knots head to B")
    _require(cmd.hand_mm == 40.0, f"hand {cmd.hand_mm}")

    at_b = b + [0.001, 0.0, 0.0]
    cmd = fc.tick(1.5, at_b, MEAS_QUAT, 0.01, 0.0)
    _require(cmd.phase == "move:B" and cmd.latched_index is None, "UR off must block B")
    cmd = fc.tick(2.0, at_b, MEAS_QUAT, 0.0, 0.0)
    _require(cmd.phase == "hold:B" and cmd.hold and cmd.latched_index == 1, cmd.phase)
    _require(cmd.hand_mm == 0.0, f"hand after latching B: {cmd.hand_mm}")
    for t, meas in ((2.0, at_b), (2.5, b + [0, 0.002, 0]), (2.9999, b)):
        cmd = fc.tick(t, meas, MEAS_QUAT, 0.0, 0.0)
        _require(cmd.phase == "hold:B" and len(cmd.times) == 2, f"t={t}: {cmd.phase}")
        _close(cmd.knots_pos, [at_b, at_b], 0.0, f"t={t} hold = latched measured pose")
        _close(cmd.times, [t, t + 0.075], 1e-15, "hold times")
    cmd = fc.tick(3.0, b, MEAS_QUAT, 0.0, 0.0)
    _require(cmd.phase == "move:C" and len(cmd.times) == 7, f"t - reached = 1.0: {cmd.phase}")

    cmd = fc.tick(4.0, c, MEAS_QUAT, 0.0, 0.0)
    _require(cmd.phase == "hold:C" and cmd.hand_mm == 0.0, f"{cmd.phase} hand {cmd.hand_mm}")
    cmd = fc.tick(4.4, c, MEAS_QUAT, 0.0, 0.0)
    _require(cmd.phase == "hold:C", cmd.phase)
    for t in (4.5, 6.0):
        cmd = fc.tick(t, c + [0, 0, 0.01], MEAS_QUAT, 0.0, 0.0)
        _require(cmd.phase == "done" and cmd.hold and cmd.target_index == 2, cmd.phase)
        _close(cmd.knots_pos, [c, c], 0.0, "last target keeps its latched pose")
    return "dwell 0 advances on the reach tick; dwell 1.0 holds until t - reached >= 1.0; " \
           "last -> hold -> done; hand changes only on latch"


def _find_magna_ur10() -> Path | None:
    for pattern in MAGNA_UR10_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    return None


def _translated(m: np.ndarray, d) -> np.ndarray:
    out = m.copy()
    out[:3, 3] += d
    return out


@check("N3")
def n3_ur_line() -> str:
    X_tt = cm.x_tool0_tracking()
    magna_urdf = _find_magna_ur10()
    if magna_urdf is not None:
        _close(cm.x_tool0_tracking(magna_urdf), X_tt, 1e-15, "repo vs magna ur10.urdf tool0")
    meas = cm.pose_mat([0.6, 0.4, 0.5], cm.quat_axis_angle([1.0, 0.5, -0.3], 2.0))
    t0 = 5.0

    def target(m, byte=None):
        return cm.UrTarget(m[:3, 3].copy(), cm.mat3_to_quat(m[:3, :3]), byte)

    urc = cm.UrLineCommander([None], P, X_tt)
    p_t, q_t = urc.to_tool0(meas)
    _close(cm.pose_mat(p_t, q_t) @ X_tt, meas, 1e-12, "tool0/tracking round trip")

    far = _translated(meas, [0.08, 0.0, 0.0])
    cmd = urc.tick(t0, meas, target(far))
    _require(cmd.regenerated, "first tick must create a line")
    _close(cmd.line.t1 - cmd.line.t0, 1.0, 1e-12, "80 mm duration")
    _close(cmd.pose_4x4, meas, 1e-12, "line start = measured")
    _close(cmd.pose_at(t0 + 0.5), _translated(meas, [0.04, 0, 0]), 1e-12, "lerp at mid")
    _close(cmd.pose_at(t0 + 7.0), far, 1e-12, "clamped at t1")
    _close(cmd.pose_at(t0 - 1.0), meas, 1e-12, "clamped at t0")
    dxyz, drot = cm.line_action(cmd, t0 + 0.1, 0.075)
    _close(dxyz, [0.006, 0, 0], 1e-12, "line_action position")
    _close(drot, np.zeros(3), 1e-12, "line_action rotation")

    on_line = cmd.pose_at(t0 + 0.5)
    _require(not urc.tick(t0 + 0.5, on_line, None).regenerated, "on the line: regenerated")
    _require(not urc.tick(t0 + 0.6, on_line, target(far)).regenerated, "same target: regen")
    lagging = cmd.pose_at(t0 + 0.9)
    _require(urc.tick(t0 + 1.0, lagging, target(far)).regenerated, "t >= t1 must regenerate")
    moved = _translated(far, [0.0, 0.001, 0.0])
    cmd = urc.tick(t0 + 1.1, lagging, target(moved))
    _require(cmd.regenerated, "1 mm target move must regenerate")
    near = _translated(moved, [0.0, 0.0, 0.003])
    _require(not urc.tick(t0 + 1.2, near, target(moved)).regenerated, "within 5 mm: regen")
    _require(not urc.tick(t0 + 9.0, near, target(moved)).regenerated, "within 5 mm after t1")

    urc = cm.UrLineCommander([None], P, X_tt)
    short = _translated(meas, [0.0, 0.02, 0.0])
    cmd = urc.tick(t0, meas, target(short))
    _close(cmd.line.t1 - cmd.line.t0, 0.5, 1e-12, "20 mm -> min duration")
    # rotate tool0 in place (rotating the tracking frame would also swing tool0's origin)
    rot = cm.pose_mat(p_t + [0.0, 0.02, 0.0], cm._qmul(_qz(0.6), q_t)) @ X_tt
    cmd = cm.UrLineCommander([None], P, X_tt).tick(t0, meas, target(rot))
    _close(cmd.line.t1 - cmd.line.t0, 1.2, 1e-12, "0.6 rad -> 1.2 s")

    urc = cm.UrLineCommander([None], P, X_tt)
    cmd = urc.tick(t0, meas, target(far))
    mid = cmd.pose_at(t0 + 0.5)
    _require(not urc.tick(t0 + 0.5, _translated(mid, [0, 0.02, 0]), None).regenerated,
             "2 cm deviation regenerated")
    _require(urc.tick(t0 + 0.5, _translated(mid, [0, 0.03, 0]), None).regenerated,
             "3 cm deviation did not regenerate")

    urc = cm.UrLineCommander([target(far, 200), target(far)], P, X_tt, byte=10)
    urc.on_latch(1)
    _require(urc.byte == 10, "a None byte must keep the byte in force")
    urc.on_latch(0)
    _require(urc.byte == 200 and urc.tick(t0, meas, None).byte == 200, "byte on latch")
    src = "magna ur10.urdf" if magna_urdf is not None else "repo ur10.urdf (magna's not found)"
    return f"80 mm -> 1.0 s, 20 mm -> 0.5 s, lerp/clamp, regen rules, tool0 via {src}"


@check("N4")
def n4_excitation() -> str:
    cap = 1.5
    lin_cap, ang_cap = 0.08 * 0.075 * cap, 0.5 * 0.075 * cap
    far = MEAS_POS + [0.06, 0.0, 0.0]
    base = cm.FrankaWaypointCommander([_target("x", far)]).tick(1.0, MEAS_POS, MEAS_QUAT, 0, 0)
    ex = cm.Excite(np.array([0.0, 0.03, 0.0]), np.array([1.0, 0, 0]), 0.0, cap, 0.002)
    cmd = cm.FrankaWaypointCommander([_target("x", far)]).tick(1.0, MEAS_POS, MEAS_QUAT, 0, 0,
                                                                excite=ex)
    _close(np.linalg.norm(cmd.knots_pos[1] - cmd.knots_pos[0]), lin_cap, 1e-12, "moving cap")
    _close(cmd.knots_pos[0], base.knots_pos[0], 0.0, "knot 0 moved")
    _close(cmd.knots_quat[0], base.knots_quat[0], 0.0, "knot 0 rotated")
    _close(cmd.knots_pos[1:] - base.knots_pos[1:],
           np.tile(cmd.excite_applied.dpos, (6, 1)), 1e-15, "offset on knots 1..6")
    dxyz, _ = cm.knot_action(cmd)
    _close(dxyz, cmd.knots_pos[1] - cmd.knots_pos[0], 1e-12, "knot_action")

    held = cm.FrankaWaypointCommander([_target("h", MEAS_POS)])
    held.tick(1.0, MEAS_POS, MEAS_QUAT, 0, 0)
    cmd = held.tick(1.1, MEAS_POS, MEAS_QUAT, 0, 0, excite=ex)
    _close(np.linalg.norm(cmd.excite_applied.dpos), lin_cap, 1e-12, "30 mm capped in hold")

    down = cm.Excite(np.array([0.0, 0.0, -0.03]), np.array([0, 0, 1.0]), 0.0, 10.0, 0.002)
    cmd = cm.FrankaWaypointCommander([_target("x", far)]).tick(1.0, MEAS_POS, MEAS_QUAT, 0, 0,
                                                                excite=down)
    _require(bool(np.all(cmd.knots_pos[:, 2] >= base.knots_pos[:, 2] - 0.002 - 1e-15)),
             "z floor violated")
    _close(cmd.excite_applied.dpos, [0, 0, -0.002], 1e-15, "z floor offset")

    q_t = cm._qmul(_qz(0.3), MEAS_QUAT)
    rbase = cm.FrankaWaypointCommander([_target("r", far, q_t)])
    rex = cm.Excite(np.zeros(3), np.array([1.0, 0, 0]), 0.5, cap, 0.002)
    cmd = rbase.tick(1.0, MEAS_POS, MEAS_QUAT, 0, 0, excite=rex)
    step = cm.angular_distance(cmd.knots_quat[0], cmd.knots_quat[1])
    _require(step <= ang_cap + 1e-12 and step > ang_cap - 1e-9, f"rot cap: {step} vs {ang_cap}")
    _require(0.0 < cmd.excite_applied.angle < 0.5, "rotation not scaled down")
    _, drot = cm.knot_action(cmd)
    _close(np.linalg.norm(drot), step, 1e-12, "knot_action rotation")

    small = cm.Excite(np.array([0.0, 0.001, 0.0]), np.array([0, 1.0, 0]), 0.01, cap, 0.002)
    cmd = cm.FrankaWaypointCommander([_target("x", far)]).tick(1.0, MEAS_POS, MEAS_QUAT, 0, 0,
                                                                excite=small)
    _close(cmd.excite_applied.dpos, small.dpos, 0.0, "small offset untouched")
    _require(cmd.excite_applied.angle == 0.01, "small rotation untouched")

    rng = np.random.default_rng(0)
    for _ in range(200):
        d = cm.draw_excite(rng, 0.004, 0.05, cap, 0.002)
        _require(np.linalg.norm(d.dpos) <= 0.004 and 0.0 <= d.angle <= 0.05
                 and abs(np.linalg.norm(d.axis) - 1.0) < 1e-12, "draw_excite out of range")
    return f"30 mm -> {lin_cap * 1e3:.1f} mm step, z floor, knot 0 fixed, rot cap " \
           f"{ang_cap:.4f} rad, knot_action = knot1 - knot0"


def _recorded_target_message() -> tuple[Path, dict] | None:
    for pattern in RECORDING_GLOBS:
        for path in sorted(glob.glob(str(REPO_ROOT / pattern))):
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    item = json.loads(line)
                    if item.get("channel") != TARGET_CHANNEL:
                        continue
                    blocks = item["payload"].get("blocks", {})
                    pos = blocks.get(cm.BLOCK_NAMES[0])
                    if pos is not None and len(pos["t"]) == 7:
                        return Path(path), item["payload"]
    return None


@check("N5")
def n5_message() -> str:
    fc = cm.FrankaWaypointCommander([_target("m", MEAS_POS + [0.06, 0, 0],
                                             cm._qmul(_qz(0.3), MEAS_QUAT))])
    cmd = fc.tick(12.3, MEAS_POS, MEAS_QUAT, 0, 0)
    msg = cm.saved_traj_message(12_300_000, cmd.knots_pos, cmd.knots_quat, cmd.times)
    back = cm.lcmt_timestamped_saved_traj.decode(msg.encode())
    _require(back.utime == msg.utime == 12_300_000, f"utime {back.utime}")
    traj = back.saved_traj
    _require(traj.num_trajectories == 3 and tuple(traj.trajectory_names) == cm.BLOCK_NAMES
             and tuple(b.trajectory_name for b in traj.trajectories) == cm.BLOCK_NAMES,
             f"block order {traj.trajectory_names}")
    _require(traj.metadata.name == "lcs_collector", "metadata name")
    for block, rows in zip(traj.trajectories, (3, 4, 3)):
        data = np.asarray(block.datapoints)
        _require(data.shape == (rows, 7) and block.num_points == 7
                 and block.num_datatypes == rows and list(block.datatypes) == ["double"] * rows,
                 f"{block.trajectory_name}: shape {data.shape}")
        _close(block.time_vec, cmd.times, 0.0, f"{block.trajectory_name} time_vec")
    _close(traj.trajectories[2].datapoints, np.zeros((3, 7)), 0.0, "force block")
    pos, quat, times = cm.parse_saved_traj_message(back)
    _close(pos, cmd.knots_pos, 1e-12, "parsed positions")
    _close(quat, cmd.knots_quat, 1e-12, "parsed wxyz")
    _close(times, cmd.times, 1e-12, "parsed times")
    for bad_utime, bad_times in ((0, cmd.times), (1, cmd.times - cmd.times[0])):
        try:
            cm.saved_traj_message(bad_utime, cmd.knots_pos, cmd.knots_quat, bad_times)
        except ValueError:
            continue
        raise AssertionError("utime 0 / time_vec[0] == 0 accepted")

    found = _recorded_target_message()
    if found is None:
        print("[SKIP] N5 recorded magna target message: no 7-knot "
              f"{TARGET_CHANNEL} under {' / '.join(RECORDING_GLOBS)}")
        return "3 blocks 3x7 / 4x7 / 3x7 round-trip through encode/decode; recorded half skipped"
    path, payload = found
    blocks = payload["blocks"]
    _require(tuple(blocks) == cm.BLOCK_NAMES, f"{path}: block names {tuple(blocks)}")
    for name, rows in zip(cm.BLOCK_NAMES, (3, 4, 3)):
        _require(blocks[name]["datatypes"] == ["double"] * rows, f"{path}: {name} datatypes")
        _require(np.asarray(blocks[name]["data"]).shape == (rows, 7), f"{path}: {name} shape")
        _close(np.diff(blocks[name]["t"]), np.full(6, 0.075), 1e-9, f"{path}: {name} spacing")
    return f"round trip OK; recorded {path.parent.name} matches names/datatypes/7 knots/0.075 s"


def _lag1(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0)
    return (x[1:] * x[:-1]).sum(axis=0) / (x * x).sum(axis=0)


@check("N6")
def n6_ou_excitation() -> str:
    n, dt, tau, sig_p, sig_r = 200_000, 0.075, 0.4, 0.002, np.radians(1.0)
    a = np.exp(-dt / tau)
    ou = cm.OuExcitation(np.random.default_rng(5), sig_p, sig_r, tau, dt, 2.0, 0.002)
    first = ou.excite()
    _require(not first.dpos.any() and first.angle == 0.0, "OU must start at 0")
    x = np.empty((n, 6))
    for k in range(n):
        ex = ou.step()
        x[k] = np.concatenate([ex.dpos, ex.axis * ex.angle])
    sigma = np.array([sig_p] * 3 + [sig_r] * 3)
    std, r1 = x.std(axis=0) / sigma, _lag1(x)
    dstd = np.diff(x, axis=0).std(axis=0) / (sigma * np.sqrt(2.0 * (1.0 - a)))
    # n(1-a)/(1+a) ~ 19k independent samples: std error ~0.5 %, lag-1 error ~0.002.
    _require(bool(np.all(np.abs(std - 1.0) < 0.03)), f"stationary std / sigma {std.round(3)}")
    _require(bool(np.all(np.abs(r1 - a) < 0.01)), f"lag-1 {r1.round(4)} vs a {a:.4f}")
    _require(bool(np.all(np.abs(dstd - 1.0) < 0.03)), f"step std ratio {dstd.round(3)}")

    again = cm.OuExcitation(np.random.default_rng(5), sig_p, sig_r, tau, dt, 2.0, 0.002)
    _close(again.step().dpos, x[0, :3], 0.0, "same seed, same draw")

    held = ou.x.copy()
    ramp = [ou.fade(100.0 + 0.005 * k) for k in range(round(cm.EXCITE_RAMP_S / 0.005) + 2)]
    _close(ramp[0].dpos, held[:3], 1e-15, "fade starts at the held offset")
    _close(ramp[1].dpos, held[:3] * (1.0 - 0.005 / cm.EXCITE_RAMP_S), 1e-12, "linear ramp")
    _require(ramp[-1] is None and ramp[-2] is None, "fade must end at None after ramp_s")
    _close(ou.x, held, 0.0, "fade draws no noise")

    rng = np.random.default_rng(6)
    w = np.empty((20_000, 6))
    for k in range(len(w)):
        ex = cm.draw_excite(rng, 0.004, np.radians(2.0), 2.0, 0.002)
        w[k] = np.concatenate([ex.dpos, ex.axis * ex.angle])
    rw = _lag1(w)
    _require(bool(np.all(np.abs(rw) < 0.03)), f"white lag-1 {rw.round(4)}")
    return (f"OU std/sigma {std.min():.3f}..{std.max():.3f}, lag-1 {r1.min():.3f}..{r1.max():.3f} "
            f"(a {a:.3f}), step std ratio {dstd.min():.3f}..{dstd.max():.3f}; ramp 0.3 s; white "
            f"lag-1 max |{np.abs(rw).max():.3f}|")


def main() -> int:
    t0 = time.perf_counter()
    exit_code = 0
    for name, fn in CHECKS:
        try:
            detail = fn()
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            exit_code = 1
            break
        except Exception as exc:  # noqa: BLE001 - report the crash as a failure
            print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
            traceback.print_exc()
            exit_code = 1
            break
        print(f"[PASS] {name}: {detail}")
    runtime = time.perf_counter() - t0
    if exit_code == 0:
        print(f"ALL COMMANDER CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
