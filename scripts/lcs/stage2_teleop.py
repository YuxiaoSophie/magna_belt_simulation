#!/usr/bin/env python3
"""SpaceMouse teleop phase of the whole-run chain (``eval_learned_mpc.py --stage2-teleop``).

After the scripted UR exit the operator drives the Franka: ``SPACE_MOUSE_FRANKA_STATE`` (the
spacemouse driver, on the harness's private LCM URL) -> :class:`TeleopHook` (world-frame velocity
integration on SIM time, tip floor / tilt cap / lead clamp vs the measured pose) -> a 7-knot
``TARGET_CARTESIAN_POSE_TRAJECTORY`` every step -> magna's OSC (``--input_mode=1``, unchanged).
Frames are sampled on the episode grid with ``sim_cmd_source`` 9; their knots are
``bridge.last_traj`` = the trajectory published with the sampled state (the first frame's is the
exit's 2-knot hold). Terminal keys: ``d`` done (release runs), ``a`` abort (episode discarded),
``p`` pause. No tty: the phase ends at ``--teleop-max-s`` (= done).
"""

from __future__ import annotations

import math
import os
import select
import shutil
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Self

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_learned_mpc as harness
import newton
from loguru import logger
from stage2_collect import cap_tilt

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from round_belt_task.arm_kinematics import UrTracking
from round_belt_task.commander import (
    CommanderParams,
    _qmul,
    angular_distance,
    parse_saved_traj_message,
    saved_traj_message,
    slerp,
)
from round_belt_task.episode_io import pose7_mat
from round_belt_task.offline_simulation import held_for
from spacemouse import lcmt_spacemouse_state

CHANNEL = "SPACE_MOUSE_FRANKA_STATE"
KNOT_DT = CommanderParams().dt
STATUS_EVERY_S = 0.5
PAUSE_POLL_S = 0.05
KEYS = {"d": "done (release runs)", "a": "abort (episode discarded)", "p": "pause / resume"}


class SpaceMouseInput:
    """Latest ``lcmt_spacemouse_state`` on ``bridge.lc``; axes in [-1, 1] scaled to m/s, rad/s."""

    def __init__(self, bridge, channel: str = CHANNEL, lin_mps: float = 0.05,
                 ang_rps: float = 0.1666) -> None:
        self.lin, self.ang = float(lin_mps), float(ang_rps)
        self.linear, self.angular = np.zeros(3), np.zeros(3)
        self.utime, self.wall, self.n_msgs = -1, None, 0
        self._lc = bridge.lc
        self._sub = self._lc.subscribe(channel, self._on)

    def _on(self, channel: str, data: bytes) -> None:
        try:
            msg = lcmt_spacemouse_state.decode(data)
        except ValueError:
            return
        # Driver output is untrusted: non-finite -> 0, clipped to the documented [-1, 1].
        self.linear = np.clip(np.nan_to_num(np.asarray(msg.linear, np.float64)), -1.0, 1.0)
        self.angular = np.clip(np.nan_to_num(np.asarray(msg.angular, np.float64)), -1.0, 1.0)
        self.utime, self.wall = int(msg.utime), time.perf_counter()
        self.n_msgs += 1

    def velocity(self, now_wall: float, stale_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """``(v m/s, w rad/s)`` world frame; zero without a message newer than ``stale_s``."""
        if self.wall is None or now_wall - self.wall > stale_s:
            return np.zeros(3), np.zeros(3)
        return self.linear * self.lin, self.angular * self.ang

    def close(self) -> None:
        self._lc.unsubscribe(self._sub)


def integrate(target7, v, w, dt: float) -> np.ndarray:
    """World frame: ``p += v dt``, ``R = exp(w dt) R``; ``xyz`` + ``wxyz`` (``w >= 0``)."""
    out = np.array(target7, dtype=np.float64)
    out[:3] += np.asarray(v, np.float64) * dt
    rv = np.asarray(w, np.float64) * dt
    a = float(np.linalg.norm(rv))
    if a > 0.0:
        q = _qmul(np.concatenate([[math.cos(0.5 * a)], math.sin(0.5 * a) * rv / a]), out[3:])
        out[3:] = q / np.linalg.norm(q)
    if out[3] < 0.0:
        out[3:] *= -1.0
    return out


def clamp(target7, measured7, plate_top_z: float, floor_m: float, tilt_deg: float,
          lead_m: float = 0.02, lead_deg: float = 15.0) -> np.ndarray:
    """Tip floor, tilt cap, then the lead vs the measured pose cut to ``lead_m`` / ``lead_deg``
    (anti-runaway when the arm is blocked)."""
    out = np.array(target7, dtype=np.float64)
    meas = np.asarray(measured7, np.float64)
    out[2] = max(out[2], plate_top_z + floor_m)
    out[3:] = cap_tilt(out[3:], tilt_deg)
    d = out[:3] - meas[:3]
    n = float(np.linalg.norm(d))
    if n > lead_m:
        out[:3] = meas[:3] + d * (lead_m / n)
    ang, cap = angular_distance(meas[3:], out[3:]), math.radians(lead_deg)
    if ang > cap:
        q = slerp(meas[3:], out[3:], cap / ang)
        out[3:] = q / np.linalg.norm(q)
    if out[3] < 0.0:
        out[3:] *= -1.0
    return out


class TeleopHook:
    """``commander_hook``: target latched at the measured pose on the first call, integrated with
    the SpaceMouse velocity every step (``sim.frame_dt``), clamped; 7 knots ``k * KNOT_DT`` ahead
    (each clamped). ``paused[0]``: hold the current target."""

    def __init__(self, sim, inp, limits: dict, plate_top_z: float, paused: list[bool]) -> None:
        self.sim, self.inp, self.limits, self.paused = sim, inp, limits, paused
        self.plate_top_z = float(plate_top_z)
        self.target: np.ndarray | None = None
        self.last_knots = self.last_times = None
        self.last_utime = -1
        self.v, self.w = np.zeros(3), np.zeros(3)
        self.n_clamped, self.clamped = 0, False

    def _clamp(self, pose, meas) -> np.ndarray:
        L = self.limits
        return clamp(pose, meas, self.plate_top_z, L["floor_m"], L["tilt_deg"], L["lead_m"],
                     L["lead_deg"])

    def __call__(self, step, t, joint_q, body_q):
        meas = self.sim.franka_measured_pose7(joint_q)
        if self.target is None:
            self.target = meas
        if self.paused[0]:
            v, w = np.zeros(3), np.zeros(3)
        else:
            v, w = self.inp.velocity(time.perf_counter(), self.limits["stale_s"])
        self.v, self.w = v, w
        free = integrate(self.target, v, w, self.sim.frame_dt)
        self.target = self._clamp(free, meas)
        self.clamped = bool(np.abs(self.target - free).max() > 1e-9)
        self.n_clamped += self.clamped
        knots = np.stack([self._clamp(integrate(self.target, v, w, k * KNOT_DT), meas)
                          for k in range(harness.N_KNOTS)])
        self.last_knots, self.last_times = knots, t + KNOT_DT * np.arange(harness.N_KNOTS)
        self.last_utime = round(t * 1e6)
        return saved_traj_message(self.last_utime, knots[:, :3], knots[:, 3:], self.last_times)


class Keys:
    """Non-blocking single keys from a tty stdin (cbreak while inside); no tty -> never a key.
    ``cmd_file``: its first char is a key too (the file is emptied once read; the GUI writes it)."""

    def __init__(self, cmd_file: Path | None = None) -> None:
        self.cmd_file = cmd_file

    def __enter__(self) -> Self:
        self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        if self._fd is not None:
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def pop(self) -> str | None:
        if self.cmd_file is not None and Path(self.cmd_file).is_file():
            ch = Path(self.cmd_file).read_text()[:1].lower()
            if ch:
                open(self.cmd_file, "w").close()
                return ch
        if self._fd is None or not select.select([self._fd], [], [], 0)[0]:
            return None
        return os.read(self._fd, 1).decode(errors="ignore").lower() or None


def _cmd_fields(msg, t: float) -> dict:
    """Frame command fields from a published trajectory, padded to ``N_KNOTS``."""
    pos, quat, times = parse_saved_traj_message(msg)
    knots, n, N = np.hstack([pos, quat]), len(times), harness.N_KNOTS
    if n < N:
        knots = np.vstack([knots, np.repeat(knots[-1:], N - n, 0)])
        times = np.concatenate([times, np.repeat(times[-1:], N - n)])
    return {"knots": knots[:N].copy(), "knot_times": times[:N].copy(), "knot0": knots[0].copy(),
            "knot1": knots[1].copy(), "traj_utime": int(msg.utime), "n_knots": n}


def run(sim, ctx, sampler, last: dict, timing: dict, args, phase_label: str = "s2:teleop",
        arms: str = "franka") -> dict:
    """The teleop phase from the current state (on the episode grid of ``last``):
    ``{frames, report, done}``; ``done`` False -> no release. ``arms`` must stay held."""
    n, dt = harness.SAMPLE_STEPS, sim.frame_dt
    s0 = last["sim_step"] - last["step"]
    limits = ctx.chain["limits"]
    phase = ctx.phase_labels.index(phase_label)
    ur = pose7_mat(UrTracking.fk(sim.arm_targets()[1]))  # UR parked, u = 0 as in stage 2
    det = timing.setdefault("teleop_detail", {"render": 0.0, "encode": 0.0})
    render = not isinstance(sim.viewer, newton.viewer.ViewerNull)
    frames: list[dict] = []

    def sample() -> None:
        fr = harness._sample(sim, ctx, sampler, sim.step_index - s0, s0, publish=False, timing=det)
        fr.update(_cmd_fields(sim.bridge.last_traj, fr["time"]), ur_t=ur.copy(), ur_t1=ur.copy(),
                  cmd_source=harness.CMD_TELEOP, ur_line=np.full(16, np.nan),
                  ur_fresh_exact=False, ur_own_delta=np.full(6, np.nan), post_phase="teleop",
                  phase=phase, stage=0, ctrl_hand_mm=math.nan, ctrl_robotiq=-1,
                  latent_pred=np.full(len(fr["latent"]), np.nan), goal_dist=np.full(1, np.nan))
        frames.append(fr)

    inp = SpaceMouseInput(sim.bridge, args.teleop_channel, limits["lin_mps"], limits["ang_rps"])
    paused = [False]
    hook = TeleopHook(sim, inp, limits, ctx.plate_top_z, paused)
    operator = abort = None
    k0, w0 = sim.step_index, time.perf_counter()
    active_s = 0.0
    st = {"wall": w0, "step": k0, "active": 0.0, "msgs": 0}
    try:
        with Keys(args.teleop_cmd_file) as keys:
            print(f"[TELEOP] keys: {', '.join(f'{k} {v}' for k, v in KEYS.items())}; max "
                  f"{args.teleop_max_s:g} s sim; viewer "
                  f"{f'http://localhost:{args.viewer_port}' if render else 'off'}", flush=True)
            end = time.perf_counter() + args.teleop_wait_s
            while inp.n_msgs == 0 and time.perf_counter() < end:
                sim.bridge.poll()
                time.sleep(0.01)
            if inp.n_msgs == 0:
                logger.warning(f"[TELEOP] no {args.teleop_channel} message in "
                               f"{args.teleop_wait_s:g} s: holding until one arrives")
            sim.commander_hook = hook
            sample()
            while True:
                if (sim.step_index - s0) % n == 0:  # end only on the sample grid
                    if abort is None and not held_for(frames[-1]["grasp_ok"], arms):
                        abort = (f"grasp lost at step {sim.step_index - s0} "
                                 f"({sim.grasp_state().describe()})")
                    if operator is None and (sim.step_index - k0) * dt >= args.teleop_max_s:
                        operator = "timeout"
                    if abort is not None or operator is not None:
                        break
                key = keys.pop()
                if key in ("d", "a") and operator is None:
                    operator, paused[0] = ("done" if key == "d" else "abort"), False
                elif key == "p" and operator is None:
                    paused[0] = not paused[0]
                now = time.perf_counter()
                if now - st["wall"] >= STATUS_EVERY_S:
                    _status(sim, ctx, frames[-1], hook, inp, st, now, k0, paused[0])
                if paused[0]:
                    sim.bridge.poll()
                    time.sleep(PAUSE_POLL_S)
                    continue
                if not sim.osc.alive():
                    abort = f"OSC exited at step {sim.step_index - s0}"
                    break
                t_step = time.perf_counter()
                sim.control_step()
                sim.bridge.poll()
                if (sim.step_index - s0) % n == 0:
                    sample()
                if render and sim.step_index % args.render_every == 0:
                    sim.render()
                spent = time.perf_counter() - t_step
                if spent < args.pace * dt and ctx.pacer is None:  # --realtime: paced in the step
                    time.sleep(args.pace * dt - spent)
                step_s = time.perf_counter() - t_step
                active_s += step_s
                st["active"] += step_s
    finally:
        sim.commander_hook = sim.make_hold_hook()
        inp.close()
        print(flush=True)
    sim_s = (sim.step_index - k0) * dt
    lab = harness._label_row(ctx, frames[-1])
    done = abort is None and operator in ("done", "timeout")
    report = {"operator": operator, "abort": abort, "duration_s": sim_s,
              "wall_s": time.perf_counter() - w0, "frames": len(frames),
              "n_clamped": hook.n_clamped, "spacemouse_msgs": inp.n_msgs,
              "realtime_factor": sim_s / active_s if active_s > 0 else None, "limits": limits,
              "large_label_end": lab["large"], "small_label_end": lab["small"],
              "held_franka": lab["held"], "wrap_deg_end": lab["small_wrap_deg"]}
    timing["teleop"] = report["wall_s"]
    logger.info(f"[TELEOP] {operator or 'aborted'}{f' ({abort})' if abort else ''}: {sim_s:.1f} s "
                f"sim, {len(frames)} frames, large {lab['large']} small {lab['small']} wrap "
                f"{lab['small_wrap_deg']:.0f} deg, clamped {hook.n_clamped} steps, realtime "
                f"{report['realtime_factor'] or 0.0:.2f}x")
    return {"frames": frames, "report": report, "done": done}


def stage1(sim, ctx, sampler, timing: dict, args) -> dict:
    """``--teleop-stage1``: stage 1 as a teleop phase (no controller; the UR keeps its start joint
    target) -> a ``_run_loop``-style dict; ``stop`` = no UR exit."""
    tp = run(sim, ctx, sampler, {"sim_step": sim.step_index, "step": 0}, timing, args,
             "s1:teleop", args.grasp_arms)
    r = tp["report"]
    return {"frames": tp["frames"], "status": "ok" if tp["done"] else "aborted",
            "reason": r["abort"] or (None if tp["done"] else "operator abort"), "timing": {},
            "tip_min_m": min(f["tip_clearance_mm"] for f in tp["frames"]) * 1e-3,
            "teleop": r, "stop": not tp["done"]}


def _status(sim, ctx, fr: dict, hook: TeleopHook, inp: SpaceMouseInput, st: dict, now: float,
            k0: int, paused: bool) -> None:
    lab = harness._label_row(ctx, fr)
    steps = sim.step_index - st["step"]
    rt = steps * sim.frame_dt / st["active"] if st["active"] > 0 else 0.0
    rate = (inp.n_msgs - st["msgs"]) / (now - st["wall"])
    sys.stdout.write(
        f"\r[TELEOP] t {(sim.step_index - k0) * sim.frame_dt:6.1f} s | large {lab['large']:<8} "
        f"small {lab['small']:<8} wrap {lab['small_wrap_deg']:4.0f} deg | held "
        f"{'y' if lab['held'] else 'N'} | |v| {np.linalg.norm(hook.v) * 1e3:3.0f} mm/s"
        f"{' CLAMP' if hook.clamped else '      '} | spacemouse {rate:4.0f}/s | rt {rt:.2f}x"
        f"{' PAUSED' if paused else '       '}")
    sys.stdout.flush()
    st.update(wall=now, step=sim.step_index, active=0.0, msgs=inp.n_msgs)


def tally(rows: list[dict]) -> tuple[int, int]:
    """``(both engaged, attempted)`` over the episodes that reached teleop (discards excluded)."""
    tele = [r for r in rows if r.get("status") != "discarded"
            and "teleop" in (r.get("chain") or {})]
    return sum(bool(r["chain"]["success_both"]) for r in tele), len(tele)


def discard(out: Path, row: dict) -> dict:
    """Operator abort: remove the episode npz + recording; the ``discarded`` row."""
    for rel in (row.get("file"), row.get("recording")):
        if rel and (out / rel).is_dir():
            shutil.rmtree(out / rel)
        elif rel:
            (out / rel).unlink(missing_ok=True)
    logger.info(f"[TELEOP] {row.get('file')} discarded (operator abort)")
    return {"file": None, "start_state": row["start_state"], "repeat": row["repeat"],
            "mode": row["mode"], "status": "discarded", "outcome": None,
            "reason": "operator abort", "chain": row["chain"]}


class Session:
    """The plan in order; with ``prompt`` (a tty) after each episode (``rows[-1]``, appended by
    the caller) its result, the running tally and ``[n]ext / [r]etry / [q]uit`` (a retry repeats
    the start as ``repeat + 100 k``)."""

    def __init__(self, plan: list, prompt: bool, rows: list[dict]) -> None:
        self.plan, self.prompt, self.rows = plan, prompt, rows

    def __iter__(self):
        for state, repeat in self.plan:
            k = 0
            while True:
                yield state, repeat + 100 * k
                if not self.prompt:
                    break
                print(f"[TELEOP] {_result_line(self.rows[-1])}; session "
                      "{}/{} both engaged".format(*tally(self.rows)), flush=True)
                ans = ""
                while ans not in ("n", "r", "q"):
                    ans = (input("[n]ext / [r]etry same start / [q]uit: ").strip().lower()
                           or "n")[:1]
                if ans == "q":
                    return
                if ans == "n":
                    break
                k += 1


def _result_line(row: dict) -> str:
    ch = row.get("chain") or {}
    name = f"{row['start_state']['id']}/{row['repeat']}"
    if row.get("status") == "discarded":
        return f"{name} discarded"
    if ch.get("stage1", {}).get("failed"):
        return f"{name} stage 1 {ch['stage1']['large_label_end']}"
    if ch.get("ur_exit", {}).get("failed"):
        return f"{name} UR exit failed ({ch['ur_exit'].get('abort') or 'not engaged / held'})"
    if "teleop" not in ch:
        return f"{name} {row.get('status')} ({row.get('reason')})"
    tp = ch["teleop"]
    return (f"{name} {row['outcome']} (both {ch['success_both']}, operator {tp['operator']}"
            f"{', ' + tp['abort'] if tp['abort'] else ''})")
