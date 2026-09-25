#!/usr/bin/env python3
"""Shell validation of ``procman/newton_assembly_learned_sim.pmd`` on a private LCM URL.

Launches the pmd's commands (asserted equal to the pmd's ``exec`` lines, URL aside) in the pmd
script's order: sim -> first ``FRANKA_STATE`` -> OSC subsystems -> 2 s -> encoder node -> first
``LATENT_STATE`` -> visualizer -> controller. Measures the latent rate/latency, the controller's
plan latency vs the 75 ms step, the sim real-time factor and the controller marker timeline,
stops its own children (reverse order, by pid), classifies the recording and writes
``report.json``.

    uv run python scripts/lcs/run_learned_stack.py --lcm-url 'udpm://239.255.76.96:7696?ttl=0'
    uv run python scripts/lcs/run_learned_stack.py --dry-run   # table + pmd check, no launch
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import importlib.util
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lcm

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from dairlib import lcmt_robot_output, lcmt_timestamped_saved_traj
from round_belt_task import perturbation as pert
from round_belt_task.episode_io import git_info, write_json
from round_belt_task.outcome import DEFAULT_THRESHOLDS, classify_episode, slant_episode
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common.latent_encoder import LATENT_STATE_CHANNEL
from task_common.magna_process import (
    CONTROLLER_STARTED,
    MAGNA_WORKTREE,
    MagnaProcess,
    check_private_url,
    marker_lines,
    parse_pre_mpc_line,
    parse_stage_line,
    sha256,
)
from task_common.recording import Recording

PROCMAN = REPO_ROOT / "procman"
PMD = PROCMAN / "newton_assembly_learned_sim.pmd"
PMD_URL = "udpm://239.255.76.95:7695?ttl=0"
DEFAULT_URL = "udpm://239.255.76.96:7696?ttl=0"
SHERIFF_CONFIG = Path("/opt/libbot2/0.0.1.20221116/lib/python3/dist-packages/bot_procman/"
                      "sheriff_config.py")
LEARNED_SIM_PARAMS = "systems/parameters/round_belt_controller_params_learned_sim.yaml"
# The encoder deploy matching LEARNED_SIM_PARAMS' lcs_file (learned_lcs_v2_flat_pp2.yaml).
DEFAULT_DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v2_20260925/"
                      "deploy_v2_flat_pp2/deploy.npz")
VIS_PARAMS = "systems/parameters/assembly_visualization_params.yaml"
DEFAULT_LABEL = "learned"
SHARED_PORT = "7667"
STEP_US = 75_000
HALF_CONTROL_STEP_US = 2_500
FRANKA_STATE = "FRANKA_STATE"
PLAN_CHANNEL = "TARGET_CARTESIAN_POSE_TRAJECTORY"
TERMINATE = "Switching to Terminate"
MPC_START = "All pre-MPC targets completed!"
MPC_DONE = "MPC completed!"
OUTCOME_WINDOW_S = 1.0
SIM_READY_TIMEOUT_S = 240.0
ENCODER_READY_TIMEOUT_S = 120.0
STOP_GRACE_S = {"newton-round-belt-simulation": 30.0}
_STATS = re.compile(r"\[STATS\] step (\d+): ([\d.]+) steps/s, compute mean ([\d.]+) ms max "
                    r"([\d.]+) ms, realtime ([\d.]+)x.*resyncs=(\d+)")
_RECORD_OPEN = re.compile(r"\[RECORD\] (\S+) state every")
_RECORD_CLOSED = re.compile(r"\[RECORD\] closed (\S+):")


@dataclass(frozen=True)
class Cmd:
    name: str
    group: str
    argv: tuple[str, ...]
    magna_binary: str | None = None  # relative to the magna root


def command_table(url: str, params: str, deploy: Path, label: str,
                  magna_root: Path) -> list[Cmd]:
    """The pmd's commands in its script's start order; lcm-spy last (never started here)."""
    magna = str(PROCMAN / "run_in_magna_learned.sh")
    board = str(Path(magna_root) / params)
    spy = "bazel-bin/lcmtypes/dair-lcm-spy"

    def m(name, group, binary, *flags):
        return Cmd(name, group, (magna, binary, *flags), binary)

    return [
        Cmd("newton-round-belt-simulation", "newton round-belt task", (
            str(PROCMAN / "run_newton_sim.sh"), "--lcm-url", url, "--cameras",
            "--publish-point-cloud", "--publish-belt-state", "--perception-every", "15",
            "--board-params", board, "--publish-belt-mesh", "--viewer", "viser",
            "--render-every", "20", "--record", "--record-label", label)),
        m("ur-control-simulation", "simulation subsystems",
          "bazel-bin/systems/simulation/ur_control_simulation", f"--lcm_url={url}"),
        m("ur_trajectory_cartesian_osc", "simulation subsystems",
          "bazel-bin/systems/controllers/ur_cartesian_trajectory_controller", "--ur_input_mode=1",
          f"--lcm_url={url}"),
        m("franka_trajectory_cartesian_osc", "simulation subsystems",
          "bazel-bin/systems/controllers/franka_cartesian_osc_controller", "--input_mode=1",
          f"--lcm_url={url}"),
        Cmd("latent-encoder", "perception", (
            str(PROCMAN / "run_learned_encoder.sh"), "--lcm-url", url, "--deploy", str(deploy),
            "--ur-state-channel", "UR_STATE_SIM", "--state-match", "exact", "--params", board)),
        m("round-belt-visualizer", "newton round-belt task",
          "bazel-bin/systems/visualization/magna_visualization",
          f"--visualization_params_file={VIS_PARAMS}",
          f"--assembly_controller_params_file={params}", f"--lcm_url={url}"),
        m("round-belt-assembly-controller", "newton round-belt task",
          "bazel-bin/systems/controllers/run_round_belt_assembly_controller",
          f"--assembly_controller_params={params}", f"--lcm_url={url}",
          f"--local_lcm_url={url}"),
        Cmd("lcm-spy", "debug", ("/usr/bin/env", f"LEARNED_LCM_URL={url}", magna, spy), spy),
    ]


def parse_pmd(path: Path) -> dict:
    """``{commands: {name: (group, argv)}, scripts: [...]}`` via procman's own parser."""
    spec = importlib.util.spec_from_file_location("sheriff_config", SHERIFF_CONFIG)
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)  # direct load: bot_procman/__init__ needs lcm on system python
    cfg = sc.config_from_filename(str(path))

    def walk(group):
        for c in group.commands:
            yield group.name, c
        for sub in group.subgroups.values():
            yield from walk(sub)

    cmds = {}
    for group, c in walk(cfg.root_group):
        cmds[c.attributes["nickname"]] = (group, tuple(shlex.split(c.attributes["exec"])))
    return {"commands": cmds, "scripts": sorted(cfg.scripts)}


def pmd_consistency(path: Path | None = None) -> dict:
    """The pmd's exec lines vs this script's table built with the pmd's URL and defaults."""
    path = PMD if path is None else path
    parsed = parse_pmd(path)
    table = {c.name: (c.group, c.argv) for c in command_table(
        PMD_URL, LEARNED_SIM_PARAMS, DEFAULT_DEPLOY, DEFAULT_LABEL, MAGNA_WORKTREE)}
    diffs = []
    for name in sorted(set(parsed["commands"]) | set(table)):
        got, want = parsed["commands"].get(name), table.get(name)
        if got != want:
            diffs.append({"cmd": name, "pmd": got, "script": want})
    no_url = [n for n, (_, argv) in parsed["commands"].items()
              if not any(PMD_URL in a for a in argv)]
    shared = [n for n, (_, argv) in parsed["commands"].items()
              if any(SHARED_PORT in a for a in argv)]
    return {"ok": not diffs and not no_url and not shared, "pmd": str(path),
            "commands": sorted(parsed["commands"]), "scripts": parsed["scripts"],
            "diffs": diffs, "lines_without_url": no_url, "lines_with_7667": shared}


# --- processes ---------------------------------------------------------------------------------


def descendants(pid: int) -> list[int]:
    """All live descendants of ``pid`` (the wrappers' ``uv run`` spawns the Python child)."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        for task in Path(f"/proc/{p}/task").glob("*"):
            try:
                kids = (task / "children").read_text().split()
            except OSError:
                continue
            for k in kids:
                out.append(int(k))
                todo.append(int(k))
    return out


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def stop_tree(proc: MagnaProcess, grace_s: float) -> dict:
    """SIGINT to the Python child (``uv run`` wrappers) or the pid, wait, SIGKILL own survivors."""
    if proc.proc is None:
        return {"pid": None}
    tree = descendants(proc.pid)
    target = next((p for p in tree if _comm(p).startswith("python")), proc.pid)
    killed = []
    if proc.alive():
        try:
            os.kill(target, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    survivors = [p for p in reversed(tree) if _alive(p)]
    if proc.alive():
        survivors.append(proc.pid)
    for p in survivors:
        try:
            os.kill(p, signal.SIGKILL)
            killed.append(p)
        except ProcessLookupError:
            pass
    code = proc.stop(grace_s=3.0)  # reaps the child and closes its log
    left = [p for p in tree if _alive(p)]
    return {"pid": proc.pid, "tree": tree, "sigint": target, "exit_code": code,
            "sigkill": killed, "left": left}


def udp_sockets(pids: list[int]) -> dict[int, list[str]]:
    lines = subprocess.run(["ss", "-ulnp"], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    out: dict[int, list[str]] = {p: [] for p in pids}
    for line in lines:
        cols = line.split()
        if len(cols) < 5:
            continue
        for p in pids:
            if f"pid={p}," in line:
                out[p].append(cols[3])
    return out


def socket_evidence(procs: dict[str, MagnaProcess], url: str) -> dict:
    """Per command: every pid of its tree and its UDP sockets; fail on 7667 / a foreign group."""
    group = re.search(r"udpm://([\d.]+):(\d+)", url)
    own = f"{group.group(1)}:{group.group(2)}"
    trees = {name: [p.pid, *descendants(p.pid)] for name, p in procs.items() if p.alive()}
    trees["run_learned_stack"] = [os.getpid()]
    socks = udp_sockets([pid for pids in trees.values() for pid in pids])
    per_cmd, bad = {}, []
    for name, pids in trees.items():
        per_cmd[name] = {str(pid): socks[pid] for pid in pids}
        for pid in pids:
            for addr in socks[pid]:
                host, _, port = addr.rpartition(":")
                if port == SHARED_PORT or (host.startswith("239.") and addr != own):
                    bad.append(f"{name} pid {pid} {addr}")
    on_group = [name for name, pids in trees.items()
                if any(own in a for pid in pids for a in socks[pid])]
    return {"url_group": own, "per_cmd": per_cmd, "bad": bad, "on_group": on_group,
            "ok": not bad}


class LogTail:
    """Complete new lines of an append-only log, read incrementally."""

    def __init__(self, path: Path) -> None:
        self.path, self._offset, self._partial = Path(path), 0, b""

    def read(self) -> list[str]:
        if not self.path.exists():
            return []
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        self._offset += len(data)
        *lines, self._partial = (self._partial + data).split(b"\n")
        return [line.decode(errors="replace") for line in lines]


# --- monitor -----------------------------------------------------------------------------------


class Monitor:
    """One LCM subscriber: FRANKA_STATE / LATENT_STATE / plan receipt wall times."""

    def __init__(self, url: str) -> None:
        self.lc = lcm.LCM(url)
        self.franka_wall: dict[int, float] = {}
        self.last_franka_utime = 0
        self.latents: list[tuple[int, float]] = []  # (utime, wall)
        self.plans: list[tuple[float, int, float]] = []  # (wall, utime, time_vec[0]) new plans
        self._last_plan_t0: float | None = None
        self.empty_plans = self.undecodable = 0
        self.lc.subscribe(FRANKA_STATE, self._on_franka)
        self.lc.subscribe(LATENT_STATE_CHANNEL, self._on_latent)
        self.lc.subscribe(PLAN_CHANNEL, self._on_plan)

    def _on_franka(self, _channel: str, data: bytes) -> None:
        now = time.perf_counter()
        utime = int(lcmt_robot_output.decode(data).utime)
        self.franka_wall.setdefault(utime, now)
        self.last_franka_utime = max(self.last_franka_utime, utime)

    def _on_latent(self, _channel: str, data: bytes) -> None:
        now = time.perf_counter()
        self.latents.append((int(lcmt_timestamped_saved_traj.decode(data).utime), now))

    def _on_plan(self, _channel: str, data: bytes) -> None:
        now = time.perf_counter()
        try:
            msg = lcmt_timestamped_saved_traj.decode(data)
        except ValueError:
            self.undecodable += 1
            return
        blocks = msg.saved_traj.trajectories
        if not blocks or not blocks[0].time_vec:
            self.empty_plans += 1
            return
        t0 = float(blocks[0].time_vec[0])
        if t0 != self._last_plan_t0:  # the controller republishes its plan every tick
            self._last_plan_t0 = t0
            self.plans.append((now, int(msg.utime), t0))

    def pump(self, timeout_ms: int = 20) -> None:
        self.lc.handle_timeout(timeout_ms)


def _stats(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "max": float(v.max()),
            "min": float(v.min()), "p95": float(np.percentile(v, 95))}


def latent_metrics(mon: Monitor, mpc_window: tuple[float, float] | None) -> dict:
    lat = sorted(mon.latents, key=lambda x: x[1])
    out: dict = {"n": len(lat)}
    if len(lat) >= 2:
        span = lat[-1][1] - lat[0][1]
        steps = np.diff([u for u, _ in lat])
        sim_span = (lat[-1][0] - lat[0][0]) / 1e6
        # Wall rate = sim rate x the sim's real-time factor.
        out.update(rate_hz=(len(lat) - 1) / span if span > 0 else None,
                   rate_hz_sim_time=(len(lat) - 1) / sim_span if sim_span > 0 else None,
                   utime_step_us=_stats(steps),
                   utime_steps_not_75ms=int(np.count_nonzero(steps != STEP_US)))
    enc = [(w - mon.franka_wall[u]) * 1e3 for u, w in lat if u in mon.franka_wall]
    out["encoder_latency_ms"] = _stats(enc)
    out["latents_without_franka_state"] = sum(u not in mon.franka_wall for u, _ in lat)
    if mpc_window is None:
        out["plan"] = {"n": 0, "note": "no MPC window (MPC never started)"}
        return out
    lo, hi = mpc_window
    in_mpc = [(u, w) for u, w in lat if lo <= w <= hi]
    plans = mon.plans
    plan_walls = [pl[0] for pl in plans]
    lats, before_next, within, offsets, exact = [], 0, 0, [], 0
    for i, (u, w) in enumerate(in_mpc):
        nxt = in_mpc[i + 1][1] if i + 1 < len(in_mpc) else float("inf")
        got = next((p for p in plans[bisect.bisect_left(plan_walls, w):]
                    if p[2] * 1e6 >= u - HALF_CONTROL_STEP_US), None)
        if got is None:
            continue
        ms = (got[0] - w) * 1e3
        lats.append(ms)
        before_next += got[0] < nxt
        within += ms <= STEP_US / 1e3
        offsets.append((got[2] * 1e6 - u) / 1e3)
        exact += abs(got[2] * 1e6 - u) < 1.0
    n = len(in_mpc)
    out["plan"] = {
        "latents_in_mpc": n, "answered": len(lats), "latency_ms": _stats(lats),
        "frac_before_next_latent": before_next / n if n else None,
        "frac_within_75ms": within / n if n else None,
        "plan_t0_minus_latent_ms": _stats(offsets), "plan_t0_equals_latent_utime": exact,
    }
    return out


def sim_stats(lines: list[str], ctrl_start_step: int | None) -> dict:
    rows = []
    for line in lines:
        m = _STATS.search(line)
        if m:
            rows.append({"step": int(m.group(1)), "steps_per_s": float(m.group(2)),
                         "compute_mean_ms": float(m.group(3)),
                         "compute_max_ms": float(m.group(4)), "rtf": float(m.group(5)),
                         "resyncs": int(m.group(6))})

    def summary(rs):
        if not rs:
            return {"n": 0}
        rtf = [r["rtf"] for r in rs]
        return {"n": len(rs), "rtf_mean": float(np.mean(rtf)), "rtf_min": float(np.min(rtf)),
                "steps_per_s_mean": float(np.mean([r["steps_per_s"] for r in rs])),
                "compute_mean_ms": float(np.mean([r["compute_mean_ms"] for r in rs])),
                "compute_max_ms": float(np.max([r["compute_max_ms"] for r in rs])),
                "resyncs": rs[-1]["resyncs"]}

    # The last window is the partial one logged at shutdown.
    full = rows[:-1] if len(rows) > 1 else rows
    later = [r for r in full if ctrl_start_step is not None and r["step"] > ctrl_start_step]
    return {"all": summary(full), "after_controller_start": summary(later), "windows": rows}


def classify_window(rec: Recording, t_end: float, tangent: np.ndarray) -> dict:
    """Outcome over the frames in ``(t_end - 1 s, t_end]`` (majority label) + final slant."""
    t = rec.state_step * rec.control_dt
    sel = np.flatnonzero((t > t_end - OUTCOME_WINDOW_S) & (t <= t_end + 1e-9))
    if sel.size == 0:
        return {"t_end": t_end, "label": None, "note": "no frames"}
    belt_bodies = rec.meta["belt_bodies"]
    large = next(b for b in rec.meta["pulley_bodies"]
                 if "large_round_pulley" in rec.meta["body_labels"][b])
    belt = rec.body_q[sel][:, belt_bodies, :3].astype(np.float64)
    pulley = rec.body_q[sel][:, large, :].astype(np.float64)
    label, metrics = classify_episode(belt, pulley, DEFAULT_THRESHOLDS, last_n=len(sel))
    slant = slant_episode(belt, pulley, tangent, DEFAULT_THRESHOLDS)
    return {"t_end": t_end, "frames": int(sel.size), "label": label,
            "wrap_deg": float(metrics["wrap_deg"][-1]),
            "h_median_mm": float(metrics["h_median_mm"][-1]),
            "slant_deg": float(slant["slant_deg"]), "slant_dir": slant["slant_dir"]}


# --- run ---------------------------------------------------------------------------------------


class Run:
    def __init__(self, args: argparse.Namespace, table: list[Cmd], out: Path) -> None:
        self.args, self.table, self.out = args, table, out
        self.t0 = time.perf_counter()
        self.mon = Monitor(args.lcm_url)
        self.procs: dict[str, MagnaProcess] = {}
        self.tails: dict[str, LogTail] = {}
        self.logs: dict[str, list[str]] = {}
        self.markers: list[dict] = []
        self.launch_s: dict[str, float] = {}
        self.dead: str | None = None

    def rel(self, wall: float | None = None) -> float:
        return (time.perf_counter() if wall is None else wall) - self.t0

    def start(self, cmd: Cmd) -> None:
        proc = MagnaProcess()
        log = self.out / "logs" / f"{cmd.name}.log"
        proc.launch(Path(cmd.argv[0]), cmd.argv[1:], REPO_ROOT, log, self.args.lcm_url)
        self.procs[cmd.name] = proc
        self.tails[cmd.name] = LogTail(log)
        self.logs[cmd.name] = []
        self.launch_s[cmd.name] = self.rel()
        print(f"[STACK] t={self.rel():6.1f} s started {cmd.name} (pid {proc.pid})", flush=True)

    def poll(self, timeout_ms: int = 20) -> None:
        self.mon.pump(timeout_ms)
        now = time.perf_counter()
        for name, tail in self.tails.items():
            lines = tail.read()
            self.logs[name].extend(lines)
            if name != "round-belt-assembly-controller":
                continue
            for _, text in marker_lines("\n".join(lines)):
                entry = {"wall_s": round(self.rel(now), 3), "line": text,
                         "sim_t_est": self.mon.last_franka_utime / 1e6}
                stage = parse_stage_line(text)
                pre = parse_pre_mpc_line(text)
                if stage is not None:
                    entry["t"] = stage[2]
                elif pre is not None:
                    entry["t"] = pre[1]
                self.markers.append(entry)
                print(f"[STACK] t={entry['wall_s']:6.1f} s {text}", flush=True)
        if self.dead is None:
            for name, proc in self.procs.items():
                if not proc.alive():
                    self.dead = f"{name} exited ({proc.proc.returncode})"

    def wait(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and self.dead is None:
            self.poll()

    def wait_until(self, what: str, cond, timeout_s: float) -> float:
        t = time.perf_counter()
        while not cond():
            if self.dead is not None:
                raise RuntimeError(f"{self.dead} while waiting for {what}")
            if time.perf_counter() - t > timeout_s:
                raise TimeoutError(f"no {what} after {timeout_s:g} s")
            self.poll()
        return time.perf_counter() - t

    def marker_wall(self, text: str) -> float | None:
        return next((m["wall_s"] for m in self.markers if text in m["line"]), None)


def run_stack(args, table: list[Cmd], out: Path, report: dict) -> Recording | None:
    run = Run(args, table, out)
    by_name = {c.name: c for c in table}
    order = ["newton-round-belt-simulation", "ur-control-simulation",
             "ur_trajectory_cartesian_osc", "franka_trajectory_cartesian_osc", "latent-encoder",
             "round-belt-visualizer", "round-belt-assembly-controller"]
    if args.no_visualizer:
        order.remove("round-belt-visualizer")
    reason, error, ctrl_start_step = None, None, None
    try:
        run.start(by_name[order[0]])
        report["startup_s"] = {"sim_first_franka_state": round(run.wait_until(
            "FRANKA_STATE", lambda: run.mon.franka_wall, SIM_READY_TIMEOUT_S), 2)}
        for name in order[1:4]:
            run.start(by_name[name])
        run.wait(2.0)
        run.start(by_name["latent-encoder"])
        report["startup_s"]["encoder_first_latent"] = round(run.wait_until(
            "LATENT_STATE", lambda: run.mon.latents, ENCODER_READY_TIMEOUT_S), 2)
        for name in order[5:]:
            run.start(by_name[name])
        ctrl_t = time.perf_counter()
        ctrl_start_step = run.mon.last_franka_utime // 5000
        report["startup_s"]["controller_started"] = round(run.wait_until(
            CONTROLLER_STARTED, lambda: run.marker_wall(CONTROLLER_STARTED) is not None, 60.0), 2)
        run.wait(1.0)
        report["sockets"] = socket_evidence(run.procs, args.lcm_url)
        print(f"[STACK] sockets ok={report['sockets']['ok']} bad={report['sockets']['bad']}",
              flush=True)
        terminate_at = None
        while True:
            run.poll()
            now = time.perf_counter()
            if terminate_at is None and run.marker_wall(TERMINATE) is not None:
                terminate_at = now + 2.0
            if terminate_at is not None and now >= terminate_at:
                reason = "terminate"
                break
            if run.dead is not None:
                reason = run.dead
                break
            if now - ctrl_t > args.max_s:
                reason = "timeout"
                break
            if not report["sockets"]["ok"]:
                reason = "socket on a foreign group"
                break
    except KeyboardInterrupt:
        reason = "interrupted"
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        reason, error = "error", repr(exc)
    finally:
        stops = {}
        for name in reversed(list(run.procs)):
            stops[name] = stop_tree(run.procs[name], STOP_GRACE_S.get(name, 5.0))
            run.poll(0)
        run.poll(0)
    report["end"] = {"reason": reason, "error": error, "wall_s": round(run.rel(), 2)}
    report["stops"] = stops
    report["left_processes"] = [p for s in stops.values() for p in s.get("left", [])]
    report["processes"] = {n: {"argv": list(by_name[n].argv), "pid": p.pid,
                               "launch_s": round(run.launch_s[n], 2), "log": str(p.log_path)}
                           for n, p in run.procs.items()}
    report["markers"] = run.markers
    mpc_lo, mpc_hi = run.marker_wall(MPC_START), run.marker_wall(MPC_DONE)
    window = None
    if mpc_lo is not None:
        window = (mpc_lo + run.t0, (mpc_hi if mpc_hi is not None else run.rel()) + run.t0)
    report["latent"] = latent_metrics(run.mon, window)
    report["plans_seen"] = {"new": len(run.mon.plans), "empty": run.mon.empty_plans,
                            "undecodable": run.mon.undecodable}
    sim_lines = run.logs.get("newton-round-belt-simulation", [])
    report["sim"] = sim_stats(sim_lines, ctrl_start_step)
    report["encoder_stats"] = [ln.strip() for ln in run.logs.get("latent-encoder", [])
                               if "latents/s" in ln]
    rec_path = next((m.group(1) for ln in sim_lines if (m := _RECORD_OPEN.search(ln))), None)
    closed = any(_RECORD_CLOSED.search(ln) for ln in sim_lines)
    report["recording"] = {"path": rec_path, "closed": closed}
    if rec_path is None:
        return None
    return Recording.load(Path(rec_path))


def outcome(rec: Recording, markers: list[dict]) -> dict:
    nominal = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first="pre_place_1", last="place_3")
    tangent = pert.belt_tangent(nominal)
    t_last = float(rec.state_step[-1] * rec.control_dt)
    res = {"final": classify_window(rec, t_last, tangent), "tangent": tangent.tolist()}
    stages = [m for m in markers if "t" in m and "[learned-mpc] stage" in m["line"]]
    done = next((m for m in markers if MPC_DONE in m["line"]), None)
    if done is not None:
        # The stage marker right before "MPC completed!" carries the controller's own t.
        t_done = stages[-1]["t"] if stages else done["sim_t_est"]
        res["at_mpc_completed"] = classify_window(rec, float(t_done), tangent)
    return res


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lcm-url", default=DEFAULT_URL, help="private LCM URL (never 7667)")
    p.add_argument("--magna-root", type=Path, default=MAGNA_WORKTREE,
                   help="exported as MAGNA_ROOT for run_in_magna_learned.sh")
    p.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY, help="encoder deploy.npz")
    p.add_argument("--params", default=LEARNED_SIM_PARAMS, help="yaml relative to the magna root")
    p.add_argument("--max-s", type=float, default=150.0, help="run limit after the controller")
    p.add_argument("--out", type=Path, default=None,
                   help="default data/lcs/stack_runs/<ts>-<record-label>")
    p.add_argument("--record-label", default=DEFAULT_LABEL)
    p.add_argument("--no-visualizer", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="table + pmd check only, no launch")
    return p


def main() -> int:
    args = create_parser().parse_args()
    args.lcm_url = check_private_url(args.lcm_url)
    if args.lcm_url == PMD_URL:
        raise SystemExit(f"{PMD_URL} is the pmd stack's group; validate on another one")
    root = args.magna_root.resolve()
    table = command_table(args.lcm_url, args.params, args.deploy, args.record_label, root)
    consistency = pmd_consistency()
    print(f"[STACK] pmd consistency ok={consistency['ok']} ({len(consistency['commands'])} "
          f"commands, scripts {consistency['scripts']})", flush=True)
    for d in consistency["diffs"]:
        print(f"[STACK]   diff {d['cmd']}:\n    pmd    {d['pmd']}\n    script {d['script']}")
    binaries = {c.name: str(root / c.magna_binary) for c in table if c.magna_binary}
    missing = [b for b in binaries.values() if not Path(b).exists()]
    params = root / args.params
    for c in table:
        print(f"[STACK] {c.group:24s} {c.name:32s} {shlex.join(c.argv)}")
    missing += [str(p) for p in (params, args.deploy) if not p.is_file()]
    if missing:
        print(f"[STACK] missing: {missing}", flush=True)
        return 1
    if not consistency["ok"]:
        return 1
    if args.dry_run:
        return 0
    os.environ["MAGNA_ROOT"] = str(root)
    os.environ["LEARNED_LCM_URL"] = args.lcm_url
    os.environ["LCM_DEFAULT_URL"] = args.lcm_url
    out = args.out or (REPO_ROOT / "data" / "lcs" / "stack_runs"
                       / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.record_label}")
    out.mkdir(parents=True, exist_ok=True)
    magna_main = Path("/home/hienbui/git/magna")
    main_status = subprocess.run(["git", "-C", str(magna_main), "status", "--porcelain"],
                                 capture_output=True, text=True, check=False).stdout
    report: dict = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "pmd_consistency": consistency,
        "binaries": {n: {"path": b, "sha256": sha256(Path(b).resolve())}
                     for n, b in binaries.items()},
        "params": {"path": str(params), "sha256": sha256(params)},
        "deploy": {"path": str(args.deploy), "sha256": sha256(args.deploy)},
        "git": {"repo": git_info(), "magna_root": git_info(root),
                "magna_main_status_before": main_status},
        "thresholds": dataclasses.asdict(DEFAULT_THRESHOLDS),
    }
    rec = run_stack(args, table, out, report)
    report["git"]["magna_main_status_unchanged"] = main_status == subprocess.run(
        ["git", "-C", str(magna_main), "status", "--porcelain"], capture_output=True,
        text=True, check=False).stdout
    if rec is not None:
        report["recording"]["frames"] = rec.frame_count
        report["recording"]["duration_s"] = rec.duration_s
        report["outcome"] = outcome(rec, report["markers"])
    write_json(out / "report.json", report)
    print_summary(report, out)
    ok = (report["end"]["reason"] == "terminate" and report.get("sockets", {}).get("ok")
          and not report["left_processes"])
    return 0 if ok else 2


def print_summary(r: dict, out: Path) -> None:
    lat, plan = r["latent"], r["latent"].get("plan", {})

    def f(d, k, fmt="{:.1f}"):
        v = d.get(k) if isinstance(d, dict) else None
        return "n/a" if v is None else fmt.format(v)

    enc = lat.get("encoder_latency_ms", {})
    pl = plan.get("latency_ms", {})
    sim = r["sim"]["all"]
    print(f"[STACK] end {r['end']['reason']} after {r['end']['wall_s']} s; startup "
          f"{r.get('startup_s')}")
    print(f"[STACK] latents n={lat['n']} rate {f(lat, 'rate_hz', '{:.2f}')} Hz wall, "
          f"{f(lat, 'rate_hz_sim_time', '{:.2f}')} Hz sim time, encoder "
          f"latency mean {f(enc, 'mean')} max {f(enc, 'max')} p95 {f(enc, 'p95')} ms")
    print(f"[STACK] plans: {plan.get('answered', 0)}/{plan.get('latents_in_mpc', 0)} latents "
          f"answered, latency mean {f(pl, 'mean')} max {f(pl, 'max')} p95 {f(pl, 'p95')} ms, "
          f"before next {f(plan, 'frac_before_next_latent', '{:.2f}')}, <=75 ms "
          f"{f(plan, 'frac_within_75ms', '{:.2f}')}")
    print(f"[STACK] sim RTF mean {f(sim, 'rtf_mean', '{:.2f}')} min "
          f"{f(sim, 'rtf_min', '{:.2f}')}, compute max {f(sim, 'compute_max_ms')} ms, "
          f"resyncs {sim.get('resyncs', 'n/a')}")
    for m in r["markers"]:
        print(f"[STACK]   {m['wall_s']:7.2f} s  {m['line']}")
    for key, o in r.get("outcome", {}).items():
        if isinstance(o, dict):
            print(f"[STACK] outcome {key}: {o.get('label')} wrap {f(o, 'wrap_deg')} deg h "
                  f"{f(o, 'h_median_mm')} mm slant {f(o, 'slant_deg')} deg "
                  f"{o.get('slant_dir')}")
    print(f"[STACK] sockets ok={r.get('sockets', {}).get('ok')}, left "
          f"{r['left_processes']} -> {out / 'report.json'}")


if __name__ == "__main__":
    sys.exit(main())
