"""Detached job runner: sim lock + GPU gate, PID file per job, stop by explicit PIDs (never pkill)."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from apps.demo_gui import telemetry
from apps.demo_gui.config import GPU_FREE, LOCK, REPO

# A process on the job's LCM URL whose argv[0] basename matches is the job's controller.
CONTROLLER_NAMES = {"run_round_belt_assembly_controller", "franka_cartesian_osc_controller"}
# The MPPI node runs as ``python mppi_controller_node.py``: matched on any argv basename.
NODE_SCRIPTS = {b"mppi_controller_node.py"}
GPU_WAIT_S = 30.0  # a sim job gated this long by gpu_free shows what holds the GPU
_GPU_APPS = [float("-inf"), ""]  # (monotonic time, nvidia-smi output)

_LIVE: dict[str, subprocess.Popen | _ThreadJob] = {}


@dataclass
class Job:
    name: str
    path: str
    argv: list[str]
    log: str
    started: str
    pid: int | None = None
    pgid: int | None = None
    lcm_url: str | None = None
    killed: list[int] = field(default_factory=list)

    def save(self) -> None:
        telemetry.write(self.path, asdict(self))


class _ThreadJob:
    """In-process mock job: ``fn(stop_event)`` on a thread."""

    def __init__(self, fn) -> None:
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, args=(fn,), daemon=True)

    def _run(self, fn) -> None:
        try:
            fn(self.stop_event)
        except BaseException as e:  # noqa: BLE001
            self.error = e


def _new(session, name: str, argv: list[str], lcm_url: str | None) -> Job:
    return Job(name=name, path=str(session.jobs / f"{name}.json"), argv=[str(a) for a in argv],
               log=str(session.jobs / f"{name}.log"), started=time.strftime("%Y-%m-%dT%H:%M:%S"),
               lcm_url=lcm_url)


def start(session, name: str, argv: list[str], *, cwd: Path = REPO, env: dict | None = None,
          sim: bool = False, gpu: bool = False, lcm_url: str | None = None) -> Job:
    """``setsid nohup argv < /dev/null`` equivalent; sim / GPU jobs wait on the lock + gpu_free."""
    argv = [str(a) for a in argv]
    if sim or gpu:
        argv = ["flock", str(LOCK), "bash", "-c",
                f"until bash {shlex.quote(str(GPU_FREE))}; do sleep 20; done; exec {shlex.join(argv)}"]
    job = _new(session, name, argv, lcm_url)
    with open(job.log, "ab") as log:
        proc = subprocess.Popen(argv, cwd=cwd, env={**os.environ, **(env or {})},
                                start_new_session=True, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT)
    job.pid = job.pgid = proc.pid
    _LIVE[job.path] = proc
    job.save()
    return job


def run_thread(session, name: str, fn) -> Job:
    """Mock job: ``fn(stop_event)`` on a daemon thread, tracked like a process job."""
    job = _new(session, name, ["thread", name], None)
    Path(job.log).touch()
    t = _LIVE[job.path] = _ThreadJob(fn)
    job.save()
    t.thread.start()
    return job


def load(path: str | Path) -> Job:
    d = json.loads(Path(path).read_text())
    return Job(**{k: d[k] for k in Job.__dataclass_fields__ if k in d})


def _stat(pid: int) -> list[str]:
    """``/proc/<pid>/stat`` fields after the comm: state, ppid, pgrp, session, ... ([] if gone)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return []


def running(session, name: str) -> bool:
    """Is the session's job ``name`` (``<s>/jobs/<name>.json``) still running?"""
    p = session.jobs / f"{name}.json"
    return p.is_file() and status(load(p)) == "running"


def _alive(pid: int) -> bool:
    st = _stat(pid)
    return bool(st) and st[0] != "Z"


def _chain(pid: int) -> list[tuple[int, int]]:
    """``(pid, session id)`` from ``pid`` up to init, ``pid`` first."""
    out = []
    while pid > 0 and (st := _stat(pid)):
        out.append((pid, int(st[3])))
        pid = int(st[1])
    return out


def status(job: Job) -> str:
    """running | done | failed (exit code from the in-process handle, else /proc liveness only)."""
    h = _LIVE.get(job.path)
    if isinstance(h, _ThreadJob):
        return "running" if h.thread.is_alive() else ("failed" if h.error else "done")
    if isinstance(h, subprocess.Popen):
        rc = h.poll()
        return "running" if rc is None else ("done" if rc == 0 else "failed")
    # shortcut: after an app restart a reused PID reads as running; record start time if it bites.
    return "running" if job.pid and _alive(job.pid) else "done"


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop(job: Job, timeout_s: float = 5.0) -> None:
    """SIGINT the job's process group, SIGKILL after ``timeout_s``, then its controllers."""
    h = _LIVE.get(job.path)
    if isinstance(h, _ThreadJob):
        h.stop_event.set()
        h.thread.join(timeout_s)
        return
    if job.pgid:
        try:
            os.killpg(job.pgid, signal.SIGINT)
        except ProcessLookupError:
            pass
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if isinstance(h, subprocess.Popen):
                h.poll()  # reaps the leader so the group can read empty
            if not _group_alive(job.pgid):
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(job.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if isinstance(h, subprocess.Popen):
            try:
                h.wait(1.0)
            except subprocess.TimeoutExpired:
                pass
    if job.lcm_url:
        job.killed = kill_port_procs(job.lcm_url, job.pgid)
        job.save()


def port_procs(lcm_url: str) -> list[int]:
    """PIDs whose cmdline contains ``lcm_url`` and whose argv[0] names a controller (or MPPI node)."""
    pids = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        name = os.path.basename(argv[0].decode(errors="replace"))
        if lcm_url.encode() in b" ".join(argv) and (
                name in CONTROLLER_NAMES or "assembly_controller" in name or "osc" in name
                or any(os.path.basename(a) in NODE_SCRIPTS for a in argv)):
            pids.append(int(p.name))
    return pids


def owned_or_orphan(pid: int, owner_pgid: int | None) -> bool:
    """``pid``'s parent chain reaches session ``owner_pgid``, or its parent is init / a subreaper.

    Controllers run in their own session (magna_process), so the harness above them carries the
    job's session id. An orphan is reparented to init or a subreaper, i.e. one of our ancestors.
    """
    chain = _chain(pid)[1:]
    if owner_pgid and any(owner_pgid in ps for ps in chain):
        return True
    return not chain or chain[0][0] in {p for p, _ in _chain(os.getpid())}


def kill_port_procs(lcm_url: str, owner_pgid: int | None = None,
                    timeout_s: float = 2.0) -> list[int]:
    """SIGTERM then SIGKILL the controllers on ``lcm_url`` that are ``owner_pgid``'s or orphans.

    Another job's live controllers on the same URL (e.g. a demo while this eval is queued) survive.
    """
    pids = [p for p in port_procs(lcm_url) if owned_or_orphan(p, owner_pgid)]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in [p for p in pids if _alive(p)]:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        end = time.monotonic() + timeout_s
        while time.monotonic() < end and any(_alive(pid) for pid in pids):
            time.sleep(0.05)
    return pids


def _gpu_apps(ttl_s: float = 5.0) -> str:
    """``nvidia-smi --query-compute-apps=pid,process_name`` (as gpu_free.sh), cached ``ttl_s``."""
    if time.monotonic() - _GPU_APPS[0] > ttl_s:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                  "--format=csv,noheader"], check=False, capture_output=True,
                                 text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        _GPU_APPS[:] = [time.monotonic(), out]
    return _GPU_APPS[1]


def gpu_wait(job: Job, min_s: float = GPU_WAIT_S) -> str | None:
    """``waiting for GPU: <pid> <name>, ...`` if ``job`` sat in its gpu_free loop over ``min_s``."""
    gate = f"until bash {shlex.quote(str(GPU_FREE))};".encode()
    try:  # flock's child is the gate's bash until it execs the job
        kids = Path(f"/proc/{job.pid}/task/{job.pid}/children").read_text().split()
        kid = next(int(k) for k in kids if gate in Path(f"/proc/{k}/cmdline").read_bytes())
        up = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, StopIteration):
        return None
    st = _stat(kid)
    if not st or up - int(st[19]) / os.sysconf("SC_CLK_TCK") < min_s:
        return None
    apps = [line.split(",", 1) for line in _gpu_apps().splitlines() if "," in line]
    held = ", ".join(f"{p.strip()} {os.path.basename(n.strip())}" for p, n in apps)
    return f"waiting for GPU: {held or '?'}"
