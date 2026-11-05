"""Core checks, CPU-only (no GPU, lock, LCM or listening port); plain asserts, pytest-compatible.

    uv run --frozen python apps/demo_gui/tests/test_core.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.demo_gui import jobs, telemetry
from apps.demo_gui.app import build_app, job_chips
from apps.demo_gui.backends.mock import MockBackend
from apps.demo_gui.config import MOCK_RUN, Cfg, lcm_url
from apps.demo_gui.session import Session, read_index

URL = "udpm://239.255.76.252:7852?ttl=0"  # off 7850, the GUI default
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_test_")  # removed at exit


def _session() -> Session:
    return Session(tempfile.mkdtemp(dir=_TMP.name))


def _until(cond, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def test_lcm_url():
    assert lcm_url(7852) == URL
    assert lcm_url(7858) == "udpm://239.255.78.58:7858?ttl=0"
    for bad in (7667, 7700, 7860):
        try:
            lcm_url(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} accepted")


def test_job_start_stop():
    s = _session()
    job = jobs.start(s, "sleep", ["bash", "-c", "sleep 30"])
    d = jobs.load(job.path)
    assert d.pid and d.pgid == d.pid and d.argv == ["bash", "-c", "sleep 30"]
    assert jobs.status(job) == "running"
    t0 = time.monotonic()
    jobs.stop(job)
    assert _until(lambda: jobs.status(job) != "running", 6.0 - (time.monotonic() - t0))
    assert not jobs._alive(job.pid)


def test_sim_job_wraps_lock_and_gpu_gate():
    calls = []

    class FakePopen:  # never runs flock / gpu_free.sh
        pid = 999999999

        def __init__(self, argv, **kw):
            calls.append((argv, kw))

    real, jobs.subprocess.Popen = jobs.subprocess.Popen, FakePopen
    try:
        jobs.start(_session(), "sim", ["python", "x.py", "--out", "a b"], sim=True)
    finally:
        jobs.subprocess.Popen = real
    argv, kw = calls[0]
    assert argv[:4] == ["flock", str(jobs.LOCK), "bash", "-c"], argv
    assert argv[4] == (f"until bash {jobs.GPU_FREE}; do sleep 20; done; "
                       "exec python x.py --out 'a b'"), argv[4]
    assert kw["start_new_session"] and kw["stdin"] == subprocess.DEVNULL


def _spawn(argv: list[str]) -> str:
    """Code that starts ``argv`` in its own session (as magna_process) and prints its pid."""
    return (f"import subprocess, sys; print(subprocess.Popen({argv!r}, executable=sys.executable,"
            f" start_new_session=True, stdout=subprocess.DEVNULL,"
            f" stderr=subprocess.DEVNULL).pid, flush=True)")


_SLEEP = "import time; time.sleep(30)"
FAKE_CTRL = _spawn(["run_round_belt_assembly_controller", "-c", _SLEEP, f"--lcm_url={URL}"])
FAKE_NODE = _spawn([sys.executable, "-c", _SLEEP, "scripts/lcs/mppi_controller_node.py",
                    "--lcm-url", URL])


def test_stop_kills_own_and_orphan_controllers_only():
    assert jobs.port_procs(URL) == [], "a real process is on the test URL; not touching it"
    s = _session()
    # stands in for a harness: its MPPI node is in its own session
    queued = jobs.start(s, "eval_x", [sys.executable, "-c", f"{FAKE_NODE}; {_SLEEP}"],
                        lcm_url=URL)
    other = jobs.start(s, "demo", [sys.executable, "-c", f"{FAKE_CTRL}; {FAKE_NODE}; {_SLEEP}"],
                       lcm_url=URL)
    fakes = []
    try:
        # spawner exits at once: its controller and node are reparented to init / a subreaper
        orphans = {int(p) for p in subprocess.run(
            [sys.executable, "-c", f"{FAKE_CTRL}; {FAKE_NODE}"], check=True, capture_output=True,
            text=True, timeout=10).stdout.split()}
        fakes = list(orphans)
        assert _until(lambda: len(jobs.port_procs(URL)) == 5, 5.0), jobs.port_procs(URL)
        fakes = jobs.port_procs(URL)
        by_parent = {}
        for p in fakes:
            by_parent.setdefault(int(jobs._stat(p)[1]), set()).add(p)
        own, mine = by_parent[queued.pid], by_parent[other.pid]
        assert len(own) == 1 and len(mine) == 2 and own | mine | orphans == set(fakes)
        assert all(jobs.owned_or_orphan(p, queued.pgid) for p in own)
        jobs.stop(queued)
        assert set(jobs.load(queued.path).killed) == own | orphans
        assert not any(jobs._alive(p) for p in own | orphans)
        assert all(jobs._alive(p) for p in mine), "killed another job's controller / node"
        assert set(jobs.kill_port_procs(URL, other.pgid)) == mine  # owned while its harness lives
        assert not any(jobs._alive(p) for p in mine)
    finally:
        for pid in fakes:
            if jobs._alive(pid):
                os.kill(pid, signal.SIGKILL)
        jobs.stop(queued)
        jobs.stop(other)


def test_gpu_wait_status():
    tmp = Path(tempfile.mkdtemp(dir=_TMP.name))
    (tmp / "nvidia-smi").write_text("#!/bin/sh\necho '4242, /usr/bin/python3'\n")
    (tmp / "nvidia-smi").chmod(0o755)
    (tmp / "gpu_free.sh").write_text("exit 1\n")
    saved = jobs.GPU_FREE, jobs.LOCK, os.environ["PATH"]
    jobs.GPU_FREE, jobs.LOCK = tmp / "gpu_free.sh", tmp / "sim.lock"  # never the real lock
    os.environ["PATH"] = f"{tmp}:{saved[2]}"
    jobs._GPU_APPS[:] = [float("-inf"), ""]
    job = jobs.start(_session(), "sim", ["true"], sim=True)
    try:
        assert _until(lambda: jobs.gpu_wait(job, 0.0) is not None, 5.0)
        assert jobs.gpu_wait(job, 0.0) == "waiting for GPU: 4242 python3"
        assert jobs.gpu_wait(job) is None  # < 30 s in the gate
        assert jobs.gpu_wait(jobs.Job("x", "", [], "", "", pid=os.getpid())) is None
    finally:
        jobs.stop(job)
        jobs.GPU_FREE, jobs.LOCK, os.environ["PATH"] = saved
        jobs._GPU_APPS[:] = [float("-inf"), ""]


def test_telemetry_round_trip():
    s = _session()
    d = {"t": 1.5, "step": 3, "held": [True, False], "cloud": [[0.1, 0.2, 0.3]], "source": "mock"}
    assert telemetry.read(s.telemetry_path) is None
    telemetry.write(s.telemetry_path, d)
    assert telemetry.read(s.telemetry_path) == d


def test_episode_streamer():
    s = _session()
    st = telemetry.EpisodeStreamer(MOCK_RUN, s.telemetry_path, pace=0.1)
    seen = set()
    st.start()
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        tel = telemetry.read(s.telemetry_path)
        if tel:
            seen.add(tel["step"])
        time.sleep(0.01)
    st.stop()
    st.join(2.0)
    assert len(seen) >= 2, seen
    tel = telemetry.read(s.telemetry_path)
    assert len(tel["belt_xyz"]) == 48 and 0 < len(tel["cloud"]) <= telemetry.MAX_CLOUD
    assert len(tel["franka_pose7"]) == 7 and len(tel["ur_pose7"]) == 7
    assert [k for k, v in tel.items() if v is None] == [], tel
    assert 0 < tel["franka_width_mm"] < 81 and 0 <= tel["ur_width_mm"] < 70
    assert tel["phase"].startswith("mpc:") and len(tel["held"]) == 2


def test_mock_start_demo_reset():
    s = _session()
    be = MockBackend(pace=0.01)
    job = be.make_start(s, {"slide_mm": 5.0, "roll_deg": -3.0}, {"slide_mm": 0, "roll_deg": 0})
    assert _until(lambda: jobs.status(job) == "done", 5.0)
    row = read_index(s.starts)[0]
    assert row["commanded"]["franka_slide_mm"] == 5.0 and Path(row["file"]).is_file()
    job = be.record_demo(s, row["id"])
    demo = s.demos / "demo_00"
    assert _until(lambda: telemetry.read(s.telemetry_path) is not None, 20.0)
    assert read_index(demo)[0]["start_state"]["id"] == row["id"]
    be.reset_demo(s)
    assert jobs.status(job) == "done" and not demo.exists()


def test_build_app_mock():
    s = _session()
    jobs.start(s, "chip", ["true"])
    app = build_app(MockBackend(), s, Cfg())
    assert app.layout is not None
    assert app.server.test_client().get("/").status_code == 200
    assert _until(lambda: "chip: done" in str(job_chips(s)), 3.0)


def test_chips():
    s = _session()
    assert "no jobs" in str(job_chips(s))
    t = time.time()

    def fake(name, i, pid):  # mtime orders the chips
        jobs.Job(name=name, path=str(s.jobs / f"{name}.json"), argv=["x"], log="", started="",
                 pid=pid, pgid=pid).save()
        os.utime(s.jobs / f"{name}.json", (t + i, t + i))

    for i in range(3):
        fake(f"run{i}", i, os.getpid())  # alive pid -> running
    for i in range(5):
        fake(f"done{i}", 10 + i, 2**22 + 7)  # dead pid -> done
    jobs.start(s, "bad", ["false"])
    assert _until(lambda: jobs.status(jobs.load(s.jobs / "bad.json")) == "failed", 3.0)
    os.utime(s.jobs / "bad.json", (t + 9, t + 9))
    chips = job_chips(s)
    run = [c for c in chips if "running" in c.className]
    assert len(run) == 3 and run[0].children == "run2: running"
    assert chips[3].children == "done4: done" and len(chips) == 5
    summ = chips[4]
    assert summ.children[0].children == "5 done · 1 failed" and len(summ.children[1].children) == 9


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
