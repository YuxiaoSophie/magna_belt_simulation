"""Panels 2 + 3 (MILES, data browser) on a session built from fixtures; CPU-only, no sim / server.

    uv run --frozen python apps/demo_gui/tests/test_miles_browser.py
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.demo_gui import episodes, jobs
from apps.demo_gui.app import build_app
from apps.demo_gui.backends.mock import MockBackend
from apps.demo_gui.backends.sim import MILES_FLAGS, SimBackend
from apps.demo_gui.config import Cfg
from apps.demo_gui.panels import browser, evaluate, miles
from apps.demo_gui.session import Session, read_buffer

FX = Path(__file__).resolve().parent / "fixtures"
E2E = FX / "e2e_run/episode_pre_place_1_osc_0.npz"
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_mb_")  # removed at exit


def _until(cond, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _session() -> Session:
    """demo_00 = e2e run, demo_01 = stage-1 demo, miles_00 = 5 rows over 3 files, one eval run."""
    s = Session(tempfile.mkdtemp(dir=_TMP.name))
    shutil.copytree(FX / "e2e_run", s.demos / "demo_00")
    d1 = s.demos / "demo_01"
    shutil.copytree(FX / "demo", d1)
    (d1 / "index.json").write_text(json.dumps({"episodes": [{
        "file": "episode_gv_07_0.npz", "frames": 53, "outcome": "engaged", "success_both": True,
        "start_state": {"id": "gv_07", "file": "data/lcs/start_states/grasp_variants/set3/gv_07_osc.npz"},
        "cmd_source_counts": {"0": 53}}]}))
    m = s.miles / "miles_00"
    shutil.copytree(FX / "miles", m)
    files = sorted((m / "miles").glob("*.npz"))
    buf = json.loads((m / "buffer.json").read_text())
    for i, r in enumerate(buf["rollouts"]):
        r["episode"] = str(files[i % len(files)])
    (m / "buffer.json").write_text(json.dumps(buf))
    shutil.copytree(FX / "e2e_run", s.eval / "armA" / "armA_nominal")
    return s


def test_scan_counts():
    rows = episodes.scan(_session())
    kinds = [r["kind"] for r in rows]
    assert kinds.count("demo") == 2 and kinds.count("miles") == 5 and kinds.count("eval") == 1, kinds
    ev = next(r for r in rows if r["kind"] == "eval")
    assert ev["name"] == "armA/armA_nominal/episode_pre_place_1_osc_0" and ev["frames"] == 459
    assert ev["cmd_sources"]["8"] == 127 and ev["success"] is True
    assert all(Path(r["path"]).is_file() for r in rows)
    assert all(r["start_state"] for r in rows)


def test_scan_skips_rows_without_file():
    s = _session()
    idx = s.demos / "demo_01" / "index.json"
    d = json.loads(idx.read_text())
    st = d["episodes"][0]["start_state"]
    d["episodes"] += [{"file": None, "start_state": st, "status": "skipped", "outcome": None},
                      {"file": None, "start_state": st, "status": "discarded", "outcome": None,
                       "reason": "operator abort"},
                      {"start_state": st}]
    idx.write_text(json.dumps(d))
    assert [r["kind"] for r in episodes.scan(s)].count("demo") == 2


def test_load_traces_and_ur_exit():
    for p in (E2E, FX / "demo/episode_gv_07_0.npz", *sorted((FX / "miles/miles").glob("*.npz"))):
        tr = episodes.load_traces(p)
        n = tr["n_frames"]
        assert tr["t"].shape == (n,) and tr["held"].shape == (n, 2) and tr["held"].dtype == bool
        assert tr["franka_xyz_mm"].shape == (n, 3) and tr["ur_xyz_mm"].shape == (n, 3)
        assert tr["wrap_large_deg"].shape == tr["wrap_small_deg"].shape == tr["phase"].shape == (n,)
        assert 100 < np.abs(tr["franka_xyz_mm"]).max() < 2000  # mm, not m
        with np.load(p, allow_pickle=True) as z:
            assert episodes.ur_exit_mask(tr).sum() == (z["sim_cmd_source"] == 8).sum()
        pc, belt = episodes.cloud_at(p, n // 2)
        assert pc.ndim == 2 and pc.shape[1] == 3 and belt.shape == (48, 3)
        assert len(episodes.traces_figure(tr, frame=n - 1).data) == 11
    tr = episodes.load_traces(E2E)
    tr["cmd_source"] = np.where(np.arange(tr["n_frames"]) >= 30, 8, 0)
    assert episodes.ur_exit_mask(tr).sum() == 10
    fig = episodes.traces_figure(tr)
    assert any(s.type == "rect" for s in fig.layout.shapes)
    assert "excluded from training" in str(fig.layout.annotations)


def test_stage1_input():
    out = Path(tempfile.mkdtemp(dir=_TMP.name))
    ep, marks = episodes.stage1_input(E2E, out, "demo_00")
    assert Path(ep).is_relative_to(out) and marks.startswith("0:"), (ep, marks)
    with np.load(ep, allow_pickle=True) as z:
        assert "sim_chain_from" not in z.files and set(np.unique(z["sim_cmd_source"])) <= {0, 1}
    seed = FX / "miles/miles" / "r_set1_gv_08_walk_026_028.npz"  # no chain: passes through
    assert episodes.stage1_input(seed, out, "x") == (str(seed), None)


def test_miles_argv():
    s = _session()
    calls = []

    class FakePopen:  # never runs flock / the sim
        pid = 999999999

        def __init__(self, argv, **kw):
            calls.append((argv, kw))

    cfg = Cfg(backend="sim")
    demos = [{"name": "demo_00_ep", "episode": str(s.demos / "demo_00" / E2E.name),
              "start": str(FX / "pre_place_1.npz")}]
    real, jobs.subprocess.Popen = jobs.subprocess.Popen, FakePopen
    try:
        job = SimBackend(cfg).run_miles(s, demos, cycles=3, wp_stride=7)
    finally:
        jobs.subprocess.Popen = real
    argv, kw = calls[0]
    assert argv[:2] == ["flock", str(jobs.LOCK)] and job.lcm_url == cfg.lcm_url
    inner = shlex.split(argv[4].split("; exec ", 1)[1])
    assert inner[:5] == ["uv", "run", "--frozen", "python", "scripts/lcs/miles_seed.py"]
    assert inner[inner.index("--out") + 1] == str(s.miles / "miles_01")  # miles_00 is the fixture
    assert inner[inner.index("--lcm-url") + 1] == cfg.lcm_url
    assert inner[inner.index("--demos") + 1] == "demo_00_ep"
    extra = inner[inner.index("--extra-demo") + 1]
    assert extra.startswith(f"demo_00_ep={s.miles / 'miles_01_in'}/") and extra.count(",") == 2
    k = inner.index("--cycles")
    assert inner[k:k + 5] == ["--cycles", "3", "--wp-stride", "7", "--record"]
    assert inner[k + 5:] == MILES_FLAGS and "--franka-grip-n" in MILES_FLAGS
    assert "7667" not in " ".join(argv)
    assert kw["env"]["ROUND_BELT_SCENE"] == str(cfg.scene)
    print("   MILES argv:", shlex.join(inner))


def test_mock_run_miles():
    s = Session(tempfile.mkdtemp(dir=_TMP.name))
    job = MockBackend(pace=0.02).run_miles(s, [])
    buf = s.miles / "miles_00" / "buffer.json"
    assert _until(lambda: len(read_buffer(buf)) >= 2, 20.0)
    jobs.stop(job)
    assert [r["kind"] for r in episodes.scan(s)][:2] == ["miles", "miles"]
    assert "miles_00: " in str(miles.progress(s, job))


def test_panels_build():
    s = _session()
    opts = miles.demo_options(s)
    assert sum(o["label"].startswith("session ") for o in opts) == 2
    assert any(o["label"].startswith("repo seed ") for o in opts)
    assert all({"name", "episode", "start"} <= json.loads(o["value"]).keys() for o in opts)
    rows = browser.table_rows(s)
    assert len(rows) == 8 and all(isinstance(r["cmd_sources"], str) for r in rows)
    app = build_app(MockBackend(), s, Cfg())
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    for cid in ("miles-demos", "miles-start", "br-table", "br-frame", "br-viewer-open", "dock-view"):
        assert f'"{cid}"' in client.get("/_dash-layout").get_data(as_text=True), cid


def test_one_shared_replay_viewer():
    s = _session()
    other = jobs.start(s, "replay", ["bash", "-c", "sleep 30"])  # e.g. started by the eval panel
    calls = []

    class FakePopen:  # never starts a viewer
        pid = 999999999

        def __init__(self, argv, **kw):
            calls.append(argv)

    real, jobs.subprocess.Popen = jobs.subprocess.Popen, FakePopen
    try:
        episodes.start_replay(s, s.demos / "demo_00" / E2E.name, 8092)
    finally:
        jobs.subprocess.Popen = real
    assert jobs.status(other) != "running" and not jobs._alive(other.pid)
    argv = calls[0]
    assert argv[argv.index("--recordings") + 1] == str(s.demos / "demo_00" / "recordings")
    assert argv[-2:] == ["--run", "episode_pre_place_1_osc_0"] and "8092" in argv
    assert evaluate.episodes.start_replay is episodes.start_replay  # eval panel shares it


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
