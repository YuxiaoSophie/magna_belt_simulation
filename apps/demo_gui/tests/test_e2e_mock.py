"""Whole mock pipeline through the panel callbacks + Flask test client, CPU-only, no server.

    uv run --frozen python apps/demo_gui/tests/test_e2e_mock.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # buffer.build imports the encoder (warp); stay off the GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.demo_gui import buffer, episodes, evalrun, jobs
from apps.demo_gui.app import TABS, Ctx, build_app, job_chips
from apps.demo_gui.backends.mock import MockBackend
from apps.demo_gui.config import Cfg
from apps.demo_gui.panels import collect, evaluate, miles, train
from apps.demo_gui.session import Session, read_buffer, read_index
from dash.exceptions import PreventUpdate

ARM = "V63j_tol50b"
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_e2e_")


def _until(cond, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def test_pipeline():
    s = Session(tempfile.mkdtemp(dir=_TMP.name))
    be = MockBackend(pace=0.02)
    ctx = Ctx(be, s, Cfg())
    state = {"franka": {"slide_mm": 3.0, "roll_deg": 0.0, "pivot_deg": 0.0, "depth_mm": 0.0},
             "ur": {"slide_mm": -2.0, "roll_deg": 0.0, "pivot_deg": 15.0, "depth_mm": 3.0}}

    # start state -> index row; pivots / depths mirror Variant.commanded(): only when non-zero
    assert collect.on_action(ctx, "col-make", state).startswith("make_start")
    assert _until(lambda: read_index(s.starts), 5.0) and read_index(s.starts)[0]["held"]
    cmd = read_index(s.starts)[0]["commanded"]
    assert cmd["ur_pivot_deg"] == 15.0 and "franka_pivot_deg" not in cmd, cmd
    assert cmd["ur_depth_mm"] == 3.0 and "franka_depth_mm" not in cmd, cmd

    # demo: telemetry updates within 3 s, then stop -> demo listed
    assert "recording scripted" in collect.on_action(ctx, "col-record", state)
    assert _until(lambda: s.telemetry_path.is_file(), 3.0)
    buf = collect.on_live(ctx, None)[0]

    def grew() -> bool:
        try:
            return len(collect.on_live(ctx, json.loads(json.dumps(buf)))[0]["t"]) > len(buf["t"])
        except PreventUpdate:
            return False
    assert _until(grew, 3.0)
    assert _until(lambda: (s.demos / "demo_00" / "index.json").is_file(), 5.0)
    collect.on_action(ctx, "col-stop", state)
    assert jobs.status(jobs.load(s.jobs / "demo.json")) != "running"
    demo_ep = s.demos / "demo_00" / read_index(s.demos / "demo_00")[0]["file"]

    # MILES -> buffer rows
    mjob = be.run_miles(s, [])
    mbuf = s.miles / "miles_00" / "buffer.json"
    assert _until(lambda: len(read_buffer(mbuf)) >= 2, 20.0)
    jobs.stop(mjob)
    assert "miles_00: " in str(miles.progress(s, mjob))
    assert sum(r["kind"] == "miles" for r in episodes.scan(s)) >= 2

    # buffer + train (mock) -> log grows -> report
    sel = [{"kind": "demo", "episode": str(demo_ep)}, {"kind": "miles", "buffer": str(mbuf)}]
    buffer.build(s, sel, buffer.path(s, "t1"))
    rows = json.loads(buffer.path(s, "t1").read_text())["rollouts"]
    assert rows
    tjob = be.train(s, "t1", rows, 5)
    run = s.train / "t1"
    assert _until(lambda: (run / "train.log").is_file() and (run / "train.log").stat().st_size, 5.0)
    n0 = (run / "train.log").stat().st_size
    assert _until(lambda: (run / "train.log").stat().st_size > n0, 5.0)
    assert _until(lambda: jobs.status(tjob) == "done", 60.0)
    assert "done" in train.status(run, tjob) and len(train.figures(train.trainlog.parse(run / "train.log"))) == 4

    # export -> arm.json
    ejob = be.export(s, "t1")
    assert _until(lambda: jobs.status(ejob) == "done", 5.0)
    assert json.loads((run / "export" / "arm.json").read_text())
    assert "arm.json" in train.export_status(run, ejob)

    # evaluate -> results rows -> scrub at frame 0 and T-1
    vjob = be.evaluate(s, ARM, evalrun.models(s)[ARM], evalrun.IN_DIST, 1)
    assert _until(lambda: jobs.status(vjob) == "done", 10.0)
    out = s.eval / f"{ARM}_00"
    table, md, eps = evaluate.results_view(out)
    assert len(table) == 1 and "strict" in md and len(eps) == 1
    npz = eps[0]["value"]
    for i in (0, evalrun.n_frames(npz) - 1):
        fig, wrap = evalrun.scrub_figures(npz, i)
        assert fig.layout.title.text == f"frame {i}" and len(wrap.data) == 2

    # pages: every tab renders and the page GETs 200
    client = build_app(be, s, ctx.cfg).server.test_client()
    assert client.get("/").status_code == 200
    layout = client.get("/_dash-layout")
    assert layout.status_code == 200 and client.get("/_dash-dependencies").status_code == 200
    text = layout.get_data(as_text=True)
    for name, _ in TABS:
        assert f'"{name}"' in text, name

    # nothing left running
    assert _until(lambda: all(jobs.status(jobs.load(p)) != "running" for p in s.jobs.glob("*.json")), 5.0)
    assert not any("running" in str(c.children) for c in job_chips(s))


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
