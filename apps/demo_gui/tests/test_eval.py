"""Eval panel checks, CPU-only (``run_hard_suite.py write`` only writes run.sh / suite.json).

    uv run --frozen python apps/demo_gui/tests/test_eval.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.demo_gui import evalrun, jobs
from apps.demo_gui.app import build_app
from apps.demo_gui.backends.mock import MockBackend
from apps.demo_gui.config import REPO, Cfg
from apps.demo_gui.panels import evaluate
from apps.demo_gui.session import Session

FX = Path(__file__).resolve().parent / "fixtures"
ARM = "V63j_tol50b"
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_eval_")  # removed at exit


def _session() -> Session:
    return Session(tempfile.mkdtemp(dir=_TMP.name))


def _until(cond, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _fixture_run(s: Session) -> Path:
    out = s.eval / f"{ARM}_00"
    shutil.copytree(FX / "e2e_run", out / f"{ARM}_nominal")
    return out


def test_models_and_keys():
    s = _session()
    exp = s.train / "t0" / "export"
    exp.mkdir(parents=True)
    (exp / "arm.json").write_text(json.dumps({"mine": {k: "x" for k in evalrun.ARM_KEYS}}))
    m = evalrun.models(s)
    assert ARM in m and m["mine"]["wp1"] == "x"
    assert len(evalrun.IN_DIST) == 8 and "set2_gv_06" not in evalrun.IN_DIST


def test_write_argv():
    s = _session()
    out = evalrun.new_out(s, ARM)
    argv = evalrun.write_argv(Cfg(), out, ARM, evalrun.models(s)[ARM], 1, evalrun.IN_DIST)
    j = " ".join(argv)
    assert "--suite full" in j and "--port 7850" in j and f"--out {s.root}/eval/" in j
    assert argv.count("--arm") == 1 and argv.count("--arm2") == 1
    assert "7857" not in j and "7667" not in j
    assert "--render-every 3 --publish-body-poses --realtime" in j, j
    for port in (7857, 7667, 7856):
        try:
            evalrun.write_argv(Cfg(sim_port=port), out, ARM, {}, 1, ["nominal"])
            raise AssertionError(port)
        except ValueError:
            pass
    r = evalrun.run_argv(out)
    assert r[:2] == ["bash", "-c"] and f"{out}/run.sh" in r[2] and "summarize --suite full" in r[2]


def test_hard_suite_write_offline():
    s = _session()
    out = evalrun.new_out(s, ARM)
    argv = evalrun.write_argv(Cfg(), out, ARM, evalrun.models(s)[ARM], 1, evalrun.IN_DIST)
    p = subprocess.run(argv, check=False, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    sh = (out / "run.sh").read_text()
    assert sum(ln.startswith("run ") for ln in sh.splitlines()) == 8
    assert "udpm://239.255.76.250:7850?ttl=0" in sh and "239.255.78." not in sh
    assert json.loads((out / "suite.json").read_text())["port"] == 7850


def test_results_fixture():
    s = _session()
    out = _fixture_run(s)
    rows = evalrun.results(out)
    assert len(rows) == 1 and rows[0]["start"] == "nominal" and rows[0]["both"] is True
    assert isinstance(rows[0]["strict"], bool) and rows[0]["strict_why"].startswith("span")
    assert math.isfinite(rows[0]["wrap_l"]) and math.isfinite(rows[0]["wrap_s"])
    assert evalrun.suite_md(out) is None and len(evalrun.episode_rows(out)) == 1


def test_scrub_figures():
    out = _fixture_run(_session())
    npz = evalrun.episode_rows(out)[0]["value"]
    n = evalrun.n_frames(npz)
    for i in (0, n - 1):
        fig, wrap = evalrun.scrub_figures(npz, i)
        assert len(fig.data) >= 3 and fig.layout.title.text == f"frame {i}"
        assert len(wrap.data) == 2 and len(wrap.layout.shapes) == 1


def test_latent_figure():
    import numpy as np

    out = _fixture_run(_session())
    npz = evalrun.episode_rows(out)[0]["value"]
    d = evalrun.load_episode(npz)
    ns, i = d["sim_goal_dist"].shape[1], 3
    fig = evalrun.latent_figure(npz, i)
    assert len(fig.data) == ns + 1  # one line per stage + the error
    t = d["sim_time"] - d["sim_time"][0]
    assert fig.layout.shapes[0].x0 == float(t[i])
    ok = np.isfinite(d["sim_latent_pred"]).all(1) & np.isfinite(d["sim_latent"]).all(1)
    assert np.isfinite(np.asarray(fig.data[-1].y)[ok]).all()
    tmp = Path(_TMP.name) / "bare.npz"
    z = np.zeros((4, 3))  # _load needs the belt/EE keys every episode has
    np.savez(tmp, sim_time=np.arange(4.0), sim_belt_xyz=np.zeros((4, 5, 3)),
             sim_ee_franka=z, sim_ee_ur=z)
    assert evalrun.latent_figure(tmp, 0).layout.title.text == "no latent data in this episode"


def test_layout_get():
    s = _session()
    app = build_app(MockBackend(), s, Cfg())
    c = app.server.test_client()
    assert c.get("/").status_code == 200
    r = c.get("/_dash-layout")
    assert r.status_code == 200 and b"eval-table" in r.data and b"eval-3d" in r.data and b"eval-latent" in r.data


def _post(app, output: str, values: dict, changed: str):
    """POST the callback writing ``output`` like the browser does -> its response dict."""
    key, cb = next(kv for kv in app.callback_map.items() if output in kv[0])
    outs = [{"id": o.rsplit(".", 1)[0], "property": o.rsplit(".", 1)[1]}
            for o in key.strip(".").split("...")]
    r = app.server.test_client().post("/_dash-update-component", json={
        "output": key, "outputs": outs if key.startswith("..") else outs[0],
        "inputs": [{**i, "value": values.get(i["id"])} for i in cb["inputs"]],
        "state": [{**i, "value": values.get(i["id"])} for i in cb["state"]],
        "changedPropIds": [changed]})
    assert r.status_code == 200, r.data[:300]
    return json.loads(r.data)["response"]


def test_frame_callback_three_figures():
    app = build_app(MockBackend(), _session(), Cfg())
    npz = evalrun.episode_rows(_fixture_run(_session()))[0]["value"]
    r = _post(app, "eval-latent.figure", {"eval-frame": 2, "eval-episode": npz},
              "eval-frame.value")
    assert {"eval-3d", "eval-wrap", "eval-latent"} <= set(r), r.keys()


def test_second_run_refused():
    s = _session()
    app = build_app(MockBackend(), s, Cfg())
    busy = jobs.run_thread(s, f"eval_{ARM}", lambda stop: stop.wait(10))
    try:
        r = _post(app, "eval-status.children", {"eval-run-btn": 1, "eval-model": ARM,
                                                "eval-starts": evalrun.IN_DIST},
                  "eval-run-btn.n_clicks")
        assert "in progress" in r["eval-status"]["children"], r
        assert not list(s.eval.glob(f"{ARM}_*")), "a second eval started"
    finally:
        jobs.stop(busy)


def test_viewer_msg_survives_tick():
    app = build_app(MockBackend(), _session(), Cfg())
    r = _post(app, "eval-viewer-msg.children", {"eval-viewer-btn": 1}, "eval-viewer-btn.n_clicks")
    assert r["eval-viewer-msg"]["children"] == "pick an episode first", r
    writers = [k for k in app.callback_map if "eval-viewer-msg." in k]
    assert len(writers) == 1 and "tick" not in str(app.callback_map[writers[0]]["inputs"])
    r = _post(app, "eval-sig.data", {"tick": 2}, "tick.n_intervals")  # the 1 s poll
    assert "eval-viewer-msg" not in r and "eval-status" in r, r


def test_mock_evaluate_results():
    s = _session()
    job = MockBackend(pace=0.01).evaluate(s, ARM, evalrun.models(s)[ARM], evalrun.IN_DIST, 1)
    assert evaluate.eval_runs(s)[0]["label"] == f"{ARM}_00"  # listed before it finishes
    assert _until(lambda: jobs.status(job) == "done", 5.0)
    out = s.eval / f"{ARM}_00"
    rows, md, eps = evaluate.results_view(out)
    assert len(rows) == 1 and rows[0]["strict"] in ("OK", "no") and len(eps) == 1
    assert "strict 0/1" in md and "Full-run hard suite" in md
    assert evaluate.run_sig(str(out))[0] == str(out)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
