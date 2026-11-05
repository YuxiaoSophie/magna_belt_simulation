"""Viewer dock: live > replay > standalone viewer > launch command by job + port state (no reload
per tick), one iframe, the toggle; and the harness's ``--live-viewer`` render wrap survives the UR
exit. CPU only; the "viewers" are local sockets and the jobs ``sleep``.

    uv run --frozen python apps/demo_gui/tests/test_live_viewer.py [eval_learned_mpc.py to check]
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import http.server
import importlib.util
import json
import socket
import sys
import tempfile
import threading
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from apps.demo_gui import jobs, viewer
from apps.demo_gui.app import build_app
from apps.demo_gui.backends.mock import MockBackend
from apps.demo_gui.config import Cfg
from apps.demo_gui.session import Session

HARNESS = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "scripts/lcs/eval_learned_mpc.py"


OUTS = [("dock-view", "children"), ("dock-title", "children"), ("dock-open", "href"),
        ("dock-open", "style"), ("dock-src", "data")]


def _post(client, app, cur):
    """The dock callback on ``tick``; None = every output no_update."""
    key = next(k for k in app.callback_map if "dock-view.children" in k)
    body = {"output": key, "changedPropIds": ["tick.n_intervals"],
            "outputs": [{"id": i, "property": p} for i, p in OUTS],
            "inputs": [{"id": "tick", "property": "n_intervals", "value": 1}],
            "state": [{"id": "dock-src", "property": "data", "value": cur}]}
    r = client.post("/_dash-update-component", json=body)
    assert r.status_code in (200, 204), r.data[:300]
    return (json.loads(r.data)["response"] or None) if r.status_code == 200 else None


_SOCKS: list[socket.socket] = []


def _http() -> http.server.HTTPServer:
    """A local "viewer" answering any GET (501 counts as up), served until shutdown."""
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class _Quiet(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass


def _port(listen: bool = True) -> int:
    """A local "viewer": serving HTTP, or bound-not-listening (not up). Kept open until exit."""
    if listen:
        return _http().server_address[1]
    x = socket.socket()
    x.bind(("127.0.0.1", 0))
    _SOCKS.append(x)
    return x.getsockname()[1]


def _replay(s, rec: Path, run: str | None):
    argv = ["bash", "-c", "sleep 60", "replay_viewer.py", "--recordings", rec, "--port", "1"]
    return jobs.start(s, "replay", argv + (["--run", run] if run else []))


def test_dock_source_rule():
    s = Session(tempfile.mkdtemp(prefix="demo_gui_dock_"))
    vsrv = _http()
    closed = _port(listen=False)
    cfg = Cfg(viewer_port=vsrv.server_address[1], replay_port=_port())
    app = build_app(MockBackend(), s, cfg)
    client = app.server.test_client()
    replay_url, sv = f"http://localhost:{cfg.replay_port}", [cfg.viewer_url, "viewer"]
    started = [jobs.start(s, "evalwrite_x", ["sleep", "60"])]  # not a sim job: ignored
    try:
        r = _post(client, app, None)  # viewer up, no job -> standalone viewer, no ?job=
        assert r["dock-src"]["data"] == sv and r["dock-title"]["children"] == "viewer", r
        kids = r["dock-view"]["children"]
        assert len(kids) == 1 and kids[0]["type"] == "Iframe" and kids[0]["props"]["src"] == sv[0]
        assert r["dock-open"]["href"] == sv[0] and r["dock-open"]["style"] is None
        assert _post(client, app, sv) is None  # unchanged: no reload

        rep = _replay(s, s.demos / "demo_00" / "recordings", "ep_pre_place_1")
        started.append(rep)
        r = _post(client, app, sv)  # replay beats the idle viewer
        rp = [f"{replay_url}/?job={rep.pid}", "replay: ep_pre_place_1"]
        assert r["dock-src"]["data"] == rp and r["dock-view"]["children"][0]["props"]["src"] == rp[0]
        assert _post(client, app, rp) is None

        live = jobs.start(s, "demo", ["sleep", "60"])
        started.append(live)
        down = Cfg(viewer_port=closed, replay_port=cfg.replay_port)
        assert viewer.dock_src(s, down) == tuple(rp)  # viewer not up: replay stays
        r = _post(client, app, rp)  # live beats replay
        assert r["dock-src"]["data"] == [sv[0], "live: demo"] and r["dock-title"]["children"] == "live: demo"
        assert r["dock-view"]["children"][0]["props"]["src"] == sv[0]
        r = _post(client, app, sv)  # same src: the title flips, the iframe is not re-sent
        assert set(r) == {"dock-title", "dock-src"} and r["dock-title"]["children"] == "live: demo"
        jobs.stop(live)
        assert _post(client, app, [sv[0], "live: demo"])["dock-title"]["children"] == "replay: ep_pre_place_1"
        jobs.stop(rep)
        r = _post(client, app, [sv[0], "live: demo"])
        assert set(r) == {"dock-title", "dock-src"} and r["dock-title"]["children"] == "viewer"

        rep = _replay(s, s.miles / "miles_demo_00" / "recordings", None)  # newest run, no --run
        started.append(rep)
        assert viewer.dock_src(s, cfg)[1] == "replay: miles_demo_00"
        assert viewer.dock_src(s, Cfg(viewer_port=closed, replay_port=closed)) is None
    finally:
        for j in started:
            jobs.stop(j)
    vsrv.shutdown()
    vsrv.server_close()
    r = _post(client, app, sv)  # viewer port closed -> placeholder with the launch command
    assert r["dock-src"]["data"] == [None, "no viewer"] and r["dock-title"]["children"] == "no viewer"
    assert cfg.viewer_cmd in json.dumps(r["dock-view"]["children"])
    assert r["dock-open"]["style"] == {"display": "none"}
    assert _post(client, app, None) is None  # the layout's own placeholder


def test_backends_alike():
    from apps.demo_gui.backends.sim import SimBackend

    s = Session(tempfile.mkdtemp(prefix="demo_gui_dock_"))
    cfg = Cfg(viewer_port=_port(), replay_port=_port(listen=False))
    backends = (MockBackend(), SimBackend(cfg))
    assert not any(hasattr(b, "viewer_url") for b in backends)
    run = jobs.run_thread(s, "demo", lambda stop: stop.wait(5))  # the mock demo is live too
    try:
        rs = [_post(a.server.test_client(), a, None) for a in (build_app(b, s, cfg) for b in backends)]
    finally:
        jobs.stop(run)
    assert rs[0] == rs[1] and rs[0]["dock-src"]["data"] == [cfg.viewer_url, "live: demo"], rs


def test_docs_launch_command():
    assert Cfg().viewer_cmd in (REPO / "docs/demo-gui.md").read_text()
    assert Cfg().viewer_cmd.endswith("--port 8091") and "7667" not in Cfg().viewer_cmd


def test_one_viewer_and_toggle():
    app = build_app(MockBackend(), Session(tempfile.mkdtemp(prefix="demo_gui_dock_")), Cfg())
    page = app.layout.to_plotly_json()
    text = json.dumps(page, default=lambda o: o.to_plotly_json())
    assert text.count('"viewer') == 1 and '"Iframe"' not in text  # only the dock's placeholder
    assert app.layout.children[-1].id == "dock"  # outside the tabs: every tab has it
    client = app.server.test_client()

    def post(out, inp, val, state=()):
        key = next(k for k in app.callback_map if k.startswith(out))
        body = {"output": key, "changedPropIds": [f"{inp[0]}.{inp[1]}"],
                "outputs": {"id": out.split(".")[0], "property": out.split(".")[1]},
                "inputs": [{"id": inp[0], "property": inp[1], "value": val}],
                "state": [{"id": i, "property": p, "value": v} for i, p, v in state]}
        r = client.post("/_dash-update-component", json=body)
        assert r.status_code == 200, r.data[:300]
        return json.loads(r.data)["response"]

    r = post("dock-collapsed.data", ("dock-toggle", "n_clicks"), 1,
             [("dock-collapsed", "data", None)])
    assert r["dock-collapsed"]["data"] is True
    r = post("dock.className", ("dock-collapsed", "data"), True)
    assert r["dock"]["className"] == "dock dock--collapsed"
    r = post("dock-collapsed.data", ("dock-toggle", "n_clicks"), 2,
             [("dock-collapsed", "data", True)])
    assert r["dock-collapsed"]["data"] is False
    assert post("dock.className", ("dock-collapsed", "data"), False)["dock"]["className"] == "dock"


def test_resize_handle():
    app = build_app(MockBackend(), Session(tempfile.mkdtemp(prefix="demo_gui_dock_")), Cfg())
    client = app.server.test_client()
    js = client.get("/assets/dock_resize.js")
    assert js.status_code == 200 and b".dock__handle" in js.data and b"localStorage" in js.data
    assert b"dock_resize.js" in client.get("/").data  # Dash includes assets/*.js in the page
    dock = app.layout.children[-1]
    assert any(getattr(c, "className", "") == "dock__handle" for c in dock.children)
    css = client.get("/assets/style.css").get_data(as_text=True)
    assert "--dock-w: 520px" in css and "width: min(var(--dock-w), 70vw)" in css
    assert "--pad: calc(min(var(--dock-w), 70vw) + 20px); padding-right: var(--pad)" in css
    assert css.index(".dock--collapsed { width: 36px") > css.index(".dock {")  # collapsed wins


def _harness():
    spec = importlib.util.spec_from_file_location("eval_learned_mpc_under_test", HARNESS)
    h = sys.modules[spec.name] = importlib.util.module_from_spec(spec)  # dataclasses need it
    sys.path.insert(0, str(REPO / "scripts"))  # a copy outside the repo misses the harness's own
    spec.loader.exec_module(h)
    return h


class _Sim:
    frame_dt = 0.01

    def __init__(self) -> None:
        self.step_index, self.renders = 0, 0
        self.bridge = types.SimpleNamespace(stats=dict, controller_stats=dict, hand_commands=[],
                                            robotiq_commands=[])
        self.state_0 = types.SimpleNamespace(body_q=types.SimpleNamespace(
            numpy=lambda: np.zeros((4, 7))))
        self.info = types.SimpleNamespace(belt_bodies=[0, 1], pulley_bodies=[2, 3])

    def control_step(self) -> None:
        self.step_index += 1

    def render(self) -> None:
        self.renders += 1

    def franka_measured_pose7(self):
        return np.zeros(7)

    def arm_targets(self):
        return None, None

    def make_hold_hook(self):
        return None

    def grasp_state(self, *_):
        return types.SimpleNamespace(held_for=lambda arm: False)  # -> return before stage 2


def _ur_exit(h, live: bool) -> _Sim:
    """``_chain_stage2`` through the UR exit (stubbed release / exit / sampling) on a fake sim."""
    sim = _Sim()
    if live:  # the harness's --live-viewer wrap, render_every 1
        step = sim.control_step
        sim.control_step = lambda: (step(), sim.step_index % 1 or sim.render())
    steps = lambda k: [sim.control_step() for _ in range(k)]
    sys.modules["phase2_probe"] = types.SimpleNamespace(
        release=lambda *a: (steps(7), ({"label_end": "x", "wrap_min": 0.0}, None))[1])
    sys.modules["stage2_starts"] = types.SimpleNamespace(ur_exit=lambda *a: (steps(5), (None, None))[1])
    h._label_row = lambda ctx, row: {"large": "engaged", "large_wrap_deg": 0.0}
    h._sample = lambda sim, *a, **k: {"sim_step": sim.step_index, "time": 0.0}
    h.pose7_mat, h.UrTracking = (lambda m: None), types.SimpleNamespace(fk=lambda q: None)
    h.frame_metrics = lambda *a, **k: types.SimpleNamespace(wrap_deg=0.0)
    h.classify = lambda *a: "engaged"
    args = types.SimpleNamespace(ur_exit_byte=0, ur_exit_settle_s=0.0, ur_exit_lift_mm=0.0,
                                 ur_exit_speed_mps=0.0, ur_exit_final_settle_s=0.04)
    ctx = types.SimpleNamespace(args=args, chain={"teleop_s1": True, "pp": {"ur9": object()}},
                                phase_labels=["ur:open", "ur:park", "ur:lift", "ur:settle"],
                                thresholds=None)
    frames = [{"step": 0, "sim_step": 0}]
    out = h._chain_stage2(sim, ctx, None, frames, 0, 0, {"status": "ok", "reason": ""}, "t", {})
    assert out["report"].get("ur_exit", {}).get("failed") and sim.step_index >= 16, (out["report"], sim.step_index)
    return sim


def test_render_wrap_survives_ur_exit():
    h = _harness()
    sim = _ur_exit(h, live=True)
    assert "control_step" in vars(sim), "render wrap dropped by the UR exit"
    sim.control_step()  # a stage-2 step
    assert sim.renders == sim.step_index, (sim.renders, sim.step_index)
    sim = _ur_exit(h, live=False)  # no --live-viewer: the class method, as before
    assert "control_step" not in vars(sim) and sim.renders == 0
    assert sim.control_step.__func__ is _Sim.control_step


if __name__ == "__main__":
    for f in (test_dock_source_rule, test_backends_alike, test_docs_launch_command,
              test_one_viewer_and_toggle, test_resize_handle, test_render_wrap_survives_ur_exit):
        f()
        print(f"ok {f.__name__}")
