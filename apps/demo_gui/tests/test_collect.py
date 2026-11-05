"""Collect panel checks, CPU-only (mock backend, Flask test client, direct callback calls).

uv run --frozen python apps/demo_gui/tests/test_collect.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.demo_gui import grasp2d, jobs
from apps.demo_gui.app import Ctx, build_app
from apps.demo_gui.backends.mock import MockBackend
from apps.demo_gui.config import Cfg
from apps.demo_gui.panels import collect
from apps.demo_gui.session import Session, read_index
from dash import no_update
from dash.exceptions import PreventUpdate

_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_collect_")  # removed at exit


def _session() -> Session:
    return Session(tempfile.mkdtemp(dir=_TMP.name))


def _until(cond, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _drag(arm: str, slide_from: float, along_mm: float, up_mm: float = 0.0) -> dict:
    """relayoutData for the side-view fingers moved ``along_mm`` along / ``up_mm`` up."""
    rings = grasp2d._rings(grasp2d.load_geom()[arm]["silhouette_A"], dx=slide_from + along_mm)
    return {"shapes[0].path": grasp2d.svg_path([r + [0.0, up_mm] for r in rings])}


def _drop(arm: str, slide: float, depth: float, pivot: float = 0.0) -> dict:
    """relayoutData of the fingers dropped at (slide, depth) at ``pivot``: turned, slid, then moved
    along the turned ``approach_A``."""
    g = grasp2d.load_geom()[arm]
    a = grasp2d._rings([[g["approach_A"]]], deg=-pivot)[0][0]
    rings = grasp2d._rings(g["silhouette_A"], dx=slide, deg=-pivot)
    return {"shapes[0].path": grasp2d.svg_path([r + depth * a for r in rings])}


def _handle(deg: float, slide: float = 0.0) -> dict:
    a = math.radians(grasp2d.HANDLE_DEG - deg)
    x, y, k = slide + 35 * math.cos(a), 35 * math.sin(a), grasp2d.HANDLE_MM
    return {"shapes[1].x0": x - k, "shapes[1].x1": x + k, "shapes[1].y0": y - k,
            "shapes[1].y1": y + k}


DEFAULT = collect.DEFAULT
REGION = "collision-free grasp point region"


def _trace(fig, name: str):
    data = fig["data"] if isinstance(fig, dict) else fig.data
    return next(t for t in data if (t["name"] if isinstance(t, dict) else t.name) == name)


def test_drag_slide():
    out = collect.on_grasp("col-canvas-franka", _drag("franka", 0.0, 10.0), None, 0, -5, None,
                           None)
    state, fig_f, txt_f = out[0], out[1], out[3]
    assert len(out) == 13 and abs(state["franka"]["slide_mm"] - 10.0) < 0.15, state
    assert state["ur"] == DEFAULT["ur"]  # untriggered slider ignored
    assert all(out[k] is no_update for k in (2, 4, 6, 8, 10, 12))
    assert out[5] is no_update  # slider already at the store's pivot
    assert (out[7], out[9]) == tuple(grasp2d.pivot_interval("franka", 10.0, 0.0, 0.0))
    assert txt_f == "slide +10.0 mm, angle +0 deg, depth +0.0 mm", txt_f
    x = grasp2d.path_mean(fig_f.layout.shapes[0].path)[0] - grasp2d._base("franka", 0.0)[0]
    assert abs(x - state["franka"]["slide_mm"]) < 0.05 and fig_f.layout.dragmode is False
    # the stale relayoutData of the other canvas is ignored; a far drag clamps to the region
    out = collect.on_grasp("col-canvas-ur", _drag("franka", 0.0, 10.0), _drag("ur", 0.0, -60.0),
                           3, 0, None, state)
    state = out[0]
    want = grasp2d.clamp("ur", -60.0, 0.0, 0.0)
    assert state["franka"]["slide_mm"] == 10.0 and out[1] is no_update
    assert tuple(state["ur"][k] for k in grasp2d.KNOBS) == want[:3], (state["ur"], want)
    assert "slide_mm" in want[3] and "(AT LIMIT)" in out[4]
    assert _trace(out[2], REGION).line.width == 3.5  # clipped slide -> region bold


def test_handle_and_slider_pivot():
    out = collect.on_grasp("col-canvas-franka", _handle(-3.0), None, 0, 0, None, None)
    assert out[0]["franka"]["pivot_deg"] == -3.0 and out[5] == -3.0 and out[6] is no_update
    assert "angle -3 deg" in out[3] and out[0]["franka"]["roll_deg"] == 0.0
    out = collect.on_grasp("col-canvas-franka", _handle(170.0), None, -3, 0, None, out[0])
    assert out[0]["franka"]["pivot_deg"] == 20.0 == out[5] and "angle +20 deg (AT LIMIT)" in out[3]
    out2 = collect.on_grasp("col-canvas-ur", None, _handle(35.0), 0, 0, None, None)
    u, hi = out2[0]["ur"], grasp2d.pivot_interval("ur", 0.0, 0.0, DEFAULT["ur"]["depth_mm"])[1]
    assert u == {**DEFAULT["ur"], "pivot_deg": hi} and f"angle {hi:+.0f} deg (AT LIMIT)" in out2[4]
    assert hi < 20 and out2[10] == hi  # a turn keeps slide / depth: the pivot stops at its run
    out = collect.on_grasp("col-pivot-ur", None, None, 20, 7, None, out[0])
    assert out[0]["ur"]["pivot_deg"] == 7.0 and out[6] is no_update and out[1] is no_update
    assert out[0]["franka"]["pivot_deg"] == 20.0
    fig = out[2]
    assert len(fig.layout.shapes) == 2 and fig.layout.shapes[1].type == "circle"
    # a slid + pivoted canvas: the handle is read about the slid grasp point
    st = {**DEFAULT, "franka": {"slide_mm": 8.0, "roll_deg": 0.0, "pivot_deg": 0.0}}
    out = collect.on_grasp("col-canvas-franka", _handle(12.0, slide=8.0), None, 0, 0, None, st)
    assert out[0]["franka"] == {"slide_mm": 8.0, "roll_deg": 0.0, "pivot_deg": 12.0,
                                "depth_mm": 0.0}, out[0]  # a store without depth_mm reads as 0
    rot = grasp2d.svg_path(grasp2d.place(grasp2d.load_geom()["franka"]["silhouette_A"], 2.0, 12.0))
    out = collect.on_grasp("col-canvas-franka", {"shapes[0].path": rot}, None, 12, 0, None,
                           out[0])
    assert abs(out[0]["franka"]["slide_mm"] - 2.0) < 0.15 and out[0]["franka"]["pivot_deg"] == 12


def test_reset_and_load_snap():
    moved = {**DEFAULT, "ur": {**DEFAULT["ur"], "slide_mm": 3.0, "pivot_deg": 6.0,
                               "depth_mm": 8.5}}
    out = collect.on_grasp("col-reset", None, None, 0, 6, 1, moved)
    assert out[0] == DEFAULT and out[1] is not no_update and out[2] is not no_update
    assert out[6] == DEFAULT["ur"]["pivot_deg"] and out[5] is no_update
    try:  # page load with a feasible store: nothing to do
        collect.on_grasp(None, None, None, 0, 0, None, DEFAULT)
        raise AssertionError("expected PreventUpdate")
    except PreventUpdate:
        pass
    old = {arm: {"slide_mm": 0.0, "roll_deg": 0.0, "pivot_deg": 0.0} for arm in collect.ARMS}
    out = collect.on_grasp(None, None, None, 0, 0, None, old)  # UR nominal is inside the plate
    assert out[0]["franka"] == old["franka"] and out[1] is no_update
    u = out[0]["ur"]
    assert tuple(u[k] for k in grasp2d.KNOBS) == grasp2d.clamp("ur", 0.0, 0.0, 0.0)[:3]
    assert grasp2d.is_feasible("ur", u["slide_mm"], u["pivot_deg"], u["depth_mm"])
    assert "(AT LIMIT)" in out[4]


def _grasp_post(client, cb: tuple, values: dict, state: dict, changed: str):
    """POST the grasp callback like the browser does -> (status, response dict or {})."""
    key, cb = cb
    outs = [o.rsplit(".", 1) for o in key.strip(".").split("...")]
    inputs = [{**i, "value": values.get(i["id"])} for i in cb["inputs"]]
    body = {"output": key, "outputs": [{"id": i, "property": p} for i, p in outs],
            "inputs": inputs, "state": [{**cb["state"][0], "value": state}],
            "changedPropIds": [changed] if changed else []}
    r = client.post("/_dash-update-component", json=body)
    assert r.status_code in (200, 204), r.data[:300]
    return r.status_code, json.loads(r.data)["response"] if r.status_code == 200 else {}


def test_drag_formats():
    """The real relayoutData formats of a finger / handle drag (the snap-back reproducer)."""
    app = build_app(MockBackend(), _session(), Cfg())
    client = app.server.test_client()
    cb = next(kv for kv in app.callback_map.items() if "col-grasp.data" in kv[0])
    slider = next(i["id"] for i in cb[1]["inputs"] if i["property"] == "value"
                  and i["id"].endswith("-franka"))
    zero = DEFAULT
    hi = 30.0
    fails = []

    def case(name, ok):
        print(f"  {name}: {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)

    _, r = _grasp_post(client, cb, {"col-canvas-franka": _drag("franka", 0.0, 6.0)}, zero,
                       "col-canvas-franka.relayoutData")
    dragged = r.get("col-grasp", {}).get("data", zero)
    case("a path drag", abs(dragged["franka"]["slide_mm"] - 6.0) < 0.15)

    _, r = _grasp_post(client, cb, {"col-canvas-franka": {"autosize": True}}, dragged,
                        "col-canvas-franka.relayoutData")
    case("b autosize after drag", "col-canvas-franka" not in r and "col-canvas-ur" not in r
         and r.get("col-grasp", {}).get("data", dragged) == dragged)

    _, r = _grasp_post(client, cb, {"col-canvas-franka": {"shapes[2].path": "M 0,0 L 1,1 Z"}},
                        dragged, "col-canvas-franka.relayoutData")
    case("c non-interactive shape", not r)

    a = math.radians(5.0)
    cx, cy, k = 6.0 - 35 * math.sin(a), -35 * math.cos(a), 3.5  # +5 deg about the slid point
    full = {"shapes[1].x0": cx - k, "shapes[1].x1": cx + k, "shapes[1].y0": cy - k,
            "shapes[1].y1": cy + k}
    for name, ev in (("d handle x0..y1", full),
                     ("d handle x0,y0 only", {q: full[q] for q in ("shapes[1].x0",
                                                                   "shapes[1].y0")})):
        _, r = _grasp_post(client, cb, {"col-canvas-franka": ev}, dragged,
                           "col-canvas-franka.relayoutData")
        f = r.get("col-grasp", {}).get("data", dragged)["franka"]
        case(name, f.get("pivot_deg") == 5.0 and abs(f["slide_mm"] - 6.0) < 0.15)

    _, r = _grasp_post(client, cb, {"col-canvas-franka": _drag("franka", 0.0, hi + 20.0)}, zero,
                       "col-canvas-franka.relayoutData")
    txt = r.get("col-readout-franka", {}).get("children", "")
    want = grasp2d.clamp("franka", hi, 0.0, 0.0)
    case("e past the limit", r["col-grasp"]["data"]["franka"]["slide_mm"] == want[0] < hi
         and f"slide {want[0]:+.1f} mm (AT LIMIT)" in txt)

    _, r = _grasp_post(client, cb, {slider: 0}, zero, f"{slider}.value")
    case("f slider == store", not r)

    def drop(arm, ev, state, name, slide, depth, centre_mm=0.05):
        """Post a finger drop; the store and the redrawn fingers must land where dropped."""
        _, r = _grasp_post(client, cb, {f"col-canvas-{arm}": ev}, state,
                           f"col-canvas-{arm}.relayoutData")
        st = r.get("col-grasp", {}).get("data", state)[arm]
        fig = r.get(f"col-canvas-{arm}", {}).get("figure", {"layout": {"shapes": [{"path": ""}]}})
        got = fig["layout"]["shapes"][0]["path"]
        err = (float(abs(grasp2d.path_mean(got) - grasp2d.path_mean(ev["shapes[0].path"])).max())
               if got else math.inf)
        ok = (st["slide_mm"] == slide and st.get("depth_mm") == depth and err < centre_mm)
        print(f"    {name}: slide {st['slide_mm']}, depth {st.get('depth_mm')}, "
              f"redrawn centre off the drop by {err:.3f} mm")
        case(name, ok)
        return r

    piv = {**zero, "ur": {**zero["ur"], "pivot_deg": 20.0}}
    r = drop("franka", _drop("franka", 0.0, -3.0), zero, "g down along the fingers", 0.0, -3.0)
    case("g readout deeper", r.get("col-readout-franka", {}).get("children", "").endswith(
        "depth -3.0 mm (deeper)"))
    r = drop("ur", _drop("ur", 3.0, 8.5), zero, "h diagonal", 3.0, 8.5)
    case("h readout shallower", "depth +8.5 mm (shallower)" in r["col-readout-ur"]["children"])
    st = r["col-grasp"]["data"]
    a = math.radians(grasp2d.HANDLE_DEG - 6.0)
    _, r = _grasp_post(client, cb, {"col-canvas-ur": {"shapes[1].x0": 3.0 + 35 * math.cos(a) - 3.5,
                                                      "shapes[1].y0": 35 * math.sin(a) - 3.5}},
                       st, "col-canvas-ur.relayoutData")
    u = r.get("col-grasp", {}).get("data", st)["ur"]
    case("h then handle x0,y0 only", u["pivot_deg"] == 6.0 and u["slide_mm"] == 3.0
         and u["depth_mm"] == 8.5)
    piv = {**zero, "ur": {**zero["ur"], "pivot_deg": 10.0}}
    drop("ur", _drop("ur", 2.0, 8.0, 10.0), piv, "i pivot 10 deg drop", 2.0, 8.0)
    up = {"shapes[0].path": grasp2d.svg_path(
        [r + [0.0, -3.0] for r in grasp2d._rings(grasp2d.load_geom()["franka"]["silhouette_A"])])}
    drop("franka", up, zero, "j screen-vertical drop (within the 0.1 / 0.5 mm steps)", 0.1, -3.0,
         0.3)
    old = {arm: {k: v for k, v in zero[arm].items() if k != "depth_mm"} for arm in collect.ARMS}
    drop("ur", _drop("ur", 1.0, 8.0), old, "k old store without depth_mm", 1.0, 8.0)
    # past a depth limit: deeper = into the holder (3D table), shallower = belt-in-jaw rule
    for arm, depth in (("franka", -20.0), ("franka", 3.0), ("ur", 20.0), ("ur", -5.0)):
        _, r = _grasp_post(client, cb, {f"col-canvas-{arm}": _drop(arm, 0.0, depth)}, zero,
                           f"col-canvas-{arm}.relayoutData")
        st = r.get("col-grasp", {}).get("data", zero)[arm]
        txt = r.get(f"col-readout-{arm}", {}).get("children", "")
        fig = r.get(f"col-canvas-{arm}", {}).get("figure", {"data": []})
        want = grasp2d.clamp(arm, 0.0, 0.0, depth)
        reg = [t for t in fig["data"] if t.get("name") == REGION]
        case(f"l {arm} past depth {depth:+g} -> {want[:3]}",
             tuple(st.get(k) for k in grasp2d.KNOBS) == want[:3] and "depth_mm" in want[3]
             and grasp2d.is_feasible(arm, *want[:3]) and "(AT LIMIT)" in txt
             and len(reg) == 1 and reg[0]["line"]["width"] == 3.5 and len(reg[0]["x"]) > 8)
    lo, hi = grasp2d.pivot_interval("ur", 3.0, 6.0, 8.5)
    _, r = _grasp_post(client, cb, {"col-canvas-ur": _drop("ur", 3.0, 8.5, 6.0)}, piv,
                       "col-canvas-ur.relayoutData")
    case("m slider bounds = pivot run at the new pose", r.get("col-pivot-ur", {}).get("min") == lo
         and r["col-pivot-ur"].get("max") == hi and lo < 6.0 < hi)
    ur_slider = slider.replace("franka", "ur")
    _, r = _grasp_post(client, cb, {ur_slider: 20}, zero, f"{ur_slider}.value")
    u = r.get("col-grasp", {}).get("data", zero)["ur"]
    case("n slider outside the run clamps", grasp2d.is_feasible(
        "ur", u["slide_mm"], u["pivot_deg"], u["depth_mm"]) and "(AT LIMIT)" in r.get(
        "col-readout-ur", {}).get("children", ""))
    old = {arm: {"slide_mm": 0.0, "roll_deg": 0.0, "pivot_deg": 0.0, "depth_mm": 0.0}
           for arm in collect.ARMS}
    _, r = _grasp_post(client, cb, {}, old, None)
    u = r.get("col-grasp", {}).get("data", old)["ur"]
    case("o page load snaps an infeasible old store", "col-canvas-franka" not in r
         and grasp2d.is_feasible("ur", u["slide_mm"], u["pivot_deg"], u["depth_mm"]))
    _, r = _grasp_post(client, cb, {}, zero, None)
    case("p page load, default store: no update", not r)
    _, r = _grasp_post(client, cb, {"col-reset": 1}, piv, "col-reset.submit_n_clicks")
    case("q reset -> defaults", r.get("col-grasp", {}).get("data") == DEFAULT)
    assert not fails, f"drag formats failing: {fails}"


def test_page_and_callback_http():
    app = build_app(MockBackend(), _session(), Cfg())
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert b"col-canvas-franka" in client.get("/_dash-layout").data
    outs = [
        ("col-grasp", "data"),
        *((f"col-canvas-{a}", "figure") for a in collect.ARMS),
        *((f"col-readout-{a}", "children") for a in collect.ARMS),
        *((f"col-pivot-{a}", p) for p in ("value", "min", "max", "marks") for a in collect.ARMS),
    ]
    body = {
        "output": ".." + "...".join(f"{i}.{p}" for i, p in outs) + "..",
        "outputs": [{"id": i, "property": p} for i, p in outs],
        "inputs": [
            {"id": "col-canvas-franka", "property": "relayoutData",
             "value": _drag("franka", 0.0, -8.0)},
            {"id": "col-canvas-ur", "property": "relayoutData", "value": None},
            {"id": "col-pivot-franka", "property": "value", "value": 4},
            {"id": "col-pivot-ur", "property": "value", "value": 0},
            {"id": "col-reset", "property": "submit_n_clicks", "value": None},
        ],
        "state": [{"id": "col-grasp", "property": "data", "value": DEFAULT}],
        "changedPropIds": ["col-canvas-franka.relayoutData"],
    }
    r = client.post("/_dash-update-component", json=body)
    assert r.status_code == 200, r.data[:300]
    resp = json.loads(r.data)["response"]
    data = resp["col-grasp"]["data"]
    assert abs(data["franka"]["slide_mm"] + 8.0) < 0.15 and data["franka"]["pivot_deg"] == 0
    assert resp["col-pivot-franka"]["value"] == 0 and "col-canvas-ur" not in resp


def test_mock_flow():
    s = _session()
    ctx = Ctx(MockBackend(pace=0.02), s, Cfg())
    collect.layout(ctx)
    assert collect.on_status(ctx)[:2] == (True, True)
    state = {
        "franka": {"slide_mm": 12.0, "roll_deg": 0.0, "pivot_deg": -5.0},
        "ur": {"slide_mm": -3.5, "roll_deg": 0.0, "pivot_deg": 0.0},
    }
    assert collect.on_action(ctx, "col-make", state).startswith("make_start")
    assert _until(lambda: read_index(s.starts), 5.0)
    assert read_index(s.starts)[0]["commanded"] == {
        "franka_slide_mm": 12.0,
        "franka_roll_deg": 0.0,
        "ur_slide_mm": -3.5,
        "ur_roll_deg": 0.0,
        "franka_pivot_deg": -5.0,
    }
    rec_off, teleop_off, start, _ = collect.on_status(ctx)
    assert not rec_off and not teleop_off and "s00" in start
    assert _until(lambda: "done" in collect.on_status(ctx)[3], 5.0)

    assert "recording scripted" in collect.on_action(ctx, "col-record", state)
    demo = s.demos / "demo_00"
    assert _until(lambda: (demo / "index.json").is_file(), 20.0)
    assert _until(lambda: s.telemetry_path.is_file(), 20.0)
    buf, cloud, pose_f, _, grip_f, _ = collect.on_live(ctx, None)
    assert [len(tr.x) > 0 for tr in cloud.data] == [True, True]  # cloud + belt
    assert len(pose_f.data) == 3 and len(buf["t"]) == 1 and "width" in grip_f.layout.title.text
    assert _until(lambda: _live_changed(ctx, buf), 5.0)

    collect.on_action(ctx, "col-stop", state)
    assert jobs.status(jobs.load(s.jobs / "demo.json")) == "done"
    try:
        buf = collect.on_live(ctx, buf)[0]
    except PreventUpdate:
        pass
    try:
        collect.on_live(ctx, buf)
        raise AssertionError("telemetry still updating after Stop")
    except PreventUpdate:
        pass

    assert demo.exists()
    collect.on_action(ctx, "col-reset", state)
    assert not demo.exists()


def test_default_start():
    """The store starts at the table's default pose (UR shallower: its nominal pick is inside the
    holder plate) and Make start passes it through to the backend."""
    s = _session()
    ctx = Ctx(MockBackend(pace=0.02), s, Cfg())
    store = next(c for c in collect.layout(ctx)._traverse() if getattr(c, "id", "") == "col-grasp")
    g = grasp2d.load_geom()
    assert store.data == DEFAULT
    for arm in collect.ARMS:
        d = DEFAULT[arm]
        assert {k: d[k] for k in grasp2d.KNOBS} == g[arm]["default_pose"]
        assert grasp2d.is_feasible(arm, *(d[k] for k in grasp2d.KNOBS)) and d["roll_deg"] == 0.0
    assert DEFAULT["franka"]["depth_mm"] == 0.0 < DEFAULT["ur"]["depth_mm"]
    assert not grasp2d.is_feasible("ur", 0.0, 0.0, 0.0)
    assert collect.on_action(ctx, "col-make", None).startswith("make_start")
    assert _until(lambda: read_index(s.starts), 5.0)
    assert read_index(s.starts)[0]["commanded"]["ur_depth_mm"] == DEFAULT["ur"]["depth_mm"]


def _live_changed(ctx, buf) -> bool:
    try:
        return len(collect.on_live(ctx, json.loads(json.dumps(buf)))[0]["t"]) == 2
    except PreventUpdate:
        return False


def test_cleanup():
    ctx = Ctx(MockBackend(), _session(), Cfg())
    tree = str(collect.layout(ctx))
    assert "Details(" not in tree and "col-demos" not in tree and "verified" not in tree
    assert "SpaceMouse driver:" in tree and len(collect.on_status(ctx)) == 4
    sl = {c.id: c for c in collect.layout(ctx)._traverse() if str(getattr(c, "id", "")).startswith("col-pivot-")}
    assert len(sl) == 2 and all(-20 <= c.min < c.max <= 20 for c in sl.values())
    for arm in collect.ARMS:
        lo, hi, _ = collect._slider(arm, DEFAULT[arm])
        assert -20 <= lo < hi <= 20 and (sl[f"col-pivot-{arm}"].min, sl[f"col-pivot-{arm}"].max) == (
            lo, hi)
        fig = grasp2d.figure(arm, *(DEFAULT[arm][k] for k in grasp2d.KNOBS))
        text = " ".join(str(t.name) + str(getattr(t, "text", "")) for t in fig.data).lower()
        assert "verified" not in text and "probed" not in text
        assert sum((t.name or "").startswith("pivot limit") for t in fig.data) == 2


def test_teleop_disabled_without_support():
    class NoTeleop(MockBackend):
        supports_teleop = False

    s = _session()
    ctx = Ctx(NoTeleop(), s, Cfg())
    page = str(collect.layout(ctx))
    assert "needs PKG-04 / SpaceMouse" in page
    (s.starts / "index.json").write_text(json.dumps({"variants": [{"id": "s00"}]}))
    assert collect.on_status(ctx)[:2] == (False, True)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
