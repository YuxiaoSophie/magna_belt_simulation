"""Tab 1, data collection: grasp canvases + pivot sliders, start/record/stop/reset, live row."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate

from apps.demo_gui import grasp2d, jobs, telemetry
from apps.demo_gui.session import read_index

ARMS = ("franka", "ur")
NAMES = {"franka": "Franka", "ur": "UR"}
BUFFER_S = 30.0
DEFAULT = {arm: grasp2d.default_pose(arm) for arm in ARMS}  # roll fixed; UR shallower (holder)
TELEOP_KEYS = {"col-tdone": ("d", "stage 1 done (UR exit, then stage-2 teleop)"),
               "col-tpause": ("p", "pause / resume")}


def readout(arm: str, st: dict, at_limit: dict | None = None) -> str:
    """``slide +x mm, angle +p deg, depth -d mm (deeper)``, ``(AT LIMIT)`` after a clipped knob."""
    at, d = at_limit or {}, st.get("depth_mm", 0.0)
    parts = [f"slide {st['slide_mm']:+.1f} mm", f"angle {st['pivot_deg']:+.0f} deg",
             f"depth {d:+.1f} mm" + (" (deeper)" if d < 0 else " (shallower)" if d > 0 else "")]
    return ", ".join(p + (" (AT LIMIT)" if k in at else "")
                     for p, k in zip(parts, ("slide_mm", "pivot_deg", "depth_mm")))


def _slider(arm: str, st: dict) -> tuple:
    """(min, max, marks) of the pivot slider = the feasible pivot run at the pose's slide / depth."""
    lo, hi = grasp2d.pivot_interval(arm, st["slide_mm"], st["pivot_deg"], st["depth_mm"])
    return lo, hi, {v: str(int(v)) for v in sorted({lo, hi, *range(-20, 21, 10)}) if lo <= v <= hi}


def on_grasp(trigger: str | None, ra_f, ra_u, piv_f, piv_u, reset, state: dict | None):
    """-> (state, figs x2, readouts x2, pivot slider values x2, (min, max, marks) x2). Only the
    triggering arm's outputs change; zoom / autosize / unknown shapes / an unchanged slider ->
    PreventUpdate (a figure rebuilt for those resets the canvas: the snap-back). Reset -> both arms
    to ``DEFAULT``; the first call (page load) snaps an infeasible stored pose (older session)."""
    state = {arm: dict(v) for arm, v in (state or DEFAULT).items()}
    out = {arm: [no_update] * 6 for arm in ARMS}
    for arm, ra, piv in (("franka", ra_f, piv_f), ("ur", ra_u, piv_u)):
        st = state[arm] = dict(DEFAULT[arm]) if trigger == "col-reset" else state[arm]
        slide, pivot, depth = st["slide_mm"], st["pivot_deg"], st.get("depth_mm", 0.0)
        if trigger in (None, "col-reset"):
            if trigger is None and grasp2d.clamp(arm, slide, pivot, depth)[:3] == (
                    slide, pivot, depth):
                continue
        elif trigger == f"col-canvas-{arm}":
            v = grasp2d.drag_from_relayout(arm, ra, pivot)
            w = grasp2d.pivot_from_relayout(arm, ra, slide)
            if v is None and w is None:
                continue
            slide, depth = (slide, depth) if v is None else v
            pivot = pivot if w is None else w
        elif trigger == f"col-pivot-{arm}" and piv is not None and piv != pivot:
            pivot = piv
        else:
            continue
        turn = pivot != st["pivot_deg"] and trigger != "col-reset"
        st["slide_mm"], st["pivot_deg"], st["depth_mm"], at = grasp2d.clamp(arm, slide, pivot,
                                                                            depth, turn)
        out[arm] = [grasp2d.figure(arm, st["slide_mm"], st["pivot_deg"], st["depth_mm"], at),
                    readout(arm, st, at), no_update if st["pivot_deg"] == piv else st["pivot_deg"],
                    *_slider(arm, st)]
    if all(o[0] is no_update for o in out.values()):
        raise PreventUpdate
    return state, *(out[a][i] for i in range(6) for a in ARMS)


def driver_command(url: str) -> str:
    """docs/teleop-stage2.md section 3 on the app's LCM URL."""
    return ("~/Workspace/spacemouse_driver/bazel-bin/core/run_franka_driver \\\n"
            f"  --lcm_url='{url}' \\\n  --gripper_command_lcm_channel=TELEOP_HAND_IGNORED")


def on_action(ctx, trigger: str | None, state: dict | None) -> str:
    """Button dispatch -> status line."""
    be, s, state = ctx.backend, ctx.session, state or DEFAULT
    if trigger == "col-make":
        job = be.make_start(s, franka=state["franka"], ur=state["ur"])
        return f"{job.name}: started ({readout('franka', state['franka'])} | {readout('ur', state['ur'])})"
    if trigger in ("col-record", "col-teleop"):
        rows = [r for r in read_index(s.starts) if r.get("held", True)]
        if not rows:
            return "no held start state yet: press Make start first"
        mode = "scripted" if trigger == "col-record" else "teleop"
        job = be.record_demo(s, rows[-1]["id"], mode=mode)
        return f"{job.name}: recording {mode} demo from start {rows[-1]['id']}"
    if trigger in TELEOP_KEYS:
        send = getattr(be, "teleop_key", None)
        if send is None:
            return "teleop keys: not on this backend"
        send(s, TELEOP_KEYS[trigger][0])
        return f"sent {TELEOP_KEYS[trigger][0]}: {TELEOP_KEYS[trigger][1]}"
    if trigger == "col-stop":
        be.stop_demo(s)
        return "stopped"
    if trigger == "col-reset":
        be.reset_demo(s)
        return "reset: in-progress demo deleted"
    raise PreventUpdate


def on_status(ctx) -> tuple:
    """-> (record disabled, teleop disabled, start text, make_start job text)."""
    s = ctx.session
    rows = read_index(s.starts)
    p = s.jobs / "make_start.json"
    job = f"make_start: {jobs.status(jobs.load(p))}" if p.is_file() else ""
    start = f"start {rows[-1]['id']} ready ({len(rows)} made)" if rows else "no start state yet"
    if rows and rows[-1].get("held") is False:
        start = f"start {rows[-1]['id']} not held: {rows[-1].get('reason')}"
    teleop = getattr(ctx.backend, "supports_teleop", False)
    return not rows, not (rows and teleop), start, job


def _ranges(tel: dict) -> list:
    pts = np.array([p for k in ("belt_xyz", "cloud") for p in (tel.get(k) or [])]
                   + [tel[k][:3] for k in ("franka_pose7", "ur_pose7") if tel.get(k)], dtype=float)
    if not len(pts):
        return [[-1.0, 1.0]] * 3
    lo, hi = pts.min(0) - 0.05, pts.max(0) + 0.05
    return [[float(a), float(b)] for a, b in zip(lo, hi)]


def cloud_figure(tel: dict, ranges: list) -> go.Figure:
    fig = go.Figure()
    for k, color, size in (("cloud", "#999", 2), ("belt_xyz", "#d62728", 4)):
        p = np.array(tel.get(k) or np.zeros((0, 3)), dtype=float).reshape(-1, 3)
        fig.add_trace(
            go.Scatter3d(
                x=p[:, 0],
                y=p[:, 1],
                z=p[:, 2],
                mode="markers",
                name=k,
                marker={"size": size, "color": color},
            )
        )
    fig.update_layout(
        scene={f"{a}axis": {"range": r} for a, r in zip("xyz", ranges)} | {"aspectmode": "cube"},
        uirevision="cloud",
        height=320,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title={
            "text": f"t {tel.get('t', 0):.1f} s, phase {tel.get('phase')}",
            "font": {"size": 13},
        },
    )
    return fig


def pose_figure(buf: dict, arm: str) -> go.Figure:
    p = np.array([[np.nan] * 3 if v is None else v for v in buf[arm]], dtype=float).reshape(-1, 3)
    fig = go.Figure(
        [
            go.Scatter(x=buf["t"], y=p[:, i], mode="lines", name=f"{a} mm")
            for i, a in enumerate("xyz")
        ]
    )
    fig.update_layout(
        title={"text": f"{NAMES[arm]} EE position", "font": {"size": 13}},
        height=220,
        margin={"l": 40, "r": 10, "t": 30, "b": 30},
        xaxis={"title": "t s"},
        uirevision=arm,
    )
    return fig


def gripper_figure(arm: str, width_mm, grip_n) -> go.Figure:
    w = width_mm or 0.0
    fingers = [
        {
            "type": "rect",
            "x0": x0,
            "x1": x0 + 8,
            "y0": 0,
            "y1": 30,
            "xref": "x",
            "yref": "y",
            "fillcolor": "#555",
            "line": {"width": 0},
        }
        for x0 in (-w / 2 - 8, w / 2)
    ]
    fig = go.Figure(
        go.Bar(x=["N"], y=[grip_n or 0.0], xaxis="x2", yaxis="y2", marker={"color": "#ff7f0e"})
    )
    num = lambda v, u: "n/a" if v is None else f"{v:.1f} {u}"
    fig.update_layout(
        shapes=fingers,
        height=180,
        showlegend=False,
        margin={"l": 10, "r": 10, "t": 30, "b": 20},
        title={
            "text": f"{NAMES[arm]}: width {num(width_mm, 'mm')}, grip {num(grip_n, 'N')}",
            "font": {"size": 12},
        },
        xaxis={"range": [-55, 55], "domain": [0, 0.7], "visible": False},
        yaxis={"range": [-5, 35], "visible": False},
        xaxis2={"domain": [0.8, 1], "anchor": "y2"},
        yaxis2={"range": [0, 150], "anchor": "x2", "title": "N"},
    )
    return fig


def on_live(ctx, buf: dict | None) -> tuple:
    """-> (buffer, cloud, pose_franka, pose_ur, grip_franka, grip_ur); PreventUpdate if stale."""
    tel = telemetry.read(ctx.session.telemetry_path)
    if not tel:
        raise PreventUpdate
    key = [tel.get("run_dir"), tel.get("step"), tel.get("t")]
    if buf and buf["key"] == key:
        raise PreventUpdate
    t = float(tel.get("t") or 0.0)
    if not buf or buf["key"][0] != key[0] or t < buf["t"][-1]:
        buf = {"t": [], "franka": [], "ur": [], "ranges": _ranges(tel)}
    buf["key"] = key
    buf["t"].append(t)
    for arm in ARMS:
        pose = tel.get(f"{arm}_pose7")
        buf[arm].append(None if pose is None else [round(v * 1e3, 1) for v in pose[:3]])
    keep = next(i for i, ti in enumerate(buf["t"]) if ti >= t - BUFFER_S)
    for k in ("t", *ARMS):
        buf[k] = buf[k][keep:]
    return (
        buf,
        cloud_figure(tel, buf["ranges"]),
        pose_figure(buf, "franka"),
        pose_figure(buf, "ur"),
        gripper_figure("franka", tel.get("franka_width_mm"), tel.get("franka_grip_n")),
        gripper_figure("ur", tel.get("ur_width_mm"), None),
    )


def _canvas(arm: str) -> html.Div:
    st = DEFAULT[arm]
    lo, hi, marks = _slider(arm, st)
    cfg = {"edits": {"shapePosition": True}, "displayModeBar": False}
    fig = grasp2d.figure(arm, st["slide_mm"], st["pivot_deg"], st["depth_mm"])
    return html.Div(
        [
            dcc.Graph(id=f"col-canvas-{arm}", figure=fig, config=cfg),
            html.Div(
                [
                    html.Span("finger–belt angle deg"),
                    dcc.Slider(id=f"col-pivot-{arm}", min=lo, max=hi, step=1,
                               value=st["pivot_deg"], marks=marks),
                ]
            ),
            html.Div(
                readout(arm, st), id=f"col-readout-{arm}", style={"fontFamily": "monospace"}
            ),
        ]
    )


def layout(ctx) -> html.Div:
    teleop = getattr(ctx.backend, "supports_teleop", False)
    graph = {"config": {"displayModeBar": False}}
    return html.Div(
        [
            dcc.Store(id="col-grasp", data=DEFAULT),
            dcc.Store(id="col-buf"),
            dcc.Interval(id="col-tick", interval=1000),
            html.Div(
                [
                    html.Div("Grasp", className="card__title"),
                    html.Div([_canvas(arm) for arm in ARMS], className="grid-2"),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.Div("Controls", className="card__title"),
                    html.Button(
                        "Make start (pick up to workspace middle)", id="col-make", className="btn"
                    ),
                    html.Button(
                        "Record scripted demo", id="col-record", disabled=True, className="btn"
                    ),
                    html.Button(
                        "Start teleop",
                        id="col-teleop",
                        disabled=True,
                        className="btn",
                        title=f"SpaceMouse driver: {driver_command(ctx.cfg.lcm_url)}"
                        if teleop else "needs PKG-04 / SpaceMouse",
                    ),
                    html.Button("Done stage 1", id="col-tdone", className="btn btn--secondary"),
                    html.Button("Pause", id="col-tpause", className="btn btn--secondary"),
                    html.Button("Stop", id="col-stop", className="btn btn--danger"),
                    dcc.ConfirmDialogProvider(
                        html.Button("Reset", className="btn btn--secondary"),
                        id="col-reset",
                        message="Stop and delete the in-progress demo?",
                    ),
                    html.Div(
                        [
                            html.Span(id="col-status"),
                            html.Span(id="col-job", style={"marginLeft": "12px"}),
                            html.Span(id="col-start", className="muted", style={"marginLeft": "12px"}),
                        ],
                        className="row",
                    ),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.Div("Point cloud", className="card__title"),
                    dcc.Graph(id="col-cloud", **graph),
                ],
                className="card",
            ),
            html.Div(
                [
                    html.Div("Telemetry", className="card__title"),
                    html.Div(
                        [dcc.Graph(id=f"col-pose-{arm}", **graph) for arm in ARMS]
                        + [dcc.Graph(id=f"col-grip-{arm}", **graph) for arm in ARMS],
                        className="grid-2",
                    ),
                ],
                className="card",
            ),
        ]
    )


def register(app, ctx) -> None:
    @app.callback(
        Output("col-grasp", "data"),
        *(Output(f"col-canvas-{a}", "figure") for a in ARMS),
        *(Output(f"col-readout-{a}", "children") for a in ARMS),
        *(Output(f"col-pivot-{a}", p) for p in ("value", "min", "max", "marks") for a in ARMS),
        *(Input(f"col-canvas-{a}", "relayoutData") for a in ARMS),
        *(Input(f"col-pivot-{a}", "value") for a in ARMS),
        Input("col-reset", "submit_n_clicks"),
        State("col-grasp", "data"),
    )
    def _grasp(*args):
        return on_grasp(callback_context.triggered_id, *args)

    @app.callback(
        Output("col-status", "children"),
        *(Input(i, "n_clicks") for i in ("col-make", "col-record", "col-teleop", *TELEOP_KEYS,
                                          "col-stop")),
        Input("col-reset", "submit_n_clicks"),
        State("col-grasp", "data"),
        prevent_initial_call=True,
    )
    def _action(*args):
        return on_action(ctx, callback_context.triggered_id, args[-1])

    @app.callback(
        Output("col-record", "disabled"),
        Output("col-teleop", "disabled"),
        Output("col-start", "children"),
        Output("col-job", "children"),
        Input("col-tick", "n_intervals"),
    )
    def _status(_):
        return on_status(ctx)

    @app.callback(
        Output("col-buf", "data"),
        Output("col-cloud", "figure"),
        *(Output(f"col-pose-{a}", "figure") for a in ARMS),
        *(Output(f"col-grip-{a}", "figure") for a in ARMS),
        Input("col-tick", "n_intervals"),
        State("col-buf", "data"),
    )
    def _live(_, buf):
        return on_live(ctx, buf)
