"""Panel 2: pick demos, run MILES augmentation, watch its buffer grow."""

from __future__ import annotations

import json
from pathlib import Path

from dash import Input, Output, State, callback_context, dcc, html, no_update

from apps.demo_gui import episodes, jobs
from apps.demo_gui.config import REPO
from apps.demo_gui.session import read_buffer


def demo_options(session) -> list[dict]:
    """Session demos, then the repo seed demos (``run_hard_suite.SEED_BUFFERS``, read-only)."""
    from run_hard_suite import SEED_BUFFERS

    opts = [{"label": f"session {r['name']} ({r['outcome']}, {r['frames']} fr)",
             "value": json.dumps({"name": r["name"].replace("/", "_"), "episode": r["path"],
                                  "start": str(REPO / r["start_state"])})}
            for r in episodes.scan(session) if r["kind"] == "demo" and r["start_state"]]
    for b in SEED_BUFFERS:
        opts += [{"label": f"repo seed {r['id']} ({r['start']['set']}/{r['start']['id']})",
                  "value": json.dumps({"name": r["id"], "episode": r["episode"],
                                       "start": str(REPO / r["start"]["file"])})}
                 for r in read_buffer(REPO / b) if r["kind"] == "demo"]
    return opts


def progress(session, job: jobs.Job | None) -> list:
    dirs = sorted((p for p in session.miles.iterdir() if (p / "buffer.json").is_file()),
                  key=lambda p: p.stat().st_mtime)
    if not dirs:
        return [html.Div("no MILES run yet" if job is None else f"{job.name}: starting")]
    d = dirs[-1]
    rows = read_buffer(d / "buffer.json")
    out = [html.Div(f"{d.name}: {len(rows)} rollouts, "
                    f"{sum(r.get('status') == 'ok' for r in rows)} valid"
                    + (f" | job {jobs.status(job)}" if job else ""))]
    if (rep := d / "miles_report.md").is_file():
        out.append(html.Pre("\n".join(rep.read_text().splitlines()[-25:]),
                            style={"maxHeight": "360px", "overflow": "auto", "fontSize": "12px"}))
    return out


def layout(ctx) -> html.Div:
    num = {"type": "number", "min": 1, "step": 1, "style": {"width": "70px"}}
    return html.Div([
        html.Div([
            html.Div("Demos to augment", className="card__title"),
            dcc.Checklist(id="miles-demos", options=[], value=[],
                          labelStyle={"display": "block"}),
            html.Div([html.Label("cycles "),
                      dcc.Input(id="miles-cycles", value=4, className="control", **num),
                      html.Label("wp-stride "),
                      dcc.Input(id="miles-stride", value=5, className="control", **num),
                      html.Button("Start augmenting", id="miles-start", className="btn"),
                      html.Button("Stop", id="miles-stop", className="btn btn--secondary")],
                     className="row"),
            html.Div(id="miles-msg", className="err"),
        ], className="card"),
        html.Div([html.Div("Progress", className="card__title"), html.Div(id="miles-progress")],
                 className="card"),
    ])


def register(app, ctx) -> None:
    state: dict = {"job": None}

    @app.callback(Output("miles-demos", "options"), Input("tabs", "value"))
    def _options(tab):
        return demo_options(ctx.session) if tab == "miles" else no_update

    @app.callback(Output("miles-msg", "children"), Input("miles-start", "n_clicks"),
                  Input("miles-stop", "n_clicks"), State("miles-demos", "value"),
                  State("miles-cycles", "value"), State("miles-stride", "value"),
                  prevent_initial_call=True)
    def _action(_s, _x, picked, cycles, stride):
        if callback_context.triggered_id == "miles-stop":
            if state["job"]:
                jobs.stop(state["job"])
            return ""
        if state["job"] and jobs.status(state["job"]) == "running":
            return "a MILES run is in progress; stop it first"
        if not picked:
            return "pick at least one demo"
        try:
            state["job"] = ctx.backend.run_miles(ctx.session, [json.loads(v) for v in picked],
                                                 cycles=int(cycles or 4), wp_stride=int(stride or 5))
        except (ValueError, OSError) as e:
            return str(e)
        return f"started {Path(state['job'].path).stem}"

    @app.callback(Output("miles-progress", "children"), Input("tick", "n_intervals"),
                  State("tabs", "value"))
    def _progress(_, tab):
        return progress(ctx.session, state["job"]) if tab == "miles" else []
