"""Panel 5: run a trained policy on the e2e hard suite, results table, scrub a finished episode."""

from __future__ import annotations

from pathlib import Path

from dash import Input, Output, State, callback_context, dash_table, dcc, html, no_update

from apps.demo_gui import episodes, evalrun, jobs

COLUMNS = ["start", "repeat", "both", "strict", "strict_why", "large", "small", "wrap_l",
           "wrap_s", "duration_s", "reason", "run"]


def eval_runs(session) -> list[dict]:
    """``<s>/eval/*`` newest first."""
    ds = sorted((p for p in session.eval.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"label": p.name, "value": str(p)} for p in ds]


def results_view(out) -> tuple[list[dict], str, list[dict]]:
    """Table rows, summary + ``hard_suite.md`` markdown, scrub dropdown options of one run."""
    rows = evalrun.results(out)
    for r in rows:
        r["both"], r["strict"] = ("yes" if r["both"] else "no"), ("OK" if r["strict"] else "no")
    n = len(rows)
    md = (f"**strict {sum(r['strict'] == 'OK' for r in rows)}/{n}, both engaged "
          f"{sum(r['both'] == 'yes' for r in rows)}/{n}**" if n else "no finished episode yet")
    return rows, md + "\n\n" + (evalrun.suite_md(out) or ""), evalrun.episode_rows(out)


def _status(path: str | None, d: str | None) -> str:
    """The run job's state + the last ``run.status`` line of the selected eval run."""
    out = f"{(job := jobs.load(path)).name}: {jobs.status(job)}" if path and Path(
        path).is_file() else "idle"
    if out.endswith("failed") and Path(job.log).is_file() and (
            tail := Path(job.log).read_text(errors="replace").strip().splitlines()):
        out += f" ({tail[-1][-160:]})"
    st = Path(d) / "run.status" if d else None
    if st and st.is_file() and (lines := st.read_text().splitlines()):
        out += f" | {lines[-1]}"
    return out


def run_sig(d: str | None) -> list[str] | None:
    """Changes when an episode index or ``hard_suite.md`` of run ``d`` is (re)written."""
    if not d:
        return None
    files = [*sorted(Path(d).glob("*/index.json")), Path(d) / "hard_suite.md"]
    return [d, *(str(p.stat().st_mtime) for p in files if p.is_file())]


def layout(ctx):
    ms = sorted(evalrun.models(ctx.session))
    return html.Div([
        html.Div([
            html.Div("Run evaluation", className="card__title"),
            html.Div([
                html.Label("model"),
                dcc.Dropdown(id="eval-model", options=ms, className="control",
                             style={"width": "260px"},
                             value="V63j_tol50b" if "V63j_tol50b" in ms else (ms or [None])[0]),
                html.Label("repeats"),
                dcc.Input(id="eval-repeats", type="number", min=1, max=5, step=1, value=1,
                          className="control", style={"width": "60px"}),
                html.Button("Run", id="eval-run-btn", className="btn"),
                html.Button("Stop", id="eval-stop-btn", className="btn btn--secondary"),
                html.Span("idle", id="eval-status", className="muted"),
            ], className="row"),
            dcc.Checklist(id="eval-starts", options=evalrun.rhs.DEFAULT_KEYS,
                          value=evalrun.IN_DIST, inline=True, inputStyle={"marginLeft": "10px"}),
            dcc.Store(id="eval-job"), dcc.Store(id="eval-sig"),
        ], className="card"),
        html.Div([
            html.Div("Results", className="card__title"),
            html.Div([html.Label("eval run"),
                      dcc.Dropdown(id="eval-dir", options=eval_runs(ctx.session),
                                   className="control", style={"width": "400px"})],
                     className="row"),
            dcc.Markdown(id="eval-md"),
            dash_table.DataTable(id="eval-table", columns=[{"name": c, "id": c} for c in COLUMNS],
                                 sort_action="native", style_cell={"fontSize": "13px"},
                                 style_data_conditional=[
                                     {"if": {"filter_query": '{strict} = "OK"'},
                                      "backgroundColor": "#e6f4ea"}]),
        ], className="card"),
        html.Div([
            html.Div("Scrub an episode", className="card__title"),
            html.Div([dcc.Dropdown(id="eval-episode", className="control",
                                   style={"width": "520px"}),
                      html.Button("Play / pause", id="eval-play-btn", className="btn"),
                      html.Button("Open replay viewer", id="eval-viewer-btn",
                                  className="btn btn--secondary"),
                      html.Button("Stop viewer", id="eval-viewer-stop",
                                  className="btn btn--secondary"),
                      html.Span(id="eval-viewer-msg", className="muted")], className="row"),
            dcc.Slider(id="eval-frame", min=0, max=0, step=1, value=0, marks=None,
                       tooltip={"placement": "bottom"}),
            dcc.Interval(id="eval-play", interval=200, disabled=True),
            html.Div([dcc.Graph(id="eval-3d"), dcc.Graph(id="eval-wrap")], className="grid-2"),
            dcc.Graph(id="eval-latent"),
        ], className="card"),
    ])


def register(app, ctx) -> None:
    s, be, cfg = ctx.session, ctx.backend, ctx.cfg

    @app.callback(Output("eval-job", "data"), Output("eval-status", "children"),
                  Input("eval-run-btn", "n_clicks"), State("eval-model", "value"),
                  State("eval-starts", "value"), State("eval-repeats", "value"),
                  prevent_initial_call=True)
    def _run(_, name, starts, repeats):
        if not name or not starts:
            return no_update, "pick a model and at least one start"
        if jobs.running(s, f"eval_{name}"):
            return no_update, f"an eval of {name} is in progress; stop it first"
        try:
            job = be.evaluate(s, name, evalrun.models(s)[name], starts, int(repeats or 1))
        except (RuntimeError, ValueError, KeyError) as e:
            return no_update, f"not started: {e}"
        return job.path, f"{job.name}: started"

    @app.callback(Output("eval-status", "children", allow_duplicate=True),
                  Input("eval-stop-btn", "n_clicks"), State("eval-job", "data"),
                  State("eval-dir", "value"), prevent_initial_call=True)
    def _stop(_, path, d):
        if path:
            jobs.stop(jobs.load(path))
        return _status(path, d)

    @app.callback(Output("eval-status", "children", allow_duplicate=True),
                  Output("eval-model", "options"), Output("eval-dir", "options"),
                  Output("eval-dir", "value"), Output("eval-sig", "data"),
                  Input("tick", "n_intervals"), State("eval-job", "data"),
                  State("eval-model", "options"), State("eval-dir", "options"),
                  State("eval-dir", "value"), State("eval-sig", "data"),
                  prevent_initial_call=True)
    def _poll(_, path, m_opts, d_opts, d, sig):
        ms, ds = sorted(evalrun.models(s)), eval_runs(s)
        follow = bool(ds) and (d is None or ds != d_opts)  # a new run appeared: show it
        d = ds[0]["value"] if follow else d
        new_sig = run_sig(d)
        return (_status(path, d), ms if ms != m_opts else no_update,
                ds if ds != d_opts else no_update, d if follow else no_update,
                new_sig if new_sig != sig else no_update)

    @app.callback(Output("eval-table", "data"), Output("eval-md", "children"),
                  Output("eval-episode", "options"), Output("eval-episode", "value"),
                  Input("eval-sig", "data"), State("eval-dir", "value"),
                  State("eval-episode", "value"))
    def _results(_, d, npz):
        if not d or not Path(d).is_dir():
            return [], "", [], None
        rows, md, eps = results_view(d)
        vals = [e["value"] for e in eps]
        return rows, md, eps, npz if npz in vals else (vals or [None])[0]

    @app.callback(Output("eval-frame", "max"), Output("eval-frame", "value"),
                  Input("eval-episode", "value"))
    def _episode(npz):
        return (evalrun.n_frames(npz) - 1, 0) if npz else (0, 0)

    @app.callback(Output("eval-play", "disabled"), Input("eval-play-btn", "n_clicks"),
                  State("eval-play", "disabled"), prevent_initial_call=True)
    def _play(_, disabled):
        return not disabled

    @app.callback(Output("eval-frame", "value", allow_duplicate=True),
                  Input("eval-play", "n_intervals"), State("eval-frame", "value"),
                  State("eval-frame", "max"), prevent_initial_call=True)
    def _advance(_, i, n):
        return 0 if (i or 0) >= (n or 0) else (i or 0) + 1

    @app.callback(Output("eval-3d", "figure"), Output("eval-wrap", "figure"),
                  Output("eval-latent", "figure"),
                  Input("eval-frame", "value"), State("eval-episode", "value"))
    def _frame(i, npz):
        if not npz or not Path(npz).is_file():
            return {}, {}, {}
        note = episodes.publish_frame(s, npz, i or 0, cfg.lcm_url)
        return (*evalrun.scrub_figures(npz, i or 0, note), evalrun.latent_figure(npz, i or 0))

    @app.callback(Output("eval-viewer-msg", "children"),
                  Input("eval-viewer-btn", "n_clicks"), Input("eval-viewer-stop", "n_clicks"),
                  State("eval-episode", "value"), prevent_initial_call=True)
    def _viewer(_o, _s, npz):
        if callback_context.triggered_id == "eval-viewer-stop":
            episodes.stop_replay(s)
            return "viewer stopped"
        if not npz:
            return "pick an episode first"
        episodes.start_replay(s, npz, cfg.replay_port)
        return f"replay viewer for {Path(npz).stem}: shows in the viewer dock in a few s"
