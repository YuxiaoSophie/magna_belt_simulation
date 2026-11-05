"""Dash app: header with job chips + five pipeline tabs; panels fill their tab via layout/register."""

from __future__ import annotations

from dataclasses import dataclass

from dash import Dash, Input, Output, State, dcc, html

from apps.demo_gui import jobs, theme, viewer
from apps.demo_gui.config import Cfg
from apps.demo_gui.panels import browser, collect, evaluate, miles, train
from apps.demo_gui.session import Session

TABS = [("collect", "1 Collect"), ("miles", "2 MILES"), ("browser", "3 Data"),
        ("train", "4 Train"), ("evaluate", "5 Evaluate")]


@dataclass
class Ctx:
    backend: object
    session: Session
    cfg: Cfg


def job_chips(session: Session) -> list:
    """Running jobs, the newest finished one, and one `N done · M failed` chip listing all."""
    paths = sorted(session.jobs.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    loaded = [(p.stem, jobs.load(p)) for p in paths]
    rows = [(n, jobs.status(j)) for n, j in loaded]

    def chip(n, st, note=None):
        return html.Span(f"{n}: {note or st}", className=f"chip {st}")

    chips = [chip(n, st, jobs.gpu_wait(j)) for (n, j), (_, st) in zip(loaded, rows)
             if st == "running"]
    chips += [chip(n, st) for n, st in rows if st != "running"][:1]
    nd, nf = (sum(st == k for _, st in rows) for k in ("done", "failed"))
    if nd or nf:
        chips.append(html.Details([html.Summary(f"{nd} done · {nf} failed"),
                                   html.Div([html.Div(f"{n}: {st}") for n, st in rows])],
                                  className="chip summary"))
    return chips or [html.Span("no jobs", className="muted")]


def build_app(backend, session: Session, cfg: Cfg) -> Dash:
    theme.apply()
    ctx = Ctx(backend, session, cfg)
    panels = {"collect": collect, "evaluate": evaluate, "miles": miles, "browser": browser,
              "train": train}  # tab name -> module with layout(ctx) / register(app, ctx)
    app = Dash(__name__, title="Belt assembly demo", suppress_callback_exceptions=True)
    app.layout = html.Div([
        html.Div([
            html.H1("Round-belt learned LCS demo"),
            html.Span(f"session {session.name}", className="muted"),
            html.Span(cfg.backend, className="badge"),
            html.Span(id="job-chips"),
        ], className="topbar"),
        dcc.Interval(id="tick", interval=1000),
        html.Div(dcc.Tabs(id="tabs", value=TABS[0][0], className="nav-tabs", children=[
            dcc.Tab(label=label, value=name, className="nav-tab",
                    selected_className="nav-tab--active", children=panels[name].layout(ctx)
                    if name in panels else html.Div(f"TODO {name}"))
            for name, label in TABS]), className="content"),
        html.Div([
            html.Div(className="dock__handle", title="drag to resize, double-click to reset"),
            html.Button("⇄", id="dock-toggle", title="show / hide the viewer",
                        className="btn btn--secondary dock-toggle"),
            html.Div("no viewer", id="dock-title", className="card__title"),
            html.Div(viewer.view(None, cfg), id="dock-view"),
            html.A("Open in new tab", id="dock-open", target="_blank", rel="noopener",
                   style={"display": "none"}),
            dcc.Store(id="dock-src"),
            dcc.Store(id="dock-collapsed", storage_type="session"),
        ], id="dock", className="dock"),
    ])
    for panel in panels.values():
        panel.register(app, ctx)
    viewer.register_dock(app, ctx)

    @app.callback(Output("dock-collapsed", "data"), Input("dock-toggle", "n_clicks"),
                  State("dock-collapsed", "data"), prevent_initial_call=True)
    def _toggle(_, collapsed):
        return not collapsed

    @app.callback(Output("dock", "className"), Input("dock-collapsed", "data"))
    def _collapse(collapsed):
        return "dock dock--collapsed" if collapsed else "dock"

    @app.callback(Output("job-chips", "children"), Input("tick", "n_intervals"))
    def _chips(_):
        return job_chips(session)

    return app
