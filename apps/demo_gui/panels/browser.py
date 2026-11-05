"""Panel 3: every session episode in a table; traces, cloud scrub and the replay viewer per row."""

from __future__ import annotations

from pathlib import Path

from dash import Input, Output, State, callback_context, dash_table, dcc, html, no_update

from apps.demo_gui import episodes

COLUMNS = ["kind", "name", "frames", "outcome", "success", "start_state", "cmd_sources"]


def table_rows(session) -> list[dict]:
    return [{**r, "start_state": Path(r["start_state"] or "").name,
             "success": "" if r["success"] is None else str(r["success"]),
             "cmd_sources": " ".join(f"{k}:{v}" for k, v in sorted(r["cmd_sources"].items()))}
            for r in episodes.scan(session)]


def layout(ctx) -> html.Div:
    return html.Div([
        html.Div([
            html.Div("Episodes", className="card__title"),
            html.Button("Refresh", id="br-refresh", className="btn btn--secondary"),
            dash_table.DataTable(id="br-table", columns=[{"name": c, "id": c} for c in COLUMNS],
                             data=[], sort_action="native", filter_action="native",
                             row_selectable="single", selected_rows=[], page_size=12,
                             style_cell={"fontSize": "13px", "textAlign": "left"}),
        ], className="card"),
        dcc.Store(id="br-sel"),
        html.Div([
            html.Div("Traces and cloud", className="card__title"),
            dcc.Graph(id="br-traces"),
            html.Div([html.Label("frame"),
                      dcc.Slider(id="br-frame", min=0, max=1, step=1, value=0,
                                 marks=None, tooltip={"placement": "bottom"})]),
            dcc.Graph(id="br-cloud"),
        ], className="card"),
        html.Div([
            html.Div("Replay viewer", className="card__title"),
            html.Div([html.Button("Open replay viewer", id="br-viewer-open", className="btn"),
                      html.Button("Stop viewer", id="br-viewer-stop",
                                  className="btn btn--secondary"),
                      html.Span(id="br-viewer-msg", className="muted")], className="row"),
        ], className="card"),
    ])


def register(app, ctx) -> None:
    @app.callback(Output("br-table", "data"), Input("br-refresh", "n_clicks"),
                  Input("tabs", "value"))
    def _table(_, tab):
        return table_rows(ctx.session) if tab == "browser" else no_update

    @app.callback(Output("br-sel", "data"), Output("br-traces", "figure"),
                  Output("br-frame", "max"), Output("br-frame", "value"),
                  Input("br-table", "selected_rows"), State("br-table", "data"),
                  prevent_initial_call=True)
    def _select(sel, data):
        if not sel or sel[0] >= len(data or []):
            return no_update, no_update, no_update, no_update
        row = data[sel[0]]
        tr = episodes.load_traces(row["path"])
        return row, episodes.traces_figure(tr), max(tr["n_frames"] - 1, 1), 0

    @app.callback(Output("br-cloud", "figure"), Input("br-frame", "value"),
                  Input("br-sel", "data"), prevent_initial_call=True)
    def _cloud(i, row):
        if not row:
            return no_update
        pc, belt = episodes.cloud_at(row["path"], i or 0)
        note = episodes.publish_frame(ctx.session, row["path"], i or 0, ctx.cfg.lcm_url)
        return episodes.cloud_figure(pc, belt, f"{row['name']} frame {i} · {note}")

    @app.callback(Output("br-viewer-msg", "children"),
                  Input("br-viewer-open", "n_clicks"), Input("br-viewer-stop", "n_clicks"),
                  State("br-sel", "data"), prevent_initial_call=True)
    def _viewer(_o, _s, row):
        if callback_context.triggered_id == "br-viewer-stop":
            episodes.stop_replay(ctx.session)
            return "viewer stopped"
        if not row:
            return "select an episode first"
        job = episodes.start_replay(ctx.session, row["path"], ctx.cfg.replay_port)
        return f"replay viewer for {row['name']} (log {job.log}): shows in the viewer dock in a few s"
