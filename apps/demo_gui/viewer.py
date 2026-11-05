"""Viewer dock shared by every tab: the standalone LCM viewer (8091), the replay viewer, or idle."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from dash import Input, Output, State, html, no_update

from apps.demo_gui import jobs


def live_job(session) -> jobs.Job | None:
    """Newest running sim job (demo / eval_*) — those publish body poses."""
    for p in sorted(session.jobs.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if (p.stem == "demo" or p.stem.startswith("eval_")) and jobs.status(
                j := jobs.load(p)) == "running":
            return j
    return None


_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def port_open(url: str, timeout_s: float = 0.2) -> bool:
    """Is the viewer serving? A bare TCP connect makes viser log a handshake traceback; an unknown
    path gets a quiet empty 404."""
    try:
        _DIRECT.open(url.rstrip("/") + "/__probe__", timeout=timeout_s).close()
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False
    return True


def episode_name(argv: list[str]) -> str:
    """``--run`` (the episode stem) if given, else the recordings dir's parent (newest run)."""
    if "--run" in argv:
        return argv[argv.index("--run") + 1]
    return Path(argv[argv.index("--recordings") + 1]).parent.name


def dock_src(session, cfg) -> tuple[str, str] | None:
    """(iframe src, title): live > replay > standalone viewer > None. The standalone src has no
    ``?job=``: one iframe across runs (it keeps the last frame); a new replay job reloads."""
    up = port_open(cfg.viewer_url)
    if up and (j := live_job(session)):
        return cfg.viewer_url, f"live: {j.name}"
    p, url = session.jobs / "replay.json", f"http://localhost:{cfg.replay_port}"
    if p.is_file() and jobs.status(j := jobs.load(p)) == "running" and port_open(url):
        return f"{url}/?job={j.pid}", f"replay: {episode_name(j.argv)}"
    return (cfg.viewer_url, "viewer") if up else None


def view(src: str | None, cfg) -> list:
    if not src:
        return [html.Div(html.Div([
            html.Div("No viewer. Launch it once (it keeps the last frame between runs):"),
            html.Code(cfg.viewer_cmd),
            html.Div("or Open replay viewer (Data / Evaluate)")]),
            className="viewer viewer--idle")]
    return [html.Iframe(src=src, className="viewer")]


def register_dock(app, ctx) -> None:
    @app.callback(Output("dock-view", "children"), Output("dock-title", "children"),
                  Output("dock-open", "href"), Output("dock-open", "style"),
                  Output("dock-src", "data"), Input("tick", "n_intervals"),
                  State("dock-src", "data"))
    def _dock(_, cur):
        src, title = dock_src(ctx.session, ctx.cfg) or (None, "no viewer")
        cur = cur or [None, "no viewer"]  # the layout's initial placeholder
        if [src, title] == cur:
            return (no_update,) * 5
        if src == cur[0]:  # title only: re-sending the iframe would reload it
            return no_update, title, no_update, no_update, [src, title]
        return (view(src, ctx.cfg), title, src, None if src else {"display": "none"},
                [src, title])
