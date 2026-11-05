"""Panel 4: build the training buffer, train (v12_152 recipe q), live curves, export, recon vs GT."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update

from apps.demo_gui import buffer, jobs, recon, trainlog
from apps.demo_gui.config import REPO
from apps.demo_gui.session import read_index

V63J = REPO / "data/lcs/realfirst/cmp/encoder_study/sim/V63j/deploy_v39j/deploy.npz"
# the source of the fixture demo (the fixture's clouds are cut to 128 points)
REPO_DEMO = REPO / "data/lcs/realfirst/cmp/v12_148/tiers/eps/s1/v12_147_demo__episode_gv_07_0.npz"
TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
NOT_LOSS = ("train/lr_ae", "train/rho_A", "train/B_fro", "train/mode_delta_rel")
FAMILIES = [  # (title, key filter, log y)
    ("Training loss", lambda k: k.startswith("train/") and not k.endswith("_mm")
     and k not in NOT_LOSS, True),
    ("Validation loss", lambda k: k.startswith("val/") and not k.endswith(("_mm", "_deg")), True),
    ("Reconstruction (mm)", lambda k: "recon" in k and k.endswith("_mm"), True),
    ("Next-step / multi-step prediction (mm)",
     lambda k: k.endswith("_mm") and ("next" in k or "ms_" in k), True),
]


def runs(session) -> list[Path]:
    """``<s>/train/<tag>`` run dirs (not the buffers or the split ``_h<k>`` heads), newest last."""
    return sorted((p for p in session.train.iterdir() if (p / "train.log").is_file()
                   and not re.search(r"_h\d$", p.name)), key=lambda p: p.stat().st_mtime)


def figures(log: dict) -> list:
    out = []
    for title, keep, logy in FAMILIES:
        keys = [k for k in log if k != "epoch" and keep(k)
                and any(v is not None and v > 0 for v in log[k])]
        if not keys:
            continue
        fig = go.Figure([go.Scatter(x=log["epoch"], y=log[k], name=k, mode="lines") for k in keys])
        fig.update_layout(title=title, height=300, margin={"l": 50, "r": 10, "t": 40, "b": 30},
                          xaxis_title="epoch", uirevision=title)
        if logy:
            fig.update_yaxes(type="log")
        out.append(dcc.Graph(figure=fig, style={"width": "50%", "display": "inline-block"}))
    return out


def status(run: Path, job) -> str:
    rep = trainlog.report(run)
    if rep:
        m = rep.get("metrics", {})
        return (f"{run.name}: done, {rep.get('epochs')} epochs; held-out recon "
                f"{m.get('recon_chamfer_mm', float('nan')):.2f} mm, one-step "
                f"{m.get('one_step_chamfer_mm', float('nan')):.2f} mm, open-loop 7 "
                f"{m.get('open_loop7_chamfer_mm', float('nan')):.2f} mm; {rep.get('checkpoint')}")
    ep = trainlog.last_epoch(run / "train.log")
    js = f" | job {jobs.status(job)}" if job else ""
    return f"{run.name}: " + (f"epoch {ep}" if ep else "starting") + js


def export_status(run: Path, job) -> str:
    d = run / "export"
    lines = (d / "status").read_text().splitlines() if (d / "status").is_file() else []
    js = f"job {jobs.status(job)}" if job else ""
    arm = f"arm: {d / 'arm.json'}" if (d / "arm.json").is_file() else ""
    return " | ".join(x for x in (js, lines[-1] if lines else "", arm) if x)


def episode_options(session) -> list[dict]:
    opts = [{"label": f"session {d.name}/{ep['file']}", "value": str(d / ep["file"])}
            for d in sorted(p for p in session.demos.iterdir() if p.is_dir())
            for ep in read_index(d) if ep.get("file")]
    if REPO_DEMO.is_file():
        opts.append({"label": "repo demo gv_07 (stage-1 cut)", "value": str(REPO_DEMO)})
    return opts


def model_options(session) -> list[dict]:
    """Session exports (``export/arm.json`` stage-1 deploy), then V63j; the decoder is the
    ``decoder_decfit`` beside the deploy dir (prep_v52_s1.py layout)."""
    deps = [(f"export {p.parent.parent.name}", Path(next(iter(json.loads(p.read_text()).values()))
                                                     ["deploy1"]))
            for p in sorted(session.train.glob("*/export/arm.json"))]
    deps.append(("V63j (repo)", V63J))
    return [{"label": name, "value": json.dumps(
        {"deploy": str(d), "decoder": str(d.parent.parent / "decoder_decfit/decoder.npz")})}
            for name, d in deps]


def recon_figure(gt: np.ndarray, pred: np.ndarray, title: str) -> go.Figure:
    mk = {"size": 2}
    fig = go.Figure([
        go.Scatter3d(x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], mode="markers", name="observed (GT)",
                     marker={**mk, "color": "#555"}),
        go.Scatter3d(x=pred[:, 0], y=pred[:, 1], z=pred[:, 2], mode="markers", name="decoded",
                     marker={**mk, "color": "#1f77b4"})])
    fig.update_layout(title=title, height=520, scene={"aspectmode": "data"}, uirevision="recon",
                      margin={"l": 0, "r": 0, "t": 40, "b": 0})
    return fig


def layout(ctx) -> html.Div:
    num = {"type": "number", "className": "control", "style": {"width": "80px"}}
    return html.Div([
        dcc.Interval(id="train-tick", interval=2000),
        html.Div([
            html.Div("Training data", className="card__title"),
            dcc.Checklist(id="train-data", options=[], value=[], labelStyle={"display": "block"}),
            html.Div([
                html.Label("tag "), dcc.Input(id="train-tag", value="t1", type="text",
                                              className="control", style={"width": "140px"}),
                html.Label("seed "), dcc.Input(id="train-seed", value=5, min=0, step=1, **num),
                html.Button("Start training", id="train-start", className="btn"),
            ], className="row"),
            html.Div([html.Label("epochs"),
                      dcc.Slider(id="train-epochs", min=10, max=300, step=10, value=60,
                                 marks={v: str(v) for v in (10, 60, 100, 200, 300)})],
                     style={"width": "480px"}),
            html.Div(id="train-msg", className="err"),
        ], className="card"),
        html.Div([
            html.Div("Progress", className="card__title"),
            dcc.Dropdown(id="train-run", options=[], placeholder="training run",
                         className="control", style={"width": "320px"}),
            html.Div(id="train-status", style={"margin": "6px 0"}),
            html.Div(id="train-plots"),
            html.Div([html.Button("Export for evaluation", id="train-export", className="btn"),
                      html.Span(id="train-export-status")], className="row"),
        ], className="card"),
        html.Div([
            html.Div("Reconstruction vs ground truth", className="card__title"),
            html.Div([
                dcc.Dropdown(id="recon-episode", options=[], placeholder="episode",
                             className="control", style={"width": "460px"}),
                dcc.Dropdown(id="recon-model", options=[], placeholder="deploy + decoder",
                             className="control", style={"width": "320px"}),
            ], className="row"),
            dcc.Slider(id="recon-frame", min=0, max=1, step=1, value=0, marks=None,
                       updatemode="mouseup", tooltip={"placement": "bottom"}),
            html.Div(id="recon-msg", className="err"),
            dcc.Graph(id="recon-fig"),
        ], className="card"),
    ])


def register(app, ctx) -> None:
    state: dict = {"train": None, "export": None}
    s = ctx.session

    @app.callback(Output("train-data", "options"), Output("train-run", "options"),
                  Output("recon-episode", "options"), Output("recon-model", "options"),
                  Input("tabs", "value"), Input("train-tick", "n_intervals"))
    def _options(tab, _):
        if tab != "train":
            return (no_update,) * 4
        rs = [{"label": p.name, "value": p.name} for p in runs(s)]
        if callback_context.triggered_id == "train-tick":  # only the run list grows while training
            return no_update, rs, no_update, no_update
        data = [{"label": c["label"], "value": json.dumps(c)} for c in buffer.candidates(s)]
        return data, rs, episode_options(s), model_options(s)

    @app.callback(Output("train-msg", "children"), Output("train-run", "value"),
                  Input("train-start", "n_clicks"), State("train-data", "value"),
                  State("train-tag", "value"), State("train-epochs", "value"),
                  State("train-seed", "value"), prevent_initial_call=True)
    def _start(_, picked, tag, epochs, seed):
        tag = (tag or "").strip()
        if not TAG.match(tag) or re.search(r"_h\d$", tag) or tag == "buffers":
            return "tag: letters, digits, _ . - (not 'buffers', not ending in _h<k>)", no_update
        if (s.train / tag).exists() or buffer.path(s, tag).exists():
            return f"tag {tag} already used in this session", no_update
        if not picked:
            return "pick training data", no_update
        try:
            counts = buffer.build(s, [json.loads(v) for v in picked], buffer.path(s, tag))
            rows = json.loads(buffer.path(s, tag).read_text())["rollouts"]
            state["train"] = ctx.backend.train(s, tag, rows, int(epochs), seed=int(seed or 5))
        except (ValueError, OSError, KeyError) as e:
            return str(e), no_update
        return f"started {tag}: {counts}", tag

    @app.callback(Output("train-status", "children"), Output("train-plots", "children"),
                  Input("train-tick", "n_intervals"), Input("train-run", "value"),
                  State("tabs", "value"))
    def _progress(_, run, tab):
        if tab != "train" or not run:
            return no_update, no_update
        d = s.train / run
        job = state["train"] if state["train"] and state["train"].name == f"train_{run}" else None
        return status(d, job), figures(trainlog.parse(d / "train.log"))

    @app.callback(Output("train-export-status", "children"), Input("train-export", "n_clicks"),
                  Input("train-tick", "n_intervals"), State("train-run", "value"),
                  State("tabs", "value"))
    def _export(_, __, run, tab):
        if tab != "train" or not run:
            return no_update
        d = s.train / run
        if callback_context.triggered_id == "train-export":
            if jobs.running(s, f"export_{run}"):
                return f"export {run} is in progress"
            if not trainlog.report(d):
                return f"{run}: training not finished"
            try:
                state["export"] = ctx.backend.export(s, run)
            except (ValueError, OSError) as e:
                return str(e)
        job = state["export"] if state["export"] and state["export"].name == f"export_{run}" else None
        return export_status(d, job)

    @app.callback(Output("recon-fig", "figure"), Output("recon-frame", "max"),
                  Output("recon-msg", "children"), Input("recon-episode", "value"),
                  Input("recon-model", "value"), Input("recon-frame", "value"))
    def _recon(npz, model, i):
        if not npz or not model:
            return go.Figure(), no_update, ""
        m = json.loads(model)
        try:
            r = recon.episode(m["deploy"], m["decoder"], npz, s.train / "recon_cache")
        except Exception as e:  # noqa: BLE001 - shown to the user
            return go.Figure(), no_update, f"recon failed: {e}"
        i = min(int(i or 0), len(r["gt"]) - 1)
        title = f"frame {i}: Chamfer {r['chamfer_mm'][i]:.2f} mm ({Path(npz).name})"
        return recon_figure(r["gt"][i], r["pred"][i], title), len(r["gt"]) - 1, ""
