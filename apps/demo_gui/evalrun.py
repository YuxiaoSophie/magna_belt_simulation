"""Panel 5 helpers (pure): model registry, hard-suite argv, results + strict score, scrub figures."""

from __future__ import annotations

import json
import math
import shlex
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.demo_gui.config import REPO, lcm_url

# isort: split
import run_hard_suite as rhs  # on sys.path via config

ARMS = REPO / "data/lcs/realfirst/cmp/v12_151/arms.json"
HELD_OUT = ("set2_gv_06", "ur_z+15")  # e2e152's held-out test keys
IN_DIST = [k for k in rhs.DEFAULT_KEYS if k not in HELD_OUT]
EVAL_EXTRA = "--end-snapshot --ur-exit-final-settle-s 3 --log-plan-age --franka-grip-n 30"
SPAN_MM, WRAP_DEG = 1.5, 100.0
MAX_CLOUD = 544
ARM_KEYS = ("deploy1", "wp1", "cfg1", "deploy2", "wp2", "cfg2")


def models(session) -> dict:
    """``arms.json`` + every ``<s>/train/*/export/arm.json`` (same ``{name: {deploy1..cfg2}}``)."""
    m = json.loads(ARMS.read_text()) if ARMS.is_file() else {}
    for p in sorted(session.train.glob("*/export/arm.json")):
        m.update(json.loads(p.read_text()))
    return m


def new_out(session, name: str) -> Path:
    """Next free ``<s>/eval/<name>_<nn>`` (not created)."""
    k = 0
    while (session.eval / f"{name}_{k:02d}").exists():
        k += 1
    return session.eval / f"{name}_{k:02d}"


def write_argv(cfg, out, name: str, model: dict, repeats: int, keys: list[str]) -> list[str]:
    """``run_hard_suite.py write --suite full`` as ``v12_152/e2e152.sh``, on the app's port."""
    if 7856 <= cfg.sim_port <= 7859:
        raise ValueError(f"port {cfg.sim_port}: needs the 239.255.78 URL form, use <= 7855")
    lcm_url(cfg.sim_port)  # raises on a refused port
    bad = [k for k in keys if k not in rhs.KEYS]
    if bad or not keys:
        raise ValueError(f"start keys {bad or keys}: need a subset of {rhs.KEYS}")
    return ["uv", "run", "--frozen", "python", "scripts/lcs/run_hard_suite.py", "write",
            "--suite", "full", "--scene", str(cfg.scene), "--out", str(out),
            "--port", str(cfg.sim_port), "--max-episode-s", "25", "--early-stop", "4,14,12",
            "--demo-actions", "none", "--repeats", str(int(repeats)),
            "--label", f"demo_gui_{name}", "--starts", ",".join(keys), "--eval-extra",
            f"{EVAL_EXTRA} --render-every {cfg.render_every} --publish-body-poses --realtime",
            "--arm", name, *(str(model[k]) for k in ARM_KEYS[:3]),
            "--arm2", name, *(str(model[k]) for k in ARM_KEYS[3:])]


def run_argv(out) -> list[str]:
    """``bash <out>/run.sh`` then ``summarize --suite full`` into ``<out>/summarize.out``."""
    q = shlex.quote
    return ["bash", "-c", (f"bash {q(f'{out}/run.sh')} && uv run --frozen python "
                           f"scripts/lcs/run_hard_suite.py summarize --suite full --dir "
                           f"{q(str(out))} > {q(f'{out}/summarize.out')} 2>&1")]


def strict(z, labels) -> tuple[bool, str]:
    # port of data/lcs/realfirst/cmp/v12_150/strict_e2e.py strict(): seated before the Franka lets go
    a = int(np.asarray(z["sim_chain_from"])[1]) if "sim_chain_from" in z.files else -1
    if a < 0:
        return False, "no stage 2"
    ends = [int(i) for i in np.flatnonzero(z["sim_phase"] == labels.index("post:release"))[:1]]
    ends += [a + int(i) for i in np.flatnonzero(~np.asarray(z["sim_grasp_ok"])[a:, 0])[:1]]
    i = min(ends, default=len(z["sim_phase"])) - 1
    b, pl = z["sim_belt_xyz"][i], z["sim_pulley_large_pose"][i][:3]
    on = np.linalg.norm(b[:, :2] - pl[:2], axis=1) * 1e3 < 62
    span = float(np.ptp((b[on, 2] - pl[2]) * 1e3)) if on.any() else float("nan")
    wl, ws = float(z["sim_wrap_deg"][i]), float(z["sim_wrap_small_deg"][i])
    return (span <= SPAN_MM and wl >= WRAP_DEG and ws >= WRAP_DEG,
            f"span {span:.1f} wrapL {wl:.0f} wrapS {ws:.0f}")


def _start(run: str) -> str:
    return next((k for k in rhs.KEYS if run.endswith(f"_{k}")), run)


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


def results(out) -> list[dict]:
    """One row per recorded episode of ``<out>/*/index.json`` with the strict score."""
    rows = []
    for idx in sorted(Path(out).glob("*/index.json")):
        js = json.loads(idx.read_text())
        for e in js.get("episodes", []):
            if not e.get("file") or not (idx.parent / e["file"]).is_file():
                continue
            with np.load(idx.parent / e["file"], allow_pickle=True) as z:
                ok, why = strict(z, js["phase_labels"])
            rows.append({"run": idx.parent.name, "start": _start(idx.parent.name),
                         "repeat": e.get("repeat"), "both": bool(e.get("success_both")),
                         "strict": ok, "strict_why": why,
                         "large": e.get("outcome_large"), "small": e.get("outcome_small"),
                         "wrap_l": _r(e.get("final_wrap_deg")),
                         "wrap_s": _r(e.get("final_wrap_small_deg")),
                         "duration_s": _r(e.get("duration_s")), "reason": e.get("reason"),
                         "npz": str(idx.parent / e["file"])})
    return rows


def suite_md(out) -> str | None:
    p = Path(out) / "hard_suite.md"
    return p.read_text() if p.is_file() else None


def episode_rows(out) -> list[dict]:
    """Scrub dropdown options ``{label, value=npz path}``."""
    return [{"label": f"{Path(p).parent.name}/{Path(p).name}", "value": str(p)}
            for idx in sorted(Path(out).glob("*/index.json"))
            for e in json.loads(idx.read_text()).get("episodes", [])
            if e.get("file") and (p := idx.parent / e["file"]).is_file()]


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict:
    with np.load(path, allow_pickle=True) as z:
        d = {k: z[k] for k in ("sim_time", "sim_belt_xyz", "sim_ee_franka", "sim_ee_ur",
                               "sim_wrap_deg", "sim_wrap_small_deg", "pcd", "sim_goal_dist",
                               "sim_latent_pred", "sim_latent") if k in z.files}
    pts = np.concatenate([d["sim_belt_xyz"].reshape(-1, 3), d["sim_ee_franka"][:, :3],
                          d["sim_ee_ur"][:, :3]])
    lo, hi = pts.min(0) - 0.05, pts.max(0) + 0.05
    d["range"] = [[float(a), float(b)] for a, b in zip(lo, hi)]
    return d


def load_episode(path) -> dict:
    """Cached arrays of one episode npz (re-read when the file changes)."""
    return _load(str(path), Path(path).stat().st_mtime)


def n_frames(path) -> int:
    return len(load_episode(path)["sim_time"])


def scrub_figures(path, i: int, note: str = "") -> tuple[go.Figure, go.Figure]:
    """Frame ``i``: 3D cloud + belt + EE markers; wrap angles vs t with a cursor at frame i."""
    d = load_episode(path)
    i = int(np.clip(i, 0, len(d["sim_time"]) - 1))
    traces = []
    pc = d["pcd"][i] if "pcd" in d else None
    if pc is not None and len(pc):
        pc = np.asarray(pc)[::math.ceil(len(pc) / MAX_CLOUD)]
        traces.append(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                                   name="cloud", marker={"size": 2, "color": "#999"}))
    b = d["sim_belt_xyz"][i]
    traces.append(go.Scatter3d(x=b[:, 0], y=b[:, 1], z=b[:, 2], mode="lines+markers",
                               name="belt", marker={"size": 3, "color": "#d62728"}))
    for k, name, c in (("sim_ee_franka", "Franka EE", "#1f77b4"),
                       ("sim_ee_ur", "UR EE", "#2ca02c")):
        p = d[k][i]
        traces.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name=name,
                                   marker={"size": 8, "color": c, "symbol": "diamond"}))
    fig = go.Figure(traces)
    rng = d["range"]
    fig.update_layout(
        uirevision="scrub", height=480, margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title=f"frame {i}" + (f" · {note}" if note else ""),
        scene={"aspectmode": "data", "xaxis": {"range": rng[0]}, "yaxis": {"range": rng[1]},
               "zaxis": {"range": rng[2]}})
    t = d["sim_time"] - d["sim_time"][0]
    wrap = go.Figure([go.Scatter(x=t, y=d[k], name=n) for k, n in
                      (("sim_wrap_deg", "wrap large"), ("sim_wrap_small_deg", "wrap small"))
                      if k in d])
    wrap.add_vline(x=float(t[i]), line_color="black", line_dash="dot")
    wrap.update_layout(height=260, margin={"l": 40, "r": 10, "t": 10, "b": 30},
                       xaxis_title="t (s)", yaxis_title="wrap (deg)", uirevision="wrap")
    return fig, wrap


def latent_figure(path, i: int) -> go.Figure:
    """Goal distance per stage and one-step latent prediction error vs t, cursor at frame i."""
    d = load_episode(path)
    if not all(k in d for k in ("sim_goal_dist", "sim_latent_pred", "sim_latent")):
        return go.Figure(layout={"title": "no latent data in this episode", "height": 360})
    t = d["sim_time"] - d["sim_time"][0]
    i = int(np.clip(i, 0, len(t) - 1))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    for k in range(d["sim_goal_dist"].shape[1]):
        fig.add_scatter(x=t, y=d["sim_goal_dist"][:, k], name=f"stage {k}", row=1, col=1)
    fig.add_scatter(x=t, y=np.linalg.norm(d["sim_latent_pred"] - d["sim_latent"], axis=1),
                    name="1-step pred err", row=2, col=1)
    fig.add_vline(x=float(t[i]), line_color="black", line_dash="dot")
    fig.update_layout(height=360, margin={"l": 50, "r": 10, "t": 10, "b": 30}, uirevision="latent")
    fig.update_yaxes(title_text="goal dist", row=1, col=1)
    fig.update_yaxes(title_text="latent units", row=2, col=1)
    fig.update_xaxes(title_text="t (s)", row=2, col=1)
    return fig
