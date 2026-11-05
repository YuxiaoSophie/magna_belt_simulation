"""Session episodes (demos, MILES, eval): scan, per-frame traces, clouds and their plotly figures."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from pathlib import Path

import lcm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.demo_gui import jobs
from apps.demo_gui.config import REPO
from apps.demo_gui.session import read_buffer, read_index

# isort: split
from task_common.lcm_contract import VIEWER_DRAW_CHANNEL, viewer_draw_msg  # src via config
from task_common.recording import Recording

UR_EXIT = 8  # sim_cmd_source code of the scripted UR exit (meta cmd_source_codes)
MAX_PTS = 2000
MAKE_TIER = REPO / "data/lcs/realfirst/cmp/v12_128/make_tier.py"


def _row(kind: str, d: Path, ep: str, **kw) -> dict:
    p = Path(ep) if Path(ep).is_absolute() else d / ep
    return {"kind": kind, "dir": str(d), "file": os.path.relpath(p, d), "path": str(p),
            "name": f"{d.name}/{p.stem}"} | kw


def _index_rows(kind: str, d: Path, prefix: str = "") -> list[dict]:
    return [_row(kind, d, e["file"], name=f"{prefix}{d.name}/{Path(e['file']).stem}", frames=e.get("frames"), outcome=e.get("outcome"),
                 success=e.get("success_both"),
                 start_state=(e.get("start_state") or {}).get("file"),
                 cmd_sources=e.get("cmd_source_counts") or {})
            # skipped / discarded rows have file None; listed but not on disk: skip
            for e in read_index(d) if e.get("file") and (d / e["file"]).is_file()]


def scan(session) -> list[dict]:
    """Every episode of the session: ``<s>/demos/*``, ``<s>/miles/*/buffer.json``, ``<s>/eval/*/*``."""
    rows = []
    for p in sorted(session.demos.glob("*/index.json")):
        rows += _index_rows("demo", p.parent)
    for p in sorted(session.miles.glob("*/buffer.json")):
        rows += [_row("miles", p.parent, r["episode"], frames=r.get("n_frames"),
                      outcome=r.get("outcome_gt"), success=None,
                      start_state=(r.get("start") or {}).get("file"), cmd_sources={})
                 for r in read_buffer(p)]
    for p in sorted(session.eval.glob("*/*/index.json")):
        rows += _index_rows("eval", p.parent, f"{p.parent.parent.name}/")
    return rows


def load_traces(npz) -> dict:
    """Full-rate per-frame signals of one episode npz; keys absent in older files read as NaN / 0."""
    with np.load(npz, allow_pickle=True) as z:
        n = len(z["sim_time"])

        def get(k, fill, dtype=np.float64):
            return np.asarray(z[k], dtype) if k in z.files else np.full(n, fill, dtype)
        labels = json.loads(str(z["sim_meta"])).get("phase_labels") if "sim_meta" in z.files else None
        phase = get("sim_phase", 0, np.int64)
        return {
            "t": get("sim_time", np.nan), "n_frames": n,
            "franka_xyz_mm": 1e3 * np.asarray(z["sim_ee_franka"], np.float64)[:, :3],
            "ur_xyz_mm": 1e3 * np.asarray(z["sim_ee_ur"], np.float64)[:, :3],
            "wrap_large_deg": get("sim_wrap_deg", np.nan),
            "wrap_small_deg": get("sim_wrap_small_deg", np.nan),
            "held": (np.asarray(z["sim_grasp_ok"], bool) if "sim_grasp_ok" in z.files
                     else np.zeros((n, 2), bool)),
            "phase": phase,
            "phase_label": [labels[p] if labels and 0 <= p < len(labels) else str(p) for p in phase],
            "cmd_source": get("sim_cmd_source", -1, np.int64),
        }


def ur_exit_mask(traces: dict) -> np.ndarray:
    """Frames of the scripted UR exit: never training data, shaded in the browser."""
    return np.asarray(traces["cmd_source"]) == UR_EXIT


@functools.lru_cache(maxsize=4)
def _clouds(npz: str) -> tuple:
    with np.load(npz, allow_pickle=True) as z:
        pc = z["pcd"] if "pcd" in z.files else z["pcd_belt"]
        return pc, (np.asarray(z["sim_belt_xyz"]) if "sim_belt_xyz" in z.files else None)


def cloud_at(npz, i: int) -> tuple[np.ndarray, np.ndarray | None]:
    """(measured cloud ``pcd[i]`` (N, 3) m, sim GT belt points (48, 3) m or None) at frame ``i``."""
    pc, belt = _clouds(str(npz))
    i = int(np.clip(i, 0, len(pc) - 1))
    return np.asarray(pc[i], np.float64)[:, :3], (None if belt is None else belt[i])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))


def traces_figure(tr: dict, frame: int | None = None, max_pts: int = MAX_PTS) -> go.Figure:
    """EE xyz (Franka, UR), wrap angles, held + phase strips vs t; UR-exit frames shaded."""
    k = np.arange(0, tr["n_frames"], max(1, -(-tr["n_frames"] // max_pts)))
    t = tr["t"][k]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.25, 0.25, 0.25, 0.12, 0.13],
                        subplot_titles=("Franka EE mm", "UR EE mm", "wrap deg", "held", "phase"))
    for row, key in ((1, "franka_xyz_mm"), (2, "ur_xyz_mm")):
        for j, ax in enumerate("xyz"):
            fig.add_scatter(x=t, y=tr[key][k, j], name=f"{key[:2]} {ax}", row=row, col=1)
    fig.add_scatter(x=t, y=tr["wrap_large_deg"][k], name="wrap large", row=3, col=1)
    fig.add_scatter(x=t, y=tr["wrap_small_deg"][k], name="wrap small", row=3, col=1)
    for j, arm in enumerate(("franka", "ur")):
        fig.add_scatter(x=t, y=tr["held"][k, j] * 0.8 + j, name=f"held {arm}", line_shape="hv",
                        row=4, col=1)
    fig.add_scatter(x=t, y=tr["phase"][k], text=[tr["phase_label"][i] for i in k],
                    hovertemplate="%{text}<extra></extra>", name="phase", line_shape="hv",
                    row=5, col=1)
    for a, b in _runs(ur_exit_mask(tr)):
        fig.add_vrect(x0=tr["t"][a], x1=tr["t"][b], fillcolor="#999", opacity=0.3, line_width=0,
                      annotation_text="UR exit: excluded from training",
                      annotation_position="top left", row="all", col=1)
    if frame is not None:
        fig.add_vline(x=tr["t"][int(np.clip(frame, 0, tr["n_frames"] - 1))], line_color="red")
    fig.update_layout(height=760, margin={"l": 40, "r": 10, "t": 30, "b": 30},
                      xaxis5_title="t s", uirevision="traces")
    return fig


def cloud_figure(pc: np.ndarray, belt: np.ndarray | None, title: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
                                 marker={"size": 2, "color": "#1f77b4"}, name="cloud"))
    if belt is not None:
        fig.add_scatter3d(x=belt[:, 0], y=belt[:, 1], z=belt[:, 2], mode="lines+markers",
                          marker={"size": 3, "color": "#d62728"}, name="belt (sim GT)")
    fig.update_layout(title=title, height=460, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                      scene={"aspectmode": "data"}, uirevision="cloud")
    return fig


def stop_replay(session) -> None:
    """One replay viewer per session: stops the ``replay`` job whichever panel started it."""
    p = session.jobs / "replay.json"
    if p.is_file() and jobs.status(job := jobs.load(p)) == "running":
        jobs.stop(job)


def recording_dir(npz) -> Path | None:
    """``<npz dir>/recordings/<npz stem>`` if recorded per episode, else the newest run there."""
    rec = Path(npz).parent / "recordings"
    if (rec / Path(npz).stem / "meta.json").is_file():
        return rec / Path(npz).stem
    # MILES recordings are per demo (<stamp>-miles_<demo>), not per episode: newest run then
    return next(iter(Recording.list_runs(rec)), None)


def start_replay(session, npz, port: int) -> jobs.Job:
    """Replay viewer on ``<npz dir>/recordings`` (``--run <stem>`` if recorded per episode)."""
    stop_replay(session)
    rec, run = Path(npz).parent / "recordings", Path(npz).stem
    argv = (["uv", "run", "--frozen", "python", "scripts/replay_viewer.py", "--recordings", rec,
             "--port", port] + (["--run", run] if recording_dir(npz) == rec / run else []))
    return jobs.start(session, "replay", argv)


@functools.lru_cache(maxsize=2)
def _recording(path: str, mtime: float) -> Recording:
    return Recording.load(Path(path))


@functools.cache
def _lcm(url: str) -> lcm.LCM:
    return lcm.LCM(url)


def publish_frame(session, npz, i: int, url: str) -> str:
    """Send frame ``i``'s recorded body poses to the standalone viewer; a short status, never raises."""
    try:
        if any((j := jobs.load(p)).lcm_url == url and jobs.status(j) == "running"
               for p in session.jobs.glob("*.json")):
            return "live run: not sent"  # a sim on this URL owns the viewer
        d = Path(npz).parent / "recordings" / Path(npz).stem
        if not (d / "meta.json").is_file():  # no newest-run fallback: wrong poses for this episode
            return "no recording"
        with np.load(npz, allow_pickle=True) as z:
            if "sim_step" not in z.files:
                return "no sim_step"
            steps = z["sim_step"]
        rec = _recording(str(d), (d / "meta.json").stat().st_mtime)
        f = rec.frame_at_step(int(steps[int(np.clip(i, 0, len(steps) - 1))]))
        _lcm(url).publish(VIEWER_DRAW_CHANNEL, viewer_draw_msg(
            int(rec.state_step[f] * rec.control_dt * 1e6), rec.meta["body_labels"],
            rec.body_q[f]).encode())
    except Exception as e:  # noqa: BLE001 - a bad recording must not break the scrub figures
        return f"viewer: {type(e).__name__}"
    return f"viewer: frame {f}"


def _make_tier():
    if "demo_gui_make_tier" not in sys.modules:
        spec = importlib.util.spec_from_file_location("demo_gui_make_tier", MAKE_TIER)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["demo_gui_make_tier"]


def stage1_input(episode, out_dir: Path, tag: str) -> tuple[str, str | None]:
    """``(episode, marks)`` for ``miles_seed --extra-demo``: whole-run episodes become stage-1 cuts.

    Cut by ``v12_128/make_tier.s1_cut`` (UR-exit start, grasp loss, belt checks) into
    ``out_dir/tiers/eps/s1/``; marks ``0:latch:hold`` by ``v12_147/s1_demo_copy.py``'s rule.
    Episodes without a chain (repo seed demos, ``move:*`` phases) pass through with no marks.
    """
    mt = _make_tier()
    mt.HERE = Path(out_dir)  # s1_cut writes under HERE
    res = mt.s1_cut({"src": str(episode), "tag": tag})
    if res.get("skip") == "no stage 1":
        return str(episode), None
    if "skip" in res:
        raise ValueError(f"{episode}: no usable stage 1 ({res['skip']})")
    with np.load(res["dst"], allow_pickle=True) as z:
        s = np.asarray(z["sim_stage"])
        if (s >= 3).any():
            hold = int(np.argmax(s >= 3))
        else:  # the baseline stalled short of place_3: the hold starts where both arms stop
            step = sum(np.linalg.norm(np.diff(z[k][:, :3], axis=0), axis=1)
                       for k in ("sim_ee_franka", "sim_ee_ur"))
            hold = int(np.flatnonzero(step > 0.5e-3)[-1]) + 1
    return res["dst"], f"0:{int(np.argmax(s >= 1))}:{hold}"
