"""Replay of collected LCS dataset episodes (``episode_*.npz``) with the one-step prediction.

:class:`EpisodeReplayApp` is a :class:`~task_common.replay_app.ReplayApp` whose "runs" are the
episode files of one directory: each frame's ``body_q`` is rebuilt by
:class:`~task_common.prediction_video.EpisodeScene` and frames play at the episode's sample
period. :class:`PredictionPanel` draws ``decode(LCS(encode(obs_k), u_k))`` (and optionally the
true belt k+1) as belt tubes and plots the per-frame one-step RMSE with a "now" cursor; both come
from :func:`~task_common.prediction_video.predict_episode`, computed once per episode.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from task_common import prediction_video as pv
from task_common.recording import Recording
from task_common.replay_app import ReplayApp
from task_common.replay_panels import NOW_SERIES_CSS, PLOT_PERIOD_S, TrajectoryChart

if TYPE_CHECKING:
    import newton

EPISODE_GLOB = "episode_*.npz"
RMSE_LABEL = "one-step prediction RMSE (mm)"
CACHE_EPISODES = 8
# Just over the 3.3 mm belt radius: the video's 1.5 mm tubes hide inside the scene belt.
PRED_TUBE_M, TRUE_TUBE_M = 0.0036, 0.0034


def list_episodes(root: Path) -> list[str]:
    return sorted(p.name for p in Path(root).glob(EPISODE_GLOB) if p.is_file())


class EpisodeRecording(Recording):
    """A :class:`Recording` view of one episode: frame k is sample k (``control_dt`` = dt)."""

    def __init__(self, path: Path, episode: pv.Episode, body_q: np.ndarray,
                 labels: list[str]) -> None:
        n = episode.frames
        steps = np.arange(n, dtype=np.int64)
        arrays = {"step": steps, "sim_time": steps * episode.dt, "state_step": steps,
                  "body_q": body_q, "wall_time": np.full(n, np.nan),
                  "compute_ms": np.full(n, np.nan)}
        meta = {"control_dt": episode.dt, "signal_coords": [], "body_labels": labels,
                "label": path.stem}
        super().__init__(path, meta, arrays, [], [])
        self.episode = episode

    @classmethod
    def load_episode(cls, path: Path, model: newton.Model) -> EpisodeRecording:
        ep = pv.Episode.load(path)
        scene = pv.EpisodeScene(model, ep)
        body_q = np.stack([scene.body_q(k) for k in range(ep.frames)])
        return cls(Path(path), ep, body_q, [str(b) for b in model.body_label])


class EpisodeReplayApp(ReplayApp):
    """The replay app over the episodes of ``root`` (only ``only`` if given)."""

    def __init__(self, root: Path, build_model, *, only: str | None = None,
                 **kwargs: Any) -> None:
        self._only = only
        kwargs["analysis"] = False  # the replay metrics need a recorded run's signals
        super().__init__(root, build_model, run=only, **kwargs)

    def runs(self) -> list[str]:
        names = list_episodes(self.recordings_root)
        return [n for n in names if n == self._only] if self._only else names

    def _load_checked(self, name: str) -> Recording:
        path = self.recordings_root / name
        if name not in self.runs():
            raise RuntimeError(f"no episode {name!r} under {self.recordings_root}")
        try:
            rec = EpisodeRecording.load_episode(path, self.model)
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"cannot load {name}: {exc!r}") from exc
        if rec.frame_count < 2:
            raise RuntimeError(f"{name}: {rec.frame_count} frame(s), need 2")
        return rec

    def _show_info(self, *, error: str | None = None, warning: str | None = None) -> None:
        rec = self.recording
        if isinstance(rec, EpisodeRecording):
            ep = rec.episode
            lines = [f"**{self.current_run}**", f"- dir: `{self.recordings_root}`",
                     f"- frames: {ep.frames} at dt {ep.dt:g} s ({ep.frames * ep.dt:.2f} s)",
                     f"- outcome: {ep.outcome}"]
        else:
            lines = [f"No episode loaded from `{self.recordings_root}`."]
        if error:
            lines.append(f"\n**Error:** {error}")
        self._info.content = "\n".join(lines)


class PredictionPanel:
    """Display toggles for the predicted / true belt k+1 and a ``One-step prediction`` folder
    with the whole-episode RMSE chart; the model is loaded once, predictions once per episode."""

    def __init__(self, deploy: Path = pv.DEPLOY, decoder: Path = pv.DECODER, *,
                 solver: str = "lcp", show_pred: bool = True, show_true: bool = False,
                 true_color=pv.GOLD, pred_color=pv.PRED_BLUE) -> None:
        self.model = pv.OneStepModel(deploy, decoder, solver)
        self.show = {"pred": bool(show_pred), "true": bool(show_true)}
        self.colors = {"pred": pred_color, "true": true_color}
        self.results: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.result: dict[str, Any] | None = None
        self.chart: TrajectoryChart | None = None
        self._time = 0.0
        self._last_plot = 0.0
        self._dirty = False

    def build_gui(self, app: ReplayApp) -> None:
        gui = app.gui
        self.tubes = {"pred": pv.BeltTubes(app.server, radius=PRED_TUBE_M),
                      "true": pv.BeltTubes(app.server, radius=TRUE_TUBE_M)}
        with app._display_folder:
            self.boxes = {"pred": gui.add_checkbox("Predicted belt k+1", self.show["pred"]),
                          "true": gui.add_checkbox("True belt k+1", self.show["true"])}
        with gui.add_folder("One-step prediction") as self.folder:
            self.status = gui.add_markdown("")
            self.now_text = gui.add_markdown("")
            gui.add_html(NOW_SERIES_CSS)
        for name, box in self.boxes.items():
            box.on_update(lambda e, n=name: app.call_soon(
                lambda v=bool(e.target.value): self.set_layer(app, n, v)))

    def set_layer(self, app: ReplayApp, name: str, shown: bool) -> None:
        self.show[name] = bool(shown)
        app._set_gui(self.boxes[name], self.show[name])
        self._draw(app.current_frame)

    def predict(self, path: Path, ep: pv.Episode) -> dict[str, Any]:
        """``predict_episode`` for ``path``, cached for the last few episodes."""
        key = str(Path(path).resolve())
        if key in self.results:
            self.results.move_to_end(key)
            return self.results[key]
        start = time.perf_counter()
        res = pv.predict_episode(self.model, ep)
        res["compute_s"] = time.perf_counter() - start
        self.results[key] = res
        while len(self.results) > CACHE_EPISODES:
            self.results.popitem(last=False)
        return res

    def on_run_loaded(self, app: ReplayApp, rec: Recording) -> None:
        if not isinstance(rec, EpisodeRecording):
            raise TypeError("PredictionPanel needs an EpisodeReplayApp")
        res = self.result = self.predict(rec.path, rec.episode)
        rmse = res["rmse_model_mm"]
        short = lambda p: f"{p.parent.name}/{p.name}"
        self.status.content = (
            f"model `{short(self.model.deploy)}`, `{short(self.model.decoder_path)}` "
            f"({self.model.solver})  \nRMSE mean {rmse.mean():.3f} / max {rmse.max():.3f} mm "
            f"(no-motion {res['rmse_nomotion_mm'].mean():.3f}); {res['compute_s']:.1f} s")
        logger.info(f"[REPLAY] one-step {rec.path.name}: RMSE mean {rmse.mean():.3f} / max "
                    f"{rmse.max():.3f} mm, {len(rmse)} predictions in {res['compute_s']:.1f} s "
                    f"({self.model.deploy}, {self.model.decoder_path})")
        if self.chart is not None:
            self.chart.remove()
        with self.folder:
            self.chart = TrajectoryChart(app.gui, res["time_s"], [rmse], ("RMSE mm",),
                                         title=None, y_label=RMSE_LABEL, aspect=0.9)

    def on_frame(self, app: ReplayApp, frame_index: int, step: int, sim_time: float) -> None:
        self._time = sim_time
        self._draw(frame_index)
        now = time.perf_counter()
        if app.playing and now - self._last_plot < PLOT_PERIOD_S:
            self._dirty = True
            return
        self._plot(now)

    def on_tick(self, app: ReplayApp) -> None:
        if self._dirty and (not app.playing
                            or time.perf_counter() - self._last_plot >= PLOT_PERIOD_S):
            self._plot(time.perf_counter())

    def _plot(self, now: float) -> None:
        self._last_plot, self._dirty = now, False
        if self.chart is not None:
            self.chart.set_time(self._time)

    def _draw(self, k: int) -> None:
        res = self.result
        valid = res is not None and k < len(res["pred"])
        for name, key in (("pred", "pred"), ("true", "true_next")):
            tubes = self.tubes[name]
            if valid and self.show[name]:
                tubes.tube(name, res[key][k], self.colors[name])
            elif name in tubes.handles:
                tubes.handles[name].visible = False
        if res is not None:
            self.now_text.content = (f"k = {k}: RMSE {res['rmse_model_mm'][k]:.3f} mm" if valid
                                     else f"k = {k}: last frame, no k+1")
