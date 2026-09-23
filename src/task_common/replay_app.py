"""Viser replay of recorded runs: run selector, timeline scrubbing and 3D playback.

The model is rebuilt from the scene (no physics) and each recorded ``body_q`` frame is drawn
with ``ViewerViser``.  GUI callbacks run on viser's threads and only queue requests
(:attr:`ReplayApp.pending`, :meth:`ReplayApp.call_soon`); :meth:`ReplayApp.tick` applies them
on the main thread, which does every Newton/warp call.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import newton
import newton.viewer
import numpy as np
import warp as wp
from loguru import logger

from task_common import REPO_ROOT
from task_common.point_cloud import CroppedPointCloud
from task_common.recording import Event, Recording, file_digest
from task_common.replay_metrics import Metrics, compute_metrics, derive_events
from task_common.replay_panels import PlotsPanel, TriadsPanel

SPEED_OPTIONS = ("0.1x", "0.25x", "0.5x", "1x", "2x", "4x")
TRANSPORT_OPTIONS = ("|<", "-10", "-1", "Play", "+1", "+10", ">|")
TEXT_PERIOD_S = 0.1
MAX_TICK_DT_S = 0.1
DEFAULT_RENDER_FPS = 25.0
DEFAULT_DEVICE = "cpu"
STATS_PERIOD_S = 10.0
DEFAULT_POINT_SIZE_MM = 1.0

# A private model (on CUDA when available: ~5 ms a frame vs ~0.5 s on CPU) plus its clouds.
PointCloudFactory = Callable[[], tuple[newton.Model, list[CroppedPointCloud]]]


class ReplayHook(Protocol):
    """Extension panel: builds its GUI once, then follows run loads and rendered frames."""

    def build_gui(self, app: ReplayApp) -> None: ...

    def on_run_loaded(self, app: ReplayApp, rec: Recording) -> None: ...

    def on_frame(self, app: ReplayApp, frame_index: int, step: int, sim_time: float) -> None: ...

    # Optional: on_tick(app), called every tick after the pending requests are applied.


def merged_events(rec: Recording, derived: list[Event]) -> list[Event]:
    """Recorded + derived events by step; repeated identical ``hand_command`` goals dropped."""
    recorded, last_goal = [], None
    for event in rec.events:
        if event.kind == "hand_command":
            goal = (event.data.get("target_mm"), event.data.get("force"))
            if goal == last_goal:
                continue
            last_goal = goal
        recorded.append(event)
    return sorted(recorded + derived, key=lambda e: e.step)


class ReplayApp:
    """A ViewerViser that plays back the ``body_q`` frames of recordings under a root dir."""

    def __init__(self, recordings_root: Path, build_model: Callable[[], newton.Model], *,
                 port: int = 8081, show_collision: bool = False, run: str | None = None,
                 verbose: bool = True, hooks: Sequence[ReplayHook] = (),
                 analysis: bool = True, render_fps: float = DEFAULT_RENDER_FPS,
                 device: str | None = DEFAULT_DEVICE,
                 build_point_clouds: PointCloudFactory | None = None,
                 show_point_cloud: bool = False) -> None:
        if not render_fps > 0.0:
            raise ValueError(f"render_fps must be > 0, got {render_fps}")
        self.recordings_root = Path(recordings_root)
        self.recording: Recording | None = None
        self.current_run: str | None = None
        self.current_frame = 0
        self.playing = False
        self.speed = 1.0
        self._loop = False
        self._play_time = 0.0
        self._last_tick: float | None = None
        self._last_text = 0.0
        self.render_fps = float(render_fps)
        self._since_render = 0.0
        self.render_count = 0
        self._render_ms: list[float] = []
        self._syncing = False
        self._closed = False
        self._lock = threading.Lock()
        self.pending: dict[str, Any] = {}
        self.hooks: list[ReplayHook] = []
        self.metrics: Metrics | None = None
        self.all_events: list[Event] = []
        self._analysis = analysis
        self._plots_panel = self._triads_panel = None
        self._build_point_clouds = build_point_clouds
        self._clouds: list[CroppedPointCloud] | None = None
        self._cloud_state: newton.State | None = None
        self.show_point_cloud = False
        self.point_size_mm = DEFAULT_POINT_SIZE_MM

        start = time.perf_counter()
        # CPU by default: on a GPU every frame costs ~100 synchronous device-to-host copies.
        with wp.ScopedDevice(device):
            self.viewer = newton.viewer.ViewerViser(port=port, label="Newton replay",
                                                    verbose=verbose)
            self.server = self.viewer._server
            self.gui = self.server.gui
            self.model = build_model()
            self.viewer.set_model(self.model)
            if show_collision:
                # After set_model: attaching the model resets the viewer layer's defaults.
                self.viewer.show_collision = True
            self.state = self.model.state()
        self._model_labels = [str(label) for label in self.model.body_label]
        self._build_gui()
        if show_point_cloud:
            self.set_display(point_cloud=True)
        if analysis:
            self._plots_panel, self._triads_panel = PlotsPanel(), TriadsPanel()
            for hook in (self._plots_panel, self._triads_panel):
                self.add_hook(hook)
        for hook in hooks:
            self.add_hook(hook)
        names = self.runs()
        if run is not None or names:
            self.select_run(run or names[0])
        else:
            self._render(0)
        logger.info(f"[REPLAY] ready in {time.perf_counter() - start:.1f} s on "
                    f"{self.model.device}: {len(names)} run(s) under {self.recordings_root}, "
                    f"3D at <= {self.render_fps:g} fps during playback")

    # ---- headless API -------------------------------------------------------------------

    def runs(self) -> list[str]:
        """Run directory names under the recordings root, newest first."""
        return [p.name for p in Recording.list_runs(self.recordings_root)]

    @property
    def frame_count(self) -> int:
        return 0 if self.recording is None else self.recording.frame_count

    @property
    def current_step(self) -> int:
        if self.recording is None or self.frame_count == 0:
            return 0
        return int(self.recording.state_step[self.current_frame])

    @property
    def current_time(self) -> float:
        return self._frame_time(self.current_frame)

    @property
    def loop(self) -> bool:
        return self._loop

    @loop.setter
    def loop(self, value: bool) -> None:
        self._loop = bool(value)
        self._set_gui(self._loop_box, self._loop)

    def select_run(self, name: str) -> None:
        """Load run ``name``; on any error the previous run stays (load errors: RuntimeError)."""
        start = time.perf_counter()
        try:
            rec = self._load_checked(name)
        except RuntimeError as exc:
            logger.error(f"[REPLAY] {exc}")
            self._show_info(error=str(exc))
            self._set_gui(self._run_dropdown, self.current_run)
            raise
        metrics, events = None, []
        if self._analysis:
            metrics_start = time.perf_counter()
            metrics = compute_metrics(rec)
            derived = derive_events(rec, metrics)
            events = merged_events(rec, derived)
            logger.info(f"[REPLAY] analysis: {len(events)} events ({len(derived)} "
                        f"derived), {len(rec.targets)} target messages, "
                        f"{time.perf_counter() - metrics_start:.2f} s")
        previous = (self.current_run, self.recording, self.metrics, self.all_events)
        previous_frame = self.current_frame
        try:
            self._install(name, rec, metrics, events)
        except Exception:
            if previous[1] is None:
                self.current_run, self.recording, self.metrics, self.all_events = (
                    None, None, None, [])
                self.current_frame = 0
                raise
            try:
                self._install(*previous)
                self.seek(previous_frame)
            except Exception as exc:  # noqa: BLE001 - keep the original error
                logger.opt(exception=exc).error(f"[REPLAY] restoring {previous[0]} failed")
            raise
        logger.info(f"[REPLAY] loaded {name}: {rec.frame_count} frames, {rec.duration_s:.1f} s "
                    f"in {time.perf_counter() - start:.2f} s")

    def _install(self, name: str, rec: Recording, metrics: Metrics | None,
                 events: list[Event]) -> None:
        self.recording = rec
        self.current_run = name
        self.metrics = metrics
        self.all_events = events
        # Before any hook runs: the previous run's frame may be past this run's end.
        self.current_frame = 0
        self._play_time = self._frame_time(0)
        self.playing = False
        warning = self._scene_warning(rec)
        if warning:
            logger.warning(f"[REPLAY] {warning}")
        self._show_info(warning=warning)
        self._syncing = True
        try:
            if name not in self._run_dropdown.options:
                self._run_dropdown.options = self.runs()
            self._run_dropdown.value = name
            self._frame_slider.value = 0
            self._frame_slider.max = max(rec.frame_count - 1, 0)
            self._transport.options = TRANSPORT_OPTIONS
        finally:
            self._syncing = False
        for hook in self.hooks:
            hook.on_run_loaded(self, rec)
        self.seek(0)

    def seek(self, frame: int) -> None:
        """Show ``frame`` (clamped) now; playback continues from there if playing."""
        if self.recording is None:
            return
        frame = int(np.clip(frame, 0, max(self.frame_count - 1, 0)))
        self._play_time = self._frame_time(frame)
        self._render(frame)

    def step_frames(self, n: int) -> None:
        self.seek(self.current_frame + int(n))

    def set_playing(self, playing: bool) -> None:
        self.playing = bool(playing) and self.recording is not None
        self._last_tick = time.perf_counter()
        self._set_gui_options()
        self._sync_timeline(force=True)

    def set_speed(self, speed: float) -> None:
        if not speed > 0.0:
            raise ValueError(f"speed must be > 0, got {speed}")
        self.speed = float(speed)
        label = f"{self.speed:g}x"
        if label in SPEED_OPTIONS:
            self._set_gui(self._speed_dropdown, label)

    def set_display(self, *, visual: bool | None = None, collision: bool | None = None,
                    point_cloud: bool | None = None,
                    point_size_mm: float | None = None) -> None:
        """Show or hide visual meshes, collision shapes and the camera point clouds."""
        if point_size_mm is not None:
            if not point_size_mm > 0.0:
                raise ValueError(f"point_size_mm must be > 0, got {point_size_mm}")
            self.point_size_mm = float(point_size_mm)
            self._set_gui(self._point_size_slider, self.point_size_mm)
        if visual is not None:
            self.viewer.show_visual = bool(visual)
            self._set_gui(self._visual_box, self.viewer.show_visual)
        if collision is not None:
            self.viewer.show_collision = bool(collision)
            self._set_gui(self._collision_box, self.viewer.show_collision)
        if point_cloud is not None:
            if point_cloud and self._build_point_clouds is None:
                raise RuntimeError("no point clouds: ReplayApp(build_point_clouds=None)")
            if point_cloud and self._clouds is None:
                self._init_point_clouds()
            self.show_point_cloud = bool(point_cloud)
            self._set_gui(self._cloud_box, self.show_point_cloud)
        self._render(self.current_frame)

    def tick(self, wall_dt: float | None = None) -> None:
        """Apply pending GUI requests, then advance playback by ``speed * wall_dt`` seconds.

        During playback the scene is redrawn at most ``render_fps`` times per second of
        ``wall_dt``; seeks, steps, the loop wrap and the last frame always redraw at once.
        """
        now = time.perf_counter()
        if wall_dt is None:
            # Clamped so a stall (run load, slow client) does not jump the playback.
            wall_dt = 0.0 if self._last_tick is None else min(now - self._last_tick,
                                                               MAX_TICK_DT_S)
        self._last_tick = now
        self._apply_pending()
        for hook in self.hooks:
            on_tick = getattr(hook, "on_tick", None)
            if on_tick is not None:
                on_tick(self)
        if not self.playing or self.recording is None or self.frame_count == 0:
            return
        self._play_time += self.speed * wall_dt
        self._since_render += wall_dt
        last = self.frame_count - 1
        if self._play_time >= self._frame_time(last):
            if self._loop:
                self.seek(0)
            else:
                self.seek(last)
                self.set_playing(False)
            return
        frame = self.recording.frame_at_time(self._play_time)
        # Small tolerance: 50 Hz ticks against a 25 fps budget land a hair short of 40 ms.
        if frame != self.current_frame and self._since_render >= 0.95 / self.render_fps:
            self._render(frame)

    def run_forever(self, fps: float = 50.0, stats: bool = False) -> None:
        """Tick at up to ``fps`` until the viewer closes or Ctrl-C; ``stats`` logs rates."""
        period = 1.0 / fps
        ticks: list[float] = []
        stats_start, counts = time.perf_counter(), self._stat_counts()
        self._render_ms = []
        try:
            while self.viewer.is_running():
                start = time.perf_counter()
                self.tick()
                ticks.append(time.perf_counter() - start)
                if stats and start - stats_start >= STATS_PERIOD_S:
                    self._log_stats(start - stats_start, ticks, counts)
                    ticks, stats_start, counts = [], start, self._stat_counts()
                time.sleep(max(period - (time.perf_counter() - start), 0.0))
        except KeyboardInterrupt:
            logger.info("[REPLAY] interrupted")
        finally:
            self.close()

    def _stat_counts(self) -> tuple[int, int, int]:
        data, scales = (0, 0) if self._plots_panel is None else self.plot_send_counts()
        return self.render_count, data, scales

    def _log_stats(self, elapsed: float, ticks: list[float],
                   before: tuple[int, int, int]) -> None:
        renders, data, scales = (b - a for a, b in zip(before, self._stat_counts(), strict=True))
        if not renders and not data:
            return  # idle: nothing worth a line
        tick_ms = 1e3 * np.asarray(ticks or [0.0])
        render_ms = np.asarray(self._render_ms or [0.0])
        self._render_ms = []
        logger.info(f"[REPLAY] stats: {len(ticks) / elapsed:.1f} ticks/s (mean "
                    f"{tick_ms.mean():.1f} / max {tick_ms.max():.1f} ms), {renders / elapsed:.1f} "
                    f"renders/s (mean {render_ms.mean():.1f} / max {render_ms.max():.1f} ms), "
                    f"plot updates {data / elapsed:.1f}/s, scale resends {scales}, "
                    f"{len(self.server.get_clients())} client(s)")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.viewer.close()

    def call_soon(self, fn: Callable[[], None]) -> None:
        """Queue ``fn`` for the main thread's next :meth:`tick` (GUI callbacks use this)."""
        if self._syncing:
            return
        with self._lock:
            self.pending.setdefault("calls", []).append(fn)

    # ---- analysis API (``analysis=True``) ------------------------------------------------

    def triads(self) -> dict[str, Any]:
        """Triad name -> viser frame handle, for the triads currently enabled and shown."""
        return self._panel("_triads_panel").shown()

    def set_triad(self, name: str, enabled: bool) -> None:
        """Show or remove a body triad (any body label) or a target triad (by its name)."""
        self._panel("_triads_panel").set_enabled(self, name, enabled)

    def set_axes_length(self, value: float) -> None:
        self._panel("_triads_panel").set_axes_length(self, value)

    def plot_handles(self) -> dict[str, Any]:
        """Chart name -> ``GuiUplotHandle``."""
        return {name: c.handle for name, c in self._panel("_plots_panel").charts.items()}

    def plot_send_counts(self) -> tuple[int, int]:
        """``(data, scales)`` updates sent to the browser by the charts so far."""
        panel = self._panel("_plots_panel")
        return panel.data_sends, panel.scale_sends

    def set_plot_window(self, seconds: float) -> None:
        self._panel("_plots_panel").set_window(self, seconds)

    def _panel(self, attr: str) -> Any:
        panel = getattr(self, attr)
        if panel is None:
            raise RuntimeError("analysis panels are off (ReplayApp(analysis=False))")
        return panel

    def add_hook(self, hook: ReplayHook) -> None:
        """Register a panel; it is caught up with the current run and frame at once."""
        self.hooks.append(hook)
        hook.build_gui(self)
        if self.recording is not None:
            hook.on_run_loaded(self, self.recording)
            hook.on_frame(self, self.current_frame, self.current_step, self.current_time)

    # ---- internals ----------------------------------------------------------------------

    def _frame_time(self, frame: int) -> float:
        if self.recording is None or self.frame_count == 0:
            return 0.0
        return float(self.recording.state_step[frame]) * self.recording.control_dt

    def _init_point_clouds(self) -> None:
        start = time.perf_counter()
        device = "cuda:0" if wp.is_cuda_available() else self.model.device
        with wp.ScopedDevice(device):
            model, self._clouds = self._build_point_clouds()
            self._cloud_state = model.state()
        names = [cloud.spec.name for cloud in self._clouds]
        logger.info(f"[REPLAY] point clouds {names} on {model.device} in "
                    f"{time.perf_counter() - start:.1f} s")

    def _log_point_clouds(self, frame: int) -> None:
        if not self.show_point_cloud:
            for cloud in self._clouds:
                self.viewer.log_points(cloud.spec.name, None, hidden=True)
            return
        if self.recording is not None and self.frame_count:
            self._cloud_state.body_q.assign(np.ascontiguousarray(self.recording.body_q[frame]))
        for cameras in {id(c.cameras): c.cameras for c in self._clouds}.values():
            cameras.update(self._cloud_state)
        for cloud in self._clouds:
            cloud.log(self.viewer, point_size=1e-3 * self.point_size_mm)

    def _load_checked(self, name: str) -> Recording:
        path = self.recordings_root / name
        if not (path / "meta.json").is_file():
            raise RuntimeError(f"no run {name!r} under {self.recordings_root}")
        try:
            rec = Recording.load(path)
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"cannot load {name}: {exc}") from exc
        labels = list(rec.meta.get("body_labels", []))
        if labels != self._model_labels:
            for i, (ours, theirs) in enumerate(zip(self._model_labels, labels, strict=False)):
                if ours != theirs:
                    raise RuntimeError(f"{name}: body {i} is {theirs!r} in the recording but "
                                       f"{ours!r} in the model")
            raise RuntimeError(f"{name}: {len(labels)} recorded bodies, the model has "
                               f"{len(self._model_labels)}")
        if rec.frame_count == 0:
            raise RuntimeError(f"{name}: no body_q frames recorded")
        return rec

    @staticmethod
    def _scene_warning(rec: Recording) -> str | None:
        scene = rec.meta.get("scene_directives")
        if not scene:
            return None
        path = Path(scene["path"])
        path = path if path.is_absolute() else REPO_ROOT / path
        if not path.is_file():
            return f"scene file {scene['path']} not found; geometry may differ"
        if file_digest(path)["sha256"] != scene["sha256"]:
            return f"{scene['path']} changed since recording; geometry may differ"
        return None

    def _render(self, frame: int) -> None:
        start = time.perf_counter()
        self.current_frame = frame
        self._since_render = 0.0
        sim_time = self._frame_time(frame)
        self.viewer.begin_frame(sim_time)
        if self.recording is not None and self.frame_count:
            self.state.body_q.assign(np.ascontiguousarray(self.recording.body_q[frame]))
        self.viewer.log_state(self.state)
        if self._clouds is not None:
            self._log_point_clouds(frame)
        self.viewer.end_frame()
        self._sync_timeline(force=not self.playing)
        for hook in self.hooks:
            hook.on_frame(self, frame, self.current_step, sim_time)
        self.render_count += 1
        if len(self._render_ms) < 10000:  # emptied by the --stats log
            self._render_ms.append(1e3 * (time.perf_counter() - start))

    def _set_gui(self, handle: Any, value: Any) -> None:
        """Assign a widget from the main thread without re-entering our own callbacks."""
        if value is None or handle.value == value:
            return
        self._syncing = True
        try:
            handle.value = value
        finally:
            self._syncing = False

    def _set_gui_options(self) -> None:
        options = tuple("Pause" if o == "Play" and self.playing else o for o in TRANSPORT_OPTIONS)
        if self._transport.options != options:
            self._transport.options = options

    def _sync_timeline(self, force: bool) -> None:
        now = time.perf_counter()
        if not force and now - self._last_text < TEXT_PERIOD_S:
            return
        self._last_text = now
        self._set_gui(self._frame_slider, self.current_frame)
        self._time_text.content = (f"`t = {self.current_time:.3f} s  step {self.current_step}  "
                                   f"frame {self.current_frame}/{max(self.frame_count - 1, 0)}`")

    def _show_info(self, *, error: str | None = None, warning: str | None = None) -> None:
        lines = []
        rec = self.recording
        if rec is not None:
            meta = rec.meta
            state = "finished" if meta.get("finished") else "unfinished"
            if meta.get("reason"):
                state += f" ({meta['reason']})"
            commit = str(meta.get("git_commit") or "?")[:10]
            if meta.get("git_dirty"):
                commit += " (dirty)"
            resyncs = sum(1 for e in rec.events if e.kind == "resync")
            lines += [
                f"**{self.current_run}**",
                f"- label: {meta.get('label')}",
                f"- created: {meta.get('created')}",
                f"- duration: {rec.duration_s:.2f} s",
                f"- steps: {len(rec.step)}",
                f"- frames: {rec.frame_count} (state_every {meta.get('state_every')})",
                f"- {state}",
                f"- git: {commit}",
                f"- resyncs: {resyncs}",
            ]
        else:
            lines.append(f"No run loaded from `{self.recordings_root}`.")
        if warning:
            self._info_warning = warning
        elif rec is not None and error is None:
            self._info_warning = None
        if self._info_warning:
            lines.append(f"\n**Warning:** {self._info_warning}")
        if error:
            lines.append(f"\n**Error:** {error}")
        self._info.content = "\n".join(lines)

    def _request(self, key: str, value: Any) -> None:
        if self._syncing:
            return
        with self._lock:
            self.pending[key] = value

    def _on_transport(self, value: str) -> None:
        if self._syncing:
            return
        with self._lock:
            if value in ("Play", "Pause"):
                self.pending["toggle"] = not self.pending.get("toggle", False)
            elif value == "|<":
                self.pending["seek"] = 0
                self.pending.pop("step", None)
            elif value == ">|":
                self.pending["seek"] = -1
                self.pending.pop("step", None)
            else:
                self.pending["step"] = self.pending.get("step", 0) + int(value)

    def _apply_pending(self) -> None:
        with self._lock:
            pending, self.pending = self.pending, {}
        if not pending:
            return
        if pending.get("rescan"):
            self._rescan()
        if "run" in pending and pending["run"] != self.current_run:
            self._select_run_guarded(pending["run"])
        if "speed" in pending:
            self.set_speed(float(pending["speed"].rstrip("x")))
        if "loop" in pending:
            self.loop = pending["loop"]
        display = {k: pending[k] for k in ("visual", "collision", "point_cloud", "point_size_mm")
                   if k in pending}
        if display:
            self.set_display(**display)
        if "seek" in pending:
            frame = pending["seek"]
            self.seek(self.frame_count - 1 if frame < 0 else frame)
        if "step" in pending:
            self.set_playing(False)
            self.step_frames(pending["step"])
        if pending.get("toggle"):
            if not self.playing and self.current_frame >= self.frame_count - 1:
                self.seek(0)
            self.set_playing(not self.playing)
        for fn in pending.get("calls", ()):
            fn()

    def _rescan(self) -> None:
        names = self.runs()
        self._syncing = True
        try:
            self._run_dropdown.options = names or ("(no runs)",)
            if self.current_run in names:
                self._run_dropdown.value = self.current_run
        finally:
            self._syncing = False
        logger.info(f"[REPLAY] rescan: {len(names)} run(s)")
        if self.current_run is None and names:
            self._select_run_guarded(names[0])

    def _select_run_guarded(self, name: str) -> None:
        """``select_run`` for GUI requests: an error is logged and never ends ``run_forever``."""
        try:
            self.select_run(name)
        except RuntimeError:
            pass  # already logged and shown; the previous run stays
        except Exception as exc:  # noqa: BLE001 - a hook bug must not kill the viewer
            logger.opt(exception=exc).error(f"[REPLAY] loading {name} failed: {exc!r}")
            self._show_info(error=f"loading {name} failed: {exc!r}")
            self._set_gui(self._run_dropdown, self.current_run)

    def _build_gui(self) -> None:
        gui = self.gui
        names = self.runs()
        self._info_warning: str | None = None
        with gui.add_folder("Recording"):
            self._run_dropdown = gui.add_dropdown("Run", names or ("(no runs)",))
            self._rescan_button = gui.add_button("Rescan")
            self._info = gui.add_markdown("")
        with gui.add_folder("Timeline"):
            self._frame_slider = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self._time_text = gui.add_markdown("")
            self._transport = gui.add_button_group("Transport", TRANSPORT_OPTIONS)
            self._speed_dropdown = gui.add_dropdown("Speed", SPEED_OPTIONS, initial_value="1x")
            self._loop_box = gui.add_checkbox("Loop", False)
        with gui.add_folder("Display"):
            self._visual_box = gui.add_checkbox("Visual meshes", self.viewer.show_visual)
            self._collision_box = gui.add_checkbox("Collision geometry",
                                                   self.viewer.show_collision)
            self._cloud_box = gui.add_checkbox("Point cloud", False,
                                               disabled=self._build_point_clouds is None)
            self._point_size_slider = gui.add_slider(
                "Point size [mm]", min=0.2, max=2.0, step=0.05,
                initial_value=DEFAULT_POINT_SIZE_MM,
                disabled=self._build_point_clouds is None)

        self._run_dropdown.on_update(lambda e: self._request("run", e.target.value))
        self._rescan_button.on_click(lambda _: self._request("rescan", True))
        # Slider drags fire continuously: the last value wins, rendered once per tick.
        self._frame_slider.on_update(lambda e: self._request("seek", int(e.target.value)))
        self._transport.on_click(lambda e: self._on_transport(e.target.value))
        self._speed_dropdown.on_update(lambda e: self._request("speed", e.target.value))
        self._loop_box.on_update(lambda e: self._request("loop", bool(e.target.value)))
        for key, box in (("visual", self._visual_box), ("collision", self._collision_box),
                         ("point_cloud", self._cloud_box)):
            box.on_update(lambda e, key=key: self._request(key, bool(e.target.value)))
        self._point_size_slider.on_update(
            lambda e: self._request("point_size_mm", float(e.target.value)))
        self._show_info()
