"""Replay app panels (:class:`~task_common.replay_app.ReplayHook`): events, charts, triads.

GUI callbacks only queue work with ``app.call_soon``; viser handles are touched on the main thread.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from viser import uplot

from task_common.recording import Event, Recording
from task_common.replay_metrics import (
    event_summary,
    target_channels,
    target_world_pose,
    xyzw_to_wxyz,
)

if TYPE_CHECKING:
    from task_common.replay_app import ReplayApp

BLANK_EVENT = "(events)"
MAX_EVENT_ROWS = 200
MAX_LABEL_DATA = 60
PLOT_PERIOD_S = 0.2
MAX_PLOT_POINTS = 300
DEFAULT_WINDOW_S = 10.0
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04")
NOW_COLOR = "#111827"
BLANK_TRIAD = "(pick a body or target)"
REMOVE_LABEL = "✕"
DEFAULT_AXES_LENGTH = 0.05
DEFAULT_AXES_RADIUS = 0.002
TRIAD_ROOT = "/replay/triads"
TRIAD_ROW_SELECTOR = '.mantine-Flex-root:has(> div > p > label[for="{uuid}"])'
TRIAD_ROW_CSS = (
    "{row} > div:first-child {{ width: auto !important; flex: 1 1 auto; min-width: 0; }}",
    "{row} > div:first-child label {{ white-space: nowrap; }}",
    "{row} > div:first-child p {{ overflow: hidden; text-overflow: ellipsis; }}",
    "{row} > div:last-child {{ flex: 0 0 auto !important; }}",
    "{row} > div:last-child button {{ flex: 0 0 auto !important; min-width: 2em; }}",
)


def event_label(event: Event) -> str:
    summary = event_summary(event)
    if len(summary) > MAX_LABEL_DATA:
        summary = summary[:MAX_LABEL_DATA - 3] + "..."
    return f"{event.sim_time:8.3f}s  {event.kind}  {summary}".rstrip()


class EventsPanel:
    """``Events`` folder: a jump-to dropdown and a table of recorded + derived events."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    def build_gui(self, app: ReplayApp) -> None:
        gui = app.gui
        with gui.add_folder("Events", expand_by_default=False):
            self.dropdown = gui.add_dropdown("Jump to", (BLANK_EVENT,))
            self.table = gui.add_markdown("")
        self.dropdown.on_update(lambda e: self._on_pick(app, e.target.value))

    def on_run_loaded(self, app: ReplayApp, rec: Recording) -> None:
        labels, seen = [], set()
        for i, event in enumerate(app.all_events):
            label = event_label(event)
            if label in seen:
                label = f"{label}  #{i}"
            seen.add(label)
            labels.append(label)
        self.labels = labels
        app._syncing = True
        try:
            self.dropdown.options = (BLANK_EVENT, *labels)
            self.dropdown.value = BLANK_EVENT
        finally:
            app._syncing = False
        self.table.content = self._table(app.all_events)

    def on_frame(self, app: ReplayApp, frame_index: int, step: int, sim_time: float) -> None:
        pass

    def _on_pick(self, app: ReplayApp, value: str) -> None:
        if value == BLANK_EVENT or value not in self.labels:
            return
        index = self.labels.index(value)
        app.call_soon(lambda: self._jump(app, index))

    def _jump(self, app: ReplayApp, index: int) -> None:
        app.jump_to_event(index)
        # Blank again so picking the same event twice still fires on_update.
        app._set_gui(self.dropdown, BLANK_EVENT)

    @staticmethod
    def _table(events: list[Event]) -> str:
        if not events:
            return "No events."
        lines = [f"{len(events)} events" + (f" (first {MAX_EVENT_ROWS} shown)"
                                            if len(events) > MAX_EVENT_ROWS else ""),
                 "", "| t s | step | kind | data |", "|---:|---:|---|---|"]
        for event in events[:MAX_EVENT_ROWS]:
            data = event_summary(event).replace("|", "/")
            lines.append(f"| {event.sim_time:.3f} | {event.step} | {event.kind} | {data} |")
        return "\n".join(lines)


def _franka_efforts(rec: Recording) -> tuple[np.ndarray, list[np.ndarray]]:
    layout = rec.meta.get("effort_layout", [])
    columns = [i for i, name in enumerate(layout) if name.startswith("franka/")]
    return rec.step * rec.control_dt, [rec.signals["efforts"][:, c] for c in columns]


# name, title, shown by default, series labels, source(rec) -> (x, per-series y arrays)
CHARTS = (
    ("Franka efforts", "Franka efforts N m", True, tuple(f"j{i}" for i in range(1, 8)),
     _franka_efforts),
)


@dataclass
class _Chart:
    name: str
    title: str
    labels: tuple[str, ...]
    source: Callable[[Recording], tuple[np.ndarray, list[np.ndarray]]]
    checkbox: Any
    handle: Any = None
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    series: list[np.ndarray] = field(default_factory=list)
    y_range: tuple[float, float] = (0.0, 1.0)
    sent_key: tuple | None = None


class PlotsPanel:
    """``Plots`` folder: a sliding window of uPlot charts around the current time."""

    def __init__(self) -> None:
        self.window_s = DEFAULT_WINDOW_S
        self.charts: dict[str, _Chart] = {}
        self._last_update = 0.0
        self._dirty = False
        self._time = 0.0
        self.data_sends = 0
        self.scale_sends = 0

    def build_gui(self, app: ReplayApp) -> None:
        gui = app.gui
        with gui.add_folder("Plots"):
            self.window = gui.add_number("Window s", initial_value=DEFAULT_WINDOW_S, min=1.0,
                                         max=120.0, step=1.0)
            for name, title, shown, labels, source in CHARTS:
                checkbox = gui.add_checkbox(name, shown)
                self.charts[name] = _Chart(name, title, labels, source, checkbox)
            for chart in self.charts.values():
                chart.handle = self._make_handle(app, chart)
        self.window.on_update(
            lambda e: app.call_soon(lambda v=float(e.target.value): self.set_window(app, v)))
        for chart in self.charts.values():
            chart.checkbox.on_update(
                lambda e, c=chart: app.call_soon(
                    lambda v=bool(e.target.value): self.set_visible(app, c.name, v)))

    @staticmethod
    def _make_handle(app: ReplayApp, chart: _Chart) -> Any:
        series = [uplot.Series(label="t s")]
        for i, label in enumerate(chart.labels):
            series.append(uplot.Series(label=label, stroke=COLORS[i % len(COLORS)], width=1.5))
        series.append(uplot.Series(label="now", stroke=NOW_COLOR, width=0,
                                   points={"show": True, "size": 8, "fill": NOW_COLOR}))
        data = (np.array([0.0, 1.0]), *([np.full(2, np.nan)] * (len(series) - 1)))
        return app.gui.add_uplot(data=data, series=tuple(series),
                                 scales=PlotsPanel._scales(chart.y_range), title=chart.title,
                                 aspect=2.0, visible=chart.checkbox.value)

    @staticmethod
    def _scales(y_range: tuple[float, float]) -> dict[str, Any]:
        # Fixed per run: uPlot's auto-range blanks on NaN, and new scales rebuild the chart.
        return {"x": uplot.Scale(time=False), "y": uplot.Scale(auto=False, range=y_range)}

    @staticmethod
    def _range(values: list[np.ndarray]) -> tuple[float, float]:
        finite = [v[np.isfinite(v)] for v in values]
        finite = [v for v in finite if v.size]
        if not finite:
            return 0.0, 1.0
        lo = float(min(v.min() for v in finite))
        hi = float(max(v.max() for v in finite))
        pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 0.5)
        return lo - pad, hi + pad

    def on_run_loaded(self, app: ReplayApp, rec: Recording) -> None:
        for chart in self.charts.values():
            x, series = chart.source(rec)
            width = len(chart.labels)
            series = [np.ascontiguousarray(v[:len(x)], dtype=np.float64) for v in series[:width]]
            series += [np.full(len(x), np.nan)] * (width - len(series))
            chart.x = np.asarray(x, dtype=np.float64)
            chart.series = series
            chart.sent_key = None
            y_range = self._range(series)
            if y_range != chart.y_range:
                chart.y_range = y_range
                chart.handle.scales = self._scales(y_range)
                self.scale_sends += 1
        self._dirty = True

    def on_frame(self, app: ReplayApp, frame_index: int, step: int, sim_time: float) -> None:
        self._time = sim_time
        now = time.perf_counter()
        if app.playing and now - self._last_update < PLOT_PERIOD_S:
            self._dirty = True
            return
        self.update(now)

    def on_tick(self, app: ReplayApp) -> None:
        if self._dirty and (not app.playing
                            or time.perf_counter() - self._last_update >= PLOT_PERIOD_S):
            self.update()

    def update(self, now: float | None = None) -> None:
        self._last_update = time.perf_counter() if now is None else now
        self._dirty = False
        for chart in self.charts.values():
            if chart.handle.visible and len(chart.x):
                key, data = self._window_data(chart, self._time)
                if key != chart.sent_key:
                    chart.sent_key = key
                    chart.handle.data = data
                    self.data_sends += 1

    def _window_data(self, chart: _Chart, t: float) -> tuple[tuple, tuple[np.ndarray, ...]]:
        """``(key, data)`` of the window around ``t``; an equal key means identical data."""
        x = chart.x
        half = 0.5 * self.window_s
        i0 = int(np.searchsorted(x, t - half, side="left"))
        i1 = int(np.searchsorted(x, t + half, side="right"))
        if i1 <= i0:
            i0 = min(i0, len(x) - 1)
            i1 = i0 + 1
        ys = [y[i0:i1] for y in chart.series]
        xs = x[i0:i1]
        if i1 - i0 > MAX_PLOT_POINTS:
            xs, ys = _min_max_decimate(xs, ys, MAX_PLOT_POINTS // 2)
        # The "now" point sits on the first series (0 where that is not finite).
        anchor = np.where(np.isfinite(ys[0]), ys[0], 0.0) if ys else np.zeros(len(xs))
        now = np.full(len(xs), np.nan)
        k = int(np.argmin(np.abs(xs - t)))
        now[k] = anchor[k]
        return (i0, i1, k), (xs, *ys, now)

    def set_window(self, app: ReplayApp, seconds: float) -> None:
        self.window_s = float(np.clip(seconds, 1.0, 120.0))
        app._set_gui(self.window, self.window_s)
        self.update()

    def set_visible(self, app: ReplayApp, name: str, visible: bool) -> None:
        chart = self.charts[name]
        chart.handle.visible = bool(visible)
        chart.sent_key = None
        app._set_gui(chart.checkbox, bool(visible))
        if visible:
            self.update()


def _min_max_decimate(x: np.ndarray, ys: list[np.ndarray],
                      buckets: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per bucket, each series' min and max in time order (keeps spikes) at the bucket's ends."""
    n = len(x)
    size = math.ceil(n / buckets)
    count = math.ceil(n / size)
    starts = np.arange(count) * size
    ends = np.minimum(starts + size, n) - 1
    xs = np.column_stack((x[starts], x[ends])).ravel()
    pad = count * size - n
    out = []
    for y in ys:
        grid = np.concatenate((y, np.full(pad, np.nan))).reshape(count, size)
        nan = np.isnan(grid)
        lo_i = np.argmin(np.where(nan, np.inf, grid), axis=1)
        hi_i = np.argmax(np.where(nan, -np.inf, grid), axis=1)
        rows = np.arange(count)
        lo, hi = grid[rows, lo_i], grid[rows, hi_i]
        lo_first = lo_i <= hi_i
        out.append(np.column_stack((np.where(lo_first, lo, hi),
                                    np.where(lo_first, hi, lo))).ravel())
    return xs, out


def _sanitise(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "frame"


class TriadsPanel:
    """``Triads`` folder: the dropdown adds a body or target frame, its row's X removes it."""

    def __init__(self) -> None:
        self.axes_length = DEFAULT_AXES_LENGTH
        self.axes_radius = DEFAULT_AXES_RADIUS
        self.enabled: set[str] = set()
        self.handles: dict[str, Any] = {}
        self.remove_buttons: dict[str, Any] = {}
        self.body_index: dict[str, int] = {}
        self.target_channel: dict[str, str] = {}
        self.target_available: dict[str, bool] = {}

    def build_gui(self, app: ReplayApp) -> None:
        gui = app.gui
        with gui.add_folder("Triads", expand_by_default=False) as folder:
            self.folder = folder
            self.length_input = gui.add_number("Axes length m", initial_value=DEFAULT_AXES_LENGTH,
                                               min=0.005, max=0.5, step=0.005)
            self.radius_input = gui.add_number("Axes radius m", initial_value=DEFAULT_AXES_RADIUS,
                                               min=0.0002, max=0.05, step=0.0002)
            self.picker = gui.add_dropdown("Add triad", (BLANK_TRIAD,), initial_value=BLANK_TRIAD)
            self.row_style = gui.add_html("")
        self.length_input.on_update(lambda e: app.call_soon(
            lambda v=float(e.target.value): self.set_axes_length(app, v)))
        self.radius_input.on_update(lambda e: app.call_soon(
            lambda v=float(e.target.value): self.set_axes_radius(app, v)))
        self.picker.on_update(lambda e: app.call_soon(
            lambda v=str(e.target.value): self._pick(app, v)))

    def on_run_loaded(self, app: ReplayApp, rec: Recording) -> None:
        self.body_index = {label: i for i, label in enumerate(rec.meta["body_labels"])}
        channels = {msg.channel for msg in rec.targets}
        self.target_channel = target_channels(rec)
        self.target_available = {name: channel in channels
                                 for name, channel in self.target_channel.items()}
        targets = [name for name, ok in self.target_available.items() if ok]
        app._set_gui(self.picker, BLANK_TRIAD)
        self.picker.options = (BLANK_TRIAD, *targets, *self.body_index)
        for name in list(self.enabled):
            if name not in self.body_index and not self.target_available.get(name, False):
                self.set_enabled(app, name, False)
            else:
                self._update_one(app, name, app.current_frame, app.current_step)

    def _pick(self, app: ReplayApp, name: str) -> None:
        if name == BLANK_TRIAD:
            return
        # Blank again so picking the same item after removing it still fires on_update.
        app._set_gui(self.picker, BLANK_TRIAD)
        self.set_enabled(app, name, True)

    def set_enabled(self, app: ReplayApp, name: str, enabled: bool) -> None:
        if name in self.target_available:
            enabled = enabled and self.target_available[name]
        elif name not in self.body_index:
            raise ValueError(f"unknown triad {name!r}")
        if not enabled:
            self.enabled.discard(name)
            if name in self.handles:
                self.handles[name].visible = False
            if name in self.remove_buttons:
                self.remove_buttons.pop(name).remove()
                self._restyle_rows()
            return
        self.enabled.add(name)
        if name not in self.remove_buttons:
            with self.folder:
                button = app.gui.add_button_group(name, (REMOVE_LABEL,), hint=name)
            button.on_click(lambda _, n=name: app.call_soon(
                lambda: self.set_enabled(app, n, False)))
            self.remove_buttons[name] = button
            self._restyle_rows()
        if name not in self.handles:
            self.handles[name] = app.server.scene.add_frame(
                f"{TRIAD_ROOT}/{_sanitise(name)}", axes_length=self.axes_length,
                axes_radius=self.axes_radius, visible=False)
        if app.recording is not None:
            self._update_one(app, name, app.current_frame, app.current_step)

    def _restyle_rows(self) -> None:
        # viser fixes the label column at ~7 em and stretches buttons: widen the name, shrink the X.
        rules = [rule.format(row=TRIAD_ROW_SELECTOR.format(uuid=button._impl.uuid))
                 for button in self.remove_buttons.values() for rule in TRIAD_ROW_CSS]
        self.row_style.content = f"<style>{' '.join(rules)}</style>" if rules else ""

    def set_axes_length(self, app: ReplayApp, value: float) -> None:
        self.axes_length = float(value)
        app._set_gui(self.length_input, self.axes_length)
        for handle in self.handles.values():
            handle.axes_length = self.axes_length

    def set_axes_radius(self, app: ReplayApp, value: float) -> None:
        self.axes_radius = float(value)
        app._set_gui(self.radius_input, self.axes_radius)
        for handle in self.handles.values():
            handle.axes_radius = self.axes_radius
            handle.origin_radius = 2.0 * self.axes_radius

    def shown(self) -> dict[str, Any]:
        return {name: handle for name, handle in self.handles.items()
                if name in self.enabled and handle.visible}

    def on_frame(self, app: ReplayApp, frame_index: int, step: int, sim_time: float) -> None:
        for name in self.enabled:
            if name in self.handles:
                self._update_one(app, name, frame_index, step)

    def _update_one(self, app: ReplayApp, name: str, frame: int, step: int) -> None:
        rec, handle = app.recording, self.handles[name]
        if name in self.body_index:
            row = rec.body_q[frame, self.body_index[name]].astype(np.float64)
            handle.position = row[:3]
            handle.wxyz = xyzw_to_wxyz(row[3:7])
            handle.visible = True
            return
        pose = target_world_pose(rec, self.target_channel[name], step, frame)
        if pose is None:
            handle.visible = False
            return
        handle.position, handle.wxyz = pose
        handle.visible = True
