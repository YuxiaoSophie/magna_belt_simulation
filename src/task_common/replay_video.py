"""Frame-exact MP4 export of replayed runs, rendered on the server.

A headless Chromium is the viser client: each output frame seeks the :class:`ReplayApp`,
waits until the scene update is on the client's socket (:func:`sync_scene`), then asks that
client for a render at a fixed camera and pipes the RGB frame to ``ffmpeg``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import viser
import viser.transforms as vtf
from loguru import logger

from task_common.replay_learned_mpc import (
    ACT_FRANKA,
    ACT_UR,
    AUG_MARKER,
    FRANKA_PLAN,
    PLAN_ALPHA,
    PLAN_BLUE,
    TARGET_GREEN,
    LearnedMpcPanel,
    _alphas,
    _tube_faces,
    tube_vertices,
)

FFMPEG = "/usr/bin/ffmpeg"
CHROMIUM = "chromium-browser"
CRF = 20
VIDEO_TUBE_STRIDE = 1  # videos draw the full 150-point belts
VIDEO_TUBE_SIDES = 8
BACKGROUND = 255  # PNG renders are transparent where nothing is drawn


@dataclass(frozen=True)
class CameraPose:
    """A fixed viewpoint; ``fov`` is the vertical field of view in radians (as viser)."""

    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 0.7854

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraPose:
        missing = {"position", "look_at"} - data.keys()
        if missing:
            raise ValueError(f"camera json lacks {sorted(missing)}")
        vec = lambda key, default=None: tuple(float(v) for v in data.get(key, default))
        return cls(vec("position"), vec("look_at"), vec("up", (0.0, 0.0, 1.0)),
                   float(data.get("fov", 0.7854)))

    @classmethod
    def load(cls, text: str) -> CameraPose:
        """From a json file path or the json text itself (the viewer's Display > Camera)."""
        text = text.strip()
        return cls.from_dict(json.loads(text if text.startswith("{") else Path(text).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {"position": list(self.position), "look_at": list(self.look_at),
                "up": list(self.up), "fov": self.fov}

    def wxyz(self) -> np.ndarray:
        """viser's camera orientation: z forward, y down (see ``CameraHandle._update_wxyz``)."""
        z = np.subtract(self.look_at, self.position).astype(np.float64)
        z /= np.linalg.norm(z)
        up = np.asarray(self.up, dtype=np.float64)
        y = -(up - (up @ z) * z)
        if np.linalg.norm(y) < 1e-9:
            raise ValueError("camera up is parallel to the view direction")
        y /= np.linalg.norm(y)
        return vtf.SO3.from_matrix(np.stack([np.cross(y, z), y, z], axis=1)).wxyz


# The user's view, low from the front right (copied from the viewer's Display > Camera).
DEFAULT_CAMERA = CameraPose(position=(0.622, -0.2402, 0.1969), look_at=(0.3984, 0.1624, -0.068),
                            up=(0.0, 0.0, 1.0), fov=1.309)


class VideoLearnedMpcPanel(LearnedMpcPanel):
    """The Learned MPC panel with its own tube resolution (the live viewer keeps the light
    default)."""

    def __init__(self, *, tube_stride: int = VIDEO_TUBE_STRIDE,
                 tube_sides: int = VIDEO_TUBE_SIDES, **kwargs: Any) -> None:
        if tube_stride < 1 or tube_sides < 3:
            raise ValueError(f"tube stride >= 1 and sides >= 3, got {tube_stride}, {tube_sides}")
        super().__init__(**kwargs)
        self.tube_stride, self.tube_sides = int(tube_stride), int(tube_sides)

    def _tube(self, app: Any, name: str, pts: np.ndarray, radius: float, color,
              opacity: float) -> Any:
        pts = np.asarray(pts)[::self.tube_stride]
        return self._mesh(app, name, tube_vertices(pts, radius, self.tube_sides),
                          _tube_faces(len(pts), self.tube_sides), color, opacity)


LEGEND_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
LEGEND_TEXT = (17, 24, 39)
LEGEND_PANEL = (255, 255, 255, 217)  # white at 0.85
LEGEND_CORNERS = ("tl", "tr", "bl", "br")


@dataclass(frozen=True)
class LegendEntry:
    """One legend row: ``kind`` is ``belt`` (bar), ``fade`` (bar in ``alphas`` segments),
    ``dots`` or ``arrows`` (one per colour)."""

    label: str
    kind: str
    colors: tuple[tuple[int, int, int], ...]
    alphas: tuple[float, ...] = ()


def belt_rgb(color: Sequence[float]) -> tuple[int, int, int]:
    """A scene shape colour (0-1) as displayed: newton sends ``uint8(c * 255)`` and viser's
    batched meshes take it as linear, so the screen shows its sRGB encoding ((1, .5, 0) is gold).
    """
    c = (np.asarray(color[:3], dtype=np.float64) * 255).astype(np.uint8) / 255.0
    srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return tuple(int(v) for v in np.round(srgb * 255))


def legend_entries(panel: LearnedMpcPanel, belt: tuple[int, int, int]) -> list[LegendEntry]:
    """The recorded belt plus the learned layers shown for the loaded run; empty if it has no
    solves."""
    if not panel.solves:
        return []
    out = [LegendEntry("current belt", "belt", (belt,))]
    if panel.shown("planned_belt"):
        s = panel.solves[0]
        n = len(s.belts) - 1 if s.belts is not None else s.n
        dt = float(np.median(np.diff(s.t))) if len(s.t) > 1 else float("nan")
        out.append(LegendEntry(f"planned belt (MPC horizon, {n} steps \u00d7 {dt:.3g} s)",
                               "fade", (PLAN_BLUE,), tuple(_alphas(n, PLAN_ALPHA))))
    if panel.shown("target_belt"):
        out.append(LegendEntry("target belt", "belt", (TARGET_GREEN,)))
    if panel.shown("planned_ee"):
        out.append(LegendEntry("planned EE (Franka plan knots, MPC state)", "dots",
                               (FRANKA_PLAN, AUG_MARKER)))
    if panel.shown("actions"):
        out.append(LegendEntry("planned actions (Franka, UR)", "arrows", (ACT_FRANKA, ACT_UR)))
    return out


def _over_white(rgb: Sequence[int], alpha: float) -> tuple[int, int, int, int]:
    return (*(round(alpha * c + (1.0 - alpha) * 255) for c in rgb), 255)


def render_legend(entries: Sequence[LegendEntry], frame_height: int) -> np.ndarray:
    """(h, w, 4) uint8 RGBA legend panel, text sized to ``frame_height`` (19 px at 720)."""
    from PIL import Image, ImageDraw, ImageFont

    size = max(10, round(frame_height * 0.026))
    font = (ImageFont.truetype(str(LEGEND_FONT), size) if LEGEND_FONT.is_file()
            else ImageFont.load_default(size))
    pad, gap, row = round(size * 0.7), round(size * 0.6), round(size * 1.5)
    sw, bar = round(size * 2.8), max(4, round(size * 0.4))
    probe_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text_w = max(round(probe_draw.textlength(e.label, font=font)) for e in entries)
    w, h = pad + sw + gap + text_w + pad, 2 * pad + row * len(entries) - (row - size)
    img = Image.new("RGBA", (w, h), LEGEND_PANEL)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, h - 1), outline=(156, 163, 175, 255))
    for i, e in enumerate(entries):
        cy, x0 = pad + row * i + size // 2, pad
        if e.kind in ("belt", "fade"):
            alphas = e.alphas or (1.0,)
            edges = np.linspace(x0, x0 + sw, len(alphas) + 1).round().astype(int)
            for a, xa, xb in zip(alphas, edges[:-1], edges[1:], strict=True):
                draw.rectangle((xa, cy - bar // 2, xb - 1, cy - bar // 2 + bar - 1),
                               fill=_over_white(e.colors[0], a))
        elif e.kind == "dots":
            r = max(2, round(size * 0.25))
            for j, c in enumerate(e.colors):
                cx = x0 + round(sw * (j + 0.5) / len(e.colors))
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*c, 255))
        elif e.kind == "arrows":
            head, shaft = max(4, round(size * 0.4)), max(1, round(size * 0.1))
            span = sw / len(e.colors)
            for j, c in enumerate(e.colors):
                xa, xb = round(x0 + j * span), round(x0 + (j + 1) * span) - gap // 2
                draw.rectangle((xa, cy - shaft, xb - head, cy + shaft), fill=(*c, 255))
                half = round(head * 0.7)
                draw.polygon([(xb - head, cy - half), (xb, cy), (xb - head, cy + half)],
                             fill=(*c, 255))
        else:
            raise ValueError(f"unknown legend kind {e.kind!r}")
        draw.text((x0 + sw + gap, cy), e.label, fill=(*LEGEND_TEXT, 255), font=font, anchor="lm")
    return np.asarray(img)


class Overlay:
    """Alpha-blends a fixed RGBA image into a corner of every frame (in place)."""

    def __init__(self, rgba: np.ndarray, width: int, height: int, corner: str = "tr",
                 margin: int = 12) -> None:
        if corner not in LEGEND_CORNERS:
            raise ValueError(f"corner must be one of {LEGEND_CORNERS}, got {corner!r}")
        h, w = rgba.shape[:2]
        if w + 2 * margin > width or h + 2 * margin > height:
            raise ValueError(f"legend {w}x{h} does not fit a {width}x{height} frame")
        self.x = margin if corner[1] == "l" else width - w - margin
        self.y = margin if corner[0] == "t" else height - h - margin
        self.rgb = rgba[..., :3].astype(np.uint16)
        self.alpha = rgba[..., 3:4].astype(np.uint16)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.rgb.shape[:2]
        region = frame[self.y:self.y + h, self.x:self.x + w]
        region[:] = ((self.rgb * self.alpha + region * (255 - self.alpha) + 127) // 255)
        return frame


INSET_BORDER = (75, 85, 99)

@dataclass(frozen=True)
class Inset:
    """A small extra view: ``label=<camera json or path>``."""

    label: str
    camera: CameraPose

    @classmethod
    def parse(cls, text: str) -> Inset:
        label, sep, cam = text.partition("=")
        if not sep or not label.strip() or not cam.strip():
            raise ValueError(f"inset must be LABEL=<camera json or path>, got {text!r}")
        return cls(label.strip(), CameraPose.load(cam))


def render_label(text: str, frame_height: int) -> np.ndarray:
    """(h, w, 4) uint8 RGBA text tag, 15 px at 720."""
    from PIL import Image, ImageDraw, ImageFont

    size = max(9, round(frame_height * 0.021))
    font = (ImageFont.truetype(str(LEGEND_FONT), size) if LEGEND_FONT.is_file()
            else ImageFont.load_default(size))
    pad = max(2, round(size * 0.35))
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
        (0, 0), text, font=font)
    img = Image.new("RGBA", (right - left + 2 * pad, bottom - top + 2 * pad), LEGEND_PANEL)
    ImageDraw.Draw(img).text((pad - left, pad - top), text, fill=(*LEGEND_TEXT, 255), font=font)
    return np.asarray(img)


class InsetLayout:
    """Picture-in-picture windows stacked in the top-right corner, each ``fraction`` of the
    frame width at the frame's aspect, shrunk if needed to end above ``bottom`` (the legend).

    Insets are rendered at the full frame size and box-downsampled: any other size makes the
    client resize its WebGL canvas, which leaks GPU memory in Chromium (see HeadlessBrowser).
    """

    def __init__(self, insets: Sequence[Inset], width: int, height: int, *,
                 bottom: int | None = None, fraction: float = 0.25, margin: int = 12,
                 gap: int = 10, border: int = 2) -> None:
        self.insets, self.border, self.frame = list(insets), border, (width, height)
        n = len(self.insets)
        room = (height if bottom is None else bottom) - 2 * margin - (n - 1) * gap
        h = min(round(width * fraction * height / width), room // n - 2 * border)
        w = round(h * width / height)
        if h < 32:
            raise ValueError(f"{n} insets do not fit a {width}x{height} frame")
        self.w, self.h = w, h
        x = width - margin - w - 2 * border
        self.origins = [(x, margin + i * (h + 2 * border + gap)) for i in range(n)]
        self.labels = [Overlay(render_label(i.label, height), w, h, "tl", margin=4)
                       for i in self.insets]

    @property
    def cameras(self) -> list[CameraPose]:
        return [i.camera for i in self.insets]

    @property
    def render_size(self) -> tuple[int, int]:
        return self.frame

    def rects(self) -> list[tuple[int, int, int, int]]:
        """(x0, y0, x1, y1) of each window, border included, exclusive ends."""
        b = 2 * self.border
        return [(x, y, x + self.w + b, y + self.h + b) for x, y in self.origins]

    def __call__(self, frame: np.ndarray, renders: Sequence[np.ndarray]) -> np.ndarray:
        from PIL import Image

        b = self.border
        for (x0, y0, x1, y1), render, label in zip(self.rects(), renders, self.labels,
                                                    strict=True):
            small = np.asarray(Image.fromarray(render).resize((self.w, self.h),
                                                              Image.Resampling.BOX))
            frame[y0:y1, x0:x1] = INSET_BORDER
            frame[y0 + b:y1 - b, x0 + b:x1 - b] = label(small.copy())
        return frame


def sync_scene(server: viser.ViserServer, client: viser.ClientHandle,
               timeout: float = 30.0) -> None:
    """Return once every scene update queued so far is written to ``client``'s socket.

    ``get_render`` flushes the broadcast buffer but its request rides the per-client buffer,
    so it can overtake a scene update; this closes that race.
    """
    buf = server._websock_server.get_message_buffer()
    target = buf.message_counter - 1
    server.flush()
    deadline = time.perf_counter() + timeout
    while buf.generator_cursors.get(client.client_id, -1) < target:
        if client.client_id not in server._connected_clients:
            raise RuntimeError("render client disconnected")
        if time.perf_counter() > deadline:
            raise TimeoutError(f"scene sync: client stuck before message {target}")
        time.sleep(2e-4)
    # The producer sends a window in the same loop step that advances the cursor.
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), server._event_loop).result(timeout)


def capture(client: viser.ClientHandle, camera: CameraPose, width: int, height: int,
            timeout: float = 60.0) -> np.ndarray:
    """(H, W, 3) uint8 render of the client's scene from ``camera``, over a white background."""
    rgba = client.get_render(height, width, wxyz=camera.wxyz(), position=camera.position,
                             fov=float(camera.fov), transport_format="png",
                             timeout=timeout)
    alpha = rgba[..., 3:4].astype(np.uint16)
    rgb = (rgba[..., :3] * alpha + BACKGROUND * (255 - alpha) + 127) // 255
    return np.ascontiguousarray(rgb.astype(np.uint8))


BROWSER_CHROME_PX = 87  # --headless=new counts the (unseen) toolbar in --window-size


class HeadlessBrowser:
    """A headless Chromium on ``url`` as the render client; stopped by its own pid on exit.

    The page canvas is sized to exactly ``width`` x ``height`` so every capture of that size
    renders without resizing the WebGL canvas: in this Chromium (ANGLE/Vulkan) each resize
    leaks ~25-50 MB of GPU memory, and viser's ``get_render`` resizes twice per capture of any
    other size, so long renders ran the GPU out of memory and lost the context (blank frames).
    """

    def __init__(self, server: viser.ViserServer, url: str, width: int, height: int,
                 timeout: float = 90.0, debug_port: int | None = None,
                 log: Path | None = None) -> None:
        self.server, self.url, self.timeout = server, url, timeout
        self.debug_port = debug_port  # Chrome DevTools, for checks
        self.log = log  # chromium stderr (default: discarded)
        self.size = (width, height)
        self.window = (width, height + BROWSER_CHROME_PX)
        self.proc: subprocess.Popen | None = None
        self.profile: Path | None = None
        self.client: viser.ClientHandle | None = None
        self.restarts = 0

    def __enter__(self) -> viser.ClientHandle:
        try:
            for attempt in range(2):
                self._launch()
                canvas = self._canvas_size()
                if canvas == self.size:
                    break
                logger.warning(f"[VIDEO] page canvas {canvas}, want {self.size}"
                               + ("; relaunching" if attempt == 0 else
                                  "; captures resize the canvas (leaks GPU memory)"))
                if attempt == 0:
                    self.stop()
                    self.window = (self.window[0] + self.size[0] - canvas[0],
                                   self.window[1] + self.size[1] - canvas[1])
        except BaseException:
            self.stop()
            raise
        return self.client

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def restart(self) -> viser.ClientHandle:
        """A fresh Chromium (new GPU context); the new client gets the current scene."""
        self.stop()
        self.restarts += 1
        return self.__enter__()

    def _launch(self) -> None:
        before = set(self.server.get_clients())
        # The snap Chromium can only write under ~/snap/chromium/common.
        self.profile = (Path.home() / "snap/chromium/common"
                        / f"replay-video-{os.getpid()}-{time.monotonic_ns()}")
        err = open(self.log, "ab") if self.log else subprocess.DEVNULL  # noqa: SIM115
        self.proc = subprocess.Popen(
            [CHROMIUM, "--headless=new", f"--user-data-dir={self.profile}",
             f"--window-size={self.window[0]},{self.window[1]}", "--force-device-scale-factor=1",
             "--enable-gpu", "--use-angle=vulkan", "--enable-features=Vulkan",
             "--ignore-gpu-blocklist", "--disable-background-timer-throttling",
             "--disable-renderer-backgrounding", "--no-first-run",
             *([f"--remote-debugging-port={self.debug_port}"] if self.debug_port else []),
             self.url],
            stdout=subprocess.DEVNULL, stderr=err, start_new_session=True)
        if self.log:
            err.close()
        logger.info(f"[VIDEO] headless chromium pid {self.proc.pid} -> {self.url}")
        deadline = time.perf_counter() + self.timeout
        while True:
            new = [c for i, c in self.server.get_clients().items() if i not in before]
            if new:
                self.client = new[0]
                return
            if self.proc.poll() is not None:
                raise RuntimeError(f"chromium exited with {self.proc.returncode}")
            if time.perf_counter() > deadline:
                raise TimeoutError(f"no viser client from chromium in {self.timeout:g} s")
            time.sleep(0.1)

    def _canvas_size(self, timeout: float = 20.0) -> tuple[int, int]:
        """The client's canvas size once it has held still for 0.5 s."""
        deadline, last, since = time.perf_counter() + timeout, None, time.perf_counter()
        while time.perf_counter() < deadline:
            cam = self.client.camera
            size = (int(cam.image_width), int(cam.image_height))
            if size != last:
                last, since = size, time.perf_counter()
            elif size[0] > 0 and time.perf_counter() - since > 0.5:
                break
            time.sleep(0.05)
        return last

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)  # our own session only
                self.proc.wait(10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(5)
                except ProcessLookupError:
                    pass
        if self.profile is not None:
            shutil.rmtree(self.profile, ignore_errors=True)
        self.proc = self.client = None


class Encoder:
    """RGB frames on stdin -> ``ffmpeg`` libx264 yuv420p (or lossless intermediate)."""

    def __init__(self, path: Path, width: int, height: int, fps: float, *,
                 vf: str | None = None, lossless: bool = False, crf: int = CRF) -> None:
        self.path, self.frames = Path(path), 0
        quality = (["-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv444p"] if lossless
                   else ["-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"])
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
               *(["-vf", vf] if vf else []), "-c:v", "libx264", *quality,
               "-movflags", "+faststart", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> None:
        _, err = self.proc.communicate()
        if self.proc.returncode:
            raise RuntimeError(f"ffmpeg failed ({self.proc.returncode}): {err.decode()[-2000:]}")


def has_drawtext() -> bool:
    out = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True,
                         check=False)
    return " drawtext " in out.stdout


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")


def text_filter(text: str, x: str, y: str = "12", size: int = 24) -> str:
    return (f"drawtext=text='{_escape(text)}':x={x}:y={y}:fontsize={size}:fontcolor=black"
            ":box=1:boxcolor=white@0.7:boxborderw=6")


def time_filter(start_s: float, step_s: float, size: int = 24) -> str:
    """``t = 3.2 s`` in the bottom-left corner, from the output frame number ``n``."""
    ds = f"round(({start_s!r}+n*{step_s!r})*10)"  # whole tenths: no float truncation
    text = (f"t = %{{eif\\:trunc({ds}/10)\\:d}}.%{{eif\\:mod({ds}\\,10)\\:d}} s")
    return (f"drawtext=text='{text}':x=12:y=h-th-12:fontsize={size}:fontcolor=black"
            ":box=1:boxcolor=white@0.7:boxborderw=6")


def frame_times(duration_s: float, fps: float, speed: float, start_s: float = 0.0,
                end_s: float | None = None) -> np.ndarray:
    """Recording-relative times of the output frames: ``start_s + n * speed / fps``."""
    end = duration_s if end_s is None else min(float(end_s), duration_s)
    if not end >= start_s:
        raise ValueError(f"empty time range [{start_s}, {end}]")
    step = speed / fps
    return start_s + step * np.arange(int(np.floor((end - start_s) / step + 1e-6)) + 1)


def run_duration(app: Any) -> float:
    """Time of the last recorded frame relative to the first."""
    return app._frame_time(app.frame_count - 1) - app._frame_time(0)


BLANK_STD = 1.0  # luma std of a scene render is ~60; a lost WebGL context renders uniform white
RENDER_RETRIES = 3


class RenderFailed(RuntimeError):
    """A frame stayed blank (or the capture failed) after every browser restart."""


def check_render(rgb: np.ndarray) -> None:
    """Raise :class:`RenderFailed` on a uniform render (lost WebGL context, empty page)."""
    luma = rgb[::4, ::4].astype(np.float32) @ np.float32([0.299, 0.587, 0.114])
    if luma.std() < BLANK_STD:
        raise RenderFailed(f"blank render (luma std {luma.std():.2f}, mean {luma.mean():.1f})")


def render_run(app: Any, browser: HeadlessBrowser, camera: CameraPose, times: Sequence[float],
               width: int, height: int, sink: Callable[[np.ndarray], None],
               insets: InsetLayout | None = None, retries: int = RENDER_RETRIES) -> list[float]:
    """Render the loaded run at ``times`` (relative s) into ``sink``; returns ms per frame.

    Every raw render is checked before it is composited; a bad one restarts Chromium and
    re-renders that frame, up to ``retries`` times, then raises :class:`RenderFailed`.
    """
    t0, rec = app._frame_time(0), app.recording
    cameras = [(camera, width, height)] + [(c, *insets.render_size) for c in
                                           (insets.cameras if insets else [])]
    per_frame = []
    for n, t in enumerate(times):
        start = time.perf_counter()
        app.seek(rec.frame_at_time(t0 + float(t)))
        for attempt in range(retries + 1):
            try:
                sync_scene(app.server, browser.client)
                # Each request carries its own camera and nothing is queued between them.
                renders = [capture(browser.client, *cam) for cam in cameras]
                for render in renders:
                    check_render(render)
                break
            except (RenderFailed, RuntimeError, TimeoutError) as exc:
                if attempt == retries:
                    raise RenderFailed(f"frame {n} (t {t:.2f} s): {exc}; gave up after "
                                       f"{retries} browser restarts") from exc
                logger.warning(f"[VIDEO] frame {n} (t {t:.2f} s): {exc}; restarting chromium "
                               f"({attempt + 1}/{retries})")
                browser.restart()
                for cam in cameras:
                    warm_up(app, browser.client, *cam)
        frame = renders[0]
        if insets is not None:
            frame = insets(frame, renders[1:])
        sink(frame)
        per_frame.append(1e3 * (time.perf_counter() - start))
    return per_frame


def warm_up(app: Any, client: viser.ClientHandle, camera: CameraPose, width: int,
            height: int, tries: int = 20) -> None:
    """Render until two captures agree: textures and meshes load asynchronously on connect."""
    sync_scene(app.server, client)
    last = None
    for _ in range(tries):
        frame = capture(client, camera, width, height)
        if last is not None and np.array_equal(frame, last):
            return
        last = frame
        time.sleep(0.5)
    logger.warning(f"[VIDEO] warm-up: renders still changing after {tries} tries")


def ms_summary(ms: Sequence[float]) -> str:
    a = np.asarray(ms)
    return (f"{len(a)} frames, render mean {a.mean():.1f} / p95 {np.percentile(a, 95):.1f} / "
            f"max {a.max():.1f} ms per frame")


def probe(path: Path) -> dict[str, Any]:
    """Frame count, size and duration of a video, from ``ffprobe``."""
    out = subprocess.run(
        [str(Path(FFMPEG).with_name("ffprobe")), "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    stream = info["streams"][0]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "frames": int(stream["nb_read_frames"]),
            "duration_s": float(info["format"]["duration"]), "bytes": path.stat().st_size}


def read_frames(path: Path, width: int, height: int) -> np.ndarray:
    """All frames of a video as (N, H, W, 3) uint8."""
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
