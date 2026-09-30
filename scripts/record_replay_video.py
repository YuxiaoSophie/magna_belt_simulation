#!/usr/bin/env python3
"""Render a recorded run (or two, side by side) to MP4 on the server, frame-exactly.

Builds the replay viewer's scene and layers headless on a scratch viser port; a headless
Chromium is the render client. Frame ``n`` shows recording time ``start + n * speed / fps``
(times are relative to the run's first recorded frame).

Run:
    uv run python scripts/record_replay_video.py --recordings DIR --run NAME --out run.mp4
    uv run python scripts/record_replay_video.py --recordings DIR --run A --compare B \
        --out a_vs_b.mp4 --learned-layers planned_belt,actions --decoder decoder.npz
    uv run python scripts/record_replay_video.py ... --camera '{"position":[...],...}'  # Display
    uv run python scripts/record_replay_video.py ... --inset 'side={...}' --inset front=cam.json
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from replay_viewer import (
    build_model,
    build_point_clouds,
    configure_logging,
)

from round_belt_task import BELT_COLOR
from task_common.recording import Recording
from task_common.replay_app import ReplayApp
from task_common.replay_learned_mpc import (
    ACTION_SCALE_DEFAULT,
    LAYERS,
    any_learned_runs,
    parse_layers,
)
from task_common.replay_video import (
    CRF,
    DEFAULT_CAMERA,
    FFMPEG,
    VIDEO_TUBE_SIDES,
    VIDEO_TUBE_STRIDE,
    CameraPose,
    Encoder,
    HeadlessBrowser,
    Inset,
    InsetLayout,
    Overlay,
    RenderFailed,
    VideoLearnedMpcPanel,
    belt_rgb,
    frame_times,
    has_drawtext,
    legend_entries,
    ms_summary,
    probe,
    render_legend,
    render_run,
    run_duration,
    text_filter,
    time_filter,
    warm_up,
)
from utils.viewer_patches import (
    patch_viewer_shape_names,
    patch_viser_texture_material,
)

DEFAULT_PORT = 19281
LEGEND_CORNER = "br"  # the floor; t = ... is bottom-left


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recordings", type=Path, default=REPO_ROOT / "recordings",
                        help="directory holding the run directories")
    parser.add_argument("--run", default=None, help="run directory name (default: newest)")
    parser.add_argument("--compare", default=None,
                        help="second run: rendered from the same camera, stacked to the right")
    parser.add_argument("--out", type=Path, required=True, help="output .mp4")
    parser.add_argument("--fps", type=float, default=25.0, help="video frame rate")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="recording seconds per video second (1 = real time)")
    parser.add_argument("--start-s", type=float, default=0.0,
                        help="start, s after the run's first frame")
    parser.add_argument("--end-s", type=float, default=None, help="end, s (default: run end)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", default=None,
                        help="camera: a .json file or the json itself {position, look_at, up, "
                             "fov (rad)}, as shown under the viewer's Display")
    parser.add_argument("--inset", action="append", default=[], metavar="LABEL=CAMERA",
                        help="extra small view, top-right (repeatable): LABEL=<camera json "
                             "or .json path>")
    parser.add_argument("--no-overlay", action="store_true", help="no time/label text")
    parser.add_argument("--no-legend", action="store_true",
                        help="no belt/layer colour legend (drawn by default on learned-MPC runs)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="scratch viser port")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--point-cloud", action="store_true")
    learned = parser.add_argument_group("learned MPC layers", "as in replay_viewer.py")
    learned.add_argument("--deploy", type=Path, default=None)
    learned.add_argument("--decoder", type=Path, default=None)
    learned.add_argument("--demo-goals", type=Path, default=None,
                         help="replaces the run's recorded demo_goals.npz (as in replay_viewer.py)")
    learned.add_argument("--demo-episode", type=Path, default=None)
    learned.add_argument("--learned-layers", default="",
                         help=f"comma list from {','.join(LAYERS)}")
    learned.add_argument("--action-scale", type=float, default=ACTION_SCALE_DEFAULT)
    learned.add_argument("--tube-stride", type=int, default=VIDEO_TUBE_STRIDE,
                         help="draw every n-th belt point (viewer: 3)")
    learned.add_argument("--tube-sides", type=int, default=VIDEO_TUBE_SIDES,
                         help="tube cross-section sides (viewer: 6)")
    return parser


def quiet_websockets() -> None:
    """Chromium's pre-connect probes log handshake tracebacks; they are harmless."""
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)


def video_hooks(args: argparse.Namespace) -> list[VideoLearnedMpcPanel]:
    paths = {"deploy": args.deploy, "decoder": args.decoder, "demo_goals": args.demo_goals,
             "demo_episode": args.demo_episode}
    if not any(paths.values()) and not any_learned_runs(args.recordings):
        return []
    return [VideoLearnedMpcPanel(**paths, layers=parse_layers(args.learned_layers),
                                 tube_stride=args.tube_stride, tube_sides=args.tube_sides)]


def main(args: argparse.Namespace) -> int:
    if args.fps <= 0 or args.speed <= 0 or args.width <= 0 or args.height <= 0:
        logger.error("--fps, --speed, --width and --height must be > 0")
        return 2
    if args.width % 2 or args.height % 2:
        logger.error("--width and --height must be even (yuv420p)")
        return 2
    names = [p.name for p in Recording.list_runs(args.recordings)]
    run = args.run or (names[0] if names else None)
    runs = [run] + ([args.compare] if args.compare else [])
    missing = [r for r in runs if r not in names]
    if missing:
        logger.error(f"no run(s) {missing} under {args.recordings}")
        return 2
    camera = CameraPose.load(args.camera) if args.camera else DEFAULT_CAMERA
    try:
        insets = [Inset.parse(text) for text in args.inset]
    except (ValueError, OSError) as exc:
        logger.error(f"--inset: {exc}")
        return 2
    overlay = not args.no_overlay and has_drawtext()
    if not args.no_overlay and not overlay:
        logger.warning("ffmpeg has no drawtext filter: rendering without text")
    hooks = video_hooks(args)
    quiet_websockets()
    patch_viewer_shape_names()
    patch_viser_texture_material()
    app = ReplayApp(args.recordings, build_model, port=args.port, run=run, hooks=hooks,
                    verbose=False, analysis=False, show_collision=args.show_collision,
                    build_point_clouds=build_point_clouds, show_point_cloud=args.point_cloud)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        durations = {}
        for name in runs:
            app.select_run(name)
            durations[name] = run_duration(app)
        times = frame_times(max(durations.values()), args.fps, args.speed, args.start_s,
                            args.end_s)
        step_s = args.speed / args.fps
        logger.info(f"[VIDEO] {len(times)} frames ({times[0]:.2f}-{times[-1]:.2f} s, "
                    f"{step_s * 1e3:.0f} ms of recording per frame) of {runs}; camera "
                    f"{camera.to_dict()}")
        browser = HeadlessBrowser(app.server, f"http://127.0.0.1:{args.port}", args.width,
                                  args.height)
        with browser, tempfile.TemporaryDirectory(prefix=".video-", dir=args.out.parent) as tmp:
            parts = []
            for i, name in enumerate(runs):
                app.select_run(name)
                for hook in hooks:
                    hook.set_action_scale(app, args.action_scale)
                legend = None if args.no_legend else legend_overlay(hooks, args)
                layout = (InsetLayout(insets, args.width, args.height,
                                      bottom=legend.y if legend else None) if insets else None)
                if i == 0:
                    warm_up(app, browser.client, camera, args.width, args.height)
                    for cam in layout.cameras if layout else []:
                        warm_up(app, browser.client, cam, *layout.render_size)
                    if layout:
                        logger.info(f"[VIDEO] insets {[x.label for x in insets]}: "
                                    f"{layout.w}x{layout.h} (rendered at "
                                    f"{layout.render_size}) at {layout.rects()}")
                single = len(runs) == 1
                vf = time_filter(args.start_s, step_s) if overlay and single else None
                path = Path(tmp) / (args.out.name if single else f"part{i}.mp4")
                encoder = Encoder(path, args.width, args.height, args.fps, vf=vf,
                                  lossless=not single)
                sink = (encoder.write if legend is None
                        else lambda f, e=encoder, o=legend: e.write(o(f)))
                try:
                    ms = render_run(app, browser, camera, times, args.width, args.height, sink,
                                    layout)
                finally:
                    encoder.close()
                logger.info(f"[VIDEO] {name}: {ms_summary(ms)} "
                            f"(run is {durations[name]:.2f} s; later frames hold the last)")
                parts.append(path)
            if len(parts) == 2:
                stack(parts, runs, args, step_s, overlay)
            else:
                os.replace(parts[0], args.out)  # only a complete video replaces the old one
    except RenderFailed as exc:
        logger.error(f"[VIDEO] {exc}; {args.out} not written")
        return 1
    finally:
        app.close()
    info = probe(args.out)
    if browser.restarts:
        logger.warning(f"[VIDEO] recovered from bad renders: {browser.restarts} chromium restarts")
    logger.info(f"[VIDEO] wrote {args.out}: {info['frames']} frames {info['width']}x"
                f"{info['height']}, {info['duration_s']:.2f} s, {info['bytes'] / 1e6:.1f} MB "
                f"in {time.perf_counter() - start:.0f} s")
    return 0


def legend_overlay(hooks: list[VideoLearnedMpcPanel],
                   args: argparse.Namespace) -> Overlay | None:
    """The colour legend for the loaded run, or None if it has no learned solves."""
    entries = legend_entries(hooks[0], belt_rgb(BELT_COLOR)) if hooks else []
    if not entries:
        return None
    logger.info(f"[VIDEO] legend: {[e.label for e in entries]}")
    return Overlay(render_legend(entries, args.height), args.width, args.height, LEGEND_CORNER)


def stack(parts: list[Path], runs: list[str], args: argparse.Namespace, step_s: float,
          overlay: bool) -> None:
    chain = "[0:v][1:v]hstack=inputs=2"
    if overlay:
        chain += "," + ",".join([text_filter(runs[0], "12"), text_filter(runs[1], "w/2+12"),
                                 time_filter(args.start_s, step_s)])
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(parts[0]),
           "-i", str(parts[1]), "-filter_complex", chain + "[v]", "-map", "[v]",
           "-c:v", "libx264", "-crf", str(CRF), "-preset", "medium", "-pix_fmt", "yuv420p",
           "-r", f"{args.fps:g}", "-movflags", "+faststart", str(args.out)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    configure_logging()
    sys.exit(main(create_parser().parse_args()))
