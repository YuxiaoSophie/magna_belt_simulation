#!/usr/bin/env python3
"""Viser replay of recorded round-belt runs (``--record``): pick a run, scrub, play.

Rebuilds the scene from the directives and replays the recorded body poses: no physics, no
LCM, no magna.  Open the printed URL (default port 8081; the live viewer uses 8080).

Run:
    uv run python scripts/replay_viewer.py
    uv run python scripts/replay_viewer.py --recordings /path/to/recordings --run NAME
    uv run python scripts/replay_viewer.py --port 8082 --show-collision --point-cloud
    uv run python scripts/replay_viewer.py --render-fps 15 --stats   # slow link / client
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import newton

from round_belt_task.scene import build_scene
from task_common.cameras import RgbdCameras
from task_common.point_cloud import CroppedPointCloud
from task_common.recording import Recording
from task_common.replay_app import DEFAULT_DEVICE, DEFAULT_RENDER_FPS, ReplayApp
from task_common.scene import make_builder
from utils.viewer_patches import (
    patch_viewer_shape_names,
    patch_viser_texture_material,
)


def configure_logging(level: str = "INFO") -> None:
    """Install this script's loguru sink.  Called from ``__main__`` ONLY."""
    logger.remove()
    logger.add(sys.stdout, level=level,
               format="<level>{level: <7}</level> | <level>{message}</level>")


def build_model() -> newton.Model:
    builder = make_builder()
    build_scene(builder)
    return builder.finalize()


def build_point_clouds() -> tuple[newton.Model, list[CroppedPointCloud]]:
    builder = make_builder()
    info = build_scene(builder)
    model = builder.finalize()
    if not info.point_clouds:
        return model, []
    cameras = RgbdCameras(model, info.cameras)
    return model, [CroppedPointCloud(cameras, spec) for spec in info.point_clouds]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recordings", type=Path, default=REPO_ROOT / "recordings",
                        help="directory holding the run directories")
    parser.add_argument("--run", default=None, help="run directory name (default: newest)")
    parser.add_argument("--port", type=int, default=8081, help="viser port")
    parser.add_argument("--show-collision", action="store_true",
                        help="start with collision shapes shown (toggle under Display)")
    parser.add_argument("--point-cloud", action="store_true",
                        help="start with the camera point cloud shown (toggle under Display)")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help="Warp device of the replay model (default %(default)s; a GPU adds "
                             "per-frame device syncs and competes with a running sim)")
    parser.add_argument("--render-fps", type=float, default=DEFAULT_RENDER_FPS,
                        help="max 3D redraws per second during playback (default %(default)g)")
    parser.add_argument("--stats", action="store_true",
                        help="log tick/render/plot rates every 10 s")
    return parser


if __name__ == "__main__":
    configure_logging()
    args = create_parser().parse_args()
    if not Recording.list_runs(args.recordings):
        logger.error(f"no recorded runs under {args.recordings} "
                     "(record one with round_belt_lcm_simulation.py --record)")
        sys.exit(2)
    # Before the viewer exists: set_model populates shapes through the patched method.
    patch_viewer_shape_names()
    patch_viser_texture_material()
    app = ReplayApp(args.recordings, build_model, port=args.port,
                    show_collision=args.show_collision, run=args.run,
                    build_point_clouds=build_point_clouds, show_point_cloud=args.point_cloud,
                    render_fps=args.render_fps, device=args.device)
    app.run_forever(stats=args.stats)
