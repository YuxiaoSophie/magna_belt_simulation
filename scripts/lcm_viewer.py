#!/usr/bin/env python3
"""Standalone CPU viser viewer of a sim's NEWTON_VIEWER_DRAW body poses on a private LCM URL.

Builds the scene once (ROUND_BELT_SCENE, same as the sim's), renders the newest body-pose message
at <= --hz and POINT_CLOUD_CROPPED when the "point cloud" checkbox is on; shows the default pose
before the first message and keeps the last frame between runs. Never uses the GPU.

Run (the sim needs --publish-body-poses):
    ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_wr_pin640b1.yaml \
        uv run --frozen python scripts/lcm_viewer.py --lcm-url 'udpm://239.255.76.250:7850?ttl=0' \
        --port 8091
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before any warp import: the viewer must stay off the GPU

import argparse
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lcm
import newton
import numpy as np
import warp as wp
from loguru import logger
from replay_viewer import build_model, configure_logging

from task_common.lcm_contract import (
    VIEWER_DRAW_CHANNEL,
    LcmChannels,
    lcmt_viewer_draw,
    viewer_draw_body_q,
)
from task_common.magna_process import check_private_url
from task_common.perception_lcm import lcmt_point_cloud, point_cloud_xyz
from task_common.replay_video import CameraPose
from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material

CLOUD_CHANNEL = LcmChannels().point_cloud_channel


# Default workspace view for new clients.
CAMERA = CameraPose(position=(1.0066, 0.5276, 0.436), look_at=(0.1734, -0.0393, -0.1182),
                    up=(0.0, 0.0, 1.0), fov=1.309)


class BodyPoseMap:
    """Message link names -> rows of the model's body_q (names rule; unknown names ignored)."""

    def __init__(self, labels) -> None:
        self.index = {str(label): i for i, label in enumerate(labels)}

    def apply(self, names, body_q_msg: np.ndarray, body_q: np.ndarray) -> int:
        pairs = [(self.index[n], i) for i, n in enumerate(names) if n in self.index]
        if pairs:
            dst, src = (list(c) for c in zip(*pairs, strict=True))
            body_q[dst] = body_q_msg[src]
        return len(pairs)


class LcmViewer:
    def __init__(self, lcm_url: str, port: int, hz: float, point_cloud: bool,
                 show_collision: bool) -> None:
        self.period = 1.0 / hz
        self.draws = self.clouds = self.dropped = 0
        self._draw = self._xyz = None  # newest decoded payloads
        self._dirty, self._cloud_dirty = False, True
        self._matched_logged = False
        self._timestamp = 0
        self._last_draw = self._last_render = self._last_text = 0.0
        self.lc = lcm.LCM(check_private_url(lcm_url))
        # LCM's default queue (30) drops the NEWEST message when full: a stall shows a stale frame
        self.lc.subscribe(VIEWER_DRAW_CHANNEL, self.on_draw).set_queue_capacity(1000)
        self.lc.subscribe(CLOUD_CHANNEL, self.on_cloud).set_queue_capacity(1000)
        patch_viewer_shape_names()
        patch_viser_texture_material()
        start = time.perf_counter()
        with wp.ScopedDevice("cpu"):
            self.viewer = newton.viewer.ViewerViser(port=port, label="Newton LCM viewer",
                                                    verbose=True)
            # viser silently moves to the next free port; the GUI dock expects exactly this one
            if self.viewer._server.get_port() != port:
                self.viewer.close()
                raise OSError(f"viewer port {port} is busy")
            if hasattr(self.viewer, "_pause"):
                self.viewer._pause = False
            self.model = build_model()
            self.viewer.set_model(self.model)
            if show_collision:
                self.viewer.show_collision = True  # after set_model, which resets it
            self._frame_workspace()
            self.state = self.model.state()
        self.build_s = time.perf_counter() - start
        logger.info(f"[VIEWER] {self.model.body_count} bodies on cpu in {self.build_s:.1f} s")
        self.map = BodyPoseMap(self.model.body_label)
        self.body_q = np.ascontiguousarray(self.state.body_q.numpy(), dtype=np.float32)
        gui = self.viewer._server.gui
        self.cloud_box = gui.add_checkbox("point cloud", initial_value=point_cloud)
        self.cloud_box.on_update(self._on_toggle)
        self.status = gui.add_text("status", initial_value="msgs: 0 / cloud: 0 / last: -",
                                   disabled=True)
        self.render()  # default pose: the page is never empty

    def _frame_workspace(self) -> None:
        """New clients start on ``CAMERA``, orbiting its look-at point."""
        cam = self.viewer._server.initial_camera  # Newton's set_camera would orbit 1 m out
        cam.position, cam.look_at, cam.up, cam.fov = (CAMERA.position, CAMERA.look_at, CAMERA.up,
                                                      CAMERA.fov)

    def _on_toggle(self, _event) -> None:
        self._dirty = self._cloud_dirty = True

    def on_draw(self, _channel, data) -> None:
        try:
            msg = lcmt_viewer_draw.decode(data)
            self._draw = (msg.timestamp, *viewer_draw_body_q(msg))
        except (ValueError, struct.error):  # a bad packet must never kill the viewer
            self.dropped += 1
            return
        self.draws += 1
        self._last_draw = time.monotonic()
        self._dirty = True

    def on_cloud(self, _channel, data) -> None:
        try:
            self._xyz = point_cloud_xyz(lcmt_point_cloud.decode(data))
        except (ValueError, struct.error):
            self.dropped += 1
            return
        self.clouds += 1
        self._dirty = self._cloud_dirty = True

    def pump(self) -> bool:
        """One loop iteration: drain LCM, render if dirty and the --hz period has passed."""
        self.lc.handle_timeout(5)
        while self.lc.handle_timeout(0) > 0:
            pass
        now = time.monotonic()
        if now - self._last_text >= 0.5:
            age = f"{now - self._last_draw:.1f} s" if self.draws else "-"
            self.status.value = f"msgs: {self.draws} / cloud: {self.clouds} / last: {age}"
            self._last_text = now
        if not self._dirty or now - self._last_render < self.period:
            return False
        self.render()
        return True

    def render(self) -> None:
        if self._draw is not None:
            self._timestamp, names, q = self._draw
            self._draw = None
            k = self.map.apply(names, q, self.body_q)
            if not self._matched_logged:
                self._matched_logged = True
                logger.info(f"[VIEWER] matched {k}/{len(names)} links")
                if k == 0:
                    # shortcut: the viewer trusts link names; a scene mismatch renders nothing and
                    # warns once, upgrade to a per-name mismatch report if that ever bites.
                    logger.warning("[VIEWER] no link matches the model: set ROUND_BELT_SCENE to "
                                   "the sim's scene")
        self.viewer.begin_frame(self._timestamp * 1e-6)
        self.state.body_q.assign(self.body_q)
        self.viewer.log_state(self.state)
        if self._cloud_dirty:  # re-sending an unchanged cloud every frame is wasted bandwidth
            if self.cloud_box.value and self._xyz is not None:
                self.viewer.log_points("point_cloud", wp.array(self._xyz, dtype=wp.vec3,
                                                               device="cpu"),
                                       radii=0.002, colors=(0.2, 0.5, 1.0))
            else:
                self.viewer.log_points("point_cloud", None, hidden=True)
            self._cloud_dirty = False
        self.viewer.end_frame()
        self._dirty = False
        self._last_render = time.monotonic()

    def run(self) -> None:
        try:
            while self.viewer.is_running():
                self.pump()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self.viewer.close()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lcm-url", required=True, help="private LCM URL (never 7667)")
    parser.add_argument("--port", type=int, default=8091, help="viser port (default %(default)s)")
    parser.add_argument("--hz", type=float, default=60.0,
                        help="max renders/s (default %(default)g)")
    parser.add_argument("--point-cloud", action="store_true",
                        help="start with the point-cloud checkbox on")
    parser.add_argument("--show-collision", action="store_true",
                        help="start with collision shapes shown")
    return parser


def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.hz <= 0:
        parser.error("--hz must be > 0")
    try:
        app = LcmViewer(args.lcm_url, args.port, args.hz, args.point_cloud, args.show_collision)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    configure_logging()
    sys.exit(main())
