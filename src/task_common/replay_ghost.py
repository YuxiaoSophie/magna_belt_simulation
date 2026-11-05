"""Demo ghost replay layer: the DTW-aligned demo belt and EE points over a recorded run.

Reads ``<ghost_dir>/<run dir name>.npz`` from ``scripts/lcs/demo_ghost.py`` (episode utimes,
``shown``, demo ``belt``/``ee_*`` per episode frame, the run's own ``ee_*_run``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from task_common.replay_learned_mpc import _tube_faces, tube_vertices

GHOST = (192, 38, 211)
GHOST_ALPHA = 0.45
RUN_EE = (31, 41, 55)
GHOST_TUBE_M = 0.0035
EE_POINT_M = 0.008
MAX_GAP_US = 100_000  # no episode frame this close: hide (exit, post, before the first frame)
ROOT = "/replay/demo_ghost"


class DemoGhost:
    def __init__(self, ghost_dir: Path, tube_sides: int = 8) -> None:
        self.ghost_dir, self.sides = Path(ghost_dir), tube_sides
        self.data: dict[str, np.ndarray] | None = None
        self.offset_us = 0
        self.handles: dict[str, Any] = {}

    def build_gui(self, app: Any) -> None:
        pass

    def on_run_loaded(self, app: Any, rec: Any) -> None:
        path = self.ghost_dir / f"{Path(rec.path).resolve().name}.npz"
        self.data = dict(np.load(path)) if path.is_file() else None
        self.offset_us = int(rec.meta.get("osc_utime_offset_us") or 0)
        if self.data is None:
            logger.warning(f"[GHOST] no {path}")
        self._hide()

    def on_frame(self, app: Any, frame_index: int, step: int, sim_time: float) -> None:
        d = self.data
        if d is None:
            return
        utime = sim_time * 1e6 + self.offset_us
        i = int(np.clip(np.searchsorted(d["utime"], utime), 1, len(d["utime"]) - 1))
        i -= abs(d["utime"][i - 1] - utime) < abs(d["utime"][i] - utime)
        if abs(d["utime"][i] - utime) > MAX_GAP_US or not d["shown"][i]:
            self._hide()
            return
        belt = d["belt"][i].astype(np.float64)
        self._mesh(app, "belt", tube_vertices(belt, GHOST_TUBE_M, self.sides),
                   _tube_faces(len(belt), self.sides))
        self._points(app, "ee_demo", np.stack([d["ee_franka"][i, :3], d["ee_ur"][i, :3]]), GHOST)
        self._points(app, "ee_run", np.stack([d["ee_franka_run"][i, :3], d["ee_ur_run"][i, :3]]),
                     RUN_EE)

    def _hide(self) -> None:
        for handle in self.handles.values():
            handle.visible = False

    def _mesh(self, app: Any, name: str, verts: np.ndarray, faces: np.ndarray) -> None:
        verts = np.ascontiguousarray(verts, dtype=np.float32)
        handle = self.handles.get(name)
        if handle is None:
            handle = self.handles[name] = app.server.scene.add_mesh_simple(
                f"{ROOT}/{name}", verts, faces, color=GHOST, opacity=GHOST_ALPHA,
                cast_shadow=False, receive_shadow=False)
        handle.vertices = verts
        handle.visible = True

    def _points(self, app: Any, name: str, pts: np.ndarray, color) -> None:
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        handle = self.handles.get(name)
        if handle is None:
            handle = self.handles[name] = app.server.scene.add_point_cloud(
                f"{ROOT}/{name}", pts, color, point_size=EE_POINT_M, point_shape="circle",
                precision="float32")
        handle.points = pts
        handle.visible = True
