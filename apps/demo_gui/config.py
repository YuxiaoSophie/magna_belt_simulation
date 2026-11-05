"""Ports, paths and session layout of the demo GUI (one place)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HTTP_PORT, VIEWER_PORT, REPLAY_PORT, SIM_PORT = 8090, 8091, 8092, 7850
# 7840/7841 SpaceMouse teleop, 7855 v12_147, 7857 a running job.
REFUSED_SIM_PORTS = (7667, 7840, 7841, 7855, 7857)
MIN_SIM_PORT = 7780
CONTROL_HZ = 200  # 1 / CONTROL_DT in src/task_common/lcm_simulation.py (not imported: pulls in warp)

SCENE = REPO / "assets/round_belt_task/round_belt_scene_wr_pin640b1.yaml"
LOCK = REPO / "data/lcs/realfirst/cmp/v12_79/sim.lock"
GPU_FREE = REPO / "data/lcs/realfirst/cmp/v12_152/gpu_free.sh"
SESSION_ROOT = REPO / "data/demo_gui/sessions"
# the harness's own --magna-root default (magna-deploy-learned-lcs) was merged into magna main
MAGNA_ROOT = Path("/home/hienbui/git/magna")

MOCK_START = REPO / "data/lcs/start_states/grasp_variants/set2/gv_00_osc.npz"
MOCK_RUN = REPO / "data/lcs/realfirst/cmp/v12_152/e2e_V63j/V63j_tol50b_nominal"
MOCK_MILES = REPO / "data/lcs/realfirst/cmp/v12_147/miles1_d3"
MOCK_TRAIN = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v12_20261006/v12_152/"
                  "J152_fpz0r05")


def lcm_url(port: int) -> str:
    """Private LCM URL: group 239.255.76.<port-7600> up to 7855, 239.255.78.<port-7800> to 7859."""
    port = int(port)
    if port == 7667 or not MIN_SIM_PORT <= port <= 7859:
        raise ValueError(f"LCM port {port} refused: need {MIN_SIM_PORT}..7859, never 7667")
    if port <= 7855:
        return f"udpm://239.255.76.{port - 7600}:{port}?ttl=0"
    return f"udpm://239.255.78.{port - 7800}:{port}?ttl=0"


@dataclass
class Cfg:
    backend: str = "mock"
    host: str = "127.0.0.1"
    http_port: int = HTTP_PORT
    viewer_port: int = VIEWER_PORT
    replay_port: int = REPLAY_PORT
    sim_port: int = SIM_PORT
    scene: Path = SCENE
    magna_root: Path = MAGNA_ROOT
    viewer_hz: float = 60.0

    @property
    def render_every(self) -> int:
        """Harness ``--render-every``: control steps per published body-pose frame."""
        return max(1, round(CONTROL_HZ / self.viewer_hz))

    @property
    def lcm_url(self) -> str:
        return lcm_url(self.sim_port)

    @property
    def viewer_url(self) -> str:
        return f"http://localhost:{self.viewer_port}"

    @property
    def viewer_cmd(self) -> str:
        """Launch line of the standalone viewer the dock shows (the GUI never starts it)."""
        return (f"ROUND_BELT_SCENE={os.path.relpath(self.scene, REPO)} uv run --frozen python "
                f"scripts/lcm_viewer.py --lcm-url '{self.lcm_url}' --port {self.viewer_port}")
