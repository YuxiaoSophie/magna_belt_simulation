#!/usr/bin/env python3
"""The learned-LCS encoder as an LCM node: robot states + camera cloud + belt state -> LATENT_STATE.

Builds exactly the training inputs (``camera_points``, material ``belt_points_ordered`` of the
48 belt bodies, ``state_vector`` with the numpy FK poses) and publishes
``latent_state_message(utime = cloud utime, ...)`` for each (cloud, belt) pair with equal utime
whose robot states match (``exact``: the states stamped with that utime, sim; ``nearest``: the
latest states with utime <= it, hardware), at most once per ``--min-period-s``.

Sim (with ``round_belt_lcm_simulation.py --cameras --publish-point-cloud --publish-belt-state``):
    uv run python scripts/lcs/latent_encoder_node.py --lcm-url 'udpm://239.255.76.97:7697?ttl=0' \\
        --ur-state-channel UR_STATE_SIM --state-match exact
Hardware mapping: ``--ur-state-channel UR_STATE --state-match nearest --belt-input points150``
(the estimator's 150 vertices, resized like the loader; the model was trained on 48 bodies).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import lcm

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from dairlib import lcmt_robot_output
from drake import lcmt_point_cloud
from magna import lcmt_round_belt_state
from round_belt_task.arm_kinematics import FrankaTip, UrTracking
from round_belt_task.episode_io import pose7_mat
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, board_pose
from task_common import lcs_dataset as lcs
from task_common.latent_encoder import (
    LATENT_STATE_CHANNEL,
    LatentEncoder,
    latent_state_message,
    resize_points_ordered,
)
from task_common.lcm_contract import _UR_JOINT_NAMES
from task_common.perception_lcm import (
    WORLD_FRAME,
    from_frame,
    point_cloud_xyz,
    round_belt_points,
)

DEFAULT_DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_ablation_20260924/"
                      "ckpt_decoded_only/deploy_demo/deploy.npz")
FRANKA_POSITION_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
UR_POSITION_NAMES = _UR_JOINT_NAMES
STATE_BUFFER_US = 2_000_000
PAIR_TIMEOUT_US = 2_000_000
BELT_INPUTS = ("bodies48", "points150")


@dataclass
class ArmState:
    utime: int
    q: np.ndarray
    v: np.ndarray


def arm_state(msg: lcmt_robot_output, names) -> ArmState:
    """Positions/velocities of ``names`` (and ``<name>dot``) picked by name, float64."""
    pos = dict(zip(msg.position_names, msg.position))
    vel = dict(zip(msg.velocity_names, msg.velocity))
    q = np.array([pos[n] for n in names], dtype=np.float64)
    v = np.array([vel[f"{n}dot"] for n in names], dtype=np.float64)
    return ArmState(int(msg.utime), q, v)


def franka_pose7(q_franka) -> np.ndarray:
    """``finger_tip`` xyz + wxyz (w >= 0): ``RoundBeltOscSimulation.franka_measured_pose7``."""
    return pose7_mat(FrankaTip.fk(q_franka))


def ur_pose7(q_ur) -> np.ndarray:
    return pose7_mat(UrTracking.fk(q_ur))


def proprio(franka: ArmState, ur: ArmState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ee_franka7, ee_ur7, state_vector40)``."""
    ee_f, ee_u = franka_pose7(franka.q), ur_pose7(ur.q)
    return ee_f, ee_u, lcs.state_vector(franka.q, ur.q, franka.v, ur.v, ee_f, ee_u)


def belt_world(msg: lcmt_round_belt_state, X_WB: np.ndarray | None) -> np.ndarray:
    """Belt points in world (float64); non-world frames go through ``X_WB`` (taskboard)."""
    pts = round_belt_points(msg)
    if msg.frame_name == WORLD_FRAME:
        return pts
    if X_WB is None:
        raise ValueError(f"belt frame {msg.frame_name!r} needs the board pose (--params)")
    return from_frame(X_WB, pts)


def belt_input(pts_world: np.ndarray, mode: str, n_points: int = lcs.BELT_POINTS
               ) -> np.ndarray:
    """``bodies48``: material ``belt_points_ordered``; ``points150``: loader resize as-is."""
    if mode == "bodies48":
        if len(pts_world) != lcs.BELT_BODIES:
            raise ValueError(f"bodies48: {len(pts_world)} points != {lcs.BELT_BODIES}")
        return lcs.belt_points_ordered(pts_world, n_points)
    if mode == "points150":
        return resize_points_ordered(pts_world, n_points)
    raise ValueError(f"unknown belt input {mode!r}")


class StateRing:
    """The last ``STATE_BUFFER_US`` of one robot's states, keyed by utime."""

    def __init__(self) -> None:
        self.states: deque[ArmState] = deque()

    def add(self, state: ArmState) -> None:
        self.states.append(state)
        while self.states and self.states[0].utime < state.utime - STATE_BUFFER_US:
            self.states.popleft()

    def newest(self) -> int | None:
        return self.states[-1].utime if self.states else None

    def match(self, utime: int, mode: str) -> ArmState | None:
        if mode == "exact":
            return next((s for s in reversed(self.states) if s.utime == utime), None)
        return next((s for s in reversed(self.states) if s.utime <= utime), None)


class LatentEncoderNode:
    def __init__(self, cli: argparse.Namespace, lc: lcm.LCM | None = None) -> None:
        self.cli = cli
        self.encoder = LatentEncoder.load(cli.deploy)
        self.X_WB = board_pose(Path(cli.params))
        self.lc = lc or lcm.LCM(cli.lcm_url)
        self.franka, self.ur = StateRing(), StateRing()
        self.clouds: dict[int, tuple[np.ndarray, float]] = {}
        self.belts: dict[int, np.ndarray] = {}
        self.last_utime: int | None = None
        self.min_period_us = round(cli.min_period_s * 1e6)
        self.published = self.misses = self.dropped = self.errors = 0
        self._reset_stats(time.perf_counter())
        self.lc.subscribe(cli.franka_state_channel, self._on_franka)
        self.lc.subscribe(cli.ur_state_channel, self._on_ur)
        self.lc.subscribe(cli.point_cloud_channel, self._on_cloud)
        self.lc.subscribe(cli.belt_state_channel, self._on_belt)

    def _reset_stats(self, now: float) -> None:
        self._t0 = now
        self._n = 0
        self._lat: list[float] = []
        self._age: list[float] = []

    def _on_franka(self, _ch: str, data: bytes) -> None:
        self.franka.add(arm_state(lcmt_robot_output.decode(data), FRANKA_POSITION_NAMES))
        self._process()

    def _on_ur(self, _ch: str, data: bytes) -> None:
        self.ur.add(arm_state(lcmt_robot_output.decode(data), UR_POSITION_NAMES))
        self._process()

    def _on_cloud(self, _ch: str, data: bytes) -> None:
        wall = time.perf_counter()
        msg = lcmt_point_cloud.decode(data)
        if msg.frame_name != WORLD_FRAME:
            self.errors += 1
            print(f"[NODE] cloud frame {msg.frame_name!r} != 'world': dropped", flush=True)
            return
        try:
            cloud = lcs.camera_points(point_cloud_xyz(msg))
        except ValueError as exc:
            self.errors += 1
            print(f"[NODE] cloud dropped: {exc}", flush=True)
            return
        self.clouds[int(msg.utime)] = (cloud, wall)
        self._process()

    def _on_belt(self, _ch: str, data: bytes) -> None:
        msg = lcmt_round_belt_state.decode(data)
        try:
            belt = belt_input(belt_world(msg, self.X_WB), self.cli.belt_input,
                              self.encoder.belt_num_points)
        except ValueError as exc:
            self.errors += 1
            print(f"[NODE] belt state dropped: {exc}", flush=True)
            return
        self.belts[int(msg.utime)] = belt
        self._process()

    def _process(self) -> None:
        for utime in sorted(set(self.clouds) & set(self.belts)):
            if self.last_utime is not None and (
                    utime <= self.last_utime or utime - self.last_utime < self.min_period_us):
                self.dropped += 1
                self._discard(utime)
                continue
            franka = self.franka.match(utime, self.cli.state_match)
            ur = self.ur.match(utime, self.cli.state_match)
            if franka is None or ur is None:
                # Exact states never arrive once both robots have moved past this utime.
                past = all(r.newest() is not None and r.newest() > utime
                           for r in (self.franka, self.ur))
                if self.cli.state_match == "exact" and past:
                    self.misses += 1
                    self._discard(utime)
                continue
            self._publish(utime, franka, ur)
        newest = max(list(self.clouds) + list(self.belts), default=None)
        if newest is not None:
            for utime in [u for u in (*self.clouds, *self.belts) if u < newest - PAIR_TIMEOUT_US]:
                self.misses += utime in self.clouds and utime in self.belts
                self._discard(utime)

    def _discard(self, utime: int) -> None:
        self.clouds.pop(utime, None)
        self.belts.pop(utime, None)

    def _publish(self, utime: int, franka: ArmState, ur: ArmState) -> None:
        cloud, wall = self.clouds[utime]
        belt = self.belts[utime]
        ee_f, ee_u, prop = proprio(franka, ur)
        z = self.encoder.encode(cloud, prop, belt)
        msg = latent_state_message(utime, utime * 1e-6, z, ee_f, ee_u, prop)
        self.lc.publish(self.cli.latent_channel, msg.encode())
        self._discard(utime)
        # Older pairs can no longer be published (monotonic utimes).
        for u in [u for u in (*self.clouds, *self.belts) if u < utime]:
            self._discard(u)
        self.last_utime = utime
        self.published += 1
        self._n += 1
        self._lat.append(time.perf_counter() - wall)
        self._age.append((self.franka.newest() - utime) * 1e-6)

    def log_stats(self, now: float) -> None:
        wall = now - self._t0
        lat = np.asarray(self._lat) * 1e3
        age = np.asarray(self._age) * 1e3
        lat_s = f"mean {lat.mean():.1f} max {lat.max():.1f} ms" if len(lat) else "n/a"
        age_s = f"mean {age.mean():.1f} max {age.max():.1f} ms" if len(age) else "n/a"
        print(f"[NODE] {self._n / wall:.2f} latents/s ({self.published} total), cloud->latent "
              f"{lat_s}, cloud age {age_s}, state-match misses {self.misses}, rate-limited "
              f"{self.dropped}, errors {self.errors}, pending {len(self.clouds)}/"
              f"{len(self.belts)}", flush=True)
        self._reset_stats(now)

    def spin(self, stop) -> None:
        next_stats = time.perf_counter() + self.cli.stats_every
        while not stop():
            try:
                self.lc.handle_timeout(100)
            except OSError:
                # A signal interrupts the select (EINTR); anything else is real.
                if not stop():
                    raise
            now = time.perf_counter()
            if now >= next_stats:
                self.log_stats(now)
                next_stats = now + self.cli.stats_every


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lcm-url", required=True, help="LCM URL (use a private group)")
    p.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY, help="deploy.npz")
    p.add_argument("--franka-state-channel", default="FRANKA_STATE")
    p.add_argument("--ur-state-channel", default="UR_STATE", help="sim: UR_STATE_SIM")
    p.add_argument("--point-cloud-channel", default="POINT_CLOUD_CROPPED")
    p.add_argument("--belt-state-channel", default="RoundBeltState")
    p.add_argument("--latent-channel", default=LATENT_STATE_CHANNEL)
    p.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML,
                   help="params yaml with task_board_position/orientation (taskboard frame)")
    p.add_argument("--belt-input", choices=BELT_INPUTS, default="bodies48")
    p.add_argument("--state-match", choices=("exact", "nearest"), default="exact")
    p.add_argument("--min-period-s", type=float, default=0.075)
    p.add_argument("--stats-every", type=float, default=5.0, help="stats period [s]")
    return p


def main() -> int:
    cli = create_parser().parse_args()
    node = LatentEncoderNode(cli)
    stopping = []
    signal.signal(signal.SIGINT, lambda *_: stopping.append(True))
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    print(f"[NODE] {cli.lcm_url}: {cli.franka_state_channel} + {cli.ur_state_channel} + "
          f"{cli.point_cloud_channel} + {cli.belt_state_channel} -> {cli.latent_channel} "
          f"(z {node.encoder.latent_dim}, belt {cli.belt_input}, states {cli.state_match}, "
          f"min period {cli.min_period_s:g} s, deploy {cli.deploy})", flush=True)
    node.spin(lambda: bool(stopping))
    node.log_stats(time.perf_counter())
    print("[NODE] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
