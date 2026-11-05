"""Live ``telemetry.json`` (``telemetry`` schema, ``source: "sim"``) from the sim's LCM traffic.

EE poses = FK of FRANKA_STATE / UR_STATE_SIM, Franka width from PANDA_HAND_STATUS, squeeze from
FRANKA_HAND_ROBOT_OUTPUT efforts, UR width from the ROBOTIQ_STATUS byte, cloud from
POINT_CLOUD_CROPPED (only if the sim publishes it; the eval harness does not). ``belt_xyz``,
phase, wraps and held stay null: belt bodies are not on LCM. FK imports Warp
(``round_belt_task.constants``), so this runs as its own CPU-only process, never inside Dash:

    CUDA_VISIBLE_DEVICES= uv run --frozen python -m apps.demo_gui.telemetry_lcm \
        --url <private url> --out <s>/telemetry.json [--run-dir <d>] [--until-pid <pid>]
"""

from __future__ import annotations

import argparse
import functools
import math
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from apps.demo_gui import jobs, telemetry
from apps.demo_gui.config import REPO
from dairlib import lcmt_robot_output
from drake import lcmt_point_cloud, lcmt_schunk_wsg_status
from robotiq import lcmt_robotiq_status
from task_common import perception_lcm

CHANNELS = {"franka": "FRANKA_STATE", "ur": "UR_STATE_SIM", "hand_out": "FRANKA_HAND_ROBOT_OUTPUT",
            "hand": "PANDA_HAND_STATUS", "robotiq": "ROBOTIQ_STATUS",
            "cloud": "POINT_CLOUD_CROPPED"}
FRANKA_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
             "wrist_2_joint", "wrist_3_joint")
LCM_SIM_YAML = REPO / "assets/round_belt_task/round_belt_lcm_sim.yaml"
WRITE_PERIOD_S = 0.1


@functools.cache
def _fk():
    from round_belt_task.arm_kinematics import FrankaTip, UrTracking
    from round_belt_task.episode_io import pose7_mat

    return FrankaTip, UrTracking, pose7_mat


@functools.cache
def ur_open_gap_m() -> float:
    """Free-air pad gap at byte 0; the sim reports byte = 255 * (1 - gap / open gap)."""
    return float(yaml.safe_load(LCM_SIM_YAML.read_text())["gripper_drive"]
                 ["width_calibration"][0][1])


def _joints(msg, names) -> np.ndarray:
    by = dict(zip(msg.position_names, msg.position))
    return np.array([by[n] for n in names], dtype=np.float64)


def decode(key: str, data: bytes) -> dict:
    """One ``CHANNELS[key]`` message -> the telemetry fields it carries (raises if malformed)."""
    if key in ("franka", "ur"):
        franka, ur, pose7 = _fk()
        m = lcmt_robot_output.decode(data)
        chain, names = (franka, FRANKA_JOINTS) if key == "franka" else (ur, UR_JOINTS)
        out = {f"{key}_pose7": telemetry._r(pose7(chain.fk(_joints(m, names))))}
        return {**out, "utime": int(m.utime)} if key == "franka" else out
    if key == "hand_out":
        m = lcmt_robot_output.decode(data)
        return {"franka_grip_n": round(float(np.abs(m.effort).sum()), 2)}
    if key == "hand":
        return {"franka_width_mm":
                round(float(lcmt_schunk_wsg_status.decode(data).actual_position_mm), 2)}
    if key == "robotiq":
        byte = lcmt_robotiq_status.decode(data).position
        return {"ur_width_mm": round(ur_open_gap_m() * 1e3 * (1.0 - byte / 255.0), 2)}
    if key == "cloud":
        xyz = perception_lcm.point_cloud_xyz(lcmt_point_cloud.decode(data))
        return {"cloud": telemetry._r(xyz[::max(1, math.ceil(len(xyz) / telemetry.MAX_CLOUD))])}
    raise KeyError(key)


class LcmTelemetry(threading.Thread):
    """Subscribes ``channels`` on ``url`` and writes ``path`` at <= 10 Hz until ``stop()``."""

    def __init__(self, url: str, path: str | Path, channels: dict = CHANNELS,
                 run_dir: str | None = None) -> None:
        super().__init__(daemon=True)
        self.url, self.path, self.channels, self.run_dir = url, Path(path), channels, run_dir
        self.state = {"t": None, "step": 0, "phase": None, "franka_pose7": None,
                      "ur_pose7": None, "franka_width_mm": None, "franka_grip_n": None,
                      "ur_width_mm": None, "belt_xyz": None, "cloud": None,
                      "wrap_large_deg": None, "wrap_small_deg": None, "held": None}
        self.received = self.dropped = self.written = 0
        self._t0: int | None = None
        self._dirty = False
        self._halt = threading.Event()

    def handle(self, key: str, channel: str, data: bytes) -> None:
        """LCM handler: never raises; a message that does not decode is dropped and counted."""
        try:
            upd = decode(key, data)
        except Exception:  # noqa: BLE001
            self.dropped += 1
            return
        self.received += 1
        utime = upd.pop("utime", None)
        if utime is not None:
            self._t0 = utime if self._t0 is None else self._t0
            upd.update(t=round((utime - self._t0) * 1e-6, 4), step=self.state["step"] + 1)
        self.state.update(upd)
        self._dirty = True

    def frame(self) -> dict:
        return {**self.state, "t": self.state["t"] or 0.0, "source": "sim",
                "run_dir": self.run_dir}

    def run(self) -> None:
        import lcm

        lc = lcm.LCM(self.url)
        for key, ch in self.channels.items():
            lc.subscribe(ch, functools.partial(self.handle, key))
        last = 0.0
        while not self._halt.is_set():
            lc.handle_timeout(50)
            now = time.monotonic()
            if self._dirty and now - last >= WRITE_PERIOD_S:
                telemetry.write(self.path, self.frame())
                self._dirty, last = False, now
                self.written += 1

    def stop(self) -> None:
        self._halt.set()


def main() -> None:
    from task_common.magna_process import check_private_url

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True, help="the sim's private LCM URL")
    ap.add_argument("--out", required=True, help="telemetry.json path")
    ap.add_argument("--run-dir", default=None, help="the run's out dir (telemetry run_dir)")
    ap.add_argument("--until-pid", type=int, default=None, help="exit when this PID is gone")
    a = ap.parse_args()
    t = LcmTelemetry(check_private_url(a.url), a.out, run_dir=a.run_dir)
    t.start()
    try:
        while t.is_alive():
            t.join(0.5)
            if a.until_pid and not jobs._alive(a.until_pid):
                break
    except KeyboardInterrupt:
        pass
    t.stop()
    t.join(2.0)
    print(f"[telemetry] {t.received} msgs, {t.dropped} dropped, {t.written} writes", flush=True)


if __name__ == "__main__":
    main()
