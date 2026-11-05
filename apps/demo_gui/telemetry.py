"""``telemetry.json``: one JSON object the GUI polls; written by the mock streamer or the sim.

Schema: ``{t, step, phase, franka_pose7, ur_pose7, franka_width_mm, franka_grip_n, ur_width_mm,
belt_xyz [[x,y,z]*48], cloud [[x,y,z]*<=544], wrap_large_deg, wrap_small_deg, held [franka, ur],
source: "mock"|"sim", run_dir}``; ``t`` s since the episode start, ``step`` the frame index, poses
xyz + quaternion (m), any value missing from the source is ``null``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import yaml

from apps.demo_gui.config import REPO
from task_common.lcs_dataset import SAMPLE_PERIOD_S

MAX_CLOUD = 544


def write(path: str | Path, d: dict) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


def read(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _r(a) -> list | float | None:
    return None if a is None else np.round(np.asarray(a, dtype=np.float64), 4).tolist()


def _gripper(rec: Path, steps: np.ndarray) -> dict[str, np.ndarray | None]:
    """Per-frame hand widths (mm) and Franka squeeze (N) from the recording chunks."""
    try:
        meta = json.loads((rec / "meta.json").read_text())
        chunks = [np.load(rec / c["file"]) for c in meta["chunks"]]
        step = np.concatenate([c["step"] for c in chunks])
        q = np.concatenate([c["joint_q"] for c in chunks])
        eff = np.concatenate([c["efforts"] for c in chunks])
        cal = np.asarray(yaml.safe_load((REPO / meta["lcm_sim_params"]["path"]).read_text())
                         ["gripper_drive"]["width_calibration"])
        fingers = meta["robot_io"]["franka_hand"]["coords"]
        motors = [i for i, n in enumerate(meta["effort_layout"]) if n.startswith("franka_hand/")]
        j = np.clip(np.searchsorted(step, steps), 0, len(step) - 1)
        return {"franka_width_mm": np.abs(q[j][:, fingers]).sum(1) * 1e3,
                "franka_grip_n": np.abs(eff[j][:, motors]).sum(1),
                "ur_width_mm": np.interp(q[j][:, meta["driver_coords"]].mean(1),
                                         cal[:, 0], cal[:, 1]) * 1e3}
    except (OSError, KeyError, ValueError, TypeError):
        return dict.fromkeys(("franka_width_mm", "franka_grip_n", "ur_width_mm"))


class EpisodeStreamer(threading.Thread):
    """Replays one recorded episode of ``run_dir`` into ``path`` at ``pace`` x the frame period."""

    def __init__(self, run_dir: str | Path, path: str | Path, pace: float = 1.0,
                 episode: str | None = None) -> None:
        super().__init__(daemon=True)
        self.run_dir, self.path, self.pace = Path(run_dir), Path(path), pace
        idx = json.loads((self.run_dir / "index.json").read_text())
        episode = episode or idx["episodes"][0]["file"]
        self.labels = idx.get("phase_labels", [])
        z = np.load(self.run_dir / episode, allow_pickle=True)
        self.z = {k: z[k] for k in ("sim_ee_franka", "sim_ee_ur", "sim_belt_xyz", "pcd",
                                    "sim_wrap_deg", "sim_wrap_small_deg", "sim_grasp_ok",
                                    "sim_phase", "sim_time", "sim_step") if k in z.files}
        self.n = len(self.z["sim_time"])
        self.grip = _gripper(self.run_dir / "recordings" / Path(episode).stem,
                             self.z.get("sim_step", np.arange(self.n)))
        self.frames_written = 0
        self._halt = threading.Event()

    def frame(self, i: int) -> dict:
        z, g = self.z, self.grip

        def at(k):
            return None if k not in z else z[k][i]

        pc = at("pcd")
        if pc is not None and len(pc):
            pc = pc[::math.ceil(len(pc) / MAX_CLOUD)]
        phase = at("sim_phase")
        if phase is not None:
            phase = self.labels[phase] if 0 <= phase < len(self.labels) else int(phase)
        held = at("sim_grasp_ok")
        return {"t": round(float(z["sim_time"][i] - z["sim_time"][0]), 4), "step": i,
                "phase": phase, "franka_pose7": _r(at("sim_ee_franka")),
                "ur_pose7": _r(at("sim_ee_ur")),
                **{k: None if v is None else round(float(v[i]), 2) for k, v in g.items()},
                "belt_xyz": _r(at("sim_belt_xyz")), "cloud": _r(pc),
                "wrap_large_deg": _r(at("sim_wrap_deg")),
                "wrap_small_deg": _r(at("sim_wrap_small_deg")),
                "held": None if held is None else [bool(h) for h in held],
                "source": "mock", "run_dir": str(self.run_dir)}

    def run(self) -> None:
        for i in range(self.n):
            if self._halt.is_set():
                return
            write(self.path, self.frame(i))
            self.frames_written += 1
            self._halt.wait(SAMPLE_PERIOD_S * self.pace)

    def stop(self) -> None:
        self._halt.set()
