"""Small run/episode I/O helpers shared by the LCS collector and the MPC evaluation harness."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from round_belt_task.commander import mat3_to_quat
from task_common import REPO_ROOT
from task_common import lcs_dataset as lcs


def pose7_mat(m: np.ndarray) -> np.ndarray:
    """``xyz`` + ``wxyz`` (``w >= 0``), the convention of ``franka_measured_pose7``."""
    return np.concatenate([np.asarray(m[:3, 3], dtype=np.float64), mat3_to_quat(m[:3, :3])])


def git_info(root: Path = REPO_ROOT) -> dict:
    def run(*cmd) -> str:
        return subprocess.run(["git", *cmd], cwd=root, capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    """Atomic ``json.dump`` (numpy-aware)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lcs.json_default) + "\n")
    os.replace(tmp, path)


def finish_recording(sim, out: Path, name: str, reason: str) -> str | None:
    """Close ``sim``'s run recording and move it to ``<out>/recordings/<name>``."""
    recorder = sim.recorder
    if recorder is None:
        return None
    sim.close_recording(reason)
    sim.recorder = None
    target = out / "recordings" / name
    if target.exists():
        return str(recorder.path.relative_to(out))
    recorder.path.rename(target)
    return str(target.relative_to(out))


def slant_row(metrics: dict) -> dict:
    """The final slant of ``slant_episode`` metrics as index.json fields."""
    def num(x) -> float | None:  # NaN (no plane fit) -> null in index.json
        return float(x) if np.isfinite(x) else None

    return {"final_slant_deg": num(metrics["slant_deg"]),
            "final_slant_axis_deg": num(metrics["slant_axis_deg"]),
            "final_slant_dir": str(metrics["slant_dir"])}
