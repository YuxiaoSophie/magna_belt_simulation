#!/usr/bin/env python3
"""``control_step()`` == ``control_step_begin(); control_step_end()``, bit for bit.

Three child processes each build a RoundBeltOfflineSimulation, restore a start-state snapshot and
play one 200-step arm/gripper command sequence: whole, whole again (the control: a fresh build is
deterministic) and split. ``joint_q`` / ``joint_qd`` / ``body_q`` after every step must match
exactly, and ``control_step_end``'s host state must equal a fresh readback. CPU (bit-exact):
``CUDA_VISIBLE_DEVICES=``; on the GPU (sim lock) fresh builds already differ (atomics), so the
split is only held to the rerun's noise there:
    uv run --frozen python scripts/checks/check_control_step_split.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("ROUND_BELT_SCENE",
                      str(ROOT / "assets/round_belt_task/round_belt_scene_wr_pin640b1.yaml"))
SNAP = ROOT / "data/lcs/start_states/grasp_variants/set2/gv_00_osc.npz"
STEPS = 200


def play(out: str, split: bool) -> None:
    """Child: build, restore, play; save the per-step state bytes."""
    from round_belt_task.offline_simulation import RoundBeltOfflineSimulation
    from task_common import sim_snapshot

    sim = RoundBeltOfflineSimulation.build()
    sim.restore(sim_snapshot.load(SNAP))
    qf, qu = sim.arm_targets()
    rows = []
    for i in range(STEPS):
        a = 0.02 * np.sin(2 * np.pi * i / STEPS)
        sim.set_arm_targets(qf + a, qu - a)
        if i == STEPS // 2:
            sim.set_grippers(20.0, 191)
        if split:
            sim.control_step_begin()
            got = sim.control_step_end()
        else:
            got = sim.control_step()
        s = sim.state_0
        now = (s.joint_q.numpy(), s.joint_qd.numpy(), s.body_q.numpy())
        assert got is None or all(np.array_equal(x, y) for x, y in zip(got, now, strict=True))
        rows.append(np.frombuffer(b"".join(x.tobytes() for x in now), np.uint8))
    np.save(out, np.stack(rows))
    print(f"device {sim.model.device}")


def main() -> int:
    if sys.argv[1:2] in (["--help"], ["-h"]):
        print(__doc__)
        return 0
    if sys.argv[1:2] == ["--play"]:
        play(sys.argv[2], sys.argv[3] == "split")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        runs = {}
        for name in ("whole", "again", "split"):
            path = f"{tmp}/{name}.npy"
            p = subprocess.run([sys.executable, __file__, "--play", path, name], check=True,
                               capture_output=True, text=True)
            runs[name] = np.load(path)
        dev = next(ln for ln in p.stdout.splitlines() if ln.startswith("device"))
    w = runs["whole"]

    def mismatch(other) -> tuple[int, float]:
        d = np.abs(w.view(np.float32) - other.view(np.float32))
        return int((w != other).any(axis=1).sum()), float(d.max())

    (rerun, rerun_d), (diff, diff_d) = mismatch(runs["again"]), mismatch(runs["split"])
    print(f"{dev}: {STEPS} steps, mismatched steps (max |d|): rerun {rerun} ({rerun_d:.3g}), "
          f"split {diff} ({diff_d:.3g}), moved {bool((w[0] != w[-1]).any())}")
    assert (w[0] != w[-1]).any()
    if rerun == 0:
        assert diff == 0, "split differs from whole"
    else:  # GPU atomics: fresh builds differ; bit-exactness is the CPU run's to prove
        print("not bit-deterministic on this device: split vs whole compared to rerun noise")
        assert diff_d <= 10 * rerun_d, "split differs beyond the rerun noise"
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
