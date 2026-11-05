#!/usr/bin/env python3
"""V12-131: the kinematics / frames lcs_learning's ``seg_ctx`` inputs need, from magna's own models.

Writes ``X_WB`` (board pose of ``--params``), ``X_BW``, the known large-pulley pose ``X_WP`` and
``X_BP = X_BW X_WP``, and per arm the joint origins / axes / ``X_post`` of ``FrankaTip`` and
``UrTracking`` (``arm_kinematics.UrdfChain``). magna's runtime uses its FK directly;
``scripts/checks/check_ctx_encoder.py`` checks lcs_learning's vector against it.

    uv run --frozen python scripts/lcs/export_ctx_kin.py \\
        --out /home/hienbui/git/lcs_learning/lcs_learning/data/ctx_kin_sim.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from round_belt_task.arm_kinematics import FrankaTip, UrTracking
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, board_pose
from task_common import belt_cloud as bc
from task_common.latent_encoder import ctx_frames


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = ctx_frames(board_pose(a.params), bc.known_large_pulley_pose7())
    for arm, ch in (("franka", FrankaTip), ("ur", UrTracking)):
        out[f"{arm}_origins"] = np.stack([j.origin for j in ch.joints])
        out[f"{arm}_axes"] = np.stack([j.axis for j in ch.joints])
        out[f"{arm}_X_post"] = ch.X_post
    out["params"] = np.array(str(a.params.resolve()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {sorted(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
