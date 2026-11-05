#!/usr/bin/env python3
"""CPU checks for the stage-2 plumbing (V12-91): the opt-in scene variant (625 mm belt, bend x3,
pulley damping 0.03) differs from the default only where intended, the per-seat wrap merge gap
(small pulley ~72 deg, large unchanged at 30) and ``MppiParams.u_franka_only``. No sim, no LCM.

Run:
    uv run --frozen python scripts/checks/check_stage2_scene.py
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from round_belt_task.outcome import LARGE_SEAT_MM, SMALL_SEAT_MM, classify, frame_metrics
from task_common import mppi_lcs
from task_common.replay_metrics import WRAP_MERGE_DEG, _largest_arc_deg, wrap_merge_deg

ASSETS = REPO_ROOT / "assets" / "round_belt_task"


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _rod(path: Path) -> dict:
    doc = yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # BaseLoader: the !Rpy tags stay strings
    return next(d["add_rod_ellipse"] for d in doc["directives"] if "add_rod_ellipse" in d)


def _polygon_mm(semi_axes, n: int) -> float:
    th = 2 * np.pi * np.arange(n + 1) / n
    p = np.c_[float(semi_axes[0]) * np.cos(th), float(semi_axes[1]) * np.sin(th)]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * 1e3)


def check_scene_variant() -> None:
    base, var = ASSETS / "round_belt_scene.yaml", ASSETS / "round_belt_scene_stage2.yaml"
    a, b = _rod(base), _rod(var)
    changed = {k for k in a if a[k] != b.get(k)}
    _require(changed == {"semi_axes", "bend_stiffness"}, f"rod keys changed: {changed}")
    n = int(b["num_elements"])
    la, lb = _polygon_mm(a["semi_axes"], n), _polygon_mm(b["semi_axes"], n)
    _require(abs(la - 652.2) < 0.5 and abs(lb - 625.0) < 0.5, f"rest lengths {la:.1f} / {lb:.1f} mm")
    _require(math.isclose(float(b["bend_stiffness"]), 3 * float(a["bend_stiffness"])), "bend x3")
    diff = [(x, y) for x, y in zip(base.read_text().splitlines(), var.read_text().splitlines()[3:])
            if x != y]
    _require(len(base.read_text().splitlines()) + 3 == len(var.read_text().splitlines())
             and len(diff) == 4, f"scene yaml differs in {len(diff)} lines besides the header")
    damp = {j.get("name"): float(j.find("dynamics").get("damping"))
            for j in ET.parse(ASSETS / "round_belt_task_board_stage2.urdf").getroot().findall("joint")
            if j.find("dynamics") is not None}
    _require(damp == {"small_round_pulley_joint": 0.03, "large_round_pulley_joint": 0.03},
             f"stage-2 pulley damping {damp}")
    print(f"ok scene variant: rest {la:.1f} -> {lb:.1f} mm, bend x3, pulley damping 0.03")


def _ring(seat_mm: float, az_deg: np.ndarray, h_mm: float = 0.0) -> np.ndarray:
    a = np.radians(az_deg)
    return np.c_[seat_mm * np.cos(a), seat_mm * np.sin(a), np.full_like(a, h_mm)] * 1e-3


def check_wrap_merge() -> None:
    _require(wrap_merge_deg(LARGE_SEAT_MM) == WRAP_MERGE_DEG, "large merge must stay 30 deg")
    small = wrap_merge_deg(SMALL_SEAT_MM)
    _require(60.0 < small < 80.0, f"small merge {small:.1f} deg")
    az = np.array([-55.0, -22.0, 16.0, 54.0])  # the F_up3 end state on the small seat
    _require(_largest_arc_deg(az) == 0.0 and abs(_largest_arc_deg(az, small) - 109.0) < 1e-6,
             "a 4-body small-pulley wrap must merge only with the small gap")
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    fs = frame_metrics(_ring(SMALL_SEAT_MM, az), pose, SMALL_SEAT_MM)
    _require(classify(fs) == "engaged" and abs(fs.wrap_deg - 109.0) < 1e-6, f"small ring {fs}")
    large_az = np.arange(-80.0, 81.0, 15.0)
    fl = frame_metrics(_ring(LARGE_SEAT_MM, large_az), pose, LARGE_SEAT_MM)
    _require(abs(fl.wrap_deg - 150.0) < 1e-6, f"large ring {fl.wrap_deg}")
    gap = np.array([-80.0, -40.0, 0.0])  # 40 deg gaps: no merge on the large seat
    _require(frame_metrics(_ring(LARGE_SEAT_MM, gap), pose, LARGE_SEAT_MM).wrap_deg == 0.0,
             "large seat must not merge 40 deg gaps")
    print(f"ok wrap merge: small {small:.1f} deg, large {WRAP_MERGE_DEG:.0f} deg")


def check_franka_only() -> None:
    lcs = SimpleNamespace(n_u=12, u_lb_np=-np.ones(12), u_ub_np=np.ones(12))
    lb, ub = mppi_lcs.MppiParams(u_franka_only=True, cuda_graph=False).bounds(lcs)
    ur = list(mppi_lcs.UR_CHANNELS)
    fr = [i for i in range(12) if i not in ur]
    _require(np.all(lb[ur] == 0) and np.all(ub[ur] == 0), "UR channels must be pinned")
    _require(np.all(lb[fr] == -1) and np.all(ub[fr] == 1), "Franka channels must stay free")
    lb0, ub0 = mppi_lcs.MppiParams(cuda_graph=False).bounds(lcs)
    _require(np.all(lb0 == -1) and np.all(ub0 == 1), "default must not pin anything")
    print("ok u_franka_only: UR channels 3-5, 9-11 pinned, Franka free")


def main() -> int:
    for fn in (check_scene_variant, check_wrap_merge, check_franka_only):
        fn()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
