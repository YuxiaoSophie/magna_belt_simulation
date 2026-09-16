#!/usr/bin/env python3
"""Pick the LCM simulation's solver settings by measurement.

For each ``substeps/vbd_iterations`` pair: a rest bench of ``RoundBeltLcmSimulation``
(non-realtime ``control_step``, no peers), then, for settings within the step budget, a
diagnostic grasp-and-drag test in the task's real configuration: ``grasp_sequence`` puts the
belt into the closed Franka fingers via the sim's own trigger, then ``panda_joint1`` swings
+-0.3 rad under the in-script PD.  Prints one row per setting and the pick (most VBD
iterations within the budget).

Run:
    uv run python scripts/bench_lcm_sim_settings.py
    uv run python scripts/bench_lcm_sim_settings.py --settings 2/5,2/10 --write-yaml
    uv run python scripts/bench_lcm_sim_settings.py --settings 2/5,2/10 --rest-only
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import newton.examples

from drake import lcmt_schunk_wsg_status
from round_belt_task.constants import BELT_TRIGGER_BODY, LCM_SIM_PARAMS
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from utils.labels import body_index

sys.path.insert(0, str(Path(__file__).parent))
from lcm_peer_utils import HAND_CLOSED_WIDTH, StatePeer, grasp_sequence, pd_step

PRIVATE_LCM_URL = "udpm://239.255.76.68:7668?ttl=0"
DEFAULT_SETTINGS = "2/5,2/10,2/20,4/5,4/10"
WARMUP_STEPS, TIMED_STEPS, DRAG_WARMUP_STEPS = 200, 600, 20
HEADROOM_MS = 3.5
DRAG_S, HOLD_S, DRAG_AMPLITUDE, SAMPLE_S, END_SPEED_S = 4.0, 1.0, 0.3, 0.1, 0.5
MAX_TIP_GAP, MAX_PERIMETER_CHANGE, MAX_END_SPEED = 0.02, 3.0, 0.05
WIDTH_RANGE_MM = (4.1, 9.1)


@dataclass
class Row:
    substeps: int
    vbd_iterations: int
    ms_per_step: float = math.nan
    rest_stable: bool = False
    grasp: str = "not run"
    max_tip_gap_mm: float = math.nan
    perimeter_change_pct: float = math.nan
    end_speed: float = math.nan
    width_mm: float = math.nan
    close_width_mm: float = math.nan
    note: str = ""

    def cells(self) -> list[str]:
        hz = 1e3 / self.ms_per_step
        return [
            str(self.substeps), str(self.vbd_iterations), f"{self.ms_per_step:.2f}",
            f"{hz:.0f}", f"{0.005 * hz:.2f}", str(self.rest_stable), self.grasp,
            f"{self.max_tip_gap_mm:.1f}", f"{self.perimeter_change_pct:+.2f}",
            f"{self.end_speed:.4f}", f"{self.width_mm:.2f}", self.note or "-",
        ]


HEADER = [
    "substeps", "vbd_it", "ms/step", "Hz", "realtime_x", "rest_stable", "drag_diagnostic",
    "max_tip_gap_mm", "perimeter_change_%", "end_speed_m/s", "width_mm", "note",
]


def build_sim(row: Row, device: str | None) -> tuple[newton.viewer.ViewerBase, object]:
    parser = RoundBeltLcmSimulation.create_parser()
    parser.set_defaults(viewer="null", realtime=False, cameras=False, lcm_url=PRIVATE_LCM_URL)
    argv = ["--substeps", str(row.substeps), "--vbd-iterations", str(row.vbd_iterations)]
    sys.argv = [sys.argv[0], *argv, *(["--device", device] if device else [])]
    viewer, args = newton.examples.init(parser)
    return viewer, RoundBeltLcmSimulation(viewer, args)


def rest_bench(row: Row, device: str | None) -> None:
    viewer, sim = build_sim(row, device)
    for _ in range(WARMUP_STEPS):
        sim.control_step()
    wp.synchronize_device(sim.device)
    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        sim.control_step()
    wp.synchronize_device(sim.device)
    row.ms_per_step = (time.perf_counter() - t0) / TIMED_STEPS * 1e3
    body_q = sim.state_0.body_q.numpy()
    belt_z = body_q[np.asarray(sim.info.belt_bodies), 2]
    row.rest_stable = bool(np.isfinite(body_q).all() and belt_z.min() > sim.table_top_z - 0.01)
    viewer.close()


def perimeter(points: np.ndarray) -> float:
    return float(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1).sum())


def drag_test(row: Row, device: str | None) -> None:
    viewer, sim = build_sim(row, device)
    peer = StatePeer(PRIVATE_LCM_URL, sim.channels)
    widths: deque[float] = deque(maxlen=round(SAMPLE_S / sim.frame_dt))
    # Median over 0.1 s: at 2 substeps single PANDA_HAND_STATUS samples jitter by mm.
    peer.lc.subscribe(sim.channels.franka_hand_state_channel, lambda _, data: widths.append(
        lcmt_schunk_wsg_status.decode(data).actual_position_mm))
    try:
        for _ in range(DRAG_WARMUP_STEPS):
            sim.control_step()
        failure = _drag(sim, peer, row, widths)
    except RuntimeError as exc:
        failure = f"step {sim.step_index}: {exc}"
    row.grasp = "False" if failure else "True"
    row.note = failure
    viewer.close()


def _drag(sim, peer: StatePeer, row: Row, widths: deque[float]) -> str:
    """Grasp, drag and hold; returns the first failed criterion ("" when all hold)."""
    q_t, anchor = grasp_sequence(sim, peer)
    row.close_width_mm = float(np.median(widths))
    tip_body = body_index(list(sim.model.body_label), BELT_TRIGGER_BODY)
    belt = np.asarray(sim.info.belt_bodies, dtype=np.int64)
    centres = sim.state_0.body_q.numpy()[belt, :3].astype(np.float64)
    initial_perimeter = perimeter(centres)
    steps = round((DRAG_S + HOLD_S) / sim.frame_dt)
    sample_every, speed_steps = round(SAMPLE_S / sim.frame_dt), round(END_SPEED_S / sim.frame_dt)
    row.max_tip_gap_mm = row.perimeter_change_pct = row.end_speed = 0.0
    for i in range(1, steps + 1):
        t = i * sim.frame_dt
        q_target = q_t.copy()
        if t <= DRAG_S:
            q_target[0] += DRAG_AMPLITUDE * math.sin(2.0 * math.pi * t / DRAG_S)
        pd_step(sim, peer, q_target, HAND_CLOSED_WIDTH)
        body_q = sim.state_0.body_q.numpy()
        prev, centres = centres, body_q[belt, :3].astype(np.float64)
        if i > steps - speed_steps:
            speed = float(np.linalg.norm(centres - prev, axis=1).max()) / sim.frame_dt
            row.end_speed = max(row.end_speed, speed)
        if i % sample_every:
            continue
        where = f"step {sim.step_index} (drag t={t:.1f} s)"
        if not np.isfinite(body_q).all():
            return f"{where}: non-finite body_q"
        row.width_mm = float(np.median(widths))
        if not WIDTH_RANGE_MM[0] <= row.width_mm <= WIDTH_RANGE_MM[1]:
            return f"{where}: hand width {row.width_mm:.2f} mm"
        gap_mm = 1e3 * float(np.linalg.norm(body_q[anchor, :3] - body_q[tip_body, :3]))
        row.max_tip_gap_mm = max(row.max_tip_gap_mm, gap_mm)
        change = 100.0 * (perimeter(centres) / initial_perimeter - 1.0)
        if abs(change) > abs(row.perimeter_change_pct):
            row.perimeter_change_pct = change
        if gap_mm > 1e3 * MAX_TIP_GAP:
            return f"{where}: anchor-tip gap {gap_mm:.1f} mm"
        if abs(change) > MAX_PERIMETER_CHANGE:
            return f"{where}: perimeter change {change:+.2f} %"
        if centres[:, 2].min() <= sim.table_top_z - 0.01:
            return f"{where}: belt below the table top - 0.01"
    if row.end_speed >= MAX_END_SPEED:
        return f"end speed {row.end_speed:.4f} m/s"
    return ""


def pick(rows: list[Row]) -> Row | None:
    """Most VBD iterations among rest-stable rows within budget (ties: fewer ms/step)."""
    within = [r for r in rows if r.rest_stable and r.ms_per_step <= HEADROOM_MS]
    return min(within, key=lambda r: (-r.vbd_iterations, r.ms_per_step), default=None)


def write_yaml(row: Row) -> None:
    text = LCM_SIM_PARAMS.read_text()
    new, count = re.subn(
        r"^solver: \{substeps: \d+, vbd_iterations: \d+\}$",
        f"solver: {{substeps: {row.substeps}, vbd_iterations: {row.vbd_iterations}}}",
        text, flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"{LCM_SIM_PARAMS}: expected one 'solver:' line, found {count}")
    LCM_SIM_PARAMS.write_text(new)
    print(f"[YAML] {LCM_SIM_PARAMS.relative_to(REPO_ROOT)} solver <- "
          f"{row.substeps}/{row.vbd_iterations}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", default=DEFAULT_SETTINGS,
                        help="comma-separated substeps/vbd_iterations pairs")
    parser.add_argument("--device", default=None, help="Warp device, e.g. cuda:0")
    parser.add_argument("--out", type=Path, default=None, help="also write a markdown table")
    parser.add_argument("--rest-only", action="store_true", help="skip the grasp-and-drag test")
    parser.add_argument("--write-yaml", action="store_true",
                        help="write the pick into round_belt_lcm_sim.yaml solver:")
    cli = parser.parse_args()
    logger.remove()
    logger.add(sys.stdout, level="WARNING", format="{level: <7} | {message}")

    rows = []
    for item in cli.settings.split(","):
        substeps, iterations = (int(v) for v in item.split("/"))
        row = Row(substeps, iterations)
        rest_bench(row, cli.device)
        if cli.rest_only:
            row.grasp = "not run (--rest-only)"
        elif row.ms_per_step > HEADROOM_MS:
            row.grasp = "not run (over budget)"
        else:
            # Diagnostic only: the drag slips at every setting, 4 substeps included.
            drag_test(row, cli.device)
            print(f"[DRAG] {substeps}/{iterations}: width after close {row.close_width_mm:.2f} mm, "
                  f"end {row.width_mm:.2f} mm, max tip gap {row.max_tip_gap_mm:.1f} mm")
        print("  ".join(row.cells()), flush=True)
        rows.append(row)

    widths = [max(len(h), *(len(r.cells()[i]) for r in rows)) for i, h in enumerate(HEADER)]
    for cells in (HEADER, *(r.cells() for r in rows)):
        print("  ".join(c.rjust(w) for c, w in zip(cells, widths)))
    if cli.out is not None:
        lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
        lines += ["| " + " | ".join(r.cells()) + " |" for r in rows]
        cli.out.write_text("\n".join(lines) + "\n")

    chosen = pick(rows)
    if chosen is None:
        print(f"[PICK] none: no rest-stable setting at <= {HEADROOM_MS} ms/step", file=sys.stderr)
        return 1
    print(f"[PICK] substeps={chosen.substeps} vbd_iterations={chosen.vbd_iterations} "
          f"({chosen.ms_per_step:.2f} ms/step, {1e3 / chosen.ms_per_step:.0f} Hz)")
    if cli.write_yaml:
        write_yaml(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
