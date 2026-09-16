#!/usr/bin/env python3
"""Summarize one end-to-end round-belt run from its sim log and assembly-controller log.

Works on the Newton LCM sim log (``[STATS]``, ``[BELT]``, ``[GRASP]`` lines) and on the Drake
``magna_simulation`` log (those fields print ``n/a``).  Lines may carry a ``[<unix seconds>] ``
prefix added by the launcher; wall times are then relative to the controller log's first line.

Run:
    uv run python scripts/summarize_e2e_logs.py <sim.log> <controller.log>
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

STATS_PERIOD = 5.0
HOLD_GAP_MM = 15.0
FRANKA_HOLD_WIDTH_MM = 10.0
FRANKA_MIN_WIDTH_MM = 1.0
UR_HOLD_BYTE = 200
UR_RELEASE_BYTE = 100
MAX_LISTED = 20
MAX_ERRORS = 5

STAMP = re.compile(r"^\[(\d+\.\d+)\] ?(.*)$")
STATS = re.compile(
    r"\[STATS\] step (\d+): ([\d.]+) steps/s, compute mean ([\d.]+) ms max ([\d.]+) ms"
)
STALE = re.compile(r"\[LCM\] franka input: stale-damping")
PLACED = re.compile(r"\[BELT\] placed at step (\d+) \(sim t=([\d.]+) s\)")
DRAKE_PLACED = re.compile(r"Set belt position at robot EE position: \[(.*)\]")
FINAL = re.compile(r"\[BELT\] final centroid \(([^)]*)\), min z (-?[\d.]+)")
GRASP = re.compile(
    r"\[GRASP\] t ([\d.]+) s: hand width (-?[\d.]+) mm, belt->finger_tip (\S+) mm, "
    r"belt->2f85 pads (\S+)/(\S+) mm,(?: belt->2f85 tip (\S+) mm,)? robotiq byte (\S+) "
    r"\(status (\d+)\)"
)
PHASES = (
    "[compiled-playback]",
    "Pre-MPC target",
    "All pre-MPC targets completed",
    "Switching to MPC phase",
    "MPC reached target",
    "All MPC targets completed",
    "MPC completed",
    "Post-MPC target",
    "All post-MPC targets completed",
)
ERROR = re.compile(r"error|abort|exception|terminate called|what\(\)", re.IGNORECASE)


@dataclass
class Line:
    number: int
    stamp: float | None
    text: str


@dataclass
class Grasp:
    t: float
    width: float
    tip: float
    pad: float
    ur_tip: float
    byte: int
    status: int


def read_lines(path: Path) -> list[Line]:
    lines = []
    for number, raw in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        match = STAMP.match(raw)
        if match:
            lines.append(Line(number, float(match.group(1)), match.group(2)))
        else:
            lines.append(Line(number, None, raw))
    return lines


def first_stamp(lines: list[Line]) -> float | None:
    return next((line.stamp for line in lines if line.stamp is not None), None)


def when(line: Line, origin: float | None) -> str:
    if line.stamp is None or origin is None:
        return f"line {line.number}"
    return f"{line.stamp - origin:7.1f} s"


def _mm(value: str) -> float:
    return float("inf") if value == "n/a" else float(value)


def windows(samples: list[Grasp], held: list[bool]) -> list[tuple[float, float]]:
    spans, start, last = [], None, 0.0
    for sample, is_held in zip(samples, held):
        if is_held:
            start = sample.t if start is None else start
            last = sample.t
        elif start is not None:
            spans.append((start, last))
            start = None
    if start is not None:
        spans.append((start, last))
    return spans


def fmt_windows(spans: list[tuple[float, float]]) -> str:
    if not spans:
        return "none"
    return ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in spans)


def summarize_sim(lines: list[Line], origin: float | None) -> None:
    stats = [(line, STATS.search(line.text)) for line in lines]
    stats = [(line, m) for line, m in stats if m]
    if stats:
        rates = [float(m.group(2)) for _, m in stats]
        print(f"steps/s (all {len(rates)} samples): min {min(rates):.1f}, "
              f"mean {sum(rates) / len(rates):.1f}")
        # Drop the final partial window, and windows that began before the controller started.
        attached = [
            float(m.group(2)) for line, m in stats[:-1]
            if origin is None or line.stamp is None or line.stamp - STATS_PERIOD >= origin
        ]
        if attached:
            print(f"steps/s (stack attached, {len(attached)} samples): min {min(attached):.1f}, "
                  f"mean {sum(attached) / len(attached):.1f}")
        else:
            print("steps/s (stack attached): n/a")
        print(f"max compute: {max(float(m.group(4)) for _, m in stats):.2f} ms")
        print(f"stale-damping windows: {sum(1 for line in lines if STALE.search(line.text))}")
    else:
        print("steps/s: n/a\nmax compute: n/a\nstale-damping windows: n/a")

    placed = next(((line, m) for line in lines if (m := PLACED.search(line.text))), None)
    drake = next(((line, m) for line in lines if (m := DRAKE_PLACED.search(line.text))), None)
    if placed:
        line, m = placed
        print(f"[BELT] placed: yes at {when(line, origin)} (sim t {m.group(2)} s)")
    elif drake:
        line, m = drake
        print(f"[BELT] placed: n/a; Drake trigger at {when(line, origin)} (EE [{m.group(1)}])")
    else:
        print("[BELT] placed: no (or n/a)")
    final = next((m for line in reversed(lines) if (m := FINAL.search(line.text))), None)
    print(f"belt final: centroid ({final.group(1)}), min z {final.group(2)}" if final
          else "belt final: n/a")

    samples = [
        Grasp(float(m.group(1)), float(m.group(2)), _mm(m.group(3)),
              min(_mm(m.group(4)), _mm(m.group(5))), _mm(m.group(6) or "n/a"),
              0 if m.group(7) == "none" else int(m.group(7)), int(m.group(8)))
        for line in lines if (m := GRASP.search(line.text))
    ]
    if not samples:
        print("franka hold: n/a\nur hold: n/a\nur release events: n/a")
        return
    franka = [
        s.tip <= HOLD_GAP_MM and FRANKA_MIN_WIDTH_MM <= s.width <= FRANKA_HOLD_WIDTH_MM
        for s in samples
    ]
    ur = [s.ur_tip <= HOLD_GAP_MM and s.byte >= UR_HOLD_BYTE for s in samples]
    print(f"[GRASP] samples: {len(samples)} (sim t {samples[0].t:.0f}-{samples[-1].t:.0f} s)")
    print(f"franka hold (tip <= {HOLD_GAP_MM:g} mm, width {FRANKA_MIN_WIDTH_MM:g}-"
          f"{FRANKA_HOLD_WIDTH_MM:g} mm): {fmt_windows(windows(samples, franka))}")
    print(f"ur hold (2f85 tip <= {HOLD_GAP_MM:g} mm, byte >= {UR_HOLD_BYTE}): "
          f"{fmt_windows(windows(samples, ur))}")
    releases, partial = [], []
    for prev, cur, was_held in zip(samples, samples[1:], ur):
        if not was_held or cur.byte >= UR_HOLD_BYTE:
            continue
        event = (f"t {cur.t:.0f} s byte {prev.byte}->{cur.byte} (status {cur.status}, "
                 f"tip {cur.ur_tip:.1f} mm, pad {cur.pad:.1f} mm)")
        (releases if cur.byte < UR_RELEASE_BYTE else partial).append(event)
    print(f"ur release events (byte < {UR_RELEASE_BYTE}): {'; '.join(releases) or 'none'}")
    print(f"ur partial releases (byte {UR_RELEASE_BYTE}-{UR_HOLD_BYTE - 1}): "
          f"{'; '.join(partial) or 'none'}")


def summarize_controller(lines: list[Line], origin: float | None) -> None:
    markers = [line for line in lines if any(p in line.text for p in PHASES)]
    print(f"phase markers ({len(markers)}):")
    for line in markers[:MAX_LISTED]:
        print(f"  {when(line, origin)}  {line.text.strip()[:110]}")
    if len(markers) > MAX_LISTED:
        print(f"  ... {len(markers) - MAX_LISTED} more; last: {when(markers[-1], origin)}  "
              f"{markers[-1].text.strip()[:110]}")
    for label, needle in (("MPC phase reached", "Switching to MPC phase"),
                          ("MPC completed", "MPC completed"),
                          ("terminate", "Switching to Terminate")):
        hit = next((line for line in lines if needle in line.text), None)
        print(f"{label}: {when(hit, origin) if hit else 'no'}")
    errors = [line for line in lines if ERROR.search(line.text)]
    print(f"error lines: {len(errors)}")
    for line in errors[:MAX_ERRORS]:
        print(f"  {when(line, origin)}  {line.text.strip()[:110]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sim_log", type=Path)
    parser.add_argument("controller_log", type=Path)
    args = parser.parse_args()
    sim, controller = read_lines(args.sim_log), read_lines(args.controller_log)
    origin = first_stamp(controller)
    print(f"== sim log {args.sim_log}")
    summarize_sim(sim, origin)
    print(f"== controller log {args.controller_log}")
    summarize_controller(controller, origin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
