#!/usr/bin/env python3
"""Synthetic check of the large-pulley outcome classifier (``round_belt_task.outcome``).

Builds a 48-body closed loop (~652 mm) wrapping 180 deg of the seat circle of a pulley at the
origin, moves it into each outcome and checks the labels, then repeats in rotated/spun frames.
No sim, no LCM.

Run:
    uv run python scripts/checks/check_lcs_outcome.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from round_belt_task.outcome import (
    LABELS,
    LARGE_SEAT_MM,
    OutcomeThresholds,
    classify,
    classify_episode,
    frame_metrics,
    slant_episode,
    slant_metrics,
)
from task_common.replay_metrics import quat_rotate

NUM_BODIES = 48
LOOP_MM = 652.0
# Diverging legs keep the loop clear of the neighbour band once it is pulled 40 mm outward.
LEG_ANGLE_DEG = 38.0
IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _quat(axis, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    half = np.radians(angle_deg) / 2.0
    return np.append(axis * np.sin(half), np.cos(half))


def _rotate_about(points: np.ndarray, pivot, axis, angle_deg: float) -> np.ndarray:
    q = np.broadcast_to(_quat(axis, angle_deg), (len(points), 4))
    return pivot + quat_rotate(q, points - pivot)


def _loop_m() -> np.ndarray:
    """Near arc (azimuth 90..270 deg at the seat), two diverging legs, a far closing arc."""
    r = LARGE_SEAT_MM
    c, s = np.cos(np.radians(LEG_ANGLE_DEG)), np.sin(np.radians(LEG_ANGLE_DEG))
    leg = (LOOP_MM - 2.0 * np.pi * r) / (2.0 + np.pi * s)
    far_r, far_x = r + leg * s, leg * c
    lengths = np.array([np.pi * r, leg, np.pi * far_r, leg])
    edges = np.concatenate(([0.0], np.cumsum(lengths)))
    # Half-step phase so the near arc is symmetric about its midpoint (azimuth 180 deg).
    arc_len = (np.arange(NUM_BODIES) + 0.5) * LOOP_MM / NUM_BODIES
    pts = np.zeros((NUM_BODIES, 3))
    for i, t in enumerate(arc_len):
        seg = int(np.searchsorted(edges, t, side="right")) - 1
        u = t - edges[seg]
        if seg == 0:
            a = np.radians(90.0) + u / r
            pts[i, :2] = r * np.cos(a), r * np.sin(a)
        elif seg == 1:
            pts[i, :2] = u * c, -r - u * s
        elif seg == 2:
            a = np.radians(-90.0) + u / far_r
            pts[i, :2] = far_x + far_r * np.cos(a), far_r * np.sin(a)
        else:
            pts[i, :2] = (leg - u) * c, r + (leg - u) * s
    return pts * 1e-3


def _scenarios() -> dict[str, tuple[np.ndarray, str]]:
    base = _loop_m()
    seat = LARGE_SEAT_MM * 1e-3
    z = np.array([0.0, 0.0, 1.0])
    return {
        "seated": (base, "engaged"),
        "lifted": (base + 0.010 * z, "over"),
        "lowered": (base - 0.010 * z, "under"),
        # Pivot on the horizontal tangent at the arc's y<0 end: that end seated, the far end up.
        "tilted": (_rotate_about(base, np.array([0.0, -seat, 0.0]), [1.0, 0.0, 0.0], 20.0),
                   "slanted"),
        "pulled_out": (base + np.array([-0.040, 0.0, 0.0]), "outside"),
    }


def _label(belt: np.ndarray, pose: np.ndarray, th: OutcomeThresholds):
    fm = frame_metrics(belt, pose, th=th)
    return classify(fm, th), fm


def _run(ctx: SimpleNamespace, name: str):
    belt, _ = ctx.scenarios[name]
    return _label(belt, IDENTITY_POSE, ctx.th)


@check("O0 seated arc -> engaged")
def check_o0(ctx: SimpleNamespace) -> str:
    label, fm = _run(ctx, "seated")
    _require(label == "engaged", f"label {label!r} != 'engaged' ({fm})")
    _require(170.0 <= fm.wrap_deg <= 190.0, f"wrap {fm.wrap_deg:.1f} deg outside [170, 190]")
    loop = np.linalg.norm(np.diff(ctx.base, axis=0, append=ctx.base[:1]), axis=1).sum() * 1e3
    _require(abs(loop - LOOP_MM) < 10.0, f"synthetic loop {loop:.1f} mm, want ~{LOOP_MM}")
    return f"wrap {fm.wrap_deg:.1f} deg, {fm.seated_bodies} seated, loop {loop:.0f} mm"


@check("O1 lifted +10 mm -> over")
def check_o1(ctx: SimpleNamespace) -> str:
    label, fm = _run(ctx, "lifted")
    _require(label == "over", f"label {label!r} != 'over' ({fm})")
    return f"h_median {fm.h_median_mm:+.1f} mm, wrap {fm.wrap_deg:.1f} deg"


@check("O2 lowered -10 mm -> under")
def check_o2(ctx: SimpleNamespace) -> str:
    label, fm = _run(ctx, "lowered")
    _require(label == "under", f"label {label!r} != 'under' ({fm})")
    return f"h_median {fm.h_median_mm:+.1f} mm, wrap {fm.wrap_deg:.1f} deg"


@check("O3 tilted 20 deg -> slanted")
def check_o3(ctx: SimpleNamespace) -> str:
    label, fm = _run(ctx, "tilted")
    _require(label == "slanted", f"label {label!r} != 'slanted' ({fm})")
    _require(fm.h_max_mm >= 15.0, f"high side h_max {fm.h_max_mm:.1f} mm < 15")
    return (f"wrap {fm.wrap_deg:.1f} deg, h {fm.h_min_mm:+.1f}..{fm.h_max_mm:+.1f} mm, "
            f"{fm.seated_bodies} seated")


@check("O4 pulled 40 mm outward -> outside")
def check_o4(ctx: SimpleNamespace) -> str:
    label, fm = _run(ctx, "pulled_out")
    _require(label == "outside", f"label {label!r} != 'outside' ({fm})")
    return f"{fm.n_neighbour} bodies in the neighbourhood"


@check("O5 frame invariance")
def check_o5(ctx: SimpleNamespace) -> str:
    q = _quat([0.3, -0.8, 0.5], 67.0)
    origin = np.array([0.41, -0.27, 0.93])
    spin = _quat([0.0, 0.0, 1.0], 123.0)
    for name, (belt, want) in ctx.scenarios.items():
        ref_label, ref = _label(belt, IDENTITY_POSE, ctx.th)
        moved = origin + quat_rotate(np.broadcast_to(q, (len(belt), 4)), belt)
        label, fm = _label(moved, np.concatenate((origin, q)), ctx.th)
        _require(label == want, f"{name} rotated: label {label!r} != {want!r}")
        _require(abs(fm.wrap_deg - ref.wrap_deg) < 1e-6,
                 f"{name} rotated: wrap {fm.wrap_deg} != {ref.wrap_deg}")
        label, fm = _label(belt, np.concatenate((np.zeros(3), spin)), ctx.th)
        _require(label == ref_label == want, f"{name} spun: label {label!r} != {want!r}")
        _require(abs(fm.wrap_deg - ref.wrap_deg) < 1e-6,
                 f"{name} spun: wrap {fm.wrap_deg} != {ref.wrap_deg}")
    # Episode API: last frame wins; majority over last_n with ties to the latest frame.
    seq = [ctx.scenarios[n][0] for n in ("lifted", "seated", "seated", "lowered")]
    poses = np.tile(IDENTITY_POSE, (len(seq), 1))
    label, metrics = classify_episode(np.stack(seq), poses, ctx.th)
    _require(label == "under", f"episode last-frame label {label!r} != 'under'")
    _require(all(v.shape == (len(seq),) for v in metrics.values()), "metrics_T shape mismatch")
    label, _ = classify_episode(np.stack(seq), poses, ctx.th, last_n=3)
    _require(label == "engaged", f"episode last_n=3 label {label!r} != 'engaged'")
    _require(set(LABELS) >= {w for _, w in ctx.scenarios.values()}, "label outside LABELS")
    return f"{len(ctx.scenarios)} scenarios rotated + spun, episode API ok"


@check("O6 slant metrics")
def check_o6(ctx: SimpleNamespace) -> str:
    tangent = np.array([1.0, 0.0, 0.0])  # Franka -> UR along +X
    tol = 1e-6
    sm = slant_metrics(ctx.base, IDENTITY_POSE, tangent, th=ctx.th)
    _require(sm.slant_deg < tol and sm.slant_dir == "level", f"seated: {sm}")
    sm = slant_metrics(ctx.scenarios["pulled_out"][0], IDENTITY_POSE, tangent, th=ctx.th)
    _require(np.isnan(sm.slant_deg) and sm.slant_dir == "n/a", f"pulled_out: {sm}")
    sm = slant_metrics(ctx.scenarios["tilted"][0], IDENTITY_POSE, tangent, th=ctx.th)
    _require(abs(sm.slant_deg - 20.0) < tol and abs(sm.slant_axis_deg) < tol
             and sm.slant_dir == "roll+", f"tilted 20 deg about +X: {sm}")
    # +alpha about azimuth phi: slant alpha, axis phi; about +Y the +X (UR) side goes down.
    cases = ((90.0, "franka_high"), (-90.0, "ur_high"), (0.0, "roll+"), (180.0, "roll-"),
             (35.0, "roll+"), (125.0, "franka_high"))
    q = _quat([0.3, -0.8, 0.5], 67.0)
    origin = np.array([0.41, -0.27, 0.93])
    for phi, want in cases:
        axis = [np.cos(np.radians(phi)), np.sin(np.radians(phi)), 0.0]
        belt = _rotate_about(ctx.base, np.zeros(3), axis, 8.0)
        sm = slant_metrics(belt, IDENTITY_POSE, tangent, th=ctx.th)
        d_axis = (sm.slant_axis_deg - phi + 180.0) % 360.0 - 180.0
        _require(abs(sm.slant_deg - 8.0) < tol and abs(d_axis) < tol and sm.slant_dir == want,
                 f"phi {phi}: {sm} (want 8 deg, axis {phi}, {want})")
        moved = origin + quat_rotate(np.broadcast_to(q, (len(belt), 4)), belt)
        sm2 = slant_metrics(moved, np.concatenate((origin, q)), quat_rotate(q, tangent),
                            th=ctx.th)
        _require(abs(sm2.slant_deg - sm.slant_deg) < tol
                 and abs(sm2.slant_axis_deg - sm.slant_axis_deg) < tol
                 and sm2.slant_dir == sm.slant_dir, f"phi {phi} rotated: {sm2} != {sm}")
    ep = slant_episode(np.stack([ctx.base, ctx.scenarios["tilted"][0]]),
                       np.tile(IDENTITY_POSE, (2, 1)), tangent, ctx.th)
    _require(ep["slant_deg_t"].shape == (2,) and ep["slant_deg"] == ep["slant_deg_t"][-1]
             and ep["slant_dir"] == "roll+", f"episode API: {ep}")
    return f"level/n/a/tilted 20 deg ok, {len(cases)} axes x (identity, rotated frame)"


def main() -> int:
    t0 = time.perf_counter()
    base = _loop_m()
    ctx = SimpleNamespace(th=OutcomeThresholds(), base=base, scenarios=_scenarios())
    for name, fn in CHECKS:
        try:
            detail = fn(ctx)
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - report the crash as a failed check
            print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[PASS] {name}: {detail}")
    print(f"ALL OUTCOME CHECKS PASSED ({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
