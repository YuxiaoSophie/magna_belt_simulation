#!/usr/bin/env python3
"""CPU checks for the stage-2 collector (V12-93, ``scripts/lcs/stage2_collect.py``): parameter
sampling, the plan vs the validated D10L_rel scratch plan, the tilt cap, knot noise and the trip
return path, trip logic on synthetic signal sequences, the budget halving, loop draws, segment
spans and the small-pulley cloud features. No sim, no LCM.

Run:
    ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_stage2.yaml \\
      uv run --frozen python scripts/checks/check_stage2_collect.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import miles_seed as ms
import stage2_collect as s2

from round_belt_task.outcome import SMALL_SEAT_MM
from task_common import belt_cloud as bc

# D10L_rel (scratch p2/plans42.json, 4/4 in sim): world mm of w0..w10, out, lift; quats of w4..w10
D10L_MM = [[415.9947, 99.0561, 64.5743], [416.5018, 159.6757, 64.6471],
           [485.2150, 172.0021, 65.0611], [485.4263, 172.1262, 48.7158],
           [485.6260, 164.5946, 38.0535], [454.2440, 185.4598, 33.3244],
           [442.7192, 180.8730, 33.4270], [433.8626, 173.0642, 30.6337],
           [429.2841, 161.5670, 33.6231], [431.2105, 150.0806, 33.6211],
           [435.7862, 140.1222, 33.6362]]
D10L_POST_MM = [[423.0705, 124.6852, 33.5445], [422.9070, 124.6529, 61.6440]]
D10L_Q = [[0.9675272377, 0.0791366749, 0.2358097733, 0.0449686758],
          [0.9691375172, 0.0114348360, 0.2416166494, -0.0475721761],
          [0.9567525320, -0.0305907162, 0.2020825134, 0.2070059380],
          [0.8325031134, -0.0114744159, 0.1544499962, 0.5319324230],
          *[[0.7375110787, 0.0391640069, -0.0289847989, 0.6735751411]] * 3]
# V45c_nominal_0's Franka pose at the stage-2 start (start index ``measured.ee_franka``)
EE0 = np.array([0.3688, 0.0804, 0.0470, 0.6363, -0.0611, -0.1360, 0.7569])
EE0[3:] /= np.linalg.norm(EE0[3:])


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _qdist_deg(a, b) -> float:
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


@check("S1 parameter draws inside PARAM_RANGES (nominal too)")
def check_params() -> str:
    rng = np.random.default_rng(0)
    draws = [s2.draw_params(rng) for _ in range(2000)]
    for k, (lo, hi) in s2.PARAM_RANGES.items():
        v = np.array([d[k] for d in draws])
        _require(lo <= v.min() and v.max() <= hi, f"{k} outside [{lo}, {hi}]")
        _require(v.max() - v.min() > 0.9 * (hi - lo), f"{k} does not span its range")
        _require(lo <= s2.NOMINAL[k] <= hi, f"nominal {k} outside the range")
    _require(set(s2.NOMINAL) == set(s2.PARAM_RANGES), "NOMINAL keys != PARAM_RANGES keys")
    a = s2.draw_params(np.random.default_rng([1, 2, 3, 0]))
    _require(a == s2.draw_params(np.random.default_rng([1, 2, 3, 0])), "draws not seeded")
    return f"{len(draws)} draws x {len(s2.PARAM_RANGES)} params in range, seeded"


@check("S2 nominal plan reproduces D10L_rel (positions, wrap/arc quats, out, lift)")
def check_plan() -> str:
    hw = s2.hw_poses()
    plan = s2.build_plan(s2.NOMINAL, EE0, hw)
    _require([lab for lab, _, _ in plan] == [f"w{i}" for i in range(11)], "labels")
    _require([ph for _, ph, _ in plan] == ["transit"] * 4 + ["descent"] + ["wrap"] * 3
             + ["arc"] * 3, "phases")
    pos = np.array([p[:3] for _, _, p in plan]) * 1e3
    err = float(np.abs(pos - np.array(D10L_MM)).max())
    _require(err < 0.01, f"plan positions {err:.4f} mm off D10L_rel")
    qerr = max(_qdist_deg(p[3:], q) for (_, _, p), q in zip(plan[4:], D10L_Q, strict=True))
    _require(qerr < 0.01, f"w4..w10 orientation {qerr:.2e} deg off D10L_rel")
    out7, lift7 = s2.post_poses(s2.NOMINAL, hw)
    perr = float(np.abs(np.array([out7[:3], lift7[:3]]) * 1e3 - np.array(D10L_POST_MM)).max())
    _require(perr < 0.01, f"post targets {perr:.4f} mm off D10L_rel")
    r, az, h = s2.small_polar(plan[3][2][:3])
    _require(abs(r - 36.15) < 0.05 and abs(az + 68.91) < 0.05 and abs(h - 13.65) < 0.05,
             f"P10 polar {r:.2f}/{az:.2f}/{h:.2f}")
    p = {**s2.NOMINAL, "wrap_dh_mm": 2.0, "arc_raise_mm": 4.0, "end_az_deg": 147.0,
         "hover_h_mm": 25.0}
    plan2 = s2.build_plan(p, EE0, hw)
    dh = [s2.small_polar(b[:3])[2] - s2.small_polar(a[:3])[2]
          for (_, _, a), (_, _, b) in zip(plan[4:8], plan2[4:8], strict=True)]
    _require(np.allclose(dh, 2.0, atol=1e-6), f"wrap dh {dh}")
    _require(all(abs(s2.small_polar(q[:3])[2] - 25.0) < 1e-6 for _, _, q in plan2[:3]),
             "hover height")
    _require(abs(s2.small_polar(plan2[10][2][:3])[1] - 147.0) < 1e-6
             and abs(s2.small_polar(plan2[10][2][:3])[2] - (s2.ARC_H0_MM + 4.0)) < 1e-6,
             "end azimuth / arc raise")
    return (f"positions {err:.4f} mm, quats {qerr:.1e} deg, post {perr:.4f} mm; dh / raise / "
            "end az / hover h move the right waypoints")


@check("S3 transit orientation: slerp fractions, tilt cap, heading kept")
def check_cap() -> str:
    hw = s2.hw_poses()
    q10 = hw["P10"][3:]
    worst, n = 0.0, 0
    for cap in (5.0, 7.5, 10.0):
        for f in (0.05, 0.1, 0.2, 0.4, 1.0):
            q = s2.slerp(EE0[3:], q10, f)
            c = s2.cap_tilt(q, cap)
            _require(abs(np.linalg.norm(c) - 1.0) < 1e-9, "not unit")
            _require(s2.tilt_deg(c) <= cap + 0.05, f"tilt {s2.tilt_deg(c):.2f} > cap {cap}")
            if s2.tilt_deg(q) > cap:
                dh = (s2.heading_deg(c) - s2.heading_deg(q) + 180.0) % 360.0 - 180.0
                worst = max(worst, abs(dh))
                n += 1
            else:
                _require(_qdist_deg(c, q) < 1e-3, "under-cap rotation changed")
    _require(worst < 1.0, f"heading moved {worst:.2f} deg")
    plan = s2.build_plan({**s2.NOMINAL, "tilt_cap_deg": 6.0}, EE0, hw)
    _require(all(s2.tilt_deg(p[3:]) <= 6.05 for _, _, p in plan[:4]), "plan transit over cap")
    fr = [s2.NOMINAL[k] for k in ("slerp_f1", "slerp_f2", "slerp_f3")] + [1.0]
    yaw = [s2.heading_deg(p[3:]) for _, _, p in plan[:4]]
    _require(abs((yaw[3] - s2.heading_deg(q10) + 180) % 360 - 180) < 0.5, "w3 heading != P10")
    return (f"{n} capped cases, max heading change {worst:.3f} deg; transit headings "
            f"{np.round(yaw, 1).tolist()} at fractions {fr}")


@check("S4 knot noise and the loop path's trip return")
def check_paths() -> str:
    rng = np.random.default_rng(3)
    nz = s2.KnotNoise(rng, 2.0, 2.0, 1.0)
    _require(np.allclose(nz(0.0), 0) and np.allclose(nz(2.0), 0) and np.allclose(nz(5.0), 0),
             "noise not zero at the ends / after")
    _require(np.all(np.diff(nz.t) <= s2.KNOT_DT + 1e-9), "grid coarser than the knot dt")
    big = s2.KnotNoise(np.random.default_rng(4), 300.0, 1.0, 0.5)
    sd = big.v[1:-1].std(axis=0)
    _require(np.allclose(sd[:3], 1.0, rtol=0.1) and np.allclose(sd[3:], 0.5, rtol=0.1),
             f"knot sigma {sd.round(3)}")
    W = np.r_[0.45, 0.17, 0.05, 1.0, 0.0, 0.0, 0.0]
    D = ms.offset_pose(W, [5.0, 0, 0, 0, 0, 3.0])
    cyc = ms.CyclePath(W, D, 0.6, 0.6, 0.2)
    cmd = s2.Cmd(cyc, None, seg_of=lambda t: ms.SEGMENTS[cyc.segment(t)], home=W, ret_s=0.6,
                 reroute_until=cyc.t1)
    before = cmd.pose7(0.3)
    _require(cmd.segment(0.3) == "dev" and cmd.trip(0.3), "trip during dev not rerouted")
    _require(np.allclose(cmd.pose7(0.3), before), "command jumps at the trip")
    _require(np.allclose(cmd.pose7(0.9), W) and np.allclose(cmd.pose7(cyc.end), W),
             "not home after ret_s")
    _require(cmd.segment(0.5) == "return" and cmd.segment(1.0) == "hold", "segments after trip")
    _require(not cmd.trip(0.4), "second trip rerouted")
    late = s2.Cmd(cyc, None, home=W, ret_s=0.6, reroute_until=cyc.t1)
    _require(not late.trip(0.8), "trip while returning rerouted")
    walk = s2.Cmd(s2.walk_path(W, [D], 0.05), None)
    _require(not walk.trip(0.1), "walk command rerouted")
    _require(np.allclose(s2.command_at(cmd, 0.3)["knot0"], before), "command_at knot0")
    noisy = s2.Cmd(cyc, s2.KnotNoise(np.random.default_rng(5), cyc.end, 2.0, 1.0))
    _require(np.allclose(noisy.pose7(0.0), W) and np.allclose(noisy.pose7(cyc.end), W),
             "noisy loop does not start / end on the plan pose")
    ur = s2.Fixed(W)
    _require(np.allclose(ur.pose7(0.0), ur.pose7(3.0)), "UR path moves")
    return f"knot sigma {sd.round(2).tolist()}, trip return continuous and home at +ret_s"


def _sig(hand=9.0, h=0.5, wrap=110.0, n=40) -> dict:
    return {"hand_mm": hand, "large_h_mm": h, "large_wrap_deg": wrap, "small": {"n_near": n}}


@check("S5 trip logic on synthetic signal sequences")
def check_trips() -> str:
    ref = _sig()
    cases = [
        ("hover", _sig(), None), ("hover", _sig(hand=10.9), None),
        ("hover", _sig(hand=11.1), "hand_width"), ("hover", _sig(hand=7.9), "hand_width"),
        ("hover", _sig(h=3.4), None), ("hover", _sig(h=3.6), "large_h"),
        ("hover", _sig(h=-2.6), "large_h"), ("hover", _sig(h=math.nan), "large_h"),
        ("hover", _sig(wrap=96.0), None), ("hover", _sig(wrap=94.0), "large_wrap"),
        ("hover", _sig(n=5), None), ("wrap", _sig(n=19), "small_count"),
        ("arc", _sig(n=21), None), ("end", _sig(n=10), "small_count"),
        ("wrap", _sig(hand=12.0, n=1), "hand_width"),
    ]
    for phase, cur, want in cases:
        got = s2.trip_signal(ref, cur, phase)
        _require(got == want, f"{phase} {cur}: {got} != {want}")
    _require(s2.trip_signal(_sig(h=math.nan), _sig(h=9.0), "hover") is None, "NaN ref h trips")
    _require(s2.trip_signal(_sig(n=6), _sig(n=0), "wrap") is None, "thin small ref trips")
    seq = [_sig(), _sig(hand=9.5), _sig(hand=11.5), _sig(hand=10.0), _sig(hand=9.2)]
    first = next(i for i, x in enumerate(seq) if s2.trip_signal(seq[0], x, "end"))
    _require(first == 2 and s2.trip_signal(seq[0], seq[-1], "end") is None,
             "sequence: trip at 2, recovered at the end")
    return f"{len(cases)} cases + NaN / thin-reference guards + a trip-and-recover sequence"


@check("S6 budget: halve on trip, floor, persisted, seeds the next run")
def check_budget() -> str:
    with tempfile.TemporaryDirectory() as d:
        b = s2.Budget(Path(d) / "budget.json")
        _require(b.mult == {p: 1.0 for p in s2.BOXES}, "initial multipliers")
        _require(b.box("hover") == s2.BOXES["hover"], "box at 1")
        _require(b.halve("hover", "large_h", "e0") == 0.5, "first halving")
        _require(b.box("hover")["mm"] == 4.0 and b.box("hover")["away_mm"] == 4.0, "box scale")
        for _ in range(5):
            b.halve("hover", "hand_width", "e1")
        _require(b.mult["hover"] == s2.MIN_MULT and b.mult["end"] == 1.0, "floor / other phase")
        saved = json.loads((Path(d) / "budget.json").read_text())
        _require(saved["multipliers"] == b.mult and len(saved["history"]) == 6, "not persisted")
        b2 = s2.Budget(Path(d) / "b2.json", Path(d) / "budget.json")
        _require(b2.mult == b.mult and b2.seeded_from.endswith("budget.json"), "seed_from")
    return f"1 -> 0.5 -> floor {s2.MIN_MULT}; json persisted and reloaded"


@check("S7 loop offsets inside the (scaled) phase boxes")
def check_loops() -> str:
    rng = np.random.default_rng(7)
    away = np.array([0.0, 1.0, 0.0])
    n_single = 0
    for phase in s2.BOXES:
        for mult in (1.0, 0.5):
            box = {k: v * mult for k, v in s2.BOXES[phase].items()}
            for _ in range(500):
                o = s2.draw_loop(rng, box, 0.5, away)
                _require(np.all(np.abs(o[:2]) <= box["mm"] + 1e-9), f"{phase} xy {o[:2]}")
                _require(abs(o[2]) <= box["z_mm"] + 1e-9, f"{phase} z {o[2]}")
                _require(abs(o[5]) <= box["yaw_deg"] + 1e-9, f"{phase} yaw {o[5]}")
                _require(np.hypot(*o[3:5]) <= box["tilt_deg"] + 1e-9, f"{phase} tilt {o[3:5]}")
                if "away_mm" in box:
                    _require(o[:3] @ away <= box["away_mm"] + 1e-9, f"{phase} away {o[:3]}")
                n_single += int(np.count_nonzero(o) == 1)
    _require(1000 < n_single < 2000, f"single-axis draws {n_single} of 3000")
    _require(s2.LOOPS["wrap"][1] <= 1 and set(s2.LOOPS) == set(s2.BOXES) == set(s2.LOOP_WP),
             "loop phases")
    _require(not set(s2.LOOPS) & {"transit", "descent", "arc"}, "loops in a fragile phase")
    return f"3000 draws in the boxes, {n_single} single-axis, away cap held"


@check("S8 segment spans are contiguous and end with the post tail")
def check_segments() -> str:
    row = {"loops": [{"span": [10, 29], "phase": "hover", "waypoint": 3, "loop": 0,
                      "trip": None, "recovered": None},
                     {"span": [29, 48], "phase": "hover", "waypoint": 3, "loop": 1,
                      "trip": "large_h", "recovered": True},
                     {"span": [90, 109], "phase": "end", "waypoint": 10, "loop": 0,
                      "trip": None, "recovered": None}],
           "post_from": 120, "n_frames": 180}
    seg = s2.segments_of(row)
    spans = [x["span"] for x in seg]
    _require(spans[0][0] == 0 and spans[-1][1] == 180, "not covering the file")
    _require(all(a[1] == b[0] for a, b in itertools.pairwise(spans)), "gap")
    _require([x["kind"] for x in seg] == ["walk", "loop", "loop", "walk", "loop", "walk", "post"],
             f"kinds {[x['kind'] for x in seg]}")
    return f"{len(seg)} spans {spans}"


@check("S9 small-pulley cloud features of a large-frame ring")
def check_small_cloud() -> str:
    large7, small7 = bc.known_large_pulley_pose7(), bc.known_small_pulley_pose7()
    az = np.radians(np.arange(0.0, 120.0, 2.0))
    r = (SMALL_SEAT_MM + 3.3) * 1e-3  # tube surface outside the seat
    ring = np.c_[r * np.cos(az), r * np.sin(az), np.full(az.shape, 1.1e-3)]
    world = small7[:3] + ring @ s2.quat_to_mat3(small7[3:]).T
    seg = bc.to_pulley_frame(world, large7)
    f = s2.small_cloud(seg, large7)
    _require(f["n_near"] == len(az) and f["n_band"] == len(az), f"counts {f}")
    _require(100.0 <= f["wrap_deg"] <= 130.0, f"wrap {f['wrap_deg']:.1f}")
    _require(abs(f["h_mm"]) < 0.2 and abs(f["r_mm"] - SMALL_SEAT_MM - 3.3) < 0.2, f"h / r {f}")
    empty = s2.small_cloud(np.zeros((0, 3)), large7)
    _require(empty["n_near"] == 0 and empty["wrap_deg"] == 0.0 and math.isnan(empty["h_mm"]),
             "empty cloud")
    return (f"ring 0-118 deg: n {f['n_near']}, wrap {f['wrap_deg']:.0f} deg, h "
            f"{f['h_mm']:.2f} mm, r {f['r_mm']:.2f} mm; empty cloud ok")


def main() -> int:
    t0 = time.perf_counter()
    for name, fn in CHECKS:
        try:
            detail = fn()
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - report the crash as a failed check
            print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(f"[PASS] {name}: {detail}")
    print(f"ALL STAGE-2 COLLECTOR CHECKS PASSED ({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
