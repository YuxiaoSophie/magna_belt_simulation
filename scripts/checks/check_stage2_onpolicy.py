#!/usr/bin/env python3
"""CPU checks for the stage-2 on-policy collector (V12-99, ``scripts/lcs/stage2_onpolicy.py``):
hook install / uninstall leaves the harness untouched, region split, MPC / excursion trip truth
tables, frame merge + segment spans, the budget, the command-resolution and writer hooks, the
reset rule and start rows. No sim, no LCM.

Run:
    ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_stage2.yaml \\
      uv run --frozen python scripts/checks/check_stage2_onpolicy.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_learned_mpc as harness
import stage2_collect as s2
import stage2_onpolicy as op

from task_common import belt_cloud as bc


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(name: str):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _sig(hand=10.0, h=0.5, wrap=150.0, n_near=30) -> dict:
    return {"hand_mm": hand, "large_h_mm": h, "large_wrap_deg": wrap, "small": {"n_near": n_near}}


@check("C1 install / uninstall: harness functions restored by identity, defaults untouched")
def c1():
    before = {k: getattr(harness, k) for k in op.HOOKED}
    with tempfile.TemporaryDirectory() as d:
        op.install(op.create_parser().parse_args([]), op.Budget(Path(d) / "budget.json"))
        _require(all(getattr(harness, k) is not before[k] for k in op.HOOKED), "not hooked")
        op.uninstall()
    _require(all(getattr(harness, k) is before[k] for k in op.HOOKED), "not restored")
    a = harness.create_parser().parse_args([])
    _require(a.post_mpc is None and a.controller == "magna" and a.grasp_arms == "both",
             "harness defaults")
    _require(op.OP.ep is None, "episode state leaked")


@check("C2 region split over the small pulley (near: r < 45 mm and h < 12 mm)")
def c2():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for (r, h), want in {(22.5, -4.0): "near", (30.0, 11.0): "near", (30.0, 13.0): "far",
                         (50.0, 0.0): "far", (70.0, 30.0): "far"}.items():
        got = op.region_of(np.r_[s2.small_to_world(r, 100.0, h), q])
        _require(got == want, f"r {r} h {h}: {got} != {want}")
    v = op.away_dir(np.r_[s2.small_to_world(22.5, 100.0, 0.0), q])
    _require(abs(np.linalg.norm(v) - 1.0) < 1e-9 and v[2] == 0.0, "away dir not horizontal unit")
    w = np.r_[s2.small_to_world(22.5, 100.0, 0.0)] - bc.known_large_pulley_pose7()[:3]
    _require(float(v[:2] @ w[:2]) > 0.0, "away dir points to the large pulley")


@check("C3 MPC trip truth table (hand range / rate, large h / wrap / no band)")
def c3():
    t = op.MPC_TRIP
    cases = [((None, _sig(), 0), None),
             ((None, _sig(hand=t.hand_lo_mm - 0.1), 0), "hand_width"),
             ((None, _sig(hand=t.hand_hi_mm + 0.1), 0), "hand_width"),
             ((_sig(hand=10.0), _sig(hand=10.0 - t.hand_rate_mm - 0.1), 0), "hand_rate"),
             ((_sig(hand=10.0), _sig(hand=10.0 - t.hand_rate_mm + 0.1), 0), None),
             ((None, _sig(h=t.large_h_mm + 0.1), 0), "large_h"),
             ((None, _sig(h=-t.large_h_mm - 0.1), 0), "large_h"),
             ((None, _sig(wrap=t.large_wrap_min_deg - 1.0), 0), "large_wrap"),
             ((None, _sig(h=math.nan, wrap=0.0), 1), None),
             ((None, _sig(h=math.nan, wrap=0.0), t.large_none_frames), "large_none")]
    for (prev, cur, run), want in cases:
        got = op.mpc_trip(prev, cur, run)
        _require(got == want, f"{cur} prev {prev} run {run}: {got} != {want}")


@check("C4 excursion trip vs its start (hand delta, large h / wrap drop, small count near only)")
def c4():
    t, ref = op.EX_TRIP, _sig(n_near=40)
    cases = [(_sig(n_near=40), "near", None),
             (_sig(hand=10.0 + t.hand_lo_mm - 0.1, n_near=40), "far", "hand_width"),
             (_sig(hand=10.0 + t.hand_hi_mm + 0.1, n_near=40), "far", "hand_width"),
             (_sig(h=0.5 + t.large_h_mm + 0.1, n_near=40), "far", "large_h"),
             (_sig(wrap=150.0 - t.large_wrap_drop_deg - 1.0, n_near=40), "far", "large_wrap"),
             (_sig(wrap=150.0 - t.large_wrap_drop_deg + 1.0, n_near=40), "far", None),
             (_sig(n_near=19), "near", "small_count"),
             (_sig(n_near=19), "far", None)]
    for cur, region, want in cases:
        got = op.ex_trip(ref, cur, region)
        _require(got == want, f"{cur} {region}: {got} != {want}")
    _require(op.ex_trip(_sig(n_near=5), _sig(n_near=0), "near") is None, "thin ref compared")


@check("C5 frame merge: excursion frames before their loop frame, spans")
def c5():
    loop = [{"i": i} for i in range(5)]
    r0, r1 = {}, {}
    ex0, ex1 = [{"e": 0, "j": j} for j in range(3)], [{"e": 1, "j": j} for j in range(2)]
    out = op.merge_frames(loop, [(1, ex0, r0), (4, ex1, r1)])
    tags = [(f.get("i"), f.get("e")) for f in out]
    want = ([(0, None)] + [(None, 0)] * 3 + [(1, None), (2, None), (3, None)] + [(None, 1)] * 2
            + [(4, None)])
    _require(tags == want, f"order {tags}")
    _require(r0["span"] == [1, 4] and r1["span"] == [7, 9], f"spans {r0} {r1}")
    r2 = {}
    out = op.merge_frames(loop[:2], [(2, ex1, r2)])
    _require(len(out) == 4 and r2["span"] == [2, 4], "trailing excursion")
    _require(op.merge_frames(loop, []) == loop, "no excursions")


@check("C6 segment labels and spans")
def c6():
    frames = [{}, {"onpolicy_seg": "dev", "onpolicy_ex": 0}, {"onpolicy_seg": "hold",
                                                               "onpolicy_ex": 0},
              {}, {"onpolicy_seg": "dev", "onpolicy_ex": 1}, {}, {"post_phase": "release"}]
    seg = [op.segment_of(f) for f in frames]
    _require(seg == ["mpc", "ex:dev", "ex:hold", "mpc", "ex:dev", "mpc", "post"], f"{seg}")
    _require(all(s in op.SEGMENTS for s in seg), "label set")
    spans = op.segments_of(seg, [f.get("onpolicy_ex", -1) for f in frames])
    want = [{"kind": "mpc", "span": [0, 1]}, {"kind": "excursion", "span": [1, 3], "excursion": 0},
            {"kind": "mpc", "span": [3, 4]}, {"kind": "excursion", "span": [4, 5], "excursion": 1},
            {"kind": "mpc", "span": [5, 6]}, {"kind": "post", "span": [6, 7]}]
    _require(spans == want, f"{spans}")
    spans = op.segments_of(["ex:dev", "ex:dev"], [0, 1])
    _require(len(spans) == 2, "adjacent excursions merged")


@check("C7 budget: halve per region, floor, persisted, seeded")
def c7():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "budget.json"
        b = op.Budget(p)
        _require(b.box("near")["mm"] == op.REGIONS["near"]["mm"], "box at 1")
        for _ in range(5):
            b.halve("near", "hand_width", "ep")
        _require(b.mult == {"far": 1.0, "near": op.MIN_MULT}, f"{b.mult}")
        _require(abs(b.box("near")["mm"] - op.REGIONS["near"]["mm"] * op.MIN_MULT) < 1e-12, "box")
        doc = json.loads(p.read_text())
        _require(doc["multipliers"] == b.mult and len(doc["history"]) == 5, "persisted")
        b2 = op.Budget(Path(d) / "b2.json", p)
        _require(b2.mult == b.mult and b2.seeded_from == str(p), "seeded")


@check("C8 command resolution skips excursion frames")
def c8():
    seen = []
    op.OP.orig = {"_resolve_commands": lambda fr, b, dt: seen.append(list(fr))}
    try:
        frames = [{"a": 0}, {"onpolicy_seg": "dev"}, {"a": 2}]
        op._resolve_commands(frames, None, 0.005)
    finally:
        op.OP.orig = {}
    _require(len(seen) == 1 and seen[0] == [frames[0], frames[2]], f"{seen}")


@check("C9 writer hook: arrays + meta added, harness meta kept")
def c9():
    class W:
        def write(self, path, label, meta, arrays=None, omit=()):
            self.got = (path, label, meta, arrays, omit)

    w = W()
    op.OP.orig = {"build_writer": lambda frames, pcd, definition: (w, np.zeros(1))}
    op.OP.cfg = op.create_parser().parse_args([])
    op.OP.ep = op.Ep("episode_x_0", np.random.default_rng(0))
    sig = {"small": {"n_near": 7}}
    frames = [{"onpolicy_sig": sig}, {"onpolicy_sig": sig, "onpolicy_seg": "dev", "onpolicy_ex": 0,
                                      "onpolicy_trip": "large_h"},
              {"onpolicy_sig": sig, "post_phase": "out"}]
    try:
        writer, _ = op._build_writer(frames, True, "cmd_delta")
        writer.write("p.npz", "engaged", {"cmd_source_codes": {"response": 0}, "k": 1},
                     arrays={"sim_x": 1}, omit=("pcd",))
    finally:
        op.OP.orig, op.OP.ep = {}, None
    path, label, meta, arrays, omit = w.got
    _require(path == "p.npz" and label == "engaged" and omit == ("pcd",), "args")
    _require(meta["k"] == 1 and meta["cmd_source_codes"] == {
        "response": 0, "onpolicy_excursion": op.CMD_EXCURSION}, f"{meta['cmd_source_codes']}")
    _require(meta["onpolicy"]["n_excursions"] == 0 and "config" in meta["onpolicy"], "meta")
    _require(arrays["sim_x"] == 1, "harness arrays kept")
    _require(arrays["sim_onpolicy_segment"].tolist() == ["mpc", "ex:dev", "post"], "segment")
    _require(arrays["sim_onpolicy_excursion"].tolist() == [-1, 0, -1], "excursion id")
    _require(arrays["sim_onpolicy_trip"].tolist() == ["", "large_h", ""], "trip")
    _require(arrays["sim_cloud_small_n_near"].tolist() == [7, 7, 7], "small count")


@check("C10 reset rule and start rows")
def c10():
    def row(large, held):
        return {"post_mpc": {"final": {"large": large}, "mpc_end": {"held": held}}}
    _require(not op.extra_reset(row("engaged", True)), "engaged + held")
    _require(op.extra_reset(row("outside", True)) and op.extra_reset(row("engaged", False)),
             "off / dropped")
    st = op.start_row({"id": "V48b_set3_gv_02_0",
                       "file": str(REPO_ROOT / "data/lcs/start_states/stage2/x_osc.npz")})
    _require(st["key"] == "set3_gv_02" and st["arm"] == "V48b"
             and st["file"] == "data/lcs/start_states/stage2/x_osc.npz", f"{st}")
    _require(op.start_row({"id": "V45c_up+20_1", "file": "/x"})["key"] == "up+20", "up+20")


@check("C11 resume hold: holds until a plan answering the latent arrives, then hands back")
def c11():
    class Tr:
        def __init__(self, u):
            self.utime = u

    class Sim:
        def __init__(self):
            self.bridge = argparse.Namespace(franka_trajs=[Tr(100), Tr(100), Tr(200)])
            self.commander_hook = "x"

        def make_hold_hook(self):
            return lambda step, t, q, b: "hold"

    sim = Sim()
    _require(op._answers(sim.bridge, 200) == 1 and op._answers(sim.bridge, 100) == 3
             and op._answers(sim.bridge, 300) == 0, "answer count")
    hook = op.hold_until_answer(sim, 300)
    sim.commander_hook = hook
    _require(hook(0, 0.0, None, None) == "hold" and sim.commander_hook is hook, "holds")
    sim.bridge.franka_trajs.append(Tr(150))  # a stale republish does not release the hold
    _require(hook(1, 0.0, None, None) == "hold", "stale plan released the hold")
    sim.bridge.franka_trajs.append(Tr(300))
    _require(hook(2, 0.0, None, None) is None and sim.commander_hook is None, "hand back")


@check("C12 excursion speed gate (measured hand motion over the last frame)")
def c12():
    a = np.array([0.4, 0.1, 0.05, 1.0, 0.0, 0.0, 0.0])
    b = a.copy()
    b[0] += 0.0029
    _require(op.slow_enough(a, b, 3.0) and not op.slow_enough(a, b, 2.8), "threshold")
    _require(not op.slow_enough(None, b, 3.0), "no previous frame")


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:  # noqa: BLE001 - report every check
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print("ALL PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
