#!/usr/bin/env python3
"""CPU checks for the Stage-1 phase-2 plumbing (V12-90): ``held_for``, the known pulley poses,
the ``human_label`` small / both rules and the parametrised phase-label marks.
``eval_learned_mpc``'s stream label and the miles_seed ``ctx`` guard on ``_sample``. No sim, no LCM.

Run:
    uv run --frozen python scripts/checks/check_phase2_plumbing.py
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import collect_lcs_dataset as cli
import eval_learned_mpc as elm
from demo_prefix import demo_marks
from human_label import _xyzw, emulated_human_label
from latent_stage_tols import latch_frame
from latent_waypoints import first_hold_frame
from realfirst_loop import goal_frames

from round_belt_task.offline_simulation import held_for
from round_belt_task.outcome import LARGE_SEAT_MM, SMALL_SEAT_MM
from task_common.belt_cloud import known_large_pulley_pose7, known_small_pulley_pose7
from task_common.replay_metrics import quat_rotate

SMALL_XYZ = np.array([0.4517, 0.1594, 0.0349])
LARGE_XYZ = np.array([0.4499, -0.0556, 0.0346])
N_BODIES, T_FRAMES, LAST_N = 48, 5, 3
PHASE1 = ["hold:pre_place_1", "move:pre_place_2", "move:pre_place_2", "hold:place_3", "done"]


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _raises(exc: type, fn, *args, **kw) -> bool:
    try:
        fn(*args, **kw)
    except exc:
        return True
    return False


def _circle(pose7_wxyz: np.ndarray, seat_mm: float) -> np.ndarray:
    """48 bodies on the seat circle (h = 0) of a pulley given as xyz_wxyz."""
    a = np.linspace(0.0, 2.0 * np.pi, N_BODIES, endpoint=False)
    local = np.stack([seat_mm * 1e-3 * np.cos(a), seat_mm * 1e-3 * np.sin(a), 0.0 * a], axis=1)
    q = np.broadcast_to(np.r_[pose7_wxyz[4:7], pose7_wxyz[3]], (N_BODIES, 4))
    return pose7_wxyz[:3] + quat_rotate(q, local)


def _mapping(*loops: np.ndarray, small_pose: bool = False) -> dict:
    belt = np.tile(np.concatenate(loops), (T_FRAMES, 1, 1))
    m = {"sim_belt_xyz": belt,
         "sim_pulley_large_pose": np.tile(known_large_pulley_pose7(), (T_FRAMES, 1))}
    if small_pose:
        m["sim_pulley_small_pose"] = np.tile(known_small_pulley_pose7(), (T_FRAMES, 1))
    return m


@check("P0 held_for truth table")
def check_p0(ctx: SimpleNamespace) -> str:
    for held in ((True, True), (True, False), (False, True), (False, False)):
        _require(held_for(held, "both") is all(held), f"both {held}")
        _require(held_for(held, "franka") is held[0], f"franka {held}")
        _require(held_for(held) is all(held), f"default arms {held}")
    _require(_raises(ValueError, held_for, (True, True), "ur"), "'ur' did not raise")
    return "both = all, franka = held[0], 'ur' raises"


@check("P1 known pulley poses")
def check_p1(ctx: SimpleNamespace) -> str:
    s, lg = known_small_pulley_pose7(), known_large_pulley_pose7()
    _require(np.abs(s[:3] - SMALL_XYZ).max() < 1e-3, f"small centre {s[:3]}")
    _require(np.abs(lg[:3] - LARGE_XYZ).max() < 1e-3, f"large centre {lg[:3]}")
    _require(np.allclose(s[3:], lg[3:], atol=1e-9), f"quats differ {s[3:]} {lg[3:]}")
    dy = s[1] - lg[1]
    _require(abs(dy - 0.215) < 1e-3, f"y offset {dy * 1e3:.2f} mm != 215")
    return f"small {np.round(s[:3], 4)}, large {np.round(lg[:3], 4)}, dy {dy * 1e3:.1f} mm"


@check("P2 emulated_human_label: large loop only")
def check_p2(ctx: SimpleNamespace) -> str:
    m = _mapping(_circle(known_large_pulley_pose7(), LARGE_SEAT_MM))
    got = {r: emulated_human_label(m, r, LAST_N)["success"] for r in
           ("engaged", "strict", "small", "both")}
    _require(got == {"engaged": True, "strict": True, "small": False, "both": False}, str(got))
    return str(got)


@check("P3 emulated_human_label: both loops")
def check_p3(ctx: SimpleNamespace) -> str:
    loops = (_circle(known_large_pulley_pose7(), LARGE_SEAT_MM),
             _circle(known_small_pulley_pose7(), SMALL_SEAT_MM))
    for with_pose in (False, True):  # known-pose fallback vs recorded small pose
        m = _mapping(*loops, small_pose=with_pose)
        got = {r: emulated_human_label(m, r, LAST_N)["success"] for r in
               ("engaged", "small", "both")}
        _require(all(got.values()), f"small_pose={with_pose}: {got}")
    return "engaged / small / both all True, with and without sim_pulley_small_pose"


@check("P4 emulated_human_label: small loop only")
def check_p4(ctx: SimpleNamespace) -> str:
    m = _mapping(_circle(known_small_pulley_pose7(), SMALL_SEAT_MM))
    got = {r: emulated_human_label(m, r, LAST_N)["success"] for r in
           ("engaged", "small", "both")}
    # both needs the large pulley engaged too (label = large's when only small is)
    _require(got == {"engaged": False, "small": True, "both": False}, str(got))
    return str(got)


@check("P5 phase-label marks")
def check_p5(ctx: SimpleNamespace) -> str:
    _require(latch_frame(["a", "move:x", "move:x", "b"], after="move:x") == 3, "latch after")
    _require(first_hold_frame(["a", "hold:y", "hold:y"], label="hold:y") == 1, "first_hold")
    want = {"move0": 0, "latch": 2, "first_hold": 2}
    got = demo_marks(["move:x", "move:x", "hold:y"], latch_after="move:x", first_hold="hold:y")
    _require(got == want, f"demo_marks {got} != {want}")
    _require(latch_frame(PHASE1) == 3, "default latch")
    _require(first_hold_frame(PHASE1) == 3, "default first_hold")
    _require(demo_marks(PHASE1) == {"move0": 1, "latch": 3, "first_hold": 3}, "default marks")
    with tempfile.TemporaryDirectory() as tmp:
        ep = Path(tmp) / "ep.npz"
        names = sorted(set(PHASE1))
        np.savez(ep, sim_meta=json.dumps({"phase_labels": names}),
                 sim_phase=np.array([names.index(lab) for lab in PHASE1]))
        gf = goal_frames(ep)
    _require(gf["frames"] == [3, 4] and gf["labels"] == ["hold:place_3", "done"], str(gf))
    for fn, args in ((latch_frame, (PHASE1, "move:place_9")),
                     (first_hold_frame, (PHASE1, "hold:place_10")),
                     (latch_frame, (["move:x"], "move:x"))):  # nothing after the last mark
        _require(_raises(ValueError, fn, *args), f"{fn.__name__}{args[1:]} did not raise")
    _require(_raises(ValueError, demo_marks, PHASE1, "move:place_9"), "demo_marks missing")
    return "synthetic + phase-1 defaults (latch 3, first_hold 3, final 4) + missing labels"


# miles_seed.py builds its _sample ctx as an argparse.Namespace with exactly these keys.
MILES_CTX_KEYS = {"args", "mode", "encoder", "sim", "board", "gripper", "thresholds", "tangent",
                  "cloud_rng", "cloud", "plate_top_z"}


@check("P6 _stream_label / _pulley_labels table")
def check_p6(ctx: SimpleNamespace) -> str:
    both = _circle(known_large_pulley_pose7(), LARGE_SEAT_MM), _circle(
        known_small_pulley_pose7(), SMALL_SEAT_MM)
    far = np.full((N_BODIES, 3), 5.0)
    lg7, sm7 = _xyzw(known_large_pulley_pose7()), _xyzw(known_small_pulley_pose7())
    cases = {  # (large engaged, small engaged) -> belt
        (True, True): np.concatenate(both), (True, False): both[0],
        (False, True): both[1], (False, False): far}
    th = cli.parse_thresholds([])
    for mode in ("large", "small", "both"):
        c = SimpleNamespace(thresholds=th, small_thresholds=th, outcome_pulley=mode)
        for (le, se), belt in cases.items():
            lab_l, lab_s, ok = elm._pulley_labels(c, belt, lg7, sm7)
            _require(ok is (le and se), f"{mode} {le, se}: success_both {ok}")
            _require((lab_l == "engaged") is le and (lab_s == "engaged") is se,
                     f"{mode} {le, se}: labels {lab_l}, {lab_s}")
            got = elm._stream_label(c, lab_l, lab_s, ok)
            if mode == "small":
                _require(got == lab_s, f"small {le, se}: {got} != {lab_s}")
                _require((got == "engaged") is se, f"small {le, se}: {got}")
            if mode == "both":
                _require((got == "engaged") is ok, f"both {le, se}: engaged != success_both")
                if se and not le:  # accepted P3 deviation: the large pulley's label
                    _require(got == lab_l, f"both small-only: {got} != large label {lab_l}")
    return "4 cases x large/small/both; both engaged iff both, small-only -> large label"


def _ctx_reads(tree: ast.Module, name: str, seen: set[str]) -> dict[str, set[str]]:
    """``ctx.<attr>`` reads (not via getattr) of ``name`` and of module functions it passes ctx to."""
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    seen.add(name)
    out: dict[str, set[str]] = {}
    for node in ast.walk(fns[name]):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "ctx"):
            out.setdefault(node.attr, set()).add(name)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            callee = node.func.id
            passes = any(isinstance(a, ast.Name) and a.id == "ctx" for a in node.args)
            if passes and callee in fns and callee not in seen:
                for k, v in _ctx_reads(tree, callee, seen).items():
                    out.setdefault(k, set()).update(v)
    return out


@check("P7 _sample ctx reads satisfied by miles_seed's Namespace")
def check_p7(ctx: SimpleNamespace) -> str:
    tree = ast.parse((REPO_ROOT / "scripts" / "lcs" / "eval_learned_mpc.py").read_text())
    reads = _ctx_reads(tree, "_sample", set())
    missing = {k: sorted(v) for k, v in reads.items() if k not in MILES_CTX_KEYS}
    _require(not missing, f"ctx attrs read without getattr but absent in miles_seed: {missing}")
    return f"direct ctx reads {sorted(reads)} all in the miles_seed key set"


def main() -> int:
    t0 = time.perf_counter()
    ctx = SimpleNamespace()
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
    print(f"ALL PHASE-2 PLUMBING CHECKS PASSED ({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
