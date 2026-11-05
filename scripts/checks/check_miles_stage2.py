#!/usr/bin/env python3
"""Checks for V12-125: ``--stage2-start-snapshot`` / ``--log-plan-age`` are opt-in (parser, defaults,
CHAIN_ARGS and chain index blocks identical to ``--pre-dir`` without it), stage-1 MILES / the
scripted walk / the suite writer byte-identical, the branch path and peel rule, and the layout of a
``miles_stage2.py`` run (``--run``; SKIP if absent), plan-age units (bridge receipt record, per-frame
age, absent when off). CPU only, no sim.

Run:
    uv run --frozen python scripts/checks/check_miles_stage2.py --pre-dir DIR [--run DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs",
           REPO_ROOT / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import check_full_run as cfr
import eval_learned_mpc as harness
import miles_stage2 as m2

NEW = {**harness.S2SNAP_ARG_DEFAULTS, **harness.PLANAGE_ARG_DEFAULTS,
       **harness.SCRIPTED_UR_ARG_DEFAULTS,  # + V12-130's later flag
       **harness.GRIP_ARG_DEFAULTS,  # + V12-132's
       "publish_body_poses": False, "realtime": False}  # + lcm-viewer's, demo GUI's
BASE = [*cfr.MPPI, *cfr.FULL_PP]
ARGV = {"plain": cfr.MPPI, "stage-1 suite line": BASE, "node chain": [*BASE, "--belt-seg", *cfr.CHAIN],
        "scripted chain": [*BASE, "--stage2-chain", "--stage2-scripted"],
        "teleop chain": [*BASE, "--stage2-chain", "--stage2-teleop"]}
# stage2_scripted.py: V12-130 hunks, identity at default in check_stage2_scripted.py C7
SAME = ("miles_seed.py", "run_hard_suite.py")
BRIDGE = REPO_ROOT / "src/round_belt_task/controller_bridge.py"
PRE_DIR: Path | None = None
RUN: Path | None = None
CHECKS: list[tuple[str, callable]] = []
_require = cfr._require


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _pre():
    spec = importlib.util.spec_from_file_location("elm_pre_m2", PRE_DIR / "eval_learned_mpc.py")
    pre = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pre
    spec.loader.exec_module(pre)
    return pre


def _block(mod, argv):
    a = mod.create_parser().parse_args(argv)
    ch = mod._chain_options(a, mod._mppi_options(a, False))
    if ch is None:
        return None
    blk = mod._chain_index_block(a, ch)
    blk.get("stage2", {}).pop("node", None)
    return json.dumps(blk, sort_keys=True, default=str)


@check("M1 flag registered, default off, refused without --stage2-chain, in the index block only when on")
def check_flag() -> str:
    d = vars(harness.create_parser().parse_args(cfr.MPPI))
    for k, v in harness.S2SNAP_ARG_DEFAULTS.items():
        _require(k in harness.STAGE2_ARG_DEFAULTS and k in harness.CHAIN_ARGS and d[k] == v, k)
    a = harness.create_parser().parse_args([*BASE, "--stage2-start-snapshot"])
    try:
        harness._chain_options(a, harness._mppi_options(a, False))
    except ValueError as exc:
        _require("only with --stage2-chain" in str(exc), str(exc))
    else:
        raise AssertionError("accepted without --stage2-chain")
    on = json.loads(_block(harness, [*ARGV["scripted chain"], "--stage2-start-snapshot"]))
    off = json.loads(_block(harness, ARGV["scripted chain"]))
    _require(on["flags"].get("stage2_start_snapshot") is True and "stage2_start_snapshot" not in off["flags"],
             "index block flags")
    return "default False; refused off-chain; index block key only when on"


@check("M2 --pre-dir identity: parser (5 argv sets), defaults, CHAIN_ARGS, chain index blocks")
def check_pre() -> str:
    if PRE_DIR is None:
        return "SKIP: no --pre-dir"
    pre = _pre()

    def norm(d: dict) -> dict:
        return {k: REPO_ROOT / v.relative_to(pre.REPO_ROOT) if isinstance(v, Path)
                and v.is_relative_to(pre.REPO_ROOT) else v for k, v in d.items()}

    for what, argv in ARGV.items():
        a = norm(vars(pre.create_parser().parse_args(argv)))
        b = {k: v for k, v in vars(harness.create_parser().parse_args(argv)).items() if k not in NEW}
        _require(a == b, f"{what}: {sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))}")
        _require(_block(pre, argv) == _block(harness, argv), f"{what}: chain index block differs")
    _require(pre.STAGE2_ARG_DEFAULTS.keys() <= harness.STAGE2_ARG_DEFAULTS.keys()
             and not hasattr(pre, "PLANAGE_ARG_DEFAULTS"), "pre has the new keys")
    _require({k: v for k, v in harness.STAGE2_ARG_DEFAULTS.items() if k not in NEW}
             == pre.STAGE2_ARG_DEFAULTS, "STAGE2_ARG_DEFAULTS minus the new key")
    _require([k for k in harness.CHAIN_ARGS if k not in NEW] == pre.CHAIN_ARGS, "CHAIN_ARGS")
    return f"{len(ARGV)} argv sets + index blocks identical; defaults / CHAIN_ARGS identical minus {list(NEW)}"


@check("M3 stage-1 MILES, scripted walk, suite writer byte-identical; bridge additions only")
def check_same() -> str:
    if PRE_DIR is None:
        return "SKIP: no --pre-dir"
    for f in SAME:
        a, b = (hashlib.sha256(p.read_bytes()).hexdigest() for p in (PRE_DIR / f, REPO_ROOT / "scripts/lcs" / f))
        _require(a == b, f"{f} changed")
    import difflib

    old, new = (p.read_text().splitlines() for p in (PRE_DIR / BRIDGE.name, BRIDGE))
    gone = [ln for ln in difflib.ndiff(old, new) if ln.startswith("- ")]
    _require(not gone, f"controller_bridge.py lines removed/changed: {gone[:3]}")
    return ", ".join(SAME) + f"; {BRIDGE.name} additions only ({len(new) - len(old)} lines)"


@check("M4 branch path (offset 0 -> D -> 0 on the moving demo) and the large_off / peel-stop rule")
def check_units() -> str:
    import demo_prefix as dp

    knots = np.tile([0.4, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], (40, 1))
    knots[:, 0] += 0.001 * np.arange(40)
    base = dp.KnotPath(knots, dp.FRAME_DT_S)
    off = np.array([10.0, 0, 0, 0, 0, 0])
    p = m2.Shifted(base, 0.3, off, 0.6, 0.6)
    _require(np.allclose(p.pose7(0.0), base.pose7(0.3)), "tau 0 != base")
    _require(np.allclose(p.pose7(0.6)[:3], base.pose7(0.9)[:3] + [0.01, 0, 0]), "tau dev_s != base + D")
    _require(np.allclose(p.pose7(1.5), base.pose7(1.8)), "after the return != base")
    _require([p.segment(t) for t in (0.1, 0.7, 1.3)] == [1, 2, 3], "segments")
    _require(np.allclose(m2.Shifted(base, 0.3).pose7(0.5), base.pose7(0.8)), "walk shift")
    v109 = importlib.util.spec_from_file_location("v109", REPO_ROOT / "data/lcs/realfirst/cmp/v12_109/make_tier.py")
    if Path(v109.origin).is_file():
        mod = importlib.util.module_from_spec(v109)
        v109.loader.exec_module(mod)
        _require((mod.OFF_WRAP_DEG, mod.MARGIN) == (m2.OFF_WRAP_DEG, m2.OFF_MARGIN), "v109 constants")
    w = m2.Watch(type("R", (), {"frames": []})(), 3)
    w.wrap = [200.0] * 5 + [10.0] * 3 + [200.0] + [10.0] * 9
    _require(w.large_off() == 9 + 1 + m2.OFF_MARGIN, f"large_off {w.large_off()}")
    w.wrap = [200.0] * 5 + [10.0] * 7
    _require(w.large_off() is None, "7 off frames counted as off")
    return "offset path, segments, large_off == v12_109 rule"


@check("M5 recorded miles_stage2 run: files, cmd_source 4, grid, peel stop, buffer rows")
def check_run() -> str:
    if RUN is None or not (RUN / "index.json").is_file():
        return f"SKIP: no run at {RUN}"
    from task_common import lcs_dataset as lcs

    idx = json.loads((RUN / "index.json").read_text())
    buf = json.loads((RUN / "buffer.json").read_text())["rollouts"]
    files = sorted((RUN / "miles").glob("*.npz"))
    _require(len(buf) == len(files) and all(Path(r["episode"]).is_file() for r in buf), "buffer rows vs files")
    peel_tail = round(idx["args"]["peel_tail_s"] / lcs.SAMPLE_PERIOD_S)
    n = 0
    for f in files[:: max(1, len(files) // 40)]:
        z = np.load(f, allow_pickle=True)
        _require((z["sim_cmd_source"] == 4).all(), f"{f.name}: cmd_source")
        _require((np.diff(z["sim_step"]) == lcs.SAMPLE_STEPS).all(), f"{f.name}: step grid")
        lcs.validate_episode(f, period_us=round(lcs.SAMPLE_STEPS * lcs.SIM_DT_S * 1e6))
        n += 1
    for b in idx["branches"]:
        if b["large_off"] is not None and b["stopped"] == "peel":
            _require(b["frames"] == b["large_off"] + peel_tail, f"{b['file']}: peel stop at {b['frames']}")
    out = {}
    for b in idx["branches"]:
        out[b["outcome"]] = out.get(b["outcome"], 0) + 1
    return f"{len(files)} files ({n} validated), {len(idx['branches'])} branches {out}"


@check("M6 --log-plan-age: bridge records the first receipt only when on; per-frame age; off = no key")
def check_plan_age() -> str:
    from types import SimpleNamespace

    from round_belt_task.controller_bridge import ControllerBridge

    step = harness.UTIME_PER_STEP
    b = object.__new__(ControllerBridge)
    b.franka_trajs, b.plan_recv_utime, b.last_sent_utime = [], None, 10 * step
    knots = (np.zeros((7, 3)), np.tile([1.0, 0, 0, 0], (7, 1)), 0.075 * np.arange(7))
    b._decode_traj = lambda data, ch: (SimpleNamespace(utime=4 * step), *knots)
    b._on_franka_traj("c", b"")
    _require(len(b.franka_trajs) == 1 and b.plan_recv_utime is None, "off: recorded")
    b.plan_recv_utime = {}
    b._on_franka_traj("c", b"")
    b.last_sent_utime = 12 * step  # a republish of the same plan keeps the first receipt
    b._on_franka_traj("c", b"")
    _require(b.plan_recv_utime == {4 * step: 10 * step}, f"receipts {b.plan_recv_utime}")
    b.ur_lines = []
    t = b.franka_trajs[0].times[0]

    def frames():
        return [{"time": t, "wall": 0.0, "ee_f": np.zeros(7), "ee_u": np.zeros(7)}]

    on, b.plan_recv_utime = frames(), {4 * step: 10 * step}
    harness._resolve_commands(on, b, 0.0025)
    _require(abs(on[0]["plan_age_s"] - 7 * step * 1e-6) < 1e-12, f"age {on[0].get('plan_age_s')}")
    off, b.plan_recv_utime = frames(), None
    harness._resolve_commands(off, b, 0.0025)
    _require("plan_age_s" not in off[0], "off: plan_age_s set")
    d = vars(harness.create_parser().parse_args(cfr.MPPI))
    _require(d["log_plan_age"] is False, "default")
    return f"receipt kept once, age = (receipt state + 1 step - plan utime) = {on[0]['plan_age_s'] * 1e3:.3f} ms; off: none"


def main() -> int:
    global PRE_DIR, RUN
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="dir with the pre-change eval_learned_mpc.py, miles_seed.py, stage2_scripted.py, "
                         "run_hard_suite.py")
    ap.add_argument("--run", type=Path, default=None, help="a miles_stage2.py output dir (M5)")
    a = ap.parse_args()
    PRE_DIR, RUN = a.pre_dir, a.run
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
        print(f"[{'SKIP' if detail.startswith('SKIP') else 'PASS'}] {name}: {detail}", flush=True)
    print(f"ALL MILES-STAGE2 CHECKS PASSED ({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
