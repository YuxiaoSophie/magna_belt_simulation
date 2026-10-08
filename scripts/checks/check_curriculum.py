#!/usr/bin/env python3
"""Check of the V12 reverse-curriculum state machine in ``realfirst_loop.py`` (no sim, no GPU).

R1 advance (>= advance_when successes -> next lower frac); R2 partial success stays and resets
the zero streak; R3 retreat after retreat_after consecutive 0/n (one 0/n only counts); R4 clamp
at the lowest frac; R5 clamp at the highest; R6 collection pool switch at frac 0.0 (rule, the
demo_starts expansion to eval-disjoint set2 starts, the harness --prefix-* flags); R7
``mppi_stages`` auto / fixed; R8 the iteration chain through ``iter<i>/curriculum.json`` +
``labels.json`` on a scratch root (and the config validation). ``--loop`` checks another copy
of the loop module.

Run:
    uv run --frozen python scripts/checks/check_curriculum.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LADDER = [2.0, 1.5, 1.0, 0.5, 0.0]
DEMOS = {f"set2/gv_0{k}": f"data/lcs/realfirst/v1_1/{d}/demo/episode_000{e}.npz"
         for k, d, e in ((0, "seed", 0), (1, "seed", 2), (2, "seed_ext", 4))}
RULE = {"while_frac_gt": 0.0, "pool": ["demo_starts"],
        "then_pool": ["grasp_variants/set2", "grasp_variants/set3"]}


def load_loop(path: Path):
    spec = importlib.util.spec_from_file_location("realfirst_loop_under_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step(rl, frac, s, n=5, z=0):
    return rl.ladder_step(LADDER, frac, s, n, z, 3, 2)


def expect(got, want, what):
    if got != want:
        raise AssertionError(f"{what}: {got} != {want}")


def r1(rl):
    expect(step(rl, 1.0, 3), (0.5, 0, "advance"), "1.0 at 3/5")
    expect(step(rl, 2.0, 5), (1.5, 0, "advance"), "2.0 at 5/5")
    return "1.0 -> 0.5 at 3/5, 2.0 -> 1.5 at 5/5"


def r2(rl):
    expect(step(rl, 1.0, 2, z=1), (1.0, 0, "stay"), "1.0 at 2/5 after one 0/5")
    expect(step(rl, 0.5, 0, n=0, z=1), (0.5, 1, "stay (no labelled rollout)"), "0 labelled")
    return "2/5 stays and clears the streak; 0 labelled rollouts keep frac and streak"


def r3(rl):
    expect(step(rl, 0.5, 0, z=0), (0.5, 1, "stay (1 x 0/n)"), "first 0/5")
    expect(step(rl, 0.5, 0, z=1), (1.0, 0, "retreat"), "second 0/5")
    return "0.5: first 0/5 stays (streak 1), second retreats to 1.0"


def r4(rl):
    expect(step(rl, 0.0, 5), (0.0, 0, "clamp (lowest)"), "0.0 at 5/5")
    return "0.0 at 5/5 stays 0.0"


def r5(rl):
    expect(step(rl, 2.0, 0, z=1), (2.0, 0, "clamp (highest)"), "2.0 after two 0/5")
    return "2.0 after two 0/5 stays 2.0"


def ctx_for(rl, root: Path, over: dict, i: int = 0):
    cfg = root / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(over))
    a = argparse.Namespace(smoke=False, root=str(root), iter=i, config=str(cfg), dry_run=True,
                           collect_port=None, eval_port=None, force=None, ignore_stop=False)
    return rl.Ctx(a)


def r6(rl):
    cfg = {**rl.DEFAULTS, "collection_pool_rule": RULE}
    expect(rl.active_pool(cfg, 1.0), ["demo_starts"], "frac 1.0")
    expect(rl.active_pool(cfg, 0.5), ["demo_starts"], "frac 0.5")
    expect(rl.active_pool(cfg, 0.0), RULE["then_pool"], "frac 0.0")
    expect(rl.active_pool(rl.DEFAULTS, 1.0), rl.DEFAULTS["collection_pool"], "no rule")
    with tempfile.TemporaryDirectory() as td:
        ctx = ctx_for(rl, Path(td), {"prefix": {"demos": DEMOS}, "collection_pool_rule": RULE})
        demo = rl.pool_starts(ctx, ["demo_starts"])
        keys = [c["key"] for c in demo]
        expect(keys, [f"grasp_variants/set2/gv_0{k}" for k in range(3)], "demo_starts")
        wide = rl.pool_starts(ctx, RULE["then_pool"])
        if not set(keys) <= {c["key"] for c in wide} or len(wide) <= len(keys):
            raise AssertionError("then_pool must widen beyond the demo starts")
        d0 = rl.draw_start(ctx, 1.0)
        expect(d0["pool"], ["demo_starts"], "draw at 1.0")
        args = rl.prefix_args(ctx, d0["start"], 1.0)
        k = d0["start"]["key"].split("/", 1)[1]
        expect(args, ["--prefix-demos", json.dumps({k: DEMOS[k]}), "--prefix-frac", "1"],
               "prefix args")
        expect(rl.prefix_args(ctx, d0["start"], 0.0), [], "no prefix flags at 0.0")
        expect(rl.draw_start(ctx, 0.0)["pool"], RULE["then_pool"], "draw at 0.0")
    return (f"demo_starts -> {len(keys)} set2 starts while frac > 0, set2 + set3 ({len(wide)}) "
            "at 0.0; --prefix-* only while frac > 0")


def r7(rl):
    auto = {**rl.DEFAULTS, "mppi_stages": "auto"}
    got = [rl.stages_for(auto, f) for f in (2.0, 1.5, 1.0, 0.5, 0.0, None)]
    expect(got, ["last", "last", "last", "all", "all", "all"], "auto")
    expect([rl.stages_for({**rl.DEFAULTS, "mppi_stages": s}, 1.0) for s in ("all", "last")],
           ["all", "last"], "fixed")
    expect(rl.DEFAULTS["mppi_stages"], "all", "default (v1.1)")
    return "auto: last for frac >= 1.0 else all; all / last fixed; default all"


def r8(rl):
    seq, decs = [], []
    successes = [3, 0, 0, 1]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        over = {"prefix": {"demos": DEMOS, "start_frac": 1.0}}
        for i in range(5):
            ctx = ctx_for(rl, root, over, i)
            cur = rl.curriculum(ctx, i)
            seq.append(cur["frac"])
            decs.append(cur["decision"])
            (root / f"iter{i}").mkdir(exist_ok=True)
            (root / f"iter{i}" / "curriculum.json").write_text(json.dumps(cur))
            expect(rl.curriculum(ctx, i), cur, f"iter {i} re-read")
            if i < len(successes):
                eps = [{"success": j < successes[i]} for j in range(5)]
                (root / f"iter{i}" / "labels.json").write_text(json.dumps({"episodes": eps}))
        expect(seq, [1.0, 0.5, 0.5, 1.0, 1.0], "fracs")
        expect(decs, ["start", "advance", "stay (1 x 0/n)", "retreat", "stay"], "decisions")
        fixed = rl.curriculum(ctx_for(rl, root / "iter0", {"prefix": {
            "demos": DEMOS, "schedule": None, "frac": 0.0}}), 0)
        expect((fixed["frac"], fixed["decision"]), (0.0, "fixed"), "schedule null")
        expect(rl.curriculum(ctx_for(rl, root / "iter1", {}), 0), None, "no demos -> off")
        for bad in ({"mppi_stages": "some"}, {"goal_mode": "track"},
                    {"prefix": {"start_frac": 0.7}}):
            try:
                ctx_for(rl, root / "iter2", bad)
            except SystemExit:
                continue
            raise AssertionError(f"config {bad} accepted")
    return (f"fracs {seq} ({', '.join(decs)}) from 3, 0, 0, 1 successes; schedule null fixed; "
            "bad configs rejected")


CHECKS = [("R1 advance", r1), ("R2 stay", r2), ("R3 retreat", r3), ("R4 clamp lowest", r4),
          ("R5 clamp highest", r5), ("R6 pool switch", r6), ("R7 mppi_stages", r7),
          ("R8 iteration chain", r8)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--loop", type=Path, default=REPO_ROOT / "scripts/lcs/realfirst_loop.py")
    a = ap.parse_args()
    rl = load_loop(a.loop)
    t0 = time.perf_counter()
    ok = 0
    for name, fn in CHECKS:
        try:
            detail = fn(rl)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"[FAIL] {name}: {exc!r}")
            continue
        ok += 1
        print(f"[PASS] {name}: {detail}")
    print(f"{ok}/{len(CHECKS)} {'PASS' if ok == len(CHECKS) else 'FAIL'} "
          f"({time.perf_counter() - t0:.1f} s)")
    return 0 if ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
