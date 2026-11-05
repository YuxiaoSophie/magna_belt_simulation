#!/usr/bin/env python3
"""Checks for the chain's scripted stage-2 walk (V12-123, ``--stage2-scripted``; V12-130
``--scripted-ur hold|feed``): flag defaults / refusals, ``--pre-dir`` identity (parser + chain index
block of a node / teleop / default scripted chain), the walk legs vs
``stage2_collect.Collector._run``, the UR feed ramp, and the layout of a recorded scripted-chain
run (``--run``; SKIP if absent). CPU only, no sim.

Run:
    uv run --frozen python scripts/checks/check_stage2_scripted.py [--pre-dir DIR_WITH_PRE_eval_learned_mpc.py] [--run DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import re
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
import stage2_collect as s2
import stage2_scripted as ss
from check_grip_encoder import strip_v131

RUN = REPO_ROOT / "data/lcs/realfirst/cmp/v12_123/smoke"
NEW = {"stage2_scripted": False, "scripted_params": "nominal", "scripted_dart": 0.0,
       "scripted_seed": 20261019}
NEW_UR = {"scripted_ur": "exit"}  # V12-130
FLAGS = {"--scripted-params": "draw", "--scripted-dart": "1.0", "--scripted-seed": "7"}
BASE = [*cfr.MPPI, *cfr.FULL_PP]
SCRIPTED = [*BASE, "--stage2-chain", "--stage2-scripted"]
PRE_DIR: Path | None = None
CHECKS: list[tuple[str, callable]] = []
_require = cfr._require


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _opts(mod, argv: list[str]):
    a = mod.create_parser().parse_args(argv)
    return a, mod._mppi_options(a, False)


def _refused(argv: list[str], needle: str) -> None:
    a, m = _opts(harness, argv)
    try:
        harness._chain_options(a, m)
    except ValueError as exc:
        _require(needle in str(exc), f"{argv[-3:]}: {exc}")
    else:
        raise AssertionError(f"accepted: {argv[-4:]}")


@check("C1 scripted flags: in STAGE2_ARG_DEFAULTS and CHAIN_ARGS, default when absent; code 5")
def check_defaults() -> str:
    d = vars(harness.create_parser().parse_args(cfr.MPPI))
    for k, v in NEW.items():
        _require(k in harness.STAGE2_ARG_DEFAULTS and k in harness.CHAIN_ARGS, f"{k} unregistered")
        _require(harness.SCRIPTED_ARG_DEFAULTS[k] == v == d[k], f"{k}: {d[k]} != {v}")
    for k, v in NEW_UR.items():
        _require(k in harness.STAGE2_ARG_DEFAULTS and k in harness.CHAIN_ARGS, f"{k} unregistered")
        _require(harness.SCRIPTED_UR_ARG_DEFAULTS[k] == v == d[k], f"{k}: {d[k]} != {v}")
    codes = {k: v for k, v in vars(harness).items() if k.startswith("CMD_") and isinstance(v, int)}
    _require(s2.CMD_STAGE2 == 5 and 5 not in codes.values(), f"harness codes {codes}")
    return (f"{len(NEW) + len(NEW_UR)} flags default-identical; walk frames use stage2_collect's "
            "code 5")


@check("C2 refusals: without the chain / --stage2-scripted, with the node triple / teleop")
def check_refusals() -> str:
    n = 0
    for flag, val in [*FLAGS.items(), ("--stage2-scripted", None)]:
        _refused([*BASE, flag, *([val] if val else [])], "only with --stage2-chain")
        n += 1
    for flag, val in FLAGS.items():
        _refused([*BASE, *cfr.CHAIN, flag, val], "only with --stage2-scripted")
        n += 1
    _refused([*SCRIPTED, "--stage2-deploy", cfr.S2[0]], "no stage-2 node")
    _refused([*SCRIPTED, "--stage2-teleop"], "not with --stage2-teleop")
    _refused([*SCRIPTED, "--teleop-max-s", "5"], "not with --teleop-max-s")
    a, m = _opts(harness, [*SCRIPTED, *itertools.chain(*FLAGS.items())])
    ch = harness._chain_options(a, m)
    _require(ch["scripted"] is True and ch["stage_labels"] == list(ss.PHASES)
             and ch["cmd_source"] == 5 and (ch["params"], ch["dart"], ch["seed"]) == ("draw", 1.0, 7)
             and harness._nodeless(ch), f"options {ch}")
    blk = harness._chain_index_block(a, ch)
    _require(set(blk) == {"flags", "scene", "scripted", "ur_exit"}
             and blk["scripted"]["flags"]["scripted_dart"] == 1.0 and "ur" not in ch
             and "scripted_ur" not in blk["flags"]
             and not set(harness.TELEOP_ARG_DEFAULTS) & set(blk["flags"]), f"index block {blk}")
    _refused([*BASE, "--scripted-ur", "hold"], "only with --stage2-chain")
    _refused([*BASE, *cfr.CHAIN, "--scripted-ur", "feed"], "only with --stage2-scripted")
    for mode in ss.UR_MODES:
        a, m = _opts(harness, [*SCRIPTED, "--scripted-ur", mode])
        ch = harness._chain_options(a, m)
        blk = harness._chain_index_block(a, ch)
        _require(ch["ur"] == mode and set(blk) == {"flags", "scene", "scripted", "ur_keep"}
                 and blk["flags"]["scripted_ur"] == mode and blk["ur_keep"]["mode"] == mode,
                 f"{mode}: {blk}")
    return f"{n + 5} refusals; options + index blocks (exit / {', '.join(ss.UR_MODES)}) ok"


@check("C3 --pre-dir identity: parser (3 argv sets), chain index block (node, teleop)")
def check_pre() -> str:
    if PRE_DIR is None:
        return "SKIP: no --pre-dir"
    spec = importlib.util.spec_from_file_location("elm_pre", PRE_DIR / "eval_learned_mpc.py")
    pre = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pre  # dataclasses look the module up
    spec.loader.exec_module(pre)

    def norm(d: dict) -> dict:
        return {k: REPO_ROOT / v.relative_to(pre.REPO_ROOT) if isinstance(v, Path)
                and v.is_relative_to(pre.REPO_ROOT) else v for k, v in d.items()}

    new = {k for k in {**NEW, **NEW_UR, "publish_body_poses": False,  # + lcm-viewer's
                       "realtime": False}  # + demo GUI's
           if k not in pre.STAGE2_ARG_DEFAULTS}  # keys newer than pre
    sets = {"stage-1 suite line": BASE, "node chain": [*BASE, *cfr.CHAIN], "plain": cfr.MPPI}
    for what, argv in sets.items():
        a, b = norm(vars(pre.create_parser().parse_args(argv))), vars(
            harness.create_parser().parse_args(argv))
        for k in [*new, *harness.GRIP_ARG_DEFAULTS]:  # + V12-132's later flag
            b.pop(k)
        _require(a == b, f"{what}: {sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))}")
    _require({k: v for k, v in harness.STAGE2_ARG_DEFAULTS.items() if k not in new}
             == pre.STAGE2_ARG_DEFAULTS, "STAGE2_ARG_DEFAULTS minus the new keys")
    _require([k for k in harness.CHAIN_ARGS if k not in new] == pre.CHAIN_ARGS, "CHAIN_ARGS")
    blocks_argv = [("node", [*BASE, "--belt-seg", *cfr.CHAIN]),
                   ("teleop", [*BASE, "--stage2-chain", "--stage2-teleop"])]
    if hasattr(pre, "SCRIPTED_ARG_DEFAULTS"):
        blocks_argv.append(("scripted", [*SCRIPTED, *itertools.chain(*FLAGS.items())]))
    for what, argv in blocks_argv:
        blocks = []
        for mod in (pre, harness):
            a, m = _opts(mod, argv)
            ch = mod._chain_options(a, m)
            blk = mod._chain_index_block(a, ch)
            blk.get("stage2", {}).pop("node", None)  # binary path: same in both
            blocks.append(json.dumps(blk, sort_keys=True, default=str))
        _require(blocks[0] == blocks[1], f"{what} chain index block differs")
        _require(not harness._nodeless(ch) if what == "node" else harness._nodeless(ch), what)
    return (f"{len(sets)} argv sets, defaults, CHAIN_ARGS, "
            f"{'/'.join(w for w, _ in blocks_argv)} index blocks identical")


@check("C4 walk legs == stage2_collect.Collector._run's walk calls (loops off)")
def check_legs() -> str:
    src = Path(s2.__file__).read_text()
    i = src.index("    def _run(self")
    body = src[i:src.index("        frames = ep.rec.frames", i)]
    calls = re.findall(r'self\._walk\(ep, (poses\[(\d+):(\d+)\]|\[\]), "(\w+)", \d+, v, rng_n'
                       r'(?:, (\w+))?\)', body)
    want = [(ph, (int(a), int(b)) if a else None, {"HOVER_HOLD_S": s2.HOVER_HOLD_S,
             "END_HOLD_S": s2.END_HOLD_S}.get(h, 0.0)) for _, a, b, ph, h in calls]
    poses = [np.full(7, float(i)) for i in range(11)]
    got = [(ph, (int(t[0][0]), int(t[-1][0]) + 1) if t else None, h)
           for ph, t, h in ss.legs(poses)]
    _require(len(calls) == 7 and got == want, f"legs {got} vs collector {want}")
    _require(tuple(dict.fromkeys(p for p, *_ in got)) == ss.PHASES, "PHASES order")
    return f"{len(got)} legs match ({', '.join(p for p, *_ in got)})"


@check("C6 --scripted-ur feed ramp: min-jerk ends, monotone, orientation kept; modes == choices")
def check_ur_ramp() -> str:
    a = np.r_[0.4, -0.1, 0.03, 1.0, 0.0, 0.0, 0.0]
    b = np.r_[a[:2] + [ss.FEED_MM * 1e-3, 0.0], a[2:]]
    r = ss.Ramp(a, b, 2.0)
    xs = [r.pose7(t)[0] for t in np.linspace(-0.5, 2.5, 61)]
    _require(np.allclose(r.pose7(0.0), a) and np.allclose(r.pose7(2.0), b)
             and np.all(np.diff(xs) >= -1e-15) and np.allclose(r(1.0)[1], a[3:]), "ramp")
    _require(abs(r.pose7(1.0)[0] - (a[0] + b[0]) / 2) < 1e-12, "min-jerk midpoint")
    choices = next(x.choices for x in harness.create_parser()._actions
                   if x.dest == "scripted_ur")
    _require(tuple(choices) == ("exit", *ss.UR_MODES), f"choices {choices}")
    return f"Ramp ok (feed {ss.FEED_MM:g} mm over {ss.FEED_OUT}, back over {ss.FEED_BACK})"


UR_PRE_CHANGED = {  # the only pre lines V12-130 changes; every other difference is an insertion
    "eval_learned_mpc.py": [
        "              or k in TELEOP_ARG_DEFAULTS or k in SCRIPTED_ARG_DEFAULTS]",
        "    sim.set_grippers(s2.OPEN_MM, None)",
        "    bad = [k for k in SCRIPTED_ARG_DEFAULTS if flags[k] != SCRIPTED_ARG_DEFAULTS[k]]",
        '                "seed": flags["scripted_seed"]}',
        "                          and (k not in S2SNAP_ARG_DEFAULTS or getattr(args, k)))}"],
    "stage2_scripted.py": [
        "def run(sim, ctx, sampler, last: dict, timing: dict, args, name: str) -> dict:",
        '    ``{frames, report, done, tip_min_m}``; ``done`` False (an abort) -> no release."""']}
UR_MARKERS = ("SCRIPTED_UR", "scripted_ur", "ur_kept", "ur_mode", "UR_MODES", "Ramp", "FEED_",
              "feed", 'ch.get("ur")', "u0", "u_feed", "stage2_scripted", "ur_keep", "held_ur",
              "flags[k] != STAGE2_ARG_DEFAULTS[k]]", "(k not in S2SNAP_ARG_DEFAULTS or getattr(args, k))",
              "\"ur\": flags", "``ur_mode``", "release.", "hold / feed")


@check("C7 --pre-dir sources: V12-130 changes only the listed pre lines; every added block is ur-keep")
def check_ur_sources() -> str:
    if PRE_DIR is None or not (PRE_DIR / "stage2_scripted.py").is_file():
        return "SKIP: no --pre-dir with stage2_scripted.py"
    import difflib

    n = 0
    for f, changed in UR_PRE_CHANGED.items():
        if not (PRE_DIR / f).is_file():
            continue
        old, new = ((PRE_DIR / f).read_text().splitlines(),
                    strip_v131(f"scripts/lcs/{f}",  # later opt-in V12-131 hunks
                               (REPO_ROOT / "scripts/lcs" / f).read_text()).splitlines())
        sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
        gone = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            gone += old[i1:i2]
            blk = "\n".join(new[j1:j2])
            _require(any(m in blk for m in UR_MARKERS) or not blk.strip(), f"{f}: block {blk[:120]!r}")
            n += 1
        _require(gone == changed, f"{f}: changed pre lines {gone}")
    return f"{n} diff blocks, all ur-keep; changed pre lines == the listed {sum(map(len, UR_PRE_CHANGED.values()))}"


def _runs(a) -> list[int]:
    return [int(k) for k, _ in itertools.groupby(a)]


@check("C5 recorded scripted-chain run: cmd runs, chain_from, labels, grid, index blocks")
def check_run() -> str:
    idx_p = next(iter(sorted(RUN.glob("*/index.json"))), None)
    if idx_p is None:
        return f"SKIP: no run under {RUN}"
    from task_common import lcs_dataset as lcs

    idx = json.loads(idx_p.read_text())
    row = next((r for r in idx["episodes"] if (r.get("chain") or {}).get("stage2")), None)
    if row is None:
        return f"SKIP: {idx_p.parent.name}: no episode reached stage 2"
    lab, d = idx["phase_labels"], idx_p.parent
    _require(lab[lab.index("ur:settle") + 1:lab.index("post:release")]
             == [f"s2:{p}" for p in ss.PHASES], "s2 labels (no s2:done)")
    z = np.load(d / row["file"], allow_pickle=True)
    cs, ph, st = z["sim_cmd_source"], z["sim_phase"], z["sim_step"]
    runs = _runs(cs)
    c0, c1 = (int(x) for x in z["sim_chain_from"])
    w = np.flatnonzero(cs == 5)
    _require(runs[-3:] == [8, 5, 6] or runs[-2:] == [8, 5], f"cmd runs {runs}")
    _require(c1 == w[0] and (cs[c0:c1] == 8).all(), f"chain_from {c0},{c1}")
    _require({lab[p] for p in ph[w]} <= {f"s2:{p}" for p in ss.PHASES}, "walk phase labels")
    _require((np.diff(st) == idx["sample_steps"]).all(), "step grid")
    ur = [j for j, n in enumerate(lcs.ACTION_DIM_NAMES) if n.startswith("u_")]
    _require(np.allclose(z["actions"][np.ix_(w, ur)], 0.0, atol=1e-12), "UR u != 0 in the walk")
    _require(set(idx["chain"]) == {"flags", "scene", "scripted", "ur_exit"}
             and row["chain"]["stage2"]["scripted"]["frames"] == len(w), "index / row blocks")
    _require("mppi_stage2" not in row, "stage-2 solve stats without a node")
    lcs.validate_episode(d / row["file"], period_us=round(idx["sample_steps"] * lcs.SIM_DT_S * 1e6))
    return (f"{d.name}: cmd runs {runs[-4:]}, {len(w)} walk frames, chain_from [{c0},{c1}], "
            f"stage 2 {row['chain']['stage2']['status']}, both {row['chain']['success_both']}")


def main() -> int:
    global PRE_DIR, RUN
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="dir with the pre-change eval_learned_mpc.py (C3)")
    ap.add_argument("--run", type=Path, default=RUN, help="a scripted-chain suite dir (C5)")
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
    print(f"ALL SCRIPTED CHECKS PASSED ({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
