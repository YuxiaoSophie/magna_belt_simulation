#!/usr/bin/env python3
"""CPU checks for the whole-run chain (V12-108): chain flags opt-in with default-identical output,
chain validation, phase / npz layout of a chain smoke run, ``run_hard_suite.py write/summarize
--suite full`` (incl. ``--scene``). No sim, no LCM.

Run:
    uv run --frozen python scripts/checks/check_full_run.py [--pre-dir DIR_WITH_PRE_run_hard_suite.py]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import re
import subprocess
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

import eval_learned_mpc as harness
import run_hard_suite as rhs

E = "data/lcs/realfirst/cmp/encoder_study"
S1 = (f"{E}/sim/V45c/deploy_v17c/deploy.npz", f"{E}/sim/V45c/latent_waypoints.npz",
      f"{E}/sim87_V45c_ur30/configs/V45c_ur30_H7.yaml")
S2 = (f"{E}/sim/V50b/deploy_v18b/deploy.npz", f"{E}/sim/V50b/latent_waypoints.npz",
      f"{E}/sim104_V50b/configs/V50b_ffg29_ws1_du.yaml")
ALT_SCENE = "assets/round_belt_task/round_belt_scene_wr640.yaml"
RUNS = REPO_ROOT / "data/lcs/realfirst/cmp/v12_108"
PRE_DIR: Path | None = None
CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _port() -> int:
    used = subprocess.run(["ss", "-lupn"], check=False, capture_output=True, text=True).stdout
    return next(p for p in range(7781, 7800) if not re.search(rf":{p}\s", used))


PORT = _port()


def _write(mod, argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["run_hard_suite.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main()
    finally:
        sys.argv = old


def _refused(argv: list[str], needle: str) -> None:
    try:
        _write(rhs, argv)
    except SystemExit as exc:
        _require(needle in str(exc), f"{argv[-3:]}: {exc}")
    else:
        raise AssertionError(f"accepted: {argv}")


def _full(out: Path, *extra: str, arm2=True) -> list[str]:
    return ["write", "--suite", "full", "--out", str(out), "--port", str(PORT),
            "--demo-actions", "none", "--max-episode-s", "25", "--early-stop", "4,14,12",
            "--arm", "base", *S1, *(["--arm2", "base", *S2] if arm2 else []), *extra]


MPPI = ["--mode", "learned", "--controller", "mppi", "--mppi-goal", "waypoints",
        "--mppi-waypoints", S1[1], "--mppi-config", S1[2]]
CHAIN = ["--stage2-chain", "--stage2-deploy", S2[0], "--stage2-waypoints", S2[1],
         "--stage2-mppi-config", S2[2]]
FULL_PP = ["--post-mpc", "stage2_release", "--outcome-pulley", "both"]


def _opts(argv: list[str]):
    a = harness.create_parser().parse_args(argv)
    return a, harness._mppi_options(a, False)


@check("R1 chain / ur-exit flags: in STAGE2_ARG_DEFAULTS, parse to their default when absent")
def check_defaults() -> str:
    d = vars(harness.create_parser().parse_args(MPPI))
    want = {"stage2_chain", "stage2_deploy", "stage2_waypoints", "stage2_mppi_config",
            "stage2_max_episode_s", "stage2_early_stop", "stage2_mppi_max_duration_s",
            "ur_exit_byte", "ur_exit_settle_s", "ur_exit_lift_mm", "ur_exit_speed_mps",
            "ur_exit_final_settle_s", *harness.TELEOP_ARG_DEFAULTS,
            *harness.SCRIPTED_ARG_DEFAULTS, *harness.S2SNAP_ARG_DEFAULTS,
            *harness.SCRIPTED_UR_ARG_DEFAULTS}
    _require(set(harness.CHAIN_ARGS) == want, f"CHAIN_ARGS {set(harness.CHAIN_ARGS) ^ want}")
    bad = {k: (d[k], harness.STAGE2_ARG_DEFAULTS[k]) for k in want
           if d[k] != harness.STAGE2_ARG_DEFAULTS[k]}
    _require(not bad, f"defaults {bad}")
    return f"{len(want)} flags default-identical"


@check("R2 chain-only flags refused without --stage2-chain; chain requirements / refusals")
def check_validation() -> str:
    a, m = _opts([*MPPI, *FULL_PP])
    _require(harness._chain_options(a, m) is None, "flags off -> None")
    n = 0
    for flag, val in (("--stage2-deploy", S2[0]), ("--stage2-waypoints", S2[1]),
                      ("--stage2-mppi-config", S2[2]), ("--stage2-max-episode-s", "20"),
                      ("--stage2-early-stop", "1,2,3"), ("--stage2-mppi-max-duration-s", "5"),
                      ("--ur-exit-byte", "0"), ("--ur-exit-settle-s", "3"),
                      ("--ur-exit-lift-mm", "50"), ("--ur-exit-speed-mps", "0.1"),
                      ("--ur-exit-final-settle-s", "2")):
        a, m = _opts([*MPPI, flag, val])
        try:
            harness._chain_options(a, m)
        except ValueError as exc:
            _require("only with --stage2-chain" in str(exc) and flag in str(exc), str(exc))
            n += 1
        else:
            raise AssertionError(f"{flag} accepted without --stage2-chain")
    refusals = [("no --post-mpc", [*MPPI, "--outcome-pulley", "both", *CHAIN], ValueError),
                ("no --outcome-pulley both", [*MPPI, "--post-mpc", "stage2_release", *CHAIN],
                 ValueError),
                ("--grasp-arms franka", [*MPPI, *FULL_PP, "--grasp-arms", "franka", *CHAIN],
                 ValueError),
                ("demo actions", [*MPPI, *FULL_PP, "--mppi-demo-actions", "auto", *CHAIN],
                 ValueError),
                ("missing stage-2 deploy", [*MPPI, *FULL_PP, "--stage2-chain",
                                            "--stage2-waypoints", S2[1],
                                            "--stage2-mppi-config", S2[2]], FileNotFoundError),
                ("bogus stage-2 yaml", [*MPPI, *FULL_PP, *CHAIN[:-1], "/nonexistent.yaml"],
                 FileNotFoundError)]
    for what, argv, exc_t in refusals:
        a, m = _opts(argv)
        try:
            harness._chain_options(a, m)
        except exc_t:
            n += 1
        else:
            raise AssertionError(f"{what} accepted")
    try:  # not --controller mppi
        harness._chain_options(harness.create_parser().parse_args([*FULL_PP, *CHAIN]), None)
    except ValueError:
        n += 1
    else:
        raise AssertionError("chain without mppi accepted")
    return f"{n} refusals (11 flags without the chain, 7 chain requirements)"


@check("R3 flags-off identity (smoke_off vs smoke_off_pre): index.args, index and npz key sets")
def check_flags_off() -> str:
    on, pre = RUNS / "smoke_off", RUNS / "smoke_off_pre"
    if not (on / "index.json").is_file() or not (pre / "index.json").is_file():
        return "SKIP: v12_108/smoke_off{,_pre} not present"
    i_on, i_pre = (json.loads((x / "index.json").read_text()) for x in (on, pre))
    _require(set(i_on["args"]) == set(i_pre["args"]), "index.args keys")
    _require(not set(harness.CHAIN_ARGS) & set(i_on["args"]), "chain flags in index.args")
    _require(set(i_on) == set(i_pre) and "chain" not in i_on, "index keys")
    _require(i_on["phase_labels"] == i_pre["phase_labels"], "phase_labels")
    f_on, f_pre = (next(iter(sorted(x.glob("episode_*.npz")))) for x in (on, pre))
    with np.load(f_on, allow_pickle=True) as a, np.load(f_pre, allow_pickle=True) as b:
        _require(set(a.files) == set(b.files), f"npz keys {set(a.files) ^ set(b.files)}")
        _require("sim_chain_from" not in a.files, "sim_chain_from with flags off")
        _require(all(a[k].dtype.kind == b[k].dtype.kind and a[k].shape[1:] == b[k].shape[1:]
                 for k in a.files), "npz dtype kinds / shapes")
        _require(set(np.unique(a["sim_cmd_source"])) <= {0, 1, 2}, "cmd sources")
    return f"index.args ({len(i_on['args'])} keys), index keys, {len(a.files)} npz keys identical"


@check("R4 chain run layout: ur:/s2: labels before post:, cmd_source 8, sim_chain_from, grid")
def check_layout() -> str:
    d = RUNS / "seated_chain2"
    if not (d / "index.json").is_file():
        return "SKIP: v12_108/seated_chain2 not present"
    idx = json.loads((d / "index.json").read_text())
    lab = idx["phase_labels"]

    def first(p: str) -> int:
        return next(i for i, x in enumerate(lab) if x.startswith(p))

    def last(p: str) -> int:
        return max(i for i, x in enumerate(lab) if x.startswith(p))

    _require(lab.index("done") < first("ur:") and lab[first("ur:"):first("ur:") + 4]
             == list(harness.UR_EXIT_PHASES), "ur:* after done")
    _require(last("ur:") + 1 == first("s2:") and last("s2:") < first("post:"), "s2:* then post:*")
    _require(lab[last("s2:")] == "s2:done" and lab[first("post:"):]
             == [f"post:{p}" for p in harness.POST_PHASES], "s2:done / post tail")
    z = np.load(d / idx["episodes"][0]["file"], allow_pickle=True)
    c0, c1 = (int(x) for x in z["sim_chain_from"])
    cs, ph, st = z["sim_cmd_source"], z["sim_phase"], z["sim_step"]
    _require(0 < c0 < c1 < len(cs), f"chain_from {c0},{c1}")
    _require((cs[c0:c1] == harness.CMD_UR_EXIT).all() and not (cs[:c0] == 8).any()
             and not (cs[c1:] == 8).any(), "cmd_source 8 exactly on [from0, from1)")
    _require(all(lab[p].startswith("ur:") for p in set(ph[c0:c1])), "ur phases on the exit")
    _require(all(lab[p].startswith(("s2:", "post:")) for p in set(ph[c1:])), "s2/post after")
    _require((np.diff(st) == idx["sample_steps"]).all(), f"step grid {idx['sample_steps']}")
    _require(set(idx["chain"]) == {"flags", "scene", "stage2", "ur_exit", "rule"}
             and set(idx["episodes"][0]["chain"]) == {"stage1", "ur_exit", "stage2",
                                                       "success_both"}, "chain blocks")
    return (f"labels ok; cmd 8 on [{c0},{c1}) of {len(cs)}; steps all +{idx['sample_steps']}")


@check("R5 CMD_* codes: CMD_UR_EXIT == 8, all distinct (5 collector, 7 onpolicy)")
def check_cmds() -> str:
    codes = {k: v for k, v in vars(harness).items() if k.startswith("CMD_") and isinstance(v, int)}
    _require(harness.CMD_UR_EXIT == 8, "CMD_UR_EXIT")
    _require(len(set(codes.values())) == len(codes) and not {5, 7} & set(codes.values()),
             f"codes {codes}")
    return str(dict(sorted(codes.items(), key=lambda kv: kv[1])))


@check("R6 write --suite full: refusals, run.sh content, --scene override; stage 1/2 identity")
def check_write() -> str:
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        _refused(_full(t / "a", arm2=False), "--arm2")
        _refused(_full(t / "a", "--arm2", "other", *S2), "--arm2")
        _refused(_full(t / "a", "--demo-actions", "auto"), "--demo-actions none")
        _refused(_full(t / "a", "--scene", "assets/nope.yaml"), "no such file")
        _refused(_full(t / "a", "--starts", "bogus"), "--starts")
        _refused(["write", "--suite", "full", "--out", str(t / "a"), "--port", "7000",
                  "--arm", "base", *S1], "private ports")
        _write(rhs, _full(t / "f", "--repeats", "2", "--starts", "nominal,up+20"))
        sh = (t / "f/run.sh").read_text()
        for s in ("--stage2-chain", "--stage2-deploy", "--post-mpc stage2_release",
                  f"export ROUND_BELT_SCENE={rhs.STAGE2_SCENE}"):
            _require(s in sh, f"run.sh lacks {s}")
        _require(len(re.findall(r"^run base_", sh, re.MULTILINE)) == 2, "run lines")
        suite = json.loads((t / "f/suite.json").read_text())
        _require(suite["scene"] == rhs.STAGE2_SCENE and suite["repeats"] == 2, "suite.json")
        _write(rhs, _full(t / "g"))
        _require(len(re.findall(r"^run base_", (t / "g/run.sh").read_text(), re.MULTILINE))
                 == len(rhs.DEFAULT_KEYS), "10 starts")
        # --scene override (full and stage2); default stays the stage-2 scene
        _write(rhs, _full(t / "h", "--scene", ALT_SCENE))
        _require(f"export ROUND_BELT_SCENE={ALT_SCENE}" in (t / "h/run.sh").read_text()
                 and json.loads((t / "h/suite.json").read_text())["scene"] == ALT_SCENE,
                 "full --scene")
        s2 = ["write", "--suite", "stage2", "--out", str(t / "s"), "--port", str(PORT),
              "--demo-actions", "none", "--franka-min-mm", "0", "--arm", "base", *S2]
        _write(rhs, [*s2, "--scene", ALT_SCENE])
        _require(f"export ROUND_BELT_SCENE={ALT_SCENE}" in (t / "s/run.sh").read_text(),
                 "stage2 --scene")
        msg = "full+stage2 --scene ok"
        if PRE_DIR is None:
            return f"{msg}; SKIP byte-identical vs pre (no --pre-dir)"
        spec = importlib.util.spec_from_file_location("rhs_pre", PRE_DIR / "run_hard_suite.py")
        pre = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pre)
        pre.REPO = REPO_ROOT
        arms = [x for n in ("V49a", "V49b") for x in
                ("--arm", n, f"{E}/sim/{n}/deploy_v17{n[-1]}/deploy.npz",
                 f"{E}/sim/{n}/latent_waypoints.npz", f"{E}/sim/{n}/mppi_hard_suite.yaml")]
        sets = [["write", "--out", str(t / "o"), "--port", str(PORT), *arms],
                ["write", "--out", str(t / "o"), "--port", str(PORT), *arms, "--resets",
                 "snapshot", "--starts", "all", "--order", "arm", "--repeats", "3"],
                ["write", "--suite", "stage2", "--out", str(t / "o"), "--port", str(PORT),
                 "--demo-actions", "none", "--early-stop", "4,20,12", "--label", "v",
                 "--franka-min-mm", "2", *arms]]
        for argv in sets:
            got = []
            for mod in (pre, rhs):
                _write(mod, argv)
                got.append([(t / "o" / f).read_bytes() for f in ("run.sh", "suite.json")])
                for f in ("run.sh", "suite.json"):
                    (t / "o" / f).unlink()
            _require(got[0] == got[1], f"differs: {argv[:4]}")
        return f"{msg}; {len(sets)} stage-1/2 argument sets byte-identical vs {PRE_DIR}"


def _row_dir(root: Path, name: str, s1: str, ux, s2, outcome: str, final_small: str = "engaged"):
    d = root / name
    d.mkdir(parents=True)
    ep = {"file": "ep.npz", "start_state": {"id": name}, "repeat": 0, "outcome": outcome,
          "status": "ok", "reason": "early stop: engaged held 4.0 s", "controller_events": {},
          "stage_ends": [], "duration_s": 20.0,
          "chain": {"stage1": {"status": "ok", "reason": "x", "large_label_end": s1},
                    "ur_exit": ux, "stage2": s2}}
    if s2 is not None:
        fin = {"large": "engaged", "small": final_small}
        ep["post_mpc"] = {"mpc_end": fin, "final": fin, "ok": True}
        ep["mppi_stage2"] = {"solve_ms": {"p50": 30.0, "p95": 60.0}}
    q = np.tile([1.0, 0.0, 0.0, 0.0], (8, 1))
    np.savez(d / "ep.npz", sim_ee_franka=np.c_[np.zeros((8, 3)), q],
             sim_chain_from=np.int32([2, 4]), sim_post_from=np.int32(8))
    (d / "index.json").write_text(json.dumps({
        "chain": {}, "summary": {}, "episodes": [ep],
        "phase_labels": ["mpc:stage0", "done", "s2:stage0", "s2:stage1", "s2:done"]}))


@check("R7 summarize --suite full on synthetic rows: counters and failure classes")
def check_summary() -> str:
    ok, st2 = {"failed": False, "large_label_end": "engaged", "abort": None}, {
        "status": "ok", "reason": "early stop: engaged held 4.0 s"}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "s"
        for arm, rows in (("A", [("nominal", "over", None, None, "over"),
                                 ("up+20", "engaged", {**ok, "failed": True, "abort": "x",
                                                       "large_label_end": "under"}, None, "under"),
                                 ("set3_gv_03", "engaged", ok, st2, "under", "over")]),
                          ("B", [("nominal", "engaged", ok, st2, "engaged"),
                                 ("up+20", "engaged", ok, st2, "engaged"),
                                 ("set3_gv_03", "under", None, None, "under")])):
            for key, *r in rows:
                _row_dir(root, f"{arm}_{key}", *r)
        a = argparse.Namespace(suite="full", dir=[str(root)], out=None, starts="nominal,up+20,"
                               "set3_gv_03", collector=None, stage2_src="V45c")
        with contextlib.redirect_stdout(io.StringIO()):
            rhs.cmd_summarize(a)
        rows = json.loads((root / "hard_suite.json").read_text())["rows"]
        md = (root / "hard_suite.md").read_text()
    cnt = {arm: tuple(sum(bool(r[k]) for r in rows if r["arm"] == arm) for k in
                      ("stage1_seated", "ur_exit_ok", "stage2_ran", "engaged"))
           for arm in "AB"}
    _require(cnt == {"A": (2, 1, 1, 0), "B": (2, 2, 2, 2)}, f"counters {cnt}")
    fail = {(r["arm"], r["start"]): r["failure"] for r in rows}
    _require(fail == {("A", "nominal"): "stage1", ("A", "up+20"): "ur_exit",
                      ("A", "set3_gv_03"): "stage2:small over (engaged/over)",
                      ("B", "nominal"): "success", ("B", "up+20"): "success",
                      ("B", "set3_gv_03"): "stage1"}, f"failures {fail}")
    _require("| A | 2/3 | 1/3 | 1/3 | **0/3** |" in md and "| B | 2/3 | 2/3 | 2/3 | **2/3** |"
             in md, "md counters")
    return f"counters A {cnt['A']} B {cnt['B']}; stage1 / ur_exit / stage2:small / success classes"


def main() -> int:
    global PRE_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="dir with the pre-chain run_hard_suite.py (R6 byte-identical part)")
    PRE_DIR = ap.parse_args().pre_dir
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
        print(f"[{'SKIP' if detail.startswith('SKIP') else 'PASS'}] {name}: {detail}")
    print(f"ALL FULL-RUN CHECKS PASSED ({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
