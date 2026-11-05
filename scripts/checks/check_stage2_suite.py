#!/usr/bin/env python3
"""CPU checks for the stage-2 hard suite (V12-95): ``run_hard_suite.py write`` stage-1 output
byte-identical to a reference runner (default: git HEAD's), the stage-2 run.sh executed with a stub
``uv`` and every harness command parsed by the harness parser, the post-MPC release targets vs the
collector's post targets, the harness stage-2 flag defaults / validation / phase index, and the
stage-2 summary on synthetic run dirs. No sim, no LCM.

Run:
    uv run --frozen python scripts/checks/check_stage2_suite.py [--pre OLD_run_hard_suite.py]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_learned_mpc as harness
import run_hard_suite as rhs
import stage2_collect as s2

SIM = "data/lcs/realfirst/cmp/encoder_study/sim"
ARMS = [(a, f"{SIM}/{a}/deploy_v17{a[-1]}/deploy.npz", f"{SIM}/{a}/latent_waypoints.npz",
         f"{SIM}/{a}/mppi_hard_suite.yaml") for a in ("V49a", "V49b", "V49c")]


def _free_port() -> int:
    """A private port no socket holds (run.sh refuses a held one)."""
    used = subprocess.run(["ss", "-lupn"], check=False, capture_output=True, text=True).stdout
    return next(p for p in range(7781, 7800) if not re.search(rf":{p}\s", used))


PORT = _free_port()
PRE: Path | None = None
CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _load_runner(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO = REPO_ROOT  # a copy outside the repo resolves paths against the repo
    return mod


def _write(mod: types.ModuleType, argv: list[str]) -> None:
    old = sys.argv
    sys.argv = ["run_hard_suite.py", *argv]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main()
    finally:
        sys.argv = old


def _arm_args(arms) -> list[str]:
    return [x for a in arms for x in ("--arm", *a)]


@check("Q1 stage-1 write: run.sh + suite.json byte-identical to the reference runner")
def check_stage1_identity() -> str:
    with tempfile.TemporaryDirectory() as td:
        ref_path = PRE
        if ref_path is None:
            ref_path = Path(td) / "run_hard_suite_head.py"
            ref_path.write_text(subprocess.run(
                ["git", "show", "HEAD:scripts/lcs/run_hard_suite.py"], cwd=REPO_ROOT,
                check=True, capture_output=True, text=True).stdout)
        ref, cur = _load_runner(ref_path, "rhs_ref"), rhs
        sets = [[], ["--resets", "snapshot", "--starts", "all", "--order", "arm",
                     "--early-stop", "off", "--repeats", "3"],
                ["--scene", "assets/round_belt_task/round_belt_scene_stage2.yaml",
                 "--demo-actions", "none", "--max-episode-s", "25", "--early-stop", "4,14,12",
                 "--cloud-noise", "default", "--label", "x", "--starts", "nominal,up+20"]]
        if "--eval-extra" in ref_path.read_text():
            sets.append(["--eval-extra", "--end-snapshot", "--demo-actions", "none"])
        out = Path(td) / "suite"
        for extra in sets:
            argv = ["write", "--out", str(out), "--port", str(PORT), *_arm_args(ARMS[:2]),
                    *extra]
            got = {}
            for tag, mod in (("ref", ref), ("cur", cur)):
                _write(mod, argv)
                got[tag] = {f: (out / f).read_bytes() for f in ("run.sh", "suite.json")}
                for f in ("run.sh", "suite.json"):
                    (out / f).unlink()
            for f in ("run.sh", "suite.json"):
                _require(got["ref"][f] == got["cur"][f], f"{f} differs for {extra}")
    return (f"{len(sets)} argument sets identical vs "
            f"{'git HEAD' if PRE is None else PRE}")


def _stage2_argv(out: Path, fmin: str = "2", arms=ARMS, extra=()) -> list[str]:
    return ["write", "--suite", "stage2", "--out", str(out), "--port", str(PORT),
            "--demo-actions", "none", "--early-stop", "4,20,12", "--max-episode-s", "30",
            "--final-hold-s", "3", "--franka-min-mm", fmin, "--label", "v12_95",
            *_arm_args(arms), *extra]


def _run_stub(out: Path) -> list[dict]:
    """Execute ``out/run.sh`` with a stub ``uv`` on PATH; one record per harness call."""
    stub = out / "stub"
    stub.mkdir()
    log = out / "calls.jsonl"
    (stub / "uv").write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'scene': "
        "os.environ.get('ROUND_BELT_SCENE'), 'cwd': os.getcwd()}) + '\\n')\n")
    (stub / "uv").chmod(0o755)
    subprocess.run(["bash", str(out / "run.sh")], check=True,
                   env={**os.environ, "PATH": f"{stub}:{os.environ['PATH']}"})
    return [json.loads(x) for x in log.read_text().splitlines()]


@check("Q2 stage-2 write: run.sh calls parse with the stage-2 harness flags (stub uv)")
def check_stage2_write() -> str:
    p = harness.create_parser()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "s2"
        _write(rhs, _stage2_argv(out))
        sh = (out / "run.sh").read_text()
        _require(f"export ROUND_BELT_SCENE={rhs.STAGE2_SCENE}" in sh, "scene export")
        calls = _run_stub(out)
        suite = json.loads((out / "suite.json").read_text())
        _require(len(calls) == len(ARMS) * len(rhs.DEFAULT_KEYS),
                 f"{len(calls)} calls, want {len(ARMS) * len(rhs.DEFAULT_KEYS)}")
        n_ep, seen = 0, set()
        for c in calls:
            _require(c["argv"][:4] == ["run", "--frozen", "python",
                                       "scripts/lcs/eval_learned_mpc.py"], c["argv"][:4])
            _require(c["scene"] == rhs.STAGE2_SCENE and Path(c["cwd"]) == REPO_ROOT,
                     f"env {c['scene']} / {c['cwd']}")
            a = p.parse_args(c["argv"][4:])
            arm, key = next((n, a.label[len(f"v12_95_{n}_"):]) for n, *_ in ARMS
                            if a.label.startswith(f"v12_95_{n}_"))
            dep, wp, cfg = next(x[1:] for x in ARMS if x[0] == arm)
            want = {"start_states": Path(rhs.STAGE2_STARTS), "repeats": 1,
                    "variants": f"V45c_{key}_0,V45c_{key}_1", "grasp_arms": "franka",
                    "outcome_pulley": "both", "post_mpc": "stage2_release",
                    "start_settle_s": 2.0, "franka_width_max": 20.0,
                    "clearance_jaw_bytes": [191], "controller": "mppi",
                    "mppi_goal": "waypoints", "mppi_stages": "all", "mppi_goal_mode": "stage",
                    "mppi_demo_offset_frames": 0, "mppi_demo_actions": None,
                    "mppi_max_duration_s": 3.0, "mppi_clearance": True,
                    "mppi_clearance_franka_min_mm": 2.0, "mppi_clearance_ur_min_mm": 3.0,
                    "reset": "snapshot", "reset_spec": [], "record": True,
                    "max_episode_s": 30.0, "early_stop": "4,20,12", "belt_seg": True,
                    "cloud_noise": "off", "mode": "learned", "prefix_frac": 0.0,
                    "deploy": (REPO_ROOT / dep).resolve(), "mppi_waypoints": Path(wp),
                    "mppi_config": (REPO_ROOT / cfg).resolve(),
                    "lcm_url": rhs.url(PORT)}
            bad = {k: (getattr(a, k), v) for k, v in want.items() if getattr(a, k) != v}
            _require(not bad, f"{arm} {key}: {bad}")
            states, _ = harness.load_start_states(REPO_ROOT / a.start_states, a.variants)
            _require(len(states) == 2 and all(s.file.is_file() for s in states),
                     f"{key}: start files")
            n_ep += len(states) * a.repeats
            seen.add((arm, key))
        _require(len(seen) == len(calls), "duplicate runs")
        _require(n_ep == 20 * len(ARMS), f"{n_ep} episodes")
        _require(suite["suite"] == "stage2" and suite["franka_min_mm"] == 2.0, "suite.json")
        # the yaml's clearance.franka_min_m wins in the node: a mismatch is refused
        try:
            _write(rhs, _stage2_argv(Path(td) / "bad", fmin="1"))
        except SystemExit as exc:
            _require("franka_min_m" in str(exc), str(exc))
        else:
            raise AssertionError("franka-min 1 mm vs yaml 2 mm accepted")
    return (f"{len(calls)} harness calls ({len(ARMS)} arms x {len(rhs.DEFAULT_KEYS)} keys), "
            f"{n_ep} episodes, stage-2 scene exported, all flags parsed; yaml mismatch refused")


@check("Q3 release targets / plan = the collector's post targets and phases")
def check_release_targets() -> str:
    hw = s2.hw_poses()
    ee0 = np.array([0.3688, 0.0804, 0.0470, 0.6363, -0.0611, -0.1360, 0.7569])
    ee0[3:] /= np.linalg.norm(ee0[3:])
    rng = np.random.default_rng(0)
    worst = 0.0
    for prm in [dict(s2.NOMINAL), *(s2.draw_params(rng) for _ in range(30))]:
        w10 = s2.build_plan(prm, ee0, hw)[-1][2]
        got, want = s2.release_targets(w10), s2.post_poses(prm, hw)
        for g, w in zip(got, want, strict=True):
            worst = max(worst, float(np.abs(g - w).max()))
        plan = s2.release_plan(w10, prm["speed_mps"])
        _require([(ph, len(t), v, h) for ph, t, v, h in plan] == [
            ("release", 0, prm["speed_mps"], s2.RELEASE_S),
            ("out", 1, s2.OUT_SPEED_MPS, s2.POST_HOLD_S),
            ("lift", 1, prm["speed_mps"], s2.POST_HOLD_S),
            ("done", 0, prm["speed_mps"], s2.DONE_S)], "plan phases")
    _require(worst < 1e-9, f"targets differ by {worst:.3g} m")
    # the collector's own release sequence still uses the same constants in the same order
    src = Path(s2.__file__).read_text()
    body = src[src.index("    def _run(self, row"):src.index("\ndef _rel(")]
    seq = re.findall(r'self\._walk\(ep, (\[\]|\[out7\]|\[lift7\]), "(\w+)", \d+, '
                     r"(\w+), rng_n,\s*(\w+)", body)
    _require(seq[-4:] == [("[]", "release", "v", "RELEASE_S"),
                          ("[out7]", "out", "OUT_SPEED_MPS", "POST_HOLD_S"),
                          ("[lift7]", "lift", "v", "POST_HOLD_S"),
                          ("[]", "done", "v", "DONE_S")], f"collector sequence {seq[-4:]}")
    _require("sim.set_grippers(OPEN_MM, None)" in body, "collector opens to OPEN_MM")
    # guards: never inwards, never down
    far = np.r_[s2.small_to_world(60.0, 120.0, 40.0), ee0[3:]]
    o, lft = s2.release_targets(far)
    r_o, az_o, h_o = s2.small_polar(o[:3])
    r_l, _, h_l = s2.small_polar(lft[:3])
    _require(abs(r_o - 60) < 1e-6 and abs(az_o - 120) < 1e-6 and abs(h_o - 40) < 1e-6
             and abs(r_l - 60) < 1e-6 and abs(h_l - 40) < 1e-6, "far / high pose guards")
    near = np.r_[s2.small_to_world(25.0, 141.0, -1.0), ee0[3:]]
    o, lft = s2.release_targets(near)
    _require(abs(s2.small_polar(o[:3])[0] - s2.OUT_R_MM) < 1e-6
             and abs(s2.small_polar(lft[:3])[2] - s2.LIFT_H_MM) < 1e-6
             and np.allclose(o[3:], ee0[3:]), "out r 45 / lift h 27 / orientation kept")
    return (f"31 plans: max |release - collector post| {worst:.1e} m; phases release/out/lift/"
            f"done at {s2.RELEASE_S}/{s2.OUT_SPEED_MPS} m/s/{s2.POST_HOLD_S}/{s2.DONE_S} s, hand "
            f"{s2.OPEN_MM:g} mm; collector sequence in sync; no inward / downward moves")


@check("Q4 harness: stage-2 flag defaults, validation before the sim, phase index with post")
def check_harness() -> str:
    p = harness.create_parser()
    d = vars(p.parse_args([]))
    bad = {k: (d[k], v) for k, v in harness.STAGE2_ARG_DEFAULTS.items() if d[k] != v}
    _require(not bad, f"defaults vs STAGE2_ARG_DEFAULTS (dropped from index.args): {bad}")
    url = rhs.url(PORT)
    for argv, msg in ((["--post-mpc", "stage2_release"], "--grasp-arms franka"),
                      (["--post-mpc", "stage2_release", "--grasp-arms", "franka"],
                       "--outcome-pulley both")):
        try:
            harness.evaluate(p.parse_args(["--lcm-url", url, "--dry-run", *argv]))
        except ValueError as exc:
            _require(msg in str(exc), str(exc))
        else:
            raise AssertionError(f"{argv} accepted")
    with tempfile.TemporaryDirectory() as td:  # the suite's real command, dry run
        out = Path(td) / "s2"
        _write(rhs, _stage2_argv(out, arms=ARMS[2:]))
        line = next(x for x in (out / "run.sh").read_text().splitlines()
                    if x.startswith("run V49c_nominal "))
        sh = (out / "run.sh").read_text()
        body = sh[sh.index("  uv run"):sh.index("> $B/$1.log")].replace("\\\n", " ")
        _, _, variants, dep, wp, cfg = shlex.split(line)
        body = (body.replace('"$2"', variants).replace("$3", dep).replace("$4", wp)
                .replace("$5", cfg).replace("$B/$1", str(out / "x")).replace("$1", "x"))
        a = p.parse_args(shlex.split(body)[5:] + ["--dry-run"])
        with contextlib.redirect_stdout(io.StringIO()):
            plan = harness.evaluate(a)
        _require(len(plan["episodes"]) == 2, f"{len(plan['episodes'])} planned episodes")
    labels = ["mpc:stage0", "mpc:stage1", "done", *(f"post:{x}" for x in harness.POST_PHASES)]
    ctx = types.SimpleNamespace(n_pre=0, phase_labels=labels, mode="learned", n_stages=2)
    ev = {"pre_reached": {}, "stages": {0: {"t": 1.0}}, "all_pre_completed": None,
          "mpc_completed": 3.0}
    got = [harness._phase_and_stage(ctx, ev, t) for t in (0.5, 2.0, 3.5)]
    _require(got == [(0, 0), (1, 1), (2, 1)], f"phase/stage {got}")
    return (f"{len(harness.STAGE2_ARG_DEFAULTS)} stage-2 defaults match; post-mpc without "
            "franka/both refused before the sim; the suite command dry-runs 2 episodes; 'done' "
            "found by name with post labels after it")


def _fake_run(root: Path) -> None:
    d = root / "V49x_nominal"
    d.mkdir(parents=True)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([np.cos(np.radians(30)), 0.0, 0.0, np.sin(np.radians(30))])  # 60 deg
    ee = np.zeros((6, 7))
    ee[:, 3:] = [q0, q0, q1, q0, q1, q1]
    np.savez(d / "episode_a.npz", sim_ee_franka=ee, sim_post_from=np.int32(3))
    np.savez(d / "episode_b.npz", sim_ee_franka=ee, sim_post_from=np.int32(6))
    lab = {"large": "engaged", "small": "engaged", "both": True, "large_wrap_deg": 120.0,
           "small_wrap_deg": 150.0, "large_h_mm": 0.1, "small_h_mm": 0.2, "held": True,
           "label": "engaged"}
    bad = {**lab, "small": "under", "both": False, "label": "under"}
    eps = [{"file": "episode_a.npz", "start_state": {"id": "V45c_nominal_0"},
            "outcome": "engaged", "status": "ok", "reason": "early stop: engaged held 4.0 s",
            "stage_ends": ["reached", "timeout"], "controller_events": {},
            "mppi": {"solve_ms": {"p95": 30.0}, "clear_min_mm": {"franka": 0.5}},
            "duration_s": 20.0, "post_mpc": {"mpc_end": lab, "final": lab, "ok": True,
                                             "mpc_duration_s": 15.0, "phases": []}},
           {"file": "episode_b.npz", "start_state": {"id": "V45c_nominal_1"},
            "outcome": "under", "status": "aborted", "reason": "grasp lost at step 9",
            "stage_ends": ["reached"], "controller_events": {}, "duration_s": 12.0,
            "post_mpc": {"mpc_end": lab, "final": bad, "ok": True, "mpc_duration_s": 7.0,
                         "phases": [{"abort": "x", "abort_kind": "unreachable"}]}},
           {"file": None, "start_state": {"id": "V45c_nominal_2"}, "reason": "grasp lost"}]
    (d / "index.json").write_text(json.dumps({
        "post_mpc": {"kind": "stage2_release"}, "summary": {},
        "phase_labels": ["mpc:s0", "mpc:s1", "done"], "episodes": eps}))
    c = root / "collector"
    c.mkdir()
    (c / "index.json").write_text(json.dumps({"episodes": [
        {"file": "c.npz", "start": {"key": "nominal", "arm": "V45c", "id": "V45c_nominal_0"},
         "final": lab, "mpc_end": lab, "walk_end": None, "robot_s": 25.0},
        {"file": "d.npz", "start": {"key": "nominal", "arm": "V48b", "id": "V48b_nominal_0"},
         "final": lab, "mpc_end": lab, "walk_end": None, "robot_s": 25.0}]}))


@check("Q5 stage-2 summary on synthetic run + collector dirs")
def check_summary() -> str:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "suite"
        _fake_run(root)
        a = argparse.Namespace(suite="stage2", dir=[str(root)], out=None, starts=None,
                               collector=[str(root / "collector")], stage2_src="V45c")
        with contextlib.redirect_stdout(io.StringIO()):
            rhs.cmd_summarize(a)
        rows = json.loads((root / "hard_suite.json").read_text())["rows"]
        md = (root / "hard_suite.md").read_text()
    by = {(r["arm"], r["variant"]): r for r in rows}
    _require(len(rows) == 4, f"{len(rows)} rows (3 MPPI + 1 scripted V45c)")
    ra, rb = by[("V49x", "V45c_nominal_0")], by[("V49x", "V45c_nominal_1")]
    _require(ra["engaged"] and ra["mpc_end"] == "early_stop:engaged"
             and abs(ra["rot_max_deg"] - 60) < 1e-6 and abs(ra["tilt_max_deg"]) < 1e-6,
             f"row a {ra}")
    _require(not rb["engaged"] and rb["mpc_end"] == "abort:grasp_lost"
             and rb["failure"] == "lost in the release (engaged/under)"
             and rb["post_aborts"] == ["unreachable"], f"row b {rb}")
    _require(by[("V49x", "V45c_nominal_2")]["failure"].startswith("skipped"), "skipped row")
    _require(by[("scripted:collector", "V45c_nominal_0")]["engaged"], "collector row")
    _require("| V49x | **1/3** | 2/3 |" in md and "| nominal | 1/3 (2) | 1/1 (1) |" in md,
             "md tables")
    return ("MPC end kinds, rotation / tilt max, failure modes, skipped + scripted rows, md "
            "tables ok")


def main() -> int:
    global PRE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre", type=Path, default=None,
                    help="reference run_hard_suite.py for Q1 (default: git HEAD's)")
    PRE = ap.parse_args().pre
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
    print(f"ALL STAGE-2 SUITE CHECKS PASSED ({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
