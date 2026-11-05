#!/usr/bin/env python3
"""Checks for the stage-2 SpaceMouse teleop (V12-120): flag defaults / refusals, ``--pre-dir`` parse
identity, integrator / clamp / hook units, launcher; with ``--sim`` a fake-SpaceMouse smoke through
the whole chain (port 7841, under the sim lock) and the flags-off identity run.

Run:
    uv run --frozen python scripts/checks/check_teleop_stage2.py [--pre-dir DIR_WITH_PRE_eval_learned_mpc.py] [--sim]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import check_full_run as cfr
import eval_learned_mpc as harness
import run_hard_suite as rhs
import stage2_teleop as st
import teleop_stage2 as launcher
from stage2_collect import cap_tilt, tilt_deg

from round_belt_task.commander import quat_to_mat3

PORT = 7841
LOCK = REPO_ROOT / "data/lcs/realfirst/cmp/v12_79/sim.lock"
SEATED = ("data/lcs/realfirst/cmp/v12_106/scan_wr625b15/V45c_wr625b15_nominal/end_states/"
          "episode_pre_place_1_osc_0.npz")
OFF = REPO_ROOT / "data/lcs/realfirst/cmp/v12_108"
SIM_TIMEOUT_S = 1500
PRE_DIR: Path | None = None
RUN_SIM = False
CHECKS: list[tuple[str, callable]] = []
_require = cfr._require

NEW_DEFAULTS = {"stage2_teleop": False, "teleop_lin_mps": 0.05, "teleop_ang_rps": 0.1666,
                "teleop_tilt_deg": 29.5, "teleop_floor_mm": 2.0, "teleop_max_s": 300.0,
                "teleop_wait_s": 10.0, "teleop_stale_s": 0.5, "live_viewer": False,
                "viewer_port": 8080, "render_every": 20,
                "teleop_channel": "SPACE_MOUSE_FRANKA_STATE"}
NEW_FLAGS = {"--teleop-lin-mps": "0.08", "--teleop-ang-rps": "0.2", "--teleop-tilt-deg": "20",
             "--teleop-floor-mm": "4", "--teleop-max-s": "8", "--teleop-wait-s": "3",
             "--teleop-stale-s": "0.2", "--viewer-port": "8090", "--render-every": "10",
             "--teleop-channel": "X", "--live-viewer": None}
BASE = [*cfr.MPPI, *cfr.FULL_PP]


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _opts(argv: list[str]):
    a = harness.create_parser().parse_args(argv)
    return a, harness._mppi_options(a, False)


def _refused(argv: list[str], exc_t, needle: str = "") -> None:
    a, m = _opts(argv)
    try:
        harness._chain_options(a, m)
    except exc_t as exc:
        _require(needle in str(exc), f"{argv[-2:]}: {exc}")
    else:
        raise AssertionError(f"accepted: {argv[-3:]}")


@check("T1 teleop flags: in STAGE2_ARG_DEFAULTS and CHAIN_ARGS, default when absent; CMD codes")
def check_defaults() -> str:
    d = vars(harness.create_parser().parse_args(cfr.MPPI))
    for k, v in NEW_DEFAULTS.items():
        _require(k in harness.STAGE2_ARG_DEFAULTS and k in harness.CHAIN_ARGS, f"{k} not registered")
        _require(harness.STAGE2_ARG_DEFAULTS[k] == v and d[k] == v, f"{k}: {d[k]} != {v}")
    _require(harness.CMD_TELEOP == 9, f"CMD_TELEOP {harness.CMD_TELEOP}")
    codes = {k: v for k, v in vars(harness).items() if k.startswith("CMD_") and isinstance(v, int)}
    _require(len(set(codes.values())) == len(codes) and not {4, 5, 7} & set(codes.values()),
             f"codes {codes}")
    return f"{len(NEW_DEFAULTS)} flags default-identical; CMD_TELEOP 9, {len(codes)} codes distinct"


@check("T2 refusals: teleop flags need --stage2-chain (and --stage2-teleop); triple; options")
def check_refusals() -> str:
    n = 0
    for flag, val in NEW_FLAGS.items():
        _refused([*BASE, flag, *([val] if val else [])], ValueError, "only with --stage2-chain")
        n += 1
    _refused([*BASE, "--stage2-teleop"], ValueError, "only with --stage2-chain")
    for flag, val in NEW_FLAGS.items():  # chain without --stage2-teleop
        _refused([*BASE, *cfr.CHAIN, flag, *([val] if val else [])], ValueError,
                 "only with --stage2-teleop")
        n += 1
    _refused([*BASE, *cfr.CHAIN, "--stage2-teleop"], ValueError, "no stage-2 node")
    _refused([*BASE, "--stage2-chain", "--stage2-teleop", "--stage2-deploy", cfr.S2[0]],
             ValueError, "no stage-2 node")
    _refused([*BASE, "--stage2-chain"], FileNotFoundError)
    a, m = _opts([*BASE, "--stage2-chain", "--stage2-teleop"])
    ch = harness._chain_options(a, m)
    _require(isinstance(ch, dict) and ch["teleop"] is True and ch["stage_labels"] == ["teleop"],
             f"teleop options {ch}")
    return f"{n + 4} refusals; teleop options dict ok"


@check("T3 --pre-dir parse identity: pre vs new parser on 3 argv sets")
def check_pre_parse() -> str:
    if PRE_DIR is None:
        return "SKIP: no --pre-dir"
    spec = importlib.util.spec_from_file_location("elm_pre", PRE_DIR / "eval_learned_mpc.py")
    pre = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pre  # dataclasses look the module up
    spec.loader.exec_module(pre)
    sets = {"stage-1 suite line": [*cfr.MPPI, *cfr.FULL_PP], "chain": [*BASE, *cfr.CHAIN],
            "plain mppi": cfr.MPPI}
    for what, argv in sets.items():
        a, b = (vars(m.create_parser().parse_args(argv)) for m in (pre, harness))
        # defaults rooted at the module's own REPO_ROOT differ when pre-dir is elsewhere
        a = {k: REPO_ROOT / v.relative_to(pre.REPO_ROOT) if isinstance(v, Path)
             and v.is_relative_to(pre.REPO_ROOT) else v for k, v in a.items()}
        for k in NEW_DEFAULTS:
            b.pop(k)
        _require(a == b, f"{what}: {sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))}")
    _require(all(harness.STAGE2_ARG_DEFAULTS.get(k) == v
                 for k, v in pre.STAGE2_ARG_DEFAULTS.items()),
             "STAGE2_ARG_DEFAULTS not a sub-dict")
    return f"{len(sets)} argv sets identical vs {PRE_DIR}; STAGE2_ARG_DEFAULTS superset; " \
           "SKIP phase_labels (needs a sim)"


class _FakeSim:
    frame_dt = 0.005

    def __init__(self, pose7) -> None:
        self.pose7 = pose7

    def franka_measured_pose7(self, joint_q):
        return self.pose7.copy()


class _FakeInput:
    def __init__(self, age: float, v) -> None:
        self.age, self.v = age, np.asarray(v, np.float64)

    def velocity(self, now_wall, stale_s):
        return (np.zeros(3), np.zeros(3)) if self.age > stale_s else (self.v, np.zeros(3))


@check("T4 integrator / clamp / TeleopHook units")
def check_units() -> str:
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    p = np.concatenate([[0.4, 0.0, 0.1], q0])
    a = st.integrate(p, [0.05, 0, 0], [0, 0, 0], 1.0)
    _require(np.allclose(a[:3] - p[:3], [0.05, 0, 0], atol=1e-12) and np.allclose(a[3:], q0),
             f"linear {a}")
    b = np.array([0, 0, 0, 1.0, 0, 0, 0])
    for _ in range(200):
        b = st.integrate(b, [0, 0, 0], [0, 0, math.pi / 2], 0.005)
    _require(np.allclose(quat_to_mat3(b[3:])[:, 0], [0, 1, 0], atol=1e-9), "yaw pi/2: x -> y")
    c = st.clamp([0.4, 0, -0.1, *q0], p * [1, 1, 0, 1, 1, 1, 1], 0.0, 0.002, 29.5)
    _require(abs(c[2] - 0.002) < 1e-12, f"floor {c[2]}")
    tilted = st.integrate(p, [0, 0, 0], [0, math.radians(45), 0], 1.0)  # 45 deg about y
    _require(abs(tilt_deg(tilted[3:]) - 45.0) < 1e-6, f"test pose tilt {tilt_deg(tilted[3:])}")
    meas = np.concatenate([tilted[:3], cap_tilt(tilted[3:], 29.5)])  # lead clamp runs last
    c = st.clamp(tilted, meas, -1.0, 0.002, 29.5)
    _require(tilt_deg(c[3:]) <= 29.5 + 1e-6, f"tilt {tilt_deg(c[3:])}")
    c = st.clamp([0.5, 0, 0.1, *q0], p, -1.0, 0.002, 29.5)
    _require(abs(np.linalg.norm(c[:3] - p[:3]) - 0.02) < 1e-9, f"lead {c[:3] - p[:3]}")
    lim = {"floor_m": 0.002, "tilt_deg": 29.5, "lead_m": 0.02, "lead_deg": 15.0, "stale_s": 0.5}
    v = np.array([0.0, 0.0, 0.05])
    hook = st.TeleopHook(_FakeSim(p), _FakeInput(0.0, v), lim, -1.0, [False])
    hook(1, 1.0, None, None)
    k = hook.last_knots
    _require(len(k) == 7 == harness.N_KNOTS, f"{len(k)} knots")
    _require(np.allclose(k[1, :3] - k[0, :3], v * 0.075, atol=1e-9), f"knot step {k[1, :3] - k[0, :3]}")
    stale = st.TeleopHook(_FakeSim(p), _FakeInput(1.0, v), lim, -1.0, [False])
    stale(1, 1.0, None, None)
    _require(not stale.v.any() and not stale.w.any(), "stale input not zero")
    return "integrate +50 mm, yaw x->y, floor, tilt <= 29.5, lead 20 mm, hook 7 knots, stale zero"


@check("T5 launcher --dry-run: argv per model / start; --port 7667 refused")
def check_launcher() -> str:
    def run(*args: str):
        return subprocess.run([sys.executable, str(REPO_ROOT / "scripts/lcs/teleop_stage2.py"),
                               *args], capture_output=True, text=True, check=False)

    for model, start, reset in (("V45c", "nominal", None), ("V55b", "set2_gv_00", True)):
        r = run("--dry-run", "--model", model, "--start", start)
        _require(r.returncode == 0, f"{model} rc {r.returncode}")
        for s in ("--stage2-chain --stage2-teleop --live-viewer", *launcher.MODELS[model],
                  launcher.SCENE):
            _require(s in r.stdout, f"{model}: lacks {s}")
        has = "--reset scripted --reset-spec slack_mm=10 seed=20261008" in r.stdout
        _require(has == bool(reset), f"{model} scripted reset {has}")
    _require(run("--dry-run", "--port", "7667").returncode != 0, "--port 7667 accepted")
    return "V45c nominal, V55b set2_gv_00 argv ok; 7667 refused"


def _publisher(url: str, log: Path, stop: threading.Event) -> threading.Thread:
    """Fake SpaceMouse at 100 Hz; z profile +1 2 s, -1 2 s, 0 starts at the log's '[TELEOP] keys'."""
    import lcm

    from spacemouse import lcmt_spacemouse_state

    channel = harness.TELEOP_ARG_DEFAULTS["teleop_channel"]

    def loop() -> None:
        lc, t_trig, t_check = lcm.LCM(url), None, 0.0
        while not stop.is_set():
            now = time.time()
            if t_trig is None and now - t_check > 0.5:
                t_check = now
                if log.is_file() and "[TELEOP] keys" in log.read_text(errors="ignore"):
                    t_trig = now
            tau = -1.0 if t_trig is None else now - t_trig
            m = lcmt_spacemouse_state()
            m.utime, m.n_buttons = int(now * 1e6), 2
            m.linear = [0.0, 0.0, 1.0 if 0 <= tau < 2 else -1.0 if 2 <= tau < 4 else 0.0]
            m.angular, m.button_pressed = [0.0, 0.0, 0.0], [False, False]
            lc.publish(channel, m.encode())
            time.sleep(0.01)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th


def _sim(argv: list[str], log: Path, env: dict, publish: bool) -> int:
    """argv under the sim lock in its own process group; SIGINT to the group on timeout."""
    stop = threading.Event()
    th = _publisher(rhs.url(PORT), log, stop) if publish else None
    try:
        with log.open("w") as f:
            proc = subprocess.Popen(["flock", str(LOCK), *argv], cwd=REPO_ROOT, env=env, stdout=f,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            try:
                return proc.wait(timeout=SIM_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=60)
                return -1
    finally:
        stop.set()
        if th:
            th.join(timeout=5)


def _runs(a) -> list[int]:
    return [int(k) for k, _ in itertools.groupby(a)]


def _teleop_run(td: Path, extra: list[str], tag: str) -> tuple[Path, Path]:
    out = td / tag
    start = next(s for s in rhs.SUITE if s[0] == "nominal")
    argv = launcher.harness_argv("V45c", start, PORT, str(out / "V45c_nominal"), 25.0, extra)
    argv = [x for x in argv if x != "--live-viewer"]
    log = td / f"{tag}.log"
    env = {**os.environ, "ROUND_BELT_SCENE": launcher.SCENE}
    rc = _sim(argv, log, env, publish=True)
    _require(rc == 0, f"{tag} rc {rc}, log {log}")
    return out / "V45c_nominal", log


@check("S1 sim smoke: fake SpaceMouse through the whole chain; npz / index layout")
def check_smoke() -> str:
    if not RUN_SIM:
        return "SKIP: no --sim"
    from task_common import lcs_dataset as lcs

    td = Path(tempfile.mkdtemp(prefix="teleop_s1_"))
    tele = ["--teleop-max-s", "8", "--teleop-wait-s", "5"]
    t0 = time.perf_counter()
    run, log = _teleop_run(td, tele, "a")
    idx = json.loads((run / "index.json").read_text())
    if "teleop" not in idx["episodes"][0].get("chain", {}):  # stage 1 failed: seated start
        run, log = _teleop_run(td, [*tele, "--start-states", SEATED, "--max-episode-s", "0.1"], "b")
        idx = json.loads((run / "index.json").read_text())
    where = f"(run {run}, log {log})"
    row = idx["episodes"][0]
    z = np.load(run / row["file"], allow_pickle=True)
    cs = z["sim_cmd_source"]
    runs = _runs(cs)
    _require(runs[-3:] == [8, 9, 6] and set(runs[:-3]) <= {0, 1, 2}, f"cmd runs {runs} {where}")
    t9 = np.flatnonzero(cs == 9)
    _require(int(z["sim_chain_from"][1]) == t9[0], f"chain_from {z['sim_chain_from']} {where}")
    _require(int(z["sim_post_from"]) == np.flatnonzero(cs == 6)[0], f"post_from {where}")
    az = z["actions"][t9, 2] * 1e3
    up, dn = np.flatnonzero(az > 0.1), np.flatnonzero(az < -0.1)
    _require(len(up) and len(dn) and up.max() < dn.min(), f"z profile +/- order {where}")
    _require(2 <= az[up].mean() <= 5 and 2 <= -az[dn].mean() <= 5,
             f"|mean| up {az[up].mean():.2f} dn {az[dn].mean():.2f} mm {where}")
    d = z["sim_cmd_knot1_franka"][t9, :3] - z["sim_cmd_knot0_franka"][t9, :3]
    _require(np.allclose(d, z["actions"][t9, :3]), f"knot1 - knot0 != actions {where}")
    _require(np.isfinite(z["sim_realised_delta"][t9[:-1]]).all(), f"realised delta {where}")
    lab = idx["phase_labels"]
    _require({lab[p] for p in z["sim_phase"][t9]} == {"s2:teleop"}, f"phase labels {where}")
    ch = row["chain"]
    _require(ch["teleop"]["operator"] == "timeout" and isinstance(ch["success_both"], bool),
             f"chain block {where}")
    _require(len(row["post_mpc"]["phases"]) == 4, f"post_mpc phases {where}")
    _require((run / row["recording"] / "meta.json").is_file(), f"recording {where}")
    lcs.validate_episode(run / row["file"], period_us=round(15 * lcs.SIM_DT_S * 1e6))
    return (f"cmd runs {runs}, {len(t9)} teleop frames, z +{az[up].mean():.2f}/{az[dn].mean():.2f} "
            f"mm, validate ok, {time.perf_counter() - t0:.0f} s")


def _cli(args: dict, **over) -> list[str]:
    args, argv = {**args, **over}, ["uv", "run", "--frozen", "python",
                                    "scripts/lcs/eval_learned_mpc.py"]
    for k, v in args.items():
        if v is None or v is False or v == []:
            continue
        argv += [f"--{k.replace('_', '-')}", *([] if v is True else [str(v)])]
    return argv


@check("S2 flags-off sim identity vs v12_108/smoke_off_pre (index, labels, npz key set)")
def check_flags_off() -> str:
    if not RUN_SIM:
        return "SKIP: no --sim"
    pre = OFF / "smoke_off_pre"
    if not (OFF / "smoke_off/index.json").is_file() or not (pre / "index.json").is_file():
        return "SKIP: v12_108/smoke_off{,_pre} not present"
    i_pre = json.loads((pre / "index.json").read_text())
    td = Path(tempfile.mkdtemp(prefix="teleop_s2_"))
    argv = _cli(json.loads((OFF / "smoke_off/index.json").read_text())["args"],
                lcm_url=rhs.url(PORT), out=td / "off", max_episode_s=2, label="V12-120_check_off")
    env = {k: v for k, v in os.environ.items() if k != "ROUND_BELT_SCENE"}
    rc = _sim(argv, td / "off.log", env, publish=False)
    _require(rc == 0, f"rc {rc}, log {td / 'off.log'}")
    on = td / "off"
    i_on = json.loads((on / "index.json").read_text())
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
        n = len(a.files)
    return f"index.args ({len(i_on['args'])} keys), index keys, labels, {n} npz keys identical"


def main() -> int:
    global PRE_DIR, RUN_SIM
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="dir with the pre-change eval_learned_mpc.py (T3)")
    ap.add_argument("--sim", action="store_true", help="also S1 / S2 (sim on port 7841, locked)")
    a = ap.parse_args()
    PRE_DIR, RUN_SIM = a.pre_dir, a.sim
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
    print(f"ALL TELEOP CHECKS PASSED ({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
