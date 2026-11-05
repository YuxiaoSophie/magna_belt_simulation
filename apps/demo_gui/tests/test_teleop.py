"""Teleop demo checks (PKG-04), CPU-only: no sim, GPU, lock, viewer, SpaceMouse or LCM socket.

    uv run --frozen python apps/demo_gui/tests/test_teleop.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # harness imports init Warp: keep them off the GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.demo_gui import jobs
from apps.demo_gui.backends import sim as simb
from apps.demo_gui.config import REPO, Cfg
from apps.demo_gui.session import Session

FX = Path(__file__).resolve().parent / "fixtures/e2e_run"
START = REPO / "data/lcs/start_states/grasp_variants/set3/gv_07_osc.npz"
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_teleop_test_")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def _session() -> Session:
    return Session(tempfile.mkdtemp(dir=_TMP.name))


def _teleop_argv(s: Session, cfg: Cfg | None = None) -> list[str]:
    return simb.SimBackend(cfg or Cfg()).argv_for("record_demo", session=s, start_id="s00",
                                                  out=s.demos / "demo_00", mode="teleop")


def test_keys_cmd_file():
    import stage2_teleop

    f = Path(_TMP.name) / "cmd.json"
    f.write_text("D")
    with stage2_teleop.Keys(f) as keys:
        assert keys.pop() == "d" and f.read_text() == "" and keys.pop() is None
    missing = Path(_TMP.name) / "none.json"
    with stage2_teleop.Keys(missing) as keys:
        assert keys.pop() is None and not missing.exists()


def test_teleop_argv():
    s, cfg = _session(), Cfg(magna_root=Path("/opt/magna"))
    argv = _teleop_argv(s, cfg)
    j = " ".join(argv)
    assert "--teleop-stage1 --stage2-teleop" in j and f"--teleop-cmd-file {s.cmd_path}" in j, j
    assert "--stage2-scripted" not in j and "--scripted-set" not in j
    assert f"--lcm-url {cfg.lcm_url}" in j
    assert j.endswith("--render-every 3 --publish-body-poses --realtime"), j
    assert argv[argv.index("--magna-root") + 1] == "/opt/magna"
    assert argv[argv.index("--mode") + 1] == "baseline"
    scripted = simb.SimBackend(cfg).argv_for("record_demo", session=s, start_id="s00",
                                             out=s.demos / "demo_00")
    assert "--teleop-stage1" not in scripted and "--stage2-scripted" in scripted


def test_harness_chain_options():
    import eval_learned_mpc as h

    s = _session()
    args = h.create_parser().parse_args(_teleop_argv(s)[5:])
    assert args.teleop_stage1 and args.teleop_cmd_file == s.cmd_path
    ch = h._chain_options(args, None)
    assert ch["teleop"] and ch["teleop_s1"] and ch["stage_labels"] == ["teleop"]
    argv = _teleop_argv(s)[5:]
    for drop, add in ((["--stage2-teleop"], []), ([], ["--stage2-scripted"]),
                      (["--teleop-stage1"], ["--stage2-scripted"])):
        a = h.create_parser().parse_args([x for x in argv if x not in drop] + add)
        try:
            h._chain_options(a, None)
            raise AssertionError((drop, add))
        except ValueError:
            pass
    a = h.create_parser().parse_args([x for x in argv if x != "--teleop-stage1"])
    try:  # --stage2-teleop needs an MPPI stage 1 unless --teleop-stage1
        h._chain_options(a, None)
        raise AssertionError("teleop stage 2 after a baseline stage 1 accepted")
    except ValueError:
        pass


def test_harness_help_lists_teleop_flags():
    p = subprocess.run([sys.executable, "scripts/lcs/eval_learned_mpc.py", "--help"], check=False,
                       cwd=REPO, env=ENV, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "--teleop-stage1" in p.stdout and "--teleop-cmd-file" in p.stdout


def _index(argv: list[str], out: Path) -> dict:
    """The harness's index.json for ``argv`` (sim build stubbed: stops right after writing it)."""
    import eval_learned_mpc as h

    import round_belt_task.osc_simulation as o

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    argv = list(argv)
    argv[argv.index("--out") + 1] = str(out)
    real, o.RoundBeltOscSimulation.build = o.RoundBeltOscSimulation.build, stop
    try:
        h.evaluate(h.create_parser().parse_args(argv))
        raise AssertionError("sim build not reached")
    except Stop:
        pass
    finally:
        o.RoundBeltOscSimulation.build = real
    return json.loads((out / "index.json").read_text())


def test_recorded_argv_index_identity():
    """The fixture run's argv still writes its recorded index args / chain flags (no new keys)."""
    ref = json.loads((FX / "index.json").read_text())
    argv = json.loads((FX / "recordings/episode_pre_place_1_osc_0/meta.json").read_text())["argv"]
    argv = argv[1:]
    argv[argv.index("--lcm-url") + 1] = Cfg().lcm_url
    out = Path(tempfile.mkdtemp(dir=_TMP.name))
    got = _index(argv, out)
    want = {**ref["args"], "out": str(out), "lcm_url": Cfg().lcm_url}
    assert json.dumps(got["args"]) == json.dumps(want), set(got["args"]) ^ set(want)
    assert got["chain"]["flags"] == ref["chain"]["flags"]
    assert got["phase_labels"] == ref["phase_labels"]
    assert "teleop_stage1" not in json.dumps(got) and "teleop_cmd_file" not in json.dumps(got)


def test_teleop_argv_index():
    s = _session()
    argv = _teleop_argv(s)[5:-2]  # no viewer / realtime flag needed
    argv[argv.index("--start-states") + 1] = str(START)
    got = _index(argv, Path(tempfile.mkdtemp(dir=_TMP.name)))
    assert got["args"]["teleop_stage1"] is True
    assert got["args"]["teleop_cmd_file"] == str(s.cmd_path)
    assert got["chain"]["flags"]["teleop_stage1"] is True
    assert "stage1" in got["chain"]["teleop"]
    labels = got["phase_labels"]
    assert labels.index("s1:teleop") + 1 == labels.index("ur:open") and "s2:teleop" in labels


def test_record_demo_teleop_with_fake_popen():
    s, cfg = _session(), Cfg(sim_port=7849, magna_root=Path("/nonexistent/magna"))
    (s.starts / "s00_osc.npz").touch()
    s.cmd_path.write_text("d")  # stale key from an earlier run
    be = simb.SimBackend(cfg)
    calls = []

    class FakePopen:  # never runs flock / gpu_free.sh / the harness
        pid = 999999999

        def __init__(self, argv, **kw):
            calls.append(argv)

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    real, jobs.subprocess.Popen = jobs.subprocess.Popen, FakePopen
    try:
        try:
            be.record_demo(s, "s00")
            raise AssertionError("missing baseline controller not refused")
        except FileNotFoundError as e:
            assert "/nonexistent/magna/bazel-bin/systems/controllers/" in str(e)
        assert not calls
        be.record_demo(s, "s00", mode="teleop")  # no controller in a teleop stage 1
        assert s.cmd_path.read_text() == ""
        demo = calls[0]
        assert "--teleop-stage1" in demo[4] and str(s.cmd_path) in demo[4], demo
        be.teleop_key(s, "p")
        be.stop_demo(s)  # still stage 1: no d sent, the job is stopped
        assert s.cmd_path.read_text() == "p"
        be.teleop_key(s, "d")
        s.cmd_path.write_text("")  # the harness popped it
        be.stop_demo(s)  # stage 2: Stop = d, then wait for the job
        assert s.cmd_path.read_text() == "d"
        be.reset_demo(s)
        assert s.cmd_path.read_text() == "a" and not (s.demos / "demo_00").exists()
    finally:
        jobs.subprocess.Popen = real


def test_collect_teleop_buttons():
    from apps.demo_gui.app import Ctx
    from apps.demo_gui.backends.mock import MockBackend
    from apps.demo_gui.panels import collect

    s, cfg = _session(), Cfg()
    ctx = Ctx(simb.SimBackend(cfg), s, cfg)
    page = str(collect.layout(ctx))
    assert "col-tdone" in page and "col-tpause" in page and cfg.lcm_url in page
    assert "--gripper_command_lcm_channel=TELEOP_HAND_IGNORED" in collect.driver_command("u")
    assert collect.on_action(ctx, "col-tdone", None).startswith("sent d")
    assert s.cmd_path.read_text() == "d"
    assert collect.on_action(ctx, "col-tpause", None).startswith("sent p")
    mock = Ctx(MockBackend(), s, cfg)
    assert "not on this backend" in collect.on_action(mock, "col-tdone", None)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
