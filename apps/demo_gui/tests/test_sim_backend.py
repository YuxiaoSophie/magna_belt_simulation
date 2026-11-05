"""Sim backend checks, CPU-only (no sim, GPU, lock, viewer or LCM socket); pytest-compatible.

    uv run --frozen python apps/demo_gui/tests/test_sim_backend.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # FK / harness imports init Warp: keep them off the GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from apps.demo_gui import jobs
from apps.demo_gui.backends import sim as simb
from apps.demo_gui.config import REPO, SCENE, Cfg
from apps.demo_gui.session import Session

FX = Path(__file__).resolve().parent / "fixtures"
REC = FX / "e2e_run/recordings/episode_pre_place_1_osc_0"
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_sim_test_")
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def _session() -> Session:
    return Session(tempfile.mkdtemp(dir=_TMP.name))


def _no_shared_port(argv: list[str]) -> None:
    j = " ".join(argv)
    assert "7667" not in j and "7857" not in j, j


def test_make_start_argv():
    s, cfg = _session(), Cfg()
    argv = simb.SimBackend(cfg).argv_for("make_start", session=s, sid="s00",
                                         franka={"slide_mm": 5, "roll_deg": -3, "pivot_deg": 0,
                                                 "depth_mm": -4},
                                         ur={"slide_mm": 0, "roll_deg": 2, "pivot_deg": -12})
    assert argv[4] == "apps/demo_gui/backend/make_start.py", argv
    assert argv[argv.index("--lcm-url") + 1] == cfg.lcm_url == "udpm://239.255.76.250:7850?ttl=0"
    assert argv[argv.index("--out") + 1] == str(s.starts)
    assert argv[argv.index("--franka-slide-mm") + 1] == "5.0"
    assert argv[argv.index("--ur-roll-deg") + 1] == "2.0"
    assert argv[argv.index("--franka-pivot-deg") + 1] == "0.0"
    assert argv[argv.index("--ur-pivot-deg") + 1] == "-12.0"
    assert argv[argv.index("--franka-depth-mm") + 1] == "-4.0"
    assert argv[argv.index("--ur-depth-mm") + 1] == "0.0"  # missing depth_mm (old store) = 0
    _no_shared_port(argv)


def test_record_demo_argv():
    s, cfg = _session(), Cfg()
    out = s.demos / "demo_00"
    argv = simb.SimBackend(cfg).argv_for("record_demo", session=s, start_id="s00", out=out)
    j = " ".join(argv)
    assert f"--lcm-url {cfg.lcm_url}" in j
    assert j.endswith("--render-every 3 --publish-body-poses --realtime"), j
    assert f"--out {out}" in j and f"--start-states {s.starts}/s00_osc.npz" in j
    assert "--label demo_gui_demo_00" in j and "--scripted-set hover_h_mm=36" in j
    assert argv[argv.index("--params") + 1] == (
        "../magna_belt_simulation-main/data/lcs/realfirst/cmp/v12_147/e2etune/yaml/set3_gv_07.yaml")
    _no_shared_port(argv)


def test_harness_accepts_demo_argv():
    """The demo argv parses and passes the chain checks with --publish-body-poses (scripted)."""
    import eval_learned_mpc as h

    s = _session()
    argv = simb.SimBackend(Cfg()).argv_for("record_demo", session=s, start_id="s00",
                                           out=s.demos / "demo_00")
    args = h.create_parser().parse_args(argv[5:])
    assert args.publish_body_poses and args.realtime and not args.live_viewer
    assert args.render_every == 3
    ch = h._chain_options(args, None)
    assert ch["scripted"] and "teleop" not in ch
    args = h.create_parser().parse_args(argv[5:-2])  # without --publish-body-poses --realtime
    assert not args.publish_body_poses and not args.realtime
    assert h._chain_options(args, None)["scripted"]


def test_record_stop_reset_with_fake_popen():
    s, cfg = _session(), Cfg(sim_port=7849)  # not the app's 7850: stop() scans the URL's procs
    (s.starts / "s00_osc.npz").touch()
    be = simb.SimBackend(cfg)
    calls = []

    class FakePopen:  # never runs flock / gpu_free.sh / the harness
        pid = 999999999

        def __init__(self, argv, **kw):
            calls.append((argv, kw))

    real, jobs.subprocess.Popen = jobs.subprocess.Popen, FakePopen
    real_rel, simb.CONTROLLER_REL = simb.CONTROLLER_REL, Path("/nonexistent/controller")
    try:
        try:
            be.record_demo(s, "s00")
            raise AssertionError("missing controller not refused")
        except FileNotFoundError:
            pass
        simb.CONTROLLER_REL = Path(sys.executable)  # stands in for an existing binary
        for bad in (("s01", "scripted"), ("s00", "bogus")):
            try:
                be.record_demo(s, *bad)
                raise AssertionError(bad)
            except (FileNotFoundError, ValueError):
                pass
        job = be.record_demo(s, "s00")
    finally:
        jobs.subprocess.Popen, simb.CONTROLLER_REL = real, real_rel
    (demo, dkw), (tel, tkw) = calls
    assert demo[:2] == ["flock", str(jobs.LOCK)] and "eval_learned_mpc.py" in demo[4], demo
    assert dkw["env"]["ROUND_BELT_SCENE"] == str(SCENE)
    assert "--url udpm://239.255.76.249:7849?ttl=0" in " ".join(tel) and "flock" not in tel
    assert tel[tel.index("--until-pid") + 1] == str(job.pid)
    assert tkw["env"]["CUDA_VISIBLE_DEVICES"] == "" and job.lcm_url == cfg.lcm_url
    assert (s.demos / "demo_00").is_dir()
    be.stop_demo(s)
    be.reset_demo(s)
    assert not (s.demos / "demo_00").exists()


def test_make_start_dry_run():
    with tempfile.TemporaryDirectory(dir=_TMP.name) as d:
        p = subprocess.run(
            [sys.executable, "apps/demo_gui/backend/make_start.py", "--out", d, "--id", "s03",
             "--franka-slide-mm", "7.5", "--ur-roll-deg", "-4", "--franka-pivot-deg", "9",
             "--franka-depth-mm", "-2.5", "--ur-depth-mm", "3",
             "--lcm-url", Cfg().lcm_url,
             "--dry-run"], check=False, cwd=REPO, env=ENV, capture_output=True, text=True,
            timeout=300)
        assert p.returncode == 0, p.stderr[-2000:]
        plan = json.loads(p.stdout[p.stdout.index("{"):])
        assert plan["variant"] == {"id": "s03", "franka_slide_mm": 7.5, "franka_roll_deg": 0.0,
                                   "ur_slide_mm": 0.0, "ur_roll_deg": -4.0, "ur_depth_mm": 3.0,
                                   "ur_pivot_deg": 0.0, "franka_pivot_deg": 9.0,
                                   "franka_depth_mm": -2.5}
        assert plan["file"] == f"{d}/s03_osc.npz" and not list(Path(d).iterdir())


def test_make_start_index_row():
    from apps.demo_gui.backend import make_start

    d = Path(tempfile.mkdtemp(dir=_TMP.name))
    row = {"id": "s00", "commanded": {}, "held": False, "reason": "osc: x", "measured": None,
           "file": None, "notes": None}
    make_start.append_row(d, row)
    make_start.append_row(d, {**row, "id": "s01", "held": True, "file": "s01_osc.npz"})
    idx = json.loads((d / "index.json").read_text())
    assert idx["set"] == "demo_gui" and [r["id"] for r in idx["variants"]] == ["s00", "s01"]


def test_collect_panel_with_sim_rows():
    """A not-held start shows its reason and is skipped by Record; no-cloud telemetry plots."""
    from apps.demo_gui.app import Ctx
    from apps.demo_gui.backend import make_start
    from apps.demo_gui.panels import collect

    s = _session()
    row = {"id": "s00", "commanded": {}, "held": True, "reason": None, "measured": None,
           "file": "s00_osc.npz", "notes": None}
    make_start.append_row(s.starts, row)
    make_start.append_row(s.starts, {**row, "id": "s01", "held": False, "reason": "osc: held"})
    ctx = Ctx(simb.SimBackend(Cfg()), s, Cfg())
    assert "s01 not held: osc: held" in collect.on_status(ctx)[2]
    assert collect._ranges({"cloud": None, "belt_xyz": None, "franka_pose7": None}) == [[-1, 1]] * 3
    assert len(collect._ranges({"franka_pose7": [0.1, 0.2, 0.3, 1, 0, 0, 0]})) == 3


def _meta():
    return json.loads((REC / "meta.json").read_text())


def test_telemetry_decoders_on_recorded_state():
    """encode (the sim's own message builders) -> decode -> the episode's sim_ee / widths."""
    from apps.demo_gui import telemetry_lcm as tl

    from task_common.lcm_contract import robot_output_msg, robotiq_status_msg, schunk_status_msg

    io = _meta()["robot_io"]
    c = np.load(REC / "chunks/chunk_00000.npz")
    z = np.load(FX / "e2e_run/episode_pre_place_1_osc_0.npz", allow_pickle=True)
    common = np.intersect1d(c["step"], z["sim_step"])
    assert len(common)
    for step in common:
        i, j = int(np.flatnonzero(c["step"] == step)[0]), int(np.flatnonzero(z["sim_step"] == step)[0])
        q, eff = c["joint_q"][i].astype(float), c["efforts"][i].astype(float)
        for arm, key in (("franka", "franka"), ("ur10", "ur")):
            spec = io[arm]
            msg = robot_output_msg(int(step) * 2000, spec["position_names"], q[spec["coords"]],
                                   spec["velocity_names"], np.zeros(len(spec["dofs"])),
                                   spec["effort_names"], np.zeros(len(spec["dofs"])))
            got = tl.decode(key, msg.encode())
            assert np.abs(np.array(got[f"{key}_pose7"]) - z[f"sim_ee_{key}"][j]).max() < 2e-4
        hand = io["franka_hand"]
        msg = robot_output_msg(0, hand["position_names"], q[hand["coords"]],
                               hand["velocity_names"], [0, 0], hand["effort_names"], eff[7:9])
        assert abs(tl.decode("hand_out", msg.encode())["franka_grip_n"]
                   - np.abs(eff[7:9]).sum()) < 0.01
        w = tl.decode("hand", schunk_status_msg(0, q[hand["coords"]], [0, 0]).encode())
        assert abs(w["franka_width_mm"] - np.abs(q[hand["coords"]]).sum() * 1e3) < 0.01
        byte = int(c["robotiq_status"][i][0])
        ur = tl.decode("robotiq", robotiq_status_msg(0, byte / 255.0, 0.0, 0).encode())
        assert abs(ur["ur_width_mm"] - tl.ur_open_gap_m() * 1e3 * (1 - byte / 255)) < 0.01
        assert 0.0 <= ur["ur_width_mm"] <= 70.0


def test_telemetry_cloud_and_handler():
    from apps.demo_gui import telemetry
    from apps.demo_gui import telemetry_lcm as tl

    from task_common.lcm_contract import robot_output_msg
    from task_common.perception_lcm import point_cloud_msg

    xyz = np.random.default_rng(0).uniform(-0.3, 0.3, (2000, 3)).astype(np.float32)
    got = tl.decode("cloud", point_cloud_msg(0, xyz, np.zeros((2000, 3), np.uint8)).encode())
    assert 0 < len(got["cloud"]) <= telemetry.MAX_CLOUD
    assert np.allclose(got["cloud"][1], xyz[4], atol=1e-4)  # stride ceil(2000 / 544) = 4

    io = _meta()["robot_io"]["franka"]
    t = tl.LcmTelemetry(Cfg().lcm_url, Path(_TMP.name) / "telemetry.json", run_dir="d")
    for utime in (5_000_000, 5_250_000):
        msg = robot_output_msg(utime, io["position_names"], np.zeros(7), io["velocity_names"],
                               np.zeros(7), io["effort_names"], np.zeros(7))
        t.handle("franka", "FRANKA_STATE", msg.encode())
    t.handle("franka", "FRANKA_STATE", b"not an lcm message")
    t.handle("robotiq", "ROBOTIQ_STATUS", b"\x00")
    f = t.frame()
    assert t.received == 2 and t.dropped == 2
    assert f["t"] == 0.25 and f["step"] == 2 and len(f["franka_pose7"]) == 7
    assert f["source"] == "sim" and f["run_dir"] == "d" and f["belt_xyz"] is None
    assert set(f) >= {"t", "step", "phase", "franka_pose7", "ur_pose7", "franka_width_mm",
                      "franka_grip_n", "ur_width_mm", "belt_xyz", "cloud", "held"}


def test_harness_help_lists_viewer_flags():
    p = subprocess.run([sys.executable, "scripts/lcs/eval_learned_mpc.py", "--help"], check=False,
                       cwd=REPO, env=ENV, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 and "--publish-body-poses" in p.stdout and "--live-viewer" in p.stdout
    assert "--realtime" in p.stdout


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
