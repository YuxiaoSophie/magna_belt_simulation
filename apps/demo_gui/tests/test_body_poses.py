"""NEWTON_VIEWER_DRAW body poses (lcm-viewer PKG-01), CPU-only: no sim, GPU, lock or viewer.

    uv run --frozen python apps/demo_gui/tests/test_body_poses.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # harness imports init Warp: keep them off the GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import lcm
import numpy as np
from apps.demo_gui.config import Cfg, lcm_url

from task_common.lcm_contract import (  # importing task_common puts lcmtypes/ on sys.path
    VIEWER_DRAW_CHANNEL,
    LcmChannels,
    lcmt_viewer_draw,
    viewer_draw_body_q,
    viewer_draw_msg,
)

URL = lcm_url(7852)
FX = Path(__file__).resolve().parent / "fixtures/e2e_run"
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_body_poses_test_")


def _body_q(n: int) -> np.ndarray:
    q = np.random.default_rng(0).normal(size=(n, 7)).astype(np.float32)
    q[:, 3:] /= np.linalg.norm(q[:, 3:], axis=1, keepdims=True)
    return q


def test_round_trip():
    q, names = _body_q(5), [f"body_{i}" for i in range(5)]
    msg = lcmt_viewer_draw.decode(viewer_draw_msg(123456, names, q).encode())
    assert msg.timestamp == 123456 and msg.num_links == 5 and list(msg.robot_num) == [0] * 5
    assert np.array_equal(np.asarray(msg.quaternion)[:, 0], q[:, 6])  # w first on the wire
    got_names, got = viewer_draw_body_q(msg)
    assert got_names == names and got.dtype == np.float32 and np.array_equal(got, q)


def test_encode_ms():
    msg = viewer_draw_msg(0, [f"robot/link_{i:02d}" for i in range(86)], _body_q(86))
    t0 = time.perf_counter()
    for _ in range(100):
        data = msg.encode()
    ms = (time.perf_counter() - t0) * 10
    print(f"encode 86 bodies: {ms:.3f} ms, {len(data)} B")
    assert ms < 20


def test_meta_channels_unchanged():
    # recording meta.json "channels" = asdict(LcmChannels()): must match the pre-change fixture
    meta = json.loads((FX / "recordings/episode_pre_place_1_osc_0/meta.json").read_text())
    assert dataclasses.asdict(LcmChannels()) == meta["channels"]


def test_osc_bridge_passes_viewer_draw():
    from round_belt_task.osc_bridge import OscBridge

    ch = LcmChannels()
    bridge, sub, got = OscBridge(URL), lcm.LCM(URL), []
    for c in (VIEWER_DRAW_CHANNEL, ch.round_belt_pulley_state_channel):
        sub.subscribe(c, lambda c, d: got.append(c))
    bridge.publish(ch.round_belt_pulley_state_channel, viewer_draw_msg(0, ["a"], _body_q(1)))
    bridge.publish(VIEWER_DRAW_CHANNEL, viewer_draw_msg(0, ["a"], _body_q(1)))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and sub.handle_timeout(200) > 0:
        pass
    assert got == [VIEWER_DRAW_CHANNEL], got


def _build_kw(argv: list[str]) -> tuple[dict, dict]:
    """(index.json, RoundBeltOscSimulation.build kw) for ``argv``; the build is stubbed."""
    import eval_learned_mpc as h

    import round_belt_task.osc_simulation as o

    class Stop(Exception):
        pass

    seen = {}

    def stop(*a, **k):
        seen.update(k)
        raise Stop

    out = Path(tempfile.mkdtemp(dir=_TMP.name))
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
    return json.loads((out / "index.json").read_text()), seen


def test_harness_flag():
    import eval_learned_mpc as h

    assert "publish_body_poses" in h.STAGE2_ARG_DEFAULTS and "publish_body_poses" in h.VIEWER_ARGS
    assert "publish_body_poses" not in h.CHAIN_ARGS
    args = h.create_parser().parse_args(["--out", _TMP.name, "--publish-body-poses"])
    assert args.publish_body_poses and not args.stage2_chain
    assert h._chain_options(args, None) is None  # allowed without --stage2-chain
    assert h.create_parser().parse_args(["--out", _TMP.name]).publish_body_poses is False


def test_harness_build_kw_identity():
    argv = json.loads((FX / "recordings/episode_pre_place_1_osc_0/meta.json").read_text())["argv"]
    argv = argv[1:]
    argv[argv.index("--lcm-url") + 1] = Cfg().lcm_url
    off_index, off = _build_kw(argv)
    assert "publish_body_poses" not in off and "body_poses_every" not in off, off
    assert "publish_body_poses" not in off_index["args"]
    on_index, on = _build_kw([*argv, "--publish-body-poses", "--render-every", "4"])
    assert on["publish_body_poses"] is True and on["body_poses_every"] == 4
    assert on_index["args"]["publish_body_poses"] is True
    assert set(on) - set(off) == {"publish_body_poses", "body_poses_every"}



def test_realtime_flag():
    import eval_learned_mpc as h

    assert "realtime" in h.STAGE2_ARG_DEFAULTS and "realtime" in h.VIEWER_ARGS
    assert "realtime" not in h.CHAIN_ARGS
    args = h.create_parser().parse_args(["--out", _TMP.name, "--realtime"])
    assert args.realtime and h._chain_options(args, None) is None  # legal without the chain
    try:
        h.create_parser().parse_args(["--realtime", "--live-viewer"])
        raise AssertionError("--realtime --live-viewer accepted")
    except SystemExit as exc:
        assert exc.code == 2
    argv = json.loads((FX / "recordings/episode_pre_place_1_osc_0/meta.json").read_text())["argv"]
    argv = argv[1:]
    argv[argv.index("--lcm-url") + 1] = Cfg().lcm_url
    off_index, off = _build_kw(argv)
    on_index, on = _build_kw([*argv, "--realtime"])
    assert "realtime" not in off_index["args"] and on_index["args"]["realtime"] is True
    assert on == off


def test_pacer():
    """Absolute deadlines: early -> sleep to the deadline, late -> catch up, lag > cap -> rebase."""
    import types

    import eval_learned_mpc as h

    now, slept = [0.0], []

    def sleep(s):
        slept.append(round(s * 1e3, 6))
        now[0] += s

    real, h.time = h.time, types.SimpleNamespace(perf_counter=lambda: now[0], sleep=sleep)
    try:
        pacer = h.Pacer(0.005)
        for work_ms in (3, 12, 1, 1, 200, 1):  # each step's duration after its wait
            pacer.wait()
            now[0] += work_ms * 1e-3
        pacer.wait()
    finally:
        h.time = real
    # t=0 start; 3 -> sleep 2; 17 (due 10) none; 18 (15) none; 19 -> 1; 220 rebase; 221 -> 4
    assert slept == [2.0, 1.0, 4.0], slept


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
