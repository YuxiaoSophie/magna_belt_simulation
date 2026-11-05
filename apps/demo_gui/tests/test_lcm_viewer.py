"""Standalone LCM viewer (lcm-viewer PKG-02), CPU-only: name matching, bad packets, and an
in-process viewer on an ephemeral port draining 50 body-pose messages (newest wins).

    CUDA_VISIBLE_DEVICES= uv run --frozen python apps/demo_gui/tests/test_lcm_viewer.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import lcm
import numpy as np
from apps.demo_gui.config import lcm_url
from lcm_viewer import BodyPoseMap, LcmViewer

from task_common.lcm_contract import VIEWER_DRAW_CHANNEL, viewer_draw_msg

URL = lcm_url(7852)


def test_body_pose_map():
    body_q = np.zeros((3, 7), dtype=np.float32)
    msg_q = np.arange(21, dtype=np.float32).reshape(3, 7) + 1
    assert BodyPoseMap(["a", "b", "c"]).apply(["c", "zzz", "a"], msg_q, body_q) == 2
    assert np.array_equal(body_q[2], msg_q[0]) and np.array_equal(body_q[0], msg_q[2])
    assert not body_q[1].any()


def test_cli():
    script = str(REPO / "scripts/lcm_viewer.py")
    r = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True,
                       check=False)
    flags = {w.strip("[],") for w in r.stdout.split() if w.startswith(("--", "[--"))} - {"--help"}
    assert r.returncode == 0 and flags == {"--lcm-url", "--port", "--hz", "--point-cloud",
                                           "--show-collision"}, flags
    r = subprocess.run([sys.executable, script, "--lcm-url", "udpm://239.255.76.67:7667?ttl=0"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 2 and "shared LCM group" in r.stdout + r.stderr, r.stdout + r.stderr


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_port_busy():
    with socket.socket() as s:
        s.bind(("", 0))
        s.listen()
        try:
            LcmViewer(URL, port=s.getsockname()[1], hz=60, point_cloud=False,
                      show_collision=False)
        except OSError as exc:
            assert "busy" in str(exc)
        else:
            raise AssertionError("a busy port must refuse, not move to the next one")


def test_live_drain():
    t0 = time.perf_counter()
    app = LcmViewer(URL, port=_free_port(), hz=1000, point_cloud=False, show_collision=False)
    print(f"model build on cpu: {app.build_s:.1f} s")
    try:
        app.on_draw(VIEWER_DRAW_CHANNEL, b"not a viewer draw")
        assert app.dropped == 1 and app.draws == 0
        labels = [str(x) for x in app.model.body_label]
        rows = [0, len(labels) // 2, len(labels) - 1]
        names = [labels[i] for i in rows] + ["no_such_body"]
        before = app.state.body_q.numpy().copy()
        pub, sent = lcm.LCM(URL), None
        for k in range(50):
            sent = np.zeros((4, 7), dtype=np.float32)
            sent[:, :3] = k + np.arange(12, dtype=np.float32).reshape(4, 3)
            sent[:, 6] = 1.0
            pub.publish(VIEWER_DRAW_CHANNEL, viewer_draw_msg(k, names, sent).encode())
        deadline = time.monotonic() + 2.0
        while not app.pump():
            assert time.monotonic() < deadline, "no render within 2 s"
        got = app.state.body_q.numpy()
        assert app.draws == 50, app.draws  # > LCM's default queue of 30
        assert np.allclose(got[rows], sent[:3]), got[rows]
        others = np.setdiff1d(np.arange(len(labels)), rows)
        assert np.array_equal(got[others], before[others])  # unknown name touched nothing
    finally:
        app.close()
    print(f"in-process viewer test: {time.perf_counter() - t0:.1f} s wall")


def test_suite_port_guard():
    """run.sh's port guard lets a passive lcm_viewer.py socket through, refuses any other."""
    sys.path.insert(0, str(REPO / "scripts/lcs"))
    from run_hard_suite import port_guard

    bind = "import socket,sys; s=socket.socket(2,2); s.bind(('',7853)); print(1, flush=True); sys.stdin.read()"
    for argv, want in (([sys.executable, "-c", bind, "lcm_viewer.py"], 0),
                       ([sys.executable, "-c", bind, "test_lcm_viewer.py"], 3)):
        holder = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            holder.stdout.readline()
            rc = subprocess.run(["bash", "-c", port_guard(7853)], check=False,
                                capture_output=True).returncode
            assert rc == want, (argv[-1], rc)
        finally:
            holder.kill()
            holder.wait()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
