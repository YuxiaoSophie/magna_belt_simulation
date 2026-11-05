"""Evaluate / Data scrub publishes the recorded frame on NEWTON_VIEWER_DRAW (lcm-viewer PKG-04).

CPU-only, private LCM 7852, no sim or viewer.

    uv run --frozen python apps/demo_gui/tests/test_scrub_publish.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import lcm
import numpy as np
from apps.demo_gui import episodes, jobs
from apps.demo_gui.app import build_app
from apps.demo_gui.backends.mock import MockBackend
from apps.demo_gui.config import Cfg, lcm_url
from apps.demo_gui.session import Session

from task_common.lcm_contract import VIEWER_DRAW_CHANNEL, lcmt_viewer_draw, viewer_draw_body_q
from task_common.recording import Recording

URL = lcm_url(7852)
FX = Path(__file__).resolve().parent / "fixtures/e2e_run"
EP = "episode_pre_place_1_osc_0"
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_scrub_")


def _run() -> Path:
    """Fixture run copy whose recording loads: the fixture keeps only chunk 0, cut to 40 steps
    but 40 state frames, so list chunk 0 only and drop the frames after its last step."""
    d = Path(tempfile.mkdtemp(dir=_TMP.name)) / "run"
    shutil.copytree(FX, d)
    m = d / "recordings" / EP / "meta.json"
    meta = json.loads(m.read_text())
    m.write_text(json.dumps(meta | {"chunks": meta["chunks"][:1]}))
    f = d / "recordings" / EP / "chunks/chunk_00000.npz"
    with np.load(f) as z:
        c = {k: z[k] for k in z.files}
    keep = c["state_step"] <= c["step"][-1]
    c["state_step"], c["body_q"] = c["state_step"][keep], c["body_q"][keep]
    np.savez(f, **c)
    return d


def _sub() -> tuple[lcm.LCM, list]:
    lc, got = lcm.LCM(URL), []
    lc.subscribe(VIEWER_DRAW_CHANNEL, lambda _c, data: got.append(lcmt_viewer_draw.decode(data)))
    return lc, got


def _drain(lc: lcm.LCM, ms: int) -> None:
    end = time.monotonic() + ms / 1e3
    while (left := int((end - time.monotonic()) * 1e3)) > 0:
        lc.handle_timeout(left)


def test_publish_frame():
    npz = _run() / f"{EP}.npz"
    rec = Recording.load(episodes.recording_dir(npz))
    f = rec.frame_at_step(int(np.load(npz)["sim_step"][3]))
    lc, got = _sub()
    assert episodes.publish_frame(Session(tempfile.mkdtemp(dir=_TMP.name)), npz, 3, URL) == \
        f"viewer: frame {f}"
    _drain(lc, 500)
    assert len(got) == 1, len(got)
    names, q = viewer_draw_body_q(got[0])
    assert len(names) == 86 and names == list(rec.meta["body_labels"])
    assert np.array_equal(q, rec.body_q[f])


def test_live_run_not_sent():
    s = Session(tempfile.mkdtemp(dir=_TMP.name))
    npz = _run() / f"{EP}.npz"
    lc, got = _sub()
    for name in ("demo", "miles_00"):  # any sim job on the URL, not only demo / eval_*
        job = jobs.start(s, name, ["sleep", "60"], lcm_url=URL)
        try:
            assert episodes.publish_frame(s, npz, 3, URL) == "live run: not sent", name
            _drain(lc, 300)
            assert not got, name
        finally:
            jobs.stop(job)


def test_no_recording():
    d = Path(tempfile.mkdtemp(dir=_TMP.name))
    shutil.copy(FX / f"{EP}.npz", d)
    s = Session(tempfile.mkdtemp(dir=_TMP.name))
    assert episodes.publish_frame(s, d / f"{EP}.npz", 3, URL) == "no recording"


def _post(app, output: str, values: dict, changed: str):
    key, cb = next(kv for kv in app.callback_map.items() if output in kv[0])
    outs = [{"id": o.rsplit(".", 1)[0], "property": o.rsplit(".", 1)[1]}
            for o in key.strip(".").split("...")]
    r = app.server.test_client().post("/_dash-update-component", json={
        "output": key, "outputs": outs if key.startswith("..") else outs[0],
        "inputs": [{**i, "value": values.get(i["id"])} for i in cb["inputs"]],
        "state": [{**i, "value": values.get(i["id"])} for i in cb["state"]],
        "changedPropIds": [changed]})
    assert r.status_code == 200, r.data[:300]
    return json.loads(r.data)["response"]


def test_eval_frame_title():
    app = build_app(MockBackend(), Session(tempfile.mkdtemp(dir=_TMP.name)), Cfg(sim_port=7852))
    npz = str(_run() / f"{EP}.npz")
    r = _post(app, "eval-3d.figure", {"eval-frame": 3, "eval-episode": npz}, "eval-frame.value")
    title = r["eval-3d"]["figure"]["layout"]["title"]["text"]
    assert "viewer: frame" in title, title


def test_recording_dir_and_replay_argv():
    npz = FX / f"{EP}.npz"
    assert episodes.recording_dir(npz) == FX / "recordings" / EP
    calls = []
    real, jobs.start = jobs.start, lambda _s, _n, argv: calls.append(argv)  # never starts a viewer
    try:
        episodes.start_replay(Session(tempfile.mkdtemp(dir=_TMP.name)), npz, 8092)
    finally:
        jobs.start = real
    assert calls[0][-2:] == ["--run", EP], calls


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
