"""Panel 4 checks (train log, buffer, argv, recon, export, mock), CPU-only; pytest-compatible.

    uv run --frozen python apps/demo_gui/tests/test_train.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the encoder imports warp; keep this process off the GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from apps.demo_gui import buffer, jobs, recon, trainlog
from apps.demo_gui.app import build_app
from apps.demo_gui.backends import sim
from apps.demo_gui.backends.mock import MOCK_ARMS, MockBackend
from apps.demo_gui.config import MOCK_RUN, REPO, Cfg
from apps.demo_gui.panels import train as panel
from apps.demo_gui.session import Session, read_index

FIX = Path(__file__).resolve().parent / "fixtures"
V63J = REPO / "data/lcs/realfirst/cmp/encoder_study/sim/V63j"
DEPLOY, DECODER = V63J / "deploy_v39j/deploy.npz", V63J / "decoder_decfit/decoder.npz"
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_train_")


def _session() -> Session:
    return Session(tempfile.mkdtemp(dir=_TMP.name))


def _until(cond, timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def test_parse_fixture_log():
    log = trainlog.parse(FIX / "train/train.log")
    n = sum(line.startswith("[epoch ") for line in (FIX / "train/train.log").read_text().splitlines())
    assert len(log["epoch"]) == n > 0
    assert all(b > a for a, b in zip(log["epoch"], log["epoch"][1:], strict=False))
    for k in ("train/total", "val/recon_chamfer_mm", "val/next_rms_mm", "val/ms_chamfer_mm_k7"):
        assert k in log, k
    assert all(len(v) == n for v in log.values())
    assert trainlog.last_epoch(FIX / "train/train.log") == log["epoch"][-1] == 250
    assert trainlog.report(FIX / "train")["checkpoint"].endswith("checkpoint_epoch_0250.pt")
    assert trainlog.report(FIX / "nope") is None and trainlog.parse(FIX / "nope")["epoch"] == []
    assert len(panel.figures(log)) == 4


def test_build_schema():
    s = _session()
    ep = MOCK_RUN / read_index(MOCK_RUN)[0]["file"]
    sel = [{"kind": "demo", "episode": str(ep)},
           {"kind": "miles", "buffer": str(FIX / "miles/buffer.json")}]
    out = buffer.path(s, "t1")
    counts = buffer.build(s, sel, out)
    doc = json.loads(out.read_text())
    ref = json.loads((FIX / "buffer_W150all_head.json").read_text())
    assert set(ref) <= set(doc), set(ref) - set(doc)
    rows = doc["rollouts"]
    assert doc["n"] == len(rows) == sum(v for k, v in counts.items() if k != "skipped")
    assert counts["s1_chain"] == 1 and counts["miles"] == 5, counts
    s1 = next(r for r in rows if r["kind"] == "s1_chain")
    assert set(ref["rollouts"][0]) <= set(s1), set(ref["rollouts"][0]) - set(s1)
    s2 = next(r for r in rows if r["stage"] == 2)
    assert s2["kind"] in ("stage2_demo", "stage2_mpc")
    for r in rows:
        assert r["weight"] == 1.0 and r["stage"] in (1, 2) and Path(r["episode"]).is_absolute()
    with np.load(ep) as z:
        exit_ = int(z["sim_chain_from"][0])
    with np.load(s1["episode"]) as z:  # stage 1 ends before the UR exit
        assert len(z["actions"]) == s1["n_frames"] <= exit_
    assert all(Path(r["episode"]).is_relative_to(out.parent) for r in (s1, s2))
    cand = buffer.candidates(s)
    assert cand[-1]["kind"] == "seed" if buffer.SEED.is_file() else True


def test_train_argv():
    s = _session()
    argv = sim.train_argv(s, "t1", 60, 5)
    joined = " ".join(argv)
    assert argv[0] == str(sim.LCS / ".venv/bin/python") and argv[1] == "scripts/refit_from_buffer.py"
    assert f"--out-dir {s.train} --tag t1" in joined and "--epochs 60 --seed 5" in joined
    assert f"--buffer {buffer.path(s, 't1')}" in joined
    assert "v12_152" not in joined and "lcs_learning/outputs" not in joined
    for flag in ("--train-only-kinds s1_chain,stage2_mpc,stage2_demo", "--heldout-frac 0.05",
                 "--no-violation-grad-to-encoder", "--n-dyn-heads 2", "--mode-gate-mats A,B,d"):
        assert flag in joined, flag
    calls = []

    class FakePopen:  # never runs flock / gpu_free.sh / the trainer
        pid = 999999999

        def __init__(self, a, **kw):
            calls.append((a, kw))

    buffer.path(s, "t1").parent.mkdir(parents=True)
    buffer.path(s, "t1").write_text("{}")
    real, jobs.subprocess.Popen = jobs.subprocess.Popen, FakePopen
    try:
        sim.SimBackend(Cfg()).train(s, "t1", [], 60, seed=5)
        sim.SimBackend(Cfg()).export(s, "t1")
    finally:
        jobs.subprocess.Popen = real
    (a, kw), (e, _) = calls
    assert a[:2] == ["flock", str(jobs.LOCK)] and joined.split()[-1] in a[-1]
    assert kw["cwd"] == sim.LCS and kw["env"]["PYTHONPATH"] == str(sim.LCS)
    assert "post_export.sh" in e[-1] and str(s.root) in e[-1]


def test_train_argv_relative_session():
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        assert Session("data/demo_gui/sessions/x").root == REPO / "data/demo_gui/sessions/x"
        tmp = Path(tempfile.mkdtemp(dir=_TMP.name)).resolve()
        os.chdir(tmp)
        argv = sim.train_argv(Session("rel"), "t1", 60, 5)
    finally:
        os.chdir(cwd)
    for flag in ("--buffer", "--out-dir"):  # the trainer runs with cwd=lcs_learning
        p = Path(argv[argv.index(flag) + 1])
        assert p.is_absolute() and p.is_relative_to(tmp / "rel"), (flag, p)


def test_predict_frame():
    gt, pred, ch = recon.predict_frame(DEPLOY, DECODER, FIX / "demo/episode_gv_07_0.npz", 0)
    assert gt.shape == pred.shape == (544, 3) and np.isfinite(ch), ch
    if panel.REPO_DEMO.is_file():  # the fixture's clouds are cut to 128 points; its source is whole
        _, _, ch = recon.predict_frame(DEPLOY, DECODER, panel.REPO_DEMO, 0)
        assert ch < 50.0, ch
    r = recon.episode(DEPLOY, DECODER, FIX / "e2e_run/episode_pre_place_1_osc_0.npz", _TMP.name)
    assert r["gt"].shape == r["pred"].shape and np.isfinite(r["chamfer_mm"]).all()


def test_post_export_script():
    sh = REPO / "apps/demo_gui/backend/post_export.sh"
    assert subprocess.run(["bash", "-n", sh], check=False).returncode == 0
    assert "v12_152" not in "".join(x for x in sh.read_text().splitlines() if not x.startswith("#"))
    out = Path(tempfile.mkdtemp(dir=_TMP.name))  # export_arms on V63j == the v12_151 arm
    subprocess.run([sys.executable, REPO / "apps/demo_gui/backend/export_arms.py", out,
                    V63J.parent, "V63j"], check=True, capture_output=True)
    a = json.loads((out / "arm.json").read_text())["V63j"]
    b = json.loads((REPO / "data/lcs/realfirst/cmp/v12_151/arms.json").read_text())["V63j_tol50b"]
    assert set(a) == set(b)
    for k in b:
        if k.startswith("wp"):
            with np.load(a[k]) as x, np.load(b[k]) as y:
                assert all(np.array_equal(x[f], y[f]) for f in y.files), k
        elif k.startswith("cfg"):
            body = [Path(p).read_text().split("\n", 1)[1] for p in (a[k], b[k])]
            assert body[0] == body[1], k
        else:
            assert a[k] == b[k], k


def test_layout_get():
    s = _session()
    app = build_app(MockBackend(), s, Cfg())
    assert app.server.test_client().get("/").status_code == 200
    assert "train-start" in str(app.layout) and "recon-fig" in str(app.layout)
    opts = panel.model_options(s)
    assert json.loads(opts[-1]["value"])["decoder"] == str(DECODER)


def _post(app, output: str, values: dict, changed: str):
    """POST the callback writing ``output`` like the browser does -> its response dict."""
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


def test_second_export_refused():
    s = _session()
    app = build_app(MockBackend(), s, Cfg())
    busy = jobs.run_thread(s, "export_t1", lambda stop: stop.wait(10))
    try:
        r = _post(app, "train-export-status.children", {"train-export": 1, "train-run": "t1",
                                                        "tabs": "train"}, "train-export.n_clicks")
        assert "in progress" in r["train-export-status"]["children"], r
    finally:
        jobs.stop(busy)


def test_mock_train_and_export():
    s = _session()
    be = MockBackend(pace=0.01)
    job = be.train(s, "m1", [], 60, seed=5)
    log = s.train / "m1/train.log"
    assert _until(lambda: log.is_file() and log.stat().st_size > 0, 5.0)
    n0 = log.stat().st_size
    assert _until(lambda: log.stat().st_size > n0, 5.0)
    assert _until(lambda: jobs.status(job) == "done", 60.0)
    assert trainlog.report(s.train / "m1") and len(trainlog.parse(log)["epoch"]) > 10
    assert [p.name for p in panel.runs(s)] == ["m1"]
    ex = be.export(s, "m1")
    assert _until(lambda: jobs.status(ex) == "done", 10.0)
    arm = json.loads((s.train / "m1/export/arm.json").read_text())["m1"]
    assert arm == json.loads(MOCK_ARMS.read_text())["V63j_tol50b"]
    assert "export m1" in panel.model_options(s)[0]["label"]


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
