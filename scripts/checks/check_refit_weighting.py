#!/usr/bin/env python3
"""Checks of the V12-36 task-driven refit weighting. No sim; CPU only (CUDA_VISIBLE_DEVICES="").

W1 frame-weight math (lcs_learning ``refit_from_buffer.py`` in its venv): ``frame_weight_array``
   expansion, None without the field / with no segment; bad segments refused (overlap, empty,
   negative, NaN, bool, float index, beyond the episode); ``fit_lcs_only``'s tuple weights
   (captured before the first epoch) == row weight x the frame weight at the tuple start, None
   (the unweighted shuffle) without weights; all-1 frame weights == none; a weighted draw lands
   at the weight shares.
W2 refit identity (CPU, 1 epoch, 3 demos + 3 MILES + 2 MPC rows of V12-35's B_2): pre-change vs
   tree without ``frame_weights``: refit.log, lcs_params, refit_report.json (minus tag / paths /
   timings) identical; tree with all-1 frame weights: lcs_params identical, report gains
   ``frame_weights``; with a 0.2 tail: params differ, removed mass as computed.
W3 off-track rule (``task_common.onpolicy_weighting``): synthetic latents vs brute force (tube
   ratio, first frame, "from every demo latent", streak, NaN, after_s, segments, fail_weight,
   rule validation); V12-35 iteration 2: ``frame_tols`` == ``Waypoints.tol_at``, the recorded
   ``sim_latent`` whitened distance to the seated waypoint == the node's ``sim_goal_dist`` and
   == deploy_2's numpy encoding of the frame (the deploy encoder's latent).
W4 loop buffer step: ``st_buffer`` replay of V12-35 iterations 0-3 (recorded attempts / labels,
   merged seed, anchors 0.4) on a scratch root: off -> buffer.json / allowlist.json bytes pre ==
   tree; on -> each new MPC row's ``frame_weights`` == ``episode_weights`` recomputed, carried
   rows unchanged, anchor share == frac of the frame-weighted total, stats; the encoded-latent
   fallback (an episode without ``sim_latent``) gives the same weights; learning curve: pre ==
   tree without the key, the section with it.
W5 config: ``check_cfg`` refusals (bad blocks, no waypoints, joint mode, fresh_each_iter,
   from_seed not bool / with fresh_each_iter); v12_36.yaml and ``{}`` accepted.
W6 from_seed: --dry-run of v12_36.yaml on the V12-35 root, iterations 2 / 3: --init-checkpoint
   = theta_0's (v12_35.yaml: iteration 1's rejected candidate); ``st_refit`` with a stubbed
   refit: model.json / refit.json ``from_seed``, default pre == tree bytes.
W7 loop --dry-run identity pre vs tree (stdout bytes): C1_transport.yaml and v12_18.yaml
   iterations 0 / 1, v12_35.yaml iterations 0-4.

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_refit_weighting.py \\
        --pre-dir <pre-change files in the repo layout> \\
        --pre-refit <pre-change lcs_learning scripts/refit_from_buffer.py> [--out <json>]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from task_common import onpolicy_weighting as ow

LCS = Path.home() / "git/lcs_learning"
PY_LCS = LCS / ".venv/bin/python"
CMP = REPO / "data/lcs/realfirst/cmp"
V35 = CMP / "v12_35"
CFG = CMP / "configs"
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def load_loop(d: Path | None, tag: str):
    path = REPO / "scripts/lcs/realfirst_loop.py" if d is None else d / "scripts/lcs/realfirst_loop.py"
    mod = types.ModuleType(f"realfirst_loop_{tag}")
    mod.__file__ = str(REPO / "scripts/lcs/realfirst_loop.py")
    sys.modules[mod.__name__] = mod
    exec(compile(path.read_text(), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def lcs_layout(refit: Path, root: Path) -> Path:
    """A lcs_learning-like dir running ``refit`` (package + trainer symlinked)."""
    (root / "scripts").mkdir(parents=True)
    shutil.copy(refit, root / "scripts/refit_from_buffer.py")
    (root / "lcs_learning").symlink_to(LCS / "lcs_learning")
    (root / "scripts/train_joint_pointnet_lcs.py").symlink_to(
        LCS / "scripts/train_joint_pointnet_lcs.py")
    return root


def seed_ckpt() -> str:
    return json.loads((V35 / "iter0/model.json").read_text())["checkpoint"]


def small_buffer(n_demo=3, n_miles=3, n_mpc=2) -> dict:
    b = json.loads((V35 / "iter1/buffer.json").read_text())
    r = b["rollouts"]
    rows = ([x for x in r if x["kind"] == "demo"][:n_demo]
            + [x for x in r if x["kind"] == "miles"][:n_miles]
            + [x for x in r if x["kind"] == "mpc"][:n_mpc])
    return {**b, "n": len(rows), "rollouts": copy.deepcopy(rows)}


# ---------------------------------------------------------------- W1

W1_SNIPPET = r'''
import importlib.util, io, json, sys
import numpy as np, torch
from torch.utils.data import WeightedRandomSampler
spec = importlib.util.spec_from_file_location("rfb", sys.argv[1])
rfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rfb)
buf = json.loads(open(sys.argv[2]).read())["rollouts"]
out = {}
w = rfb.frame_weight_array({"episode": "x", "frame_weights": [[2, 5, 0.5], [7, 10, 0.0]]}, 10)
assert w.tolist() == [1, 1, .5, .5, .5, 1, 1, 0, 0, 0], w
assert rfb.frame_weight_array({"episode": "x"}, 10) is None
assert rfb.frame_weight_array({"episode": "x", "frame_weights": []}, 10) is None
bad = [[[0, 5, 1.0], [4, 8, 0.5]], [[5, 5, 1.0]], [[3, 2, 1.0]], [[-1, 2, 1.0]], [[0, 2, -0.1]],
       [[0, 2, float("nan")]], [[0, 2, True]], [[0.0, 2, 1.0]], [[0, 2]], "x", [[0, 2, "a"]],
       [[True, 2, 1.0]]]
n_bad = 0
for b in bad:
    try:
        rfb.check_frame_weights(b, "x")
    except SystemExit:
        n_bad += 1
assert n_bad == len(bad), n_bad
rfb.check_frame_weights([[4, 8, 0.5], [0, 4, 1.0]], "x")
try:
    rfb.frame_weight_array({"episode": "x", "frame_weights": [[0, 11, 1.0]]}, 10)
    raise AssertionError("beyond the episode accepted")
except SystemExit:
    pass
dev = torch.device("cpu")
b = rfb.Bundle(rfb.Path(sys.argv[3]), dev)
eps = [rfb.load_belt_seg_episode(r["episode"], int(b.a["num_points"])) for r in buf]
train = list(range(len(buf)))
class Stop(Exception):
    pass
cap = {}
def fake(n, w, gen):
    cap["n"], cap["w"] = n, None if w is None else w.numpy().copy()
    raise Stop
orig = rfb.epoch_order
rfb.epoch_order = fake
def tw(weights, fws):
    try:
        rfb.fit_lcs_only(b, eps, train, 1, 1e-4, 0, io.StringIO(), weights, fws)
    except Stop:
        pass
    return cap["w"]
ones = [1.0] * len(buf)
rw = [float(r.get("weight", 1.0)) for r in buf]
fws = [None] * len(buf)
mpc = [i for i, r in enumerate(buf) if r["kind"] == "mpc"]
for i in mpc:
    n = len(eps[i]["u"])
    fws[i] = rfb.frame_weight_array({"episode": "x", "frame_weights": [[n // 2, n, 0.2]]}, n)
all1 = [None if f is None else np.ones_like(f) for f in fws]
want = np.concatenate([rw[e] * (np.ones(len(rfb.tuples(eps[e], 1))) if fws[e] is None else
                       fws[e][rfb.tuples(eps[e], 1)]) for e in train])
got = tw(rw, fws)
assert tw(ones, None) is None and tw(ones, all1) is None
assert np.array_equal(tw(rw, None), tw(rw, all1))
assert np.allclose(got, want, rtol=0, atol=0), "tuple weights"
assert (got != np.concatenate([np.full(len(rfb.tuples(eps[e], 1)), rw[e]) for e in train])).sum() > 0
out["n_tuples"] = int(len(got))
out["n_down"] = int((want != np.concatenate([np.full(len(rfb.tuples(eps[e], 1)), rw[e])
                                              for e in train])).sum())
g = torch.Generator().manual_seed(0)
idx = np.asarray(list(WeightedRandomSampler(torch.as_tensor(got, dtype=torch.double), 200000,
                                            replacement=True, generator=g)))
cnt = np.bincount(idx, minlength=len(got))
share = got / got.sum()
groups = {}
off = 0
for e in train:
    k = len(rfb.tuples(eps[e], 1))
    for name, sl in (("head", slice(off, off + k // 2)), ("tail", slice(off + k // 2, off + k))):
        groups[f"{e}{name}"] = (cnt[sl].sum() / len(idx), share[sl].sum())
    off += k
out["max_share_err"] = float(max(abs(a - b) for a, b in groups.values()))
assert out["max_share_err"] < 0.01, groups
print(json.dumps(out))
'''


def w1(ctx) -> str:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "buf.json").write_text(json.dumps(small_buffer(2, 2, 1)))
        (td / "w1.py").write_text(W1_SNIPPET)
        p = subprocess.run([str(PY_LCS), str(td / "w1.py"), str(LCS / "scripts/refit_from_buffer.py"),
                            str(td / "buf.json"), seed_ckpt()], cwd=LCS, env=ENV,
                           capture_output=True, text=True, check=False)
        _require(p.returncode == 0, f"W1 snippet failed: {p.stderr[-2000:]}")
        r = json.loads(p.stdout.strip().splitlines()[-1])
    return (f"expansion / 12 bad segments refused / beyond-episode refused; tuple weights == row x "
            f"frame weight over {r['n_tuples']} tuples ({r['n_down']} down-weighted); none / all-1"
            f" -> unweighted shuffle; 200k draws within {r['max_share_err']:.4f} of the shares")


# ---------------------------------------------------------------- W2


def run_refit(lcs_dir: Path, buf: Path, out: Path, tag: str) -> Path:
    cmd = [str(PY_LCS), "scripts/refit_from_buffer.py", "--buffer", str(buf), "--init-checkpoint",
           seed_ckpt(), "--epochs", "1", "--mode", "lcs_only", "--lr", "0.0003", "--seed", "4",
           "--out-dir", str(out), "--tag", tag, "--recon-max-mm", "6.0", "--device", "cpu"]
    return subprocess.Popen(cmd, cwd=lcs_dir, env=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True), out / tag


PARAMS = r'''
import sys, torch
a = [torch.load(p, map_location="cpu", weights_only=False)["lcs_params"] for p in sys.argv[1:]]
print(" ".join(str(bool(torch.equal(a[0], x))) for x in a[1:]))
'''

VOLATILE = ("tag", "checkpoint", "timings_s", "finished_at", "buffer")


def w2(ctx) -> str:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        b = small_buffer()
        bufs = {"none": b, "all1": copy.deepcopy(b), "tail": copy.deepcopy(b)}
        cut = {}
        for name in ("all1", "tail"):
            for r in bufs[name]["rollouts"]:
                if r["kind"] == "mpc":
                    n = r["n_frames"]
                    cut[r["id"]] = n // 2
                    r["frame_weights"] = [[0, n, 1.0]] if name == "all1" else [[n // 2, n, 0.2]]
        for k, v in bufs.items():
            (td / f"{k}.json").write_text(json.dumps(v))
        tree = lcs_layout(LCS / "scripts/refit_from_buffer.py", td / "tree_lcs")
        jobs = [("tree_none", tree, "none"), ("tree_all1", tree, "all1"),
                ("tree_tail", tree, "tail")]
        if ctx.pre_refit:
            jobs.append(("pre_none", lcs_layout(ctx.pre_refit, td / "pre_lcs"), "none"))
        procs = [(tag, *run_refit(d, td / f"{bk}.json", td / "out", tag)) for tag, d, bk in jobs]
        runs = {}
        for tag, p, rd in procs:
            o = p.communicate()[0]
            _require(p.returncode == 0, f"{tag} failed: {o[-1500:]}")
            runs[tag] = rd
        rep = {t: json.loads((d / "refit_report.json").read_text()) for t, d in runs.items()}
        ck = [str(runs[t] / "checkpoint_lcs_only.pt") for t in
              ("tree_none", "tree_all1", "tree_tail", *(["pre_none"] if ctx.pre_refit else []))]
        p = subprocess.run([str(PY_LCS), "-c", PARAMS, *ck], cwd=LCS, env=ENV,
                           capture_output=True, text=True, check=True)
        eq = p.stdout.split()
        _require(eq[0] == "True", "all-1 frame weights changed lcs_params")
        _require(eq[1] == "False", "a 0.2 tail left lcs_params unchanged")
        _require("frame_weights" not in rep["tree_none"], "report key without the field")
        fa, ft = rep["tree_all1"]["frame_weights"], rep["tree_tail"]["frame_weights"]
        _require(fa["n_tuples_weighted"] == 0 and fa["mass_share_removed"] == 0.0, f"all-1 {fa}")
        _require(ft["n_tuples_weighted"] > 0 and ft["mass_share_removed"] > 0, f"tail {ft}")

        def strip(r):  # baseline_theta0.source names the run dir
            out = {k: v for k, v in r.items() if k not in VOLATILE + ("frame_weights",)}
            out["baseline_theta0"] = {k: v for k, v in r["baseline_theta0"].items()
                                      if k != "source"}
            return out

        dk = [k for k in strip(rep["tree_none"]) if strip(rep["tree_none"])[k]
              != strip(rep["tree_all1"]).get(k)]
        _require(not dk, f"all-1 report differs: {[(k, rep['tree_none'][k], rep['tree_all1'][k]) for k in dk]}")
        msg = ""
        if ctx.pre_refit:
            _require(eq[2] == "True", "pre vs tree lcs_params differ")
            _require(strip(rep["pre_none"]) == strip(rep["tree_none"]), "pre vs tree report")
            _require((runs["pre_none"] / "refit.log").read_bytes()
                     == (runs["tree_none"] / "refit.log").read_bytes(), "pre vs tree refit.log")
            msg = "pre == tree (refit.log, lcs_params, report); "
        sp = rep["tree_none"]["split"]
    return (msg + f"all-1 frame weights == none (params, metrics); 0.2 tail: params differ, "
            f"{ft['n_tuples_weighted']}/{ft['n_tuples']} tuples, mass removed "
            f"{ft['mass_share_removed']:.3f}; split {sp['n_train']} / {sp['n_heldout']}")


# ---------------------------------------------------------------- W3


def brute_ratio(z, zt, zs, tf):
    out = []
    for x in z:
        if not np.all(np.isfinite(x)):
            out.append(np.nan)
            continue
        out.append(min(np.sqrt(np.sum(((x - zt[j]) / zs) ** 2)) / tf[j] for j in range(len(zt))))
    return np.asarray(out)


def w3(ctx) -> str:
    rng = np.random.default_rng(0)
    T, nx = 60, 4
    zt = np.zeros((T, nx))
    zt[:, 0] = np.linspace(0.0, 6.0, T)
    zs = np.full(nx, 0.5)
    frames, tol = np.array([10, 20, 40, 59]), np.array([0.5, 1.0, 0.8, -1.0])
    tf = ow.frame_tols(frames, tol, T)
    want_tf = [tol[np.flatnonzero(tol >= 0)][np.argmin(np.abs(frames[tol >= 0] - j))]
               for j in range(T)]
    _require(np.array_equal(tf, want_tf), "frame_tols")
    n = 100
    z = zt[np.minimum(np.arange(n), T - 1)] + rng.normal(0, 0.01, (n, nx))
    z[60:, 1] = 0.05 * np.arange(40)  # leaves the tube sideways
    z[30] = zt[5]  # back at an early demo latent: on track
    z[25, 2] = 2.0  # one-frame spike
    z[70] = np.nan
    r = ow.tube_ratio(z, zt, zs, tf)
    rb = brute_ratio(z, zt, zs, tf)
    _require(np.allclose(r, rb, equal_nan=True) and np.isnan(r[70]), "tube ratio vs brute force")
    k = 3.0
    exp1 = next(t for t in range(n) if np.isfinite(rb[t]) and rb[t] > k)
    _require(ow.first_off_track(r, k) == exp1 == 25, f"first frame {exp1}")
    exp3 = next(t for t in range(n - 2) if all(np.isfinite(rb[t + j]) and rb[t + j] > k
                                              for j in range(3)))
    _require(ow.first_off_track(r, k, 3) == exp3 and exp3 > 60, f"streak {exp3}")
    _require(r[30] < 0.5, "a latent at an early demo frame is on track")
    _require(ow.first_off_track(np.full(5, np.nan), 1.0) is None, "NaN counted")
    wp = {"z_traj": zt, "z_std": zs, "frames": frames, "tol": tol}
    rule = {"off_track": {"tol_k": k, "after_s": 1.0, "streak": 3}, "off_track_weight": 0.2,
            "fail_weight": 0.5}
    fw, info = ow.episode_weights(z, wp, 0.075, True, rule)
    _require(info["after_frames"] == 13 and info["weighted_from"] == exp3 + 13
             and fw == [[exp3 + 13, n, 0.2]], f"success {fw} {info}")
    fw, _ = ow.episode_weights(z, wp, 0.075, False, rule)
    _require(fw == [[0, exp3 + 13, 0.5], [exp3 + 13, n, 0.1]], f"failure {fw}")
    _require(abs(ow.effective_frames({"frame_weights": fw}, n)
                 - (0.5 * (exp3 + 13) + 0.1 * (n - exp3 - 13))) < 1e-9, "effective frames")
    _require(ow.episode_weights(z[:exp3 + 10], wp, 0.075, True, rule)[0] == [], "cut past end")
    _require(ow.segments(10, None, 0.2) == [] and ow.segments(10, 0, 0.2) == [[0, 10, 0.2]],
             "segments")
    good = [{}, {"off_track": {"tol_k": 2}}, {"off_track_weight": 0.0}, rule]
    bad = [[], {"x": 1}, {"off_track": {"k": 1}}, {"off_track": {"tol_k": 0}},
           {"off_track": {"after_s": -1}}, {"off_track": {"streak": 0}},
           {"off_track": {"streak": True}}, {"off_track_weight": -0.1}, {"fail_weight": 0},
           {"fail_weight": float("nan")}, {"off_track": {"tol_k": "3"}}]
    _require(all(ow.check_rule(g) is None for g in good), "good rule refused")
    _require(all(ow.check_rule(b) is not None for b in bad), "bad rule accepted")
    # real data: V12-35 iteration 2
    from latent_waypoints import Waypoints

    from task_common.latent_encoder import LatentEncoder

    wpp = V35 / "iter2/latent_waypoints.npz"
    W = Waypoints.load(wpp)
    with np.load(wpp, allow_pickle=True) as d:
        rw = {k: np.asarray(d[k]) for k in ("z_traj", "z_std", "frames", "tol", "z_wp")}
    tfr = ow.frame_tols(rw["frames"], rw["tol"], len(rw["z_traj"]))
    _require(np.array_equal(tfr, [W.tol_at(j) for j in range(len(rw["z_traj"]))]), "tol_at")
    ep = V35 / "iter2/collect/episode_gv_02_4.npz"
    with np.load(ep, allow_pickle=True) as d:
        zl, gd = np.asarray(d["sim_latent"], np.float64), np.asarray(d["sim_goal_dist"])
        seg = [d["pcd_belt_seg"][t] for t in range(0, len(zl), 40)]
    dseat = np.linalg.norm((zl - rw["z_wp"][-1]) / rw["z_std"], axis=1)
    e_goal = float(np.max(np.abs(dseat - gd[:, 1])))
    _require(e_goal < 1e-3, f"sim_latent vs sim_goal_dist {e_goal}")
    enc = LatentEncoder.load(json.loads((V35 / "iter2/deploy.json").read_text())["deploy"])
    ze = np.stack([enc.encode(np.asarray(s, np.float32)) for s in seg])
    e_enc = float(np.max(np.abs(ze - zl[::40]) / rw["z_std"]))
    _require(e_enc < 1e-3, f"sim_latent vs deploy encoding {e_enc}")
    sub = slice(0, len(zl), 7)
    _require(np.allclose(ow.tube_ratio(zl[sub], rw["z_traj"], rw["z_std"], tfr),
                         brute_ratio(zl[sub], rw["z_traj"], rw["z_std"], tfr)), "real ratio")
    return (f"synthetic: ratio == brute force, first frame {exp1} (spike), streak 3 -> {exp3}, "
            f"early-demo latent on track, NaN never off, after 13 frames, failure x0.5, "
            f"{len(bad)} bad rules refused; V12-35 i2: tol_at identical, sim_latent vs node "
            f"goal distance {e_goal:.1e}, vs deploy encoding {e_enc:.1e} (whitened)")


# ---------------------------------------------------------------- W4


@contextlib.contextmanager
def no_print():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def fake_run(cmd, log, cwd=None, ctx=None, step=None) -> int:
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"\n# fake {' '.join(map(str, cmd))}\nPASS\n")
    return 0


def ns(root: Path, i: int, cfg: Path, dry=True):
    return argparse.Namespace(smoke=False, root=str(root), iter=i, config=str(cfg), dry_run=dry,
                              collect_port=None, eval_port=None, force=None, ignore_stop=False)


def replay(rl, root: Path, over: dict, n_it: int) -> list[dict]:
    cfg = yaml.safe_load((CFG / "v12_35.yaml").read_text())
    cfg.update(seed_buffer_merged=str(V35 / "seed_buffer_merged.json"), **over)
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    rl.run = fake_run
    out = []
    for i in range(n_it):
        it = root / f"iter{i}"
        it.mkdir(parents=True, exist_ok=True)
        for f in ("attempts.json", "labels.json", "latent_waypoints.npz", "deploy.json"):
            shutil.copy(V35 / f"iter{i}" / f, it / f)
        with no_print():
            rl.st_buffer(rl.Ctx(ns(root, i, root / "cfg.yaml")), False)
        out.append({"buffer": (it / "buffer.json").read_bytes(),
                    "allow": (it / "allowlist.json").read_bytes()})
    return out


def frames_of(ep: str, cache: dict) -> int:
    if ep not in cache:
        with np.load(ep, allow_pickle=True) as d:
            cache[ep] = len(d["actions"])
    return cache[ep]


def w4(ctx) -> str:
    n_it = min(4, len([i for i in range(10) if (V35 / f"iter{i}/labels.json").is_file()]))
    rule = {"off_track": {"tol_k": 2.0, "after_s": 1.0}, "off_track_weight": 0.2,
            "fail_weight": 0.5}
    lines = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tree = replay(ctx.tree_loop, td / "off", {}, n_it)
        rec = sum(json.loads(t["buffer"])["rollouts"] == json.loads(
            (V35 / f"iter{i}/buffer.json").read_text())["rollouts"] for i, t in enumerate(tree))
        if ctx.pre_loop is not None:
            shutil.rmtree(td / "off")
            pre = replay(ctx.pre_loop, td / "off", {}, n_it)
            _require(all(p == t for p, t in zip(pre, tree, strict=True)), "off: pre != tree bytes")
        on = replay(ctx.tree_loop, td / "on", {"onpolicy_weighting": rule}, n_it)
        cache, prev = {}, {}
        for i, x in enumerate(on):
            b = json.loads(x["buffer"])
            with np.load(V35 / f"iter{i}/latent_waypoints.npz", allow_pickle=True) as d:
                wp = {k: np.asarray(d[k]) for k in ("z_traj", "z_std", "frames", "tol")}
            for r in b["rollouts"]:
                if r["kind"] != "mpc":
                    _require("frame_weights" not in r, "frame_weights on a non-MPC row")
                    continue
                if r["iter"] == i:
                    with np.load(r["episode"], allow_pickle=True) as d:
                        z = np.asarray(d["sim_latent"], np.float64)
                    fw, info = ow.episode_weights(z, wp, 0.075, r["label_human"], rule)
                    _require(r["frame_weights"] == fw and r["off_track"]["first_off_track"]
                             == info["first_off_track"], f"i{i} {r['id']} weights")
                    prev[r["id"]] = r["frame_weights"]
                else:
                    _require(r["frame_weights"] == prev[r["id"]], f"i{i} carried {r['id']}")
            eff = {r["episode"]: ow.effective_frames(r, frames_of(r["episode"], cache))
                   for r in b["rollouts"]}
            wf = [(r["weight"] * eff[r["episode"]], r["kind"] in ("demo", "miles"))
                  for r in b["rollouts"]]
            share = sum(w for w, a in wf if a) / sum(w for w, _ in wf)
            _require(abs(share - 0.4) < 1e-9, f"i{i}: anchor share {share}")
            st = b["onpolicy_weighting"]
            lines.append(f"i{i}: new {st['new']['frames_down']}/{st['new']['frames']}, buffer "
                         f"{st['pool']['share_down']:.0%} down, mass kept "
                         f"{st['pool']['mass_kept']:.2f}")
        # encoded fallback: the same episode prefix with / without sim_latent
        ep = V35 / "iter1/collect/episode_gv_06_0.npz"
        with np.load(ep, allow_pickle=True) as d:
            keep = {k: d[k][:60] for k in ("actions", "utime", "sim_latent")}
            seg = np.empty(60, dtype=object)
            seg[:] = [d["pcd_belt_seg"][t] for t in range(60)]
        np.savez(td / "with.npz", **keep, pcd_belt_seg=seg)
        np.savez(td / "without.npz", **{k: v for k, v in keep.items() if k != "sim_latent"},
                 pcd_belt_seg=seg)
        rl = ctx.tree_loop
        c = rl.Ctx(ns(td / "on", 1, td / "on/cfg.yaml"))
        c.cfg = {**c.cfg, "onpolicy_weighting": {"off_track": {"tol_k": 0.3, "after_s": 0.3},
                                                 "off_track_weight": 0.2}}
        rows = [{"id": k, "episode": str(td / f"{k}.npz"), "label_human": False}
                for k in ("with", "without")]
        a, b2 = rl.onpolicy_rows(c, rows)
        _require(a["off_track"]["latents"] == "sim_latent" and b2["off_track"]["latents"]
                 == "encoded" and a["frame_weights"] == b2["frame_weights"]
                 and a["off_track"]["first_off_track"] is not None, f"fallback {a} {b2}")
        # learning curve
        lc = json.loads((V35 / "learning_curve.json").read_text())
        bud = json.loads((V35 / "budget.json").read_text())
        cx = ctx.tree_loop.Ctx(ns(V35, 0, CFG / "v12_35.yaml"))
        md_t = ctx.tree_loop.learning_curve_md(cx, lc["rows"], bud, lc["verdict"], lc["g4"])
        if ctx.pre_loop is not None:
            cp = ctx.pre_loop.Ctx(ns(V35, 0, CFG / "v12_35.yaml"))
            _require(md_t == ctx.pre_loop.learning_curve_md(cp, lc["rows"], bud, lc["verdict"],
                                                            lc["g4"]), "learning curve pre != tree")
        rows = copy.deepcopy(lc["rows"])
        rows[-1]["onpolicy"] = {k: json.loads(on[-1]["buffer"])["onpolicy_weighting"][k]
                                for k in ("new", "pool")}
        md_on = ctx.tree_loop.learning_curve_md(cx, rows, bud, lc["verdict"], lc["g4"])
        _require("On-policy refit weighting" in md_on and "On-policy" not in md_t, "lc section")
    return (f"off: {n_it} iterations {'pre == tree bytes' if ctx.pre_loop else 'tree only'} "
            f"({rec}/{n_it} rows == the recorded V12-35 buffers); on (tol_k 2, fail 0.5): "
            + "; ".join(lines) + "; anchor share 0.400 (frame-weighted) every iteration; "
            "encoded fallback == sim_latent weights; learning-curve section")


# ---------------------------------------------------------------- W5


def w5(ctx) -> str:
    rl = ctx.tree_loop
    base = rl.merge(rl.DEFAULTS, yaml.safe_load((CFG / "v12_36.yaml").read_text()))
    rl.check_cfg(base)
    rl.check_cfg(rl.merge(base, {"onpolicy_weighting": {}}))
    rl.check_cfg(rl.merge(rl.DEFAULTS, yaml.safe_load((CFG / "v12_35.yaml").read_text())))
    bad = [{"onpolicy_weighting": {"x": 1}}, {"onpolicy_weighting": {"off_track": {"tol_k": 0}}},
           {"onpolicy_weighting": {"fail_weight": 0}}, {"onpolicy_weighting": []},
           {"waypoints": None, "goal_mode": None, "demo_prior": None},
           {"refit": {"mode": "joint"}},
           {"refit": {"fresh_each_iter": True, "from_seed": False}},
           {"refit": {"from_seed": "yes"}}, {"refit": {"from_seed": 1}}]
    n = 0
    for b in bad:
        cfg = rl.merge(base, b)
        if "onpolicy_weighting" in b:
            cfg["onpolicy_weighting"] = b["onpolicy_weighting"]
        try:
            rl.check_cfg(cfg)
        except SystemExit as e:
            _require("onpolicy_weighting" in str(e) or "from_seed" in str(e), f"{b}: {e}")
            n += 1
            continue
        raise AssertionError(f"config {b} accepted")
    c2 = rl.merge(base, {"refit": {"fresh_each_iter": True}})
    c2["onpolicy_weighting"] = None
    try:
        rl.check_cfg(c2)
        raise AssertionError("from_seed + fresh_each_iter accepted")
    except SystemExit as e:
        _require("from_seed" in str(e), str(e))
        n += 1
    return f"{n} bad configs refused; v12_36.yaml, onpolicy_weighting {{}} and v12_35.yaml accepted"


# ---------------------------------------------------------------- W6 / W7


def dry_main(rl, argv: list[str]) -> str:
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["realfirst_loop.py", *argv]
    try:
        with contextlib.redirect_stdout(buf):
            rl.main()
    finally:
        sys.argv = old
    return buf.getvalue()


def init_of(out: str) -> str:
    line = next(x for x in out.splitlines() if "refit_from_buffer.py" in x)
    t = line.split()
    return t[t.index("--init-checkpoint") + 1]


def w6(ctx) -> str:
    rl = ctx.tree_loop
    seed = seed_ckpt()
    m1 = json.loads((V35 / "iter1/model.json").read_text())
    got = {}
    for i in (2, 3):
        o36 = dry_main(rl, ["--root", str(V35), "--config", str(CFG / "v12_36.yaml"), "--iter",
                            str(i), "--step", "refit", "--dry-run"])
        o35 = dry_main(rl, ["--root", str(V35), "--config", str(CFG / "v12_35.yaml"), "--iter",
                            str(i), "--step", "refit", "--dry-run"])
        _require(init_of(o36) == seed and "refit.from_seed: init = theta_0 (V15a)" in o36,
                 f"i{i} from_seed init {init_of(o36)}")
        _require("from_seed" not in o35 and init_of(o35) != seed, f"i{i} default init")
        got[i] = Path(init_of(o35)).parent.name
    _require(init_of(dry_main(rl, ["--root", str(V35), "--config", str(CFG / "v12_35.yaml"),
                                   "--iter", "2", "--step", "refit", "--dry-run"]))
             == m1["rejected"]["checkpoint"], "default i2 init != iteration 1's candidate")

    def stub(mod, root: Path, cfg_name: str):
        cfg = yaml.safe_load((CFG / cfg_name).read_text())
        cfg["lcs_out"] = str(root / "lcs")
        root.mkdir(parents=True)
        (root / "cfg.yaml").write_text(yaml.safe_dump(cfg))
        for i in (0, 1):
            (root / f"iter{i}").mkdir(parents=True, exist_ok=True)
            shutil.copy(V35 / f"iter{i}/model.json", root / f"iter{i}/model.json")
        seen = {}
        rep = json.loads(Path(m1["refit_report"]).read_text())

        def fake_refit(c, cmd, tag, step):
            seen["cmd"] = [str(x) for x in cmd]
            return {**rep, "decision": "accepted", "checkpoint": str(root / "cand.pt")}

        mod.run_refit = fake_refit
        mod.behaviour_check = lambda c, ck: {"accept": True, "line": "ACCEPT (stub)"}
        c = mod.Ctx(ns(root, 2, root / "cfg.yaml", dry=False))
        with no_print():
            mod.st_refit(c, False)
        cmd = seen["cmd"]
        return (cmd[cmd.index("--init-checkpoint") + 1],
                (root / "iter2/model.json").read_bytes(), (root / "iter2/refit.json").read_bytes())

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        init, mj, rj = stub(rl, td / "on", "v12_36.yaml")
        m, r = json.loads(mj), json.loads(rj)
        _require(init == seed and m["from_seed"] == "V15a" and r["from_seed"]["checkpoint"] == seed
                 and "initialised from the seed V15a" in m["source"]
                 and "its rejected candidate" not in m["source"], f"from_seed st_refit {m}")
        t_off = stub(load_loop(None, "tree_w6"), td / "off", "v12_35.yaml")
        _require(t_off[0] == m1["rejected"]["checkpoint"], "default stub init")
        same = ""
        if ctx.pre_dir is not None:
            p_off = stub(load_loop(ctx.pre_dir, "pre_w6"), td / "off_pre", "v12_35.yaml")
            norm = [x.replace(str(td / "off_pre"), "R") for x in (p_off[1].decode(),
                                                                  p_off[2].decode())]
            normt = [x.replace(str(td / "off"), "R") for x in (t_off[1].decode(), t_off[2].decode())]
            _require(p_off[0] == t_off[0] and norm == normt, "default st_refit pre != tree")
            same = "; default st_refit model.json / refit.json pre == tree"
    return (f"dry-run i2/i3 init = theta_0 V15a (default: {got[2]}, {got[3]}); st_refit stub: "
            f"model.json from_seed V15a, refit.json from_seed{same}")


def w7(ctx) -> str:
    if ctx.pre_loop is None:
        return "skipped (no --pre-dir)"
    cases = [("C1_transport", CMP / "C1_transport", (0, 1)), ("v12_18", CMP / "v12_18", (0, 1)),
             ("v12_35", V35, (0, 1, 2, 3, 4))]
    n = 0
    for name, root, iters in cases:
        for i in iters:
            argv = ["--root", str(root), "--config", str(CFG / f"{name}.yaml"), "--iter", str(i),
                    "--step", "all", "--dry-run"]
            a = dry_main(load_loop(ctx.pre_dir, f"p7_{name}{i}"), argv)
            b = dry_main(load_loop(None, f"t7_{name}{i}"), argv)
            _require(a == b and len(a) > 200, f"{name} iter {i}: dry-run differs")
            n += 1
    return f"{n}/{n} dry-runs identical (C1_transport 0/1, v12_18 0/1, v12_35 0-4)"


CHECKS = [("W1 frame-weight math", w1), ("W2 refit identity", w2), ("W3 off-track rule", w3),
          ("W4 loop buffer step", w4), ("W5 config", w5), ("W6 from_seed", w6),
          ("W7 dry-run identity", w7)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="pre-change src/task_common + scripts/lcs/realfirst_loop.py (repo layout)")
    ap.add_argument("--pre-refit", type=Path, default=None,
                    help="pre-change lcs_learning scripts/refit_from_buffer.py")
    ap.add_argument("--only", default=None, help="comma list of W1..W7")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit('run with CUDA_VISIBLE_DEVICES=""')
    ctx = types.SimpleNamespace(tree_loop=load_loop(None, "tree"), pre_dir=a.pre_dir,
                                pre_loop=load_loop(a.pre_dir, "pre") if a.pre_dir else None,
                                pre_refit=a.pre_refit)
    t0, ok, res = time.perf_counter(), 0, {}
    sel = [c for c in CHECKS if not a.only or c[0].split()[0] in a.only.split(",")]
    for name, fn in sel:
        t1 = time.perf_counter()
        try:
            detail = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - report every check
            traceback.print_exc()
            print(f"[FAIL] {name}: {exc!r}", flush=True)
            res[name] = {"pass": False, "detail": repr(exc)}
            continue
        ok += 1
        res[name] = {"pass": True, "detail": detail, "s": round(time.perf_counter() - t1, 1)}
        print(f"[PASS] {name}: {detail} ({time.perf_counter() - t1:.0f} s)", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{ok}/{len(sel)} {'PASS' if ok == len(sel) else 'FAIL'} ({time.perf_counter() - t0:.0f} s)")
    return 0 if ok == len(sel) else 1


if __name__ == "__main__":
    sys.exit(main())
