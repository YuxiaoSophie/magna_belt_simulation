#!/usr/bin/env python3
"""Checks of the opt-in waypoint schedule (``latent_waypoints.py --min-frame``, node
``stage_skip_ahead`` / ``stage_no_timeout_into``, loop ``waypoints.{min_frame, skip_ahead,
no_timeout_into}``) and the behaviour gate's pre-latch rule.

W1 params: defaults False / None; every realfirst ``mppi*.yaml`` loads with ``to_dict`` == the
   pre-change module's plus the two keys at default; bad values refused; solver shared on / off.
W2 artifact: ``waypoint_frames`` on synthetic marks; ``derive`` pre vs tree byte-identical
   (goal demo + 2 demos); ``--min-frame`` 20 / 21 / 33 == the full artifact sliced; > latch
   refused.
W3 node rule (synthetic latents, solve stubbed): hand cases (timeouts, skip ahead, never back,
   gate blocks the timeout and the skip, streak 3, gate index with / without min_frame,
   refusals) and 200 random sequences == a reference implementation of the documented rules.
W4 node on a recorded V12-18 stream (real solves): pre vs tree with the options off: plans,
   log rows minus timing, stdout identical; on (skip + latch gate): per-solve stage == the
   reference on the recorded latents, JSONL ``blocked`` only when on.
W5 loop: ``mppi_yaml`` bytes pre == tree without the keys (C1, V12-18); with them the doc ==
   pre + the two MppiParams keys; ``waypoints_cmd`` + ``--min-frame`` only when set (offset ->
   mppi_demo_offset_frames); ``check_cfg`` refusals; ``--dry-run`` stdout pre == tree (C1 and
   V12-18 iters 0 / 1; nothing under those iterations written).
W6 gate: ``decide`` pre == tree on recorded gate rows (defaults); the pre-latch rule on
   synthetic stats (strict thresholds); pre-latch frames == [first ``move:`` label, latch) of
   every seed demo; loop ``behaviour_cmds`` adds the flags only when set.
W7 with ``prior_sync: stage`` (V12-20 UR z-30 #0): idx == ``prior_sync_index`` on the node's stage
   starts incl. skips, ``u_demo`` == ``u_seq(idx)``, zero from the gate's frame below the gate.

Clearance is off on the CPU (its torch path is minutes per solve there).

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_waypoint_schedule.py \\
        --pre-dir <dir with the pre-change files in the repo layout>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import overshoot_offline as oo
import torch

from task_common.latent_encoder import LATENT_STATE_CHANNEL, latent_state_message

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C1 = REPO / "data/lcs/realfirst/cmp/C1_transport"
V18 = REPO / "data/lcs/realfirst/cmp/v12_18"
FRAMES = [10, 20, 30, 33, 43, 53, 199]
LATCH = 33
DT_US = 75_000
LATER = (("stage_skip_ahead", False), ("stage_no_timeout_into", None))
POLICY = (("prior_mode", "demo"), ("bc_prior", None),
          ("stage_start", None), ("goal_set", None), ("goal_set_tau", 0.0),
          ("value", None), ("latent_support", None),
          ("ee_support", None))  # later fields (P-M3, V12-30/31/39)


def load_pair(d: Path | None, tag: str):
    """``(mppi_lcs, node)`` from ``d`` (repo layout) or the tree."""
    root = REPO if d is None else d
    spec = importlib.util.spec_from_file_location(f"mppi_lcs_{tag}",
                                                  root / "src/task_common/mppi_lcs.py")
    lcs_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lcs_mod
    spec.loader.exec_module(lcs_mod)
    sys.modules["task_common.mppi_lcs"] = lcs_mod
    spec = importlib.util.spec_from_file_location(
        f"mppi_node_{tag}", root / "scripts/lcs/mppi_controller_node.py")
    node_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = node_mod
    spec.loader.exec_module(node_mod)
    return lcs_mod, node_mod


def load_script(d: Path | None, name: str, tag: str):
    """``scripts/lcs/<name>.py`` from ``d``, run as if at the tree's path (same REPO)."""
    path = (REPO if d is None else d) / f"scripts/lcs/{name}.py"
    mod = types.ModuleType(f"{name}_{tag}")
    mod.__file__ = str(REPO / f"scripts/lcs/{name}.py")
    sys.modules[mod.__name__] = mod
    exec(compile(path.read_text(), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def c1_deploy() -> str:
    return json.loads((C1 / "iter1/collect/index.json").read_text())["args"]["deploy"]


# ---------------------------------------------------------------- W1


def w1(pre, tree) -> tuple[bool, str]:
    p = tree.MppiParams()
    ok, notes = p.stage_skip_ahead is False and p.stage_no_timeout_into is None, []
    yamls = sorted((REPO / "data/lcs/realfirst").rglob("mppi*.yaml"))
    n_on = 0
    for y in yamls:
        doc = yaml.safe_load(y.read_text()) or {}
        if {k for k, _ in LATER + POLICY} & set(doc):  # schedule / policy on: tree only
            n_on += 1
            tree.MppiParams.from_yaml(y)
            continue
        a, b = pre.MppiParams.from_yaml(y).to_dict(), tree.MppiParams.from_yaml(y).to_dict()
        for k, v in LATER:
            if b.pop(k, "missing") is not v or a.pop(k, v) is not v:  # pre may have them
                ok = False
        for k, v in POLICY:
            if k in b and b[k] == v and k not in a:
                b.pop(k)
        if a != b:
            ok = False
            notes.append(f"diff {y}")
    refused = 0
    for bad in ({"stage_skip_ahead": 1}, {"stage_skip_ahead": "yes"},
                {"stage_no_timeout_into": 0}, {"stage_no_timeout_into": -2},
                {"stage_no_timeout_into": "sweep"}, {"stage_no_timeout_into": True},
                {"stage_no_timeout_into": 1.5}):
        try:
            tree.MppiParams(**bad)
        except ValueError:
            refused += 1
    for good in ({"stage_skip_ahead": True}, {"stage_no_timeout_into": "latch"},
                 {"stage_no_timeout_into": 4}):
        tree.MppiParams(**good)
    lcs = tree.TorchLcs.from_deploy(c1_deploy(), DEVICE)
    q = tree.MppiParams(K=8, H=6, n_iter=1, cuda_graph=False)
    import dataclasses
    same = tree.solver_for(lcs, q) is tree.solver_for(lcs, dataclasses.replace(
        q, stage_skip_ahead=True, stage_no_timeout_into="latch"))
    ok = ok and refused == 7 and same
    return ok, (f"{len(yamls) - n_on} yamls: to_dict == pre + 2 keys at default; {n_on} with "
                f"them load; {refused}/7 bad refused; solver shared {same}"
                + (f"; {notes[:3]}" if notes else ""))


# ---------------------------------------------------------------- W2


def w2(pre_wp, tree_wp) -> tuple[bool, str]:
    ok, notes = True, []
    cases = {0: FRAMES, 20: [20, 30, 33, 43, 53, 199], 21: [30, 33, 43, 53, 199],
             33: [33, 43, 53, 199]}
    if tree_wp.waypoint_frames(33, 53, 200, 10) != pre_wp.waypoint_frames(33, 53, 200, 10):
        ok = False
    dep = Path(c1_deploy())
    demo = REPO / "data/lcs/realfirst/v1_1/seed/demo"
    demos = [demo / "episode_0000.npz", demo / "episode_0001.npz", demo / "episode_0002.npz"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        memo_p, memo_t = {}, {}
        d_pre = pre_wp.derive(dep, demos, demos[0], 10, 3, 90.0, 0.5, memo_p)
        d_tree = tree_wp.derive(dep, demos, demos[0], 10, 3, 90.0, 0.5, memo_t)
        pre_wp.save_npz(tmp / "pre.npz", d_pre)
        tree_wp.save_npz(tmp / "tree.npz", d_tree)
        same = (tmp / "pre.npz").read_bytes() == (tmp / "tree.npz").read_bytes()
        same_md = pre_wp.summary(d_pre) == tree_wp.summary(d_tree)
        ok = ok and same and same_md and d_tree["frames"].tolist() == cases[0]
        per = ("frames", "z_wp", "tol", "tol_percentile_raw", "n_tol_frames", "max_duration_s")
        for mf in (20, 21, 33):
            d = tree_wp.derive(dep, demos, demos[0], 10, 3, 90.0, 0.5, memo_t, min_frame=mf)
            keep = np.asarray(d_tree["frames"]) >= mf
            sl = all(np.array_equal(np.asarray(d[k]), np.asarray(d_tree[k])[keep]) for k in per)
            rest = all(np.array_equal(np.asarray(d[k]), np.asarray(d_tree[k]))
                       for k in d_tree if k not in (*per, "per_demo"))
            pd, pf = json.loads(str(d["per_demo"])), json.loads(str(d_tree["per_demo"]))
            al = all(a["aligned_frames"] == np.asarray(b["aligned_frames"])[keep].tolist()
                     for a, b in zip(pd, pf))
            tree_wp.save_npz(tmp / f"m{mf}.npz", d)
            w = tree_wp.Waypoints.load(tmp / f"m{mf}.npz")
            if not (sl and rest and al and int(d["min_frame"]) == mf and w.latch == LATCH
                    and d["frames"].tolist() == cases[mf]
                    and "min frame" in tree_wp.summary(d)):
                ok = False
                notes.append(f"min_frame {mf} != slice")
        try:
            tree_wp.derive(dep, demos, demos[0], 10, 3, 90.0, 0.5, memo_t, min_frame=34)
            ok = False
        except SystemExit:
            pass
    return ok, (f"frames {cases}; derive pre == tree bytes {same}, summary {same_md}; "
                f"min_frame 20 / 21 / 33 == full sliced (tols, durations, aligned frames); 34 "
                f"refused" + (f"; FAIL {notes[:3]}" if notes else ""))


# ---------------------------------------------------------------- W3


def synth_wp(path: Path, lcs, frames=FRAMES) -> np.ndarray:
    """Artifact with goal k at 10 sigma along latent axis k; tol 1 (last -1), 3 s (last 30)."""
    n = len(frames)
    zs = np.asarray(lcs.z_std_np, np.float64)
    z_wp = np.stack([10.0 * np.eye(lcs.n_x)[k] * zs for k in range(n)])
    z_traj = np.zeros((200, lcs.n_x))
    z_traj[frames] = z_wp
    doc = {"frames": np.asarray(frames, np.int64), "z_wp": z_wp,
           "tol": np.r_[np.ones(n - 1), -1.0], "max_duration_s": np.r_[np.full(n - 1, 3.0), 30.0],
           "z_traj": z_traj, "u_traj": np.zeros((200, 12)), "dt": np.float64(lcs.dt),
           "latch": np.int64(LATCH), "first_hold": np.int64(53), "deploy_sha256": np.str_(""),
           "demo_episode": np.str_("synthetic")}
    np.savez(path, **doc)
    return z_wp


def make_node(node_mod, wp_path: Path, frag: dict, tmp: Path, stub: bool = True,
              extra=(), goal=None):
    doc = yaml.safe_load((V18 / "iter0/mppi_collect.yaml").read_text())
    doc.update(clearance=None, prior_alpha=0.0, prior_weight=0.0, action_noise_frac=0.0,
               stage_reach_streak=1, K=64, n_iter=1)
    doc.update(frag)
    cfg = tmp / "m.yaml"
    cfg.write_text(yaml.safe_dump(doc))
    argv = ["--lcm-url", "fake", "--deploy", c1_deploy(), "--config", str(cfg),
            "--goal", goal or f"waypoints:{wp_path}", "--stages", "all", "--goal-mode", "stage",
            "--max-duration-s", "30", *extra]
    node = node_mod.MppiControllerNode(node_mod.create_parser().parse_args(argv), oo.FakeLc())
    if stub:
        node._solve = lambda *a, **k: None
    node.start()
    return node


def feed(node, zs, ef=None, eu=None) -> list[tuple[int, bool]]:
    ef = np.r_[0.36, 0.0, 0.09, 1.0, 0.0, 0.0, 0.0] if ef is None else ef
    eu = np.r_[0.51, -0.02, 0.1, 1.0, 0.0, 0.0, 0.0] if eu is None else eu
    out = []
    k0 = getattr(node, "_k", 0)
    for k, z in enumerate(zs):
        if node.done:
            break
        ut = 1_000_000 + (k0 + k) * DT_US
        node.lc.cb(LATENT_STATE_CHANNEL, latent_state_message(ut, ut * 1e-6, z, ef, eu,
                                                              np.zeros(40)).encode())
        node._process(*node.pending)
        node.pending = None
        out.append((node.stage, node.blocked))
    node._k = k0 + len(zs)
    return out


def ref_schedule(D, tol, dur, ts, skip, gate, streak=1) -> list[tuple[int, bool]]:
    """Documented rules on whitened distances ``D (n, n_stages)``: skip -> furthest later stage
    whose tol holds (below the gate only up to gate - 1), reach (streak) or timeout advances,
    a timeout into the gate stage keeps the stage (blocked)."""
    n_st = len(tol)
    s, t0, rs, blocked, out = 0, ts[0], 0, False, []
    for i, t in enumerate(ts):
        if skip:
            hi = n_st - 1 if gate is None or s >= gate else gate - 1
            js = [j for j in range(s + 1, hi + 1) if tol[j] >= 0 and 0 <= D[i, j] < tol[j]]
            if js:
                s, t0, rs, blocked = js[-1], t, 0, False
        reached = 0 <= D[i, s] < tol[s]
        if streak > 1 and s + 1 < n_st:
            rs = rs + 1 if reached else 0
            reached = rs >= streak
        if reached or t - t0 >= dur[s]:
            if not reached and s + 1 == gate:
                blocked = True
            elif s + 1 < n_st:
                s, t0, rs, blocked = s + 1, t, 0, False
            else:
                out.append((s, blocked))
                break
        out.append((s, blocked))
    return out


def w3(node_mod) -> tuple[bool, str]:
    ok, notes = True, []
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as so:
        tmp = Path(tmp)
        lcs = node_mod.TorchLcs.from_deploy(c1_deploy(), DEVICE)
        g = synth_wp(tmp / "wp.npz", lcs)
        nowhere = np.zeros(lcs.n_x)
        on = {"stage_skip_ahead": True, "stage_no_timeout_into": "latch"}

        def run(frag, segs, streak=1):
            node = make_node(node_mod, tmp / "wp.npz", {**frag, "stage_reach_streak": streak},
                             tmp)
            res = []
            for zs in segs:
                res.append(feed(node, zs))
            return node, res

        def expect(name, cond):
            nonlocal ok
            if not cond:
                ok = False
                notes.append(name)

        _, (r,) = run({}, [[nowhere] * 200])
        st = [s for s, _ in r]
        expect("H1 off timeouts", sorted(set(st)) == [0, 1, 2, 3, 4]
               and set(np.diff(st)) <= {0, 1} and not any(b for _, b in r))
        n0 = so.getvalue().count("timeout blocked")
        node, (r1, r2, r3) = run({"stage_no_timeout_into": "latch"},
                                 [[nowhere] * 300, [g[3]], [nowhere] * 50])
        expect("H2 gate", max(s for s, _ in r1) == 3 and r1[-1] == (3, True)
               and so.getvalue().count("timeout blocked") - n0 == 1 and node.gate_stage == 4
               and r2[-1] == (4, False) and r3[-1][0] == 5)
        _, (r1, r2, r3) = run({"stage_skip_ahead": True}, [[g[2]], [g[1]] * 5, [g[5]]])
        expect("H3 skip", r1 == [(3, False)] and all(s == 3 for s, _ in r2) and r3 == [(6, False)])
        _, (r1, r2, r3) = run(on, [[g[5]] * 3, [g[3]], [g[5]]])
        expect("H4 skip + gate", [s for s, _ in r1] == [0, 0, 0] and r2 == [(4, False)]
               and r3 == [(6, False)])
        _, (r1,) = run({"stage_skip_ahead": True}, [[g[2]] * 3], streak=3)
        expect("H5 streak 3", [s for s, _ in r1] == [2, 2, 3])
        _, (r1, r2, r3) = run(on, [[nowhere] * 200, [g[5]] * 3, [g[3]]])
        expect("H6 blocked, no skip past the gate", r1[-1] == (3, True)
               and r2 == [(3, True)] * 3 and r3 == [(4, False)])
        # gate index: min_frame artifact (frames from 20), int, one stage, refusals
        synth_wp(tmp / "wp20.npz", lcs, FRAMES[1:])
        n20 = make_node(node_mod, tmp / "wp20.npz", on, tmp)
        n_int = make_node(node_mod, tmp / "wp.npz", {"stage_no_timeout_into": 2}, tmp)
        n_last = make_node(node_mod, tmp / "wp.npz", on, tmp, extra=["--stages", "last"])
        expect("H7 gate index", n20.gate_stage == 3 and n_int.gate_stage == 2
               and n_last.gate_stage is None)
        refused = 0
        for frag, goal in (({"stage_no_timeout_into": "latch"}, "deploy"),
                           ({"stage_no_timeout_into": 7}, None)):
            try:
                make_node(node_mod, tmp / "wp.npz", frag, tmp, goal=goal)
            except ValueError:
                refused += 1
        expect("H7 refusals", refused == 2)
        # random sequences vs the reference
        rng = np.random.default_rng(7)
        tol, dur = np.r_[np.ones(6), -1.0], np.r_[np.full(6, 3.0), 30.0]
        n_rand = 0
        for trial in range(200):
            frag = [{}, {"stage_skip_ahead": True}, {"stage_no_timeout_into": "latch"}, on,
                    {"stage_no_timeout_into": 2, "stage_skip_ahead": True}][trial % 5]
            streak = 1 if trial % 3 else 3
            zs = []
            while len(zs) < 400:
                k = rng.integers(-1, 7)
                z = nowhere if k < 0 else g[k] + rng.normal(size=lcs.n_x) * 0.3 * lcs.z_std_np
                zs += [z] * int(rng.integers(1, 30))
            node = make_node(node_mod, tmp / "wp.npz", {**frag, "stage_reach_streak": streak},
                             tmp)
            got = feed(node, zs)
            ts = (1_000_000 + np.arange(len(zs)) * DT_US) * 1e-6
            D = np.array([[lcs.whitened_dist(z, gk) for gk in g] for z in zs])
            want = ref_schedule(D, tol, dur, ts, frag.get("stage_skip_ahead", False),
                                node.gate_stage, streak)
            if got != want[:len(got)] or len(got) != len(want):
                ok = False
                notes.append(f"random {trial} {frag} streak {streak}")
            n_rand += 1
    return ok, (f"hand cases H1-H7 (timeouts; gate blocks once, reach passes, later timeouts "
                f"unaffected; skip, never back; skip capped below the gate; streak 3; gate index "
                f"4 / 3 with min_frame / int / None for one stage; 2 refusals); {n_rand} random "
                f"sequences == reference" + (f"; FAIL {notes[:4]}" if notes else ""))


# ---------------------------------------------------------------- W4


def replay(node_mod, run: Path, i: int, frag: dict, n: int, wp: str | None = None) -> dict:
    idx = json.loads((run / "index.json").read_text())
    a, ep = idx["args"], idx["episodes"][i]
    if wp is not None:
        a = {**a, "mppi_waypoints": wp}
    if DEVICE == "cpu":
        a, frag = {**a, "mppi_clearance": False}, {**frag, "clearance": None}
    r = oo.replay(node_mod, run, a, ep, frag, n)
    for row in r["rows"]:
        row.pop("solve_ms", None)
    r["stdout_clean"] = "\n".join(ln for ln in r["stdout"].splitlines()
                                  if not ln.startswith(("Module ", "Warp ", "   ")))
    return r


def w4(pre_node, tree_node) -> tuple[bool, str]:
    run = V18 / "iter0/collect"
    off_p = replay(pre_node, run, 0, {}, 120)
    off_t = replay(tree_node, run, 0, {}, 120)
    same = (off_p["rows"] == off_t["rows"] and off_p["stdout_clean"] == off_t["stdout_clean"]
            and all(np.array_equal(a[1], b[1]) for a, b in zip(off_p["plans"], off_t["plans"]))
            and len(off_p["plans"]) == len(off_t["plans"]))
    no_key = all("blocked" not in r for r in off_t["rows"])
    on = replay(tree_node, run, 0, {"stage_skip_ahead": True, "stage_no_timeout_into": "latch"},
                120)
    wp = tree_node.Waypoints.load(REPO / json.loads((run / "index.json").read_text())[
        "args"]["mppi_waypoints"])
    ut, z, k0 = on["ut"], on["z"], on["k0"]
    zs = z[k0:k0 + 120]
    ts = ut[k0:k0 + 120] * 1e-6
    lcs = tree_node.TorchLcs.from_deploy(json.loads((run / "index.json").read_text())[
        "args"]["deploy"], DEVICE)
    D = np.array([[lcs.whitened_dist(x, gk) for gk in wp.z_wp] for x in zs])
    tol = np.r_[wp.tol[:-1], -1.0]
    dur = np.r_[wp.max_duration_s[:-1], 30.0]
    streak = yaml.safe_load((run.parent / "mppi_collect.yaml").read_text())["stage_reach_streak"]
    want = ref_schedule(D, tol, dur, ts, True, 4, streak)
    by_ut = {int(u): s for u, s in zip(ut[k0:k0 + 120], want)}
    match = all(by_ut[r["utime"]] == (r["stage"], r["blocked"]) for r in on["rows"])
    stages = sorted({r["stage"] for r in on["rows"]})
    ok = same and no_key and match and all("blocked" in r for r in on["rows"])
    return ok, (f"V12-18 iter0 gv_06_0, 120 latents: off pre == tree (plans {len(off_t['plans'])}"
                f", rows, stdout) {same}, no 'blocked' key {no_key}; on (skip + latch gate): "
                f"{len(on['rows'])} solves, stages {stages}, stage/blocked == reference {match}; "
                f"stdout {[ln for ln in on['stdout'].splitlines() if 'skip' in ln][:2]}")


# ---------------------------------------------------------------- W7

STAGE_RE = re.compile(r"^\[learned-mpc\] stage (\d+) (?:reached|timeout) t=([0-9.]+)")
SKIP_RE = re.compile(r"^\[mppi\] skip ahead: stage \d+ -> (\d+) t=([0-9.]+)")


def w7(tree_node) -> tuple[bool, str]:
    """With ``prior_sync: stage`` (V12-20 UR z-30 #0, V12h): per solve the logged idx ==
    ``prior_sync_index`` on the node's own stage starts (advances and skips) and the passed
    ``u_demo`` == ``u_seq(idx)``, zero from the gate's waypoint frame while below the gate."""
    run = REPO / "data/lcs/realfirst/cmp/ur_slant/ee_ur_z-30"
    a1 = "data/lcs/realfirst/cmp/wp_schedule/artifacts/V12h_min20.npz"
    ok, res = True, []
    for name, frag, wp in (
            ("sync+A2", {"stage_skip_ahead": True}, None),
            ("sync+A1+A2+A3", {"stage_skip_ahead": True, "stage_no_timeout_into": "latch"}, a1),
            ("sync+A3", {"stage_no_timeout_into": "latch"}, None)):
        calls = []
        real = tree_node.mppi_plan

        def spy(*args, real=real, calls=calls, **kw):
            calls.append(None if kw.get("u_demo") is None else kw["u_demo"].copy())
            return real(*args, **kw)

        tree_node.mppi_plan = spy
        try:
            r = replay(tree_node, run, 0, {"prior_sync": "stage", **frag}, 200, wp)
        finally:
            tree_node.mppi_plan = real
        rows = r["rows"]
        calls = calls[-len(rows):]
        w = tree_node.Waypoints.load(REPO / (wp or json.loads((run / "index.json").read_text())[
            "args"]["mppi_waypoints"]))
        frames = [int(f) for f in w.frames]
        gate = (None if "stage_no_timeout_into" not in frag else
                int(np.flatnonzero(w.frames == w.latch)[0]) + 1)
        starts = {0: rows[0]["t"]}
        for ln in r["stdout"].splitlines():
            if m := STAGE_RE.match(ln):
                starts[int(m[1]) + 1] = float(m[2])
            elif m := SKIP_RE.match(ln):
                starts[int(m[1])] = float(m[2])
        bad = 0
        for row, ud in zip(rows, calls):
            s = row["stage"]
            want = tree_node.prior_sync_index(frames, s, r["offset"], row["t"] - starts[s], w.dt)
            u = w.u_seq(want, len(ud))
            if gate is not None and s < gate:
                u[max(0, frames[gate - 1] - want):] = 0.0
            bad += row["idx"] != want or not np.array_equal(ud, u)
        ok = ok and bad == 0 and len(rows) > 0
        res.append(f"{name}: {len(rows)} solves, stages {sorted({x['stage'] for x in rows})}, "
                   f"idx {rows[0]['idx']}..{rows[-1]['idx']}, {bad} mismatches")
    return ok, "; ".join(res)


# ---------------------------------------------------------------- W5


def mtimes(dirs) -> dict:
    return {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for d in dirs
            for p in sorted(d.rglob("*")) if p.is_file()}


def dry(mod, root: str, cfg: str, it: int) -> str:
    argv = sys.argv
    sys.argv = ["realfirst_loop.py", "--root", root, "--config", cfg, "--iter", str(it),
                "--step", "all", "--dry-run"]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            mod.main()
    finally:
        sys.argv = argv
    return out.getvalue()


def w5(pre_loop, tree_loop) -> tuple[bool, str]:
    ok, notes, res = True, [], []
    on_w = {"waypoints": {"min_frame": "offset", "skip_ahead": True, "no_timeout_into": "latch"}}
    for cell, root in (("C1_transport", C1), ("v12_18", V18)):
        cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
            (REPO / f"data/lcs/realfirst/cmp/configs/{cell}.yaml").read_text()))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "goal.json").write_bytes((root / "iter1/goal.json").read_bytes())

            def ctx_for(c, tmp=tmp, root=root):
                return types.SimpleNamespace(
                    cfg=c, i=1, seed_buffer=REPO / c["seed_buffer_merged"],
                    seed_buffers=[REPO / b for b in c["seed_buffers"]],
                    it=lambda i=None: tmp if i is None or i == 1 else root / f"iter{i}")

            outs, cmds = {}, {}
            with contextlib.redirect_stdout(io.StringIO()):
                for tag, mod, c in (("pre", pre_loop, cfg), ("tree", tree_loop, cfg),
                                    ("on", tree_loop, tree_loop.merge(cfg, on_w))):
                    p = tmp / f"{tag}.yaml"
                    mod.mppi_yaml(ctx_for(c), p, 0.2, 7)
                    outs[tag] = p.read_text()
                    cmds[tag] = [str(x) for x in mod.waypoints_cmd(ctx_for(c), "D.npz")]
            dp, do = yaml.safe_load(outs["pre"]), yaml.safe_load(outs["on"])
            added = {k: do.pop(k, None) for k in ("stage_skip_ahead", "stage_no_timeout_into")}
            good = (outs["pre"] == outs["tree"] and dp == do
                    and added == {"stage_skip_ahead": True, "stage_no_timeout_into": "latch"}
                    and cmds["pre"] == cmds["tree"]
                    and cmds["on"] == cmds["pre"][:-2] + ["--min-frame", "20"] + cmds["pre"][-2:])
            if outs["on"].splitlines()[0].count("waypoints.") != 2:
                good = False
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write(outs["on"].split("\n", 1)[1])
                f.flush()
                from task_common.mppi_lcs import MppiParams
                MppiParams.from_yaml(f.name)
            if not good:
                ok = False
                notes.append(f"{cell} yaml / cmd")
            res.append(f"{cell}: yaml + waypoints_cmd off == pre, on == pre + 2 keys / "
                       f"--min-frame 20")
    cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
        (REPO / "data/lcs/realfirst/cmp/configs/v12_18.yaml").read_text()))
    tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg), on_w))
    refused = 0
    bads = ({"waypoints": {"min_frame": -1}}, {"waypoints": {"min_frame": "motion"}},
            {"waypoints": {"min_frame": True}},
            {"waypoints": {"min_frame": "offset"}, "mppi_demo_offset_frames": None},
            {"waypoints": {"min_frame": 20}, "waypoint_bank": {"knn": 1}},
            {"waypoints": {"skip_ahead": "yes"}}, {"waypoints": {"no_timeout_into": 0}},
            {"waypoints": {"no_timeout_into": "sweep"}},
            {"waypoints": {"skip_ahead": True}, "goal_mode": "track"})
    for over in bads:
        try:
            tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg), over))
        except SystemExit:
            refused += 1
    ok = ok and refused == len(bads)
    # dry runs on the live roots: read only
    snap_dirs = [C1 / "iter0", C1 / "iter1", V18 / "iter0", V18 / "iter1"]
    before = mtimes(snap_dirs)
    n_dry = 0
    for root, cfgp in ((C1, "data/lcs/realfirst/cmp/configs/C1_transport.yaml"),
                       (V18, "data/lcs/realfirst/cmp/configs/v12_18.yaml")):
        for it in (0, 1):
            a = dry(pre_loop, str(root.relative_to(REPO)), cfgp, it)
            b = dry(tree_loop, str(root.relative_to(REPO)), cfgp, it)
            n_dry += 1
            if a != b:
                ok = False
                notes.append(f"dry {root.name} iter {it}")
    untouched = mtimes(snap_dirs) == before
    ok = ok and untouched
    return ok, ("; ".join(res) + f"; check_cfg refusals {refused}/{len(bads)}; dry-run stdout "
                f"pre == tree {n_dry}/4 (C1, V12-18 iters 0/1), iteration dirs untouched "
                f"{untouched}" + (f"; FAIL {notes}" if notes else ""))


# ---------------------------------------------------------------- W6


def w6(pre_gate, tree_gate, tree_loop) -> tuple[bool, str]:
    ok, notes = True, []
    jsons = sorted((REPO / "data/lcs/realfirst/cmp").glob("*/iter*/behaviour_gate/"
                                                           "behaviour_gate.json"))
    n = n_rec = 0
    for p in jsons:
        d = json.loads(p.read_text())
        r = d["rule"]
        args = (d["rows"]["cand"], d["rows"]["ref"], r["max_drop"], r["min_n"], r["phases"],
                r["max_hold_push_mm"], r.get("max_hold_push_increase_mm"))
        a, b = pre_gate.decide(*args), tree_gate.decide(*args)
        if a != b:
            ok = False
            notes.append(f"decide {p}")
        n += 1
        n_rec += a == d["decision"]
    pre = {"cand": {"n": 80, "fy_mm": 1.0, "ur_down_mm": 3.0},
           "ref": {"n": 80, "fy_mm": 0.5, "ur_down_mm": 3.5}}
    base = ({}, {}, 0.1, 10, [], None, None)
    cases = [((None, None), True, 0), ((0.5, None), True, 0), ((0.49, None), False, 1),
             ((None, 0.5), True, 0), ((None, 0.49), False, 1), ((0.2, 0.2), False, 2)]
    for (x, y), acc, nr in cases:
        dec = tree_gate.decide(*base, pre, x, y)
        if dec["accept"] != acc or len(dec["reasons"]) != nr or (
                ("prelatch_mm" in dec) != (x is not None or y is not None)):
            ok = False
            notes.append(f"rule {x} {y}: {dec}")
    if "prelatch_mm" in tree_gate.decide(*base):
        ok = False
    # pre-latch frames of the seed demos
    from model_gates import load_demo
    cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
        (REPO / "data/lcs/realfirst/cmp/configs/v12_18.yaml").read_text()))
    demos = [x["episode"] for b in cfg["seed_buffers"] for x in json.loads(
        (REPO / b).read_text())["rollouts"] if x["kind"] == "demo"]
    ds = [load_demo(Path(x)) for x in demos]
    fr = tree_gate.frame_list(ds, [], 1)
    spans = []
    for j, d in enumerate(ds):
        m0 = next(t for t, lab in enumerate(d["labels"]) if lab.startswith("move:"))
        got = [f["t"] for f in fr if f["src"] == ("demo", j) and f["prelatch"]]
        spans.append((m0, d["latch"]))
        if got != list(range(m0, d["latch"])):
            ok = False
            notes.append(f"prelatch frames demo {j}")
    st = tree_gate.prelatch_stats(np.tile([0, 0.002, 0, 0, 0, -0.004], (len(fr), 1)), fr)
    ok = ok and abs(st["fy_mm"] - 2.0) < 1e-9 and abs(st["ur_down_mm"] - 4.0) < 1e-9
    # loop gate command
    a = types.SimpleNamespace(iter=1, root=str(V18.relative_to(REPO)),
                              config="data/lcs/realfirst/cmp/configs/v12_18.yaml", smoke=False,
                              dry_run=True, collect_port=None, eval_port=None)
    cm = {}
    for tag, over in (("off", {}), ("on", {"behaviour_gate": {
            "max_prelatch_push_increase_mm": 0.3, "max_prelatch_descent_drop_mm": 0.4}})):
        ctx = tree_loop.Ctx(a)
        ctx.cfg = tree_loop.merge(ctx.cfg, over)
        cm[tag] = [str(x) for x in tree_loop.behaviour_cmds(ctx, "CK")[-1]]
    k = cm["on"].index("--max-hold-push-increase-mm")
    extra = cm["on"][k + 2:k + 6]
    good = (extra == ["--max-prelatch-push-increase-mm", "0.3", "--max-prelatch-descent-drop-mm",
                      "0.4"] and cm["on"][:k + 2] + cm["on"][k + 6:] == cm["off"])
    ok = ok and good
    return ok, (f"decide pre == tree on {n} gate jsons (== recorded {n_rec}); pre-latch rule "
                f"{len(cases)} threshold cases; pre-latch frames == [move0, latch) on {len(ds)} "
                f"demos (spans {sorted(set(spans))[:4]}...); loop gate cmd + 4 tokens only when "
                f"set {good}" + (f"; FAIL {notes[:3]}" if notes else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True)
    ap.add_argument("--only", default=None, help="comma list, e.g. W3,W4")
    ap.add_argument("--out", type=Path, default=None, help="json report")
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    pre_lcs, pre_node = load_pair(a.pre_dir, "pre")
    tree_lcs, tree_node = load_pair(None, "tree")
    pre_loop, tree_loop = load_script(a.pre_dir, "realfirst_loop", "pre"), \
        load_script(None, "realfirst_loop", "tree")
    pre_wp, tree_wp = load_script(a.pre_dir, "latent_waypoints", "pre"), \
        load_script(None, "latent_waypoints", "tree")
    pre_gate, tree_gate = load_script(a.pre_dir, "behaviour_gate", "pre"), \
        load_script(None, "behaviour_gate", "tree")
    tests = (("W1", lambda: w1(pre_lcs, tree_lcs)), ("W2", lambda: w2(pre_wp, tree_wp)),
             ("W3", lambda: w3(tree_node)), ("W4", lambda: w4(pre_node, tree_node)),
             ("W5", lambda: w5(pre_loop, tree_loop)),
             ("W6", lambda: w6(pre_gate, tree_gate, tree_loop)),
             ("W7", lambda: w7(tree_node)))
    only = set(a.only.split(",")) if a.only else None
    res = {}
    for name, fn in tests:
        if only and name not in only:
            continue
        ok, msg = fn()
        res[name] = {"ok": bool(ok), "msg": msg}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"device": DEVICE, **res}, indent=1) + "\n")
    n = sum(r["ok"] for r in res.values())
    print(f"{n}/{len(res)} PASS")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
