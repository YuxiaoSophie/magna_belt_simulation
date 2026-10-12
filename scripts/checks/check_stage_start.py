#!/usr/bin/env python3
"""Checks of V12-31's opt-in progress-gated start (``MppiParams.stage_start: progress``, node
``progress_start`` at the first latent, loop ``waypoints.start``).

S1 params: default None; every realfirst ``mppi*.yaml`` loads with ``to_dict`` == the pre
   module's + ``stage_start: None``; bad values refused; solver shared on / off.
S2 node rule (synthetic latents, solve stubbed): hand cases (out-of-reach start-state waypoints
   dropped, in-reach start == off for the whole sequence, nearest pick, nothing left -> latch
   stage, latch at stage 0 -> no-op, never back, an int gate caps the start, one stage -> no-op,
   refusals) and 200 random sequences (with skip
   ahead / latch gate / streak 3) == a reference implementation; ``progress_start`` == a brute
   force on 2000 random cases.
S3 node on recorded streams (real solves): pre vs tree with the option off: plans, log rows minus
   timing, stdout identical (3 streams); on: the first solve's JSONL ``start`` == the rule on the
   recorded first latent, no ``start`` key elsewhere or when off.
S4 with ``prior_sync: stage``: idx == ``prior_sync_index`` from the start stage on, ``u_demo`` ==
   ``u_seq(idx)``.
S5 loop: ``mppi_yaml`` bytes pre == tree without the key (C1, V12-18), with ``waypoints.start``
   the doc == pre + ``stage_start``; ``check_cfg`` refusals; ``--dry-run`` stdout pre == tree
   (C1 and V12-18 iters 0 / 1; nothing under those iterations written).

Clearance is off on the CPU.

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_stage_start.py \\
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
for _p in (REPO / "src", REPO / "scripts" / "lcs", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = importlib.util.spec_from_file_location(
    "cws", REPO / "scripts/checks/check_waypoint_schedule.py")
cws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cws)
torch = cws.torch
DEVICE, C1, V18, DT_US = cws.DEVICE, cws.C1, cws.V18, cws.DT_US
CMP = REPO / "data/lcs/realfirst/cmp"
FRAMES = [10, 20, 30, 33, 43, 53, 199]
ON = {"stage_start": "progress"}


# ---------------------------------------------------------------- S1


def s1(pre, tree) -> tuple[bool, str]:
    ok, notes = tree.MppiParams().stage_start is None, []
    yamls = sorted((REPO / "data/lcs/realfirst").rglob("mppi*.yaml"))
    n_on = 0
    for y in yamls:
        doc = yaml.safe_load(y.read_text()) or {}
        if "stage_start" in doc:
            n_on += tree.MppiParams.from_yaml(y).stage_start == "progress"
            continue
        if "value" in doc:  # later key (V12-39): tree only
            continue
        a, b = pre.MppiParams.from_yaml(y).to_dict(), tree.MppiParams.from_yaml(y).to_dict()
        for k in ("value", "latent_support", "ee_support"):  # later keys at defaults
            if k in b and b[k] is None and k not in a:
                b.pop(k)
        if b.pop("stage_start", "missing") is not None \
                or a.pop("stage_start", None) is not None or a != b:  # pre may have it
            ok = False
            notes.append(f"diff {y}")
    refused = 0
    for bad in ("nearest", True, 1, "Progress", ""):
        try:
            tree.MppiParams(stage_start=bad)
        except ValueError:
            refused += 1
    tree.MppiParams(stage_start="progress")
    lcs = tree.TorchLcs.from_deploy(cws.c1_deploy(), DEVICE)
    q = tree.MppiParams(K=8, H=6, n_iter=1, cuda_graph=False)
    import dataclasses
    same = tree.solver_for(lcs, q) is tree.solver_for(lcs, dataclasses.replace(
        q, stage_start="progress"))
    ok = ok and refused == 5 and same
    return ok, (f"{len(yamls) - n_on} yamls: to_dict == pre + stage_start None; {n_on} with it; "
                f"{refused}/5 bad refused; solver shared {same}"
                + (f"; {notes[:3]}" if notes else ""))


# ---------------------------------------------------------------- S2


def synth(path: Path, lcs, frames=FRAMES, static=2, latch=33) -> np.ndarray:
    """Goals: the first ``static`` at the start latent (0, then 0.3 sigma on axis 1), the rest
    10 sigma along axis k; z_traj[0] = 0; tol 1 (last -1), 3 s (last 30)."""
    n = len(frames)
    zs = np.asarray(lcs.z_std_np, np.float64)
    z_wp = np.stack([10.0 * np.eye(lcs.n_x)[k % lcs.n_x] * zs for k in range(n)])
    for k in range(static):
        z_wp[k] = 0.3 * k * np.eye(lcs.n_x)[1] * zs
    z_traj = np.zeros((200, lcs.n_x))
    z_traj[frames] = z_wp
    z_traj[0] = 0.0
    np.savez(path, frames=np.asarray(frames, np.int64), z_wp=z_wp,
             tol=np.r_[np.ones(n - 1), -1.0], max_duration_s=np.r_[np.full(n - 1, 3.0), 30.0],
             z_traj=z_traj, u_traj=np.zeros((200, 12)), dt=np.float64(lcs.dt),
             latch=np.int64(latch), first_hold=np.int64(53), deploy_sha256=np.str_(""),
             demo_episode=np.str_("synthetic"))
    return z_wp


def start_ref(d0, d_start, tol, n) -> int:
    """Brute force of the documented rule."""
    cands = []
    for k in range(n):
        if d_start[k] < tol[k] and not d0[k] < tol[k]:
            continue  # start-state waypoint the latent is outside
        cands.append(k)
    if not cands:
        return n
    best = cands[0]
    for k in cands[1:]:
        if d0[k] < d0[best]:
            best = k
    return best


def ref_schedule(D, tol, dur, ts, skip, gate, streak, s0) -> list[tuple[int, bool]]:
    """``check_waypoint_schedule.ref_schedule`` from stage ``s0``."""
    n_st = len(tol)
    s, t0, rs, blocked, out = s0, ts[0], 0, False, []
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


def s2(node_mod) -> tuple[bool, str]:
    ok, notes = True, []

    def expect(name, cond):
        nonlocal ok
        if not cond:
            ok = False
            notes.append(name)

    rng = np.random.default_rng(31)
    n_pure = 0
    for _ in range(2000):
        n = int(rng.integers(0, 6))
        tol = rng.uniform(0.5, 2.0, n)
        d0, ds = rng.uniform(0.0, 3.0, n), rng.uniform(0.0, 3.0, n)
        if n and rng.random() < 0.3:
            d0[rng.integers(0, n)] = d0[rng.integers(0, n)]  # ties
        expect("pure", node_mod.progress_start(d0, ds, tol, n) == start_ref(d0, ds, tol, n))
        n_pure += 1
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as so:
        tmp = Path(tmp)
        lcs = node_mod.TorchLcs.from_deploy(cws.c1_deploy(), DEVICE)
        zs = lcs.z_std_np
        g = synth(tmp / "wp.npz", lcs)
        far = 3.0 * np.eye(lcs.n_x)[5] * zs

        def run(frag, segs, wp="wp.npz", streak=1, **kw):
            node = cws.make_node(node_mod, tmp / wp, {**frag, "stage_reach_streak": streak}, tmp,
                                 **kw)
            return node, [cws.feed(node, z) for z in segs]

        n_off, (r_off,) = run({}, [[far] * 200])
        n_on, (r_on,) = run(ON, [[far] * 200])
        expect("H1 far: start 2, drop [0, 1]", r_off[0] == (0, False) and r_on[0] == (2, False)
               and n_on.start_info["stage"] == 2 and n_on.start_info["dropped"] == [0, 1]
               and n_off.start_info is None and "[mppi] start stage 2 (frame 30)" in so.getvalue())
        near0 = 0.1 * np.eye(lcs.n_x)[0] * zs
        seq = [near0] * 5 + [far] * 60 + [g[3]] * 3 + [far] * 100
        _, (a,) = run({}, [seq])
        _, (b,) = run(ON, [seq])
        expect("H2 in reach == off", a == b)
        node, (r,) = run(ON, [[g[2] + 0.2 * zs * np.eye(lcs.n_x)[0]] * 3])
        expect("H3 nearest 2", node.start_info["stage"] == 2
               and node.start_info["dropped"] == [0, 1])
        node, (r,) = run(ON, [[g[1]] * 2])
        expect("H4 nearest in reach", node.start_info["stage"] == 1
               and node.start_info["dropped"] == [])
        synth(tmp / "wp3.npz", lcs, static=3)
        node, (r,) = run(ON, [[far] * 3], wp="wp3.npz")
        expect("H5 none left -> latch", node.start_info["stage"] == 3 and r[0][0] == 3)
        synth(tmp / "wp33.npz", lcs, frames=[33, 43, 53, 199], static=1)
        node, (r,) = run(ON, [[far] * 3], wp="wp33.npz")
        expect("H6 latch at 0: no-op", node.start_info["stage"] == 0
               and node.start_info["dropped"] == [] and r[0][0] == 0)
        _, (_r1, r2) = run(ON, [[far], [g[0]] * 50])
        expect("H7 never back", min(s for s, _ in r2) >= 2)
        refused = 0
        for kw in ({"extra": ["--goal-mode", "track"]}, {"goal": "deploy"}):
            cws.make_node(node_mod, tmp / "wp.npz", {}, tmp, **kw)  # fine when off
            try:
                cws.make_node(node_mod, tmp / "wp.npz", ON, tmp, **kw)
            except ValueError as exc:
                refused += "stage_start" in str(exc)
        node, (r,) = run({**ON, "stage_no_timeout_into": 2}, [[far] * 3])
        n_last, (_r_last,) = run(ON, [[far] * 3], extra=["--stages", "last"])
        expect("H8 refusals; int gate 2 caps the start at 1; one stage: no-op",
               refused == 2 and node.start_info["stage"] == 1 and r[0][0] == 1
               and n_last.start_info["stage"] == 0 and len(n_last.stages) == 1)
        # random sequences vs the reference
        tol, dur = np.r_[np.ones(6), -1.0], np.r_[np.full(6, 3.0), 30.0]
        d_start = [lcs.whitened_dist(np.zeros(lcs.n_x), gk) for gk in g]
        n_rand, s0s = 0, {}
        frags = [ON, {**ON, "stage_skip_ahead": True}, {**ON, "stage_no_timeout_into": "latch"},
                 {**ON, "stage_skip_ahead": True, "stage_no_timeout_into": "latch"},
                 {**ON, "stage_no_timeout_into": 2}]
        for trial in range(200):
            frag = frags[trial % 5]
            streak = 1 if trial % 3 else 3
            zl = []
            while len(zl) < 300:
                k = rng.integers(-1, 7)
                z = far if k < 0 else g[k] + rng.normal(size=lcs.n_x) * 0.4 * zs
                zl += [z] * int(rng.integers(1, 30))
            node = cws.make_node(node_mod, tmp / "wp.npz", {**frag, "stage_reach_streak": streak},
                                 tmp)
            got = cws.feed(node, zl)
            ts = (1_000_000 + np.arange(len(zl)) * DT_US) * 1e-6
            D = np.array([[lcs.whitened_dist(z, gk) for gk in g] for z in zl])
            s0 = start_ref(D[0], d_start, tol, 3 if node.gate_stage is None
                           else min(3, node.gate_stage - 1))
            want = ref_schedule(D, tol, dur, ts, frag.get("stage_skip_ahead", False),
                                node.gate_stage, streak, s0)
            s0s[s0] = s0s.get(s0, 0) + 1
            if got != want or node.start_info["stage"] != s0:
                ok = False
                notes.append(f"random {trial} {frag} streak {streak}")
            n_rand += 1
    return ok, (f"progress_start == brute force on {n_pure} cases; hand cases H1-H8 (far -> 2 "
                f"dropping [0, 1]; in reach == off for 168 latents; nearest; none left -> latch "
                f"3; latch at 0 no-op; never back; 2 refusals, int gate caps, --stages last no-op); {n_rand} random sequences "
                f"(skip / gate / int gate / streak 3; start stages {dict(sorted(s0s.items()))}) == "
                f"reference"
                + (f"; FAIL {notes[:4]}" if notes else ""))


# ---------------------------------------------------------------- S3 / S4

STREAMS = (("v12_18/iter0/collect", 0), ("a1_sim/boxsync_up-20", 0),
           ("ur_slant/pivot_box_up-20", 0))


def s3(pre_node, tree_node, n: int) -> tuple[bool, str]:
    ok, res = True, []
    for rel, i in STREAMS:
        run = CMP / rel
        a = cws.replay(pre_node, run, i, {}, n)
        b = cws.replay(tree_node, run, i, {}, n)
        same = (a["rows"] == b["rows"] and a["stdout_clean"] == b["stdout_clean"]
                and len(a["plans"]) == len(b["plans"])
                and all(np.array_equal(x[1], y[1]) for x, y in zip(a["plans"], b["plans"])))
        no_key = all("start" not in r for r in b["rows"])
        ok = ok and same and no_key
        res.append(f"{rel} #{i}: off pre == tree ({len(b['plans'])} plans) {same}")
    run = CMP / "a1_sim/boxsync_up-20"
    on = cws.replay(tree_node, run, 0, ON, 20)
    args = json.loads((run / "index.json").read_text())["args"]
    w = tree_node.Waypoints.load(REPO / args["mppi_waypoints"])
    lcs = tree_node.TorchLcs.from_deploy(args["deploy"], DEVICE)
    z0 = on["z"][on["k0"]]
    n_l = int(np.flatnonzero(w.frames == w.latch)[0])
    d0 = [lcs.whitened_dist(z0, g) for g in w.z_wp[:n_l]]
    ds = [lcs.whitened_dist(w.z_traj[0], g) for g in w.z_wp[:n_l]]
    want = start_ref(d0, ds, w.tol, n_l)
    st = on["rows"][0].get("start", {})
    good = (st.get("stage") == want and all("start" not in r for r in on["rows"][1:])
            and np.allclose(st.get("d_now", []), d0) and on["rows"][0]["stage"] >= want
            and f"[mppi] start stage {want} " in on["stdout"])
    ok = ok and good
    res.append(f"on (A1 up-20): start {st.get('stage')} == rule {want} (d0 "
               f"{np.round(d0, 2).tolist()}, dropped {st.get('dropped')}), key on the first "
               f"solve only {good}")
    return ok, "; ".join(res)


STAGE_RE = re.compile(r"^\[learned-mpc\] stage (\d+) (?:reached|timeout) t=([0-9.]+)")


def s4(tree_node) -> tuple[bool, str]:
    run = CMP / "ur_slant/pivot_sync_up-20"
    calls = []
    real = tree_node.mppi_plan

    def spy(*args, **kw):
        calls.append(None if kw.get("u_demo") is None else kw["u_demo"].copy())
        return real(*args, **kw)

    tree_node.mppi_plan = spy
    try:
        r = cws.replay(tree_node, run, 0, ON, 60)
    finally:
        tree_node.mppi_plan = real
    rows = r["rows"]
    calls = calls[-len(rows):]
    w = tree_node.Waypoints.load(REPO / json.loads((run / "index.json").read_text())[
        "args"]["mppi_waypoints"])
    frames = [int(f) for f in w.frames]
    s0 = rows[0]["start"]["stage"]
    starts = {s0: r["ut"][r["k0"]] * 1e-6}
    for ln in r["stdout"].splitlines():
        if m := STAGE_RE.match(ln):
            starts[int(m[1]) + 1] = float(m[2])
    bad = 0
    for row, ud in zip(rows, calls):
        s = row["stage"]
        want = tree_node.prior_sync_index(frames, s, r["offset"], row["t"] - starts[s], w.dt)
        bad += row["idx"] != want or not np.array_equal(ud, w.u_seq(want, len(ud)))
    ok = bad == 0 and len(rows) > 0 and s0 == 2
    return ok, (f"sync + start (V12-21 sync up-20 #0): start stage {s0}, {len(rows)} solves, "
                f"stages {sorted({x['stage'] for x in rows})}, idx {rows[0]['idx']}.."
                f"{rows[-1]['idx']}, {bad} mismatches")


# ---------------------------------------------------------------- S5


def s5(pre_loop, tree_loop) -> tuple[bool, str]:
    ok, notes, res = True, [], []
    on_w = {"waypoints": {"start": "progress"}}
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
            added = do.pop("stage_start", None)
            good = (outs["pre"] == outs["tree"] and dp == do and added == "progress"
                    and cmds["pre"] == cmds["tree"] == cmds["on"]
                    and "stage_start (waypoints.start)" in outs["on"].splitlines()[0])
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write(outs["on"].split("\n", 1)[1])
                f.flush()
                good = good and tree_loop_params(f.name) == "progress"
            if not good:
                ok = False
                notes.append(f"{cell} yaml / cmd")
            res.append(f"{cell}: yaml + waypoints_cmd off == pre, on == pre + stage_start")
    cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
        (REPO / "data/lcs/realfirst/cmp/configs/v12_18.yaml").read_text()))
    tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg), on_w))
    refused = 0
    bads = ({"waypoints": {"start": "nearest"}}, {"waypoints": {"start": True}},
            {"waypoints": {"start": "progress"}, "goal_mode": "track"})
    for over in bads:
        try:
            tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg), over))
        except SystemExit:
            refused += 1
    ok = ok and refused == len(bads)
    snap_dirs = [C1 / "iter0", C1 / "iter1", V18 / "iter0", V18 / "iter1"]
    before = cws.mtimes(snap_dirs)
    n_dry = 0
    for root, cfgp in ((C1, "data/lcs/realfirst/cmp/configs/C1_transport.yaml"),
                       (V18, "data/lcs/realfirst/cmp/configs/v12_18.yaml")):
        for it in (0, 1):
            a = cws.dry(pre_loop, str(root.relative_to(REPO)), cfgp, it)
            b = cws.dry(tree_loop, str(root.relative_to(REPO)), cfgp, it)
            n_dry += a == b
            if a != b:
                ok = False
                notes.append(f"dry {root.name} iter {it}")
    untouched = cws.mtimes(snap_dirs) == before
    ok = ok and untouched
    return ok, ("; ".join(res) + f"; check_cfg refusals {refused}/{len(bads)}; dry-run stdout "
                f"pre == tree {n_dry}/4 (C1, V12-18 iters 0/1), iteration dirs untouched "
                f"{untouched}" + (f"; FAIL {notes}" if notes else ""))


def tree_loop_params(path: str):
    from task_common.mppi_lcs import MppiParams
    return MppiParams.from_yaml(path).stage_start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True)
    ap.add_argument("--only", default=None, help="comma list, e.g. S2,S3")
    ap.add_argument("--latents", type=int, default=60, help="S3 latents per stream")
    ap.add_argument("--out", type=Path, default=None, help="json report")
    a = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    pre_lcs, pre_node = cws.load_pair(a.pre_dir, "pre")
    tree_lcs, tree_node = cws.load_pair(None, "tree")
    pre_loop = cws.load_script(a.pre_dir, "realfirst_loop", "pre")
    tree_loop = cws.load_script(None, "realfirst_loop", "tree")
    tests = (("S1", lambda: s1(pre_lcs, tree_lcs)), ("S2", lambda: s2(tree_node)),
             ("S3", lambda: s3(pre_node, tree_node, a.latents)), ("S4", lambda: s4(tree_node)),
             ("S5", lambda: s5(pre_loop, tree_loop)))
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
