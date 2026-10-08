#!/usr/bin/env python3
"""Checks of the stage-synced demo action prior (``MppiParams.prior_sync``, node
``prior_sync_index``, ``realfirst_loop.py`` ``demo_prior.sync``).

P1 params: default None; every realfirst ``mppi*.yaml`` loads with ``to_dict`` == the pre-change
   module's plus ``prior_sync: None``; "stage" accepted, other values refused; the cached
   solver ignores the field.
P2 node identity (sync off): pre-change node + solver vs the tree's on recorded streams (FULL
   iter0 ep4 with the prior, the same without it, C1 iter1 ep0): plans, log rows minus
   timing and stdout identical.
P3 rule: synthetic stage sequences -> expected indices (start at max(previous waypoint frame,
   offset), demo rate, clamp at the stage's frame, jump on advance, offset floor, one stage =
   the wall clock); random sequences: never past the stage frame, never below its start,
   monotone; the node's ``_progress`` == the rule; refusals (deploy goal, track mode).
P4 node with sync on (FULL iter0 ep4): per solve the logged idx == the rule on the stage
   start times of the node's own stage lines, and the passed ``u_demo`` == ``u_seq(idx)``.
P5 loop: ``mppi_yaml`` bytes identical pre vs tree without ``sync``; with ``sync: stage``
   the doc == pre + ``prior_sync: stage``; ``check_cfg`` refuses a bad sync / track mode.

    uv run --frozen python scripts/checks/check_prior_sync.py --pre-dir <dir with the
        pre-change src/task_common/mppi_lcs.py, scripts/lcs/{mppi_controller_node,
        realfirst_loop}.py>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # cpu: clearance off (torch path slow)
FULL = REPO / "data/lcs/realfirst/cmp/full/iter0/collect"
C1 = REPO / "data/lcs/realfirst/cmp/C1_transport/iter1/collect"
N_LAT = 120
STAGE_RE = re.compile(r"^\[learned-mpc\] stage (\d+) (?:reached|timeout) t=([0-9.]+)")


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


def load_loop(d: Path | None, tag: str):
    """``realfirst_loop`` from ``d``, run as if at the tree's path (same REPO)."""
    path = (REPO if d is None else d) / "scripts/lcs/realfirst_loop.py"
    mod = types.ModuleType(f"realfirst_loop_{tag}")
    mod.__file__ = str(REPO / "scripts/lcs/realfirst_loop.py")
    sys.modules[mod.__name__] = mod
    exec(compile(path.read_text(), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def episode(run: Path, i: int) -> tuple[dict, dict]:
    idx = json.loads((run / "index.json").read_text())
    return idx["args"], idx["episodes"][i]


def run_replay(node_mod, run: Path, i: int, frag: dict, n: int | None, spy=None) -> dict:
    a, ep = episode(run, i)
    if DEVICE == "cpu":
        a, frag = {**a, "mppi_clearance": False}, {**frag, "clearance": None}
    real = node_mod.mppi_plan
    if spy is not None:
        node_mod.mppi_plan = spy(real)
    try:
        r = oo.replay(node_mod, run, a, ep, frag, n)
    finally:
        node_mod.mppi_plan = real
    for row in r["rows"]:
        row.pop("solve_ms", None)
    r["stdout_clean"] = "\n".join(ln for ln in r["stdout"].splitlines()
                                  if not ln.startswith(("Module ", "Warp ", "   ")))
    return r


def p1(pre, tree) -> tuple[bool, str]:
    ok, notes = tree.MppiParams().prior_sync is None, []
    yamls = sorted((REPO / "data/lcs/realfirst").rglob("mppi*.yaml"))
    n_on = 0
    for y in yamls:
        if {"stage_skip_ahead", "stage_no_timeout_into", "prior_mode", "bc_prior", "goal_set",
                "goal_set_tau", "stage_start", "value"} & set(yaml.safe_load(y.read_text()) or {}):
            continue  # schedule / policy-prior keys: tree only
        if (yaml.safe_load(y.read_text()) or {}).get("prior_sync"):  # sync on: tree only
            n_on += tree.MppiParams.from_yaml(y).prior_sync == "stage"
            continue
        a, b = pre.MppiParams.from_yaml(y).to_dict(), tree.MppiParams.from_yaml(y).to_dict()
        for k, v in (("support_box", None), ("stage_skip_ahead", False),
                     ("stage_no_timeout_into", None), ("prior_mode", "demo"),
                     ("bc_prior", None),
                     ("goal_set", None), ("goal_set_tau", 0.0),
                     ("stage_start", None), ("value", None)):  # later fields at their defaults
            if k in b and b[k] == v and k not in a:
                b.pop(k)
        if b.pop("prior_sync", "missing") is not None or a.pop("prior_sync", None) is not None \
                or a != b:
            ok = False
            notes.append(f"diff {y}")
    ok = ok and tree.MppiParams(prior_sync="stage").prior_sync == "stage"
    refused = 0
    for bad in ("time", "Stage", 1, ""):
        try:
            tree.MppiParams(prior_sync=bad)
        except ValueError:
            refused += 1
    ok = ok and refused == 4
    lcs = tree.TorchLcs.from_deploy(episode(FULL, 0)[0]["deploy"], DEVICE)
    p = tree.MppiParams()
    same = tree.solver_for(lcs, p) is tree.solver_for(lcs, tree.MppiParams(prior_sync="stage"))
    return ok and same, (f"{len(yamls) - n_on} yamls: to_dict == pre + prior_sync None; {n_on} "
                         f"with sync on load; 'stage' "
                         f"accepted, {refused}/4 bad refused; solver shared {same}"
                         + (f"; {notes[:3]}" if notes else ""))


def p2(pre_node, tree_node) -> tuple[bool, str]:
    res = []
    for name, run, i, frag in (("full ep4 prior", FULL, 4, {}),
                               ("full ep4 no prior", FULL, 4,
                                {"prior_alpha": 0.0, "prior_weight": 0.0}),
                               ("C1 iter1 ep0 prior", C1, 0, {})):
        a = run_replay(pre_node, run, i, frag, N_LAT)
        b = run_replay(tree_node, run, i, frag, N_LAT)
        plans = len(a["plans"]) == len(b["plans"]) and all(
            ka == kb and np.array_equal(pa, pb) for (ka, pa), (kb, pb) in
            zip(a["plans"], b["plans"]))
        same = (plans, a["rows"] == b["rows"], a["stdout_clean"] == b["stdout_clean"])
        res.append((name, all(same), len(a["plans"]), same))
    return all(r[1] for r in res), "; ".join(
        f"{n}: {k} solves, plans/rows/stdout {s}" for n, _, k, s in res)


def p3(tree_node) -> tuple[bool, str]:
    f = tree_node.prior_sync_index
    ok, notes = True, []
    dt = 0.075

    def chk(c, msg):
        nonlocal ok
        if not c:
            ok = False
            notes.append(msg)

    fr = [10, 20, 30, 33, 43, 53, 199]
    # (stage, elapsed frames in the stage, offset) -> index
    cases = [((0, 0, 0), 0), ((0, 7, 0), 7), ((0, 10, 0), 10), ((0, 25, 0), 10),
             ((1, 0, 0), 10), ((1, 4, 0), 14), ((1, 40, 0), 20),
             ((0, 0, 20), 20), ((0, 30, 20), 20), ((1, 5, 20), 20), ((2, 0, 20), 20),
             ((2, 6, 20), 26), ((2, 99, 20), 30), ((3, 0, 20), 30), ((3, 2, 20), 32),
             ((3, 9, 20), 33), ((4, 1, 20), 34), ((6, 0, 20), 53), ((6, 100, 20), 153),
             ((6, 999, 20), 199), ((2, 3, 40), 40), ((4, 0, 40), 40), ((4, 5, 40), 43),
             ((5, 0, 40), 43)]
    for (s, e, off), want in cases:
        got = f(fr, s, off, e * dt, dt)
        chk(got == want, f"stage {s} +{e} off {off}: {got} != {want}")
    # one stage (--stages last) = the wall clock
    for e in range(0, 260, 7):
        chk(f([199], 0, 20, e * dt, dt) == min(20 + e, 199), f"last-only +{e}")
    rng = np.random.default_rng(0)
    n_seq = 0
    for _ in range(200):
        n = int(rng.integers(2, 9))
        frs = np.sort(rng.choice(np.arange(1, 200), n, replace=False)).tolist()
        off = int(rng.integers(0, 60))
        t, s, t0, prev = 0.0, 0, 0.0, -1
        while s < n:
            t += float(rng.uniform(0.02, 0.2))
            k = f(frs, s, off, t - t0, dt)
            lo = max(off, frs[s - 1] if s > 0 else 0)
            chk(lo <= k <= max(lo, frs[s]) and k >= prev, f"seq {frs} off {off} s {s}")
            prev = k
            if rng.uniform() < 0.08:  # stage advance (reach or timeout)
                s, t0 = s + 1, t
        n_seq += 1
    # the node's _progress on scripted stage switches
    a, _ = episode(FULL, 4)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "m.yaml"
        doc = yaml.safe_load(Path(a["mppi_config"]).read_text())
        cfg.write_text(yaml.safe_dump({**doc, "prior_sync": "stage", "clearance": None}))
        argv = ["--lcm-url", "fake", "--deploy", a["deploy"], "--config", str(cfg),
                "--goal", f"waypoints:{REPO / a['mppi_waypoints']}", "--stages", "all",
                "--goal-mode", "stage", "--demo-offset-frames", "20", "--demo-actions", "auto"]
        with contextlib.redirect_stdout(io.StringIO()):
            node = tree_node.MppiControllerNode(tree_node.create_parser().parse_args(argv),
                                                oo.FakeLc())
            frames = [int(node.wp.frames[st.index]) for st in node.stages]
            node.t_mpc_start = node.stage_t0 = 100.0
            want_node, got_node = [], []
            script = [(0.0, 0), (0.3, 0), (0.45, 1), (2.0, 1), (3.45, 2), (3.6, 2), (4.5, 2),
                      (5.0, 3), (5.1, 4), (5.4, 4), (9.0, 5), (9.3, 6), (30.0, 6)]
            for dts, s in script:
                if s != node.stage:
                    node.stage, node.stage_t0 = s, 100.0 + dts
                got_node.append(node._progress(100.0 + dts, None)[0])
                want_node.append(f(frames, s, 20, 100.0 + dts - node.stage_t0, node.dt))
            chk(got_node == want_node, f"node {got_node} vs {want_node}")
            chk(got_node == [20, 20, 20, 20, 20, 22, 30, 30, 33, 37, 43, 53, 199],
                f"node expected {got_node}")
            refused = 0
            for extra in (["--goal", "deploy"], ["--goal-mode", "track"]):
                try:
                    tree_node.MppiControllerNode(
                        tree_node.create_parser().parse_args(argv + extra), oo.FakeLc())
                except ValueError:
                    refused += 1
        chk(refused == 2, f"refusals {refused}/2")
    return ok, (f"{len(cases)} cases + last-only wall clock + {n_seq} random sequences (clamp, "
                f"start floor, monotone); node _progress {got_node}; refusals {refused}/2"
                + (f"; FAIL {notes[:4]}" if notes else ""))


def p4(tree_node) -> tuple[bool, str]:
    calls = []

    def spy(real):
        def fn(lcs, params, z0, p0, rot0, goal, u_prev, gen, **kw):
            calls.append(None if kw.get("u_demo") is None else kw["u_demo"].copy())
            return real(lcs, params, z0, p0, rot0, goal, u_prev, gen, **kw)
        return fn

    r = run_replay(tree_node, FULL, 4, {"prior_sync": "stage"}, 260, spy=spy)
    rows = r["rows"]
    calls = calls[-len(rows):]  # drop the warm-up solves
    from latent_waypoints import Waypoints

    a, _ = episode(FULL, 4)
    wp = Waypoints.load(REPO / a["mppi_waypoints"])
    starts = {0: rows[0]["t"]}
    for ln in r["stdout"].splitlines():
        m = STAGE_RE.match(ln)
        if m:
            starts[int(m.group(1)) + 1] = float(m.group(2))
    ok, bad = True, []
    for row, ud in zip(rows, calls):
        s = row["stage"]
        want = tree_node.prior_sync_index(wp.frames, s, r["offset"], row["t"] - starts[s],
                                          wp.dt)
        if row["idx"] != want or ud is None or not np.array_equal(ud, wp.u_seq(want, len(ud))):
            ok = False
            bad.append((row["t"], s, row["idx"], want))
    idx_by_stage = {s: (min(x["idx"] for x in rows if x["stage"] == s),
                        max(x["idx"] for x in rows if x["stage"] == s))
                    for s in sorted({x["stage"] for x in rows})}
    return ok, (f"{len(rows)} solves, stages {sorted(starts)}; idx range per stage "
                f"{idx_by_stage}; idx and u_demo == rule" + (f"; FAIL {bad[:4]}" if bad else ""))


def p5(pre_loop, tree_loop) -> tuple[bool, str]:
    cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
        (REPO / "data/lcs/realfirst/cmp/configs/C1_transport.yaml").read_text()))
    ok, notes = True, []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        it = REPO / "data/lcs/realfirst/cmp/C1_transport/iter1"

        def ctx_for(c):
            return types.SimpleNamespace(cfg=c, it=lambda i=None: it)

        outs = {}
        for tag, mod, c in (("pre", pre_loop, cfg), ("tree", tree_loop, cfg),
                            ("sync", tree_loop, tree_loop.merge(
                                cfg, {"demo_prior": {"sync": "stage"}}))):
            p = tmp / f"{tag}.yaml"
            mod.mppi_yaml(ctx_for(c), p, 0.2, 7)
            outs[tag] = p.read_text()
        if outs["pre"] != outs["tree"]:
            ok = False
            notes.append("off bytes differ")
        dp, ds = yaml.safe_load(outs["pre"]), yaml.safe_load(outs["sync"])
        if ds.pop("prior_sync", None) != "stage" or dp != ds:
            ok = False
            notes.append("sync doc")
        refused = 0
        for over in ({"demo_prior": {"sync": "time"}},
                     {"demo_prior": {"sync": "stage"}, "goal_mode": "track"}):
            try:
                tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg), over))
            except SystemExit:
                refused += 1
        tree_loop.check_cfg(tree_loop.merge(cfg, {"demo_prior": {"sync": "stage"}}))
        ok = ok and refused == 2
    head = outs["sync"].splitlines()[0]
    return ok, (f"off: bytes identical; sync: doc == pre + prior_sync stage (header '...{head[-60:]}');"
                f" check_cfg refusals {refused}/2" + (f"; FAIL {notes}" if notes else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="json report")
    a = ap.parse_args()
    pre_lcs, pre_node = load_pair(a.pre_dir, "pre")
    tree_lcs, tree_node = load_pair(None, "tree")
    pre_loop, tree_loop = load_loop(a.pre_dir, "pre"), load_loop(None, "tree")
    res = {}
    for name, fn in (("P1", lambda: p1(pre_lcs, tree_lcs)), ("P2", lambda: p2(pre_node, tree_node)),
                     ("P3", lambda: p3(tree_node)), ("P4", lambda: p4(tree_node)),
                     ("P5", lambda: p5(pre_loop, tree_loop))):
        ok, msg = fn()
        res[name] = {"ok": bool(ok), "msg": msg}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    n = sum(r["ok"] for r in res.values())
    print(f"{n}/{len(res)} PASS")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
