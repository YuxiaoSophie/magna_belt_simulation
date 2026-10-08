#!/usr/bin/env python3
"""Checks of the seated goal set (``MppiParams.goal_set`` / ``goal_set_tau``, ``goal_set.py``,
node last stage, ``realfirst_loop.py`` ``goal_set``; V12-30).

G1 params: defaults None / 0.0; every realfirst ``mppi*.yaml`` the pre-change module reads loads
   with ``to_dict`` == pre + the two defaulted keys; bad values / npz refused; the cached solver
   keys on the set.
G2 cost (synthetic, float32 + float64): a set of 1 (the goal) and of K copies of it give a
   ``plan_cost`` bit-identical to the single goal; a random set: on - off == w_q sum_knots +
   w_T last knot of (soft-min - single) from a numpy reference (tau 0 / 0.5 / 3); soft-min in
   [min, mean]; an explicit ``gset`` == from params; ``goal_set_dist`` == the reference.
G3 solver: pre vs tree ``mppi_plan`` with the set off (cold + warm, with / without the prior):
   plans and stats identical; a set of 1 == off exactly; with 14 seated demos the solver's
   ``plan_cost`` == ``rollout_cost`` of its plan and ``goal_dist_H`` == ``goal_set_dist``.
G4 node: recorded C1 stream (V12h, 7 stages): off pre == tree (rows minus timing, plans,
   stdout); set of 1 == off (plans; rows + ``d_single``); 14 demos: plans before the last stage
   == off, last-stage ``dist`` == ``goal_set_dist`` and ``d_single`` == the stage-goal distance;
   "reached" / skip-ahead on a set member far from the goal demo; track mode refused.
G5 loop: ``mppi_yaml`` bytes pre == tree without ``goal_set`` (C1, V12-18); with it the doc ==
   pre + ``goal_set`` / ``goal_set_tau``, the npz == ``goal_set.build`` on B_1 with deploy_1,
   built once; row counts with include_successes / strict; ``check_cfg`` refusals; ``--dry-run`` stdout pre == tree (C1, V12-18 iters 0/1),
   iteration dirs untouched.

CPU: clearance off (its torch path is minutes per solve there).

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_goal_set.py \\
        --pre-dir <dir with the pre-change mppi_lcs.py, mppi_controller_node.py, realfirst_loop.py>
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import io
import json
import math
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

import goal_set as gsm
import overshoot_offline as oo
import torch
from check_support_box import load_loop, load_pair
from check_waypoint_schedule import dry, mtimes

from task_common.latent_encoder import LATENT_STATE_CHANNEL, latent_state_message

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C1 = REPO / "data/lcs/realfirst/cmp/C1_transport"
V18 = REPO / "data/lcs/realfirst/cmp/v12_18"
RUN = C1 / "iter1/collect"
N_LAT = 30
N_FULL = 160  # reaches the last stage (8.0 s) of C1 iter1 episode 0
DEPLOY = json.loads((C1 / "iter1/deploy.json").read_text())["deploy"]
WP = REPO / "data/lcs/realfirst/cmp/C1_transport/iter1/latent_waypoints.npz"


def c1_args() -> tuple[dict, dict]:
    idx = json.loads((RUN / "index.json").read_text())
    return idx["args"], idx["episodes"][0]


def base_doc(no_prior: bool = False, small: bool = False) -> dict:
    a, _ = c1_args()
    doc = yaml.safe_load(Path(a["mppi_config"]).read_text())
    doc["action_noise_frac"] = 0.0
    if DEVICE == "cpu":
        doc["clearance"] = None
    if no_prior:
        doc.update(prior_alpha=0.0, prior_weight=0.0)
    if small:
        doc.update(K=256, n_iter=2)
    return doc


def recorded(n: int):
    _, ep = c1_args()
    rows = [json.loads(x) for x in (RUN / "controller" / Path(ep["file"]).with_suffix(
        ".solves.jsonl").name).read_text().splitlines()]
    with np.load(RUN / ep["file"], allow_pickle=True) as d:
        ut, z = np.asarray(d["utime"]), np.asarray(d["sim_latent"], np.float64)
        ef, eu = np.asarray(d["sim_ee_franka"]), np.asarray(d["sim_ee_ur"])
    k0 = int(np.flatnonzero(ut == rows[0]["utime"])[0])
    sl = slice(k0, min(len(ut), k0 + n))
    return ut[sl], z[sl], ef[sl], eu[sl], int(rows[0].get("idx", 20))


def sets(tmp: Path) -> dict:
    """Goal sets for V12h: ``one`` = the goal demo's seated latent (== the last waypoint),
    ``copies`` = it 5x, ``demo14`` = B_1's seated demos."""
    rows = json.loads((C1 / "iter0/buffer.json").read_text())["rollouts"]
    gs = gsm.build(DEPLOY, rows, ["demo"])
    with np.load(WP) as w:
        last = np.asarray(w["z_wp"][-1])
    out = {}
    for name, Z in (("one", last[None]), ("copies", np.repeat(last[None], 5, 0)),
                    ("demo14", gs["z_goals"])):
        np.savez(tmp / f"{name}.npz", z_goals=Z)
        out[name] = str(tmp / f"{name}.npz")
    out["Z14"], out["last"], out["episodes"] = gs["z_goals"], last, gs["episodes"]
    return out


def g1(pre, tree, S) -> tuple[bool, str]:
    p = tree.MppiParams()
    ok, notes, n = p.goal_set is None and p.goal_set_tau == 0.0, [], 0
    for y in sorted((REPO / "data/lcs/realfirst").rglob("mppi*.yaml")):
        if "goal_set" in (yaml.safe_load(y.read_text()) or {}):  # set on: tree only
            continue
        try:
            a = pre.MppiParams.from_yaml(y).to_dict()
        except ValueError:
            continue  # keys newer than the pre module
        b = tree.MppiParams.from_yaml(y).to_dict()
        n += 1
        dflt = {f.name: f.default for f in dataclasses.fields(tree.MppiParams)}
        for k in set(b) - set(a) - {"goal_set", "goal_set_tau"}:  # other later fields
            if b[k] == dflt.get(k, "x"):
                b.pop(k)
        if b.pop("goal_set", "x") is not None or b.pop("goal_set_tau", "x") != 0.0 \
                or a.pop("goal_set", None) is not None or a.pop("goal_set_tau", 0.0) != 0.0 \
                or a != b:  # pre may have them
            ok = False
            notes.append(f"diff {y}")
    refused = 0
    for bad in ({"goal_set_tau": -1.0}, {"goal_set_tau": math.nan}, {"goal_set_tau": math.inf},
                {"goal_set": 3}, {"goal_set": ["a.npz"]}):
        try:
            tree.MppiParams(**bad)
        except (ValueError, TypeError):
            refused += 1
    with tempfile.TemporaryDirectory() as tmp:
        bad_npz = 0
        for Z in (np.zeros((0, 10)), np.zeros((3, 9)), np.full((2, 10), np.nan)):
            np.savez(Path(tmp) / "b.npz", z_goals=Z)
            try:
                tree.load_goal_set(Path(tmp) / "b.npz", 10)
            except ValueError:
                bad_npz += 1
    lcs = tree.TorchLcs.from_deploy(DEPLOY, DEVICE)
    q = tree.MppiParams(K=8, H=6, n_iter=1, cuda_graph=False)
    s0 = tree.solver_for(lcs, q)
    s1 = tree.solver_for(lcs, dataclasses.replace(q, goal_set=S["demo14"]))
    s2 = tree.solver_for(lcs, dataclasses.replace(q, goal_set=S["demo14"], goal_set_tau=1.0))
    keyed = (s0 is not s1 and s1 is not s2 and s0.s_gset is None
             and torch.equal(s1.s_gset, lcs.tensor(S["Z14"])))
    ok = ok and refused == 5 and bad_npz == 3 and keyed
    return ok, (f"{n} yamls: to_dict == pre + goal_set None / tau 0; {refused}/5 bad params, "
                f"{bad_npz}/3 bad npz refused; solver keyed on the set {keyed}"
                + (f"; {notes[:3]}" if notes else ""))


def g2(tree, S) -> tuple[bool, str]:
    ok, notes = True, []
    rng = np.random.default_rng(0)
    K, H = 9, 6
    for dt in (torch.float64, torch.float32):
        lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu", dt)
        Z = lcs.tensor(rng.normal(size=(K, H + 1, lcs.n_x)) * 2.0)
        P = lcs.tensor(rng.normal(size=(K, H + 1, 6)) * 0.01)
        R, U = lcs.tensor(np.zeros((K, H + 1, 2))), lcs.tensor(rng.normal(size=(K, H, 12)) * 1e-3)
        half = lcs.tensor(np.full(12, 0.003))
        g = lcs.tensor(S["last"])
        p0 = tree.MppiParams(K=K, H=H)
        c0, n0, _ = tree.plan_cost(lcs, p0, Z, P, R, U, g, half, None)
        for name, tau in (("one", 0.0), ("one", 2.0), ("copies", 0.0), ("copies", 2.0)):
            c1, n1, _ = tree.plan_cost(lcs, dataclasses.replace(p0, goal_set=S[name],
                                                                goal_set_tau=tau),
                                       Z, P, R, U, g, half, None)
            if not (torch.equal(c0, c1) and torch.equal(n0, n1)):
                ok = False
                notes.append(f"{dt} {name} tau {tau} != single")
        if dt != torch.float64:
            continue
        G = rng.normal(size=(7, lcs.n_x)) * 2.0
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / "r.npz", z_goals=G)
            Zn, zs = Z.numpy(), lcs.z_std_np
            single = (((Zn[:, 1:] - S["last"]) / zs) ** 2).sum(-1)
            D = ((((Zn[:, 1:, None] - G) / zs) ** 2).sum(-1))
            for tau in (0.0, 0.5, 3.0):
                m = D.min(-1)
                ref = m if tau == 0 else m - tau * np.log(np.exp(-(D - m[..., None]) / tau)
                                                          .mean(-1))
                pr = dataclasses.replace(p0, goal_set=str(Path(tmp) / "r.npz"), goal_set_tau=tau)
                c1, n1, _ = tree.plan_cost(lcs, pr, Z, P, R, U, g, half, None)
                c2, _, _ = tree.plan_cost(lcs, pr, Z, P, R, U, g, half, None,
                                          gset=lcs.tensor(G))
                want = p0.w_q * (ref - single).sum(1) + p0.w_T * (ref - single)[:, -1]
                if not (np.allclose((c1 - c0).numpy(), want, rtol=1e-10, atol=1e-9)
                        and torch.equal(c1, c2) and torch.equal(n0, n1)
                        and np.all(ref >= m - 1e-12) and np.all(ref <= D.mean(-1) + 1e-12)):
                    ok = False
                    notes.append(f"random set tau {tau}")
                z = Zn[0, 3]
                gd = tree.goal_set_dist(z, G, zs, tau)
                d2 = (((z - G) / zs) ** 2).sum(-1)
                r = d2.min() if tau == 0 else d2.min() - tau * np.log(np.exp(-(d2 - d2.min())
                                                                            / tau).mean())
                if abs(gd - math.sqrt(r)) > 1e-12:
                    ok = False
                    notes.append(f"goal_set_dist tau {tau}")
    return ok, ("set of 1 / 5 copies (tau 0, 2) == single goal bit-identical (float32, float64); "
                "random 7-set tau 0 / 0.5 / 3: on - off == numpy reference (1e-10), in [min, "
                "mean], explicit gset == params, goal_set_dist == reference"
                + (f"; FAIL {notes[:4]}" if notes else ""))


def g3(pre, tree, S) -> tuple[bool, str]:
    _, z, ef, eu, offset = recorded(5)
    p0 = np.r_[ef[0, :3], eu[0, :3]]
    from latent_waypoints import Waypoints

    wp = Waypoints.load(WP)

    def solve(mod, doc, frag=None, seed=3):
        lcs = mod.TorchLcs.from_deploy(DEPLOY, DEVICE)
        prm = mod.MppiParams(**{**doc, **(frag or {})})
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        ud = wp.u_seq(offset + 14, prm.H) if prm.prior_weight > 0 else None
        U1, i1 = mod.mppi_plan(lcs, prm, z[0], p0, None, wp.z_wp[-1], None, gen, u_demo=ud)
        U2, i2 = mod.mppi_plan(lcs, prm, z[1], p0, None, wp.z_wp[-1], U1, gen, u_demo=ud)
        info = {k: v for k, v in {**i1, **{f"w_{k}": v for k, v in i2.items()}}.items()
                if "solve_ms" not in k}
        return U1, U2, info, lcs, prm, ud

    def same(x, y, skip=()):
        ok = np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
        for k in x[2]:
            if k in skip:
                continue
            va, vb = x[2][k], y[2][k]
            ok = ok and (np.array_equal(va, vb) if isinstance(va, np.ndarray) else
                         (va == vb or (isinstance(va, float) and math.isnan(va)
                                       and math.isnan(vb))))
        return ok

    ok, n = True, 0
    for doc in (base_doc(), base_doc(no_prior=True)):
        off_pre, off_tree = solve(pre, doc), solve(tree, doc)
        one = solve(tree, doc, {"goal_set": S["one"]})
        gd = abs(one[2]["goal_dist_H"] - off_tree[2]["goal_dist_H"]) < 1e-12
        ok = ok and same(off_pre, off_tree) and gd and same(
            off_tree, one, ("goal_dist_H", "w_goal_dist_H"))
        n += 1
    U1, _, info, lcs, prm, ud = solve(tree, base_doc(), {"goal_set": S["demo14"],
                                                         "goal_set_tau": 1.0})
    c, nb, _ = tree.rollout_cost(lcs, prm, z[0], p0, None, wp.z_wp[-1], U1, u_demo=ud)
    rel = abs(c - info["plan_cost"]) / max(abs(c), 1.0)
    gdh = tree.goal_set_dist(info["Z"][-1], S["Z14"], lcs.z_std_np, 1.0)
    ok = ok and rel < 1e-4 and nb == info["plan_violations"] \
        and abs(gdh - info["goal_dist_H"]) < 1e-9
    return ok, (f"{n} configs x cold + warm: set off pre == tree, set of 1 == off (plans, stats); "
                f"14-set tau 1: solver plan_cost {info['plan_cost']:.3f} == rollout_cost "
                f"{c:.3f} (rel {rel:.1e}), goal_dist_H == goal_set_dist ({gdh:.3f})")


def stream(node_mod, doc: dict, n: int, argv_extra=()) -> dict:
    ut, z, ef, eu, offset = recorded(n)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, log = Path(tmp) / "mppi.yaml", Path(tmp) / "solves.jsonl"
        cfg.write_text(yaml.safe_dump(doc))
        argv = ["--lcm-url", "fake", "--deploy", DEPLOY, "--config", str(cfg),
                "--max-duration-s", "30", "--log", str(log), "--goal", f"waypoints:{WP}",
                "--stages", "all", "--goal-mode", "stage", "--demo-offset-frames", str(offset),
                "--demo-actions", "auto", *argv_extra]
        args = node_mod.create_parser().parse_args(argv)
        out, plans = io.StringIO(), []
        with contextlib.redirect_stdout(out):
            node = node_mod.MppiControllerNode(args, oo.FakeLc())
            node.start()
            for k in range(len(ut)):
                msg = latent_state_message(int(ut[k]), ut[k] * 1e-6, z[k], ef[k], eu[k],
                                           np.zeros(40))
                n0 = node.n_solves
                node.lc.cb(LATENT_STATE_CHANNEL, msg.encode())
                node._process(*node.pending)
                node.pending = None
                if node.n_solves > n0:
                    plans.append(node.plan.copy())
            node.log.close()
        rows = [json.loads(x) for x in log.read_text().splitlines()]
    for r in rows:
        r.pop("solve_ms", None)
    stdout = "\n".join(ln for ln in out.getvalue().splitlines()
                       if not ln.startswith(("Module ", "Warp ", "   ")))
    return {"rows": rows, "plans": plans, "stdout": stdout, "node": node, "z": z}


def g4(pre_node, tree_node, tree, S) -> tuple[bool, str]:
    ok, notes = True, []
    a = stream(pre_node, base_doc(), N_LAT)
    b = stream(tree_node, base_doc(), N_LAT)
    same_off = (a["rows"] == b["rows"] and len(a["plans"]) == len(b["plans"])
                and all(np.array_equal(x, y) for x, y in zip(a["plans"], b["plans"]))
                and a["stdout"] == b["stdout"])
    ok = ok and same_off
    small = base_doc(small=True)
    off = stream(tree_node, small, N_FULL)
    one = stream(tree_node, {**small, "goal_set": S["one"]}, N_FULL)
    s14 = stream(tree_node, {**small, "goal_set": S["demo14"], "goal_set_tau": 0.5}, N_FULL)
    last = max(r["stage"] for r in off["rows"])
    n_last = sum(r["stage"] == last for r in off["rows"])
    plans_one = all(np.array_equal(x, y) for x, y in zip(off["plans"], one["plans"]))
    rows_one = all({k: v for k, v in r1.items() if k not in ("d_single", "dist", "goal_dist_H")}
                   == {k: v for k, v in r0.items() if k not in ("dist", "goal_dist_H")}
                   and abs(r1["dist"] - r0["dist"]) < 1e-9
                   and abs(r1["goal_dist_H"] - r0["goal_dist_H"]) < 1e-9
                   and (r1["stage"] != last or abs(r1["d_single"] - r0["dist"]) < 1e-12)
                   for r0, r1 in zip(off["rows"], one["rows"]))
    ok = ok and plans_one and rows_one and len(one["plans"]) == len(off["plans"])
    lcs = s14["node"].lcs
    zs = lcs.z_std_np
    before = [i for i, r in enumerate(off["rows"]) if r["stage"] < last]
    plans_before = all(np.array_equal(off["plans"][i], s14["plans"][i]) for i in before)
    d_ok, n_diff = True, 0
    for i, r in enumerate(s14["rows"]):
        if r["stage"] != last:
            d_ok = d_ok and "d_single" not in r
            continue
        n_diff += not np.array_equal(off["plans"][i], s14["plans"][i])
    # last-stage distances recomputed from the streamed latents
    ut, zz, _, _, _ = recorded(N_FULL)
    zmap = {int(u): z for u, z in zip(ut, zz)}
    goal = s14["node"].stages[-1].goal
    for r in s14["rows"]:
        if r["stage"] != last:
            continue
        z = zmap[int(r["utime"])]
        d_ok = d_ok and abs(r["dist"] - tree.goal_set_dist(z, S["Z14"], zs, 0.5)) < 1e-12 \
            and abs(r["d_single"] - lcs.whitened_dist(z, goal)) < 1e-12
    ok = ok and plans_before and d_ok and n_diff > 0
    # "reached" and skip-ahead on a set member far from the goal demo
    far = int(np.argmax([lcs.whitened_dist(g, S["last"]) for g in S["Z14"]]))
    zf = S["Z14"][far]
    res = {}
    for tag, frag in (("off", {}), ("on", {"goal_set": S["demo14"]})):
        for skip in (False, True):
            doc = {**small, **frag, "stage_skip_ahead": skip}
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                cfg = Path(tmp) / "m.yaml"
                cfg.write_text(yaml.safe_dump(doc))
                args = tree_node.create_parser().parse_args(
                    ["--lcm-url", "fake", "--deploy", DEPLOY, "--config", str(cfg), "--goal",
                     f"waypoints:{WP}", "--stages", "all", "--goal-mode", "stage",
                     "--goal-tol", "0.5", "--demo-actions", "auto"])
                nd = tree_node.MppiControllerNode(args, oo.FakeLc())
                if skip:
                    nd.stage_t0 = 0.0
                    nd._skip_ahead(0.1, zf)
                    res[(tag, "skip")] = nd.stage
                else:
                    nd.stage, nd.stage_t0 = len(nd.stages) - 1, 0.0
                    nd._stage_over(0.1, nd._stage_dist(zf, nd.stage), None)
                    res[(tag, "reach")] = nd.done
    n_st = len(s14["node"].stages)
    reach_ok = (res[("on", "reach")] is True and res[("off", "reach")] is False
                and res[("on", "skip")] == n_st - 1 and res[("off", "skip")] != n_st - 1)
    refused = False
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "m.yaml"
        cfg.write_text(yaml.safe_dump({**small, "goal_set": S["demo14"]}))
        args = tree_node.create_parser().parse_args(
            ["--lcm-url", "fake", "--deploy", DEPLOY, "--config", str(cfg), "--goal",
             f"waypoints:{WP}", "--goal-mode", "track", "--demo-actions", "auto"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tree_node.MppiControllerNode(args, oo.FakeLc())
        except ValueError as e:
            refused = "goal_set" in str(e)
    ok = ok and reach_ok and refused
    d_far = lcs.whitened_dist(zf, S["last"])
    return ok, (f"off pre == tree ({len(b['rows'])} rows, plans, stdout) {same_off}; {N_FULL} "
                f"latents (K 256, last stage {last}: {n_last} solves): set of 1 == off (plans "
                f"{plans_one}, rows + d_single {rows_one}); 14-set tau 0.5: plans before the "
                f"last stage == off {plans_before}, {n_diff}/{n_last} last-stage plans differ, "
                f"dist == goal_set_dist and d_single == stage-goal distance {d_ok}; demo at d "
                f"{d_far:.2f} from the goal demo (tol 0.5): reached on {res[('on', 'reach')]} / "
                f"off {res[('off', 'reach')]}, skip-ahead to stage {res[('on', 'skip')]} / "
                f"{res[('off', 'skip')]}; track mode refused {refused}"
                + (f"; FAIL {notes}" if notes else ""))


def g5(pre_loop, tree_loop) -> tuple[bool, str]:
    ok, notes, res = True, [], []
    for cell, root in (("C1_transport", C1), ("v12_18", V18)):
        cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
            (REPO / f"data/lcs/realfirst/cmp/configs/{cell}.yaml").read_text()))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for f in ("goal.json", "deploy.json"):
                (tmp / f).write_bytes((root / "iter1" / f).read_bytes())

            def ctx_for(c, tmp=tmp, root=root):
                return types.SimpleNamespace(
                    cfg=c, i=1, seed_buffer=REPO / c["seed_buffer_merged"],
                    seed_buffers=[REPO / b for b in c["seed_buffers"]],
                    it=lambda i=None: tmp if i is None or i == 1 else root / f"iter{i}")

            outs = {}
            with contextlib.redirect_stdout(io.StringIO()):
                for tag, mod, c in (("pre", pre_loop, cfg), ("tree", tree_loop, cfg),
                                    ("on", tree_loop, tree_loop.merge(cfg, {"goal_set": {}}))):
                    p = tmp / f"{tag}.yaml"
                    mod.mppi_yaml(ctx_for(c), p, 0.2, 7)
                    outs[tag] = p.read_text()
                st = (tmp / "goal_set.npz").stat().st_mtime_ns
                tree_loop.mppi_yaml(ctx_for(tree_loop.merge(cfg, {"goal_set": {}})),
                                    tmp / "on2.yaml", 0.2, 7)
                once = (tmp / "goal_set.npz").stat().st_mtime_ns == st
                g_s = tree_loop.goal_set_npz(ctx_for(tree_loop.merge(
                    cfg, {"goal_set": {"include_successes": True}})), dry=True)
                g_st = tree_loop.goal_set_npz(ctx_for(tree_loop.merge(
                    cfg, {"goal_set": {"strict": True}})), dry=True)
            dp, do = yaml.safe_load(outs["pre"]), yaml.safe_load(outs["on"])
            added = {k: do.pop(k, None) for k in ("goal_set", "goal_set_tau")}
            rows = json.loads((root / "iter0/buffer.json").read_text())["rollouts"]
            dep = json.loads((root / "iter1/deploy.json").read_text())["deploy"]
            ref = gsm.build(dep, rows, ["demo"])
            with np.load(tmp / "goal_set.npz") as f:
                Z, eps = f["z_goals"], [str(x) for x in f["episodes"]]
            js = json.loads((tmp / "goal_set.json").read_text())
            n_succ = sum(r["kind"] == "mpc" and r["label_human"] for r in rows)
            n_strict = sum(r["kind"] == "demo" and r["label_strict"] for r in rows)
            good = (outs["pre"] == outs["tree"] and dp == do
                    and added == {"goal_set": str(tmp / "goal_set.npz"), "goal_set_tau": 0.0}
                    and np.array_equal(Z, ref["z_goals"]) and eps == ref["episodes"]
                    and js["deploy_sha256"] == ref["deploy_sha256"] and once
                    and g_s["n"] == len(eps) + n_succ and g_st["n"] == n_strict
                    and "goal_set, goal_set_tau" in outs["on"].splitlines()[0])
            from task_common.mppi_lcs import MppiParams
            with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
                f.write(outs["on"].split("\n", 1)[1])
                f.flush()
                MppiParams.from_yaml(f.name)
            if not good:
                ok = False
                notes.append(f"{cell} yaml / npz")
            res.append(f"{cell}: off bytes == pre; on doc == pre + 2 keys, npz == build(B_1, "
                       f"deploy_1) ({len(eps)} seated demos, + {n_succ} successes with "
                       f"include_successes, {n_strict} with strict), built once {once}")
    cfg = tree_loop.merge(tree_loop.DEFAULTS, yaml.safe_load(
        (REPO / "data/lcs/realfirst/cmp/configs/v12_18.yaml").read_text()))
    tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg), {"goal_set": {
        "kinds": ["demo", "excitation"], "include_successes": True, "softmin_tau": 1.0}}))
    refused = 0
    bads = ({"goal_set": {"kinds": []}}, {"goal_set": {"kinds": ["miles"]}},
            {"goal_set": {"include_successes": "yes"}}, {"goal_set": {"softmin_tau": -1}},
            {"goal_set": {"softmin_tau": True}}, {"goal_set": {"tau": 1.0}},
            {"goal_set": {"strict": "yes"}},
            {"goal_set": {}, "goal_mode": "track"})
    for over in bads:
        try:
            tree_loop.check_cfg(tree_loop.merge(copy.deepcopy(cfg), over))
        except SystemExit:
            refused += 1
    ok = ok and refused == len(bads)
    snap = [C1 / "iter0", C1 / "iter1", V18 / "iter0", V18 / "iter1"]
    before = mtimes(snap)
    n_dry, on_line = 0, False
    for root, cfgp in ((C1, "data/lcs/realfirst/cmp/configs/C1_transport.yaml"),
                       (V18, "data/lcs/realfirst/cmp/configs/v12_18.yaml")):
        for it in (0, 1):
            a = dry(pre_loop, str(root.relative_to(REPO)), cfgp, it)
            b = dry(tree_loop, str(root.relative_to(REPO)), cfgp, it)
            n_dry += a == b
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        f.write(yaml.safe_dump({**yaml.safe_load(
            (REPO / "data/lcs/realfirst/cmp/configs/v12_18.yaml").read_text()),
            "goal_set": {"include_successes": True}}))
        f.flush()
        c = dry(tree_loop, str(V18.relative_to(REPO)), f.name, 1)
        on_line = "goal set 16 seated latents" in c
    untouched = mtimes(snap) == before
    ok = ok and n_dry == 4 and on_line and untouched
    return ok, ("; ".join(res) + f"; check_cfg refusals {refused}/{len(bads)}; dry-run stdout pre "
                f"== tree {n_dry}/4 (C1, V12-18 iters 0/1), on: goal set line {on_line}; "
                f"iteration dirs untouched {untouched}" + (f"; FAIL {notes}" if notes else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="json report")
    ap.add_argument("--only", nargs="+", default=None, help="e.g. G5")
    a = ap.parse_args()
    pre_lcs, pre_node = load_pair(a.pre_dir, "pre")
    tree_lcs, tree_node = load_pair(None, "tree")
    pre_loop, tree_loop = load_loop(a.pre_dir, "pre"), load_loop(None, "tree")
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        S = sets(Path(tmp))
        for name, fn in (("G1", lambda: g1(pre_lcs, tree_lcs, S)), ("G2", lambda: g2(tree_lcs, S)),
                         ("G3", lambda: g3(pre_lcs, tree_lcs, S)),
                         ("G4", lambda: g4(pre_node, tree_node, tree_lcs, S)),
                         ("G5", lambda: g5(pre_loop, tree_loop))):
            if a.only and name not in a.only:
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
