#!/usr/bin/env python3
"""Checks of the V12-57 loop keys: ``refit.extra_args``, ``graph`` (cloud graph rebuilt before
each refit), ``deploy_policy`` keep_best, ``eval_suite`` hard, ``eval_every: 1``, the JEPA export
env. No sim; CPU only (CUDA_VISIBLE_DEVICES="").

K1 config: ``check_cfg`` refuses bad blocks; v12_36 / v12_46 / v12_57 accepted; eval_suite hard
   == run_hard_suite.SUITE (11 runs x 2, grasp reset seeds); eval_due for every 0 / 1 / 2 and
   keep_best at i = 0.
K2 dry-run identity pre vs tree (stdout bytes): v12_36 and v12_46 iterations 0-9, C1_transport 0/1.
K3 v12_57 dry-run on a fresh root, iterations 0-2: rc 0, nothing written; step order (eval, deploy
   before collect); i0 adopts seed_model; i1 / i2 build_cloud_graph on B_i -> iter<i>/graph and the
   refit gets it (--graph-file, --graph-weight 1, init = seed, lcs_only); JEPA env on the lcs
   exports; 11 hard-suite eval runs (--repeats 2, scripted resets with SUITE's seeds on the
   grasps); i2: no candidate export / eval, deploy keeps the deployed model; extra_args appended.
K4 keep_best_pick table (initial, not evaluated, same model, ties, min_gain, losses).
K5 dry-run with stubbed evals / deploy records: i1 win / loss / tie, i2 keep.
K6 scratch iterations 0-1 (seed, trust, export, stubbed eval, deploy; i1: real graph build and
   lcs_only JEPA refit of 1 epoch on a 20-episode B_1, CPU): deploy_0 == the V21c sim arm's
   (float noise), JEPA env recorded; refit.json graph stats / train_objective jepa, refit
   checkpoint carries the rebuilt graph; i1 deploy decision per stubbed score (win and loss);
   learning_curve.md keep-best and graph sections.

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_keep_best.py \\
        --pre-dir <pre-change tree (repo layout)> [--only K1,..] [--out <json>]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from check_seed_model import dry_main, load_loop, ns, req

CMP = REPO / "data/lcs/realfirst/cmp"
CFG = CMP / "configs"
V21C = CMP / "encoder_study/sim/V21c"


def cfg57() -> dict:
    return yaml.safe_load((CFG / "v12_57.yaml").read_text())


def k1(ctx) -> str:
    rl = ctx.tree
    import run_hard_suite as rhs

    for n in ("v12_36", "v12_46", "v12_57"):
        rl.check_cfg(rl.merge(rl.DEFAULTS, yaml.safe_load((CFG / f"{n}.yaml").read_text())))
    base = rl.merge(rl.DEFAULTS, cfg57())
    bad = [{"refit": {**base["refit"], "extra_args": "--jepa"}},
           {"refit": {**base["refit"], "extra_args": [1]}},
           {"graph": True}, {"graph": {"stride": 0}}, {"graph": {"weight": -1}},
           {"graph": {"rebuild": 1}}, {"graph": {"extra": "--k 10"}}, {"graph": {"k": 3}},
           {"deploy_policy": {"mode": "best"}}, {"deploy_policy": {"metric": "eval_rmse"}},
           {"deploy_policy": {"min_gain": -1}}, {"deploy_policy": {"min_gain": 0.5}},
           {"deploy_policy": {"mode": "keep_best"}, "eval_every": 0},
           {"eval_suite": {"name": "seat6"}}, {"eval_suite": {"repeats": 0}}, {"eval_suite": "hard"},
           {"eval_every": -1}, {"eval_every": 1.5},
           {"eval_runs": [{**base["eval_runs"][0], "reset_seed": "x"}]}]
    for b in bad:
        try:
            rl.check_cfg({**copy.deepcopy(base), **b})
        except SystemExit:
            continue
        raise AssertionError(f"accepted {b}")
    runs = rl.eval_runs_of(base)
    req(len(runs) == 11 and sum(r["repeats"] for r in runs) == 22
        and [(r["start_states"], r["variants"] or "", r["reset_seed"]) for r in runs]
        == [(s, v, seed) for _, s, v, seed in rhs.SUITE], "eval_suite runs")
    req(rl.eval_runs_of(rl.merge(rl.DEFAULTS, {})) == rl.DEFAULTS["eval_runs"], "eval_runs default")

    def due(i, every, kb=False):
        c = types.SimpleNamespace(smoke=False, i=i, a=types.SimpleNamespace(force=None),
                                  cfg={"eval_every": every, **({"deploy_policy": {
                                      "mode": "keep_best"}} if kb else {})})
        return rl.eval_due(c)

    req([due(i, 2) for i in range(5)] == [False, True, False, True, False]
        and [due(i, 0) for i in range(3)] == [False] * 3
        and [due(i, None) for i in range(3)] == [False] * 3
        and [due(i, 1) for i in range(3)] == [True] * 3
        and [due(i, 3) for i in range(5)] == [False, True, False, False, True]
        and [due(i, 2, True) for i in range(4)] == [True, True, False, True], "eval_due")
    return (f"{len(bad)} bad blocks refused; v12_36 / v12_46 / v12_57 accepted; hard suite 11 runs "
            "x 2 == SUITE; eval_due every 0 / 1 / 2 / 3 + keep_best i0")


def k2(ctx) -> str:
    if ctx.pre_dir is None:
        return "skipped (no --pre-dir)"
    cases = [("v12_36", range(10)), ("v12_46", range(10)), ("C1_transport", (0, 1))]
    n = 0
    for name, iters in cases:
        for i in iters:
            argv = ["--root", str(CMP / name), "--config", str(CFG / f"{name}.yaml"), "--iter",
                    str(i), "--step", "all", "--dry-run"]
            a = dry_main(load_loop(ctx.pre_dir, f"p_{name}{i}"), argv)
            b = dry_main(load_loop(None, f"t_{name}{i}"), argv)
            req(a == b and len(a) > 200, f"{name} iter {i}: dry-run differs")
            n += 1
    return f"{n}/{n} dry-runs identical (v12_36 0-9, v12_46 0-9, C1_transport 0/1)"


def k3(ctx) -> str:
    import run_hard_suite as rhs

    c = cfg57()
    sm = c["seed_model"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "v12_57"
        c2 = {**c, "refit": {**c["refit"], "extra_args": ["--heldout-frac", "0.1"]}}
        (Path(td) / "c2.yaml").write_text(yaml.safe_dump(c2, sort_keys=False))
        outs = [dry_main(load_loop(None, f"k3_{i}"), ["--root", str(root), "--config", str(
            CFG / "v12_57.yaml"), "--iter", str(i), "--step", "all", "--dry-run"]) for i in (0, 1, 2)]
        ex = dry_main(load_loop(None, "k3_x"), ["--root", str(root), "--config", str(
            Path(td) / "c2.yaml"), "--iter", "1", "--step", "refit", "--dry-run"])
        req(not root.exists(), "dry-run wrote the root")
    for i, o in enumerate(outs):
        st = [x[1:-1] for x in o.splitlines() if x.startswith("[")]
        req(st == ["seed" if i == 0 else "refit", "trust", "export", "eval", "deploy", "collect",
                   "label", "buffer", "report", "register"], f"i{i} steps {st}")
        req("udpm://239.255.76.200:7800" in o and "7801" in o, f"i{i} ports")
        req(not any(x in o for x in ("--mppi-demo-actions", "prior_")), f"i{i} prior")
        lcs = [x for x in o.splitlines() if "encode_latents_cache.py" in x
               or "export_learned_lcs_deploy.py --checkpoint" in x and "behaviour_gate" not in x]
        req(lcs and all("LCS_CACHE_REPRO_TOL_REL=1e-5" in x for x in lcs), f"i{i} JEPA env")
    req("seed_model: theta_0 = " + sm["checkpoint"] in outs[0]
        and "refit_from_buffer.py" not in outs[0] and "build_cloud_graph" not in outs[0]
        and f"--checkpoint {sm['checkpoint']}" in outs[0], "i0 seed")
    for i in (1, 2):
        o = outs[i]
        g = next(x for x in o.splitlines() if "build_cloud_graph.py" in x).split()
        t = next(x for x in o.splitlines() if "refit_from_buffer.py" in x).split()
        npz = str(root / f"iter{i}/graph/graph.npz")
        req(g[g.index("--buffer") + 1] == str(root / f"iter{i - 1}/buffer.json")
            and g.count("--buffer") == 1 and g[g.index("--out") + 1] == npz
            and g[g.index("--stride") + 1] == "3" and g[g.index("--device") + 1] == "cuda"
            and "--validate-gt" not in g, f"i{i} graph cmd")
        req(t[t.index("--graph-file") + 1] == npz and t[t.index("--graph-weight") + 1] == "1"
            and t[t.index("--init-checkpoint") + 1] == sm["checkpoint"]
            and t[t.index("--mode") + 1] == "lcs_only" and "--jepa" not in t, f"i{i} refit")
        req("graph enters metrics.graph only" in o, f"i{i} lcs_only note")
    ev = [x.split() for x in outs[1].splitlines() if "eval_learned_mpc.py" in x
          and "7801" in x]
    want = [(s, v, seed) for _, s, v, seed in rhs.SUITE]
    got = [(x[x.index("--start-states") + 1],
            x[x.index("--variants") + 1] if "--variants" in x else "",
            next((int(y[5:]) for y in x if y.startswith("seed=")), None)) for x in ev]
    req(got == want and all(x[x.index("--repeats") + 1] == "2" for x in ev)
        and all("V11_iter1_eval_hs_" in " ".join(x) for x in ev)
        and all(str(root / "iter1/cand/latent_waypoints.npz").endswith(
            x[x.index("--mppi-waypoints") + 1]) for x in ev), "i1 eval runs")
    req("keep_best: candidate theta_1 -> " in outs[1] and "cand/deploy_v22?" in outs[1]
        and "-> deploy ?: deploy theta_1 iff its eval_engaged >= deployed_0's + 0" in outs[1],
        "i1 candidate / deploy")
    req("no candidate export" in outs[2] and "eval skipped (iteration 2" in outs[2]
        and "-> deploy ?: keep (candidate not evaluated)" in outs[2]
        and "eval_learned_mpc.py" in outs[2], "i2 keep")
    req(ex.count("--recon-max-mm 6.0 --heldout-frac 0.1 --graph-file") == 1, "extra_args")
    return ("iterations 0-2 resolve: eval + deploy before collect, i0 seed_model, i1/i2 graph of "
            "B_i -> --graph-file / --graph-weight 1, JEPA env, 11 x 2 hard-suite eval runs, i2 "
            "keep; extra_args appended")


def k4(ctx) -> str:
    rl = ctx.tree

    def m(i, s, n=22, mid=None):
        return {"id": mid or f"V22{'abcdefghij'[i]}", "iter": i, "score": s, "n": n}

    rows = [(None, m(0, 11), "a", "initial"), (None, m(0, None), "a", "initial"),
            (m(0, 11), m(1, None), "a", "keep (candidate not evaluated)"),
            (m(0, 11), m(1, 12), "b", "deploy (gain +1 >= 0)"),
            (m(0, 11), m(1, 11), "b", "deploy (gain +0 >= 0)"),
            (m(0, 11), m(1, 10), "a", "keep (gain -1 < 0)"),
            (m(0, 11, mid="V22a"), m(1, 13, mid="V22a"), "a", "keep (same model)"),
            (m(0, None), m(1, 3), "b", "deploy (deployed model not evaluated)")]
    for inc, cand, want, why in rows:
        pick, w = rl.keep_best_pick(inc, cand, 0)
        req(pick["id"][-1] == want and w.startswith(why), f"{inc} {cand} -> {pick} {w}")
    for gain, mg, want in ((0, 1, "a"), (1, 1, "b"), (1, 2, "a"), (2, 2, "b")):
        pick, _ = rl.keep_best_pick(m(0, 11), m(1, 11 + gain), mg)
        req(pick["id"][-1] == want, f"min_gain {mg} gain {gain}")
    seq, inc = [11, 12, 9, 12, 15, 14], None  # i = 0..5, all evaluated
    dep = []
    for i, s in enumerate(seq):
        inc, _ = rl.keep_best_pick(inc, m(i, s), 0)
        dep.append(inc["iter"])
    req(dep == [0, 1, 1, 3, 4, 4], f"sequence {dep}")
    req(rl.policy_score({"summary": {"engaged": 7, "strict": 2}}, "eval_engaged") == 7
        and rl.policy_score({"summary": {"engaged": 7, "strict": 2}}, "eval_strict") == 2, "score")
    return f"{len(rows) + 4} cases + a 6-iteration sequence (deployed iters {dep})"


def stub_root(rl, root: Path, inc_score: int | None, cand_score: int | None):
    """iter0 (deployed V22a, score inc_score) + iter1 (theta_1 V22b, eval cand_score)."""
    ck = cfg57()["seed_model"]
    m0 = {"id": "V22a", "letter": "a", "iter": 0, "new": True, "checkpoint": ck["checkpoint"],
          "refit_report": ck["refit_report"]}
    m1 = {**m0, "id": "V22b", "letter": "b", "iter": 1, "checkpoint": "/x/theta1.pt"}
    rl.jdump(root / "iter0/model.json", m0)
    rl.jdump(root / "iter0/deploy_policy.json", {"deployed": {"id": "V22a", "iter": 0,
                                                              "score": inc_score, "n": 22,
                                                              "model": m0}})
    rl.jdump(root / "iter1/model.json", m1)
    if cand_score is not None:
        rl.jdump(root / "iter1/eval.json", {"model": "V22b", "summary": {
            "n": 22, "engaged": cand_score, "strict": 0}})


def k5(ctx) -> str:
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for name, inc, cand in (("win", 11, 12), ("loss", 11, 10), ("tie", 11, 11)):
            root = Path(td) / name
            stub_root(ctx.tree, root, inc, cand)
            o = dry_main(load_loop(None, f"k5_{name}"), [
                "--root", str(root), "--config", str(CFG / "v12_57.yaml"), "--iter", "1",
                "--step", "deploy", "--dry-run"])
            out[name] = o
            o2 = dry_main(load_loop(None, f"k5_{name}2"), [
                "--root", str(root), "--config", str(CFG / "v12_57.yaml"), "--iter", "2",
                "--step", "deploy", "--dry-run"])
            req("theta_2 not evaluated" in o2 and "-> deploy ?: keep (candidate not evaluated)"
                in o2, f"{name} i2")
    req("-> deploy V22b: deploy (gain +1 >= 0)" in out["win"]
        and "--checkpoint /x/theta1.pt" in out["win"] and "deploy_v22b" in out["win"]
        and "V22a (iter 0, 11/22); theta_1 12/22" in out["win"], "win")
    req("-> deploy V22a: keep (gain -1 < 0)" in out["loss"]
        and "--checkpoint " + cfg57()["seed_model"]["checkpoint"] in out["loss"]
        and "iter1/deploy_v22a" in out["loss"], "loss")
    req("-> deploy V22b: deploy (gain +0 >= 0)" in out["tie"], "tie")
    return "i1 win -> V22b exported, loss -> V22a re-exported with box_1, tie -> V22b; i2 keep"


def k6(ctx) -> str:
    rl = load_loop(None, "k6")
    _e = rl.export_cmds

    def cpu(*a, **k):
        c, e, d, ch = _e(*a, **k)
        return c + ["--device", "cpu"], e, d, ch  # the cache encoder defaults to cuda

    rl.export_cmds = cpu
    rl.policy_screen = lambda *a, **k: {"n": 0, "stub": True}
    sm = cfg57()["seed_model"]
    keep = ctx.keep_dir
    with tempfile.TemporaryDirectory() as td:
        td = Path(keep) if keep else Path(td)
        cfg = cfg57()
        cfg.update(lcs_out=str(td / "lcs"), seed_buffer_merged=str(td / "root/seed_buffer_merged.json"),
                   graph={**cfg["graph"], "device": "cpu",  # CPU: fewer candidates / samples
                          "extra": ["--k", "6", "--detour-samples", "500"]},
                   refit={**cfg["refit"], "epochs": 1})
        (td / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        t0 = time.perf_counter()
        c = rl.Ctx(ns(td / "root", 0, td / "cfg.yaml", False))
        with contextlib.redirect_stdout(io.StringIO()):
            for st in ("seed", "trust", "export"):
                rl.do_step(c, st, False)
        it = c.it()
        req((td / "root/seed_buffer_merged.json").read_bytes()
            == (CMP / "v12_35/seed_buffer_merged.json").read_bytes(), "merged seed buffer bytes")
        m0 = rl.jload(it / "model.json")
        req(m0["id"] == "V22a" and m0["checkpoint"] == sm["checkpoint"]
            and m0["seed_model_same_buffer_rows"], f"model0 {m0}")
        cd = rl.jload(it / "cand/deploy.json")
        req(cd["model"] == "V22a" and cd["env"] == rl.JEPA_ENV and "cand" in cd["deploy"]
            and not (it / "deploy.json").exists(), "cand export")

        def stub_eval(i, engaged):
            cdir = c.it(i) / "cand"
            ev = {"iter": i, "model": rl.jload(cdir / "model.json")["id"], "rows": [],
                  "summary": {"n": 22, "engaged": engaged, "strict": 0, "cloud_success": 0,
                              "rmse_mean_mm": float("nan"), "n_unsafe": 0, "aborts": 0,
                              "solve_p95_ms": None, "max_rot_deg": [0, 0]},
                  "dir": rl.rel(cdir / "eval"), "label": "stub", "box_id": f"ub-tr{i}"}
            rl.jdump(cdir / "eval.json", ev)
            rl.jdump(c.it(i) / "eval.json", {**ev, "candidate": True})
            rl.set_step(rl.Ctx(ns(td / "root", i, td / "cfg.yaml", False)), i, "eval",
                        status="done", outputs=[str(c.it(i) / "eval.json")])

        stub_eval(0, 11)
        with contextlib.redirect_stdout(io.StringIO()):
            rl.do_step(c, "deploy", False)
        d0 = rl.jload(it / "deploy.json")
        dp0 = rl.jload(it / "deploy_policy.json")
        req(d0["model"] == "V22a" and dp0["deployed"]["score"] == 11
            and dp0["decision"] == "initial (theta_0)", f"deploy0 {dp0['decision']}")
        a = np.load(d0["deploy"], allow_pickle=True)
        b = np.load(rl.jload(V21C / "deploy.json")["deploy"], allow_pickle=True)
        cc = np.load(cd["deploy"], allow_pickle=True)
        req(set(a.files) == set(b.files) == set(cc.files), "deploy keys")
        fl = [k for k in a.files if a[k].dtype.kind == "f" and k != "exported_at"]
        dmax = max(float(np.max(np.abs(a[k] - b[k]))) for k in fl)
        dcand = max(float(np.max(np.abs(a[k] - cc[k]))) for k in fl)
        req(dmax < 1e-5 and dcand < 1e-5 and all(np.array_equal(a[k], b[k]) for k in a.files
                                                if a[k].dtype.kind not in "fU"
                                                and k != "exported_at"),
            f"deploy vs V21c sim arm {dmax}, vs cand {dcand}")
        # B_1 stub: the 3 demos + 17 MILES rows of the seed buffer (anchor weights as written)
        seed = rl.jload(td / "root/seed_buffer_merged.json")
        rows = [r for r in seed["rollouts"] if r["kind"] == "demo"][:3] + \
            [r for r in seed["rollouts"] if r["kind"] != "demo"][:17]
        rl.jdump(it / "buffer.json", {**seed, "iter": 0, "n": len(rows), "rollouts": rows,
                                      "n_success": sum(r["label_human"] for r in rows)})
        t_i0 = time.perf_counter() - t0
        c1 = rl.Ctx(ns(td / "root", 1, td / "cfg.yaml", False))
        t1 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            for st in ("refit", "trust", "export"):
                rl.do_step(c1, st, False)
        t_refit = time.perf_counter() - t1
        rf = rl.jload(c1.it() / "refit.json")
        m1 = rl.jload(c1.it() / "model.json")
        g = rf["graph"]
        gj = rl.jload(c1.it() / "graph/graph.json")
        import torch

        ca = torch.load(m1["checkpoint"], map_location="cpu", weights_only=False)["args"]
        req(rf["train_objective"] == "jepa" and m1["id"] == "V22b" and m1["from_seed"] == "V22a"
            and g["n_episodes"] == 20 and gj["args"]["buffer"] == [str(it / "buffer.json")]
            and gj["args"]["k"] == 6 and gj["args"]["detour_samples"] == 500
            and g["n_nodes"] == gj["stats"]["n_nodes"] > 0 and g["refit_metrics"]["stress"]
            is not None and ca["graph_file"] == str(c1.it() / "graph/graph.npz")
            and ca["jepa"] is True and ca["lcs_symmetric_f"] is True
            and ca["graph_sha256"] == rl.sha(c1.it() / "graph/graph.npz"), f"refit1 {g}")
        req(rl.jload(c1.it() / "cand/deploy.json")["model"] == "V22b", "cand1")
        res = {}
        for name, score, want in (("loss", 10, "V22a"), ("win", 12, "V22b")):
            stub_eval(1, score)
            with contextlib.redirect_stdout(io.StringIO()):
                c1.a.force = "deploy" if name == "win" else None
                rl.do_step(c1, "deploy", False)
            c1.a.force = None
            d1 = rl.jload(c1.it() / "deploy.json")
            dp1 = rl.jload(c1.it() / "deploy_policy.json")
            req(d1["model"] == want and dp1["deployed"]["id"] == want
                and Path(d1["deploy"]).parent.name == f"deploy_v22{want[-1]}"
                and rl.jload(d1["box_json"]) == rl.jload(c1.it() / "box_1.json"), f"{name} {dp1}")
            res[name] = dp1["decision"]
        with contextlib.redirect_stdout(io.StringIO()):
            rl.do_step(c1, "report", False)
        md = (td / "root/learning_curve.md").read_text()
        req("Keep-best deployment" in md and "| 1 | V22b | 12/22 | V22a (11/22) | **V22b** (1, 12/22)"
            in md and "Cloud graph per refit" in md and "| 1 | jepa | " in md, "learning curve")
    return (f"i0 deploy == V21c arm (max {dmax:.1e}; cand == main {dcand:.1e}), JEPA env; i1 graph "
            f"{g['n_nodes']} nodes / {g['n_episodes']} episodes, rejected {g['rejected_frac']:.2%},"
            f" components {g['n_components']}, refit jepa + graph keys; deploy loss -> "
            f"{res['loss']}, win -> {res['win']}; learning curve sections (i0 {t_i0:.0f} s, i1 "
            f"refit+export {t_refit:.0f} s)")


CHECKS = [("K1 config", k1), ("K2 dry-run identity", k2), ("K3 v12_57 dry-run", k3),
          ("K4 keep_best_pick", k4), ("K5 stubbed-eval dry-run", k5),
          ("K6 scratch iterations 0-1", k6)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="pre-change tree with scripts/lcs/realfirst_loop.py (repo layout)")
    ap.add_argument("--only", default=None, help="comma list of K1..K6")
    ap.add_argument("--keep-dir", type=Path, default=None, help="K6 scratch dir (kept)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit('run with CUDA_VISIBLE_DEVICES=""')
    ctx = types.SimpleNamespace(tree=load_loop(None, "tree"), pre_dir=a.pre_dir,
                                keep_dir=a.keep_dir)
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
