#!/usr/bin/env python3
"""V12-40 checks of the C3+ controller path (CPU, offline: no sim, no LCM, no C++ binary).

K1 the python mirror of the C++ params load: known-good archived params (V8a, V10a, v2_long30,
   seatonly30, a 09-24 demo_traj yaml) pass; the raw belt_seg export yaml (no
   n_engaged_episodes), an unknown key and a stage-list length mismatch fail.
K2 the generated V12h / V16b files (c3plus/<model>/manifest.json): pass the mirror; the LCS
   matrices equal the export's yaml and deploy.npz (the MPPI's model), u bounds == box_i, stage
   goals == the waypoints' z_wp rows, params top level == V10a's, costs / floors as intended,
   track: demo_traj == z_traj / EE poses from the offset frame; files == manifest sha256.
K3 loop default identity: realfirst_loop.py --dry-run pre vs tree for C1_transport and v12_36
   iterations 0 / 1 (byte-identical stdout).
K4 loop controller c3plus: dry-run harness commands use --params (no MPPI flags) and E3 at
   1e-2; c3plus_params() on a scratch iteration dir writes passing files (+ box_z); refusals.
K5 hard suite: write with MPPI arms only pre == tree (run.sh, suite.json); a c3plus arm adds
   run_c3 with --params and passing worktree params.
K6 E3 at 1e-2 (check_latent_encoder logs) PASS for V12h and V16b; exact-LCP table present.

    uv run --frozen python scripts/checks/check_c3plus.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import c3plus_params as c3p

C3 = REPO / "data/lcs/realfirst/cmp/c3plus"
A = "systems/parameters/learned_archive"
P = "round_belt_controller_params_learned_eval"
KNOWN_GOOD = [f"{A}/2026-10-01/v8a/{P}_v8a_flat_engaged.yaml",
              f"{A}/2026-10-01/v10/{P}_v10a_flat_engaged.yaml",
              f"{A}/2026-10-05/v2_long30/{P}_flat_engaged_long30.yaml",
              f"{A}/2026-10-01/v10/{P}_v10a_flat_engaged_seatonly30.yaml"]
TRAJ_GOOD = (f"{A}/2026-09-24/learned_tuning/round_belt_controller_params_learned_eval_ubnd_"
             "wr30_s0skip_tol2_eez_ur13_traj_progress_admm2.yaml")
CELLS = {"C1_transport": "data/lcs/realfirst/cmp/C1_transport",
         "v12_36": "data/lcs/realfirst/cmp/v12_36"}
RUNNER = '''import sys
src, target, *argv = sys.argv[1:]
sys.argv = [target, *argv]
sys.path.insert(0, str(__import__("pathlib").Path(target).parent))
exec(compile(open(src).read(), target, "exec"), {"__name__": "__main__", "__file__": target})
'''
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def req(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run_src(ctx, src: Path, target: str, argv: list[str]) -> str:
    r = subprocess.run([sys.executable, str(ctx.runner), str(src), str(REPO / target), *argv],
                       cwd=REPO, capture_output=True, text=True, check=False,
                       env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "",
                            "OMP_NUM_THREADS": "2"})
    req(r.returncode == 0, f"{src.name} {argv[:3]} rc {r.returncode}: {r.stderr[-400:]}")
    return r.stdout + r.stderr


def tmp_params(ctx, doc: dict, name: str) -> str:
    """A throwaway params yaml in the worktree check folder (worktree-relative path)."""
    rel = f"{c3p.ARCHIVE}/check/{name}.yaml"
    (c3p.WORKTREE / rel).parent.mkdir(parents=True, exist_ok=True)
    (c3p.WORKTREE / rel).write_text(yaml.safe_dump(doc, sort_keys=False))
    return rel


@check("K1 C++ load mirror: known good pass, bad fail")
def k1(ctx) -> str:
    for p in KNOWN_GOOD:
        r = c3p.check_params(p)
        req(r["pass"], f"{p}: {r['errors']}")
    r = c3p.check_params(TRAJ_GOOD, base=TRAJ_GOOD)
    req(r["pass"] and r.get("demo_traj_frames"), f"{TRAJ_GOOD}: {r['errors']}")
    base = yaml.safe_load((c3p.WORKTREE / KNOWN_GOOD[1]).read_text())
    raw = Path(json.loads((REPO / "data/lcs/realfirst/cmp/v12_36/iter1/deploy.json")
                          .read_text())["deploy_dir"]) / "learned_lcs.yaml"
    lrel = f"{c3p.ARCHIVE}/check/raw_export_lcs.yaml"
    (c3p.WORKTREE / lrel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, c3p.WORKTREE / lrel)
    bad = []
    d = yaml.safe_load(yaml.safe_dump(base))
    d["learned_mpc"]["lcs_file"] = lrel
    r = c3p.check_params(tmp_params(ctx, d, "raw_export"), base=KNOWN_GOOD[1])
    req(not r["pass"] and any("n_engaged_episodes" in e for e in r["errors"]),
        f"raw export not caught: {r['errors']}")
    bad.append("raw export: " + r["errors"][0])
    d = yaml.safe_load(yaml.safe_dump(base))
    d["learned_mpc"]["w_typo"] = 1.0
    r = c3p.check_params(tmp_params(ctx, d, "unknown_key"), base=KNOWN_GOOD[1])
    req(not r["pass"] and any("unknown key" in e for e in r["errors"]), "unknown key passed")
    d = yaml.safe_load(yaml.safe_dump(base))
    d["learned_mpc"]["stage_goal_tols"] = [1e-6]
    r = c3p.check_params(tmp_params(ctx, d, "tols_len"), base=KNOWN_GOOD[1])
    req(not r["pass"] and any("stage_goal_tols" in e for e in r["errors"]), "tols length passed")
    return f"{len(KNOWN_GOOD) + 1} known-good pass (n_z 50 = the C++ log); 3 bad fail ({bad[0]})"


def lcs_of(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@check("K2 generated V12h / V16b files")
def k2(ctx) -> str:
    n = 0
    for model in ("v12h", "v16b"):
        man = json.loads((C3 / model / "manifest.json").read_text())
        dj = json.loads((REPO / man["deploy_json"]).read_text())
        exp = lcs_of(Path(dj["deploy_dir"]) / "learned_lcs.yaml")
        box = json.loads(Path(dj["box_json"]).read_text())
        with np.load(dj["deploy"], allow_pickle=True) as z:
            npz = {k: z[k] for k in z.files}
        with np.load(REPO / man["variants"][0]["waypoints"], allow_pickle=False) as f:
            wp = {k: f[k] for k in f.files}
        k_latch = list(wp["frames"]).index(int(wp["latch"]))
        base = yaml.safe_load((c3p.WORKTREE / c3p.BASE_PARAMS).read_text())
        mppi = yaml.safe_load((REPO / c3p.BASE_MPPI).read_text())
        for m in man["variants"]:
            v = m["variant"]
            for k, f in m["files"].items():
                req(c3p.sha(c3p.WORKTREE / f) == m["sha256"][k], f"{f} changed since write")
            r = c3p.check_params(m["files"]["params"])
            req(r["pass"], f"{model} {v}: {r['errors']}")
            lc = lcs_of(c3p.WORKTREE / m["files"]["lcs_yaml"])
            for k in ("n_x", "n_u", "n_lam", "N", "dt", "A", "B", "D", "d", "E", "F", "H", "c",
                      "z_std", "z_start_example", "goal_tol_whitened", "checkpoint_sha256"):
                req(lc[k] == exp[k], f"{model} {v}: {k} != export yaml")
            for k in ("A", "B", "D", "d", "E", "F", "H", "c", "z_std"):
                req(np.abs(np.asarray(lc[k]) - npz[k]).max() <= 1e-6 * max(
                    1.0, np.abs(npz[k]).max()), f"{model} {v}: {k} != deploy.npz (MPPI model)")
            req(lc["u_lb"] == box["u_lb"] and lc["u_ub"] == box["u_ub"], f"{model} {v}: u bounds")
            z2 = wp["z_wp"][-1].tolist() if v != "pp2" else exp["z_goal_stage2"]
            req(lc["z_goal"] == z2, f"{model} {v}: z_goal")
            if v in ("wp2", "track"):
                req(lc["z_goal_stage1"] == wp["z_wp"][k_latch].tolist()
                    and lc["z_goal_stage2"] == z2, f"{model} {v}: stage goals != waypoints")
            if v == "pp2":
                req(lc["z_goal_stage1"] == exp["z_goal_stage1"], f"{model} pp2 stage 1")
            if v == "seat":
                req("z_goal_stage1" not in lc and r["n_stages_effective"] == 1, "seat stages")
            pd = yaml.safe_load((c3p.WORKTREE / m["files"]["params"]).read_text())
            req({k: x for k, x in pd.items() if k != "learned_mpc"}
                == {k: x for k, x in base.items() if k != "learned_mpc"}, "top level != V10a")
            lm = pd["learned_mpc"]
            ee = lm["ee_constraints"]
            req(lm["w_q"] == 1.0 and lm["w_r"] == 0.3 and lm["admm_iter"] == 2
                and lm["projection_type"] == "C3+" and ee["w_p"] == 0.0
                and ee["z_min_franka"] == mppi["z_min_franka"]
                and ee["z_min_ur"] == mppi["z_min_ur"] and lm["honor_penalize_input_change"]
                and not lm["penalize_input_change"], f"{model} {v}: costs / floors")
            if v == "wp2":
                req(lm["stage_goal_tols"] == [float(wp["tol"][k_latch]), 1e-6]
                    and lm["stage_max_durations_s"] == [float(sum(wp["max_duration_s"][
                        :k_latch + 1])), 30.0], f"{model} wp2 tols / durations")
            if v == "track":
                tr = yaml.safe_load((c3p.WORKTREE / m["files"]["demo_traj"]).read_text())
                with np.load(str(wp["demo_episode"]), allow_pickle=True) as e:
                    ee_f = np.asarray(e["sim_ee_franka"], float)[c3p.DEMO_OFFSET:]
                req(np.array_equal(np.asarray(tr["z"]), wp["z_traj"][c3p.DEMO_OFFSET:])
                    and np.array_equal(np.asarray(tr["ee_pose_franka"]), ee_f)
                    and lm["demo_traj"]["w_p"] == 0.0, f"{model} track demo_traj")
            n += 1
    return f"{n} files sets pass; matrices == export yaml == deploy.npz; goals == waypoints"


@check("K3 loop default: dry-run pre == tree")
def k3(ctx) -> str:
    out = []
    for cell, root in CELLS.items():
        for i in (0, 1):
            argv = ["--root", root, "--config", f"data/lcs/realfirst/cmp/configs/{cell}.yaml",
                    "--iter", str(i), "--step", "all", "--dry-run"]
            a = run_src(ctx, ctx.pre / "realfirst_loop.py", "scripts/lcs/realfirst_loop.py", argv)
            b = run_src(ctx, REPO / "scripts/lcs/realfirst_loop.py",
                        "scripts/lcs/realfirst_loop.py", argv)
            req(a == b, f"{cell} iter {i}: dry-run differs")
            out.append(f"{cell}/{i} {len(b.splitlines())} lines")
    return "identical: " + ", ".join(out)


@check("K4 loop controller c3plus")
def k4(ctx) -> str:
    import realfirst_loop as rl

    cfg = yaml.safe_load((REPO / "data/lcs/realfirst/cmp/configs/v12_36.yaml").read_text())
    cfg.update(yaml.safe_load((C3 / "configs/v12_36_c3plus.yaml").read_text()))
    cpath = ctx.tmp / "v12_36_c3plus.yaml"
    cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    txt = run_src(ctx, REPO / "scripts/lcs/realfirst_loop.py", "scripts/lcs/realfirst_loop.py",
                  ["--root", CELLS["v12_36"], "--config", str(cpath), "--iter", "1", "--step",
                   "all", "--dry-run"])
    h = [ln for ln in txt.splitlines() if "eval_learned_mpc.py" in ln]
    req(len(h) == 9 and all("--params systems/parameters/learned_archive/2026-10-07/c3plus/"
                            "loop/v12_36/iter1/" in ln and "--mppi" not in ln
                            and "--controller" not in ln for ln in h), "c3plus harness cmds")
    req("--exact-tol 0.01" in txt, "E3 bound for c3plus")
    # c3plus_params on a scratch iteration (writes worktree files under c3plus/check)
    root = ctx.tmp / "loop_root"
    src = REPO / CELLS["v12_36"]
    (root / "iter1").mkdir(parents=True)
    (root / "iter0").mkdir()
    for f in ("deploy.json", "latent_waypoints.npz"):
        shutil.copy2(src / "iter1" / f, root / "iter1" / f)
    shutil.copy2(src / "iter0" / "buffer.json", root / "iter0" / "buffer.json")
    for box_z in (False, True):
        c = {**cfg, "c3plus": {**cfg["c3plus"], "box_z": box_z,
                               "archive": f"{c3p.ARCHIVE}/check/loop"}}
        (ctx.tmp / "c.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        a = argparse.Namespace(smoke=False, root=str(root), iter=1, config=str(ctx.tmp / "c.yaml"),
                               dry_run=False, collect_port=None, eval_port=None)
        x = rl.Ctx(a)
        p = rl.c3plus_params(x, "collect")
        r = c3p.check_params(p)
        js = json.loads((root / "iter1" / "c3plus_collect.json").read_text())
        req(r["pass"] and js["check"]["pass"] and js["support_box_z"] == box_z, f"box_z {box_z}")
        ee = yaml.safe_load((c3p.WORKTREE / p).read_text())["learned_mpc"]["ee_constraints"]
        if box_z:
            sb = rl.support_box(x, True)
            req(ee["z_max_franka"] == sb["hi"][2] and ee["z_max_ur"] == sb["hi"][5],
                "box_z ceilings")
        p2 = rl.c3plus_params(x, "eval_urtip6", {"z_min_franka": 0.0376, "z_min_ur": 0.0321})
        ee2 = yaml.safe_load((c3p.WORKTREE / p2).read_text())["learned_mpc"]["ee_constraints"]
        req(ee2["z_min_ur"] == max(0.0321, sb["lo"][5]) if box_z else ee2["z_min_ur"] == 0.0321,
            "eval floors")
        req(rl.c3plus_params(x, "c", stages="last", dry=True).endswith("_seat.yaml"), "last")
    for bad in ({"controller": "c3"}, {"controller": "c3plus", "demo_prior": {"alpha": 0.5,
                                                                            "weight": 0.1}},
                {"controller": "c3plus", "waypoint_bank": {"knn": 1}},
                {"controller": "c3plus", "c3plus": {"variant": "seat"}},
                {"c3plus": {}}):
        try:
            rl.check_cfg(rl.merge(rl.DEFAULTS, {**cfg, "controller": "mppi", "c3plus": None,
                                                **bad}))
        except SystemExit:
            continue
        raise AssertionError(f"not refused: {bad}")
    return ("9 harness cmds with --params, E3 1e-2; c3plus_params writes passing files "
            "(box_z ceilings = support box hi z; eval floors); 5 refusals")


@check("K5 hard suite: MPPI arms pre == tree, C3+ arm")
def k5(ctx) -> str:
    d16 = json.loads((REPO / CELLS["v12_36"] / "iter1/deploy.json").read_text())["deploy"]
    wp = f"{CELLS['v12_36']}/iter1/latent_waypoints.npz"
    yml = f"{CELLS['v12_36']}/iter1/eval/mppi_eval_fe_nominal.yaml"
    outs = {}
    for v, src in (("pre", ctx.pre / "run_hard_suite.py"),
                   ("tree", REPO / "scripts/lcs/run_hard_suite.py")):
        o = ctx.tmp / f"hs_{v}" / "s"
        run_src(ctx, src, "scripts/lcs/run_hard_suite.py",
                ["write", "--out", str(o), "--port", "7850", "--arm", "A", d16, "auto:20",
                 "data/lcs/realfirst/cmp/sbox_sim/mppi_box.yaml", "--arm", "B", d16, wp, yml])
        outs[v] = [(o / f).read_text().replace(f"hs_{v}", "hs_X")
                   for f in ("run.sh", "suite.json")]
    req(outs["pre"] == outs["tree"], "MPPI-only run.sh / suite.json differ")
    o = ctx.tmp / "hs_c3" / "check_hs"
    run_src(ctx, REPO / "scripts/lcs/run_hard_suite.py", "scripts/lcs/run_hard_suite.py",
            ["write", "--out", str(o), "--port", "7850", "--arm", "A", d16, wp, yml,
             "--arm", "C", d16, wp, "c3plus:wp2"])
    sh = (o / "run.sh").read_text()
    c3 = [ln for ln in sh.splitlines() if ln.startswith("run_c3 ")]
    req(len(c3) == 11 and "--params $7" in sh and all(
        "c3plus/hard_suite/check_hs/C/" in ln for ln in c3), "run_c3 lines")
    arm = json.loads((o / "suite.json").read_text())["arms"][1]
    req(arm["controller"] == "c3plus" and c3p.check_params(arm["params"])["pass"], "C3+ arm")
    return "MPPI-only identical; c3plus:wp2 arm -> 11 run_c3 lines, params pass"


@check("K6 E3 at 1e-2 and exact LCP")
def k6(ctx) -> str:
    out = []
    for m in ("v12h", "v16b"):
        t = (C3 / f"checks/check_latent_encoder_{m}_tol1e-2.log").read_text()
        req("ALL LATENT ENCODER CHECKS PASSED" in t, f"{m}: E3 log")
        out.append(next(ln for ln in t.splitlines() if ln.startswith("[PASS] E3"))
                   .split("vs exact ")[1].split(";")[0])
    lcp = json.loads((C3 / "lcp/lcp.json").read_text())
    for k, r in lcp.items():
        req(r["unique"] == r["states"] and r["pgd25"]["max"] < 1e-2, f"{k}: exact LCP")
    mx = ", ".join(f"{k} {r['pgd25']['max']:.1e}" for k, r in lcp.items())
    return (f"E3 loose vs exact V12h {out[0]}, V16b {out[1]} (<= 1e-2); task-state PGD-25 vs "
            f"exact max {mx}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=C3 / "checks/pre",
                    help="realfirst_loop.py / run_hard_suite.py before V12-40")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="check_c3plus_"))
    ctx = argparse.Namespace(pre=a.pre_dir, tmp=tmp, runner=tmp / "runner.py")
    ctx.runner.write_text(RUNNER)
    rc = 0
    try:
        for name, fn in CHECKS:
            try:
                print(f"[PASS] {name}: {fn(ctx)}", flush=True)
            except Exception as exc:  # noqa: BLE001
                rc = 1
                print(f"[FAIL] {name}: {exc}", flush=True)
                traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(c3p.WORKTREE / c3p.ARCHIVE / "check", ignore_errors=True)
        shutil.rmtree(c3p.WORKTREE / c3p.ARCHIVE / "hard_suite" / "check_hs", ignore_errors=True)
    print("ALL C3PLUS CHECKS PASSED" if rc == 0 else "C3PLUS CHECKS FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
