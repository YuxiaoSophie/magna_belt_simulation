#!/usr/bin/env python3
"""V8 on-policy loop driver (plan handoffs/PLAN-20261001-v8-onpolicy.md §2), one step at a time.

Steps of iteration i (model M_i = iter<i>/model.json; M_0 = V4a + ub-ins):
    collect   sample the manifest (unless present), pool check --all, one harness run per cell
    eval      seat6 + yaw12 of M_i (skipped when M_i's rows exist: V4a 09-29, or a kept model)
    encode    latent cache of the new rollouts (+ per-rollout splits), buffer.json (FIFO 50)
    trust     box_<i+1> = clip(u_mean +- max(eta sigma, floor half_ins), ub-ins), buffer MPC frames
    finetune  candidates every steps/6; pick the best held-out h7 among the one-step-gate passes;
              h7 gate (1.25x V4a); ms7 loss from the start; 2nd rejection in a row -> lr 1e-4; M_<i+1>
    export    --from-deploy deploy_t1 --u-bounds-json box_<i+1>.json + check_latent_encoder
    assets    per-target goal deploys + worktree yamls (V4a files with the LCS lines swapped)
    c3check   learned_lcs_c3_check on every params yaml of M_<i+1>
    tables    learning_curve.{json,md,png} row i + the stop-rule verdict line (never acted on)
    register  M_<i+1> model rows + M_i eval rows in docs/models.{md,yaml} (idempotent)
    all       the above; --concurrency 1 (default): eval starts after collect, beside the GPU
              steps (2 sims at once cost +28-32 % solve p95 in the smoke); 2: eval beside collect
    control   once: V4a LCS + box_1, seat6 (plan Q6), registered as a V4a eval row
A done step is skipped; --force STEP moves its outputs aside (.prev<n>) and redoes it.

Run (this repo, uv run --frozen python scripts/lcs/onpolicy_loop.py ...):
    --iter 0 --step all --dry-run
    --iter 0 --step all                       # real run, URLs 7752 / 7750 (/ 7751)
    --smoke --iter 0 --step all               # 2 attempts, 300 steps, 7749, root .../v8/smoke_loop
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
LCS = HOME / "git/lcs_learning"
WT = HOME / "git/magna-deploy-learned-lcs"
PY_LCS = LCS / ".venv/bin/python"
C3_BIN = WT / "bazel-bin/systems/controllers/learned_lcs_c3_check"
V4 = LCS / "outputs/sim_belt_v4_20260929"
V4A_CKPT = V4 / "ckpt_t1/2026-09-29/1niukyhl/checkpoint_epoch_0300.pt"
V4A_DEPLOY = V4 / "deploy_t1"
V8OUT = LCS / "outputs/sim_belt_v8_20261001"
CACHE = V8OUT / "cache"
EVAL = REPO / "data/lcs/mpc_eval"
GOALS = EVAL / "20261001-v8/goals"
V4R = EVAL / "20260929-v4-retrain"
POOL = Path("data/lcs/onpolicy/v8/pool.json")
ARCHIVE_REL = "systems/parameters/learned_archive/2026-10-01"
DATE = "20261001"

sys.path.insert(0, str(V4R / "scripts"))
sys.path.insert(0, str(REPO / "src"))

T0 = ["flat_engaged", "flat_engaged_urtip", "flat_engaged_urtip6", "flat_engaged_urtip6_high"]
TYAW = ["flat_engaged_yawp05", "flat_engaged_yawm05", "flat_engaged_yawp10",
        "flat_engaged_yawm10"]
YAWS = ("yawp15", "yawm15")
EVAL_TARGETS = T0 + [f"{t}_{y}" for y in YAWS for t in T0]
ALL_TARGETS = EVAL_TARGETS + TYAW
SS = "data/lcs/start_states"
# (run, target, start states, variants, [(recording link, episode)])
SEAT_RUNS = [
    ("fe_nominal", "flat_engaged", f"{SS}/pre_place_1_osc.npz", None,
     [("nominal", "pre_place_1_osc")]),
    ("fe_set1", "flat_engaged", f"{SS}/grasp_variants/set1", "gv_01,gv_05",
     [("gv_01", "gv_01"), ("gv_05", "gv_05")]),
    ("urtip", "flat_engaged_urtip", f"{SS}/ur_depth", "ud+3_r0", [("ud+3_r0", "ud+3_r0")]),
    ("urtip6", "flat_engaged_urtip6", f"{SS}/ur_depth", "ud+6_r0", [("ud+6_r0", "ud+6_r0")]),
    ("urtip6_high", "flat_engaged_urtip6_high", f"{SS}/ur_depth", "ud+6_r0",
     [("ud+6_r0", "ud+6_r0")]),
]
# lines of a per-target LCS yaml that a new LCS replaces (everything else = V4a's file)
YAML_KEYS = ("A", "B", "D", "d", "E", "F", "H", "c", "u_lb", "u_ub", "checkpoint",
             "checkpoint_sha256", "exported_at")
NPZ_KEYS = ("A", "B", "D", "d", "E", "F", "H", "c", "G", "J", "u_lb", "u_ub", "checkpoint",
            "checkpoint_sha256", "exported_at")
CH = ["f_x", "f_y", "f_z", "ur_x", "ur_y", "ur_z", "fr_x", "fr_y", "fr_z", "urr_x", "urr_y",
      "urr_z"]
INS_HARD, INS_SOFT, RHO_MAX, BFRO_MAX = 0.635, 0.560, 1.05, 2 * 48.3
V4A_INS_H7, H7_FACTOR = 1.98, 1.25  # V4a insertion h7 (eval_results.json), gate = 1.25x
MS_WEIGHT = 0.1
C3_TOL_ONLY = 1e-5  # c3check bound violations below this count as tolerance-only
BOX_INS = False  # --box-ins: promoted models keep V4a ub-ins bounds (no trust region)
NO_H7_GATE = False  # --no-h7-gate: h7 logged only; candidates still need the one-step gates
MAX_FT = 5
STEPS = ["collect", "eval", "encode", "trust", "finetune", "export", "assets", "c3check",
         "tables", "register"]
LETTERS = "abcdefghij"


def url(port: int) -> str:
    return f"udpm://239.255.76.{port - 7600}:{port}?ttl=0"


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p)
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


class Ctx:
    """Paths, URLs and sizes of one invocation (normal run or --smoke)."""

    def __init__(self, a):
        self.a = a
        self.smoke = a.smoke
        self.root = REPO / (a.root or ("data/lcs/onpolicy/v8/smoke_loop" if a.smoke
                                       else "data/lcs/onpolicy/v8"))
        self.i = a.iter
        self.it = self.root / f"iter{a.iter}"
        self.logs = self.it / "logs"
        self.out = V8OUT / "smoke" if a.smoke else V8OUT
        self.cache = self.out / "cache" if a.smoke else CACHE
        self.eval_base = EVAL / "20261001-v8/smoke_eval" if a.smoke else EVAL
        p = 7749 if a.smoke else None
        self.url_collect = url(a.collect_port or p or 7752)
        self.url_seat = url(a.seat_port or p or 7750)
        self.url_yaw = url(a.yaw_port or p or (7751 if a.concurrency >= 3 else 7750))
        self.attempts = a.attempts or (2 if a.smoke else 12)
        self.steps = a.steps or (300 if a.smoke else 3000)
        self.eval_cells = a.eval_cells.split(",") if a.eval_cells else (
            ["fe_nominal"] if a.smoke else None)
        self.suites = ["seat6"] if a.smoke else ["seat6", "yaw12"]


# ---------------------------------------------------------------- state


@contextmanager
def locked(ctx: Ctx, i: int):
    d = ctx.root / f"iter{i}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".state.lock", "w") as fl:
        fcntl.flock(fl, fcntl.LOCK_EX)
        yield d / "state.json"


def get_state(ctx: Ctx, i: int) -> dict:
    p = ctx.root / f"iter{i}" / "state.json"
    return jload(p) if p.is_file() else {"iter": i, "steps": {}}


def set_step(ctx: Ctx, i: int, step: str, **kw):
    with locked(ctx, i) as p:
        st = jload(p) if p.is_file() else {"iter": i, "steps": {}}
        st["steps"].setdefault(step, {}).update(kw)
        jdump(p, st)


def step_done(ctx: Ctx, i: int, step: str) -> bool:
    s = get_state(ctx, i)["steps"].get(step, {})
    return s.get("status") in ("done", "skipped") and all(
        Path(o).exists() for o in s.get("outputs", []))


def move_aside(paths):
    for p in map(Path, paths):
        if p.exists() or p.is_symlink():
            n = 0
            while Path(f"{p}.prev{n}").exists():
                n += 1
            p.rename(f"{p}.prev{n}")
            print(f"  moved {p} -> {p}.prev{n}")


def run(cmd, log: Path, cwd=REPO, ctx=None, step=None) -> int:
    """Subprocess with its own log (appended); pid recorded in the state while it runs."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n# {now()} {' '.join(map(str, cmd))}\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
        if ctx is not None and step:
            set_step(ctx, ctx.i, step, pids=get_state(ctx, ctx.i)["steps"].get(step, {}).get(
                "pids", []) + [p.pid])
        return p.wait()


# ---------------------------------------------------------------- models and assets


def v4a_goal_dir(t: str) -> Path:
    if t in T0:
        return V4R / "t1/goals" / t
    if t.endswith(YAWS):
        return V4R / "t1_yaw/goals" / t
    return EVAL / "20261001-v8/goals_v4a/goals" / t


def v4a_assets() -> dict:
    out = {}
    for t in ALL_TARGETS:
        g = v4a_goal_dir(t)
        gj = jload(g / "goal.json")
        out[t] = {"deploy": rel(g / "deploy.npz"), "demo_goals": rel(g / "demo_goals.npz"),
                  "params": gj["params"], "lcs_yaml": gj["lcs_yaml"], "goal_dir": rel(g)}
    return out


def model_path(ctx: Ctx, i: int) -> Path:
    return ctx.root / f"iter{i}" / "model.json"


def load_model(ctx: Ctx, i: int) -> dict:
    p = model_path(ctx, i)
    if i == 0 and not p.is_file():
        m0 = {"id": "V4a", "iter": 0, "tag": "v4a", "checkpoint": str(V4A_CKPT),
              "deploy_dir": str(V4A_DEPLOY), "box_id": "ub-ins", "box_json": None,
              "buffer_size": 0, "source": "V4a (T1), 09-29 export", "eval_of": 0,
              "goals_dir": None, "assets": v4a_assets()}
        if ctx.a.dry_run:
            return m0
        jdump(p, m0)
    if not p.is_file():
        raise SystemExit(f"{rel(p)} missing: run --iter {i - 1} finetune..c3check first")
    return jload(p)


def n_accepted_before(ctx: Ctx, i: int) -> int:
    n = 0
    for j in range(i):
        f = ctx.root / f"iter{j}" / "finetune.json"
        n += f.is_file() and jload(f)["decision"] == "accepted"
    return n


def new_ids(ctx: Ctx, i: int):
    """(model id, tag) of the fine-tune of iteration i."""
    if ctx.smoke:
        return (f"V8smoke{i + 1}", f"smoke{i + 1}") if i else ("V8smoke", "smoke")
    x = LETTERS[n_accepted_before(ctx, i)]
    return f"V8{x}", f"v8{x}"


def yaml_blocks(lines):
    """Top-level key -> (start, end) line span (key line + its indented / '- ' lines)."""
    spans, key, start = {}, None, None
    for n, ln in enumerate(lines + [""]):
        top = ln and not ln.startswith((" ", "-", "#")) and ":" in ln
        if top or n == len(lines):
            if key is not None:
                spans[key] = (start, n)
            key, start = (ln.split(":", 1)[0], n) if top else (None, None)
    return spans


def swap_lcs_yaml(src_text: str, new_text: str) -> tuple[list[str], int]:
    """V4a per-target LCS yaml with the YAML_KEYS blocks taken from the new deploy's yaml."""
    src, new = src_text.splitlines(), new_text.splitlines()
    ss, ns = yaml_blocks(src), yaml_blocks(new)
    out, changed, n = [], 0, 0
    while n < len(src):
        k = next((k for k, (a, _) in ss.items() if a == n), None)
        if k in YAML_KEYS:
            a, b = ss[k]
            na, nb = ns[k]
            blk = new[na:nb]
            changed += sum(x != y for x, y in zip(src[a:b], blk)) + abs(len(blk) - (b - a))
            out += blk
            n = b
        else:
            out.append(src[n])
            n += 1
    return out, changed


def build_assets(model: dict, targets, tag: str, log) -> dict:
    """Goal deploys + worktree yamls for `model` (deploy_dir) from V4a's per-target files."""
    import yaml

    dep_dir = Path(model["deploy_dir"])
    new_yaml = (dep_dir / "learned_lcs.yaml").read_text()
    with np.load(dep_dir / "deploy.npz", allow_pickle=True) as d:
        newd = {k: d[k] for k in d.files}
    with np.load(V4A_DEPLOY / "deploy.npz", allow_pickle=True) as d:
        v4d = {k: d[k] for k in d.files}
    for k in set(newd) - set(NPZ_KEYS):  # fast export: encoder side copied from deploy_t1
        assert np.array_equal(newd[k], v4d[k]), f"{dep_dir}: {k} differs from deploy_t1"
    gdir_root = GOALS / tag
    arch_rel = f"{ARCHIVE_REL}/{tag}"
    arch = WT / arch_rel
    (arch / "learned_lcs").mkdir(parents=True, exist_ok=True)
    out = {}
    for t in targets:
        src = v4a_goal_dir(t)
        gj = jload(src / "goal.json")
        g = gdir_root / t
        g.mkdir(parents=True, exist_ok=True)
        with np.load(src / "deploy.npz", allow_pickle=True) as d:
            gd = {k: d[k] for k in d.files}
        for k in NPZ_KEYS:
            gd[k] = newd[k]
        np.savez(g / "deploy.npz", **gd)
        with np.load(g / "deploy.npz", allow_pickle=True) as a, \
                np.load(src / "deploy.npz", allow_pickle=True) as b:
            for k in ("z_goal", "z_goal_stage1", "z_goal_stage2", "z_mean", "z_std"):
                assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), (t, k)
            assert set(a.files) == set(b.files)
        with np.load(src / "demo_goals.npz", allow_pickle=True) as d:
            dg = {k: d[k] for k in d.files}
        dg["deploy_sha256"] = np.array(sha(g / "deploy.npz"))
        np.savez(g / "demo_goals.npz", **dg)
        # LCS yaml: V4a's per-target file, header + YAML_KEYS blocks replaced
        src_lcs = WT / gj["lcs_yaml"]
        lines, n_changed = swap_lcs_yaml(src_lcs.read_text(), new_yaml)
        head = [(f"# {DATE} V8 {model['id']} ({tag}): copy of {gj['lcs_yaml']} (sha256 "
                 f"{sha(src_lcs)[:12]}...) with {', '.join(YAML_KEYS)} from {dep_dir}"
                 "/learned_lcs.yaml; goals unchanged (frozen encoder).")]
        lcs_name = f"learned_lcs_{tag}_{t}.yaml"
        lcs_p = arch / "learned_lcs" / lcs_name
        lcs_p.write_text("\n".join(head + lines) + "\n")
        doc, ref = yaml.safe_load(lcs_p.read_text()), yaml.safe_load(src_lcs.read_text())
        nd = yaml.safe_load(new_yaml)
        assert set(doc) == set(ref), t
        for k in doc:
            want = nd[k] if k in YAML_KEYS else ref[k]
            assert doc[k] == want, (t, k)
        # params: V4a's params yaml, header + lcs_file only
        src_p = WT / gj["params"]
        old_lcs = gj["lcs_yaml"]
        body = src_p.read_text()
        assert body.count(old_lcs) == 1, (t, old_lcs)
        p_name = f"round_belt_controller_params_learned_eval_{tag}_{t}.yaml"
        p_path = arch / p_name
        p_path.write_text(
            f"# {DATE} V8 {model['id']}: copy of {gj['params']} (sha256 {sha(src_p)[:12]}...); "
            "only lcs_file changed.\n"
            + body.replace(old_lcs, f"{arch_rel}/learned_lcs/{lcs_name}"))
        pa, pb = yaml.safe_load(p_path.read_text()), yaml.safe_load(src_p.read_text())
        assert pa["learned_mpc"].pop("lcs_file") != pb["learned_mpc"].pop("lcs_file")
        assert pa == pb, t
        gj2 = {**gj, "model": model["id"], "v4a_goal_dir": rel(src),
               "deploy": rel(g / "deploy.npz"), "deploy_sha256": sha(g / "deploy.npz"),
               "demo_goals": rel(g / "demo_goals.npz"),
               "demo_goals_sha256": sha(g / "demo_goals.npz"),
               "lcs_yaml": f"{arch_rel}/learned_lcs/{lcs_name}",
               "params": f"{arch_rel}/{p_name}", "lcs_yaml_sha256": sha(lcs_p),
               "params_sha256": sha(p_path), "lcs_source_deploy": str(dep_dir),
               "lcs_yaml_changed_lines": n_changed}
        jdump(g / "goal.json", gj2)
        out[t] = {"deploy": gj2["deploy"], "demo_goals": gj2["demo_goals"],
                  "params": gj2["params"], "lcs_yaml": gj2["lcs_yaml"], "goal_dir": rel(g),
                  "deploy_sha256": gj2["deploy_sha256"], "lcs_yaml_sha256": sha(lcs_p),
                  "params_sha256": sha(p_path), "lcs_yaml_changed_lines": n_changed,
                  "lcs_yaml_header_lines_added": len(head), "params_changed_lines": 2}
        print(f"  assets {t}: LCS yaml {n_changed} changed lines (+{len(head)} header)", file=log)
    return out


def c3check(assets: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for t, a in assets.items():
        lg = out_dir / f"c3check_{t}.log"
        with open(lg, "w") as f:
            rc = subprocess.run([str(C3_BIN), "--params", a["params"]], cwd=WT, stdout=f,
                                stderr=subprocess.STDOUT, check=False).returncode
        text = lg.read_text()
        last = (text.strip().splitlines() or [""])[-1]
        ok = rc == 0 and last.startswith("PASS")
        # bound violations below C3_TOL_ONLY are OSQP tolerance, as accepted for earlier models
        viol = [float(m) for m in re.findall(r"bounds: worst violation ([0-9.eE+-]+)", text)]
        other = re.search(r"-> VIOLATED", re.sub(r"bounds: worst violation .*", "", text))
        tol_only = (not ok and not other and viol and max(viol) < C3_TOL_ONLY)
        res[t] = {"rc": rc, "last": last, "pass": ok or bool(tol_only),
                  "tolerance_only": bool(tol_only), "max_bound_violation": max(viol, default=0.0)}
    return res


# ---------------------------------------------------------------- harness / eval


def harness(ctx, out: Path, lcm: str, a: dict, starts: str, variants, log: Path) -> dict:
    """One eval_learned_mpc run (skipped if its index.json already has every episode)."""
    n_want = len(variants.split(",")) if variants else 1
    idx = out / "index.json"
    if idx.is_file() and len(jload(idx)["episodes"]) >= n_want:
        return {"skipped": True, "rc": 0, "wall_s": None}
    cmd = [sys.executable, "scripts/lcs/eval_learned_mpc.py", "--mode", "learned",
           "--lcm-url", lcm, "--record", "--max-episode-s", "16", "--settle-s", "0.5",
           "--pace", "1", "--params", a["params"], "--deploy", a["deploy"],
           "--demo-goals", a["demo_goals"], "--start-states", starts, "--out", rel(out)]
    if variants:
        cmd += ["--variants", variants]
    t0 = time.time()
    rc = run(cmd, log)
    return {"skipped": False, "rc": rc, "wall_s": time.time() - t0}


def eval_runs(suite: str, cells=None):
    runs = []
    for run_, t, s, v, links in SEAT_RUNS:
        if suite == "seat6":
            runs.append((run_, t, s, v, [(f"{t}__{ln}", f"{run_}/recordings/episode_{e}_0")
                                         for ln, e in links]))
        else:
            for y in YAWS:
                runs.append((f"{run_}_{y}", f"{t}_{y}", s, v,
                             [(f"{t}_{y}__{ln}", f"{run_}_{y}/recordings/episode_{e}_0")
                              for ln, e in links]))
    return [r for r in runs if cells is None or r[0] in cells or r[0].rsplit("_y", 1)[0] in cells]


def eval_dir(ctx: Ctx, model: dict, suite: str) -> Path:
    n = len(eval_runs(suite, ctx.eval_cells))
    name = suite if ctx.eval_cells is None else f"{suite}sub{n}"
    return ctx.eval_base / f"{DATE}_{model['id']}_{name}_Qw_{model['box_id']}"


def run_suite(ctx: Ctx, model: dict, suite: str, lcm: str, log: Path) -> dict:
    d = eval_dir(ctx, model, suite)
    (d / "recordings").mkdir(parents=True, exist_ok=True)
    for t in {r[1] for r in eval_runs(suite, ctx.eval_cells)}:
        gl = d / "goals" / t  # the 09-29 table code reads <root>/goals/<target>
        gl.parent.mkdir(exist_ok=True)
        if not gl.is_symlink():
            gl.symlink_to(REPO / model["assets"][t]["goal_dir"])
    res = []
    for run_, t, s, v, links in eval_runs(suite, ctx.eval_cells):
        r = harness(ctx, d / run_, lcm, model["assets"][t], s, v,
                    d / f"{run_}.stdout.log")
        res.append({"run": run_, **r})
        with open(log, "a") as f:
            print(f"  {suite} {run_} rc {r['rc']} wall {r['wall_s']}", file=f)
        for name, target in links:
            if (d / target).parent.parent.is_dir():
                lk = d / "recordings" / name
                if lk.is_symlink():
                    lk.unlink()
                lk.symlink_to(Path("..") / target)
    table = suite_table(model, suite, d, ctx.eval_cells)
    return {"dir": rel(d), "runs": res, "table": rel(d / "table.md"), "summary": table}


def safety(rows) -> dict:
    unsafe = [r["run"] for r in rows if not (r["ur_all"] and r["f_all"]) or r["F_rot_max"] > 90
              or r["hand_zmin"] < 11.2]
    return {"unsafe_cells": unsafe, "n_unsafe": len(unsafe),
            "aborts": sum(bool(r["aborts"]) for r in rows)}


def summarize(rows) -> dict:
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "strict": sum(not r["strict_fail"] for r in rows),
            "engaged": sum(r["outcome"] == "engaged" for r in rows),
            "rmse": float(np.mean([r["rmse"] for r in rows])),
            "F_rot_max": float(max(r["F_rot_max"] for r in rows)),
            "hand_zmin": float(min(r["hand_zmin"] for r in rows)),
            "stale": int(sum(r["stale"] for r in rows)),
            "solve_p95_med": float(np.median([r["solve_p95"] for r in rows])),
            **safety(rows)}


def suite_table(model: dict, suite: str, d: Path, cells=None) -> dict:
    """table.md / table.json with the 09-29 columns (eval_table_v4x / eval_table_yaw)."""
    import eval_table_v4 as T
    import eval_table_v4x as X
    import eval_table_yaw as Y

    dec = Path(model["deploy_dir"]) / "decoder.npz"
    rows = []
    for run_, t, _, _, links in eval_runs(suite, cells):
        base = t.rsplit("_yaw", 1)[0] if suite == "yaw12" else t
        label = model["id"] + (f"_{t.rsplit('_', 1)[1]}" if suite == "yaw12" else "")
        T.DEC[label] = dec
        for name, _ in links:
            start = name.split("__", 1)[1]
            if not (d / run_ / "index.json").is_file():
                continue
            x = T.row(label, d, run_, t, start)
            x.update(X.extra(d, run_, start))
            if suite == "yaw12":
                x.update(Y.yaw_cols(d, run_, t, start, base))
            x["s0"] = float("nan") if x["s0"] is None else x["s0"]
            rows.append(x)
    hdr, fmt = (Y.HDR, Y.fmt) if suite == "yaw12" else (X.HDR, X.fmt)
    s = summarize(rows)
    md = [f"# {model['id']} · {suite} · Qw · {model['box_id']}", "",
          f"deploy `{model['deploy_dir']}`; checkpoint `{model['checkpoint']}`", "", hdr]
    md += [fmt(r) for r in rows]
    if rows:
        md += ["", ("| model | n | strict | engaged | mean belt RMSE mm | max F rot deg | "
                    "min panda_hand z mm | unsafe cells | aborts | stale | solve p95 ms med |"),
               "|---" * 11 + "|",
               (f"| {model['id']} | {s['n']} | {s['strict']}/{s['n']} | "
                f"{s['engaged']}/{s['n']} | {s['rmse']:.1f} | {s['F_rot_max']:.0f} | "
                f"{s['hand_zmin']:.1f} | {s['n_unsafe']} | {s['aborts']} | {s['stale']} | "
                f"{s['solve_p95_med']:.0f} |")]
    jdump(d / "table.json", rows)
    (d / "table.md").write_text("\n".join(md) + "\n")
    return s


def v4a_eval_rows() -> dict:
    """Iteration-0 rows: V4a seat6 / yaw12 of 2026-09-29 (plan Q12)."""
    seat = [r for r in jload(V4R / "table_t1.json") if r["model"] == "t1"]
    yaw = [r for r in jload(V4R / "t1_yaw/table.json") if r["model"] in ("t1_yawp15",
                                                                          "t1_yawm15")]
    return {"seat6": {"dir": rel(V4R / "t1"), "table": rel(V4R / "table_t1.md"),
                      "summary": summarize(seat), "reused": True},
            "yaw12": {"dir": rel(V4R / "t1_yaw"), "table": rel(V4R / "t1_yaw/table.md"),
                      "summary": summarize(yaw), "reused": True}}


# ---------------------------------------------------------------- steps


def select_cells(cells, k):
    """First k cells interleaved over strata (all when k >= len)."""
    if k >= len(cells):
        return cells
    by = {}
    for c in cells:
        by.setdefault(c["stratum"], []).append(c)
    out, n = [], 0
    while len(out) < k:
        for v in by.values():
            if n < len(v) and len(out) < k:
                out.append(v[n])
        n += 1
    return out


def cell_name(k, c):
    return f"c{k:02d}_{c['target']}__{c['start_set'].replace('/', '-')}-{c['start_id']}"


def st_collect(ctx: Ctx, dry: bool):
    man = ctx.it / "manifest.json"
    n = 12 * max(1, -(-ctx.attempts // 12))  # manifest = whole quotas; first k cells run
    sample = [sys.executable, "scripts/lcs/onpolicy_pool.py", "sample", "--pool", str(POOL),
              "--iter", str(ctx.i), "--n", str(n), "--out", rel(man)]
    check = [sys.executable, "scripts/lcs/onpolicy_pool.py", "check", "--all"]
    model = load_model(ctx, ctx.i)
    if dry:
        if not man.is_file():
            tmp = Path(f"/tmp/onpolicy_dry_manifest_{os.getpid()}.json")
            subprocess.run(sample[:-1] + [str(tmp)], cwd=REPO, check=True,
                           stdout=subprocess.DEVNULL)
            cells = jload(tmp)["cells"]
            tmp.unlink()
            print(f"  would run: {' '.join(sample)}")
        else:
            cells = jload(man)["cells"]
        print(f"  would run: {' '.join(check)}")
        for k, c in enumerate(select_cells(cells, ctx.attempts)):
            a = model["assets"].get(c["target"], {})
            print(f"  cell {cell_name(k, c)} on {ctx.url_collect}: params {a.get('params')}"
                  f" deploy {a.get('deploy')} starts {c['start_states_arg']}"
                  f" --variants {c['variants_arg']} -> {rel(ctx.it / 'collect' / cell_name(k, c))}")
        return {}
    log = ctx.logs / "collect.log"
    if not man.is_file() and run(sample, log):
        raise SystemExit("pool sample failed")
    if run(check, log):
        raise SystemExit("onpolicy_pool.py check --all failed (see collect.log)")
    cells = select_cells(jload(man)["cells"], ctx.attempts)
    att = []
    for k, c in enumerate(cells):
        name = cell_name(k, c)
        out = ctx.it / "collect" / name
        r = harness(ctx, out, ctx.url_collect, model["assets"][c["target"]],
                    c["start_states_arg"], c["variants_arg"], ctx.logs / f"collect_{name}.log")
        with open(log, "a") as f:
            print(f"  {name} rc {r['rc']} wall {r['wall_s']}", file=f)
        att.append(attempt_row(k, c, name, out, r))
    s = {"iter": ctx.i, "model": model["id"], "box_id": model["box_id"], "url":
         ctx.url_collect, "n": len(att), "attempts": att,
         "strict": sum(a["strict"] for a in att), "engaged": sum(a["outcome"] == "engaged"
                                                               for a in att),
         "aborts": sum(bool(a["reason"]) for a in att),
         "wall_s": [a["wall_s"] for a in att]}
    jdump(ctx.it / "collect" / "summary.json", s)
    bad = [a["name"] for a in att if a["rc"] != 0]
    if bad:
        raise SystemExit(f"harness exit != 0: {bad}")
    return {"outputs": [str(ctx.it / "collect" / "summary.json")]}


def attempt_row(k, c, name, out: Path, r) -> dict:
    ep = jload(out / "index.json")["episodes"][0]
    with np.load(out / ep["file"], allow_pickle=True) as d:
        src = np.asarray(d["sim_cmd_source"])
    wrap, h = ep["final_wrap_deg"], ep["final_h_median_mm"]
    return {"k": k, "cell": c["cell"], "name": name, "target": c["target"],
            "stratum": c["stratum"], "episode": rel(out / ep["file"]),
            "outcome": ep["outcome"], "strict": bool(wrap >= 90 and abs(h) <= 1),
            "wrap": wrap, "h": h, "rmse": ep["alignment"]["final"]["belt_rmse_mm"],
            "reason": ep.get("reason"), "grasp_ok_final": ep.get("grasp_ok_final"),
            "n_frames": len(src), "n_mpc_frames": int(np.isin(src, (0, 1)).sum()),
            "wall_s": r["wall_s"], "rc": r["rc"]}


def st_eval(ctx: Ctx, dry: bool):
    model = load_model(ctx, ctx.i)
    forced = ctx.a.force == "eval"
    if (ctx.i == 0 or model.get("eval_of", ctx.i) != ctx.i) and not forced:
        j = model.get("eval_of", 0)
        print(f"  eval skipped: {model['id']} {model['box_id']} rows of iteration {j} reused")
        if not dry:
            ev = v4a_eval_rows() if j == 0 else jload(ctx.root / f"iter{j}" / "eval.json")
            jdump(ctx.it / "eval.json", {**ev, "reused_from_iter": j})
        return {"status": "skipped", "outputs": [str(ctx.it / "eval.json")]}
    urls = {"seat6": ctx.url_seat, "yaw12": ctx.url_yaw}
    if dry:
        for s in ctx.suites:
            d = eval_dir(ctx, model, s)
            for run_, t, st, v, _ in eval_runs(s, ctx.eval_cells):
                print(f"  eval {s} {run_} on {urls[s]}: {model['assets'][t]['params']}"
                      f" starts {st} {v or ''} -> {rel(d / run_)}")
        return {}
    res, log = {}, ctx.logs / "eval.log"

    def go(s):
        res[s] = run_suite(ctx, model, s, urls[s], log)

    if urls["seat6"] != urls["yaw12"] and len(ctx.suites) > 1:
        th = [threading.Thread(target=go, args=(s,)) for s in ctx.suites]
        [t.start() for t in th]
        [t.join() for t in th]
    else:
        for s in ctx.suites:
            go(s)
    jdump(ctx.it / "eval.json", res)
    bad = [r["run"] for s in res.values() for r in s["runs"] if r["rc"] != 0]
    if bad:
        raise SystemExit(f"eval harness exit != 0: {bad}")
    return {"outputs": [str(ctx.it / "eval.json")] + [str(REPO / r["dir"]) for r in
                                                        res.values()]}


def split_cache(src: Path, dst_dir: Path) -> list[str]:
    """One cache per rollout (file) of `src`, same format (file_idx reset)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    with np.load(src) as d:
        c = {k: d[k] for k in d.files}
    out = []
    for fi, f in enumerate(c["files"]):
        m = c["file_idx"] == fi
        part = {k: (v[m] if v.ndim and len(v) == len(m) else v) for k, v in c.items()}
        part["file_idx"] = np.zeros(int(m.sum()), c["file_idx"].dtype)
        part["files"] = np.array([f])
        p = dst_dir / f"r{fi:02d}.npz"
        np.savez(p, **part)
        out.append(str(p))
    return out


def st_encode(ctx: Ctx, dry: bool):
    summ = jload(ctx.it / "collect" / "summary.json") if not dry else None
    out = ctx.cache / f"iter{ctx.i}.npz"
    lst = ctx.it / "encode_files.txt"
    cmd = [PY_LCS, "scripts/encode_latents_cache.py", "--checkpoint", V4A_CKPT,
           "--files", lst, "--out", out]
    vcache = CACHE / "v3_val_episodes.npz"
    if dry:
        print(f"  would run (lcs_learning): {' '.join(map(str, cmd))}")
        print(f"  per-rollout caches -> {ctx.cache}/iter{ctx.i}_rollouts/; buffer.json FIFO "
              f"{ctx.a.buffer_max}; held-out val cache {vcache} "
              f"({'present' if vcache.is_file() else 'built on first use'})")
        return {}
    eps = [str(REPO / a["episode"]) for a in summ["attempts"]]
    lst.write_text("\n".join(eps) + "\n")
    if not vcache.is_file():
        vl = CACHE / "v3_val_episodes_files.txt"
        vl.write_text("\n".join(map(str, jload(REPO / "data/lcs/v3/split.json")["val"])) + "\n")
        if run([PY_LCS, "scripts/encode_latents_cache.py", "--checkpoint", V4A_CKPT, "--files",
                vl, "--out", vcache], ctx.logs / "encode.log", cwd=LCS):
            raise SystemExit("val-episodes cache failed")
    if run(cmd, ctx.logs / "encode.log", cwd=LCS, ctx=ctx, step="encode"):
        raise SystemExit("encode failed")
    parts = split_cache(out, ctx.cache / f"iter{ctx.i}_rollouts")
    with np.load(out) as d:
        files = [str(f) for f in d["files"]]
    assert files == eps, "cache file order != collection order"
    new = [{"iter": ctx.i, "k": a["k"], "cell": a["cell"], "name": a["name"],
            "episode": a["episode"], "cache": p, "n_frames": a["n_frames"],
            "n_mpc_frames": a["n_mpc_frames"], "outcome": a["outcome"], "strict": a["strict"]}
           for a, p in zip(summ["attempts"], parts)]
    jdump(ctx.it / "rollouts.json", new)
    allr = []
    for j in range(ctx.i + 1):
        f = ctx.root / f"iter{j}" / "rollouts.json"
        if f.is_file():
            allr += jload(f)
    buf = sorted(allr, key=lambda r: (r["iter"], r["k"]))[-ctx.a.buffer_max:]
    b = {"iter": ctx.i, "max": ctx.a.buffer_max, "n": len(buf), "rollouts": buf,
         "n_mpc_frames": sum(r["n_mpc_frames"] for r in buf)}
    jdump(ctx.it / "buffer.json", b)
    jdump(ctx.root / "buffer.json", b)
    return {"outputs": [str(out), str(ctx.it / "buffer.json")]}


def trust_region(buf: dict, eta: float, floor: float):
    """Plan §5 per channel over the buffer's MPC frames (sim_cmd_source 0/1)."""
    us = []
    for r in buf["rollouts"]:
        with np.load(REPO / r["episode"], allow_pickle=True) as d:
            u, src = np.asarray(d["actions"], float), np.asarray(d["sim_cmd_source"])
        us.append(u[np.isin(src, (0, 1))])
    u = np.concatenate(us)
    with np.load(V4A_DEPLOY / "deploy.npz") as d:
        lb_ins, ub_ins = d["u_lb"].astype(float), d["u_ub"].astype(float)
    half = (ub_ins - lb_ins) / 2
    mean, std = u.mean(0), u.std(0)
    delta = np.maximum(eta * std, floor * half)
    lb_raw, ub_raw = mean - delta, mean + delta
    lb, ub = np.maximum(lb_raw, lb_ins), np.minimum(ub_raw, ub_ins)
    w = 2 * floor * half  # keep a positive width if the mean sits outside ub-ins
    recenter = ub - lb < w
    lb = np.where(recenter, np.clip(np.minimum(lb, ub_ins - w), lb_ins, None), lb)
    ub = np.where(recenter, np.clip(np.maximum(ub, lb_ins + w), None, ub_ins), ub)
    assert np.all(lb < ub)
    sc = np.array([1e3] * 6 + [np.degrees(1.0)] * 6)
    unit = ["mm"] * 6 + ["deg"] * 6
    ch = []
    for j, n in enumerate(CH):
        ch.append({"channel": n, "unit": unit[j], "mean": mean[j], "std": std[j],
                   "delta": delta[j], "lb_ins": lb_ins[j], "ub_ins": ub_ins[j],
                   "half_ins": half[j], "lb_raw": lb_raw[j], "ub_raw": ub_raw[j],
                   "lb": lb[j], "ub": ub[j], "floor_hit": bool(eta * std[j] < floor * half[j]),
                   "clip_lb_hit": bool(lb_raw[j] < lb_ins[j]),
                   "clip_ub_hit": bool(ub_raw[j] > ub_ins[j]), "recenter_hit": bool(recenter[j]),
                   "delta_over_half_ins": delta[j] / half[j],
                   "box_over_ins": (ub[j] - lb[j]) / (2 * half[j]),
                   "disp": {k: float(v * sc[j]) for k, v in (
                       ("mean", mean[j]), ("std", std[j]), ("delta", delta[j]),
                       ("lb", lb[j]), ("ub", ub[j]), ("lb_ins", lb_ins[j]),
                       ("ub_ins", ub_ins[j]))}})
    r = delta / half
    b = (ub - lb) / (2 * half)

    def gm(x):
        return float(np.exp(np.mean(np.log(x))))

    ratios = {"delta_trans_gm": gm(r[:6]), "delta_rot_gm": gm(r[6:]),
              "box_trans_gm": gm(b[:6]), "box_rot_gm": gm(b[6:]),
              "min_delta_over_half_ins": float(r.min()),
              "min_channel": CH[int(np.argmin(r))]}
    return {"eta": eta, "floor": floor, "n_rollouts": len(us), "n_frames": len(u),
            "units": "SI: m / rad per 75 ms; disp: mm / deg", "channels": ch,
            "ratios": ratios, "u_lb": lb.tolist(), "u_ub": ub.tolist(),
            "n_floor_hits": sum(c["floor_hit"] for c in ch),
            "n_clip_hits": sum(c["clip_lb_hit"] or c["clip_ub_hit"] for c in ch)}


def st_trust(ctx: Ctx, dry: bool):
    eta = ctx.a.eta_base + ctx.a.eta_step * ctx.i  # eta of box_<i+1>
    box = ctx.it / f"box_{ctx.i + 1}.json"
    if dry:
        print(f"  box_{ctx.i + 1}: eta {eta}, floor {ctx.a.floor} x half(ub-ins), buffer <= "
              f"{ctx.a.buffer_max} -> {rel(ctx.it / 'trust_region.json')}, {rel(box)}")
        return {}
    buf = jload(ctx.it / "buffer.json")
    tr = trust_region(buf, eta, ctx.a.floor)
    tr["buffer"] = rel(ctx.it / "buffer.json")
    jdump(ctx.it / "trust_region.json", tr)
    jdump(box, {"u_lb": tr["u_lb"], "u_ub": tr["u_ub"],
                "source": f"V8 trust region ub-tr{ctx.i + 1}: eta {eta}, floor {ctx.a.floor}, "
                          f"{tr['n_rollouts']} rollouts / {tr['n_frames']} MPC frames of "
                          f"{rel(ctx.it / 'buffer.json')}, clipped to deploy_t1 (ub-ins)"})
    return {"outputs": [str(ctx.it / "trust_region.json"), str(box)]}


def cand_rows(rep: dict) -> list[dict]:
    by = {e["step"]: e["metrics"] for e in rep["evals"]}
    out = []
    for c in rep["candidates"]:
        m = by[c["step"]]
        ins, ve = m["insertion_test"], m["val_episodes"]
        hard = (ins["one_step_mm"] <= INS_HARD and m["rho_A"] <= RHO_MAX
                and m["B_fro"] <= BFRO_MAX)
        out.append({"step": c["step"], "checkpoint": c["checkpoint"],
                    "ins_1": ins["one_step_mm"], "ins_h7": ins["h7_decoded_mm"],
                    "valep_1": ve["one_step_mm"], "valep_h7": ve["h7_decoded_mm"],
                    "val_dec": m["val"]["decoded_rmse_mm"],
                    "buf_1": m["buffer"]["decoded_rmse_mm"],
                    "buf_h7": m["buffer"]["h7_decoded_mm"],
                    "fresh_1": m.get("fresh", {}).get("decoded_rmse_mm"),
                    "rho_A": m["rho_A"], "B_fro": m["B_fro"], "hard_ok": hard,
                    "soft_ok": hard and ins["one_step_mm"] <= INS_SOFT,
                    "score_h7": ins["h7_decoded_mm"] + ve["h7_decoded_mm"]})
    return out


def pick(cands):
    pool = [c for c in cands if c["soft_ok"]] or [c for c in cands if c["hard_ok"]]
    return min(pool, key=lambda c: c["score_h7"]) if pool else None


def st_finetune(ctx: Ctx, dry: bool):
    if ctx.i >= MAX_FT:
        print(f"  finetune skipped: budget of {MAX_FT} fine-tunes")
        if not dry:
            jdump(ctx.it / "finetune.json", {"decision": "budget"})
        return {"status": "skipped", "outputs": [str(ctx.it / "finetune.json")]}
    model = load_model(ctx, ctx.i)
    mid, tag = new_ids(ctx, ctx.i)
    ft_root = ctx.out / f"iter{ctx.i}"
    base = [PY_LCS, "scripts/finetune_lcs_latent.py", "--checkpoint", model["checkpoint"],
            "--replay", CACHE / "v4a_replay.npz", "--val", CACHE / "v4a_val.npz",
            "--insertion-test", CACHE / "v2_insertion_test.npz",
            "--val-episodes", CACHE / "v3_val_episodes.npz",
            "--fresh", ctx.cache / f"iter{ctx.i}.npz", "--steps", str(ctx.steps),
            "--save-every", str(max(ctx.steps // 6, 1))]
    ladder = [("primary", 3e-4, MS_WEIGHT), ("lr1e-4_ms7", 1e-4, MS_WEIGHT)]  # ms7 from the start
    prev = ctx.root / f"iter{ctx.i - 1}" / "finetune.json"
    prev_rej = ctx.i > 0 and prev.is_file() and jload(prev).get(
        "gate_decision", jload(prev)["decision"]) == "rejected"
    gate_h7 = H7_FACTOR * V4A_INS_H7
    if dry:
        print(f"  finetune from {model['id']} ({model['checkpoint']}) -> {mid}; "
              f"{ctx.steps} steps, candidates every {max(ctx.steps // 6, 1)}; ladder "
              f"{[x[0] for x in ladder] if prev_rej else ['primary']} under {ft_root}")
        print(f"  gates: candidates ins 1-step <= {INS_HARD} (prefer <= {INS_SOFT}), rho(A) <= "
              f"{RHO_MAX}, |B|_F <= {BFRO_MAX}; pick min ins h7 + val-episodes h7; accept if "
              f"ins h7 <= {gate_h7:.3f}")
        return {}
    buf = jload(ctx.it / "buffer.json")
    bargs = [x for r in buf["rollouts"] for x in ("--buffer", r["cache"])]
    attempts, decision, chosen = [], None, None
    for name, lr, ms in ladder if prev_rej else ladder[:1]:
        out = ft_root / f"ft_{name}"
        cmd = base + bargs + ["--lr", str(lr), "--out-dir", out, "--tag", f"{tag}_{name}"]
        if ms:
            cmd += ["--multistep-weight", str(ms)]
        if not (out / "finetune_report.json").is_file() and run(
                cmd, ctx.logs / f"finetune_{name}.log", cwd=LCS, ctx=ctx, step="finetune"):
            raise SystemExit(f"finetune {name} failed")
        rep = jload(out / "finetune_report.json")
        cands = cand_rows(rep)
        best = pick(cands)
        ok = best is not None and (NO_H7_GATE or best["ins_h7"] <= gate_h7)
        attempts.append({"name": name, "lr": lr, "multistep_weight": ms, "out_dir": str(out),
                         "fresh_before": rep["fresh_decoded_mm_before"],
                         "initial": cand_rows({"evals": rep["evals"], "candidates": [
                             {"step": 0, "checkpoint": model["checkpoint"]}]})[0],
                         "candidates": cands, "chosen": best, "h7_gate_ok": ok,
                         "train_s": rep["timing_s"]["train"],
                         "total_s": rep["timing_s"]["total"]})
        if best is None:
            decision = "failed"
            break
        if ok:
            decision, chosen = "accepted", best
            break
    decision = decision or "rejected"
    gate_decision = decision
    if ctx.smoke and decision == "rejected":  # smoke exercises export..c3check anyway
        decision, chosen = "accepted", attempts[-1]["chosen"]
    path = attempts[-1]["name"]
    res = {"iter": ctx.i, "from": model["id"], "candidate_id": mid, "decision": decision,
           "gate_decision": gate_decision, "smoke_override": gate_decision != decision,
           "path": path, "prev_rejected": prev_rej, "h7_gate_mm": gate_h7,
           "gates": {"ins_hard": INS_HARD, "ins_soft": INS_SOFT, "rho_A": RHO_MAX,
                     "B_fro": BFRO_MAX, "ins_h7": gate_h7},
           "chosen": chosen or attempts[-1]["chosen"], "attempts": attempts}
    jdump(ctx.it / "finetune.json", res)
    nxt = model_path(ctx, ctx.i + 1)
    if decision == "accepted":
        jdump(nxt, {"id": mid, "iter": ctx.i + 1, "tag": tag, "checkpoint": chosen["checkpoint"],
                    "deploy_dir": str(ctx.out / f"deploy_{tag}"),
                    "box_id": "ub-ins" if BOX_INS else f"ub-tr{ctx.i + 1}",
                    "box_json": str(ctx.root / "box_ub_ins.json") if BOX_INS
                    else str(ctx.it / f"box_{ctx.i + 1}.json"),
                    "buffer_size": buf["n"], "eval_of": ctx.i + 1,
                    "offline": {k: chosen[k] for k in ("ins_1", "ins_h7", "rho_A", "B_fro")},
                    "source": f"finetune iter {ctx.i} ({path}, step {chosen['step']}) from "
                              f"{model['id']}", "goals_dir": None, "assets": None})
    elif decision == "rejected":
        jdump(nxt, {**model, "iter": ctx.i + 1, "buffer_size": buf["n"],
                    "kept": f"iter {ctx.i} fine-tune rejected (h7 gate); {model['id']} kept"})
    print(f"  finetune {decision} via {path}: chosen {res['chosen'] and res['chosen']['step']}")
    if decision == "failed":
        raise SystemExit("finetune: no candidate passes the one-step / rho / |B| gates; stop")
    return {"outputs": [str(ctx.it / "finetune.json"), str(nxt)]}


def accepted(ctx: Ctx) -> bool:
    f = ctx.it / "finetune.json"
    return f.is_file() and jload(f)["decision"] == "accepted"


def st_export(ctx: Ctx, dry: bool):
    if not dry and not accepted(ctx):
        print("  export skipped: no accepted fine-tune")
        return {"status": "skipped"}
    nm = load_model(ctx, ctx.i + 1) if not dry else None
    dep = Path(nm["deploy_dir"]) if nm else ctx.out / "deploy_<tag>"
    box = ctx.it / f"box_{ctx.i + 1}.json"
    cmd = [PY_LCS, "scripts/export_learned_lcs_deploy.py", "--checkpoint",
           nm["checkpoint"] if nm else "<chosen candidate>", "--device", "cpu",
           "--from-deploy", V4A_DEPLOY, "--u-bounds-json", box, "--out-dir", dep]
    chk = [sys.executable, "scripts/checks/check_latent_encoder.py", "--deploy",
           dep / "deploy.npz"]
    if dry:
        print(f"  would run (lcs_learning): {' '.join(map(str, cmd))}")
        print(f"  would run: {' '.join(map(str, chk))}")
        return {}
    log = ctx.logs / "export.log"
    if not (dep / "deploy.npz").is_file() and run(cmd, log, cwd=LCS, ctx=ctx, step="export"):
        raise SystemExit("export failed")
    rc = run(chk, log)
    if rc or "ALL LATENT ENCODER CHECKS PASSED" not in log.read_text().split("# ")[-1]:
        raise SystemExit("check_latent_encoder failed")
    return {"outputs": [str(dep / "deploy.npz")]}


def targets_for(ctx: Ctx) -> list[str]:
    if ctx.a.targets:
        return ctx.a.targets.split(",")
    if ctx.smoke:
        return T0  # covers the smoke's seat cell and the first fe_GS / urtip_U cells
    return ALL_TARGETS


def st_assets(ctx: Ctx, dry: bool):
    if not dry and not accepted(ctx):
        print("  assets skipped: no accepted fine-tune")
        return {"status": "skipped"}
    if dry:
        n = len(targets_for(ctx)) if (ctx.it / "manifest.json").is_file() else "?"
        print(f"  assets for {n} targets -> {rel(GOALS)}/<tag>/<target>/, worktree "
              f"{ARCHIVE_REL}/<tag>/")
        return {}
    nm = load_model(ctx, ctx.i + 1)
    with open(ctx.logs / "assets.log", "a") as log:
        assets = build_assets(nm, targets_for(ctx), nm["tag"], log)
    nm.update(assets=assets, goals_dir=rel(GOALS / nm["tag"]))
    jdump(model_path(ctx, ctx.i + 1), nm)
    jdump(ctx.root / f"iter{ctx.i + 1}" / "assets.json", assets)
    return {"outputs": [str(ctx.root / f"iter{ctx.i + 1}" / "assets.json")]}


def st_c3check(ctx: Ctx, dry: bool):
    if not dry and not accepted(ctx):
        print("  c3check skipped: no accepted fine-tune")
        return {"status": "skipped"}
    out = ctx.root / f"iter{ctx.i + 1}" / "c3check"
    if dry:
        print(f"  would run {C3_BIN} --params <each params yaml> (cwd {WT}) -> {rel(out)}")
        return {}
    res = c3check(load_model(ctx, ctx.i + 1)["assets"], out)
    jdump(out / "summary.json", res)
    bad = [t for t, r in res.items() if not r["pass"]]
    print(f"  c3check {len(res) - len(bad)}/{len(res)} PASS")
    if bad:
        raise SystemExit(f"c3check failed: {bad}")
    return {"outputs": [str(out / "summary.json")]}


# ---------------------------------------------------------------- tables


def wall(ctx: Ctx, i: int) -> dict:
    out = {}
    for k, v in get_state(ctx, i)["steps"].items():
        if v.get("wall_s") is not None:
            out[k] = round(v["wall_s"], 1)
    return out


def lc_row(ctx: Ctx, i: int) -> dict:
    d = ctx.root / f"iter{i}"
    m = jload(d / "model.json")
    row = {"iter": i, "model": m["id"], "box_id": m["box_id"], "kept": m.get("kept")}
    if m.get("box_json") and m["box_id"].startswith("ub-tr"):
        tr = jload(Path(m["box_json"]).parent / "trust_region.json")
        row["box"] = {**tr["ratios"], "eta": tr["eta"], "n_floor_hits": tr["n_floor_hits"],
                      "n_clip_hits": tr["n_clip_hits"]}
    else:
        row["box"] = {"delta_trans_gm": 1.0, "delta_rot_gm": 1.0, "box_trans_gm": 1.0,
                      "box_rot_gm": 1.0, "min_delta_over_half_ins": 1.0, "eta": None}
    if (d / "collect/summary.json").is_file():
        c = jload(d / "collect/summary.json")
        w = [x for x in c["wall_s"] if x]
        row["collect"] = {"n": c["n"], "strict": c["strict"], "engaged": c["engaged"],
                          "aborts": c["aborts"], "wall_mean_s": float(np.mean(w)) if w else None}
    if (d / "eval.json").is_file():
        ev = jload(d / "eval.json")
        row["eval"] = {s: ev[s]["summary"] for s in ("seat6", "yaw12") if s in ev}
        row["eval_dirs"] = {s: ev[s]["dir"] for s in ("seat6", "yaw12") if s in ev}
        row["eval_reused_from"] = ev.get("reused_from_iter")
    if (d / "buffer.json").is_file():
        row["buffer_size"] = jload(d / "buffer.json")["n"]
    if (d / "finetune.json").is_file():
        f = jload(d / "finetune.json")
        row["finetune"] = {"decision": f["decision"], "path": f.get("path"),
                           "gate_decision": f.get("gate_decision", f["decision"]),
                           "candidate_id": f.get("candidate_id"), "chosen": f.get("chosen"),
                           "attempts": [{"name": a["name"], "fresh_before": a["fresh_before"],
                                         "initial": a["initial"],
                                         "candidates": [(c["step"], c["ins_1"], c["ins_h7"],
                                                         c["hard_ok"], c["soft_ok"])
                                                        for c in a["candidates"]],
                                         "chosen": a["chosen"],
                                         "h7_gate_ok": a["h7_gate_ok"], "train_s": a["train_s"]}
                                        for a in f.get("attempts", [])]}
    if (d / "trust_region.json").is_file():
        tr = jload(d / "trust_region.json")
        row["next_box"] = {**tr["ratios"], "eta": tr["eta"]}
    row["wall_s"] = wall(ctx, i)
    return row


def score(row):
    e = row.get("eval", {})
    s, y = e.get("seat6", {}), e.get("yaw12", {})
    return (s.get("strict", 0) + y.get("strict", 0), s.get("engaged", 0) + y.get("engaged", 0),
            -float(np.mean([x["rmse"] for x in (s, y) if x.get("n")] or [np.inf])))


def verdict(rows) -> str:
    """Plan §6 stop rules (printed only)."""
    last = rows[-1]
    e = last.get("eval", {})
    unsafe = sum(e.get(s, {}).get("n_unsafe", 0) for s in ("seat6", "yaw12"))
    aborts = sum(e.get(s, {}).get("aborts", 0) for s in ("seat6", "yaw12"))
    if unsafe >= 2 or aborts >= 2:
        return f"STOP (safety): {unsafe} unsafe cells, {aborts} aborts in iteration {last['iter']}"
    ft = last.get("finetune", {})
    if ft.get("decision") == "failed":
        return "STOP (model gates): no candidate passes insertion one-step / rho(A) / |B|_F"
    ev_rows = [r for r in rows if r.get("eval")]
    if len(ev_rows) >= 3:
        best_before = max(score(r) for r in ev_rows[:-2])
        if all(score(r) <= best_before for r in ev_rows[-2:]):
            return "STOP (plateau): score S not above the best for 2 iterations"
    if last["iter"] >= MAX_FT:
        return f"STOP (budget): {MAX_FT} fine-tunes done"
    return "continue"


def fmt_eval(s):
    if not s or not s.get("n"):
        return "-"
    safe = "ok" if not s["n_unsafe"] and not s["aborts"] else f"{s['n_unsafe']}u/{s['aborts']}a"
    return f"{s['strict']}/{s['n']} ({s['engaged']}) {s['rmse']:.1f} {safe}"


def f3(x, spec=".3f"):
    return "-" if x is None else format(x, spec)


def st_tables(ctx: Ctx, dry: bool):
    if dry:
        print(f"  would write {rel(ctx.root)}/learning_curve.{{json,md,png}} row {ctx.i}")
        return {}
    rows = []
    for j in range(ctx.i + 2):
        if (ctx.root / f"iter{j}" / "model.json").is_file() and (
                (ctx.root / f"iter{j}" / "eval.json").is_file() or
                (ctx.root / f"iter{j}" / "collect/summary.json").is_file()):
            rows.append(lc_row(ctx, j))
    v = verdict(rows)
    lc = {"rows": rows, "verdict": v, "updated": now(),
          "comparators": {"V4a": {"seat6": "0/6, RMSE 13.1", "yaw12": "0/12, 13.6"},
                          "V2": {"seat6": "4/6 (6/6 engaged), 6.2"},
                          "V7a": {"seat6": "2/6 (3/6), 10.0", "yaw12": "3/12, 23.5"}}}
    jdump(ctx.root / "learning_curve.json", lc)
    md = ["# V8 learning curve", "",
          ("Closed loop per model M_i (seat6 / yaw12: strict/n (engaged) mean belt RMSE mm, "
           "safety ok or unsafe-cells u / aborts a). Comparators: V4a seat6 0/6 13.1, yaw12 "
           "0/12 13.6; V2 seat6 4/6 6.2; V7a 2/6, 3/12."), "",
          ("| iter | model | box | box vol trans / rot (min Δ/half) | buffer | collection "
           "strict / engaged / aborts (n) | seat6 | yaw12 | wall per attempt s |"),
          "|---" * 9 + "|"]
    for r in rows:
        b, c = r["box"], r.get("collect")
        md.append(
            f"| {r['iter']} | {r['model']}{' (kept)' if r.get('kept') else ''} | {r['box_id']} | "
            f"{b['delta_trans_gm']:.2f} / {b['delta_rot_gm']:.2f} "
            f"({b['min_delta_over_half_ins']:.2f}) | {r.get('buffer_size', '-')} | "
            + (f"{c['strict']} / {c['engaged']} / {c['aborts']} ({c['n']})" if c else "-")
            + f" | {fmt_eval(r.get('eval', {}).get('seat6'))}"
            + (" (09-29)" if r.get("eval_reused_from") == 0 else "")
            + f" | {fmt_eval(r.get('eval', {}).get('yaw12'))} | "
            + (f"{c['wall_mean_s']:.0f}" if c and c["wall_mean_s"] else "-") + " |")
    md += ["", ("Fine-tune of iteration i (θ_i → M_{i+1}): fresh = this iteration's rollouts "
                "(one-step decoded mm, before → after); candidates `step:ins1/insh7` (* = "
                "passes the one-step gates); chosen by min insertion h7 + held-out "
                f"val-episodes h7; gate: insertion h7 ≤ {H7_FACTOR * V4A_INS_H7:.3f} mm."), "",
           ("| iter | attempt | fresh before → after | candidates | chosen | ins 1-step / h7 | "
            "val-ep 1-step / h7 | val dec | buffer 1-step / h7 | ρ(A) | ‖B‖_F | decision | "
            "next box η |"), "|---" * 13 + "|"]
    for r in rows:
        f = r.get("finetune")
        for a in (f or {}).get("attempts", []):
            c, init = a["chosen"], a["initial"]
            cs = " ".join(f"{st}:{i1:.3f}/{h7:.2f}{'*' if ok else ''}"
                          for st, i1, h7, ok, _ in a["candidates"])
            gate = "h7 ok" if a["h7_gate_ok"] else "h7 FAIL"
            if a is f["attempts"][-1]:
                gate += f" → {f['gate_decision']}"
                if f["gate_decision"] != f["decision"]:
                    gate += " (smoke: exported anyway)"
            md.append(
                f"| {r['iter']} | {a['name']} | {f3(a['fresh_before'])} → "
                f"{f3(c and c['fresh_1'])} | {cs} | {c['step'] if c else '-'} | "
                f"{f3(c and c['ins_1'])} / {f3(c and c['ins_h7'], '.2f')} (θ_i "
                f"{init['ins_1']:.3f} / {init['ins_h7']:.2f}) | "
                f"{f3(c and c['valep_1'])} / {f3(c and c['valep_h7'], '.2f')} (θ_i "
                f"{init['valep_1']:.3f} / {init['valep_h7']:.2f}) | {f3(c and c['val_dec'])} | "
                f"{f3(c and c['buf_1'])} / {f3(c and c['buf_h7'], '.2f')} (θ_i "
                f"{init['buf_1']:.3f} / {f3(init['buf_h7'], '.2f')}) | "
                f"{f3(c and c['rho_A'], '.4f')} | {f3(c and c['B_fro'], '.1f')} | {gate} | "
                f"{r.get('next_box', {}).get('eta', '-')} |")
    md += ["", "Wall per step (s): " + "; ".join(
        f"iter {r['iter']}: " + ", ".join(f"{k} {v:.0f}" for k, v in r["wall_s"].items())
        for r in rows if r["wall_s"]), "", f"**Stop rules (plan §6):** {v}", ""]
    (ctx.root / "learning_curve.md").write_text("\n".join(md) + "\n")
    plot(ctx.root / "learning_curve.png", rows)
    print("\n".join(md))
    return {"outputs": [str(ctx.root / "learning_curve.md")]}


def plot(path: Path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = [r["iter"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    def frac(r, k):
        s = r.get("eval", {}).get(k) if k != "collect" else r.get("collect")
        return s["strict"] / s["n"] if s and s.get("n") else np.nan

    for k, mk in (("seat6", "o-"), ("yaw12", "s-"), ("collect", "^-")):
        ax[0, 0].plot(it, [frac(r, k) for r in rows], mk, label=k)
    ax[0, 0].set(title="strict success fraction", xlabel="iteration", ylim=(-0.05, 1.05))
    ax[0, 0].legend()

    def ft(r, key, which="chosen"):
        f = r.get("finetune") or {}
        c = f.get("chosen")
        if which == "before":
            return f["attempts"][0]["fresh_before"] if f.get("attempts") else np.nan
        return c[key] if c and c.get(key) is not None else np.nan

    ax[0, 1].plot(it, [ft(r, None, "before") for r in rows], "o-", label="fresh before")
    ax[0, 1].plot(it, [ft(r, "fresh_1") for r in rows], "s-", label="fresh after")
    ax[0, 1].plot(it, [ft(r, "ins_h7") for r in rows], "^-", label="insertion h7")
    ax[0, 1].axhline(H7_FACTOR * V4A_INS_H7, ls=":", c="k", lw=0.8)
    ax[0, 1].set(title="fine-tune (mm)", xlabel="iteration")
    ax[0, 1].legend()
    ax[1, 0].plot(it, [r["box"]["delta_trans_gm"] for r in rows], "o-", label="trans")
    ax[1, 0].plot(it, [r["box"]["delta_rot_gm"] for r in rows], "s-", label="rot")
    ax[1, 0].set(title="box Δ / half(ub-ins), geometric mean", xlabel="iteration")
    ax[1, 0].legend()
    ax[1, 1].plot(it, [ft(r, "rho_A") for r in rows], "o-", label="ρ(A)")
    ax[1, 1].axhline(RHO_MAX, ls=":", c="k", lw=0.8)
    ax2 = ax[1, 1].twinx()
    ax2.plot(it, [ft(r, "B_fro") for r in rows], "s-", c="C1", label="‖B‖_F")
    ax[1, 1].set(title="ρ(A) (left), ‖B‖_F (right)", xlabel="iteration")
    for a in ax.flat:
        a.set_xticks(it)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- registry


def md_append_row(lines, header_prefix, row):
    """Insert `row` after the last row of the table whose header starts with header_prefix."""
    h = next(n for n, ln in enumerate(lines) if ln.startswith(header_prefix))
    n = h + 2
    while n < len(lines) and lines[n].startswith("|"):
        n += 1
    lines.insert(n, row)


def register_model(ctx: Ctx, nm: dict, ft: dict, log) -> bool:
    import yaml

    ydoc = REPO / "docs/models.yaml"
    text = ydoc.read_text()
    if any(m["id"] == nm["id"] for m in yaml.safe_load(text)["models"]):
        print(f"  {nm['id']} already registered", file=log)
        return False
    c = ft["chosen"]
    ck = Path(nm["checkpoint"])
    entry = {"id": nm["id"], "tag": f"mix-rmse-fixB (ft-onpolicy {ctx.i}, tr{ctx.i + 1})",
             "run_dir": "~/git/lcs_learning/outputs/sim_belt_v8_20261001",
             "config": None, "checkpoint": str(ck).replace(str(HOME), "~"),
             "step": c["step"], "exports": [nm["deploy_dir"].replace(str(HOME), "~")],
             "date": "2026-10-01", "parent": ft["from"],
             "change": f"{ft['from']} LCS fine-tuned on {nm['buffer_size']} on-policy rollouts "
                       f"+ V4a replay (AE frozen), {ft['path']}, step {c['step']}; box "
                       f"ub-tr{ctx.i + 1}",
             "u_bounds": f"ub-tr{ctx.i + 1}", "status": "candidate",
             "status_note": "V8 on-policy iteration model",
             "results": {"offline": {
                 "insertion_one_step_h7_mm": f"{c['ins_1']:.3f} / {c['ins_h7']:.2f}",
                 "rho_A": round(c["rho_A"], 4), "B_fro": round(c["B_fro"], 2),
                 "source": f"data/lcs/onpolicy/v8/iter{ctx.i}/finetune.json"},
                 "closed_loop": None}}
    block = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=1000)
    assert text.count("\nevals:\n") == 1
    text = text.replace("\nevals:\n", "\n" + block + "evals:\n")
    ydoc.write_text(text)
    yaml.safe_load(ydoc.read_text())
    mdp = REPO / "docs/models.md"
    lines = mdp.read_text().splitlines()
    md_append_row(lines, "| ID | tag | old names", f"| {nm['id']} | `{entry['tag']}` | - | "
                  f"candidate | {ft['from']} | on-policy fine-tune iteration {ctx.i} "
                  f"({ft['path']}, step {c['step']}), box ub-tr{ctx.i + 1} |")
    md_append_row(lines, "| ID | run folder", f"| {nm['id']} | `sim_belt_v8_20261001` | "
                  f"`finetune_lcs_latent.py` | `iter{ctx.i}/{Path(ck).parent.name}` | - | "
                  f"step {c['step']} | `{Path(nm['deploy_dir']).name}` | ub-tr{ctx.i + 1} |")
    mdp.write_text("\n".join(lines) + "\n")
    print(f"  registered model {nm['id']}", file=log)
    return True


def register_evals(ctx: Ctx, model: dict, ev: dict, log, note="") -> list[str]:
    import yaml

    ydoc = REPO / "docs/models.yaml"
    have = {e["path"] for e in yaml.safe_load(ydoc.read_text())["evals"]}
    added = []
    mdp = REPO / "docs/models.md"
    lines = mdp.read_text().splitlines()
    text = ydoc.read_text()
    for s, e in ev.items():
        if not isinstance(e, dict) or e.get("reused") or e["dir"] in have:
            continue
        n = e["summary"]["n"]
        label = f"{model['id']} · {s} · Qw · {model['box_id']}"
        entry = {"label": label, "model": model["id"], "suite": s, "cost": "Qw",
                 "limits": model["box_id"], "path": e["dir"], "date": "2026-10-01",
                 "log_day": "2026-10-01", "n_cells": n, "n_episodes": n, "table": e["table"],
                 "note": note or f"V8 iteration {ctx.i}: {e['summary']['strict']}/{n} strict, "
                                 f"RMSE {e['summary']['rmse']:.1f}"}
        text = text.rstrip("\n") + "\n" + yaml.safe_dump(
            [entry], sort_keys=False, allow_unicode=True, width=1000)
        md_append_row(lines, "| label | path | date", f"| {label} | `mpc_eval/"
                      f"{Path(e['dir']).name}` | 10-01 | {n} / {n} | `…/table.md` | "
                      f"{entry['note']} |")
        added.append(label)
    if added and model["id"].startswith("V8") and {"seat6", "yaw12"} <= set(ev):
        o, se, ye = model["offline"], ev["seat6"]["summary"], ev["yaw12"]["summary"]
        md_append_row(lines, "| ID | insertion 1-step / h7", (
            f"| {model['id']} | {o['ins_1']:.3f} / {o['ins_h7']:.2f} | {se['strict']}/{se['n']} "
            f"({se['engaged']}/{se['n']} engaged), {se['rmse']:.1f} ({model['box_id']}) | "
            f"{ye['strict']}/{ye['n']}, {ye['rmse']:.1f} | - | V8 on-policy, iteration "
            f"{ctx.i} |"))
    if added:
        ydoc.write_text(text)
        yaml.safe_load(ydoc.read_text())
        mdp.write_text("\n".join(lines) + "\n")
    print(f"  registered evals {added}", file=log)
    return added


def st_register(ctx: Ctx, dry: bool):
    if ctx.smoke:
        print("  register skipped (smoke)")
        return {"status": "skipped"}
    if dry:
        print("  would append the new model row (if accepted) and this iteration's eval rows to "
              "docs/models.{md,yaml}")
        return {}
    with open(ctx.logs / "register.log", "a") as log:
        ev = jload(ctx.it / "eval.json")
        register_evals(ctx, load_model(ctx, ctx.i), ev, log)
        if accepted(ctx):
            register_model(ctx, load_model(ctx, ctx.i + 1), jload(ctx.it / "finetune.json"), log)
    return {}


# ---------------------------------------------------------------- control (plan Q6)


def st_control(ctx: Ctx, dry: bool):
    """V4a LCS + box_1 on seat6 (once, after iteration 0)."""
    d = ctx.root / "control"
    box = ctx.root / "iter0" / "box_1.json"
    dep = ctx.out / "deploy_v4a_tr1"
    m = {"id": "V4a", "tag": "v4a_tr1", "checkpoint": str(V4A_CKPT), "deploy_dir": str(dep),
         "box_id": "ub-tr1", "box_json": str(box), "goals_dir": rel(GOALS / "v4a_tr1")}
    cmd = [PY_LCS, "scripts/export_learned_lcs_deploy.py", "--checkpoint", V4A_CKPT,
           "--device", "cpu", "--from-deploy", V4A_DEPLOY, "--u-bounds-json", box,
           "--out-dir", dep]
    if dry:
        print(f"  control: {' '.join(map(str, cmd))}; assets v4a_tr1 (T0); seat6 on "
              f"{ctx.url_seat} -> {rel(eval_dir(ctx, m, 'seat6'))}")
        return {}
    log = d / "control.log"
    d.mkdir(parents=True, exist_ok=True)
    if not (dep / "deploy.npz").is_file() and run(cmd, log, cwd=LCS):
        raise SystemExit("control export failed")
    if run([sys.executable, "scripts/checks/check_latent_encoder.py", "--deploy",
            dep / "deploy.npz"], log):
        raise SystemExit("control encoder check failed")
    if not (d / "assets.json").is_file():
        with open(log, "a") as f:
            m["assets"] = build_assets(m, T0, "v4a_tr1", f)
        jdump(d / "assets.json", m["assets"])
    m["assets"] = jload(d / "assets.json")
    res = c3check(m["assets"], d / "c3check")
    jdump(d / "c3check/summary.json", res)
    if not all(r["pass"] for r in res.values()):
        raise SystemExit("control c3check failed")
    jdump(d / "model.json", m)
    ev = {"seat6": run_suite(ctx, m, "seat6", ctx.url_seat, log)}
    jdump(d / "eval.json", ev)
    if not ctx.smoke:
        with open(log, "a") as f:
            register_evals(ctx, m, ev, f, note="box control: V4a LCS with box_1 (plan Q6)")
    print(f"  control seat6: {fmt_eval(ev['seat6']['summary'])}")
    return {"outputs": [str(d / "eval.json")]}


# ---------------------------------------------------------------- main

FUNCS = {"collect": st_collect, "eval": st_eval, "encode": st_encode, "trust": st_trust,
         "finetune": st_finetune, "export": st_export, "assets": st_assets,
         "c3check": st_c3check, "tables": st_tables, "register": st_register,
         "control": st_control}


def do_step(ctx: Ctx, step: str, dry: bool):
    if dry:
        print(f"[{step}]")
        FUNCS[step](ctx, True)
        return
    if ctx.a.force == step:
        prev = get_state(ctx, ctx.i)["steps"].get(step, {})
        move_aside(prev.get("outputs", []))
        set_step(ctx, ctx.i, step, status="forced")
    elif step_done(ctx, ctx.i, step):
        print(f"[{step}] done ({get_state(ctx, ctx.i)['steps'][step]['status']}); skipped")
        return
    print(f"[{step}] start {now()}", flush=True)
    t0 = time.time()
    set_step(ctx, ctx.i, step, status="running", started=now(), pids=[os.getpid()])
    try:
        r = FUNCS[step](ctx, False) or {}
    except BaseException as e:
        set_step(ctx, ctx.i, step, status="failed", finished=now(), wall_s=time.time() - t0,
                 error=str(e))
        raise
    set_step(ctx, ctx.i, step, status=r.get("status", "done"), finished=now(),
             wall_s=time.time() - t0, outputs=r.get("outputs", []))
    print(f"[{step}] {r.get('status', 'done')} in {time.time() - t0:.0f} s", flush=True)


def child_argv(a, step):
    argv = [sys.executable, str(Path(__file__).resolve()), "--iter", str(a.iter), "--step", step]
    for k in ("root", "attempts", "steps", "concurrency", "eval_cells", "seat_port", "yaw_port",
              "collect_port"):
        v = getattr(a, k)
        if v is not None:
            argv += [f"--{k.replace('_', '-')}", str(v)]
    if a.smoke:
        argv.append("--smoke")
    if a.force == "eval":
        argv += ["--force", "eval"]
    return argv


def run_all(ctx: Ctx, dry: bool):
    """collect (+ eval beside it at --concurrency >= 2) -> GPU/CPU steps (+ eval) -> tables."""
    early = ctx.a.concurrency >= 2 and ctx.url_collect not in (ctx.url_seat, ctx.url_yaw)
    child = None
    if dry:
        print(f"# iteration {ctx.i}: collect on {ctx.url_collect}; eval seat {ctx.url_seat} / yaw "
              f"{ctx.url_yaw} ({'beside' if early else 'after'} collect); root "
              f"{rel(ctx.root)}; lcs outputs {ctx.out}")

    def start_eval():
        nonlocal child
        if dry:
            do_step(ctx, "eval", dry)
        elif not step_done(ctx, ctx.i, "eval"):
            load_model(ctx, ctx.i)
            ctx.logs.mkdir(parents=True, exist_ok=True)
            with open(ctx.logs / "eval_child.log", "a") as f:
                child = subprocess.Popen(child_argv(ctx.a, "eval"), cwd=REPO, stdout=f,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            print(f"[eval] child pid {child.pid} (log {rel(ctx.logs / 'eval_child.log')})")
        else:
            print("[eval] done; skipped")

    if early:
        start_eval()
    do_step(ctx, "collect", dry)
    if not early:
        start_eval()
    if ctx.i < MAX_FT:
        for s in ("encode", "trust", "finetune", "export", "assets", "c3check"):
            do_step(ctx, s, dry)
    if child is not None:
        rc = child.wait()
        if rc:
            raise SystemExit(f"eval child exit {rc}")
    for s in ("tables", "register"):
        do_step(ctx, s, dry)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[1:]))
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--step", choices=STEPS + ["all", "control"], required=True)
    ap.add_argument("--root", default=None, help="default data/lcs/onpolicy/v8 (smoke: "
                                                 "…/v8/smoke_loop)")
    ap.add_argument("--attempts", type=int, default=None, help="12 (smoke 2)")
    ap.add_argument("--eta-base", type=float, default=1.0)
    ap.add_argument("--eta-step", type=float, default=0.25)
    ap.add_argument("--floor", type=float, default=0.2)
    ap.add_argument("--buffer-max", type=int, default=50)
    ap.add_argument("--steps", type=int, default=None, help="fine-tune steps: 3000 (smoke 300)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="1: one sim at a time; 2: eval beside collect; 3: + yaw12 on 7751")
    ap.add_argument("--eval-cells", default=None, help="comma list of run names (subset)")
    ap.add_argument("--targets", default=None, help="assets: comma list (default all 16)")
    ap.add_argument("--collect-port", type=int, default=None)
    ap.add_argument("--seat-port", type=int, default=None)
    ap.add_argument("--yaw-port", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", choices=STEPS + ["control"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-ft", type=int, default=MAX_FT, help="fine-tune budget (default 5)")
    ap.add_argument("--no-h7-gate", action="store_true", help="promote the best candidate even if ins h7 > gate")
    ap.add_argument("--box-ins", action="store_true", help="promoted models use ub-ins, not the trust box")
    a = ap.parse_args()
    globals()["MAX_FT"] = a.max_ft  # budget is read by the steps and the verdict
    globals()["NO_H7_GATE"] = a.no_h7_gate
    globals()["BOX_INS"] = a.box_ins
    ports = [p for p in (a.collect_port, a.seat_port, a.yaw_port) if p]
    if any(p in (7667, 8081, 8082) for p in ports):
        raise SystemExit("reserved port")
    ctx = Ctx(a)
    if a.step == "all":
        run_all(ctx, a.dry_run)
    elif a.step == "control":
        ctx.i = 0
        do_step(ctx, "control", a.dry_run)
    else:
        do_step(ctx, a.step, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
