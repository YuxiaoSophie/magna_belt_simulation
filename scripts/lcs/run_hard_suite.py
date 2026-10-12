#!/usr/bin/env python3
"""Closed-loop hard suite (PLAN-20261006-miles E4): run.sh generator + summary. No sim in here.

The 11 starts of V12-22 (``cmp/a1_sim/run.sh``): V12-17's 4 grasps (set2 gv_00 / gv_06, set3
gv_02 / gv_03), V12-20's UR z -30 / +15 mm, V12-21's UR grasp angles up-30 / -20 / +20 / +30,
nominal pre_place_1; 2 episodes each, 42 s, the harness flags of a1_sim (clean belt-seg cloud,
clearance, stage goals on latent waypoints, demo prior, offset 20). ``--resets suite`` (default)
= a1_sim's: the grasps restore the snapshot, then a scripted reset with V12-17's seed; the rest
2 snapshot episodes (``snapshot``: every start 2 snapshot episodes). Run dirs
``<out>/<arm>_<start>``, so V12-22's ``a1_sim`` dirs summarize as controls.

``write``: ``<out>/run.sh`` (one sim run at a time, start-major so a partial run compares arms on
the same starts; skips a complete run dir, moves a partial one aside; refuses while a socket
holds the port) + ``<out>/suite.json``. An arm's waypoints ``auto`` / ``auto:<min_frame>``: the
script first builds ``<out>/waypoints/<arm>.npz`` with latent_waypoints.py on the CPU (the loop's
rule: stride 10, window 3, p90, floor 0.5, the 20 seed demos, goal = seed episode_0000).
C3+ arms (V12-40): MPPI_YAML ``c3plus[:wp2|pp2|seat|track]`` (default wp2) writes the C++
controller's LCS + params yaml for the arm's deploy (its u bounds) and waypoints npz (not auto)
with c3plus_params.py into the worktree (``<archive>/hard_suite/<out name>/<arm>/``); a params
yaml with a learned_mpc block (inside the worktree) is used as is. Those arms run magna's C++
learned MPC (--params; no MPPI flags, no clearance) on the same starts / resets.
``summarize``: per episode GT outcome, final wrap / h, stage 0, the first frame with Franka y
>= 45 mm (time from the episode start, UR z, belt h there), the final sweep start (last upward
crossing of 45 mm; over diagnosis), Franka y max (and while UR z > 50 mm), UR z min / end, solve
p95; per arm engaged and an arm x start table -> ``hard_suite.{json,md}`` (GT report-only).

    uv run --frozen python scripts/lcs/run_hard_suite.py write --out data/lcs/realfirst/cmp/miles_e4 \\
        --port 7798 --arm A <deploy.npz> auto:20 data/lcs/realfirst/cmp/sbox_sim/mppi_box.yaml \\
        --arm B <deploy.npz> auto:20 data/lcs/realfirst/cmp/sbox_sim/mppi_box.yaml \\
        --arm C <deploy.npz> <waypoints.npz> c3plus:wp2
    bash data/lcs/realfirst/cmp/miles_e4/run.sh
    uv run --frozen python scripts/lcs/run_hard_suite.py summarize \\
        --dir data/lcs/realfirst/cmp/miles_e4 [--dir data/lcs/realfirst/cmp/a1_sim]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
STARTS = "data/lcs/start_states"
G = f"{STARTS}/grasp_variants"
# key, start-states, variants, scripted reset seed (suite mode; None = snapshot)
SUITE = [("set2_gv_00", f"{G}/set2", "gv_00", 20261008),
         ("set2_gv_06", f"{G}/set2", "gv_06", 20261009),
         ("set3_gv_02", f"{G}/set3", "gv_02", 20261007),
         ("set3_gv_03", f"{G}/set3", "gv_03", 20261006),
         ("ur_z-30", f"{STARTS}/ur_slant/ee_ur_z-30_osc.npz", "", None),
         ("ur_z+15", f"{STARTS}/ur_slant/ee_ur_z+15_osc.npz", "", None),
         *[(p, f"{STARTS}/ur_pivot/{p}_osc.npz", "", None)
           for p in ("up-30", "up-20", "up+20", "up+30")],
         ("nominal", f"{STARTS}/pre_place_1_osc.npz", "", None)]
KEYS = [k for k, *_ in SUITE]
SEED_BUFFERS = ["data/lcs/realfirst/v1_1/seed/buffer.json",
                "data/lcs/realfirst/v1_1/seed_ext/buffer.json",
                "data/lcs/realfirst/cmp/demos_ext2/buffer.json"]
GOAL_DEMO = "data/lcs/realfirst/v1_1/seed/demo/episode_0000.npz"
WP_RULE = {"stride": 10, "window": 3, "percentile": 90, "floor": 0.5}
SWEEP_MM = 45.0
C3_SOLVE = re.compile(r"\[learned-mpc\] t=\S+ stage=\d+ .*solve=([0-9.]+)ms")


def seed_demos() -> list[str]:
    return [r["episode"] for b in SEED_BUFFERS
            for r in json.loads((REPO / b).read_text())["rollouts"] if r["kind"] == "demo"]


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def url(port: int) -> str:
    return f"udpm://239.255.76.{port - 7600}:{port}?ttl=0"


def waypoints_cmd(deploy: str, out: Path, min_frame: int | None) -> list[str]:
    return ["CUDA_VISIBLE_DEVICES=", "uv", "run", "--frozen", "python",
            "scripts/lcs/latent_waypoints.py", "--deploy", deploy, "--demos", *seed_demos(),
            "--goal-demo", GOAL_DEMO, *[x for k, v in WP_RULE.items()
                                        for x in (f"--{k}", str(v))],
            *(["--min-frame", str(min_frame)] if min_frame is not None else []),
            "--out", rel(out)]


def c3plus_arm(name: str, dep: str, wp: str, cfg: str, out: Path) -> dict:
    """A C3+ arm: generated (``c3plus[:variant]``) or a given worktree params yaml."""
    import c3plus_params as c3p

    if cfg.startswith("c3plus"):
        variant = cfg.split(":", 1)[1] if ":" in cfg else "wp2"
        if wp.startswith("auto") or not (REPO / wp).is_file():
            raise SystemExit(f"arm {name}: c3plus needs an existing waypoints npz, not {wp}")
        m = c3p.build(Path(dep).resolve().parent, (REPO / wp).resolve(), None, None,
                      name.lower(), variant, f"{c3p.ARCHIVE}/hard_suite/{out.name}/{name}")
        params = m["files"]["params"]
    else:
        p = (REPO / cfg).resolve()
        if not p.is_relative_to(c3p.WORKTREE):
            raise SystemExit(f"arm {name}: C3+ params {cfg} outside {c3p.WORKTREE}")
        params, m = str(p.relative_to(c3p.WORKTREE)), None
    chk = c3p.check_params(params)
    if not chk["pass"]:
        raise SystemExit(f"arm {name}: {params}: {chk['errors']}")
    return {"name": name, "deploy": str(Path(dep).resolve()), "waypoints": wp,
            "controller": "c3plus", "params": params, "params_sha256": chk["params_sha256"],
            "c3plus": m}


def is_c3(cfg: str) -> bool:
    if cfg.startswith("c3plus"):
        return True
    p = REPO / cfg
    return p.suffix in (".yaml", ".yml") and p.is_file() and "learned_mpc" in (
        yaml.safe_load(p.read_text()) or {})


def cmd_write(a) -> int:
    out = (REPO / a.out).resolve()
    if a.port < 7780 or a.port in (7667, 8081, 8082):
        raise SystemExit(f"--port {a.port}: private ports >= 7780 only")
    names = [x[0] for x in a.arm]
    if len(set(names)) != len(names) or not all(n.replace("-", "").replace("_", "").isalnum()
                                                for n in names):
        raise SystemExit(f"arm names {names}: unique, [A-Za-z0-9_-]")
    starts = [s for s in SUITE if not a.starts or s[0] in a.starts.split(",")]
    if a.starts and len(starts) != len(a.starts.split(",")):
        raise SystemExit(f"--starts {a.starts}: keys within {KEYS}")
    arms, pre = [], []
    for name, dep, wp, cfg in a.arm:
        for p in (dep, cfg):
            if not (REPO / p).is_file() and not (p is cfg and cfg.startswith("c3plus")):
                raise SystemExit(f"arm {name}: {p} missing")
        if is_c3(cfg):
            arms.append(c3plus_arm(name, dep, wp, cfg, out))
            continue
        if wp.startswith("auto"):
            mf = int(wp.split(":", 1)[1]) if ":" in wp else None
            wpo = out / "waypoints" / f"{name}.npz"
            pre.append((wpo, waypoints_cmd(str(Path(dep).resolve()), wpo, mf)))
            wp = rel(wpo)
        elif not (REPO / wp).is_file():
            raise SystemExit(f"arm {name}: waypoints {wp} missing")
        arms.append({"name": name, "deploy": str(Path(dep).resolve()), "waypoints": wp,
                     "mppi_config": str((REPO / cfg).resolve())})
    da = a.demo_actions == "auto"
    runs = [(s, arm) for s in starts for arm in arms] if a.order == "start" else \
        [(s, arm) for arm in arms for s in starts]
    q = shlex.quote
    sh = ["#!/bin/bash",
          (f"# hard suite (run_hard_suite.py write): arms {names}, {len(starts)} starts x "
           f"{a.repeats}, {a.max_episode_s:g} s, resets {a.resets}, port {a.port}"
           + ("" if da else ", no --mppi-demo-actions")),
          "set -u", f"cd {REPO}", f"B={rel(out)}", "STATUS=$B/run.status",
          (f"if ss -lupn 2>/dev/null | grep -qE ':{a.port}[[:space:]]'; then "
           f"echo 'run.sh: a socket holds port {a.port}' >&2; exit 3; fi")]
    for wpo, c in pre:
        sh.append(f"[ -f {q(rel(wpo))} ] || {' '.join(q(x) if i else x for i, x in enumerate(c))}"
                  f" > $B/waypoints_{wpo.stem}.log 2>&1 || exit 4")
    sh += [("run() {  # $1 out, $2 start-states, $3 variants|'', $4 snapshot|scripted, $5 seed|'',"
            " $6 deploy, $7 waypoints, $8 mppi yaml"),
           ("  if [ -f $B/$1/index.json ] && grep -q '\"summary\"' $B/$1/index.json; then "
            "echo \"$1 skip (complete) $(date -Is)\" >> $STATUS; return; fi"),
           "  [ -e $B/$1 ] && mv $B/$1 $B/$1.partial$(date +%s)",
           "  echo \"$1 start $(date -Is)\" >> $STATUS",
           "  vv=(); [ -n \"$3\" ] && vv=(--variants \"$3\")",
           "  rs=(--reset \"$4\"); [ \"$4\" = scripted ] && rs+=(--reset-spec slack_mm=10 seed=$5)",
           "  uv run --frozen python scripts/lcs/eval_learned_mpc.py --mode learned \\",
           (f"    --lcm-url '{url(a.port)}' --record --max-episode-s {a.max_episode_s:g} "
            "--settle-s 0.5 \\"),
           (f"    --pace 1 --deploy $6 --demo-goals none --start-states $2 \"${{vv[@]}}\" "
            f"--repeats {a.repeats} \\"),
           "    --belt-seg --cloud-noise off --controller mppi --mppi-config $8 \\",
           "    --mppi-goal waypoints --mppi-stages all --mppi-max-duration-s 30.0 --mppi-debug \\",
           f"    --label {a.label}_$1 --out $B/$1 --mppi-clearance --mppi-clearance-ur-min-mm 3.0 \\",
           "    --mppi-clearance-franka-min-mm 2.0 \"${rs[@]}\" --mppi-waypoints $7 \\",
           (f"    --mppi-goal-mode stage{' --mppi-demo-actions auto' if da else ''} "
            "--mppi-demo-offset-frames 20 > $B/$1.log 2>&1"),
           "  echo \"$1 rc $? $(date -Is)\" >> $STATUS",
           "}"]
    if any(arm.get("controller") == "c3plus" for arm in arms):
        sh += [("run_c3() {  # C3+: $1 out, $2 start-states, $3 variants|'', $4 snapshot|"
                "scripted, $5 seed|'', $6 deploy, $7 worktree params"),
               ("  if [ -f $B/$1/index.json ] && grep -q '\"summary\"' $B/$1/index.json; then "
                "echo \"$1 skip (complete) $(date -Is)\" >> $STATUS; return; fi"),
               "  [ -e $B/$1 ] && mv $B/$1 $B/$1.partial$(date +%s)",
               "  echo \"$1 start $(date -Is)\" >> $STATUS",
               "  vv=(); [ -n \"$3\" ] && vv=(--variants \"$3\")",
               ("  rs=(--reset \"$4\"); [ \"$4\" = scripted ] && rs+=(--reset-spec slack_mm=10 "
                "seed=$5)"),
               "  uv run --frozen python scripts/lcs/eval_learned_mpc.py --mode learned \\",
               (f"    --lcm-url '{url(a.port)}' --record --max-episode-s {a.max_episode_s:g} "
                "--settle-s 0.5 \\"),
               (f"    --pace 1 --deploy $6 --demo-goals none --start-states $2 \"${{vv[@]}}\" "
                f"--repeats {a.repeats} \\"),
               "    --belt-seg --cloud-noise off --params $7 \\",
               f"    --label {a.label}_$1 --out $B/$1 \"${{rs[@]}}\" > $B/$1.log 2>&1",
               "  echo \"$1 rc $? $(date -Is)\" >> $STATUS",
               "}"]
    for (key, st, var, seed), arm in runs:
        scripted = a.resets == "suite" and seed is not None
        rs = f"{'scripted' if scripted else 'snapshot'} {seed if scripted else q('')}"
        if arm.get("controller") == "c3plus":
            sh.append(f"run_c3 {arm['name']}_{key} {st} {q(var)} {rs} {q(arm['deploy'])} "
                      f"{q(arm['params'])}")
            continue
        sh.append(f"run {arm['name']}_{key} {st} {q(var)} "
                  f"{'scripted' if scripted else 'snapshot'} {seed if scripted else q('')} "
                  f"{q(arm['deploy'])} {q(arm['waypoints'])} {q(arm['mppi_config'])}")
    sh.append("echo \"DONE $(date -Is)\" >> $STATUS")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.sh").write_text("\n".join(sh) + "\n")
    (out / "run.sh").chmod(0o755)
    (out / "suite.json").write_text(json.dumps(
        {"arms": arms, "starts": [dict(zip(("key", "start_states", "variants", "reset_seed"), s))
                                  for s in starts],
         "repeats": a.repeats, "max_episode_s": a.max_episode_s, "resets": a.resets,
         "port": a.port, "order": a.order, **({} if da else {"demo_actions": None}),
         "waypoint_rule": {**WP_RULE, "goal_demo": GOAL_DEMO, "demos": SEED_BUFFERS}},
        indent=1) + "\n")
    print(f"wrote {rel(out / 'run.sh')}: {len(runs)} runs ({len(arms)} arms x {len(starts)} "
          f"starts x {a.repeats} episodes, ~{len(runs) * a.repeats * (a.max_episode_s + 8) / 60:.0f}"
          f" min), port {a.port}")
    return 0


# ---------------------------------------------------------------- summary


def nan_or(x) -> float:
    return float("nan") if x is None else float(x)


def episode_metrics(d: Path, ep: dict) -> dict:
    with np.load(d / ep["file"], allow_pickle=True) as z:
        t = np.asarray(z["sim_time"]) - float(z["sim_time"][0])
        F, U = np.asarray(z["sim_ee_franka"])[:, :3] * 1e3, np.asarray(z["sim_ee_ur"])[:, :3] * 1e3
        h = np.asarray(z["sim_h_median_mm"], np.float64)
    fy, uz = F[:, 1], U[:, 2]
    i45 = np.flatnonzero(fy >= SWEEP_MM)
    up = np.flatnonzero((fy[1:] >= SWEEP_MM) & (fy[:-1] < SWEEP_MM))
    isw = int(up[-1] + 1) if len(up) else None  # final sweep start (over diagnosis)
    hi = uz > 50.0
    ev = ep.get("controller_events") or {}
    s0 = (ep.get("stage_ends") or [None])[0]
    m = (ep.get("mppi") or {}).get("solve_ms") or {}
    if not ep.get("mppi") and ep.get("controller_log") and (d / ep["controller_log"]).is_file():
        ms = [float(x) for x in C3_SOLVE.findall((d / ep["controller_log"]).read_text(
            errors="replace"))]  # C++ controller: ~1 Hz log samples
        m = {"p95": float(np.percentile(ms, 95))} if ms else {}
    return {"file": ep["file"], "repeat": ep.get("repeat"),
            "reset": (ep.get("reset_before") or {}).get("kind") or "snapshot",
            "outcome": ep.get("outcome"), "engaged": ep.get("outcome") == "engaged",
            "status": ep.get("status"), "wrap_deg": nan_or(ep.get("final_wrap_deg")),
            "h_end_mm": nan_or(ep.get("final_h_median_mm")), "stage0": s0,
            "t_stage0_s": nan_or(ev.get(f"stage_0_{s0}")) if s0 else float("nan"),
            "t45_s": float(t[i45[0]]) if len(i45) else float("nan"),
            "urz_at45_mm": float(uz[i45[0]]) if len(i45) else float("nan"),
            "h_at45_mm": float(h[i45[0]]) if len(i45) else float("nan"),
            "t_sweep_s": float(t[isw]) if isw is not None else float("nan"),
            "urz_at_sweep_mm": float(uz[isw]) if isw is not None else float("nan"),
            "h_at_sweep_mm": float(h[isw]) if isw is not None else float("nan"),
            "fy_max_mm": float(fy.max()),
            "fy_max_urz_gt50_mm": float(fy[hi].max()) if hi.any() else float("nan"),
            "urz_min_mm": float(uz.min()), "urz_end_mm": float(uz[-1]),
            "solve_p95_ms": nan_or(m.get("p95")), "board_contact": ep.get("board_contact")}


def complete(idx: dict) -> bool:
    return "summary" in idx


def collect(dirs: list[Path]) -> list[dict]:
    rows = []
    for root in dirs:
        tag = root.name if len(dirs) > 1 else None
        for d in sorted(p for p in root.iterdir() if (p / "index.json").is_file()):
            key = next((k for k in KEYS if d.name.endswith("_" + k)), None)
            if key is None:
                continue
            idx = json.loads((d / "index.json").read_text())
            arm = d.name[:-len(key) - 1]
            for ep in idx["episodes"]:
                if ep.get("file") is None:
                    rows.append({"arm": f"{tag}/{arm}" if tag else arm, "start": key,
                                 "file": None, "outcome": ep.get("outcome") or "no episode",
                                 "engaged": False, "dir": rel(d), "complete": complete(idx)})
                    continue
                rows.append({"arm": f"{tag}/{arm}" if tag else arm, "start": key, "dir": rel(d),
                             "complete": complete(idx), **episode_metrics(d, ep)})
    return rows


def f(x, s=".0f") -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, s)


def med(rows, k) -> float:
    v = [r[k] for r in rows if isinstance(r.get(k), float) and math.isfinite(r[k])]
    return float(np.median(v)) if v else float("nan")


def summary_md(rows: list[dict]) -> str:
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    starts = [k for k in KEYS if any(r["start"] == k for r in rows)]
    md = ["# Hard suite (GT report-only)", "",
          ("Times from the episode's first frame. 45 mm: the first frame with Franka y >= 45 mm; "
           "sweep: the last upward crossing of 45 mm (over diagnosis); h = sim_h_median_mm; "
           "* = run incomplete."), "",
          ("| arm | engaged | outcomes | t45 s med | UR z at 45 med | h at 45 med | UR z at sweep "
           "med | h at sweep med | F y max med | solve p95 med |"), "|" + "---|" * 10]
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        oc: dict = {}
        for r in rr:
            oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
        md.append(f"| {arm} | **{sum(r['engaged'] for r in rr)}/{len(rr)}** | "
                  f"{', '.join(f'{k} {v}' for k, v in sorted(oc.items()))} | "
                  f"{f(med(rr, 't45_s'), '.1f')} | {f(med(rr, 'urz_at45_mm'))} | "
                  f"{f(med(rr, 'h_at45_mm'), '.1f')} | {f(med(rr, 'urz_at_sweep_mm'))} | "
                  f"{f(med(rr, 'h_at_sweep_mm'), '.1f')} | {f(med(rr, 'fy_max_mm'))} | "
                  f"{f(med(rr, 'solve_p95_ms'), '.1f')} |")
    md += ["", "Engaged per start:", "", "| start | " + " | ".join(arms) + " |",
           "|---|" + "---|" * len(arms)]
    for k in starts:
        cells = []
        for arm in arms:
            rr = [r for r in rows if r["arm"] == arm and r["start"] == k]
            cells.append(f"{sum(r['engaged'] for r in rr)}/{len(rr)}" if rr else "-")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md += ["", ("| arm | start | ep | reset | GT | wrap | h end | stage 0 (s) | t45 s | UR z at "
                "45 | h at 45 | t sweep s | UR z at sweep | h at sweep | F y max (UR z > 50) | UR z "
                "min / end | p95 ms |"), "|" + "---|" * 17]
    for r in rows:
        st = r["start"] + ("" if r["complete"] else "*")
        if r.get("file") is None:
            md.append(f"| {r['arm']} | {st} | - | - | {r['outcome']} |" + " - |" * 12)
            continue
        md.append(
            f"| {r['arm']} | {st} | {r['repeat']} | {r['reset']} | {r['outcome']} | "
            f"{f(r['wrap_deg'])} | {f(r['h_end_mm'], '.1f')} | {r['stage0'] or '-'} "
            f"({f(r['t_stage0_s'], '.1f')}) | {f(r['t45_s'], '.1f')} | {f(r['urz_at45_mm'])} | "
            f"{f(r['h_at45_mm'], '.1f')} | {f(r['t_sweep_s'], '.1f')} | "
            f"{f(r['urz_at_sweep_mm'])} | {f(r['h_at_sweep_mm'], '.1f')} | {f(r['fy_max_mm'])} "
            f"({f(r['fy_max_urz_gt50_mm'])}) | {f(r['urz_min_mm'])} / {f(r['urz_end_mm'])} | "
            f"{f(r['solve_p95_ms'], '.1f')} |")
    return "\n".join(md) + "\n"


def cmd_summarize(a) -> int:
    dirs = [(REPO / d).resolve() for d in a.dir]
    rows = collect(dirs)
    if not rows:
        raise SystemExit(f"no hard-suite run dirs (<arm>_<start>, start in {KEYS}) under {a.dir}")
    out = (REPO / a.out).resolve() if a.out else dirs[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "hard_suite.json").write_text(json.dumps(
        {"dirs": [rel(d) for d in dirs], "rows": rows}, indent=1, default=float) + "\n")
    md = summary_md(rows)
    (out / "hard_suite.md").write_text(md)
    print(md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write <out>/run.sh + suite.json")
    w.add_argument("--out", required=True)
    w.add_argument("--port", type=int, required=True)
    w.add_argument("--arm", nargs=4, action="append", required=True,
                   metavar=("NAME", "DEPLOY", "WAYPOINTS", "MPPI_YAML"),
                   help="WAYPOINTS: npz | auto | auto:<min_frame>; MPPI_YAML: an MPPI yaml, "
                        "c3plus[:variant] or a C3+ params yaml (C++ controller arm)")
    w.add_argument("--resets", choices=("suite", "snapshot"), default="suite")
    w.add_argument("--repeats", type=int, default=2)
    w.add_argument("--max-episode-s", type=float, default=42.0)
    w.add_argument("--order", choices=("start", "arm"), default="start")
    w.add_argument("--starts", default=None, help=f"comma subset of {','.join(KEYS)}")
    w.add_argument("--label", default="hard_suite")
    w.add_argument("--demo-actions", choices=("auto", "none"), default="auto",
                   help="none: no --mppi-demo-actions (the node gets no demo action inputs)")
    s = sub.add_parser("summarize", help="hard_suite.{json,md}")
    s.add_argument("--dir", action="append", required=True)
    s.add_argument("--out", default=None, help="default: the first --dir")
    a = ap.parse_args()
    return cmd_write(a) if a.cmd == "write" else cmd_summarize(a)


if __name__ == "__main__":
    sys.exit(main())
