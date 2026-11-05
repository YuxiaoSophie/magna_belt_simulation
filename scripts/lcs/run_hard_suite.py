#!/usr/bin/env python3
"""Closed-loop hard suite (PLAN-20261006-miles E4): run.sh generator + summary. No sim in here.

The 11 starts of V12-22 (``cmp/a1_sim/run.sh``): V12-17's 4 grasps (set2 gv_00 / gv_06, set3
gv_02 / gv_03), V12-20's UR z -30 / +15 mm, V12-21's UR grasp angles up-30 / -20 / +20 / +30,
nominal pre_place_1 (up-30 dropped by default since V12-75: its belt starts hooked on the small
pulley); 2 episodes each, 42 s, the harness flags of a1_sim (clean belt-seg cloud,
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

``--suite stage2`` (V12-95): starts = the stage-2 snapshots ``data/lcs/start_states/stage2``
(variants ``<src>_<key>_<rep>``, default src V45c, reps 0..repeats-1 = the two stage-1 repeats),
snapshot resets, one episode per variant; harness flags: Franka-only grasp test, both-pulley
labels, the post-MPC scripted release (success = both engaged after it), demo offset 0, a short
final stage (``--final-hold-s``), the stage-2 scene, the parked UR's jaw (191), 2 s start settle,
the collector's 20 mm hand-width limit, ``--franka-min-mm`` (the stage-1 rule on the stage-2
data). ``summarize --suite stage2`` [``--collector DIR``: scripted stage2_collect runs as control
rows] -> ``hard_suite.{json,md}`` with the MPC-end and after-release labels.

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
from collections import Counter
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
DROPPED = ("up-30",)  # belt hooked on the small pulley at t 0: no controller can seat it (V12-75)
DEFAULT_KEYS = [k for k in KEYS if k not in DROPPED]
SEED_BUFFERS = ["data/lcs/realfirst/v1_1/seed/buffer.json",
                "data/lcs/realfirst/v1_1/seed_ext/buffer.json",
                "data/lcs/realfirst/cmp/demos_ext2/buffer.json"]
GOAL_DEMO = "data/lcs/realfirst/v1_1/seed/demo/episode_0000.npz"
WP_RULE = {"stride": 10, "window": 3, "percentile": 90, "floor": 0.5}
SWEEP_MM = 45.0
STAGE2_STARTS = f"{STARTS}/stage2"
STAGE2_SCENE = "assets/round_belt_task/round_belt_scene_stage2.yaml"
STAGE2_SRC = "V45c"  # stage-1 arm whose end states seed the default stage-2 suite
STAGE2_JAW = 191  # the parked UR's open jaw (stage2_starts.py)
STAGE2_SETTLE_S = 2.0  # start settle: the collector's (large pulley kept 9/10 vs 6/10 at 0.5 s)
STAGE2_WIDTH_MAX_MM = 20.0  # stage2_collect --franka-width-max
STAGE2_EARLY_STOP = "4,20,12"  # --suite full: the chain's stage-2 early stop
C3_SOLVE = re.compile(r"\[learned-mpc\] t=\S+ stage=\d+ .*solve=([0-9.]+)ms")


def seed_demos() -> list[str]:
    return [r["episode"] for b in SEED_BUFFERS
            for r in json.loads((REPO / b).read_text())["rollouts"] if r["kind"] == "demo"]


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def url(port: int) -> str:
    # 239.255.76.<octet> runs out at 7855 (octet 255)
    return (f"udpm://239.255.76.{port - 7600}:{port}?ttl=0" if port <= 7855
            else f"udpm://239.255.78.{port - 7800}:{port}?ttl=0")


def port_guard(port: int) -> str:
    """run.sh line: exit 3 if a socket holds ``port``; the passive lcm_viewer.py is allowed."""
    return (f"while read -r l; do p=$(grep -oE 'pid=[0-9]+' <<<\"$l\" | head -1 | cut -d= -f2); "
            f"[ -n \"$p\" ] && tr '\\0' '\\n' < /proc/$p/cmdline 2>/dev/null | "
            f"grep -qE '(^|/)lcm_viewer\\.py$' || "
            f"{{ echo 'run.sh: a socket holds port {port}' >&2; exit 3; }}; "
            f"done < <(ss -lupnH 2>/dev/null | grep -E ':{port}[[:space:]]')")


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
    if a.suite == "stage2":
        return cmd_write_stage2(a)
    if a.suite == "full":
        return cmd_write_full(a)
    out = (REPO / a.out).resolve()
    if a.port < 7780 or a.port in (7667, 8081, 8082):
        raise SystemExit(f"--port {a.port}: private ports >= 7780 only")
    if a.scene and not (REPO / a.scene).is_file():
        raise SystemExit(f"--scene {a.scene}: no such file")
    names = [x[0] for x in a.arm]
    if len(set(names)) != len(names) or not all(n.replace("-", "").replace("_", "").isalnum()
                                                for n in names):
        raise SystemExit(f"arm names {names}: unique, [A-Za-z0-9_-]")
    keys = start_keys(a.starts)
    starts = [s for s in SUITE if s[0] in keys]
    if len(starts) != len(keys):
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
    es = "" if a.early_stop == "off" else f" --early-stop {a.early_stop}"
    ex = f" {a.eval_extra}" if a.eval_extra else ""
    runs = [(s, arm) for s in starts for arm in arms] if a.order == "start" else \
        [(s, arm) for arm in arms for s in starts]
    q = shlex.quote
    sh = ["#!/bin/bash",
          (f"# hard suite (run_hard_suite.py write): arms {names}, {len(starts)} starts x "
           f"{a.repeats}, {a.max_episode_s:g} s, resets {a.resets}, port {a.port}"
           + ("" if da else ", no --mppi-demo-actions") + (f", early stop {a.early_stop}" if es else "")
           + f", cloud noise {a.cloud_noise}" + (f", scene {a.scene}" if a.scene else "")
           + (f", eval extra {a.eval_extra}" if ex else "")),
          "set -u", f"cd {REPO}", f"B={rel(out)}", "STATUS=$B/run.status",
          *([f"export ROUND_BELT_SCENE={q(a.scene)}"] if a.scene else []),
          port_guard(a.port)]
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
            "--settle-s 0.5" + es + " \\"),
           (f"    --pace 1 --deploy $6 --demo-goals none --start-states $2 \"${{vv[@]}}\" "
            f"--repeats {a.repeats} \\"),
           f"    --belt-seg --cloud-noise {q(a.cloud_noise)} --controller mppi --mppi-config $8 \\",
           "    --mppi-goal waypoints --mppi-stages all --mppi-max-duration-s 30.0 --mppi-debug \\",
           f"    --label {a.label}_$1 --out $B/$1 --mppi-clearance --mppi-clearance-ur-min-mm 3.0 \\",
           "    --mppi-clearance-franka-min-mm 2.0 \"${rs[@]}\" --mppi-waypoints $7 \\",
           (f"    --mppi-goal-mode stage{' --mppi-demo-actions auto' if da else ''} "
            "--mppi-demo-offset-frames 20" + ex + " > $B/$1.log 2>&1"),
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
                "--settle-s 0.5" + es + " \\"),
               (f"    --pace 1 --deploy $6 --demo-goals none --start-states $2 \"${{vv[@]}}\" "
                f"--repeats {a.repeats} \\"),
               f"    --belt-seg --cloud-noise {q(a.cloud_noise)} --params $7 \\",
               f"    --label {a.label}_$1 --out $B/$1 \"${{rs[@]}}\"{ex} > $B/$1.log 2>&1",
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
         "cloud_noise": a.cloud_noise, **({"scene": a.scene} if a.scene else {}),
         **({"eval_extra": a.eval_extra} if ex else {}),
         "waypoint_rule": {**WP_RULE, "goal_demo": GOAL_DEMO, "demos": SEED_BUFFERS}},
        indent=1) + "\n")
    print(f"wrote {rel(out / 'run.sh')}: {len(runs)} runs ({len(arms)} arms x {len(starts)} "
          f"starts x {a.repeats} episodes, ~{len(runs) * a.repeats * (a.max_episode_s + 8) / 60:.0f}"
          f" min), port {a.port}")
    return 0


def stage2_starts(src: str, keys: list[str], reps: int) -> list[tuple[str, list[str]]]:
    """``[(key, [variant ids])]`` of the stage-2 start set; every variant held with a file."""
    idx = json.loads((REPO / STAGE2_STARTS / "index.json").read_text())
    by_id = {str(v["id"]): v for v in idx["variants"]}
    out = []
    for k in keys:
        vids = [f"{src}_{k}_{r}" for r in range(reps)]
        bad = [v for v in vids if v not in by_id or not by_id[v].get("held")
               or not (REPO / STAGE2_STARTS / by_id[v]["file"]).is_file()]
        if bad:
            raise SystemExit(f"stage-2 starts {bad}: not held / no file in {STAGE2_STARTS}")
        out.append((k, vids))
    return out


def cmd_write_stage2(a) -> int:
    out = (REPO / a.out).resolve()
    if a.port < 7780 or a.port in (7667, 8081, 8082):
        raise SystemExit(f"--port {a.port}: private ports >= 7780 only")
    scene = a.scene or STAGE2_SCENE
    if not (REPO / scene).is_file():
        raise SystemExit(f"--scene {scene}: no such file")
    names = [x[0] for x in a.arm]
    if len(set(names)) != len(names) or not all(n.replace("-", "").replace("_", "").isalnum()
                                                for n in names):
        raise SystemExit(f"arm names {names}: unique, [A-Za-z0-9_-]")
    if a.resets != "suite" or a.demo_actions != "none":
        raise SystemExit("--suite stage2: snapshot resets and --demo-actions none only")
    keys = start_keys(a.starts)
    if not set(keys) <= set(KEYS):
        raise SystemExit(f"--starts {a.starts}: keys within {KEYS}")
    starts = stage2_starts(a.stage2_src, keys, a.repeats)
    arms = []
    for name, dep, wp, cfg in a.arm:
        for x in (dep, wp, cfg):
            if not (REPO / x).is_file():
                raise SystemExit(f"arm {name}: {x} missing (stage 2: deploy, waypoints npz, "
                                 "MPPI yaml)")
        if is_c3(cfg):
            raise SystemExit(f"arm {name}: --suite stage2 runs MPPI arms only")
        fmin = ((yaml.safe_load((REPO / cfg).read_text()) or {}).get("clearance")
                or {}).get("franka_min_m")
        if fmin is not None and abs(fmin * 1e3 - a.franka_min_mm) > 1e-6:
            raise SystemExit(f"arm {name}: {cfg} clearance.franka_min_m {fmin * 1e3:g} mm != "
                             f"--franka-min-mm {a.franka_min_mm:g} (the yaml wins in the node)")
        arms.append({"name": name, "deploy": str(Path(dep).resolve()), "waypoints": wp,
                     "mppi_config": str((REPO / cfg).resolve())})
    es = "" if a.early_stop == "off" else f" --early-stop {a.early_stop}"
    ex = f" {a.eval_extra}" if a.eval_extra else ""
    harness = (f"--grasp-arms franka --outcome-pulley both --post-mpc stage2_release "
               f"--start-settle-s {STAGE2_SETTLE_S:g} --franka-width-max {STAGE2_WIDTH_MAX_MM:g} "
               f"--clearance-jaw-bytes {STAGE2_JAW}")
    runs = [(s, arm) for s in starts for arm in arms] if a.order == "start" else \
        [(s, arm) for arm in arms for s in starts]
    q = shlex.quote
    sh = ["#!/bin/bash",
          (f"# stage-2 hard suite (run_hard_suite.py write --suite stage2): arms {names}, "
           f"{len(starts)} starts x {a.repeats} reps ({a.stage2_src}), {a.max_episode_s:g} s MPC "
           f"+ release, port {a.port}" + (f", early stop {a.early_stop}" if es else "")
           + f", cloud noise {a.cloud_noise}, scene {scene}, Franka clearance "
             f"{a.franka_min_mm:g} mm, final hold {a.final_hold_s:g} s"
           + (f", eval extra {a.eval_extra}" if ex else "")),
          "set -u", f"cd {REPO}", f"B={rel(out)}", "STATUS=$B/run.status",
          f"export ROUND_BELT_SCENE={q(scene)}",
          port_guard(a.port),
          "run() {  # $1 out, $2 variants, $3 deploy, $4 waypoints, $5 mppi yaml",
          ("  if [ -f $B/$1/index.json ] && grep -q '\"summary\"' $B/$1/index.json; then "
           "echo \"$1 skip (complete) $(date -Is)\" >> $STATUS; return; fi"),
          "  [ -e $B/$1 ] && mv $B/$1 $B/$1.partial$(date +%s)",
          "  echo \"$1 start $(date -Is)\" >> $STATUS",
          "  uv run --frozen python scripts/lcs/eval_learned_mpc.py --mode learned \\",
          (f"    --lcm-url '{url(a.port)}' --record --max-episode-s {a.max_episode_s:g} "
           "--settle-s 0.5" + es + " \\"),
          (f"    --pace 1 --deploy $3 --demo-goals none --start-states {STAGE2_STARTS} "
           "--variants \"$2\" --repeats 1 \\"),
          f"    --belt-seg --cloud-noise {q(a.cloud_noise)} --controller mppi --mppi-config $5 \\",
          (f"    --mppi-goal waypoints --mppi-stages all --mppi-max-duration-s "
           f"{a.final_hold_s:g} --mppi-debug \\"),
          f"    --label {a.label}_$1 --out $B/$1 --mppi-clearance --mppi-clearance-ur-min-mm 3.0 \\",
          f"    --mppi-clearance-franka-min-mm {a.franka_min_mm:g} --mppi-waypoints $4 \\",
          "    --mppi-goal-mode stage --mppi-demo-offset-frames 0 \\",
          f"    {harness}{ex} > $B/$1.log 2>&1",
          "  echo \"$1 rc $? $(date -Is)\" >> $STATUS",
          "}"]
    for (key, vids), arm in runs:
        sh.append(f"run {arm['name']}_{key} {','.join(vids)} {q(arm['deploy'])} "
                  f"{q(arm['waypoints'])} {q(arm['mppi_config'])}")
    sh.append("echo \"DONE $(date -Is)\" >> $STATUS")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.sh").write_text("\n".join(sh) + "\n")
    (out / "run.sh").chmod(0o755)
    n_ep = len(runs) * a.repeats
    (out / "suite.json").write_text(json.dumps(
        {"suite": "stage2", "arms": arms,
         "starts": [{"key": k, "start_states": STAGE2_STARTS, "variants": v} for k, v in starts],
         "stage2_src": a.stage2_src, "repeats": a.repeats, "max_episode_s": a.max_episode_s,
         "resets": "snapshot", "port": a.port, "order": a.order, "demo_actions": None,
         "early_stop": a.early_stop, "cloud_noise": a.cloud_noise, "scene": scene,
         "franka_min_mm": a.franka_min_mm, "final_hold_s": a.final_hold_s,
         "harness": harness, **({"eval_extra": a.eval_extra} if ex else {})},
        indent=1) + "\n")
    print(f"wrote {rel(out / 'run.sh')}: {len(runs)} runs ({len(arms)} arms x {len(starts)} "
          f"starts x {a.repeats} reps = {n_ep} episodes, ~{n_ep * (a.max_episode_s + 20) / 60:.0f}"
          f" min), port {a.port}")
    return 0


def cmd_write_full(a) -> int:
    """``--suite full``: stage-1 starts / resets in the stage-2 scene; per episode the harness
    chain stage 1 -> UR exit -> stage 2 (``--arm2``) -> release."""
    out = (REPO / a.out).resolve()
    if a.port < 7780 or a.port in (7667, 8081, 8082):
        raise SystemExit(f"--port {a.port}: private ports >= 7780 only")
    scene = a.scene or STAGE2_SCENE
    if not (REPO / scene).is_file():
        raise SystemExit(f"--scene {scene}: no such file")
    names = [x[0] for x in a.arm]
    if len(set(names)) != len(names) or not all(n.replace("-", "").replace("_", "").isalnum()
                                                for n in names):
        raise SystemExit(f"arm names {names}: unique, [A-Za-z0-9_-]")
    s2 = {x[0]: x[1:] for x in a.arm2 or []}
    if sorted(x[0] for x in a.arm2 or []) != sorted(names):
        raise SystemExit("--suite full: one --arm2 NAME DEPLOY WAYPOINTS MPPI_YAML (stage 2) per "
                         f"--arm name {names}")
    if a.demo_actions != "none":
        raise SystemExit("--suite full: --demo-actions none only")
    keys = start_keys(a.starts)
    starts = [s for s in SUITE if s[0] in keys]
    if len(starts) != len(keys):
        raise SystemExit(f"--starts {a.starts}: keys within {KEYS}")
    arms = []
    for name, dep, wp, cfg in a.arm:
        for x in (dep, wp, cfg, *s2[name]):
            if not (REPO / x).is_file():
                raise SystemExit(f"arm {name}: {x} missing (full: deploy, waypoints npz, MPPI "
                                 "yaml per stage)")
        if is_c3(cfg) or is_c3(s2[name][2]):
            raise SystemExit(f"arm {name}: --suite full runs MPPI arms only")
        d2, w2, c2 = s2[name]
        arms.append({"name": name, "deploy": str(Path(dep).resolve()), "waypoints": wp,
                     "mppi_config": str((REPO / cfg).resolve()),
                     "stage2": {"deploy": str(Path(d2).resolve()), "waypoints": w2,
                                "mppi_config": str((REPO / c2).resolve())}})
    es = "" if a.early_stop == "off" else f" --early-stop {a.early_stop}"
    ex = f" {a.eval_extra}" if a.eval_extra else ""
    chain = ("--outcome-pulley both --post-mpc stage2_release --stage2-chain --stage2-deploy $9 "
             "--stage2-waypoints ${10} --stage2-mppi-config ${11} --stage2-max-episode-s "
             f"{a.stage2_max_episode_s:g} --stage2-early-stop {STAGE2_EARLY_STOP} "
             f"--franka-width-max {STAGE2_WIDTH_MAX_MM:g} --stage2-mppi-max-duration-s "
             f"{a.final_hold_s:g}")
    runs = [(s, arm) for s in starts for arm in arms] if a.order == "start" else \
        [(s, arm) for arm in arms for s in starts]
    q = shlex.quote
    sh = ["#!/bin/bash",
          (f"# full-run hard suite (run_hard_suite.py write --suite full): arms {names}, "
           f"{len(starts)} starts x {a.repeats}, stage 1 {a.max_episode_s:g} s"
           + (f" (early stop {a.early_stop})" if es else "") + " -> UR exit -> stage 2 "
           f"{a.stage2_max_episode_s:g} s (early stop {STAGE2_EARLY_STOP}, final hold "
           f"{a.final_hold_s:g} s) -> release, resets {a.resets}, port {a.port}, cloud noise "
           f"{a.cloud_noise}, scene {scene}" + (f", eval extra {a.eval_extra}" if ex else "")),
          "set -u", f"cd {REPO}", f"B={rel(out)}", "STATUS=$B/run.status",
          f"export ROUND_BELT_SCENE={q(scene)}",
          port_guard(a.port),
          ("run() {  # $1 out, $2 start-states, $3 variants|'', $4 snapshot|scripted, $5 seed|'',"
           " $6 deploy, $7 waypoints, $8 mppi yaml, $9 / ${10} / ${11} the stage-2 ones"),
          ("  if [ -f $B/$1/index.json ] && grep -q '\"summary\"' $B/$1/index.json; then "
           "echo \"$1 skip (complete) $(date -Is)\" >> $STATUS; return; fi"),
          "  [ -e $B/$1 ] && mv $B/$1 $B/$1.partial$(date +%s)",
          "  echo \"$1 start $(date -Is)\" >> $STATUS",
          "  vv=(); [ -n \"$3\" ] && vv=(--variants \"$3\")",
          "  rs=(--reset \"$4\"); [ \"$4\" = scripted ] && rs+=(--reset-spec slack_mm=10 seed=$5)",
          "  uv run --frozen python scripts/lcs/eval_learned_mpc.py --mode learned \\",
          (f"    --lcm-url '{url(a.port)}' --record --max-episode-s {a.max_episode_s:g} "
           "--settle-s 0.5" + es + " \\"),
          (f"    --pace 1 --deploy $6 --demo-goals none --start-states $2 \"${{vv[@]}}\" "
           f"--repeats {a.repeats} \\"),
          f"    --belt-seg --cloud-noise {q(a.cloud_noise)} --controller mppi --mppi-config $8 \\",
          "    --mppi-goal waypoints --mppi-stages all --mppi-max-duration-s 30.0 --mppi-debug \\",
          f"    --label {a.label}_$1 --out $B/$1 --mppi-clearance --mppi-clearance-ur-min-mm 3.0 \\",
          "    --mppi-clearance-franka-min-mm 2.0 \"${rs[@]}\" --mppi-waypoints $7 \\",
          "    --mppi-goal-mode stage --mppi-demo-offset-frames 20 \\",
          f"    {chain}{ex} > $B/$1.log 2>&1",
          "  echo \"$1 rc $? $(date -Is)\" >> $STATUS",
          "}"]
    for (key, st, var, seed), arm in runs:
        scripted = a.resets == "suite" and seed is not None
        b = arm["stage2"]
        sh.append(f"run {arm['name']}_{key} {st} {q(var)} "
                  f"{'scripted' if scripted else 'snapshot'} {seed if scripted else q('')} "
                  f"{q(arm['deploy'])} {q(arm['waypoints'])} {q(arm['mppi_config'])} "
                  f"{q(b['deploy'])} {q(b['waypoints'])} {q(b['mppi_config'])}")
    sh.append("echo \"DONE $(date -Is)\" >> $STATUS")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.sh").write_text("\n".join(sh) + "\n")
    (out / "run.sh").chmod(0o755)
    n_ep = len(runs) * a.repeats
    (out / "suite.json").write_text(json.dumps(
        {"suite": "full", "arms": arms,
         "starts": [dict(zip(("key", "start_states", "variants", "reset_seed"), s))
                    for s in starts],
         "repeats": a.repeats, "max_episode_s": a.max_episode_s, "resets": a.resets,
         "port": a.port, "order": a.order, "demo_actions": None, "early_stop": a.early_stop,
         "cloud_noise": a.cloud_noise, "scene": scene,
         "stage2_max_episode_s": a.stage2_max_episode_s, "stage2_early_stop": STAGE2_EARLY_STOP,
         "final_hold_s": a.final_hold_s, "chain": chain,
         **({"eval_extra": a.eval_extra} if ex else {})}, indent=1) + "\n")
    print(f"wrote {rel(out / 'run.sh')}: {len(runs)} runs ({len(arms)} arms x {len(starts)} "
          f"starts x {a.repeats} episodes, ~{n_ep * 2:.0f} min), port {a.port}")
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


def start_keys(arg: str | None) -> list[str]:
    """--starts: None = DEFAULT_KEYS, 'all' = KEYS, else the comma list."""
    return DEFAULT_KEYS if arg is None else KEYS if arg == "all" else arg.split(",")


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
    if a.suite == "stage2":
        return cmd_summarize_stage2(a)
    if a.suite == "full":
        return cmd_summarize_full(a)
    dirs = [(REPO / d).resolve() for d in a.dir]
    keys = start_keys(a.starts)
    rows = [r for r in collect(dirs) if r["start"] in keys]
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


# ---------------------------------------------------------------- stage-2 summary

ABORT_KINDS = (("grasp lost", "grasp_lost"), ("finger_tip", "tip_floor"),
               ("board clearance", "board"), ("controller exited", "controller_exit"),
               ("UR IK", "ur_ik"))


def mpc_end_kind(status: str | None, reason: str | None, completed: bool) -> str:
    if status == "aborted":
        return "abort:" + next((k for m, k in ABORT_KINDS if m in (reason or "")), "other")
    if status == "timeout":
        return "timeout"
    if (reason or "").startswith("early stop"):
        return "early_stop:" + reason.split()[2]  # "early stop: <label> held ..."
    return "completed" if completed else str(status)


def stage2_failure(r: dict) -> str:
    if r["engaged"]:
        return "success"
    if r.get("mpc_both"):
        why = "lost in the release"
    elif r.get("final_large") != "engaged":
        why = "large off"
    else:
        why = f"small {r.get('final_small')}"
    return f"{why} ({r.get('final_large')}/{r.get('final_small')})"


def rot_max_deg(q) -> float:
    """Max rotation angle (deg) of the quaternions ``q`` from ``q[0]``."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return float(np.degrees(2.0 * np.arccos(np.clip(np.abs(q @ q[0]), 0.0, 1.0))).max())


def tilt_max_deg(q) -> float:
    """Max tilt (deg) of the ``wxyz`` quaternions ``q``: acos(|z_ee . z_world|)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    c = np.abs(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))).max())


def stage2_metrics(d: Path, ep: dict, n_stages: int) -> dict:
    pm = ep.get("post_mpc") or {}
    me, fi = pm.get("mpc_end") or {}, pm.get("final") or {}
    with np.load(d / ep["file"], allow_pickle=True) as z:
        q = np.asarray(z["sim_ee_franka"])[:, 3:7]
        n = int(z["sim_post_from"]) if "sim_post_from" in z.files else len(q)
        a0 = max(int(z["sim_chain_from"][1]), 0) if "sim_chain_from" in z.files else 0  # chain
    ev = ep.get("controller_events") or {}
    stages = ep.get("stage_ends") or []
    mp = ep.get("mppi") or {}
    r = {"file": ep["file"], "variant": ep["start_state"]["id"], "status": ep.get("status"),
         "reason": ep.get("reason"), "outcome": ep.get("outcome"),
         "engaged": ep.get("outcome") == "engaged",
         "mpc_end": mpc_end_kind(ep.get("status"), ep.get("reason"), "mpc_completed" in ev),
         "final_large": fi.get("large"), "final_small": fi.get("small"),
         "final_large_wrap_deg": nan_or(fi.get("large_wrap_deg")),
         "final_small_wrap_deg": nan_or(fi.get("small_wrap_deg")),
         "final_small_h_mm": nan_or(fi.get("small_h_mm")),
         "mpc_label": me.get("label"), "mpc_large": me.get("large"), "mpc_small": me.get("small"),
         "mpc_both": me.get("both"), "mpc_held": me.get("held"),
         "mpc_large_wrap_deg": nan_or(me.get("large_wrap_deg")),
         "mpc_small_wrap_deg": nan_or(me.get("small_wrap_deg")),
         "mpc_small_h_mm": nan_or(me.get("small_h_mm")),
         "stages_done": len(stages), "stage_timeouts": stages.count("timeout"),
         "n_stages": n_stages, "rot_max_deg": rot_max_deg(q[a0:n]),
         "tilt_max_deg": tilt_max_deg(q[a0:n]),
         "tip_min_mm": nan_or(ep.get("min_franka_tip_clearance_mm")),
         "clear_min_franka_mm": nan_or((mp.get("clear_min_mm") or {}).get("franka")),
         "solve_p95_ms": nan_or((mp.get("solve_ms") or {}).get("p95")),
         "solves": mp.get("solves"), "fallbacks": mp.get("fallbacks"),
         "plans_violating": mp.get("plans_violating"),
         "tilt_plan_max_deg": nan_or((mp.get("tilt_max_deg") or {}).get("plan_franka")),
         "mpc_s": nan_or(pm.get("mpc_duration_s")), "episode_s": nan_or(ep.get("duration_s")),
         "post_ok": pm.get("ok"),
         "post_aborts": [p["abort_kind"] for p in pm.get("phases", []) if p.get("abort")],
         "recording": ep.get("recording")}
    r["failure"] = stage2_failure(r)
    return r


def collect_stage2(dirs: list[Path]) -> list[dict]:
    rows = []
    for root in dirs:
        tag = root.name if len(dirs) > 1 else None
        for d in sorted(p for p in root.iterdir() if (p / "index.json").is_file()):
            key = next((k for k in KEYS if d.name.endswith("_" + k)), None)
            if key is None:
                continue
            idx = json.loads((d / "index.json").read_text())
            if "post_mpc" not in idx:
                continue
            arm = d.name[:-len(key) - 1]
            n_st = sum(lab.startswith("mpc:") for lab in idx.get("phase_labels", []))
            for ep in idx["episodes"]:
                base = {"arm": f"{tag}/{arm}" if tag else arm, "start": key, "dir": rel(d),
                        "complete": complete(idx)}
                if ep.get("file") is None:
                    rows.append({**base, "file": None, "variant": ep["start_state"]["id"],
                                 "outcome": "no episode", "engaged": False,
                                 "failure": f"skipped: {ep.get('reason')}"})
                    continue
                rows.append({**base, **stage2_metrics(d, ep, n_st)})
    return rows


def collector_rows(root: Path, src: str | None) -> list[dict]:
    """Scripted stage2_collect episodes (``index.json``) as control rows."""
    idx = json.loads((root / "index.json").read_text())
    rows = []
    for e in idx["episodes"]:
        st = e.get("start") or {}
        if not e.get("file") or (src and st.get("arm") != src) or st.get("key") not in KEYS:
            continue
        fi, me = e["final"], e["mpc_end"]
        r = {"arm": f"scripted:{root.name}", "start": st["key"], "dir": rel(root),
             "complete": True, "file": e["file"], "variant": st["id"],
             "outcome": "engaged" if fi["both"] else fi["small"], "engaged": bool(fi["both"]),
             "mpc_end": ("walk_end:" + e["walk_end"]["why"]) if e.get("walk_end") else "completed",
             "final_large": fi["large"], "final_small": fi["small"],
             "final_large_wrap_deg": fi["large_wrap_deg"],
             "final_small_wrap_deg": fi["small_wrap_deg"], "final_small_h_mm": fi["small_h_mm"],
             "mpc_label": None, "mpc_large": me["large"], "mpc_small": me["small"],
             "mpc_both": me["both"], "mpc_held": me["held"],
             "mpc_large_wrap_deg": me["large_wrap_deg"],
             "mpc_small_wrap_deg": me["small_wrap_deg"], "mpc_small_h_mm": me["small_h_mm"],
             "episode_s": nan_or(e.get("robot_s")), "recording": e.get("recording")}
        r["failure"] = stage2_failure(r)
        rows.append(r)
    return rows


def stage2_md(rows: list[dict]) -> str:
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    starts = [k for k in KEYS if any(r["start"] == k for r in rows)]

    def cnt(rr, k):
        return sum(bool(r.get(k)) for r in rr)

    md = ["# Stage-2 hard suite (GT report-only)", "",
          ("Success = both pulleys engaged after the scripted release (hand open 40 mm, out to "
           "r 45 mm, lift to h 27 mm, 1.5 s settle). MPC end = the last MPC frame (before the "
           "release). Scripted rows: stage2_collect episodes (MPC end = walk end). Rotation = "
           "max Franka rotation from the start over the MPC frames; tilt = max Franka "
           "acos(|z_ee . z_world|) over them. * = run incomplete."), "",
          ("| arm | both after release | both at MPC end | large engaged after | small engaged "
           "after | held at MPC end | MPC ends | stages done med | rot max med | tilt max med | "
           "solve p95 med | MPC s med | episode s med |"), "|" + "---|" * 13]
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        n = len(rr)
        ends = Counter(r.get("mpc_end") for r in rr if r.get("file"))
        md.append(
            f"| {arm} | **{cnt(rr, 'engaged')}/{n}** | {cnt(rr, 'mpc_both')}/{n} | "
            f"{sum(r.get('final_large') == 'engaged' for r in rr)}/{n} | "
            f"{sum(r.get('final_small') == 'engaged' for r in rr)}/{n} | "
            f"{cnt(rr, 'mpc_held')}/{n} | "
            f"{', '.join(f'{k} {v}' for k, v in sorted(ends.items()))} | "
            f"{f(float(np.median(sd)) if (sd := [r['stages_done'] for r in rr if 'stages_done' in r]) else None)} | "
            f"{f(med(rr, 'rot_max_deg'))} | {f(med(rr, 'tilt_max_deg'))} | "
            f"{f(med(rr, 'solve_p95_ms'), '.1f')} | "
            f"{f(med(rr, 'mpc_s'), '.1f')} | {f(med(rr, 'episode_s'), '.1f')} |")
    md += ["", "Both engaged after the release, per start (MPC-end both in brackets):", "",
           "| start | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for k in starts:
        cells = []
        for arm in arms:
            rr = [r for r in rows if r["arm"] == arm and r["start"] == k]
            cells.append(f"{cnt(rr, 'engaged')}/{len(rr)} ({cnt(rr, 'mpc_both')})" if rr else "-")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md += ["", "Failure modes (final large/small label):", ""]
    for arm in arms:
        fm = Counter(r["failure"] for r in rows if r["arm"] == arm and not r["engaged"])
        md.append(f"- {arm}: " + (", ".join(f"{k} x{v}" for k, v in fm.most_common()) or "none"))
    md += ["", ("| arm | start | variant | MPC end | stages | MPC-end L/S (wrap S) | final L/S "
                "(wrap L/S, h S) | both | rot max | tilt max | tip min | p95 ms | MPC s | ep s | "
                "failure |"), "|" + "---|" * 15]
    for r in rows:
        st = r["start"] + ("" if r["complete"] else "*")
        if r.get("file") is None:
            md.append(f"| {r['arm']} | {st} | {r['variant']} | - |" + " - |" * 10
                      + f" {r['failure']} |")
            continue
        stg = f"{r['stages_done']}/{r['n_stages']} ({r['stage_timeouts']} t/o)" \
            if "stages_done" in r else "-"
        md.append(
            f"| {r['arm']} | {st} | {r['variant']} | {r['mpc_end']} | {stg} | "
            f"{r['mpc_large']}/{r['mpc_small']} ({f(r['mpc_small_wrap_deg'])}) | "
            f"{r['final_large']}/{r['final_small']} ({f(r['final_large_wrap_deg'])}/"
            f"{f(r['final_small_wrap_deg'])}, {f(r['final_small_h_mm'], '.1f')}) | "
            f"{'yes' if r['engaged'] else 'no'} | {f(r.get('rot_max_deg'))} | "
            f"{f(r.get('tilt_max_deg'))} | "
            f"{f(r.get('tip_min_mm'), '.1f')} | {f(r.get('solve_p95_ms'), '.1f')} | "
            f"{f(r.get('mpc_s'), '.1f')} | {f(r.get('episode_s'), '.1f')} | {r['failure']} |")
    return "\n".join(md) + "\n"


def cmd_summarize_stage2(a) -> int:
    dirs = [(REPO / d).resolve() for d in a.dir]
    keys = start_keys(a.starts)
    rows = [r for r in collect_stage2(dirs) if r["start"] in keys]
    if not rows:
        raise SystemExit(f"no stage-2 run dirs (<arm>_<start> with post_mpc) under {a.dir}")
    for c in a.collector or []:
        rows += [r for r in collector_rows((REPO / c).resolve(), a.stage2_src)
                 if r["start"] in keys]
    out = (REPO / a.out).resolve() if a.out else dirs[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "hard_suite.json").write_text(json.dumps(
        {"suite": "stage2", "dirs": [rel(d) for d in dirs], "collector": a.collector or [],
         "rows": rows}, indent=1, default=float) + "\n")
    md = stage2_md(rows)
    (out / "hard_suite.md").write_text(md)
    print(md)
    return 0


# ---------------------------------------------------------------- full-run summary


def full_metrics(d: Path, ep: dict, n_st2: int) -> dict:
    """One chain episode: stage-1 / exit / stage-2 flags + ``stage2_metrics`` of its stage 2."""
    ch = ep.get("chain") or {}
    s1, ux, s2 = ch.get("stage1") or {}, ch.get("ur_exit"), ch.get("stage2")
    sm1 = (ep.get("mppi") or {}).get("solve_ms") or {}
    sm2 = (ep.get("mppi_stage2") or {}).get("solve_ms") or {}
    r = {"file": ep["file"], "variant": ep["start_state"]["id"], "repeat": ep.get("repeat"),
         "outcome": ep.get("outcome"), "stage1_large": s1.get("large_label_end"),
         "stage1_end": mpc_end_kind(s1.get("status"), s1.get("reason"),
                                    "mpc_completed" in (ep.get("controller_events") or {})),
         "stage1_seated": s1.get("large_label_end") == "engaged",
         "ur_exit_ok": bool(ux) and not ux["failed"], "ur_exit_large": (ux or {}).get(
             "large_label_end"), "ur_exit_abort": (ux or {}).get("abort"),
         "stage2_ran": s2 is not None, "s1_solve_p50_ms": nan_or(sm1.get("p50")),
         "s1_solve_p95_ms": nan_or(sm1.get("p95")), "s2_solve_p50_ms": nan_or(sm2.get("p50")),
         "s2_solve_p95_ms": nan_or(sm2.get("p95")), "episode_s": nan_or(ep.get("duration_s")),
         "recording": ep.get("recording")}
    if s2 is not None:
        e2 = ep.get("events_stage2") or {}
        r.update(stage2_metrics(d, {**ep, "status": s2["status"], "reason": s2["reason"],
                                    "mppi": ep.get("mppi_stage2"),
                                    "controller_events": e2.get("controller_events"),
                                    "stage_ends": e2.get("stage_ends")}, n_st2))
    r["engaged"] = r["stage2_ran"] and ep.get("outcome") == "engaged"
    r["failure"] = ("success" if r["engaged"] else "stage1" if not r["stage1_seated"]
                    else "ur_exit" if not r["ur_exit_ok"] else f"stage2:{r['failure']}")
    return r


def collect_full(dirs: list[Path]) -> list[dict]:
    rows = []
    for root in dirs:
        tag = root.name if len(dirs) > 1 else None
        for d in sorted(p for p in root.iterdir() if (p / "index.json").is_file()):
            key = next((k for k in KEYS if d.name.endswith("_" + k)), None)
            idx = json.loads((d / "index.json").read_text()) if key else {}
            if "chain" not in idx:
                continue
            arm = d.name[:-len(key) - 1]
            n_st2 = sum(lab.startswith("s2:stage") for lab in idx.get("phase_labels", []))
            for ep in idx["episodes"]:
                base = {"arm": f"{tag}/{arm}" if tag else arm, "start": key, "dir": rel(d),
                        "complete": complete(idx)}
                if ep.get("file") is None:
                    rows.append({**base, "file": None, "variant": ep["start_state"]["id"],
                                 "outcome": "no episode", "engaged": False,
                                 "failure": f"skipped: {ep.get('reason')}"})
                    continue
                rows.append({**base, **full_metrics(d, ep, n_st2)})
    return rows


def full_md(rows: list[dict]) -> str:
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    starts = [k for k in KEYS if any(r["start"] == k for r in rows)]

    def cnt(rr, k):
        return sum(bool(r.get(k)) for r in rr)

    md = ["# Full-run hard suite (GT report-only)", "",
          ("Stage 1 (both arms, large pulley) -> scripted UR open + exit -> stage 2 (Franka, "
           "second node) -> scripted release. Seated = large engaged at the stage-1 end; exit ok "
           "= no abort, Franka holding, large still engaged; success = both pulleys engaged "
           "after the release. * = run incomplete."), "",
          ("| arm | stage-1 seated | UR exit ok | stage 2 ran | both engaged | s1 solve p50/p95 "
           "med | s2 solve p50/p95 med | episode s med |"), "|" + "---|" * 8]
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        n = len(rr)
        md.append(f"| {arm} | {cnt(rr, 'stage1_seated')}/{n} | {cnt(rr, 'ur_exit_ok')}/{n} | "
                  f"{cnt(rr, 'stage2_ran')}/{n} | **{cnt(rr, 'engaged')}/{n}** | "
                  f"{f(med(rr, 's1_solve_p50_ms'), '.1f')}/{f(med(rr, 's1_solve_p95_ms'), '.1f')}"
                  f" | {f(med(rr, 's2_solve_p50_ms'), '.1f')}/"
                  f"{f(med(rr, 's2_solve_p95_ms'), '.1f')} | {f(med(rr, 'episode_s'), '.1f')} |")
    md += ["", "Both engaged per start (stage-1 seated in brackets):", "",
           "| start | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for k in starts:
        cells = []
        for arm in arms:
            rr = [r for r in rows if r["arm"] == arm and r["start"] == k]
            cells.append(f"{cnt(rr, 'engaged')}/{len(rr)} ({cnt(rr, 'stage1_seated')})"
                         if rr else "-")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    md += ["", "Failure classes:", ""]
    for arm in arms:
        fm = Counter(r["failure"] for r in rows if r["arm"] == arm and not r["engaged"])
        md.append(f"- {arm}: " + (", ".join(f"{k} x{v}" for k, v in fm.most_common()) or "none"))
    md += ["", ("| arm | start | ep | stage-1 end | stage-1 large | UR exit (large) | stage-2 "
                "end | final L/S | both | s1/s2 p95 ms | ep s | failure |"), "|" + "---|" * 12]
    for r in rows:
        st = r["start"] + ("" if r["complete"] else "*")
        if r.get("file") is None:
            md.append(f"| {r['arm']} | {st} | - |" + " - |" * 8 + f" {r['failure']} |")
            continue
        ux = ("-" if r["stage1_large"] != "engaged" else
              f"{'ok' if r['ur_exit_ok'] else 'fail'} ({r['ur_exit_large']})")
        md.append(
            f"| {r['arm']} | {st} | {r['repeat']} | {r['stage1_end']} | {r['stage1_large']} | "
            f"{ux} | {r.get('mpc_end', '-')} | {r.get('final_large', '-')}/"
            f"{r.get('final_small', '-')} | {'yes' if r['engaged'] else 'no'} | "
            f"{f(r['s1_solve_p95_ms'], '.1f')}/{f(r['s2_solve_p95_ms'], '.1f')} | "
            f"{f(r['episode_s'], '.1f')} | {r['failure']} |")
    return "\n".join(md) + "\n"


def cmd_summarize_full(a) -> int:
    dirs = [(REPO / d).resolve() for d in a.dir]
    keys = start_keys(a.starts)
    rows = [r for r in collect_full(dirs) if r["start"] in keys]
    if not rows:
        raise SystemExit(f"no full-run dirs (<arm>_<start> with index chain) under {a.dir}")
    out = (REPO / a.out).resolve() if a.out else dirs[0]
    out.mkdir(parents=True, exist_ok=True)
    (out / "hard_suite.json").write_text(json.dumps(
        {"suite": "full", "dirs": [rel(d) for d in dirs], "rows": rows}, indent=1,
        default=float) + "\n")
    md = full_md(rows)
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
    w.add_argument("--early-stop", default="10,16,15", metavar="W_ENGAGED,W_FAIL,T_MIN|off",
                   help="eval --early-stop (early_stop/sweep.py: re-score older suites with the same rule); off = full episodes")
    w.add_argument("--order", choices=("start", "arm"), default="start")
    w.add_argument("--cloud-noise", default="off", help="eval --cloud-noise preset of every arm")
    w.add_argument("--scene", default=None,
                   help="ROUND_BELT_SCENE for every run (a scene variant, e.g. assets/round_belt_task/round_belt_scene_zed720.yaml)")
    w.add_argument("--starts", default=None, help=f"comma subset of {','.join(KEYS)} or all (default: without {','.join(DROPPED)})")
    w.add_argument("--label", default="hard_suite")
    w.add_argument("--demo-actions", choices=("auto", "none"), default="auto",
                   help="none: no --mppi-demo-actions (the node gets no demo action inputs)")
    w.add_argument("--eval-extra", default="",
                   help="flags appended verbatim to every harness command (e.g. --end-snapshot)")
    w.add_argument("--suite", choices=("stage1", "stage2", "full"), default="stage1",
                   help="stage2: the stage-2 snapshot starts + harness flags; full: stage-1 "
                        "starts, harness --stage2-chain with --arm2 (module doc)")
    w.add_argument("--arm2", nargs=4, action="append", default=None,
                   metavar=("NAME", "DEPLOY", "WAYPOINTS", "MPPI_YAML"),
                   help="full: the stage-2 node of arm NAME")
    w.add_argument("--stage2-max-episode-s", type=float, default=30.0,
                   help="full: the chain's stage-2 timeout")
    w.add_argument("--stage2-src", default=STAGE2_SRC,
                   help="stage 2: start variants <src>_<key>_<rep>, rep < --repeats")
    w.add_argument("--final-hold-s", type=float, default=3.0,
                   help="stage 2: --mppi-max-duration-s (full: --stage2-mppi-max-duration-s; "
                        "the last waypoint's hold; tol -1)")
    w.add_argument("--franka-min-mm", type=float, default=2.0,
                   help="stage 2: --mppi-clearance-franka-min-mm (match the arms' yaml "
                        "clearance.franka_min_m: the yaml wins in the node)")
    s = sub.add_parser("summarize", help="hard_suite.{json,md}")
    s.add_argument("--dir", action="append", required=True)
    s.add_argument("--out", default=None, help="default: the first --dir")
    s.add_argument("--starts", default=None, help=f"comma subset or all (default: without {','.join(DROPPED)})")
    s.add_argument("--suite", choices=("stage1", "stage2", "full"), default="stage1")
    s.add_argument("--collector", action="append", default=None,
                   help="stage 2: a stage2_collect run dir as scripted control rows")
    s.add_argument("--stage2-src", default=STAGE2_SRC,
                   help="stage 2: collector rows of this start arm only ('' = all)")
    it = iter(sys.argv[1:])  # glue --eval-extra's value: argparse rejects one starting '--'
    a = ap.parse_args([f"--eval-extra={next(it, '')}" if x == "--eval-extra" else x
                       for x in it])
    return cmd_write(a) if a.cmd == "write" else cmd_summarize(a)


if __name__ == "__main__":
    sys.exit(main())
