#!/usr/bin/env python3
"""Operator launcher: SpaceMouse teleop of stage 2 in the whole-run chain (V12-120).

Per start (``run_hard_suite.SUITE`` key) one harness run, sequentially, in the foreground under the
sim lock (stdin stays the terminal: keys d / a / p, the next / retry / quit prompt): the
``run_hard_suite.py write --suite full`` stage-1 line for ``--model`` + ``--stage2-chain
--stage2-teleop --live-viewer``. Unknown flags pass through to the harness (later flags win, e.g.
``--teleop-max-s 8``). The SpaceMouse driver must publish ``SPACE_MOUSE_FRANKA_STATE`` on the
printed ``--lcm-url`` (docs/teleop-stage2.md).

    uv run --frozen python scripts/lcs/teleop_stage2.py --model V45c --start nominal
    uv run --frozen python scripts/lcs/teleop_stage2.py --model V55b --start set2_gv_00 --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "lcs"))

import run_hard_suite as rhs

E = "data/lcs/realfirst/cmp/encoder_study"
MODELS = {"V45c": (f"{E}/sim/V45c/deploy_v17c/deploy.npz", f"{E}/sim/V45c/latent_waypoints.npz",
                   f"{E}/sim87_V45c_ur30/configs/V45c_ur30_H7.yaml"),
          "V55b": (f"{E}/sim/V55b/deploy_v23b/deploy.npz", f"{E}/sim/V55b/latent_waypoints.npz",
                   "data/lcs/realfirst/cmp/v12_113/configs/V55b_stage1.yaml")}
SCENE = "assets/round_belt_task/round_belt_scene_wr625b15.yaml"
LOCK = "data/lcs/realfirst/cmp/v12_79/sim.lock"


def harness_argv(model: str, start: tuple, port: int, out: str, max_episode_s: float,
                 extra: list[str]) -> list[str]:
    """``cmd_write_full``'s stage-1 line (resets ``suite``) + the teleop chain flags."""
    key, states, variants, seed = start
    dep, wp, cfg = MODELS[model]
    reset = (["--reset", "scripted", "--reset-spec", "slack_mm=10", f"seed={seed}"]
             if seed is not None else ["--reset", "snapshot"])
    return ["uv", "run", "--frozen", "python", "scripts/lcs/eval_learned_mpc.py",
            "--mode", "learned", "--lcm-url", rhs.url(port), "--record",
            "--max-episode-s", f"{max_episode_s:g}", "--settle-s", "0.5",
            "--early-stop", "4,14,12", "--pace", "1", "--deploy", dep, "--demo-goals", "none",
            "--start-states", states, *(["--variants", variants] if variants else []),
            "--repeats", "1", "--belt-seg", "--cloud-noise", "off", "--controller", "mppi",
            "--mppi-config", cfg, "--mppi-goal", "waypoints", "--mppi-stages", "all",
            "--mppi-max-duration-s", "30.0", "--mppi-debug", "--label", f"teleop_{model}_{key}",
            "--out", out, "--mppi-clearance", "--mppi-clearance-ur-min-mm", "3.0",
            "--mppi-clearance-franka-min-mm", "2.0", *reset, "--mppi-waypoints", wp,
            "--mppi-goal-mode", "stage", "--mppi-demo-offset-frames", "20",
            "--outcome-pulley", "both", "--post-mpc", "stage2_release", "--stage2-chain",
            "--stage2-teleop", "--live-viewer",
            "--franka-width-max", f"{rhs.STAGE2_WIDTH_MAX_MM:g}", "--end-snapshot", *extra]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), default="V45c", help="stage-1 controller")
    ap.add_argument("--start", default="nominal", help=f"comma list of {','.join(rhs.KEYS)}")
    ap.add_argument("--port", type=int, default=7840)
    ap.add_argument("--out", default=None,
                    help="default data/lcs/realfirst/cmp/v12_120/session_<YYYYmmdd-HHMM>; one "
                         "<out>/<model>_<start> run dir per start")
    ap.add_argument("--scene", default=SCENE, help="exported as ROUND_BELT_SCENE")
    ap.add_argument("--max-episode-s", type=float, default=25.0, help="stage-1 timeout")
    ap.add_argument("--dry-run", action="store_true", help="print the command(s) and exit")
    a, extra = ap.parse_known_args()
    if a.port < 7780 or a.port in (7667, 8081, 8082):
        raise SystemExit(f"--port {a.port}: private ports >= 7780 only (never 7667 / 8081 / 8082)")
    keys = [k.strip() for k in a.start.split(",") if k.strip()]
    bad = [k for k in keys if k not in rhs.KEYS]
    if bad or not keys:
        raise SystemExit(f"--start {a.start}: keys within {rhs.KEYS}")
    if not (REPO / a.scene).is_file():
        raise SystemExit(f"--scene {a.scene}: no such file")
    out = a.out or f"data/lcs/realfirst/cmp/v12_120/session_{time.strftime('%Y%m%d-%H%M')}"
    starts = {s[0]: s for s in rhs.SUITE}
    cmds = [["flock", LOCK, *harness_argv(a.model, starts[k], a.port, f"{out}/{a.model}_{k}",
                                          a.max_episode_s, extra)] for k in keys]
    for cmd in cmds:
        print(f"cd {REPO} && ROUND_BELT_SCENE={shlex.quote(a.scene)} {shlex.join(cmd)}",
              flush=True)
    if a.dry_run:
        return 0
    env = {**os.environ, "ROUND_BELT_SCENE": a.scene}
    for i, (key, cmd) in enumerate(zip(keys, cmds, strict=True)):
        if i and sys.stdin.isatty() and input(f"next start {key}? [Y/n] ").strip().lower() == "n":
            break
        proc = subprocess.Popen(cmd, cwd=REPO, env=env)
        while True:
            try:
                rc = proc.wait()
                break
            except KeyboardInterrupt:  # the harness got the same SIGINT: let it clean up
                continue
        if rc != 0:
            print(f"harness exited {rc}; remaining starts skipped", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
