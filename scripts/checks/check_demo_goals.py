#!/usr/bin/env python3
"""Headless check of the demonstration episode, its demo re-export and ``demo_goals.npz``.

No sim, no LCM. D0 the demo episode validates, is engaged, nominal/excitation off, with a
recording, frames as defined; D1 the ``deploy_demo`` yaml/npz carry the stage keys, name the demo
and match ``LatentEncoder.encode`` on frames 0 / T-1; D2 ``demo_goals.npz`` schema + calibration;
D3 re-running ``make_demo_goals.py`` reproduces every array and the report except ``created``.

Run:
    uv run python scripts/checks/check_demo_goals.py [--demo-dir DIR] [--deploy PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for sub in ("src", "scripts/lcs"):
    if str(REPO_ROOT / sub) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / sub))

import make_demo_goals as mdg

from task_common import lcs_dataset as lcs
from task_common.latent_encoder import DemoGoals, LatentEncoder, LearnedLcs

DEFAULT_DEMO_DIR = REPO_ROOT / "data" / "lcs" / "demo"
DEFAULT_DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_ablation_20260924/"
                      "ckpt_decoded_only/deploy_demo/deploy.npz")
STAGE_KEYS = ("z_goal_stage1", "z_goal_stage2")
ENCODE_TOL = 1e-5
# The stage goals are the exporter's torch z; the stored z is the numpy re-encode.
GOAL_DIST_ZERO_TOL = 1e-4

_require = mdg._require
CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


@check("D0 demo episode")
def check_d0(ctx: SimpleNamespace) -> str:
    lcs.validate_episode(ctx.episode)
    with np.load(ctx.episode, allow_pickle=True) as d:
        ctx.data = {k: d[k] for k in d.files}
    data = ctx.data
    _require(str(data["sim_outcome"]) == "engaged", f"outcome {data['sim_outcome']!r}")
    meta = json.loads(str(data["sim_meta"]))
    nominal = meta.get("scenario") == "nominal" and meta["intent"] == "nominal"
    fallback = meta.get("scenario") is None and meta["intent"] == "engaged"
    _require(nominal or fallback, f"scenario {meta.get('scenario')!r}, intent {meta['intent']!r}")
    _require(not meta["excitation"]["on"], "excitation on")
    _require(not np.any(data["sim_excite_dpos_m"]) and not np.any(data["sim_excite_rotvec"]),
             "non-zero excitation rows")
    ctx.frames = mdg.demo_frames(data)
    src = mdg.recording_dir(ctx.episode, mdg.sha256(ctx.episode))
    _require(src is not None, "no collection run holds a copy of the demo")
    _require(src["recording"] is not None and (REPO_ROOT / src["recording"]).is_dir(),
             f"recording missing: {src['recording']}")
    fr = ctx.frames
    return (f"scenario {meta.get('scenario')}, intent {meta['intent']}, T {fr['T']}, frames "
            f"{fr['i']},{fr['j']} ({'/'.join(fr['labels'])}), first hold:place_3 "
            f"{fr['first_hold_place_3_frame']}, start {fr['start_err_mm']:.2f} mm, "
            f"run {src['run_dir']}")


@check("D1 deploy_demo stage goals")
def check_d1(ctx: SimpleNamespace) -> str:
    y = yaml.safe_load((ctx.deploy.parent / "learned_lcs.yaml").read_text())
    missing = [k for k in (*STAGE_KEYS, "goal_source", "z_goal") if k not in y]
    _require(not missing, f"yaml lacks {missing}")
    fr = ctx.frames
    want = f"#{fr['i']},{fr['j']}"
    src = str(y["goal_source"])
    _require(src.startswith("demo:") and src.endswith(want)
             and Path(src[5:-len(want)]).resolve() == ctx.episode.resolve(),
             f"goal_source {src!r} does not name {ctx.episode}{want}")
    _require(np.array_equal(y["z_goal"], y["z_goal_stage2"]), "yaml z_goal != z_goal_stage2")
    model = LearnedLcs.load(ctx.deploy)
    _require(model.goal_source == src, "npz goal_source != yaml")
    _require(np.array_equal(model.z_goal, model.stage_goals[1]), "npz z_goal != z_goal_stage2")
    yaml_err = float(np.abs(np.array([y[k] for k in STAGE_KEYS]) - model.stage_goals).max())
    _require(yaml_err <= 1e-12, f"yaml vs npz stage goals {yaml_err:.2e}")
    _require(model.goal_frames is not None
             and model.goal_frames.tolist() == [fr["i"], fr["j"]],
             f"goal_frames {model.goal_frames}")
    enc = LatentEncoder.load(ctx.deploy)
    d = ctx.data
    err = max(float(np.abs(enc.encode(d["pcd"][t], d["state"][t], d["pcd_belt"][t])
                           - model.stage_goals[k]).max())
              for k, t in enumerate((fr["i"], fr["j"])))
    _require(err <= ENCODE_TOL, f"encode vs stage goals {err:.2e} > {ENCODE_TOL}")
    ctx.model = model
    return f"goal_source {src}, encode err {err:.2e}, yaml vs npz {yaml_err:.1e}"


@check("D2 demo_goals.npz")
def check_d2(ctx: SimpleNamespace) -> str:
    with np.load(ctx.goals, allow_pickle=False) as d:
        ctx.arrays = {k: d[k] for k in d.files}
    a = ctx.arrays
    goals = DemoGoals(a)
    fr = ctx.frames
    _require(goals.n_stages == 2, f"n_stages {goals.n_stages}")
    _require(goals.stage_labels == list(mdg.STAGE_LABELS), f"labels {goals.stage_labels}")
    _require(goals.stage_frames.tolist() == [fr["i"], fr["j"]], f"frames {goals.stage_frames}")
    _require(goals.first_hold_place_3_frame == fr["first_hold_place_3_frame"],
             "first_hold_place_3_frame")
    _require(np.array_equal(goals.z_goals, ctx.model.stage_goals), "z_goals != deploy stage keys")
    _require(np.array_equal(goals.z_std, ctx.model.z_std), "z_std != deploy")
    _require(goals.z.shape[0] == fr["T"], f"z rows {goals.z.shape[0]} != T {fr['T']}")
    zero = max(abs(goals.goal_dist[f, k]) for k, f in enumerate(goals.stage_frames))
    _require(zero <= GOAL_DIST_ZERO_TOL, f"goal_dist at the stage frames {zero:.2e}")
    _require(goals.max_durations_s.tolist() == list(mdg.MAX_DURATIONS_S), "max_durations_s")
    d = ctx.data
    idx = goals.stage_frames
    for key, src in (("ee_pose_franka", "sim_ee_franka"), ("ee_pose_ur", "sim_ee_ur"),
                     ("pcd_belt_stage", "pcd_belt"), ("belt_xyz_stage", "sim_belt_xyz"),
                     ("pulley_pose_stage", "sim_pulley_large_pose"), ("state_stage", "state")):
        _require(np.array_equal(getattr(goals, key), np.asarray(d[src])[idx].astype(np.float64)),
                 f"{key} != demo {src}")
    _require(goals.demo_sha256 == mdg.sha256(ctx.episode), "demo_sha256")
    _require(goals.deploy_sha256 == mdg.sha256(ctx.deploy), "deploy_sha256")
    rep = json.loads(mdg.report_path(ctx.goals).read_text())
    s0, s1 = rep["stage_0_pre_place_1"], rep["stage_1_place_3"]
    nums = [s0["tol"], s0["floor_p50"], s0["floor_p90"], s1["tol"],
            *(v for row in s1["separation"]["by_outcome"].values()
              for v in (row["p10"], row["p50"], row["p90"], row["frac_below_tol"]))]
    _require(all(math.isfinite(v) for v in nums), "non-finite calibration number")
    _require(np.all(np.isfinite(goals.goal_tols)) and np.all(goals.goal_tols > 0), "goal_tols")
    _require(np.isclose(s0["tol"], mdg.STAGE0_TOL_FACTOR * s0["floor_p90"]), "stage 0 tol")
    _require(s1["tol"] == ctx.model.goal_tol, "stage 1 tol != deploy goal_tol_whitened")
    rec = rep["recommended_learned_mpc_yaml"]
    _require(rec["stage_goal_tols"] == goals.goal_tols.tolist()
             and rec["stage_max_durations_s"] == goals.max_durations_s.tolist(),
             "recommended yaml values != npz")
    return (f"tols {goals.goal_tols.round(4).tolist()}, stage-frame dist {zero:.1e}, stage-0 "
            f"floor p50 {s0['floor_p50']:.4f} p90 {s0['floor_p90']:.4f} over "
            f"{s0['n_train_files']} files")


@check("D3 make_demo_goals reproducible")
def check_d3(ctx: SimpleNamespace) -> str:
    rep = json.loads(mdg.report_path(ctx.goals).read_text())
    arrays, report = mdg.build(ctx.episode, ctx.deploy, rep["stage_0_pre_place_1"]["train_glob"])
    out, rep_path = mdg.write(arrays, report, ctx.tmp / ctx.goals.name)
    with np.load(out, allow_pickle=False) as d:
        new = {k: d[k] for k in d.files}
    _require(sorted(new) == sorted(ctx.arrays), "key sets differ")
    diff = [k for k in new if k != "created" and (
        new[k].dtype != ctx.arrays[k].dtype or new[k].shape != ctx.arrays[k].shape
        or new[k].tobytes() != ctx.arrays[k].tobytes())]
    _require(not diff, f"arrays differ: {diff}")
    old_rep = json.loads(mdg.report_path(ctx.goals).read_text())
    new_rep = json.loads(rep_path.read_text())
    old_rep.pop("created")
    new_rep.pop("created")
    _require(old_rep == new_rep, "report differs beyond created")
    return f"{len(new)} arrays byte-identical (created excepted), report identical"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    args = parser.parse_args()
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="check_demo_goals_"))
    ctx = SimpleNamespace(tmp=tmp, deploy=args.deploy,
                          episode=(args.demo_dir / "demo_episode.npz").resolve(),
                          goals=args.demo_dir / "demo_goals.npz")
    exit_code = 0
    try:
        for name, fn in CHECKS:
            try:
                detail = fn(ctx)
            except AssertionError as exc:
                print(f"[FAIL] {name}: {exc}", file=sys.stderr)
                exit_code = 1
                break
            except Exception as exc:  # noqa: BLE001 - report, then still clean up
                print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
                traceback.print_exc()
                exit_code = 1
                break
            print(f"[PASS] {name}: {detail}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if exit_code == 0:
        print(f"ALL DEMO GOALS CHECKS PASSED ({time.perf_counter() - t0:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
