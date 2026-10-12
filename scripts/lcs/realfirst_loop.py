#!/usr/bin/env python3
"""V11 real-first J&P loop driver (PLAN-20261005-realfirst-v1 §4), one step at a time.

Steps of iteration i (theta_i = iter<i>/model.json, deploy_i = iter<i>/deploy.json):
    seed      (i = 0) theta_0 = joint refit from scratch on the seed buffer (60 epochs); G2
              numbers + the recon-vs-noise-floor gate -> iter0/gates.json
    refit     (i >= 1) joint warm refit of theta_{i-1} on B_i (30 epochs, --gate-ref); the loop
              accepts iff refit_from_buffer accepts AND recon <= 1.1 x the real_like floor;
              rejected -> theta_{i-1} kept
    trust     box_i = mean +- max(k_i sigma, 0.2 x outer half) of B_i's demo/MPC actions, clipped
              to the outer safety box (+-6 mm / +-2 deg per step); k_i = min(1 + 0.25 i, 2)
    export    latent cache of B_i (z_std), deploy_i (belt_cloud_pointnet, box_i, the seed demo's
              two goals: its pre_place_2 frame and its final seated frame) + decoder;
              check_latent_encoder.py must PASS; goal.json
    collect   one harness run: first start drawn from the collection pool (set2 + set3, seeded,
              unused first), then --repeats chained by scripted resets; MPPI, two stages (4 s /
              30 s), action noise 0.2, --belt-seg --cloud-noise real_like; attempts.json
    label     human_label.py (engaged rule, one bit per episode) -> labels.json
    buffer    FIFO 50 (successes exempt until > 20) -> iter<i>/buffer.json + root buffer.json;
              allowlist.json; c1's deleted-keys loader test on 3 buffer files
    eval      (i odd) the 6 seat6 cells x 1, MPPI without action noise, snapshot resets
    report    budget.json (seed + every index.json), policy screen, learning_curve.{json,md,png},
              stop-rule verdict and the G4 line (printed, never acted on silently)
    register  model row (new model) + eval rows in docs/models.{md,yaml} (idempotent)
v1.1 (PLAN-20261005-realfirst-v1_1 rev 2; --root data/lcs/realfirst/v1_1 --config
v1_1/loop.yaml): merged seed manifests, Sinkhorn-10 refit gated on clean-cloud recon (G2'), the
planner box = trust region U a zero-centred minimum box, stage-1 latent tolerance
(latent_stage_tols.py) after every export, clearance in MPPI, buffer_policy.apply, G4'. Every
v1.1 knob is a config key whose default reproduces v1.
V12 campaign keys (PKG-20261006-cmp-harness-prefix-curriculum; defaults reproduce v1.1):
``prefix`` (demo-prefix replay before MPPI in `collect`, frac per iteration from a reverse
curriculum: iter<i>/curriculum.json; on iff ``prefix.demos``), ``collection_pool_rule``
(demo starts while frac > x), ``mppi_stages`` (all | last | auto), ``goal_mode`` /
``waypoints`` (latent_waypoints.py after every export, --mppi-goal waypoints), ``demo_prior``,
``mppi_H``, ``mppi_demo_offset_frames``, ``refit.fresh_each_iter``. `eval` never replays a
prefix. ``waypoint_bank`` ({knn, success_only}): after every export a per-demo waypoint bank
with the new encoder (latent_waypoints.py --bank-out), the node picks the demo with the
nearest initial latent per episode (--mppi-waypoint-bank).
Waypoint schedule (opt-in ``waypoints`` keys): ``min_frame`` (int, or "offset" =
mppi_demo_offset_frames) -> latent_waypoints.py --min-frame; ``skip_ahead`` -> MppiParams
stage_skip_ahead; ``no_timeout_into`` ("latch" | int) -> MppiParams stage_no_timeout_into;
``start`` ("progress") -> MppiParams stage_start (V12-31).
``behaviour_gate`` ({stride, max_drop, min_n, phases, max_hold_push_mm,
max_hold_push_increase_mm, max_prelatch_push_increase_mm, max_prelatch_descent_drop_mm,
n_successes, no_prior}): a refit passing the prediction gates is
exported to a scratch deploy (box_{i-1}) and screened by behaviour_gate.py against deploy_{i-1}
(MPPI u_0 vs the demos' / MPC successes' next actions, Franka u_0 y on demo hold frames;
iter<i>/behaviour_gate/); a failed rule rejects it (theta_{i-1} kept).
``mppi_overshoot``: MppiParams.overshoot in the collect/eval yaml (seat_tol "tol_2" = goal.json).
``support_box`` ({kinds, lo_pct, hi_pct, margin_mm, weight, hard_margin_mm}, missing keys from
SBOX_DEFAULTS): per iteration the [p_lo, p_hi] of B_i's EE positions (kinds: demo / excitation /
mpc, success = label_human) widened by margin_mm -> iter<i>/support_box.json and
MppiParams.support_box in the collect/eval yaml.
MILES anchor buffer (PLAN-20261006-miles; opt-in): ``anchor`` ({kinds, frac, decay_to, decay_by},
needs buffer_policy): rows of ``kinds`` are never evicted and outside the cap; with ``frac`` the
`buffer` step writes per-episode ``weight`` so their frames make frac (linearly to decay_to by
iteration decay_by) of the frame-weighted total of the next refit (refit_from_buffer.py reads
``weight``). ``collection_schedule`` ([start keys "<set>/<id>"]): iteration i collects from entry
i mod n (eval-disjoint like the pool) instead of the seeded pool draw.
Seated goal set (V12-30; opt-in): ``goal_set`` ({kinds, include_successes, all_demos, strict,
softmin_tau}, missing keys from GOAL_SET_DEFAULTS; {} = the defaults): per iteration
goal_set.py on B_i with deploy_i (final-frame latents of the seated rows of kinds, label_human or
with strict label_strict, + seated MPC rows) -> iter<i>/
goal_set.npz / .json and MppiParams.goal_set / goal_set_tau in the collect/eval yaml (the node's
last stage: soft-min over the set instead of the goal demo's seated latent).
Task-driven refits (V12-36; opt-in, need ``refit.mode`` lcs_only): ``onpolicy_weighting``
({off_track: {tol_k, after_s, streak}, off_track_weight, fail_weight}, task_common.
onpolicy_weighting): the `buffer` step gives each new MPC row ``frame_weights`` from its recorded
``sim_latent`` (the deploy encoder's; else encoded with deploy_i) vs iteration i's
latent_waypoints.npz tube, plus ``off_track`` stats; anchor shares then use frame-weighted
frames; buffer.json ``onpolicy_weighting``. ``refit.from_seed`` (bool): every refit starts from
theta_0's checkpoint (the seed) instead of theta_{i-1}.
External seed (V12-46; opt-in): ``seed_model`` ({checkpoint, refit_report[, source]}): the `seed`
step adopts this checkpoint and its refit_from_buffer report as theta_0 (no seed refit; the
merged seed buffer, gates.json, model.json, refit.json as usual), so ``refit.from_seed`` refits
start from it. ``response_gate`` ({stride, max_cos}; report-only): in `report`, the V12-44
response-direction statistic of deploy_i on iteration i's collect states (response_gate.py) ->
iter<i>/response_gate.json and a learning-curve section.
V12-57 keys (opt-in): ``refit.extra_args`` ([str]): appended to every refit_from_buffer.py command.
``graph`` ({rebuild, stride, weight, device, extra, buffers}, GRAPH_LOOP_DEFAULTS): before each
refit (i >= 1) build_cloud_graph.py on B_i (+ ``buffers``; ``extra`` = its extra args) ->
iter<i>/graph/graph.{npz,json}, passed as --graph-file (+ --graph-weight); stats in refit.json and
the learning curve (lcs_only: the encoder is frozen, the graph only enters metrics.graph).
``deploy_policy`` ({mode keep_best | latest, metric eval_engaged | eval_strict, min_gain}):
keep_best evaluates theta_i (exported to iter<i>/cand/) before `collect` when eval is due (always
at i = 0) and the new `deploy` step deploys it iff score >= the deployed model's + min_gain, else
keeps the deployed model; iter<i>/deploy.json is the deployed model's export (collect, buffer,
report); iter<i>/deploy_policy.json. ``eval_suite`` ({name: hard, repeats}): the eval runs are
run_hard_suite.SUITE (11 starts, its scripted-reset seeds) instead of ``eval_runs``; an eval run's
``reset_seed`` = scripted resets with that seed. ``eval_every: 1`` = every iteration. JEPA
checkpoints (report train_objective jepa) export with LCS_CACHE_REPRO_TOL_REL=1e-5.
C3+ controller (V12-40; opt-in): ``controller: c3plus`` (default mppi) runs magna's C++ learned
MPC (worktree magna-deploy-learned-lcs, projection C3+) in `collect` / `eval` instead of the MPPI
node: per run c3plus_params.py writes the LCS + params yaml into the worktree
(``c3plus.archive``/<root>/iter<i>/<run>/) and iter<i>/c3plus_<run>.json, the harness gets
--params. ``c3plus`` ({variant wp2 | pp2 | track, w_q, w_r, admm_iter, box_z, exact_tol,
archive}, missing keys from C3PLUS_DEFAULTS): stage goals (wp2: the latch + seated waypoints;
"last" stages: seat), costs, the support box's z faces as hard EE bounds, the export's E3 bound
(C3 runs the exact LCP). The MPPI yaml is still written (behaviour gate / policy screen proxy).
A done step is skipped; --force STEP moves its outputs aside (.prev<n>) and redoes it. `all`
refuses to start iteration i when the verdict after i-1 is STOP (--ignore-stop overrides).
Sim ground truth is read only by `label` (human_label.py) and `report`/`eval` (reporting).

Run (this repo):
    uv run --frozen python scripts/lcs/realfirst_loop.py --iter 0 --step all --dry-run
    uv run --frozen python scripts/lcs/realfirst_loop.py --smoke --iter 0 --step all
    uv run --frozen python scripts/lcs/realfirst_loop.py --iter 0 --step all   # 7764 / 7765
"""

from __future__ import annotations

import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
HOME = Path.home()
LCS = HOME / "git/lcs_learning"
PY_LCS = LCS / ".venv/bin/python"
LOADER_TEST = LCS / "outputs/sim_belt_v11_20261005/tests/test_belt_seg_loader.py"
V1 = "data/lcs/realfirst/v1"
STARTS = "data/lcs/start_states"
DATE = "20261005"

for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


STEPS = ["seed", "refit", "trust", "export", "collect", "label", "buffer", "eval", "deploy",
         "report", "register"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"
START_SETTLE_S = 0.5  # eval_learned_mpc.START_SETTLE_S: settle after every restore / reset
LAST_N = 13
# loader allowlist (lcs_learning dataset_round_belt_tuples.BELT_SEG_KEYS) + what the loop reads
LOADER_KEYS = ["pcd_belt_seg", "state", "actions", "sim_meta", "sim_ee_franka", "sim_ee_ur",
               "sim_cloud_h_mm", "sim_cloud_wrap_deg", "sim_cloud_r_mm", "sim_cmd_source"]
SIM_ONLY = ["pcd_belt", "sim_belt_xyz", "sim_wrap_deg", "sim_h_median_mm", "sim_outcome",
            "sim_pulley_large_pose"]

DEFAULTS = {
    "buffer_max": 50, "new_per_iter": 5, "success_exempt_until": 20,
    "k_base": 1.0, "k_step": 0.25, "k_max": 2.0, "floor_frac": 0.2,
    "outer_box": {"trans_m": 0.006, "rot_deg": 2.0},
    "trust_kinds": ["demo", "mpc"],  # miles excluded on purpose: scripted excursions, not policy
    "moving_eps": 1.0e-6,  # collector frames with max|u| above this count as moving
    "refit": {"mode": "joint", "seed_epochs": 60, "epochs": 30, "lr": 3.0e-4,
              "belt_a_init": 0.9, "state_ee_positions": False, "seed": 0,
              "fresh_each_iter": False},  # V12: true = every refit fresh (seed_epochs)
    # v1 gates recon vs the real_like floor via recon_floor_mm / recon_floor_factor (v1
    # loop.yaml); without them the refit's recon_max_mm (clean clouds) decides
    "g2_auc_min": 0.85,
    "eval_every": 2,
    "eval_runs": [
        {"run": "fe_nominal", "start_states": f"{STARTS}/pre_place_1_osc.npz", "variants": None,
         "repeats": 1, "cells": ["fe_nominal"], "floors": None},
        {"run": "fe_set1", "start_states": f"{STARTS}/grasp_variants/set1",
         "variants": "gv_01,gv_05", "repeats": 1, "cells": ["fe_set1_gv_01", "fe_set1_gv_05"],
         "floors": None},
        {"run": "urtip", "start_states": f"{STARTS}/ur_depth", "variants": "ud+3_r0",
         "repeats": 1, "cells": ["urtip"], "floors": None},
        # urtip6 + urtip6_high: same start, same goal here -> one run, 2 repeats (distinct noise)
        {"run": "urtip6", "start_states": f"{STARTS}/ur_depth", "variants": "ud+6_r0",
         "repeats": 2, "cells": ["urtip6", "urtip6_high"],
         "floors": {"z_min_franka": 0.037634701754760747, "z_min_ur": 0.03209393023877791}},
    ],
    "stop": {"zero_success_attempts": 25, "max_consecutive_rejections": 2,
             "done_successes": 4, "done_consecutive": 2},
    "mppi_config": f"{V1}/mppi.yaml",
    "stage_max_durations_s": [4.0, 30.0],
    "action_noise_frac": 0.2,
    "max_episode_s": 36.0,
    "noise_preset": "real_like",
    "reset_spec": ["slack_mm=10"],
    "label_rule": "engaged",
    "goal_episode": f"{V1}/seed/demo/episode_0000.npz",
    "collection_pool": ["grasp_variants/set2", "grasp_variants/set3"],
    "seed": 20261005,
    "policy_screen_n": 32,
    # v1.1 knobs; these defaults reproduce v1
    "lcs_out": "outputs/sim_belt_v11_20261005",
    "seed_buffers": [f"{V1}/seed/buffer.json"],
    "seed_buffer_merged": None,
    "collect_port": 7764, "eval_port": 7765,
    "export_seeds": 4,  # v1: c1's float32 guard retry; v1.1 (P5): 1, a failed export is an error
    "box_min_frac": None,  # planner box = trust region U zero-centred +-frac x outer half
    # V12: {kinds, lo_pct, hi_pct}: planner box U the seed demos' signed [p_lo, p_hi] per channel
    "box_demo_union": None,
    "mppi_clearance": False, "mppi_clearance_ur_min_mm": None,
    "mppi_clearance_franka_min_mm": None,
    "stage_tol": None,  # {percentile, floor, window}: latent_stage_tols.py after every export
    "stage_reach_streak": None, "rot_envelope_decision": None,
    "buffer_policy": None,  # {cap, short_abort_max, min_failures, short_abort_weight}
    "model_letter_start": "a",
    "model_id_prefix": "V11",  # V12 cells: "V12" (V11a-c are v1's models)
    "registry": {"suite": "seat6-rf", "tag": "rf-rmse+vg+aux-fixB", "cloud": "real_like",
                 "date": "2026-10-05",
                 "suite_text": "seat6 starts, the seed demo's two goals (pre_place_2 4 s, seated "
                               "30 s), MPPI node, belt-seg cloud real_like (V11)"},
    "g2_ref_report": None,  # one-step gate reference (P2's C16 control on the same split)
    "g2_one_step_ratio": 1.25, "g2_rho_max": 1.05, "g2_zgc_max": 1.5,
    # V12 knobs; these defaults reproduce v1.1. prefix: on iff demos ({"<set>/<variant>":
    # episode}); schedule null = the fixed frac, else a ladder from start_frac
    "prefix": {"demos": {}, "frac": 0.0, "schedule": [2.0, 1.5, 1.0, 0.5, 0.0],
               "advance_when": 3, "retreat_after": 2, "start_frac": 1.0},
    "collection_pool_rule": None,  # {while_frac_gt, pool, then_pool}; "demo_starts" = demos
    "mppi_stages": "all",  # all | last | auto (last while the prefix frac >= 1.0)
    "goal_mode": None,  # stage | track (needs waypoints)
    # {stride, window, percentile, floor[, min_frame, skip_ahead, no_timeout_into, start]}:
    # latent_waypoints.py per export (+ the opt-in stage schedule)
    "waypoints": None,
    "demo_prior": None,  # {alpha, weight, sync, mode, bc_prior} -> mppi yaml prior_* / bc_prior
    "mppi_H": None,
    "mppi_demo_offset_frames": None,  # node demo clock at the first latent (20 = skip prehold)
    # {knn, success_only}: per-demo waypoint bank after every export (needs waypoints)
    "waypoint_bank": None,
    # {stride, max_drop, min_n, phases, max_hold_push_mm, max_hold_push_increase_mm,
    #  max_prelatch_push_increase_mm, max_prelatch_descent_drop_mm, n_successes, no_prior}:
    #  behaviour refit gate (behaviour_gate.py)
    "behaviour_gate": None,
    "refit_gate": "enforce",  # "report": every refit is deployed, gate verdicts only logged
    "check_exact_tol": None,  # E3 loose-PGD bound (default 1e-2); MPPI uses the train-fixed PGD
    "mppi_overshoot": None,  # MppiParams.overshoot (seat_tol "tol_2": goal.json's)
    # {kinds, lo_pct, hi_pct, margin_mm, weight, hard_margin_mm}: MppiParams.support_box from B_i
    "support_box": None,
    "anchor": None,  # {kinds, frac, decay_to, decay_by}: MILES anchor buffer (needs buffer_policy)
    "collection_schedule": None,  # [start keys]: iteration i collects from entry i mod n
    # {kinds, include_successes, all_demos, strict, softmin_tau}: MppiParams.goal_set (V12-30)
    "goal_set": None,
}
SBOX_KINDS = ("demo", "excitation", "mpc", "miles", "success")  # success: label_human, any kind
ANCHOR_DEFAULTS = {"kinds": ["demo", "miles"], "frac": 0.4, "decay_to": None, "decay_by": 9}
NON_TASK_KINDS = ("miles",)  # not task attempts: neither success nor failure in separation()
# weight 0.5: one knot 5 mm outside costs 12.5 ~ the MPPI temperature (11-15 in the last stage)
SBOX_DEFAULTS = {"kinds": ["demo", "success"], "lo_pct": 0.0, "hi_pct": 100.0, "margin_mm": 15.0,
                 "weight": 0.5, "hard_margin_mm": None}
GOAL_SET_DEFAULTS = {"kinds": ["demo"], "include_successes": False, "all_demos": False,
                     "strict": False, "softmin_tau": 0.0}
SEED_MODEL_KEYS = {"checkpoint", "refit_report", "source"}
RGATE_DEFAULTS = {"stride": 20, "max_cos": 0.3}
GRAPH_LOOP_DEFAULTS = {"rebuild": True, "stride": 3, "weight": None, "device": "cuda", "extra": [],
                       "buffers": []}
DEPLOY_POLICY_DEFAULTS = {"mode": "latest", "metric": "eval_engaged", "min_gain": 0}
EVAL_SUITE_DEFAULTS = {"name": "hard", "repeats": 2}
JEPA_ENV = {"LCS_CACHE_REPRO_TOL_REL": "1e-5"}  # unit-scale JEPA latents fail the 1e-6 cache check
# V12-40 controller c3plus: costs of the last C3+ campaign (V8/V10); exact_tol = E3 at 1e-2
N_C3 = 7  # the LCS yaml's N (C3 horizon)
C3PLUS_DEFAULTS = {"variant": "wp2", "w_q": 1.0, "w_r": 0.3, "admm_iter": 2, "box_z": False,
                   "exact_tol": 1e-2,
                   "archive": "systems/parameters/learned_archive/2026-10-07/c3plus/loop"}
SMOKE = {"new_per_iter": 2, "refit": {"seed_epochs": 2, "epochs": 2},
         "eval_runs": [DEFAULTS["eval_runs"][0], DEFAULTS["eval_runs"][2]],
         "policy_screen_n": 8}


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
    Path(p).write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def c3_on(cfg: dict) -> bool:
    return cfg.get("controller", "mppi") == "c3plus"


def check_c3plus(cfg: dict) -> None:
    """``controller`` / ``c3plus`` (V12-40): MPPI-only options are refused, not ignored."""
    ctl = cfg.get("controller", "mppi")
    if ctl not in ("mppi", "c3plus"):
        raise SystemExit(f"controller {ctl!r}: mppi | c3plus")
    if ctl == "mppi":
        if cfg.get("c3plus") is not None:
            raise SystemExit("c3plus block needs controller: c3plus")
        return
    c = {**C3PLUS_DEFAULTS, **(cfg.get("c3plus") or {})}
    w = cfg["waypoints"] or {}
    bad = [k for k, on in (("demo_prior", cfg["demo_prior"]), ("waypoint_bank",
                           cfg["waypoint_bank"] is not None), ("goal_set", cfg["goal_set"]
                                                               is not None),
                           ("mppi_overshoot", cfg["mppi_overshoot"]),
                           ("goal_mode track", cfg["goal_mode"] == "track"),
                           *((f"waypoints.{k}", w.get(k) is not None)
                             for k in ("skip_ahead", "no_timeout_into", "start", "min_frame")))
           if on]
    if (set(c) != set(C3PLUS_DEFAULTS) or c["variant"] not in ("wp2", "pp2", "track")
            or not cfg["waypoints"] or bad or (c["box_z"] and not cfg["support_box"])
            or not float(c["exact_tol"]) > 0 or int(c["admm_iter"]) < 1):
        raise SystemExit(f"controller c3plus {c}: keys {sorted(C3PLUS_DEFAULTS)}, variant wp2 | "
                         "pp2 | track, needs waypoints (box_z: support_box); MPPI-only options "
                         f"set: {bad}")


def check_cfg(cfg: dict) -> None:
    """The V12 keys' values (the v1.1 defaults pass)."""
    check_c3plus(cfg)
    if cfg.get("refit_gate", "enforce") not in ("enforce", "report"):
        raise SystemExit(f"refit_gate {cfg['refit_gate']!r} must be enforce | report")
    if cfg["mppi_stages"] not in ("all", "last", "auto"):
        raise SystemExit(f"mppi_stages {cfg['mppi_stages']!r}: all | last | auto")
    if cfg["goal_mode"] not in (None, "stage", "track"):
        raise SystemExit(f"goal_mode {cfg['goal_mode']!r}: stage | track | null")
    if cfg["goal_mode"] and not cfg["waypoints"]:
        raise SystemExit("goal_mode needs waypoints (the latent_waypoints.py artifact)")
    dp = cfg["demo_prior"] or {}
    if dp.get("mode", "demo") not in ("demo", "policy") or (dp.get("mode") == "policy" and (
            not dp.get("bc_prior") or dp.get("sync") is not None
            or (cfg["waypoints"] or {}).get("no_timeout_into") is not None)):
        raise SystemExit(f"demo_prior.mode {dp.get('mode')!r}: demo | policy (needs bc_prior; "
                         "not with sync or waypoints.no_timeout_into)")
    if cfg["demo_prior"] and cfg["demo_prior"].get("sync") is not None \
            and (cfg["demo_prior"]["sync"] != "stage" or cfg["goal_mode"] != "stage"):
        raise SystemExit(f"demo_prior.sync {cfg['demo_prior']['sync']!r}: stage (with "
                         "goal_mode stage) | null")
    if cfg["waypoint_bank"] is not None and not cfg["waypoints"]:
        raise SystemExit("waypoint_bank needs waypoints (its rule and the base artifact)")
    w = cfg["waypoints"] or {}
    mf, nt = w.get("min_frame"), w.get("no_timeout_into")
    if mf is not None and (cfg["waypoint_bank"] is not None or (
            mf != "offset" and (isinstance(mf, bool) or not isinstance(mf, int) or mf < 0))
            or (mf == "offset" and cfg["mppi_demo_offset_frames"] is None)):
        raise SystemExit(f"waypoints.min_frame {mf!r}: int >= 0 | offset (needs "
                         "mppi_demo_offset_frames); not with waypoint_bank")
    if w.get("skip_ahead") not in (None, True, False) or (nt is not None and (
            nt != "latch" and (isinstance(nt, bool) or not isinstance(nt, int) or nt < 1))):
        raise SystemExit(f"waypoints.skip_ahead {w.get('skip_ahead')!r} (bool) / no_timeout_into "
                         f"{nt!r} (latch | int >= 1)")
    if (w.get("skip_ahead") or nt is not None) and cfg["goal_mode"] != "stage":
        raise SystemExit("waypoints.skip_ahead / no_timeout_into need goal_mode stage")
    if w.get("start") not in (None, "progress") or (w.get("start")
                                                    and cfg["goal_mode"] != "stage"):
        raise SystemExit(f"waypoints.start {w.get('start')!r}: progress (goal_mode stage) | null")
    if cfg["mppi_overshoot"] and cfg["mppi_overshoot"].get("seat_tol") == "tol_2" \
            and not cfg["stage_tol"]:
        raise SystemExit("mppi_overshoot seat_tol tol_2 needs stage_tol (goal.json tol_2)")
    if cfg["box_demo_union"] and not cfg["box_min_frac"]:
        raise SystemExit("box_demo_union widens the planner box: needs box_min_frac")
    sb = cfg["support_box"] and {**SBOX_DEFAULTS, **cfg["support_box"]}
    if sb and (not set(sb["kinds"]) <= set(SBOX_KINDS) or not sb["kinds"]
                           or not 0.0 <= float(sb["lo_pct"]) < float(sb["hi_pct"]) <= 100.0
                           or float(sb["margin_mm"]) < 0.0 or not float(sb["weight"]) >= 0.0
                           or (sb.get("hard_margin_mm") is not None
                               and float(sb["hard_margin_mm"]) < 0.0)):
        raise SystemExit(f"support_box {sb}: kinds within {SBOX_KINDS}, 0 <= lo_pct < hi_pct "
                         "<= 100, margin_mm >= 0, weight >= 0, hard_margin_mm >= 0 | null")
    an = cfg["anchor"] and {**ANCHOR_DEFAULTS, **cfg["anchor"]}
    if an and (not cfg["buffer_policy"] or not isinstance(an["kinds"], list) or not an["kinds"]
               or not all(isinstance(k, str) for k in an["kinds"])
               or (an["frac"] is not None and not 0.0 < float(an["frac"]) < 1.0)
               or (an["decay_to"] is not None and (an["frac"] is None
                                                   or not 0.0 < float(an["decay_to"]) < 1.0))
               or isinstance(an["decay_by"], bool) or not isinstance(an["decay_by"], int)
               or an["decay_by"] < 2):
        raise SystemExit(f"anchor {an}: needs buffer_policy; kinds a non-empty list of kinds, "
                         "0 < frac < 1 | null, 0 < decay_to < 1 | null (needs frac), int "
                         "decay_by >= 2")
    gs = cfg["goal_set"] is not None and {**GOAL_SET_DEFAULTS, **cfg["goal_set"]}
    if gs and (set(gs) != set(GOAL_SET_DEFAULTS) or not isinstance(gs["kinds"], list)
               or not gs["kinds"] or not set(gs["kinds"]) <= {"demo", "excitation", "mpc"}
               or not isinstance(gs["include_successes"], bool)
               or not isinstance(gs["all_demos"], bool) or not isinstance(gs["strict"], bool)
               or isinstance(gs["softmin_tau"], bool)
               or not float(gs["softmin_tau"]) >= 0.0 or cfg["goal_mode"] == "track"):
        raise SystemExit(f"goal_set {gs}: keys {sorted(GOAL_SET_DEFAULTS)}, kinds a non-empty "
                         "list within demo / excitation / mpc, bools, softmin_tau >= 0; not with "
                         "goal_mode track")
    fs = cfg["refit"].get("from_seed", False)
    if not isinstance(fs, bool) or (fs and cfg["refit"].get("fresh_each_iter")):
        raise SystemExit(f"refit.from_seed {fs!r}: bool; not with refit.fresh_each_iter")
    sm = cfg.get("seed_model")
    if sm is not None and (not isinstance(sm, dict) or not set(sm) <= SEED_MODEL_KEYS
                           or not all(isinstance(sm.get(k), str) and sm[k]
                                      for k in ("checkpoint", "refit_report"))
                           or not isinstance(sm.get("source", ""), str)):
        raise SystemExit(f"seed_model {sm!r}: {{checkpoint, refit_report[, source]}} (paths)")
    rgc = cfg.get("response_gate")
    rgm = rgc if isinstance(rgc, dict) else {}
    rgm = {**RGATE_DEFAULTS, **rgm}
    if rgc is not None and (not isinstance(rgc, dict) or set(rgm) != set(RGATE_DEFAULTS)
                            or isinstance(rgm["stride"], bool)
                            or not isinstance(rgm["stride"], int) or rgm["stride"] < 1
                            or isinstance(rgm["max_cos"], bool)
                            or not isinstance(rgm["max_cos"], (int, float))):
        raise SystemExit(f"response_gate {rgc!r}: {{stride: int >= 1, max_cos: float}}")
    check_v12_57(cfg)
    ow = cfg.get("onpolicy_weighting")
    if ow is not None:
        from task_common import onpolicy_weighting

        why = onpolicy_weighting.check_rule(ow)
        if why or not cfg["waypoints"] or cfg["refit"]["mode"] != "lcs_only" \
                or cfg["refit"].get("fresh_each_iter"):
            raise SystemExit(f"onpolicy_weighting {ow}: {why or 'ok'}; needs waypoints and "
                             "refit.mode lcs_only (not fresh_each_iter)")
    cs = cfg["collection_schedule"]
    if cs is not None and (not isinstance(cs, list) or not cs or cfg["collection_pool_rule"]
                           or not all(isinstance(k, str) and "/" in k for k in cs)):
        raise SystemExit(f"collection_schedule {cs!r}: a non-empty list of '<set>/<id>' start "
                         "keys; not with collection_pool_rule")
    pc = cfg["prefix"]
    fr = [pc["frac"], pc["start_frac"], *(pc.get("schedule") or [])]
    if not all(0.0 <= float(x) <= 2.0 for x in fr):
        raise SystemExit(f"prefix fracs outside [0, 2]: {fr}")
    if pc.get("schedule") and pc["start_frac"] not in pc["schedule"]:
        raise SystemExit(f"prefix.start_frac {pc['start_frac']} not in the schedule")


def _strs(x) -> bool:
    return isinstance(x, list) and all(isinstance(s, str) for s in x)


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def check_v12_57(cfg: dict) -> None:
    """``refit.extra_args``, ``graph``, ``deploy_policy``, ``eval_suite``, ``eval_every``."""
    ea = cfg["refit"].get("extra_args")
    if ea is not None and not _strs(ea):
        raise SystemExit(f"refit.extra_args {ea!r}: a list of strings")
    gc = cfg.get("graph")
    if gc is not None:
        g = {**GRAPH_LOOP_DEFAULTS, **gc} if isinstance(gc, dict) else {}
        if (not isinstance(gc, dict) or set(g) != set(GRAPH_LOOP_DEFAULTS)
                or not isinstance(g["rebuild"], bool) or not _int(g["stride"]) or g["stride"] < 1
                or (g["weight"] is not None and (isinstance(g["weight"], bool) or not isinstance(
                    g["weight"], (int, float)) or g["weight"] < 0))
                or not isinstance(g["device"], str) or not _strs(g["extra"])
                or not _strs(g["buffers"])):
            raise SystemExit(f"graph {gc!r}: keys {sorted(GRAPH_LOOP_DEFAULTS)}; rebuild bool, "
                             "stride int >= 1, weight >= 0 | null, device str, extra / buffers "
                             "lists of str")
    dp = cfg.get("deploy_policy")
    if dp is not None:
        d = {**DEPLOY_POLICY_DEFAULTS, **dp} if isinstance(dp, dict) else {}
        if (not isinstance(dp, dict) or set(d) != set(DEPLOY_POLICY_DEFAULTS)
                or d["mode"] not in ("latest", "keep_best")
                or d["metric"] not in ("eval_engaged", "eval_strict")
                or not _int(d["min_gain"]) or d["min_gain"] < 0):
            raise SystemExit(f"deploy_policy {dp!r}: mode latest | keep_best, metric eval_engaged "
                             "| eval_strict, min_gain int >= 0")
        if d["mode"] == "keep_best" and (not cfg["eval_every"] or c3_on(cfg)):
            raise SystemExit("deploy_policy keep_best needs eval_every >= 1 (the candidates' "
                             "evals); not with controller c3plus")
    es = cfg.get("eval_suite")
    if es is not None:
        e = {**EVAL_SUITE_DEFAULTS, **es} if isinstance(es, dict) else {}
        if (not isinstance(es, dict) or set(e) != set(EVAL_SUITE_DEFAULTS) or e["name"] != "hard"
                or not _int(e["repeats"]) or e["repeats"] < 1):
            raise SystemExit(f"eval_suite {es!r}: {{name: hard, repeats: int >= 1}}")
    ev = cfg["eval_every"]
    if ev is not None and (not _int(ev) or ev < 0):
        raise SystemExit(f"eval_every {ev!r}: int >= 0 | null")
    for er in cfg["eval_runs"]:
        rs = er.get("reset_seed")
        if rs is not None and not _int(rs):
            raise SystemExit(f"eval run {er['run']}: reset_seed {rs!r} (int | null)")


def keep_best(cfg: dict) -> bool:
    return (cfg.get("deploy_policy") or {}).get("mode") == "keep_best"


def eval_runs_of(cfg: dict) -> list[dict]:
    """``eval_suite`` hard: run_hard_suite.SUITE (one run per start, its reset seeds), else
    ``eval_runs``."""
    es = cfg.get("eval_suite")
    if es is None:
        return cfg["eval_runs"]
    import run_hard_suite as rhs

    n = int({**EVAL_SUITE_DEFAULTS, **es}["repeats"])
    return [{"run": f"hs_{k}", "start_states": s, "variants": v or None, "repeats": n,
             "cells": [k], "floors": None, "reset_seed": seed} for k, s, v, seed in rhs.SUITE]


class Ctx:
    """Paths, URLs and config of one invocation (normal run or --smoke)."""

    def __init__(self, a):
        self.a = a
        self.smoke = a.smoke
        if a.smoke:
            self.root = REPO / a.root / "smoke_loop" if a.root else \
                REPO / "data/lcs/realfirst/smoke_loop"
        else:
            self.root = REPO / (a.root or V1)
        self.i = a.iter
        cfg_path = self.root / "loop.yaml"
        base = DEFAULTS
        if cfg_path.is_file():
            base = merge(base, yaml.safe_load(cfg_path.read_text()) or {})
        if a.config:
            base = merge(base, yaml.safe_load(Path(a.config).read_text()) or {})
        if a.smoke:  # last: smoke sizes win over the root's config
            base = merge(base, SMOKE)
        if not cfg_path.is_file() and not a.dry_run:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(
                f"# realfirst_loop.py defaults (plan §4 + orchestrator decisions), written {now()}"
                "\n" + yaml.safe_dump(base, sort_keys=False, width=100))
        check_cfg(base)
        self.cfg = base
        self.out = LCS / base["lcs_out"] / "smoke_loop" if a.smoke else LCS / base["lcs_out"]
        self.seed_buffers = [REPO / b for b in base["seed_buffers"]]
        self.seed_budgets = [b.parent / "budget.json" for b in self.seed_buffers]
        self.seed_buffer = (REPO / base["seed_buffer_merged"] if base["seed_buffer_merged"]
                            else self.seed_buffers[0])
        p = base["collect_port"] if a.smoke else None
        self.url_collect = url(a.collect_port or p or base["collect_port"])
        self.url_eval = url(a.eval_port or p or base["eval_port"])

    def it(self, i=None) -> Path:
        return self.root / f"iter{self.i if i is None else i}"

    def logs(self, i=None) -> Path:
        return self.it(i) / "logs"

    def lout(self, i=None) -> Path:
        return self.out / f"iter{self.i if i is None else i}"


class CandCtx(Ctx):
    """keep_best: iteration i's candidate theta_i, exported / evaluated under iter<i>/cand/."""

    def __init__(self, ctx: Ctx):
        self.__dict__.update(ctx.__dict__)

    def it(self, i=None) -> Path:
        d = super().it(i)
        return d / "cand" if i is None or i == self.i else d

    def lout(self, i=None) -> Path:
        d = super().lout(i)
        return d / "cand" if i is None or i == self.i else d


# ---------------------------------------------------------------- state


@contextmanager
def locked(ctx: Ctx, i: int):
    d = ctx.it(i)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".state.lock", "w") as fl:
        fcntl.flock(fl, fcntl.LOCK_EX)
        yield d / "state.json"


def get_state(ctx: Ctx, i: int) -> dict:
    p = ctx.it(i) / "state.json"
    return jload(p) if p.is_file() else {"iter": i, "steps": {}, "failures": []}


def set_step(ctx: Ctx, i: int, step: str, **kw):
    with locked(ctx, i) as p:
        st = jload(p) if p.is_file() else {"iter": i, "steps": {}, "failures": []}
        st["steps"].setdefault(step, {}).update(kw)
        if kw.get("status") == "failed":
            st.setdefault("failures", []).append({"step": step, "at": kw.get("finished"),
                                                  "error": kw.get("error")})
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


def run(cmd, log: Path, cwd=REPO, ctx=None, step=None, env=None) -> int:
    """Subprocess with its own log (appended); pid recorded in the state while it runs;
    ``env``: extra environment variables."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n# {now()} {''.join(f'{k}={v} ' for k, v in (env or {}).items())}"
                f"{' '.join(map(str, cmd))}\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, **env} if env else None)
        if ctx is not None and step:
            set_step(ctx, ctx.i, step, pids=get_state(ctx, ctx.i)["steps"].get(step, {}).get(
                "pids", []) + [p.pid])
        return p.wait()


def show(cmd, cwd=REPO):
    where = "" if cwd == REPO else f"(cwd {cwd}) "
    print(f"  would run: {where}{' '.join(map(str, cmd))}")


# ---------------------------------------------------------------- models / buffer


def model_of(ctx: Ctx, i: int) -> dict:
    p = ctx.it(i) / "model.json"
    if not p.is_file():
        raise SystemExit(f"{rel(p)} missing: run --iter {i} --step {'seed' if i == 0 else 'refit'}")
    return jload(p)


def buffer_in(ctx: Ctx, i: int) -> Path:
    """B_i: the seed buffer for i = 0, else iteration i-1's buffer step output."""
    return ctx.seed_buffer if i == 0 else ctx.it(i - 1) / "buffer.json"


def new_model_id(ctx: Ctx, i: int) -> tuple[str, str]:
    if ctx.smoke:
        return (f"V11smoke{i}" if i else "V11smoke"), "smoke"
    ids = []
    for j in range(i):
        f = ctx.it(j) / "model.json"
        if f.is_file() and jload(f)["id"] not in ids:
            ids.append(jload(f)["id"])
    x = LETTERS[LETTERS.index(ctx.cfg["model_letter_start"]) + len(ids)]
    return f"{ctx.cfg['model_id_prefix']}{x}", x


def ckpt_arg(ck: str, key: str):
    out = subprocess.run([str(PY_LCS), "-c", (
        "import sys, torch; a = torch.load(sys.argv[1], map_location='cpu', "
        "weights_only=False)['args']; print(repr(a.get(sys.argv[2])))"), ck, key],
        cwd=LCS, capture_output=True, text=True, check=True).stdout.strip()
    return yaml.safe_load(out.replace("None", "null").replace("True", "true").replace(
        "False", "false"))


def recon_gate(ctx: Ctx, rep: dict) -> dict:
    r = rep["metrics"]["recon_chamfer_mm"]
    if "recon_floor_mm" not in ctx.cfg:  # v1.1: held-out Chamfer RMS on clean clouds
        mx = ctx.cfg["refit"].get("recon_max_mm")
        return {"recon_chamfer_mm": r, "max_mm": mx, "ratio": None,
                "ok": None if r is None or mx is None else bool(r <= mx)}
    floor, fac = ctx.cfg["recon_floor_mm"], ctx.cfg["recon_floor_factor"]
    return {"recon_chamfer_mm": r, "floor_mm": floor, "factor": fac,
            "ratio": None if r is None else r / floor,
            "ok": None if r is None else bool(r <= fac * floor)}


def refit_cmd(ctx: Ctx, i: int, init: str, epochs: int, tag: str, gate_ref=None):
    r = ctx.cfg["refit"]
    cmd = [PY_LCS, "scripts/refit_from_buffer.py", "--buffer", buffer_in(ctx, i),
           "--init-checkpoint", init, "--epochs", str(epochs),
           "--mode", "joint" if init == "none" else r["mode"],  # lcs_only needs a checkpoint
           "--lr", str(r["lr"]), "--seed", str(r["seed"]), "--out-dir", ctx.lout(i) / "refit",
           "--tag", tag]
    if gate_ref:
        cmd += ["--gate-ref", gate_ref]
    if r["state_ee_positions"]:
        cmd.append("--state-ee-positions")
    if init == "none":  # fresh-only; warm starts inherit the checkpoint's
        for k in ("latent_dim", "pc_recon_loss", "sinkhorn_epsilon", "sinkhorn_iters"):
            if r.get(k) is not None:
                cmd += [f"--{k.replace('_', '-')}", str(r[k])]
    # refit_from_buffer gates recon <= 6 mm by default since P2; v1 had no such gate
    mx = r.get("recon_max_mm")
    cmd += ["--recon-max-mm", "none" if mx is None else str(mx)]
    return cmd + list(r.get("extra_args") or [])


def run_refit(ctx: Ctx, cmd, tag: str, step: str) -> dict:
    rep_p = ctx.lout() / "refit" / tag / "refit_report.json"
    if not rep_p.is_file():
        if (rep_p.parent).exists():
            move_aside([rep_p.parent])  # a crashed refit leaves its dir; the CLI refuses it
        if run(cmd, ctx.logs() / f"{step}.log", cwd=LCS, ctx=ctx, step=step):
            raise SystemExit(f"refit_from_buffer failed (see {rel(ctx.logs() / f'{step}.log')})")
    return jload(rep_p)


def merge_seed(ctx: Ctx) -> dict:
    """The seed manifests concatenated (ids / episodes unique; ``weight`` 1.0, ``status``
    "ok" where missing) -> ``seed_buffer_merged``."""
    rows, src = [], []
    for b in ctx.seed_buffers:
        rows += [{"weight": 1.0, "status": "ok", **x} for x in jload(b)["rollouts"]]
        src.append({"buffer": rel(b), "sha256": sha(b)})
    for k in ("id", "episode"):
        if len({x[k] for x in rows}) != len(rows):
            raise SystemExit(f"seed manifests: duplicate {k}")
    doc = {"version": 2, "iter": -1, "max": ctx.cfg["buffer_max"], "n": len(rows),
           "source": "realfirst_loop.py: " + " + ".join(x["buffer"] for x in src),
           "sources": src, "rollouts": rows}
    jdump(ctx.seed_buffer, doc)
    return doc


def g2_gates(ctx: Ctx, m: dict, rep: dict, rg: dict) -> dict:
    """G2' (plan v1.1 §3) on the seed refit's held-out split."""
    ref = None
    if ctx.cfg["g2_ref_report"]:
        rr = jload(ctx.cfg["g2_ref_report"])
        ref = {"report": ctx.cfg["g2_ref_report"],
               "one_step_chamfer_mm": rr["metrics"]["one_step_chamfer_mm"],
               "same_split": rr["split"]["heldout"] == rep["split"]["heldout"]}

    def le(x, y):
        return None if x is None or y is None else bool(x <= y)

    g = {"auc_ok": None if m["separation_auc"] is None
         else bool(m["separation_auc"] >= ctx.cfg["g2_auc_min"]),
         "recon_clean_ok": rg["ok"],
         "one_step_ok": None if ref is None else le(
             m["one_step_chamfer_mm"], ctx.cfg["g2_one_step_ratio"] * ref["one_step_chamfer_mm"]),
         "rho_ok": le(m["rho_A"], ctx.cfg["g2_rho_max"]),
         "z_goal_consistency_ok": le(m.get("z_goal_consistency"), ctx.cfg["g2_zgc_max"])}
    g["all_ok"] = all(v is True for v in g.values())
    return {"gates": g, "one_step_ref": ref, "z_goal_consistency": m.get("z_goal_consistency"),
            "n_z_goal_demos": m.get("n_z_goal_demos"),
            "recon_train_loss_mm": m.get("recon_train_loss_mm"), "recon_loss": m.get("recon_loss")}


def st_seed(ctx: Ctx, dry: bool):
    if ctx.i != 0:
        print("  seed is iteration 0 only")
        return {"status": "skipped"}
    r, sm = ctx.cfg["refit"], ctx.cfg.get("seed_model")
    tag = "r0_seed"
    cmd = refit_cmd(ctx, 0, "none", r["seed_epochs"], tag)
    if dry:
        if ctx.cfg["seed_buffer_merged"]:
            print(f"  merge {[rel(b) for b in ctx.seed_buffers]} -> {rel(ctx.seed_buffer)}")
        if sm:
            print(f"  seed_model: theta_0 = {sm['checkpoint']} (report {sm['refit_report']}"
                  f"{'; ' + sm['source'] if sm.get('source') else ''}); no seed refit; G2 "
                  f"numbers of its report -> {rel(ctx.it(0))}/gates.json")
            return {}
        show(cmd, LCS)
        print(f"  -> theta_0 = {ctx.lout(0)}/refit/{tag}; G2 numbers -> "
              f"{rel(ctx.it(0))}/gates.json")
        return {}
    seed = merge_seed(ctx) if ctx.cfg["seed_buffer_merged"] else jload(ctx.seed_buffer)
    assert all(isinstance(x["label_human"], bool) for x in seed["rollouts"])
    jdump(ctx.it(0) / "buffer_seed.json", {**seed, "imported_from": rel(ctx.seed_buffer),
                                           "imported_sha256": sha(ctx.seed_buffer)})
    rep_p = str(ctx.lout(0) / "refit" / tag / "refit_report.json")
    if sm:
        rep, rep_p, same = seed_model_report(ctx, sm, seed)
        tag = rep["tag"]
    else:
        rep = run_refit(ctx, cmd, tag, "seed")
    ck = rep["checkpoint"]
    a_init = ckpt_arg(ck, "belt_a_init")
    if not ctx.smoke and abs(float(a_init) - r["belt_a_init"]) > 1e-12:
        raise SystemExit(f"belt_a_init {a_init} != {r['belt_a_init']}")
    m = rep["metrics"]
    rg = recon_gate(ctx, rep)
    gates = {"gate": "G2 (seed refit)", "source": rep_p,
             "separation_auc": m["separation_auc"], "separation_auc_set": m["separation_auc_set"],
             "auc_min": ctx.cfg["g2_auc_min"],
             "auc_ok": None if m["separation_auc"] is None
             else bool(m["separation_auc"] >= ctx.cfg["g2_auc_min"]),
             "recon": rg, "split": rep["split"], "rho_A": m["rho_A"], "B_fro_w": m["B_fro_w"],
             "drift_u0_w": m["drift_u0_w"], "one_step_chamfer_mm": m["one_step_chamfer_mm"],
             "open_loop7_chamfer_mm": m["open_loop7_chamfer_mm"],
             "aux_mae_h_mm": m["aux_mae_h_mm"], "aux_mae_wrap_deg": m["aux_mae_wrap_deg"]}
    if "recon_floor_mm" in ctx.cfg:
        gates["not_computed"] = ("z_goal consistency across 3 seated episodes; recon <= 6 mm on "
                                 "clean clouds (no clean cloud is recorded with --cloud-noise)")
    else:
        gates.update(gate="G2' (seed refit, clean clouds)", **g2_gates(ctx, m, rep, rg))
    if sm:
        gates["seed_model"] = {**sm, "same_buffer_rows": same}
    jdump(ctx.it(0) / "gates.json", gates)
    mid, letter = new_model_id(ctx, 0)
    model = {"id": mid, "letter": letter, "iter": 0, "new": True, "checkpoint": ck,
             "refit_report": rep_p,
             "refit_tag": tag, "decision": "seed", "buffer": str(ctx.seed_buffer),
             "belt_a_init": a_init, "epochs": rep["epochs"], "mode": rep["mode"],
             "source": f"seed refit (fresh, {rep['epochs']} epochs) on the e1 seed buffer"}
    if sm:
        model.update(seed_model=sm, seed_model_same_buffer_rows=same, source=(
            f"seed_model {sm.get('source') or ck}: {rep['mode']} {rep['epochs']} epochs on "
            f"{rep['buffer']}, adopted without a seed refit"))
    jdump(ctx.it(0) / "model.json", model)
    jdump(ctx.it(0) / "refit.json", {"decision": "seed", "loop_decision": "accepted",
                                     "report_decision": rep["decision"], "recon_gate": rg,
                                     "report": model["refit_report"],
                                     "gpu_s": rep["timings_s"]["total"]})
    print(f"  seed: {mid} one-step {m['one_step_chamfer_mm']} recon {m['recon_chamfer_mm']} "
          f"(floor ratio {rg['ratio']}) auc {m['separation_auc']}"
          + (f"; G2' {gates['gates']}" if "gates" in gates else ""))
    return {"outputs": [str(ctx.it(0) / "model.json"), str(ctx.it(0) / "gates.json")]}


def seed_model_report(ctx: Ctx, sm: dict, seed: dict) -> tuple[dict, str, bool]:
    """``seed_model``'s report (its checkpoint must be the given one) and whether it was
    trained on the seed buffer's rows (episode, kind, label, weight)."""
    rp = Path(sm["refit_report"]).resolve()
    rep = jload(rp)
    ck = Path(sm["checkpoint"]).resolve()
    if not ck.is_file() or Path(rep["checkpoint"]).resolve() != ck:
        raise SystemExit(f"seed_model: checkpoint {ck} missing or != its report's "
                         f"{rep['checkpoint']}")

    def key(rows):
        return [(x["episode"], x["kind"], x["label_human"], float(x.get("weight", 1.0)))
                for x in rows]

    same = Path(rep["buffer"]).is_file() and key(jload(rep["buffer"])["rollouts"]) == key(
        seed["rollouts"])
    if not same:
        print(f"  WARNING seed_model: trained on {rep['buffer']}, not the rows of "
              f"{rel(ctx.seed_buffer)}")
    return rep, str(rp), same


def graph_rule(cfg: dict) -> dict | None:
    g = cfg.get("graph")
    g = g is not None and {**GRAPH_LOOP_DEFAULTS, **g}
    return g if g and g["rebuild"] else None


def graph_npz(ctx: Ctx, i=None) -> Path:
    return ctx.it(i) / "graph" / "graph.npz"


def graph_buffers(ctx: Ctx) -> list[str]:
    return [str(b) for b in (buffer_in(ctx, ctx.i), *[REPO / x for x in
                                                       graph_rule(ctx.cfg)["buffers"]])]


def graph_cmd(ctx: Ctx) -> list:
    """build_cloud_graph.py on B_i (+ ``graph.buffers``) -> iter<i>/graph/graph.npz."""
    g = graph_rule(ctx.cfg)
    return [PY_LCS, "scripts/build_cloud_graph.py",
            *[x for b in graph_buffers(ctx) for x in ("--buffer", b)], "--out", graph_npz(ctx),
            "--stride", str(g["stride"]), "--device", g["device"], *g["extra"]]


def graph_args(ctx: Ctx) -> list[str]:
    w = graph_rule(ctx.cfg)["weight"]
    return ["--graph-file", str(graph_npz(ctx))] + ([] if w is None else ["--graph-weight", str(w)])


def build_graph(ctx: Ctx) -> dict:
    """The graph of B_i (rebuilt when its buffers changed) -> summary stats."""
    npz, cmd, want = graph_npz(ctx), graph_cmd(ctx), graph_buffers(ctx)
    js = npz.with_suffix(".json")
    if npz.is_file() and js.is_file() and jload(js)["args"]["buffer"] == want:
        wall = None
    else:
        if npz.parent.exists():
            move_aside([npz.parent])
        t0 = time.time()
        if run(cmd, ctx.logs() / "graph.log", cwd=LCS, ctx=ctx, step="refit"):
            raise SystemExit(f"build_cloud_graph failed (see {rel(ctx.logs() / 'graph.log')})")
        wall = time.time() - t0
    st = jload(js)["stats"]
    rc = st["coverage"]["rejected_candidates"]
    return {"npz": str(npz), "sha256": sha(npz), "buffers": want, "n_nodes": st["n_nodes"],
            "n_edges": st["n_edges"], "n_episodes": st["n_episodes"], "n_goal": st["n_goal"],
            "n_rejected": rc["n"], "rejected_frac": rc["frac"],
            "n_components": st["n_components"],
            "largest_component_frac": st["largest_component_frac"],
            "unreachable_frac": st["unreachable_frac"],
            "build_s": st["timings_s"]["total"] + st["timings_s"].get("coverage", 0.0),
            "wall_s": wall, "rule": graph_rule(ctx.cfg)}


def st_refit(ctx: Ctx, dry: bool):
    if ctx.i == 0:
        print("  refit: iteration 0 uses the seed step")
        return {"status": "skipped"}
    i = ctx.i
    sm = ctx.cfg.get("seed_model")
    sm0 = sm and {"checkpoint": sm["checkpoint"], "refit_report": sm["refit_report"],
                  "id": "theta_0 (seed_model)"}  # dry-run before the seed step
    prev = model_of(ctx, i - 1) if not dry or (ctx.it(i - 1) / "model.json").is_file() else (
        sm0 if i == 1 and sm else {"checkpoint": "<theta_{i-1}>",
                                   "refit_report": "<theta_{i-1} report>", "id": "?"})
    tag = f"r{i}_{ctx.cfg['refit']['mode']}"
    fresh = bool(ctx.cfg["refit"].get("fresh_each_iter"))
    from_seed = bool(ctx.cfg["refit"].get("from_seed"))
    if fresh:  # J&P: refit from scratch on B_i each iteration; gates still vs theta_{i-1}
        cmd = refit_cmd(ctx, i, "none", ctx.cfg["refit"]["seed_epochs"], tag,
                        prev["refit_report"])
    else:  # report mode: a kept model's rejected candidate is the latest parameters
        init = (prev["rejected"]["checkpoint"] if ctx.cfg["refit_gate"] == "report"
                and prev.get("rejected") else prev["checkpoint"])
        if from_seed:  # no drift accumulation: theta_0 + all of B_i every time
            seed = (model_of(ctx, 0) if not dry or (ctx.it(0) / "model.json").is_file()
                    else sm0 or {"checkpoint": "<theta_0>", "id": "?"})
            init = seed["checkpoint"]
        cmd = refit_cmd(ctx, i, init, ctx.cfg["refit"]["epochs"], tag, prev["refit_report"])
    gr = graph_rule(ctx.cfg)
    if gr:
        cmd += graph_args(ctx)
    if dry:
        if gr:
            show(graph_cmd(ctx), LCS)
            if ctx.cfg["refit"]["mode"] == "lcs_only" and not fresh:
                print("  graph: lcs_only refit (frozen encoder): the graph enters metrics.graph "
                      "only")
        if from_seed:
            print(f"  refit.from_seed: init = theta_0 ({seed['id']}), gates vs theta_{i - 1}")
        show(cmd, LCS)
        lim = (f"{ctx.cfg['recon_floor_factor']} x {ctx.cfg['recon_floor_mm']} mm"
               if "recon_floor_mm" in ctx.cfg
               else f"{ctx.cfg['refit'].get('recon_max_mm')} mm (clean)")
        print(f"  accept iff report decision accepted and recon <= {lim}; else keep "
              f"{prev['id']}")
        if ctx.cfg["behaviour_gate"]:
            for c in behaviour_cmds(ctx, f"<theta_{i} checkpoint>"):
                show(c, LCS if c[0] == PY_LCS else REPO)
            print("  then accept iff behaviour_gate.py accepts")
        return {}
    gs = build_graph(ctx) if gr else None
    rep = run_refit(ctx, cmd, tag, "refit")
    rg = recon_gate(ctx, rep)
    ok = rep["decision"] == "accepted" and rg["ok"] is True
    report_only = ctx.cfg["refit_gate"] == "report"
    bg = (behaviour_check(ctx, rep["checkpoint"]) if (ok or report_only)
          and ctx.cfg["behaviour_gate"] else None)
    if bg is not None and bg.get("accept") is False:
        ok = False
    verdict = "accepted" if ok else "rejected"
    ok = ok or report_only
    rep_path = str(ctx.lout() / "refit" / tag / "refit_report.json")
    res = {"decision": "accepted" if ok else "rejected", "gate_verdict": verdict,
           "refit_gate": ctx.cfg["refit_gate"], "report_decision": rep["decision"],
           "recon_gate": rg, "gates": rep["gates"], "report": rep_path,
           "ref_metrics": (rep.get("ref") or {}).get("metrics_same_split"),
           "metrics": rep["metrics"], "from": prev["id"], "gpu_s": rep["timings_s"]["total"]}
    if bg is not None:
        res["behaviour_gate"] = bg
    if from_seed:
        res["from_seed"] = {"id": seed["id"], "checkpoint": init}
    if "train_objective" in rep:
        res["train_objective"] = rep["train_objective"]
    if gs is not None:
        res["graph"] = {**gs, "refit_metrics": {k: (rep["metrics"].get("graph") or {}).get(k)
                                                for k in ("stress", "far", "pearson", "spearman",
                                                          "spearman_goal")}}
    if ok:
        mid, letter = new_model_id(ctx, i)
        model = {"id": mid, "letter": letter, "iter": i, "new": True,
                 "checkpoint": rep["checkpoint"], "refit_report": rep_path, "refit_tag": tag,
                 "decision": "accepted", "gate_verdict": verdict,
                 "parent": prev["id"], "buffer": str(buffer_in(ctx, i)),
                 "epochs": rep["epochs"], "mode": rep["mode"],
                 "source": f"joint {'fresh' if fresh else 'warm'} refit iter {i} "
                           f"({rep['epochs']} epochs) from {prev['id']}"
                           + (" (its rejected candidate)" if not fresh and not from_seed
                              and init != prev["checkpoint"] else "")}
        if fresh:
            model["fresh"] = True
        if from_seed:
            model["from_seed"] = seed["id"]
            model["source"] += f", initialised from the seed {seed['id']}"
    else:
        model = {**prev, "iter": i, "new": False, "kept": True,
                 "rejected": {"checkpoint": rep["checkpoint"], "report": rep_path,
                              "report_decision": rep["decision"], "recon_ok": rg["ok"],
                              "gates": rep["gates"]}}
        if bg is not None:
            model["rejected"]["behaviour_gate"] = bg
    jdump(ctx.it() / "refit.json", res)
    jdump(ctx.it() / "model.json", model)
    print(f"  refit {res['decision']} (gates {verdict}, report {rep['decision']}, recon ok {rg['ok']}"
          + (f", behaviour {bg['line']}" if bg is not None else "") + f"): model {model['id']}")
    if gs is not None:
        print(f"  graph {gs['n_nodes']} nodes, {gs['n_edges']} edges, rejected "
              f"{gs['rejected_frac']:.2%}, components {gs['n_components']}; held-out stress "
              f"{res['graph']['refit_metrics']['stress']}")
    return {"outputs": [str(ctx.it() / "refit.json"), str(ctx.it() / "model.json")]}


def behaviour_cmds(ctx: Ctx, ck: str) -> list[list]:
    """Candidate export (box_{i-1}), its waypoints, behaviour_gate.py vs deploy_{i-1}."""
    bc, prev_it = ctx.cfg["behaviour_gate"], ctx.it(ctx.i - 1)
    gd, ld = ctx.it() / "behaviour_gate", ctx.lout() / "behaviour_gate"
    pd = prev_it / "deploy.json"
    pdep = jload(pd) if pd.is_file() else {"deploy": "<deploy_{i-1}>", "box_json": "<box>"}
    gf = goal_frames(REPO / ctx.cfg["goal_episode"])
    dep = ld / "deploy_cand"
    exp = [PY_LCS, "scripts/export_learned_lcs_deploy.py", "--checkpoint", ck, "--data-files",
           gd / "export_files.txt", "--goal-episode", gf["episode"],
           f"--goal-frames={gf['frames'][0]},{gf['frames'][1] - gf['n_frames']}",
           "--u-bounds-json", pdep["box_json"], "--out-dir", dep]
    gate = [sys.executable, "scripts/lcs/behaviour_gate.py", "--cand", dep / "deploy.npz",
            "--ref", pdep["deploy"], "--mppi-config", prev_it / "mppi_collect.yaml",
            "--stride", str(bc.get("stride", 4)), "--max-drop", str(bc.get("max_drop", 0.1)),
            "--min-n", str(bc.get("min_n", 10)),
            "--phases", ",".join(bc.get("phases", ["approach", "descent"])),
            "--out", gd / "behaviour_gate.json"]
    for k in ("max_hold_push_mm", "max_hold_push_increase_mm", "max_prelatch_push_increase_mm",
              "max_prelatch_descent_drop_mm"):
        if bc.get(k) is not None:
            gate += [f"--{k.replace('_', '-')}", str(bc[k])]
    if bc.get("no_prior"):
        gate.append("--no-prior")
    cmds = [exp]
    if ctx.cfg["waypoints"]:
        cmds.append(waypoints_cmd(ctx, dep / "deploy.npz", gd / "latent_waypoints_cand.npz"))
        gate += ["--cand-waypoints", gd / "latent_waypoints_cand.npz", "--ref-waypoints",
                 prev_it / "latent_waypoints.npz"]
    succ = [r["episode"] for r in jload(buffer_in(ctx, ctx.i))["rollouts"]
            if r["kind"] == "mpc" and r.get("label_human")] if not ctx.a.dry_run or \
        buffer_in(ctx, ctx.i).is_file() else []
    succ = [e for e in succ if (Path(e).parent / "controller" / Path(e).with_suffix(
        ".solves.jsonl").name).is_file()][-int(bc.get("n_successes", 5)):]
    gate += ["--demos", *stage_demos(ctx)] + (["--successes", *succ] if succ else [])
    return cmds + [gate]


def behaviour_check(ctx: Ctx, ck: str) -> dict:
    """The behaviour gate on a prediction-accepted refit -> summary (``accept`` None: skipped)."""
    prev_it = ctx.it(ctx.i - 1)
    if not (prev_it / "deploy.json").is_file() or not (prev_it / "mppi_collect.yaml").is_file():
        return {"accept": None, "line": "skipped (no deploy / mppi yaml of iteration i-1)"}
    gd, ld = ctx.it() / "behaviour_gate", ctx.lout() / "behaviour_gate"
    if gd.exists() or ld.exists():
        move_aside([p for p in (gd, ld) if p.exists()])
    gd.mkdir(parents=True)
    (gd / "export_files.txt").write_text(
        "\n".join(r["episode"] for r in jload(buffer_in(ctx, ctx.i))["rollouts"]) + "\n")
    log = ctx.logs() / "behaviour_gate.log"
    for c in behaviour_cmds(ctx, ck):
        if run(c, log, cwd=LCS if c[0] == PY_LCS else REPO, ctx=ctx, step="refit"):
            raise SystemExit(f"behaviour gate failed (see {rel(log)})")
    doc = jload(gd / "behaviour_gate.json")
    d = doc["decision"]
    cos = {ph: {k: (doc["rows"][k].get(ph) or {}).get("cos_median") for k in ("cand", "ref")}
           for ph in ("approach", "descent", "hold", "success")}
    out = {"accept": d["accept"], "reasons": d["reasons"], "deltas": d["deltas"],
           "hold_push_mm": d["hold_push_mm"], "cos_median": cos,
           "cross_cos_median": doc["cross_cos_median"], "n_frames": doc["n_frames"],
           "n_successes": doc["n_successes"], "rule": doc["rule"], "wall_s": doc["wall_s"],
           "json": rel(gd / "behaviour_gate.json")}
    if "prelatch_mm" in d:
        out["prelatch_mm"] = d["prelatch_mm"]
    out["line"] = ("ACCEPT" if d["accept"] else "REJECT: " + "; ".join(d["reasons"])) + " (" + \
        ", ".join(f"{k} {v:+.3f}" for k, v in d["deltas"].items()) + ")"
    return out


# ---------------------------------------------------------------- trust region


def buffer_actions(ctx: Ctx, buf: dict, kinds=None) -> tuple[np.ndarray, dict]:
    kinds, eps = kinds or ctx.cfg["trust_kinds"], ctx.cfg["moving_eps"]
    us, per = [], {}
    for r in buf["rollouts"]:
        if r["kind"] not in kinds:
            continue
        with np.load(r["episode"], allow_pickle=True) as d:
            u = np.asarray(d["actions"], dtype=np.float64)
            src = np.asarray(d["sim_cmd_source"]) if "sim_cmd_source" in d.files else None
        m = np.isin(src, (0, 1)) if src is not None else np.abs(u).max(1) > eps
        us.append(u[m])
        per.setdefault(r["kind"], [0, 0, 0])
        per[r["kind"]][0] += 1
        per[r["kind"]][1] += int(m.sum())
        per[r["kind"]][2] += int(m.sum()) if r.get("status") == "aborted" else 0
    out = {k: {"rollouts": v[0], "frames": v[1]} for k, v in per.items()}
    if any(v[2] for v in per.values()):
        for k, v in per.items():
            out[k]["frames_aborted"] = v[2]
    return np.concatenate(us), out


def k_of(ctx: Ctx, i: int) -> float:
    c = ctx.cfg
    return float(min(c["k_base"] + c["k_step"] * i, c["k_max"]))


def demo_range(ctx: Ctx) -> dict:
    """Signed per-channel [p_lo, p_hi] of the seed manifest's demo actions (moving frames)."""
    c = ctx.cfg["box_demo_union"]
    u, per = buffer_actions(ctx, jload(ctx.seed_buffer), c["kinds"])
    lo, hi = np.percentile(u, [c["lo_pct"], c["hi_pct"]], axis=0)
    return {"lb": lo, "ub": hi, "frames_by_kind": per, "buffer": str(ctx.seed_buffer),
            **{k: c[k] for k in ("kinds", "lo_pct", "hi_pct")}}


def st_trust(ctx: Ctx, dry: bool):
    k = k_of(ctx, ctx.i)
    box = ctx.it() / f"box_{ctx.i}.json"
    ob = ctx.cfg["outer_box"]
    if dry:
        print(f"  box_{ctx.i}: k {k}, floor {ctx.cfg['floor_frac']} x outer half (+-"
              f"{ob['trans_m'] * 1e3:g} mm / +-{ob['rot_deg']:g} deg), kinds "
              f"{ctx.cfg['trust_kinds']} of {rel(buffer_in(ctx, ctx.i))} -> {rel(box)}"
              + (f"; planner box U zero-centred +-{ctx.cfg['box_min_frac']} x outer half"
                 if ctx.cfg["box_min_frac"] else "")
              + (f" U seed demo range {ctx.cfg['box_demo_union']}"
                 if ctx.cfg["box_demo_union"] else ""))
        return {}
    from task_common import trust_region

    buf = jload(buffer_in(ctx, ctx.i))
    u, per = buffer_actions(ctx, buf)
    lo, hi = trust_region.outer_box(ob["trans_m"], ob["rot_deg"])
    tr = trust_region.compute(u, k, ctx.cfg["floor_frac"], lo, hi)
    tr.update(buffer=str(buffer_in(ctx, ctx.i)), frames_by_kind=per,
              frame_rule="harness: sim_cmd_source in {0, 1}; collector: max|u| > "
                         f"{ctx.cfg['moving_eps']}")
    n_ab = sum(v.get("frames_aborted", 0) for v in per.values())
    tr["aborted_frame_share"] = n_ab / tr["n_frames"]
    src = (f"V11 trust region ub-tr{ctx.i}: k {k}, floor {ctx.cfg['floor_frac']}, "
           f"{tr['n_frames']} frames ({per}) of {rel(buffer_in(ctx, ctx.i))}, "
           f"outer +-{ob['trans_m'] * 1e3:g} mm / +-{ob['rot_deg']:g} deg")
    bx = {"u_lb": tr["u_lb"], "u_ub": tr["u_ub"], "source": src}
    frac = ctx.cfg["box_min_frac"]
    if frac:
        pb = trust_region.with_min_box(tr["u_lb"], tr["u_ub"], lo, hi, frac)
        tr["planner_box"] = pb
        bx = {"u_lb": pb["u_lb"], "u_ub": pb["u_ub"],
              "source": src + f"; U zero-centred +-{frac} x outer half (planner box)",
              "trust_u_lb": tr["u_lb"], "trust_u_ub": tr["u_ub"],
              "reversible": pb["reversible"], "widened": pb["widened"]}
        if ctx.cfg["box_demo_union"]:  # the box must never exclude a demo action
            dr = demo_range(ctx)
            blb, bub = np.asarray(pb["u_lb"]), np.asarray(pb["u_ub"])
            nlb = np.maximum(np.minimum(blb, dr["lb"]), lo)
            nub = np.minimum(np.maximum(bub, dr["ub"]), hi)
            pb.update(u_lb=nlb.tolist(), u_ub=nub.tolist(),
                      widened=((nlb < np.asarray(tr["u_lb"])) | (nub > np.asarray(tr["u_ub"])))
                      .tolist(), reversible=((nlb < 0) & (nub > 0)).tolist(),
                      disp=[{"channel": c, "unit": un, "lb": float(a * sc), "ub": float(b * sc)}
                            for c, un, a, b, sc in zip(trust_region.CHANNELS,
                                                       trust_region.DISPLAY_UNIT, nlb, nub,
                                                       trust_region.DISPLAY_SCALE)])
            dwid = ((nlb < blb) | (nub > bub)).tolist()
            pb["demo_union"] = {**dr, "lb": dr["lb"].tolist(), "ub": dr["ub"].tolist(),
                                "widened_by_demo": dwid}
            bx.update(u_lb=pb["u_lb"], u_ub=pb["u_ub"], widened=pb["widened"],
                      reversible=pb["reversible"], demo_union=pb["demo_union"],
                      source=bx["source"] + f"; U seed demo actions p{dr['lo_pct']:g}.."
                      f"p{dr['hi_pct']:g} ({dr['frames_by_kind']}), outer-clipped")
        if not all(pb["reversible"]):
            raise SystemExit(f"planner box: a channel cannot reverse: {pb['disp']}")
    jdump(ctx.it() / "trust_region.json", tr)
    jdump(box, bx)
    ra = tr["ratios"]
    print(f"  box_{ctx.i}: k {k}, {tr['n_frames']} frames, trans/rot gm {ra['box_trans_gm']:.2f}"
          f"/{ra['box_rot_gm']:.2f}, floor hits {tr['n_floor_hits']}, clip {tr['n_clip_hits']}"
          + (f", aborted frames {n_ab / tr['n_frames']:.0%}" if n_ab else ""))
    if frac:
        print("  planner box (mm | deg): " + ", ".join(
            f"{c['channel']} [{c['lb']:+.2f}, {c['ub']:+.2f}]" for c in pb["disp"])
            + f"; widened {sum(pb['widened'])}/12, reversible {sum(pb['reversible'])}/12"
            + (f", by demo {sum(pb['demo_union']['widened_by_demo'])}/12"
               if "demo_union" in pb else ""))
    return {"outputs": [str(ctx.it() / "trust_region.json"), str(box)]}


def support_box(ctx: Ctx, dry: bool = False) -> dict:
    """``support_box`` of B_i: per-coordinate [p_lo, p_hi] of every frame's [p_franka, p_ur]
    (sim_ee_*[:3], the node's P) of the rollouts of ``kinds``, widened by ``margin_mm``; written
    to iter<i>/support_box.json unless ``dry``."""
    c, src = {**SBOX_DEFAULTS, **ctx.cfg["support_box"]}, buffer_in(ctx, ctx.i)
    ps, per = [], {}
    for r in jload(src)["rollouts"]:
        tags = [k for k in c["kinds"] if r["kind"] == k or (k == "success" and r["label_human"])]
        if not tags:
            continue
        with np.load(r["episode"], allow_pickle=True) as d:
            p = np.concatenate([np.asarray(d["sim_ee_franka"], np.float64)[:, :3],
                                np.asarray(d["sim_ee_ur"], np.float64)[:, :3]], axis=1)
        ps.append(p)
        for k in tags:
            v = per.setdefault(k, {"rollouts": 0, "frames": 0, "lo_mm": None, "hi_mm": None})
            v["rollouts"] += 1
            v["frames"] += len(p)
            v["lo_mm"] = np.fmin(v["lo_mm"], p.min(0) * 1e3) if v["lo_mm"] is not None \
                else p.min(0) * 1e3
            v["hi_mm"] = np.fmax(v["hi_mm"], p.max(0) * 1e3) if v["hi_mm"] is not None \
                else p.max(0) * 1e3
    if not ps:
        raise SystemExit(f"support_box: no {c['kinds']} rollouts in {rel(src)}")
    p = np.concatenate(ps)
    lo, hi = np.percentile(p, [float(c["lo_pct"]), float(c["hi_pct"])], axis=0)
    m = float(c["margin_mm"]) * 1e-3
    hm = c.get("hard_margin_mm")
    box = {"lo": (lo - m).tolist(), "hi": (hi + m).tolist(), "weight": float(c["weight"]),
           "hard_margin_m": None if hm is None else float(hm) * 1e-3}
    out = {**box, "coords": ["franka_x", "franka_y", "franka_z", "ur_x", "ur_y", "ur_z"],
           "data_lo": lo.tolist(), "data_hi": hi.tolist(), "n_frames": len(p),
           "by_kind": {k: {**v, "lo_mm": np.round(v["lo_mm"], 2).tolist(),
                           "hi_mm": np.round(v["hi_mm"], 2).tolist()} for k, v in per.items()},
           "buffer": rel(src), **{k: c.get(k) for k in ("kinds", "lo_pct", "hi_pct", "margin_mm",
                                                         "hard_margin_mm")}}
    if not dry:
        jdump(ctx.it() / "support_box.json", out)
    return out


def goal_set_npz(ctx: Ctx, dry: bool = False) -> dict:
    """``goal_set`` of B_i encoded with deploy_i (goal_set.py) -> iter<i>/goal_set.npz + .json;
    built once per iteration (rebuilt when deploy_i changed); ``dry`` only selects the rows."""
    import goal_set as gsm

    c, src = {**GOAL_SET_DEFAULTS, **ctx.cfg["goal_set"]}, buffer_in(ctx, ctx.i)
    rows = gsm.select_rows(jload(src)["rollouts"], c["kinds"], c["include_successes"],
                           c["all_demos"], c["strict"])
    npz, js = ctx.it() / "goal_set.npz", ctx.it() / "goal_set.json"
    out = {"path": str(npz), "n": len(rows), "buffer": rel(src), "tau": float(c["softmin_tau"]),
           "kinds": {k: sum(r["kind"] == k for r in rows) for k in ("demo", "excitation", "mpc")}}
    if dry:
        return out
    dep = jload(ctx.it() / "deploy.json")
    if npz.is_file() and js.is_file() and jload(js).get("deploy_sha256") == dep["deploy_sha256"] \
            and jload(js).get("rule") == c:
        return {**out, **jload(js)}
    gs = gsm.build(dep["deploy"], rows, c["kinds"], c["include_successes"], c["all_demos"],
                   c["strict"])
    if npz.exists():
        move_aside([npz, js])
    gsm.save(npz, gs)
    D = gsm.pairwise(gs)
    summ = {**out, "rule": c, "deploy": dep["deploy"], "deploy_sha256": gs["deploy_sha256"],
            "episodes": gs["episodes"], "npz_sha256": sha(npz),
            "pairwise_median": float(np.median(D[np.triu_indices(len(D), 1)])) if len(D) > 1
            else 0.0}
    jdump(js, summ)
    return summ


def sbox_line(sb: dict) -> str:
    n = {k: v["rollouts"] for k, v in sb["by_kind"].items()}
    return (f"support box (mm) lo {np.round(np.asarray(sb['lo']) * 1e3, 1).tolist()} hi "
            f"{np.round(np.asarray(sb['hi']) * 1e3, 1).tolist()} ({sb['n_frames']} frames, "
            f"rollouts {n} of {sb['buffer']}), weight {sb['weight']:g}, hard margin "
            f"{sb['hard_margin_m']}")


# ---------------------------------------------------------------- export / goal


def goal_frames(ep: Path) -> dict:
    """Stage 1 = the first frame after move:pre_place_2 (V2 demo_goals convention), stage 2 =
    the final (done) frame of the seed demo."""
    with np.load(ep, allow_pickle=True) as d:
        meta = json.loads(str(d["sim_meta"]))
        phase = np.asarray(d["sim_phase"], dtype=np.int64)
        feats = {k: np.asarray(d[k]) for k in ("sim_cloud_h_mm", "sim_cloud_wrap_deg",
                                               "sim_cloud_r_mm", "sim_cloud_n_pts",
                                               "sim_cloud_success") if k in d.files}
    labels = [meta["phase_labels"][p] for p in phase]
    mv = [t for t, lab in enumerate(labels) if lab == "move:pre_place_2"]
    if not mv or mv[-1] + 1 >= len(labels) or labels[-1] != "done":
        raise SystemExit(f"{ep}: no move:pre_place_2 -> ... -> done phase sequence")
    i, j = mv[-1] + 1, len(labels) - 1
    return {"episode": str(ep), "frames": [i, j], "labels": [labels[i], labels[j]],
            "n_frames": len(labels), "stage_names": ["pre_place_2", "seated"],
            "cloud_features": {k: [v[i].item(), v[j].item()] for k, v in feats.items()}}


def export_cmds(ctx: Ctx, model: dict, dep: Path, files_txt: Path, box: Path, gf: dict):
    ck = model["checkpoint"]
    cache = [PY_LCS, "scripts/encode_latents_cache.py", "--model", "belt_cloud_pointnet",
             "--checkpoint", ck, "--files", files_txt, "--out", ctx.lout() / "cache_buffer.npz",
             "--workers", "8"]
    exp = [PY_LCS, "scripts/export_learned_lcs_deploy.py", "--checkpoint", ck, "--data-files",
           files_txt, "--goal-episode", gf["episode"],
           f"--goal-frames={gf['frames'][0]},{gf['frames'][1] - gf['n_frames']}",
           "--u-bounds-json", box, "--out-dir", dep]
    dec = [PY_LCS, "scripts/export_learned_lcs_deploy.py", "--checkpoint", ck,
           "--decoder-out", dep / "decoder.npz"]
    chk = [sys.executable, "scripts/checks/check_latent_encoder.py", "--deploy",
           dep / "deploy.npz"]
    if c3_on(ctx.cfg):  # C3 plans on the exact LCP
        chk += ["--exact-tol", str({**C3PLUS_DEFAULTS, **(ctx.cfg.get("c3plus") or {})}[
            "exact_tol"])]
    elif ctx.cfg.get("check_exact_tol") is not None:  # MPPI never runs the loose PGD
        chk += ["--exact-tol", str(ctx.cfg["check_exact_tol"])]
    return cache, exp, dec, chk


def stage_demos(ctx: Ctx) -> list[str]:
    """The seed manifests' demo episodes (the clean demos of plan v1.1)."""
    return [x["episode"] for b in ctx.seed_buffers for x in jload(b)["rollouts"]
            if x["kind"] == "demo"]


def stage_tol_cmd(ctx: Ctx, deploy) -> list:
    st = ctx.cfg["stage_tol"]
    return [sys.executable, "scripts/lcs/latent_stage_tols.py", "--deploy", deploy,
            "--demos", *stage_demos(ctx), "--out", rel(ctx.it() / "stage_tols.json"),
            "--window", str(st["window"]), "--floor", str(st["floor"]), "--percentile",
            str(st["percentile"])]


def wp_min_frame(cfg: dict) -> int | None:
    """``waypoints.min_frame`` ("offset" = mppi_demo_offset_frames); None = off."""
    mf = (cfg["waypoints"] or {}).get("min_frame")
    return int(cfg["mppi_demo_offset_frames"]) if mf == "offset" else mf


def waypoints_cmd(ctx: Ctx, deploy, out=None) -> list:
    """W1-B's latent_waypoints.py on the goal demo + the seed demos -> iter<i>/
    latent_waypoints.npz (or ``out``)."""
    w, mf = ctx.cfg["waypoints"], wp_min_frame(ctx.cfg)
    return [sys.executable, "scripts/lcs/latent_waypoints.py", "--deploy", deploy,
            "--demos", *stage_demos(ctx), "--goal-demo", ctx.cfg["goal_episode"],
            "--stride", str(w["stride"]), "--window", str(w["window"]), "--percentile",
            str(w["percentile"]), "--floor", str(w["floor"]),
            *(["--min-frame", str(mf)] if mf is not None else []), "--out",
            rel(out or ctx.it() / "latent_waypoints.npz")]


def bank_demos(ctx: Ctx) -> list[str]:
    """Bank entries: the seed demos (human-labelled successes only with ``success_only``)."""
    ok = (ctx.cfg["waypoint_bank"] or {}).get("success_only", False)
    return [x["episode"] for b in ctx.seed_buffers for x in jload(b)["rollouts"]
            if x["kind"] == "demo" and (not ok or x.get("label_human"))]


def bank_cmd(ctx: Ctx, deploy) -> list:
    """latent_waypoints.py --bank-out iter<i>/waypoint_bank (tols pooled over the seed demos)."""
    w = ctx.cfg["waypoints"]
    return [sys.executable, "scripts/lcs/latent_waypoints.py", "--deploy", deploy,
            "--demos", *stage_demos(ctx), "--goal-demos", *bank_demos(ctx),
            "--stride", str(w["stride"]), "--window", str(w["window"]), "--percentile",
            str(w["percentile"]), "--floor", str(w["floor"]), "--bank-out",
            rel(ctx.it() / "waypoint_bank")]


def is_jepa(model: dict) -> bool:
    p = model.get("refit_report")
    return bool(p) and Path(p).is_file() and jload(p).get("train_objective") == "jepa"


def export_env(ctx: Ctx, model: dict) -> dict | None:
    """JEPA_ENV for a JEPA checkpoint (a dry-run placeholder: seed_model's objective, which
    refits inherit), else None."""
    sm = ctx.cfg.get("seed_model")
    if "refit_report" not in model and sm:
        model = {"refit_report": sm["refit_report"]}
    return dict(JEPA_ENV) if is_jepa(model) and not all(
        os.environ.get(k) for k in JEPA_ENV) else None


def show_env(cmd, env, cwd=REPO):
    show([*(f"{k}={v}" for k, v in (env or {}).items()), *cmd], cwd)


def prepare_cand(ctx: Ctx) -> CandCtx:
    """iter<i>/cand/ with theta_i's model.json and box_i (keep_best)."""
    c = CandCtx(ctx)
    c.it().mkdir(parents=True, exist_ok=True)
    jdump(c.it() / "model.json", model_of(ctx, ctx.i))
    (c.it() / f"box_{ctx.i}.json").write_bytes((ctx.it() / f"box_{ctx.i}.json").read_bytes())
    return c


def export_candidate(ctx: Ctx, dry: bool):
    """keep_best `export`: theta_i -> iter<i>/cand/ when its eval is due."""
    if not eval_due(ctx):
        print(f"  keep_best: theta_{ctx.i} is not evaluated at iteration {ctx.i} (eval every "
              f"{ctx.cfg['eval_every']}): no candidate export; `deploy` exports the deployed "
              "model")
        return {"status": "skipped"}
    if dry:
        mp, sm = ctx.it() / "model.json", ctx.cfg.get("seed_model")
        print(f"  keep_best: candidate theta_{ctx.i} -> {rel(CandCtx(ctx).it())}")
        return st_export(CandCtx(ctx), True, jload(mp) if mp.is_file() else {
            "checkpoint": sm["checkpoint"] if ctx.i == 0 and sm else f"<theta_{ctx.i}>",
            "id": f"theta_{ctx.i}", "letter": "?"})
    return st_export(prepare_cand(ctx), False)


def st_export(ctx: Ctx, dry: bool, model: dict | None = None):
    """``model``: export this model (keep_best `deploy`) instead of theta_i."""
    if keep_best(ctx.cfg) and model is None and not isinstance(ctx, CandCtx):
        return export_candidate(ctx, dry)
    gep = REPO / ctx.cfg["goal_episode"]
    gf = goal_frames(gep)
    box = ctx.it() / f"box_{ctx.i}.json"
    files_txt = ctx.it() / "export_files.txt"
    if dry:
        mp = ctx.it() / "model.json"
        model = model or (jload(mp) if mp.is_file() else {"checkpoint": "<theta_i>", "id": "V11?",
                                                          "letter": "?"})
        dep = ctx.lout() / f"deploy_{ctx.cfg['model_id_prefix'].lower()}{model.get('letter', '?')}"
        env = export_env(ctx, model)
        for c in export_cmds(ctx, model, dep, files_txt, box, gf):
            show_env(c, env if c[0] == PY_LCS else None, LCS if c[0] == PY_LCS else REPO)
        if ctx.cfg["stage_tol"]:
            c = stage_tol_cmd(ctx, dep / "deploy.npz")
            show(c[:c.index("--demos") + 1] + [f"<{len(stage_demos(ctx))} seed demos>"]
                 + c[c.index("--out"):])
        if ctx.cfg["waypoints"]:
            c = waypoints_cmd(ctx, dep / "deploy.npz")
            show(c[:c.index("--demos") + 1] + [f"<{len(stage_demos(ctx))} seed demos>"]
                 + c[c.index("--goal-demo"):])
        if ctx.cfg["waypoint_bank"] is not None:
            c = bank_cmd(ctx, dep / "deploy.npz")
            show(c[:c.index("--demos") + 1] + [f"<{len(stage_demos(ctx))} seed demos>",
                                               "--goal-demos", f"<{len(bank_demos(ctx))}>"]
                 + c[c.index("--stride"):])
        print(f"  goal: {rel(gep)} frames {gf['frames']} ({gf['labels']})")
        return {}
    model = model or model_of(ctx, ctx.i)
    env = export_env(ctx, model)
    dep = ctx.lout() / f"deploy_{ctx.cfg['model_id_prefix'].lower()}{model['letter']}"
    buf = jload(buffer_in(ctx, ctx.i))
    files_txt.write_text("\n".join(r["episode"] for r in buf["rollouts"]) + "\n")
    cache, exp, dec, chk = export_cmds(ctx, model, dep, files_txt, box, gf)
    log = ctx.logs() / "export.log"
    t0 = time.time()
    if not (ctx.lout() / "cache_buffer.npz").is_file() and run(cache, log, cwd=LCS, ctx=ctx,
                                                               step="export", env=env):
        raise SystemExit("encode_latents_cache failed")
    clog = ctx.logs() / "check_latent_encoder.log"
    tries, check = [], None
    # c1's float32 guards (exporter reference z, check E2: 1e-6 x max|z|) sit on the batch-order
    # noise of these models; --seed only redraws the reference frames, deploy arrays are the same
    n_seeds = ctx.cfg["export_seeds"]
    for seed in range(n_seeds):
        if not (dep / "deploy.npz").is_file():
            if dep.exists():
                move_aside([dep])
            if run(exp + (["--seed", str(seed)] if n_seeds > 1 else []), log, cwd=LCS,
                   ctx=ctx, step="export", env=env):
                tail = log.read_text().split("\n# ")[-1].strip().splitlines()[-1:]
                tries.append({"seed": seed, "export": "FAIL", "error": tail})
                continue
        rc = run(chk, clog)
        if rc == 0 and "ALL LATENT ENCODER CHECKS PASSED" in clog.read_text().split("\n# ")[-1]:
            check = "PASS"
            tries.append({"seed": seed, "export": "ok", "check": "PASS"})
            break
        fails = [ln for ln in clog.read_text().split("\n# ")[-1].splitlines() if "[FAIL]" in ln]
        tries.append({"seed": seed, "export": "ok", "check": "FAIL", "error": fails})
        if ctx.smoke:
            check = "FAIL (smoke: continued; a 2-epoch model)"
            print(f"  check_latent_encoder FAILED (see {rel(clog)}); smoke continues")
            break
        move_aside([dep])
    if check is None:
        raise SystemExit(f"export / check_latent_encoder failed for seeds 0..{n_seeds - 1}"
                         f": {tries}")
    if not (dep / "decoder.npz").is_file() and run(dec, log, cwd=LCS, ctx=ctx, step="export",
                                                   env=env):
        raise SystemExit("decoder export failed")
    gpu_s = time.time() - t0
    with np.load(ctx.lout() / "cache_buffer.npz") as c:
        z = np.asarray(c["z_t"], dtype=np.float64)
    with np.load(dep / "deploy.npz", allow_pickle=True) as d:
        z_std, g1, g2 = d["z_std"], d["z_goal_stage1"], d["z_goal_stage2"]
        tol = float(d["goal_tol_whitened"])
        gsrc, gfr = str(d["goal_source"]), d["goal_frames"].tolist()
        ub, lb = d["u_ub"], d["u_lb"]
    bx = jload(box)
    assert np.allclose(lb, bx["u_lb"]) and np.allclose(ub, bx["u_ub"]), "deploy box != box_i"
    goal = {**gf, "deploy_goal_frames": gfr, "goal_source": gsrc, "goal_tol_whitened": tol,
            "stage_dist_whitened": float(np.linalg.norm((g2 - g1) / z_std)),
            "z_goal_stage1": g1.tolist(), "z_goal_stage2": g2.tolist(),
            "z_std_cache_vs_deploy_max_rel": float(np.max(np.abs(z.std(0) - z_std) / z_std)),
            "cache_tuples": len(z), "model": model["id"]}
    if ctx.cfg["stage_tol"]:
        if run(stage_tol_cmd(ctx, dep / "deploy.npz"), ctx.logs() / "stage_tols.log"):
            raise SystemExit("latent_stage_tols.py failed")
        stt = jload(ctx.it() / "stage_tols.json")
        goal.update(tol_1=stt["tol"][0], tol_2=stt["tol"][1], stage_tols=rel(
            ctx.it() / "stage_tols.json"), stage_tol_rule=ctx.cfg["stage_tol"],
            n_latch_inside_tol1=stt["n_latch_inside_tol1"],
            n_streak_in_window=stt["n_streak_in_window"],
            per_demo=[{k: r[k] for k in ("episode", "latch", "outcome", "d_latch",
                                         "d_window_min", "frames_inside_tol1")}
                      for r in stt["per_demo"]])
    if ctx.cfg["waypoints"]:
        if run(waypoints_cmd(ctx, dep / "deploy.npz"), ctx.logs() / "latent_waypoints.log"):
            raise SystemExit("latent_waypoints.py failed")
        goal.update(latent_waypoints=rel(ctx.it() / "latent_waypoints.npz"),
                    latent_waypoints_sha256=sha(ctx.it() / "latent_waypoints.npz"),
                    waypoints_rule=ctx.cfg["waypoints"])
    if ctx.cfg["waypoint_bank"] is not None:
        bj = ctx.it() / "waypoint_bank" / "bank.json"
        if run(bank_cmd(ctx, dep / "deploy.npz"), ctx.logs() / "waypoint_bank.log"):
            raise SystemExit("latent_waypoints.py --bank-out failed")
        goal.update(waypoint_bank=rel(bj), waypoint_bank_sha256=sha(bj),
                    waypoint_bank_rule=ctx.cfg["waypoint_bank"],
                    waypoint_bank_n=len(jload(bj)["entries"]))
    jdump(ctx.it() / "goal.json", goal)
    deploy = {"model": model["id"], "iter": ctx.i, "deploy_dir": str(dep),
              "deploy": str(dep / "deploy.npz"), "deploy_sha256": sha(dep / "deploy.npz"),
              "decoder": str(dep / "decoder.npz"), "box_id": f"ub-tr{ctx.i}",
              "box_json": str(box), "goal_json": str(ctx.it() / "goal.json"),
              "cache": str(ctx.lout() / "cache_buffer.npz"), "check_latent_encoder": check,
              "export_attempts": tries,
              "gpu_s": gpu_s, **({"env": env} if env else {})}
    jdump(ctx.it() / "deploy.json", deploy)
    print(f"  export {model['id']} ub-tr{ctx.i} -> {dep}; goal frames {gfr}, stage distance "
          f"{goal['stage_dist_whitened']:.2f} (tol {tol:.2f}); encoder check {check}"
          + (f"; tol_1 {goal['tol_1']:.3f} (latch inside {goal['n_latch_inside_tol1']}/"
             f"{len(goal['per_demo'])})" if "tol_1" in goal else ""))
    return {"outputs": [str(ctx.it() / "deploy.json"), str(dep / "deploy.npz")]}


# ---------------------------------------------------------------- harness


def mppi_yaml(ctx: Ctx, path: Path, noise: float, seed: int, floors=None, dry=False) -> Path:
    base_p = REPO / ctx.cfg["mppi_config"]
    doc = yaml.safe_load(base_p.read_text())
    doc.update(action_noise_frac=float(noise),
               stage_max_durations_s=list(ctx.cfg["stage_max_durations_s"]), seed=int(seed))
    doc.update(floors or {})
    extra = []
    if ctx.cfg["stage_tol"]:
        gj = ctx.it() / "goal.json"
        tol1 = jload(gj)["tol_1"] if gj.is_file() else float("nan")
        doc["stage_goal_tols"] = [float(tol1), -1.0]  # stage 2 ends on its timeout (P4)
        extra.append("stage_goal_tols [tol_1, -1]")
    if ctx.cfg["stage_reach_streak"]:
        doc["stage_reach_streak"] = int(ctx.cfg["stage_reach_streak"])
        extra.append("stage_reach_streak")
    if ctx.cfg["rot_envelope_decision"]:
        doc["rot_envelope_deg"] = dict(jload(REPO / ctx.cfg["rot_envelope_decision"])[
            "rot_envelope_deg"])
        extra.append(f"rot_envelope_deg ({ctx.cfg['rot_envelope_decision']})")
    if ctx.cfg["mppi_H"]:
        doc["H"] = int(ctx.cfg["mppi_H"])
        extra.append("H")
    if ctx.cfg["demo_prior"]:
        doc.update(prior_alpha=float(ctx.cfg["demo_prior"]["alpha"]),
                   prior_weight=float(ctx.cfg["demo_prior"]["weight"]))
        extra.append("prior_alpha, prior_weight (demo_prior)")
        if ctx.cfg["demo_prior"].get("sync") is not None:
            doc["prior_sync"] = ctx.cfg["demo_prior"]["sync"]
            extra.append("prior_sync (demo_prior.sync)")
        if ctx.cfg["demo_prior"].get("mode", "demo") == "policy":
            doc.update(prior_mode="policy", bc_prior=str(REPO / ctx.cfg["demo_prior"]["bc_prior"]))
            extra.append("prior_mode, bc_prior (demo_prior.mode policy)")
    w = ctx.cfg["waypoints"] or {}
    if w.get("skip_ahead"):
        doc["stage_skip_ahead"] = True
        extra.append("stage_skip_ahead (waypoints.skip_ahead)")
    if w.get("no_timeout_into") is not None:
        doc["stage_no_timeout_into"] = w["no_timeout_into"]
        extra.append("stage_no_timeout_into (waypoints.no_timeout_into)")
    if w.get("start") is not None:
        doc["stage_start"] = w["start"]
        extra.append("stage_start (waypoints.start)")
    if ctx.cfg["mppi_overshoot"]:
        ov = copy.deepcopy(ctx.cfg["mppi_overshoot"])
        if ov.get("seat_tol") == "tol_2":
            gj = ctx.it() / "goal.json"
            ov["seat_tol"] = float(jload(gj)["tol_2"]) if gj.is_file() else float("nan")
        doc["overshoot"] = ov
        extra.append("overshoot (mppi_overshoot)")
    if ctx.cfg["support_box"] and dry and not buffer_in(ctx, ctx.i).is_file():
        print(f"  {path.name}: support box of {rel(buffer_in(ctx, ctx.i))} (not built yet)")
    elif ctx.cfg["support_box"]:
        sb = support_box(ctx, dry)
        doc["support_box"] = {k: sb[k] for k in ("lo", "hi", "weight", "hard_margin_m")}
        extra.append(f"support_box ({rel(ctx.it() / 'support_box.json')})")
        print(f"  {path.name}: {sbox_line(sb)}")
    if ctx.cfg["goal_set"] is not None:  # {} = the defaults
        g = goal_set_npz(ctx, dry)
        doc.update(goal_set=g["path"], goal_set_tau=g["tau"])
        extra.append(f"goal_set, goal_set_tau ({rel(g['path'])})")
        print(f"  {path.name}: goal set {g['n']} seated latents {g['kinds']} of {g['buffer']}, "
              f"tau {g['tau']:g}")
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel(base_p)} (sha256 {sha(base_p)[:12]}) + action_noise_frac, "
                        f"stage_max_durations_s, seed{', floors' if floors else ''}"
                        + "".join(", " + e for e in extra)
                        + " (realfirst_loop.py)\n" + yaml.safe_dump(doc, sort_keys=False))
    return path


def c3plus_params(ctx: Ctx, run: str, floors=None, stages: str = "all", dry=False) -> str:
    """controller c3plus: c3plus_params.py files of deploy_i for harness run ``run`` in the
    worktree + iter<i>/c3plus_<run>.json; returns the worktree-relative params (--params)."""
    import c3plus_params as c3p

    c = {**C3PLUS_DEFAULTS, **(ctx.cfg.get("c3plus") or {})}
    variant = "seat" if stages == "last" else c["variant"]
    out_rel = f"{c['archive']}/{ctx.root.name}/iter{ctx.i}/{run}"
    dj, wp = ctx.it() / "deploy.json", ctx.it() / "latent_waypoints.npz"
    if not (dj.is_file() and wp.is_file()):
        p = f"{out_rel}/round_belt_controller_params_learned_eval_<model>_{variant}.yaml"
        print(f"  c3plus {run}: {variant} -> {p} (after export)")
        return p
    d = jload(dj)
    base = yaml.safe_load((REPO / ctx.cfg["mppi_config"]).read_text())
    fl = {k: float(base[k]) for k in ("z_min_franka", "z_min_ur")}
    fl.update(floors or {})
    m = c3p.build(Path(d["deploy_dir"]), wp, Path(d["box_json"]), Path(d["goal_json"]),
                  d["model"].lower(), variant, out_rel, floors=fl,
                  stage_max_durations_s=ctx.cfg["stage_max_durations_s"],
                  costs={k: c[k] for k in ("w_q", "w_r", "admm_iter")},
                  box=support_box(ctx, True) if c["box_z"] else None,
                  demo_offset=int(ctx.cfg["mppi_demo_offset_frames"] or 0), dry=dry)
    p = m["files"]["params"]
    print(f"  c3plus {run}: {variant} {p}: stage frames "
          f"{m['mapping'].get('stage_frames') or m['mapping'].get('traj_frames')}, tols "
          f"{np.round(m['stage_goal_tols'], 4).tolist()}, durations {m['stage_max_durations_s']}"
          f", ee z {m['ee_constraints']}")
    if run == "collect":
        off = [k for k, on in (("action noise", ctx.cfg["action_noise_frac"]),
                               ("clearance", ctx.cfg["mppi_clearance"]),
                               ("support box x/y" if c["box_z"] else "support box",
                                ctx.cfg["support_box"]), ("rotation envelope", True)) if on]
        print(f"  c3plus: MPPI-only, not applied: {', '.join(off)}; H {N_C3} knots (MPPI "
              f"{ctx.cfg['mppi_H'] or 'yaml'})")
    if not dry:
        chk = c3p.check_params(p)
        jdump(ctx.it() / f"c3plus_{run}.json", {**m, "check": chk})
        if not chk["pass"]:
            raise SystemExit(f"c3plus params {p}: {chk['errors']}")
    return p


def harness_cmd(ctx: Ctx, lcm: str, deploy: str, cfg: Path, starts: str, variants, repeats: int,
                out: Path, label: str, scripted: bool, reset_seed: int, stages: str = "all",
                extra=()):
    """``stages``: --mppi-stages; ``extra``: appended last (collect: the --prefix-* flags);
    controller c3plus: ``cfg`` = the worktree-relative C3+ params (c3plus_params)."""
    if c3_on(ctx.cfg):
        cmd = [sys.executable, "scripts/lcs/eval_learned_mpc.py", "--mode", "learned",
               "--lcm-url", lcm, "--record", "--max-episode-s", str(ctx.cfg["max_episode_s"]),
               "--settle-s", "0.5", "--pace", "1", "--params", str(cfg), "--deploy", deploy,
               "--demo-goals", "none", "--start-states", starts, "--repeats", str(repeats),
               "--belt-seg", "--cloud-noise", ctx.cfg["noise_preset"], "--label", label,
               "--out", rel(out)]
        if variants:
            cmd += ["--variants", variants]
        if scripted:
            cmd += ["--reset", "scripted", "--reset-spec", *ctx.cfg["reset_spec"],
                    f"seed={reset_seed}"]
        return cmd + list(extra)
    goal = "waypoints" if ctx.cfg["waypoints"] else "deploy"
    cmd = [sys.executable, "scripts/lcs/eval_learned_mpc.py", "--mode", "learned", "--lcm-url",
           lcm, "--record", "--max-episode-s", str(ctx.cfg["max_episode_s"]), "--settle-s",
           "0.5", "--pace", "1", "--deploy", deploy, "--demo-goals", "none",
           "--start-states", starts, "--repeats", str(repeats), "--belt-seg", "--cloud-noise",
           ctx.cfg["noise_preset"], "--controller", "mppi", "--mppi-config", cfg,
           "--mppi-goal", goal, "--mppi-stages", stages, "--mppi-max-duration-s",
           str(ctx.cfg["stage_max_durations_s"][-1]), "--mppi-debug", "--label", label,
           "--out", rel(out)]
    if ctx.cfg["mppi_clearance"]:  # per-run clearance model at the run's first start
        cmd.append("--mppi-clearance")
        for k in ("ur", "franka"):
            v = ctx.cfg[f"mppi_clearance_{k}_min_mm"]
            if v is not None:
                cmd += [f"--mppi-clearance-{k}-min-mm", str(v)]
    if variants:
        cmd += ["--variants", variants]
    if scripted:
        cmd += ["--reset", "scripted", "--reset-spec", *ctx.cfg["reset_spec"],
                f"seed={reset_seed}"]
    if ctx.cfg["waypoints"]:  # W1-B's harness flags (PKG-20261006-cmp-mppi-transport)
        cmd += ["--mppi-waypoints", rel(ctx.it() / "latent_waypoints.npz")]
    if ctx.cfg["goal_mode"]:
        cmd += ["--mppi-goal-mode", ctx.cfg["goal_mode"]]
    if ctx.cfg["demo_prior"]:
        cmd += ["--mppi-demo-actions", "auto"]
    if ctx.cfg["mppi_demo_offset_frames"] is not None:  # a prefix overrides it per episode
        cmd += ["--mppi-demo-offset-frames", str(int(ctx.cfg["mppi_demo_offset_frames"]))]
    if ctx.cfg["waypoint_bank"] is not None:
        cmd += ["--mppi-waypoint-bank", rel(ctx.it() / "waypoint_bank" / "bank.json")]
        if int(ctx.cfg["waypoint_bank"].get("knn", 1)) != 1:
            cmd += ["--mppi-bank-knn", str(int(ctx.cfg["waypoint_bank"]["knn"]))]
    return cmd + list(extra)


def harness(ctx: Ctx, cmd, out: Path, n_want: int, log: Path, step: str) -> dict:
    """One eval_learned_mpc run; skipped when its index.json has every episode, a partial run
    (crash) is moved aside and redone."""
    idx = out / "index.json"
    if idx.is_file():
        if len(jload(idx).get("episodes", [])) >= n_want and "summary" in jload(idx):
            return {"skipped": True, "rc": 0, "wall_s": None}
        move_aside([out])
    t0 = time.time()
    rc = run(cmd, log, ctx=ctx, step=step)
    return {"skipped": False, "rc": rc, "wall_s": time.time() - t0}


def quat_angle_deg(q0, q) -> np.ndarray:
    d = np.abs(np.sum(np.asarray(q0)[None, :] * np.asarray(q), axis=1))
    return np.degrees(2 * np.arccos(np.clip(d, -1.0, 1.0)))


_C3_SOLVE = re.compile(r"\[learned-mpc\] t=\S+ stage=\d+ .*solve=([0-9.]+)ms")


def solves_ms(out: Path, row: dict) -> list[float]:
    """MPPI: every solve (JSONL); C++ controller: its ~1 Hz ``solve=`` log samples."""
    p = out / row.get("mppi", {}).get("solve_log", "")
    if not row.get("mppi") and row.get("controller_log"):
        lg = out / row["controller_log"]
        return ([float(x) for x in _C3_SOLVE.findall(lg.read_text(errors="replace"))]
                if lg.is_file() else [])
    if not row.get("mppi") or not p.is_file():
        return []
    return [json.loads(x)["solve_ms"] for x in p.read_text().splitlines() if x.strip()
            and json.loads(x).get("solve_ms") is not None]


def episode_row(k: int, out: Path, row: dict) -> dict:
    """attempts/eval row; GT columns (outcome, wrap, h) are report-only."""
    rb = row.get("reset_before") or {}
    pre = row.get("prefix")
    base = {"k": k, "start_id": row["start_state"]["id"], "repeat": row.get("repeat"),
            "from_reset": bool(row["start_state"].get("from_reset")),
            "jitter": row["start_state"].get("jitter"),
            "reset_before": rb.get("kind"), "reset_ok": rb.get("ok"),
            "reset_reason": rb.get("reason"), "reset_s": rb.get("duration_s"),
            "status": row.get("status"), "reason": row.get("reason"),
            "outcome_gt": row.get("outcome")}
    if pre is not None:  # V12 prefix rows only (v1.1 rows unchanged)
        base.update(prefix_frac=pre["frac"], prefix_ok=pre["ok"], prefix_s=pre["duration_s"],
                    prefix_end_frame=pre["end_frame"])
    if row.get("file") is None:
        return {**base, "episode": None, "duration_s": 0.0, "n_frames": 0, "n_mpc_frames": 0,
                **({"n_prefix_frames": 0} if pre is not None else {}), "unsafe": True}
    path = out / row["file"]
    with np.load(path, allow_pickle=True) as d:
        src = np.asarray(d["sim_cmd_source"])
        cs = np.asarray(d["sim_cloud_success"]) if "sim_cloud_success" in d.files else None
        ch = np.asarray(d["sim_cloud_h_mm"]) if "sim_cloud_h_mm" in d.files else None
        cw = np.asarray(d["sim_cloud_wrap_deg"]) if "sim_cloud_wrap_deg" in d.files else None
        ef, eu = np.asarray(d["sim_ee_franka"]), np.asarray(d["sim_ee_ur"])
        belt = np.asarray(d["pcd_belt"])[-1]
    tail = slice(-LAST_N, None)
    rf, ru = quat_angle_deg(ef[0, 3:7], ef[:, 3:7]), quat_angle_deg(eu[0, 3:7], eu[:, 3:7])
    mp = row.get("mppi") or {}
    grasp = row.get("grasp_ok_final")
    unsafe = (row.get("status") == "aborted" or grasp != [True, True]
              or bool(row.get("board_contact")) or float(rf.max()) > 25 or float(ru.max()) > 25)
    if pre is not None:
        base["n_prefix_frames"] = int((src == 3).sum())  # eval_learned_mpc.CMD_PREFIX
    return {**base, "episode": str(path.resolve()), "duration_s": row["duration_s"],
            "n_frames": len(src), "n_mpc_frames": int(np.isin(src, (0, 1)).sum()),
            "final_wrap_deg_gt": row.get("final_wrap_deg"),
            "final_h_mm_gt": row.get("final_h_median_mm"),
            "strict_gt_final": bool(row.get("final_wrap_deg", 0) >= 90
                                    and abs(row.get("final_h_median_mm", 99)) <= 1),
            "cloud_success": None if cs is None else bool(2 * int(cs[tail].sum())
                                                          > len(cs[tail])),
            "cloud_h_mm_last": None if ch is None else float(np.nanmedian(ch[tail])),
            "cloud_wrap_deg_last": None if cw is None else float(np.nanmedian(cw[tail])),
            "solve_ms": mp.get("solve_ms"), "solves": mp.get("solves"),
            "fallbacks": mp.get("fallbacks"), "plans_violating": mp.get("plans_violating"),
            "grasp_ok_final": grasp, "board_contact": row.get("board_contact"),
            "min_board_clearance_mm": row.get("min_board_clearance_mm"),
            "min_franka_tip_clearance_mm": row.get("min_franka_tip_clearance_mm"),
            "max_rot_deg": [float(rf.max()), float(ru.max())],
            "stage_ends": row.get("stage_ends"), "controller_exit": row.get("controller_exit"),
            "stage_0_how": (row.get("stage_ends") or [None])[0],
            "dist1_at_switch": (row.get("controller_events") or {}).get("stage_0_dist"),
            "t_switch_s": next((v for k, v in (row.get("controller_events") or {}).items()
                                if k in ("stage_0_reached", "stage_0_timeout")), None),
            "clear_min_mm": mp.get("clear_min_mm"),
            "clear_violating_plans": mp.get("clear_violating_plans"),
            "board_abort": is_board_abort(row),
            "final_belt_gt": belt, "unsafe": bool(unsafe)}


def is_board_abort(row: dict) -> bool:
    """Harness safety abort on board/plate contact (2F-85 clearance or Franka tip floor)."""
    r = str(row.get("reason") or "")
    return row.get("status") == "aborted" and r.startswith(("2F-85 board clearance",
                                                             "franka finger_tip"))


def demo_start_sets(ctx: Ctx) -> dict[str, set]:
    """prefix.demos keys ("<set>/<variant>") -> {start set dir under STARTS: {variant ids}}."""
    out: dict[str, set] = {}
    for key in ctx.cfg["prefix"]["demos"]:
        name, vid = key.rsplit("/", 1)
        s = next((c for c in (name, f"grasp_variants/{name}")
                  if (REPO / STARTS / c / "index.json").is_file()), None)
        if s is None:
            raise SystemExit(f"prefix.demos {key}: no start set {name} under {STARTS}")
        out.setdefault(s, set()).add(vid)
    return out


def pool_starts(ctx: Ctx, pool=None) -> list[dict]:
    """Start candidates of ``pool`` (default collection_pool); "demo_starts" = the prefix demos'
    starts. Eval-disjoint (sha)."""
    import onpolicy_pool as op

    bad = set(op.eval_start_shas().values())
    out = []
    sets = []
    for s in pool or ctx.cfg["collection_pool"]:
        if s == "demo_starts":
            sets += list(demo_start_sets(ctx).items())
        else:
            sets.append((s, None))
    for s, only in sets:
        index = jload(REPO / STARTS / s / "index.json")
        for r in index["variants"]:
            if not r.get("held") or r.get("file") is None:
                continue
            if only is not None and str(r["id"]) not in only:
                continue
            f = REPO / STARTS / s / r["file"]
            h = sha(f)
            if h in bad:
                raise SystemExit(f"collection start {s}/{r['id']} is an eval start (sha)")
            out.append({"start_set": s, "start_id": str(r["id"]), "file": rel(f), "sha256": h,
                        "key": f"{s}/{r['id']}"})
    out = list({c["key"]: c for c in out}.values())  # a set listed twice
    return sorted(out, key=lambda c: c["key"])


# ---------------------------------------------------------------- curriculum (V12)


def ladder_step(schedule, frac: float, successes: int, n: int, zero_streak: int,
                advance_when: int, retreat_after: int) -> tuple[float, int, str]:
    """``(frac, zero_streak, decision)`` for the next iteration after one at ``frac``: >=
    advance_when successes -> the next lower frac; retreat_after consecutive 0/n -> the next
    higher; clamped at both ends."""
    lad = sorted({float(x) for x in schedule}, reverse=True)
    j = lad.index(float(frac))
    if n == 0:
        return float(frac), int(zero_streak), "stay (no labelled rollout)"
    if successes >= advance_when:
        return (lad[j + 1], 0, "advance") if j + 1 < len(lad) else (lad[j], 0, "clamp (lowest)")
    if successes == 0:
        z = int(zero_streak) + 1
        if z < retreat_after:
            return lad[j], z, f"stay ({z} x 0/n)"
        return (lad[j - 1], 0, "retreat") if j > 0 else (lad[j], 0, "clamp (highest)")
    return lad[j], 0, "stay"


def prefix_on(cfg: dict) -> bool:
    return bool((cfg.get("prefix") or {}).get("demos"))


def curriculum(ctx: Ctx, i: int) -> dict | None:
    """Iteration i's prefix frac: iter<i>/curriculum.json once written (`collect`), else from
    iteration i-1's state + labels (start_frac without one); None when no prefix is set."""
    if not prefix_on(ctx.cfg):
        return None
    p = ctx.it(i) / "curriculum.json"
    if p.is_file():
        return jload(p)
    pc = ctx.cfg["prefix"]
    rule = {k: pc[k] for k in ("schedule", "advance_when", "retreat_after", "start_frac")}
    if not pc.get("schedule"):
        return {"iter": i, "frac": float(pc["frac"]), "zero_streak": 0, "decision": "fixed",
                "rule": rule}
    prev_p = ctx.it(i - 1) / "curriculum.json" if i > 0 else None
    if prev_p is None or not prev_p.is_file():
        return {"iter": i, "frac": float(pc["start_frac"]), "zero_streak": 0,
                "decision": "start", "rule": rule}
    prev = jload(prev_p)
    lab = ctx.it(i - 1) / "labels.json"
    if not lab.is_file():
        return {"iter": i, "frac": prev["frac"], "zero_streak": prev["zero_streak"],
                "decision": f"pending (no labels for iteration {i - 1})", "rule": rule}
    eps = jload(lab)["episodes"]
    s = sum(bool(e["success"]) for e in eps)
    frac, z, dec = ladder_step(pc["schedule"], prev["frac"], s, len(eps), prev["zero_streak"],
                               pc["advance_when"], pc["retreat_after"])
    return {"iter": i, "frac": frac, "zero_streak": z, "decision": dec, "rule": rule,
            "prev": {"iter": i - 1, "frac": prev["frac"], "successes": s, "n": len(eps),
                     "zero_streak": prev["zero_streak"]}}


def active_pool(cfg: dict, frac) -> list[str]:
    """collection_pool_rule on the current frac (None = no prefix = 0.0), else
    collection_pool."""
    rule = cfg.get("collection_pool_rule")
    if not rule:
        return list(cfg["collection_pool"])
    f = 0.0 if frac is None else float(frac)
    return list(rule["pool"] if f > float(rule["while_frac_gt"]) else rule["then_pool"])


def stages_for(cfg: dict, frac) -> str:
    """--mppi-stages of a collection run (auto: the seated goal only from pre_place_2 on)."""
    s = cfg["mppi_stages"]
    if s == "auto":
        return "last" if frac is not None and float(frac) >= 1.0 else "all"
    return s


def prefix_args(ctx: Ctx, start: dict, frac) -> list[str]:
    if frac is None or float(frac) <= 0.0:
        return []
    demos = ctx.cfg["prefix"]["demos"]
    key = next((k for k in demos if start["key"] == k or start["key"].endswith("/" + k)), None)
    if key is None:
        raise SystemExit(f"start {start['key']}: no prefix demo (prefix.demos {sorted(demos)})")
    return ["--prefix-demos", json.dumps({key: demos[key]}), "--prefix-frac", f"{float(frac):g}"]


def scheduled_start(ctx: Ctx) -> dict:
    """``collection_schedule`` entry i mod n, resolved like a pool start (eval-disjoint)."""
    import onpolicy_pool as op

    sched = ctx.cfg["collection_schedule"]
    j = ctx.i % len(sched)
    key = sched[j]
    name, vid = key.rsplit("/", 1)
    ip = REPO / STARTS / name / "index.json"
    r = next((r for r in jload(ip)["variants"] if str(r["id"]) == vid and r.get("held")
              and r.get("file") is not None), None) if ip.is_file() else None
    if r is None:
        raise SystemExit(f"collection_schedule {key}: no held start {vid} in {rel(ip)}")
    f = REPO / STARTS / name / r["file"]
    h = sha(f)
    if h in set(op.eval_start_shas().values()):
        raise SystemExit(f"collection start {key} is an eval start (sha)")
    s = {"start_set": name, "start_id": vid, "file": rel(f), "sha256": h, "key": key}
    used = {jload(ctx.it(k) / "collect_plan.json")["start"]["key"] for k in range(ctx.i)
            if (ctx.it(k) / "collect_plan.json").is_file()}
    return {"start": s, "seed": ctx.cfg["seed"] + ctx.i, "n_pool": len(sched),
            "n_unused": len([k for k in sched if k not in used]), "eval_disjoint": True,
            "schedule": {"index": j, "n": len(sched), "key": key}}


def draw_start(ctx: Ctx, frac=None) -> dict:
    if ctx.cfg["collection_schedule"]:
        return scheduled_start(ctx)
    pool_cfg = active_pool(ctx.cfg, frac)
    pool = pool_starts(ctx, pool_cfg)
    used = set()
    for j in range(ctx.i):
        p = ctx.it(j) / "collect_plan.json"
        if p.is_file():
            used.add(jload(p)["start"]["key"])
    fresh = [c for c in pool if c["key"] not in used] or pool
    rng = np.random.default_rng(ctx.cfg["seed"] + ctx.i)
    out = {"start": fresh[int(rng.integers(len(fresh)))], "seed": ctx.cfg["seed"] + ctx.i,
           "n_pool": len(pool), "n_unused": len([c for c in pool if c["key"] not in used]),
           "eval_disjoint": True}
    if frac is not None or ctx.cfg.get("collection_pool_rule"):
        out["pool"] = pool_cfg
    return out


def st_collect(ctx: Ctx, dry: bool):
    plan_p = ctx.it() / "collect_plan.json"
    cur = curriculum(ctx, ctx.i)
    frac = None if cur is None else cur["frac"]
    plan = jload(plan_p) if plan_p.is_file() else draw_start(ctx, frac)
    s = plan["start"]
    seed = ctx.cfg["seed"] + ctx.i
    cfg = mppi_yaml(ctx, ctx.it() / "mppi_collect.yaml", ctx.cfg["action_noise_frac"], seed,
                    dry=dry)
    c3 = c3plus_params(ctx, "collect", stages=stages_for(ctx.cfg, frac), dry=dry) \
        if c3_on(ctx.cfg) else None  # no action noise: C3+ is deterministic
    out = ctx.it() / "collect"
    dj = ctx.it() / "deploy.json"
    deploy = jload(dj)["deploy"] if dj.is_file() else "<deploy_i>"
    n = ctx.cfg["new_per_iter"]
    cmd = harness_cmd(ctx, ctx.url_collect, deploy, c3 or cfg, f"{STARTS}/{s['start_set']}",
                      s["start_id"], n, out, f"V11_iter{ctx.i}_collect", True, seed,
                      stages=stages_for(ctx.cfg, frac), extra=prefix_args(ctx, s, frac))
    if dry:
        show(cmd)
        print(f"  start {s['key']} (pool {plan['n_pool']}, unused {plan['n_unused']}"
              + (f"; schedule entry {plan['schedule']['index']}/{plan['schedule']['n']}"
                 if "schedule" in plan else "") + f"); {n} rollouts chained by scripted resets")
        if cur is not None:
            print(f"  curriculum: prefix frac {frac:g} ({cur['decision']}), pool "
                  f"{plan.get('pool')}, --mppi-stages {stages_for(ctx.cfg, frac)}")
        return {}
    jdump(plan_p, plan)
    if cur is not None and not (ctx.it() / "curriculum.json").is_file():
        jdump(ctx.it() / "curriculum.json", {**cur, "written": now()})
    r = harness(ctx, cmd, out, n, ctx.logs() / "collect.log", "collect")
    if r["rc"] != 0:
        raise SystemExit(f"collection harness exit {r['rc']} (see {rel(ctx.logs())}/collect.log)")
    idx = jload(out / "index.json")
    rows = [episode_row(k, out, e) for k, e in enumerate(idx["episodes"])]
    for x in rows:
        x.pop("final_belt_gt", None)
    att = {"iter": ctx.i, "url": ctx.url_collect, "start": s, "deploy": deploy,
           "mppi_config": str(cfg), "n": len(rows), "harness": r, "attempts": rows,
           "index_budget": idx.get("budget"), "wall_s": idx["summary"]["wall_s"],
           **({"controller": "c3plus", "c3plus_params": c3} if c3 else {})}
    if cur is not None:
        att.update(prefix_frac=frac, prefix_ok=sum(bool(x.get("prefix_ok")) for x in rows),
                   curriculum=cur, mppi_stages=stages_for(ctx.cfg, frac))
    jdump(ctx.it() / "attempts.json", att)
    write_budget(ctx)
    print(f"  collect {s['key']}: {len(rows)} rollouts, GT outcomes "
          f"{[x['outcome_gt'] for x in rows]}, resets {[x['reset_before'] for x in rows]}")
    return {"outputs": [str(ctx.it() / "attempts.json"), str(out / "index.json")]}


# ---------------------------------------------------------------- label / buffer


def st_label(ctx: Ctx, dry: bool):
    out = ctx.it() / "labels.json"
    if dry:
        print(f"  would run: {sys.executable} scripts/lcs/human_label.py --episodes "
              f"<{rel(ctx.it())}/collect episodes> --rule {ctx.cfg['label_rule']} --out {rel(out)}")
        return {}
    att = jload(ctx.it() / "attempts.json")
    eps = [a["episode"] for a in att["attempts"] if a["episode"]]
    if out.exists():
        move_aside([out])
    if eps:
        cmd = [sys.executable, "scripts/lcs/human_label.py", "--episodes", *eps, "--rule",
               ctx.cfg["label_rule"], "--out", rel(out)]
        if run(cmd, ctx.logs() / "label.log"):
            raise SystemExit("human_label.py failed")
    else:
        jdump(out, {"rule": ctx.cfg["label_rule"], "episodes": [], "note": "no episode"})
    lab = jload(out)
    print(f"  labels ({ctx.cfg['label_rule']}): "
          f"{sum(e['success'] for e in lab['episodes'])}/{len(eps)}")
    write_budget(ctx)
    return {"outputs": [str(out)]}


def fifo(rows: list[dict], cap: int, exempt_until: int) -> tuple[list[dict], list[dict]]:
    """Oldest first out; successes are exempt while the buffer holds <= exempt_until of them."""
    rows, evicted = list(rows), []
    while len(rows) > cap:
        n_succ = sum(r["label_human"] for r in rows)
        pick = next((r for r in rows if not r["label_human"]), None) if n_succ <= exempt_until \
            else None
        pick = pick or rows[0]
        rows.remove(pick)
        evicted.append(pick)
    return rows, evicted


def st_buffer(ctx: Ctx, dry: bool):
    if dry:
        bp = ctx.cfg["buffer_policy"]
        rule = (f"buffer_policy.apply({rel(buffer_in(ctx, ctx.i))} + labelled rollouts, {bp})"
                if bp else
                f"fifo({rel(buffer_in(ctx, ctx.i))} + labelled rollouts, max "
                f"{ctx.cfg['buffer_max']}, successes exempt until > "
                f"{ctx.cfg['success_exempt_until']})")
        print(f"  B_{ctx.i + 1} = {rule} -> {rel(ctx.it())}/buffer.json + "
              f"{rel(ctx.root)}/buffer.json; deleted-keys test: {PY_LCS} {LOADER_TEST} <file> x3")
        if ctx.cfg["anchor"]:
            an = {**ANCHOR_DEFAULTS, **ctx.cfg["anchor"]}
            print(f"  anchors {an['kinds']}: never evicted, outside cap {bp['cap']}"
                  + ("" if an["frac"] is None else
                     f"; weight -> anchor frame share {anchor_frac_at(ctx, ctx.i + 1):.3f} of "
                     f"the refit at iteration {ctx.i + 1}"))
        if ctx.cfg.get("onpolicy_weighting") is not None:
            from task_common import onpolicy_weighting

            print(f"  onpolicy_weighting {onpolicy_weighting.rule_of(ctx.cfg['onpolicy_weighting'])}"
                  f": new MPC rows' frame_weights vs {rel(ctx.it())}/latent_waypoints.npz")
        return {}
    att = jload(ctx.it() / "attempts.json")
    lab = {e["episode"]: e for e in jload(ctx.it() / "labels.json")["episodes"]}
    prev = jload(buffer_in(ctx, ctx.i))
    new = []
    for a in att["attempts"]:
        if not a["episode"]:
            continue
        lb = lab[a["episode"]]
        new.append({"id": f"mpc_i{ctx.i}_{a['k']}", "iter": ctx.i, "kind": "mpc",
                    **({"status": a["status"], "reason": a["reason"]}
                       if ctx.cfg["buffer_policy"] else {}),
                    "episode": a["episode"], "label_human": bool(lb["success"]),
                    "label_human_detail": {k: lb[k] for k in ("success", "rule", "source",
                                                              "wrap_deg", "h_mm", "n_pass",
                                                              "last_n")},
                    "label_strict": bool(lb["labels"]["strict"]["success"]),
                    "cloud_success": a["cloud_success"], "cloud_success_role": "monitor only",
                    "start": {"set": att["start"]["start_set"], "id": a["start_id"],
                              "file": att["start"]["file"], "from_reset": a["from_reset"],
                              "jitter": a["jitter"]},
                    "reset_before": a["reset_before"], "n_frames": a["n_frames"],
                    "n_mpc_frames": a["n_mpc_frames"], "outcome_gt": a["outcome_gt"],
                    **({"prefix_frac": a["prefix_frac"], "n_prefix_frames": a["n_prefix_frames"]}
                       if "prefix_frac" in a else {})})
    ow = ctx.cfg.get("onpolicy_weighting")
    if ow is not None:
        from task_common import onpolicy_weighting

        new = onpolicy_rows(ctx, new)
    bp = ctx.cfg["buffer_policy"]
    policy = anchor = None
    if bp:
        from task_common import buffer_policy

        if bp["short_abort_weight"] != buffer_policy.SHORT_ABORT_WEIGHT:
            raise SystemExit("buffer_policy.short_abort_weight must match buffer_policy.py")
        an = ctx.cfg["anchor"] and {**ANCHOR_DEFAULTS, **ctx.cfg["anchor"]}
        rows, evicted, policy = buffer_policy.apply(
            prev["rollouts"] + new, cap=bp["cap"], short_abort_max=bp["short_abort_max"],
            min_failures=bp["min_failures"], **({"anchor_kinds": an["kinds"]} if an else {}))
        if an and an["frac"] is not None:
            rows, anchor = buffer_policy.anchor_weights(
                rows, an["kinds"], anchor_frac_at(ctx, ctx.i + 1),
                {r["episode"]: n_frames(r["episode"]) for r in rows} if ow is None else
                {r["episode"]: onpolicy_weighting.effective_frames(r, n_frames(r["episode"]))
                 for r in rows})
            anchor.update(refit_iter=ctx.i + 1, rule=an)
    else:
        rows, evicted = fifo(prev["rollouts"] + new, ctx.cfg["buffer_max"],
                             ctx.cfg["success_exempt_until"])
    buf = {"version": 2, "iter": ctx.i, "max": bp["cap"] if bp else ctx.cfg["buffer_max"],
           "n": len(rows), "n_success": sum(r["label_human"] for r in rows),
           "source": f"realfirst_loop.py iteration {ctx.i}: {rel(buffer_in(ctx, ctx.i))} + "
                     f"{len(new)} MPC rollouts", "evicted": [r["id"] for r in evicted],
           "rollouts": rows}
    if policy is not None:
        buf.update(policy=policy, n_short_abort=sum(r["short_abort"] for r in rows),
                   weight_sum=sum(r["weight"] for r in rows))
    if anchor is not None:
        buf["anchor"] = anchor
    if ow is not None:
        buf["onpolicy_weighting"] = onpolicy_stats(ctx, new, rows)
    jdump(ctx.it() / "buffer.json", buf)
    jdump(ctx.root / "buffer.json", buf)
    pick = [next((r for r in rows if r["kind"] == k), None) for k in ("demo", "excitation")]
    ext = next((r for r in rows if str(r.get("id", "")).startswith("ext_")), None)
    pick[0] = ext or pick[0]  # v1.1: an ext demo; v1 has none
    pick.append(next((r for r in reversed(rows) if r["kind"] == "mpc"), None))
    tests = []
    for r in [p for p in pick if p]:
        lg = ctx.logs() / "deleted_keys.log"
        rc = run([PY_LCS, LOADER_TEST, r["episode"]], lg, cwd=LCS)
        tail = lg.read_text().split("\n# ")[-1]
        tests.append({"episode": r["episode"], "kind": r["kind"], "rc": rc,
                      "pass": rc == 0 and tail.rstrip().endswith("PASS")})
    allow = {"loader_keys": LOADER_KEYS, "buffer_keys": ["episode", "label_human", "kind"],
             "trust_keys": ["actions", "sim_cmd_source"],
             "export_keys": "loader_keys (load_belt_seg_episode) + pcd_belt_seg of the goal "
                            "episode; goal frame choice: the demo's sim_phase/sim_meta (operator "
                            "phase script)",
             "sim_only_never_read_by_refit_export_trust": SIM_ONLY,
             "label_only": ["sim_belt_xyz", "sim_pulley_large_pose"],
             "guard": "lcs_learning test_belt_seg_loader.py (deleted-keys) on 3 buffer files",
             "deleted_keys_tests": tests}
    jdump(ctx.it() / "allowlist.json", allow)
    if not all(t["pass"] for t in tests):
        raise SystemExit(f"deleted-keys loader test failed: {tests}")
    print(f"  buffer {buf['n']} ({buf['n_success']} successes), +{len(new)}, evicted "
          f"{len(evicted)}; deleted-keys {sum(t['pass'] for t in tests)}/{len(tests)} PASS"
          + (f"; short aborts {buf['n_short_abort']}, weight sum {buf['weight_sum']:.2f}"
             if policy is not None else "")
          + (f"; anchors {anchor['n_anchor']} x{anchor['scale'] or 1:.3f} -> frame share "
             f"{anchor.get('share_after', anchor['share_before'])}" if anchor is not None else "")
          + (f"; off track: new {buf['onpolicy_weighting']['new']['share_down']:.1%}, pool "
             f"{buf['onpolicy_weighting']['pool']['share_down']:.1%} of MPC frames down-weighted"
             if ow is not None else ""))
    return {"outputs": [str(ctx.it() / "buffer.json"), str(ctx.it() / "allowlist.json")]}


def onpolicy_rows(ctx: Ctx, rows: list[dict]) -> list[dict]:
    """Copies of iteration i's new MPC rows with ``frame_weights`` + ``off_track``: latents =
    the episode's ``sim_latent`` (the deploy encoder's, logged by the node), else deploy_i's
    encoding of ``pcd_belt_seg``; tube = iteration i's latent_waypoints.npz."""
    from task_common import onpolicy_weighting
    from task_common.latent_encoder import LatentEncoder

    rule = onpolicy_weighting.rule_of(ctx.cfg["onpolicy_weighting"])
    with np.load(ctx.it() / "latent_waypoints.npz", allow_pickle=True) as d:
        wp = {k: np.asarray(d[k]) for k in ("z_traj", "z_std", "frames", "tol", "dt")}
    enc, out = None, []
    for r in rows:
        with np.load(r["episode"], allow_pickle=True) as d:
            n = len(d["actions"])
            ut = np.asarray(d["utime"], np.float64)
            z = np.asarray(d["sim_latent"], np.float64) if "sim_latent" in d.files else None
            seg = (None if z is not None and z.shape == (n, wp["z_traj"].shape[1])
                   else list(d["pcd_belt_seg"]))
        src = "sim_latent"
        if seg is not None:
            enc = enc or LatentEncoder.load(jload(ctx.it() / "deploy.json")["deploy"])
            z = np.stack([enc.encode(np.asarray(s, np.float32)) for s in seg]).astype(np.float64)
            src = "encoded"
        dt = float(np.median(np.diff(ut))) * 1e-6 if n > 1 else float(wp["dt"])
        fw, info = onpolicy_weighting.episode_weights(z, wp, dt, bool(r["label_human"]), rule)
        out.append({**r, "frame_weights": fw, "off_track": {**info, "latents": src, "dt": dt}})
    return out


def onpolicy_stats(ctx: Ctx, new: list[dict], rows: list[dict]) -> dict:
    """Share of on-policy (MPC) frames past the off-track cut and the frame mass kept: this
    iteration's rollouts and every MPC row of the buffer that carries ``off_track``."""
    from task_common import onpolicy_weighting

    def part(rs):
        rs = [r for r in rs if "off_track" in r]
        n = sum(r["off_track"]["n_frames"] for r in rs)
        down = sum(r["off_track"]["n_down"] for r in rs)
        eff = sum(onpolicy_weighting.effective_frames(r, r["off_track"]["n_frames"]) for r in rs)
        return {"n_rows": len(rs), "n_off_track": sum(r["off_track"]["weighted_from"] is not None
                                                      for r in rs),
                "frames": n, "frames_down": down, "share_down": down / n if n else 0.0,
                "mass_kept": eff / n if n else None}

    return {"rule": onpolicy_weighting.rule_of(ctx.cfg["onpolicy_weighting"]),
            "waypoints": rel(ctx.it() / "latent_waypoints.npz"), "new": part(new),
            "pool": part([r for r in rows if r.get("kind") == "mpc"]),
            "rows": {r["id"]: {k: r["off_track"][k] for k in (
                "first_off_track", "weighted_from", "n_down", "n_frames", "ratio_max",
                "latents")} for r in new}}


def anchor_frac_at(ctx: Ctx, refit_iter: int) -> float:
    from task_common import buffer_policy

    an = {**ANCHOR_DEFAULTS, **ctx.cfg["anchor"]}
    return buffer_policy.anchor_frac(an["frac"], an["decay_to"], refit_iter, an["decay_by"])


def n_frames(ep) -> int:
    with np.load(ep, allow_pickle=True) as d:
        return len(d["actions"])


# ---------------------------------------------------------------- eval


def eval_due(ctx: Ctx) -> bool:
    every = ctx.cfg["eval_every"]  # 0 / null: never (V12 screening cells); 1: every iteration
    return (ctx.smoke or ctx.a.force == "eval" or bool(every) and ctx.i % every == 1 % every
            or keep_best(ctx.cfg) and ctx.i == 0)  # keep_best: theta_0's eval is the baseline


def demo_final_belt(ctx: Ctx) -> np.ndarray:
    with np.load(REPO / ctx.cfg["goal_episode"], allow_pickle=True) as d:
        return np.asarray(d["pcd_belt"])[-1].astype(np.float64)


def belt_rmse_mm(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(index-wise, best cyclic shift) RMSE in mm of two ordered belt polylines (sim GT)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    direct = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))) * 1e3
    best = min(float(np.sqrt(np.mean(np.sum((np.roll(a, s, axis=0) - b) ** 2, axis=1))))
               for s in range(len(a))) * 1e3
    return direct, best


def eval_candidate(ctx: Ctx, dry: bool):
    """keep_best `eval` (before `collect`): theta_i's export under iter<i>/cand/ (exported here
    when missing) -> cand/eval.json, copied to iter<i>/eval.json."""
    c = CandCtx(ctx)
    if dry:
        print(f"  keep_best: eval of the candidate theta_{ctx.i} ({rel(c.it())})")
        return st_eval(c, True)
    if not (c.it() / "deploy.json").is_file():
        st_export(prepare_cand(ctx), False)
    st_eval(c, False)
    jdump(ctx.it() / "eval.json", {**jload(c.it() / "eval.json"), "candidate": True})
    write_budget(ctx)
    return {"outputs": [str(ctx.it() / "eval.json"), str(c.it() / "eval.json")]}


def st_eval(ctx: Ctx, dry: bool):
    if not eval_due(ctx):
        print(f"  eval skipped (iteration {ctx.i}: every {ctx.cfg['eval_every']}, odd i)")
        return {"status": "skipped"}
    if keep_best(ctx.cfg) and not isinstance(ctx, CandCtx):
        return eval_candidate(ctx, dry)
    dj = ctx.it() / "deploy.json"
    deploy = jload(dj)["deploy"] if dj.is_file() else "<deploy_i>"
    base = ctx.it() / "eval"
    runs, c3s = [], {}
    for er in eval_runs_of(ctx.cfg):  # full task, no prefix, all stages (V12 too)
        cfg = mppi_yaml(ctx, base / f"mppi_eval_{er['run']}.yaml", 0.0, ctx.cfg["seed"],
                        er.get("floors"), dry=dry)
        if c3_on(ctx.cfg):
            cfg = c3s[er["run"]] = c3plus_params(ctx, f"eval_{er['run']}", er.get("floors"),
                                                 dry=dry)
        rs = er.get("reset_seed")  # hard suite: grasp starts reset by script after episode 1
        cmd = harness_cmd(ctx, ctx.url_eval, deploy, cfg, er["start_states"], er["variants"],
                          er["repeats"], base / er["run"], f"V11_iter{ctx.i}_eval_{er['run']}",
                          rs is not None, rs or 0)
        runs.append((er, cmd))
    if dry:
        for _, c in runs:
            show(c)
        return {}
    goal_belt = demo_final_belt(ctx)
    res, rows, n_tot = [], [], 0
    for er, cmd in runs:
        n_want = er["repeats"] * (len(er["variants"].split(",")) if er["variants"] else 1)
        r = harness(ctx, cmd, base / er["run"], n_want, ctx.logs() / f"eval_{er['run']}.log",
                    "eval")
        res.append({"run": er["run"], **r})
        if r["rc"] != 0:
            raise SystemExit(f"eval harness {er['run']} exit {r['rc']}")
        idx = jload(base / er["run"] / "index.json")
        for e in idx["episodes"]:
            x = episode_row(n_tot, base / er["run"], e)
            belt = x.pop("final_belt_gt", None)
            if belt is not None:
                x["belt_rmse_mm"], x["belt_rmse_shift_mm"] = belt_rmse_mm(belt, goal_belt)
            x["run"], x["cell"] = er["run"], er["cells"][min(len(er["cells"]) - 1,
                                                             len([y for y in rows
                                                                  if y["run"] == er["run"]]))]
            rows.append(x)
            n_tot += 1
    eps = [x["episode"] for x in rows if x["episode"]]
    lab_p = base / "labels_gt.json"
    if lab_p.exists():
        move_aside([lab_p])
    if run([sys.executable, "scripts/lcs/human_label.py", "--episodes", *eps, "--out",
            rel(lab_p)], ctx.logs() / "eval_label.log"):
        raise SystemExit("human_label.py (eval, report only) failed")
    lab = {e["episode"]: e["labels"] for e in jload(lab_p)["episodes"]}
    for x in rows:
        lb = lab.get(x["episode"])
        x["engaged_gt"] = bool(lb and lb["engaged"]["success"])
        x["strict_gt"] = bool(lb and lb["strict"]["success"])
    ms = [v for er, _ in runs for e in jload(base / er["run"] / "index.json")["episodes"]
          for v in solves_ms(base / er["run"], e)]
    summ = {"n": len(rows), "strict": sum(x["strict_gt"] for x in rows),
            "engaged": sum(x["engaged_gt"] for x in rows),
            "cloud_success": sum(bool(x.get("cloud_success")) for x in rows),
            "rmse_mean_mm": float(np.mean([x["belt_rmse_shift_mm"] for x in rows
                                           if "belt_rmse_shift_mm" in x] or [np.nan])),
            "n_unsafe": sum(x["unsafe"] for x in rows),
            "aborts": sum(x["status"] == "aborted" for x in rows),
            "solve_p95_ms": float(np.percentile(ms, 95)) if ms else None,
            "max_rot_deg": [max((x.get("max_rot_deg") or [0, 0])[j] for x in rows)
                            for j in (0, 1)]}
    model = model_of(ctx, ctx.i)
    ev = {"iter": ctx.i, "model": model["id"], "box_id": f"ub-tr{ctx.i}", "deploy": deploy,
          "url": ctx.url_eval, "dir": rel(base), "runs": res, "rows": rows, "summary": summ,
          "label": f"{model['id']} · {ctx.cfg['registry']['suite']} · Qw · ub-tr{ctx.i}",
          "cloud": ctx.cfg["noise_preset"], "stages_s": ctx.cfg["stage_max_durations_s"],
          "clearance": bool(ctx.cfg["mppi_clearance"]),
          "rmse_ref": f"final pcd_belt of {ctx.cfg['goal_episode']} (sim GT, best cyclic shift)"}
    if c3s:
        ev.update(controller="c3plus", c3plus_params=c3s, clearance=False)
    jdump(ctx.it() / "eval.json", ev)
    summ.update(stage0_reached=sum(x.get("stage_0_how") == "reached" for x in rows),
                board_aborts=sum(bool(x.get("board_abort")) for x in rows))
    (base / "table.md").write_text(eval_table_md(ev))
    write_budget(ctx)
    print(f"  eval {model['id']}: strict {summ['strict']}/{summ['n']}, engaged "
          f"{summ['engaged']}/{summ['n']}, unsafe {summ['n_unsafe']}")
    return {"outputs": [str(ctx.it() / "eval.json")]}


def eval_table_md(ev: dict) -> str:
    s = ev["summary"]
    st = ev.get("stages_s", [4.0, 30.0])
    md = [f"# {ev['label']}", "",
          (f"deploy `{ev['deploy']}`; MPPI two stages (pre_place_2 {st[0]:g} s, seated "
           f"{st[1]:g} s), no action noise, snapshot resets, --belt-seg --cloud-noise "
           f"{ev.get('cloud', 'real_like')}{', clearance on' if ev.get('clearance') else ''}; "
           "strict/engaged = emulated label rules on sim GT (report only); RMSE vs "
           f"{ev['rmse_ref']}."), "",
          ("| cell | start | outcome GT | engaged | strict | wrap / h GT | cloud success | "
           "RMSE mm | max rot F/UR deg | solve p95 ms | unsafe |"), "|---" * 11 + "|"]
    for x in ev["rows"]:
        sm = x.get("solve_ms") or {}
        md.append(f"| {x['cell']} | {x['start_id']} | {x['outcome_gt']} | {x['engaged_gt']} | "
                  f"{x['strict_gt']} | {x.get('final_wrap_deg_gt', 0):.0f} / "
                  f"{x.get('final_h_mm_gt', 0):+.2f} | {x.get('cloud_success')} | "
                  f"{x.get('belt_rmse_shift_mm', float('nan')):.1f} | "
                  f"{(x.get('max_rot_deg') or [0, 0])[0]:.0f} / "
                  f"{(x.get('max_rot_deg') or [0, 0])[1]:.0f} | {sm.get('p95') or 0:.1f} | "
                  f"{x['unsafe']} |")
    md += ["", (f"**{s['strict']}/{s['n']} strict, {s['engaged']}/{s['n']} engaged, cloud "
                f"{s['cloud_success']}, RMSE {s['rmse_mean_mm']:.1f} mm, unsafe "
                f"{s['n_unsafe']}, solve p95 {s['solve_p95_ms'] or 0:.1f} ms**")]
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- deploy policy (V12-57)


def keep_best_pick(inc: dict | None, cand: dict, min_gain: int) -> tuple[dict, str]:
    """``inc`` (the deployed model; None at iteration 0) / ``cand`` (theta_i): {id, iter, score
    (None: not evaluated), n} -> (the model to deploy, reason). Ties go to the candidate when
    min_gain is 0."""
    if inc is None:
        return cand, "initial (theta_0)"
    if cand["score"] is None:
        return inc, "keep (candidate not evaluated)"
    if cand["id"] == inc["id"]:
        return inc, "keep (same model)"
    if inc["score"] is None:
        return cand, "deploy (deployed model not evaluated)"
    gain = cand["score"] - inc["score"]
    if gain >= min_gain:
        return cand, f"deploy (gain {gain:+d} >= {min_gain})"
    return inc, f"keep (gain {gain:+d} < {min_gain})"


def policy_score(ev: dict, metric: str) -> int:
    return int(ev["summary"]["engaged" if metric == "eval_engaged" else "strict"])


def st_deploy(ctx: Ctx, dry: bool):
    """keep_best: deploy theta_i or keep the deployed model (keep_best_pick on the evals) ->
    iter<i>/deploy_policy.json; the deployed model is exported to iter<i>/ (box_i)."""
    if not keep_best(ctx.cfg):
        print("  deploy: deploy_policy keep_best only (else `export` deploys theta_i)")
        return {"status": "skipped"}
    rule = {**DEPLOY_POLICY_DEFAULTS, **ctx.cfg["deploy_policy"]}
    i, met = ctx.i, rule["metric"]
    pp = ctx.it(i - 1) / "deploy_policy.json" if i > 0 else None
    inc = jload(pp)["deployed"] if pp is not None and pp.is_file() else None
    mp, ep = ctx.it() / "model.json", ctx.it() / "eval.json"
    model = jload(mp) if mp.is_file() else None
    ev = jload(ep) if ep.is_file() else None
    cand = None if model is None else {"id": model["id"], "iter": i, "score": None, "n": None}
    if cand is not None and ev is not None and ev["model"] == model["id"]:
        cand.update(score=policy_score(ev, met), n=ev["summary"]["n"])
    if dry:
        print(f"  deploy_policy keep_best ({met}, min_gain {rule['min_gain']}): deployed before "
              + ("none" if i == 0 else f"{inc['id']} (iter {inc['iter']}, {inc['score']}/"
                 f"{inc['n']})" if inc else f"<{rel(pp)}>") + f"; theta_{i} "
              + (("<eval>" if ev is None else f"{cand['score']}/{cand['n']}")
                 if eval_due(ctx) else "not evaluated"))
        sm = ctx.cfg.get("seed_model")
        th = model or {"checkpoint": sm["checkpoint"] if i == 0 and sm else f"<theta_{i}>",
                       "id": f"theta_{i}", "letter": "?"}
        if cand is not None and (i == 0 or inc) and (cand["score"] is not None
                                                     or not eval_due(ctx)):
            pick, why = keep_best_pick(inc, cand, rule["min_gain"])
            dm = model if pick is cand else pick["model"]
        elif i == 0:
            why, dm = "initial (theta_0)", th
        elif not eval_due(ctx):
            why = "keep (candidate not evaluated)"
            dm = inc["model"] if inc else {"checkpoint": f"<deployed_{i - 1}>", "id": "?",
                                           "letter": "?"}
        else:
            why = (f"deploy theta_{i} iff its {met} >= deployed_{i - 1}'s + {rule['min_gain']}, "
                   f"else keep deployed_{i - 1}")
            dm = {"checkpoint": f"<theta_{i} | deployed_{i - 1}>", "id": "?", "letter": "?"}
        print(f"  -> deploy {dm['id']}: {why}")
        st_export(ctx, True, dm)
        return {}
    if i > 0 and inc is None:
        raise SystemExit(f"{rel(pp)} missing: run --iter {i - 1} --step deploy")
    model = model_of(ctx, i)
    pick, why = keep_best_pick(inc, cand, rule["min_gain"])
    dm = model if pick is cand else pick["model"]
    rec = {"iter": i, "rule": rule, "candidate": cand,
           "incumbent": None if inc is None else {k: inc[k] for k in ("id", "iter", "score", "n")},
           "deployed": {**{k: pick[k] for k in ("id", "iter", "score", "n")}, "model": dm},
           "decision": why, "eval": None if ev is None else rel(ep), "decided": now()}
    jdump(ctx.it() / "deploy_policy.json", rec)
    print(f"  deploy_policy: theta_{i} {model['id']} {met} {cand['score']}/{cand['n']}; deployed_"
          f"{i - 1} " + ("-" if inc is None else f"{inc['id']} {inc['score']}/{inc['n']}")
          + f" -> deploy {pick['id']}: {why}")
    r = st_export(ctx, False, dm)
    return {"outputs": [str(ctx.it() / "deploy_policy.json"), *r["outputs"],
                        str(ctx.lout() / "cache_buffer.npz")]}


# ---------------------------------------------------------------- budget


def iters_present(ctx: Ctx) -> list[int]:
    out = []
    for p in sorted(ctx.root.glob("iter*")):
        m = re.fullmatch(r"iter(\d+)", p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def write_budget(ctx: Ctx) -> dict:
    """Recomputed from scratch: the seed budget + every iteration's index.json files."""
    b = {}
    for sb in ctx.seed_budgets:  # v1.1: seed + seed_ext summed
        for k, v in jload(sb).items():
            if not isinstance(v, (dict, str)) and k != "wall_s":
                b[k] = b.get(k, 0) + v
    b.update(robot_s_detail={"seed": float(b["robot_s"]), "mpc_episodes": 0.0,
                             "mpc_resets": 0.0, "eval": 0.0},
             seed_budgets=[rel(x) for x in ctx.seed_budgets],
             resets_human_eval=0, rollouts_mpc_skipped=0, per_iter=[], consistency=[])
    b["robot_s"] = float(b["robot_s"])
    for i in iters_present(ctx):
        it = ctx.it(i)
        row = {"iter": i}
        idx_p = it / "collect" / "index.json"
        if idx_p.is_file() and (it / "attempts.json").is_file():
            idx = jload(idx_p)
            eps = idx["episodes"]
            bud = idx.get("budget") or {}
            n_ok = sum(e.get("file") is not None for e in eps)
            ep_s = sum(e.get("duration_s", 0.0) + START_SETTLE_S for e in eps)
            rs = float(bud.get("reset_s", 0.0))
            ps = sum((e.get("prefix") or {}).get("duration_s", 0.0) for e in eps)
            if ps:  # V12 prefix replays: robot time, not MPC
                b["robot_s_detail"]["mpc_prefix"] = b["robot_s_detail"].get("mpc_prefix",
                                                                            0.0) + ps
                b["robot_s"] += ps
            b["rollouts_mpc"] += n_ok
            b["rollouts_mpc_skipped"] += len(eps) - n_ok
            b["resets_scripted"] += int(bud.get("resets_scripted", 0))
            b["resets_human"] += int(bud.get("resets_human", 0))
            b["robot_s_detail"]["mpc_episodes"] += ep_s
            b["robot_s_detail"]["mpc_resets"] += rs
            b["robot_s"] += ep_s + rs
            att = jload(it / "attempts.json")["attempts"]
            rb = [e.get("reset_before", {}).get("kind") for e in eps]
            cons = {"iter": i, "index_episodes": len(eps), "attempts": len(att),
                    "index_resets_scripted": bud.get("resets_scripted"),
                    "rows_scripted": rb.count("scripted"),
                    "index_resets_human": bud.get("resets_human"), "rows_human": rb.count("human")}
            row.update(mpc=n_ok, resets_scripted=rb.count("scripted"),
                       resets_human=rb.count("human"), robot_s=ep_s + rs)
            if ps:
                row.update(robot_s=ep_s + rs + ps, prefix_s=ps,
                           prefix_failed=sum(not (e.get("prefix") or {}).get("ok", True)
                                             for e in eps))
            if (it / "labels.json").is_file():
                nl = len(jload(it / "labels.json")["episodes"])
                b["labels_human"] += nl
                cons.update(labels=nl, labelled_episodes=n_ok)
            cons["ok"] = (cons["index_episodes"] == cons["attempts"]
                          and cons["index_resets_scripted"] == cons["rows_scripted"]
                          and cons["index_resets_human"] == cons["rows_human"]
                          and cons.get("labels", n_ok) == n_ok)
            b["consistency"].append(cons)
        if (it / "eval.json").is_file():
            ev = jload(it / "eval.json")
            n = len(ev["rows"])
            es = sum(x["duration_s"] + START_SETTLE_S for x in ev["rows"])
            b["rollouts_eval"] += n
            b["resets_human"] += n
            b["resets_human_eval"] += n
            b["robot_s_detail"]["eval"] += es
            b["robot_s"] += es
            row.update(eval=n, eval_robot_s=es)
        for f, key in (("refit.json", "gpu_s"), ("deploy.json", "gpu_s")):
            if (it / f).is_file():
                b["gpu_s"] += float(jload(it / f).get(key) or 0.0)
        row["cum"] = {k: b[k] for k in ("rollouts_mpc", "rollouts_eval", "resets_scripted",
                                        "resets_human", "labels_human", "robot_s", "gpu_s")}
        b["per_iter"].append(row)
    b["consistent"] = all(c["ok"] for c in b["consistency"])
    b["updated"] = now()
    b["notes"] = ("seed counts from seed/budget.json; resets_human includes every snapshot "
                  "restore (the first episode of each collection run, every eval episode); "
                  "robot_s = episodes (duration + 0.5 s settle) + scripted resets + eval")
    jdump(ctx.root / "budget.json", b)
    return b


# ---------------------------------------------------------------- report


def policy_screen(ctx: Ctx, deploy: Path, buf: dict, cfg: Path, clear_npz=None) -> dict:
    """MPPI u_0 Franka-z / UR-z signs on the buffer's riding-over states (seated stage, GT h
    3..35 mm: report only); z from the numpy encoder on pcd_belt_seg."""
    import dataclasses

    import torch

    from task_common.latent_encoder import LatentEncoder
    from task_common.mppi_lcs import MppiParams, TorchLcs, mppi_plan

    late = {"move:place_3", "hold:place_3", "done"}
    cand = []
    for r in buf["rollouts"]:
        with np.load(r["episode"], allow_pickle=True) as d:
            meta = json.loads(str(d["sim_meta"]))
            labels = meta.get("phase_labels", [])
            mpc = [lab for lab in labels if lab.startswith("mpc:")]
            ok = late | ({mpc[-1]} if mpc else set())
            ph = np.asarray(d["sim_phase"], dtype=np.int64)
            h = np.asarray(d["sim_h_median_mm"], dtype=np.float64)
            for t in np.flatnonzero((h >= 3) & (h <= 35)):
                if labels and labels[ph[t]] in ok:
                    cand.append((r["episode"], int(t), r["kind"]))
    out = {"rule": "seated stage (collector move/hold:place_3, done; harness last mpc stage, "
                   "done) and GT h 3..35 mm", "n_candidates": len(cand)}
    if not cand:
        return {**out, "n": 0}
    rng = np.random.default_rng(ctx.cfg["seed"])
    pick = [cand[i] for i in sorted(rng.choice(len(cand), min(len(cand),
                                                               ctx.cfg["policy_screen_n"]),
                                               replace=False))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc, lcs = LatentEncoder.load(deploy), TorchLcs.from_deploy(deploy, dev)
    params = dataclasses.replace(MppiParams.from_yaml(cfg), action_noise_frac=0.0)
    if params.clearance is not None and clear_npz is not None:  # the run's exported model
        params = dataclasses.replace(params, clearance={**params.clearance,
                                                        "model": str(clear_npz)})
    gen = torch.Generator(device=lcs.device).manual_seed(ctx.cfg["seed"])
    goal = lcs.stage_goals[-1]
    us, kinds = [], []
    for ep, t, kind in pick:
        with np.load(ep, allow_pickle=True) as d:
            z = enc.encode(d["pcd_belt_seg"][t])
            p0 = np.r_[d["sim_ee_franka"][t, :3], d["sim_ee_ur"][t, :3]]
            q = np.stack([d["sim_ee_franka"][t, 3:7], d["sim_ee_ur"][t, 3:7]])
        c = params.z_ceiling_above_start_m
        prm = dataclasses.replace(params, z_max_franka=min(params.z_max_franka, p0[2] + c),
                                  z_max_ur=min(params.z_max_ur, p0[5] + c))
        kw = {"q_now": q} if prm.clearance is not None else {}
        U, _ = mppi_plan(lcs, prm, z, p0, np.zeros((2, 3)), goal, None, gen, **kw)
        us.append(U[0])
        kinds.append(kind)
    U = np.asarray(us)
    return {**out, "n": len(U), "kinds": {k: kinds.count(k) for k in set(kinds)},
            "both_down": float(np.mean((U[:, 2] < 0) & (U[:, 5] < 0))),
            "both_up": float(np.mean((U[:, 2] > 0) & (U[:, 5] > 0))),
            "fz_mean_mm": float(U[:, 2].mean() * 1e3), "uz_mean_mm": float(U[:, 5].mean() * 1e3)}


def auc(pos, neg) -> float | None:
    """P(d_success < d_failure), ties half."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p < n) + 0.5 * (p == n)).mean())


def separation(ctx: Ctx, i: int) -> dict | None:
    """GT-free: whitened distance of each B_i episode's last frame to deploy_i's seated goal,
    human-label successes vs failures (all buffer episodes but the goal episode; optimistic,
    most are training episodes). Cached in iter<i>/separation.json."""
    p = ctx.it(i) / "separation.json"
    if p.is_file():
        return jload(p)
    dj = ctx.it(i) / "deploy.json"
    if not dj.is_file() or not Path(jload(dj)["cache"]).is_file():
        return None
    d = jload(dj)
    with np.load(d["cache"]) as c:
        zn, fi, files = c["z_next"], c["file_idx"], [str(f) for f in c["files"]]
    with np.load(d["deploy"], allow_pickle=True) as dd:
        z_std, g2 = np.asarray(dd["z_std"]), np.asarray(dd["z_goal_stage2"])
    lab = {r["episode"]: r["label_human"] for r in jload(buffer_in(ctx, i))["rollouts"]
           if r.get("kind") not in NON_TASK_KINDS}
    goal_ep = str((REPO / ctx.cfg["goal_episode"]).resolve())
    dist = {}
    for k, f in enumerate(files):
        m = np.flatnonzero(fi == k)
        if len(m) and str(Path(f).resolve()) != goal_ep:
            dist[f] = float(np.linalg.norm((zn[m[-1]] - g2) / z_std))
    pos = [v for f, v in dist.items() if lab.get(f) is True]
    neg = [v for f, v in dist.items() if lab.get(f) is False]
    out = {"auc_final_to_goal": auc(pos, neg), "n_success": len(pos), "n_failure": len(neg),
           "median_success": float(np.median(pos)) if pos else None,
           "median_failure": float(np.median(neg)) if neg else None,
           "goal_tol_whitened": jload(ctx.it(i) / "goal.json").get("goal_tol_whitened")
           if (ctx.it(i) / "goal.json").is_file() else None}
    jdump(p, out)
    return out


def lc_row(ctx: Ctx, i: int, budget: dict) -> dict:
    it = ctx.it(i)
    m = jload(it / "model.json")
    row = {"iter": i, "model": m["id"], "kept": bool(m.get("kept"))}
    if (it / "refit.json").is_file():
        rf = jload(it / "refit.json")
        rep = jload(rf["report"])
        mt = rep["metrics"]
        row["refit"] = {"decision": rf.get("decision"), "report_decision": rf["report_decision"],
                        "recon_ratio": rf["recon_gate"]["ratio"],
                        "recon_ok": rf["recon_gate"]["ok"], "gates": rep.get("gates"),
                        **{k: mt[k] for k in ("one_step_chamfer_mm", "open_loop7_chamfer_mm",
                                              "recon_chamfer_mm", "rho_A", "B_fro_w",
                                              "drift_u0_w", "separation_auc", "aux_mae_h_mm")},
                        "ref_one_step": ((rep.get("ref") or {}).get("metrics_same_split") or {})
                        .get("one_step_chamfer_mm"), "train_s": rep["timings_s"]["train"]}
        if "behaviour_gate" in rf:
            row["refit"]["behaviour"] = rf["behaviour_gate"]
    if (it / "trust_region.json").is_file():
        tr = jload(it / "trust_region.json")
        row["box"] = {**tr["ratios"], "k": tr["k"], "n_floor_hits": tr["n_floor_hits"],
                      "n_clip_hits": tr["n_clip_hits"], "all_at_floor": tr["all_at_floor"]}
    if (it / "attempts.json").is_file():
        att = jload(it / "attempts.json")
        a = att["attempts"]
        ms = [v for e in jload(it / "collect/index.json")["episodes"]
              for v in solves_ms(it / "collect", e)]
        row["collect"] = {
            "start": att["start"]["key"], "n": len(a),
            "strict_gt": sum(bool(x.get("strict_gt_final")) for x in a),
            "outcomes_gt": [x["outcome_gt"] for x in a],
            "cloud_success": sum(bool(x.get("cloud_success")) for x in a),
            "aborts": sum(x["status"] == "aborted" for x in a),
            "unsafe": sum(x["unsafe"] for x in a),
            "resets": [x["reset_before"] for x in a],
            "solve_p95_ms": float(np.percentile(ms, 95)) if ms else None,
            "solve_p95_worst_ep_ms": max(((x.get("solve_ms") or {}).get("p95") or 0)
                                         for x in a),
            "fallbacks": sum(x.get("fallbacks") or 0 for x in a),
            "plans_violating": sum(x.get("plans_violating") or 0 for x in a),
            "max_rot_deg": [max((x.get("max_rot_deg") or [0, 0])[j] for x in a) for j in (0, 1)],
            "min_tip_clearance_mm": min((x.get("min_franka_tip_clearance_mm") or 1e9)
                                        for x in a),
            "stage0_reached": sum(x.get("stage_0_how") == "reached" for x in a),
            "stage0_timeout": sum(x.get("stage_0_how") == "timeout" for x in a),
            "board_aborts": sum(bool(x.get("board_abort")) for x in a),
            "clear_min_ur_mm": min([x["clear_min_mm"]["ur"] for x in a
                                    if (x.get("clear_min_mm") or {}).get("ur") is not None]
                                   or [None]),
            "clear_violating_plans": sum(x.get("clear_violating_plans") or 0 for x in a)}
    if (it / "curriculum.json").is_file():
        cu = jload(it / "curriculum.json")
        row["prefix_frac"] = cu["frac"]
        row["curriculum"] = {k: cu.get(k) for k in ("decision", "zero_streak", "prev")}
        if (it / "attempts.json").is_file():
            att = jload(it / "attempts.json")
            row["prefix_ok"] = att.get("prefix_ok")
            row["mppi_stages"] = att.get("mppi_stages")
            row["pool"] = jload(it / "collect_plan.json").get("pool") \
                if (it / "collect_plan.json").is_file() else None
    if (it / "labels.json").is_file():
        lab = jload(it / "labels.json")["episodes"]
        row["successes"] = sum(e["success"] for e in lab)
        row["strict_label"] = sum(e["labels"]["strict"]["success"] for e in lab)
        row["n_labelled"] = len(lab)
    if (it / "buffer.json").is_file():
        bf = jload(it / "buffer.json")
        row["buffer"] = {"n": bf["n"], "n_success": bf["n_success"]}
        if "n_short_abort" in bf:
            row["buffer"].update(n_short_abort=bf["n_short_abort"], weight_sum=bf["weight_sum"],
                                 n_evicted=len(bf["evicted"]),
                                 by_kind=bf["policy"]["kept"]["by_kind"])
        if "onpolicy_weighting" in bf:
            row["onpolicy"] = {k: bf["onpolicy_weighting"][k] for k in ("new", "pool")}
            if "anchor" in bf:
                row["onpolicy"]["anchor_share"] = bf["anchor"].get(
                    "share_after", bf["anchor"]["share_before"])
    if (it / "goal.json").is_file() and "tol_1" in jload(it / "goal.json"):
        row["tol_1"] = jload(it / "goal.json")["tol_1"]
    bxp = it / f"box_{i}.json"
    if bxp.is_file() and "reversible" in jload(bxp):
        bx = jload(bxp)
        row["planner_box"] = {"reversible": sum(bx["reversible"]),
                              "widened": sum(bx["widened"]),
                              "mm_deg": [[round(a * sc, 3), round(b * sc, 3)] for a, b, sc in zip(
                                  bx["u_lb"], bx["u_ub"], [1e3] * 6 + [57.29577951308232] * 6)]}
        if "demo_union" in bx:
            row["planner_box"]["widened_by_demo"] = sum(bx["demo_union"]["widened_by_demo"])
    if (it / "eval.json").is_file():
        row["eval"] = jload(it / "eval.json")["summary"]
    if (it / "policy_screen.json").is_file():
        ps = jload(it / "policy_screen.json")
        row["policy_screen"] = {k: ps.get(k) for k in ("n", "both_down", "both_up",
                                                        "fz_mean_mm", "uz_mean_mm")}
    if (it / "response_gate.json").is_file():
        rq = jload(it / "response_gate.json")
        row["response_gate"] = {"pass": rq["pass"], "n_states": rq["n_states"],
                                **{k: v["cos_median"] for k, v in rq["channels"].items()}}
    if (it / "refit.json").is_file():
        row["recon_clean_mm"] = jload(jload(it / "refit.json")["report"])["metrics"][
            "recon_chamfer_mm"]
        rf = jload(it / "refit.json")
        if "train_objective" in rf:
            row["train_objective"] = rf["train_objective"]
        if "graph" in rf:
            row["graph"] = {k: rf["graph"][k] for k in (
                "n_nodes", "n_edges", "n_episodes", "rejected_frac", "n_components",
                "largest_component_frac", "unreachable_frac", "build_s", "refit_metrics")}
    if (it / "deploy_policy.json").is_file():
        dp = jload(it / "deploy_policy.json")
        row["deploy_policy"] = {"trained": model_of(ctx, i)["id"], "candidate": dp["candidate"],
                                "incumbent": dp["incumbent"],
                                "deployed": {k: dp["deployed"][k] for k in ("id", "iter",
                                                                            "score", "n")},
                                "decision": dp["decision"]}
    row["separation"] = separation(ctx, i)
    cum = next((p["cum"] for p in budget["per_iter"] if p["iter"] == i), None)
    row["budget"] = cum
    st = get_state(ctx, i)
    row["wall_s"] = {k: round(v["wall_s"], 1) for k, v in st["steps"].items()
                     if v.get("wall_s") is not None}
    row["failures"] = len(st.get("failures", []))
    return row


def verdict(ctx: Ctx, rows: list[dict]) -> str:
    """Stop rules (plan §4 + package): logged, acted on only by refusing the next `all`."""
    s = ctx.cfg["stop"]
    att = [ok for r in rows for ok in ([True] * r.get("successes", 0)
                                       + [False] * (r.get("n_labelled", 0)
                                                    - r.get("successes", 0)))]
    nosucc = None
    if len(att) >= s["zero_success_attempts"] and not any(att[-s["zero_success_attempts"]:]):
        nosucc = (f"STOP (no success): 0 successes in the last {s['zero_success_attempts']} MPC "
                  "attempts")
        if not s.get("zero_success_report_only"):
            return nosucc
    k = s.get("safety_consecutive")
    if k:
        bad = [bool((r.get("collect") or {}).get("board_aborts")
                    or (r.get("eval") or {}).get("n_unsafe")) for r in rows]
        if len(bad) >= k and all(bad[-k:]):
            return (f"STOP (safety): board-contact aborts / unsafe evals in {k} consecutive "
                    "iterations")
    rej = 0
    for r in rows:
        rej = rej + 1 if (r.get("refit") or {}).get("decision") == "rejected" else 0
    if rej >= s["max_consecutive_rejections"]:
        return f"STOP (model): {rej} consecutive refit rejections"
    tail = rows[-s["done_consecutive"]:]
    if len(tail) == s["done_consecutive"] and all(
            r.get("successes", 0) >= s["done_successes"] for r in tail):
        return f"DONE: >= {s['done_successes']}/{ctx.cfg['new_per_iter']} in " \
               f"{s['done_consecutive']} consecutive iterations"
    return "continue" + (f" (report-only: {nosucc})" if nosucc else "")


def g4_line(rows: list[dict], budget: dict, v: str) -> str | None:
    if len(rows) < 3 or [r["iter"] for r in rows[:3]] != [0, 1, 2]:
        return None
    r3 = rows[:3]
    crashes = sum(r["failures"] for r in r3)
    floor_all = [r["iter"] for r in r3 if (r.get("box") or {}).get("all_at_floor")]
    succ = sum(r.get("successes", 0) for r in r3)
    stop_clean = v.startswith("STOP")
    ok = (crashes == 0 and budget["consistent"] and not floor_all
          and (succ >= 1 or stop_clean))
    return (f"G4 {'PASS' if ok else 'FAIL'}: crashes {crashes}, budget consistent "
            f"{budget['consistent']}, box at the floor on all channels in iterations "
            f"{floor_all or 'none'}, successes in iterations 0-2: {succ}/15"
            + ("" if succ else f"; stop rule: {v}"))


def g4p_line(rows: list[dict], budget: dict, v: str) -> str | None:
    """G4' (plan v1.1 §3) over iterations 0-2."""
    if len(rows) < 3 or [r["iter"] for r in rows[:3]] != [0, 1, 2]:
        return None
    r3 = rows[:3]
    crashes = sum(r["failures"] for r in r3)
    c = [r.get("collect") or {} for r in r3]
    n = sum(x.get("n", 0) for x in c)
    board = sum(x.get("board_aborts", 0) for x in c)
    reached = sum(x.get("stage0_reached", 0) for x in c)
    succ = sum(r.get("successes", 0) for r in r3)
    ok = (crashes == 0 and budget["consistent"] and board == 0 and n and reached >= 0.6 * n
          and (succ >= 1 or v.startswith("STOP")))
    return (f"G4' {'PASS' if ok else 'FAIL'}: crashes {crashes}, budget consistent "
            f"{budget['consistent']}, board-contact aborts {board}/{n}, stage 1 reached "
            f"{reached}/{n} ({reached / max(n, 1):.0%}, gate >= 60 %), successes in iterations "
            f"0-2: {succ}/{n}" + ("" if succ else f"; stop rule: {v}"))


def fmt(x, spec=".2f", none="-"):
    return none if x is None else format(x, spec)


def st_report(ctx: Ctx, dry: bool):
    if dry:
        print(f"  would write {rel(ctx.root)}/budget.json, {rel(ctx.it())}/policy_screen.json, "
              f"{rel(ctx.root)}/learning_curve.{{json,md,png}} row {ctx.i}")
        return {}
    budget = write_budget(ctx)
    dj, bj = ctx.it() / "deploy.json", ctx.it() / "buffer.json"
    if dj.is_file() and bj.is_file():
        cn = ctx.it() / "collect/clearance_model.npz"
        ps = policy_screen(ctx, Path(jload(dj)["deploy"]), jload(bj),
                           ctx.it() / "mppi_collect.yaml", cn if cn.is_file() else None)
        jdump(ctx.it() / "policy_screen.json", {**ps, "deploy": jload(dj)["deploy"],
                                                "buffer": str(bj)})
    if ctx.cfg.get("response_gate") is not None and dj.is_file() \
            and (ctx.it() / "attempts.json").is_file():
        response_gate_of(ctx)
    rows = [lc_row(ctx, j, budget) for j in iters_present(ctx)
            if (ctx.it(j) / "model.json").is_file() and j <= ctx.i]
    v = verdict(ctx, rows)
    g4 = (g4p_line if ctx.cfg.get("gate4") == "G4'" else g4_line)(rows, budget, v)
    lc = {"rows": rows, "verdict": v, "g4": g4, "budget": {k: budget[k] for k in (
        "rollouts_demo", "rollouts_excitation", "rollouts_mpc", "rollouts_eval",
        "resets_scripted", "resets_human", "labels_human", "robot_s", "gpu_s", "consistent")},
        "updated": now()}
    jdump(ctx.root / "learning_curve.json", lc)
    md = learning_curve_md(ctx, rows, budget, v, g4)
    (ctx.root / "learning_curve.md").write_text(md)
    plot(ctx.root / "learning_curve.png", rows)
    print(md)
    return {"outputs": [str(ctx.root / "learning_curve.md")]}


def response_gate_of(ctx: Ctx) -> dict:
    """``response_gate``: deploy_i on iteration i's collect episodes (response_gate.py) ->
    iter<i>/response_gate.json (report-only)."""
    import response_gate as rgm

    c = {**RGATE_DEFAULTS, **ctx.cfg["response_gate"]}
    dep = jload(ctx.it() / "deploy.json")
    eps = [a["episode"] for a in jload(ctx.it() / "attempts.json")["attempts"] if a["episode"]]
    res = rgm.compute(dep["deploy"], eps, int(c["stride"]), float(c["max_cos"]))
    res.update(model=dep["model"], deploy_sha256=dep["deploy_sha256"], line=rgm.line(res))
    jdump(ctx.it() / "response_gate.json", res)
    print(f"  response gate {dep['model']}: {res['line']}")
    return res


def learning_curve_md(ctx: Ctx, rows, budget, v, g4) -> str:
    md = ["# V11 real-first learning curve", "",
          (f"Success = the emulated human label ({ctx.cfg['label_rule']}, one bit per episode); "
           "strict and GT outcomes are report-only; cloud success is monitor-only. Budget "
           f"columns are cumulative (seed included: {budget['rollouts_demo']} demos + "
           f"{budget['rollouts_excitation']} excitation, "
           f"{budget['robot_s_detail']['seed']:.0f} s)."), "",
          ("| iter | model | refit | successes / n | strict | cloud | GT outcomes | rollouts_mpc "
           "| robot_s | resets scr / hum | box k, trans / rot gm (floor hits) | policy both-down "
           "| solve p95 ms (worst ep) | unsafe / aborts | eval strict / engaged (n) | wall s |"),
          "|---" * 16 + "|"]
    for r in rows:
        c, b, rf = r.get("collect") or {}, r.get("box") or {}, r.get("refit") or {}
        bu, ev, ps = r.get("budget") or {}, r.get("eval"), r.get("policy_screen") or {}
        dec = rf.get("decision") or ("seed" if r["iter"] == 0 else "-")
        md.append(
            f"| {r['iter']} | {r['model']}{' (kept)' if r['kept'] else ''} | {dec} | "
            f"{r.get('successes', '-')}/{r.get('n_labelled', '-')} | {r.get('strict_label', '-')}"
            f" | {c.get('cloud_success', '-')} | "
            f"{', '.join(map(str, c.get('outcomes_gt', []))) or '-'} | "
            f"{bu.get('rollouts_mpc', '-')} | {fmt(bu.get('robot_s'), '.0f')} | "
            f"{bu.get('resets_scripted', '-')} / {bu.get('resets_human', '-')} | "
            f"{b.get('k', '-')}, {fmt(b.get('box_trans_gm'))} / {fmt(b.get('box_rot_gm'))} "
            f"({b.get('n_floor_hits', '-')}) | {fmt(ps.get('both_down'), '.0%')} | "
            f"{fmt(c.get('solve_p95_ms'), '.1f')} ({fmt(c.get('solve_p95_worst_ep_ms'), '.1f')})"
            f" | {c.get('unsafe', '-')} / {c.get('aborts', '-')} | "
            + (f"{ev['strict']} / {ev['engaged']} ({ev['n']})" if ev else "-")
            + f" | {sum(r['wall_s'].values()):.0f} |")
    rg = (f"(vs the real_like floor {ctx.cfg['recon_floor_mm']} mm, gate <= "
          f"{ctx.cfg['recon_floor_factor']}x)" if "recon_floor_mm" in ctx.cfg else
          f"(clean clouds, gate <= {ctx.cfg['refit'].get('recon_max_mm')} mm)")
    md += ["", ("Refit of iteration i (theta_i on B_i; i = 0 fresh): held-out one-step / 7-step "
                f"Chamfer, recon {rg}, "
                "rho(A), ||B||_F whitened, u = 0 drift, separation AUC; one-step of theta_{i-1} "
                "on the same split."), "",
           ("| iter | decision (report) | one-step mm (prev) | h7 mm | recon mm (x floor) | "
            "rho(A) | B_fro_w | drift | AUC held-out | AUC final-to-goal, all B_i (median "
            "d succ / fail) | aux h mm | train s |"), "|---" * 12 + "|"]
    for r in rows:
        rf, sp = r.get("refit"), r.get("separation") or {}
        if not rf:
            continue
        md.append(
            f"| {r['iter']} | {rf.get('decision') or 'seed'} ({rf['report_decision']}) | "
            f"{fmt(rf['one_step_chamfer_mm'])} ({fmt(rf['ref_one_step'])}) | "
            f"{fmt(rf['open_loop7_chamfer_mm'])} | {fmt(rf['recon_chamfer_mm'])} "
            f"({fmt(rf['recon_ratio'], '.3f')}) | {fmt(rf['rho_A'], '.4f')} | "
            f"{fmt(rf['B_fro_w'], '.1f')} | {fmt(rf['drift_u0_w'], '.3f')} | "
            f"{fmt(rf['separation_auc'], '.3f')} | {fmt(sp.get('auc_final_to_goal'), '.3f')} "
            f"({fmt(sp.get('median_success'))} / {fmt(sp.get('median_failure'))}) | "
            f"{fmt(rf['aux_mae_h_mm'])} | "
            f"{fmt(rf['train_s'], '.0f')} |")
    if any("behaviour" in (r.get("refit") or {}) for r in rows):
        md += ["", ("Behaviour refit gate (behaviour_gate.py, candidate vs the previous accepted "
                    "model on the same frames): MPPI u_0 cosine to the next recorded actions, "
                    "cos_median cand / ref per phase, Franka u_0 y on demo hold frames "
                    "(mm/step), decision."), "",
               "| iter | approach | descent | success | hold u0 y cand / ref | decision |",
               "|---" * 6 + "|"]
        for r in rows:
            b = (r.get("refit") or {}).get("behaviour")
            if not b:
                continue
            cm = b.get("cos_median") or {}
            hp = b.get("hold_push_mm") or {}
            md.append(f"| {r['iter']} | " + " | ".join(
                f"{fmt((cm.get(ph) or {}).get('cand'))} / {fmt((cm.get(ph) or {}).get('ref'))}"
                for ph in ("approach", "descent", "success"))
                + f" | {fmt(hp.get('cand'))} / {fmt(hp.get('ref'))} | {b['line']} |")
    if ctx.cfg.get("gate4") == "G4'":
        md += ["", ("v1.1 columns: stage 1 (pre_place_2) reached on the latent (streak) vs "
                    "timeout, board-contact aborts, buffer after the iteration (short aborts, "
                    "successes, weight sum), tol_1, held-out recon Chamfer RMS on clean clouds, "
                    "planner box channels able to reverse / widened by the minimum box."), "",
               ("| iter | stage0 reached / timeout (n) | board aborts / n | min UR clear mm "
                "(plans < min) | short aborts in B | n_success in B | B n (w sum) | tol_1 | "
                "recon_clean_mm | box reversible / widened |"), "|---" * 10 + "|"]
        for r in rows:
            c, bu, pb = r.get("collect") or {}, r.get("buffer") or {}, r.get("planner_box") or {}
            md.append(
                f"| {r['iter']} | {c.get('stage0_reached', '-')} / {c.get('stage0_timeout', '-')}"
                f" ({c.get('n', '-')}) | {c.get('board_aborts', '-')} / {c.get('n', '-')} | "
                f"{fmt(c.get('clear_min_ur_mm'), '.1f')} ({c.get('clear_violating_plans', '-')})"
                f" | {bu.get('n_short_abort', '-')} | {bu.get('n_success', '-')} | "
                f"{bu.get('n', '-')} ({fmt(bu.get('weight_sum'))}) | {fmt(r.get('tol_1'), '.3f')}"
                f" | {fmt(r.get('recon_clean_mm'))} | {pb.get('reversible', '-')} / "
                f"{pb.get('widened', '-')} |")
    if any("onpolicy" in r for r in rows):
        md += ["", ("On-policy refit weighting (onpolicy_weighting): MPC frames past the "
                    "off-track cut (latent outside tol_k x the waypoint tol of every goal-demo "
                    "frame, + after_s), this iteration's rollouts and the whole buffer, frame "
                    "mass kept, anchor frame share (frame-weighted) of the next refit."), "",
               ("| iter | new rows off track / n | new frames down | buffer MPC frames down | "
                "MPC mass kept | anchor share |"), "|---" * 6 + "|"]
        for r in rows:
            o = r.get("onpolicy")
            if not o:
                continue
            nw, pl = o["new"], o["pool"]
            md.append(f"| {r['iter']} | {nw['n_off_track']} / {nw['n_rows']} | "
                      f"{nw['frames_down']} / {nw['frames']} ({fmt(nw['share_down'], '.0%')}) | "
                      f"{pl['frames_down']} / {pl['frames']} ({fmt(pl['share_down'], '.0%')}) | "
                      f"{fmt(pl['mass_kept'], '.2f')} | {fmt(o.get('anchor_share'), '.2f')} |")
    if any("response_gate" in r for r in rows):
        mc = {**RGATE_DEFAULTS, **(ctx.cfg.get("response_gate") or {})}["max_cos"]
        md += ["", ("Response-direction gate (V12-44, report-only): median cosine of the one-step "
                    "whitened response to a 1 mm Franka +x / +y input with the direction to the "
                    "seated goal, deploy_i on iteration i's collect states; pass iff both <= "
                    f"{mc:g}."),
               "", "| iter | model | F+x | F+y | states | gate |", "|---" * 6 + "|"]
        for r in rows:
            q = r.get("response_gate")
            if q:
                md.append(f"| {r['iter']} | {r['model']} | {fmt(q.get('F+x'), '+.2f')} | "
                          f"{fmt(q.get('F+y'), '+.2f')} | {q['n_states']} | "
                          f"{'-' if q['pass'] is None else 'pass' if q['pass'] else 'FAIL'} |")
    if any("deploy_policy" in r for r in rows):
        met = {**DEPLOY_POLICY_DEFAULTS, **(ctx.cfg.get("deploy_policy") or {})}["metric"]
        md += ["", (f"Keep-best deployment (deploy_policy, {met}): theta_i = the model trained at "
                    "iteration i (the main table's model), its eval before collection, the model "
                    "deployed before it, and the model deployed for this iteration's collection "
                    "(eval column of the main table = theta_i)."), "",
               ("| iter | trained | trained eval | deployed before | deployed (iter, eval) | "
                "decision |"), "|---" * 6 + "|"]

        def sc(x):
            return "-" if not x or x["score"] is None else f"{x['score']}/{x['n']}"

        for r in rows:
            d = r.get("deploy_policy")
            if d:
                inc, dep = d["incumbent"], d["deployed"]
                md.append(f"| {r['iter']} | {d['trained']} | {sc(d['candidate'])} | "
                          + ("-" if not inc else f"{inc['id']} ({sc(inc)})")
                          + f" | **{dep['id']}** ({dep['iter']}, {sc(dep)}) | {d['decision']} |")
    if any("graph" in r for r in rows):
        md += ["", ("Cloud graph per refit (build_cloud_graph.py on B_i, iter<i>/graph/): nodes, "
                    "edges, episodes, rejected candidate edges, components (largest share), "
                    "unreachable (node, landmark) share, build s; held-out graph stress / "
                    "Spearman(d, D) of the refit's encoder vs this graph."), "",
               ("| iter | objective | nodes | edges | episodes | rejected | components | "
                "unreachable | build s | stress | spearman |"), "|---" * 11 + "|"]
        for r in rows:
            g = r.get("graph")
            if g:
                m = g.get("refit_metrics") or {}
                md.append(f"| {r['iter']} | {r.get('train_objective', '-')} | {g['n_nodes']} | "
                          f"{g['n_edges']} | {g['n_episodes']} | {g['rejected_frac']:.2%} | "
                          f"{g['n_components']} ({g['largest_component_frac']:.3f}) | "
                          f"{g['unreachable_frac']:.3f} | {g['build_s']:.0f} | "
                          f"{fmt(m.get('stress'), '.4f')} | {fmt(m.get('spearman'), '.3f')} |")
    if any("prefix_frac" in r for r in rows):
        md += ["", ("V12 curriculum: demo-prefix frac of the iteration (iter<i>/curriculum.json),"
                    " the decision that set it, prefix replays ok / rollouts, successes, "
                    "collection pool, --mppi-stages."), "",
               "| iter | prefix frac | decision | prefix ok / n | successes / n | pool | stages |",
               "|---" * 7 + "|"]
        for r in rows:
            if "prefix_frac" not in r:
                continue
            c = r.get("collect") or {}
            md.append(f"| {r['iter']} | {r['prefix_frac']:g} | {r['curriculum']['decision']} | "
                      f"{r.get('prefix_ok', '-')} / {c.get('n', '-')} | "
                      f"{r.get('successes', '-')}/{r.get('n_labelled', '-')} | "
                      f"{', '.join(r.get('pool') or []) or '-'} | {r.get('mppi_stages', '-')} |")
    bd = budget
    md += ["", (f"Budget (cumulative): demo {bd['rollouts_demo']}, excitation "
                f"{bd['rollouts_excitation']}, MPC {bd['rollouts_mpc']} "
                f"(+{bd['rollouts_mpc_skipped']} skipped), eval {bd['rollouts_eval']}; resets "
                f"scripted {bd['resets_scripted']} / human {bd['resets_human']} (eval "
                f"{bd['resets_human_eval']}); labels {bd['labels_human']}; robot "
                f"{bd['robot_s']:.0f} s; GPU {bd['gpu_s']:.0f} s; index.json counts "
                f"consistent: {bd['consistent']}."), "",
           f"**Stop rules:** {v}", ""]
    if g4:
        md += [f"**{g4}**", ""]
    return "\n".join(md) + "\n"


def plot(path: Path, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def frac(r):
        return r["successes"] / r["n_labelled"] if r.get("n_labelled") else np.nan

    it = [r["iter"] for r in rows]
    y = [frac(r) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(it, y, "o-", label="collection (human label)")
    ax[0].plot(it, [r["eval"]["engaged"] / r["eval"]["n"] if r.get("eval") else np.nan
                    for r in rows], "s", label="eval engaged (GT)")
    ax[0].plot(it, [r["eval"]["strict"] / r["eval"]["n"] if r.get("eval") else np.nan
                    for r in rows], "^", label="eval strict (GT)")
    ax[0].set(xlabel="iteration", ylabel="success fraction", ylim=(-0.05, 1.05))
    ax[0].legend()
    bx = [(r.get("budget") or {}) for r in rows]
    ax[1].plot([b.get("rollouts_mpc", np.nan) for b in bx], y, "o-")
    ax[1].set(xlabel="cumulative MPC rollouts", ylim=(-0.05, 1.05))
    ax[2].plot([b.get("robot_s", np.nan) for b in bx], y, "o-")
    ax[2].set(xlabel="cumulative robot s", ylim=(-0.05, 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- registry


def md_append_row(lines, header_prefix, row):
    h = next(n for n, ln in enumerate(lines) if ln.startswith(header_prefix))
    n = h + 2
    while n < len(lines) and lines[n].startswith("|"):
        n += 1
    lines.insert(n, row)


def register_model(ctx: Ctx, m: dict, log) -> bool:
    ydoc = REPO / "docs/models.yaml"
    text = ydoc.read_text()
    if any(x["id"] == m["id"] for x in yaml.safe_load(text)["models"]):
        print(f"  {m['id']} already registered", file=log)
        return False
    rep = jload(m["refit_report"])
    mt = rep["metrics"]
    tr = jload(ctx.it() / "trust_region.json") if (ctx.it() / "trust_region.json").is_file() \
        else None
    dep = next((jload(p) for p in (ctx.it() / "deploy.json", ctx.it() / "cand" / "deploy.json")
                if p.is_file() and jload(p)["model"] == m["id"]), None)  # keep_best: or none
    reg = ctx.cfg["registry"]
    tag = reg["tag"]
    entry = {"id": m["id"], "tag": tag,
             "run_dir": str(ctx.out).replace(str(HOME), "~"),
             "config": str(Path(m["refit_report"]).parent / "config.yaml").replace(str(HOME), "~"),
             "checkpoint": m["checkpoint"].replace(str(HOME), "~"), "step": rep["epochs"],
             "exports": [dep["deploy_dir"].replace(str(HOME), "~")] if dep else [],
             "date": reg["date"],
             "parent": m.get("parent"),
             "change": (f"belt-cloud PointNet encoder (pcd_belt_seg, {reg['cloud']} cloud"
                        f"{', ' + reg['model_note'] if reg.get('model_note') else ''}) "
                        "+ latent LCS, "
                        f"joint {'fresh' if m['iter'] == 0 or m.get('fresh') else 'warm'} "
                        f"refit, {rep['epochs']} "
                        f"epochs on the real-first buffer B_{m['iter']} ({rep['split']['n_train']} "
                        f"train / {rep['split']['n_heldout']} held-out episodes); box ub-tr"
                        f"{m['iter']}; MPPI node"),
             "encoder": "belt_cloud_pointnet", "u_bounds": f"ub-tr{m['iter']}",
             "status": "candidate", "status_note": "V11 real-first loop model",
             "results": {"offline": {
                 "heldout_one_step_h7_recon_mm": f"{mt['one_step_chamfer_mm']:.2f} / "
                                                 f"{mt['open_loop7_chamfer_mm']:.2f} / "
                                                 f"{mt['recon_chamfer_mm']:.2f}",
                 "rho_A": round(mt["rho_A"], 4), "B_fro_w": round(mt["B_fro_w"], 2),
                 "separation_auc": mt["separation_auc"],
                 "box_trans_rot_gm": None if tr is None else
                 f"{tr['ratios']['box_trans_gm']:.2f} / {tr['ratios']['box_rot_gm']:.2f}",
                 "source": m["refit_report"].replace(str(HOME), "~")}, "closed_loop": None}}
    block = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=1000)
    assert text.count("\nevals:\n") == 1
    ydoc.write_text(text.replace("\nevals:\n", "\n" + block + "evals:\n"))
    yaml.safe_load(ydoc.read_text())
    mdp = REPO / "docs/models.md"
    lines = mdp.read_text().splitlines()
    md_append_row(lines, "| ID | tag | old names", (
        f"| {m['id']} | `{tag}` | - | candidate | {m.get('parent') or '-'} | real-first loop "
        f"iteration {m['iter']}: belt-cloud encoder + LCS, joint "
        f"{'fresh' if m['iter'] == 0 or m.get('fresh') else 'warm'} refit on the buffer, box "
        f"ub-tr{m['iter']} |"))
    md_append_row(lines, "| ID | run folder", (
        f"| {m['id']} | `{ctx.out.name}` | `refit_from_buffer.py` | "
        f"`iter{m['iter']}/refit/{m['refit_tag']}` | - | epoch {rep['epochs']} | "
        + (f"`iter{m['iter']}/{Path(dep['deploy_dir']).name}`" if dep else "-")
        + f" | ub-tr{m['iter']} |"))
    mdp.write_text("\n".join(lines) + "\n")
    print(f"  registered model {m['id']}", file=log)
    return True


def register_eval(ctx: Ctx, ev: dict, log) -> bool:
    ydoc = REPO / "docs/models.yaml"
    text = ydoc.read_text()
    if ev["dir"] in {e["path"] for e in yaml.safe_load(text)["evals"]}:
        print(f"  eval {ev['label']} already registered", file=log)
        return False
    s = ev["summary"]
    reg = ctx.cfg["registry"]
    ctl = "C3+" if ev.get("controller") == "c3plus" else "MPPI"
    note = (f"V11 loop iteration {ctx.i}, {ctl} 2 stages (seed demo goals): {s['strict']}/{s['n']}"
            f" strict, {s['engaged']}/{s['n']} engaged, RMSE {s['rmse_mean_mm']:.1f} vs the demo "
            f"belt, unsafe {s['n_unsafe']}, solve p95 {s['solve_p95_ms'] or 0:.0f} ms")
    suite = reg["suite"]
    nc = len(eval_runs_of(ctx.cfg)) if ctx.cfg.get("eval_suite") else 6
    entry = {"label": ev["label"], "model": ev["model"], "suite": suite, "cost": "Qw",
             "limits": ev["box_id"], "controller": ev.get("controller", "mppi"), "path": ev["dir"],
             "date": reg["date"], "log_day": reg["date"], "n_cells": nc,
             "n_episodes": s["n"], "table": f"{ev['dir']}/table.md", "note": note}
    if f"  {suite}:" not in text:
        q = reg["suite_text"].replace("'", "''")
        text = text.replace("\ncosts:\n", f"\n  {suite}: '{q}'\ncosts:\n", 1)
    ydoc.write_text(text.rstrip("\n") + "\n" + yaml.safe_dump(
        [entry], sort_keys=False, allow_unicode=True, width=1000))
    yaml.safe_load(ydoc.read_text())
    mdp = REPO / "docs/models.md"
    lines = mdp.read_text().splitlines()
    if not any(ln.startswith(f"| | `{suite}`") for ln in lines):
        k = max(n for n, ln in enumerate(lines)
                if ln.startswith(("| | `seat6-1goal30`", "| | `seat6-rf")))
        lines.insert(k + 1, f"| | `{suite}` | {reg['suite_text']} |")
    md_append_row(lines, "| label | path | date", (
        f"| {ev['label']} | `{ev['dir']}` | {reg['date'][5:]} | {nc} / {s['n']} | `…/table.md` | "
        f"{note} |"))
    mdp.write_text("\n".join(lines) + "\n")
    print(f"  registered eval {ev['label']}", file=log)
    return True


def st_register(ctx: Ctx, dry: bool):
    if ctx.smoke:
        print("  register skipped (smoke)")
        return {"status": "skipped"}
    if dry:
        print("  would add the new model row (seed / accepted refit) and this iteration's eval "
              "rows to docs/models.{md,yaml}")
        return {}
    with open(ctx.logs() / "register.log", "a") as log:
        m = model_of(ctx, ctx.i)
        if m.get("new"):
            register_model(ctx, m, log)
        if (ctx.it() / "eval.json").is_file():
            register_eval(ctx, jload(ctx.it() / "eval.json"), log)
    return {}


# ---------------------------------------------------------------- main

FUNCS = {"seed": st_seed, "refit": st_refit, "trust": st_trust, "export": st_export,
         "collect": st_collect, "label": st_label, "buffer": st_buffer, "eval": st_eval,
         "deploy": st_deploy, "report": st_report, "register": st_register}


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


def run_all(ctx: Ctx, dry: bool):
    if ctx.i > 0 and not dry and not ctx.a.ignore_stop:
        lc = ctx.root / "learning_curve.json"
        v = jload(lc)["verdict"] if lc.is_file() else "continue"
        if not v.startswith("continue"):
            raise SystemExit(f"verdict after iteration {ctx.i - 1}: {v}; not starting "
                             f"iteration {ctx.i} (--ignore-stop to override)")
    if dry:
        print(f"# iteration {ctx.i}: collect {ctx.url_collect}, eval {ctx.url_eval}; root "
              f"{rel(ctx.root)}; lcs outputs {ctx.lout()}")
    steps = ["seed" if ctx.i == 0 else "refit", "trust", "export", "collect", "label", "buffer",
             "eval", "report", "register"]
    if keep_best(ctx.cfg):  # theta_i's eval decides the deployed model before collect
        steps = steps[:3] + ["eval", "deploy"] + steps[3:6] + steps[7:]
    for s in steps:
        do_step(ctx, s, dry)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[1:]))
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--step", choices=STEPS + ["all"], required=True)
    ap.add_argument("--root", default=None, help=f"default {V1} (smoke: "
                                                 "data/lcs/realfirst/smoke_loop)")
    ap.add_argument("--config", default=None, help="yaml merged over <root>/loop.yaml")
    ap.add_argument("--collect-port", type=int, default=None)
    ap.add_argument("--eval-port", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1 iteration, 2 rollouts, 2 refit epochs, 2 eval cells, URL 7764 only")
    ap.add_argument("--force", choices=STEPS, default=None)
    ap.add_argument("--ignore-stop", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if any(p in (7667, 8081, 8082) for p in (a.collect_port, a.eval_port) if p):
        raise SystemExit("reserved port")
    ctx = Ctx(a)
    if a.step == "all":
        run_all(ctx, a.dry_run)
    else:
        do_step(ctx, a.step, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
