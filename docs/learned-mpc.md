# Learned MPC: engineering log

Deploys `lcs_learning`'s trained latent LCS as the MPC model of magna's round-belt assembly
controller: the controller's `kMPC` phase plans with C3+ over the learned model instead of the
classic hand-built LCS, targeting a demonstration episode's states instead of a hand-set goal.
This doc is a day-by-day log of what was tried, what broke, and how it was fixed or left open.
For the wire contract, the params-yaml schema and step-by-step commands, see
`docs/learned-mpc-reference.md`. Where any of this and the code disagree, the code wins.

## Current status (2026-09-25)

**Best controller so far:** `honor_penalize_input_change: true`, `w_r 0.3`,
`demo_traj.w_p 0.03` (an EE-path cost) on the `v2_decoded_only` model — held-out engaged
22/40 = 0.55 vs the waypoint baseline's 14/28 = 0.50 (`eval-v2-honor`; the 95% CIs overlap, so
this is "on par," not a demonstrated win). The same ranking holds against a flat, UR-held
target (`flat-hold-eval`: matched 6/10 vs 3/10; held-out 17/32 vs 6/13). **The user has
rejected the EE-path cost** (`demo_traj.w_p`) as a design, because it costs the EE pose to the
demo's own path and grasps vary by construction — dropping it (`two-fixed-targets`, `w_p 0`)
collapses engagement to 1/10. So the result above is not adoptable as-is.

**Kept default (worktree `round_belt_controller_params_learned_eval.yaml`):** the latent-only
two-fixed-target setting (flat demo frames 13 / 59, both stages run to their 4 s / 6 s
timeouts, `w_p 0`): 3/10 engaged (2 strict) on the matched starts. Every other yaml is in
`systems/parameters/learned_archive/<date>/`.

**Current model:** `v2_decoded_only`, W&B run `nsxihz32`, epoch 300, trained on 464 episodes /
33,382 train tuples (`data/lcs/v2/split.json`).

**Main open issues:** the demo-tracking progress index stalls/deadlocks on a large fraction of
held-out starts; the goal tolerance does not separate `engaged` from `slanted`; no progress fix
has been found yet that does not lean on the rejected EE-path cost.

## 2026-09-25 — action redefinition, coverage re-collection, v2 training, deploy, eval, and the controller-failure diagnosis

**Summary:** Fixed the three causes the 2026-09-24 diagnosis named (non-causal UR action,
missing hold-still/UR-excited coverage, one-step training), added board-height constraints and
demo-trajectory tracking to the controller, retrained (`v2_decoded_only`), and evaluated. The
straightforward result was still negative (v2 full: 0/40 held-out). Root-causing that failure
found a C3 option-parsing bug; fixing it plus retuning got the first result on par with the
baseline, but only with an EE-path cost the user then rejected. A same-day UR-held/flat-target
demo and a fixed-two-latent-goal variant did not change the picture.

**Tried:**
- **Board-height constraints on the controller** (no retraining): augmented C3 state
  `x = [z; p_franka; p_ur]` with a hard world-z floor per arm. The naive `z_min_ur = plate + 2 mm`
  rule could not meet the target (board aborts stuck at 9-10/10 by construction, see Issues);
  an origin-based rule (demo's own minimum UR-origin height minus 3 mm) cut board aborts
  9/10 -> 4/10 on the pre-existing model, at the cost of 0 engaged (both arms pin on their floors).
- **Demo-trajectory tracking:** instead of one or two fixed latent goals, track the whole demo
  `z_demo[k0..k0+N]`, `k0` advancing by wall time or by nearest-latent progress. Progress mode +
  `admm_iter 2` was the first configuration to engage anything with the height constraints on:
  2/10, 1 strict, chamfer 8.9 mm (vs 0/10 for the two-stage baseline under the same constraints).
- **Action redefinition (`cmd_delta`):** Franka `knot1 - knot0` of the *published* trajectory,
  UR `line(t+dt) - line(t)` of the line actually in force, instead of `knot1 - measured` (which
  carries the tracking lag and was not causal for 4 of the 12 UR input dims, 2026-09-24). Added
  1.5 s pre-hold frames (`u = 0`) and UR OU excitation. Rewrote the old 320 episodes + the demo
  into the new definition (`rewrite_actions.py`) instead of re-collecting them.
- **Coverage re-collection:** 264 new episodes (nominal + a 12/13-state grasp-varied `set2`,
  excited + held-still each) plus the rewritten 320 old episodes -> 584 episodes / 33,382 train
  tuples after an 80/10/10 split. Causality gate (realised-vs-`u` slope in [0.6, 1.2]) passed on
  every dim; hold-still drift 0.14 mm/frame (0.80 mm total) on quiet rows.
- **Training:** two configs on the new data, `v2_decoded_only` (one-step) and `v2_multistep`
  (7-step open-loop loss through the differentiable PGD). `v2_decoded_only` (epoch 300, W&B
  `nsxihz32`): one-step RMSE 0.58 mm, h7 1.92 mm. `v2_multistep` failed its own pick rule (h7
  3.25 mm vs the 0.8x target 1.53 mm) after an epoch-11 loss collapse (see Issues) and was not
  exported; **`v2_decoded_only` was deployed.**
- **Deploy + eval-v2 (straightforward tuning, no controller fix):** height constraints (UR
  origin >= plate + 13.8 mm) + demo-trajectory progress tracking + `w_r 30` (carried over from
  the 2026-09-24 winner). Smoke: 0/2 engaged. Full eval on held-out grasps: **v2 full 0/40
  engaged vs baseline 14/28** — worse than doing nothing; both arms sit pinned on their height
  floors for the whole episode and the model finds no input that advances (§ Issues, root cause).
- **Controller-failure diagnosis (Phase A, offline; Phase B, sim):** found and fixed the C3
  `penalize_input_change` bug (below); with the bug honoured and `w_r` dropped from 30 to 0.3
  (R was dominating Q ~20x), plus a small EE-path cost (`demo_traj.w_p 0.03`, since the latent
  alone is nearly blind to Franka height), a 10-start sim screen went **6/10 engaged vs the
  waypoint baseline's 4/10** on the same starts — the first controller variant to beat the
  baseline on anything.
- **`eval-v2-honor` (full held-out re-run of the Phase-B winner):** **22/40 = 0.55 engaged vs
  baseline 14/28 = 0.50** (CIs [0.40, 0.69] vs [0.33, 0.67] — overlapping, "on par"). Chamfer is
  worse (6.7 vs 5.3 mm). 15/18 held-out failures are progress stalls at `k0` <= 19 of 59;
  `k0` >= 29 gives 22/25 engaged. One episode's outcome flipped between two otherwise-identical
  reps, showing the stall is solve-time-jitter sensitive, not fully deterministic.
- **Recording + replay of learned-MPC plans:** added an opt-in per-solve debug channel
  (`LEARNED_MPC_DEBUG`: `x_sol`, `u_sol`, `z_ref`, `p_ref`, scalars) and a viewer panel to see
  the planned belt, planned EE knots, planned actions and the fixed target belt against the
  recorded run (`docs/lcm-simulation.md` §10 "Learned-MPC layers").
- **UR-held, flatter demo:** found that every training episode (and the original demo) replays
  magna's waypoint yaml, which releases the UR grip at `place_3` — the eval harness holds both
  grippers throughout, so the MPC had been chasing a released-belt target while holding the
  belt. Swept UR-height offsets with the grip held; no setting reached <= 3 deg (see Issues);
  picked the flattest held setting (dz 0) and recorded a new demo, slant 3.65 deg.
- **`flat-hold-eval` (Phase-B winner against the new flat/held target):** ranking unchanged —
  matched n=10: learned 6/10 vs baseline 3/10 (same as the old target); held-out (everything
  that ran) learned 17/32 = 0.53 vs baseline 6/13 = 0.46. 0 episodes end at <= 3 deg in any
  condition (the demo itself is 3.65 deg). Same progress-stall failure mode as before.
- **`two-fixed-targets`:** per the user, re-ran with the EE-path cost removed (`w_p 0`, no
  `demo_traj`) and two fixed latent goals (`pre_place_2`, `place_3`) instead of the
  demo-trajectory reference. Result: **1/10 engaged** — worse than either the demo-traj winner
  (6/10) or the plain baseline (3/10). Root cause: the completion rule, not unreachable goals
  (see Issues).
- Long trainings and eval sweeps were launched detached (`setsid nohup`, driver `PPID 1`) so
  they keep running across an SSH disconnect — used for every multi-minute run this day.

**Issues / bugs -> resolution:**

| symptom | root cause | fix / status |
|---|---|---|
| UR MPC channel drives x by -1.6..-1.8 mm/step all episode, pulls the belt out (2026-09-24 finding) | UR action label `knot1 - measured` was dominated by the steady UR tracking offset (commanded vs measured), not by motion: several UR dims had slope -0.01..0.3 against realised motion | redefined the action as `cmd_delta` (Franka `knot1-knot0` of the published plan, UR `line(t+dt)-line(t)` of the line in force); causality gate now passes all 12 dims |
| model predicts ~0.65 whitened/step drift at `u=0` from the demo start; costs real inputs to fight a model artefact | training data had no held-still (`u=0`) frames to teach the model the true (near-zero) drift | added a 1.5 s pre-hold (`u=0`) to every collected episode; hold-still drift now 0.14 mm/frame measured, 0.067 whitened/step predicted |
| eval-v2 pre-flight: recorded UR `cmd_delta` action differs from the controller's actual `u0` by up to 0.29 mm (~5%) | the UR line used in the recorded delta starts 5 ms into the frame (`t+5ms`), so `line(t+dt)-line(t)` covers 70/75 ms of the true span, not the full `dt` | harness now uses "line end minus the frame's `ee_u`" for any exact-dt line starting in the frame (matches the Franka convention); UR `cmd_delta` == `u0` to float round-off |
| v2 full (straightforward tuning) engages 0/40; both arms pinned on their height floors the whole episode with no board aborts | `C3Options::penalize_input_change` is `std::optional<bool>`; C3 tests "has a value", not the value, so the yaml's explicit `false` turned the u-change penalty **ON** — every replan became a stiff proximal creep off the previous plan | added `honor_penalize_input_change` (opt-in, default off). Classic magna yamls set `penalize_input_change: true`, which behaves as intended, so they are unaffected; worth reporting upstream to C3 |
| even with the bug fixed, plans barely move (best scale of the demo's own actions ~0.05) | `w_r 30` (tuned on the buggy solver) made `R` dominate `Q` by ~20x once the penalty bug was fixed | dropped to `w_r 0.3` |
| latent-only reference gives no sweep / no notion of Franka height once off-manifold | the latent cost is nearly blind to Franka height after the descent (cos ~0.07 to the progress direction) | added `demo_traj.w_p 0.03` (EE-path cost) — **the user later rejected this fix** because it costs the EE to the demo's specific grasp path |
| demo-tracking progress stalls/deadlocks at low `k0` (early: `k0<=6`; mid: `k0` 13-19) on 15/18 held-out failures | the progress index only ever matches the *current* nearest demo frame; when the grasp offset puts the EE ahead of/behind the latent's read of progress, `k0` gets stuck | **open** — proposed fix: a time floor `k0 >= floor(s*t/dt)` or a progress index derived from EE-path distance too, not attempted |
| the naive `z_min_ur = plate + 2 mm` rule could not push board aborts below 9-10/10 (both in the first height-constraints package and again, independently, in the flat-hold-demo package) | conflated the board/pad clearance number (~2 mm) with the UR **tracking-origin** height above the plate (~14-17 mm) — the origin sits ~12-13 mm above the 2F-85 colliders it is meant to bound | switched to an origin-based rule (`plate + (demo's own minimum UR-origin height - 3 mm)`, later `- 0.8 mm`); recomputed per demo (11.63 / 13.1 / 13.8 / 16.08 mm across the day) |
| `v2_multistep` training: reconstruction loss jumps 0.004 -> 0.022 (RMSE, m) when the multistep loss switches on at epoch 11; 7-step rollouts of the random LCS reach 857 mm | multistep loss activated against a still-random-init LCS; the AE partially collapsed to compensate | training recovered to 0.007 by epoch 26 but ended at 2.19 mm recon vs the one-step model's 0.39 mm; `v2_multistep` failed its own pick rule and was **not exported** (deploy blocked separately on an unrelated exporter float32 tolerance) |
| the goal tolerance does not separate `engaged` from `slanted` (v2: 0.67 of slanted frames fall under it; the flat demo: 0.87; occurred once for real in `eval-v2-honor`) | the tolerance is calibrated as the engaged-episode p90 whitened distance, and slanted endings can land inside that radius too | **open**, noted as a caveat everywhere the tolerance is used; `two-fixed-targets` shows the sharpest version of it |
| `two-fixed-targets` engages only 1/10, worse than either alternative, even though the demo's own actions do reach the goals | the MPC's completion rule stops the episode once whitened distance falls under tol — which happens 2.0-4.7 s in, well before the belt is actually placed (dist-only completion fires on a belt still above the pulley) | **open** — the final stage needs a stop rule that separates "close in latent space" from "actually placed" |
| UR grip released at `place_3` in every training episode and the original demo; the eval harness holds both grippers throughout | the training/demo data replayed magna's own waypoint yaml verbatim, which commands a partial UR release at `place_3` | recorded a new UR-held demo (`--hold-ur-gripper`); swept UR height and found no setting reaches <= 3 deg slant (flattest held: dz 0, 3.70/3.72 deg mean/max); the new demo (episode 0) is 3.65 deg |
| harness recordings had an empty `targets.jsonl` (no plan/target data to replay) | `ControllerBridge` had no `take_targets()` method | added `take_targets()` (plan channels + the new debug channel) and an additive `meta.learned_mpc` provenance block |

**Decisions (user):**
- Action/data fixes for "smaller first": UR excitation on; convert the old 320 episodes + the
  demo to `cmd_delta` as extra training data; ~260 new episodes, 2 training configs (one-step +
  multistep); DAgger left undecided pending eval-v2's result.
- "Pls have one training that applies DAgger to see if there is any improvement" — made the
  DAgger round unconditional (later superseded).
- After eval-v2's negative result: "may be leave DAgger aside and investigate why the controller
  fails" — DAgger round put on hold; pursued the controller diagnosis instead.
- UR-held/flat demo: scoped to recording a new demo and evaluating the current v2 model only; a
  UR-held re-collection + retrain was explicitly left out of scope.
- Rejected the EE-path cost (`demo_traj.w_p`) as a controller design, because it costs the EE
  pose to the demo's specific grasp path while the whole point is to generalise across grasps;
  asked for a re-run with two fixed latent goals and no EE targets instead
  (`two-fixed-targets`, result: 1/10, worse — see Issues).

## 2026-09-24 — deploy the learned latent-LCS MPC in magna, targeting a demonstration episode

**Summary:** First end-to-end deployment: encoder, `LATENT_STATE` message, a C3+ controller
branch behind an opt-in `learned_mpc:` params block, a demo-driven two-stage goal, grasp-varied
start states and a lock-step eval harness. Tuned against the waypoint baseline on 13 grasp-varied
starts. **Result: negative** (engaged 11/26 vs baseline 14/26, only 2 strict; chamfer 11.8 vs
5.4 mm) — diagnosed why, in enough detail to drive the next day's fixes. Also validated the same
controller as a real procman stack on the free-running sim.

**Tried:**
- Exported the learned LCS from the `decoded_only` epoch-300 checkpoint (numpy sidecar encoder,
  no torch/C++ on the hot path; verified against the torch reference to <= 1e-5).
- Built the magna C++ integration: `LATENT_STATE` subscriber, `learned_mpc:` params block, a
  C3+ branch in `assembly_controller.cc`, checked first against a synthetic fixture, then the
  real export.
- Recorded a nominal (unperturbed) demo episode as the source of two staged goals
  (`z_demo[pre_place_1]`, `z_demo[place_3]`), with tolerance/timeout calibration derived from
  the training set's own same-state noise floor.
- Built 13 grasp-varied `pre_place_1` start states (`set1`) by perturbing the *pick*, not the
  belt (slide/roll the grasp along the belt's rest tangent, replay to the nominal target poses,
  keep only variants that still hold).
- Built the lock-step eval harness (OSC + a fresh controller per episode on one private URL) with
  three belt-alignment metrics (index-wise RMSE, chamfer, best-cyclic-shift RMSE).
- Ran untuned defaults on `set1`: **0/13 engaged, 13/13 aborted** (8 board, 5 grasp-lost).
- Diagnosed the failure (see Issues) and tuned per-diagnosis, not a grid: larger `w_r` (30),
  per-dim `u` bounds for the 4 non-causal UR dims, skip stage 0. Winner: **3/10 engaged, 1
  strict** on a 10-start tuning subset (still well under the baseline's 6-8/10 on the same
  starts); full 26-episode set1 run: **11/26 engaged (2 strict) vs baseline 14/26**, chamfer
  11.8 vs 5.4 mm, 24/26 board aborts.
- Validated the same controller under procman against the free-running sim (async processes,
  real-time pacing, the encoder as its own LCM node, 13.3 Hz latents): 2 live runs, both end
  `outside` (belt 66-71 mm from the large pulley at the stage-1 timeout).

**Issues / bugs -> resolution:**

| symptom | root cause | fix / status |
|---|---|---|
| untuned defaults abort 16/16 on "grasp lost" | inputs are effectively free (`R` cost <= 0.1/dim at `w_r 0.1`); C3 saturates most input dims most of the time (bang-bang) | raised `w_r` to 30 and added per-dim `u` bounds for the non-causal UR dims — reduces but does not eliminate it |
| the MPC drives UR x by -1.6..-1.8 mm/step for the whole episode, pulling the belt out | the UR action channel is not causal in the training data: the training line was not re-anchored at the measured pose every step the way the deployed line is, so a steady command-minus-measured offset was learned as "no motion" | **open** — fixed the next day (`cmd_delta`, 2026-09-25) |
| stage 0 never reaches its tolerance and *worsens* the real belt's alignment in every setting tried | the model has an autonomous phantom drift at the demo's `pre_place_1` state (`u=0` rollout drifts ~0.65 whitened/step) that the real (held) system does not have | **open** — fixed the next day (hold-still pre-hold data, 2026-09-25); the winner works around it by skipping stage 0 entirely (tol 5.0, reached on the first latent) |
| the 2F-85 contacts the board at ~1.8 s in 24/26 `set1` episodes | the MPC's planned poses have no board-clearance awareness (only the waypoint controller's own clamp guards that) | **open** — fixed (partially) the next day (`ee_constraints` height bounds, 2026-09-25) |
| a shared-group LCM leak on the controller's debug channels | `run_round_belt_assembly_controller` had no `--local_lcm_url` flag | added the flag; every run passes a private URL to both `--lcm_url` and `--local_lcm_url` |

**Decisions (user):**
- The MPC's goal must come from a real demonstration episode, staged (start, end), not a
  hand-set/mean-engaged goal.
- Grasp variation must perturb the *pick* pose (grasp), not the belt geometry, so the varied
  starts stay physically consistent.
- "Do B" — also deploy the same controller as a real procman stack against the free-running
  sim (async, real-time, hardware-shaped), not only the lock-step eval harness.

## Background (2026-09-22/23)

Earlier work built the data-collection pipeline this all rests on: the OSC-backend collector,
OU excitation, material belt-point sampling, and slant/grasp-variant scenarios. See
`handoffs/archive/RUN-STATE-2026-09-22-lcs-collection.md` and
`handoffs/archive/RUN-STATE-2026-09-23-lcs-slant-variants.md`; not detailed here.

## How to run

Full commands and flags: `docs/learned-mpc-reference.md` §5. The short version:

```bash
# 1. export the learned LCS (in ~/git/lcs_learning)
uv run python scripts/export_learned_lcs_deploy.py --checkpoint <ckpt.pt> --out-dir <deploy>

# 2. evaluate learned vs baseline on a grasp-variant set (this repo); the defaults are the
#    live params + its matching deploy/demo goals (see below)
uv run python scripts/lcs/eval_learned_mpc.py --mode learned --repeats 2 --record \
    --start-states data/lcs/start_states/grasp_variants/set1 \
    --lcm-url udpm://239.255.76.90:7690?ttl=0

# 3. replay a recorded run's plans/actions/targets
uv run python scripts/replay_viewer.py --recordings <run dir>/recordings --run <episode> \
    --port 8081 --learned-layers planned_belt,planned_ee,actions,target_belt
```

The live yamls in the worktree (`systems/parameters/`) are the latent-only, two-fixed-target
setting (no EE pose target): `round_belt_controller_params_learned_eval.yaml` (harness default)
on `learned_lcs/learned_lcs_v2_flat_pp2.yaml`, with the same `learned_mpc:` block in
`round_belt_controller_params_learned_sim.yaml` (procman). Its matching encoder deploy is
`deploy_v2_flat_pp2/deploy.npz`, with demo goals `data/lcs/demo_flat_pp2/demo_goals.npz`.
Every earlier params and LCS yaml named in this log is under
`systems/parameters/learned_archive/<date>/`, at its old subpath. The folder's README gives
each file's result.

Running the same controller as a real procman stack against the free-running sim (async
processes, real-time pacing, the encoder as its own LCM node) instead of the lock-step harness:
`docs/lcm-simulation.md` §3, subsection "Learned MPC with procman".
