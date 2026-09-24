# Learned MPC: driving the belt with a latent LCS toward a demonstration

Deploys `lcs_learning`'s trained latent LCS (`decoded_only`, epoch 300) as the MPC model of
magna's round-belt assembly controller: the controller's `kMPC` phase plans with C3+ over the
learned model instead of the classic hand-built LCS, targeting a demonstration episode's states
instead of a hand-set goal. Where this doc and the code disagree, the code wins.
**Bottom line, 2026-09-24: negative.** Tuned, the learned MPC does not beat the waypoint
baseline (§6); read that section before deploying anything from here.

## 1. What runs where

```
this sim (Newton)                magna (worktree magna-deploy-learned-lcs)
  camera cloud + belt bodies
  + FRANKA_STATE / UR_STATE  -->  scripts/lcs/latent_encoder_node.py (or the in-process
                                   encoder in the eval harness / offline tools)
                                     |
                                     v  LATENT_STATE (z in R^16, ee poses, proprio)
                                   run_round_belt_assembly_controller
                                     kMPC phase (learned_ branch): C3+ on the learned LCS,
                                     staged demo goals (stage 0 = pre_place_1, stage 1 = place_3)
                                     |
                        Franka trajectory (TARGET_CARTESIAN_POSE_TRAJECTORY)
                        UR line (UR_TARGET_CARTESIAN_POSE_TRAJECTORY)
                                     v
                              OSC / UR executors  -->  back into this sim
```

Everything up to `pre_place_1` is unchanged magna (compiled pick-and-place waypoints). From
`pre_place_1` the controller's `kMPC` branch is either the classic hand-built LCS (unmodified
if `learned_mpc:` is absent from the params yaml) or the learned path above. Encoder, message
and controller are all covered below; §7 covers running the whole thing under procman against
the free-running sim instead of the lock-step eval harness.

## 2. The `LATENT_STATE` contract

`src/task_common/latent_encoder.py`: `latent_state_message` / `parse_latent_state_message`.
Wire type `dairlib::lcmt_timestamped_saved_traj` (vendored both sides), channel `LATENT_STATE`
(`LATENT_STATE_CHANNEL`). One single-column `lcmt_trajectory_block` per name, `utime` = the
tick's `FRANKA_STATE` utime:

| block | size | meaning |
|---|---|---|
| `latent` | 16 (`n_x`) | `z`, the encoder's output |
| `ee_pose_franka` | 7 | `xyz_wxyz`, world, at the moment `z` was computed |
| `ee_pose_ur` | 7 | `xyz_wxyz`, world (UR tracking frame, same convention as the classic UR line) |
| `proprio` | 40 (`STATE_DIM`) | the training state vector |

Producers: the eval harness's in-process encoder (one message every `SAMPLE_STEPS` = 15 control
steps, before the UR and Franka state of that tick — `ControllerBridge.set_latent` /
`_send_latent` in `src/round_belt_task/controller_bridge.py`), or the standalone node
`scripts/lcs/latent_encoder_node.py` (§7). The controller (`assembly_controller.cc`, learned
branch) subscribes it on the `latent_state` port, replans only when `utime` changes, and warms
up on a held pose if no message has arrived yet.

## 3. The learned LCS yaml and the `learned_mpc:` params block

`LearnedLcsParams` / `LearnedMpcParams`
(`systems/controllers/parameter_headers/learned_lcs_params.h` in the worktree) load
`systems/parameters/learned_lcs/*.yaml` (copied verbatim from an `lcs_learning` export, plus a
2-line `# source` / `# sha256` header) and the params yaml's `learned_mpc:` block.

**LCS yaml** — `z' = A z + B u + D lam + d`, `0 <= lam _|_ E z + F lam + H u + c >= 0`:
`n_x=16, n_u=12, n_lam=8, N=7, dt=0.075`, `A/B/D/d/E/F/H/c`, `u_lb/u_ub` (12, training-action
percentiles), `z_std` (16, whitening), `goal_tol_whitened`, `goal_source` (string provenance:
`mean_engaged_final` or `demo:<path>#i,j`), `z_goal` (= `z_goal_stage2` when stages are
present), and — demo exports only — `z_goal_stage1`/`z_goal_stage2` (both or neither;
`Validate()` demands `z_goal_stage2 == z_goal` to 1e-9). `u = [dxyz_franka, dxyz_ur,
drotvec_franka, drotvec_ur]`, world frame, per `dt` (§4).

**`learned_mpc:` block** (controller params yaml; presence alone switches the `kMPC` phase to
the learned path — absent, the controller is byte-identical to classic):

| key | default | meaning |
|---|---|---|
| `lcs_file` | required | path to the LCS yaml above, relative to cwd |
| `latent_state_channel` | `LATENT_STATE` | |
| `w_q` / `w_r` | `1.0` / `0.1` | `Q = w_q diag(1/z_std^2)`, `R = w_r diag(1/((u_ub-u_lb)/2)^2)` |
| `g_x,g_lambda,g_u,g_eta` / `u_x,u_lambda,u_u,u_eta` | `0,2,0,1` / `0,10,0,1` | block-diagonal `G`/`U` per `z`/`lambda`/`u`/`eta` block, scaled by `w_g`/`w_u` |
| `w_g` / `w_u` | `0.2` / `0.5` | |
| `goal_tol` | `-1.0` | whitened; `< 0` uses the LCS file's `goal_tol_whitened` |
| `max_duration_s` | `6.0` | |
| `stage_goal_tols` / `stage_max_durations_s` | absent | one entry per `Stages()` (`[stage1, stage2]` if the LCS file has them, else `[z_goal]`); an entry `< 0` or the whole key absent falls back to `goal_tol`/`max_duration_s` |
| `penalize_input_change` | `false` | if true, C3's `R` term becomes `(u - u_prev_plan)^T R (...)` — not tried in the e2e pass (§6) |
| `admm_iter` / `rho_scale` / `gamma` | `3` / `3.0` / `1.0` | C3(+) solver knobs |
| `projection_type` | `C3+` | `C3+` or `QP` |
| `u_bound_scale` | `1.0` | multiplies `u_lb`/`u_ub` at load time |

`Stages()`/stage index reuse the classic multi-target machinery
(`mpc_current_target_idx`): the controller ends a stage when the whitened distance to its goal
drops below `StageGoalTolerance(stage)` or `StageMaxDuration(stage)` elapses (checked every
tick, so a stalled latent stream still times out), logs `[learned-mpc] stage <k>
reached|timeout t=<t> dist=<d>` (suffixed `(stage1)`/`(stage2)` only when there are 2 stages),
then either starts the next stage on the *next* latent (the stage-switch latent itself is
consumed without a solve) or, after the last stage, falls into the same `MPC completed!` ->
post-MPC path as the classic controller.

**`demo_goals.npz`** (`DemoGoals` in `latent_encoder.py`; written by
`scripts/lcs/make_demo_goals.py`, §5.5) is the eval harness's source of alignment ground truth
and per-stage tolerance/duration recommendations — a superset of the two `z_goal_stage*`
vectors baked into the LCS yaml: per stage, `stage_labels`, `stage_frames`, `z_goals`,
`goal_tols`, `max_durations_s`, `ee_pose_franka/ur`, `pcd_belt_stage` (150 material points),
`belt_xyz_stage` (48 bodies), `pulley_pose_stage`, `state_stage`; plus the whole demo's
`z`/`goal_dist` (T, n_stages) traces and provenance (`demo_episode`, `demo_sha256`,
`deploy_sha256`, `first_hold_place_3_frame`).

## 4. Semantics

- **`u = knot1 - measured`, per dim.** The Franka block of `u` is a pose delta in the *world*
  frame: position as `p1 - p0`, rotation as `rotvec(R1 . R0^T)`; the UR block is identical for
  the UR's tracking pose. "measured" is not the current tick's state but the state the
  `LATENT_STATE` message carries (`ee_pose_franka`/`ee_pose_ur`, i.e. the pose at the moment the
  encoder ran) — the only state the plan is computed against.
- **Franka: cumulative knots.** `knot_0 = ee_pose_franka` (the latent's pose); for `i=1..N-1`,
  `p_i = p_{i-1} + u_{i-1}[0:3]` and `q_i = exp(u_{i-1}[6:9]) . q_{i-1}` (left-multiplied,
  world frame). All `N=7` knots go out as one `TARGET_CARTESIAN_POSE_TRAJECTORY`, timestamped
  `t_context + i*dt`.
- **UR: 2-knot line over `dt`.** Only `u_0` (the first planned input) is used: target = the
  latent's UR pose (`ee_pose_ur`) plus `u_0[3:6]`/`u_0[9:12]`, transformed world -> UR base ->
  `tool0`. The line runs from the **currently measured** UR tool0 pose (read fresh from
  `UR_STATE` at trajectory-build time, not the latent's) to that target, over
  `max(dt, |dp|/v_lin, angle/v_ang)` (the classic controller's own linear/angular speed
  params) — so the line is re-anchored at the measured pose on every replan, unlike the Franka
  knots which chain off the latent's pose.
- **Replan only on a new latent.** `msg->utime` unchanged => the previous plan keeps
  publishing (`GenerateLearnedMpcTrajectory` returns early); the stage-switch tick is the one
  exception, which consumes a fresh latent to measure `last_goal_distance_` against the new
  goal but does not re-solve.
- **Stage 0 = align to `z_demo(pre_place_1)`, stage 1 = `z_demo(place_3)`.** `pre_place_1` is
  demo frame `0` (episode step 0, "the settled `pre_place_1` start, before any command moved
  the arms"); `place_3` is the **last** frame `T-1` (after the large-pulley latch, the 1.0 s
  dwell and the 0.5 s settle — the state `classify_episode` calls `engaged`). The demo's first
  `hold:place_3` frame is recorded as a diagnostic alternative
  (`first_hold_place_3_frame`, `demo_goals.npz`) but is **not** used as a goal.
- **Tolerance/timeout calibration** (`make_demo_goals.py`, §5.5): stage-1 tol = the export's
  `goal_tol_whitened` (engaged-episode p90 whitened distance to the goal). Stage-0 tol = 1.5x
  the p90 whitened distance from `z_goal_stage1` to frame 0 of every training episode (the
  same-snapshot noise floor, since every episode starts at the same `pre_place_1` state).
  Defaults `4.0 s` / `6.0 s`. The 2026-09-24 demo's numbers: stage-0 tol `0.0278` (p50/p90 floor
  `0.0152`/`0.0186`), stage-1 tol `0.9903`
  (`data/lcs/demo/demo_goals_report.json`, `recommended_learned_mpc_yaml`).
- **Eval-yaml convention:** `pre_mpc_motion: []`, so the MPC phase (and stage 0) starts at
  tick 0 from the start state — there is no separate "pre-MPC toward `pre_place_1`" phase to
  skip, because the start state already *is* `pre_place_1` (§5.6, the offline/OSC pick + the
  grasp-varied states are all built there).

## 5. How to

Runs `A -> B` below read left to right: export, check, record the demo, re-export with it,
build `demo_goals.npz`, build grasp variants, build the worktree, run the harness.

### 5.1 Export the learned LCS (in `~/git/lcs_learning`)

```bash
cd ~/git/lcs_learning
CKPT=outputs/sim_belt_ablation_20260924/ckpt_decoded_only
uv run python scripts/export_learned_lcs_deploy.py \
    --checkpoint $CKPT/2026-09-24/5afcg1zb/checkpoint_epoch_0300.pt \
    --out-dir $CKPT/deploy
```
Writes `learned_lcs.yaml`, `deploy.npz`, `reference_vectors.npz`, `report.json` under
`--out-dir` (default: `<checkpoint's ckpt dir>/deploy`). Without `--goal-episode`, `z_goal` is
the mean of the engaged training episodes' final frames (a diagnostic/fallback goal, not a
demonstration). `--u-percentile` (default `0.5`) and `--goal-percentile` (default `90.0`)
control the `u_lb/u_ub` and `goal_tol_whitened` percentiles.

### 5.2 Check the encoder (this repo)

```bash
uv run python scripts/checks/check_latent_encoder.py --deploy \
    ~/git/lcs_learning/outputs/sim_belt_ablation_20260924/ckpt_decoded_only/deploy/deploy.npz
```
E0-E7 (§8 table). `[SKIP]`, exit 0, if `--deploy` is absent — the check never requires a live
export.

### 5.3 Record the demo (this repo)

```bash
uv run python scripts/collect_lcs_dataset.py --out data/lcs/demo/<label> \
    --scenario nominal --episodes 3 --record \
    --lcm-url udpm://239.255.76.94:7694?ttl=0
```
`--scenario nominal` runs the unperturbed waypoints (no perturbation, no excitation); the demo
is the first `engaged` episode of the 3 repeats. Fallback rule (only if the nominal waypoints do
not engage): the smallest fixed `engaged`-band offset with excitation off that does engage,
recorded as a deviation in the run log. The chosen episode's `.npz` is copied to
`data/lcs/demo/demo_episode.npz`; its source recording directory is discovered by sha256 among
`data/lcs/demo/*/episode_*.npz` (never inside the training data dirs — the demo is trivially
held out of any checkpoint trained on data collected before it).

### 5.4 Re-export with the demo (in `~/git/lcs_learning`)

```bash
cd ~/git/lcs_learning
CKPT=outputs/sim_belt_ablation_20260924/ckpt_decoded_only
DEMO=/home/hienbui/git/magna_belt_simulation-main/data/lcs/demo/demo_episode.npz
uv run python scripts/export_learned_lcs_deploy.py \
    --checkpoint $CKPT/2026-09-24/5afcg1zb/checkpoint_epoch_0300.pt \
    --out-dir $CKPT/deploy_demo --goal-episode $DEMO --goal-frames 0,-1
```
`--goal-frames i,j` (negative indices allowed) picks the two demo frames encoded into
`z_goal_stage1`/`z_goal_stage2` (§4); the original mean-engaged `deploy/` export is untouched
and stays available as a fallback goal.

### 5.5 Build `demo_goals.npz` (this repo)

```bash
uv run python scripts/lcs/make_demo_goals.py --episode data/lcs/demo/demo_episode.npz \
    --deploy ~/git/lcs_learning/outputs/sim_belt_ablation_20260924/ckpt_decoded_only/\
deploy_demo/deploy.npz --out data/lcs/demo/demo_goals.npz
```
`--train-glob` (default the run's own training glob) selects the training episodes the stage-0
noise floor is measured over. Also writes `<out stem>_report.json` with the calibration and
`recommended_learned_mpc_yaml` values. Checked by `scripts/checks/check_demo_goals.py` D0-D3.

### 5.6 Build the grasp-varied start states (this repo)

```bash
# feasibility probe: single-knob sweeps, writes only <set>/probe.json
uv run python scripts/lcs/make_grasp_variants.py --probe --set set1

# the set: nominal + --count draws inside the probe's box
uv run python scripts/lcs/make_grasp_variants.py --set set1 --count 12 --seed 0
```
A "variant" perturbs the *grasp*, not the belt: each gripper's `pick` pose is slid along /
rolled about the rest belt's local tangent, the pick and lift are replayed (position backend) to
the **nominal** `pre_place_1` poses, the result settles 2 s under magna's OSC, and only variants
where both grippers still hold are kept. Writes
`data/lcs/start_states/grasp_variants/<set>/gv_<id>_osc.npz` plus `index.json` (per-variant
measured material slide/roll vs `pre_place_1_osc.npz`, the reference). The feasible box (no
`--box` override) is the probe's own full sweep range: franka slide +-20 mm, franka roll
+-15 deg, ur slide +-20 mm, ur roll +-15 deg (2026-09-24 probe, `set1/probe.json`) — rolls
transmit partially and noisily (the round tube rotates inside the 2F-85 jaw), slides transmit
faithfully. Checked by `scripts/checks/check_grasp_variants.py` G0-G3 (§8 table).

### 5.7 Build the worktree (magna, `magna-deploy-learned-lcs`)

```bash
cd /home/hienbui/git/magna-deploy-learned-lcs
bazel build //systems/controllers:run_round_belt_assembly_controller \
             //systems/controllers:learned_lcs_c3_check \
             //systems/controllers:franka_cartesian_osc_controller \
             //systems/controllers:ur_cartesian_trajectory_controller \
             //systems/simulation:ur_control_simulation \
             //systems/visualization:magna_visualization
```
Only the worktree has the learned branch and `--local_lcm_url`; the main magna checkout's
controller rejects that flag. Incremental builds only — never run two bazel commands in the
worktree at once. `learned_lcs_c3_check --params <yaml>` is the fastest offline sanity check of
a params/LCS-yaml pair: it solves the C3(+) problem `20` times per stage from `z_start_example`
and reports timing, bound violations and the plan/rollout distance to the goal, with no LCM, no
sim, no other process.

### 5.8 Run the harness (this repo)

```bash
# baseline waypoint controller, nominal start state, 3 repeats
uv run python scripts/lcs/eval_learned_mpc.py --mode baseline --repeats 3 \
    --label baseline-nominal --lcm-url udpm://239.255.76.90:7690?ttl=0

# baseline, grasp-variant set1, 2 repeats per variant
uv run python scripts/lcs/eval_learned_mpc.py --mode baseline --repeats 2 \
    --start-states data/lcs/start_states/grasp_variants/set1 \
    --label baseline-set1 --lcm-url udpm://239.255.76.90:7690?ttl=0

# learned MPC (tuned params, §6), grasp-variant set1, 2 repeats per variant, recorded
WINNER=learned_tuning/round_belt_controller_params_learned_eval_ubnd_wr30_s0skip_tol2.yaml
uv run python scripts/lcs/eval_learned_mpc.py --mode learned --repeats 2 --record \
    --params systems/parameters/$WINNER \
    --start-states data/lcs/start_states/grasp_variants/set1 \
    --label learned-set1 --lcm-url udpm://239.255.76.90:7690?ttl=0
```
`--start-states` defaults to `data/lcs/start_states/pre_place_1_osc.npz` ("nominal"); pass a
grasp-variant set dir instead to sweep `index.json`'s variants (`--variants` restricts to a
comma list). `--deploy`/`--demo-goals` default to the demo export (§5.4/§5.5); without
`--demo-goals`, alignment is null and `sim_goal_dist` is measured against the LCS file's own
`z_goal`/`stage_goals` instead. `--params` defaults to the mode's `*_eval.yaml`
(`round_belt_controller_params_{learned,baseline}_eval.yaml`); pass a params-yaml copy to try a
different `learned_mpc:` tuning (§6) without touching the shipped file. The Franka OSC starts
once per run; each episode restarts only a fresh controller process next to it, on one private
URL — always pass `--lcm-url` (`eval_learned_mpc.py`'s own default is `...88:7688`, only safe
for one run at a time). Every run's episodes are written in dataset format (`.npz` +
`index.json`) under `data/lcs/mpc_eval/<label>/`; `--dry-run` prints the episode plan
without touching LCM or the sim. Checked end to end by `scripts/checks/check_mpc_harness.py`
H0-H6 (§8 table).

### 5.9 Read `index.json`

`summary`: `engaged`/`engaged_rate`/`engaged_ci95` (Wilson 95% interval over classified
episodes — the classifier's `engaged` label, which counts an aborted episode's *final* belt
state, not whether the run finished cleanly), `outcomes` (label counts), `aborted`/`timeouts`,
`alignment_final` (per metric: `n`/`median`/`mean` over episodes with a demo to align to) and
`goal_dist_final` (whitened distance to each stage's goal at the last frame, learned mode only).
`alignment_final` has three belt metrics (`src/task_common/belt_metrics.py`), all in mm,
comparing the final `pcd_belt` (150 material-ordered points) to the demo's stage-2 belt:
**index-wise RMSE** (same material point vs same material point — only meaningful when the
grasp holds the same material offset as the demo, since a shifted grasp shifts which point is
"point 0"), **chamfer** (symmetric nearest-neighbour distance — shape only, indifferent to a
material-index shift) and **best-cyclic-shift RMSE** (the minimum index-wise RMSE over every
cyclic roll of one belt against the other, plus the winning shift — approximately the material
offset between the two grasps). Per-episode rows carry the same three metrics for `initial`
(first frame, i.e. how far off the start state already is) and `final`.

## 6. Results (2026-09-24, `PKG-20260924-e2e-learned-mpc`)

**Negative: the tuned learned MPC does not beat the waypoint baseline.** All numbers below are
from `handoffs/RUN-STATE.md` (`PKG-20260924-e2e-learned-mpc` entries) and the run dirs under
`data/lcs/mpc_eval/` (kept, all 146 episodes). "Strict" = engaged and not aborted.

| condition | n | eng/over/under/slant/other | engaged rate [95% CI] | strict | rmse / chamfer / shift mm | aborts |
|---|---|---|---|---|---|---|
| baseline set1 x2 | 26 | 14/0/2/8/2 | 0.54 [0.35, 0.71] | 14 | 9.1 / 5.4 / 8.4 | 0 |
| **learned (winner) set1 x2** | 26 | **11**/14/0/1/0 | 0.42 [0.26, 0.61] | **2** | 18.5 / 11.8 / 17.6 | 24 (board) |
| baseline nominal x3 | 3 | 3/0/0/0/0 | 1.00 [0.44, 1.00] | 3 | 4.9 / 3.3 / 4.9 | 0 |
| learned (winner) nominal x3 | 3 | 0/3/0/0/0 | 0.00 [0.00, 0.56] | 0 | 18.5 / 12.1 / 18.4 | 3 (board) |
| learned (untuned defaults) set1 x1 | 13 | 0/13/0/0/0 | 0.00 [0.00, 0.23] | 0 | 76.0 / 55.2 / 74.9 | 13 (board 8, grasp 5) |

Run dirs: `20260924-131014-baseline-set1`, `20260924-131302-baseline-nominal`,
`20260924-133132-final-learned-set1`, `20260924-133342-final-learned-nominal`,
`20260924-131328-learned-default-set1`. "Winner" params:
`learned_tuning/round_belt_controller_params_learned_eval_ubnd_wr30_s0skip_tol2.yaml` (tuning
below).

**11/26 engaged, only 2 of those strict, against the baseline's 14/26** — the learned policy's
"engaged" episodes are mostly the classifier scoring an *aborted* run's final belt position, not
a controller that finished the episode driving toward the goal.

### Diagnosis (why, before tuning — untuned defaults aborted 16/16 on "grasp lost")

1. **`u`/knot convention verified correct** — no sign or frame bug (every commanded dim has a
   positive realised-motion slope, Franka 0.74-1.02, UR x 2.0, rot 0.84-1.1).
2. **Inputs are free -> bang-bang.** `R` costs <= 0.1/dim at `w_r=0.1` against `dist^2 ~ 90` at
   the stage-1 start; C3 saturates most input dims most of the time (Franka rot 75-95% of
   steps) and holds some there for ~50 consecutive steps (~140 deg cumulative rotation).
3. **The UR action channel is not causal in the training data.** Fitting realised motion = a.u
   against the 19.7k training rows: UR x slope -0.01 (corr -0.03), y 0.16, rot-y 0.3, rot-z
   0.06 — those four dims are a steady command-minus-measured *offset* in the data the model was
   trained on, not a real motion, because the training UR line was **not** re-anchored at the
   measured pose every step the way the deployed controller's line is (§4). Deployed, the same
   `u` moves the UR ~1:1, so the model's learned (near-zero) response to those dims is wrong:
   the MPC drives UR x by -1.6..-1.8 mm/step for the whole episode (its interior optimum),
   -132 mm over 5 s against the demo's -7 mm, and the gripper separation goes
   146 -> 86 -> 200 mm (demo: 146 -> 183 mm, monotone) — the fingers are forced apart and the
   belt is pulled out.
4. **The model predicts phantom drift at the stage-0 start.** With `u=0`, the learned LCS
   rolled out from the demo's own `pre_place_1` state drifts ~0.65 whitened/step — the real
   system does not drift there (it is a held snapshot). So holding `z` near
   `z_demo(pre_place_1)` costs real, non-zero inputs on the actual arms purely to fight a
   model artefact; stage 0 never reaches its 0.028 tolerance and *worsens* the real belt's
   alignment to the demo in every setting tried (belt RMSE to the demo's `pre_place_1` belt goes
   8-11 mm -> 43-87 mm; improved in 0-10% of episodes). One-step whitened prediction error is
   0.14-0.26 open-loop but 0.5-1.0 in the first closed-loop second and 2.4-3.9 rms per episode
   once bang-bang inputs push the real state off the encoder's training manifold (the
   checkpoint's own decoder's belt reconstruction error goes from 8 mm to 140 mm).
5. **The gripper hits the board.** With the tuned winner, the UR overshoots the demo's -80 mm
   z travel and the 2F-85 contacts the board at ~1.8 s in **24/26** `set1` episodes (the harness
   classifies this as an "aborted" episode, kept, not discarded — see the `aborts` column
   above); the collector's board-clearance clamp (`docs/lcs-data-collection.md` §5.1) guards the
   waypoint controller's targets, but the learned MPC's planned poses have no such guard.
6. Latency is not the cause (plans answer latents 12-25 ms sim-late on average, well under one
   knot). `penalize_input_change` was not tried (it removes the *magnitude* penalty, not the
   saturation).

### Tuning (10 grasp variants x baseline + learned, chosen per-diagnosis, not a 1-knob grid)

| variant | `lcs_file` u bounds | `w_r` | stage 0 | engaged / strict (of 10) | chamfer med mm | board aborts |
|---|---|---|---|---|---|---|
| wr100 | training p5/p95 | 100 | on | 0 / 0 | 17.1 | 8 |
| ubnd | training p5/p95 for causal dims, **realised**-motion p5/p95 for UR x/y/rot-y/rot-z | 0.1 | on | 0 / 0 | 28.1 | 6 |
| ubnd_wr30_s0skip | ubnd | 30 | **skipped** (tol 5.0) | 3 / 0 | 11.5 | 10 |
| **ubnd_wr30_s0skip_tol2 (winner)** | ubnd | 30 | skipped (tol 5.0) | 3 / **1** | 11.4 | 9 |
| wr100_s0skip | training p5/p95 | 100 | skipped | 1 / 0 | 14.1 | 10 |
| ubnd_wr30_s0skip_ub05 | ubnd, `u_bound_scale: 0.5` | 30 | skipped | 0 / 0 | 14.4 | 10 |

Each variant's params yaml is
`systems/parameters/learned_tuning/round_belt_controller_params_learned_eval_<variant>.yaml` in
the worktree (plus `wr10`, screened but not in this table).
`round_belt_controller_params_learned_sim.yaml` carries the winner's `learned_mpc:` block
verbatim.

Skipping stage 0 (tol `5.0` reaches on the very first latent) and raising `w_r` to `30` (leaves
saturation) plus the per-dim `u_lb/u_ub` fix for the 4 non-causal UR dims (new LCS yaml copy
`decoded_only_epoch0300_demo_ubnd.yaml`) together move engaged 0->3 of 10 and strict 0->1 of 10
— still far short of the baseline's 6-8/10 on the same variants. By offset magnitude
(`max(|slide mm|)`): under 5 mm the baseline holds 6/6 vs the tuned learned's 4 engaged / 0
strict; >= 12 mm the baseline drops to 0/12 while the tuned learned reaches 4/12 (2 strict) —
the only regime where the learned policy adds anything, at large material offsets the waypoint
baseline cannot reach at all (`gv_02`, `gv_07`).

### Interpretation

1. No, the tuned learned MPC does not beat the waypoint baseline on engaged rate (0.42 vs 0.54)
   or alignment (chamfer 11.8 vs 5.4 mm); it only wins where the baseline already fails, at
   large material offsets.
2. Stage 0 never helps (point 4 above); the winning config skips it entirely.
3. Closed-loop model error only matches open-loop once inputs stay in the training range;
   bang-bang inputs (untuned) push the state off the encoder's manifold.
4. The MPC's latent-space path is not the demo's path (both arms move down together instead of
   the demo's lower-then-wrap order) and has no notion of the ~2 mm board clearance the demo
   keeps — hence the board hits.
5. The material-indexed belt input is *not* the binding limit here (alignment is flat against
   grasp offset for the learned policy); the binding limits are the non-causal UR channel, the
   stage-0 drift artefact, and too few goal frames. Next steps (not attempted): intermediate
   demo goal frames (`place_1`/`place_2`) as extra stages, retraining with a UR action defined
   as commanded motion (not line-minus-measured), and a board-height penalty/constraint in the
   MPC itself.

## 7. Procman deployment (option B)

Running the same learned controller as a real procman stack against the free-running sim (async
processes, real-time pacing, the encoder as its own LCM node) is documented in
`docs/lcm-simulation.md` §3, subsection "Learned MPC with procman" — including the pmd, the URL
policy, the sim's opt-in perception publishers, the encoder node's flags, the shell validation
tool (`scripts/lcs/run_learned_stack.py`) and the two measured live runs. Not duplicated here.
Its live-run outcome is consistent with §6: the grasp holds, but the policy does not engage
(`outside`, belt 60-110 mm from the large pulley at the stage-1 timeout).

## 8. Limits and next steps

- **Belt input is material-indexed**, not the binding limit in §6, but still a real risk: a
  grasp that shifts the held material point changes `z` even when the belt geometry matches the
  demo, so stage 0 is not fully reachable by construction for a shifted grasp. The best-shift
  RMSE metric (§5.9) exposes this independently of the §6 failure modes.
- **Hardware node.** `scripts/lcs/latent_encoder_node.py` is the same module on hardware
  channels (`POINT_CLOUD_CROPPED`, `RoundBeltState`, `UR_STATE` with `--state-match nearest`).
  The model was trained on the 48 belt-body centres (`--belt-input bodies48`, the default); the
  hardware estimator's 150 unordered vertices need `--belt-input points150` and a model trained
  on that input — not solved here (`docs/lcm-simulation.md` §8, the node's check row).
- **C++ encoder** was deferred; the numpy sidecar (§2) is the only implementation, in-process in
  the eval harness and as a standalone LCM node.
- **`--local_lcm_url`** (worktree controller only) carries the controller's `local_lcm`
  traffic (C3 debug/state channels, `OSC_TARGET_TRACKING_DEBUG`). Both `--lcm_url` and
  `--local_lcm_url` default to the shared group; pass a private URL to both, as every run in
  §5/§6/§7 does (`docs/lcm-simulation.md`, "URL policy").

## 9. Private LCM groups (this run, harness + tooling side, `7688`-`7694`)

Never `239.255.76.67:7667` (magna's shared group). `7695`-`7697` (the procman stack, its shell
validation and the encoder-node check) are documented with §7 / `docs/lcm-simulation.md` §8.

| URL | user |
|---|---|
| `udpm://239.255.76.88:7688?ttl=0` | `scripts/lcs/eval_learned_mpc.py` (own default; pass an explicit URL for concurrent runs) |
| `udpm://239.255.76.89:7689?ttl=0` | `scripts/checks/check_mpc_harness.py` |
| `udpm://239.255.76.90:7690?ttl=0` | the 2026-09-24 e2e evaluation runs (§6), passed explicitly |
| `udpm://239.255.76.91:7691?ttl=0` | ad hoc magna launch/smoke checks (worktree build verification) |
| `udpm://239.255.76.92:7692?ttl=0` | `scripts/lcs/make_grasp_variants.py` (own default) |
| `udpm://239.255.76.93:7693?ttl=0` | `scripts/checks/check_grasp_variants.py` |
| `udpm://239.255.76.94:7694?ttl=0` | the demo recording (`collect_lcs_dataset.py --scenario nominal`, §5.3), passed explicitly |

Every magna binary on one of these exposes `--lcm_url`, and the controller additionally needs
`--local_lcm_url` (only the worktree's controller has that flag) — omit either on even one
process and it falls back onto the shared group (see `docs/lcm-simulation.md`'s private-URL
recipe, which this doc's tooling follows).
