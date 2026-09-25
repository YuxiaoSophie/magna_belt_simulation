# Learned MPC: reference (contract, params, how-to)

Long-form reference for the learned latent-LCS MPC work: the wire contract, the params-yaml
schema, the u/knot semantics and the step-by-step commands. For current status, day-by-day
progress and issues, see `docs/learned-mpc.md`. Where this doc and the code disagree, the code
wins.

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
                                     staged/demo-traj goals from a demonstration episode
                                     |
                        Franka trajectory (TARGET_CARTESIAN_POSE_TRAJECTORY)
                        UR line (UR_TARGET_CARTESIAN_POSE_TRAJECTORY)
                                     v
                              OSC / UR executors  -->  back into this sim
```

Everything up to `pre_place_1` is unchanged magna (compiled pick-and-place waypoints). From
`pre_place_1` the controller's `kMPC` branch is either the classic hand-built LCS (unmodified
if `learned_mpc:` is absent from the params yaml) or the learned path above. §7 covers running
the whole thing under procman against the free-running sim instead of the lock-step eval
harness.

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

## 3. The learned LCS yaml, the `learned_mpc:` params block, and the debug channel

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
| `penalize_input_change` | `false` | **bug (2026-09-25): `std::optional<bool>` in C3, and C3 tests "has a value", not the value — so `false` turns the penalty ON.** Use `honor_penalize_input_change` (below) to get the literal behaviour |
| `honor_penalize_input_change` | `false` | opt-in fix: when true, C3 receives `std::nullopt` for `false` / `true` for `true`, i.e. the field is honoured; classic magna yamls never set this field and are unaffected either way |
| `admm_iter` / `rho_scale` / `gamma` | `3` / `3.0` / `1.0` | C3(+) solver knobs |
| `projection_type` | `C3+` | `C3+` or `QP` |
| `u_bound_scale` | `1.0` | multiplies `u_lb`/`u_ub` at load time |
| `ee_constraints` | absent | `LearnedEeConstraints {enabled, z_min_franka, z_min_ur, z_max_franka, z_max_ur (±inf), w_p (0)}`: one-sided world-z floor/ceiling on the augmented EE state `x = [z; p_franka; p_ur]` (`A_aug = blkdiag(A, I6)`, `B_aug = [B; S]`, positions carried in mm internally), bound as a hard C3 `STATE` constraint on `x_1..x_{N-1}` |
| `ur_line_exact_dt` | `false` | the UR line spans exactly one `dt` instead of the classic speed-limited duration |
| `demo_traj` | absent | `LearnedDemoTraj {file, ref_mode time\|progress, time_scale 1, progress_window 10, w_p 0, complete_on_final_dist false}`: tracks a whole demo trajectory (`z_demo[k]`, EE positions) instead of one or two fixed goals; `ref_mode: progress` finds `k0` by nearest whitened distance in `[k_prev, k_prev+window]` (monotone); `w_p > 0` costs the EE position error to the demo's path (requires `ee_constraints`); `complete_on_final_dist` lets the stage end on distance to the final demo frame alone, without requiring `k0 == T-1` |
| `debug_channel` | absent | set: one `dairlib::lcmt_timestamped_saved_traj` per learned solve on this channel of `--lcm_url` (layout below); absent: nothing is published |

**`debug_channel`** (opt-in, worktree only; e.g. `round_belt_controller_params_learned_eval.yaml`
sets `LEARNED_MPC_DEBUG`). Published once per solve (a forced publisher that skips repeated utimes),
including failed solves (`solve_ok` 0); `utime` = the latent's utime. Blocks (rows x columns,
`time_vec` = the plan's knot times `t_context + i*dt`, OSC clock):

| block | shape | content |
|---|---|---|
| `x_sol` | `n_x x (N+1)` | C3's state solution `x_0..x_{N-1}` plus `x_N` = the model step from `x_{N-1}` (C3 keeps N states); rows `[z (16); p_franka (3); p_ur (3)]` with the EE rows in metres (C3's mm scaling undone). A QP solution: `x_0` and the augmentation hold to OSQP tolerance (~1e-7..1e-4), not exactly |
| `u_sol` | `12 x N` | the planned inputs; `u_sol[:, 0]` is what was executed (Franka knot 1 - knot 0, UR line end - latent pose) |
| `z_ref` | `16 x (N+1)` | the latent references passed to `UpdateTarget` (demo frames `min(k0+i, T-1)`, or the stage goal repeated) |
| `p_ref` | `6 x (N+1)` | the demo's Franka/UR EE positions at those frames when tracking the demo, else the current ones |
| `scalars` | `10 x 1` | rows named in `datatypes`: `k0` (-1 without demo), `stage`, `dist_ref` (whitened, to `z_ref[:, 0]`), `dist_final` (to the last goal), `solve_ms`, `t_mpc_start`, `ref_mode` (0 time, 1 progress, -1 none), `n_x`, `N`, `solve_ok` |

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

- **`u = knot1 - measured`, per dim** (the default `knot1_minus_measured` definition; the
  collector/harness default since 2026-09-25 is `cmd_delta`, defined the same way but re-anchored
  at the *commanded* pose so it cancels tracking lag — see `docs/lcs-dataset.md` §3). The Franka
  block of `u` is a pose delta in the *world* frame: position as `p1 - p0`, rotation as
  `rotvec(R1 . R0^T)`; the UR block is identical for the UR's tracking pose. "measured" is the
  state the `LATENT_STATE` message carries (`ee_pose_franka`/`ee_pose_ur`, i.e. the pose at the
  moment the encoder ran) — the only state the plan is computed against.
- **Franka: cumulative knots.** `knot_0 = ee_pose_franka` (the latent's pose); for `i=1..N-1`,
  `p_i = p_{i-1} + u_{i-1}[0:3]` and `q_i = exp(u_{i-1}[6:9]) . q_{i-1}` (left-multiplied,
  world frame). All `N=7` knots go out as one `TARGET_CARTESIAN_POSE_TRAJECTORY`, timestamped
  `t_context + i*dt`.
- **UR: 2-knot line over `dt`.** Only `u_0` (the first planned input) is used: target = the
  latent's UR pose (`ee_pose_ur`) plus `u_0[3:6]`/`u_0[9:12]`, transformed world -> UR base ->
  `tool0`. Without `ur_line_exact_dt`, the line runs from the **currently measured** UR tool0
  pose (read fresh from `UR_STATE` at trajectory-build time, not the latent's) to that target,
  over `max(dt, |dp|/v_lin, angle/v_ang)` (the classic controller's own linear/angular speed
  params); with the flag, the line always spans exactly `dt` — so the line is re-anchored at the
  measured pose on every replan, unlike the Franka knots which chain off the latent's pose.
- **Replan only on a new latent.** `msg->utime` unchanged => the previous plan keeps
  publishing (`GenerateLearnedMpcTrajectory` returns early); the stage-switch tick is the one
  exception, which consumes a fresh latent to measure `last_goal_distance_` against the new
  goal but does not re-solve.
- **Two ways to define the goal:** fixed stage goals (`Stages()`, one or two `z_goal_stage*`
  vectors — stage 0 aligns to the demo's start, stage 1 to its end) or a full trajectory
  (`demo_traj`, tracking `z_demo[k0..k0+N]` with `k0` advancing by wall time or by nearest-latent
  progress). See `docs/learned-mpc.md`'s day log for which was tried when and why.
- **Tolerance/timeout calibration** (`make_demo_goals.py`, §5.5): stage-1 tol = the export's
  `goal_tol_whitened` (engaged-episode p90 whitened distance to the goal). Stage-0 tol = 1.5x
  the p90 whitened distance from `z_goal_stage1` to frame 0 of every training episode (the
  same-snapshot noise floor, since every episode starts at the same `pre_place_1` state).
  Defaults `4.0 s` / `6.0 s`. **Caveat found 2026-09-25: this tolerance does not separate
  `engaged` from `slanted`** — a sizeable fraction of slanted endings also fall under it (v2:
  0.67 of slanted frames; the 2026-09-24 demo: 0.11) — so a controller that completes on
  whitened distance alone can stop on a slanted belt.
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
demonstration). `--goal-episode <demo> --goal-frames i,j` (negative indices allowed) instead
encodes the demo's two frames into `z_goal_stage1`/`z_goal_stage2` (§4); `--decoder-out` also
writes `decoder.npz` (the belt decoder, for the replay viewer, §5.10); `--data-glob` picks which
collected episodes set `u_lb/u_ub` and `goal_tol_whitened` (pass the glob(s) matching the
`action_definition` you are exporting for — the exporter's own default is stale knot1 data).
`--u-percentile` (default `0.5`) and `--goal-percentile` (default `90.0`) control the
`u_lb/u_ub` and `goal_tol_whitened` percentiles.

### 5.2 Check the encoder (this repo)

```bash
uv run python scripts/checks/check_latent_encoder.py --deploy \
    ~/git/lcs_learning/outputs/sim_belt_ablation_20260924/ckpt_decoded_only/deploy/deploy.npz
```
E0-E8 (§8 of `docs/lcm-simulation.md`). `[SKIP]`, exit 0, if `--deploy` is absent — the check
never requires a live export.

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
`<out>/demo_episode.npz`; its source recording directory is discovered by sha256 among
`<out>/*/episode_*.npz` (never inside the training data dirs — the demo is trivially held out
of any checkpoint trained on data collected before it). `--hold-ur-gripper` keeps the UR closed
through `place_3` instead of magna's own partial-release command (2026-09-25: every training
episode releases the UR at `place_3`, so a UR-held demo/target is out of that distribution —
`docs/learned-mpc.md`, 2026-09-25); `--nominal-ur-dz-mm A[:B]` raises the nominal UR waypoint
z by A mm at `pre_place_2` and B (default A) at `place_3`, for a UR-height sweep.

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
and stays available as a fallback goal. Pick whichever frame indices match the demo's own
`sim_phase` labels for the two stages you want (e.g. `0,59` for start/end, `13,59` for
`move:place_3`/end — the exporter does not look these up itself).

### 5.5 Build `demo_goals.npz` (this repo)

```bash
uv run python scripts/lcs/make_demo_goals.py --episode data/lcs/demo/demo_episode.npz \
    --deploy ~/git/lcs_learning/outputs/sim_belt_ablation_20260924/ckpt_decoded_only/\
deploy_demo/deploy.npz --out data/lcs/demo/demo_goals.npz
```
`--train-glob` (default the run's own training glob) selects the training episodes the stage-0
noise floor is measured over — pass the glob matching your `action_definition`. Also writes
`<out stem>_report.json` with the calibration and `recommended_learned_mpc_yaml` values.
`--traj-out PATH` (additive; makes `--out` optional) instead writes a whole-episode
`demo_traj.yaml` (`z`, EE positions, `dt`) for `demo_traj` tracking (§3). Checked by
`scripts/checks/check_demo_goals.py` D0-D3.

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
faithfully. Checked by `scripts/checks/check_grasp_variants.py` G0-G3 (§8 of
`docs/lcm-simulation.md`). `--start-states PATH --variants a,b` on the collector
(`scripts/collect_lcs_dataset.py`) restores episodes from a set of held states instead of the
usual snapshot, for collecting grasp-varied training coverage.

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
(or from `--diag_x0_file` states, a 2026-09-25 diagnostic addition) and reports timing, bound
violations and the plan/rollout distance to the goal, with no LCM, no sim, no other process.

### 5.8 Run the harness (this repo)

```bash
# baseline waypoint controller, nominal start state, 3 repeats
uv run python scripts/lcs/eval_learned_mpc.py --mode baseline --repeats 3 \
    --label baseline-nominal --lcm-url udpm://239.255.76.90:7690?ttl=0

# baseline, grasp-variant set1, 2 repeats per variant
uv run python scripts/lcs/eval_learned_mpc.py --mode baseline --repeats 2 \
    --start-states data/lcs/start_states/grasp_variants/set1 \
    --label baseline-set1 --lcm-url udpm://239.255.76.90:7690?ttl=0

# learned MPC (the default params/deploy/demo goals), grasp-variant set1, 2 repeats, recorded
uv run python scripts/lcs/eval_learned_mpc.py --mode learned --repeats 2 --record \
    --start-states data/lcs/start_states/grasp_variants/set1 \
    --label learned-set1 --lcm-url udpm://239.255.76.90:7690?ttl=0

# an archived params copy (see systems/parameters/learned_archive/README.md in the worktree)
# needs its own matching --deploy and --demo-goals
OLD=learned_archive/2026-09-24/learned_tuning/round_belt_controller_params_learned_eval_v2_honor_wr0p3_wp0p03.yaml
uv run python scripts/lcs/eval_learned_mpc.py --mode learned --params systems/parameters/$OLD \
    --deploy ~/git/lcs_learning/outputs/sim_belt_v2_20260925/deploy_v2_decoded_only/deploy.npz \
    --demo-goals data/lcs/demo/v2/demo_goals.npz --lcm-url udpm://239.255.76.90:7690?ttl=0
```
`--start-states` defaults to `data/lcs/start_states/pre_place_1_osc.npz` ("nominal"); pass a
grasp-variant set dir instead to sweep `index.json`'s variants (`--variants` restricts to a
comma list). `--deploy`/`--demo-goals` default to the pair that matches the default learned
params: `deploy_v2_flat_pp2/deploy.npz` and `data/lcs/demo_flat_pp2/demo_goals.npz`. Without
`--demo-goals`, alignment is null and `sim_goal_dist` is measured against the LCS file's own
`z_goal`/`stage_goals` instead. `--params` defaults to the mode's `*_eval.yaml`
(`round_belt_controller_params_{learned,baseline}_eval.yaml`). Since 2026-09-25 the learned one
is the two-fixed-target setting on `learned_lcs/learned_lcs_v2_flat_pp2.yaml`, and earlier
tunings are under `systems/parameters/learned_archive/<date>/`. Pass a params-yaml copy to try
a different `learned_mpc:` tuning without touching the shipped file. `--action-definition
{cmd_delta,knot1_minus_measured}` (default `cmd_delta` since 2026-09-25) picks which formula
fills the recorded `actions` and the `sim_latent_pred` diagnostic — pass
`knot1_minus_measured` when evaluating a model trained on that definition. The Franka OSC starts
once per run; each episode restarts only a fresh controller process next to it, on one private
URL — always pass `--lcm-url` (`eval_learned_mpc.py`'s own default is `...88:7688`, only safe
for one run at a time). Every run's episodes are written in dataset format (`.npz` +
`index.json`) under `data/lcs/mpc_eval/<label>/`; `--dry-run` prints the episode plan
without touching LCM or the sim. Checked end to end by `scripts/checks/check_mpc_harness.py`
H0-H6 (§8 of `docs/lcm-simulation.md`).

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

### 5.10 Replay the learned MPC's plans (this repo)

Record with a debug yaml (`learned_mpc.debug_channel: LEARNED_MPC_DEBUG`) and `--record` (§5.8).
Then open the recordings in the replay viewer. Pick your own `--port`; 8081 is the default and
the live viewer uses 8080.

```bash
uv run python scripts/replay_viewer.py \
    --recordings <run dir>/recordings --run <episode> --port <PORT> \
    --learned-layers planned_belt,planned_ee,actions,target_belt
# baseline on the same starts: the standard viewer, no --learned-layers needed
```

The four layers (`planned_belt` blue tubes, `planned_ee` Franka knot + augmented dots,
`actions` length-only arrows default scale 9, `target_belt` green tubes at the fixed goal
frames) and their exactness guarantees are documented in `docs/lcm-simulation.md` §10 "Learned-
MPC layers". `--deploy`, `--decoder`, `--demo-goals` and `--demo-episode` override the paths
`meta.json` recorded; pass `--decoder` explicitly if the run's own deploy dir has no
`decoder.npz` next to it (only exports run with `--decoder-out` have one).

## 6. Private LCM groups

Every private-URL run in the day log picks its own group so concurrent runs never collide, and
so nothing lands on magna's shared group `239.255.76.67:7667`. See the day-by-day entries in
`docs/learned-mpc.md` and `handoffs/RUN-STATE.md` for which URL a given run used; the pattern is
always `udpm://239.255.76.<NN>:76<NN>?ttl=0`, passed to every magna binary's `--lcm_url` (and
`--local_lcm_url` for the worktree's controller) and to this repo's `--lcm-url`.

## 7. Limits and next steps

- **Belt input is material-indexed**: a grasp that shifts the held material point changes `z`
  even when the belt geometry matches the demo, so a fixed stage-0 goal is not fully reachable
  by construction for a shifted grasp. The best-shift RMSE metric (§5.9) exposes this
  independently of any controller failure mode.
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
