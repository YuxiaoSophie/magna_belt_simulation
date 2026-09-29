# LCS data collection (round-belt engagement)

How to generate `(observation_t, action_t, observation_t+1)` episodes of the belt being engaged
into the large pulley, for `~/git/lcs_learning`. On-disk format: `docs/lcs-dataset.md` (this doc
does not repeat it). Implemented by `src/task_common/{lcs_dataset,sim_snapshot,osc_process}.py`,
`src/round_belt_task/{waypoints,arm_kinematics,motion,offline_simulation,osc_simulation,
osc_bridge,commander,perturbation,clearance,outcome}.py`, `scripts/lcs/make_start_state.py`,
`scripts/collect_lcs_dataset.py`. Free-space primitives, approach episodes and contact branches
(2026-09-28): §4.5-4.7.

## 1. What this is

One episode covers `pre_place_1 -> place_3` plus a settle: the belt is already held by both arms
(picked at `pre_place_1`) and is carried to, and engaged with, the board's large pulley. Each
episode perturbs the `pre_place_2` / `place_3` waypoints so the outcome varies across
`engaged` / `over` / `under` / `slanted` / `outside` / `other` (`round_belt_task.outcome`).

Two backends (`--backend`):

- **`osc`** (default): the Franka is torque-driven by magna's real `franka_cartesian_osc_controller`
  over LCM, lock-step, exactly as on hardware; `state` is the measured pose, `action` is the
  commanded displacement over one C3 step, `cmd_delta` (§3, `docs/lcs-dataset.md` §3).
- **`position`** (legacy): both arms are joint-PD driven along a precomputed Cartesian trajectory,
  no magna, no LCM; `state`/`action` are the commanded targets (§3.1).

## 2. Pipeline

1. Once: build the start-state snapshot the osc backend restores
   (`scripts/lcs/make_start_state.py`): the nominal in-process pick `pre_pick_0 -> pre_place_1`
   (belt gripped physically) -> `pre_place_1.npz`, then `--backend osc` settles that snapshot 2 s
   under magna's OSC hold -> `pre_place_1_osc.npz` (§4.1).
2. Per episode (`--backend osc`):
   - restore `pre_place_1_osc.npz` (rebase the OSC clock, hold at the measured pose, settle
     `--osc-settle-s`);
   - sample an intent and perturb `pre_place_2` / `place_3` (`round_belt_task.perturbation`), then
     clamp the perturbed UR waypoints so the 2F-85 stays clear of the board (§5.1);
   - drive the Franka torque through magna's real OSC controller in lock-step (§3), commanded every
     control step by an emulation of magna's waypoint generator (`round_belt_task.commander`) from
     the MEASURED pose, with optional bounded excitation of Franka knots 1..6 and of the UR
     target (§5.2); drive the UR by per-step IK to magna's 2-knot line; `--pre-hold-s` of hold
     frames come first;
   - sample every `--sample-period` (default 0.075 s = 15 control steps, the C3 knot spacing): the
     measured state, the commanded action, a camera point cloud render, and (optionally) a
     `RunRecorder` frame; track clearances every 4 control steps;
   - classify the outcome from the last frame's belt geometry (`round_belt_task.outcome`);
   - write `episode_XXXX.npz` (`docs/lcs-dataset.md`) and append a row to `index.json`.
3. Per episode (`--backend position`, legacy): restore `pre_place_1.npz`, build a Cartesian
   trajectory to the perturbed waypoints and solve it with numpy IK, play it back with the sim's own
   joint position drive, sample the same way; `state`/`action` are the commanded targets, not a
   measurement (§3.1).

## 3. Why the OSC backend

The previous (position-backend) tuples did not match what MPC (C3) will query on hardware: `state`
was the *commanded* target, not the measured pose C3 actually conditions on, and the sample period
(0.1 s, magna's log rate) was not the C3 `dt` (0.075 s). The osc backend fixes both: it runs magna's
own `franka_cartesian_osc_controller` binary (unmodified) as a child process and drives it exactly
as the hardware does, so `state` is the measured pose and one tuple is one C3 step.

**What is emulated:** magna's waypoint generator (`assembly_controller.cc`, not run here) — 7
Franka knots spaced by `dt=0.075 s`, each knot walking from the measured pose towards the target at
`0.08 m/s` / `0.5 rad/s`, latched once within `5.5 mm` / `0.15 rad`, then held (2-knot hold at the
latched pose) for the waypoint's dwell before advancing; and the UR's 2-knot `tool0` line (current
pose -> target at `0.08 m/s` / `0.5 rad/s`, `>= 0.5 s`, regenerated per magna's rule). **What is NOT
run:** magna's assembly controller itself (no bazel recompile per waypoint set, no shared LCM
group) — the Python collector (`round_belt_task.commander`) reproduces its rules directly.

**Lock-step protocol** (`round_belt_task.osc_bridge.OscBridge`, `task_common.osc_process`):

- The OSC binary (`bazel-bin/systems/controllers/franka_cartesian_osc_controller`) is launched from
  the magna root (its params resolve relative to CWD) on a PRIVATE `--lcm_url`, `--input_mode=1`.
  It reads `FRANKA_STATE` + `TARGET_CARTESIAN_POSE_TRAJECTORY`, publishes one `FRANKA_INPUT` per
  state. Side effect: it writes `<magna root>/../diagrams/franka_cartesian_osc_controller{,.svg}`.
  No other magna process is needed (no hand driver, no translator, no assembly controller).
- Every control step publishes the tick's trajectory, then `FRANKA_STATE` (utime rewritten to a
  monotonic OSC clock, bumped at every restore so magna's `LcmDrivenLoop` never treats a restore as
  a backwards time jump / resets); the next step blocks until the `FRANKA_INPUT` whose echoed utime
  matches within 2 µs (magna rounds its reply time to µs), with a 5 s timeout and one republish.
- Warm-up: once per run, republish the current state every 100 ms until the OSC answers.
- Restore = snapshot restore -> clock rebase -> 2-knot hold at the measured pose ->
  `--osc-settle-s` (default 0.5 s). The OSC's own internal state (trajectory buffers, filters) is
  NOT restored by a snapshot; the hold + settle bounds the resulting drift, not eliminates it (§7).
- Shutdown: SIGINT to the OSC's own pid, 3 s grace, then SIGKILL.

**One OSC build per process for ad-hoc scripts.** A probe script crashed natively on its fourth
`RoundBeltOscSimulation` build (`malloc(): unaligned tcache chunk detected`; root cause unknown).
The checks build up to 7 in sequence without issue, but prefer one build per process for ad-hoc
scripts.

### 3.1 position backend (legacy)

magna's assembly controller needs a bazel recompile per waypoint set and hard-codes the shared LCM
debug group, so it cannot batch headlessly even before the osc backend existed; `--backend position`
keeps the original in-process path for cases that don't need OSC fidelity (fast iteration, no
magna checkout needed). It interpolates a Cartesian trajectory to each waypoint and solves it with
numpy IK, then feeds the result to the sim's existing joint position drive (`ke`/`kd` = 700/110,
same as the LCM sim). Headless, batchable, deterministic (`check_inproc_motion.py` M5).

**Fidelity caveat:** this tracks a Cartesian target with a joint position drive, not magna's
Cartesian-space OSC controller, and `state`/`action` are the *commanded* targets, not measurements.
Tracking error stays within a few mm / degrees away from contact (`check_inproc_motion.py` M4), but
this is not what MPC sees; prefer `--backend osc` for training data.

**The belt is picked physically** (both grippers close on it at the `pick` waypoint) and the
Drake belt-teleport trigger is disabled, so a start-state snapshot captures a physically
consistent grasp, not a scripted one. This applies to both backends' start states.

## 4. How to run

### 4.1 Build the start state (once, or after a scene change)

```bash
uv run python scripts/lcs/make_start_state.py
uv run python scripts/lcs/make_start_state.py --backend osc
```

| flag | default | meaning |
|---|---|---|
| `--backend` | `position` | `position` builds `pre_place_1.npz`; `osc` settles it under magna's OSC into `pre_place_1_osc.npz` |
| `--lcm-url` | `udpm://239.255.76.83:7683?ttl=0` | osc backend: private LCM URL (never magna's shared group) |
| `--start` | `pre_place_1.npz` | osc backend: the position-backend snapshot to settle |
| `--out` | `pre_place_1.npz` / `pre_place_1_osc.npz` | snapshot path |
| `--record [DIR]` | off (const `data/lcs/recordings`) | also record the run (position backend) |
| `--params` | magna's `round_belt_controller_params_sim.yaml` | waypoints source (read-only) |
| `--arm-ke` / `--arm-kd` | `700.0` / `110.0` | arm position-drive gains |
| `--no-velocity-lead` | off | position backend: plain position targets, no `kd/ke * qdot` feed-forward |
| `--hold` | `1.0` | position backend: hold at the end [s] |

Fails with `GRASP FAILED: ...` (exit 1) if either gripper does not end up holding the belt
(position backend), or `OSC LOG ERROR: ...` / `GRASP FAILED: ...` if the settle drops the grasp or
the OSC log shows a reset/exception (osc backend). Run `--backend osc` once **after** the position
one exists: it loads `--start` (default `pre_place_1.npz`) and writes `pre_place_1_osc.npz`.

`scripts/lcs/make_grasp_variants.py` builds *grasp-varied* `pre_place_1` start states from
`pre_place_1_osc.npz` (each gripper holding the belt elsewhere, same nominal target poses),
for the learned-MPC evaluation harness — see `docs/learned-mpc-reference.md` §5.6.

### 4.2 Collect episodes

```bash
uv run python scripts/collect_lcs_dataset.py --episodes 40 --seed 0
uv run python scripts/collect_lcs_dataset.py --backend position --episodes 40 --seed 0
```

| flag | default | meaning |
|---|---|---|
| `--out` | `data/lcs/<YYYYmmdd-HHMMSS>-<label>` | output dir |
| `--label` | `lcs` | used in the default `--out` |
| `--episodes` | `20` | episode count |
| `--seed` | `0` | RNG seed (reproducible: each episode's draw is keyed on `[seed, episode]`) |
| `--intents` | `engaged,over,under,slanted` | sampled per episode (see `--weights`); osc also accepts the slant variant `slanted_franka_high_steep` (§5.3) |
| `--weights` | uniform | one weight per `--intents` entry |
| `--backend` | `osc` | `osc`: Franka on magna's OSC (lock-step); `position`: in-process PD only |
| `--lcm-url` | `udpm://239.255.76.83:7683?ttl=0` | osc backend: private LCM URL of the OSC (never magna's shared group); one URL per concurrent run |
| `--osc-timeout-s` | `5.0` | osc backend: seconds to wait for a `FRANKA_INPUT` reply |
| `--osc-settle-s` | `0.5` | osc backend: hold under the OSC after each restore |
| `--osc-log` | `<out>/osc.log` | osc backend: OSC process log path |
| `--start-state` | `pre_place_1_osc.npz` (osc) / `pre_place_1.npz` (position) | snapshot from 4.1 |
| `--start-states` / `--variants` | off | osc: a snapshot or a grasp-variant set dir (`make_grasp_variants.py`); episode `i` restores `variants[i % len]` (held states only), recorded as `start_variant` in `index.json` and `start_state` in `sim_meta` |
| `--fresh-pick` | off | position backend only: re-run the nominal pick per episode instead of restoring (slow) |
| `--record` | off | a `RunRecorder` run per episode under `<out>/recordings/` |
| `--settle-s` | `1.0` | dwell at the end of the move before the last sample |
| `--min-clearance` | `4.0` | mm of 2F-85 -> board clearance to clamp the UR waypoints to (§5.1); `0` = measure only |
| `--sample-period` | `0.075` | seconds; a multiple of the 5 ms control step; the C3 knot spacing (`0.1` = magna's log rate) |
| `--excite-mode` | `ou` | osc backend: `ou` = smooth Ornstein-Uhlenbeck offset; `white` = a fresh draw per sample (data before 2026-09-23 evening; §5.2) |
| `--excite-pos-mm` | `1.5` (ou) / `4.0` (white) | osc backend: Franka knot 1..6 offset; ou: per-axis stationary std; white: ball radius (0 with `--excite-rot-deg 0` = off) |
| `--excite-rot-deg` | `1.0` (ou) / `2.0` (white) | osc backend: ou: per-axis rotation-vector std; white: max angle |
| `--excite-tau-s` | `0.4` | osc backend, ou: correlation time |
| `--excite-cap-factor` | `2.0` | osc backend: knot step capped to `factor * speed * dt` |
| `--excite-down-mm` | `2.0` | osc backend: largest downward excitation offset |
| `--excite-ur-pos-mm` | `2.0` | osc backend, `ou` only: UR target offset per-axis std (z clipped >= 0, never below the nominal target); `0` with `--excite-ur-rot-deg 0` = off (§5.2) |
| `--excite-ur-rot-deg` | `1.0` | osc backend, `ou` only: UR target rotation-vector per-axis std |
| `--action-definition` | `cmd_delta` | osc backend: recorded `actions`, `cmd_delta` or `knot1_minus_measured` (`docs/lcs-dataset.md` §3); both, `sim_cmd_*` and `sim_realised_delta` are always stored |
| `--pre-hold-s` | `1.5` | osc backend: after the settle, `round(S / 0.075)` frames under the hold before the first move (`sim_episode_step < 0`, phase `prehold`, `u = 0` exactly, asserted) |
| `--max-episode-s` | `20.0` | osc backend: skip an episode whose targets are not all reached by then |
| `--scenario` | none | osc backend: `pure_translation` = one Franka target 30 mm -x / 10 mm +z, UR still, excitation forced off; `nominal` = the unperturbed waypoints, excitation off — used to record the learned-MPC demonstration episode, `docs/learned-mpc-reference.md` §5.3 |
| `--arm-ke` / `--arm-kd` | `700.0` / `110.0` | arm position-drive gains (UR, and the Franka under the position backend) |
| `--params` | magna's `round_belt_controller_params_sim.yaml` | waypoints source (read-only) |
| `--no-pcd` | off | skip camera renders (files then fail dataset validation; speed tests only) |
| `--dry-run` | off | print the sampled perturbation table and exit, nothing written |
| `--thresholds` | none | `KEY=VALUE` overrides of `OutcomeThresholds`, e.g. `engaged_arc_deg=75` |

Prints a confusion table (`intent` rows x `outcome` columns), a per-intent clearance table
(min / median clearance, largest guard lift, contact count) and a summary line (episodes
ok/skipped, episodes/min, mean per-episode timing) when done. An episode is `skipped`, not
written, if the perturbed waypoints are unreachable (IK failure, osc backend timeout, or the
episode does not reach its targets within `--max-episode-s`).

### 4.3 Inspect episodes

```bash
uv run python scripts/replay_viewer.py --recordings data/lcs/<run>/recordings
```

Only works for episodes collected with `--record`; recording dirs are named
`episode_XXXX-<intent>`.

### 4.4 Train with lcs_learning

```bash
cd ~/git/lcs_learning && uv run python scripts/train_joint_pointnet_lcs.py \
    --data-glob "/home/hienbui/git/magna_belt_simulation-main/data/lcs/<run>/episode_*.npz" \
    --point-cloud-source camera_plus_belt --num-points 1800 --belt-num-points 150
```

`--data-glob` may be repeated to combine run subdirectories, and overrides a config's own
hardware-log glob, e.g. a 2-epoch pilot over one 10-episode osc run's excitation on/off subsets:

```bash
cd ~/git/lcs_learning && uv run python scripts/train_joint_pointnet_lcs.py \
    --config configs/fair_comparison_latent16.yaml \
    --data-glob "<run>/excite/episode_*.npz" --data-glob "<run>/noexcite/episode_*.npz" \
    --epochs 2 --ae-warmup-epochs 0 --disable-saving --no-wandb --disable-wandb-viz
```

### 4.5 Free-space motion primitives (`scripts/lcs/collect_motion_primitives.py`)

Scripted, smooth EE motions of both arms holding the belt, away from the pulley. Each episode
restores `pre_place_1_osc.npz` (or a grasp-variant state) under magna's OSC, holds
`--pre-hold-s` (`u = 0`), plays one motion and holds `--post-hold-s`. The Franka gets 7 knots
sampled on the scripted pose; the UR gets per-step IK plus a 2-knot line `pose(t) -> pose(t +
dt)`, so `cmd_delta` is `pose(t + dt) - pose(t)` exactly. Files use the collector's format plus
`sim_motion_*` / `sim_rod_stretch_pct`. Originally the OOD one-step test sets
(`data/lcs/motion_primitives/`, v1 and v2-big); since 2026-09-28 also training data
(`data/lcs/free_space/`, `docs/lcs-dataset.md` §10). Flags off, the plans are bit-identical to
the v1 / v2-big ones.

**Live guards** (a failing episode is retried, see `--retry-policy`): grasp held; belt inside
the crop box x/y and above its z-bottom; **crop rule:** the belt top may not rise more than
`--crop-z-tol-mm` above its own start top. The crop top stays at 0.11 m (2026-09-28 decision
D1), and the start belt already sits ~3 mm above it, so vertical motions only go down. Also: a
belt-near-pulley keep-out, the 2F-85 board clearance, and a rod stretch gain ≤
`--stretch-cap-pct`.

**Families** (`--primitives`, default the 10 v1 ones): `both_up_down`, `both_fwd_back`,
`both_sideways`, `opposite_fwd_back`, `opposite_sideways`, `opposite_up_down`, `franka_only`,
`ur_only`, `wrist_roll`, `random_mix` (v1); `good_mix`, `good_mix_tilt` (v2-big); and the
deformation families of 2026-09-28: `twist`, `bend`, `bend_lift`, `stretch_cycle`,
`franka_sweep`, `ur_sweep`, `random_mix_holds` (random mix with `u = 0` plateaus),
`wrist_tilt`, `hold_only` (no motion). twist / bend / bend_lift / wrist_tilt first drop both
arms (−z, 0.5× amp) and deform in τ ∈ [0.2, 0.8]: without that drop the gripper rotations
lift the belt past the crop rule.

| flag | default | meaning |
|---|---|---|
| `--lcm-url` | required | private LCM URL of the OSC |
| `--out` | `data/lcs/motion_primitives/<YYYYmmdd-HHMMSS>` | output dir |
| `--append` | off | add episodes to an existing `--out` (extends its `index.json`) |
| `--primitives` | the 10 v1 families | comma list (above) |
| `--amp-mm` / `--amp-deg` | `20` / `5` | max translation / rotation amplitude |
| `--bound-frac` | `0.8` | planned `\|cmd_delta\|` ≤ this fraction of the `--deploy` export's `u_lb/u_ub` (or of `--step-cap`) |
| `--deploy` | v2 `deploy_v2_decoded_only/deploy.npz` | export whose `u_lb/u_ub` bound the plan without `--step-cap` |
| `--step-cap F_MM,F_MRAD,U_MM,U_MRAD` | off | symmetric per-step caps per arm replacing the export bounds; retries then scale the capped plan (`row.retry_mul`) |
| `--motion-s` | `8.0` | motion duration; `<= 0`: per primitive, the shortest in [`--motion-s-min`, `--motion-s-max`] (`15` / `25`) that keeps `--bound-frac` at full amplitude |
| `--motion-s-range LO,HI` | off | per-episode duration ~ U[LO, HI] (not with `--motion-s <= 0`) |
| `--motion-s-mul NAME=X[,NAME=X]` | off | per-family multiplier on the `--motion-s-range` draw (row field `motion_s_mul`); errors without `--motion-s-range` |
| `--amp-range LO,HI` | off | per-episode amplitude multiplier ~ U[LO, HI] (row `amp_mul`) |
| `--repeats` | `1` | episodes per primitive; `> 1` adds the repeat to the RNG key |
| `--start-state` | `pre_place_1_osc.npz` | snapshot to restore |
| `--start-states DIR` / `--variants a,b` | off | grasp-variant set; episodes round-robin over `--variants` (`start_state` in the index = set meta + sha256) |
| `--retry-policy` | `halve` | `halve`: halve the amplitude on any failure (v1); `lobe`: on a stretch failure first halve the stretching lobe, else amplitude ×0.7 |
| `--retries` | `2` | retries per episode |
| `--pre-hold-s` / `--post-hold-s` | `1.5` / `0.5` | holds before / after the motion |
| `--stretch-cap-pct` | `2.0` | max rod stretch gain over the episode |
| `--crop-z-tol-mm` | `2.0` | crop rule above |
| `--no-pulley-keepout` | off | drop the live belt-near-pulley check |
| `--min-clearance` | `4.0` | 2F-85 board clearance (§5.1) |
| `--osc-settle-s` / `--osc-timeout-s` | `0.5` / `5.0` | as in §4.2 |
| `--seed` | `0` | RNG seed; amplitude / duration draws use their own stream `[seed, idx, rep, 1]` |
| `--record` / `--no-pcd` | off | `RunRecorder` run per episode / skip camera renders |

The per-step cap used on 2026-09-28 is `3,25,3,25` (3 mm / 25 mrad per 75 ms step, both arms):
the cap probe met the realised-vs-`u` slope gate at 3 mm (slopes 0.956-1.067), so smaller caps
were not run. Runs actually used (`data/lcs/free_space/`; train on `...133:7733`, held-out on
`...134:7734`):

```bash
C="--step-cap 3,25,3,25 --bound-frac 1.0 --amp-mm 40 --amp-deg 20 --retry-policy lobe
   --retries 4 --motion-s-range 6,14 --amp-range 0.4,1.0"
F14=twist,bend,bend_lift,stretch_cycle,franka_sweep,ur_sweep,random_mix,random_mix_holds
F14=$F14,wrist_tilt,wrist_roll,opposite_sideways,both_up_down,good_mix,hold_only
F8=twist,bend,stretch_cycle,franka_sweep,ur_sweep,random_mix_holds,wrist_tilt,hold_only
S2="--start-states data/lcs/start_states/grasp_variants/set2 --variants gv_01,gv_04,gv_07,gv_10"
S3="--start-states data/lcs/start_states/grasp_variants/set3 --variants gv_02,gv_05"
MP="uv run --frozen python scripts/lcs/collect_motion_primitives.py $C"
T='udpm://239.255.76.133:7733?ttl=0'; H='udpm://239.255.76.134:7734?ttl=0'
$MP --lcm-url $T --out data/lcs/free_space/v1/nominal_families --primitives $F14 --repeats 3 \
    --motion-s-mul twist=2 --seed 301
$MP --lcm-url $T --out data/lcs/free_space/v1/nominal_random \
    --primitives random_mix_holds,random_mix --repeats 6 --seed 302
$MP --lcm-url $T --out data/lcs/free_space/v1/set2_families $S2 --primitives $F8 --repeats 4 \
    --motion-s-mul twist=2 --seed 303
$MP --lcm-url $T --out data/lcs/free_space/v1/set2_random $S2 --primitives random_mix \
    --repeats 4 --seed 304
$MP --lcm-url $H --out data/lcs/free_space/v1_heldout/nominal --primitives $F14 --repeats 1 \
    --amp-range 0.6,1.0 --motion-s-mul twist=2 --seed 401
$MP --lcm-url $H --out data/lcs/free_space/v1_heldout/set3 $S3 --primitives $F8 --repeats 1 \
    --motion-s-mul twist=2 --seed 402
```

With `--start-states`, variant = repeat index (set2: each of gv_01/04/07/10 sees each of the 8
families once). **QC / reports:** `scripts/lcs/deform_report.py RUN [RUN ...] [--write-qc]
[--json OUT]` (per-episode table, per-family stats, holds, causality, `<run>/qc.json`, pooled
coverage; non-rigid deformation = belt RMSE after the best rigid Kabsch fit of the motion-start
belt); `scripts/lcs/start_check.py RUN [--tol-mm 2]` (every episode's frame-0 belt is closest
to its own variant); `check_lcs_tuples.py --causality-dir RUN`.

Measured 2026-09-28: train 89 ok / 15,689 tuples, held-out 20 ok / 3,692 tuples (3 episodes
failed on the crop rule). Causality gate [0.8, 1.1] met in every table (pooled train
0.949-1.046). Stretch gain max 1.99 %, grasp held, no board contact. Holds `u == 0` exactly,
mean belt drift ≤ 0.93 mm. `free_space/v1/nominal_random/episode_0010` has a Franka tracking
excursion (6.2 mm RMS, 29 mm peak) and is excluded from training.

### 4.6 Approach episodes (`collect_lcs_dataset.py`, opt-in)

Insertion episodes approached from other directions, with a continuous `place_3` and a
post-place contact tail. All flags are opt-in; at their defaults they are kept out of
`index.args` and the collector is unchanged (`check_lcs_collector.py` C0-C8).

- **Approach transform** (`--approach`): both arms' `pre_place_1/2` and `place_3` are moved
  rigidly: yaw about the large pulley's vertical axis, offset along the rotated horizontal
  normal, `+elev` in z (not on `place_3`), tilt about the rotated tangent.
- **`place_3` sampling** (`--place3-mode`): `intents` = today's `--intents` draw; `continuous`
  = with probability `--engaged-share` the engaged band (the OSC `engaged` intent box plus ±2°
  yaw per arm), else the wide box drawn independently per arm (depth > 0 above the nominal,
  < 0 pressing past it; normal, tangent, roll, yaw); `engaged_band` = the band only; `fixed` =
  `--place3-values`. The Franka's `place_3` is floored at plate + 12 mm
  (`clamp.franka_lift_mm`); the board guard (§5.1) still applies.
- **Tail** (`--tail`, `scripts/lcs/place_tail.py`): after the `place_3` settle, one scripted
  motion of both arms for U[MIN, MAX] s + a 0.5 s hold, from the commanded hold poses (no
  jump). Families: `groove_slide`, `press_deeper`, `lift_repress`, `partial_pullout`,
  `tension_release`, `random_contact`. Engaged-band episodes get gentle tails (groove_slide /
  tension_release / lift_repress at 0.3× amplitude). Live guards: grasp (a loss discards the
  episode), crop x/y + top, 2F-85 clearance, stretch gain, Franka tip floor; any other guard
  ends the tail early and keeps the frames.
- **Labels and extras:** the outcome at `place_3` (`sim_outcome_place3`, row `outcome_place3`)
  and at the end; per-frame `sim_n_neighbour`, `sim_in_contact`, `sim_rod_stretch_pct`,
  `sim_ur_guard_scale`, `sim_motion_offset_*`; rows `place3_frame`, `approach`,
  `place3_sample`, `tail`, `tag`. With `--hold-ur-gripper` the UR holds through the tail, so
  end labels are not comparable with UR-released episodes; `outcome_place3` is.

| flag | default | meaning |
|---|---|---|
| `--approach YAW,ELEV,OFFSET,TILT\|start` | off | approach transform (deg, mm, mm, deg); `start` = each start state's own `index.json` entry (`make_approach_starts.py`) |
| `--approach-range YLO:YHI,ELO:EHI,OLO:OHI,TLO:THI` | off | per-episode approach ~ U[range] (RNG `[seed, i, 13]`) |
| `--place3-mode {intents,continuous,engaged_band,fixed}` | `intents` | above |
| `--engaged-share` | `0.40` | `continuous`: share of episodes drawn from the engaged band |
| `--place3-box DLO:DHI,NORMAL,TANGENT,ROLL,YAW` | depth −6..20 mm, ±10 mm, ±10 mm, ±10°, ±15° | wide box (symmetric except depth) |
| `--place3-box-ur` | `--place3-box` | the UR's wide box |
| `--place3-values F:U[;F:U]` | off | `fixed`: per arm `depth,normal,tangent,roll,yaw`; episode i uses entry `i % n` |
| `--tail MIN_S,MAX_S` | off | post-`place_3` contact tail |
| `--tail-families` | all (gentle set in the band) | comma list of tail families |
| `--tail-gentle {auto,on,off}` | `auto` | `auto`: engaged-band episodes get gentle tails |
| `--tail-cap F_MM,F_MRAD,U_MM,U_MRAD` | `3,25,3,25` | per-step `cmd_delta` caps of the tail |
| `--tail-stretch-cap-pct` | `2.0` | tail stretch-gain guard |
| `--row-tag` | none | recorded as `row["tag"]` (e.g. `top_up`) |

Helper scripts:

- `scripts/lcs/approach_probe.py --lcm-url URL --out screen.json [--path-min-mm 2]`: offline
  screen (no physics) of the approach grid and the per-arm `place_3` box corners: UR IK reach,
  Franka tip / crop proxy, 2F-85 pre-guard path clearance, `place_3` clamp lift.
- `scripts/lcs/make_approach_starts.py --lcm-url URL --out DIR --settings
  'id=yaw,elev,offset,tilt;...'`: start states `pre_place_1` moved rigidly through the
  approach transform (min-jerk `--move-s`, hold, settle, grasp check, re-check on restore);
  grasp-variant format + `approach` in `index.json`.
- `scripts/lcs/approach_report.py RUN [RUN ...] [--write-qc] [--pool] [--probe A|B|C]
  [--json OUT]`: labels at `place_3` / end per start state, transitions, engaged fractions,
  tails, clearances, grasp, latch, causality on approach and tail rows, `<run>/qc.json`.
- `scripts/lcs/place_tail.py`: the tail module (imported lazily by the collector).

Usable box after the 2026-09-28 probes: yaw [-30, 30]°, elev [0, 30] mm, offset [-10, 10] mm,
tilt [-4, 0]° (tilt +4 / +8 never wrap); UR tangent [0, 10] mm (any UR tangent < 0 stalls the
held UR short of `pre_place_2`). Runs actually used (`...137:7737` for probes and start states,
`...138:7738` for collection; train states a01-a12, held-out h01 (+25, 10, 0, -2), h02 (-25,
10, 0, -2), h03 (0, 22, 5, -2), h04 (-10, 15, -5, -4)):

```bash
P='udpm://239.255.76.137:7737?ttl=0'; U='udpm://239.255.76.138:7738?ttl=0'
uv run --frozen python scripts/lcs/approach_probe.py --lcm-url $P \
    --out data/lcs/approach/probe/screen.json
S='a01=-30,30,10,-4;a02=-30,0,10,-4;a03=30,0,-10,0;a04=-30,0,-10,-4;a05=-30,30,-10,0;'
S+='a06=30,30,-10,-4;a07=5.4,4.7,-7.3,-0.9;a08=29.1,7.6,3.3,-2.3;a09=7.9,16.7,-8.3,-0.4;'
S+='a10=6.6,3.5,-3.9,-3.5;a11=1.8,10.1,7.3,-1.4;a12=11.6,5.2,-7.3,-3.9;'
S+='h01=25,10,0,-2;h02=-25,10,0,-2;h03=0,22,5,-2;h04=-10,15,-5,-4'
uv run --frozen python scripts/lcs/make_approach_starts.py --lcm-url $P \
    --out data/lcs/start_states/approach --settings "$S"
A="--start-states data/lcs/start_states/approach --hold-ur-gripper --approach start --tail 4,6"
COL="uv run --frozen python scripts/collect_lcs_dataset.py --lcm-url $U $A"
$COL --out data/lcs/approach/v1/train --episodes 120 --seed 601 \
    --variants a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12 \
    --place3-mode continuous --place3-box-ur=-6:20,10,0:10,10,15
$COL --out data/lcs/approach/v1_heldout --episodes 24 --seed 701 --variants h01,h02,h03,h04 \
    --place3-mode continuous --place3-box-ur=-6:20,10,0:10,10,15
# engaged-band top-ups (train_topup_1..4, v1_heldout_topup_1..2), states rotated per batch
$COL --out data/lcs/approach/v1/train_topup_1 --episodes 10 --seed 611 \
    --variants a01,a02,a03,a04,a05,a06,a07,a08,a09,a10 --place3-mode engaged_band --row-tag top_up
uv run --frozen python scripts/lcs/approach_report.py data/lcs/approach/v1/train \
    data/lcs/approach/v1/train_topup_* --write-qc --pool
```

Measured 2026-09-28: train 120/120 + top-ups 38/40, held-out 23/24 + top-ups 7/8 (5 skips,
all `timeout`), ≈ 161 frames per episode, 3.91 ep/min. Engaged at `place_3` / end: train
20.3 / 22.8 %, held-out 20.0 / 16.7 %, **below the 30 / 25 % targets (accepted)**. Tails ≥ 4 s
in 98.1 % (held-out 100 %), contact-frame fraction 1.0 in every tail, grasp held on every frame,
0 board contacts, tail stretch gain max 2.08 %. Causality: approach rows 0.68-0.96 (gate
[0.6, 1.2]), tail rows 0.93-1.01 (gate [0.8, 1.1]).

### 4.7 Contact branching (`scripts/lcs/collect_contact_branches.py`)

Contact-rich rollouts branched from sim snapshots taken during insertions. Four phases:

- **`snap`**: runs insertion source episodes through `collect_lcs_dataset.collect`
  (`run_osc_episode` wrapped in-process with the opt-in `on_frame` callback; uniform intents,
  no excitation, no pre-hold, `--no-pcd`, UR held) and snapshots `first_contact` (first frame
  with `n_neighbour >= 3` and `h_min <= 10 mm`, or wrap > 0), `partial` (the first later frame
  with 15 <= wrap < 60°) and `final_<label>`. Each snapshot is validated right away (restore +
  0.5 s settle + grasp). The source episodes (`--source-out`) have no point clouds and are
  **not** training data.
- **`select`** (offline): a balanced working set of the validated snapshots
  (`working_set.json`); `--all` = every snapshot, 1 rollout each.
- **`branch`**: per (snapshot, family): restore + 0.5 s settle, 1.0 s pre-hold (`u = 0`), a
  scripted 4-8 s rollout of both arms (per-step cap 3,25,3,25), 0.5 s hold, in the
  collector's training format. Families: `rim_press`, `top_cross`, `pullout_reseat`,
  `groove_slide`, `recover` (lift, re-descend, hold), `random_contact`. Guard failures get one
  ×0.7 retry. Per-frame extras: `sim_wrap_deg`, `sim_h_median_mm`, `sim_slant_deg_t`,
  `sim_n_neighbour`, `sim_in_contact`, `sim_outcome`; plus `sim_outcome_start`, `sim_family`,
  `sim_branch`.
- **`report`**: `qc.json` per branch run, plus `--rebalance OUT` (contact-frame share of the
  v2 insertion, approach and branch sets).

| phase | flags |
|---|---|
| `snap` | `--lcm-url` (req.), `--src NAME` (snapshot id prefix), `--episodes`, `--seed`, `--start-states`, `--variants`, `--snap-dir` (default `data/lcs/contact/snapshots`), `--source-out` (req.) |
| `select` | `--snap-dir`, `--out` (req.), `--all` |
| `branch` | `--lcm-url` (req.), `--snap-dir`, `--working-set` (req.), `--out` (req.), `--seed`, `--append`, `--families`, `--limit`, `--no-pcd`, `--record`, `--min-clearance` |
| `report` | `RUN [RUN ...]`, `--rebalance OUT` |

Runs actually used (`...139:7739` snap, `...140:7740` branch):

```bash
CB="uv run --frozen python scripts/lcs/collect_contact_branches.py"
S='udpm://239.255.76.139:7739?ttl=0'; B='udpm://239.255.76.140:7740?ttl=0'
GV=data/lcs/start_states/grasp_variants; C=data/lcs/contact
$CB snap --lcm-url $S --src nominal --episodes 20 --seed 501 --source-out $C/source/nominal
$CB snap --lcm-url $S --src set2 --episodes 20 --seed 502 --start-states $GV/set2 \
    --variants gv_01,gv_04,gv_07,gv_10 --source-out $C/source/set2
$CB snap --lcm-url $S --src set3 --episodes 8 --seed 503 --start-states $GV/set3 \
    --variants gv_02,gv_05 --snap-dir $C/snapshots_heldout --source-out $C/source/set3_heldout
$CB select --out $C/snapshots/working_set.json
$CB select --snap-dir $C/snapshots_heldout --all --out $C/snapshots_heldout/working_set.json
$CB branch --lcm-url $B --working-set $C/snapshots/working_set.json --out $C/v1/train --seed 601
$CB branch --lcm-url $B --snap-dir $C/snapshots_heldout \
    --working-set $C/snapshots_heldout/working_set.json --out $C/v1/heldout --seed 602
$CB report $C/v1/train $C/v1/heldout --rebalance $C/v1/rebalance.json
```

Measured 2026-09-28: 48 source episodes (engaged 14, over 11, under 12, slanted 11) gave 98
train / 18 held-out validated snapshots. Working set 40 (first_contact 10, partial 10, 5 per
final label). Rollouts: train 80/80 ok, held-out 17/18 ok. Strict contact-frame fraction
91.2 / 89.0 %, wrap max 143 / 148°. Recovery (engaged end from a failure snapshot): `recover`
1/15 (held-out 0/6). Held-out causality u_x 0.751 is below the 0.8 gate; it is attributed to
the held UR lagging under belt load.

## 5. Perturbation classes and outcome thresholds

Each episode samples an intent, then a UR-arm position offset + tilt about the belt tangent (the
horizontal Franka-tip -> UR-tip direction at the nominal `pre_place_2`); the Franka gets shared
jitter. `--backend osc` uses `round_belt_task.perturbation.OSC_RANGES` (mm / deg, UR `dxy` and
Franka jitter shared across intents unless noted):

| intent | UR `dz` mm | UR `tilt` deg | UR `dxy` mm | Franka `dxyz` mm | Franka `tilt` deg |
|---|---|---|---|---|---|
| `engaged` | 1.5 .. 4.0 | 2.5 .. 5.0 | -3 .. 3 | -2 .. 2 | -3 .. 3 |
| `over` | 11 .. 20 | -3 .. 1 | -3 .. 3 | -2 .. 2 | -3 .. 3 |
| `under` | 2 .. 4 | 10 .. 15 | -2 .. 2 | -2 .. 2 | -3 .. 3 |
| `slanted` | -2.0 .. -0.5 | -5.5 .. -4.5 (sign fixed) | -3 .. 3 | -2 .. 2 | -3 .. 3 |

Why these differ from the position backend's ranges (below): the Franka latches ~5.3 mm short of
`place_3` under the OSC's `5.5 mm` reach tolerance (magna's hardware tolerance, §7). A ~0 or
negative UR tilt then slants the belt where it used to engage, and a positive tilt now engages
where it used to slant. `slanted` needed a larger, fixed-sign tilt band: an earlier osc-backend
retune (tilt -4.5 .. -2.0 deg at dz 2..4 mm) gave a 12.1 deg mean belt slant
(`atan((h_max - h_min) / seat diameter)` over the seat neighbourhood); the shipped range gives
13.5 deg mean (range 11.4..15.7).

Outcome thresholds (`round_belt_task.outcome.DEFAULT_THRESHOLDS`, unchanged from the
position-backend tuning run): `neighbour_radial_mm=12`, `neighbour_inner_mm=10`,
`in_groove_axial_mm=4`, `in_groove_radial_mm=5`, `engaged_arc_deg=60`, `partial_arc_deg=15`,
`over_under_h_mm=5`, `slant_spread_mm=8`, `outside_min_bodies=2`.

Measured 2026-09-23, `--episodes 40 --seed 0` (0 skipped in either run):

```text
excitation ON, white (data/lcs/20260923-182315-tune40-osc)
intent     engaged  over  under  slanted  outside  other
engaged          8     0      0        1        0      0
over             0    11      0        0        0      0
under            0     0     10        0        0      0
slanted          0     0      0       10        0      0
total            8    11     10       11        0      0

excitation OFF (data/lcs/20260923-182727-tune40-osc-noexcite)
intent     engaged  over  under  slanted  outside  other
engaged          9     0      0        0        0      0
over             0    11      0        0        0      0
under            0     0     10        0        0      0
slanted          0     1      0        9        0      0
total            9    12     10        9        0      0
```

Every class has >= 3/40 episodes and every intent >= 89 % on its own class, in both runs.

`under` is **tilt-driven, not dz-driven**: a `10..15 deg` tilt about the belt tangent tips the loop
below the pulley groove while the gripper stays clear (the position-backend rationale, unchanged);
a negative dz alone would drag the 2F-85 fingers through the board plate (§5.1).

**`--backend position`** uses `round_belt_task.perturbation.DEFAULT_RANGES` instead (mm / deg):

| intent | UR `dz` mm | UR `tilt` deg | UR `dxy` mm | Franka `dxyz` mm | Franka `tilt` deg |
|---|---|---|---|---|---|
| `engaged` | 1.5 .. 5 | -2 .. 1.5 | -3 .. 3 | -2 .. 2 | -3 .. 3 |
| `over` | 11 .. 20 | -3 .. 1 | -3 .. 3 | -2 .. 2 | -3 .. 3 |
| `under` | 2 .. 4 | 10 .. 18 | -3 .. 3 | -2 .. 2 | -3 .. 3 |
| `slanted` | 2 .. 4 | \|3.5 .. 5.5\|, random sign | -3 .. 3 | -2 .. 2 | -3 .. 3 |

Measured 2026-09-22, `--episodes 20 --seed 11 --record` (`data/lcs/20260922-151152-clearance`,
0 skipped, 0 board contacts, 5 episodes per intent):

```text
intent     engaged  over  under  slanted  outside  other      min clearance mm
engaged          4     0      0        1        0      0                  2.25
over             0     5      0        0        0      0                  3.30
under            0     0      5        0        0      0                  2.20
slanted          0     0      0        5        0      0                  3.62
total            4     5      5        6        0      0                  2.20
```

### 5.1 Board clearance guard

The UR's Robotiq 2F-85 grips the belt ~75 mm from the large pulley's axis, i.e. right over the
board, and its ALOHA fingertips reach 13.5 mm past the `tracking_frame` origin. The board's
*collision* geometry is the plate box (top at world z ~ 0.0099, `board/board/collision0`) plus the
two grooved pulleys (large: outer r 51 mm, groove bottom r 47 mm, half-height 5 mm, top at world
z ~ 0.0394); every other raised board feature (the pulley axle bolt, the small-pulley mounting
plate) is visual-only and stays > 40 mm from the gripper's path. The guard models only the 2F-85
(UR); it does not cover the Franka.

`round_belt_task.clearance` measures the distance between the 2F-85's mesh colliders and that
geometry two ways:

- **Preventive** (`clamp_waypoints`, before the episode runs): the gripper is frozen in the
  `tracking_frame` — once with the jaws closed on the belt and once per Robotiq byte the segment
  commands, since the jaws open at `place_3` and that swings the fingertips ~6 mm outward, right
  onto the pulley's rim — and evaluated by FK at each perturbed UR waypoint. If the prediction is
  below `--min-clearance`, the waypoint is lifted along world `+z` until it is not. A lift raises
  every gripper point equally and leaves the tilt — which is what produces `under`/`slanted` —
  untouched; the tilt is only shrunk if even a 50 mm lift cannot clear the board (never observed).
  After planning, `path_clearance` re-checks the whole commanded UR path (every 4th pose, with the
  jaw byte commanded there) and lifts again if the closest approach happens *between* waypoints,
  re-planning up to 3 times. The osc backend runs this same path check on straight lines built from
  the CURRENT MEASURED Franka/UR poses (not the nominal FK), so it accounts for wherever the sim
  actually is at the start of the perturbed move. The report goes into the episode's `sim_clamp`
  and the `index.json` row's `clamp` / `clamp_lift_mm` / `clamp_tilt_scale`.
- **Detective** (`Gripper.measure`, every 4 control steps during playback): every collider placed
  by its own measured body pose. The per-episode minimum is `sim_min_board_clearance_mm` in the
  file and `min_board_clearance_mm` in the index row; the per-frame value is
  `sim_board_clearance_mm`. Negative means the collision geometries overlapped, i.e. real contact.

**The Franka is not covered by the guard**, only bounded indirectly: its excitation offset is
capped to `lin_speed * dt * cap_factor` (12 mm / 4.3 deg at the defaults, §5.2), floored at
`--excite-down-mm` so it cannot dive further than that below the un-excited knot, and its
`finger_tip` height above the board plate is measured every control step
(`sim_franka_tip_clearance_mm`, `sim_min_franka_tip_clearance_mm`) but not clamped. Measured
2026-09-23 (40-episode runs): min Franka tip-to-plate 23.4 mm (white excitation) / 29.7 mm (off);
never negative. With `ou` (20 episodes, `--seed 1000`): 17.3 mm.

**Contact episodes are kept, not skipped**: the physics is valid, only the scenario is unwanted.
They are flagged `sim_board_contact` (file) / `board_contact` (index row), listed in the index
summary's `board_contact_files`, and logged at WARNING level, so a training set can filter them
with one predicate. Measured 2026-09-23 (40-episode osc runs): min board clearance 2.11 mm
(white excitation) / 2.12 mm (off), 0 contacts in either run; largest guard lift 17.0 mm (`over`).

**Why 4 mm.** Measured over the 100 recorded episodes of `data/lcs/20260922-131647-ep100` (the
pre-guard position-backend run, replayed from its recordings):

| intent | measured min clearance (mm) | FK prediction at the waypoints (mm) |
|---|---|---|
| `engaged` | 1.99 .. 5.63 | 4.36 .. 7.00 |
| `over` | 1.15 .. 3.40 | 1.51 .. 6.76 |
| `slanted` | 2.40 .. 5.85 | 4.48 .. 7.03 |
| `under` (old dz -11 .. -6) | **-5.00 .. -2.13 (all 30 in contact)** | -5.00 .. -3.48 |

(Those predictions used the closed-jaw set only; the shipped guard also uses the open-jaw set and
the path check, which is why `over` now needs a lift.) The prediction has the same sign as the
measurement in all 100 episodes. The *unperturbed* `place_3` already predicts only 2.81 mm — the
fingers have to reach past the pulley rim to seat the belt — and the `engaged`/`slanted` draws all
predict >= 4.36 mm, so 4 mm is the largest threshold that never touches those two classes while
still being a real guard.

Clearance is **not monotone in z**: from ~10 to ~25 mm above the nominal `place_3` the binding
obstacle switches from the plate to the large pulley's flange, a few tenths of a mm radially from
the open finger, and lifting barely helps until the finger clears the flange top. That valley is
exactly where the `over` class lives, which is why `over` episodes are lifted the most while
`engaged`, `under` and `slanted` are usually not lifted at all.

### 5.2 Input excitation

`--backend osc` optionally perturbs the commander's Franka knots 1..6 (never knot 0, the measured
pose) every control step, independently of the episode's `pre_place_2`/`place_3` perturbation:

- **What (`--excite-mode ou`, default):** an Ornstein-Uhlenbeck offset `x = (dpos, rotvec)`
  (`round_belt_task.commander.OuExcitation`), 0 at the episode start and stepped once per
  `--sample-period`: `x <- a x + sqrt(1 - a^2) sigma xi`, `a = exp(-dt / --excite-tau-s)`
  (0.83 at 0.075 s / 0.4 s). Each axis has stationary std `sigma` = `--excite-pos-mm` (1.5 mm) /
  `--excite-rot-deg` (1 deg); the per-sample change has std `sigma sqrt(2 (1 - a))` (0.88 mm).
  The rotation is a left rotation by `rotvec`.
- **`--excite-mode white`:** the pre-OU scheme, kept so earlier runs stay reproducible: a fresh
  uniform-ball offset (radius `--excite-pos-mm`, default 4 mm) plus a rotation about a random
  axis (angle `~ U(0, --excite-rot-deg)`, default 2 deg) every sample (`draw_excite`).
  Uncorrelated draws zig-zag the command at 13 Hz with offsets comparable to the 6 mm nominal
  knot step; OU replaces it for that reason.
- **Why:** MPC (C3) will query the learned model at states/actions it chooses, not only the ones
  the nominal waypoint-follower visits. Exciting the Franka knots makes the collected `u_t` cover a
  neighbourhood of the nominal command instead of only it.
- **Bounds:** the position offset's `z` is floored at minus `--excite-down-mm` (2 mm) first, then
  the whole offset is scaled (never past magnitude 1) so the excited knot step stays inside
  `--excite-cap-factor * lin_speed * dt` (12 mm at the defaults); the rotation is scaled the same
  way against `--excite-cap-factor * ang_speed * dt` (4.3 deg) by bisection on the angle. The
  index row / summary `excite_clip` counts the control steps where the floor / cap bound.
- **Phase rule:** on through the move to the last target (`place_3`), off through its
  hold/dwell/settle, so the outcome label reflects the sampled perturbation. `white` switches off
  when `place_3` is reached. `ou` ramps the held offset linearly to 0 over 0.3 s (no new noise, no
  jump). The ramp starts once the Franka is within `lin_speed * 0.3 s + pos_tol` (29.5 mm) of
  `place_3`, or at the reach, whichever comes first. A held offset at the reach would shift the
  latched pose.
- **RNG stream:** its own per-episode generator (`[seed, episode_index, 7]`), independent of the
  `pre_place_2`/`place_3` perturbation draw (`[seed, episode_index]`) — enabling/disabling
  excitation does not change which perturbation a given `--seed`/episode draws.
- **Recorded:** `excitation` (mode, resolved `pos_mm`/`rot_deg`, `tau_s`, `ramp_s`,
  `fade_dist_mm`) in `index.json` and every episode's `sim_meta`; the applied offset per frame in
  `sim_excite_dpos_m` / `sim_excite_rotvec`.
- **How to disable:** `--excite-pos-mm 0 --excite-rot-deg 0` (also off automatically under
  `--scenario`).
- **UR (`--excite-ur-pos-mm 2 --excite-ur-rot-deg 1`, `ou` mode only):** a second
  `OuExcitation` (own RNG stream `[seed, episode_index, 11]`, same `tau`/fade) offsets the UR
  TARGET pose fed to `UrLineCommander.tick` (`dpos` added, rotation applied on the left in the
  world frame); its z offset is clipped to >= 0 and it fades out like the Franka's near
  `place_3`. The target then moves every sample, so the UR line regenerates every sample (from
  the measured pose), which is intended. Clearance: the guard's margin is unchanged; instead
  `UrExciteGuard` scales the offset each time it changes to the largest `s` in [0, 1] whose
  excited target needs no `clearance.required_lift` (the 2F-85 colliders, so a rotation's lever
  arm counts), checked with both the current and the target's Robotiq byte. The applied scale
  falls to that value at once and recovers at `1 / ramp_s` per second; `check_lcs_tuples.py` T7
  checks this adds no jerk (a synthetic binding guard: 45 % of samples scaled, jerk ratio 0.82).
  Recorded: `excitation["ur"]`, `sim_excite_ur_raw` (6, before the scale), `sim_excite_ur_dpos_m`
  / `sim_excite_ur_rotvec` (applied), `sim_ur_excite_scale`, and per episode `ur_excite`
  (`frames_active`, `frames_scaled`, `scale_mean_active`, `scale_min`) in `index.json`.

Measured 2026-09-23, `--episodes 20 --seed 1000` (same perturbations in all rows). Move frames
only. "Jerk" is the RMS of the Franka action's second difference
`|u_{t+1} - 2 u_t + u_{t-1}|`.

| run | jerk pos / rot | `dxyz_f` std x/y/z (mm) | `drotvec_f` std (deg) | tracking rms / p95 (mm) |
|---|---|---|---|---|
| no excitation | 1.9 mm / 0.7 deg | 0.91 / 4.49 / 2.21 | 0.41 / 1.12 / 0.40 | 0.73 / 1.51 |
| `ou` (defaults) | 2.9 mm / 1.8 deg | 1.53 / 4.50 / 2.44 | 0.80 / 1.27 / 0.79 | 0.80 / 1.51 |
| `white` (4 mm / 2 deg) | 8.3 mm / 3.6 deg | 2.20 / 4.84 / 2.76 | 0.90 / 1.37 / 0.84 | 1.49 / 2.09 |

With `ou` the z floor binds on 9.5 % of the excited control steps and the cap on 0.1 %. A 2.0 mm
position std flipped 3 of 20 outcomes against the no-excitation run with the same perturbations
(engaged 4/6, slanted 2/3); 1.5 mm keeps every intent >= 80 % on-intent. Over 40 episodes at
`--seed 0` the confusion is engaged 8/0/1/0, over 0/11/0/0, under 0/0/10/0, slanted 0/2/0/8, with
min board clearance 0.96 mm and no contact.

### 5.3 Slant variants

The plain `slanted` intent always slants the loop the same way: its high side faces the Franka
(`sim_slant_dir` = `franka_high`, `docs/lcs-dataset.md` §1). Variant intents aim for other
slants. The outcome label is still `slanted`; the variant is recorded as the `intent`.
`ClassRanges.franka_dz_mm` sets the Franka `dz` on its own. When it is unset, `dz` comes from
`franka_dxyz_mm`, so the existing intents draw the same values as before.

| intent (osc only) | UR `dz` mm | UR `tilt` deg | UR `dxy` mm | Franka `dz` mm | Franka `dxy` mm / `tilt` deg |
|---|---|---|---|---|---|
| `slanted_franka_high_steep` | -2.0 .. -1.0 | -5.5 .. -4.5 | -3 .. 3 | 3.5 .. 6.5 | -2 .. 2 / -3 .. 3 |

Measured 2026-09-23 (OU excitation unless noted; final-frame slant, median [min, max]):

| run | slanted / n | other outcomes | slant deg (slanted) | `slant_dir` (slanted) |
|---|---|---|---|---|
| tune, seed 24 | 27 / 30 | over 3 | 13.8 [11.5, 15.4] | franka_high 27 |
| collected, seed 3000 | 12 / 16 | over 4 | 14.2 [11.2, 16.6] | franka_high 12 |
| collected, no excitation, seed 4000 | 4 / 4 | - | 14.4 [13.4, 15.6] | franka_high 4 |

For comparison, the plain `slanted` intent measures 12.6 [8.1, 14.6] deg (p5/p95) on ep300. The
`over` episodes (h median 15-20 mm) have the same slant; they fell on the other side of
"≥ 1 body seated". A first band of Franka `dz` 4..10 mm gave 11 / 30 `over`.

**Other directions: tried and dropped.** We tuned over 827 osc episodes (4 mm guard on, UR
`dz` -2..20 mm, UR tilt ±14 deg, Franka `dz` -20..10 mm, Franka tilt ±16 deg, UR/Franka `dxy` up
to ±10 mm, plus a probe-only UR pitch of ±15 deg about `z x tangent`).

- **`ur_high`: infeasible.** It never occurred in a `slanted` episode (1 of 827 episodes overall:
  an `under` at 1.5 deg). The furthest the uphill direction turned toward the UR was
  `roll-`, at slant axis <= 125 deg. Three limits block it:
  - The Franka end sits ~20 mm above the UR's at `place_3`.
  - UR `dz` 8..20 mm falls in the clearance guard's flange valley (§5.1). The guard lifts the UR
    9-34 mm, which gives `over`.
  - Lowering the Franka by more than ~15 mm takes its finger tip to the plate: 0.2 / -0.1 mm
    tip clearance at Franka `dz` -19 / -20.

  UR `|dxy|` up to 10 mm moved the axis little and caused board contacts.
- **`roll+`: dropped.** It shows up only as `under` (h median -6..-19 mm), from a UR tilt of
  +4..+12 deg. The belt goes straight from `engaged` to `under`, with 1 `slanted` roll+ episode
  (3.1 deg) in ~140 tries.
- **`roll-`: dropped (below the 50 % bar).** The best band gave 46 / 100 `slanted` (43 roll-,
  3 franka_high; 5.5..10.2 deg), with over 36 and engaged 18. The band was:
  - UR `dz` 4..5.5 mm, `dxy` ±1 mm, tilt -5..-3.5 deg;
  - Franka `dz` -14..-12.5 mm, tilt 1..4 deg;
  - UR pitch 0..3 or 1..4 deg.

  Inside that band the outcome does not track any knob; it varies with excitation and latch
  noise. Every outcome in the band is roll-, so relabelled `over` / `engaged` data with a roll-
  slant is available at this rate.
- **`combined` (UR-high pitch + roll): dropped** because `ur_high` is infeasible.

### 5.4 Yaw fields

`Perturbation` has two additive fields, `ur_yaw_deg` and `franka_yaw_deg` (default 0). Each
rotates that arm's perturbed waypoint orientation about world z, after the tilt. `to_dict`
adds them only when either is non-zero, so older rows are unchanged. The intent ranges above do
not draw them; `--place3-mode continuous` / `engaged_band` (§4.6) sets them.

## 6. Frames and conventions

- **Board -> world:** `X_WF = X_WB * X_BF`. Compiled-segment waypoints (`pre_pick_*` .. `place_3`)
  use the `board` weld of magna's compiler default scene (`round-belt-scene.dmd.yaml`, identical
  to this scene's `X_W_BOARD`); the two live waypoints (`start`, `place_11`) use the controller
  yaml's `task_board_position` / `task_board_orientation` instead — the two board positions
  differ by 27.2 mm in x (y -0.10 mm, z -0.12 mm) and are not interchangeable
  (`round_belt_task.waypoints`).
- Waypoint quaternions in the magna yaml are `wxyz`; the dataset's pose fields are also `wxyz`
  (`POSE_LAYOUT = "xyz_wxyz"`, `docs/lcs-dataset.md` §2).
- **Franka EE frame** = `finger_tip` (`panda_link0 -> panda_link8 -> X_LINK8_HAND -> finger_tip`).
  **UR EE frame** = magna's `tracking_frame` (`X_W_UR10 -> base_link -> wrist_3_link ->
  T(0,0,0.194) * RPY(pi,0,pi/2)`). Both `round_belt_task.arm_kinematics`.
- **Gripper commands:** Franka `mm = cmd * 2000`; UR `byte = clamp(cmd / 0.04, 0, 1) * 255`
  (integer truncation, so `cmd=0.01` -> byte 63, not 64).
- **osc backend `state`:** the EE pose fields are the MEASURED pose (FK of the measured joints),
  i.e. what the OSC itself computes from `FRANKA_STATE`, stored as `sim_ee_franka`/`sim_ee_ur` too.
  This equals the commander's knot 0 (`sim_cmd_knot0_franka`) only on rows where the target is still
  more than the 5.5 mm reach tolerance away; on the last few move rows and every hold/dwell row
  knot 0 lags the measurement by up to a few mm (`docs/lcs-dataset.md` §3).

## 7. Known limits and tuning knobs

- **OSC internal state is not restored** across a snapshot (§3): only the sim's own state/control
  is captured; the OSC's own trajectory buffers/filters carry over unchanged, so a restore rebases
  the clock and then holds + settles rather than resetting the controller. Measured determinism
  (`check_osc_backend.py` K2, 2 restores + 100-step settle each): 403 strictly increasing OSC
  utimes, held `(True, True)` both times, Franka joint drift 4e-6 rad, belt body position drift
  (max over all bodies) 0.011 mm.
- **Reach tolerance** (`pos_tol=5.5 mm`, `ori_tol=0.15 rad`, `CommanderParams`, magna's hardware
  values): the Franka latches up to 5.5 mm short of a waypoint exactly as on hardware — this
  squashes small perturbations near `place_3` and drove the `OSC_RANGES` retune (§5).
- Deep `under` perturbations can make the belt slip off the pulley mid-move and land `over`
  instead — a real physical outcome, not a bug.
- The clearance guard predicts from a rigid snapshot of the gripper, so it is optimistic by a
  couple of mm against the measurement (PD tracking lag, belt reaction). Raise `--min-clearance`
  if a run needs more margin, at the cost of lifting more `over` (and eventually `engaged`)
  waypoints away from their sampled `dz`.
- `--fresh-pick` (position backend only) re-runs the ~40 s nominal pick per episode instead of
  restoring a snapshot; use it only to sanity-check the snapshot path, not for bulk collection.
- `--backend position` is legacy: commanded (not measured) `state`/`action`, joint-PD tracking, no
  OSC/LCM dependency (§3.1). Keep it for fast iteration or when magna is unavailable.
- Throughput, `--backend osc` (measured 2026-09-23, point clouds on, GPU otherwise idle): mean
  6.10 s/episode (restore ~0, settle 0.42, guard 0.52, render 0.44, clearance 0.17, commander 0.43,
  lcm_wait 0.03, physics 3.96, write 0.08 s), ~9.75 episodes/min, mean 895 control steps / 60.6
  sampled frames per episode. `--backend position` throughput: ~6.3 s/episode, ~9.5 episodes/min
  (unchanged, `data/lcs/20260922-151152-clearance`).
- Tracking error, `--backend osc` (`sim_tracking_err_mm`, move frames, mm, 40-episode runs):
  Franka rms/p95/max 1.42/2.12/8.01 (white excitation) / 0.77/1.51/7.39 (off); UR 1.36/3.03/4.77
  (both); `ou` Franka rms/p95 0.80/1.51 (§5.2). The Franka latches short of `pre_place_2` /
  `place_3` by up to 5.49/5.50 mm (mean 2.84/5.34 mm) — the reach tolerance above, not tracking
  error.

### 7.1 Hardware caveats

Differences between this sim's `--backend osc` collection and a hardware run at the same sample
period:

- **Camera cadence.** Hardware ZED clouds arrive at 20 Hz (every 50 ms); 75 ms sampling therefore
  does not land exactly on a cloud. Use nearest-cloud matching bounded to <= 25 ms skew, or run the
  camera at `1 / 0.075 ≈ 13.3 Hz` if the driver supports it. The sim has no such mismatch: it
  renders the point cloud at the exact sample step every time.
- **Controller rate.** This sim's control loop is 200 Hz (5 ms); the real OSC can run up to 1 kHz.
  The OSC replans from whatever `FRANKA_STATE` it receives, so a slower feed here means fewer,
  larger re-plans between samples than hardware would produce, not a different control law.
- **UR execution path.** On hardware the UR follows Cartesian commands through its own driver; here
  the UR's line target is converted to a joint command by per-step numpy IK plus the sim's position
  drive (§2). Both track the same 2-knot line, but the low-level tracking is not identical.

## 8. Checks

`docs/lcm-simulation.md` §8 has the full table. Covering this pipeline:
`check_commander.py` (the waypoint-generator / UR-line emulation), `check_osc_backend.py` (the
lock-step OSC child process), `check_sim_snapshot.py` (restore fidelity), `check_inproc_motion.py`
(waypoints, FK/IK, the nominal pick and a placement move), `check_lcs_dataset.py` (the `.npz`
writer/validator), `check_lcs_outcome.py` (the outcome classifier), `check_lcs_collector.py` (an
end-to-end collection run on both backends, the clearance guard and an optional `lcs_learning`
loader smoke test), `check_lcs_tuples.py` (osc-backend tuple semantics: timing, `cmd_delta` and
old-definition alignment, pre-hold/hold zeros, tracking error, belt-point identity, the
`lcs_learning` loader, Franka + UR excitation smoothness, the `rewrite_actions.py` round trip,
and the realised-vs-`u` causality table; `--causality-dir <run>` prints only the table). The live
checks take `--lcm-url` (use a private group per run).
