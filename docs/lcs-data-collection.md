# LCS data collection (round-belt engagement)

How to generate `(observation_t, action_t, observation_t+1)` episodes of the belt being engaged
into the large pulley, for `~/git/lcs_learning`. On-disk format: `docs/lcs-dataset.md` (this doc
does not repeat it). Implemented by `src/task_common/{lcs_dataset,sim_snapshot}.py`,
`src/round_belt_task/{waypoints,arm_kinematics,motion,offline_simulation,perturbation,
outcome}.py`, `scripts/lcs/make_start_state.py`, `scripts/collect_lcs_dataset.py`.

## 1. What this is

One episode covers `pre_place_1 -> place_3` plus a settle: the belt is already held by both arms
(picked at `pre_place_1`) and is carried to, and engaged with, the board's large pulley. Each
episode perturbs the `pre_place_2` / `place_3` waypoints so the outcome varies across
`engaged` / `over` / `under` / `slanted` / `outside` / `other` (`round_belt_task.outcome`).

## 2. Pipeline

1. Once: run the nominal in-process pick (`pre_pick_0 -> pre_place_1`, belt gripped physically)
   and snapshot the resulting sim state (`scripts/lcs/make_start_state.py`).
2. Per episode:
   - restore the snapshot (full state/control restore, no belt teleport);
   - sample an intent and perturb `pre_place_2` / `place_3` (`round_belt_task.perturbation`), then
     clamp the perturbed UR waypoints so the 2F-85 stays clear of the board (§5.1);
   - build a Cartesian trajectory to the perturbed waypoints and solve it with numpy
     damped-least-squares IK on both arm chains (`round_belt_task.motion`);
   - play it back with the sim's own MuJoCo joint position drive (`round_belt_task.
     offline_simulation`);
   - sample every `--sample-period`: state, action, a camera point cloud render, and (optionally)
     a `RunRecorder` frame; track the measured gripper -> board clearance every 4 control steps;
   - classify the outcome from the last frame's belt geometry (`round_belt_task.outcome`);
   - write `episode_XXXX.npz` (`docs/lcs-dataset.md`) and append a row to `index.json`.

## 3. Why in-process motion instead of magna over LCM

magna's assembly controller needs a bazel recompile per waypoint set and hard-codes the shared
LCM debug group, so it cannot batch headlessly. The in-process path instead interpolates a
Cartesian trajectory to each waypoint and solves it with numpy IK, then feeds the result to the
sim's existing joint position drive (`ke`/`kd` = 700/110, same as the LCM sim). This is headless,
batchable and deterministic (see `check_inproc_motion.py` M5).

**Fidelity caveat:** this tracks a Cartesian target with a joint position drive, not magna's
Cartesian-space OSC controller. Tracking error stays within a few mm / degrees away from contact
(`check_inproc_motion.py` M4), but the two controllers are not the same.

**The belt is picked physically** (both grippers close on it at the `pick` waypoint) and the
Drake belt-teleport trigger is disabled, so a start-state snapshot captures a physically
consistent grasp, not a scripted one.

## 4. How to run

### 4.1 Build the start state (once, or after a scene change)

```bash
uv run python scripts/lcs/make_start_state.py
```

| flag | default | meaning |
|---|---|---|
| `--out` | `data/lcs/start_states/pre_place_1.npz` | snapshot path |
| `--record [DIR]` | off (const `data/lcs/recordings`) | also record the run |
| `--params` | magna's `round_belt_controller_params_sim.yaml` | waypoints source (read-only) |
| `--arm-ke` / `--arm-kd` | `700.0` / `110.0` | arm position-drive gains |
| `--no-velocity-lead` | off | plain position targets, no `kd/ke * qdot` feed-forward |
| `--hold` | `1.0` | hold at the end [s] |

Fails with `GRASP FAILED: ...` (exit 1) if either gripper does not end up holding the belt.

### 4.2 Collect episodes

```bash
uv run python scripts/collect_lcs_dataset.py --episodes 40 --seed 0
```

| flag | default | meaning |
|---|---|---|
| `--out` | `data/lcs/<YYYYmmdd-HHMMSS>-<label>` | output dir |
| `--label` | `lcs` | used in the default `--out` |
| `--episodes` | `20` | episode count |
| `--seed` | `0` | RNG seed (reproducible: each episode's draw is keyed on `[seed, episode]`) |
| `--intents` | `engaged,over,under,slanted` | sampled per episode (see `--weights`) |
| `--weights` | `1,1,1,1` | one weight per `--intents` entry |
| `--start-state` | `data/lcs/start_states/pre_place_1.npz` | snapshot from 4.1 |
| `--fresh-pick` | off | re-run the nominal pick per episode instead of restoring (slow) |
| `--record` | off | a `RunRecorder` run per episode under `<out>/recordings/` |
| `--settle-s` | `1.0` | dwell at the end of the move before the last sample |
| `--min-clearance` | `4.0` | mm of 2F-85 -> board clearance to clamp the UR waypoints to (§5.1); `0` = measure only |
| `--sample-period` | `0.1` | seconds; must be a multiple of the 5 ms control step |
| `--arm-ke` / `--arm-kd` | `700.0` / `110.0` | arm position-drive gains |
| `--params` | magna's `round_belt_controller_params_sim.yaml` | waypoints source (read-only) |
| `--no-pcd` | off | skip camera renders (files then fail dataset validation; speed tests only) |
| `--dry-run` | off | print the sampled perturbation table and exit, nothing written |
| `--thresholds` | none | `KEY=VALUE` overrides of `OutcomeThresholds`, e.g. `engaged_arc_deg=75` |

Prints a confusion table (`intent` rows x `outcome` columns), a per-intent clearance table
(min / median clearance, largest guard lift, contact count) and a summary line (episodes
ok/skipped, episodes/min, mean per-episode timing) when done. An episode is `skipped`, not
written, if the perturbed waypoints are unreachable (IK failure).

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

## 5. Perturbation classes and outcome thresholds

Each episode samples an intent, then a UR-arm position offset + tilt about the belt tangent (the
horizontal Franka-tip -> UR-tip direction at the nominal `pre_place_2`); the Franka gets shared
jitter. Shipped defaults (`round_belt_task.perturbation.DEFAULT_RANGES`, mm / deg, UR `dxy` and
Franka jitter shared across intents):

| intent | UR `dz` mm | UR `tilt` deg | UR `dxy` mm | Franka `dxyz` mm | Franka `tilt` deg |
|---|---|---|---|---|---|
| `engaged` | 1.5 .. 5 | -2 .. 1.5 | -3 .. 3 | -2 .. 2 | -3 .. 3 |
| `over` | 11 .. 20 | -3 .. 1 | -3 .. 3 | -2 .. 2 | -3 .. 3 |
| `under` | 2 .. 4 | 10 .. 18 | -3 .. 3 | -2 .. 2 | -3 .. 3 |
| `slanted` | 2 .. 4 | \|3.5 .. 5.5\|, random sign | -3 .. 3 | -2 .. 2 | -3 .. 3 |

Outcome thresholds (`round_belt_task.outcome.DEFAULT_THRESHOLDS`, unchanged from the tuning run):
`neighbour_radial_mm=12`, `neighbour_inner_mm=10`, `in_groove_axial_mm=4`, `in_groove_radial_mm=5`,
`engaged_arc_deg=60`, `partial_arc_deg=15`, `over_under_h_mm=5`, `slant_spread_mm=8`,
`outside_min_bodies=2`. A wrapped arc >= `engaged_arc_deg` always classifies `engaged`, even over a
tilted belt; `--thresholds engaged_arc_deg=75` reclassifies the borderline ~6 deg-tilt / ~68 deg-
wrap group as `slanted` if that is preferred for a given run.

`under` is **tilt-driven, not dz-driven**: a negative UR `dz` is the only way to drag the 2F-85
fingers through the board plate, and every one of the 30 `under` episodes of the earlier
`-11 .. -6 mm` range did exactly that (§5.1). A `+10 .. 18 deg` tilt about the belt tangent tips
the loop below the pulley groove while the gripper stays 2-6 mm clear, which is why the shipped
`under` row looks like the `slanted` row with a bigger tilt.

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
plate) is visual-only and stays > 40 mm from the gripper's path.

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
  re-planning up to 3 times. The report goes into the episode's `sim_clamp` and the `index.json`
  row's `clamp` / `clamp_lift_mm` / `clamp_tilt_scale`.
- **Detective** (`Gripper.measure`, every 4 control steps during playback): every collider placed
  by its own measured body pose. The per-episode minimum is `sim_min_board_clearance_mm` in the
  file and `min_board_clearance_mm` in the index row; the per-frame value is
  `sim_board_clearance_mm`. Negative means the collision geometries overlapped, i.e. real contact.

**Contact episodes are kept, not skipped**: the physics is valid, only the scenario is unwanted.
They are flagged `sim_board_contact` (file) / `board_contact` (index row), listed in the index
summary's `board_contact_files`, and logged at WARNING level, so a training set can filter them
with one predicate.

**Why 4 mm.** Measured over the 100 recorded episodes of `data/lcs/20260922-131647-ep100` (the
pre-guard run, replayed from its recordings):

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
still being a real guard. With it, the 20-episode verification run measured 2.20 mm worst case and
no contact. `--min-clearance 0` turns the clamp off and keeps the measurement.

Clearance is **not monotone in z**: from ~10 to ~25 mm above the nominal `place_3` the binding
obstacle switches from the plate to the large pulley's flange, a few tenths of a mm radially from
the open finger, and lifting barely helps until the finger clears the flange top. That valley is
exactly where the `over` class lives, which is why `over` episodes are lifted 4 .. 15 mm (their
`place_3` predicts 0.1 .. 3.6 mm before the clamp) while `engaged`, `under` and `slanted` are
usually not lifted at all. The lift does not change what the belt does — the measured `over`
`h_median` stays 9 .. 14 mm, the same band as the unguarded run.

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

## 7. Known limits and tuning knobs

- Joint position drive vs magna's OSC (§3): don't expect bit-identical trajectories to a hardware
  or Drake run, only physically similar ones.
- A tilted-but-well-seated belt (>= 60 deg wrapped arc) classifies `engaged`, not `slanted`; use
  `--thresholds engaged_arc_deg=<N>` to move that boundary (§5).
- Deep `under` perturbations can make the belt slip off the pulley mid-move and land `over`
  instead — a real physical outcome, not a bug.
- The clearance guard predicts from a rigid snapshot of the gripper, so it is optimistic by a
  couple of mm against the measurement (PD tracking lag, belt reaction). Raise `--min-clearance`
  if a run needs more margin, at the cost of lifting more `over` (and eventually `engaged`)
  waypoints away from their sampled `dz`; the guard adds ~0.2 s/episode of measurement plus one
  re-plan (~0.3 s) whenever the path check fires.
- `--fresh-pick` re-runs the ~40 s nominal pick per episode instead of restoring a snapshot; use
  it only to sanity-check the snapshot path, not for bulk collection.
- Throughput (measured 2026-09-22, point clouds on, GPU otherwise idle): ~6.3 s/episode, ~9.5
  episodes/min. File size ~1.5 MB/episode; `--record` adds ~1.3 MB/episode.

## 8. Checks

`docs/lcm-simulation.md` §8 has the full table. Covering this pipeline:
`check_sim_snapshot.py` (restore fidelity), `check_inproc_motion.py` (waypoints, FK/IK, the
nominal pick and a placement move), `check_lcs_dataset.py` (the `.npz` writer/validator),
`check_lcs_outcome.py` (the outcome classifier), `check_lcs_collector.py` (an end-to-end
collection run, the clearance guard (C7) and an optional `lcs_learning` loader smoke test).
