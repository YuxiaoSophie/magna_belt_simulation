# LCS episode format (`.npz`)

The on-disk format of one simulated episode for `~/git/lcs_learning`
(`RoundBeltTupleDataset`). Implemented by `src/task_common/lcs_dataset.py`
(`EpisodeWriter`, `validate_episode`); checked by `scripts/checks/check_lcs_dataset.py`.

**Source of truth:** magna's reference collector,
`git show origin/stephen/non-rigid-lcs-learning:python/collect_round_belt_dataset.py`
(commit `51e1814`, 2026-06-22), plus `batch_collect_round_belt_dataset.py` on the same branch,
which adds `trajectory_label`. Where the collector and the lcs_learning README disagree, the
collector wins, because it produced the logs lcs_learning trains on.

## 1. Keys

`T` = number of samples. All positions are in the world frame, in metres. The file is written with
`np.savez_compressed`; object arrays need `np.load(path, allow_pickle=True)`.

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `pcd` | object | `(T,)`, each `(N_t, 3)` float32 | Cropped, voxelised camera cloud (ragged, `N_t` varies) |
| `pcd_rgb` | object | `(T,)`, each `(N_t, 3)` uint8 | Colour of each `pcd` point |
| `pcd_belt` | float32 | `(T, 150, 3)` | Belt points, temporally consistent per index (see §4) |
| `pcd_kinematic` | float32 | `(T, K, 3)`; `K = 0` by default | Sampled rigid gripper/board geometry (off by default) |
| `state` | float64 | `(T, 40)` | Proprioception, §2 |
| `actions` | float64 | `(T, 12)` | Commanded end-effector deltas, §3 |
| `utime` | int64 | `(T,)` | Sim time of the sample in µs, spaced by exactly the sample period |
| `trajectory_label` | str | `()` | Episode class (magna: `success`, `tilted_success`, `slip_off`) |
| `point_cloud_source` | str | `()` | `"camera_plus_belt"` |
| `record_kinematic_points` | bool | `()` | `True` iff any `pcd_kinematic` frame is non-empty |
| `kinematic_model_names` | str | `(3,)` | Collector's default sampler config; informational only |
| `kinematic_body_name_patterns` | str | `(8,)` | Collector's default sampler config; informational only |
| `kinematic_sampled_body_names` | float64 | `(0,)` | Empty, like `np.array([])` in the collector |
| `sim_step` | int64 | `(T,)` | Sim-only: control-step index of the sample |
| `sim_time` | float64 | `(T,)` | Sim-only: time of the sample in seconds; `--backend osc`: the OSC clock (`utime`·1e-6), not `sim_step`·0.005 — use `sim_step` for the sim step |
| `sim_meta` | str | `()` | Sim-only: JSON of the episode metadata plus an `lcs_format` block |
| `sim_<name>` | any | `(T, ...)` | Sim-only: per-frame extras passed by the caller |
| `sim_slant_deg`, `sim_slant_axis_deg`, `sim_slant_dir` | float64, float64, str | `()` | Sim-only: final-frame belt slant at the large pulley (see below); NaN / `n/a` without a plane fit |
| `sim_slant_deg_t`, `sim_slant_axis_deg_t` | float64 | `(T,)` | Sim-only: the same per frame |

`sim_*` keys never collide with collector keys. The loader and visualisers ignore them.

Slant (`round_belt_task.outcome.slant_metrics`, descriptive only; labels are unchanged): in the
large pulley's frame (axis = its +Z), an SVD plane is fitted to the belt bodies in the
classifier's neighbourhood (radial band around the seat; >= 3 non-collinear bodies).
`slant_deg` is the angle between that plane's normal `n` and the pulley axis `z`.
`slant_axis_deg` is the azimuth (from the pulley +X) of `z x n`, the axis that rotates `z`
onto `n`. `slant_dir` compares the uphill direction of the plane with the Franka -> UR tangent
(`index.json` `belt_tangent`, projected into the pulley frame): `ur_high` (within 45 deg),
`franka_high` (beyond 135 deg), otherwise `roll+` (uphill to the left of the tangent, seen from
+z) or `roll-`; `level` below 1 deg. `index.json` rows carry `final_slant_deg`,
`final_slant_axis_deg` (null if NaN) and `final_slant_dir`.

The loader reads only `state`, `actions` and the cloud keys for its `point_cloud_source`:
- `pcd_belt` whenever it is present
- `pcd_kinematic` for `belt_plus_static` and `belt_plus_kinematic`
- `pcd` for `camera` and `camera_plus_belt`

It casts every array to float32.

## 2. State layout (40)

`[q_franka(7), q_ur(6), v_franka(7), v_ur(6), ee_pose_franka(7), ee_pose_ur(7)]`

- `q`/`v`: joint positions and velocities, taken from `FRANKA_STATE` and `UR_STATE_SIM`
  (`lcmt_robot_output`). Only the 7 arm joints and the 6 arm joints; no gripper joints.
- Pose 7-vectors are `[x, y, z, qw, qx, qy, qz]` (`POSE_LAYOUT = "xyz_wxyz"`), in the world frame.
  The collector reads knot 0 of the controller's commanded trajectory, not a measured pose:
  - Franka: `end_effector_position_target` / `end_effector_orientation_target` on
    `TARGET_CARTESIAN_POSE_TRAJECTORY`. The frame is `finger_tip`
    (`franka_cartesian_osc_controller_params.yaml` `end_effector_name`); in this sim that is the
    body `panda_hand/finger_tip`.
  - UR: `ur_ee_position_target_world` / `ur_ee_orientation_target_world` on
    `UR_TARGET_CARTESIAN_POSE_TRAJECTORY`, i.e. `X_world_ur_base * X_base_flange(knot) *
    X_ur_flange_ur_ee`. The EE frame is magna's tracking frame, URDF `wrist_3_link * T(0,0,0.194)
    * RPY(pi,0,pi/2)`. In this sim that is `/ur10/wrist_3_link * X_USDWRIST3_URDFWRIST3 *` that
    offset (`src/round_belt_task/constants.py`, `LCM_UR_GRIPPER_TIP_Z`).
  - `round_belt_assembly_controller.cc` builds knot 0 from the current measured EE pose at plan
    time. Once a waypoint is reached, knot 0 is the latched reached position. So `ee_pose` is
    "commanded-at-plan-time", which is roughly the measured pose.
  - The collector normalises the UR quaternion; the Franka one is stored as received.

## 3. Action layout (12) and delta definition

`[dxyz_franka(3), dxyz_ur(3), drotvec_franka(3), drotvec_ur(3)]`, float64, world frame.

Collector definition (`_parse_trajectory_pose_and_delta`, `_delta_orientation_rotvec`), using the
two first knots of the same commanded trajectory:

- `dxyz = p_knot1 - p_knot0`, in the world frame.
- `drotvec = axis * angle` of `R_delta = R_knot1 * R_knot0^T`. This is a world-frame (spatial,
  left-multiplied) relative rotation, with angle in `[0, pi]` (Drake `AngleAxis`).
- If the trajectory has one knot, both deltas are zero.

The controller spaces the knots by the C3 `dt = 0.075 s` (`ACTION_KNOT_DT_S`). So the collector's
action is the commanded displacement over one C3 step, not the displacement to the next sample
(0.1 s later). When a waypoint is latched, knot 0 equals knot 1, and the action is exactly zero:
15 of 21 Franka rows in `log_001` are zero.

`action_vector(pose_franka_t, pose_franka_t1, pose_ur_t, pose_ur_t1)` implements this delta
(`dxyz = p1 - p0`, `drotvec = rotvec(R1 * R0^T)`) for any pose pair; the caller picks the pair.
Each file records its choice in `sim_meta["lcs_format"]["action_definition"]`, and
`validate_episode` rejects values outside `ACTION_DEFINITIONS`:

| `action_definition` | Pose 0 | Pose 1 | Written by |
|---|---|---|---|
| `cmd_delta` | Franka: knot 0 of the command published at `t`; UR: the line in force at `t`, sampled at `t` | Franka: knot 1 of that command; UR: the same line at `t + 0.075 s` | `--backend osc` (default since 2026-09-25), `eval_learned_mpc.py` (default), `rewrite_actions.py` |
| `knot1_minus_measured` | measured EE pose at `t` (= `state[:, 26:33]` / `state[:, 33:40]`) | Franka: knot 1 of the command published at `t`; UR: the commanded line at `t + 0.075 s` | `--backend osc` / harness before 2026-09-25, or `--action-definition knot1_minus_measured`; always kept as the extra `sim_action_knot1_minus_measured` |
| `knot1_minus_knot0` | knot 0 | knot 1 | magna's reference collector (not written by the sim) |
| `cmd_t1_minus_cmd_t` | commanded target at sample `t` | commanded target at `t+1` | `--backend position` (no key in the file) |

Files without the key (the position backend, older runs, magna logs) are accepted.
`command_action(definition, ...)` / `command_actions(...)` compute either OSC definition from the
stored `sim_cmd_*` and the measured poses; `realised_delta(state)` is `measured_{t+1} -
measured_t` in the action layout (last row NaN), stored as `sim_realised_delta`.

**osc backend (`cmd_delta`, default).** `u_t` = what the controller commands the arm to do over
the next 0.075 s: Franka `knot1_t - knot0_t`, UR `line_t(t + dt) - line_t(t)`. Why: under
`knot1_minus_measured` the steady tracking lag is part of `u` (hold rows and the UR between line
regenerations are dominated by it: UR x slope -0.01, -0.8 mm mean offset against 0.1 mm realised
motion), while at deployment knots and lines are re-anchored at the measured pose every replan, so
the same `u` is realised ~1:1 — a train/deploy mismatch. The difference of two commanded poses
cancels the lag, is exactly 0 on holds (`delta_rotvec` returns exact zeros for equal quaternions),
and is what the deployed controller already realises (knot 0 = the measured pose at the replan;
a UR line spanning exactly `dt`). The realised delta is only a diagnostic (an MPC output must be
a command); a per-arm hybrid was rejected. Row types: pre-hold (`sim_episode_step < 0`, phase
`prehold`): both arms exactly 0; free move: the `lin_speed * dt` step (+ excitation); reached
while moving / quiet hold (`sim_cmd_hold`, no excitation): Franka exactly 0 (the UR is 0 once its
line has ended). On the converted 2026-09-23 run (19 387 tuples) the realised-vs-`u` OLS slopes
are Franka x/y/z 0.89/0.92/0.94, rot 0.75/0.85/0.76; UR x/y/z 0.92/0.94/1.00, rot 0.98/0.90/1.11
(UR x/y/rot-y/rot-z have `u` std <= 0.4 mm / 1.4 mrad: never excited there), printed by
`check_lcs_tuples.py --causality-dir <run>`. `scripts/lcs/rewrite_actions.py` converts older
files into NEW dirs (`data/lcs/20260925-ep300-ou-cmd_delta/`, `data/lcs/demo/
20260925-nominal-cmd_delta/`, `data/lcs/demo/demo_episode_cmd_delta.npz`), non-action keys
byte-identical, provenance in `sim_meta["action_rewrites"]`.

**`knot1_minus_measured` (osc backend before 2026-09-25).** `u_t = knot1_t - ee_t`: the pose the controller is
told to reach one C3 step ahead, minus where the EE is now. This matches MPC, whose `x_sol[0]` is
the current (measured) state. It also makes the one-step residual exactly the tracking error:
`ee_{t+1} - (ee_t + u_t) = ee_{t+1} - knot1_t`, which is `sim_tracking_err_mm[t+1]` (Franka
column 0, UR column 1; 0 at `t = 0`). Per row type (4-episode smoke run, seed 1):

- **Free move** (`dist >= 5.5 mm` from the target): knot 0 is the measured pose bit for bit, so
  `u_t` equals the reference-style `knot1 - knot0` exactly (mean 6.4 mm, cap 12 mm).
- **Reached while moving** (`dist < 5.5 mm`, waiting on the UR): every knot is the target, so
  `u_t = target - ee_t` (1.4..5.2 mm). The reference action would be 0 or the excitation only.
- **Hold / done** (`sim_cmd_hold`): the command is the latched pose (the measured pose at the
  latch tick, up to 5.5 mm short of the target), so `u_t = latched - ee_t`, a small non-zero pull
  back against drift (0.07..1.2 mm, 0.35 deg max). It is not `target - ee_t`. The reference
  action is exactly 0 here.

The commanded knots are kept, so the reference-style action can be rebuilt:
`sim_cmd_knot1_franka - sim_cmd_knot0_franka` (position; rotation via `delta_rotvec(knot0,
knot1)`), and for the UR `sim_cmd_ur_t1 - sim_cmd_ur_t` (the line at `t + dt` minus the line at
`t`). `sim_cmd_knots_franka` holds all 7 knots.

## 4. Point clouds

- `pcd`: the collector stores the `POINT_CLOUD_CROPPED` message as received. It does no extra
  subsampling or ordering. The magna sim has already cropped and voxelised the cloud. `N_t`
  varies per frame (ragged), so `PCD_POINTS = None`. In this sim the cloud comes from
  `CroppedPointCloud.compute()` (voxel 5 mm, crop `[0.2,-0.25,0.015]..[0.7,0.25,0.11]`), and
  `camera_points()` passes it through unchanged except for a float32 cast and a finiteness check.
- `pcd_belt` (magna): `RoundBeltState.point_positions`, the vertices of Drake's deformable belt
  mesh (`drake_deformable_state_to_round_belt_state_converter.cc`). They are published in the
  taskboard frame and the collector maps them to world. There are 150 vertices with a fixed
  index-to-material mapping, so they are temporally consistent. They are **not** ordered along
  the loop: consecutive-index gaps reach 0.24 m, and the closed index polyline in `log_001` frame
  0 is 10.1 m long.
- `pcd_belt` (sim): `belt_points_ordered(belt_xyz_48x3)` samples the belt at fixed material
  coordinates (`sim_meta["lcs_format"]["belt_sampling"] == "material"`). The input is the 48
  rod-body origins, in index order, closed back to body 0. `material_table()` gives each point
  `k` a segment `i_k` and fraction `f_k`, fixed once: 150 points at equal arc length along the
  rest belt (`rest_belt_bodies()`: the scene ellipse's segment midpoints, semi-axes
  `REST_BELT_SEMI_AXES`), point 0 at body 0. Each frame, `p_k = (1-f_k) body[i_k] + f_k
  body[i_k+1]`, so point `k` stays on the same piece of belt however it stretches (rest spacing
  4.3 mm; stretched frames are not equally spaced). `spline=True` (not used by the collector)
  swaps the chord for a closed centripetal Catmull-Rom with the same `(i_k, f_k)`. To recompute:
  `np.stack([belt_points_ordered(b) for b in sim_belt_xyz])`. Files without `belt_sampling` used
  the old equal-current-arc-length resampling, whose index drifts up to ~0.9 body units (~12 mm)
  along the material as the belt stretches; `scripts/lcs/rewrite_belt_points.py RUN_DIR`
  rewrites them (idempotent, `--dry-run`).
- `pcd_kinematic`: by default the collector does not record kinematic points
  (`record_kinematic_points=False`), so it stores `(0, 3)` per frame. `camera_plus_belt` does not
  use it. The sim writes the same empty `(T, 0, 3)` by default (`kinematic_points()`).
- Loader behaviour (`_resize_points_ordered`): each frame is resized to a fixed count with
  `linspace` indices or deterministic tiling, not random sampling. Index-wise errors on
  `pcd_belt` therefore need temporally consistent indices.

## 5. Sample period

Constants: `SAMPLE_PERIOD_S = 0.075`, `SAMPLE_STEPS = 15` sim steps of 5 ms (`utime` step
`75000` µs). `--sample-period 0.1` gives 20 steps / `100000` µs.

- 0.075 s is the C3 `dt` and the knot spacing (`ACTION_KNOT_DT_S`), so each tuple
  `(x_t, u_t, x_{t+1})` is one MPC query: `u_t` is aimed at the pose one knot ahead, and
  `x_{t+1}` is measured exactly one knot later. The point cloud is rendered at the same step.
- magna's reference collector instead samples at 0.1 s (`MAGNA_LOG_PERIOD_S`): it is triggered
  by the 20 fps camera and rate-limited to 13.3 Hz, so the first cloud after 75.19 ms is the one
  100 ms later. Every inspected magna log has `utime` diffs of exactly `100000` µs; validate them
  with `period_us=MAGNA_LOG_PERIOD_US`.
- In the osc backend `utime` is the OSC clock (`FRANKA_STATE.utime`, equal to `sim_osc_utime`).
  It is monotonic across episodes of one run, because the offset is bumped at every restore.

## 6. Where the sim deviates from the collector, and why

| Point | Collector | Sim | Why |
|---|---|---|---|
| `pcd_belt` points | 150 FEM mesh vertices, index-consistent but unordered | 150 fixed-material-coordinate samples of the 48-body centreline, ordered | The sim belt is 48 rods, not a FEM mesh. Ordered points satisfy the loader's index-consistency assumption and the README's "ordered 150-point" wording. |
| `pcd_belt` storage | `np.array(list, dtype=object)`, which numpy turns into a `(T,150,3)` object array of Python floats | dense float32 `(T,150,3)` | Same indexing and `len`; no pickled scalars. |
| `pcd_kinematic` storage | `(T,0,3)` object array | dense float32 `(T,K,3)`; 1-D object array if `K` varies | Same reason. |
| `pcd` storage | `np.array(list, dtype=object)`, which collapses to N-D if all `N_t` happen to be equal | always a 1-D object array | Avoids the numpy collapse trap. |
| Action poses | knot 1 - knot 0 of the controller's trajectory (`knot1_minus_knot0`) | osc: knot 1 (Franka) / line at `t + 0.075 s` (UR) minus the measured pose at `t` (`knot1_minus_measured`); position: commanded target at `t+1` minus at `t` | osc: matches MPC (`x_sol[0]` = current state) and gives a real pull on reached/hold rows (§3). Same delta formula and frame. |
| `ee_pose` | trajectory knot 0 (the measured pose at plan time, except within 5.5 mm of a waypoint, where it is the target, and during holds, where it is the latched pose) | osc: the measured pose (FK of the measured joints), also stored as `sim_ee_franka`/`sim_ee_ur`; knot 0 is stored as `sim_cmd_knot0_franka`. position: the commanded target, with the measured pose in `sim_ee_*` | osc: the state MPC and the hardware see. `state == knot 0` only on free-move rows (§3). |
| `sim_*` keys | absent | present | Sim provenance (step, time, perturbation, outcome). The loader ignores them. |
| `kinematic_*` metadata | the collector's argparse values | the collector's defaults, verbatim | Parity only; the loader never reads them. |

## 7. Legacy format (`pcd_kept` / `pcd_removed`)

Older rollouts, produced by `mask_round_belt_dataset.py`, have no `pcd_belt`. They carry:
- `pcd_kept` (+ `pcd_rgb_kept`): points inside the masked region of interest.
- `pcd_removed` (+ `pcd_rgb_removed`): points the mask removed.

The loader branches on `"pcd_belt" in data`. For these files the encoder input is
`pcd_kept ∪ pcd_removed` (concatenated) and the reconstruction target is `pcd_kept` only,
whatever the `point_cloud_source`. `validate_episode(path, legacy=True)` accepts that layout.
The 2026-06-22 logs are the current generation (they have `pcd_belt`), so the default path
accepts them.

## 8. Inspected: `/home/hienbui/git/magna-logs/2026-06-22/log_001.npz`

| Key | dtype | Shape | Notes |
|---|---|---|---|
| `pcd` | object | `(21,)` | each `(N,3)` float32, N = 2004..2053; bbox x 0.30..0.57, y -0.15..0.18, z 0.015..0.11 |
| `pcd_rgb` | object | `(21,)` | each `(N,3)` uint8 |
| `pcd_belt` | object | `(21, 150, 3)` | Python floats; frame-to-frame index motion <= 4.4 mm |
| `pcd_kinematic` | object | `(21, 0, 3)` | empty |
| `kinematic_model_names` | `<U10` | `(3,)` | `panda_hand robotiq_85 nist_board` |
| `kinematic_body_name_patterns` | `<U22` | `(8,)` | the collector's defaults |
| `kinematic_sampled_body_names` | float64 | `(0,)` | |
| `point_cloud_source` | `<U16` | `()` | `camera_plus_belt` |
| `record_kinematic_points` | bool | `()` | `False` |
| `state` | float64 | `(21, 40)` | quaternion norms 1.0 |
| `actions` | float64 | `(21, 12)` | abs max: dxyz <= 16 mm, drotvec <= 0.03 rad |
| `utime` | int64 | `(21,)` | 13 500 000 .. 15 500 000, diffs all 100 000 |
| `trajectory_label` | `<U7` | `()` | `success` |

The other logs in that directory have the same keys:
- `log_003`: T=24. `log_004`: T=26. `log_006`: T=21. `log_007`: T=22. `log_008`: T=26.
  `log_009`: T=20. All have `utime` diffs of 100 000 µs.
- `log_002` and `log_005` (`slip_off`) have T=0. The loader skips them, and `validate_episode`
  rejects them (T < 2).

## 9. Episodes written by the MPC harness (`scripts/lcs/eval_learned_mpc.py`)

Same on-disk format (`EpisodeWriter` / `validate_episode`); the harness's `actions` come from the
**controller's own published messages** (its Franka trajectory's knot 1, its UR line's target),
not the offline commander's — `--action-definition` (default `cmd_delta`: `knot1 - knot0` of the
plan answering the tick, else the plan in force / `line(t+dt) - line(t)` of the UR line;
`knot1_minus_measured` for models trained on it, which also feeds `sim_latent_pred`), plus the
extras `sim_action_knot1_minus_measured` and `sim_realised_delta`. Extra `sim_*` keys per frame, on top of §1's:
`sim_latent` (16, the encoder's `z`; NaN in baseline mode — no encoder runs), `sim_latent_pred`
(16, the learned LCS's one-step-ahead prediction from the previous latent, learned mode only),
`sim_goal_dist` (per-stage whitened distance to that stage's goal, learned mode only) and
`sim_stage` (int, non-decreasing, `mpc_current_target_idx`; always 0 in baseline mode, which has
no stages). See `docs/learned-mpc-reference.md` for the full contract and how to run the harness; the
demonstration episode itself is collected with `--scenario nominal` (no perturbation, no
excitation) and lives under `data/lcs/demo/`, never inside a training data dir.
