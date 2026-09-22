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
| `sim_time` | float64 | `(T,)` | Sim-only: sim time of the sample, in seconds |
| `sim_meta` | str | `()` | Sim-only: JSON of the episode metadata plus an `lcs_format` block |
| `sim_<name>` | any | `(T, ...)` | Sim-only: per-frame extras passed by the caller |

`sim_*` keys never collide with collector keys. The loader and visualisers ignore them.

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

`action_vector(pose_franka_t, pose_franka_t1, pose_ur_t, pose_ur_t1)` implements exactly this
delta for any pose pair. The sim collector feeds it the commanded EE target at sample `t` and at
`t+1` (§6).

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
- `pcd_belt` (sim): `belt_points_ordered(belt_xyz_48x3)` resamples the belt centreline by arc
  length. The input is the 48 rod-body origins, in index order, closed back to body 0. The output
  is 150 equally spaced points, point 0 at body 0, walking in body-index order. These points are
  temporally consistent, and additionally ordered.
- `pcd_kinematic`: by default the collector does not record kinematic points
  (`record_kinematic_points=False`), so it stores `(0, 3)` per frame. `camera_plus_belt` does not
  use it. The sim writes the same empty `(T, 0, 3)` by default (`kinematic_points()`).
- Loader behaviour (`_resize_points_ordered`): each frame is resized to a fixed count with
  `linspace` indices or deterministic tiling, not random sampling. Index-wise errors on
  `pcd_belt` therefore need temporally consistent indices.

## 5. Sample period

Constants: `SAMPLE_PERIOD_S = 0.1`, `SAMPLE_STEPS = 20` sim steps of 5 ms.

- The collector is triggered by the point cloud and rate-limited by `--sample_rate_hz 13.3`
  (13.33 in the batch collector), so consecutive samples are at least 75.19 ms apart.
- The magna sim camera publishes at 20 fps (`round_belt_scene.yaml` `fps: 20`), one cloud every
  50 ms. The first cloud that passes the 75.19 ms limit is therefore the one 100 ms later.
- Every inspected log has `utime` diffs of exactly `100000` µs.

The architect's fallback of 0.075 s (the C3 `dt`) is only the action knot spacing (§3). `utime`
is sim time in µs; the magna logs start around 13.5 s, at the waypoint-5 trigger.

## 6. Where the sim deviates from the collector, and why

| Point | Collector | Sim | Why |
|---|---|---|---|
| `pcd_belt` points | 150 FEM mesh vertices, index-consistent but unordered | 150 arc-length samples of the 48-body centreline, ordered | The sim belt is 48 rods, not a FEM mesh. Ordered points satisfy the loader's index-consistency assumption and the README's "ordered 150-point" wording. |
| `pcd_belt` storage | `np.array(list, dtype=object)`, which numpy turns into a `(T,150,3)` object array of Python floats | dense float32 `(T,150,3)` | Same indexing and `len`; no pickled scalars. |
| `pcd_kinematic` storage | `(T,0,3)` object array | dense float32 `(T,K,3)`; 1-D object array if `K` varies | Same reason. |
| `pcd` storage | `np.array(list, dtype=object)`, which collapses to N-D if all `N_t` happen to be equal | always a 1-D object array | Avoids the numpy collapse trap. |
| Action poses | knot 0 / knot 1 of the controller's trajectory, 0.075 s apart | commanded EE target at sample `t` / `t+1`, 0.1 s apart | The in-process motion has no C3 knot trajectory. Same delta formula and frame. |
| `ee_pose` | trajectory knot 0 (roughly the measured pose at plan time) | the commanded Cartesian target at the sample time, so `state_t.ee + action_t == state_t+1.ee`; the measured pose is stored separately as `sim_ee_franka`/`sim_ee_ur` (PKG-lcs-collector) | No C3 plan in the sim. |
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
