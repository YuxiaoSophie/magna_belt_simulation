# Fixture key inventory

Shapes/dtypes are from the ORIGINAL files (full T).

## e2e_run/episode_pre_place_1_osc_0.npz

| key | shape | dtype |
|---|---|---|
| pcd | (459,) | object |
| pcd_rgb | (459,) | object |
| pcd_belt | (459, 150, 3) | float32 |
| pcd_kinematic | (459, 0, 3) | float32 |
| kinematic_model_names | (3,) | <U10 |
| kinematic_body_name_patterns | (8,) | <U22 |
| kinematic_sampled_body_names | (0,) | float64 |
| point_cloud_source | () | <U16 |
| record_kinematic_points | () | bool |
| state | (459, 40) | float64 |
| actions | (459, 12) | float64 |
| utime | (459,) | int64 |
| trajectory_label | () | <U7 |
| sim_step | (459,) | int64 |
| sim_time | (459,) | float64 |
| sim_meta | () | <U9025 |
| sim_ee_franka | (459, 7) | float64 |
| sim_ee_ur | (459, 7) | float64 |
| sim_cmd_knot0_franka | (459, 7) | float64 |
| sim_cmd_knot1_franka | (459, 7) | float64 |
| sim_cmd_knots_franka | (459, 7, 7) | float64 |
| sim_cmd_knot_times_franka | (459, 7) | float64 |
| sim_cmd_n_knots | (459,) | int16 |
| sim_cmd_traj_utime | (459,) | int64 |
| sim_cmd_ur_t | (459, 7) | float64 |
| sim_cmd_ur_t1 | (459, 7) | float64 |
| sim_cmd_ur_line | (459, 16) | float64 |
| sim_cmd_ur_fresh_exact | (459,) | int8 |
| sim_ur_line_own_window_delta | (459, 6) | float64 |
| sim_cmd_source | (459,) | int8 |
| sim_cmd_ctrl_hand_mm | (459,) | float32 |
| sim_cmd_ctrl_robotiq_byte | (459,) | int16 |
| sim_phase | (459,) | int16 |
| sim_stage | (459,) | int16 |
| sim_latent | (459, 32) | float64 |
| sim_latent_pred | (459, 32) | float64 |
| sim_goal_dist | (459, 2) | float64 |
| sim_osc_utime | (459,) | int64 |
| sim_render_step | (459,) | int64 |
| sim_belt_xyz | (459, 48, 3) | float32 |
| sim_pulley_large_pose | (459, 7) | float64 |
| sim_hand_mm | (459,) | float32 |
| sim_robotiq_byte | (459,) | uint8 |
| sim_grasp_ok | (459, 2) | bool |
| sim_episode_step | (459,) | int64 |
| sim_board_clearance_mm | (459,) | float32 |
| sim_franka_tip_clearance_mm | (459,) | float32 |
| sim_tracking_err_mm | (459, 2) | float64 |
| sim_action_knot1_minus_measured | (459, 12) | float64 |
| sim_realised_delta | (459, 12) | float64 |
| sim_cloud_h_mm | (459,) | float32 |
| sim_cloud_wrap_deg | (459,) | float32 |
| sim_cloud_r_mm | (459,) | float32 |
| sim_cloud_n_pts | (459,) | int32 |
| sim_cloud_success | (459,) | bool |
| sim_plan_age_s | (459,) | float64 |
| pcd_belt_seg | (459,) | object |
| sim_outcome | () | <U7 |
| sim_status | () | <U2 |
| sim_mode | () | <U7 |
| sim_start_state | () | <U129 |
| sim_wrap_deg | (459,) | float64 |
| sim_h_median_mm | (459,) | float64 |
| sim_slant_deg | () | float64 |
| sim_slant_axis_deg | () | float64 |
| sim_slant_dir | () | <U5 |
| sim_slant_deg_t | (459,) | float64 |
| sim_slant_axis_deg_t | (459,) | float64 |
| sim_min_board_clearance_mm | () | float32 |
| sim_board_contact | () | bool |
| sim_min_franka_tip_clearance_mm | () | float32 |
| sim_alignment | () | <U4 |
| sim_backend | () | <U3 |
| sim_osc_utime_offset_us | () | int64 |
| sim_wrap_small_deg | (459,) | float64 |
| sim_h_small_median_mm | (459,) | float64 |
| sim_outcome_large | () | <U7 |
| sim_outcome_small | () | <U7 |
| sim_success_both | () | bool |
| sim_chain_from | (2,) | int32 |
| sim_post_from | () | int32 |
| sim_mpc_end_outcome | () | <U7 |

## e2e_run/recordings/.../chunks/chunk_00000.npz

| key | shape | dtype |
|---|---|---|
| step | (2000,) | int64 |
| sim_time | (2000,) | float64 |
| wall_time | (2000,) | float64 |
| compute_ms | (2000,) | float32 |
| hand_target_mm | (2000,) | float32 |
| hand_stale | (2000,) | bool |
| franka_stale | (2000,) | bool |
| robotiq_cmd | (2000, 3) | uint8 |
| robotiq_cmd_valid | (2000,) | bool |
| robotiq_status | (2000, 3) | uint8 |
| robotiq_opening | (2000,) | float32 |
| lcm_rx | (2000,) | int32 |
| joint_q | (2000, 33) | float32 |
| joint_qd | (2000, 31) | float32 |
| efforts | (2000, 15) | float32 |
| state_step | (500,) | int64 |
| body_q | (500, 86, 7) | float32 |

## demo/episode_gv_07_0.npz

| key | shape | dtype |
|---|---|---|
| pcd | (53,) | object |
| pcd_rgb | (53,) | object |
| pcd_belt | (53, 150, 3) | float32 |
| pcd_kinematic | (53, 0, 3) | float32 |
| kinematic_model_names | (3,) | <U10 |
| kinematic_body_name_patterns | (8,) | <U22 |
| kinematic_sampled_body_names | (0,) | float64 |
| point_cloud_source | () | <U16 |
| record_kinematic_points | () | bool |
| state | (53, 40) | float64 |
| actions | (53, 12) | float64 |
| utime | (53,) | int64 |
| trajectory_label | () | <U7 |
| sim_step | (53,) | int64 |
| sim_time | (53,) | float64 |
| sim_meta | () | <U12298 |
| sim_ee_franka | (53, 7) | float64 |
| sim_ee_ur | (53, 7) | float64 |
| sim_cmd_knot0_franka | (53, 7) | float64 |
| sim_cmd_knot1_franka | (53, 7) | float64 |
| sim_cmd_knots_franka | (53, 7, 7) | float64 |
| sim_cmd_knot_times_franka | (53, 7) | float64 |
| sim_cmd_n_knots | (53,) | int16 |
| sim_cmd_traj_utime | (53,) | int64 |
| sim_cmd_ur_t | (53, 7) | float64 |
| sim_cmd_ur_t1 | (53, 7) | float64 |
| sim_cmd_ur_line | (53, 16) | float64 |
| sim_cmd_ur_fresh_exact | (53,) | int8 |
| sim_ur_line_own_window_delta | (53, 6) | float64 |
| sim_cmd_source | (53,) | int8 |
| sim_cmd_ctrl_hand_mm | (53,) | float32 |
| sim_cmd_ctrl_robotiq_byte | (53,) | int16 |
| sim_phase | (53,) | int16 |
| sim_stage | (53,) | int16 |
| sim_latent | (53, 16) | float64 |
| sim_latent_pred | (53, 16) | float64 |
| sim_goal_dist | (53, 2) | float64 |
| sim_osc_utime | (53,) | int64 |
| sim_render_step | (53,) | int64 |
| sim_belt_xyz | (53, 48, 3) | float32 |
| sim_pulley_large_pose | (53, 7) | float64 |
| sim_hand_mm | (53,) | float32 |
| sim_robotiq_byte | (53,) | uint8 |
| sim_grasp_ok | (53, 2) | bool |
| sim_episode_step | (53,) | int64 |
| sim_board_clearance_mm | (53,) | float32 |
| sim_franka_tip_clearance_mm | (53,) | float32 |
| sim_tracking_err_mm | (53, 2) | float64 |
| sim_action_knot1_minus_measured | (53, 12) | float64 |
| sim_realised_delta | (53, 12) | float64 |
| sim_cloud_h_mm | (53,) | float32 |
| sim_cloud_wrap_deg | (53,) | float32 |
| sim_cloud_r_mm | (53,) | float32 |
| sim_cloud_n_pts | (53,) | int32 |
| sim_cloud_success | (53,) | bool |
| sim_plan_age_s | (53,) | float64 |
| pcd_belt_seg | (53,) | object |
| sim_outcome | () | <U7 |
| sim_status | () | <U2 |
| sim_mode | () | <U8 |
| sim_start_state | () | <U1538 |
| sim_wrap_deg | (53,) | float64 |
| sim_h_median_mm | (53,) | float64 |
| sim_slant_deg | () | float64 |
| sim_slant_axis_deg | () | float64 |
| sim_slant_dir | () | <U5 |
| sim_slant_deg_t | (53,) | float64 |
| sim_slant_axis_deg_t | (53,) | float64 |
| sim_min_board_clearance_mm | () | float32 |
| sim_board_contact | () | bool |
| sim_min_franka_tip_clearance_mm | () | float32 |
| sim_alignment | () | <U623 |
| sim_backend | () | <U3 |
| sim_osc_utime_offset_us | () | int64 |
| sim_wrap_small_deg | (53,) | float64 |
| sim_h_small_median_mm | (53,) | float64 |
| sim_outcome_large | () | <U7 |
| sim_outcome_small | () | <U7 |
| sim_success_both | () | bool |
| sim_post_from | () | int32 |
| sim_mpc_end_outcome | () | <U7 |

## pre_place_1.npz (whole copy)

| key | shape | dtype |
|---|---|---|
| joint_q | (232,) | float32 |
| joint_qd | (229,) | float32 |
| body_q | (86, 7) | float32 |
| body_qd | (86, 6) | float32 |
| joint_target_q | (232,) | float32 |
| joint_f | (229,) | float32 |
| hand_target_q | (2,) | float64 |
| hand_goal_q | (2,) | float64 |
| hand_ramp_q | (2,) | float64 |
| step_index | () | int64 |
| sim_time | () | float64 |
| hand_target_mm | () | float64 |
| robotiq_position_byte | () | int64 |
| robotiq_force_byte | () | int64 |
| belt_placed | () | bool |
| belt_anchor_body | () | int64 |
| meta_json | () | <U7898 |

## miles/miles/*.npz (3 whole copies, 2 frames each)

| key | shape | dtype |
|---|---|---|
| pcd | (2,) | object |
| pcd_rgb | (2,) | object |
| pcd_belt | (2, 150, 3) | float32 |
| pcd_kinematic | (2, 0, 3) | float32 |
| kinematic_model_names | (3,) | <U10 |
| kinematic_body_name_patterns | (8,) | <U22 |
| kinematic_sampled_body_names | (0,) | float64 |
| point_cloud_source | () | <U16 |
| record_kinematic_points | () | bool |
| state | (2, 40) | float64 |
| actions | (2, 12) | float64 |
| utime | (2,) | int64 |
| trajectory_label | () | <U7 |
| sim_step | (2,) | int64 |
| sim_time | (2,) | float64 |
| sim_meta | () | <U2445 |
| sim_ee_franka | (2, 7) | float64 |
| sim_ee_ur | (2, 7) | float64 |
| sim_cmd_knot0_franka | (2, 7) | float64 |
| sim_cmd_knot1_franka | (2, 7) | float64 |
| sim_cmd_knots_franka | (2, 7, 7) | float64 |
| sim_cmd_knot_times_franka | (2, 7) | float64 |
| sim_cmd_n_knots | (2,) | int16 |
| sim_cmd_traj_utime | (2,) | int64 |
| sim_cmd_ur_t | (2, 7) | float64 |
| sim_cmd_ur_t1 | (2, 7) | float64 |
| sim_cmd_ur_line | (2, 16) | float64 |
| sim_cmd_ur_fresh_exact | (2,) | int8 |
| sim_ur_line_own_window_delta | (2, 6) | float64 |
| sim_cmd_source | (2,) | int8 |
| sim_cmd_ctrl_hand_mm | (2,) | float32 |
| sim_cmd_ctrl_robotiq_byte | (2,) | int16 |
| sim_phase | (2,) | int16 |
| sim_stage | (2,) | int16 |
| sim_latent | (2, 16) | float64 |
| sim_latent_pred | (2, 16) | float64 |
| sim_goal_dist | (2, 1) | float64 |
| sim_osc_utime | (2,) | int64 |
| sim_render_step | (2,) | int64 |
| sim_belt_xyz | (2, 48, 3) | float32 |
| sim_pulley_large_pose | (2, 7) | float64 |
| sim_hand_mm | (2,) | float32 |
| sim_robotiq_byte | (2,) | uint8 |
| sim_grasp_ok | (2, 2) | bool |
| sim_episode_step | (2,) | int64 |
| sim_board_clearance_mm | (2,) | float32 |
| sim_franka_tip_clearance_mm | (2,) | float32 |
| sim_tracking_err_mm | (2, 2) | float64 |
| sim_action_knot1_minus_measured | (2, 12) | float64 |
| sim_realised_delta | (2, 12) | float64 |
| sim_cloud_h_mm | (2,) | float32 |
| sim_cloud_wrap_deg | (2,) | float32 |
| sim_cloud_r_mm | (2,) | float32 |
| sim_cloud_n_pts | (2,) | int32 |
| sim_cloud_success | (2,) | bool |
| pcd_belt_seg | (2,) | object |
| sim_outcome | () | <U7 |
| sim_status | () | <U2 |
| sim_mode | () | <U5 |
| sim_start_state | () | <U97 |
| sim_wrap_deg | (2,) | float64 |
| sim_h_median_mm | (2,) | float64 |
| sim_slant_deg | () | float64 |
| sim_slant_axis_deg | () | float64 |
| sim_slant_dir | () | <U11 |
| sim_slant_deg_t | (2,) | float64 |
| sim_slant_axis_deg_t | (2,) | float64 |
| sim_min_board_clearance_mm | () | float32 |
| sim_board_contact | () | bool |
| sim_min_franka_tip_clearance_mm | () | float32 |
| sim_backend | () | <U3 |
| sim_osc_utime_offset_us | () | int64 |
| sim_miles_demo_frame | (2,) | int16 |
| sim_miles | () | <U200 |
| miles_segment | (2,) | <U4 |

## Episode npz key presence (e2e run; demo identical except no sim_chain_from)

| asked name | present | actual name |
|---|---|---|
| sim_ee_franka | yes | sim_ee_franka |
| sim_ee_ur | yes | sim_ee_ur |
| sim_belt_xyz | yes | sim_belt_xyz |
| pc | renamed | pcd_belt (+ pcd: per-frame object clouds) |
| state | yes | state |
| u | NO | - |
| actions | yes | actions |
| sim_wrap_deg | yes | sim_wrap_deg |
| sim_wrap_small_deg | yes | sim_wrap_small_deg |
| sim_grasp_ok | yes | sim_grasp_ok |
| sim_phase | yes | sim_phase |
| sim_cmd_source | yes | sim_cmd_source |
| ctrl_hand_mm | renamed | sim_cmd_ctrl_hand_mm |
| time | renamed | sim_time |
| utime | yes | utime |
| sim_chain_from | yes | sim_chain_from |

## meta.json

- franka_hand.coords: `[9, 10]`
- driver_coords: `[17, 21]`
- effort_layout: `['franka/panda_motor1', 'franka/panda_motor2', 'franka/panda_motor3', 'franka/panda_motor4', 'franka/panda_motor5', 'franka/panda_motor6', 'franka/panda_motor7', 'franka_hand/panda_finger_motor1', 'franka_hand/panda_finger_motor2', 'ur10/shoulder_pan_joint_actuator', 'ur10/shoulder_lift_joint_actuator', 'ur10/elbow_joint_actuator', 'ur10/wrist_1_joint_actuator', 'ur10/wrist_2_joint_actuator', 'ur10/wrist_3_joint_actuator']`
- chunks entry still lists steps=2000/state_frames=500; fixture chunk is cut to 40

## index.json

- episodes[0] keys: `['file', 'start_state', 'repeat', 'mode', 'outcome', 'status', 'reason', 'duration_s', 'steps', 'frames', 'controller_events', 'stage_times', 'stage_ends', 'latent', 'alignment', 'tracking_rms_mm', 'tracking_max_mm', 'final_wrap_deg', 'final_h_median_mm', 'outcome_large', 'outcome_small', 'success_both', 'final_wrap_small_deg', 'final_h_small_mm', 'final_slant_deg', 'final_slant_axis_deg', 'final_slant_dir', 'min_board_clearance_mm', 'board_contact', 'min_franka_tip_clearance_mm', 'grasp_ok_final', 'cmd_source_counts', 'controller_gripper_commands', 'timing_s', 'osc_stats', 'controller_log', 'controller_exit', 'recording', 'size_bytes', 'end_state', 'mppi', 'post_mpc', 'chain', 'mppi_stage2', 'events_stage2']`
- phase_labels: `['mpc:stage0', 'mpc:stage1', 'mpc:stage2', 'mpc:stage3', 'mpc:stage4', 'mpc:stage5', 'done', 'ur:open', 'ur:lift', 'ur:park', 'ur:settle', 's2:stage0', 's2:stage1', 's2:stage2', 's2:stage3', 's2:stage4', 's2:stage5', 's2:stage6', 's2:stage7', 's2:stage8', 's2:stage9', 's2:stage10', 's2:stage11', 's2:stage12', 's2:stage13', 's2:stage14', 's2:stage15', 's2:stage16', 's2:stage17', 's2:stage18', 's2:done', 'post:release', 'post:out', 'post:lift', 'post:settle']`

## train.log `[epoch N]` k=v keys

`train/recon`, `train/aux`, `train/vio`, `train/lcp`, `train/dyn`, `train/next`, `train/hold`, `train/total`, `train/recon_rms_mm`, `train/lr_ae`, `train/mode_delta_rel`, `val/recon_rms_mm`, `val/recon_chamfer_mm`, `val/vio`, `val/dyn`, `val/hold`, `val/aux_mae_h_mm`, `val/aux_mae_wrap_deg`, `train/rho_A`, `train/B_fro`, `val/next_rms_mm`, `train/ms_skip`, `train/ms`, `train/ms_rec`, `val/ms`, `val/ms_rec`, `val/ms_chamfer_mm_k1`, `val/ms_chamfer_mm_k7`

## Not copied (paths + sizes)

- `/home/hienbui/git/magna_belt_simulation-main/data/lcs/realfirst/cmp/encoder_study/sim/V63j/deploy_v39j/deploy.npz` (796512 bytes)
- `/home/hienbui/git/magna_belt_simulation-main/data/lcs/realfirst/cmp/encoder_study/sim/V63j/decoder_decfit/decoder.npz` (4875056 bytes)
