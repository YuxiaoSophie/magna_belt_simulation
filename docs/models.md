# Learned-LCS models and closed-loop evaluations: registry

One place to look up every trained latent-LCS model and every closed-loop evaluation of the
learned MPC. The machine-readable twin is `docs/models.yaml` (it lives in `docs/` because
`data/` is gitignored); it has the full per-model and per-eval fields (config, checkpoint,
flags, exports, sources). Day-by-day context: `docs/learned-mpc.md`; commands:
`docs/learned-mpc-reference.md` §5. Numbers here are copied from existing tables, each with its
source; nothing is recomputed. Folder names are unchanged: old names stay valid as aliases.

## 1. Naming scheme

**Model ID = `V<campaign>[letter]`.** The campaign is the lcs_learning run folder
`outputs/sim_belt_v<N>_<date>/`; letters a, b, c… are variants within a campaign; no letter if
the campaign has a single model. Exceptions and mappings:

- **V1** = `sim_belt_ablation_20260924/`, the first campaign (no `v1` in its folder name).
  Letters follow the config names: a `both`, b `decoded_only`, c `violation_only`.
- **V2** (no letter) is the deployed `v2_decoded_only`, kept letterless because it is the
  deployed model and the name everyone uses; `V2a` is an accepted alias. The campaign's other
  model is V2b (`v2_multistep`).
- **V6** (no letter) is the unbounded T6 run, stopped at epoch 25; V6a / V6b are its bounded
  variants (as agreed).
- **Old names V1 / V2 of 2026-09-30** (the state-dependent-B models) are now **V7a / V7b**.
  "V1"/"V2" in the 2026-09-30 day log and in lcs_learning's v7 files mean V7a/V7b.
- **Runs** (as opposed to models) are `V<campaign>-<nn>`, indexed in `docs/realfirst-campaign.md`:
  V11-01 / V11-02 = the real-first loops in `data/lcs/realfirst/v1`, `v1_1` (models V11a-i);
  V12-01… = the 2026-10-06 campaign (models V12a-u, V13a). Lowercase "v1" elsewhere in this
  file means campaign V1's insertion data.

**Descriptive tag = `<data>-<objective>-<dynamics>`:**

| field | token | meaning |
|---|---|---|
| data | `ins1` | v1 insertion data (`20260923-210508-ep300-ou`, old `knot1 − measured` actions); V1 only |
| | `ins` | v2 insertion data (`data/lcs/v2/split.json`, 464 train eps, `cmd_delta` actions) |
| | `mix` | the v3 union: insertion + free space + approach + contact (`data/lcs/v3/split.json`) |
| objective | `mse` / `rmse` | decoded next-state loss form (`mse` = m², effectively off; `rmse` = the fix) |
| | `vg` | LCS violation gradient reaches the encoder (V1a, V1c; V1c has no decoded loss) |
| | `ms7` | multistep H7 loss |
| | `db` | Δbelt loss |
| | `np3` | near-pulley sampling ×3 |
| | `lm` | latent-metric loss |
| | `nceS` / `nceA` | state / action InfoNCE (σ-whitened, τ 0.1) |
| | `cfm` / `cfm-band` / `cfm-pin` | CFM-form state InfoNCE (σ 1, τ 1): unbounded / latent-std band / pin |
| | `gd` | graph-distance loss: ‖z_s − z_l‖ ≈ obstacle-aware path length / 50 mm (V10) |
| | `rec<w>` | joint belt reconstruction at weight w (V10; the V2-V8 recon is implicit) |
| | `aug30` | random ±30° rotation about the pulley axis on every loss (V10) |
| | `hold<w>` | joint-LCS hold term (u = 0 keeps z) at weight w, inside encoder training (V10) |
| | `2blk` | two-block latent: 8 task dims (graph loss) + 8 shape dims (recon, cost κ) (V10) |
| dynamics | `fixB` / `Bz` | fixed B / state-dependent B(z) = B0 + ΔB(z) |
| encoder | (V10 only, not a tag field) | belt-only MLP encoder (`belt_mlp`): no proprio, no camera cloud |
| | `+frz` | trained with the head frozen at z_0 over the multistep window (deploy semantics) |

Fine-tuning is recorded separately ("ft from V2"). `ins1` and `vg` are additions to the agreed
token list (§6).

**Evaluation label = `<ID> · <suite> · <cost> · <limits>`** (`BL` = the waypoint baseline):

| field | token | meaning |
|---|---|---|
| suite | `seat6` | flat_engaged × {nominal, gv_01, gv_05}, urtip × ud+3_r0, urtip6 × ud+6_r0, urtip6_high × ud+6_r0 |
| | `yaw12` / `yaw12-m15` | the seat6 starts, targets yawed ±15°; `-m15` = the −15° half only |
| | `fs6` | single-stage free-space goals (twist, bend, bend_lift, stretch, wrist_tilt, random_mix) |
| | `press48` | press-down set, 12 snapshots × 4 Franka-only presses (open loop, no model) |
| | `nom` | nominal start `pre_place_1_osc`, repeated |
| | `gv1` / `gv3` | grasp-variant set1 (13 starts) / set3 (8 starts), 2026-09-24/25 |
| | `gv1-2` / `gv1-10` / `gv1-d10` | set1 subsets: screen gv_00+gv_05 / tuning gv_00-09 / diagnosis gv_01-10 |
| | `stages10` | the 10 matched starts of 2026-09-25 (nominal + 8 of set1 + 1 of set3) |
| | `rec6` / `rec4` | recorded replay sets (nominal + 3/2 of set1 + 2/1 of set3) |
| | `eeoff4` / `long25` | EE-offset starts (4) / set1 gv_03 as one 35 s episode |
| | `seat6-1goal30` | seat6 starts, seated goal only (stage 1 0.075 s), stage 2 30 s |
| | `seat6-rf` | seat6 starts, the seed demo's two goals (pre_place_2 4 s, seated 30 s), MPPI node, belt-seg cloud real_like (V11) |
| | `seat6-rf11` | seat6 starts, MPPI two stages on the latent (demo-spread tol, streak 3, 10 s / 30 s), clearance, belt-seg clean cloud |
| | `seat6-rf12` | seat6 starts, MPPI on the latent, no prefix (V12 campaign cells: eval is always the full task), clearance, belt-seg clean cloud |
| | `hard11x2-rf` | hard suite: 11 starts x 2 (run_hard_suite.SUITE resets), noise 0, MPPI on the latent, stage goals on waypoints, clearance, belt-seg clean cloud |
| | `under12` | 12 belt-under-pulley snapshots (`data/lcs/contact/snapshots{,_heldout}`), seated goal only, 30 s |
| | `cell-s30` | one seat6 cell re-run with the stage-2 timeout 10 → 30 s |
| cost | `Qw` | whitened latent cost (default) |
| | `QA` / `QB` | metric A (learned global Q) / metric B (decoder pullback) |
| | `Qi` | isotropic Q = w_q·I/σ² (V10; one σ for every dim, two-block: shape dims × κ) |
| | `Qw+wp` | Qw + the `demo_traj.w_p` EE-path cost (2026-09-25; rejected design) |
| limits | `ub-ins` | `u` bounds from insertion data only (for V1b: its own v1 data) |
| | `ub-pool` | `u` bounds pooled over the v3 union (V3a's export bounds) |
| | `ub-hand` / `ub-hand-x0.5` | V1b demo export with 4 UR dims hand-set / × `u_bound_scale` 0.5 |
| | `admmN` | `admm_iter` N, written when not the default 2 (or in the E4 sweep) |

Other controller differences (reference mode, `w_r`, stage tolerances, head off) go in the
note column.

**Future runs.**

- New training campaign: `outputs/sim_belt_v<N>_<date>/` with N the next number (V11 next; V9
  is reserved by `docs/learned-lcs-improvement-plan.md`), one
  config/checkpoint/export per variant letter: `v<N><letter>.yaml`, `ckpt_v<N><letter>/`,
  `deploy_v<N><letter>/`, W&B run name `v<N><letter>`.
- New closed-loop eval: `data/lcs/mpc_eval/<date>_<ID>_<suite>_<cost>_<limits>/`, e.g.
  `20261001_V7a_seat6_QA_ub-ins/`; one run per label; write `table.md` in it.
- Add both to `docs/models.yaml` and to the tables below in the same change.

## 2. Models

| ID | tag | old names | status | parent | what changed vs parent |
|---|---|---|---|---|---|
| V1a | `ins1-mse+vg-fixB` | both | rejected | - | v1 ablation arm: decoded loss + violation gradient to the encoder |
| V1b | `ins1-mse-fixB` | decoded_only, decoded_only_epoch0300 | rejected (deployed 09-24, replaced by V2) | - | v1 ablation arm: decoded loss only |
| V1c | `ins1-vg-fixB` | violation_only | rejected | - | v1 ablation arm: violation gradient only, no decoded loss |
| **V2** | `ins-mse-fixB` | v2, v2_decoded_only, V2a | **deployed** | V1b | re-collected insertion data (`cmd_delta`, pre-hold, UR excitation) |
| V2b | `ins-mse+ms7-fixB` | v2_multistep | rejected | V2 | V2 + multistep H7 (mse); AE collapse at epoch 11, failed its pick rule |
| V3a | `mix-mse-fixB` | v3_mix, PICK A, "model A" | rejected | V2 | V2 recipe from scratch on the v3 union |
| V3b | `mix-mse-fixB` (ft from V2) | v3_ft, B | rejected | V2 | V2 epoch 300 fine-tuned on the v3 union, 120 epochs, lr 3e-4 |
| V4a | `mix-rmse-fixB` | T1 | rejected (base recipe of V5-V7) | V3a | decoded loss in rmse form (the loss-scale fix, now the default) |
| V4b | `mix-rmse+db+ms7+np3-fixB` | T2 | rejected | V4a | + Δbelt, multistep H7 (rmse), near-pulley ×3 |
| V4c | `mix-rmse+db+ms7+np3+lm-fixB` | T3 | rejected | V4b | + latent-metric loss 0.02 |
| V5a | `mix-rmse+nceS-fixB` | T5a | rejected | V4a | + state InfoNCE (τ 0.1, hard negatives, batch σ) |
| V5b | `mix-rmse+nceS+nceA-fixB` | T5b | rejected | V5a | + action InfoNCE |
| V6 | `mix-rmse+cfm-fixB` | T6 | stopped (epoch 25) | V4a | + CFM-form InfoNCE, unbounded latent scale |
| V6a | `mix-rmse+cfm-band-fixB` | T6a, t6a_w1 | rejected | V6 | + latent-std band 0.18-0.35, weight 1 |
| V6b | `mix-rmse+cfm-pin-fixB` | T6b, t6b_w10_r2 | rejected | V6 | + latent-std pin 0.35, weight 10 (retrain r2) |
| V7a | `mix-rmse-Bz` | V1 (09-30), sdv1 | **candidate** | V4a | + state-dependent B(z), H 64, zero-init head |
| V7b | `mix-rmse+ms7-Bz+frz` | V2 (09-30), sdv2 | rejected | V7a | + multistep H7 (rmse) with B(z_0) frozen; inflates B, ρ(A) 1.264 |
| V8a | `mix-rmse-fixB (ft-onpolicy 6, tr7)` | - | candidate | V4a | on-policy fine-tune iteration 6 (primary, step 3000), box ub-tr7 |
| V8b | `mix-rmse-fixB (ft-onpolicy 7, tr8)` | - | candidate | V8a | on-policy fine-tune iteration 7 (primary, step 2500), box ub-tr8 |
| V8c | `mix-rmse-fixB (ft-onpolicy 8, tr9)` | - | candidate | V8b | on-policy fine-tune iteration 8 (primary, step 2500), box ub-tr9 |
| V8d | `mix-rmse-fixB (ft-onpolicy 9, tr10)` | - | candidate | V8c | on-policy fine-tune iteration 9 (primary, step 3000), box ub-tr10 |
| V10a | `mix-gd+vg-fixB` | graphemb_dyn1.0 | **candidate** (not reproducible) | - | belt-only MLP encoder, graph-distance loss + joint LCS (weight 1); only V10 that seats |
| V10b | `mix-gd+vg+rec0.03+aug30+hold1-fixB` | graphemb_aug_rec0.03 | rejected | V10a | + recon 0.03, aug30, joint hold 1; LCS refit with hold 10 |
| V10c | same as V10b | lever_b_nohold | rejected | V10b | V10b encoder, LCS refit without hold |
| V10d | same as V10b | - | rejected | V10b | V10b encoder, LCS refit with hold 1 |
| V10e | `mix-gd+vg+aug30-fixB` | graphemb_abl_aug | rejected | V10a | + aug30 only (ablation) |
| V10f | `mix-gd+vg+rec0.03-fixB` | graphemb_abl_rec | rejected | V10a | + recon 0.03 only (ablation) |
| V10g | `mix-gd+vg+hold1-fixB` | graphemb_abl_hold | rejected | V10a | + joint hold 1 only (ablation); ρ(A) 1.076 |
| V10h | `mix-gd+vg+rec0.03+2blk-fixB` | graphemb_twoblock | rejected | V10a | two-block 8 + 8, recon 0.03, κ 0.25 |
| V10i | `mix-gd+vg-fixB` (seed 1) | graphemb_dyn1.0_s1 | rejected | V10a | seed control: V10a recipe, seed 1 |
| V10j | `mix-gd+vg+rec0.03+aug30+2blk-fixB` | graphemb_twoblock_aug | rejected | V10h | + aug30 |
| V10k | `mix-gd+vg-fixB` (rerun) | graphemb_rs_rec0 | rejected | V10a | V10a recipe and seed re-run (current code; GPU nondeterminism) |
| V10l | `mix-gd+vg+rec0.001-fixB` | graphemb_rs_rec0.001 | rejected | V10a | + recon 0.001 (from scratch, seed 0) |
| V10m | `mix-gd+vg+rec0.003-fixB` | graphemb_rs_rec0.003 | rejected (closest recon variant) | V10a | + recon 0.003 (from scratch, seed 0) |
| V10n | `mix-gd+vg+rec0.01-fixB` | graphemb_rs_rec0.01 | rejected | V10a | + recon 0.01 (from scratch, seed 0) |
| V11a | `rf-rmse+vg+aux-fixB` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V11b | `rf-rmse+vg+aux-fixB` | - | candidate | V11a | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |
| V11c | `rf-rmse+vg+aux-fixB` | - | candidate | V11b | real-first loop iteration 2: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr2 |
| V11d | `rf-sk10+vg+aux-fixB` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V11e | `rf-sk10+vg+aux-fixB` | - | candidate | V11d | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |
| V11f | `rf-sk10+vg+aux-fixB` | - | candidate | V11e | real-first loop iteration 2: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr2 |
| V11g | `rf-sk10+vg+aux-fixB` | - | candidate | V11f | real-first loop iteration 3: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr3 |
| V11h | `rf-sk10+vg+aux-fixB` | - | candidate | V11g | real-first loop iteration 4: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr4 |
| V11i | `rf-sk10+vg+aux-fixB` | - | candidate | V11h | real-first loop iteration 5: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr5 |
| V12a | `rf-sk10+vg+aux-fixB-C2_prefix` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V12b | `rf-sk10+vg+aux-fixB-C2_prefix` | - | candidate | V12a | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |
| V12c | `rf-sk10+vg+aux-fixB-C2_prefix` | - | candidate | V12b | real-first loop iteration 2: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr2 |
| V12d | `rf-sk10+vg+aux-fixB-C3_both` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V12g | `rf-sk10+vg+aux-fixB-C1_transport` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V12h | `rf-sk10+vg+aux-fixB-C1_transport` | - | candidate | V12g | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |
| V12j | `rf-sk10+vg+aux-fixB-full` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V12k | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12j | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |
| V12l | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12k | real-first loop iteration 2: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr2 |
| V12m | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12l | real-first loop iteration 3: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr3 |
| V12n | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12m | real-first loop iteration 4: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr4 |
| V12o | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12n | real-first loop iteration 5: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr5 |
| V12p | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12o | real-first loop iteration 6: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr6 |
| V12q | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12p | real-first loop iteration 7: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr7 |
| V12r | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12q | real-first loop iteration 8: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr8 |
| V12s | `rf-sk10+vg+aux-fixB-full` | - | candidate | V12r | real-first loop iteration 9: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr9 |
| V13a | `rf-sk10+vg+aux-fixB-full2_lcsonly` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V12t | `rf-sk10+vg+aux-fixB-full2_gate` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V12u | `rf-sk10+vg+aux-fixB-full2_gate` | - | candidate | V12t | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |
| V12v | `rf-sk10+vg+aux-fixB-v12_18` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V12w | `rf-sk10+vg+aux-fixB-v12_18` | - | candidate | V12v | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |
| V12x | `rf-sk10+vg+aux-fixB-v12_18` | - | candidate | V12w | real-first loop iteration 2: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr2 |
| V14a | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V15a | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V15b | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V15a | real-first loop iteration 2: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr2 |
| V15c | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V15b | real-first loop iteration 3: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr3 |
| V15d | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V15c | real-first loop iteration 4: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr4 |
| V16a | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V16b | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16a | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |
| V16c | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16b | real-first loop iteration 2: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr2 |
| V16d | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16c | real-first loop iteration 3: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr3 |
| V16e | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16d | real-first loop iteration 4: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr4 |
| V16f | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16e | real-first loop iteration 5: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr5 |
| V16g | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16f | real-first loop iteration 6: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr6 |
| V16h | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16g | real-first loop iteration 7: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr7 |
| V16i | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16h | real-first loop iteration 8: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr8 |
| V16j | `rf-sk10+vg+aux-fixB-v12_27` | - | candidate | V16i | real-first loop iteration 9: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr9 |
| V17a | `rf-sk10-t1-e0` | - | candidate | - | V12-42 T1 (3 demos + V12-29b MILES) fresh joint, today's recipe, seed 4; hard suite 8/22 (V12-43), 7/22 with progress start (V12-44) |
| V17b | `rf-sk10-t1-aug-act0.1` | - | rejected | V17a | + point augmentation + Delta-JEPA action head λ 0.1 (train-only), seed 4; off-demo −15 % but hard suite 1/22 (V12-43, +y bias) |
| V17c | `rf-sk10-t1-aug` | - | rejected | V17a | + point augmentation only, seed 4; hard suite 0/22 (V12-44, +x bias) |
| V17d | `rf-sk10-t1-e0` | - | candidate | V17a | V17a's recipe, seed 6; hard suite **11/22** with progress start (V12-45); seed of V12-46 (= V18a) |
| V17e | `rf-sk10-t1-e0` | - | candidate | V17a | V17a's recipe, seed 5; hard suite 9/22 with progress start (V12-45) |
| V18a | `rf-sk10-t1e0s6-v12_46` | - | candidate | - | V12-46 iteration 0: = V17d (adopted via `seed_model`, no refit); eval 8/16 |
| V18b | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18a | V12-46 iteration 1: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to set2 gv_00; eval 11/16; hard suite 10/22 (V12-47) |
| V18c | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18b | V12-46 iteration 2: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to set2 gv_06 |
| V18d | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18c | V12-46 iteration 3: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to set3 gv_02; eval 11/16 |
| V18e | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18d | V12-46 iteration 4: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to set3 gv_03 |
| V18f | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18e | V12-46 iteration 5: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to UR −30 mm; eval 5/16 |
| V18g | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18f | V12-46 iteration 6: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to UR +15 mm |
| V18h | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18g | V12-46 iteration 7: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to grasp −30°; eval 10/16 |
| V18i | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18h | V12-46 iteration 8: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to grasp −20° |
| V18j | `rf-sk10-t1e0s6-v12_46` | - | candidate | V18i | V12-46 iteration 9: LCS-only refit from V17d's checkpoint (`refit.from_seed`) on the anchored buffer + on-policy rollouts up to grasp +20°; eval 5/16 |
| V22a | `rf-jepa+graph-v12_52` | - | candidate | - | real-first loop iteration 0: belt-cloud encoder + LCS, joint fresh refit on the buffer, box ub-tr0 |
| V22b | `rf-jepa+graph-v12_52` | - | candidate | V22a | real-first loop iteration 1: belt-cloud encoder + LCS, joint warm refit on the buffer, box ub-tr1 |

Side runs without an ID (`models.yaml` lists them under V6a/V6b): `t6a_w10` (`trafqf0l`) and
`t6b_w1` (`6n5j9jvb`) stopped at epoch 30; `t6b_w10` (`q25alk7k`) segfaulted at epoch 216 and
was retrained as V6b. Smoke, identity, debug and τ-sweep checkpoints (`smoke/`, `identity*/`,
`sim_belt_v5_20260929/{debug,sweep}/`, V7 `smoke/deploy_smoke_*`) are not models.

**V10 is a prototype, not an lcs_learning trainer run.** Encoders: `embed.py` (graph-distance
loss + joint LCS, optional recon/hold/aug/two-block), 40k steps, in
`data/lcs/diag/20261001-temporal-latent/scripts/`; the deployed LCS is refit on the frozen
latent by `lcs_fit.py` (the trainer's `lcs.loss`, ε 0.01, n_λ 8, stiffness 0.1, 25 PGD
iterations; whitened, converted to raw units) and exported by `export_v10.py`. Side encoders
without an ID (`runs/`: `graphemb`, `graphemb_dyn0.01..0.3`, `graphemb_rec0.003..0.03`,
`graphemb_aug_rec0.003/0.01`, `hilp_*`, `stab*`, `exp0*`) were offline only.

**Provenance** (lcs_learning `outputs/`; checkpoint =
`<run>/<ckpt>/<date>/<W&B>/checkpoint_epoch_<E>.pt`; `u` bounds as the export's):

| ID | run folder | config | ckpt dir | W&B | epoch | export dir(s) | `u` bounds |
|---|---|---|---|---|---|---|---|
| V1a | `sim_belt_ablation_20260924` | `both.yaml` | `ckpt_both` | `5r0a403t` | 300 | - | ub-ins (v1) |
| V1b | same | `decoded_only.yaml` | `ckpt_decoded_only` | `5afcg1zb` | 300 | `ckpt_decoded_only/deploy`, `…/deploy_demo` | ub-ins (v1); ub-hand in tuning |
| V1c | same | `violation_only.yaml` | `ckpt_violation_only` | `ylcjn8b9` | 300 | - | ub-ins (v1) |
| V2 | `sim_belt_v2_20260925` | `v2_decoded_only.yaml` | `ckpt_v2_decoded_only` | `nsxihz32` | 300 | `deploy_v2_decoded_only`, `deploy_v2_flat`, `deploy_v2_flat_pp2` (live) | ub-ins |
| V2b | same | `v2_multistep.yaml` | `ckpt_v2_multistep` | `5js34zl7` | 300 | `deploy_v2_multistep_cuda_REJECTED_E2` | ub-ins |
| V3a | `sim_belt_v3_20260928` | `v3_mix.yaml` | `ckpt_v3_mix` | `rt2j3k45` | 300 | `deploy_v3_mix` | ub-pool |
| V3b | same | `v3_ft.yaml` | `ckpt_v3_ft` | `3y5b4zac` | 120 | `deploy_v3_ft` | ub-pool |
| V4a | `sim_belt_v4_20260929` | `v4_t1.yaml` | `ckpt_t1` | `1niukyhl` | 300 | `deploy_t1` | ub-ins |
| V4b | same | `v4_t2.yaml` | `ckpt_t2` | `gwz0d9jk` | 300 | `deploy_t2` | ub-ins |
| V4c | same | `v4_t3.yaml` | `ckpt_t3` | `1q2gt22n` | 300 | `deploy_t3` | ub-ins |
| V5a | `sim_belt_v5_20260929` | `v5_t5a.yaml` | `ckpt_t5a` | `a4liiwsl` | 300 | `deploy_t5a` (+ `superseded/deploy_t5a`) | ub-ins |
| V5b | same | `v5_t5b.yaml` | `ckpt_t5b` | `mkgsga96` | 300 | `deploy_t5b` | ub-ins |
| V6 | `sim_belt_v6_20260929` | `v6_t6.yaml` | `ckpt_t6` | `29na8zty` | 20 (last saved) | - | - |
| V6a | same | `v6_t6a_w1.yaml` | `ckpt_t6a_w1` | `04ium1ly` | 300 | `deploy_t6a` | ub-ins |
| V6b | same | `v6_t6b_w10_r2.yaml` | `ckpt_t6b_w10_r2` | `w3cqrq86` | 300 | `deploy_t6b` | ub-ins |
| V7a | `sim_belt_v7_20260930` | `v7_v1.yaml` | `ckpt_v1` | `grn9fisu` | 300 | `deploy_v1` | ub-ins |
| V7b | same | `v7_v2.yaml` | `ckpt_v2` | `vo97r4cg` | 300 | `deploy_v2` | ub-ins |
| V8a | `sim_belt_v8_20261001` | `finetune_lcs_latent.py` | `iter6/ft_primary` | - | step 3000 | `deploy_v8a` | ub-tr7 |
| V8b | `sim_belt_v8_20261001` | `finetune_lcs_latent.py` | `iter7/ft_primary` | - | step 2500 | `deploy_v8b` | ub-tr8 |
| V8c | `sim_belt_v8_20261001` | `finetune_lcs_latent.py` | `iter8/ft_primary` | - | step 2500 | `deploy_v8c` | ub-tr9 |
| V8d | `sim_belt_v8_20261001` | `finetune_lcs_latent.py` | `iter9/ft_primary` | - | step 3000 | `deploy_v8d` | ub-tr10 |
| V10a | `sim_belt_v10_20261001` | `embed.py --dyn-weight 1.0` | diag `runs/graphemb_dyn1.0` | - | 40k steps | `deploy_v10a` | ub-ins |
| V10b | same | `--dyn-weight 1 --hold-weight 1 --recon-weight 0.03 --rot-aug 30` | `runs/graphemb_aug_rec0.03` | - | 40k | `deploy_v10b` (LCS hold 10) | ub-ins |
| V10c | same | as V10b | `runs/graphemb_aug_rec0.03` | - | 40k | `deploy_v10c` (LCS no hold) | ub-ins |
| V10d | same | as V10b | `runs/graphemb_aug_rec0.03` | - | 40k | `deploy_v10d` (LCS hold 1) | ub-ins |
| V10e | same | `--dyn-weight 1 --rot-aug 30` | `runs/graphemb_abl_aug` | - | 40k | `deploy_v10e` | ub-ins |
| V10f | same | `--dyn-weight 1 --recon-weight 0.03` | `runs/graphemb_abl_rec` | - | 40k | `deploy_v10f` | ub-ins |
| V10g | same | `--dyn-weight 1 --hold-weight 1` | `runs/graphemb_abl_hold` | - | 40k | `deploy_v10g` | ub-ins |
| V10h | same | `--dyn-weight 1 --recon-weight 0.03 --split 8` | `runs/graphemb_twoblock` | - | 40k | `deploy_v10h` (κ 0.25) | ub-ins |
| V10i | same | `--dyn-weight 1 --seed 1` | `runs/graphemb_dyn1.0_s1` | - | 40k | `deploy_v10i` | ub-ins |
| V10j | same | `… --split 8 --rot-aug 30` | `runs/graphemb_twoblock_aug` | - | 40k | `deploy_v10j` (κ 0.25) | ub-ins |
| V10k | same | `--dyn-weight 1 --seed 0` | `runs/graphemb_rs_rec0` | - | 40k | `deploy_v10k` | ub-ins |
| V10l | same | `… --recon-weight 0.001` | `runs/graphemb_rs_rec0.001` | - | 40k | `deploy_v10l` | ub-ins |
| V10m | same | `… --recon-weight 0.003` | `runs/graphemb_rs_rec0.003` | - | 40k | `deploy_v10m` | ub-ins |
| V10n | same | `… --recon-weight 0.01` | `runs/graphemb_rs_rec0.01` | - | 40k | `deploy_v10n` | ub-ins |
| V11a | `sim_belt_v11_20261005` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v11a` | ub-tr0 |
| V11b | `sim_belt_v11_20261005` | `refit_from_buffer.py` | `iter1/refit/r1_joint` | - | epoch 30 | `iter1/deploy_v11b` | ub-tr1 |
| V11c | `sim_belt_v11_20261005` | `refit_from_buffer.py` | `iter2/refit/r2_joint` | - | epoch 30 | `iter2/deploy_v11c` | ub-tr2 |
| V11d | `sim_belt_v11_20261006` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v11d` | ub-tr0 |
| V11e | `sim_belt_v11_20261006` | `refit_from_buffer.py` | `iter1/refit/r1_joint` | - | epoch 30 | `iter1/deploy_v11e` | ub-tr1 |
| V11f | `sim_belt_v11_20261006` | `refit_from_buffer.py` | `iter2/refit/r2_joint` | - | epoch 30 | `iter2/deploy_v11f` | ub-tr2 |
| V11g | `sim_belt_v11_20261006` | `refit_from_buffer.py` | `iter3/refit/r3_joint` | - | epoch 30 | `iter3/deploy_v11g` | ub-tr3 |
| V11h | `sim_belt_v11_20261006` | `refit_from_buffer.py` | `iter4/refit/r4_joint` | - | epoch 30 | `iter4/deploy_v11h` | ub-tr4 |
| V11i | `sim_belt_v11_20261006` | `refit_from_buffer.py` | `iter5/refit/r5_joint` | - | epoch 30 | `iter5/deploy_v11i` | ub-tr5 |
| V12a | `C2_prefix` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v12a` | ub-tr0 |
| V12b | `C2_prefix` | `refit_from_buffer.py` | `iter1/refit/r1_joint` | - | epoch 30 | `iter1/deploy_v12b` | ub-tr1 |
| V12c | `C2_prefix` | `refit_from_buffer.py` | `iter2/refit/r2_joint` | - | epoch 30 | `iter2/deploy_v12c` | ub-tr2 |
| V12d | `C3_both` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v12d` | ub-tr0 |
| V12g | `C1_transport` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v12g` | ub-tr0 |
| V12h | `C1_transport` | `refit_from_buffer.py` | `iter1/refit/r1_joint` | - | epoch 30 | `iter1/deploy_v12h` | ub-tr1 |
| V12j | `full` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v12j` | ub-tr0 |
| V12k | `full` | `refit_from_buffer.py` | `iter1/refit/r1_joint` | - | epoch 30 | `iter1/deploy_v12k` | ub-tr1 |
| V12l | `full` | `refit_from_buffer.py` | `iter2/refit/r2_joint` | - | epoch 30 | `iter2/deploy_v12l` | ub-tr2 |
| V12m | `full` | `refit_from_buffer.py` | `iter3/refit/r3_joint` | - | epoch 30 | `iter3/deploy_v12m` | ub-tr3 |
| V12n | `full` | `refit_from_buffer.py` | `iter4/refit/r4_joint` | - | epoch 30 | `iter4/deploy_v12n` | ub-tr4 |
| V12o | `full` | `refit_from_buffer.py` | `iter5/refit/r5_joint` | - | epoch 30 | `iter5/deploy_v12o` | ub-tr5 |
| V12p | `full` | `refit_from_buffer.py` | `iter6/refit/r6_joint` | - | epoch 30 | `iter6/deploy_v12p` | ub-tr6 |
| V12q | `full` | `refit_from_buffer.py` | `iter7/refit/r7_joint` | - | epoch 30 | `iter7/deploy_v12q` | ub-tr7 |
| V12r | `full` | `refit_from_buffer.py` | `iter8/refit/r8_joint` | - | epoch 30 | `iter8/deploy_v12r` | ub-tr8 |
| V12s | `full` | `refit_from_buffer.py` | `iter9/refit/r9_joint` | - | epoch 30 | `iter9/deploy_v12s` | ub-tr9 |
| V13a | `full2_lcsonly` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v13a` | ub-tr0 |
| V12t | `full2_gate` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v12t` | ub-tr0 |
| V12u | `full2_gate` | `refit_from_buffer.py` | `iter1/refit/r1_joint` | - | epoch 30 | `iter1/deploy_v12u` | ub-tr1 |
| V12v | `v12_18` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v12v` | ub-tr0 |
| V12w | `v12_18` | `refit_from_buffer.py` | `iter1/refit/r1_joint` | - | epoch 30 | `iter1/deploy_v12w` | ub-tr1 |
| V12x | `v12_18` | `refit_from_buffer.py` | `iter2/refit/r2_joint` | - | epoch 30 | `iter2/deploy_v12x` | ub-tr2 |
| V14a | `v12_27` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v14a` | ub-tr0 |
| V15a | `v12_35` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v15a` | ub-tr0 |
| V15b | `v12_35` | `refit_from_buffer.py` | `iter2/refit/r2_lcs_only` | - | epoch 30 | `iter2/deploy_v15b` | ub-tr2 |
| V15c | `v12_35` | `refit_from_buffer.py` | `iter3/refit/r3_lcs_only` | - | epoch 30 | `iter3/deploy_v15c` | ub-tr3 |
| V15d | `v12_35` | `refit_from_buffer.py` | `iter4/refit/r4_lcs_only` | - | epoch 30 | `iter4/deploy_v15d` | ub-tr4 |
| V16a | `v12_36` | `refit_from_buffer.py` | `iter0/refit/r0_seed` | - | epoch 60 | `iter0/deploy_v16a` | ub-tr0 |
| V16b | `v12_36` | `refit_from_buffer.py` | `iter1/refit/r1_lcs_only` | - | epoch 30 | `iter1/deploy_v16b` | ub-tr1 |
| V16c | `v12_36` | `refit_from_buffer.py` | `iter2/refit/r2_lcs_only` | - | epoch 30 | `iter2/deploy_v16c` | ub-tr2 |
| V16d | `v12_36` | `refit_from_buffer.py` | `iter3/refit/r3_lcs_only` | - | epoch 30 | `iter3/deploy_v16d` | ub-tr3 |
| V16e | `v12_36` | `refit_from_buffer.py` | `iter4/refit/r4_lcs_only` | - | epoch 30 | `iter4/deploy_v16e` | ub-tr4 |
| V16f | `v12_36` | `refit_from_buffer.py` | `iter5/refit/r5_lcs_only` | - | epoch 30 | `iter5/deploy_v16f` | ub-tr5 |
| V16g | `v12_36` | `refit_from_buffer.py` | `iter6/refit/r6_lcs_only` | - | epoch 30 | `iter6/deploy_v16g` | ub-tr6 |
| V16h | `v12_36` | `refit_from_buffer.py` | `iter7/refit/r7_lcs_only` | - | epoch 30 | `iter7/deploy_v16h` | ub-tr7 |
| V16i | `v12_36` | `refit_from_buffer.py` | `iter8/refit/r8_lcs_only` | - | epoch 30 | `iter8/deploy_v16i` | ub-tr8 |
| V16j | `v12_36` | `refit_from_buffer.py` | `iter9/refit/r9_lcs_only` | - | epoch 30 | `iter9/deploy_v16j` | ub-tr9 |
| V18a | `v12_46` | `refit_from_buffer.py` | `iter0/refit/E0_s6` | - | epoch 60 | `iter0/deploy_v18a` | ub-tr0 |
| V18b | `v12_46` | `refit_from_buffer.py` | `iter1/refit/r1_lcs_only` | - | epoch 30 | `iter1/deploy_v18b` | ub-tr1 |
| V18c | `v12_46` | `refit_from_buffer.py` | `iter2/refit/r2_lcs_only` | - | epoch 30 | `iter2/deploy_v18c` | ub-tr2 |
| V18d | `v12_46` | `refit_from_buffer.py` | `iter3/refit/r3_lcs_only` | - | epoch 30 | `iter3/deploy_v18d` | ub-tr3 |
| V18e | `v12_46` | `refit_from_buffer.py` | `iter4/refit/r4_lcs_only` | - | epoch 30 | `iter4/deploy_v18e` | ub-tr4 |
| V18f | `v12_46` | `refit_from_buffer.py` | `iter5/refit/r5_lcs_only` | - | epoch 30 | `iter5/deploy_v18f` | ub-tr5 |
| V18g | `v12_46` | `refit_from_buffer.py` | `iter6/refit/r6_lcs_only` | - | epoch 30 | `iter6/deploy_v18g` | ub-tr6 |
| V18h | `v12_46` | `refit_from_buffer.py` | `iter7/refit/r7_lcs_only` | - | epoch 30 | `iter7/deploy_v18h` | ub-tr7 |
| V18i | `v12_46` | `refit_from_buffer.py` | `iter8/refit/r8_lcs_only` | - | epoch 30 | `iter8/deploy_v18i` | ub-tr8 |
| V18j | `v12_46` | `refit_from_buffer.py` | `iter9/refit/r9_lcs_only` | - | epoch 30 | `iter9/deploy_v18j` | ub-tr9 |
| V22a | `v12_57` | `refit_from_buffer.py` | `iter0/refit/JPG_s4` | - | epoch 40 | `iter0/deploy_v22a` | ub-tr0 |
| V22b | `v12_57` | `refit_from_buffer.py` | `iter1/refit/r1_joint` | - | epoch 30 | `iter1/deploy_v22b` | ub-tr1 |

Checkpoint date folders: V1 and V2 `2026-09-24` (V2's campaign folder is dated 20260925),
V3 `2026-09-28`, V4/V5 `2026-09-29`, V6/V7 `2026-09-30`. V10 encoder runs (`phi.pt`, `z.npz`,
optional `decoder_weights.npz`) are in `data/lcs/diag/20261001-temporal-latent/runs/<tag>/`;
LCS fits in the same folder as `lcs_<Name>[_hold<w>].pt`.

**Headline results** (insertion test one-step / h7 belt RMSE in mm; closed loop: strict
counts, mean final belt RMSE mm; free space: mean final / min mm; "-" = not run):

| ID | insertion 1-step / h7 | seat6 (Qw, ub-ins) | yaw12 (Qw) | fs6 (Qw) | other |
|---|---|---|---|---|---|
| V1b | 0.62 / 2.06 (old split) | - | - | - | gv1 11/26 engaged, 2 strict (BL 14/26) |
| V2 | 0.577 / 1.92 | 4/6 (6/6 engaged), 6.2 | - | 23.4 / 10.0 | admm5 4/6, admm10 0/6; gv1+gv3 Qw+wp 22/40 engaged (BL 14/28); stages10 3/10 |
| V2b | 2.28 / 3.25 | - | - | - | - |
| V3a | 0.551 / 1.88 | 1/6 (ub-pool), 79.8; ub-ins 1/6 (E3) | - | 8.4 / 5.5 (ub-pool) | admm5 0/6, admm10 0/6 |
| V3b | 0.583 / 1.88 | - | - | - | - |
| V4a | 0.509 / 1.98 | 0/6, 13.1 | 0/12, 13.6 | 16.0 / 8.7; ub-pool 11.4 / 7.3 | QA seat 0/6 (10.9), yaw 6/12 (9.9); QB seat 1/6, yaw 3/12 |
| V4b | 0.711 / 1.42 | 0/6, 132.7 | - | - | - |
| V4c | 0.643 / 1.93 | 1/6, 90.4 | - | - | QA 0/6, 0/12; QB 0/6, 0/12 |
| V5a | 0.722 / 2.42 | 0/6, 82.9 | 0/12, 94.9 | 125.9 / 12.4 (ub-pool) | - |
| V5b | 1.192 / 21.25 | 0/6, 141.1 | 0/12, 118.6 | 64.7 / 15.1 (ub-pool) | - |
| V6a | 0.825 / 3.62 | 0/6, 95.8 | 0/12, 109.9 | 69.8 (ub-pool) | - |
| V6b | 0.834 / 3.13 | 0/6, 80.7 | 2/12, 66.4 | 65.6 (ub-pool) | - |
| V7a | 0.509 / 2.29 | 2/6 (3/6 engaged), 10.0 | 3/12, 23.5 | 12.7 / 7.3 (ub-pool) | head off 0/6 |
| V7b | 0.569 / 1.32 | - | - | - | - |
| V8a | 0.574 / 2.57 | 0/6 (0/6 engaged), 19.9 (ub-ins) | 0/12, 24.4 | - | V8 on-policy, iteration 7 |
| V8b | 0.569 / 2.54 | 0/6 (0/6 engaged), 19.9 (ub-ins) | 0/12, 25.6 | - | V8 on-policy, iteration 8 |
| V8c | 0.564 / 2.54 | 0/6 (0/6 engaged), 19.3 (ub-ins) | 0/12, 26.9 | - | V8 on-policy, iteration 9 |
| V10a | 0.513 / 0.455 rel | 2/6 (3/6 engaged), 21.7; R2 3/6 (4/6), 20.9 (Qi) | 0/12 (4/12 engaged), 155.3; 6 unsafe | - | cell-s30 urtip6_high strict; seat6-1goal30 0/6 (2/6 eng.); under12 0/12; policy gate both-down 100 % (below both-up 10 %) |
| V10b | 0.409 / 0.328 rel (hold-10 LCS) | 0/6, 22.5 (Qi) | 0/12, 36.7 | - | policy gate both-down 28 % (below both-up 100 %) |
| V10c | 0.413 / 0.333 rel | 0/6, 24.6 (Qi) | - | - | policy gate both-down 100 % (below both-up 0 %) |
| V10d | 0.408 / 0.328 rel (hold-1 LCS) | 0/6, 15.2 (Qi) | - | - | policy gate both-down 100 % (below both-up 0 %) |
| V10e | 0.624 / 0.51 rel | 0/6, 71.2 (Qi) | - | - | policy gate both-down 38 % (below both-up 0 %) |
| V10f | 0.456 / 0.394 rel | 0/6, 55.9 (Qi) | - | - | policy gate both-down 62 % (below both-up 90 %) |
| V10g | 0.526 / 0.468 rel | 0/6, 103.4 (Qi) | - | - | policy gate both-down 15 % (below both-up 10 %) |
| V10h | 0.819 / 0.695 rel | 0/6, 42.2 (Qi); 1 unsafe | - | - | policy gate both-down 40 % (below both-up 35 %) |
| V10i | 0.547 / 0.469 rel | 0/6, 19.7 (Qi) | - | - | policy gate both-down 42 % (below both-up 100 %) |
| V10j | 0.718 / 0.598 rel | 0/6, 20.1 (Qi) | - | - | policy gate both-down 62 % (below both-up 0 %) |
| V10k | 0.513 / 0.452 rel | 0/6, 45.5 (Qi) | - | - | policy gate both-down 50 % (below both-up 38 %) |
| V10l | 0.548 / 0.481 rel | 0/6, 108.0 (Qi) | - | - | policy gate both-down 22 % (below both-up 0 %) |
| V10m | 0.509 / 0.444 rel | 0/6, 27.1 (Qi) | - | - | policy gate both-down 88 % (below both-up 0 %) |
| V10n | 0.486 / 0.42 rel | 0/6, 49.0 (Qi) | - | - | policy gate both-down 68 % (below both-up 100 %) |

V1a, V1c and V6 have no tabled results (V1a/V1c raw numbers: lcs_learning
`outputs/sim_belt_ablation_20260924/eval_results.json`). Sources: one-step/h7 from
lcs_learning `outputs/sim_belt_v{2,3,4,7}_*/eval_table*.md`, `v5`/`v6` `eval_table_v{5,6}.md`;
closed loop from the tables in §3 and `docs/learned-mpc.md` (2026-09-24 to 2026-09-30).
**V10 offline numbers are not in mm:** "rel" = held-out one-step / h7 latent error ÷ the
no-motion error (whitened), for the encoder's no-hold LCS refit, from
`data/lcs/diag/20261001-temporal-latent/lcs_fit_all_nohold.json`; V10b / V10d quote their
deployed hold-10 / hold-1 fits (`lcs_fit_hold{10.0,1.0}.json`). Refits are not bitwise
reproducible (V10a: 0.513 / 0.455 here vs 0.51 / 0.46 in the session); the files'
`cl_drift_over_step` is not quoted (it varies between refits). Policy gate
(`…/policy_gate.txt`): share of riding-over states where an offline 7-step constant-input
solve of the deployed model + cost presses both arms down (below: lifts both). It is a
screen, not a test: V10a scores 100 % and every model below 90 % failed seat6, but V10c
and V10d also score 100 % and seat 0/6. The one-step lever gate is in `…/lever_gate.txt`.
Obstacle separation (held-out, blocked
farther than an equal-RMSE unblocked state, fraction / median distance ratio; V4a 0.687 / 1.11):
V10a 0.870 / 1.54, V10b-d 0.840 / 1.95, V10e 0.840 / 1.88, V10f 0.889 / 1.55, V10g 0.863 / 1.47,
V10h 0.849 / 1.44, V10i 0.848 / 1.43, V10j 0.839 / 1.91, V10k 0.870 / 1.54, V10l 0.888 / 1.54,
V10m 0.891 / 1.55, V10n 0.891 / 1.56 (`…/blocked_vs_unblocked_heldout.json`). Held-out recon
median (joint decoders; `…/embed_<tag>.log`): V10b-d 2.53, V10f 2.72, V10h 4.03, V10j 3.07,
V10l 6.08, V10m 5.29, V10n 3.86 mm. V10 seat6 cells use the same harness and params as V4a
except the model files; the cost is isotropic (`Qi`) although the folder names say `Qw`.

### Encoder-study models (V12-42 to V12-61, not made by a loop)

Trained directly with `train_joint_pointnet_lcs.py` (two-stage for JEPA arms), exported by
`cmp/encoder_study/sim/prep_*.py` to `data/lcs/realfirst/cmp/encoder_study/sim/<ID>/`.
Checkpoints: lcs_learning `outputs/sim_belt_v12_20261006/<run>/`. Hard suite = 11 starts × 2,
no demo prior, engaged count (noise 0.2 for V17*, 0 from V12-49 on). Tiers: T1 = 3 demos +
V12-29b MILES; T2 = T1 + V12-35/36 on-policy; T1ga = T1 + 4 grasp-angle demos + their MILES;
T1ga_f = T1ga + 42 more operator attempts (39 failed). Details: `docs/realfirst-campaign.md`.

| ID | run (checkpoint) | recipe | tier | seed | hard suite | run |
|---|---|---|---|---|---|---|
| V17a | `v12_42/t1/E0_s4` | reconstruction + LCS (E0) | T1 | 4 | 8/22; 7/22 progress start | V12-43/44 |
| V17b | `v12_42/t1/E2_aug_act0.1_s4` | E0 + point augmentation + Delta-JEPA action head 0.1 | T1 | 4 | 1/22 | V12-43 |
| V17c | `v12_42/t1/E1_aug_s4` | E0 + point augmentation | T1 | 4 | 0/22 | V12-44 |
| V17d | `v12_42/t1/E0_s6` | E0 | T1 | 6 | 11/22; 9/22 at noise 0 | V12-45/49 |
| V17e | `v12_42/t1/E0_s5` | E0 | T1 | 5 | 9/22 | V12-45 |
| V19a | `v12_49/t1/B_s6` | E0 with J fixed to 0 (B) | T1 | 6 | 7/22 | V12-50 |
| V19b | `v12_49/t1/C_s6` | J learned + violation-QP loss + full-LCP forward (C) | T1 | 6 | not run (MPPI has no full-LCP solver) | V12-50 |
| V19c | `v12_49/t1/B_s5` | B | T1 | 5 | 11/22 | V12-50 |
| V20a | `v12_51/t1/JP0_s6` | JEPA + SIGReg, J 0, post-hoc decoder | T1 | 6 | 0/22 | V12-53 |
| V21a | `v12_52/t1/JPG_s6` | JEPA + graph stress (JPG), SIGReg off, J 0 | T1 | 6 | 11/22 | V12-54 |
| V21b | `v12_52/t1/BG_s6` | reconstruction + graph stress (BG) | T1 | 6 | 0/22 | V12-54 |
| V21c | `v12_52/t1/JPG_s4` | JPG | T1 | 4 | 11/22 (V22a re-export: 12/22) | V12-55 |
| V21d | `v12_52/t1/JPG_s5` | JPG | T1 | 5 | 5/22 | V12-55 |
| V23a | `v12_56/t2/JPG2_s4` | JPG, T2 graph | T2 | 4 | 1/22 | V12-56 |
| V23b | `v12_56/t2/JPG2_s6` | JPG, T2 graph | T2 | 6 | 1/22 | V12-56 |
| V24a | `v12_58/t1ga/JPG_s4` | JPG, T1ga graph | T1ga | 4 | 0/22 | V12-58 |
| V24b | `v12_58/t1ga/JPG_s6` | JPG, T1ga graph | T1ga | 6 | 2/22 | V12-58 |
| V25a | `v12_60/t2_gm/JPG2GM_s4` | JPG + goal margin W 0.1 / M 0.3 | T2 | 4 | 6/22 | V12-60 |
| V25b | `v12_60/t2_gm/JPG2GM_s6` | JPG + goal margin W 0.1 / M 0.3 | T2 | 6 | 1/22 | V12-60 |
| V26a | `v12_61/t1gaf_gm/JPGGM_s4` | JPG + goal margin W 0.1 / M 0.3, T1ga_f graph | T1ga_f | 4 | 0/22 | V12-61 |
| **V26b** | `v12_61/t1gaf_gm/JPGGM_s6` | JPG + goal margin W 0.1 / M 0.3, T1ga_f graph | T1ga_f | 6 | **16/22** (best) | V12-61 |
| V27a | `v12_62/t1gaf_gt_gm/JPGGM_s4` | V26 recipe on sim GT belt points (`pcd_belt_seg` := `pcd_belt`, pulley frame; diagnostic, not real-available) | T1ga_f GT twin | 4 | 0/22 | V12-62 |
| V27b | `v12_62/t1gaf_gt_gm/JPGGM_s6` | as V27a | T1ga_f GT twin | 6 | 0/22 | V12-62 |
| V28a | `v12_63/t1gaf_gt_gm16/JPGGM_s4` | V27 recipe (sim GT points), latent 16 | T1ga_f GT twin | 4 | 0/22 | V12-63 |
| V28b | `v12_63/t1gaf_gt_gm16/JPGGM_s6` | V27 recipe (sim GT points), latent 16 | T1ga_f GT twin | 6 | 14/22 | V12-63 |
| V29a | `v12_64/t1gaf_gm16/JPGGM_s4` | V26 recipe (camera seg), latent 16 | T1ga_f | 4 | 0/22 | V12-64 |
| V29b | `v12_64/t1gaf_gm16/JPGGM_s6` | V26 recipe (camera seg), latent 16 | T1ga_f | 6 | 0/22 | V12-64 |
| V30a | `v12_65/t1gaf_gt_gm25/JPGGM_s4` | V27 recipe (sim GT points), latent 25 | T1ga_f GT twin | 4 | 0/22 | V12-65 |
| V30b | `v12_65/t1gaf_gt_gm25/JPGGM_s6` | V27 recipe (sim GT points), latent 25 | T1ga_f GT twin | 6 | 13/22 | V12-65 |
| V31a | `v12_66/t1gaf_gm25/JPGGM_s4` | V26 recipe (camera seg), latent 25 | T1ga_f | 4 | 16/22 | V12-66 |
| V31b | `v12_66/t1gaf_gm25/JPGGM_s6` | V26 recipe (camera seg), latent 25 | T1ga_f | 6 | 0/22 | V12-66 |
| V32a | `v12_67/t1gaf_gt_gm5/JPGGM_s4` | V27 recipe (sim GT points), latent 5 | T1ga_f GT twin | 4 | 1/22 | V12-67 |
| V32b | `v12_67/t1gaf_gt_gm5/JPGGM_s6` | V27 recipe (sim GT points), latent 5 | T1ga_f GT twin | 6 | 14/22 | V12-67 |
| V33a | `v12_68/t1gaf_gm5/JPGGM_s4` | V26 recipe (camera seg), latent 5 | T1ga_f | 4 | 0/22 | V12-68 |
| V33b | `v12_68/t1gaf_gm5/JPGGM_s6` | V26 recipe (camera seg), latent 5 | T1ga_f | 6 | 1/22 | V12-68 |
| V34a | `v12_69/t1gaf_gm_ee/JPGGMEE_s4` | V26 recipe (camera seg, latent 10) + EE positions in the LCS (A_p/E_p, world-frame fold) | T1ga_f | 4 | 0/22 | V12-69 |
| V34b | `v12_69/t1gaf_gm_ee/JPGGMEE_s6` | V26 recipe (camera seg, latent 10) + EE positions in the LCS (A_p/E_p, world-frame fold) | T1ga_f | 6 | 0/22 | V12-69 |
| V34c | `v12_69/t1gaf_gm_ee/JPGGMEE_s5` | V26 recipe (camera seg, latent 10) + EE positions in the LCS (A_p/E_p, world-frame fold) | T1ga_f | 5 | 0/22 | V12-69 |
| V35a | `v12_70/t1gaf_gm_sbp/JPGGMSBP_s4` | V26b recipe on V4a encoder inputs (scene cloud 1800 + 150 sim-GT belt points + 40-d state, latent 16); post-hoc decoder RMSE on GT belt points (diagnostic: GT input) | T1ga_f | 4 | 0/22 | V12-70 |
| V35b | `v12_70/t1gaf_gm_sbp/JPGGMSBP_s6` | V26b recipe on V4a encoder inputs (scene cloud 1800 + 150 sim-GT belt points + 40-d state, latent 16); post-hoc decoder RMSE on GT belt points (diagnostic: GT input) | T1ga_f | 6 | 1/22 | V12-70 |
| V35c | `v12_70/t1gaf_gm_sbp/JPGGMSBP_s5` | V26b recipe on V4a encoder inputs (scene cloud 1800 + 150 sim-GT belt points + 40-d state, latent 16); post-hoc decoder RMSE on GT belt points (diagnostic: GT input) | T1ga_f | 5 | 0/22 | V12-70 |
| V36a | `v12_71/t1gaftr_gm/JPGGMTR_s4` | V26b recipe on T1ga_f_tr (demos cut 30 frames after 'done') | T1ga_f_tr | 4 | 0/22 | V12-71 |
| V36b | `v12_71/t1gaftr_gm/JPGGMTR_s6` | V26b recipe on T1ga_f_tr (demos cut 30 frames after 'done') | T1ga_f_tr | 6 | 16/22 | V12-71 |
| V36c | `v12_71/t1gaftr_gm/JPGGMTR_s5` | V26b recipe on T1ga_f_tr (demos cut 30 frames after 'done') | T1ga_f_tr | 5 | 7/22 | V12-71 |
| V37a | `v12_72/t1gaftr2_gm/JPGGMTR2_s4` | V26b recipe on T1ga_f_tr2 (T1ga_f_tr minus MILES past the cut) | T1ga_f_tr2 | 4 | 8/22 | V12-72 |
| V37b | `v12_72/t1gaftr2_gm/JPGGMTR2_s6` | V26b recipe on T1ga_f_tr2 (T1ga_f_tr minus MILES past the cut) | T1ga_f_tr2 | 6 | 2/22 | V12-72 |
| V37c | `v12_72/t1gaftr2_gm/JPGGMTR2_s5` | V26b recipe on T1ga_f_tr2 (T1ga_f_tr minus MILES past the cut) (collapsed, eff_rank 1.00) | T1ga_f_tr2 | 5 | 1/22 | V12-72 |
| V38a | `v12_75/t1gaftr3_gm/JPGGMTR3_s4` | V26b recipe on T1ga_f_tr3 (T1ga_f_tr minus the 15 hooked-start episodes) | T1ga_f_tr3 | 4 | 5/20 (H 7: 11/20) | V12-75 |
| V38b | `v12_75/t1gaftr3_gm/JPGGMTR3_s6` | V26b recipe on T1ga_f_tr3 (T1ga_f_tr minus the 15 hooked-start episodes) | T1ga_f_tr3 | 6 | 2/20 (H 7: 6/20) | V12-75 |
| V38c | `v12_75/t1gaftr3_gm/JPGGMTR3_s5` | V26b recipe on T1ga_f_tr3 (T1ga_f_tr minus the 15 hooked-start episodes) | T1ga_f_tr3 | 5 | 5/20 (H 7: 7/20) | V12-75 |
| V39a | `v12_79/dense/JPGGMD_s4` | V26b recipe on T1ga_f_tr re-rendered with the ZED720 camera (2.5 mm voxel, clean), N 1600, Chamfer decoder | T1ga_f_tr_d | 4 | H 7: 3/20 | V12-79 |
| V39b | `v12_79/dense/JPGGMD_s6` | V26b recipe on T1ga_f_tr re-rendered with the ZED720 camera (2.5 mm voxel, clean), N 1600, Chamfer decoder | T1ga_f_tr_d | 6 | H 7: 14/20 | V12-79 |
| V39c | `v12_79/dense/JPGGMD_s5` | V26b recipe on T1ga_f_tr re-rendered with the ZED720 camera (2.5 mm voxel, clean), N 1600, Chamfer decoder | T1ga_f_tr_d | 5 | H 7: 16/20 | V12-79 |
| V40a | `v12_79/dense_noisy/JPGGMDN_s4` | as V39, `real_like` clouds | T1ga_f_tr_dn | 4 | H 7: 9/20 | V12-79 |
| V40b | `v12_79/dense_noisy/JPGGMDN_s6` | as V39, `real_like` clouds | T1ga_f_tr_dn | 6 | H 7: 0/20 (under 18) | V12-79 |
| V40c | `v12_79/dense_noisy/JPGGMDN_s5` | as V39, `real_like` clouds | T1ga_f_tr_dn | 5 | H 7: 0/20 (under 18) | V12-79 |
| V44a | `v12_79/orig544/JPGGM544_s4` | V26b recipe on T1ga_f_tr at N 544 (every point, no resample jitter), Sinkhorn decoder | T1ga_f_tr | 4 | H 7: 9/20 | V12-79 |
| V44b | `v12_79/orig544/JPGGM544_s6` | V26b recipe on T1ga_f_tr at N 544 (every point, no resample jitter), Sinkhorn decoder | T1ga_f_tr | 6 | H 7: 14/20 | V12-79 |
| V44c | `v12_79/orig544/JPGGM544_s5` | V26b recipe on T1ga_f_tr at N 544 (every point, no resample jitter), Sinkhorn decoder | T1ga_f_tr | 5 | H 7: 13/20 | V12-79 |
| V45a | `v12_80/v4a544/V4A544_s4` | V4a recipe on V44's input (N 544): joint recon + decoded-next + LCS, latent 16, lr 1e-3, batch 154, 300 epochs, no violation grad to the encoder, Chamfer; no JEPA / graph / goal margin | T1ga_f_tr | 4 | H 7: 16/20 | V12-80 |
| V45b | `v12_80/v4a544/V4A544_s6` | V4a recipe on V44's input (N 544): joint recon + decoded-next + LCS, latent 16, lr 1e-3, batch 154, 300 epochs, no violation grad to the encoder, Chamfer; no JEPA / graph / goal margin | T1ga_f_tr | 6 | H 7: 6/20 | V12-80 |
| V45c | `v12_80/v4a544/V4A544_s5` | V4a recipe on V44's input (N 544): joint recon + decoded-next + LCS, latent 16, lr 1e-3, batch 154, 300 epochs, no violation grad to the encoder, Chamfer; no JEPA / graph / goal margin | T1ga_f_tr | 5 | H 7: 18/20 (only up-20 fails); **20/20** with the UR 30° / Franka 20° rotation envelope (V12-87) | V12-80, V12-87 |
| V45d | `v12_83/ft/V4A544FT_s5` | V45c fine-tuned (warm, 30 epochs, lr 3e-4) on its two failed up-20 rollouts (label False, weight 5, train-only) | T1ga_f_tr + 2 | 5 | H 7: 18/20 (up-20 0/2; a training start) | V12-83 |
| V45e | `v12_84/ft/V4A544FT2_s5` | V45c fine-tuned (warm, 30 epochs, lr 3e-4) on T1ga_f_tr + 563 MILES segments around V45d's failed up-20 rollouts (train-only) | T1ga_f_tr + MILES(fail) | 5 | H 7: 18/20 (up-20 0/2; a training start) | V12-84 |
| V45f | `v12_85/ft/V4A544FT3_s5` | as V45e on the new MILES segments only (old rows weight 1e-6): forgets (gate rejected, recon 4.58 mm) | MILES(fail) | 5 | partial: 2/10, paused by the user | V12-85 |
| V47a | `v12_82/orig544l16/JPGGM544L16_s4` | V44 recipe, latent 16 | T1ga_f_tr | 4 | not run (decoder probe 3.87 mm) | V12-82 |
| V47b | `v12_82/orig544l16/JPGGM544L16_s6` | V44 recipe, latent 16 | T1ga_f_tr | 6 | not run (decoder probe 2.86 mm) | V12-82 |
| V47c | `v12_82/orig544l16/JPGGM544L16_s5` | V44 recipe, latent 16 (collapsed dynamics, eff_rank 1.26) | T1ga_f_tr | 5 | not run (decoder probe 2.57 mm) | V12-82 |
| V48a | `v12_88/v48/V48_s4` | V45 recipe without current-state recon or the one-step decoded next state; decoded LCS rollout recon at k 1..7 (belt_ms K 7, γ 0.9, multi-step latent weight 0); violation grad to the encoder on | T1ga_f_tr | 4 | H 7 UR30/F20: 18/20 (up-20 0/2) (held-out h7 3.62 mm) | V12-88 |
| V48b | `v12_88/v48/V48_s6` | V45 recipe without current-state recon or the one-step decoded next state; decoded LCS rollout recon at k 1..7 (belt_ms K 7, γ 0.9, multi-step latent weight 0); violation grad to the encoder on (h24 rollout diverges: 506 mm) | T1ga_f_tr | 6 | H 7 UR30/F20: **20/20** (h7 3.12 mm) | V12-88 |
| V48c | `v12_88/v48/V48_s5` | V45 recipe without current-state recon or the one-step decoded next state; decoded LCS rollout recon at k 1..7 (belt_ms K 7, γ 0.9, multi-step latent weight 0); violation grad to the encoder on; 1 contact mode on held-out data | T1ga_f_tr | 5 | H 7 UR30/F20: 11/20 (slanted 9) (h7 3.49 mm) | V12-88 |

V21c with the EE data-support MPPI cost (weights 1 / 10): 12/22 each (V12-59b, eval tags
`V21c_n0_ees1/10`, no new model).

## 3. Closed-loop evaluation index

Every closed-loop run folder under `data/lcs/mpc_eval/` plus the E3 bounds runs, the E4
c3-consistency ADMM runs and the snapshot re-runs (`capture/`) under `data/lcs/diag/`. One row per
(model, suite, cost, limits) group; `mpc_eval/` = `data/lcs/mpc_eval/`, `diag/` =
`data/lcs/diag/`, `…/` = the same top folder. cells = distinct (target, start) pairs, eps =
episodes, both counted from each run's `index.json`. Folder stamps `20260924-2xxxxx` and
`20260925-*` belong to the 2026-09-25 day log (shown in brackets). The 2026-09-24/25 runs have
no per-folder table; their results are in the worktree
`systems/parameters/learned_archive/README.md` and `docs/learned-mpc.md`.

| label | path | date (log day) | cells / eps | table | note |
|---|---|---|---|---|---|
| BL · nom · - · - | `mpc_eval/20260924-124753-baseline-nominal` | 09-24 | 1 / 2 | worktree `learned_archive/README.md` | waypoint baseline |
| V1b · nom · Qw · ub-ins admm3 | `mpc_eval/20260924-124821-learned-nominal` | 09-24 | 1 / 2 | worktree `learned_archive/README.md` | v1 defaults, w_r 0.1, 2 stages, demo 0/59 |
| V1b · nom · Qw · ub-ins admm3 | `mpc_eval/20260924-130800-learned-default-nominal` | 09-24 | 1 / 3 | worktree `learned_archive/README.md` | v1 defaults |
| BL · gv1 · - · - | `mpc_eval/20260924-131014-baseline-set1` | 09-24 | 13 / 26 | worktree `learned_archive/README.md` | baseline set1 x2 (14/26) |
| BL · nom · - · - | `mpc_eval/20260924-131302-baseline-nominal` | 09-24 | 1 / 3 | worktree `learned_archive/README.md` | waypoint baseline |
| V1b · gv1 · Qw · ub-ins admm3 | `mpc_eval/20260924-131328-learned-default-set1` | 09-24 | 13 / 13 | worktree `learned_archive/README.md` | v1 defaults: 0/13, 13 aborts |
| V1b · gv1-2 · Qw · ub-ins admm3 | `mpc_eval/20260924-131721-screen-wr10` | 09-24 | 2 / 2 | worktree `learned_archive/README.md` | screen wr10 |
| V1b · gv1-2 · Qw · ub-ins admm3 | `mpc_eval/20260924-131753-screen-wr100` | 09-24 | 2 / 2 | worktree `learned_archive/README.md` | screen wr100 |
| V1b · gv1-2 · Qw · ub-hand admm3 | `mpc_eval/20260924-131828-screen-ubnd` | 09-24 | 2 / 2 | worktree `learned_archive/README.md` | screen ubnd |
| V1b · gv1-2 · Qw · ub-ins admm3 | `mpc_eval/20260924-131931-screen-wr100_s0skip` | 09-24 | 2 / 2 | worktree `learned_archive/README.md` | screen wr100_s0skip |
| V1b · gv1-2 · Qw · ub-hand admm3 | `mpc_eval/20260924-131947-screen-ubnd_wr30_s0skip` | 09-24 | 2 / 2 | worktree `learned_archive/README.md` | screen ubnd_wr30_s0skip |
| V1b · gv1-2 · Qw · ub-hand admm3 | `mpc_eval/20260924-132059-screen-ubnd_wr30_s0skip_tol2` | 09-24 | 2 / 2 | worktree `learned_archive/README.md` | screen ubnd_wr30_s0skip_tol2 |
| V1b · gv1-2 · Qw · ub-hand-x0.5 admm3 | `mpc_eval/20260924-132116-screen-ubnd_wr30_s0skip_ub05` | 09-24 | 2 / 2 | worktree `learned_archive/README.md` | screen ubnd_wr30_s0skip_ub05 |
| V1b · gv1-10 · Qw · ub-ins admm3 | `mpc_eval/20260924-132203-tune-wr100` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | tune wr100 |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-132429-tune-ubnd` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | tune ubnd |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-132709-tune-ubnd_wr30_s0skip` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | tune ubnd_wr30_s0skip |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-132800-tune-ubnd_wr30_s0skip_tol2` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | tune ubnd_wr30_s0skip_tol2 |
| V1b · gv1-10 · Qw · ub-ins admm3 | `mpc_eval/20260924-132851-tune-wr100_s0skip` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | tune wr100_s0skip |
| V1b · gv1-10 · Qw · ub-hand-x0.5 admm3 | `mpc_eval/20260924-132943-tune-ubnd_wr30_s0skip_ub05` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | tune ubnd_wr30_s0skip_ub05 |
| V1b · gv1 · Qw · ub-hand admm3 | `mpc_eval/20260924-133132-final-learned-set1` | 09-24 | 13 / 26 | worktree `learned_archive/README.md` | 09-24 winner ubnd_wr30_s0skip_tol2: 11/26 (2 strict) vs BL 14/26 |
| V1b · nom · Qw · ub-hand admm3 | `mpc_eval/20260924-133342-final-learned-nominal` | 09-24 | 1 / 3 | worktree `learned_archive/README.md` | 09-24 winner |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-163529-exp-eez` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | height floors eez |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-163633-exp-eez_noexact` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | height floors eez_noexact |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-164211-exp-eez_ur13` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | height floors eez_ur13 |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-164322-exp-eez_franka_free` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | height floors eez_franka_free |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-165113-exp-eez_ur13_traj_time` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | height floors eez_ur13_traj_time (demo-traj ref) |
| V1b · gv1-10 · Qw · ub-hand admm3 | `mpc_eval/20260924-165323-exp-eez_ur13_traj_progress` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | height floors eez_ur13_traj_progress (demo-traj ref) |
| V1b · gv1-10 · Qw · ub-hand admm2 | `mpc_eval/20260924-165712-exp-eez_ur13_traj_progress_admm2` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | height floors eez_ur13_traj_progress_admm2 (demo-traj ref) |
| V1b · gv1-10 · Qw · ub-hand admm1 | `mpc_eval/20260924-170028-exp-eez_ur13_traj_progress_admm1` | 09-24 | 10 / 10 | worktree `learned_archive/README.md` | height floors eez_ur13_traj_progress_admm1 (demo-traj ref) |
| V2 · nom · Qw · ub-ins | `mpc_eval/20260924-200235-deploy-v2-smoke` | 09-24 (09-25) | 1 / 2 | `learned-mpc.md` 09-25; archive README | v2 smoke, demo-traj progress, w_r 30 |
| V2 · nom · Qw · ub-ins | `mpc_eval/20260924-200601-deploy-v2-stages-smoke` | 09-24 (09-25) | 1 / 2 | `learned-mpc.md` 09-25; archive README | v2 smoke, 2 fixed stages |
| BL · gv1 · - · - | `mpc_eval/20260924-201659-evalv2-baseline-set1` | 09-24 (09-25) | 13 / 13 | `learned-mpc.md` 09-25; archive README | eval-v2 baseline |
| BL · gv3 · - · - | `mpc_eval/20260924-201842-evalv2-baseline-set3` | 09-24 (09-25) | 8 / 16 | `learned-mpc.md` 09-25; archive README | eval-v2 baseline |
| BL · nom · - · - | `mpc_eval/20260924-202029-evalv2-baseline-nominal` | 09-24 (09-25) | 1 / 3 | `learned-mpc.md` 09-25; archive README | eval-v2 baseline |
| V2 · gv1 · Qw · ub-ins | `mpc_eval/20260924-202056-evalv2-v2full-set1` | 09-24 (09-25) | 13 / 26 | `learned-mpc.md` 09-25; archive README | v2 full (traj progress, w_r 30, penalty bug): 0/40 held-out |
| V2 · nom · Qw · ub-ins | `mpc_eval/20260924-202855-evalv2-v2full-nominal` | 09-24 (09-25) | 1 / 3 | `learned-mpc.md` 09-25; archive README | v2 full |
| V2 · gv1 · Qw · ub-ins | `mpc_eval/20260924-202954-evalv2-abl-v2stages-set1` | 09-24 (09-25) | 13 / 13 | `learned-mpc.md` 09-25; archive README | ablation: 2 fixed stages, 0/13 |
| V1b · gv1 · Qw · ub-hand admm2 | `mpc_eval/20260924-203301-evalv2-abl-oldmodel-phase0-set1` | 09-24 (09-25) | 13 / 13 | `learned-mpc.md` 09-25; archive README | ablation: V1b + traj progress admm2 |
| V2 · gv3 · Qw · ub-ins | `mpc_eval/20260924-203612-evalv2-v2full-set3` | 09-24 (09-25) | 8 / 16 | `learned-mpc.md` 09-25; archive README | v2 full |
| V2 · gv1-d10 · Qw · ub-ins | `mpc_eval/20260924-211039-ctrldiag-honor_wr0p3` | 09-24 (09-25) | 10 / 10 | `learned-mpc.md` 09-25; archive README | honor penalize_input_change, w_r 0.3 |
| V2 · gv1-d10 · Qw+wp · ub-ins | `mpc_eval/20260924-211342-ctrldiag-honor_wr0p3_wp0p03` | 09-24 (09-25) | 10 / 10 | `learned-mpc.md` 09-25; archive README | Phase-B winner: 6/10 vs BL 4/10 |
| V2 · gv1-d10 · Qw · ub-ins | `mpc_eval/20260924-211646-ctrldiag-penalty_wr0p3` | 09-24 (09-25) | 10 / 10 | `learned-mpc.md` 09-25; archive README | w_r 0.3, penalty bug on |
| V2 · gv1 · Qw+wp · ub-ins | `mpc_eval/20260924-212141-evalv2honor-set1` | 09-24 (09-25) | 13 / 26 | `learned-mpc.md` 09-25; archive README | eval-v2-honor (22/40 held-out with set3) |
| V2 · gv3 · Qw+wp · ub-ins | `mpc_eval/20260924-212906-evalv2honor-set3` | 09-24 (09-25) | 8 / 16 | `learned-mpc.md` 09-25; archive README | eval-v2-honor |
| V2 · nom · Qw+wp · ub-ins | `mpc_eval/20260924-213353-evalv2honor-nominal` | 09-24 (09-25) | 1 / 3 | `learned-mpc.md` 09-25; archive README | eval-v2-honor |
| BL · rec6 · - · - | `mpc_eval/20260924-223112-replayset-baseline` | 09-24 (09-25) | 6 / 6 | `learned-mpc.md` 09-25; archive README | recorded replay set (nominal/set1/set3 subdirs) |
| V2 · rec6 · Qw+wp · ub-ins | `mpc_eval/20260924-223112-replayset-learned` | 09-24 (09-25) | 6 / 9 | `learned-mpc.md` 09-25; archive README | honor wp + debug channel, recorded |
| BL · gv1 · - · - | `mpc_eval/20260924-234527-flateval-baseline-set1` | 09-24 (09-25) | 12 / 12 | `learned-mpc.md` 09-25; archive README | flat-hold-eval baseline (flat demo) |
| BL · gv1 · - · - | `mpc_eval/20260924-234527-flateval-baseline-set1-rerun` | 09-24 (09-25) | 5 / 5 | `learned-mpc.md` 09-25; archive README | flat-hold-eval baseline re-run gv_04-08 |
| V2 · nom · Qw+wp · ub-ins | `mpc_eval/20260924-234527-flateval-learned-nominal` | 09-24 (09-25) | 1 / 3 | `learned-mpc.md` 09-25; archive README | flat-hold-eval, deploy_v2_flat |
| V2 · gv1 · Qw+wp · ub-ins | `mpc_eval/20260924-234527-flateval-learned-set1` | 09-24 (09-25) | 12 / 24 | `learned-mpc.md` 09-25; archive README | flat-hold-eval; index.json episodes rebuilt 09-30 from the npz (outcomes only) |
| V2 · gv3 · Qw+wp · ub-ins | `mpc_eval/20260924-234527-flateval-learned-set3` | 09-24 (09-25) | 8 / 8 | `learned-mpc.md` 09-25; archive README | flat-hold-eval, deploy_v2_flat |
| BL · rec4 · - · - | `mpc_eval/20260924-234527-flatrec-baseline` | 09-24 (09-25) | 4 / 4 | `learned-mpc.md` 09-25; archive README | flat recorded set |
| V2 · rec4 · Qw+wp · ub-ins | `mpc_eval/20260924-234527-flatrec-learned` | 09-24 (09-25) | 4 / 4 | `learned-mpc.md` 09-25; archive README | flat recorded set, debug channel |
| V2 · stages10 · Qw · ub-ins | `mpc_eval/20260925-002801-stages` | 09-25 | 10 / 10 | `learned-mpc.md` 09-25; archive README | two-fixed-targets (deploy_v2_flat_pp2): 1/10 |
| V2 · stages10 · Qw · ub-ins | `mpc_eval/20260925-002801-stages-s1timeout` | 09-25 | 10 / 10 | `learned-mpc.md` 09-25; archive README | final stage runs to timeout: 2/10 |
| V2 · stages10 · Qw · ub-ins | `mpc_eval/20260925-002801-stages-timeout` | 09-25 | 10 / 10 | `learned-mpc.md` 09-25; archive README | both stages to timeout = live default: 3/10 (2 strict) |
| V2 · eeoff4 · Qw · ub-ins | `mpc_eval/20260925-203831-eeoff/learned` | 09-25 | 4 / 4 | `handoffs/archive/…-part5.md` | EE-offset starts, live params |
| BL · eeoff4 · - · - | `mpc_eval/20260925-203831-eeoff/baseline` | 09-25 | 4 / 4 | `handoffs/archive/…-part5.md` | EE-offset starts |
| V2 · long25 · Qw · ub-ins | `mpc_eval/20260925-long25-gv03` | 09-25 | 1 / 1 | - | gv_03, 35 s episode, long25 params |
| V2 · seat6 · Qw · ub-ins | `mpc_eval/20260926-231837-synthtargets` | 09-26 | 6 / 6 | `…/table.json` | first seat6 run: 4/6 strict |
| V3a · seat6 · Qw · ub-pool | `mpc_eval/20260928-223352-synthtargets-v3` | 09-28 | 6 / 6 | `…/table.md` | PICK A closed loop: 1/6 strict |
| V2 · seat6 · Qw · ub-ins admm2 | `diag/20260929-c3-consistency/closed/admm2_p10/v2` | 09-29 | 3 / 3 | `…/closed/table.md` | E4, paced p10 (3 of 6 cells) |
| V3a · seat6 · Qw · ub-pool admm2 | `diag/20260929-c3-consistency/closed/admm2_p10/v3` | 09-29 | 3 / 3 | `…/closed/table.md` | E4, paced p10 (3 of 6 cells) |
| V2 · seat6 · Qw · ub-ins admm5 | `diag/20260929-c3-consistency/closed/admm5_p5/v2` | 09-29 | 6 / 6 | `…/closed/table.md` | E4, paced p5 |
| V3a · seat6 · Qw · ub-pool admm5 | `diag/20260929-c3-consistency/closed/admm5_p5/v3` | 09-29 | 6 / 6 | `…/closed/table.md` | E4, paced p5 |
| V2 · seat6 · Qw · ub-ins admm10 | `diag/20260929-c3-consistency/closed/admm10_p10/v2` | 09-29 | 6 / 6 | `…/closed/table.md` | E4, paced p10 |
| V3a · seat6 · Qw · ub-pool admm10 | `diag/20260929-c3-consistency/closed/admm10_p10/v3` | 09-29 | 6 / 6 | `…/closed/table.md` | E4, paced p10 |
| V2 · seat6 · Qw · ub-ins | `diag/20260929-sim-response/capture/v2` | 09-29 | 6 / 6 | `…/e3/table.md` | re-run with per-frame snapshots ("rerun" rows) |
| V3a · seat6 · Qw · ub-pool | `diag/20260929-sim-response/capture/v3` | 09-29 | 6 / 6 | `…/e3/table.md` | re-run with per-frame snapshots ("rerun" rows) |
| V3a · seat6 · Qw · ub-ins | `diag/20260929-sim-response/e3` | 09-29 | 6 / 6 | `…/e3/table.md` | E3: V3a with V2's u bounds |
| V4a · seat6 · Qw · ub-ins | `mpc_eval/20260929-v4-retrain/t1` | 09-29 | 6 / 6 | `…/table_v4.md` |  |
| V4b · seat6 · Qw · ub-ins | `mpc_eval/20260929-v4-retrain/t2` | 09-29 | 6 / 6 | `…/table_v4.md` |  |
| V4c · seat6 · Qw · ub-ins | `mpc_eval/20260929-v4-retrain/t3` | 09-29 | 6 / 6 | `…/table_v4.md` |  |
| V4a · yaw12 · Qw · ub-ins | `mpc_eval/20260929-v4-retrain/t1_yaw` | 09-29 | 12 / 12 | `…/t1_yaw/table.md` |  |
| V4a · fs6 · Qw · ub-ins | `mpc_eval/20260929-v4-retrain/t1_freespace/t1` | 09-29 | 6 / 6 | `…/t1_freespace/table.md` |  |
| V4a · fs6 · Qw · ub-pool | `mpc_eval/20260929-v4-retrain/t1_freespace/t1_v3bounds` | 09-29 | 6 / 6 | `…/t1_freespace/table.md` | "T1-v3b" |
| V2 · fs6 · Qw · ub-ins | `mpc_eval/20260929-v4-retrain/t1_freespace/v2` | 09-29 | 6 / 6 | `…/t1_freespace/table.md` |  |
| V3a · fs6 · Qw · ub-pool | `mpc_eval/20260929-v4-retrain/t1_freespace/v3_mix` | 09-29 | 6 / 6 | `…/t1_freespace/table.md` |  |
| V4a · seat6 · QA · ub-ins | `mpc_eval/20260929-metric/t1_metricA` | 09-29 | 6 / 6 | `…/table_metric.md` |  |
| V4a · yaw12 · QA · ub-ins | `mpc_eval/20260929-metric/t1_metricA_yaw` | 09-29 | 12 / 12 | `…/table_metric_yaw.md` |  |
| V4a · seat6 · QB · ub-ins | `mpc_eval/20260929-metric/t1_metricB` | 09-29 | 6 / 6 | `…/table_metric.md` |  |
| V4a · yaw12 · QB · ub-ins | `mpc_eval/20260929-metric/t1_metricB_yaw` | 09-29 | 12 / 12 | `…/table_metric_yaw.md` |  |
| V4c · seat6 · QA · ub-ins | `mpc_eval/20260929-metric/t3_metricA` | 09-29 | 6 / 6 | `…/table_metric.md` |  |
| V4c · yaw12 · QA · ub-ins | `mpc_eval/20260929-metric/t3_metricA_yaw` | 09-29 | 12 / 12 | `…/table_metric_yaw.md` |  |
| V4c · seat6 · QB · ub-ins | `mpc_eval/20260929-metric/t3_metricB` | 09-29 | 6 / 6 | `…/table_metric.md` |  |
| V4c · yaw12 · QB · ub-ins | `mpc_eval/20260929-metric/t3_metricB_yaw` | 09-29 | 12 / 12 | `…/table_metric_yaw.md` |  |
| V4a · yaw12-m15 · QA · ub-ins | `mpc_eval/20260929-metric/t1_metricA_yaw_rep` | 09-29 | 6 / 6 | `…/table_metric_yaw.md` | repeat of the -15 deg half |
| V5a · seat6 · Qw · ub-ins | `mpc_eval/20260929-v5-infonce/t5a/seat` | 09-29 | 6 / 6 | `…/table_seat.md` |  |
| V5a · yaw12 · Qw · ub-ins | `mpc_eval/20260929-v5-infonce/t5a/yaw` | 09-29 | 12 / 12 | `…/table_yaw.md` | urtip6_high_yawp15.crash139 set aside, re-run |
| V5a · fs6 · Qw · ub-pool | `mpc_eval/20260929-v5-infonce/t5a/freespace` | 09-29 | 6 / 6 | `…/table_freespace.md` |  |
| V5b · seat6 · Qw · ub-ins | `mpc_eval/20260929-v5-infonce/t5b/seat` | 09-29 | 6 / 6 | `…/table_seat.md` |  |
| V5b · yaw12 · Qw · ub-ins | `mpc_eval/20260929-v5-infonce/t5b/yaw` | 09-29 | 12 / 12 | `…/table_yaw.md` |  |
| V5b · fs6 · Qw · ub-pool | `mpc_eval/20260929-v5-infonce/t5b/freespace` | 09-29 | 6 / 6 | `…/table_freespace.md` |  |
| V6a · seat6 · Qw · ub-ins | `mpc_eval/20260929-v6-cfm/t6a/seat` | 09-29 | 6 / 6 | `…/table_seat.md` |  |
| V6a · yaw12 · Qw · ub-ins | `mpc_eval/20260929-v6-cfm/t6a/yaw` | 09-29 | 12 / 12 | `…/table_yaw.md` |  |
| V6a · fs6 · Qw · ub-pool | `mpc_eval/20260929-v6-cfm/t6a/freespace` | 09-29 | 6 / 6 | `…/table_freespace.md` |  |
| V6b · seat6 · Qw · ub-ins | `mpc_eval/20260929-v6-cfm/t6b/seat` | 09-29 | 6 / 6 | `…/table_seat.md` |  |
| V6b · yaw12 · Qw · ub-ins | `mpc_eval/20260929-v6-cfm/t6b/yaw` | 09-29 | 12 / 12 | `…/table_yaw.md` |  |
| V6b · fs6 · Qw · ub-pool | `mpc_eval/20260929-v6-cfm/t6b/freespace` | 09-29 | 6 / 6 | `…/table_freespace.md` |  |
| V7a · seat6 · Qw · ub-ins | `mpc_eval/20260930-sdlcs/v1/seat` | 09-30 | 6 / 6 | `…/table_seat.md` | verdict.md |
| V7a · seat6 · Qw · ub-ins | `mpc_eval/20260930-sdlcs/v1/seat_headoff` | 09-30 | 6 / 6 | `…/table_seat.md` | head off (use_state_dependent_lcs absent = fixed B0) |
| V7a · yaw12 · Qw · ub-ins | `mpc_eval/20260930-sdlcs/v1/yaw` | 09-30 | 12 / 12 | `…/table_yaw.md` |  |
| V7a · fs6 · Qw · ub-pool | `mpc_eval/20260930-sdlcs/v1/freespace` | 09-30 | 6 / 6 | `…/table_freespace.md` |  |
| - · press48 · D · - | `data/lcs/contact/press_down/eval` | 09-30 | 12 / 48 | `data/lcs/contact/press_down/eval/press_down_report.md` | open-loop Franka-only press-down set (12 snapshots x 4 variants); model-free, used by gate G2b |
| V4a · seat6 · Qw · ub-tr1 | `mpc_eval/20261001_V4a_seat6_Qw_ub-tr1` | 10-01 | 6 / 6 | `…/table.md` | box control: V4a LCS with box_1 (plan Q6) |
| V8a · seat6 · Qw · ub-ins | `mpc_eval/20261001_V8a_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | V8 iteration 7: 0/6 strict, RMSE 19.9 |
| V8a · yaw12 · Qw · ub-ins | `mpc_eval/20261001_V8a_yaw12_Qw_ub-ins` | 10-01 | 12 / 12 | `…/table.md` | V8 iteration 7: 0/12 strict, RMSE 24.4 |
| V8b · seat6 · Qw · ub-ins | `mpc_eval/20261001_V8b_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | V8 iteration 8: 0/6 strict, RMSE 19.9 |
| V8b · yaw12 · Qw · ub-ins | `mpc_eval/20261001_V8b_yaw12_Qw_ub-ins` | 10-01 | 12 / 12 | `…/table.md` | V8 iteration 8: 0/12 strict, RMSE 25.6 |
| V8c · seat6 · Qw · ub-ins | `mpc_eval/20261001_V8c_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | V8 iteration 9: 0/6 strict, RMSE 19.3 |
| V8c · yaw12 · Qw · ub-ins | `mpc_eval/20261001_V8c_yaw12_Qw_ub-ins` | 10-01 | 12 / 12 | `…/table.md` | V8 iteration 9: 0/12 strict, RMSE 26.9 |
| V10a · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10A_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 2/6 strict, 3/6 engaged, RMSE 21.7 |
| V10a · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10A_R2_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | repeat: 3/6 strict, 4/6 engaged, RMSE 20.9 |
| V10a · yaw12 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10A_yaw12_Qw_ub-ins` | 10-01 | 12 / 12 | `…/table.md` | 0/12 strict, 4/12 engaged (+15°); −15°: 6 unsafe (Franka ~131°), 6 aborts |
| V10a · cell-s30 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10A_long30_urtip6_high` | 10-01 | 1 / 1 | `…/index.json` | urtip6_high × ud+6_r0, stage 2 30 s: strict (wrap 106.2°, h 0.23) |
| V10a · seat6-1goal30 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10A_seatonly30_seat6` | 10-01 | 6 / 6 | `…/*/index.json` | seated goal only: 0/6 strict, 2/6 engaged (gv_05 strict at 6.8 s, then under) |
| V10a · under12 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10A_from_under` | 10-01 | 12 / 12 | `…/*/index.json` | 0/12; all end under (h −9.4 to −12.1); both arms move down |
| V10b · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10B_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, all over (h 18-42), RMSE 22.5 |
| V10b · yaw12 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10B_yaw12_Qw_ub-ins` | 10-01 | 12 / 12 | `…/table.md` | 0/12, all over, safe (rot ≤ 8°), RMSE 36.7 |
| V10c · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10C_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, RMSE 24.6 |
| V10d · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10D_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, RMSE 15.2 |
| V10e · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10E_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, belt pulled off (outside), RMSE 71.2 |
| V10f · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10F_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, RMSE 55.9 |
| V10g · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10G_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, belt pulled off, RMSE 103.4 |
| V10h · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10H_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, 1 unsafe (rot 107°), RMSE 42.2 |
| V10i · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10I_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | seed control: 0/6, RMSE 19.7 |
| V10j · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10J_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, RMSE 20.1 |
| V10k · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10K_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | V10a rerun: 0/6, RMSE 45.5 |
| V10l · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10L_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, belt lifted (h ~127), RMSE 108.0 |
| V10m · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10M_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, closest (h 5.7-9.5, 5 slanted), RMSE 27.1 |
| V10n · seat6 · Qi · ub-ins | `mpc_eval/20261001-v10/20261001_V10N_seat6_Qw_ub-ins` | 10-01 | 6 / 6 | `…/table.md` | 0/6, RMSE 49.0 |
| V2 · cell-s30 · Qw · ub-ins | `mpc_eval/20261001-v10/20261001_V2_long30` | 10-01 | 2 / 2 | `…/*/index.json` | 09-26 params, stage 2 30 s: urtip strict (wrap 142.1°, h 0.03); gv_01 engaged (81.4°, h 1.94) |
| V11b · seat6-rf · Qw · ub-tr1 | `data/lcs/realfirst/v1/iter1/eval` | 10-05 | 6 / 6 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 59.6 vs the demo belt, unsafe 6, solve p95 11 ms |
| V11e · seat6-rf11 · Qw · ub-tr1 | `data/lcs/realfirst/v1_1/iter1/eval` | 10-05 | 6 / 6 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 141.9 vs the demo belt, unsafe 0, solve p95 14 ms |
| V11g · seat6-rf11 · Qw · ub-tr3 | `data/lcs/realfirst/v1_1/iter3/eval` | 10-05 | 6 / 6 | `…/table.md` | V11 loop iteration 3, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 145.0 vs the demo belt, unsafe 0, solve p95 17 ms |
| V11i · seat6-rf11 · Qw · ub-tr5 | `data/lcs/realfirst/v1_1/iter5/eval` | 10-05 | 6 / 6 | `…/table.md` | V11 loop iteration 5, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 121.2 vs the demo belt, unsafe 0, solve p95 17 ms |
| V11i · seat6-rf11 · Qw · ub-tr7 | `data/lcs/realfirst/v1_1/iter7/eval` | 10-05 | 6 / 6 | `…/table.md` | V11 loop iteration 7, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 129.3 vs the demo belt, unsafe 0, solve p95 41 ms |
| V12k · seat6-rf12 · Qw · ub-tr1 | `data/lcs/realfirst/cmp/full/iter1/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 0/6 strict, 1/6 engaged, RMSE 90.5 vs the demo belt, unsafe 0, solve p95 16 ms |
| V12m · seat6-rf12 · Qw · ub-tr3 | `data/lcs/realfirst/cmp/full/iter3/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 3, MPPI 2 stages (seed demo goals): 1/6 strict, 1/6 engaged, RMSE 112.0 vs the demo belt, unsafe 0, solve p95 16 ms |
| V12o · seat6-rf12 · Qw · ub-tr5 | `data/lcs/realfirst/cmp/full/iter5/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 5, MPPI 2 stages (seed demo goals): 1/6 strict, 1/6 engaged, RMSE 80.1 vs the demo belt, unsafe 0, solve p95 30 ms |
| V12q · seat6-rf12 · Qw · ub-tr7 | `data/lcs/realfirst/cmp/full/iter7/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 7, MPPI 2 stages (seed demo goals): 2/6 strict, 2/6 engaged, RMSE 49.3 vs the demo belt, unsafe 0, solve p95 23 ms |
| V12s · seat6-rf12 · Qw · ub-tr9 | `data/lcs/realfirst/cmp/full/iter9/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 9, MPPI 2 stages (seed demo goals): 1/6 strict, 1/6 engaged, RMSE 70.2 vs the demo belt, unsafe 0, solve p95 16 ms |
| V12u · seat6-rf12 · Qw · ub-tr1 | `data/lcs/realfirst/cmp/full2_gate/iter1/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 138.1 vs the demo belt, unsafe 0, solve p95 14 ms |
| V12u · seat6-rf12 · Qw · ub-tr3 | `data/lcs/realfirst/cmp/full2_gate/iter3/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 3, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 136.4 vs the demo belt, unsafe 0, solve p95 14 ms |
| V12w · seat6-rf12 · Qw · ub-tr1 | `data/lcs/realfirst/cmp/v12_18/iter1/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 23.3 vs the demo belt, unsafe 0, solve p95 15 ms |
| V12x · seat6-rf12 · Qw · ub-tr3 | `data/lcs/realfirst/cmp/v12_18/iter3/eval` | 10-06 | 6 / 6 | `…/table.md` | V11 loop iteration 3, MPPI 2 stages (seed demo goals): 0/6 strict, 0/6 engaged, RMSE 20.9 vs the demo belt, unsafe 0, solve p95 15 ms |
| V15a · seat6-rf12 · Qw · ub-tr1 | `data/lcs/realfirst/cmp/v12_35/iter1/eval` | 10-06 | 6 / 16 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 3/16 strict, 3/16 engaged, RMSE 23.2 vs the demo belt, unsafe 0, solve p95 16 ms |
| V15c · seat6-rf12 · Qw · ub-tr3 | `data/lcs/realfirst/cmp/v12_35/iter3/eval` | 10-06 | 6 / 16 | `…/table.md` | V11 loop iteration 3, MPPI 2 stages (seed demo goals): 0/16 strict, 0/16 engaged, RMSE 30.6 vs the demo belt, unsafe 0, solve p95 16 ms |
| V16b · seat6-rf12 · Qw · ub-tr1 | `data/lcs/realfirst/cmp/v12_36/iter1/eval` | 10-06 | 6 / 16 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 4/16 strict, 4/16 engaged, RMSE 12.5 vs the demo belt, unsafe 0, solve p95 14 ms |
| V16d · seat6-rf12 · Qw · ub-tr3 | `data/lcs/realfirst/cmp/v12_36/iter3/eval` | 10-06 | 6 / 16 | `…/table.md` | V11 loop iteration 3, MPPI 2 stages (seed demo goals): 2/16 strict, 3/16 engaged, RMSE 13.6 vs the demo belt, unsafe 0, solve p95 15 ms |
| V16f · seat6-rf12 · Qw · ub-tr5 | `data/lcs/realfirst/cmp/v12_36/iter5/eval` | 10-06 | 6 / 16 | `…/table.md` | V11 loop iteration 5, MPPI 2 stages (seed demo goals): 2/16 strict, 6/16 engaged, RMSE 11.9 vs the demo belt, unsafe 0, solve p95 15 ms |
| V16h · seat6-rf12 · Qw · ub-tr7 | `data/lcs/realfirst/cmp/v12_36/iter7/eval` | 10-06 | 6 / 16 | `…/table.md` | V11 loop iteration 7, MPPI 2 stages (seed demo goals): 3/16 strict, 3/16 engaged, RMSE 11.5 vs the demo belt, unsafe 0, solve p95 15 ms |
| V16j · seat6-rf12 · Qw · ub-tr9 | `data/lcs/realfirst/cmp/v12_36/iter9/eval` | 10-06 | 6 / 16 | `…/table.md` | V11 loop iteration 9, MPPI 2 stages (seed demo goals): 1/16 strict, 1/16 engaged, RMSE 24.3 vs the demo belt, unsafe 4, solve p95 15 ms |
| V18b · seat6-rf12 · Qw · ub-tr1 | `data/lcs/realfirst/cmp/v12_46/iter1/eval` | 10-08 | 6 / 16 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 11/16 strict, 11/16 engaged, RMSE 15.0 vs the demo belt, unsafe 0, solve p95 18 ms |
| V18d · seat6-rf12 · Qw · ub-tr3 | `data/lcs/realfirst/cmp/v12_46/iter3/eval` | 10-08 | 6 / 16 | `…/table.md` | V11 loop iteration 3, MPPI 2 stages (seed demo goals): 11/16 strict, 11/16 engaged, RMSE 17.9 vs the demo belt, unsafe 0, solve p95 17 ms |
| V18f · seat6-rf12 · Qw · ub-tr5 | `data/lcs/realfirst/cmp/v12_46/iter5/eval` | 10-08 | 6 / 16 | `…/table.md` | V11 loop iteration 5, MPPI 2 stages (seed demo goals): 3/16 strict, 5/16 engaged, RMSE 15.9 vs the demo belt, unsafe 0, solve p95 19 ms |
| V18h · seat6-rf12 · Qw · ub-tr7 | `data/lcs/realfirst/cmp/v12_46/iter7/eval` | 10-08 | 6 / 16 | `…/table.md` | V11 loop iteration 7, MPPI 2 stages (seed demo goals): 10/16 strict, 10/16 engaged, RMSE 13.8 vs the demo belt, unsafe 0, solve p95 19 ms |
| V18j · seat6-rf12 · Qw · ub-tr9 | `data/lcs/realfirst/cmp/v12_46/iter9/eval` | 10-08 | 6 / 16 | `…/table.md` | V11 loop iteration 9, MPPI 2 stages (seed demo goals): 5/16 strict, 5/16 engaged, RMSE 13.0 vs the demo belt, unsafe 0, solve p95 18 ms |
| V22a · hard11x2-rf · Qw · ub-tr0 | `data/lcs/realfirst/cmp/v12_57/iter0/cand/eval` | 10-08 | 11 / 22 | `…/table.md` | V11 loop iteration 0, MPPI 2 stages (seed demo goals): 12/22 strict, 12/22 engaged, RMSE 30.0 vs the demo belt, unsafe 0, solve p95 19 ms |
| V22b · hard11x2-rf · Qw · ub-tr1 | `data/lcs/realfirst/cmp/v12_57/iter1/cand/eval` | 10-08 | 11 / 22 | `…/table.md` | V11 loop iteration 1, MPPI 2 stages (seed demo goals): 1/22 strict, 2/22 engaged, RMSE 31.2 vs the demo belt, unsafe 0, solve p95 17 ms |

Fixed 2026-09-30: the empty `mpc_eval/20260924-202837-evalv2-v2full-set3` was deleted (its
re-run is `…-203612`); `mpc_eval/20260924-234527-flateval-learned-set1/index.json` had 0
episodes and was rebuilt from the npz (outcomes only; original kept as `index.json.orig`). `mpc_eval/20260929-v5-infonce/t5a/yaw/urtip6_high_yawp15.crash139`
is a set-aside crash (re-run exit 0).

## 4. Offline-only evaluations

Not closed loop, listed for lookup: open-loop gates and diagnostics in `data/lcs/diag/`
(`20260929-model-vs-mpc`, `20260929-v4-offline`, `20260929-v5-infonce`, `20260929-v6-cfm`,
`20260930-sdlcs` gates G1-G8, `20260930-v1-yawp15-plans`), E4 offline solves
(`20260929-c3-consistency/solves/`), the metric offline tables (`mpc_eval/20260929-metric/offline/`)
and the free-space open-loop gates (`data/lcs/free_space/eval_v3/`). V10 (2026-10-01/02):
`data/lcs/diag/20261001-temporal-latent/` — HILP temporal-distance runs (`runs/hilp_*`,
`sep.md`), the obstacle-aware graph (`graph/`), separation tests (`sep.json`,
`blocked_vs_unblocked*.json`), LCS fits (`lcs_fit*.json`), shortest-path figures (`paths/`)
and the scripts for the lever / policy gates and plan checks (`scripts/`).

## 5. How to use

- In docs and notes, write the ID first and the old name once: "V4a (T1)".
- Quote an evaluation by its label, e.g. "V4a · yaw12 · QA · ub-ins: 6/12 strict".
- Look up paths in `docs/models.yaml` (`models[].checkpoint`, `exports`; `evals[].path`,
  `table`).

## 6. Mapping decisions

- V1 = the 2026-09-24 ablation: it is the only campaign before `sim_belt_v2_20260925`, and
  V1b's export was the model deployed on 2026-09-24 (V2's parent in the data lineage).
- V2 stays letterless as the deployed model; `v2_multistep` is V2b and `V2a` is an alias of V2.
- V6 keeps no letter next to V6a/V6b (as agreed), although the scheme would give it one.
- Tokens added to the agreed list: data `ins1` (v1 data, old action definition), objective `vg`
  (violation gradient to the encoder), limits `ub-hand`/`ub-hand-x0.5`, and `admm3`/`admm1` for
  the 2026-09-24 runs (their default was `admm_iter` 3).
- V7b's `+frz` means the multistep window uses B(z_0) frozen (the head's weights still train).
- V9 is skipped: the 2026-10-01 plan reserves V9a/V9b for the structured physical-state retrain.
  The obstacle-aware graph latent is campaign V10 (`sim_belt_v10_20261001`).
- V10 tokens: objective `gd`, `rec<w>`, `aug30`, `hold<w>`, `2blk` (with `vg`, since the joint
  LCS loss reaches the encoder); cost `Qi`. The encoder (`belt_mlp`) is not a tag field. V10b,
  V10c and V10d share one encoder and differ only in the LCS refit (hold 10 / none / 1).
- V10 eval folders are named `…_Qw_…` by the shared harness; their cost is `Qi`, and the labels
  use `Qi`. Suites `cell-s30`, `seat6-1goal30` and `under12` are added for the 10-01/02 runs.
