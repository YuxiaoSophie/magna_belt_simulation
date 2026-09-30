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
| dynamics | `fixB` / `Bz` | fixed B / state-dependent B(z) = B0 + ΔB(z) |
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
| cost | `Qw` | whitened latent cost (default) |
| | `QA` / `QB` | metric A (learned global Q) / metric B (decoder pullback) |
| | `Qw+wp` | Qw + the `demo_traj.w_p` EE-path cost (2026-09-25; rejected design) |
| limits | `ub-ins` | `u` bounds from insertion data only (for V1b: its own v1 data) |
| | `ub-pool` | `u` bounds pooled over the v3 union (V3a's export bounds) |
| | `ub-hand` / `ub-hand-x0.5` | V1b demo export with 4 UR dims hand-set / × `u_bound_scale` 0.5 |
| | `admmN` | `admm_iter` N, written when not the default 2 (or in the E4 sweep) |

Other controller differences (reference mode, `w_r`, stage tolerances, head off) go in the
note column.

**Future runs.**

- New training campaign: `outputs/sim_belt_v<N>_<date>/` with N the next number (V8 next), one
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

Side runs without an ID (`models.yaml` lists them under V6a/V6b): `t6a_w10` (`trafqf0l`) and
`t6b_w1` (`6n5j9jvb`) stopped at epoch 30; `t6b_w10` (`q25alk7k`) segfaulted at epoch 216 and
was retrained as V6b. Smoke, identity, debug and τ-sweep checkpoints (`smoke/`, `identity*/`,
`sim_belt_v5_20260929/{debug,sweep}/`, V7 `smoke/deploy_smoke_*`) are not models.

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

Checkpoint date folders: V1 and V2 `2026-09-24` (V2's campaign folder is dated 20260925),
V3 `2026-09-28`, V4/V5 `2026-09-29`, V6/V7 `2026-09-30`.

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

V1a, V1c and V6 have no tabled results (V1a/V1c raw numbers: lcs_learning
`outputs/sim_belt_ablation_20260924/eval_results.json`). Sources: one-step/h7 from
lcs_learning `outputs/sim_belt_v{2,3,4,7}_*/eval_table*.md`, `v5`/`v6` `eval_table_v{5,6}.md`;
closed loop from the tables in §3 and `docs/learned-mpc.md` (2026-09-24 to 2026-09-30).

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

Fixed 2026-09-30: the empty `mpc_eval/20260924-202837-evalv2-v2full-set3` was deleted (its
re-run is `…-203612`); `mpc_eval/20260924-234527-flateval-learned-set1/index.json` had 0
episodes and was rebuilt from the npz (outcomes only; original kept as `index.json.orig`). `mpc_eval/20260929-v5-infonce/t5a/yaw/urtip6_high_yawp15.crash139`
is a set-aside crash (re-run exit 0).

## 4. Offline-only evaluations

Not closed loop, listed for lookup: open-loop gates and diagnostics in `data/lcs/diag/`
(`20260929-model-vs-mpc`, `20260929-v4-offline`, `20260929-v5-infonce`, `20260929-v6-cfm`,
`20260930-sdlcs` gates G1-G8, `20260930-v1-yawp15-plans`), E4 offline solves
(`20260929-c3-consistency/solves/`), the metric offline tables (`mpc_eval/20260929-metric/offline/`)
and the free-space open-loop gates (`data/lcs/free_space/eval_v3/`).

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
