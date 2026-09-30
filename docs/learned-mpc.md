# Learned MPC: engineering log

Deploys `lcs_learning`'s trained latent LCS as the MPC model of magna's round-belt assembly
controller: the controller's `kMPC` phase plans with C3+ over the learned model instead of the
classic hand-built LCS, targeting a demonstration episode's states instead of a hand-set goal.
This doc is a day-by-day log of what was tried, what broke, and how it was fixed or left open.
For the wire contract, the params-yaml schema and step-by-step commands, see
`docs/learned-mpc-reference.md`. Where any of this and the code disagree, the code wins.

## Glossary

**Model versions** (lcs_learning `outputs/<run>/`; each `deploy_<m>/` holds deploy.npz,
decoder.npz, learned_lcs.yaml). IDs are `V<campaign>[letter]`; the full registry (tags,
checkpoints, W&B ids, every closed-loop eval with its label) is `docs/models.md` /
`docs/models.yaml`. Older sections use the old names:

| ID | old name | run dir | what it is |
|---|---|---|---|
| V1a / V1b / V1c | both / `decoded_only` / violation_only | `sim_belt_ablation_20260924` | v1 insertion data; V1b was deployed 2026-09-24 |
| V2 (alias V2a) | v2 (`v2_decoded_only`) | `sim_belt_v2_20260925` | insertion data only (464 eps); the deployed model |
| V2b | `v2_multistep` | same | V2 + multistep H7 (mse); failed its pick rule, not used |
| V3a | v3_mix (PICK A, "model A") | `sim_belt_v3_20260928` | v2 recipe on v2 + free-space + approach + contact data (790 eps) |
| V3b | v3_ft (B) | same | v2 epoch 300 fine-tuned on the v3 data (`--init-checkpoint`, 120 epochs, lr 3e-4) |
| V4a | T1 | `sim_belt_v4_20260929` | v3_mix data + decoded next-state loss as RMSE (the loss-scale fix; now the default) |
| V4b | T2 | same | T1 + Δbelt loss + multistep H7 (rmse) + near-pulley ×3 sampling |
| V4c | T3 | same | T2 + latent-metric loss (latent distance ≈ belt RMSE + height term) |
| V5a / V5b | T5a / T5b | `sim_belt_v5_20260929` | T1 + state InfoNCE (whitened, τ 0.1, hard negatives) / + action InfoNCE |
| V6 / V6a / V6b | T6 / T6a / T6b | `sim_belt_v6_20260929` | T1 + CFM-form InfoNCE (σ 1, τ 1, in-batch only): unbounded / std band 0.18-0.35 / std pin 0.35 |
| V7a / V7b | V1 / V2 (2026-09-30) | `sim_belt_v7_20260930` | T1 + state-dependent B(z) = B0 + ΔB(z); V7a one-step, V7b + multistep H7 with B frozen at z0 |

In the 2026-09-30 section, "V1" / "V2" mean V7a / V7b, not the V1 / V2 campaigns.
V4-V7 use insertion-only `u` bounds (`--u-bounds-glob`); V3a's bounds are pooled over all
data. "-v3b" (e.g. T1-v3b) = the same model run with v3_mix's `u` bounds (label `ub-pool`).

**Costs:**
- **current cost:** Σ (z − z_g)ᵀ Q (z − z_g) + uᵀRu with Q = w_q·diag(1/z_std²) (whitened latent)
  and R = w_r·diag(1/half_range²) from the `u` bounds.
- **metric A:** one global learned Q = LᵀL per model, fitted so the quadratic matches
  D² = belt RMSE² + β·Δh² (both states near the seat) + γ·Δwrap², β = 2 (2026-09-29).
- **metric B:** decoder pullback Q = JᵀWJ at the goal latent (J = decoder Jacobian, W up-weights
  the vertical error of near-seat points); no fitting.
- Both are loaded by the opt-in `learned_mpc.use_q_matrix`.

**Task measures:**
- **wrap:** largest arc (deg) of belt bodies seated in the large pulley's groove (|h| ≤ 4 mm and
  radial offset ≤ 5 mm).
- **h:** median height (mm) of belt bodies near the seat, relative to the groove plane.
- **outcome** (`src/round_belt_task/outcome.py`): *engaged* wrap ≥ 60°; *slanted* wrap ≥ 15° or
  a height spread > 8 mm with a seated body; *over* h > +5 mm; *under* h < −5 mm; *outside*
  fewer than 2 bodies near the seat; else *other*.
- **strict:** engaged with wrap ≥ 90° and |h| ≤ 1 mm.
- **D:** task distance to the target belt (belt RMSE plus height/wrap terms as in metric A);
  **d** or "latent distance": the whitened ‖z − z_g‖ the MPC sees; **goal_tol**: p90 of d over
  engaged final frames (exported).
- **belt RMSE:** per-point RMSE over the 150 belt points, mm (some diag tables use the
  per-coordinate form, √3 smaller; tables say which).

**Evaluation sets:**
- **seat cells (6):** flat_engaged × {nominal, gv_01, gv_05}, flat_engaged_urtip × ud+3_r0,
  flat_engaged_urtip6 × ud+6_r0, flat_engaged_urtip6_high × ud+6_r0 (target × start state).
- **yaw cells (12):** the same 6 starts with the targets rotated ±15° about the pulley axis
  (`<target>_yaw{p15,m15}`).
- **free-space goals (6):** held-out free-space frames (twist, bend, bend_lift, stretch,
  wrist_tilt, random_mix), single-stage, from each episode's own start state.
- **stage 1 / stage 2:** the pre-seat goal (demo frame 13) and the seated goal.

**Controller terms:**
- **B0 / ΔB(z):** the fixed part and the network part of the state-dependent input matrix; the
  head is evaluated once per solve and frozen over the 7-step horizon.
- **λ:** LCS complementarity (contact) variables. C3+ runs `admm_iter` 2 ADMM iterations, so the
  planned λ is only loosely complementary; planned states can differ from the model's own
  rollout of the planned actions.
- **no-motion / copy baseline:** predicting the belt does not move; a model should beat it.

**Offline gates** (2026-09-30, `data/lcs/diag/20260930-sdlcs/gates.md`):
- G1: insertion one-step error.
- G2a/G2b/G2c: action-direction agreement at stage-2 onset / at T1's failure states (press-down
  pairs) / predicted vs measured sign.
- G3: contact-region action gain.
- G4: latent consistency.
- G5: stability (ρ(A)) and B size.
- G6: C3 check and solve cost.
- G7: frozen-vs-per-step B mismatch.
- G8: 7-step error.
- The 2026-09-29 diagnostics Q1-Q7 are the same kinds of checks (`data/lcs/diag/20260929-*`).

## Current status (2026-09-30)

**Best controller so far:** `honor_penalize_input_change: true`, `w_r 0.3`,
`demo_traj.w_p 0.03` (an EE-path cost) on the `v2_decoded_only` (V2) model — held-out engaged
22/40 = 0.55 vs the waypoint baseline's 14/28 = 0.50 (`eval-v2-honor`; the 95% CIs overlap, so
this is "on par," not a demonstrated win). The same ranking holds against a flat, UR-held
target (`flat-hold-eval`: matched 6/10 vs 3/10; held-out 17/32 vs 6/13). **The user has
rejected the EE-path cost** (`demo_traj.w_p`) as a design, because it costs the EE pose to the
demo's own path and grasps vary by construction — dropping it (`two-fixed-targets`, `w_p 0`)
collapses engagement to 1/10. So the result above is not adoptable as-is.

**Kept default (worktree `round_belt_controller_params_learned_eval.yaml`):** the latent-only
two-fixed-target setting (flat demo frames 13 / 59, both stages run to their 4 s / 6 s
timeouts, `w_p 0`): 3/10 engaged (2 strict) on the matched starts. Every other yaml is in
`systems/parameters/learned_archive/<date>/`. `admm_iter` stays 2: more ADMM iterations make
the closed loop worse (2026-09-29, E4).

**Current model:** `v2_decoded_only` (V2), W&B run `nsxihz32`, epoch 300, trained on 464
episodes / 33,382 train tuples (`data/lcs/v2/split.json`). It stays the deployed MPC model after
2026-09-30: on the 6 synthetic-target cells it is still the best (strict 4/6).

**v3 (2026-09-28), not adopted:** `v3_mix` (V3a; W&B `rt2j3k45`, the v2 recipe on the union
of v2 + free-space + approach + contact data, 790 episodes / 82,834 tuples,
`data/lcs/v3/split.json`) is better than v2 open loop on every held-out set (e.g. approach
one-step 1.236 vs 8.588 mm) and was the eval pick (PICK A). **In closed loop it is worse**:
strict 1/6 vs v2's 4/6 on the same synthetic-target cells. The 2026-09-29 diagnosis:
prediction accuracy is not the cause. The latent distance the MPC minimises is not a task
distance (v3's scores a belt riding over the pulley about as close to the goal as a seated
one), and C3's 2-iteration plans are not consistent with the model. See 2026-09-29.

**v4 (2026-09-29), not adopted:** three retrains of the v3_mix recipe with a corrected
objective (T1 = V4a: rmse decoded loss; T2 = V4b: + Δbelt, multistep H7, near-pulley
sampling; T3 = V4c: + a latent-metric loss), all with insertion-only `u` bounds. Closed loop
on the same 6 cells: T1 0/6 strict (calm, rests on top of the pulley), T2 0/6 (the Franka
spins), T3 1/6 (spins, penetrates the board). T3 fixes the latent ranking offline, but not
the closed loop.

**Cost metric (2026-09-29), opt-in, not adopted:** replacing the whitened latent cost with a
task-fitted metric (A: a learned global Q; B: the decoder pullback at the goal) ranks states
almost perfectly offline (Spearman ≈ 0.99), but seating stays T1 0/6 (A) / 1/6 (B), T3 0/6.
On yawed targets T1 + A reaches 6/12 (T1 0/12). The metric is not the main bottleneck; the
model's action response near the pulley is. Worktree switch `learned_mpc.use_q_matrix`,
default off.

**v5 InfoNCE (2026-09-29), not adopted:** T1 + a state InfoNCE (T5a = V5a) and + an action
InfoNCE (T5b = V5b). Both are worse than T1 offline and fail everywhere in closed loop (seat
0/6, yaw 0/12, free space ends further from the goal than it started), with spins, grasp losses
and the hand below the board. T5b's LCS is unstable (ρ(A) 1.30). The trainer options stay off
by default.

**v6 CFM InfoNCE (2026-09-29/30), not adopted:** the CFM loss on raw latents (σ = 1, τ = 1,
in-batch negatives, no action term) inflates the latent scale unless bounded; with a std band
(T6a = V6a) or pin (T6b = V6b) the scale holds, but both fail in closed loop (seat 0/6, yaw
0/12 and 2/12, free space ends further from the goal), with spins and the hand below the board.

**v7 state-dependent LCS (2026-09-30), most promising retrain, not adopted:** the T1 recipe
with a state-dependent input matrix B(z) = B0 + ΔB(z) (a z-only tanh MLP, evaluated once per
C3 solve and frozen over the horizon). V1 (V7a, one-step) fixes T1's riding-over sign: the early
stage-2 press goes down (Franka dz −4.4 vs +8.3 mm) and no seat cell ends over. Seat 2/6 strict,
3/6 engaged, RMSE 10.0 mm ("promising", not a pass); but it overshoots to "under" in 2 seat
cells, fails the yaw set (3/12) with a safety failure at +15° (5 grasp losses, Franka spins,
hand below the board) and free space (12.7 mm). V2 (V7b, + frozen multistep H7) inflates B and
destabilises A (ρ(A) 1.264) and was not run in closed loop. All switches are opt-in, off by
default (`docs/learned-mpc-reference.md` §5.13).

**Ranking (end of 2026-09-30):** v2 is the only reliable seater (4/6). V1 is the first retrain
that presses down instead of riding over (2/6 strict), but it is unsafe on the +15° yawed
targets. T1 is safe but rests on top of the pulley; T1 + metric A gives 6/12 on yawed targets.
v3_mix and T1 with v3_mix's `u` bounds lead in free space. T3, T5a, T5b, T6a and T6b are unsafe
(spins, penetration, grasp loss).

**Trainer default (lcs_learning, user decision 2026-09-29):** the decoded next-state loss uses
the `rmse` form by default (it was MSE in m², effectively off next to the RMSE reconstruction).
Every other v4 training option stays opt-in and off by default (`docs/learned-mpc-reference.md`
§5.12).

**Main open issues:** the latent is not task-shaped: whitened latent distance does not rank
riding-over vs seated states (the fix has to reach the closed loop, not only the offline
ranking). Rotation spinning is a separate failure: plans that lean on large Franka/UR rotations
leave the data, where the models are wrong. The demo-tracking progress index stalls/deadlocks
on a large fraction of held-out starts, and the goal tolerance does not separate `engaged` from
`slanted` (the v4 over states also fall inside it). A task-fitted cost metric does not fix
seating. B(z) (V1) fixes the press direction but overshoots, and the latent cost still never
prefers Franka Fz− at the riding-over states. Franka-only pressing does not seat in the sim
(the UR holds its side up), and the training data has few UR-down steps in contact. The
C3 plans stay inconsistent with the model (relaxed λ at `admm_iter` 2), and near the seat the
latent is nearly blind (V1: latent distance 0.52 at 7.7 mm true belt error). Next (none
run): V1 + metric A; V1b (head warm-up so B0 learns); coordinated-press data (UR down / both
arms); orientation-envelope safety constraints for the spinning.

## 2026-09-30 — state-dependent LCS B(z), press-down data, V1 (V7a) closed loop

**Summary:** On 2026-09-29 the model's action response near the pulley was the main
bottleneck: at T1's (V4a's) riding-over end states the fixed-B LCS's lever is a coin flip
(|E|-weighted sign agreement 0.39-0.52) and it prefers Fz+ where Fz− lowers the task distance.
This day made the input matrix state-dependent, B(z) = B0 + ΔB(z), trained in lcs_learning
and swapped into C3 once per solve. Two variants on the T1 recipe: V1 (= V7a, one-step) and V2
(= V7b, + a 7-step loss with the head frozen at z_0, as in deployment). In this section V1 /
V2 always mean V7a / V7b; the deployed model (ID V2) is written v2 or `v2_decoded_only`. By
the offline gate rule the pick is STOP (neither passes G1-G6); V1 is the better variant and
the first model to pass the onset lever gate G2a.
V1 in closed loop (on the user's go) fixes T1's riding-over: the press goes down and no seat
cell ends over. Seat 2/6 strict (T1 0/6), but 2 cells overshoot to "under", the +15° yawed set
is unsafe (5 grasp losses) and free space is 12.7 mm. A paired ±Fz press-down eval set (48
rollouts) confirms the lever sign in the sim, but a Franka-only press never seats: the UR holds
its side up. `v2_decoded_only` stays deployed; V1 is the most promising retrain but is not
adopted. Plan: `handoffs/PLAN-20260930-state-dependent-lcs.md`; gates:
`data/lcs/diag/20260930-sdlcs/{gates,pick}.md`; closed loop:
`data/lcs/mpc_eval/20260930-sdlcs/`; paths and commands: `docs/learned-mpc-reference.md` §5.13.

**Tried:**
- **State-dependent LCS design.** With z ∈ R^16, u ∈ R^12, λ ∈ R^8:

  ```
  h(z)  = tanh(W1 z + b1)             W1: 64×16
  o(z)  = W2 h(z) + b2                W2: 192×64 (+16 rows with d(z))
  ΔB(z) = reshape(o[:192], 16×12)     row-major: o[i·12 + j] = ΔB[i, j]
  z'    = A z + (B0 + ΔB(z)) u + D λ + d (+ Δd(z)),  0 ≤ λ ⊥ E z + F λ + H u + c ≥ 0
  ```

  - z only (never u), so at a fixed z every C3 subproblem is the same QP. λ does not depend
    on B, so the residual sits outside the PGD solve (trainer) and the LCP solve (exporter).
  - Zero-init output layer (the model starts as the fixed-B LCS); W1 from its own seeded
    generator so `lcs_params` init and shuffling match T1. AdamW, weight decay 1e-4, own param
    group at `lr_lcs`. No norm penalty, no d(z) (flags exist, off).
  - Deployment: the controller evaluates ΔB at the `LATENT_STATE` z once per solve and calls
    `C3::UpdateLCS`; the time-invariant LCS is frozen over the 7-step horizon.
  - **V1** (W&B `grn9fisu`): `v4_t1.yaml` (rmse decoded loss, v3_mix data/split, seed 0, 300
    epochs) + `--bz-hidden 64`, one-step only.
  - **V2** (`vo97r4cg`): V1 + multistep H7 (rmse, weight 1/7, warm-up 10, grad to the encoder)
    with B(z_0) frozen over the window (`--multistep-freeze-bz`, default on).
- **Everything is opt-in and off by default; identity proven.**
  - Trainer (`--bz-*`, `--multistep-freeze-bz`): 2 epochs on 12 files vs HEAD `9b7f0c4`,
    defaults / multistep H3 / `--bz-hidden 64 --bz-lr 0 --bz-weight-decay 0` / HEAD rerun, all
    weights, `lcs_params` and metrics max |diff| 0.0.
  - Exporter: the `bz_*` keys appear only when the checkpoint has a head. The v4_t1 re-export
    equals `deploy_t1/` except `exported_at` (yaml 1 line, deploy.npz 59 arrays,
    reference_vectors 18/18).
  - C++ (worktree, `learned_mpc.use_state_dependent_lcs`, absent = false): `learned_lcs_c3_check`
    HEAD vs new on head-free yamls differs only in the `solve mean` lines; a head yaml with the
    flag off equals its stripped twin except one log line and the ref-check line; a zero-W2 head
    on vs off is bitwise identical (17 digits). `bz_ref_check` on the real exports 2.6e-16 (V1)
    / 2.2e-16 (V2) relative.
- **Training** (lcs_learning `outputs/sim_belt_v7_20260930/`, epoch 300, val):

  | run | recon mm | decoded mm | ‖ΔB‖/‖B0‖ p50 / p95 | ‖B(z)‖_F p95 | ‖B0‖_F | ρ(A) | s/epoch |
  |---|---|---|---|---|---|---|---|
  | T1 | 0.346 | 0.436 | - | (48.3) | 48.3 | 1.012 | 18.7 |
  | V1 | 0.334 | 0.421 | 8.41 / 10.22 | 67.8 | 6.46 | 1.011 | 13.9 |
  | V2 | 0.440 | 0.479 | 17.8 / 19.3 | 159.4 | 8.00 | **1.264** | 22.1 |

  - **B0 never learns with a zero-init head.** ‖B0‖_F stays at its random init (5.94-5.96 at
    epochs 20-43, 6.46 at 300; T1 reaches 48.3): Adam moves each of the head's 12k output weights
    at lr 1e-3, so the head takes the fast path and carries the mean B. On 975 tuples:
    ‖B̄‖_F 51.8 (B̄ = B0 + mean ΔB; T1's ‖B‖_F 48.3) and the state-varying part
    ‖B(z) − B̄‖/‖B̄‖ p50 0.42 / p95 0.88. So the planned G5 ratio ‖ΔB‖/‖B0‖ ≤ 1 measures the
    B0/head split, not inflation; G5 was revised to ‖B(z) − B̄‖/‖B̄‖ (B̄ over every v3 frame).
  - **V2 inflates B and destabilises A** once the multistep loss is on (from epoch ~120):
    ‖B(z)‖ p95 159 (3.3× T1), ρ(A) 1.264 with a peak of 1.53 at epoch 210. Its val h7 (1.15 mm)
    beats T2's 1.29, likely bought with the inflation.
- **Freeze mismatch (G7, V1):** h7 decoded RMSE (mm), B frozen at z_0 (deploy) vs re-evaluated
  per step on the predicted latent (the oracle B(z_k true) equals per-step within 0.02 mm):

  | window | frozen | per-step | gap | B0 only |
  |---|---|---|---|---|
  | contact | 2.18 | 2.18 | +0.2 % | 2.48 |
  | approach, contact frames | 2.48 | 2.35 | +5.2 % | 4.62 |
  | contact, riding-over | 2.40 | 2.41 | −0.4 % | 2.75 |
  | approach, riding-over | 2.66 | 2.57 | +3.4 % | 4.76 |
  | insertion (test) | 2.29 | 1.76 | +23.0 % | 8.11 |
  | free space | 0.96 | 0.96 | +0.5 % | 1.97 |
  | approach, free frames | 3.38 | 2.70 | +20.2 % | - |

  Small at contact (≤ 5 %, below the 20 % that would justify V2 there); large on insertion and
  free motion (states that move far in 7 steps). V2's frozen training closes it (≤ 1.6 %).
- **Offline gates** (`data/lcs/diag/20260930-sdlcs/`; plan §6 with the coordinator's G5 and G2b
  adjustments):

  | gate (pass) | v2 | T1 | V1 | V2 |
  |---|---|---|---|---|
  | G1 insertion one-step mm (≤ 0.560) | 0.577 | **0.509** | **0.509** | 0.569 |
  | G2a onset lever agree (≥ 0.80) | 0.764 | 0.750 | **0.814** | 0.742 |
  | G2b end-state kNN agree (≥ 0.70) | 0.62 | 0.50 | 0.60 | 0.53 |
  | G2b press pairs dn10 < up6 at k7 / k20 / end (≥ 5/6 at k7, end) | 6 / 6 / 2 | 4 / 5 / 0 | 5 / 6 / 2 | 6 / 6 / 6 |
  | G2b press pairs dn4 < up6 at k7 / k20 / end | 6 / 6 / 3 | 4 / 5 / 0 | 3 / 5 / 4 | 6 / 6 / 6 |
  | G2c pred vs measured sign, informative pairs at end | 12/27 | 4/27 | 17/27 | 20/27 |
  | G3 contact-onset gain (≥ 0.85) / Q5 cl riding-over (< 1) | 0.816 / 2.48 | 0.847 / 2.16 | 0.811 / 2.19 | 0.819 / 2.54 |
  | G4 insertion e1 / e7 ÷ copy (≤ 0.519 / 0.401) | 0.579 / 0.428 | 0.472 / 0.365 | 0.384 / 0.396 | 0.465 / 0.406 |
  | G5 ρ(A) / ‖B(z)‖ p95 / ‖B − B̄‖/‖B̄‖ p95 (≤ 1.02 / 96.7 / 1) | 1.029 / 31.9 / 0 | 1.012 / 48.3 / 0 | 1.011 / 66.8 / 0.99 | 1.264 / 159.7 / 0.78 |
  | G6 C3 check / ref check / solve on÷off (≤ 1.05) | - | - | 4/4 / PASS / 0.54 | 4/4 / PASS / 1.18 |
  | G8 h7 insertion / contact / free mm (≤ 1.70 / 2.50 / 1.22) | 1.92 / 3.92 / 2.58 | 1.98 / 3.13 / 1.06 | 2.29 / 2.18 / 0.96 | 1.32 / 1.96 / 0.85 |

  - Pick by the rule (V2 if G1-G6 + G8, else V1 if G1-G6, else STOP): **STOP**. V1 fails G2b
    and G3; V2 fails G1, G2a, G2b, G3, G4, G5 and G6. T1 itself fails G2a, G2b and G3; no model
    has passed G3 yet.
  - V1 is the first model to pass G2a with the current cost (offset 0: 0.692 → 0.848). It beats
    T1 on every lever measure, and contact h7 drops 3.13 → 2.18 mm; the contact-onset gain
    regresses (0.847 → 0.811).
  - G6's on÷off compares against a B0-only QP (‖B0‖ 6-8), so it also measures problem
    difficulty; the head's own cost (head eval + `UpdateLCS`) is 0.60 % (V1) / 0.55 % (V2) of a
    flag-on solve.
  - **Which lever lowers D.** At the 6 T1 end states, measured by each model's decoded D over 7
    held steps, the best channel is UR z− (Uz−): V1 4/6, V2 6/6, T1 6/6, v2 6/6. By the latent
    cost no model prefers Fz−: Fz− lowers the latent cost in 0/6 states for T1, V1 and V2 (v2
    1/6) and ranks 16-24 (V1 prefers Fx−, V2 Fy+). This matches the press-down result
    below.
- **Press-down eval set** (`data/lcs/contact/press_down/eval/`, 7747, `--record`). New
  collector family `press_down` (`scripts/lcs/collect_contact_branches.py`): a Franka-only
  min-jerk ramp over 1.5 s to (dz, lateral) = (−4, 0), (−10, 0), (+6, 0) or (−10, ±3) mm, then
  held; the UR holds. 12 snapshots (T1's 6 seat-cell end states, re-captured; 3 held-out
  final_over; 3 held-out first_contact) × 4 = 48/48 ok, 0 grasp losses, 0 board contacts.
  Labels: task distance D (`mc.d2_task`) to the stage-2 goal belt.
  - At the T1 end states the sign is right: dn10 lowers D at k7 6/6 and at the end 5/6, dn4
    lowers it 6/6, up6 raises it 6/6, paired dn10 < up6 6/6 at every k. Mean ΔD per mm: dn10
    −0.155, up6 +0.28.
  - **Nothing seats: 0/48 engaged, wrap 0 everywhere.** A 10 mm Franka press lowers h only
    ~2.5 mm (+12.6 → +10.1) because the UR holds its side up. Pressing with the Franka alone
    lowers D in the right direction but cannot seat.
  - first_contact states (D0 ≈ 64 mm) are near-flat (|ΔD| ≤ 0.6). k1/k3 are small-signal (the
    `u = 0` drift has median 0.09 / p90 0.42 mm).
  - **Data coverage** (coordinator check on the v3 training data, steps with the belt above the
    groove, a loose label: h 5-60 mm, wrap < 60°): in the contact data the UR moves down on
    only 7-10 % of steps (5th percentile −0.5 mm); insertion/approach have 25-35 % UR-down
    steps. So the data barely shows the coordinated press that seating needs.
- **V1 closed loop** (`data/lcs/mpc_eval/20260930-sdlcs/v1/`; current cost: `w_p 0`,
  `admm_iter 2`, no `q_matrix`; V1's insertion-only bounds for seat/yaw, v3_mix's for free
  space; 1 episode per cell; 7745 / 7746). C3 check 22/22 PASS, `bz_ref_check` max rel 2.6e-16.
  30 runs, all exit 0, 0 stale replies.

  | model | strict | engaged | mean RMSE mm | max rot F / UR ° | early F dz mm | min hand z mm | aborts | solve p95 ms |
  |---|---|---|---|---|---|---|---|---|
  | v2 | 4/6 | 6/6 | 6.2 | 28 / 5 | −0.5 | 223 | 0 | 57 |
  | T1 | 0/6 | 0/6 | 13.1 | 26 / 4 | +8.3 | 231 | 0 | 75 |
  | T1 + A | 0/6 | 0/6 | 10.9 | 39 / 8 | +9.4 | 195 | 0 | 76 |
  | **V1** | **2/6** | **3/6** | **10.0** | 32 / 4 | **−4.4** | 193 | 0 | 83 |
  | V1 head off | 0/6 | 0/6 | 381.2 | 180 / 50 | +3.2 | 1 | 5 | 98 |

  - **Seat:** nominal and gv_05 seat strictly (wrap 132° / 146°, RMSE 6.0 / 5.2), urtip6_high
    engages (h −1.8, RMSE 22.1), urtip6 is slanted. No cell rides over any more, but gv_01 and
    urtip overshoot to "under" (h −8.6 / −8.5). Gate: "promising" (≥ 2/6, RMSE < 13.1), not a
    pass (≥ 3/6). Safe: Franka rotation ≤ 37°, 0 aborts.
  - **Yaw (12):** 3/12 strict (T1 0/12, T1 + A 6/12). −15°: 3/6 strict, 4/6 engaged, RMSE
    11.5, Franka rotation 44-51°. **+15°: 0/6 and a safety failure**: 5 grasp losses, Franka
    spins 111-167°, 5 cells "under" and gv_05 slanted, panda_hand z down to −40 mm (below the
    board). Pooled RMSE 23.5 (T1 13.6).
  - **Free space** (v3_mix bounds): mean final / min 12.7 / 7.3 mm (T1-v3b 11.4 / 7.3, v3_mix
    8.4 / 5.5); stretch diverges 8.9 → 22.5 mm, final ≤ start in 5/6. FAIL.
  - **Head off** (the same export with the flag absent, fixed B0): degenerate as expected
    (‖B0‖_F 6.5, B0 ≈ 0 next to B̄): 5 grasp losses, ~180° Franka spins.
  - |ΔB|/|B0| in closed loop, median 7.5 (seat) / 8.1 (yaw) / 10.1 (free space). Solve p95
    median ~83 / 86 / 50 ms (T1 75 / 77 / 26); the V2 export ran on the CPU during seat, yaw
    and free space, so part of it may be load. Head eval + `UpdateLCS` ~57 µs per solve.
- **Yawp15 gv_05 plan diagnosis** (`data/lcs/diag/20260930-v1-yawp15-plans/`, offline: each
  recorded solve re-solved with `learned_lcs_c3_check --diag_stage_file` to recover the planned
  λ; V1 yawp15 vs V1 seat, V1 yawm15 and T1 yawp15, all gv_05):
  - **The planned states are inconsistent with the model** because λ is relaxed at
    `admm_iter` 2 (planned λ +45 % over the exact LCP at late stage 2). At the stage switch
    the step-1 jumps are ~2 cm. The plan ends ~3.5 cm short of the seated target at the
    switch, closing to ~8 mm 0.45 s later and ~3 mm by 0.9 s, while the real belt stays
    ~7.7 mm off and slanted. Late stage 2: decoded plan k7 is 9.1 mm from a static true belt
    (0.25 mm motion); the plan-vs-rollout gap is 36 % of the k7 error² (76 % in stage 1, 93 %
    for T1).
  - **Latent blind zone:** latent goal distance 0.52-0.54 at 7.7 mm true belt error (seat:
    0.66 at 3.4 mm; train pairs at d 0.4-0.7 are 4.4 mm p50, 6.4 mm p90). Rolled Δd with the
    plan's u is −0.31 vs a claimed +0.10 (means), so the MPC stalls.
  - Reconstruction floor 4.2 mm at that state (train p50 0.44 mm, test p95 4.1 mm), but z0 and
    the planned z are in distribution (kNN5 0.76 / 0.67), so it is not decoder garbage.
  - **The head is not the cause:** ‖ΔB‖ 46 (train p5), local relative change 0.14, frozen vs
    per-step B 0.025; T1 without a head plans worse. The model also drifts at rest (`u = 0`
    rollout 0.49 whitened, seat 0.25).
- **One-step prediction on yawed held-out approach data** (coordinator check, 24 episodes,
  mean mm):

  | | V1 | T1 |
  |---|---|---|
  | model | 1.54 | 1.66 |
  | recon floor | 1.50 | 1.63 |
  | no-motion | 0.66 | 0.66 |
  | moving frames: model vs no-motion | 1.46 vs 1.97 | 1.51 vs 1.97 |

  The dynamics add little error on top of reconstruction; the encoder offset dominates.
- **Viewer and tools.**
  - `prediction_video.OneStepModel.step`, `lcp` path (replay episode mode and prediction
    videos), used `B`/`d` for head models, i.e. the near-zero B0. It now uses
    `LearnedLcs.B_at(z)` / `d_at(z)`; checked against the PGD step within 1e-4.
  - `--target-belt` is removed from `replay_viewer.py` and `record_replay_video.py`. It was an
    opt-in workaround from 2026-09-27 for synthetic targets whose goal frames are not demo
    frames, and it made it easy to show the wrong target. Target belts now come from each
    run's recorded `demo_goals.npz` (`pcd_belt_stage`), checked against the recorded
    `demo_goals_sha256` (a mismatch is reported and not drawn), with a fallback to the demo
    frames (`docs/lcm-simulation.md` §10). Demo-traj runs keep the final-demo-frame target.
  - The viewer shows the absolute sim clock; for the V1 yaw runs MPC starts at 45.12 s.

**Issues / bugs -> resolution:**

| symptom | root cause | fix / status |
|---|---|---|
| G5's ‖ΔB‖/‖B0‖ p95 is 10.2 (V1) / 19.4 (V2), far over 1 | B0 stays at its random init (‖B0‖_F ≈ 6) with a zero-init head; the head carries the mean B | G5 revised to ‖B(z) − B̄‖/‖B̄‖ (V1 0.99, passes); **open**: V1b with a head warm-up so B0 learns |
| V2: ρ(A) 1.264 (peak 1.53), ‖B(z)‖ p95 159 | the frozen multistep loss is met by inflating B from epoch ~120 | not adopted, not run in closed loop; V2b (+ `--bz-reg-weight` or `--bz-lr 1e-4`) proposed |
| V2 export E3 fails: `step(100, 1e-5)` vs `z_next_pgd` 1.86e-4 > 1.1e-4 on frame 14 | all D·Δλ (head residual 4e-9): numpy f64 and torch f32 PGD early-stop at different iterations on a slowly converging frame; fails head-free too | accepted as a known mismatch; tolerance not loosened; E4-E9 pass |
| first `check_v1` printed [SKIP] and exited 0 | relative deploy path in the export runner | `realpath -m` before `cd`; re-run, E0-E9 PASS |
| G6 on÷off 1.18 for V2 (gate ≤ 1.05) | flag off is a different QP (B0 only) | head eval + `UpdateLCS` measured at 0.55-0.60 % of a solve; pick unchanged |
| V1 overshoots to "under" (2 seat cells, 5/6 +15° cells) | not isolated; the yawp15 gv_05 diagnosis shows plans inconsistent with the model (relaxed λ) and a latent blind to the last ~8 mm | **open** |
| V1 +15° yaw: 5 grasp losses, Franka spins 111-167°, hand z −40 mm | not isolated | **open**; V1 not adoptable |
| Franka-only press lowers D but never seats (0/48) | the UR holds its side up; contact data has few UR-down steps (7-10 %) | **open**: coordinated-press data (UR down / both arms) |
| replay episode mode / prediction videos showed wrong one-step predictions for head models | `OneStepModel.step` (`lcp`) used the fixed `B`/`d`, i.e. B0 | uses `B_at(z)` / `d_at(z)`; matches the PGD step within 1e-4 |
| the viewer could show the wrong target belt | `--target-belt` was a manual override | removed; recorded demo_goals with a SHA check and a fallback |
| T1's seat episodes end at 14.55 s, not 16 s | T1 episodes finish early | press-down capture window 14.0-14.6 s, last frame used |
| `bz_ref_z` has 6 rows, not 4 as planned | one per outcome class on the v3 globs | C++ reads K from the rows |

**Decisions:**
- Keep `v2_decoded_only` deployed. V1 is the most promising retrain but is not adopted (the
  +15° safety failure, the "under" overshoot, free space); V2 is not adopted (unstable A).
- All B(z) switches stay opt-in and off by default: the trainer's `--bz-*` (head off at
  `--bz-hidden 0`), the exporter's `bz_*` keys (only when the checkpoint has a head), the
  worktree's `learned_mpc.use_state_dependent_lcs` (absent = false).
- (coordinator) The V2 E3 failure is accepted as a known PGD early-stop mismatch; the tolerance
  is not changed.
- Demo-traj runs keep the final-demo-frame target in the viewer.
- The `external/newton` pointer was committed at `2bc2f63` (fork `hien/viser-opaque-batched`).
- c2's identity artifacts and synthetic yamls were moved out of the worktree to
  `data/lcs/diag/20260930-sdlcs/c2_identity/`; the worktree archive holds only eval yamls.

**Open / next:**
- **V1 + metric A** (recommended next): V1 fixes the press direction, metric A helped T1 on the
  yawed targets (6/12).
- **V1b:** a head warm-up (`--bz-warmup-epochs`) or a lower `--bz-lr`, so B0 learns the mean B
  and the head only the state dependence.
- **Coordinated-press data:** press-down branches with the UR down or both arms, since a
  Franka-only press never seats.
- A viewer layer for the model rollout of the plan's `u` (next to the planned belts), to see
  plan-vs-model inconsistency directly.
- Lower priority: a latent-dimension analysis (the blind zone), V2 + a norm penalty, and an
  ADMM re-test on V1.

## 2026-09-29 — why v3_mix controls worse, v4 retrain, cost-metric test, InfoNCE ablation

**Summary:** A root-cause pass on the 2026-09-28 result (v3_mix (V3a) predicts better but
controls worse) found that prediction accuracy is not the cause. The latent distance the MPC
minimises is not a task distance, and the trainer's decoded next-state loss was effectively off
(m² next to an RMSE in m). C3's 2-iteration plans are inconsistent with the model, but making
them consistent (more ADMM iterations) makes the closed loop worse. Three models (v4 T1-T3 =
V4a-V4c) were retrained with a corrected objective and insertion-only `u` bounds. T3 fixes the
latent ranking offline; none beats v2 in closed loop (T1 0/6, T2 0/6, T3 1/6 strict vs v2 4/6).
Extra tests: T1 on ±15°-yawed targets (0/12) and single-stage free-space goals (v3_mix > T1 >
v2). Two follow-ups: a task-fitted MPC cost metric (A: global Q = LᵀL; B: decoder pullback)
ranks states at Spearman ≈ 0.99 offline but seats no better (T1 0/6 A, 1/6 B; T3 0/6), though
T1 + A reaches 6/12 on the yawed targets. So the metric is not the main bottleneck; the model's
action response near the pulley is. An InfoNCE ablation (v5: T5a = V5a state, T5b = V5b state
+ action) is worse than T1 offline and unsafe in closed loop (0/6 seat, 0/12 yaw). The deployed
model stays `v2_decoded_only` (V2). Diagnosis summary: `data/lcs/diag/20260929-report.md`;
paths and commands: `docs/learned-mpc-reference.md` §5.12.

**Tried:**
- **Replay viewer episode mode** (committed): `scripts/replay_viewer.py --episode/--episodes`
  replays LCS dataset episodes with the one-step LCS predicted belt and a per-episode RMSE plot
  (`docs/lcm-simulation.md` §10).
- **Offline model-vs-MPC diagnosis** (`data/lcs/diag/20260929-model-vs-mpc/`, q1-q6, no sim).
  Ranked findings:
  1. v3's latent distance d tracks global belt RMSE and ignores belt height (|h| coefficient
     0.04 vs v2's 0.20). The riding-over state 1 s into stage 2 (h 12-14 mm) has v3 d 2.3-2.5,
     equal to the seated states of v2's runs; v2 gives it 4.9-5.3. The v3 MPC hits a false
     minimum.
  2. On riding-over states every model's 7-step prediction is worse than no-motion (v3 6.9 vs
     1.4 mm), with ~1 mm/step drift at `u = 0`, so recovery can't be planned.
  3. v3's MPC used the UR yaw/pitch/roll channels (+2.0σ Urz), the ones v3 models worst (onset
     cos vs kNN 0.2-0.4); latent-vs-empirical lever ρ 0.02 (v2 0.42).
  4. States go out of distribution after about 2 s (belt NN 7.9 mm, 30 % beyond the held-out
     p99 latent kNN).
  5. The LCS is affine near the pulley in every model. One-step accuracy is not the cause.
- **Training-code audit** (read-only, `scripts/train_joint_pointnet_lcs.py`). Reconstruction is
  RMSE (m) but the decoded next-state loss was MSE (m²), both at weight 1.0: its gradient is
  ~2·RMSE ≈ 6e-4 of reconstruction's, i.e. effectively off. The LCS fit uses detached z
  (`enable_violation_grad_to_encoder` false), so the latent gets no dynamics or control
  shaping. There is no Δz/Δbelt loss and `multistep_horizon` is 0 (deploy horizon 7). The
  exporter's `u` bounds are the 0.5/99.5 % of the pooled data, and R = 1/half_range², so v3's
  wider UR rotation range was also 10-16× cheaper. v2 and v3 configs differ only in data
  files; mean |Δbelt| per tuple fell from 2.06 mm (v2) to 1.26 mm (v3).
- **Sim-response diagnostics E1-E3** (`data/lcs/diag/20260929-sim-response/`): 12 closed-loop
  re-runs with per-frame snapshots (reproduce Phase 6: v2 4/6, v3 1/6 strict), 600 open-loop
  probes (E1), 72 plan-vs-zero-u rollouts (E2), and v3 with v2's `u` bounds (E3).
  - Early in stage 2, v3's metric scores the lifted route and v2's successful route alike
    (+1.94 vs +1.97 per 7 steps); v2's metric prefers the good one (+1.94 vs +1.20). v3's LCS
    predicts the lift correctly (h +8.0 predicted / +7.3 actual): this is not model
    exploitation.
  - v2's latent carries a strong Franka-yaw term (3-step yaw probe −0.216 vs v3 −0.013, belt
    RMSE effect ≈ 0) that keeps the arm near the demo pose; v3's does not.
  - C3 `x_sol` is inconsistent with its own model for both models: the plans claim 3-4× the
    decrease that rolling `u_sol` through the same LCS gives.
  - E3: v3 with v2's bounds is still 1/6 strict. The early lift persists (+12-27 mm in 0.5 s);
    the late drag-off disappears (end RMSE 13-21 vs 60-114 mm).
- **E4, C3 plan consistency** (`data/lcs/diag/20260929-c3-consistency/`; opt-in
  `--diag_stage_file` and cost overrides in the worktree's `learned_lcs_c3_check`). Offline
  on 2235 recorded solve states, more ADMM iterations close the plan-vs-rollout gap (early
  stage 2, v3: 2.72 at admm 2, 1.36 at 10, 0.65 at 20). In closed loop, paced to the admm-2
  solve latency, it gets worse:

  | model | admm 2 | admm 5 | admm 10 |
  |---|---|---|---|
  | v2 strict | 4/6 | 4/6 (2 grasp losses) | 0/6 (4 grasp losses) |
  | v3_mix strict | 1/6 | 0/6 | 0/6 (grasp lost on 4/6) |

  At admm 2 the relaxed λ acts as a free drift toward the goal, so plans use small `u`.
  Consistent plans must earn the decrease with `u`, so they saturate the rotation bounds
  (u0 rotation channels at a bound: 0-6 % at admm 2, 43-79 % at admm 10), the arm spins
  off the data (max Franka rotation 113-179°) and the grasp is lost. admm 5 and 10 also miss
  the 75 ms budget (closed-loop solve median 62-86 / 106-163 ms).
- **v4 retrain** (lcs_learning `outputs/sim_belt_v4_20260929/`): the v3_mix yaml, data, split,
  seed and 300 epochs, from scratch, with new opt-in trainer flags. Loss weights come from a
  grad-norm calibration at v3.
  - T1 (W&B `1niukyhl`): `--decoded-next-state-loss-form rmse` only.
  - T2 (`gwz0d9jk`): T1 + Δbelt loss (weight 1), multistep H7 (rmse form, weight 1/7, grads to
    the encoder, warm-up 10) and near-pulley sampling K 3 (the near label covers 66 % of train
    tuples, 85.5 % once sampled).
  - T3 (`1q2gt22n`): T2 + latent-metric loss 0.02.

  All three are exported with insertion-only `u` bounds (`--u-bounds-glob`, equal to v2's
  exactly); z stats and `goal_tol` still use the full v3 globs (`goal_tol` 5.40 / 5.09 / 5.49).
  E0-E8 pass for all three. Offline (mm; Q metrics are the mean of 4 targets unless fe):

  | offline | v2 | v3_mix | T1 | T2 | T3 |
  |---|---|---|---|---|---|
  | insertion test one-step / h7 (G1 ≤ 0.635) | 0.577 / 1.92 | 0.551 / 1.88 | **0.509** / 1.98 | 0.711 / **1.42** | 0.643 / 1.93 |
  | free-space / approach / contact one-step | 2.02 / 9.61 / 1.74 | **0.58 / 1.27 / 1.07** | 0.63 / 1.55 / 1.30 | 0.74 / 1.86 / 1.64 | 0.75 / 1.71 / 1.34 |
  | free-space / approach / contact h7 | 2.58 / 13.0 / 3.92 | **1.03 / 1.95 / 1.79** | 1.06 / 2.36 / 3.13 | 1.08 / 2.13 / 2.16 | 1.68 / 2.34 / 2.15 |
  | Q4 abs-h coef (fe drivers) | 0.198 | 0.041 | −0.043 | −0.060 | **0.337** |
  | Q4 d(over 1 s into s2) / d(v2-run) | 2.08 | 0.99 | 0.90 | 1.05 | **1.79** |
  | Q4 over states inside goal_tol | **0.010** | 0.610 | 0.745 | 0.720 | 0.670 |
  | Q4 AUC d over>engaged (fe, beltR 0-10) | 0.892 | 0.969 | 0.938 | 0.923 | **1.000** |
  | Q4 improved over v3_mix (≥ 2 of abs-h, ratio, goal_tol share) | - | - | no (0/3) | no (1/3) | **yes (2/3)** |

  Offline ranking: T3 > T2 ≈ T1. T3 is the only variant whose latent separates riding-over
  from seated (|h| coefficient above v2's, 1 s ratio 1.79), at the cost of one-step accuracy
  (0.643, just over G1) and free-space h7. T2 has the best insertion h7 but no geometry gain
  and an ill-conditioned LCS. T1 alone improves insertion one-step but hurts the held-out sets
  and the geometry.
- **v4 closed loop** (`data/lcs/mpc_eval/20260929-v4-retrain/t{1,2,3}/`): the 6 Phase 6 cells,
  latent goals re-encoded per model, `w_p 0`, `admm_iter 2`, floors unchanged, each model's
  insertion-only bounds. All 18 runs exit 0, no stale replies:

  | model | strict | engaged | mean RMSE mm | max rot F / UR ° | early F dz mm | min hand z mm | aborts |
  |---|---|---|---|---|---|---|---|
  | v2 | 4/6 | 6/6 | 6.2 | 28 / 5 | −0.5 | 223 | 0 |
  | v3_mix | 1/6 | 1/6 | 79.8 | 136 / 45 | +15.9 | −22 | 1 |
  | T1 | 0/6 | 0/6 | 13.1 | 26 / 4 | +8.3 | 231 | 0 |
  | T2 | 0/6 | 0/6 | 132.7 | 178 / 46 | −15.6 | 144 | 0 |
  | T3 | 1/6 | 1/6 | 90.4 | 124 / 12 | +16.6 | −55 | 1 |

  T1 is calm and never spins, but every cell ends over or slanted (h +8.8 to +13.5 mm) resting
  on top of the pulley, and its latent reads that as about the goal (end d ≈ 2.0 < goal_tol
  5.4). T2's Franka spins ~180° and the belt comes off. T3's only success is gv_01; it shows a
  v3-like early lift and a Franka spin of 111-161°, with the hand below the board top in 3
  cells (penetration) and one grasp loss. Ranking: v2 > T1 > v3 ≈ T3 > T2.
- **T1 on yawed targets** (±15° about the large-pulley axis, 12 runs,
  `data/lcs/mpc_eval/20260929-v4-retrain/t1_yaw/`). New yawed start states and synthetic
  targets `data/lcs/synthetic_targets/<t>_yaw{p15,m15}/`, built with the same recipes. All 8
  targets seated and held in QC (achieved yaw +14.8 to +15.1 / −14.4 to −14.9°). Result: strict
  0/6 at both yaws, as unyawed. +15°: 4 over / 2 slanted, belt RMSE 12.9-15.1 mm. −15°: RMSE
  6.3-17.2 mm, with a v3-like early lift (F dz +12.6 to +14.2 mm). No board penetration, no
  aborts.
- **Free-space single-stage goals** (`data/lcs/mpc_eval/20260929-v4-retrain/t1_freespace/`):
  one latent goal per free-space family from a held-out episode frame, started from that
  episode's own start state, 9 s, 1 episode per cell. Results (mean final / min belt RMSE to
  the goal, mm; mean % of the gap closed final / min; cells within 5 mm):

  | model | final / min | % closed final / min | within 5 mm |
  |---|---|---|---|
  | v3_mix | 8.4 / 5.5 | 46 / 66 | 3/6 |
  | T1 with v3_mix's `u` bounds | 11.4 / 7.3 | 41 / 61 | 2/6 |
  | T1 | 16.0 / 8.7 | 17 / 51 | 1/6 |
  | v2 | 23.4 / 10.0 | −55 / 43 | 0/6 |

  None reaches 2 mm. Free space reverses the insertion ranking: v3_mix, trained on free-space
  data, tracks best. Every model reaches its minimum early (usually 0.7-2 s) and then drifts
  away for the rest of the 9 s (latent d at the end > its minimum in 18/18). T1 bend_lift and
  v2 bend_lift/stretch diverge (Franka rotation 63-92°). With v3_mix's bounds, T1 improves on
  every goal. The bounds are rarely hit (u0 within 1 % of a bound on 1-5 % of solves), so the
  gain comes from the cheaper input cost R, not from unclipping.
- **Cost-metric test** (`data/lcs/mpc_eval/20260929-metric/`): does a task-shaped MPC cost fix
  seating without retraining? Task distance D² = per-point belt RMSE² + β·Δh² (both states
  within 30 mm of the seat) + γ·Δwrap², β = 2, γ = 25/90². Two metrics for T1 and T3:
  - **A**, one global learned Q = LᵀL per model, fitted on train-split pairs (120k, near-seat
    oversampled) and 40k triplets: log-space regression of zᵀQz to D², a ranking hinge, and a
    ridge toward the current diag(1/z_std²). β swept 0 / 0.5 / 2 / 8; β 2 deployed.
  - **B**, the decoder pullback Q = JᵀWJ at the goal (J the decoder Jacobian; W up-weights the
    pulley-axis component of the belt points within 30 mm of the seat, β_z 6.26). It is poorly
    conditioned: cond 2e3-6e4 (T1) and 5e4-7e5 (T3), vs 95 / 33 for A.

  Each metric is scaled so the median stage-start cost equals the current Q's; `goal_tol_q` is
  the engaged-final p90 in that metric. Offline (β 2; current Q in brackets):

  | model | metric | held-out pair Spearman vs D | near-seat Spearman | lever sign agreement | cond(Q) |
  |---|---|---|---|---|---|
  | T1 | current | [0.920] | 0.952 | 0.750 | 53 |
  | T1 | A | 0.984 | 0.990 | 0.785 | 95 |
  | T1 | B | - | 0.973 | 0.789 | 2.1e3-6.3e4 |
  | T3 | current | [0.922] | 0.962 | 0.777 | 20 |
  | T3 | A | 0.995 | 0.994 | 0.804 | 33 |
  | T3 | B | - | 0.970 | 0.844 | 5.0e4-7.2e5 |

  Closed loop (7741, the 6 Phase 6 cells and the 12 ±15° yawed cells, 1 episode each, no stale
  replies; mean final belt RMSE in mm):

  | model | metric | seat strict | seat RMSE | yaw strict (12) | yaw RMSE | aborts |
  |---|---|---|---|---|---|---|
  | T1 | current | 0/6 | 13.1 | 0/12 | 13.6 | 0 |
  | T1 | A | 0/6 | 10.9 | 6/12 (−15°: 5/6, +15°: 1/6) | 9.9 | 0 |
  | T1 | B | 1/6 (gv_01) | 17.6 | 3/12 | 22.9 | 0 |
  | T3 | current | 1/6 | 90.4 | - | - | 1 |
  | T3 | A | 0/6 | 44.8 | 0/12 | 69.1 | 2 (yaw; hand z −22 mm) |
  | T3 | B | 0/6 | 43.0 | 0/12 | 42.6 | 0 |

  A repeat of T1 + A at −15° reproduced 5/6. Both metrics roughly halve T3's RMSE (less spin,
  no penetration), but none seats. Why seating still fails (`offline/cl_analysis.md`,
  `offline/lever_end.md`):
  - The true D already rates T1's riding-over end states as 87-94 % done (D² end/onset
    0.06-0.13), and they stay inside `goal_tol` in every metric. The over-end / v2-success d
    ratio grows only to 2.6 (A) / 2.3 (B) from 1.7 (true D: 4.4).
  - At those states the LCS + metric lever is a coin flip (|E|-weighted sign agreement 0.39-
    0.52). With the current Q, T1's model prefers Fz+ in 6/6 end states, which raises the true
    D (+0.17 to +0.81 mm); Fz− lowers it in every one. The metric-best step lowers true D in
    0/6 (current), 2/6 (A), 4/5 (B).
  - Even the true D evaluated on the model's decoded predictions reaches only 0.69-0.78 sign
    agreement, and its best step lowers true D in 3/6, 2/6 and 5/5 end states.

  Conclusion: the model's action response near the pulley, not the cost metric, is the main
  bottleneck. The metric matters at the margin (T1 + A on the yawed cells).
- **Worktree `learned_mpc.use_q_matrix`** (opt-in, default off): the LCS yaml gains
  `q_matrix`, `q_matrix_stage1` / `q_matrix_stage2` (both or none; override `q_matrix`) and
  `goal_tol_q`. Q = w_q·M, goal distance sqrt(dzᵀ M dz), per-stage Q swapped at stage switches.
  Default path verified: `learned_lcs_c3_check` on v4_t1_flat_engaged and
  v4_t3_flat_engaged_urtip6 is identical to the HEAD binary except the timing lines.
- **v5 InfoNCE ablation** (lcs_learning `outputs/sim_belt_v5_20260929/`; diag
  `data/lcs/diag/20260929-v5-infonce/`; closed loop `data/lcs/mpc_eval/20260929-v5-infonce/`).
  The v4 T1 recipe (data, split, seed, 300 epochs) plus a contrastive term on the predicted
  next latent ẑ = LCS(z_t, u_t):
  - score −‖Δ⊘σ‖²/τ with σ the per-dimension latent std, τ 0.1.
  - **State term (T5a, T5b):** ẑ vs encode(o_{t+1}) against in-batch negatives, an episode
    hard negative (a frame 3-10 steps from t+1 in the same episode) and a branch hard negative
    (the t+1 frame of a sibling branch from the same contact snapshot, only after the whitened
    actions diverge; 40 sibling pairs in contact/v1/train). Near-duplicate negatives (belt RMSE
    and both EE shifts < 0.5 mm) are masked.
  - **Action term (T5b only):** ẑ vs LCS(z_t, u_j) for up to 32 in-batch alternative actions
    at least 0.5 σ_u away, scored at encode(o_{t+1}).
  - **Weights** by calibration at T1's epoch 300 (encoder grad = 0.5 × recon's): 0.0065 for
    both terms. At NCE start the NCE gradient is ~1e5 × recon's, so an early calibration is
    unusable.
  - **σ:** the spec's detached EMA σ diverged from scratch at every τ (latent std ran away to
    393 / 86 at τ 0.03 / 0.1 and collapsed to the 1e-6 clamp at τ 0.3; the decoder went
    constant by epoch 13-14). State-only on a subset collapsed too, so it is not the action
    term: the lagging σ lets the encoder set the effective temperature by rescaling z. Fix:
    opt-in `--nce-sigma batch` (this batch's std, with grad; scale-invariant). τ swept 0.03 /
    0.1 / 0.3 with batch σ; 0.1 picked (steady climb, best val recon).
  - Training identity with the NCE weights at 0 vs HEAD: max |diff| 0.

  Offline (`eval_table_v5.md`, `summary.md`; mm):

  | offline | T1 | T3 | T5a | T5b |
  |---|---|---|---|---|
  | insertion one-step / h7 | **0.509 / 1.98** | 0.643 / 1.93 | 0.722 / 2.42 | 1.192 / 21.25 |
  | contact one-step / h7 | 1.295 / 3.13 | 1.344 / **2.15** | 1.393 / 2.36 | 2.289 / 27.78 |
  | Q4 abs-h coef | −0.043 | **0.337** | −0.133 | −0.007 |
  | Q4 over states inside goal_tol | 0.745 | 0.670 | **0.293** | 0.296 |
  | Q4 AUC d over>engaged (beltR 0-10, mean of 4) | 0.932 | **1.000** | 0.729 | 0.825 |
  | latent e7 ÷ copy7 (insertion) | **0.36** | 0.58 | 0.82 | 1.84 |
  | ρ(A) / ‖B‖ | 1.01 / 29 | - | 1.008 / 32 | 1.30 / 95 |
  | C3 check (4 targets) | 4/4 | 3/4 | 3/4 | 4/4 |

  Both lose belt accuracy. Fewer over states fall inside `goal_tol` (0.29 / 0.30 vs 0.745),
  but the latent ranks over vs engaged worse (AUC 0.73 / 0.83 vs 0.93). T5b's LCS is
  unstable: ρ(A) 1.30 with 7 eigenvalues outside the unit circle, `u = 0` hold drift 16 mm in 7
  steps (true 0.11), not a solver artefact (PGD-25 vs exact LCP 3.8e-4). Its whitened ‖B‖ is
  inflated about 10-18× vs T1: ‖B/z_std‖ 2034 vs 110 (training agent); a separate check on 10
  contact + insertion files gives Frobenius norms v3_mix 325, T1 225, T5a 405, T5b 2229, and
  mean latent std 0.15 / 0.27 / 0.18 / 0.08.

  Closed loop (current cost, latent goals only, `w_p 0`, admm 2; seat/yaw with each model's
  insertion-only bounds, free space with v3_mix's; 1 episode per cell; mean final belt RMSE mm):

  | model | seat strict | seat RMSE | yaw strict | yaw RMSE | free space final / min | aborts seat / yaw / fs | min hand z mm (all runs) |
  |---|---|---|---|---|---|---|---|
  | v2 | 4/6 | 6.2 | - | - | 23.4 / 10.0 | 0 / - / 0 | 141 |
  | T1 | 0/6 | 13.1 | 0/12 | 13.6 | 16.0 / 8.7 | 0 / 0 / 0 | 219 |
  | T1 + A | 0/6 | 10.9 | 6/12 | 9.9 | - | 0 / 0 / - | 195 |
  | T5a | 0/6 | 82.9 | 0/12 | 94.9 | 125.9 / 12.4 | 2 / 8 / 2 | −67 |
  | T5b | 0/6 | 141.1 | 0/12 | 118.6 | 64.7 / 15.1 | 6 / 11 / 3 | −81 |

  (T1 free space with v3_mix's bounds: 11.4 / 7.3; v3_mix 8.4 / 5.5.) Both are unsafe: Franka
  spins of 70-180°, grasp losses (T5b loses the Franka grasp at 4.7-5.5 s in every seat cell),
  T5a's finger tip below the plate at +15°, UR IK misses, the hand below the board. Every T5
  free-space cell ends further from its goal than it started.
  **Diagnosis:** the action term's negatives are generated by the model itself, so it can win
  by inflating B (spreading LCS(z, u_j) apart) rather than by predicting better; the per-batch
  σ removed the scale exploit but left the latent scale free (T5b's latent std shrinks while B
  grows). **Proposed fixes, not run:** real branch counterfactuals only as action negatives;
  soft, outcome-aware labels instead of hard negatives; a 7-step and a Δ accuracy anchor; a
  fixed latent scale (variance floor or a norm layer); a spectral guard on A.
- **Comparison with Yan et al. 2020 (CFM, arXiv:2003.05436).** CFM scores
  h = exp(−‖z₁ − z₂‖²) with no τ or normalisation, uses in-batch negatives only (127), has no
  decoder, and its forward model is an MLP that outputs a linear map applied to z_t; 8-d
  latent, random pick-and-place data, 1-step sampling MPC over 100 actions. T5a matches its
  core state loss. Ours adds the action term, σ whitening and τ, the recon / LCS losses, the
  fixed-B LCS and the 7-step C3 planner. Their ablation found a pure linear forward model worse,
  which our fixed-B LCS is closer to. Possible next step: a faithful T6 (raw exp(−‖Δ‖²),
  in-batch negatives only, a modest recon anchor).
- **v6 CFM-faithful InfoNCE (T6, T6a, T6b = V6, V6a, V6b; run overnight into 2026-09-30)**
  (lcs_learning `outputs/sim_belt_v6_20260929/`; diag `data/lcs/diag/20260929-v6-cfm/`; closed loop
  `data/lcs/mpc_eval/20260929-v6-cfm/`). T1 + state InfoNCE with `--nce-sigma none` (σ ≡ 1),
  τ 1, in-batch negatives only, action weight 0; state weight 7.1e-3 (NCE encoder gradient
  0.5× recon at T1 epoch 300).
  - **T6 (unbounded)** stopped at epoch 25: the mean latent std went 0.18 (epoch 10) → 0.88
    (epoch 11) → 3.10 (epoch 25), still rising; LCS violation 100× T1's at the same epoch.
    Raw exp(−‖Δ‖²) keeps lowering the loss as every latent is scaled up (the softmax sharpens),
    so nothing bounds ‖z‖. A floor alone cannot stop growth.
  - **Bounded:** opt-in `--latent-std-weight` with `--latent-std-band LO,HI` or
    `--latent-std-pin F` on the per-dim batch std of z. T6a: band 0.18-0.35, weight 1 (σ held
    ≈ 0.33 to epoch 300). T6b: pin 0.35, weight 10 (weight 1 drifted to 0.38; σ held ≈ 0.354;
    the first weight-10 run segfaulted at epoch 216 with no traceback and was retrained).

  | offline | T1 | T5a | T6a | T6b |
  |---|---|---|---|---|
  | insertion one-step / h7 mm | 0.509 / 1.98 | 0.722 / 2.42 | 0.825 / 3.62 | 0.834 / 3.13 |
  | contact one-step / h7 mm | 1.295 / 3.13 | 1.393 / 2.36 | 1.711 / 4.60 | 1.627 / 3.05 |
  | Q4 \|h\| coef / over inside goal_tol | −0.043 / 0.745 | −0.133 / 0.293 | 0.225 / 0.187 | 0.014 / 0.229 |
  | latent e7 ÷ copy7 (insertion) | 0.36 | 0.82 | 1.19 | 1.01 |
  | ρ(A) (# \|eig\| > 1) | 1.012 (3) | 1.008 (2) | 1.067 (11) | 1.060 (10) |
  | whitened ‖B/z_σ‖₂ | 110 | 275 | 147 | 145 |
  | C3 check (4 seat targets) | 4/4 | 3/4 | 3/4 | 4/4 |

  | closed loop | seat strict | seat RMSE mm | yaw strict | yaw RMSE mm | free space final mm | aborts seat / yaw / fs | min hand z mm (seat) |
  |---|---|---|---|---|---|---|---|
  | T1 | 0/6 | 13.1 | 0/12 | 13.6 | 16.0 (11.4 with v3_mix bounds) | 0 / 0 / 0 | 231 |
  | T6a | 0/6 | 95.8 | 0/12 | 109.9 | 69.8 | 5 / 11 / 5 | 25 |
  | T6b | 0/6 | 80.7 | 2/12 | 66.4 | 65.6 | 6 / 11 / 4 | −22 |

  The bound fixes the scale and T6a even ranks height (|h| coef 0.225), but latent consistency
  is worse than copy at 7 steps, A has 10-11 eigenvalues above 1, and the closed loop is unsafe
  like T5a/T5b (Franka rotation up to 99° / 156° in seat cells). T6b's 2 yaw seats come from
  otherwise violent runs. Across five InfoNCE variants, contrastive latent shaping has not
  helped this LCS + C3 setup.

**Issues / bugs -> resolution:**

| symptom | root cause | fix / status |
|---|---|---|
| v3_mix predicts better than v2 but controls worse (1/6 vs 4/6) | the whitened latent distance is not a task distance: v3's latent scores a riding-over belt about as close as a seated one, and nothing in training shapes it (reconstruction only, dynamics loss detached, decoded loss effectively off). v2's success partly rests on an incidental Franka-yaw term | retrained with a corrected objective (v4); **open**: T3 fixes the offline ranking, not the closed loop |
| the decoded next-state loss had no effect on training | MSE in m² next to a reconstruction RMSE in m, both weight 1.0: gradient ≈ 6e-4 of reconstruction's (the multistep loss is m² too) | new `--decoded-next-state-loss-form` / `--multistep-loss-form {mse,rmse}`; `rmse` is the decoded loss's default after the user's decision (below) |
| C3 plans claim 3-4× the goal-distance decrease their own `u_sol` gives | 2 ADMM iterations with relaxed λ; λ acts as free drift toward the goal | **kept**: more iterations make plans consistent but saturate the rotation bounds and spin the arm (E4). `admm_iter` stays 2 |
| T2/T3 export failed: "reference z not reproducible 1.9e-06" | float32 batch-order noise scales with \|z\| (~3e-7 × max\|z\|); the T2/T3 latents are ~10× larger than v3's | exporter and `check_latent_encoder.py` E3 tolerances scaled by max(1, max\|z\|), unchanged for \|z\| ≤ 1; re-exported, E0-E8 pass |
| `learned_lcs_c3_check` fails on T2 ×4 (OSQP IterationLimit, solve 54-73 ms) | the T2 LCS is ill-conditioned: cond(F) 201 vs 7-30 for v2 / v3 / T1 / T3 | **open**; run anyway: T2 spins ~180° in closed loop |
| `learned_lcs_c3_check` fails on T3 urtip6_high | stage-2 bound violation 1.4e-6 vs tol 1e-9 | tolerance only; run anyway |
| T1 ends over / slanted in 6/6 cells | the latent still reads the over state as the goal (end d ≈ 2.0 < goal_tol 5.4; 0.745 of over states fall inside goal_tol offline) | **open** — the latent must be task-shaped |
| T2 / T3 / v3_mix spin the Franka (T2 ~180°, T3 111-161°, v3_mix up to 136°) and lose the belt, the grasp or the board clearance | plans lean on large Franka/UR rotations, where every model is wrong | **open**; proposed: orientation-envelope safety constraints (not run) |
| urtip −15° yawed target failed tension at the default stepping floor (wrap 81 at step 1) | wrap 81° at step 1, under the default stepping floor | re-ran tension/observe with `--step-min-wrap-deg 75`; verify still requires wrap ≥ 90 |
| free-space runs reach their minimum in 0.7-2 s, then drift away | not isolated (λ drift or model bias at rest) | **open**; next: compare with a `u = 0` hold |
| a task-fitted cost metric (A / B) ranks states at Spearman ≈ 0.99 but seating stays 0/6 / 1/6 | T1's riding-over end states are already 87-94 % done in the true D, and the model's lever there is a coin flip (prefers Fz+, Fz− lowers D) | **open**: needs contact data and a model fix, not a cost fix; `use_q_matrix` kept opt-in |
| `learned_lcs_c3_check` fails on t3 urtip6_yawm15 metric A | bound violation 3.3e-6 | tolerance only; run anyway |
| the spec'd InfoNCE (detached EMA σ) diverges at every τ: latent std runs away or collapses, decoder goes constant | the lagging σ lets the encoder set the effective temperature by rescaling z | new opt-in `--nce-sigma batch` (per-batch std with grad); default stays `ema` |
| T5b's LCS is unstable (ρ(A) 1.30, 7 \|eig\| > 1, 16 mm drift at `u = 0` in 7 steps) | the action term's model-generated negatives are separable by inflating B; the per-batch σ leaves the latent scale free | **open**; fixes proposed, not run (above) |
| first T5a export flagged `no_input_normalisation False` | the exporter's source grep matched NCE comments / `.std(` in the trainer | comments reworded (math identical, identity re-run 0/0/0); re-exported, old export in `superseded/` |
| T5a C3 check 3/4 seat, 9/14 yaw/fs; T5b 11/14 | bound-tolerance violations ≤ 6.4e-5, plus OSQP IterationLimit for T5a | run anyway, as T2/T3 |
| t5a `urtip6_high_yawp15` segfaulted at sim init (exit 139) | not isolated | set aside as `*.crash139`, rerun exit 0 |

**Decisions:**
- Keep `v2_decoded_only` deployed; none of v3_mix, T1, T2, T3 is adopted.
- Keep `admm_iter` 2. If consistency is revisited, it needs a larger/structured input cost or
  tighter rotation bounds with it, or a train-time fix so the LCS does not rely on λ drift.
- (user) The `rmse` form of the decoded next-state loss becomes the lcs_learning trainer's
  default. The other options stay opt-in and off by default: `--delta-belt-weight`,
  `--multistep-horizon` / `--multistep-loss-form`, `--sample-weight-near-pulley`,
  `--latent-metric-weight`, and the exporter's `--u-bounds-glob`. Identity (12 v2 files, 2
  epochs): new defaults vs HEAD + sqrt, and `--decoded-next-state-loss-form mse` vs HEAD, both
  max |diff| 0; the exporter without `--u-bounds-glob` reproduces v3_mix's bounds exactly.
  Configs that enable the decoded loss without the key (v2_decoded_only, v2_multistep,
  v3_mix, v3_ft, ablation both / decoded_only) now train with rmse on re-run.
- T2 and T3 ran in closed loop despite their `learned_lcs_c3_check` failures (coordinator).
- Keep `learned_mpc.use_q_matrix` opt-in, default off; no metric-A/B yaml is the default.
- Do not adopt T5a / T5b. The InfoNCE trainer options stay opt-in and off by default
  (`docs/learned-mpc-reference.md` §5.12); `--nce-sigma batch` is the only stable setting.
- Do not adopt T6a / T6b. `--nce-sigma none` needs `--latent-std-weight` with a band or pin;
  the latent-std regulariser stays off by default.

**Open:**
- The latent must be task-shaped. The latent-metric loss (T3) fixes the ranking offline, but
  not in closed loop. Ideas, none run: a latent whose distance tracks steps to the goal; letting
  the LCS consistency gradient reach the encoder.
- Rotation spinning is a separate failure: orientation-envelope safety constraints on both
  arms are proposed, not run.
- `goal_tol` (engaged p90) grew to 5.1-5.5 for v4 and admits over states; a stricter goal
  percentile is worth a look.
- The T2 LCS conditioning (cond(F) 201).
- Free-space drift after the early minimum; repeats per cell (every cell here is 1 episode,
  so these are sensitivities, not rates).
- The model's action response near the pulley: at riding-over states it prefers Fz+ where Fz−
  lowers the task distance. Next: contact data (press-down from riding-over states, branches),
  then a retrain with a fixed dynamics objective.
- InfoNCE fixes, none run: real branch counterfactuals only, soft outcome-aware labels, a
  7-step and Δ accuracy anchor, a spectral guard on A. The faithful CFM loss with a bounded
  latent scale (T6a/T6b) was run and failed; next is a state-dependent LCS (B(z)), planned in
  `handoffs/PLAN-20260930-state-dependent-lcs.md`.

## 2026-09-27/28 — data diversity, v3 training, OOD gates and the closed-loop re-test

**Summary:** v2 (464 insertion episodes, all along one waypoint family) was nearly blind in
free space to grasp-axis translation, stretch/slack and wrist yaw (displacement gain 0.06-0.28,
model ≈ recon, ~1 mm static offset), and had seen contact only along the nominal approach. The
data was diversified three ways: deformation-rich free-space primitives; approaches to the
large pulley from varied yaw, with continuous `place_3` sampling, post-place contact tails and
the UR holding the belt; and contact-rich rollouts branched from snapshots, including recovery.
Two models were trained on the union (A `v3_mix`, B `v3_ft`) and gated open loop (G1-G5).
A was picked, then re-tested in closed loop on the 2026-09-26 synthetic-target cells.
**Result: negative.** A beats v2 open loop on every new held-out set, but in closed loop it
reaches strict 1/6 against v2's 4/6. The deployed model stays `v2_decoded_only`. Collector
flags: `docs/lcs-data-collection.md` §4.5-4.7. Sources and split: `docs/lcs-dataset.md` §10.

**Tried:**
- **Deformation-rich free space** (`collect_motion_primitives.py`, new opt-in flags and the
  families twist / bend / bend_lift / stretch_cycle / franka_sweep / ur_sweep /
  random_mix_holds / wrist_tilt / hold_only). Per-step cap 3 mm / 25 mrad on both arms: the
  cap probe met the realised-vs-`u` slope gate at 3 mm, so smaller caps were not run. 6 runs:
  train 89 ok / 15,689 tuples, held-out 20 ok / 3,692 tuples. Causality gate [0.8, 1.1] met in
  every table (pooled train 0.949-1.046). Stretch gain max 1.99 %, grasp held, no board
  contact. 33 train episodes have a stretch lobe (grasp separation +≥ 8 mm), 52 reach ≥ 5 mm
  non-rigid deformation. Holds: `u == 0` exactly, mean belt drift ≤ 0.93 mm.
- **Rotated and varied approaches** (`collect_lcs_dataset.py --approach start --place3-mode
  continuous --tail 4,6 --hold-ur-gripper`). The offline screen passed 182/189 approach
  settings; sim probes A (approach) 62/62, B (`place_3`) 30/30 and C (tails) 12/12 ok. Usable
  box: yaw [-30, 30]°, elev [0, 30] mm, offset [-10, 10] mm, tilt [-4, 0]°. 16 start states
  (train a01-a12, held-out h01-h04). Runs: train 120/120 + top-ups 38/40, held-out 23/24 +
  top-ups 7/8, ≈ 161 frames per episode. Tails (train 158): lift_repress 44, tension_release 42,
  groove_slide 41, random_contact 15, press_deeper 10, partial_pullout 6. The contact-frame
  fraction was 1.0 in every tail. Labels were recorded at `place_3` and at the end.
- **Contact branching** (`collect_contact_branches.py`, trimmed per D10). 48 new source
  episodes gave 98 train / 18 held-out validated snapshots. Working set: 40 (first_contact 10,
  partial 10, 5 per final label). Rollouts: train 80/80 ok (rim_press 9, top_cross 9,
  pullout_reseat 10, groove_slide 10, recover 15, random_contact 27), held-out 17/18 ok. Strict
  contact-frame fraction 91.2 % / 89.0 %. Strict contact share of the union's train frames:
  57.0 % (target 35 %).
- **Deliberately excluded:** force, stretch and tension features, and force/actuation signals.
  The hardware has no F/T sensor and the UR10 is position-controlled, so the model gets
  neither as an input.
- **Training** (lcs_learning `outputs/sim_belt_v3_20260928/`). Union split: 790 episodes /
  82,834 tuples (v2 insertion 464 / 33,382, free space 88 / 15,484, approach 158 / 25,969,
  contact 80 / 7,999). A `v3_mix` is the v2 recipe on the union, 300 epochs, W&B `rt2j3k45`.
  B `v3_ft` fine-tunes from `nsxihz32` (new `--init-checkpoint`), 120 epochs, W&B `3y5b4zac`.
  Both exports pass E0-E8 and share the same, wider `u` bounds (UR rz up to 15.4 mrad, where
  v2 had ±3.5-4.1).
- **Open-loop gates** (`eval_motion_primitives.py --out` × 21, `summarize_v3.py`). Pooled
  motion one-step, mm:

  | set | v2 | A `v3_mix` | B `v3_ft` | no-motion |
  |---|---|---|---|---|
  | v1 primitives | 1.098 | 0.239 | 0.242 | 0.215 |
  | v2-big | 2.125 | 0.457 | 0.444 | 0.185 |
  | v2-big fast | 2.594 | 0.612 | 0.579 | 0.741 |
  | free-space held-out, nominal | 1.777 | 0.254 | 0.294 | 0.259 |
  | free-space held-out, set3 | 2.802 | 1.133 | 1.161 | 0.205 |
  | approach held-out | 8.588 | 1.236 | 1.322 | 0.751 |
  | contact held-out | 1.841 | 1.100 | 1.109 | 0.249 |

  | gate | rule | v2 | A | B |
  |---|---|---|---|---|
  | G1 | v2 insertion test one-step ≤ 0.635 mm | 0.577 PASS | 0.551 PASS | 0.583 PASS |
  | G2 | displacement gain ≥ 0.90 on every family | min 0.04 FAIL | 0.83 FAIL (1 of 30) | 0.89 FAIL (1 of 30) |
  | G3 | model < no-motion on ≥ 50 % of motion tuples | FAIL | FAIL (deform 24.8 %) | FAIL (deform 21.1 %) |
  | G4 | held-out recon ≤ 0.6 mm | FAIL | FAIL (deform 0.564, contact 1.028) | FAIL (0.609, 1.064) |
  | G5 | approach / contact one-step ≤ 0.8 × v2's | — | PASS (0.14 / 0.61) | PASS (0.15 / 0.62) |

  **PICK = A `v3_mix`**: the mean deform / approach / contact motion one-step is 0.976 mm,
  against B's 1.020 and v2's 4.201. A and B fail G2 only on v2-big `both_sideways`. G3 and G4
  fail because of a static encoder offset on unseen starts (set3, h01-h04, set3 snapshots):
  held-out recon is 3-5× the train-file recon, so one-step ≈ recon. The dynamics-only error is
  0.14-0.34 mm. On approaches v2 blows up at |yaw| > 20° (13.9 mm), while A stays flat at
  1.14-1.41 mm across yaw bins.
- **Videos** (`data/lcs/free_space/videos_v3/`): 13 PICK clips + `all_pick.mp4`, plus 2 v2
  reference clips (`docs/lcm-simulation.md` §10).
- **Closed loop** (`data/lcs/mpc_eval/20260928-223352-synthtargets-v3/`). PICK A in the same
  6 cells as the 2026-09-26 v2 run: the same goal frames re-encoded, and the 2026-09-26 params
  with only `lcs_file` changed (`u_bound_scale 1`). `learned_lcs_c3_check` passed, as did
  `check_mpc_harness.py` H0-H6. 1 episode per cell, so these are sensitivities, not rates:

  | target | start | v2 (strict) | v3 `v3_mix` (strict) |
  |---|---|---|---|
  | flat_engaged | nominal | engaged (y), wrap 117.0°, h +0.00 | over (n), wrap 0, h +54.4 |
  | flat_engaged | gv_01 | engaged (n), wrap 82.0°, h +2.80 | engaged (y), wrap 127.7°, h +0.02 |
  | flat_engaged | gv_05 | engaged (y), wrap 117.2°, h +0.01 | slanted (n), wrap 0, h +10.4 |
  | urtip | ud+3_r0 | engaged (n), wrap 112.2°, h −2.43 | slanted (n), wrap 0, h +10.9; Franka grasp lost 12.9 s |
  | urtip6 | ud+6_r0 | engaged (y), wrap 116.3°, h +0.04 | over (n), wrap 0, h +9.3 |
  | urtip6_high | ud+6_r0 | engaged (y), wrap 127.3°, h +0.58 | over (n), wrap 0, h +11.7 |

  Strict (wrap ≥ 90 and |h| ≤ 1): v3 1/6, v2 4/6. v3 encodes the seated targets about 2×
  better than v2 (re-encode belt RMSE 0.45-1.12 vs 1.84-2.05 mm), and stage 1 ends as well as
  v2's (belt RMSE to the `pre_place_2` goal 6.9 vs 6.6 mm).

**Issues / bugs -> resolution:**

| symptom | root cause | fix / status |
|---|---|---|
| twist / bend episodes failed the crop rule (0 of 4 ok), even at 0.2× amplitude | the gripper rotations lifted the belt > start + 2 mm | these families add a carrier drop (both arms −z, 0.5× amp) and deform in τ ∈ [0.2, 0.8] |
| a retry left `random_mix_holds` failing 5× at the same amplitude | scaling the requested amplitude is a no-op where the per-step cap binds | in the new mode a retry scales the capped plan (`row.retry_mul`); the legacy retry is unchanged |
| twist non-rigid deformation stayed at 2.9-3.0 mm (target 5) | the rot group sat on the 25 mrad cap | `--motion-s-mul twist=2`: rot group scale 1.0, max 6.7 mm; ≥ 5 mm in only 2/7 train episodes (accepted) |
| `free_space/v1/nominal_random/episode_0010`: Franka 6.2 mm RMS, 29 mm peak | a tracking excursion mid-motion (no OSC warning, grasp ok) | excluded from training (it still enters the export's `u` bounds / z stats) |
| `uv run` rewrote `uv.lock` | warp-nn git source via the dirty `external/newton` | reverted; use `uv run --frozen` |
| any UR tangent < 0 at `place_3` timed out 8/8 | the held UR stalls 6-12 mm short of `pre_place_2` | UR box tangent [0, 10] mm (`--place3-box-ur`) |
| approach tilt +4 / +8° never wraps | the belt lands "under" | usable tilt range [-4, 0]° |
| engaged share train 20.3 / 22.8 %, held-out 20.0 / 16.7 % (`place_3` / end), under the 30 / 25 % targets | band yield at the approach states is 30 %; the elev-30 / tilt ≈ -4 settings never engage | **accepted as is** (user); near-nominal engagement comes from the v2 insertion data |
| labels "under" and "outside" below 8 % | "outside" is unreachable while the UR holds the belt | accepted |
| the literal `first_contact` rule fired ~50 mm above the pulley | `n_neighbour` counts bodies in the radial band only (no height term) | first contact = `n_neighbour ≥ 3 AND h_min ≤ 10 mm`, or wrap > 0 |
| `recover` seats only 1/15 (held-out 0/6); over/under never recover | failure finals sit ~1-2 mm from the demo EE poses; lifting pulls the partly seated belt out, and it lands on top again | accepted as a coverage fact |
| contact held-out causality u_x 0.751 (gate 0.8) | the held UR lags along the grasp axis while the belt is loaded on the pulley | accepted as physical, not mislabelling |
| held-out one-step > copy-last-frame on all three new held-out sets; G3/G4 fail | static encoder offset on unseen starts (held-out recon 0.55-1.3 mm vs 0.18-0.38 on train files) | **open**; the dynamics-only error is 0.14-0.34 mm |
| `check_prediction_video.py --scan` P2 failed on every PICK video | the check always rebuilt the v2 model | new `--deploy` / `--decoder` |
| no `groove_slide` / `pullout_reseat` episode in `contact/v1/heldout` | that set holds only random_contact (11) and recover (6) | the video uses a train episode, named `contact_groove_slide_TRAIN`; flagged |
| v3 closed loop: 5/6 cells end over / slanted, end RMSE 64-119 mm | at stage-2 onset the plan lifts the Franka (+17.5 mm z in 0.5 s; v2 +0.6), turns the UR (rz +77 mrad, later rx +100-126 mrad, inside the wider bounds) and counter-yaws the Franka, so the belt rises to h 21-30 mm and rides over the pulley. The encoder is not the cause: along v2's trajectories v3's d(z_target) falls to 0.7-1.0. The dynamics are: on its own closed-loop states v3 barely beats no-motion early in stage 2 (1.8-2.4 vs 2.7-2.8 mm), and its LCS gives Franka yaw almost no lever on d(z_target) (±0.004 per bound step, v2 −0.093 / +0.064) | **open**; next: v3 with v2's `u` bounds, B `v3_ft` in the same cells, multi-step / closed-loop-aware training |

**Decisions:**
- D1: keep the crop top at 0.11 m.
- D2: start small. Free space ≈ 90 train + 22 held-out; approaches 120 + 24 plus engaged-band
  top-ups (≤ 40 / ≤ 8); contact ≈ 80 + 20. Scale up only if the eval shows a gain.
- D3: per-step caps 3 mm / 25 mrad on both arms.
- D4: keep the soft-pulley v2 insertion split (the 0.577 mm reference).
- D6: 10 % insertion regression margin, so G1 is test one-step ≤ 0.635 mm.
- D7: the UR holds the belt through `place_3` and the tail (`--hold-ur-gripper`). Because v2
  episodes released the UR at `place_3`, `outcome_place3` is the label comparable with v2's
  mix; end labels are reported separately (caveat accepted by the user).
- D8: contact snapshots come from new source episodes.
- D9: two training configs, A `v3_mix` and B `v3_ft`. The capacity variant was dropped.
- D10 (user): trim contact branching to what tails cannot give. Keep mid-descent branching and
  recovery; drop `lift_repress` and `tension_release`, which the tails cover.
- D11 (user): per-arm `place_3` wide box depth −6..+20 mm, lateral ±10 mm, roll ±10°,
  yaw ±15°; 40 % engaged-band share; targets ≥ 30 % / ≥ 25 % engaged, with top-ups.
- Gates G1-G5 as tabled above. Pick = the lowest mean held-out motion one-step among G1
  passers.
- Excluded `nominal_random/episode_0010`; no free-space held-out top-up. Kept every
  approach `top_up` row. Never train on `contact/source/*` (no point clouds).
- Accepted (user) the approach engaged shortfall as is, without further top-ups or re-probes.
  Accepted the contact phase as is.
- Ran the closed loop with PICK A, since G1 passed. The deployed model stays
  `v2_decoded_only`. `v3_mix` is not a drop-in replacement despite its better open-loop
  numbers.

## 2026-09-26 — stiffer pulley anchors (sim default change)

The pulleys' VBD axle anchors are now pinned at 1e7 N/m / N m/rad
(`defaults.VBD_RIGID_JOINT_*_KE`, `simulation._pin_pulley_joint_stiffness`); before, legacy AVBD
held them at ~300 N/m and the large pulley bobbed 1.6 mm z / 0.8 mm xy under belt load (now
< 0.01 mm after a 65 ms restore transient). The belt's rod joints are unaffected. **All existing
data (v1/v2 collections, `v2_decoded_only`), the demos (`demo_flat`, `demo_flat_pp2`) and the
synthetic target states were produced with the soft anchors** and were not regenerated. A fresh
UR-held nominal collection (as `demo_flat`) gave 3/3 engaged, wrap 126°, h 0.02-0.04 mm, slant
3.70° (the soft run: 1/3 at wrap 126° / slant 3.65°, 2/3 at wrap 85° / h 2.4 mm / slant 5.8°).

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
