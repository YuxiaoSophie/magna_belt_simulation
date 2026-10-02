# Learned latent LCS + C3 MPC for belt seating: improvement plan (2026-10-01)

Plan of record for improving the learned latent LCS + C3+ MPC so that it seats the round belt
into the large pulley robustly across grasp points, grasp slant/roll, initial EE poses and target
yaw, and so that the state the MPC plans over captures the belt's behaviour under contact. It
draws on every campaign since 2026-09-22 (V1 through V8; IDs per `docs/models.md`). Planning only;
sim pinned at newton 79aa32ba for the whole round. User constraints honoured throughout: no
EE-pose targets in any MPC cost; belt/latent goals only; safety bounds allowed; C3+ stays the
controller (the model stays affine in u at a fixed state).

## How this plan was made

Four architect lenses wrote independent proposals and then argued them out over five rounds:
A representation, B dynamics/MPC co-design, C data/training/evaluation, D skeptic/alternatives.

- Round 1: four proposals from the four lenses (diagnosis, ranked fixes, gates, non-goals).
- Round 2: each lens rebutted the other three (agreements, disputes, concessions, conflicts).
- Round 3: a synthesis architect wrote a consolidated draft; all four critiqued it (24 blocking
  items, every disputed fact re-checked against the code and the day logs).
- Round 4: revised plan resolving every blocking item; verification by all four (B ACCEPT;
  A, C, D BLOCK on specific new items).
- Round 5: targeted fixes; final verdicts **ACCEPT from all four lenses**, with the non-blocking
  amendments folded into this published version.

The round documents (proposals, rebuttals, critiques, verifications, drafts R3/R4/R5) live in
`handoffs/rebuttal-20261001/` (local, git-ignored). Sources checked: `docs/learned-mpc.md`
(09-24 → 09-30, Current status), `docs/models.md`, `docs/learned-mpc-reference.md`,
`data/lcs/diag/20260929-report.md`, `data/lcs/diag/20260930-sdlcs/{gates,g2_lever_end}.md`,
`data/lcs/diag/20260930-v1-yawp15-plans/table.md`,
`data/lcs/contact/press_down/eval/press_down_report.md`, `data/lcs/onpolicy/v8/learning_curve.md`,
`handoffs/RUN-STATE-20261001-v8.md`, lcs_learning `lcs_model.py`, `dataset_round_belt_tuples.py`,
`train_joint_pointnet_lcs.py`, magna `learned_lcs_c3.cc`, `src/round_belt_task/outcome.py`.

## Key disagreements and how they were resolved

| disagreement | positions | resolution |
|---|---|---|
| proprio in the encoder / goal | A, C, D: out (V2's 4/6 is an EE-pose target through the encoder); B: keep for dynamics | Out of the goal and the encoder; EE translations stay in the C3 state with zero cost; p_rel may enter the dynamics (A_p) if the Markov pre-check says it is needed |
| structured state vs opaque latent + readout vs full keypoints | A: 16-d hybrid; B: readout on z; D: 48–80-d keypoints | Analytic physical state (per-bin height + per-bin surface gap + centroid + radial) plus 4 free dims, n_x 20; keypoints only after the oracle; readout as an ablation |
| λ / contact-mode supervision | A: per-bin gaps; B: global over/seated/under modes with a negative-gap hinge; D: analytic gap functions | Per-bin pulley-surface gap tied to the predicted next-step gap (B's hinge is infeasible under the trainer's `relu(−slack)²`); analytic-gap-only init as the fallback |
| orientation envelope form and width | B: ±45° state box; A, C: ±60° (the −15° successes used 44–51°); B later: input-bound "lite" | θ-state box ±60° / ±20°, clamped to x_0 on violation; lite only if the QP budget forces it |
| cost-to-go / demo subgoals | B: terminal cost-to-go and subgoals from demos; A, C, D: EE-path cost in disguise | Dropped from demo data; only on pulley-frame coordinates with off-demo successes, after the oracle |
| oracle sampling MPC vs scripted press search | D: oracle first; A, B, C: scripted search first | Scripted `press_both` + `lift_shift_lower` first; oracle on four pre-registered triggers, always before a third retrain or any wave-3 training |
| baselines with or without the envelope | C: 7 h of baselines first; B: after the envelope | With envelope + physical R; dev grid first, `final` last; the press search runs while the C++ is built |
| on-policy / DAgger loops | C: now, encoder included; A, B, D: after the state fix | Only after V9 exists and a success from a riding-over state has been found; stop at 0/24 |
| seat6 vs a held-out grid as the bar | A, B, D: seat6 with repeats; C: grid with CIs | Grid-final with cell-level exact paired tests; seat6/yaw12 legacy rows only |
| stage completion signal | B: decode(z) → h, wrap; A: measured belt points | Belt features from the encoder node's own belt points, published to the controller; latent completion switched off |
| is one-step error a selector | C: Spearman ≈ 0 (retracted: +0.37, driven by unstable models) | Floor only; V9 gates in coordinate units |

## 1. Executive summary

**Diagnosis:**
1. The latent is a reconstruction code with an accidental metric: 61–75 % of riding-over states
   fall inside `goal_tol` (V3a/V4a); the |h| coefficient is ±0.04; V7a's latent distance to its
   own target is 0.54 at 7.7 mm true error (yawp15 cell) vs 0.66 at 3.4 mm (seat cell).
2. V2's 4/6 rests on an EE-pose term by the back door (proprio is an encoder input; 3-step yaw
   probe −0.216 vs V3a −0.013). Off the demo pose without the EE-path cost V2 is 1/10 and 3/10.
3. λ never switches; Dλ cancels an affine drift; the deployed LCS is affine where seating is
   decided.
4. The latent cost never prefers Fz− at riding-over states (V2 1/6, V4a/V7a/V7b 0/6) while the
   decoded task distance says UR z− is the lever; V7a's B(z) fixed the onset sign and then
   overshot to "under" (radial slip below the groove; `outcome.py`: under = h_median < −5 mm).
5. C3 at admm 2 works through an accidental damper (relaxed λ); admm 5/10 saturate rotations.
6. Rotation runaway (> 100°) is the only safety failure; R follows data percentiles by accident.
7. Seated frames exist in three regimes — v2 insertion (464 train episodes, about a quarter end
   engaged, **UR released at `place_3` in every one**), approach (yaw ±30°, UR held, ~20 %
   engaged) and contact branches (wrap ≤ 143°) — but essentially none from the riding-over states
   the MPC reaches (recover 1/15, press48 0/48, on-policy 1/120); UR-down is 7–10 % of contact
   steps.
8. One-step/h7 error is a floor, not a selector (ρ ≈ +0.37 over 13 models, driven by the unstable
   ones; among stable models V4a/V7a 0.509 mm → 0/6 and 2/6, V2 0.577 mm → 4/6).
9. The encoder dominates error on yawed approach frames (1.54 vs floor 1.50 mm); at contact the
   dynamics gap is real (V4a contact h7 3.13 mm vs 2.18 for V7a; contact gain ≈ 0.6).
10. Every verdict since 09-26 is one episode per cell; the trainer's val is tuple-level; seat6 is
    the design loop's training set; the live yaml pins `stage_goal_tols: [1e-6, 1e-6]` (stages run
    to timeout; latent completion is de-facto off but not switchable).

**Plan in phases.** Wave 0: safety and measurement floor (feasibility-safe orientation envelope,
physical R, demo-replay feasibility with and without the envelope, stage-1 ceiling from BL seated
episodes, belt-feature stage rule as its own package, held-out grid with cell-level paired tests)
in parallel with the offline bundle (proprio-leak test, probes, gate retrodiction, episode-level
val, the Markov-sufficiency pre-check that freezes V9's state config) and the scripted
reachability search from pool snapshots. Wave 1: two no-retrain paired arms, the PCA positive
control, an admm re-test; the oracle planner runs before any V9 closed loop if both arms and the
control fail. Wave 2: at most two structured retrains (V9a on the v3 union, V9b with press data)
on an analytic physical state plus 4 free dims, λ tied to the predicted next-step surface gap,
gates in coordinate units. Oracle before any third retrain or wave-3 training. Serial sim queue
in §4.5 (firm ≈ 19.0 h, ≈ 27–31 h with conditionals and slack, 5 days).

## 2. Consolidated diagnosis with evidence

| # | failure | evidence | campaigns |
|---|---|---|---|
| D1 | cost geometry ≠ task; blind in the last ~8 mm; blind directions count in Q | 09-29 Q4; yawp15 table | V3a, V4a, metric A/B, V4c |
| D2 | proprio leak into z and z_g; material-index grasp dependence of z | 09-29 report §1; archive README; reference §7 | V2 vs V3a; 09-25 |
| D3 | no contact modes; λ is a bias absorber because the rest state is not a fixed point | yawp15 table (‖WΣDλ*‖ ≈ ‖WΣ((A−I)z+d)‖); 09-24 phantom drift; u = 0 rollouts 0.25–0.49 | all |
| D4 | lever wrong in the latent cost; gain under-predicted (≈ 0.6); overshoot to under | g2_lever_end.md; press48; 09-28 G3; 09-30 | V4a, V7a/b, metric A |
| D5 | C3 inconsistency as damper | yawp15 table; E4 admm sweep | V2, V3a |
| D6 | rotation runaway; R from bounds; free inputs → bang-bang → grasp loss (09-24) | 09-30 +15°; 09-29; V8 box; 09-24 | V3a, V4b/c, V5, V6, V7a, V8 |
| D7 | no over → seated transitions; seated spans UR-released and UR-held regimes; causality under load (u_x slope 0.751); transmission u → Δs depends on grasp/arm pose | 09-28; press48; V8; 09-25 | V3a, V8 |
| D8 | encoder OOD offset on yawed starts; contact dynamics gap | 09-30 table; 09-28 | V3, V7a yaw |
| D9 | evaluation: n = 1 cells; tuple-level val; completion rule pinned off by 1e-6 tolerances; BL undefined on yawed cells; mixed target provenance (soft-anchor demo/synthetic vs stiff-anchor yaw targets); gravity tilt; Newton pin | R1-C §1.5; trainer:803; live yaml l.67; RUN-STATE-v8 | all |
| D10 | multistep/contrastive objectives exploit scale or B; loss-scale traps (MSE m² ≈ 6e-4 of RMSE) | 09-29/30 | V2b, V4b/c, V5, V6, V7b |
| D11 | on-policy LCS-only loop on failures; gate loosened 2.178 → 2.475 | learning_curve.md | V8 |
| D12 | free-space goals reach their minimum at 0.7–2 s then drift away (18/18) | 09-29 fs6 | V4a, V7a |

Facts corrected during the rounds: V2's "0/40 off-demo" was the `penalize_input_change`-bug run
(clean latent-only numbers are 1/10 and 3/10); `sim_belt_xyz` and `sim_pulley_large_pose` are in
every training npz (`_pulley_labels` reads them), so λ/occupancy labels come from belt bodies and
`pcd_belt` is the encoder input; "464 seated episodes" is 464 insertion train episodes, about a
quarter engaged, UR-released; "dynamics add 0.04 mm" holds only on yawed approach frames; the
spinning models are the ones insensitive to rotation (the spin is cheap rotation plus a model wrong
off-data, not a pull toward demo poses); a negative-gap hinge is infeasible under `relu(−slack)²`.

## 3. Decisions and resolved conflicts

| conflict | decision | rationale |
|---|---|---|
| proprio in encoder / goal | Out of the goal for every new model; EE translations in x with zero cost (deployed). Dynamics may see p_rel through A_p (C++ `A_aug.topRightCorner(n_x, 6) = A_p`, one block, no QP growth); A_p on/off is decided by the Markov pre-check. | D2, D7 |
| state form (sim campaign) | Analytic physical coordinates computed directly from the 150 belt points + pulley pose in the encoder node (no encoder offset; goal and state in the same units) plus a small learned encoder for the free dims. Default z = [s (6); g (6); c (3); ρ̄ (1); z_free (4)] = 20, n_λ 6. The PointNet regressor of the 16 physical coordinates is the hardware-path twin (same loss; not used in sim closed loop). Budget fallback if quiet p95 at 20 + 6 (+6 θ) states > 70 ms: K = 5, ρ̄ dropped → [s 5; g 5; c 3; z_free 3] = 16, with the g rows added to the stage-2 Q (g_g = 0) since s alone no longer costs a radially-off belt at seat height. | height-only gaps cannot represent flange-top or under contact; stage-1 grasp invariance; QP budget |
| sim vs hardware caveat | In sim the 16 physical coordinates are exact, so the encoder OOD offset (D8) is removed by construction, not by design: **every sim comparison V9 vs V7a conflates "better representation" with "no encoder error"**. No hardware or transfer claim is made by this plan. Before any such claim: a twin gate (held-out PointNet regression error on s/g ≤ 1 mm on yawed and grasp-varied states) and one twin-in-the-loop dev ×1 run of V9 with the PointNet coordinates replacing the analytic ones; the twin runs once on grid-dev regardless. | D, C (Round 5) |
| Markov sufficiency | Offline pre-check before V9a's config is frozen (W0.3): LCS fits via the cached-latent path on held-out grasp sets (set1/set3/set4) from (i) the analytic state, (ii) + p_rel (A_p on), (iii) the V4a latent; one-step and h7 on s in coordinate units. If (i) loses > 20 % vs (ii) or (iii) → **A_p on**; free dims stay 4 on both branches. | D-N1, A-N1 |
| λ form | Gap rows tied to the predicted next-step gap by parameter sharing in `unpack_params` (`--tie-gap-rows`: E_g ≡ A_g, H_g ≡ B_g, c_g ≡ d_g, G_gg = sqrt(α − ε)·I so F_gg = αI ≡ D_gg; **when A_p is on the tie extends to the p block, E_p,g ≡ A_p,g, and the C++ E_aug carries it**); flag off = HEAD with an identity test; exporter writes plain E, F, H, c. Regressed to measured g(t+1) through the g rows of the dynamics loss; occupancy hinge secondary; λ = 0 on free/pre-contact frames; `--no-lcs-warm-init`; J = 0 on structured rows. λ is identifiable only where the free prediction would penetrate, so λ gates are defined on press / penetrating-contact frames; holds expect λ ≈ 0. With F = αI each λ_k is an independent per-bin floor (no inter-bin support coupling), accepted for V9a; the learned D_s,g / D_c,g columns still couple the cost coordinates. Analytic-gap-only fallback. | A-B1/B4, A-N2, B-N2/N5 |
| envelope form | θ-state first (+6 integrator states, STATE bounds ±60° Franka / ±20° UR about the episode start, binding x_1..x_{N−1}); **feasibility-safe: when x_0 violates a box, floor or ceiling the violated bound is clamped to x_0 (u = 0 stays feasible; only moves back toward the box are allowed — no outward widening, which would ratchet a runaway); infeasible or clamped solves are logged as a safety-failure class**; θ gets its own scale in C3; lite (per-solve input bounds) only if quiet p95 > 70 ms. No extra per-step caps beyond ub-ins. Demo replay with and without the envelope. Hand-z floor; stage-1 z ceilings set from BL seated dev episodes and frozen before `final`. | B-B7, B-N1, D-B2, D-N2, D (Round 5) |
| physical R | R = w_r·diag(1/s_i²) with s_i frozen to the insertion half-ranges (numerically unchanged for V2/V4a/V7a at w_rot 1×); w_rot 3× only as a paired arm; the 2× per-channel guard applies to translation channels. | three campaigns changed behaviour through R by accident |
| Q/R calibration for V9 | w_q scaled so the median stage-2-start Q cost equals V2's (09-29 metric rule); per-stage Q via `use_q_matrix` + diag `q_matrix_stage*` extended to N stages; bound-hit ≤ 5 % of solves on all channels (09-24: free inputs → bang-bang). | B-B2, D-B5 |
| stage rule path | The encoder node computes belt features → new `LATENT_STATE` fields → C++ N-stage `Stages()` + N-stage `q_matrix_stage*` + `latent_completion: off`. Own package W0.1c; W1.1 and V9 depend on it; W0.4 baselines do not (live tolerances 1e-6 = stages to timeout). Stage-1 yaw error logged; a tangential-angle coordinate (one free dim converted) if > 10° on yawed targets. | A-B2, B-B1, B-B6 |
| V9 gates | All V9 gates in coordinate units (whitened per-row one-step / h7 of s, g, c, ρ̄; σ 2 mm for s/g/ρ̄, 5 mm for c) with no-motion and V7a-decoded-feature baselines in the same units; L_dec down-weighted to 0.1× (free dims only); decoded-mm gates apply to opaque-latent models only. | A-N1 |
| cost-to-go | Not from demo data; later only on pulley-frame coordinates with off-demo successes. | EE-path cost in disguise otherwise |
| oracle sampling MPC | Scripted `press_both` first. The oracle (true-sim planner under envelope/bounds/R; positive control = reproduce the demo seat) runs on any of: (a) scripted grid + lift-shift-lower find nothing; (b) V9 passes offline gates but fails plan realisation; (c) before any third retrain or any wave-3 training; (d) if both W1.1 arms and W1.2 fail — queued before V9a's closed loop. | D-B1 |
| baselines | With envelope (w_rot 1×) + physical R + ceilings, stages-timeout default, pinned yaml (`honor_penalize_input_change: true`, w_r 0.3, admm 2, floors) in `index.json`; V2 unmasked/no-envelope once as a continuity row. | C-B6 |
| on-policy loops | Only after V9 and ≥ 1 found success; buffer seeded with successes; pinned coordinates; per-iteration re-export of z stats/goals; stop at 0 successes in 24 attempts. | V8 |
| leakage | Collection snapshots and the plan-realisation bank from pool cells only; recover12 = 12 pool overs never used by press48 / W0.2 / W2.2 / training; set4 held out, set5 (new seed) for training. | C-B2 |
| statistics | Cell-level pairing; exact two-sided test on discordant cells; dev "promising" = net ≥ +5 with ≤ 0 reversals (p 0.0625) or p < 0.1; "pass to final" p < 0.05; non-inferiority with the paired (McNemar) variance and margin 0.15 on n = 96 episode pairs. | C-B1, D-B6 |

## 4. Phased plan

Rates: episode ≈ 35 s; 300-epoch training ≈ 1.6 h; LCS fit via the cached-latent path ≈ 25 s;
fast export 1 s, full 23 min; one sim at a time (two concurrent cost +30 % solve latency). Owner
types: developer (C++/trainer/collector logic) and test-writer (assets, harness flags, tables).
Every new model and eval is registered in `docs/models.{md,yaml}` in the same change.

### Wave 0 — safety and measurement floor

**W0.1a Envelope + physical R + re-solve at switch (developer, ~1 day, no sim).** θ-state
augmentation (x = [z; p_f; p_ur; θ_f; θ_ur], θ' = θ + u_rot, STATE bounds about the episode start,
re-anchored per episode never per stage, θ scaled in C3), feasibility-safe bounds (violated bound
clamped to x_0; infeasible or clamped solves logged as the "infeasible" safety class; same for z
floors and ceilings), hand-z floor, stage-1 z ceilings (yaml key; value from W0.4's BL check),
R from frozen insertion half-ranges + `w_rot`, re-solve on the stage-switch latent,
`honor_penalize_input_change: true`; `x_sol` debug rows sized from `n_x`. Opt-in keys; HEAD path
byte-identical; `learned_lcs_c3_check` PASS. Timing check: quiet-machine solve p95 for V4a/V7a at
22 and 28 states, and synthetic 26/32 (n_x 20 ± θ); > 70 ms at 28 → envelope-lite; > 70 ms at 26 →
V9 budget fallback (K = 5, n_x 16). Feasibility check: offline clip analysis of the demo's recorded
u against ub-ins + the θ box (clipped fraction per channel); sim replay of the demo u under the full
W0.1a yaml (10 min) and the same replay with the envelope off (2 min, soft-vs-stiff-anchor control);
the envelope run must match the control's outcome. Gate: identity + replay + continuity seat6 ×3
for V2 at w_rot 1× (mean strict within one cell of 4/6).

**W0.1c Belt-feature state + stage rule (developer, ~1 day, separate package).** Encoder node:
compute the analytic coordinates [s, g, c, ρ̄] and h_median / wrap / slant / UR-side-arc height
from its 150 belt points + pulley pose; append the stage features to `LATENT_STATE` (contract
change in reference §2; harness `take_targets`/`index.json`; stage-switch time and stage-1 yaw error
logged). C++: N-stage `Stages()` and N-stage `q_matrix_stage*` (`QMatrix(stage)`,
`SetLearnedStageCost`), rules 2a (h ≤ 3 mm), 2b (wrap ≥ 90°, |h| ≤ 2 mm), 3 consecutive frames,
timeouts kept, `latent_completion: off`. Identity: rule off ⇒ byte-identical. Dependency of W1.1
and all V9 runs.

**W0.2 Scripted reachability search (developer; sim 1.9 h + 0.5 h, conditional +1.9 h).**
`press_both` family (UR dz {−4, −8, −12}, Franka dz {−4, −8}, lateral {0, 3, 6} mm, UR-first delay
{0, 0.3 s}, roll ±3°, min-jerk 1.5 s + hold; 16 seeded variants) and `lift_shift_lower` family
(lift 10–20 mm, shift 5–10 mm toward the axis, UR-first lower; 4 variants) from 3 held-out
final_over + 9 pool riding-over snapshots (never reused by recover12), within `ee_constraints`
floors and ub-ins per-step ranges; causality gate (realised-vs-u slope [0.6, 1.2]) per channel on
UR z / Franka z on every rollout; `sim_realised_delta` recorded. The 12 V8-buffer overs only if the
first set is ambiguous (10–30 %). Branch: ≥ 30 % of snapshots engaged and ≥ 1 strict → W2.2;
10–30 % → collect from the recoverable stratum and plan prevention; < 10 % in both families →
recovery modelling dropped, CEM fallback once (3 h; oracle trigger (a)), prevention data (W2.2b) and
the stage design become the path.

**W0.3 Offline bundle (developer; test-writer for split/val flags and tables; ~1 day, parallel).**
- Markov-sufficiency pre-check (hours, GPU, before V9a's config is frozen): analytic coordinates
  cached for the v3 union and the set1/set3/set4 held-out episodes; LCS fits (lcs_params only,
  `finetune_lcs_latent.py` path) on (i) the analytic state, (ii) analytic + p_rel, (iii) the V4a
  latent with s/g/c/ρ̄ read from its decoded belt; one-step and h7 on s per held-out grasp set in
  coordinate units. Rule: (i) worse than (ii) or (iii) by > 20 % → A_p on. Output: the frozen V9a
  config and the n_x/budget decision (with W0.1a's timing check).
- E3 proprio leak: offline decomposition of the whitened goal distance into its proprio part for
  V2/V4a; closed loop with goals re-encoded once per episode with the start frame's proprio (fixed
  `z_goal` per cell; no C++ change); V2 seat6 ×3. V2-masked < 2/6 mean → V2 retired as the
  reference; the adoption reference becomes max(V2-envelope, BL) and is named in the results table.
- E4 ridge probes from the V4a/V7a latent → h_median, wrap, radial offset, UR-side-arc height,
  per-bin surface distance; in-distribution vs held-out reported separately (expressivity vs
  coverage).
- Metric A fitted for V7a (the 09-29 recipe; hours) — prerequisite of W1.1 (A exists only for
  V4a and V4c today).
- Gate retrodiction over the 13 models with closed-loop results (decoded-form lever; per-bin gain);
  plausibility screen only (n = 6 per model).
- Splits: `split_v4.json` with an episode-level `val_mix` (~10 % of free-space/approach/contact
  train episodes removed from train; v3 test lists untouched); held/released flag per frame; set5
  grasps (new seed, same box) for training; set4 held out.
- Quiet-machine solve-time re-measurement (V7a's 83 ms p95 was measured under CPU load).
- Target hygiene: regenerate the 4 unyawed seat targets under stiff anchors (≈ 2 min sim); label
  demo frames 13/59 as soft-anchor.

**W0.4 Held-out grid + baselines (test-writer; sim 2.9 h + 0.9 h).** Grid: grasp set1 (13), set3
(8), set4 (8, roll ±15°, slide ±5 mm); depth/slant nominal, ud+3, ud+4.5, ud+6, ud+6_high; EE
offsets (4); yaw 0, ±5, ±10, ±15 (±15 only in `final`); `dev` 24 cells, `final` 48 cells, published
seed, disjoint from each other and from every pool/collection cell. recover12 = 12 pool overs
disjoint from press48 / W0.2 / W2.2, ×2. fs6 report-only with min-vs-final. Order: BL dev ×2 first
(no envelope or ceiling applies to the waypoint baseline); from BL's seated episodes report
max(EE z − start z) per arm; if > 10 mm on > 10 % of seated episodes the ceiling is widened or made
relative to the flange top; the ceiling is then frozen before `final`; then V2, V7a under the W0.1a
yaml; then BL-yawed (the yawed waypoint replay that built the yaw targets, where available), V4a+A,
V2-masked. Protocol check: repeat-to-repeat strict agreement ≥ 80 % of cells, else repeats → 3.
`final` for baselines runs last in the queue.

### Wave 1 — no-retrain paired arms

**W1.1 P7 arm (developer; sim 0.7 h + 0.7 h):** V7a + W0.1a + W0.1c rule + metric A(V7a), the
existing two goals, on dev ×2 + recover ×2, cell-level paired vs V7a-envelope; second arm w_rot 3×.
"Promising" = net ≥ +5 discordant cells (p ≤ 0.0625) and 0 safety failures; a miss parks P7 only
(a win means "controller headroom exists"; it is not attributable to P7 alone since the arm includes
W0.1c).
**W1.2 PCA positive control (developer; 0.5 day GPU + 0.5 h sim):** PCA-16 of pulley-frame belt
points as a linear encoder type in the numpy sidecar / `latent_encoder.py`; LCS via the
cached-latent path (same pipeline as V9); exact quadratic task cost. Pass = offline S1–S4 and not
worse than V7a-envelope on dev ×2; fail = no inference.
**W1.3 admm-5 re-test (developer; 0.2 h):** V2 under W0.1a on seat6 ×3 (legacy): settles whether
the damping was only rotation exploitation.
**Oracle trigger (d):** if both W1.1 arms and W1.2 fail → the oracle (3 h) runs before V9a's
closed loop.

### Wave 2 — at most two structured retrains before the oracle

**W2.1 V9a (v3 + press48 holds) and V9b (+ W2.2 data; mandatory twin) (developer; 1.6 h GPU each).**
Config frozen by W0.3's pre-check and W0.1a's timing check. Recipe in §5; `--tie-gap-rows`,
`--no-lcs-warm-init` (new trainer flags, identity-tested); checkpoints every 25 epochs; `val_mix`
curves. Offline gates in coordinate units: S0 whitened one-step on [s, g, c, ρ̄] ≤ 1.1× the
V7a-decoded-feature baseline and better than no-motion on moving frames; S1 ρ(A) ≤ 1.02, u = 0
drift ≤ 1 mm (s) on free and contact holds, and on press / penetrating-contact frames (press48
ramps, W2.2 presses, contact frames with commanded z− on the loaded arm) λ active on ≥ 90 %,
λ-contribution ‖Dλ‖/‖Δz‖ ≥ 0.3, every D column norm > 0.1·α after ≥ 1 epoch with press data; on
holds and free frames ≤ 5 % active; exporter LCP-vs-PGD check at tight tolerance; S2 ≤ 5 % of
over/slanted/under held-out frames inside goal tol; S3 lever sign ≥ 10/12 (s-form) and per-bin
gain ∂s_k/∂u_z within [0.6, 1.4] of the press48 finite differences on ≥ 5/6 end states; S4
realised ΔD < 0 in ≥ 80 % of the 50-snapshot bank, realised/claimed ≥ 0.25, bound-hit ≤ 5 % (all
channels); GC4 (V2 seated vs V4a over runs separated by ≥ 2× hold drift); GC6 (±20° yaw probe
|ΔD| ≤ 2 mm); GC8 whitened (‖B_w‖ p95 ≤ 2× V9's own epoch-50 checkpoint; |∂s_k/∂u_z| ≤ 1.5 mm/mm).
Closed loop: top-3 → dev ×1; top-1 → one more dev repeat (reusing the ×1 run) + recover ×2 + fs6 +
legacy seat6/yaw12; pass to final = exact p < 0.05 vs the W1.1 reference, 0 safety failures, no
cell ending over/under. Branches: S1 λ fail → analytic-gap-only init once (the V9b slot); a second
S1 fail → oracle + data audit before any representation conclusion; S2 fail → κ/w_q recalibration
(export only); held-out s residual > 1 mm → coverage audit first; offline pass + closed-loop fail →
snapshot diagnosis + oracle (b). Hardware-path twin: the PointNet regressor trained alongside,
gated at held-out s/g regression ≤ 1 mm on yawed and grasp-varied states, run once twin-in-the-loop
on grid-dev ×1.

**W2.2 Coordinated-press data (developer; sim 3.1 h + 0.4 h; gated by W0.2).** 40 pool riding-over
snapshots (disjoint from recover12; balanced over h 8–20 mm and grasp sets) × (6 best variants + 2
random) = 320 episodes with per-frame `sim_h_median_mm`, `sim_wrap_deg`, `sim_outcome`,
held/released flag, `sim_realised_delta`; causality gate per channel; target ≥ 60 seat transitions
(h +5 → ≤ 1 mm) and UR-down ≥ 40 % of contact steps; plus 40 `lift_shift_lower` recover
episodes and held frames at over/seated states (L_hold targets). W2.2b prevention fallback:
approaches with the UR box tangent / elev [0, 10] mm, 80 episodes, engaged share ≥ 50 %.

### Wave 3 — conditional (after the oracle has run)

W3.1 B(s, g, p) head (V7a's recipe on physical inputs, zero init, warm-up so B0 learns the mean,
head lr 1e-4, weak sign prior on the bins nearest each gripper; needs W2.2 for the UR-side rows);
W3.2 real-counterfactual ranking hinge on the MPC's own cost with data-provided sibling rollouts
only; W3.3 λ-block consistency re-test at admm 2 and the rollout-consistent fallback QP if the
budget allows; W3.4 success-seeded DAgger (pinned coordinates, per-iteration re-export, stop at
0/24); W3.5 belt-feature cost-to-go on off-demo successes; W3.6 Newton upgrade + gravity-tilt fix at
the round boundary with the BL + V2 dev ×2 sensitivity test (shift > 0.15 → re-collect targets,
grid starts and press data; pre-cutover data kept as a legacy stratum).

### 4.5 Published serial sim queue (one sim; attended ≈ 6 h/day, long batches overnight)

| slot | item | h | cum. | prerequisite | owner |
|---|---|---|---|---|---|
| Q1 (d1) | set5 grasps + stiff-anchor target regen | 0.2 | 0.2 | — | test-writer |
| Q2 (d1) | W0.2 press_both, 12 snapshots × 16 | 1.9 | 2.1 | collector ext. | developer |
| Q3 (d1) | W0.2 lift_shift_lower, 12 × 4 | 0.5 | 2.6 | Q2 | developer |
| Q4 (d1, cond.) | W0.2 V8-buffer overs × 16 (only if Q2 ambiguous) | 1.9 | 4.5 | Q2 | developer |
| Q5 (d2) | demo replay under W0.1a yaml + no-envelope control | 0.25 | 4.75 | W0.1a identity | developer |
| Q6 (d2) | continuity V2 seat6 ×3 (w_rot 1×) | 0.2 | 4.95 | Q5 | developer |
| Q7 (d2) | E3 V2-masked seat6 ×3 | 0.2 | 5.15 | W0.3 E3 assets | developer |
| Q8a (d2) | BL dev ×2 → ceiling feasibility check (offline, minutes); ceiling frozen | 0.5 | 5.65 | grid json | test-writer |
| Q8b (d2) | V2, V7a dev ×2 under W0.1a | 0.9 | 6.55 | Q8a, Q6 | test-writer |
| Q9 (d2) | recover12 ×2 for V2, V7a (12 pool overs) | 0.5 | 7.05 | Q8b | test-writer |
| Q10 (d2/3) | dev ×2 BL-yawed, V4a+A, V2-masked | 1.4 | 8.45 | A fit, Q7 | test-writer |
| Q11 (d3) | W1.1 P7 arm dev ×2 + recover ×2 | 0.7 | 9.15 | W0.1c, A(V7a) | developer |
| Q12 (d3) | W1.1 w_rot 3× arm | 0.7 | 9.85 | Q11 | developer |
| Q13 (d3) | W1.3 admm-5 V2 seat6 ×3 | 0.2 | 10.05 | W0.1a | developer |
| Q14 (d3) | W1.2 PCA dev ×2 | 0.5 | 10.55 | PCA export | developer |
| Q15 (d3 night) | W2.2 press collection 320 + recover 40 | 3.5 | 14.05 | Q2 pass | developer |
| Q15b (d4, cond.) | oracle sampling MPC, trigger (d) — before any V9 closed loop | 3.0 | 17.05 | Q11–Q14 all fail | developer |
| Q16 (d4) | V9a top-3 dev ×1, top-1 +1 repeat, recover ×2, fs6, legacy | 1.25 | 18.3 | V9a gates (+Q15b) | developer |
| Q17 (d4) | V9b (data twin) dev ×2 + recover ×2; hardware-twin dev ×1 | 1.0 | 19.3 | Q15, training | developer |
| Q18 (d4 night) | final ×2 baselines BL, BL-yawed, V2, V7a (4 × 56 min) | 3.7 | 23.0 | Q8 | test-writer |
| Q19 (d5) | V9 final ×2 (if dev pass) | 0.9 | 23.9 | Q16/Q17 | developer |
| Q20 (d5, cond.) | oracle, triggers (a)/(b)/(c) / CEM fallback | 3.0 | 26.9 | per §3 | developer |
| slack | 20 % of firm items | 3.8 | 30.7 | — | — |

Firm sim ≈ 19.0 h (18.7 + the hardware-twin dev ×1 run); conditional ≈ 7.9 h (Q4, Q15b, Q20;
Q15b and Q20 are rarely both triggered → ≈ 4.9 h typical); with slack ≈ 27–31 h ≈ 5 days. GPU in
parallel: d1 analytic-coordinate cache + Markov pre-check + E4 + retrodiction + metric-A fit + PCA
cache; d2 split_v4 + PCA LCS fit + config freeze; d3 V9a (1.6 h); d4 V9b (1.6 h) + readout twin if
E4 passed + the hardware-path PointNet twin.

## 5. The representation design

**Frame.** Large-pulley frame: origin at the groove-plane centre on the axis, ẑ_p = pulley axis
(vertical in the scene: "over" = resting on the flange top, "under" = below the groove). For belt
body / point i: axial height a_i above the groove plane, radius r_i, angle θ_i. λ/occupancy labels
come from `sim_belt_xyz` + `sim_pulley_large_pose` (as `outcome.py`: seated ⇔ |a| within the groove
axial band and |r − R_seat| ≤ 5 mm; over/under ⇔ h_median > +5 / < −5 mm); the profile targets from
`pcd_belt` in the same frame; both use the same body → point map (`belt_points_ordered`) so a bin's
s and g never disagree about emptiness. Engagement arc ≈ 200° centred on the seat; K = 6 bins of 33°
(wrap ≥ 90° ⇔ ≥ 3 adjacent bins seated).

**Coordinates (n_x = 20 default).**
- s_k (6): soft-weighted mean axial height of the bin's belt segment (Gaussian in θ, σ 8°; sigmoid
  radial membership [R_g − 10, R_g + 15] mm); smooth blend to a_fill = +40 mm when the bin is
  empty; soft saturation 60·tanh(s/60). The **cost coordinate** ("where on the surface").
- g_k (6): analytic distance from the bin's belt segment to the pulley surface (flange top, groove,
  lower flange; the pulley is a known rigid shape): 0 when resting anywhere on the surface, > 0
  when free. The **gap coordinate** for λ_k.
- c (3): belt-centroid offset (axial, radial, tangential) in the pulley frame — the grasp-invariant
  stage-1 target (the free code of material-indexed points is grasp-specific, reference §7). If the
  stage-1 yaw error exceeds 10° on yawed targets, one free dim becomes the belt's tangential angle
  about ẑ_p.
- ρ̄ (1): soft-weighted radial offset over the seat band.
- z_free (4): learned code (PointNet over the cropped belt-region cloud + belt points in the pulley
  frame, no proprio) for belt shape (slack, twist, UR-side drape); cost weight κ ≤ 0.25; never a
  goal by itself.
Sim path: the 16 physical coordinates are computed analytically in the encoder node (exact); the
PointNet regressor of the same 16 (L_coord) is the hardware-path twin. p_rel enters the dynamics
through A_p when the pre-check says so. Rotation augmentation about ẑ_p on belt points / the
belt-region crop only; set5 grasps in train. Decoder z → 150 points kept for the viewer and the
down-weighted decoded loss.

**LCS structure.** z' = A z + B u (+ A_p p) + D λ + d; 0 ≤ λ ⊥ w = E z + F λ + H u + c ≥ 0 with
n_λ = 6. `--tie-gap-rows`: E_g ≡ A_g, H_g ≡ B_g, c_g ≡ d_g, G_gg = sqrt(α − ε)·I (so F_gg = αI ≡
D_gg), and E_p,g ≡ A_p,g when A_p is on, so that w_k = ĝ_k(t+1): λ_k > 0 only when the free
prediction would penetrate the surface; D λ pushes ĝ back to 0 and, through the learned D_s,g,
D_c,g, D_ρ,g columns, stops s/c/ρ̄ where the surface stops the belt — the "stop at the groove / on
the flange" V7a lacks. At rest on a surface A ≈ I, d_g ≈ 0 and λ = 0 (consistent with L_hold);
the support force is identifiable only on press frames, which is where the λ gates look. J = 0 on
structured rows (PGD ignores the skew part, the exporter LCP does not); `--no-lcs-warm-init` (the
c-shift to a target active fraction would corrupt the gap semantics). Over → seated → under are
different (s, g = 0) locations with the same gap structure; the active set changes at every
free ↔ contact transition and λ carries the support force C3 plans over.

**Losses** (grad-norm calibrated at the V4a epoch-300 operating point; dynamics terms on detached
z; per-dimension whitened with σ fixed at 2 mm for s, g, ρ̄, 5 mm for c, batch σ for z_free; units
written into the exporter — the unweighted MSE in `lcs_model.py` would otherwise re-create the
09-29 loss-scale trap):
- L_recon (RMSE); L_dec decoded next state at 0.1× (free dims only; decoded mm is not a V9 gate).
- L_coord = ‖ẑ_phys − [s*, g*, c*, ρ̄*](t)‖₁ (hardware-path twin; identity on the sim path).
- L_dyn = whitened ‖ẑ' − [s*, g*, c*, ρ̄*, z_free(t+1)]‖²; its g rows are the gap regression.
  L_λ hinge secondary: o_k·|w_k| + (1 − o_k)·relu(m − w_k) + (1 − o_k)·λ_k with o_k = 1 ⇔
  g*_k(t+1) ≤ 1 mm, on every frame; m = 2 mm; the held/released flag stratifies the report.
- Guards: L_ρ = relu(ρ̂(A) − 1.0)² (power iteration); L_B = relu(‖B_w‖_F − 1.5·‖B_w,ref‖_F)² with
  B_w = rows/σ_dim and the reference from V9's own epoch-50 checkpoint; L_hold on u = 0 frames
  including contact holds (press48 pre-holds, W2.2 holds, insertion pre-holds): whitened ‖ẑ' − z‖ +
  0.1·‖decode(ẑ') − belt_t‖. No multistep gradient through the AE; an optional 7-step loss on the
  s/g rows only, with L_B active. No InfoNCE; no model-generated negatives.

**C3 state, cost, goal.** x = [z (20); p_f; p_ur] (+ θ (6)) — identity rows for p/θ, zero cost;
A_p block when on. Per-stage diag `q_matrix_stage1..N`: stage 1 on c (s at its fill value is flat;
z_free at κ); stages 2a/2b on s, ρ̄ (and g under the K = 5 fallback; c at κ = 0.25, z_free at κ);
w_q scaled so the median stage-start Q cost equals V2's; R from frozen insertion half-ranges;
bound-hit ≤ 5 % verified offline. Goals: the 16 physical coordinates of z_g computed analytically
from the target belt (no encoder offset on the goal); z_free,g = encode(target) with no proprio;
`goal_tol_q` ≡ the W0.1c rule (max_k |s_k − s_g,k| ≤ 2 mm and |ρ̄ − ρ̄_g| ≤ 2 mm — the strict
criterion by construction). QP size per knot 20 + 6 + 12 (+6 θ) = 38–44 vs V4a's 36; the W0.1a
timing check decides before training.

## 6. Evaluation protocol and adoption bar

Suites: grid-dev 24 × 2 (iteration); grid-final 48 × 2 (once per campaign, never tuned on);
recover12 × 2 (12 pool overs, disjoint from press48 / W0.2 / W2.2 / training); fs6 report-only (min
and final); seat6/yaw12 legacy rows for V2, V7a, V9 only. All starts held out; set4 never trained
on; collection snapshots and the plan-realisation bank from pool cells only (`onpolicy_pool.py
check --all` discipline). Per-episode measures: strict (wrap ≥ 90°, |h| ≤ 1 mm), engaged (≥ 60°),
final belt RMSE, end class (over/under/slanted/seated), max Franka/UR rotation (continuous), min
hand z, grasp loss, abort, infeasible/clamped solves, solve p95, early Franka dz, stage-switch time,
stage-1 yaw error. Safety failure = grasp loss, hand below the board, abort, infeasible or clamped
solve, or rotation > 60° (legacy rows). Pinned controller yaml, newton sha and target provenance
(soft/stiff anchor) in every `index.json`; stage-1 ceiling frozen before `final`.

**Paired tests.** Per cell: strict count over the 2 repeats (0/1/2); candidate vs reference
compared cell-wise; exact two-sided McNemar/sign test on discordant cells (n = 24 cells at dev, 48
at final). Dev "promising": net ≥ +5 discordant cells with ≤ 0 reversals (p 0.0625) or p < 0.1;
"pass to final": p < 0.05 (e.g. 6–0 or 8–1). Wilson CIs at the episode level with n stated;
single episodes never decide.

**Offline ladder** (floors, retrodicted on the 13 known models before use): S0 in coordinate units
for V9 (decoded mm for opaque models), held-out yawed s/g residual ≤ 1.0 mm; S1 stability / λ on
press frames / exporter LCP check; S2 cost geometry; S3 lever sign + per-bin gain; S4 plan
realisation + bound-hit; S5 2 s closed loop from the snapshot bank; GC4; GC6; GC8 whitened.
Promotion: proxy → dev ×1 (top-3 checkpoints) → dev ×2 paired → final.

**Adoption bar ("robust seating", grid-final, 96 episodes):** strict ≥ 0.70 with Wilson lower
bound ≥ 0.55; engaged ≥ 0.85; zero safety failures; per-stratum table reported, hard floor strict ≥
0.50 only for strata with ≥ 24 episodes (yaw sign, grasp set); non-inferiority vs the reference
(max(V2-envelope, BL)) on **n = 96 episode pairs** (cell × repeat): with b/c the discordant counts,
(b − c)/n + 1.96·sqrt([(b + c) − (b − c)²/n]/n²) ≤ 0.15 (an equal candidate passes up to ~55 %
discordance; at n = 48 cells the same margin would tolerate only 28 %); vs BL better with exact
p < 0.05 on yaw-0 cells, or on all cells where BL-yawed exists; recover12 ≥ 4/12 engaged if W0.2
showed recovery is reachable, else "no episode ends over"; solve p95 ≤ 75 ms; C3 check PASS. The
bar is a sim bar: no hardware or transfer claim follows from it (see §3 "sim vs hardware caveat").

Sim pin and cutover: the whole round at newton 79aa32ba with the ≈ 0.8° gravity tilt as-is; the
Newton upgrade + gravity fix is one change at the round boundary, validated by the BL + V2 dev ×2
sensitivity test; post-cutover numbers are reported separately until the baselines are re-run.

## 7. Stop rules and non-goals

Stop rules (pre-registered): no closed loop without the envelope; no model with ρ(A) > 1.05 or C3
check FAIL; W0.2 < 10 % in both families → recovery modelling and data stop; E3 masked-V2 < 2/6
mean → V2 retired as the reference; any on-policy loop with 0 successes in 24 attempts stops; any
fine-tune whose acceptance gate must be loosened stops (V8's 2.178 → 2.475); at most two
structured retrains (V9a, V9b) before the oracle has run; no wave-3 training before it; a V9 S1
failure after the analytic-gap fallback triggers the oracle + a data audit, not a representation
verdict; the current paradigm (opaque latent + whitened cost) gets exactly the W1.1 run; V9's
config is frozen only after the Markov pre-check and the timing check.

Non-goals / what we stop doing: EE-pose or demo-path costs in any form (V2's yaw lever is a bug,
not a feature); InfoNCE/CFM/latent-metric shaping on the whole code (six variants, one mechanism);
post-hoc metric fits as primary fixes; LCS-only on-policy fine-tuning on failures-only buffers;
raising `admm_iter` as a standalone fix; widening u bounds; whitening the cost by z_std on task
coordinates; one-step/h7 as a selector; decoded-mm gates for V9; cost-to-go from demos; a longer
horizon by itself; sampling MPC as a deployment controller (diagnostic only); an 80-d keypoint
state before a budget plan; new encoder backbones; hardware perception; making the sim physically
exact; re-collecting the 790-episode union; seat6 as a decision suite; free-space tracking as a
goal for V9.

## 8. Risks and open questions for the user

1. **Belt points and pulley pose at runtime.** In sim the physical coordinates are analytic from
   privileged belt points and the fixed pulley pose; on hardware the PointNet twin must regress
   them from the camera and the pulley pose must be known (calibration or perception). Approve the
   sim-path / hardware-twin split and the twin gate (≤ 1 mm on s/g) as the transfer precondition.
2. **QP budget.** Is 75 ms hard? n_x 20 + θ gives 44 per knot vs V4a's 36; the fallback is K = 5 /
   n_x 16 and envelope-lite. Decided by the W0.1a timing check on a quiet machine.
3. **Wire-contract change** (W0.1c): `LATENT_STATE` gains belt-feature fields; approve.
4. **Envelope priors** ±60° / ±20°, clamping to x_0 on violation, stage-1 ceilings from BL seated
   episodes (default start + 10 mm) frozen before `final`; w_rot 3× only as an arm.
5. **Sim time:** ≈ 27–31 h serial over 5 days including slack; `final` baselines overnight on d4.
6. **Seated-regime mismatch:** training seated frames are mostly UR-released; the targets are
   UR-held. Accept the held/released flag as a loss stratum, or re-collect held insertion seats
   (≈ 1.3 h sim)?
7. **Cutover timing** (round boundary) and the readout-only twin (one extra 1.6 h GPU).
8. Risk register: sim results for V9 remove the encoder offset by construction (the twin gate and
   twin-in-the-loop run are the only check); 4 free dims may still under-describe belt shape
   (recon reported, not gated); surface-gap labels may fight the fit (accept ≤ 1.10× in coordinate
   units; analytic-gap-only fallback); over-state λ rows are learnable only from presses; the
   envelope anchored at a slanted start (clamping covers it; logged); dev CIs are wide (paired tests
   only); jitter-driven flips are real variance; the stage-1 target on c may be weakly constrained
   on yawed targets (tangential coordinate if > 10°); with F = αI the bins have no inter-bin
   support coupling (accepted for V9a).

## 9. Appendix — campaign → lesson → plan item

| campaign | result | lesson | plan item |
|---|---|---|---|
| V1a/b/c (09-24) | gv1 11/26 vs BL 14/26; phantom u = 0 drift; free inputs → 16/16 grasp loss | rest state must be a fixed point; Q/R must be calibrated | L_hold incl. contact holds; median-cost rule; bound-hit gate |
| V2 | 4/6 seat6; 1/10, 3/10 off-demo latent-only | hidden EE term via proprio; goal_tol excludes over states only here | E3; proprio out; analytic goals; grid |
| V2b | AE collapse under ms7 | no multistep gradient into the AE | detached dynamics; s/g-row multistep with L_B only |
| 09-25 controller | w_p 22/40 rejected; completion fired 2–4.7 s early; UR released at `place_3` | EE-path cost forbidden; latent completion must be switchable off; two seated regimes | W0.1c `latent_completion: off`; held/released flag |
| V3a/V3b | 1/6, 79.8 mm; better one-step; UR rotation 10–16× cheaper; early lift +12–27 mm; held UR lags under load (0.751) | diversity ≠ task structure; R from bounds; stage-1 lift; transmission depends on grasp/arm pose | physical R; ceilings from BL; Markov pre-check / A_p |
| V4a | 0/6 calm, rests on top; 4/16 latent dims carry little belt motion | height-only metric blind to flange contact; the code spends dims off-task | s_k cost + g_k gap; free dims limited to 4 |
| V4a + metric A / B | 0/6, 1/6; yaw 6/12 (A) | cost fixes ranking, not the lever; pullback ill-conditioned | A(V7a) as the W1.1 stand-in; exact cost on analytic coordinates |
| V4b | 0/6 spins 178°; cond(F) 201 | position-keyed weighting hits statics; grads inflate | sampler as a separate ablation; guards |
| V4c | 1/6 spins 111–161°; AUC 1.0 | scale-free metric loss re-scales z; ranking gates pass unsafe models | pinned units; S2 necessary only |
| V5a/V5b | 0/6 unsafe; ρ(A) 1.30 | model-generated negatives exploited by B inflation | real counterfactuals only (W3.2); L_B/L_ρ |
| V6/V6a/V6b | 0/6, 0–2/12 unsafe; unstable A at pinned σ | a std band is not a guard | spectral penalty; physical units |
| E4 admm sweep | admm 5: 2 grasp losses; admm 10: 0/6, rotation at bound 43–79 % | relaxed λ is an accidental damper; consistency exploits rotation | envelope first; W1.3/W3.3 |
| press48 | dn10 lowers D 6/6; 0/48 seat; h +12.6 → +10.1 on the flange | Franka-only cannot seat; flange-top contact is a mode; λ identifiable on presses | g_k surface gap; λ gates on press frames; W0.2/W2.2 |
| V7a | 2/6, 3/12; presses; overshoots under; +15° unsafe; p95 83 ms (loaded) | state-dependent lever is right; no support mode; budget unknown | λ on ĝ(t+1); W3.1 head on [s, g, p]; timing check |
| V7b | ‖B‖ 159, ρ(A) 1.26 | multistep buys h7 by inflation | whitened L_B gate |
| yawp15 plan diagnosis | d 0.54 @ 7.7 mm; claimed +0.10 vs rolled −0.31; recon 4.2 mm | blind zone + inconsistency + encoder offset compound | analytic state; belt-feature rule; S4 realisation gate |
| V8a–c (+ box control) | 0/6, 19–20 mm; 1/120 successes; gate loosened; R 15–25× | failures-only buffers teach failure; frozen encoder fits noise; trust region ≡ R change | W3.4 after V9 + successes; R decoupled; "gate loosened ⇒ stop" |
| fs6 (09-29) | min at 0.7–2 s then drift away 18/18 | rest drift is a standing bug | L_hold; fs6 min-vs-final reported |

## 10. Revision log (condensed)

| round | items | outcome |
|---|---|---|
| R3 → R4 | A-B1 gap tied to ĝ(t+1), surface-distance gap; A-B2/B-B1 W0.1c package, baselines independent of it; A-B3 labels from `sim_belt_xyz` + `sim_pulley_large_pose`; A-B4/B-B3 warm-init off, J = 0, exporter LCP check; B-B2 per-stage Q via `q_matrix_stage*`, w_q median-cost rule, whitened dynamics loss; B-B4 fixed-goal E3; B-B5 metric A for V7a; B-B6 centroid coordinates; B-B7 θ-state first by timing; C-B1/D-B6 cell-level exact tests; C-B2 pool-only snapshots, set5; C-B3 BL-yawed / yaw-0; C-B4 causality gate; C-B5 464 / seated regimes; C-B6/D-B3/D-B4 continuity ×3, w_rot 1× baselines; C-B7 sim queue; D-B1 oracle triggers (c)/(d); D-B2 demo-replay feasibility, caps dropped; D-B5 Q/R calibration | all 24 adopted |
| R4 → R5 | A-N1 + D-N1 n_x 20 with 4 free dims, Markov pre-check, gates in coordinate units, L_dec 0.1×; A-N2 λ gates on press frames; D-N2 BL ceiling check, no-envelope replay control; C recover12 from pool overs, non-inferiority margin 0.15, firm hours; B-N1 feasibility-safe bounds + infeasible logging; B-N2 `--tie-gap-rows` with identity test; B-B1 N-stage `q_matrix_stage*`; B-B6 yaw watch; B-N3 analytic sim state + PointNet hardware twin; D-B1 Q15b; typos (n_λ 6, whitened L_B/GC8, shared body → point map, L_w removed) | all adopted; ACCEPT from B, then A/C/D |
| R5 → published | A: tie extended to the p block when A_p is on; g rows in the stage-2 Q under K = 5. C: n = 96 episode pairs stated; BL ceiling frozen before `final`; hardware twin once on grid-dev. D: "A_p on" wording; sim-vs-hardware conflation stated with the twin gate and twin-in-the-loop run; violated bounds clamped to x_0 (no widening), infeasible/clamped solves logged | all folded in; final verdicts ACCEPT × 4 |

Deferred (not scheduled): transition-aware sampler twin (after V9b); no-augmentation twin; growing
`final` to 72 cells (only if the stratum table is needed as a hard gate); a UR re-anchoring rate cap
on u_ur (W0.1a design review).
