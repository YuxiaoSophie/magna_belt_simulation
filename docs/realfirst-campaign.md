# Real-first campaign (V12): making the latent-LCS MPC loop seat the belt

Living record of every attempt and result of the autonomous campaign started 2026-10-05 night
(user mandate in `handoffs/RUN-STATE-20261005-realfirst.md`). Plan:
`handoffs/PLAN-20261006-campaign.md`; run log `handoffs/RUN-STATE-20261006-campaign.md`;
packages `handoffs/packages/PKG-20261006-cmp-*.md`. Glossary and earlier campaigns:
`docs/learned-mpc.md`, `docs/models.md`.

**IDs.** Runs are `V<campaign>-<nn>` (V11-01, V12-07, …; index below); models are
`V<campaign><letter>` (V12g, V13a, …; `docs/models.md`). Campaign V11 = the first two real-first
loops (2026-10-05), V12 = this campaign (2026-10-06); V13 = the frozen-encoder models of V12-13.

## 0. Status summary (updated 2026-10-12)

**Where we are.** Without any demo action prior, **V26b seats 16/22 on the hard suite** (V12-61,
noise 0). It is JEPA + graph stress + a label-aware goal margin, trained on 3 demos + MILES
+ 4 grasp-angle demos and their MILES + 42 operator attempts at grasp-angle starts. That is
12/14 on the non-angle starts and the first seats at positive grasp angles (up+20, up+30 2/2).
Open:
- Seed variance: seed 4 of the same recipe seats 0/22, and no offline metric predicts it.
- Negative grasp angles: still "over" for every model.
- The Jin & Posa loop: V12-57 is paused after one refit dropped 12 → 2/22.

**Everything tried, by theme** (details in the run index and §7; model registry `docs/models.md`):

| theme | runs | what was tried | outcome |
|---|---|---|---|
| loop basics | V11-01/02, V12-01..08 | real-first loop, clean clouds, Sinkhorn recon, warm refits, demo prefixes, latent waypoints + demo action prior | 8/10 near the demos with the prior (V12-07); varied grasps 7/50 |
| controller | V12-09/10/15/16/17/19/22/30/31/32/33/34/38/39/40, V12-48 | nearest-demo waypoints, overshoot guard, failure census, data-support EE box, waypoint schedule, seated goal set, progress start, MPPI sweep, H 16 → 24 / 32, learned cost-to-go, C3+, smoothness w_s, exploration noise | the support box is the one gain (5/8 vs 1/8); noise 0 for eval; the rest trade starts |
| demo action prior | V12-06/07/14/28/37 | demo action prior, stage-synced prior, MILES-BC prior (planned); removing the prior (user: cheating) | without the prior 5-7/22 (20 demos, V12h); the rule since 2026-10-07 |
| refit rules (J&P) | V12-08/11/12/13/18/27/35/36/46/57 | warm vs from-seed refits, frozen encoder, anchor 0.7, behaviour and response gates, on-policy weighting, keep-best | warm refits drift; from-seed flat; V12-46's first refit 8 → 11/16 then degrades; V12-57 iteration 1 2/22 |
| seed data and starts | V12-04, V12-20/21, V12-23..26, V12-29b, V12-58 | extra demos, slanted-belt and grasp-angle starts, MILES around demos, scripted grasp-angle demos (UR tilt ≈ −1.0 to −1.3 × grasp angle) | 3 demos + 12 min MILES (T1) ≈ 20 demos (V12-43) |
| encoder | V12-41/42/43/44/45 | data tiers T0/T1/T2, point augmentation, Delta-JEPA action head, latent 32, Point-MAE (pretrained) | better offline error does not mean better control (aug + head 1/22, aug 0/22); seeds 7-11/22 |
| LCS loss | V12-49/50, loss audit | multi-step loss K ≤ 7, learned J with violation-QP + full-LCP forward, contact-mode statistic | B (J 0) 7-11/22; J learned needs a full-LCP MPPI solver (not run) |
| latent objective | V12-51..55 | JEPA + SIGReg; JEPA + graph stress (JPG); reconstruction + graph | SIGReg 0/22; JPG 11 / 11 / 5 (seeds 6 / 4 / 5); recon + graph 0/22 |
| more data, same recipe | V12-56, V12-58 | JPG on T2 (on-policy data) and on T1ga (grasp-angle demos) | collapse: 1/22, 0-2/22; MPPI-seated belts drift away from the goal in the latent |
| data support in MPPI | V12-59/59b, diag gv_00 | latent-NN support cost; 6-D EE NN support cost | latent cost wrong direction in replay; EE cost 12/22 (no gain) |
| goal margin | V12-60/61 | pull human-labelled seated finals together, push failed finals ≥ M away | W 1 / M 1 collapses the latent; W 0.1 / M 0.3: T2 1 → 6/22, T1ga_f **16/22** (seed 6) |
| tooling | viewer, diagnoses | replay viewer LCS-accuracy chart, planned-vs-recorded check (V26b up+30) | no bug: the chart is decoded vs decoded; the JEPA post-hoc decoder is ~13 mm off on closed-loop belts |

**What we learned.**
1. Offline prediction error does not select controller models (V12-43, V12-50, V12-56).
   Closed-loop checks are needed, and seeds of one recipe span 0-16/22.
2. The latent's geometry near the goal decides success. When data is added, MPPI-seated
   belts can move away from the goal while "over" belts move closer. A human success label
   (the goal margin) is the first fix that holds this geometry in place.
3. Graph stress in place of SIGReg makes JEPA usable at this data scale. Reconstruction
   + graph does not.
4. Grasp-angle starts need data there: positive angles seat once their demos, failed
   attempts and the goal margin are in training. Negative angles still fail.

**Next.**
- Seed robustness of the V26 recipe (seeds 5, 7, 8; ensemble or closed-loop selection).
- Resume the J&P loop from V26b with the goal margin in the refits and labelled on-policy
  attempts.
- Operator demos at negative grasp angles.
- An H 7 vs H 24 A/B on V26b.

Constraints unchanged: no demo prior, no sim pretraining, real-available signals only,
multi-step training horizon ≤ 7.

**Controller used for every eval from V12-49 on (MPPI):**
- 1024 samples, H 24 steps of 75 ms (1.8 s), 6 iterations; 7 knots are sent.
- u is the per-step EE displacement. Its box is the training data's range ∩ 6 mm / 2° per step.
- Effort cost w_r 0.3 and smoothness cost w_s 0.3. There is no separate EE-velocity cost.

"Whitened" latent distances are z / z_std per dimension, with z_std taken from the training
latents at export. They are not comparable across models.

**Deferred by the user:** robust resets; seed variant D (A's encoder frozen + MILES).

## Run index

Every run of the real-first work. Type: **loop** = on-policy learning loop (collect → label →
refit, several iterations); **sim** = fixed-model closed-loop check in the simulator; **offline**
= analysis on recorded data, no sim. Data dirs are under `data/lcs/realfirst/`.

| run | type | about | models | data | result |
|---|---|---|---|---|---|
| V11-01 | loop | first real-first loop: noisy (`real_like`) clouds, Chamfer recon, latent 16, 2 stages, no gripper clearance | V11a-c | `v1/` | 0/15; latent goal "reached" with the arms 20-50 mm off, UR into the board |
| V11-02 | loop | V11-01 + clean clouds, Sinkhorn recon, latent 10, stage tolerance from the demos, gripper clearance, buffer hygiene | V11d-i | `v1_1/` | 1/40 over 8 iterations; stopped after 2 refit rejections |
| V12-01 | offline | model gates on V11-02's models: rollout error, latent monotonicity, MPPI vs demo actions, fresh vs warm refit, horizon | V11d-i | `cmp/offline/` | warm refits and H 16 kept |
| V12-02 | offline + sim smoke | transport screen: single goal vs latent waypoints vs per-frame tracking, each ± demo action prior | V11d | `cmp/offline/`, `cmp/smoke_transport/` | waypoints + prior best; stage goal mode chosen |
| V12-03 | sim smoke | demo-prefix tooling: replay the demo to the pre_place_2 latch, then MPPI | V11d | `cmp/smoke_prefix/` | 2/2 seated from prefix 1.0 |
| V12-04 | data | 8 extra UR-held demos (set2 gv_03/05, set3 gv_02/03) | - | `cmp/demos_ext2/` | 4/8 engaged; added to every seed buffer |
| V12-05 | loop | demo prefix to the pre_place_2 latch, MPPI with the seated goal only | V12a-c | `cmp/C2_prefix/` | 1/15 |
| V12-06 | loop | V12-05 + demo action prior | V12d-e | `cmp/C3_both/` | 0/4; crashed on export check E3 |
| V12-07 | loop | full task from pre_place_1, latent waypoints + demo action prior, demo grasps only, no prefix | V12g-h | `cmp/C1_transport/` | **8/10**, loop DONE |
| V12-08 | loop | V12-07's config + prefix curriculum + set2/set3 grasp pool, 10 iterations | V12j-s | `cmp/full/` | 7/50, eval best 2/6 |
| V12-09 | offline + sim | waypoint bank: reference demo = nearest by initial latent | V12j-n | `cmp/bank_select/` | 0/4 vs control 0/4 |
| V12-10 | offline + sim | overshoot guard: hold / fade the prior near the seated goal | V12h, V12n | `cmp/overshoot/` | V12h 0/8 vs 1/8 off; V12n overshoot cut, 0/4 |
| V12-11 | offline | behavioural refit gate: which signal separates good from bad refits | V12g-s | `cmp/overshoot/gate_sweep/`, `gate_test/` | hold-frame +y push separates them |
| V12-12 | loop | set2/set3 pool, no curriculum, behavioural gate, warm joint refits | V12t-u | `cmp/full2_gate/` | 1/25 (iteration 0 GPU-contended); stopped after iteration 4 |
| V12-13 | loop | V12-12 with the encoder frozen after the seed, LCS-only refits | V13a | `cmp/full2_lcsonly/` | iteration 0 0/5 (contended); paused |
| V12-14 | offline + sim | demo action prior tied to the current stage instead of the clock | V12h | `cmp/prior_sync/` | 1/2 vs 0/2 (too few to judge) |
| V12-15 | offline | failure census of 165 recorded rollouts | all | `cmp/census/` | +y sweep runaway in 56/140 failures |
| V12-16 | offline | data-support position box for MPPI: build, checks, replays on the learned model | V12g, h, u | `cmp/support_box/` | stops the runaway on the model when the last stage starts inside the box |
| V12-17 | sim | V12-16's box on V12h, 4 hard grasps: box vs box + V12-14's stage-synced prior | V12h | `cmp/sbox_sim/` | **box 5/8, box + sync 4/8, no box 1/8**; the box stops a sideways (−x) drift |
| V12-18 | loop | V12-12 + the data-support position box (demos only, margin 20 mm), 10 iterations: does on-policy learning improve once the planner stays where the model has data? | V12v… | `cmp/v12_18/` | 2/20 over iterations 0-3, nearly all "over"; paused at iteration 4 (15:52) for V12-20 |
| V12-19 | offline | opt-in waypoint schedule against "over" (A1 drop the pre-motion waypoints, A2 skip ahead, A3 no timeout into the sweep stage) and a pre-latch rule for the behavioural refit gate; open-loop replays of recorded rollouts (incl. V12-20) and a gate sweep over every refit | V12h, V12v, V12w (sweep V12g-x) | `cmp/wp_schedule/` | A1 turns the early sweep into a descent (Σu₀ Franka y while UR z > 50 mm: +62..+83 → −151..−179 mm), engaged controls unchanged; A2 alone skips the latch on 4/4 engaged; A3 blocks 2/4 engaged (latch tol never reached); the pre-latch rule cannot reject V12w |
| V12-20 | sim | belt slanted toward / away from the UR fingers at the start (UR end-effector moved −10/−20/−30/+15 mm): what does the MPC do? V12h + box, 2 snapshot episodes per start + nominal control | V12h | `cmp/ur_slant/`, starts `data/lcs/start_states/ur_slant/` | **moderate slant 5/6, strong slant 0/4**, 2/4 with the stage-synced prior (+15 mm fixed, −30 mm not) |
| V12-21 | sim | belt grasped at an angle by the UR 2F-85 (`make_grasp_variants.py --ur-pivot-deg`: jaw turned about its closing axis at the pick, arm at the nominal pre_place_1 pose; kept angle 9-27°): what does the MPC do? V12h + box (+ stage-synced prior) | V12h | `cmp/ur_slant/pivot_*`, starts `data/lcs/start_states/ur_pivot/` | box: +10° 2/2, +20° 1/2, +30° 0/2, −10/−20/−30° 0/6; + synced prior: +20° 2/2, +30° 2/2, −20/−30° 0/4 |
| V12-22 | sim | V12-19's A1 (waypoints from demo frame 20: drop the pre-motion waypoints) on V12h + box, ± stage-synced prior, on the hard suite (V12-17 grasps, V12-20 UR −30/+15 mm, V12-21 ±20/±30°, nominal); controls = the box arms of V12-17/20/21 | V12h | `cmp/a1_sim/` | box 8/22 → box + A1 9/22; box + sync 12/20 (10 starts) → box + A1 + sync 9/20 (+ nominal 2/2): **A1 not adopted**; E4 controller = box + stage-synced prior |
| V12-23 | data (sim) | MILES collector smoke (`scripts/lcs/miles_seed.py`): set2 gv_00 demo, waypoints 20/30/33/43/53, 4 deviate-and-return cycles each; plus a full-schedule diagnostic pass (41 waypoints, sim-only restore after an invalid cycle) to score the belt check | V12h (latent distance, report-only) | `miles/smoke/` | 18 cycles, 17 valid, 0 grasp lost / unreachable, stop at waypoint 53 (belt check, Chamfer 1.97 > tau 1.82 mm); format loads in lcs_learning; the check vs the demo is dominated by replay drift (121/164 flagged in the diagnostic pass, GT label unchanged in 162/164) |
| V12-24 | data (sim) | MILES seed collection: set2 gv_00 / gv_01 / gv_02 demos, all 41 waypoints, 4 (1×) and 8 (2×) cycles; belt check vs the waypoint arrival cloud, tau 1.82 mm, a disturbed cycle skips the rest of its waypoint (`--on-disturb skip-waypoint`) | V12h (latent distance, report-only) | `miles/v1/`, `miles/v2x/`, manifests `miles/seed_{B,C,B2}.json` | 1×: 482 cycles, 5 disturbed, 0 grasp lost / unreachable, 9692 frames, 716 s robot; 2×: 957 cycles, 8 disturbed, 18717 frames, 1381 s robot; no demo stopped; the check catches 3/42 and 4/94 GT flags, which are marginal (no cycle moves the belt to over / outside) |
| V12-25 | offline | seed models A (V12v: demos + OU excitation) / B (demos + MILES 1×) / C (A's seed + MILES 1×) / B2 (demos + MILES 2×), same recipe; prediction on 87 held-out off-demo rollouts, probes, behaviour (`seed_compare.py`) | V12v, seed_B, seed_C, seed_B2 | `cmp/seed_compare/` | off-demo one-step 13.5 / 12.8 / 11.5 / **9.8 mm**; MILES models descend first before the latch |
| V12-26 | sim | closed-loop hard suite (11 starts × 2) with box + stage-synced prior, seed models A / B2 / C / B | as V12-25 | `cmp/miles_e4/` | **A 11/22, B 2/22, B2 1/22, C 1/22**: the MILES seeds lose in closed loop (seated states no longer close to the seated goal in their latents: d 2.5-5.7 vs 1.6) |
| V12-27 | loop | Jin & Posa loop on the V12-29b MILES seed (V14 models): V12-18's seed + V12-29b detours weighted to 40 % of the seed's frames, anchor buffer (demo + miles, 40 %), LCS-only refits, fixed rotation over 10 hard starts, box + stage-synced prior, eval every 2 (seat6 + hard suite) | V14a… | `cmp/v12_27/` | **stopped** after iteration 1 (10:11): its seed was V12-18's seed (20 demos + 14 OU) + MILES, not the intended 3 demos + MILES → V12-35. Iter 0 (V14a, set2/gv_00) 4/5; iter 1 refit rejected (hold push +1.20 vs −0.10), set2/gv_06 1/5 |
| V12-28 | plan | MILES-BC prior as the MPPI mean (P-M3; π(z) only) | - | `cmp/bc_prior/` | not run: on hold under the no-demo-prior rule (a state-conditioned policy as the MPPI mean needs the user's go-ahead) |
| V12-29 | data (sim) | larger-deviation MILES (user request): 3 demos with different grasps (set2 gv_00, set2 gv_02, set3 gv_02), both arms every cycle, translation + rotation every draw, free ±30 mm / ±8°, contact ±10 mm / ±4° (clearance-limited), 4 cycles / waypoint, recorded for the viewer | - | `data/lcs/realfirst/miles/v12_29/` | 337 cycles (319 valid), 8455 frames, 636 s robot, 3 human resets; every demo stopped at an unreachable return (gv_00 at waypoint 55, gv_02 at 188, s3_gv_02 at 198); superseded by V12-29b's motion-based waypoints |
| V12-29b | data (sim) | = V12-29 with motion-based waypoints (user choice): one per 5 mm of summed Franka + UR EE path from motion start (every frame 20-52 here) + 4 in the static rest; 38-39 waypoints / 152-156 cycles per demo | - | `data/lcs/realfirst/miles/v12_29b/` | 333 cycles (285 valid), no demo stopped; most descent waypoints (20-33) skipped after their first cycle (belt check at 1.82 mm everywhere, but free-phase demo spread is 3.3 mm); seed V14a: one-step 3.82 mm, seed check 1/5 |
| V12-32 | offline | MPPI hyperparameter sensitivity on V12h + box + synced prior (open-loop replay of 12 hard-start episodes + closed loop on the learned LCS; 21 one-factor settings + 6 pairs) | V12h | `cmp/mppi_sweep/` (`tables.md`, `metrics.json`, `candidates/`) | shortlist: H 24 + w_q 0.5; w_q 0.5 + prior weight 0.3 |
| V12-33 | sim | controller A/B on V12h, hard suite 11 × 2: box + synced prior vs + `stage_start: progress` (V12-31) vs + strict goal set (V12-30) vs both | V12h | `cmp/ctrl_ab/` | sync 10/22, + start 11/22, + goal set 11/22, both 10/22: no net gain; per-start wins and losses swap (start: UR −30 mm 0→2/2 but grasp +20/+30° 3/4→0/4) |
| V12-34 | sim | V12-32's top two MPPI settings on V12h, hard suite 11 × 2 | V12h | `cmp/mppi_ab/` | H 24 + w_q 0.5 8/22, w_q 0.5 + prior weight 0.3 8/22: worse than the base (10/22) |
| V12-35 | loop | Jin & Posa loop seeded ONLY by the 3 MILES reference demos (set2 gv_00, set2 gv_02, set3 gv_02) + V12-29b MILES (unweighted): anchor buffer (demo + miles, 40 %), LCS-only refits, fixed rotation over 10 hard starts, box (demos + MILES support, 10 mm) + stage-synced prior, eval every 2 (seat6 + hard suite) | V15a… | `cmp/v12_35/`, config `cmp/configs/v12_35.yaml` | from 10:11; iter 0 V15a 0/5 (set2/gv_00); iter 1 refit rejected (7-step 6.85 mm), V15a kept, 3/5 (set2/gv_06). From iteration 2: `refit_gate: report` (user): every refit is deployed, gate verdicts logged only; iteration 2 warm-starts from iteration 1's rejected candidate |; iterations 2-5 (V15b-e, all deployed): 0/5 each (over / outside), behaviour gate descent cosine 0.95 → 0.43 → 0.27, recon 3.3 → 6.1 mm; **stopped by the user** after iteration 5 (11:45) → V12-36 |
| V12-37 | sim | honest baseline without the demo action prior (user rule): V12h + support box + latent waypoints, prior off, horizon 16 vs 24; hard suite 11 × 2 | V12h | `cmp/noprior/` | **no prior: H 16 5/22, H 24 7/22** vs 10/22 with the prior (V12-33 base); failures over / slanted; the prior carried about half of the successes |
| V12-38 | sim | no prior, longer planning: V12h + box + waypoints, H 32 vs H 24 + 10 MPPI iterations; hard suite 11 × 2 | V12h | `cmp/noprior2/` | stopped by the user at 14:16 after 5-6 of 11 starts: H 32 0/10, H 24 + 10 iterations 3/10 (= plain H 24 on the same starts); longer horizons compound model error, more iterations add nothing |
| V12-36 | code + offline | task-driven refits: opt-in per-frame refit weights (`frame_weights`), on-policy frames past the off-track cut (latent outside tol_k × waypoint tol of every goal-demo frame) down-weighted (`onpolicy_weighting`), every refit from the seed (`refit.from_seed`); config v12_36.yaml = V12-35 + these + anchor 0.7, **no demo action prior** (banned) | V15a/b (offline arms) | `cmp/refit_weighting/`, config `cmp/configs/v12_36.yaml` | loop 16:45-19:32 (iteration 5 eval crashed once on the sim's malloc bug and was redone): collections 0,4,0,1,0,1,0,0,0,0 /5 (rotation over 10 hard starts); **evals 4, 3, 6, 3, 1 /16** at iterations 1/3/5/7/9; from-seed refits stopped V12-35's collapse but did not improve steadily, and the last two refits (trained on all-under / all-over rotations) fell to 1/16 |
| V12-39 | offline + code | learned cost-to-go V(z) in V12h's latent as an opt-in MPPI term (`MppiParams.value`): TD(λ) low-expectile value from demos, MILES, 397 labelled rollouts (hindsight seat time, failure = never), trained also on LCS-predicted latents, V_max off the data; open-loop / learned-LCS replays of V12-37 | V12h | `cmp/value_cost/` | sim (15:30-16:45, V12h, no prior, H 24): waypoints + V weight 30 **6/22**, weight 100 2/22, seated goal only + V 100 2/22 vs control 7/22; gains UR −30 mm (0→2/2) and grasp +30° (0→2/2) but loses nominal (2→0) and the set2/set3 grasps (3→0, all over): the value holds back the sweep (Franka y max median 42-43 vs 60 mm) |
| V12-30 | offline | seated goal SET for the last stage: soft-min over encoded seated latents (demos ± MPC successes) instead of the goal demo's one latent; separation on 219 rollouts, open-loop replays, `MppiParams.goal_set`, loop key `goal_set` | V12h, V12v, B, C, B2 | `cmp/goal_set/` | AUC engaged vs not, V12h / B2: single 0.956 / 0.446, strict demos + successes 0.962 / 0.784; replay: the 14 `label_human` demos add +y on over states (a borderline demo), the 9 strict demos cut it (−76..−113 vs −1..−40 mm); sim A/B pending |
| V12-31 | offline | grasp −20°/−30° starts: why the planner sweeps early when the first waypoint (the demo's static start) is out of reach (LCS cost terms at the start latent), and an opt-in progress-gated start (`MppiParams.stage_start: progress`, loop `waypoints.start`): drop the start-state waypoints the latent is outside, start at the nearest remaining pre-latch stage | V12h | `cmp/start_stage/` | the tracking + terminal cost toward the static start drives the sweep (−20°: 'sweep' 127-129 cheaper than 'descend UR', the prior +4); replay Σu₀ Franka y / UR z 0-3 s +4..+71 / −21..−82 → −71..−95 / −162..−189 mm on 12 failing episodes; in-reach controls unchanged; +30° changes (risk); −10..−30° belt stays 11-27 mm up at the UR's lowest z (second cause); sim pending |
| V12-40 | code + offline | C3+ (magna's C++ learned MPC, `projection_type: C3+`) as the controller for the current latent deploys: params generator, loop key `controller: c3plus`, hard-suite C3+ arms, C++ load mirror, E3 at 1e-2 + exact-LCP consistency on task states, solve-time estimate, what the support box / clearance did in MPPI runs | V12h, V16b | `cmp/c3plus/` | A/B 19:32-20:30 (no prior, 11 hard starts × 2): **C3+ with V12h 6/22** (engaged 6, under 6, slanted 7, over 3; nominal 2/2, UR +15 mm 2/2, UR −30 mm 1/2) vs MPPI with V12h 5/22 (H 16, V12-37) / 7/22 (H 24); **C3+ with V16b 0/22** (Franka y max median 8 mm: no sweep, under 10) vs MPPI with V16b 5/22 (set2/gv_06 2/2, grasp +20/+30° 3/4); C3+ solve p95 7-8 ms |
| V12-41 | offline | encoder accuracy readout: held-out reconstruction of the seed / loop models vs reconstruction and one-step prediction on V12-25's 87 off-demo closed-loop rollouts, by distance from the demo tube | V12v, V12h, V16a, V16b, V16f, seeds B / C / B2 | refit reports, `cmp/seed_compare/report.md` | held-out 2.5-3.2 mm (V16f 5.0) vs off-demo 9.6-13.4 mm (12.4-17.4 mm beyond 40 mm from the tube); one-step ≈ recon: the encoder limits accuracy where the controller goes |
| V12-42 | offline | encoder study for little real data: real-budget tiers T0 (3 demos, 45 s) / T1 (+ V12-29b MILES) / T2 (+ V12-35/36 on-policy; no all-data tier, user: too slow) × arms E0 today's recipe, E1 + point augmentation, E2 + Delta-JEPA action head (Δz → actions, k 1 and 5, λ 1/10/50), E3 + latent 32, E4 external Point-MAE backbone (ShapeNet-pretrained; frozen / partly fine-tuned); scored on V12-25's off-demo rollouts and the no-prior hard-suite rollouts (recon, one-/7-step, action-response and action-decodability probes, +y probe) | V17… | `cmp/encoder_study/` | started 2026-10-08 evening (code: trainer options, Point-MAE wrapper, tiers + `encoder_eval.py`) |
| V12-43 | sim | hard suite (11 × 2, no prior, box) with V12-42's winners, MPPI H 24 (+ C3+ for the best) vs V12h 7/22 / 6/22 | V17a, V17b | `cmp/encoder_study/sim/` (`vs_v12_37/hard_suite.md`) | 02:55-03:45: **V17a (E0, T1 = 3 demos + MILES) 8/22** (over 13, slanted 1; nominal 2/2, UR +15 mm 2/2, set2/set3 grasps 4/8) vs **V17b (E2 aug + head λ 0.1) 1/22** (over 18, under 2): the better-predicting encoder controls far worse (+y bias of the action head, diagnosis in §7). V17a matches V12h's 7/22 (V12-37, 20 demos) with 3 demos + 12 min of MILES |
| V12-44 | sim | V12-43's diagnosis fix: `stage_start: progress` (start at the nearest reachable pre-latch waypoint) on V17a (E0) and V17c (E1, augmentation only); hard suite 11 × 2, no prior | V17a, V17c | `cmp/encoder_study/sim44/` (`vs_v12_43/hard_suite.md`) | 03:45-04:35: **V17a + progress 7/22** (vs 8/22 without: + UR −30 mm 1/2, the first without the prior, and set3 gv_02 2/2; − UR +15 mm 1/2 and nominal 0/2 slanted): mode 1 removed, no net gain (as V12-33 with the prior); **V17c (augmentation only) 0/22** (mode 3: belt down, then lifted out without the Franka −x pull) |
| V12-45 | sim | gate validation within today's recipe: V17d = T1 E0 seed 6 (response gate pass) vs V17e = T1 E0 seed 5 (gate fail), `stage_start: progress`, hard suite 11 × 2, no prior | V17d, V17e | `cmp/encoder_study/sim45/` (`vs_all/hard_suite.md`) | 04:35-05:23: **V17d (E0 seed 6, gate pass) 11/22** (the best without a demo prior so far; all 4 set2/set3 grasps 7/8, UR +15 mm 2/2, nominal 2/2), **V17e (E0 seed 5, gate fail) 9/22** (UR −30 mm 2/2, under 8). Same recipe and data: seeds 4/5/6 give 7, 9, 11/22 with the progress start; the gate does not rank seeds within E0 (V17e beats V17a); it only flags the extreme models (V17b, V17c) |
| V12-46 | loop | Jin & Posa loop seeded by V17d (E0 seed 6, 11/22; new opt-in loop key `seed_model`), = V12-36's rules (no prior, anchor 0.7, from-seed LCS-only refits, on-policy weighting, `refit_gate: report`, rotation over the 10 hard starts) + H 24 + `waypoints.start: progress`; report-only response gate per iteration (`scripts/lcs/response_gate.py`); eval (16 episodes, V12-36's set) at iteration 0 (extra baseline) and 1/3/5/7/9 | V18a (= V17d)… | `cmp/v12_46/`, config `cmp/configs/v12_46.yaml` | 06:15-09:18 (~3 h, 50 MPC rollouts + 96 eval episodes, 6524 s robot): collects 5, 5, 5, 4, 3, 5, 0, 0, 1, 2 /5 over the rotation (set2 gv_00, gv_06, set3 gv_02, gv_03, UR −30, UR +15, grasp −30°, −20°, +20°, +30°); **evals (16 episodes) 8 → 11 → 11 → 5 → 10 → 5** at iterations 0/1/3/5/7/9. The first on-policy refit gives the campaign's best no-prior model (V18b/V18d 11/16: every start but the grasp angles and one more); the grasp angles stay 0 in eval (collect +20° 1/5, +30° 2/5: the first grasp-angle seats); after the grasp-angle iterations the model degrades on the other starts (V18j 5/16, slanted/other). Response gate (report-only) fails on every refit, including the 11/16 ones |
| V12-47 | sim | V18b (V12-46 iteration 1, 11/16 on the loop eval) on the hard suite (11 × 2, noise 0.2, `stage_start: progress`, its loop yaml), to compare directly with V17d's 11/22 (V12-45) | V18b | `cmp/encoder_study/sim47/` (`vs_all/hard_suite.md`) | 11:00-11:23: **10/22** (over 4, slanted 4, under 4) vs V17d 11/22: same level, different starts (+ UR −30 mm 2/2, set3 gv_03 2/2; − set2 gv_06 0/2, UR +15 mm 1/2, nominal 1/2); grasp angles 0/8. The loop eval's 8 → 11/16 gain is not confirmed on the hard suite: within noise, the first refit traded starts |
| V12-48 | sim | robot-motion oscillation (user observation): V18b on the hard suite with `action_noise_frac` 0 (the V12-37-derived hard-suite yaml injects 0.2 exploration noise into every executed command) vs 0 + `w_s` 1.0 (smoothness, incl. continuity with the last executed command; was 0.3). Measured on V12-47: commands change by 0.9-1.1× their size per step (demos 0.1-0.3), UR y reverses on 35 % of steps even without noise (demos 8 %), injected noise doubles UR x/z flips and EE second difference | V18b | `cmp/encoder_study/sim48/` (`vs_all/hard_suite.md`, oscillation in `chain.log`) | 11:42-12:30: **noise 0: 10/22** (= V12-47's 10/22 with noise 0.2; set2 gv_06 0 → 2/2, set3 gv_03 / UR −30 2 → 1/2); **noise 0 + w_s 1.0: 2/22** (slanted 9: smoothing damps the final corrections). Oscillation, noise 0.2 → 0: UR x/z sign flips 0.21/0.26 → 0.07/0.11, EE second difference F y/z 0.49/0.24 → 0.34/0.11 mm (demo-like); UR y keeps flipping (0.35) in every arm, incl. w_s 1.0 (0.34): a near-null latent direction, not a smoothing problem. Decision: noise 0 for evaluation (smoother, same success); keep w_s 0.3 |
| V12-49 | code + offline + sim | multi-step LCS loss for the belt model (user request): rollout ẑ_{t+j+1} = LCS(ẑ_{t+j}, recorded u_{t+j}) from ẑ_t = enc(o_t) with the training PGD; multi-step violation (dynamics vs the encoded recorded states + LCP terms at each step, γ^j, steps 1..K−1) up to K = 24 + a multi-step Sinkhorn reconstruction anchor at j = 1, 3, 7, 24; T1, V17d's recipe, 3 seeds × K 7 / 24; baseline = V17d at noise 0 on the hard suite | V19… | `cmp/encoder_study/sim49/` (baseline), trainer flags `--belt-ms-*` | from 12:50. Baseline: **V17d at noise 0: 9/22** (13:24-14:05; vs 11/22 at noise 0.2 in V12-45, V18b 10/22 at noise 0): within noise the T1 models sit at 9-11/22. Code done (`--belt-ms-*`, `--lcs-symmetric-f`, contact-mode statistic, 28/28 checks). Multi-step horizon ≤ 7 (user; K 24 also has too few windows in T1: MILES segments are 24 frames). Training from 14:40 (T1, seeds 4/5/6): B = J 0; E7 = K 7 + recon 1,3,7 (J learned); E7s = both. Next: C = J learned + L4DC eq.-9 violation + eq.-7 full-F forward solver; D = C + K 7 (code in progress) |
| V12-50 | sim | closed loop for the V12-49 candidates that MPPI simulates exactly (J 0): V19a = B seed 6 (most accurate: held-out 2.99 / 4.26 mm, off-demo 9.06 / 10.07), V19c = B seed 5 (3 contact modes); noise 0, progress start, H 24; vs V17d 9/22 (V12-49 baseline). C models (J learned) are excluded until MPPI runs the full-LCP solver: C seed 6's export fails E3 (MPPI's 25-iteration symmetric PGD vs the trained full-LCP step 0.025-0.044) | V19a, V19c | `cmp/encoder_study/sim50/` | 17:03-18:05: **V19c (B seed 5, J 0, 3 contact modes) 11/22** (best at noise 0; nominal 2/2, UR −30 2/2, UR +15 2/2, set3 gv_03 2/2), **V19a (B seed 6, most accurate offline) 7/22**; vs V17d 9/22, V18b 10/22 at noise 0. The offline-best seed again did not win; the multi-mode seed did. Grasp angles 0/8 for both |
| V12-51 | code + offline + sim | JEPA-style training (user): no reconstruction, no decoded next state, no aux head; LCS as the predictor with multi-step latent prediction (K ≤ 7, γ^k) against an EMA target encoder (stop-gradient) + LCP terms, SIGReg anti-collapse; decoder fitted afterwards on the frozen encoder (viewer / scoring only). Arms JP0 (J 0) and JPq (J learned, violation-QP loss, full-LCP forward), 3 seeds each on T1 | V20… | lcs_learning `outputs/sim_belt_v12_20261006/v12_51/` | code done (`--jepa`, `--jepa-horizon/-gamma/-ema-tau`, `--sigreg-weight/-projections`, `--jepa-decoder-epochs`; `scripts/check_jepa.py` all pass, defaults bit-identical). Single-stage K 7 from scratch collapses on T1 (effective rank 1.004 / 10 after 60 epochs; SIGReg up to weight 50 does not stop it); K 1 for 20 epochs then warm K 7 keeps rank 8 → two-stage recipe. Caution from the probes: latent prediction is no better than predicting no change (whitened k 1 0.46 vs hold 0.41, k 7 1.83 vs 1.84). Training queued after V12-50: JP0 seeds 4/5/6 (J 0, ~17 min each), JPq seed 4 (J learned, ~95 min) |
| V12-52 | code + offline + sim | JEPA with the V10 graph stress loss instead of SIGReg (user): obstacle-aware graph rebuilt on T1's segmented camera clouds (Sinkhorn/OT matching for the blend, pulley SDF on cloud points, robust clearance and edge-length quantiles, duplicate edges removed — V10's graph.py summed them), stress + far terms on the latent (1 unit = 50 mm of legal motion); arm JPG (J 0, two-stage) 3 seeds | V21… | lcs_learning `outputs/sim_belt_v12_20261006/v12_52/` (graphs `graphs/t1_s3.npz`, `t2_s16.npz`) | code done (`scripts/build_cloud_graph.py`, `lcs_learning/graph_latent.py`, trainer/refit `--graph-*`; `check_graph_reg.py` 27/27, defaults bit-identical; builder on GT centrelines = graph.py exactly; the dedupe fix changes V10-style D by median 12.8 %). Cloud graphs: T1 3,016 nodes, T2 7,407; one component each; only 0.2-0.3 % of candidates rejected; cloud test catches 30-38 % of GT-blocked candidates; D vs GT-graph D Pearson 0.96 (T1) / 0.90 (T2); straight moves blocked 11 % (T1) / 28 % (T2) of node-landmark pairs, which then take 3.1-3.5 × vs 2.3-2.4 × the straight distance: weak but real obstacle signal. Training from 19:05 on the T1 graph, W 1: JPG (JEPA + graph, SIGReg off, J 0, two-stage) and BG (today's recipe, J 0, + graph), seeds 4/5/6 |
| V12-53 | sim | JEPA model in closed loop (user): V20a = V12-51 JP0 seed 6 (SIGReg JEPA, J 0, post-hoc decoder; best JEPA seed offline but off-demo one-step 11.96 mm vs B 9.4); hard suite 11 × 2, noise 0, progress start, vs V19c 11/22 / V17d 9/22. JPq (J learned) stopped at stage 2 to free the GPU (MPPI cannot run full_lcp yet). Cache-reproducibility tolerance made opt-in configurable (`LCS_CACHE_REPRO_TOL_REL`, default unchanged) because unit-scale JEPA latents exceed the 1e-6 relative float32 check | V20a | `cmp/encoder_study/sim53/` (`vs_all/hard_suite.md`) | 18:25-18:58: **0/22** (over 17, under 4, other 1), every start including nominal and UR +15 mm; the belt is swept while still 10.7 mm above the groove (median h at Franka y 45 mm, vs 1.7 for V19c). Response probe: Franka +x toward the seat at 86 % of states (as V17c, 0/22), but V19c (11/22) shows similar +x and stronger +y, so the probe does not separate them. **SIGReg JEPA does not work at this data scale** (offline off-demo 12-13 mm vs 9.4, weaker action readout, closed loop 0/22) |
| V12-54 | sim | graph-regularised models in closed loop: V21a = V12-52 JPG seed 6 (JEPA + graph stress, SIGReg off, J 0; held-out 3.29 / 4.48 mm but off-demo 12.30 / 13.11) and V21b = BG seed 6 (reconstruction recipe + graph stress, J 0; held-out 3.12 / 4.23, off-demo 9.21 / 10.15, hard-suite rollouts 10.88 / 11.56 — the best BG seed; BG mean off-demo 10.7 vs B 9.4, action R² 0.13-0.15 vs 0.21); hard suite 11 × 2, noise 0, vs V19c 11/22 | V21a, V21b | `cmp/encoder_study/sim54/` (`vs_all/hard_suite.md`) | 19:30-20:20: **V21a (JEPA + graph, no reconstruction) 11/22** = the best (ties V19c; nominal 2/2, UR −30 2/2, UR +15 2/2, set3 gv_03 2/2), sweep starts with the belt 2.8 mm up (V19c 1.7). **V21b (reconstruction + graph) 0/22**, over 18, sweep with the belt 9.9 mm up, like SIGReg JEPA (V20a 0/22, 10.7 mm). Replacing SIGReg by the graph stress turns JEPA from 0/22 into 11/22 (one seed); adding the graph to the reconstruction recipe breaks it although its offline errors equal B seed 6's. Offline prediction error again does not rank closed-loop results |
| V12-55 | sim | seed robustness of JEPA + graph: V21c = JPG seed 4, V21d = JPG seed 5; hard suite 11 × 2, noise 0, vs V21a (seed 6) 11/22 | V21c, V21d | `cmp/encoder_study/sim55/` (`vs_all/hard_suite.md`) | 20:30-21:22: **JPG seed 4 11/22, seed 5 5/22** (seed 6 11/22 in V12-54): JEPA + graph mean 9.0 / 22 over 3 seeds vs SIGReg JEPA 0/22 and reconstruction B seeds 4-6 (offline-chosen 5 and 6) 11 and 7 /22; seed 5 was also the weakest offline (held-out 4.39 / 5.54). Seed 4 starts the sweep with the belt fully down (median h 0.0 mm at Franka y 45). Grasp angles 0/8 for every seed |
| V12-56 | offline + sim | JEPA + graph on tier T2 (T1 + V12-35/36 on-policy, 150 min robot) with the T2 cloud graph (2.5× more lip-blocked pairs): JPG2 seeds 4 and 6, two-stage, J 0, SIGReg off; then offline scoring and the hard suite vs the T1 JPG seeds (11/22) | V23a (s4), V23b (s6) | lcs_learning `outputs/sim_belt_v12_20261006/v12_56/`; offline `cmp/encoder_study/v12_56_eval_s4/`; sim `cmp/encoder_study/sim56/` | seed 4 (stage 2 cut to 5 epochs: 174 s/epoch at the first attempt): offline one-step / H7 Chamfer **5.37 / 5.59 mm off-demo, 8.22 / 8.48 hard-suite rollouts** vs T1 JPG seed 4 12.69 / 15.68 and 13.76 / 16.15; action R² unchanged (k5 0.39 / 0.18). Seed 6: 6.23 / 6.33 and 8.48 / 8.63. **Hard suite (noise 0, 22:21-23:13): V23a 1/22 (over 11, slanted 6, under 4), V23b 1/22 (over 13, under 4, slanted 2, other 2)** vs T1 JPG seeds 4 and 6 11/22: halving the offline error broke the closed loop. Sweep starts with the belt up (median h at Franka y 45: 7.2 / 5.5 mm vs 0.0 / 2.8 for the T1 seeds; V21b 9.9 at 0/22); h at the sweep ranks closed-loop success across all six JEPA arms, offline Chamfer does not. Seeding V12-57 stays on T1 JPG seed 4 |
| V12-57 | loop (prep) | Jin & Posa loop seeded by JPG seed 4 (V21c, 11/22) with the JEPA latent loss in the refits, the cloud graph rebuilt from each iteration's buffer (on-policy failures add lip-blocked states), and keep-best deployment (every refit trained and kept; the best model by eval is deployed) | V22… | `cmp/v12_57/`, config `cmp/configs/v12_57.yaml` | config final: joint refits from the seed (JEPA K=7 + graph inherited from the checkpoint, 30 epochs, no on-policy frame weights), graph rebuilt each iteration (`--knn-coarse 64`), keep-best with min_gain 1 (a candidate must engage ≥ 1 more of 22); launched 23:11; iteration 0: V22a 12/22 (no J&P iteration yet; 12/14 without the grasp-angle starts). **Iteration 1: V22b 2/22** (over 18: nominal, ur_z±, set3 gv_03 all over; keep-best kept V22a). Cause (offline, `v12_57` deploys): not the graph (rebuilt graph = T1 graph on the T1 nodes, distance-to-goal corr 0.999; the 5 new on-policy episodes end 97-237 mm from the goal) but where the refit puts seated belts: V22a's 12 engaged final clouds are 1.99 (whitened) from the nearest demo final in V22a's latent and 5.47 in V22b's, its over finals 2.29 / 4.47 (over closer than seated in V22b). The only seated training states are the 3 demo finals + MILES rest frames, so the place of MPPI-seated belts in the latent is unconstrained and moves between trainings (also the seed spread 11 / 11 / 5 and the T2 collapse); with anchor 0.7 the 5 on-policy failure episodes carry 30 % of the refit's frames. Paused after iteration 1 for V12-58 |
| V12-58 | data (sim) + offline + sim | grasp-angle coverage (user request): new UR grasp-angle starts ±15° / ±25° (`start_states/ur_pivot_train`; the hard suite keeps ±20° / ±30° for eval), scripted demos (the teleop stand-in: nominal place, then a UR place-offset grid dz / tilt = the operator adapting; one success per angle), MILES around them (V12-29b settings), tier T1ga = T1 + these; V22a's recipe (JPG two-stage, J 0, SIGReg off, graph rebuilt on T1ga) seeds 4 / 6; hard suite noise 0 vs V22a 12/22 (grasp angles 0/8). V12-57 paused after iteration 1 for the GPU, resumed at the end | V24a (s4), V24b (s6) | `cmp/v12_58/` (`night3.sh`, `night.status`, `probe_*`), sim `cmp/encoder_study/sim58/` | starts ±15/±25 made 00:50 (all held). Nominal scripted place fails at every angle (negative: over, h +11..+17 mm, roll−; positive: under, h −18..−19 mm, roll+; the fallback offset too). Per-angle grid (2 demos per setting, UR tilt opposite to the grasp angle): **up−15 UR +1.5 mm / +14° 2/2, up+15 +1.5 / −14° 2/2, up−25 +1.5 / +26° 1/2 (Franka-down offsets, +3 mm UR and ≤ +20° all over), up+25 +1.5 / −32° 2/2** (−8..−26° under, h −19 → −12 mm): the operator's compensating tilt ≈ −1.0 to −1.3 × the grasp angle. 01:20 MILES on the 4 picked demos (16 min, 499 segments); T1ga = 352 + 4 demos + 499 MILES (graph 7178 nodes, one component). **Hard suite (02:02-02:53): V24a 0/22 (over 17, outside 3, other 2), V24b 2/22 (engaged 2 = up+30 2/2, the first grasp-angle seat without a demo prior; up+20 slanted 2/2 instead of under; every other start over)**. Same collapse as T2 / V22b: in V24a's latent V22a's seated finals are 3.62 from the demo finals and V24a's own over / outside finals 2.05-2.25; in V24b 1.71 vs 2.39 (MPPI settles in over) |
| V12-59 | sim | EE data-support MPPI cost (`MppiParams.ee_support`, opt-in: weight × Σ_k relu((d_k − 35 mm)/10 mm)², d_k = 6-D NN distance of the planned Franka + UR positions to V21c's training EE positions, pulley frame; replay on set2 gv_00 r1 slows then reverses the +x drive, `cmp/latent_support/README.md`) on V21c: weights 1 and 10, hard suite noise 0 vs V21c 11/22 | V21c_n0_ees1, V21c_n0_ees10 | `cmp/encoder_study/sim59/` (`chain.sh`); re-run `sim59b/` | 02:54-03:25, **invalid**: every finished episode over (10/10 starts done incl. V21c's 2/2 starts) because the term's NN (cdist direct-difference mode) made MPPI solves 757 ms (median) vs 17 ms, so the arms acted on 3 Hz plans; logged EE distances were sane (0-32 mm). Stopped; `nn_dist` now a centred float32 matmul expansion (2.7 ms per call on the GPU, 2e-4 mm from fp64; `check_latent_support.py` 9/9 with chunk tolerances 1e-4 / 1e-3 mm). **V12-59b** (07:0x-07:48, fast NN, solves 39 ms median vs 18 without the term): **ees1 12/22, ees10 12/22** vs V21c 11/22 and V22a 12/22: no measurable gain; grasp-angle starts unchanged (up+ under, up− over). Not adopted |
| V12-60 | offline + sim | label-aware goal margin (new opt-in loss, lcs_learning `goal_margin.py`, `--goal-margin-weight/-margin/-pull-radius/-last-n`): last 13 frames of every task attempt labelled success by the human label (+ demo finals) = SEATED, of every failed attempt = FAILED (MILES excluded; sim GT never used); pull SEATED within 0.3 latent units of other attempts' seated frames, push FAILED ≥ 1.0 (≈ 50 mm with the graph scale) from them. T2 has 36 seated / 174 failed attempts. T2 JPG2 (V12-56's recipe, collapsed to 1/22) + goal margin W 1, M 1.0, seeds 4 / 6; hard suite noise 0 | V25a (s4), V25b (s6) | `cmp/v12_60/` (`chain.sh`), sim `cmp/encoder_study/sim60/` | training from 03:26 (V12-57 iteration 2 stopped at its start, stays paused). **W 1 / M 1.0 collapses the latent**: stage 1 seed 4 val eff rank 1.0-1.2 (6.0 without the term), graph Spearman 0.39 (0.98), val graph stress 11 → 157 (0.03), held-out push violation still 42 %; the baseline latent is ~0.44 per dim, so M 1.0 (≈ 50 mm) is far beyond the over / seated gap. Stopped 03:37; restart (`chain2.sh`): stage-1 sweep (W, M) ∈ (0.1, 0.3), (0.3, 0.3), (0.1, 0.6), pick = lowest held-out violation with eff rank ≥ 4.5 and Spearman ≥ 0.95, then stage 2 + seed 6 + hard suite; V12-61 uses the same pick. Sweep (stage 1 seed 4, val, epoch 20; baseline eff rank 5.98, Spearman 0.982): W 0.1 / M 0.3 rank 3.90, Spearman 0.969, push violation 0, sep 5.8; W 0.3 / M 0.3 4.04 / 0.961 / 0 / 5.3; W 0.1 / M 0.6 1.56 / 0.925 / 0.07 / 5.6. None reached rank 4.5; continued with **W 0.1 / M 0.3** (the T1 JPG models that seated 11/22 had rank ~3) from 04:12. **Hard suite (04:37-05:26): V25a 6/22 (engaged 6, slanted 6, over 5, under 5), V25b 1/22 (slanted 12)** vs the same recipe without the term V23a/b 1/22 each (over 11-13): the term moves T2's failures from over to slanted and lifts seed 4 from 1 to 6; still below T1 JPG (11/22) |
| V12-61 | offline + sim | V22a's recipe + goal margin (V12-60's sweep pick) on T1ga_f = T1ga + every V12-58 probe demo (42 more: 39 failed operator attempts at the ±15/±25 starts = the FAILED set, 3 more successes); graph rebuilt on T1ga_f; seeds 4 / 6; hard suite noise 0 vs V24a/b | V26a (s4), V26b (s6) | `cmp/v12_61/` (`chain3.sh`), sim `cmp/encoder_study/sim61/` | goal margin W 0.1 / M 0.3 (V12-60's pick). Hard suite 06:1x-06:55: **V26b 16/22 — best so far** (both repeats of nominal, set2 gv_00, set3 gv_02, set3 gv_03, ur_z+15, ur_z−30, **up+20 and up+30**; set2 gv_06 other 2/2; up−20 / up−30 over 2/2; 12/14 on the non-angle starts = V22a, + 4/4 positive grasp angles; median h at the sweep −0.7 mm). **V26a 0/22** (over 17, other 4): the seed lottery remains (V12-55: JPG seeds 11 / 11 / 5). Negative grasp angles still fail for every model |

## 1. Goal

A framework that learns dual-arm round-belt seating from little real data: colour-segmented
belt cloud -> belt-cloud PointNet encoder -> latent LCS -> MPPI, improved iteratively from
on-policy rollouts in the style of Jin & Posa (arXiv 2211.16657; the Newton sim stands in for
the robot). Success = the emulated human label (GT `engaged` at the end of the episode) on
MPC rollouts, rising with iterations; eval = the held-out seat6 cells (sim GT, report-only).

## 2. Constraints

No EE-pose targets in any MPC cost (EE positions are state with zero cost; floors, ceilings,
+-20 deg envelope, +-6 mm / +-2 deg input box and the gripper clearance are the only EE
constraints). Only real-available signals in training and control; sim GT for reporting and
the one-bit human label. No hand-designed alignment features; belt-observation goals (encoded
demo clouds, per-frame demo latents) and demo action priors are allowed. MPC covers all steps
(a replayed demo prefix is a start-state choice, eval always runs the full task from
pre_place_1 without a prefix). **No demo action prior (user, 2026-10-07):** the demo action prior (clock- or
stage-indexed demo actions as the MPPI mean / cost) counts as cheating and fails for different
initial conditions; from V12-37 on it is off, and earlier prior-on results are historical. **No simulation pretraining (user, 2026-10-08):** the sim stands in for the real system, so no component (encoder included) is pretrained on extra sim data; every sim frame counts as real budget. External models pretrained on public data are allowed. Clean clouds (`--cloud-noise off`, segmentation on). Private LCM
ports >= 7780; one sim at a time; new data dirs only (`data/lcs/realfirst/cmp/`,
`data/lcs/mpc_eval/20261006_cmp_*`, lcs_learning `outputs/sim_belt_v12_20261006/`); commits only when the user asks.

**Aux head (in every belt-cloud model since V11-01, `+aux` in the model tags; made explicit 2026-10-08).**
The belt-cloud encoder trains a small aux head on the latent, weight 0.1 (L1), whose targets are hand-designed
geometric features of the segmented belt cloud relative to the known large pulley
(`src/task_common/belt_cloud_features.py`, stored as `sim_cloud_h_mm` / `sim_cloud_wrap_deg` / `sim_cloud_r_mm`):
median height of the belt over the groove seat plane h (h/50), wrap arc (sin, cos) and radial offset r
((r − 47)/10). They are real-available (camera cloud + known pulley pose) and used only as training
supervision, not in the MPC cost or state, but they inject task knowledge (the success measures h and wrap)
into the latent. Refit default `aux_feature_weight: 0.1` (no CLI flag yet). Ablation pending.

## 3. Baseline: V11-01 and V11-02

| run | config | MPC successes | stage 1 reached | eval (seat6) | main failure |
|---|---|---|---|---|---|
| V11-01 (V11a-c), 2026-10-05 | real_like clouds, Chamfer-16, 2 stages [4, 30] s, no clearance | 0/15 | n/a (tol fired falsely with arms 20-50 mm off) | 0/6, 6 unsafe | UR driven down state-independently, 2F-85 hits the board (aborts), aborted rollouts swamp the buffer |
| V11-02 (V11d-h..), 2026-10-05 | clean clouds, Sinkhorn-10, stage tol from demo spread (streak 3, 10 s), clearance UR 3 / F 2 mm, min box, buffer hygiene, 26-episode seed | 0/20 (iters 0-3; run continuing to 9) | 0/20 | 0/6 (iters 1, 3), 0 unsafe | stage 1 never reached: at the 10 s switch the Franka is 35-83 mm and the UR 9-42 mm off the demo's pre-place pose; the UR descends ~70 mm per rollout and drags the belt outside; the UR sits on the 20 deg envelope 50-84 % of frames |

What is settled: perception/representation (recon 2.6 mm, final-frame AUC 1.00 engaged vs
over/outside), safety (0 board aborts with clearance, composed envelope, reversible box), the
stage rule on clean demos (12/12 fire within 2 s of the latch), MPPI solve p95 14-17 ms.
What is not: transport to the pre-place configuration under a one-goal latent cost over a
1.2 s horizon, and a buffer that never contains an MPC success.

**V11-02 final (orchestrator, 2026-10-06 00:33):** the run stopped after iteration 7 on the
stop rule "2 consecutive refit rejections" (iterations 8-9 not run; the user's 10-iteration
request kept that rule). MPC successes **1/40**: iteration 5 (V11i) collected 1/5 engaged
(also strict) — the first on-policy success — every other iteration 0/5. Eval 0/6 in
iterations 1, 3, 5, 7 (0 unsafe except 1 in iteration 7). Stage 1 never reached by the
latent rule. Iterations 6 and 7 rejected their refits (one-step gate; iteration 7's refit
degraded to one-step 7.74 mm vs 3.46 for the kept model, recon 6.67 mm), so V11i was kept;
the warm-started joint refit drifts as failure rollouts accumulate — relevant to V12-01's
fresh-vs-warm test. One simulator segfault (iteration 2 collect, exit −11) left orphan
controller/MPPI processes on the run's port; they were killed and the loop resumed (the
harness should kill its children on a sim crash).

| iteration | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| model | V11d | V11e | V11f | V11g | V11h | V11i | V11i (rejected) | V11i (rejected) |
| collected engaged | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | **1/5** | 0/5 | 0/5 |
| eval (6 cells) | - | 0/6 | - | 0/6 | - | 0/6 | - | 0/6 |

**Correction (orchestrator, 2026-10-06, after V12-02):** the seed demo itself drives the UR down
106 → 30 mm and the Franka 91 → 43 mm within ~1 s (to the pre_place_2 latch, frame 33), then
the Franka sweeps +108 mm in y (frames 33 → 60) to wrap the belt while the UR holds. The V11-02
rollouts' UR descent (~70 mm) is therefore demo-like; the failure is the missing Franka sweep,
which leaves the belt dragged "outside". Earlier text calling the UR descent the failure is
superseded by this note.

## 4. Levers and ranking (from the plan)

1. Reverse curriculum (replayed demo prefix, start moves earlier with success).
2. Transport along the demo: latent waypoints / per-knot demo-latent tracking + demo action
   prior on the MPPI mean.
3. Curriculum schedule + start-pool widening.
4. J&P fidelity: fresh refit per iteration, horizon.
5. EE positions in the dynamics (A_p/E_p), no EE cost.
6. More demos (grasp variants).
7-11 (held): multistep loss, ensemble/support penalty, C3+ comparison, MPPI knobs,
observation-space cost.

## 5. Offline screens

| id | package | what | headline numbers | recommendation |
|---|---|---|---|---|
| V12-01 | cmp-offline-model-gates | demo-action rollout error (M1), latent monotonicity (M2), MPPI vs demo on demo frames (M3), fresh vs warm (M4), H sweep (V11d-i + 4 M4 refits on the `v1_1/iter5` buffer, 12 clean demos, H 8/16/24, V11-02 yaml, clearance on; `cmp/offline/model_gates.md`) | M1 H16 whitened error on moving windows: V11d 0.51, V11i 0.34 vs no-motion 1.12 / 0.72 (every model ~2x better than no motion; 7956 solves, viol0 100 % except the M4 deploy-box rows at H24, 58 %, percentile box). M2: from the first moving frame the pp2, seated and +10-waypoint potentials are monotone on 12/12 demos for every model (max uphill <= 0.09), so the single-goal failure is off the demo path. M3 approach cos_median (seated goal): V11d 0.39 -> V11i 0.85 over the warm iterations; descent 0.73-0.87, but MPPI's Franka y in descent is +0.4..+1.2 vs the demo's +5.1 mm/step and it lifts the UR instead of holding it. M4 (same split): warm30 one-step 4.07 / M1 0.100 / cos 0.93; fresh60 4.74 / 0.127 / 0.33; fresh120 4.23 / 0.124 / 0.69 (box-matched cos 0.84 / 0.13 / 0.70). H: approach cos H8 -0.007, H24 +0.016 vs H16 | **`refit.fresh_each_iter = false`** (fresh60 / warm30 M1 ratio 1.28 > 0.8 and cos 0.33 < 0.93; same for fresh120 and every pair); **`H = 16`** (no horizon gains >= 0.15). Flag: the trust-region box cut the planner's Franka +y bound from 6.0 (V11d) to 2.98 (V11i) / 2.71 (box_6) mm/step, below the demo sweep's 5.98 |
| V12-02 | cmp-mppi-transport | transport screen: baseline / wp / track / +prior, M3 metric per phase (V11d, V11-02 yaml, clearance on, 12 clean demos, every 4th frame; `cmp/offline/transport_screen.md`) | approach cos_median: baseline 0.734, wp 0.770, track 0.726, wp+prior 0.921, track+prior 0.907 (best - baseline +0.187; seeds 1/2: +0.131 / +0.147); descent 0.771 -> 0.960 with the prior; u_0 speed vs the demo 0.12 -> 0.40 (approach) with the prior; replay guard: u_0 near the demo action 0 % (all configs); H 8/24 and alpha 0.25/0.75 within 0.03 of track+prior | rule: do NOT route on the +0.2 cosine margin (missed by 0.013-0.07 over 3 seeds); goal mode: **stage** (waypoints; track - wp = -0.014); the demo action prior is the effective lever (orchestrator's call whether V12-07/C3 run with `wp+prior`). Sim smoke 7790 (`cmp/smoke_transport/`, 15 s, pre_place_1): track+prior and wp-stage+prior both PASS (0 bad msgs, u_0 in the box, idx 0 -> 160 / 199, hold capped at 3.0 s, p95 38 / 41 ms at 97 % GPU); the UR still descends 106 -> 36 mm (track) / 42 mm (stage) in 15 s, so the on-demo cosine does not capture the closed-loop UR descent |

## 6. Data

| id | package | what | episodes | successes (human) | notes |
|---|---|---|---|---|---|
| seed V11-02 | P6 | 6 demos + 14 excitation, clean | 20 | 10 | reference |
| seed_ext | P6 | 6 demos set2 gv_00/01/02 | 6 | 6 | 2 near-duplicates (4 distinct) |
| V12-04 | cmp-demos-ext2 | 8 clean UR-held demos (seed 20261007, set2 gv_03/gv_05, set3 gv_02/gv_03, 2 each, scripted `slack_mm=10` between; ids `ext2_demo_0000..0007`; `cmp/demos_ext2/buffer.json`) | 8 | 4 (strict 2; GT engaged 4 / slanted 4) | per variant human: set2 gv_03 0+1, gv_05 0+0, set3 gv_02 1+1, gv_03 1+0 (human-reset + scripted); 152 s robot sim, 4 scripted + 4 human resets; gv_05 gave no success; ep 7 had 12/200 frames outside the action box |

## 7. Experiment log

One row per cell; the per-iteration detail lives in each cell's `learning_curve.md`.

| run | config diff vs V11-02 | seed buffer | iters x n | successes / iter | stage 1 reached | GT outcomes (report) | unsafe / aborts | both-down % | refit (1-step / h7 / rho / AUC) | solve p95 ms | budget (MPC rollouts, robot s) | root | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V11-02 iterations 0-2 (control) | - | 26 | 3 x 5 | 0, 0, 0 | 0/15 | over 3, outside 11, slanted 1 | 3 / 3 (iter 0 grasp) | 19, 12, 19 | 2.7 / 4.2 / 0.98 / 1.00 | 14-15 | 15, 1409 | `v1_1/` | control |
| V12-05 prefix 1.0 | `cmp/configs/C2_prefix.yaml`: prefix frac 1.0 fixed (3 set2 demo starts), `mppi_stages auto` -> last (deploy seated goal), no eval, `box_demo_union`, V12a- | 34 (V11-02 seed 20 + ext 6 + ext2 8) | 3 x 5 (3 prefix skips: UR end err 5.3 / 3.1 / 4.4 mm > 3) | 0/4, 0/4, **1/4** (1/15) | n/a (frac 1.0) | outside 6, slanted 4, over 1, other 1, engaged 1, skipped 3 | 0 real (the loop's `unsafe` 5/iter = prefix rotation counted from frame 0 + skips) / board aborts 0, 0, 1 | 9, 25, 44 | 2.53/3.71/0.989/1.00; 2.59/3.59/1.000/0.75 (acc); 2.60/3.39/1.002/1.00 (acc) | 14-16 | 12, 464 s (cum. 1161 incl. seed) | `cmp/C2_prefix/` | weak seating from pre_place_2 (1/15 < 3): iter 0 sweeps +230 mm past the demo; refits cut the overshoot to +33..+51 mm |
| V12-06 prefix 1.0 + transport + prior | `C3_both.yaml`: V12-05 + waypoints {10, 3, 90, 0.5}, goal mode stage, prior {0.5, 0.1}; at frac 1.0 stages last = seated waypoint only, i.e. effectively V12-05 + the demo prior (demo clock from frame 33), V12d- | 34 (same) | **1 x 5 + crash** (iter 1 export: check E3 FAIL) | 0/4 (1 prefix skip) | n/a | over 2, outside 2, skipped 1 | 0 real / 0 | 0 | 2.49/3.77/0.980/1.00 (seed); iter 1 refit V12e accepted (one-step ok) but its export failed E3 | 14.4 | 4, ~150 s | `cmp/C3_both/` | stopped (crash rule); the prior bounds the sweep (overshoot -2..+38 mm vs V12-05 iter 0 +230) but the belt ends over (h 10-11 mm) / outside |
| V12-07 transport + prior | `C1_transport.yaml`: no prefix (frac 0 fixed), demo starts, stages all = 7 latent waypoints (stage mode, 3 s each), prior {0.5, 0.1}, offset 20, V12g- | 34 (same) | **2 x 5, loop verdict DONE** (>= 4/5 twice; iteration 2 refused; `--ignore-stop` not permitted) | **4/5, 4/5 (8/10; strict 8)** | latch waypoint (k 3) on the latent 1/5, 4/5 (5/10 = 50 %); loop's first-waypoint column 2/5, 4/5 | engaged 8, outside 1, slanted 1 | 0 / 0 | 0, 12 | 2.51/4.10/0.976/1.00; 2.66/3.58/1.002/0.75 (acc) | 15.6-16.2 | 10, 466 s (cum. 1163) | `cmp/C1_transport/` | **transport + prior seats from pre_place_1**: +77 mm in 2.3-2.5 s, overshoot +33..+47 (iter 0) -> -22..+5 mm (iter 1); V12-08 = V12-07 config |
| V12-08 | `full.yaml` = V12-07 config + eval every 2 + ladder [2.0 .. 0.0] from 0.0 (advance >= 3/5, retreat after 2 x 0/5) + pool set2 + set3 (22 starts) at 0.0 / demo starts above 0, V12j-s | 34 (same) | 10 x 5 | 3, 0, 0, 0, 1, 0, 1, 1, 0, 1 (**7/50**; frac 0.0: 3/15, frac 0.5: 4/35) | latent latch 8/15 in iters 0-2 (53 %) | outside 19, over 12, engaged 7, other 7, slanted 5 | 0 / 0 board aborts; eval unsafe 0 | 0, 0, 12, 34, 53, 69, 66, 69, 44, 41 | 9/9 refits accepted; one-step 2.49 -> 3.42 (iter 6) -> 2.99, h7 4.37 -> 3.17 | 14-17 | 50 MPC + 30 eval, robot 4258 s cum. | `cmp/full/` | **G6 FAIL** (>= 2/5 at frac 0.0 only in iter 0; eval engaged 1, 1, 1, **2**, 1 of 6) |

### Tooling for V12-03 to V12-08 (`PKG-20261006-cmp-harness-prefix-curriculum`)

| item | what | status |
|---|---|---|
| `scripts/lcs/demo_prefix.py` | relative replay of a demo's EE poses (both arms) to a frac of the task before MPPI starts; frac 0..1 = motion start..pre_place_2 latch, 1..2 = latch..first hold:place_3 (demo `prehold` skipped); end gate 3 mm / 1 deg, grasp held; frames recorded with `sim_cmd_source` 3, actions = the replay's commanded deltas | sim (7791, `cmp/smoke_prefix/cli`): frac 0 no-op, 0.5 / 1.0 / 1.5 -> end frames 27 / 33 / 43, end error <= 0.82 mm / 0.14 deg, grasps held, 4/4 ok |
| harness `--prefix-demos/--prefix-frac/--prefix-offset-to-node` | prefix between settle and controller start; `index.prefix`, row `prefix`, `budget.prefix_s/robot_s`; prefix failure = skipped episode (next episode human reset); flags at default dropped from `index.args` | identity pre vs tree (mppi 3 s, baseline) PASS (only the run-to-run `sim_alignment` string length differs); `check_mpc_harness.py` H0-H8 + H10 PASS on 7791; smoke below |
| loop V12 keys | `mppi_demo_offset_frames` (20 in the transport cells), `prefix` (ladder 2.0 / 1.5 / 1.0 / 0.5 / 0.0, advance at >= 3/5, retreat after 2 x 0/5, state `iter<i>/curriculum.json`), `collection_pool_rule` (`demo_starts`), `mppi_stages auto`, `goal_mode` / `waypoints` / `demo_prior` (V12-02 flags), `mppi_H`, `refit.fresh_each_iter`, `model_id_prefix`; eval never prefixes | V11-02 dry-run identity 10/10 iterations, non-dry identity on V11-02 data, `check_curriculum.py` 8/8 |
| `cmp/configs/*.yaml`, `cmp/run_cell.sh` | five cell configs (V11-02 loop.yaml + diff); runner refuses while V11-02 may still hold the sim | guard tested on fake status files |

**V12-03 prefix smoke (2026-10-06, 7791, V11d = the V11-02 seed model S10, V11-02 `mppi.yaml`, clearance,
clean belt-seg, set2 gv_00, 20 s; GT report-only; `cmp/smoke_prefix/`)** — first answer to "does
MPPI seat from pre_place_2 with S10?": **yes, 2/2** (the V11-02 loop never reached pre_place_2).

| run | prefix (end frame, s, end err F / UR mm) | reset before | GT outcome (wrap deg, h mm) | Franka y: start -> max (to +77 mm) | UR z start / min / end mm | max rot F / UR deg | solve p95 ms |
|---|---|---|---|---|---|---|---|
| frac 1.0, stages last, ep 0 | 33, 1.28, 0.14 / 0.82 | human (initial) | engaged (118, +0.10) | -31 -> +171 (+77 at 3.4 s) | 29 / 29 / 36 | 19.1 / 19.3 | 38.4 |
| frac 1.0, stages last, ep 1 | 33, 1.28, 0.14 / 0.93 | scripted ok (jitter -4.6 deg kept) | engaged (118, +0.07) | -27 -> +174 (+77 at 2.8 s) | 29 / 29 / 36 | 19.6 / 19.4 | 35.8 |
| frac 0.5, stages all | 27, 0.90, 0.14 / 0.74 | human (initial) | over (0, +26.0), stage 0 timeout | -21 -> +214 (+77 at 11.8 s) | 63 / 35 / 36 | 19.3 / 19.3 | 35.6 |

Demo reference (seed demo 0): Franka y -31 (latch, frame 33) -> +77 mm (frames 53-60), UR z 30 ->
28 mm. From the latch MPPI does the y-sweep (faster than the demo, +77 mm in 2.8-3.4 s) and then
keeps going to +171-174 mm by 20 s (the sweep is not stopped by the seated-goal cost; watch it in
the 42 s cells). Prefix frames: `sim_cmd_source` 3 on exactly the first 17 (12) frames, phase
`prefix:demo`, `n_mpc_frames` excludes them, budget `robot_s` = episodes + settle + prefix +
reset (50.6 s for frac 1.0).

### Pre-flight for V12-05 / V12-06 / V12-07 (2026-10-06 01:05, before the first sim start)

- Sim free: V11-02 status ends `iter 8 rc 1` (STOP after 7), no V11-02 process, no socket on
  7774/7775 or 7780-7799; V12-02/W1-C smokes and V12-04 done (run log).
- **Seed buffer (same for every cell):** V11-02 seed (20) + seed_ext (6) + `cmp/demos_ext2` (8,
  4/8 engaged, meets the >= 4 rule) = 34 episodes, merged per cell (`<cell>/seed_buffer_merged.json`).
- **V12-01 applied to ALL cells:** `refit.fresh_each_iter: false`, `mppi_H: 16` (both = V11-02, written
  explicitly). **Orchestrator box decision applied to ALL cells:** new opt-in loop key
  `box_demo_union {kinds [demo], lo_pct 1, hi_pct 99}`: planner box = (trust region U min box) U
  the merged seed's demo actions [p1, p99] per channel (moving frames), outer-clipped. Off =
  identity (V11-02 `box_6` reproduced exactly from `iter5/buffer.json`; dry-run V11-02 iters 0/1/3
  identical). On V11-02 `iter5` data it widens 7/12 channels: f_y [-1.45, +2.71] -> [-3.95, +5.98] mm,
  f_z -> -4.97, ur_z -> -6.00 (outer), fr_y +-1.8 deg.
- **Goal mode:** `stage` (transport-screen verdict + orchestrator routing `wp+prior`), prior {0.5, 0.1},
  offset 20 without a prefix (a prefix sets it per episode: frac 1.0 -> frame 33).
- **Per-cell config diff vs `v1_1/loop.yaml`** (common: the seed above, ports/roots/`lcs_out`
  per plan §5, `model_id_prefix V12`, `eval_every 0` (never), `prefix.demos` = set2 gv_00/01/02 seed
  demos, `mppi_stages auto`, registry tag, `box_demo_union`, `mppi_H 16`, `fresh_each_iter false`):
  | cell | prefix | pool | stages / goal | transport |
  |---|---|---|---|---|
  | V12-05 | frac 1.0 fixed (`schedule: null`) | demo starts | last = deploy seated goal | - |
  | V12-06 | frac 1.0 fixed | demo starts | last = the waypoint artifact's last (seated) goal | prior {0.5, 0.1} on the demo clock from frame 33 |
  | V12-07 | none (frac 0) | demo starts | all = 7 latent waypoints (3 s each, last 30 s) | prior {0.5, 0.1}, offset 20 |
- Deviation: V12-03's V12-05/C3 templates carried the ladder from 1.0; with it a 0/5, 0/5 cell
  would retreat to frac 1.5 at iteration 2 (a separate prefix-1.5 variant, not run) and a 3/5 would advance to
  0.5, so the screening cells are pinned at the plan's frac 1.0; the ladder is for V12-08.
- Note: at frac 1.0, `mppi_stages auto` -> `last`, so V12-06 uses only the seated waypoint: V12-06 = V12-05 +
  the demo prior (the sweep waypoints 43 / 53 are not used; `stages all` would restart at the
  frame-10 waypoint). V12-07 is the only cell that exercises the waypoints.
- Control finding (`cmp/diagnose_cell.py` on V11-02 iters 0-7, report only): the Franka **does**
  sweep in V11-02 (after the 10 s switch) and **overshoots** the demo's +77 mm by +76 to +212 mm
  in 37/40 rollouts (y max 153-289 mm); the only 3 rollouts with y max <= 104 mm are the only
  non-outside/over outcomes (iter 5 engaged +104, iter 2/4 slanted +87/+99). Overshoot fell as the
  trust region narrowed f_y (iter 0: +143..+212; iters 6-7: +76..+92) — a caution for the demo-range
  widening (f_y ub 2.7 -> 6.0 mm/step).

### Decision tree (plan §3) and the branch taken (V12-05/06/07 → V12-08)

```
V12-05 >= 3/15                              -> V12-08: V12-06 config if V12-06 >= V12-05 else V12-05
V12-05 == 0 and V12-06 >= 2                     -> V12-08 with V12-06
V12-05 == 0 and V12-06 <= 1                     -> MODEL branch (no V12-08; a prefix-1.5 variant may start)
V12-07 >= V12-06 and V12-07 stage-1 reached >= 60 % -> V12-08 = V12-07 config, prefix.start_frac 0.0
```

Numbers: V12-05 **1/15** (1/12 labelled, 3 prefix skips); V12-06 **0/4** (stopped by a crash after
iteration 0); V12-07 **8/10** (2 iterations, loop DONE), latent latch waypoint reached **5/10 = 50 %**.

Literal reading: no line fires. V12-05 = 1 falls between the ">= 3" and "== 0" lines, and the V12-07 line's
successes clause holds (8 >= 0) but its stage-1 clause misses (50 % < 60 %). **Branch taken (operator
judgement): the V12-07 line, "transport alone suffices" -> V12-08 = V12-07 config with `start_frac 0.0`.**
Reason: the 60 % clause is a proxy for "transport solved" (plan §2), and V12-07 does better than the
proxy: 8/10 human-labelled (and strict) seats from pre_place_1 with no prefix, i.e. the full task.
In stage mode the waypoints also advance on their 3 s timeouts, so the latent latch firing is not
needed for transport. The MODEL branch's premise ("seating from pre_place_2 fails with the learned
model") is refuted by V12-07. A prefix-1.5 variant was prepared (`cmp/configs/C2b_prefix15.yaml`) but not
run.

V12-08 config = `C1_transport.yaml` + `eval_every 2` + ladder [2.0, 1.5, 1.0, 0.5, 0.0] from 0.0
(advance at >= 3/5, retreat after 2 x 0/5) + pool rule (demo starts while frac > 0, set2 + set3 (22
starts) at 0.0); stop rules unchanged (incl. DONE >= 4/5 twice). Launched 09:43 on 7786/7787.

### V12-08 (`cmp/full/`, 7786/7787, from 09:43)

One row per iteration as it lands (`cmp/full/learning_curve.md`, `cmp/full/diagnosis/diagnosis.md`). Successes = human label (strict in brackets); overshoot = Franka y max - demo +77 mm per rollout; eval = seat6 cells, full task from pre_place_1, no prefix (report-only GT).

| iter | model | refit | prefix frac (decision) | start | successes (strict) | GT outcomes | overshoot mm | refit one-step / h7 mm | both-down | box f_y mm | eval strict / engaged (n) | board aborts | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | V12j | seed | 0.0 (start) | grasp_variants/set2/gv_06 | **3/5** (3) | engaged, engaged, slanted, engaged, outside | +32..+110 | 2.49 / 4.37 | 0% | [-4.17, 6.0] | - | 0 | continue |
| 1 | V12k | accepted | 0.0 (clamp (lowest)) | grasp_variants/set3/gv_03 | **0/5** (0) | over, over, over, over, over | +48..+57 | 2.51 / 3.68 | 0% | [-4.17, 5.98] | 0 / 1 (6), unsafe 0 | 0 | continue |
| 2 | V12l | accepted | 0.0 (stay (1 x 0/n)) | grasp_variants/set3/gv_02 | **0/5** (0) | other, other, other, other, over | +53..+60 | 2.51 / 3.42 | 12% | [-4.17, 5.98] | - | 0 | continue |
| 3 | V12m | accepted | 0.5 (retreat) | grasp_variants/set2/gv_00 | **0/5** (0) | outside, outside, outside, outside, outside | +119..+129 | 2.99 / 3.62 | 34% | [-4.17, 5.98] | 1 / 1 (6), unsafe 0 | 0 | continue |
| 4 | V12n | accepted | 0.5 (stay (1 x 0/n)) | grasp_variants/set2/gv_02 | **1/5** (1) | outside, slanted, slanted, outside, engaged | +27..+140 | 2.88 / 3.52 | 53% | [-4.17, 5.98] | - | 0 | continue |
| 5 | V12o | accepted | 0.5 (stay) | grasp_variants/set2/gv_01 | **0/5** (0) | outside, outside, outside, outside, over | +54..+171 | 3.25 / 3.61 | 69% | [-4.17, 5.98] | 1 / 1 (6), unsafe 0 | 0 | continue |
| 6 | V12p | accepted | 0.5 (stay (1 x 0/n)) | grasp_variants/set2/gv_02 | **1/5** (1) | outside, slanted, outside, engaged, outside | +62..+194 | 3.42 / 3.68 | 66% | [-4.17, 5.98] | - | 0 | continue |
| 7 | V12q | accepted | 0.5 (stay) | grasp_variants/set2/gv_01 | **1/5** (1) | outside, engaged, outside, outside, outside | +48..+165 | 3.08 / 3.29 | 69% | [-4.17, 5.98] | 2 / 2 (6), unsafe 0 | 0 | continue |
| 8 | V12r | accepted | 0.5 (stay) | grasp_variants/set2/gv_01 | **0/5** (0) | over, over, slanted, over, over | +39..+79 | 3.10 / 3.43 | 44% | [-4.17, 5.98] | - | 0 | continue |
| 9 | V12s | accepted | 0.5 (stay (1 x 0/n)) | grasp_variants/set2/gv_02 | **1/5** (0) | over, other, other, other, engaged | +63..+103 | 2.99 / 3.17 | 41% | [-4.17, 5.98] | 1 / 1 (6), unsafe 0 | 0 | continue |

**V12-08 reading (09:43-12:04).** Iteration 0 (V12j, new start set2 gv_06, frac 0.0) 3/5. The pool
rule then drew two set3 starts (gv_03: 5 x over, h 10-12 mm; gv_02: 4 x other + over) at 0/5 each,
so the ladder retreated to frac 0.5 at iteration 3 and never came back (best 1/5 at 0.5). At 0.5
the prefix and the waypoint stages desynchronise (stage index restarts at waypoint 0, frame 10;
prior clock at frame 27): 6 s of waypoint timeouts, +77 mm at 13-15 s, overshoot +120 mm, on the
same start (set2 gv_00) where V12-07 got 4/5 at 0.0. So the curriculum moved the run to a harder
operating point. Eval (always the full task, no prefix) improved 1 -> 2 of 6 by iteration 7
(fe_set1_gv_01 seats every time; gv_05 added in iter 7; fe_nominal and urtip never: over / outside).
Refits: 9/9 accepted, one-step drifts 2.49 -> 3.42 mm by iteration 6, both-down 0 -> 69 % (the V11-02
warm-refit drift pattern). 0 board aborts, 0 eval unsafe, solve p95 14-17 ms.

### Per-run readings (V12-05, V12-06, V12-07)

(one 5-line block per cell, appended by the operator package)

**V12-05 (prefix 1.0, seated goal only; V12a-c; 01:00-01:19).** 1/15 (iter 2 k3 engaged, strict; GT
wrap 94 deg). Primary diagnostic, Franka y over the MPC segment (`C2_prefix/diagnosis/diagnosis.md`):
+77 mm reached in 2.2-2.7 s in every rollout (demo 1.43 s), i.e. the sweep is there; iter 0 (V12a)
never stops: y max +303..+320 mm (overshoot +226..+243), still +20..+39 mm in the last 10 s, all
outside. The seated-goal distance bottoms at ~2.0 at t ~ 2 s (y ~ +60, GT "over", h 13 mm) and then
rises while MPPI keeps pushing +y (model vs sim disagree past the goal). After two warm refits on
the cell's own rollouts (both accepted, one-step 2.59/2.60, h7 3.59 -> 3.39) the overshoot drops to
+33..+51 mm (y max 110-128) and outcomes move outside -> slanted/over/engaged; both-down 9 -> 44 %.
UR: z 29 -> min 29 (18 in the board abort, iter 2 k1: 11 violating plans, 1 fallback) -> end
36-42 mm, net +7..+13 mm (lifts; demo holds at 28). Box: trust f_y ub 6.0 -> 5.09 -> 4.74, the demo
union keeps 5.98; Franka never at the box (0 %), UR on its envelope 11-73 % of frames in iters 1-2.
Contrast: the V12-03 smoke (V11d, 20 s) seated 2/2 from the same prefix — V12a (34-episode seed) is
a worse seater; the 42 s episodes also leave 20 s more for the drift.

**V12-06 (prefix 1.0 + prior; V12d; 01:19-01:28) — stopped after iteration 0 (crash rule).** 0/4 (1
prefix skip): over (h 11.2, 10.4 mm) x 2, outside x 2. The demo prior bounds the sweep: y max +75..+115
mm (overshoot -2..+38 vs V12-05 iter 0 +226..+243; +77 reached in 2.6-2.9 s or not at all), but the belt
stays above the groove ("over") or slips outside; UR z 29 -> end 36-39 mm (net +6..+10, lifts while
the demo holds 28), UR on its envelope 45-63 % of frames, both-down 0 %. Iteration 1: the warm refit
V12e was accepted, then export's float32 guard failed (`check_latent_encoder` E3: numpy LCS step vs
the exporter's PGD reference 3.13e-4 > 3.4e-5; `export_seeds 1` = no retry) -> loop rc 1. Not a code
bug; the deploy arrays do not depend on the export seed (only the check's reference frames do), so
a resume is `export_seeds: 4` in `C3_both.yaml` + `run_cell.sh C3_both 1 2` (state resumes at export).

**V12-07 (waypoints + prior, no prefix; V12g-h; 01:29-01:46) — loop verdict DONE after iteration 1.**
8/10 human labels, all 8 strict too (GT wrap 91-115 deg, h 0.0-0.8 mm); failures: 1 outside (iter 0
k1: no waypoint reached, y max +253, overshoot +176), 1 slanted (iter 1 k2). Franka y: +77 mm at
2.3-2.5 s after MPC start (the demo's own timing from motion start ~2.4 s); overshoot +33..+47 mm
in iter 0, -22..+5 mm (demo-like) in iter 1 after the warm refit V12h (accepted, one-step 2.66,
h7 4.10 -> 3.58). UR descends 106 -> 32-35 mm min, ends 37-40 (net -66..-69, demo-like), Franka
net dz -39..-42; UR envelope binding 0-15 %, box binding ~1 %, 0 fallbacks, 0 violating plans,
both-down 0 / 12 %. The latent latch waypoint fires in 1/5 then 4/5 rollouts (others advance on the
3 s timeouts). Iteration 2 was refused by the loop's DONE rule (`stop.done_successes 4`,
`done_consecutive 2`); the `--ignore-stop` override was not permitted, so V12-07 has 2 x 5.

### V12-09 Grasp-adaptive demo selection (waypoint bank; 2026-10-06 10:20-11:00)

Motivation: the waypoints + demo action prior come from one demo (seed `episode_0000`, set2
gv_00), so on other grasps the reference belt path and prior may not match (V12-08: set3/gv_03 0/5
"over", set3/gv_02 0/5 "other"). Rule (real-compatible): at the node's first latent, the reference
demo (waypoints, tols, `--demo-actions auto` prior) = the bank entry whose initial latent
(encoded first demo frame) is nearest in whitened distance; k-NN mean optional, off.

| item | what |
|---|---|
| bank | `latent_waypoints.py --bank-out DIR [--goal-demos ...]`: one artifact per seed demo (20 = 6 V11-02 seed + 6 seed_ext + 8 demos_ext2; tols pooled over all 20 as today) + `move0` -> `wp_<j>.npz`, `bank.json` (`z_init`, tag report-only); 20 entries in ~2 s; entry 0 == the single artifact byte for byte except `move0` |
| node | `--waypoint-bank BANK_JSON`, `--bank-knn K` (1); `--goal waypoints:PATH` stays the base until the first latent; a non-zero `--demo-offset-frames` is mapped onto each entry's clock (prefix frac mapping); log: stdout `[mppi] bank: entry j <demo> dist d (next ...)`, first JSONL row `bank` |
| harness | `--mppi-waypoint-bank`, `--mppi-bank-knn`; `index.controller.bank` and per-episode `mppi.bank` only when set; stage labels cover the longest entry |
| loop | `waypoint_bank: {knn, success_only}` (needs `waypoints`): bank rebuilt after every export with the new encoder (`iter<i>/waypoint_bank/`), `--mppi-waypoint-bank` on collect and eval; config `cmp/bank_select/configs/full_bank.yaml` (not run) |
| identity | options off: node bytes/stdout/log rows == pre-change node (deploy-goal and waypoints+prior streams); harness sim run pre vs tree (7795, 3 s, V12-08 flags): index keys/args/controller/rows/npz/sim_meta identical up to out path, pid, clearance sha; loop dry-run V12-08 + V11-02 configs iters 0-3 identical 8/8; `check_curriculum.py` 8/8; `check_waypoint_bank.py` B1-B8 PASS |

**Offline screen** (`cmp/bank_select/screen/screen.md`; first frame of the 20 demos and 25 V12-08
rollouts, iters 0-4; tags = start-state set/variant, report only). Pattern identical for V12j-n:

| key = initial latent | V12j | V12k | V12l | V12m | V12n |
|---|---|---|---|---|---|
| snapshot (human-reset) rollouts, own variant picked | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| scripted-reset rollouts, own variant picked | 4/16 | 7/16 | 7/16 | 5/16 | 4/16 |
| median d to the pick, snapshot / scripted | 0.09 / 1.88 | 0.13 / 1.13 | 0.07 / 1.14 | 0.07 / 1.12 | 0.07 / 1.21 |
| demos leave-one-out, own variant | 10/20 | 7/20 | 7/20 | 7/20 | 6/20 |

V12n rollouts (rows = true, cols = pick): set3/gv_02 5/5 (all picks set3/gv_02); set3/gv_03 1/5
(picks set2/gv_00 1, set2/gv_01 2, set2/gv_05 1, set3/gv_03 1); set2/gv_00 1/5; set2/gv_02 1/5 (4
pick set3/gv_02); set2/gv_06 (no demo) -> set2/gv_03 4, set2/gv_05 1. Reading: a snapshot start
re-creates its demo's start (d 0.04-0.13); after a scripted reset (re-drape + jitter dx/dy +-7 mm,
yaw +-5 deg) the initial latent moves 0.8-2.2, more than the gap between grasp variants, so the
nearest demo is mostly another variant. The initial belt latent encodes the absolute belt
placement more than the grasp.

**Sim check** (7795, V12n = V12-08 iter4 (latest accepted at 10:46), V12-08 iter4 collect yaml +
waypoints, 42 s, snapshot + 1 scripted reset with the V12-08 run's seeds; control = same without
the bank; `cmp/bank_select/sim/summary.md`; GT report-only):

| arm | grasp | start | GT | pick (d) | Franka y max (demo ~78) mm | solve p50 / p95 ms |
|---|---|---|---|---|---|---|
| bank | set3/gv_03 | snapshot | outside | set3/gv_03 (0.06) | 203 | 25.6 / 34.0 |
| bank | set3/gv_03 | scripted | outside | set2/gv_00 (1.27) | 203 | 14.2 / 43.0 |
| bank | set3/gv_02 | snapshot | outside | set3/gv_02 (0.04) | 173 | 20.4 / 46.5 |
| bank | set3/gv_02 | scripted | outside | set3/gv_02 (0.85) | 180 | 22.0 / 48.0 |
| ctrl | set3/gv_03 | snapshot / scripted | outside / outside | - | 222 / 222 | 20.8-21.5 / 43.0-43.3 |
| ctrl | set3/gv_02 | snapshot / scripted | outside / slanted | - | 205 / 112 | 17.3-20.1 / 39.1-42.5 |

Bank 0/4, control 0/4 (V12n's own V12-08 iteration: 1/5). Both arms fail the same way: the Franka
+y sweep overshoots the demo's +78 mm by +95..+145 mm (the V12-05 iter-0 failure mode), 0 fallbacks,
0 violating plans, no board contact. Verdict: with V12n the reference demo is not the bottleneck
on these grasps; the drifting model's unbounded sweep is. Solve p95 34-48 ms in both arms (vs
13-16 ms in V12-08 iters 0-3) = GPU sharing with the V12-08 run; the selection itself is 20
distances once per episode. **Side effect: V12-08 iter5's collect (10:48-10:56) overlapped this
check and ran at p95 42.6-47.7 ms** (iter4's last two episodes, p95 19.6 / 23.6 ms, overlapped an
earlier CPU-heavy check of mine); treat V12-08 iter5 (0/5) as confounded by latency.

### V12-10 overshoot guard and V12-11 behavioural refit gate (2026-10-06 11:00-12:55)

Motivation: V12-08 degrades by overshooting the Franka +y sweep (+30..+140 mm on every grasp;
nearest-demo check y max 173-203 mm vs the demo's +78). Both levers are opt-in (off = identity);
root `data/lcs/realfirst/cmp/overshoot/`.

| item | what |
|---|---|
| guard (`MppiParams.overshoot`, `mppi_lcs.OvershootGuard`, node hook) | d = whitened latent distance to the seated (last) goal, `seat_tol` (loop: goal.json `tol_2`). (1) `fade [k1, k2]`: demo-prior actions x clip((d - k1 tol) / ((k2 - k1) tol), 0, 1); `fade_last_stage`: x 0 in the last stage (a zero-action prior, not off). (2) `hold`: enters after `streak` latents with d < `enter_k` tol, or (`stall {latents, eps, k}`) in the last stage once d has not dropped `eps` below its anchor for `latents` solves with min d < k tol; then prior off, goal `seat` / `entry` (latent at entry) / `best` (min-d latent), weights x `w_q/w_T/w_r/w_s`, first hold solve cold (`cold`), latches unless `exit_k`. (3) `near {k, w_T}`: w_T x scale while d < k tol. No EE terms. Node JSONL `d_seat`, `prior_scale`, `mode`; stdout `[mppi] overshoot hold: ...`; hold/near solvers captured at warm-up |
| loop | `mppi_overshoot` (yaml block, `seat_tol: tol_2` substituted per iteration; needs `stage_tol`); `behaviour_gate` (below); template `overshoot/configs/full_gate.yaml` (not run) |
| behavioural gate (`scripts/lcs/behaviour_gate.py`, loop `behaviour_gate`) | after a refit passes the prediction gates: candidate exported to a scratch deploy (box_{i-1}, `lcs_out/iter<i>/behaviour_gate/`) + its waypoints; MPPI (iteration i-1's collect yaml, noise 0, cold, clearance) on every 4th frame of the 20 seed demos (active waypoint + demo prior) and every 4th solve of B_i's last 5 MPC successes (their logged goal index / prior clock), candidate vs deploy_{i-1}, same seed; `model_gates.screen_rows` cosine per phase + Franka u_0 y on demo hold frames ("hold push"); reject iff approach/descent cos_median drops > `max_drop` (0.1), or hold push > `max_hold_push_mm`, or > ref + `max_hold_push_increase_mm`; rejected = theta_{i-1} kept (counts toward the 2-rejection stop rule); `iter<i>/behaviour_gate/behaviour_gate.json`, refit.json / model.json `behaviour_gate`, learning-curve section |
| checks | `scripts/checks/check_overshoot.py --pre-dir` O1-O5 PASS: 87 realfirst yamls to_dict == pre + `overshoot: None` (the 2 guard-on yamls load), 9/9 bad blocks refused; pre vs tree solver plans + stats identical (3 configs, cold + warm); pre vs tree node on a recorded V12-07 stream (waypoints + prior 56 solves, deploy goal 60): published bytes, stdout, log rows identical; guard logic; node with the guard (prior x fade per solve, hold: no prior / goal / w_r x10 / first solve cold). Loop `--dry-run` V12-08 + V11-02 configs iters 0/1/2/3/5/9: 12/12 identical; `check_curriculum.py` 8/8; ruff 0.16.10 clean |

**Offline replay** (`scripts/lcs/overshoot_offline.py`, `overshoot/offline/replay.md`): recorded
V12-07/FULL rollouts' latents + EE poses fed to the node (open loop, noise 0); u_0 Franka y (mm/step)
over the 3 s after the recorded seated-goal distance bottoms. Off reproduces the recording (e.g.
+1.39 vs +1.37). `stall_entry` = enter_k 1.0 + stall {13, 0.05, 2.0}, goal entry, w_T x2, w_r x10,
cold; `hold_entry` = enter_k 1.2, no stall.

| episode (GT) | F y at d min / max mm | off | stall_entry (hold at y mm) | hold_entry (hold at y mm) |
|---|---|---|---|---|
| V12-07 i0 00_0 (engaged) | 61 / 112 | +1.25 | +0.94 (57) | +0.80 (52) |
| V12-07 i0 00_1 (outside) | 77 / 253 | +1.39 | +1.43 (86) | +1.39 (no hold) |
| V12-07 i1 02_0 / 02_2 / 02_3 (V12h) | 37-64 / 60-82 | +0.39 / +0.45 / +0.07 | +0.20 / +0.24 / -0.35 (38-48) | +0.13 / +0.09 / -0.36 (37-42) |
| V12-08 i0 06_1 (engaged) / 06_4 (outside) | 100-102 / 121-187 | +0.97 / +0.65 | +0.41 / +0.75 (89-114) | +0.23 / -0.17 (75-80) |
| V12-08 i1 03_0 (over) | 103 / 130 | +0.27 | -0.37 (89) | +0.27 (no hold) |
| V12-08 i3 00_0 / i4 02_0 / i5 01_0 (outside) | 90-115 / 203-248 | +0.86 / +0.77 / +1.33 | +0.25 / -1.14 / -0.36 (86-120) | -1.22 / -1.83 / +1.33 (59-75; i5 no hold) |
| V12-08 i4 02_4 (engaged) | 92 / 104 | +0.92 | -1.48 (76) | -1.91 (56) |

Reading: the plan stops pushing +y (or pulls back) once hold is in force in 10/12 episodes
(stall_entry); it fails where hold enters late (V12-07 i0 00_1) or the model maps +y toward the hold
latent. `fade` alone = off exactly (the demo prior is already zero after demo frame 53, and d
bottoms at 7-36 s); `near` (w_T x3) no effect (+1.01 / +0.54 / +0.60 vs +0.97 / +0.65 / +0.77);
goal `seat` keeps pushing (the model believes +y approaches the seated goal); `best` < `entry`;
w_r x30-x300 not monotone (open-loop states diverge after hold).

**Sim checks** (7796, 42 s, snapshot + scripted reset with fixed seeds, off/on interleaved, same
model / starts; guard on = enter_k 0.8, streak 3, stall {13, 0.05, 2.0}, goal entry, w_T x2,
w_r x10, cold, fade [1, 2] + last stage; GT report-only; `overshoot/sim*/summary.md`):

| model (why) | grasps x 2 | arm | GT engaged | outcomes | overshoot (y max - 77) mm | hold at (t s, F y mm) | solve p95 ms |
|---|---|---|---|---|---|---|---|
| V12h = V12-07 iter1 (best: 4/5, tie with V12g; less overshoot) | set2 gv_00 / gv_06, set3 gv_02 / gv_03 | off | **1/8** | slanted 4, over 3, engaged 1 | median -16, max +19 | - | 15.1-20.0 |
| | | on | 0/8 | slanted 4, over 4 | median -13, max +19 | 7.5-18.1 s, 18-57 mm (8/8) | 13.8-14.9 |
| V12n = V12-08 iter4 (the overshooting model of the bank check) | set3 gv_02 / gv_03 | off | 0/4 | outside 3, over 1 | **+95..+152** (y max 172-229) | - | 14.1-14.7 |
| | | on | 0/4 | over 4 (h 13-15 mm) | **+42..+55** (y max 119-132) | 7.0-15.1 s, 72-90 mm (4/4) | 13.8-14.5 |

Reading: (1) on the overshooting model the guard does what it is for in closed loop: overshoot
-53..-101 mm per start, belt kept on the pulley ("over" instead of dragged "outside"), but no seat. (2) The
best model V12h does not overshoot on these grasps (it under-sweeps, y max 41-96 and often pulls
back), so there is nothing to fix; the guard cost its one success (set3/gv_03 scripted: hold at
8.2 s, y 42, slanted) because V12h's waypoint stages complete in ~5-8 s, the last stage starts
early and the stall trigger (k 2 x tol_2 2.96 = 5.9) fires before the sweep ends. (3) Hold on a
latent goal is not an EE hold: the Franka still moves in hold (e.g. set3/gv_02: -44 mm after
entry). Solve p95 ~14 ms in all arms (V12-08 had finished at 12:04, GPU unshared). Next: a stall
`k` near 1 x tol and a minimum time / stage index before stall, judged on both models.

**V12-11 behavioural gate sweep** (`overshoot/gate_sweep/`, every accepted V12-07/FULL refit vs its
predecessor, as run; with and without the demo prior; 10-13 s per pair):

| pair | hold push cand / ref mm/step | cand's collect | appr / desc cos cand / ref | succ cos cand / ref | rule appr+desc+succ | appr+desc | + hold push +0.5 |
|---|---|---|---|---|---|---|---|
| V12-07 V12h vs V12g | 0.53 / 0.54 | 4/5 | 0.95 / 0.95, 0.94 / 0.95 | **-0.31 / 0.93** | reject | accept | accept |
| V12-08 V12k vs V12j | **2.02 / 0.74** | 0/5 | 0.96 / 0.95, 0.96 / 0.96 | 0.86 / 0.87 | accept | accept | **reject** |
| V12-08 V12l..V12s vs predecessor | 1.67-2.49 / 1.67-2.49 | 0-1/5 | 0.90-0.95 | 0.66-0.90 / 0.75-0.92 | reject at i2, i4 | accept | reject at i5 (2.26 / 1.67) |

Reading: the demo-frame cosines are saturated (0.90-0.96, with or without the prior) and do not
see the degradation; the success-frame cosine favours the model that generated those successes
(it would have rejected V12h, the campaign's best refit), so `success` is report-only by default;
the hold-frame +y push (u_0 Franka y where the demo is still) is the discriminating signal: 0.5
for V12-07's seating models, 0.74 -> 2.02 at V12-08's V12j -> V12k (3/5 -> 0/5), 1.7-2.5 after.
Recommended gate for a next run: `phases [approach, descent]`, `max_drop 0.1`,
`max_hold_push_increase_mm 0.5` (or `max_hold_push_mm 1.0`); counterfactually (if later
candidates push like the actual ones, >= 1.67) it keeps V12j from V12-08 iteration 1 and the
2-rejection stop rule ends the run at iteration 2. Loop test (`overshoot/gate_test/`,
V12-08 iter4 -> iter5 data, V12o checkpoint): accept path 24 s incl. the scratch export; reject path
(`max_hold_push_mm 0.5`) -> refit.json rejected, model.json keeps V12n with
`rejected.behaviour_gate`, learning-curve section rendered, verdict continue.

Side effects: my paced offline replays overlapped V12-08 iter7 / iter8 collects (solve p95 20.1 /
22.0 ms vs 15.7 in iter6); V12-08 iter9's loop process predates the loop install (11:49) and ran
the old loop; its nodes ran the new node at default (identity O2/O3). Nothing under `cmp/full/`
written.

### V12-12 / V12-13: two loops from 12:43 (`cmp/full2_gate/` 7788/7789, `cmp/full2_lcsonly/` 7790/7791)

Why: V12-08 showed that warm joint refits drift the policy (iter0 V12j 3/5 → iter1 V12k 0/5, "over"
×5; both-down share 0 → 69 %) while every prediction gate accepted. The overshoot guard did not
seat anything, so it stays off. Both cells = `full.yaml` with the prefix ladder removed (frac 0.0,
pool set2 + set3 from iteration 0), `max_consecutive_rejections` 4, and the behaviour refit gate
(`max_hold_push_increase_mm 0.5`, phases approach + descent). They differ only in the refit:

| cell | refit | question |
|---|---|---|
| V12-12 (V12t…) | warm joint (encoder + LCS) | does rejecting refits that push harder at the hold stop the drift? |
| V12-13 (V13a…) | encoder frozen after the seed, LCS-only refits | does a fixed latent (fixed goals, waypoints, tolerances) let the LCS improve from on-policy data the way Jin & Posa's loop does? |

Code: `realfirst_loop.refit_cmd` now always runs the seed (init none) as `joint`, since
`lcs_only` needs a checkpoint. Running alongside: the stage-synced demo prior (opt-in fix of the
prior/stage desync), offline and on 7796.

Start-up fix (12:49): `full2_gate`'s seed V12t failed export check E3 on all 4 export seeds
(100-iteration PGD vs the 25-iteration training solve: 1.7-1.8e-2 > 1e-2), the same failure that
stopped V12-06. The 25-iteration check (what MPPI rolls out) matches to 2e-6, so E3's loose-vs-exact
bound only matters for the C++ C3 controller. New `check_latent_encoder.py --exact-tol` and loop
key `check_exact_tol` (0.05 in both configs); export seeds 4 for V12-12.

GPU contention (13:02): both seeds went 0/5 on set2/gv_06 (V12t: other, under ×4; V13a:
outside ×3, over, other) with solve p95 65-68 ms. During those collects the GPU also ran the other
cell's seed training and the prior-sync sim check; V12-08's seed had p95 15.6 ms and went 3/5. The
iteration-0 rows are therefore marked contended. `full2_lcsonly` was stopped (by pid) during its
iteration-1 collect, to resume after `full2_gate`. From now on, one loop at a time on the GPU.

### V12-14 stage-synced demo prior (2026-10-06 12:45-13:00, `cmp/prior_sync/`)

Bug: the demo action prior is indexed by wall clock (frames since the first latent + offset),
independent of stage progress. V12-08 iter0 ep4: stage 1 (goal = demo frame 20, pre-sweep) times
out after 3 s while the prior index runs 23 → 61, i.e. through the whole +y sweep (demo frames
33-52, +6 mm/frame); the Franka goes y 10 → 89 mm, the pre-sweep goals of stages 2-3 pull it
back to 11, and the sweep is redone in stage 5 with the prior already at zero actions (idx > 53).

Fix (opt-in, `MppiParams.prior_sync`, default None = unchanged): `prior_sync: stage` (needs
`--goal waypoints:` + `--goal-mode stage`): in stage k the index starts at max(frame of waypoint
k-1, demo offset) (0 before stage 0), advances one frame per dt from the stage start, clamps at
waypoint k's frame, and jumps to the next stage's start on advance (`prior_sync_index` in the
node; frames = the artifact's existing `frames`; a bank / knn entry uses its own). One stage
(`--stages last`) = the wall clock. Loop: `demo_prior: {alpha: 0.5, weight: 0.1, sync: stage}`
→ yaml `prior_sync: stage` (`check_cfg` needs goal_mode stage). `eval_learned_mpc.py` passes the
yaml through, no change.

Checks: `scripts/checks/check_prior_sync.py` P1-P5 PASS (90 realfirst yamls to_dict == pre +
`prior_sync: None`, bad values refused, solver shared; pre vs tree node with sync off on 3
recorded streams, 114-116 solves each: plans, log rows, stdout identical; synthetic sequences +
200 random ones: clamp, start floor, monotone, jump; node with sync on: logged idx and passed
`u_demo` == the rule on 254 solves; loop yaml bytes identical without `sync`, + one key with it).
`check_overshoot.py` 5/5 (O1 allows the new defaulted key). Loop `--dry-run` V12-07 iter 0/1
(V12-07 root and an empty root) identical pre vs tree.

**Offline** (`prior_sync/offline_replay.py`, `offline/replay.md`; V12-08 iter0 ep4, recorded
latents open loop, noise 0, identical stage times in both arms), u_0 Franka y summed per stage:

| stage (goal frame) | t s | idx off / sync | u_0 y sum off / sync mm |
|---|---|---|---|
| 1 (20) | 0.2-3.1 | 23-61 / 20 | +88 / +18 |
| 2 (30) | 3.2-4.4 | 63-79 / 21-30 | -35 / -37 |
| 3 (33, latch) | 4.6-7.4 | 81-119 / 31-33 | -48 / +47 |
| 4 (43) | 7.6-10.4 | 121-158 / 34-43 | +20 / +114 |
| 5 (53) | 10.5-12.8 | 160-190 / 44-53 | +66 / +77 |

The sweep prior now lands in the sweep stages (4-5: +86 → +191 mm commanded) instead of stage 1
(+88 → +18). Caveat: the clamp holds idx at waypoint k's frame, whose action is the first one
of the next segment (stage 3 clamps at the latch frame 33 = the sweep's first +6 mm), and the
H = 16 window always reaches past the frame; a stricter variant would clamp at frame - 1 or
zero the window past it (not implemented).

**Sim** (`prior_sync/sim/`, 7796, V12h = V12-07 iter1 deploy, set2/gv_06 (the desync start) x 2,
scripted reset seed 20261006, noise 0, `summary.md`; GT report-only; solve p95 63-72 ms in both
arms because V12-12's and V12-13's sims shared the GPU):

| arm | outcomes | F y max mm | pull-back before the last stage mm | stage-1 u_0 y sum mm |
|---|---|---|---|---|
| off | over, slanted | 91, 59 | 108, 72 | +85, +46 |
| sync | over, **engaged** | 87, 73 | 79, 31 | +48, +32 |

Reading: less premature sweep and less pull-back with sync, one seat vs none, but n = 2 (not
evidence of a success-rate change). Not enabled in any running cell.

### V12-15 failure census (2026-10-06 13:30, `cmp/census/`, 165 rollouts, 25 engaged)

CPU-only pass over every recorded rollout of the campaign (`census.py`, tables `census_tables.md`).

| mode | n | median overshoot | what happens |
|---|---|---|---|
| M1 sweep runaway → outside | 56 | +130 mm | last stage: the Franka keeps sweeping +y to the episode end (F y max at ~38 s) |
| M2 belt above / off groove after an over-sweep | 35 | +58 mm | |
| M3 belt stays above the groove | 25 | +19 mm | UR does not descend (min z 33-34 mm vs demo end 28 mm) |
| M4 partial seat (slanted) | 20 | +19 mm | |
| M5 lost before seating | 4 | +37 mm | |

- M1 also hits demo-like starts: `pre_place_1` (initial latent distance 0.13) is 0/6, overshoot +120 mm.
  The failures are mostly not a grasp-coverage problem.
- Models alternate between two habits: runaway (V12k-s, V12u: M1/M2) and under-sweep (V12h: M3/M4).
- Grasps: set1/gv_01 5/6, set2/gv_02 7/24, set3/gv_02 0/17 (its own demos are engaged),
  set2/gv_06 4/23 (no demo of its own; nearest demo belt RMS 10 mm).
- Snapshot vs scripted-reset starts: 9/64 vs 11/50, no difference.
- Since the per-step action box cannot stop cumulative drift, the next lever is a data-support position
  box in MPPI (Jin & Posa's trust region in position space, built from demos and successes; a
  bound, not a target). It is in development, offline only.

Reboot (~13:34): the machine rebooted during `full2_gate` iteration 2's collect (3 of 5
episodes). The loop moves a partial collect aside and redoes it; resumed at 13:56.

### V12-16 data-support position box (2026-10-06 14:00-14:30, `cmp/support_box/`, offline, CPU)

`MppiParams.support_box` = `{lo [6], hi [6], weight, hard_margin_m}` on P = [p_franka, p_ur] (the
rollout's P: `sim_ee_*[:3]`, world, m). It adds `weight` × Σ over knots 1..H and the 6 coordinates
of (hinge in mm)², where hinge = max(lo − P, 0) + max(P − hi, 0). With `hard_margin_m`, knots
outside by more than the margin also count as violating. This is a bound, not a target. Costs are
torch-only for both rollout paths (the Warp kernel returns Z/P/R), and `[lo, hi]` is a static
solver tensor, so CUDA graphs are unaffected. The node logs JSONL `sbox_mm` = [mm outside now,
max over the plan's knots] only when the box is on. Loop `support_box`: per iteration, the
per-coordinate [p_lo, p_hi] over every frame of B_i's demos and label_human successes (any kind,
so successful excitation rollouts count), widened by `margin_mm`. Written to
`iter<i>/support_box.json` and added to the collect and eval yaml. Off by default.

**Weight 0.5** (`realfirst_loop.SBOX_DEFAULTS`). Last-stage numbers with the box off, from 6
replayed episodes (`offline/replay.md`):
- The adaptive MPPI temperature (ESS 5 %) is 11-16. That is the cost resolution.
- J is 20-1040, almost all track + terminal; effort, smoothness and prior are each ≤ 5.
- The pull toward +y is weak: dJ/d(u_0 F y) is −0.34 / −0.08 per mm (p10 / p50, M1 last stages;
  min −0.75).
- So at w = 0.5, one knot 5 mm out costs 12.5 (about one temperature). A plan with all 16 knots
  5 mm out costs 200 (about 15 temperatures, weight e⁻¹⁵). The box gradient at a 5 mm hinge is
  5 per mm per knot, about 15× the p10 pull. One knot 2 mm out costs 2 (0.15 temperature).
- The result: soft near the face, dominant past about 5 mm. Closed-loop results barely change
  between w = 0.1 and 2 (below).

**Box from `full2_gate/iter1/buffer.json`** (= B_2, margin 15 mm, 20 demos + 21 successes, 5816
frames), in mm:

| | F x | F y | F z | UR x | UR y | UR z |
|---|---|---|---|---|---|---|
| lo | 323.7 | −53.1 | 25.7 | 492.1 | −76.4 | 13.0 |
| hi | 398.3 | 147.0 | 105.7 | 541.2 | 4.4 | 121.3 |

- One MPC success, `mpc_i1_1` (V12u set3/gv_03, engaged, F y up to 132), sets F y hi, F x lo and
  both UR y faces.
- With demos only, F y hi is 93.5 (demo max 78.5 + 15). B_1's box (iter 1) has F y hi 93.5.

**Offline** (`offline_replay.py`; clearance off in every arm, since its torch path takes minutes
per solve on the CPU). Six episodes, each with the box its own loop iteration would have used
(B_i):
- 4 M1: V12-12 i1 set3/gv_03 #0, V12-12 i1 eval pre_place_1, full i5 set2/gv_01 #0, V12-07 i0
  set2/gv_00 #1.
- 2 engaged controls: V12-07 i0 gv_00 #0 (V12g), V12-07 i1 gv_02 #0 (V12h).

*Open loop* (recorded latents) is a poor test here. In every M1 last stage the recorded state is
already outside the box (F y and F x, UR y). With the box on, every solve then commands the
largest −y step the action box allows (u_0 y sum −950 to −1490 mm vs +215 to +350 off). The
predicted terminal F y falls from 184-202 to 132-153 (p50) for any w from 0.1 to 3.

*Closed loop on the learned LCS*: recorded latents until the final stage, then
z' = LCS(z, u_0, P) and P' = P + u_0:

| episode | box F y hi | recorded F y end | off: F y end (u_0 y sum) | w 0.5: F y end (u_0 y sum), max hinge |
|---|---|---|---|---|
| full i5 gv_01 (M1) | 136 | 248 | 364 (+272) | 132 (+39), 0.0 |
| V12-07 i0 gv_00 #1 (M1) | 94 | 253 | 473 (+430) | 84 (+39), 0.0 |
| V12-12 i1 gv_03 (M1) | 94 | 188 | 440 (+320) | inside by step 20, min 72; leaves at step 152 as the latent diverges (d_seat 4.8 → 12 → non-finite at 313); ends at 264 |
| V12-12 i1 pre_place_1 (M1) | 94 | 196 | 423 (+291) | inside by step 20, min 67; leaves at step 145 (non-finite at 299); ends at 272 |
| V12-07 i0 gv_00 #0 (engaged) | 94 | 104 | 657 (+603) | 86 (+30), 0.0 |
| V12-07 i1 gv_02 #0 (engaged) | 139 | 58 | 22 (−10), another coordinate 170 mm out | 121 (+88), 0.0 |

- When the final stage starts inside the box, the sweep stops at the face for every w from 0.1
  to 2.
- On the model, with the box off, every episode runs away, including the two that engaged in the
  sim. The runaway is a property of the model; the plant stopped it.
- V12u's latent diverges in every arm (box on or off). In those runs the box only delays the
  runaway, by about 150 steps (11 s).

*Controls while the box is not binding* (w 0.5, open loop, solves whose box-off plan stays
inside):
- u_0 translation changes by a mean of 0.17 mm (V12g) and 0.02 mm (V12h); p90 0.82 / 0.07 mm.
- Before the last stage the change is 0.000 / 0.003 mm.
- The box does bind on both controls:
  - V12g swept to F y 106-112, past B_0's demo-only face at 94.
  - V12h's gv_02 grasp puts the Franka x 6 mm below B_1's lo in 65 % of solves, so u_0 changes by
    1.3 mm (mean max-channel).

**Risks.**
1. Pooled min/max lets a single over-swept success widen the box (B_2's F y hi is 147, which is
   +70 mm past the demos). The census has 4/98 engaged above +50 mm overshoot. `kinds: [demo]`
   gives 93.5.
2. Grasps unlike the data sit outside on x by a few mm. The box then moves them, which is a
   grasp-dependent bias.
3. The offline evidence is a proxy. Only a sim cell can show the effect on M1.

Checks (CPU, `CUDA_VISIBLE_DEVICES=""`):
- `check_support_box.py` B1-B5: 5/5.
- `check_overshoot.py` and `check_prior_sync.py` against the pre-change files: 5/5 each. In CPU
  mode both drop clearance. Their P1/O1 treat later defaulted keys and tree-only yamls (sync on)
  like O1 already did.
- Loop `--dry-run` for C1_transport and V12-12, iters 0/1: identical pre vs tree.

Enable with a loop config overlay
(`support_box/configs/full2_gate_sbox.yaml`; dry-run in `dry_full2_gate_iter2.txt`):

```yaml
support_box: {kinds: [demo, success], lo_pct: 0, hi_pct: 100, margin_mm: 15, weight: 0.5,
              hard_margin_mm: null}
```

Used from V12-17 on: `support_box: {kinds: [demo], margin_mm: 20, weight: 0.5}` (demos only, so
one over-swept success cannot widen the box; 20 mm covers the grasp offsets).

### V12-12 readout (iterations 0-4) and V12-17 support-box sim check (14:45)

| iter | model | refit | engaged | note |
|---|---|---|---|---|
| 0 | V12t seed | - | 0/5 | contended GPU (p95 65 ms) |
| 1 | V12u | accepted (hold push 0.95 → 1.23) | 1/5 | M1 runaway in 4/5 |
| 2 | V12u kept | rejected: refit diverged (train recon 13 mm, held-out 7.4 mm > 6) | 0/5 | all outside |
| 3 | V12u kept | rejected: same (12 mm / 7.2 mm) | 0/5 | all outside, eval 0/6 |
| 4 | V12u kept | rejected by the behaviour gate (hold push +1.73 > 1.11 + 0.5) | 0/5 | first gate rejection; all outside |

- Two failure sources in warm joint refits: occasional optimisation blow-ups (recon doubles), and
  policy drift toward a stronger +y push (caught by the gate in iteration 4).
- The cell was stopped after iteration 4: with V12u kept it only collected more runaways.
- Support-box offline finding (the developer's closed-loop replays on the learned LCS): with the box
  off, even engaged episodes' last stages run away on the model, so the +y runaway is in the
  model, and the sim plant (contact) had stopped it in the engaged runs.
- V12-17 sim check `cmp/sbox_sim/` (7796): V12h on the overshoot study's 4 grasps and reset seeds (its
  off arm = control, 1/8 engaged). Arms: `box` (support box from V12-07 B_1's demos only, margin
  20 mm, weight 0.5; Franka y face 98.5 mm) and `boxsync` (box + `prior_sync: stage`).
  Demos-only, because one over-swept success widened the min/max box by 70 mm in Franka y.

**V12-17 result (14:39-14:58, solve p95 14-18 ms, GPU unshared; `sbox_sim/summary.md`):**

| arm | engaged | outcomes | Franka y max median | end-effectors outside the demo box |
|---|---|---|---|---|
| no box (V12-10's off arm) | 1/8 | slanted 4, over 3, engaged 1 | 61 mm | **every episode**: Franka x below the demo range by 25-86 mm for 60-78 % of frames; on set2/gv_06 and set3/gv_02 #1 also UR x (45-52 mm) and UR y (8-16 mm) |
| box | **5/8** | engaged 5, over 3 | 59 mm | inside in 6/8; UR y up to 2 mm out in the other 2 |
| box + stage-synced prior | 4/8 | engaged 4, over 4 | 67 mm | inside or within 8 mm |

- Per grasp (no box / box / box + sync): set2/gv_00 0/2, 2/2, 1/2; set2/gv_06 0/2, 0/2, 0/2 (all
  "over"); set3/gv_02 0/2, 1/2, 1/2; set3/gv_03 1/2, 2/2, 2/2.
- V12h does not over-sweep here (Franka y max about 60 mm against the box's 98.5 mm face), so the
  gain is not from stopping the +y runaway. It comes from a sideways drift: without the box the
  Franka wanders 25-86 mm in −x (and the UR in x / y) out of the demonstrated region. Directions in
  which the learned model sees no effect on the latent cost nothing, so MPPI's sampling noise and
  any model bias move the arms freely there. The data-support box is a generic remedy for both this
  drift and the +y runaway: the planner stays where the model has data (the Jin & Posa trust
  region).
- The stage-synced prior adds nothing on top of the box here (4/8 vs 5/8, n = 8); it stays off.
- n = 8 per arm (5/8 vs 1/8, Fisher one-sided p ≈ 0.06): strong enough to adopt the box in the
  next loop, not a final number.

### V12-18 varied-grasp loop with the position box (from 15:00, `cmp/v12_18/`, 7792/7793)

V12-12's config + `support_box {kinds [demo], margin 20 mm, w 0.5}`; per-iteration detail in
`cmp/v12_18/learning_curve.md`.

| iter | model | refit | start grasp | engaged | outcomes | outside the box (now, per episode) | solve p95 |
|---|---|---|---|---|---|---|---|
| 0 | V12v seed | - | set2/gv_06 | **2/5** | over 3, engaged 2 | ≤ 2 mm, 0-32 % of solves | 15.0 ms |
| 1 | V12w | accepted (hold push 0.15 vs 0.28) | set3/gv_03 | 0/5 | over 5; eval: over on every cell | | |
| 2 | V12x | accepted | set3/gv_02 | 0/5 | over 4, slanted 1 | | |
| 3 | V12x kept | rejected by the behavioural gate (hold push +1.10 > −0.02 + 0.5) | set2/gv_01 | 0/5 | over 2, slanted 3; eval 0/6 (all over) | | |

**Diagnosis of "over" with the box** (offline, CPU; `cmp/over_diag/over_diag.md`; V12-17 box arms,
V12-18 iterations 0-1, 20 demos): **the Franka's final +y sweep starts before the UR has pressed
the belt into the groove.** Belt height when the final sweep starts (F y last crosses 45 mm):
engaged ≤ 2.4 mm (11/11), demos −6..2.4, over ≥ 5.5 mm (21/21). The belt then lies across the
flange; the sweep's tension drags the UR ~25 mm +y (uncommanded) and the run stalls in the
seated stage at h 8-17 mm.

| group | n | UR z when F y reaches +45 mm | F y max while UR z > 50 mm | UR z end |
|---|---|---|---|---|
| demos | 20 | 28 | 2 | 28 |
| engaged (V12h, V12v) | 11 | 34-38 | 2-24 | 34-35 |
| over V12h / V12v | 10 | 85 | 80-93 | 40-44 |
| over V12w | 11 | 39 | 8 | 38 |

- Route 1 (V12h, V12v): waypoints 0-1 are demo frames 10 and 20, before the demo moves. Grasps
  unlike the goal demo start outside stage 0's tolerance; two 3 s timeouts pass while the Franka
  already sweeps with the UR at z 60-90 mm (over: stage 0 missed 10/10; engaged 1/11).
- Route 2 (V12w): every stage "reached" early, the UR stops ~35 mm and the sweep starts ~2 s
  earlier than V12v's engaged runs. The behavioural gate accepted V12w because it only checks the
  hold frames.
- Not latent ambiguity (seated-goal distance separates GT seated from over: AUC 0.96-1.0), not the
  box (no face limits the UR descent), not "sweep stops short" (over runs reach F y 80-93 mm).
- Not setpoint anchoring either: the demos do not press by leading the arm (UR target lead
  0.07 mm median, ≤ 0.2 mm p95 while the UR is below 40 mm), so the demos' 28 mm comes from
  descending before the sweep.
- Next: V12-19 (opt-in waypoint schedule: drop pre-motion waypoints, skip ahead, no timeout into
  the sweep; and a descent-frame rule for the behavioural gate).

### V12-20 belt slant at the start (16:00, `cmp/ur_slant/`, 7798)

Starts: `make_ee_offset_starts.py` (new variants `ee_ur_z±…`) moves the UR end-effector at the
nominal pre_place_1 pose; the grasp holds in all four. Belt slant measured on the 48 belt bodies.
Model V12h + support box (V12-17's box arm yaml), 2 snapshot-restored episodes per start, 42 s.

| start | belt height at UR − at Franka | plane tilt | engaged | stage 0 | Franka sweep reaches +45 mm at | UR z then | UR z min / end |
|---|---|---|---|---|---|---|---|
| demo (set2/gv_00) | +10 mm | 11.4° | (demo) | - | 3.5 s | 28 | 28 / 28 |
| nominal | +10 mm | 11.8° | 2/2 | reached 2.7 s (d 0.30) | 7.0-9.7 s | 34-35 | 32 / 36 |
| UR −10 mm | +1 mm | 11.8° | 1/2 (1 slanted) | reached | 7.0-9.9 s | 34-35 | 32 / 35 |
| UR −20 mm | −9 mm (down to the UR) | 12.8° | 2/2 | reached 2.7 s (d 0.99) | 7.2-7.3 s | 33-34 | 32 / 35 |
| UR −30 mm | −19 mm (down to the UR) | 14.6° | **0/2** over | timeout 5.6 s (d 2.86 > tol 1.07) | **1.6 s** | **64-65** | 33 / 36 |
| UR +15 mm | +25 mm (up at the UR) | 14.0° | **0/2** over | timeout 5.6 s (d 2.38) | **1.9-2.0 s** | **89** | 34 / 36 |

- Up to a moderate slant (UR −20 mm: the belt 19 mm lower at the UR than nominal) the MPC does what
  it does on the nominal start: the UR descends first (z 106 → 35 by 3 s), and the sweep starts
  only at 7-10 s with the belt down (h 3 mm). 5/6 engaged.
- At a stronger slant either way the initial belt latent is too far from the first waypoint (a
  pre-motion demo frame), stage 0 and stage 1 time out, and the Franka sweeps at 1.6-2.0 s while
  the UR is still at z 64-89 mm (belt h about 40 mm): the "over" signature of V12-18 / route 1 of
  the over diagnosis. Afterwards the UR does descend (min 33-34 mm), but the belt is already on the
  flange.
- The sweep at 1.6-2.0 s comes while stage 0 is still active (before any timeout): the demo
  action prior runs on the clock from frame 20 and reaches the demo's sweep frames by ~1.5 s.

**With the stage-synced prior** (V12-14's `prior_sync: stage`, `ur_slant/sync_*`, 16:11-16:17):

| start | engaged | stage 0 | sweep +45 mm at | UR z then | belt h then |
|---|---|---|---|---|---|
| UR +15 mm | **2/2** (was 0/2) | reached 3.0 s (d 0.98) | 5.6 s | 33 | 0 mm |
| UR −30 mm | 0/2 over (was 0/2) | timeout 5.6 s (d 3.44), stages 1-2 time out too | 2.2-2.3 s | 59-60 | 29-30 mm |

- With the prior held at the pre-motion frames until stage 0 is reached, the UR +15 mm start
  behaves like the nominal one: descend first, sweep with the belt down.
- At UR −30 mm the latent never comes within tolerance of the first waypoint and the planner
  still sweeps early with the prior held, so the latent goal itself (or the model) pulls the
  Franka +y there; V12-19's skip-ahead / pre-motion-waypoint removal is the next test for it.
- Combined over V12-20: box + stage-synced prior fixes one of the two strong-slant failures.
  (On the V12-17 grasps the synced prior added nothing: 4/8 vs 5/8.)

### V12-19 waypoint schedule and pre-latch refit-gate rule (offline, CPU, `cmp/wp_schedule/`)

Acts on the over diagnosis (`cmp/over_diag/over_diag.md`). All rules opt-in; off = byte-identical.

| rule | where | what |
|---|---|---|
| A1 `waypoints.min_frame` (int, or `offset` = `mppi_demo_offset_frames`) | `latent_waypoints.py --min-frame` | drop the waypoints before that goal-demo frame; the kept ones are the full schedule's (same tols, durations; prior_sync maps stages to the kept frames) |
| A2 `waypoints.skip_ahead` | `MppiParams.stage_skip_ahead`, node | at each latent, jump to the furthest later stage whose tol holds it (never back) |
| A3 `waypoints.no_timeout_into` (`latch` or a stage index) | `MppiParams.stage_no_timeout_into`, node | that stage is entered only by reaching the previous stage's tol; its timeout keeps the stage (stdout `timeout blocked`, JSONL `blocked`); A2 cannot skip past it; until then the demo prior is zero from the gated waypoint's frame on |
| B `behaviour_gate.max_prelatch_push_increase_mm` / `max_prelatch_descent_drop_mm` | `behaviour_gate.py` | on demo frames from the motion start (first `move:` label, frame 20) to the latch: reject if the candidate's mean u₀ Franka +y exceeds the reference's by X, or its mean UR descent (−u₀ z) falls short by Y (mm/step) |

The goal demo's motion starts at frame 20 (first `move:` label = first non-zero action, as the
prefix / offset code). Its frame-20 latent is still the static start (0.12 from frame 10), so
`min_frame: offset` keeps a frame-20 stage 0 (tol 1.68-2.09 vs frame 10's 1.07-1.51); `21` drops it.

**Open-loop replays** (`offline/replay.md`; recorded latents through the node, noise 0, clearance off;
the states are the recording's, so only the schedule and the commands change). Commanded sums in
the first stage (about 3 s), Franka y / UR z, mm:

| episode (GT, model) | off: goal frame 10 | A1: goal frame 30 | prior_sync | prior_sync + A1 |
|---|---|---|---|---|
| V12-17 box set2/gv_06 #0 (over, V12h) | +85 / −33 | −66 / −171 | +27 / −62 | −95 / −186 |
| V12-18 i0 set2/gv_06 #0 (over, V12v) | +93 / −44 | −50 / −125 | +26 / −92 | −87 / −169 |
| V12-20 UR z−30 #0 (over, V12h) | +80 / −16 | −72 / −142 | +14 / −26 | −97 / −172 |
| V12-20 UR z+15 #0 (over, V12h) | +66 / −44 | −62 / −167 | +9 / −71 | −94 / −186 |
| V12-20 sync UR z−30 #0 (over, recorded with sync) | +29 / −27 (sync) | −92 / −167 | = off | = A1 |

- **Route 1 (V12h/V12v, V12-20's strong slant):** with the frame-10 goal the planner sweeps +y
  while the UR is high, prior or no prior (with the synced prior the push is smaller but still
  positive, which matches V12-20's UR −30 mm). With A1 the first active goal is frame 30 (frame 20
  is reached at 0.1-0.5 s), and the same states get −y and a 125-186 mm UR descent. Σu₀ Franka y
  while the recorded UR z > 50 mm: +62..+83 off → −151..−179 with A1 (−98 on the synced recording).
  A2 alone does the same on these episodes (skips 0 → 1 at the first latent).
- **Engaged controls** (V12-17 set3/gv_03 #0, V12-18 gv_06 #2, V12-20 UR z−20 #0 and sync UR z+15 #0):
  A1 leaves the sweep stage's start unchanged (3.8-5.0 s, belt h −1.4..1.7 mm there).
  - **A2 alone skips past the latch in 4/4** (to the first sweep stage at 1.7-2.9 s with h 5-16 mm),
    also with A1 and the synced prior. Unsafe without A3.
  - **A3 blocks 2/4 engaged runs at the latch** (V12v gv_06 #2, sync UR z+15: the latch stage timed
    out with the belt down) and the open-loop sweep goal never comes. `latch_dist/latch_dist.md`:
    the latch tol (3 latents in a row) is reached in only 4/11 engaged rollouts of the over set,
    while the belt is down (h ≤ 2.5 mm) at d 1.0-3.5 vs tol 1.1-1.3. V12w reaches it in 4/5 overs
    with h 3-5 mm (one undefined), UR z 37-38 mm (route 2). The latch latent tol is not a belt-down signal yet.
  - With A3, before the latch the planner commands −y throughout (no sweep goal, prior masked).
- **V12w (route 2):** every stage is reached anyway; A1 / A3 do not change the sweep time (3.5 s,
  h 3.4 mm). No schedule rule fixes route 2.

**Gate sweep** (`gate_sweep/sweep_{prior,noprior}.md`, `gate_alt.md`; every refit of V12-07,
V12-08, V12-12, V12-18 incl. the 2 gate-rejected candidates; candidate vs the model before it;
iteration i−1's yaml; CPU without clearance; hold push reproduces the loop's: V12w / V12v 0.08 /
0.29 vs 0.15 / 0.28 logged). Pre-latch means, cand − ref, mm/step (prior on):

| pair | cand engaged | d Franka y | d UR descent | hold push cand / ref |
|---|---|---|---|---|
| V12h vs V12g (V12-07 i1) | 4/5 | −0.28 | +0.91 | 0.55 / 0.58 |
| V12w vs V12v (V12-18 i1) | 0/5 | **−0.27** | **+0.27** | 0.08 / 0.29 |
| V12x vs V12w (V12-18 i2) | 0/5 | −0.57 | +0.31 | −0.06 / 0.08 |
| V12-18 i3 candidate (rejected by hold) | - | +0.40 | −0.27 | 1.09 / −0.06 |
| V12-08 i1-i9, V12-12 i1, i4 candidate | 0-1/5 | −0.68..+0.23 | −0.10..+0.68 | 1.29-2.53 |

- **The rule cannot reject V12w vs V12v for any X, Y ≥ 0**: on demo states V12w pushes less +y and
  descends more than V12v (also without the prior, and on the late-descent window). On post-latch
  and hold frames its UR descent is 0.08-0.09 mm/step smaller, less than V12h vs V12g's 0.11-0.12,
  so no threshold there separates them either. V12w's failure is in where its latent "reaches"
  the latch (above), not in its first actions on demo states.
- X 0.3, Y 0.2 accepts every deployed refit (including V12h) and rejects only the V12-18 i3
  candidate, which the hold rule already rejects. It is safe but redundant; not recommended as a
  V12w catcher.

**Checks** (CPU): `check_waypoint_schedule.py --pre-dir` W1-W7 7/7 (yamls == pre + 2 defaulted keys;
derive pre == tree bytes, `--min-frame` == the full artifact sliced; 7 hand cases + 200 random
latent sequences == a reference implementation; pre vs tree node on a V12-18 stream; loop yaml /
command / `check_cfg`, `--dry-run` C1 + V12-18 iters 0/1 identical; gate `decide` pre == tree on 6
recorded gates; with `prior_sync: stage` the idx and the passed prior == the rule incl. skips and the
mask). `check_prior_sync` 5/5, `check_overshoot` 5/5, `check_support_box` 5/5 (P1/O1/B1 now allow
the new defaulted keys). Regenerated `--min-frame 20/21` artifacts == the recorded ones sliced (6/6).

**Enable** (loop config overlay; `configs/wp_a1.yaml`, `configs/wp_all.yaml`, dry runs beside them):

```yaml
waypoints: {stride: 10, window: 3, percentile: 90, floor: 0.5, min_frame: offset}  # A1
# not yet: skip_ahead: true, no_timeout_into: latch  (A2 / A3, see above)
# B: behaviour_gate: {..., max_prelatch_push_increase_mm: 0.3, max_prelatch_descent_drop_mm: 0.2}
```

Next: a sim check of A1 (with and without `prior_sync: stage`) on V12-20's strong-slant starts and
the V12-17 grasps; a belt-down latch criterion before A3 can be used.

### V12-21 grasp angle at the UR jaw (16:59-17:22, `cmp/ur_slant/pivot_*`, 7798)

Starts: `make_grasp_variants.py --ur-pivot-deg=-30,-20,-10,10,20,30` (set `ur_pivot`): the UR
picks with its jaw turned about the closing axis (approach × belt tangent) through the grasp
point, then goes to the nominal pre_place_1, so the UR end-effector pose is the nominal one
(≤ 0.1 mm; up−30 6 mm). 7/7 held. Kept grasp angle (belt tangent at the jaw vs `up+0`, in the UR
gripper frame): ±10° → 9.4-9.9°, ±20° → 16.4-19.7°, ±30° → 25.1-26.5°
(`ur_slant/grasp_angle_in_gripper_frame.png`). Negative angles hang the loop lower (lowest point
33-45 mm vs 56), positive higher (63-73). V12h + box, 2 snapshot episodes each.
(Moving the UR end-effector instead, V12-20 and `ee_ur_r*` starts, changes the angle at the jaw
by only 2-7°: the round belt rolls / pivots in the flat pads.)

| angle | box | box + stage-synced prior | stage 0 (box) |
|---|---|---|---|
| +10° | 2/2 | - | reached 2.7 s (d 1.00) |
| +20° | 1/2 (1 other) | **2/2** | - |
| +30° | 0/2 (under) | **2/2** | timeout (d 3.63) |
| −10° | 0/2 over | - | timeout (d 2.84) |
| −20° | 0/2 over | 0/2 over | timeout (d 4.00 with sync) |
| −30° | 0/2 over | 0/2 over | timeout |

- Same mechanism as V12-20: when the first waypoint (a pre-motion demo frame) is out of reach,
  stages 0-1 time out and the Franka sweeps at 1.8-2.3 s with the UR at z 82-87 mm (belt h 42-48
  mm) → "over". The synced prior removes the early sweep for +20°/+30°, not for the negative
  angles, whose start latents are 2.8-4.0 from the first waypoint. V12-22 tests A1 on them.

Resets on a real robot: discussed (a reset ladder with real-signal checks); deferred by the user.

**MILES plan** (`handoffs/PLAN-20261006-miles.md`, from 18:45): MILES-style deviate-and-return
cycles around demo waypoints as seed data (P-M1, V12-23/24), demos + MILES as a never-evicted
anchor pool with a fixed frame share in every refit (P-M2), seed comparison A (today) / B (demos +
MILES) / C (A + MILES) offline and in closed loop (V12-25/26), the learning loop on the winner
(V12-27), and a state-conditioned BC prior from the MILES data as the MPPI mean (P-M3, V12-28).

### MILES tooling (P-M2)

Offline / CPU only; every option is opt-in and off = byte-identical.
- **Anchor buffer** (loop key `anchor: {kinds, frac, decay_to, decay_by}`, needs `buffer_policy`):
  - Rows of `kinds` (default demo, miles) are never evicted and do not count toward `cap`. The
    short-abort cap, the FIFO and `min_failures` act on the other rows (the on-policy pool).
  - With `frac`, the `buffer` step of iteration i scales the anchors' per-episode `weight`. Their
    frames then make `frac` of the frame-weighted total, Σ weight × frames. That total is what
    `refit_from_buffer.py` samples from: each tuple is drawn with its episode's weight.
  - Pool rows keep 1.0 (short aborts 0.25). `decay_to` lowers the share linearly from refit 1
    to refit `decay_by` (9).
  - Written into `iter<i>/buffer.json`: the rows' `weight` plus an `anchor` block (scale, shares).
  - The seed refit (iteration 0) is unweighted, as before.
  - `buffer_policy.apply(..., anchor_kinds)`, `anchor_weights`, `anchor_frac`.
- **`collection_schedule`** (`[<set>/<id>]`): iteration i collects its 5 rollouts from entry
  i mod n, instead of the seeded pool draw. Eval starts are refused, as in the pool (sha check).
- **`support_box.kinds`** accepts `miles`. `seed_buffers` takes the MILES manifest. `merge_seed`
  is unchanged, so MILES rows need `kind: miles`, absolute `episode` paths and a bool `label_human`.
- **MILES rows are not task attempts** (`kind: miles`, `label_human` False, `sim_cmd_source` 4):
  - `refit_from_buffer.py --non-task-kinds` (default `miles`; `none` = old behaviour): they train
    like any row and are held out by episode as a third group, drawn after the success / failure
    classes, so the task rows' split is unchanged. They are left out of the separation AUC. The
    report gets `split.non_task_*` keys only when such rows exist.
  - The loop's `separation()` (`auc_final_to_goal`) skips them.
  - The trust region (`box_i`) keeps `trust_kinds [demo, mpc]`: MILES excursions are scripted, not
    the policy's actions, so they are excluded on purpose. Its frame rule (`sim_cmd_source` 0/1)
    would drop them anyway.
  - `finetune_lcs_latent.py` (its `is_mpc` filter drops MILES tuples) is only used by the V8
    `onpolicy_loop.py`, not by this loop; unchanged.
  - Without anchors, `buffer_policy` counts MILES rows as failures (FIFO, `min_failures`); with
    `anchor.kinds` containing miles they are anchors.
  - Identity (CPU): split pre == tree on 34 loop / seed buffers × 10 seeds × 2 fracs, and == the
    recorded held-out lists of 26 V12 refit reports; `evaluate` pre == tree for V12v on the V12-18
    seed buffer; loop `separation()` pre == tree == recorded for V12-18 iterations 0-1.
  - With 15 MPC rows relabelled miles: the task rows' held-out split is unchanged, 2 miles rows
    are held out, and the all-episode AUC is 0.846 (it would be 0.898 with them counted as
    failures).
- **`scripts/lcs/seed_compare.py`** (V12-25): runs on `--model NAME=PATH`, where PATH is a
  deploy, a loop iteration dir, or a checkpoint plus `--export-buffer` (exported on the CPU with
  the loop's commands and one `--export-box`). Metrics:
  - one-step / 7-step / recon Chamfer RMS on the off-demo rollouts of V12-17, V12-18 (collects
    and evals), V12-20 and V12-21, split by run, kinematic phase and the 6-d distance to the
    nearest seed-demo frame. Episodes in a model's training buffer are dropped.
  - probes from the final latents of engaged and over episodes (16 × 3 mm Franka +y, ±x, UR ±y,
    minus the zero-input change of the seated-goal distance);
  - `behaviour_gate.py`'s u_0 metrics on the demo frames.
  - Validation: on V12v's held-out split it gives 2.51 / 3.56 / 2.50 mm (the refit report:
    2.50 / 3.56 / 2.49), and on V12x's (2 MPC episodes) 3.12 / 4.15 / 3.10, equal to the report.
    A checkpoint exported here equals V12v's deploy except `exported_at`.
  - Smoke readout (24 episodes, stride 4, 5 demos):
    - V12h predicts better off-demo than V12v (one-step 9.0 vs 13.3 mm, recon-dominated; error
      grows with the distance from the demo tube: 12-18 mm beyond 40 mm).
    - On over / seated latents, V12v predicts that Franka +y lowers the seated distance in
      75-90 % of states (the "+y always helps" signature). V12h predicts the opposite in 100 %.
- **`scripts/lcs/run_hard_suite.py`** (V12-26):
  - `write`: generates `<out>/run.sh` for the 11 hard-suite starts × 2 episodes with
    `cmp/a1_sim/run.sh`'s flags and resets (the grasps: snapshot, then a scripted reset with
    V12-17's seeds). It runs start-major, skips complete runs, and refuses while a socket holds
    the port. `--arm NAME DEPLOY WAYPOINTS MPPI_YAML`; WAYPOINTS `auto:20` builds the A1
    artifact on the CPU (byte-identical to `wp_schedule/artifacts/V12v_min20.npz` for V12v).
  - `summarize --dir ...`: outcome, final wrap / h, stage 0, the first frame with Franka y ≥ 45
    mm (t, UR z, belt h), the final sweep start, F y max, UR z, solve p95, and an arm × start
    table. It also reads `cmp/a1_sim` (V12-22).
- **`data/lcs/realfirst/cmp/configs/v12_27.yaml`** (V12-27 template): V12-18's config plus:
  - the MILES seed (`<MILES_SEED_BUFFER>`; B or C per V12-25 / V12-26);
  - `anchor {frac 0.4}`, `refit.mode lcs_only`;
  - `collection_schedule` = the 10 non-eval hard-suite starts (the nominal is a seat6 eval start);
  - eval every 2 on seat6 plus the other 10 hard-suite starts;
  - ports 7796 / 7797, and `model_id_prefix <MODEL_ID_PREFIX>`.
  With V12-22's winner, also set `waypoints.min_frame` / `demo_prior.sync`. A raw dry-run stops
  on the placeholder; filled with a test buffer, the iteration 0 / 1 dry-runs are clean.
- **Checks** (CPU):
  - `check_anchor_buffer.py` A1-A7: 7/7. It covers identity vs the pre-change module,
    anchors never evicted, the pool decision == `apply(pool)`, cap on the pool only, frame share
    == frac to 1e-15 (and 200k weighted draws within 0.004), and the decay. It also replays
    V12-18 iterations 0-3: off, pre == tree bytes; on, 20/20 demos kept vs 18 off, demo share
    0.400 → 0.325 with decay 0.2. Plus the config refusals and the schedule.
  - Existing checks: `check_buffer_policy` 5/5, `check_curriculum` 8/8, `check_support_box`
    5/5, `check_prior_sync` 5/5, `check_overshoot` 5/5. `check_waypoint_schedule` W2-W7 pass;
    W1 (MppiParams only) passes against its own pre-V12-19 copy.
  - In these runs the pre side is the current `mppi_lcs.py` / node with this package's pre
    `realfirst_loop.py` / `buffer_policy.py`, so the comparisons isolate this change from
    P-M3's concurrent MppiParams edits.
  - Loop `--dry-run` C1_transport and V12-18 iterations 0 / 1: pre == tree.

### MILES-BC prior (P-M3, for V12-28; offline, CPU, `cmp/bc_prior/`)

Opt-in (`prior_mode: demo` default = unchanged): the MPPI prior sequence is a state-conditioned BC
policy rolled through the LCS instead of the demo's actions at the prior index. π sees the belt
latent only (decision: no absolute EE positions, the no-EE-target rule; `--inputs zp` keeps
π(z, p_franka, p_ur) opt-in for comparison).

| piece | what |
|---|---|
| `scripts/lcs/train_bc_prior.py` | MLP π(z) → u (12-d cmd_delta, 2×128 ReLU; `--inputs zp`: + world EE positions), inputs standardised, output `u_mid + u_half y` (deploy box); latents from the deploy's numpy encoder (or `--cache`); demo rows: first non-zero action to the last + `--max-hold-frames` (20); miles rows: return / hold / walk frames only (npz `miles_segment`, row `segments` {name: [a, b)} or `segment: walk`; a miles row without them is refused); best held-out-episode epoch → `bc_prior.npz` (numpy weights, `inputs`, `encoder_sha256` of the deploy encoder) + `.json` |
| `MppiParams.prior_mode` / `bc_prior`, `mppi_lcs.PolicyPrior` | `policy`: u_k = clip(π(z_k)) to the MPPI box, z_{k+1} = LCS step, P_{k+1} = P_k + u_k[:6] (P carried for the LCS only; a `zp` prior also reads it), k < H; computed before the solve (own CUDA graph on CUDA; the solver graph is unchanged), passed as `u_demo`, alpha / weight as today; refused with `prior_sync` or `stage_no_timeout_into`, and on a deploy whose encoder differs from the BC's (retrain after a joint refit; LCS-only refits keep it) |
| node | JSONL `prior_u0`, `prior_ms` only in policy mode; `--demo-actions` not needed; the overshoot guard's `prior_scale` / hold apply to the policy prior too |
| loop | `demo_prior: {alpha, weight, mode: policy, bc_prior: PATH}` → yaml `prior_mode`, `bc_prior` (absolute); `check_cfg` refuses a bad mode, policy without `bc_prior`, with `sync` or `waypoints.no_timeout_into` |

BC for V12h (`v12h_demo/`, π(z); C1_transport iter1 buffer, 20 demos, 1058 frames, 2 held-out demos):
val MSE 0.027 vs 0.21 for the mean (normalised), val MAE Franka y 0.98 mm/step (π(z, p) `v12h_demo_zp/`:
0.0087, 0.77).

Checks (CPU): `check_bc_prior.py` B1-B6 6/6 (params and refusals, solver shared; pre vs tree node
with `prior_mode: demo` on 3 recorded streams (FULL i0 ep4, C1 i1 ep0, V12-20 UR z−30 #0, 114-118
solves): plans, rows, stdout identical; numpy vs torch forward 1.5e-6 (z) / 9.6e-6 (zp) and
closed-loop rollout ≤ 1.8e-5 of the box half-width; a z prior is bit-identical under changed P
(forward and rollout), a zp prior is not; node `u_demo` == the numpy rollout on 59 solves,
other-encoder deploy refused; frame rules, toy fits (z and zp), cache == numpy encoder 1.2e-6; loop
yaml bytes identical off, + 2 keys on). `check_prior_sync` 5/5, `check_overshoot` 5/5,
`check_support_box` 5/5, `check_waypoint_schedule` 7/7 (their P1/O1/B1/W1 allow the 2 new defaulted
keys); loop `--dry-run` C1 iters 0/1 pre == tree. The CUDA-graph path of `PolicyPrior` is untested
(no GPU run): B3 runs a graph-vs-eager comparison when CUDA is available.

**Offline** (`offline_replay.py`, `offline/replay.md`; V12h + box yaml, recorded latents open loop,
noise 0, clearance off; high = recorded UR z > 50 mm, low = < 40 mm; prior = its first action,
clipped; sums of MPPI's u₀, mm; policy = π(z), zp = π(z, p)):

| episodes (GT) | prior while UR high, F y / UR z mm/step: demo → policy (zp) | Σu₀ F y while UR high: demo / policy / zp / no prior | Σu₀ UR z while UR high: demo / policy / zp | Σu₀ F y while UR low: demo / policy / zp / no prior |
|---|---|---|---|---|
| V12-20 UR z−30 #0, #1 (over) | +1.0 / −1.0 → −4.0 / −5.1 (−3.8 / −3.2) | +74, +75 / **−24, −19** / −31, −26 / +59, +62 | −30, −28 / −50, −46 / −29, −26 | −5, −1 / +323, +301 / +210, +228 / +80, +82 |
| V12-21 pivot up−20 #0, #1 (over) | +0.9 / −0.9 → −3.8 / −5.0 (−4.1 / −4.3) | +56, +53 / **−31, −30** / −51, −50 / +40, +38 | −66, −63 / −97, −95 / −84, −84 | −2, 0 / +165, +175 / +119, +145 / +73, +81 |
| V12-20 UR z−20 #0, #1 (engaged) | +0.1..0.4 / −4.2 → −3.0 / −5.5 (−2.6 / −5.3) | −4, −3 / −15, −17 / −16, −18 / −9, −10 | −43, −45 / −37, −38 / −40, −43 | +133, +128 / **+788, +830** / +1087, +1096 / +266, +267 |

- While the UR is high, π(z) no longer suggests the sweep: it pulls the Franka −y and the UR down
  (more UR descent than π(z, p)), and MPPI's commanded Franka y over that phase turns from
  +53..+75 to −19..−31 mm (π(z, p): −26..−51; A1 of V12-19 gave −151..−179 on this kind of episode).
- Once the belt is down the prior still pushes +y: +3.2..3.4 mm/step in the engaged controls
  (π(z, p): +5.9, at the box edge), Σu₀ +788..+830 vs the demo prior's +128..+133. On their seated
  states (h 0.2 mm, Franka y 45 mm vs the demos' 77) π extrapolates "keep sweeping". More hold frames
  do not teach the stop (`probe_states.py`, `offline/probe_states.md`, unclipped, engaged low phase
  π(z) F y +3.3 / +11.7 / +9.3 mm/step with 20 / 60 / all hold frames; π(z, p) +9.9 / +11.5 / +33.7).
  Open loop, so the recorded states never move; in closed loop the support box bounds the drift,
  but this is the +y runaway mode of V12-15.
- Reading: the demo-only BC fixes the early sweep but not the stop; it needs off-demo contact-phase
  data (MILES returns, E2) before a sim run, or `overshoot.fade_last_stage` to drop the prior in the
  last stage.

Enable (loop overlay `cmp/bc_prior/configs/bc_policy.yaml`; the BC must match the deploy's encoder):

```yaml
demo_prior: {alpha: 0.5, weight: 0.1, mode: policy, bc_prior: data/lcs/realfirst/cmp/bc_prior/v12h_demo/bc_prior.npz}
# node yaml: prior_alpha: 0.5, prior_weight: 0.1, prior_mode: policy, bc_prior: <abs path>
```

### V12-22 A1 in closed loop (18:48-19:50, `cmp/a1_sim/`, 7798, solve p95 14-25 ms)

V12h + support box, waypoints from demo frame 20 (`wp_schedule/artifacts/V12h_min20.npz`), ±
stage-synced prior, 2 episodes per start (V12-17's grasps: snapshot then scripted reset with its
seeds; the rest: snapshot). Controls = the box / box + sync arms of V12-17, V12-20 and V12-21
(same starts, resets and seeds). `run_hard_suite.py summarize` → `a1_sim/summary.md`.

| start | box | box + A1 | box + sync | box + A1 + sync |
|---|---|---|---|---|
| nominal | 2/2 | 2/2 | - | 2/2 |
| set2/gv_00 | 2/2 | 1/2 | 1/2 | 1/2 |
| set2/gv_06 | 0/2 | 1/2 | 0/2 | 1/2 |
| set3/gv_02 | 1/2 | 1/2 | 1/2 | 1/2 |
| set3/gv_03 | 2/2 | 2/2 | 2/2 | 2/2 |
| UR −30 mm | 0/2 | 0/2 slanted | 0/2 | **2/2** |
| UR +15 mm | 0/2 | 0/2 slanted | 2/2 | 2/2 |
| grasp +20° | 1/2 | 0/2 | 2/2 | 0/2 under |
| grasp +30° | 0/2 | 2/2 | 2/2 | 0/2 slanted |
| grasp −20° | 0/2 | 0/2 | 0/2 | 0/2 |
| grasp −30° | 0/2 | 0/2 | 0/2 | 0/2 |
| **total** | **8/22** | **9/22** | **12/20** | **11/22** (9/20 without nominal) |

- A1 turns the "over" of the slant starts into "slanted" (box) and, with the synced prior, seats
  UR −30 mm (2/2, the start V12-20 could not), but loses grasp +20°/+30° (now under / slanted).
  Net: no gain at n = 2 per start; not adopted.
- The negative grasp angles fail in every arm: stage 0 times out and the Franka sweeps at
  1.6-2.2 s with the UR at 80-90 mm (belt h 66-87 mm), even with A1.
- Best controller so far: **box + stage-synced prior** (12/20 on 10 hard starts) → E4 (V12-26)
  compares the seed models with it.

### V12-23 MILES collector smoke (19:47-19:58, `miles/smoke/`, 7796)

Collector `scripts/lcs/miles_seed.py` (package P-M1). Per demo: restore its start snapshot,
settle 0.5 s, walk the demo's recorded EE poses (`demo_prefix.relative_targets` + `KnotPath`,
7-knot Franka trajectory, UR IK); at each waypoint, cycles of min-jerk deviation (0.6 s),
straight min-jerk return (0.6 s), hold 0.2 s. Offsets: box free ±15 mm / ±4°, contact
±6 mm / ±2°; half the draws on one axis at the box edge (Franka x/y, UR x/y/z); modes rotate
Franka / UR / both. Not in the plan: each arm's offset is scaled down until the frozen clearance
model keeps ≥ min(2 mm, clearance at the waypoint) (the 2F-85 sits ~4 mm above the board after
the latch, so a −6 mm UR step would otherwise hit it). One file per cycle and per walk segment
(`eval_learned_mpc.build_writer` format, `cmd_delta`, `sim_cmd_source` 4, string key
`miles_segment` walk/dev/return/hold), `buffer.json` rows `kind: miles` (`segment`, `segments`,
offsets, validity, tau, R), `budget.json`, `miles_report.md`.

Belt check: symmetric Chamfer (mean nearest neighbour) of the segmented cloud vs the demo's
`pcd_belt_seg` at frame k ≤ tau. tau = p95 of the same Chamfer between every same-grasp pair of
the 10 clean demos (v1_1 seed + seed_ext, label_human true; 13 pairs) at matched frames: free
3.32 mm, contact 1.82 mm (max 4.00 / 2.50). Sinkhorn / exact OT on the 256-point resampled clouds
was dropped: same-grasp pairs at rest already give 14-16 mm (point density differs with
visibility), Chamfer 1.2-2.4 mm. GT for scoring only: GT label changed or Chamfer of the 150 GT
belt points vs the same reference > its own p95 spread (free 4.00, contact 1.72 mm).

| pass | waypoints | cycles | valid | R | why | check vs GT |
|---|---|---|---|---|---|---|
| `r2_demo` (plan's check) | 20, 30, 33, 43, 53 | 18 | 17 | 53 | disturbed (1.97 > 1.82) | 17/18 |
| `r2_arrival` (`--belt-ref arrival`) | same | 17 | 16 | 53 | disturbed (1.85 > 1.82) | 14/17 |
| `diag` (`--diag-restore`, sim-only) | all 41 (20..198) | 164 | 43 | 43 | disturbed | see below |

`r1` = the first pass with an index-wise GT belt RMSE (counts sliding along the belt as
disturbance; same cycles as `r2_demo`), superseded.

- Reach and grasp are not limiting: EE back within 0.68 mm (Franka) / 0.80 mm (UR) of the
  waypoint target after the 0.2 s hold in all 164 diagnostic cycles; 0 grasp losses, 0 aborts;
  the clearance scaling bound in 25/164 cycles.
- The check vs the demo at k measures the replay's drift, not the cycle: the replayed belt
  already arrives 1.6 mm (latch) to 3.0 mm (end of the hold phase) from the demo's cloud, so the
  diagnostic pass flags 121/164 cycles, from waypoint 43 on. The GT label equals the arrival's in
  162/164 cycles (2 changes at waypoint 49: slanted → engaged, the cycle helped seating).
- Against the arrival cloud (`--belt-ref arrival`) the Chamfer after a cycle is p50 1.13 / p95
  1.58 / max 2.06 mm (contact): tau 1.82 flags 5/164 (all at waypoints 49-52, mid-seat; GT
  agrees on those 5, 11 GT flags at 1.7-2.7 mm missed with the label unchanged); tau 2.5 (the
  demo spread max) flags 0.
- The latent distance to the demo (V12h, whitened) grows 0.2 → 1.9 along the demo; report-only.
- Format: `test_belt_seg_loader.py` PASS, `load_belt_seg_episode` on every file PASS (diagnostic
  pass: 204 files, 3090 pairs), `refit_from_buffer.read_buffer` + `split` read the buffer
  (`refit_from_buffer.py` has no dry-run). `sim_cmd_source` 4 makes `is_mpc` false: the
  trust-region frame rule and `finetune_lcs_latent.py` skip these frames.
- Timing: 1.4 s robot / 1.5-1.7 s wall per cycle; a full demo (41 waypoints × 4) is 243 s robot,
  5.1 min wall, 3294 frames, 93 MB.

For V12-24: the plan's rule (vs the demo, tau p95) stops each demo near the latch / hold
(waypoint 43-53 of 198), i.e. ~30-50 cycles per demo instead of 164. V12-24 therefore checks
against the arrival cloud (tau 1.82 mm) and skips the rest of a waypoint after a disturbed cycle.

### V12-24 MILES seed collection (20:01-20:45, `miles/v1/`, `miles/v2x/`, 7796)

`run_e2.sh`: `miles_seed.py --demos gv_00,gv_01,gv_02 --cycles 4|8 --belt-ref arrival --tau-mm
1.82 --on-disturb skip-waypoint --deploy <V12h>`; all 41 waypoints per demo (20, 25, 28..55 every
3, 58..198 every 5). `--on-disturb skip-waypoint` (new): a disturbed cycle (frames kept) ends that
waypoint's cycles and the walk goes on; only grasp loss or an unreachable pose stops a demo.
Note: `--tau-mm` sets both phases, so the free phase uses 1.82 instead of its own 3.32.

| run | demo | cycles | valid | skipped waypoints (after cycle) | stopped |
|---|---|---|---|---|---|
| 1× (`v1`) | gv_00 | 157 | 154 | 28 (c0), 49 (c1), 52 (c1) | no |
| | gv_01 | 161 | 159 | 28 (c1), 49 (c2) | no |
| | gv_02 | 164 | 164 | - | no |
| 2× (`v2x`) | gv_00 | 314 | 310 | 25 (c3), 28 (c0), 49 (c4), 55 (c7) | no |
| | gv_01 | 322 | 321 | 28 (c1) | no |
| | gv_02 | 321 | 318 | 28 (c6), 49 (c7), 52 (c1) | no |

| run | cycles | disturbed | grasp lost / unreachable | files | frames | robot s | wall | size |
|---|---|---|---|---|---|---|---|---|
| 1× | 482 | 5 | 0 / 0 | 602 | 9692 | 716 | 15.3 min | 269 MB |
| 2× | 957 | 8 | 0 / 0 | 1077 | 18717 | 1381 | 28.2 min | 519 MB |

- Reach: back within 1.05 mm of the waypoint target after every cycle; clearance scaling
  bound in 55 (1×) / 140 (2×) cycles; 0 aborts. Per-waypoint tables: `miles_report.md`.
- Belt check after a cycle (Chamfer vs arrival): p50 0.88 / p95 1.6 / max 2.0-2.1 mm. The
  flags are at waypoints 25-28 (free phase, 1.82-1.92 mm, GT not disturbed) and 49-55
  (mid-seat, 1.88-2.06 mm, GT disturbed).
- Check vs sim GT: 441/482 (1×), 863/957 (2×). The check flags 3/42 (1×) and 4/94 (2×) GT
  flags; it raises 2 / 4 false alarms. The GT flags are marginal: GT Chamfer 1.5-2.5 mm
  against gt_tau 1.72, at waypoints 34-52. In 16 / 31 the GT label changes, mostly under ↔
  other at h ≈ −5 mm and toward engaged / slanted mid-seat. No cycle moves the belt to over
  or outside. So the GT measure is as noisy as the check here, and the agreement figure is a
  weak score.
- Latent distance to the demo (V12h, whitened): at most 1.4 (1×) / 2.4 (2×); report-only.
- Loader: `test_belt_seg_loader.py` PASS; `load_belt_seg_episode` on every file PASS
  (1×: 9090 pairs, 2×: 17640); `refit_from_buffer.read_buffer` + `split` read the manifests.

Seed manifests (`miles/make_manifests.py`): rows are copied as-is with absolute paths, and
the header is the schema of `cmp/v12_18/seed_buffer_merged.json`. The MILES rows carry
label_strict / cloud_success / n_prehold_frames / fallback_demo with neutral values and a
`label_human_detail` dict.

| manifest | content | rows |
|---|---|---|
| `miles/seed_B.json` | the 20 demo rows of V12-18's merged seed + MILES 1× | 622 (20 demo + 602 miles) |
| `miles/seed_C.json` | V12-18's merged seed + MILES 1× | 636 (20 demo + 14 excitation + 602 miles) |
| `miles/seed_B2.json` | the 20 demo rows + MILES 2× | 1097 (20 demo + 1077 miles) |

MILES rows have `label_human` false and `sim_cmd_source` 4 (`is_mpc` false). The refit's
success-stratified split and its AUC therefore treat them as failures; that handling belongs to
the refit side.

### V12-25 seed comparison (21:07, `cmp/seed_compare/report.md`)

Seeds trained with V12-18's seed recipe (joint, 60 epochs, latent 10, Sinkhorn, refit seed 4):
A = V12v (V12-18's seed: 20 demos + 14 OU excitation, 6,952 frames), B = 20 demos + MILES 1×
(13.7k frames, 5.3 min), C = A's seed + MILES 1× (6.5 min), B2 = 20 demos + MILES 2× (8.8 min).
Held-out (own split): one-step 2.79 / 2.55 / 2.56 mm, 7-step 4.09 / 4.16 / 3.59 mm for B / C / B2.

Off-demo prediction (87 closed-loop rollouts of V12-17/18/20/21 none of them trained on; every 3rd
frame; Chamfer RMS mm, one-step / 7-step):

| split | A | B | C | B2 |
|---|---|---|---|---|
| all (15,083) | 13.5 / 14.3 | 12.8 / 13.8 | 11.5 / 12.5 | **9.8 / 12.4** |
| descent | 13.5 / 14.6 | 12.1 / 13.4 | 12.2 / 12.2 | **9.2 / 11.7** |
| seat | 13.8 / 14.5 | 13.2 / 14.1 | 11.7 / 12.6 | **9.9 / 12.7** |
| 20-40 mm from the demo tube | 11.6 / 12.2 | 11.0 / 12.3 | 10.0 / 10.9 | **8.0 / 10.1** |
| > 40 mm from the tube | 17.5 / 18.4 | 16.5 / 17.0 | 14.7 / 15.5 | **12.8 / 16.4** |

- The error is dominated by reconstruction (recon ≈ one-step) and falls with MILES data volume
  (B2 > C > B > A); inside the demo tube (0-10 mm) A is as good or better.
- Probes (16 × 3 mm constant input from engaged / over final latents, change of the seated-goal
  distance vs zero input): the MILES models rate Franka −x and UR −y as worse in 96-100 % of states
  (A: 32-90 %), i.e. they learned the drift directions. All four models still predict that +y
  lowers the seated-goal distance from engaged final states (worse in 0-14 %); no ground truth
  for that, so the closed loop decides.
- Behaviour on demo frames (MPPI u_0, box yaml): before the latch the MILES models descend the UR
  more (+2.1-2.6 vs +1.7 mm/step) and pull the Franka back (y −0.8..−1.7 vs −0.65): "descend
  first". Hold push is higher for C / B2 (1.02 / 0.80 vs 0.24 mm/step).

### V12-30 seated goal set (offline, CPU, `cmp/goal_set/`)

Today the last stage tracks one latent: the goal demo's (set2/gv_00 episode_0000) final latent.
Other grasps seat the belt in other shapes, so their seated latents are far from it (V12-25/26).
V12-30 replaces that one latent with a **set of seated latents**: the encoded final belt clouds
of every seated demo, and optionally of the MPC successes. These are belt observations only, so
the set contains no EE-pose targets.

**Separation** (`analysis.py` → `analysis.md`): the final frame of 219 recorded closed-loop
rollouts from V12-17/18/20/21/22/26 (64 engaged, 126 over, 16 slanted, 12 under, 1 other).
Each final frame is encoded by each model and its whitened distance to the goal is measured.
The table gives the AUC of engaged vs not engaged, with the median distance of each group in
brackets (engaged / not). The GT outcome is report-only. "LOO" (leave-one-out) means a rollout
is never in its own goal set.

| goal | V12h | V12v | B | C | B2 |
|---|---|---|---|---|---|
| single goal (today) | 0.956 (2.70 / 5.16) | 0.876 (1.39 / 2.64) | 0.764 | 0.653 | 0.446 |
| min over 20 demos | 0.940 (1.76 / 3.15) | 0.882 | 0.841 | 0.838 | 0.666 |
| min over 14 seated demos (`label_human`) | 0.908 (1.92 / 3.21) | 0.853 | 0.792 | 0.789 | 0.621 |
| soft-min τ 1 / τ 4, 14 | 0.927 / 0.951 | 0.878 / 0.882 | 0.792 / 0.787 | 0.755 / 0.678 | 0.605 / 0.529 |
| min over 9 `label_strict` demos | **0.961** (2.48 / 5.02) | 0.884 | 0.793 | 0.705 | 0.427 |
| 14 seated demos + 20 MPC successes (LOO) | 0.955 (0.84 / 2.80) | 0.921 | 0.714 | 0.847 | **0.792** |
| 9 strict demos + 18 strict successes (LOO) | **0.962** (0.84 / 3.12) | **0.933** | 0.710 | **0.854** | 0.784 |

- **V12h / V12v:** the single goal already separates engaged from over (0.98 / 0.91). Using the
  set brings the engaged states much closer: V12h median d 2.70 → 0.84 with successes.
- **One borderline demo lowers the 14-demo AUC:** the 14-demo set scores 0.908 for V12h. Without
  demos_ext2 episode_0004 (set3/gv_02, labelled engaged at h 4.3 mm, wrap 62°) it scores 0.951.
  The strict human label (wrap ≥ 90°, |h| ≤ 1 mm) drops that demo.
- **MILES seeds (B, C, B2):** here the set helps a lot. C goes 0.653 → 0.854 and B2 0.446 →
  0.784, but only with successes in the set. These seeds' "under" states lie closer to the
  seated latents than their engaged states (B2 vs under: 0.04-0.52).
- **Demo spread:** the seated demos' latents lie 0.12-2.87 from the goal demo's (median 0.84,
  V12h). The other models give medians of 0.47 (V12v), 0.61 (B), 1.08 (C) and 0.94 (B2). The
  slanted demos lie 2.08-3.18 away (V12h).

**Implementation** (all opt-in; off = byte-identical):
- **Params:** `MppiParams.goal_set` (a `goal_set.py` npz of K latents) and `goal_set_tau` (τ,
  default 0 = hard min).
- **Cost:** for each knot, the track / terminal cost becomes a soft-min over the set:
  m − τ log mean exp(−(d² − m)/τ), with m = min d². It stays in [min, mean] and equals the
  single-goal cost for a set of 1. The set is a static tensor, so the solve stays CUDA-graph
  safe.
- **Node, last stage:** both the cost and the "reached" / skip-ahead distance (JSONL `dist`)
  use the set. JSONL `d_single` keeps the distance to the stage goal. Earlier stages use a
  solver without the set. The overshoot guard's `d_seat` is unchanged. Track mode is refused.
- **Builder:** `scripts/lcs/goal_set.py` encodes the final frame (numpy twin) of the seated
  rows: `label_human`, or `label_strict` with `--strict`. The goal demo's entry equals the
  waypoint artifact's last waypoint exactly.
- **Loop key:** `goal_set` builds `iter<i>/goal_set.npz` / `.json` from B_i with deploy_i, once
  per iteration. It is rebuilt only if deploy_i or the rule changes.

**Open-loop replay** (`replay.py` → `replay.md`): V12-22 box + sync episodes on V12h, using the
recorded latents. Clearance is off on the CPU. The stage sequence is the same in every variant.
The cells are sums of u₀ over the last (seated) stage, about 20-30 s: Franka y / UR z in mm
(negative z = descent). Sets come from C1's B_1.

| episode (GT) | off | 14 seated | 14, τ 1 | 14 + 4 successes | 9 strict |
|---|---|---|---|---|---|
| set2/gv_00 #1 (over) | −23 / −218 | +191 / −191 | +65 / −207 | +192 / −191 | **−88 / −227** |
| set3/gv_02 #1 (over) | −40 / −181 | +158 / −157 | +13 / −175 | +177 / −155 | **−113 / −196** |
| grasp −20° #0 (over) | −1 / −56 | +182 / −38 | +181 / −39 | +182 / −38 | **−76 / −68** |
| grasp −30° #0 (over) | −309 / −102 | −305 / −98 | −304 / −99 | −281 / −96 | −312 / −103 |
| set2/gv_06 #1 (engaged) | +107 / +19 | −107 / +38 | −35 / +27 | −80 / +40 | +28 / +18 |
| set3/gv_02 #0 (engaged) | +130 / +11 | +31 / 0 | +2 / +1 | +442 / +15 | −3 / −10 |

- **The planned Franka +y does change with the set, and the members decide which way.**
  - In all 4 over episodes, the 14-demo set's nearest member is the borderline demo above.
    Pulling the state toward it adds +149..+162 mm of +y after the first 10 s of the stage:
    the runaway signature.
  - Without that demo (13 demos: −55..−112) or with the 9 strict demos (−76..−113), the planner
    commands less +y than off (−1..−40) and descends the UR as much or more.
  - On the 2 engaged episodes, the strict set halves or removes the late +y push: +28 / −3 vs
    +107 / +130.
- **Grasp −30° is unchanged:** its latent is far from every member (d 4.1-6.0).
- **Caveat:** this is open loop, so the states do not respond to the commands. The test is
  whether the push direction changes, not whether the belt seats.

**Checks** (CPU):
- `check_goal_set.py --pre-dir` G1-G5 5/5:
  - G1 params: 122 yamls == pre + 2 defaulted keys; bad params and bad npz are refused.
  - G2 cost: a set of 1 or 5 copies is bit-identical to the single goal in float32 and
    float64; a random set matches the numpy reference.
  - G3 solver: off pre == tree; a set of 1 == off; with a set, `plan_cost` == `rollout_cost`.
  - G4 node: off pre == tree; a set of 1 == off over 160 latents into the last stage; the
    14-demo set leaves plans before the last stage unchanged, logs `dist` / `d_single`
    correctly, and fires "reached" / skip-ahead on a set member at d 2.87; track mode is refused.
  - G5 loop: yaml bytes == pre without the key; the npz == the builder on B_1; check_cfg gives
    8/8 refusals; loop `--dry-run` for C1_transport and V12-18 iterations 0/1 is pre == tree.
- `check_support_box` 5/5, `check_prior_sync` 5/5 and `check_overshoot` 5/5. Their B1/P1/O1
  now accept the two new defaulted keys.
- The pre side is the current tree with only the V12-30 edits removed, which isolates them
  from V12-31's concurrent `stage_start` edits.

**Enable** (loop config):

```yaml
goal_set: {kinds: [demo], strict: true, include_successes: true, softmin_tau: 0.0}
# goal_set: {}  # = 14 label_human demos, hard min (defaults)
```

Recommendation: use `strict: true` (with `include_successes` once the loop has strict
successes). The default `label_human` set contains a borderline demo, and on over states that
member pulls the plan toward +y. Next: a sim A/B on the hard suite (box + sync ± the strict
goal set).

### V12-31 progress-gated start for an out-of-reach first waypoint (offline, CPU, `cmp/start_stage/`)

Starts: grasp −20° / −30° at the UR jaw (V12-21 box / box + synced prior, V12-22 the same + A1),
V12h. Diagnosis `diag.py` → `diag.md`; replays `offline_replay.py` → `offline/replay.md`.

**Diagnosis.**
- The first two waypoints (frames 10, 20) are the goal demo's static start: 0.13-0.14 from its
  first latent. On −20° / −30° the start latent is 3.04-3.12 from them (tol 1.07 / 1.97) and
  6.5-9.8 from every later waypoint. So "start at the nearest pre-latch waypoint" picks stage 0
  anyway; it is a no-op here.
- Recorded, first 3 s with the static goal (Σu₀ Franka y / UR z, mm, including the 0.2 action
  noise): −20° box +70 / −24..−25, synced +38..+39 / −42; −30° box +6..+8 / −33, synced
  −56..−57 / −78..−80. On −20° the
  distance to the static goal falls 3.08 → 2.2-2.3 by 1.3-1.5 s while the Franka sweeps +35 mm,
  then rises to 4.0-4.3. It is never within tol.
- LCS cost at the recorded first latent (TorchLcs on CPU, the run's yaml), static goal, −20°:
  - Constant-rate argmin of tracking + terminal: Franka y +6 mm/step, UR z 0.
  - 'Sweep' (+4 mm/step y) costs 127-129 less than 'descend UR' (−4 mm/step z): tracking −72..−74,
    terminal −57, effort −3, prior +4, box 0.
  - The demo's own descent costs 881-888 more than the sweep: the LCS predicts it moves the latent
    8.4 away from the static start.
  - CPU MPPI u₀ +0.3 / −0.5 mm (Σ over H +17 / −14).
  - −30°: the argmin is −4 / 0 (hold, −y), so there is no push to sweep at t = 0.
- Under the frame-30 goal every start flips: argmin −4 / −6 mm/step, the demo's descent is the
  cheapest sequence, MPPI u₀ −2.5..−2.7 / −4.7 (Σ −11..−24 / −42..−51).
- **The early sweep comes from the tracking and terminal terms toward an unreachable
  static-start latent.** It does not come from the demo prior: weight 0.1, 10-40× smaller,
  pulling toward descent. The box adds 0. In-reach starts (nominal, set3/gv_03, UR z −30 mm)
  feel the same pull, but they are within tol in 0.1-0.3 s.
- Why not (b) or (c):
  - (b) Timeouts scaled with the remaining distance (d / tol 1.6-2.9) would keep the static goal
    longer.
  - (c) A freeze while the distance falls would hold it through the 1.3-1.5 s in which the
    distance falls *because of* the sweep.
- **A second cause that no stage schedule fixes** (`diag.md` §5, GT report only):
  - At the start, the near-seat belt sits 20 / 11 / −1 mm below the UR end-effector on
    −10 / −20 / −30°. It sits 28-35 mm below on the nominal and engaged starts and 38-56 mm
    below on +10..+30°.
  - At the UR's lowest z (32-37 mm; MPPI floor 27 mm) the belt is still 11-27 mm above the
    groove. This also holds in runs that never swept early (synced −30°: UR 23-32 mm, h 12-19 mm).
  - Seating these grasps needs more than descend-then-sweep.

**Rule** (opt-in; off = unchanged):
- Where: `MppiParams.stage_start: progress`, node `progress_start`, loop
  `waypoints.start: progress`.
- At the first latent (after a bank pick), take the stages before the latch waypoint's. If a
  `stage_no_timeout_into` gate is set, only those below its previous stage count.
- Drop the start-state waypoints whose tol does not hold the current latent. A start-state
  waypoint is one whose tol holds the goal demo's first latent `z_traj[0]`.
- Start at the nearest remaining stage by latent distance. If none is left, start at the latch
  stage. The node never steps back afterwards.
- Uses only the artifact's latents: no frame numbers, no task constants.
- Logs: stdout `[mppi] start stage k (frame f) … dropped start-state stages [...]`; JSONL `start`
  on the first solve.
- With `prior_sync: stage` the prior index starts as on a normal entry.
- With one stage (`--stages last`) it is a no-op.

**Replay** (open loop, recorded latents, noise 0, clearance off, 100 latents = 7.5 s; Σu₀ mm):

| starts (episodes) | off: F y / UR z 0-3 s | start: F y / UR z 0-3 s | F y while rec. UR z > 50 mm, 0-5 s: off → start |
|---|---|---|---|
| −20° box (3) | +70..+71 / −21..−22 | −71..−73 / −168..−172 | +36..+77 → −109..−130 |
| −20° synced (3) | +40..+43 / −36..−37 | −93..−95 / −185..−189 | +22..+42 → −104..−132 |
| −30° box (3) | +4..+6 / −33..−34 | −84..−87 / −162..−165 | −30..−12 → −121..−144 |
| −30° synced (3) | −38..−60 / −78..−82 | −73..−77 / −167..−180 | −35..−59 → −73..−78 |

- Out-of-reach starts drop frames 10/20 (A1 artifact: frame 20) and track frame 30 from t = 0.
  The commands equal V12-19's A1(21) artifact.
- Engaged controls:
  - Nominal #0 and UR z −30 mm #0 (V12-22 sync + A1) are in reach: start 0. Stage events and
    commands are identical to off. A1(21) does change them (nominal F y / UR z −10 / −95 vs
    −2 / −82).
  - set3/gv_03 #1 (d 2.53 > tol 1.97) starts at frame 30; off reached frame 20 at 0.2 s.
    0-3 s goes +1 / −84 → −10 / −98, and the events are identical from 1.2 s on.
- Risk: V12-21 synced +30° #0 (engaged off, by waiting through the two static stages with the
  descent prior). It drops both (d 2.26 / 2.24) and descends at once: −95 / −180 vs +10 / −65.
  A1 (frame 20) + sync was slanted 0/2 there in V12-22.
- Open-loop limit: after t = 0 the states are the off run's, so the frame-30 stage times out at
  3 s in every "on" replay, and the latch stage (synced prior at frame 33 = the first sweep
  action) follows. Only the sim shows two things:
  - whether the latent comes within frame 30's tol 2.07 in closed loop (V12-22 sync + A1 −20°
    after its 3 s timeout: d 5.3 → 2.8 in 1.5 s);
  - whether descending first seats the belt, given the second cause above.

**Checks** (CPU; pre = the tree with only V12-31's edits reverted):
- `check_stage_start.py` S1-S5 5/5 (`start_stage/checks/`):
  - S1: 120 yamls to_dict == pre + `stage_start: None`.
  - S2: the rule == brute force on 2000 cases; 8 hand cases; 200 random latent sequences ==
    reference, with skip / gate / streak.
  - S3: pre == tree, option off, on 3 recorded streams (59 plans each); on: JSONL `start` == the
    rule.
  - S4: synced prior idx / `u_demo` == rule from the start stage.
  - S5: loop yaml off == pre, on == pre + 1 key; 3 refusals; `--dry-run` C1 + V12-18 iters 0/1
    pre == tree, dirs untouched.
- `check_waypoint_schedule` 6/7. W1 fails only on V12-30's new `goal_set` / `goal_set_tau` keys
  (with those allowed: PASS).
- `check_prior_sync` 5/5, `check_support_box` 5/5, `check_overshoot` 5/5. P1 / O1 / B1 / W1
  now allow the new defaulted key.

**Enable** for a sim test: box + synced prior + start, full waypoints (frames 10/20 dropped only
when out of reach):

```
--mppi-config data/lcs/realfirst/cmp/start_stage/configs/mppi_boxsync_start.yaml \
--mppi-waypoints data/lcs/realfirst/cmp/C1_transport/iter1/latent_waypoints.npz  # + V12-22's other flags
# loop: waypoints: {stride: 10, window: 3, percentile: 90, floor: 0.5, start: progress}
```

`mppi_box_start.yaml` is the same without the synced prior. Suggested starts: grasp ±20°/±30°,
UR z −30 / +15 mm, nominal, ×2. Watch +30° (the risk above) and, on −20° / −30°, belt h at the
UR's lowest z.

### V12-32 MPPI sensitivity (offline, CPU; `cmp/mppi_sweep/`)

V12h + `sbox_sim/mppi_boxsync.yaml`, driven through the deployed node code on a fake LCM (clearance
off, noise 0). 12 hard-start episodes (6 over, 6 engaged). Screens: (1) open-loop replay of the
recorded latents, "sweep share" = solves planning Franka +y > 1 mm/step while the recorded UR is
above 50 mm (the over signature); (2) closed loop on the learned LCS from the start (21 s) and from
the final stage (12 s), 2 seeds. Noise floor ±2 points on group sweep shares.

| setting | sweep share over / engaged | model closed loop from the start |
|---|---|---|
| base | 14 / 18 % | under-sweeps on 7/12 starts |
| H 24 + w_q 0.5 | **4 / 11 %** | descend then sweep 24/24, no runaway |
| w_q 0.5 + prior weight 0.3 | 4 / 9 % | descend then sweep 18/24 |
| H 24 | 8 / 10 % | 23/24 |
| w_q 0.5 | 6 / 9 % | 11/24 |
| traps: sigma 0.25 / prior alpha 0.75 / ess 0.1 | best open loop | latent diverges / runaways on the model |
| worse: sigma 0.75, H 8, alpha 0.25, w_q 2, ess 0.02 | 20-42 % | |

- No knob changes the stage-0/1 timeouts behind the negative-grasp-angle failures; the settings only
  change what the planner does while they run (less sweep, more UR descent).
- The final-stage model runs do not separate the settings (V12h pulls −y on the model: its
  under habit). Screens are proxies; contact (drag, stalled UR) is invisible to both.
- (The study's subagent could not write its report file; this summary and `tables.md` stand in.)

### V12-33 / V12-34 controller tweaks in closed loop (01:25-04:00, V12h, hard suite 11 × 2)

| start | sync (base) | + start | + goal set | both | H 24 + w_q 0.5 | w_q 0.5 + pw 0.3 |
|---|---|---|---|---|---|---|
| nominal | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| set2/gv_00 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| set2/gv_06 | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| set3/gv_02 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| set3/gv_03 | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 |
| grasp +20° | 2/2 | 0/2 | 2/2 | 0/2 | 2/2 | 2/2 |
| grasp +30° | 1/2 | 0/2 | 2/2 | 0/2 | 0/2 | 0/2 |
| grasp −20° / −30° | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| UR +15 mm | 1/2 | 2/2 | 2/2 | 2/2 | 0/2 | 0/2 |
| UR −30 mm | 0/2 | 2/2 | 0/2 | 2/2 | 0/2 | 0/2 |
| **total** | **10/22** | **11/22** | **11/22** | **10/22** | **8/22** | **8/22** |

- With V12h, the controller is saturated around 10-11/22: each tweak fixes some starts and breaks
  others (n = 2 per start). The offline screens (V12-30/31/32) predicted the push directions but
  not the closed-loop gain; the MPPI settings that looked best offline lose on the slant starts.
- The negative grasp angles fail in every arm, consistent with V12-31's finding that the belt sits
  almost level with the UR fingertips there (needs another strategy or more UR depth).
- Next: the controller is not the bottleneck; the model / data are. V12-27 (the Jin & Posa loop on
  the MILES seed, box + synced prior) started 09:43.

### V12-37 honest baseline without the demo action prior (12:11-13:20, `cmp/noprior/`)

V12h + support box + latent waypoints (C1 iteration 1 artifact), `prior_weight 0`, `prior_alpha 0`,
hard suite 11 starts × 2 (V12-17 grasps: snapshot then scripted reset; the rest snapshot).

| start | prior on (V12-33 base) | no prior, H 16 | no prior, H 24 |
|---|---|---|---|
| nominal | 2/2 | 1/2 | 2/2 |
| set2/gv_00 | 1/2 | 1/2 | 1/2 |
| set2/gv_06 | 0/2 | 0/2 | 0/2 |
| set3/gv_02 | 1/2 | 0/2 | 1/2 |
| set3/gv_03 | 2/2 | 1/2 | 1/2 |
| grasp +20° | 2/2 | 2/2 | 2/2 |
| grasp +30° | 1/2 | 0/2 | 0/2 |
| grasp −20° / −30° | 0/4 | 0/4 | 0/4 |
| UR +15 / −30 mm | 1/4 | 0/4 | 0/4 |
| **total** | **10/22** | **5/22** (over 9, slanted 8) | **7/22** (over 12, slanted 3) |

- The prior carried about half of the successes; model + latent cost alone seat 5-7/22.
- The longer horizon helps (H 24: 7/22, fewer slanted); the planner is myopic without the prior.
- All runs are linked for the viewer in `cmp/noprior/all_recordings/` (`<arm>__<episode>__<outcome>`).
- Next: make the cost carry the task — a learned cost-to-go in the latent as MPPI's terminal cost
  (steps-to-seated from demos, MILES and successes; V10's obstacle-aware distance as a cost),
  longer horizon / better sampling, then the Jin & Posa loop improving model + cost-to-go.

### V12-36 task-driven refit weighting (code)

Why: in V12-35 every deployed refit drifted (behaviour gate descent cosine 0.95 → 0.43 → 0.27,
hold-frame +y push −1.32 → +0.94 mm/step, collections 3/5 → 0/5 → 0/5). Hypothesis: the on-policy
data is mostly failure tails far from the demos, and fitting them bends the one global LCS. All
new options are opt-in; defaults are byte-identical. Outputs: `cmp/refit_weighting/`.

| item | what |
|---|---|
| per-frame weights (lcs_learning `refit_from_buffer.py`) | an optional row field `frame_weights` [[start, stop, w], ...] over the episode's frames (non-overlapping, w ≥ 0, frames outside a segment 1). The tuple at frame t is weighted row weight × w(t) in the sampler. `--mode lcs_only` only (joint refuses it). The report gains `frame_weights` stats (weighted tuples, mass removed). |
| `onpolicy_weighting` (loop key; `task_common/onpolicy_weighting.py`) | `{off_track: {tol_k 3, after_s 1, streak 1}, off_track_weight 0.2, fail_weight 1}`. The tube ratio of a latent is the min over the goal demo's per-frame latents `z_traj` of the whitened distance / tol_j, where tol_j is the tol of the nearest waypoint (`Waypoints.tol_at`). The rollout goes off track at the first frame whose ratio is above tol_k for `streak` frames. Frames from there + after_s get off_track_weight, and failed rollouts (human label) are scaled by fail_weight. The `buffer` step writes `frame_weights` + `off_track` on iteration i's new MPC rows. It uses their recorded `sim_latent` (the deploy encoder's latent; numpy encoding with deploy_i if absent) and iteration i's `latent_waypoints.npz`. Carried rows keep theirs. Anchor shares use frame-weighted frames. Stats go to buffer.json `onpolicy_weighting` and a learning-curve section. Needs `waypoints` and `refit.mode` lcs_only. |
| `refit.from_seed` (loop key) | every refit starts from θ_0's checkpoint (the seed) and is fit on all of B_i, instead of warm-starting from θ_{i-1} or its rejected candidate. The gates still compare against θ_{i-1}. model.json / refit.json record `from_seed`. Refused with `fresh_each_iter`. |
| checks | `scripts/checks/check_refit_weighting.py` 7/7 (`cmp/refit_weighting/check_refit_weighting.{log,json}`): W1 weight math; W2 refit pre == tree without the field (refit.log, lcs_params, report), all-1 weights == none; W3 off-track rule vs brute force; the recorded `sim_latent` equals deploy_2's encoding (0.0) and the node's goal distance (3e-6); W4 `st_buffer` replay of V12-35 iterations 0-3: off = pre bytes, on = recomputed weights, anchor share 0.400 frame-weighted, encoded fallback == sim_latent; W5 config refusals; W6 from_seed init in --dry-run + stubbed `st_refit`; W7 loop --dry-run pre == tree (C1_transport 0/1, V12-18 0/1, V12-35 0-4). Also `check_anchor_buffer` 7/7 (pre), `check_buffer_policy` PASS, `check_curriculum` 8/8. The running V12-35 loop picked up the new loop at iteration 5 (default path, no change). |
| config | `cmp/configs/v12_36.yaml` = V12-35 + `onpolicy_weighting` (above) + `refit.from_seed: true` + anchor frac 0.7, root `cmp/v12_36`, models V16, `refit_gate: report`. **The demo action prior is removed** (`demo_prior: null`, user decision: it is banned from now on), so the collect/eval yaml has no `prior_*` and the harness gets no `--mppi-demo-actions`. The behaviour gate gets `no_prior: true`. Nothing in the weighting uses the prior: the off-track rule uses only the latent tube. Not run. Run: `data/lcs/realfirst/cmp/run_cell.sh v12_36 0 9`. |

**Offline evidence on V12-35** (`scripts/lcs/refit_weighting_offline.py`,
`cmp/refit_weighting/offtrack_fractions.md`). Each column is the share of each iteration's
on-policy frames down-weighted (after_s 1 s). Latents are the recorded `sim_latent`, scored
against that iteration's waypoints.

| iter (GT outcomes) | tol_k 2 | tol_k 3 | tol_k 4 |
|---|---|---|---|
| 0 (4 over, outside) | 49 % (4/5 rows) | 0 % | 0 % |
| 1 (3 engaged, 2 over) | 24 % (2/5; also 20 % of the successes' frames) | 0 % | 0 % |
| 2 (5 over) | 55 % | 6 % (1/5) | 0 % |
| 3 (5 outside) | 0 % | 0 % | 0 % |
| buffer B_4 (all MPC rows of iterations 0-3) | 32 % | 2 % | 0 % |

Reading: the hypothesis does not hold in latent space. The failure tails are **not** far from the
demo tube. "Outside" (belt off the pulley) and most "over" endings sit next to goal-demo frames
26-30, the descent before the latch (ratio 0.3-0.7 for outside, 1-2.8 for over). The demo passes
through such belt shapes on its way down, and the min over every demo frame accepts them. Against
the seated demo frames only (first hold on), the same tails are at 3.8-4.0 × tol (outside) and
2.1-5.3 (over), while engaged endings are at 1.1-1.3. With the requested rule, tol_k 3 down-weights
almost nothing (B_2: 0 of 5600 MPC frames), so V12-36 as configured is effectively from_seed +
anchor 0.7. tol_k 2 cuts 24-55 % but also hits successes. A progress-aware tube (only the demo
frames past the stage the rollout has reached) would separate them; that needs a decision.

**Refit arms for iteration 2** (`cmp/refit_weighting/report_arms{,_noprior}.md`). Each arm is
V12-35's LCS-only refit on B_2 (= `iter1/buffer.json`), fit on CPU for 10 of the loop's 30 epochs
(4 threads each, about 3 h). Each candidate is then exported with box_1 and run through the loop's
iteration-2 behaviour gate against V15a (the seed): 3 demos + iteration 1's 3 successes, stride 4.
The gate runs on CPU **without the clearance term**: the CPU path of the clearance term takes over
10 s per plan, without it 0.3 s.

Check on the CPU setup: the as-run V15b (GPU, 30 epochs) gets descent 0.38 / hold −1.44 here. The
loop's GPU gate had 0.43 / −1.30, with the same verdict.

| arm (init; MPC frames down) | held-out one-step / h7 mm, ρ(A) | with prior: approach / descent cos, hold push | no prior: approach / descent cos, hold push |
|---|---|---|---|
| V15a (seed, the reference) | 5.25 / 9.08 (same split) | 0.93 / **0.96**, +0.50 | 0.56 / 0.67, +0.53 |
| V15b as run (warm, GPU 30 ep) | - | 0.84 / 0.38, −1.44 | 0.80 / −0.61, −1.46 |
| (a) original (warm from the iteration-1 candidate; 0) | 4.98 / 14.64, 1.119 | 0.91 / 0.41, −1.13 | 0.80 / −0.77, −1.17 |
| (b) from_seed, tol_k 3, anchor 0.4 (seed; 0/5600) | 5.02 / 12.95, 1.019 | 0.95 / 0.83, −0.00 | 0.80 / −0.47, −0.04 |
| (c) from_seed, tol_k 2, anchor 0.4 (seed; 2042/5600) | 5.05 / 14.35, 1.067 | 0.95 / 0.73, +0.40 | 0.90 / −0.50, +0.39 |
| (d) = v12_36.yaml: from_seed, tol_k 3, anchor 0.7 (seed; 0/5600) | 5.05 / 13.33, 1.017 | 0.93 / **0.91**, +0.37 (**accept**) | 0.88 / 0.20, +0.38 (reject) |

Reading (one seed per arm, 10 epochs):
- Starting from the seed removes most of the drift: descent 0.41 → 0.83, hold push −1.13 → 0.00,
  ρ(A) 1.12 → 1.02. Adding anchor 0.7 brings the descent back to 0.91 / +0.37, and the
  prediction and behaviour gates both accept.
- Off-track down-weighting at tol_k 2 does not help: descent 0.73 vs 0.83, ρ(A) 1.07. At tol_k 3
  it has no data to act on.
- **Without the demo prior** (V12-36's controller), the seed itself has only 0.67 descent cosine.
  Every refit turns the descent negative except (d), which reaches 0.20. The banned prior was
  masking part of the model's descent error. V12-36 should expect weaker descent than V12-35's
  iterations 0-1.

### V12-39 learned cost-to-go (offline + code, CPU; `cmp/value_cost/`)

Without the demo prior the planner is myopic (V12-37: 5-7/22). V12-39 adds a learned cost-to-go
V(z) in the deploy's latent (V12h) as an opt-in MPPI cost term. V10's idea (states blocked by the
pulley far from the goal) follows from outcomes: a belt on the flange gets V ≈ "never".

**Data** (real-available only; `train_value.py encode`). Every frame is encoded with V12h's numpy
encoder from `pcd_belt_seg`. Total: 1,382 episodes, 236,583 frames.

| source | episodes | label |
|---|---|---|
| demos + OU excitations (seed, seed_ext, demos_ext2 buffers) | 20 + 14 | `label_human` |
| closed-loop rollouts (`*/iter*/collect`, a1_sim, ctrl_ab, noprior, sbox_sim, ur_slant) | 397 (128 engaged) | index.json outcome == engaged (the bit only) |
| MILES v1 + v12_29b | 815 cycles, 136 walk segments (6 sessions) | session: last walk segment engaged |

**Value** (seconds; cost dt per frame until success; γ = 1 − dt / 20 s, so V_max = 20 s = never):
- Success episode: the seat time t* is found in hindsight, as the first frame from which the 5-frame
  median whitened distance to the episode's final latent stays ≤ 1.5. Against the GT seat time it is
  0 frames off on the demos (median) and +4 frames on MPC runs (p90 +81). Frames from t* on cost 0.
- Failed episode: terminal V_max.
- MILES: a session's walk segments are joined into one demo-progress chain. A valid cycle bootstraps
  on V(its arrival latent) after the return; a disturbed cycle on its last latent.
- Targets: TD(λ 0.9) over 12 fitted rounds with the low expectile 0.2. This is action-free optimistic
  value learning (HIQL-style): the best outcome the data shows from similar latents.
- Network: MLP 10 → 128 → 128 → 1 on (z − mean) / z_std, output V_max·sigmoid. Batch mix: MPC 0.5,
  demos 0.25, MILES 0.25.
- **Model augmentation (`aug`)**: pairs (LCS prediction ẑ(t+H | z_t, recorded commanded actions),
  target at t+H) for H = 8 / 16 / 24. MPPI evaluates V on such predictions, and V12h's 24-step
  error is 1.5-4 whitened (worse than no motion off the demos).
- **Conservatism (`ood`)**: points 1-4 whitened from the data and > 1.0 from every training input get
  target V_max (weight 0.2). Without it, `tde` gives V ≈ 0 on the latents the LCS drifts to
  (2-6 whitened from any data), and the planner can exploit that.
- Model for the sim: `tde_aug_ood_all` (all data). Offline numbers below use its held-out twin
  `tde_aug_ood`.

**Offline evaluation** (`eval/`, GT for scoring only). Held-out = all 44 V12-37 episodes + 20 % of
the other rollouts / demos by episode: 114 rollouts (35 engaged, 41 over, 18 slanted). Score on
the final frame; lower = closer to seated. Columns:
- d-matched: P(V over > V engaged) for over / engaged finals whose single-goal distances are within
  0.5 of each other.
- ρ ttg: Spearman of the score vs the GT time to go on held-out successes (pooled / median per episode).
- ρ GT value: Spearman of the score vs the GT time to go (failed runs = never), on the true latent /
  on the LCS 24-step prediction.

| score | AUC eng vs over + slanted | eng vs over | median eng / over | d-matched | ρ ttg | ρ GT value true / LCS-24 |
|---|---|---|---|---|---|---|
| V `tde_aug_ood` | **0.967** | 1.000 | 0.13 / 19.9 s | **1.00** (34) | 0.33 / 0.41 | 0.74 / **0.72** |
| V `tde` (no aug / ood) | 0.975 | 1.000 | 0.02 / 19.4 s | 1.00 | 0.43 / 0.74 | 0.75 / 0.61 |
| V `td` (MSE, λ 0.9) | 0.994 | 1.000 | 0.05 / 19.9 s | 1.00 | 0.46 / 0.72 | 0.76 / 0.61 |
| V `mc` (Monte Carlo) | 0.993 | 0.999 | 0.02 / 20.0 s | 0.97 | 0.41 / 0.41 | 0.76 / 0.60 |
| d single goal (today) | 0.917 | 0.999 | 3.28 / 5.17 | 0.94 | 0.34 / **0.87** | 0.67 / 0.64 |
| d strict set of 9 (V12-30) | 0.921 | 0.999 | 3.07 / 5.05 | 0.97 | 0.34 / 0.87 | 0.67 / 0.64 |

- In-sample (all 397 rollouts), `tde_aug_ood` scores 0.987 vs 0.967 / 0.971, matching V12-30's
  single-goal 0.956.
- **Over states rank far:** median V 19.9 s, every held-out over final is above the engaged p90, and
  over is ranked farther than an engaged state at the same latent distance in 34/34 pairs (distance:
  0.94).
- **V orders progress within an episode worse than the distance does** (episode median ρ 0.41 vs
  0.87). Along a run, V starts at 5-10 s, rises into the region where the data mostly fails, and
  drops to 0 at seating. So it is mostly a success / failure funnel, not a clock.
- Hard starts (up−20/−30, UR z−30 mm) have no success in the data. V can say "not this way" there,
  not how to seat.

**MPPI** (opt-in; off = byte-identical):
- `MppiParams.value = {path, weight, where: terminal | all_knots}`: cost += w·V(z_H) or w·Σ_k V(z_k).
  `task_common/latent_value.py` holds the numpy weights and the torch twin `value_forward`.
- The solver holds static tensors (CUDA-graph safe; checked graph == eager). JSONL `value_H`.
- The npz is refused on another z_std / n_x / format. The node also refuses another encoder
  (fingerprint), applies the term in every stage, and logs JSONL `v_now`.

**Replays** (`replay.py`, CPU, the 22 V12-37 H 24 episodes, the run's yaml, clearance off, noise 0;
`replay_tde_aug_ood.md`, `replay_tde.md`, `replay_tde_aug.md`):
- Open loop: recorded latents, first 10 s; Σu₀ over 0-3 s in mm.
- Closed loop: the learned LCS as the plant, 21 s. "Descend first" = the UR goes down 20 mm before the
  Franka goes +10 mm y. "Over-like" = the Franka reaches y 45 mm while the UR is above 50 mm.
- "seat" = the C1 artifact cut to its seated waypoint (`waypoints_seated_only.npz`, one 30 s stage).

| setting (`tde_aug_ood`) | Σu₀ F y / UR z, 15 failed | 7 engaged | sweep share failed / engaged | closed: descend first | over-like | F y gain while UR high, failed |
|---|---|---|---|---|---|---|
| off (V12-37 H 24) | +12 / −44 | +1 / −55 | 24 % / 19 % | 14/22 | 5/22 | +30 mm |
| + V w 30 | **−26 / −41** | −17 / −49 | **5 % / 2 %** | **20/22** | **0/22** | +6 |
| + V w 100 | −22 / −42 | −8 / −50 | 10 % / 8 % | 20/22 | 0/22 | +7 |
| + V all knots w 4 | −34 / −37 | +12 / −42 | 18 % / 24 % | 20/22 | 0/22 | +3 |
| seated goal only | +46 / −127 | +74 / −112 | 43 % / 58 % | 16/22 | 18/22 | +60 |
| seated goal only + V w 100 | +11 / −108 | +42 / −97 | 33 % / 40 % | 18/22 | 16/22 | +57 |

- **With the waypoints, the value removes the early +y sweep on every hard start.** Σu₀ F y / UR z
  over 0-3 s, off → V w 30:
  - up−20: +60 / −18 → −24 / −19; up+20 / +30: +58 / −39 → −11..−32 / −45..−48;
  - UR z−30 mm: +71 / −12 → −19 / −14; set3/gv_02 #1: +32 / −23 → −18 / −15;
    set2/gv_00 #1: +36 / −29 → −4 / −20.
  - Cost: some UR descent on the in-reach starts (nominal −69 → −59 mm).
- **Seated goal only** descends hard (UR −110..−160 mm in 3 s) but also sweeps +y early on the
  nominal-like starts (+110). The value cuts this to +40..+90 (and to −28..−50 on up−20 / set3 #1 /
  set2 #1), yet on the model it stays over-like (16/22).
- **The weaker models change little and mixed.** `tde` (no aug / ood) + w 100: sweep share 21 % /
  26 %, descend first 10/22. Its V on the LCS-predicted knots is unreliable (above).
- Open loop the states do not respond, and the closed loop runs on the learned model. Contact (UR
  depth, the −20/−30° belt-level problem of V12-31) is invisible to both.

**Checks** (CPU; pre = the tree before V12-39; `checks/`):
- `check_value_cost.py` V1-V5 5/5, also with `--cuda` (graph == eager, max |dU| 0):
  - V1: 171 yamls == pre + `value: None`; 9/9 bad blocks and 3/3 bad npz refused.
  - V2: numpy == torch (float64 4e-15 s, float32 8e-6 s).
  - V3: on − off == w·V (float64 3e-16 rel); weight 0 == off bit for bit.
  - V4: off pre == tree; solver `plan_cost` == `rollout_cost`; `value_H` == numpy.
  - V5: node off pre == tree (rows, plans, stdout); `v_now` == numpy; other encoder refused.
- Reruns: `check_support_box` 5/5, `check_prior_sync` 5/5, `check_overshoot` 5/5, `check_stage_start`
  5/5, `check_waypoint_schedule` 7/7, `check_goal_set` 5/5.
  - Their B1 / P1 / O1 / W1 / S1 / G1 now allow the defaulted `value` key and a pre that already has
    their own keys.
  - G1 skips yamls with the set on (tree only); the first G1 run failed only on the two ctrl_ab
    goal-set yamls.

**Enable** (sim A/B; yamls = `noprior/configs/mppi_noprior_H24.yaml` + `value`, model
`models/tde_aug_ood_all/value.npz`):

```
uv run --frozen python scripts/lcs/run_hard_suite.py write --out data/lcs/realfirst/cmp/value_cost/sim --port <p> \
  --arm value_wp_V30 <V12h deploy.npz> data/lcs/realfirst/cmp/C1_transport/iter1/latent_waypoints.npz \
        data/lcs/realfirst/cmp/value_cost/configs/mppi_value_aug_wp_V30.yaml \
  --arm value_wp_V100 <V12h deploy.npz> data/lcs/realfirst/cmp/C1_transport/iter1/latent_waypoints.npz \
        data/lcs/realfirst/cmp/value_cost/configs/mppi_value_aug_wp_V100.yaml \
  --arm value_seat_V100 <V12h deploy.npz> data/lcs/realfirst/cmp/value_cost/waypoints_seated_only.npz \
        data/lcs/realfirst/cmp/value_cost/configs/mppi_value_aug_seat_V100.yaml
```

- Control: V12-37's `noprior_H24` (7/22).
- `configs/mppi_value_{wp,seat}_*.yaml` use the superseded `tde_all` (no aug / ood) and are kept
  only for traceability.
- Next: the sim A/B. If it helps, refit V each iteration of the Jin & Posa loop from the new labelled
  rollouts (`train_value.py` on B_i).

### V12-40 C3+ controller path (prep; code + offline, CPU; `cmp/c3plus/`)

Goal: run the current latent deploys (belt-cloud PointNet, latent 10) with magna's C++ learned MPC
(worktree `magna-deploy-learned-lcs`, C3+) instead of the MPPI node, in the loop and the hard
suite, for an MPPI-vs-C3+ A/B with the same model. No sim was run and no C++ binary was launched.

| item | what |
|---|---|
| `scripts/lcs/c3plus_params.py` | `write`: per variant an LCS yaml + controller params yaml (+ demo_traj yaml) in the worktree under `systems/parameters/learned_archive/2026-10-07/c3plus/<tag>/` (the controller resolves paths from its CWD), manifest + copies in `cmp/c3plus/<tag>/`. `check`: a python mirror of the C++ load (drake's strict keys parsed from `learned_lcs_params.h`, the `DRAKE_DEMAND`s of `Validate` / `Serialize` / `LoadLcs`) |
| `scripts/lcs/c3plus_offline.py` | `lcp` (exact LCP vs the trained PGD on task states), `solve-times` (earlier C++ logs), `box-loss` (support box / clearance in MPPI runs) |
| loop | `controller: c3plus` (default `mppi`, byte-identical) + `c3plus: {variant, w_q, w_r, admm_iter, box_z, exact_tol, archive}`: collect / eval write the C3+ files per run (`iter<i>/c3plus_<run>.json`) and pass `--params`; the export checks E3 at 1e-2; MPPI-only options (prior, bank, goal set, overshoot, waypoint schedule) are refused; the MPPI yaml is still written (behaviour gate / policy screen proxy). Overlay: `cmp/c3plus/configs/v12_36_c3plus.yaml` |
| hard suite | arm config `c3plus[:wp2\|pp2\|seat\|track]` (built from the arm's deploy and waypoints npz) or a worktree params yaml: `run_c3` lines with `--params`, no MPPI flags; summary solve p95 from the C++ log samples |

**Mapping.** The C++ `Stages()` holds at most two fixed goals; its only many-reference mode is
`demo_traj` (one stage). Costs are V10's (the last C3+ campaign).

| | MPPI (V12-36) | C3+ `wp2` (default) |
|---|---|---|
| goals | 7 waypoint stages: frames 10, 20, 30, 33 (latch), 43, 53, 199; waypoint tol, 3 s each, reach streak 3; seated 30 s | stage 0 = the latch waypoint (tol = its waypoint tol: V16b 1.62, V12h 1.34; timeout = Σ max_s up to the latch = 12 s); stage 1 = seated (tol 1e-6: runs to 30 s). One latent below tol ends a stage (no streak). Frames 10, 20, 30, 43, 53 are dropped |
| cost | w_q 1 (whitened), w_T 5, w_r 0.3, w_s 0.3, H 16 | Q = diag(1/z_std²) on every knot, R = 0.3 diag(1/half²), no input-change term, N 7, admm_iter 2 |
| inputs | box_i ∩ ±6 mm / ±2° | box_i, hard (the LCS yaml's u_lb / u_ub) |
| EE | floors, ceiling start + 60 mm, ±20° rotation envelope, support box (soft, 6 coordinates), clearance | the same floors (hard); opt-in `box_z`: the box's z faces as hard bounds |
| exploration | action noise 0.2 in collect | none (deterministic) |

- Other variants: `pp2` (fallback: the export's own goals, frames 33 / 199, tols [goal.json
  tol_1, 1e-6], [10, 30] s), `seat` (seated only; the loop's `last` stages), `track` (demo_traj
  progress over the goal demo's per-frame latents from frame 20, w_p 0).
- The waypoint goals differ from the export's by 7e-6 / 9e-6 whitened (numpy vs torch encoding of
  the same frames). On the goal demo, stage 0 ends at frame 31, 2 frames before the latch.
- **Found:** no belt-seg export's `learned_lcs.yaml` (V12a-V16d) has `n_engaged_episodes`. The C++
  loader (drake, `allow_cpp_with_no_yaml` false) refuses it ("is missing"); the generator adds
  `n_engaged_episodes: 0`.

**What C3+ cannot express** (`box_loss.md`; executed EE vs each run's box, MPPI JSONL):
- Support box: the C++ `ee_constraints` are world-z bounds on the two EEs (hard, knots 1..N−1).
  Its quadratic `w_p` pulls the EE to its current position (damping), not into a box. The z faces
  were never used: 0 % of frames outside or within 3 mm of a z face (V12-17 box, V12-37, V12-36).
  All the box's work is on x / y faces. Without the box (V12-17 control, V12h) 70 % of frames are
  outside an x / y face (Franka x lo); with it 5-11 % (UR y hi), and 28-33 % of V12h's frames sit
  within 3 mm of the Franka x lo face. So `box_z` adds nothing, and C3+ loses V12-17's effect
  (box 5/8 vs 1/8).
- Gripper clearance: not expressible (floors only). MPPI's term hits 35-44 % of the samples per
  solve; the chosen plans keep ≥ 2 mm margin in 99.6-100 % of solves; executed min clearance p1
  5.0 mm without the box, 6.9-7.8 mm with it.
- Rotation envelope ±20°: not expressible (per-step rotation bounds only); MPPI's term hits 20-43 %
  of the samples. Earlier C3 runs turned the Franka 26-39° (seat) and 111-167° (+15° yaw).
- 7 waypoints → 2 (or `track`), horizon 7 vs 16, no exploration noise, no reach streak.

**E3 / LCP consistency.** `check_latent_encoder.py` at the default `--exact-tol 1e-2`: V12h PASS
(loose PGD vs exact 2.3e-5), V16b PASS (9.3e-5; the loop had checked V16b at 0.05). On task states
(goal demo + 5 collect rollouts per model, `lcp/lcp.md`), the exact LCP (active-set enumeration;
F is PD: one solution in all 2861 / 2694 states, one active λ) vs the trained PGD-25: per step max
1.3e-5 (V12h) / 2.0e-7 (V16b) whitened, after 7 steps 5.6e-5 / 1.9e-6, against stage-0 tols of
1.3-1.6. C3 plans on the trained model. C3+'s own plan consistency at admm_iter 2 is a separate
issue (2026-09-29 E4: plans claim 3-4× the decrease their inputs give).

**Solve time** (`solve_times.md`; the `solve=` lines of 995 earlier C++ logs, ~1 Hz samples, N 7):

| C3 size | admm_iter | p50 ms | p95 ms | > 75 ms |
|---|---|---|---|---|
| n_z 50 (n_x 16 + EE) | 2 | 25.7 | 79.9 | 6.8 % |
| n_z 44 (n_x 16, no EE) | 2 | 23.2 | 36.2 | 0 % |
| n_z 50 | 3 / 5 / 10 | 44 / 71 / 116 | 68 / 129 / 187 | 2 / 40 / 94 % |

V12h / V16b are n_x 10 + 6 EE = n_z 44: expect p50 20-25 ms and p95 35-80 ms (the n_z 50 tail is
CPU load) against the 75 ms period. admm_iter 2 stays.

**Checks** (`checks/check_c3plus.log`, CPU): K1-K6 6/6. K1 the mirror passes 5 archived params the
C++ loaded (n_z 50 = the C++ log) and fails the raw export, an unknown key and a wrong stage-list
length; K2 the 8 V12h / V16b file sets pass, the matrices equal the export yaml and deploy.npz
(the MPPI's model), u bounds = box_1, goals = the waypoints; K3 loop `--dry-run` pre == tree for
C1_transport and v12_36, iterations 0 / 1; K4 with `controller: c3plus` the 9 harness commands use
`--params`, E3 at 1e-2, files written and checked on a scratch iteration (also `box_z`, eval
floors), 5 refusals; K5 hard-suite MPPI-only `run.sh` / `suite.json` pre == tree, a C3+ arm adds
11 `run_c3` lines; K6 E3 and LCP. Also: the harness `--dry-run` accepts a generated params yaml.
The worktree binary (built 2026-09-30 10:13) predates the last edit of `learned_lcs_c3.cc`
(10:27); V8 / V10 ran with it.

**A/B** (run 2026-10-07 19:32-20:30; result in the run index: C3+ with V12h 6/22, C3+ with V16b 0/22, MPPI with V16b 5/22; one sim, port 7798 in `run.sh`, it refuses a taken port):

```
bash data/lcs/realfirst/cmp/c3plus/ab/run.sh
uv run --frozen python scripts/lcs/run_hard_suite.py summarize --dir data/lcs/realfirst/cmp/c3plus/ab
```

- Arms: `mppi_v16b` (V16b + `v12_36/iter1/eval/mppi_eval_fe_nominal.yaml`: box, clearance, H 16, no
  prior), `c3p_v16b` (V16b, wp2), `c3p_v12h` (V12h, wp2). The V12h MPPI control is V12-37's
  `noprior` (H 16, 5/22) in `cmp/noprior/`.
- Loop with C3+: `--config` overlay `cmp/c3plus/configs/v12_36_c3plus.yaml` on a new root (or
  `C1_transport_c3plus.yaml`, which drops C1's demo prior).

### Replay viewer fix and regression check (2026-10-06 14:15-14:45, tooling)

- Bug: `replay_learned_mpc.py` assumed a 16-d latent and world-frame ordered belt decoders. For
  the 10-d `belt_cloud_pointnet` models (V11-02 on) the planned-EE layer crashed and the planned
  belt was drawn ~270 mm away as a tube through 256 unordered points.
- Fix: latent size from each recorded plan (`x_sol` = [z; p_franka; p_ur]); `LatentDecoder.frame`
  read from `decoder.npz` (default `world`); pulley-frame clouds moved to world with the known
  pulley pose and drawn as two point clouds: light blue = the current latent (knot 0), dark blue
  = the plan's last knot (7 steps, ~0.5 s ahead). MPPI runs record no demo-goal files, so the
  target layer now falls back to the decoded goal latent MPPI tracks (`z_ref`, the active stage's
  waypoint): green cloud.
- Check `scripts/checks/check_replay_learned.py` (headless, git HEAD module vs tree, fake viser
  scene): 15 old 16-d `mpc_eval` runs identical (6 also with a world-frame decoder: belts
  identical); V11-01 / V11-02 / V12-07 / V12-12 runs draw without error and the decoded current
  belt matches the GT belt to 3-7 mm at best (V11-01's noisy-cloud model drifts to 40-97 mm later
  in its episode: model error, not the viewer).

### V12-41 encoder accuracy readout (2026-10-07, offline)

Question (user): can the encoder + LCS capture the belt well enough? Belt-cloud Chamfer RMS (mm):

| model | held-out recon (demo-like frames) | off-demo recon (V12-25, 87 rollouts) | off-demo one-step | > 40 mm from the demo tube (recon) |
|---|---|---|---|---|
| V12v (seed A: 20 demos + 14 OU) | 2.49 | 13.44 | 13.47 | 17.36 |
| seed B (demos + MILES 1×) | - | 12.94 | 12.84 | 16.64 |
| seed C (A + MILES 1×) | - | 11.41 | 11.53 | 14.56 |
| seed B2 (demos + MILES 2×) | - | 9.58 | 9.81 | 12.41 |
| V12h | 2.66 | - | - | - |
| V16a / V16b / V16f | 3.21 / 3.17 / 5.00 | - | - | - |

- Off-demo error is 4-5× the held-out error and grows with the distance from the demo tube
  (B2: 6.7 mm at 10-20 mm, 12.4 mm beyond 40 mm).
- One-step prediction ≈ reconstruction at every split: the latent cannot represent those belts,
  so a better LCS step cannot help; the encoder (or its training coverage) is the bottleneck.
- Coverage matters more than size: MILES 2× (23 min robot) cuts off-demo error by 30 %.

Encoder options discussed with the user (2026-10-07/08), for the next study:
- train on every real frame (labels are not needed: failures, aborts, MILES deviations count) with
  point augmentation (resampling, dropout, noise, occlusion; no rotation about the pulley axis,
  the belt also wraps the small pulley);
- action-sensitive latent: a train-only head that recovers the executed actions from z_{t+k} − z_t
  (Delta-JEPA style, arXiv 2608.29434) and multi-step latent prediction; larger latent (10 → 32);
  lower reconstruction weight (keep the decoder for the viewer);
- local geometry: Point-BERT-style tokens (FPS + radius groups) + ViT-Tiny, or PointNet++;
- an external backbone pretrained on public data (Point-MAE, Point-BERT, Sonata), frozen + head
  or partly fine-tuned, to cut the real data needed;
- **rejected: simulation pretraining** (user): the sim is the real system here, so pretraining on
  extra sim data would hide the real data cost.
- in the loop, keep the encoder learning from on-policy data (warm start, small learning rate,
  anchored old frames) so it improves where the controller goes.

### V12-42 encoder study (2026-10-08 00:50-, `cmp/encoder_study/`, GPU trainings, CPU scoring)

Question: which encoder recipe represents the belt best off the demos, for the least real data?
Simulation pretraining is excluded (the sim is the real system).

Setup:
- Code: lcs_learning trainer options `--pc-aug*` (resample from the raw segmented cloud, occlusion
  ball, dropout, 1 mm noise; training inputs only), `--act-recon-weight/--act-recon-steps`
  (Delta-JEPA head: MLP from Δz_k = enc(o_{t+k}) − enc(o_t) to the k executed actions, k 1 and 5,
  train-only, not exported), `--model belt_cloud_pointmae` (Point-MAE, official ShapeNet
  `pretrain.pth`, N 256 / G 32 / M 16; deploy = torch encoder in the node, ~2.5-6 ms per frame).
  Checks: `check_act_head_aug.py` 18/18 (defaults bit-identical), `check_pointmae_integration.py` 8/8.
- Tiers (`scripts/lcs/encoder_tiers.py`, `encoder_study/tiers/`): T0 3 demos (601 frames, 45 s robot),
  T1 + V12-29b MILES (9,022 frames, 11.4 min, = V12-35's seed), T2 + V12-35/36 on-policy (113k
  frames, 150 min). No all-data tier (user).
- Scoring (`scripts/lcs/encoder_eval.py`): E_off = V12-25's 87 off-demo rollouts, E_hs = 176 no-prior
  hard-suite rollouts (V12-37/39/40); disjoint from every tier. One-step / 7-step / recon Chamfer
  RMS, probes, action response, ridge action decodability from Δz.
- Trainings: V12-35 seed recipe (fresh joint, 60 epochs, latent 10, Sinkhorn, lr 3e-4), frozen code
  copies (`outputs/sim_belt_v12_20261006/v12_42/code_snapshot_*`), ~4.5 min each (PointNet).

T1 results (one-step mm off-demo / hard suite; action decodability R² k=1 Franka / UR):

| arm | E_off | E_hs | R² k=1 |
|---|---|---|---|
| E0 today's recipe, seed 4 / 5 | 9.18 / 10.41 | 11.34 / 11.49 | 0.19 / 0.07 |
| V15a (V12-35's seed model) | 9.18 | 11.25 | 0.21 / 0.08 |
| E1 + augmentation | 8.35 | 10.04 | 0.21 / 0.09 |
| E2 aug + action head λ 0.1 | **8.03** | **9.99** | **0.33 / 0.12** |
| E2 aug + action head λ 1 | 11.02 | 11.95 | 0.32 / 0.11 |
| E2 aug + action head λ 10 | 11.19 (7-step 250) | 12.86 | 0.34 / 0.11 |
| E2 action head λ 1, no aug | 12.28 | 13.18 | 0.33 / 0.12 |
| E4 Point-MAE frozen | 17.69 | 18.49 | 0.00 / 0.00 |
| E4 Point-MAE, last 2 blocks fine-tuned | 18.61 | 18.88 | 0.00 / 0.00 |
| E4 Point-MAE absolute scale, last 2 blocks | 15.28 | 15.56 | 0.06 / 0.02 |
| E4 Point-MAE absolute scale, all fine-tuned | 14.37 | 14.50 | 0.02 / 0.01 |

- Augmentation: −10-15 % on both sets. The action head at λ 0.1 adds a little and makes the
  Franka action readable from Δz (R² 0.19 → 0.33); the UR stays weak (≈ 0 in the seat phase).
  λ ≥ 1 shrinks the LCS's action response 5-10× and hurts prediction.
- Seed-to-seed spread of the same recipe is ~1.2 mm, as large as the arm differences: replicates
  queued (E0 s6, E1 s5/s6, E2 λ 0.1 s5/s6, λ 0.03, latent 32 ± head).
- Point-MAE underfits (train recon 7.9 vs 5.9 mm, latent std ~1/3 of PointNet's): ShapeNet features
  with per-cloud normalisation do not carry mm-level belt geometry. Follow-up with absolute-scale
  input (`--pm-norm fixed`): last 2 blocks fine-tuned held-out recon 5.8 mm, E_off 15.3 / E_hs 15.6;
  whole network fine-tuned held-out 3.8 mm (near PointNet's 3.2) but E_off 14.4 / E_hs 14.5 and
  action R² ≈ 0. With 22M parameters on 9k frames it fits the demo neighbourhood and generalises
  worse than the small PointNet: **Point-MAE dropped** for this data scale.
- Latent 32 (aug + head λ 0.1, 3 seeds): E_off 7.86 ± 0.28, E_hs 9.87 ± 0.28 vs latent 10's 8.05 ± 0.40 /
  9.83 ± 0.21: no clear gain, so latent 10 is kept (smaller C3+ problem).
- 3-seed summary (`encoder_study/t1_summary.md`): E0 9.52 ± 0.77 / 11.51 ± 0.17, E1 aug 8.96 ± 0.56 /
  10.70 ± 0.68, E2 aug + head λ 0.1 **8.05 ± 0.40 / 9.83 ± 0.21** (−15 % on both sets).
- V12-43 (sim) started 02:57: V17a = E0 seed 4 vs V17b = E2 λ 0.1 seed 4 (the median seed of each).
  Then T0 / T2 trainings of both recipes for the learning curve.
- **Learning curve (today's recipe E0, seed 4; one-step mm E_off / E_hs, `encoder_study/t02_eval/`):**
  T0 (3 demos, 45 s robot) 16.32 / 16.01 (7-step 38 / 47: unstable); T1 (+ MILES, 11 min) 9.18 /
  11.34 (3-seed mean 9.52 / 11.51); T2 (+ V12-35/36 on-policy, 150 min) **4.21 / 6.53** (7-step 4.49 /
  6.99). Coverage of the states the controller visits halves the error; T2's on-policy rollouts
  start from the same hard-suite starts as the eval sets (other episodes), so part of the gain is
  in-distribution. E2 on T0: 15.78 / 15.74 (no gain); E2 on T2 was stopped (closed loop already lost).
- The +y probe varies strongly with the seed (E0: 92 % vs 60 % "worse" from over states); E2 λ 0.1
  rates +y as improving everywhere (0 %), a risk for the sweep runaway in closed loop.

### V12-43 diagnosis: why V17b goes "over" (03:20, offline on the first 4 starts; scripts and outputs `encoder_study/sim/diag/`)

Episodes: V17a 9 (4 engaged, 4 over, 1 slanted), V17b 8 (all over).
- **Mode 1 (both models): stage 0 never reached → sweep with the UR high.** Waypoints 0 and 1 (demo
  frames 10, 20) are the demo's static start. From a start outside the frame-10 tol, MPPI spends two
  3 s timeouts on it while commanding Franka +y 1.3-1.9 mm/step (Franka y +45 mm at 1.2-2.0 s with
  UR z 72-95 mm; the demo sweeps at frame 33 with the UR down); the belt ends over (h 30-33 mm).
  The LCS drifts under zero input from static states (24 steps: V17a 2.7, V17b 4.75 whitened), and
  every constant input moves away from waypoint 0, +y least: sweeping is the least-bad option (as
  V12-31). V17b: 6/8 episodes (median start distance / frame-10 tol ≈ 2.0); V17a 3/9 (≈ 1.0).
- **Mode 2 (V17b only): no −y dip before the latch.** In V17b a one-step Franka +y input points
  toward the seated goal at 100 % of 476 recorded states (median cosine +0.56; V17a 41 %, −0.04), and
  −y is rated worse at the latch. MPPI skips the demo's −y dip (lowest Franka y −1..−2 vs V17a
  −16..−20 mm, demo −32), misses the latch, and sweeps with the belt 7.5 mm up. Likely from the
  action head: Δz must encode actions, and the dominant demo action is the +y sweep ending at the seat.
- Waypoint tols did not declare stages early (V17b's wide frame-20 tol never triggered early).
- Fixes ranked: (1) `stage_start: progress` (V12-31; config) → V12-44; (2) do not take the action
  head to closed loop without a probe gate (+y "worse" from over states ≥ 50 % and the +y/seat
  cosine positive at ≤ 60 % of states; E2 λ 0.1 fails both, E0 passes); (3) make the LCS hold under
  zero input at static states (hold segments in the data, or a zero-input consistency loss).

V12-44 (prepared 03:25, `cmp/encoder_study/sim44/`, port 7797, after V12-43): V17a and V17c (= T1 E1
seed 4, point augmentation only; frame-10 tol 1.33) with `stage_start: progress`, otherwise the
V12-43 yaml. Then T0 / T2 trainings of E0 and E2.

### V12-44 diagnosis and a response-direction gate (04:10, offline; `encoder_study/sim44/diag/`)

- `stage_start: progress` removed mode 1 in both arms: no stage-0/1 timeouts; Franka y reaches 45 mm
  at 3.9-8.4 s with the UR at 35-46 mm (V12-43 mode-1 episodes: 1.2-2.0 s, UR 72-95 mm).
- V17c (augmentation only) fails in a new mode 3: the belt is down (h 1.4-3.5 mm at sweep onset),
  then the sweep lifts it out (h 7-11 mm within 0.4-0.8 s) because MPPI never pulls Franka −x
  (V17c 0..+5 mm vs V17a −7..−10 mm from sweep onset to max y). Its LCS rates Franka +x as toward the
  seat at 86 % of 1719 recorded closed-loop states (median cosine +0.47; V17a +0.10, V12h −0.46).
- **Response-direction gate** (`sim44/diag/gate.py`, `gate_t1.py`): median cosine of the one-step
  whitened response to Franka +x / +y (1 mm) with the direction to the seated goal, over recorded
  closed-loop states; pass if both ≤ 0.3. Over the five sim-tested models it orders the closed-loop
  results (V12h −0.47/−0.14, V17a +0.10/+0.01 pass; V16b +0.43, V17c +0.47, V17b +0.70/+0.52 fail).
  Threshold set after seeing results, n = 5. Offline prediction error and zero-input drift did not
  predict closed loop (V17c beat V17a on both).
- Over all 17 T1 PointNet models the gate varies with the seed, not only the recipe: E0 2/3 pass
  (s4, s6; s5 fails F+x +0.34), E1 aug 1/3, E2 aug + head λ 0.1 0/3, latent 32 + head 0/3, latent 32
  aug 1/1. Augmentation and the action head make a +x / +y "toward the seat" geometry more likely.
- V12-45 (queued after V12-44): validates the gate within one recipe: V17d = E0 seed 6 (pass) vs
  V17e = E0 seed 5 (fail), both with `stage_start: progress`, hard suite 11 × 2. Then T0 / T2 trainings.

### V12-43 to V12-45 readout (2026-10-08 05:25)

Hard suite (11 starts × 2, no demo prior, MPPI H 24, data-support box, models trained on T1 = 3 demos
+ 12 min of MILES only):

| model | recipe / seed | response gate | V12-43 (static start) | with `stage_start: progress` |
|---|---|---|---|---|
| V17a | E0 seed 4 | pass | 8/22 | 7/22 (V12-44) |
| V17d | E0 seed 6 | pass | - | **11/22** (V12-45) |
| V17e | E0 seed 5 | fail (F+x +0.34) | - | 9/22 (V12-45) |
| V17b | E2 aug + head λ 0.1, seed 4 | fail | 1/22 | - |
| V17c | E1 aug, seed 4 | fail | - | 0/22 (V12-44) |
| V12h (V12-37) | 20 demos + OU, 2 loop iterations | pass | 7/22 | - |

- **3 demos + 12 min of MILES are enough** for the no-prior hard-suite level of the 20-demo model
  (7-11/22 vs 7/22): the real-data cost of a working seed is small.
- **Offline prediction error does not choose the controller model.** Augmentation and the action
  head cut off-demo error 10-15 % but give 0-1/22: they change the LCS's action-response geometry
  (Franka +x / +y "toward the seat" at 85-100 % of states), and MPPI exploits it.
- **Seed variance is large within one recipe** (7, 9, 11/22): the same data and recipe give different
  action geometries. The response-direction gate flags the broken recipes but does not rank good
  seeds. Practical rule for now: train a few seeds, reject gate failures, pick by a short closed-loop
  check (on hardware: a handful of rollouts per candidate), then run the Jin & Posa loop from it.
- **Unsolved starts:** every grasp-angle start (±20°, ±30°) fails for every no-prior model (over at
  −20/−30°, under at +20/+30°); the T1 demos have no such grasp, so these are out of the model's data.
  This is where on-policy data (the J&P loop) has to add coverage.
- `stage_start: progress` removes the early high sweep (mode 1) without a net gain for V17a; kept on
  for V12-45. Point-MAE (ShapeNet) is dropped at this data scale.

### V12-46 readout (2026-10-08 09:20, `cmp/v12_46/learning_curve.md`)

| iter | model | collect start | collect | eval /16 |
|---|---|---|---|---|
| 0 | V18a = V17d | set2 gv_00 | 5/5 | 8 |
| 1 | V18b | set2 gv_06 | 5/5 | **11** |
| 2 | V18c | set3 gv_02 | 5/5 | - |
| 3 | V18d | set3 gv_03 | 4/5 | **11** |
| 4 | V18e | UR −30 mm | 3/5 | - |
| 5 | V18f | UR +15 mm | 5/5 | 5 |
| 6 | V18g | grasp −30° | 0/5 | - |
| 7 | V18h | grasp −20° | 0/5 | 10 |
| 8 | V18i | grasp +20° | 1/5 | - |
| 9 | V18j | grasp +30° | 2/5 | 5 |

- **On-policy refits help when the new data is near what the model can already do:** one refit on
  5 rollouts from a start the model already seats lifts the eval from 8 to 11/16 (nominal and UR
  −30 mm seat for the first time without a prior) and it holds at iteration 3.
- **Far-off failures hurt the shared model:** after rollouts from starts it cannot solve (UR −30 mm
  slants, grasp angles over), the from-seed LCS-only refit degrades the solved starts (5/16 at
  iterations 5 and 9, mostly slanted). Same pattern as V12-36. Collecting on the grasp angles
  gave the first grasp-angle seats in collection (+20° 1/5, +30° 2/5), but not in eval.
- Evals are one episode per start, no action noise: ±2 is within noise; 11 vs 5 is not.
- Next to try: (1) keep the best model for deployment and evaluation (the loop deploys the latest),
  while still training on every rollout (user rule: never discard data); (2) curriculum over the
  rotation: stay on a start until it seats before moving on, so failures far from the model's
  competence do not dominate a refit; (3) per-start or per-region weighting of failures (the
  V12-36 off-track weight applies only after a tube exit); (4) joint (encoder) refits from T2-sized
  data: the T2 learning curve (4.2 mm off-demo) shows the encoder improves a lot with on-policy data,
  while LCS-only refits keep the T1 encoder.

### LCS loss audit: torch_pgd_fixed vs the L4DC violation loss (2026-10-08, offline; `encoder_study/lcs_loss_cmp/`)

User question: with CasADi's original violation-based loss (Jin et al. L4DC 2022, eq. 9) as ground truth, how good is
the `torch_pgd_fixed` loss every V11-V18 model was trained with? Measured on V17d (n_λ 8) with 3000 tuples each from
T1 and from its own hard-suite rollouts (sim45), plus a synthetic stress θ (V17d with c shifted so 1-7 constraints
are active, real mode switches).
- **V17d uses a single contact mode everywhere:** exactly λ₁ is active on all 6000 tuples (λ₁ ≈ 0.40 ± 0.04, smallest
  inactive slack median 1.4): the complementarity does no switching, so the model is effectively affine. Here
  torch and eq. 9 agree to machine precision (λ error 1.5e-7; gradient cosine 1.000 on every block, torch = 0.2 ×
  eq. 9 from MSE vs ½‖·‖²).
- **With real switching (stress θ) torch_pgd_fixed is wrong, and the cause is the dropped skew part:** PGD solves
  LCP(F_pgd) while the slack uses F_lcp = F_pgd + J − Jᵀ (‖J−Jᵀ‖/‖F_pgd‖ = 0.61): λ off 40-55 %, 8-12 % mode
  agreement, predictions shift 10-18× the true step, gradient cosine 0.59 vs eq. 9 with J pushed the opposite way and
  E/H/G/J/c gradients 33-37× too large. 25 vs 500 PGD iterations changes nothing (cosine 0.592 → 0.588); the torch
  loss form with exact-LCP λ matches eq. 9 (cosine 1.000). Hypothesis (untested): the mismatch penalty is smallest
  in single-mode solutions, pushing training toward degenerate models like V17d.
- Side finding: V17d's one-step latent residual exceeds the per-step latent motion (median ratio 1.13 T1, 1.90 sim45):
  in latent space its one-step prediction is worse than "no change", although decoded clouds stay within ~5 mm
  (the λ term is a near-constant offset 17-20× the step).
- MPPI also uses F_pgd (consistent with training), C3+ uses F_lcp (inconsistent).
- Fix options: drop J (symmetric F: PGD = exact LCP, training = MPPI = C3+), or solve the full LCP exactly
  (n_λ 8: 256 active sets) with an active-set / implicit gradient, or train with eq. 9 (convex inner QP, envelope
  gradient). Report the number of distinct active sets as a standard model metric.

### Diagnosis: early prediction error in V21c set2 gv_00 repeat 1 ("over") (2026-10-08 23:40, offline; scripts `encoder_study/diag_gv00/`)

Viewer frames are 20 ms, so frames 200-300 = 4-6 s. Comparison run: V21c nominal repeat 0 (engaged).

| t (s) | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Franka x, pulley frame (mm) | 58 | 85 | 136 | 143 | 142 | 143 | 145 |
| belt h (mm) (nominal) | 73 (72) | 58 (37) | 52 (19) | 50 (13) | 49 (8) | 48 (5) | 48 (3) |
| belt cloud NN to training (mm) (nominal) | 9.9 (5.2) | 11.2 (8.8) | 19.5 (9.9) | 22.8 (8.8) | 27.1 (6.9) | 28.0 (7.7) | 28.1 (8.1) |
| latent NN, whitened (nominal) | 0.63 (0.18) | 1.04 (0.52) | 2.03 (0.51) | 2.20 (0.86) | 2.76 (0.49) | 2.77 (0.60) | 2.75 (0.69) |
| decoded vs GT belt (mm) (nominal) | 7.2 (3.2) | 9.9 (5.5) | 18.0 (6.5) | 25.2 (3.9) | 30.3 (4.2) | 32.1 (4.3) | 32.4 (4.6) |
| k7 error / no change / zero action (mm) | 6.5 / 6.3 / 6.3 | 11.6 / 9.2 / 11.7 | 12.1 / 3.5 / 11.7 | 6.9 / 6.0 / 7.1 | 6.4 / 2.0 / 6.8 | 7.3 / 1.8 / 7.7 | 4.0 / 8.1 / 4.8 |

Training reference: cloud NN between episodes median 5.5 mm (p90 7.2); latent NN spacing 0.17. Segmentation is fine (segmented vs GT 3.4 mm, as nominal). The reset jitter (dx -6, dy 7.5 mm, yaw 4.9 deg) starts the EEs 11-13 mm from the nominal start. In 0-2 s MPPI drives the Franka +85 mm along pulley x with the UR held high (z 53 vs 12 nominal at 3 s); the belt stays up (h ~50 mm). That joint state is outside the training data (the EE support box is per-axis, so each coordinate is still inside it), the encoder extrapolates (decoded belt 25-32 mm from the real one), and the LCS there is pure drift: the k7 error equals the zero-action rollout error and exceeds "no change". MPPI exploited the model on the way out: in 0-2 s the plans promised a goal-distance decrease of 0.14-0.25 per 7 steps and got an increase of 0.32-0.80 (nominal: promised 0.55, got 1.90). From 3 s the planner finds no downhill (promised < 0), barely moves (0.7 mm/step) and every stage times out. Screen over all 44 V21c / V22a hard-suite episodes (`diag_gv00/ood_screen.py`, max whitened NN distance of the encoded latent to V21c's training latents in the first 10 s): engaged 0.94-1.58; the non-angle "over" runs split into out-of-data (set2 gv_00 r1 3.15, set3 gv_02 r1 2.74 / 2.86) and in-data (set2 gv_06 1.25-1.40); every grasp-angle start is out of the data from t = 0 (up−20 1.6-1.9, up+20 3.0-3.2, up+30 3.2-3.4, up−30 3.9-4.0). A latent-support MPPI cost (`MppiParams.latent_support`, opt-in; `cmp/latent_support/README.md`) does not hold the state in the data on a CPU replay of this run: the LCS's predicted latents stay near the data while the real ones leave it, and where the term acts it speeds the +x drive. EE positions are propagated exactly, and their 6-D NN distance to the training EE positions (pulley frame) separates the out-of-data failures: engaged 17-34 mm, set2 gv_00 r1 61, set3 gv_02 r1 40, up−30 70 (set2 gv_06 33 either way); an `ee_support` term is being added. Remedy candidates: a data-support cost on the rollout latents (kNN distance to training latents, real-available) in place of or beside the per-axis EE box; on-policy data from these starts (the V12-57 loop collects them).

### Diagnosis: viewer plans vs recorded belt in V26b up+30 r0 (2026-10-09, offline; `encoder_study/diag_v26b/`)

Viewer: the planned belt (decoded MPPI knots) misses the recorded belt, while the LCS-accuracy chart reports < 2 mm (k 1) / 3.6 mm (k 7). No bug found:
the recorded plans are exactly the LCS rollouts of the recorded inputs (DiagLcs vs `x_sol` knot 7: 0.000 whitened), knots are 75 ms apart like the solves, solve and episode utimes match exactly, and the
world transform is not involved (the same errors in the pulley frame). The chart compares a decoded prediction with the decoded encoding of the later observation, so decoder error cancels; against the recorded belt:

| median over the run (mm) | |
|---|---|
| decoded knot 0 vs recorded belt (decoder error now) | 13.3 |
| planned end (knot 7) vs recorded belt 7 steps later | 14.6 |
| planned end vs decoded observation 7 steps later (= chart k 7) | 3.65 |
| recorded belt motion over 7 steps | 0.96 |

The 13 mm is the JEPA latent + post-hoc decoder: decoded vs the observed segmented cloud is 2.7-2.8 mm on training episodes (seed demo, up+25 demo) but 9.7 (V26b nominal) and 13.6 (this run) on closed-loop episodes; along this run it is 3.3 mm at t 0, 11-17 mm from 0.8 s on.
In the seated phase the latent is inside the data (whitened NN 0.66-0.87) yet the training cloud of the nearest latent is 13-14 mm from the observed cloud: the encoder maps these belts onto training states that differ by ~13 mm (JEPA does not ask the latent to keep that detail), so the decoder draws the training shape.
The chart therefore measures the LCS's self-consistency in its own latent, not the accuracy of the drawn belts; the drawn planned / target belts are only as good as the decoder (here ~13 mm off).

## 8. Conclusions

### Night of 2026-10-08/09 (V12-57 to V12-61)

- **Best model: V26b 16/22** (V22a's JEPA + graph recipe + goal margin W 0.1 / M 0.3, seed 6, on T1ga_f = T1 + 4 grasp-angle demos (UR tilt ≈ −1.0 to −1.3 × the grasp angle) + their MILES + 39 failed operator attempts): 12/14 on the non-angle starts and the first positive-grasp-angle seats (up+20, up+30 2/2 each). Seed 4 of the same recipe: 0/22.
- **Recurring collapse** (T2 1/22, V22b 2/22, V24a/b 0-2/22): adding data moves where the latent puts MPPI-seated belts, and "over" finals can end up nearer the seated goal than truly seated ones; nothing in training separated them. The label-aware goal margin (human success label, real-available) is the first fix that helps: T2 1 → 6/22 (seed 4), T1ga 0-2 → 16/22 (seed 6). At W 1 / M 1.0 it collapses the latent (rank 1).
- **Seed variance is the open problem**: 16 vs 0 for seeds 6 / 4 of V12-61; offline metrics do not predict it.
- EE data-support cost (V12-59b): no gain (12/22). Latent support cost: wrong direction in replay. Negative grasp angles: still over for every model.
- V12-57 (J&P loop) is paused after iteration 1 (V22b 2/22, kept V22a); its refits need the goal margin and labelled on-policy attempts before resuming.

### Current (2026-10-08)

1. **Near the demos the framework works; on varied starts it does not, once the demo action prior
   is gone.** With the prior: 8/10 on demo grasps (V12-07), 10-11/22 on the hard suite (V12-33).
   Without it (the rule since 2026-10-07): 5-7/22 with MPPI (V12-37), 6/22 with C3+ (V12-40).
2. **The data-support box is the one controller change that holds** (V12-17, 5/8 vs 1/8). Start
   rule, goal set, MPPI settings, longer horizon and a learned cost-to-go trade wins between
   starts (V12-33/34/38/39).
3. **The Jin & Posa loop has not improved the controller.** Warm refits drift (V12-08, V12-12,
   V12-35: 3/5 → 0/5, descent cosine 0.95 → 0.27); from-seed refits with a 70 % anchor are stable
   but flat (V12-36: 4, 3, 6, 3, 1 /16). Logging gate verdicts instead of rejecting (user rule)
   exposes the drift but does not stop it.
4. **MILES** gives cheap, two-signed data around the demos (12 min robot for 9.7k frames, one reset
   per demo) and better off-demo prediction (V12-25), but its seed models control worse (V12-26):
   their seated states are no longer close to the seated goal in the latent.
5. **The encoder is the bottleneck** (V12-41): 2.5-3 mm near the demos, 10-17 mm where the
   controller goes, with one-step error tracking reconstruction error. Next: the encoder study in
   §0 (all real frames + augmentation + action-sensitive objectives; external pretrained
   backbones; no simulation pretraining), scored offline against minutes of real robot time.

### Earlier (2026-10-06 15:10)

1. **The framework works near the demos** (V12-07: 8/10 from pre_place_1, full task) and fails
   mostly on varied grasps (V12-08 7/50, V12-12 1/25).
2. **The main failure is the planner leaving the region the model has data for.** Two forms: the
   Franka +y sweep running ~130 mm past the demo (V12-15: 56/140 failures; on the learned model
   the sweep never stops, the contact in the sim stopped it in the successes), and a sideways
   drift of 25-86 mm in directions the model ignores (V12-17). A data-support position box
   (V12-16) fixes both: **5/8 vs 1/8** on four hard grasps (V12-17).
3. **On-policy refits have not improved the policy yet.** Warm joint refits drift (V12-08: every
   refit accepted, both-down 0 → 69 %) or blow up (V12-12 iterations 2-3); prediction metrics do
   not see it. The behavioural gate (V12-11) catches the drift on the hold-frame +y push and
   rejected its first bad refit in V12-12 iteration 4.
4. Not helping: demo-prefix curriculum (V12-05/06/08), nearest-demo selection (V12-09), the
   overshoot guard (V12-10), the stage-synced prior on top of the box (V12-17).
5. Open: whether J&P-style iterations improve once the planner is kept in the data's support
   (V12-18, running), whether a frozen encoder makes refits stable (V12-13, paused), and why
   set2/gv_06 ends "over" in every arm.

Status of the 12:10 "try next" list: 1 done (V12-12); 2 the prefix desync is moot without a
prefix, and the clock desync was fixed opt-in (V12-14); 3 not done (V12-09 showed the reference
demo is not the bottleneck); 4 partly (behavioural gate, frozen-encoder loop queued); 5 not done
(the seed refit is still not deterministic).

### Readout of V12-05 to V12-08 (draft of 2026-10-06 12:10, kept as written)

**G6: FAIL.** G6 needs >= 2/5 at prefix 0.0 in two consecutive iterations AND eval engaged >= 2/6.
V12-08 had >= 2/5 at 0.0 only in iteration 0 (3/5), then the ladder retreated to 0.5 and stayed there.
Eval reached 2/6 once (iteration 7). But the strongest result of the campaign is in the screening
cells, not in V12-08: **V12-07 (latent waypoints + demo prior, no prefix, the seed demo starts) seated
8/10 from pre_place_1, all strict**, against 1/55 over V11-01 and V11-02.

Learning curve (human-label successes per 5): V11-02 0, 0, 0; V12-05 0, 0, 1; V12-06 0 (crash);
V12-07 4, 4 (DONE); V12-08 3, 0, 0, 0, 1, 0, 1, 1, 0, 1 with eval 1, 1, 1, 2, 1 of 6.

What each lever did:
- **Demo action prior + latent waypoints (rank 2):** the lever that works. It turns the sweep into a
  demo-timed one (+77 mm at 2.3-2.5 s) and bounds it (overshoot +33..+47 -> -22..+5 mm in V12-07).
  Without it (V12-05) the sweep runs to +300 mm. Across all 76 labelled V12 rollouts, an overshoot >
  +50 mm gave 2/43 successes vs 14/33 at <= +50 mm.
- **Reverse curriculum (rank 1):** did not help. From the pre_place_2 latch with only the seated goal,
  the learned model seats 1/15 (V12-05) and with the prior 0/4 (V12-06, belt "over"). With waypoints the
  prefix desynchronises the stage index from the prior's demo clock, so retreating made V12-08 worse.
- **More demos (rank 6):** V12-04 4/8 engaged, added to every seed; it confounds V12-05 vs the V11d smoke
  (V12a seats worse). The seed refit is also not deterministic (same buffer / split / seed gives
  different models per cell), so cell-to-cell differences of 1-2 successes are noise.
- **Warm refit, H 16 (rank 4, V12-01):** refits never got rejected (9/9 in V12-08), but one-step error and
  both-down drift up as failures accumulate (both-down 0 -> 69 %), as in V11-02.
- **Demo-range box (orchestrator):** kept f_y ub at 5.98 mm/step where the trust region fell to
  3.97-4.74. Not isolated, since every cell had it on.
- **Pool widening:** set3 starts (gv_02 / gv_03) failed 0/10 with "over" / "other", while new set2
  starts (gv_06) seated 3/5. The set3 grasp likely needs a different sweep end than the seed demo's
  +77 mm.

What to try next (in order):
1. A V12-08 follow-up at frac 0.0 fixed (`schedule: null`, no retreat), V12-07 config, pool = set2
   first (set3 only after 2 iterations >= 3/5). Run as V12-12, with set2 + set3 from the start.
2. Fix the prefix / waypoint desync (start the node's stage index at the first waypoint at or after
   `--demo-offset-frames`) before any further curriculum use.
3. Per-set goal demos (set3's own demo for waypoints + prior) for set3 starts.
4. Refit drift: a buffer cap that keeps successes / demos weighted, or a fresh refit every k
   iterations (V12-01's fresh-vs-warm test was on a failure-only buffer).
5. Seed the refit's torch RNG (determinism) so cells compare like with like.
