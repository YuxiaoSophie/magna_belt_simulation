# Obstacle-aware graph latent (V10): method and results

Prototype of 2026-10-01/02. Idea (user): two belt states that a straight move cannot connect
because the pulley is in the way (e.g. riding over the flange vs seated) must be far apart in
the latent, so the MPC cost ‖z − z_goal‖² stops rating a belt on the lip as nearly seated.
Day log: `docs/learned-mpc.md` (2026-10-01/02); registry: `docs/models.md` (V10a–V10n).
Everything here is a **prototype outside the lcs_learning trainer** (scripts in
`data/lcs/diag/20261001-temporal-latent/scripts/`, not committed to lcs_learning).

## 1. Pulley geometry (`pulley_sdf.py`)

- The large pulley is axisymmetric: its solid is a (radius r, height h) section read from the
  collision mesh (`large_round_pulley_half.obj`, two mirrored halves): flange r = 51 mm for
  |h| ≥ 3.97 mm, groove narrowing to r = 47 mm at h = 0, height ±5 mm.
- Signed distance (SDF) of a belt point = distance of its (r, h) to that polygon, negative
  inside. The 150 belt points are the centreline; seated belts sit at SDF ≈ 3.3 mm (tube radius).

## 2. Graph (`graph.py`, train-only graph `graph/graph_train.npz` + `_edges.npz`)

- **Nodes:** every 3rd frame of the v3 union train split (09-29 diag cache, same 790
  episodes as V3–V8) + the 4 seated targets: 28,137 nodes, 150 points each in the pulley frame.
- **Edges:**
  1. data edges: consecutive kept frames of one episode (always valid: the sim made the move,
     including stretch-and-snap over the lip);
  2. geometric edges: 15 nearest neighbours (Euclidean on the 450-d point vector), kept only if
     the swept blend is clear (below). 422k candidates, 99.8 % clear.
- **Collision test (`blend_ok`):** X(w) = (1 − w)·A + w·B per point, w at 9 samples;
  m(w) = min over points of SDF; keep iff min_w m(w) ≥ min(m(0), m(1), 2 mm) − 0.3 mm, i.e. the
  blend may not dig deeper into the pulley than the shallower endpoint (contact states may
  slide; a move through the lip is rejected).
- **Edge length:** largest single-point displacement (mm).
- **Distances:** Dijkstra from 1,504 landmarks (1,500 random train nodes + the 4 goals) to
  every node → D (1,504 × 28,137, mm). An island of 2,921 approach frames is unreachable
  (19 % ∞).
- **Example path** (`paths/path_over_flange.{png,gif,html}`, `path_viz.py`,
  `path_plotly.py`): riding over (h +17 mm) → seated: straight move cuts 0.8 mm into the lip;
  shortest legal path 71 mm vs 34 mm direct (×2.1), 11 edges from 8 episodes. From under
  (h −11 mm): 70 mm vs 41 mm (×1.7).

## 3. Encoder φ and losses (`embed.py`)

- φ: (belt − pulley centre) flattened, normalised by train mean/std → MLP
  450 → [512, LayerNorm, GELU] × 3 → 16. No camera cloud, no proprio.
- **L_graph** (batch 4,096 pairs (node s, landmark l), 25 % of landmarks are the 4 goals):

  ```
  d = ‖φ(s) − φ(l)‖,   t = D(l, s) / 50 mm              (1 latent unit = 50 mm of path)
  stress = mean_{t finite} (d − t)² / max(t, 0.1)       (Sammon weighting: near pairs count most)
  far    = mean_{t = ∞}    max(0, cap − d)²             (cap = 1.2 × largest finite t)
  L_graph = stress + 0.1 · far
  ```
  Landmark metric MDS (Isomap with collision-checked edges). Converged stress ≈ 0.0025.
- **Joint dynamics (weight 1.0; V10a):** an LCS (n_λ 8, stiffness 0.1, 10 PGD iterations)
  trained with φ on 1,024 train transitions per step; dyn = mean (‖ẑ′ − z′‖ / mean step)²
  (scale-free), plus the trainer's LCP violation terms / 0.01. **The gradient reaches φ**
  (V4a's violation gradient does not).
- **Optional terms (later variants):** `--recon-weight` (joint decoder, belt RMSE in cm),
  `--hold-weight` (u = 0 keeps z), `--rot-aug 30` (random rotation about the pulley axis,
  the same for both states of a pair / tuple, actions rotated too; D is unchanged since the
  pulley is axisymmetric), `--split K` (two-block latent: graph loss on the first K dims,
  reconstruction on the rest, no recon gradient into the task block).
- Adam 3e-4, cosine, 40k steps (~3 min on the 4090).

## 4. Deployment (`lcs_fit.py`, `export_v10.py`, `build_v10_assets.py`, `run_v10_eval.py`)

- **LCS refit:** φ frozen; the trainer's `LinearComplementaritySystemTorchFixedPGD.loss`
  (ε 0.01, n_λ 8, 25 PGD iterations, 15k steps) on whitened z / normalised u, then converted to
  raw latent units (max error 1e-6). Optional `--hold=w` adds ‖LCS(z, 0) − z‖².
- **Cost:** z_std = σ for every dimension (Q = w_q·I/σ²: plain Euclidean distance, the
  geometry the graph loss trained); σ chosen so the median goal distance of near-pulley eval
  states equals V4a's whitened one, so w_q 1 / w_r 0.3 keep their meaning. No fitted Q.
  Two-block: the shape block's σ is scaled by 1/√κ (κ 0.25).
- **Encoder in the sim:** `deploy.npz` with `encoder_type = belt_mlp`;
  `task_common.latent_encoder.BeltMlpEncoder` (opt-in; old deploys load the PointNet encoder
  unchanged) matches torch to 2e-6, 0.15 ms per encode. The C++ controller is unchanged.
- **Goals:** z_goal = φ(stage belts of V4a's per-target `demo_goals.npz`); per-target LCS yamls
  and params under the worktree `learned_archive/2026-10-01/v10/` (params: lcs_file only).
- Decoder: V10a has none (a report-only decoder was fitted afterwards, `train_decoder.py`);
  the reconstruction variants export their jointly trained decoder.

## 5. Results

**Offline (held-out):**

| check | V4a | V10a |
|---|---|---|
| blocked vs equal-RMSE unblocked: blocked farther (paired, 3,525 pairs) | 0.69 | **0.87** |
| blocked / unblocked distance (median) | 1.11 | **1.54** |
| distance vs steps-to-go on held-out demos (ρ) | - | 0.97 |
| 1-step / 7-step error vs no-motion (whitened) | 0.48 / 0.36 | 0.51 / 0.46 |
| ground truth: blocked states reach seated later in the episode | 0.4 % (vs 29 % unblocked) | |

Data-driven alternative tried first: HILP temporal distance (6 runs) — orders demo states by
steps-to-go (ρ 0.92–0.96) but separates dead-end riding-over states worse than raw belt RMSE
(AUC 0.55–0.63 vs 0.70): the data has few dead-end → seated paths and the belt-only state
cannot see whether the UR holds.

**Closed loop (seat6 = 6 cells, 10 s seating stage unless stated):**

| model | change vs V10a | seat6 strict / engaged | notes |
|---|---|---|---|
| V10a | – | 2/6, 3/6 (repeat 3/6, 4/6) | yaw12: +15° 4/6 engaged (0 strict), −15° 6/6 unsafe (Franka ~130°) |
| V10b | + aug + recon 0.03 + joint hold, LCS hold 10 | 0/6 | yaw12 0/12, safe (rot ≤ 8°), stalls h 14–54 mm |
| V10c / V10d | V10b encoder, LCS no hold / hold 1 | 0/6, 0/6 | riding over |
| V10e / V10f / V10g | + aug / + recon 0.03 / + joint hold (one factor each) | 0/6 each | e, g drag the belt off the pulley; f parks at h ≈ 55 |
| V10h / V10j | two-block, recon 0.03 (j: + aug), κ 0.25 | 0/6, 0/6 | h: 107° spin in one cell; j: h 4–16 mm, best RMSE |
| V10i | seed 1 (exact V10a recipe) | 0/6 | riding over |
| V10k | V10a recipe rerun, seed 0 | 0/6 | GPU nondeterminism: latents differ ≤ 0.16 |
| V10l / V10m / V10n | + recon 0.001 / 0.003 / 0.01 | 0/6 each | m closest (h ≈ 7 mm); l lifts the belt away |

**Special V10a runs:** `urtip6_high` with a 30 s seating stage → strict (wrap 106°, h 0.2,
seated at 16 s and held); 12 belt-under-pulley starts (seated goal only, 30 s) → 0/12, both
arms press down; seat6 with the seated goal only (no pre-place stage, 30 s) → 0/6 strict,
2/6 engaged (`gv_05` seats at 6.8 s then is pushed under).

## 6. What it means

1. The graph latent does what it was designed for offline: blocked states are ranked farther
   from the goal (0.84–0.90 across all V10 encoders vs 0.69).
2. Closed-loop seating is decided by the learned **action response** near the pulley, not by
   the geometry: V10a happened to learn "both arms down lowers the cost" (a coordinated press);
   13 other trainings, including a same-seed rerun, did not. The policy gate (offline 7-step
   constant-input MPC proxy, `policy_gate.py`, results `policy_gate.txt`) is a screen: every
   model below 90 % "both down" failed, V10a scores 100 %, but V10c and V10d also score 100 %
   and fail. The one-step lever gate (`lever_gate.py`) does not predict the closed loop.
   The closed-loop drift gate of `lcs_fit.py` varies between refits of the same encoder
   (V10a: 11.8 and 30.0), so it is not used as a criterion.
3. V10a is a fixed press habit, not a state-aware decision: it presses down from under the
   pulley (0/12) and past the seat with a single goal.
4. Its plans are inconsistent with its own model (plan claims −1.6 goal distance per 7 steps,
   the model rollout of the same inputs +3.8, reality −0.1): C3 at `admm_iter` 2 plus a large
   LCS drift (12× the real step). It works as receding-horizon descent on the cost.
5. The 10 s seating stage cuts off slow seats (V10a `urtip6_high`, V2 `urtip` strict with 30 s).

## 7. Limitations and next steps

- Collision only, not stretch: the graph detours around the lip in ways a closed belt can
  only manage by stretching, so separation is moderate (×1.5).
- Geometry only: nothing teaches which action crosses the barrier; the data has almost no
  coordinated two-arm presses or from-under recoveries → collect them (plan W0.2 / W2.2).
- Coverage: the train data has no approach yaw in (−30°, 0°); without augmentation the −15°
  goals are out of distribution (held-out yaw −10° reconstructs at 72–114 mm).
- Goal leakage: the 4 seat6 targets were graph landmarks during training.
- Prototype code: port into lcs_learning (graph-distance term, belt-only encoder, violation
  gradient to the encoder) before further comparisons.

## 8. Files

| what | where |
|---|---|
| scripts | `data/lcs/diag/20261001-temporal-latent/scripts/` (`pulley_sdf, graph, embed, hilp, lcs_fit, export_v10, train_decoder, build_v10_assets, run_v10_eval, eval_sep, compare_blocked, lever_gate, policy_gate, plan_check, cost_check, path_viz, path_plotly`) |
| graph, encoder runs, paths | `…/20261001-temporal-latent/{graph, runs/<tag>, paths}` |
| deploys | `~/git/lcs_learning/outputs/sim_belt_v10_20261001/deploy_v10{a..n}` |
| goals, evals | `data/lcs/mpc_eval/20261001-v10/{goals/<id>, 20261001_<ID>_*}` |
| worktree yamls | `~/git/magna-deploy-learned-lcs/systems/parameters/learned_archive/2026-10-01/{v10, v2_long}` |
| sim encoder | `src/task_common/latent_encoder.py` (`BeltMlpEncoder`, opt-in via `encoder_type`) |
