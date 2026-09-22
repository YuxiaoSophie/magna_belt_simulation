# Belt Simulation Notes

This project develops a Newton/Warp belt simulation.

## Expected folder structure

```text
my_projects/
├── src/                            # the Python packages (entry points put src/ on sys.path)
│   ├── round_belt_task/            # the round-belt task (built from directives)
│   │   ├── constants.py            #   typed view over round_belt_scene.yaml + round_belt_lcm_sim.yaml
│   │   ├── directives.py           #   custom directives (belt rod only; tabletop/ground are
│   │   │                           #   task-agnostic, see src/task_common/directives.py)
│   │   ├── scene.py                #   build_scene: load directives -> SceneInfo
│   │   ├── joint_state.py          #   seeds joint values on the finalized Model
│   │   ├── simulation.py           #   RoundBeltTaskSimulation: solver, stepping
│   │   └── lcm_simulation.py       #   RoundBeltLcmSimulation: belt trigger over LCM
│   ├── task_common/                # task-agnostic scene/joint/simulation scaffolding, RGBD cameras
│   │   ├── __init__.py             #   REPO_ROOT; puts lcmtypes/ on sys.path
│   │   ├── defaults.py             #   contact materials, 2F-85 drive gains, solver iterations
│   │   ├── directives.py           #   task-agnostic custom directives (tabletop, ground)
│   │   ├── lcm_contract.py         #   the Drake magna_simulation LCM contract (channels, messages)
│   │   ├── lcm_bridge.py           #   non-blocking LCM I/O (latest-message inputs, publishes)
│   │   ├── lcm_simulation.py       #   LcmBeltTaskSimulation: task-agnostic LCM control-step loop
│   │   ├── belt_mesh_lcm.py        #   optional DRAKE_VIEWER_DEFORMABLE tube-mesh publish
│   │   ├── recording.py            #   RunRecorder / Recording: --record run capture + reader
│   │   ├── replay_app.py           #   ReplayApp: viser run selection, timeline, 3D playback
│   │   ├── replay_metrics.py       #   metrics, derived events, target-pose evaluation
│   │   └── replay_panels.py        #   Plots/Triads panels for the replay app
│   ├── timing_belt_task/           # timing-belt model spike (belt.py, belt_strip.py)
│   └── utils/                      # task-agnostic Newton helpers
│       ├── directives/             #   Drake-style scene directives loader
│       └── labels.py meshes.py transforms.py urdf.py viewer_patches.py
│
├── assets/
│   ├── README.md                   # provenance + edits for every vendored asset
│   ├── common/                     # shared robots and fixtures
│   │   ├── directives/ur10_2f85.yaml
│   │   ├── franka/                 #   arm + long-finger hand (URDF + meshes)
│   │   ├── ur10/                   #   URDF: kinematic source of truth for the pose check
│   │   ├── robotiq_2f85/           #   2F-85 MJCF (2f85.xml, full linkage + ALOHA fingers), STLs, fingers/
│   │   ├── belt_chain_holder/
│   │   └── table/  franka_mount/  scene.urdf
│   └── round_belt_task/
│       ├── round_belt_scene.yaml   # THE SCENE (schema: docs/scene-directives.md)
│       ├── round_belt_lcm_sim.yaml # LCM sim params: solver, drive gains, belt trigger (docs/lcm-simulation.md)
│       ├── round_belt_task_board.urdf  # board + two free-spinning pulley axles
│       └── round_belt_task_board/  #   board + small/large pulleys
│
├── lcmtypes/                       # LCM types: vendored .lcm sources (byte-identical to magna) + the
│   ├── dairlib/  drake/  robotiq/  #   generated Python packages beside them, checked in
│   ├── magna/                      #   lcmt_spatial_pose (UR target pose, docs/lcm-simulation.md)
│   └── gen_lcmtypes.sh             #   regenerates the Python packages in place from the .lcm sources
├── procman/                        # newton_assembly_sim.pmd / _hw.pmd + run_in_magna.sh / run_newton_sim.sh wrappers
│
├── recordings/                     # --record output (gitignored), <timestamp>-<label>/ per run
│
├── scripts/
│   ├── round_belt_lcm_simulation.py     # LCM sim entry point: speaks magna's contract (docs/lcm-simulation.md)
│   ├── replay_viewer.py                 # viser replay of a --record run: scrub, play, plots, triads
│   ├── checks/                          # regression checks; run all before a commit
│   │   ├── check_round_belt_task_poses.py   # independent-FK pose check vs the Drake yaml
│   │   ├── check_scene_directives.py        # directives loader / scene checks
│   │   ├── check_scene_cameras.py           # ZED intrinsics/extrinsics + cropped point cloud
│   │   ├── check_lcmtypes.py                # generated LCM types vs magna's generated modules
│   │   ├── check_lcm_contract.py            # 12 checks against a live LCM sim (docs/lcm-simulation.md)
│   │   ├── check_pulleys.py                 # the task-board pulleys spin freely on fixed axles
│   │   ├── check_robotiq_width.py           # Robotiq command byte -> jaw width calibration
│   │   ├── check_recording.py               # --record output is complete and bounded overhead
│   │   ├── check_replay_viewer.py           # headless check of the replay app (ReplayApp)
│   │   ├── lcm_peer_utils.py                # controller-side LCM peer for the checks (and the bench)
│   │   └── data/                            # reference data for the checks
│   └── debug/                           # tuning and analysis tools, not checks:
│       ├── bench_lcm_sim_settings.py    #   measures/picks the LCM sim's solver settings
│       ├── summarize_e2e_logs.py        #   summarizes a sim log + controller log from an E2E run
│       ├── check_timing_belt_behaviour.py  # timing-belt model spike (belt.py / belt_strip.py)
│       └── view_timing_belt.py          #   views that spike's scenes
│
├── docs/
│   ├── scene-directives.md         # directive schema + how to add a task
│   └── lcm-simulation.md           # the LCM contract, running, CLI, tuning, divergences
├── external/newton/                # Newton source (git submodule)
└── external/task_board_urdf/       # DAIRLab task-board meshes (submodule); not used by the current code
```

## Run

### Terminal 1: Start the VirtualGL client

```bash
/opt/VirtualGL/bin/vglclient
```

### Terminal 2: Run the simulation with a viewer

```bash
vglrun -d :1 uv run python scripts/round_belt_lcm_simulation.py --viewer gl
```

### Record a run

```bash
uv run python scripts/round_belt_lcm_simulation.py --record
```

### Replay a run

```bash
uv run python scripts/replay_viewer.py
```

---

## Files

### 1. Main files

#### The round-belt scene (`src/round_belt_task/`)

##### Current setup

This is a Newton port of the Drake round-belt scene
(`magna/models/round_belt_task/round-belt-scene.dmd.yaml`), in the `src/round_belt_task/`
package (`constants.py` — a typed view over the scene YAML; `scene.py` — assembly into a
`ModelBuilder`; `joint_state.py` — seeding the finalized `Model`; `simulation.py` —
`RoundBeltTaskSimulation`, the scene held at its default configuration, used by
`scripts/checks/check_pulleys.py` and `scripts/checks/check_scene_cameras.py`; `lcm_simulation.py` — the
LCM-driven sim below). Contact materials, 2F-85 drive gains and solver iterations come from
`src/task_common/defaults.py`.

The scene is authored as data, not code: `assets/round_belt_task/round_belt_scene.yaml`
(plus the shared `assets/common/directives/ur10_2f85.yaml` for the UR10 + Robotiq 2F-85) is
a Drake-shaped directives file — `add_model` / `add_weld` / `add_frame` / `add_directives`
with `X_PC` + `!Rpy { deg: ... }` poses transcribed verbatim from the Drake yaml, plus the
Newton-native custom directives: `add_tabletop_collision` and `add_ground_plane`
(task-agnostic, `src/task_common/directives.py`) and `add_rod_ellipse` (belt-specific,
`src/round_belt_task/directives.py`). It is loaded onto the
`ModelBuilder` by `src/utils/directives/`, and `src/round_belt_task/constants.py` reads its numbers
from the parsed file, so every scene number lives once, in the
YAML. Directive order is load-bearing: it fixes every body and shape index. Schema
reference: `docs/scene-directives.md`.

It reproduces, with the Drake world-frame poses and default joint angles:

* table + Franka mount (`assets/common/scene.urdf`)
* round-belt task board with its two free-spinning pulleys
  (`assets/round_belt_task/round_belt_task_board.urdf`)
* belt chain holder (`assets/common/belt_chain_holder/belt_chain_holder.urdf`)
* Franka Panda arm + long-finger hand (`assets/common/franka/urdf/`)
* UR10 -- the textured NVIDIA `universal_robots_ur10` USD, **not**
  `assets/common/ur10/ur10.urdf` -- + Robotiq 2F-85 (`2f85.xml`)
* the deformable round belt (48-element closed rod, resting on the holder's slotted outer rim)

Key points:

* **World frame = the Drake world frame verbatim.** The origin is at
  `panda_link0`, Z is up, the table top is at `z = -0.02858` and the floor is at
  `z = -0.81852` (the table visual's own AABB min z), i.e. about 0.82 m below
  the origin.
* **Start pose.** The UR10 uses the `default_joint_positions` from
  `ur.dmd.yaml`. The round-belt scene yaml gives the Franka *no*
  `default_joint_positions`, so the Franka arm/hand start at what the Drake sim
  actually seeds — `q_init_franka` / `q_init_franka_hand` from
  `magna/systems/parameters/round_belt_simulation_params.yaml` — and **not** the
  "ready" pose in `franka.dmd.yaml`, which this scene never loads.
* Drake `!Rpy { deg: [r, p, y] }` maps to `R = Rz(y) · Ry(p) · Rx(r)`, which is
  what `utils.transforms.quat_from_rpy` computes. `_assert_rpy_convention()` proves
  this numerically at import time; it will raise if anyone changes the
  convention.
* Table / board / holder are added as static world shapes (body `-1`) so the
  VBD half of the coupled solver owns them.
* The Robotiq gripper uses the existing `2f85.xml` MJCF (Newton has no SDF
  importer), welded to the UR10 wrist-3 frame with `Rz(+90°)`.
* **UR10 asset.** The scene builds the UR10 from
  `newton.utils.download_asset("universal_robots_ur10")/usd/ur10_instanceable.usda`.
  The ported Drake `assets/common/ur10/ur10.urdf` is kept on disk as the *kinematic*
  source of truth (the pose check walks it with an independent numpy FK) but is
  no longer built into the scene: its 7 visual glTFs contain **zero images** --
  only `baseColorFactor` greys plus a pale UR blue -- so the arm rendered flat
  grey/white. The USD's root frame is identical to the URDF's `base_link` frame,
  so the Drake weld `X_W_UR10` is unchanged; its `wrist_3_link` *frame* differs
  by `T(0, 0.0922, 0)·Rx(-90°)`, which `X_USDWRIST3_URDFWRIST3` absorbs so the
  gripper, its pads and every pose number stay exactly where they were. See
  `assets/README.md` for the measurements.
* Physics is the repo's MuJoCo + VBD proxy-coupled solver, configured only to
  hold both arms at their default configuration. There is **no** teleop, IK,
  recording, replay or ADMM here.
* **CUDA graph capture (performance only).** The frame is a fixed 10-substep
  sequence of the same kernels, so it is captured once into a CUDA graph and
  replayed. This removes repeated GPU launch
  overhead and nothing else — `sim_substeps`, `sim_dt`, the solver entries and
  every iteration count are unchanged. Measured here: **0.139 s/frame (7.2 FPS)
  before, 0.020 s/frame (49 FPS) after**, sim-only (`--viewer null`, 200 vs 400
  frames differenced to remove the ~5 s startup). Pass `--no-cuda-graph` to run
  the uncaptured path for A/B testing; capture is skipped automatically on CPU
  and falls back with a printed reason if it fails.

##### Assets

The Franka, UR10 and belt-holder models live under `assets/` and were ported
out of the `magna` checkout; see `assets/README.md` for provenance and the
normalisations that were applied (relative mesh paths, robot renames, glTF
up-axis fix on the visual origins), plus why the UR10 URDF is now kinematic
reference only.

The Franka's own textures (9 × 2048² PNG baseColor) **do** import and reach the
viewer intact — on `--viewer viser` the Panda renders its real white-and-black
livery. On `--viewer gl` those textures are multiplied by Newton's rotating
debug palette because the glTF import path leaves `Mesh.color = None`; that is a
Newton bug under `external/newton/`, documented with the exact code path in
`assets/README.md`.

The simulation prints a world-pose table for the key robot frames on startup, which
is the quickest way to confirm the scene matches Drake.

---

#### `scripts/round_belt_lcm_simulation.py`

##### Current setup

The scene above, driven over LCM: it speaks magna's `magna_simulation` LCM contract, so the unchanged magna round-belt controllers
(`franka_cartesian_osc_controller`, `ur_cartesian_trajectory_controller`,
`run_round_belt_assembly_controller`) can run against this repo's Newton sim in place of Drake.
It replaces magna's own `franka_hand_simulation` and `robotiq_control_simulation`: this sim
drives the Panda hand and the 2F-85 gripper itself from `PANDA_HAND_COMMAND` /
`ROBOTIQ_COMMAND`. Full contract, CLI, tuning numbers and known divergences from Drake:
`docs/lcm-simulation.md`.

##### Run

```bash
# smoke check (publishes on whatever --lcm-url is; use a private group, not the shared default)
uv run python scripts/round_belt_lcm_simulation.py --test --lcm-url "udpm://239.255.76.68:7668?ttl=0"

# real-time, default LCM group, ready for the magna controllers
uv run python scripts/round_belt_lcm_simulation.py

# hardware-parameter mode: start from the joint positions a hardware run starts from
# (procman/newton_assembly_hw.pmd points the magna binaries at ..._params_hw.yaml)
uv run python scripts/round_belt_lcm_simulation.py \
    --initial-state /home/hienbui/git/magna/python/data/generated/hw_initial_state.yaml

# record the run for offline replay with scripts/replay_viewer.py (docs/lcm-simulation.md §10)
uv run python scripts/round_belt_lcm_simulation.py --record --record-label my_run
```
