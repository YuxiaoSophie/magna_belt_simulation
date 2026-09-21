# Belt Simulation Notes

This project develops a Newton/Warp belt simulation.

## Expected folder structure

```text
my_projects/
├── round_belt_task_simulation.py   # entry point: parser, logging, run
├── round_belt_lcm_simulation.py    # LCM entry point: speaks magna's contract (docs/lcm-simulation.md)
├── round_belt_task/                # the round-belt task (built from directives)
│   ├── constants.py                #   typed view over round_belt_scene.yaml + round_belt_lcm_sim.yaml
│   ├── directives.py               #   custom directives (belt rod only; tabletop/ground are
│   │                                #   task-agnostic, see task_common/directives.py)
│   ├── scene.py                    #   build_scene: load directives -> SceneInfo
│   ├── joint_state.py              #   seeds joint values on the finalized Model
│   ├── simulation.py               #   RoundBeltTaskSimulation: solver, stepping
│   └── lcm_simulation.py           #   RoundBeltLcmSimulation: belt trigger over LCM
├── task_common/                    # task-agnostic scene/joint/simulation scaffolding, RGBD cameras
│   ├── directives.py               #   task-agnostic custom directives (tabletop, ground)
│   ├── lcm_contract.py             #   the Drake magna_simulation LCM contract (channels, messages)
│   ├── lcm_bridge.py               #   non-blocking LCM I/O (latest-message inputs, publishes)
│   ├── lcm_simulation.py           #   LcmBeltTaskSimulation: task-agnostic LCM control-step loop
│   └── belt_mesh_lcm.py            #   optional DRAKE_VIEWER_DEFORMABLE tube-mesh publish
│
├── utils/                          # task-agnostic Newton helpers
│   ├── directives/                 #   Drake-style scene directives loader
│   └── labels.py meshes.py transforms.py urdf.py viewer_patches.py
│
├── assets/
│   ├── README.md                   # provenance + edits for every vendored asset
│   ├── common/                     # shared robots and fixtures
│   │   ├── directives/ur10_2f85.yaml
│   │   ├── franka/                 #   arm + long-finger hand (URDF + meshes)
│   │   ├── ur10/                   #   URDF: kinematic source of truth for the pose check
│   │   ├── robotiq_2f85/           #   2F-85 STLs (meshdir of 2f85.xml) + fingers/
│   │   ├── belt_chain_holder/
│   │   └── table/  franka_mount/  scene.urdf
│   └── round_belt_task/
│       ├── round_belt_scene.yaml   # THE SCENE (schema: docs/scene-directives.md)
│       ├── round_belt_lcm_sim.yaml # LCM sim params: solver, drive gains, belt trigger (docs/lcm-simulation.md)
│       ├── round_belt_task_board.urdf  # board + two free-spinning pulley axles
│       └── round_belt_task_board/  #   board + small/large pulleys
│
├── lcmtypes/                       # vendored .lcm sources (dairlib/drake/robotiq), byte-identical to magna
├── dairlib/  drake/  robotiq/      # generated Python LCM types (scripts/gen_lcmtypes.sh), checked in
├── procman/                        # newton_assembly_sim.pmd / _hw.pmd + run_in_magna.sh / run_newton_sim.sh wrappers
│
├── scripts/
│   ├── check_round_belt_task_poses.py   # independent-FK pose check vs the Drake yaml
│   ├── check_scene_directives.py        # directives loader / scene checks
│   ├── check_scene_cameras.py           # ZED intrinsics/extrinsics + cropped point cloud
│   ├── check_lcmtypes.py                # vendored LCM types vs magna's generated modules
│   ├── check_lcm_contract.py            # 12 checks against a live LCM sim (docs/lcm-simulation.md)
│   ├── check_pulleys.py                 # the task-board pulleys spin freely on fixed axles
│   ├── check_robotiq_width.py           # Robotiq command byte -> jaw width calibration
│   ├── gen_lcmtypes.sh                  # regenerates dairlib/ drake/ robotiq/ from lcmtypes/*/*.lcm
│   ├── lcm_peer_utils.py                # shared LCM peer tooling for the checks
│   ├── data/                            # reference data for the checks
│   └── debug/                           # tuning and analysis tools, not checks:
│       ├── bench_lcm_sim_settings.py    #   measures/picks the LCM sim's solver settings
│       ├── summarize_e2e_logs.py        #   summarizes a sim log + controller log from an E2E run
│       ├── check_timing_belt_behaviour.py  # timing-belt model spike (belt.py / belt_strip.py)
│       └── view_timing_belt.py          #   views that spike's scenes
│
├── docs/
│   ├── scene-directives.md         # directive schema + how to add a task
│   └── lcm-simulation.md           # the LCM contract, running, CLI, tuning, divergences
├── 2f85.xml                        # Robotiq 2F-85 MJCF (shared with the other sims)
├── external/newton/                # Newton source (git submodule)
├── task_board_urdf/                # optional submodule; no longer needed by this task
├── round_belt.py  round_belt_two_arms.py  round_belt_command.py  timing_belt.py
└── test/
```

## Run

### Terminal 1: Start the VirtualGL client

```bash
/opt/VirtualGL/bin/vglclient
```

### Terminal 2: Run the full simulation

```bash
vglrun -d :1 uv run python code.py
```

---

## Files

### 1. Main files

#### `round_belt.py`

##### Current setup

This is the main full-scene simulation.

The setup includes everything in `round_belt_old.py`, along with additional features:

* Round belt and task environment

* UR10 and Robotiq 2F-85

* SpaceMouse control

  * enables Cartesian teleoperation of the UR10 end effector using the SpaceMouse
  * reads the target position, orientation, and gripper command from the shared-memory target buffer

* Recording and replay

  * supports recording teleoperation episodes with `--record-episode`
  * supports selecting the recording directory with `--record-dir`
  * supports naming an episode with `--record-name`
  * saves the initial simulation state, per-frame control commands, and Newton state snapshots
  * supports replaying a recorded episode with `--replay-episode`

##### Previous issues

* The IK solver could select different valid robot configurations while following the same end-effector target, which could cause inconsistent or unnecessary joint motion during SpaceMouse teleoperation.

  * A null-space posture preference is used to bias the IK solution toward the previous robot posture while still satisfying the end-effector position and orientation target.

* The full coupled simulation was too slow when using the more expensive coupling solver for every frame.

  * CUDA graph capture is used for the fast simulation path to reduce repeated GPU launch overhead.
  * ADMM coupling is used during contact-critical stages to provide stronger and more stable robot-belt coupling.

---

#### `round_belt_two_arms.py`

##### Current setup

Building on `round_belt.py`, this setup adds another UR10 and Robotiq 2F-85.

---

#### `round_belt_task_simulation.py`

##### Current setup

This is a Newton port of the Drake round-belt scene
(`magna/models/round_belt_task/round-belt-scene.dmd.yaml`). The file itself is a
~50-line entry point; the scene lives in the `round_belt_task/` package
(`constants.py` — a typed view over the scene YAML; `scene.py` — assembly into a
`ModelBuilder`; `joint_state.py` — seeding the finalized `Model`; `simulation.py` —
the runnable `RoundBeltTaskSimulation`). It is standalone in the sense that it does
not build on
`round_belt.py`, it only reuses its helpers and its contact/solver constants.

The scene is authored as data, not code: `assets/round_belt_task/round_belt_scene.yaml`
(plus the shared `assets/common/directives/ur10_2f85.yaml` for the UR10 + Robotiq 2F-85) is
a Drake-shaped directives file — `add_model` / `add_weld` / `add_frame` / `add_directives`
with `X_PC` + `!Rpy { deg: ... }` poses transcribed verbatim from the Drake yaml, plus the
Newton-native custom directives: `add_tabletop_collision` and `add_ground_plane`
(task-agnostic, `task_common/directives.py`) and `add_rod_ellipse` (belt-specific,
`round_belt_task/directives.py`). It is loaded onto the
`ModelBuilder` by `utils/directives/`, and `round_belt_task/constants.py` reads its numbers
from the parsed file, so every scene number lives once, in the
YAML. Directive order is load-bearing: it fixes every body and shape index. Schema
reference: `docs/scene-directives.md`.

It reproduces, with the Drake world-frame poses and default joint angles:

* table + Franka mount (`assets/common/scene.urdf`)
* round-belt task board with its two free-spinning pulleys
  (`assets/round_belt_task/round_belt_task_board.urdf`)
* belt chain holder (`assets/common/belt_chain_holder/belt_chain_holder.urdf`)
* Franka Panda arm + long-finger hand (`assets/common/franka/urdf/`)
* UR10 -- the textured NVIDIA `universal_robots_ur10` USD (same asset as
  `round_belt.py`), **not** `assets/common/ur10/ur10.urdf` -- + Robotiq 2F-85 (`2f85.xml`)
* the deformable round belt (48-element closed rod, resting on the holder's slotted outer rim)

Key points:

* **World frame = the Drake world frame verbatim.** The origin is at
  `panda_link0`, Z is up, the table top is at `z = -0.02858` and the floor is at
  `z = -0.81852` (the table visual's own AABB min z), i.e. about 0.82 m below
  the origin. This is deliberately *not* the `round_belt.py` convention
  (`TABLE_TOP_Z = 0.72`).
* **Start pose.** The UR10 uses the `default_joint_positions` from
  `ur.dmd.yaml`. The round-belt scene yaml gives the Franka *no*
  `default_joint_positions`, so the Franka arm/hand start at what the Drake sim
  actually seeds — `q_init_franka` / `q_init_franka_hand` from
  `magna/systems/parameters/round_belt_simulation_params.yaml` — and **not** the
  "ready" pose in `franka.dmd.yaml`, which this scene never loads.
* Drake `!Rpy { deg: [r, p, y] }` maps to `R = Rz(y) · Ry(p) · Rx(r)`, which is
  what `round_belt.quat_from_rpy` computes. `_assert_rpy_convention()` proves
  this numerically at import time; it will raise if anyone changes the
  convention.
* Table / board / holder are added as static world shapes (body `-1`) so the
  VBD half of the coupled solver owns them, exactly as `round_belt.py` does for
  its table and board.
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
  replayed, exactly as `round_belt.py` does. This removes repeated GPU launch
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

##### Run

```bash
# headless self-check
uv run python round_belt_task_simulation.py --viewer null --num-frames 120 --test

# interactive viewer
vglrun -d :1 uv run python round_belt_task_simulation.py

# same physics, no CUDA graph (A/B comparison)
uv run python round_belt_task_simulation.py --viewer null --num-frames 200 --no-cuda-graph
```

The script prints a world-pose table for the key robot frames on startup, which
is the quickest way to confirm the scene matches Drake.

---

#### `round_belt_lcm_simulation.py`

##### Current setup

Same scene as `round_belt_task_simulation.py`, but driven over LCM instead of a viewer loop: it
speaks magna's `magna_simulation` LCM contract, so the unchanged magna round-belt controllers
(`franka_cartesian_osc_controller`, `ur_cartesian_trajectory_controller`,
`run_round_belt_assembly_controller`) can run against this repo's Newton sim in place of Drake.
It replaces magna's own `franka_hand_simulation` and `robotiq_control_simulation`: this sim
drives the Panda hand and the 2F-85 gripper itself from `PANDA_HAND_COMMAND` /
`ROBOTIQ_COMMAND`. Full contract, CLI, tuning numbers and known divergences from Drake:
`docs/lcm-simulation.md`.

##### Run

```bash
# smoke check (publishes on whatever --lcm-url is; use a private group, not the shared default)
uv run python round_belt_lcm_simulation.py --test --lcm-url "udpm://239.255.76.68:7668?ttl=0"

# real-time, default LCM group, ready for the magna controllers
uv run python round_belt_lcm_simulation.py

# hardware-parameter mode: start from the joint positions a hardware run starts from
# (procman/newton_assembly_hw.pmd points the magna binaries at ..._params_hw.yaml)
uv run python round_belt_lcm_simulation.py \
    --initial-state /home/hienbui/git/magna/python/data/generated/hw_initial_state.yaml
```

---

#### `spacemouse.py`

##### Current setup

This is the SpaceMouse input translator used for robot teleoperation.

---

#### `spacemouse_via_socket_sender.py`

##### Current setup

This is the optional SpaceMouse socket sender for running the SpaceMouse on our own computer while the Newton simulation runs on another computer.

---

#### `round_belt_old.py`

##### Current setup

This is the previous full-scene simulation.

The setup includes, in this order:

* Round belt and task environment
  * table
  * task board
  * small pulley
  * large pulley
  * deformable closed-loop round belt
  * real belt dimensions: 248 mm × 168 mm × 6.6 mm
  * 48 rod elements
  * cable radius: 0.0033 m
  * target belt mass: 22 grams

* UR10 and Robotiq 2F-85
  * UR10 robot arm
  * Robotiq 2F-85 gripper
  * MuJoCo-controlled articulated robot model

##### Previous issues

* Some parts appeared on the floor because of coordinate / height mismatch.

* The belt dimension needed to match the real 248 mm × 168 mm × 6.6 mm size.

* High stiffness caused instability: when user force pulled the belt, the rod tried to maintain stiff constraints and could explode.

* `add_rod_graph` builds the cable from an explicit graph topology: nodes, edges, and connection data must be provided manually.

* `add_rod(..., closed=True)` builds the rod from an ordered list of points along one continuous path. For this belt, the geometry is just one closed ellipse, so `closed=True` automatically connects the last segment back to the first and is simpler and less error-prone.

* Use the latest Newton source to reduce cable explosion

  This project now uses:

  ```bash
  git clone https://github.com/newton-physics/newton
  ```

  The cloned Newton source is used because it may include newer solver fixes that are not yet available in the released `pip` package.

  The important improvement is in Newton’s VBD rigid contact behavior for finite-radius objects such as cables. Previously, when a small-radius cable contacted another object while rotating, the normal contact response could act at a rotating surface anchor point. This could inject artificial kinetic energy into the simulation, making the cable suddenly jump, spin, or explode. The newer Newton source improves this contact handling by applying the normal contact response more stably for cable-like objects, reducing non-physical energy gain during contact.

* Use center-of-mass body frames for rod segments

  The rod setup was also changed to use:

  ```python
  body_frame_origin="com"
  ```

  This places each rod segment’s body frame at its center of mass instead of at the segment start point.

* The interaction is two-way: the cable feels the robot through the VBD proxy bodies that carry the gripper’s motion and inertia, while the robot feels the cable because the cable’s contact reaction forces are fed back to the corresponding MuJoCo gripper bodies.

* I am also using standard Newton contact (since the belt remains stable and does not slip away in the simulation, so I have not switched to hydroelastic contact).

---

#### `round_belt_command.py`

##### Current setup

This is the commanded-motion version of the full simulation.

---

### 2. Test files

The test scripts are stored in:

```text
test/
```

See:

```text
test/README.md
```

for a short description of each test setup.