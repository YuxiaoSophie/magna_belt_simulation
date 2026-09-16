# LCM simulation

`round_belt_lcm_simulation.py` speaks magna's `magna_simulation` LCM contract, so the unchanged
magna round-belt controllers can drive this repo's Newton sim in place of Drake. Where this doc
and the code disagree, the code wins — file an issue against this doc, not the code.

## 1. What this is

For the round-belt task, `round_belt_lcm_simulation.py` replaces magna's `magna_simulation`
process: it subscribes the same input channels, publishes the same six state channels at the
same 5 ms cadence, and the magna controllers
(`franka_cartesian_osc_controller --input_mode=1`, `ur_cartesian_trajectory_controller
--ur_input_mode=1`, `run_round_belt_assembly_controller`) run against it unmodified. Two magna
simulation processes are **not** run against it: `franka_hand_simulation` (this sim drives the
Panda fingers itself, see §2/§7) and `robotiq_control_simulation` (this sim is the Robotiq
driver directly). Drake (`magna_simulation`) stays the sim of record for every other task; this
is a Newton-only alternative for the round-belt task.

Entry point: `round_belt_lcm_simulation.py`. Everything it wires up lives in
`task_common/lcm_simulation.py` (`LcmBeltTaskSimulation`, task-agnostic control-step loop),
`round_belt_task/lcm_simulation.py` (`RoundBeltLcmSimulation`, the belt trigger), and
`task_common/lcm_contract.py` / `task_common/lcm_bridge.py` (the wire contract and the non-
blocking LCM I/O).

## 2. Contract table

All channels are `dairlib`/`drake`/`robotiq` LCM types vendored under `lcmtypes/` (see §8,
`scripts/check_lcmtypes.py`). Every output publishes once per 5 ms control step (200 Hz) unless
noted.
`utime = step_index * 5000` (µs) on every message.

| channel | type | dir | names | formula |
|---|---|---|---|---|
| `FRANKA_INPUT` | `lcmt_robot_input` | in | `panda_motor1..7` | applied joint torque `control.joint_f` |
| `FRANKA_STATE` | `lcmt_robot_output` | out | `panda_joint1..7[dot]`, `panda_motor1..7` | position/velocity as simulated; effort = last `FRANKA_INPUT` value, or `-K*v` (`K = [37.5,50,37.5,25,5,3.75,2.5]`) once the input is older than 0.1 s sim time (stale damping) |
| `PANDA_HAND_COMMAND` | `lcmt_schunk_wsg_command` | in | `target_position_mm` | `p = target_position_mm / 2000`; goal `panda_finger_joint1 <- -p`, `panda_finger_joint2 <- +p`, **not** clamped to the joint range; the goal is slewed at `hand_drive.max_speed` (0.1 m/s per finger) with a `kd/ke` velocity lead, never overshooting; `panda_finger_joint2` is coupled `= -panda_finger_joint1` by a Newton joint mimic (as the real hand, so a one-sided belt load cannot push the fingers apart); default 0 mm (closed) before any command; `force` ignored; `\|sim_time - utime/1e6\| > 1.0 s` -> target 0 mm (closed) |
| `PANDA_HAND_STATUS` | `lcmt_schunk_wsg_status` | out | `actual_position_mm`, `actual_speed_mm_per_s`, `actual_force` | width `(-q1 + q2) * 1000`; speed likewise; `actual_force` always 0 (as Drake) |
| `FRANKA_HAND_ROBOT_OUTPUT` | `lcmt_robot_output` | out | `panda_finger_joint1/2[dot]`, `panda_finger_motor1/2` | position/velocity as simulated; effort = the host-evaluated drive law `ke*(target - q) - kd*qd`, clamped to `+-effort_limit` |
| `UR_INPUT_SIM` | `lcmt_robot_input` | in | `shoulder_pan..wrist_3_actuator` | applied joint torque `control.joint_f`; no stale rule (an `LcmSubscriberSystem` holds the last message forever, matching Drake) |
| `UR_STATE_SIM` | `lcmt_robot_output` | out | `shoulder_pan..wrist_3[dot]`, `*_actuator` | position/velocity as simulated; effort = the raw `UR_INPUT_SIM` value (no gravity term, see §7) |
| `ROBOTIQ_COMMAND` | `lcmt_robotiq_command` | in | `position`, `speed`, `force` (bytes) | driver target `q = open + position/255 * (0.8 - open)` on both `*_driver_joint`s via a position PD (ke 400 / kd 80); the driver's upper joint limit is clamped to `gripper_drive.stop` (0.648 rad), a mechanical stop where the pad/fingertip colliders meet, so a command past the stop cannot drive the fingers into each other; the command mapping and the 0.8 rad span used for `ROBOTIQ_STATUS`/`ROBOTIQ_ROBOT_OUTPUT` are unaffected by the stop; `speed`/`force` echoed on `ROBOTIQ_STATUS`, not applied (as Drake) |
| `ROBOTIQ_STATUS` | `lcmt_robotiq_status` | out | `position`, `speed`, `force`, `activation_status`, `gripper_mode`, `goto_status` | `position = round(fraction * 255)` (`fraction` = mean driver opening / 0.795); `speed = round(clip(\|mapped_speed\|/2.0, 0, 1)*255)`; `force` = last commanded byte; the three status bools always `True`; a full close (255) on nothing now stops at the driver's mechanical limit and reads `position` 206 (a real 2F-85 reads ~227 closed empty), not 255 |
| `ROBOTIQ_ROBOT_OUTPUT` | `lcmt_robot_output` | out | `left/right_finger_joint[dot]` | opening fraction mapped onto Drake's `0.04588` m prismatic range; effort always `[0, 0]` |
| `DRAKE_VIEWER_DEFORMABLE` | `lcmt_viewer_link_data` | out | one MESH geom, `string_data="round_belt::round_belt"` | a tube mesh (8 sides/body) around the 48 belt-body centres; only with `--publish-belt-mesh`, every 10th step (20 Hz) |

`FRANKA_HAND_ROBOT_INPUT` (`lcmt_robot_input`) is in magna's contract but is **not** subscribed
here — nothing else publishes it once `franka_hand_simulation` is not run (see §7).

## 3. Running

### With procman

`procman/newton_assembly_sim.pmd` has two roots baked into every `exec` line: this repo's
`procman/` (absolute path) and magna's checkout, via `procman/run_in_magna.sh`, which `cd`s to
`${MAGNA_ROOT:-/home/hienbui/git/magna}` before exec'ing (procman itself never `cd`s, and the
magna binaries resolve `systems/parameters/...` relative to the magna root). Export `MAGNA_ROOT`
first if your magna checkout is elsewhere.

```bash
bot-procman-sheriff -l procman/newton_assembly_sim.pmd
```

then run script `run_newton_round_belt_sim` (starts the Newton sim, waits 15 s for it to come up,
then the OSC controllers, the visualizer and the assembly controller) or `end_experiment` to tear
everything down. `procman/run_newton_sim.sh` is the Newton sim's own wrapper (`cd` to the repo
root, then run the sim with any extra args passed through); the procman config passes it
`--publish-belt-mesh`. This `.pmd` script sequence has only been validated by parsing (procman's
own `sheriff_config.py`), not by running the sheriff; the E2E runs in §5/§9 launched each process
directly from a shell instead.

### Manual order (equivalent, for debugging one process at a time)

1. This repo, sim first (it owns gravity compensation and the default closed-hand/torque-mode
   arm targets the OSC controllers expect to find):
   ```bash
   uv run python round_belt_lcm_simulation.py
   ```
2. In the magna checkout: `ur_control_simulation`, `ur_cartesian_trajectory_controller
   --ur_input_mode=1`, `franka_cartesian_osc_controller --input_mode=1` (all under
   `bazel-bin/systems/...`, see `procman/newton_assembly_sim.pmd` for the exact paths/flags).
3. Optionally, `magna_visualization` (needs the compiled trajectory, see §6/§8).
4. `run_round_belt_assembly_controller --assembly_controller_params=systems/parameters/
   round_belt_controller_params_sim.yaml --lcm_url=udpm://239.255.76.67:7667?ttl=0`.

### Two roots and `MAGNA_ROOT`

Every magna binary in the procman config is wrapped by `procman/run_in_magna.sh`; every path
this repo touches (the Newton sim, `procman/run_newton_sim.sh`) is relative to this repo's own
root. Only `run_round_belt_assembly_controller`'s `exec` line sets `--lcm_url` explicitly in this
procman config, but every magna binary this stack runs — the OSC controllers,
`ur_control_simulation` and `magna_visualization` — defines the same `--lcm_url` flag
(`DEFINE_string(lcm_url, ...)`, verified in the magna source) with the same compiled-in default
(`udpm://239.255.76.67:7667?ttl=0`). Each constructs its `DrakeLcm` as `DrakeLcm(FLAGS_lcm_url)`,
a **non-empty** string, so `LCM_DEFAULT_URL` is never consulted for any of them (Drake's
`DrakeLcm` only falls back to that env var when the URL it's given is empty) — the only way to
move the whole stack onto a non-default URL is to pass `--lcm_url` to every one of them (see the
procman `exec` lines, or the private-URL recipe below for a manual run).

### Private-URL recipe (isolated test runs, does not touch a running magna stack)

- Every magna binary run in this stack exposes `--lcm_url` — `franka_cartesian_osc_controller`,
  `ur_cartesian_trajectory_controller`, `ur_control_simulation`, `magna_visualization` and
  `run_round_belt_assembly_controller` (`franka_hand_simulation` also has the flag but is not run
  in this stack, see §1) — pass it explicitly to every one of them (§3 "Two roots" above: their
  compiled-in default is non-empty, so `LCM_DEFAULT_URL` has no effect on any of them, and
  omitting `--lcm_url` for even one puts it back on the shared default group).
- `export LCM_DEFAULT_URL=<private-url>` as a belt-and-braces extra: it only matters for a
  process invoked with no `--lcm_url` of its own (e.g. the debug group's `lcm-spy`, which is not
  one of magna's gflags binaries).
- The Newton sim: `--lcm-url <private-url>` (`round_belt_lcm_simulation.py`'s own flag).
- Known leak: `run_round_belt_assembly_controller` also publishes a handful of C3 debug channels
  on a hard-coded shared group, independent of `--lcm_url` — a private-URL run is isolated for
  the contract channels but not perfectly silent on that debug group.
- `scripts/check_lcm_contract.py` and `scripts/bench_lcm_sim_settings.py` use their own private
  group `udpm://239.255.76.68:7668?ttl=0` internally; they never touch a running stack.

### Viewer options

`--viewer null` (default) is headless and is what keeps real time at 200 Hz; `--viewer gl` /
`viser` render the scene inside the paced loop, so each render costs real time.
Measured (`--no-realtime`, 2000 steps, private URL; physics alone stays ~3.1 ms/step throughout):
no viewer 324 steps/s (1.62x real time); `viser --render-every 4` (50 Hz render) 130 steps/s
(0.65x, below real time — each viser render costs ~18 ms CPU); `--render-every 10` (20 Hz) 202
steps/s (1.01x, right at the edge); `--render-every 20` (10 Hz, the default) 250 steps/s (1.25x).
The GL viewer's cost was not measured. To watch a run, use `--viewer viser` (the procman config
runs headless; add the flag to `procman/run_newton_sim.sh`'s arguments if wanted). Contrast
with the two other viewers in this stack: `magna_visualization`'s MeshCat shows the controller's
own overlays (C3 plans, targets, keypoints, forces) but draws Drake's simplified two-finger
Robotiq model and the belt only if this sim is also passed `--publish-belt-mesh`; this sim's own
viewer shows the actual simulated 2F-85 linkage and belt rod (plus collision geometry via
`--show-collision`), but no controller overlays. `--publish-belt-mesh` adds a cheap (~0.02 ms/step
measured) `DRAKE_VIEWER_DEFORMABLE` publish at 20 Hz so an
external Drake-protocol viewer can render the belt without running this sim's own viewer.

## 4. CLI reference (`round_belt_lcm_simulation.py --help`)

Task/Newton-inherited flags (`task_common/simulation.py`, Newton's `newton.examples.init`):

| flag | default | what |
|---|---|---|
| `--device DEVICE` | Warp default | override the Warp device |
| `--viewer {gl,usd,rtx,rerun,null,viser}` | `null` | rendering backend |
| `--headless` | off | initialize the OpenGL viewer headless |
| `--test` | off | run `TEST_NUM_STEPS=400` steps then `test_final()` and exit |
| `--quiet` | off | suppress Warp compilation messages |
| `--paused` | off | start the viewer paused |
| `--warp-config KEY=VALUE` | none | override a `warp.config` attribute (repeatable) |
| `--coupled-view NAME` | `combined` | which coupled-solver view to render |
| `--no-cuda-graph` | graph capture on | disable CUDA graph capture (A/B testing only; solver settings unchanged) |
| `--substeps SUBSTEPS` | `2` | physics substeps per 5 ms control step; **must be even** (the captured graph's state ping-pong only lands back on `state_0` for even counts) |
| `--vbd-iterations N` | `10` | VBD solver iterations per substep |
| `--no-cameras` | cameras off (this script overrides the base parser's `cameras=True` default) | do not render the scene's RGBD cameras; redundant here, see `--cameras` below |
| `--show-collision` | off | draw collision geometry in every viewer |

This script's own flags (`round_belt_lcm_simulation.py` / `LcmBeltTaskSimulation.create_parser`),
which also override some of the above defaults (`viewer=null`, `substeps`/`vbd_iterations` from
the YAML, `cameras=False`):

| flag | default | what |
|---|---|---|
| `--lcm-url LCM_URL` | `udpm://239.255.76.67:7667?ttl=0` | LCM provider URL |
| `--lcm-channels-file PATH` | none | a channels-override YAML (subset of the 12 `LcmChannels` fields, e.g. magna's own `/home/hienbui/git/magna/systems/parameters/lcm_channels.yaml`) |
| `--no-realtime` | realtime pacing on | step as fast as possible instead of pacing to wall-clock time |
| `--num-steps N` | `0` (until stopped / viewer closes) | control steps to run |
| `--render-every N` | `20` | render every N control steps (10 Hz; lower values cost real time) |
| `--cameras` | off | render the scene's RGBD cameras |
| `--publish-belt-mesh` | off | publish the 20 Hz `DRAKE_VIEWER_DEFORMABLE` tube mesh (§2, §3) |

`--test` runs 400 steps non-interactively, prints one `[STATS]` line and asserts
`test_final()`; useful as a smoke check independent of any LCM peer. It still publishes on
whatever `--lcm-url` resolves to (the shared default group if omitted), so pass a private URL,
e.g. `--test --lcm-url "udpm://239.255.76.68:7668?ttl=0"`.

## 5. Solver settings and measured performance

`assets/round_belt_task/round_belt_lcm_sim.yaml` `solver:` picks `substeps: 2`,
`vbd_iterations: 10`, chosen by `scripts/bench_lcm_sim_settings.py` on two criteria: a rest bench
(`control_step` timed non-realtime, GPU idle) and a static Franka-grasp hold (contract check 9).
A third criterion — a diagnostic grasp-and-drag test (`panda_joint1` swung ±0.3 rad while the
Franka holds the belt closed) — is kept in the script but is **not** part of the selection: the
belt slips out of the grip under that specific drag at every setting tried (2/5 through 8/20,
including finer substeps, more VBD iterations, and various contact-stiffness probes), so it
cannot discriminate settings; treat its columns (`drag_diagnostic`, `max_tip_gap_mm`,
`perimeter_change_%`, ...) as informational only.

| substeps | vbd_it | ms/step | Hz | realtime headroom | rest_stable | picked |
|---|---|---|---|---|---|---|
| 2 | 5 | 2.27-2.32 | ~440 | 2.2x | True | |
| **2** | **10** | **3.03-3.24** | **~330** | **1.6x** | True | **yes** |
| 2 | 20 | 4.60 | 217 | 1.09x | True | over budget (> 3.5 ms) |
| 4 | 5 | 4.17 | 240 | 1.20x | True | over budget |
| 4 | 10 | 5.62 | 178 | 0.89x | True | over budget, < 200 Hz |

2/10 over 2/5: both meet the 3.5 ms/step budget, but 2/10 gives the VBD solver twice the contact-
convergence margin and is what the Franka-grasp check (§8, check 9) was tuned and verified
against.

Run: `uv run python scripts/bench_lcm_sim_settings.py --settings 2/5,2/10 --write-yaml`.

**End-to-end rate**, sim attached to the full magna controller stack (procman, 2 final 72 s runs
from controller start, after the hand-slew/finger-stop/gripper-stop fixes below): this Newton sim
holds **198.7-199.6 steps/s min, 200.0 mean** (12 `[STATS]` samples per run, max compute
6.25 ms/step); the Drake baseline (`magna_simulation` + `robotiq_control_simulation`) runs at
**70-150 steps/s, mean 113** (realtime 0.35-0.84x, Drake's own sim step is the bottleneck, not
this repo). Franka holds the belt 20-49 s in both runs (width 6.25-6.69 mm), lifts it to
z ≈ 0.10 m and carries it to the UR; the UR 2F-85 then holds it 19.0-32.5 s at 0.4-5.4 mm pad
contact until the commanded partial release (§7). Belt final centroid: (0.4636, 0.0833, 0.0134)
and (0.4524, 0.0723, 0.0230) m across the two runs, vs Drake's (0.430, 0.087, 0.013) m — run-to-
run variation is real, not just noise: the second run's belt caught on a pulley flange (max z
43 mm) rather than settling flat.

## 6. Belt initial condition

The belt starts resting in the holder (`add_rod_ellipse` in
`assets/round_belt_task/round_belt_scene.yaml`); `round_belt_task/lcm_simulation.py` reproduces
magna's own "place the belt into the hand" trigger from `magna_simulation.cc`
(`assets/round_belt_task/round_belt_lcm_sim.yaml` `belt_trigger:`): once
`panda_hand/finger_tip` comes within `tolerance` (0.005 m) of `point` (a fixed world point), the
belt bodies are **rigidly translated** — not re-fit — onto the hand, their velocities zeroed, and
the solver is reset (`reset_body_poses`, which drops and re-captures the CUDA graph after the
next step). Rigid translation, not a shape re-fit: the FEM rod is a few percent stretched at rest
and a re-fit copy would spring back violently the instant contact is live again. The re-capture
stalls the real-time paced loop for one step (~0.3 s), logging one resync warning; harmless.

The translation target is `finger_tip - 0.010 m` along the hand's local +Z axis
(`belt_trigger.grasp_depth`, `assets/round_belt_task/round_belt_lcm_sim.yaml`), i.e. anchored
10 mm inside the tip and centred between the two finger faces (hand-frame X = Y = 0). The
**anchor body** — which belt body lands exactly on that target — is whichever belt body is
already within `belt_trigger.nearest_body_radius` (0.03 m) of the finger tip if one is (the belt
was already placed in the hand), else the belt body with the largest world-Y coordinate (the
"+Y end", matching where the FEM belt naturally sits at the moment magna's trigger point is
reached).

`semi_axes` in `assets/round_belt_task/round_belt_scene.yaml`'s `add_rod_ellipse` is
`[0.0807, 0.1207]` — the Drake mesh's outer XY extent minus the tube radius, i.e. the rod's
**centreline**, corrected from an earlier value that put the rod ~3.3 mm too large and made it
rest on the holder's bottom plate instead of its step.

The trigger point is the exact point magna's own trajectory drives the Franka finger tip toward,
but **the trigger does not fire with the current magna compiled trajectory, in either the Drake
or the Newton sim** (no `Set belt position` / `[BELT] placed` line appears in either baseline
run); the belt is instead already resting in the fingers' reach by the time the controller closes
the hand at the holder. See §8 (check 9) for a synthetic reproduction of the trigger + grasp via
`scripts/lcm_peer_utils.py`.

## 7. Divergences from Drake

| Drake | here | why |
|---|---|---|
| `franka_hand_simulation` runs an external OSC PD over `FRANKA_HAND_ROBOT_INPUT` | this sim drives the Panda fingers itself from `PANDA_HAND_COMMAND` with an implicit MuJoCo position drive (ke 1000 / kd 100, effort limit 20 N, armature 0.5 kg, all in `assets/round_belt_task/round_belt_lcm_sim.yaml` `hand_drive:`); `franka_hand_simulation` is not run and `FRANKA_HAND_ROBOT_INPUT` is not subscribed | a 5 ms-lagged *external* hand loop diverges on finger-belt contact in this sim (width to 168 mm, finger speed 5.19 m/s, belt thrown); an in-sim implicit drive removes the lag instead of tuning around it. The 0.5 kg finger armature is a deliberate deviation from Drake (which has none) — needed to keep the lagged VBD proxy finger-rod contact stable |
| the real hand's finger stroke slews at its own mechanical rate; the two fingers are mechanically coupled (one lead screw) | the finger goal is slewed at `hand_drive.max_speed` (0.1 m/s per finger) with a `kd/ke` velocity lead, and `panda_finger_joint2` is coupled `= -panda_finger_joint1` by a Newton joint mimic; the finger joint stops are critically damped (`hand_drive.limit_ke`/`limit_kd` 1e5/500) | an un-slewed goal hit the rod at 0.55 m/s and yawed the belt off the UR holder step on first contact; soft, uncoupled stops let a squeeze-on-nothing overshoot ~14 mm per finger and let a one-sided belt load slide both fingers sideways — both fixed by matching the real hand's own mechanical limits, not by tuning the contact model |
| `robotiq_control_simulation` runs the 2F-85 control loop | this sim is the Robotiq driver directly (`ROBOTIQ_COMMAND` in, gains ke 400 / kd 80) | one fewer external process; `robotiq_control_simulation` must not be run against this sim |
| published Franka/UR effort = `tau + tau_g` (gravity torque added back for telemetry) | published effort = the torque actually applied (`control.joint_f`), gravity compensation happens inside MuJoCo and is never added back | no round-belt consumer reads efforts except the (unrelated) timing-belt controller; not worth the extra mapping |
| Robotiq `speed`/`force` command bytes are ignored by the physics, only echoed on status | same here | Drake's own behaviour, reproduced verbatim |
| simplified two-finger Robotiq collision model | the full 2F-85 linkage (proxied into the VBD entry alongside the Franka long fingers) | this repo keeps the full gripper mesh; contact is real, not scripted |
| no belt state estimation channels | none here either | out of scope for both sims |
| Drake viewer draws the FEM mesh directly | `DRAKE_VIEWER_DEFORMABLE` here is a tube mesh (8-sided ring per belt body) around the rod centreline, optional via `--publish-belt-mesh` | this sim's belt is a rod, not an FEM volume |
| 2F-85 fully closed on an empty grasp reads status ≈227 (a real gripper's mechanical stop) | fully closed (command 255) on nothing reads status 206: the driver's upper joint limit is clamped to `gripper_drive.stop` (0.648 rad, where the pad/fingertip proxy colliders meet), so the command saturates against that mechanical stop instead of driving the fingers through each other | close but not identical to the real gripper's ≈227 — both are the driver stalled against a physical stop rather than an arbitrary status value; irrelevant unless a controller keys exact logic off the empty-close status value |
| UR 2F-85 holds the belt through the place motion | in the Newton E2E the UR 2F-85 grasps the belt and holds it (pad contact every step, fingertip 0.4-5.4 mm from the belt) until the controller itself commands a partial release (`ROBOTIQ_COMMAND` position 127) | matches Drake's behaviour now that the finger-stop and hand-slew fixes above removed the spurious belt yaw that used to knock the belt off the UR pads within ~2 s; final belt pose still differs slightly from Drake's (§5) |

## 8. Checks and scripts

Every script below runs as `uv run python <script>` (bash for `scripts/gen_lcmtypes.sh`) from
the repo root; the "extra args" column is what follows the script path.

| script | checks | extra args |
|---|---|---|
| `scripts/check_lcmtypes.py` | 8 vendored LCM types: fingerprint match against magna's own generated modules (skipped if that bazel tree is absent) + an encode/decode round trip | none |
| `scripts/check_lcm_contract.py` | 11 checks against a live, non-realtime `RoundBeltLcmSimulation` on a private LCM group: (1) layouts vs `scripts/data/drake_lcm_layouts.json`, (2) utime monotonic, (3) gravity hold + default (closed) hand, (4) Franka torque sign, (5) Franka stale damping, (6) UR torque sign (no stale rule), (7) hand command open/close/squeeze/stale timing, (8) Robotiq round trip, (9) belt trigger + Franka grasp + a 50 mm lift, (10) `LcmChannels.from_yaml` against magna's own `/home/hienbui/git/magna/systems/parameters/lcm_channels.yaml`, (11) belt tube mesh geometry + wire layout | none |
| `scripts/bench_lcm_sim_settings.py` | rest bench (speed + stability) + the diagnostic grasp-and-drag, per `substeps/vbd_iterations` pair; picks and can write the YAML (§5) | `--settings 2/5,2/10 --write-yaml` |
| `scripts/gen_lcmtypes.sh` | not a check — regenerates `dairlib/`, `drake/`, `robotiq/` from `lcmtypes/*/*.lcm` with the venv's `lcm-gen`; idempotent (re-running produces byte-identical output) | none |
| `scripts/summarize_e2e_logs.py` | not a check — parses a sim log + a controller log from an end-to-end run (procman or manual) into rate, phase-marker, `[BELT]`/`[GRASP]` and error-line summaries; works on either sim's log (Drake fields read `n/a`) | `<sim.log> <controller.log>` |

`scripts/lcm_peer_utils.py` is shared tooling (not a script to run directly): a finger-tip IK,
an in-script Franka joint PD + hand-command peer (`StatePeer`, `pd_step`), and `grasp_sequence`
(approach, trigger, close) used by both `scripts/check_lcm_contract.py` and
`scripts/bench_lcm_sim_settings.py`.

**Regenerating the magna trajectory this sim's trigger targets.** The assembly controller refuses
to start without a compiled `python/data/generated/round_belt_pre_mpc_sim.lcmtraj` (gitignored
generated output in the magna checkout). The regeneration command is documented in
`/home/hienbui/git/magna/systems/parameters/round_belt_controller_params_sim.yaml` next to
`pre_mpc_motion:`:

```bash
bazel run //python:compile_bimanual_trajectory -- \
    --waypoints=systems/parameters/round_belt_controller_params_sim.yaml \
    --sequence-key=pre_mpc_motion --segment=pick_and_place \
    --scene=models/round_belt_task/round-belt-scene-planning.dmd.yaml \
    --ignore-collision robotiq_85::left_finger belt_holder::belt_chain_holder_first_half \
    --ignore-collision robotiq_85::right_finger belt_holder::belt_chain_holder_first_half \
    --linear-speed=0.18 --angular-speed=1.125 \
    --output-prefix=python/data/generated/round_belt_pre_mpc_sim
```

Run from the magna checkout; this repo never writes into magna.

## 9. Troubleshooting

- **No messages arrive on either side.** Check the LCM URL matches on every process
  (`--lcm-url`/`--lcm_url`/`LCM_DEFAULT_URL`, §3) and that UDP multicast is routed on loopback —
  LCM's default `udpm://239.255.76.67:7667` needs a multicast route on `lo`
  (`sudo ip route add 239.255.76.67/32 dev lo` or equivalent) on a fresh machine/container.
- **A controller does not start.** Follow the start order in §3: the Newton sim must be running
  first (it is what makes the arms hold their default/gravity-compensated pose that the OSC
  controllers expect to find); the assembly controller additionally needs the compiled trajectory
  (§8).
- **Rate drops below 200 Hz.** Check `nvidia-smi` for other GPU compute processes sharing the
  device; a `--viewer gl`/`viser` window competes with the physics step for the same GPU — the
  `--render-every` below 10 drops the sim below real time with viser, see §3 "Viewer
  options" for the measured table; `--cameras` adds
  RGBD rendering cost. `[STATS]` reports `compute mean`/`max ms` per step — compare against the
  §5 table for the configured `substeps`/`vbd_iterations`.
- **`franka input: stale-damping` appears unexpectedly.** This fires whenever `FRANKA_INPUT` is
  more than 0.1 s (sim time) old — normal for the first ~0.1 s before any controller publishes,
  and expected again if `franka_cartesian_osc_controller` dies or is paused; the arm damps to a
  stop under `-K*v` rather than free-falling or holding the last torque.
