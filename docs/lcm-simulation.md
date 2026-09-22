# LCM simulation

`scripts/round_belt_lcm_simulation.py` speaks magna's `magna_simulation` LCM contract, so the unchanged
magna round-belt controllers can drive this repo's Newton sim in place of Drake. Where this doc
and the code disagree, the code wins — file an issue against this doc, not the code.

## 1. What this is

For the round-belt task, `scripts/round_belt_lcm_simulation.py` replaces magna's `magna_simulation`
process: it subscribes the same input channels, publishes the same six state channels at the
same 5 ms cadence, and the magna controllers
(`franka_cartesian_osc_controller --input_mode=1`, `ur_cartesian_trajectory_controller
--ur_input_mode=1`, `run_round_belt_assembly_controller`) run against it unmodified. Two magna
simulation processes are **not** run against it: `franka_hand_simulation` (this sim drives the
Panda fingers itself, see §2/§7) and `robotiq_control_simulation` (this sim is the Robotiq
driver directly). Drake (`magna_simulation`) stays the sim of record for every other task; this
is a Newton-only alternative for the round-belt task.

Entry point: `scripts/round_belt_lcm_simulation.py`. Everything it wires up lives in
`src/task_common/lcm_simulation.py` (`LcmBeltTaskSimulation`, task-agnostic control-step loop),
`src/round_belt_task/lcm_simulation.py` (`RoundBeltLcmSimulation`, the belt trigger), and
`src/task_common/lcm_contract.py` / `src/task_common/lcm_bridge.py` (the wire contract and the non-
blocking LCM I/O).

## 2. Contract table

All channels are `dairlib`/`drake`/`robotiq`/`magna` LCM types vendored under `lcmtypes/`, `.lcm`
sources and generated Python side by side (`import task_common` puts `lcmtypes/` on `sys.path`;
see §8, `scripts/checks/check_lcmtypes.py`). Every output publishes once per 5 ms control step (200 Hz)
unless noted.
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
| `ROBOTIQ_COMMAND` | `lcmt_robotiq_command` | in | `position`, `speed`, `force` (bytes) | the byte is a **jaw width**, not an angle: target pad gap `69.20 mm * (1 - position/255)`, inverted onto the driver target through `gripper_drive.width_calibration` (see "Robotiq byte to jaw width" below), applied to both `*_driver_joint`s via a position PD (ke 400 / kd 80) whose torque is clamped to `gripper_drive.effort_limit` (2 N m per driver; stalled on the belt this clamped torque is the grip, ~30-45 N per pad) plus a passive `gripper_drive.damping` (1 N m s/rad) that still acts while the torque is clamped; empty close 0 -> 255 reaches the stop in ~0.53 s, open 255 -> 0 in ~0.65 s; the driver's upper joint limit is clamped to `gripper_drive.stop` (0.648 rad), a mechanical stop where the pad/fingertip colliders meet, so a command past the stop cannot drive the fingers into each other; byte 255 keeps the full-close target 0.8 rad (the overdrive past the stop is what makes the grip), so the stop changes the reached angle but not the mapping; `speed`/`force` echoed on `ROBOTIQ_STATUS`, not applied (as Drake; magna's assembly controller sends a constant speed 255 / force 0) |
| `ROBOTIQ_STATUS` | `lcmt_robotiq_status` | out | `position`, `speed`, `force`, `activation_status`, `gripper_mode`, `goto_status` | `position = round(fraction * 255)` (`fraction` = closed fraction of the jaw **width**, `1 - gap(mean driver angle) / 69.20 mm`, on the same scale as the command byte, so a reached command reads back as itself to within a count); `speed = round(clip(\|mapped_speed\|/2.0, 0, 1)*255)`; `force` = last commanded byte; the three status bools always `True`; a full close (255) on nothing stalls on the mechanical stop 0.09 mm short of a shut jaw and reads `position` 255 (a real 2F-85 reads ~227 closed empty) |
| `ROBOTIQ_ROBOT_OUTPUT` | `lcmt_robot_output` | out | `left/right_finger_joint[dot]` | the same width fraction mapped onto Drake's `0.04588` m prismatic range (linear in the byte, as Drake's own prismatic model is); effort always `[0, 0]` |
| `DRAKE_VIEWER_DEFORMABLE` | `lcmt_viewer_link_data` | out | one MESH geom, `string_data="round_belt::round_belt"` | a tube mesh (8 sides/body) around the 48 belt-body centres; only with `--publish-belt-mesh`, every 10th step (20 Hz) |
| `ROUND_BELT_PULLEY_STATE` | `dairlib::lcmt_object_state` | out | `object_name` `nist_board` (Drake's board model instance, `pulley_state.object_name` in `round_belt_lcm_sim.yaml`); `small_round_pulley_joint`, `large_round_pulley_joint` / `...jointdot` | the two free pulley axles' `joint_q` [rad] and `joint_qd` [rad/s], small first, as dairlib `ObjectStateSender` names a 1-dof joint; raw values (a continuous joint may wrap at +-pi); 200 Hz; no magna process consumes it (magna's local `lcm_channels.yaml` has no key for it, so `--lcm-channels-file` keeps this default) |

`FRANKA_HAND_ROBOT_INPUT` (`lcmt_robot_input`) is in magna's contract but is **not** subscribed
here — nothing else publishes it once `franka_hand_simulation` is not run (see §7).

### Robotiq byte to jaw width

Robotiq's 2F-85/2F-140 manual (POSITION REQUEST register) defines the byte as a **width**:
"0x00 — Open, 0xFF — Closed", "Opening / count: 0.4 mm (for 85 mm stroke)", "quasi-linear
between 0 and 255", with 0 and 255 always at the mechanical extremes whatever fingertips are
fitted. The 2F-85 four-bar is *not* linear in the driver angle, so this sim maps

    intended pad gap [m] = open_gap * (1 - position / 255)

and inverts it through `gripper_drive.width_calibration` in
`assets/round_belt_task/round_belt_lcm_sim.yaml`: nine `[driver angle rad, free-air pad gap m]`
points measured on this scene's own jaw (minimum distance between the two pad collision meshes,
0.02 rad sweep). `open_gap` is the first row, so the calibration is anchored on **our** endpoints,
not on the manual's literal 85 mm: our ALOHA fingertips are thicker, and the usable stroke is
**69.20 mm** (byte 0, driver 0.005 rad) down to **0.09 mm** (the `gripper_drive.stop` 0.648 rad,
where the pad colliders meet). The linkage spreads 87 mm/rad at the open end against 121 mm/rad at
the stop, which is the whole reason the calibration exists.

| byte | 0 | 32 | 63 | 96 | 127 | 160 | 191 | 203 | 223 | 255 |
|---|---|---|---|---|---|---|---|---|---|---|
| intended gap [mm] | 69.20 | 60.52 | 52.10 | 43.15 | 34.74 | 25.78 | 17.37 | 14.11 | 8.68 | 0.00 |
| measured gap [mm] | 69.20 | 60.52 | 52.10 | 43.15 | 34.73 | 25.84 | 17.34 | 14.08 | 8.64 | 0.09 |
| driver target [rad] | 0.0050 | 0.0996 | 0.1856 | 0.2728 | 0.3515 | 0.4320 | 0.5044 | 0.5316 | 0.5768 | 0.8000 |
| gap before this fix [mm] | 69.20 | 60.06 | 50.55 | 39.85 | 29.34 | 17.45 | 5.82 | 1.28 | 0.09 | 0.09 |

The last row is the old linear-in-angle map (`q = open + position/255 * 0.795`), which squeezed
everything above ~byte 160 into the last few millimetres of travel: the hardware place sequence's
`ur_gripper_pos_command: 0.03` (byte 191) means a 21 mm opening on the real gripper and got
5.8 mm here, so the UR never released the 6.6 mm belt. Byte 0 and byte 255 are exact endpoints by
construction — 0 is the open value, 255 keeps the full-close target (0.8 rad, well past the
0.648 rad stop) because that overdrive against the stop is what produces the grip force.
`scripts/checks/check_robotiq_width.py` re-measures the jaw and holds the sweep to 1 mm (§8).

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
`--publish-belt-mesh --record --record-label sim` (`newton_assembly_hw.pmd` passes
`--record-label hw` instead), so every procman run is recorded under
`recordings/<timestamp>-sim/` / `-hw/` (§10; the CLI's own default is unrecorded, §4). This
`.pmd` script sequence has only been validated by parsing (procman's
own `sheriff_config.py`), not by running the sheriff; the E2E runs in §5/§9 launched each process
directly from a shell instead.

### Hardware-parameter mode (`procman/newton_assembly_hw.pmd`)

Same stack, but the magna binaries read `systems/parameters/round_belt_controller_params_hw.yaml`
(the parameters the real cell runs, playing `python/data/generated/round_belt_pre_mpc_hw.lcmtraj`)
and this sim starts from the joint positions a hardware run starts from:

```bash
bot-procman-sheriff -l procman/newton_assembly_hw.pmd    # same two scripts as the sim config
```

The only difference in the `.pmd` is the params file in the visualizer / assembly controller and
`--initial-state /home/hienbui/git/magna/python/data/generated/hw_initial_state.yaml` on the sim
(magna's own `procman/assembly_hwd.pmd` also passes `--is_simulation=false` to the visualizer;
that is deliberately *not* copied here — only the parameters change, the plant is still this sim).

`--initial-state` reads magna's `*_initial_state.yaml` (`q_init_franka` 7, `q_init_franka_hand` 2,
`q_init_ur` 6) and replaces the scene's `default_joint_positions` by joint label; every key must
be present with exactly that length or the sim refuses to start. The applied values are logged as
`[INIT]` lines, one per joint that moved, and the `[LCM]` startup line names the file. Without the
flag the scene defaults are used, unchanged.

What the hardware parameters change, and what that does here:

| hw vs sim parameter | effect in this sim |
|---|---|
| `q_init_franka` / `q_init_ur` from `hw_initial_state.yaml` | the arms start at the hardware `start` waypoint: the controller's live pre-MPC target 0 is reached immediately and the compiled playback's joint gate passes with `franka dq=0.002, ur dq=0.002-0.008` (sim mode: `franka dq=0.049` from the scene defaults) |
| Franka pick `z -0.024962` vs `-0.026802` | both pick the rim-resting belt (§6). The hw pick is the shallower of the two: its tube centre sits level with the fingertip plane at the close and 0.45 mm inside once the fingers settle, against 1.8-1.9 mm for the sim parameters. `task_board_position` differs too, but it does **not** move this waypoint: the compiled segment is baked against the planning scene's own board (§8), so only the live pre/post-MPC and MPC targets see it |
| `franka_gripper_pos_command: 0.0` instead of `-0.05` | the hand is commanded to 0 mm width instead of -100 mm (`assembly_controller.cc`: `target_position_mm = command * 2000`). Both close fully, but the in-sim finger drive is a position PD, so the *grip force* differs: 0 mm gives `ke * half-width` ≈ 3.0-4.7 N per finger on the 6.6 mm belt, where -100 mm saturates the 20 N effort limit. See §7 |
| slower motion (`0.08` m/s / `0.5` rad/s, `--joint-velocity-limit=0.4`), looser tolerances, an extra live `start` waypoint, a re-cut place sequence, `ur_gripper_pos_command` 0.03 (release byte 191, not 127), `spring_stiffness` 1300 | the run takes longer and the UR releases at a 17.4 mm jaw instead of 34.7 mm (§2); no change to the sim itself |
| `ur_gripper_speed: 255` / `ur_gripper_force: 0` | nothing — these are identical in both parameter files, and this sim ignores both bytes anyway (§7) |

Not modelled in this mode (it is the *parameters* that are hardware's, not the plant):

- no hardware drivers or bridges (`franka_driver_v4`, `ur_driver`, `drake_robotiq_driver`,
  `franka_bridge_driver_in/out`): this sim is the plant and speaks the same contract (§1, §2).
- the Robotiq `force`/`speed` bytes are still ignored; a real 2F-85 at force 0 grips with its
  ~20 N minimum, here the grip is the fixed `gripper_drive.effort_limit` (§7).
- the Franka hand has no commanded grasp force, only the position drive above (§7).
- `task_board_position` in the hw parameters is the *real* board's calibration; the scene's board
  weld is unchanged, so the *live* hw targets (pre/post-MPC waypoints, MPC states) land ~9.7 mm
  from where the sim parameters put them relative to this scene's board. The compiled
  `pick_and_place` segment is unaffected — it is baked against the planning scene's board, which
  both parameter files share (`board_pose_in_world` in the `.json` sidecar, §8). The scene is
  deliberately not re-welded.
- `belt_trigger` (§2) was derived from the sim parameters, so the `[BELT] placed` teleport does
  not fire in hardware mode (it does not fire in sim mode either with the current belt).

### Manual order (equivalent, for debugging one process at a time)

1. This repo, sim first (it owns gravity compensation and the default closed-hand/torque-mode
   arm targets the OSC controllers expect to find):
   ```bash
   uv run python scripts/round_belt_lcm_simulation.py
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
- The Newton sim: `--lcm-url <private-url>` (`scripts/round_belt_lcm_simulation.py`'s own flag).
- Known leak: `run_round_belt_assembly_controller` also publishes a handful of C3 debug channels
  on a hard-coded shared group, independent of `--lcm_url` — a private-URL run is isolated for
  the contract channels but not perfectly silent on that debug group.
- `scripts/checks/check_lcm_contract.py` and `scripts/debug/bench_lcm_sim_settings.py` use their own private
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

## 4. CLI reference (`scripts/round_belt_lcm_simulation.py --help`)

Task/Newton-inherited flags (`src/task_common/simulation.py`, Newton's `newton.examples.init`):

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

This script's own flags (`scripts/round_belt_lcm_simulation.py` / `LcmBeltTaskSimulation.create_parser`),
which also override some of the above defaults (`viewer=null`, `substeps`/`vbd_iterations` from
the YAML, `cameras=False`):

| flag | default | what |
|---|---|---|
| `--lcm-url LCM_URL` | `udpm://239.255.76.67:7667?ttl=0` | LCM provider URL |
| `--lcm-channels-file PATH` | none | a channels-override YAML (subset of the 16 `LcmChannels` fields, e.g. magna's own `/home/hienbui/git/magna/systems/parameters/lcm_channels.yaml`) |
| `--no-realtime` | realtime pacing on | step as fast as possible instead of pacing to wall-clock time |
| `--num-steps N` | `0` (until stopped / viewer closes) | control steps to run |
| `--render-every N` | `20` | render every N control steps (10 Hz; lower values cost real time) |
| `--cameras` | off | render the scene's RGBD cameras |
| `--publish-belt-mesh` | off | publish the 20 Hz `DRAKE_VIEWER_DEFORMABLE` tube mesh (§2, §3) |
| `--initial-state PATH` | none (the scene's `default_joint_positions`) | a magna `*_initial_state.yaml` (`q_init_franka`, `q_init_franka_hand`, `q_init_ur`) whose joint positions seed the arms and fingers instead (§3 "Hardware-parameter mode") |
| `--record [DIR]` | off (`const` = `recordings/`) | record the run under `DIR/<timestamp>-<label>/` for later replay (§10) |
| `--record-label LABEL` | `round_belt` | label appended to the run directory name |
| `--record-state-every N` | `4` | record body poses every Nth control step (`N=1` = every step) |
| `--record-chunk-steps N` | `2000` | control steps per recording chunk file |

`--test` runs 400 steps non-interactively, prints one `[STATS]` line and asserts
`test_final()`; useful as a smoke check independent of any LCM peer. It still publishes on
whatever `--lcm-url` resolves to (the shared default group if omitted), so pass a private URL,
e.g. `--test --lcm-url "udpm://239.255.76.68:7668?ttl=0"`.

## 5. Solver settings and measured performance

`assets/round_belt_task/round_belt_lcm_sim.yaml` `solver:` picks `substeps: 2`,
`vbd_iterations: 10`, chosen by `scripts/debug/bench_lcm_sim_settings.py` on two criteria: a rest bench
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

Run: `uv run python scripts/debug/bench_lcm_sim_settings.py --settings 2/5,2/10 --write-yaml`.

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
`assets/round_belt_task/round_belt_scene.yaml`); `src/round_belt_task/lcm_simulation.py` reproduces
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

The belt rests on the holder's **outer rim** (top face z -0.01858, the ledge between the
rim's outer edge and the middle tier's wall). The rim is cut by four 4 cm slots (holder-local
|x| < 0.02 on the long sides = the belt's world-X sides, |y| < 0.02 at the ends = its world-Y
ends) whose floor is the base plate top 5 mm lower; the belt spans them. `add_rod_ellipse` in
`assets/round_belt_task/round_belt_scene.yaml` sets the rod **centreline** to `semi_axes`
`[0.08558, 0.12066]`, `center` z -0.0152: the axis-aligned ellipse whose 48 capsules sit
against the middle tier's wall (the second outermost rim), fitted to that tier's collision
(`quarter_ellipse_bottom.obj`, 1.05 mm outside the visual wall): 0.05 mm minimum gap between the
tube and the collision wall, at most 0.37 mm where the wall exists (visually 1.04-1.40 mm). The
wall's outline is boxier than an ellipse, so the gap opens to 3.9 mm at the end
slots. z is rim top + tube radius + the settled contact gap. The rim collision is four
`quarter_ellipse_rim.obj` quarters (see `assets/README.md`), so the slots stay open.

**Stretch stiffness comes from the real belt's datasheet.** The hardware belt is a MISUMI
**MBT6-640** polyurethane round belt: 6 mm diameter, 640 mm loop, JIS A 88, tensile strength
>= 24.5 MPa, elongation at break >= 400%, minimum pulley diameter 50 mm, and for the 6 mm size
**16.6 N produces 5% elongation**. That secant gives a section rigidity EA = 16.6 / 0.05 =
**332 N**. Newton stores `stretch_stiffness` directly as the rod joint's `target_ke` in N/m and
the stretch force is `ke * (l - l0)` per element (Newton's own rigidity-to-stiffness helper
divides a section rigidity by the local dual length to get exactly this), so the loop's
compliance is `n / ke` and matching the material needs `ke = EA / mean(l0)`. The 48 elements are
sampled at uniform ellipse parameter, so they run 11.22-15.77 mm with a **mean rest length of
13.5865 mm** (total 652.15 mm) and `ke = 332 / 0.0135865 = 24436 N/m`, i.e.
`stretch_stiffness: 2.44e+4`. The shipped `2.0e+4` was EA 272 N — 22% too soft.

Verified with a headless axial pull test (a straight 48-element rod with the same `ke`, damping,
bend stiffness, radius and mean rest length, one end pinned, no gravity and no contacts): at
2.44e+4 the measured secant is **EA 325 N** and **16.6 N gives 5.10% elongation** (datasheet
5.00%); at the old 2.0e+4 it was EA 266 N and 6.24%. The ~2% softness below the analytic 331.5 N
is the VBD penalty/augmented-Lagrangian residual, not the model.

`stretch_damping` stays at **1.0e-1**. It is a per-element viscous damper on the elongation rate
[N.s/m], so it sets a stress-relaxation time `kd / ke` of 5.0 us before and 4.1 us after — both
several hundred times shorter than one solver substep — and the loop's first axial mode sits at a
damping ratio of ~3e-4 either way. The datasheet gives no loss modulus, so scaling it with `ke`
would be an invented number with no measurable effect.

Note the modelled loop is the Drake ellipse resized to the holder rim, **652 mm** of centreline
against the real belt's 640 mm; the taut two-pulley path is 649 mm, so the simulated belt has
~3 mm of slack where the real one would have to stretch ~1.4% (about 4.6 N) to fit. The real
board carries a slide tensioner on the small-pulley assembly.

The trigger point is the exact point magna's own trajectory drives the Franka finger tip toward,
but **the trigger does not fire with the current magna compiled trajectory, in either the Drake
or the Newton sim** (no `Set belt position` / `[BELT] placed` line appears in either baseline
run); the belt is instead already resting in the fingers' reach by the time the controller closes
the hand at the holder. See §8 (check 9) for a synthetic reproduction of the trigger + grasp via
`scripts/checks/lcm_peer_utils.py`.

## 7. Divergences from Drake

| Drake | here | why |
|---|---|---|
| `franka_hand_simulation` runs an external OSC PD over `FRANKA_HAND_ROBOT_INPUT` | this sim drives the Panda fingers itself from `PANDA_HAND_COMMAND` with an implicit MuJoCo position drive (ke 1000 / kd 100, effort limit 20 N, armature 0.5 kg, all in `assets/round_belt_task/round_belt_lcm_sim.yaml` `hand_drive:`); `franka_hand_simulation` is not run and `FRANKA_HAND_ROBOT_INPUT` is not subscribed | a 5 ms-lagged *external* hand loop diverges on finger-belt contact in this sim (width to 168 mm, finger speed 5.19 m/s, belt thrown); an in-sim implicit drive removes the lag instead of tuning around it. The 0.5 kg finger armature is a deliberate deviation from Drake (which has none) — needed to keep the lagged VBD proxy finger-rod contact stable |
| the real hand's finger stroke slews at its own mechanical rate; the two fingers are mechanically coupled (one lead screw) | the finger goal is slewed at `hand_drive.max_speed` (0.1 m/s per finger) with a `kd/ke` velocity lead, and `panda_finger_joint2` is coupled `= -panda_finger_joint1` by a Newton joint mimic; the finger joint stops are critically damped (`hand_drive.limit_ke`/`limit_kd` 1e5/500) | an un-slewed goal hit the rod at 0.55 m/s and yawed the belt off the UR holder step on first contact; soft, uncoupled stops let a squeeze-on-nothing overshoot ~14 mm per finger and let a one-sided belt load slide both fingers sideways — both fixed by matching the real hand's own mechanical limits, not by tuning the contact model |
| `robotiq_control_simulation` runs the 2F-85 control loop | this sim is the Robotiq driver directly (`ROBOTIQ_COMMAND` in, gains ke 400 / kd 80) | one fewer external process; `robotiq_control_simulation` must not be run against this sim |
| a real 2F-85 grips with 20-235 N set by the `force` byte and its non-backdrivable drive does not ring on a stalled grip | a fixed grip: the driver PD torque clamps at `gripper_drive.effort_limit` 2 N m (~30-45 N per pad on the belt) whatever the `force` byte, with a passive `gripper_drive.damping` 1 N m s/rad on the drivers | magna sends force 0 (a real gripper's ~20 N minimum), so a byte map would change nothing in this stack. With the old 1 N m clamp and no damping outside it, the stalled PD had no damping: the pads rang at ~2.5 Hz after every close (pad force 0-77 N, pad separation 5 mm p2p) and a belt hanging from the pads ratcheted out in 2.2 s. Damping 1 needs 2 N m to keep the ~0.53 s close |
| the real Panda hand holds with a commanded grasp force, independent of how far the width command is past the object | the grip is whatever the finger position PD produces: `hand_drive.ke * half-width error`, clamped at `effort_limit` 20 N. A `franka_gripper_pos_command` of `-0.05` (sim parameters, -100 mm) therefore saturates at 20 N per finger, while `0.0` (hardware parameters, 0 mm, a plain full close) gives 3.0-4.7 N per finger on the 6.6-9.4 mm belt | the drive is a position drive by design (see the `franka_hand_simulation` row above); measured over a 157 s hardware-mode hold the 3-5 N grip did not slip, so no force floor was added. A grasp-force model would mean commanding effort (not position) once the fingers stall, which the current `PANDA_HAND_COMMAND` path has no input for |
| published Franka/UR effort = `tau + tau_g` (gravity torque added back for telemetry) | published effort = the torque actually applied (`control.joint_f`), gravity compensation happens inside MuJoCo and is never added back | no round-belt consumer reads efforts except the (unrelated) timing-belt controller; not worth the extra mapping |
| Robotiq `speed`/`force` command bytes are ignored by the physics, only echoed on status | same here | Drake's own behaviour, reproduced verbatim |
| simplified two-finger Robotiq collision model | the full 2F-85 linkage (proxied into the VBD entry alongside the Franka long fingers) | this repo keeps the full gripper mesh; contact is real, not scripted |
| no belt state estimation channels | none here either | out of scope for both sims |
| task-board pulleys: the Drake sim on the local magna branch has them fixed (magna `2d9b0ca` makes them revolute) | the pulleys rotate freely: VBD bodies on world revolute axles at their centres (axis = board normal, SDF inertials, axle damping 0.001 N m s/rad from the URDF, applied as a zero-stiffness VBD drive because `SolverVBD` ignores `joint_damping`), in the VBD entry with the belt; each carries a thin marker strip | as `2d9b0ca`, except the damping (magna: 0): undamped, a pulley kicked by the Franka fingers or the belt free-spun for minutes; 0.001 stops it in ~0.2 s (time constant Izz/c = 60 ms) and resists a belt-driven turn with only 1e-3 N m per rad/s. The axle anchor is a hard AL constraint (steady sag ~0.1-0.2 mm under gravity); costs ~0.33 ms/step (two more VBD bodies + the damper row) |
| the holder URDF has no collision for its outer rim (bottom plate and middle/top tiers only); the Drake belt ellipse is 0.168 x 0.248 | four `quarter_ellipse_rim.obj` quarters (z 0.005-0.010, slots left open) and a belt centreline of `[0.08558, 0.12066]` resting on that rim against the middle-tier wall (§6) | user request: the belt starts on the holder's outermost rim |
| Drake viewer draws the FEM mesh directly | `DRAKE_VIEWER_DEFORMABLE` here is a tube mesh (8-sided ring per belt body) around the rod centreline, optional via `--publish-belt-mesh` | this sim's belt is a rod, not an FEM volume |
| 2F-85 fully closed on an empty grasp reads status ≈227 (a real gripper's mechanical stop) | fully closed (command 255) on nothing reads status 255: the driver's upper joint limit is clamped to `gripper_drive.stop` (0.648 rad, where the pad/fingertip proxy colliders meet), which is 0.09 mm of pad gap, i.e. a shut jaw on the width scale the status uses (§2) | a real 2F-85 stalls its fingertips against each other a few counts before its own position command runs out; ours stalls level with it. Irrelevant unless a controller keys exact logic off the empty-close status value |
| Drake's `robotiq_arg85` finger is a prismatic joint the byte drives linearly, so the byte is a width by construction | the full four-bar linkage, with the byte inverted onto the driver angle through a measured `gripper_drive.width_calibration` (§2) | same contract (byte = width, per Robotiq's own manual) reached from a real linkage. Before the calibration the byte drove the *angle* linearly: byte 191 gave a 5.8 mm jaw where the real gripper and Drake give ~21 mm, and the hardware place sequence never released the belt |
| UR 2F-85 holds the belt through the place motion | in the Newton E2E the UR 2F-85 grasps the belt and holds it (pad contact every step, fingertip 0.4-5.4 mm from the belt) until the controller itself commands a partial release (`ROBOTIQ_COMMAND` position 127) | matches Drake's behaviour now that the finger-stop and hand-slew fixes above removed the spurious belt yaw that used to knock the belt off the UR pads within ~2 s, and the driver damping above stopped the pads ringing after the close; final belt pose still differs slightly from Drake's (§5) |

## 8. Checks and scripts

Every script below runs as `uv run python <script>` (bash for `lcmtypes/gen_lcmtypes.sh`) from
the repo root; the "extra args" column is what follows the script path.

| script | checks | extra args |
|---|---|---|
| `scripts/checks/check_lcmtypes.py` | the four packages resolve to `lcmtypes/<pkg>/`; then 14 vendored LCM types: fingerprint match against magna's own generated modules (skipped if that bazel tree is absent) + an encode/decode round trip | none |
| `scripts/checks/check_lcm_contract.py` | 12 checks against a live, non-realtime `RoundBeltLcmSimulation` on a private LCM group: (1) layouts vs `scripts/checks/data/drake_lcm_layouts.json`, (2) utime monotonic, (3) gravity hold + default (closed) hand, (4) Franka torque sign, (5) Franka stale damping, (6) UR torque sign (no stale rule), (7) hand command open/close/squeeze/stale timing, (8) Robotiq round trip, (9) belt trigger + Franka grasp + a 50 mm lift, (10) `LcmChannels.from_yaml` against magna's own `/home/hienbui/git/magna/systems/parameters/lcm_channels.yaml`, (11) belt tube mesh geometry + wire layout, (12) `ROUND_BELT_PULLEY_STATE` once per step with the names, utime and the simulated pulley `joint_q`/`joint_qd` | none |
| `scripts/checks/check_pulleys.py` | the free pulley axles in the position-PD `RoundBeltTaskSimulation`: (T0) 2 VBD bodies on world revolute joints at the board weld composed with the joint origins, axis = board normal, (T1) 2 s zero-torque hold (angle < 0.005 rad, centre drift/sag < 0.5 mm), (T2/T3) 2e-3 N m `joint_f` for 0.5 s drives each pulley to torque/damping (2 rad/s, +-15 %) and the damped-rigid angle (+-20 %) without moving its centre or the other pulley, then after release it spins down with the time constant Izz/damping | none |
| `scripts/checks/check_robotiq_width.py` | the `ROBOTIQ_COMMAND` byte to jaw width map (§2) against a live `RoundBeltLcmSimulation` on a private LCM group: (T0) the 256-entry byte -> driver target table (byte 0 at the open value, 255 at the full-close target, monotone, calibration spanning open to `gripper_drive.stop`), (T1) a 10-byte free-air sweep whose measured pad gap (minimum distance between the two pad collision meshes) is within 1 mm of `open_gap * (1 - byte/255)`, endpoints included, (T2) the `ROBOTIQ_STATUS` position echoes each reached byte to within 3 counts | none |
| `scripts/checks/check_recording.py` | 10 checks (`R0`-`R9`) against a live, non-realtime `RoundBeltLcmSimulation` recording (`--record`) on a private LCM group: build, a 600-step scripted sequence (a hand command republished unchanged then changed — one `hand_command` event per goal, two Robotiq commands, a deliberate belt trigger), the on-disk files, `Recording.load`, events/signals, `Recording.list_runs`, the recording's per-step overhead against an unrecorded baseline, loading an unfinished copy (chunk discovery) and a copy with a chunk removed (`ValueError`), and run-dir name collisions (§10) | none |
| `scripts/checks/check_replay_viewer.py` | 11 checks (`V0`-`V10`) against a headless `ReplayApp` (§10) over two recorded runs (a scripted 600-step run and a second, shorter one): run selection, seeking, playback, metrics, events, plots, triads, refusal of a body-label mismatch, per-seek render timing, and synthetic target messages in both runs (target triads vs. `target_world_pose`, a run switch from past the shorter run's end, a failing hook during a GUI run switch), and a 5 s real-time looped playback's rates (redraws/s near `render_fps`, chart updates <= 6/s per chart, no scale resends) | none |
| `scripts/checks/check_sim_snapshot.py` | 6 checks (`S0`-`S5`) that `task_common.sim_snapshot` restores a `RoundBeltLcmSimulation` state, on a private LCM group: (S0) build + settle, (S1) baseline restore-vs-continue noise, (S2) restore fidelity mid-motion (hand opening, Robotiq closing) in the same sim, (S3) the same restore in a second sim built fresh from the saved `.npz`, (S4) refusal of an edited/truncated snapshot (no side effects), (S5) the CUDA graph is dropped and re-captured and step/time bookkeeping is exact after restore | none |
| `scripts/checks/check_inproc_motion.py` | 6 checks (`M0`-`M5`) of the in-process waypoint motion (no magna, no LCM): (M0) `pre_mpc_motion` waypoints from magna's yaml, world frame, gripper commands, (M1) numpy FK vs the built model's bodies, (M2) IK round trips at every waypoint + the nominal joint trajectory's per-step jump, (M3) the nominal pick holds the belt and is snapshotted, (M4) restore + `pre_place_1 -> place_3` tracking, the grasp held at `pre_place_2` and belt bodies reaching the large pulley, (M5) two restored replays of M4 agree (determinism) | none |
| `scripts/checks/check_lcs_dataset.py` | 4 checks (`F0`-`F3`) of the `.npz` writer/validator (`task_common.lcs_dataset`) on synthetic data, no sim/GPU/LCM: (F0) format constants, (F1) `belt_points_ordered` arc-length resampling, (F2) `EpisodeWriter` -> `validate_episode` round trip incl. the action-delta formula, (F3) copies with a missing/malformed key are rejected naming that key; also validates a magna log if present (INFO) | none |
| `scripts/checks/check_lcs_outcome.py` | 6 checks (`O0`-`O5`) of the large-pulley outcome classifier (`round_belt_task.outcome`) on a synthetic 48-body loop, no sim/LCM: seated/lifted/lowered/tilted/pulled-out belts classify `engaged`/`over`/`under`/`slanted`/`outside`, and labels + wrap angle are invariant to the pulley frame's pose (rotated and spun) and to the episode-label majority-vote API | none |
| `scripts/checks/check_lcs_collector.py` | 8 checks (`C0`-`C7`) end to end against `scripts/collect_lcs_dataset.py` (no magna, no LCM): (C0) a start-state snapshot (reused if its scene digests match, else built), (C1) one collected episode per intent, (C2) per-file `.npz` contents (shapes, joint limits, `utime` spacing), (C3) `index.json`, (C4) `--record` recordings + a headless `ReplayApp`, (C5) a same-seed rerun reproduces the perturbation/outcome/final belt state, (C6) the real `lcs_learning` loader on the written files (skipped if that venv is absent), (C7) the board clearance guard: every C1 episode reports a non-negative `sim_min_board_clearance_mm` and no contact, and a deliberately unsafe UR `dz` of -15 mm is clamped back above `--min-clearance` instead of running into the plate | none |
| `scripts/replay_viewer.py` | not a check — opens the viser replay GUI on a recordings directory (§10) | `--recordings DIR --run NAME --port N --show-collision --device DEV --render-fps FPS --stats` |
| `scripts/lcs/make_start_state.py` | not a check — runs the nominal in-process pick and saves the `pre_place_1` start-state snapshot LCS episodes restore from (`docs/lcs-data-collection.md`) | `--out PATH --record [DIR] --params PATH --arm-ke K --arm-kd K --no-velocity-lead --hold S` |
| `scripts/collect_lcs_dataset.py` | not a check — collects LCS episodes of the round-belt engagement segment into `.npz` + `index.json` (`docs/lcs-data-collection.md`) | `--out DIR --label L --episodes N --seed S --intents LIST --weights LIST --start-state PATH --fresh-pick --record --settle-s S --sample-period S --arm-ke K --arm-kd K --params PATH --no-pcd --dry-run --thresholds KEY=VALUE...` |
| `scripts/debug/bench_lcm_sim_settings.py` | rest bench (speed + stability) + the diagnostic grasp-and-drag, per `substeps/vbd_iterations` pair; picks and can write the YAML (§5) | `--settings 2/5,2/10 --write-yaml` |
| `lcmtypes/gen_lcmtypes.sh` | not a check — regenerates the `dairlib`/`drake`/`robotiq`/`magna` Python packages in place in `lcmtypes/<pkg>/` (beside the `.lcm` sources, which it never deletes) with the venv's `lcm-gen`; idempotent (re-running produces byte-identical output) | none |
| `scripts/debug/summarize_e2e_logs.py` | not a check — parses a sim log + a controller log from an end-to-end run (procman or manual) into rate, phase-marker, `[BELT]`/`[GRASP]` (incl. per-pulley unwrapped rotation, >= 3 deg/sample turning windows and the final pulley drift) and error-line summaries; works on either sim's log (Drake fields read `n/a`) | `<sim.log> <controller.log>` |

`scripts/checks/lcm_peer_utils.py` is shared tooling (not a script to run directly): a finger-tip IK,
an in-script Franka joint PD + hand-command peer (`StatePeer`, `pd_step`), and `grasp_sequence`
(approach, trigger, close) used by both `scripts/checks/check_lcm_contract.py` and
`scripts/debug/bench_lcm_sim_settings.py`.

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

**The sim `pick` waypoint was retuned for the rim-resting belt (magna-side change, 2026-09-17).**
On `hien/speed_up_belt_tasks` the `pick` waypoint's `franka_position` z went
`-0.022802311862471163` -> `-0.026802311862471163` (4.0 mm lower) and the trajectory above was
recompiled. Nothing else in magna changed; the edit is left uncommitted there. With the belt on the
outer rim (§6) the old z put the tube centre 2.1 mm *past* the fingertip plane, so the close swept
the belt out instead of capturing it; 4.0 mm lower puts it 1.8 mm inside the jaw at the close and
the Franka picks, lifts (~125 mm) and carries it. Measured with a scratch `[JAW]` probe on the
Franka finger meshes; the closed fingers keep 4.6 mm of vertical clearance to the holder below
(the belt spans a rim slot there). The hardware parameters and
`round_belt_pre_mpc_hw.lcmtraj` were **not** touched.

**Wrapping the belt onto the pulleys was attempted and does not work yet; magna's place waypoints
are back to the shipped ones (2026-09-17).** Seven place-phase configurations were compiled and run
three times each (22 full-stack E2E runs); none seats the belt reproducibly, so nothing was landed
in magna beyond the `pick` change above. The geometry and the failure mechanism are now measured,
which is the useful part:

- Groove geometry off the board meshes: the large pulley's channel bottoms at 47.0 mm with a
  51.0 mm outer radius and a 4.0 mm half width, so a 3.3 mm belt seats at **50.73 mm** from the
  axis; the small pulley's channel bottoms at 13.5 mm with a 15.0 mm outer radius and a 2.3 mm
  half width — shallower than the belt is thick — so the belt rides its rim corners at
  **17.38 mm**. The pulley discs span only +-4.5 mm axially and both mid-planes sit 21.4 mm above
  the board, so a belt that is not already at the mid-plane slides under them.
- The taut two-pulley path is **649.1 mm** (2 x 212.4 tangent + 175.2 large arc + 49.2 small arc)
  against the rod's 652.15 mm rest polygon.
- **The two grip points are one element out of phase.** The UR holds belt node 46/47 and the Franka
  node 23/24 (both fixed at the `pick`). Node **22** is the exact antipode of node 46 — 326.08 mm
  of rest belt each way against the 324.57 mm taut half-path, i.e. a 1.5 mm slack fit. Node 23 has
  only **310.44 mm** one way and 341.71 mm the other, so the strand the Franka pulls is 14 mm
  short and has to stretch **~4.5%** before it can even reach the small pulley, before any
  over-travel.
- What the runs show, reproducibly: the UR carrying its half to the large pulley's far apex and
  holding it there does work — at `place_11`/`place_12` the belt is **in the large groove**
  (4 nodes in band, 52.5-52.8 deg of arc, closest node 51.2-51.7 mm at h +0.1..+0.5 mm) in every
  run of that family. But it is held there by ~6-7% of stretch (loop polygon 694-705 mm, ~22-25 N),
  and the instant either gripper opens that energy snaps the belt straight and off the rim:
  the loop polygon drops to ~652 mm and the winding number about the large pulley goes 1 -> 0
  within 2 s. 52 deg of wrap is not enough capstan friction to hold 25 N. Configurations that
  avoid the stretch instead leave the loop slack and it falls to the board (h -21.4 mm).
- Two hazards worth recording. (1) The apex pose compiles to UR `wrist_3` = **6.19 rad**; the
  compiler normalises `wrist_3` to [0, 2 pi) so the equivalent -0.09 rad branch is never chosen,
  and the UR simulation sometimes tracks the other way — the commanded tool jumps ~1.2 m in one
  control step and the solver goes non-finite. Measured **4 aborts in 12 runs** of that family.
  Rotating the tool 180 deg about its approach axis gives `wrist_3` = 3.05 rad and is abort-free,
  but it flips the loop over (large-pulley winding -1 instead of +1) and never wraps, 3/3.
  Pitching the approach 0.8 rad gives 4.80 rad but fails the post-playback settle check, 3/3.
  (2) The repo's `belt_trigger` teleport (`round_belt_lcm_sim.yaml`, point
  [0.482235, 0.166332, 0.039995], tolerance 5 mm) sits right where the Franka's final press goes;
  it fired once in the previous session's attempt and rigidly lifted the whole belt off the pulley.
  It never fired in this session's 22 runs.
- Recommended next step, outside this package's scope: move the `pick` waypoint ~13.6 mm along the
  belt (one element) so the Franka grips node **22** rather than 23. That makes the two grips
  antipodal and the loop then fits both grooves with 1.5 mm of slack and **no stretch at all**, so
  releasing cannot snap it off. Failing that, model the small-pulley slide tensioner.

**Partially releasing the UR gripper so the belt can pay out does not work either (2026-09-18).**
The idea was to let the 2F-85 slip instead of hold while the Franka pulls, so the loop would stay
near its rest length while being laid on the pulleys. Six place-phase configurations, three runs
each (18 more E2E); nothing was landed and magna's sim waypoints are still the shipped ones.

- **Command to jaw.** `ur_gripper_pos_command` becomes a Robotiq byte as
  `static_cast<uint8_t>(clamp(cmd / 0.04, 0, 1) * 255)` (0.04 -> 255, 0.03 -> 191, 0.02 -> 127).
  **These runs predate the width calibration (§2)**: the driver target was still
  `0.005 + byte / 255 * 0.795` rad against the 0.648 rad stop, so every byte below reads about
  12 mm tighter than the same byte does now (191 is now a 17.4 mm jaw, not 5.8 mm). Free-air pad
  separation, measured as the minimum distance between the two pad collision meshes in the built
  scene; the faces stay parallel to within 0.05 mm along the pad, so this is a plain gap:

  | byte | 127 | 177 | 183 | 187 | 188 | 189 | 190 | 191 | 195 | 203 | >= 207 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | gap [mm] | 29.30 | 11.05 | 8.79 | 7.29 | 6.91 | 6.53 | 6.15 | 5.78 | 4.26 | 1.23 | 0.09 |

  0.376 mm per byte through the band, so for the 6.6 mm belt byte 189 just touches, 187 leaves
  0.7 mm of clearance and 191 would need 0.8 mm of squeeze. (Under the calibrated map the same
  jaw openings are bytes 234/231/228 instead of 191/189/187; the conclusions below are about the
  openings, not the bytes, and are unchanged.)
- **The jaw never pinches the belt at its diameter.** Commanded 255 with the belt in it, the driver
  stalls at a **1.23 mm** nominal gap (status 203 on the old scale, 250 on the width scale): the
  round belt is squeezed out from between the flat pads and is retained at their edge, not held
  across its 6.6 mm. For contrast the Franka hand holds the same belt at a **6.44-6.63 mm** width
  with both fingers on their 20 N effort limit.
- **So no partial release makes the belt pay out.** Slip is the belt material coordinate (in
  13.5865 mm rest elements) of the loop point nearest the 2F-85 tip, against its value at the grip;
  the arc column is the largest wrapped arc on the large pulley before the UR lets go:

  | # | UR byte during the pull | slip over the pull | loop peak | arc | result |
  |---|---|---|---|---|---|
  | 1 | 191 `place_3..place_9`, 255 for the seat press | -1.3 .. +0.8 mm | 705-708 mm | 89-106 deg | 0/3 |
  | 2 | 189 at `place_3..place_11` | -1.6 .. +1.1 mm | 704 mm | 0-18 deg | 0/3 |
  | 3 | 187 at `place_3..place_11` | -2.6 .. -0.9 mm | ~705 mm | 87-90 deg | 0/3 |
  | 4 | 187, plus a 6 s dwell after the release | -2.6 .. -0.9 mm | 704-708 mm | 36-89 deg | 0/3 |
  | 5 | 177 (11.05 mm, 4.4 mm of clearance all round) | -3.6 .. -0.5 mm | 702 mm | 0 deg | 0/3 |
  | 6 | as 1, plus a 6 s dwell after the release | -1.3 .. +0.5 mm | 706-708 mm | 36-88 deg | 0/3 |

  The largest slip anywhere is **3.6 mm** against the **45-56 mm** that would have to pay out.
  Opening further does not free the belt, it only lets it sag out of the groove plane (candidate 5
  never forms a wrap at all). Candidates 3 and 4 do put the belt in the large groove reproducibly
  (7 nodes in band, closest node 50.6-50.7 mm at h 0.0), and lose it at the release exactly as
  before.
- **Changing the byte mid-playback also made the UR aborts much worse**: 14 of these 18 runs
  aborted, 12 of them on the `wrist_3` branch (`ur: joint 12.46 rad` = 4 pi - 0.1 while the UR tool
  position and orientation errors are inside their limits, i.e. an unwrapped-joint false positive),
  one on a marginal 0.37 rad joint error and one on a settle timeout; 7 then went non-finite. The
  same waypoints with a constant byte aborted 0/3. Rolling the approach tangentially by
  +-0.3/0.6/1.0 rad does not move the branch (the compiler still walks `wrist_3` 0 -> 6.28 rad) and
  positive rolls fail IK at `post_pick -> pre_place_1`.

The recommendation above is unchanged and is still the thing to fix: the two grips are one rod
element out of phase, the loop therefore has to stretch 6-8% to reach both pulleys, and no gripper
opening removes that.

The hardware parameters are unaffected: hw-mode belt final centroid (0.4839, 0.0169, 0.0620),
unchanged.

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
- **`scripts/replay_viewer.py` refuses a recording (body labels differ).** The recording's
  `body_labels` no longer match the current scene's directive order — either the recording is
  stale (the scene changed since it was made) or it is from a different task. The error names the
  first mismatched body and index; the previously loaded run (if any) stays shown (§10).
- **Recording overhead or `resync` events.** Compare the sim's `[RECORD]` startup/close log lines
  (path, state-every rate, MB/min, final size) against §10's measured numbers; a `resync` event
  (§10) means the real-time loop fell behind (e.g. under `--record-state-every 1` with a busy
  GPU) — lower it back toward the default, or check `nvidia-smi` for other GPU load (§9 above).

## 10. Recording and replay

`--record` (§4) captures a run — sim state, LCM traffic, controller target poses, structured
events — to disk without perturbing the 200 Hz loop; `scripts/replay_viewer.py` is a standalone
viser app that loads a recording and scrubs/plays it back, with no LCM and no magna involved.
Recording is off by default; the CLI keeps its own default unrecorded, and only the procman
`.pmd` files turn it on (§3).

### Recording a run

```bash
# CLI: label + decimation defaults (200 Hz signals, 50 Hz body poses, 10 s chunks)
uv run python scripts/round_belt_lcm_simulation.py --record --record-label my_run

# override the body-pose rate / chunk size
uv run python scripts/round_belt_lcm_simulation.py --record --record-state-every 1 \
    --record-chunk-steps 500
```

Both procman `.pmd` files pass `--record --record-label sim|hw` (§3), so every procman-driven
run lands under `recordings/<timestamp>-sim/` or `-hw/` automatically.

### Run directory layout

`recordings/` is gitignored. Each run is `<root>/<YYYYmmdd-HHMMSS>-<label>/` (`-2`, `-3`, ...
appended if a run with that name already exists, e.g. a procman restart within one second):

- `meta.json` — schema version, labels/index maps, args, git provenance, sha256 of the scene
  files, and the recorder's settings. Key fields: `created`, `label`, `argv`, `git_commit`,
  `git_dirty`, `control_dt`, `lcm_url`, `channels` (the 16-field `LcmChannels`, §2),
  `scene_directives`/`lcm_sim_params` (`{path, sha256}`), `initial_state`, `body_labels` (index -> body,
  fixes `body_q`'s column order), `body_q_layout` (`"xyz_xyzw"`), `joint_labels`/`joint_q_start`/
  `joint_qd_start`/`joint_coord_count`/`joint_dof_count`, `signal_coords`/`signal_dofs` (which
  model coords/dofs are recorded, §2 "Contract table" plus the pulleys — the belt rod joints are
  excluded as useless signals), `belt_bodies`, `pulley_bodies`/`pulley_joints`/`pulley_coords`/
  `pulley_dofs`, `gripper_pad_bodies`, `franka_finger_bodies`, `driver_coords`/`driver_dofs`,
  `hand_body`, `robot_io` (per-robot channel/coord/effort layout), `effort_layout`,
  `finger_tip_body`, `ur_wrist_body`/`ur_tip_in_wrist`, `belt_trigger_point`, `state_every`,
  `chunk_steps`, `finished`, `reason`, `num_steps`, `last_step`, `chunks` (per-chunk file/
  first_step/steps/state_frames). `recorder_failed: true` is added only if the writer thread
  died mid-run; `writer_timeout: true` (with `finished: false`, so the reader still finds the
  chunks written after that snapshot) if the writer was still busy 30 s after the run ended. The
  sim logs a warning when the writer falls more than 3 chunks behind.
- `chunks/chunk_NNNNN.npz` — one file per `--record-chunk-steps` control steps (2000 = 10 s at
  the default). Two row groups, at different rates:

  | key | dtype | shape | rate |
  |---|---|---|---|
  | `step`, `sim_time`, `wall_time`, `compute_ms` | i64/f64/f64/f32 | `(n,)` | 200 Hz |
  | `hand_target_mm`, `hand_stale`, `franka_stale` | f32/bool/bool | `(n,)` | 200 Hz |
  | `robotiq_cmd`, `robotiq_status` | u8 | `(n, 3)` | 200 Hz |
  | `robotiq_cmd_valid`, `robotiq_opening`, `lcm_rx` | bool/f32/i32 | `(n,)` | 200 Hz |
  | `joint_q`, `joint_qd` | f32 | `(n, signal_coords)`, `(n, signal_dofs)` | 200 Hz |
  | `efforts` | f32 | `(n, effort_layout)` | 200 Hz |
  | `state_step` | i64 | `(m,)` | 50 Hz (every `state_every`-th step) |
  | `body_q` | f32 | `(m, B, 7)` | 50 Hz |

  `n` = chunk steps, `m` = state frames in the chunk, `B` = body count (86 in the round-belt
  scene). `robotiq_cmd`/`robotiq_status` are `(position, speed, force)` bytes; `body_q` rows are
  `xyz_xyzw`.
- `events.jsonl` — one `{step, sim_time, kind, data}` object per line, recorded kinds: `run_start`,
  `input_first`, `input_stale`, `hand_command`, `hand_stale`, `robotiq_command`, `belt_placed`,
  `resync`, `run_end`.
- `targets.jsonl` — one `{step, sim_time, channel, payload}` object per controller target message
  received while recording (see "Target poses" below).

### Size and overhead

At the defaults (200 Hz signals restricted to the non-belt joints, C' = 33 coords / D' = 31 dofs,
15 efforts; 50 Hz `body_q`, B = 86 bodies; 10 s chunks): **~11.6 MB/min**, measured **35.0 MB**
over a 3-minute real-time run (18 chunks, 9000 state frames; measured 2026-09-21). Per-step
overhead measured the same day: **+0.04 ms/step mean** (`--record` vs. off, both non-realtime,
GPU idle), well under the 0.10 ms design bar; `--record-state-every 1` (every step's `body_q`,
~90-100 MB per 3 min) still measured only +0.02 ms/step mean. With `--record` off the step time
is unchanged within measurement noise (-0.01 ms). A killed run is crash-safe up to its last
fully written chunk — `Recording.load` reads every chunk still listed or found on disk and raises
`ValueError` naming any chunk that is missing or breaks the step sequence.

### Events and derived metrics

Recorded events (above) are raw facts logged by the sim as they happen. Everything else is
computed at load time by `replay_metrics.py` from `body_q`/`joint_q`, so the recording format
never has to change to add a metric:

- **Hold rules** (debounced 0.2 s, runs under 10 frames ignored): Franka hold = belt-to-
  `finger_tip` gap <= 15 mm and hand width in [1, 10] mm; UR hold = belt-to-2F-85-tip gap
  <= 15 mm and the Robotiq command byte >= 200. `ur_release` / `ur_partial_release` fire when the
  byte drops below 100 / into [100, 200) right after a UR hold.
- **Pulley wrap** — the largest contiguous azimuthal arc (gaps <= 30 deg merged) of belt bodies
  seated in the groove: within 6 mm axially of the pulley mid-plane and within 5 mm radially of
  the belt's seat radius (small 17.38 mm, large 50.73 mm, §8). `pulley_turning_start/_end` fire
  on a >= 3 deg change in the unwrapped joint angle over a 1 s window.
- Other metrics: `belt_loop_mm`/`belt_loop_change_pct` (closed polygon over `belt_bodies`, vs.
  frame 0), per-pad/per-tip gaps, `compute_ms` (shifted to the row it timed; row 0 is the CUDA
  graph capture, not a real step, see Known limits) and `realtime_factor`.

### Target poses

While recording only, the bridge additionally subscribes the controllers' end-effector target
channels (magna source-verified, §2): `TARGET_CARTESIAN_POSE_TRAJECTORY` and
`UR_TARGET_CARTESIAN_POSE_TRAJECTORY` (`dairlib::lcmt_timestamped_saved_traj`, a trajectory of
position/orientation knots) and `UR_TARGET_SPATIAL_POSE` (`magna::lcmt_spatial_pose`, one pose).
Orientation is always quaternion **wxyz**. The UR trajectory is `tool0` expressed in the UR
`base` frame (`base_link` rotated 180 deg about Z); the Franka trajectory is in `panda_link0`;
the spatial pose is already world. Knot times are on this sim's own clock (verified against
magna's timestamp sources), with a +-0.5 s fallback ("hold the first knot", warned once per
channel) if a trajectory's own clock assumption does not hold.

**Not captured:** `OSC_TARGET_TRACKING_DEBUG` (magna's assembly controller hard-codes it to a
separate `local_lcm` on the shared default group, independent of `--lcm-url`) and `OSC_DEBUG`
(`lcmt_osc_output`, a large nested type) — both out of scope; nor the belt mesh
(`DRAKE_VIEWER_DEFORMABLE`) or the RGBD cameras, neither of which is LCM traffic this sim needs
to replay the scene from `body_q` alone.

### Why not Newton's `ViewerFile`

Newton's own autosave viewer was not reused as the recording format:

- it clones every `State` array into GPU memory for the whole run and re-serialises the entire
  history on every autosave — O(n^2) cost over a long run;
- it writes a single CBOR blob with no random access and no channels for signals or events
  (`log_scalar` is a no-op there);
- it serialises the whole `Model` (meshes, warp arrays) instead of just the per-frame state.

The replay app borrows only its idea of driving `viewer.log_state` from a rebuilt model; the
chunked, columnar format above is otherwise unrelated.

### Replaying a run

```bash
uv run python scripts/replay_viewer.py                              # newest run, port 8081
uv run python scripts/replay_viewer.py --recordings PATH --run NAME
uv run python scripts/replay_viewer.py --port 8082 --show-collision --device cuda:0
uv run python scripts/replay_viewer.py --render-fps 15 --stats     # slow link or weak client
```

Port 8081 by default (the live `--viewer viser` uses 8080, so both can run at once). The GUI has
four folders: **Recording** (run dropdown + rescan, provenance/warning markdown), **Timeline**
(frame slider, transport buttons `|< -10 -1 Play/Pause +1 +10 >|`, speed 0.1x-4x, loop), **Plots** (uPlot charts in a
sliding time window, default 10 s, with a dashed vertical line at the current time; currently one chart, Franka joint
efforts, hideable; more are added in `CHARTS` in `src/task_common/replay_panels.py`), and
**Triads** (none shown by default: pick a body, or a target channel above that the run
recorded, from the "Add triad" dropdown to show its frame; each shown triad gets a button with
an X that removes it; axes length/radius). Loading a run refuses one whose
`body_labels` differ from the rebuilt scene's (directive order fixes body indices) and warns, but
does not refuse, if the recorded scene YAML's sha256 no longer matches the file on disk.

**Playback rates.** The app ticks at 50 Hz (GUI requests and the playback clock). During
playback, it redraws the 3D scene at no more than `--render-fps` (default 25). It updates the
frame slider and time text at 10 Hz, and the charts at 5 Hz. A seek, a step, the loop wrap and
the last frame always redraw at once. Each chart's y ranges are fixed per run (whole-run
min/max plus 5 % padding) and are sent once when the run loads. Sending new ranges would make
the browser rebuild the chart, so they are never resent during playback. The window is min-max
decimated to at most 300 points per series, which keeps spikes. A chart whose window has not
changed (for example, while paused) is not resent. The replay model is built on the CPU by
default (`--device`), because on a GPU each frame needs about 100 blocking device-to-host copies
and competes with a running sim. `--stats` logs the tick, redraw and chart-update rates every
10 s while something plays.

Measured on 2026-09-21 (1x, looped, default charts, local headless Chromium, 3-min run):

| | before | after |
|---|---|---|
| 3D redraws/s (server) | 47, main thread ~100 % busy (21 ms each, GPU model) | 25 (11 ms each, CPU model) |
| data sent to the browser | 1.27 MB/s | 0.40 MB/s |
| uPlot chart rebuilds/s (browser) | 17 | 0 |
| lag behind the server over a 1 MB/s link | 3.3 s mean, still growing | 22 ms |

The initial page load still sends the scene meshes (about 6 MB) once. On a very slow link, the
page takes that long to appear before playback can keep up.

### Known limits

- Target-pose replay was validated on synthetic target data only: magna's own assembly
  controller (`run_round_belt_assembly_controller.cc:70`) hard-codes its C3 debug publishers to
  `udpm://239.255.76.67:7667`, the forbidden shared group, so it could not be run in this
  session; the trajectory-frame math (above) is verified against the magna source instead.
- `Recording.frame_at_step`/`frame_at_time` round **down** to the last state frame at or before
  the given step — there is no step-0 frame at the default `--record-state-every 4` (the first
  frame is at step 4), so jumping to an event before step `state_every` lands on frame 0.
- On disk, `compute_ms[i]` is the compute time of the step that produced row `i - 1` (a step is
  timed after its row is written): row 0 holds `0` and row 1 the one-off CUDA graph capture.
  `replay_metrics.Metrics.compute_ms` shifts it back by one row, so there row 0 is the capture
  and the last row is `NaN`.
- The recorder logs a `hand_command` event only when the goal (`target_mm`, `force`) changes, not
  per message (magna republishes the same goal every controller tick); the event keeps the
  `utime` of the first message carrying that goal. The replay app still drops repeats of the
  same goal when it merges events (`ReplayApp.all_events`), for older recordings.

### How the checks exercise this headlessly

`scripts/checks/check_recording.py` drives a `RoundBeltLcmSimulation` with `--record` through a
scripted sequence via `control_step()` and checks the on-disk files, `Recording.load`, events,
signals and per-step overhead. `scripts/checks/check_replay_viewer.py` records two runs the same
way, then drives a headless `ReplayApp` (no browser) through run selection, seeking, playback,
metrics, events, plots, triads (including synthetic target messages), a label-mismatch
refusal and the playback redraw/chart-update rates. Both use a private LCM group and never open
a browser or touch `recordings/` in the repo. See §8 for what each checks in detail.
