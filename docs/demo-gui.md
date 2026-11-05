# Demo GUI (`apps/demo_gui`)

One app, five tabs, two backends. Launcher: `scripts/demo_gui.py`.

## Launch

```bash
# mock: replays recorded data, no GPU / LCM / sim
uv run --frozen python scripts/demo_gui.py --backend mock
# sim
uv run --frozen python scripts/demo_gui.py --backend sim --sim-port 7850 --viewer-port 8091 --replay-port 8092
```

Ports: GUI 8090 (`--http-port`), standalone viewer 8091 (`scripts/lcm_viewer.py`, launched by you;
the GUI never starts it), replay viewer 8092, private LCM 7850.
`--sim-port` must be in 7780..7859 and not 7667 (magna's shared port), 7840/7841 (SpaceMouse teleop),
7855, 7857; the launcher refuses those. `--magna-root` defaults to `/home/hienbui/git/magna`.
The launcher prints the URL; it never opens a browser or a viewer. Viewers started from the GUI
(Data / Evaluate tabs) are served on 8092. Sim jobs (demo, teleop, eval) pass `--publish-body-poses`
instead of a viewer: `--viewer-hz` (default 60) sets the body-pose publish period
`--render-every round(200 / hz)` sim steps (60 -> 3 = 66.7 Hz); publishing is cheap (no viser inside the sim).

They also pass the harness's `--realtime` (off by default; off, runs are unchanged): every sim step
of an episode starts on one absolute wall clock (`--pace` x sim time; a late step does not sleep and
catches up, a lag over 50 ms rebases the clock instead of a fast-forward burst), the stage-2 MPPI
node starts up during the UR exit instead of freezing the sim before stage 2, and the MPC loop's
per-step checks run while the GPU steps (their aborts act one step later). The sim runs at 1x where
the GPU step fits in 5 ms and as fast as it can elsewhere; the stage-1 node start-up and the node
shut-downs at the stage-1 / stage-2 ends still pause it. Not with `--live-viewer`.
Launch the standalone viewer once; it keeps the last frame between runs:

```bash
ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_wr_pin640b1.yaml uv run --frozen python scripts/lcm_viewer.py --lcm-url 'udpm://239.255.76.250:7850?ttl=0' --port 8091
```

One viewer dock on the right, visible from every tab, shows (1) the standalone viewer (8091) titled
`live: <job>` while a demo or eval job runs and 8091 accepts connections, else (2) the replay viewer
(8092) last opened from Data / Evaluate, else (3) the standalone viewer titled `viewer`, else (4) a
placeholder with the launch command above. The standalone iframe keeps its URL across runs (no reload;
only the title changes). The ⇄ button collapses it; "Open in new tab" opens the same URL.
Drag the dock's left edge to resize it (320 px .. 70 % of the window, kept per browser); double-click
the edge to reset to 520 px. Below 1000 px window width the dock overlays the content.

## Backends

- mock: copies and replays recorded runs (fixture start state, e2e run, MILES buffer, training log,
  the pre-exported `V63j_tol50b` arm) with a pace scale. Exercises every callback, no GPU.
- sim: real jobs. Every sim job is wrapped in `flock data/lcs/realfirst/cmp/v12_79/sim.lock` and
  waits for `data/lcs/realfirst/cmp/v12_152/gpu_free.sh`. Dash itself imports no warp/newton.

## Sim preconditions

1. `bash data/lcs/realfirst/cmp/v12_152/gpu_free.sh` succeeds (GPU idle); no other sim or collection running.
2. The baseline controller is built (scripted demos refuse otherwise; teleop does not need it):
   `cd /home/hienbui/git/magna && bazel build //systems/controllers:run_round_belt_assembly_controller`
3. No orphan `run_round_belt_assembly_controller` or `mppi_controller_node.py` on port 7850.
4. Whole-run scene is `wr_pin640b1` (`Cfg.scene`).
5. For a live view: the standalone viewer running on 8091 (command under Launch).

## Tabs

1. Collect. Styled UI; per arm one side view of the grasp along the belt, with the real Franka /
   Robotiq finger silhouettes and the belt in orange: drag the fingers along the belt = `slide_mm`,
   up/down = grasp depth `depth_mm` (down = deeper; `Variant`'s `franka_depth_mm` / `ur_depth_mm`,
   + = shallower = moved back along the pick frame's tip-to-wrist +z, applied after slide and
   pivot), drag the handle below the fingertips (or the slider) = the finger-belt crossing angle
   `pivot_deg` (`Variant`'s `franka_pivot_deg` / `ur_pivot_deg`: a turn about the jaw axis
   `approach x t` through the slid grasp point; + = finger tops lean toward +t, clockwise on the
   canvas). A drag is split exactly into slide + depth, so the fingers stay where dropped (to the
   0.1 mm / 0.5 mm steps). No roll (sent as 0). Accepted poses keep the finger collision meshes >= 2 mm
   from the holder in 3D (table in `assets/grasp_geom.json`, below); dashed = the region the grasp point
   (red dot) may take at the current angle and the angle range at the current slide / depth (= slider
   bounds, within +-20 deg). A drag outside snaps to the nearest allowed pose (keeping the angle; an
   angle change keeps slide / depth), reads "AT LIMIT" and draws that limit bold. Start / Reset pose:
   Franka nominal; UR depth +6.5 mm (shallower), because the default UR pick sits 3.2 mm inside the
   holder's bottom plate in 3D (sim runs held the belt at `ur_depth_mm` +6 / +9).
   Both canvases are seen from outside the holder (+t = increasing belt index) and draw every holder
   part within 70 mm across the belt. Buttons: Make start (variant -> held start state,
   row in `starts/index.json`), Record scripted demo, Record teleop demo, Done stage 1 (`d`), Pause (`p`),
   Stop, Reset. Top bar: running jobs, the newest finished job and one expandable `N done · M failed`
   chip. Live cloud / pose / gripper plots from `telemetry.json`.
2. MILES. Pick session or repo-seed demos, cycles, wp-stride; runs `miles_seed`, shows the buffer growing
   and the report. Note: with the tuned `--wp-motion-mm` flags miles_seed ignores wp-stride.
3. Data. Table of all episodes (demos, MILES, eval), trace plots, cloud scrub, replay viewer on 8092.
4. Train. Pick data (demos, MILES, seed rows) -> `buffer.json` -> `refit_from_buffer` (curves from the log,
   report) -> reconstruction vs ground truth per frame -> Export (`post_export.sh`, writes `arm.json`).
5. Evaluate. Run the e2e hard suite with an exported arm (or a pre-exported one) on the chosen starts;
   results table + `hard_suite.md`; scrub a finished episode (frame 0..T-1)
   + latent distance to goal per stage and one-step prediction error (raw latent norm) from the
   episode npz, replay viewer.

Data / Evaluate scrub also sends that frame's recorded body poses (`<npz dir>/recordings/<episode>`, by
`sim_step`) to the standalone viewer in the dock; refused while a sim job runs (status in the plot title).

## Teleop flow (stage 1, Franka)

Make start -> start the SpaceMouse driver on the app's LCM URL:

```bash
~/Workspace/spacemouse_driver/bazel-bin/core/run_franka_driver \
  --lcm_url='udpm://239.255.76.250:7850?ttl=0' \
  --gripper_command_lcm_channel=TELEOP_HAND_IGNORED
```

(the Start teleop tooltip shows the same command for the configured port; details in `docs/teleop-stage2.md`
section 10) -> Record teleop demo -> drive the Franka -> Done stage 1 (`d`) lets the scripted UR exit and
stage 2 finish -> Stop. Pause (`p`) holds; Reset sends `a` (discard) and deletes the demo.

## Sessions and stopping

`data/demo_gui/sessions/<YYYYmmdd-HHMM>/` (or `--session <dir>`): `starts/ demos/ miles/ train/ eval/ jobs/`,
`telemetry.json`, `cmd.json`. Each job writes `jobs/<name>.json` (pid, pgid, argv) and a log.
Stop with the panel Stop buttons; otherwise `kill <pid>` with the pid from `<session>/jobs/*.json`
(sim jobs: `kill -- -<pgid>` to take the whole group). After stopping a sim run check no
`run_round_belt_assembly_controller` / `mppi_controller_node.py` is left on the port (`ps`); orphans
corrupt later runs and the node holds the GPU. Panel Stop kills the job's own and orphaned
controllers / MPPI nodes on the job's URL (by PID; another live job's are kept). A sim job gated by
`gpu_free.sh` for over 30 s shows `waiting for GPU: <pid> <name>` (nvidia-smi) on its chip.

## Known limits

- Teleop drives the Franka only; the UR stays static in stage 1 (may not seat the large pulley: stage 1
  can fail, then no UR exit / stage 2 / release; the episode is kept).
- Teleop Stop after the stage-1 timeout (`--teleop-max-s`) ends the run without release / save: the
  harness publishes no phase to the GUI, so it cannot tell the run moved past teleop.
- Teleop demos cannot feed MILES yet (`s1_cut` refuses stage-1 cmd sources other than 0/1).
- Sim telemetry has poses and gripper widths only: `cloud` and `belt_xyz` are null (the harness has no
  perception), so the live cloud plot is empty in sim mode.
- Sim `make_start` runs under the default scene, the demo under `pin640b1`.
- The old harness `--magna-root` default (`magna-deploy-learned-lcs`) no longer exists; the app passes
  `/home/hienbui/git/magna`, whose `bazel-bin` controller must be rebuilt (command above) or scripted
  demos are refused.
- MILES uses run11.sh's tuned flags; `--belt-ref arrival`, `--on-disturb skip-waypoint` are not passed.
- Grasp limits follow the held-state silhouettes: regenerate `assets/grasp_geom.json`
  (`backend/make_silhouettes.py --device cpu`) if the scene, snapshot or pick waypoint changes
  (`test_grasp_geom.py` checks the hashes). The pivot is drawn as an in-plane turn; the real axis is
  12.4 deg (Franka) / 11.7 deg (UR) off the view normal, so the drawn fingers are within ~0.2 mm of
  the sim's at 10 deg (`test_variant_pivot.py`). Slide, pivot and depth limits are each found with the
  others at 0 (nominal); the combination is not checked. Depth: deeper = the same holder rule plus
  the bottom plate's top as a floor (Franka -6.5 mm; the nominal UR pick already sits 3.3 mm below the
  plate top, so UR -0.5 mm); shallower = the fingertips stay 0.5 mm below the belt disc about the grasp
  point (UR +10.5 mm; the nominal Franka tips sit only 0.5 mm below the grasp point, so Franka 0 —
  no shallower Franka grasp). `franka_depth_mm` is untested in sim.
- Holder clearance: `make_silhouettes.py` (CPU, ~30 s, `--margin-mm`, default 2) writes a 3D table per arm
  (`feasible` + `default_pose` in `grasp_geom.json`: finger collision meshes at the exact `variant_waypoints`
  pick vs the holder's collision prisms, slide 0.5 mm x pivot 1 deg x depth 0.5 mm); the shallower depth
  limit is still the belt-in-jaw rule. It replaces the one-knob slide / deeper-depth / pivot limits above.
- Export has no fallback path: `post_export.sh` is fully parametrized (no pre-exported-arm-only mode needed).
- Not verified in a browser (no headless one): plotly shape drag, the CSS look, and all sim paths.
- Mock `train` replays the recorded log whatever the epoch count; mock `evaluate` returns one fixture episode.

## Standalone LCM viewer

`scripts/lcm_viewer.py`: a long-lived CPU-only viser viewer you launch once. It builds the scene,
then renders the newest `NEWTON_VIEWER_DRAW` body-pose message on a private LCM URL (robots,
grippers, belt, pulleys) at <= `--hz` (60). It shows the default pose until the first message
and keeps the last frame between runs.

```bash
ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_wr_pin640b1.yaml uv run --frozen python scripts/lcm_viewer.py --lcm-url 'udpm://239.255.76.250:7850?ttl=0' --port 8091
```

- The sim must publish: `--publish-body-poses` (harness) or `--publish-body-poses --body-poses-every N` (sim CLI).
- `ROUND_BELT_SCENE` must match the sim's scene: poses match by link name, and no match warns once
  and renders nothing.
- The "point cloud" checkbox (`--point-cloud` starts it on) shows a cloud only when a sim publishes
  `POINT_CLOUD_CROPPED` (`--cameras --publish-point-cloud`).
- A busy `--port` or the shared 7667 group exits 2.

## Tests

`uv run --frozen python apps/demo_gui/tests/run_all.py` runs every `test_*.py` as a plain script
(CPU only, ~25 s; pytest-compatible but not installed). `test_e2e_mock.py` drives the full mock pipeline;
`test_ports.py` checks the LCM URL ranges, launcher refusals and that `7667` appears only in refusals.
`test_collect.py::test_drag_formats` posts the real plotly drag events (path, handle x0..y1 or a
subset, autosize, other shapes, past the limit, unchanged slider, depth drags at 0 / 20 deg, old
stores without `depth_mm`); `test_variant_pivot.py` / `test_variant_depth.py` prove
`Variant.franka_pivot_deg` / `franka_depth_mm` leave the default `make_grasp_variants` outputs
byte-identical to git HEAD's copy (pass another reference path as the argument; the combined UR
slide + pivot + depth case needs a post-PKG-11 reference). `test_live_viewer.py` proves the dock's
4-state rule on fake sockets, the stable standalone src, and that the launch command above equals
`Cfg().viewer_cmd`.
Lint: `.venv/bin/ruff check apps/ scripts/demo_gui.py`.
