# SpaceMouse teleop of stage 2 (operator guide)

One continuous sim in the wr625b15 scene: a learned controller does stage 1, the scripted UR
exit runs, then you drive the Franka with a SpaceMouse to wrap the belt round the small pulley.
Press `d`: the scripted release runs, the outcome is labelled, the episode is saved.

```
SpaceMouse -> spacenavd -> run_franka_driver --(LCM, private URL)--> sim process
   (eval_learned_mpc.py --stage2-chain --stage2-teleop) -> 7-knot Franka trajectory -> OSC
```

## 1. SpaceMouse daemon

```
sudo apt install -y spacenavd
sudo systemctl enable --now spacenavd
systemctl status spacenavd
lsusb | grep -i 3dconnexion
```

The socket is `/var/run/spnav.sock`. If apt has no package, build from source as in
`~/Workspace/spacemouse_driver/README.md`.

## 2. Build the publisher (once)

```
cd ~/Workspace/spacemouse_driver && bazel build //core:run_franka_driver
```

Bazel fetches libspnav itself.

## 3. Start the publisher on the private sim URL

The URL is `udpm://239.255.76.<port-7600>:<port>?ttl=0`; the default port 7840 gives:

```
~/Workspace/spacemouse_driver/bazel-bin/core/run_franka_driver \
  --lcm_url='udpm://239.255.76.240:7840?ttl=0' \
  --gripper_command_lcm_channel=TELEOP_HAND_IGNORED
```

Never use port 7667 (the launcher refuses it, and 8081/8082, and ports below 7780). The renamed
gripper channel is explicit only: the buttons cannot open the hand (grippers are set in-process
and the sim never publishes `PANDA_HAND_STATUS` here). The launcher does not start the driver.

## 4. Launch

```
cd /home/hienbui/git/magna_belt_simulation-main
uv run --frozen python scripts/lcs/teleop_stage2.py --model V45c --start nominal
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `V45c` | stage-1 controller: `V45c`, `V55b` |
| `--start` | `nominal` | comma list: `set2_gv_00`, `set2_gv_06`, `set3_gv_02`, `set3_gv_03`, `ur_z-30`, `ur_z+15`, `up-30`, `up-20`, `up+20`, `up+30`, `nominal` (`up-30` is a known-unseatable start) |
| `--port` | 7840 | private LCM port (match the driver URL) |
| `--out` | `data/lcs/realfirst/cmp/v12_120/session_<YYYYmmdd-HHMM>` | one run dir per start: `<out>/<model>_<start>` |
| `--scene` | `round_belt_scene_wr625b15.yaml` | exported as `ROUND_BELT_SCENE` |
| `--max-episode-s` | 25 | stage-1 timeout |
| `--dry-run` | off | print the command(s), run nothing |

Unknown flags pass through to the harness; later flags win (`--teleop-max-s`, `--teleop-wait-s`,
`--teleop-lin-mps`, `--teleop-ang-rps`, `--teleop-tilt-deg`, `--teleop-floor-mm`,
`--teleop-stale-s`, `--teleop-channel`, `--viewer-port`, `--render-every`).

The terminal shows stage 1 and the UR exit, then `[TELEOP] keys: ...` and a live status line
(time, pulley labels, wrap angle, held, speed, `CLAMP`, SpaceMouse msgs/s, real-time factor).
Live viewer: http://localhost:8080. Start moving the mouse only after the `[TELEOP] keys` line.

## 5. Keys

| Key | Action |
|---|---|
| `d` | done: scripted release, outcome, save |
| `a` | abort: the episode is discarded (files removed, row marked `discarded`) |
| `p` | pause / resume (the target holds) |

Automatic ends: grasp lost (aborted, episode kept); `--teleop-max-s` reached (counts as done).

After each episode: the result line, the session tally, then
`[n]ext / [r]etry same start / [q]uit:` (Enter = next). Between starts the launcher asks
`next start <key>? [Y/n]`. Without a tty there are no keys or prompts; the phase ends at
`--teleop-max-s`.

## 6. Axes, speeds, limits

Motion is integrated in the world frame (the Franka OSC plant frame), not the camera frame. The
driver publishes axes `[x, z, y]` of the raw device (`linear_axis [0, 2, 1]`, signs +1), so
published x = device x, y = device z, z = device y; angular likewise. The sim applies these as
world x, y, z (and rotations about them). Check the directions in the first session: no sign
flag exists.

| Quantity | Value |
|---|---|
| Full-deflection linear speed | 0.05 m/s (`--teleop-lin-mps`) |
| Full-deflection angular speed | 0.1666 rad/s (`--teleop-ang-rps`) |
| Tip floor | 2 mm above the plate (`--teleop-floor-mm`) |
| Tilt cap | 29.5 deg (`--teleop-tilt-deg`) |
| Target lead vs measured pose | clamped to 20 mm / 15 deg (shows `CLAMP`) |
| Silent SpaceMouse | stop after 0.5 s (`--teleop-stale-s`) |
| Phase limit | 300 sim s (`--teleop-max-s`) |
| Wait for first message | 10 s (`--teleop-wait-s`), then holds until one arrives |

## 7. Real-time factor

With the viewer, teleop runs at about 0.6x real time, so motion looks slow. Integration is on sim
time, so the data are unaffected. A larger `--render-every` (default 20) may help.

## 8. What is saved

Per start, under `<out>/<model>_<start>`: one continuous episode (stage 1, UR exit, teleop,
release) in the `--stage2-chain` format with `sim_cmd_source` 9 on teleop frames (8 = UR exit,
6 = release), `sim_chain_from`, `sim_post_from`, actions = commanded 12-D delta; `index.json`
rows with `chain.teleop` and `chain.success_both`; and `recordings/` for
`uv run python scripts/replay_viewer.py --recordings <out>/<model>_<start>/recordings`.

Success: both pulleys engaged after the release (`chain.success_both`). The session prints
`session N/M both engaged` after each episode, and the harness logs
`[EVAL] teleop: N/M both engaged` at the end. Discarded aborts and episodes that never reached
teleop are not counted.

## 9. Troubleshooting

- No SpaceMouse messages (`spacemouse 0/s`, or the `--teleop-wait-s` warning): check
  `systemctl status spacenavd`, that the driver URL matches the launcher's `--lcm-url` (printed
  in the command), and that you did not use 7667.
- Sim lock busy (launcher sits at `flock`): another sim is running; wait for it.
- Stage 1 fails (result line `stage 1 ...`): retry (`r`) or take the next start.

## 10. GUI / stage-1 teleop

The demo GUI (`scripts/demo_gui.py --backend sim`, panel 1 "Start teleop") runs the chain with
`--mode baseline --teleop-stage1 --stage2-teleop --teleop-cmd-file <session>/cmd.json` on the
app's LCM URL (default port 7850: start the driver of section 3 with
`--lcm_url='udpm://239.255.76.250:7850?ttl=0'`; the panel prints the command).

- `--teleop-stage1`: stage 1 is a teleop phase too (no controller; the UR keeps its start joint
  target). Frames `sim_cmd_source` 9, stage 0, phase `s1:teleop`; `d` (or `--teleop-max-s`) ends
  it -> scripted UR exit -> stage-2 teleop; `a` discards the episode. Needs `--mode baseline
  --stage2-teleop`.
- `--teleop-cmd-file PATH`: keys also come from this file (its first char; the harness empties it
  once read), so no tty is needed. The GUI buttons write it: Done stage 1 = `d`, Pause = `p`,
  Stop = `d` in stage 2 (release + save), Reset = `a` (discard).
