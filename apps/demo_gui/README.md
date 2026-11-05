# Demo GUI

Dash app for the belt-assembly pipeline: collect a demo, augment (MILES), train, export, evaluate.
Full description and limits: `docs/demo-gui.md`.

```bash
# mock backend: replays recorded data, no GPU / LCM / sim
uv run --frozen python scripts/demo_gui.py --backend mock
# sim backend (private LCM 7850, live viewer 8091, replay viewer 8092; GUI on 8090)
uv run --frozen python scripts/demo_gui.py --backend sim --sim-port 7850 --viewer-port 8091 --replay-port 8092
```

Open `http://127.0.0.1:8090`. The launcher never opens a browser or viewer.

Sim preconditions: `bash data/lcs/realfirst/cmp/v12_152/gpu_free.sh`, no other sim running, the
`data/lcs/realfirst/cmp/v12_79/sim.lock` lock is taken automatically by sim jobs.

Controller build (scripted demos need it): `cd /home/hienbui/git/magna && bazel build //systems/controllers:run_round_belt_assembly_controller`

Tests (no pytest in the venv): `uv run --frozen python apps/demo_gui/tests/run_all.py`
