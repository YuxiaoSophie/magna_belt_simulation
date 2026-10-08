#!/usr/bin/env python3
"""Headless regression check of the learned-MPC replay layers (``replay_learned_mpc.py``) on
recorded runs: the git HEAD module vs the tree's, no viser server.

Per run (learned-MPC debug channel present): load with both modules (deploy / decoder / demo
paths from the run's meta, or ``--deploy`` / ``--decoder`` overrides), then render every layer
at ``--frames`` evenly spaced frames into a fake scene that asserts viser's shape rules.

R1 parse: same layer availability (except the tree's MPPI ``z_ref`` target fallback), same solve
   count and steps, ``ee_franka`` / ``ee_ur`` identical (HEAD's crashes are reported, not failures).
R2 decode: world-frame decoders: ``belts`` identical; pulley-frame decoders (tree only):
   the best knot-0 cloud within ``--max-chamfer-mm`` of the recorded GT belt (``pcd_belt``);
   min / median / max reported (the max is model quality, e.g. v1's noisy-cloud encoder).
R3 render: the tree renders every layer at every sampled frame without an exception.

    uv run --frozen python scripts/checks/check_replay_learned.py [--runs RUN_DIR ...]
        [--glob 'data/lcs/mpc_eval/*/recordings/*'] [--limit 40]
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.util
import subprocess
import sys
import tempfile
import traceback
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from task_common.recording import Recording

MOD = "src/task_common/replay_learned_mpc.py"
HEAD_N_LATENT = 16  # HEAD's fixed slice of x_sol = [z; p_f; p_ur]
DEFAULT_RUNS = [  # one per controller era
    "data/lcs/mpc_eval/20260924-133342-final-learned-nominal/recordings",
    "data/lcs/mpc_eval/20260925-002801-stages/recordings",
    "data/lcs/mpc_eval/20261001-v10/20261001_V10A_seat6_Qw_ub-ins/recordings",
    "data/lcs/realfirst/cmp/C1_transport/iter1/collect/recordings",
    "data/lcs/realfirst/cmp/full2_gate/iter1/collect/recordings",
]


def load_head(tmp: Path) -> types.ModuleType:
    src = subprocess.run(["git", "show", f"HEAD:{MOD}"], cwd=REPO, check=True,
                         capture_output=True, text=True).stdout
    p = tmp / "replay_learned_mpc_head.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("replay_learned_mpc_head", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


class Widget(types.SimpleNamespace):
    def on_update(self, fn):
        return fn


class Scene:
    """viser's add_point_cloud / add_mesh_simple shape asserts, recorded handles."""

    def __init__(self):
        self.n = 0

    def add_point_cloud(self, name, points, colors, **kw):
        assert points.ndim == 2 and points.shape[-1] == 3, f"{name}: points {points.shape}"
        assert np.isfinite(points).all(), f"{name}: non-finite points"
        self.n += 1
        return types.SimpleNamespace(points=points, visible=True, remove=lambda: None)

    def add_mesh_simple(self, name, vertices, faces, **kw):
        assert vertices.ndim == 2 and vertices.shape[-1] == 3, f"{name}: vertices {vertices.shape}"
        assert faces.ndim == 2 and faces.shape[-1] == 3, f"{name}: faces {faces.shape}"
        self.n += 1
        return types.SimpleNamespace(vertices=vertices, visible=True, remove=lambda: None,
                                     opacity=kw.get("opacity"), color=kw.get("color"))


def fake_app() -> types.SimpleNamespace:
    @contextlib.contextmanager
    def folder(*a, **k):
        yield Widget()
    gui = types.SimpleNamespace(
        add_folder=folder, add_markdown=lambda *a, **k: Widget(content=""),
        add_checkbox=lambda label, value: Widget(value=value, disabled=False),
        add_slider=lambda *a, **k: Widget(value=k.get("initial_value"), disabled=False))
    return types.SimpleNamespace(gui=gui, server=types.SimpleNamespace(scene=Scene()),
                                 call_soon=lambda fn: fn(), _set_gui=lambda *a: None,
                                 current_frame=0, current_step=0)


def panel(mod, rec, args):
    p = mod.LearnedMpcPanel(deploy=args.deploy, decoder=args.decoder, layers=mod.LAYERS)
    app = fake_app()
    p.build_gui(app)
    p.on_run_loaded(app, rec)
    return p, app


def chamfer_mm(a, b) -> float:
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return 1e3 * 0.5 * (d.min(1).mean() + d.min(0).mean())


def gt_belt(run: Path, rec, step: int) -> np.ndarray | None:
    """GT centreline of the run's episode npz (``<collect>/<run name>.npz``) at ``step``."""
    ep = run.parent.parent / f"{run.name}.npz"
    if not ep.is_file():
        return None
    with np.load(ep, allow_pickle=True) as e:
        if "pcd_belt" not in e.files or "sim_step" not in e.files:
            return None
        k = int(np.searchsorted(e["sim_step"], step, side="right")) - 1
        return None if k < 0 else e["pcd_belt"][k].astype(np.float64)


def check_run(run: Path, head, tree, args) -> dict:
    rec = Recording.load(run)
    if not tree.has_learned_debug(rec.meta):
        return {"run": str(run), "skip": "no learned-MPC debug channel"}
    out = {"run": str(run.relative_to(REPO) if run.is_relative_to(REPO) else run)}
    try:
        ph, _ = panel(head, rec, args)
        out["head"] = "ok"
    except Exception as exc:  # noqa: BLE001 - HEAD crashing on new runs is the bug being fixed
        ph, out["head"] = None, f"crash {exc!r}"[:160]
    pt, app = panel(tree, rec, args)
    out["solves"] = len(pt.solves)
    out["available"] = {k: v for k, v in pt.available.items() if v}
    fails = []
    n_x = pt.solves[0].x_sol.shape[0] - 6 if pt.solves else HEAD_N_LATENT
    out["n_x"] = n_x
    if ph is not None and n_x != HEAD_N_LATENT:  # HEAD slices 16 latents: wrong by design
        out["head"] = f"{out['head']} (n_x {n_x}: HEAD mis-slices, compared to GT only)"
        ph = None
    if ph is not None:  # R1, R2 (world decoders)
        new = {"target_belt"} if pt.target_source == "z_ref" else set()  # MPPI goal fallback
        if {k: v for k, v in ph.available.items() if k not in new} != \
                {k: v for k, v in pt.available.items() if k not in new}:
            fails.append(f"R1 availability {ph.available} != {pt.available}")
        if [s.step for s in ph.solves] != [s.step for s in pt.solves]:
            fails.append("R1 solve steps differ")
        for a, b in zip(ph.solves, pt.solves):
            if not (np.array_equal(a.ee_franka, b.ee_franka)
                    and np.array_equal(a.ee_ur, b.ee_ur)):
                fails.append(f"R1 ee differ at step {a.step}")
                break
            if getattr(a, "belts", None) is not None and pt.decoder.frame == "world" \
                    and not np.array_equal(a.belts, b.belts):
                fails.append(f"R2 belts differ at step {a.step}")
                break
    if pt.decoder is not None:
        out["decoder_frame"] = pt.decoder.frame
        if pt.decoder.frame != "world" and pt.solves:  # R2 (pulley frame)
            errs = [chamfer_mm(s.belts[0], g) for s in pt.solves[::max(1, len(pt.solves) // 8)]
                    if (g := gt_belt(run, rec, s.step)) is not None]
            if errs:
                out["knot0_chamfer_mm"] = [round(float(f(errs)), 1) for f in (min, np.median, max)]
                if min(errs) > args.max_chamfer_mm:  # a frame error is ~270 mm on every solve
                    fails.append(f"R2 knot-0 chamfer min {min(errs):.1f} mm")
    try:  # R3
        steps = pt.solve_steps if len(pt.solve_steps) else np.zeros(1, np.int64)
        for st in np.linspace(steps[0], steps[-1], args.frames).astype(np.int64):
            pt._update(app, 0, int(st))
        out["drawn"] = app.server.scene.n
    except Exception:  # noqa: BLE001 - any render error fails R3
        fails.append("R3 render: " + traceback.format_exc(limit=2).splitlines()[-1][:160])
    out["fails"] = fails
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="*", default=None, help="run dirs or recordings dirs")
    ap.add_argument("--glob", default=None, help="extra run-dir glob (sorted, then --limit)")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--per-dir", type=int, default=1, help="runs per recordings dir")
    ap.add_argument("--deploy", type=Path, default=None)
    ap.add_argument("--decoder", type=Path, default=None)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--max-chamfer-mm", type=float, default=20.0)
    args = ap.parse_args()
    dirs = [REPO / d for d in (args.runs if args.runs is not None else DEFAULT_RUNS)]
    if args.glob:
        dirs += [Path(p) for p in sorted(glob.glob(str(REPO / args.glob)))]
    runs = []
    for d in dirs:
        if (d / "meta.json").is_file():
            runs.append(d)
        else:
            runs += sorted(p for p in d.iterdir() if (p / "meta.json").is_file())[:args.per_dir]
    runs = runs[:args.limit]
    with tempfile.TemporaryDirectory() as tmp:
        head = load_head(Path(tmp))
        from task_common import replay_learned_mpc as tree
        n_fail = n_ok = 0
        for r in runs:
            try:
                res = check_run(r, head, tree, args)
            except Exception as exc:  # noqa: BLE001 - one bad run must not stop the sweep
                res = {"run": str(r), "fails": [f"load: {exc!r}"[:200]]}
            if "skip" in res:
                print(f"[SKIP] {res['run']}: {res['skip']}")
                continue
            ok = not res["fails"]
            n_ok += ok
            n_fail += not ok
            extra = {k: res[k] for k in ("head", "n_x", "solves", "decoder_frame", "knot0_chamfer_mm",
                                         "drawn", "available") if k in res}
            print(f"[{'PASS' if ok else 'FAIL'}] {res['run']} {extra}")
            for f in res["fails"]:
                print(f"    {f}")
    print(f"{n_ok} pass, {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
