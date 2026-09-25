#!/usr/bin/env python3
"""Headless check of the replay viewer's learned-MPC layers (``LearnedMpcPanel``).

Drives one ``ReplayApp`` (no browser) over a learned, a baseline and a staged recording
(read-only; linked into a temp root) and checks the drawn geometry against the recorded debug
messages, the published plans, ``LatentDecoder`` and the demo belts.

Run:
    uv run python scripts/checks/check_replay_learned_mpc.py
    uv run python scripts/checks/check_replay_learned_mpc.py --skip-x4
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import newton

from round_belt_task.scene import build_scene
from task_common.latent_encoder import LatentDecoder
from task_common.recording import Recording
from task_common.replay_app import ReplayApp
from task_common.replay_learned_mpc import (
    ACTION_SCALE_DEFAULT,
    ARROW_SIDES,
    FRANKA_PLAN_CHANNEL,
    HEAD_LEN_M,
    HEAD_R_M,
    LAYERS,
    SHAFT_R_M,
    TARGET_ALPHA,
    TUBE_SIDES,
    U0_THICK,
    LearnedMpcPanel,
    any_learned_runs,
    arrow_vertices,
)
from task_common.scene import make_builder
from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material

sys.path.insert(0, str(Path(__file__).parent))
from private_url import pgrep_others, url_port

SET_TS = "20260924-223112"
LEARNED_SET = REPO_ROOT / f"data/lcs/mpc_eval/{SET_TS}-replayset-learned"
BASELINE_SET = REPO_ROOT / f"data/lcs/mpc_eval/{SET_TS}-replayset-baseline"
LEARNED_RUN = "set1/recordings/episode_gv_01_0"
STAGED_RUN = REPO_ROOT / "data/lcs/mpc_eval/20260925-002801-stages/recordings/stages_set1_gv_01"
STAGED_FRAMES = [13, 59]
STAGED_DEMO = "data/lcs/demo_flat/demo_episode.npz"
BASELINE_RUN = "set1/recordings/episode_gv_01_0"
REPLAY_PORT = 18095
LCM_URL = "udpm://239.255.76.105:7705?ttl=0"
KNOT_TOL = 1e-6  # m, float32 scene buffers vs float64 transforms
F32_TOL = 1e-6  # m, float32 rounding at ~1 m
EXACT_TOL = 1e-9
COS_MIN = 0.999999
REDRAW_LIMIT_MS = 50.0
N_TIMED_FRAMES = 200


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def build_model() -> newton.Model:
    builder = make_builder()
    build_scene(builder)
    return builder.finalize()


def _debug(rec: Recording) -> list:
    return [t for t in rec.targets if t.channel == "LEARNED_MPC_DEBUG"]


def _plan(rec: Recording, channel: str, t0: float) -> dict | None:
    for t in rec.targets:
        pos = t.payload.get("blocks", {}).get("end_effector_position_target")
        if t.channel == channel and pos and pos["t"] and round(pos["t"][0] * 1e6) == round(
                t0 * 1e6):
            return t.payload
    return None


def _ref_franka(rec: Recording, payload: dict) -> np.ndarray:
    # panda_link0 is welded at the world origin in this scene (asserted in X1).
    return np.asarray(payload["blocks"]["end_effector_position_target"]["data"]).T


def _arrow_errors(panel: LearnedMpcPanel, s, knots: dict[str, np.ndarray]) -> dict[str, float]:
    """Arrow base vs its knot, direction vs u_i, length vs scale*|u_i|, tip vs
    base + scale*u_i, f32 buffer vs f64."""
    err = dict.fromkeys(("base", "cos", "len", "tip", "f32", "chain"), 0.0)
    err["cos"] = 1.0
    arrows = {a[0]: a for a in panel.action_arrows(s)}
    for arm, rows in (("franka", slice(0, 3)), ("ur", slice(3, 6))):
        tips = []
        for i in range(s.n):
            name, base, vec, _, _, thick = arrows[f"actions/{arm}_{i}"]
            v64 = arrow_vertices(base, vec, thick)
            u = s.u_sol[rows, i]
            d = v64[-1] - v64[0]
            err["base"] = max(err["base"], float(np.abs(v64[0] - knots[arm][i]).max()))
            err["cos"] = min(err["cos"], float(d @ u / np.linalg.norm(d) / np.linalg.norm(u)))
            err["len"] = max(err["len"], abs(np.linalg.norm(d) / (
                panel.action_scale * np.linalg.norm(u)) - 1.0))
            err["tip"] = max(err["tip"], float(np.abs(
                v64[-1] - (knots[arm][i] + panel.action_scale * u)).max()))
            err["f32"] = max(err["f32"], float(np.abs(panel.handles[name].vertices - v64).max()))
            tips.append(v64[-1])
        if panel.action_scale == 1.0:
            err["chain"] = max(err["chain"], float(np.abs(
                np.array(tips[:-1]) - knots[arm][1:s.n]).max()))
    return err


def _arrow_dims(verts: np.ndarray) -> tuple[float, float, float]:
    """``(shaft radius, head length, head radius)`` measured from an arrow's vertices."""
    v, k = verts.astype(np.float64), ARROW_SIDES
    neck = v[1 + k:1 + 2 * k].mean(0)
    return (float(np.linalg.norm(v[1] - v[0])), float(np.linalg.norm(v[-1] - neck)),
            float(np.linalg.norm(v[1 + 2 * k] - neck)))


def _scene_nodes(app: ReplayApp) -> int:
    return len(getattr(app.server.scene, "_handle_from_node_name", {}))


def _centres(handle) -> np.ndarray:
    return handle.vertices.reshape(150, TUBE_SIDES, 3).mean(1)


def _visible(panel: LearnedMpcPanel, layer: str) -> list[str]:
    return [n for n, h in panel.handles.items() if n.startswith(f"{layer}/") and h.visible]


@check("X0 panel on a learned run, disabled on a baseline run")
def check_x0(ctx: SimpleNamespace) -> str:
    _require(any_learned_runs(ctx.learned_parent), "learned set not detected as learned")
    _require(not any_learned_runs(ctx.baseline_parent), "baseline set detected as learned")
    spec = importlib.util.spec_from_file_location("replay_viewer_cli",
                                                  REPO_ROOT / "scripts/replay_viewer.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base_args = cli.create_parser().parse_args(["--recordings", str(ctx.baseline_parent)])
    learned_args = cli.create_parser().parse_args(["--recordings", str(ctx.learned_parent),
                                                   "--learned-layers", "readout,actions,action_rotation"])
    _require(cli.learned_hooks(base_args) == [], "baseline root registers the panel")
    hooks = cli.learned_hooks(learned_args)
    _require(len(hooks) == 1 and hooks[0].requested == {"actions"},
             "learned root: panel not registered, layers not pre-enabled or removed names kept")
    over = cli.create_parser().parse_args(["--recordings", str(ctx.baseline_parent),
                                           "--decoder", "/nonexistent/decoder.npz"])
    _require(len(cli.learned_hooks(over)) == 1, "a CLI path does not register the panel")

    app, panel = ctx.app, ctx.panel
    app.select_run(ctx.learned_name)
    _require(all(panel.available[n] is None for n in LAYERS),
             f"learned run: layers unavailable {panel.available}")
    _require(all(not panel.boxes[n].disabled for n in LAYERS), "learned run: boxes disabled")
    _require(not panel.requested, "layers not default off")
    _require(panel.action_scale == ACTION_SCALE_DEFAULT == 9.0
             and panel.scale_slider.value == 9.0, "action scale not default 9")
    n = len(panel.solves)
    _require(n == len(_debug(app.recording)) and n > 0, f"{n} solves indexed")
    status = panel.status.content.splitlines()[0]
    app.select_run(ctx.baseline_name)
    _require(all(panel.available[n] for n in LAYERS) and all(
        panel.boxes[n].disabled for n in LAYERS), "baseline run: layers not disabled")
    for name in LAYERS:
        panel.set_layer(app, name, True)
    app.seek(app.frame_count // 2)
    shown = [k for k, h in panel.handles.items() if h.visible]
    _require(not shown, f"baseline run shows {shown}")
    for name in LAYERS:
        panel.set_layer(app, name, False)
    return (f"learned root registers (layers pre-enabled, readout/action_rotation dropped), "
            f"baseline root does not, CLI path does; action scale 9; learned run: {status}, {len(LAYERS)} layers enabled, default off; "
            f"baseline run: "
            f"disabled ({panel.available['planned_ee']!r}), nothing drawn")


@check("X1 per-frame geometry == recorded plan, decoder, demo")
def check_x1(ctx: SimpleNamespace) -> str:
    app, panel = ctx.app, ctx.panel
    app.select_run(ctx.learned_name)
    rec = app.recording
    labels = rec.meta["body_labels"]
    link0 = rec.body_q[:, labels.index("panda_arm/panda_link0")]
    _require(np.allclose(link0, [0, 0, 0, 0, 0, 0, 1]), "panda_link0 not at the world origin")
    for name in LAYERS:
        panel.set_layer(app, name, True)
    panel.set_action_scale(app, 1.0)  # chain check
    dbg = _debug(rec)
    steps = np.array([m.step for m in dbg])
    decoder = LatentDecoder.load(panel.paths["decoder"])
    demo = np.load(panel.paths["demo_episode"])["pcd_belt"]
    picks = [0, len(dbg) // 4, len(dbg) // 2, 3 * len(dbg) // 4, len(dbg) - 1]
    err = dict.fromkeys(("knot_f", "aug32", "aug64", "act"), 0.0)
    for i in picks:
        frame = min(rec.frame_at_step(int(dbg[i].step)) + 1, rec.frame_count - 1)
        app.seek(frame)
        want = int(np.searchsorted(steps, app.current_step, side="right")) - 1
        _require(panel.current == want, f"frame {frame}: picked {panel.current}, want {want}")
        msg = dbg[want].payload
        x = np.asarray(msg["blocks"]["x_sol"]["data"])
        u = np.asarray(msg["blocks"]["u_sol"]["data"])
        t0 = msg["blocks"]["x_sol"]["t"][0]
        h = panel.handles
        franka = _plan(rec, FRANKA_PLAN_CHANNEL, t0)
        _require(franka is not None, f"solve {want}: no plan with t0 {t0}")
        err["knot_f"] = max(err["knot_f"], float(np.abs(
            h["planned_ee/franka_knots"].points - _ref_franka(rec, franka)).max()))
        aug = np.concatenate([x[16:19].T, x[19:22].T])
        err["aug32"] = max(err["aug32"], float(np.abs(
            h["planned_ee/augmented"].points - aug).max()))
        s = panel.solves[want]
        err["aug64"] = max(err["aug64"], float(np.abs(
            np.concatenate([s.ee_franka, s.ee_ur]) - aug).max()))
        planned = np.stack([decoder.decode(z) for z in x[:16, 1:].T])
        _require(np.array_equal(s.belts[1:], planned),
                 f"solve {want}: planned belts != decode(x_sol[:16, 1:])")
        rings = np.stack([h[f"planned_belt/step_{i}"].vertices.reshape(150, TUBE_SIDES, 3)
                          .mean(1) for i in range(1, len(planned) + 1)])
        _require(np.abs(rings - planned).max() <= F32_TOL,
                 f"solve {want}: planned tube centres off by {np.abs(rings - planned).max():.1e}")
        _require(_visible(panel, "target_belt") == ["target_belt/goal_0"],
                 f"solve {want}: target tubes {_visible(panel, 'target_belt')}")
        _require(np.abs(_centres(h["target_belt/goal_0"]) - demo[-1]).max() <= F32_TOL
                 and h["target_belt/goal_0"].opacity == TARGET_ALPHA[0],
                 f"solve {want}: target tube != opaque pcd_belt[-1]")
        _require(np.array_equal(s.u_sol, u), f"solve {want}: u_sol != message")
        a = _arrow_errors(panel, s, {"franka": _ref_franka(rec, franka), "ur": x[19:22].T})
        _require(a["base"] <= KNOT_TOL and a["cos"] > COS_MIN and a["len"] <= EXACT_TOL
                 and a["tip"] <= KNOT_TOL and a["f32"] <= F32_TOL and a["chain"] <= EXACT_TOL,
                 f"solve {want}: arrows {a}")
        err["act"] = max(err["act"], a["base"])
        err["cos"] = min(err.get("cos", 1.0), a["cos"])
    _require(err["knot_f"] <= KNOT_TOL, f"knots {err['knot_f']:.1e} > {KNOT_TOL:g}")
    _require(sorted(n for n in panel.handles if n.startswith("planned_ee/"))
             == ["planned_ee/augmented", "planned_ee/franka_knots"],
             f"planned_ee handles {sorted(panel.handles)}")
    _require(err["aug32"] <= F32_TOL and err["aug64"] <= EXACT_TOL,
             f"augmented {err['aug32']:.1e} (f32) / {err['aug64']:.1e} (f64)")
    app.seek(0)
    before = panel.current == -1 and not any(h.visible for h in panel.handles.values())
    _require(before, "before the first solve: layers not hidden")
    panel.set_action_scale(app, ACTION_SCALE_DEFAULT)
    staged = _check_staged(ctx)
    return (f"{len(picks)} frames: picked solve == latest step <= frame step; Franka knots "
            f"{err['knot_f']:.1e} m, no connecting paths; augmented f32 "
            f"{err['aug32']:.1e} / f64 {err['aug64']:.1e} m; decoded belts bit-exact, one target tube == pcd_belt[{len(demo) - 1}];"
            f" arrows start at their knots ({err['act']:.1e} m), chain tip-to-tail, "
            f"min cos {err['cos']:.9f}, length rel <= {EXACT_TOL:g}; frame 0 hidden; {staged}")


def _check_staged(ctx: SimpleNamespace) -> str:
    app, panel = ctx.app, ctx.panel
    app.select_run(ctx.staged_name)
    _require(panel.staged and panel.goal_frames.tolist() == STAGED_FRAMES
             and panel.paths["demo_episode"] == REPO_ROOT / STAGED_DEMO,
             f"staged goals {panel.goal_frames} from {panel.paths['demo_episode']}")
    panel.set_layer(app, "target_belt", True)
    demo = np.load(panel.paths["demo_episode"])["pcd_belt"]
    names = [f"target_belt/goal_{i}" for i in range(len(STAGED_FRAMES))]
    seen = set()
    for f in range(0, app.frame_count, 5):
        app.seek(f)
        if panel.current < 0:
            continue
        stage = int(panel.solves[panel.current].scalars["stage"])
        seen.add(stage)
        _require(_visible(panel, "target_belt") == names, f"frame {f}: {_visible(panel, 'target_belt')}")
        for i, name in enumerate(names):
            h = panel.handles[name]
            _require(np.abs(_centres(h) - demo[STAGED_FRAMES[i]]).max() <= F32_TOL,
                     f"{name} centres != pcd_belt[{STAGED_FRAMES[i]}]")
            _require(h.opacity == TARGET_ALPHA[0 if i == stage else 1],
                     f"frame {f} stage {stage}: {name} opacity {h.opacity}")
    _require(seen == {0, 1}, f"stages seen {seen}")
    _require(not any("/ref_" in n for n in panel.handles), "reference tubes exist")
    panel.set_layer(app, "target_belt", False)
    return (f"staged run: goals {STAGED_FRAMES} of {STAGED_DEMO}, tubes == pcd_belt, opacity "
            f"follows stage (0 and 1 seen), no ref tubes")


@check("X2 toggling keeps the handle count; redraw time")
def check_x2(ctx: SimpleNamespace) -> str:
    app, panel = ctx.app, ctx.panel
    app.select_run(ctx.learned_name)
    mid = app.recording.frame_at_step(int(panel.solves[len(panel.solves) // 2].step))
    for name in LAYERS:
        panel.set_layer(app, name, True)
    app.seek(mid)
    full = (len(panel.handles), _scene_nodes(app))
    for name in LAYERS:
        panel.set_layer(app, name, False)
        _require(not _visible(panel, name), f"{name} still visible when off")
        panel.set_layer(app, name, True)
        _require(_visible(panel, name), f"{name} not visible when on")
    for name in LAYERS:
        panel.set_layer(app, name, False)
    off = (len(panel.handles), _scene_nodes(app))
    _require(not any(h.visible for h in panel.handles.values()), "all off: something visible")
    for name in LAYERS:
        panel.set_layer(app, name, True)
    again = (len(panel.handles), _scene_nodes(app))
    _require(full == off == again, f"handles/nodes {full} -> off {off} -> on {again}")
    s = panel.solves[panel.current]
    knots = {"franka": panel.action_bases(s)[0], "ur": s.ee_ur}
    dims, n_full = {}, 0
    for scale in (1.0, 20.0):
        panel.set_action_scale(app, scale)
        a = _arrow_errors(panel, s, knots)
        _require(a["base"] <= EXACT_TOL and a["cos"] > COS_MIN and a["len"] <= EXACT_TOL
                 and a["tip"] <= F32_TOL and a["f32"] <= F32_TOL, f"action scale {scale:g}: {a}")
        _require((len(panel.handles), _scene_nodes(app)) == full,
                 f"handles changed at scale {scale:g}")
        for name, _, vec, _, _, thick in panel.action_arrows(s):
            length = float(np.linalg.norm(vec))
            got = _arrow_dims(panel.handles[name].vertices)
            want = (SHAFT_R_M * thick, HEAD_LEN_M, HEAD_R_M * thick)
            if length < HEAD_LEN_M:  # head only, shrunk to the length
                want = tuple(w * length / HEAD_LEN_M for w in want)
            _require(np.allclose(got, want, atol=F32_TOL, rtol=0), f"{name} scale {scale:g}: "
                     f"dims {got} != {want}")
            dims.setdefault(name, []).append((length >= HEAD_LEN_M, got))
    for name, ((full1, d1), (full20, d20)) in dims.items():
        if full1 and full20:
            n_full += 1
            _require(np.allclose(d1, d20, atol=F32_TOL, rtol=0), f"{name}: dims {d1} -> {d20}")
    _require(n_full > 0, "no arrow at full size at both scales")
    u0 = _arrow_dims(panel.handles["actions/franka_0"].vertices)
    _require(abs(u0[0] - U0_THICK * SHAFT_R_M) <= F32_TOL, f"u0 shaft {u0[0]}")
    start = time.perf_counter()
    panel.set_action_scale(app, 1.0)
    rebuild_ms = 1e3 * (time.perf_counter() - start)
    first = app.recording.frame_at_step(int(panel.solves[0].step))
    frames = range(first, min(first + N_TIMED_FRAMES, app.frame_count))
    panel_ms, render_ms = [], []
    for f in frames:
        start = time.perf_counter()
        app.seek(f)
        render_ms.append(1e3 * (time.perf_counter() - start))
        panel_ms.append(panel.last_frame_ms)
    p, r = np.asarray(panel_ms), np.asarray(render_ms)
    _require(p.mean() < REDRAW_LIMIT_MS and r.mean() < REDRAW_LIMIT_MS,
             f"panel {p.mean():.1f} ms / render {r.mean():.1f} ms mean > {REDRAW_LIMIT_MS:g}")
    return (f"{full[0]} panel handles / {full[1]} scene nodes, unchanged over toggles and scale; "
            f"scales 1/20: tip == base + scale*u, shaft/head constant ({n_full} full-size arrows "
            f"equal at both, short ones head-only), u0 x{U0_THICK:g}; arrow rebuild on a scale change "
            f"{rebuild_ms:.1f} ms; {len(p)} consecutive frames, all layers: panel mean "
            f"{p.mean():.1f} / "
            f"max {p.max():.1f} ms, whole redraw mean {r.mean():.1f} / p95 "
            f"{np.percentile(r, 95):.1f} ms")


@check("X3 missing files degrade cleanly")
def check_x3(ctx: SimpleNamespace) -> str:
    app = ctx.app
    app.select_run(ctx.learned_name)
    missing = Path(ctx.tmp_root) / "missing"
    panel = LearnedMpcPanel(deploy=missing / "deploy.npz", demo_goals=missing / "goals.npz",
                            demo_episode=missing / "demo.npz", layers=LAYERS,
                            root="/replay/learned_mpc_x3")
    app.add_hook(panel)
    for f in (0, app.frame_count // 2, app.frame_count - 1):
        app.seek(f)
    off = {n for n in LAYERS if panel.available[n]}
    _require(off == {"planned_belt", "target_belt"}, f"unavailable layers {off}")
    _require(all(panel.boxes[n].disabled for n in off), "boxes of missing files not disabled")
    _require(all("missing" in panel.available[n] for n in off), f"reasons {panel.available}")
    _require("missing" in panel.status.content, "status lacks the reason")
    _require(not _visible(panel, "planned_belt") and not _visible(panel, "target_belt"),
             "layers with missing files drawn")
    _require(_visible(panel, "planned_ee") and _visible(panel, "actions"),
             "remaining layers not drawn")
    for name in LAYERS:
        panel.set_layer(app, name, False)
    ctx.app.hooks.remove(panel)
    return (f"decoder/demo missing: {sorted(off)} disabled ({panel.available['planned_belt']!r});"
            " EE and actions still drawn, no exception")


@check("X4 check_replay_viewer.py V0-V10")
def check_x4(ctx: SimpleNamespace) -> str:
    if ctx.args.skip_x4:
        return "skipped (--skip-x4)"
    left = pgrep_others(url_port(ctx.args.lcm_url))
    _require(not left, f"processes already on {ctx.args.lcm_url}: {left}")
    runner = ("import importlib.util, sys; sys.argv = [sys.argv[1]]; "
              "sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[0]).parent)); "
              "spec = importlib.util.spec_from_file_location('check_under_test', sys.argv[0]); "
              "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
              f"mod.PRIVATE_LCM_URL = {ctx.args.lcm_url!r}; sys.exit(mod.main())")
    script = REPO_ROOT / "scripts/checks/check_replay_viewer.py"
    out = subprocess.run([sys.executable, "-c", runner, str(script)], cwd=REPO_ROOT,
                         capture_output=True, text=True, timeout=600, check=False)
    passes = [line for line in out.stdout.splitlines() if line.startswith("[PASS]")]
    _require(out.returncode == 0 and "ALL REPLAY CHECKS PASSED" in out.stdout,
             f"exit {out.returncode}: {(out.stderr or out.stdout)[-800:]}")
    return f"{len(passes)} PASS on {ctx.args.lcm_url}"


def _link_runs(tmp_root: Path, args: argparse.Namespace) -> tuple[str, str, str]:
    learned, baseline = args.learned / LEARNED_RUN, args.baseline / BASELINE_RUN
    for path in (learned, baseline, args.staged):
        if not (path / "meta.json").is_file():
            raise FileNotFoundError(f"{path}: no recording")
    names = ("learned_" + learned.name, "baseline_" + baseline.name, "staged_" + args.staged.name)
    for name, path in zip(names, (learned, baseline, args.staged), strict=True):
        (tmp_root / name).symlink_to(path.resolve(), target_is_directory=True)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--learned", type=Path, default=LEARNED_SET,
                        help="learned replay-set dir ({set1,set3,nominal}/recordings)")
    parser.add_argument("--baseline", type=Path, default=BASELINE_SET)
    parser.add_argument("--staged", type=Path, default=STAGED_RUN, help="staged-goal run dir")
    parser.add_argument("--port", type=int, default=REPLAY_PORT, help="headless viser port")
    parser.add_argument("--lcm-url", default=LCM_URL, help="private LCM group for X4")
    parser.add_argument("--skip-x4", action="store_true")
    args = parser.parse_args()
    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_replay_learned_"))
    app, exit_code = None, 0
    try:
        learned_name, baseline_name, staged_name = _link_runs(tmp_root, args)
        patch_viewer_shape_names()
        patch_viser_texture_material()
        panel = LearnedMpcPanel()
        app = ReplayApp(tmp_root, build_model, port=args.port, verbose=False, analysis=False,
                        hooks=[panel], run=baseline_name)
        ctx = SimpleNamespace(app=app, panel=panel, args=args, tmp_root=tmp_root,
                              learned_name=learned_name, baseline_name=baseline_name,
                              staged_name=staged_name,
                              learned_parent=args.learned / "set1/recordings",
                              baseline_parent=args.baseline / "set1/recordings")
        for name, fn in CHECKS:
            try:
                detail = fn(ctx)
            except AssertionError as exc:
                print(f"[FAIL] {name}: {exc}", file=sys.stderr)
                exit_code = 1
                continue
            except Exception as exc:  # noqa: BLE001 - report, then continue
                print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
                traceback.print_exc()
                exit_code = 1
                continue
            print(f"[PASS] {name}: {detail}", flush=True)
    finally:
        if app is not None:
            app.close()
        shutil.rmtree(tmp_root, ignore_errors=True)
    if exit_code == 0:
        print(f"ALL LEARNED-MPC REPLAY CHECKS PASSED ({time.perf_counter() - t0:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
