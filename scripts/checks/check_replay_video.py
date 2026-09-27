#!/usr/bin/env python3
"""Check of the MP4 export (``scripts/record_replay_video.py``) and the viewer's camera line.

V0: an in-process ``ReplayApp`` + headless Chromium: captures right after ``sync_scene`` equal
settled ones; the Display "Camera" block follows the client camera, and a real click on its
"Copy camera JSON" puts that json on the clipboard (DevTools); the video panel draws full tubes
(stride 1, 8 sides).
V1: the CLI renders 10 frames of a learned run with that copied json as ``--camera``: count, size,
non-blank, consecutive frames differ while playing. V2: a rerun gives identical frames.
V3: ``--compare`` stacks two runs side by side. V4: the colour legend is drawn by default
(bottom-right) and ``--no-legend`` removes it, leaving the rest of the frame identical.
V5: ``--target-belt`` (off by default) with the run's own goal belts renders V1 bit-exactly; a
shifted belt changes the frames. V6: two ``--inset`` views: frame size kept, the inset windows
differ from V1 while the rest of the frame does not, and a rerun is bit-identical.
V7: the blank-render guard: a real WebGL context loss mid-render (DevTools) restarts Chromium
and re-renders that frame (same frames as a clean pass); a render that stays blank raises
``RenderFailed`` after the retries, and the CLI exits 1 without writing the video.

Run (no LCM needed; ports PORT, PORT+1 and DevTools PORT+2):
    uv run python scripts/checks/check_replay_video.py --port 19295
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import record_replay_video as rv

from task_common import replay_video
from task_common.replay_app import ReplayApp
from task_common.replay_video import (
    DEFAULT_CAMERA,
    CameraPose,
    HeadlessBrowser,
    RenderFailed,
    capture,
    probe,
    read_frames,
    render_run,
    sync_scene,
    warm_up,
)
from utils.viewer_patches import (
    patch_viewer_shape_names,
    patch_viser_texture_material,
)

RECORDINGS = REPO_ROOT / "data/lcs/mpc_eval/20260925-203831-eeoff/recordings"
RUN, OTHER = "learned_ee_ur_y-8", "baseline_ee_ur_y-8"
DECODER = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v2_20260925/"
               "deploy_v2_decoded_only/decoder.npz")
LAYERS = "planned_belt,planned_ee,actions,target_belt"
PORT = 19295
W, H = 640, 360
N_FRAMES, FPS, START_S = 10, 25.0, 3.0
TEST_POSE = {"position": (0.52, -0.55, 0.48), "look_at": (0.45, -0.02, 0.1)}
OVERLAY_ROWS = 60  # the time text sits in the bottom-left corner
INSETS = (
    'side={"position":[0.6904,-0.0038,0.0404],"look_at":[0.2459,0.0171,0.0321],"fov":1.309}',
    'front={"position":[0.4517,-0.2212,0.0449],"look_at":[0.3618,0.422,0.0227],"fov":1.309}',
)


def common_args(args: argparse.Namespace) -> list[str]:
    return ["--recordings", str(args.recordings), "--decoder", str(args.decoder),
            "--learned-layers", LAYERS]


def check_camera_line(args: argparse.Namespace) -> tuple[str, str]:
    cli = rv.create_parser().parse_args([*common_args(args), "--run", RUN, "--out", "x.mp4"])
    hooks = rv.video_hooks(cli)
    assert len(hooks) == 1, "no learned panel"
    panel = hooks[0]
    assert (panel.tube_stride, panel.tube_sides) == (1, 8), \
        f"video tubes stride {panel.tube_stride} sides {panel.tube_sides}, want 1 / 8"
    patch_viewer_shape_names()
    patch_viser_texture_material()
    app = ReplayApp(args.recordings, rv.build_model, port=args.port, run=RUN, hooks=hooks,
                    verbose=False, analysis=False)
    try:
        app.seek(app.recording.frame_at_time(app._frame_time(0) + START_S))
        tube = panel.handles.get("planned_belt/step_1")
        assert tube is not None and tube.visible, "no planned belt tube drawn"
        n_belt = panel.solves[panel.current].belts.shape[1]
        assert tube.vertices.shape == (n_belt * 8, 3), \
            f"tube has {tube.vertices.shape[0]} vertices, want {n_belt} x 8"
        with HeadlessBrowser(app.server, f"http://127.0.0.1:{args.port}", W, H,
                             debug_port=args.port + 2) as client:
            warm_up(app, client, DEFAULT_CAMERA, W, H)
            # The fence: a capture right after sync_scene equals one after the scene settled.
            t0, previous = app._frame_time(0), None
            for t in np.linspace(1.0, 9.0, 5):
                app.seek(app.recording.frame_at_time(t0 + t))
                sync_scene(app.server, client)
                fenced = capture(client, DEFAULT_CAMERA, W, H)
                time.sleep(0.3)
                settled = capture(client, DEFAULT_CAMERA, W, H)
                assert np.array_equal(fenced, settled), f"t={t:g}: fenced capture is stale"
                assert previous is None or not np.array_equal(fenced, previous), \
                    f"t={t:g}: same image as the previous seek"
                previous = fenced
            client.camera.position = TEST_POSE["position"]
            client.camera.look_at = TEST_POSE["look_at"]
            client.camera.up_direction = (0.0, 0.0, 1.0)
            deadline, text = time.monotonic() + 20.0, None
            while time.monotonic() < deadline:
                text = app.camera_text(client.client_id)
                if text and CameraPose.load(text).position == TEST_POSE["position"]:
                    break
                time.sleep(0.1)
            assert text, "no Camera line for the client"
            pose = CameraPose.load(text)
            assert pose.position == TEST_POSE["position"], f"camera line {text}"
            assert np.allclose(pose.look_at, TEST_POSE["look_at"], atol=1e-4), f"{text}"
            assert np.isclose(pose.fov, client.camera.fov, atol=1e-4), f"{text}"
            copied = click_copy_button(args.port + 2, f"http://127.0.0.1:{args.port}")
            assert copied == text, f"clipboard {copied!r} != camera line {text!r}"
    finally:
        app.close()
    return text, (f"line {text}; tubes {n_belt} pts x 8 sides "
                  f"({tube.vertices.shape[0]} vertices); 5 fenced captures == settled; "
                  "Copy button -> clipboard == line")


def click_copy_button(debug_port: int, origin: str) -> str:
    """Real mouse click on "Copy camera JSON" over the DevTools protocol; the clipboard text."""
    pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json").read())
    page = next(p for p in pages if p["type"] == "page")
    with connect(page["webSocketDebuggerUrl"], max_size=2**24) as ws:
        ids = iter(range(1, 10**6))

        def cdp(method: str, **params) -> dict:
            i = next(ids)
            ws.send(json.dumps({"id": i, "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv(timeout=30))
                if msg.get("id") == i:
                    assert "error" not in msg, f"{method}: {msg['error']}"
                    return msg["result"]

        def js(expr: str) -> object:
            out = cdp("Runtime.evaluate", expression=expr, awaitPromise=True,
                      returnByValue=True)
            assert "exceptionDetails" not in out, f"{expr}: {out['exceptionDetails']}"
            return out["result"].get("value")

        cdp("Browser.grantPermissions", origin=origin,
            permissions=["clipboardReadWrite", "clipboardSanitizedWrite"])
        cdp("Emulation.setFocusEmulationEnabled", enabled=True)
        js("navigator.clipboard.writeText('before')")
        box = js("(() => { const b = [...document.querySelectorAll('button')].find("
                 "e => e.textContent === 'Copy camera JSON'); if (!b) return null; "
                 "b.scrollIntoView({block: 'center'}); const r = b.getBoundingClientRect(); "
                 "return [r.x + r.width / 2, r.y + r.height / 2]; })()")
        assert box, "no Copy camera JSON button on the page"
        for kind in ("mousePressed", "mouseReleased"):
            cdp("Input.dispatchMouseEvent", type=kind, x=box[0], y=box[1], button="left",
                clickCount=1)
        deadline = time.monotonic() + 5.0
        while (text := js("navigator.clipboard.readText()")) == "before":
            assert time.monotonic() < deadline, "click did not change the clipboard"
            time.sleep(0.1)
        return text


def render(args: argparse.Namespace, out: Path, camera: str, log: Path,
           extra: tuple[str, ...] = ()) -> dict:
    step = 1.0 / FPS
    cmd = [sys.executable, str(REPO_ROOT / "scripts/record_replay_video.py"),
           *common_args(args), "--run", RUN, "--out", str(out), "--camera", camera,
           "--width", str(W), "--height", str(H), "--fps", f"{FPS:g}",
           "--start-s", f"{START_S:g}", "--end-s", f"{START_S + (N_FRAMES - 1) * step:.4f}",
           "--port", str(args.port + 1), *extra]
    with log.open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=600,
                            check=False).returncode
    assert rc == 0, f"record_replay_video.py exited {rc}; see {log}"
    return probe(out)


def check_video(args: argparse.Namespace, tmp: Path, camera: str) -> tuple[np.ndarray, str]:
    out = tmp / "v1.mp4"
    info = render(args, out, camera, tmp / "v1.log")
    assert info["frames"] == N_FRAMES, f"{info['frames']} frames, want {N_FRAMES}"
    assert (info["width"], info["height"]) == (W, H), f"size {info['width']}x{info['height']}"
    frames = read_frames(out, W, H)
    assert len(frames) == N_FRAMES
    std = frames.reshape(N_FRAMES, -1).std(1)
    drawn = (frames < 235).any(-1).mean((1, 2))  # not background white
    assert std.min() > 20 and drawn.min() > 0.2, f"blank frame: std {std}, drawn {drawn}"
    scene = frames[:, : H - OVERLAY_ROWS].astype(np.int16)
    diff = np.abs(np.diff(scene, axis=0)).max(-1)
    changed = (diff > 12).mean((1, 2))
    assert changed.min() > 1e-4, f"consecutive frames identical while playing: {changed}"
    return frames, (f"{info['frames']} frames {W}x{H}, {info['duration_s']:.2f} s; std >= "
                    f"{std.min():.0f}, drawn >= {drawn.min():.0%}, changed pixels per step "
                    f"{changed.min():.2%}-{changed.max():.2%}")


def check_deterministic(args: argparse.Namespace, tmp: Path, camera: str,
                        first: np.ndarray) -> str:
    render(args, tmp / "v2.mp4", camera, tmp / "v2.log")
    again = read_frames(tmp / "v2.mp4", W, H)
    assert again.shape == first.shape, f"shape {again.shape} vs {first.shape}"
    worst = int(np.abs(again.astype(np.int16) - first).max())
    assert worst == 0, f"rerun differs: max pixel diff {worst}"
    return "rerun bit-identical"


def check_compare(args: argparse.Namespace, tmp: Path, camera: str) -> str:
    info = render(args, tmp / "v3.mp4", camera, tmp / "v3.log", ("--compare", OTHER))
    assert (info["width"], info["height"]) == (2 * W, H), f"{info['width']}x{info['height']}"
    assert info["frames"] == N_FRAMES, f"{info['frames']} frames"
    frames = read_frames(tmp / "v3.mp4", 2 * W, H)
    halves = frames[:, : H - OVERLAY_ROWS, :W], frames[:, : H - OVERLAY_ROWS, W:]
    assert not np.array_equal(*halves), "both halves identical"
    return f"{info['frames']} frames {2 * W}x{H} ({RUN} | {OTHER})"


def check_legend(args: argparse.Namespace, tmp: Path, camera: str,
                 first: np.ndarray) -> str:
    render(args, tmp / "v4.mp4", camera, tmp / "v4.log", ("--no-legend",))
    bare = read_frames(tmp / "v4.mp4", W, H)
    assert bare.shape == first.shape, f"shape {bare.shape} vs {first.shape}"
    # In every frame: the legend is static; x264 noise elsewhere is not.
    changed = (np.abs(first.astype(np.int16) - bare).max(-1) > 12).all(0)
    corner = np.zeros_like(changed)
    corner[H // 2:, W // 3:] = True
    inside = int((changed & corner).sum())
    assert inside > 0.02 * W * H, f"no legend: {inside} px differ with --no-legend"
    assert inside >= 0.98 * changed.sum(), \
        f"legend outside the bottom-right: {changed.sum() - inside} of {changed.sum()} px"
    ys, xs = np.nonzero(changed & corner)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    assert "legend:" in (tmp / "v1.log").read_text(), "no legend line in the V1 log"
    return f"legend box x {box[0]}-{box[2]}, y {box[1]}-{box[3]}; --no-legend removes it"


def check_target_belt(args: argparse.Namespace, tmp: Path, camera: str,
                      first: np.ndarray) -> str:
    base = rv.create_parser().parse_args([*common_args(args), "--run", RUN, "--out", "x.mp4"])
    assert rv.video_hooks(base)[0].target_belt is None, "target_belt set by default"
    meta = json.loads((args.recordings / RUN / "meta.json").read_text())["learned_mpc"]
    with np.load(meta["deploy"]) as d:
        goal = np.asarray(d["goal_frames"], dtype=np.int64)
    belts = np.load(meta["demo_episode"])["pcd_belt"][goal].astype(np.float32)
    np.savez(tmp / "same.npz", pcd_belt_stage=belts)
    np.savez(tmp / "shift.npz", pcd_belt_stage=belts + np.float32(0.03))
    cli = rv.create_parser().parse_args([*common_args(args), "--run", RUN, "--out", "x.mp4",
                                         "--target-belt", str(tmp / "same.npz")])
    assert rv.video_hooks(cli)[0].target_belt == tmp / "same.npz", "flag not passed"
    render(args, tmp / "v5a.mp4", camera, tmp / "v5a.log", ("--target-belt", str(tmp / "same.npz")))
    same = read_frames(tmp / "v5a.mp4", W, H)
    worst = int(np.abs(same.astype(np.int16) - first).max())
    assert worst == 0, f"same belts via --target-belt differ from the meta path: {worst}"
    render(args, tmp / "v5b.mp4", camera, tmp / "v5b.log",
           ("--target-belt", str(tmp / "shift.npz")))
    moved = read_frames(tmp / "v5b.mp4", W, H)
    changed = (np.abs(moved.astype(np.int16) - first).max(-1) > 12).mean((1, 2))
    assert changed.min() > 1e-3, f"shifted target belt not drawn: {changed}"
    return (f"default off; goals {goal.tolist()} via --target-belt == meta path (bit-exact); "
            f"+30 mm shift changes {changed.min():.2%}-{changed.max():.2%} of pixels")


def check_insets(args: argparse.Namespace, tmp: Path, camera: str,
                 first: np.ndarray) -> str:
    extra = tuple(x for text in INSETS for x in ("--inset", text))
    frames = []
    for tag in ("v6a", "v6b"):
        info = render(args, tmp / f"{tag}.mp4", camera, tmp / f"{tag}.log", extra)
        assert (info["width"], info["height"], info["frames"]) == (W, H, N_FRAMES), f"{info}"
        frames.append(read_frames(tmp / f"{tag}.mp4", W, H))
    worst = int(np.abs(frames[1].astype(np.int16) - frames[0]).max())
    assert worst == 0, f"inset rerun differs: max pixel diff {worst}"
    line = re.search(r"insets .*? at (\[.*\])", (tmp / "v6a.log").read_text())
    assert line, "no insets line in the V6 log"
    rects = ast.literal_eval(line.group(1))
    assert (len(rects) == 2 and rects[0][1] < rects[0][3] <= rects[1][1] < rects[1][3] < 0.7 * H
            and all(x1 == W - 12 for _, _, x1, _ in rects)), f"insets not stacked top-right: {rects}"
    changed = np.abs(frames[0].astype(np.int16) - first).max(-1) > 12
    mask = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in rects:
        inside = changed[:, y0:y1, x0:x1].mean((1, 2))
        assert inside.min() > 0.2, f"inset {x0},{y0} matches the plain render: {inside}"
        mask[y0:y1, x0:x1] = True
    outside = changed[:, ~mask].mean(1)
    assert outside.max() < 0.01, f"inset renders changed the main view: {outside}"
    return (f"{N_FRAMES} frames {W}x{H}; windows {rects}; inset pixels changed >= "
            f"{inside.min():.0%}, outside <= {outside.max():.2%}; rerun bit-identical")


def devtools_js(debug_port: int, expr: str) -> object:
    pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json").read())
    page = next(p for p in pages if p["type"] == "page")
    with connect(page["webSocketDebuggerUrl"], max_size=2**24) as ws:
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
            "expression": expr, "returnByValue": True}}))
        while (msg := json.loads(ws.recv(timeout=30))).get("id") != 1:
            pass
    assert "exceptionDetails" not in msg["result"], f"{expr}: {msg['result']}"
    return msg["result"]["result"].get("value")


LOSE_CONTEXT = ("document.querySelector('canvas[data-engine]').getContext('webgl2')"
                ".getExtension('WEBGL_lose_context').loseContext() ?? 'lost'")


def check_guard(args: argparse.Namespace, tmp: Path) -> str:
    cli = rv.create_parser().parse_args([*common_args(args), "--run", RUN, "--out", "x.mp4"])
    app = ReplayApp(args.recordings, rv.build_model, port=args.port, run=RUN,
                    hooks=rv.video_hooks(cli), verbose=False, analysis=False)
    times, real_capture = START_S + np.arange(4) / FPS, replay_video.capture
    try:
        browser = HeadlessBrowser(app.server, f"http://127.0.0.1:{args.port}", W, H,
                                  debug_port=args.port + 2)
        with browser:
            warm_up(app, browser.client, DEFAULT_CAMERA, W, H)
            clean: list[np.ndarray] = []
            # First pass discarded: a newly added point cloud can draw once at a stale size.
            render_run(app, browser, DEFAULT_CAMERA, times, W, H, lambda _f: None)
            render_run(app, browser, DEFAULT_CAMERA, times, W, H, clean.append)
            assert browser.restarts == 0, "clean frames tripped the guard"
            got: list[np.ndarray] = []

            def lose_after_two(frame: np.ndarray) -> None:
                got.append(frame)
                if len(got) == 2:
                    devtools_js(args.port + 2, LOSE_CONTEXT)

            render_run(app, browser, DEFAULT_CAMERA, times, W, H, lose_after_two)
            assert browser.restarts == 1, f"{browser.restarts} restarts after a context loss"
            assert len(got) == len(times) and all(
                np.array_equal(a, b) for a, b in zip(got, clean, strict=True)), \
                "recovered frames differ from the clean pass"
            replay_video.capture = lambda _c, _cam, w, h, **_k: np.full((h, w, 3), 255, np.uint8)
            sunk: list[np.ndarray] = []
            try:
                render_run(app, browser, DEFAULT_CAMERA, times, W, H, sunk.append, retries=2)
                raise AssertionError("an always-blank render did not raise")
            except RenderFailed as exc:
                failed = str(exc)
            assert not sunk and browser.restarts == 3, f"{len(sunk)} frames, {browser.restarts}"
    finally:
        replay_video.capture = real_capture
        app.close()
    out = tmp / "v7.mp4"
    cli = rv.create_parser().parse_args([*common_args(args), "--run", RUN, "--out", str(out),
                                         "--width", str(W), "--height", str(H),
                                         "--start-s", f"{START_S:g}", "--end-s",
                                         f"{START_S + 0.1:g}", "--port", str(args.port + 1)])
    replay_video.capture = lambda _c, _cam, w, h, **_k: np.full((h, w, 3), 255, np.uint8)
    try:
        rc = rv.main(cli)
    finally:
        replay_video.capture = real_capture
    assert rc == 1 and not out.exists(), f"always-blank CLI exited {rc}, wrote {out.exists()}"
    return (f"context loss after frame 2 -> 1 restart, {len(times)} frames == clean pass; "
            f"always blank -> {failed!r}; CLI exit 1, no file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=PORT,
                        help="viser PORT, PORT+1; DevTools PORT+2")
    parser.add_argument("--recordings", type=Path, default=RECORDINGS)
    parser.add_argument("--decoder", type=Path, default=DECODER)
    args = parser.parse_args()
    rv.quiet_websockets()
    for path in (args.recordings / RUN, args.recordings / OTHER, args.decoder):
        if not path.exists():
            print(f"[FAIL] setup: missing {path}", file=sys.stderr)
            return 1
    start, failed = time.perf_counter(), 0
    with tempfile.TemporaryDirectory(prefix="check_replay_video_") as tmp:
        tmp = Path(tmp)
        state: dict = {}
        checks = [
            ("V0 camera line + tubes", lambda: _v0(args, state)),
            ("V1 render 10 frames", lambda: _v1(args, tmp, state)),
            ("V2 deterministic", lambda: check_deterministic(args, tmp, state["camera"],
                                                             state["frames"])),
            ("V3 side by side", lambda: check_compare(args, tmp, state["camera"])),
            ("V4 legend", lambda: check_legend(args, tmp, state["camera"], state["frames"])),
            ("V5 target belt", lambda: check_target_belt(args, tmp, state["camera"],
                                                         state["frames"])),
            ("V6 insets", lambda: check_insets(args, tmp, state["camera"], state["frames"])),
            ("V7 blank-render guard", lambda: check_guard(args, tmp)),
        ]
        for name, fn in checks:
            try:
                detail = fn()
            except Exception as exc:  # noqa: BLE001 - report and go on
                failed += 1
                traceback.print_exc()
                print(f"[FAIL] {name}: {exc}", file=sys.stderr)
                if name.startswith(("V0", "V1")):
                    break  # later checks need the camera / frames
                continue
            print(f"[PASS] {name}: {detail}", flush=True)
    runtime = time.perf_counter() - start
    if failed:
        print(f"{failed} VIDEO CHECK(S) FAILED ({runtime:.1f} s)")
        return 1
    print(f"ALL VIDEO CHECKS PASSED ({runtime:.1f} s)")
    return 0


def _v0(args: argparse.Namespace, state: dict) -> str:
    state["camera"], detail = check_camera_line(args)
    return detail


def _v1(args: argparse.Namespace, tmp: Path, state: dict) -> str:
    state["frames"], detail = check_video(args, tmp, state["camera"])
    return detail


if __name__ == "__main__":
    sys.exit(main())
