#!/usr/bin/env python3
"""Render the learned LCS's one-step belt prediction on a recorded LCS episode to MP4.

Left half: the episode's scene at frame k (default camera). Right half: belts only, same view
direction, up and fov, the camera moved along its view direction to zoom on the belt: current
belt k (grey), true belt k+1 (gold) and ``decode(LCS(encode(obs_k), u_k))``. Each episode frame
k is held for ``--repeat`` video frames.

Run:
    uv run python scripts/record_prediction_video.py --episode data/lcs/v2/set2_excite/\
episode_0007.npz --out data/lcs/prediction_videos/ep0007.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import newton
import newton.viewer
import warp as wp
from PIL import Image
from replay_viewer import build_model, configure_logging

from round_belt_task import BELT_COLOR
from task_common import prediction_video as pv
from task_common.replay_video import (
    DEFAULT_CAMERA,
    RENDER_RETRIES,
    CameraPose,
    Encoder,
    HeadlessBrowser,
    RenderFailed,
    belt_rgb,
    capture,
    check_render,
    probe,
    sync_scene,
)
from utils.viewer_patches import (
    patch_viewer_shape_names,
    patch_viser_texture_material,
)

DEFAULT_PORT = 19381
LAYOUTS = ("A", "B", "C", "D")  # A-C: the preview options; D: the chosen one
ERROR_COLORS = {"A": True, "B": False, "C": True, "D": False}
PLOT_STRIP = {"A": False, "B": False, "C": True, "D": True}
FILL = 0.8  # the belt's extent spans this fraction of the right view (binding axis)


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Layout:
    """The right half stacks its label, the belts-only view and the legend; the RMSE plot sits
    under the right half or along the whole bottom (strip)."""

    name: str
    left: Rect
    right_half: Rect
    right: Rect  # the belts-only view
    legend: tuple[int, int]
    plot: Rect
    error_colors: bool


def make_layout(name: str, W: int, H: int, label_h: int, legend_hw: tuple[int, int],
                margin: int) -> Layout:
    if name not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {name!r}")
    half = W // 2
    lh, lw = legend_hw
    if PLOT_STRIP[name]:
        plot = Rect(0, H - round(H * 0.2), W, round(H * 0.2))
        left = Rect(0, 0, half, plot.y)
    else:
        plot = Rect(half, H - round(H * 0.22), W - half, round(H * 0.22))
        left = Rect(0, 0, half, H)
    top = label_h + 2 * margin
    legend_y = plot.y - lh - margin
    h = legend_y - margin - top
    return Layout(name, left, Rect(half, 0, W - half, plot.y),
                  Rect(half, top, W - half, h - h % 2), (half + (W - half - lw) // 2, legend_y),
                  plot, ERROR_COLORS[name])


class Renderer:
    """The episode scene (newton viewer) and the belts-only tubes on one viser server."""

    def __init__(self, port: int) -> None:
        patch_viewer_shape_names()
        patch_viser_texture_material()
        with wp.ScopedDevice("cpu"):
            self.viewer = newton.viewer.ViewerViser(port=port, label="Prediction video",
                                                    verbose=False)
            self.model = build_model()
            self.viewer.set_model(self.model)
            self.state = self.model.state()
        self.server = self.viewer._server
        self.tubes = pv.BeltTubes(self.server)

    def scene(self, body_q: np.ndarray, t: float) -> None:
        self.viewer.begin_frame(t)
        self.state.body_q.assign(np.ascontiguousarray(body_q))
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def mode(self, belts_only: bool) -> None:
        for h in pv.viewer_handles(self.viewer):
            if h.visible == belts_only:
                h.visible = not belts_only
        self.tubes.set_visible(belts_only)

    def close(self) -> None:
        self.viewer.close()


def view(client, camera: CameraPose, W: int, H: int, w: int, h: int) -> np.ndarray:
    """A ``w`` x ``h`` view at ``camera``'s vertical fov: captured at the canvas size W x H
    (no canvas resize, which leaks GPU memory), centre-cropped to the view's aspect, then
    box-downsampled."""
    img = capture(client, camera, W, H)
    check_render(img)
    cw = round(H * w / h)
    if cw > W:
        raise ValueError(f"view {w}x{h} is wider than the {W}x{H} canvas")
    x0 = (W - cw) // 2
    crop = Image.fromarray(img[:, x0:x0 + cw])
    return np.asarray(crop.resize((w, h), Image.Resampling.BOX))


def warm(r: Renderer, client, camera: CameraPose, W: int, H: int, tries: int = 20) -> None:
    """Capture until two renders agree: textures load asynchronously on connect."""
    last = None
    for _ in range(tries):
        sync_scene(r.server, client)
        img = capture(client, camera, W, H)
        if last is not None and np.array_equal(img, last):
            return
        last = img
        time.sleep(0.5)
    logger.warning(f"[PRED] warm-up: renders still changing after {tries} tries")


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--episode", type=Path, required=True, help="LCS episode .npz")
    p.add_argument("--out", type=Path, required=True, help="output .mp4")
    p.add_argument("--layout", choices=LAYOUTS, default="D")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--start-s", type=float, default=0.0, help="episode time of the first frame")
    p.add_argument("--end-s", type=float, default=None, help="episode time of the last frame")
    p.add_argument("--speed", type=float, default=0.5, help="episode s per video s")
    p.add_argument("--repeat", type=int, default=3, help="video frames per episode frame")
    p.add_argument("--still-s", type=float, action="append", default=[],
                   help="also write <out>_still_t<s>.png at this episode time (repeatable)")
    p.add_argument("--plot-ymax", type=float, default=None,
                   help="plot y limit, mm (default: rounded up from the data)")
    p.add_argument("--plot-clip", action="store_true",
                   help="plot only the rendered frames (default: the whole episode)")
    p.add_argument("--plot-title", default=None, help="bottom plot title (default per metric)")
    p.add_argument("--belt-colors", default="#9ca3af,gold,#2563eb",
                   help="right-half CURRENT,TRUE,PRED colours: #rrggbb or gold (the left "
                        "half's belt as rendered); CURRENT none hides the current belt")
    p.add_argument("--plot-metric", choices=PLOT_METRICS, default="rmse",
                   help="bottom plot: index-wise RMSE, lcs_learning's Sinkhorn EMD, or both")
    p.add_argument("--solver", choices=pv.SOLVERS, default="lcp")
    p.add_argument("--deploy", type=Path, default=pv.DEPLOY)
    p.add_argument("--decoder", type=Path, default=pv.DECODER)
    p.add_argument("--camera", default=None, help="left camera json/path (default: the user's)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="scratch viser port")
    return p


def parse_colors(text: str, gold) -> tuple[tuple[int, int, int], ...]:
    """``CURRENT,TRUE,PRED``: ``#rrggbb`` or ``gold`` (the belt as rendered in the left half);
    CURRENT may be ``none`` (current belt k not drawn, not in the legend)."""
    parts = [c.strip() for c in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--belt-colors needs 3 colours, got {text!r}")
    rgb = lambda c: gold if c == "gold" else tuple(int(c.lstrip("#")[i:i + 2], 16)
                                                   for i in (0, 2, 4))
    return (None if parts[0] == "none" else rgb(parts[0]), *(rgb(c) for c in parts[1:]))


def legend_rgba(error_colors: bool, size: int, colors) -> np.ndarray:
    cur, true, pred_c = colors
    pred = (pv.Row("predicted belt k+1 (1 LCS step), coloured by error", "ramp")
            if error_colors else pv.Row("predicted belt k+1 (1 LCS step)", "bar", pred_c))
    rows = [] if cur is None else [pv.Row("current belt k", "bar", cur)]
    return pv.legend_rgba([*rows, pv.Row("true belt k+1", "bar", true), pred],
                          size, colorbar=error_colors)


PLOT_METRICS = ("rmse", "sinkhorn", "both")
SINKHORN_LABEL = (f"Sinkhorn EMD (\u03b5 = {pv.SINKHORN_EPS:g} m, {pv.SINKHORN_ITERS} it)")
SINKHORN_COLOR = (213, 94, 0)  # Okabe-Ito vermillion, for "both"


def rmse_plot(res: dict, rect: Rect, size: int, color, metric: str = "rmse",
              title_override: str | None = None, ymax: float | None = None,
              ks: np.ndarray | None = None) -> pv.RmsePlot:
    """``ks``: plot only these episode frames (else the whole episode)."""
    if ks is not None:
        res = {key: res[key][ks] for key in ("time_s", "rmse_model_mm", "sinkhorn_emd_mm")
               if key in res}
    rmse = pv.Series("RMSE (per coordinate, index-wise)", res["rmse_model_mm"], color)
    if metric == "rmse":
        series, title = [rmse], "one-step prediction RMSE (mm)"
    elif metric == "sinkhorn":
        series = [pv.Series("sinkhorn", res["sinkhorn_emd_mm"], color)]
        title = f"one-step prediction {SINKHORN_LABEL}, mm"
    else:
        series = [rmse, pv.Series(SINKHORN_LABEL, res["sinkhorn_emd_mm"], SINKHORN_COLOR)]
        title = "one-step prediction error (mm)"
    return pv.RmsePlot(res["time_s"], series, rect.w, rect.h, max(9, round(size * 0.85)),
                       ymax=ymax, title=title_override or title, legend=metric == "both")


def main(args: argparse.Namespace) -> int:
    W, H = args.width, args.height
    if W % 4 or H % 2 or args.repeat < 1 or args.speed <= 0:
        logger.error("--width a multiple of 4, --height even, --repeat >= 1, --speed > 0")
        return 2
    ep = pv.Episode.load(args.episode)
    model = pv.OneStepModel(args.deploy, args.decoder, args.solver)
    t0 = time.perf_counter()
    res = pv.predict_episode(model, ep)
    logger.info(f"[PRED] {args.episode}: {ep.frames} frames, outcome {ep.outcome}; one-step "
                f"RMSE mean {res['rmse_model_mm'].mean():.3f} / max "
                f"{res['rmse_model_mm'].max():.3f} mm, recon {res['rmse_recon_mm'].mean():.3f},"
                f" no-motion {res['rmse_nomotion_mm'].mean():.3f} ({args.solver}; "
                f"{time.perf_counter() - t0:.1f} s)")
    end = args.end_s if args.end_s is not None else np.inf
    ks = np.flatnonzero((res["time_s"] >= args.start_s - 1e-9) & (res["time_s"] <= end + 1e-9))
    if not len(ks):
        logger.error("empty frame range")
        return 2
    gold = belt_rgb(BELT_COLOR)
    left_cam = CameraPose.load(args.camera) if args.camera else DEFAULT_CAMERA
    size = max(10, round(H * 0.024))
    try:
        colors = parse_colors(args.belt_colors, gold)
    except ValueError as exc:
        logger.error(str(exc))
        return 2
    legend = legend_rgba(ERROR_COLORS[args.layout], size, colors)
    labels = [None, pv.text_rgba(f"one-step prediction (Δt = {ep.dt:g} s)", size)]
    margin = max(4, size // 2)
    layout = make_layout(args.layout, W, H, labels[1].shape[0], legend.shape[:2], margin)
    if args.plot_metric != "rmse":
        # lcs_learning's own sinkhorn_emd_loss (torch, via its venv): predicted vs true k+1.
        res["sinkhorn_emd_mm"] = pv.sinkhorn_emd_mm(res["pred"], res["true_next"])
        logger.info(f"[PRED] {SINKHORN_LABEL}: mean {res['sinkhorn_emd_mm'].mean():.3f} / max "
                    f"{res['sinkhorn_emd_mm'].max():.3f} mm")
    plot = rmse_plot(res, layout.plot, size, colors[2], args.plot_metric,
                     args.plot_title, args.plot_ymax, ks if args.plot_clip else None)
    plot_k0 = int(ks[0]) if args.plot_clip else 0
    # One fixed zoom over every belt drawn in the rendered range; verified unclipped below.
    drawn = np.concatenate([res[key][ks] for key in ("current", "true_next", "pred")])
    aspect = layout.right.w / layout.right.h
    right_cam = pv.fit_camera(drawn, left_cam, aspect, margin=1.0 - FILL)
    ndc = pv.project_ndc(drawn, right_cam, aspect)
    if np.any(np.abs(ndc) > 1.0 - 0.02):
        logger.error(f"right view clips the belt (|ndc| max {np.abs(ndc).max(0)})")
        return 1
    logger.info(f"[PRED] layout {layout}; right camera {right_cam.to_dict()}; belt spans "
                f"{np.ptp(ndc[:, 0]) / 2:.2f} x {np.ptp(ndc[:, 1]) / 2:.2f} of the view (x, y), "
                f"|ndc| max {np.abs(ndc).max(0).round(3)}")
    fps = args.speed / ep.dt * args.repeat
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    r = Renderer(args.port)
    sc = pv.EpisodeScene(r.model, ep)
    logger.info(f"[PRED] FK check: finger_tip vs recorded {sc.ee_error_mm(int(ks[0])):.4f} mm")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stills = {int(np.argmin(np.abs(res["time_s"] - s))): s for s in args.still_s}
    L, R = layout.left, layout.right
    sent, stats, ms = {}, [], []
    browser = HeadlessBrowser(r.server, f"http://127.0.0.1:{args.port}", W, H)
    try:
        with browser, tempfile.TemporaryDirectory(prefix=".pred-", dir=args.out.parent) as tmp:
            tmp_out = Path(tmp) / args.out.name
            enc = Encoder(tmp_out, W, H, fps)
            try:
                for i, k in enumerate(ks):
                    start = time.perf_counter()
                    r.scene(sc.body_q(int(k)), float(res["time_s"][k]))
                    if colors[0] is not None:
                        r.tubes.tube("current", res["current"][k], colors[0])
                    r.tubes.tube("true", res["true_next"][k], colors[1])
                    if layout.error_colors:
                        r.tubes.colored_tube("pred", res["pred"][k],
                                             pv.err_rgb(res["point_err_mm"][k]))
                    else:
                        r.tubes.tube("pred", res["pred"][k], colors[2])
                    if i == 0:
                        r.mode(belts_only=False)
                        warm(r, browser.client, left_cam, W, H)
                    for attempt in range(RENDER_RETRIES + 1):
                        try:
                            r.mode(belts_only=False)
                            sync_scene(r.server, browser.client)
                            left = view(browser.client, left_cam, W, H, L.w, L.h)
                            r.mode(belts_only=True)
                            sync_scene(r.server, browser.client)
                            right = view(browser.client, right_cam, W, H, R.w, R.h)
                            break
                        except (RenderFailed, RuntimeError, TimeoutError) as exc:
                            if attempt == RENDER_RETRIES:
                                raise RenderFailed(f"frame k {k}: {exc}") from exc
                            logger.warning(f"[PRED] frame k {k}: {exc}; restarting chromium "
                                           f"({attempt + 1}/{RENDER_RETRIES})")
                            browser.restart()
                            r.mode(belts_only=False)
                            warm(r, browser.client, left_cam, W, H)
                    frame = np.full((H, W, 3), 255, np.uint8)
                    frame[L.y:L.y + L.h, L.x:L.x + L.w] = left
                    frame[R.y:R.y + R.h, R.x:R.x + R.w] = right
                    sent[int(k)] = r.tubes.sent["pred"].copy()
                    compose(frame, layout, legend, plot.frame(int(k) - plot_k0), labels, size, res,
                            int(k), ep)
                    for _ in range(args.repeat):
                        enc.write(frame)
                    stats.append([pv.luma_stats(left)[1], pv.luma_stats(right)[1]])
                    if int(k) in stills:
                        p = args.out.with_name(f"{args.out.stem}_still_t"
                                               f"{stills[int(k)]:g}.png")
                        Image.fromarray(frame).save(p)
                        logger.info(f"[PRED] still {p} (k {k})")
                    ms.append(1e3 * (time.perf_counter() - start))
            finally:
                enc.close()
            tmp_out.replace(args.out)
    except RenderFailed as exc:
        logger.error(f"[PRED] {exc}; {args.out} not written")
        return 1
    finally:
        r.close()
    side = args.out.with_suffix(".npz")
    np.savez_compressed(
        side, episode=str(args.episode), solver=args.solver, deploy=str(args.deploy),
        decoder=str(args.decoder), layout=layout.name, k=ks, repeat=args.repeat,
        width=W, height=H, rects=json.dumps({"left": vars(L), "right": vars(R),
                                             "plot": vars(layout.plot)}),
        **{key: res[key] for key in ("pred", "recon_next", "true_next", "current",
                                     "point_err_mm", "rmse_model_mm", "rmse_recon_mm",
                                     "rmse_nomotion_mm", "time_s")},
        plot_metric=args.plot_metric, plot_title=plot.title, belt_colors=args.belt_colors,
        plot_ymax=plot.ymax, plot_clip=args.plot_clip,
        **({"sinkhorn_emd_mm": res["sinkhorn_emd_mm"], "sinkhorn_eps": pv.SINKHORN_EPS,
            "sinkhorn_iters": pv.SINKHORN_ITERS} if "sinkhorn_emd_mm" in res else {}),
        pred_vertices=np.stack([sent[int(k)] for k in ks]),
        view_luma_std=np.asarray(stats), cameras=json.dumps(
            {"left": left_cam.to_dict(), "right": right_cam.to_dict()}))
    info = probe(args.out)
    a = np.asarray(ms)
    logger.info(f"[PRED] wrote {args.out}: {info['frames']} frames {info['width']}x"
                f"{info['height']}, {info['duration_s']:.2f} s, {info['bytes'] / 1e6:.2f} MB; "
                f"render {a.mean():.0f} mean / {np.percentile(a, 95):.0f} p95 / {a.max():.0f} max"
                f" ms per episode frame; {browser.restarts} chromium restarts; min view luma std "
                f"{np.asarray(stats).min():.1f}; sidecar {side}")
    return 0


def compose(frame, layout: Layout, legend, plot_img, labels, size, res, k, ep) -> None:
    m = max(4, size // 2)
    L, R, P = layout.left, layout.right_half, layout.plot
    pv.blend(frame, labels[1], R.x + m, R.y + m)  # the left half carries only the time
    t = pv.text_rgba(f"t = {res['time_s'][k]:.2f} s", size)
    pv.blend(frame, t, L.x + m, L.y + L.h - t.shape[0] - m)
    pv.blend(frame, legend, *layout.legend)
    pv.blend(frame, plot_img, P.x, P.y)
    frame[L.y:L.y + L.h, L.x + L.w - 1:L.x + L.w + 1] = (156, 163, 175)  # divider


if __name__ == "__main__":
    configure_logging()
    sys.exit(main(create_parser().parse_args()))
