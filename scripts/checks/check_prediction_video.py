#!/usr/bin/env python3
"""Check of the one-step prediction video (``scripts/record_prediction_video.py``).

P0: the CLI renders a short clip (exit 0, video + sidecar written).
P1: frame count (= episode frames x ``--repeat``) and size.
P2: exact geometry: at a few frames, an independent ``decode(step(encode(obs_k), u_k))`` (exact
LCP) and ``decode(encode(obs_{k+1}))`` equal the sidecar's ``pred`` / ``recon_next`` bit for
bit, the RMSE columns follow, and the predicted-tube vertices sent to the scene equal
``tube_vertices(pred)`` in float32 exactly.
P3: no blank frame: luma std of the left view, the right view and the whole frame over every
decoded frame.
P5 (sidecars with ``sinkhorn_emd_mm``): at a few frames, lcs_learning's ``sinkhorn_emd_loss``
re-run on the sidecar's pred / true k+1 gives the same values; vs an exact EMD
(``linear_sum_assignment``) the entropic bias stays < 0.5 mm.
P4: the fixed right-view zoom never clips a drawn belt; the scene's FK matches the recorded EE.

``--scan VIDEO.mp4 ...`` runs P1-P4 on existing videos (their sidecars) instead of rendering.

Run (scratch viser port):
    uv run python scripts/checks/check_prediction_video.py --port 19391
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task_common import prediction_video as pv
from task_common.replay_learned_mpc import tube_vertices
from task_common.replay_video import CameraPose, probe, read_frames

EPISODE = REPO_ROOT / "data/lcs/v2/set2_excite/episode_0007.npz"
MIN_VIEW_STD = 5.0  # real views are >= 11; a lost context renders uniform white (std 0)
N_GEOMETRY = 4


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def independent(model: pv.OneStepModel, ep: pv.Episode, k: int) -> tuple[np.ndarray, ...]:
    """The prediction and reconstruction without ``one_step``: the numpy pieces directly."""
    s = model.lcs
    z = model.enc.encode(ep.pcd[k], ep.prop[k], ep.belt[k])[None]
    u = ep.u[k].astype(np.float64)[None]
    lam = pv.lcp_exact(s.F, z @ s.E.T + u @ s.H.T + s.c)
    z_hat = (z @ s.A.T + u @ s.B.T + lam @ s.D.T + s.d)[0]  # row form, as the recorder
    z1 = model.enc.encode(ep.pcd[k + 1], ep.prop[k + 1], ep.belt[k + 1])
    return model.dec.decode(z_hat), model.dec.decode(z1), z_hat


def p1(video: Path, side: dict) -> str:
    info = probe(video)
    want = len(side["k"]) * int(side["repeat"])
    require(info["frames"] == want, f"{info['frames']} frames, want {want}")
    size = (int(side["width"]), int(side["height"]))
    require((info["width"], info["height"]) == size, f"size {info['width']}x{info['height']}")
    return f"{info['frames']} frames {size[0]}x{size[1]} {info['duration_s']:.2f} s"


def p2(side: dict, model: pv.OneStepModel) -> str:
    ep = pv.Episode.load(Path(str(side["episode"])))
    ks = side["k"]
    picks = sorted({int(i) for i in np.linspace(0, len(ks) - 1, N_GEOMETRY).round()})
    for i in picks:
        k = int(ks[i])
        pred, recon, _ = independent(model, ep, k)
        require(np.array_equal(side["pred"][k], pred), f"pred k {k} differs "
                f"(max {np.abs(side['pred'][k] - pred).max():.3g} m)")
        require(np.array_equal(side["recon_next"][k], recon), f"recon k {k} differs")
        require(np.array_equal(side["true_next"][k], ep.belt[k + 1]), f"true belt k {k}")
        require(np.array_equal(side["current"][k], ep.belt[k]), f"current belt k {k}")
        verts = tube_vertices(pred, pv.TUBE_M, pv.TUBE_SIDES).astype(np.float32)
        require(np.array_equal(side["pred_vertices"][i], verts), f"sent tube k {k} differs")
        require(np.isclose(side["rmse_model_mm"][k], pv.rmse_mm(pred, ep.belt[k + 1]),
                           rtol=0, atol=1e-12), f"rmse k {k}")
        require(np.isclose(side["rmse_recon_mm"][k], pv.rmse_mm(recon, ep.belt[k + 1]),
                           rtol=0, atol=1e-12), f"recon rmse k {k}")
    return (f"frames k {[int(ks[i]) for i in picks]}: pred, recon, belts and sent vertices "
            f"bit-exact; solver {side['solver']}")


def p3(video: Path, side: dict) -> str:
    rects = json.loads(str(side["rects"]))
    W, H = int(side["width"]), int(side["height"])
    frames = read_frames(video, W, H)
    worst = {"left": 1e9, "right": 1e9, "frame": 1e9}
    for f in frames:
        for name in ("left", "right"):
            r = rects[name]
            worst[name] = min(worst[name], pv.luma_stats(
                f[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]])[1])
        worst["frame"] = min(worst["frame"], pv.luma_stats(f)[1])
    require(min(worst.values()) >= MIN_VIEW_STD, f"blank frame: min luma std {worst}")
    return (f"{len(frames)} frames; min luma std left {worst['left']:.1f} right "
            f"{worst['right']:.1f} frame {worst['frame']:.1f}")


def p4(side: dict) -> str:
    cams = json.loads(str(side["cameras"]))
    r = json.loads(str(side["rects"]))["right"]
    ks = side["k"]
    drawn = np.concatenate([side[key][ks] for key in ("current", "true_next", "pred")])
    ndc = pv.project_ndc(drawn, CameraPose.from_dict(cams["right"]), r["w"] / r["h"])
    reach = np.abs(ndc).max(0)
    require(np.all(reach < 0.98), f"right view clips a belt: |ndc| max {reach}")
    left, right = CameraPose.from_dict(cams["left"]), CameraPose.from_dict(cams["right"])
    d = lambda c: np.subtract(c.look_at, c.position) / np.linalg.norm(
        np.subtract(c.look_at, c.position))
    require(np.allclose(d(left), d(right), atol=1e-9) and left.up == right.up
            and left.fov == right.fov, "right camera changed direction/up/fov")
    return f"belt reach |ndc| {reach.round(3)} (< 0.98); same direction, up and fov"


def p5(side: dict) -> str:
    ks = side["k"]
    picks = [int(ks[i]) for i in sorted({int(i) for i in np.linspace(0, len(ks) - 1, 4).round()})]
    eps, iters = float(side["sinkhorn_eps"]), int(side["sinkhorn_iters"])
    again = pv.sinkhorn_emd_mm(side["pred"][picks], side["true_next"][picks], eps, iters)
    err = np.abs(again - side["sinkhorn_emd_mm"][picks]).max()
    require(err <= 1e-9, f"sinkhorn recompute differs by {err:.3g} mm")
    exact = pv.emd_exact_mm(side["pred"][ks], side["true_next"][ks])
    bias = side["sinkhorn_emd_mm"][ks] - exact
    require(np.abs(bias).max() < 0.5, f"sinkhorn vs exact EMD max |diff| {np.abs(bias).max():.3f}")
    return (f"k {picks} recomputed (max diff {err:.1e} mm); vs exact EMD over {len(ks)} frames: "
            f"mean {bias.mean():+.3f}, max |diff| {np.abs(bias).max():.3f} mm")


def fk_check(side: dict) -> str:
    import warp as wp
    from replay_viewer import build_model

    with wp.ScopedDevice("cpu"):
        model = build_model()
    ep = pv.Episode.load(Path(str(side["episode"])))
    sc = pv.EpisodeScene(model, ep)
    err = max(sc.ee_error_mm(int(k)) for k in side["k"][:: max(1, len(side["k"]) // 5)])
    require(err < 0.05, f"FK finger_tip vs recorded {err:.3f} mm")
    return f"FK finger_tip vs recorded max {err:.4f} mm"


def run_checks(video: Path, model: pv.OneStepModel, results: list, fk: bool = True) -> None:
    with np.load(video.with_suffix(".npz"), allow_pickle=False) as d:
        side = {k: d[k] for k in d.files}
    checks = [("P1 frames/size", lambda: p1(video, side)),
              ("P2 exact geometry", lambda: p2(side, model)),
              ("P3 non-blank", lambda: p3(video, side)),
              ("P4 zoom/camera", lambda: p4(side) + ("; " + fk_check(side) if fk else ""))]
    if "sinkhorn_emd_mm" in side:
        checks.append(("P5 sinkhorn", lambda: p5(side)))
    for name, fn in checks:
        results.append(attempt(f"{name} [{video.name}]", fn))


def attempt(name: str, fn) -> bool:
    start = time.perf_counter()
    try:
        detail = fn()
        print(f"PASS {name}: {detail} ({time.perf_counter() - start:.1f} s)", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {exc}", flush=True)
        traceback.print_exc()
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=19391, help="scratch viser port")
    p.add_argument("--scan", type=Path, nargs="*", default=None,
                   help="check these rendered videos instead of rendering a clip")
    p.add_argument("--deploy", type=Path, default=pv.DEPLOY,
                   help="model the scanned videos were rendered with (default: v2)")
    p.add_argument("--decoder", type=Path, default=pv.DECODER)
    args = p.parse_args()
    model = pv.OneStepModel(args.deploy, args.decoder)
    results: list[bool] = []
    if args.scan:
        for video in args.scan:
            run_checks(video, model, results)
    else:
        with tempfile.TemporaryDirectory(prefix="check_prediction_video_") as tmp:
            video = Path(tmp) / "clip.mp4"
            cmd = [sys.executable, str(REPO_ROOT / "scripts/record_prediction_video.py"),
                   "--episode", str(EPISODE), "--out", str(video), "--width", "960",
                   "--height", "540", "--start-s", "4.0", "--end-s", "4.4", "--repeat", "2",
                   "--plot-metric", "both",
                   "--port", str(args.port)]

            def p0() -> str:
                with open(Path(tmp) / "render.log", "wb") as log:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False,
                                        timeout=900).returncode
                tail = (Path(tmp) / "render.log").read_text()[-1500:]
                require(rc == 0 and video.is_file() and video.with_suffix(".npz").is_file(),
                        f"render exit {rc}:\n{tail}")
                return f"rendered {video.name} (exit 0)"

            ok = attempt("P0 render", p0)
            results.append(ok)
            if ok:
                run_checks(video, model, results)
    print("ALL PASS" if all(results) else f"FAILED {results.count(False)}/{len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
