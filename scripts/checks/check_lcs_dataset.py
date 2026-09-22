#!/usr/bin/env python3
"""Headless check of the LCS episode format (``task_common.lcs_dataset``) on synthetic data.

No sim, no GPU, no LCM. F0 constants, F1 belt resampling, F2 writer -> validator round trip,
F3 broken copies are rejected naming the key. If the magna logs are present, the validator is
also run on ``log_001.npz`` (reported as INFO).

Run:
    uv run python scripts/checks/check_lcs_dataset.py
"""

from __future__ import annotations

import math
import shutil
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

from task_common import lcs_dataset as ld

MAGNA_LOG = Path("/home/hienbui/git/magna-logs/2026-06-22/log_001.npz")
N_FRAMES = 12


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _ellipse(n: int = ld.BELT_BODIES, a: float = 0.12, b: float = 0.07) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([0.45 + a * np.cos(t), b * np.sin(t), np.full(n, 0.03)], axis=1)


def _signed_area_xy(p: np.ndarray) -> float:
    x, y = p[:, 0].astype(np.float64), p[:, 1].astype(np.float64)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _quat_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


def _quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


def _synthetic_frame(rng: np.random.Generator, i: int) -> dict:
    ee_f = np.concatenate([[0.40, -0.05, 0.04 + 0.001 * i], _quat_axis_angle([0, 0, 1], 0.1 * i)])
    ee_u = np.concatenate([[0.50, 0.10, 0.05], _quat_axis_angle([1, 0, 0], np.pi - 0.05 * i)])
    state = ld.state_vector(rng.normal(size=7), rng.normal(size=6), rng.normal(size=7),
                            rng.normal(size=6), ee_f, ee_u)
    n = int(rng.integers(1900, 2100))
    return {
        "state": state, "ee_f": ee_f, "ee_u": ee_u,
        "pcd": rng.uniform([0.2, -0.25, 0.015], [0.7, 0.25, 0.11], size=(n, 3)),
        "rgb": rng.integers(0, 256, size=(n, 3)),
        "belt": ld.belt_points_ordered(_ellipse() + 0.001 * i),
    }


@check("F0 constants")
def check_f0(ctx: SimpleNamespace) -> str:
    _require(abs(ld.SAMPLE_STEPS * ld.SIM_DT_S - ld.SAMPLE_PERIOD_S) < 1e-12,
             f"SAMPLE_STEPS {ld.SAMPLE_STEPS} * {ld.SIM_DT_S} != {ld.SAMPLE_PERIOD_S}")
    _require(ld.SAMPLE_PERIOD_US == round(ld.SAMPLE_PERIOD_S * 1e6), "SAMPLE_PERIOD_US mismatch")
    _require((ld.STATE_DIM, ld.ACTION_DIM, ld.BELT_POINTS) == (40, 12, 150),
             f"dims {(ld.STATE_DIM, ld.ACTION_DIM, ld.BELT_POINTS)} != (40, 12, 150)")
    _require(ld.POSE_LAYOUT == "xyz_wxyz", f"POSE_LAYOUT {ld.POSE_LAYOUT}")
    _require(not set(ld.KEYS_REQUIRED) & set(ld.KEYS_OPTIONAL), "required/optional overlap")
    _require(not any(k.startswith(ld.EXTRA_PREFIX) for k in ld.KEYS_REQUIRED + ld.KEYS_OPTIONAL),
             "a collector key uses the sim_ prefix")
    return (f"period {ld.SAMPLE_PERIOD_S} s = {ld.SAMPLE_STEPS} x {ld.SIM_DT_S} s, "
            f"state {ld.STATE_DIM}, action {ld.ACTION_DIM}, belt {ld.BELT_POINTS}")


@check("F1 belt_points_ordered")
def check_f1(ctx: SimpleNamespace) -> str:
    bodies = _ellipse()
    out = ld.belt_points_ordered(bodies)
    _require(out.shape == (ld.BELT_POINTS, 3) and out.dtype == np.float32,
             f"shape/dtype {out.shape} {out.dtype}")
    _require(np.allclose(out[0], bodies[0], atol=1e-7), f"first point {out[0]} != body 0")
    gaps = np.linalg.norm(np.diff(np.vstack([out, out[:1]]).astype(np.float64), axis=0), axis=1)
    spread = (gaps.max() - gaps.min()) / gaps.mean()
    _require(spread < 0.02, f"closed-loop spacing spread {spread:.4f} >= 2 %")
    _require(np.array_equal(out, ld.belt_points_ordered(bodies)), "not deterministic")

    shifted = ld.belt_points_ordered(np.roll(bodies, -1, axis=0))
    _require(np.allclose(shifted[0], bodies[1], atol=1e-7), "shifted: first point != body 1")
    _require(np.sign(_signed_area_xy(shifted)) == np.sign(_signed_area_xy(out)),
             "shifted input reversed the loop orientation")
    # Each shifted point maps to a strictly forward-walking point of the original loop.
    dense = ld.belt_points_ordered(bodies, n_points=ld.BELT_POINTS * 20).astype(np.float64)
    idx = np.array([np.argmin(np.linalg.norm(dense - p, axis=1)) for p in shifted])
    steps = np.mod(np.diff(idx), len(dense))
    _require(np.all((steps > 0) & (steps < len(dense) // 4)),
             "shifted output does not roll forward")
    return f"150 pts, spacing spread {100 * spread:.2f} %, 1-body shift rolls forward"


@check(f"F2 writer round trip ({N_FRAMES} frames)")
def check_f2(ctx: SimpleNamespace) -> str:
    rng = np.random.default_rng(0)
    frames = [_synthetic_frame(rng, i) for i in range(N_FRAMES + 1)]
    writer = ld.EpisodeWriter()
    step0 = 2700
    for i in range(N_FRAMES):
        f, g = frames[i], frames[i + 1]
        action = ld.action_vector(f["ee_f"], g["ee_f"], f["ee_u"], g["ee_u"])
        step = step0 + i * ld.SAMPLE_STEPS
        writer.add_frame(step, step * ld.SIM_DT_S, f["state"], action, f["pcd"], f["belt"],
                         None, {"wrap_deg": 10.0 * i, "h_mm": np.full(3, i)}, pcd_rgb=f["rgb"])
    try:
        writer.add_frame(step0, 0.0, frames[0]["state"], np.zeros(12), frames[0]["pcd"],
                         frames[0]["belt"])
        raise AssertionError("add_frame accepted a step not SAMPLE_STEPS after the last")
    except ValueError:
        pass
    path = writer.write(ctx.tmp / "episode_0000.npz", "engaged", {"seed": np.int64(7)})
    ctx.episode = path
    summary = ld.validate_episode(path)
    _require(summary["T"] == N_FRAMES, f"T {summary['T']} != {N_FRAMES}")
    with np.load(path, allow_pickle=True) as d:
        _require(np.all(np.diff(d["utime"]) == ld.SAMPLE_PERIOD_US), "utime diffs != period")
        _require(d["pcd"].shape == (N_FRAMES,) and d["pcd"].dtype == object, "pcd not 1-D object")
        _require(d["pcd_belt"].shape == (N_FRAMES, 150, 3), f"pcd_belt {d['pcd_belt'].shape}")
        _require(d["pcd_kinematic"].shape == (N_FRAMES, 0, 3), "pcd_kinematic not (T, 0, 3)")
        _require(d["state"].dtype == np.float64 and d["actions"].dtype == np.float64,
                 "state/actions not float64")
        _require(str(d["trajectory_label"]) == "engaged", "trajectory_label")
        _require(d["sim_wrap_deg"].shape == (N_FRAMES,) and d["sim_h_mm"].shape == (N_FRAMES, 3),
                 "sim_ extras shapes")
        _require('"seed": 7' in str(d["sim_meta"]), "sim_meta lost extras_meta")

    # Hand-computed delta for a known pose pair.
    p0f = np.array([0.1, 0.2, 0.3, *_quat_axis_angle([0, 0, 1], 0.3)])
    p1f = np.array([0.15, 0.1, 0.33, *_quat_axis_angle([0, 1, 1], 0.5)])
    p0u = np.array([0.5, -0.1, 0.2, *_quat_axis_angle([1, 0, 0], 3.0)])
    p1u = np.array([0.5, -0.12, 0.25, *_quat_axis_angle([1, 0, 0], -3.0)])
    got = ld.action_vector(p0f, p1f, p0u, p1u)
    want = np.concatenate([
        p1f[:3] - p0f[:3], p1u[:3] - p0u[:3],
        _rotvec_from_matrix(_quat_to_matrix(p1f[3:]) @ _quat_to_matrix(p0f[3:]).T),
        _rotvec_from_matrix(_quat_to_matrix(p1u[3:]) @ _quat_to_matrix(p0u[3:]).T),
    ])
    err = float(np.abs(got - want).max())
    _require(err < 1e-9, f"action_vector error {err:.3e} >= 1e-9")
    _require(abs(np.linalg.norm(got[9:12]) - (2 * np.pi - 6.0)) < 1e-9,
             "UR delta across +-pi did not take the short way")
    return (f"T {summary['T']}, utime step {summary['period_us']} us, "
            f"pcd {summary['points']['pcd']}, action err {err:.1e}")


def _broken_copy(src: Path, dst: Path, mutate) -> Path:
    with np.load(src, allow_pickle=True) as d:
        payload = {k: d[k] for k in d.files}
    mutate(payload)
    with open(dst, "wb") as f:
        np.savez_compressed(f, **payload)
    return dst


def _set_nan(p: dict) -> None:
    frames = list(p["pcd"])
    frames[3] = frames[3].copy()
    frames[3][5, 1] = np.nan
    p["pcd"] = ld._ragged(frames)


@check("F3 broken copies rejected")
def check_f3(ctx: SimpleNamespace) -> str:
    cases = [
        ("pcd_belt", lambda p: p.pop("pcd_belt")),
        ("state", lambda p: p.update(state=p["state"][:, :39])),
        ("pcd", _set_nan),
    ]
    for i, (key, mutate) in enumerate(cases):
        path = _broken_copy(ctx.episode, ctx.tmp / f"broken_{i}.npz", mutate)
        try:
            ld.validate_episode(path)
        except ValueError as exc:
            _require(f": {key}:" in str(exc) and path.name in str(exc),
                     f"error does not name {path.name} / {key}: {exc}")
            continue
        raise AssertionError(f"validate_episode accepted a copy with broken {key}")
    return "missing pcd_belt, state width 39, NaN in pcd -> ValueError naming the key"


def main() -> int:
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="check_lcs_dataset_"))
    ctx = SimpleNamespace(tmp=tmp)
    exit_code = 0
    try:
        for name, fn in CHECKS:
            try:
                detail = fn(ctx)
            except AssertionError as exc:
                print(f"[FAIL] {name}: {exc}", file=sys.stderr)
                exit_code = 1
                break
            except Exception as exc:  # noqa: BLE001 - report, then still clean up
                print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
                traceback.print_exc()
                exit_code = 1
                break
            print(f"[PASS] {name}: {detail}")
        if exit_code == 0 and MAGNA_LOG.is_file():
            s = ld.validate_episode(MAGNA_LOG)
            print(f"[INFO] {MAGNA_LOG.name}: T {s['T']}, label {s.get('trajectory_label')}, "
                  f"period {s['period_us']} us, pcd {s['points']['pcd']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        print(f"ALL LCS DATASET CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
