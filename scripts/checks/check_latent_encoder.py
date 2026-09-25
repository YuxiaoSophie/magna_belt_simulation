#!/usr/bin/env python3
"""Headless check of the numpy latent encoder, learned-LCS step and belt metrics.

No sim, no GPU, no LCM traffic. Against an ``lcs_learning`` deploy export: E0 load + dims,
E1 preprocessing bit-exact, E2 encoder vs the torch reference ``z``, E3 LCS step vs the torch PGD,
E4 timing, E5 ``LATENT_STATE`` round trip, E6 ``DemoGoals`` on a synthetic file, E7 belt metrics,
E8 ``LatentDecoder`` vs the torch decodes in ``decoder.npz`` beside the deploy (``[SKIP]`` without it).
``[SKIP]`` (exit 0) if the deploy export is absent.

Run:
    uv run python scripts/checks/check_latent_encoder.py [--deploy PATH]
"""

from __future__ import annotations

import argparse
import json
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

from task_common import belt_metrics as bm
from task_common import latent_encoder as le
from task_common.lcs_dataset import BELT_BODIES, BELT_POINTS, STATE_DIM

DEFAULT_DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_ablation_20260924/"
                      "ckpt_decoded_only/deploy/deploy.npz")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


class Skip(Exception):
    pass


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


@check("E0 load")
def check_e0(ctx: SimpleNamespace) -> str:
    ctx.enc = le.LatentEncoder.load(ctx.deploy)
    ctx.lcs = le.LearnedLcs.load(ctx.deploy)
    enc, lcs = ctx.enc, ctx.lcs
    a = json.loads(ctx.report.read_text())["checkpoint_args"]
    dims = {"num_points": (enc.num_points, a["num_points"]),
            "belt_num_points": (enc.belt_num_points, a["belt_num_points"]),
            "proprio_dim": (enc.proprio_dim, a["proprio_dim"]),
            "point_feat_dim": (enc.point_feat_dim, a["point_feat_dim"]),
            "latent_dim": (enc.latent_dim, a["latent_dim"]),
            "n_x": (lcs.n_x, a["latent_dim"]), "n_u": (lcs.n_u, a["control_dim"]),
            "n_lam": (lcs.n_lam, a["n_lam"]), "stiffness": (lcs.stiffness, a["lcs_stiffness"])}
    bad = {k: v for k, v in dims.items() if v[0] != v[1]}
    _require(not bad, f"dims differ from report.json checkpoint_args: {bad}")
    f_ref = lcs.G @ lcs.G.T + lcs.stiffness * np.eye(lcs.n_lam) + lcs.J - lcs.J.T
    f_err = float(np.abs(lcs.F - f_ref).max())
    _require(f_err <= 1e-12, f"F != GG^T + sI + J - J^T (max {f_err:.2e})")
    with np.load(ctx.deploy, allow_pickle=False) as d:
        demo = "z_goal_stage1" in d.files
    n_rows = 2 if demo else 1
    _require(lcs.stage_goals.shape == (n_rows, lcs.n_x),
             f"stage_goals shape {lcs.stage_goals.shape}, expected ({n_rows}, {lcs.n_x})")
    if demo:
        _require(np.array_equal(lcs.z_goal, lcs.stage_goals[1]), "z_goal != z_goal_stage2")
        _require(lcs.goal_frames is not None, "demo export without goal_frames")
    _require(lcs.goal_source != "", "empty goal_source")
    return (f"dims match checkpoint_args (pts {enc.num_points}, belt {enc.belt_num_points}, "
            f"prop {enc.proprio_dim}, feat {enc.point_feat_dim}, z {enc.latent_dim}, u {lcs.n_u}, "
            f"lam {lcs.n_lam}); F identity {f_err:.1e}; goal_source {lcs.goal_source!r}; "
            f"stage_goals {lcs.stage_goals.shape[0]} row(s); goal_tol {lcs.goal_tol:.3f}")


@check("E1 preprocessing bit-exact")
def check_e1(ctx: SimpleNamespace) -> str:
    r = ctx.ref
    n_lin = n_tile = 0
    for k, pc in enumerate(r["pc_raw"]):
        _require(np.array_equal(ctx.enc.preprocess(pc), r["pc_resized"][k]),
                 f"frame {k}: preprocess != pc_resized")
        _require(np.array_equal(le.resize_points_ordered(r["belt_raw"][k], BELT_POINTS),
                                r["belt_raw"][k]), f"frame {k}: belt resize is not identity")
        if len(pc) >= ctx.enc.num_points:
            n_lin += 1
        else:
            n_tile += 1
    _require(n_lin > 0 and n_tile > 0, f"linspace {n_lin} / tiling {n_tile}: a path is missing")
    return f"K={len(r['pc_raw'])} bit-exact (linspace {n_lin}, tiling {n_tile}); belt identity"


@check("E2 encoder vs torch z")
def check_e2(ctx: SimpleNamespace) -> str:
    r = ctx.ref
    z = np.stack([ctx.enc.encode(r["pc_raw"][k], r["prop"][k], r["belt_raw"][k])
                  for k in range(len(r["z"]))])
    z_err = float(np.abs(z - r["z"]).max())
    zb = ctx.enc.encode_batch(list(r["pc_raw"]), r["prop"], r["belt_raw"])
    b_err = float(np.abs(zb - r["z"]).max())
    g_err = float(np.abs(ctx.enc.pc_global(r["pc_resized"]) - r["pc_global"]).max())
    _require(z.dtype == np.float64 and z.shape == r["z"].shape, f"z {z.dtype} {z.shape}")
    _require(z_err <= 1e-5, f"max |encode - z| = {z_err:.2e} > 1e-5")
    _require(b_err <= 1e-5, f"max |encode_batch - z| = {b_err:.2e} > 1e-5")
    _require(g_err <= 1e-5, f"max |pc_global - ref| = {g_err:.2e} > 1e-5")
    return (f"max |encode - z| {z_err:.2e}, encode_batch {b_err:.2e}, pc_global {g_err:.2e} "
            f"(|z| max {np.abs(r['z']).max():.2f})")


@check("E3 learned LCS step")
def check_e3(ctx: SimpleNamespace) -> str:
    r, lcs = ctx.ref, ctx.lcs
    fixed, loose, lam_min = [], [], math.inf
    for k in range(len(r["z"])):
        zn, lam, _ = lcs.step(r["z"][k], r["u"][k], iters=25, tol=0.0)
        fixed.append(zn)
        lam_min = min(lam_min, float(lam.min()))
        zn, lam, _ = lcs.step(r["z"][k], r["u"][k], iters=100, tol=1e-5)
        loose.append(zn)
        lam_min = min(lam_min, float(lam.min()))
    e_fixed = float(np.abs(np.stack(fixed) - r["z_next_pgd_trainfixed"]).max())
    e_pgd = float(np.abs(np.stack(loose) - r["z_next_pgd"]).max())
    e_exact = float(np.abs(np.stack(loose) - r["z_next_exact"]).max())
    _require(e_fixed <= 1e-6, f"step(25, 0) vs z_next_pgd_trainfixed {e_fixed:.2e} > 1e-6")
    _require(e_pgd <= 1e-5, f"step(100, 1e-5) vs z_next_pgd {e_pgd:.2e} > 1e-5")
    _require(e_exact <= 1e-2, f"step(100, 1e-5) vs z_next_exact {e_exact:.2e} > 1e-2")
    _require(lam_min >= 0.0, f"lam < 0 ({lam_min})")
    _require(lcs.whitened_dist(lcs.z_goal) == 0.0, "whitened_dist(z_goal) != 0")
    Z, L = lcs.rollout(r["z"][0], np.stack([r["u"][0]] * 3))
    _require(Z.shape == (4, lcs.n_x) and L.shape == (3, lcs.n_lam), f"rollout {Z.shape}")
    _require(np.array_equal(Z[1], fixed[0]), "rollout[1] != step")
    u_big = lcs.u_ub + 1.0
    _require(np.array_equal(lcs.clip_u(u_big), lcs.u_ub), "clip_u does not clip to u_ub")
    return (f"25-it vs trainfixed {e_fixed:.2e}, 100-it/tol vs pgd {e_pgd:.2e}, vs exact "
            f"{e_exact:.2e}; lam min {lam_min:.1e}; rollout + clip_u ok")


@check("E4 timing")
def check_e4(ctx: SimpleNamespace) -> str:
    r = ctx.ref
    k = int(np.argmax([len(p) for p in r["pc_raw"]]))
    args = (r["pc_raw"][k], r["prop"][k], r["belt_raw"][k])
    ctx.enc.encode(*args)
    t0 = time.perf_counter()
    for _ in range(50):
        ctx.enc.encode(*args)
    t_enc = (time.perf_counter() - t0) / 50 * 1e3
    t0 = time.perf_counter()
    for _ in range(200):
        ctx.lcs.step(r["z"][0], r["u"][0])
    t_step = (time.perf_counter() - t0) / 200 * 1e3
    _require(t_enc < 50.0, f"encode {t_enc:.1f} ms >= 50 ms")
    _require(t_step < 2.0, f"step {t_step:.3f} ms >= 2 ms")
    warn = " [WARN encode >= 20 ms target]" if t_enc >= 20.0 else ""
    return (f"encode mean {t_enc:.2f} ms over 50 ({len(args[0])} pts); step mean "
            f"{t_step:.3f} ms{warn}")


@check("E5 LATENT_STATE round trip")
def check_e5(ctx: SimpleNamespace) -> str:
    rng = np.random.default_rng(0)
    z, pf, pu, prop = (rng.normal(size=n) for n in (ctx.lcs.n_x, 7, 7, STATE_DIM))
    msg = le.latent_state_message(1234567, 12.345, z, pf, pu, prop)
    dec = le.lcmt_timestamped_saved_traj.decode(msg.encode())
    _require(list(dec.saved_traj.trajectory_names) == list(le.LATENT_BLOCK_NAMES),
             f"names {dec.saved_traj.trajectory_names}")
    _require(dec.saved_traj.metadata.name == le.LATENT_METADATA_NAME, "metadata name")
    _require(all(b.num_points == 1 and list(b.time_vec) == [12.345]
                 for b in dec.saved_traj.trajectories), "blocks are not one column at [t]")
    utime, t, *vals = le.parse_latent_state_message(dec)
    _require(utime == 1234567 and t == 12.345, f"utime/t {utime} {t}")
    err = max(float(np.abs(a - b).max()) for a, b in zip(vals, (z, pf, pu, prop), strict=True))
    _require(err <= 1e-12, f"round trip error {err:.1e}")
    for bad in (0, -5):
        try:
            le.latent_state_message(bad, 1.0, z, pf, pu, prop)
        except ValueError:
            continue
        raise AssertionError(f"utime {bad} accepted")
    try:
        le.latent_state_message(1, 1.0, z, pf[:6], pu, prop)
    except ValueError:
        pass
    else:
        raise AssertionError("a 6-element ee_pose_franka was accepted")
    return (f"4 one-column blocks on {le.LATENT_STATE_CHANNEL!r}, max error {err:.1e}; "
            "utime 0/-5 and a short pose raise")


def _demo_goals_dict(n: int = 2, nx: int = 16, n_t: int = 20) -> dict:
    rng = np.random.default_rng(1)
    return {
        "n_stages": np.array(n), "stage_labels": np.array(["pre_place_1", "place_3"][:n]),
        "stage_frames": np.arange(n, dtype=np.int64) * 10,
        "z_goals": rng.normal(size=(n, nx)), "goal_tols": np.full(n, 0.9),
        "max_durations_s": np.full(n, 6.0), "ee_pose_franka": rng.normal(size=(n, 7)),
        "ee_pose_ur": rng.normal(size=(n, 7)), "z": rng.normal(size=(n_t, nx)),
        "goal_dist": rng.random((n_t, n)), "pcd_belt_stage": rng.normal(size=(n, BELT_POINTS, 3)),
        "belt_xyz_stage": rng.normal(size=(n, BELT_BODIES, 3)),
        "pulley_pose_stage": rng.normal(size=(n, 7)),
        "state_stage": rng.normal(size=(n, STATE_DIM)),
        "first_hold_place_3_frame": np.array(15), "z_std": rng.random(nx) + 0.5,
        "demo_episode": np.array("demo/episode_0000.npz"), "demo_sha256": np.array("0" * 64),
        "deploy_sha256": np.array("1" * 64), "created": np.array("2026-09-24T00:00:00+00:00"),
        "notes": np.array("synthetic"),
    }


@check("E6 DemoGoals")
def check_e6(ctx: SimpleNamespace) -> str:
    path = ctx.tmp / "demo_goals.npz"
    good = _demo_goals_dict()
    np.savez(path, **good)
    g = le.DemoGoals.load(path)
    _require(g.n_stages == 2 and g.stage_labels == ["pre_place_1", "place_3"], "labels")
    _require(g.demo_episode == "demo/episode_0000.npz" and g.first_hold_place_3_frame == 15,
             "scalar keys")
    for k in range(2):
        _require(g.dist(g.z_goals[k], k) == 0.0, f"dist(z_goals[{k}], {k}) != 0")
        _require(np.array_equal(g.stage_goal(k), good["z_goals"][k]), f"stage_goal({k})")
    d01 = g.dist(g.z_goals[0], 1)
    ref = float(np.linalg.norm((good["z_goals"][0] - good["z_goals"][1]) / good["z_std"]))
    _require(abs(d01 - ref) <= 1e-12, f"dist(stage0, 1) {d01} != {ref}")
    broken = {"z_goals": np.zeros((2, 15)), "pcd_belt_stage": np.zeros((2, 149, 3)),
              "goal_dist": np.zeros((20, 3)), "state_stage": np.zeros((2, 39))}
    for key, value in broken.items():
        bad_path = ctx.tmp / f"demo_goals_bad_{key}.npz"
        np.savez(bad_path, **{**good, key: value})
        try:
            le.DemoGoals.load(bad_path)
        except ValueError as exc:
            _require(key in str(exc), f"error does not name {key}: {exc}")
            continue
        raise AssertionError(f"DemoGoals accepted a wrong {key} shape")
    return f"2-stage synthetic file loads, dist(own goal) = 0; wrong {sorted(broken)} raise"


def _loop(n: int = BELT_POINTS) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([0.45 + 0.12 * np.cos(t), 0.07 * np.sin(t), np.full(n, 0.03)], axis=1)


@check("E7 belt metrics")
def check_e7(ctx: SimpleNamespace) -> str:
    a = _loop()
    _require(bm.belt_rmse_mm(a, a) == 0.0, "rmse(a, a) != 0")
    b = np.roll(a, 7, axis=0)
    r_idx, cham = bm.belt_rmse_mm(a, b), bm.belt_chamfer_mm(a, b)
    r_best, shift = bm.belt_best_shift_rmse_mm(a, b)
    _require(r_best < 1e-9 and shift == 7, f"best shift {r_best:.2e} mm at {shift}")
    _require(cham < 1e-9, f"chamfer {cham:.2e} mm on a pure index shift")
    _require(r_idx > 20.0, f"index-wise rmse {r_idx:.1f} mm not large on a shift by 7")
    c = a + np.array([0.0, 0.0, 0.003])
    t_idx, t_cham = bm.belt_rmse_mm(a, c), bm.belt_chamfer_mm(a, c)
    t_best, t_shift = bm.belt_best_shift_rmse_mm(a, c)
    for name, v in (("rmse", t_idx), ("chamfer", t_cham), ("best-shift", t_best)):
        _require(abs(v - 3.0) < 1e-6, f"3 mm translation: {name} {v:.4f} mm")
    _require(t_shift == 0, f"3 mm translation: best shift {t_shift}")
    q = _quat_z(10.0)
    mm, deg = bm.pose_error([0, 0, 0, 1, 0, 0, 0], [0.003, 0, 0, *q])
    _require(abs(mm - 3.0) < 1e-9 and abs(deg - 10.0) < 1e-9, f"pose_error {mm} mm {deg} deg")
    return (f"shift 7: index rmse {r_idx:.1f} mm, best {r_best:.1e} mm @ {shift}, chamfer "
            f"{cham:.1e} mm; 3 mm z: {t_idx:.3f}/{t_cham:.3f}/{t_best:.3f} mm; pose 3 mm/10 deg")


@check("E8 LatentDecoder")
def check_e8(ctx: SimpleNamespace) -> str:
    path = ctx.deploy.parent / "decoder.npz"
    if not path.is_file():
        raise Skip(f"{path} absent")
    dec = le.LatentDecoder.load(path)
    with np.load(path, allow_pickle=False) as d:
        z_ref, belt_ref = d["z_ref"], d["belt_dec_ref"]
    with np.load(ctx.deploy, allow_pickle=False) as d:
        deploy_sha = str(d["checkpoint_sha256"])
    _require(dec.checkpoint_sha256 == deploy_sha, "decoder and deploy checkpoints differ")
    belts = dec.decode_batch(z_ref)
    _require(belts.dtype == np.float32 and belts.shape == (len(z_ref), dec.num_points, 3),
             f"decode_batch {belts.dtype} {belts.shape}")
    err = float(np.abs(belts - belt_ref).max())
    one = float(np.abs(dec.decode(z_ref[0]) - belt_ref[0]).max())
    _require(err <= 1e-5 and one <= 1e-5, f"max |decode - torch| {err:.2e} / {one:.2e} > 1e-5 m")
    return (f"K={len(z_ref)} decodes ({dec.num_points} pts) vs torch max {err:.2e} m, "
            f"decode() {one:.2e} m; checkpoint sha matches the deploy")


def _quat_z(deg: float) -> list[float]:
    h = math.radians(deg) / 2.0
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY,
                        help="deploy.npz (reference_vectors.npz + report.json beside it)")
    args = parser.parse_args()
    reference = args.deploy.parent / "reference_vectors.npz"
    report = args.deploy.parent / "report.json"
    missing = [p for p in (args.deploy, reference, report) if not p.is_file()]
    if missing:
        print(f"[SKIP] deploy export absent: {', '.join(map(str, missing))}")
        return 0

    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="check_latent_encoder_"))
    # pc_raw is a ragged object array in our own export.
    with np.load(reference, allow_pickle=True) as d:
        ref = {k: d[k] for k in d.files}
    ctx = SimpleNamespace(tmp=tmp, deploy=args.deploy, report=report, ref=ref)
    exit_code = 0
    try:
        for name, fn in CHECKS:
            try:
                detail = fn(ctx)
            except Skip as exc:
                print(f"[SKIP] {name}: {exc}")
                continue
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        print(f"ALL LATENT ENCODER CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
