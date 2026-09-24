#!/usr/bin/env python3
"""Per-stage latent goals of the demonstration episode for the learned-MPC harness.

Reads the demo episode (``data/lcs/demo/demo_episode.npz``) and the demo re-export's
``deploy.npz``, re-encodes every demo frame with ``task_common.latent_encoder`` and writes
``demo_goals.npz`` (the ``DemoGoals`` schema: stage goals ``pre_place_1`` = frame 0 and
``place_3`` = frame T-1, the demo's poses/belts, ``z``, ``goal_dist``, tolerances) plus
``<out stem>_report.json`` with the tolerance calibration and the recommended ``learned_mpc``
yaml values. Stage 0 tol = 1.5 x p90 of the whitened distance from ``z_goal_stage1`` to frame 0
of every training file; stage 1 tol = the export's ``goal_tol_whitened``.

Run:
    uv run python scripts/lcs/make_demo_goals.py --episode data/lcs/demo/demo_episode.npz \\
        --deploy <deploy_demo>/deploy.npz \\
        --train-glob 'data/lcs/20260923-210508-ep300-ou/*/episode_*.npz' \\
        --out data/lcs/demo/demo_goals.npz
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.latent_encoder import DemoGoals, LatentEncoder, LearnedLcs
from task_common.osc_process import sha256

STAGE_LABELS = ("pre_place_1", "place_3")
MAX_DURATIONS_S = (4.0, 6.0)
STAGE0_TOL_FACTOR = 1.5
STAGE0_PERCENTILE = 90.0
START_POSE_TOL_M = 1e-3
REENCODE_TOL = 1e-5
FINGER_TIP_BODY = "panda_hand/finger_tip"
TRAIN_GLOB = "data/lcs/20260923-210508-ep300-ou/*/episode_*.npz"


def rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def demo_frames(data) -> dict:
    """Frame indices of the stage goals, asserted against the phase labels and the start state."""
    meta = json.loads(str(data["sim_meta"]))
    labels = meta["phase_labels"]
    phase = np.asarray(data["sim_phase"], dtype=np.int64)
    n = len(phase)
    i, j = 0, n - 1
    _require(labels[phase[i]].startswith("move:"), f"frame 0 phase {labels[phase[i]]!r}")
    _require(labels[phase[j]] == "done", f"frame {j} phase {labels[phase[j]]!r} != 'done'")
    hold = [t for t in range(n) if labels[phase[t]] == f"hold:{STAGE_LABELS[1]}"]
    _require(bool(hold), f"no hold:{STAGE_LABELS[1]} frame")
    snap = sim_snapshot.load(Path(meta["start_state"]))
    tip = snap.body_q[snap.meta["body_labels"].index(FINGER_TIP_BODY), :3]
    start_err = float(np.linalg.norm(np.asarray(data["state"])[i, 26:29] - tip))
    _require(start_err <= START_POSE_TOL_M,
             f"frame 0 finger_tip {start_err * 1e3:.3f} mm from the start snapshot")
    return {"i": i, "j": j, "T": n, "first_hold_place_3_frame": hold[0],
            "labels": [labels[phase[i]], labels[phase[j]]], "start_err_mm": start_err * 1e3,
            "start_state": meta["start_state"], "meta": meta}


def encode_frames(enc: LatentEncoder, data) -> np.ndarray:
    pcd, state, belt = data["pcd"], np.asarray(data["state"]), np.asarray(data["pcd_belt"])
    return np.stack([enc.encode(pcd[t], state[t], belt[t]) for t in range(len(state))])


def recording_dir(episode: Path, demo_sha: str) -> dict | None:
    """The collection run (and its recording) the demo file was copied from."""
    for cand in sorted(episode.parent.glob("*/episode_*.npz")):
        if sha256(cand) != demo_sha:
            continue
        index = json.loads((cand.parent / "index.json").read_text())
        row = next(r for r in index["episodes"] if r.get("file") == cand.name)
        rec = row.get("recording")
        return {"run_dir": rel(cand.parent), "file": cand.name,
                "recording": None if rec is None else rel(cand.parent / rec),
                "scenario": index.get("scenario"), "seed": index["args"].get("seed")}
    return None


def stage0_floor(enc: LatentEncoder, goal: np.ndarray, z_std: np.ndarray, pattern: str
                 ) -> tuple[np.ndarray, int]:
    files = sorted(glob.glob(str(REPO_ROOT / pattern) if not Path(pattern).is_absolute()
                             else pattern))
    _require(bool(files), f"no training files match {pattern!r}")
    dist = []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            z0 = enc.encode(d["pcd"][0], d["state"][0], d["pcd_belt"][0])
        dist.append(float(np.linalg.norm((z0 - goal) / z_std)))
    return np.asarray(dist), len(files)


def _first_under(col: np.ndarray, tol: float) -> int | None:
    hit = np.flatnonzero(col < tol)
    return int(hit[0]) if hit.size else None


def build(episode: Path, deploy: Path, train_glob: str) -> tuple[dict, dict]:
    episode, deploy = Path(episode), Path(deploy)
    lcs.validate_episode(episode)
    enc, model = LatentEncoder.load(deploy), LearnedLcs.load(deploy)
    _require(model.stage_goals.shape[0] == 2, f"{deploy} has no stage goals")
    with np.load(episode, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files}
    outcome = str(data["sim_outcome"])
    _require(outcome == "engaged", f"demo outcome {outcome!r} != 'engaged'")
    fr = demo_frames(data)
    frames = np.array([fr["i"], fr["j"]], dtype=np.int64)
    z = encode_frames(enc, data)
    z_goals = model.stage_goals.copy()
    reencode_err = float(np.abs(z[frames] - z_goals).max())
    _require(reencode_err <= REENCODE_TOL, f"deploy stage goals vs re-encode {reencode_err:.2e}")
    z_std = model.z_std
    goal_dist = np.stack([[np.linalg.norm((zt - g) / z_std) for g in z_goals] for zt in z])

    floor, n_train = stage0_floor(enc, z_goals[0], z_std, train_glob)
    p50, p90 = (float(np.percentile(floor, q)) for q in (50.0, STAGE0_PERCENTILE))
    tols = np.array([STAGE0_TOL_FACTOR * p90, model.goal_tol])

    demo_sha, deploy_sha = sha256(episode), sha256(deploy)
    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta = fr["meta"]
    pert = json.loads(str(data["sim_perturbation"]))
    excite_on = bool(meta["excitation"]["on"])
    notes = (f"stage goals = demo frames {fr['i']} ({fr['labels'][0]}) and {fr['j']} "
             f"({fr['labels'][1]}); z_goals copied from the deploy stage keys; stage 0 tol = "
             f"{STAGE0_TOL_FACTOR} x p{STAGE0_PERCENTILE:g} of the training frame-0 floor, "
             f"stage 1 tol = deploy goal_tol_whitened")
    arrays = {
        "n_stages": np.int64(2), "stage_labels": np.array(STAGE_LABELS),
        "stage_frames": frames, "z_goals": z_goals, "goal_tols": tols,
        "max_durations_s": np.array(MAX_DURATIONS_S),
        "ee_pose_franka": np.asarray(data["sim_ee_franka"])[frames].astype(np.float64),
        "ee_pose_ur": np.asarray(data["sim_ee_ur"])[frames].astype(np.float64),
        "z": z, "goal_dist": goal_dist,
        "pcd_belt_stage": np.asarray(data["pcd_belt"])[frames].astype(np.float64),
        "belt_xyz_stage": np.asarray(data["sim_belt_xyz"])[frames].astype(np.float64),
        "pulley_pose_stage": np.asarray(data["sim_pulley_large_pose"])[frames].astype(
            np.float64),
        "state_stage": np.asarray(data["state"])[frames].astype(np.float64),
        "first_hold_place_3_frame": np.int64(fr["first_hold_place_3_frame"]),
        "z_std": z_std.copy(), "demo_episode": np.array(rel(episode)),
        "demo_sha256": np.array(demo_sha), "deploy_sha256": np.array(deploy_sha),
        "created": np.array(created), "notes": np.array(notes),
    }
    DemoGoals(arrays)

    export = json.loads((deploy.parent / "report.json").read_text())
    report = {
        "created": created, "demo_episode": rel(episode), "demo_sha256": demo_sha,
        "deploy": str(deploy.resolve()), "deploy_sha256": deploy_sha,
        "goal_source": model.goal_source,
        "demo": {
            "source": recording_dir(episode, demo_sha), "scenario": meta.get("scenario"),
            "intent": meta.get("intent"), "perturbation": pert, "excitation_on": excite_on,
            "outcome": outcome, "final_wrap_deg": float(data["sim_wrap_deg"][-1]),
            "final_h_median_mm": float(data["sim_h_median_mm"][-1]),
            "slant_deg": float(data["sim_slant_deg"]), "slant_dir": str(data["sim_slant_dir"]),
            "min_board_clearance_mm": float(data["sim_min_board_clearance_mm"]),
            "board_contact": bool(data["sim_board_contact"]),
        },
        "frames": {"T": fr["T"], "stage_frames": frames.tolist(),
                   "stage_phase_labels": fr["labels"],
                   "first_hold_place_3_frame": fr["first_hold_place_3_frame"],
                   "frame0_vs_start_state_mm": fr["start_err_mm"],
                   "start_state": fr["start_state"]},
        "reencode_max_abs_err": reencode_err,
        "stage_1_place_3": {
            "tol": float(tols[1]), "source": "deploy goal_tol_whitened (engaged p90 vs stage 2)",
            "separation": export["separation"],
        },
        "stage_0_pre_place_1": {
            "tol": float(tols[0]), "factor": STAGE0_TOL_FACTOR, "percentile": STAGE0_PERCENTILE,
            "train_glob": train_glob, "n_train_files": n_train,
            "floor_p50": p50, "floor_p90": p90, "floor_min": float(floor.min()),
            "floor_max": float(floor.max()),
            "source": "whitened distance from z_goal_stage1 to frame 0 of every training file",
        },
        "demo_goal_dist": {
            label: {"tol": float(tols[k]),
                    "first_frame_under_tol": _first_under(goal_dist[:, k], tols[k]),
                    "at_frame_0": float(goal_dist[0, k]),
                    "at_frame_last": float(goal_dist[-1, k]),
                    "trajectory": [round(float(v), 4) for v in goal_dist[:, k]]}
            for k, label in enumerate(STAGE_LABELS)
        },
        "recommended_learned_mpc_yaml": {
            "stage_goal_tols": [float(t) for t in tols],
            "stage_max_durations_s": list(MAX_DURATIONS_S),
        },
    }
    return arrays, report


def report_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}_report.json")


def write(arrays: dict, report: dict, out: Path) -> tuple[Path, Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    rep = report_path(out)
    rep.write_text(json.dumps(report, indent=1) + "\n")
    return out, rep


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--deploy", type=Path, required=True)
    p.add_argument("--train-glob", default=TRAIN_GLOB)
    p.add_argument("--out", type=Path, required=True)
    return p


def main() -> int:
    args = create_parser().parse_args()
    arrays, report = build(args.episode, args.deploy, args.train_glob)
    out, rep = write(arrays, report, args.out)
    s0, s1 = report["stage_0_pre_place_1"], report["stage_1_place_3"]
    print(f"wrote {out} and {rep}")
    print(f"frames {report['frames']['stage_frames']} (T {report['frames']['T']}), first "
          f"hold:place_3 {report['frames']['first_hold_place_3_frame']}, re-encode err "
          f"{report['reencode_max_abs_err']:.2e}")
    print(f"stage 0 floor over {s0['n_train_files']} files: p50 {s0['floor_p50']:.3f} "
          f"p90 {s0['floor_p90']:.3f} -> tol {s0['tol']:.3f}")
    print(f"stage 1 tol {s1['tol']:.3f}")
    print(f"{'outcome':<9}{'n':>4}{'p10':>8}{'p50':>8}{'p90':>8}{'<tol':>7}")
    for name, row in s1["separation"]["by_outcome"].items():
        print(f"{name:<9}{row['n']:>4}{row['p10']:>8.2f}{row['p50']:>8.2f}{row['p90']:>8.2f}"
              f"{row['frac_below_tol']:>7.2f}")
    for label, row in report["demo_goal_dist"].items():
        print(f"demo {label}: first frame under tol {row['first_frame_under_tol']}, "
              f"frame 0 {row['at_frame_0']:.3f}, last {row['at_frame_last']:.3f}")
    print(f"learned_mpc: {json.dumps(report['recommended_learned_mpc_yaml'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
