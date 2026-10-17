#!/usr/bin/env python3
"""Re-render the camera clouds of recorded LCS episodes from their recorded scene states.

No physics, controller or LCM: the scene is built once, each frame's ``body_q`` is set from
:class:`~task_common.prediction_video.EpisodeScene` (FK of the recorded arm joints, Franka fingers
from ``sim_hand_mm``, the recorded large-pulley pose, belt bodies at ``sim_belt_xyz``; Robotiq
linkage, small pulley and belt twist from the start snapshot) and the scene's cameras are
rendered. ``pcd``/``pcd_rgb``/``pcd_belt_seg``/``sim_cloud_*`` are rebuilt as the live collector
does with cloud noise ``off`` (``eval_learned_mpc._sample``); every other key is copied as is.
``sim_meta.belt_seg.source = rerender`` and ``sim_meta.rerender`` names the source episode.
The scene is the default one or ``ROUND_BELT_SCENE`` (e.g. the ZED720 variant).

    # compare against the recorded clouds (default scene), no writes
    uv run --frozen python scripts/lcs/rerender_clouds.py validate --episode A.npz --episode B.npz
    # re-render every episode of tier buffers, write re-pointed tiers
    ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_zed720.yaml \\
    uv run --frozen python scripts/lcs/rerender_clouds.py run \\
        --tier T1ga_f_tr_d=data/lcs/realfirst/cmp/v12_71/tiers/T1ga_f_tr/buffer.json \\
        --episodes-out data/lcs/realfirst/dense_rr/episodes --tiers-out data/lcs/realfirst/dense_rr/tiers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import belt_seg_offline as bso  # noqa: E402
import warp as wp  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from round_belt_task.scene import build_scene  # noqa: E402
from task_common import belt_cloud as bc  # noqa: E402
from task_common import belt_cloud_features as bcf  # noqa: E402
from task_common import lcs_dataset as lcs  # noqa: E402
from task_common import prediction_video as pv  # noqa: E402
from task_common.cameras import RgbdCameras  # noqa: E402
from task_common.point_cloud import CroppedPointCloud  # noqa: E402
from task_common.scene import make_builder  # noqa: E402

REALFIRST = REPO_ROOT / "data" / "lcs" / "realfirst"
NOISE_OFF = bc.CLOUD_NOISE_PRESETS["off"]
CLOUD_KEYS = ("pcd", "pcd_rgb", "pcd_belt_seg", *(lcs.EXTRA_PREFIX + k for k in bcf.EXTRA_KEYS))


class RerenderScene(pv.EpisodeScene):
    """EpisodeScene with the belt axis change taken from the non-uniform 3-point tangent at each body
    centre (capsule axis error vs the chord: median 0.45 vs 0.68 deg over start-state pairs)."""

    @staticmethod
    def _dirs(p: np.ndarray) -> np.ndarray:
        a, b = np.roll(p, 1, axis=0) - p, np.roll(p, -1, axis=0) - p
        ha, hb = (np.linalg.norm(v, axis=1, keepdims=True) for v in (a, b))
        t = b * ha ** 2 - a * hb ** 2
        return t / np.linalg.norm(t, axis=1, keepdims=True)


class Renderer:
    """The scene's cameras + cropped cloud on a kinematic state (no solver)."""

    def __init__(self) -> None:
        builder = make_builder()
        info = build_scene(builder)
        builder.color()  # as BeltTaskSimulation
        self.model = builder.finalize()
        self.cameras = RgbdCameras(self.model, info.cameras)
        spec = next(s for s in info.point_clouds if s.name == bc.POINT_CLOUD_NAME)
        self.cloud = CroppedPointCloud(self.cameras, spec)
        self.state = self.model.state()
        self.origins, self.pose = bc.scene_camera_origins(), bc.known_large_pulley_pose7()

    def episode(self, ep: pv.Episode, scene_cls=RerenderScene) -> dict:
        """Per-frame ``pcd``, ``pcd_rgb``, ``pcd_belt_seg`` (noise off) + ``sim_cloud_*``."""
        scene = scene_cls(self.model, ep)
        d, T = ep.data, ep.frames
        if "sim_render_step" in d and not np.array_equal(d["sim_render_step"], d["sim_step"]):
            raise ValueError(f"{ep.path}: recorded clouds not rendered at the recorded states")
        pcd, rgb, seg = (np.empty(T, dtype=object) for _ in range(3))
        feats: dict[str, list] = {k: [] for k in bcf.EXTRA_KEYS}
        rng = np.random.default_rng(0)  # unused: noise off draws nothing
        for k in range(T):
            with wp.ScopedDevice(self.model.device):
                self.state.body_q.assign(scene.body_q(k))
                self.cameras.update(self.state)
                xyz, colours = self.cloud.compute()
            jaws = bc.jaw_centres_from_poses(d["sim_ee_franka"][k], d["sim_ee_ur"][k])
            out = bc.belt_seg_frame(xyz, colours, NOISE_OFF, bc.DEFAULT_SEG, rng, self.origins,
                                    jaws, self.pose)
            pcd[k], rgb[k], seg[k] = lcs.camera_points(out["pcd"]), out["pcd_rgb"], out["pcd_belt_seg"]
            for key, v in bcf.cloud_extras(out["pcd_belt_seg"]).items():
                feats[key].append(v)
        out = {"pcd": pcd, "pcd_rgb": rgb, "pcd_belt_seg": seg}
        out.update({lcs.EXTRA_PREFIX + k: np.stack(v) for k, v in feats.items()})
        return out


def rerender_block(src: Path) -> dict:
    return {"source_episode": str(src), "source_sha256": bso._sha256(src), "scene": bc.scene_info(),
            "state": "EpisodeScene (recorded joints/hand/pulley/belt; Robotiq linkage + belt twist "
                     "from the start snapshot)", "script": "scripts/lcs/rerender_clouds.py",
            "git": bso._git()}


def write_episode(src: Path, dst: Path, ep: pv.Episode, clouds: dict, pose: np.ndarray,
                  origins: np.ndarray) -> dict:
    arrays = dict(ep.data)
    missing = [k for k in CLOUD_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"{src}: no {missing}")
    meta = json.loads(str(arrays["sim_meta"]))
    meta["belt_seg"] = bc.belt_seg_meta(NOISE_OFF, bc.DEFAULT_SEG, "rerender", pose)
    meta["cloud_noise"] = {**NOISE_OFF.to_dict(), "camera_origins": origins.tolist()}
    meta["rerender"] = rerender_block(src)
    arrays.update(clouds)
    arrays["sim_meta"] = np.array(json.dumps(meta, sort_keys=True, default=lcs.json_default))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, dst)
    lcs.validate_episode(dst, period_us=None)
    n = np.array([len(s) for s in clouds["pcd_belt_seg"]])
    return {"frames": len(n), "seg_median": float(np.median(n)), "seg_max": int(n.max())}


# --- validation -------------------------------------------------------------------------------

def compare_clouds(a: np.ndarray, b: np.ndarray) -> dict:
    """``a`` re-rendered vs ``b`` recorded: count diff, symmetric Chamfer RMS (mm), exact matches."""
    a, b = np.asarray(a, np.float64).reshape(-1, 3), np.asarray(b, np.float64).reshape(-1, 3)
    if not len(a) or not len(b):
        return {"n_new": len(a), "n_rec": len(b), "chamfer_rms_mm": float("nan"),
                "exact_frac": float(len(a) == len(b) == 0), "set_equal": len(a) == len(b) == 0}
    dab, _ = cKDTree(b).query(a)
    dba, _ = cKDTree(a).query(b)
    exact = int((dab == 0).sum())
    return {"n_new": len(a), "n_rec": len(b),
            "chamfer_rms_mm": 1e3 * float(np.sqrt(0.5 * ((dab ** 2).mean() + (dba ** 2).mean()))),
            "max_nn_mm": 1e3 * float(max(dab.max(), dba.max())),
            "nn_median_mm": 1e3 * float(np.median(np.concatenate([dab, dba]))),
            "within_0p1mm": float(np.mean(np.concatenate([dab, dba]) <= 1e-4)),
            "exact_frac": exact / max(len(a), len(b)),
            "set_equal": len(a) == len(b) and exact == len(a) and int((dba == 0).sum()) == len(b),
            "_unmatched_new": a[dab > 1e-4], "_unmatched_rec": b[dba > 1e-4]}


def validate(paths: list[Path], report: Path | None, chord: bool = False) -> int:
    r = Renderer()
    scene_cls = pv.EpisodeScene if chord else RerenderScene
    rows, near = [], {"ur": 0, "franka": 0, "other": 0}
    for path in paths:
        ep = pv.Episode.load(path)
        t0 = time.perf_counter()
        new = r.episode(ep, scene_cls)
        dt = time.perf_counter() - t0
        scene = scene_cls(r.model, ep)
        for key in ("pcd", "pcd_belt_seg"):
            res = [compare_clouds(new[key][k], ep.data[key][k]) for k in range(ep.frames)]
            if key == "pcd":
                for k, c in enumerate(res):  # where the unmatched points are (world frame)
                    ur, fr = ep.data["sim_ee_ur"][k, :3], ep.data["sim_ee_franka"][k, :3]
                    for p in np.concatenate([c["_unmatched_new"], c["_unmatched_rec"]]):
                        du, df = np.linalg.norm(p - ur), np.linalg.norm(p - fr)
                        near["ur" if du < 0.06 and du <= df else "franka" if df < 0.06 else "other"] += 1
            dn = np.array([c["n_new"] - c["n_rec"] for c in res])
            ch = np.array([c["chamfer_rms_mm"] for c in res])
            row = {"episode": str(path), "key": key, "frames": ep.frames,
                   "render_s_per_frame": dt / ep.frames,
                   "set_equal_frac": float(np.mean([c["set_equal"] for c in res])),
                   "exact_frac_mean": float(np.mean([c["exact_frac"] for c in res])),
                   "dn_mean_abs": float(np.abs(dn).mean()), "dn_max_abs": int(np.abs(dn).max()),
                   "n_rec_median": float(np.median([c["n_rec"] for c in res])),
                   "nn_median_mm": float(np.median([c.get("nn_median_mm", np.nan) for c in res])),
                   "within_0p1mm_mean": float(np.mean([c.get("within_0p1mm", np.nan) for c in res])),
                   "chamfer_rms_mm_median": float(np.nanmedian(ch)),
                   "chamfer_rms_mm_max": float(np.nanmax(ch)),
                   "max_nn_mm_max": float(np.nanmax([c.get("max_nn_mm", np.nan) for c in res]))}
            rows.append(row)
            print(json.dumps(row), flush=True)
        feats = {k: float(np.nanmax(np.abs(new[lcs.EXTRA_PREFIX + k].astype(np.float64)
                                           - ep.data[lcs.EXTRA_PREFIX + k].astype(np.float64))))
                 for k in bcf.EXTRA_KEYS}
        ee = [scene.ee_error_mm(k) for k in range(ep.frames)]
        byte = ep.data["sim_robotiq_byte"]
        extra = {"episode": str(path), "sim_cloud_max_abs_diff": feats,
                 "fk_ee_error_mm_max": float(np.max(ee)),
                 "robotiq_byte_range": [int(byte.min()), int(byte.max())]}
        rows.append(extra)
        print(json.dumps(extra), flush=True)
    print(json.dumps({"unmatched_pcd_points_near": near}), flush=True)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"rows": rows, "unmatched_near": near}, indent=1) + "\n")
    return 0


# --- run over tier buffers --------------------------------------------------------------------

def out_rel(src: Path) -> Path:
    """``src`` relative to data/lcs/realfirst (else to data/lcs under ``_lcs``)."""
    src = src.resolve()
    return src.relative_to(REALFIRST) if src.is_relative_to(REALFIRST) else Path("_lcs") / bso.relative_dir(src)


def _status(path: Path | None, **kw) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(kw, indent=1) + "\n")
        os.replace(tmp, path)


def run(args: argparse.Namespace) -> int:
    tiers = [t.split("=", 1) for t in args.tier]
    ep_out, tiers_out = args.episodes_out.resolve(), args.tiers_out.resolve()
    bufs = {name: (Path(src).resolve(), json.loads(Path(src).read_text())) for name, src in tiers}
    for name in bufs:
        if (tiers_out / name).exists():
            raise SystemExit(f"{tiers_out / name} exists; refusing to overwrite")
    srcs = list(dict.fromkeys(Path(r["episode"]).resolve() for _, b in bufs.values() for r in b["rollouts"]))
    dsts = {s: ep_out / out_rel(s) for s in srcs}
    if len(set(dsts.values())) != len(dsts):
        raise SystemExit("output name collision")
    for s, d in dsts.items():
        if d.resolve().parent == s.parent or ep_out.is_relative_to(s.parent):
            raise SystemExit(f"{d} would be written into the input dir {s.parent}")
    existing = [d for d in dsts.values() if d.exists()]
    if existing and not args.resume:
        raise SystemExit(f"{len(existing)} outputs exist (e.g. {existing[0]}); --resume to skip them")
    todo = [s for s in srcs if not dsts[s].exists()]
    for s in srcs:
        if dsts[s].exists():
            with np.load(dsts[s], allow_pickle=True) as d:
                got = json.loads(str(d["sim_meta"])).get("rerender", {}).get("source_episode")
            if got != str(s):
                raise SystemExit(f"{dsts[s]} exists but re-renders {got}, not {s}")
    n_frames = {Path(row["episode"]).resolve(): row["n_frames"] for _, b in bufs.values() for row in b["rollouts"]}
    total = sum(n_frames[s] for s in todo)
    print(f"[RR] scene {bc.scene_info()}; {len(srcs)} episodes, {len(todo)} to render ({total} frames)",
          flush=True)
    r = Renderer()
    t0, frames, rows = time.perf_counter(), 0, []
    pool, pending = ThreadPoolExecutor(max_workers=args.writers), []
    for i, src in enumerate(todo):
        ep = pv.Episode.load(src)
        clouds = r.episode(ep)
        pending.append((src, pool.submit(write_episode, src, dsts[src], ep, clouds, r.pose, r.origins)))
        while len(pending) > 2 * args.writers or (i == len(todo) - 1 and pending):
            s, fut = pending.pop(0)
            row = fut.result()
            frames += row["frames"]
            rows.append({"episode": str(s), **row})
        el = time.perf_counter() - t0
        rate = frames / el if el > 0 else 0.0
        _status(args.status, done=len(rows), queued=i + 1, total=len(todo), frames_done=frames,
                frames_total=total, elapsed_s=round(el, 1), frames_per_s=round(rate, 2),
                eta_min=round((total - frames) / rate / 60, 1) if rate else None, last=str(src))
        if (i + 1) % 20 == 0 or i == len(todo) - 1:
            print(f"[RR] {i + 1}/{len(todo)} episodes, {frames} frames written, {rate:.2f} frames/s, "
                  f"{el / 60:.1f} min", flush=True)
    pool.shutdown()
    for name, (src, buf) in bufs.items():
        out = []
        for row in buf["rollouts"]:
            s = Path(row["episode"]).resolve()
            out.append(dict(row, episode=str(dsts[s]), source_episode=row.get("source_episode", row["episode"]),
                            rerender_of=row["episode"]))
        prev = buf.get("derived_from")
        buf = dict(buf, rollouts=out, tier=name)
        buf["derived_from"] = {"buffer": str(src), "rerender": {"scene": bc.scene_info(),
                               "noise": "off", "script": "scripts/lcs/rerender_clouds.py", "git": bso._git()},
                               "source_derived_from": prev}
        (tiers_out / name).mkdir(parents=True)
        (tiers_out / name / "buffer.json").write_text(json.dumps(buf, indent=1) + "\n")
        print(f"[RR] tier {name}: {len(out)} rows -> {tiers_out / name / 'buffer.json'}", flush=True)
    _status(args.status, done=len(rows), total=len(todo), frames_done=frames, finished=True,
            elapsed_s=round(time.perf_counter() - t0, 1), tiers=list(bufs))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="re-render and compare with the recorded clouds")
    v.add_argument("--episode", type=Path, action="append", required=True)
    v.add_argument("--report", type=Path)
    v.add_argument("--chord", action="store_true", help="EpisodeScene's chord belt axes (A/B)")
    r = sub.add_parser("run", help="re-render all episodes of tier buffers")
    r.add_argument("--tier", action="append", required=True, metavar="NAME=BUFFER_JSON")
    r.add_argument("--episodes-out", type=Path, required=True)
    r.add_argument("--tiers-out", type=Path, required=True)
    r.add_argument("--status", type=Path)
    r.add_argument("--resume", action="store_true", help="skip episodes already written")
    r.add_argument("--writers", type=int, default=3, help="npz compression threads")
    a = p.parse_args()
    return validate(a.episode, a.report, a.chord) if a.cmd == "validate" else run(a)


if __name__ == "__main__":
    sys.exit(main())
