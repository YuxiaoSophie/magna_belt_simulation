#!/usr/bin/env python3
"""Gate G1: cloud h / wrap / success estimates vs sim ground truth on recorded episodes.

For every ``episode_*.npz`` under ``--dirs`` (recursive), every ``--stride``-th frame is a
candidate, labelled with the GT class ``outcome.classify(outcome.frame_metrics(sim_belt_xyz,
sim_pulley_large_pose))``. The sample is stratified: up to ``--per-class`` frames of each GT class
(seeded draw), topped up at random to ``--min-frames``. Per preset, each sampled frame's recorded
``pcd``/``pcd_rgb`` goes through a1's noise + colour segmentation in memory
(``belt_cloud.belt_seg_frame``, RNG ``[seed, crc32(episode), t, 11]``, jaws = ``sim_ee_*``, known
pulley pose), then ``belt_cloud_features.cloud_features``.

``h_bias_mm`` = median(raw h - GT h) over GT-engaged frames under ``off`` (applied to every
preset). Detector rules vs GT: strict (wrap >= 90, |h| <= 1) and engaged (classify == engaged).
The PASS line uses the plan's G1 numbers on ``real_like`` with the detector scored vs GT
engaged (user decision Q1); strict is reported. Thresholds are swept and the best F1 on
``real_like`` written as ``chosen``. Info rows (not gated): ``episode_cloud_success`` over each
episode's last ``--last-n`` frames vs the emulated human label (``human_label.py``).

    uv run --frozen python scripts/lcs/perception_gate.py --dirs data/lcs/approach/v1/train \\
        data/lcs/contact/v1/train data/lcs/mpc_eval/20260929-v4-retrain/t1 \\
        --presets off,mild,real_like,harsh --out data/lcs/realfirst/v1/perception_gate/
    uv run --frozen python scripts/lcs/perception_gate.py --self-test
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import subprocess
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import human_label as hl

from round_belt_task import outcome as oc
from task_common import belt_cloud as bc
from task_common import belt_cloud_features as bcf

CLOUD_NOISE_RNG = 11
GT_STRICT_WRAP_DEG, GT_STRICT_H_MM = 90.0, 1.0
GATE_PRESET = "real_like"
GATE = {"dh_median_mm": 2.0, "dh_p90_mm": 4.0, "dwrap_median_deg": 15.0, "precision": 0.90,
        "recall": 0.80}
SWEEP_STRICT_WRAP = (70.0, 75.0, 80.0, 85.0, 90.0)
SWEEP_STRICT_H = (1.5, 2.0, 2.5, 3.0, 3.5)
SWEEP_ENGAGED_WRAP = (40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0)
NEEDED = ("pcd", "pcd_rgb", "sim_belt_xyz", "sim_pulley_large_pose", "sim_ee_franka", "sim_ee_ur")


def _git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _rel(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def gt_frame(belt_xyz, pose_xyz_wxyz) -> dict:
    """GT metrics of one frame (``sim_pulley_large_pose`` is xyz_wxyz; outcome wants xyzw)."""
    p = np.asarray(pose_xyz_wxyz, dtype=np.float64)
    fm = oc.frame_metrics(np.asarray(belt_xyz, dtype=np.float64),
                          np.concatenate([p[:3], p[4:7], p[3:4]]))
    return {"cls": oc.classify(fm), "h": fm.h_median_mm, "wrap": fm.wrap_deg,
            "strict": bool(fm.wrap_deg >= GT_STRICT_WRAP_DEG
                           and abs(fm.h_median_mm) <= GT_STRICT_H_MM)}


# --- sampling ---------------------------------------------------------------------------------

def candidates(dirs: list[Path], stride: int) -> tuple[list[dict], list[dict]]:
    rows, skipped = [], []
    for d in dirs:
        for f in sorted(d.rglob("episode_*.npz")):
            with np.load(f, allow_pickle=True) as data:
                missing = [k for k in NEEDED if k not in data.files]
                if missing:
                    skipped.append({"file": _rel(f), "missing": missing})
                    continue
                belt, pose = data["sim_belt_xyz"], data["sim_pulley_large_pose"]
            for t in range(0, len(belt), stride):
                rows.append({"file": _rel(f), "t": t, **gt_frame(belt[t], pose[t])})
    return rows, skipped


def stratify(rows: list[dict], per_class: int, min_frames: int, seed: int) -> list[dict]:
    rng = np.random.default_rng([seed, 1])
    order = rng.permutation(len(rows))
    picked, count = [], Counter()
    for i in order:
        if count[rows[i]["cls"]] < per_class:
            picked.append(int(i))
            count[rows[i]["cls"]] += 1
    taken = set(picked)
    rest = [int(i) for i in order if int(i) not in taken]
    picked += rest[:max(0, min_frames - len(picked))]
    return [rows[i] for i in sorted(picked)]


def segment_frames(sample: list[dict], presets: list[str], seed: int) -> dict[str, list]:
    """``{preset: [(M, 3) pulley-frame belt cloud]}`` in sample order."""
    origins, pose, seg = bc.scene_camera_origins(), bc.known_large_pulley_pose7(), bc.BeltSegSpec()
    specs = {p: bc.cloud_noise_spec(p) for p in presets}
    out = {p: [None] * len(sample) for p in presets}
    by_file: dict[str, list[int]] = {}
    for k, row in enumerate(sample):
        by_file.setdefault(row["file"], []).append(k)
    for n, (rel, ks) in enumerate(by_file.items()):
        with np.load(REPO_ROOT / rel, allow_pickle=True) as data:
            pcd, rgb = data["pcd"], data["pcd_rgb"]
            ee_f, ee_u = data["sim_ee_franka"], data["sim_ee_ur"]
        for k in ks:
            t = sample[k]["t"]
            jaws = bc.jaw_centres_from_poses(ee_f[t], ee_u[t])
            for p, spec in specs.items():
                rng = np.random.default_rng([seed, zlib.crc32(rel.encode()), t, CLOUD_NOISE_RNG])
                fr = bc.belt_seg_frame(np.asarray(pcd[t], dtype=np.float32),
                                       np.asarray(rgb[t], dtype=np.uint8), spec, seg, rng,
                                       origins, jaws, pose)
                out[p][k] = fr["pcd_belt_seg"]
        if (n + 1) % 25 == 0:
            print(f"[G1] segmented {n + 1}/{len(by_file)} episodes", flush=True)
    return out


# --- statistics -------------------------------------------------------------------------------

def _q(x, q) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.percentile(x, q)) if x.size else float("nan")


def detector(pred, truth) -> dict:
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def evaluate(sample: list[dict], feats: list[bcf.CloudFeatures],
             th: bcf.CloudSuccessThresholds) -> dict:
    h_gt = np.array([r["h"] for r in sample])
    w_gt = np.array([r["wrap"] for r in sample])
    h_est = np.array([f.h_median_mm for f in feats])
    w_est = np.array([f.wrap_deg for f in feats])
    both = np.isfinite(h_gt) & np.isfinite(h_est)
    dh, dw = np.abs(h_est - h_gt)[both], np.abs(w_est - w_gt)
    gt_strict = np.array([r["strict"] for r in sample])
    gt_eng = np.array([r["cls"] == "engaged" for r in sample])
    det_s = np.array([bcf.cloud_success(f, "strict", th) for f in feats])
    det_e = np.array([bcf.cloud_success(f, "engaged", th) for f in feats])
    classes = {}
    for c in oc.LABELS:
        m = np.array([r["cls"] == c for r in sample])
        if not m.any():
            continue
        mb = m & both
        classes[c] = {"n": int(m.sum()), "n_h": int(mb.sum()),
                      "h_bias_mm": _q((h_est - h_gt)[mb], 50),
                      "dh_median_mm": _q(np.abs(h_est - h_gt)[mb], 50),
                      "wrap_bias_deg": _q((w_est - w_gt)[m], 50),
                      "dwrap_median_deg": _q(np.abs(w_est - w_gt)[m], 50),
                      "cloud_nan": int((~np.isfinite(h_est) & m).sum()),
                      "gt_strict": int((gt_strict & m).sum()),
                      "det_strict": int((det_s & m).sum()), "det_engaged": int((det_e & m).sum())}
    return {"n": len(sample), "n_h": int(both.sum()),
            "dh_median_mm": _q(dh, 50), "dh_p90_mm": _q(dh, 90),
            "dwrap_median_deg": _q(dw, 50), "dwrap_p90_deg": _q(dw, 90),
            "strict": detector(det_s, gt_strict), "engaged": detector(det_e, gt_eng),
            "by_class": classes}


def sweep(sample: list[dict], feats: list[bcf.CloudFeatures]) -> dict:
    gt_strict = [r["strict"] for r in sample]
    gt_eng = [r["cls"] == "engaged" for r in sample]
    strict = []
    for w, h in itertools.product(SWEEP_STRICT_WRAP, SWEEP_STRICT_H):
        th = bcf.CloudSuccessThresholds(strict_wrap_deg=w, strict_h_mm=h)
        strict.append({"strict_wrap_deg": w, "strict_h_mm": h,
                       **detector([bcf.cloud_success(f, "strict", th) for f in feats],
                                  gt_strict)})
    engaged = []
    for w in SWEEP_ENGAGED_WRAP:
        th = bcf.CloudSuccessThresholds(engaged_wrap_deg=w)
        engaged.append({"engaged_wrap_deg": w,
                        **detector([bcf.cloud_success(f, "engaged", th) for f in feats],
                                   gt_eng)})
    return {"strict": strict, "engaged": engaged}


def episode_level(files: list[str], presets: list[str], spec: bcf.CloudFeatureSpec,
                  seed: int, last_n: int) -> dict:
    """Info: ``episode_cloud_success`` (last ``last_n`` frames) vs the emulated human label."""
    origins, pose, seg = bc.scene_camera_origins(), bc.known_large_pulley_pose7(), bc.BeltSegSpec()
    noise = {p: bc.cloud_noise_spec(p) for p in presets}
    gt = {r: [] for r in bcf.RULES}
    det = {p: {r: [] for r in bcf.RULES} for p in presets}
    for rel in files:
        with np.load(REPO_ROOT / rel, allow_pickle=True) as d:
            data = {k: d[k] for k in NEEDED}
        for r in bcf.RULES:
            gt[r].append(hl.emulated_human_label(data, r, last_n)["success"])
        T = len(data["pcd"])
        for p, spec_n in noise.items():
            feats = []
            for t in range(max(0, T - last_n), T):
                rng = np.random.default_rng([seed, zlib.crc32(rel.encode()), t, CLOUD_NOISE_RNG])
                jaws = bc.jaw_centres_from_poses(data["sim_ee_franka"][t], data["sim_ee_ur"][t])
                fr = bc.belt_seg_frame(np.asarray(data["pcd"][t], dtype=np.float32),
                                       np.asarray(data["pcd_rgb"][t], dtype=np.uint8), spec_n,
                                       seg, rng, origins, jaws, pose)
                feats.append(bcf.cloud_features(fr["pcd_belt_seg"], spec))
            for r in bcf.RULES:
                det[p][r].append(bcf.episode_cloud_success(feats, r, last_n))
    return {"n_episodes": len(files), "last_n": last_n,
            "gt_successes": {r: int(sum(v)) for r, v in gt.items()},
            "presets": {p: {r: detector(det[p][r], gt[r]) for r in bcf.RULES} for p in presets}}


def _best(rows: list[dict], keys: tuple[str, ...], default: dict) -> dict:
    """Max F1, then precision, then the closest to the package default."""
    def score(r):
        prec = r["precision"] if np.isfinite(r["precision"]) else -1.0
        return (round(r["f1"], 6), round(prec, 6),
                -sum(abs(r[k] - default[k]) for k in keys))
    best = max(rows, key=score)
    return {k: best[k] for k in (*keys, "precision", "recall", "f1")}


def verdict(res: dict) -> tuple[bool, list[str]]:
    r = res[GATE_PRESET]
    checks = [("median |dh|", r["dh_median_mm"], "<=", GATE["dh_median_mm"], "mm"),
              ("p90 |dh|", r["dh_p90_mm"], "<=", GATE["dh_p90_mm"], "mm"),
              ("median |dwrap|", r["dwrap_median_deg"], "<=", GATE["dwrap_median_deg"], "deg"),
              ("engaged precision", r["engaged"]["precision"], ">=", GATE["precision"], ""),
              ("engaged recall", r["engaged"]["recall"], ">=", GATE["recall"], "")]
    misses = []
    for name, v, op, lim, unit in checks:
        ok = v <= lim if op == "<=" else v >= lim
        if not (np.isfinite(v) and ok):
            u = f" {unit}" if unit else ""
            misses.append(f"{name} {v:.3f}{u} vs {op} {lim} (miss by {abs(v - lim):.3f}{u})")
    return not misses, misses


# --- report -----------------------------------------------------------------------------------

def _f(v, nd=2) -> str:
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def markdown(report: dict) -> str:
    res, cfg = report["results"], report["config"]
    lines = [f"# G1 perception gate ({report['created']})", "",
             f"- dirs: {', '.join(cfg['dirs'])}",
             (f"- frames: every {cfg['stride']}th frame of {report['n_episodes']} episodes "
              f"({report['n_candidates']} candidates); stratified by GT per-frame class: up to "
              f"{cfg['per_class']} per class (seed {cfg['seed']}), topped up to "
              f"{cfg['min_frames']}; sampled {report['n_frames']} (sampled/available): "
              + ", ".join(f"{c} {n}/{report['candidates_by_class'].get(c, 0)}"
                          for c, n in report["sample_by_class"].items())),
             (f"- h_bias_mm = {_f(report['h_bias_mm'], 3)} (median raw h - GT h on "
              f"{report['h_bias_n']} GT-engaged frames under `off`; spec default "
              f"{cfg['spec']['h_bias_mm']})"),
             f"- thresholds (spec defaults): {json.dumps(cfg['thresholds'])}",
             f"- chosen on `{GATE_PRESET}` (best F1): {json.dumps(report['chosen'])}",
             ("- GT strict = wrap >= 90 deg and |h| <= 1 mm; GT engaged = classify == engaged; "
              "detector rules = `belt_cloud_features.cloud_success`"), "",
             "## Summary (one row per preset)", "",
             ("| preset | n | n_h | med abs dh mm | p90 abs dh mm | med abs dwrap deg | "
              "p90 abs dwrap deg | engaged P / R / F1 | strict P / R / F1 |"),
             "|---|---|---|---|---|---|---|---|---|"]
    for p, r in res.items():
        e, s = r["engaged"], r["strict"]
        lines.append(f"| {p} | {r['n']} | {r['n_h']} | {_f(r['dh_median_mm'])} | "
                     f"{_f(r['dh_p90_mm'])} | {_f(r['dwrap_median_deg'], 1)} | "
                     f"{_f(r['dwrap_p90_deg'], 1)} | {_f(e['precision'], 3)} / "
                     f"{_f(e['recall'], 3)} / {_f(e['f1'], 3)} | {_f(s['precision'], 3)} / "
                     f"{_f(s['recall'], 3)} / {_f(s['f1'], 3)} |")
    for p, r in res.items():
        lines += ["", f"## `{p}`", "",
                  ("| GT class | n | n_h | h bias mm | med abs dh mm | wrap bias deg | "
                   "med abs dwrap deg | cloud NaN | GT strict | det strict | det engaged |"),
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, v in r["by_class"].items():
            lines.append(f"| {c} | {v['n']} | {v['n_h']} | {_f(v['h_bias_mm'])} | "
                         f"{_f(v['dh_median_mm'])} | {_f(v['wrap_bias_deg'], 1)} | "
                         f"{_f(v['dwrap_median_deg'], 1)} | {v['cloud_nan']} | "
                         f"{v['gt_strict']} | {v['det_strict']} | {v['det_engaged']} |")
        for rule in ("engaged", "strict"):
            d = r[rule]
            lines.append(f"\n{rule} detector vs GT {rule}: TP {d['tp']} FP {d['fp']} "
                         f"FN {d['fn']} TN {d['tn']}")
        sw = report["sweeps"][p]
        lines += ["", ("engaged_wrap_deg sweep (P / R / F1): "
                       + "; ".join(f"{r_['engaged_wrap_deg']:g}: {_f(r_['precision'], 2)} / "
                                   f"{_f(r_['recall'], 2)} / {_f(r_['f1'], 2)}"
                                   for r_ in sw["engaged"])),
                  "", "strict sweep F1 (rows strict_wrap_deg, cols strict_h_mm):", "",
                  "| wrap \\ h | " + " | ".join(f"{h:g}" for h in SWEEP_STRICT_H) + " |",
                  "|---" * (len(SWEEP_STRICT_H) + 1) + "|"]
        for w in SWEEP_STRICT_WRAP:
            cells = [x for x in sw["strict"] if x["strict_wrap_deg"] == w]
            lines.append(f"| {w:g} | " + " | ".join(
                f"{_f(x['f1'], 2)} ({_f(x['precision'], 2)}/{_f(x['recall'], 2)})"
                for x in cells) + " |")
    ep = report.get("episode_level")
    if ep:
        lines += ["", (f"## Episode level (info, not gated): `episode_cloud_success` on the "
                       f"last {ep['last_n']} frames vs the emulated human label "
                       f"(`human_label.py`), {ep['n_episodes']} episodes, GT successes "
                       f"{json.dumps(ep['gt_successes'])}"),
                  "", "| preset | engaged P / R / F1 | strict P / R / F1 |", "|---|---|---|"]
        for p, v in ep["presets"].items():
            e, s_ = v["engaged"], v["strict"]
            lines.append(f"| {p} | {_f(e['precision'], 3)} / {_f(e['recall'], 3)} / "
                         f"{_f(e['f1'], 3)} | {_f(s_['precision'], 3)} / {_f(s_['recall'], 3)}"
                         f" / {_f(s_['f1'], 3)} |")
    lines += ["", (f"**G1 {report['verdict']}** ({GATE_PRESET}: median |dh| <= "
                   f"{GATE['dh_median_mm']} mm, p90 <= {GATE['dh_p90_mm']} mm, median |dwrap| <= "
                   f"{GATE['dwrap_median_deg']} deg, engaged precision >= {GATE['precision']}, "
                   f"recall >= {GATE['recall']})")]
    lines += [f"- miss: {m}" for m in report["misses"]]
    return "\n".join(lines) + "\n"


# --- self test --------------------------------------------------------------------------------

def synthetic_ring(spec: bcf.CloudFeatureSpec, h0_mm: float, gaps_deg) -> np.ndarray:
    """Torus surface (metres, pulley frame): centreline r = seat + tube radius (the outer face
    the cameras see), h = ``h0_mm`` + ``spec.h_bias_mm`` (the surface offset the bias removes),
    minus azimuth intervals ``gaps_deg``."""
    az = np.radians(np.arange(0.0, 360.0, 0.5))
    tube = np.radians(np.arange(0.0, 360.0, 30.0))
    a, b = np.meshgrid(az, tube, indexing="ij")
    rc = spec.seat_mm + spec.tube_radius_mm + spec.tube_radius_mm * np.cos(b)
    h = h0_mm + spec.h_bias_mm + spec.tube_radius_mm * np.sin(b)
    pts = np.stack([rc * np.cos(a), rc * np.sin(a), h], axis=-1).reshape(-1, 3)
    deg = np.mod(np.degrees(a.reshape(-1)), 360.0)
    keep = np.ones(len(pts), dtype=bool)
    for lo, hi in gaps_deg:
        keep &= ~((np.mod(deg - lo, 360.0)) < (hi - lo))
    return pts[keep] * 1e-3


def self_test() -> bool:
    spec = bcf.DEFAULT_SPEC
    jaws = [(-30.0, 0.0), (0.0, 30.0)]  # two adjacent 30 deg jaw footprints
    seated = bcf.cloud_features(synthetic_ring(spec, 0.0, jaws), spec)
    lifted = bcf.cloud_features(synthetic_ring(spec, 15.0, jaws), spec)
    opposed = bcf.cloud_features(synthetic_ring(spec, 0.0, [(-15.0, 15.0), (165.0, 195.0)]),
                                 spec)
    empty = bcf.cloud_features(np.zeros((0, 3)), spec)
    checks = [
        ("seated wrap in [280, 300]", 280.0 <= seated.wrap_deg <= 300.0, seated.wrap_deg),
        ("seated |h_median| <= 0.5", abs(seated.h_median_mm) <= 0.5, seated.h_median_mm),
        ("lifted 15 mm: |h - 15| <= 0.5", abs(lifted.h_median_mm - 15.0) <= 0.5,
         lifted.h_median_mm),
        ("lifted 15 mm: wrap 0", lifted.wrap_deg == 0.0, lifted.wrap_deg),
        ("seated success (engaged, strict)", bcf.cloud_success(seated, "engaged")
         and bcf.cloud_success(seated, "strict"), None),
        ("lifted no success", not (bcf.cloud_success(lifted, "engaged")
                                   or bcf.cloud_success(lifted, "strict")), None),
        ("empty cloud -> NaN, wrap 0, no success",
         np.isnan(empty.h_median_mm) and empty.wrap_deg == 0.0
         and not bcf.cloud_success(empty, "engaged"), None),
    ]
    ok = True
    for name, passed, value in checks:
        ok &= bool(passed)
        print(f"[SELF-TEST] {'PASS' if passed else 'FAIL'} {name}"
              + ("" if value is None else f" ({value:.3f})"))
    print(f"[SELF-TEST] info: opposed 30 deg jaws (0, 180) -> wrap {opposed.wrap_deg:.1f} "
          "(largest contiguous arc)")
    print(f"[SELF-TEST] {'PASS' if ok else 'FAIL'}")
    return ok


# --- main -------------------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dirs", type=Path, nargs="+")
    p.add_argument("--presets", default="off,mild,real_like,harsh")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--per-class", type=int, default=80)
    p.add_argument("--min-frames", type=int, default=200)
    p.add_argument("--last-n", type=int, default=hl.LAST_N,
                   help="episode-level info rows (0 = skip)")
    p.add_argument("--out", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return 0 if self_test() else 1
    if not args.dirs or args.out is None:
        p.error("--dirs and --out are required (or --self-test)")
    presets = [s.strip() for s in args.presets.split(",") if s.strip()]
    if "off" not in presets or GATE_PRESET not in presets:
        p.error(f"--presets must include 'off' (bias calibration) and '{GATE_PRESET}' (gate)")
    out = args.out.resolve()
    if (out / "gate.json").exists():
        raise SystemExit(f"{out / 'gate.json'} exists; refusing to overwrite")
    for d in args.dirs:
        if not d.is_dir():
            raise SystemExit(f"--dirs {d} is not a directory")

    t0 = time.perf_counter()
    rows, skipped = candidates([d.resolve() for d in args.dirs], args.stride)
    sample = stratify(rows, args.per_class, args.min_frames, args.seed)
    print(f"[G1] {len(rows)} candidates, {len(sample)} sampled "
          f"{dict(Counter(r['cls'] for r in sample))}; skipped files {len(skipped)}", flush=True)
    clouds = segment_frames(sample, presets, args.seed)

    raw = dataclasses.replace(bcf.DEFAULT_SPEC, h_bias_mm=0.0)
    eng = [k for k, r in enumerate(sample) if r["cls"] == "engaged"]
    diffs = [bcf.cloud_features(clouds["off"][k], raw).h_median_mm - sample[k]["h"] for k in eng]
    diffs = [d for d in diffs if np.isfinite(d)]
    h_bias = float(np.median(diffs)) if diffs else float("nan")
    spec = dataclasses.replace(bcf.DEFAULT_SPEC, h_bias_mm=round(h_bias, 1))
    print(f"[G1] h_bias_mm {h_bias:.3f} on {len(diffs)} GT-engaged frames (off); "
          f"features with {spec.h_bias_mm} (spec default {bcf.DEFAULT_SPEC.h_bias_mm})",
          flush=True)

    results, sweeps = {}, {}
    for pr in presets:
        feats = [bcf.cloud_features(c, spec) for c in clouds[pr]]
        results[pr] = evaluate(sample, feats, bcf.DEFAULT_THRESHOLDS)
        sweeps[pr] = sweep(sample, feats)
    d = bcf.DEFAULT_THRESHOLDS.to_dict()
    chosen = {"preset": GATE_PRESET,
              "strict": _best(sweeps[GATE_PRESET]["strict"], ("strict_wrap_deg", "strict_h_mm"),
                              d),
              "engaged": _best(sweeps[GATE_PRESET]["engaged"], ("engaged_wrap_deg",), d)}
    passed, misses = verdict(results)
    episodes = sorted({r["file"] for r in rows})
    ep_level = (episode_level(episodes, presets, spec, args.seed, args.last_n)
                if args.last_n > 0 else None)
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": _git(),
        "script": "scripts/lcs/perception_gate.py",
        "config": {"dirs": [_rel(d) for d in args.dirs], "presets": presets, "seed": args.seed,
                   "stride": args.stride, "per_class": args.per_class,
                   "min_frames": args.min_frames, "spec": bcf.DEFAULT_SPEC.to_dict(),
                   "spec_used": spec.to_dict(), "thresholds": d,
                   "noise_specs": {pr: bc.cloud_noise_spec(pr).to_dict() for pr in presets},
                   "seg_spec": bc.BeltSegSpec().to_dict(),
                   "rng": "[seed, crc32(episode path), t, 11]",
                   "gt_strict": {"wrap_deg": GT_STRICT_WRAP_DEG, "h_mm": GT_STRICT_H_MM},
                   "gate": {"preset": GATE_PRESET, **GATE, "detector": "engaged vs GT engaged"}},
        "n_candidates": len(rows), "n_episodes": len({r["file"] for r in rows}),
        "candidates_by_class": dict(Counter(r["cls"] for r in rows)),
        "n_frames": len(sample), "sample_by_class": dict(Counter(r["cls"] for r in sample)),
        "skipped_files": skipped, "h_bias_mm": h_bias, "h_bias_n": len(diffs),
        "results": results, "sweeps": sweeps, "chosen": chosen,
        "episode_level": ep_level, "verdict": "PASS" if passed else "FAIL", "misses": misses,
        "wall_s": time.perf_counter() - t0,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "gate.md").write_text(markdown(report))
    for pr, r in results.items():
        print(f"[G1] {pr}: n {r['n']} |dh| med {_f(r['dh_median_mm'])} p90 "
              f"{_f(r['dh_p90_mm'])} mm, |dwrap| med {_f(r['dwrap_median_deg'], 1)} p90 "
              f"{_f(r['dwrap_p90_deg'], 1)} deg, engaged P/R {_f(r['engaged']['precision'], 3)}"
              f"/{_f(r['engaged']['recall'], 3)}, strict P/R "
              f"{_f(r['strict']['precision'], 3)}/{_f(r['strict']['recall'], 3)}")
    if ep_level:
        for pr, v in ep_level["presets"].items():
            print(f"[G1] episode level (info) {pr}: " + ", ".join(
                f"{r} P/R {_f(v[r]['precision'], 3)}/{_f(v[r]['recall'], 3)}"
                for r in bcf.RULES))
    print(f"[G1] chosen {json.dumps(chosen)}")
    print(f"G1 {report['verdict']}" + "".join(f"\n  miss: {m}" for m in misses))
    print(f"[G1] wrote {out / 'gate.json'}, {out / 'gate.md'} ({report['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
