#!/usr/bin/env python3
"""Evidence (report, not a gate): does the stage-1 latent rule reject e2's false "reached"
switch states and fire on the demos near their own pre_place_2 latch?

Per rollout: whitened distance trace to ``z_goal_stage1`` (numpy twin on ``pcd_belt_seg``),
the distance at e2's switch frame (last ``sim_stage == 0`` + 1), the first frame the
``tol_1`` / streak rule of ``stage_tols.json`` fires, and the EE errors (``sim_ee_*``, GT,
report-only) vs the goal demo's latch poses and vs the nearest demo latch. ``--also``: the same
rollouts under another encoding (e.g. ``belt_seg_offline.py --cloud-noise off`` copies). Demos
(from ``stage_tols.json``): the first fire relative to the latch.

    uv run --frozen python scripts/lcs/latent_reach_evidence.py --deploy <deploy.npz> \\
        --tols data/lcs/realfirst/v1_1/stage_rule/stage_tols.json \\
        --rollouts data/lcs/realfirst/v1/iter0/collect/episode_*.npz \\
        --also data/lcs/realfirst/v1_1/stage_rule/iter0_off/realfirst/v1/iter0/collect/*.npz \\
        --out data/lcs/realfirst/v1_1/stage_rule/evidence
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latent_stage_tols import (
    dists,
    encode_obs,
    latch_frame,
    load_episode,
    sha256,
    stage_goals,
    streak_fire,
)

from task_common.latent_encoder import LatentEncoder, episode_input_keys

DT = 0.075
WITHIN_S = 2.0


def ee_err_mm(ep: dict, t: int, ref: dict) -> list[float]:
    return [float(np.linalg.norm(ep["ee_f"][t, :3] - ref["ef"]) * 1e3),
            float(np.linalg.norm(ep["ee_u"][t, :3] - ref["eu"]) * 1e3)]


def nearest_err_mm(ep: dict, t: int, refs: list[dict]) -> list[float]:
    """Errors vs the demo latch whose summed F + UR error is smallest (grasps differ)."""
    errs = [ee_err_mm(ep, t, r) for r in refs]
    return errs[int(np.argmin([sum(e) for e in errs]))]


def rollout_row(path: Path, enc, lcs, g1, tol: float, streak: int, goal_ref: dict,
                refs: list[dict]) -> dict:
    ep = load_episode(path, episode_input_keys(enc))
    n = ep["n"]
    Z, n_empty = encode_obs(enc, ep, range(n))
    d1 = dists(lcs, Z, g1)
    s0 = np.flatnonzero(ep["stage"] == 0)
    sw = int(s0[-1]) + 1 if len(s0) and s0[-1] + 1 < n else None
    fire = streak_fire(d1, tol, streak)
    row = {"episode": str(path), "n_frames": n, "n_empty": n_empty, "switch": sw,
           "d_t0": float(d1[0]), "d_min_stage0": float(d1[s0].min()) if len(s0) else None,
           "t_d_min_stage0": float(np.argmin(d1[s0]) * DT) if len(s0) else None,
           "d_at_switch": None if sw is None else float(d1[sw]),
           "v1_d_at_switch": (None if sw is None or ep["goal_dist"] is None
                              else float(ep["goal_dist"][sw, 0])),
           "inside_at_switch": None if sw is None else bool(0.0 <= d1[sw] < tol),
           "fired_by_switch": None if sw is None else bool(fire is not None and fire <= sw),
           "fire": fire, "t_fire": None if fire is None else fire * DT,
           "frames_inside": int(np.sum(d1 < tol)),
           "jitter_median": float(np.median(np.abs(np.diff(d1)))),
           "ee_err_switch_goal_mm": None if sw is None else ee_err_mm(ep, sw, goal_ref),
           "ee_err_switch_nearest_mm": None if sw is None else nearest_err_mm(ep, sw, refs),
           "ee_err_fire_goal_mm": None if fire is None else ee_err_mm(ep, fire, goal_ref),
           "ee_err_fire_nearest_mm": None if fire is None else nearest_err_mm(ep, fire, refs),
           "d_trace": np.round(d1, 3).tolist()}
    return row


def demo_row(path: Path, enc, lcs, g1, tol: float, streak: int) -> dict:
    ep = load_episode(path, episode_input_keys(enc))
    i = latch_frame(ep["labels"])
    Z, n_empty = encode_obs(enc, ep, range(ep["n"]))
    d1 = dists(lcs, Z, g1)
    fire = streak_fire(d1, tol, streak)
    off = None if fire is None else fire - i
    return {"episode": str(path), "outcome": ep["outcome"], "latch": i, "n_empty": n_empty,
            "d_latch": float(d1[i]), "d_window_min": float(d1[i: i + 7].min()),
            "jitter_median": float(np.median(np.abs(np.diff(d1)))),
            "fire": fire, "fire_minus_latch": off,
            "fire_minus_latch_s": None if off is None else off * DT,
            "within_2s_after_latch": off is not None and off * DT <= WITHIN_S,
            "fires_before_latch": off is not None and off < 0,
            "label_at_fire": None if fire is None else ep["labels"][fire],
            "d_trace_to_latch_plus_30": np.round(d1[: i + 31], 3).tolist()}


PROXY_BINS_MM = (0.0, 5.0, 10.0, 20.0, 35.0, 60.0, np.inf)


def approach_proxy(paths: list[Path], enc, lcs, g1, tol: float, k: int) -> dict:
    """Clean demo frames binned by max(F, UR) EE distance from the demo's own latch pose (GT):
    the fraction inside tol_1 and where the streak rule would hold (clean false-reached proxy)."""
    err, d, held, pre = [], [], [], []
    for path in paths:
        ep = load_episode(path, episode_input_keys(enc))
        i = latch_frame(ep["labels"])
        Z, _ = encode_obs(enc, ep, range(ep["n"]))
        di = dists(lcs, Z, g1)
        e = np.maximum(np.linalg.norm(ep["ee_f"][:, :3] - ep["ee_f"][i, :3], axis=1),
                       np.linalg.norm(ep["ee_u"][:, :3] - ep["ee_u"][i, :3], axis=1)) * 1e3
        inside = (di >= 0.0) & (di < tol)
        run = np.zeros(len(di), dtype=int)
        for t in range(len(di)):
            run[t] = (run[t - 1] + 1 if t else 1) if inside[t] else 0
        err += e.tolist()
        d += di.tolist()
        held += (run >= k).tolist()
        pre += (np.arange(len(di)) < i).tolist()
    err, d, held, pre = (np.asarray(x) for x in (err, d, held, pre))
    bins = []
    for lo, hi in itertools.pairwise(PROXY_BINS_MM):
        m = (err >= lo) & (err < hi)
        bins.append({"lo_mm": lo, "hi_mm": None if np.isinf(hi) else hi, "n": int(m.sum()),
                     "frac_inside": float(np.mean(d[m] < tol)) if m.any() else None,
                     "frac_streak": float(np.mean(held[m])) if m.any() else None,
                     "d_median": float(np.median(d[m])) if m.any() else None,
                     "n_pre": int((m & pre).sum()), "n_streak_pre": int((m & pre & held).sum()),
                     "n_streak_post": int((m & ~pre & held).sum())})
    ins = held & (err >= 20.0)
    return {"bins": bins, "n_frames": len(err),
            "max_err_mm_where_streak_holds": float(err[held].max()) if held.any() else None,
            "n_streak_frames_err_ge_20mm": int(ins.sum()),
            "n_streak_frames_err_ge_20mm_pre_latch": int((ins & pre).sum())}


def headline(rows: list[dict]) -> dict:
    sw = [r for r in rows if r["switch"] is not None]
    return {"n": len(rows), "n_with_switch": len(sw),
            "flagged_not_reached_at_switch": sum(not r["fired_by_switch"] for r in sw),
            "outside_tol_at_switch_frame": sum(not r["inside_at_switch"] for r in sw),
            "ever_fire": sum(r["fire"] is not None for r in rows)}


def fmt(v, spec=".2f") -> str:
    if v is None:
        return "-"
    if isinstance(v, list):
        return "/".join(format(x, ".0f") for x in v)
    return format(v, spec)


def demo_table(d: dict) -> list[str]:
    head = (f"**{d['within_2s']}/{d['n']} fire within {WITHIN_S:g} s after the latch** "
            f"(<= {WITHIN_S:g} s, early fires included); {d['before_latch']}/{d['n']} fire "
            f"before the latch; {d['never']}/{d['n']} never fire.")
    out = [head, "",
           ("| demo | outcome | latch | d latch | min latch+6 | median abs(d diff) | first fire "
            "| fire - latch (s) | phase at fire |"),
           "|---|---|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        name = "/".join(Path(r["episode"]).parts[-4:])
        out.append(f"| {name} | {r['outcome']} | {r['latch']} | {fmt(r['d_latch'])} | "
                   f"{fmt(r['d_window_min'])} | {fmt(r['jitter_median'])} | "
                   f"{fmt(r['fire'], 'd')} | {fmt(r['fire_minus_latch_s'], '+.3f')} | "
                   f"{r['label_at_fire'] or '-'} |")
    return out


def markdown(doc: dict) -> str:
    tol, k = doc["tol_1"], doc["streak"]
    intro = (f"deploy `{doc['deploy']}` (sha {doc['deploy_sha'][:12]}); tol_1 {tol:.3f}, "
             f"streak {k} (stage_tols.json); v1 (e2 iter 0) tol 4.67, single latent. EE errors "
             f"are GT (`sim_ee_*`), report-only: vs the goal demo latch (`{doc['goal_ref']}`) "
             f"and vs the nearest of the {doc['n_demos']} demo latches, F/UR mm. "
             "median abs(d diff) = frame-to-frame jitter of the distance.")
    out = ["# Stage-1 latent rule evidence (report, not a gate)", "", intro, ""]
    for enc in doc["encodings"]:
        h = enc["headline"]
        head = (f"**{h['flagged_not_reached_at_switch']}/{h['n_with_switch']} flagged \"not "
                f"reached\" at their switch frame** (the rule has not fired by then; "
                f"{h['outside_tol_at_switch_frame']}/{h['n_with_switch']} outside tol_1 on the "
                f"switch frame itself); **{h['ever_fire']}/{h['n']} ever fire** over the whole "
                "rollout.")
        out += [f"## e2 iter-0 rollouts, {enc['name']}", "", enc["note"], "", head, "",
                ("| rollout | switch t | d t0 | d min st0 (t) | d switch (v1 d) | median "
                 "abs(d diff) | fired by switch | first fire t | EE err switch goal / nearest "
                 "| EE err fire goal / nearest |"),
                "|---|---|---|---|---|---|---|---|---|---|"]
        for r in enc["rollouts"]:
            sw_t = None if r["switch"] is None else r["switch"] * DT
            out.append(
                f"| {Path(r['episode']).stem} | {fmt(sw_t)} | {fmt(r['d_t0'])} | "
                f"{fmt(r['d_min_stage0'])} ({fmt(r['t_d_min_stage0'], '.1f')}) | "
                f"{fmt(r['d_at_switch'])} ({fmt(r['v1_d_at_switch'])}) | "
                f"{fmt(r['jitter_median'])} | "
                f"{'yes' if r['fired_by_switch'] else 'no'} | {fmt(r['t_fire'])} | "
                f"{fmt(r['ee_err_switch_goal_mm'])} / {fmt(r['ee_err_switch_nearest_mm'])} | "
                f"{fmt(r['ee_err_fire_goal_mm'])} / {fmt(r['ee_err_fire_nearest_mm'])} |")
        if enc.get("identical_to") is not None:
            out.append(f"\nEvery `pcd_belt_seg` frame is bit-identical to `{enc['identical_to']}`"
                       " (the stored `pcd` is the noised cloud; preset `off` adds nothing).")
        out.append("")
    out += ["## Clean demos: first fire vs the pre_place_2 latch", "", *demo_table(doc["demos"]),
            ""]
    px = doc.get("approach_proxy")
    if px:
        out += ["## Clean proxy: demo frames by EE distance from their own latch", "",
                ("All frames of the clean demos, binned by max(F, UR) EE distance (GT) from the "
                 "demo's own latch pose; inside = d < tol_1 on that frame; streak = the rule "
                 f"holds there. Streak frames with the arms >= 20 mm off: "
                 f"{px['n_streak_frames_err_ge_20mm']}; largest EE distance where the streak "
                 f"holds: {fmt(px['max_err_mm_where_streak_holds'], '.1f')} mm; of those "
                 f">= 20 mm, before the latch: {px['n_streak_frames_err_ge_20mm_pre_latch']}."),
                "", ("| EE distance mm | frames (pre-latch) | inside tol_1 | streak holds | "
                     "streak frames pre / post latch | median d |"), "|---|---|---|---|---|---|"]
        for bn in px["bins"]:
            hi = "inf" if bn["hi_mm"] is None else f"{bn['hi_mm']:g}"
            out.append(f"| {bn['lo_mm']:g}-{hi} | {bn['n']} ({bn['n_pre']}) | "
                       f"{fmt(bn['frac_inside'], '.0%')} | {fmt(bn['frac_streak'], '.0%')} | "
                       f"{bn['n_streak_pre']} / {bn['n_streak_post']} | {fmt(bn['d_median'])} |")
        out.append("")
    if doc.get("noised_demos"):
        out += ["## Control: noised demos (real_like clouds, same recipe) under this encoder", "",
                doc["noised_note"], "", *demo_table(doc["noised_demos"]), ""]
    return "\n".join(out)


def same_seg(a: Path, b: Path) -> bool:
    with np.load(a, allow_pickle=True) as x, np.load(b, allow_pickle=True) as y:
        sa, sb = x["pcd_belt_seg"], y["pcd_belt_seg"]
        return len(sa) == len(sb) and all(np.array_equal(p, q) for p, q in zip(sa, sb))


def demo_summary(paths: list[Path], enc, lcs, g1, tol: float, k: int) -> dict:
    rows = [demo_row(pth, enc, lcs, g1, tol, k) for pth in paths]
    return {"n": len(rows), "rows": rows,
            "within_2s": sum(r["within_2s_after_latch"] for r in rows),
            "before_latch": sum(r["fires_before_latch"] for r in rows),
            "never": sum(r["fire"] is None for r in rows)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deploy", type=Path, required=True)
    p.add_argument("--tols", type=Path, required=True, help="latent_stage_tols.py output")
    p.add_argument("--rollouts", type=Path, nargs="+", required=True)
    p.add_argument("--also", type=Path, nargs="*", default=[],
                   help="the same rollouts under another encoding (sorted pairing by name)")
    p.add_argument("--noised-demos", type=Path, nargs="*", default=[],
                   help="control: demos recorded with noised clouds (e.g. the v1 seed)")
    p.add_argument("--rollouts-note", default="As recorded: `pcd_belt_seg` of the real_like "
                   "noised cloud, out of distribution for a clean-cloud encoder.")
    p.add_argument("--also-note", default="Re-segmented with preset `off` "
                   "(`belt_seg_offline.py`); their `pcd` is already the real_like noised cloud, "
                   "so this is noised, not re-noised.")
    p.add_argument("--noised-note", default="v1 seed demos (real_like `pcd_belt_seg`), the "
                   "clean seed's recipe and seed: how much of the e2 distance is cloud noise.")
    p.add_argument("--out", type=Path, required=True, help="prefix: <out>.json, <out>.md")
    a = p.parse_args()
    tols = json.loads(a.tols.read_text())
    if tols["deploy_sha"] != sha256(a.deploy):
        raise SystemExit(f"{a.tols} was derived for another deploy")
    tol, k = float(tols["tol"][0]), int(tols["streak"])
    lcs, g1, _ = stage_goals(a.deploy)
    enc = LatentEncoder.load(a.deploy)
    with np.load(a.deploy, allow_pickle=True) as d:
        src = str(d["goal_source"]).split(":", 1)[-1].split("#")[0]
        gframe = int(d["goal_frames"][0])
    demos = [Path(r["episode"]) for r in tols["per_demo"]]
    refs = []
    for path in demos:
        ep = load_episode(path)
        i = latch_frame(ep["labels"])
        refs.append({"ef": ep["ee_f"][i, :3], "eu": ep["ee_u"][i, :3]})
    gep = load_episode(Path(src))
    goal_ref = {"ef": gep["ee_f"][gframe, :3], "eu": gep["ee_u"][gframe, :3]}
    rolls, also = sorted(a.rollouts), sorted(a.also, key=lambda q: q.name)
    if also and [q.name for q in also] != [q.name for q in sorted(rolls, key=lambda q: q.name)]:
        raise SystemExit("--also must hold the same episode names as --rollouts")
    encodings = []
    for name, paths, note in (("as recorded", rolls, a.rollouts_note),
                              ("re-segmented off", also, a.also_note)):
        if not paths:
            continue
        rows = [rollout_row(pth, enc, lcs, g1, tol, k, goal_ref, refs) for pth in paths]
        item = {"name": name, "note": note, "rollouts": rows, "headline": headline(rows)}
        if paths is also:
            same = all(same_seg(x, y) for x, y in zip(sorted(rolls, key=lambda q: q.name), also))
            item["identical_to"] = "as recorded" if same else None
        encodings.append(item)
    doc = {"deploy": str(a.deploy.resolve()), "deploy_sha": sha256(a.deploy),
           "tols": str(a.tols), "tol_1": tol, "streak": k, "dt": DT,
           "goal_ref": f"{Path(src).name}#{gframe}", "n_demos": len(demos),
           "encodings": encodings, "demos": demo_summary(demos, enc, lcs, g1, tol, k),
           "approach_proxy": approach_proxy(demos, enc, lcs, g1, tol, k)}
    if a.noised_demos:
        doc["noised_note"] = a.noised_note
        doc["noised_demos"] = demo_summary(sorted(a.noised_demos), enc, lcs, g1, tol, k)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(doc, indent=1) + "\n")
    text = markdown(doc)
    a.out.with_suffix(".md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
