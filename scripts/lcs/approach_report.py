#!/usr/bin/env python3
"""QC + label report of approach runs (``collect_lcs_dataset.py`` with the opt-in modes).

Per run (and pooled): labels at ``place_3`` and at the end (per start state), the ``place_3 ->
end`` transition matrix, engaged fractions, tail families / stop reasons / durations / contact-
frame fraction, wrap / h histograms through the tails, stretch, clearances, grasp, latch errors,
causality (realised vs u) on approach rows and tail rows with their gates, achieved approach
parameters. ``--write-qc`` writes ``<run>/qc.json``; ``--probe A|B|C`` prints the probe criteria
per setting.

Run:
    uv run --frozen python scripts/lcs/approach_report.py RUN [RUN ...] [--write-qc] [--pool]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from task_common import lcs_dataset as lcs

LABELS = ("engaged", "over", "under", "slanted", "outside", "other")
GATES = {"approach": (0.6, 1.2), "tail": (0.8, 1.1)}
UR_XY_STD_MIN_MM = 0.8
LATCH_MAX_MM, UR_REACH_MAX_MM, CLAMP_MAX_MM, STRETCH_MAX_PCT = 5.0, 3.0, 8.0, 2.0
WRAP_BINS = (0, 1, 30, 60, 120, 180, 360)
H_BINS = (-50, -5, -2, 2, 5, 10, 20, 100)


def load_run(run: Path) -> tuple[dict, list[dict]]:
    index = json.loads((run / "index.json").read_text())
    return index, [dict(r, _run=str(run)) for r in index["episodes"]]


def episode_arrays(row: dict) -> dict:
    d = np.load(Path(row["_run"]) / row["file"], allow_pickle=True)
    meta = json.loads(str(d["sim_meta"]))
    labels = meta["phase_labels"]
    ph = np.array([labels[k] for k in d["sim_phase"]])
    p3 = int(d["sim_place3_frame"]) if "sim_place3_frame" in d else len(ph) - 1
    out = {"phase": ph, "p3": p3, "u": d["actions"].astype(np.float64),
           "r": d["sim_realised_delta"].astype(np.float64),
           "wrap": d["sim_wrap_deg"], "h": d["sim_h_median_mm"],
           "in_contact": d.get("sim_in_contact", None),
           "grasp": d["sim_grasp_ok"], "ee_ur": d["sim_ee_ur"], "cmd_ur": d["sim_cmd_ur_t"],
           "robotiq": d["sim_robotiq_byte"], "rod": d.get("sim_rod_stretch_pct", None)}
    return out


def pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:5.1f}%" if total else "   - "


def label_table(rows: list[dict], key: str, group: str | None = None) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(group) if group else "all"].append(r)
    head = f"{'group':<14}{'n':>4}" + "".join(f"{lb:>9}" for lb in LABELS[:5])
    out = [head]
    for g in sorted(groups, key=str):
        c = Counter(r[key] for r in groups[g])
        n = len(groups[g])
        out.append(f"{g!s:<14}{n:>4}" + "".join(f"{pct(c[lb], n):>9}" for lb in LABELS[:5]))
    return out


def transition(rows: list[dict]) -> list[str]:
    c = Counter((r["outcome_place3"], r["outcome"]) for r in rows)
    out = [f"{'p3 / end':<10}" + "".join(f"{lb:>9}" for lb in LABELS[:5])]
    for a in LABELS[:5]:
        if any(c[(a, b)] for b in LABELS):
            out.append(f"{a:<10}" + "".join(f"{c[(a, b)]:>9}" for b in LABELS[:5]))
    return out


def causality(arrs: list[dict], which: str) -> dict:
    u, r = [], []
    for a in arrs:
        ph = a["phase"]
        if which == "tail":
            m = np.array([p.startswith("tail:") or p == "hold:tail" for p in ph])
        else:
            m = np.array([not (p.startswith("tail:") or p in ("hold:tail", "prehold"))
                          for p in ph])
        u.append(a["u"][m])
        r.append(a["r"][m])
    if not u or not sum(len(x) for x in u):
        return {}
    rows = lcs.causality_slopes(np.vstack(u), np.vstack(r))
    lo, hi = GATES[which]
    return {"rows": rows[0]["rows"], "gate": [lo, hi],
            "slopes": {x["dim"]: round(x["slope"], 4) for x in rows},
            "u_std": {x["dim"]: round(x["u_std"] * 1e3, 4) for x in rows},
            "pass": bool(all(lo <= x["slope"] <= hi for x in rows))}


def summarize(rows_all: list[dict], name: str, approach_index: dict | None = None) -> dict:
    ok = [r for r in rows_all if r["status"] == "ok"]
    arrs = [episode_arrays(r) for r in ok]
    n = len(ok)
    qc = {"run": name, "episodes_ok": n, "episodes_skipped": len(rows_all) - n,
          "skipped_reasons": dict(Counter(r.get("reason_key", r.get("reason"))
                                          for r in rows_all if r["status"] != "ok"))}
    if not n:
        return qc
    has_p3 = all("outcome_place3" in r for r in ok)
    lab_end = Counter(r["outcome"] for r in ok)
    qc["labels_end"] = {lb: lab_end[lb] for lb in LABELS}
    if has_p3:
        lab_p3 = Counter(r["outcome_place3"] for r in ok)
        qc["labels_place3"] = {lb: lab_p3[lb] for lb in LABELS}
        qc["engaged_frac_place3"] = lab_p3["engaged"] / n
        qc["transition"] = {f"{a}->{b}": c for (a, b), c in Counter(
            (r["outcome_place3"], r["outcome"]) for r in ok).items()}
    qc["engaged_frac_end"] = lab_end["engaged"] / n
    qc["place3_modes"] = dict(Counter((r.get("place3_sample") or {}).get("mode") for r in ok))
    qc["top_up_rows"] = sum(r.get("tag") == "top_up" for r in ok)
    qc["min_board_clearance_mm"] = float(min(r["min_board_clearance_mm"] for r in ok))
    qc["board_contact_files"] = [r["file"] for r in ok if r["board_contact"]]
    qc["min_franka_tip_clearance_mm"] = float(min(r["min_franka_tip_clearance_mm"] for r in ok))
    qc["clamp_lift_mm_max"] = float(max(r["clamp_lift_mm"] for r in ok))
    qc["franka_floor_lift_mm_max"] = float(max((r.get("clamp") or {}).get("franka_lift_mm", 0.0)
                                               for r in ok))
    latch = [r["final_target_err_mm"].get("place_3") for r in ok]
    qc["latch_err_place3_mm"] = {"max": float(max(v for v in latch if v is not None)),
                                 "n_over_5": int(sum(v > LATCH_MAX_MM for v in latch if v))}
    grasp_all = np.array([a["grasp"].all(axis=0) for a in arrs])
    qc["grasp_ok_all_frames"] = {"franka": int(grasp_all[:, 0].sum()),
                                 "ur": int(grasp_all[:, 1].sum()), "of": n}
    qc["ur_release_stripped"] = int(sum(bool(r.get("hold_ur_gripper")) for r in ok))
    reach = [float(np.linalg.norm(a["ee_ur"][a["p3"], :3] - a["cmd_ur"][a["p3"], :3]) * 1e3)
             for a in arrs]
    qc["ur_reach_place3_mm"] = {"max": max(reach), "median": float(np.median(reach))}
    tr = np.array([r["tracking_rms_mm"] for r in ok])
    qc["tracking_rms_mm_max"] = tr.max(axis=0).tolist()
    tails = [r["tail"] for r in ok if r.get("tail")]
    if tails:
        dt = 0.075
        dur = np.array([t["frames"] * dt for t in tails])
        cf = np.array([t["contact_frame_frac"] for t in tails
                       if t["contact_frame_frac"] is not None])
        qc["tails"] = {
            "n": len(tails), "families": dict(Counter(t["family"] for t in tails)),
            "gentle": sum(t["gentle"] for t in tails),
            "stopped": dict(Counter((t["stopped"] or "completed").split(" at k")[0]
                                    .split(" (")[0] for t in tails)),
            "motion_ge_4s_frac": float(np.mean([min(t["T_s"], t["frames"] * dt) >= 4.0 - 1e-6
                                                and t["stopped"] is None for t in tails])),
            "tail_s": {"min": float(dur.min()), "median": float(np.median(dur))},
            "contact_frame_frac": {"pooled": float(np.mean(np.concatenate(
                [a["in_contact"][a["p3"] + 1:] for a in arrs if a["in_contact"] is not None]))),
                "episode_min": float(cf.min()), "episode_median": float(np.median(cf))},
            "stretch_gain_max_pct": float(max(t["stretch_gain_max_pct"] for t in tails)),
            "ur_guard_scaled_frames": int(sum(t["ur_guard_scaled_frames"] for t in tails)),
            "tracking_rms_mm_max": np.array([t["tracking_rms_mm"] for t in tails]).max(0).tolist(),
            "retries": sum("tail_retry" in r for r in ok),
        }
        wr = np.concatenate([a["wrap"][a["p3"] + 1:] for a in arrs])
        hh = np.concatenate([a["h"][a["p3"] + 1:] for a in arrs])
        hh = hh[np.isfinite(hh)]
        qc["tails"]["wrap_hist"] = dict(zip([f"{a}-{b}" for a, b in pairwise(WRAP_BINS)],
                                            np.histogram(wr, WRAP_BINS)[0].tolist()))
        qc["tails"]["h_median_hist"] = dict(zip([f"{a}..{b}" for a, b in pairwise(H_BINS)],
                                                np.histogram(hh, H_BINS)[0].tolist()))
    rods = [a["rod"] for a in arrs if a["rod"] is not None]
    if rods:
        qc["rod_stretch_pct_max"] = float(max(np.nanmax(x) for x in rods))
    qc["causality"] = {w: causality(arrs, w) for w in ("approach", "tail")}
    ct = qc["causality"].get("tail") or {}
    if ct:
        qc["causality"]["tail"]["ur_xy_u_std_ge_0p8"] = bool(
            min(ct["u_std"]["u_x"], ct["u_std"]["u_y"]) >= UR_XY_STD_MIN_MM)
    if approach_index:
        ach = [v for k, v in approach_index.items()
               if k in {r.get("start_variant") for r in ok}]
        qc["approach_achieved"] = {
            "yaw_err_deg_max": float(max(abs(v["achieved"]["yaw_deg"] - v["approach"]["yaw_deg"])
                                         for v in ach)) if ach else None,
            "normal_err_mm_max": float(max(abs(v["achieved"][arm]["normal_err_mm"])
                                           for v in ach for arm in ("franka", "ur")))
            if ach else None}
    return qc


def print_run(name: str, rows_all: list[dict], qc: dict) -> None:
    ok = [r for r in rows_all if r["status"] == "ok"]
    print(f"== {name}: ok {qc['episodes_ok']} skipped {qc['episodes_skipped']} "
          f"{qc['skipped_reasons'] or ''}")
    if not ok:
        return
    if "labels_place3" in qc:
        print("labels at place_3 (per start state):")
        print("\n".join(label_table(ok, "outcome_place3", "start_variant")))
        print("\n".join(label_table(ok, "outcome_place3")[1:]))
        print("labels at the end:")
        print("\n".join(label_table(ok, "outcome", "start_variant")))
        print("\n".join(label_table(ok, "outcome")[1:]))
        print("transition place_3 -> end:")
        print("\n".join(transition(ok)))
        print(f"engaged: place_3 {qc['engaged_frac_place3']:.3f}, end {qc['engaged_frac_end']:.3f}")
    keys = ("place3_modes", "min_board_clearance_mm", "board_contact_files",
            "min_franka_tip_clearance_mm", "clamp_lift_mm_max", "franka_floor_lift_mm_max",
            "latch_err_place3_mm", "grasp_ok_all_frames", "ur_release_stripped",
            "ur_reach_place3_mm", "tracking_rms_mm_max", "rod_stretch_pct_max",
            "approach_achieved", "top_up_rows")
    for k in keys:
        if k in qc:
            print(f"{k}: {json.dumps(qc[k], default=float)}")
    if "tails" in qc:
        print("tails: " + json.dumps(qc["tails"], default=float))
    for w, c in qc["causality"].items():
        if c:
            print(f"causality {w} ({c['rows']} rows, gate {c['gate']}, pass {c['pass']}): "
                  f"slopes {c['slopes']}")
            print(f"   u std (mm|mrad) {c['u_std']}")


def probe_table(kind: str, rows_all: list[dict]) -> None:
    ok = {r["file"]: r for r in rows_all if r["status"] == "ok"}
    groups = defaultdict(list)
    for k, r in enumerate(rows_all):
        if kind == "A":
            key = r.get("start_variant")
        elif kind == "B":
            key = json.dumps({arm: r["place3_sample"][arm] for arm in ("franka", "ur")})
        else:
            key = (r.get("tail") or {}).get("family") or f"skipped#{k}"
        groups[key].append(r)
    for key, rs in groups.items():
        cells = []
        for r in rs:
            if r["status"] != "ok":
                cells.append(f"SKIP({r.get('reason_key', r.get('reason'))})")
                continue
            a = episode_arrays(ok[r["file"]])
            grasp = bool(a["grasp"].all())
            latch = r["final_target_err_mm"].get("place_3", np.nan)
            reach = float(np.linalg.norm(a["ee_ur"][a["p3"], :3] - a["cmd_ur"][a["p3"], :3]) * 1e3)
            wrap3 = float(a["wrap"][a["p3"]])
            c = (f"{r['outcome_place3']}->{r['outcome']} grasp {int(grasp)} contact "
                 f"{int(r['board_contact'])} clr {r['min_board_clearance_mm']:.1f} latch "
                 f"{latch:.1f} reach {reach:.1f} lift {r['clamp_lift_mm']:.1f} wrap3 {wrap3:.0f}")
            if r.get("tail"):
                t = r["tail"]
                tw = a["wrap"][a["p3"] + 1:]
                th = a["h"][a["p3"] + 1:]
                c += (f" stretch+ {t['stretch_gain_max_pct']:.2f} stop {t['stopped']} "
                      f"wrap {tw.min():.0f}..{tw.max():.0f} h {np.nanmin(th):.1f}.."
                      f"{np.nanmax(th):.1f} contact {t['contact_frame_frac']:.2f}")
            cells.append(c)
        print(f"{key}: " + " | ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--write-qc", action="store_true")
    ap.add_argument("--pool", action="store_true", help="also the pooled summary")
    ap.add_argument("--pool-name", default="pooled")
    ap.add_argument("--probe", choices=("A", "B", "C"), default=None)
    ap.add_argument("--json", type=Path, default=None, help="save the summaries")
    args = ap.parse_args()
    pooled, out = [], {}
    for run in args.runs:
        index, rows = load_run(run)
        ss = index.get("start_state") or {}
        a_index = None
        if ss.get("kind") == "variant_set":
            p = Path(ss["path"]) / "index.json"
            if p.is_file():
                a_index = {v["id"]: v for v in json.loads(p.read_text())["variants"]
                           if v.get("approach") and v.get("achieved")}
        qc = summarize(rows, str(run), a_index)
        qc["wall_s"] = (index.get("summary") or {}).get("wall_s")
        qc["episodes_per_min"] = (index.get("summary") or {}).get("episodes_per_min")
        qc["osc_log_errors"] = (index.get("summary") or {}).get("osc_log_errors")
        print_run(str(run), rows, qc)
        if args.probe:
            probe_table(args.probe, rows)
        if args.write_qc:
            (run / "qc.json").write_text(json.dumps(qc, indent=1, default=float) + "\n")
        out[str(run)] = qc
        pooled += [dict(r, start_variant=f"{run.name}/{r.get('start_variant')}") for r in rows]
    if args.pool and len(args.runs) > 1:
        qc = summarize(pooled, args.pool_name)
        print_run(args.pool_name, pooled, qc)
        out[args.pool_name] = qc
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
