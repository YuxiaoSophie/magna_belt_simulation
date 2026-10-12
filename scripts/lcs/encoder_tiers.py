#!/usr/bin/env python3
"""Real-budget training tiers and held-out eval sets for the encoder study (V12-42). CPU, no sim.

Tiers (buffer.json manifests, every row weight 1.0, no frame weights / anchors):
  T0 = the 3 MILES reference demos (``miles/demos3``);
  T1 = T0 + V12-29b MILES (every valid / invalid cycle and walk; = V12-35's seed buffer);
  T2 = T1 + every on-policy rollout of V12-35 and V12-36 (iter*/collect and iter*/eval/*).
Eval sets (episode lists, never in a tier; asserted by resolved path):
  E_off = V12-25's off-demo rollouts (``seed_compare.GROUPS``: V12-17, V12-18, V12-20, V12-21);
  E_hs  = the no-prior hard-suite rollouts of V12-37 (``noprior/``), V12-39 (``value_cost/sim/``)
          and V12-40 (``c3plus/ab/``); any E_hs episode found in a tier is dropped (reported).
Writes ``<out>/T{0,1,2}/buffer.json``, ``<out>/eval_sets.json``, ``<out>/tiers.{md,json}``.

    uv run --frozen python scripts/lcs/encoder_tiers.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_compare import GROUPS, group_episodes, jload

RF = REPO / "data/lcs/realfirst"
CMP = RF / "cmp"
DEMOS3 = RF / "miles/demos3"
MILES = RF / "miles/v12_29b"
LOOPS = ("v12_35", "v12_36")
SEED_MERGED = CMP / "v12_35/seed_buffer_merged.json"
HS_GROUPS = {"V12-37": "data/lcs/realfirst/cmp/noprior/*/index.json",
             "V12-39": "data/lcs/realfirst/cmp/value_cost/sim/*/index.json",
             "V12-40": "data/lcs/realfirst/cmp/c3plus/ab/*/index.json"}
DT = 0.075
DROP_KEYS = ("frame_weights", "off_track")


def resolved(p) -> str:
    return str(Path(p).resolve())


def plain(row: dict) -> dict:
    """Equal weight: weight 1.0, per-frame weights and off-track marks dropped."""
    return {k: v for k, v in row.items() if k not in DROP_KEYS} | {"weight": 1.0}


def index_rows(index: Path, it: int, kind: str) -> list[dict]:
    idx = jload(index)
    if "summary" not in idx:
        return []  # an aborted / restarted run dir (e.g. fe_set1.prev0)
    rows = []
    for e in idx["episodes"]:
        if not e.get("file"):
            continue
        ep = resolved(index.parent / e["file"])
        rows.append({"id": str(Path(ep).relative_to(CMP)),
                     "iter": it, "kind": kind, "status": e.get("status", "ok"), "weight": 1.0,
                     "episode": ep, "label_human": e.get("outcome") == "engaged",
                     "label_human_detail": {"source": "GT outcome in the run index"},
                     "outcome_gt": e.get("outcome"), "start": e.get("start_state", {}).get("id"),
                     "reset_before": "human", "n_frames": e.get("frames"),
                     "duration_s": e.get("duration_s")})
    return rows


def loop_rows(loop: str) -> tuple[list[dict], list[dict], dict]:
    """(collect rows from the loop buffer, eval rows from the eval indexes, unbudgeted evals)."""
    root = CMP / loop
    buf = [plain(r) for r in jload(root / "buffer.json")["rollouts"] if r["kind"] == "mpc"]
    have = {resolved(r["episode"]) for r in buf}
    for ix in sorted(root.glob("iter*/collect/index.json")):
        for r in index_rows(ix, int(ix.parent.parent.name[4:]), "mpc"):
            if r["episode"] not in have:
                raise SystemExit(f"{loop}: collect episode {r['episode']} not in buffer.json")
    budget = jload(root / "budget.json")
    budgeted = {p["iter"] for p in budget["per_iter"] if "eval" in p}
    ev, extra = [], {"episodes": 0, "robot_s": 0.0, "resets_human": 0, "iters": []}
    for ix in sorted(root.glob("iter*/eval/*/index.json")):
        it = int(ix.parent.parent.parent.name[4:])
        rows = index_rows(ix, it, "eval")
        ev += rows
        if it not in budgeted and rows:  # e.g. V12-35 iteration 5: the loop stopped before budget
            extra["episodes"] += len(rows)
            extra["robot_s"] += sum(r["duration_s"] or 0.0 for r in rows)
            extra["resets_human"] += len(rows)
            extra["iters"] = sorted(set(extra["iters"]) | {it})
    return buf, ev, extra


def episode_stats(rows: list[dict]) -> dict:
    frames = nonempty = 0
    for r in rows:
        with np.load(r["episode"], allow_pickle=True) as d:
            if "pcd_belt_seg" not in d.files:
                raise SystemExit(f"{r['episode']}: no pcd_belt_seg")
            seg = d["pcd_belt_seg"]
        frames += len(seg)
        nonempty += sum(len(s) > 0 for s in seg)
    return {"episodes": len(rows), "frames": frames, "frames_nonempty": nonempty,
            "episode_s": round(frames * DT, 1),
            "kinds": dict(collections.Counter(r["kind"] for r in rows))}


def write_buffer(path: Path, rows: list[dict], name: str, source: str) -> None:
    eps = [r["episode"] for r in rows]
    if len(set(eps)) != len(eps):
        raise SystemExit(f"{name}: duplicate episodes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "iter": -1, "n": len(rows), "tier": name,
                                "source": source, "rollouts": rows}, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/lcs/realfirst/cmp/encoder_study/tiers")
    a = ap.parse_args()
    out = (REPO / a.out).resolve()

    t0 = [plain(r) for r in jload(DEMOS3 / "buffer.json")["rollouts"]]
    miles = [plain(r) for r in jload(MILES / "buffer.json")["rollouts"]]
    t1 = t0 + miles
    seed = {resolved(r["episode"]) for r in jload(SEED_MERGED)["rollouts"]}
    if seed != {resolved(r["episode"]) for r in t1}:
        raise SystemExit("T1 != V12-35's seed buffer (demos3 + v12_29b)")
    on, budgets = [], {}
    for loop in LOOPS:
        buf, ev, extra = loop_rows(loop)
        on += buf + ev
        budgets[loop] = extra
    seen, t2 = set(), list(t1)
    for r in on:
        r["episode"] = resolved(r["episode"])
        if r["episode"] not in seen:
            seen.add(r["episode"])
            t2.append(r)
    for r in t2:
        r["episode"] = resolved(r["episode"])
    n_dup = len(on) - len(seen)

    b_d, b_m = jload(DEMOS3 / "budget.json"), jload(MILES / "budget.json")
    seed_s, seed_h = b_d["robot_s"] + b_m["robot_s"], b_d["resets_human"] + b_m["resets_human"]
    loop_s = loop_h = loop_sc = 0.0
    for loop in LOOPS:
        b = jload(CMP / loop / "budget.json")
        if abs(b["robot_s_detail"]["seed"] - seed_s) > 1e-6:
            raise SystemExit(f"{loop}: budget seed robot_s != demos3 + v12_29b")
        loop_s += b["robot_s"] - seed_s + budgets[loop]["robot_s"]
        loop_h += b["resets_human"] - seed_h + budgets[loop]["resets_human"]
        loop_sc += b["resets_scripted"]
    budget = {"T0": {"robot_s": b_d["robot_s"], "resets_human": b_d["resets_human"],
                     "resets_scripted": 0},
              "T1": {"robot_s": seed_s, "resets_human": seed_h, "resets_scripted": 0},
              "T2": {"robot_s": seed_s + loop_s, "resets_human": seed_h + loop_h,
                     "resets_scripted": loop_sc}}

    tiers = {"T0": t0, "T1": t1, "T2": t2}
    src = {"T0": "miles/demos3/buffer.json",
           "T1": "miles/demos3 + miles/v12_29b (= cmp/v12_35/seed_buffer_merged.json)",
           "T2": "T1 + cmp/v12_35, cmp/v12_36 iter*/collect + iter*/eval/* (deduped)"}
    for k, rows in tiers.items():
        write_buffer(out / k / "buffer.json", rows, k, src[k])
    trained = {r["episode"] for rows in tiers.values() for r in rows}

    e_off = [r | {"group": g} for g, rows in group_episodes([]).items() for r in rows]
    if bad := [r["episode"] for r in e_off if r["episode"] in trained]:
        raise SystemExit(f"E_off overlaps a tier: {bad[:5]}")
    hs_all = [r | {"group": g} for g, rows in
              group_episodes([f"{g}={p}" for g, p in HS_GROUPS.items()]).items() for r in rows]
    e_hs = [r for r in hs_all if r["episode"] not in trained]
    hs_dropped = [r["episode"] for r in hs_all if r["episode"] in trained]
    off_eps, hs_eps = {r["episode"] for r in e_off}, {r["episode"] for r in e_hs}
    if len(off_eps) != len(e_off) or len(hs_eps) != len(e_hs) or off_eps & hs_eps:
        raise SystemExit("E_off / E_hs have duplicates or overlap")
    for name, rows in (("E_off", e_off), ("E_hs", e_hs)):
        missing = [r["episode"] for r in rows if not Path(r["episode"]).is_file()]
        if missing:
            raise SystemExit(f"{name}: missing episodes {missing[:3]}")
    sets = {"E_off": e_off, "E_hs": e_hs}
    (out / "eval_sets.json").write_text(json.dumps(
        {"groups": {"E_off": GROUPS, "E_hs": HS_GROUPS}, "dropped_from_E_hs": hs_dropped,
         "disjoint_from": list(tiers), "sets": sets}, indent=1) + "\n")

    stats = {k: episode_stats(rows) | budget[k] for k, rows in tiers.items()}
    estats = {k: episode_stats([r | {"kind": r["group"]} for r in rows]) for k, rows in
              sets.items()}
    doc = {"tiers": stats, "eval_sets": estats, "dedupe_on_policy": n_dup,
           "unbudgeted_evals": budgets, "E_hs_dropped": len(hs_dropped),
           "disjoint": {"E_off": True, "E_hs": True, "E_off_vs_E_hs": True}}
    (out / "tiers.json").write_text(json.dumps(doc, indent=1) + "\n")
    md = ["# Encoder-study tiers (`encoder_tiers.py`)", "",
          ("Equal weights; robot s and resets from the runs' budget.json (seed counted once; "
           "evals the loop did not budget added from index durations: ")
          + ", ".join(f"{k} iter {v['iters']} {v['episodes']} ep {v['robot_s']:.0f} s"
                      for k, v in budgets.items() if v["episodes"]) + ").", "",
          ("| tier | episodes | frames (non-empty) | episode s | robot s (budget) | robot min | "
           "human resets | scripted resets | kinds |"), "|---|---|---|---|---|---|---|---|---|"]
    for k, s in stats.items():
        md.append(f"| {k} | {s['episodes']} | {s['frames']} ({s['frames_nonempty']}) | "
                  f"{s['episode_s']:.0f} | {s['robot_s']:.0f} | {s['robot_s'] / 60:.1f} | "
                  f"{s['resets_human']:.0f} | {s['resets_scripted']:.0f} | "
                  + ", ".join(f"{a} {b}" for a, b in s["kinds"].items()) + " |")
    md += ["", "| eval set | episodes | frames (non-empty) | groups |", "|---|---|---|---|"]
    for k, s in estats.items():
        md.append(f"| {k} | {s['episodes']} | {s['frames']} ({s['frames_nonempty']}) | "
                  + ", ".join(f"{a} {b}" for a, b in s["kinds"].items()) + " |")
    md += ["", (f"Disjointness (resolved paths): E_off ∩ T0-T2 = 0, E_hs ∩ T0-T2 = 0 "
                f"({len(hs_dropped)} dropped), E_off ∩ E_hs = 0. On-policy duplicates removed: "
                f"{n_dup}.")]
    (out / "tiers.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
