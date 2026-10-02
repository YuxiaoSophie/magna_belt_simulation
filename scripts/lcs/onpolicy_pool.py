#!/usr/bin/env python3
"""V8 on-policy collection pool: (target x start) cells disjoint from the seat6 + yaw12 evals.

Subcommands:
    build  --out data/lcs/onpolicy/v8/pool.json      enumerate starts, targets, strata, eval cells
    stats  --pool pool.json                           pool size per stratum
    sample --pool pool.json --iter i --n 12 --out data/lcs/onpolicy/v8/iter<i>/manifest.json
           stratified 4/2/2/4 (x n/12), seed 20261001 + i; unused cells first (reads earlier
           manifests)
    check  --manifest m.json [...] | --all           exit 1 on any eval-cell overlap

Start groups: G = grasp_variants set1 minus {gv_01, gv_05}, set2, set3; S = set4 minus gv_00
(its nominal grasp); E = ee_offsets; U = ur_depth minus {ud+3_r0, ud+6_r0}, ur_depth_4p5.
Targets: T0 = the four seat targets (V4a t1 goals); Tyaw = flat_engaged_yaw{p05,m05,p10,m10}.
Strata: fe_GS = flat_engaged x (G+S); urtip_U = urtip family x U; T0_E = T0 x E;
Tyaw_GS = Tyaw x (G+S), one cell per yaw target per iteration.

Run:
    uv run --frozen python scripts/lcs/onpolicy_pool.py build --out data/lcs/onpolicy/v8/pool.json
    uv run --frozen python scripts/lcs/onpolicy_pool.py sample \\
        --pool data/lcs/onpolicy/v8/pool.json --iter 0 \\
        --out data/lcs/onpolicy/v8/iter0/manifest.json
    uv run --frozen python scripts/lcs/onpolicy_pool.py check --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STARTS = Path("data/lcs/start_states")
SYN = Path("data/lcs/synthetic_targets")
V8 = Path("data/lcs/onpolicy/v8")
T0_GOALS = Path("data/lcs/mpc_eval/20260929-v4-retrain/t1/goals")
TYAW_GOALS = Path("data/lcs/mpc_eval/20261001-v8/goals_v4a/goals")
SEED0 = 20261001
LCM_URL_PLACEHOLDER = "<LCM_URL>"

T0 = ["flat_engaged", "flat_engaged_urtip", "flat_engaged_urtip6", "flat_engaged_urtip6_high"]
URTIP = T0[1:]
TYAW = {"flat_engaged_yawp05": 5.0, "flat_engaged_yawm05": -5.0,
        "flat_engaged_yawp10": 10.0, "flat_engaged_yawm10": -10.0}
EVAL_YAWS = (15.0, -15.0)

# (start_set, start_id); start_set "" = a single snapshot file under data/lcs/start_states
NOMINAL = ("", "pre_place_1_osc")
SEAT6 = [("flat_engaged", NOMINAL), ("flat_engaged", ("grasp_variants/set1", "gv_01")),
         ("flat_engaged", ("grasp_variants/set1", "gv_05")),
         ("flat_engaged_urtip", ("ur_depth", "ud+3_r0")),
         ("flat_engaged_urtip6", ("ur_depth", "ud+6_r0")),
         ("flat_engaged_urtip6_high", ("ur_depth", "ud+6_r0"))]
EVAL_CELLS = SEAT6 + [(f"{t}_yaw{s}", st) for s in ("p15", "m15") for t, st in SEAT6]

GROUPS = {  # group -> [(start_set, excluded ids)]
    "G": [("grasp_variants/set1", {"gv_01", "gv_05"}), ("grasp_variants/set2", set()),
          ("grasp_variants/set3", set())],
    "S": [("grasp_variants/set4", {"gv_00"})],
    "E": [("ee_offsets", set())],
    "U": [("ur_depth", {"ud+3_r0", "ud+6_r0"}), ("ur_depth_4p5", set())],
}
STRATA = {  # stratum -> (targets, start groups, cells per iteration)
    "fe_GS": (["flat_engaged"], ("G", "S"), 4),
    "urtip_U": (URTIP, ("U",), 2),
    "T0_E": (T0, ("E",), 2),
    "Tyaw_GS": (list(TYAW), ("G", "S"), 4),
}
EXPECTED = {"fe_GS": 41, "urtip_U": 30, "T0_E": 16, "Tyaw_GS": 164}


def sha(path: Path) -> str:
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()


def start_file(start_set: str, start_id: str) -> Path:
    if not start_set:
        return STARTS / f"{start_id}.npz"
    index = json.loads((REPO / STARTS / start_set / "index.json").read_text())
    row = next(r for r in index["variants"] if str(r["id"]) == start_id)
    return STARTS / start_set / row["file"]


def eval_start_shas() -> dict[str, str]:
    keys = sorted({st for _, st in EVAL_CELLS})
    return {f"{s}/{i}".lstrip("/"): sha(start_file(s, i)) for s, i in keys}


def list_starts() -> dict[str, list[dict]]:
    out = {}
    for group, sets in GROUPS.items():
        rows = []
        for start_set, excluded in sets:
            index = json.loads((REPO / STARTS / start_set / "index.json").read_text())
            for r in index["variants"]:
                vid = str(r["id"])
                if vid in excluded or not r.get("held") or r.get("file") is None:
                    continue
                f = STARTS / start_set / r["file"]
                rows.append({"group": group, "start_set": start_set, "start_id": vid,
                             "start_file": str(f), "start_sha256": sha(f),
                             "start_states_arg": str(STARTS / start_set), "variants_arg": vid})
        out[group] = rows
    return out


def list_targets() -> dict[str, dict]:
    out = {}
    for t in T0 + list(TYAW):
        gdir = (T0_GOALS if t in T0 else TYAW_GOALS) / t
        gj = json.loads((REPO / gdir / "goal.json").read_text())
        obs = SYN / t / "observation.npz"
        assert sha(obs) == gj["observation_sha256"], f"{t}: observation sha != goal.json"
        assert gj.get("params"), f"{t}: goal.json has no params yaml"
        out[t] = {"target": t, "family": "T0" if t in T0 else "Tyaw",
                  "target_yaw": TYAW.get(t, 0.0), "observation": str(obs),
                  "observation_sha256": gj["observation_sha256"], "goals_dir": str(gdir),
                  "deploy": str(gdir / "deploy.npz"), "deploy_sha256": sha(gdir / "deploy.npz"),
                  "demo_goals": str(gdir / "demo_goals.npz"),
                  "demo_goals_sha256": sha(gdir / "demo_goals.npz"),
                  "params_yaml": gj["params"], "lcs_yaml": gj["lcs_yaml"]}
    return out


def cell_key(c: dict) -> str:
    return f"{c['target']}|{c['start_set']}/{c['start_id']}"


def violations(cells: list[dict], shas: dict[str, str]) -> list[str]:
    """Why each cell collides with the eval set (empty = disjoint)."""
    ev = {(t, s, i) for t, (s, i) in EVAL_CELLS}
    bad_sha = set(shas.values())
    out = []
    for c in cells:
        k = cell_key(c)
        if (c["target"], c["start_set"], c["start_id"]) in ev:
            out.append(f"{k}: eval cell")
        if c["start_sha256"] in bad_sha:
            out.append(f"{k}: eval start sha {c['start_sha256'][:12]}")
        if float(c["target_yaw"]) in EVAL_YAWS or c["target"].endswith(("_yawp15", "_yawm15")):
            out.append(f"{k}: eval yaw {c['target_yaw']}")
        if sha(Path(c["start_file"])) != c["start_sha256"]:
            out.append(f"{k}: start file changed since the pool was built")
    return out


def build(args) -> int:
    starts, targets, shas = list_starts(), list_targets(), eval_start_shas()
    strata = {}
    for name, (tnames, groups, _) in STRATA.items():
        cells = []
        for t in tnames:
            for g in groups:
                for s in starts[g]:
                    cells.append({"stratum": name, "target": t,
                                  "target_yaw": targets[t]["target_yaw"], **s})
        strata[name] = sorted(cells, key=cell_key)
    allc = [c for v in strata.values() for c in v]
    bad = violations(allc, shas)
    if bad:
        print("\n".join(bad))
        return 1
    assert len({cell_key(c) for c in allc}) == len(allc), "duplicate cells"
    pool = {"kind": "v8_onpolicy_pool", "seed0": SEED0,
            "quota": {k: v[2] for k, v in STRATA.items()},
            "groups": {g: [s for s, _ in sets] for g, sets in GROUPS.items()},
            "excluded_from_groups": {g: {s: sorted(ex) for s, ex in sets} for g, sets in
                                     GROUPS.items()},
            "eval": {"cells": [{"target": t, "start_set": s, "start_id": i}
                               for t, (s, i) in EVAL_CELLS],
                     "start_sha256": shas, "yaws": list(EVAL_YAWS)},
            "starts": starts, "targets": targets,
            "sizes": {k: len(v) for k, v in strata.items()}, "strata": strata}
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=1) + "\n")
    print(f"wrote {args.out}: " + ", ".join(f"{k} {len(v)}" for k, v in strata.items()))
    return 0


def stats(args) -> int:
    pool = json.loads((REPO / args.pool).read_text())
    ok = True
    for k, cells in pool["strata"].items():
        n = len(cells)
        ok &= n == EXPECTED[k]
        by_t = {}
        for c in cells:
            by_t[c["target"]] = by_t.get(c["target"], 0) + 1
        print(f"{k:8s} {n:4d} (expected {EXPECTED[k]}, quota {pool['quota'][k]}) {by_t}")
    print(f"starts: {({g: len(v) for g, v in pool['starts'].items()})}; "
          f"total cells {sum(len(v) for v in pool['strata'].values())}")
    return 0 if ok else 1


def earlier_used(pool_dir: Path, it: int) -> set[str]:
    used = set()
    for j in range(it):
        m = pool_dir / f"iter{j}" / "manifest.json"
        if m.is_file():
            used |= {cell_key(c) for c in json.loads(m.read_text())["cells"]}
    return used


def draw(rng, cells: list[dict], n: int, used: set[str]) -> list[dict]:
    """``n`` cells, unused ones first; each part drawn from a sorted list."""
    fresh = sorted((c for c in cells if cell_key(c) not in used), key=cell_key)
    old = sorted((c for c in cells if cell_key(c) in used), key=cell_key)
    k = min(n, len(fresh))
    pick = [fresh[i] for i in rng.choice(len(fresh), size=k, replace=False)] if k else []
    if n > k:
        pick += [old[i] for i in rng.choice(len(old), size=n - k, replace=False)]
    return pick


def sample(args) -> int:
    pool_path = REPO / args.pool
    pool = json.loads(pool_path.read_text())
    quota = dict(pool["quota"])
    total = sum(quota.values())
    if args.n % total:  # --n 24 = the quota twice (plan Q1)
        print(f"--n {args.n} is not a multiple of the quota total {total} ({quota})")
        return 2
    quota = {k: v * (args.n // total) for k, v in quota.items()}
    rng = np.random.default_rng(SEED0 + args.iter)
    used = earlier_used(pool_path.parent, args.iter)
    cells = []
    for name in STRATA:
        pool_cells = pool["strata"][name]
        if name == "Tyaw_GS":  # one per yaw target
            per = quota[name] // len(TYAW)
            for t in sorted(TYAW):
                cells += draw(rng, [c for c in pool_cells if c["target"] == t], per, used)
        else:
            cells += draw(rng, pool_cells, quota[name], used)
    tg = pool["targets"]
    out_cells = []
    for c in cells:
        t = tg[c["target"]]
        out_cells.append({
            "cell": cell_key(c), "stratum": c["stratum"], "target": c["target"],
            "target_yaw": c["target_yaw"], "target_observation_sha256": t["observation_sha256"],
            "start_group": c["group"], "start_set": c["start_set"], "start_id": c["start_id"],
            "start_file": c["start_file"], "start_sha256": c["start_sha256"],
            "start_states_arg": c["start_states_arg"], "variants_arg": c["variants_arg"],
            "goals_dir": t["goals_dir"], "deploy": t["deploy"], "demo_goals": t["demo_goals"],
            "params_yaml": t["params_yaml"], "lcs_yaml": t["lcs_yaml"],
            "lcm_url": LCM_URL_PLACEHOLDER})
    man = {"kind": "v8_onpolicy_manifest", "iter": args.iter, "seed": SEED0 + args.iter,
           "pool": str(args.pool), "pool_sha256": sha(Path(args.pool)),
           "n_previously_used": len(used),
           "reused": sorted(c["cell"] for c in out_cells if c["cell"] in used),
           "counts": {k: sum(c["stratum"] == k for c in out_cells) for k in STRATA},
           "cells": out_cells}
    text = json.dumps(man, indent=1) + "\n"
    out = REPO / args.out
    if out.exists() and out.read_text() != text:
        print(f"{args.out} exists with different content; not overwritten")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {args.out}: {man['counts']} (reused {len(man['reused'])})")
    return 0


def check(args) -> int:
    paths = [Path(p) for p in args.manifest or []]
    if args.all:
        found = sorted((REPO / V8).rglob("manifest.json"))
        paths += [p.relative_to(REPO) if p.is_relative_to(REPO) else p for p in found]
    shas = eval_start_shas()
    bad = []
    pool_path = REPO / V8 / "pool.json"
    if args.all and pool_path.is_file():
        pool = json.loads(pool_path.read_text())
        pc = [c for v in pool["strata"].values() for c in v]
        pv = violations(pc, shas)
        bad += [f"{V8 / 'pool.json'}: {v}" for v in pv]
        print(f"{V8 / 'pool.json'}: {len(pc)} cells, {'FAIL' if pv else 'PASS'}")
    for p in paths:
        cells = json.loads((REPO / p).read_text())["cells"]
        v = violations(cells, shas)
        bad += [f"{p}: {x}" for x in v]
        counts = {k: sum(c["stratum"] == k for c in cells) for k in STRATA}
        print(f"{p}: {len(cells)} cells {counts} {'FAIL' if v else 'PASS'}")
    if not paths and not args.all:
        print("nothing to check (--manifest or --all)")
        return 2
    if bad:
        print("\n".join(bad))
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=V8 / "pool.json")
    s = sub.add_parser("stats")
    s.add_argument("--pool", type=Path, default=V8 / "pool.json")
    m = sub.add_parser("sample")
    m.add_argument("--pool", type=Path, default=V8 / "pool.json")
    m.add_argument("--iter", type=int, required=True)
    m.add_argument("--n", type=int, default=12)
    m.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--manifest", nargs="*")
    c.add_argument("--all", action="store_true", help="pool.json + every manifest under v8/")
    args = ap.parse_args()
    return {"build": build, "stats": stats, "sample": sample, "check": check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
