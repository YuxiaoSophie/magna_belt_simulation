#!/usr/bin/env python3
"""Offline screen of nearest-initial-latent demo selection (``WaypointBank`` rule).

Per deploy: encode (numpy twin, ``pcd_belt_seg``) frame 0 of every seed demo and of every
rollout; each rollout picks the demo whose initial latent is nearest (whitened by the deploy's
``z_std``), each demo the nearest OTHER demo (leave-one-out). Reports whether the pick has the
same grasp variant (start-state ``set/variant``, sim metadata: report only) as a confusion table,
the distances to the pick and to the nearest same-variant demo.

    uv run --frozen python scripts/lcs/bank_screen.py --deploy V12m=<deploy.npz> \\
        --demos <demo npz ...> --rollouts data/lcs/realfirst/cmp/full/iter*/collect/*.npz \\
        --out data/lcs/realfirst/cmp/bank_select/screen
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from latent_waypoints import demo_tag

from task_common.latent_encoder import LatentEncoder, LearnedLcs

SEG_KEY = "pcd_belt_seg"


def first_cloud(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as d:
        return np.asarray(d[SEG_KEY][0], np.float32)


def outcome(path: Path) -> str:
    with np.load(path, allow_pickle=True) as d:
        return str(d["sim_outcome"]) if "sim_outcome" in d.files else "?"


def reset_kind(path: Path) -> str:
    """``scripted`` (jittered re-grasp after a scripted reset) or ``snapshot``."""
    with np.load(path, allow_pickle=True) as d:
        if "sim_start_state" not in d.files:
            return "?"
        return "scripted" if json.loads(str(d["sim_start_state"])).get("from_reset") else (
            "snapshot")


def screen(deploy: Path, demos: list[Path], rollouts: list[Path], clouds: dict) -> dict:
    enc, lcs = LatentEncoder.load(deploy), LearnedLcs.load(deploy)
    zs = np.asarray(lcs.z_std, np.float64)

    def z0(p):
        return np.asarray(enc.encode(clouds[str(p)]), np.float64)

    Zd = np.stack([z0(p) for p in demos])
    tags = [demo_tag(p) for p in demos]
    D = np.linalg.norm((Zd[:, None] - Zd[None]) / zs, axis=-1)
    rows = []
    for i, p in enumerate(demos):
        d = D[i].copy()
        d[i] = np.inf
        rows.append(pick_row("demo", p, tags[i], d, tags))
    for p in rollouts:
        d = np.linalg.norm((Zd - z0(p)) / zs, axis=-1)
        rows.append(pick_row("rollout", p, demo_tag(p), d, tags))
    return {"deploy": str(deploy), "rows": rows}


def pick_row(kind: str, path: Path, tag: str, d: np.ndarray, tags: list[str]) -> dict:
    j = int(np.argmin(d))
    same = [k for k, t in enumerate(tags) if t == tag and np.isfinite(d[k])]
    order = np.argsort(d)
    return {"kind": kind, "episode": str(path), "tag": tag, "pick": tags[j], "pick_index": j,
            "d_pick": float(d[j]), "d_second": float(d[order[1]]),
            "d_same": float(min(d[k] for k in same)) if same else None,
            "has_demo": bool(same), "correct": tags[j] == tag,
            "outcome": outcome(path) if kind == "rollout" else None,
            "reset": reset_kind(path) if kind == "rollout" else None}


def confusion(rows: list[dict]) -> str:
    true = sorted({r["tag"] for r in rows})
    pick = sorted({r["pick"] for r in rows})
    c = Counter((r["tag"], r["pick"]) for r in rows)
    lines = ["| true \\ pick | " + " | ".join(pick) + " | correct |",
             "|---" * (len(pick) + 2) + "|"]
    for t in true:
        n = sum(c[(t, q)] for q in pick)
        ok = c[(t, t)]
        lines.append(f"| {t} | " + " | ".join(str(c[(t, q)] or "") for q in pick)
                     + f" | {ok}/{n} |")
    return "\n".join(lines)


def _split(rol_c: list[dict], rol: list[dict], kind: str) -> str:
    sub = [x for x in rol_c if x["reset"] == kind]
    same_set = sum(x["pick"].split("/")[0] == x["tag"].split("/")[0] for x in sub)
    d = [x["d_pick"] for x in rol if x["reset"] == kind] or [np.nan]
    return (f"  - {kind} starts: {sum(x['correct'] for x in sub)}/{len(sub)} same variant, "
            f"{same_set} same set; median d pick (all) {np.median(d):.2f}")


def report(res: dict[str, dict]) -> str:
    out = ["# Nearest-initial-latent demo selection: offline screen", "",
           ("Key = encoded first frame (`pcd_belt_seg[0]`), whitened distance (deploy `z_std`). "
            "Demos: leave-one-out among the demos; rollouts: among all demos. Tags = "
            "start-state `set/variant` (sim metadata, report only)."), ""]
    for label, r in res.items():
        rows = r["rows"]
        dem = [x for x in rows if x["kind"] == "demo"]
        rol = [x for x in rows if x["kind"] == "rollout"]
        rol_c = [x for x in rol if x["has_demo"]]
        dem_c = [x for x in dem if x["has_demo"]]
        same_set = sum(x["pick"].split("/")[0] == x["tag"].split("/")[0] for x in dem)
        out += [f"## {label} (`{r['deploy']}`)", "",
                (f"- demos (leave-one-out, variant has another demo): "
                 f"{sum(x['correct'] for x in dem_c)}/{len(dem_c)} same variant; set-level "
                 f"{same_set}/{len(dem)}"),
                (f"- rollouts whose variant has a demo: {sum(x['correct'] for x in rol_c)}/"
                 f"{len(rol_c)} same variant; rollouts without a demo of their variant: "
                 f"{len(rol) - len(rol_c)}"),
                *[_split(rol_c, rol, k) for k in ("snapshot", "scripted")],
                "", "Demos (leave-one-out):", "", confusion(dem), "",
                "Rollouts:", "", confusion(rol), "",
                "| rollout | start | true | outcome | pick | d pick | d 2nd | d same variant |",
                "|---|---|---|---|---|---|---|---|"]
        for x in rol:
            name = "/".join(Path(x["episode"]).parts[-3:])
            ds = "-" if x["d_same"] is None else f"{x['d_same']:.2f}"
            out.append(f"| {name} | {x['reset']} | {x['tag']} | {x['outcome']} | {x['pick']} | "
                       f"{x['d_pick']:.2f} | {x['d_second']:.2f} | {ds} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deploy", nargs="+", required=True, metavar="LABEL=PATH")
    p.add_argument("--demos", type=Path, nargs="+", required=True)
    p.add_argument("--rollouts", type=Path, nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True, help="dir: screen.{md,json}")
    a = p.parse_args()
    clouds = {str(x): first_cloud(x) for x in [*a.demos, *a.rollouts]}
    res = {}
    for spec in a.deploy:
        label, path = spec.split("=", 1)
        res[label] = screen(Path(path), a.demos, a.rollouts, clouds)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "screen.json").write_text(json.dumps(res, indent=1) + "\n")
    text = report(res)
    (a.out / "screen.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
