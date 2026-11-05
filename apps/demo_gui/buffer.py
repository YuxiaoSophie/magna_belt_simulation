"""Training buffer of a session: its demos cut per stage (as make_tier148.py) + MILES rows.

Each demo -> a stage-1 row (``s1_chain``, v12_128 ``s1_cut``: to the UR exit, so no UR-exit
frames) and a stage-2 row (v12_127 ``cut_one``: from ``chain_from[1]``, failures cut at onset),
cut copies under ``<out>/tiers/eps/{s1,cd}``. MILES rows (``miles_seed.py``, stage 1) verbatim
+ ``stage: 1``. Optional repo seed = the W150all tier rows verbatim.
"""

from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path

from apps.demo_gui import telemetry
from apps.demo_gui.config import REPO
from apps.demo_gui.session import read_buffer, read_index

CMP = REPO / "data/lcs/realfirst/cmp"
SEED = CMP / "v12_152/tiers/W150all/buffer.json"


def path(session, tag: str) -> Path:
    """Outside ``<s>/train/<tag>``: the trainer refuses an existing run dir."""
    return session.train / "buffers" / tag / "buffer.json"


def _cutters():
    """v12_128 ``make_tier`` (its ``mt`` = v12_127 ``make_tier``), imported, never edited."""
    if "mt128" not in sys.modules:
        spec = importlib.util.spec_from_file_location("mt128", CMP / "v12_128/make_tier.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mt128"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["mt128"]


def candidates(session) -> list[dict]:
    """Checklist rows: one per session demo episode, one per MILES run, the repo seed tier."""
    out = []
    for d in sorted(p for p in session.demos.iterdir() if p.is_dir()):
        out += [{"kind": "demo", "label": f"demo {d.name}/{ep['file']} ({ep.get('outcome')})",
                 "episode": str(d / ep["file"])}
                for ep in read_index(d) if ep.get("file")]
    for d in sorted(p for p in session.miles.iterdir() if (p / "buffer.json").is_file()):
        n = sum(r.get("status", "ok") == "ok" for r in read_buffer(d / "buffer.json"))
        out.append({"kind": "miles", "label": f"MILES {d.name} ({n} rows)",
                    "buffer": str(d / "buffer.json")})
    if SEED.is_file():
        out.append({"kind": "seed", "label": "repo seed: v12_152 tier W150all (5 demos + MILES)",
                    "buffer": str(SEED)})
    return out


def demo_rows(src: Path, tag: str, out: Path) -> tuple[list[dict], list[str]]:
    """The stage-1 and stage-2 rows of one demo episode, and why a stage was skipped."""
    m8 = _cutters()
    m8.HERE = m8.mt.HERE = out  # cut copies -> <out>/tiers/eps/{s1,cd}, as make_tier148.py
    job = {"src": str(src), "key": src.stem, "tag": tag}
    rows, skipped = [], []
    x1, x2 = m8.s1_cut(job), m8.mt.cut_one(job)
    common = {"iter": 0, "status": "ok", "weight": 1.0, "start": {"set": "demo_gui", "key": tag},
              "scene": "wr_pin640b1"}
    if "dst" in x1:
        rows.append({"id": f"{tag}__{src.stem}_s1", "kind": "s1_chain", "episode": x1["dst"],
                     "label_human": x1["seated"], "n_frames": x1["n"], "cut": x1["cut"],
                     "source_episode": x1["src"], "stage": 1, **common})
    else:
        skipped.append(f"{tag} stage 1: {x1.get('skip')}")
    if "dst" in x2:
        rows.append({"id": f"{tag}__{src.stem}_s2", "kind": x2["kind"], "episode": x2["dst"],
                     "label_human": x2["success"], "n_frames": x2["n"] - x2["a"], "cut": x2["cut"],
                     "source_episode": x2["src"], "stage": 2, **common})
    else:
        skipped.append(f"{tag} stage 2: {x2.get('skip')}")
    return rows, skipped


def build(session, selected: list[dict], out_path: str | Path) -> dict:
    """Write ``out_path`` (W150all schema, weight 1.0, absolute ``episode``); returns counts per
    kind (+ ``skipped`` reasons)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for c in selected:
        if c["kind"] == "demo":
            src = Path(c["episode"])
            r, s = demo_rows(src, src.parent.name, out_path.parent)
            rows += r
            skipped += s
        elif c["kind"] == "miles":
            rows += [{**r, "episode": str(Path(r["episode"]).resolve()), "weight": 1.0, "stage": 1}
                     for r in read_buffer(c["buffer"]) if r.get("status", "ok") == "ok"]
        elif c["kind"] == "seed":
            rows += [{**r, "weight": 1.0} for r in read_buffer(c["buffer"])]
    rows = list({r["episode"]: r for r in reversed(rows)}.values())[::-1]  # first row per episode
    telemetry.write(out_path, {"version": 2, "iter": 0, "n": len(rows), "tier": out_path.parent.name,
                               "rollouts": rows, "action_definition": "cmd_delta",
                               "source": f"demo GUI session {session.name} (apps/demo_gui/buffer.py)"})
    counts = dict(collections.Counter(r["kind"] for r in rows))
    return {**counts, "skipped": skipped} if skipped else counts
