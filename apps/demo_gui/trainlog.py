"""Parse the refit trainer's ``train.log`` (``[epoch N] k=v ...`` lines) and ``refit_report.json``."""

from __future__ import annotations

import json
import re
from pathlib import Path

EPOCH = re.compile(r"^\[epoch (\d+)\]")
KV = re.compile(r"(\S+)=(\S+)")


def parse(path: str | Path) -> dict[str, list]:
    """``{"epoch": [...], key: [...]}`` of every numeric ``k=v`` on ``[epoch N]`` lines; a key
    missing on some epoch reads ``None`` there, so every list is as long as ``epoch``."""
    p = Path(path)
    out: dict[str, list] = {"epoch": []}
    if not p.is_file():
        return out
    for line in p.read_text(errors="replace").splitlines():
        if not (m := EPOCH.match(line)):
            continue
        n = len(out["epoch"])
        out["epoch"].append(int(m.group(1)))
        for k, v in KV.findall(line):
            try:
                x = float(v.rstrip(","))
            except ValueError:
                continue
            out.setdefault(k, [None] * n).append(x)
        for col in out.values():
            if len(col) == n:
                col.append(None)
    return out


def last_epoch(path: str | Path) -> int | None:
    ep = parse(path)["epoch"]
    return ep[-1] if ep else None


def report(path: str | Path) -> dict | None:
    """``refit_report.json`` of a run dir (or the file itself); ``None`` until it exists."""
    p = Path(path)
    p = p / "refit_report.json" if p.is_dir() else p
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None
