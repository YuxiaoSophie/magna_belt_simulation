"""Session directory model: <root>/{starts,demos,miles,train,eval,jobs,telemetry.json,cmd.json}."""

from __future__ import annotations

import json
import time
from pathlib import Path

from apps.demo_gui.config import SESSION_ROOT


class Session:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()  # jobs run with other cwds (training: lcs_learning)

    def _dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    starts = property(lambda self: self._dir("starts"))
    demos = property(lambda self: self._dir("demos"))
    miles = property(lambda self: self._dir("miles"))
    train = property(lambda self: self._dir("train"))
    eval = property(lambda self: self._dir("eval"))
    jobs = property(lambda self: self._dir("jobs"))

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def telemetry_path(self) -> Path:
        return self.root / "telemetry.json"

    @property
    def cmd_path(self) -> Path:
        return self.root / "cmd.json"

    @classmethod
    def new(cls, stamp: str | None = None, base: Path = SESSION_ROOT) -> Session:
        root = Path(base) / (stamp or time.strftime("%Y%m%d-%H%M"))
        root.mkdir(parents=True, exist_ok=True)
        return cls(root)

    @staticmethod
    def list(base: Path = SESSION_ROOT) -> list[Session]:
        base = Path(base)
        return [Session(p) for p in sorted(base.iterdir()) if p.is_dir()] if base.is_dir() else []


def read_index(d: str | Path) -> list[dict]:
    """``<d>/index.json`` rows: ``episodes`` (harness runs) or ``variants`` (start sets)."""
    p = Path(d) / "index.json"
    if not p.is_file():
        return []
    idx = json.loads(p.read_text())
    return idx.get("episodes", idx.get("variants", []))


def read_buffer(path: str | Path) -> list[dict]:
    """``buffer.json`` ``rollouts`` rows ([] if the file is absent)."""
    p = Path(path)
    return json.loads(p.read_text()).get("rollouts", []) if p.is_file() else []
