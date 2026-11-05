"""What every backend (mock, sim) offers the panels; long operations return a ``jobs.Job``."""

from __future__ import annotations

from typing import Protocol

from apps.demo_gui.jobs import Job
from apps.demo_gui.session import Session


class Backend(Protocol):
    supports_teleop: bool = False

    def make_start(self, session: Session, franka: dict, ur: dict) -> Job:
        """New start state from per-arm ``{slide_mm, roll_deg, pivot_deg, depth_mm}`` (pivot /
        depth optional, 0 = none; ``Variant``'s ``*_pivot_deg`` / ``*_depth_mm``, depth + =
        shallower; the GUI keeps roll 0) -> ``<s>/starts/``."""

    def record_demo(self, session: Session, start_id: str, mode: str = "scripted") -> Job:
        """Record one demo from ``<s>/starts/<start_id>_osc.npz`` into ``<s>/demos/<n>/``."""

    def stop_demo(self, session: Session) -> None: ...

    def reset_demo(self, session: Session) -> None:
        """Stop and delete the in-progress demo dir."""

    def run_miles(self, session: Session, demos: list[dict], cycles: int = 4,
                  wp_stride: int = 5) -> Job:
        """MILES around ``demos`` (``{name, episode, start}``) -> ``<s>/miles/<n>/``."""

    def train(self, session: Session, tag: str, rows: list[dict], epochs: int,
              seed: int = 5) -> Job: ...

    def export(self, session: Session, tag: str) -> Job: ...

    def evaluate(self, session: Session, name: str, model: dict, starts: list[str],
                 repeats: int) -> Job: ...
