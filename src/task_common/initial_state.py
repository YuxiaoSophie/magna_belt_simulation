"""magna-style initial joint state: a yaml of ``q_init_*`` lists that replaces the scene's
``default_joint_positions`` for a run.

The file format is magna's ``python/data/generated/*_initial_state.yaml`` -- the same file the
trajectory compiler is given with ``--initial-state`` -- so a simulated run can start from the
joint positions a hardware run starts from.  Validation is strict (exact key set, exact
lengths, finite numbers): a silently half-applied start pose would look like a controller bug.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import yaml


def load_initial_state(path: str | Path, expected: Mapping[str, int]) -> dict[str, list[float]]:
    """Read ``path`` and return ``{key: values}`` for exactly ``expected`` (key -> length)."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"initial-state file not found: {path}") from error
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path}: expected a yaml mapping, got {type(raw).__name__}")

    missing = sorted(set(expected) - set(raw))
    extra = sorted(set(raw) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{path}: initial state must define exactly {sorted(expected)}; "
            f"missing {missing or 'none'}, unexpected {extra or 'none'}"
        )

    state: dict[str, list[float]] = {}
    for key, length in expected.items():
        values = raw[key]
        if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
            raise ValueError(f"{path}: {key} must be a list of {length} numbers, got {values!r}")
        if len(values) != length:
            raise ValueError(
                f"{path}: {key} must have {length} values, got {len(values)}: {list(values)}"
            )
        try:
            state[key] = [float(v) for v in values]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: {key} has a non-numeric entry: {list(values)}") from error
        if not all(math.isfinite(v) for v in state[key]):
            raise ValueError(f"{path}: {key} has a non-finite entry: {state[key]}")
    return state
