"""Label lookup and shape-flag helpers."""

from __future__ import annotations

from collections.abc import Sequence

import newton


def _index_by_label(labels: Sequence[str], label: str, kind: str) -> int:
    for i, name in enumerate(labels):
        if name == label:
            return i
    leaf = label.rsplit("/", 1)[-1]
    near = [f"{i}:{n}" for i, n in enumerate(labels) if leaf in str(n)]
    raise KeyError(
        f"{kind} label {label!r} not found. Near misses: {near[:20] or '(none)'}. "
        f"Total {kind} labels: {len(labels)}"
    )


def body_index(labels: Sequence[str], label: str) -> int:
    """Exact-match body label lookup; raises KeyError listing near-misses."""
    return _index_by_label(labels, label, "body")


def joint_index(labels: Sequence[str], label: str) -> int:
    """Exact-match joint label lookup; raises KeyError listing near-misses."""
    return _index_by_label(labels, label, "joint")


def body_label_endswith(labels: Sequence[str], suffix: str) -> int:
    """Index of the single body label ending in ``suffix``; raises unless unique."""
    matches = [i for i, n in enumerate(labels) if str(n).endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"expected exactly one body label ending in {suffix!r}, got "
            f"{[(i, labels[i]) for i in matches]}"
        )
    return matches[0]


def label_shapes_by_body(builder: newton.ModelBuilder, shape_start: int, shape_end: int) -> int:
    """Relabel anonymous shapes as ``<body>/<kind><n>``; returns how many were renamed.

    Newton's USD and MJCF importers label every shape, but ``add_urdf`` does not:
    the Franka's 77 shapes all arrive as ``shape_<n>``, and the viewer names its
    render batches from those labels. The owning body is labelled correctly, so
    derive from it and the arm nests per link instead of as anonymous nodes.
    """
    visible_flag = int(newton.ShapeFlags.VISIBLE)
    counters: dict[tuple[str, str], int] = {}
    relabelled = 0
    for shape in range(shape_start, shape_end):
        label = builder.shape_label[shape] or ""
        if label and not label.startswith("shape_"):
            continue  # the importer gave it a real name; leave it alone
        body = builder.shape_body[shape]
        base = builder.body_label[body] if 0 <= body < len(builder.body_label) else "world"
        kind = "visual" if int(builder.shape_flags[shape]) & visible_flag else "collision"
        index = counters.get((base, kind), 0)
        counters[(base, kind)] = index + 1
        builder.shape_label[shape] = f"{base}/{kind}{index}"
        relabelled += 1
    return relabelled
