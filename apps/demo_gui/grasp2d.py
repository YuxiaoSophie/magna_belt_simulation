"""Along-belt side-view grasp canvas (pure) from ``assets/grasp_geom.json``
(``backend/make_silhouettes.py``).

mm, origin = the Variant grasp point, x = rest tangent ``t``, y = up, seen from outside the
holder. Dragging the fingers (shape 0) sets ``slide_mm`` (along x) and ``depth_mm`` (along the
fingers' tip-to-wrist axis ``approach_A``; + = shallower = up, Variant's ``*_depth_mm``); the
handle (shape 1) on a ring about the slid grasp point sets ``pivot_deg`` = Variant's
``*_pivot_deg`` (turn about the jaw axis ``approach x t`` ~ ``n``; + = finger tops lean toward +t
= clockwise here). Drawn as the sim moves them: turned about the slid grasp point, then moved
along the turned axis by the depth. Accepted poses: the 3D holder-clearance table ``feasible``
(nearest node collision-free at the margin) within the belt-in-jaw shallower depth limit.
"""

from __future__ import annotations

import base64
import functools
import json
import math
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from contourpy import contour_generator

GEOM = Path(__file__).resolve().parent / "assets/grasp_geom.json"
RING_MM, HANDLE_MM, HANDLE_DEG = 35.0, 3.5, -90.0  # handle below the fingertips at pivot 0
APPROX_DEG = 15.0  # pivot axis further than this from the view normal -> "shown approximately"
BELT, FINGER, FINGER_LINE = "#f97316", "rgba(51,65,85,0.85)", "#1e293b"
HOLDER, HOLDER_FILL = "#94a3b8", "rgba(148,163,184,0.35)"
LIMIT, HANDLE = "#dc2626", "#2563eb"
KNOBS = ("slide_mm", "pivot_deg", "depth_mm")
X_RANGE, Y_RANGE = [-70.0, 70.0], [-45.0, 90.0]
NAMES = {"franka": "Franka", "ur": "UR"}
_NUM = r"-?\d*\.?\d+(?:[eE][+-]?\d+)?"


@functools.cache
def load_geom(path: Path = GEOM) -> dict:
    return json.loads(Path(path).read_text())


def unpack(arm: str) -> np.ndarray:
    """The arm's 3D holder-clearance table: bool (slide, pivot, depth) nodes, True = feasible."""
    f = load_geom()[arm]["feasible"]
    shape = (f["slide"][2], f["pivot"][2], f["depth"][2])
    bits = np.unpackbits(np.frombuffer(base64.b64decode(f["bits"]), dtype=np.uint8))
    return bits[:math.prod(shape)].astype(bool).reshape(shape)


def _axes(arm: str) -> list:
    f = load_geom()[arm]["feasible"]
    return [f["slide"], f["pivot"], f["depth"]]


@functools.cache
def feasible(arm: str) -> np.ndarray:
    """``unpack`` within the belt-in-jaw shallower depth limit (the table is holder-only)."""
    m = unpack(arm).copy()
    lo, st, n = _axes(arm)[2]
    m[:, :, lo + st * np.arange(n) > load_geom()[arm]["limits"]["depth_mm"][1] + 1e-9] = False
    return m


@functools.cache
def _nodes(arm: str) -> np.ndarray:
    return np.argwhere(feasible(arm))


def default_pose(arm: str) -> dict:
    """The store's per-arm start: nominal, or the UR's shallowest collision-free depth."""
    return {"roll_deg": 0.0, **load_geom()[arm]["default_pose"]}


def _node(arm: str, *pose: float) -> tuple[int, ...] | None:
    """Indices of the nearest table node, None outside the table."""
    idx = tuple(math.floor((v - lo) / st + 0.5) for v, (lo, st, _) in zip(pose, _axes(arm)))
    return idx if all(0 <= i < n for i, (*_, n) in zip(idx, _axes(arm))) else None


def is_feasible(arm: str, slide_mm: float, pivot_deg: float, depth_mm: float) -> bool:
    node = _node(arm, slide_mm, pivot_deg, depth_mm)
    return node is not None and bool(feasible(arm)[node])


def clamp(arm: str, slide_mm: float, pivot_deg: float, depth_mm: float, turn: bool = False
          ) -> tuple[float, float, float, dict]:
    """Slide (0.1 mm), pivot (1 deg), depth (0.5 mm), kept if feasible, else the nearest feasible
    node in grid units, keeping the pivot first (the drawn region), or with ``turn`` (a pivot
    change) the slide / depth first; then the slide. + ``{knob: 1 request above / 0 below}`` for
    each knob that moved."""
    req = [round(round(float(v) / q) * q, 1) for v, q in zip((slide_mm, pivot_deg, depth_mm),
                                                           (0.1, 1.0, 0.5))]
    if is_feasible(arm, *req):
        return req[0], req[1], req[2], {}
    ax, nodes = _axes(arm), _nodes(arm)
    d = nodes - [(v - lo) / st for v, (lo, st, _) in zip(req, ax)]
    pose, piv = np.round(d[:, 0] ** 2 + d[:, 2] ** 2, 9), np.abs(d[:, 1])
    best = nodes[np.lexsort((np.abs(d[:, 0]), piv, pose) if turn else (np.abs(d[:, 0]), pose, piv))[0]]
    out = [round(lo + st * float(k), 1) for k, (lo, st, _) in zip(best, ax)]
    at = {knob: int(r > o) for knob, r, o in zip(KNOBS, req, out) if abs(r - o) > 1e-9}
    return out[0], out[1], out[2], at


def pivot_interval(arm: str, slide_mm: float, pivot_deg: float, depth_mm: float) -> list[float]:
    """The run of feasible pivots at (slide, depth) that holds ``pivot_deg``."""
    node = _node(arm, slide_mm, pivot_deg, depth_mm)
    if node is None or not feasible(arm)[node]:
        return [pivot_deg, pivot_deg]
    k, j, i = node
    col = feasible(arm)[k, :, i]
    lo, hi = j, j
    while lo > 0 and col[lo - 1]:
        lo -= 1
    while hi < len(col) - 1 and col[hi + 1]:
        hi += 1
    p0, st, _ = _axes(arm)[1]
    return [p0 + st * lo, p0 + st * hi]


def region(arm: str, pivot_deg: float) -> list[np.ndarray]:
    """Boundary rings (canvas mm) of the feasible grasp-point positions ``slide*(1,0) +
    depth*axis(pivot)`` at the pivot node nearest ``pivot_deg``."""
    (s0, ss, _), (p0, ps, n), (d0, ds, _) = _axes(arm)
    j = min(max(math.floor((pivot_deg - p0) / ps + 0.5), 0), n - 1)
    z = np.pad(feasible(arm)[:, j, :], 1).astype(np.float64)
    a = axis(arm, pivot_deg)
    return [np.c_[s0 + ss * (ln[:, 1] - 1), np.zeros(len(ln))] + np.outer(d0 + ds * (ln[:, 0] - 1), a)
            for ln in map(np.asarray, contour_generator(z=z).lines(0.5))]


def _rings(rings, dx: float = 0.0, deg: float = 0.0) -> list[np.ndarray]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [np.asarray(r, dtype=np.float64) @ [[c, s], [-s, c]] + [dx, 0.0] for r in rings]


def svg_path(rings) -> str:
    return " ".join("M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in r) + " Z" for r in rings)


def path_mean(path: str) -> np.ndarray:
    return np.array(re.findall(_NUM, path), dtype=np.float64).reshape(-1, 2).mean(0)


def shape_center(relayout: dict, k: int, half: float = 0.0) -> np.ndarray | None:
    """Centre of shape ``k`` from a plotly ``relayoutData``: its path, else whatever of
    x0/x1/y0/y1 is present (a box of half-size ``half``); None if neither axis is given."""
    p = relayout.get(f"shapes[{k}].path")
    if p is not None:
        return path_mean(p)
    c = []
    for a in "xy":
        v = [float(relayout[q]) + d for q, d in ((f"shapes[{k}].{a}0", half),
                                                 (f"shapes[{k}].{a}1", -half)) if q in relayout]
        if not v:
            return None
        c.append(sum(v) / len(v))
    return np.array(c)


def axis(arm: str, pivot_deg: float) -> np.ndarray:
    """The arm's tip-to-wrist axis ``approach_A`` turned like the fingers at ``pivot_deg``."""
    return _rings([[load_geom()[arm]["approach_A"]]], deg=-pivot_deg)[0][0]


def place(rings, slide_mm: float, pivot_deg: float, depth_mm: float = 0.0,
          arm: str | None = None) -> list[np.ndarray]:
    """Fingers turned by the pivot about the grasp point, slid (= about the slid point), then
    moved ``depth_mm`` along ``axis(arm, pivot)`` (``arm`` needed only for a depth)."""
    d = depth_mm * axis(arm, pivot_deg) if depth_mm else 0.0
    return [r + d for r in _rings(rings, dx=slide_mm, deg=-pivot_deg)]


@functools.cache
def _base(arm: str, pivot_deg: float) -> np.ndarray:
    """Mean of the silhouette path's points at slide 0, depth 0 (what a drag is measured against)."""
    return path_mean(svg_path(place(load_geom()[arm]["silhouette_A"], 0.0, pivot_deg)))


def drag_from_relayout(arm: str, relayout: dict | None, pivot_deg: float = 0.0
                       ) -> tuple[float, float] | None:
    """Unclamped (slide, depth) of a finger drag (shape 0) at the current pivot, else None: the
    drag's displacement solved exactly in the basis (x, ``axis``), so the redraw lands on the drop."""
    c = shape_center(relayout or {}, 0)
    if c is None:
        return None
    a, d = axis(arm, pivot_deg), c - _base(arm, pivot_deg)
    assert abs(a[1]) > 0.3, a  # never horizontal within the pivot limits
    depth = d[1] / a[1]
    return float(d[0] - depth * a[0]), float(depth)


def pivot_from_relayout(arm: str, relayout: dict | None, slide_mm: float = 0.0) -> float | None:
    """Unclamped pivot of a handle drag (shape 1) about the slid grasp point, else None."""
    c = shape_center(relayout or {}, 1, HANDLE_MM)
    if c is None:
        return None
    return (HANDLE_DEG - math.degrees(math.atan2(c[1], c[0] - slide_mm)) + 180.0) % 360.0 - 180.0


def _poly(rings, fill: str, line: str, name: str) -> go.Scatter:
    xy = [p for r in rings for p in [*r, [None, None]]]
    return go.Scatter(x=[p[0] for p in xy], y=[p[1] for p in xy], fill="toself", fillcolor=fill,
                      line={"color": line, "width": 1}, name=name, hoverinfo="name")


def _limit(x, y, bold: bool, name: str) -> go.Scatter:
    return go.Scatter(x=x, y=y, mode="lines", hoverinfo="name", name=name,
                      line={"color": LIMIT, "dash": "dash", "width": 3.5 if bold else 1.5})


def figure(arm: str, slide_mm: float, pivot_deg: float, depth_mm: float = 0.0,
           at_limit: dict | None = None) -> go.Figure:
    """Side view: fingers (shape 0) and pivot handle (shape 1) at the given slide / pivot / depth,
    the grasp point and the region it may take at this pivot, the feasible pivot rays; the limit of
    a clipped knob (``at_limit`` from ``clamp``) is drawn bold."""
    g, at = load_geom()[arm], at_limit or {}
    gp = np.array([slide_mm, 0.0]) + depth_mm * axis(arm, pivot_deg)
    reg = [p for r in region(arm, pivot_deg) for p in [*r, [None, None]]]
    b, r = np.asarray(g["belt_A"]), g["belt_r_mm"]
    ring = np.radians(np.linspace(0.0, 360.0, 73))
    ray = RING_MM + 10.0
    fig = go.Figure([
        _poly(g["holder_A"], HOLDER_FILL, HOLDER, "holder"),
        go.Scatter(x=np.r_[b[:, 0], b[::-1, 0]], y=np.r_[b[:, 1] + r, b[::-1, 1] - r],
                   fill="toself", fillcolor=BELT, line={"color": BELT, "width": 0},
                   name="belt", hoverinfo="name"),
        go.Scatter(x=(slide_mm + RING_MM * np.cos(ring)).tolist(),
                   y=(RING_MM * np.sin(ring)).tolist(), mode="lines", hoverinfo="skip",
                   line={"color": "#cbd5e1", "dash": "dot", "width": 1}),
        _limit([p[0] for p in reg], [p[1] for p in reg], bool({"slide_mm", "depth_mm"} & set(at)),
               "collision-free grasp point region"),
        *(_limit([slide_mm, slide_mm + ray * math.cos(math.radians(HANDLE_DEG - v))],
                 [0.0, ray * math.sin(math.radians(HANDLE_DEG - v))],
                 at.get("pivot_deg") == i, f"pivot limit {v:+g} deg")
          for i, v in enumerate(pivot_interval(arm, slide_mm, pivot_deg, depth_mm))),
        go.Scatter(x=[0.0], y=[0.0], mode="markers", name="nominal grasp point",
                   marker={"symbol": "x", "size": 8, "color": "#64748b"}, hoverinfo="name"),
        go.Scatter(x=[gp[0]], y=[gp[1]], mode="markers", name="grasp point",
                   marker={"size": 6, "color": LIMIT}, hoverinfo="name"),
    ])
    a = math.radians(HANDLE_DEG - pivot_deg)
    cx, cy = slide_mm + RING_MM * math.cos(a), RING_MM * math.sin(a)
    shapes = [
        {"type": "path",
         "path": svg_path(place(g["silhouette_A"], slide_mm, pivot_deg, depth_mm, arm)),
         "fillcolor": FINGER, "line": {"color": FINGER_LINE, "width": 1}},
        {"type": "circle", "x0": cx - HANDLE_MM, "x1": cx + HANDLE_MM, "y0": cy - HANDLE_MM,
         "y1": cy + HANDLE_MM, "fillcolor": HANDLE, "line": {"color": "#ffffff", "width": 2}},
    ]
    title = f"{NAMES[arm]} grasp — side view along the belt"
    if g["pivot_axis_deg"] > APPROX_DEG:
        title += " (pivot shown approximately)"
    fig.update_layout(
        shapes=shapes, dragmode=False, showlegend=False, height=380,
        title={"text": title, "font": {"size": 13}}, margin={"l": 44, "r": 8, "t": 36, "b": 36},
        xaxis={"range": X_RANGE, "title": "along belt mm", "zeroline": False},
        yaxis={"range": Y_RANGE, "title": "up mm", "scaleanchor": "x", "zeroline": False},
    )
    return fig
