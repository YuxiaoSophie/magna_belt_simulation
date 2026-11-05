"""One plotly template for every figure in the app."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT = 'Inter, "Segoe UI", system-ui, -apple-system, Roboto, sans-serif'


def apply() -> None:
    t = go.layout.Template(pio.templates["plotly_white"])
    t.layout.update(
        font={"family": FONT, "size": 12}, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin={"l": 48, "r": 16, "t": 40, "b": 40},
        colorway=["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2"],
        legend={"orientation": "h", "y": 1.1, "x": 0},
    )
    t.layout.xaxis.gridcolor = t.layout.yaxis.gridcolor = "#eef2f6"
    pio.templates["demo"] = t
    pio.templates.default = "demo"
