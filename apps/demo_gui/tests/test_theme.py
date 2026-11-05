"""Theme checks, CPU-only (mock backend, Flask test client): css served, template, cards, ids.

uv run --frozen python apps/demo_gui/tests/test_theme.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import plotly.io as pio
from apps.demo_gui.app import Ctx, build_app
from apps.demo_gui.backends.mock import MockBackend
from apps.demo_gui.config import Cfg
from apps.demo_gui.panels import browser, collect, evaluate, miles, train
from apps.demo_gui.session import Session

# component ids recorded from the pre-theme layouts (styling must not change them); app: topbar + viewer dock
IDS = {
    "app": ['dock', 'dock-collapsed', 'dock-open', 'dock-src', 'dock-title', 'dock-toggle', 'dock-view', 'job-chips', 'tabs', 'tick'],
    "browser": ['br-cloud', 'br-frame', 'br-refresh', 'br-sel', 'br-table', 'br-traces', 'br-viewer-msg', 'br-viewer-open', 'br-viewer-stop'],
    "collect": ['col-buf', 'col-canvas-franka', 'col-canvas-ur', 'col-cloud', 'col-grasp', 'col-grip-franka', 'col-grip-ur', 'col-job', 'col-make', 'col-pivot-franka', 'col-pivot-ur', 'col-pose-franka', 'col-pose-ur', 'col-readout-franka', 'col-readout-ur', 'col-record', 'col-reset', 'col-start', 'col-status', 'col-stop', 'col-tdone', 'col-teleop', 'col-tick', 'col-tpause'],
    "evaluate": ['eval-3d', 'eval-dir', 'eval-episode', 'eval-frame', 'eval-job', 'eval-latent', 'eval-md', 'eval-model', 'eval-play', 'eval-play-btn', 'eval-repeats', 'eval-run-btn', 'eval-sig', 'eval-starts', 'eval-status', 'eval-stop-btn', 'eval-table', 'eval-viewer-btn', 'eval-viewer-msg', 'eval-viewer-stop', 'eval-wrap'],
    "miles": ['miles-cycles', 'miles-demos', 'miles-msg', 'miles-progress', 'miles-start', 'miles-stop', 'miles-stride'],
    "train": ['recon-episode', 'recon-fig', 'recon-frame', 'recon-model', 'recon-msg', 'train-data', 'train-epochs', 'train-export', 'train-export-status', 'train-msg', 'train-plots', 'train-run', 'train-seed', 'train-start', 'train-status', 'train-tag', 'train-tick'],
}  # fmt: skip
PANELS = {"browser": browser, "collect": collect, "evaluate": evaluate, "miles": miles,
          "train": train}


def _walk(n):
    yield n
    c = getattr(n, "children", None)
    for x in c if isinstance(c, (list, tuple)) else [c]:
        if hasattr(x, "to_plotly_json"):
            yield from _walk(x)


def test_css_and_template():
    app = build_app(MockBackend(), Session(tempfile.mkdtemp()), Cfg())
    r = app.server.test_client().get("/assets/style.css")
    assert r.status_code == 200 and b"--accent" in r.data
    assert pio.templates.default == "demo"


def test_panels():
    ctx = Ctx(MockBackend(), Session(tempfile.mkdtemp()), Cfg())
    for name, mod in PANELS.items():
        nodes = list(_walk(mod.layout(ctx)))
        ids = sorted(str(n.id) for n in nodes if getattr(n, "id", None) is not None)
        assert ids == IDS[name], name
        classes = {str(getattr(n, "className", None)).split()[0] for n in nodes}
        assert "card" in classes, name
        assert name != "collect" or "grid-2" in classes
    panel_ids = {i for k, v in IDS.items() if k != "app" for i in v}
    nodes = _walk(build_app(MockBackend(), Session(tempfile.mkdtemp()), Cfg()).layout)
    ids = sorted(str(n.id) for n in nodes if getattr(n, "id", None) is not None)
    assert [i for i in ids if i not in panel_ids] == IDS["app"], ids


if __name__ == "__main__":
    for f in (test_css_and_template, test_panels):
        f()
        print("ok", f.__name__)
