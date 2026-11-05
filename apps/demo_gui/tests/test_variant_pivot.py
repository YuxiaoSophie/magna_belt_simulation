"""``make_grasp_variants``: ``Variant.franka_pivot_deg`` is opt-in (defaults byte-identical to a
reference copy of the script), the pivot turns about the slid grasp point, and the canvas turns the
fingers the way ``variant_waypoints`` does. CPU only (magna params yaml, no sim).

uv run --frozen python apps/demo_gui/tests/test_variant_pivot.py [reference make_grasp_variants.py]
(default reference: the copy at 7177c1d, before pivot/depth)
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from apps.demo_gui import grasp2d

SCRIPT = REPO / "scripts/lcs/make_grasp_variants.py"
REF_COMMIT = "7177c1d"  # last commit before pivot/depth: HEAD now has them
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_pivot_")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)  # @dataclass looks it up
    spec.loader.exec_module(mod)
    return mod


def _reference() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    p = Path(_TMP.name) / "make_grasp_variants_ref.py"
    p.write_text(subprocess.run(["git", "show", f"{REF_COMMIT}:scripts/lcs/make_grasp_variants.py"],
                                cwd=REPO, check=True, capture_output=True, text=True).stdout)
    return p


NEW = _load("mgv_new", SCRIPT)
OLD = _load("mgv_old", _reference())
NOMINAL = NEW.load_pre_mpc_segment(NEW.MAGNA_PARAMS_SIM_YAML, first=NEW.PICK_FIRST,
                                   last=NEW.PICK_LAST)
GEOM = grasp2d.load_geom()
_PICK = next(w for w in NOMINAL if w.label == "pick")
FRAMES = {arm: {"point": np.asarray(getattr(_PICK, f"{arm}_pos"), float),  # as phase_a
                "tangent": np.array(GEOM[arm]["t"]) / np.linalg.norm(GEOM[arm]["t"])}
          for arm in ("franka", "ur")}
GV06 = next(v for v in json.loads((REPO / GEOM["source"]["probed"]).read_text())["variants"]
            if v["id"] == "gv_06")["commanded"]


def _bytes(w) -> list:
    return [np.asarray(x, dtype=np.float64).tobytes() if isinstance(x, np.ndarray | list)
            else x for x in vars(w).values()]


def _pick(wps, arm: str) -> np.ndarray:
    w = next(w for w in wps if w.label == "pick")
    return w.franka_mat() if arm == "franka" else w.ur_mat()


def test_defaults_byte_identical():
    for kw in ({}, {"ur_pivot_deg": 10.0}, GV06):
        a, b = NEW.Variant("x", **kw), OLD.Variant("x", **kw)
        assert json.dumps(a.commanded()) == json.dumps(b.commanded()), kw
        new = NEW.variant_waypoints(NOMINAL, FRAMES, a)
        old = OLD.variant_waypoints(NOMINAL, FRAMES, b)
        assert [_bytes(w) for w in new] == [_bytes(w) for w in old], kw
    assert "franka_pivot_deg" not in NEW.Variant("x", franka_pivot_deg=0.0).commanded()
    assert NEW.Variant("x", franka_pivot_deg=-4).commanded()["franka_pivot_deg"] == -4.0
    assert NEW.KNOBS == OLD.KNOBS


def test_franka_pivot_only_moves_the_franka_pick():
    new = NEW.variant_waypoints(NOMINAL, FRAMES, NEW.Variant("f", franka_pivot_deg=10.0))
    nom = NEW.variant_waypoints(NOMINAL, FRAMES, NEW.Variant("n"))
    for w, n in zip(new, nom):
        same_f = np.array_equal(w.franka_mat(), n.franka_mat())
        assert same_f == (w.label != "pick"), w.label
        assert w.ur_pos is None or np.array_equal(w.ur_mat(), n.ur_mat()), w.label


def test_combined_slide_pivot_stays_on_the_belt():
    """Slide 15 mm + pivot 10 deg: the grasp point stays at p + s t (the reference lifts it)."""
    p, t = FRAMES["ur"]["point"], FRAMES["ur"]["tangent"]
    target = p + 15e-3 * t
    kw = {"ur_slide_mm": 15.0, "ur_pivot_deg": 10.0}
    new = _pick(NEW.variant_waypoints(NOMINAL, FRAMES, NEW.Variant("c", **kw)), "ur")[:3, 3]
    old = _pick(OLD.variant_waypoints(NOMINAL, FRAMES, OLD.Variant("c", **kw)), "ur")[:3, 3]
    d_new, d_old = (float(np.linalg.norm(x - target)) * 1e3 for x in (new, old))
    print(f"  ur slide 15 mm + pivot 10 deg: grasp point off p + s*t by {d_new:.4f} mm (now), "
          f"{d_old:.3f} mm (reference; z {1e3 * (old - target)[2]:+.3f} mm)")
    assert d_new < 1e-6 and abs(d_old - 30.0 * math.sin(math.radians(5.0))) < 0.01


def test_canvas_turns_like_the_sim():
    """A finger point 50 mm up the approach axis, moved by variant_waypoints vs drawn by grasp2d."""
    for arm in ("franka", "ur"):
        p, t = FRAMES[arm]["point"], FRAMES[arm]["tangent"]
        L = np.stack([t, [0.0, 0.0, 1.0]])
        v = NEW.Variant("s", **{f"{arm}_slide_mm": 5.0, f"{arm}_pivot_deg": 10.0})
        X0 = _pick(NOMINAL, arm)
        T = _pick(NEW.variant_waypoints(NOMINAL, FRAMES, v), arm) @ np.linalg.inv(X0)
        q = X0 @ [0.0, 0.0, 0.05, 1.0]
        sim = (T @ q)[:3]
        canvas = grasp2d.place([((q[:3] - p) @ L.T * 1e3)[None]], 5.0, 10.0)[0][0]
        err = float(np.linalg.norm((sim - p) @ L.T * 1e3 - canvas))
        lean = canvas[0] - 5.0 - (q[:3] - p) @ t * 1e3
        print(f"  {arm}: +10 deg leans the point {lean:+.1f} mm along t; "
              f"sim vs canvas {err:.2f} mm")
        assert lean > 0 and err < 1.0, (arm, lean, err)
        assert np.linalg.norm((T @ [*p, 1.0])[:3] - (p + 5e-3 * t)) < 1e-9  # tip stays on the belt


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
