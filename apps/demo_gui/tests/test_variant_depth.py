"""``make_grasp_variants``: ``Variant.franka_depth_mm`` is opt-in (defaults byte-identical to a
reference copy of the script), moves the Franka ``pick`` along its post-pose ``finger_tip`` +z, and
the canvas moves the fingers the way ``variant_waypoints`` does. CPU only (magna params yaml).

uv run --frozen python apps/demo_gui/tests/test_variant_depth.py [reference make_grasp_variants.py]
(default reference: the copy at 7177c1d, before pivot/depth)
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import json
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
_TMP = tempfile.TemporaryDirectory(prefix="demo_gui_depth_")


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


NEW = _load("mgv_new_depth", SCRIPT)
OLD = _load("mgv_old_depth", _reference())
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


def _wps(mod, **kw) -> list:
    return mod.variant_waypoints(NOMINAL, FRAMES, mod.Variant("x", **kw))


def _pick(wps) -> object:
    return next(w for w in wps if w.label == "pick")


def test_defaults_byte_identical():
    cases = [{}, {"ur_depth_mm": 6.0}, {"ur_pivot_deg": 10.0}, GV06]
    # slide + pivot changed in PKG-11 (pivot about the slid point): only vs a post-PKG-11 copy
    if "franka_pivot_deg" in OLD.Variant.__dataclass_fields__:
        cases.append({"ur_slide_mm": 15.0, "ur_pivot_deg": 10.0, "ur_depth_mm": 6.0})
    else:
        print("  combined ur slide+pivot+depth: skipped (reference predates the PKG-11 pivot)")
    for kw in cases:
        a, b = NEW.Variant("x", **kw), OLD.Variant("x", **kw)
        assert json.dumps(a.commanded()) == json.dumps(b.commanded()), kw
        assert [_bytes(w) for w in _wps(NEW, **kw)] == [_bytes(w) for w in _wps(OLD, **kw)], kw
    print(f"  byte-identical: {len(cases)} cases")
    assert "franka_depth_mm" not in NEW.Variant("x", franka_depth_mm=0.0).commanded()
    assert NEW.Variant("x", franka_depth_mm=5).commanded()["franka_depth_mm"] == 5.0
    assert NEW.KNOBS == OLD.KNOBS


def test_franka_depth_moves_only_the_franka_pick():
    assert _pick(NOMINAL).franka_mat()[2, 2] > 0.9  # finger_tip +z points up: + = shallower
    new, nom = _wps(NEW, franka_depth_mm=5.0), _wps(NEW)
    for w, n in zip(new, nom):
        assert np.array_equal(w.franka_mat()[:3, :3], n.franka_mat()[:3, :3]), w.label
        assert w.ur_pos is None or np.array_equal(w.ur_mat(), n.ur_mat()), w.label
        d = np.asarray(w.franka_pos, float) - np.asarray(n.franka_pos, float)
        if w.label == "pick":
            assert np.linalg.norm(d - 5e-3 * w.franka_mat()[:3, 2]) < 1e-12, d
        else:
            assert not d.any(), w.label


def test_canvas_moves_like_the_sim():
    """Slide 5 + pivot 10 + depth -4: a finger point 50 mm up the approach axis, moved by
    variant_waypoints vs drawn by grasp2d."""
    for arm in ("franka", "ur"):
        p, t = FRAMES[arm]["point"], FRAMES[arm]["tangent"]
        L = np.stack([t, [0.0, 0.0, 1.0]])
        mat = "franka_mat" if arm == "franka" else "ur_mat"
        kw = {f"{arm}_slide_mm": 5.0, f"{arm}_pivot_deg": 10.0, f"{arm}_depth_mm": -4.0}
        X0 = getattr(_pick(NOMINAL), mat)()
        T = getattr(_pick(_wps(NEW, **kw)), mat)() @ np.linalg.inv(X0)
        q = X0 @ [0.0, 0.0, 0.05, 1.0]
        sim = (T @ q)[:3]
        canvas = grasp2d.place([((q[:3] - p) @ L.T * 1e3)[None]], 5.0, 10.0, -4.0, arm)[0][0]
        err = float(np.linalg.norm((sim - p) @ L.T * 1e3 - canvas))
        print(f"  {arm}: sim vs canvas at 50 mm up the fingers {err:.2f} mm")
        assert err < 0.5, (arm, err)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
