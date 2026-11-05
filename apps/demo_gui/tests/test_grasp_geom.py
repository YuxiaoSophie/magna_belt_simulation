"""Side-view grasp geometry checks, CPU-only: the precomputed JSON, figure, drag -> slide / depth /
pivot.

uv run --frozen python apps/demo_gui/tests/test_grasp_geom.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from apps.demo_gui import grasp2d
from apps.demo_gui.app import build_app  # noqa: F401  (the app must not pull in the sim stack)

ARMS = ("franka", "ur")


def test_json():
    g = grasp2d.load_geom()
    src = g["source"]
    for k in ("scene", "snapshot"):  # stale geometry -> rerun backend/make_silhouettes.py
        assert hashlib.sha256((REPO / src[k]).read_bytes()).hexdigest() == src["sha256"][k], k
    gv00 = json.loads((REPO / src["probed"]).read_text())["variants"][0]["measured"]["absolute"]
    for arm in ARMS:
        a = g[arm]
        for k in ("silhouette_A", "holder_A"):
            assert sum(len(r) for r in a[k]) >= 20, (arm, k)
        assert a["belt_r_mm"] == 3.3 and len(a["belt_A"]) > 20
        assert a["rest_body"] == gv00[arm]["nearest_body"], arm
        assert 0.0 <= a["pivot_axis_deg"] < 90.0 and abs(math.hypot(*a["pivot_axis"]) - 1) < 1e-4
        for knob in ("slide_mm", "pivot_deg"):
            lo, hi = a["limits"][knob]
            assert math.isfinite(lo) and math.isfinite(hi) and lo < 0 < hi, (arm, knob)
            print(f"  {arm} {knob}: limits {lo:+g}..{hi:+g}")
        d0, d1 = a["limits"]["depth_mm"]
        assert -15.0 <= d0 < 0 <= d1 <= 15.0 and d0 % 0.5 == 0 == d1 % 0.5, (arm, d0, d1)
        assert abs(math.hypot(*a["approach_A"]) - 1) < 1e-4 and a["approach_A"][1] > 0.9
        for piv in range(a["limits"]["pivot_deg"][0], a["limits"]["pivot_deg"][1] + 1):
            assert abs(grasp2d.axis(arm, piv)[1]) > 0.3, (arm, piv)
        print(f"  {arm} depth_mm: limits {d0:+g}..{d1:+g}, approach {a['approach_deg']:+.1f} deg "
              f"off vertical, rule {a['depth_rule_mm']}")
        s0, s1 = a["limits"]["slide_mm"]
        assert a["cutout_mm"][0] < s0 < 0 < s1 < a["cutout_mm"][1]
        assert "silhouette_B" not in a and "roll_deg" not in a["limits"]


def test_holder_same_rule_both_arms():
    """The drawn holder: the same part rule for both arms (not each gripper's finger width), the
    same layer stack, and both views from outside the holder (n of each points at the other)."""
    g = grasp2d.load_geom()
    f, u = (g[arm]["holder_parts"] for arm in ARMS)
    assert len(f) == len(u) == 7, (f, u)
    assert sorted(round(z1 - z0, 1) for _, z0, z1 in f) == sorted(
        round(z1 - z0, 1) for _, z0, z1 in u)
    for arm in ARMS:
        ys = [y for r in g[arm]["holder_A"] for _, y in r]
        lo = min(z0 for _, z0, _ in g[arm]["holder_parts"])
        hi = max(z1 for _, _, z1 in g[arm]["holder_parts"])
        assert abs(min(ys) - lo) < 0.5 and abs(max(ys) - hi) < 0.5, arm  # all layers drawn
    pf, pu = (np.array(g[arm]["grasp_point"]) for arm in ARMS)
    assert (pu - pf) @ g["franka"]["n"] > 0 and (pf - pu) @ g["ur"]["n"] > 0


def test_feasible_table():
    """3D holder-clearance table: axes, the nominal node vs its clearance, the stop rule (enabled
    only if both nominal grasps clear the margin), deeper = infeasible, one region per pivot."""
    g = grasp2d.load_geom()
    for arm in ARMS:
        f, m = g[arm]["feasible"], grasp2d.unpack(arm)
        assert f["enabled"] is True and f["order"] == "slide,pivot,depth", arm
        assert f["pivot"] == [-20.0, 1.0, 41] and f["depth"] == [-15.0, 0.5, 61], arm
        s_lo, s_step, ns = f["slide"]
        c0, c1 = g[arm]["cutout_mm"]
        assert s_step == 0.5 and s_lo <= c0 - 5 and s_lo + s_step * (ns - 1) >= c1 + 5, arm
        assert m.shape == (ns, 41, 61) and int(m.sum()) == f["n_feasible"], arm
        assert len(base64.b64decode(f["bits"])) == math.ceil(m.size / 8)
        k0 = round(-s_lo / s_step)
        half = g["source"]["method"]["collision"]["slide_half_step_mm"]
        assert m[k0, 20, 30] == (f["nominal_clearance_mm"] >= f["margin_mm"] + half), arm
        dp = g[arm]["default_pose"]  # nominal if feasible, else the depth nearest 0 at (0, 0)
        ok = np.flatnonzero(grasp2d.feasible(arm)[k0, 20])
        assert dp["slide_mm"] == dp["pivot_deg"] == 0.0 and grasp2d.is_feasible(arm, 0, 0, dp[
            "depth_mm"]) and dp["depth_mm"] == -15.0 + 0.5 * ok[np.argmin(np.abs(ok - 30))], arm
        assert (dp["depth_mm"] == 0.0) == bool(m[k0, 20, 30]), arm
        ok = np.flatnonzero(m[k0, 20])
        deepest = -15.0 + 0.5 * ok.min()
        if deepest - 10.0 >= -15.0:
            assert not m[k0, 20, round((deepest - 10.0 + 15.0) / 0.5)], (arm, deepest)
        comps = [label(m[:, j, :])[1] for j in range(41)]
        assert all(c <= 1 for c in comps), (arm, comps)
        print(f"  {arm}: nominal clearance {f['nominal_clearance_mm']:+.2f} mm "
              f"({f['min_clearance_at_nominal_part']}), {f['n_feasible']}/{m.size} feasible, "
              f"deepest at slide 0 / pivot 0 {deepest:+g} mm, default {dp}")


def test_regenerate_byte_identical():
    """Rerun the CPU precompute (CUDA hidden): the committed JSON is reproduced byte for byte."""
    with tempfile.TemporaryDirectory(prefix="demo_gui_geom_") as d:
        out = Path(d) / "grasp_geom.json"
        t0 = time.monotonic()
        subprocess.run([sys.executable, str(REPO / "apps/demo_gui/backend/make_silhouettes.py"),
                        "--device", "cpu", "--out", str(out)], cwd=REPO, check=True,
                       capture_output=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        print(f"  make_silhouettes.py: {time.monotonic() - t0:.1f} s")
        assert out.read_bytes() == grasp2d.GEOM.read_bytes()


def _spot(n: int = 10) -> None:
    """Independent recheck (CUDA-hidden subprocess, sim stack): per arm ``n`` random + ``n``
    boundary feasible nodes, clearance at the exact ``variant_waypoints`` pick by footprint EDT
    (0.25 mm) + z gap per holder prism, minus the sample radius and half a pixel diagonal."""
    sys.path.insert(0, str(REPO / "apps/demo_gui/backend"))
    import make_silhouettes as ms
    import newton
    from scipy.ndimage import binary_erosion, distance_transform_edt

    import round_belt_task
    from round_belt_task.arm_kinematics import model_pose_franka_tip, model_pose_ur_tracking

    g, rng, res = grasp2d.load_geom(), np.random.default_rng(0), 0.25
    snap = np.load(REPO / g["source"]["snapshot"], allow_pickle=True)
    b = round_belt_task.make_builder()
    info = round_belt_task.build_scene(b, REPO / g["source"]["scene"])
    labels, held = list(b.body_label), np.asarray(snap["body_q"], dtype=np.float64)
    tips = {"franka": model_pose_franka_tip(held, labels.index("panda_hand/finger_tip")),
            "ur": model_pose_ur_tracking(held, labels.index("/ur10/wrist_3_link"))}
    mgv = ms.load_mgv()
    nominal = ms.load_pre_mpc_segment(REPO / g["source"]["params"], first="pick", last="pick")
    rest = np.array([[float(v) for v in b.body_q[i]][:3] for i in info.belt_bodies])
    frames = {}
    for arm in ARMS:
        pt = np.asarray(getattr(nominal[0], f"{arm}_pos"), dtype=np.float64)
        frames[arm] = {"point": pt, "tangent": mgv.rest_tangent(rest, pt)[1]}
    holder = [s for s in range(b.shape_count) if b.shape_body[s] < 0
              and b.shape_label[s].startswith("belt_chain_holder/")
              and not b.shape_flags[s] & newton.ShapeFlags.VISIBLE]
    gt, gn = (np.arange(round((hi - lo) / res) + 1) * res + lo for lo, hi in (ms.WIN["t"],
                                                                               ms.WIN["n"]))
    for arm in ARMS:
        p, t = frames[arm]["point"], frames[arm]["tangent"]
        L = np.stack([t, np.cross([0.0, 0.0, 1.0], t), [0.0, 0.0, 1.0]])

        def loc(x, p=p, L=L):
            return (np.asarray(x) - p) @ L.T * 1e3

        bodies = {k for k, lab in enumerate(labels) if lab.rsplit("/", 1)[-1] in ms.FINGERS[arm]
                  or lab in ms.FINGERS[arm]}
        pts = ms.surface_samples(np.concatenate([
            ms.shape_tris(b, s, np.linalg.inv(tips[arm]) @ ms.pose4(held[b.shape_body[s]]))
            for s in range(b.shape_count) if b.shape_body[s] in bodies
            and b.shape_flags[s] & newton.ShapeFlags.COLLIDE_SHAPES]), ms.SAMPLE_MM * 1e-3)
        prisms = []
        for s in holder:
            q = loc(ms.shape_tris(b, s, np.eye(4)))
            fp = ms.raster(q[..., :2], [gt[0], gn[0]], (len(gt), len(gn)), res)  # rows = n
            prisms.append((distance_transform_edt(~fp) * res, q[..., 2].min(), q[..., 2].max()))
        m = grasp2d.feasible(arm)
        edge = np.argwhere(m & ~binary_erosion(m, border_value=1))
        nodes = np.argwhere(m)
        picks = [*nodes[rng.choice(len(nodes), n)], *edge[rng.choice(len(edge), n)]]
        worst = np.inf
        for k, j, i in picks:
            ax = grasp2d._axes(arm)
            kw = dict(zip(grasp2d.KNOBS, (ax[0][0] + 0.5 * k, -20.0 + j, -15.0 + 0.5 * i)))
            v = mgv.Variant("s", **{f"{arm}_{knob}": val for knob, val in kw.items()})
            w = next(w for w in mgv.variant_waypoints(nominal, frames, v) if w.label == "pick")
            X = w.franka_mat() if arm == "franka" else w.ur_mat()
            q = loc(pts @ X[:3, :3].T + X[:3, 3])
            it = np.clip(np.rint((q[:, 0] - gt[0]) / res).astype(int), 0, len(gt) - 1)
            iN = np.clip(np.rint((q[:, 1] - gn[0]) / res).astype(int), 0, len(gn) - 1)
            c = min(float(np.hypot(e[iN, it], np.maximum(0, np.maximum(z0 - q[:, 2], q[:, 2] - z1)))
                          .min()) for e, z0, z1 in prisms) - ms.SAMPLE_R - res / math.sqrt(2)
            worst = min(worst, c)
            assert c >= g[arm]["feasible"]["margin_mm"], (arm, kw, c)
        print(f"  {arm}: {len(picks)} feasible nodes ({n} on the boundary), min clearance "
              f"{worst:.2f} mm (margin {g[arm]['feasible']['margin_mm']})")


def test_spot_clearance():
    r = subprocess.run([sys.executable, __file__, "--spot"], check=False, cwd=REPO,
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    print("\n".join(ln for ln in r.stdout.splitlines() if "min clearance" in ln))
    assert r.returncode == 0, r.stderr[-2000:]


def test_no_sim_stack_at_runtime():
    assert not {"warp", "newton", "trimesh"} & set(sys.modules)


def test_figure():
    for arm in ARMS:
        f = grasp2d.figure(arm, 3.0, 5.0, -0.5)
        assert len(f.data) >= 3 and len(f.layout.shapes) == 2
        assert grasp2d.BELT in {t.fillcolor for t in f.data}
        assert f.layout.yaxis.scaleanchor == "x" and f.layout.dragmode is False
        assert f.layout.title.text == f"{grasp2d.NAMES[arm]} grasp — side view along the belt"
        assert f.layout.shapes[1].type == "circle" and f.layout.yaxis.range[0] <= -45
        # +pivot = clockwise about the slid grasp point (finger tops toward +t)
        b = grasp2d.path_mean(f.layout.shapes[0].path) - [3.0, 0.0] + 0.5 * grasp2d.axis(arm, 5.0)
        b0 = grasp2d._base(arm, 0.0)
        turn = math.degrees(math.atan2(b[1], b[0]) - math.atan2(b0[1], b0[0]))
        assert abs(turn + 5.0) < 0.05, turn
        rays = [t for t in f.data if (t.name or "").startswith("pivot limit")]
        assert len(rays) == 2 and rays[0].x[0] == 3.0 and rays[0].y[0] == 0.0
        assert not [t for t in f.data if "depth limit" in (t.name or "")
                    or "slide limit" in (t.name or "")]
        # the region at pivot 0 = the table's feasible (slide, depth) slice, mapped back
        dp = grasp2d.default_pose(arm)
        f = grasp2d.figure(arm, *(dp[k] for k in grasp2d.KNOBS))
        reg = next(t for t in f.data if t.name == "collision-free grasp point region")
        xy = np.array([(x, y) for x, y in zip(reg.x, reg.y) if x is not None])
        a = grasp2d.axis(arm, 0.0)
        depth = xy[:, 1] / a[1]
        slide = xy[:, 0] - depth * a[0]
        ks = np.flatnonzero(grasp2d.feasible(arm)[:, 20, :].any(1))
        s_lo = grasp2d.load_geom()[arm]["feasible"]["slide"][0]
        assert abs(slide.min() - (s_lo + 0.5 * ks.min() - 0.25)) < 0.02, (arm, slide.min())
        assert abs(slide.max() - (s_lo + 0.5 * ks.max() + 0.25)) < 0.02, (arm, slide.max())
        lo, hi = grasp2d.pivot_interval(arm, *(dp[k] for k in grasp2d.KNOBS))
        assert [r.name for r in f.data if (r.name or "").startswith("pivot limit")] == [
            f"pivot limit {lo:+g} deg", f"pivot limit {hi:+g} deg"]
        gp = next(t for t in f.data if t.name == "grasp point")
        assert abs(gp.y[0] - dp["depth_mm"] * a[1]) < 1e-9


def test_relayout_mapping_and_clamp():
    for arm in ARMS:
        g = grasp2d.load_geom()[arm]

        def drag(dx, piv=0.0, depth=0.0, sil=g["silhouette_A"], arm=arm):
            return {"shapes[0].path": grasp2d.svg_path(grasp2d.place(sil, dx, piv, depth, arm))}

        def handle(deg, slide=0.0, keys="x0 x1 y0 y1"):
            a = math.radians(grasp2d.HANDLE_DEG - deg)
            c = {"x": slide + 35 * math.cos(a), "y": 35 * math.sin(a)}
            k = grasp2d.HANDLE_MM
            return {f"shapes[1].{q}": c[q[0]] + (k if q[1] == "1" else -k) for q in keys.split()}

        for dx, piv, depth in ((-2.5, 0.0, 0.0), (4.0, 9.0, 0.0), (1.3, -20.0, -3.7)):
            got = grasp2d.drag_from_relayout(arm, drag(dx, piv, depth), piv)
            assert abs(got[0] - dx) < 0.02 and abs(got[1] - depth) < 0.02, (arm, got)
        assert grasp2d.drag_from_relayout(arm, {"autosize": True}) is None
        assert grasp2d.pivot_from_relayout(arm, {"autosize": True}) is None
        assert grasp2d.pivot_from_relayout(arm, {"shapes[2].path": "M 0,0 L 1,1 Z"}) is None
        for keys in ("x0 x1 y0 y1", "x0 y0", "x1 y1", "x0 y1"):
            assert abs(grasp2d.pivot_from_relayout(arm, handle(-7.0, 5.0, keys), 5.0) + 7.0) < 1e-6
        assert grasp2d.pivot_from_relayout(arm, handle(-2.0, keys="x0")) is None
        d = grasp2d.default_pose(arm)["depth_mm"]
        assert grasp2d.clamp(arm, -1.24, 0.4, d) == (-1.2, 0.0, d, {})  # feasible: kept
        s, p, dd, at = grasp2d.clamp(arm, 1e3, 0.0, d)
        ks = np.flatnonzero(grasp2d.feasible(arm)[:, 20].any(1))
        assert p == 0.0 and at["slide_mm"] == 1 and "pivot_deg" not in at
        assert s == g["feasible"]["slide"][0] + 0.5 * ks.max() and grasp2d.is_feasible(arm, s, p, dd)
        s, p, dd, at = grasp2d.clamp(arm, 0.0, 0.0, -1e3)  # a finger drag keeps the pivot
        assert p == 0.0 and at.get("depth_mm") == 0 and "pivot_deg" not in at
        assert dd == -15.0 + 0.5 * np.flatnonzero(grasp2d.feasible(arm)[:, 20].any(0)).min()
        col = np.flatnonzero(grasp2d.feasible(arm)[round(-g["feasible"]["slide"][0] / 0.5), :,
                                                   round((d + 15.0) / 0.5)])
        assert grasp2d.clamp(arm, 0.0, 1e3, d, turn=True) == (  # a turn keeps slide / depth
            0.0, -20.0 + col.max(), d, {"pivot_deg": 1})
        assert grasp2d.clamp(arm, 0.0, 20.0, 1e3)[3]["depth_mm"] == 1  # belt-in-jaw rule
        assert not grasp2d.is_feasible(arm, 0.0, 0.0, g["limits"]["depth_mm"][1] + 0.5)
        rng = np.random.default_rng(0)
        for req in rng.uniform([-40, -40, -20], [40, 40, 20], (200, 3)):
            for turn in (False, True):
                assert grasp2d.is_feasible(arm, *grasp2d.clamp(arm, *req, turn)[:3]), (arm, req)


if __name__ == "__main__":
    if "--spot" in sys.argv:
        _spot()
        raise SystemExit(0)
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        t0 = time.monotonic()
        fn()
        print(f"ok {name} ({time.monotonic() - t0:.1f} s)")
    print(f"{len(tests)} passed")
