#!/usr/bin/env python3
"""Checks of a scene_rv (V61e) deploy and its goal rule (V12-133).

S1 encoder: magna's ``SceneRvEncoder`` == the exporter's reference z (relative 1e-5 of max |z|);
   ``encode == scene_part + robot_part``.
S2 goal rule: goal = scene_part(goal frame) + robot_part(CURRENT rv). For each reference frame,
   the whitened distance to that goal under EE +-10 mm / +-10 deg (both arms) and grip changes
   of the current rv must not change (float rounding only, <= 1e-9).
S3 node: ``MppiControllerNode._rv_goals`` sets z_traj / z_wp / every stage goal to the stored
   scene part + the current robot part.
S4 (``--waypoints``): the waypoints npz is rv_free and its z_wp are the goal demo's scene parts.

    uv run --frozen python scripts/checks/check_scene_rv.py --deploy <deploy dir> [--waypoints npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from task_common import latent_encoder as le


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def _rotz(q, deg: float) -> np.ndarray:
    h = np.radians(deg) / 2.0
    w1, z1 = np.cos(h), np.sin(h)
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deploy", type=Path, required=True, help="deploy dir")
    ap.add_argument("--waypoints", type=Path, default=None)
    a = ap.parse_args()
    enc = le.LatentEncoder.load(a.deploy / "deploy.npz")
    r = dict(np.load(a.deploy / "reference_vectors.npz", allow_pickle=True))
    with np.load(a.deploy / "deploy.npz") as d:
        zstd = np.asarray(d["z_std"], float)
    ok, K = True, len(r["z"])
    belt = getattr(enc, "rv_cloud", "scene") == "belt"  # V61f: the point branch reads belt_seg
    cl, ck = ("pc_raw", "pcd_belt_seg") if belt else ("scene_raw", "pcd")
    try:
        _require(isinstance(enc, le.SceneRvEncoder), "not a scene_rv deploy")
        z = np.stack([enc.encode(r["pc_raw"][k], r["prop"][k], grip=r["grip"][k], scene=r["scene_raw"][k])
                      for k in range(K)])
        err = float(np.abs(z - r["z"]).max())
        _require(err <= 1e-5 * max(1.0, float(np.abs(r["z"]).max())), f"z vs exporter {err:.1e}")
        parts = enc.scene_part(r[cl][0]) + enc.robot_part(r["prop"][0], r["grip"][0])
        _require(np.array_equal(parts, z[0]), "encode != scene_part + robot_part")
        print(f"[PASS] S1 K={K}: z vs exporter {err:.1e}; encode == scene_part + robot_part", flush=True)
    except AssertionError as e:
        ok = False
        print(f"[FAIL] S1 {e}", flush=True)
    try:
        zs_goal = enc.scene_part(r[cl][-1])
        worst = 0.0
        for k in range(K):
            zs = enc.scene_part(r[cl][k])
            d0 = None
            for dp, deg, dg in ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (-0.01, 0.0, 0.0), (0.0, 10.0, 0.0),
                                (0.0, -10.0, 0.0), (0.0, 0.0, 0.05)):
                p = np.asarray(r["prop"][k], np.float64).copy()
                for sl in (slice(26, 33), slice(33, 40)):
                    p[sl][:3] += dp
                    p[sl][3:] = _rotz(p[sl][3:], deg)
                g = np.asarray(r["grip"][k], np.float64).copy()
                g[0] += dg
                zr = enc.robot_part(p, g)
                d = float(np.linalg.norm(((zs + zr) - (zs_goal + zr)) / zstd))
                d0 = d if d0 is None else d0
                worst = max(worst, abs(d - d0))
        _require(worst <= 1e-9, f"goal distance moves with rv: {worst:.2e}")
        print(f"[PASS] S2 EE +-10 mm / +-10 deg and gap changes of the current rv: goal distance change max "
              f"{worst:.1e} (exact 0 up to float rounding)", flush=True)
    except AssertionError as e:
        ok = False
        print(f"[FAIL] S2 {e}", flush=True)
    try:
        import mppi_controller_node as node

        rng = np.random.default_rng(0)
        base = (rng.normal(size=(20, enc.latent_dim)), rng.normal(size=(4, enc.latent_dim)))
        wp = SimpleNamespace(z_traj=base[0].copy(), z_wp=base[1].copy())
        stages = [node.Stage(base[1][k].copy(), 1.0, 1.0, k) for k in range(4)]
        fake = SimpleNamespace(rv_enc=enc, rv_goals=base, wp=wp, stages=stages)
        prop = np.r_[r["prop"][0], r["grip"][0]]
        node.MppiControllerNode._rv_goals(fake, prop)
        zr = enc.robot_part(r["prop"][0], r["grip"][0])
        _require(np.array_equal(wp.z_traj, base[0] + zr) and np.array_equal(wp.z_wp, base[1] + zr)
                 and all(np.array_equal(s.goal, base[1][s.index] + zr) for s in stages), "node goals")
        print("[PASS] S3 node _rv_goals: z_traj, z_wp and stage goals = scene part + current robot part", flush=True)
    except AssertionError as e:
        ok = False
        print(f"[FAIL] S3 {e}", flush=True)
    if a.waypoints is not None:
        try:
            with np.load(a.waypoints, allow_pickle=True) as w:
                _require(bool(w.get("rv_free", False)), "waypoints not rv_free")
                ep = Path(str(w["demo_episode"]))
                frames, z_wp = w["frames"], w["z_wp"]
            with np.load(ep, allow_pickle=True) as d:
                zs = np.stack([enc.scene_part(d[ck][int(f)]) for f in frames])
            err = float(np.abs(zs - z_wp).max())
            _require(err <= 1e-9, f"z_wp != scene parts ({err:.1e})")
            print(f"[PASS] S4 waypoints rv_free; z_wp == the goal demo's scene parts ({err:.1e})", flush=True)
        except AssertionError as e:
            ok = False
            print(f"[FAIL] S4 {e}", flush=True)
    print("ALL SCENE-RV CHECKS PASSED" if ok else "SCENE-RV CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
