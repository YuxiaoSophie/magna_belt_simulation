#!/usr/bin/env python3
"""Precompute the side-view grasp canvas geometry -> ``apps/demo_gui/assets/grasp_geom.json``.

Per arm, in ``make_grasp_variants``' Variant frame (origin = the ``pick`` waypoint point, ``t`` =
rest-belt tangent at the nearest rest body, ``n = z x t``, ``z`` up; mm): the finger silhouette of
the HELD snapshot (finger bodies carried from the held tip frame onto the ``pick`` pose) in the
along-belt view (``t``-``z``), holder outline, rest belt, holder cutout, slide / pivot / depth
limits, and ``approach_A`` = the pick frame's tip-to-wrist ``+z`` projected on the view (unit).

Holder = its collision parts (z-prisms); the obstacles are the parts inside the fingers' ``n``
range below the holder top. A slide / pivot (Variant's turn about ``approach x t``, ~``n``; + =
finger tops toward +t = clockwise in the view) is feasible while no finger pixel sits deeper in
any part than at the nominal grasp + ``TOL_MM`` (the nominal UR pick already touches the bottom
plate). Depth (``*_depth_mm``, + = shallower, along ``approach_A``): deeper by the same rule;
shallower while the fingertip end stays below the belt disc about the grasp point by ``TOL_MM``.
All limits at the nominal slide / pivot. CPU only, deterministic.

``feasible`` (per arm): the joint slide x pivot x depth grid where the finger COLLISION meshes, at the
exact ``variant_waypoints`` pick pose, keep ``--margin-mm`` from every holder collision part in 3D
(surface samples <= 0.5 mm apart, 0.5 mm occupancy grid + EDT, minus half a voxel diagonal and
the sample radius and half a slide step, so 0.1 mm slides between nodes are covered).
``default_pose``: nominal if feasible, else the feasible depth nearest 0 at slide 0 / pivot 0
(within the belt-in-jaw shallower limit).

Run:
    uv run --frozen python apps/demo_gui/backend/make_silhouettes.py
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only scene build: never touch the GPU

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import newton

import round_belt_task
from round_belt_task.arm_kinematics import (
    model_pose_franka_tip,
    model_pose_ur_tracking,
    quat_xyzw_to_mat3,
)
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment

OUT = REPO / "apps/demo_gui/assets/grasp_geom.json"
SNAPSHOT = REPO / "data/lcs/start_states/pre_place_1_osc.npz"
PROBED = REPO / "data/lcs/start_states/grasp_variants/set2/index.json"
UR_DEPTH = REPO / "data/lcs/start_states/ur_depth/index.json"  # optional "verified" depth ticks
RES = 0.25  # raster mm / pixel
SIMPLIFY_MM = 0.1
MIN_RING_MM2 = 2.0  # drop raster specks / pin holes
TOL_MM = 0.5
SLIDE_STEP, SLIDE_CAP, PIVOT_CAP, DEPTH_CAP = 0.5, 60.0, 45, 15.0
BELT_ARC_MM = 120.0
WIN = {"t": (-90.0, 90.0), "n": (-70.0, 70.0), "z": (-30.0, 30.0)}  # holder windows, local mm
VOX_MM, MARGIN_MM = 0.5, 2.0
VOX_HALF_DIAG = VOX_MM * math.sqrt(3.0) / 2.0
SAMPLE_MM = 0.5  # finger surface sample spacing (sub-triangle edge); any surface point is
SAMPLE_R = SAMPLE_MM / math.sqrt(3.0)  # within this of a sample, so it is subtracted too
PIVOT_GRID, DEPTH_GRID = (-20.0, 1.0, 41), (-15.0, 0.5, 61)  # [lo, step, n]
SLIDE_PAD_MM = 5.0  # slide grid = cutout span +- this
FINGERS = {"franka": ("panda_hand/panda_leftfinger", "panda_hand/panda_rightfinger"),
           "ur": ("left_pad", "right_pad", "left_follower", "right_follower")}


def mat4(R, p) -> np.ndarray:
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, p
    return X


def pose4(row) -> np.ndarray:
    """(px py pz qx qy qz qw) -> 4x4."""
    row = [float(v) for v in row]
    return mat4(quat_xyzw_to_mat3(row[3:7]), row[:3])


def shape_tris(builder, s, X_WB) -> np.ndarray:
    """World triangles (F, 3, 3) of mesh shape ``s`` on a body at ``X_WB``."""
    src = builder.shape_source[s]
    v = np.asarray(src.vertices, dtype=np.float64) * np.asarray(builder.shape_scale[s])
    X = X_WB @ pose4(builder.shape_transform[s])
    return (v @ X[:3, :3].T + X[:3, 3])[np.asarray(src.indices).reshape(-1, 3)]


def raster(polys, lo, size, res: float = RES) -> np.ndarray:
    """Union of 2D polygons (mm) on a grid with pixel centre ``lo + i * res``; bool (rows=y)."""
    img = Image.new("1", (int(size[0]), int(size[1])))
    draw = ImageDraw.Draw(img)
    for p in (np.asarray(polys) - lo) / res:
        draw.polygon([tuple(q) for q in p], fill=1, outline=1)
    return np.asarray(img, dtype=bool)


def rdp(p: np.ndarray, eps: float) -> np.ndarray:
    keep = np.zeros(len(p), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(p) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d, seg = p[j] - p[i], p[i + 1:j] - p[i]
        n = math.hypot(*d)
        dist = (np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / n if n > 1e-9
                else np.hypot(seg[:, 0], seg[:, 1]))
        k = int(np.argmax(dist))
        if dist[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return p[keep]


def outline(mask: np.ndarray, lo) -> list:
    """Closed rings (mm, 0.01 mm) of a bool mask; holes included (draw with even-odd fill)."""
    m = np.pad(mask, 1).astype(np.float64)
    lines = contour_generator(z=m).lines(0.5)
    rings = [rdp(np.asarray(ln) - 1.0, SIMPLIFY_MM / RES) * RES + lo for ln in lines]
    area = [abs(np.cross(r[:-1], r[1:]).sum()) / 2 for r in rings]
    return [np.round(r, 2).tolist() for r, a in zip(rings, area) if a >= MIN_RING_MM2]


def sil_mask(tris2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.floor(tris2d.reshape(-1, 2).min(0) / RES) * RES - 2 * RES
    hi = tris2d.reshape(-1, 2).max(0) + 2 * RES
    return raster(tris2d, lo, np.ceil((hi - lo) / RES) + 1), lo


def grid(axis: str) -> np.ndarray:
    a, b = WIN[axis]
    return np.arange(round((b - a) / RES) + 1) * RES + a


def lookup(field: np.ndarray, gx: np.ndarray, gy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """``field[y, x]`` at the nearest pixel of each point; 0 outside the window."""
    ix = np.rint((pts[:, 0] - gx[0]) / RES).astype(int)
    iy = np.rint((pts[:, 1] - gy[0]) / RES).astype(int)
    ok = (ix >= 0) & (ix < len(gx)) & (iy >= 0) & (iy < len(gy))
    out = np.zeros(len(pts))
    out[ok] = field[iy[ok], ix[ok]]
    return out


def depths(obstacles, gx, gy, pts) -> np.ndarray:
    return np.array([lookup(e, gx, gy, pts).max() for e in obstacles])


def search(feasible, step: float, cap: float) -> list[float]:
    """Largest feasible ``k * step`` on each side, capped at ``cap``."""
    out = []
    for sign in (-1, 1):
        k = 0
        while (k + 1) * step <= cap + 1e-9 and feasible(sign * (k + 1) * step):
            k += 1
        out.append(sign * k * step)
    return out


def rotate(pts: np.ndarray, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return pts @ np.array([[c, s], [-s, c]])


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def surface_samples(tris: np.ndarray, h: float) -> np.ndarray:
    """Unique vertices after longest-edge bisection of every triangle down to edges <= ``h``."""
    while True:
        e = np.linalg.norm(np.roll(tris, -1, axis=1) - tris, axis=2)  # edge k: v_k -> v_k+1
        k = e.argmax(1)
        big = e[np.arange(len(e)), k] > h
        if not big.any():
            return np.unique(np.round(tris.reshape(-1, 3), 9), axis=0)
        t, k, r = tris[big], k[big], np.arange(big.sum())
        v0, v1, v2 = t[r, k], t[r, (k + 1) % 3], t[r, (k + 2) % 3]
        m = (v0 + v1) / 2
        tris = np.concatenate([tris[~big], np.stack([v0, m, v2], 1), np.stack([m, v1, v2], 1)])


def load_mgv():
    """``scripts/lcs/make_grasp_variants.py`` (imported, not edited): the exact pick poses."""
    spec = importlib.util.spec_from_file_location(
        "make_grasp_variants", REPO / "scripts/lcs/make_grasp_variants.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)  # @dataclass looks it up
    spec.loader.exec_module(mod)
    return mod


def feasibility(mgv, nominal, frames, arm, pts, loc, holder, names, cutout, margin
                ) -> tuple[dict, np.ndarray]:
    """Joint slide x pivot x depth table: finger sample points ``pts`` (tip frame, m) at the
    ``variant_waypoints`` pick pose keep >= ``margin`` mm from the holder prisms ``holder``."""
    def pick(slide=0.0, pivot=0.0, depth=0.0):
        v = mgv.Variant("g", **{f"{arm}_slide_mm": slide, f"{arm}_pivot_deg": pivot,
                                f"{arm}_depth_mm": depth})
        w = next(w for w in mgv.variant_waypoints(nominal, frames, v) if w.label == "pick")
        return w.franka_mat() if arm == "franka" else w.ur_mat()

    g = [np.arange(round((b - a) / VOX_MM) + 1) * VOX_MM + a for a, b in WIN.values()]  # t, n, z
    occ = np.zeros([len(x) for x in g], dtype=bool)
    prisms = []
    for tri in holder:
        q = loc(tri)
        fp = raster(q[..., :2], [g[0][0], g[1][0]], (len(g[0]), len(g[1])), VOX_MM).T  # (t, n)
        z0, z1 = float(q[..., 2].min()), float(q[..., 2].max())
        # centres within half a voxel count, like the outlined footprint: conservative
        occ |= fp[:, :, None] & ((g[2] >= z0 - VOX_MM / 2) & (g[2] <= z1 + VOX_MM / 2))
        prisms.append((fp, z0, z1))
    D = distance_transform_edt(~occ, sampling=VOX_MM)
    near = D < margin + VOX_HALF_DIAG + SAMPLE_R + SLIDE_STEP / 2  # a sample here is too close
    nt, nn, nz = near.shape
    z_top = int(np.flatnonzero(near.any((0, 1))).max())
    assert z_top < nz - 1, "holder + margin reaches the window top"
    # t runs [on, off) of `near` per (n, z) row, CSR by row: a point hits a run at a range of slides
    edge = np.diff(np.pad(near.transpose(1, 2, 0), ((0, 0), (0, 0), (1, 1))).astype(np.int8),
                   axis=2).reshape(nn * nz, -1)
    r_on, t_on = np.nonzero(edge == 1)
    t_off = np.nonzero(edge == -1)[1]
    ptr = np.searchsorted(r_on, np.arange(nn * nz + 1))

    assert SLIDE_STEP == VOX_MM  # a slide node = a whole-voxel shift along t
    s_lo = math.floor((cutout[0] - SLIDE_PAD_MM) / SLIDE_STEP) * SLIDE_STEP
    ns = round((math.ceil((cutout[1] + SLIDE_PAD_MM) / SLIDE_STEP) * SLIDE_STEP - s_lo)
               / SLIDE_STEP) + 1
    shifts = np.arange(ns) + round(s_lo / SLIDE_STEP)
    pivots, depths = (lo + st * np.arange(n) for lo, st, n in (PIVOT_GRID, DEPTH_GRID))
    mask = np.zeros((ns, len(pivots), len(depths)), dtype=bool)
    forced = 0
    for j, piv in enumerate(pivots):
        X = pick(0.0, piv, 0.0)
        a, b = (pick(1.0, piv, 0.0)[:3, 3] - X[:3, 3], pick(0.0, piv, 1.0)[:3, 3] - X[:3, 3])
        far = pick(s_lo, piv, depths[0])[:3, 3] - X[:3, 3] - s_lo * a - depths[0] * b
        assert np.abs(far).max() < 1e-9 and np.allclose(pick(s_lo, piv, depths[0])[:3, :3],
                                                        X[:3, :3], atol=1e-12), (arm, piv)
        aL, bL = loc(frames[arm]["point"] + a), loc(frames[arm]["point"] + b)
        assert np.allclose(aL, [1.0, 0.0, 0.0], atol=1e-9), aL  # slide = +t exactly
        Q = loc(pts @ X[:3, :3].T + X[:3, 3])
        Q = Q[Q[:, 2] + min(depths[0] * bL[2], depths[-1] * bL[2]) <= g[2][z_top] + VOX_MM]
        for i, d in enumerate(depths):
            it, iN, iz = (np.rint((q - g[k][0]) / VOX_MM).astype(np.int64)
                          for k, q in enumerate((Q + d * bL).T))
            band = iz <= z_top  # higher rows hold no `near` voxel
            it, iN, iz = it[band], iN[band], iz[band]
            if (iz < 0).any() or (iN < 0).any() or (iN >= nn).any():
                forced += 1  # a point near the holder outside the window: conservative
                continue
            ok = np.ones(ns, dtype=bool)
            if len(it):
                ok = (it.min() + shifts >= 0) & (it.max() + shifts < nt)
                r = iN * nz + iz
                cnt = ptr[r + 1] - ptr[r]
                sel = cnt > 0
                c = cnt[sel]
                run = np.repeat(ptr[r[sel]] - np.cumsum(c) + c, c) + np.arange(c.sum())
                ip = np.repeat(it[sel], c) + shifts[0]
                lo, hi = (np.clip(t[run] - ip, 0, ns) for t in (t_on, t_off))
                ok &= np.cumsum(np.bincount(lo, minlength=ns + 1)
                                - np.bincount(hi, minlength=ns + 1))[:ns] == 0
            mask[:, j, i] = ok

    # nominal: signed (negative = penetration depth), nearest prism by footprint EDT + z gap
    q0 = loc(pts @ pick()[:3, :3].T + pick()[:3, 3])
    it, iN, iz = (np.clip(np.rint((q - g[k][0]) / VOX_MM).astype(np.int64), 0, len(g[k]) - 1)
                  for k, q in enumerate(q0.T))
    D_in = distance_transform_edt(occ, sampling=VOX_MM)
    nominal_mm = float((D[it, iN, iz] - D_in[it, iN, iz]).min()) - VOX_HALF_DIAG - SAMPLE_R
    per_part = sorted(
        (float(np.hypot(distance_transform_edt(~fp, sampling=VOX_MM)[it, iN],
                        np.maximum(0.0, np.maximum(z0 - q0[:, 2], q0[:, 2] - z1))).min()), nm)
        for (fp, z0, z1), nm in zip(prisms, names))
    k0, j0, i0 = (round(-lo / st) for lo, st in ((s_lo, SLIDE_STEP), PIVOT_GRID[:2],
                                                  DEPTH_GRID[:2]))
    assert mask[k0, j0, i0] == (nominal_mm >= margin + SLIDE_STEP / 2), (arm, nominal_mm)
    if forced:
        print(f"{arm}: {forced} (pivot, depth) cells infeasible by the holder window rule")
    return {"slide": [s_lo, SLIDE_STEP, ns], "pivot": list(PIVOT_GRID),
            "depth": list(DEPTH_GRID), "order": "slide,pivot,depth",
            "bits": base64.b64encode(np.packbits(mask.ravel())).decode(),
            "margin_mm": margin, "vox_mm": VOX_MM, "n_samples": len(pts),
            "n_feasible": int(mask.sum()), "enabled": True, "nominal_clearance_mm": round(nominal_mm, 2),
            "min_clearance_at_nominal_part": per_part[0][1],
            "nominal_part_distance_mm": [[nm, round(v, 2)] for v, nm in per_part[:4]]}, mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--scene", type=Path, default=None,
                    help="scene yaml (default: the snapshot's scene, which Variant's rest uses)")
    ap.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML)
    ap.add_argument("--device", choices=["cpu"], default="cpu")
    ap.add_argument("--margin-mm", type=float, default=MARGIN_MM,
                    help="finger-holder clearance of a feasible grasp (3D, collision shapes)")
    a = ap.parse_args()
    t_start = time.monotonic()
    mgv = load_mgv()

    snap = np.load(a.snapshot, allow_pickle=True)
    meta = json.loads(str(snap["meta_json"]))
    scene = a.scene or REPO / meta["scene_directives"]["path"]
    builder = round_belt_task.make_builder()
    info = round_belt_task.build_scene(builder, scene)
    labels = list(builder.body_label)
    if meta["body_labels"] != labels:
        raise SystemExit(f"{a.snapshot}: body labels differ from {scene}")
    held_q = np.asarray(snap["body_q"], dtype=np.float64)
    belt_ids = [int(b) for b in info.belt_bodies]
    rest = np.array([[float(v) for v in builder.body_q[b]][:3] for b in belt_ids])
    held = held_q[belt_ids, :3]
    belt_r = next(float(builder.shape_scale[s][0]) for s in range(builder.shape_count)
                  if builder.shape_body[s] == belt_ids[0]) * 1e3
    pick = load_pre_mpc_segment(a.params, first="pick", last="pick")[0]
    picks = {"franka": mat4(quat_xyzw_to_mat3(pick.franka_quat_xyzw), pick.franka_pos),
             "ur": mat4(quat_xyzw_to_mat3(pick.ur_quat_xyzw), pick.ur_pos)}
    tips = {"franka": model_pose_franka_tip(held_q, labels.index("panda_hand/finger_tip")),
            "ur": model_pose_ur_tracking(held_q, labels.index("/ur10/wrist_3_link"))}
    holder_ids = [s for s in range(builder.shape_count) if builder.shape_body[s] < 0
                  and builder.shape_label[s].startswith("belt_chain_holder/")
                  and not builder.shape_flags[s] & newton.ShapeFlags.VISIBLE]
    holder = [shape_tris(builder, s, np.eye(4)) for s in holder_ids]
    z_floor = min(float(h[..., 2].min()) for h in holder)
    probed = json.loads(PROBED.read_text())["box"]
    frames = {arm: {"point": picks[arm][:3, 3], "tangent": mgv.rest_tangent(rest, picks[arm][:3, 3])[1]}
              for arm in picks}

    geom = {}
    n_rest = len(rest)
    for arm in ("franka", "ur"):
        p = picks[arm][:3, 3]
        i = int(np.argmin(np.linalg.norm(rest - p, axis=1)))
        t = rest[(i + 1) % n_rest] - rest[(i - 1) % n_rest]
        t /= np.linalg.norm(t)
        z = np.array([0.0, 0.0, 1.0])
        n = np.cross(z, t)
        L = np.stack([t, n, z])

        def loc(x, p=p, L=L):
            return (np.asarray(x) - p) @ L.T * 1e3

        M = picks[arm] @ np.linalg.inv(tips[arm])
        bodies = [b for b, lab in enumerate(labels) if lab in FINGERS[arm]
                  or lab.rsplit("/", 1)[-1] in FINGERS[arm]]
        fingers = np.concatenate([
            shape_tris(builder, s, M @ pose4(held_q[builder.shape_body[s]]))
            for s in range(builder.shape_count)
            if builder.shape_body[s] in bodies and builder.shape_flags[s] & newton.ShapeFlags.VISIBLE])
        F = loc(fingers)
        j = int(np.argmin(np.linalg.norm(held - tips[arm][:3, 3], axis=1)))
        t_held = M[:3, :3] @ (held[(j + 1) % n_rest] - held[(j - 1) % n_rest])
        t_held /= np.linalg.norm(t_held)
        axis = np.cross(picks[arm][:3, 2], t)
        axis /= np.linalg.norm(axis)
        approach = np.array([picks[arm][:3, 2] @ t, picks[arm][:3, 2] @ z])
        approach /= np.linalg.norm(approach)

        # holder footprints in the local (t, n) window, one per collision prism
        gt, gn, gz = grid("t"), grid("n"), grid("z")
        lo_tn = np.array([gt[0], gn[0]])
        parts = []
        for tri in holder:
            q = loc(tri)
            parts.append((raster(q[..., :2], lo_tn, (len(gt), len(gn))),
                          float(q[..., 2].min()), float(q[..., 2].max()),
                          float(tri[..., 2].min()) <= z_floor + 1e-6))

        # cutout: arc around the nominal point with no part above the bottom plate
        loop = np.vstack([rest, rest[:1]])
        seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)]) * 1e3
        s = np.arange(0.0, cum[-1], RES)
        k = np.minimum(np.searchsorted(cum, s, side="right") - 1, n_rest - 1)
        dense = loc(loop[k] + ((s - cum[k]) / (seg[k] * 1e3))[:, None] * (loop[k + 1] - loop[k]))
        s0 = s[int(np.argmin(np.hypot(dense[:, 0], dense[:, 1])))]
        ds = (s - s0 + cum[-1] / 2) % cum[-1] - cum[-1] / 2
        near = np.abs(ds) <= 60.0
        covered = np.zeros(near.sum(), dtype=bool)
        for fp, _, _, plate in parts:
            if not plate:
                covered |= lookup(fp, gt, gn, dense[near][:, :2]) > 0
        dsn = ds[near]
        if covered[np.argmin(np.abs(dsn))]:
            raise SystemExit(f"{arm}: nominal grasp point is not inside a holder cutout")
        order = np.argsort(dsn)
        dsn, covered = dsn[order], covered[order]
        c0 = int(np.argmin(np.abs(dsn)))
        lo_i, hi_i = c0, c0
        while lo_i > 0 and not covered[lo_i - 1]:
            lo_i -= 1
        while hi_i < len(dsn) - 1 and not covered[hi_i + 1]:
            hi_i += 1
        cutout = [float(dsn[lo_i]), float(dsn[hi_i])]

        # obstacles: parts inside the fingers' n range below the holder top
        top = max(z1 for _, _, z1, _ in parts)
        low = F.reshape(-1, 3)
        low = low[low[:, 2] <= top]
        n_band = (gn >= low[:, 1].min()) & (gn <= low[:, 1].max())
        obs = [((gz >= z0) & (gz <= z1))[:, None] & fp[n_band].any(0)[None, :]
               for fp, z0, z1, _ in parts]
        # drawn holder: every part in the fixed WIN["n"] window, the same rule for both arms
        shown = [k for k, (fp, *_) in enumerate(parts) if fp.any()]
        drawn = [((gz >= parts[k][1]) & (gz <= parts[k][2]))[:, None] & parts[k][0].any(0)[None, :]
                 for k in shown]

        out = {"grasp_point": np.round(p, 5).tolist(), "t": np.round(t, 5).tolist(),
               "n": np.round(n, 5).tolist(), "rest_body": i, "held_body": j,
               "held_tangent_deg": round(math.degrees(math.acos(min(1.0, abs(t_held @ t)))), 1),
               "belt_r_mm": round(belt_r, 2), "pivot_axis": np.round(axis, 5).tolist(),
               "pivot_axis_deg": round(math.degrees(math.acos(min(1.0, abs(axis @ n)))), 1),
               "cutout_mm": np.round(cutout, 2).tolist(),
               "probed_box": {"slide_mm": probed[f"{arm}_slide_mm"],
                              "roll_deg": probed[f"{arm}_roll_deg"]}}
        mask, lo = sil_mask(F[..., [0, 2]])
        iy, ix = np.nonzero(mask)
        pts = np.stack([ix, iy], axis=1) * RES + lo
        fields = [distance_transform_edt(o) * RES for o in obs]
        base = depths(fields, gt, gz, pts) + TOL_MM

        def ok(moved, fields=fields, gt=gt, gz=gz, base=base):
            return bool((depths(fields, gt, gz, moved) <= base).all())

        # depth also against the plate top as a floor: a 5 mm plate's edt saturates at 2.5 mm
        z_top = max(z1 for _, _, z1, plate in parts if plate)
        plate_fields = [*fields, distance_transform_edt(
            np.repeat((gz <= z_top)[:, None], len(gt), 1)) * RES]
        plate_base = depths(plate_fields, gt, gz, pts) + TOL_MM

        limits = {"slide_mm": search(lambda v, pts=pts: ok(pts + [v, 0.0]), SLIDE_STEP, SLIDE_CAP),
                  "pivot_deg": [int(v) for v in search(lambda v, pts=pts: ok(rotate(pts, -v)), 1,
                                                       PIVOT_CAP)]}
        tip = float((pts @ approach).min())
        shallow = -belt_r - tip - TOL_MM  # fingertip end stays below the belt disc
        limits["depth_mm"] = [
            search(lambda v, pts=pts, a=approach, f=plate_fields, b=plate_base:
                   ok(pts + v * a, f, base=b), SLIDE_STEP, DEPTH_CAP)[0],
            min(DEPTH_CAP, max(0.0, math.floor(shallow / SLIDE_STEP) * SLIDE_STEP))]
        out["approach_A"] = np.round(approach, 5).tolist()
        out["approach_deg"] = round(math.degrees(math.atan2(approach[0], approach[1])), 1)
        out["depth_rule_mm"] = {"tip": round(tip, 2), "shallower_raw": round(shallow, 2)}
        if arm == "ur" and UR_DEPTH.is_file():
            rows = json.loads(UR_DEPTH.read_text())["variants"]
            out["verified_depth_mm"] = sorted({r["commanded"].get("ur_depth_mm", 0.0)
                                               for r in rows if r.get("held")})
        out["silhouette_A"] = outline(mask, lo)
        out["holder_A"] = outline(np.any(drawn, axis=0), [gt[0], gz[0]])
        out["holder_parts"] = [[builder.shape_label[holder_ids[k]].split("/", 1)[1],
                                round(parts[k][1], 2), round(parts[k][2], 2)] for k in shown]
        out["limits"] = limits
        arc = dense[np.abs(ds) <= BELT_ARC_MM][np.argsort(ds[np.abs(ds) <= BELT_ARC_MM])]
        out["belt_A"] = np.round(arc[::8][:, [0, 2]], 2).tolist()
        cbodies = {builder.shape_body[s] for s in range(builder.shape_count)
                   if builder.shape_flags[s] & newton.ShapeFlags.COLLIDE_SHAPES}
        for b in sorted(set(bodies) - cbodies):
            print(f"{arm}: {labels[b]} has no collision shape, using its visible shapes")
        tris = np.concatenate([
            shape_tris(builder, s, np.linalg.inv(tips[arm]) @ pose4(held_q[builder.shape_body[s]]))
            for s in range(builder.shape_count) if builder.shape_body[s] in bodies
            and builder.shape_flags[s] & (newton.ShapeFlags.COLLIDE_SHAPES
                                          if builder.shape_body[s] in cbodies
                                          else newton.ShapeFlags.VISIBLE)])
        samples = surface_samples(tris, SAMPLE_MM * 1e-3)
        t_arm = time.monotonic()
        f, m = feasibility(
            mgv, [pick], frames, arm, samples, loc, holder,
            [builder.shape_label[k].split("/", 1)[1] for k in holder_ids], cutout, a.margin_mm)
        out["feasible"] = f
        k0, j0, i0 = (round(-lo / st) for lo, st, _ in (f["slide"], PIVOT_GRID, DEPTH_GRID))
        on = [np.flatnonzero(m[:, j0, i0]), np.flatnonzero(m[k0, :, i0]),
              np.flatnonzero(m[k0, j0, :])]
        rng = [f"{lo + st * v[0]:+g}..{lo + st * v[-1]:+g}" if len(v) else "none"
               for (lo, st, _), v in zip((f["slide"], f["pivot"], f["depth"]), on)]
        print(f"{arm}: 3D nominal clearance {f['nominal_clearance_mm']:+.2f} mm "
              f"(nearest {f['nominal_part_distance_mm']}), {f['n_samples']} samples, grid "
              f"{m.shape}, {f['n_feasible']} feasible; at the nominal others: slide {rng[0]} mm, "
              f"pivot {rng[1]} deg, depth {rng[2]} mm ({time.monotonic() - t_arm:.1f} s)")
        dz = DEPTH_GRID[0] + DEPTH_GRID[1] * np.arange(DEPTH_GRID[2])
        d_ok = np.flatnonzero(m[k0, j0] & (dz <= limits["depth_mm"][1]))
        if not len(d_ok):
            raise SystemExit(f"{arm}: no collision-free depth at slide 0 / pivot 0")
        out["default_pose"] = {"slide_mm": 0.0, "pivot_deg": 0.0,
                               "depth_mm": float(dz[d_ok[np.argmin(np.abs(d_ok - i0))]])}
        print(f"{arm}: default pose {out['default_pose']}")
        geom[arm] = out
        print(f"{arm}: rest body {i}, held body {j}, held-vs-rest tangent "
              f"{out['held_tangent_deg']} deg, cutout {cutout[0]:+.1f}..{cutout[1]:+.1f} mm, "
              f"slide {limits['slide_mm'][0]:+.1f}..{limits['slide_mm'][1]:+.1f} mm, "
              f"pivot {limits['pivot_deg'][0]:+d}..{limits['pivot_deg'][1]:+d} deg (axis "
              f"{out['pivot_axis_deg']} deg off n), depth {limits['depth_mm'][0]:+.1f}.."
              f"{limits['depth_mm'][1]:+.1f} mm (tip {tip:+.2f} mm, shallower rule "
              f"{shallow:+.2f} mm, approach {out['approach_deg']:+.1f} deg off vertical) "
              f"(probed {out['probed_box']})")

    srcs = {"scene": scene, "snapshot": a.snapshot, "params": a.params, "probed": PROBED}
    geom["source"] = {k: str(Path(v).resolve().relative_to(REPO)) if Path(v).resolve()
                      .is_relative_to(REPO) else str(v) for k, v in srcs.items()}
    geom["source"]["sha256"] = {k: sha256(v) for k, v in srcs.items()}
    geom["source"]["method"] = {"res_mm": RES, "simplify_mm": SIMPLIFY_MM, "tol_mm": TOL_MM,
                                "slide_step_mm": SLIDE_STEP, "pivot_cap_deg": PIVOT_CAP,
                                "depth_cap_mm": DEPTH_CAP}
    geom["source"]["method"]["collision"] = {
        "shapes": "finger collision meshes vs holder collision prisms, 3D, at the variant_waypoints pick",
        "vox_mm": VOX_MM, "margin_mm": a.margin_mm, "half_voxel_diag_mm": round(VOX_HALF_DIAG, 3),
        "sample_mm": SAMPLE_MM, "sample_radius_mm": round(SAMPLE_R, 3),
        "slide_half_step_mm": SLIDE_STEP / 2, "nominal_clearance_mm": {
            arm: [geom[arm]["feasible"]["nominal_clearance_mm"],
                  geom[arm]["feasible"]["min_clearance_at_nominal_part"]] for arm in picks}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(geom, separators=(",", ":")) + "\n")
    print(f"wrote {a.out} ({a.out.stat().st_size // 1024} KB) in {time.monotonic() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
