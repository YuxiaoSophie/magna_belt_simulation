"""Bike-chain extension directives (``EXTENSION_DIRECTIVES``): the shared ones + ``add_roller_chain``.

Same rules as ``round_belt_task.directives`` (``docs/scene-directives.md`` section 8): strict keys
through ``task_common.directives.params``, numbers through ``utils.directives`` helpers, output in
``ctx.scene.extras[params["name"]]``.  Chain geometry: ``handoffs/research/bike-chain-facts.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import newton
import numpy as np
import warp as wp

from task_common.defaults import CABLE_CONTACT_KD, CABLE_CONTACT_KE
from task_common.directives import COMMON_DIRECTIVES, REQUIRED
from task_common.directives import params as _params
from utils.directives import DirectiveContext, DirectiveFn, as_floats, as_vec3
from utils.meshes import load_meshes

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "bike_chain_task"
# magna bike_chain.urdf link inertia at 5 g (COM z is re-centred on the plate stack).
LINK_MASS = 0.005  # magna's per-link mass: the inertia reference
CHAIN_LINK_MASS = 0.2 / 64  # #35 ~ 0.31 kg/m x 0.61 m = 0.20 kg; magna's 5 g/link is 1.6x heavy
LINK_INERTIA_DIAG = (6.97e-8, 6.97e-8, 8.33e-8)
REAL_INNER_WIDTH = 0.0045
OUTER_TOP_CLEARANCE = 0.0005  # top inner plate -> top outer plate (magna stack)
CHAIN_CONTACT_MU = 0.15  # lubricated steel; MuJoCo pairs use max(mu), so sprockets need it too

_CHAIN_SPEC: dict[str, Any] = {
    "name": REQUIRED, "center": REQUIRED, "radius": None, "semi_axes": None,
    "num_links": REQUIRED, "pitch": REQUIRED, "color": [0.3, 0.3, 0.3], "mass": CHAIN_LINK_MASS,
    "gap": 0.0002, "margin": 0.0, "ke": None, "kd": None, "mu": None, "plate": None,
    "roller": None, "inner_width": REAL_INNER_WIDTH, "visual_meshes": True, "collision_group": -2,
    "armature": 1.0e-6, "plate_shape": "mesh", "play": False, "play_mass": 1.0e-4,
    "play_damping": 0.0, "play_frictionloss": 0.0, "play_slide": True, "damping": 0.0, "frictionloss": 1.0e-3,
}
PLAY_TILT = 0.02  # rad about the link's long axis (magna bike_chain.urdf)
PLAY_SLIDE = 0.0005  # m along the pin: inner -> outer [-0.5, 0] mm, outer -> inner [0, 0.5] mm
PLAY_LIMIT_TIMECONST = 0.004  # s, >= 2 dt at 60 fps x 10 substeps
SOLREF_MODE_RAW = 1  # newton/_src/solvers/mujoco/constants.py (not public)
_PLATE_SPEC: dict[str, Any] = {"length": 0.017725, "thickness": 0.00125, "height": 0.0082}
_ROLLER_SPEC: dict[str, Any] = {"radius": 0.0025, "length": REAL_INNER_WIDTH}  # magna's


def chord_walk(curve: np.ndarray, start: int, n: int, pitch: float) -> tuple[np.ndarray, float]:
    """``n`` pins chord-walked at ``pitch`` along a closed dense polyline from ``curve[start]``;
    returns the pins and the signed arc overshoot of pin n past the start [m]."""
    seg = np.linalg.norm(np.diff(curve, axis=0, append=curve[:1]), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    loop = np.concatenate([curve, curve, curve[:1]])
    sl = np.concatenate([s[:-1], s[:-1] + total, [2 * total]])
    i, pins = start, [curve[start]]
    cur, s_cur = curve[start], s[start]
    for _ in range(n):
        d = np.linalg.norm(loop[i + 1:] - cur, axis=1)
        j = i + 1 + int(np.argmax(d >= pitch))
        d0 = np.linalg.norm(loop[j - 1] - cur)
        f = (pitch - d0) / max(d[j - i - 1] - d0, 1e-12)
        cur = loop[j - 1] + f * (loop[j] - loop[j - 1])
        s_cur = sl[j - 1] + f * (sl[j] - sl[j - 1])
        pins.append(cur)
        i = j - 1
    return np.array(pins[:n]), s_cur - (s[start] + total)


def ellipse_pins(center_xy, semi_axes, n: int, pitch: float) -> tuple[np.ndarray, float]:
    """(n, 2) pins chord-walked CCW round the ellipse ``semi_axes`` (world x, y) about
    ``center_xy``, its scale bisected until n pitches close; pin 0 at the +x end, y -pitch/2.
    Returns the pins and the scale applied to ``semi_axes``."""
    t = np.linspace(0.0, 2.0 * math.pi, 40000, endpoint=False)
    unit = np.c_[semi_axes[0] * np.cos(t), semi_axes[1] * np.sin(t)]
    start = int(np.argmin(np.abs(t - (2.0 * math.pi - math.asin(0.5 * pitch / semi_axes[1])))))
    s0 = n * pitch / np.linalg.norm(np.diff(unit, axis=0, append=unit[:1]), axis=1).sum()
    lo, hi = 0.98 * s0, 1.05 * s0  # chords are shorter than the arc
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if chord_walk(mid * unit, start, n, pitch)[1] > 0 else (lo, mid)
    scale = 0.5 * (lo + hi)
    pins, err = chord_walk(scale * unit, start, n, pitch)
    if abs(err) > 5e-5:
        raise RuntimeError(f"ellipse chord walk does not close: {err * 1e3:.4f} mm")
    return pins + np.asarray(center_xy, dtype=np.float64), scale


def ring_pins(center_xy, radius: float, n: int) -> np.ndarray:
    """(n, 2) vertices of the regular n-gon, vertex k at azimuth (2k - 1) pi / n."""
    a = (2.0 * np.arange(n) - 1.0) * math.pi / n
    return np.asarray(center_xy, dtype=np.float64) + radius * np.c_[np.cos(a), np.sin(a)]


def chain_joint_q(pins_xy: np.ndarray, z: float = 0.0,
                  frame: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Free-joint q (xyz + quat xyzw) of link 0 and the n pin angles for a planar pose.

    ``pins_xy`` (n, 2) lie in the plane z = ``z`` of ``frame`` (world pose, 7: xyz + quat xyzw;
    None = world), CCW; ``z`` is the inner-link frame height.  Pin k angle = heading change at
    vertex k+1 minus 2 pi / n (the joint's built-in yaw), so the regular n-gon is all zeros.
    """
    n = len(pins_xy)
    d = np.roll(pins_xy, -1, axis=0) - pins_xy
    heading = np.arctan2(d[:, 1], d[:, 0])
    angles = (np.roll(heading, -1) - heading - 2.0 * math.pi / n + math.pi) % (2.0 * math.pi)
    yaw0 = heading[0] - 0.5 * math.pi  # link +y along edge 0
    p = np.array([pins_xy[0, 0], pins_xy[0, 1], z])
    q = np.array([0.0, 0.0, math.sin(0.5 * yaw0), math.cos(0.5 * yaw0)])
    if frame is not None:
        fq = wp.quat(*frame[3:7])
        p = np.array(wp.quat_rotate(fq, wp.vec3(*p))) + frame[:3]
        q = np.array(fq * wp.quat(*q))
    return np.r_[p, q], angles - math.pi


def add_roller_chain(ctx: DirectiveContext, params: Mapping[str, Any]) -> None:
    """A closed roller chain of ``num_links`` rigid links joined by REVOLUTE pins (axis world z).

    Built for ``SolverMuJoCo``: a free joint on link 0 and one articulation over pins 0..n-2; pin
    n-1 stays OUTSIDE the articulation, which MuJoCo turns into the loop closure (two CONNECT
    equalities along the pin axis).  Link k (even = inner, odd = outer, magna's alternation) has
    its origin at pin k, link +y along the edge to pin k+1, +x outward.  The flat start polygon
    is the regular n-gon of circumradius ``radius`` (vertex k at azimuth ``(2k - 1) pi / n``) or,
    with ``semi_axes`` [x, y] instead, n pins chord-walked round that ellipse with its scale
    bisected so the polygon closes (pin 0 at the +x end); ``builder.joint_q`` carries the pose.
    ``center`` z is the INNER-link frame z (the bottom face of its bottom plate); outer-link
    frames sit ``plate.thickness`` (1.25 mm) lower, as in magna's URDF.  Stack in the inner frame,
    t = plate thickness, w = ``inner_width``: outer plate -t..0, inner plate 0..t, rollers
    t..t+w, inner plate t+w..2t+w, outer plate 2t+w+0.5 mm..3t+w+0.5 mm.  Default w is
    magna's 4.5 mm (MISUMI CHE35-64): plates at z 0 / 5.75 (inner), 0 / 8.75 mm (outer), rollers at
    3.5 mm, 0.1 mm per side over the 4.3 mm rim; a larger ``inner_width`` widens the gap.
    Plates (``plate_shape`` "mesh") are magna's ``chain_link_collision.obj``: a 1.25 mm stadium,
    4.1 mm end radii centred on the two pins (MuJoCo collides its convex hull, <= 64 vertices).
    ``plate_shape: box`` = the V13-02b boxes.
    Every chain collision shape gets ``collision_group`` (negative: chain self-contact off).

    Params: ``name``, ``center`` (3), ``radius`` (must equal ``pitch / (2 sin(pi / n))``) or
    ``semi_axes`` (2), ``num_links`` (even int >= 8), ``pitch``; optional ``color``
    ([0.3, 0.3, 0.3]), ``mass`` per link (0.003125 = 0.20 kg / 64: #35 ~ 0.31 kg/m; magna's 5 g/link is 1.6x
    heavy; magna's inertia scaled by mass), ``gap``
    (0.0002), ``margin`` (0.0), ``ke``/``kd`` (``CABLE_CONTACT_*``), ``mu``
    (``CHAIN_CONTACT_MU`` 0.15: at the cable's 1.0 rollers are carried ~20 deg past the
    tangent; MuJoCo uses max(mu) per pair, so the sprockets must be 0.15 as well), ``plate`` {``length``
    0.017725, ``thickness`` 0.00125, ``height`` 0.0082}, ``roller`` {``radius`` 0.0025,
    ``length`` 0.0045}, ``inner_width`` (0.0045), ``visual_meshes`` (true: magna's
    link ``.obj``s, z-stretched if ``inner_width`` is widened), ``collision_group`` (-2; must be unused),
    ``armature`` (1e-6 kg m^2 on the free joint and every pin DOF; less blows up in mujoco_warp),
    ``plate_shape`` ("mesh" stadium or "box" 17.725 x 8.2 x 1.25 mm), ``play`` (false; true =
    magna's per-pitch play: pin -> tilt about x (+-0.02 rad) -> slide along z (inner -> outer
    [-0.5, 0] mm, outer -> inner [0, 0.5] mm) through two dummies of ``play_mass`` (1e-4 kg; the
    links lose the same mass so the chain stays 64 x ``mass``), none on the closing pitch),
    ``play_damping`` (0.0 on the tilt/slide DOFs), ``play_frictionloss`` (0.0 N m / N on them),
    ``play_slide`` (true; false = tilt-only play, one dummy per pitch), ``damping`` (0.0 N m s/rad on the pins; magna
    0.5) and ``frictionloss`` (1e-3 N m, MuJoCo dry friction on the pins: damps the free spans'
    in-plane whip from chordal action; 1e-4 NaNs at ~2 s of drive, viscous damping NaNs, V13-02b).
    With ``visual_meshes`` the plates and rollers are hidden (collision only).
    Rollers are magna's (r 2.5 mm, length 4.5 mm, centred in the gap).  magna's rim cylinders
    reach 0.33 mm (16T) / 0.15 mm (32T) into a 2.5 mm roller on the pitch circle
    (``gen_bike_chain_board_urdf.py --root-rims`` shrinks them; the default keeps magna's).
    Stores ``extras[name] = {"bodies" (links), "dummy_bodies", "joints" (all), "pin_joints",
    "shapes", "roller_shapes", "plate_shapes", "pitch", "num_links"}``.
    """
    p, where, num = _params("add_roller_chain", params, _CHAIN_SPEC)
    _, _, plate_num = _params("add_roller_chain plate", p["plate"] or {}, _PLATE_SPEC)
    _, _, roller_num = _params("add_roller_chain roller", p["roller"] or {}, _ROLLER_SPEC)
    builder, name, n = ctx.builder, str(p["name"]), p["num_links"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 8 or n % 2:
        raise ValueError(f"{where}: num_links must be an even integer >= 8, got {n!r}")
    if not isinstance(p["visual_meshes"], bool):
        raise TypeError(f"{where}: visual_meshes must be true or false, got {p['visual_meshes']!r}")
    group = p["collision_group"]
    if isinstance(group, bool) or not isinstance(group, int):
        raise TypeError(f"{where}: collision_group must be an integer, got {group!r}")
    if group in builder.shape_collision_group:
        raise ValueError(f"{where}: collision_group {group} is already used by another shape")
    if p["plate_shape"] not in ("mesh", "box"):
        raise ValueError(f"{where}: plate_shape {p['plate_shape']!r} is not 'mesh' or 'box'")
    if (p["radius"] is None) == (p["semi_axes"] is None):
        raise ValueError(f"{where}: give exactly one of radius and semi_axes")
    pitch = num("pitch")
    cx, cy, cz = as_vec3(p["center"], where, "center")
    if p["radius"] is not None:
        radius = num("radius")
        if abs(radius - pitch / (2.0 * math.sin(math.pi / n))) > 1.0e-6:
            raise ValueError(f"{where}: radius {radius} != pitch / (2 sin(pi/n)) "
                             f"= {pitch / (2.0 * math.sin(math.pi / n)):.8f}")
        pins = ring_pins((cx, cy), radius, n)
    else:
        pins, _ = ellipse_pins((cx, cy), as_floats(p["semi_axes"], where, "semi_axes", 2), n,
                               pitch)
    free_q, pin_q = chain_joint_q(pins, cz)
    color = as_vec3(p["color"], where, "color")
    t, w = plate_num("thickness"), num("inner_width")
    half_len, half_h = 0.5 * plate_num("length"), 0.5 * plate_num("height")
    r_roll = roller_num("radius")
    l_roll = roller_num("length")
    mass = num("mass")
    if not isinstance(p["play"], bool):
        raise TypeError(f"{where}: play must be true or false, got {p['play']!r}")
    m_play = num("play_mass") if p["play"] else 0.0
    n_dummy = 2 if p["play_slide"] else 1  # dummies per pitch
    mass -= n_dummy * (n - 1) * m_play / n  # the play dummies keep the chain's total mass
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, ke=num("ke", CABLE_CONTACT_KE), kd=num("kd", CABLE_CONTACT_KD),
        mu=num("mu", CHAIN_CONTACT_MU), margin=num("margin"), gap=num("gap"),
        collision_group=group, is_visible=not p["visual_meshes"])  # meshes draw the links
    # Plate bottoms in each link's own frame; outer = inner-frame z + t.
    inner_plates = (0.0, t + w)
    outer_plates = (0.0, 3.0 * t + w + OUTER_TOP_CLEARANCE)
    plate_mesh = None
    if p["plate_shape"] == "mesh":  # magna's stadium; MuJoCo uses its hull (pin holes filled)
        plate_mesh, *rest = load_meshes(ASSETS / "chain_link_collision.obj", per_material=False)
        if rest:
            raise RuntimeError(f"{where}: chain_link_collision.obj is not one mesh")
    visuals = {}
    if p["visual_meshes"]:
        stretch = {"inner": (2 * t + w) / (2 * t + REAL_INNER_WIDTH),
                   "outer": (4 * t + w + OUTER_TOP_CLEARANCE)
                   / (4 * t + REAL_INNER_WIDTH + OUTER_TOP_CLEARANCE)}
        visuals = {kind: (load_meshes(ASSETS / f"{kind}_chain_link.obj"), stretch[kind])
                   for kind in ("inner", "outer")}

    body_start, joint_start, shape_start = (
        builder.body_count, builder.joint_count, builder.shape_count)
    bodies, roller_shapes, plate_shapes = [], [], []
    ixx, iyy, izz = (d * mass / LINK_MASS for d in LINK_INERTIA_DIAG)
    inertia = wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)
    d = np.roll(pins, -1, axis=0) - pins
    yaws = np.arctan2(d[:, 1], d[:, 0]) - 0.5 * math.pi
    for k in range(n):
        kind = "inner" if k % 2 == 0 else "outer"
        z = cz if kind == "inner" else cz - t
        plates = inner_plates if kind == "inner" else outer_plates
        stack_mid = 0.5 * (plates[0] + plates[1] + t)
        label = f"{name}/{kind}_chain_link_{k // 2}"
        body = builder.add_link(
            xform=wp.transform(wp.vec3(float(pins[k, 0]), float(pins[k, 1]), z),
                               wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaws[k]))),
            com=wp.vec3(0.0, 0.5 * pitch, stack_mid), inertia=inertia, mass=mass,
            label=label, lock_inertia=True)
        bodies.append(body)
        for side, z0 in zip(("lo", "hi"), plates):
            if plate_mesh is not None:
                plate_shapes.append(builder.add_shape_mesh(
                    body, xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()),
                    mesh=plate_mesh, cfg=cfg, color=color, label=f"{label}/plate_{side}"))
                continue
            plate_shapes.append(builder.add_shape_box(
                body, xform=wp.transform(wp.vec3(0.0, 0.5 * pitch, z0 + 0.5 * t),
                                         wp.quat_identity()),
                hx=half_h, hy=half_len, hz=0.5 * t, cfg=cfg, color=color,
                label=f"{label}/plate_{side}"))
        if kind == "inner":
            for j, y in enumerate((0.0, pitch)):
                roller_shapes.append(builder.add_shape_cylinder(
                    body, xform=wp.transform(wp.vec3(0.0, y, t + 0.5 * w), wp.quat_identity()),
                    radius=r_roll, half_height=0.5 * l_roll, cfg=cfg, color=color,
                    label=f"{label}/roller_{j}"))
        if visuals:
            meshes, sz = visuals[kind]
            for m, mesh in enumerate(meshes):
                builder.add_shape_mesh(
                    body, mesh=mesh, scale=wp.vec3(1.0, 1.0, sz), cfg=ctx.visual_cfg,
                    color=color, label=f"{label}/visual{m}")
    if bodies != list(range(body_start, builder.body_count)):
        raise RuntimeError(f"{where}: chain bodies are not contiguous from {body_start}")

    dummies = []
    if p["play"]:  # magna's dummies, near-massless: pin -> tilt (x) -> slide (z) per pitch
        i_play = wp.mat33(*(np.eye(3) * LINK_INERTIA_DIAG[0] * m_play / LINK_MASS).ravel())
        for k in range(n - 1):
            for j in range(1, n_dummy + 1):
                dummies.append(builder.add_link(
                    xform=builder.body_q[bodies[k + 1]], mass=m_play, inertia=i_play,
                    label=f"{name}/dummy_{k}_{j}", lock_inertia=True))
    free = builder.add_joint_free(bodies[0], label=f"{name}/free_joint")
    yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 2.0 * math.pi / n)
    pins_j, tree = [], [free]
    for k in range(n):
        dz = -t if k % 2 == 0 else t  # inner -> outer frame is t lower
        child = bodies[(k + 1) % n] if not dummies or k == n - 1 else dummies[n_dummy * k]
        pins_j.append(builder.add_joint_revolute(
            bodies[k], child, parent_xform=wp.transform(wp.vec3(0.0, pitch, dz), yaw),
            child_xform=wp.transform_identity(), axis=newton.Axis.Z, label=f"{name}/pin_{k}",
            damping=num("damping"), friction=num("frictionloss")))
        if k == n - 1:
            break  # the loop joint: no play on the closing pitch (magna: weld, no slide)
        tree.append(pins_j[-1])
        if dummies:
            lo, hi = (-PLAY_SLIDE, 0.0) if k % 2 == 0 else (0.0, PLAY_SLIDE)
            # Raw MuJoCo limits, stiffest refsafe allows: at (0.02 s, dmax 0.95) the plate loads
            # push the slide ~4 mm past +-0.5 mm (soft-constraint equilibrium (1-d)/d a0 / k).
            lim = {"damping": num("play_damping"), "friction": num("play_frictionloss"),
                   "custom_attributes": {
                "mujoco:solreflimit": [wp.vec2(PLAY_LIMIT_TIMECONST, 1.0)],
                "mujoco:solreflimit_mode": [SOLREF_MODE_RAW],
                "mujoco:solimplimit": [(0.95, 0.99, 0.001, 0.5, 2.0)]}}
            tilt_child = dummies[2 * k + 1] if p["play_slide"] else bodies[k + 1]
            tree.append(builder.add_joint_revolute(
                dummies[n_dummy * k], tilt_child, axis=newton.Axis.X, limit_lower=-PLAY_TILT,
                limit_upper=PLAY_TILT, label=f"{name}/tilt_{k}", **lim))
            if p["play_slide"]:
                tree.append(builder.add_joint_prismatic(
                    dummies[2 * k + 1], bodies[k + 1], axis=newton.Axis.Z, limit_lower=lo,
                    limit_upper=hi, label=f"{name}/slide_{k}", **lim))
    builder.add_articulation(tree, label=f"{name}/articulation")
    # 5 g / 8e-8 kg m^2 links: mujoco_warp NaNs below ~1e-6 kg m^2 of armature (V13-02b).
    for d in range(builder.joint_qd_start[free], builder.joint_dof_count):
        builder.joint_armature[d] = num("armature")
    q0 = builder.joint_q_start[free]
    builder.joint_q[q0:q0 + 7] = [float(v) for v in free_q]
    for j, a in zip(pins_j, pin_q):
        builder.joint_q[builder.joint_q_start[j]] = float(a)
    ctx.scene.extras[name] = {
        "bodies": bodies,
        "dummy_bodies": dummies,
        "joints": list(range(joint_start, builder.joint_count)),
        "pin_joints": pins_j,
        "shapes": list(range(shape_start, builder.shape_count)),
        "roller_shapes": roller_shapes,
        "plate_shapes": plate_shapes,
        "pitch": pitch,
        "num_links": n,
    }


EXTENSION_DIRECTIVES: dict[str, DirectiveFn] = {
    **COMMON_DIRECTIVES, "add_roller_chain": add_roller_chain,
}
