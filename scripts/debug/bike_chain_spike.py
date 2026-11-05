"""Chain spike take 2 (V13-02b): the 64-link chain + sprockets in ``SolverMuJoCo`` (mujoco_warp).

Builds ``assets/bike_chain_task/bike_chain_spike.yaml`` (table, board + 3 sprockets, holder,
chain, ground; ``full``: ``bike_chain_scene.yaml`` with the robots) into ``make_builder()``, poses
the chain through ``joint_q`` (free joint of link 0 + pin angles; MuJoCo closes the loop with two
CONNECT rows on pin 63), steps ONE ``bike_chain_task.solver.make_mujoco_solver`` with MuJoCo's own
collision (no Newton pipeline) and scores S1-S6:

  rest        S1 + S2: flat 64-gon chord-walked round the holder ellipse, on its tier, 3 s.
  rest_board  S2: the yaml's flat ring 1 mm above the board's free corner, 3 s.
  seated      S3: pre-seated on small + large (slack V over the tensioner), 3 s.
  drive       S4: seated start, large sprocket position-driven at 0.5 rad/s for 2 s.
  full        S6: bike_chain_scene.yaml (robots, default targets), 3 s.

Seated wrap (board frame, CCW, link +x outward): pins chord-walked at exactly one pitch along a
closed curve = small pitch-circle arc -> tangent from the small circle to a kink point A ->
tangent from A to the large circle -> large arc -> straight upper external tangent back.  A sits
at the tensioner's x on the far side of it (the tensioner ends up inside the loop, as an idler);
its depth is bisected until 64 pitches close the loop.  The walk starts in a small-sprocket tooth
gap; the large sprocket's joint_q puts its gaps under the walked pins.  A pure-hinge chain is
planar, so it lies in the plane least-squares fitted through the small and large tooth planes.

``--fit-holder`` (CPU): the holder-ellipse start (``fit_holder``) written into
``bike_chain_scene.yaml``; no tier fit -> the board-corner ring.

Output: ``<out>/<run>/{<test>.json,summary.json,model.xml}`` and ``<out>/../spike.{md,html}``.
``--viewer viser --viewer-port N`` (one ``--test``): watch it live (every ``--render-every``-th
frame), then the rendered frames loop until Ctrl-C.  ``--record-viser PATH`` (one ``--test``,
with or without ``--viewer``) saves the rendered frames as a scrubbable ``.viser`` timeline
(play it with ``scripts/debug/play_viser.py``).
flock data/lcs/realfirst/cmp/v12_79/sim.lock uv run --frozen python \\
    scripts/debug/bike_chain_spike.py --test all --out data/bike_chain/v13_02b/spike
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import newton
import trimesh
import warp as wp

from bike_chain_task.directives import (
    ASSETS,
    EXTENSION_DIRECTIVES,
    chain_joint_q,
    chord_walk,
    ellipse_pins,
)
from bike_chain_task.solver import make_mujoco_solver
from round_belt_task.constants import PANDA_FINGER_LABELS, PANDA_JOINT_LABELS, UR10_JOINT_LABELS
from round_belt_task.outcome import belt_in_pulley_frame
from task_common.defaults import make_robust_table_collision_cfg, make_visual_cfg
from task_common.joint_state import apply_default_joint_state
from task_common.scene import build_task_scene, make_builder
from task_common.simulation import BeltTaskSimulation
from utils.directives import load_directives, parse_directives

SCENE = ASSETS / "bike_chain_spike.yaml"
SCENE_FULL = ASSETS / "bike_chain_scene.yaml"
TABLE_TOP_Z = -0.02858
T_PLATE, W_INNER, PLATE_H = 0.00125, 0.0045, 0.0082  # add_roller_chain defaults
FPS = 60
TESTS = ("rest", "rest_board", "seated", "drive", "full")
DURATION = {"rest": 3.0, "rest_board": 3.0, "seated": 3.0, "drive": 15.0, "full": 3.0}
SAMPLE_EVERY = 6  # frames: 10 Hz metrics
DRIVE_RATE = -0.5  # rad/s; < 0 puts the tensioner on the slack side
DRIVE_KE, DRIVE_KD = 10.0, 0.1
MESH_TOL_DEG, BAND_R_MM, BAND_H_MM = 4.0, 3.0, 5.0
SPROCKETS = ("small", "large")
HOLDER_SEMI_AXES = (0.08558, 0.12066)  # the round belt's holder ellipse (x, y)
HOLDER_CLEARANCE = 0.002  # plate edge -> next step up
HOLDER_LIFT = 0.0005  # bottom plate above the tier
BOARD_RING = "center: [0.393, -0.090, 0.012]"  # spike yaml's board-corner ring (fallback start)


def _tf(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """World points of local points ``x`` (n, 3) on poses ``q`` (n, 7 xyz + quat xyzw)."""
    return q[:, :3] + Rotation.from_quat(q[:, 3:7]).apply(x)


def _board_height(X_wb: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """World z of the board's top plane (board z = 0) at world ``xy`` (n, 2)."""
    n = Rotation.from_quat(X_wb[3:7]).apply([0.0, 0.0, 1.0])
    return X_wb[2] - (n[0] * (xy[:, 0] - X_wb[0]) + n[1] * (xy[:, 1] - X_wb[1])) / n[2]


def _ccw_arc(c: np.ndarray, r: float, a0: float, a1: float, m: int) -> np.ndarray:
    a1 = a1 if a1 > a0 else a1 + 2 * math.pi
    a = np.linspace(a0, a1, m)
    return c + r * np.c_[np.cos(a), np.sin(a)]


def _tangent_point(c: np.ndarray, r: float, a: np.ndarray, leaving: bool) -> float:
    """Angle of the point on circle (c, r) whose CCW tangent runs to (leaving) / from ``a``."""
    d = a - c
    base, off = math.atan2(d[1], d[0]), math.acos(r / np.linalg.norm(d))
    for ang in (base + off, base - off):
        q = c + r * np.array([math.cos(ang), math.sin(ang)])
        vel = np.array([-math.sin(ang), math.cos(ang)])
        if ((a - q) @ vel > 0) == leaving:
            return ang
    raise RuntimeError("no tangent")


def seated_pins(S, L, T, r_s, r_l, tip_t, n, pitch, gap_az_s):
    """Board-frame (n, 2) pins of the seated wrap (see the module docstring) and A's depth."""
    beta = math.asin((r_l - r_s) / np.linalg.norm(L - S))
    up = math.atan2(math.cos(beta), -math.sin(beta))  # upper external tangent normal angle

    def curve(h):
        a = np.array([T[0], T[1] - h])  # outside the loop, beyond the tensioner
        a_s, a_l = _tangent_point(S, r_s, a, True), _tangent_point(L, r_l, a, False)
        parts = [_ccw_arc(S, r_s, up, a_s, 4000), np.linspace(S + r_s * np.array(
            [math.cos(a_s), math.sin(a_s)]), a, 4000), np.linspace(a, L + r_l * np.array(
                [math.cos(a_l), math.sin(a_l)]), 4000), _ccw_arc(L, r_l, a_l, up, 6000),
            np.linspace(L + r_l * np.array([math.cos(up), math.sin(up)]),
                        S + r_s * np.array([math.cos(up), math.sin(up)]), 8000)]
        c = np.concatenate([p[:-1] for p in parts])
        start = int(np.argmin(np.linalg.norm(
            c - (S + r_s * np.array([math.cos(gap_az_s), math.sin(gap_az_s)])), axis=1)))
        return chord_walk(c, start, n, pitch)

    lo, hi = tip_t, 0.08
    assert curve(lo)[1] > 0 > curve(hi)[1], "seated wrap: closure not bracketed"
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if curve(mid)[1] > 0 else (lo, mid)
    pins, err = curve(0.5 * (lo + hi))
    assert abs(err) < 2e-5 and np.allclose(np.linalg.norm(
        np.roll(pins, -1, 0) - pins, axis=1), pitch, atol=2e-5), err
    return pins, 0.5 * (lo + hi)


def _load(path: Path, builder, roller_radius: float, full: bool, plate_shape: str = "mesh",
          rims: str = "magna", play: bool = False, damping: float = 0.0,
          frictionloss: float = 1.0e-3, mu: float | None = None, play_frictionloss: float = 0.0,
          play_slide: bool = True):
    """Scene into ``builder`` with the chain's roller radius, plate shape and board rims."""
    text = path.read_text().replace("file: ../", f"file: {ASSETS.parent}/").replace(
        "file: bike_chain", f"file: {ASSETS}/bike_chain").replace(
        "      pitch: 0.009525\n", f"      pitch: 0.009525\n      roller: {{radius: {roller_radius}}}\n"
        f"      plate_shape: {plate_shape}\n      play: {str(play).lower()}\n"
        f"      damping: {damping}\n      frictionloss: {frictionloss}\n"
        + (f"      mu: {mu}\n" if mu is not None else "")
        + (f"      play_frictionloss: {play_frictionloss}\n      play_slide: "
           f"{str(play_slide).lower()}\n" if play else ""))
    if rims == "root":
        text = re.sub(r"(bike_chain_task_board\w*)\.urdf", r"\1_root_rims.urdf", text)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / path.name
        tmp.write_text(text)
        if full:
            return build_task_scene(builder, tmp, directives=EXTENSION_DIRECTIVES,
                                    belt_name="bike_chain", table_visual_label="table/scene/visual0",
                                    table_top_z=TABLE_TOP_Z)
        return load_directives(builder, tmp, directives=EXTENSION_DIRECTIVES,
                               visual_cfg=make_visual_cfg(),
                               collision_cfg=make_robust_table_collision_cfg(visible=False))


def fit_holder(roller_radius: float = 0.0025) -> dict:
    """Holder-ellipse start: the round belt's holder ellipse scaled until 64 pitches close, tested
    against the holder's collision meshes (trimesh rays).  ``literal``: every pin's plate band
    (+-4.1 mm + 2 mm) on one tier; ``bridged``: the same but pins over the holder's 40 mm finger
    channels may bridge them (a pure-hinge chain cannot sag) -- no step UP within the band
    anywhere and >= half the pins on the tier."""
    builder = make_builder()
    scene = _load(SCENE, builder, roller_radius, full=False)
    rec, chain = scene.models["belt_chain_holder"], scene.extras["bike_chain"]
    flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    parts = []
    for s in rec.shapes:
        if builder.shape_type[s] != newton.GeoType.MESH or not builder.shape_flags[s] & flag:
            continue
        assert builder.shape_body[s] == -1, "holder shapes are expected in the world frame"
        src, X = builder.shape_source[s], np.array(builder.shape_transform[s])
        v = Rotation.from_quat(X[3:7]).apply(np.asarray(src.vertices) * np.array(
            builder.shape_scale[s])) + X[:3]
        parts.append(trimesh.Trimesh(v, np.asarray(src.indices).reshape(-1, 3), process=False))
    mesh = trimesh.util.concatenate(parts)

    def surface(xy: np.ndarray) -> np.ndarray:
        loc, ray, _ = mesh.ray.intersects_location(
            np.c_[xy, np.ones(len(xy))], np.tile([0.0, 0.0, -1.0], (len(xy), 1)))
        z = np.full(len(xy), TABLE_TOP_Z)
        np.maximum.at(z, ray, loc[:, 2])
        return z

    n, pitch = chain["num_links"], chain["pitch"]
    centre = np.array(rec.xform)[:2]
    pins, scale = ellipse_pins(centre, HOLDER_SEMI_AXES, n, pitch)
    d = np.roll(pins, -1, 0) - pins
    edge_n = np.c_[d[:, 1], -d[:, 0]] / pitch  # outward for a CCW loop
    nrm = edge_n + np.roll(edge_n, 1, 0)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    off = np.linspace(-0.5 * PLATE_H - HOLDER_CLEARANCE, 0.5 * PLATE_H + HOLDER_CLEARANCE, 25)
    z = surface((pins[:, None, :] + off[None, :, None] * nrm[:, None, :]).reshape(-1, 2))
    z = z.reshape(n, len(off))
    zc = z[:, len(off) // 2]
    tier = float(zc.max())
    on = np.abs(zc - tier) < 1e-4
    up_ok = bool((z <= tier + 1e-4).all())
    outer = [off[(off > 0) & (z[k] < tier - 1e-4)] for k in np.flatnonzero(on)]
    edge = min((float(o.min()) for o in outer if len(o)), default=float("inf"))
    res = {
        "centre": centre.tolist(), "scale": scale,
        "semi_axes": [scale * a for a in HOLDER_SEMI_AXES], "tier_z": tier,
        "tier_above_table_mm": (tier - TABLE_TOP_Z) * 1e3,
        "center_z": tier + HOLDER_LIFT + T_PLATE, "pins_on_tier": int(on.sum()),
        "pins_bridging": int((~on).sum()), "up_step_clear": up_ok,
        "outer_down_step_mm": edge * 1e3,
        "literal": bool(on.all() and (np.abs(z - tier) < 1e-4).all()),
        "bridged": bool(up_ok and on.sum() >= n // 2),
        "scale_in_range": bool(0.90 <= scale <= 1.00),
        "pins": pins.tolist(),
    }
    res["fit"] = res["bridged"] and res["scale_in_range"]
    return res


def write_scene_start(fit: dict) -> None:
    """The fitted holder ellipse (or the board-corner ring) as the chain start of SCENE_FULL."""
    text = SCENE_FULL.read_text()
    if fit["fit"]:
        c, a = fit["centre"], fit["semi_axes"]
        block = (
            "  # bike_chain_task.directives.add_roller_chain (SolverMuJoCo). Start (V13-02b\n"
            "  # --fit-holder): a flat 64-gon chord-walked round the round belt's holder ellipse\n"
            f"  # x{fit['scale']:.4f} (long axis along y) on the holder's "
            f"{fit['tier_above_table_mm']:.0f} mm tier, bottom plate\n"
            f"  # {HOLDER_LIFT * 1e3:g} mm above it; {fit['pins_bridging']} pins bridge the "
            "holder's finger channels. center z is the\n"
            "  # inner-link frame z (outer links 1.25 mm lower).\n"
            "  - add_roller_chain:\n      name: bike_chain\n"
            f"      center: [{c[0]:.5f}, {c[1]:.5f}, {fit['center_z']:.5f}]\n"
            f"      semi_axes: [{a[0]:.5f}, {a[1]:.5f}]\n")
    else:
        block = (
            "  # bike_chain_task.directives.add_roller_chain (SolverMuJoCo). Start: no holder tier\n"
            "  # fits (V13-02b --fit-holder), so the flat ring 1 mm above the board's free corner\n"
            "  # (bike_chain_spike.yaml); center z is the inner-link frame z.\n"
            "  - add_roller_chain:\n      name: bike_chain\n"
            f"      {BOARD_RING}\n"
            "      radius: 0.09705983                # 64-gon circumradius "
            "0.009525 / (2 sin(pi/64))\n")
    new, k = re.subn(r"  # bike_chain_task\.directives\.add_roller_chain.*?(?=      num_links:)",
                     lambda _: block, text, flags=re.DOTALL)
    assert k == 1, "add_roller_chain block not found in the scene yaml"
    SCENE_FULL.write_text(new)


def scene_support(text: str) -> tuple[str, float | None]:
    """("holder", tier z) when the scene yaml's chain start is the holder ellipse, else board."""
    m = re.search(r"add_roller_chain:.*?center: \[[^\]]*, ([-0-9.e]+)\]\s*\n\s*semi_axes:", text,
                  re.DOTALL)
    return ("holder", float(m.group(1)) - HOLDER_LIFT - T_PLATE) if m else ("board", None)


class Spike:
    def __init__(self, test: str, args: argparse.Namespace, run_dir: Path) -> None:
        self.test, self.args = test, args
        self.viewer, self.frames = None, []
        builder = make_builder()
        self.info: dict = {}
        self.arm_coords: list[int] = []
        if test == "full":
            scene_info = _load(SCENE_FULL, builder, args.roller_radius, full=True,
                               plate_shape=args.plate_shape, rims=args.rims,
                               play=args.play, damping=args.damping,
                               frictionloss=args.frictionloss, mu=args.mu,
                               play_frictionloss=args.play_frictionloss,
                               play_slide=args.play_slide)
            self.bodies = np.asarray(scene_info.belt_bodies)
            self.chain_joints = list(scene_info.belt_joints)
            sprocket_bodies, self.sprocket_joints = (scene_info.pulley_bodies,
                                                     list(scene_info.pulley_joints))
            self.pitch, self.n = 0.009525, len(self.bodies)
            rollers = [s for s in scene_info.belt_shapes if "/roller_" in builder.shape_label[s]]
            self.support = scene_support(SCENE_FULL.read_text())
            # Board pose from the chain-only scene (same weld in both yamls).
            self.X_wb = np.array(_load(SCENE, make_builder(), args.roller_radius,
                                       full=False).models["board"].xform, dtype=np.float64)
        else:
            scene = _load(SCENE, builder, args.roller_radius, full=False,
                          plate_shape=args.plate_shape, rims=args.rims,
                          play=args.play, damping=args.damping, frictionloss=args.frictionloss,
                          mu=args.mu, play_frictionloss=args.play_frictionloss,
                          play_slide=args.play_slide)
            chain = scene.extras["bike_chain"]
            self.bodies = np.asarray(chain["bodies"])
            self.chain_joints = list(chain["joints"])
            board = scene.models["board"]
            self.X_wb = np.array(board.xform, dtype=np.float64)
            sprocket_bodies, self.sprocket_joints = board.bodies, list(board.joints)
            self.pitch, self.n = chain["pitch"], chain["num_links"]
            rollers = chain["roller_shapes"]
            self.support = ("board", None)
        self.sprocket_bodies = dict(zip(("small", "large", "tensioner"), sprocket_bodies))
        self._sprocket_geometry(builder)
        self.drive_offset = 0.0
        if test in ("seated", "drive"):
            self._pose_seated(builder)
        elif test == "rest":
            fit = fit_holder(args.roller_radius)
            self._set_chain_q(builder, *chain_joint_q(np.array(fit["pins"]), fit["center_z"]))
            self.support = ("holder", fit["tier_z"])
            self.info["holder_fit"] = {k: v for k, v in fit.items() if k != "pins"}
        builder.color()
        self.model = model = builder.finalize()
        mu = model.shape_material_mu.numpy()  # steel sprockets (URDF has no friction slot)
        mu[np.isin(model.shape_body.numpy(), list(self.sprocket_bodies.values()))] = args.sprocket_mu
        model.shape_material_mu.assign(mu)
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)  # body_q from the posed q
        sb = model.shape_body.numpy()
        self.roller_body = sb[rollers]
        self.roller_local = model.shape_transform.numpy()[rollers, :3]
        labels = list(model.joint_label)
        self.rev = [j for j in self.chain_joints if "/pin_" in labels[j]]
        qs = model.joint_q_start.numpy()
        self.tilt_q = [qs[j] for j in self.chain_joints if "/tilt_" in labels[j]]
        self.slide_q = [qs[j] for j in self.chain_joints if "/slide_" in labels[j]]
        self.Xp, self.Xc = model.joint_X_p.numpy()[self.rev], model.joint_X_c.numpy()[self.rev]
        self.jp, self.jc = model.joint_parent.numpy()[self.rev], model.joint_child.numpy()[self.rev]
        self.com = model.body_com.numpy()[self.bodies]

        dofs = model.joint_qd_start.numpy()[self.sprocket_joints]
        self.drive_dof = int(dofs[1])
        assert model.joint_q_start.numpy()[self.sprocket_joints[1]] == self.drive_dof
        self.control = None
        if test == "full":
            dirs = parse_directives(SCENE_FULL)

            def defaults(model_name, labels):
                pos = dirs.model(model_name).default_joint_positions
                return [float(pos[label.rsplit("/", 1)[-1]][0]) for label in labels]

            apply_default_joint_state(
                model, scene_info, arm_labels=PANDA_JOINT_LABELS + UR10_JOINT_LABELS,
                arm_defaults=defaults("panda_arm", PANDA_JOINT_LABELS)
                + defaults("ur10", UR10_JOINT_LABELS), finger_labels=PANDA_FINGER_LABELS,
                finger_defaults=defaults("panda_hand", PANDA_FINGER_LABELS))
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)
            self.arm_coords = list(scene_info.joint_config["arm_coord_indices"])
            self.control = model.control()
            BeltTaskSimulation._seed_control_targets(
                SimpleNamespace(control=self.control, model=model), scene_info.joint_config)
        if test == "drive":
            for arr, v in ((model.joint_target_ke, DRIVE_KE), (model.joint_target_kd, DRIVE_KD),
                           (model.joint_target_mode, int(newton.JointTargetMode.POSITION))):
                a = arr.numpy()
                a[self.drive_dof] = v
                arr.assign(a)
        mjcf = run_dir / ("model_full.xml" if test == "full" else "model.xml")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.solver = make_mujoco_solver(
                model, iterations=args.iterations, ls_iterations=args.ls_iterations,
                nconmax=args.nconmax, njmax=args.njmax, impratio=args.impratio,
                solref=(args.solref_timeconst, 1.0),
                save_to_mjcf=str(mjcf) if args.mjcf and not mjcf.exists() else None)
        self.warnings = [str(w.message)[:300] for w in caught]
        if self.control is None:
            self.control = model.control()
        mj = self.solver.mj_model
        self.mj_counts = {"nbody": int(mj.nbody), "nv": int(mj.nv), "neq": int(mj.neq),
                          "ngeom": int(mj.ngeom), "nu": int(mj.nu),
                          "naconmax": int(self.solver.mjw_data.naconmax),
                          "njmax": int(self.solver.mjw_data.njmax),
                          "geom_solref": np.unique(mj.geom_solref.round(5), axis=0).tolist()}
        self.anchor_err_mm = self._anchor_check() if test != "full" else None
        self.s0, self.s1 = model.state(), model.state()
        self.dt = 1.0 / FPS / args.substeps
        self.graph = None
        self.peak = {"nacon": 0, "nefc": 0}
        self.q_sprocket0 = self.s0.body_q.numpy()[list(self.sprocket_bodies.values())]
        self.arm_q0 = self.s0.joint_q.numpy()[self.arm_coords]
        self.target = self.control.joint_target_q.numpy()

    def _set_chain_q(self, builder, free_q: np.ndarray, angles: np.ndarray) -> None:
        qs = builder.joint_q_start
        free = self.chain_joints[0]
        pins = [j for j in self.chain_joints if "/pin_" in builder.joint_label[j]]
        builder.joint_q[qs[free]:qs[free] + 7] = [float(v) for v in free_q]
        for j, a in zip(pins, angles):
            builder.joint_q[qs[j]] = float(a)

    def _anchor_check(self) -> float:
        """Max |child-side CONNECT anchor - pin 63's joint_X_c (+0.1 m along the axis)| [mm]."""
        eq = self.solver.mjw_model.eq_data.numpy()[0][:, 3:6]
        Xc = self.Xc[-1]
        axis = self.model.joint_axis.numpy()[self.model.joint_qd_start.numpy()[self.rev[-1]]]
        want = np.array([Xc[:3], Xc[:3] + 0.1 * Rotation.from_quat(Xc[3:7]).apply(axis)])
        return float(np.abs(eq - want).max() * 1e3) if eq.shape == want.shape else float("inf")

    def _sprocket_geometry(self, builder) -> None:
        """Rim z, pitch radius, tooth transforms (sprocket frame) per sprocket from the URDF."""
        self.geo = {}
        flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for name, body in self.sprocket_bodies.items():
            shapes = [s for s in range(builder.shape_count) if builder.shape_body[s] == body
                      and builder.shape_flags[s] & flag]
            teeth = [s for s in shapes if builder.shape_type[s] == newton.GeoType.MESH]
            tooth_x = np.array([list(builder.shape_transform[s]) for s in teeth])
            # URDF labels are generic: the rim is the cylinder nearest the teeth's base radius.
            cyl = [s for s in shapes if builder.shape_type[s] == newton.GeoType.CYLINDER]
            rims = [min(cyl, key=lambda s: abs(builder.shape_scale[s][0]
                                               - np.hypot(*tooth_x[0, :2])))]
            n = len(teeth)
            self.geo[name] = {
                "rim_z": float(builder.shape_transform[rims[0]][2]),
                "rim_r": float(builder.shape_scale[rims[0]][0]),
                "teeth": n, "pitch_r": self.pitch / (2 * math.sin(math.pi / n)),
                "az0": math.degrees(math.atan2(tooth_x[0, 1], tooth_x[0, 0])),
                "tooth_x": tooth_x,
            }

    def _pose_seated(self, builder) -> None:
        g, n, pitch = self.geo, self.n, self.pitch
        Rwb = Rotation.from_quat(self.X_wb[3:7])
        centre = {k: Rwb.inv().apply(np.array(builder.body_q[b])[:3] - self.X_wb[:3])
                  for k, b in self.sprocket_bodies.items()}
        S, L, T = (centre[k][:2] for k in ("small", "large", "tensioner"))
        tp = 2 * math.pi / g["small"]["teeth"]
        gap0 = math.radians(g["small"]["az0"]) + 0.5 * tp
        k_gap = round((math.pi - gap0) / tp)  # the gap nearest the small sprocket's -x side
        tip_t = g["tensioner"]["rim_r"] + 0.0049 + self.args.roller_radius + 0.001
        pins, h = seated_pins(S, L, T, g["small"]["pitch_r"], g["large"]["pitch_r"], tip_t, n,
                              pitch, gap0 + k_gap * tp)
        on = {k: np.abs(np.linalg.norm(pins - c, axis=1) - g[k]["pitch_r"]) < 1e-5
              for k, c in (("small", S), ("large", L))}
        # Large phase: its joint_q puts a gap under the walked pins on its arc.
        tp_l = 360.0 / g["large"]["teeth"]
        az = np.degrees(np.arctan2(*(pins[on["large"]] - L).T[::-1]))
        resid = (az - g["large"]["az0"]) % tp_l - 0.5 * tp_l  # pin az - gap az
        self.drive_offset = math.radians(float(np.median(resid)))
        builder.joint_q[builder.joint_q_start[self.sprocket_joints[1]]] = self.drive_offset
        # Planar chain: plane z = a + b x (board frame) fitted to both tooth planes.
        zt = {k: centre[k][2] + g[k]["rim_z"] - (T_PLATE + 0.5 * W_INNER) for k in SPROCKETS}
        sel = on["small"] | on["large"]
        b, a = np.polyfit(pins[sel, 0], np.where(on["small"], zt["small"], zt["large"])[sel], 1)
        R = Rwb * Rotation.from_euler("y", -math.atan(b))
        frame = np.r_[self.X_wb[:3] + Rwb.apply([0.0, 0.0, a]), R.as_quat()]
        self._set_chain_q(builder, *chain_joint_q(pins, 0.0, frame))
        dev = {k: float(np.abs(a + b * pins[on[k], 0] - zt[k]).max() * 1e3) for k in SPROCKETS}
        self.info = {"kink_depth_mm": h * 1e3, "large_phase_deg": math.degrees(self.drive_offset),
                     "pins_on_small": int(on["small"].sum()), "pins_on_large": int(on["large"].sum()),
                     "plane_tilt_deg": math.degrees(math.atan(b)), "plane_dev_mm": dev}

    def _frame(self) -> None:
        for _ in range(self.args.substeps):
            self.solver.step(self.s0, self.s1, self.control, None, self.dt)
            self.s0, self.s1 = self.s1, self.s0

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
            return
        self._frame()
        if self.args.cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture() as cap:
                self._frame()  # substeps even: the swap restores s0/s1
            self.graph = cap.graph

    def metrics(self, t: float) -> dict:
        q = self.s0.body_q.numpy().astype(np.float64)
        nan = not np.isfinite(q).all()
        bad = ~np.isfinite(q).all(axis=1) | (np.linalg.norm(q[:, 3:7], axis=1) < 1e-9)
        q[bad] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # blown up: score a FAIL, don't raise
        qc = q[self.bodies]
        gaps = np.linalg.norm(_tf(q[self.jp], self.Xp[:, :3]) - _tf(q[self.jc], self.Xc[:, :3]),
                              axis=1) * 1e3
        com = _tf(qc, self.com).mean(axis=0)
        if self.support[0] == "holder":
            min_h = float((qc[:, 2] - self.support[1]).min() * 1e3)
        else:
            up = Rotation.from_quat(self.X_wb[3:7]).apply([0.0, 0.0, 1.0])
            min_h = float(((qc[:, :3] - self.X_wb[:3]) @ up).min() * 1e3)
        d = self.solver.mjw_data
        nacon, nefc = int(d.nacon.numpy()[0]), int(d.nefc.numpy().max())
        pen = float(max(0.0, -d.contact.dist.numpy()[:nacon].min() * 1e3)) if nacon else 0.0
        self.peak = {"nacon": max(self.peak["nacon"], nacon),
                     "nefc": max(self.peak["nefc"], nefc)}
        out = {"t": round(t, 4), "nan": nan,
               "gap_p95_mm": float(np.percentile(gaps, 95)), "gap_max_mm": float(gaps.max()),
               "closure_gap_mm": float(gaps[-1]), "com": com.tolist(), "min_h_support_mm": min_h,
               "nacon": nacon, "nefc": nefc, "pen_max_mm": pen,
               "w_rms": float(np.sqrt((np.nan_to_num(
                   self.s0.body_qd.numpy()[self.bodies, 3:6]) ** 2).sum(1).mean()))}
        if self.tilt_q:
            jq = self.s0.joint_q.numpy()
            out["tilt_max_rad"] = float(np.abs(jq[self.tilt_q]).max())
            out["slide_max_mm"] = float(np.abs(jq[self.slide_q]).max() * 1e3)
            out["play_at_limit"] = int((np.abs(jq[self.tilt_q]) > 0.95 * 0.02).sum()
                                       + (np.abs(jq[self.slide_q]) > 0.95 * 0.0005).sum())
        if self.arm_coords:
            dq = self.s0.joint_q.numpy()[self.arm_coords] - self.arm_q0
            out["arm_drift_rad"] = float(np.abs(dq).max())
        if self.test == "drive":
            out["drive_target_rad"] = DRIVE_RATE * t
        rollers = _tf(q[self.roller_body], self.roller_local)
        for name in SPROCKETS:
            g, b = self.geo[name], self.sprocket_bodies[name]
            h, r, az = belt_in_pulley_frame(rollers, q[b])
            h = h - g["rim_z"] * 1e3
            band = (np.abs(r - g["pitch_r"] * 1e3) <= BAND_R_MM) & (np.abs(h) <= BAND_H_MM)
            tp = 360.0 / g["teeth"]
            off = np.abs((az - g["az0"]) % tp - 0.5 * tp)
            meshed = band & (off <= MESH_TOL_DEG) & (np.abs(h) <= 2.0)
            rel = Rotation.from_quat(self.q_sprocket0[list(self.sprocket_bodies).index(name), 3:7]
                                     ).inv() * Rotation.from_quat(q[b, 3:7])
            out[name] = {"band": int(band.sum()), "meshed": int(meshed.sum()),
                         "frac": float(meshed.sum() / band.sum()) if band.any() else float("nan"),
                         "h_abs_max_mm": float(np.abs(h[band]).max()) if band.any() else 0.0,
                         "angle_rad": float(rel.as_rotvec() @ [0.0, 0.0, 1.0]),
                         "tooth_dist_min_mm": self._tooth_dist(rollers[band], q[b], g)}
        return out

    def _render(self, t: float) -> None:
        self.viewer.begin_frame(t)
        self.viewer.log_state(self.s0)
        self.viewer.end_frame()

    def hold(self) -> None:
        """Loop the rendered frames in the viewer until Ctrl-C or the viewer closes."""
        period = self.args.render_every / FPS
        try:
            while self.viewer.is_running():
                for k, q in enumerate(self.frames):
                    self.s0.body_q.assign(q)
                    self._render(k * period)
                    time.sleep(period)
        except KeyboardInterrupt:
            pass

    def _tooth_dist(self, pts: np.ndarray, pose: np.ndarray, g: dict) -> float:
        """Min roller-surface to tooth-mesh distance [mm] (negative = roller inside a tooth)."""
        if not len(pts):
            return float("nan")
        local = Rotation.from_quat(pose[3:7]).inv().apply(pts - pose[:3])
        tx = g["tooth_x"]
        d_min = np.inf
        az_t = np.arctan2(tx[:, 1], tx[:, 0])
        for p in local:
            near = np.argsort(np.abs(np.angle(np.exp(1j * (az_t - math.atan2(p[1], p[0]))))))[:2]
            for k in near:
                lp = Rotation.from_quat(tx[k, 3:7]).inv().apply(p - tx[k, :3])
                sd = trimesh.proximity.signed_distance(TOOTH, lp[None])[0]
                d_min = min(d_min, -sd - self.args.roller_radius)
        return float(d_min * 1e3)

    def run(self) -> dict:
        frames = round(DURATION[self.test] * FPS)
        dof = self.drive_dof
        samples, step_s, t0 = [self.metrics(0.0)], 0.0, time.perf_counter()
        for f in range(1, frames + 1):
            if self.test == "drive":
                self.target[dof] = self.drive_offset + DRIVE_RATE * f / FPS
                self.control.joint_target_q.assign(self.target)
            ts = time.perf_counter()
            self.step()
            wp.synchronize_device(self.model.device)
            if f > 1:  # frame 1 is the uncaptured warm-up + capture
                step_s += time.perf_counter() - ts
            if self.viewer is not None and f % self.args.render_every == 0:
                q = self.s0.body_q.numpy()
                if np.isfinite(q).all():  # a blown-up state is never recorded
                    self.frames.append(q)
                    self._render(f / FPS)
            if f % SAMPLE_EVERY == 0:
                samples.append(self.metrics(f / FPS))
                if samples[-1]["nan"]:
                    break
        wall = time.perf_counter() - t0
        a = self.args
        return {"test": self.test, "settings": settings(a), "info": self.info,
                "cuda_graph": self.graph is not None, "device": str(self.model.device),
                "chain_bodies": len(self.bodies), "chain_joints": len(self.rev),
                "mj": self.mj_counts, "peak": self.peak, "warnings": self.warnings,
                "anchor_err_mm": self.anchor_err_mm, "support": list(self.support),
                "rtf_step": (frames - 1) / FPS / max(step_s, 1e-9),
                "rtf_wall": samples[-1]["t"] / wall, "samples": samples}


TOOTH = trimesh.load(ASSETS / "sprocket_tooth.obj", force="mesh")


def settings(a: argparse.Namespace) -> dict:
    return {"iterations": a.iterations, "ls_iterations": a.ls_iterations, "nconmax": a.nconmax,
            "njmax": a.njmax, "impratio": a.impratio, "roller_radius": a.roller_radius,
            "substeps": a.substeps, "solref_timeconst": a.solref_timeconst,
            "plates": a.plate_shape, "rims": a.rims, "play": a.play, "damping": a.damping,
            "frictionloss": a.frictionloss, "mu": a.mu, "sprocket_mu": a.sprocket_mu,
            "play_frictionloss": a.play_frictionloss, "play_slide": a.play_slide, "teeth": "mesh", "closure": "pin63 loop joint (2 CONNECT)", "contacts": "mujoco"}


def run_name(a: argparse.Namespace) -> str:
    s = settings(a)
    return (f"i{s['iterations']}_ls{s['ls_iterations']}_imp{s['impratio']:g}"
            f"_tc{s['solref_timeconst']:g}_r{s['roller_radius'] * 1e3:g}_s{s['substeps']}"
            + ("_box" if a.plate_shape == "box" else "")
            + ("_rootrims" if a.rims == "root" else "") + ("_play" if a.play else "")
            + (f"_d{a.damping:g}" if a.damping else "")
            + (f"_f{a.frictionloss:g}" if a.frictionloss != 1.0e-3 else "")
            + (f"_rate{a.drive_rate:g}" if getattr(a, "drive_rate", DRIVE_RATE) != DRIVE_RATE
               else "") + (f"_mu{a.mu:g}" if getattr(a, "mu", None) is not None else "")
            + (f"_smu{a.sprocket_mu:g}" if getattr(a, "sprocket_mu", 0.15) != 0.15 else "")
            + (f"_pf{a.play_frictionloss:g}" if getattr(a, "play_frictionloss", 0.0) else "")
            + ("_tiltonly" if a.play and not getattr(a, "play_slide", True) else ""))


def _jitter(r: list) -> float:
    """Mean link |angular velocity| RMS over the samples of the last 1 s [rad/s]."""
    late = [s.get("w_rms", float("nan")) for s in r if s["t"] >= r[-1]["t"] - 1.0]
    return float(np.mean(late))


def _rest(run: dict) -> tuple[bool, str]:
    """S2 bounds on one rest run: pin gaps, COM drift over the last 2 s, min height."""
    r = run["samples"]
    gmax = max(s["gap_max_mm"] for s in r)
    g95 = max(s["gap_p95_mm"] for s in r)
    late = [s for s in r if s["t"] >= r[-1]["t"] - 2.0]
    drift = max(np.linalg.norm(np.subtract(s["com"][:2], late[0]["com"][:2])) for s in late)
    min_h = min(s["min_h_support_mm"] for s in r)
    ok = gmax < 0.5 and g95 < 0.2 and drift * 1e3 < 5.0 and min_h > -2.0 and not r[-1]["nan"]
    return ok, (f"{run['support'][0]}: jitter |w| rms (last 1 s) {_jitter(r):.3f} rad/s, gap max {gmax:.3f} mm (closure "
                f"{max(s['closure_gap_mm'] for s in r):.3f}), p95 {g95:.3f} mm, COM xy drift "
                f"(last 2 s) {drift * 1e3:.2f} mm, min height above support {min_h:.2f} mm")


def score(runs: dict[str, dict], checks: dict) -> dict[str, tuple[bool, str]]:
    """S1-S6 pass/fail + the measured numbers for one settings directory."""
    res = {}
    if "rest" in runs:
        r = runs["rest"]
        first = [s for s in r["samples"] if s["t"] <= 1.0]
        nan = any(s["nan"] for s in first)
        m = r["mj"]
        nv = 6 + 63 * ((3 if r["settings"].get("play_slide", True) else 2)
                       if r["settings"].get("play") else 1) + 3
        ok = (r["chain_bodies"] == 64 and r["chain_joints"] == 64 and m["neq"] == 2
              and m["nv"] == nv and r["anchor_err_mm"] < 0.1 and "cuda" in r["device"]
              and not nan and checks.get("model_xml", False)
              and all(checks.get(k) is not False for k in ("directives", "belt_pin")))
        res["S1"] = (ok, (f"{r['chain_bodies']} bodies / {r['chain_joints']} pins, neq {m['neq']}, "
                          f"nv {m['nv']}, anchor-2 err {r['anchor_err_mm']:.2e} mm, model.xml "
                          f"{checks.get('model_xml')}, NaN in 1 s: {nan}, "
                          f"check_scene_directives {checks.get('directives')}, "
                          f"check_belt_pin {checks.get('belt_pin')}"))
    rest = [_rest(runs[t]) for t in ("rest", "rest_board") if t in runs]
    if rest:
        res["S2"] = (all(ok for ok, _ in rest), "; ".join(txt for _, txt in rest))
    for test, key in (("seated", "S3"), ("drive", "S4")):
        if test not in runs:
            continue
        r = runs[test]["samples"]
        last = [s for s in r if s["t"] >= r[-1]["t"] - 1.0]
        parts, ok = [], not any(s["nan"] for s in r)
        for name in SPROCKETS:
            frac = float(np.nan_to_num([s[name]["frac"] for s in last]).min())
            band0, band1 = r[0][name]["band"], min(s[name]["band"] for s in last)
            h = max(s[name]["h_abs_max_mm"] for s in last)
            dist = min(s[name]["tooth_dist_min_mm"] for s in last)
            ok &= frac >= 0.9 and h < 2.0 and dist > -1.0 and band1 >= band0 - 1
            parts.append(f"{name}: meshed {frac:.0%} (min, last 1 s), band {band0}->{band1}, "
                         f"|h| max {h:.2f} mm, tooth dist min {dist:.2f} mm")
        parts.append(f"jitter |w| rms (last 1 s) {_jitter(r):.3f} rad/s")
        parts.append(f"gap max {max(s['gap_max_mm'] for s in r):.3f} mm, contact penetration "
                     f"max {max(s.get('pen_max_mm', 0.0) for s in r):.2f} mm")
        if "tilt_max_rad" in r[0]:
            parts.append(f"play used: tilt max {max(s['tilt_max_rad'] for s in r):.4f} rad, "
                         f"slide max {max(s['slide_max_mm'] for s in r):.3f} mm, DOFs at limit "
                         f"max {max(s['play_at_limit'] for s in r)}")
        if test == "drive":
            ang = {k: np.unwrap([s[k]["angle_rad"] for s in r]) for k in SPROCKETS}  # > pi runs
            a_s, a_l = (ang[k][-1] - ang[k][0] for k in SPROCKETS)
            ratio = a_s / a_l if abs(a_l) > 1e-6 else float("nan")
            lag = max(abs(s["drive_target_rad"] - q) for s, q in zip(r, ang["large"]))
            ok &= abs(ratio - 2.0) <= 0.2 and lag < 0.2
            parts.append(f"large {a_l:.3f} rad, small {a_s:.3f} rad, ratio {ratio:.2f}, "
                         f"drive lag max {lag:.3f} rad")
        res[key] = (bool(ok), "; ".join(parts))
    chain_only = {t: run for t, run in runs.items() if t != "full"}
    if chain_only:
        blown = [t for t, run in chain_only.items() if run["samples"][-1]["nan"]]
        res["S5"] = (min(run["rtf_step"] for run in chain_only.values()) >= 0.5 and not blown,
                     "RTF (stepping, CUDA graph) " + ", ".join(
                         f"{t} {run['rtf_step']:.2f}" for t, run in chain_only.items())
                     + "".join(f"; {t} NaN at {chain_only[t]['samples'][-1]['t']} s"
                               for t in blown)
                     + "; peak contacts " + ", ".join(
                         f"{t} {run['peak']['nacon']}/{run['mj']['naconmax']}"
                         for t, run in chain_only.items()))
    if "full" in runs:
        r = runs["full"]
        drift = max(s["arm_drift_rad"] for s in r["samples"])
        ok2, txt = _rest(r)
        res["S6"] = (r["rtf_step"] >= 0.25 and drift < 1e-3 and ok2,
                     (f"RTF {r['rtf_step']:.2f} (target 0.5, accept 0.25), arm drift max "
                      f"{drift:.2e} rad, peak contacts {r['peak']['nacon']}/"
                      f"{r['mj']['naconmax']}, nefc {r['peak']['nefc']}/{r['mj']['njmax']}, "
                      f"nv {r['mj']['nv']}, neq {r['mj']['neq']}; {txt}"))
    return res


ACCEPTED = (
    "Accepted default (coordinator, 2026-10-22): i30_ls10_imp1_tc0.02_r2.5_s10 = aligned board "
    "(bike_chain_task_board_aligned.urdf), magna-exact planar chain (stadium plates, rollers "
    "2.5 x 4.5 mm, 4.5 mm gap, no play, 0.20 kg = 3.125 g/link; the 0.32 kg run is "
    "..._m0.32), magna rims, pin frictionloss 1e-3 N m (no damping); "
    "chain and sprocket contact mu 0.15 (lubricated steel; at 1.0 rollers are carried ~20 deg "
    "past the tangent); S4 = 15 s drive at -0.5 rad/s (tensioner on the slack side). Known "
    "limitations: closure gap "
    "~1.0 mm under drive (1.9 mm at mu 1.0; soft 2-CONNECT loop closure); S5 RTF 0.43-0.63. The old S4 (2 s at "
    "+0.5 rad/s, frictionloss 1e-4: run ..._f0.0001_rate0.5, band 9 -> 6, ratio 1.74) masked a "
    "NaN at ~2.05 s. Runs without an _f/_rate suffix older than this note used frictionloss 0 "
    "and the 2 s +0.5 rad/s drive. Root rims: 0.05 mm clearance NaNs at 0.5 s; 0.15 mm passes "
    "S3, S4 band still 9 -> 6.")


def report(out: Path) -> None:
    """``spike.md`` (one table) and ``spike.html`` from every run directory under ``out``."""
    runs, checks = {}, {}
    for d in sorted(out.iterdir()):
        if d.is_dir():
            runs[d.name] = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))
                            if p.stem in TESTS}
            c = d / "checks.json"
            checks[d.name] = json.loads(c.read_text()) if c.exists() else {}
            checks[d.name]["model_xml"] = (d / "model.xml").exists()
    scores = {name: score(r, checks[name]) for name, r in runs.items() if r}
    chosen = next((n for n, s in scores.items()
                   if all(s.get(k, (False,))[0] for k in ("S1", "S2", "S3", "S4", "S5"))), None)
    if chosen:
        st = next(iter(runs[chosen].values()))["settings"]
        head = ("First run passing S1-S5 (reference): " + chosen + " -- " + ", ".join(f"{k} {v}" for k, v in st.items()))
    else:
        head = "No run passes S1-S5"
    lines = [f"# V13-02b chain spike (SolverMuJoCo)\n\n{ACCEPTED}\n\n{head}.\n",
             "| run | criterion | pass | measured |", "|---|---|---|---|"]
    for name, s in scores.items():
        for key in sorted(s):
            lines.append(f"| {name} | {key} | {'PASS' if s[key][0] else 'FAIL'} | {s[key][1]} |")
    (out.parent / "spike.md").write_text("\n".join(lines) + "\n")
    series = {f"{n}/{t}": [{"t": s["t"], "gap_max_mm": s["gap_max_mm"],
                            "closure_gap_mm": s["closure_gap_mm"],
                            "min_h_support_mm": s["min_h_support_mm"], "nacon": s["nacon"],
                            "com_x_mm": s["com"][0] * 1e3, "com_y_mm": s["com"][1] * 1e3,
                            **{f"{k}_meshed_frac": s[k]["frac"] for k in SPROCKETS},
                            **{f"{k}_angle_rad": s[k]["angle_rad"] for k in SPROCKETS},
                            **({"arm_drift_rad": s["arm_drift_rad"]} if "arm_drift_rad" in s
                               else {})}
                           for s in run["samples"]] for n, r in runs.items() for t, run in r.items()}
    rtf = {f"{n}/{t}": run["rtf_step"] for n, r in runs.items() for t, run in r.items()}
    (out.parent / "spike.html").write_text(HTML.replace("__DATA__", json.dumps(series)).replace(
        "__RTF__", json.dumps(rtf)).replace("__MD__", "\n".join(lines).replace("<", "&lt;")))


HTML = """<!doctype html><meta charset="utf-8"><title>V13-02b chain spike (SolverMuJoCo)</title>
<body style="font-family:sans-serif;max-width:1100px">
<h2>V13-02b chain spike (SolverMuJoCo)</h2><pre style="white-space:pre-wrap;font-size:12px">__MD__</pre>
<p>Metric <select id="m"></select> runs: <span id="runs"></span></p>
<svg id="plot" width="1000" height="360" style="border:1px solid #ccc"></svg>
<h3>Real-time factor (stepping, CUDA graph)</h3><svg id="rtf" width="1000"></svg>
<script>
const D=__DATA__, R=__RTF__, keys=Object.keys(D), cols=["#36c","#c33","#3a3","#c90","#93c",
"#099","#666","#e6a","#8a5","#05a","#a50","#5a0"];
const metrics=[...new Set(keys.flatMap(k=>Object.keys(D[k][0])))].filter(k=>k!="t"), on={};
const sel=document.getElementById("m"); metrics.forEach(m=>sel.add(new Option(m,m)));
const rs=document.getElementById("runs");
keys.forEach((k,i)=>{on[k]=true;const l=document.createElement("label");
 l.style.color=cols[i%cols.length];l.innerHTML=`<input type=checkbox checked> ${k} `;
 l.firstChild.onchange=e=>{on[k]=e.target.checked;draw()};rs.appendChild(l)});
function draw(){const m=sel.value,W=1000,H=360,P=40,s=document.getElementById("plot");
 let tx=0,lo=Infinity,hi=-Infinity;keys.filter(k=>on[k]).forEach(k=>D[k].forEach(p=>{
  tx=Math.max(tx,p.t);if(isFinite(p[m])&&p[m]!==null){lo=Math.min(lo,p[m]);hi=Math.max(hi,p[m])}}));
 if(hi==lo){hi+=1;lo-=1} const X=t=>P+t/tx*(W-2*P),Y=v=>H-P-(v-lo)/(hi-lo)*(H-2*P);
 let g=`<text x=5 y=15 font-size=12>${hi.toPrecision(4)}</text><text x=5 y=${H-P} font-size=12>`+
  `${lo.toPrecision(4)}</text><text x=${W-P} y=${H-5} font-size=12>${tx}s</text>`;
 keys.forEach((k,i)=>{if(!on[k])return;const pts=D[k].filter(p=>p[m]!==null&&isFinite(p[m]))
  .map(p=>X(p.t).toFixed(1)+","+Y(p[m]).toFixed(1)).join(" ");
  g+=`<polyline points="${pts}" fill=none stroke="${cols[i%cols.length]}" stroke-width=1.5>`+
   `<title>${k}</title></polyline>`});s.innerHTML=g}
sel.onchange=draw;draw();
const rt=document.getElementById("rtf"),ks=Object.keys(R),mx=Math.max(1,...Object.values(R));
rt.setAttribute("height",ks.length*18+10);rt.innerHTML=ks.map((k,i)=>
 `<rect x=260 y=${i*18+4} width=${R[k]/mx*700} height=14 fill="${R[k]>=0.5?'#3a3':'#c33'}">`+
 `<title>${R[k].toFixed(3)}</title></rect><text x=0 y=${i*18+15} font-size=12>${k}</text>`+
 `<text x=${265+R[k]/mx*700} y=${i*18+15} font-size=12>${R[k].toFixed(2)}</text>`).join("")+
 `<line x1=${260+0.5/mx*700} x2=${260+0.5/mx*700} y1=0 y2=${ks.length*18+10} stroke=#000 `+
 `stroke-dasharray=4><title>0.5x</title></line>`;
</script></body>
"""


def _check(script: str) -> bool:
    return subprocess.run([sys.executable, str(ROOT / "scripts/checks" / script)], check=False,
                          capture_output=True).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", choices=(*TESTS, "all"), default="all")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--ls-iterations", type=int, default=10)
    ap.add_argument("--nconmax", type=int, default=2048)
    ap.add_argument("--njmax", type=int, default=8192)
    ap.add_argument("--impratio", type=float, default=1.0)
    ap.add_argument("--solref-timeconst", type=float, default=0.02,
                    help="contact solref time constant for every shape (MuJoCo default 0.02 s)")
    ap.add_argument("--substeps", type=int, default=10)
    ap.add_argument("--roller-radius", type=float, default=0.0025)
    ap.add_argument("--play", action=argparse.BooleanOptionalAction, default=False,
                    help="magna's per-pitch tilt/slide play DOFs")
    ap.add_argument("--drive-rate", type=float, default=DRIVE_RATE,
                    help="drive test large-sprocket rate [rad/s]; < 0 puts the tensioner on the "
                         "slack side")
    ap.add_argument("--drive-seconds", type=float, default=None,
                    help=f"drive test length [s] (default {DURATION['drive']:g}, the S4 length)")
    ap.add_argument("--mu", type=float, default=None,
                    help="chain contact friction (default: the directive's CHAIN_CONTACT_MU)")
    ap.add_argument("--sprocket-mu", type=float, default=0.15,
                    help="sprocket shapes' contact friction (MuJoCo pairs use max(mu))")
    ap.add_argument("--damping", type=float, default=0.0, help="pin damping [N m s/rad]")
    ap.add_argument("--frictionloss", type=float, default=1.0e-3, help="pin dry friction [N m]")
    ap.add_argument("--play-frictionloss", type=float, default=0.0,
                    help="dry friction on the play DOFs (with --play)")
    ap.add_argument("--play-slide", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-play-slide: tilt-only play")
    ap.add_argument("--rims", choices=("magna", "root"), default="magna",
                    help="sprocket rims: magna's SDF radii, or pitch radius - 2.55 mm")
    ap.add_argument("--plate-shape", choices=("mesh", "box"), default="mesh",
                    help="chain plates: magna's stadium mesh or the V13-02b boxes")
    ap.add_argument("--mjcf", action=argparse.BooleanOptionalAction, default=True,
                    help="write <out>/<run>/model.xml (save_to_mjcf)")
    ap.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false")
    ap.add_argument("--viewer", choices=("viser",), default=None,
                    help="watch the --test live in Newton's viser viewer (one test, not 'all')")
    ap.add_argument("--viewer-port", type=int, default=8083)
    ap.add_argument("--render-every", type=int, default=2, help="frames per rendered frame")
    ap.add_argument("--record-viser", type=Path, default=None,
                    help="save the --test as a .viser timeline (scripts/debug/play_viser.py)")
    ap.add_argument("--fit-holder", action="store_true",
                    help="CPU: fit the holder-ellipse start and write it into bike_chain_scene.yaml")
    ap.add_argument("--out", type=Path, default=None,
                    help="default data/bike_chain/v13_02b/spike; --viewer/--record-viser runs: "
                         "data/bike_chain/v13_02b/view/spike (never the official report)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    globals()["DRIVE_RATE"] = args.drive_rate
    if args.drive_seconds is not None:
        DURATION["drive"] = args.drive_seconds
    if args.out is None:
        sub = "view" if args.viewer or args.record_viser else "."
        args.out = ROOT / "data/bike_chain/v13_02b" / sub / "spike"
    if (args.viewer or args.record_viser) and args.out.resolve() == (
            ROOT / "data/bike_chain/v13_02b/spike").resolve():
        raise SystemExit("viewer/recording runs must not write the official spike report")
    if args.substeps < 2 or args.substeps % 2:
        raise SystemExit("--substeps must be even (graph capture swaps state twice per frame)")
    if (args.viewer or args.record_viser) and args.test == "all":
        raise SystemExit("--viewer / --record-viser need one --test "
                         "(rest, rest_board, seated, drive or full)")
    viewer = None
    if args.viewer or args.record_viser:
        import newton.viewer

        from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material

        with socket.socket() as sock:
            if args.viewer:  # viser silently moves to the next free port
                if sock.connect_ex(("127.0.0.1", args.viewer_port)) == 0:
                    raise SystemExit(f"--viewer-port {args.viewer_port} is in use")
                port = args.viewer_port
            else:  # record only: any free port, nobody connects
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        patch_viewer_shape_names()
        patch_viser_texture_material()
        if args.record_viser:
            args.record_viser.parent.mkdir(parents=True, exist_ok=True)
        viewer = newton.viewer.ViewerViser(
            port=port, label=f"bike chain {args.test}",
            record_to_viser=str(args.record_viser) if args.record_viser else None)
    if args.fit_holder:
        fit = fit_holder(args.roller_radius)
        write_scene_start(fit)
        print(json.dumps({k: v for k, v in fit.items() if k != "pins"}, indent=1))
        return 0
    run_dir = args.out / run_name(args)
    if not args.report_only:
        run_dir.mkdir(parents=True, exist_ok=True)
        for test in (TESTS if args.test == "all" else (args.test,)):
            spike = Spike(test, args, run_dir)
            if viewer is not None:
                spike.viewer = viewer
                viewer.set_model(spike.model)
                spike._render(0.0)
            res = spike.run()
            if args.record_viser:
                viewer.save_recording()
            (run_dir / f"{test}.json").write_text(json.dumps(res, indent=1))
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            summary[test] = {k: v for k, v in res.items() if k != "samples"}
            summary_path.write_text(json.dumps(summary, indent=1))
            last = res["samples"][-1]
            print(f"[{test}] t {last['t']} rtf {res['rtf_step']:.2f} gap max "
                  f"{last['gap_max_mm']:.3f} mm small {last['small']['frac']:.2f} large "
                  f"{last['large']['frac']:.2f} contacts peak {res['peak']['nacon']}", flush=True)
        if args.test == "all":
            checks = {"directives": _check("check_scene_directives.py"),
                      "belt_pin": _check("check_belt_pin.py")}
            (run_dir / "checks.json").write_text(json.dumps(checks))
    report(args.out)
    print((args.out.parent / "spike.md").read_text())
    if viewer is not None and not args.viewer:
        viewer.close()
    elif viewer is not None:
        print(f"Looping {len(spike.frames)} frames on port {args.viewer_port}; Ctrl-C to quit.",
              flush=True)
        spike.hold()
        viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
