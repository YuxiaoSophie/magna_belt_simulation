"""Clearance between the UR 2F-85 gripper and the task board's collision geometry.

The board's colliders are a plate box (``board/board/collision0``, top at board z = 0) and the two
pulleys (grooved discs on ``board/*_round_pulley``); every other raised board feature is
visual-only and sits > 40 mm from the gripper's path. The gripper's colliders are the 2F-85 mesh
shapes, of which only the ALOHA fingers, followers and spring links can ever be the closest
(:data:`PRUNE_MARGIN`).

Two ways to ask the same question:

* :meth:`Gripper.predict` -- the lowest point of the gripper frozen in the UR ``tracking_frame``,
  evaluated at a commanded pose. Cheap, needs no sim state, so it guards a *waypoint* before the
  episode runs (:func:`clamp_waypoints`).
* :meth:`Gripper.measure` -- every collider placed by its own measured body pose. This is the
  truth during playback; negative means the collision geometries overlap, i.e. real contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import newton
import numpy as np

from round_belt_task.arm_kinematics import wp_transform_to_mat4

BOARD_PLATE_SHAPE = "board/board/collision0"
GRIPPER_MODEL = "robotiq_2f85"
PRUNE_MARGIN = 0.08
PROFILE_BINS = 24
DEFAULT_MIN_CLEARANCE_MM = 4.0
CLAMP_ITERS = 10
CLAMP_TOL_M = 1e-5
MAX_CLAMP_LIFT_M = 0.05
PATH_STRIDE = 4
PATH_CHUNK = 64


def _quat_mat(q) -> np.ndarray:
    """``xyzw`` quaternion -> 3x3 (same convention as ``arm_kinematics``)."""
    return wp_transform_to_mat4([0.0, 0.0, 0.0, *q])[:3, :3]


@dataclass(frozen=True)
class Box:
    """An oriented box; :meth:`distance` is its exact signed distance."""

    rot: np.ndarray
    pos: np.ndarray
    half: np.ndarray

    def distance(self, points: np.ndarray) -> np.ndarray:
        q = np.abs((np.asarray(points, float) - self.pos) @ self.rot) - self.half
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)


@dataclass(frozen=True)
class Revolved:
    """A solid of revolution about its own z: ``profile`` is the closed ``(r, z)`` section."""

    rot: np.ndarray
    pos: np.ndarray
    profile: np.ndarray

    def section(self, points: np.ndarray) -> np.ndarray:
        """``(r, z)`` of ``points`` in the solid's own frame."""
        local = (np.asarray(points, float) - self.pos) @ self.rot
        return np.stack([np.linalg.norm(local[..., :2], axis=-1), local[..., 2]], axis=-1)

    def bound(self, rz: np.ndarray) -> np.ndarray:
        """Distance to the enclosing cylinder: a lower bound of :meth:`distance`."""
        a = rz[..., 0] - self.profile[:, 0].max()
        b = np.abs(rz[..., 1]) - np.abs(self.profile[:, 1]).max()
        return (np.hypot(np.maximum(a, 0.0), np.maximum(b, 0.0))
                + np.minimum(np.maximum(a, b), 0.0))

    def distance(self, points: np.ndarray) -> np.ndarray:
        return _polygon_distance(self.section(points), self.profile)


def _polygon_distance(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance from 2-D ``points`` to a closed polygon (negative inside)."""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nsi,si->ns", ap, ab) / np.einsum("si,si->s", ab, ab), 0.0, 1.0)
    d = np.linalg.norm(ap - t[..., None] * ab[None, :, :], axis=-1).min(axis=1)
    # Crossing-number test, vectorised over the polygon edges.
    py, ay, by = points[:, None, 1], a[None, :, 1], b[None, :, 1]
    straddles = (ay > py) != (by > py)
    x_at = a[None, :, 0] + (py - ay) / np.where(ab[None, :, 1] == 0.0, 1.0, ab[None, :, 1]) \
        * ab[None, :, 0]
    inside = (straddles & (points[:, None, 0] < x_at)).sum(axis=1) % 2 == 1
    return np.where(inside, -d, d)


def _revolved_profile(local: np.ndarray, bins: int = PROFILE_BINS) -> np.ndarray:
    """``(r, z)`` section of a revolved mesh: the largest radius in each z bin, capped on axis."""
    r, z = np.linalg.norm(local[:, :2], axis=1), local[:, 2]
    edges = np.linspace(z.min(), z.max(), bins + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, bins - 1)
    rows = [(float(r[idx == i].max()), float(z[idx == i].max()))
            for i in range(bins) if np.any(idx == i)]
    rows.sort(key=lambda row: row[1])
    return np.array([(0.0, z.min()), *rows, (0.0, z.max())])


@dataclass(frozen=True)
class Board:
    """The board's collision geometry: the plate box and the two pulleys."""

    plate: Box
    pulleys: tuple[Revolved, ...]

    def distance(self, points: np.ndarray) -> np.ndarray:
        d = self.plate.distance(points)
        for pulley in self.pulleys:
            d = np.minimum(d, pulley.distance(points))
        return d

    def min_distance(self, points: np.ndarray) -> float:
        """The smallest :meth:`distance`, skipping the profile query where a bound rules it out."""
        best = float(self.plate.distance(points).min())
        for pulley in self.pulleys:
            rz = pulley.section(points)
            near = pulley.bound(rz) < best
            if near.any():
                best = min(best, float(_polygon_distance(rz[near], pulley.profile).min()))
        return best


@dataclass(frozen=True)
class Gripper:
    """The 2F-85 colliders: per-shape points in their body frame, plus a frozen tracking set."""

    bodies: tuple[int, ...]
    shape_points: tuple[np.ndarray, ...]
    shape_labels: tuple[str, ...]
    local: np.ndarray
    by_jaw: dict[int, np.ndarray] = field(default_factory=dict)

    def points(self, jaw: int | None = None) -> np.ndarray:
        """The frozen tracking-frame set for a Robotiq command byte (falls back to the base one)."""
        return self.local if jaw is None else self.by_jaw.get(int(jaw), self.local)

    def world_points(self, body_q: np.ndarray) -> np.ndarray:
        """Every kept collider point, each placed by its own measured body pose."""
        return np.vstack([
            points @ _quat_mat(body_q[body][3:7]).T + np.asarray(body_q[body][:3], float)
            for body, points in zip(self.bodies, self.shape_points)])

    def measure(self, board: Board, body_q: np.ndarray) -> float:
        """Smallest gripper -> board distance with every collider at its measured pose."""
        return board.min_distance(self.world_points(body_q))

    def predict(self, board: Board, X_W_track: np.ndarray, jaw: int | None = None) -> float:
        """Smallest distance with the gripper frozen in the UR tracking frame at ``X_W_track``."""
        X = np.asarray(X_W_track, float)
        return board.min_distance(self.points(jaw) @ X[:3, :3].T + X[:3, 3])


def _shape_world(model, shape: int, body_q: np.ndarray) -> np.ndarray:
    body = int(model.shape_body.numpy()[shape])
    local = wp_transform_to_mat4(model.shape_transform.numpy()[shape])
    return wp_transform_to_mat4(body_q[body]) @ local if body >= 0 else local


def _shape_vertices(model, shape: int) -> np.ndarray:
    scale = np.asarray(model.shape_scale.numpy()[shape], float)[:3]
    return np.asarray(model.shape_source[shape].vertices, float) * scale


def board_obstacles(model, body_q: np.ndarray) -> Board:
    """The board's plate box and pulley solids of revolution, in world frame."""
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    flags = model.shape_flags.numpy()
    shape_body = model.shape_body.numpy()
    labels = list(model.shape_label)
    plate_shape = labels.index(BOARD_PLATE_SHAPE)
    X = _shape_world(model, plate_shape, body_q)
    plate = Box(rot=X[:3, :3], pos=X[:3, 3],
                half=np.asarray(model.shape_scale.numpy()[plate_shape], float)[:3])
    pulleys = []
    for body in sorted({int(shape_body[s]) for s in range(model.shape_count)
                        if int(shape_body[s]) >= 0 and "round_pulley" in (labels[s] or "")}):
        shapes = [s for s in range(model.shape_count)
                  if int(shape_body[s]) == body and int(flags[s]) & collide]
        X_wb = wp_transform_to_mat4(body_q[body])
        local = np.vstack([
            _shape_vertices(model, s) @ wp_transform_to_mat4(
                model.shape_transform.numpy()[s])[:3, :3].T
            + wp_transform_to_mat4(model.shape_transform.numpy()[s])[:3, 3] for s in shapes])
        pulleys.append(Revolved(rot=X_wb[:3, :3], pos=X_wb[:3, 3],
                                profile=_revolved_profile(local)))
    if not pulleys:
        raise RuntimeError("no pulley colliders found on the board")
    return Board(plate=plate, pulleys=tuple(pulleys))


def gripper_geometry(model, body_qs, X_W_track: np.ndarray, margin: float = PRUNE_MARGIN,
                     jaw_keys: tuple[int, ...] = ()) -> Gripper:
    """The 2F-85 colliders, pruned to the points within ``margin`` of the lowest one.

    ``body_qs[0]`` is the base (as-held) configuration; any further entry is the gripper driven to
    the Robotiq byte at the same index of ``jaw_keys``. The jaws open at ``place_3``, which swings
    the fingers several mm lower over the pulley, so :meth:`Gripper.predict` needs the right set
    per commanded byte rather than one worst-case union.
    """
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    flags = model.shape_flags.numpy()
    shape_body = model.shape_body.numpy()
    types = model.shape_type.numpy()
    bodies = {b for b in range(model.body_count) if GRIPPER_MODEL in (model.body_label[b] or "")}
    poses = ([np.asarray(body_qs, dtype=np.float64)] if np.ndim(body_qs) == 2
             else [np.asarray(q, dtype=np.float64) for q in body_qs])
    rows = []
    for shape in range(model.shape_count):
        body = int(shape_body[shape])
        if body not in bodies or not int(flags[shape]) & collide:
            continue
        if int(types[shape]) != int(newton.GeoType.MESH):
            raise RuntimeError(f"{model.shape_label[shape]}: non-mesh 2F-85 collider")
        X_bs = wp_transform_to_mat4(model.shape_transform.numpy()[shape])
        in_body = _shape_vertices(model, shape) @ X_bs[:3, :3].T + X_bs[:3, 3]
        local = []
        for body_q in poses:
            X_wb = wp_transform_to_mat4(body_q[body])
            world = in_body @ X_wb[:3, :3].T + X_wb[:3, 3]
            local.append((world - X_W_track[:3, 3]) @ X_W_track[:3, :3])
        rows.append((body, model.shape_label[shape], in_body, local))
    if not rows:
        raise RuntimeError(f"no {GRIPPER_MODEL} colliders found")
    floor = min(float(one[:, 2].min()) for _, _, _, local in rows for one in local) + margin
    kept = []
    for body, label, pts, local in rows:
        mask = np.any([one[:, 2] <= floor for one in local], axis=0)
        if mask.any():
            kept.append((body, label, pts[mask], [one[mask] for one in local]))
    stacked = [np.vstack([local[i] for _, _, _, local in kept]) for i in range(len(poses))]
    return Gripper(bodies=tuple(int(b) for b, _, _, _ in kept),
                   shape_points=tuple(p for _, _, p, _ in kept),
                   shape_labels=tuple(str(label) for _, label, _, _ in kept),
                   local=stacked[0],
                   by_jaw={int(k): stacked[i + 1] for i, k in enumerate(jaw_keys)})


@dataclass
class ClampReport:
    """What :func:`clamp_waypoints` had to change to keep the gripper clear."""

    min_clearance_mm: float
    lift_mm: dict[str, float]
    tilt_scale: float
    before_mm: dict[str, float]
    after_mm: dict[str, float]

    path_lift_mm: float = 0.0
    path_before_mm: float | None = None
    path_after_mm: float | None = None

    @property
    def max_lift_mm(self) -> float:
        return max(self.lift_mm.values(), default=0.0) + self.path_lift_mm

    def to_dict(self) -> dict:
        return {"min_clearance_mm": self.min_clearance_mm, "lift_mm": self.lift_mm,
                "tilt_scale": self.tilt_scale, "before_mm": self.before_mm,
                "after_mm": self.after_mm, "path_lift_mm": self.path_lift_mm,
                "path_before_mm": self.path_before_mm, "path_after_mm": self.path_after_mm}


def required_lift(board: Board, gripper: Gripper, X_W_track: np.ndarray, min_clearance: float,
                  jaw: int | None = None) -> float:
    """World ``+z`` lift that brings :meth:`Gripper.predict` up to ``min_clearance`` (>= 0).

    Clearance is *not* monotone in z: between ~10 and ~25 mm above the nominal ``place_3`` the
    binding obstacle is the large pulley's flange, a few tenths of a mm radially from the finger,
    and lifting barely helps until the finger clears its top. Adding the deficit each pass walks
    out of that valley instead of stalling in it.
    """
    lift = 0.0
    X = np.array(X_W_track, dtype=np.float64, copy=True)
    for _ in range(CLAMP_ITERS):
        deficit = min_clearance - gripper.predict(board, X, jaw)
        if deficit <= CLAMP_TOL_M:
            break
        lift += deficit
        X[2, 3] = X_W_track[2, 3] + lift
    return lift


def jaw_bytes(waypoints) -> dict[str, int | None]:
    """The Robotiq byte in force at each waypoint (``None`` means "keep the previous one")."""
    out, current = {}, None
    for w in waypoints:
        current = w.ur_gripper_byte if w.ur_gripper_byte is not None else current
        out[w.label] = current
    return out


def clamp_waypoints(waypoints, perturbation, tangent, board: Board, gripper: Gripper, *,
                    min_clearance: float = DEFAULT_MIN_CLEARANCE_MM * 1e-3,
                    max_lift: float = MAX_CLAMP_LIFT_M):
    """``(waypoints, ClampReport)``: perturbed UR waypoints lifted until the gripper is clear.

    A world ``+z`` lift raises every gripper point by the same amount, so it converges in a few
    passes and leaves the tilt (which is what makes an ``under``/``slanted`` loop) untouched. The
    tilt is only shrunk if even ``max_lift`` cannot clear the board.
    """
    from round_belt_task import perturbation as pert  # circular at import time

    jaws = jaw_bytes(waypoints)
    scale = 1.0
    while True:
        moved = pert.apply(waypoints, pert.scale_tilt(perturbation, scale), tangent)
        lifts, before = {}, {}
        for w in moved:
            if w.label not in pert.PERTURBED_LABELS or w.ur_mat() is None:
                continue
            before[w.label] = gripper.predict(board, w.ur_mat(), jaws[w.label]) * 1e3
            lifts[w.label] = required_lift(board, gripper, w.ur_mat(), min_clearance,
                                           jaws[w.label])
        if max(lifts.values(), default=0.0) <= max_lift or scale <= 0.0:
            break
        scale = max(0.0, round(scale - 0.25, 6))
    out, after = [], dict(before)
    for w in moved:
        lift = lifts.get(w.label, 0.0)
        if lift <= 0.0:
            out.append(w)
            continue
        lifted = replace(w, ur_pos=np.asarray(w.ur_pos, float) + np.array([0.0, 0.0, lift]))
        after[w.label] = gripper.predict(board, lifted.ur_mat(), jaws[w.label]) * 1e3
        out.append(lifted)
    report = ClampReport(min_clearance_mm=min_clearance * 1e3,
                         lift_mm={k: v * 1e3 for k, v in lifts.items()}, tilt_scale=scale,
                         before_mm=before, after_mm=after)
    return out, report


def path_clearance(board: Board, gripper: Gripper, poses: np.ndarray, jaws=None,
                   stride: int = PATH_STRIDE) -> float:
    """Smallest :meth:`Gripper.predict` over a sequence of commanded UR tracking poses (4x4).

    The waypoint clamp is blind to what happens *between* waypoints, and an ``over`` episode's
    closest approach is the finger swinging past the large pulley's rim mid-move. ``jaws`` is the
    commanded Robotiq byte per pose, so the release poses use the open-jaw geometry.
    """
    step = max(1, int(stride))
    poses = np.asarray(poses, dtype=np.float64)[::step]
    jaws = (np.full(len(poses), None) if jaws is None
            else np.asarray(jaws)[::step][:len(poses)])
    best = math.inf
    for jaw in dict.fromkeys(jaws.tolist()):
        points = gripper.points(jaw)
        block_all = poses[jaws == jaw] if jaw is not None else poses
        for start in range(0, len(block_all), PATH_CHUNK):
            block = block_all[start:start + PATH_CHUNK]
            world = np.einsum("kij,pj->kpi", block[:, :3, :3], points) + block[:, None, :3, 3]
            best = min(best, board.min_distance(world.reshape(-1, 3)))
    return best


def lift_waypoints(waypoints, labels, lift: float):
    """Copies of ``waypoints`` with the UR pose of every label in ``labels`` raised by ``lift``."""
    out = []
    for w in waypoints:
        if w.label in labels and w.ur_pos is not None:
            w = replace(w, ur_pos=np.asarray(w.ur_pos, float) + np.array([0.0, 0.0, lift]))
        out.append(w)
    return out
