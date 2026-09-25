"""Replay panel (:class:`~task_common.replay_app.ReplayHook`) for the learned MPC's solves.

Draws the ``LEARNED_MPC_DEBUG`` message answering the current frame (the latest with
``msg.step <= step``): decoded planned belts, planned EE knots, planned actions and the demo's
fixed target belts (stage goals, or the final frame of a demo-traj run). Files come from ``meta["learned_mpc"]`` unless overridden; a layer
whose files are missing is disabled with a reason. Every layer defaults off.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from task_common.latent_encoder import LatentDecoder
from task_common.recording import Recording
from task_common.replay_metrics import FRANKA_BASE_BODY, quat_rotate

if TYPE_CHECKING:
    from task_common.replay_app import ReplayApp

LAYERS = ("planned_belt", "planned_ee", "actions", "target_belt")
REMOVED_LAYERS = ("action_rotation", "readout")  # old --learned-layers names, ignored
LAYER_LABELS = {"planned_belt": "Planned belt", "planned_ee": "Planned EE",
                "actions": "Planned actions", "target_belt": "Target belt"}
DEFAULT_DEBUG_CHANNEL = "LEARNED_MPC_DEBUG"
FRANKA_PLAN_CHANNEL = "TARGET_CARTESIAN_POSE_TRAJECTORY"
N_LATENT = 16
ROOT = "/replay/learned_mpc"
ACTION_SCALE_MAX = 100.0
ACTION_SCALE_DEFAULT = 9.0
POINT_M = 0.003
MARKER_M = 0.0015
# viser 1.1 line segments have no opacity, so faded loops are tube meshes (per-mesh opacity).
PLAN_TUBE_M = 0.003  # radius; the real belt is 3.3 mm
TARGET_TUBE_M = 0.003
TUBE_SIDES = 8
PLAN_ALPHA = (0.9, 0.2)  # step 1 -> step N
TARGET_ALPHA = (0.9, 0.25)  # current stage, other stages
# Arrows are meshes: viser 1.1 add_arrows has a fixed absolute head size.
ARROW_SIDES = 12
SHAFT_R_M, HEAD_LEN_M, HEAD_R_M = 0.0008, 0.004, 0.002  # constant; only the length scales
U0_THICK = 1.4  # radii factor for u0
ACT_ALPHA = (0.75, 0.2)  # step 1 -> step N-1; u0 is opaque

# One hue per category; the recorded belt is gold, so no yellow/orange here.
PLAN_BLUE = (37, 99, 235)
TARGET_GREEN = (22, 163, 74)
FRANKA_PLAN = (6, 182, 212)
AUG_MARKER = (31, 41, 55)
ACT_FRANKA = (8, 145, 178)
ACT_UR = (220, 38, 38)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _goal_fields(path: Path) -> dict[str, Any]:
    """``goal_frames``/``stage_frames`` (else the ``#a,b`` of ``goal_source``) and the demo the
    goals were taken from, from a ``deploy.npz`` or ``demo_goals.npz``."""
    with np.load(path, allow_pickle=False) as d:
        out: dict[str, Any] = {"frames": None, "demo_path": None, "demo_sha256": None}
        for key in ("goal_frames", "stage_frames"):
            if key in d.files:
                out["frames"] = np.asarray(d[key], dtype=np.int64).reshape(-1)
                break
        if "goal_source" in d.files:
            src = str(d["goal_source"])
            if src.startswith("demo:") and "#" in src:
                demo, _, idx = src[len("demo:"):].rpartition("#")
                out["demo_path"] = demo
                if out["frames"] is None:
                    out["frames"] = np.array([int(v) for v in idx.split(",")], dtype=np.int64)
        if "demo_sha256" in d.files:
            out["demo_sha256"] = str(d["demo_sha256"])
    return out


def learned_meta(meta: dict) -> dict:
    block = meta.get("learned_mpc")
    return block if isinstance(block, dict) else {}


def has_learned_debug(meta: dict) -> bool:
    """True when the run's controller published the debug channel (``meta.learned_mpc``)."""
    return bool(learned_meta(meta).get("debug_channel"))


def any_learned_runs(root: Path) -> bool:
    """Any run under ``root`` recorded with the debug channel (reads ``meta.json`` only)."""
    for path in Recording.list_runs(root):
        try:
            if has_learned_debug(json.loads((path / "meta.json").read_text())):
                return True
        except (OSError, ValueError):
            continue
    return False


def parse_layers(text: str | None) -> tuple[str, ...]:
    """``"a,b"`` -> validated layer names (``LAYERS``); ``REMOVED_LAYERS`` are dropped with a
    warning."""
    names = tuple(s.strip() for s in (text or "").split(",") if s.strip())
    gone = [n for n in names if n in REMOVED_LAYERS]
    if gone:
        logger.warning(f"[REPLAY] learned layer(s) {gone} were removed; ignoring")
        names = tuple(n for n in names if n not in REMOVED_LAYERS)
    bad = [n for n in names if n not in LAYERS]
    if bad:
        raise ValueError(f"unknown learned layer(s) {bad}; choose from {','.join(LAYERS)}")
    return names


@dataclass
class Solve:
    """One debug message, geometry precomputed in world (float64 unless noted)."""

    step: int
    utime: int
    frame: int
    t: np.ndarray
    x_sol: np.ndarray
    u_sol: np.ndarray
    z_ref: np.ndarray
    p_ref: np.ndarray
    scalars: dict[str, float]
    ee_franka: np.ndarray  # (N+1, 3) x_sol Franka rows
    ee_ur: np.ndarray  # (N+1, 3) x_sol UR rows
    franka_knots: np.ndarray | None = None  # (K, 3) published plan knots
    belts: np.ndarray | None = None  # (N+1, 150, 3) float32 decoded x_sol[:16, i]

    @property
    def n(self) -> int:
        return self.u_sol.shape[1]


def _blocks(payload: dict) -> dict[str, np.ndarray]:
    return {k: np.asarray(v["data"], dtype=np.float64) for k, v in payload["blocks"].items()}


def _t0_key(t: float) -> int:
    return round(float(t) * 1e6)


def _alphas(n: int, ramp: tuple[float, float]) -> np.ndarray:
    return np.linspace(ramp[0], ramp[1], n) if n > 1 else np.array([ramp[0]])


def _tube_faces(n_pts: int, sides: int = TUBE_SIDES) -> np.ndarray:
    j, k = np.meshgrid(np.arange(n_pts), np.arange(sides), indexing="ij")
    a = j * sides + k
    b = j * sides + (k + 1) % sides
    c = ((j + 1) % n_pts) * sides + k
    d = ((j + 1) % n_pts) * sides + (k + 1) % sides
    faces = np.concatenate([np.stack([a, b, c], -1), np.stack([b, d, c], -1)], axis=0)
    return faces.reshape(-1, 3).astype(np.uint32)


def tube_vertices(pts: np.ndarray, radius: float, sides: int = TUBE_SIDES) -> np.ndarray:
    """Closed-loop tube around ``pts`` (P, 3) -> (P * sides, 3); ring ``j`` is centred on
    ``pts[j]``."""
    pts = np.asarray(pts, dtype=np.float64)
    tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    # The loop's plane normal is ~perpendicular to every tangent: a twist-free frame.
    m = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][2]
    nrm = m - (tan @ m)[:, None] * tan
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    bin_ = np.cross(tan, nrm)
    th = 2.0 * np.pi * np.arange(sides) / sides
    ring = np.cos(th)[None, :, None] * nrm[:, None] + np.sin(th)[None, :, None] * bin_[:, None]
    return (pts[:, None] + radius * ring).reshape(-1, 3)


def _perp(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def _ring(center: np.ndarray, e1: np.ndarray, e2: np.ndarray, r: float,
          sides: int = ARROW_SIDES) -> np.ndarray:
    th = 2.0 * np.pi * np.arange(sides) / sides
    return center + r * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)


def _fan(center: int, ring: int, sides: int = ARROW_SIDES, flip: bool = False) -> np.ndarray:
    k = np.arange(sides)
    a, b = ring + k, ring + (k + 1) % sides
    return np.stack([np.full(sides, center), b, a] if flip else [np.full(sides, center), a, b], -1)


def _band(r0: int, r1: int, sides: int = ARROW_SIDES) -> np.ndarray:
    k = np.arange(sides)
    a, b = r0 + k, r0 + (k + 1) % sides
    c, d = r1 + k, r1 + (k + 1) % sides
    return np.concatenate([np.stack([a, b, c], -1), np.stack([b, d, c], -1)])


def arrow_dims(length: float, thick: float = 1.0) -> tuple[float, float, float]:
    """``(shaft radius, head length, head radius)``: constant, except an arrow shorter than
    the head is all head, shrunk uniformly to ``length`` (the tip stays at base + vec)."""
    f = min(1.0, length / HEAD_LEN_M)
    return SHAFT_R_M * thick * f, HEAD_LEN_M * f, HEAD_R_M * thick * f


def arrow_faces(sides: int = ARROW_SIDES) -> np.ndarray:
    # vertices: base centre, shaft ring at base, shaft ring at head, head ring, tip
    r0, r1, r2, tip = 1, 1 + sides, 1 + 2 * sides, 1 + 3 * sides
    return np.concatenate([_fan(0, r0, sides, flip=True), _band(r0, r1, sides),
                           _band(r1, r2, sides), _fan(tip, r2, sides)]
                          ).astype(np.uint32)


def arrow_vertices(base: np.ndarray, vec: np.ndarray, thick: float = 1.0,
                   sides: int = ARROW_SIDES) -> np.ndarray:
    """Cylinder + cone from ``base`` to ``base + vec`` -> (3 * sides + 2, 3); vertex 0 is the
    base, the last is the tip. A zero vector collapses to the base."""
    base = np.asarray(base, dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length < 1e-12:
        return np.broadcast_to(base, (3 * sides + 2, 3)).copy()
    d = np.asarray(vec, dtype=np.float64) / length
    e1, e2 = _perp(d)
    shaft, head, head_r = arrow_dims(length, thick)
    neck = base + (length - head) * d
    return np.concatenate([base[None], _ring(base, e1, e2, shaft, sides),
                           _ring(neck, e1, e2, shaft, sides), _ring(neck, e1, e2, head_r, sides),
                           (base + vec)[None]])


def _base_pose(rec: Recording, frame: int, label: str) -> tuple[np.ndarray, np.ndarray] | None:
    labels = rec.meta["body_labels"]
    if label not in labels:
        return None
    row = rec.body_q[frame, labels.index(label)].astype(np.float64)
    return row[:3], row[3:7]


class LearnedMpcPanel:
    """``Learned MPC`` folder: four toggleable layers over the recorded solves."""

    def __init__(self, *, deploy: Path | None = None, decoder: Path | None = None,
                 demo_goals: Path | None = None, demo_episode: Path | None = None,
                 layers: tuple[str, ...] = (), root: str = ROOT) -> None:
        self.root = root
        self.overrides = {"deploy": deploy, "decoder": decoder, "demo_goals": demo_goals,
                          "demo_episode": demo_episode}
        self.requested = set(parse_layers(",".join(layers)))
        self.action_scale = ACTION_SCALE_DEFAULT
        self.paths: dict[str, Path | None] = {}
        self.available: dict[str, str | None] = dict.fromkeys(LAYERS, "no run loaded")
        self.notes: list[str] = []
        self.solves: list[Solve] = []
        self.solve_steps = np.zeros(0, dtype=np.int64)
        self.decoder: LatentDecoder | None = None
        self.demo_belts: np.ndarray | None = None
        self.goal_frames: np.ndarray | None = None  # fixed target frames into demo_belts
        self.staged = False
        self.current: int = -1
        self.handles: dict[str, Any] = {}
        self._drawn: dict[str, tuple] = {}
        self._cache: dict[tuple[str, str], Any] = {}
        self.last_frame_ms = 0.0

    # ---- GUI --------------------------------------------------------------------------

    def build_gui(self, app: ReplayApp) -> None:
        gui = app.gui
        with gui.add_folder("Learned MPC", expand_by_default=bool(self.requested)) as folder:
            self.folder = folder
            self.status = gui.add_markdown("")
            self.boxes = {name: gui.add_checkbox(LAYER_LABELS[name], name in self.requested)
                          for name in LAYERS}
            self.scale_slider = gui.add_slider("Action scale", min=1.0, max=ACTION_SCALE_MAX,
                                               step=1.0, initial_value=self.action_scale)
        for name, box in self.boxes.items():
            box.on_update(lambda e, n=name: app.call_soon(
                lambda v=bool(e.target.value): self.set_layer(app, n, v)))
        self.scale_slider.on_update(lambda e: app.call_soon(
            lambda v=float(e.target.value): self.set_action_scale(app, v)))

    def set_layer(self, app: ReplayApp, name: str, enabled: bool) -> None:
        if name not in LAYERS:
            raise ValueError(f"unknown layer {name!r}")
        (self.requested.add if enabled else self.requested.discard)(name)
        app._set_gui(self.boxes[name], bool(enabled))
        self._update(app, app.current_frame, app.current_step)

    def set_action_scale(self, app: ReplayApp, value: float) -> None:
        self.action_scale = float(np.clip(value, 1.0, ACTION_SCALE_MAX))
        app._set_gui(self.scale_slider, self.action_scale)
        self._update(app, app.current_frame, app.current_step)

    def shown(self, name: str) -> bool:
        return name in self.requested and self.available.get(name) is None

    # ---- run loading ------------------------------------------------------------------

    def _resolve(self, block: dict) -> None:
        def pick(key: str) -> Path | None:
            value = self.overrides.get(key) or block.get(key)
            return Path(value) if value else None

        self.paths = {k: pick(k) for k in ("deploy", "demo_goals", "demo_episode")}
        decoder = self.overrides.get("decoder")
        if decoder is None and self.paths["deploy"] is not None:
            decoder = self.paths["deploy"].parent / "decoder.npz"
        self.paths["decoder"] = Path(decoder) if decoder else None

    def _missing(self, key: str) -> str | None:
        path = self.paths.get(key)
        if path is None:
            return f"no {key} path (meta or --{key.replace('_', '-')})"
        if not path.is_file():
            return f"{key} missing: {path}"
        return None

    def _load(self, key: str, loader, path: Path | None = None) -> Any:
        path = self.paths[key] if path is None else path
        cache_key = (key, str(path.resolve()))
        if cache_key not in self._cache:
            self._cache[cache_key] = loader(path)
        return self._cache[cache_key]

    def on_run_loaded(self, app: ReplayApp, rec: Recording) -> None:
        start = time.perf_counter()
        self.solves, self.current, self._drawn = [], -1, {}
        self.decoder = self.demo_belts = self.goal_frames = None
        self.staged = False
        self.notes = []
        block = learned_meta(rec.meta)
        self._resolve(block)
        channel = block.get("debug_channel") or rec.meta.get("channels", {}).get(
            "learned_mpc_debug_channel", DEFAULT_DEBUG_CHANNEL)
        msgs = [t for t in rec.targets if t.channel == channel]
        reason = None if msgs else f"no {channel} messages in this run"
        if msgs:
            try:
                self._index(rec, msgs)
            except (KeyError, ValueError, IndexError) as exc:
                reason = f"cannot parse {channel}: {exc!r}"
                self.solves = []
        self.solve_steps = np.array([s.step for s in self.solves], dtype=np.int64)
        self.available = dict.fromkeys(LAYERS, reason)
        if reason is None:
            self._load_files(block)
        for name, box in self.boxes.items():
            box.disabled = self.available[name] is not None
        self.scale_slider.disabled = self.available["actions"] is not None
        self._show_status(start)

    def _load_files(self, block: dict) -> None:
        for key in ("deploy", "demo_goals", "demo_episode"):
            path, want = self.paths.get(key), block.get(f"{key}_sha256")
            checkable = path is not None and path.is_file() and want
            if (checkable and self.overrides.get(key) is None
                    and self._load(f"sha:{key}", _sha256, path) != want):
                self.notes.append(f"{key} sha256 differs from the recording's")
        why = self._missing("decoder")
        if why is None:
            try:
                self.decoder = self._load("decoder", LatentDecoder.load)
                self._decode_all()
            except (OSError, KeyError, ValueError) as exc:
                why = f"decoder unusable: {exc!r}"
        self.available["planned_belt"] = why
        why = self._missing("demo_episode")
        if why is None:
            try:
                belts = self._load("demo_episode", lambda p: np.asarray(
                    np.load(p)["pcd_belt"], dtype=np.float32))
                if belts.ndim != 3 or belts.shape[2] != 3 or not len(belts):
                    raise ValueError(f"pcd_belt shape {belts.shape}")
                self.demo_belts = belts
            except (OSError, KeyError, ValueError) as exc:
                why = f"demo_episode unusable: {exc!r}"
        if why is None:
            why = self._resolve_goals(block)
        self.available["target_belt"] = why

    def _resolve_goals(self, block: dict) -> str | None:
        """Sets ``goal_frames``/``staged`` from deploy + demo_goals; returns a reason if the
        frames cannot be pinned down exactly."""
        if "demo_traj_yaml" not in block:
            return "meta.learned_mpc has no demo_traj_yaml: staged vs demo-traj unknown"
        staged = not block["demo_traj_yaml"]
        found: dict[str, dict] = {}
        for key in ("deploy", "demo_goals"):
            path = self.paths.get(key)
            if path is not None and path.is_file():
                try:
                    found[key] = self._load(f"goals:{key}", _goal_fields, path)
                except (OSError, ValueError) as exc:
                    return f"{key} goal frames unreadable: {exc!r}"
        frames = {k: tuple(v["frames"].tolist()) for k, v in found.items()
                  if v["frames"] is not None}
        if not frames:
            return "no goal_frames/stage_frames/goal_source in deploy or demo_goals"
        if len(set(frames.values())) > 1:
            return f"goal frames disagree: {frames}"
        goal = np.array(next(iter(frames.values())), dtype=np.int64)
        last = len(self.demo_belts) - 1
        if goal.min() < 0 or goal.max() > last:
            return f"goal frames {goal.tolist()} outside demo frames 0..{last}"
        demo_sha = self._load("sha:demo_episode", _sha256, self.paths["demo_episode"])
        for key, fields in found.items():
            want = fields["demo_sha256"]
            if want is None and fields["demo_path"] and Path(fields["demo_path"]).is_file():
                want = self._load("sha:demo_episode", _sha256, Path(fields["demo_path"]))
            if want is not None and want != demo_sha:
                return f"{key} goals were taken from another demo episode"
        if staged and any("stage" not in s.scalars for s in self.solves):
            return "staged run without a stage scalar"
        if not staged and goal[-1] != last:
            self.notes.append(f"final goal frame {goal[-1]} is not the demo's last ({last})")
        self.staged = staged
        self.goal_frames = goal if staged else goal[-1:]
        return None

    def _index(self, rec: Recording, msgs: list) -> None:
        keys = set()
        solves = []
        for msg in msgs:
            b = _blocks(msg.payload)
            names = msg.payload["blocks"]["scalars"]["datatypes"]
            s = {str(k): float(v) for k, v in zip(names, b["scalars"][:, 0], strict=True)}
            x = b["x_sol"]
            solve = Solve(step=msg.step, utime=int(msg.payload["utime"]),
                          frame=rec.frame_at_step(msg.step),
                          t=np.asarray(msg.payload["blocks"]["x_sol"]["t"], dtype=np.float64),
                          x_sol=x, u_sol=b["u_sol"], z_ref=b["z_ref"], p_ref=b["p_ref"],
                          scalars=s, ee_franka=x[N_LATENT:N_LATENT + 3].T.copy(),
                          ee_ur=x[N_LATENT + 3:N_LATENT + 6].T.copy())
            solves.append(solve)
            keys.add(_t0_key(solve.t[0]))
        plans: dict[tuple[str, int], Any] = {}
        for t in rec.targets:
            if t.channel != FRANKA_PLAN_CHANNEL:
                continue
            pos = t.payload.get("blocks", {}).get("end_effector_position_target")
            if pos and pos["t"] and _t0_key(pos["t"][0]) in keys:
                plans.setdefault((t.channel, _t0_key(pos["t"][0])), t.payload)
        for solve in solves:
            key = _t0_key(solve.t[0])
            franka = plans.get((FRANKA_PLAN_CHANNEL, key))
            if franka is not None:
                solve.franka_knots = self._franka_world(rec, solve.frame, franka)
        self.solves = solves

    @staticmethod
    def _franka_world(rec: Recording, frame: int, payload: dict) -> np.ndarray | None:
        pos = np.asarray(payload["blocks"]["end_effector_position_target"]["data"],
                         dtype=np.float64).T
        base = _base_pose(rec, frame, FRANKA_BASE_BODY)
        if base is None:
            return pos
        return base[0] + quat_rotate(base[1], pos)

    def _decode_all(self) -> None:
        if not self.solves:
            return
        # One latent per call: decode_batch may round differently from decode (BLAS blocking).
        for s in self.solves:
            s.belts = np.stack([self.decoder.decode(z) for z in s.x_sol[:N_LATENT].T])

    def _show_status(self, start: float) -> None:
        lines = []
        if self.solves:
            lines.append(f"{len(self.solves)} solves, N {self.solves[0].n}, loaded in "
                         f"{1e3 * (time.perf_counter() - start):.0f} ms")
        reasons = {why for why in self.available.values() if why}
        for why in sorted(reasons):
            names = [LAYER_LABELS[n] for n in LAYERS if self.available[n] == why]
            lines.append(f"- off ({', '.join(names)}): {why}")
        lines += [f"- {note}" for note in self.notes]
        self.status.content = "\n".join(lines)
        logger.info(f"[REPLAY] learned MPC: {'; '.join(lines) or 'nothing to show'}")

    # ---- per frame --------------------------------------------------------------------

    def solve_at(self, step: int) -> int:
        """Index of the latest solve with ``step <= step`` (-1 if none)."""
        return int(np.searchsorted(self.solve_steps, step, side="right")) - 1

    def on_frame(self, app: ReplayApp, frame_index: int, step: int, sim_time: float) -> None:
        self._update(app, frame_index, step)

    def _update(self, app: ReplayApp, frame: int, step: int) -> None:
        start = time.perf_counter()
        self.current = self.solve_at(step) if self.solves else -1
        solve = self.solves[self.current] if self.current >= 0 else None
        for name, draw in (("planned_belt", self._draw_planned_belt),
                           ("planned_ee", self._draw_planned_ee),
                           ("actions", self._draw_actions),
                           ("target_belt", self._draw_target_belt)):
            if not self.shown(name) or solve is None:
                self._hide(name)
                continue
            key = (self.current, self.action_scale if name == "actions" else None)
            if self._drawn.get(name) != key:
                draw(app, solve)
                self._drawn[name] = key
        self.last_frame_ms = 1e3 * (time.perf_counter() - start)

    def _hide(self, layer: str) -> None:
        self._drawn.pop(layer, None)
        for name, handle in self.handles.items():
            if name.startswith(f"{layer}/"):
                handle.visible = False

    def _points(self, app: ReplayApp, name: str, points: np.ndarray, color,
                size: float = POINT_M) -> Any:
        points = np.ascontiguousarray(points, dtype=np.float32)
        handle = self.handles.get(name)
        if handle is not None and handle.points.shape == points.shape:
            handle.points = points
            handle.visible = True
            return handle
        if handle is not None:
            handle.remove()
        self.handles[name] = app.server.scene.add_point_cloud(
            f"{self.root}/{name}", points, color, point_size=size, point_shape="circle",
            precision="float32")
        return self.handles[name]

    def _tube(self, app: ReplayApp, name: str, pts: np.ndarray, radius: float, color,
              opacity: float) -> Any:
        return self._mesh(app, name, tube_vertices(pts, radius), _tube_faces(len(pts)), color,
                          opacity)

    def _tubes(self, app: ReplayApp, prefix: str, loops, radius: float, color,
               ramp: tuple[float, float]) -> None:
        alphas = _alphas(len(loops), ramp)
        for i, (pts, a) in enumerate(zip(loops, alphas, strict=True), start=1):
            self._tube(app, f"{prefix}{i}", pts, radius, color, float(a))
        i = len(loops) + 1
        while f"{prefix}{i}" in self.handles:  # a shorter horizon than a previous run
            self._set_hidden(f"{prefix}{i}")
            i += 1

    def _set_hidden(self, name: str) -> None:
        if name in self.handles:
            self.handles[name].visible = False

    def _draw_planned_belt(self, app: ReplayApp, s: Solve) -> None:
        self._tubes(app, "planned_belt/step_", s.belts[1:], PLAN_TUBE_M, PLAN_BLUE, PLAN_ALPHA)

    def _draw_planned_ee(self, app: ReplayApp, s: Solve) -> None:
        if s.franka_knots is not None and len(s.franka_knots):
            self._points(app, "planned_ee/franka_knots", s.franka_knots, FRANKA_PLAN)
        else:
            self._set_hidden("planned_ee/franka_knots")
        self._points(app, "planned_ee/augmented", np.concatenate([s.ee_franka, s.ee_ur]),
                     AUG_MARKER, size=MARKER_M)

    def action_bases(self, s: Solve) -> tuple[np.ndarray, np.ndarray]:
        """``(franka, ur)`` (N, 3) arrow bases: the Franka plan knots (``x_sol`` rows if the
        plan is missing) and the UR ``x_sol`` rows."""
        n = s.n
        fk = s.franka_knots
        franka = fk[:n] if fk is not None and len(fk) >= n else s.ee_franka[:n]
        return franka, s.ee_ur[:n]

    def action_arrows(self, s: Solve) -> list[tuple[str, np.ndarray, np.ndarray, Any, float,
                                                    float]]:
        """``(name, base, vec, colour, opacity, thickness)`` per arm and step, float64."""
        n, k = s.n, self.action_scale
        alphas = np.concatenate([[1.0], _alphas(n - 1, ACT_ALPHA)]) if n > 1 else [1.0]
        out = []
        for arm, bases, rows, color in (("franka", self.action_bases(s)[0], slice(0, 3),
                                         ACT_FRANKA),
                                        ("ur", self.action_bases(s)[1], slice(3, 6), ACT_UR)):
            for i in range(n):
                out.append((f"actions/{arm}_{i}", bases[i], k * s.u_sol[rows, i], color,
                            float(alphas[i]), U0_THICK if i == 0 else 1.0))
        return out

    def _mesh(self, app: ReplayApp, name: str, verts: np.ndarray, faces: np.ndarray, color,
              opacity: float) -> Any:
        verts = np.ascontiguousarray(verts, dtype=np.float32)
        handle = self.handles.get(name)
        if handle is not None and handle.vertices.shape == verts.shape:
            handle.vertices = verts
            if handle.opacity != opacity:
                handle.opacity = opacity
            handle.visible = True
            return handle
        if handle is not None:
            handle.remove()
        self.handles[name] = app.server.scene.add_mesh_simple(
            f"{self.root}/{name}", verts, faces, color=color, opacity=opacity,
            cast_shadow=False, receive_shadow=False)
        return self.handles[name]

    def _draw_actions(self, app: ReplayApp, s: Solve) -> None:
        faces = arrow_faces()
        names = set()
        for name, base, vec, color, alpha, thick in self.action_arrows(s):
            self._mesh(app, name, arrow_vertices(base, vec, thick), faces, color, alpha)
            names.add(name)
        for name, handle in self.handles.items():  # a shorter horizon than a previous run
            if name.startswith("actions/") and name not in names:
                handle.visible = False

    def target_opacities(self, s: Solve) -> np.ndarray:
        """Per ``goal_frames`` entry: opaque for the solve's stage (always, demo-traj)."""
        if not self.staged:
            return np.array([TARGET_ALPHA[0]])
        cur = int(s.scalars["stage"])
        return np.where(np.arange(len(self.goal_frames)) == cur, *TARGET_ALPHA)

    def _draw_target_belt(self, app: ReplayApp, s: Solve) -> None:
        loops = self.demo_belts[self.goal_frames]
        for i, (pts, a) in enumerate(zip(loops, self.target_opacities(s), strict=True)):
            self._tube(app, f"target_belt/goal_{i}", pts, TARGET_TUBE_M, TARGET_GREEN, float(a))
        i = len(loops)
        while f"target_belt/goal_{i}" in self.handles:  # fewer goals than a previous run
            self._set_hidden(f"target_belt/goal_{i}")
            i += 1
