"""Directive execution: parsed directives -> a populated ``newton.ModelBuilder``.

The per-kind importers, the weld look-ahead, frame resolution, the custom-directive
registry and the index bookkeeping (:class:`LoadedScene`).  What a directive looks like:
``utils.directives.schema``; the schema reference: ``docs/scene-directives.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from loguru import logger

import newton

from utils.directives.schema import (
    CORE_DIRECTIVES,
    ComponentColorRule,
    DirectiveFile,
    FrameDirective,
    ModelDirective,
    Vec3,
    WeldDirective,
    _fail,
    parse_directives,
)
from utils.labels import label_shapes_by_body
from utils.meshes import fix_inverted_mesh_winding, neutralize_textured_shape_colors
from utils.urdf import add_urdf_as_static_shapes

_IDENTITY = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())


@dataclass
class ModelRecord:
    """What one ``add_model`` produced in the builder."""

    directive: ModelDirective
    file: Path
    body_start: int
    body_end: int
    joint_start: int
    joint_end: int
    shape_start: int
    shape_end: int
    parent_body: int
    xform: wp.transform
    shape_labels: dict[str, int] = field(default_factory=dict)

    @property
    def bodies(self) -> list[int]:
        return list(range(self.body_start, self.body_end))

    @property
    def joints(self) -> list[int]:
        return list(range(self.joint_start, self.joint_end))

    @property
    def shapes(self) -> list[int]:
        return list(range(self.shape_start, self.shape_end))


@dataclass
class FrameRecord:
    name: str
    body: int
    X_BF: wp.transform


@dataclass
class LoadedScene:
    """Everything :func:`load_directives` learned while executing a directives file."""

    builder: newton.ModelBuilder
    directives: DirectiveFile
    models: dict[str, ModelRecord] = field(default_factory=dict)
    frames: dict[str, FrameRecord] = field(default_factory=dict)
    aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def _lookup(self, labels: Sequence[str], start: int, end: int, leaf: str, what: str) -> int:
        matches = [i for i in range(start, end) if str(labels[i]).rsplit("/", 1)[-1] == leaf]
        if len(matches) != 1:
            raise KeyError(
                f"expected exactly one {what} whose label ends in {leaf!r}, got "
                f"{[(i, labels[i]) for i in matches]}; candidates: "
                f"{[str(labels[i]) for i in range(start, end)]}"
            )
        return matches[0]

    def body(self, ref: str) -> int:
        """``world`` -> -1, ``<model>::<link>`` -> its body index, or an ``add_frame`` name."""
        if ref == "world":
            return -1
        if "::" in ref:
            name, link = ref.split("::", 1)
            record = self.models.get(name)
            if record is None:
                raise KeyError(f"{ref!r}: no loaded model {name!r} (have {list(self.models)})")
            return self._lookup(
                self.builder.body_label, record.body_start, record.body_end, link,
                f"body in {name}",
            )
        if ref in self.frames:
            return self.frames[ref].body
        raise KeyError(
            f"{ref!r} is not 'world', '<model>::<link>' or one of the frames {list(self.frames)}"
        )

    def frame_transform(self, ref: str) -> tuple[int, wp.transform]:
        """``(body, X_body_ref)``; identity for ``world`` and ``<model>::<link>``."""
        if ref in self.frames and "::" not in ref and ref != "world":
            frame = self.frames[ref]
            return frame.body, frame.X_BF
        return self.body(ref), wp.transform(_IDENTITY)

    def joint(self, model: str, joint_name: str) -> int:
        record = self.models.get(model)
        if record is None:
            raise KeyError(f"no loaded model {model!r} (have {list(self.models)})")
        return self._lookup(
            self.builder.joint_label, record.joint_start, record.joint_end, joint_name,
            f"joint in {model}",
        )

    def default_joint_positions(self) -> list[tuple[int, list[float]]]:
        """``(joint index, values)`` per ``default_joint_positions`` entry, in YAML order."""
        return [
            (self.joint(name, joint), list(values))
            for name, record in self.models.items()
            for joint, values in record.directive.default_joint_positions.items()
        ]


@dataclass
class DirectiveContext:
    """What a custom directive function is handed."""

    builder: newton.ModelBuilder
    scene: LoadedScene
    source: Path
    visual_cfg: newton.ModelBuilder.ShapeConfig
    collision_cfg: newton.ModelBuilder.ShapeConfig

    def resolve_path(self, file: str) -> Path:
        return self.scene.directives.resolve_path(file, self.source)


DirectiveFn = Callable[[DirectiveContext, Mapping[str, Any]], None]


def _enable_gravity_compensation(builder: newton.ModelBuilder, bodies: Sequence[int]) -> None:
    """MuJoCo gravity compensation for both arms + gripper (as round_belt.py); done on
    the builder because ``mujoco:gravcomp`` is a builder custom attribute, so it cannot
    be set later from ``apply_default_joint_state(model, ...)``."""
    try:
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in bodies:
            gravcomp.values[body] = 1.0
    except (KeyError, AttributeError):
        logger.warning("mujoco:gravcomp attribute not available; skipping gravity compensation.")


def _component_color_fn(
    rules: Sequence[ComponentColorRule],
) -> Callable[[newton.Mesh], Vec3 | None]:
    """Per split component: skip anything spanning >= ``max_span`` in XY, then take the first
    rule whose ``near_local_xy`` is within ``radius`` of the component's bbox centre."""

    def color_fn(mesh: newton.Mesh) -> Vec3 | None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.size == 0:
            return None
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        span = max(upper[0] - lower[0], upper[1] - lower[1])
        centre = 0.5 * (lower + upper)
        for rule in rules:
            if span >= rule.max_span:
                continue
            offset = math.hypot(
                centre[0] - rule.near_local_xy[0], centre[1] - rule.near_local_xy[1]
            )
            if offset <= rule.radius:
                return rule.color
        return None

    return color_fn


def _is_identity(pose: Sequence[float]) -> bool:
    return bool(
        np.allclose(
            [float(v) for v in pose], [0.0] * 6 + [1.0], atol=1.0e-9, rtol=0.0
        )
    )


def _pose_close(builder: newton.ModelBuilder, body: int, expected: wp.transform) -> bool:
    got = np.array([float(v) for v in builder.body_q[body]], dtype=np.float64)
    want = np.array([float(v) for v in expected], dtype=np.float64)
    if not np.allclose(got[:3], want[:3], atol=1.0e-6, rtol=0.0):
        return False
    # q and -q are the same rotation.
    return bool(
        np.allclose(got[3:], want[3:], atol=1.0e-6, rtol=0.0)
        or np.allclose(got[3:], -want[3:], atol=1.0e-6, rtol=0.0)
    )


def _load_model(
    context: DirectiveContext, directive: ModelDirective, weld: WeldDirective | None, where: str
) -> None:
    builder, scene = context.builder, context.scene
    path = context.resolve_path(directive.file)
    if weld is None:
        parent_body, xform = -1, wp.transform(_IDENTITY)
    else:
        try:
            parent_body, X_parent_ref = scene.frame_transform(weld.parent)
        except KeyError as error:
            raise ValueError(
                f"{where}: weld parent {weld.parent!r} is not resolvable here -- a parent model "
                f"must be loaded before its child {directive.name!r} ({error})"
            ) from error
        xform = X_parent_ref * weld.X_PC.to_transform()

    body_start = builder.body_count
    joint_start = builder.joint_count
    shape_start = builder.shape_count
    labels: dict[str, int] = {}
    if directive.kind == "static":
        if weld is not None and weld.parent != "world":
            _fail(
                where,
                f"static model {directive.name!r} must be welded to world, not {weld.parent!r}",
            )
        labels = add_urdf_as_static_shapes(
            builder, path, xform,
            label_prefix=directive.name,
            visual_cfg=context.visual_cfg,
            collision_cfg=context.collision_cfg,
            visual_color=directive.color,
            collect_aabbs=scene.aabbs,
            split_components=directive.split_components,
            color_fn=(
                _component_color_fn(directive.component_colors)
                if directive.component_colors
                else None
            ),
        )
        for link, rgb in directive.link_colors.items():
            prefix = f"{directive.name}/{link}/visual"
            for shape in range(shape_start, builder.shape_count):
                if str(builder.shape_label[shape] or "").startswith(prefix):
                    builder.shape_color[shape] = rgb
    elif directive.kind == "urdf":
        builder.add_urdf(
            str(path), xform=xform, parent_body=parent_body,
            floating=weld is None, **directive.importer_options,
        )
    elif directive.kind == "mjcf":
        floating = {} if weld is not None else {"floating": True}
        builder.add_mjcf(
            str(path), xform=xform, parent_body=parent_body,
            **floating, **directive.importer_options,
        )
    else:
        floating = {} if weld is not None else {"floating": True}
        builder.add_usd(
            str(path), xform=xform, parent_body=parent_body,
            **floating, **directive.importer_options,
        )

    if directive.kind == "urdf" and directive.urdf_fixups:
        label_shapes_by_body(builder, shape_start, builder.shape_count)
        flipped = fix_inverted_mesh_winding(builder, shape_start, builder.shape_count)
        if flipped:
            logger.info(
                f"Flipped {len(flipped)} inward-wound mesh(es) in "
                f"{directive.name}: {', '.join(flipped)}"
            )
        neutralize_textured_shape_colors(builder, shape_start, builder.shape_count)

    if directive.gravity_compensation:
        _enable_gravity_compensation(builder, range(body_start, builder.body_count))

    scene.models[directive.name] = ModelRecord(
        directive=directive, file=path,
        body_start=body_start, body_end=builder.body_count,
        joint_start=joint_start, joint_end=builder.joint_count,
        shape_start=shape_start, shape_end=builder.shape_count,
        parent_body=parent_body, xform=xform, shape_labels=labels,
    )

    if weld is None or "::" not in weld.child:
        return
    link = weld.child.split("::", 1)[1]
    if directive.kind == "static":
        # Static models have no bodies; check the link at least produced shapes.
        if not any(label.startswith(f"{directive.name}/{link}/") for label in labels):
            _fail(
                where,
                f"weld child {weld.child!r} names no geometry-bearing link of {path.name}",
            )
        return
    body = scene.body(weld.child)
    # Newton applies the weld to whichever body the importer makes the articulation root, so
    # naming any other link silently misplaces the model. The importer-independent witness is
    # the base joint it just created: the one joint in this model's range whose parent is the
    # weld parent. (A pose comparison alone cannot see this: ``add_urdf`` does no build-time
    # FK, leaving every ``builder.body_q`` at identity, while MJCF/USD do populate it.)
    roots = [
        int(builder.joint_child[j])
        for j in range(joint_start, builder.joint_count)
        if int(builder.joint_parent[j]) == parent_body
        and body_start <= int(builder.joint_child[j]) < builder.body_count
    ]
    if body not in roots:
        _fail(
            where,
            f"weld child {weld.child!r} is body {body} "
            f"({builder.body_label[body]}), but the importer rooted this model at "
            f"{[str(builder.body_label[r]) for r in roots]}: only a model's ROOT link may be a "
            "weld child, because Newton applies the weld to the articulation root",
        )
    if body != body_start:
        logger.warning(
            f"{where}: weld child {weld.child!r} is body {body}, not the model's "
            f"first body {body_start}"
        )
    if any(not _is_identity(builder.body_q[b]) for b in range(body_start, builder.body_count)):
        # MJCF/USD compose world poses at import time, so the weld can be verified directly.
        X_W_parent = (
            wp.transform(*builder.body_q[parent_body])
            if parent_body >= 0
            else wp.transform(_IDENTITY)
        )
        requested = X_W_parent * xform
        if not _pose_close(builder, body, requested):
            landed = np.array([float(v) for v in builder.body_q[body]][:3])
            miss_mm = 1.0e3 * float(np.linalg.norm(landed - np.array(requested.p, dtype=float)))
            _fail(
                where,
                f"welded body {weld.child!r} landed at {list(builder.body_q[body])}, not at the "
                f"requested {list(requested)} (translation off by {miss_mm:.3f} mm): the root "
                f"body is offset from {path.name}'s import frame, so use the bare model name as "
                f"the weld child (child: {directive.name}) and X_PC then poses the import frame",
            )


def load_directives(
    builder: newton.ModelBuilder,
    path: Path | str,
    *,
    directives: Mapping[str, DirectiveFn] | None = None,
    visual_cfg: newton.ModelBuilder.ShapeConfig,
    collision_cfg: newton.ModelBuilder.ShapeConfig,
) -> LoadedScene:
    """Execute a directives file onto ``builder``, returning the :class:`LoadedScene`."""
    custom: Mapping[str, DirectiveFn] = dict(directives or {})
    reserved = sorted(set(custom) & set(CORE_DIRECTIVES))
    if reserved:
        raise ValueError(
            f"{reserved} are core directive names and cannot be registered as custom"
        )

    parsed = parse_directives(path)
    scene = LoadedScene(builder=builder, directives=parsed)
    welds = [(i, d) for i, (d, _) in enumerate(parsed.entries) if isinstance(d, WeldDirective)]
    consumed: set[int] = set()
    custom_count = 0

    for index, (directive, source) in enumerate(parsed.entries):
        where = f"{source}[entry {index}]"
        context = DirectiveContext(builder, scene, source.parent, visual_cfg, collision_cfg)
        if isinstance(directive, ModelDirective):
            matches = [
                (i, weld) for i, weld in welds
                if weld.child == directive.name or weld.child.startswith(f"{directive.name}::")
            ]
            if len(matches) > 1:
                _fail(where, f"model {directive.name!r} is the child of {len(matches)} welds: "
                             f"{[weld.child for _, weld in matches]}")
            weld = matches[0][1] if matches else None
            if matches:
                consumed.add(matches[0][0])
            logger.debug(f"add_model {directive.name} ({directive.kind}) welded to "
                         f"{weld.parent if weld else 'nothing (floating)'}")
            _load_model(context, directive, weld, where)
        elif isinstance(directive, WeldDirective):
            continue  # consumed by the add_model of its child
        elif isinstance(directive, FrameDirective):
            try:
                body, X_base_ref = scene.frame_transform(directive.base_frame)
            except KeyError as error:
                raise ValueError(
                    f"{where}: add_frame base_frame {directive.base_frame!r}: {error}"
                ) from error
            scene.frames[directive.name] = FrameRecord(
                directive.name, body, X_base_ref * directive.X_PF.to_transform()
            )
            logger.debug(f"add_frame {directive.name} on {directive.base_frame} (body {body})")
        else:
            function = custom.get(directive.kind)
            if function is None:
                _fail(where, f"unknown directive {directive.kind!r}; known custom kinds: "
                             f"{sorted(custom)} (core: {list(CORE_DIRECTIVES)})")
            logger.debug(f"{directive.kind} {directive.params.get('name', '')}")
            function(context, directive.params)
            custom_count += 1

    orphans = [weld.child for i, weld in welds if i not in consumed]
    if orphans:
        raise ValueError(
            f"{parsed.path}: no add_model consumed the weld(s) for child {orphans}; "
            "a weld child must be '<model>' or '<model>::<link>' of a model in this file"
        )
    logger.info(
        f"Loaded {parsed.path}: {len(scene.models)} models, {len(scene.frames)} frames, "
        f"{custom_count} custom directives"
    )
    return scene
