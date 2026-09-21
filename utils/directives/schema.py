"""Directive schema and parsing: the builder-free half of the loader.

Dataclasses for the four Drake-shaped core directives plus the custom-kind catch-all,
the ``!Rpy`` tag, strict key validation, number coercion (:func:`as_float` and friends,
public for custom directives) and path resolution.  Nothing here touches a
``ModelBuilder``.  The one Newton call is ``newton.utils.download_asset``, made whenever a
``newton_asset://`` path is resolved -- at parse time for an ``add_directives`` include.
Execution is ``utils.directives.runtime``; the schema reference is ``docs/scene-directives.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NoReturn, TypeVar

import warp as wp
import yaml

import newton.utils

from utils.transforms import drake_xform

Vec3 = tuple[float, float, float]
ModelKind = Literal["static", "urdf", "mjcf", "usd"]

CORE_DIRECTIVES = ("add_model", "add_weld", "add_frame", "add_directives")
_ASSET_SCHEME = "newton_asset://"
_EXTENSION_KINDS = {
    ".urdf": "urdf", ".xml": "mjcf", ".usd": "usd", ".usda": "usd", ".usdc": "usd",
}
_POSE_KEYS = frozenset({"translation", "rotation"})
_STATIC_ONLY = {
    "color", "link_colors", "split_components", "component_colors", "keep_visual_material",
}
_ARTICULATED_ONLY = {"default_joint_positions", "gravity_compensation", "importer_options"}
_MODEL_KEYS: dict[str, frozenset[str]] = {
    "static": frozenset({"name", "file", "static", "kind"} | _STATIC_ONLY),
    "urdf": frozenset({"name", "file", "static", "kind", "urdf_fixups"} | _ARTICULATED_ONLY),
    "mjcf": frozenset({"name", "file", "static", "kind"} | _ARTICULATED_ONLY),
    "usd": frozenset({"name", "file", "static", "kind"} | _ARTICULATED_ONLY),
}


# --------------------------------------------------------------------------------------
# parsing primitives
# --------------------------------------------------------------------------------------


def _fail(where: str, message: str) -> NoReturn:
    raise ValueError(f"{where}: {message}")


def as_float(value: Any, where: str, key: str) -> float:
    """Coerce a YAML scalar to ``float``; ``ValueError`` naming ``where`` and ``key`` if it is
    not a number.  PyYAML is YAML 1.1, so ``8.95207485e01`` and ``2e4`` arrive as strings;
    ``float()`` coerces those, and only a non-numeric value (``abc``) fails."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, dict)) or value is None:
        _fail(where, f"{key} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        _fail(where, f"{key} = {value!r} is not a number")


def as_floats(value: Any, where: str, key: str, count: int) -> tuple[float, ...]:
    """A list of exactly ``count`` numbers, each through :func:`as_float`."""
    if not isinstance(value, (list, tuple)) or len(value) != count:
        _fail(where, f"{key} must be a list of {count} numbers, got {value!r}")
    return tuple(as_float(v, where, f"{key}[{i}]") for i, v in enumerate(value))


def as_vec3(value: Any, where: str, key: str) -> Vec3:
    """A list of 3 numbers as a :data:`Vec3`, each through :func:`as_float`."""
    x, y, z = as_floats(value, where, key, 3)
    return (x, y, z)


def _mapping(value: Any, where: str, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(where, f"{key} must be a mapping, got {value!r}")
    return value


def _check_keys(block: Mapping[str, Any], allowed: frozenset[str], where: str, what: str) -> None:
    unknown = [str(k) for k in block if k not in allowed]
    if unknown:
        _fail(where, f"unknown {what} key(s) {unknown}; allowed: {sorted(allowed)}")


class _Rpy:
    """The payload of a ``!Rpy { deg: [...] }`` tag, validated where it is used."""

    __slots__ = ("deg",)

    def __init__(self, deg: Any) -> None:
        self.deg = deg


class _DirectivesLoader(yaml.SafeLoader):
    """SafeLoader plus Drake's ``!Rpy`` tag."""


def _rpy_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> _Rpy:
    block = loader.construct_mapping(node, deep=True)
    if "rad" in block:
        raise ValueError("!Rpy { rad: ... } is not supported; use !Rpy { deg: [r, p, y] }")
    unknown = [str(k) for k in block if k != "deg"]
    if unknown or "deg" not in block:
        raise ValueError(f"!Rpy takes exactly one key 'deg', got {sorted(map(str, block))}")
    return _Rpy(block["deg"])


_DirectivesLoader.add_constructor("!Rpy", _rpy_constructor)


# --------------------------------------------------------------------------------------
# directive dataclasses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """A Drake ``translation`` + ``!Rpy { deg: ... }`` pose."""

    translation: Vec3 = (0.0, 0.0, 0.0)
    rpy_deg: Vec3 = (0.0, 0.0, 0.0)

    def to_transform(self) -> wp.transform:
        return drake_xform(self.translation, self.rpy_deg)


@dataclass
class ComponentColorRule:
    """Colour one split mesh component; first matching rule wins, non-matching -> asset colour."""

    color: Vec3
    max_span: float
    near_local_xy: tuple[float, float]
    radius: float


@dataclass
class ModelDirective:
    name: str
    file: str
    kind: ModelKind
    default_joint_positions: dict[str, list[float]] = field(default_factory=dict)
    color: Vec3 | None = None
    link_colors: dict[str, Vec3] = field(default_factory=dict)
    split_components: bool = False
    component_colors: list[ComponentColorRule] = field(default_factory=list)
    keep_visual_material: list[str] = field(default_factory=list)
    gravity_compensation: bool = False
    urdf_fixups: bool = True
    importer_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeldDirective:
    parent: str
    child: str
    X_PC: Pose = field(default_factory=Pose)


@dataclass
class FrameDirective:
    name: str
    base_frame: str
    X_PF: Pose = field(default_factory=Pose)


@dataclass
class CustomDirective:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


Directive = ModelDirective | WeldDirective | FrameDirective | CustomDirective
_D = TypeVar("_D", bound=Directive)


# --------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------


def _parse_pose(block: Any, where: str, key: str, extra: frozenset[str] = frozenset()) -> Pose:
    if block is None:
        return Pose()
    body = _mapping(block, where, key)
    _check_keys(body, frozenset(_POSE_KEYS | extra), where, key)
    translation = as_vec3(
        body.get("translation", [0.0, 0.0, 0.0]), where, f"{key}.translation"
    )
    rotation = body.get("rotation")
    if rotation is None:
        return Pose(translation, (0.0, 0.0, 0.0))
    if not isinstance(rotation, _Rpy):
        _fail(where, f"{key}.rotation must be a !Rpy {{ deg: [r, p, y] }} tag, got {rotation!r}")
    return Pose(translation, as_vec3(rotation.deg, where, f"{key}.rotation.deg"))


def _parse_model(block: Mapping[str, Any], where: str) -> ModelDirective:
    for required in ("name", "file"):
        if required not in block:
            _fail(where, f"add_model needs a {required!r}")
    name, source = str(block["name"]), str(block["file"])
    kind = block.get("kind")
    if block.get("static", False):
        if kind not in (None, "static"):
            _fail(where, f"static: true conflicts with kind: {kind!r}")
        kind = "static"
    if kind is None:
        kind = _EXTENSION_KINDS.get(Path(source).suffix.lower())
        if kind is None:
            _fail(where, f"cannot infer kind from {source!r}; set kind: urdf|mjcf|usd|static")
    if kind not in _MODEL_KEYS:
        _fail(where, f"kind must be one of {sorted(_MODEL_KEYS)}, got {kind!r}")
    if kind == "static" and Path(source).suffix.lower() != ".urdf":
        _fail(where, f"static models must be .urdf files, got {source!r}")
    _check_keys(block, _MODEL_KEYS[kind], where, f"add_model (kind {kind})")

    color = block.get("color")
    link_colors = {
        str(link): as_vec3(rgb, where, f"link_colors[{link!r}]")
        for link, rgb in _mapping(block.get("link_colors", {}), where, "link_colors").items()
    }
    split_components = bool(block.get("split_components", False))
    rules: list[ComponentColorRule] = []
    for i, raw in enumerate(block.get("component_colors", []) or []):
        rule = _mapping(raw, where, f"component_colors[{i}]")
        _check_keys(
            rule, frozenset({"color", "max_span", "near_local_xy", "radius"}),
            where, f"component_colors[{i}]",
        )
        near_key = f"component_colors[{i}].near_local_xy"
        near = as_floats(rule.get("near_local_xy", [0.0, 0.0]), where, near_key, 2)
        rules.append(
            ComponentColorRule(
                color=as_vec3(rule["color"], where, f"component_colors[{i}].color"),
                max_span=as_float(
                    rule.get("max_span", float("inf")), where, f"component_colors[{i}].max_span"
                ),
                near_local_xy=(near[0], near[1]),
                radius=as_float(
                    rule.get("radius", float("inf")), where, f"component_colors[{i}].radius"
                ),
            )
        )
    if rules and not split_components:
        _fail(where, "component_colors requires split_components: true")
    keep = block.get("keep_visual_material", []) or []
    if not isinstance(keep, (list, tuple)):
        _fail(where, f"keep_visual_material must be a list of visual names, got {keep!r}")

    defaults: dict[str, list[float]] = {}
    for joint, values in _mapping(
        block.get("default_joint_positions", {}), where, "default_joint_positions"
    ).items():
        if not isinstance(values, (list, tuple)):
            _fail(where, f"default_joint_positions[{joint!r}] must be a list, got {values!r}")
        defaults[str(joint)] = [
            as_float(v, where, f"default_joint_positions[{joint!r}][{i}]")
            for i, v in enumerate(values)
        ]

    return ModelDirective(
        name=name,
        file=source,
        kind=kind,  # type: ignore[arg-type]
        default_joint_positions=defaults,
        color=None if color is None else as_vec3(color, where, "color"),
        link_colors=link_colors,
        split_components=split_components,
        component_colors=rules,
        keep_visual_material=[str(name) for name in keep],
        gravity_compensation=bool(block.get("gravity_compensation", False)),
        urdf_fixups=bool(block.get("urdf_fixups", True)),
        importer_options=dict(
            _mapping(block.get("importer_options", {}), where, "importer_options")
        ),
    )


def _parse_weld(block: Mapping[str, Any], where: str) -> WeldDirective:
    _check_keys(block, frozenset({"parent", "child", "X_PC"}), where, "add_weld")
    for required in ("parent", "child"):
        if required not in block:
            _fail(where, f"add_weld needs a {required!r}")
    return WeldDirective(
        parent=str(block["parent"]),
        child=str(block["child"]),
        X_PC=_parse_pose(block.get("X_PC"), where, "X_PC"),
    )


def _parse_frame(block: Mapping[str, Any], where: str) -> FrameDirective:
    _check_keys(block, frozenset({"name", "X_PF"}), where, "add_frame")
    if "name" not in block or "X_PF" not in block:
        _fail(where, "add_frame needs a 'name' and an 'X_PF'")
    pose_block = _mapping(block["X_PF"], where, "X_PF")
    if "base_frame" not in pose_block:
        _fail(where, "add_frame's X_PF needs a 'base_frame'")
    return FrameDirective(
        name=str(block["name"]),
        base_frame=str(pose_block["base_frame"]),
        X_PF=_parse_pose(pose_block, where, "X_PF", frozenset({"base_frame"})),
    )


def _resolve(source: Path, file: str, where: str) -> Path:
    """``newton_asset://<asset>/<rel>`` | absolute | relative to the yaml's directory."""
    base = source.parent if source.suffix else source
    if file.startswith(_ASSET_SCHEME):
        asset, _, relative = file[len(_ASSET_SCHEME):].partition("/")
        if not asset or not relative:
            _fail(where, f"{file!r} must be newton_asset://<asset>/<relative/path>")
        path = Path(newton.utils.download_asset(asset)).joinpath(*relative.split("/"))
    else:
        path = Path(file)
        path = path if path.is_absolute() else (base / path)
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{where}: {file!r} resolves to {path}, which does not exist")
    return path


@dataclass
class DirectiveFile:
    """A parsed directives file: ``entries`` is flat (``add_directives`` expanded in place)."""

    path: Path
    entries: list[tuple[Directive, Path]] = field(default_factory=list)

    def _find(self, kind: type[_D], match: Callable[[_D], bool], what: str) -> _D:
        for directive, _ in self.entries:
            if isinstance(directive, kind) and match(directive):
                return directive
        raise KeyError(f"no {what} in {self.path}")

    def model(self, name: str) -> ModelDirective:
        return self._find(ModelDirective, lambda d: d.name == name, f"add_model named {name!r}")

    def weld(self, child: str) -> WeldDirective:
        return self._find(
            WeldDirective, lambda d: d.child == child, f"add_weld for child {child!r}"
        )

    def frame(self, name: str) -> FrameDirective:
        return self._find(FrameDirective, lambda d: d.name == name, f"add_frame named {name!r}")

    def custom(self, kind: str, name: str | None = None) -> CustomDirective:
        return self._find(
            CustomDirective,
            lambda d: d.kind == kind and (name is None or d.params.get("name") == name),
            f"{kind!r} directive named {name!r}",
        )

    def resolve_path(self, file: str, source: Path) -> Path:
        """Resolve an authored ``file:`` against ``source`` (a yaml path or its directory)."""
        return _resolve(Path(source), file, str(self.path))


def _parse_into(
    path: Path, entries: list[tuple[Directive, Path]], stack: tuple[Path, ...]
) -> None:
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"add_directives recursion: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.load(handle, Loader=_DirectivesLoader)
    if not isinstance(document, Mapping) or list(document) != ["directives"]:
        raise ValueError(f"{path}: expected a single top-level 'directives:' key")
    items = document["directives"]
    if not isinstance(items, list):
        raise ValueError(f"{path}: 'directives:' must be a list")
    for index, item in enumerate(items):
        where = f"{path}[{index}]"
        if not isinstance(item, Mapping) or len(item) != 1:
            raise ValueError(f"{where}: each entry must be a single-key mapping, got {item!r}")
        (kind, raw), = item.items()
        block = _mapping(raw if raw is not None else {}, where, str(kind))
        where = f"{where} {kind}"
        if kind == "add_model":
            entries.append((_parse_model(block, where), path))
        elif kind == "add_weld":
            entries.append((_parse_weld(block, where), path))
        elif kind == "add_frame":
            entries.append((_parse_frame(block, where), path))
        elif kind == "add_directives":
            _check_keys(block, frozenset({"file"}), where, "add_directives")
            if "file" not in block:
                _fail(where, "add_directives needs a 'file'")
            _parse_into(_resolve(path, str(block["file"]), where), entries, (*stack, path))
        else:
            entries.append((CustomDirective(str(kind), dict(block)), path))


def parse_directives(path: Path | str) -> DirectiveFile:
    """Parse a directives yaml (and everything it includes) into a flat file; no builder."""
    root = Path(path).resolve()
    entries: list[tuple[Directive, Path]] = []
    _parse_into(root, entries, ())
    models: set[str] = set()
    frames: set[str] = set()
    for directive, source in entries:
        if isinstance(directive, ModelDirective):
            if directive.name in models:
                raise ValueError(f"{source}: duplicate model name {directive.name!r}")
            models.add(directive.name)
        elif isinstance(directive, FrameDirective):
            if directive.name in frames:
                raise ValueError(f"{source}: duplicate frame name {directive.name!r}")
            frames.add(directive.name)
    clash = models & frames
    if clash:
        raise ValueError(f"{root}: {sorted(clash)} used as both a model and a frame name")
    return DirectiveFile(root, entries)
