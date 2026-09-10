"""Scene directives: a Drake-shaped YAML recipe executed onto a ``newton.ModelBuilder``.

``schema`` parses (dataclasses, ``!Rpy``, strict keys, number coercion, path resolution);
``runtime`` executes (per-kind importers, weld look-ahead, frames, custom-directive registry).
The schema reference -- syntax, paths, float literals, weld and frame semantics, extension
directives -- is ``docs/scene-directives.md``.

Invariants: at most one weld per model, every weld must be consumed by exactly one model,
and the weld child must be the model's root link.  Poses go through
:func:`utils.transforms.drake_xform` and nothing else.
"""

from __future__ import annotations

from utils.directives.runtime import (
    DirectiveContext,
    DirectiveFn,
    FrameRecord,
    LoadedScene,
    ModelRecord,
    load_directives,
)
from utils.directives.schema import (
    CORE_DIRECTIVES,
    ComponentColorRule,
    CustomDirective,
    Directive,
    DirectiveFile,
    FrameDirective,
    ModelDirective,
    ModelKind,
    Pose,
    Vec3,
    WeldDirective,
    as_float,
    as_floats,
    as_vec3,
    parse_directives,
)

__all__ = [
    "as_float",
    "as_floats",
    "as_vec3",
    "ComponentColorRule",
    "CORE_DIRECTIVES",
    "CustomDirective",
    "Directive",
    "DirectiveContext",
    "DirectiveFile",
    "DirectiveFn",
    "FrameDirective",
    "FrameRecord",
    "load_directives",
    "LoadedScene",
    "ModelDirective",
    "ModelKind",
    "ModelRecord",
    "parse_directives",
    "Pose",
    "Vec3",
    "WeldDirective",
]
