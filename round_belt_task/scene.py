"""Assembly of the Drake round-belt scene into a ``newton.ModelBuilder``.

The scene is authored as directives -- ``assets/round_belt_task/round_belt_scene.yaml``
(+ ``assets/common/directives/ur10_2f85.yaml``) -- so :func:`build_scene` is
:func:`task_common.scene.build_task_scene` on that file, with the extension directives from
``round_belt_task.directives``.
"""

from __future__ import annotations

from pathlib import Path

import newton

from round_belt_task.constants import SCENE_DIRECTIVES, TABLE_TOP_Z, TABLE_VISUAL_LABEL
from round_belt_task.directives import EXTENSION_DIRECTIVES
from task_common.scene import (  # noqa: F401
    JointConfig,
    SceneInfo,
    build_task_scene,
    make_builder,
)


def build_scene(
    builder: newton.ModelBuilder, directives_path: Path = SCENE_DIRECTIVES
) -> SceneInfo:
    """Build the whole Drake scene into ``builder`` from ``directives_path``; no solver."""
    return build_task_scene(
        builder, directives_path, directives=EXTENSION_DIRECTIVES,
        belt_name="flexible_ellipse_cable", table_visual_label=TABLE_VISUAL_LABEL,
        table_top_z=TABLE_TOP_Z,
    )
