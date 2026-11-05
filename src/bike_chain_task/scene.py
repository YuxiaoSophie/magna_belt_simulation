"""Assembly of the bike-chain scene (``assets/bike_chain_task/bike_chain_scene.yaml``)."""

from __future__ import annotations

from pathlib import Path

import newton

from bike_chain_task.constants import SCENE_DIRECTIVES, TABLE_TOP_Z, TABLE_VISUAL_LABEL
from bike_chain_task.directives import EXTENSION_DIRECTIVES
from task_common.scene import SceneInfo, build_task_scene

CHAIN_NAME = "bike_chain"


def build_scene(
    builder: newton.ModelBuilder, directives_path: Path = SCENE_DIRECTIVES
) -> SceneInfo:
    """Build the whole scene into ``builder``; the chain is ``info.belt_*``, the sprockets
    ``info.pulley_*``; no solver."""
    return build_task_scene(
        builder, directives_path, directives=EXTENSION_DIRECTIVES, belt_name=CHAIN_NAME,
        table_visual_label=TABLE_VISUAL_LABEL, table_top_z=TABLE_TOP_Z,
    )
