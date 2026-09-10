"""Entry point for the Newton port of the Drake round-belt scene.

The scene itself lives in the ``round_belt_task`` package (constants / scene /
simulation); this file only wires up logging, the viewer and the argument parser.

Run:
    uv run python round_belt_task_simulation.py --viewer null --num-frames 120 --test
    vglrun -d :1 uv run python round_belt_task_simulation.py
"""

from __future__ import annotations

import sys

from loguru import logger

import newton.examples

from round_belt_task.simulation import RoundBeltTaskSimulation
from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material


def configure_logging(level: str = "INFO") -> None:
    """Install this script's loguru sink.  Called from ``__main__`` ONLY.

    Nothing outside ``__main__`` may touch loguru's handlers: a module that calls
    ``logger.remove()`` on import hijacks logging for every importer, and
    scripts/check_round_belt_task_poses.py imports the scene.
    """
    logger.remove()
    logger.add(sys.stdout, level=level,
               format="<level>{level: <7}</level> | <level>{message}</level>")


if __name__ == "__main__":
    # The only place loguru is configured -- see configure_logging().
    configure_logging()
    parser = RoundBeltTaskSimulation.create_parser()
    patch_viewer_shape_names()
    patch_viser_texture_material()
    viewer, args = newton.examples.init(parser)
    viewer._pause = False
    simulation = RoundBeltTaskSimulation(viewer, args)
    if getattr(args, "show_collision", False):
        # Must be set AFTER the simulation is built: attaching the model resets the
        # viewer layer's defaults (viewer.py:609 sets layer.show_collision = False),
        # so anything set before construction is silently discarded.
        viewer.show_collision = True
        logger.info("[VIEWER] show_collision enabled; colliders are drawn alongside visuals.")
    newton.examples.run(simulation, args)
