"""Entry point for the real-time Newton round-belt simulation over LCM.

Speaks the Drake ``magna_simulation`` LCM contract (see ``task_common/lcm_simulation.py``)
so the magna controllers can drive it; this file only wires up logging, the viewer and the
argument parser.

Run:
    uv run python round_belt_lcm_simulation.py --test
    uv run python round_belt_lcm_simulation.py
    vglrun -d :1 uv run python round_belt_lcm_simulation.py --viewer gl
"""

from __future__ import annotations

import sys

import newton.examples
from loguru import logger

from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material

TEST_NUM_STEPS = 400


def configure_logging(level: str = "INFO") -> None:
    """Install this script's loguru sink.  Called from ``__main__`` ONLY."""
    logger.remove()
    logger.add(sys.stdout, level=level,
               format="<level>{level: <7}</level> | <level>{message}</level>")


if __name__ == "__main__":
    # The only place loguru is configured -- see configure_logging().
    configure_logging()
    parser = RoundBeltLcmSimulation.create_parser()
    patch_viewer_shape_names()
    patch_viser_texture_material()
    viewer, args = newton.examples.init(parser)
    viewer._pause = False
    simulation = RoundBeltLcmSimulation(viewer, args)
    if getattr(args, "show_collision", False):
        # Set after construction: attaching the model resets the viewer layer's defaults.
        viewer.show_collision = True
    num_steps = args.num_steps or (TEST_NUM_STEPS if args.test else 0)
    simulation.run(num_steps, args.realtime)
    if args.test:
        simulation.test_final()
    viewer.close()
