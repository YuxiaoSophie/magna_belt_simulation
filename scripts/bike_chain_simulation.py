"""Entry point for the bike-chain simulation (one ``SolverMuJoCo``, robots held at their defaults).

Run:
    uv run --frozen python scripts/bike_chain_simulation.py --viewer null --test --num-steps 600
    uv run --frozen python scripts/bike_chain_simulation.py --viewer viser --viewer-port 8085
    vglrun -d :1 uv run --frozen python scripts/bike_chain_simulation.py --viewer gl
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import newton.examples
import warp as wp
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bike_chain_task.simulation import BikeChainTaskSimulation
from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material


def configure_logging(level: str = "INFO") -> None:
    """Install this script's loguru sink.  Called from ``__main__`` ONLY."""
    logger.remove()
    logger.add(sys.stdout, level=level,
               format="<level>{level: <7}</level> | <level>{message}</level>")


def make_viser(port: int) -> newton.viewer.ViewerViser:
    """Viser on exactly ``port`` (viser would silently move to the next free one)."""
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"--viewer-port {port} is in use")
    return newton.viewer.ViewerViser(port=port, label="bike chain")


if __name__ == "__main__":
    configure_logging()
    parser = BikeChainTaskSimulation.create_parser()
    parser.add_argument("--num-steps", dest="num_frames", type=int, default=argparse.SUPPRESS,
                        help="alias of --num-frames (frames of the null viewer)")
    parser.add_argument("--viewer-port", type=int, default=8080, help="viser port")
    patch_viewer_shape_names()
    patch_viser_texture_material()
    args = parser.parse_args()
    if args.viewer not in ("gl", "viser", "null"):
        raise SystemExit(f"--viewer {args.viewer}: use gl, viser or null")
    if args.viewer == "viser":  # newton.examples.init has no viser port
        if args.device:
            wp.set_device(args.device)
        viewer = make_viser(args.viewer_port)
    else:
        viewer, args = newton.examples.init(parser)
    simulation = BikeChainTaskSimulation(viewer, args)
    if args.show_collision:
        # Set after construction: attaching the model resets the viewer layer's defaults.
        viewer.show_collision = True
    newton.examples.run(simulation, args)
    logger.info(f"[RTF] physics {simulation.rtf():.2f}x realtime")
