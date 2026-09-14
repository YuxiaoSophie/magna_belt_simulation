#!/usr/bin/env python3
"""Preview the timing-belt prototype (belt + pulleys only, no robots) in a viewer.

Run:
    uv run python scripts/view_timing_belt.py --viewer viser
    uv run python scripts/view_timing_belt.py --model chain --scene loop --num-elements 68
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import newton.examples

_spec = importlib.util.spec_from_file_location(
    "check_timing_belt_behaviour", Path(__file__).with_name("check_timing_belt_behaviour.py"))
chk = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = chk  # dataclasses resolve string annotations through sys.modules
_spec.loader.exec_module(chk)


class BeltPreview:
    def __init__(self, viewer, args):
        n = args.num_elements or (135 if args.model == "strip" else 68)
        chk.ARGS.num_elements = n
        chk.STRIP_ARGS.n_w = args.width_cells
        mat = chk.operating_material(args.tri_ke, args.cord_ke)
        if mat.edge_ke_flat is None:  # uncalibrated tri_ke: fall back to the operating point's bending
            mat = chk.replace(mat, edge_ke_flat=chk.SOFT_STRIP_OPERATING_POINT.edge_ke_flat)
        if args.model == "strip":
            print(f"[INFO] strip tri_ke={mat.tri_ke:g} cord_ke={mat.cord_ke:g} edge_ke_flat={mat.edge_ke_flat:.4g}")
        pts = chk.ellipse_points((0.0, 0.0, 0.05), (0.08597, 0.12691), n)  # loop: zero gravity, released from an ellipse
        if args.model == "strip" and args.scene == "pulley":
            self.sim = chk.build_strip_pulley_scene(n, args.width_cells, mat=mat)
        elif args.model == "strip":
            builder = chk.make_strip_builder(0.0)
            info = chk.add_strip(builder, pts, up=(0.0, 0.0, 1.0), closed=True, mat=mat)
            self.sim = chk.StripSim(builder, info)
        elif args.scene == "pulley":
            self.sim = chk.build_pulley_scene(n)
        else:
            builder = chk.make_builder(0.0)
            bodies, joints = chk.add_belt(builder, pts, up=(0.0, 0.0, 1.0), closed=True)
            self.sim = chk.BeltSim(builder, bodies, joints)
        self.viewer = viewer
        self.sim_time = 0.0
        viewer.set_model(self.sim.model)

    def step(self):
        self.sim.step()
        self.sim_time += chk.FRAME_DT

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.sim.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--model", choices=("strip", "chain"), default="strip")
    parser.add_argument("--scene", choices=("pulley", "loop"), default="pulley")
    parser.add_argument("--num-elements", type=int, default=None, help="default 135 (strip) / 68 (chain)")
    parser.add_argument("--width-cells", type=int, default=chk.STRIP_N_W, help="strip cells across the width")
    parser.add_argument("--tri-ke", type=float, default=None, help="strip membrane tri_ke override (default: operating point)")
    parser.add_argument("--cord-ke", type=float, default=None, help="strip edge-cord spring ke override [N/m]")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(BeltPreview(viewer, args), args)
