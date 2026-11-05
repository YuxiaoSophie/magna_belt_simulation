"""The bike-chain scene's one physics solver: ``SolverMuJoCo`` (mujoco_warp) with its own contacts."""

from __future__ import annotations

import newton
import numpy as np
from newton.solvers import SolverMuJoCo


def solref_gains(timeconst: float, dampratio: float = 1.0) -> tuple[float, float]:
    """Newton (ke, kd) that ``SolverMuJoCo`` compiles into geom solref (timeconst, dampratio)."""
    return 1.0 / (timeconst * dampratio) ** 2, 2.0 / timeconst


def make_mujoco_solver(model: newton.Model, *, iterations: int = 30, ls_iterations: int = 10,
                       nconmax: int = 2048, njmax: int = 8192, impratio: float = 1.0,
                       solref: tuple[float, float] | None = (0.02, 1.0),
                       solref_overrides: tuple = (), enable_multiccd: bool = False,
                       save_to_mjcf: str | None = None) -> SolverMuJoCo:
    """MuJoCo for chain, sprockets, robots and statics; ``solref`` (None: leave ke/kd) is written
    to every shape's ke/kd first, then each ``(shapes, (timeconst, dampratio))`` override."""
    if solref is not None:  # ke/kd -> solref is unit-mass: the cable default (1e4, 0.1) is 20 s
        ke, kd = solref_gains(*solref)
        model.shape_material_ke.fill_(ke)
        model.shape_material_kd.fill_(kd)
    for shapes, ref in solref_overrides:
        for array, value in zip((model.shape_material_ke, model.shape_material_kd),
                                solref_gains(*ref)):
            host = array.numpy()
            host[list(shapes)] = value
            array.assign(host)
    solver = SolverMuJoCo(
        model, solver="newton", integrator="implicitfast", cone="elliptic", impratio=impratio,
        iterations=iterations, ls_iterations=ls_iterations, nconmax=nconmax, njmax=njmax,
        use_mujoco_contacts=True, enable_multiccd=enable_multiccd, save_to_mjcf=save_to_mjcf)
    restore_body_invweight0(solver)
    return solver


def restore_body_invweight0(solver: SolverMuJoCo) -> None:
    """MuJoCo C's ``body_invweight0`` into mujoco_warp; re-apply after any set_const
    (``notify_model_changed`` with body/joint flags)."""
    # warp swaps a zero translational invweight (sprocket hinge through its COM) for the
    # rotational one (~5500 vs ~50 for a link): chain-sprocket contacts ~100x soft, chain sinks.
    w = solver.mjw_model.body_invweight0
    w.assign(np.broadcast_to(solver.mj_model.body_invweight0.astype(np.float32), w.shape + (2,)))
