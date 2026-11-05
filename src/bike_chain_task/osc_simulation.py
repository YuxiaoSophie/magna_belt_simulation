"""The bike-chain sim with the Franka torque-driven by magna's OSC in lock-step
(:class:`RoundBeltOscSimulation` on the chain scene, one ``SolverMuJoCo``)."""

from __future__ import annotations

from bike_chain_task.lcm_simulation import _BikeChainLcmTask
from round_belt_task.osc_simulation import RoundBeltOscSimulation


class BikeChainOscSimulation(_BikeChainLcmTask, RoundBeltOscSimulation):
    """Franka on magna's OSC (private URL), UR position-driven, grippers in-process."""
