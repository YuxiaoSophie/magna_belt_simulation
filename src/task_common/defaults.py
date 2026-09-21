"""Contact materials, gripper drive and solver settings shared by the belt-task simulations."""

from __future__ import annotations

import newton

# Cable material: the builder default and every shape's material before the pad override.
CABLE_CONTACT_KE = 1.0e4
CABLE_CONTACT_KD = 1.0e-5 * CABLE_CONTACT_KE
CABLE_CONTACT_MU = 1.0

# Gripper-pad contact used by the mjc -> vbd proxy coupling.
GRIPPER_CONTACT_KE = 2.0e4
GRIPPER_CONTACT_KD = 20.0
GRIPPER_CONTACT_MU = 4.0

# 2F-85 driver position gains and effort limit.
GRIPPER_DRIVE_KE = 180.0
GRIPPER_DRIVE_KD = 80.0
GRIPPER_EFFORT_LIMIT = 1.0

PROXY_ITERATIONS = 1
PROXY_MASS_SCALE = 10.0
PROXY_COUPLING_MODE = "lagged"

VBD_ITERATIONS = 20
VBD_RIGID_AVBD_BETA = 1.0e2
VBD_RIGID_CONTACT_K_START = 3.0e3
VBD_RIGID_CONTACT_BUFFER_SIZE = 256
MUJOCO_ITERATIONS = 30
MUJOCO_LS_ITERATIONS = 10


def make_visual_cfg() -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=0.0, has_shape_collision=False, has_particle_collision=False,
        collision_group=0, is_visible=True,
    )


def make_robust_table_collision_cfg(visible=True) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(
        density=0.0, ke=CABLE_CONTACT_KE, kd=CABLE_CONTACT_KD, mu=CABLE_CONTACT_MU,
        has_shape_collision=True, has_particle_collision=True,
        collision_group=1, is_visible=visible,
    )
