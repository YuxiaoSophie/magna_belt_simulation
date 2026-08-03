# SPDX-License-Identifier: Apache-2.0
#
# Elliptical closed ring belt with a strong, smoothly distributed twist.
# The ellipse centerline is preserved exactly, so the belt never collapses
# into a figure-eight or opens like a loose paper strip.

from __future__ import annotations

import math

import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


# SI units: metres, kilograms, seconds.
BELT_MAJOR_DIAMETER = 0.248
BELT_MINOR_DIAMETER = 0.168
BELT_STRIP_WIDTH = 0.05

# Mesh resolution.
CIRCUMFERENCE_CELLS = 128
WIDTH_CELLS = 10

PARTICLE_RADIUS = 0.0010
GROUND_EPSILON = 0.0001
CLOTH_DENSITY = 10.0

# Stronger twist. Each cross-section still rotates only around the local tangent,
# so every cross-section center remains on the original elliptical ring.
# 60 degrees is visibly stronger without forcing the closed strip into a figure-eight.
MAX_TWIST_ANGLE = math.radians(60.0)
TWIST_END_TIME = 2.0
TWIST_ANGULAR_VELOCITY = MAX_TWIST_ANGLE / TWIST_END_TIME


@wp.kernel
def apply_closed_ring_twist(
    rest_positions: wp.array(dtype=wp.vec3),
    theta_values: wp.array(dtype=float),
    width_offsets: wp.array(dtype=float),
    time_value: wp.array(dtype=float),
    angular_velocity: float,
    max_twist_angle: float,
    end_time: float,
    dt: float,
    major_radius: float,
    minor_radius: float,
    center_z: float,
    pos_0: wp.array(dtype=wp.vec3),
    pos_1: wp.array(dtype=wp.vec3),
    vel_0: wp.array(dtype=wp.vec3),
    vel_1: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    t = time_value[0]
    clamped_t = wp.min(t, end_time)
    amplitude = wp.min(clamped_t * angular_velocity, max_twist_angle)

    theta = theta_values[tid]
    offset = width_offsets[tid]

    # Ellipse centerline.
    center = wp.vec3(
        major_radius * wp.cos(theta),
        minor_radius * wp.sin(theta),
        center_z,
    )

    # Unit tangent of the ellipse.
    tangent_raw = wp.vec3(
        -major_radius * wp.sin(theta),
        minor_radius * wp.cos(theta),
        0.0,
    )
    tangent = wp.normalize(tangent_raw)

    # Horizontal in-plane normal. This and +Z span the belt cross-section.
    horizontal_normal = wp.normalize(
        wp.vec3(
            minor_radius * wp.cos(theta),
            major_radius * wp.sin(theta),
            0.0,
        )
    )
    vertical = wp.vec3(0.0, 0.0, 1.0)

    # A periodic distributed twist. sin(2 theta) is zero at four locations and
    # is exactly periodic at the seam, so the closed loop remains continuous.
    local_angle = amplitude * wp.sin(2.0 * theta)
    c = wp.cos(local_angle)
    s = wp.sin(local_angle)

    # Rotate the original vertical strip direction around the local tangent.
    # Rodrigues simplifies to vertical*cos + horizontal_normal*sin here.
    cross_section_direction = vertical * c + horizontal_normal * s
    p = center + offset * cross_section_direction

    pos_0[tid] = p
    pos_1[tid] = p
    vel_0[tid] = wp.vec3(0.0, 0.0, 0.0)
    vel_1[tid] = wp.vec3(0.0, 0.0, 0.0)

    if tid == 0 and t <= end_time:
        time_value[0] = t + dt


def build_elliptical_ring_mesh(
    major_diameter: float,
    minor_diameter: float,
    strip_width: float,
    circumference_cells: int,
    width_cells: int,
    center_z: float,
) -> tuple[list[wp.vec3], list[int], list[float], list[float]]:
    """Create a seamless vertical cloth strip wrapped around an ellipse."""
    if major_diameter <= 0.0 or minor_diameter <= 0.0:
        raise ValueError("The two belt diameters must be positive.")
    if strip_width <= 0.0:
        raise ValueError("The belt strip width must be positive.")
    if circumference_cells < 8:
        raise ValueError("circumference_cells must be at least 8.")
    if width_cells < 1:
        raise ValueError("width_cells must be at least 1.")

    a = 0.5 * major_diameter
    b = 0.5 * minor_diameter
    row_size = width_cells + 1

    vertices: list[wp.vec3] = []
    indices: list[int] = []
    theta_values: list[float] = []
    width_offsets: list[float] = []

    for i in range(circumference_cells):
        theta = 2.0 * math.pi * i / circumference_cells
        x = a * math.cos(theta)
        y = b * math.sin(theta)

        for j in range(row_size):
            offset = -0.5 * strip_width + strip_width * j / width_cells
            vertices.append(wp.vec3(x, y, center_z + offset))
            theta_values.append(theta)
            width_offsets.append(offset)

    def vertex_id(i: int, j: int) -> int:
        return (i % circumference_cells) * row_size + j

    for i in range(circumference_cells):
        i_next = (i + 1) % circumference_cells
        for j in range(width_cells):
            v00 = vertex_id(i, j)
            v10 = vertex_id(i_next, j)
            v11 = vertex_id(i_next, j + 1)
            v01 = vertex_id(i, j + 1)

            indices.extend((v00, v10, v01))
            indices.extend((v10, v11, v01))

    return vertices, indices, theta_values, width_offsets


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.iterations = 30
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.center_z = PARTICLE_RADIUS + GROUND_EPSILON + 0.5 * args.strip_width

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.add_ground_plane()

        vertices, indices, theta_values, width_offsets = build_elliptical_ring_mesh(
            major_diameter=BELT_MAJOR_DIAMETER,
            minor_diameter=BELT_MINOR_DIAMETER,
            strip_width=args.strip_width,
            circumference_cells=args.circumference_cells,
            width_cells=args.width_cells,
            center_z=self.center_z,
        )

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=CLOTH_DENSITY,
            # Keep all of your stiff belt material settings.
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=5.0e2,
            edge_ke=2.5e3,
            edge_kd=5.0e1,
            particle_radius=PARTICLE_RADIUS,
        )

        builder.color(include_bending=True)
        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 1.0

        # The whole belt is driven kinematically by a closed-loop-preserving
        # twist map. This guarantees that its centerline always remains a ring.
        flags = self.model.particle_flags.numpy()
        flags[:] = flags[:] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=PARTICLE_RADIUS,
            particle_self_contact_margin=1.5 * PARTICLE_RADIUS,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.rest_positions = wp.array(vertices, dtype=wp.vec3)
        self.theta_values = wp.array(theta_values, dtype=float)
        self.width_offsets = wp.array(width_offsets, dtype=float)
        self.twist_time = wp.zeros(1, dtype=float)

        self.viewer.set_model(self.model)
        self.camera_set = False
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                kernel=apply_closed_ring_twist,
                dim=self.state_0.particle_q.shape[0],
                inputs=[
                    self.rest_positions,
                    self.theta_values,
                    self.width_offsets,
                    self.twist_time,
                    TWIST_ANGULAR_VELOCITY,
                    MAX_TWIST_ANGLE,
                    TWIST_END_TIME,
                    self.sim_dt,
                    0.5 * BELT_MAJOR_DIAMETER,
                    0.5 * BELT_MINOR_DIAMETER,
                    self.center_z,
                ],
                outputs=[
                    self.state_0.particle_q,
                    self.state_1.particle_q,
                    self.state_0.particle_qd,
                    self.state_1.particle_qd,
                ],
            )

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if not self.camera_set and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                wp.vec3(0.0, -0.40, 0.25),
                -28.0,
                90.0,
            )
            self.camera_set = True

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "belt particles remain near or above the ground",
            lambda q, qd: q[2] > -PARTICLE_RADIUS,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--gravity",
            type=float,
            default=0.0,
            help="Gravity acceleration along Z.",
        )
        parser.add_argument(
            "--strip-width",
            type=float,
            default=BELT_STRIP_WIDTH,
            help="Width of the cloth belt strip in meters.",
        )
        parser.add_argument(
            "--circumference-cells",
            type=int,
            default=CIRCUMFERENCE_CELLS,
            help="Number of cells around the closed elliptical loop.",
        )
        parser.add_argument(
            "--width-cells",
            type=int,
            default=WIDTH_CELLS,
            help="Number of cells across the belt strip width.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)