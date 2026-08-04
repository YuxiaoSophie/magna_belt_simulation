from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

# Belt geometry (metres, kilograms, seconds)
BELT_MAJOR_DIAMETER = 0.248
BELT_MINOR_DIAMETER = 0.168
BELT_STRIP_WIDTH = 0.050

# Mesh resolution.
CIRCUMFERENCE_CELLS = 128
WIDTH_CELLS = 10

PARTICLE_RADIUS = 0.0010
GROUND_EPSILON = 0.0001
CLOTH_DENSITY = 10.0  # kg/m^2

# Ordinary membrane stiffness.
TRI_KE = 1.0e3
TRI_KA = 1.0e3
TRI_KD = 1.0e2

# Ordinary cloth bending stiffness. This stays relatively low so the belt can
# still bend around its circumference and wrap around pulleys.
BASE_EDGE_KE = 1.0e1
BASE_EDGE_KD = 0.0

# Very high bending stiffness for hinges that bend a cross-sectional rib.
RIB_BENDING_KE = 2.0e4
RIB_BENDING_KD = 2.0e1

# Additional distance springs inside every cross-sectional line.
RIB_SPRING_KE = 2.0e4
RIB_SPRING_KD = 2.0e1

# More iterations/substeps help the stiff constraints converge.
SIM_FPS = 60
SIM_SUBSTEPS = 20
VBD_ITERATIONS = 20

# Twisting drive 
TWIST_WAVENUMBER = 2 # number of twist lobes around the loop (k)
TWIST_MAX_ANGLE = math.radians(60.0) # peak twist of the band (radians)
TWIST_RAMP_TIME = 4.0 # seconds to ramp from 0 to the peak, then hold


# Twist kernels
@wp.kernel
def init_twist(
    # input
    driven_indices: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    rot_centers: wp.array[wp.vec3],
    rot_axes: wp.array[wp.vec3],
    t: wp.array[float],
    # output
    roots: wp.array[wp.vec3],
    roots_to_ps: wp.array[wp.vec3],
):
    """Cache each driven particle's offset relative to its twist axis.

    The twist axis passes through the rib's pinned centerline point along the
    ring's local tangent, so 'root' (the foot on the axis) is the centerline
    point and 'root_to_p' is the perpendicular Z offset of the top/bottom edge.
    """
    tid = wp.tid()
    v_index = driven_indices[tid]

    p = pos[v_index]
    rot_center = rot_centers[tid]
    rot_axis = rot_axes[tid]

    op = p - rot_center
    axial = wp.dot(op, rot_axis) * rot_axis
    foot = rot_center + axial       # nearest point on the twist axis line

    roots[tid] = foot
    roots_to_ps[tid] = p - foot     # perpendicular offset from the axis

    if tid == 0:
        t[0] = 0.0


@wp.kernel
def apply_twist(
    # input
    driven_indices: wp.array[wp.int32],
    rot_axes: wp.array[wp.vec3],
    roots: wp.array[wp.vec3],
    roots_to_ps: wp.array[wp.vec3],
    angle_mults: wp.array[float],
    t: wp.array[float],
    max_angle: float,
    ramp_time: float,
    dt: float,
    # output
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    tid = wp.tid()
    cur_t = t[0]

    v_index = driven_indices[tid]
    rot_axis = rot_axes[tid]

    ux = rot_axis[0]
    uy = rot_axis[1]
    uz = rot_axis[2]

    # Ramp the amplitude linearly, then hold.
    amp = max_angle
    if ramp_time > 0.0:
        s = cur_t / ramp_time
        if s < 1.0:
            amp = max_angle * s

    theta = amp * angle_mults[tid]

    R = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )

    # Rotate the original offset by the current absolute angle (non-incremental,
    # so it cannot drift).
    foot = roots[tid]
    perp = roots_to_ps[tid]
    p_rot = foot + R * perp

    pos_0[v_index] = p_rot
    pos_1[v_index] = p_rot

    if tid == 0:
        t[0] = cur_t + dt


# Mesh creation
def build_elliptical_ring_mesh(
    major_diameter: float,
    minor_diameter: float,
    strip_width: float,
    circumference_cells: int,
    width_cells: int,
    bottom_z: float,
) -> tuple[list[wp.vec3], list[int]]:
    """Create a seamless vertical cloth strip wrapped around an ellipse.

    local particle id = i * (width_cells + 1) + j
        i = circumferential position around the closed belt
        j = position across the belt width
    """
    if major_diameter <= 0.0 or minor_diameter <= 0.0:
        raise ValueError("The two belt diameters must be positive.")
    if strip_width <= 0.0:
        raise ValueError("The belt strip width must be positive.")
    if circumference_cells < 8:
        raise ValueError("circumference_cells must be at least 8.")
    if width_cells < 2:
        raise ValueError("width_cells must be at least 2 for rib bending.")

    a = 0.5 * major_diameter
    b = 0.5 * minor_diameter
    row_size = width_cells + 1

    vertices: list[wp.vec3] = []
    indices: list[int] = []

    # The loop lies in XY; belt width extends along Z.
    for i in range(circumference_cells):
        theta = 2.0 * math.pi * i / circumference_cells
        x = a * math.cos(theta)
        y = b * math.sin(theta)

        for j in range(row_size):
            z = bottom_z + strip_width * j / width_cells
            vertices.append(wp.vec3(x, y, z))

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

    return vertices, indices


# Rib-stiffening helpers
def belt_particle_id(
    particle_start: int,
    circumference_index: int,
    width_index: int,
    circumference_cells: int,
    width_cells: int,
) -> int:
    row_size = width_cells + 1
    local_id = (circumference_index % circumference_cells) * row_size + width_index
    return particle_start + local_id


def make_cross_section_bending_stiff(
    builder: newton.ModelBuilder,
    *,
    particle_start: int,
    edge_start: int,
    edge_end: int,
    circumference_cells: int,
    width_cells: int,
    rib_edge_ke: float,
    rib_edge_kd: float,
) -> int:
    """Assign high bending stiffness only to cross-sectional rib hinges."""
    row_size = width_cells + 1
    belt_particle_end = particle_start + circumference_cells * row_size
    stiffened_count = 0

    def decode(global_particle: int) -> tuple[int, int]:
        local_particle = global_particle - particle_start
        return divmod(local_particle, row_size)

    for edge_id in range(edge_start, edge_end):
        opposite_0, opposite_1, hinge_0, hinge_1 = builder.edge_indices[edge_id]

        if opposite_0 == -1 or opposite_1 == -1:
            continue

        if not (
            particle_start <= hinge_0 < belt_particle_end
            and particle_start <= hinge_1 < belt_particle_end
        ):
            continue

        circ_0, width_0 = decode(hinge_0)
        circ_1, width_1 = decode(hinge_1)

        circumferential_delta = (circ_1 - circ_0) % circumference_cells
        are_circumferential_neighbours = circumferential_delta in (
            1,
            circumference_cells - 1,
        )

        is_rib_bending_hinge = width_0 == width_1 and are_circumferential_neighbours

        if is_rib_bending_hinge:
            builder.edge_bending_properties[edge_id] = (
                float(rib_edge_ke),
                float(rib_edge_kd),
            )
            stiffened_count += 1

    return stiffened_count


def add_cross_section_rib_springs(
    builder: newton.ModelBuilder,
    *,
    particle_start: int,
    circumference_cells: int,
    width_cells: int,
    spring_ke: float,
    spring_kd: float,
) -> int:
    """Add a stiff distance network separately to every cross-sectional rib."""
    spring_count_before = builder.spring_count

    for i in range(circumference_cells):
        bottom = belt_particle_id(
            particle_start, i, 0, circumference_cells, width_cells
        )
        top = belt_particle_id(
            particle_start, i, width_cells, circumference_cells, width_cells
        )

        builder.add_spring(bottom, top, spring_ke, spring_kd, 0.0)

        for j in range(1, width_cells):
            particle = belt_particle_id(
                particle_start, i, j, circumference_cells, width_cells
            )
            builder.add_spring(bottom, particle, spring_ke, spring_kd, 0.0)
            builder.add_spring(particle, top, spring_ke, spring_kd, 0.0)

    return builder.spring_count - spring_count_before


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0

        self.fps = SIM_FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.iterations = args.iterations
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(gravity=args.gravity)
        builder.add_ground_plane()

        bottom_z = PARTICLE_RADIUS + GROUND_EPSILON

        vertices, indices = build_elliptical_ring_mesh(
            major_diameter=BELT_MAJOR_DIAMETER,
            minor_diameter=BELT_MINOR_DIAMETER,
            strip_width=args.strip_width,
            circumference_cells=args.circumference_cells,
            width_cells=args.width_cells,
            bottom_z=bottom_z,
        )

        belt_particle_start = builder.particle_count
        belt_edge_start = builder.edge_count

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=CLOTH_DENSITY,
            tri_ke=args.tri_ke,
            tri_ka=args.tri_ka,
            tri_kd=args.tri_kd,
            edge_ke=args.base_edge_ke,
            edge_kd=args.base_edge_kd,
            add_springs=False,
            particle_radius=PARTICLE_RADIUS,
        )

        belt_edge_end = builder.edge_count

        stiff_hinge_count = make_cross_section_bending_stiff(
            builder,
            particle_start=belt_particle_start,
            edge_start=belt_edge_start,
            edge_end=belt_edge_end,
            circumference_cells=args.circumference_cells,
            width_cells=args.width_cells,
            rib_edge_ke=args.rib_bending_ke,
            rib_edge_kd=args.rib_bending_kd,
        )

        rib_spring_count = 0
        if not args.disable_rib_springs:
            rib_spring_count = add_cross_section_rib_springs(
                builder,
                particle_start=belt_particle_start,
                circumference_cells=args.circumference_cells,
                width_cells=args.width_cells,
                spring_ke=args.rib_spring_ke,
                spring_kd=args.rib_spring_kd,
            )

        print(
            "Belt rib reinforcement: "
            f"{stiff_hinge_count} high-stiffness bending hinges, "
            f"{rib_spring_count} rib springs."
        )

        builder.color(include_bending=True)
        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 1.0

        # Twist that keeps the ring: pin the centerline, drive the band edges.
        row_size = args.width_cells + 1
        C = args.circumference_cells
        j_center = args.width_cells // 2 # middle row = ring centerline
        a = 0.5 * BELT_MAJOR_DIAMETER
        b = 0.5 * BELT_MINOR_DIAMETER
        z_center = bottom_z + 0.5 * args.strip_width
        k = args.twist_wavenumber

        # (1) Centerline particles: pinned in place to hold the ring footprint.
        centerline_indices: list[int] = [
            belt_particle_start + i * row_size + j_center for i in range(C)
        ]

        # (2) Top/bottom edge particles: driven to rotate about the local
        #     tangent through the centerline point => a pure twist of the band.
        driven_indices: list[int] = []
        driven_axes: list[list[float]] = []
        driven_centers: list[list[float]] = []
        driven_mults: list[float] = []
        for i in range(C):
            theta = 2.0 * math.pi * i / C
            # Ring centerline point and local tangent direction.
            cx, cy = a * math.cos(theta), b * math.sin(theta)
            tx, ty = -a * math.sin(theta), b * math.cos(theta)
            tn = math.hypot(tx, ty)
            tx, ty = tx / tn, ty / tn
            mult = math.sin(k * theta)  # seam-consistent twist field
            for j in (0, args.width_cells):
                driven_indices.append(belt_particle_start + i * row_size + j)
                driven_axes.append([tx, ty, 0.0])
                driven_centers.append([cx, cy, z_center])
                driven_mults.append(mult)

        # Deactivate all prescribed particles (centerline pins + driven edges).
        flags = self.model.particle_flags.numpy()
        for idx in centerline_indices:
            flags[idx] = flags[idx] & ~int(ParticleFlags.ACTIVE)
        for idx in driven_indices:
            flags[idx] = flags[idx] & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags)

        self.twist_max_angle = args.twist_max_angle
        self.twist_ramp_time = args.twist_ramp_time

        print(
            f"Twist drive: {len(centerline_indices)} pinned centerline points, "
            f"{len(driven_indices)} driven edge points, "
            f"k={k}, max_angle={math.degrees(self.twist_max_angle):.1f} deg, "
            f"ramp={self.twist_ramp_time:.1f}s."
        )

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

        # Twist bookkeeping arrays.
        n_drv = len(driven_indices)
        self.driven_indices = wp.array(driven_indices, dtype=int)
        self.driven_axes = wp.array(driven_axes, dtype=wp.vec3)
        self.driven_centers = wp.array(driven_centers, dtype=wp.vec3)
        self.driven_mults = wp.array(driven_mults, dtype=float)
        self.roots = wp.zeros(n_drv, dtype=wp.vec3)
        self.roots_to_ps = wp.zeros(n_drv, dtype=wp.vec3)
        self.t = wp.zeros((1,), dtype=float)

        wp.launch(
            kernel=init_twist,
            dim=n_drv,
            inputs=[
                self.driven_indices,
                self.state_0.particle_q,
                self.driven_centers,
                self.driven_axes,
                self.t,
            ],
            outputs=[self.roots, self.roots_to_ps],
        )

        self.viewer.set_model(self.model)

        picking = getattr(self.viewer, "picking", None)
        if picking is not None:
            if hasattr(picking, "pick_stiffness"):
                picking.pick_stiffness = 0.0
            if hasattr(picking, "pick_damping"):
                picking.pick_damping = 0.0
            if hasattr(picking, "pick_state") and picking.pick_state is not None:
                pick_state_np = picking.pick_state.numpy()
                pick_state_np[0]["pick_stiffness"] = 0.0
                pick_state_np[0]["pick_damping"] = 0.0
                pick_state_np[0]["pick_max_acceleration"] = 0.0
                picking.pick_state.assign(pick_state_np)

        self.camera_set = False
        self.graph = None

    def simulate(self):
        if hasattr(self.solver, "rebuild_bvh"):
            self.solver.rebuild_bvh(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                kernel=apply_twist,
                dim=self.driven_indices.shape[0],
                inputs=[
                    self.driven_indices,
                    self.driven_axes,
                    self.roots,
                    self.roots_to_ps,
                    self.driven_mults,
                    self.t,
                    self.twist_max_angle,
                    self.twist_ramp_time,
                    self.sim_dt,
                ],
                outputs=[
                    self.state_0.particle_q,
                    self.state_1.particle_q,
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
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if not self.camera_set and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -0.40, 0.25), -28.0, 90.0)
            self.camera_set = True

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # Centerline is pinned, so the ring must stay near its rest footprint.
        newton.examples.test_particle_state(
            self.state_0,
            "belt particles stay within a bounded region (ring preserved)",
            lambda q, qd: (-0.3 < q[0] < 0.3)
            and (-0.3 < q[1] < 0.3)
            and (-0.05 < q[2] < 0.2),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "belt particle velocities remain bounded",
            lambda q, qd: max(abs(qd[0]), abs(qd[1]), abs(qd[2])) < 5.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)

        parser.add_argument("--gravity", type=float, default=0.0)
        parser.add_argument("--strip-width", type=float, default=BELT_STRIP_WIDTH)
        parser.add_argument(
            "--circumference-cells", type=int, default=CIRCUMFERENCE_CELLS
        )
        parser.add_argument("--width-cells", type=int, default=WIDTH_CELLS)
        parser.add_argument("--substeps", type=int, default=SIM_SUBSTEPS)
        parser.add_argument("--iterations", type=int, default=VBD_ITERATIONS)

        parser.add_argument("--tri-ke", type=float, default=TRI_KE)
        parser.add_argument("--tri-ka", type=float, default=TRI_KA)
        parser.add_argument("--tri-kd", type=float, default=TRI_KD)

        parser.add_argument("--base-edge-ke", type=float, default=BASE_EDGE_KE)
        parser.add_argument("--base-edge-kd", type=float, default=BASE_EDGE_KD)
        parser.add_argument("--rib-bending-ke", type=float, default=RIB_BENDING_KE)
        parser.add_argument("--rib-bending-kd", type=float, default=RIB_BENDING_KD)
        parser.add_argument("--rib-spring-ke", type=float, default=RIB_SPRING_KE)
        parser.add_argument("--rib-spring-kd", type=float, default=RIB_SPRING_KD)
        parser.add_argument("--disable-rib-springs", action="store_true")

        # Twist controls
        parser.add_argument(
            "--twist-wavenumber",
            type=int,
            default=TWIST_WAVENUMBER,
            help="Number of twist lobes around the loop (integer keeps the "
            "seam consistent).",
        )
        parser.add_argument(
            "--twist-max-angle",
            type=lambda d: math.radians(float(d)),
            default=TWIST_MAX_ANGLE,
            help="Peak twist angle of the band, in DEGREES.",
        )
        parser.add_argument(
            "--twist-ramp-time",
            type=float,
            default=TWIST_RAMP_TIME,
            help="Seconds to ramp the twist from 0 to the peak, then hold.",
        )

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)