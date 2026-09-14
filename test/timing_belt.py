from __future__ import annotations

import math

import warp as wp

import newton
import newton.examples

# Belt geometry (metres, kilograms, seconds)
BELT_MAJOR_DIAMETER = 0.248
BELT_MINOR_DIAMETER = 0.168
BELT_STRIP_WIDTH = 0.050

# Mesh resolution.
CIRCUMFERENCE_CELLS = 64
WIDTH_CELLS = 4

PARTICLE_RADIUS = 0.0010
GROUND_EPSILON = 0.0001

# Total belt mass.
TARGET_BELT_MASS = 0.033 # kg (33 g)

# Ordinary membrane stiffness.
TRI_KE = 1.0e4
TRI_KA = 1.0e4
TRI_KD = 2.0e2

# Ordinary cloth bending stiffness.
BASE_EDGE_KE = 5.0e1
BASE_EDGE_KD = 5.0

RIB_BENDING_KE = 2.0e4
RIB_BENDING_KD = 2.0e1

# Additional distance springs inside every cross-sectional line.
RIB_SPRING_KE = 2.0e4
RIB_SPRING_KD = 2.0e1

# Circumferential reinforcement. 
LOOP_SPRING_KE = 1.0e4
LOOP_SPRING_KD = 5.0e1

SIM_FPS = 60
SIM_SUBSTEPS = 20
VBD_ITERATIONS = 20


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

    Particle layout:

        local particle id = i * (width_cells + 1) + j

    where:
        i = circumferential position around the closed belt
        j = position across the belt width

    For one fixed i, all particles j=0...width_cells form one cross-sectional
    rib. Initially, they have identical x/y and vary only in z.
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

    # Two triangles per quadrilateral cell.
    for i in range(circumference_cells):
        i_next = (i + 1) % circumference_cells

        for j in range(width_cells):
            v00 = vertex_id(i, j)
            v10 = vertex_id(i_next, j)
            v11 = vertex_id(i_next, j + 1)
            v01 = vertex_id(i, j + 1)

            if (i + j) % 2 == 0:
                indices.extend((v00, v10, v01))
                indices.extend((v10, v11, v01))
            else:
                indices.extend((v00, v10, v11))
                indices.extend((v00, v11, v01))

    return vertices, indices


def compute_cloth_area(vertices: list[wp.vec3], indices: list[int]) -> float:
    """Sum the area of every triangle in the mesh.
    """
    total = 0.0
    for t in range(0, len(indices), 3):
        p0 = vertices[indices[t + 0]]
        p1 = vertices[indices[t + 1]]
        p2 = vertices[indices[t + 2]]

        ax, ay, az = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        bx, by, bz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]

        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx

        total += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)

    return total


# Rib-stiffening helpers
def belt_particle_id(
    particle_start: int,
    circumference_index: int,
    width_index: int,
    circumference_cells: int,
    width_cells: int,
) -> int:
    """Return the global Newton particle id of one belt mesh vertex."""
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
    """Assign high bending stiffness only to cross-sectional rib hinges.
    """
    row_size = width_cells + 1
    belt_particle_end = particle_start + circumference_cells * row_size
    stiffened_count = 0

    def decode(global_particle: int) -> tuple[int, int]:
        local_particle = global_particle - particle_start
        return divmod(local_particle, row_size)

    for edge_id in range(edge_start, edge_end):
        opposite_0, opposite_1, hinge_0, hinge_1 = builder.edge_indices[edge_id]

        # Boundary edges have only one adjacent triangle and cannot provide a
        # proper dihedral bending constraint.
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

        is_rib_bending_hinge = (
            width_0 == width_1 and are_circumferential_neighbours
        )

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
    """Add a stiff distance network separately to every cross-sectional rib.
    """
    spring_count_before = builder.spring_count

    for i in range(circumference_cells):
        bottom = belt_particle_id(
            particle_start,
            i,
            0,
            circumference_cells,
            width_cells,
        )
        top = belt_particle_id(
            particle_start,
            i,
            width_cells,
            circumference_cells,
            width_cells,
        )

        # Preserve the complete physical belt width.
        builder.add_spring(bottom, top, spring_ke, spring_kd, 0.0)

        # Lock every interior point to its original fractional location between
        # the two endpoints of this rib.
        for j in range(1, width_cells):
            particle = belt_particle_id(
                particle_start,
                i,
                j,
                circumference_cells,
                width_cells,
            )
            builder.add_spring(bottom, particle, spring_ke, spring_kd, 0.0)
            builder.add_spring(particle, top, spring_ke, spring_kd, 0.0)

    return builder.spring_count - spring_count_before


def add_circumferential_springs(
    builder: newton.ModelBuilder,
    *,
    particle_start: int,
    circumference_cells: int,
    width_cells: int,
    spring_ke: float,
    spring_kd: float,
) -> int:
    """Preserve segment lengths around every horizontal row of the belt.
    """
    spring_count_before = builder.spring_count

    for j in range(width_cells + 1):
        for i in range(circumference_cells):
            p0 = belt_particle_id(
                particle_start, i, j, circumference_cells, width_cells
            )
            p1 = belt_particle_id(
                particle_start,
                (i + 1) % circumference_cells,
                j,
                circumference_cells,
                width_cells,
            )
            builder.add_spring(p0, p1, spring_ke, spring_kd, 0.0)

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

        vertices, indices = build_elliptical_ring_mesh(
            major_diameter=BELT_MAJOR_DIAMETER,
            minor_diameter=BELT_MINOR_DIAMETER,
            strip_width=args.strip_width,
            circumference_cells=args.circumference_cells,
            width_cells=args.width_cells,
            bottom_z=PARTICLE_RADIUS + GROUND_EPSILON,
        )

        # Derive density from the requested total mass and the true mesh area so
        # the belt weighs exactly args.target_mass regardless of resolution.
        belt_area = compute_cloth_area(vertices, indices)
        cloth_density = args.target_mass / belt_area

        # Record ranges so the code remains correct even if other particles or
        # cloth objects are added before this belt later.
        belt_particle_start = builder.particle_count
        belt_edge_start = builder.edge_count

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=cloth_density,
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

        loop_spring_count = add_circumferential_springs(
            builder,
            particle_start=belt_particle_start,
            circumference_cells=args.circumference_cells,
            width_cells=args.width_cells,
            spring_ke=LOOP_SPRING_KE,
            spring_kd=LOOP_SPRING_KD,
        )

        print(
            f"Belt mesh: {len(vertices)} particles, "
            f"area {belt_area * 1e4:.1f} cm^2, "
            f"density {cloth_density:.4f} kg/m^2, "
            f"target mass {args.target_mass * 1e3:.1f} g."
        )
        print(
            "Belt rib reinforcement: "
            f"{stiff_hinge_count} high-stiffness bending hinges, "
            f"{rib_spring_count} rib springs, "
            f"{loop_spring_count} circumferential springs."
        )

        builder.color(include_bending=True)
        self.model = builder.finalize()

        try:
            actual_mass = float(self.model.particle_mass.numpy().sum())
            print(f"Finalized belt mass: {actual_mass * 1e3:.2f} g.")
        except Exception:
            pass

        self.model.soft_contact_ke = 2.0e4
        self.model.soft_contact_kd = 2.0e2
        self.model.soft_contact_mu = 1.0

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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

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
            "belt particles remain above the ground",
            lambda q, qd: q[2] > 0.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        parser.add_argument(
            "--gravity",
            type=float,
            default=-9.81,
            help="Gravity along Z (negative = down). 0.0 makes the belt float.",
        )
        parser.add_argument(
            "--target-mass",
            type=float,
            default=TARGET_BELT_MASS,
            help="Total belt mass in kilograms (default 0.033 kg = 33 g).",
        )
        parser.add_argument(
            "--strip-width",
            type=float,
            default=BELT_STRIP_WIDTH,
            help="Width of the cloth belt strip in metres.",
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
        parser.add_argument(
            "--substeps",
            type=int,
            default=SIM_SUBSTEPS,
            help="Simulation substeps per rendered frame.",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=VBD_ITERATIONS,
            help="VBD iterations per substep.",
        )

        parser.add_argument("--tri-ke", type=float, default=TRI_KE)
        parser.add_argument("--tri-ka", type=float, default=TRI_KA)
        parser.add_argument("--tri-kd", type=float, default=TRI_KD)

        parser.add_argument(
            "--base-edge-ke",
            type=float,
            default=BASE_EDGE_KE,
            help="Bending stiffness for non-rib hinges.",
        )
        parser.add_argument(
            "--base-edge-kd",
            type=float,
            default=BASE_EDGE_KD,
            help="Bending damping for non-rib hinges.",
        )
        parser.add_argument(
            "--rib-bending-ke",
            type=float,
            default=RIB_BENDING_KE,
            help="High bending stiffness used across each belt cross-section.",
        )
        parser.add_argument(
            "--rib-bending-kd",
            type=float,
            default=RIB_BENDING_KD,
            help="Bending damping used across each belt cross-section.",
        )
        parser.add_argument(
            "--rib-spring-ke",
            type=float,
            default=RIB_SPRING_KE,
            help="Stiffness of the extra per-rib distance springs.",
        )
        parser.add_argument(
            "--rib-spring-kd",
            type=float,
            default=RIB_SPRING_KD,
            help="Damping of the extra per-rib distance springs.",
        )
        parser.add_argument(
            "--disable-rib-springs",
            action="store_true",
            help="Use only anisotropic bending and do not create rib springs.",
        )

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)