import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def create_ellipse_cable_geometry(
        self, pos: wp.vec3 | None = None, num_elements=32, twisting_angle=0.0
    ):
        """Create an elliptical closed-loop cable route with parallel-transported quaternions.

        Generates a closed elliptical path lying on the XY-plane.
        Dimensions: 248 mm x 168 mm (radii: 124 mm x 84 mm)
        """
        if pos is None:
            pos = wp.vec3()

        if num_elements <= 0:
            raise ValueError("num_elements must be positive")

        num_points = num_elements + 1
        points = []

        # Exact dimensions converted to meters
        a = 0.248 / 2.0  # 0.124 m (124 mm)
        b = 0.168 / 2.0  # 0.084 m (84 mm)

        for i in range(num_points):
            theta = 2.0 * np.pi * i / num_elements
            x = a * np.cos(theta)
            y = b * np.sin(theta)
            z = 0.0
            points.append(pos + wp.vec3(x, y, z))

        edge_q = newton.utils.create_parallel_transport_cable_quaternions(
            points,
            twist_total=float(twisting_angle),
        )
        return points, edge_q

    def configure_safe_picking(
        self,
        pick_stiffness=1.2,
        pick_damping=1.0,
        pick_max_acceleration=0.25,
    ):
        """Make mouse interaction stable for a tiny flexible cable.
        """
        # Make all cable bodies receive only linear picking force, no picking torque.
        if hasattr(self.viewer, "set_picking_linear_only_bodies"):
            self.viewer.set_picking_linear_only_bodies(self.cable_body_ids)

        picking = getattr(self.viewer, "picking", None)
        if picking is not None:
            picking.pick_stiffness = float(pick_stiffness)
            picking.pick_damping = float(pick_damping)

            # Update the GPU/CPU picking state array used by Newton's picking kernel.
            if hasattr(picking, "pick_state") and picking.pick_state is not None:
                pick_state_np = picking.pick_state.numpy()
                pick_state_np[0]["pick_stiffness"] = float(pick_stiffness)
                pick_state_np[0]["pick_damping"] = float(pick_damping)
                pick_state_np[0]["pick_max_acceleration"] = float(pick_max_acceleration)
                picking.pick_state.assign(pick_state_np)

    def __init__(self, viewer, args):
        # Store viewer and arguments
        self.viewer = viewer
        self.args = args

        # Simulation cadence
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.update_step_interval = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable parameters
        self.num_elements = 48
        cable_radius = 0.0033  # 3.3 mm radius (6.6 mm diameter)

        a = 0.248 / 2.0
        b = 0.168 / 2.0
        h = ((a - b) ** 2) / ((a + b) ** 2)
        self.cable_length = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))

        stretch_stiffness = 5.0e3
        bend_stiffness = 1.5e-1

        # Create builder for the simulation
        builder = newton.ModelBuilder()

        # Set default material properties before adding any shapes
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.mu = 1.5

        self.cable_bodies_list = []
        self.cable_body_ids = []

        # Position loop center at the origin resting flat on the ground plane (z = cable_radius)
        start_pos = wp.vec3(0.0, 0.0, cable_radius)

        cable_points, cable_edge_q = self.create_ellipse_cable_geometry(
            pos=start_pos,
            num_elements=self.num_elements,
            twisting_angle=0.0,
        )

        rod_bodies, _rod_joints = builder.add_rod(
            positions=cable_points,
            quaternions=cable_edge_q,
            radius=cable_radius,
            stretch_stiffness=stretch_stiffness,
            bend_stiffness=bend_stiffness,
            bend_damping=1.0e-1,
            closed=True,
            label="flexible_ellipse_cable",
        )

        self.cable_bodies_list.append(rod_bodies)
        self.cable_body_ids = list(rod_bodies)

        # Add ground plane
        builder.add_ground_plane()

        # Color particles and rigid bodies for VBD solver
        builder.color()

        # Finalize model
        self.model = builder.finalize()

        # Use full hard-contact correction (contact alpha 0.0) for stronger repulsion with low iterations.
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.sim_iterations,
            rigid_avbd_contact_alpha=0.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

        self.configure_safe_picking(
            pick_stiffness=1.2,
            pick_damping=1.0,
            pick_max_acceleration=0.25,
        )

        # Camera will be forced once inside render().
        self.camera_set = False
        self.capture()

    def capture(self):
        """Capture simulation loop into a CUDA graph for optimal GPU performance."""
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Execute all simulation substeps for one frame."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # User mouse interaction is enabled.
            # Safe picking settings above prevent the close-view drag from exploding.
            self.viewer.apply_forces(self.state_0)

            # Collision detection and contact refresh cadence.
            refresh_contacts = (substep % self.update_step_interval) == 0
            if refresh_contacts:
                self.model.collide(self.state_0, self.contacts)

            self.solver.set_rigid_history_update(refresh_contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance simulation by one frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current simulation state to the viewer."""

        # Force close-up camera once after viewer starts rendering.
        if not self.camera_set and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                wp.vec3(0.0, -0.70, 0.40),
                -28.0,
                90.0,
            )
            self.camera_set = True

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Test cable simulation for stability and correctness."""
        segment_length = self.cable_length / self.num_elements

        if self.state_0.body_q is not None and self.state_0.body_qd is not None:
            body_positions = self.state_0.body_q.numpy()
            body_velocities = self.state_0.body_qd.numpy()

            # Test 1: Check for numerical stability
            assert np.isfinite(body_positions).all(), "Non-finite values in body positions"
            assert np.isfinite(body_velocities).all(), "Non-finite values in body velocities"

            # Test 2: Check cable connectivity
            for cable_idx, cable_bodies in enumerate(self.cable_bodies_list):
                num_bodies = len(cable_bodies)
                for segment in range(num_bodies):
                    body1_idx = cable_bodies[segment]
                    body2_idx = cable_bodies[(segment + 1) % num_bodies]

                    pos1 = body_positions[body1_idx][:3]
                    pos2 = body_positions[body2_idx][:3]
                    distance = np.linalg.norm(pos2 - pos1)

                    expected_distance = segment_length
                    joint_tolerance = expected_distance * 0.25
                    assert distance < expected_distance + joint_tolerance, (
                        f"Cable {cable_idx} segments connection too far apart: {distance:.3f}"
                    )

            # Test 3: Check ground boundaries
            ground_tolerance = 0.1
            min_z = np.min(body_positions[:, 2])
            assert min_z > -ground_tolerance, f"Cable penetrated ground too much: min_z = {min_z:.3f}"


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    newton.examples.run(Example(viewer, args), args)