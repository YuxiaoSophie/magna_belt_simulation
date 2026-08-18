import argparse
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MJCF_PATH = PROJECT_DIR / "2f85.xml"


def as_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def quat_to_vec4(q):
    return wp.vec4(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def parse_indices(text):
    if text is None or text.strip() == "":
        return []
    return [int(v.strip()) for v in text.split(",") if v.strip() != ""]


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def find_body_index(model, name_hint):
    labels = getattr(model, "body_label", [])

    if name_hint:
        name_hint = name_hint.lower()
        for i, label in enumerate(labels):
            if name_hint in label.lower():
                return i

    for key in ["ee_link", "tool0", "wrist_3_link"]:
        for i, label in enumerate(labels):
            if key in label.lower():
                return i

    return model.body_count - 1


def find_gripper_driver_coords(model, arm_dofs):
    """
    Find only the real Robotiq driver joint coordinate(s).
    """
    joint_labels = getattr(model, "joint_label", [])
    q_start = as_numpy(model.joint_q_start)
    dof_dim = as_numpy(model.joint_dof_dim)

    driver_coords = []

    for j, name in enumerate(joint_labels):
        if j < arm_dofs:
            continue

        lname = name.lower()
        dofs = int(dof_dim[j, 0] + dof_dim[j, 1])

        if dofs <= 0:
            continue

        # Only select true driver joints.
        if "driver_joint" in lname:
            driver_coords.append(int(q_start[j]))

    return sorted(set(driver_coords))


def print_model_tree(model):
    print("\n========== BODY LABELS ==========")
    for i, name in enumerate(getattr(model, "body_label", [])):
        print(f"body {i:3d}: {name}")

    print("\n========== JOINT LABELS / COORD INDICES ==========")
    joint_labels = getattr(model, "joint_label", [])
    q_start = as_numpy(model.joint_q_start)
    dof_dim = as_numpy(model.joint_dof_dim)

    for j, name in enumerate(joint_labels):
        dofs = int(dof_dim[j, 0] + dof_dim[j, 1])
        q0 = int(q_start[j])
        q_indices = list(range(q0, q0 + dofs))
        print(f"joint {j:3d}: q={q_indices}  name={name}")

    print("=================================\n")


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True

        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.arm_dofs = 6

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        height = 1.2
        base_tf = wp.transform(
            wp.vec3(0.0, 0.0, height),
            wp.quat_identity(),
        )

        # Load robot / gripper
        if args.asset_file:
            if args.asset_type == "usd":
                builder.add_usd(
                    args.asset_file,
                    xform=base_tf,
                    collapse_fixed_joints=False,
                    enable_self_collisions=False,
                    hide_collision_shapes=False,
                )
            elif args.asset_type == "urdf":
                builder.add_urdf(
                    args.asset_file,
                    xform=base_tf,
                    floating=False,
                    enable_self_collisions=False,
                    parse_visuals_as_colliders=False,
                )
            else:
                raise ValueError(f"Unknown asset type: {args.asset_type}")

        else:
            asset_path = newton.utils.download_asset("universal_robots_ur10")
            asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")

            builder.add_usd(
                asset_file,
                xform=base_tf,
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                hide_collision_shapes=True,
            )

            tool_body_idx = 7
            print(f"[INFO] Attaching 2f85.xml centered to body {tool_body_idx}.")
            gripper_pos = wp.vec3(0.0, 0.0, 0.0)
            gripper_rot = wp.quat_from_axis_angle(
                wp.vec3(0.0, 1.0, 0.0),
                np.pi / 2.0,
            )
            gripper_offset_tf = wp.transform(gripper_pos, gripper_rot)
            builder.add_mjcf(
                str(MJCF_PATH),
                parent_body=tool_body_idx,
                xform=gripper_offset_tf,
            )

        # Add simple support column and ground
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, height / 2.0)),
            half_height=height / 2.0,
            radius=0.08,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()

        if args.print_model:
            print_model_tree(self.model)

        self.n_coords = self.model.joint_coord_count

        # Find gripper driver coordinate(s)
        gripper_args = args.gripper_indices

        if not gripper_args and not args.asset_file:
            driver_coords = find_gripper_driver_coords(self.model, self.arm_dofs)

            if driver_coords:
                gripper_args = ",".join(map(str, driver_coords))

        self.gripper_coord_indices = parse_indices(gripper_args)

        print(
            "[INFO] Actively driving ONLY gripper driver coordinate indices: "
            f"{self.gripper_coord_indices}"
        )

        if len(self.gripper_coord_indices) == 0:
            print(
                "[WARNING] No gripper driver joints were found. "
                "Run with --print-model and pass the real driver q-index using "
                "--gripper-indices."
            )

        # Compute open / closed values from limits
        lower_np = as_numpy(self.model.joint_limit_lower)
        upper_np = as_numpy(self.model.joint_limit_upper)

        self.gripper_open_values = []
        self.gripper_closed_values = []

        limit_margin = 0.005

        for q_idx in self.gripper_coord_indices:
            lo = float(lower_np[q_idx])
            hi = float(upper_np[q_idx])

            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                open_q = lo + limit_margin
                closed_q = hi - limit_margin
            else:
                open_q = 0.0
                closed_q = float(args.gripper_closed)

            self.gripper_open_values.append(open_q)
            self.gripper_closed_values.append(closed_q)

        print(f"[INFO] Gripper open targets:   {self.gripper_open_values}")
        print(f"[INFO] Gripper closed targets: {self.gripper_closed_values}")

        # Static initial pose: arm pose + open gripper
        initial_joints = np.zeros(self.n_coords, dtype=np.float32)

        # Arm pose
        initial_joints[0] = 0.0
        initial_joints[1] = -1.35
        initial_joints[2] = 1.75
        initial_joints[3] = -1.95
        initial_joints[4] = -1.57
        initial_joints[5] = 0.0

        # Start with gripper fully open
        for q_idx, open_q in zip(
            self.gripper_coord_indices,
            self.gripper_open_values,
        ):
            initial_joints[q_idx] = open_q

        self.base_targets_np = initial_joints.copy()

        self.model.joint_q.assign(initial_joints)
        self.model.joint_qd.zero_()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Position controller gains
        ke_np = self.model.joint_target_ke.numpy()
        kd_np = self.model.joint_target_kd.numpy()
        mode_np = self.model.joint_target_mode.numpy()

        for i in range(len(ke_np)):
            ke_np[i] = 0.0
            kd_np[i] = 0.0
            mode_np[i] = int(JointTargetMode.NONE)

        # Hold arm joints fixed
        for i in range(self.arm_dofs):
            ke_np[i] = 1000.0
            kd_np[i] = 100.0
            mode_np[i] = int(JointTargetMode.POSITION)

        # Drive only the gripper driver coordinate(s)
        for q_idx in self.gripper_coord_indices:
            ke_np[q_idx] = 200.0
            kd_np[q_idx] = 15.0
            mode_np[q_idx] = int(JointTargetMode.POSITION)

        self.model.joint_target_ke.assign(ke_np)
        self.model.joint_target_kd.assign(kd_np)
        self.model.joint_target_mode.assign(mode_np)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=False,
        )

        self.viewer.set_model(self.model)

        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0,
        )

        self.joint_target_q_view = self.control.joint_target_q.reshape(
            (1, self.n_coords)
        )

        # Initial control target equals initial pose
        wp.copy(self.control.joint_target_q, self.model.joint_q)

        self.ee_index = find_body_index(self.model, args.ee_body)

    def solve_static_targets(self):
        targets = self.base_targets_np.copy()

        cycle_period = 2.0
        t = (self.sim_time % cycle_period) / cycle_period

        if t < 0.5:
            # Open to closed
            alpha = smoothstep(t / 0.5)
        else:
            # Closed back to open
            alpha = smoothstep(1.0 - ((t - 0.5) / 0.5))

        for q_idx, open_q, closed_q in zip(
            self.gripper_coord_indices,
            self.gripper_open_values,
            self.gripper_closed_values,
        ):
            targets[q_idx] = open_q + alpha * (closed_q - open_q)

        self.joint_target_q_view.assign(targets.reshape(1, -1))

    def simulate(self):
        self.solve_static_targets()
        self.model.collide(self.state_0, self.contacts)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

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
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        parser.add_argument("--asset-file", type=str, default="")
        parser.add_argument(
            "--asset-type",
            type=str,
            default="usd",
            choices=["usd", "urdf"],
        )
        parser.add_argument("--ee-body", type=str, default="ee_link")

        # Leave blank for auto-detection.
        # Or manually pass only the real driver coordinate index, for example:
        # --gripper-indices 6
        # or if the MJCF has two real driver coordinates:
        # --gripper-indices 6,10
        parser.add_argument("--gripper-indices", type=str, default="")
        parser.add_argument("--gripper-closed", type=float, default=0.8)
        parser.add_argument("--print-model", action="store_true")

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)