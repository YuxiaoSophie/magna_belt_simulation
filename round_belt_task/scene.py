"""Assembly of the Drake round-belt scene into a ``newton.ModelBuilder``.

:func:`build_scene` adds the statics (table, task board, belt-chain holder), both arms
(Franka + hand, USD UR10 + 2f85 gripper), the belt rod and the ground, and returns the
:class:`SceneInfo` index bookkeeping the solver partition needs.
No viewer and no solver here, and nothing touches the finalized ``Model``: seeding the
default joint state is ``joint_state.py``, running the thing is ``simulation.py``.
Every number lives in ``constants.py``; task-agnostic Newton plumbing lives in ``utils/``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
import warp as wp
from loguru import logger

import newton
import newton.utils
from newton.solvers import SolverMuJoCo, SolverVBD

import round_belt
from round_belt_task.constants import (
    ALOHA_FINGER_COLOR, ALOHA_FINGER_OFFSET_X, ALOHA_FINGER_OFFSET_Z, BELT_CENTER,
    BELT_COLOR, BELT_NUM_ELEMENTS, BELT_RADIUS, BELT_SEMI_AXIS_X, BELT_SEMI_AXIS_Y,
    BOARD_COLOR, BOARD_PANEL_MIN_SPAN, BOARD_URDF, HOLDER_URDF, LARGE_PULLEY_COLOR,
    MJCF_BASE_MOUNT_OFFSET_Z, PANDA_ARM_URDF, PANDA_HAND_URDF, PULLEY_MOUNT_COLOR,
    ROBOTIQ_FINGER_DIR, ROBOTIQ_MJCF, SMALL_PULLEY_COLOR, SMALL_PULLEY_MOUNT_LOCAL_XY,
    SMALL_PULLEY_MOUNT_RADIUS, TABLE_TOP_Z, TABLE_URDF, TABLE_VISUAL_LABEL,
    TABLETOP_COLLISION_THICKNESS, UR10_USD_ASSET, UR10_USD_RELPATH, UR10_WRIST3_LABEL,
    X_LINK8_HAND, X_USDWRIST3_GRIPPER, X_W_BOARD, X_W_HOLDER, X_W_PANDA, X_W_TABLE,
    X_W_UR10,
)

from utils.labels import body_index, hide_shapes, label_shapes_by_body
from utils.meshes import (
    fix_inverted_mesh_winding, load_meshes, neutralize_textured_shape_colors,
)
from utils.urdf import Color, add_urdf_as_static_shapes


class JointConfig(TypedDict, total=False):
    """Joint bookkeeping produced by ``joint_state.apply_default_joint_state``; empty
    until it
    runs, hence ``total=False``.  ``gains_layout`` records which space Newton's gain
    arrays turned out to be indexed in, and the ``*_target_indices`` are the coord- or
    dof-index lists already chosen to match it."""

    gains_layout: str
    arm_labels: list[str]
    arm_defaults: list[float]
    arm_coord_indices: list[int]
    finger_coord_indices: list[int]
    gripper_open_values: list[float]
    arm_target_indices: list[int]
    finger_target_indices: list[int]
    gripper_target_indices: list[int]


@dataclass
class SceneInfo:
    """Builder-index bookkeeping produced by :func:`build_scene`."""

    robot_bodies: list[int]
    robot_joints: list[int]
    robot_shapes: list[int]
    franka_bodies: list[int]
    ur10_bodies: list[int]
    gripper_bodies: list[int]
    gripper_pad_bodies: list[int]
    gripper_pad_shapes: list[int]
    belt_bodies: list[int]
    belt_joints: list[int]
    belt_shapes: list[int]
    static_shapes: list[int]
    tabletop_collision_shape: int
    ground_height: float
    table_visual_aabb: tuple[np.ndarray, np.ndarray]
    static_shape_labels: dict[str, int]
    # Extras (not part of the published interface, safe to ignore).
    franka_joints: list[int] = field(default_factory=list)
    ur10_joints: list[int] = field(default_factory=list)
    gripper_joints: list[int] = field(default_factory=list)
    ground_shape: int = -1
    joint_config: JointConfig = field(default_factory=JointConfig)


def make_builder() -> newton.ModelBuilder:
    """Builder configured exactly like ``round_belt.py`` (Z up, -9.81, cable material)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.rigid_gap = 0.001
    SolverMuJoCo.register_custom_attributes(builder)
    try:
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
    except TypeError:
        SolverVBD.register_custom_attributes(builder)

    builder.default_shape_cfg.ke = round_belt.CABLE_CONTACT_KE
    builder.default_shape_cfg.kd = round_belt.CABLE_CONTACT_KD
    builder.default_shape_cfg.mu = round_belt.CABLE_CONTACT_MU
    return builder


def _small_pulley_mount_color(mesh: newton.Mesh) -> Color | None:
    """Black for the small pulley's mounting plate and bolts, ``None`` otherwise.

    Called per split component of the board mesh.  The board panel is excluded by span so
    it can never be recoloured wholesale; what is left within ``SMALL_PULLEY_MOUNT_RADIUS``
    of the mount is the plate and its bolts.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        return None
    lower, upper = verts.min(axis=0), verts.max(axis=0)
    if max(upper[0] - lower[0], upper[1] - lower[1]) >= BOARD_PANEL_MIN_SPAN:
        return None
    centre = 0.5 * (lower + upper)
    dx = centre[0] - SMALL_PULLEY_MOUNT_LOCAL_XY[0]
    dy = centre[1] - SMALL_PULLEY_MOUNT_LOCAL_XY[1]
    if (dx * dx + dy * dy) ** 0.5 > SMALL_PULLEY_MOUNT_RADIUS:
        return None
    return PULLEY_MOUNT_COLOR


def _add_aloha_fingers(
    builder: newton.ModelBuilder, gripper_root_body: int, pad_bodies: list[int]
) -> list[int]:
    """Attach the ALOHA-style Robotiq finger meshes to the 2f85's pad bodies.

    The fingers are placed at the Drake SDF's ``left_finger``/``right_finger`` offsets
    (``ALOHA_FINGER_OFFSET_X/Z``, the right one yawed by pi) in the Drake gripper base
    frame -- which is where the MJCF is welded, less the MJCF root's own
    ``MJCF_BASE_MOUNT_OFFSET_Z``.  Each finger rides whichever pad body it is nearest so
    it tracks the gripper as the pads close; the 2f85 linkage rotates its pads where
    Drake's fingers translate, so the match is exact only at the configuration used here.
    That is the point: these are the visible, contacting geometry, not a re-articulated
    gripper.  Returns the collider shapes, i.e. what GRIPPER_CONTACT_MU/KE/KD applies to.
    """
    identity = wp.quat_identity()
    X_W_root = wp.transform(*builder.body_q[gripper_root_body])
    # Undo the MJCF's own base_mount offset to land on the Drake gripper base frame.
    X_W_base = X_W_root * wp.transform(wp.vec3(0.0, 0.0, -MJCF_BASE_MOUNT_OFFSET_Z), identity)

    yaw_pi = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi)
    placements = [
        ("left_finger", +ALOHA_FINGER_OFFSET_X, identity),
        ("right_finger", -ALOHA_FINGER_OFFSET_X, yaw_pi),
    ]

    shapes: list[int] = []
    remaining = list(pad_bodies)
    for mesh_name, offset_x, rotation in placements:
        X_W_finger = X_W_base * wp.transform(
            wp.vec3(float(offset_x), 0.0, ALOHA_FINGER_OFFSET_Z), rotation
        )
        finger_position = np.array([X_W_finger.p[0], X_W_finger.p[1], X_W_finger.p[2]])
        # Pair each finger with the pad it actually sits on rather than assuming an order.
        pad = min(
            remaining,
            key=lambda body: float(
                np.linalg.norm(np.array(builder.body_q[body][:3]) - finger_position)
            ),
        )
        remaining.remove(pad)
        X_pad_finger = wp.transform_inverse(wp.transform(*builder.body_q[pad])) * X_W_finger
        mesh_path = ROBOTIQ_FINGER_DIR / f"{mesh_name}.obj"
        mesh = load_meshes(mesh_path, per_material=False)[0]
        # The Drake SDF declares a <visual> and a <collision> from the same mesh, so mirror
        # that: one visible non-colliding shape plus one hidden collider.  One shape
        # carrying both flags is physically equivalent but collapses into a single viewer
        # batch, leaving no collision entry for --show-collision to toggle.
        builder.add_shape_mesh(
            body=pad, xform=X_pad_finger, mesh=mesh, cfg=round_belt.make_visual_cfg(),
            color=wp.vec3(*ALOHA_FINGER_COLOR),
            label=f"robotiq_2f85/aloha_finger/{mesh_name}/visual",
        )
        shapes.append(
            builder.add_shape_mesh(
                body=pad, xform=X_pad_finger, mesh=mesh,
                cfg=round_belt.make_robust_table_collision_cfg(visible=False),
                color=wp.vec3(*ALOHA_FINGER_COLOR),
                label=f"robotiq_2f85/aloha_finger/{mesh_name}/collision",
            )
        )
    return shapes


def _add_belt(builder: newton.ModelBuilder) -> tuple[list[int], list[int], list[int]]:
    """Add the belt -- a 48-element closed VBD rod on the Drake round_belt.sdf ellipse --
    and return its (bodies, joints, shapes)."""
    body_start = builder.body_count
    joint_start = builder.joint_count
    shape_start = builder.shape_count

    center = np.asarray(BELT_CENTER, dtype=np.float64)
    points = []
    for i in range(BELT_NUM_ELEMENTS + 1):
        theta = 2.0 * np.pi * i / BELT_NUM_ELEMENTS
        points.append(wp.vec3(
            float(center[0] + BELT_SEMI_AXIS_X * np.cos(theta)),
            float(center[1] + BELT_SEMI_AXIS_Y * np.sin(theta)),
            float(center[2]),
        ))
    edge_q = newton.utils.create_parallel_transport_cable_quaternions(points, twist_total=0.0)

    bodies, _joints = builder.add_rod(
        positions=points, quaternions=edge_q, radius=BELT_RADIUS,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=round_belt._estimate_belt_density(), ke=round_belt.CABLE_CONTACT_KE,
            kd=round_belt.CABLE_CONTACT_KD, mu=round_belt.CABLE_CONTACT_MU,
            margin=0.0, gap=0.001),
        stretch_stiffness=2.0e4, stretch_damping=1.0e-1,
        bend_stiffness=1.5e-1, bend_damping=1.0e-1,
        closed=True, body_frame_origin="com",
        label="flexible_ellipse_cable", color=BELT_COLOR,
    )
    assert list(bodies) == list(range(body_start, builder.body_count))
    return (
        list(bodies),
        list(range(joint_start, builder.joint_count)),
        list(range(shape_start, builder.shape_count)),
    )


def _add_tabletop_collision(
    builder: newton.ModelBuilder,
    table_aabb: tuple[np.ndarray, np.ndarray],
    cfg: newton.ModelBuilder.ShapeConfig,
) -> int:
    """Add the safety floor under the belt; NOT part of the Drake scene.

    The Drake table is visual-only, so without this the belt drops through it before it
    settles on the holder.  The box spans the table footprint, top face at TABLE_TOP_Z.
    """
    center = 0.5 * (table_aabb[0] + table_aabb[1])
    half = 0.5 * (table_aabb[1] - table_aabb[0])
    return builder.add_shape_box(
        body=-1,
        xform=wp.transform(
            wp.vec3(float(center[0]), float(center[1]),
                    TABLE_TOP_Z - 0.5 * TABLETOP_COLLISION_THICKNESS),
            wp.quat_identity(),
        ),
        hx=float(half[0]), hy=float(half[1]), hz=0.5 * TABLETOP_COLLISION_THICKNESS,
        cfg=cfg, color=wp.vec3(0.55, 0.35, 0.14), label="tabletop_collision",
    )


def _enable_gravity_compensation(builder: newton.ModelBuilder, bodies: Sequence[int]) -> None:
    """MuJoCo gravity compensation for both arms + gripper (as round_belt.py); done on
    the builder because ``mujoco:gravcomp`` is a builder custom attribute, so it cannot
    be set later from ``apply_default_joint_state(model, ...)``."""
    try:
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in bodies:
            gravcomp.values[body] = 1.0
    except (KeyError, AttributeError):
        logger.warning("mujoco:gravcomp attribute not available; skipping gravity compensation.")


def build_scene(builder: newton.ModelBuilder) -> SceneInfo:
    """Build the whole Drake scene into ``builder``. No viewer, no solver."""
    visual_cfg = round_belt.make_visual_cfg()
    collision_cfg = round_belt.make_robust_table_collision_cfg(visible=False)
    add_static = functools.partial(
        add_urdf_as_static_shapes, builder, visual_cfg=visual_cfg, collision_cfg=collision_cfg
    )

    static_shape_start = builder.shape_count
    static_labels: dict[str, int] = {}
    aabbs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 1. table + Franka mount (visual only; Drake has no table collision).
    static_labels.update(
        add_static(TABLE_URDF, X_W_TABLE, label_prefix="table",
                   visual_color=(0.55, 0.35, 0.14), collect_aabbs=aabbs)
    )
    table_aabb = aabbs.get(TABLE_VISUAL_LABEL)
    if table_aabb is None:
        candidates = sorted(l for l in aabbs if l.startswith("table/"))
        raise RuntimeError(
            f"table visual AABB {TABLE_VISUAL_LABEL!r} not found in the collected AABBs; "
            f"scene.urdf's visual labels changed. Available table labels: {candidates}"
        )
    ground_height = float(table_aabb[0][2])
    tabletop_collision_shape = _add_tabletop_collision(builder, table_aabb, collision_cfg)
    static_labels["tabletop_collision"] = int(tabletop_collision_shape)

    # 2. task board (board mesh + collision box + the two fixed pulleys).
    static_labels.update(
        add_static(BOARD_URDF, X_W_BOARD, label_prefix="board", visual_color=BOARD_COLOR,
                   split_components=True, color_fn=_small_pulley_mount_color)
    )

    # White small pulley, black large pulley -- matching round_belt.py, not the board SDF
    # (see the colour constants for why).
    for shape_index in range(static_shape_start, builder.shape_count):
        label = builder.shape_label[shape_index] or ""
        if label.startswith("board/small_round_pulley/visual"):
            builder.shape_color[shape_index] = SMALL_PULLEY_COLOR
        elif label.startswith("board/large_round_pulley/visual"):
            builder.shape_color[shape_index] = LARGE_PULLEY_COLOR

    # 3. belt chain holder.
    static_labels.update(
        add_static(HOLDER_URDF, X_W_HOLDER, label_prefix="belt_chain_holder",
                   visual_color=(0.20, 0.60, 1.00))
    )

    static_shapes = list(range(static_shape_start, builder.shape_count))

    # 4. Franka arm (welded at the world origin).
    franka_body_start = builder.body_count
    franka_joint_start = builder.joint_count
    franka_shape_start = builder.shape_count
    builder.add_urdf(str(PANDA_ARM_URDF), xform=X_W_PANDA, floating=False,
                     enable_self_collisions=False, collapse_fixed_joints=False)

    # 5. Franka long-finger hand (welded to panda_link8, must follow the arm).
    link8 = body_index(builder.body_label, "panda_arm/panda_link8")
    builder.add_urdf(str(PANDA_HAND_URDF), parent_body=link8, xform=X_LINK8_HAND,
                     floating=False, enable_self_collisions=False,
                     collapse_fixed_joints=False)
    franka_bodies = list(range(franka_body_start, builder.body_count))
    franka_joints = list(range(franka_joint_start, builder.joint_count))
    label_shapes_by_body(builder, franka_shape_start, builder.shape_count)
    flipped = fix_inverted_mesh_winding(builder, franka_shape_start, builder.shape_count)
    if flipped:
        logger.info(f"Flipped {len(flipped)} inward-wound Franka mesh(es): {', '.join(flipped)}")
    neutralize_textured_shape_colors(builder, franka_shape_start, builder.shape_count)

    # 6. UR10 (USD asset; its root frame == the URDF base_link frame, so the xform is the
    #    Drake weld unchanged).
    ur10_body_start = builder.body_count
    ur10_joint_start = builder.joint_count
    builder.add_usd(
        # Downloaded/cached exactly as round_belt.py does.
        str(Path(newton.utils.download_asset(UR10_USD_ASSET)).joinpath(*UR10_USD_RELPATH)),
        xform=X_W_UR10, collapse_fixed_joints=False, enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    ur10_bodies = list(range(ur10_body_start, builder.body_count))
    ur10_joints = list(range(ur10_joint_start, builder.joint_count))

    # 7. Robotiq 2F-85 (welded to wrist_3_link, must follow the UR10).
    wrist3 = body_index(builder.body_label, UR10_WRIST3_LABEL)
    gripper_body_start = builder.body_count
    gripper_joint_start = builder.joint_count
    builder.add_mjcf(str(ROBOTIQ_MJCF), parent_body=wrist3, xform=X_USDWRIST3_GRIPPER,
                     enable_self_collisions=False)
    gripper_body_end = builder.body_count
    gripper_bodies = list(range(gripper_body_start, gripper_body_end))
    gripper_joints = list(range(gripper_joint_start, builder.joint_count))

    gripper_pad_bodies = round_belt._select_gripper_proxy_bodies(
        builder, gripper_body_start, gripper_body_end
    )
    if len(gripper_pad_bodies) != 2:
        raise RuntimeError(
            f"Two-plane gripper contact requires exactly 2 pad bodies; got {gripper_pad_bodies}"
        )
    round_belt._replace_proxy_pad_colliders_with_two_planes(builder, gripper_pad_bodies)

    # Show (and contact the belt with) the ALOHA-style fingers instead of the 2f85's own
    # pads: add the finger meshes, then drop the pad visuals and the two flat
    # ``simple_belt_contact`` planes round_belt.py substitutes for the pad colliders.
    # The fingers carry collision, so the gripper keeps its belt contact geometry.
    aloha_finger_shapes = _add_aloha_fingers(builder, gripper_body_start, gripper_pad_bodies)
    dropped_pad_visuals = hide_shapes(
        builder, lambda label: label.endswith("pad_geom_0_visual") or "silicone_pad_geom" in label
    )
    dropped_planes = hide_shapes(builder, lambda label: label.startswith("simple_belt_contact"))
    gripper_pad_shapes = list(aloha_finger_shapes)
    logger.info(
        f"ALOHA fingers: added {len(aloha_finger_shapes)} finger meshes; "
        f"dropped {dropped_pad_visuals} 2f85 pad visuals and {dropped_planes} "
        f"simple_belt_contact planes."
    )

    robot_bodies = list(range(franka_body_start, builder.body_count))
    robot_joints = list(range(franka_joint_start, builder.joint_count))
    robot_shapes = list(range(franka_shape_start, builder.shape_count))

    _enable_gravity_compensation(builder, robot_bodies)

    # 8. round belt.
    belt_bodies, belt_joints, belt_shapes = _add_belt(builder)

    # 9. ground plane, level with the bottom of the table visual.
    ground_shape = builder.add_ground_plane(height=ground_height)

    logger.info(
        f"Scene: {len(static_shapes)} static shapes, {len(robot_bodies)} robot bodies, "
        f"{len(belt_bodies)} belt bodies; table AABB z = [{table_aabb[0][2]:.5f}, "
        f"{table_aabb[1][2]:.5f}], ground z = {ground_height:.5f}, "
        f"tabletop_collision top z = {TABLE_TOP_Z:.5f}."
    )

    return SceneInfo(
        robot_bodies=robot_bodies, robot_joints=robot_joints, robot_shapes=robot_shapes,
        franka_bodies=franka_bodies, franka_joints=franka_joints,
        ur10_bodies=ur10_bodies, ur10_joints=ur10_joints,
        gripper_bodies=gripper_bodies, gripper_joints=gripper_joints,
        gripper_pad_bodies=[int(b) for b in gripper_pad_bodies],
        gripper_pad_shapes=[int(s) for s in gripper_pad_shapes],
        belt_bodies=belt_bodies, belt_joints=belt_joints, belt_shapes=belt_shapes,
        static_shapes=static_shapes, static_shape_labels=static_labels,
        tabletop_collision_shape=int(tabletop_collision_shape),
        ground_shape=int(ground_shape), ground_height=ground_height,
        table_visual_aabb=(table_aabb[0], table_aabb[1]),
    )
