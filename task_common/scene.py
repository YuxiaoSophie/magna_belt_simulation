"""Assembly of a belt task's directives scene into a ``newton.ModelBuilder``.

:func:`build_task_scene` is :func:`utils.directives.load_directives` followed by the mapping
of the loaded model ranges onto the :class:`SceneInfo` index bookkeeping the solver
partition needs.  The directive order fixes every body and shape index.  No viewer and no
solver here, and nothing touches the finalized ``Model``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
from loguru import logger

import newton
from newton.solvers import SolverMuJoCo, SolverVBD

import round_belt
from task_common.cameras import CameraSpec
from task_common.point_cloud import PointCloudSpec
from utils.directives import DirectiveFn, ModelRecord, load_directives


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
    """Builder-index bookkeeping produced by :func:`build_task_scene`."""

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
    cameras: list[CameraSpec] = field(default_factory=list)
    point_clouds: list[PointCloudSpec] = field(default_factory=list)
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


def span(records: Sequence[ModelRecord], what: str) -> tuple[list[int], list[int], list[int]]:
    """``(bodies, joints, shapes)`` of consecutive models; raises unless their ranges abut."""
    for before, after in zip(records, records[1:]):
        ends = (before.body_end, before.joint_end, before.shape_end)
        starts = (after.body_start, after.joint_start, after.shape_start)
        if ends != starts:
            raise RuntimeError(
                f"{what}: {before.directive.name!r} ends at (body, joint, shape) {ends} but "
                f"{after.directive.name!r} starts at {starts}; something in the directives "
                "file is ordered between them"
            )
    first, last = records[0], records[-1]
    return (
        list(range(first.body_start, last.body_end)),
        list(range(first.joint_start, last.joint_end)),
        list(range(first.shape_start, last.shape_end)),
    )


FINGER_COLLIDER_SUFFIX = "_aloha_finger_collision"
UR10_BASE_LABEL = "/ur10/base_link"
UR10_WRIST3_LABEL = "/ur10/wrist_3_link"


def finger_colliders(
    builder: newton.ModelBuilder, gripper: ModelRecord, pad_bodies: Sequence[int]
) -> list[int]:
    """The one ALOHA finger collider on each pad body, in ``pad_bodies`` order."""
    shapes = []
    for body in pad_bodies:
        hits = [
            s for s in gripper.shapes
            if int(builder.shape_body[s]) == int(body)
            and str(builder.shape_label[s] or "").endswith(FINGER_COLLIDER_SUFFIX)
        ]
        if len(hits) != 1:
            raise RuntimeError(
                f"pad body {builder.body_label[body]!r} carries {len(hits)} shapes labelled "
                f"*{FINGER_COLLIDER_SUFFIX}, expected exactly 1; 2f85.xml's finger geoms changed"
            )
        shapes.append(int(hits[0]))
    return shapes


def build_task_scene(
    builder: newton.ModelBuilder,
    directives_path: Path,
    *,
    directives: Mapping[str, DirectiveFn],
    belt_name: str,
    table_visual_label: str,
    table_top_z: float,
) -> SceneInfo:
    """Build the whole Drake scene into ``builder`` from ``directives_path``; no solver."""
    scene = load_directives(
        builder, directives_path, directives=directives,
        visual_cfg=round_belt.make_visual_cfg(),
        collision_cfg=round_belt.make_robust_table_collision_cfg(visible=False),
    )
    models = scene.models
    statics = [record for record in models.values() if record.directive.kind == "static"]
    robots = [record for record in models.values() if record.directive.kind != "static"]
    tabletop = scene.extras["tabletop_collision"]
    belt = scene.extras[belt_name]
    ground = scene.extras["ground"]

    # Statics: table, the tabletop safety box (a custom directive, not a model), board,
    # holder -- one contiguous shape range with nothing else inside it.
    static_shapes = list(range(statics[0].shape_start, statics[-1].shape_end))
    tabletop_collision_shape = int(tabletop["shape"])
    owned = {s for record in statics for s in record.shapes} | {tabletop_collision_shape}
    if set(static_shapes) != owned:
        raise RuntimeError(
            f"static shape range {static_shapes[0]}..{static_shapes[-1]} and the static "
            f"models' shapes differ by {sorted(set(static_shapes) ^ owned)}; fix the order"
        )
    labels = {label: s for record in statics for label, s in record.shape_labels.items()}
    labels["tabletop_collision"] = tabletop_collision_shape
    static_labels = dict(sorted(labels.items(), key=lambda item: item[1]))

    # Robots: Franka arm + hand, UR10, gripper -- contiguous in bodies, joints and shapes.
    franka_bodies, franka_joints, _ = span([models["panda_arm"], models["panda_hand"]], "franka")
    ur10, gripper = models["ur10"], models["robotiq_2f85"]
    robot_bodies, robot_joints, robot_shapes = span(robots, "robots")
    pad_bodies = round_belt._select_gripper_proxy_bodies(
        builder, gripper.body_start, gripper.body_end
    )
    if len(pad_bodies) != 2:
        raise RuntimeError(f"gripper contact needs exactly 2 pad bodies; got {pad_bodies}")
    pad_shapes = finger_colliders(builder, gripper, pad_bodies)

    table_aabb = scene.aabbs[table_visual_label]
    ground_height = float(ground["height"])
    logger.info(
        f"Scene: {len(static_shapes)} static shapes, {len(robot_bodies)} robot bodies, "
        f"{len(belt['bodies'])} belt bodies; table AABB z = [{table_aabb[0][2]:.5f}, "
        f"{table_aabb[1][2]:.5f}], ground z = {ground_height:.5f}, "
        f"tabletop_collision top z = {table_top_z:.5f}."
    )

    return SceneInfo(
        robot_bodies=robot_bodies, robot_joints=robot_joints, robot_shapes=robot_shapes,
        franka_bodies=franka_bodies, franka_joints=franka_joints,
        ur10_bodies=ur10.bodies, ur10_joints=ur10.joints,
        gripper_bodies=gripper.bodies, gripper_joints=gripper.joints,
        gripper_pad_bodies=[int(b) for b in pad_bodies], gripper_pad_shapes=pad_shapes,
        belt_bodies=belt["bodies"], belt_joints=belt["joints"], belt_shapes=belt["shapes"],
        static_shapes=static_shapes, static_shape_labels=static_labels,
        tabletop_collision_shape=tabletop_collision_shape,
        ground_shape=int(ground["shape"]), ground_height=ground_height,
        table_visual_aabb=(table_aabb[0], table_aabb[1]),
        cameras=[e for e in scene.extras.values() if isinstance(e, CameraSpec)],
        point_clouds=[e for e in scene.extras.values() if isinstance(e, PointCloudSpec)],
    )
