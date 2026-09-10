#!/usr/bin/env python3
"""Headless, solver-free pose check for the Drake round-belt scene port.

Builds the Newton scene from ``round_belt_task_simulation.py``, runs forward
kinematics only (no solver stepping, no viewer), and asserts every weld pose,
default joint angle and belt placement against the numbers transcribed from
the Drake yaml (see ``round_belt_task_simulation.py``'s own docstring/constants
block for the source files). Check 6 recomputes the Franka/UR10 chain FK
directly from the raw URDF ``<origin>``/``<axis>`` data with an independent
numpy implementation, so the check does not just compare Newton's kinematics
against itself.

Run:
    uv run python scripts/check_round_belt_task_poses.py
    uv run python scripts/check_round_belt_task_poses.py --device cpu
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # round_belt_task_simulation.py lives at the repo root; make this importable
    # regardless of the CWD the script is invoked from.
    sys.path.insert(0, str(REPO_ROOT))

import newton
import round_belt_task as scene


# ----------------------------------------------------------------------------
# Small numpy/Warp transform helpers (see package implementation notes).
# ----------------------------------------------------------------------------


def rpy_to_mat3(rpy_rad) -> np.ndarray:
    """R = Rz(yaw) . Ry(pitch) . Rx(roll) -- the Drake/URDF rpy convention."""
    r, p, y = rpy_rad
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def mat4_from_xyz_rpy(xyz, rpy_rad) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rpy_to_mat3(rpy_rad)
    m[:3, 3] = xyz
    return m


def mat4_translate(xyz) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = xyz
    return m


def mat4_rz(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return m


def rot_axis_angle(axis, angle: float) -> np.ndarray:
    """Rodrigues' formula: 3x3 rotation matrix for an arbitrary unit axis."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    cc = 1.0 - c
    return np.array(
        [
            [x * x * cc + c, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, y * y * cc + c, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, z * z * cc + c],
        ]
    )


def tf_to_mat4(t: wp.transform) -> np.ndarray:
    p = np.array([float(v) for v in wp.transform_get_translation(t)], dtype=np.float64)
    q = wp.transform_get_rotation(t)
    r = np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = p
    return m


def mat4_from_pos_quat(pos, quat_xyzw) -> np.ndarray:
    q = wp.quat(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3]))
    r = np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = [float(v) for v in pos]
    return m


def body_mat4(body_q: np.ndarray, idx: int) -> np.ndarray:
    row = body_q[idx]
    return mat4_from_pos_quat(row[:3], row[3:7])


def tf_mat4(tf) -> np.ndarray:
    """wp.transform (px py pz qx qy qz qw) -> 4x4 homogeneous matrix."""
    row = [float(v) for v in tf]
    return mat4_from_pos_quat(row[:3], row[3:7])


def rpy_deg_from_mat3(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rpy_to_mat3: extract Rz.Ry.Rx angles in degrees."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 0.999999:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))


def assert_close_tf(name: str, a: np.ndarray, b: np.ndarray, atol_pos: float, atol_rot: float) -> None:
    """Compare two 4x4 world transforms; message reports mm / degrees on failure."""
    pos_err = np.linalg.norm(a[:3, 3] - b[:3, 3])
    rot_err_max = float(np.max(np.abs(a[:3, :3] - b[:3, :3])))
    r_rel = a[:3, :3].T @ b[:3, :3]
    cos_ang = max(-1.0, min(1.0, (np.trace(r_rel) - 1.0) / 2.0))
    ang_deg = math.degrees(math.acos(cos_ang))
    if pos_err > atol_pos or rot_err_max > atol_rot:
        raise AssertionError(
            f"{name}: position error {pos_err * 1000.0:.4f} mm (atol {atol_pos * 1000.0:.4f} mm), "
            f"rotation error {ang_deg:.4f} deg (max abs matrix-entry diff {rot_err_max:.3e}, "
            f"atol {atol_rot:.1e})\nA=\n{a}\nB=\n{b}"
        )


# ----------------------------------------------------------------------------
# Independent numpy FK, computed straight from the raw URDF <origin>/<axis>.
# ----------------------------------------------------------------------------


def _parse_vec3(text: str | None, default) -> tuple[float, float, float]:
    if text is None:
        return default
    parts = [float(v) for v in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3 numbers, got {text!r}")
    return (parts[0], parts[1], parts[2])


def urdf_chain_transform(urdf_path: Path, root_link: str, target_link: str, q_by_joint: dict) -> np.ndarray:
    """Compose T(origin.xyz) . R_rpy(origin.rpy) . Rot(axis, q) for every joint
    on the path from ``root_link`` to ``target_link`` (fixed joints contribute
    only their origin). Walks the URDF's own parent/child graph -- nothing
    about chain length or joint order is assumed up front.
    """
    root = ET.parse(str(urdf_path)).getroot()
    child_to_joint = {j.find("child").get("link"): j for j in root.findall("joint")}

    chain = []
    link = target_link
    while link != root_link:
        joint = child_to_joint.get(link)
        if joint is None:
            raise KeyError(f"{urdf_path.name}: no path from {root_link!r} to {target_link!r} (stuck at {link!r})")
        chain.append(joint)
        link = joint.find("parent").get("link")
    chain.reverse()

    m = np.eye(4)
    for joint in chain:
        origin = joint.find("origin")
        xyz = _parse_vec3(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_vec3(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        m = m @ mat4_from_xyz_rpy(xyz, rpy)

        jtype = joint.get("type")
        if jtype in ("revolute", "continuous", "prismatic"):
            name = joint.get("name")
            if name not in q_by_joint:
                raise KeyError(f"{urdf_path.name}: no q supplied for movable joint {name!r}")
            axis_elem = joint.find("axis")
            axis = _parse_vec3(axis_elem.get("xyz") if axis_elem is not None else None, (0.0, 0.0, 1.0))
            q = q_by_joint[name]
            if jtype == "prismatic":
                m = m @ mat4_translate(np.asarray(axis, dtype=np.float64) * q)
            else:
                rot = np.eye(4)
                rot[:3, :3] = rot_axis_angle(axis, q)
                m = m @ rot
        elif jtype != "fixed":
            raise ValueError(f"{urdf_path.name}: unsupported joint type {jtype!r} on joint {joint.get('name')!r}")

    return m


# ----------------------------------------------------------------------------
# Source-of-truth constants, transcribed HERE from the Drake yaml.
# ----------------------------------------------------------------------------
# These are this script's own copies, typed independently of
# ``round_belt_task_simulation.py``.  Every check below compares the scene module
# against THESE numbers, never against the module's own constants -- otherwise a
# transcription typo in a translation would compare equal to itself and pass.
#
# Source: magna/models/round_belt_task/round-belt-scene.dmd.yaml
#   add_weld world -> nist_board::board
YAML_BOARD_XYZ = (0.64483928, -0.19718233, 0.01076393)
YAML_BOARD_RPY_DEG = (-3.32822058e-01, -6.87450103e-02, 8.95207485e01)
#   add_weld world -> belt_holder::belt_chain_holder_first_half
YAML_HOLDER_XYZ = (0.4736603358808432, 0.3520562100563749, -0.02858)
YAML_HOLDER_RPY_DEG = (0.0, 0.0, 90.0)
#   add_weld world -> ur10::base_link
YAML_UR10_XYZ = (1.33235648, -0.15491427, 0.03076191)
YAML_UR10_RPY_DEG = (-0.35720724, 0.72781324, 179.78723941)
#   add_weld world -> panda::panda_link0  (no X_PC => identity)
YAML_PANDA_XYZ = (0.0, 0.0, 0.0)
YAML_PANDA_RPY_DEG = (0.0, 0.0, 0.0)
#   add_weld panda::panda_link8 -> panda_hand::panda_hand
YAML_LINK8_HAND_XYZ = (0.0, 0.0, 0.0)
YAML_LINK8_HAND_RPY_DEG = (0.0, 0.0, -45.0)
#   add_weld world -> table::scene  (no X_PC => identity)
YAML_TABLE_XYZ = (0.0, 0.0, 0.0)
YAML_TABLE_RPY_DEG = (0.0, 0.0, 0.0)


def yaml_mat4(xyz, rpy_deg) -> np.ndarray:
    """4x4 pose from a Drake ``translation`` + ``!Rpy { deg: ... }`` pair."""
    return mat4_from_xyz_rpy(xyz, tuple(math.radians(a) for a in rpy_deg))


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------

PANDA_HAND_LINKS = ["panda_hand", "panda_leftfinger", "panda_rightfinger", "finger_tip", "panda_finger_tip_center"]
# The scene builds the UR10 from the NVIDIA USD asset, not assets/common/ur10/ur10.urdf
# (the Drake glTFs carry no textures).  The USD exposes 8 bodies under the prim
# path prefix "/ur10/"; the URDF-only massless frames (base_link_inertia, base,
# ft_frame, flange, tool0) simply do not exist in it.  The URDF is still the
# kinematic source of truth -- check 7 walks it with an independent numpy FK and
# asserts the USD arm lands in the same world pose.
UR10_LINKS = [
    "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link", "ee_link",
]
UR10_LABEL_PREFIX = "/ur10/"


def check_build(ctx: dict) -> None:
    """1. Build the scene, finalize, seed defaults, run eval_fk."""
    builder = scene.make_builder()
    info = scene.build_scene(builder)
    model = builder.finalize()
    scene.apply_default_joint_state(model, info)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    body_q = state.body_q.numpy()
    body_labels = list(model.body_label)
    joint_labels = list(model.joint_label)
    assert model.body_count > 0, "model has no bodies"
    assert np.isfinite(body_q).all(), "non-finite body_q after eval_fk"
    ctx.update(
        model=model, info=info, state=state, body_q=body_q,
        body_labels=body_labels, joint_labels=joint_labels,
    )


def check_convention(ctx: dict) -> None:
    """2. Drake !Rpy{deg} == Rz(yaw).Ry(pitch).Rx(roll) -- board rotation."""
    rpy_rad = tuple(math.radians(a) for a in YAML_BOARD_RPY_DEG)
    expected = rpy_to_mat3(rpy_rad)
    q = wp.transform_get_rotation(scene.X_W_BOARD)
    got = np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3)
    # atol=1e-6, rtol=0: the scene's quaternions are float32, so the achieved
    # agreement is ~1.3e-7; 1e-9 would be a false precision claim and would only
    # pass because np.allclose's default rtol=1e-5 dominates.  A wrong rotation
    # order errs by 7e-3 and an inverted rotation by 2.0, so 1e-6 keeps the check
    # fully load-bearing.
    if not np.allclose(got, expected, atol=1e-6, rtol=0.0):
        raise AssertionError(
            f"X_W_BOARD rotation != independent Rz@Ry@Rx(board rpy); "
            f"max abs error = {np.abs(got - expected).max():.3e}"
        )


def check_scene_constants(ctx: dict) -> None:
    """3. Scene weld constants == this script's own yaml transcription."""
    for name, tf, xyz, rpy_deg in (
        ("X_W_BOARD", scene.X_W_BOARD, YAML_BOARD_XYZ, YAML_BOARD_RPY_DEG),
        ("X_W_HOLDER", scene.X_W_HOLDER, YAML_HOLDER_XYZ, YAML_HOLDER_RPY_DEG),
        ("X_W_UR10", scene.X_W_UR10, YAML_UR10_XYZ, YAML_UR10_RPY_DEG),
        ("X_W_PANDA", scene.X_W_PANDA, YAML_PANDA_XYZ, YAML_PANDA_RPY_DEG),
        ("X_W_TABLE", scene.X_W_TABLE, YAML_TABLE_XYZ, YAML_TABLE_RPY_DEG),
        ("X_LINK8_HAND", scene.X_LINK8_HAND, YAML_LINK8_HAND_XYZ, YAML_LINK8_HAND_RPY_DEG),
    ):
        assert_close_tf(
            f"round_belt_task_simulation.{name} vs round-belt-scene.dmd.yaml",
            tf_to_mat4(tf),
            yaml_mat4(xyz, rpy_deg),
            # wp.transform stores float32, so ~1e-7 relative is the floor here;
            # 1e-6 m still catches a 1 mm transcription typo by 1000x.
            atol_pos=1e-6,
            atol_rot=1e-6,
        )

    # The belt-holder weld translation IS the belt centre in the Drake yaml, and
    # the holder weld z IS the Drake table-top height; both are re-used as
    # separate module constants, so tie them back to the yaml here too.
    if abs(scene.TABLE_TOP_Z - YAML_HOLDER_XYZ[2]) > 1e-12:
        raise AssertionError(
            f"scene.TABLE_TOP_Z={scene.TABLE_TOP_Z} != holder weld z {YAML_HOLDER_XYZ[2]}"
        )
    belt_xy_err = float(np.linalg.norm(np.asarray(scene.BELT_CENTER[:2]) - np.asarray(YAML_HOLDER_XYZ[:2])))
    if belt_xy_err > 1e-12:
        raise AssertionError(
            f"scene.BELT_CENTER xy {scene.BELT_CENTER[:2]} != holder weld xy "
            f"{YAML_HOLDER_XYZ[:2]}, error={belt_xy_err:.3e}"
        )

    # UR10 default joint positions, transcribed from ur.dmd.yaml.
    yaml_ur10_q = [0.0, -1.57079632679, 1.57079632679, -1.57079632679, -1.57079632679, 0.0]
    if len(scene.UR10_DEFAULT_Q) != len(yaml_ur10_q) or any(
        abs(a - b) > 1e-12 for a, b in zip(scene.UR10_DEFAULT_Q, yaml_ur10_q)
    ):
        raise AssertionError(
            f"scene.UR10_DEFAULT_Q={scene.UR10_DEFAULT_Q} != ur.dmd.yaml default_joint_positions {yaml_ur10_q}"
        )

    # Franka start pose: NOT franka.dmd.yaml's "ready" pose -- the round-belt
    # scene yaml gives panda_arm.urdf no default_joint_positions at all, so the
    # configuration the sim starts in is what magna_simulation.cc:187-189 seeds
    # from round_belt_simulation_params.yaml:9-10.
    yaml_franka_q = [1.71717, 1.20002, -1.4548, -2.25837, 1.259265, 1.94719, 0.331838]
    yaml_franka_hand_q = [-0.004, 0.004]
    if len(scene.PANDA_DEFAULT_Q) != len(yaml_franka_q) or any(
        abs(a - b) > 1e-12 for a, b in zip(scene.PANDA_DEFAULT_Q, yaml_franka_q)
    ):
        raise AssertionError(
            f"scene.PANDA_DEFAULT_Q={scene.PANDA_DEFAULT_Q} != q_init_franka {yaml_franka_q}"
        )
    if len(scene.PANDA_FINGER_DEFAULT_Q) != 2 or any(
        abs(a - b) > 1e-12 for a, b in zip(scene.PANDA_FINGER_DEFAULT_Q, yaml_franka_hand_q)
    ):
        raise AssertionError(
            f"scene.PANDA_FINGER_DEFAULT_Q={scene.PANDA_FINGER_DEFAULT_Q} != "
            f"q_init_franka_hand {yaml_franka_hand_q}"
        )


def check_static_shapes(ctx: dict) -> None:
    """4. Static shape world poses: board/board/collision0, holder collision0."""
    model = ctx["model"]
    info = ctx["info"]
    shape_transform = model.shape_transform.numpy()

    board_label = "board/board/collision0"
    idx = info.static_shape_labels[board_label]
    got = mat4_from_pos_quat(shape_transform[idx][:3], shape_transform[idx][3:7])
    expected = yaml_mat4(YAML_BOARD_XYZ, YAML_BOARD_RPY_DEG) @ mat4_translate((0.192, 0.192, -0.005))
    assert_close_tf(board_label, got, expected, atol_pos=1e-6, atol_rot=1e-6)

    holder_label = "belt_chain_holder/belt_chain_holder_first_half/collision0"
    idx = info.static_shape_labels[holder_label]
    got = mat4_from_pos_quat(shape_transform[idx][:3], shape_transform[idx][3:7])
    expected = yaml_mat4(YAML_HOLDER_XYZ, YAML_HOLDER_RPY_DEG)
    assert_close_tf(holder_label, got, expected, atol_pos=1e-6, atol_rot=1e-6)


def check_body_poses(ctx: dict) -> None:
    """5. panda_link0, ur10/base_link, panda_hand, 2f85 root weld poses."""
    body_q = ctx["body_q"]
    body_labels = ctx["body_labels"]

    link0 = scene.body_index(body_labels, "panda_arm/panda_link0")
    assert_close_tf("panda_arm/panda_link0", body_mat4(body_q, link0), np.eye(4), 1e-5, 1e-5)

    # The USD's root/base_link frame is identical to the Drake URDF's base_link
    # frame, so this weld assertion is unchanged (and it is what proves it).
    ur10_base = scene.body_index(body_labels, scene.UR10_BASE_LABEL)
    assert_close_tf(
        scene.UR10_BASE_LABEL,
        body_mat4(body_q, ur10_base),
        yaml_mat4(YAML_UR10_XYZ, YAML_UR10_RPY_DEG),
        1e-5,
        1e-5,
    )

    link8 = scene.body_index(body_labels, "panda_arm/panda_link8")
    hand = scene.body_index(body_labels, "panda_hand/panda_hand")
    expected_hand = body_mat4(body_q, link8) @ yaml_mat4(YAML_LINK8_HAND_XYZ, YAML_LINK8_HAND_RPY_DEG)
    assert_close_tf("panda_hand/panda_hand", body_mat4(body_q, hand), expected_hand, 1e-5, 1e-5)

    # The USD's wrist_3_link BODY frame is not the ROS/Drake URDF's frame for the
    # same link (it sits d6 = 92.2 mm back and is rotated Rx(-90 deg)).  Every
    # wrist-3-relative assertion below is therefore expressed in the Drake frame
    # obtained by mapping through scene.X_USDWRIST3_URDFWRIST3, so the numbers
    # are directly comparable to the previous URDF-based build.  That constant is
    # independently validated in check 7.
    wrist3 = scene.body_index(body_labels, scene.UR10_WRIST3_LABEL)
    wrist3_drake_mat = body_mat4(body_q, wrist3) @ tf_mat4(scene.X_USDWRIST3_URDFWRIST3)
    gripper_matches = [i for i, lbl in enumerate(body_labels) if str(lbl).endswith("/base_mount")]
    if len(gripper_matches) != 1:
        raise AssertionError(f"expected exactly one body label ending in '/base_mount', got {gripper_matches}")
    gripper_root = gripper_matches[0]
    ctx["gripper_root_label"] = body_labels[gripper_root]
    # 2f85.xml's own <body name="base_mount" pos="0 0 0.007" .../> (the MJCF
    # root's first child body) puts the actual "base_mount" body 7 mm along
    # local +Z of the weld frame X(wrist3).Rz(90) that add_mjcf's xform= is
    # applied at; that weld frame itself (not a body) is exactly
    # X(wrist3).Rz(90) with zero translation, per the ur.dmd.yaml weld.
    expected_gripper = wrist3_drake_mat @ mat4_rz(90.0) @ mat4_translate((0.0, 0.0, 0.007))
    assert_close_tf(
        body_labels[gripper_root], body_mat4(body_q, gripper_root), expected_gripper, 1e-5, 1e-5
    )
    ctx["gripper_root"] = gripper_root
    ctx["wrist3"] = wrist3
    ctx["wrist3_drake_mat"] = wrist3_drake_mat
    ctx["link8"] = link8


def check_gripper_geometry(ctx: dict) -> None:
    """6. Pad separation/orientation relative to the wrist-3 frame."""
    body_q = ctx["body_q"]
    body_labels = ctx["body_labels"]
    info = ctx["info"]

    left_matches = [i for i, lbl in enumerate(body_labels) if str(lbl).endswith("/left_pad")]
    right_matches = [i for i, lbl in enumerate(body_labels) if str(lbl).endswith("/right_pad")]
    if len(left_matches) == 1 and len(right_matches) == 1:
        p_left_idx, p_right_idx = left_matches[0], right_matches[0]
    else:
        if len(info.gripper_pad_bodies) != 2:
            raise AssertionError(
                f"no unique /left_pad,/right_pad body labels and info.gripper_pad_bodies "
                f"is not length 2: {info.gripper_pad_bodies}"
            )
        p_left_idx, p_right_idx = info.gripper_pad_bodies

    p_left = np.array(body_q[p_left_idx][:3], dtype=np.float64)
    p_right = np.array(body_q[p_right_idx][:3], dtype=np.float64)
    wrist3_mat = ctx["wrist3_drake_mat"]
    r_w3 = wrist3_mat[:3, :3]
    x_w3, y_w3, z_w3 = r_w3[:, 0], r_w3[:, 1], r_w3[:, 2]

    d = p_left - p_right
    dx = abs(float(np.dot(d, x_w3)))
    dy = abs(float(np.dot(d, y_w3)))
    if dx >= 0.002:
        raise AssertionError(f"pad separation has {dx * 1000:.3f} mm along wrist3 +X (expected < 2 mm)")
    if dy <= 0.04:
        raise AssertionError(f"pad separation along wrist3 +Y is only {dy * 1000:.3f} mm (expected > 40 mm)")

    mean_pad = 0.5 * (p_left + p_right)
    rel = mean_pad - wrist3_mat[:3, 3]
    proj_z = float(np.dot(rel, z_w3))
    if not (0.10 <= proj_z <= 0.20):
        raise AssertionError(f"mean pad position projects to {proj_z * 1000:.2f} mm along wrist3 +Z (expected [100, 200] mm)")

    ctx["pad_dx_mm"] = dx * 1000.0
    ctx["pad_dy_mm"] = dy * 1000.0
    ctx["pad_proj_z_mm"] = proj_z * 1000.0


def check_independent_fk(ctx: dict) -> None:
    """7. Independent numpy FK from the raw URDF origins/axes."""
    body_q = ctx["body_q"]
    link8 = ctx["link8"]

    panda_q = {f"panda_joint{i}": q for i, q in zip(range(1, 8), scene.PANDA_DEFAULT_Q)}
    chain_panda = urdf_chain_transform(scene.PANDA_ARM_URDF, "panda_link0", "panda_link8", panda_q)
    x_w_link8_numpy = yaml_mat4(YAML_PANDA_XYZ, YAML_PANDA_RPY_DEG) @ chain_panda

    ur10_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    ur10_q = dict(zip(ur10_joint_names, scene.UR10_DEFAULT_Q))
    chain_ur10 = urdf_chain_transform(scene.UR10_URDF, "base_link", "wrist_3_link", ur10_q)
    x_w_wrist3_numpy = yaml_mat4(YAML_UR10_XYZ, YAML_UR10_RPY_DEG) @ chain_ur10

    assert_close_tf("panda_arm/panda_link8 (independent FK)", body_mat4(body_q, link8), x_w_link8_numpy, 1e-5, 1e-5)
    # This simultaneously validates (a) that the USD arm lands where the Drake
    # URDF kinematics say it should and (b) that scene.X_USDWRIST3_URDFWRIST3
    # (written structurally as T(0, 0.0922, 0).Rx(-90 deg), 0.0922 being the
    # wrist_3_joint origin in assets/common/ur10/ur10.urdf) is the correct USD->URDF
    # wrist-3 frame conversion.  A wrong conversion fails by >= 92 mm.
    assert_close_tf(
        f"{scene.UR10_WRIST3_LABEL} . X_USDWRIST3_URDFWRIST3 (independent FK)",
        ctx["wrist3_drake_mat"], x_w_wrist3_numpy, 1e-5, 1e-5,
    )

    # Absolute regression anchor for panda_link8 in the world frame.  Recomputed
    # from this file's own numpy FK after the start pose moved from the
    # franka.dmd.yaml "ready" pose to the sim's q_init_franka
    # (round_belt_simulation_params.yaml:9); the previous anchor for the ready
    # pose was (0.307, 0, 0.590).  It catches a regression in the weld/pose
    # constants, not an error in the FK itself.
    link8_pos = x_w_link8_numpy[:3, 3]
    anchor = np.array([0.48362, 0.15700, 0.23423])
    dist = float(np.linalg.norm(link8_pos - anchor))
    if dist >= 0.005:
        raise AssertionError(f"panda_link8 independent-FK position {link8_pos} is {dist * 1000:.3f} mm from the {tuple(anchor)} anchor (expected < 5 mm)")

    ctx["link8_pos"] = link8_pos


def check_joint_values(ctx: dict) -> None:
    """8. Default joint angles for the 7 Franka arm joints, 2 fingers, 6 UR10 joints."""
    model = ctx["model"]
    joint_labels = ctx["joint_labels"]
    joint_q = model.joint_q.numpy()
    q_start = model.joint_q_start.numpy()

    checks = (
        list(zip(scene.PANDA_JOINT_LABELS, scene.PANDA_DEFAULT_Q))
        + list(zip(scene.PANDA_FINGER_LABELS, scene.PANDA_FINGER_DEFAULT_Q))
        + list(zip(scene.UR10_JOINT_LABELS, scene.UR10_DEFAULT_Q))
    )
    for label, default in checks:
        j = scene.joint_index(joint_labels, label)
        coord = int(q_start[j])
        value = float(joint_q[coord])
        err = abs(value - default)
        if err >= 1e-7:
            raise AssertionError(f"{label}: joint_q={value!r} default={default!r} error={err:.3e} (atol 1e-7)")


def check_belt(ctx: dict) -> None:
    """9. Belt AABB centre/half-extents/planarity."""
    body_q = ctx["body_q"]
    info = ctx["info"]
    idx = np.asarray(info.belt_bodies, dtype=np.int32)
    xyz = body_q[idx, :3].astype(np.float64)

    center = 0.5 * (xyz.min(axis=0) + xyz.max(axis=0))
    half_extent = 0.5 * (xyz.max(axis=0) - xyz.min(axis=0))

    center_err = np.linalg.norm(center - np.asarray(scene.BELT_CENTER))
    if center_err >= 0.003:
        raise AssertionError(f"belt AABB centre {center} is {center_err * 1000:.3f} mm from BELT_CENTER (expected < 3 mm)")

    dx = abs(half_extent[0] - scene.BELT_SEMI_AXIS_X)
    dy = abs(half_extent[1] - scene.BELT_SEMI_AXIS_Y)
    if dx >= 0.005 or dy >= 0.005:
        raise AssertionError(
            f"belt AABB half-extents {half_extent[:2]} vs expected "
            f"({scene.BELT_SEMI_AXIS_X}, {scene.BELT_SEMI_AXIS_Y}) (dx={dx * 1000:.3f} mm, dy={dy * 1000:.3f} mm, tol 5 mm)"
        )

    max_abs_z = float(np.max(np.abs(xyz[:, 2])))
    if max_abs_z >= 0.001:
        raise AssertionError(f"belt max |z| = {max_abs_z * 1000:.3f} mm (expected < 1 mm)")

    ctx["belt_center"] = center
    ctx["belt_half_extent"] = half_extent


def check_floor_table(ctx: dict) -> None:
    """10. Ground height and the safety tabletop-collision box top face."""
    model = ctx["model"]
    info = ctx["info"]

    if not (info.ground_height < -0.6):
        raise AssertionError(f"ground_height={info.ground_height} is not < -0.6")
    diff = abs(info.ground_height - float(info.table_visual_aabb[0][2]))
    if diff > 1e-6:
        raise AssertionError(
            f"ground_height ({info.ground_height}) != table_visual_aabb min z "
            f"({info.table_visual_aabb[0][2]}), diff={diff:.3e}"
        )

    shape_transform = model.shape_transform.numpy()
    shape_scale = model.shape_scale.numpy()
    idx = info.tabletop_collision_shape
    top_z = float(shape_transform[idx][2]) + float(shape_scale[idx][2])
    if abs(top_z - (-0.02858)) > 1e-6:
        raise AssertionError(f"tabletop_collision top face z={top_z:.8f}, expected -0.02858000 (atol 1e-6)")

    table_max_z = float(info.table_visual_aabb[1][2])
    warn_diff = abs(table_max_z - (-0.02858))
    if warn_diff > 0.005:
        print(
            f"[WARN] table visual AABB max z ({table_max_z:.5f}) differs from -0.02858 by "
            f"{warn_diff * 1000:.2f} mm (> 5 mm)"
        )


def check_counts(ctx: dict) -> None:
    """11. Body/shape counts and label-set membership."""
    info = ctx["info"]
    body_labels = ctx["body_labels"]
    label_set = set(str(l) for l in body_labels)

    if len(info.belt_bodies) != 48:
        raise AssertionError(f"len(info.belt_bodies)={len(info.belt_bodies)}, expected 48")
    if len(info.gripper_pad_bodies) != 2:
        raise AssertionError(f"len(info.gripper_pad_bodies)={len(info.gripper_pad_bodies)}, expected 2")

    panda_labels = {f"panda_arm/panda_link{i}" for i in range(9)}
    missing = panda_labels - label_set
    if missing:
        raise AssertionError(f"missing panda_arm link labels: {sorted(missing)}")

    hand_labels = {f"panda_hand/{l}" for l in PANDA_HAND_LINKS}
    missing = hand_labels - label_set
    if missing:
        raise AssertionError(f"missing panda_hand link labels: {sorted(missing)}")

    ur10_labels = {f"{UR10_LABEL_PREFIX}{l}" for l in UR10_LINKS}
    missing = ur10_labels - label_set
    if missing:
        raise AssertionError(f"missing ur10 link labels: {sorted(missing)}")


def print_pose_table(ctx: dict) -> None:
    body_q = ctx["body_q"]
    body_labels = ctx["body_labels"]
    info = ctx["info"]
    rows = [
        ("panda_arm/panda_link0", scene.body_index(body_labels, "panda_arm/panda_link0")),
        ("panda_arm/panda_link8", ctx["link8"]),
        ("panda_hand/panda_hand", scene.body_index(body_labels, "panda_hand/panda_hand")),
        (scene.UR10_BASE_LABEL, scene.body_index(body_labels, scene.UR10_BASE_LABEL)),
        (scene.UR10_WRIST3_LABEL, ctx["wrist3"]),
        (f"{ctx['gripper_root_label']} (2f85 root)", ctx["gripper_root"]),
    ]
    for i, b in enumerate(info.gripper_pad_bodies):
        rows.append((f"{body_labels[b]} (pad {i})", b))

    print("\n[POSE TABLE] world poses after eval_fk at the default configuration")
    print(f"{'body':<48}{'x':>9}{'y':>9}{'z':>9}   {'roll':>9}{'pitch':>9}{'yaw':>9}   (deg)")
    for label, idx in rows:
        m = body_mat4(body_q, idx)
        rpy = rpy_deg_from_mat3(m[:3, :3])
        p = m[:3, 3]
        print(f"{label:<48}{p[0]:>9.5f}{p[1]:>9.5f}{p[2]:>9.5f}   {rpy[0]:>9.3f}{rpy[1]:>9.3f}{rpy[2]:>9.3f}")
    print()


CHECKS = [
    ("1. Build scene, finalize model, run eval_fk", check_build),
    ("2. Rotation convention (board rpy == Rz.Ry.Rx)", check_convention),
    ("3. Scene weld/default constants vs this script's yaml transcription", check_scene_constants),
    ("4. Static shape poses (board + holder collision boxes)", check_static_shapes),
    ("5. Body poses (panda_link0, /ur10/base_link, panda_hand, 2f85 root)", check_body_poses),
    ("6. Gripper pad geometry relative to wrist_3_link", check_gripper_geometry),
    ("7. Independent numpy FK (panda_link8, ur10 wrist_3 via the USD->URDF frame constant)", check_independent_fk),
    ("8. Default joint values (7 Franka + 2 fingers + 6 UR10)", check_joint_values),
    ("9. Belt AABB centre/half-extents/planarity", check_belt),
    ("10. Floor height + tabletop_collision top face", check_floor_table),
    ("11. Body-label counts (belt, pads, panda/hand/ur10 links)", check_counts),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="Warp device to use (e.g. 'cpu'); default is Warp's default device.")
    args = parser.parse_args()
    if args.device:
        wp.set_device(args.device)

    ctx: dict = {}
    for name, fn in CHECKS:
        try:
            fn(ctx)
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            return 1
        print(f"[PASS] {name}")

    print(f"[INFO] discovered 2f85 root label: {ctx['gripper_root_label']!r}")
    print(
        f"[INFO] panda_link8 independent-FK position: "
        f"({ctx['link8_pos'][0]:.5f}, {ctx['link8_pos'][1]:.5f}, {ctx['link8_pos'][2]:.5f})"
    )
    print(
        f"[INFO] belt AABB centre {ctx['belt_center']}, half-extents {ctx['belt_half_extent']}"
    )
    print(
        f"[INFO] gripper pads: |d.x_w3|={ctx['pad_dx_mm']:.3f} mm, "
        f"|d.y_w3|={ctx['pad_dy_mm']:.3f} mm, mean pad proj on z_w3={ctx['pad_proj_z_mm']:.2f} mm"
    )

    print_pose_table(ctx)
    print("ALL POSE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
