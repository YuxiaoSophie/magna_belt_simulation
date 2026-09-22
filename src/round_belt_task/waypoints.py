"""magna's round-belt pick-and-place waypoints, read from its controller yaml, in WORLD frame.

The yaml (``MAGNA_PARAMS_SIM_YAML``, read-only, never copied) lists ``pre_mpc_motion``: a live
``start`` waypoint, a ``!CompiledTrajectorySegment`` whose ``waypoints`` are the pick and place,
and a live ``place_11``. Poses are ``finger_tip`` (Franka) / ``tracking_frame`` (UR) in the board
frame when ``target_poses_in_board_frame``: ``X_WF = X_WB * X_BF``, quaternions ``wxyz``. Live
waypoints use the yaml's ``task_board_position`` / ``task_board_orientation`` (RPY, ``Rz.Ry.Rx``;
``assembly_controller.cc`` ``ConvertOSCTargetPoseFromBoardToWorld``); compiled-segment waypoints
use the ``board`` weld of the compiler's default scene (``round-belt-scene.dmd.yaml``), as
``compile_bimanual_trajectory.py`` ``targets_in_world`` does (== this scene's ``X_W_BOARD``).
Gripper commands follow ``assembly_controller.cc``: Franka ``mm = cmd * 2000``; Robotiq
``byte = clamp(cmd / 0.04) * 255``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from round_belt_task.arm_kinematics import (
    mat3_to_quat_xyzw,
    mat4,
    quat_xyzw_to_mat3,
    rpy_to_mat3,
)
from round_belt_task.constants import SCENE_DIRECTIVES
from utils.directives import parse_directives

MAGNA_PARAMS_SIM_YAML = Path(
    "/home/hienbui/git/magna/systems/parameters/round_belt_controller_params_sim.yaml"
)
SEQUENCE_KEY = "pre_mpc_motion"
FRANKA_MM_PER_CMD = 2000.0
UR_FULL_CLOSE_CMD = 0.04
COMPILED_TAG = "CompiledTrajectorySegment"


def _planning_board_pose() -> np.ndarray:
    # magna's round-belt-scene.dmd.yaml board weld (the compiler's default X_WB) == this scene's.
    pose = parse_directives(SCENE_DIRECTIVES).weld("board::board").X_PC
    return mat4(rpy_to_mat3(np.radians(pose.rpy_deg)), pose.translation)


X_W_BOARD_PLANNING = _planning_board_pose()


class _ParamsLoader(yaml.SafeLoader):
    """SafeLoader that keeps Drake-style ``!Tag`` mappings."""


def _construct_tagged_mapping(loader, suffix, node):
    return dict(loader.construct_mapping(node, deep=True), __tag__=suffix)


_ParamsLoader.add_multi_constructor("!", _construct_tagged_mapping)


def load_params(path: Path = MAGNA_PARAMS_SIM_YAML) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=_ParamsLoader)


@dataclass
class Waypoint:
    """One waypoint in world frame; a ``None`` command means "keep the previous one"."""

    label: str
    franka_pos: np.ndarray
    franka_quat_xyzw: np.ndarray
    ur_pos: np.ndarray | None
    ur_quat_xyzw: np.ndarray | None
    franka_gripper_mm: float | None
    ur_gripper_byte: int | None
    dwell_s: float

    def franka_mat(self) -> np.ndarray:
        return mat4(quat_xyzw_to_mat3(self.franka_quat_xyzw), self.franka_pos)

    def ur_mat(self) -> np.ndarray | None:
        if self.ur_pos is None:
            return None
        return mat4(quat_xyzw_to_mat3(self.ur_quat_xyzw), self.ur_pos)


def board_pose(path: Path = MAGNA_PARAMS_SIM_YAML, params: dict | None = None) -> np.ndarray:
    """``X_WB`` (4x4) from the yaml's ``task_board_position`` / ``task_board_orientation``."""
    params = load_params(path) if params is None else params
    return mat4(rpy_to_mat3(params["task_board_orientation"]), params["task_board_position"])


def franka_gripper_mm(cmd: float | None) -> float | None:
    return None if cmd is None else float(cmd) * FRANKA_MM_PER_CMD


def ur_gripper_byte(cmd: float | None) -> int | None:
    if cmd is None:
        return None
    return int(np.clip(float(cmd) / UR_FULL_CLOSE_CMD, 0.0, 1.0) * 255)


def _world_pose(X_WB: np.ndarray, pos, quat_wxyz) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    X_BF = mat4(quat_xyzw_to_mat3([x, y, z, w]), pos)
    X_WF = X_WB @ X_BF
    return X_WF[:3, 3].copy(), mat3_to_quat_xyzw(X_WF[:3, :3])


def _flatten(sequence: list) -> list[tuple[dict, bool]]:
    """``(row, compiled)`` for every waypoint, compiled segments expanded in place."""
    rows = []
    for entry in sequence:
        if isinstance(entry, dict) and entry.get("__tag__") == COMPILED_TAG:
            rows.extend((row, True) for row in entry["waypoints"])
        else:
            rows.append((entry, False))
    return rows


def load_pre_mpc_segment(path: Path = MAGNA_PARAMS_SIM_YAML, first: str = "pre_pick_0",
                         last: str = "place_3") -> list[Waypoint]:
    """The ``pre_mpc_motion`` waypoints ``first..last`` (inclusive) in world frame."""
    params = load_params(path)
    rows = _flatten(params[SEQUENCE_KEY])
    labels = [row.get("label") for row, _ in rows]
    for name in (first, last):
        if name not in labels:
            raise KeyError(f"{path}: {SEQUENCE_KEY} has no waypoint {name!r} (have {labels})")
    i, j = labels.index(first), labels.index(last)
    if j < i:
        raise ValueError(f"{path}: {last!r} comes before {first!r}")
    in_board = bool(params.get("target_poses_in_board_frame"))
    X_WB_live = board_pose(params=params)
    waypoints = []
    for row, compiled in rows[i:j + 1]:
        X_WB = (X_W_BOARD_PLANNING if compiled else X_WB_live) if in_board else np.eye(4)
        franka_pos, franka_quat = _world_pose(X_WB, row["franka_position"],
                                              row["franka_orientation"])
        ur_pos = ur_quat = None
        if row.get("ur_position") is not None:
            ur_pos, ur_quat = _world_pose(X_WB, row["ur_position"], row["ur_orientation"])
        waypoints.append(Waypoint(
            label=str(row["label"]), franka_pos=franka_pos, franka_quat_xyzw=franka_quat,
            ur_pos=ur_pos, ur_quat_xyzw=ur_quat,
            franka_gripper_mm=franka_gripper_mm(row.get("franka_gripper_pos_command")),
            ur_gripper_byte=ur_gripper_byte(row.get("ur_gripper_pos_command")),
            dwell_s=float(row.get("dwell_seconds", 0.0)),
        ))
    return waypoints
