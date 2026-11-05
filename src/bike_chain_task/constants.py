"""Bike-chain scene constants, read from ``assets/bike_chain_task/bike_chain_scene.yaml``."""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from round_belt_task.constants import (  # noqa: F401  (scene-independent, reused as is)
    LCM_SIM_PARAMS,
    PANDA_FINGER_LABELS,
    PANDA_JOINT_LABELS,
    TABLE_VISUAL_LABEL,
    UR10_JOINT_LABELS,
)
from task_common import REPO_ROOT
from utils.directives import Directive, parse_directives

BIKE_CHAIN_ASSETS_DIR = REPO_ROOT / "assets" / "bike_chain_task"
# BIKE_CHAIN_SCENE: a scene variant, repo-relative or absolute
SCENE_DIRECTIVES = (REPO_ROOT / os.environ["BIKE_CHAIN_SCENE"] if os.environ.get("BIKE_CHAIN_SCENE")
                    else BIKE_CHAIN_ASSETS_DIR / "bike_chain_scene.yaml")

_D = parse_directives(SCENE_DIRECTIVES)


def _source(directive: Directive) -> Path:
    """The yaml that authored ``directive``; its relative paths resolve against it."""
    return next(source for entry, source in _D.entries if entry is directive)


def _joint_defaults(model: str, labels: list[str]) -> list[float]:
    """``model``'s ``default_joint_positions``, ordered by the leaf names of ``labels``."""
    positions = _D.model(model).default_joint_positions
    leaves = [label.rsplit("/", 1)[-1] for label in labels]
    if sorted(positions) != sorted(leaves) or any(len(v) != 1 for v in positions.values()):
        raise ValueError(
            f"{SCENE_DIRECTIVES}: {model} default_joint_positions {positions} must give "
            f"exactly one value for each of {leaves}"
        )
    return [positions[leaf][0] for leaf in leaves]


PANDA_DEFAULT_Q = _joint_defaults("panda_arm", PANDA_JOINT_LABELS)
PANDA_FINGER_DEFAULT_Q = _joint_defaults("panda_hand", PANDA_FINGER_LABELS)
UR10_DEFAULT_Q = _joint_defaults("ur10", UR10_JOINT_LABELS)

X_W_BOARD = _D.weld("board::board").X_PC.to_transform()
X_W_UR10 = _D.weld("ur10::base_link").X_PC.to_transform()

TABLE_TOP_Z = float(_D.custom("add_tabletop_collision").params["top_z"])

_CHAIN = _D.custom("add_roller_chain", "bike_chain").params
CHAIN_CENTER = tuple(float(v) for v in _CHAIN["center"])  # z = inner-link frame z
CHAIN_SEMI_AXES = (tuple(float(v) for v in _CHAIN["semi_axes"]) if "semi_axes" in _CHAIN
                   else None)
CHAIN_RADIUS = float(_CHAIN["radius"]) if "radius" in _CHAIN else None
CHAIN_NUM_LINKS = int(_CHAIN["num_links"])
CHAIN_PITCH = float(_CHAIN["pitch"])
# Holder tier top under the start (V13-02b fit: center z = tier + 0.5 mm lift + 1.25 mm plate).
HOLDER_TIER_Z = CHAIN_CENTER[2] - 0.0005 - 0.00125 if CHAIN_SEMI_AXES is not None else None

BOARD_URDF = _D.resolve_path(_D.model("board").file, _source(_D.model("board")))
SPROCKET_JOINT_LABELS = [
    "board/small_sprocket_joint", "board/large_sprocket_joint", "board/tensioner_sprocket_joint",
]
SPROCKET_TEETH = (16, 32, 16)
SPROCKET_PITCH_RADIUS_MM = tuple(
    CHAIN_PITCH / (2.0 * math.sin(math.pi / n)) * 1e3 for n in SPROCKET_TEETH)
_URDF = ET.parse(BOARD_URDF).getroot()
_JOINTS = {j.get("name"): j for j in _URDF.iter("joint")}
_LEAVES = [label.rsplit("/", 1)[-1] for label in SPROCKET_JOINT_LABELS]
SPROCKET_CENTERS_LOCAL = tuple(
    tuple(float(v) for v in _JOINTS[leaf].find("origin").get("xyz").split()) for leaf in _LEAVES)
_RIMS = {c.get("name"): c for c in _URDF.iter("collision")}
# Rim (tooth) mid-plane z in each sprocket's own frame; board frame = + SPROCKET_CENTERS_LOCAL z.
TOOTH_PLANE_Z_MM = tuple(
    float(_RIMS[leaf.removesuffix("_joint") + "_rim"].find("origin").get("xyz").split()[2]) * 1e3
    for leaf in _LEAVES)
