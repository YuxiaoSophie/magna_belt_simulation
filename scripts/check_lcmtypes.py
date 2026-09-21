#!/usr/bin/env python3
"""Check the vendored lcmtypes/ packages are wire-compatible with magna's generated modules.

For each of the 9 vendored LCM types: compares the packed fingerprint against magna's own
generated module (bazel-bin output, read-only), then round-trips a non-default instance
through encode()/decode() and checks field equality. Skips the fingerprint half (not the
round trip) if the magna reference tree is not present.

Run:
    uv run python scripts/check_lcmtypes.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # This script lives under scripts/; make the repo root importable regardless of CWD.
    sys.path.insert(0, str(REPO_ROOT))

from dairlib import lcmt_object_state, lcmt_robot_input, lcmt_robot_output
from drake import (
    lcmt_schunk_wsg_command,
    lcmt_schunk_wsg_status,
    lcmt_viewer_geometry_data,
    lcmt_viewer_link_data,
)
from robotiq import lcmt_robotiq_command, lcmt_robotiq_status

# Parent "lcmtypes" dir of each magna-generated package (inserted into sys.path so a nested
# `import drake` inside a loaded magna module resolves, if it isn't already cached).
MAGNA_LCMTYPES_DIRS = {
    "dairlib": Path("/home/hienbui/git/magna/bazel-bin/external/dairlib+/lcmtypes"),
    "drake": Path("/home/hienbui/git/magna/bazel-bin/external/drake+/lcmtypes"),
    "robotiq": Path("/home/hienbui/git/magna/bazel-bin/external/robotiq-driver+/lcmtypes"),
}
MAGNA_AVAILABLE = all(d.is_dir() for d in MAGNA_LCMTYPES_DIRS.values())
for _dir in MAGNA_LCMTYPES_DIRS.values():
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _load_magna_class(package: str, type_name: str):
    """Load magna's generated ``type_name`` class under a distinct module alias."""
    module_path = MAGNA_LCMTYPES_DIRS[package] / package / f"{type_name}.py"
    alias = f"_magna_ref_{package}_{type_name}"
    spec = importlib.util.spec_from_file_location(alias, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return getattr(module, type_name)


def _check_fingerprint(package: str, type_name: str, repo_cls) -> None:
    if not MAGNA_AVAILABLE:
        print("[SKIP] fingerprint vs magna (not found)")
        return
    ref_cls = _load_magna_class(package, type_name)
    _require(
        repo_cls._get_packed_fingerprint() == ref_cls._get_packed_fingerprint(),
        f"{type_name}: fingerprint {repo_cls._get_packed_fingerprint()!r} != "
        f"magna's {ref_cls._get_packed_fingerprint()!r}",
    )


# Shared probe geometry, reused standalone and nested inside lcmt_viewer_link_data. Values are
# small integers so the float32 geometry fields round-trip exactly.
def _build_geom():
    geom = lcmt_viewer_geometry_data()
    geom.type = lcmt_viewer_geometry_data.MESH
    geom.position = [1.0, 2.0, 3.0]
    geom.quaternion = [1.0, 0.0, 0.0, 0.0]
    geom.color = [1.0, 0.0, 0.0, 1.0]
    geom.string_data = "round_belt::round_belt"
    geom.float_data = [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2]
    geom.num_float_data = len(geom.float_data)
    return geom


def _require_geom_equal(note: str, got, want) -> None:
    _require(got.type == want.type, f"{note}: type {got.type} != {want.type}")
    _require(list(got.position) == want.position, f"{note}: position {got.position} != "
             f"{want.position}")
    _require(list(got.quaternion) == want.quaternion, f"{note}: quaternion {got.quaternion} != "
             f"{want.quaternion}")
    _require(list(got.color) == want.color, f"{note}: color {got.color} != {want.color}")
    _require(got.string_data == want.string_data, f"{note}: string_data {got.string_data!r} != "
             f"{want.string_data!r}")
    _require(got.num_float_data == want.num_float_data, f"{note}: num_float_data "
             f"{got.num_float_data} != {want.num_float_data}")
    _require(list(got.float_data) == want.float_data, f"{note}: float_data {got.float_data} != "
             f"{want.float_data}")


def check_robot_input() -> None:
    _check_fingerprint("dairlib", "lcmt_robot_input", lcmt_robot_input)
    msg = lcmt_robot_input()
    msg.utime = 123456789
    msg.effort_names = ["left", "right"]
    msg.efforts = [1.5, -2.5]
    msg.num_efforts = len(msg.efforts)
    decoded = lcmt_robot_input.decode(msg.encode())
    _require(decoded.utime == msg.utime, "utime mismatch")
    _require(decoded.num_efforts == msg.num_efforts, "num_efforts mismatch")
    _require(list(decoded.effort_names) == msg.effort_names, "effort_names mismatch")
    _require(list(decoded.efforts) == msg.efforts, "efforts mismatch")
    print("[PASS] lcmt_robot_input")


def check_robot_output() -> None:
    _check_fingerprint("dairlib", "lcmt_robot_output", lcmt_robot_output)
    msg = lcmt_robot_output()
    msg.utime = 987654321
    msg.position_names = ["p0", "p1"]
    msg.position = [0.1, 0.2]
    msg.num_positions = len(msg.position)
    msg.velocity_names = ["v0", "v1"]
    msg.velocity = [0.3, 0.4]
    msg.num_velocities = len(msg.velocity)
    msg.effort_names = ["e0", "e1"]
    msg.effort = [0.5, 0.6]
    msg.num_efforts = len(msg.effort)
    msg.imu_accel = [1.0, 2.0, 3.0]
    decoded = lcmt_robot_output.decode(msg.encode())
    _require(decoded.utime == msg.utime, "utime mismatch")
    _require(list(decoded.position_names) == msg.position_names, "position_names mismatch")
    _require(list(decoded.position) == msg.position, "position mismatch")
    _require(list(decoded.velocity_names) == msg.velocity_names, "velocity_names mismatch")
    _require(list(decoded.velocity) == msg.velocity, "velocity mismatch")
    _require(list(decoded.effort_names) == msg.effort_names, "effort_names mismatch")
    _require(list(decoded.effort) == msg.effort, "effort mismatch")
    _require(list(decoded.imu_accel) == msg.imu_accel, "imu_accel mismatch")
    print("[PASS] lcmt_robot_output")


def check_object_state() -> None:
    _check_fingerprint("dairlib", "lcmt_object_state", lcmt_object_state)
    msg = lcmt_object_state()
    msg.utime = 5000
    msg.object_name = "nist_board"
    msg.position_names = ["small_round_pulley_joint", "large_round_pulley_joint"]
    msg.position = [0.25, -1.5]
    msg.num_positions = len(msg.position)
    msg.velocity_names = ["small_round_pulley_jointdot", "large_round_pulley_jointdot"]
    msg.velocity = [0.5, -0.125]
    msg.num_velocities = len(msg.velocity)
    decoded = lcmt_object_state.decode(msg.encode())
    _require(decoded.utime == msg.utime, "utime mismatch")
    _require(decoded.object_name == msg.object_name, "object_name mismatch")
    _require(decoded.num_positions == msg.num_positions, "num_positions mismatch")
    _require(decoded.num_velocities == msg.num_velocities, "num_velocities mismatch")
    _require(list(decoded.position_names) == msg.position_names, "position_names mismatch")
    _require(list(decoded.position) == msg.position, "position mismatch")
    _require(list(decoded.velocity_names) == msg.velocity_names, "velocity_names mismatch")
    _require(list(decoded.velocity) == msg.velocity, "velocity mismatch")
    print("[PASS] lcmt_object_state")


def check_schunk_wsg_status() -> None:
    _check_fingerprint("drake", "lcmt_schunk_wsg_status", lcmt_schunk_wsg_status)
    msg = lcmt_schunk_wsg_status()
    msg.utime = 111
    msg.actual_position_mm = 12.5
    msg.actual_force = 3.25
    msg.actual_speed_mm_per_s = -1.5
    decoded = lcmt_schunk_wsg_status.decode(msg.encode())
    _require(decoded.utime == msg.utime, "utime mismatch")
    _require(decoded.actual_position_mm == msg.actual_position_mm, "actual_position_mm mismatch")
    _require(decoded.actual_force == msg.actual_force, "actual_force mismatch")
    _require(decoded.actual_speed_mm_per_s == msg.actual_speed_mm_per_s,
             "actual_speed_mm_per_s mismatch")
    print("[PASS] lcmt_schunk_wsg_status")


def check_schunk_wsg_command() -> None:
    _check_fingerprint("drake", "lcmt_schunk_wsg_command", lcmt_schunk_wsg_command)
    msg = lcmt_schunk_wsg_command()
    msg.utime = 123
    msg.target_position_mm = 40.0
    msg.force = 10.0
    decoded = lcmt_schunk_wsg_command.decode(msg.encode())
    _require(decoded.utime == msg.utime, "utime mismatch")
    _require(decoded.target_position_mm == msg.target_position_mm, "target_position_mm mismatch")
    _require(decoded.force == msg.force, "force mismatch")
    print("[PASS] lcmt_schunk_wsg_command")


def check_viewer_geometry_data() -> None:
    _check_fingerprint("drake", "lcmt_viewer_geometry_data", lcmt_viewer_geometry_data)
    msg = _build_geom()
    decoded = lcmt_viewer_geometry_data.decode(msg.encode())
    _require_geom_equal("lcmt_viewer_geometry_data", decoded, msg)
    print("[PASS] lcmt_viewer_geometry_data")


def check_viewer_link_data() -> None:
    _check_fingerprint("drake", "lcmt_viewer_link_data", lcmt_viewer_link_data)
    msg = lcmt_viewer_link_data()
    msg.name = "round_belt"
    msg.robot_num = 1
    msg.geom = [_build_geom()]
    msg.num_geom = len(msg.geom)
    decoded = lcmt_viewer_link_data.decode(msg.encode())
    _require(decoded.name == msg.name, "name mismatch")
    _require(decoded.robot_num == msg.robot_num, "robot_num mismatch")
    _require(decoded.num_geom == msg.num_geom, "num_geom mismatch")
    _require(len(decoded.geom) == 1, f"geom count {len(decoded.geom)} != 1")
    _require_geom_equal("lcmt_viewer_link_data.geom[0]", decoded.geom[0], msg.geom[0])
    print("[PASS] lcmt_viewer_link_data")


def check_robotiq_command() -> None:
    _check_fingerprint("robotiq", "lcmt_robotiq_command", lcmt_robotiq_command)
    msg = lcmt_robotiq_command()
    msg.utime = 42
    msg.speed = 255
    msg.force = 0
    msg.position = 128
    decoded = lcmt_robotiq_command.decode(msg.encode())
    _require(decoded.utime == msg.utime, "utime mismatch")
    _require(decoded.speed == msg.speed, "speed mismatch")
    _require(decoded.force == msg.force, "force mismatch")
    _require(decoded.position == msg.position, "position mismatch")
    print("[PASS] lcmt_robotiq_command")


def check_robotiq_status() -> None:
    _check_fingerprint("robotiq", "lcmt_robotiq_status", lcmt_robotiq_status)
    msg = lcmt_robotiq_status()
    msg.utime = 42
    msg.activation_status = True
    msg.gripper_mode = False
    msg.goto_status = True
    msg.position = 200
    msg.speed = 50
    msg.force = 10
    decoded = lcmt_robotiq_status.decode(msg.encode())
    _require(decoded.utime == msg.utime, "utime mismatch")
    _require(decoded.activation_status == msg.activation_status, "activation_status mismatch")
    _require(decoded.gripper_mode == msg.gripper_mode, "gripper_mode mismatch")
    _require(decoded.goto_status == msg.goto_status, "goto_status mismatch")
    _require(decoded.position == msg.position, "position mismatch")
    _require(decoded.speed == msg.speed, "speed mismatch")
    _require(decoded.force == msg.force, "force mismatch")
    print("[PASS] lcmt_robotiq_status")


CHECKS = [
    check_robot_input, check_robot_output, check_object_state, check_schunk_wsg_status,
    check_schunk_wsg_command, check_viewer_geometry_data, check_viewer_link_data,
    check_robotiq_command, check_robotiq_status,
]


def main() -> int:
    for fn in CHECKS:
        try:
            fn()
        except AssertionError as exc:
            print(f"[FAIL] {fn.__name__}: {exc}", file=sys.stderr)
            return 1
    print("ALL LCMTYPE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
