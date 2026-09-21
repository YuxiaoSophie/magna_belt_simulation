#!/usr/bin/env python3
"""Headless, solver-free check of the scene-directives loader's contract.

Covers parsing (YAML 1.1 numeric coercion, `!Rpy`, strict keys), weld/frame composition
onto a real articulation, static models + AABB collection + `add_directives` includes, the
error guards, custom-directive plumbing and a smoke load of the real round-belt scene, plus
the real board's colours, two guards exercised on edited copies of the real scene (directive
order, an MJCF root-body weld child) and the ALOHA finger geoms baked into `2f85.xml` (with an
edited copy of that XML that must fail). Builds synthetic directives files and XML copies at
runtime under `tempfile.TemporaryDirectory()`; nothing is committed under `assets/`. No
solver stepping, no viewer.

Run:
    uv run python scripts/check_scene_directives.py
    uv run python scripts/check_scene_directives.py --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import warp as wp
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # This script lives under scripts/; make the repo root importable regardless of CWD.
    sys.path.insert(0, str(REPO_ROOT))

import newton
import newton.utils

import round_belt
import round_belt_task
from round_belt_task.directives import EXTENSION_DIRECTIVES
from utils.directives import DirectiveContext, Pose, load_directives, parse_directives

# ----------------------------------------------------------------------------
# Small numpy/Warp transform helpers, re-implemented locally (kept independent
# of scripts/check_round_belt_task_poses.py on purpose).
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


def mat4_from_xyz_rpy_deg(xyz, rpy_deg) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rpy_to_mat3(tuple(math.radians(a) for a in rpy_deg))
    m[:3, 3] = xyz
    return m


def tf_mat4(tf) -> np.ndarray:
    """(px py pz qx qy qz qw) -- a wp.transform or a body_q/shape_transform row -> 4x4."""
    row = [float(v) for v in tf]
    m = np.eye(4)
    m[:3, :3] = np.array(wp.quat_to_matrix(wp.quat(*row[3:7])), dtype=np.float64).reshape(3, 3)
    m[:3, 3] = row[:3]
    return m


def assert_close_mat4(name: str, got: np.ndarray, expected: np.ndarray, atol: float) -> None:
    diff = float(np.max(np.abs(got - expected)))
    if diff > atol:
        raise AssertionError(
            f"{name}: max abs diff {diff:.3e} > atol {atol:.1e}\nGOT=\n{got}\nEXPECTED=\n{expected}"
        )


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


class Skipped(Exception):
    """Raised by a check to signal [SKIP] rather than [FAIL] (network-dependent checks)."""


# ----------------------------------------------------------------------------
# Fixture assets used by the synthetic scenes (read-only; nothing here is edited).
# ----------------------------------------------------------------------------

PANDA_ARM_URDF = REPO_ROOT / "assets/common/franka/urdf/panda_arm.urdf"
PANDA_HAND_URDF = REPO_ROOT / "assets/common/franka/urdf/panda_hand_with_long_fingers.urdf"
HOLDER_URDF = REPO_ROOT / "assets/common/belt_chain_holder/belt_chain_holder.urdf"
ROBOTIQ_MJCF = REPO_ROOT / "2f85.xml"
BOARD_URDF = REPO_ROOT / "assets/round_belt_task/round_belt_task_board.urdf"
SCENE_YAML = REPO_ROOT / "assets/round_belt_task/round_belt_scene.yaml"
UR10_2F85_YAML = REPO_ROOT / "assets/common/directives/ur10_2f85.yaml"


@contextlib.contextmanager
def _tmpdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _load(path: Path, directives=None):
    """``load_directives`` onto a fresh Z-up builder (kept on the result as ``.builder``)."""
    return load_directives(
        newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81), path,
        directives=directives or {}, visual_cfg=round_belt.make_visual_cfg(),
        collision_cfg=round_belt.make_robust_table_collision_cfg(visible=False),
    )


def _build_scene(path: Path):
    """The real task's ``build_scene`` on a directives file (the real one or an edited copy)."""
    return round_belt_task.build_scene(round_belt_task.make_builder(), path)


def _expect_error(build: Callable[[], object], note: str, *needles: str,
                  error: type[Exception] = ValueError) -> None:
    """Assert ``build()`` raises ``error`` (not another type, not nothing) naming every needle."""
    try:
        build()
    except error as exc:
        message = str(exc)
    except Exception as exc:  # noqa: BLE001 - re-raised as an AssertionError, on purpose
        raise AssertionError(
            f"{note}: expected {error.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    else:
        raise AssertionError(f"{note}: expected {error.__name__}, got no exception")
    missing = [needle for needle in needles if needle not in message]
    _require(not missing, f"{note}: {missing} not in the {error.__name__} message: {message}")


@functools.lru_cache(maxsize=1)
def _real_scene():
    """``(builder, SceneInfo)`` of the real round-belt scene, built once and shared."""
    builder = round_belt_task.make_builder()
    return builder, round_belt_task.build_scene(builder)


# ----------------------------------------------------------------------------
# Edited copies of the real scene. Relative paths are made absolute so the copies load
# from a temp dir; entries are edited as whole list items so comments cannot mislead.
# ----------------------------------------------------------------------------

_PATH_KEY = re.compile(r"(\bfile:[ \t]*)([^\s,}#]+)")
Edit = Callable[[list[str]], None]


def _entries(text: str) -> tuple[str, list[str]]:
    """Split a directives file into its header and one chunk per top-level list entry."""
    head, sep, body = re.split(r"(?m)^(directives:\n)", text, maxsplit=1)
    chunks = re.split(r"(?m)^(?=  - )", body)
    return head + sep + chunks[0], chunks[1:]


def _pick(chunks: list[str], predicate: Callable[[str], bool], what: str) -> int:
    hits = [i for i, chunk in enumerate(chunks) if predicate(chunk)]
    _require(len(hits) == 1, f"fixture: expected exactly one {what} entry, found {len(hits)}")
    return hits[0]


def _absolutize(text: str, base: Path, overrides: dict[Path, Path]) -> str:
    def swap(match: re.Match) -> str:
        raw = match.group(2)
        if raw.startswith("newton_asset://") or Path(raw).is_absolute():
            return match.group(0)
        target = (base / raw).resolve()
        return match.group(1) + str(overrides.get(target, target))

    return _PATH_KEY.sub(swap, text)


def _scene_copy(out_dir: Path, edit_scene: Edit | None = None,
                edit_include: Edit | None = None) -> Path:
    """Write ``round_belt_scene.yaml`` + ``ur10_2f85.yaml`` into ``out_dir``, edited."""
    head, entries = _entries(UR10_2F85_YAML.read_text())
    if edit_include:
        edit_include(entries)
    include = _write(out_dir / "ur10_2f85.yaml",
                     _absolutize(head + "".join(entries), UR10_2F85_YAML.parent, {}))
    head, entries = _entries(SCENE_YAML.read_text())
    if edit_scene:
        edit_scene(entries)
    return _write(out_dir / "scene.yaml", _absolutize(
        head + "".join(entries), SCENE_YAML.parent, {UR10_2F85_YAML.resolve(): include}
    ))


# ----------------------------------------------------------------------------
# Checks, registered in order by @check(title).
# ----------------------------------------------------------------------------

CHECKS: list[tuple[str, Callable[[], None]]] = []


def check(title: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def register(fn: Callable[[], None]) -> Callable[[], None]:
        CHECKS.append((title, fn))
        return fn

    return register


@check("1. Parse + !Rpy + YAML-1.1 numeric coercion")
def check_parse_numeric() -> None:
    with _tmpdir() as tmp:
        ok_yaml = _write(tmp / "ok.yaml", """
directives:
  - add_model: {name: dummy, file: fake.urdf, static: true}
  - add_weld:
      parent: world
      child: dummy
      X_PC:
        translation: [2e-1, 0.0, 0.0]
        rotation: !Rpy {deg: [0.0, 0.0, 9.0e+01]}
""")
        weld = parse_directives(ok_yaml).weld("dummy")
        expected_pose = Pose((0.2, 0.0, 0.0), (0.0, 0.0, 90.0))
        _require(weld.X_PC == expected_pose, f"parsed X_PC {weld.X_PC} != expected {expected_pose}")

        got_rot = tf_mat4(weld.X_PC.to_transform())[:3, :3]
        expected_rot = rpy_to_mat3((0.0, 0.0, math.pi / 2))  # yaw only: a pure Rz(90 deg)
        _require(
            np.allclose(got_rot, expected_rot, atol=1e-6, rtol=0.0),
            f"Pose.to_transform() rotation != independent Rz(90 deg); "
            f"max abs error = {np.abs(got_rot - expected_rot).max():.3e}",
        )

        bad_yaml = _write(tmp / "bad.yaml", """
directives:
  - add_model: {name: dummy, file: fake.urdf, static: true}
  - add_weld:
      parent: world
      child: dummy
      X_PC:
        translation: [abc, 0, 0]
""")
        _expect_error(functools.partial(parse_directives, bad_yaml),
                      "translation: [abc, 0, 0]", "translation[0] = 'abc' is not a number")


@check("2. Strict key validation (unknown/mismatched keys, duplicate names)")
def check_strict_keys() -> None:
    cases = [  # (what, directives entries, needles the ValueError must name)
        ("(a) colour: (unknown key) on a static add_model",
         "  - add_model: {name: d, file: fake.urdf, static: true, colour: [1, 0, 0]}\n",
         ["colour"]),
        ("(b) color: on an articulated (non-static) urdf model",
         "  - add_model: {name: d, file: fake.urdf, color: [1, 0, 0]}\n", ["color"]),
        ("(c) gravity_compensation: on a static: true model",
         "  - add_model: {name: d, file: fake.urdf, static: true, gravity_compensation: true}\n",
         ["gravity_compensation"]),
        ("(d) two models with the same name",
         "  - add_model: {name: d, file: fake.urdf, static: true}\n"
         "  - add_model: {name: d, file: fake2.urdf, static: true}\n",
         ["duplicate model name", "'d'"]),
    ]
    with _tmpdir() as tmp:
        for i, (note, entries, needles) in enumerate(cases):
            path = _write(tmp / f"{i}.yaml", "directives:\n" + entries)
            _expect_error(functools.partial(parse_directives, path), note, *needles)


@check("3. Weld + frame composition on a real (Franka) articulation")
def check_weld_frame_composition() -> None:
    with _tmpdir() as tmp:
        scene = _load(_write(tmp / "scene.yaml", f"""
directives:
  - add_model:
      name: panda_arm
      file: {PANDA_ARM_URDF}
      default_joint_positions:
        panda_joint2: [-0.5]
  - add_weld:
      parent: world
      child: panda_arm::panda_link0
      X_PC:
        translation: [0.1, 0.2, 0.3]
        rotation: !Rpy {{deg: [10.0, 20.0, 30.0]}}
  - add_frame:
      name: F
      X_PF:
        base_frame: panda_arm::panda_link8
        translation: [0.0, 0.05, 0.1]
        rotation: !Rpy {{deg: [0.0, 90.0, 0.0]}}
  - add_model:
      name: panda_hand
      file: {PANDA_HAND_URDF}
  - add_weld:
      parent: F
      child: panda_hand::panda_hand
      X_PC:
        translation: [0.0, 0.0, 0.0]
        rotation: !Rpy {{deg: [0.0, 0.0, -45.0]}}
"""))
        model = scene.builder.finalize()

        # Seed model.joint_q with the authored default_joint_positions before eval_fk.
        joint_q = model.joint_q.numpy().copy()
        q_start = model.joint_q_start.numpy()
        defaults = scene.default_joint_positions()
        for joint_idx, values in defaults:
            coord = int(q_start[joint_idx])
            for k, v in enumerate(values):
                joint_q[coord + k] = v
        model.joint_q.assign(joint_q)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()
        body_labels = list(model.body_label)
        joint_labels = list(model.joint_label)

        link0 = scene.body("panda_arm::panda_link0")
        expected_link0 = mat4_from_xyz_rpy_deg((0.1, 0.2, 0.3), (10.0, 20.0, 30.0))
        assert_close_mat4("panda_arm/panda_link0", tf_mat4(body_q[link0]), expected_link0, 1e-6)

        link8 = scene.body("panda_arm::panda_link8")
        link8_by_label = [
            i for i, lbl in enumerate(body_labels) if str(lbl) == "panda_arm/panda_link8"
        ]
        _require(link8_by_label == [link8],
                 f"scene.body('panda_arm::panda_link8') = {link8}, but the label lookup gives "
                 f"{link8_by_label}")

        frame_body, X_PF = scene.frame_transform("F")
        _require(frame_body == link8,
                 f"scene.frame_transform('F') body = {frame_body}, expected link8 = {link8}")
        expected_X_PF = mat4_from_xyz_rpy_deg((0.0, 0.05, 0.1), (0.0, 90.0, 0.0))
        assert_close_mat4("frame F's X_PF", tf_mat4(X_PF), expected_X_PF, atol=1e-6)

        hand = scene.body("panda_hand::panda_hand")
        expected_hand = tf_mat4(body_q[link8]) @ expected_X_PF @ mat4_from_xyz_rpy_deg(
            (0.0, 0.0, 0.0), (0.0, 0.0, -45.0)
        )
        assert_close_mat4("panda_hand/panda_hand", tf_mat4(body_q[hand]), expected_hand, 1e-5)

        joint4 = scene.joint("panda_arm", "panda_joint4")
        joint4_by_label = [
            i for i, lbl in enumerate(joint_labels) if str(lbl) == "panda_arm/panda_joint4"
        ]
        _require(joint4_by_label == [joint4],
                 f"scene.joint('panda_arm', 'panda_joint4') = {joint4}, but the label lookup "
                 f"gives {joint4_by_label}")

        joint2 = scene.joint("panda_arm", "panda_joint2")
        _require((joint2, [-0.5]) in defaults,
                 f"scene.default_joint_positions() = {defaults}, expected ({joint2}, [-0.5]) in it")


@check("4. Static model + AABB collection + add_directives include")
def check_static_aabb_include() -> None:
    with _tmpdir() as tmp:
        # A path that only resolves relative to sub/ (not the main yaml's directory): as
        # many '../' as it takes to walk from sub/ back to the real repo asset.
        rel_holder = os.path.relpath(HOLDER_URDF, start=tmp / "sub")
        _write(tmp / "sub/holder.yaml", f"""
directives:
  - add_model:
      name: belt_chain_holder
      file: {rel_holder}
      static: true
  - add_weld:
      parent: world
      child: belt_chain_holder::belt_chain_holder_first_half
      X_PC:
        translation: [0.4736603358808432, 0.3520562100563749, -0.02858]
        rotation: !Rpy {{deg: [0.0, 0.0, 90.0]}}
""")
        scene = _load(_write(tmp / "main.yaml",
                             "directives:\n  - add_directives: {file: sub/holder.yaml}\n"))
        record = scene.models["belt_chain_holder"]
        label = "belt_chain_holder/belt_chain_holder_first_half/collision0"
        _require(label in record.shape_labels,
                 f"shape_labels does not contain {label!r}; have {sorted(record.shape_labels)}")

        model = scene.builder.finalize()
        got = tf_mat4(model.shape_transform.numpy()[record.shape_labels[label]])
        expected = mat4_from_xyz_rpy_deg(
            (0.4736603358808432, 0.3520562100563749, -0.02858), (0.0, 0.0, 90.0)
        )
        assert_close_mat4(label, got, expected, atol=1e-6)

        _require(any(key.startswith("belt_chain_holder/") for key in scene.aabbs),
                 f"no 'belt_chain_holder/' key in scene.aabbs; have {sorted(scene.aabbs)[:5]}...")


@check("5. Load-time guards (6 cases, each a ValueError)")
def check_guards() -> None:
    arm = f"directives:\n  - add_model: {{name: panda_arm, file: {PANDA_ARM_URDF}}}\n"
    cases = [  # (what, directives file, custom directives, needles the ValueError must name)
        ("(a) weld child is a non-root link",
         arm + "  - add_weld: {parent: world, child: panda_arm::panda_link3}\n",
         None, ["panda_link3", "ROOT link"]),
        ("(b) weld whose child model never appears",
         "directives:\n  - add_weld: {parent: world, child: ghost::link}\n",
         None, ["ghost::link"]),
        ("(c) parent model appears after the child",
         arm + "  - add_weld: {parent: panda_hand::panda_hand, child: panda_arm::panda_link0}\n"
         f"  - add_model: {{name: panda_hand, file: {PANDA_HAND_URDF}}}\n",
         None, ["panda_hand", "loaded before"]),
        ("(d) unknown custom directive kind",
         "directives:\n  - totally_unknown_directive: {}\n",
         None, ["unknown directive", "totally_unknown_directive"]),
        ("(e) directives= mapping tries to override a core directive name",
         "directives: []\n", {"add_model": lambda ctx_, params: None},
         ["add_model", "core directive"]),
        ("(f) static: true welded to a non-world parent",
         arm + "  - add_weld: {parent: world, child: panda_arm::panda_link0}\n"
         f"  - add_model: {{name: holder, file: {HOLDER_URDF}, static: true}}\n"
         "  - add_weld: {parent: panda_arm::panda_link0, "
         "child: holder::belt_chain_holder_first_half}\n",
         None, ["must be welded to world"]),
    ]
    with _tmpdir() as tmp:
        for i, (note, text, custom, needles) in enumerate(cases):
            path = _write(tmp / f"{i}.yaml", text)
            _expect_error(functools.partial(_load, path, custom), note, *needles)


@check("6. Custom directive plumbing (params/extras/ctx.source/ctx.resolve_path)")
def check_custom_directive() -> None:
    with _tmpdir() as tmp:
        _write(tmp / "x.urdf", "<robot name='x'/>")
        captured: dict = {}

        def mark(directive_ctx: DirectiveContext, params) -> None:
            captured["resolved"] = directive_ctx.resolve_path("x.urdf")
            directive_ctx.scene.extras[params["name"]] = {
                "seen": params["value"], "source": directive_ctx.source.name,
            }

        path = _write(tmp / "scene.yaml", "directives:\n  - mark: {name: mymark, value: 42}\n")
        scene = _load(path, {"mark": mark})

        extra = scene.extras.get("mymark")
        expected_extra = {"seen": 42, "source": tmp.name}
        _require(extra == expected_extra,
                 f"extras['mymark'] = {extra!r}, expected {expected_extra!r}")
        expected_path = (tmp / "x.urdf").resolve()
        _require(captured.get("resolved") == expected_path,
                 f"ctx.resolve_path('x.urdf') = {captured.get('resolved')!r}, "
                 f"expected {expected_path!r}")


# Independently transcribed from assets/round_belt_task/round_belt_scene.yaml's
# `add_weld world -> board::board`. Do NOT import this from round_belt_task.constants:
# that would compare the module against itself.
BOARD_WELD_TRANSLATION = (0.64483928, -0.19718233, 0.01076393)


@check("7. Real scene smoke (round_belt_task.make_builder/build_scene)")
def check_real_scene_smoke() -> None:
    _, info = _real_scene()
    for attr, want in (("static_shapes", 58), ("robot_bodies", 36), ("belt_bodies", 48),
                       ("gripper_pad_bodies", 2), ("gripper_pad_shapes", 2),
                       ("pulley_bodies", 2), ("pulley_joints", 2), ("pulley_shapes", 12)):
        got = len(getattr(info, attr))
        _require(got == want, f"len(info.{attr}) = {got}, expected {want}")

    builder = _real_scene()[0]
    pulley_bodies = set(info.pulley_bodies)
    on_pulleys = [s for s in range(builder.shape_count)
                  if int(builder.shape_body[s]) in pulley_bodies]
    _require(sorted(info.pulley_shapes) == on_pulleys,
             f"info.pulley_shapes {info.pulley_shapes} != shapes on the pulley bodies {on_pulleys}")
    pads = [str(builder.body_label[b]).rsplit("/", 1)[-1] for b in info.gripper_pad_bodies]
    _require(pads == ["right_pad", "left_pad"],
             f"info.gripper_pad_bodies leaves = {pads}, expected ['right_pad', 'left_pad']")
    for body, shape, pad in zip(info.gripper_pad_bodies, info.gripper_pad_shapes, pads):
        label = str(builder.shape_label[shape])
        side = pad.split("_")[0]
        _require(int(builder.shape_body[shape]) == body
                 and label.endswith(f"/{pad}/{side}_aloha_finger_collision"),
                 f"gripper_pad_shapes entry {shape} ({label!r}) is not the finger collider on "
                 f"body {body}")
    for attr in ("franka_finger_bodies", "franka_finger_shapes"):
        got = len(getattr(info, attr))
        _require(got == 2, f"len(info.{attr}) = {got}, expected 2")
    fingers = {str(builder.body_label[b]).rsplit("/", 1)[-1] for b in info.franka_finger_bodies}
    _require(fingers == {"panda_leftfinger", "panda_rightfinger"},
             f"info.franka_finger_bodies leaves = {sorted(fingers)}, expected "
             "['panda_leftfinger', 'panda_rightfinger']")
    for body, shape in zip(info.franka_finger_bodies, info.franka_finger_shapes):
        label = str(builder.shape_label[shape])
        _require(int(builder.shape_body[shape]) == body and label.endswith("/collision0"),
                 f"franka_finger_shapes entry {shape} ({label!r}) is not collision0 on body {body}")
    extension = sorted(EXTENSION_DIRECTIVES)
    expected_extension = [
        "add_cropped_point_cloud", "add_ground_plane", "add_rgbd_camera", "add_rod_ellipse",
        "add_tabletop_collision",
    ]
    _require(extension == expected_extension,
             f"EXTENSION_DIRECTIVES = {extension}, expected {expected_extension}")

    for label in (
        "tabletop_collision", "board/board/collision0",
        "belt_chain_holder/belt_chain_holder_first_half/collision0",
    ):
        _require(label in info.static_shape_labels, f"{label!r} not in info.static_shape_labels")

    board_weld = parse_directives(round_belt_task.SCENE_DIRECTIVES).weld("board::board")
    translation = board_weld.X_PC.translation
    _require(translation == BOARD_WELD_TRANSLATION,
             f"board weld translation {translation} != {BOARD_WELD_TRANSLATION} (this check's "
             "own transcription of the yaml)")


@check("8. newton_asset:// resolution")
def check_newton_asset_resolution() -> None:
    """Skips only if the asset download itself fails."""
    with _tmpdir() as tmp:
        directive_file = parse_directives(_write(
            tmp / "scene.yaml",
            "directives:\n"
            "  - add_model:\n"
            "      name: ur10\n"
            "      file: newton_asset://universal_robots_ur10/usd/ur10_instanceable.usda\n",
        ))
        model = directive_file.model("ur10")
        source = next(src for entry, src in directive_file.entries if entry is model)
        try:
            # download_asset's git/network/cache failures surface as these three.
            newton.utils.download_asset("universal_robots_ur10")
        except (OSError, RuntimeError, ImportError) as exc:
            raise Skipped(f"downloading universal_robots_ur10 failed: {exc}") from exc
        resolved = directive_file.resolve_path(model.file, source)
        _require(resolved.exists(), f"resolve_path returned {resolved}, which does not exist")


@check("9. Unsigned-exponent / dot-less literals are coerced")
def check_unsigned_exponent() -> None:
    for literal in ("8.95207485e01", "2e4"):
        _require(isinstance(yaml.safe_load(f"v: {literal}")["v"], str),
                 f"PyYAML no longer reads {literal} as a string; check is moot")
    with _tmpdir() as tmp:
        path = _write(
            tmp / "unsigned.yaml",
            "directives:\n"
            "  - add_model: {name: dummy, file: fake.urdf, static: true}\n"
            "  - add_weld:\n"
            "      parent: world\n"
            "      child: dummy\n"
            "      X_PC:\n"
            "        translation: [2e4, 0.0, 0.0]\n"
            "        rotation: !Rpy {deg: [0.0, 0.0, 8.95207485e01]}\n",
        )
        try:
            pose = parse_directives(path).weld("dummy").X_PC
        except ValueError as exc:
            raise AssertionError(f"the loader rejected 8.95207485e01 / 2e4: {exc}") from exc
    expected = Pose((20000.0, 0.0, 0.0), (0.0, 0.0, 89.5207485))
    _require(pose == expected, f"parsed X_PC {pose} != expected {expected}")


# Independently transcribed from the board add_model in round_belt_scene.yaml (the colours,
# and the component_colors rule's max_span / near_local_xy / radius).
BOARD_WHITE, BOARD_BLACK = (0.8, 0.8, 0.8), (0.1, 0.1, 0.1)
# keep_visual_material markers (URDF materials): dark on the small pulley, white on the large.
MARKER_COLORS = {"small_round_pulley": (0.05, 0.05, 0.05), "large_round_pulley": (1.0, 1.0, 1.0)}
MARKER_VISUAL = "visual2"
MOUNT_MAX_SPAN, MOUNT_LOCAL_XY, MOUNT_RADIUS = 0.2, (0.3504, 0.1964), 0.05


@check("10. Board link_colors + component_colors in shape_color")
def check_board_colors() -> None:
    builder, _ = _real_scene()
    labels = [str(label or "") for label in builder.shape_label]

    def rgb(b: newton.ModelBuilder, shape: int) -> tuple[float, ...]:
        return tuple(round(float(c), 6) for c in b.shape_color[shape])

    def wrong(shapes: list[int], want: tuple[float, ...]) -> list[tuple[str, tuple]]:
        return [(labels[s], rgb(builder, s)) for s in shapes if rgb(builder, s) != want]

    for link, want in (("small_round_pulley", BOARD_WHITE), ("large_round_pulley", BOARD_BLACK)):
        marker = f"board/{link}/{MARKER_VISUAL}"
        shapes = [s for s, lbl in enumerate(labels)
                  if lbl.startswith(f"board/{link}/visual") and lbl != marker]
        if not shapes or wrong(shapes, want):
            raise AssertionError(f"link_colors[{link}] != {want}: {wrong(shapes, want) or 'none'}")
        markers = [s for s, lbl in enumerate(labels) if lbl == marker]
        _require(len(markers) == 1 and not wrong(markers, MARKER_COLORS[link]),
                 f"{marker}: {[(labels[s], rgb(builder, s)) for s in markers]}, expected one "
                 f"shape coloured {MARKER_COLORS[link]}")

    # The mount rule, restated: a split board component narrower than max_span whose XY-bbox
    # centre is within radius of the mount point is black; every other component is white.
    plate, rest, panel = [], [], []
    for s, lbl in enumerate(labels):
        if not lbl.startswith("board/board/visual"):
            continue
        vertices = np.asarray(builder.shape_source[s].vertices, dtype=np.float64)
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        span = max(upper[0] - lower[0], upper[1] - lower[1])
        centre = 0.5 * (lower + upper)
        near = math.hypot(centre[0] - MOUNT_LOCAL_XY[0], centre[1] - MOUNT_LOCAL_XY[1])
        (plate if span < MOUNT_MAX_SPAN and near <= MOUNT_RADIUS else rest).append(s)
        if span >= MOUNT_MAX_SPAN:
            panel.append(s)
    _require(plate and panel, f"board split into no mount parts ({plate}) or no panel ({panel})")
    bad = wrong(plate, BOARD_BLACK) + wrong(rest, BOARD_WHITE)
    _require(not bad, f"component_colors: mount parts must be black, the rest white: {bad}")

    # On this board link_colors repeats the URDF's own <material> colours (0.8 / 0.1), so the
    # real scene cannot show it ran; a colour nothing else produces can.
    probe = (0.3, 0.6, 0.9)
    with _tmpdir() as tmp:
        probe_builder = _load(_write(
            tmp / "board.yaml",
            "directives:\n"
            f"  - add_model:\n      name: board\n      file: {BOARD_URDF}\n      static: true\n"
            f"      link_colors: {{small_round_pulley: [{probe[0]}, {probe[1]}, {probe[2]}]}}\n"
            "  - add_weld: {parent: world, child: board::board}\n",
        )).builder
    probe_labels = [str(label or "") for label in probe_builder.shape_label]
    painted = sorted(lbl for s, lbl in enumerate(probe_labels) if rgb(probe_builder, s) == probe)
    small = sorted(lbl for lbl in probe_labels if lbl.startswith("board/small_round_pulley/visual"))
    _require(small and painted == small,
             f"link_colors {probe} painted {painted}, expected exactly {small}")


@check("11. Ordering guard: tabletop after the last static model raises")
def check_ordering_guard() -> None:
    def tabletop_after_statics(entries: list[str]) -> None:
        tabletop = entries.pop(_pick(
            entries, lambda c: c.startswith("  - add_tabletop_collision:"), "tabletop"
        ))
        cameras = _pick(
            entries, lambda c: c.startswith("  - add_directives:") and "zed_cameras.yaml" in c,
            "ZED cameras add_directives",
        )
        entries.insert(cameras + 1, tabletop)

    with _tmpdir() as tmp:
        control = _build_scene(_scene_copy(tmp / "control"))
        _require(len(control.static_shapes) == 58,
                 f"fixture: unedited copy has {len(control.static_shapes)} static shapes, "
                 "expected 58")
        moved = _scene_copy(tmp / "moved", edit_scene=tabletop_after_statics)
        _expect_error(functools.partial(_build_scene, moved),
                      "add_tabletop_collision after the ZED cameras", "static shape range",
                      error=RuntimeError)


@check("12. MJCF '::root' weld child fails the pose guard (7 mm)")
def check_mjcf_root_child_pose_guard() -> None:
    bare = re.compile(r"child: robotiq_2f85\b(?!::)")

    def name_root_body(entries: list[str]) -> None:
        weld = _pick(
            entries, lambda c: c.startswith("  - add_weld:") and bool(bare.search(c)),
            "gripper add_weld",
        )
        entries[weld] = bare.sub("child: robotiq_2f85::base_mount", entries[weld], count=1)

    with _tmpdir() as tmp:
        path = _scene_copy(tmp, edit_include=name_root_body)
        _expect_error(functools.partial(_build_scene, path), "robotiq_2f85::base_mount weld child",
                      "robotiq_2f85::base_mount", "off by 7.000 mm", "bare model name")


# Drake Robotiq SDF finger poses in the MJCF import frame; not read from 2f85.xml on purpose.
DRAKE_FINGER_X, DRAKE_FINGER_Z = 0.047285310862444, 0.1148045193817614
# pad body -> (finger mesh, pose). Left/right cross over.
EXPECTED_FINGERS = {
    "right_pad": ("left_finger",
                  mat4_from_xyz_rpy_deg((DRAKE_FINGER_X, 0.0, DRAKE_FINGER_Z), (0.0, 0.0, 0.0))),
    "left_pad": ("right_finger",
                 mat4_from_xyz_rpy_deg((-DRAKE_FINGER_X, 0.0, DRAKE_FINGER_Z), (0.0, 0.0, 180.0))),
}
FINGER_RGB = (0.1, 0.1, 0.1)
FINGER_OBJ_DIR = REPO_ROOT / "assets/common/robotiq_2f85/fingers"


def _world_aabb(tf: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """World AABB of ``vertices`` under ``tf``, as (lo, hi) concatenated."""
    world = vertices @ tf[:3, :3].T + tf[:3, 3]
    return np.concatenate([world.min(axis=0), world.max(axis=0)])


def _obj_vertices(path: Path) -> np.ndarray:
    """OBJ vertices, parsed without Newton's loader."""
    rows = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    return np.asarray(rows, dtype=np.float64)


def _finger_geom_problems(xml_path: Path) -> list[str]:
    """Ways ``xml_path``'s pads differ from one finger visual + collider at the Drake pose."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.add_mjcf(str(xml_path), enable_self_collisions=False)
    flags = newton.ShapeFlags
    leaf = {b: str(label).rsplit("/", 1)[-1] for b, label in enumerate(builder.body_label)}
    problems: list[str] = []
    for side in ("right", "left"):
        silicone = [b for b, name in leaf.items() if name == f"{side}_silicone_pad"]
        on_silicone = [str(builder.shape_label[s]) for s in range(builder.shape_count)
                       if silicone and int(builder.shape_body[s]) == silicone[0]]
        if len(silicone) != 1 or on_silicone:
            problems.append(f"{side}_silicone_pad: body count {len(silicone)} (want 1), "
                            f"shapes {on_silicone} (want none)")
    for pad, (finger, expected) in EXPECTED_FINGERS.items():
        bodies = [b for b, name in leaf.items() if name == pad]
        if len(bodies) != 1:
            problems.append(f"{pad}: {len(bodies)} bodies with that name, expected 1")
            continue
        body = bodies[0]
        shapes = [s for s in range(builder.shape_count) if int(builder.shape_body[s]) == body]
        names = {s: str(builder.shape_label[s]).rsplit("/", 1)[-1] for s in shapes}
        side = pad.split("_")[0]
        want = {f"{side}_aloha_finger_visual", f"{side}_aloha_finger_collision"}
        if sorted(names.values()) != sorted(want):
            problems.append(f"{pad}: shapes {sorted(names.values())}, expected exactly "
                            f"{sorted(want)}")
            continue
        body_tf = tf_mat4(builder.body_q[body])
        for s, name in names.items():
            f = int(builder.shape_flags[s])
            collider = name.endswith("_collision")
            has_shape = bool(f & flags.COLLIDE_SHAPES)
            has_particle = bool(f & flags.COLLIDE_PARTICLES)
            if collider:
                group = int(builder.shape_collision_group[s])
                if (group, has_shape, has_particle) != (1, True, True):
                    problems.append(f"{name}: collision group {group}, shape/particle collision "
                                    f"{has_shape}/{has_particle}; want 1, True/True")
            else:
                rgb = tuple(round(float(c), 6) for c in builder.shape_color[s])
                visible = bool(f & flags.VISIBLE)
                if has_shape or has_particle or not visible or rgb != FINGER_RGB:
                    problems.append(f"{name}: visible {visible}, collides {has_shape}/"
                                    f"{has_particle}, rgb {rgb}; want True, False/False, "
                                    f"{FINGER_RGB}")
            got = body_tf @ tf_mat4(builder.shape_transform[s])
            err = float(np.max(np.abs(got - expected)))
            if err > 1.0e-6:
                problems.append(f"{name}: frame off Drake's {finger} frame by {err:.3e} (> 1e-6)")
            # Catches a swapped mesh or lost scale="1 1 1" that keeps the frame right.
            vertices = np.asarray(builder.shape_source[s].vertices, dtype=np.float64)
            vertices = vertices * np.asarray(builder.shape_scale[s], dtype=np.float64)
            drake = _obj_vertices(FINGER_OBJ_DIR / f"{finger}.obj")
            err = float(np.max(np.abs(_world_aabb(got, vertices) - _world_aabb(expected, drake))))
            if err > 1.0e-5:
                problems.append(f"{name}: mesh AABB off Drake's {finger} by {err:.3e} m (> 1e-5)")
    return problems


def _xml_copy(out_dir: Path, *swaps: tuple[str, str]) -> Path:
    """Copy of 2f85.xml with an absolute meshdir and each ``(old, new)`` applied once."""
    text = ROBOTIQ_MJCF.read_text()
    meshdir = re.search(r'meshdir="([^"]+)"', text)
    _require(meshdir is not None, "fixture: 2f85.xml has no meshdir")
    text = text.replace(meshdir.group(0), f'meshdir="{ROBOTIQ_MJCF.parent / meshdir.group(1)}"')
    for old, new in swaps:
        _require(text.count(old) == 1, f"fixture: {old!r} occurs {text.count(old)}x in 2f85.xml")
        text = text.replace(old, new)
    return _write(out_dir / "2f85.xml", text)


@check("13. 2f85.xml pad bodies carry exactly the ALOHA fingers, at the Drake pose")
def check_aloha_fingers_in_mjcf() -> None:
    problems = _finger_geom_problems(ROBOTIQ_MJCF)
    _require(not problems, "2f85.xml: " + "; ".join(problems))
    # Negative control: same-name mesh/pad wiring must fail.
    with _tmpdir() as tmp:
        crossed = _xml_copy(
            tmp, ('mesh name="left_finger"', 'mesh name="tmp_finger"'),
            ('mesh name="right_finger"', 'mesh name="left_finger"'),
            ('mesh name="tmp_finger"', 'mesh name="right_finger"'),
        )
        problems = _finger_geom_problems(crossed)
    _require(any("mesh AABB off Drake" in p for p in problems),
             f"a 2f85.xml copy with the finger meshes swapped passed: {problems}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default=None,
        help="Warp device to use (e.g. 'cpu'); default is Warp's default device.",
    )
    args = parser.parse_args()
    if args.device:
        wp.set_device(args.device)

    for name, fn in CHECKS:
        try:
            fn()
        except Skipped as exc:
            print(f"[SKIP] {name}: {exc}")
            continue
        except AssertionError as exc:
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            return 1
        print(f"[PASS] {name}")

    print("ALL DIRECTIVE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
