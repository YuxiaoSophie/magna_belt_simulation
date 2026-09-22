"""Full-state snapshot / restore of a running :class:`LcmBeltTaskSimulation`.

:func:`capture` copies the physics state (``joint_q/joint_qd/body_q/body_qd``), the control
buffers (``joint_target_q``, ``joint_f``) and the host-side command state (step and time, the
Panda hand slew, the Robotiq bytes, the belt trigger) to host arrays; :func:`restore` writes
them back into the same or a freshly built sim, rebaselines the coupled solver's history with
``solver.reset(state_0, flags=0)`` and drops the CUDA graph (re-captured on the next step).
:func:`save` / :func:`load` round-trip a snapshot through one ``.npz``.
The LCM bridge (latest received commands) is external input and is not part of a snapshot.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from task_common import REPO_ROOT

SCHEMA_VERSION = 1
DEFAULT_START_STATE_DIR = REPO_ROOT / "data" / "lcs" / "start_states"
RESET_TOLERANCE = 1e-6
_NO_BYTE = -1  # robotiq_position_byte None (no command received yet) in the .npz

_ARRAYS = ("joint_q", "joint_qd", "body_q", "body_qd", "joint_target_q", "joint_f",
           "hand_target_q", "hand_goal_q", "hand_ramp_q")
_STATE_ARRAYS = ("joint_q", "joint_qd", "body_q", "body_qd")


@dataclass
class SimSnapshot:
    """Host copies of everything a control step reads; see the module docstring."""

    joint_q: np.ndarray
    joint_qd: np.ndarray
    body_q: np.ndarray
    body_qd: np.ndarray
    joint_target_q: np.ndarray
    joint_f: np.ndarray
    step_index: int
    sim_time: float
    hand_target_mm: float
    hand_target_q: np.ndarray
    hand_goal_q: np.ndarray
    hand_ramp_q: np.ndarray
    robotiq_position_byte: int | None
    robotiq_force_byte: int
    belt_placed: bool
    belt_anchor_body: int
    meta: dict = field(default_factory=dict)


def capture(sim, label: str, notes: str = "") -> SimSnapshot:
    """Snapshot ``sim`` between control steps (host copies only)."""
    full = sim.recorder_meta()
    meta = {
        "schema": SCHEMA_VERSION, "label": str(label),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "body_labels": full["body_labels"], "joint_labels": full["joint_labels"],
        "joint_coord_count": full["joint_coord_count"],
        "joint_dof_count": full["joint_dof_count"],
        "scene_directives": full.get("scene_directives"),
        "lcm_sim_params": full.get("lcm_sim_params"),
        "git_commit": full["git_commit"], "git_dirty": full["git_dirty"],
        "notes": str(notes),
    }
    state = sim.state_0
    hand_target_q, hand_goal_q, hand_ramp_q = sim.hand_ramp_state()
    return SimSnapshot(
        joint_q=state.joint_q.numpy().astype(np.float32),
        joint_qd=state.joint_qd.numpy().astype(np.float32),
        body_q=state.body_q.numpy().astype(np.float32),
        body_qd=state.body_qd.numpy().astype(np.float32),
        joint_target_q=sim.control.joint_target_q.numpy().copy(),
        joint_f=sim.control.joint_f.numpy().copy(),
        step_index=int(sim.step_index), sim_time=float(sim.sim_time),
        hand_target_mm=float(sim.hand_target_mm),
        hand_target_q=hand_target_q, hand_goal_q=hand_goal_q, hand_ramp_q=hand_ramp_q,
        robotiq_position_byte=(None if sim.robotiq_position_byte is None
                               else int(sim.robotiq_position_byte)),
        robotiq_force_byte=int(sim.robotiq_force_byte),
        belt_placed=bool(sim.belt_placed),
        belt_anchor_body=int(getattr(sim, "belt_anchor_body", -1)),
        meta=meta,
    )


def _first_mismatch(want: list, got: list) -> int | None:
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b:
            return i
    return None if len(want) == len(got) else min(len(want), len(got))


def _validate(sim, snap: SimSnapshot) -> None:
    """Raise ``ValueError`` if ``snap`` does not fit ``sim``'s model; touches nothing."""
    model = sim.model
    for key, labels in (("body_labels", model.body_label), ("joint_labels", model.joint_label)):
        have = [str(v) for v in labels]
        want = list(snap.meta.get(key, []))
        index = _first_mismatch(want, have)
        if index is not None:
            got_want = want[index] if index < len(want) else "<missing>"
            got_have = have[index] if index < len(have) else "<missing>"
            raise ValueError(f"snapshot {key}[{index}] {got_want!r} != model {got_have!r} "
                             f"({len(want)} vs {len(have)} entries)")
    n_coords, n_dofs = int(model.joint_coord_count), int(model.joint_dof_count)
    n_bodies = len(model.body_label)
    for key, have in (("joint_coord_count", n_coords), ("joint_dof_count", n_dofs)):
        if snap.meta.get(key) != have:
            raise ValueError(f"snapshot {key} {snap.meta.get(key)} != model {have}")
    shapes = {
        "joint_q": (n_coords,), "joint_qd": (n_dofs,), "body_q": (n_bodies, 7),
        "body_qd": (n_bodies, 6), "joint_target_q": tuple(sim.control.joint_target_q.shape),
        "joint_f": tuple(sim.control.joint_f.shape), "hand_target_q": (2,),
        "hand_goal_q": (2,), "hand_ramp_q": (2,),
    }
    for key, shape in shapes.items():
        value = np.asarray(getattr(snap, key))
        if value.shape != shape:
            raise ValueError(f"snapshot {key} shape {value.shape} != sim {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"snapshot {key} has non-finite values")


def restore(sim, snap: SimSnapshot) -> None:
    """Write ``snap`` into ``sim`` (validated first: a refusal changes nothing)."""
    _validate(sim, snap)
    model, control = sim.model, sim.control
    arrays = {key: np.asarray(getattr(snap, key), dtype=np.float32) for key in _STATE_ARRAYS}
    model.joint_q.assign(arrays["joint_q"])
    model.joint_qd.assign(arrays["joint_qd"])
    for state in (sim.state_0, sim.state_1):
        for key, value in arrays.items():
            getattr(state, key).assign(value)
    control.joint_target_q.assign(np.asarray(snap.joint_target_q, dtype=np.float32))
    control.joint_f.assign(np.asarray(snap.joint_f, dtype=np.float32))

    sim.step_index = int(snap.step_index)
    sim.frame_id = int(snap.step_index)
    sim.sim_time = float(snap.sim_time)
    sim.hand_target_mm = float(snap.hand_target_mm)
    sim.robotiq_position_byte = snap.robotiq_position_byte
    sim.robotiq_force_byte = int(snap.robotiq_force_byte)
    sim.belt_placed = bool(snap.belt_placed)
    sim.belt_anchor_body = int(snap.belt_anchor_body)

    sim.solver.reset(sim.state_0, flags=0)
    for key, value in arrays.items():
        moved = float(np.abs(getattr(sim.state_0, key).numpy() - value).max())
        if moved > RESET_TOLERANCE:
            raise RuntimeError(f"solver.reset moved state_0.{key} by {moved:.3g}")
    sim.physics_graph = None
    sim.sync_after_restore(snap.hand_target_q, snap.hand_goal_q, snap.hand_ramp_q)


def save(snap: SimSnapshot, path: Path) -> Path:
    """Write ``snap`` to ``path`` (``.npz``; the meta as a JSON string)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    byte = _NO_BYTE if snap.robotiq_position_byte is None else snap.robotiq_position_byte
    np.savez(
        path, **{key: np.asarray(getattr(snap, key)) for key in _ARRAYS},
        step_index=np.int64(snap.step_index), sim_time=np.float64(snap.sim_time),
        hand_target_mm=np.float64(snap.hand_target_mm),
        robotiq_position_byte=np.int64(byte),
        robotiq_force_byte=np.int64(snap.robotiq_force_byte),
        belt_placed=np.bool_(snap.belt_placed), belt_anchor_body=np.int64(snap.belt_anchor_body),
        meta_json=np.asarray(json.dumps(snap.meta)),
    )
    # np.savez appends .npz to a suffix-less name.
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def load(path: Path) -> SimSnapshot:
    """Read a :func:`save` file."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        if meta.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"{path}: snapshot schema {meta.get('schema')} != {SCHEMA_VERSION}")
        byte = int(data["robotiq_position_byte"])
        return SimSnapshot(
            **{key: data[key].copy() for key in _ARRAYS},
            step_index=int(data["step_index"]), sim_time=float(data["sim_time"]),
            hand_target_mm=float(data["hand_target_mm"]),
            robotiq_position_byte=None if byte == _NO_BYTE else byte,
            robotiq_force_byte=int(data["robotiq_force_byte"]),
            belt_placed=bool(data["belt_placed"]),
            belt_anchor_body=int(data["belt_anchor_body"]),
            meta=meta,
        )
