"""Chunked run recordings of the LCM simulation (:class:`RunRecorder`) and their reader.

A run directory ``<root>/<YYYYmmdd-HHMMSS>-<label>/`` holds ``meta.json``,
``chunks/chunk_NNNNN.npz``, ``events.jsonl`` and ``targets.jsonl`` (docs/lcm-simulation.md §10).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import queue
import re
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from task_common import REPO_ROOT

SCHEMA_VERSION = 1
DEFAULT_RECORDINGS_DIR = REPO_ROOT / "recordings"
DEFAULT_STATE_EVERY = 4
DEFAULT_CHUNK_STEPS = 2000
WRITER_JOIN_TIMEOUT = 30.0
BACKLOG_WARN_CHUNKS = 3

_ROW_FIELDS = (
    ("step", np.int64, ()),
    ("sim_time", np.float64, ()),
    ("wall_time", np.float64, ()),
    ("compute_ms", np.float32, ()),
    ("hand_target_mm", np.float32, ()),
    ("hand_stale", np.bool_, ()),
    ("franka_stale", np.bool_, ()),
    ("robotiq_cmd", np.uint8, (3,)),
    ("robotiq_cmd_valid", np.bool_, ()),
    ("robotiq_status", np.uint8, (3,)),
    ("robotiq_opening", np.float32, ()),
    ("lcm_rx", np.int32, ()),
)


def run_dir_name(label: str, now: datetime | None = None) -> str:
    """``YYYYmmdd-HHMMSS-<label>`` with the label reduced to ``[A-Za-z0-9_-]``."""
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", label) or "run"
    return (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S-") + clean


def file_digest(path: Path) -> dict:
    """``{path relative to REPO_ROOT, sha256}`` of a file (absolute path outside the repo)."""
    path = Path(path).resolve()
    try:
        name = str(path.relative_to(REPO_ROOT))
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git_state() -> tuple[str | None, bool | None]:
    """``(HEAD commit, dirty)`` of the repo, best effort."""
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                                 text=True, timeout=5.0, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return (None if commit is None else commit.strip(),
            None if status is None else bool(status.strip()))


class _Chunk:
    """One chunk's preallocated host arrays and fill counts."""

    def __init__(self, rows: int, frames: int, coords: int, dofs: int, efforts: int,
                 bodies: int) -> None:
        for name, dtype, shape in _ROW_FIELDS:
            setattr(self, name, np.zeros((rows, *shape), dtype=dtype))
        self.joint_q = np.zeros((rows, coords), dtype=np.float32)
        self.joint_qd = np.zeros((rows, dofs), dtype=np.float32)
        self.efforts = np.zeros((rows, efforts), dtype=np.float32)
        self.state_step = np.zeros(frames, dtype=np.int64)
        self.body_q = np.zeros((frames, bodies, 7), dtype=np.float32)
        self.rows = 0
        self.frames = 0

    def arrays(self) -> dict[str, np.ndarray]:
        n, m = self.rows, self.frames
        out = {name: getattr(self, name)[:n] for name, _, _ in _ROW_FIELDS}
        out.update(joint_q=self.joint_q[:n], joint_qd=self.joint_qd[:n],
                   efforts=self.efforts[:n], state_step=self.state_step[:m],
                   body_q=self.body_q[:m])
        return out


class RunRecorder:
    """Records one run: ``step()`` fills host buffers, a daemon thread does all file IO."""

    def __init__(self, root: Path, label: str, meta: dict, *, state_every: int,
                 chunk_steps: int, compress: bool = False) -> None:
        if state_every < 1 or chunk_steps < 1:
            raise ValueError(f"state_every {state_every}, chunk_steps {chunk_steps}: must be >= 1")
        self.path = self._make_run_dir(Path(root), run_dir_name(label))
        self.state_every = int(state_every)
        self.chunk_steps = int(chunk_steps)
        self.compress = compress
        self.failed = False
        self.closed = False
        self.chunks: list[dict] = []  # written chunks, appended by the writer thread
        self._chunk_index = 0
        self._backlog_warned = False
        self._signal_coords = np.asarray(meta["signal_coords"], dtype=np.int64)
        self._signal_dofs = np.asarray(meta["signal_dofs"], dtype=np.int64)
        max_frames = -(-self.chunk_steps // self.state_every) + 1
        self._shape = (self.chunk_steps, max_frames, len(self._signal_coords),
                       len(self._signal_dofs), len(meta["effort_layout"]),
                       len(meta["body_labels"]))
        self._buf = _Chunk(*self._shape)
        created = datetime.now().astimezone().isoformat(timespec="seconds")
        self.meta = {
            "schema": SCHEMA_VERSION, "created": created,
            "label": label, **meta, "state_every": self.state_every,
            "chunk_steps": self.chunk_steps, "finished": False, "reason": None,
            "num_steps": 0, "last_step": None, "chunks": [],
        }
        self._write_meta()
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="run-recorder", daemon=True)
        self._thread.start()

    @staticmethod
    def _make_run_dir(root: Path, name: str) -> Path:
        """``root/name``, or ``name-2``, ``name-3``... if a run started in the same second."""
        root.mkdir(parents=True, exist_ok=True)
        for n in range(1, 1000):
            path = root / (name if n == 1 else f"{name}-{n}")
            try:
                path.mkdir()
            except FileExistsError:
                continue
            (path / "chunks").mkdir()
            return path
        raise FileExistsError(f"{root / name}: too many runs with this name")

    @property
    def bytes_per_step(self) -> float:
        _, _, coords, dofs, efforts, bodies = self._shape
        row = 4 * (coords + dofs + efforts)
        row += sum(np.dtype(dtype).itemsize * int(np.prod(shape)) for _, dtype, shape in
                   _ROW_FIELDS)
        frame = 8 + 28 * bodies  # state_step i64 + body_q f32 (bodies, 7)
        return row + frame / self.state_every

    def step(self, *, step, sim_time, wall_time, joint_q, joint_qd, efforts, hand_target_mm,
             hand_stale, franka_stale, robotiq_cmd, robotiq_status, robotiq_opening, lcm_rx,
             compute_ms, body_q) -> None:
        """One row; ``efforts`` is a sequence of per-robot arrays laid end to end."""
        if self.failed:
            return
        buf = self._buf
        r = buf.rows
        buf.step[r] = step
        buf.sim_time[r] = sim_time
        buf.wall_time[r] = wall_time
        buf.compute_ms[r] = compute_ms
        np.take(joint_q, self._signal_coords, out=buf.joint_q[r], mode="clip")
        np.take(joint_qd, self._signal_dofs, out=buf.joint_qd[r], mode="clip")
        row, i = buf.efforts[r], 0
        for values in efforts:
            n = len(values)
            row[i:i + n] = values
            i += n
        buf.hand_target_mm[r] = hand_target_mm
        buf.hand_stale[r] = hand_stale
        buf.franka_stale[r] = franka_stale
        if robotiq_cmd is None:
            buf.robotiq_cmd_valid[r] = False
        else:
            buf.robotiq_cmd[r] = robotiq_cmd
            buf.robotiq_cmd_valid[r] = True
        buf.robotiq_status[r] = robotiq_status
        buf.robotiq_opening[r] = robotiq_opening
        buf.lcm_rx[r] = lcm_rx
        if step % self.state_every == 0:
            f = buf.frames
            buf.state_step[f] = step
            buf.body_q[f] = body_q
            buf.frames = f + 1
        buf.rows = r + 1
        if buf.rows == self.chunk_steps:
            self._flush()

    def _flush(self) -> None:
        buf = self._buf
        if buf.rows == 0:
            return
        self._queue.put_nowait(("chunk", (self._chunk_index, buf)))
        self._chunk_index += 1
        self._buf = _Chunk(*self._shape)
        backlog = self._chunk_index - len(self.chunks)
        if backlog > BACKLOG_WARN_CHUNKS and not self._backlog_warned:
            self._backlog_warned = True
            logger.warning(f"[RECORD] writer {backlog} chunks behind (slow disk?); "
                           "buffers grow until it catches up")
        elif backlog <= 1:
            self._backlog_warned = False

    def event(self, step: int, sim_time: float, kind: str, /, **data) -> None:
        self._put_line("event", step, sim_time, kind=kind, data=data)

    def target(self, step: int, sim_time: float, channel: str, payload: dict) -> None:
        self._put_line("target", step, sim_time, channel=channel, payload=payload)

    def _put_line(self, stream: str, step: int, sim_time: float, /, **fields) -> None:
        if not self.failed and not self.closed:
            record = {"step": int(step), "sim_time": float(sim_time), **fields}
            self._queue.put_nowait((stream, record))

    def close(self, *, finished: bool, reason: str) -> None:
        """Flush the partial chunk, stop the writer and write the final ``meta.json``."""
        if self.closed:
            return
        if not self.failed:
            self._flush()
        self.closed = True
        self._queue.put_nowait(("stop", None))
        self._thread.join(WRITER_JOIN_TIMEOUT)
        timed_out = self._thread.is_alive()
        if timed_out:
            # Unfinished: Recording.load then also globs the chunks written after this snapshot.
            logger.warning(f"[RECORD] writer did not finish within {WRITER_JOIN_TIMEOUT:g} s; "
                           "meta.json marked unfinished")
        chunks = list(self.chunks)
        written = sum(c["steps"] for c in chunks)
        self.meta.update(
            finished=bool(finished) and not timed_out, reason=reason, num_steps=written,
            last_step=chunks[-1]["first_step"] + chunks[-1]["steps"] - 1 if chunks else None,
            chunks=chunks,
        )
        if timed_out:
            self.meta["writer_timeout"] = True
        if self.failed:
            self.meta["recorder_failed"] = True
        self._write_meta()

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())

    def _write_meta(self) -> None:
        tmp = self.path / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=1) + "\n")
        tmp.replace(self.path / "meta.json")

    def _writer(self) -> None:
        try:
            with (open(self.path / "events.jsonl", "a") as events,
                  open(self.path / "targets.jsonl", "a") as targets):
                self._write_loop(events, targets)
        except Exception as exc:  # noqa: BLE001 - a recording failure must never stop the sim
            self.failed = True
            logger.opt(exception=exc).error(f"[RECORD] writer failed, recording off: {exc!r}")
            for tmp in (self.path / "chunks").glob("*.tmp"):
                with contextlib.suppress(OSError):
                    tmp.unlink()

    def _write_loop(self, events, targets) -> None:
        save = np.savez_compressed if self.compress else np.savez
        while True:
            kind, item = self._queue.get()
            if kind == "stop":
                return
            if kind == "chunk":
                index, buf = item
                name = f"chunk_{index:05d}.npz"
                tmp = self.path / "chunks" / (name + ".tmp")
                with open(tmp, "wb") as f:
                    save(f, **buf.arrays())
                tmp.replace(self.path / "chunks" / name)
                self.chunks.append({"file": f"chunks/{name}", "first_step": int(buf.step[0]),
                                    "steps": buf.rows, "state_frames": buf.frames})
            else:
                (events if kind == "event" else targets).write(json.dumps(item) + "\n")
            if self._queue.empty():
                events.flush()
                targets.flush()


@dataclass(frozen=True)
class Event:
    step: int
    sim_time: float
    kind: str
    data: dict


@dataclass(frozen=True)
class TargetMsg:
    step: int
    sim_time: float
    channel: str
    payload: dict


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a killed run can leave a torn last line
    return rows


class Recording:
    """A loaded run: whole-run arrays with random access by step, time or state frame."""

    def __init__(self, path: Path, meta: dict, arrays: dict[str, np.ndarray],
                 events: list[Event], targets: list[TargetMsg]) -> None:
        self.path = path
        self.meta = meta
        self.step = arrays.pop("step")
        self.sim_time = arrays.pop("sim_time")
        self.wall_time = arrays["wall_time"]
        self.compute_ms = arrays["compute_ms"]  # row i holds the compute time of step i - 1
        self.state_step = arrays.pop("state_step")
        self.body_q = arrays.pop("body_q")
        self.signals = arrays
        self.events = events
        self.targets = targets
        self.control_dt = float(meta["control_dt"])
        self._coord_column = {int(c): i for i, c in enumerate(meta["signal_coords"])}

    @classmethod
    def load(cls, path: Path) -> Recording:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        files = [path / c["file"] for c in meta.get("chunks", [])]
        if not meta.get("finished", False):
            listed = {f.name for f in files}
            files += [f for f in sorted((path / "chunks").glob("chunk_*.npz"))
                      if f.name not in listed]
        parts: dict[str, list[np.ndarray]] = {}
        coords, dofs = len(meta["signal_coords"]), len(meta["signal_dofs"])
        bodies = len(meta["body_labels"])
        prev_step = prev_state = None
        for f in files:
            if not f.exists():
                raise ValueError(f"{path}: chunk {f.name} listed in meta.json is missing")
            with np.load(f) as data:
                chunk = {key: data[key] for key in data.files}
            step, state_step = chunk["step"], chunk["state_step"]
            if len(step) and prev_step is not None and step[0] != prev_step + 1:
                raise ValueError(f"{f.name}: starts at step {step[0]}, expected {prev_step + 1}")
            if np.any(np.diff(step) != 1):
                raise ValueError(f"{f.name}: steps are not contiguous")
            if not np.allclose(chunk["sim_time"], step * float(meta["control_dt"]),
                               rtol=0.0, atol=1e-9):
                raise ValueError(f"{f.name}: sim_time != step * control_dt")
            if len(state_step) and prev_state is not None and state_step[0] <= prev_state:
                raise ValueError(f"{f.name}: state_step not increasing across chunks")
            if np.any(np.diff(state_step) <= 0) or not np.isin(state_step, step).all():
                raise ValueError(f"{f.name}: state_step not increasing or not a subset of step")
            if chunk["body_q"].shape[1:] != (bodies, 7):
                raise ValueError(f"{f.name}: body_q shape {chunk['body_q'].shape} does not match "
                                 f"{bodies} body_labels")
            if chunk["joint_q"].shape[1] != coords or chunk["joint_qd"].shape[1] != dofs:
                raise ValueError(f"{f.name}: joint_q/joint_qd columns do not match meta")
            if len(step):
                prev_step = int(step[-1])
            if len(state_step):
                prev_state = int(state_step[-1])
            for key, value in chunk.items():
                parts.setdefault(key, []).append(value)
        if parts:
            arrays = {key: np.concatenate(values) for key, values in parts.items()}
        else:
            empty = _Chunk(0, 0, coords, dofs, len(meta["effort_layout"]), bodies)
            arrays = {key: value.copy() for key, value in empty.arrays().items()}
        events = [Event(int(e["step"]), float(e["sim_time"]), str(e["kind"]), e.get("data", {}))
                  for e in _read_jsonl(path / "events.jsonl")]
        targets = [TargetMsg(int(t["step"]), float(t["sim_time"]), str(t["channel"]),
                             t["payload"]) for t in _read_jsonl(path / "targets.jsonl")]
        return cls(path, meta, arrays, events, targets)

    @staticmethod
    def list_runs(root: Path = DEFAULT_RECORDINGS_DIR) -> list[Path]:
        """Run directories under ``root`` (those with a ``meta.json``), newest first."""
        root = Path(root)
        if not root.is_dir():
            return []
        runs = [p for p in root.iterdir() if (p / "meta.json").is_file()]
        return sorted(runs, key=lambda p: p.name, reverse=True)

    @property
    def frame_count(self) -> int:
        return len(self.state_step)

    @property
    def duration_s(self) -> float:
        return len(self.step) * self.control_dt

    def frame_at_step(self, step: int) -> int:
        """Last state frame at or before ``step`` (0 if none)."""
        return max(int(np.searchsorted(self.state_step, step, side="right")) - 1, 0)

    def frame_at_time(self, t: float) -> int:
        return self.frame_at_step(int(np.floor(t / self.control_dt + 1e-9)))

    def body_index(self, label: str) -> int:
        return self.meta["body_labels"].index(label)

    def coord_column(self, model_coord: int) -> int:
        """Column of model coordinate ``model_coord`` in ``signals["joint_q"]``."""
        return self._coord_column[int(model_coord)]

    def hand_width_mm(self) -> np.ndarray:
        """Franka finger opening ``(-q1 + q2) * 1000`` per row."""
        coords = self.meta["robot_io"]["franka_hand"]["coords"]
        first, second = (self.coord_column(c) for c in coords)
        q = self.signals["joint_q"]
        return (-q[:, first] + q[:, second]) * 1e3
