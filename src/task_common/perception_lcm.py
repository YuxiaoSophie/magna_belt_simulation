"""Hardware perception messages: ``drake::lcmt_point_cloud`` and ``magna::lcmt_round_belt_state``.

``point_cloud_msg`` mirrors Drake's ``PointCloudToLcm`` (``perception/point_cloud_to_lcm.cc``):
unstructured (``height 1``), fields ``x``/``y``/``z`` FLOAT32 at 0/4/8, then ``rgb`` UINT32
(bytes r, g, b, 0) when colours are given, non-finite points dropped, ``IS_STRICTLY_FINITE``,
filler so the encoded size before the data is a multiple of 16. ``point_cloud_xyz`` decodes any
layout from the field table. Pure numpy + LCM types.
"""

from __future__ import annotations

import numpy as np

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from drake import lcmt_point_cloud, lcmt_point_cloud_field
from magna import lcmt_round_belt_state

WORLD_FRAME = "world"
TASKBOARD_FRAME = "taskboard"
_ALIGN = 16
_DTYPES = {
    lcmt_point_cloud_field.INT8: "i1", lcmt_point_cloud_field.UINT8: "u1",
    lcmt_point_cloud_field.INT16: "i2", lcmt_point_cloud_field.UINT16: "u2",
    lcmt_point_cloud_field.INT32: "i4", lcmt_point_cloud_field.UINT32: "u4",
    lcmt_point_cloud_field.FLOAT32: "f4", lcmt_point_cloud_field.FLOAT64: "f8",
}


def _field(name: str, offset: int, datatype: int) -> lcmt_point_cloud_field:
    field = lcmt_point_cloud_field()
    field.name, field.byte_offset, field.datatype, field.count = name, offset, datatype, 1
    return field


def point_cloud_msg(utime: int, xyz, rgb=None, frame_name: str = WORLD_FRAME
                    ) -> lcmt_point_cloud:
    """``PointCloudToLcm`` layout of ``xyz (N, 3)`` [+ ``rgb (N, 3) uint8``], little-endian."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    fields = [_field(n, 4 * i, lcmt_point_cloud_field.FLOAT32) for i, n in enumerate("xyz")]
    record = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        fields.append(_field("rgb", 12, lcmt_point_cloud_field.UINT32))
        record.append(("rgb", "u1", (4,)))
    points = np.zeros(int(finite.sum()), dtype=np.dtype(record))
    kept = xyz[finite]
    points["x"], points["y"], points["z"] = kept[:, 0], kept[:, 1], kept[:, 2]
    if rgb is not None:
        points["rgb"][:, :3] = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)[finite]

    msg = lcmt_point_cloud()
    msg.utime = int(utime)
    msg.frame_name = frame_name
    msg.height = 1
    msg.flags = lcmt_point_cloud.IS_STRICTLY_FINITE
    msg.fields = fields
    msg.num_fields = len(fields)
    msg.point_step = points.dtype.itemsize
    msg.filler, msg.filler_size = b"", 0
    msg.data, msg.data_size = b"", 0
    msg.filler_size = (_ALIGN - len(msg.encode()) % _ALIGN) % _ALIGN
    msg.filler = bytes(msg.filler_size)
    msg.data = points.tobytes()
    msg.data_size = len(msg.data)
    msg.width = len(points)
    msg.row_step = msg.data_size
    return msg


def point_cloud_xyz(msg: lcmt_point_cloud) -> np.ndarray:
    """``(N, 3)`` float32 xyz, read through the message's field table (any extra fields)."""
    by_name = {f.name: f for f in msg.fields}
    missing = [n for n in "xyz" if n not in by_name]
    if missing:
        raise ValueError(f"point cloud lacks fields {missing}")
    n = int(msg.width) * int(msg.height)
    step = int(msg.point_step)
    if len(msg.data) < n * step:
        raise ValueError(f"point cloud data {len(msg.data)} B < {n} x {step} B")
    order = ">" if msg.flags & lcmt_point_cloud.IS_BIGENDIAN else "<"
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    cols = []
    for name in "xyz":
        f = by_name[name]
        dtype = np.dtype(order + _DTYPES[f.datatype])
        start = int(f.byte_offset)
        cols.append(raw[:, start:start + dtype.itemsize].copy().view(dtype)[:, 0])
    return np.stack(cols, axis=1).astype(np.float32)


def to_frame(X_WF: np.ndarray, p_W) -> np.ndarray:
    """World points -> frame ``F`` (float64)."""
    X = np.asarray(X_WF, dtype=np.float64)
    return (np.asarray(p_W, dtype=np.float64) - X[:3, 3]) @ X[:3, :3]


def from_frame(X_WF: np.ndarray, p_F) -> np.ndarray:
    """Frame ``F`` points -> world (float64)."""
    X = np.asarray(X_WF, dtype=np.float64)
    return np.asarray(p_F, dtype=np.float64) @ X[:3, :3].T + X[:3, 3]


def round_belt_state_msg(utime: int, points, frame_name: str = TASKBOARD_FRAME
                         ) -> lcmt_round_belt_state:
    """``num_control_points 0``, ``point_positions`` = ``points (n, 3)`` as float."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    msg = lcmt_round_belt_state()
    msg.utime = int(utime)
    msg.frame_name = frame_name
    msg.num_control_points = 0
    msg.control_point_positions = []
    msg.num_points = len(pts)
    msg.point_positions = pts.astype(np.float32).tolist()
    return msg


def round_belt_points(msg: lcmt_round_belt_state) -> np.ndarray:
    """``point_positions`` as ``(n, 3)`` float64, in ``msg.frame_name``."""
    return np.asarray(msg.point_positions, dtype=np.float32).reshape(-1, 3).astype(np.float64)
