"""One-step prediction videos of the learned latent LCS on recorded LCS episodes.

Per frame ``k`` of an episode: encode the real observation, take ONE LCS step with the
recorded action ``u_k`` and decode; no rollout. The frame is two halves from the same
headless viser client: the recorded scene at ``k`` (robots from FK of the recorded joints,
belt bodies from ``sim_belt_xyz``), and a belts-only view (current, true next, predicted next).
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import newton
import numpy as np
import trimesh
import viser
import warp as wp

from task_common.latent_encoder import LatentDecoder, LatentEncoder, LearnedLcs
from task_common.replay_learned_mpc import _tube_faces, tube_vertices
from task_common.replay_video import LEGEND_FONT, LEGEND_PANEL, LEGEND_TEXT, CameraPose
from task_common.sim_snapshot import load as load_snapshot

LCS_OUT = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v2_20260925")
DEPLOY = LCS_OUT / "deploy_v2_decoded_only" / "deploy.npz"
DECODER = LCS_OUT / "deploy_v2_decoded_only" / "decoder.npz"
SPLIT = LCS_OUT / "split.json"
LCS_REPO = Path("/home/hienbui/git/lcs_learning")
LCS_PYTHON = LCS_REPO / ".venv/bin/python"  # has torch; this repo's venv does not
SINKHORN_EPS, SINKHORN_ITERS = 1e-3, 120  # the trainer's --sinkhorn-epsilon/-iters defaults
SOLVERS = ("lcp", "pgd")  # lcp = exact LCP (evaluate_v2's qpOASES); pgd = the trained 25-it PGD
ERR_MAX_MM = 5.0  # colour-bar top (fixed)
TUBE_M = 0.0015  # thinner than the 3.3 mm belt: mm offsets stay visible
TUBE_SIDES = 8

GOLD = (255, 188, 0)  # overwritten with belt_rgb(BELT_COLOR) by the script
CURRENT_GREY = (156, 163, 175)
PRED_BLUE = (37, 99, 235)
# RdYlGn reversed: green (0 mm) -> pale yellow -> red (max).
ERR_STOPS = np.array([(0, 104, 55), (26, 152, 80), (102, 189, 99), (166, 217, 106),
                      (217, 239, 139), (255, 255, 191), (254, 224, 139), (253, 174, 97),
                      (244, 109, 67), (215, 48, 39), (165, 0, 38)], dtype=np.float64)


# ---- model --------------------------------------------------------------------------------

def _subset_inverses(F: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(F)
    out = []
    for mask in itertools.product((False, True), repeat=n):
        s = np.flatnonzero(mask)
        out.append((s, np.linalg.inv(F[np.ix_(s, s)]) if len(s) else np.zeros((0, 0))))
    return out


def lcp_exact(F: np.ndarray, q: np.ndarray, subsets=None) -> np.ndarray:
    """``lam >= 0, F lam + q >= 0, lam'(F lam + q) = 0`` by active-set enumeration.

    ``F + F'`` is positive definite for the learned LCS (``GG' + sI`` plus a skew part), so the
    solution is unique and equals the qpOASES QP optimum ``lcs_model.py`` uses."""
    subsets = subsets or _subset_inverses(F)
    q = np.asarray(q, dtype=np.float64)
    best, best_v = np.zeros_like(q), np.full(len(q), np.inf)
    for s, inv in subsets:
        lam = np.zeros_like(q)
        if len(s):
            lam[:, s] = -q[:, s] @ inv.T
        w = lam @ F.T + q
        v = np.maximum(-lam.min(1), -w.min(1))
        take = v < best_v
        best[take], best_v[take] = lam[take], v[take]
    return best


class OneStepModel:
    """numpy encoder + one LCS step + decoder of the exported v2 model."""

    def __init__(self, deploy: Path = DEPLOY, decoder: Path = DECODER,
                 solver: str = "lcp") -> None:
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        self.deploy, self.decoder_path, self.solver = Path(deploy), Path(decoder), solver
        self.enc = LatentEncoder.load(deploy)
        self.lcs = LearnedLcs.load(deploy)
        self.dec = LatentDecoder.load(decoder)
        self._subsets = _subset_inverses(self.lcs.F)

    def encode(self, pcd, prop, belt) -> np.ndarray:
        return self.enc.encode(pcd, prop, belt)

    def step(self, z, u) -> np.ndarray:
        """``(B, n_x)`` next latents for ``(B, n_x)`` / ``(B, n_u)``, float64."""
        s = self.lcs
        z, u = np.atleast_2d(z).astype(np.float64), np.atleast_2d(u).astype(np.float64)
        if self.solver == "pgd":
            return np.stack([s.step(a, b)[0] for a, b in zip(z, u, strict=True)])
        lam = lcp_exact(s.F, z @ s.E.T + u @ s.H.T + s.c, self._subsets)
        return z @ s.A.T + u @ s.B.T + lam @ s.D.T + s.d

    def decode(self, z) -> np.ndarray:
        return self.dec.decode_batch(np.atleast_2d(z))


@dataclass
class Episode:
    """The encoder inputs of one LCS episode, cast as the training loader does."""

    path: Path
    pcd: list[np.ndarray]
    prop: np.ndarray  # (T, 40) float32
    belt: np.ndarray  # (T, 150, 3) float32
    u: np.ndarray  # (T, 12) float32 -> float64
    dt: float
    outcome: str
    data: dict = field(repr=False)

    @classmethod
    def load(cls, path: Path) -> Episode:
        with np.load(path, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        meta = json.loads(str(data["sim_meta"]))
        return cls(Path(path), [np.asarray(p, np.float32) for p in data["pcd"]],
                   data["state"].astype(np.float32), data["pcd_belt"].astype(np.float32),
                   data["actions"].astype(np.float32).astype(np.float64),
                   float(meta["lcs_format"]["sample_period_s"]), str(data["sim_outcome"]), data)

    @property
    def frames(self) -> int:
        return len(self.prop)


def one_step(model: OneStepModel, ep: Episode, k: int) -> dict[str, np.ndarray]:
    """``decode(step(encode(obs_k), u_k))`` and ``decode(encode(obs_{k+1}))``, single-sample
    calls (the check re-derives these bit-exactly)."""
    z = model.encode(ep.pcd[k], ep.prop[k], ep.belt[k])
    z1 = model.encode(ep.pcd[k + 1], ep.prop[k + 1], ep.belt[k + 1])
    z_hat = model.step(z, ep.u[k])
    return {"z": z, "z_next_pred": z_hat[0], "pred": model.decode(z_hat)[0],
            "recon_next": model.decode(z1)[0]}


def rmse_mm(a, b) -> np.ndarray:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return 1e3 * np.sqrt((d ** 2).mean(axis=(-1, -2)))


def predict_episode(model: OneStepModel, ep: Episode) -> dict[str, Any]:
    """Per frame k = 0..T-2: predicted / true / current belts, per-point error and RMSEs."""
    rows = [one_step(model, ep, k) for k in range(ep.frames - 1)]
    pred = np.stack([r["pred"] for r in rows])
    recon = np.stack([r["recon_next"] for r in rows])
    true, cur = ep.belt[1:], ep.belt[:-1]
    return {"pred": pred, "recon_next": recon, "true_next": true, "current": cur,
            "point_err_mm": 1e3 * np.linalg.norm(pred.astype(np.float64) - true, axis=-1),
            "rmse_model_mm": rmse_mm(pred, true), "rmse_recon_mm": rmse_mm(recon, true),
            "rmse_nomotion_mm": rmse_mm(cur, true),
            "time_s": ep.dt * np.arange(ep.frames - 1)}


_SINKHORN = """
import sys
import numpy as np
import torch
sys.path.insert(0, sys.argv[1])
from lcs_learning.losses_pointcloud import sinkhorn_emd_loss
d = np.load(sys.argv[2])
eps, iters = float(sys.argv[4]), int(sys.argv[5])
torch.set_num_threads(4)
with torch.no_grad():
    out = [float(sinkhorn_emd_loss(torch.from_numpy(p[None]), torch.from_numpy(t[None]),
                                   epsilon=eps, num_iters=iters)) for p, t in zip(d["a"], d["b"])]
np.save(sys.argv[3], np.asarray(out))
"""


def sinkhorn_emd_mm(a, b, epsilon: float = SINKHORN_EPS,
                    iters: int = SINKHORN_ITERS) -> np.ndarray:
    """Per-frame ``lcs_learning.losses_pointcloud.sinkhorn_emd_loss`` (torch, CPU, in the
    lcs_learning venv) of (T, N, 3) vs (T, M, 3) metres, in mm. Its cost is the Euclidean
    distance, so the value is an (entropic) mean transport distance, already a length."""
    a, b = np.asarray(a), np.asarray(b)
    with tempfile.TemporaryDirectory(prefix="sinkhorn_") as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npy"
        np.savez(src, a=a, b=b)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        subprocess.run([str(LCS_PYTHON), "-c", _SINKHORN, str(LCS_REPO), str(src), str(dst),
                        repr(float(epsilon)), str(int(iters))], check=True, env=env,
                       stdout=subprocess.DEVNULL)
        return 1e3 * np.load(dst)


def emd_exact_mm(a, b) -> np.ndarray:
    """Exact EMD (mean matched Euclidean distance) of equal-size uniform sets, mm: for uniform
    weights the optimal transport is a permutation (``linear_sum_assignment``)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    out = []
    for p, t in zip(np.asarray(a, np.float64), np.asarray(b, np.float64), strict=True):
        c = cdist(p, t)
        r, k = linear_sum_assignment(c)
        out.append(c[r, k].mean())
    return 1e3 * np.asarray(out)


# ---- scene reconstruction -----------------------------------------------------------------

def _quat_xyzw_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw shortest rotations taking unit rows of ``a`` onto ``b``."""
    c = np.cross(a, b)
    w = 1.0 + (a * b).sum(1)
    q = np.concatenate([c, w[:, None]], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _quat_mul_xyzw(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    px, py, pz, pw = p.T
    qx, qy, qz, qw = q.T
    return np.stack([pw * qx + px * qw + py * qz - pz * qy, pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw, pw * qw - px * qx - py * qy - pz * qz],
                    1)


class EpisodeScene:
    """``body_q`` of recorded frames: FK of the recorded arm joints on the start snapshot's
    joint state (Robotiq linkage kept at the snapshot's), the large pulley's recorded pose and
    the belt bodies at ``sim_belt_xyz`` (each body's +z along its segment, twist from the
    snapshot)."""

    def __init__(self, model: newton.Model, ep: Episode) -> None:
        meta = json.loads(str(ep.data["sim_meta"]))
        self.snapshot_path = Path(meta["start_state"])
        snap = load_snapshot(self.snapshot_path)
        labels = [str(b) for b in model.body_label]
        jlabels = [str(j) for j in model.joint_label]
        qs = model.joint_q_start.numpy()
        coord = lambda name: int(qs[jlabels.index(name)])
        self.arm = [coord(f"panda_arm/panda_joint{i}") for i in range(1, 8)]
        self.fingers = [coord(f"panda_hand/panda_finger_joint{i}") for i in (1, 2)]
        ur = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
        self.ur = [next(int(qs[i]) for i, j in enumerate(jlabels) if j.endswith(f"/{n}_joint"))
                   for n in ur]
        self.belt = [i for i, b in enumerate(labels) if "cable_edge_body" in b]
        self.pulley = labels.index("board/large_round_pulley")
        self.finger_tip = labels.index("panda_hand/finger_tip")
        self.model, self.ep, self.q0 = model, ep, snap.joint_q.astype(np.float32)
        self._state = model.state()
        p0 = snap.body_q[self.belt, :3].astype(np.float64)
        self._belt_dir0 = self._dirs(p0)
        self._belt_q0 = snap.body_q[self.belt, 3:].astype(np.float64)
        self._finger_sign = np.sign(snap.joint_q[self.fingers])

    @staticmethod
    def _dirs(p: np.ndarray) -> np.ndarray:
        d = np.roll(p, -1, axis=0) - p
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def body_q(self, k: int) -> np.ndarray:
        d = self.ep.data
        q = self.q0.copy()
        q[self.arm] = d["state"][k, :7]
        q[self.ur] = d["state"][k, 7:13]
        q[self.fingers] = self._finger_sign * float(d["sim_hand_mm"][k]) / 2000.0
        with wp.ScopedDevice(self.model.device):
            self.model.joint_q.assign(q)
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self._state)
            out = self._state.body_q.numpy().copy()
        pose = d["sim_pulley_large_pose"][k]  # xyz_wxyz
        out[self.pulley] = [*pose[:3], *pose[4:7], pose[3]]
        p = d["sim_belt_xyz"][k].astype(np.float64)
        rot = _quat_xyzw_between(self._belt_dir0, self._dirs(p))
        out[self.belt, :3] = p
        out[self.belt, 3:] = _quat_mul_xyzw(rot, self._belt_q0)
        return out.astype(np.float32)

    def ee_error_mm(self, k: int) -> float:
        """FK finger_tip vs the recorded ``sim_ee_franka`` position (sanity)."""
        return 1e3 * float(np.linalg.norm(self.body_q(k)[self.finger_tip, :3]
                                          - self.ep.data["sim_ee_franka"][k, :3]))


# ---- colours ------------------------------------------------------------------------------

def err_rgb(err_mm, vmax: float = ERR_MAX_MM) -> np.ndarray:
    """(..., 3) uint8 on the fixed green -> red scale."""
    x = np.clip(np.asarray(err_mm, np.float64) / vmax, 0.0, 1.0) * (len(ERR_STOPS) - 1)
    i = np.minimum(np.floor(x).astype(int), len(ERR_STOPS) - 2)
    f = (x - i)[..., None]
    return np.round(ERR_STOPS[i] * (1 - f) + ERR_STOPS[i + 1] * f).astype(np.uint8)


def srgb_to_linear(rgb) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ---- 3D -----------------------------------------------------------------------------------

class BeltTubes:
    """Belts-only view: solid tubes (``add_mesh_simple``) and one per-vertex coloured tube (glb,
    COLOR_0 written linear so the screen shows the given sRGB)."""

    ROOT = "/prediction"

    def __init__(self, server: viser.ViserServer, sides: int = TUBE_SIDES,
                 radius: float = TUBE_M) -> None:
        self.server, self.sides, self.radius = server, sides, radius
        self.handles: dict[str, Any] = {}
        self.sent: dict[str, np.ndarray] = {}  # last vertices per tube (for the check)

    def tube(self, name: str, pts: np.ndarray, color, opacity: float = 1.0) -> None:
        verts = tube_vertices(pts, self.radius, self.sides).astype(np.float32)
        self.sent[name] = verts
        faces = _tube_faces(len(pts), self.sides)
        h = self.handles.get(name)
        op = None if opacity >= 1.0 else float(opacity)
        if isinstance(h, viser.MeshHandle) and h.vertices.shape == verts.shape:
            h.vertices = verts
            h.color = tuple(int(c) for c in color)
            h.opacity = op
            h.visible = True
            return
        if h is not None:
            h.remove()
        self.handles[name] = self.server.scene.add_mesh_simple(
            f"{self.ROOT}/{name}", verts, faces, color=tuple(int(c) for c in color), opacity=op,
            cast_shadow=False, receive_shadow=False)

    def colored_tube(self, name: str, pts: np.ndarray, point_rgb: np.ndarray) -> None:
        verts = tube_vertices(pts, self.radius, self.sides).astype(np.float32)
        self.sent[name] = verts
        lin = srgb_to_linear(np.repeat(point_rgb, self.sides, axis=0))
        rgba = np.concatenate([np.round(lin * 255), np.full((len(verts), 1), 255)], 1)
        mesh = trimesh.Trimesh(verts, _tube_faces(len(pts), self.sides), process=False,
                               vertex_colors=rgba.astype(np.uint8))
        h = self.handles.pop(name, None)
        if h is not None:
            h.remove()
        self.handles[name] = self.server.scene.add_mesh_trimesh(
            f"{self.ROOT}/{name}", mesh, cast_shadow=False, receive_shadow=False)

    def set_visible(self, visible: bool) -> None:
        for h in self.handles.values():
            h.visible = visible


def viewer_handles(viewer: Any) -> list[Any]:
    """Every scene node the newton viewer drew (shapes, planes/grid)."""
    out = list(getattr(viewer, "_scene_handles", {}).values())
    for v in getattr(viewer, "_plane_handles", {}).values():
        out.extend(v if isinstance(v, (list, tuple)) else [v])
    return out


# ---- 2D overlays --------------------------------------------------------------------------

def _font(size: int):
    from PIL import ImageFont

    return (ImageFont.truetype(str(LEGEND_FONT), size) if LEGEND_FONT.is_file()
            else ImageFont.load_default(size))


def text_rgba(text: str, size: int, pad: int | None = None,
              panel=LEGEND_PANEL) -> np.ndarray:
    from PIL import Image, ImageDraw

    font = _font(size)
    pad = max(2, round(size * 0.35)) if pad is None else pad
    x0, y0, x1, y1 = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    img = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), panel)
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, fill=(*LEGEND_TEXT, 255), font=font)
    return np.asarray(img)


@dataclass(frozen=True)
class Row:
    """Legend row: ``kind`` is ``bar`` (one colour), ``ramp`` (the error colours), ``line``
    or ``dash`` (plot series)."""

    label: str
    kind: str
    color: tuple[int, int, int] = (0, 0, 0)


def legend_rgba(rows: Sequence[Row], size: int, colorbar: bool = False,
                vmax: float = ERR_MAX_MM) -> np.ndarray:
    """(h, w, 4) legend panel; with ``colorbar`` a labelled 0..vmax mm error bar at the
    bottom."""
    from PIL import Image, ImageDraw

    font, small = _font(size), _font(max(8, round(size * 0.8)))
    pad, gap, row = round(size * 0.6), round(size * 0.5), round(size * 1.45)
    sw, bar = round(size * 2.6), max(3, round(size * 0.42))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text_w = max(round(probe.textlength(r.label, font=font)) for r in rows)
    cb_h = round(size * 2.6) if colorbar else 0
    w = pad + sw + gap + text_w + pad
    h = 2 * pad + row * len(rows) - (row - size) + cb_h
    img = Image.new("RGBA", (w, h), LEGEND_PANEL)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, w - 1, h - 1), outline=(156, 163, 175, 255))
    for i, r in enumerate(rows):
        cy, x0 = pad + row * i + size // 2, pad
        y0, y1 = cy - bar // 2, cy - bar // 2 + bar - 1
        if r.kind == "bar":
            draw.rectangle((x0, y0, x0 + sw - 1, y1), fill=(*r.color, 255))
        elif r.kind == "ramp":
            cols = err_rgb(np.linspace(0, vmax, sw))
            for j, c in enumerate(cols):
                draw.line((x0 + j, y0, x0 + j, y1), fill=(*map(int, c), 255))
        elif r.kind in ("line", "dash"):
            lw = max(2, round(size * 0.14))
            step = round(size * 0.5) if r.kind == "dash" else sw
            for xa in range(x0, x0 + sw, step * (2 if r.kind == "dash" else 1)):
                draw.line((xa, cy, min(xa + step, x0 + sw), cy), fill=(*r.color, 255), width=lw)
        else:
            raise ValueError(f"unknown legend row kind {r.kind!r}")
        draw.text((x0 + sw + gap, cy), r.label, fill=(*LEGEND_TEXT, 255), font=font, anchor="lm")
    if colorbar:
        y0 = h - pad - cb_h + round(size * 0.35)
        bw = w - 2 * pad
        cols = err_rgb(np.linspace(0, vmax, bw))
        for j, c in enumerate(cols):
            draw.line((pad + j, y0, pad + j, y0 + bar * 2), fill=(*map(int, c), 255))
        for v in np.arange(0, vmax + 1e-9, 1.0):
            x = pad + round(v / vmax * (bw - 1))
            draw.line((x, y0 + bar * 2, x, y0 + bar * 2 + 3), fill=(*LEGEND_TEXT, 255))
            lab = f"{v:g}" + (" mm" if v == vmax else "")
            draw.text((x, y0 + bar * 2 + 4), lab, fill=(*LEGEND_TEXT, 255), font=small,
                      anchor="ma" if 0 < v < vmax else ("la" if v == 0 else "ra"))
    return np.asarray(img)


@dataclass(frozen=True)
class Series:
    label: str
    values: np.ndarray
    color: tuple[int, int, int]
    dashed: bool = False


class RmsePlot:
    """Small RMSE-over-time plot: static axes and series, a per-frame cursor."""

    def __init__(self, t: np.ndarray, series: Sequence[Series], width: int, height: int,
                 size: int, ymax: float | None = None, title: str = "one-step RMSE (mm)",
                 legend: bool = True) -> None:
        from PIL import Image, ImageDraw

        self.t, self.series, self.w, self.h, self.size = t, list(series), width, height, size
        self.title = title
        font = _font(size)
        top = max(float(np.max(s.values)) for s in self.series)
        self.ymax = ymax or float(np.ceil(top * 1.05 + 1e-9))
        self.x0, self.x1 = round(size * 2.2), width - round(size * 0.6)
        self.y0, self.y1 = round(size * 1.6), height - round(size * 1.7)
        img = Image.new("RGBA", (width, height), LEGEND_PANEL)
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, width - 1, height - 1), outline=(156, 163, 175, 255))
        d.text((self.x0, round(size * 0.3)), title, fill=(*LEGEND_TEXT, 255), font=font)
        axis = (107, 114, 128, 255)
        d.line((self.x0, self.y1, self.x1, self.y1), fill=axis)
        d.line((self.x0, self.y0, self.x0, self.y1), fill=axis)
        for v in self._ticks(self.ymax):
            y = self._y(v)
            d.line((self.x0 - 3, y, self.x0, y), fill=axis)
            d.line((self.x0 + 1, y, self.x1, y), fill=(229, 231, 235, 255))
            d.text((self.x0 - 5, y), f"{v:g}", fill=(*LEGEND_TEXT, 255), font=font, anchor="rm")
        for v in self._time_ticks(float(t[0]), float(t[-1])):
            x = self._x(v)
            d.line((x, self.y1, x, self.y1 + 3), fill=axis)
            d.text((x, self.y1 + 4), f"{v:g}", fill=(*LEGEND_TEXT, 255), font=font, anchor="ma")
        d.text((self.x1, self.y1 + 4), "s", fill=(*LEGEND_TEXT, 255), font=font, anchor="ra")
        lw = max(1, round(size * 0.12))
        for s in self.series:
            pts = [(self._x(a), self._y(min(b, self.ymax))) for a, b in zip(t, s.values,
                                                                             strict=True)]
            if not s.dashed:
                d.line(pts, fill=(*s.color, 255), width=lw, joint="curve")
                continue
            dash = max(3, round(size * 0.45))
            for x, on in self._dashes(np.asarray(pts, np.float64), dash):
                if on:
                    d.line([tuple(p) for p in x], fill=(*s.color, 255), width=lw)
        if legend:
            x = self.x1
            for s in reversed(self.series):
                tw = round(d.textlength(s.label, font=font))
                x -= tw
                d.text((x, round(size * 0.3)), s.label, fill=(*s.color, 255), font=font)
                x -= round(size * 0.8)
        self.base = np.asarray(img)

    @staticmethod
    def _dashes(pts: np.ndarray, dash: float):
        """Split a polyline into alternating on/off runs of ``dash`` px of arc length."""
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        n = np.maximum(1, np.ceil(seg).astype(int))
        dense = np.concatenate([np.linspace(a, b, m, endpoint=False)
                                for a, b, m in zip(pts[:-1], pts[1:], n, strict=True)]
                               + [pts[-1:]])
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
        on = (arc // dash).astype(int) % 2 == 0
        cut = np.flatnonzero(np.diff(on.astype(int))) + 1
        for run in np.split(np.arange(len(dense)), cut):
            if len(run) > 1:
                yield dense[run], bool(on[run[0]])

    @staticmethod
    def _ticks(top: float, n: int = 3) -> list[float]:
        raw = top / n
        mag = 10 ** np.floor(np.log10(raw))
        step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
        return [float(v) for v in np.arange(0, top + 1e-9, step)]

    def _time_ticks(self, a: float, b: float) -> list[float]:
        """Nice ticks over [a, b] (the plotted range need not start at 0)."""
        ticks = self._ticks(b - a, n=5)
        step = ticks[1] - ticks[0] if len(ticks) > 1 else max(b - a, 1e-9)
        return [float(v) for v in np.arange(np.ceil(a / step - 1e-9) * step, b + 1e-9, step)]

    def _x(self, t: float) -> int:
        t0, t1 = float(self.t[0]), float(self.t[-1])
        return round(self.x0 + (self.x1 - self.x0) * (t - t0) / max(t1 - t0, 1e-9))

    def _y(self, v: float) -> int:
        return round(self.y1 - (self.y1 - self.y0) * v / self.ymax)

    def frame(self, k: int) -> np.ndarray:
        img = self.base.copy()
        x = self._x(float(self.t[k]))
        img[self.y0:self.y1, max(x - 1, 0):x + 1, :3] = (220, 38, 38)
        return img


def blend(frame: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """In-place alpha blend of ``rgba`` at (x, y), clipped to the frame."""
    h, w = rgba.shape[:2]
    H, W = frame.shape[:2]
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xa >= xb or ya >= yb:
        return
    src = rgba[ya - y:yb - y, xa - x:xb - x]
    a = src[..., 3:4].astype(np.uint16)
    region = frame[ya:yb, xa:xb]
    region[:] = (src[..., :3].astype(np.uint16) * a + region * (255 - a) + 127) // 255


def fit_camera(pts: np.ndarray, base: CameraPose, aspect: float,
               margin: float = 0.08) -> CameraPose:
    """``base``'s view direction, up and fov, moved along the view direction so ``pts`` (any
    shape (..., 3)) fill the frame with ``margin`` (fraction of the half-extent) to spare."""
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    f = np.subtract(base.look_at, base.position)
    f = f / np.linalg.norm(f)
    up = np.asarray(base.up, np.float64)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    upc = np.cross(right, f)
    ty = np.tan(0.5 * base.fov) * (1.0 - margin)
    tx = ty * aspect
    c = 0.5 * (p.min(0) + p.max(0))

    def distance(c: np.ndarray) -> float:
        r = p - c
        x, y, z = r @ right, r @ upc, r @ f
        lo, hi = 0.0, 10.0
        for _ in range(60):
            s = 0.5 * (lo + hi)
            depth = z + s
            ok = (np.all(depth > 1e-3) and np.all(np.abs(x) <= tx * depth)
                  and np.all(np.abs(y) <= ty * depth))
            lo, hi = (lo, s) if ok else (s, hi)
        return hi

    for _ in range(8):  # re-centre on the projected box
        hi = distance(c)
        r = p - c
        depth = r @ f + hi
        x, y = (r @ right) / depth, (r @ upc) / depth
        c = c + hi * (0.5 * (x.max() + x.min()) * right + 0.5 * (y.max() + y.min()) * upc)
    hi = distance(c)
    return CameraPose(tuple(map(float, c - hi * f)), tuple(map(float, c)), base.up, base.fov)


def project_ndc(pts: np.ndarray, cam: CameraPose, aspect: float) -> np.ndarray:
    """(N, 2) image coordinates in [-1, 1] (x right, y up) of ``pts`` for ``cam`` at
    ``aspect``; |v| > 1 is outside the view."""
    p = np.asarray(pts, np.float64).reshape(-1, 3)
    f = np.subtract(cam.look_at, cam.position)
    f = f / np.linalg.norm(f)
    right = np.cross(f, np.asarray(cam.up, np.float64))
    right /= np.linalg.norm(right)
    upc = np.cross(right, f)
    r = p - np.asarray(cam.position)
    depth = r @ f
    t = np.tan(0.5 * cam.fov)
    return np.stack([(r @ right) / depth / (t * aspect), (r @ upc) / depth / t], 1)


def luma_stats(frame: np.ndarray) -> tuple[float, float]:
    y = frame[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(y.mean()), float(y.std())
