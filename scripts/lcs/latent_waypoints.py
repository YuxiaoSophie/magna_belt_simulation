#!/usr/bin/env python3
"""Latent waypoints along a demo: encoded demo belt observations as a sequence of MPPI goals.

Waypoint frames of ``--goal-demo``: ``stride, 2 stride, ...`` before the pre_place_2 latch, the
latch, then every ``stride`` up to the first ``hold:place_3`` frame (included), then the final
(seated) frame; frame 0 is excluded. Per waypoint ``k``: ``z_wp[k]`` = the numpy-twin ``encode``
of the goal demo's ``pcd_belt_seg`` (other encoders: ``episode_input_keys``) there (as the
encoder node); ``tol[k]`` = max(p``--percentile`` over ALL demos of their whitened distance to ``z_wp[k]`` at their time-aligned frame +-
``--window``, ``--floor``). The latch waypoint uses the stage-1 rule of
``latent_stage_tols.py`` (frames latch .. latch + window: an arrival, not a pass-through), so
its tol equals that script's ``tol_1`` for the same goal vector. Alignment: frames before the
latch scale by the latch ratio, latch .. first hold by that segment's ratio, later frames keep
their offset from the first hold. ``max_duration_s[k]`` = 2 x the goal demo's segment duration
(floor 3 s); the last waypoint: tol -1 (timeout only), 30 s. Also stores the goal demo's
per-frame latents ``z_traj (T, n_x)`` and actions ``u_traj (T, 12)`` (track mode and the demo
action prior of ``mppi_controller_node.py``). Deterministic: rerun -> byte-identical npz.
``--min-frame F`` (off: 0): drop the waypoints before goal-demo frame F (e.g. the static prehold
before its motion start); the kept ones are the full schedule's (same tols and durations).

``--bank-out DIR``: one such artifact per ``--goal-demos`` episode (default: every ``--demos``
episode; tolerances pooled over ``--demos`` as above) -> ``DIR/wp_<j>.npz`` (+ ``move0``) and
``DIR/bank.json`` (entries with each demo's initial latent ``z_traj[0]``), for the node's
per-episode nearest-initial-latent selection (``WaypointBank``, ``--waypoint-bank``).

    uv run --frozen python scripts/lcs/latent_waypoints.py --deploy <deploy.npz> \\
        --demos data/lcs/realfirst/v1_1/seed/demo/*.npz \\
        data/lcs/realfirst/v1_1/seed_ext/demo/*.npz \\
        --goal-demo data/lcs/realfirst/v1_1/seed/demo/episode_0000.npz --out <latent_waypoints.npz>
    uv run --frozen python scripts/lcs/latent_waypoints.py --deploy <deploy.npz> \\
        --demos <demo npz ...> --bank-out <dir>
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from latent_stage_tols import encode_obs, latch_frame, load_episode, sha256

from task_common.latent_encoder import LatentEncoder, LearnedLcs, episode_input_keys

FIRST_HOLD = "hold:place_3"
MIN_DURATION_S = 3.0
LAST_DURATION_S = 30.0
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
BANK_VERSION = 1


def first_hold_frame(labels: list[str]) -> int:
    hold = [t for t, lab in enumerate(labels) if lab == FIRST_HOLD]
    if not hold:
        raise ValueError(f"no {FIRST_HOLD} frame")
    return hold[0]


def waypoint_frames(latch: int, first_hold: int, n: int, stride: int) -> list[int]:
    if stride < 1 or not 0 < latch <= first_hold < n:
        raise ValueError(f"stride {stride}, latch {latch}, first hold {first_hold}, n {n}")
    frames = list(range(stride, latch, stride)) + [latch]
    frames += list(range(latch + stride, first_hold, stride)) + [first_hold, n - 1]
    return sorted(set(frames))


def _round(x: float) -> int:
    return math.floor(x + 0.5)


def align(t: int, src: tuple[int, int, int], dst: tuple[int, int, int]) -> int:
    """Frame of ``dst`` aligned to frame ``t`` of ``src``; ``(latch, first_hold, n)`` each."""
    (ls, hs, ns), (ld, hd, nd) = src, dst
    if t <= ls:
        out = _round(t * ld / ls)
    elif t <= hs:
        out = ld + _round((t - ls) * (hd - ld) / (hs - ls))
    else:
        out = hd + (t - hs)
    if t == ns - 1:
        out = nd - 1
    return min(max(out, 0), nd - 1)


def map_frame(t: int, src: tuple[int, int, int, int], dst: tuple[int, int, int, int]) -> int:
    """Frame of ``dst`` at the task progress of frame ``t`` of ``src``; ``(move0, latch,
    first_hold, n)`` each: prehold frames keep their offset to ``move0``, linear on [move0,
    latch] and [latch, first_hold] (the prefix frac mapping), offset from first_hold after."""
    (ms, ls, hs, ns), (md, ld, hd, nd) = src, dst
    if t <= ms:
        out = md - (ms - t)
    elif t <= ls:
        out = md + _round((t - ms) * (ld - md) / (ls - ms))
    elif t <= hs:
        out = ld + _round((t - ls) * (hd - ld) / (hs - ls))
    else:
        out = hd + (t - hs)
    if t == ns - 1:
        out = nd - 1
    return min(max(out, 0), nd - 1)


def move0_frame(labels: list[str]) -> int:
    return next(t for t, lab in enumerate(labels) if lab.startswith("move:"))


def tol_frames(t: int, latch: int, window: int, n: int, is_latch: bool) -> list[int]:
    lo = t if is_latch else t - window
    return [f for f in range(lo, t + window + 1) if 0 <= f < n]


def derive(deploy: Path, demos: list[Path], goal_demo: Path, stride: int, window: int,
           pct: float, floor: float, memo: dict | None = None, min_frame: int = 0) -> dict:
    """``memo``: episodes / encoded frames shared across calls (bank); same output."""
    memo = {} if memo is None else memo
    eps, cache = memo.setdefault("ep", {}), memo.setdefault("z", {})
    if "lcs" not in memo:
        memo.update(lcs=LearnedLcs.load(deploy), enc=LatentEncoder.load(deploy))
    lcs, enc = memo["lcs"], memo["enc"]

    def episode(path):
        if str(path) not in eps:
            eps[str(path)] = load_episode(path, episode_input_keys(enc))
        return eps[str(path)]

    with np.load(deploy, allow_pickle=False) as d:
        dt = float(np.asarray(d["dt"]).item())
    goal = episode(goal_demo)
    with np.load(goal_demo, allow_pickle=True) as d:
        u_traj = np.asarray(d["actions"], dtype=np.float64)
        frame_dt = float(np.median(np.diff(np.asarray(d["sim_time"], dtype=np.float64))))
    if abs(frame_dt - dt) > 1e-6:
        raise SystemExit(f"{goal_demo}: frame period {frame_dt} != deploy dt {dt}")
    n_g = goal["n"]
    key_g = (latch_frame(goal["labels"]), first_hold_frame(goal["labels"]), n_g)
    frames = waypoint_frames(*key_g, stride)
    if not 0 <= min_frame <= key_g[0]:
        raise SystemExit(f"--min-frame {min_frame}: 0 .. the latch frame {key_g[0]}")
    z_traj, n_empty = encode_obs(enc, goal, range(n_g))
    if n_empty:
        raise SystemExit(f"{goal_demo}: {n_empty} empty belt clouds")
    z_wp = z_traj[frames]
    pooled: list[list[float]] = [[] for _ in frames]
    per_demo = []
    for path in demos:
        ep = episode(path)
        n_j = ep["n"]
        key_j = (latch_frame(ep["labels"]), first_hold_frame(ep["labels"]), n_j)
        aligned = []
        for k, f in enumerate(frames):
            t_j = align(f, key_g, key_j)
            aligned.append(t_j)
            for t in tol_frames(t_j, key_j[0], window, n_j, f == key_g[0]):
                if (str(path), t) not in cache:
                    cache[(str(path), t)] = encode_obs(enc, ep, [t])[0][0]
                pooled[k].append(lcs.whitened_dist(cache[(str(path), t)], z_wp[k]))
        per_demo.append({"episode": str(path), "latch": key_j[0], "first_hold": key_j[1],
                         "n_frames": n_j, "aligned_frames": aligned})
    p = np.array([float(np.percentile(v, pct)) for v in pooled])
    tol = np.maximum(p, floor)
    tol[-1] = -1.0
    seg = np.diff([0, *frames]) * dt
    dur = np.maximum(2.0 * seg, MIN_DURATION_S)
    dur[-1] = LAST_DURATION_S
    n_pool = np.array([len(v) for v in pooled])
    out = {}
    if min_frame > 0:
        keep = np.asarray(frames) >= min_frame
        frames, z_wp, tol, p, n_pool, dur = (np.asarray(x)[keep] for x in
                                             (frames, z_wp, tol, p, n_pool, dur))
        for r in per_demo:
            r["aligned_frames"] = np.asarray(r["aligned_frames"])[keep].tolist()
        out["min_frame"] = np.int64(min_frame)
    return {**out, "frames": np.asarray(frames, np.int64), "z_wp": z_wp, "tol": tol,
            "tol_percentile_raw": p, "n_tol_frames": n_pool,
            "max_duration_s": dur, "z_traj": z_traj, "u_traj": u_traj,
            "latch": np.int64(key_g[0]), "first_hold": np.int64(key_g[1]), "dt": np.float64(dt),
            "z_std": np.asarray(lcs.z_std, np.float64),
            "deploy": np.str_(str(deploy.resolve())), "deploy_sha256": np.str_(sha256(deploy)),
            "demo_episode": np.str_(str(goal_demo.resolve())),
            "demos": np.array([str(Path(x).resolve()) for x in demos]),
            "stride": np.int64(stride), "window": np.int64(window),
            "percentile": np.float64(pct), "floor": np.float64(floor),
            "per_demo": np.str_(json.dumps(per_demo))}


def save_npz(path: Path, arrays: dict) -> None:
    """``np.savez`` with fixed zip timestamps (byte-identical reruns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[k]), allow_pickle=False)
            zf.writestr(zipfile.ZipInfo(f"{k}.npy", ZIP_DATE), buf.getvalue())


class Waypoints:
    """A ``latent_waypoints.npz`` (node goal modes ``waypoints:`` / ``track``, demo prior)."""

    def __init__(self, d: dict) -> None:
        self.frames = np.asarray(d["frames"], np.int64)
        self.z_wp = np.asarray(d["z_wp"], np.float64)
        self.tol = np.asarray(d["tol"], np.float64)
        self.max_duration_s = np.asarray(d["max_duration_s"], np.float64)
        self.z_traj = np.asarray(d["z_traj"], np.float64)
        self.u_traj = np.asarray(d["u_traj"], np.float64)
        self.dt = float(d["dt"])
        self.latch, self.first_hold = int(d["latch"]), int(d["first_hold"])
        self.deploy_sha256 = str(d["deploy_sha256"])
        self.demo_episode = str(d["demo_episode"])
        self.move0 = int(d["move0"]) if "move0" in d else None
        if not np.any(self.tol >= 0.0):
            raise ValueError("no waypoint with a tolerance")

    @classmethod
    def load(cls, path) -> Waypoints:
        with np.load(Path(path), allow_pickle=False) as f:
            return cls({k: f[k] for k in f.files})

    @property
    def T(self) -> int:
        return len(self.z_traj)

    def tol_at(self, idx: int) -> float:
        """Tol of the waypoint nearest to frame ``idx`` among those with a tol (>= 0)."""
        ok = np.flatnonzero(self.tol >= 0.0)
        return float(self.tol[ok[np.argmin(np.abs(self.frames[ok] - idx))]])

    def goal_seq(self, idx: int, H: int) -> np.ndarray:
        """``z_traj[idx+1 : idx+H+1]``, the final frame repeated past the end."""
        return self.z_traj[np.minimum(np.arange(idx + 1, idx + H + 1), self.T - 1)]

    def u_seq(self, idx: int, H: int) -> np.ndarray:
        """``u_traj[idx : idx+H]``, zeros past the demo's last frame."""
        out = np.zeros((H, self.u_traj.shape[1]))
        rows = self.u_traj[idx:idx + H]
        out[:len(rows)] = rows
        return out


def demo_tag(path) -> str:
    """``set/variant`` of a recording's start state (report only, never used for selection)."""
    with np.load(Path(path), allow_pickle=True) as f:
        st = json.loads(str(f["sim_meta"])).get("start_state", "")
        if "sim_start_state" in f.files:
            st = json.loads(str(f["sim_start_state"])).get("file", st)
    st = str(st)
    if "{" in st:  # dict repr in sim_meta
        st = st.split("'file': '", 1)[-1].split("'", 1)[0]
    p = Path(st)
    return f"{p.parent.name}/{p.stem.removesuffix('_osc')}" if st else "?"


def build_bank(deploy: Path, demos: list[Path], goal_demos: list[Path], out: Path, stride: int,
               window: int, pct: float, floor: float) -> dict:
    """One artifact per goal demo (+ ``move0``) -> ``out/wp_<j>.npz``; ``out/bank.json``."""
    memo: dict = {}
    entries = []
    for j, g in enumerate(goal_demos):
        doc = derive(deploy, demos, g, stride, window, pct, floor, memo)
        doc["move0"] = np.int64(move0_frame(memo["ep"][str(g)]["labels"]))
        f = out / f"wp_{j:02d}.npz"
        save_npz(f, doc)
        entries.append({"file": f.name, "sha256": sha256(f),
                        "demo_episode": str(Path(g).resolve()), "tag": demo_tag(g),
                        "move0": int(doc["move0"]), "latch": int(doc["latch"]),
                        "first_hold": int(doc["first_hold"]), "n_frames": len(doc["z_traj"]),
                        "frames": doc["frames"].tolist(),
                        "tol": [float(x) for x in doc["tol"]],
                        "z_init": [float(x) for x in doc["z_traj"][0]]})
    bank = {"version": BANK_VERSION, "deploy": str(deploy.resolve()),
            "deploy_sha256": sha256(deploy), "key": "z_traj[0] (encoded first demo frame)",
            "demos": [str(Path(x).resolve()) for x in demos], "stride": stride,
            "window": window, "percentile": pct, "floor": floor, "entries": entries}
    (out / "bank.json").write_text(json.dumps(bank, indent=1) + "\n")
    return bank


def bank_summary(bank: dict) -> str:
    head = (f"{len(bank['entries'])} entries (deploy sha {bank['deploy_sha256'][:12]}, tol "
            f"pool {len(bank['demos'])} demos)")
    lines = [head, "| j | demo | tag | move0 / latch / hold / T | waypoint frames | tols |",
             "|---|---|---|---|---|---|"]
    for j, e in enumerate(bank["entries"]):
        name = "/".join(Path(e["demo_episode"]).parts[-3:])
        lines.append(f"| {j} | {name} | {e['tag']} | {e['move0']} / {e['latch']} / "
                     f"{e['first_hold']} / {e['n_frames']} | {e['frames']} | "
                     f"{[round(x, 3) for x in e['tol']]} |")
    return "\n".join(lines)


def episode_marks(path) -> tuple[int, int, int, int]:
    """``(move0, latch, first_hold, n)`` from a recording's phase labels (no clouds read)."""
    with np.load(Path(path), allow_pickle=True) as f:
        names = json.loads(str(f["sim_meta"]))["phase_labels"]
        labels = [names[p] for p in np.asarray(f["sim_phase"], np.int64)]
    return move0_frame(labels), latch_frame(labels), first_hold_frame(labels), len(labels)


class WaypointBank:
    """``bank.json`` of ``--bank-out``: per-episode reference demo = the entry whose initial
    latent is nearest (whitened) to the controller's first latent."""

    def __init__(self, path) -> None:
        self.path = Path(path).resolve()
        doc = json.loads(self.path.read_text())
        if doc.get("version") != BANK_VERSION:
            raise ValueError(f"{path}: bank version {doc.get('version')} != {BANK_VERSION}")
        self.meta = doc["entries"]
        self.entries = [Waypoints.load(self.path.parent / e["file"]) for e in self.meta]
        if not self.entries or any(w.move0 is None for w in self.entries):
            raise ValueError(f"{path}: empty bank or an entry without move0")
        self.marks = [(w.move0, w.latch, w.first_hold, w.T) for w in self.entries]

    def base_marks(self, wp: Waypoints) -> tuple[int, int, int, int]:
        """Marks of the base artifact's demo (a bank entry, else its recording)."""
        for e, m in zip(self.meta, self.marks):
            if e["demo_episode"] == str(Path(wp.demo_episode).resolve()):
                return m
        return episode_marks(wp.demo_episode)

    def key_frames(self, offset: int, base) -> tuple[list[int], list[int]]:
        """Per entry: the demo clock at the node's first latent (``offset`` on the base demo's
        clock) and the frame compared with it (frame 0 unless past the entry's prehold)."""
        if offset > 0 and base is None:
            raise ValueError("a demo offset needs the base demo's marks")
        offs = [0 if offset <= 0 else map_frame(offset, base, m) for m in self.marks]
        return offs, [o if o > m[0] else 0 for o, m in zip(offs, self.marks)]

    def select(self, z, z_std, offset: int = 0, base=None, knn: int = 1
               ) -> tuple[Waypoints, int, dict]:
        """``(waypoints, demo clock offset, info)``; ``knn > 1``: per-frame mean of the k
        nearest entries mapped onto the nearest one's clock, tol = their max."""
        offs, keys = self.key_frames(offset, base)
        zs = np.asarray(z_std, np.float64)
        d = np.array([float(np.linalg.norm((np.asarray(z, np.float64) - w.z_traj[k]) / zs))
                      for w, k in zip(self.entries, keys)])
        order = np.argsort(d, kind="stable")
        nn = [int(j) for j in order[:max(1, int(knn))]]
        r = nn[0]
        info = {"entry": r, "demo": self.meta[r]["demo_episode"], "tag": self.meta[r]["tag"],
                "dist": float(d[r]), "key_frame": keys[r], "offset": offs[r],
                "second": ({"entry": int(order[1]), "tag": self.meta[order[1]]["tag"],
                            "dist": float(d[order[1]])} if len(order) > 1 else None)}
        if len(nn) == 1:
            return self.entries[r], offs[r], info
        info["knn"] = [{"entry": j, "tag": self.meta[j]["tag"], "dist": float(d[j])}
                       for j in nn]
        return self._average(nn), offs[r], info

    def _average(self, nn: list[int]) -> Waypoints:
        ref, mr = self.entries[nn[0]], self.marks[nn[0]]
        maps = [[map_frame(t, mr, self.marks[j]) for t in range(ref.T)] for j in nn]
        z_traj = np.mean([self.entries[j].z_traj[m] for j, m in zip(nn, maps)], axis=0)
        u_traj = np.mean([self.entries[j].u_traj[np.minimum(m, len(self.entries[j].u_traj) - 1)]
                          for j, m in zip(nn, maps)], axis=0)
        tol = np.array([max(self.entries[j].tol_at(map_frame(int(f), mr, self.marks[j]))
                            for j in nn) for f in ref.frames])
        tol[ref.tol < 0.0] = -1.0
        return Waypoints({"frames": ref.frames, "z_wp": z_traj[ref.frames], "tol": tol,
                          "max_duration_s": ref.max_duration_s, "z_traj": z_traj,
                          "u_traj": u_traj, "dt": ref.dt, "latch": ref.latch,
                          "first_hold": ref.first_hold, "deploy_sha256": ref.deploy_sha256,
                          "demo_episode": ref.demo_episode, "move0": ref.move0})


def load_actions(path) -> np.ndarray:
    """``(T, 12)`` demo actions of a waypoints artifact (``u_traj``) or an episode (``actions``)."""
    with np.load(Path(path), allow_pickle=True) as f:
        key = "u_traj" if "u_traj" in f.files else "actions"
        return np.asarray(f[key], np.float64)


def summary(doc: dict) -> str:
    lines = [(f"{len(doc['frames'])} waypoints on {doc['demo_episode']} (latch {doc['latch']}, "
              f"first hold {doc['first_hold']}, T {len(doc['z_traj'])}); stride {doc['stride']}, "
              f"window {doc['window']}, p{doc['percentile']:g}, floor {doc['floor']:g}; "
              f"{len(doc['demos'])} demos"
              + (f"; min frame {doc['min_frame']}" if "min_frame" in doc else "")),
             "| k | frame | tol | p raw | frames pooled | max s |",
             "|---|---|---|---|---|---|"]
    for k, f in enumerate(doc["frames"]):
        lines.append(f"| {k} | {f} | {doc['tol'][k]:.4f} | {doc['tol_percentile_raw'][k]:.4f} | "
                     f"{doc['n_tol_frames'][k]} | {doc['max_duration_s'][k]:.2f} |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deploy", type=Path, required=True)
    p.add_argument("--demos", type=Path, nargs="+", required=True)
    p.add_argument("--goal-demo", type=Path, default=None, help="one artifact (with --out)")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--percentile", type=float, default=90.0)
    p.add_argument("--floor", type=float, default=0.5)
    p.add_argument("--min-frame", type=int, default=0,
                   help="drop waypoints before this goal-demo frame (one artifact only)")
    p.add_argument("--out", type=Path, default=None, help="latent_waypoints.npz")
    p.add_argument("--bank-out", type=Path, default=None,
                   help="dir: one artifact per --goal-demos episode + bank.json")
    p.add_argument("--goal-demos", type=Path, nargs="+", default=None,
                   help="bank entries (default: --demos)")
    a = p.parse_args()
    one = a.goal_demo is not None and a.out is not None
    if (a.bank_out is None) != one or (a.bank_out is not None and (a.goal_demo or a.out)):
        p.error("--goal-demo + --out (one artifact) or --bank-out (bank)")
    if a.min_frame and a.bank_out is not None:
        p.error("--min-frame is for one artifact (the goal demo's clock)")
    if a.bank_out is not None:
        a.bank_out.mkdir(parents=True, exist_ok=True)
        bank = build_bank(a.deploy, a.demos, a.goal_demos or a.demos, a.bank_out, a.stride,
                          a.window, a.percentile, a.floor)
        text = bank_summary(bank)
        (a.bank_out / "bank.md").write_text(text + "\n")
        print(text)
        print(f"wrote {a.bank_out / 'bank.json'}")
        return 0
    doc = derive(a.deploy, a.demos, a.goal_demo, a.stride, a.window, a.percentile, a.floor,
                 min_frame=a.min_frame)
    save_npz(a.out, doc)
    text = summary(doc)
    a.out.with_suffix(".md").write_text(text + "\n")
    print(text)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
