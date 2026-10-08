#!/usr/bin/env python3
"""Offline comparison of seed models (PLAN-20261006-miles E3, V12-25). CPU, no sim.

Models: ``--model NAME=PATH`` (repeatable). PATH = a ``deploy.npz`` (``decoder.npz`` beside it), a
loop iteration dir / its ``deploy.json`` (+ ``latent_waypoints.npz``, ``model.json`` buffer), or a
checkpoint ``.pt`` with ``--export-buffer NAME=<its training buffer.json>``: exported here with the
loop's export commands (lcs_learning, CPU) and ``--export-box``. Every model runs the same frames.

Data (``--group NAME=GLOB``, repeatable; a glob of run ``index.json`` files or episode npz;
default the off-demo closed-loop rollouts of V12-17, V12-18 collects + evals, V12-20, V12-21):
episodes in a model's training buffer (iteration dirs, ``--train-buffer``) are dropped.
1. Prediction: start frames every ``--frame-stride`` (frames t..t+H non-empty), z_t = the
   deploy's numpy encoder on ``pcd_belt_seg``; one-step and H-step (7) open-loop LCS rollouts
   with the recorded actions (and P for EE-state models), decoded (``decoder.npz``, pulley frame)
   vs the resampled cloud at t+1 / t+H: Chamfer RMS mm (``refit_from_buffer.py``'s metric) and
   recon at t; split by group, phase and the distance from the demo tube. Phase (kinematic,
   per episode): sweep = from the first frame with Franka y >= 45 mm to its max (- 3 mm), seat =
   after, else descent if UR z <= 50 mm, else approach. Tube = the 6-d distance of [p_franka,
   p_ur] to the nearest frame of the 20 seed demos (mm).
2. Probes: from the final latent of every GT engaged ("seated") / "over" episode, roll a
   constant input for ``--probe-steps`` (16) steps of ``--probe-mm`` (3) mm: Franka +y (the
   "+y past the sweep" probe; the right answer is "worse"), Franka +-x, UR +-y (null directions)
   and zero; d = whitened distance to the deploy's seated goal; reported dd = d_16 - d_0 minus
   the zero-input change, and the fraction with dd > 0 (worse).
3. Behaviour (``behaviour_gate.py``'s metric; off with ``--no-behaviour``): MPPI u_0 on every
   ``--bg-stride``-th seed demo frame (one yaml for all, clearance dropped on the CPU unless
   ``--keep-clearance``, demo prior on), cos_median approach / descent / hold, hold push (u_0
   Franka y on hold frames, mm/step), pre-latch Franka y / UR descent. Waypoints: the iteration
   dir's, else built here with the loop's rule (``--wp-min-frame``).
Writes ``<out>/report.md`` + ``report.json`` (GT outcomes are labels only).

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/lcs/seed_compare.py \\
        --model V12h=data/lcs/realfirst/cmp/C1_transport/iter1 \\
        --model V12v=data/lcs/realfirst/cmp/v12_18/iter0 --out data/lcs/realfirst/cmp/seed_compare
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from task_common.belt_cloud import resample_deterministic
from task_common.latent_encoder import (
    BeltCloudPointNetEncoder,
    LatentDecoder,
    LatentEncoder,
)
from task_common.mppi_lcs import MppiParams, TorchLcs

LCS_REPO = Path.home() / "git/lcs_learning"
PY_LCS = LCS_REPO / ".venv/bin/python"
CMP = "data/lcs/realfirst/cmp"
GROUPS = {"V12-17": [f"{CMP}/sbox_sim/box/*/index.json", f"{CMP}/sbox_sim/boxsync/*/index.json"],
          "V12-18": [f"{CMP}/v12_18/iter*/collect/index.json",
                     f"{CMP}/v12_18/iter*/eval/*/index.json"],
          "V12-20": [f"{CMP}/ur_slant/nominal/index.json", f"{CMP}/ur_slant/ee_ur_z*/index.json",
                     f"{CMP}/ur_slant/sync_ee_ur_*/index.json",
                     f"{CMP}/ur_slant/box_ee_ur_rz*/index.json"],
          "V12-21": [f"{CMP}/ur_slant/pivot_*/index.json"]}
SEED_BUFFERS = ["data/lcs/realfirst/v1_1/seed/buffer.json",
                "data/lcs/realfirst/v1_1/seed_ext/buffer.json",
                f"{CMP}/demos_ext2/buffer.json"]
GOAL_DEMO = "data/lcs/realfirst/v1_1/seed/demo/episode_0000.npz"
WP_RULE = {"stride": 10, "window": 3, "percentile": 90, "floor": 0.5}
PHASES = ("approach", "descent", "sweep", "seat")
TUBE_MM = (0.0, 5.0, 10.0, 20.0, 40.0, math.inf)
PROBES = {"F+y": (1, 1.0), "F+x": (0, 1.0), "F-x": (0, -1.0), "UR+y": (4, 1.0),
          "UR-y": (4, -1.0)}
SWEEP_MM, UR_HIGH_MM = 45.0, 50.0
NPTS = 256
DTYPE = torch.float64


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def jload(p):
    return json.loads(Path(p).read_text())


def seed_demos() -> list[str]:
    return [r["episode"] for b in SEED_BUFFERS for r in jload(REPO / b)["rollouts"]
            if r["kind"] == "demo"]


# ---------------------------------------------------------------- models


def goal_frames_arg() -> str:
    """The loop's ``--goal-frames`` of the goal demo (realfirst_loop.goal_frames)."""
    from realfirst_loop import goal_frames

    gf = goal_frames(REPO / GOAL_DEMO)
    return f"--goal-frames={gf['frames'][0]},{gf['frames'][1] - gf['n_frames']}"


def export_checkpoint(name: str, ck: Path, buffer: Path, box: Path, out: Path) -> Path:
    """The loop's export + decoder commands on the CPU -> ``out/deploys/<name>/deploy.npz``."""
    dep = out / "deploys" / name
    if (dep / "deploy.npz").is_file() and (dep / "decoder.npz").is_file():
        return dep / "deploy.npz"
    files = out / "deploys" / f"{name}_files.txt"
    files.parent.mkdir(parents=True, exist_ok=True)
    files.write_text("\n".join(r["episode"] for r in jload(buffer)["rollouts"]) + "\n")
    env = {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"}
    for cmd in ([str(PY_LCS), "scripts/export_learned_lcs_deploy.py", "--checkpoint", str(ck),
                 "--data-files", str(files), "--goal-episode", str(REPO / GOAL_DEMO),
                 goal_frames_arg(), "--u-bounds-json", str(box), "--out-dir", str(dep)],
                [str(PY_LCS), "scripts/export_learned_lcs_deploy.py", "--checkpoint", str(ck),
                 "--decoder-out", str(dep / "decoder.npz")]):
        r = subprocess.run(cmd, cwd=LCS_REPO, env=env, capture_output=True, text=True,
                           check=False)
        if r.returncode:
            raise SystemExit(f"export of {name} failed: {r.stdout[-2000:]}{r.stderr[-2000:]}")
    return dep / "deploy.npz"


def resolve_model(spec: str, a, out: Path) -> dict:
    name, _, path = spec.partition("=")
    if not name or not path:
        raise SystemExit(f"--model {spec!r}: NAME=PATH")
    p = (REPO / path).resolve()
    m = {"name": name, "spec": path, "waypoints": None, "train_buffer": None}
    if p.is_dir() or p.name == "deploy.json":
        it = p if p.is_dir() else p.parent
        dj = jload(it / "deploy.json")
        m.update(deploy=dj["deploy"], decoder=dj["decoder"], model_id=dj["model"])
        if (it / "latent_waypoints.npz").is_file():
            m["waypoints"] = str(it / "latent_waypoints.npz")
        if (it / "model.json").is_file():
            m["train_buffer"] = jload(it / "model.json")["buffer"]
    elif p.suffix == ".pt":
        bufs = dict(x.split("=", 1) for x in a.export_buffer)
        if name not in bufs:
            raise SystemExit(f"--model {name} is a checkpoint: needs --export-buffer {name}=<buf>")
        m["train_buffer"] = str((REPO / bufs[name]).resolve())
        dep = export_checkpoint(name, p, Path(m["train_buffer"]), (REPO / a.export_box).resolve(),
                                out)
        m.update(deploy=str(dep), decoder=str(dep.parent / "decoder.npz"), checkpoint=str(p))
    else:
        m.update(deploy=str(p), decoder=str(p.parent / "decoder.npz"))
    tb = dict(x.split("=", 1) for x in a.train_buffer)
    if name in tb:
        m["train_buffer"] = str((REPO / tb[name]).resolve())
    for k in ("deploy", "decoder"):
        if not Path(m[k]).is_file():
            raise SystemExit(f"model {name}: {k} {m[k]} missing")
    return m


class Model:
    def __init__(self, m: dict) -> None:
        self.m = m
        self.enc = LatentEncoder.load(m["deploy"])
        if not isinstance(self.enc, BeltCloudPointNetEncoder):
            raise SystemExit(f"{m['name']}: not a belt_cloud_pointnet deploy")
        self.dec = LatentDecoder.load(m["decoder"])
        if self.dec.frame != self.enc.frame:
            raise SystemExit(f"{m['name']}: decoder frame {self.dec.frame} != {self.enc.frame}")
        self.lcs = TorchLcs.from_deploy(m["deploy"], "cpu", DTYPE)
        self.seat = torch.as_tensor(self.lcs.stage_goals[-1], dtype=DTYPE)
        self.z_std = torch.as_tensor(self.lcs.z_std_np, dtype=DTYPE)

    def encode(self, clouds: list) -> torch.Tensor:
        z = [self.enc.encode_batch(clouds[i:i + 256]) for i in range(0, len(clouds), 256)]
        return torch.as_tensor(np.concatenate(z), dtype=DTYPE)

    def step(self, z, u, p):
        return self.lcs.step(z, u, p if self.lcs.has_p else None)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.dec.decode_batch(z.numpy()), dtype=DTYPE)

    def d_seat(self, z: torch.Tensor) -> torch.Tensor:
        return ((z - self.seat) / self.z_std).norm(dim=-1)


# ---------------------------------------------------------------- data


def group_episodes(specs: list[str]) -> dict[str, list[dict]]:
    groups = {}
    items = [s.split("=", 1) for s in specs] if specs else [(k, g) for k, v in GROUPS.items()
                                                             for g in v]
    for name, pat in items:
        rows = groups.setdefault(name, [])
        for f in sorted(glob.glob(str(REPO / pat))):
            f = Path(f)
            if f.suffix == ".npz":
                rows.append({"episode": str(f.resolve()), "outcome": None, "run": rel(f.parent)})
                continue
            idx = jload(f)
            if "summary" not in idx:
                continue  # a run still going
            for e in idx["episodes"]:
                if e.get("file"):
                    rows.append({"episode": str((f.parent / e["file"]).resolve()),
                                 "outcome": e.get("outcome"), "run": rel(f.parent)})
    return groups


def demo_points() -> torch.Tensor:
    ps = []
    for e in seed_demos():
        with np.load(e, allow_pickle=True) as d:
            ps.append(np.c_[d["sim_ee_franka"][:, :3], d["sim_ee_ur"][:, :3]] * 1e3)
    return torch.as_tensor(np.concatenate(ps), dtype=DTYPE)


def phases(fy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    T = len(fy)
    ph = np.where(uz <= UR_HIGH_MM, "descent", "approach").astype(object)
    i45 = np.flatnonzero(fy >= SWEEP_MM)
    if len(i45):
        imax = int(np.flatnonzero(fy >= fy.max() - 3.0)[0])
        ph[i45[0]:] = "sweep"
        ph[max(imax, i45[0]):] = "seat"
    return ph[:T]


def load_episode(e: dict, H: int, stride: int, demo_p: torch.Tensor) -> dict | None:
    with np.load(e["episode"], allow_pickle=True) as d:
        seg = list(d["pcd_belt_seg"])
        u = np.asarray(d["actions"], np.float64)
        F, U = np.asarray(d["sim_ee_franka"])[:, :3], np.asarray(d["sim_ee_ur"])[:, :3]
        out = str(d["sim_outcome"]) if "sim_outcome" in d.files else None
    T = len(seg)
    pcs, empty = zip(*(resample_deterministic(s, NPTS) for s in seg))
    empty = np.asarray(empty, bool)
    ok = np.array([not empty[t:t + H + 1].any() for t in range(T - H)], bool)
    starts = np.flatnonzero(ok)
    starts = starts[starts % stride == 0]
    P = np.c_[F, U]
    tube = torch.cdist(torch.as_tensor(P * 1e3, dtype=DTYPE), demo_p).min(dim=1).values.numpy()
    last = int(np.flatnonzero(~empty)[-1]) if (~empty).any() else None
    return {**e, "outcome": e["outcome"] or out, "seg": seg, "pc": np.stack(pcs), "u": u, "P": P,
            "starts": starts, "phase": phases(F[:, 1] * 1e3, U[:, 2] * 1e3), "tube": tube,
            "last": last}


def chamfer_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(B,) refit_from_buffer's symmetric Chamfer (m^2)."""
    out = []
    for i in range(0, len(a), 64):
        d2 = torch.cdist(a[i:i + 64], b[i:i + 64]).pow(2)
        out.append(0.5 * (d2.min(dim=2).values.mean(1) + d2.min(dim=1).values.mean(1)))
    return torch.cat(out) if out else torch.zeros(0, dtype=DTYPE)


# ---------------------------------------------------------------- metrics


def prediction(model: Model, eps: list[dict], H: int) -> dict:
    """Per start frame: one-step, H-step and recon Chamfer squared (m^2)."""
    rows = {"one": [], "hk": [], "rec": [], "group": [], "phase": [], "tube": []}
    for e in eps:
        t = e["starts"]
        if not len(t):
            continue
        z = model.encode([e["seg"][k] for k in t])
        u, p, pc = (torch.as_tensor(e[k], dtype=DTYPE) for k in ("u", "P", "pc"))
        rows["rec"].append(chamfer_sq(model.decode(z), pc[t]))
        zk = model.step(z, u[t], p[t])
        rows["one"].append(chamfer_sq(model.decode(zk), pc[t + 1]))
        for k in range(1, H):
            zk = model.step(zk, u[t + k], p[t + k])
        rows["hk"].append(chamfer_sq(model.decode(zk), pc[t + H]))
        rows["group"] += [e["group"]] * len(t)
        rows["phase"] += list(e["phase"][t])
        rows["tube"].append(e["tube"][t])
    return {"one": torch.cat(rows["one"]).numpy(), "hk": torch.cat(rows["hk"]).numpy(),
            "rec": torch.cat(rows["rec"]).numpy(), "group": np.asarray(rows["group"]),
            "phase": np.asarray(rows["phase"]), "tube": np.concatenate(rows["tube"])}


def rms_mm(x: np.ndarray) -> float | None:
    return float(1e3 * math.sqrt(x.mean())) if len(x) else None


def split_table(pr: dict) -> dict:
    out = {"all": {k: rms_mm(pr[k]) for k in ("one", "hk", "rec")} | {"n": len(pr["one"])}}
    for name, mask in ([(f"group {g}", pr["group"] == g) for g in dict.fromkeys(pr["group"])]
                       + [(f"phase {p}", pr["phase"] == p) for p in PHASES]
                       + [(f"tube {lo:g}-{hi:g} mm", (pr["tube"] >= lo) & (pr["tube"] < hi))
                          for lo, hi in itertools.pairwise(TUBE_MM)]):
        out[name] = {k: rms_mm(pr[k][mask]) for k in ("one", "hk", "rec")} | {
            "n": int(mask.sum())}
    return out


def roll_dd(model: Model, z0, p0, ch, sign: float, steps: int, mm: float) -> torch.Tensor:
    """Seated-goal distance change after ``steps`` constant inputs (channel ``ch``, None = 0)."""
    u = torch.zeros(len(z0), model.lcs.n_u, dtype=DTYPE)
    if ch is not None:
        u[:, ch] = sign * mm * 1e-3
    z, p = z0, p0
    for _ in range(steps):
        z = model.step(z, u, p)
        p = p + u[:, :6]
    return model.d_seat(z) - model.d_seat(z0)


def probes(model: Model, states: list[dict], steps: int, mm: float) -> dict:
    """dd per direction (d_steps - d_0 minus the zero-input change) per state group."""
    out = {}
    for grp in ("seated", "over"):
        ss = [s for s in states if s["kind"] == grp]
        if not ss:
            continue
        z0 = model.encode([s["seg"] for s in ss])
        p0 = torch.as_tensor(np.stack([s["P"] for s in ss]), dtype=DTYPE)
        d0 = model.d_seat(z0)
        base = roll_dd(model, z0, p0, None, 0.0, steps, mm)
        res = {"n": len(ss), "d0_median": float(d0.median()),
               "zero_dd_median": float(base.median())}
        for name, (ch, sign) in PROBES.items():
            dd = (roll_dd(model, z0, p0, ch, sign, steps, mm) - base).numpy()
            res[name] = {"dd_median": float(np.median(dd)), "frac_worse": float((dd > 0).mean())}
        out[grp] = res
    return out


def probe_states(eps: list[dict]) -> list[dict]:
    out = []
    for e in eps:
        k = {"engaged": "seated", "over": "over"}.get(e["outcome"])
        if k is None or e["last"] is None:
            continue
        out.append({"kind": k, "seg": e["seg"][e["last"]], "P": e["P"][e["last"]],
                    "episode": e["episode"]})
    return out


def waypoints_for(m: dict, a, out: Path) -> str:
    if m["waypoints"] and a.wp_min_frame is None:
        return m["waypoints"]
    wp = out / "waypoints" / f"{m['name']}{'' if a.wp_min_frame is None else '_min' + str(a.wp_min_frame)}.npz"
    if not wp.is_file():
        wp.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "scripts/lcs/latent_waypoints.py", "--deploy", m["deploy"],
               "--demos", *seed_demos(), "--goal-demo", GOAL_DEMO,
               *[x for k, v in WP_RULE.items() for x in (f"--{k}", str(v))],
               *(["--min-frame", str(a.wp_min_frame)] if a.wp_min_frame is not None else []),
               "--out", str(wp)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
        if r.returncode:
            raise SystemExit(f"latent_waypoints.py for {m['name']} failed: {r.stderr[-2000:]}")
    return str(wp)


def behaviour(models: list[dict], a, out: Path) -> dict:
    from behaviour_gate import frame_list, phase_rows, prelatch_stats, run_model
    from model_gates import load_demo

    base = MppiParams.from_yaml(REPO / a.mppi_config)
    kw = {} if a.keep_clearance else {"clearance": None}
    if a.u_box_json:
        bx = jload(REPO / a.u_box_json)
        kw.update(u_lb=list(bx["u_lb"]), u_ub=list(bx["u_ub"]))
    base = dataclasses.replace(base, **kw)
    dem = seed_demos()
    if a.bg_demos:
        dem = [dem[i] for i in np.linspace(0, len(dem) - 1, a.bg_demos).round().astype(int)]
    demos = [load_demo(Path(d)) for d in dem]
    frames = frame_list(demos, [], a.bg_stride)
    res = {"mppi_config": a.mppi_config, "clearance": a.keep_clearance,
           "u_box_json": a.u_box_json, "n_demos": len(demos), "n_frames": len(frames),
           "stride": a.bg_stride, "models": {}}
    for m in models:
        wp = waypoints_for(m, a, out)
        t0 = time.perf_counter()
        u0, meta = run_model(Path(m["deploy"]), Path(wp), base, demos, [], frames, a.seed, "cpu")
        rows = phase_rows(m["name"], base.H, u0, frames)
        pre = prelatch_stats(u0, frames)
        res["models"][m["name"]] = {
            "waypoints": rel(wp), "prior": meta["prior"],
            "cos_median": {ph: (rows.get(ph) or {}).get("cos_median")
                           for ph in ("approach", "descent", "hold")},
            "n": {ph: (rows.get(ph) or {}).get("n") for ph in ("approach", "descent", "hold")},
            "hold_push_mm": (rows.get("hold") or {}).get("u0_fy_mm"),
            "prelatch_fy_mm": pre["fy_mm"], "prelatch_ur_down_mm": pre["ur_down_mm"],
            "wall_s": time.perf_counter() - t0}
        print(f"  behaviour {m['name']}: {res['models'][m['name']]}", flush=True)
    return res


# ---------------------------------------------------------------- report


def f(x, s=".2f") -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, s)


def report_md(doc: dict) -> str:
    names = [m["name"] for m in doc["models"]]
    grp = ", ".join(f"{g} {n}" for g, n in doc["groups"].items())
    md = ["# Seed model comparison (`seed_compare.py`)", "",
          "Models: " + "; ".join(f"{m['name']} = `{m['deploy']}`"
                                  + (f" ({m['model_id']})" if m.get("model_id") else "")
                                  for m in doc["models"]) + ".",
          (f"Data: {doc['n_episodes']} episodes ({grp}),"
           f" every {doc['args']['frame_stride']} frames, H {doc['args']['horizon']}; "
           f"dropped (in a training buffer): {doc['n_dropped']}."), "",
          "## Prediction (Chamfer RMS mm: one-step / H-step / recon)", "",
          "| split | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    keys = list(doc["prediction"][names[0]].keys())
    for k in keys:
        n = doc["prediction"][names[0]][k]["n"]
        md.append(f"| {k} | {n} | " + " | ".join(
            f"{f(doc['prediction'][nm][k]['one'])} / {f(doc['prediction'][nm][k]['hk'])} / "
            f"{f(doc['prediction'][nm][k]['rec'])}" for nm in names) + " |")
    md += ["", (f"## Probes ({doc['args']['probe_steps']} steps x {doc['args']['probe_mm']:g} "
                "mm; dd = change of the whitened seated-goal distance minus the zero-input "
                "change; worse = dd > 0)"), "",
           "| model | states | n | d0 | zero dd | " + " | ".join(
               f"{p} dd (worse)" for p in PROBES) + " |", "|---|---|---|---|---|"
           + "---|" * len(PROBES)]
    for nm in names:
        for grp, r in doc["probes"][nm].items():
            md.append(f"| {nm} | {grp} | {r['n']} | {f(r['d0_median'])} | "
                      f"{f(r['zero_dd_median'])} | " + " | ".join(
                          f"{f(r[p]['dd_median'], '+.2f')} ({f(r[p]['frac_worse'], '.0%')})"
                          for p in PROBES) + " |")
    if doc.get("behaviour"):
        b = doc["behaviour"]
        md += ["", (f"## Behaviour (MPPI u_0 on seed demo frames, stride {b['stride']}, "
                    f"{b['n_demos']} demos, {b['n_frames']} frames; `{b['mppi_config']}`, "
                    f"clearance {'on' if b['clearance'] else 'off'}, box "
                    f"{b['u_box_json'] or 'each deploy'})"), "",
               ("| model | cos approach | cos descent | cos hold | hold push mm/step | pre-latch "
                "F y / UR down mm/step |"), "|---|---|---|---|---|---|"]
        for nm in names:
            r = b["models"][nm]
            md.append(f"| {nm} | {f(r['cos_median']['approach'], '.3f')} | "
                      f"{f(r['cos_median']['descent'], '.3f')} | "
                      f"{f(r['cos_median']['hold'], '.3f')} | {f(r['hold_push_mm'], '+.2f')} | "
                      f"{f(r['prelatch_fy_mm'], '+.2f')} / {f(r['prelatch_ur_down_mm'], '+.2f')} |")
    return "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH")
    ap.add_argument("--group", action="append", default=[], help="NAME=GLOB (index.json | npz)")
    ap.add_argument("--train-buffer", action="append", default=[], help="NAME=buffer.json")
    ap.add_argument("--export-buffer", action="append", default=[],
                    help="NAME=buffer.json of a checkpoint model (export data files)")
    ap.add_argument("--export-box", default=f"{CMP}/v12_18/iter0/box_0.json")
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--max-episodes", type=int, default=None, help="per group (seeded pick)")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--probe-steps", type=int, default=16)
    ap.add_argument("--probe-mm", type=float, default=3.0)
    ap.add_argument("--no-behaviour", action="store_true")
    ap.add_argument("--bg-stride", type=int, default=8)
    ap.add_argument("--bg-demos", type=int, default=None, help="subsample the 20 seed demos")
    ap.add_argument("--mppi-config", default=f"{CMP}/sbox_sim/mppi_box.yaml")
    ap.add_argument("--keep-clearance", action="store_true")
    ap.add_argument("--u-box-json", default=None, help="one planner box for every model")
    ap.add_argument("--wp-min-frame", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{CMP}/seed_compare")
    a = ap.parse_args()
    t0 = time.perf_counter()
    torch.set_grad_enabled(False)
    out = (REPO / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = [resolve_model(s, a, out) for s in a.model]
    if len({m["name"] for m in models}) != len(models):
        raise SystemExit("model names must be unique")
    train = set()
    for m in models:
        if m["train_buffer"]:
            train |= {r["episode"] for r in jload(m["train_buffer"])["rollouts"]}
    rng = np.random.default_rng(a.seed)
    groups = group_episodes(a.group)
    demo_p = demo_points()
    eps, n_drop, counts = [], 0, {}
    for g, rows in groups.items():
        keep = [r for r in rows if r["episode"] not in train]
        n_drop += len(rows) - len(keep)
        if a.max_episodes and len(keep) > a.max_episodes:
            keep = [keep[i] for i in sorted(rng.choice(len(keep), a.max_episodes, replace=False))]
        counts[g] = len(keep)
        for r in keep:
            e = load_episode(r, a.horizon, a.frame_stride, demo_p)
            e["group"] = g
            eps.append(e)
        print(f"  {g}: {len(keep)} episodes ({len(rows) - len(keep)} in a training buffer"
              f"{' / not picked' if a.max_episodes else ''})", flush=True)
    if not eps:
        raise SystemExit("no held-out episodes")
    states = probe_states(eps)
    doc = {"args": vars(a), "models": models, "groups": counts, "n_episodes": len(eps),
           "n_dropped": n_drop, "n_probe_states": {k: sum(s["kind"] == k for s in states)
                                                   for k in ("seated", "over")},
           "prediction": {}, "probes": {}}
    for m in models:
        mod = Model(m)
        t1 = time.perf_counter()
        doc["prediction"][m["name"]] = split_table(prediction(mod, eps, a.horizon))
        doc["probes"][m["name"]] = probes(mod, states, a.probe_steps, a.probe_mm)
        al = doc["prediction"][m["name"]]["all"]
        print(f"  {m['name']}: one-step {f(al['one'])} / H{a.horizon} {f(al['hk'])} / recon "
              f"{f(al['rec'])} mm on {al['n']} frames ({time.perf_counter() - t1:.0f} s)",
              flush=True)
    if not a.no_behaviour:
        doc["behaviour"] = behaviour(models, a, out)
    doc["wall_s"] = time.perf_counter() - t0
    (out / "report.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    md = report_md(doc)
    (out / "report.md").write_text(md)
    print(md)
    print(f"wrote {rel(out / 'report.md')} ({doc['wall_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
