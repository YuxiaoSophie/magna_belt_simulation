#!/usr/bin/env python3
"""V12-40 offline evidence for the C3+ path (CPU, recorded data only; no sim, no binary).

``lcp``: C3 plans on the exact LCP (``0 <= lam _|_ E z + F lam + H u + c >= 0``); the model was
trained through a fixed 25-iteration PGD. On recorded task states (goal demo + rollouts of the
deploy: ``sim_latent`` + ``actions``) compare the exact step (active-set enumeration, n_lam 8)
with PGD-25 and the loose PGD (100 it, tol 1e-5; check_latent_encoder E3), per step and over an
N-step open-loop rollout of the recorded actions, in whitened units.
``solve-times``: the C++ controller's sampled ``solve=<ms>`` log lines of every earlier C3+ run
(``data/lcs/mpc_eval``), grouped by n_x / C3 n_z / N / admm_iter, vs the 75 ms latent period.
``box-loss``: what V12-16's support box and the clearance did in MPPI runs, split into what the
C++ ee_constraints can express (EE z faces, hard) and what they cannot (x / y faces, clearance,
rotation envelope): executed EE outside / near each face, MPPI per-solve flags.

    uv run --frozen python scripts/lcs/c3plus_offline.py lcp --out data/lcs/realfirst/cmp/c3plus/lcp
    uv run --frozen python scripts/lcs/c3plus_offline.py solve-times --root data/lcs \
        --out data/lcs/realfirst/cmp/c3plus/solve_times
    uv run --frozen python scripts/lcs/c3plus_offline.py box-loss \
        --out data/lcs/realfirst/cmp/c3plus/box_loss
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from task_common.latent_encoder import LearnedLcs

WORKTREE = Path("/home/hienbui/git/magna-deploy-learned-lcs")
CMP = REPO / "data/lcs/realfirst/cmp"
PERIOD_MS = 75.0
FACES = [f"{arm} {ax} {side}" for arm in ("F", "UR") for ax in "xyz" for side in ("lo", "hi")]


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def pct(a, qs=(50, 95, 99)) -> dict:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), **{f"p{q}": float(np.percentile(a, q)) for q in qs},
            "max": float(a.max()), "mean": float(a.mean())}


# ---------------------------------------------------------------- lcp


def exact_lambda(lcs: LearnedLcs, z, u) -> tuple[np.ndarray, int]:
    """All solutions of the LCP with M = F_pgd (the exported F) by active-set enumeration;
    returns the first and the count (a P-matrix gives exactly one)."""
    q = lcs.E @ z + lcs.H @ u + lcs.c
    M, n = lcs.F_pgd, lcs.n_lam
    sols = []
    for r in range(n + 1):
        for S in itertools.combinations(range(n), r):
            lam = np.zeros(n)
            if S:
                idx = list(S)
                try:
                    lam[idx] = np.linalg.solve(M[np.ix_(idx, idx)], -q[idx])
                except np.linalg.LinAlgError:
                    continue
            w = M @ lam + q
            if lam.min() >= -1e-10 and w.min() >= -1e-10:
                sols.append(np.maximum(lam, 0.0))
    return (sols[0] if sols else np.full(n, np.nan)), len(sols)


def lcp_episode(lcs: LearnedLcs, Z, U, N: int) -> dict:
    zs = lcs.z_std
    one = {"pgd25": [], "loose": [], "n_sol": [], "active": []}
    for z, u in zip(Z[:-1], U[:-1]):
        le, ns = exact_lambda(lcs, z, u)
        ze = lcs.A @ z + lcs.B @ u + lcs.D @ le + lcs.d
        z25 = lcs.step(z, u)[0]
        zl = lcs.step(z, u, iters=100, tol=1e-5)[0]
        one["pgd25"].append(np.linalg.norm((z25 - ze) / zs))
        one["loose"].append(np.linalg.norm((zl - ze) / zs))
        one["n_sol"].append(ns)
        one["active"].append(int((le > 1e-9).sum()))
    roll = []
    for t in range(0, len(Z) - N - 1, 3):
        a = b = Z[t]
        for k in range(N):
            a = lcs.step(a, U[t + k])[0]
            le, _ = exact_lambda(lcs, b, U[t + k])
            b = lcs.A @ b + lcs.B @ U[t + k] + lcs.D @ le + lcs.d
        roll.append(np.linalg.norm((a - b) / zs))
    return {**{k: np.asarray(v) for k, v in one.items()}, "roll": np.asarray(roll)}


def cmd_lcp(a) -> int:
    out = (REPO / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    md = ["# V12-40: exact LCP (C3) vs the trained PGD-25 step on task states", "",
          ("Whitened ||z_next(PGD) - z_next(exact)|| per step, and after an N = 7 open-loop "
           "rollout of the recorded actions (every 3rd frame). Exact = active-set enumeration "
           "with the exported F (= G G^T + stiffness I, PD: one solution)."), "",
          ("| model | states | solutions = 1 | active lam p50 / max | PGD-25 step p50 / p99 / max |"
           " loose (E3) step max | PGD-25 vs exact at k = 7 p50 / p95 / max | stage-0 tol |"),
          "|" + "---|" * 8]
    for spec in a.model:
        name, dj, coll = spec.split(":", 2)
        it = (REPO / dj).parent
        d = json.loads((REPO / dj).read_text())
        lcs = LearnedLcs.load(d["deploy"])
        with np.load(it / "latent_waypoints.npz", allow_pickle=False) as f:
            eps = [("goal_demo", np.asarray(f["z_traj"]), np.asarray(f["u_traj"]))]
            tol0 = float(f["tol"][list(f["frames"]).index(int(f["latch"]))])
        for p in sorted((REPO / coll).glob("episode_*.npz"))[:a.max_episodes]:
            with np.load(p, allow_pickle=True) as e:
                if "sim_latent" in e.files:
                    eps.append((p.name, np.asarray(e["sim_latent"], float),
                                np.asarray(e["actions"], float)))
        agg = {"pgd25": [], "loose": [], "n_sol": [], "active": [], "roll": []}
        per = {}
        for nm, Z, U in eps:
            r = lcp_episode(lcs, Z, U, a.N)
            for k, v in agg.items():
                v.append(r[k])
            per[nm] = {"frames": len(Z), "pgd25_max": float(r["pgd25"].max()),
                       "roll_p95": float(np.percentile(r["roll"], 95)) if r["roll"].size else None}
        agg = {k: np.concatenate(v) for k, v in agg.items()}
        res[name] = {"deploy": d["deploy"], "model": d.get("model"), "episodes": per,
                     "states": int(agg["pgd25"].size),
                     "unique": int((agg["n_sol"] == 1).sum()),
                     "active": pct(agg["active"]), "pgd25": pct(agg["pgd25"]),
                     "loose": pct(agg["loose"]), "roll7": pct(agg["roll"]), "stage0_tol": tol0}
        r = res[name]
        md.append(f"| {name} ({d.get('model')}) | {r['states']} ({len(eps)} episodes) | "
                  f"{r['unique']}/{r['states']} | {r['active']['p50']:.0f} / "
                  f"{r['active']['max']:.0f} | {r['pgd25']['p50']:.2e} / {r['pgd25']['p99']:.2e}"
                  f" / {r['pgd25']['max']:.2e} | {r['loose']['max']:.2e} | "
                  f"{r['roll7']['p50']:.2e} / {r['roll7']['p95']:.2e} / {r['roll7']['max']:.2e}"
                  f" | {tol0:.2f} |")
    (out / "lcp.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "lcp.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


# ---------------------------------------------------------------- solve times

_HEAD = re.compile(r"\[learned-mpc\] (C3\+|QP) over (\S+) .*?n_x=(\d+) n_u=\d+ n_lam=(\d+) "
                   r"N=(\d+) dt=\S+ n_z=(\d+)")
_SOLVE = re.compile(r"\[learned-mpc\] t=\S+ stage=\d+ .*solve=([0-9.]+)ms")


def learned_block_of(pp: Path, want_sha: str | None) -> dict:
    """The params' learned_mpc block; a moved file is found in learned_archive/<date>/<its old
    path under systems/parameters/> (sha256-matched when the run recorded one)."""
    cands = [pp]
    sp = "systems/parameters/"
    if sp in str(pp):
        tail = str(pp).split(sp, 1)[1]
        cands += sorted((WORKTREE / sp / "learned_archive").glob(f"*/{tail}"))
    cands = [c for c in cands if c.is_file()]
    # archived copies had lcs_file rewritten: prefer a sha match, else the first copy
    exact = [c for c in cands if want_sha and hashlib.sha256(c.read_bytes()).hexdigest()
             == want_sha]
    for c in exact or cands[:1]:
        try:
            return (yaml.safe_load(c.read_text()) or {}).get("learned_mpc") or {}
        except yaml.YAMLError:
            return {}
    return {}


def cmd_solve_times(a) -> int:
    out = (REPO / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, dict] = {}
    params_cache: dict[str, dict] = {}
    for idx in sorted((REPO / a.root).rglob("index.json")):
        try:
            ix = json.loads(idx.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        c = ix.get("controller") or {}
        if c.get("mode") == "mppi" or not c.get("params"):
            continue
        pp = Path(c.get("magna_root") or WORKTREE) / c["params"]
        key_p = str(pp)
        if key_p not in params_cache:
            params_cache[key_p] = learned_block_of(pp, c.get("params_sha256"))
        lm = params_cache[key_p]
        if not lm:
            continue
        for log in sorted((idx.parent / "controller").glob("*.log")):
            text = log.read_text(errors="replace")
            h = _HEAD.search(text)
            if h is None:
                continue
            ms = [float(x) for x in _SOLVE.findall(text)]
            k = (h.group(1), int(h.group(3)), int(h.group(6)), int(h.group(5)),
                 int(lm.get("admm_iter", 3)), bool(lm.get("demo_traj")))
            g = groups.setdefault(k, {"ms": [], "runs": set(), "logs": 0})
            g["ms"] += ms
            g["runs"].add(rel(idx.parent))
            g["logs"] += 1
    rows = []
    md = ["# C3+ solve time in earlier closed-loop runs (C++ controller logs)", "",
          ("`solve=` is logged about once per second (every ~14th solve), so these are "
           "samples. Period = 75 ms (one latent per 15 sim steps, pace 1)."), "",
          ("| projection | n_x | C3 n_z | N | admm_iter | demo_traj | logs | samples | p50 ms | "
           "p95 ms | p99 ms | max ms | > 75 ms |"), "|" + "---|" * 13]
    for k in sorted(groups):
        g = groups[k]
        s = pct(g["ms"])
        if not s["n"]:
            continue
        over = float(np.mean(np.asarray(g["ms"]) > PERIOD_MS))
        rows.append({"projection": k[0], "n_x": k[1], "n_z": k[2], "N": k[3],
                     "admm_iter": k[4], "demo_traj": k[5], "logs": g["logs"],
                     "runs": sorted(g["runs"]), **s, "frac_over_75ms": over})
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {k[5]} | {g['logs']} | "
                  f"{s['n']} | {s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | "
                  f"{s['max']:.1f} | {100 * over:.1f} % |")
    (out / "solve_times.json").write_text(json.dumps(rows, indent=1) + "\n")
    (out / "solve_times.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


# ---------------------------------------------------------------- box loss


def run_box(idx: dict, fallback: dict | None) -> tuple[dict | None, str]:
    cfg = (idx.get("controller") or {}).get("config")
    if cfg and Path(cfg).is_file():
        sb = (yaml.safe_load(Path(cfg).read_text()) or {}).get("support_box")
        if sb:
            return sb, "own"
    return fallback, "reference"


def cmd_box_loss(a) -> int:
    out = (REPO / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_box = json.loads((REPO / a.reference_box).read_text())
    res = {}
    md = ["# V12-40: what V12-16's support box and the clearance did in MPPI runs", "",
          ("Executed EE positions (sim_ee_*, every frame) vs the run's support box (its MPPI "
           "yaml; box-off arms: the reference box). out = frames outside the face; near = "
           "inside within 3 mm of it; hinge = mm outside summed over frames. C++ "
           "ee_constraints can express the z faces (hard), not x / y. Per solve (MPPI JSONL): "
           "the chosen plan leaves the box (sbox_mm[1] > 0); its min clearance margin over the "
           "set minimum (UR 3 / Franka 2 mm); violated (clear_violating > 0); the share of "
           "the K samples the clearance / rotation-envelope penalty hit."), ""]
    for spec in a.group:
        label, box_mode, *roots = spec.split(":")
        frames = 0
        out_f = np.zeros(12)
        near_f = np.zeros(12)
        hinge = np.zeros(12)
        solves = {"n": 0, "sbox_plan_out": 0, "clear_near": 0, "clear_violating": 0}
        clear_now, clear_plan, frac_clear, frac_env = [], [], [], []
        any_xy = any_z = 0
        n_ep, boxes = 0, set()
        for root in roots:
            for idx_p in sorted((REPO / root).rglob("index.json")):
                if ".partial" in str(idx_p) or ".prev" in str(idx_p):
                    continue
                ix = json.loads(idx_p.read_text())
                if (ix.get("controller") or {}).get("mode") != "mppi":
                    continue
                sb, src = run_box(ix, ref_box)
                if box_mode == "off" and src == "own":
                    continue
                if box_mode == "on" and src != "own":
                    continue
                lo, hi = np.asarray(sb["lo"], float), np.asarray(sb["hi"], float)
                boxes.add((tuple(np.round(lo * 1e3, 1)), tuple(np.round(hi * 1e3, 1))))
                for ep in ix.get("episodes", []):
                    if not ep.get("file"):
                        continue
                    with np.load(idx_p.parent / ep["file"], allow_pickle=True) as e:
                        P = np.concatenate([np.asarray(e["sim_ee_franka"])[:, :3],
                                            np.asarray(e["sim_ee_ur"])[:, :3]], axis=1)
                    n_ep += 1
                    frames += len(P)
                    below, above = (lo - P) * 1e3, (P - hi) * 1e3
                    o = np.stack([below, above], axis=2).reshape(len(P), 12) > 0
                    any_xy += int(o[:, [0, 1, 2, 3, 6, 7, 8, 9]].any(1).sum())
                    any_z += int(o[:, [4, 5, 10, 11]].any(1).sum())
                    for j in range(6):
                        out_f[2 * j] += (below[:, j] > 0).sum()
                        out_f[2 * j + 1] += (above[:, j] > 0).sum()
                        near_f[2 * j] += ((below[:, j] <= 0) & (below[:, j] > -3)).sum()
                        near_f[2 * j + 1] += ((above[:, j] <= 0) & (above[:, j] > -3)).sum()
                        hinge[2 * j] += np.clip(below[:, j], 0, None).sum()
                        hinge[2 * j + 1] += np.clip(above[:, j], 0, None).sum()
                    sl = (ep.get("mppi") or {}).get("solve_log")
                    if sl and (idx_p.parent / sl).is_file():
                        for line in (idx_p.parent / sl).read_text().splitlines():
                            if not line.strip():
                                continue
                            r = json.loads(line)
                            solves["n"] += 1
                            solves["sbox_plan_out"] += bool((r.get("sbox_mm") or [0, 0])[1] > 0)
                            solves["clear_violating"] += bool(r.get("clear_violating", 0) > 0)
                            frac_clear.append(r.get("frac_clear_penalised", np.nan))
                            frac_env.append(r.get("frac_env_penalised", np.nan))
                            if "clear_min_ur_mm" in r:  # chosen plan, margin to the set minimum
                                m = min(r["clear_min_ur_mm"] - a.ur_min_mm,
                                        r["clear_min_f_mm"] - a.franka_min_mm)
                                clear_plan.append(m)
                                solves["clear_near"] += bool(m < 2.0)
                            if "clear_now_ur_mm" in r:
                                clear_now.append(min(r["clear_now_ur_mm"], r["clear_now_f_mm"]))
        if not frames:
            continue
        z_idx = [4, 5, 10, 11]
        res[label] = {"roots": roots, "box": box_mode, "episodes": n_ep, "frames": frames,
                      "boxes_mm": sorted(boxes)[:4], "n_boxes": len(boxes),
                      "out_frac": dict(zip(FACES, (out_f / frames).tolist())),
                      "near_frac": dict(zip(FACES, (near_f / frames).tolist())),
                      "hinge_mm_frames": dict(zip(FACES, hinge.tolist())),
                      "hinge_share_z": float(hinge[z_idx].sum() / max(hinge.sum(), 1e-12)),
                      "frames_out_xy": any_xy / frames, "frames_out_z": any_z / frames,
                      "solves": solves, "clear_now_min_mm": pct(clear_now, (1, 5, 50)),
                      "clear_plan_margin_mm": pct(clear_plan, (1, 5, 50)),
                      "frac_samples_clear_penalised": pct(frac_clear, (50, 95)),
                      "frac_samples_env_penalised": pct(frac_env, (50, 95))}
        r = res[label]
        sv = solves["n"] or 1
        md += [f"## {label} ({box_mode}: {n_ep} episodes, {frames} frames; {len(boxes)} box(es))",
               "", "| face | out % frames | near % frames | hinge mm·frames |", "|---|---|---|---|"]
        for i, f in enumerate(FACES):
            md.append(f"| {f}{' (C3+: hard)' if i in z_idx else ''} | "
                      f"{100 * out_f[i] / frames:.1f} | {100 * near_f[i] / frames:.1f} | "
                      f"{hinge[i]:.0f} |")
        fc, fe, cp = (r["frac_samples_clear_penalised"], r["frac_samples_env_penalised"],
                      r["clear_plan_margin_mm"])
        md += ["", (f"- frames outside an x / y face {100 * any_xy / frames:.1f} %, a z face "
                    f"{100 * any_z / frames:.1f} %; z share of the hinge "
                    f"{100 * r['hinge_share_z']:.1f} %."),
               (f"- per solve ({solves['n']}): chosen plan leaves the box "
                f"{100 * solves['sbox_plan_out'] / sv:.1f} %; plan clearance margin < 2 mm "
                f"{100 * solves['clear_near'] / sv:.1f} % (margin p1 {cp.get('p1', np.nan):.1f}"
                f" / p5 {cp.get('p5', np.nan):.1f} / p50 {cp.get('p50', np.nan):.1f} mm), "
                f"violating {100 * solves['clear_violating'] / sv:.1f} %; share of MPPI "
                f"samples penalised by clearance p50 {fc.get('p50', np.nan):.3f} / p95 "
                f"{fc.get('p95', np.nan):.3f}, by the rotation envelope p50 "
                f"{fe.get('p50', np.nan):.3f} / p95 {fe.get('p95', np.nan):.3f}."),
               (f"- clearance now (min of UR / Franka, mm): p1 "
                f"{r['clear_now_min_mm'].get('p1', float('nan')):.1f}, p5 "
                f"{r['clear_now_min_mm'].get('p5', float('nan')):.1f}, p50 "
                f"{r['clear_now_min_mm'].get('p50', float('nan')):.1f}."), ""]
    (out / "box_loss.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "box_loss.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("lcp")
    p.add_argument("--model", action="append", default=None,
                   help="NAME:<iter>/deploy.json:<collect dir> (default V12h, V16b)")
    p.add_argument("--N", type=int, default=7)
    p.add_argument("--max-episodes", type=int, default=5)
    p.add_argument("--out", required=True)
    s = sub.add_parser("solve-times")
    s.add_argument("--root", default="data/lcs/mpc_eval")
    s.add_argument("--out", required=True)
    b = sub.add_parser("box-loss")
    b.add_argument("--group", action="append", default=None,
                   help="LABEL:on|off|any:<root>[:<root>...] (dirs searched for index.json)")
    b.add_argument("--reference-box", default="data/lcs/realfirst/cmp/sbox_sim/box.json")
    b.add_argument("--ur-min-mm", type=float, default=3.0)
    b.add_argument("--franka-min-mm", type=float, default=2.0)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "lcp":
        a.model = a.model or [
            ("V12h:data/lcs/realfirst/cmp/C1_transport/iter1/deploy.json:"
             "data/lcs/realfirst/cmp/C1_transport/iter1/collect"),
            ("V16b:data/lcs/realfirst/cmp/v12_36/iter1/deploy.json:"
             "data/lcs/realfirst/cmp/v12_36/iter1/collect")]
        return cmd_lcp(a)
    if a.cmd == "solve-times":
        return cmd_solve_times(a)
    c = "data/lcs/realfirst/cmp"
    a.group = a.group or [
        f"V12-17 no box (V12h):off:{c}/overshoot/sim/off",
        f"V12-17 box (V12h):on:{c}/sbox_sim/box",
        f"V12-37 no prior, box (V12h):on:{c}/noprior",
        f"V12-36 loop, box (V16a-d):on:{c}/v12_36"]
    return cmd_box_loss(a)


if __name__ == "__main__":
    sys.exit(main())
