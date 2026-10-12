#!/usr/bin/env python3
"""Checks of the latent and EE data-support costs (``MppiParams.latent_support`` /
``ee_support``).

S1 params: default None; ``from_yaml`` of a config with the block == pre-change module's
   ``to_dict`` + the block, without it == pre + ``latent_support: None``; bad blocks refused;
   the cached solver keys on the block; ``max_points`` = a deterministic stride subsample.
S2 off: pre-change (git HEAD) vs tree ``plan_cost`` on random rollouts and ``mppi_plan`` (cold +
   warm, fixed seed) with ``latent_support`` None: costs, plans and stats identical; on with a
   radius no knot reaches == off exactly.
S3 brute force: ``plan_cost`` on - off == weight sum_k relu(d_k - radius)^2 with d_k a numpy
   nearest-neighbour search (float64, rtol 1e-6), where all / terminal; ``nn_dist_np``
   == the torch distance.
S4 chunking: ``nn_dist`` over chunks of 1 / 7 / 64 rows / the default == unchunked (to 1e-4);
   the solver's ``plan_cost`` == ``rollout_cost`` of its plan with the term biting, and its
   ``support_d0`` / ``support_dmax`` == the brute-force distance of the returned plan. With
   ``--device cuda``: the captured graph (fused rollout) == eager.
S5 replay (numbers, no threshold): V21c set2 gv_00 repeat 1 ("over"), solves at t 0.5-2 s:
   ``mppi_plan`` from the recorded knot-0 state, stage goal (``z_ref``) and warm start (the
   previous solve's recorded 7 inputs, shifted) with the support cost off / radius 1.5 x
   weights 1, 10, 100 / radius 1 x weight 10 (and one solve at 2 s); first-action Franka
   x/y/z (world and large-pulley frame) and planned max d.
   Clearance is off (its torch path is minutes per CPU solve; the run's plans were >= 30 mm
   clear there); the z ceiling is the run's (JSONL ``z_max``).
E1-E4 the same for ``ee_support`` (S2 covers off == pre and a far radius == off for both):
   the training EE positions mapped to P's frame (``R q + t``) map back through
   ``belt_cloud.to_pulley_frame`` to the npz rows; the term == weight sum_k relu((d_k -
   radius_mm) / 10)^2 with d_k a numpy NN in the pulley frame (mm); chunking; solver ==
   ``rollout_cost``; ``ee_support_d0`` / ``ee_support_dmax`` == brute force.
E5 replay as S5 with ``ee_support`` radius 35 mm x weights 1, 10, 100: first-action Franka
   pulley x/y/z and planned max EE d; plus d0 of the recorded P over the first 10 s (full
   training set, pulley frame).

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_latent_support.py \\
        [--device cpu] [--skip-replay] [--seeds 3]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from task_common import mppi_lcs as tree

ES = REPO / "data/lcs/realfirst/cmp/encoder_study"
DEPLOY = ES / "sim/V21c/deploy_v17c/deploy.npz"
CACHE = ES / "sim/V21c/cache_buffer.npz"
CONFIG = ES / "sim55/configs/V21c_n0.yaml"
RUN = ES / "jpg_t1_recordings/V21c_n0_set2_gv_00__episode_gv_00_1__over"
SOLVES = ES / "sim55/V21c_n0_set2_gv_00/controller/episode_gv_00_1.solves.jsonl"
EPISODE = ES / "sim55/V21c_n0_set2_gv_00/episode_gv_00_1.npz"
LS = {"file": str(CACHE), "radius": 1.5, "weight": 1.0}


def pulley_pose7() -> list[float]:
    with np.load(EPISODE, allow_pickle=True) as e:
        return list(json.loads(str(e["sim_meta"]))["belt_seg"]["pulley_pose7_known"])


ES_ = {"file": str(CACHE), "pulley_pose7": pulley_pose7(), "radius_mm": 35.0, "weight": 1.0}


def load_head(tmp: Path) -> types.ModuleType:
    src = subprocess.run(["git", "show", "HEAD:src/task_common/mppi_lcs.py"], cwd=REPO,
                         check=True, capture_output=True, text=True).stdout
    p = tmp / "mppi_lcs_head.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("mppi_lcs_head", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def base_params(mod, **kw):
    doc = yaml.safe_load(CONFIG.read_text())
    doc["clearance"] = None
    return mod.MppiParams(**{**doc, "cuda_graph": False, **kw})


def nn_dist_np(Z: np.ndarray, S: np.ndarray, z_std: np.ndarray) -> np.ndarray:
    """Brute-force whitened nearest-neighbour distance, one query at a time (float64)."""
    Zw, Sw = Z.reshape(-1, Z.shape[-1]) / z_std, S / z_std
    return np.array([np.sqrt(((Sw - z) ** 2).sum(1)).min() for z in Zw]).reshape(Z.shape[:-1])


def rollouts(rng, S: np.ndarray, z_std: np.ndarray, K: int, H: int) -> np.ndarray:
    """``(K, H+1, n_x)`` latents near training rows, whitened noise scale 0.05..3 per sample."""
    base = S[rng.integers(0, len(S), size=(K, H + 1))]
    scale = np.geomspace(0.05, 3.0, K)[:, None, None]
    return base + z_std * scale * rng.normal(size=base.shape)


def s1(head) -> tuple[bool, str]:
    notes, ok = [], tree.MppiParams().latent_support is None
    a = head.MppiParams.from_yaml(CONFIG).to_dict()
    b = tree.MppiParams.from_yaml(CONFIG).to_dict()
    ok &= b.pop("latent_support", "missing") is None and b.pop("ee_support", "x") is None \
        and a == b
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        yaml.safe_dump({**yaml.safe_load(CONFIG.read_text()), "latent_support": LS}, f)
        f.flush()
        c = tree.MppiParams.from_yaml(f.name).to_dict()
    ok &= c.pop("latent_support") == LS and c.pop("ee_support") is None and c == a
    bad = [{}, {"radius": 1.0}, {**LS, "radius": -1.0}, {**LS, "weight": math.nan},
           {**LS, "max_points": 0}, {**LS, "max_points": 2.5}, {**LS, "max_points": True},
           {**LS, "where": "all_knots"}, {**LS, "other": 1}, {**LS, "file": 3}]
    refused = 0
    for ls in bad:
        try:
            tree.MppiParams(latent_support=ls)
        except (ValueError, TypeError):
            refused += 1
    ok &= refused == len(bad)
    for ls in ({"file": "x.npz"}, {**LS, "where": "terminal", "max_points": 10}):
        tree.MppiParams(latent_support=ls)
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu")
    p = tree.MppiParams(K=8, H=6, n_iter=1, cuda_graph=False)
    s0 = tree.solver_for(lcs, p)
    s1_ = tree.solver_for(lcs, dataclasses.replace(p, latent_support=LS))
    s2 = tree.solver_for(lcs, dataclasses.replace(p, latent_support={**LS, "weight": 2.0}))
    keyed = s0 is not s1_ and s1_ is not s2 and s0.s_lsup is None and s1_.s_lsup is not None
    with np.load(CACHE) as f:
        z = f["z_t"]
    sub = tree.load_latent_support(LS, lcs.n_x)
    sub7 = tree.load_latent_support({**LS, "max_points": 7})
    stride = math.ceil(len(z) / 3000)
    sub_ok = (np.array_equal(sub, z[::stride]) and len(sub) <= 3000
              and np.array_equal(sub7, z[::math.ceil(len(z) / 7)]) and len(sub7) <= 7
              and np.array_equal(tree.load_latent_support({**LS, "max_points": 10 ** 6}), z))
    ok &= keyed and sub_ok
    notes.append(f"V21c_n0.yaml to_dict == pre + None, with the block == pre + it; "
                 f"{refused}/{len(bad)} bad blocks refused; solver keyed {keyed}; subsample "
                 f"N {len(z)} -> {len(sub)} (stride {stride}) {sub_ok}")
    return bool(ok), "; ".join(notes)


def s2(head) -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    lt, lh = (m.TorchLcs.from_deploy(DEPLOY, "cpu") for m in (tree, head))
    S = tree.load_latent_support(LS, lt.n_x)
    pt, ph = base_params(tree), base_params(head)
    K, H = 64, pt.H
    Z = rollouts(rng, S, lt.z_std_np, K, H)
    P = np.tile(np.r_[0.37, 0.0, 0.08, 0.51, -0.02, 0.09], (K, H + 1, 1))
    U = rng.normal(size=(K, H, 12)) * 1e-3
    R = np.zeros((K, H + 1, 2))
    goal = Z[0, -1]
    half = np.full(12, 0.003)
    anchor = rng.normal(size=12) * 1e-3

    def cost(m, lcs, p):
        t = lcs.tensor
        c, n, e = m.plan_cost(lcs, p, t(Z), t(P), t(R), t(U), t(goal), t(half), t(anchor))
        return torch.cat([c, n, e.to(c.dtype)])

    c_head, c_off = cost(head, lh, ph), cost(tree, lt, pt)
    far_kw = {"latent_support": {**LS, "radius": 1e6}, "ee_support": {**ES_, "radius_mm": 1e9}}
    c_far = cost(tree, lt, dataclasses.replace(pt, **far_kw))
    ok_cost = torch.equal(c_head, c_off) and torch.equal(c_off, c_far)
    fails, n_cmp = [], 0
    z0, p0 = Z[3, 0], P[0, 0]
    for prm_t, prm_h, tag in ((pt, ph, "base"), (dataclasses.replace(pt, K=256, n_iter=2),
                                                 dataclasses.replace(ph, K=256, n_iter=2), "K256")):
        u_prev = None
        far = dataclasses.replace(prm_t, **far_kw)
        for step in range(3):  # cold, then warm on the tree's own plan
            outs = []
            for m, lcs, prm in ((head, lh, prm_h), (tree, lt, prm_t), (tree, lt, far)):
                gen = torch.Generator(device="cpu").manual_seed(7 + step)
                outs.append(m.mppi_plan(lcs, prm, z0, p0, None, goal, u_prev, gen))
            (ua, ia), (ub, ib), (uc, ic) = outs
            n_cmp += 1
            if not (np.array_equal(ua, ub) and np.array_equal(ub, uc)):
                fails.append(f"{tag} step {step} plan differs")
            for k in ia:
                if k == "solve_ms":
                    continue
                if not np.array_equal(np.asarray(ia[k]), np.asarray(ib[k])):
                    fails.append(f"{tag} step {step} info {k} differs")
            u_prev = ub
    return ok_cost and not fails, (f"plan_cost pre == off == far-radius on {ok_cost} (K {K}, H "
                                   f"{H}); mppi_plan pre == off == far-radius on over {n_cmp} "
                                   f"solves" + (f"; FAIL {fails[:4]}" if fails else ""))


def s3() -> tuple[bool, str]:
    rng = np.random.default_rng(2)
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu", torch.float64)
    S = tree.load_latent_support(LS, lcs.n_x)
    p_off = tree.MppiParams(K=16, H=8)
    K, H = 16, 8
    Z = rollouts(rng, S, lcs.z_std_np, K, H)
    t = lcs.tensor
    args = (t(np.zeros((K, H + 1, 6))), t(np.zeros((K, H + 1, 2))),
            t(rng.normal(size=(K, H, 12)) * 1e-3), t(Z[0, 0]), t(np.full(12, 0.003)), None)
    c_off = tree.plan_cost(lcs, p_off, t(Z), *args)[0].numpy()
    d = nn_dist_np(Z, S, lcs.z_std_np)
    worst, rows = 0.0, []
    for radius, weight, where in ((1.5, 1.0, "all"), (0.5, 3.7, "all"), (1.0, 10.0, "terminal"),
                                  (0.0, 1.0, "all")):
        ls = {**LS, "radius": radius, "weight": weight, "where": where}
        got = tree.plan_cost(lcs, dataclasses.replace(p_off, latent_support=ls), t(Z),
                             *args)[0].numpy() - c_off
        dk = d[:, -1:] if where == "terminal" else d[:, 1:]
        want = weight * (np.maximum(dk - radius, 0.0) ** 2).sum(1)
        rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-12)
        rel[want == 0.0] = np.abs(got[want == 0.0])
        worst = max(worst, float(rel.max()))
        rows.append(f"r {radius} w {weight} {where}: {int((want > 0).sum())}/{K} samples "
                    f"penalised")
    S_w = S / lcs.z_std_np
    d_np = tree.nn_dist_np(Z[5] / lcs.z_std_np, S_w)
    d_t = tree.nn_dist(t(Z[5]) / lcs.z_std, t(S_w)).numpy()
    rel_np = float(np.max(np.abs(d_np - d[5]) / d[5]))
    rel_t = float(np.max(np.abs(d_t - d[5]) / d[5]))
    ok = worst <= 1e-6 and rel_np <= 1e-9 and rel_t <= 1e-9
    return ok, (f"term vs numpy NN max rel err {worst:.1e} ({'; '.join(rows)}; d range "
                f"{d.min():.2f}..{d.max():.2f}); nn_dist_np rel {rel_np:.1e}, torch "
                f"rel {rel_t:.1e}")


def s4(device: str) -> tuple[bool, str]:
    rng = np.random.default_rng(3)
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu")
    S_w = lcs.tensor(tree.load_latent_support(LS, lcs.n_x) / lcs.z_std_np)
    Q = lcs.tensor(rng.normal(size=(37, 5, lcs.n_x)) * 2.0)
    full = tree.nn_dist(Q, S_w, chunk=10 ** 9)
    diffs = {c: float((tree.nn_dist(Q, S_w, chunk=c) - full).abs().max())
             for c in (1, 7, 64, None)}
    ok_chunk = all(v <= 1e-4 for v in diffs.values())  # float32 matmul blocking differs per chunk
    n_q, M = 1024 * 24, S_w.shape[0]
    rows = max(1, tree.LATENT_SUPPORT_CHUNK_ELEMS // M)
    mb = rows * M * 4 / 1e6
    # Solver: its plan cost == rollout_cost of its plan with the term biting.
    S = tree.load_latent_support(LS, lcs.n_x)
    z0 = S[100] + lcs.z_std_np * 1.2 * rng.normal(size=lcs.n_x)
    goal = S[-1] + lcs.z_std_np * 2.0
    p0 = np.r_[0.37, 0.0, 0.08, 0.51, -0.02, 0.09]
    prm = base_params(tree, K=256, n_iter=2, latent_support={**LS, "radius": 0.5, "weight": 10.0})
    gen = torch.Generator(device="cpu").manual_seed(3)
    U, info = tree.mppi_plan(lcs, prm, z0, p0, None, goal, None, gen)
    c_roll, _, Zr = tree.rollout_cost(lcs, prm, z0, p0, None, goal, U)
    c_off = tree.rollout_cost(lcs, dataclasses.replace(prm, latent_support=None), z0, p0, None,
                              goal, U)[0]
    rel_cost = abs(c_roll - info["plan_cost"]) / abs(c_roll)
    d_bf = nn_dist_np(info["Z"], S, lcs.z_std_np)
    ok_info = (abs(info["support_d0"] - d_bf[0]) < 1e-9
               and abs(info["support_dmax"] - d_bf[1:].max()) < 1e-9)
    ok = ok_chunk and rel_cost < 1e-5 and c_roll > c_off and ok_info and not info["fallback"]
    msg = (f"chunk max |diff| {diffs} vs unchunked; default chunk {rows} rows x {M} = "
           f"{mb:.0f} MB f32 per cdist for {n_q} queries; solver plan_cost {info['plan_cost']:.4f}"
           f" vs rollout_cost {c_roll:.4f} (term {c_roll - c_off:.3f}); info d0 "
           f"{info['support_d0']:.3f} dmax {info['support_dmax']:.3f} == brute force {ok_info}")
    if device == "cuda":
        lc = tree.TorchLcs.from_deploy(DEPLOY, "cuda")
        res = []
        for g in (False, True):
            s = tree.MppiSolver(lc, dataclasses.replace(prm, cuda_graph=g, fused_rollout=True),
                                use_graph=g)
            gen = torch.Generator(device="cuda").manual_seed(3)
            res.append(s.solve(z0, p0, None, goal, None, gen, tree.safety_limits(prm)))
        d = float(np.abs(res[0][0] - res[1][0]).max())
        ok &= d == 0.0 and res[0][1]["support_dmax"] == res[1][1]["support_dmax"]
        msg += f"; cuda graph vs eager plan max |diff| {d:.1e}"
    return bool(ok), msg


def pulley_rot() -> np.ndarray:
    from task_common.belt_cloud import known_large_pulley_pose7

    w, x, y, z = known_large_pulley_pose7()[3:]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def s5(seeds: int, out_json: Path | None) -> tuple[bool, str]:
    import check_replay_learned as crl

    from task_common import replay_learned_mpc as rlm
    from task_common.recording import Recording

    rec = Recording.load(RUN.resolve())
    pt, _ = crl.panel(rlm, rec, types.SimpleNamespace(deploy=None, decoder=None))
    rows = {r["utime"]: r for r in map(json.loads, SOLVES.read_text().splitlines())}
    sv = pt.solves
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu")
    nx = lcs.n_x
    S = tree.load_latent_support(LS, nx)
    Rp = pulley_rot()
    t_rel = np.array([s.t[0] for s in sv]) - sv[0].t[0]
    picks = sorted({int(np.argmin(np.abs(t_rel - tt))) for tt in (0.5, 0.83, 1.17, 1.5, 2.0)})
    settings = [("off", None)] + [(f"r{r:g} w{w:g}", {**LS, "radius": r, "weight": w})
                                  for r, w in ((1.5, 1.0), (1.5, 10.0), (1.5, 100.0), (1.0, 10.0))]
    table = []
    for i in picks:
        s, row = sv[i], rows[sv[i].utime]
        z0, p0 = s.x_sol[:nx, 0], s.x_sol[nx:, 0]
        if not np.allclose(p0, row["p0"], atol=1e-12):
            raise ValueError(f"solve {i}: x_sol P0 != JSONL p0")
        goal, u_prev = s.z_ref[:, 0], sv[i - 1].u_sol.T  # previous solve's 7 recorded inputs
        rot0 = np.asarray(row["rot0"])
        rec_u0 = s.u_sol[:, 0]
        d0 = float(nn_dist_np(z0[None], S, lcs.z_std_np)[0])
        d_rec = nn_dist_np(s.x_sol[:nx].T, S, lcs.z_std_np)  # recorded 7-knot plan
        entry = {"i": i, "t": round(float(t_rel[i]), 3), "stage": row["stage"], "d0": d0,
                 "recorded": {"u0_mm": (rec_u0[:3] * 1e3).tolist(),
                              "u0_pulley_mm": (Rp.T @ rec_u0[:3] * 1e3).tolist(),
                              "dmax_k1_6": float(d_rec[1:].max()),
                              "clear_min_f_mm": row.get("clear_min_f_mm"),
                              "clear_min_ur_mm": row.get("clear_min_ur_mm")},
                 "settings": {}}
        for name, ls in settings:
            prm = base_params(tree, z_max_franka=row["z_max"][0], z_max_ur=row["z_max"][1],
                              latent_support=ls)
            us, dmax, dk6, t0 = [], [], [], time.perf_counter()
            for seed in range(seeds):
                gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
                U, info = tree.mppi_plan(lcs, prm, z0, p0, rot0, goal, u_prev, gen)
                d = nn_dist_np(info["Z"], S, lcs.z_std_np)
                us.append(U[0, :3] * 1e3)
                dmax.append(float(d[1:].max()))
                dk6.append(float(d[1:7].max()))
            us = np.array(us)
            entry["settings"][name] = {
                "u0_mm": us.mean(0).tolist(), "u0_mm_sd": us.std(0).tolist(),
                "u0_pulley_mm": (us @ Rp).mean(0).tolist(),
                "dmax": float(np.mean(dmax)), "dmax_k1_6": float(np.mean(dk6)),
                "goal_dist_H": float(info["goal_dist_H"]),
                "s_per_solve": (time.perf_counter() - t0) / seeds}
        table.append(entry)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"run": str(RUN.relative_to(REPO)), "seeds": seeds,
                                        "rows": table}, indent=1))
    lines = ["", "| t (s) | i | stage | d0 | setting | u0 Franka world x/y/z (mm) | u0 pulley "
             "x/y/z (mm) | planned max d (k 1..H) | max d k 1..6 |",
             "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "/".join(f"{x:+.2f}" for x in v)  # noqa: E731
    for e in table:
        r = e["recorded"]
        lines.append(f"| {e['t']:.2f} | {e['i']} | {e['stage']} | {e['d0']:.2f} | recorded | "
                     f"{fmt(r['u0_mm'])} | {fmt(r['u0_pulley_mm'])} | - | {r['dmax_k1_6']:.2f} |")
        for name, v in e["settings"].items():
            lines.append(f"| | | | | {name} | {fmt(v['u0_mm'])} | {fmt(v['u0_pulley_mm'])} | "
                         f"{v['dmax']:.2f} | {v['dmax_k1_6']:.2f} |")
    return True, f"{len(picks)} solves x {len(settings)} settings x {seeds} seeds" + \
        "\n".join(lines)


def ee_pulley_mm(P: np.ndarray, pose7) -> np.ndarray:
    """``(..., 6)`` P (m, world) -> pulley frame (mm) per arm, ``belt_cloud.to_pulley_frame``."""
    from task_common.belt_cloud import to_pulley_frame

    P = np.asarray(P, dtype=np.float64)
    q = [to_pulley_frame(P[..., sl].reshape(-1, 3), pose7) for sl in (slice(0, 3), slice(3, 6))]
    return 1e3 * np.concatenate(q, axis=1).reshape(P.shape)


def ee_train_mm(es: dict) -> np.ndarray:
    """The npz's ``ee_pos`` (pulley frame, mm), subsampled like ``load_ee_support``."""
    with np.load(es["file"]) as f:
        E = np.asarray(f["ee_pos"], dtype=np.float64)
    return 1e3 * E[::math.ceil(len(E) / int(es.get("max_points", 3000)))]


def ee_nn_mm(P: np.ndarray, es: dict) -> np.ndarray:
    """Brute-force 6-D NN distance (mm) of world ``P (..., 6)`` in the pulley frame."""
    Q, S = ee_pulley_mm(P, es["pulley_pose7"]), ee_train_mm(es)
    Qf = Q.reshape(-1, 6)
    return np.array([np.sqrt(((S - q) ** 2).sum(1)).min() for q in Qf]).reshape(P.shape[:-1])


def ee_rollouts(rng, S_w: np.ndarray, K: int, H: int) -> np.ndarray:
    """``(K, H+1, 6)`` world P (m) near training rows, noise 1..100 mm per sample."""
    base = S_w[rng.integers(0, len(S_w), size=(K, H + 1))]
    return base + 1e-3 * np.geomspace(1.0, 100.0, K)[:, None, None] * rng.normal(size=base.shape)


def e1() -> tuple[bool, str]:
    ok = tree.MppiParams().ee_support is None
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        yaml.safe_dump({**yaml.safe_load(CONFIG.read_text()), "ee_support": ES_}, f)
        f.flush()
        ok &= tree.MppiParams.from_yaml(f.name).ee_support == ES_
    bad = [{"file": str(CACHE)}, {"pulley_pose7": ES_["pulley_pose7"]},
           {**ES_, "pulley_pose7": [0.0] * 6}, {**ES_, "pulley_pose7": [0.0] * 7},
           {**ES_, "pulley_pose7": [math.nan] + [0.0] * 6}, {**ES_, "radius_mm": -1.0},
           {**ES_, "weight": math.inf}, {**ES_, "max_points": 0}, {**ES_, "where": "x"},
           {**ES_, "radius": 1.0}, {**ES_, "file": 3}]
    refused = 0
    for es in bad:
        try:
            tree.MppiParams(ee_support=es)
        except (ValueError, TypeError):
            refused += 1
    ok &= refused == len(bad)
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu")
    p = tree.MppiParams(K=8, H=6, n_iter=1, cuda_graph=False)
    s0 = tree.solver_for(lcs, p)
    s1_ = tree.solver_for(lcs, dataclasses.replace(p, ee_support=ES_))
    s2 = tree.solver_for(lcs, dataclasses.replace(p, ee_support={**ES_, "radius_mm": 20.0}))
    keyed = s0 is not s1_ and s1_ is not s2 and s0.s_esup is None and s1_.s_esup is not None
    W = tree.load_ee_support(ES_)
    back = ee_pulley_mm(W, ES_["pulley_pose7"])
    frame_err = float(np.abs(back - ee_train_mm(ES_)).max())
    W7 = tree.load_ee_support({**ES_, "max_points": 7})
    ok &= keyed and frame_err < 1e-9 and len(W) <= 3000 and len(W7) <= 7
    return bool(ok), (f"from_yaml carries the block; {refused}/{len(bad)} bad blocks refused; "
                      f"solver keyed {keyed}; {len(W)} points (stride); P frame -> "
                      f"to_pulley_frame == npz rows (max {frame_err:.1e} mm)")


def e3() -> tuple[bool, str]:
    rng = np.random.default_rng(4)
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu", torch.float64)
    K, H = 16, 8
    P = ee_rollouts(rng, tree.load_ee_support(ES_), K, H)
    t = lcs.tensor
    Z = t(rng.normal(size=(K, H + 1, lcs.n_x)))
    rest = (t(P), t(np.zeros((K, H + 1, 2))), t(rng.normal(size=(K, H, 12)) * 1e-3),
            t(rng.normal(size=lcs.n_x)), t(np.full(12, 0.003)), None)
    p_off = tree.MppiParams(K=K, H=H)
    c_off = tree.plan_cost(lcs, p_off, Z, *rest)[0].numpy()
    d = ee_nn_mm(P, ES_)
    worst, rows = 0.0, []
    for r, w, where in ((35.0, 1.0, "all"), (10.0, 3.7, "all"), (20.0, 10.0, "terminal"),
                        (0.0, 1.0, "all")):
        es = {**ES_, "radius_mm": r, "weight": w, "where": where}
        got = tree.plan_cost(lcs, dataclasses.replace(p_off, ee_support=es), Z,
                             *rest)[0].numpy() - c_off
        dk = d[:, -1:] if where == "terminal" else d[:, 1:]
        want = w * (np.maximum((dk - r) / 10.0, 0.0) ** 2).sum(1)
        rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-12)
        rel[want == 0.0] = np.abs(got[want == 0.0])
        worst = max(worst, float(rel.max()))
        rows.append(f"r {r:g} w {w:g} {where}: {int((want > 0).sum())}/{K} penalised")
    d_np = tree.nn_dist_np(P[5] * 1e3, tree.load_ee_support(ES_) * 1e3)
    rel_np = float(np.max(np.abs(d_np - d[5]) / d[5]))
    ok = worst <= 1e-6 and rel_np <= 1e-9
    return ok, (f"term vs pulley-frame numpy NN max rel err {worst:.1e} ({'; '.join(rows)}; d "
                f"{d.min():.1f}..{d.max():.1f} mm); nn_dist_np (P frame) rel {rel_np:.1e}")


def e4() -> tuple[bool, str]:
    rng = np.random.default_rng(5)
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu")
    S = tree.load_ee_support(ES_)
    S_mm = lcs.tensor(S * 1e3)
    Q = lcs.tensor(ee_rollouts(rng, S, 37, 4) * 1e3)
    full = tree.nn_dist(Q, S_mm, chunk=10 ** 9)
    diffs = {c: float((tree.nn_dist(Q, S_mm, chunk=c) - full).abs().max())
             for c in (1, 7, 64, None)}
    ok_chunk = all(v <= 1e-3 for v in diffs.values())  # mm; float32 matmul blocking differs per chunk
    Sl = tree.load_latent_support(LS, lcs.n_x)
    z0, goal = Sl[100], Sl[-1] + lcs.z_std_np * 2.0
    p0 = S[50] + np.r_[0.0, 0.04, 0.0, 0.0, 0.0, 0.03]  # 50 mm off one training row
    prm = base_params(tree, K=256, n_iter=2, ee_support={**ES_, "radius_mm": 20.0,
                                                         "weight": 10.0})
    gen = torch.Generator(device="cpu").manual_seed(5)
    U, info = tree.mppi_plan(lcs, prm, z0, p0, None, goal, None, gen)
    c_roll = tree.rollout_cost(lcs, prm, z0, p0, None, goal, U)[0]
    c_off = tree.rollout_cost(lcs, dataclasses.replace(prm, ee_support=None), z0, p0, None,
                              goal, U)[0]
    rel_cost = abs(c_roll - info["plan_cost"]) / abs(c_roll)
    d_bf = ee_nn_mm(info["P"], ES_)
    # f32 rollout P: compare at 1e-6 mm-ish of float64 recompute on the returned P
    ok_info = (abs(info["ee_support_d0"] - d_bf[0]) < 1e-6
               and abs(info["ee_support_dmax"] - d_bf[1:].max()) < 1e-6)
    ok = ok_chunk and rel_cost < 1e-5 and c_roll > c_off and ok_info and not info["fallback"]
    return bool(ok), (f"chunk max |diff| {diffs} (mm); solver plan_cost {info['plan_cost']:.4f}"
                      f" vs rollout_cost {c_roll:.4f} (term {c_roll - c_off:.3f}); info d0 "
                      f"{info['ee_support_d0']:.2f} dmax {info['ee_support_dmax']:.2f} mm == "
                      f"brute force {ok_info}")


def e5(seeds: int, out_json: Path | None) -> tuple[bool, str]:
    import check_replay_learned as crl
    from scipy.spatial import cKDTree

    from task_common import replay_learned_mpc as rlm
    from task_common.recording import Recording

    rec = Recording.load(RUN.resolve())
    pt, _ = crl.panel(rlm, rec, types.SimpleNamespace(deploy=None, decoder=None))
    rows = {r["utime"]: r for r in map(json.loads, SOLVES.read_text().splitlines())}
    sv = pt.solves
    lcs = tree.TorchLcs.from_deploy(DEPLOY, "cpu")
    nx = lcs.n_x
    Rp = pulley_rot()
    t_rel = np.array([s.t[0] for s in sv]) - sv[0].t[0]
    P0 = np.stack([s.x_sol[nx:, 0] for s in sv])
    with np.load(CACHE) as f:
        full = np.asarray(f["ee_pos"], np.float64) * 1e3
    d_full = cKDTree(full).query(ee_pulley_mm(P0, ES_["pulley_pose7"]))[0]
    d_sub = tree.nn_dist_np(P0 * 1e3, tree.load_ee_support(ES_) * 1e3)
    m10 = t_rel < 10.0
    at = {tt: int(np.argmin(np.abs(t_rel - tt))) for tt in (0.0, 1.0, 2.0, 3.0, 5.0)}
    t_max = t_rel[m10][np.argmax(d_full[m10])]
    d0_line = (", ".join(f"t {tt:g}: {d_full[i]:.1f}" for tt, i in at.items())
               + f"; max <10 s {d_full[m10].max():.1f} mm at t {t_max:.2f}"
               + f" (3000-subsample max {d_sub[m10].max():.1f})")
    picks = sorted({int(np.argmin(np.abs(t_rel - tt))) for tt in (0.5, 0.83, 1.17, 1.5, 2.0)})
    settings = [("off", None)] + [(f"r35 w{w:g}", {**ES_, "weight": w}) for w in (1.0, 10.0, 100.0)]
    table = []
    for i in picks:
        s, row = sv[i], rows[sv[i].utime]
        z0, p0 = s.x_sol[:nx, 0], s.x_sol[nx:, 0]
        goal, u_prev = s.z_ref[:, 0], sv[i - 1].u_sol.T
        rot0 = np.asarray(row["rot0"])
        d_rec = ee_nn_mm(s.x_sol[nx:].T, ES_)
        entry = {"i": i, "t": round(float(t_rel[i]), 3), "d0_mm": float(d_full[i]),
                 "d0_sub_mm": float(d_sub[i]),
                 "recorded": {"u0_pulley_mm": (Rp.T @ s.u_sol[:3, 0] * 1e3).tolist(),
                              "dmax_k1_6": float(d_rec[1:].max())}, "settings": {}}
        for name, es in settings:
            prm = base_params(tree, z_max_franka=row["z_max"][0], z_max_ur=row["z_max"][1],
                              ee_support=es)
            us, dmax, dk6 = [], [], []
            for seed in range(seeds):
                gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
                U, info = tree.mppi_plan(lcs, prm, z0, p0, rot0, goal, u_prev, gen)
                d = ee_nn_mm(info["P"], ES_)
                us.append(U[0, :3] * 1e3)
                dmax.append(float(d[1:].max()))
                dk6.append(float(d[1:7].max()))
            us = np.array(us)
            entry["settings"][name] = {"u0_pulley_mm": (us @ Rp).mean(0).tolist(),
                                       "u0_pulley_mm_sd": (us @ Rp).std(0).tolist(),
                                       "dmax_mm": float(np.mean(dmax)),
                                       "dmax_k1_6_mm": float(np.mean(dk6))}
        table.append(entry)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"run": str(RUN.relative_to(REPO)), "seeds": seeds,
                                        "d0_recorded": d0_line, "rows": table}, indent=1))
    fmt = lambda v: "/".join(f"{x:+.2f}" for x in v)  # noqa: E731
    lines = ["", f"recorded P d0 (full set, mm): {d0_line}", "",
             "| t (s) | d0 (mm) | setting | u0 Franka pulley x/y/z (mm) | planned max EE d "
             "k 1..H (mm) | k 1..6 (mm) |", "|---|---|---|---|---|---|"]
    for e in table:
        r = e["recorded"]
        lines.append(f"| {e['t']:.2f} | {e['d0_sub_mm']:.1f} | recorded | "
                     f"{fmt(r['u0_pulley_mm'])} | - | {r['dmax_k1_6']:.1f} |")
        for name, v in e["settings"].items():
            lines.append(f"| | | {name} | {fmt(v['u0_pulley_mm'])} | {v['dmax_mm']:.1f} | "
                         f"{v['dmax_k1_6_mm']:.1f} |")
    return True, f"{len(picks)} solves x {len(settings)} settings x {seeds} seeds" + \
        "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--replay-json", type=Path, default=None)
    ap.add_argument("--ee-replay-json", type=Path, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    a = ap.parse_args()
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        head = load_head(Path(tmp))
        checks = [("S1", lambda: s1(head)), ("S2", lambda: s2(head)), ("S3", s3),
                  ("S4", lambda: s4(a.device)), ("E1", e1), ("E3", e3), ("E4", e4)]
        if not a.skip_replay:
            checks += [("S5", lambda: s5(a.seeds, a.replay_json)),
                       ("E5", lambda: e5(a.seeds, a.ee_replay_json))]
        for name, fn in checks:
            if a.only and name not in a.only:
                continue
            t0 = time.perf_counter()
            try:
                ok, msg = fn()
            except Exception as exc:  # noqa: BLE001 - report every check
                import traceback
                traceback.print_exc()
                ok, msg = False, f"exception {exc!r}"
            results[name] = ok
            print(f"[{'PASS' if ok else 'FAIL'}] {name} ({time.perf_counter() - t0:.1f} s): {msg}",
                  flush=True)
    n = sum(results.values())
    print(f"{n}/{len(results)} pass")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
