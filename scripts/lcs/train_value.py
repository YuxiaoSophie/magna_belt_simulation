#!/usr/bin/env python3
"""Learned cost-to-go ``V(z)`` in a deploy's latent for MPPI (``MppiParams.value``; V12-39).

Real-available data only: every frame's latent = the deploy's numpy encoder on the recorded
segmented belt cloud (``pcd_belt_seg``); the one-bit human label per episode (``label_human`` of
the seed buffers; rollouts: ``index.json`` outcome == engaged; a MILES session: its last walk
segment's outcome == engaged). Sim GT (per-frame wrap / h, outcome classes) is kept in the cache
for scoring only (``eval``).

Value (seconds, discount ``gamma`` per frame of ``dt``): cost ``dt`` per frame until success,
``V_max = dt / (1 - gamma)`` = never. Chains:
- success episode: the seat time ``t*`` = hindsight, the first frame from which the (5-frame
  median) whitened distance to the episode's final latent stays <= ``eps``; frames >= t* are
  absorbing (V 0);
- failed episode: terminal V_max after the last frame;
- MILES session: its walk segments concatenated (demo progress), labelled as an episode; a valid
  deviate-and-return cycle bootstraps on its own first (arrival) latent, ``V(z_0)``, after the
  return; a disturbed cycle on its last latent.
Targets = TD(lambda) returns with the current net (``--rounds`` fitted rounds; lambda 1 = Monte
Carlo where no bootstrap). Episodes are held out whole (``--holdout``).

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/lcs/train_value.py encode \\
        --deploy <deploy.npz> --out data/lcs/realfirst/cmp/value_cost
    ... train_value.py train --out data/lcs/realfirst/cmp/value_cost --tag v1
    ... train_value.py eval --out data/lcs/realfirst/cmp/value_cost --tag v1
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from task_common import latent_value as lv
from task_common.bc_prior import encoder_fingerprint
from task_common.latent_encoder import LatentEncoder

RF = REPO / "data/lcs/realfirst"
DEMO_BUFFERS = ["v1_1/seed/buffer.json", "v1_1/seed_ext/buffer.json",
                "cmp/demos_ext2/buffer.json"]
MILES_BUFFERS = ["miles/v1/buffer.json", "miles/v12_29b/buffer.json"]
ROLLOUT_INDEX_GLOBS = ["cmp/*/iter*/collect/index.json", "cmp/a1_sim/*/index.json",
                       "cmp/ctrl_ab/*/index.json", "cmp/noprior/*/index.json",
                       "cmp/sbox_sim/*/*/index.json", "cmp/ur_slant/*/index.json"]
SEG = "pcd_belt_seg"
ENGAGED = "engaged"
GT_ENGAGED_WRAP = 60.0  # outcome.OutcomeThresholds.engaged_arc_deg (scoring only)


# --- encode --------------------------------------------------------------------------------------

def _rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def collect_sources() -> list[dict]:
    eps = []
    for b in DEMO_BUFFERS:
        for r in json.loads((RF / b).read_text())["rollouts"]:
            if r["kind"] not in ("demo", "excitation"):
                continue
            eps.append({"src": r["kind"], "path": _rel(r["episode"]), "run": b,
                        "start": f"{r['start']['set']}/{r['start']['id']}",
                        "label": bool(r["label_human"]), "outcome": r.get("outcome_gt")})
    for b in MILES_BUFFERS:
        for k, r in enumerate(json.loads((RF / b).read_text())["rollouts"]):
            m = r["miles"]
            eps.append({"src": f"miles_{r['segment']}", "path": _rel(r["episode"]), "run": b,
                        "start": f"{r['start']['set']}/{r['start']['id']}",
                        "session": f"{b}:{m['demo']}", "order": k, "valid": m.get("valid"),
                        "waypoint": m.get("waypoint"), "from_frame": m.get("from_frame"),
                        "outcome": r.get("outcome_gt"), "label": None})
    for pat in ROLLOUT_INDEX_GLOBS:
        for f in sorted(glob.glob(str(RF / pat))):
            idx = json.loads(Path(f).read_text())
            if "summary" not in idx:
                continue
            run = Path(f).parent
            for e in idx.get("episodes") or []:
                if e.get("outcome") is None or e.get("status") == "skipped" \
                        or not (run / e["file"]).is_file():
                    continue
                eps.append({"src": "mpc", "path": _rel(run / e["file"]), "run": _rel(run),
                            "start": str(e.get("start_state", {}).get("file", "?")),
                            "label": e["outcome"] == ENGAGED, "outcome": e["outcome"]})
    return eps


def encode(a) -> int:
    enc = LatentEncoder.load(a.deploy)
    eps = collect_sources()
    Z, W, H, P, A, keep = [], [], [], [], [], []
    t0 = time.perf_counter()
    off = 0
    for i, e in enumerate(eps):
        with np.load(REPO / e["path"], allow_pickle=True) as d:
            if SEG not in d.files or len(d[SEG]) == 0:
                print(f"skip {e['path']}: no {SEG}")
                continue
            z = enc.encode_batch(list(d[SEG]))
            empty = np.asarray(enc.last_empty, bool)
            n = len(z)
            w = np.asarray(d["sim_wrap_deg"], np.float64) if "sim_wrap_deg" in d.files \
                else np.full(n, np.nan)
            h = np.asarray(d["sim_h_median_mm"], np.float64) if "sim_h_median_mm" in d.files \
                else np.full(n, np.nan)
            p = np.c_[np.asarray(d["sim_ee_franka"])[:, :3], np.asarray(d["sim_ee_ur"])[:, :3]]
            u = np.asarray(d["actions"], np.float64)
        if empty.any():
            e["n_empty"] = int(empty.sum())
        e.update(off=off, n=n)
        off += n
        Z.append(z)
        W.append(w)
        H.append(h)
        P.append(p)
        A.append(u)
        keep.append(e)
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(eps)}] {off} frames {time.perf_counter() - t0:.0f} s", flush=True)
    out = Path(a.out) / "cache"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(a.deploy, allow_pickle=False) as d:
        z_std, dt = np.asarray(d["z_std"], np.float64), float(d["dt"])
    np.savez(out / "frames.npz", Z=np.concatenate(Z), wrap=np.concatenate(W),
             h=np.concatenate(H), p=np.concatenate(P), u=np.concatenate(A), z_std=z_std, dt=dt)
    meta = {"deploy": str(Path(a.deploy).resolve()), "encoder_sha256": encoder_fingerprint(a.deploy),
            "n_frames": off, "episodes": keep}
    (out / "episodes.json").write_text(json.dumps(meta) + "\n")
    print(f"{len(keep)} episodes, {off} frames, {time.perf_counter() - t0:.0f} s -> {out}")
    return 0


def load_cache(out) -> tuple[dict, dict]:
    c = Path(out) / "cache"
    with np.load(c / "frames.npz") as f:
        fr = {k: f[k] for k in f.files}
    return fr, json.loads((c / "episodes.json").read_text())


# --- chains and targets --------------------------------------------------------------------------

def hindsight_seat(z, z_std, eps: float, win: int = 5) -> int:
    """First frame from which the ``win``-median whitened distance to the final latent (median of
    the last ``win``) stays <= ``eps``."""
    from scipy.ndimage import median_filter

    zf = np.median(z[-win:], axis=0)
    d = median_filter(np.linalg.norm((z - zf) / z_std, axis=1), win, mode="nearest")
    bad = np.flatnonzero(d > eps)
    return int(bad[-1] + 1) if len(bad) else 0


def build_chains(meta: dict, fr: dict, eps: float) -> list[dict]:
    """Chains over cache frame indices: ``kind`` success (``t_star``) / fail / boot_first /
    boot_last; ``eps`` = episode paths (hold-out unit)."""
    chains, sessions = [], {}
    z_std = fr["z_std"]
    for e in meta["episodes"]:
        idx = np.arange(e["off"], e["off"] + e["n"])
        if e["src"] == "miles_walk":
            sessions.setdefault(e["session"], []).append(e)
        elif e["src"] == "miles_cycle":
            chains.append({"idx": idx, "kind": "boot_first" if e["valid"] else "boot_last",
                           "src": e["src"], "eps": [e["path"]]})
        else:
            chains.append({"idx": idx, "kind": "success" if e["label"] else "fail",
                           "src": e["src"], "eps": [e["path"]]})
    for name, segs in sessions.items():
        segs = sorted(segs, key=lambda s: s["order"])
        idx = np.concatenate([np.arange(s["off"], s["off"] + s["n"]) for s in segs])
        ok = segs[-1]["outcome"] == ENGAGED  # the session's one-bit end label
        chains.append({"idx": idx, "kind": "success" if ok else "fail", "src": "miles_walk",
                       "eps": [s["path"] for s in segs], "session": name})
    for c in chains:
        if c["kind"] == "success":
            c["t_star"] = hindsight_seat(fr["Z"][c["idx"]], z_std, eps)
    return chains


def td_targets(chains: list[dict], V: np.ndarray, dt: float, gamma: float, lam: float,
               v_max: float) -> np.ndarray:
    """TD(lambda) returns per frame (NaN where not in a chain); ``V`` = current predictions."""
    G = np.full(len(V), np.nan)
    for c in chains:
        idx, n = c["idx"], len(c["idx"])
        k = c["kind"]
        if k == "success":
            n = c["t_star"]
            G[idx[n:]] = 0.0
            g_next, v_next = 0.0, 0.0
        elif k == "fail":
            g_next = v_next = v_max
        elif k == "boot_first":
            g_next = v_next = V[idx[0]]
        else:
            g_next = v_next = V[idx[-1]]
        for t in range(n - 1, -1, -1):
            g = dt + gamma * ((1.0 - lam) * v_next + lam * g_next)
            G[idx[t]] = g
            g_next, v_next = g, V[idx[t]]
    return G


def split(meta: dict, frac: float, seed: int, always: list[str]) -> set[str]:
    """Held-out episode paths: every rollout under an ``always`` run prefix + ``frac`` of the
    other rollouts and of the demos / excitations (per label), by episode."""
    rng = np.random.default_rng(seed)
    out = set()
    groups: dict = {}
    for e in meta["episodes"]:
        if e["src"] not in ("mpc", "demo", "excitation"):
            continue
        if e["src"] == "mpc" and any(e["run"].startswith(p) for p in always):
            out.add(e["path"])
            continue
        groups.setdefault((e["src"] == "mpc", bool(e["label"])), []).append(e["path"])
    for _, paths in sorted(groups.items()):
        k = round(frac * len(paths))
        out.update(rng.choice(sorted(paths), size=k, replace=False).tolist())
    return out


def frame_weights(chains: list[dict], train: np.ndarray, n: int, mix: dict) -> np.ndarray:
    """Per-frame weight: each source group (mpc / demo+excitation / miles) gets its ``mix`` share
    of the total, uniform over its train frames."""
    grp = np.full(n, "", dtype=object)
    for c in chains:
        g = "mpc" if c["src"] == "mpc" else "demo" if c["src"] in ("demo", "excitation") \
            else "miles"
        grp[c["idx"]] = g
    w = np.zeros(n)
    for g, share in mix.items():
        m = train & (grp == g)
        if m.any():
            w[m] = float(share) / m.sum()
    return w / w.sum()


class Fitter:
    """``v_max sigmoid(mlp(x))`` fitted by weighted expectile loss (seconds; ``expectile`` 0.5 =
    MSE, < 0.5 = the low (optimistic) expectile of the cost targets); one optimiser and cosine
    schedule over every round."""

    def __init__(self, n_in: int, a, v_max: float) -> None:
        import torch

        torch.manual_seed(a.seed)
        dims = [n_in, *[a.width] * a.depth, 1]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float64))
            if i + 2 < len(dims):
                mods.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*mods)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=a.lr, weight_decay=a.wd)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, a.steps * a.rounds)
        self.gen = torch.Generator().manual_seed(a.seed)
        self.a, self.v_max = a, v_max

    def __call__(self, X):
        import torch

        with torch.no_grad():
            return (self.v_max * torch.sigmoid(self.net(torch.as_tensor(X))[:, 0])).numpy()

    def fit(self, X, G, w, log=None) -> None:
        import torch

        a = self.a
        Xt, Gt, p = torch.as_tensor(X), torch.as_tensor(G), torch.as_tensor(w)
        for s in range(a.steps):
            i = torch.multinomial(p, a.batch, replacement=True, generator=self.gen)
            x = Xt[i]
            if a.noise > 0:
                x = x + a.noise * torch.randn(x.shape, dtype=x.dtype, generator=self.gen)
            u = Gt[i] - self.v_max * torch.sigmoid(self.net(x)[:, 0])
            k = torch.where(u > 0, a.expectile, 1.0 - a.expectile)
            loss = (k * u * u).mean() / max(a.expectile, 1.0 - a.expectile)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.sched.step()
        if log is not None:
            log(f"  {a.steps} steps, last batch loss {loss.item():.3f}")


def net_layers(net) -> list[tuple[np.ndarray, np.ndarray]]:
    import torch

    return [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
            for m in net if isinstance(m, torch.nn.Linear)]


def ood_pool(X, n: int, scale: tuple[float, float], min_nn: float, seed: int) -> np.ndarray:
    """Whitened points around the data (Gaussian, radius scale U[lo, hi]) whose nearest data point
    is farther than ``min_nn``: targets V_max (conservatism outside the data)."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    base = X[rng.integers(0, len(X), n)]
    r = rng.uniform(scale[0], scale[1], size=(n, 1))
    d = rng.normal(size=base.shape)
    cand = base + r * d / np.linalg.norm(d, axis=1, keepdims=True)
    nn, _ = cKDTree(X).query(cand, k=1)
    return cand[nn > min_nn]


def model_aug(fr: dict, meta: dict, horizons, stride: int, out) -> dict:
    """LCS predictions ``z_hat(t+H | z_t, p_t, u_t..u_{t+H-1})`` (the deploy's model, recorded
    commanded actions) from every ``stride``-th frame of every episode, with the target frame
    t+H: MPPI queries V on such predicted latents (its knots). Cached per (horizons, stride)."""
    f = Path(out) / "cache" / f"aug_{'-'.join(map(str, horizons))}_s{stride}.npz"
    if f.is_file():
        with np.load(f) as d:
            return {k: d[k] for k in d.files}
    import torch

    from task_common.mppi_lcs import N_P, TorchLcs

    lcs = TorchLcs.from_deploy(meta["deploy"], "cpu", torch.float64)
    betas = lcs.nesterov_betas(25)
    Zs, tg, Hs, st0 = [], [], [], []
    for H in horizons:
        st = np.concatenate([e["off"] + np.arange(0, e["n"] - H, stride)
                             for e in meta["episodes"] if e["n"] > H])
        for b in range(0, len(st), 20000):
            s = st[b:b + 20000]
            z, p = lcs.tensor(fr["Z"][s]), lcs.tensor(fr["p"][s])
            with torch.no_grad():
                for k in range(H):
                    u = lcs.tensor(fr["u"][s + k])
                    z, _ = lcs.step(z, u, p, 25, betas)
                    p = p + u[:, :N_P]
            Zs.append(z.numpy())
        tg.append(st + H)
        st0.append(st)
        Hs.append(np.full(len(st), H))
    d = {"Z": np.concatenate(Zs), "target": np.concatenate(tg), "start": np.concatenate(st0),
         "H": np.concatenate(Hs)}
    np.savez(f, **d)
    return d


def train(a) -> int:
    import torch

    torch.set_num_threads(a.threads)
    fr, meta = load_cache(a.out)
    Z, dt, z_std = fr["Z"], float(fr["dt"]), fr["z_std"]
    gamma = 1.0 - dt / a.tau_s
    v_max = dt / (1.0 - gamma)
    chains = build_chains(meta, fr, a.eps)
    held = split(meta, a.holdout, a.seed, a.always_holdout) if a.holdout > 0 else set()
    train_m = np.zeros(len(Z), bool)
    for c in chains:
        if not set(c["eps"]) & held:
            train_m[c["idx"]] = True
    z_mean = Z[train_m].mean(0)
    X = (Z - z_mean) / z_std
    w = frame_weights(chains, train_m, len(Z), json.loads(a.mix))
    out = Path(a.out) / "models" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"{a.tag}: {len(chains)} chains, {train_m.sum()} train / {(~train_m).sum()} held-out "
        f"frames, {len(held)} held-out episodes; gamma {gamma:.5f} V_max {v_max:.1f} s, eps "
        f"{a.eps}, lambda {a.lam}, expectile {a.expectile}, rounds {a.rounds} x {a.steps} steps, "
        f"mix {a.mix}")
    Xf, wf, tgt = X, w, np.arange(len(Z))
    if a.aug_share > 0:
        aug = model_aug(fr, meta, a.model_aug, a.aug_stride, a.out)
        ok = train_m[aug["target"]] & train_m[aug["start"]] & np.all(np.isfinite(aug["Z"]), 1)
        Xa = (aug["Z"][ok] - z_mean) / z_std
        Xf = np.concatenate([X, Xa])
        wf = np.concatenate([w * (1.0 - a.aug_share), np.full(len(Xa), a.aug_share / len(Xa))])
        tgt = np.concatenate([tgt, aug["target"][ok]])
        log(f"model aug: {len(Xa)} LCS-predicted latents (H {a.model_aug}, stride "
            f"{a.aug_stride}), share {a.aug_share}")
    if a.ood > 0:
        ref = np.concatenate([X[train_m], Xf[len(X):]])  # frames + LCS predictions
        P = ood_pool(ref, a.ood, (a.ood_r[0], a.ood_r[1]), a.ood_nn, a.seed)
        Xf = np.concatenate([Xf, P])
        wf = np.concatenate([wf * (1.0 - a.ood_w), np.full(len(P), a.ood_w / max(1, len(P)))])
        log(f"OOD pool: {len(P)} points (r {a.ood_r}, nn > {a.ood_nn}), weight {a.ood_w}")
    V = np.full(len(Z), 0.5 * v_max)
    fitter = Fitter(X.shape[1], a, v_max)
    for r in range(a.rounds):
        lam = 1.0 if r == 0 and a.lam < 1.0 else a.lam  # round 0: Monte Carlo (V not fitted yet)
        G = td_targets(chains, V, dt, gamma, lam, v_max)
        Gf = np.concatenate([G[tgt], np.full(len(Xf) - len(tgt), v_max)])
        log(f"round {r}: lambda {lam}, target mean {np.nanmean(G[train_m]):.2f} s")
        fitter.fit(Xf, np.nan_to_num(Gf), wf, log=log)
        V = fitter(X)
    net = fitter.net
    G = td_targets(chains, V, dt, gamma, a.lam, v_max)
    rm = lambda m: float(np.sqrt(np.mean((V[m] - G[m]) ** 2)))
    log(f"final rmse vs TD targets: train {rm(train_m):.2f} s, held-out {rm(~train_m):.2f} s")
    meta_out = {"deploy": meta["deploy"], "gamma": gamma, "tau_s": a.tau_s, "dt": dt,
                "eps": a.eps, "lambda": a.lam, "expectile": a.expectile, "rounds": a.rounds,
                "mix": json.loads(a.mix),
                "width": a.width, "depth": a.depth, "steps": a.steps, "ood": a.ood,
                "ood_w": a.ood_w, "ood_r": a.ood_r, "ood_nn": a.ood_nn, "seed": a.seed,
                "holdout": a.holdout, "always_holdout": a.always_holdout,
                "model_aug": a.model_aug, "aug_stride": a.aug_stride, "aug_share": a.aug_share,
                "noise": a.noise,
                "n_train_frames": int(train_m.sum()), "n_heldout_episodes": len(held)}
    lv.save(out / "value.npz", net_layers(net), z_mean, z_std, v_max, meta["encoder_sha256"],
            meta_out)
    (out / "split.json").write_text(json.dumps({"heldout": sorted(held)}, indent=0) + "\n")
    (out / "train.log").write_text("\n".join(lines) + "\n")
    chk = lv.LatentValue.load(out / "value.npz")
    err = np.abs(chk(Z) - V).max()
    log(f"numpy twin max |dV| {err:.2e} s -> {out / 'value.npz'}")
    return 0 if err < 1e-8 else 1


# --- offline evaluation (sim GT for scoring only) -----------------------------------------------

GOAL_WP = "data/lcs/realfirst/cmp/C1_transport/iter1/latent_waypoints.npz"
GOAL_SETS = {"set strict 9": "data/lcs/realfirst/cmp/goal_set/sets/V12h_strict.npz",
             "set 14 seated": "data/lcs/realfirst/cmp/goal_set/sets/V12h_demo14.npz"}


def gt_seat(wrap: np.ndarray) -> int | None:
    """First frame from which the GT wrap stays >= 60 deg (None if not seated at the end)."""
    ok = wrap >= GT_ENGAGED_WRAP
    if not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    return int(bad[-1] + 1) if len(bad) else 0


def auc_low(pos, neg) -> float:
    """P(score_pos < score_neg) + 0.5 P(tie): low score = positive (engaged)."""
    pos, neg = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None]
    return float(np.mean((pos < neg) + 0.5 * (pos == neg))) if pos.size and neg.size \
        else float("nan")


def scores(Z, z_std) -> dict:
    """name -> f(Z) (lower = closer to seated)."""
    with np.load(REPO / GOAL_WP) as f:
        g = np.asarray(f["z_wp"], np.float64)[-1]
    out = {"d single goal": lambda X: np.linalg.norm((X - g) / z_std, axis=-1)}
    for k, p in GOAL_SETS.items():
        if (REPO / p).is_file():
            with np.load(REPO / p) as f:
                G = np.asarray(f["z_goals"], np.float64)
            out[k] = lambda X, G=G: np.min(np.linalg.norm((X[:, None] - G) / z_std, axis=-1), 1)
    return out


def evaluate(a) -> int:
    from scipy.stats import spearmanr

    fr, meta = load_cache(a.out)
    Z, W, dt, z_std = fr["Z"], fr["wrap"], float(fr["dt"]), fr["z_std"]
    tags = a.tag
    rows, report = {}, []
    for tag in tags:
        d = Path(a.out) / "models" / tag
        vf = lv.LatentValue.load(d / "value.npz")
        vf.check_z_std(z_std)
        held = set(json.loads((d / "split.json").read_text())["heldout"])
        sc = {f"V {tag}": vf}
        if not rows:
            sc.update(scores(Z, z_std))
        for name, f in sc.items():
            rows[name] = {"f": f, "held": held}
    held = rows[f"V {tags[0]}"]["held"]
    aug = model_aug(fr, meta, [8, 16, 24], 2, a.out)
    res = {}
    for subset in ("held-out", "all"):
        eps = [e for e in meta["episodes"] if e["src"] == "mpc"
               and (subset == "all" or e["path"] in held)]
        demos = [e for e in meta["episodes"] if e["src"] in ("demo", "excitation")
                 and (subset == "all" or e["path"] in held)]
        fin = np.array([e["off"] + e["n"] - 1 for e in eps])
        oc = np.array([e["outcome"] for e in eps])
        r_sub = {}
        for name, r in rows.items():
            s = r["f"](Z[fin])
            eng = s[oc == ENGAGED]
            ovs = s[np.isin(oc, ["over", "slanted"])]
            ov = s[oc == "over"]
            r_sub[name] = {"auc_eng_vs_over_slanted": auc_low(eng, ovs),
                           "auc_eng_vs_not": auc_low(eng, s[oc != ENGAGED]),
                           "auc_eng_vs_over": auc_low(eng, ov),
                           "median_eng": float(np.median(eng)) if len(eng) else None,
                           "median_over": float(np.median(ov)) if len(ov) else None,
                           "over_above_eng_p90": float(np.mean(ov > np.percentile(eng, 90)))
                           if len(ov) and len(eng) else None}
            # time-to-go on successes (GT seat), frames before the GT seat
            xs, ys, per = [], [], []
            for e in [*(x for x in eps if x["label"]), *(x for x in demos if x["label"])]:
                w = W[e["off"]:e["off"] + e["n"]]
                g = gt_seat(w)
                if g is None or g < 5:
                    continue
                v = r["f"](Z[e["off"]:e["off"] + g])
                ttg = (g - np.arange(g)) * dt
                xs.append(v)
                ys.append(ttg)
                per.append(spearmanr(v, ttg).statistic)
            if xs:
                r_sub[name].update(rho_ttg_pooled=float(spearmanr(np.concatenate(xs),
                                                                  np.concatenate(ys)).statistic),
                                   rho_ttg_episode_median=float(np.nanmedian(per)),
                                   n_ttg_episodes=len(per))
        # distance-matched over vs engaged finals (the blocked-state test)
        ds = rows["d single goal"]["f"](Z[fin])
        for name, r in rows.items():
            s = r["f"](Z[fin])
            wins = []
            for i in np.flatnonzero(oc == "over"):
                m = (oc == ENGAGED) & (np.abs(ds - ds[i]) < a.match)
                wins += list((s[i] > s[m]) + 0.5 * (s[i] == s[m]))
            r_sub[name]["over_vs_matched_engaged"] = float(np.mean(wins)) if wins else None
            r_sub[name]["n_matched_pairs"] = len(wins)
        # on the model's 24-step predictions (what MPPI's terminal knot sees) vs the GT time to go
        sel = {e["path"] for e in [*eps, *demos]}
        ok_ep = np.zeros(len(Z), bool)
        ttg = np.full(len(Z), np.nan)
        for e in meta["episodes"]:
            if e["path"] not in sel:
                continue
            ok_ep[e["off"]:e["off"] + e["n"]] = True
            g = gt_seat(W[e["off"]:e["off"] + e["n"]])
            t = np.arange(e["n"])
            ttg[e["off"]:e["off"] + e["n"]] = (np.maximum(g - t, 0) * dt if g is not None
                                               else np.inf)
        m = (aug["H"] == 24) & ok_ep[aug["target"]] & np.all(np.isfinite(aug["Z"]), 1)
        m &= aug["start"] % 4 == 0
        y = ttg[aug["target"][m]]
        y = np.where(np.isfinite(y), y, 1e3)
        for name, r in rows.items():
            r_sub[name]["rho_pred24"] = float(spearmanr(r["f"](aug["Z"][m]), y).statistic)
            r_sub[name]["rho_true24"] = float(spearmanr(r["f"](Z[aug["target"][m]]), y).statistic)
        res[subset] = {"n_episodes": len(eps), "outcomes": {k: int(np.sum(oc == k))
                                                           for k in np.unique(oc)},
                       "scores": r_sub}
    out = Path(a.out) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{'_'.join(tags)}.json").write_text(json.dumps(res, indent=1) + "\n")
    for subset, rr in res.items():
        report.append(f"\n## {subset} rollouts ({rr['n_episodes']}: {rr['outcomes']})\n")
        report.append("| score | AUC eng vs over+slanted | eng vs not | eng vs over | median eng / "
                      "over | over > eng p90 | over vs d-matched eng | rho time-to-go pooled / "
                      "episode median (n) | rho GT value: true z / LCS 24-step z |")
        report.append("|---|---|---|---|---|---|---|---|---|")
        for name, s in rr["scores"].items():
            f = lambda x: "-" if x is None else f"{x:.3f}"
            report.append(
                f"| {name} | {f(s['auc_eng_vs_over_slanted'])} | {f(s['auc_eng_vs_not'])} | "
                f"{f(s['auc_eng_vs_over'])} | {f(s['median_eng'])} / {f(s['median_over'])} | "
                f"{f(s['over_above_eng_p90'])} | {f(s['over_vs_matched_engaged'])} "
                f"({s['n_matched_pairs']}) | {f(s.get('rho_ttg_pooled'))} / "
                f"{f(s.get('rho_ttg_episode_median'))} ({s.get('n_ttg_episodes', 0)}) | "
                f"{f(s.get('rho_true24'))} / {f(s.get('rho_pred24'))} |")
    txt = "\n".join(report)
    (out / f"eval_{'_'.join(tags)}.md").write_text(txt + "\n")
    print(txt)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--deploy", required=True)
    e.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--tag", required=True)
    t.add_argument("--tau-s", type=float, default=20.0, help="discount horizon: gamma = 1 - dt/tau")
    t.add_argument("--eps", type=float, default=1.5, help="hindsight seat: whitened distance")
    t.add_argument("--lam", type=float, default=1.0, help="TD(lambda); 1 = Monte Carlo")
    t.add_argument("--rounds", type=int, default=3, help="fitted rounds (bootstraps)")
    t.add_argument("--mix", default='{"mpc": 0.5, "demo": 0.25, "miles": 0.25}')
    t.add_argument("--width", type=int, default=128)
    t.add_argument("--depth", type=int, default=2)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--expectile", type=float, default=0.5,
                   help="0.5 = MSE; < 0.5 = low (optimistic) expectile of the cost targets")
    t.add_argument("--wd", type=float, default=1e-5)
    t.add_argument("--ood", type=int, default=0, help="OOD pool candidates (0 = off)")
    t.add_argument("--ood-r", type=float, nargs=2, default=[1.0, 4.0])
    t.add_argument("--ood-nn", type=float, default=1.0)
    t.add_argument("--ood-w", type=float, default=0.1)
    t.add_argument("--model-aug", type=lambda x: [int(v) for v in x.split(",")],
                   default=[8, 16, 24], help="LCS-prediction horizons (frames)")
    t.add_argument("--aug-stride", type=int, default=2)
    t.add_argument("--aug-share", type=float, default=0.0,
                   help="batch share of LCS-predicted latents (0 = off)")
    t.add_argument("--noise", type=float, default=0.0, help="whitened input noise (training)")
    t.add_argument("--holdout", type=float, default=0.2)
    t.add_argument("--always-holdout", nargs="*", default=["data/lcs/realfirst/cmp/noprior"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=4)
    v = sub.add_parser("eval")
    v.add_argument("--out", required=True)
    v.add_argument("--tag", nargs="+", required=True, help="models/<tag> (split of the first)")
    v.add_argument("--match", type=float, default=0.5, help="d-matched pairs: |d_single| window")
    a = ap.parse_args()
    if a.cmd == "encode":
        return encode(a)
    if a.cmd == "train":
        return train(a)
    return evaluate(a)


if __name__ == "__main__":
    sys.exit(main())
