#!/usr/bin/env python3
"""V12-36 offline evidence on a finished loop root: off-track shares and task-weighted buffers.

``fractions``: for every iteration i of ``--root``, the MPC rollouts collected in i (``attempts``
+ ``labels``) scored with ``task_common.onpolicy_weighting`` against iteration i's
latent_waypoints.npz (latents = the recorded ``sim_latent``, the deploy encoder's): per tol_k,
rows off track, frames past the cut (+ after_s), split by the human label; the buffer's MPC
frames down-weighted. GT outcomes are report-only. -> ``<out>/offtrack_fractions.{json,md}``.
``buffer``: iteration ``--iter``'s refit input B_i (= iter<i-1>/buffer.json) with the rule's
``frame_weights`` on every MPC row (each with its own iteration's waypoints) and the anchors
re-scaled to ``--anchor-frac`` of the frame-weighted total -> ``--buffer-out``.
``summary``: refit arms (``--arms tag=label``; gate/<tag>/behaviour_gate.json + the refit report
under ``--lcs-out``) -> ``<out>/report_arms.md``.

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/lcs/refit_weighting_offline.py \\
        fractions --root data/lcs/realfirst/cmp/v12_35 --out data/lcs/realfirst/cmp/refit_weighting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from task_common import buffer_policy
from task_common import onpolicy_weighting as ow


def waypoints(it: Path) -> dict:
    with np.load(it / "latent_waypoints.npz", allow_pickle=True) as d:
        return {k: np.asarray(d[k]) for k in ("z_traj", "z_std", "frames", "tol", "dt",
                                              "first_hold", "latch")}


def tail(z: np.ndarray, wp: dict, n_last: int) -> dict:
    """Last ``n_last`` frames: nearest goal-demo frame and tube ratio over every demo frame vs
    over the demo frames from its first hold on (the seated part)."""
    tf = ow.frame_tols(wp["frames"], wp["tol"], len(wp["z_traj"]))
    zl = z[-n_last:]
    d = np.linalg.norm((zl[:, None] - wp["z_traj"][None]) / wp["z_std"], axis=-1) / tf
    h = int(wp["first_hold"])
    return {"nearest_frame": int(np.median(d.argmin(1))), "ratio": float(np.median(d.min(1))),
            "ratio_seated": float(np.median(d[:, h:].min(1)))}


def episode(path: str) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=True) as d:
        ut = np.asarray(d["utime"], np.float64)
        return np.asarray(d["sim_latent"], np.float64), float(np.median(np.diff(ut))) * 1e-6


def rule(k: float, after_s: float, streak: int, w: float = 0.2, fail_w: float = 1.0) -> dict:
    return {"off_track": {"tol_k": k, "after_s": after_s, "streak": streak},
            "off_track_weight": w, "fail_weight": fail_w}


def fractions(a) -> int:
    root, out = Path(a.root), Path(a.out)
    iters = sorted(int(p.name[4:]) for p in root.glob("iter*") if (p / "labels.json").is_file()
                   and (p / "latent_waypoints.npz").is_file())
    rows, per_ep = [], []
    rules = [(k, s) for k in a.tol_k for s in a.streak]
    for i in iters:
        it = root / f"iter{i}"
        wp = waypoints(it)
        lab = {e["episode"]: e for e in json.loads((it / "labels.json").read_text())["episodes"]}
        eps = [x for x in json.loads((it / "attempts.json").read_text())["attempts"]
               if x["episode"]]
        model = json.loads((it / "model.json").read_text())["id"]
        for x in eps:
            z, dt = episode(x["episode"])
            succ = bool(lab[x["episode"]]["success"])
            rec = {"iter": i, "model": model, "episode": x["episode"], "success": succ,
                   "outcome_gt": x["outcome_gt"], "n": len(z)}
            for k, s in rules:
                _, info = ow.episode_weights(z, wp, dt, succ, rule(k, a.after_s, s))
                rec[f"k{k:g}_s{s}"] = {kk: info[kk] for kk in (
                    "first_off_track", "weighted_from", "n_down", "ratio_max", "ratio_median")}
            rec["tail_3s"] = tail(z, wp, round(3.0 / dt))
            per_ep.append(rec)
    for i in iters:
        es = [e for e in per_ep if e["iter"] == i]
        r = {"iter": i, "model": es[0]["model"], "n_rows": len(es),
             "n_success": sum(e["success"] for e in es), "frames": sum(e["n"] for e in es),
             "outcomes": [e["outcome_gt"] for e in es]}
        for k, s in rules:
            key = f"k{k:g}_s{s}"
            r[key] = {
                "rows_off": sum(e[key]["weighted_from"] is not None for e in es),
                "frames_down": sum(e[key]["n_down"] for e in es),
                "share": sum(e[key]["n_down"] for e in es) / max(1, sum(e["n"] for e in es)),
                "share_fail": (sum(e[key]["n_down"] for e in es if not e["success"])
                               / max(1, sum(e["n"] for e in es if not e["success"]))),
                "share_success": (sum(e[key]["n_down"] for e in es if e["success"])
                                  / max(1, sum(e["n"] for e in es if e["success"])))}
        rows.append(r)
    cum = []  # the buffer B_{i+1}: every MPC row up to iteration i
    for i in iters:
        es = [e for e in per_ep if e["iter"] <= i]
        c = {"iter": i, "frames": sum(e["n"] for e in es)}
        for k, s in rules:
            key = f"k{k:g}_s{s}"
            c[key] = sum(e[key]["n_down"] for e in es) / max(1, c["frames"])
        cum.append(c)
    ratio = {"median": float(np.median([e[f"k{a.tol_k[0]:g}_s1"]["ratio_median"]
                                        for e in per_ep])),
             "max_median": float(np.median([e[f"k{a.tol_k[0]:g}_s1"]["ratio_max"]
                                            for e in per_ep]))}
    doc = {"root": str(root), "after_s": a.after_s, "rules": [f"k{k:g}_s{s}" for k, s in rules],
           "per_iter": rows, "buffer_cumulative": cum, "episodes": per_ep,
           "tube_ratio_over_episodes": ratio}
    out.mkdir(parents=True, exist_ok=True)
    (out / "offtrack_fractions.json").write_text(json.dumps(doc, indent=1) + "\n")
    md = [f"# Off-track share of on-policy frames ({root.name})", "",
          (f"Rule: first frame whose latent (recorded `sim_latent`) is farther than tol_k x the "
           f"waypoint tol from every goal-demo latent (iteration i's latent_waypoints.npz) for "
           f"`streak` frames; frames from there + {a.after_s:g} s are down-weighted. GT outcomes "
           "report-only."), "",
          "| iter | model | rows (succ) | GT outcomes | frames | "
          + " | ".join(f"tol_k {k:g}{'' if s == 1 else f' streak {s}'}: rows off, frames down "
                       "(fail / succ)" for k, s in rules) + " |",
          "|---" * (5 + len(rules)) + "|"]
    for r in rows:
        md.append(f"| {r['iter']} | {r['model']} | {r['n_rows']} ({r['n_success']}) | "
                  f"{', '.join(r['outcomes'])} | {r['frames']} | " + " | ".join(
                      f"{r[f'k{k:g}_s{s}']['rows_off']}, {r[f'k{k:g}_s{s}']['share']:.0%} "
                      f"({r[f'k{k:g}_s{s}']['share_fail']:.0%} / "
                      f"{r[f'k{k:g}_s{s}']['share_success']:.0%})" for k, s in rules) + " |")
    md += ["", ("Buffer B_{i+1} (every MPC row collected up to iteration i): share of MPC frames "
                "down-weighted."), "", "| iter | MPC frames | " + " | ".join(
               f"tol_k {k:g}{'' if s == 1 else f' s{s}'}" for k, s in rules) + " |",
           "|---" * (2 + len(rules)) + "|"]
    for c in cum:
        md.append(f"| {c['iter']} | {c['frames']} | " + " | ".join(
            f"{c[f'k{k:g}_s{s}']:.0%}" for k, s in rules) + " |")
    md += ["", (f"Tube ratio (min_j d_j / tol_j) over episodes: median of per-episode medians "
                f"{ratio['median']:.2f}, median of per-episode maxima {ratio['max_median']:.2f}."),
           ""]
    md += [("Where the rollouts end (last 3 s, medians): nearest goal-demo frame (latch "
            f"{int(wp['latch'])}, first hold {int(wp['first_hold'])}), tube ratio over every "
            "demo frame vs over the seated demo frames (first hold on)."), "",
           ("| iter | GT outcome (label) | nearest demo frame | ratio, all frames | ratio, "
            "seated frames |"), "|---" * 5 + "|"]
    for e in per_ep:
        t = e["tail_3s"]
        md.append(f"| {e['iter']} | {e['outcome_gt']} ({'success' if e['success'] else 'fail'}) "
                  f"| {t['nearest_frame']} | {t['ratio']:.2f} | {t['ratio_seated']:.2f} |")
    md.append("")
    (out / "offtrack_fractions.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


def weighted_buffer(a) -> int:
    root = Path(a.root)
    src = root / f"iter{a.iter - 1}" / "buffer.json"
    buf = json.loads(src.read_text())
    r = rule(a.tol_k[0], a.after_s, a.streak[0], a.off_track_weight, a.fail_weight)
    wps, frames, n_down, n_mpc = {}, {}, 0, 0
    rows = []
    for x in buf["rollouts"]:
        x = {k: v for k, v in x.items() if k != "frame_weights"}
        with np.load(x["episode"], allow_pickle=True) as d:
            frames[x["episode"]] = len(d["actions"])
        if x["kind"] == "mpc":
            j = int(x["iter"])
            wps[j] = wps.get(j) or waypoints(root / f"iter{j}")
            z, dt = episode(x["episode"])
            fw, info = ow.episode_weights(z, wps[j], dt, bool(x["label_human"]), r)
            x.update(frame_weights=fw, off_track=info)
            n_down += info["n_down"]
            n_mpc += info["n_frames"]
        rows.append(x)
    eff = {x["episode"]: ow.effective_frames(x, frames[x["episode"]]) for x in rows}
    for x in rows:  # anchors back to the pool's base weights before re-scaling
        if x["kind"] in a.anchor_kinds:
            x["weight"] = 1.0
    rows, anchor = buffer_policy.anchor_weights(rows, a.anchor_kinds, a.anchor_frac, eff)
    out = {**buf, "rollouts": rows, "anchor": anchor, "source": f"{src} + {r}",
           "offline": {"rule": r, "mpc_frames": n_mpc, "mpc_frames_down": n_down,
                       "anchor_frac": a.anchor_frac}}
    Path(a.buffer_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.buffer_out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{a.buffer_out}: {len(rows)} rows, MPC frames down {n_down}/{n_mpc}, anchor share "
          f"{anchor.get('share_after')}")
    return 0


def summary(a) -> int:
    """Arms ``tag=label`` -> ``<out>/report.md`` (refit report + behaviour gate vs the ref)."""
    out = Path(a.out)
    md = [f"# Iteration-2 refit arms (V12-35 data, CPU) vs the seed{a.title}", "",
          ("Behaviour gate = the loop's iteration-2 call (behaviour_gate.py, MPPI noise 0, the "
           "3 demos + iteration 1's 3 successes, stride 4) with the candidate vs V15a (the seed) "
           "on CPU. cos = median cosine of MPPI u_0 to the next recorded actions; hold = Franka "
           "u_0 y on demo hold frames (mm/step)."), "",
          ("| arm | init | buffer (MPC frames down) | epochs | one-step / h7 mm (held-out) | "
           "rho(A) | approach cos | descent cos | success cos | hold push | gate |"),
          "|---" * 11 + "|"]
    ref = None
    for spec in a.arms:
        tag, label = spec.split("=", 1)
        g = json.loads((out / "gate" / tag / "behaviour_gate.json").read_text())
        rp = Path(a.lcs_out) / "refit" / tag.removesuffix("_noprior") / "refit_report.json"
        r = json.loads(rp.read_text()) if rp.is_file() else None
        cos = {ph: (g["rows"]["cand"].get(ph) or {}).get("cos_median")
               for ph in ("approach", "descent", "success")}
        ref = ref or ({ph: (g["rows"]["ref"].get(ph) or {}).get("cos_median")
                       for ph in ("approach", "descent", "success")},
                      g["decision"]["hold_push_mm"]["ref"])
        down = "-"
        if r is not None:
            b = json.loads(Path(r["buffer"]).read_text())
            off = b.get("offline")
            down = f"{off['mpc_frames_down']}/{off['mpc_frames']}" if off else "0 (as run)"
            init = "seed V15a" if "r0_seed" in r["init_checkpoint"] else Path(
                r["init_checkpoint"]).parent.name
            m = r["metrics"]
            pred = (f"{m['one_step_chamfer_mm']:.2f} / {m['open_loop7_chamfer_mm']:.2f} | "
                    f"{m['rho_A']:.4f}")
            ep = r["epochs"]
        else:
            init, pred, ep = "-", "- | -", "-"
        md.append(f"| {label} | {init} | {down} | {ep} | {pred} | "
                  + " | ".join("-" if cos[ph] is None else f"{cos[ph]:.2f}"
                               for ph in ("approach", "descent", "success"))
                  + f" | {g['decision']['hold_push_mm']['cand']:+.2f} | "
                  + ("accept" if g["decision"]["accept"] else "reject") + " |")
    if ref:
        md.append("| V15a (reference, seed) | - | - | - | - | - | " + " | ".join(
            "-" if ref[0][ph] is None else f"{ref[0][ph]:.2f}"
            for ph in ("approach", "descent", "success")) + f" | {ref[1]:+.2f} | - |")
    md.append("")
    (out / a.report).write_text("\n".join(md))
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("fractions", "buffer", "summary"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=str(REPO / "data/lcs/realfirst/cmp/refit_weighting"))
    ap.add_argument("--tol-k", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    ap.add_argument("--streak", type=int, nargs="+", default=[1])
    ap.add_argument("--after-s", type=float, default=1.0)
    ap.add_argument("--iter", type=int, default=2)
    ap.add_argument("--off-track-weight", type=float, default=0.2)
    ap.add_argument("--fail-weight", type=float, default=1.0)
    ap.add_argument("--anchor-kinds", nargs="+", default=["demo", "miles"])
    ap.add_argument("--anchor-frac", type=float, default=0.4)
    ap.add_argument("--buffer-out", default=None)
    ap.add_argument("--arms", nargs="+", default=[], help="summary: tag=label (gate/<tag>/)")
    ap.add_argument("--report", default="report_arms.md")
    ap.add_argument("--title", default="")
    ap.add_argument("--lcs-out", default=str(Path.home() / "git/lcs_learning/outputs/"
                                             "sim_belt_v12_20261006/refit_weighting"))
    a = ap.parse_args()
    return {"fractions": fractions, "buffer": weighted_buffer, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
