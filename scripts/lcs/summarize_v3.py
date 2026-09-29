#!/usr/bin/env python3
"""Summarise the eval-v3 matrix: {v2, v3_mix, v3_ft} x the free-space / approach / contact runs.

Reads ``<root>/<model>/<run>.json`` (``eval_motion_primitives.py --out``), the runs' npz extras
and lcs_learning ``eval_results.json`` (per-input response); adds open-loop h3 / h7 rollouts on
the contact held-out run. Writes ``<root>/summary.md`` and ``summary.json``: pooled matrix,
breakdowns, gates G1-G5, PICK.

Run:
    uv run --frozen python scripts/lcs/summarize_v3.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from task_common import prediction_video as pv
from task_common.lcs_dataset import ACTION_DIM_NAMES as U_NAMES

ROOT = REPO_ROOT / "data" / "lcs" / "free_space" / "eval_v3"
EVAL_RESULTS = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v3_20260928/eval_results.json")
MODELS = ("v2", "v3_mix", "v3_ft")
LABEL = {"v2": "v2", "v3_mix": "A v3_mix", "v3_ft": "B v3_ft"}
RUNS = ("v1", "v2-big", "v2-big-fast", "fs_heldout_nominal", "fs_heldout_set3",
        "approach_heldout", "contact_heldout")
DEFORM = ("fs_heldout_nominal", "fs_heldout_set3")
KEYS = ("model", "recon", "nomotion", "pred_vs_recon", "nomotion_decoded")
G1_MAX, G2_MIN, G3_MIN, G4_MAX, G5_RATIO = 0.635, 0.90, 0.50, 0.6, 0.8
HORIZONS = (3, 7)


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def tuples(res: dict, run: Path) -> list[dict]:
    """Per episode: per-tuple series + the npz extras the breakdowns need."""
    index = {r["file"]: r for r in load_json(run / "index.json")["episodes"]}
    out = []
    for name, e in res["primitives"].items():
        s = e["series"]
        with np.load(run / e["file"], allow_pickle=True) as d:
            n = len(s["moving"])
            x = {k: np.asarray(s[f"rmse_{k}_mm"]) for k in KEYS}
            x.update(name=name, file=e["file"], entry=e, moving=np.asarray(s["moving"], bool),
                     row=index[e["file"]], family=name.split("#")[0])
            if "sim_in_contact" in d.files:
                x["contact"] = np.asarray(d["sim_in_contact"], bool)[:n]
                x["contact_full"] = np.asarray(d["sim_in_contact"], bool)
            for k in ("sim_outcome_start", "sim_outcome_place3", "sim_outcome", "sim_family"):
                if k in d.files:
                    x[k] = str(d[k])
            if "sim_approach" in d.files:
                x["yaw"] = float(json.loads(str(d["sim_approach"]))["yaw_deg"])
        out.append(x)
    return out


def pool(eps: list[dict], mask_fn=lambda e: e["moving"]) -> dict | None:
    cat = {k: [] for k in KEYS}
    for e in eps:
        m = mask_fn(e)
        for k in KEYS:
            cat[k].append(e[k][m])
    c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cat.items()}
    n = len(c["model"])
    if not n:
        return None
    return {"tuples": n, **{k: float(c[k].mean()) for k in KEYS},
            "dyn_skill": float(1.0 - c["pred_vs_recon"].mean() / c["nomotion_decoded"].mean()),
            "below_nomotion": float((c["model"] < c["nomotion"]).mean())}


def yaw_bin(yaw: float) -> str:
    a = abs(yaw)
    return "|yaw|<10" if a < 10 else ("10-20" if a <= 20 else ">20")


def rollouts(res: dict, run: Path, eps: list[dict]) -> dict:
    """Open-loop h-step belt RMSE from every start, split by whether [s, s+h] crosses a
    ``sim_in_contact`` flip; copy = belt_s held."""
    model = pv.OneStepModel(Path(res["deploy"]), Path(res["decoder"]), "lcp")
    acc = {h: {"cross": [], "none": [], "copy_cross": [], "copy_none": []} for h in HORIZONS}
    for e in eps:
        ep = pv.Episode.load(run / e["file"])
        z = model.enc.encode_batch(ep.pcd, ep.prop, ep.belt).astype(np.float64)
        c = e["contact_full"]
        for h in HORIZONS:
            s = np.arange(0, ep.frames - h)
            if not len(s):
                continue
            zk = z[s]
            for k in range(h):
                zk = model.step(zk, ep.u[s + k])
            err = pv.rmse_mm(model.decode(zk), ep.belt[s + h])
            cp = pv.rmse_mm(ep.belt[s], ep.belt[s + h])
            cross = np.array([c[a:a + h + 1].min() != c[a:a + h + 1].max() for a in s])
            acc[h]["cross"].append(err[cross])
            acc[h]["none"].append(err[~cross])
            acc[h]["copy_cross"].append(cp[cross])
            acc[h]["copy_none"].append(cp[~cross])
    out = {}
    for h, d in acc.items():
        cat = {k: np.concatenate(v) for k, v in d.items()}
        out[f"h{h}"] = {k: {"mean": float(v.mean()) if len(v) else None, "n": len(v)}
                        for k, v in cat.items()}
    flips = [e for e in eps if e["contact_full"].min() != e["contact_full"].max()]
    one = [e["model"][np.flatnonzero(np.diff(e["contact_full"].astype(int)))
                      .clip(max=len(e["model"]) - 1)] for e in flips]
    out["one_step_at_flip"] = {"mean": float(np.concatenate(one).mean()) if one else None,
                               "n": int(sum(len(x) for x in one)), "episodes": len(flips)}
    return out


def f(x, d: int = 3) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--eval-results", type=Path, default=EVAL_RESULTS)
    args = p.parse_args()
    res = {m: {r: load_json(args.root / m / f"{r}.json") for r in RUNS} for m in MODELS}
    data = {m: {r: tuples(res[m][r], Path(res[m][r]["run"])) for r in RUNS} for m in MODELS}
    S: dict = {"models": {}, "files": {m: {r: str(args.root / m / f"{r}.json") for r in RUNS}
                                       for m in MODELS}}
    md = ["# eval-v3 matrix (belt RMSE mm; exact-LCP one-step; numpy export)", ""]

    # pooled matrix
    md += ["## Pooled matrix (motion tuples; tuple-weighted over the run's episodes)", "",
           ("| run | model | eps | tuples | one-step | recon | no-motion | dyn-only | dyn skill | "
            "model < no-motion | gain pred min / median | one-step (all tuples) |"),
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    matrix: dict = {}
    for r in RUNS:
        for m in MODELS:
            eps = data[m][r]
            mo, al = pool(eps), pool(eps, lambda e: np.ones_like(e["moving"]))
            g = [e["entry"]["displacement_gain"]["pred"] for e in eps]
            matrix.setdefault(r, {})[m] = {"motion": mo, "all": al, "episodes": len(eps),
                                           "gain_pred_min": float(min(g)),
                                           "gain_pred_median": float(np.median(g))}
            md.append(f"| {r} | {LABEL[m]} | {len(eps)} | {mo['tuples']} | {f(mo['model'])} | "
                      f"{f(mo['recon'])} | {f(mo['nomotion'])} | {f(mo['pred_vs_recon'])} | "
                      f"{f(mo['dyn_skill'], 2)} | {pct(mo['below_nomotion'])} | "
                      f"{min(g):.2f} / {np.median(g):.2f} | {f(al['model'])} |")
    S["matrix"] = matrix

    # test split (G1) + v2 reproduction
    md += ["", "## Test split (v2 insertion test, 60 eps) — recomputed per model", "",
           "| model | one-step | recon | no-motion | dyn skill | NN dist p50 / p95 |",
           "|---|---|---|---|---|---|"]
    for m in MODELS:
        t = res[m]["v2-big"]["reference_test_split"]
        S["models"].setdefault(m, {})["test"] = {k: t[k]["mean"] for k in ("model", "recon",
                                                                             "nomotion")}
        md.append(f"| {LABEL[m]} | {t['model']['mean']:.3f} | {t['recon']['mean']:.3f} | "
                  f"{t['nomotion']['mean']:.3f} | {t['dyn_skill']:.2f} | "
                  f"{t['nn_train_dist']['p50']:.3f} / {t['nn_train_dist']['p95']:.3f} |")
    repro = {}
    for r, d in (("v1", "20260927-free-space-v1"), ("v2-big", "20260927-free-space-v2-big"),
                 ("v2-big-fast", "20260927-free-space-v2-big-fast")):
        ref = load_json(REPO_ROOT / "data/lcs/motion_primitives" / d / "eval.json")["primitives"]
        new = res["v2"][r]["primitives"]
        repro[r] = max(abs(new[k][s][q]["mean"] - ref[k][s][q]["mean"])
                       for k in ref for s in ("all", "motion") for q in KEYS)
    S["v2_reproduction_max_abs_diff_mm"] = repro
    md += ["", "v2 vs the archived `eval.json` (max |diff| of every per-primitive mean, mm): "
           + ", ".join(f"{k} {v:.2e}" for k, v in repro.items()), ""]

    # breakdowns
    md += ["## Free-space held-out (nominal + set3) by family (motion tuples)", "",
           "| family | eps | " + " | ".join(f"{LABEL[m]} one-step / recon / <nm / gain"
                                            for m in MODELS) + " | no-motion |",
           "|---|---|" + "---|" * len(MODELS) + "---|"]
    fams = sorted({e["family"] for r in DEFORM for e in data["v2"][r]})
    S["fs_by_family"] = {}
    for fam in fams:
        cells, nm = [], None
        for m in MODELS:
            eps = [e for r in DEFORM for e in data[m][r] if e["family"] == fam]
            po = pool(eps)
            g = min(e["entry"]["displacement_gain"]["pred"] for e in eps)
            S["fs_by_family"].setdefault(fam, {})[m] = {**po, "gain_pred_min": g}
            cells.append(f"{f(po['model'])} / {f(po['recon'])} / {pct(po['below_nomotion'])} / "
                         f"{g:.2f}")
            nm = po["nomotion"]
        md.append(f"| {fam} | {len(eps)} | " + " | ".join(cells) + f" | {f(nm)} |")

    def grouped(title: str, run: str, key_fn, masks: dict, skey: str) -> None:
        nonlocal md
        md += ["", f"## {title}", "",
               "| group | " + " | ".join(f"{LABEL[m]} one-step / recon / dyn skill / <nm"
                                        for m in MODELS) + " | no-motion | tuples |",
               "|---|" + "---|" * len(MODELS) + "---|---|"]
        S[skey] = {}
        keys = sorted({key_fn(e) for e in data["v2"][run]})
        rows = [(f"{k}", (lambda e, k=k: key_fn(e) == k), None) for k in keys]
        rows += [(k, None, fn) for k, fn in masks.items()]
        for label, sel, fn in rows:
            cells, po = [], None
            for m in MODELS:
                eps = [e for e in data[m][run] if sel is None or sel(e)]
                po = pool(eps, fn or (lambda e: e["moving"]))
                S[skey].setdefault(label, {})[m] = po
                cells.append("-" if po is None else f"{f(po['model'])} / {f(po['recon'])} / "
                             f"{f(po['dyn_skill'], 2)} / {pct(po['below_nomotion'])}")
            md.append(f"| {label} | " + " | ".join(cells) + f" | "
                      f"{'-' if po is None else f(po['nomotion'])} | "
                      f"{'-' if po is None else po['tuples']} |")

    grouped("Contact held-out by frame state and by `outcome_start` (motion tuples)",
            "contact_heldout", lambda e: f"outcome_start:{e['sim_outcome_start']}",
            {"frames:in_contact": lambda e: e["moving"] & e["contact"],
             "frames:free": lambda e: e["moving"] & ~e["contact"],
             "all frames (incl. pre-hold)": lambda e: np.ones_like(e["moving"])}, "contact_by")
    appr = data["v2"]["approach_heldout"]
    grouped("Approach held-out by yaw bin, `outcome_place3` and end label (motion tuples)",
            "approach_heldout", lambda e: f"yaw:{yaw_bin(e['yaw'])}",
            {**{f"place3:{lab}": (lambda e, lab=lab: e["moving"]
                                  & (e["sim_outcome_place3"] == lab))
                for lab in sorted({e["sim_outcome_place3"] for e in appr})},
             **{f"end:{lab}": (lambda e, lab=lab: e["moving"] & (e["sim_outcome"] == lab))
                for lab in sorted({e["sim_outcome"] for e in appr})},
             "frames:in_contact": lambda e: e["moving"] & e["contact"],
             "frames:free": lambda e: e["moving"] & ~e["contact"]}, "approach_by")

    md += ["", "## Contact held-out open-loop rollouts across `sim_in_contact` flips", "",
           "window [s, s+h] crosses a flip vs not; copy = belt_s held", "",
           ("| model | h3 cross / none | h7 cross / none | copy h3 cross / none | "
            "copy h7 cross / none | windows h7 cross / none | one-step at flip (n) |"),
           "|---|---|---|---|---|---|---|"]
    S["contact_rollouts"] = {}
    for m in MODELS:
        ro = rollouts(res[m]["contact_heldout"], Path(res[m]["contact_heldout"]["run"]),
                      data[m]["contact_heldout"])
        S["contact_rollouts"][m] = ro
        h3, h7 = ro["h3"], ro["h7"]
        md.append(f"| {LABEL[m]} | {f(h3['cross']['mean'])} / {f(h3['none']['mean'])} | "
                  f"{f(h7['cross']['mean'])} / {f(h7['none']['mean'])} | "
                  f"{f(h3['copy_cross']['mean'])} / {f(h3['copy_none']['mean'])} | "
                  f"{f(h7['copy_cross']['mean'])} / {f(h7['copy_none']['mean'])} | "
                  f"{h7['cross']['n']} / {h7['none']['n']} | "
                  f"{f(ro['one_step_at_flip']['mean'])} ({ro['one_step_at_flip']['n']}) |")

    # gates
    gates: dict = {}
    for m in MODELS:
        g = {}
        t1 = S["models"][m]["test"]["model"]
        g["G1"] = {"value": t1, "pass": t1 <= G1_MAX}
        gains = [(f"v2-big/{e['name']}", e["entry"]["displacement_gain"]["pred"])
                 for e in data[m]["v2-big"]]
        gains += [(f"{r}/{e['name']}", e["entry"]["displacement_gain"]["pred"])
                  for r in DEFORM for e in data[m][r] if e["family"] != "hold_only"]
        fails = [(k, round(v, 3)) for k, v in gains if v < G2_MIN]
        g["G2"] = {"min": min(v for _, v in gains), "n": len(gains), "n_fail": len(fails),
                   "fails": fails, "pass": not fails}
        dm = pool([e for r in DEFORM for e in data[m][r]])["below_nomotion"]
        fast = res[m]["v2-big-fast"]["primitives"]["both_up_down"][
            "model_below_nomotion_frac"]["motion"]
        cm = pool(data[m]["contact_heldout"], lambda e: e["moving"] & e["contact"])[
            "below_nomotion"]
        g["G3"] = {"deform_heldout": dm, "fast_both_up_down": fast, "contact_frames": cm,
                   "v1": pool(data[m]["v1"])["below_nomotion"],
                   "v2-big": pool(data[m]["v2-big"])["below_nomotion"],
                   "pass": min(dm, fast, cm) >= G3_MIN}
        rd = pool([e for r in DEFORM for e in data[m][r]])["recon"]
        rc = pool(data[m]["contact_heldout"], lambda e: np.ones_like(e["moving"]))["recon"]
        g["G4"] = {"deform_heldout_motion": rd, "contact_frames": rc,
                   "pass": max(rd, rc) <= G4_MAX}
        g["one_step"] = {r: matrix[r][m]["all"]["model"] for r in ("approach_heldout",
                                                                   "contact_heldout")}
        gates[m] = g
    for m in MODELS:
        v2 = gates["v2"]["one_step"]
        ratio = {r: gates[m]["one_step"][r] / v2[r] for r in v2}
        gates[m]["G5"] = {"ratio": ratio, "pass": max(ratio.values()) <= G5_RATIO}
    S["gates"] = gates

    md += ["", "## Gates", "",
           "| gate | " + " | ".join(LABEL[m] for m in MODELS) + " |",
           "|---|" + "---|" * len(MODELS)]

    def pf(b: bool) -> str:
        return "**PASS**" if b else "**FAIL**"

    def g5(m: str) -> str:
        o, r = gates[m]["one_step"], gates[m]["G5"]["ratio"]
        return (f"{o['approach_heldout']:.3f} ({r['approach_heldout']:.2f}) / "
                f"{o['contact_heldout']:.3f} ({r['contact_heldout']:.2f}) "
                f"{pf(gates[m]['G5']['pass'])}")

    md.append("| G1 test one-step <= 0.635 | " + " | ".join(
        f"{gates[m]['G1']['value']:.3f} {pf(gates[m]['G1']['pass'])}" for m in MODELS) + " |")
    md.append("| G2 gain >= 0.90 (v2-big all + deform held-out ex hold_only): min, fails/n | "
              + " | ".join(f"{gates[m]['G2']['min']:.2f}, {gates[m]['G2']['n_fail']}/"
                           f"{gates[m]['G2']['n']} {pf(gates[m]['G2']['pass'])}"
                           for m in MODELS) + " |")
    md.append("| G3 model < no-motion >= 50 %: deform / fast both_up_down / contact frames | "
              + " | ".join(f"{pct(gates[m]['G3']['deform_heldout'])} / "
                           f"{pct(gates[m]['G3']['fast_both_up_down'])} / "
                           f"{pct(gates[m]['G3']['contact_frames'])} {pf(gates[m]['G3']['pass'])}"
                           for m in MODELS) + " |")
    md.append("| (G3 info) v1 / v2-big | " + " | ".join(
        f"{pct(gates[m]['G3']['v1'])} / {pct(gates[m]['G3']['v2-big'])}" for m in MODELS) + " |")
    md.append("| G4 recon <= 0.6: deform motion / contact frames | " + " | ".join(
        f"{gates[m]['G4']['deform_heldout_motion']:.3f} / {gates[m]['G4']['contact_frames']:.3f}"
        f" {pf(gates[m]['G4']['pass'])}" for m in MODELS) + " |")
    md.append("| G5 one-step <= 0.8 x v2: approach / contact (ratio) | " + " | ".join(
        g5(m) for m in MODELS) + " |")
    for m in MODELS:
        if gates[m]["G2"]["fails"]:
            md.append(f"\nG2 fails {LABEL[m]}: " + ", ".join(
                f"{k} {v:.2f}" for k, v in gates[m]["G2"]["fails"]))

    # pick
    cand = [m for m in ("v3_mix", "v3_ft") if gates[m]["G1"]["pass"]]
    score = {}
    for m in ("v2", "v3_mix", "v3_ft"):
        dm = pool([e for r in DEFORM for e in data[m][r]])["model"]
        score[m] = {"deform": dm, "approach": matrix["approach_heldout"][m]["motion"]["model"],
                    "contact": matrix["contact_heldout"][m]["motion"]["model"]}
        score[m]["mean"] = float(np.mean(list(score[m].values())))
    if cand:
        pick = min(cand, key=lambda m: (round(score[m]["mean"], 6), m != "v3_mix"))
        negative = False
    else:
        pick = min(("v3_mix", "v3_ft"), key=lambda m: gates[m]["G1"]["value"])
        negative = True
    dep = Path(res[pick]["v2-big"]["deploy"]).parent
    S["pick"] = {"model": pick, "negative": negative, "score": score, "deploy_dir": str(dep),
                 "deploy": str(dep / "deploy.npz"), "decoder": str(dep / "decoder.npz")}
    md += ["", "## PICK", "",
           "| model | deform held-out motion one-step | approach | contact | mean |",
           "|---|---|---|---|---|"]
    for m in MODELS:
        s = score[m]
        md.append(f"| {LABEL[m]} | {s['deform']:.3f} | {s['approach']:.3f} | "
                  f"{s['contact']:.3f} | {s['mean']:.3f} |")
    md += ["", f"**PICK = {LABEL[pick]}**" + (" (NEGATIVE: no model passes G1)" if negative
                                             else f" (G1 passers: {', '.join(cand)})"),
           f"- deploy `{dep / 'deploy.npz'}`", f"- decoder `{dep / 'decoder.npz'}`"]

    # per-primitive PICK vs v2
    md += ["", (f"## Per episode: {LABEL[pick]} vs v2 (gain pred / model < no-motion / motion "
                "one-step)"), "", "| run | episode | v2 | " + LABEL[pick] + " |",
           "|---|---|---|---|"]
    for r in RUNS:
        by = {e["name"]: e for e in data[pick][r]}
        for e2 in data["v2"][r]:
            e = by[e2["name"]]

            def cell(x):
                en = x["entry"]
                return (f"{en['displacement_gain']['pred']:.2f} / "
                        f"{pct(en['model_below_nomotion_frac']['motion'])} / "
                        f"{en['motion']['model']['mean']:.3f}")
            md.append(f"| {r} | {e2['name']} | {cell(e2)} | {cell(e)} |")

    # input response
    er = load_json(args.eval_results)["results"]
    md += ["", (f"## Per-input 1 sigma latent response (whitened, lcs_learning "
                f"eval_results.json): v2 vs {LABEL[pick]}"), "",
           ("| input | u std (fs held-out) | fs held-out v2 / pick | approach v2 / pick | "
            "contact v2 / pick |"), "|---|---|---|---|---|"]
    S["input_response"] = {}
    for j, nm in enumerate(U_NAMES):
        cells = []
        for st in ("test_free_space", "test_approach", "test_contact"):
            a = er["v2"][st]["input_response"]["whitened_per_sigma"][j]
            b = er[pick][st]["input_response"]["whitened_per_sigma"][j]
            S["input_response"].setdefault(nm, {})[st] = {"v2": a, pick: b}
            cells.append(f"{a:.3f} / {b:.3f}")
        sd = er["v2"]["test_free_space"]["input_response"]["u_std"][j]
        md.append(f"| {nm} | {sd:.2e} | " + " | ".join(cells) + " |")

    (args.root / "summary.json").write_text(json.dumps(S, indent=1, default=float) + "\n")
    (args.root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
