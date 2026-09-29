#!/usr/bin/env python3
"""QC, deformation, hold and coverage report for motion-primitive run dirs.

Per run: a per-episode table, per-family stats, holds, causality; ``--write-qc`` writes
``<run>/qc.json``. Several runs: also a pooled coverage table (``--json`` saves it).

Non-rigid deformation = RMSE of the belt points after the best rigid (Kabsch) fit of the
motion-start belt. Achieved offsets are measured EE poses vs the motion start, in the action
layout (``lcs_dataset.action_vector``).

Run:
    uv run --frozen python scripts/lcs/deform_report.py RUN [RUN ...] [--write-qc] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from task_common import lcs_dataset as lcs

DIMS = lcs.ACTION_DIM_NAMES
ROT = np.array([j >= 6 for j in range(12)])
STRETCH_LOBE_MM = 8.0
DEFORM_MM = 5.0
HOLD_DRIFT_MM = 1.0


def kabsch_rmse(P, Q):
    """RMSE of Q to P after the best rigid fit of P onto Q."""
    p, q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((p @ R - q) ** 2).sum(1).mean()))


def mmr(x):
    x = np.asarray(x, np.float64)
    return {"min": float(x.min()), "med": float(np.median(x)), "max": float(x.max())} \
        if x.size else None


def zero_runs(z):
    runs, cur = [], []
    for k, v in enumerate(z):
        if v:
            cur.append(k)
        elif cur:
            runs.append(cur)
            cur = []
    return runs + ([cur] if cur else [])


def drift(pb, k_a, k_b):
    """Mean and max-point belt drift [mm] over frames ``k_a..k_b`` vs frame ``k_a``."""
    dev = np.sqrt(((pb[k_a:k_b + 1] - pb[k_a]) ** 2).sum(-1))
    return float(dev.mean(-1).max()) * 1e3, float(dev.max()) * 1e3


def episode(run: Path, row: dict) -> dict:
    d = np.load(run / row["file"], allow_pickle=True)
    pb = d["pcd_belt"].astype(np.float64) if d["pcd_belt"].size else None
    if pb is None:
        pb = np.stack([lcs.belt_points_ordered(b) for b in d["sim_belt_xyz"]])
    ph, u = d["sim_phase"], d["actions"].astype(np.float64)
    r, st = d["sim_realised_delta"].astype(np.float64), d["state"].astype(np.float64)
    mv = ph == 0
    k0 = int(np.argmax(mv)) if mv.any() else 0
    dt = float(json.loads(str(d["sim_meta"]))["sample_period_s"])
    nonrig = max(kabsch_rmse(pb[k0], pb[k]) for k in range(k0, len(pb))) * 1e3
    off = np.stack([lcs.action_vector(st[k0, 26:33], st[k, 26:33], st[k0, 33:40], st[k, 33:40])
                    for k in range(k0, len(st))])
    sep = np.linalg.norm(st[k0:, 33:36] - st[k0:, 26:29], axis=1)
    rod = d["sim_rod_stretch_pct"].astype(np.float64)
    out = {"file": row["file"], "primitive": row["primitive"],
           "variant": row.get("start_variant"), "amp_mul": row["amp_mul"],
           "retry_mul": row.get("retry_mul", 1.0), "stretch_lobe": row.get("stretch_lobe", 1.0),
           "motion_s": row["plan"]["motion_s"], "frames": len(u), "k0": k0,
           "nonrigid_mm": nonrig, "peak_offset": np.abs(off).max(0).tolist(),
           "sep_gain_mm": float(sep.max() - sep[0]) * 1e3,
           "sep_loss_mm": float(sep[0] - sep.min()) * 1e3,
           "stretch_gain_pct": float(np.nanmax(rod - rod[k0])), "u": u, "r": r, "mv": mv}
    trans = [c["amp_effective"] for c in row["plan"]["components"] if c["kind"] == "trans"]
    rot = [c["amp_effective"] for c in row["plan"]["components"] if c["kind"] == "rot"]
    out["eff_mm"] = max(trans, default=0.0)
    out["eff_deg"] = max(rot, default=0.0)
    holds = []
    if row["primitive"] == "hold_only":
        holds.append({"frames": len(u), "u_zero": bool(np.all(u == 0.0)),
                      **dict(zip(("drift_mm", "drift_max_pt_mm"), drift(pb, 0, len(pb) - 1),
                                 strict=True))})
    for h in row["plan"].get("holds") or []:
        ks = [k for k in range(k0, len(u))
              if (k - k0) * dt >= h["t0_s"] - 1e-9 and (k - k0 + 1) * dt <= h["t1_s"] + 1e-9]
        if not ks:
            continue
        holds.append({"frames": len(ks), "u_zero": bool(np.all(u[ks] == 0.0)),
                      **dict(zip(("drift_mm", "drift_max_pt_mm"),
                                 drift(pb, ks[0], min(ks[-1] + 1, len(pb) - 1)), strict=True))})
    out["holds"] = holds
    return out


def run_report(run: Path, write_qc: bool) -> tuple[dict, list[dict]]:
    index = json.loads((run / "index.json").read_text())
    rows = index["episodes"]
    ok = [r for r in rows if r["status"] == "ok"]
    slots = defaultdict(list)
    for r in rows:
        slots[(r["primitive"], r.get("rep"), r.get("start_variant"))].append(r)
    failed_eps = [{"primitive": k[0], "rep": k[1], "variant": k[2],
                   "reasons": [a.get("reason") for a in v]}
                  for k, v in slots.items() if not any(a["status"] == "ok" for a in v)]
    failed_attempts = [{"primitive": r["primitive"], "rep": r.get("rep"),
                        "variant": r.get("start_variant"), "reason": r.get("reason")}
                       for r in rows if r["status"] != "ok"]
    eps = [episode(run, r) for r in ok]
    print(f"== {run}  ok {len(ok)}  failed episodes {len(failed_eps)}  failed attempts "
          f"{len(failed_attempts)}")
    print(f"{'file':<17}{'primitive':<18}{'var':<6}{'amp':>5}{'rmul':>5}{'T':>6}{'nonrig':>7}"
          f"{'sep+':>6}{'str+':>6}{'mm':>6}{'deg':>6}")
    for e in eps:
        print(f"{e['file']:<17}{e['primitive']:<18}{e['variant'] or '-'!s:<6}"
              f"{e['amp_mul']:>5.2f}{e['retry_mul']:>5.2f}{e['motion_s']:>6.2f}"
              f"{e['nonrigid_mm']:>7.2f}{e['sep_gain_mm']:>6.1f}{e['stretch_gain_pct']:>6.2f}"
              f"{e['eff_mm']:>6.1f}{e['eff_deg']:>6.1f}")
    st = [r["stats"] for r in ok]
    fam = defaultdict(list)
    for e in eps:
        fam[e["primitive"]].append(e)
    per_family = {}
    for name, es in fam.items():
        um = np.vstack([e["u"][e["mv"]] for e in es])
        per_family[name] = {
            "episodes": len(es), "eff_amp_mm": mmr([e["eff_mm"] for e in es]),
            "eff_amp_deg": mmr([e["eff_deg"] for e in es]),
            "amp_mul": mmr([e["amp_mul"] for e in es]),
            "retry_mul": mmr([e["retry_mul"] for e in es]),
            "motion_s": mmr([e["motion_s"] for e in es]),
            "nonrigid_mm": mmr([e["nonrigid_mm"] for e in es]),
            "sep_gain_mm": mmr([e["sep_gain_mm"] for e in es]),
            "u_std_motion": dict(zip(DIMS, um.std(0).tolist(), strict=True)),
        }
    holds = [dict(h, file=e["file"], primitive=e["primitive"]) for e in eps for h in e["holds"]]
    U = np.vstack([e["u"] for e in eps])
    R = np.vstack([e["r"] for e in eps])
    caus_all = lcs.causality_slopes(U, R)
    caus_mv = lcs.causality_slopes(np.vstack([e["u"][e["mv"]] for e in eps]),
                                   np.vstack([e["r"][e["mv"]] for e in eps]))
    qc = {
        "run": str(run), "episodes_ok": len(ok), "failed_episodes": failed_eps,
        "failed_attempts": failed_attempts, "wall_s": index.get("summary", {}).get("wall_s"),
        "osc_log_errors": index.get("osc_log_errors"),
        "frames": int(sum(s["frames"] for s in st)),
        "tuples": int(sum(s["frames"] - 1 for s in st)),
        "per_family": per_family,
        "bound_ratio_max_per_dim": dict(zip(DIMS, np.max(
            [s["u_bound_ratio_max_per_dim"] for s in st], 0).tolist(), strict=True)),
        "crop_margin_min_mm": {k: float(min(s["crop_margin_min_mm"][k] for s in st))
                               for k in st[0]["crop_margin_min_mm"]},
        "stretch_gain_pct_max": float(max(s["rod_stretch_gain_pct_max"] for s in st)),
        "min_board_clearance_mm": float(min(s["min_board_clearance_mm"] for s in st)),
        "min_franka_tip_clearance_mm": float(min(s["min_franka_tip_clearance_mm"] for s in st)),
        "board_contact_any": any(s["board_contact"] for s in st),
        "grasp_ok_all": all(s["grasp_ok_all"] for s in st),
        "ur_guard_scale_min": float(min(s["ur_guard_scale_min"] for s in st)),
        "tracking_rms_mm_max": np.max([s["tracking_rms_mm"] for s in st], 0).tolist(),
        "u_plan_max_err": float(max(s["u_plan_max_err"] for s in st)),
        "holds": holds,
        "holds_u_zero_all": all(h["u_zero"] for h in holds),
        "holds_drift_mm_max": max((h["drift_mm"] for h in holds), default=None),
        "causality_all_rows": caus_all, "causality_motion_rows": caus_mv,
        "episodes": [{k: v for k, v in e.items() if k not in ("u", "r", "mv")} for e in eps],
    }
    print("families:")
    for name, f in per_family.items():
        s = f["u_std_motion"]
        print(f"  {name:<18}n {f['episodes']}  nonrig mm {fmt(f['nonrigid_mm'])}  eff mm "
              f"{fmt(f['eff_amp_mm'])} deg {fmt(f['eff_amp_deg'])}  T {fmt(f['motion_s'])}  "
              f"UR u std x/y {s['u_x'] * 1e3:.2f}/{s['u_y'] * 1e3:.2f} mm rz "
              f"{s['u_rz'] * 1e3:.1f} mrad")
    print(f"holds: {len(holds)}, u == 0 all {qc['holds_u_zero_all']}, drift max "
          f"{qc['holds_drift_mm_max']}")
    print("causality all rows:\n" + lcs.causality_table(caus_all))
    print("causality motion rows:\n" + lcs.causality_table(caus_mv))
    if write_qc:
        (run / "qc.json").write_text(json.dumps(qc, indent=1, default=float) + "\n")
    return qc, eps


def fmt(m):
    return "-" if m is None else f"{m['min']:.1f}/{m['med']:.1f}/{m['max']:.1f}"


def coverage(eps: list[dict]) -> dict:
    peak = np.array([e["peak_offset"] for e in eps])
    um = np.abs(np.vstack([e["u"][e["mv"]] for e in eps]))
    scale = np.where(ROT, 180.0 / np.pi, 1e3)  # peak: mm / deg
    uscale = 1e3  # |u|: mm / mrad
    cov = {d: {"peak_p50": float(np.percentile(peak[:, j], 50) * scale[j]),
               "peak_p95": float(np.percentile(peak[:, j], 95) * scale[j]),
               "u_abs_p95": float(np.percentile(um[:, j], 95) * uscale)}
           for j, d in enumerate(DIMS)}
    n_lobe = sum(e["sep_gain_mm"] >= STRETCH_LOBE_MM for e in eps)
    n_def = sum(e["nonrigid_mm"] >= DEFORM_MM for e in eps)
    U = np.vstack([e["u"] for e in eps])
    ustd = dict(zip(DIMS, U.std(0).tolist(), strict=True))
    return {"episodes": len(eps), "per_dim": cov, "stretch_lobe_episodes": int(n_lobe),
            "deform_ge_5mm_episodes": int(n_def), "u_std_all_rows": ustd}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--write-qc", action="store_true", help="write <run>/qc.json")
    ap.add_argument("--json", type=Path, default=None, help="save the pooled coverage")
    args = ap.parse_args()
    all_eps = []
    for run in args.runs:
        all_eps += run_report(run, args.write_qc)[1]
    cov = coverage(all_eps)
    print(f"\ncoverage over {cov['episodes']} episodes (peak mm|deg, |u| mm|mrad):")
    print(f"{'dim':<6}{'peak p50':>9}{'peak p95':>9}{'|u| p95':>9}")
    for d, c in cov["per_dim"].items():
        print(f"{d:<6}{c['peak_p50']:>9.1f}{c['peak_p95']:>9.1f}{c['u_abs_p95']:>9.2f}")
    print(f"stretch lobes (grasp sep +>= {STRETCH_LOBE_MM:g} mm): {cov['stretch_lobe_episodes']}"
          f"; non-rigid >= {DEFORM_MM:g} mm: {cov['deform_ge_5mm_episodes']}")
    s = cov["u_std_all_rows"]
    print(f"UR u std all rows x/y {s['u_x'] * 1e3:.2f}/{s['u_y'] * 1e3:.2f} mm, rz "
          f"{s['u_rz'] * 1e3:.1f} mrad")
    if args.json:
        args.json.write_text(json.dumps(cov, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
