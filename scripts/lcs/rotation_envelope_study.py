"""Rotation-envelope study (PKG-20261005-v11-rotation-envelope-study): EE rotation reached in the
episodes we have (offline), the closed-loop env 20 vs 40 deg tables and the decision rule.

    uv run --frozen python scripts/lcs/rotation_envelope_study.py \
        --out data/lcs/realfirst/v1_1/rotation
    uv run --frozen python scripts/lcs/rotation_envelope_study.py \
        --closed-loop data/lcs/mpc_eval/20261005_v1_1_envelope \
        --out data/lcs/realfirst/v1_1/rotation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from realfirst_loop import quat_angle_deg

EVAL = "data/lcs/mpc_eval"
# (source, root, controller label when the index has no controller block)
SOURCES = [
    ("seed_demo", "data/lcs/realfirst/v1/seed", "osc demo"),
    ("seed_excite", "data/lcs/realfirst/v1/seed", "osc excitation"),
    ("v2_c3_seat6", f"{EVAL}/20260926-231837-synthtargets", None),
    ("v2_mppi2s", f"{EVAL}/20261005_V2_seat6_mppi/mppi2s", None),
    ("v2_magna2s", f"{EVAL}/20261005_V2_seat6_mppi/magna2s", None),
    ("v10a_seat6", f"{EVAL}/20261001-v10/20261001_V10A_seat6_Qw_ub-ins", None),
    ("v10a_r2_seat6", f"{EVAL}/20261001-v10/20261001_V10A_R2_seat6_Qw_ub-ins", None),
    ("e2_iter0", "data/lcs/realfirst/v1/iter0/collect", None),
    ("e2_iter1", "data/lcs/realfirst/v1/iter1/collect", None),
    ("e2_iter2", "data/lcs/realfirst/v1/iter2/collect", None),
]
ARMS = ("F", "UR")
LIMITS = (20, 30, 40)
E2_FRAME_DEG = 19.0
CELLS = [("fe_nominal", "pre_place_1_osc"), ("fe_set1", "gv_01"), ("fe_set1", "gv_05"),
         ("urtip", "ud+3_r0"), ("urtip6", "ud+6_r0"), ("urtip6_high", "ud+6_r0")]
HAND = "panda_hand/panda_hand"
HAND_Z_MIN_MM, ROT_MAX_DEG = 11.2, 90.0  # d2 table.py (onpolicy_loop.safety())
D2_MPPI2S = f"{EVAL}/20261005_V2_seat6_mppi/mppi2s"


def strict_of(row: dict) -> bool:
    return bool(row.get("final_wrap_deg", 0) >= 90 and abs(row.get("final_h_median_mm", 99)) <= 1)


def pp2_frame(d) -> tuple[int, str]:
    """Reference frame: harness stage switch, collector end of move:pre_place_2, else frame 0."""
    if "sim_stage" in d.files:
        s = np.flatnonzero(np.asarray(d["sim_stage"]) >= 1)
        if len(s):
            return int(s[0]), "stage>=1"
    labels = json.loads(str(d["sim_meta"])).get("phase_labels") or []
    if "move:pre_place_2" in labels:
        m = np.flatnonzero(np.asarray(d["sim_phase"]) == labels.index("move:pre_place_2"))
        if len(m) and m[-1] + 1 < len(d["sim_phase"]):
            return int(m[-1] + 1), "phase"
    return 0, "frame0"


def rotation_row(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        ee = [np.asarray(d["sim_ee_franka"], float), np.asarray(d["sim_ee_ur"], float)]
        t = np.asarray(d["sim_time"], float)
        k, ref = pp2_frame(d)
    t = t - t[0]
    out = {"n_frames": len(t), "pp2_frame": k, "pp2_ref": ref, "pp2_t": float(t[k])}
    for arm, e in zip(ARMS, ee):
        a0 = quat_angle_deg(e[0, 3:7], e[:, 3:7])
        a2 = quat_angle_deg(e[k, 3:7], e[k:, 3:7])
        out[arm] = {"rot_start_max": float(a0.max()), "t_start_max": float(t[int(a0.argmax())]),
                    "rot_start_final": float(a0[-1]), "rot_start_at_pp2": float(a0[k]),
                    "rot_pp2_max": float(a2.max()), "t_pp2_max": float(t[k + int(a2.argmax())]),
                    "frac_ge19": float((a0 >= E2_FRAME_DEG).mean())}
    return out


def index_files(root: Path) -> list[Path]:
    if (root / "index.json").is_file():
        return [root / "index.json"]
    return sorted(root.glob("*/index.json"))


def controller_of(idx: dict, default: str | None) -> str:
    c = idx.get("controller") or {}
    if c.get("mode") == "mppi":
        return "mppi"
    if c.get("mode") == "learned":
        return "C3"
    return default or str(idx.get("backend"))


def source_rows(name: str, root: str, default_ctl: str | None) -> list[dict]:
    rows = []
    for ip in index_files(REPO / root):
        idx = json.loads(ip.read_text())
        for ep in idx["episodes"]:
            rel = ep.get("path") or ep.get("file")
            if not rel:
                continue
            if name == "seed_demo" and ep.get("kind") != "demo":
                continue
            if name == "seed_excite" and ep.get("kind") != "excitation":
                continue
            path = ip.parent / rel
            with np.load(path, allow_pickle=True) as d:
                if "sim_ee_franka" not in d.files:
                    continue
                gt = str(d["sim_outcome"])
            r = {"source": name, "dir": str(ip.parent.relative_to(REPO)),
                 "file": str(path.relative_to(REPO / root)),
                 "start": (ep.get("start_state") or {}).get("id") or ep.get("start_variant"),
                 "controller": controller_of(idx, default_ctl), "gt": ep.get("outcome", gt),
                 "strict": strict_of(ep), "status": ep.get("status")}
            r.update(rotation_row(path))
            rows.append(r)
    return rows


def dist(vals: list[float]) -> dict:
    v = np.asarray(vals, float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max()),
            **{f"le{x}": float((v <= x).mean()) for x in LIMITS}}


def pooled(rows: list[dict]) -> dict:
    """Per arm and reference; pp2 excludes frame-0 fallback rows."""
    out = {}
    for arm in ARMS:
        out[arm] = {"start": dist([r[arm]["rot_start_max"] for r in rows]),
                    "pp2": dist([r[arm]["rot_pp2_max"] for r in rows if r["pp2_ref"] != "frame0"]),
                    "start_final": dist([r[arm]["rot_start_final"] for r in rows])}
    return out


def e2_frames(rows: list[dict]) -> dict:
    out = {}
    for it in ("e2_iter0", "e2_iter1", "e2_iter2"):
        rr = [r for r in rows if r["source"] == it]
        if not rr:
            continue
        out[it] = {"n": len(rr)}
        for arm in ARMS:
            f = [r[arm]["frac_ge19"] for r in rr]
            w = np.asarray([r["n_frames"] for r in rr], float)
            out[it][arm] = {"min": float(min(f)), "max": float(max(f)),
                            "pooled": float(np.dot(f, w) / w.sum())}
    return out


def offline(out: Path) -> dict:
    rows = [r for s in SOURCES for r in source_rows(*s)]
    succ = [r for r in rows if r["gt"] == "engaged"]
    res = {"angle": "quat_angle_deg (2 acos |<q0, q>|), deg", "sources": [s[0] for s in SOURCES],
           "n": len(rows), "successes": pooled(succ), "successes_n": len(succ),
           "strict": pooled([r for r in rows if r["strict"]]),
           "strict_n": sum(r["strict"] for r in rows),
           "failures": pooled([r for r in rows if r["gt"] != "engaged"]),
           "failures_n": len(rows) - len(succ),
           "successes_closed_loop": pooled([r for r in succ if r["controller"] not in
                                             ("osc demo", "osc excitation")]),
           "e2_frames_ge19": e2_frames(rows),
           "frame0_flagged": [f"{r['dir']}/{r['file']}" for r in rows if r["pp2_ref"] == "frame0"],
           "rows": rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "rotation.json").write_text(json.dumps(res, indent=1) + "\n")
    (out / "rotation.md").write_text(offline_md(res))
    return res


def f0(x) -> str:
    return f"{x:.0f}"


def pool_table(title: str, p: dict, n: int) -> list[str]:
    md = [f"### {title} (n {n})", "",
          "| arm | reference | n | p50 ° | p90 ° | max ° | <= 20 | <= 30 | <= 40 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for ref, lab in (("start", "start (max)"), ("pp2", "pre_place_2 (max)"),
                         ("start_final", "start (final)")):
            s = p[arm][ref]
            if not s["n"]:
                continue
            md.append(f"| {arm} | {lab} | {s['n']} | {s['p50']:.1f} | {s['p90']:.1f} | "
                      f"{s['max']:.1f} | " + " | ".join(f"{s[f'le{x}']:.0%}" for x in LIMITS)
                      + " |")
    return md + [""]


def offline_md(res: dict) -> str:
    md = ["# Rotation reached per episode (offline)", "",
          ("Angle = `realfirst_loop.quat_angle_deg` (geodesic, 2 acos |<q0, q>|) on `sim_ee_franka`"
           " / `sim_ee_ur` (wxyz). `start` = max over the episode of the angle to frame 0; `pp2` ="
           " max over frames from the pre_place_2 frame (harness: first `sim_stage >= 1`; "
           "collector: first frame after `move:pre_place_2`) of the angle to that frame; `final` "
           "= angle at the last frame. Success = GT `engaged` (index row `outcome`); strict = "
           "final wrap >= 90 deg and |h| <= 1 mm. Times in s from episode start."), "",
          (f"Episodes: {res['n']} ({res['successes_n']} engaged, {res['strict_n']} strict, "
           f"{res['failures_n']} not engaged). pp2 fallback to frame 0: "
           f"{len(res['frame0_flagged'])} file(s) {res['frame0_flagged'] or ''}"), "",
          "## Pooled", ""]
    md += pool_table("Successes (GT engaged)", res["successes"], res["successes_n"])
    md += pool_table("Strict successes", res["strict"], res["strict_n"])
    md += pool_table("Successes under closed-loop control only (C3 / MPPI, no OSC seed)",
                     res["successes_closed_loop"],
                     res["successes_closed_loop"]["F"]["start"]["n"])
    md += pool_table("Failures (context)", res["failures"], res["failures_n"])
    md += ["## e2 MPC rollouts: fraction of frames >= 19 deg from the start (all frames)", "",
           "| iteration | n | F min-max (pooled) | UR min-max (pooled) |", "|---|---|---|---|"]
    for it, v in res["e2_frames_ge19"].items():
        md.append(f"| {it} | {v['n']} | " + " | ".join(
            f"{v[a]['min']:.0%}-{v[a]['max']:.0%} ({v[a]['pooled']:.0%})" for a in ARMS) + " |")
    md += ["", "## Per source", "",
           ("| source | controller | n | engaged | strict | F start max p50 / max | UR start max "
            "p50 / max | F pp2 max p50 / max | UR pp2 max p50 / max |"), "|---" * 9 + "|"]
    for s in res["sources"]:
        rr = [r for r in res["rows"] if r["source"] == s]
        if not rr:
            md.append(f"| {s} | - | 0 | - | - | - | - | - | - |")
            continue
        cells = []
        for key in ("rot_start_max", "rot_pp2_max"):
            for arm in ARMS:
                v = [r[arm][key] for r in rr]
                cells.append(f"{np.median(v):.0f} / {max(v):.0f}")
        ctl = ", ".join(sorted({r["controller"] for r in rr}))
        md.append(f"| {s} | {ctl} | {len(rr)} | {sum(r['gt'] == 'engaged' for r in rr)} | "
                  f"{sum(r['strict'] for r in rr)} | " + " | ".join(cells) + " |")
    md += ["", "## Per episode", "",
           ("| source | file | start | ctl | GT | strict | pp2 frame (ref, t s) | F start max (t) |"
            " F final | F at pp2 | F pp2 max (t) | UR start max (t) | UR final | UR at pp2 | "
            "UR pp2 max (t) |"), "|---" * 15 + "|"]
    for r in res["rows"]:
        f, u = r["F"], r["UR"]
        md.append(
            f"| {r['source']} | {r['file']} | {r['start']} | {r['controller']} | {r['gt']} | "
            f"{'y' if r['strict'] else 'n'} | {r['pp2_frame']} ({r['pp2_ref']}, "
            f"{r['pp2_t']:.1f}) | " + " | ".join(
                f"{a['rot_start_max']:.1f} ({a['t_start_max']:.1f}) | {a['rot_start_final']:.1f} "
                f"| {a['rot_start_at_pp2']:.1f} | {a['rot_pp2_max']:.1f} ({a['t_pp2_max']:.1f})"
                for a in (f, u)) + " |")
    return "\n".join(md) + "\n"


def solves(d_cell: Path, ep: dict) -> tuple[np.ndarray, list[float]]:
    p = d_cell / ((ep.get("mppi") or {}).get("solve_log") or "")
    if not p.is_file():
        return np.zeros(0), []
    js = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    ms = [s["solve_ms"] for s in js if s.get("solve_ms") is not None]
    return np.asarray(ms, float), [s["frac_env_penalised"] for s in js
                                   if s.get("frac_env_penalised") is not None]


def closed_row(ctl: str, d_cell: Path, cell: str, ep: dict) -> dict:
    from task_common.recording import Recording

    path = d_cell / ep["file"]
    rot = rotation_row(path)
    with np.load(path, allow_pickle=True) as d:
        held = bool(np.asarray(d["sim_grasp_ok"]).all())
    rec = Recording.load(d_cell / ep["recording"])
    hz = 1e3 * rec.body_q[:, rec.meta["body_labels"].index(HAND), 2].astype(float)
    ms, envp = solves(d_cell, ep)
    r = {"ctl": ctl, "cell": cell, "start": ep["start_state"]["id"], "repeat": ep["repeat"],
         "outcome": ep["outcome"], "status": ep["status"], "reason": ep.get("reason"),
         "strict": strict_of(ep), "wrap": ep["final_wrap_deg"], "h": ep["final_h_median_mm"],
         "rmse": ep["alignment"]["final"]["belt_rmse_mm"],
         "F_rot": rot["F"]["rot_start_max"], "U_rot": rot["UR"]["rot_start_max"],
         "F_pp2": rot["F"]["rot_pp2_max"], "U_pp2": rot["UR"]["rot_pp2_max"],
         "pp2_ref": rot["pp2_ref"], "hand_zmin": float(hz.min()), "hand_z0": float(hz[0]),
         "tip_min": ep["min_franka_tip_clearance_mm"], "board_min": ep["min_board_clearance_mm"],
         "held": held, "duration": ep["duration_s"], "stage_ends": ep.get("stage_ends"),
         "n_solve": int(ms.size), "solve_ms": ms.tolist(),
         "p50": float(np.percentile(ms, 50)) if ms.size else np.nan,
         "p95": float(np.percentile(ms, 95)) if ms.size else np.nan,
         "max": float(ms.max()) if ms.size else np.nan,
         "env_pen_mean": float(np.mean(envp)) if envp else np.nan,
         "fallbacks": (ep.get("mppi") or {}).get("fallbacks"),
         "viol": (ep.get("mppi") or {}).get("plans_violating"),
         "tracking": ep["tracking_rms_mm"]}
    unsafe = []
    if ep["status"] == "aborted":
        unsafe.append(f"abort: {(ep.get('reason') or '')[:60]}")
    if not held:
        unsafe.append("grasp lost")
    if r["hand_zmin"] < HAND_Z_MIN_MM:
        unsafe.append(f"panda_hand z {r['hand_zmin']:.1f} mm")
    if r["F_rot"] > ROT_MAX_DEG:
        unsafe.append(f"Franka rot {r['F_rot']:.0f} deg")
    r["unsafe"] = unsafe
    return r


def closed_rows(ctl: str, root: Path) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for cell, start in CELLS:
        idx = root / cell / "index.json"
        eps = [] if not idx.is_file() else [
            e for e in json.loads(idx.read_text())["episodes"] if e["start_state"]["id"] == start]
        got = [e for e in eps if e.get("file")]
        if not got:
            missing.append(f"{cell}/{start}")
        rows += [closed_row(ctl, root / cell, cell, e) for e in got]
    return rows, missing


def closed_summary(rows: list[dict], missing: list[str]) -> dict:
    ms = np.concatenate([np.asarray(r["solve_ms"]) for r in rows]) if rows else np.zeros(0)
    return {"n": len(rows), "missing": missing, "strict": sum(r["strict"] for r in rows),
            "engaged": sum(r["outcome"] == "engaged" for r in rows),
            "unsafe": sum(bool(r["unsafe"]) for r in rows),
            "rmse": float(np.mean([r["rmse"] for r in rows])) if rows else np.nan,
            "F_rot": max((r["F_rot"] for r in rows), default=np.nan),
            "U_rot": max((r["U_rot"] for r in rows), default=np.nan),
            "F_pp2": max((r["F_pp2"] for r in rows), default=np.nan),
            "U_pp2": max((r["U_pp2"] for r in rows), default=np.nan),
            "hand_zmin": min((r["hand_zmin"] for r in rows), default=np.nan),
            "env_pen_mean": float(np.nanmean([r["env_pen_mean"] for r in rows]))
            if rows else np.nan,
            "p95_pooled": float(np.percentile(ms, 95)) if ms.size else np.nan,
            "p95_ep_max": max((r["p95"] for r in rows), default=np.nan)}


HDR = ("| run | cell | start | outcome | strict | final wrap ° | final h mm | final belt RMSE mm | "
       "max rot from start F / UR ° | max rot from pre_place_2 F / UR ° | min panda_hand z mm "
       "(vs start) | min tip / board clearance mm | held | solve ms p50 / p95 / max (n, fb, "
       "viol) | env-penalised samples (mean) | tracking rms F / UR mm | unsafe |\n"
       + "|---" * 17 + "|")
SUM_HDR = ("| run | n | missing | strict | engaged | unsafe | mean belt RMSE mm | max rot F / UR "
           "° | max rot pp2 F / UR ° | min panda_hand z mm | env-penalised mean | solve p95 "
           "pooled / worst episode ms |\n" + "|---" * 12 + "|")


def fmt_row(r: dict) -> str:
    return (f"| {r['ctl']} | {r['cell']} | {r['start']} | {r['outcome']} | "
            f"{'y' if r['strict'] else 'n'} | {r['wrap']:.1f} | {r['h']:+.2f} | {r['rmse']:.1f} | "
            f"{r['F_rot']:.0f} / {r['U_rot']:.0f} | {r['F_pp2']:.0f} / {r['U_pp2']:.0f} | "
            f"{r['hand_zmin']:.1f} ({r['hand_zmin'] - r['hand_z0']:+.1f}) | {r['tip_min']:.1f} / "
            f"{r['board_min']:.1f} | {'y' if r['held'] else 'n'} | {r['p50']:.1f} / "
            f"{r['p95']:.1f} / {r['max']:.1f} ({r['n_solve']}, {r['fallbacks']}, {r['viol']}) | "
            f"{r['env_pen_mean']:.0%} | {r['tracking'][0]:.2f} / {r['tracking'][1]:.2f} | "
            f"{'; '.join(r['unsafe']) or '-'} |")


def fmt_sum(k: str, v: dict) -> str:
    return (f"| {k} | {v['n']} | {len(v['missing'])} | {v['strict']}/{v['n']} | "
            f"{v['engaged']}/{v['n']} | {v['unsafe']} | {v['rmse']:.1f} | {v['F_rot']:.0f} / "
            f"{v['U_rot']:.0f} | {v['F_pp2']:.0f} / {v['U_pp2']:.0f} | {v['hand_zmin']:.1f} | "
            f"{v['env_pen_mean']:.0%} | {v['p95_pooled']:.1f} / {v['p95_ep_max']:.1f} |")


NOTE = ("V2 goal deploys `data/lcs/mpc_eval/20260926-231837-synthtargets/goals/<target>/`, MPPI "
        "two stages (`--mppi-stages all`, [4, 30] s, d2's `mppi2s_<target>.yaml` with "
        "`rot_envelope_deg` overridden), `--max-episode-s 36`, pace 1, 1 repeat, port 7773. "
        "Rotation = `quat_angle_deg`. Strict = final wrap >= 90 deg and |h| <= 1 mm. Unsafe = "
        "harness abort, grasp lost, panda_hand z < 11.2 mm or Franka rotation > 90 deg (d2). "
        "Solve ms from the node's `solves.jsonl`. `d2_mppi2s` = d2's original env-20 run "
        "(port 7762), context only.")


def closed_loop(root: Path, out: Path, off: dict) -> dict:
    runs = {"env20": root / "env20", "env40": root / "env40", "d2_mppi2s": REPO / D2_MPPI2S}
    data = {k: closed_rows(k, p) for k, p in runs.items()}
    s = {k: closed_summary(*v) for k, v in data.items()}
    for k in ("env20", "env40"):
        md = [f"# {k}: V2 + MPPI two-stage, seat6, envelope {k[3:]} deg", "", NOTE, "", HDR]
        md += [fmt_row(r) for r in data[k][0]] + ["", SUM_HDR, fmt_sum(k, s[k])]
        (runs[k] / "table.md").write_text("\n".join(md) + "\n")
    md = ["# Rotation envelope 20 vs 40 deg: V2 + MPPI two-stage, seat6", "", NOTE, "", HDR]
    md += [fmt_row(r) for k in runs for r in data[k][0]]
    md += ["", SUM_HDR] + [fmt_sum(k, v) for k, v in s.items()]
    e20, e40 = s["env20"], s["env40"]
    go40 = e40["engaged"] >= e20["engaged"] + 2 and e40["unsafe"] <= e20["unsafe"]
    deg = 40.0 if go40 else 20.0
    rule = ("keep 20 deg unless env40 engaged >= env20 engaged + 2 AND env40 unsafe <= env20 "
            "unsafe (fresh runs only)")
    line = (f"decision: rot_envelope_deg {deg:.0f} (env20 engaged {e20['engaged']}/{e20['n']} "
            f"strict {e20['strict']} unsafe {e20['unsafe']} p95 {e20['p95_pooled']:.1f} ms; env40 "
            f"engaged {e40['engaged']}/{e40['n']} strict {e40['strict']} unsafe {e40['unsafe']} "
            f"p95 {e40['p95_pooled']:.1f} ms)")
    md += ["", line]
    (root / "table.md").write_text("\n".join(md) + "\n")
    for k, (rows, _) in data.items():
        for r in rows:
            r.pop("solve_ms")
    (root / "table.json").write_text(json.dumps(
        {"rows": {k: v[0] for k, v in data.items()}, "summary": s, "decision": line},
        indent=1, default=float) + "\n")
    sp = off["successes"]
    offl = {a: {"successes_le20_start": sp[a]["start"]["le20"],
                "successes_le20_pp2": sp[a]["pp2"]["le20"],
                "p90_start": sp[a]["start"]["p90"], "p90_pp2": sp[a]["pp2"]["p90"],
                "n": sp[a]["start"]["n"]} for a in ARMS}
    dec = {"rot_envelope_deg": {"franka": deg, "ur": deg}, "rule": rule, "rule_fired": go40,
           "env20": e20, "env40": e40, "context_d2_mppi2s": s["d2_mppi2s"], "offline": offl,
           "decision": line}
    (out / "decision.json").write_text(json.dumps(dec, indent=1, default=float) + "\n")
    dmd = ["# Rotation envelope decision (P7)", "", f"Rule: {rule}.", "", line, "",
           SUM_HDR] + [fmt_sum(k, v) for k, v in s.items()]
    dmd += ["", "Secondary evidence (offline, GT-engaged episodes, all sources):", "",
            ("| arm | n | <= 20 deg from start | <= 20 deg from pre_place_2 | p90 start ° | "
             "p90 pre_place_2 ° |"), "|---|---|---|---|---|---|"]
    for a in ARMS:
        o = offl[a]
        dmd.append(f"| {a} | {o['n']} | {o['successes_le20_start']:.0%} | "
                   f"{o['successes_le20_pp2']:.0%} | {o['p90_start']:.1f} | {o['p90_pp2']:.1f} |")
    dmd += ["", f"P8: `rot_envelope_deg: {{franka: {deg:.1f}, ur: {deg:.1f}}}`."]
    (out / "decision.md").write_text("\n".join(dmd) + "\n")
    print("\n".join(md))
    return dec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--closed-loop", type=Path, default=None,
                    help="dir with env20/<cell>/ and env40/<cell>/ harness runs")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else REPO / a.out
    off = offline(out)
    sp = off["successes"]
    print(f"offline: {off['n']} episodes, {off['successes_n']} engaged; successes start max "
          + "; ".join(f"{x} p50 {sp[x]['start']['p50']:.1f} p90 {sp[x]['start']['p90']:.1f} "
                      f"max {sp[x]['start']['max']:.1f} <=20 {sp[x]['start']['le20']:.0%}"
                      for x in ARMS))
    if a.closed_loop is not None:
        root = a.closed_loop if a.closed_loop.is_absolute() else REPO / a.closed_loop
        closed_loop(root, out, off)


if __name__ == "__main__":
    main()
