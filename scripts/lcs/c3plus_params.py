#!/usr/bin/env python3
"""C3+ (magna's C++ learned MPC, worktree magna-deploy-learned-lcs) params for a latent deploy.

``write``: per variant an LCS yaml + a controller params yaml (+ a demo_traj yaml for ``track``)
under the worktree's ``systems/parameters/learned_archive/<date>/c3plus/<tag>/`` (the controller
resolves them relative to its CWD = the worktree), copies + ``manifest.json`` under ``--out``.
``check``: a python mirror of the C++ load (drake yaml strict keys from the headers, the
DRAKE_DEMANDs of LearnedLcsParams / LearnedMpcParams / LoadLcs). Nothing here launches a binary.

Variants (the C++ ``Stages()`` holds at most two fixed goals; ``demo_traj`` is its only
many-reference mode):
    wp2    latent_waypoints.npz: stage 0 = the latch waypoint (pre_place_2), tol = its waypoint
           tol, timeout = sum of max_s up to the latch; stage 1 = the seated waypoint, runs to
           its timeout (tol 1e-6; the C++ falls back to goal_tol_whitened for tol < 0)
    pp2    fallback: the deploy export's own goals (goal frames pre_place_2 latch, seated),
           tols [goal.json tol_1, 1e-6], the loop's stage_max_durations_s
    seat   the seated waypoint only (one stage; the loop's mppi_stages "last")
    track  one demo_traj stage, progress mode over the goal demo's per-frame latents from the
           demo offset frame (the MPPI's track mode), w_p 0
Costs as the last C3+ campaign (V8/V10, 2026-10-01 params): w_q 1 (Q = diag(1/z_std^2)),
w_r 0.3 (R = diag(1/((u_ub - u_lb)/2)^2)), admm_iter 2, input-change penalty off, EE z floors
(hard, = the MPPI floors), no EE-position cost (w_p 0). u bounds = the loop's box_i.

    uv run --frozen python scripts/lcs/c3plus_params.py write \\
        --deploy-json data/lcs/realfirst/cmp/v12_36/iter1/deploy.json --tag v16b \\
        --out data/lcs/realfirst/cmp/c3plus/v16b
    uv run --frozen python scripts/lcs/c3plus_params.py check \\
        --params systems/parameters/learned_archive/2026-10-07/c3plus/v16b/<params>.yaml
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WORKTREE = Path("/home/hienbui/git/magna-deploy-learned-lcs")
ARCHIVE = "systems/parameters/learned_archive/2026-10-07/c3plus"
# Last C3+ campaign's params (V10a, 2026-10-01): every top-level key and the cost knobs.
BASE_PARAMS = ("systems/parameters/learned_archive/2026-10-01/v10/"
               "round_belt_controller_params_learned_eval_v10a_flat_engaged.yaml")
HEADERS = ("systems/controllers/parameter_headers/learned_lcs_params.h",)
BASE_MPPI = "data/lcs/realfirst/v1_1/mppi.yaml"  # floors
VARIANTS = ("wp2", "pp2", "seat", "track")
NEVER_TOL = 1.0e-6  # stage runs to its timeout (V10's convention)
COSTS = {"w_q": 1.0, "w_r": 0.3, "admm_iter": 2}
DEMO_OFFSET = 20  # mppi_demo_offset_frames: the demo prehold is skipped
LCS_KEYS_REPLACED = ("z_goal", "u_lb", "u_ub", "goal_source", "z_goal_stage1", "z_goal_stage2")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def fnum(x: float) -> str:
    """Shortest round-trip repr; mantissa always has a '.' (YAML 1.1 floats, e.g. PyYAML)."""
    x = float(x)
    if math.isinf(x):
        return ".inf" if x > 0 else "-.inf"
    r = repr(x)
    m, e = (r.split("e") + [None])[:2]
    m = m if "." in m else m + ".0"
    return m if e is None else f"{m}e{e}"


def fvec(v) -> str:
    return "[" + ", ".join(fnum(x) for x in np.asarray(v).ravel()) + "]"


def top_spans(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Top-level key -> [start, end) line span (key line + its indented / '- ' lines)."""
    spans, key, start = {}, None, None
    for n, ln in enumerate([*lines, "end:"]):
        top = ln[:1].isalpha() and ":" in ln
        if top:
            if key is not None:
                spans[key] = (start, n)
            key, start = ln.split(":", 1)[0], n
    return spans


def lcs_yaml(src_text: str, header: list[str], set_keys: dict, drop=()) -> str:
    """The export's learned_lcs.yaml with top-level ``set_keys`` lines replaced / appended and
    ``drop`` removed; matrices keep the exporter's bytes."""
    lines = src_text.splitlines()
    spans = top_spans(lines)
    out, n = [], 0
    starts = {a: (k, b) for k, (a, b) in spans.items()}
    while n < len(lines):
        if n in starts:
            k, b = starts[n]
            if k in drop:
                n = b
                continue
            if k in set_keys:
                out.append(f"{k}: {set_keys[k]}")
                n = b
                continue
        out.append(lines[n])
        n += 1
    for k, v in set_keys.items():
        if k not in spans:
            out.append(f"{k}: {v}")
    return "\n".join(header + out) + "\n"


def learned_block(lcs_rel: str, tols, durations, floors: dict, costs: dict,
                  demo_traj: dict | None) -> list[str]:
    c = {**COSTS, **(costs or {})}
    b = ["learned_mpc:",
         f"  lcs_file: {lcs_rel}",
         "  latent_state_channel: LATENT_STATE",
         "  debug_channel: LEARNED_MPC_DEBUG",
         f"  w_q: {fnum(c['w_q'])}",
         f"  w_r: {fnum(c['w_r'])}",
         "  g_x: 0.0", "  g_lambda: 2.0", "  g_u: 0.0", "  g_eta: 1.0",
         "  u_x: 0.0", "  u_lambda: 10.0", "  u_u: 0.0", "  u_eta: 1.0",
         "  w_g: 0.2", "  w_u: 0.5",
         "  goal_tol: -1.0",
         f"  max_duration_s: {fnum(durations[-1])}",
         f"  stage_goal_tols: {fvec(tols)}",
         f"  stage_max_durations_s: {fvec(durations)}",
         "  penalize_input_change: false",
         "  honor_penalize_input_change: true  # c3 treats false as on",
         f"  admm_iter: {int(c['admm_iter'])}",
         "  rho_scale: 3.0", "  gamma: 1.0", "  projection_type: C3+", "  u_bound_scale: 1.0",
         "  ee_constraints:  # hard world-z bounds on x_1..x_{N-1}; w_p 0 = no EE cost",
         "    enabled: true",
         f"    z_min_franka: {fnum(floors['z_min_franka'])}",
         f"    z_min_ur: {fnum(floors['z_min_ur'])}",
         f"    z_max_franka: {fnum(floors.get('z_max_franka', math.inf))}",
         f"    z_max_ur: {fnum(floors.get('z_max_ur', math.inf))}",
         "    w_p: 0.0",
         "  ur_line_exact_dt: true"]
    if demo_traj is not None:
        b += ["  demo_traj:",
              f"    file: {demo_traj['file']}",
              "    ref_mode: progress",
              "    time_scale: 1.0",
              f"    progress_window: {int(demo_traj.get('progress_window', 10))}",
              "    w_p: 0.0",
              "    complete_on_final_dist: false"]
    return b


def params_yaml(base_text: str, header: list[str], block: list[str]) -> str:
    """The base params' top-level keys verbatim (comments dropped) + a new learned_mpc block."""
    lines = [ln for ln in base_text.splitlines() if not ln.lstrip().startswith("#")]
    spans = top_spans(lines)
    a, b = spans["learned_mpc"]
    body = lines[:a] + lines[b:]
    return "\n".join(header + body + block) + "\n"


def default_floors() -> dict:
    doc = yaml.safe_load((REPO / BASE_MPPI).read_text())
    return {k: float(doc[k]) for k in ("z_min_franka", "z_min_ur")}


def box_z_floors(floors: dict, box: dict) -> dict:
    """Support box z faces as hard ee_constraints (lo raises the floor, hi = ceiling)."""
    lo, hi = np.asarray(box["lo"], float), np.asarray(box["hi"], float)
    return {"z_min_franka": max(floors["z_min_franka"], lo[2]),
            "z_min_ur": max(floors["z_min_ur"], lo[5]),
            "z_max_franka": float(hi[2]), "z_max_ur": float(hi[5])}


def build(deploy_dir: Path, waypoints: Path | None, box_json: Path | None, goal_json: Path | None,
          tag: str, variant: str, out_rel: str, floors: dict | None = None,
          stage_max_durations_s=(10.0, 30.0), costs: dict | None = None,
          box: dict | None = None, demo_offset: int = DEMO_OFFSET, root: Path = WORKTREE,
          dry: bool = False) -> dict:
    """Write one variant into ``root/out_rel``; returns its manifest (worktree-relative paths).
    ``box_json`` None: the deploy's own u bounds (= box_i in a loop export)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")
    deploy_dir = Path(deploy_dir)
    src = deploy_dir / "learned_lcs.yaml"
    with np.load(deploy_dir / "deploy.npz", allow_pickle=True) as d:
        dep = {k: d[k] for k in ("z_goal_stage1", "z_goal_stage2", "z_std", "u_lb", "u_ub",
                                 "goal_tol_whitened", "goal_frames")}
    bx = json.loads(Path(box_json).read_text()) if box_json else {"u_lb": dep["u_lb"],
                                                                    "u_ub": dep["u_ub"]}
    u_lb, u_ub = np.asarray(bx["u_lb"], float), np.asarray(bx["u_ub"], float)
    if not (np.allclose(u_lb, dep["u_lb"], rtol=0, atol=1e-12)
            and np.allclose(u_ub, dep["u_ub"], rtol=0, atol=1e-12)):
        raise ValueError(f"{box_json}: u bounds != the deploy's (box_i)")
    floors = dict(floors or default_floors())
    if box is not None:
        floors = box_z_floors(floors, box)
    wp = None
    if variant != "pp2":
        if waypoints is None:
            raise ValueError(f"variant {variant} needs --waypoints")
        with np.load(waypoints, allow_pickle=False) as f:
            wp = {k: f[k] for k in f.files}
        if not np.allclose(wp["z_std"], dep["z_std"], rtol=1e-12, atol=0):
            raise ValueError(f"{waypoints}: z_std != the deploy's (another encoder)")
    tols, durations, drop, demo_traj = None, None, (), None
    set_keys = {"u_lb": fvec(u_lb), "u_ub": fvec(u_ub)}
    frames = [int(x) for x in np.asarray(dep["goal_frames"]).ravel()]
    mapping = {}
    if variant == "pp2":
        g = json.loads(Path(goal_json).read_text()) if goal_json else {}
        tol1 = float(g.get("tol_1", dep["goal_tol_whitened"]))
        tols, durations = [tol1, NEVER_TOL], [float(x) for x in stage_max_durations_s]
        z1, z2 = dep["z_goal_stage1"], dep["z_goal_stage2"]
        mapping = {"stage_frames": frames, "source": "deploy export goals",
                   "tol_source": "goal.json tol_1" if "tol_1" in g else "goal_tol_whitened"}
    else:
        fr = [int(x) for x in wp["frames"]]
        k_latch = fr.index(int(wp["latch"]))
        ms = [float(x) for x in wp["max_duration_s"]]
        last = min(ms[-1], float(stage_max_durations_s[-1]))
        z1, z2 = wp["z_wp"][k_latch], wp["z_wp"][-1]
        dropped = [f for j, f in enumerate(fr) if j not in (k_latch, len(fr) - 1)]
        if variant == "wp2":
            tols = [float(wp["tol"][k_latch]), NEVER_TOL]
            durations = [float(sum(ms[:k_latch + 1])), last]
            mapping = {"stage_frames": [fr[k_latch], fr[-1]], "waypoint_index": [k_latch,
                       len(fr) - 1], "dropped_waypoint_frames": dropped,
                       "waypoint_tols": [float(x) for x in wp["tol"]], "waypoint_max_s": ms}
        elif variant == "seat":
            tols, durations, drop = [NEVER_TOL], [last], ("z_goal_stage1", "z_goal_stage2")
            mapping = {"stage_frames": [fr[-1]], "dropped_waypoint_frames": fr[:-1]}
        else:
            tols, durations = [NEVER_TOL], [float(sum(ms[:-1])) + last]
            mapping = {"traj_frames": [demo_offset, len(wp["z_traj"]) - 1],
                       "replaces_waypoint_frames": fr}
    d12 = {"stage1_vs_export_whitened": float(np.linalg.norm((z1 - dep["z_goal_stage1"])
                                                             / dep["z_std"])),
           "stage2_vs_export_whitened": float(np.linalg.norm((z2 - dep["z_goal_stage2"])
                                                             / dep["z_std"]))}
    set_keys["z_goal"] = fvec(z2)
    if variant != "seat":
        set_keys.update(z_goal_stage1=fvec(z1), z_goal_stage2=fvec(z2))
    src_text = src.read_text()
    if not re.search(r"^n_engaged_episodes:", src_text, re.MULTILINE):
        set_keys["n_engaged_episodes"] = "0  # absent from belt_seg exports; C++ needs the key"
    gsrc = (f"c3plus {variant}: " + (f"waypoints {rel(waypoints)} frames "
                                     f"{mapping.get('stage_frames') or mapping['traj_frames']}"
                                     if wp is not None else
                                     yaml.safe_load(src_text)["goal_source"]))
    set_keys["goal_source"] = json.dumps(gsrc)
    name = f"{tag}_{variant}"
    out_dir = Path(root) / out_rel
    lcs_rel = f"{out_rel}/learned_lcs/learned_lcs_{name}.yaml"
    head = [(f"# c3plus_params.py {variant}: {src} (sha256 {sha(src)[:12]}); "
             f"u bounds {rel(box_json) if box_json else 'deploy.npz'}")]
    texts = {lcs_rel: lcs_yaml(src_text, head, set_keys, drop)}
    if variant == "track":
        import make_demo_goals as mdg

        traj_rel = f"{out_rel}/learned_lcs/demo_traj_{name}.yaml"
        ep = Path(str(wp["demo_episode"]))
        with np.load(ep, allow_pickle=True) as e:
            ee_f = np.asarray(e["sim_ee_franka"], np.float64)[demo_offset:]
            ee_u = np.asarray(e["sim_ee_ur"], np.float64)[demo_offset:]
        z = np.asarray(wp["z_traj"], np.float64)[demo_offset:]
        doc = {"n_frames": len(z), "dt": round(float(wp["dt"]), 6), "z": z,
               "ee_pose_franka": ee_f, "ee_pose_ur": ee_u, "demo_episode": str(ep),
               "demo_sha256": sha(ep), "deploy_sha256": str(wp["deploy_sha256"]),
               "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        demo_traj = {"file": traj_rel}
        if not dry:
            mdg.write_traj(doc, Path(root) / traj_rel)
    p_rel = f"{out_rel}/round_belt_controller_params_learned_eval_{name}.yaml"
    base = Path(root) / BASE_PARAMS
    sf = mapping.get("stage_frames") or mapping.get("traj_frames")
    p_head = [(f"# c3plus_params.py {variant} ({tag}): top level = {BASE_PARAMS} (sha256 "
               f"{sha(base)[:12]}); learned_mpc = V10's costs, goal frames {sf}")]
    texts[p_rel] = params_yaml(base.read_text(), p_head, learned_block(
        lcs_rel, tols, durations, floors, costs, demo_traj))
    if not dry:
        (out_dir / "learned_lcs").mkdir(parents=True, exist_ok=True)
        for r, t in texts.items():
            (Path(root) / r).write_text(t)
    files = {"params": p_rel, "lcs_yaml": lcs_rel, **({"demo_traj": demo_traj["file"]}
                                                       if demo_traj else {})}
    return {"tag": tag, "variant": variant, "files": files,
            "sha256": {} if dry else {k: sha(Path(root) / v) for k, v in files.items()},
            "deploy": str(deploy_dir / "deploy.npz"), "waypoints": None if wp is None
            else rel(waypoints), "box_json": rel(box_json) if box_json else None,
            "stage_goal_tols": tols,
            "stage_max_durations_s": durations, "ee_constraints": floors,
            "costs": {**COSTS, **(costs or {})}, "mapping": mapping, "goal_diff": d12,
            "support_box_z": box is not None}


# ---------------------------------------------------------------- check (C++ load mirror)


def header_schema(root: Path = WORKTREE) -> dict[str, dict[str, bool]]:
    """struct -> {visited field: is std::optional} from the C++ headers."""
    out = {}
    for h in HEADERS:
        text = (root / h).read_text()
        for m in re.finditer(r"\nstruct (\w+) \{(.*?)\n\};", text, re.DOTALL):
            body = m.group(2)
            opt = set(re.findall(r"std::optional<[^;]*>\s+(\w+)\s*(?:\{[^;]*\})?;", body))
            out[m.group(1)] = {v: v in opt for v in
                               re.findall(r"a->Visit\(DRAKE_NVP\((\w+)\)\)", body)}
    return out


def _keys(doc: dict, schema: dict[str, bool], where: str, errs: list[str]) -> None:
    for k in doc:
        if k not in schema:
            errs.append(f"{where}: unknown key {k!r} (drake: allow_yaml_with_no_cpp false)")
    for k, opt in schema.items():
        if k not in doc and not opt:
            errs.append(f"{where}: {k!r} is missing (drake: allow_cpp_with_no_yaml false)")


def _mat(doc, k, rows, cols, errs, where):
    a = np.asarray(doc.get(k), dtype=np.float64)
    if a.shape != (rows, cols) or not np.all(np.isfinite(a)):
        errs.append(f"{where}: {k} shape {a.shape} != ({rows}, {cols}) or non-finite")
    return a


def _vec(doc, k, n, errs, where):
    a = np.asarray(doc.get(k), dtype=np.float64)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        errs.append(f"{where}: {k} shape {a.shape} != ({n},) or non-finite")
    return a


def check_lcs(doc: dict, schema: dict, errs: list[str], where: str) -> dict:
    """LearnedLcsParams::Validate."""
    _keys(doc, schema["LearnedLcsParams"], where, errs)
    nx, nu, nl, N, dt = (doc.get(k) for k in ("n_x", "n_u", "n_lam", "N", "dt"))
    if not (all(isinstance(v, int) and v > 0 for v in (nx, nu, nl, N)) and dt > 0):
        errs.append(f"{where}: n_x/n_u/n_lam/N/dt {nx, nu, nl, N, dt}")
        return {}
    for k, r, c in (("A", nx, nx), ("B", nx, nu), ("D", nx, nl), ("E", nl, nx), ("F", nl, nl),
                    ("H", nl, nu)):
        _mat(doc, k, r, c, errs, where)
    for k, n in (("d", nx), ("c", nl), ("z_goal", nx), ("z_std", nx), ("u_lb", nu),
                 ("u_ub", nu), ("z_start_example", nx)):
        _vec(doc, k, n, errs, where)
    zs = np.asarray(doc["z_std"], float)
    if zs.min() <= 0:
        errs.append(f"{where}: z_std min <= 0")
    if (np.asarray(doc["u_ub"], float) - np.asarray(doc["u_lb"], float)).min() <= 0:
        errs.append(f"{where}: u_ub - u_lb min <= 0")
    if not float(doc.get("goal_tol_whitened", 0)) > 0:
        errs.append(f"{where}: goal_tol_whitened <= 0")
    s1, s2 = doc.get("z_goal_stage1"), doc.get("z_goal_stage2")
    if (s1 is None) != (s2 is None):
        errs.append(f"{where}: z_goal_stage1 / z_goal_stage2 both or neither")
    n_stages = 1
    if s1 is not None:
        n_stages = 2
        _vec(doc, "z_goal_stage1", nx, errs, where)
        _vec(doc, "z_goal_stage2", nx, errs, where)
        if np.abs(np.asarray(s2, float) - np.asarray(doc["z_goal"], float)).max() > 1e-9:
            errs.append(f"{where}: z_goal_stage2 != z_goal (1e-9)")
    for k in ("q_matrix", "q_matrix_stage1", "q_matrix_stage2", "bz_hidden"):
        if doc.get(k) is not None:
            errs.append(f"{where}: {k} set (not handled by this mirror)")
    return {"n_x": nx, "n_u": nu, "n_lam": nl, "N": N, "dt": float(dt), "n_stages": n_stages}


def check_params(params, root: Path = WORKTREE, base=BASE_PARAMS) -> dict:
    """Mirror of RoundBeltControllerParams load + LearnedMpcParams::Serialize/LoadLcs."""
    root = Path(root)
    p = Path(params)
    p = p if p.is_absolute() else root / p
    errs: list[str] = []
    schema = header_schema(root)
    doc = yaml.safe_load(p.read_text())
    ref = yaml.safe_load((root / base).read_text())
    if set(doc) != set(ref):
        errs.append(f"top-level keys != {base}: {sorted(set(doc) ^ set(ref))}")
    diff = [k for k in ref if k != "learned_mpc" and doc.get(k) != ref[k]]
    lm = doc.get("learned_mpc") or {}
    _keys(lm, schema["LearnedMpcParams"], "learned_mpc", errs)
    if lm.get("projection_type") not in ("C3+", "QP") or not lm.get("u_bound_scale", 0) > 0 \
            or not lm.get("max_duration_s", 0) > 0:
        errs.append("learned_mpc: projection_type / u_bound_scale / max_duration_s")
    ee = lm.get("ee_constraints")
    if ee is not None:
        _keys(ee, schema["LearnedEeConstraints"], "ee_constraints", errs)
        if not (ee["z_min_franka"] < ee["z_max_franka"] and ee["z_min_ur"] < ee["z_max_ur"]
                and ee["w_p"] >= 0):
            errs.append(f"ee_constraints {ee}")
    out = {"params": str(p), "params_sha256": sha(p), "non_learned_diffs_vs_base": diff}
    lcs_p = root / lm["lcs_file"]
    if not lcs_p.is_file():
        errs.append(f"lcs_file {lcs_p} missing")
        return {**out, "errors": errs, "pass": False}
    lcs = yaml.safe_load(lcs_p.read_text())
    info = check_lcs(lcs, schema, errs, "lcs_file")
    tracks = lm.get("demo_traj") is not None
    if tracks:
        dt_ = lm["demo_traj"]
        _keys(dt_, schema["LearnedDemoTrajParams"], "demo_traj", errs)
        if dt_.get("ref_mode") not in ("time", "progress") or dt_.get("w_p", 0) != 0.0:
            errs.append(f"demo_traj ref_mode / w_p {dt_}")
        tp = root / dt_["file"]
        tr = yaml.safe_load(tp.read_text()) if tp.is_file() else {}
        _keys(tr, schema["LearnedDemoTraj"], "demo_traj file", errs)
        T = int(tr.get("n_frames", 0))
        if T <= 0 or abs(float(tr.get("dt", 0)) - info.get("dt", -1)) > 1e-9:
            errs.append(f"demo_traj n_frames {T} / dt {tr.get('dt')} vs lcs dt")
        _mat(tr, "z", T, info.get("n_x", 0), errs, "demo_traj")
        _mat(tr, "ee_pose_franka", T, 7, errs, "demo_traj")
        _mat(tr, "ee_pose_ur", T, 7, errs, "demo_traj")
        info["demo_traj_frames"] = T
    n_st = 1 if tracks else info.get("n_stages", 0)
    for k in ("stage_goal_tols", "stage_max_durations_s"):
        if lm.get(k) is not None and len(lm[k]) != n_st:
            errs.append(f"{k} has {len(lm[k])} entries, Stages() {n_st}")
    tols = [t if t > 0 else (lm["goal_tol"] if lm["goal_tol"] > 0 else lcs["goal_tol_whitened"])
            for t in (lm.get("stage_goal_tols") or [-1.0] * n_st)]
    durs = [x if x > 0 else lm["max_duration_s"]
            for x in (lm.get("stage_max_durations_s") or [-1.0] * n_st)]
    n_aug = info.get("n_x", 0) + (6 if ee is not None and ee.get("enabled") else 0)
    plus = lm.get("projection_type") == "C3+"
    n_lam = info.get("n_lam", 0)  # C3 z = [x, lam, u] (+ eta for C3+)
    out.update(lcs_file=str(lcs_p), lcs_sha256=sha(lcs_p), **info, n_stages_effective=n_st,
               stage_tols_effective=tols, stage_durations_effective=durs,
               c3_n_z=n_aug + n_lam + info.get("n_u", 0) + (n_lam if plus else 0),
               admm_iter=lm.get("admm_iter"), errors=errs)
    out["pass"] = not errs
    return out


def stage_replay(params, root: Path = WORKTREE, waypoints: Path | None = None) -> dict | None:
    """The C++ stage switch (first latent with whitened dist < tol, no streak) on the goal
    demo's own latents: the frame stage 0 would end on."""
    if waypoints is None:
        return None
    doc = yaml.safe_load((Path(root) / params).read_text())
    lm = doc["learned_mpc"]
    lcs = yaml.safe_load((Path(root) / lm["lcs_file"]).read_text())
    if lm.get("demo_traj") is not None or lcs.get("z_goal_stage1") is None:
        return None
    with np.load(waypoints, allow_pickle=False) as f:
        z = np.asarray(f["z_traj"], float)
    zs, g1 = np.asarray(lcs["z_std"], float), np.asarray(lcs["z_goal_stage1"], float)
    d = np.linalg.norm((z - g1) / zs, axis=1)
    tol = lm["stage_goal_tols"][0]
    hit = np.flatnonzero(d[DEMO_OFFSET:] < tol)
    return {"tol": tol, "first_frame_inside": int(hit[0] + DEMO_OFFSET) if len(hit) else None,
            "dist_at_offset": float(d[DEMO_OFFSET]), "min_dist": float(d.min()),
            "argmin_frame": int(d.argmin())}


# ---------------------------------------------------------------- CLI


def cmd_write(a) -> int:
    dj = json.loads(Path(a.deploy_json).read_text())
    it = Path(a.deploy_json).parent
    wp = Path(a.waypoints) if a.waypoints else it / "latent_waypoints.npz"
    box = json.loads(Path(a.support_box).read_text()) if a.support_box else None
    out = (REPO / a.out).resolve()
    (out / "files").mkdir(parents=True, exist_ok=True)
    gj = Path(dj["goal_json"]) if dj.get("goal_json") else None
    mans = []
    for v in a.variants.split(","):
        m = build(Path(dj["deploy_dir"]), wp if wp.is_file() else None, Path(dj["box_json"]),
                  gj, a.tag + ("_boxz" if box else ""), v, f"{a.archive}/{a.tag}",
                  stage_max_durations_s=a.stage_max_durations_s, box=box)
        m["check"] = check_params(m["files"]["params"])
        m["stage_replay_goal_demo"] = stage_replay(m["files"]["params"],
                                                   waypoints=wp if wp.is_file() else None)
        for f in m["files"].values():
            shutil.copy2(WORKTREE / f, out / "files" / Path(f).name)
        mans.append(m)
        print(f"{m['tag']} {v}: {'PASS' if m['check']['pass'] else 'FAIL'} "
              f"{m['files']['params']} stages {m['check']['n_stages_effective']} tols "
              f"{np.round(m['stage_goal_tols'], 4).tolist()} durations "
              f"{m['stage_max_durations_s']} n_z {m['check']['c3_n_z']}"
              + "".join(f"\n  {e}" for e in m["check"]["errors"]))
    name = "manifest" + ("_boxz" if box else "") + ".json"
    (out / name).write_text(json.dumps({"model": dj.get("model"), "deploy_json": rel(
        a.deploy_json), "worktree": str(WORKTREE), "variants": mans}, indent=1) + "\n")
    print(f"wrote {rel(out / name)}")
    return 0 if all(m["check"]["pass"] for m in mans) else 1


def cmd_check(a) -> int:
    bad = 0
    for p in a.params:
        r = check_params(p)
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {p}: n_x {r.get('n_x')} N {r.get('N')} "
              f"stages {r.get('n_stages_effective')} tols {r.get('stage_tols_effective')} "
              f"durations {r.get('stage_durations_effective')} n_z {r.get('c3_n_z')}"
              + "".join(f"\n  {e}" for e in r["errors"]))
        bad += not r["pass"]
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--deploy-json", required=True, help="a loop iteration's deploy.json")
    w.add_argument("--waypoints", default=None, help="default: deploy.json's latent_waypoints.npz")
    w.add_argument("--tag", required=True)
    w.add_argument("--variants", default="wp2,pp2,seat,track")
    w.add_argument("--stage-max-durations-s", type=float, nargs=2, default=(10.0, 30.0))
    w.add_argument("--support-box", default=None,
                   help="support_box.json: its z faces as hard ee_constraints (opt-in)")
    w.add_argument("--archive", default=ARCHIVE, help="worktree-relative output dir")
    w.add_argument("--out", required=True, help="manifest + copies (this repo)")
    c = sub.add_parser("check")
    c.add_argument("--params", nargs="+", required=True, help="worktree-relative params yaml")
    a = ap.parse_args()
    return cmd_write(a) if a.cmd == "write" else cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
