#!/usr/bin/env python3
"""Offline overshoot test: recorded MPPI rollouts' latents replayed through the node.

Per episode of a harness collect dir: the node (this tree's, or ``--module-dir``) on a fake LCM
with the run's own flags (index.json args, the episode's demo offset from its first solve row),
fed the recorded ``sim_latent`` / ``sim_ee_*`` from the first solve's utime, 75 ms apart (open
loop: the states are the recording's). Variants = the run's mppi yaml + a yaml fragment (e.g.
``{overshoot: ...}``, ``seat_tol: tol_2`` = the run's goal.json tol_2); ``action_noise_frac`` 0
in every variant. Reported around the moment the
recorded seated-goal distance (whitened, to the node's last goal) bottoms: u_0 Franka y and the
plan's summed Franka y over H, mean over [t_min, t_min + window] and [t_min, end], the hold
entry (t, recorded Franka y), and the u_0 y summed after the bottom (the +y the node commands).

    uv run --frozen python scripts/lcs/overshoot_offline.py \\
        --runs data/lcs/realfirst/cmp/full/iter4/collect --episodes 0,3 \\
        --variant off= --variant on=data/lcs/realfirst/cmp/overshoot/offline/on.yaml \\
        --out data/lcs/realfirst/cmp/overshoot/offline/replay
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from task_common.latent_encoder import LATENT_STATE_CHANNEL, latent_state_message


def load_modules(module_dir: Path | None):
    """The node module; with ``module_dir`` its ``mppi_lcs.py`` / ``mppi_controller_node.py``
    replace the tree's (``task_common.mppi_lcs`` injected before the node import)."""
    if module_dir is None:
        import mppi_controller_node as node_mod
        return node_mod
    spec = importlib.util.spec_from_file_location("task_common.mppi_lcs",
                                                  module_dir / "mppi_lcs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["task_common.mppi_lcs"] = mod
    spec.loader.exec_module(mod)
    spec = importlib.util.spec_from_file_location("mppi_controller_node_alt",
                                                  module_dir / "mppi_controller_node.py")
    node_mod = importlib.util.module_from_spec(spec)
    sys.modules["mppi_controller_node_alt"] = node_mod
    spec.loader.exec_module(node_mod)
    return node_mod


class FakeLc:
    def __init__(self) -> None:
        self.cb = None

    def subscribe(self, _ch, cb) -> None:
        self.cb = cb

    def publish(self, _ch, _data) -> None:
        pass


def node_argv(run: Path, a: dict, cfg: Path, log: Path, offset: int) -> list[str]:
    argv = ["--lcm-url", "fake", "--deploy", a["deploy"], "--config", str(cfg),
            "--goal", a.get("mppi_goal") or "deploy", "--stages", a.get("mppi_stages") or "last",
            "--max-duration-s", str(a.get("mppi_max_duration_s") or 30.0), "--log", str(log)]
    if argv[argv.index("--goal") + 1] == "waypoints":
        argv[argv.index("--goal") + 1] = f"waypoints:{REPO_ROOT / a['mppi_waypoints']}"
    if a.get("mppi_clearance"):
        argv += ["--clearance-model", str(run / "clearance_model.npz")]
    if a.get("mppi_goal_mode"):
        argv += ["--goal-mode", a["mppi_goal_mode"]]
    argv += ["--demo-offset-frames", str(offset)]
    if a.get("mppi_demo_actions"):
        argv += ["--demo-actions", a["mppi_demo_actions"]]
    if a.get("mppi_waypoint_bank"):
        argv += ["--waypoint-bank", a["mppi_waypoint_bank"], "--bank-knn",
                 str(a.get("mppi_bank_knn") or 1)]
    return argv


def replay(node_mod, run: Path, a: dict, ep: dict, fragment: dict,
           max_latents: int | None = None, period_s: float = 0.0) -> dict:
    f = run / ep["file"]
    solves = run / "controller" / Path(ep["file"]).with_suffix(".solves.jsonl").name
    rec = [json.loads(x) for x in solves.read_text().splitlines() if x.strip()]
    with np.load(f, allow_pickle=True) as d:
        ut, z = np.asarray(d["utime"]), np.asarray(d["sim_latent"], np.float64)
        ef, eu = np.asarray(d["sim_ee_franka"]), np.asarray(d["sim_ee_ur"])
    k0 = int(np.flatnonzero(ut == rec[0]["utime"])[0])
    offset = int(rec[0].get("idx", a.get("mppi_demo_offset_frames") or 0))
    base = yaml.safe_load((REPO_ROOT / a["mppi_config"]).read_text())
    doc = {**base, **fragment, "action_noise_frac": 0.0}
    ov = doc.get("overshoot")
    if ov and ov.get("seat_tol") == "tol_2":  # the run's own seated tol (loop goal.json)
        doc["overshoot"] = {**ov, "seat_tol": json.loads((run.parent / "goal.json")
                                                         .read_text())["tol_2"]}
    with tempfile.TemporaryDirectory() as tmp:
        cfg, log = Path(tmp) / "mppi.yaml", Path(tmp) / "solves.jsonl"
        cfg.write_text(yaml.safe_dump(doc))
        args = node_mod.create_parser().parse_args(node_argv(run, a, cfg, log, offset))
        out = io.StringIO()
        plans = []
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(out):
            node = node_mod.MppiControllerNode(args, FakeLc())
            node.start()
            end = len(ut) if max_latents is None else min(len(ut), k0 + max_latents)
            for k in range(k0, end):
                if node.done:
                    break
                msg = latent_state_message(int(ut[k]), ut[k] * 1e-6, z[k], ef[k], eu[k],
                                           np.zeros(40))
                n0, tk = node.n_solves, time.perf_counter()
                node.lc.cb(LATENT_STATE_CHANNEL, msg.encode())
                node._process(*node.pending)
                node.pending = None
                if node.n_solves > n0:  # a stage switch consumes its latent without a solve
                    plans.append((k, node.plan.copy()))
                time.sleep(max(0.0, period_s - (time.perf_counter() - tk)))
            node.log.close()
        rows = [json.loads(x) for x in log.read_text().splitlines()]
        seat = node.stages[-1].goal
        z_std = node.lcs.z_std_np
    wall = time.perf_counter() - t0
    return {"rows": rows, "plans": plans, "stdout": out.getvalue(), "wall_s": wall, "rec": rec,
            "seat": seat, "z_std": z_std, "k0": k0, "ut": ut, "z": z, "fy": ef[:, 1] * 1e3,
            "offset": offset}


def metrics(r: dict, window_s: float) -> dict:
    ut, k0 = r["ut"], r["k0"]
    d = np.linalg.norm((r["z"][k0:] - r["seat"]) / r["z_std"], axis=1)
    t = (ut[k0:] - ut[k0]) * 1e-6
    kmin = int(np.argmin(d))
    ks = np.array([k for k, _ in r["plans"]]) - k0
    u0y = np.array([p[0, 1] for _, p in r["plans"]]) * 1e3
    sumy = np.array([p[:, 1].sum() for _, p in r["plans"]]) * 1e3
    tp = t[ks]
    w = (tp >= t[kmin]) & (tp <= t[kmin] + window_s)
    after = tp >= t[kmin]
    tr = np.array([x["utime"] for x in r["rec"]]) * 1e-6 - ut[k0] * 1e-6
    ry = np.array([x["u0"][1] for x in r["rec"]]) * 1e3
    rw, ra = (tr >= t[kmin]) & (tr <= t[kmin] + window_s), tr >= t[kmin]
    modes = [x.get("mode") for x in r["rows"]]
    hold = next((i for i, m in enumerate(modes) if m == "hold"), None)
    out = {"n_solves": len(r["rows"]), "t_min_s": float(t[kmin]), "d_min": float(d[kmin]),
           "fy_at_min_mm": float(r["fy"][k0 + kmin]), "fy_max_mm": float(r["fy"][k0:].max()),
           "u0y_win_mm": float(u0y[w].mean()), "sumy_win_mm": float(sumy[w].mean()),
           "u0y_after_mm": float(u0y[after].mean()), "frac_u0y_gt_0.5_after":
           float(np.mean(u0y[after] > 0.5)), "u0y_cum_after_mm": float(u0y[after].sum()),
           "u0y_at_min_mm": float(u0y[np.argmin(np.abs(tp - t[kmin]))]),
           "rec_u0y_win_mm": float(ry[rw].mean()), "rec_u0y_after_mm": float(ry[ra].mean()),
           "rec_u0y_cum_after_mm": float(ry[ra].sum()),
           "u0y_cummax_after_mm": float(np.max(np.cumsum(u0y[after]), initial=0.0)),
           "rec_u0y_cummax_after_mm": float(np.max(np.cumsum(ry[ra]), initial=0.0)),
           "series": {"t": np.round(tp, 3).tolist(), "u0y_mm": np.round(u0y, 3).tolist(),
                      "mode": modes, "rec_t": np.round(tr, 3).tolist(),
                      "rec_u0y_mm": np.round(ry, 3).tolist(),
                      "d": np.round(d, 3).tolist()},
           "hold_t_s": None if hold is None else float(tp[hold]),
           "hold_fy_mm": None if hold is None else float(r["fy"][k0 + ks[hold]]),
           "hold_d": None if hold is None else float(r["rows"][hold]["d_seat"]),
           "frac_hold": float(np.mean([m == "hold" for m in modes])),
           "prior_scale_mean": (float(np.mean([x["prior_scale"] for x in r["rows"]]))
                                if r["rows"] and "prior_scale" in r["rows"][0] else None),
           "solve_ms_p95": float(np.percentile([x["solve_ms"] for x in r["rows"]], 95)),
           "wall_s": r["wall_s"], "offset": r["offset"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="harness collect dirs")
    ap.add_argument("--episodes", default=None, help="comma list of episode indices (all)")
    ap.add_argument("--variant", action="append", required=True,
                    help="NAME=YAML (fragment merged over the run's mppi yaml; empty = as is)")
    ap.add_argument("--window-s", type=float, default=3.0)
    ap.add_argument("--max-latents", type=int, default=None, help="smoke: stop after N")
    ap.add_argument("--period-ms", type=float, default=0.0,
                    help="min wall time per latent (GPU sharing with a running sim)")
    ap.add_argument("--module-dir", type=Path, default=None,
                    help="dir with mppi_lcs.py + mppi_controller_node.py replacing the tree's")
    ap.add_argument("--out", type=Path, required=True)
    cli = ap.parse_args()
    node_mod = load_modules(cli.module_dir)
    variants = []
    for v in cli.variant:
        name, path = v.split("=", 1)
        variants.append((name, yaml.safe_load(Path(path).read_text()) if path else {}))
    cli.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in cli.runs:
        idx = json.loads((run / "index.json").read_text())
        a = idx["args"]
        eps = idx["episodes"]
        pick = (range(len(eps)) if cli.episodes is None
                else [int(x) for x in cli.episodes.split(",")])
        for i in pick:
            ep = eps[i]
            for name, frag in variants:
                r = replay(node_mod, run, a, ep, frag, cli.max_latents, cli.period_ms * 1e-3)
                m = metrics(r, cli.window_s)
                rows.append({"run": str(run), "episode": ep["file"], "outcome": ep["outcome"],
                             "variant": name, **m})
                print(f"{run.parent.parent.name}/{run.parent.name} {ep['file']} "
                      f"{ep['outcome']:8s} {name:8s} d_min {m['d_min']:.2f} @ "
                      f"{m['t_min_s']:.1f} s (F y {m['fy_at_min_mm']:.0f}, max "
                      f"{m['fy_max_mm']:.0f}); u0y win {m['u0y_win_mm']:+.2f} after "
                      f"{m['u0y_after_mm']:+.2f} cum {m['u0y_cum_after_mm']:+.0f} mm (rec "
                      f"{m['rec_u0y_win_mm']:+.2f} / {m['rec_u0y_after_mm']:+.2f} / "
                      f"{m['rec_u0y_cum_after_mm']:+.0f}); cummax {m['u0y_cummax_after_mm']:.0f}"
                      f" (rec {m['rec_u0y_cummax_after_mm']:.0f}) mm; sum_y win "
                      f"{m['sumy_win_mm']:+.1f}; "
                      "hold " + ("-" if m["hold_t_s"] is None else
                                 f"{m['hold_t_s']:.1f} s (F y {m['hold_fy_mm']:.0f}, d "
                                 f"{m['hold_d']:.2f})")
                      + f"; p95 {m['solve_ms_p95']:.1f} ms", flush=True)
    (cli.out / "replay.json").write_text(json.dumps(
        {"variants": {n: f for n, f in variants}, "window_s": cli.window_s, "rows": rows},
        indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
