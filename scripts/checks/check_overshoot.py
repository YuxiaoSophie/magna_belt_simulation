#!/usr/bin/env python3
"""Checks of the MPPI overshoot guard (``MppiParams.overshoot``, ``OvershootGuard``, node hook).

O1 params: default off; every realfirst ``mppi*.yaml`` loads; ``to_dict`` == the pre-change
   module's plus ``overshoot: None``; bad blocks are refused.
O2 solver identity: pre-change vs tree ``mppi_plan`` (3 configs, cold + warm, with / without a
   prior): plans and stats identical.
O3 node identity (guard off): pre-change node + solver vs the tree's on a recorded stream
   (waypoints + prior run, and a deploy-goal stream): published bytes (debug ``solve_ms``
   masked), stdout, log rows minus timing identical.
O4 guard logic: fade ramp / last-stage zero, hold entry after exactly ``streak`` latents,
   latch / ``exit_k``, entry / best goals, cold only on entry, ``params_for`` scales, near.
O5 node with the guard: per solve the passed ``u_demo`` == prior x fade, hold solves have no
   prior, the hold goal and scaled weights, the first hold solve is cold; JSONL keys.

    uv run --frozen python scripts/checks/check_overshoot.py --pre-dir <dir with the
        pre-change mppi_lcs.py + mppi_controller_node.py>
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

from task_common.latent_encoder import LATENT_STATE_CHANNEL, latent_state_message

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # cpu: clearance off (torch path slow)
RUN = REPO / "data/lcs/realfirst/cmp/C1_transport/iter1/collect"
DEBUG_CH = "LEARNED_MPC_DEBUG_CHECK"
N_LAT = 60


def load_pair(d: Path | None, tag: str):
    """``(mppi_lcs, node)`` modules from ``d`` (``mppi_lcs.py`` + ``mppi_controller_node.py``)
    or the tree's."""
    src = {"lcs": REPO / "src/task_common/mppi_lcs.py",
           "node": REPO / "scripts/lcs/mppi_controller_node.py"}
    if d is not None:
        src = {"lcs": d / "mppi_lcs.py", "node": d / "mppi_controller_node.py"}
    spec = importlib.util.spec_from_file_location(f"mppi_lcs_{tag}", src["lcs"])
    lcs_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lcs_mod
    spec.loader.exec_module(lcs_mod)
    sys.modules["task_common.mppi_lcs"] = lcs_mod
    spec = importlib.util.spec_from_file_location(f"mppi_node_{tag}", src["node"])
    node_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = node_mod
    spec.loader.exec_module(node_mod)
    return lcs_mod, node_mod


class FakeLc:
    def __init__(self) -> None:
        self.cb, self.raw = None, []

    def subscribe(self, _ch, cb) -> None:
        self.cb = cb

    def publish(self, ch, data) -> None:
        self.raw.append((ch, data))


def masked(ch: str, data: bytes) -> bytes:
    from dairlib import lcmt_timestamped_saved_traj

    if ch != DEBUG_CH:
        return data
    m = lcmt_timestamped_saved_traj.decode(data)
    for b in m.saved_traj.trajectories:
        if b.trajectory_name == "scalars":
            b.datapoints[list(b.datatypes).index("solve_ms")] = [0.0]
    return m.encode()


def run_args() -> tuple[dict, dict, Path]:
    idx = json.loads((RUN / "index.json").read_text())
    return idx["args"], idx["episodes"][0], RUN / idx["episodes"][0]["file"]


def recorded(ep_file: Path, n: int):
    rows = [json.loads(x) for x in (RUN / "controller" / ep_file.with_suffix(
        ".solves.jsonl").name).read_text().splitlines()]
    with np.load(ep_file, allow_pickle=True) as d:
        ut, z = np.asarray(d["utime"]), np.asarray(d["sim_latent"], np.float64)
        ef, eu = np.asarray(d["sim_ee_franka"]), np.asarray(d["sim_ee_ur"])
    k0 = int(np.flatnonzero(ut == rows[0]["utime"])[0])
    sl = slice(k0, min(len(ut), k0 + n))
    return ut[sl], z[sl], ef[sl], eu[sl], int(rows[0].get("idx", 20))


def stream(node_mod, cfg_doc: dict, waypoints: bool, n: int = N_LAT, spy=None) -> dict:
    a, _, ep_file = run_args()
    ut, z, ef, eu, offset = recorded(ep_file, n)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, log = Path(tmp) / "mppi.yaml", Path(tmp) / "solves.jsonl"
        cfg.write_text(yaml.safe_dump(cfg_doc))
        argv = ["--lcm-url", "fake", "--deploy", a["deploy"], "--config", str(cfg),
                "--max-duration-s", "30", "--log", str(log), "--debug-channel", DEBUG_CH]
        if DEVICE == "cuda":
            argv += ["--clearance-model", str(RUN / "clearance_model.npz")]
        if waypoints:
            argv += ["--goal", f"waypoints:{REPO / a['mppi_waypoints']}", "--stages", "all",
                     "--goal-mode", "stage", "--demo-offset-frames", str(offset),
                     "--demo-actions", "auto"]
        args = node_mod.create_parser().parse_args(argv)
        lc, out = FakeLc(), io.StringIO()
        real = node_mod.mppi_plan
        if spy is not None:
            node_mod.mppi_plan = spy(real)
        try:
            with contextlib.redirect_stdout(out):
                node = node_mod.MppiControllerNode(args, lc)
                node.start()
                for k in range(len(ut)):
                    msg = latent_state_message(int(ut[k]), ut[k] * 1e-6, z[k], ef[k], eu[k],
                                               np.zeros(40))
                    lc.cb(LATENT_STATE_CHANNEL, msg.encode())
                    node._process(*node.pending)
                    node.pending = None
                node.log.close()
        finally:
            node_mod.mppi_plan = real
        rows = [json.loads(x) for x in log.read_text().splitlines()]
    for r in rows:
        r.pop("solve_ms", None)
    stdout = "\n".join(ln for ln in out.getvalue().splitlines()
                       if not ln.startswith(("Module ", "Warp ", "   ")))
    return {"raw": [(c, masked(c, b)) for c, b in lc.raw], "stdout": stdout, "rows": rows,
            "node": node, "z": z}


def base_doc(no_prior: bool = False) -> dict:
    a, _, _ = run_args()
    doc = yaml.safe_load(Path(a["mppi_config"]).read_text())
    doc["action_noise_frac"] = 0.2
    if DEVICE == "cpu":
        doc["clearance"] = None
    if no_prior:
        doc.update(prior_alpha=0.0, prior_weight=0.0)
    return doc


def o1(pre, tree) -> tuple[bool, str]:
    ok, notes = tree.MppiParams().overshoot is None, []
    yamls = sorted((REPO / "data/lcs/realfirst").rglob("mppi*.yaml"))
    n_on = 0
    for y in yamls:
        if {"stage_skip_ahead", "stage_no_timeout_into", "prior_mode", "bc_prior", "goal_set",
                "goal_set_tau", "stage_start", "value"} & set(yaml.safe_load(y.read_text()) or {}):
            continue  # schedule / policy-prior keys: tree only
        if "overshoot" in (yaml.safe_load(y.read_text()) or {}):  # guard on: tree only
            n_on += tree.MppiParams.from_yaml(y).overshoot is not None
            continue
        a, b = pre.MppiParams.from_yaml(y).to_dict(), tree.MppiParams.from_yaml(y).to_dict()
        for k, v in (("prior_sync", None), ("support_box", None), ("stage_skip_ahead", False),
                     ("stage_no_timeout_into", None), ("prior_mode", "demo"),
                     ("bc_prior", None),
                     ("goal_set", None), ("goal_set_tau", 0.0),
                     ("stage_start", None), ("value", None),
                     ("latent_support", None),
                     ("ee_support", None)):  # later fields at their defaults
            if k in b and b[k] == v and k not in a:
                b.pop(k)
        if b.pop("overshoot", "missing") is not None or a.pop("overshoot", None) is not None \
                or a != b:
            ok = False
            notes.append(f"diff {y}")
    bad = [{"fade": [1, 2]}, {"seat_tol": 0.0}, {"seat_tol": 1, "fade": [2, 1]},
           {"seat_tol": 1, "hold": {"goal": "x"}}, {"seat_tol": 1, "hold": {"streak": 0}},
           {"seat_tol": 1, "hold": {"enter_k": 1.0, "exit_k": 0.5}},
           {"seat_tol": 1, "hold": {"w_r": 0}}, {"seat_tol": 1, "near": {"k": 0}},
           {"seat_tol": 1, "other": 1}]
    refused = 0
    for ov in bad:
        try:
            tree.MppiParams(overshoot=ov)
        except ValueError:
            refused += 1
    ok = ok and refused == len(bad)
    return ok, (f"{len(yamls) - n_on} yamls: to_dict == pre + overshoot None; {n_on} with the "
                f"guard load; {refused}/{len(bad)} bad blocks refused"
                + (f"; {notes[:3]}" if notes else ""))


def o2(pre, tree) -> tuple[bool, str]:
    a, _, ep_file = run_args()
    _, z, ef, eu, offset = recorded(ep_file, 5)
    p0 = np.r_[ef[0, :3], eu[0, :3]]
    q = np.stack([ef[0, 3:7], eu[0, 3:7]])
    from latent_waypoints import Waypoints

    wp = Waypoints.load(REPO / a["mppi_waypoints"])
    n_cmp, ok = 0, True
    for doc in (base_doc(), base_doc(no_prior=True), {**base_doc(), "clearance": None}):
        outs = []
        for mod in (pre, tree):
            lcs = mod.TorchLcs.from_deploy(a["deploy"], DEVICE)
            prm = mod.MppiParams(**{k: v for k, v in doc.items()})
            if prm.clearance is not None:
                prm = dataclasses.replace(prm, clearance={**prm.clearance,
                                                          "model": str(RUN /
                                                                       "clearance_model.npz")})
            gen = torch.Generator(device=DEVICE).manual_seed(3)
            kw = {"q_now": q} if prm.clearance else {}
            ud = wp.u_seq(offset + 14, prm.H) if prm.prior_weight > 0 else None
            U1, i1 = mod.mppi_plan(lcs, prm, z[0], p0, None, wp.z_wp[3], None, gen,
                                   u_demo=ud, **kw)
            U2, i2 = mod.mppi_plan(lcs, prm, z[1], p0, None, wp.z_wp[3], U1, gen, u_demo=ud,
                                   **kw)
            outs.append((U1, U2, {k: v for k, v in {**i1, **{f"w_{k}": v for k, v in
                                                              i2.items()}}.items()
                                  if "solve_ms" not in k}))
        for x, y in zip(outs[0][:2], outs[1][:2]):
            ok = ok and np.array_equal(x, y)
        for k in outs[0][2]:
            va, vb = outs[0][2][k], outs[1][2][k]
            ok = ok and (np.array_equal(va, vb) if isinstance(va, np.ndarray) else
                         (va == vb or (isinstance(va, float) and math.isnan(va)
                                       and math.isnan(vb))))
        n_cmp += 1
    return ok, f"{n_cmp} configs x cold + warm: plans and stats identical (exact)"


def o3(pre_node, tree_node) -> tuple[bool, str]:
    res = []
    for wpts in (True, False):
        doc = base_doc(no_prior=not wpts)
        a = stream(pre_node, doc, wpts)
        b = stream(tree_node, doc, wpts)
        same = (a["raw"] == b["raw"], a["stdout"] == b["stdout"], a["rows"] == b["rows"])
        res.append((all(same), len(a["rows"]), len(a["raw"]), same))
    return all(r[0] for r in res), "; ".join(
        f"{'waypoints+prior' if k == 0 else 'deploy goal'}: {r[1]} solves, {r[2]} msgs, "
        f"bytes/stdout/rows {r[3]}" for k, r in enumerate(res))


def o4(tree) -> tuple[bool, str]:
    G = tree.OvershootGuard
    ok, notes = True, []

    def chk(c, msg):
        nonlocal ok
        if not c:
            ok = False
            notes.append(msg)

    g = G({"seat_tol": 2.0, "fade": [1.0, 2.0]})
    chk(g.fade(2.0, False) == 0.0 and g.fade(4.0, False) == 1.0
        and abs(g.fade(3.0, False) - 0.5) < 1e-12 and g.fade(9.0, False) == 1.0, "fade ramp")
    chk(g.fade(3.0, True) == 0.5, "fade_last_stage off")
    g2 = G({"seat_tol": 2.0, "fade_last_stage": True})
    chk(g2.fade(9.0, True) == 0.0 and g2.fade(9.0, False) == 1.0, "fade_last_stage")
    h = G({"seat_tol": 1.0, "hold": {"enter_k": 1.0, "streak": 3, "goal": "entry", "cold": True,
                                     "w_r": 10.0, "w_T": 2.0}})
    z = [np.full(2, float(k)) for k in range(10)]
    ds = [2.0, 0.5, 0.5, 2.0, 0.9, 0.8, 0.7, 0.6, 5.0, 0.1]
    st = [h.update(z[k], ds[k], False) for k in range(10)]
    modes = [s.mode for s in st]
    chk(modes == ["normal"] * 6 + ["hold"] * 4, f"hold entry/latch {modes}")
    chk([s.entered for s in st].count(True) == 1 and st[6].entered and st[6].cold
        and not st[7].cold, "entered/cold once")
    chk(all(np.array_equal(s.goal, z[6]) for s in st[6:]), "entry goal")
    chk(st[6].prior_scale == 0.0, "hold prior scale 0")
    b = G({"seat_tol": 1.0, "hold": {"streak": 1, "goal": "best", "exit_k": 3.0}})
    sb = [b.update(z[k], d, False) for k, d in enumerate([0.9, 0.5, 0.7, 3.5, 0.95])]
    chk([s.mode for s in sb] == ["hold", "hold", "hold", "normal", "hold"], "exit_k")
    chk(np.array_equal(sb[2].goal, z[1]) and np.array_equal(sb[4].goal, z[1]), "best goal")
    sg = G({"seat_tol": 1.0, "hold": {"enter_k": 0.1, "stall": {"latents": 3, "eps": 0.1,
                                                                 "k": 2.0}}})
    dd = [3.0, 1.9, 1.85, 1.75, 1.72, 1.7, 1.75, 1.8, 1.9]
    ms = [sg.update(z[k], d, k >= 3).mode for k, d in enumerate(dd)]
    chk(ms == ["normal"] * 6 + ["hold"] * 3, f"stall {ms}")  # anchors 3.0, 1.9, 1.75
    sn = G({"seat_tol": 1.0, "hold": {"enter_k": 0.1, "stall": {"latents": 2, "k": 2.0}}})
    chk([sn.update(z[0], d, False).mode for d in (1.5, 1.6, 1.7, 1.8)] == ["normal"] * 4,
        "stall needs the last stage")
    s = G({"seat_tol": 1.0, "hold": {"streak": 1}}).update(z[0], 0.1, False)
    chk(s.goal is None and not s.cold, "seat goal (None = the stage goal), cold off")
    p = tree.MppiParams()
    hp = h.params_for(p, "hold")
    chk((hp.w_q, hp.w_T, hp.w_r, hp.w_s) == (p.w_q, 2 * p.w_T, 10 * p.w_r, p.w_s), "hold w")
    n = G({"seat_tol": 1.0, "near": {"k": 2.0, "w_T": 3.0}})
    chk(n.update(z[0], 1.9, False).mode == "near" and n.update(z[0], 2.1, False).mode
        == "normal" and n.params_for(p, "near").w_T == 3 * p.w_T, "near")
    chk(h.params_for(p, "normal") is p and len(h.variants(p)) == 1
        and len(n.variants(p)) == 1, "variants")
    return ok, ("fade ramp, last stage, hold streak 3 / latch / entry goal / cold once, exit_k + "
                "best goal, stall (anchor, last stage only), weights, near"
                + (f"; FAIL {notes}" if notes else ""))


def o5(tree, tree_node) -> tuple[bool, str]:
    _, _, ep_file = run_args()
    ut, z, _, _, _ = recorded(ep_file, N_LAT)
    at = {int(u): k for k, u in enumerate(ut)}
    calls = []

    def spy(real):
        def f(lcs, params, z0, p0, rot0, goal, u_prev, gen, **kw):
            calls.append({"params": params, "goal": np.array(goal), "u_prev": u_prev,
                          "u_demo": None if kw.get("u_demo") is None else kw["u_demo"].copy()})
            return real(lcs, params, z0, p0, rot0, goal, u_prev, gen, **kw)
        return f

    probe = stream(tree_node, base_doc(), True, N_LAT)["node"]
    d_all = np.array([probe.lcs.whitened_dist(x, probe.stages[-1].goal) for x in z])
    tol = float(np.quantile(d_all, 0.3)) / 0.9  # hold enters inside the replay window
    ov = {"seat_tol": tol, "fade": [1.0, 2.0], "fade_last_stage": True,
          "hold": {"enter_k": 0.9, "streak": 2, "goal": "seat", "w_r": 10.0, "cold": True}}
    calls.clear()
    r = stream(tree_node, {**base_doc(), "overshoot": ov}, True, N_LAT, spy=spy)
    rows, node = r["rows"], r["node"]
    ok, notes = True, []
    keys_ok = all({"d_seat", "prior_scale", "mode"} <= set(x) for x in rows)
    calls = calls[-len(rows):]  # drop the warm-up solves
    g = tree.OvershootGuard(ov)
    seat = node.stages[-1].goal
    base_w = node.params
    n_hold = 0
    for row, c in zip(rows, calls):
        k = at[row["utime"]]  # stage switches consume a latent without a solve
        d = node.lcs.whitened_dist(z[k], seat)
        s = g.update(z[k], d, row["stage"] == len(node.stages) - 1)
        if abs(row["d_seat"] - d) > 1e-12 or row["mode"] != s.mode:
            ok = False
            notes.append(f"k {k} mode {row['mode']} vs {s.mode}")
        if s.mode == "hold":
            n_hold += 1
            if c["u_demo"] is not None or not np.array_equal(c["goal"], seat) \
                    or c["params"].w_r != 10.0 * base_w.w_r or (s.cold != (c["u_prev"] is None)):
                ok = False
                notes.append(f"k {k} hold call")
        else:
            ud = node.wp.u_seq(row["idx"], base_w.H) * s.prior_scale
            if c["u_demo"] is None or not np.allclose(c["u_demo"], ud, atol=0, rtol=0):
                ok = False
                notes.append(f"k {k} prior x {s.prior_scale:.3f}")
    holds = [x for x in r["stdout"].splitlines() if "overshoot hold" in x]
    ok = ok and keys_ok and 0 < n_hold < len(rows) and len(holds) == 1
    return ok, (f"{len(rows)} solves, {n_hold} in hold (entered once: '{holds[:1]}'), prior x "
                f"fade per solve, hold: no prior / seat goal / w_r x10 / first solve cold; JSONL "
                f"keys {keys_ok}" + (f"; FAIL {notes[:4]}" if notes else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, required=True)
    ap.add_argument("--tree-dir", type=Path, default=None,
                    help="test these copies instead of the tree's (pre-install)")
    ap.add_argument("--out", type=Path, default=None, help="json report")
    a = ap.parse_args()
    pre_lcs, pre_node = load_pair(a.pre_dir, "pre")
    tree_lcs, tree_node = load_pair(a.tree_dir, "tree")
    res = {}
    for name, fn in (("O1", lambda: o1(pre_lcs, tree_lcs)), ("O2", lambda: o2(pre_lcs, tree_lcs)),
                     ("O3", lambda: o3(pre_node, tree_node)), ("O4", lambda: o4(tree_lcs)),
                     ("O5", lambda: o5(tree_lcs, tree_node))):
        ok, msg = fn()
        res[name] = {"ok": bool(ok), "msg": msg}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    n = sum(r["ok"] for r in res.values())
    print(f"{n}/{len(res)} PASS")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
