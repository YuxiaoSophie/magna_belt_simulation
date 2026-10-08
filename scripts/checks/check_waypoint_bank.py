#!/usr/bin/env python3
"""Checks of the grasp-adaptive waypoint bank (``latent_waypoints.py --bank-out``,
``WaypointBank``, node ``--waypoint-bank``, harness ``--mppi-waypoint-bank``); no sim.

B1 ``derive`` with a shared memo == without; B2 bank entries == single artifacts (+ ``move0``),
shas; B3 ``map_frame`` identity and the prefix frac mapping (<= 1 frame); B4 ``select`` on each
entry's own initial latent (distance 0), k-NN mean; B5 node with the options off == the
``--pre-node`` copy (deploy goal and waypoints + prior streams: bytes, stdout, log rows); B6 a
one-entry bank of the base artifact == no bank (bytes and rows minus ``bank``); B7 node with the
bank on a demo's latents picks that demo (stages, tols, prior, idx clock, log row); B8 harness:
defaults add nothing, the bank needs waypoints, node argv.

    uv run --frozen python scripts/checks/check_waypoint_bank.py --deploy <deploy.npz> \\
        --bank <dir/bank.json> --base <latent_waypoints.npz> [--pre-node <old node .py>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import demo_prefix as dp
import eval_learned_mpc as harness
import latent_waypoints as lw
import mppi_offline as mo

V11_MPPI = REPO_ROOT / "data" / "lcs" / "realfirst" / "v1_1" / "mppi.yaml"
PRIOR = {"prior_alpha": 0.5, "prior_weight": 0.1}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


def b1_memo(a) -> tuple[bool, str]:
    bank = json.loads(a.bank.read_text())
    demos = [Path(x) for x in bank["demos"]]
    g = Path(bank["entries"][-1]["demo_episode"])
    args = (a.deploy, demos, g, bank["stride"], bank["window"], bank["percentile"],
            bank["floor"])
    plain = lw.derive(*args)
    memo: dict = {}
    lw.derive(a.deploy, demos, Path(bank["entries"][0]["demo_episode"]), *args[3:], memo)
    shared = lw.derive(*args, memo)
    same = plain.keys() == shared.keys() and all(
        np.array_equal(plain[k], shared[k]) for k in plain)
    return same, f"derive(memo) == derive() on all {len(plain)} keys: {same}"


def b2_entries(a) -> tuple[bool, str]:
    bank = json.loads(a.bank.read_text())
    demos = [Path(x) for x in bank["demos"]]
    bad, n = [], 0
    for j in sorted({0, len(bank["entries"]) // 2, len(bank["entries"]) - 1}):
        e = bank["entries"][j]
        f = a.bank.parent / e["file"]
        doc = lw.derive(a.deploy, demos, Path(e["demo_episode"]), bank["stride"],
                        bank["window"], bank["percentile"], bank["floor"])
        with np.load(f, allow_pickle=False) as d:
            got = {k: d[k] for k in d.files}
        n += 1
        if set(got) - set(doc) != {"move0"} or not all(np.array_equal(doc[k], got[k])
                                                       for k in doc):
            bad.append(j)
        if lw.sha256(f) != e["sha256"] or int(got["move0"]) != e["move0"]:
            bad.append(j)
    shas = all(lw.sha256(a.bank.parent / e["file"]) == e["sha256"] for e in bank["entries"])
    return not bad and shas, (f"{n} entries re-derived == bank npz (+ move0 only): "
                              f"{not bad}; all {len(bank['entries'])} shas match: {shas}")


def b3_map(a) -> tuple[bool, str]:
    wb = lw.WaypointBank(a.bank)
    ident = all(lw.map_frame(t, m, m) == t for m in wb.marks for t in range(m[3]))
    worst = 0
    base = wb.marks[0]
    for frac in (0.25, 0.5, 1.0, 1.5):
        mb = dict(zip(("move0", "latch", "first_hold"), base[:3]))
        off = dp.frac_frame(mb, frac)
        for m in wb.marks:
            want = dp.frac_frame(dict(zip(("move0", "latch", "first_hold"), m[:3])), frac)
            worst = max(worst, abs(lw.map_frame(off, base, m) - want))
    offs, keys = wb.key_frames(20, base)
    k20 = all(k == 0 for k in keys) and offs == [m[0] for m in wb.marks]
    ok = ident and worst <= 1 and k20
    return ok, (f"identity map {ident}; prefix frac 0.25/0.5/1/1.5 mapped vs per-demo frac frame:"
                f" max |diff| {worst} frame(s); offset 20 (= move0) -> key frame 0, demo clock "
                f"= each move0: {k20}")


def b4_select(a) -> tuple[bool, str]:
    wb = lw.WaypointBank(a.bank)
    zs = wb.entries[0].z_traj.std(0) + 1.0  # any positive scale
    hits = 0
    for j, w in enumerate(wb.entries):
        sel, off, info = wb.select(w.z_traj[0], zs)
        hits += info["dist"] == 0.0 and np.array_equal(sel.z_traj[0], w.z_traj[0]) and off == 0
    sel, off, info = wb.select(wb.entries[3].z_traj[0], zs, knn=3)
    ref = wb.entries[info["entry"]]
    nn = [k["entry"] for k in info["knn"]]
    z_mean = np.mean([wb.entries[j].z_traj[lw.map_frame(0, wb.marks[nn[0]], wb.marks[j])]
                      for j in nn], axis=0)
    knn_ok = (sel.T == ref.T and np.array_equal(sel.frames, ref.frames)
              and np.allclose(sel.z_traj[0], z_mean) and np.all(sel.tol[:-1] >= ref.tol[:-1])
              and sel.tol[-1] == -1.0 and len(nn) == 3)
    ok = hits == len(wb.entries) and knn_ok
    return ok, (f"own initial latent -> distance 0 on {hits}/{len(wb.entries)} entries; knn 3: "
                f"entries {nn}, ref clock, mean z, tol >= ref: {knn_ok}")


def _cli(a) -> SimpleNamespace:
    return SimpleNamespace(deploy=a.deploy, config=V11_MPPI, device=a.device, override=[])


def _same(s1: dict, s2: dict, drop_bank: bool = False) -> bool:
    r2 = [{k: v for k, v in r.items() if not (drop_bank and k == "bank")} for r in s2["rows"]]
    return s1["raw"] == s2["raw"] and s1["rows"] == r2


def _text(s: dict) -> list[str]:
    """stdout minus Warp's once-per-process kernel-load lines."""
    return [ln for ln in s["stdout"].splitlines() if not ln.startswith("Module ")]


def b5_off(a) -> tuple[bool, str]:
    if a.pre_node is None:
        return True, "SKIP (no --pre-node)"
    pre = _load_module(a.pre_node, "pre_mppi_node")
    T = mo.TorchLcs.from_deploy(a.deploy, "cpu")
    base = lw.Waypoints.load(a.base)

    def far(k, node):
        return T.stage_goals[-1] + 0.2

    def on_demo(k, node):
        return base.z_traj[min(20 + k, base.T - 1)]

    wpf = {"goal": f"waypoints:{a.base}", "goal_mode": "stage", "demo_actions": "auto",
           "demo_offset_frames": 20}
    out = []
    for name, over, z_of, flags in (("deploy goal", {}, far, None),
                                    ("waypoints+prior", PRIOR, on_demo, wpf)):
        s_pre = mo._stream(_cli(a), over, z_of, 40, node_module=pre, node_flags=flags)
        s_new = mo._stream(_cli(a), over, z_of, 40, node_flags=flags)
        out.append((name, _same(s_pre, s_new) and _text(s_pre) == _text(s_new),
                    len(s_new["rows"])))
    ok = all(x[1] for x in out)
    return ok, "; ".join(f"{n}: bytes/stdout/rows identical {s} ({k} solves)" for n, s, k in out)


def b6_one_entry(a) -> tuple[bool, str]:
    base = lw.Waypoints.load(a.base)
    with tempfile.TemporaryDirectory() as tmp:
        with np.load(a.base, allow_pickle=False) as d:
            doc = {k: d[k] for k in d.files}
        marks = lw.episode_marks(base.demo_episode)
        doc["move0"] = np.int64(marks[0])
        lw.save_npz(Path(tmp) / "wp_00.npz", doc)
        bj = Path(tmp) / "bank.json"
        bj.write_text(json.dumps({"version": lw.BANK_VERSION, "deploy_sha256": "-", "entries": [
            {"file": "wp_00.npz", "demo_episode": base.demo_episode, "tag": "base",
             "frames": base.frames.tolist(), "move0": marks[0]}]}))
        wpf = {"goal": f"waypoints:{a.base}", "goal_mode": "stage", "demo_actions": "auto",
               "demo_offset_frames": 20}

        def on_demo(k, node):
            return base.z_traj[min(20 + k, base.T - 1)]

        s0 = mo._stream(_cli(a), PRIOR, on_demo, 40, node_flags=wpf)
        s1 = mo._stream(_cli(a), PRIOR, on_demo, 40, node_flags={**wpf, "waypoint_bank": bj})
    first = s1["rows"][0].get("bank", {})
    extra = [ln for ln in s1["stdout"].splitlines() if ln not in s0["stdout"].splitlines()]
    ok = (_same(s0, s1, drop_bank=True) and first.get("entry") == 0
          and first.get("offset") == 20 and sum("bank" in r for r in s1["rows"]) == 1
          and len(extra) == 1 and extra[0].startswith("[mppi] bank: entry 0"))
    return ok, (f"one-entry bank of the base == no bank (bytes, rows minus bank): "
                f"{_same(s0, s1, drop_bank=True)}; bank row once, entry {first.get('entry')} "
                f"offset {first.get('offset')} dist {first.get('dist', float('nan')):.2e}; "
                f"stdout +1 line")


def b7_pick(a) -> tuple[bool, str]:
    wb = lw.WaypointBank(a.bank)
    T = mo.TorchLcs.from_deploy(a.deploy, "cpu")
    base = lw.Waypoints.load(a.base)
    d0 = [T.whitened_dist(w.z_traj[0], base.z_traj[0]) for w in wb.entries]
    j = int(np.argmax(d0))  # the entry farthest from the base at the start
    w = wb.entries[j]
    wpf = {"goal": f"waypoints:{a.base}", "goal_mode": "stage", "demo_actions": "auto",
           "demo_offset_frames": 20, "waypoint_bank": a.bank}

    def on_j(k, node):
        return w.z_traj[0] if k == 0 else w.z_traj[min(w.move0 + k, w.T - 1)]

    s = mo._stream(_cli(a), PRIOR, on_j, 30, node_flags=wpf)
    node = s["node"]
    first = s["rows"][0]["bank"]
    twin = [i for i, x in enumerate(wb.entries) if np.array_equal(x.z_traj[0], w.z_traj[0])]
    idx = [r["idx"] for r in s["rows"]]
    t0 = s["rows"][0]["t"]
    clock = [w.move0 + round((r["t"] - t0) / w.dt) for r in s["rows"]]
    ok = (first["entry"] in twin and first["dist"] == 0.0
          and [st.tol for st in node.stages[:-1]] == wb.entries[first["entry"]].tol[:-1].tolist()
          and len(node.stages) == len(wb.entries[first["entry"]].frames)
          and np.array_equal(node.u_demo, wb.entries[first["entry"]].u_traj)
          and idx[0] == w.move0 and idx == clock
          and first["n_stages"] == len(node.stages))
    return ok, (f"latents of entry {j} ({first['tag']}, initial distance to the base "
                f"{d0[j]:.2f}): picked {first['entry']} at dist {first['dist']:.1e}, "
                f"{len(node.stages)} stages with its tols, prior = its u_traj, idx "
                f"{idx[0]}..{idx[-1]} = move0 + latent time")


class _Ctrl:
    def start(self, *args, **kw):
        self.args, self.kw = args, kw


def b8_harness(a) -> tuple[bool, str]:
    p = harness.create_parser()
    base_argv = ["--mode", "learned", "--deploy", str(a.deploy), "--controller", "mppi",
                 "--mppi-config", str(V11_MPPI), "--mppi-goal", "waypoints",
                 "--mppi-waypoints", str(a.base), "--mppi-demo-actions", "auto"]
    off = harness._mppi_options(p.parse_args(base_argv), False)
    on = harness._mppi_options(p.parse_args(base_argv + ["--mppi-waypoint-bank", str(a.bank),
                                                         "--mppi-bank-knn", "2"]), False)
    try:
        harness._mppi_options(p.parse_args(["--mode", "learned", "--deploy", str(a.deploy),
                                            "--controller", "mppi", "--mppi-config",
                                            str(V11_MPPI), "--mppi-waypoint-bank",
                                            str(a.bank)]), False)
        needs = False
    except ValueError:
        needs = True
    ctx = SimpleNamespace(mppi=on, args=SimpleNamespace(mppi_demo_offset_frames=20,
                                                        lcm_url="x", deploy=a.deploy))
    c = _Ctrl()
    harness._start_mppi(c, ctx, Path("/tmp/x.log"))
    argv = list(c.kw["extra_args"])
    i = argv.index("--waypoint-bank")
    ctx.mppi = off
    c0 = _Ctrl()
    harness._start_mppi(c0, ctx, Path("/tmp/x.log"))
    ok = ("bank" not in off and on["bank"]["knn"] == 2 and needs
          and argv[i + 1] == str(a.bank.resolve()) and argv[i + 2:i + 4] == ["--bank-knn", "2"]
          and "--waypoint-bank" not in c0.kw["extra_args"])
    return ok, (f"defaults: no bank block / no node flag {'bank' not in off}; bank without "
                f"--mppi-goal waypoints refused {needs}; node argv {argv[i:i + 4]}")


CHECKS = (("B1", b1_memo), ("B2", b2_entries), ("B3", b3_map), ("B4", b4_select),
          ("B5", b5_off), ("B6", b6_one_entry), ("B7", b7_pick), ("B8", b8_harness))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--deploy", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True, help="bank.json")
    p.add_argument("--base", type=Path, required=True, help="the base latent_waypoints.npz")
    p.add_argument("--pre-node", type=Path, default=None, help="node copy before the bank")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    n_fail = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(a)
        except Exception:  # noqa: BLE001
            ok, detail = False, traceback.format_exc()
        n_fail += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    print("ALL WAYPOINT BANK CHECKS PASSED" if not n_fail else f"{n_fail} CHECK(S) FAILED")
    return int(bool(n_fail))


if __name__ == "__main__":
    sys.exit(main())
