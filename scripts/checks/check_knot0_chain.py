#!/usr/bin/env python3
"""Checks of the node's opt-in Franka ``knot0_chain`` (V12-127): knot 0 = the previous plan's command at
t (capped from the measured pose) instead of the measured pose.

K1 params: default None; bad values refused; knot0_chain does not key a separate solver (CPU).
K2 off identical (``--pre-dir`` = pre-change tree): MppiParams.from_yaml(...).to_dict() of stage-1 /
   stage-2 configs equal pre's plus ``knot0_chain: None``; the tree's node / mppi_lcs / mppi_guards
   sources with the V12-127 hunks removed are byte-identical to pre's; franka_message bytes equal.
K3 chain_pose: no plan / t outside the plan -> the measured pose; linear / nlerp between knots;
   cap_mm and cap_deg pull back toward the measured pose; "shift" moves every knot by the lead,
   "knot0" only knot 0; the chained plan is kept for the next solve.

    uv run --frozen python scripts/checks/check_knot0_chain.py [--pre-dir <pre-change tree>]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs", REPO / "scripts" / "checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mppi_controller_node as node
import torch
from check_grip_encoder import strip_v131

from task_common import mppi_guards as guards
from task_common import mppi_lcs as tree

ES = REPO / "data/lcs/realfirst/cmp/encoder_study"
CFGS = [ES / "sim87_V45c_ur30/configs/V45c_ur30_H7.yaml", ES / "sim104_V50b/configs/V50b_ffg29_ws1_du.yaml",
        REPO / "data/lcs/realfirst/cmp/v12_125/configs/V57c_stage2.yaml",
        REPO / "data/lcs/realfirst/cmp/v12_126/configs/V57d_stage2.yaml"]
V50B_DEP = ES / "sim/V50b/deploy_v18b/deploy.npz"
ON = {"mode": "shift", "cap_mm": 15.0, "cap_deg": 5.0}
# V12-127 hunks (tree text, pre text) per file
HUNKS = {
    "scripts/lcs/mppi_controller_node.py": [
        ("OvershootGuard, PolicyPrior, chain_pose, guard_plan", "OvershootGuard, PolicyPrior, guard_plan"),
        ("        self.prev_knots = None  # knot0_chain: (t, pos, quat) of the last published Franka plan\n", ""),
        ("""        fmsg = (franka_message(utime, t, self.dt, ee_f, U_exec) if params.knot0_chain is None
                else self._chained_message(utime, t, ee_f, U_exec, params.knot0_chain, extra))
        self.out = [(CH.tracking_trajectory_actor_channel, fmsg.encode()),""",
         """        self.out = [(CH.tracking_trajectory_actor_channel,
                     franka_message(utime, t, self.dt, ee_f, U_exec).encode()),"""),
        ("""    def _chained_message(self, utime: int, t: float, ee_f, U, cfg: dict,
                         extra: dict) -> lcmt_timestamped_saved_traj:
        \"\"\"``knot0_chain``: knot 0 = the previous plan's command at ``t`` (``chain_pose``).\"\"\"
        pose0, info = chain_pose(self.prev_knots, t, self.dt, ee_f, cfg)
        pos, quat = franka_knots(pose0 if cfg["mode"] == "shift" else ee_f, U, N_KNOTS)
        if cfg["mode"] == "knot0":
            pos[0], quat[0] = pose0[:3], pose0[3:7]
        self.prev_knots = (t, pos, quat)
        extra["knot0_chain"] = info
        return _rename(saved_traj_message(utime, pos, quat, t + self.dt * np.arange(N_KNOTS)))

""", "")],
    "src/task_common/mppi_lcs.py": [
        ("""    # Node: Franka knot 0 = the previous plan's command at t (capped {cap_mm, cap_deg} from the
    # measured pose) instead of the measured pose; "shift" moves every knot with it (cmd_delta
    # actions), "knot0" only knot 0 (knot1_minus_measured). None = off.
    knot0_chain: dict | None = None
""", ""),
        ("""        if self.knot0_chain is not None:
            check_knot0_chain(self.knot0_chain)
""", ""),
        ("check_exec_guard, check_knot0_chain, check_overshoot", "check_exec_guard, check_overshoot"),
        ('"goal_blend_s", "knot0_chain"}', '"goal_blend_s"}')],
}


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def k1() -> str:
    _require(tree.MppiParams().knot0_chain is None, "default")
    refused = 0
    for bad in ({"mode": "shift", "cap_mm": 1.0}, {**ON, "mode": "x"}, {**ON, "cap_mm": -1.0},
                {**ON, "cap_deg": math.nan}, {**ON, "extra": 1}, "shift"):
        try:
            tree.MppiParams(knot0_chain=bad)
        except ValueError:
            refused += 1
    _require(refused == 6, f"refused {refused}/6")
    tree.MppiParams(knot0_chain=ON)
    tree.MppiParams(knot0_chain={**ON, "mode": "knot0"})
    lcs = tree.TorchLcs.from_deploy(V50B_DEP, torch.device("cpu"))
    prm = tree.MppiParams.from_yaml(CFGS[1])
    same = tree.solver_for(lcs, prm) is tree.solver_for(lcs, dataclasses.replace(prm, knot0_chain=ON))
    _require(same, "knot0_chain keys a separate solver")
    return "default None; 6/6 bad refused; shares the cached solver"


def _strip_guards(txt: str) -> str:
    a = txt.index("KNOT0_CHAIN_KEYS = ")
    txt = txt[:a] + txt[txt.index("\n", txt.index("KNOT0_CHAIN_MODES = ")) + 1:]
    a, b = txt.index("def check_knot0_chain("), txt.index("def check_exec_guard(")
    return txt[:a] + txt[b:]


def k2(pre_dir: Path | None) -> str:
    if pre_dir is None:
        return "no --pre-dir: skipped"
    n = 0
    for rel, hunks in HUNKS.items():
        txt = strip_v131(rel, (REPO / rel).read_text())  # later opt-in V12-131 hunks
        for new, old in hunks:
            _require(txt.count(new) == 1, f"{rel}: hunk not found once: {new[:60]!r}")
            txt = txt.replace(new, old)
        _require(txt == (pre_dir / rel).read_text(), f"{rel}: tree minus V12-127 hunks != pre")
        n += 1
    rel = "src/task_common/mppi_guards.py"
    _require(_strip_guards((REPO / rel).read_text()) == (pre_dir / rel).read_text(), f"{rel} != pre")
    n += 1
    sys.path.insert(0, str(pre_dir / "src"))
    pre = _load("pre_mppi_lcs", pre_dir / "src/task_common/mppi_lcs.py")
    sys.path.pop(0)
    for cfg in CFGS:
        a, b = pre.MppiParams.from_yaml(cfg).to_dict(), tree.MppiParams.from_yaml(cfg).to_dict()
        _require(b.pop("knot0_chain", "missing") is None and a == b, f"{cfg.name}: params differ")
    rng = np.random.default_rng(0)
    for _ in range(5):
        pose = np.r_[rng.normal(size=3), rng.normal(size=4)]
        pose[3:] /= np.linalg.norm(pose[3:])
        U = rng.normal(size=(7, 12)) * 1e-3
        _require(node.franka_message(1, 0.5, 0.075, pose, U).encode()
                 == tree_message_pre(pre_dir, pose, U), "franka_message bytes differ")
    return f"{n} sources == pre minus the hunks; {len(CFGS)} configs' params == pre + knot0_chain None; franka_message bytes =="


def tree_message_pre(pre_dir: Path, pose, U) -> bytes:
    sys.path.insert(0, str(pre_dir / "scripts/lcs"))
    try:
        pn = _load("pre_node", pre_dir / "scripts/lcs/mppi_controller_node.py")
    finally:
        sys.path.pop(0)
    return pn.franka_message(1, 0.5, 0.075, pose, U).encode()


def k3() -> str:
    dt = 0.075
    ee = np.r_[0.4, 0.1, 0.05, 1.0, 0.0, 0.0, 0.0]
    big = {"mode": "shift", "cap_mm": 100.0, "cap_deg": 90.0}
    p, info = guards.chain_pose(None, 1.0, dt, ee, big)
    _require(np.array_equal(p, ee) and info["lead_mm"] == 0.0, "no plan")
    pos = np.stack([ee[:3] + [0.002 * i, 0.0, 0.0] for i in range(7)])
    rz = [math.radians(2.0 * i) for i in range(7)]
    quat = np.stack([[math.cos(r / 2), 0.0, 0.0, math.sin(r / 2)] for r in rz])
    prev = (1.0, pos, quat)
    for t in (0.99, 1.0 + 6.01 * dt):
        p, _ = guards.chain_pose(prev, t, dt, ee, big)
        _require(np.array_equal(p, ee), f"t {t} outside the plan")
    p, info = guards.chain_pose(prev, 1.0 + 2.5 * dt, dt, ee, big)
    _require(np.allclose(p[:3], ee[:3] + [0.005, 0, 0]) and abs(info["lead_deg"] - 5.0) < 1e-6,
             f"interp {p} {info}")
    p, info = guards.chain_pose(prev, 1.0 + 2.5 * dt, dt, ee, {**big, "cap_mm": 3.0, "cap_deg": 2.0})
    ang = math.degrees(2 * math.acos(abs(p[3])))
    _require(np.allclose(p[:3], ee[:3] + [0.003, 0, 0]) and abs(ang - 2.0) < 0.05 and info["capped"],
             f"cap {p} {ang}")
    U = np.zeros((7, 12))
    U[:, 0] = 0.001
    for mode in ("shift", "knot0"):
        fake = SimpleNamespace(prev_knots=prev, dt=dt)
        extra = {}
        node.MppiControllerNode._chained_message(fake, 1, 1.0 + 2.5 * dt, ee, U, {**big, "mode": mode}, extra)
        P, _ = node.franka_knots(ee, U, 7)
        lead = np.r_[0.005, 0.0, 0.0]
        kp = fake.prev_knots[1]
        want = P + lead if mode == "shift" else np.vstack([ee[:3] + lead, P[1:]])
        _require(np.allclose(kp, want) and fake.prev_knots[0] == 1.0 + 2.5 * dt and "knot0_chain" in extra,
                 f"{mode}: knots {kp} vs {want}")
    return "no plan / outside -> measured; interpolation; caps; shift vs knot0 knots; plan kept"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, default=None)
    a = ap.parse_args()
    ok = True
    for name, fn in (("K1", k1), ("K2", lambda: k2(a.pre_dir)), ("K3", k3)):
        try:
            print(f"[PASS] {name} {fn()}")
        except AssertionError as e:
            ok = False
            print(f"[FAIL] {name} {e}")
    print("ALL KNOT0_CHAIN CHECKS PASSED" if ok else "KNOT0_CHAIN CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
