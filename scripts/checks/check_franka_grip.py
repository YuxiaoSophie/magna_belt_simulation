#!/usr/bin/env python3
"""Checks of the opt-in Franka grip force (V12-132): ``--franka-grip-n N`` = finger drive ke 25000
kd 160, effort limit N / 2 per finger, so the squeeze is force-limited (libfranka grasp()).

F1 (CPU) defaults: the round-belt sim and eval parsers default to None; the eval drops it from
   index.args at default; ``franka_grip_drive`` values / refusals; with ``--pre-dir`` (HEAD copy of
   ``src/round_belt_task/lcm_simulation.py``) the tree minus the V12-132 hunks is byte-identical.
F2 (``--sim``, GPU, under sim.lock): the nominal start (both hands on the belt), arms held at
   their positions for ``--hold-s``, default drive vs each ``--grip`` value: per-finger drive force
   (the drive law the solver applies), finger gap, pad sink into the belt ((2 r - gap) / 2),
   max belt link stretch (all links / at the grip), belt speed, all finite. 30 N must sit on its
   limit (|force| within 2 % of 15 N) with a smaller gap than the default.

    ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_wr_pin640b1.yaml uv run --frozen \\
        python scripts/checks/check_franka_grip.py --pre-dir data/lcs/realfirst/cmp/v12_132/pre \\
        [--sim --out data/lcs/realfirst/cmp/v12_132/static.json]
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

NOMINAL = REPO / "data/lcs/start_states/pre_place_1_osc.npz"
REST_LINK_MM = 640.0 / 48
REL = "src/round_belt_task/lcm_simulation.py"
HUNKS = [  # V12-132 (tree text, pre text)
    ("import argparse\nimport dataclasses\n", "import argparse\n"),
    ('''# V12-132 --franka-grip-n: a stiff finger drive whose effort limit binds on the ~7 mm belt, so the
# squeeze is the force limit (libfranka grasp()); the bike chain's ke / kd (35 N would not bind at
# 1000 N/m: ~3.4 mm x 1000 = 3.4 N per finger).
FRANKA_GRIP_KE, FRANKA_GRIP_KD = 25000.0, 160.0


def franka_grip_drive(drive: HandDrive, squeeze_n: float) -> HandDrive:
    """``drive`` with ke / kd FRANKA_GRIP_* and an effort limit of half the total squeeze."""
    if not squeeze_n > 0.0:
        raise ValueError(f"--franka-grip-n must be > 0, got {squeeze_n}")
    return dataclasses.replace(drive, ke=FRANKA_GRIP_KE, kd=FRANKA_GRIP_KD,
                               effort_limit=0.5 * float(squeeze_n))


''', ""),
    ('''        grip = getattr(args, "franka_grip_n", None)
        if grip is not None:
            self.hand_drive = franka_grip_drive(self.hand_drive, grip)
''', ""),
    ('''    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        parser = super().create_parser()
        parser.add_argument(
            "--franka-grip-n", type=float, default=None, metavar="N",
            help="Franka squeeze force [N, both fingers]: finger drive ke "
                 f"{FRANKA_GRIP_KE:g} kd {FRANKA_GRIP_KD:g}, effort limit N / 2 per finger "
                 "(default: the yaml hand_drive)")
        return parser

''', ""),
    ('''            **({"franka_grip_n": float(self.args.franka_grip_n),
                "hand_drive": dataclasses.asdict(self.hand_drive)}
               if getattr(self.args, "franka_grip_n", None) is not None else {}),
''', ""),
]


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def f1(pre_dir: Path | None) -> str:
    import eval_learned_mpc as harness

    from round_belt_task import lcm_simulation as ls

    _require(ls.RoundBeltLcmSimulation.create_parser().parse_args([]).franka_grip_n is None,
             "sim parser default")
    _require(harness.create_parser().parse_args(["--mode", "learned"]).franka_grip_n is None
             and harness.GRIP_ARG_DEFAULTS == {"franka_grip_n": None}, "eval default")
    d = ls.franka_grip_drive(ls.RoundBeltLcmSimulation.hand_drive, 30.0)
    _require((d.ke, d.kd, d.effort_limit) == (25000.0, 160.0, 15.0), f"drive {d}")
    base = ls.RoundBeltLcmSimulation.hand_drive
    _require((base.ke, base.kd, base.effort_limit) == (1000.0, 100.0, 20.0), f"yaml drive {base}")
    refused = 0
    for bad in (0.0, -5.0, float("nan")):
        try:
            ls.franka_grip_drive(base, bad)
        except ValueError:
            refused += 1
    _require(refused == 3, f"refused {refused}/3")
    msg = "drive 25000 / 160 / 15 N at 30 N; 3/3 bad refused; parsers default None"
    if pre_dir is None:
        return msg + "; no --pre-dir: source identity skipped"
    txt = (REPO / REL).read_text()
    for new, old in HUNKS:
        _require(txt.count(new) == 1, f"hunk not found once: {new[:60]!r}")
        txt = txt.replace(new, old)
    _require(txt == (pre_dir / "lcm_simulation.py").read_text(), f"{REL} minus hunks != pre")
    return msg + f"; {REL} minus the {len(HUNKS)} V12-132 hunks == pre"


def hold(grip: float | None, hold_s: float) -> dict:
    from round_belt_task.constants import BELT_RADIUS
    from round_belt_task.offline_simulation import RoundBeltOfflineSimulation
    from task_common import sim_snapshot

    kw = {} if grip is None else {"franka_grip_n": grip}
    sim = RoundBeltOfflineSimulation.build(**kw)
    try:
        sim.restore(sim_snapshot.load(NOMINAL))
        q_f, q_u = sim.arm_positions()
        sim.set_arm_targets(q_f, q_u)
        hand = sim._io_by_name["franka_hand"]
        drive = sim.hand_drive
        belt_ids = np.asarray(sim.info.belt_bodies)
        rows = []
        for i in range(round(hold_s / sim.frame_dt)):
            sim.control_step()
            body_q = sim.state_0.body_q.numpy()
            belt = body_q[belt_ids, :3].astype(np.float64)
            g = sim.grasp_state(body_q)
            force = drive.ke * (sim._hand_target_q - hand.positions) - drive.kd * hand.velocities
            link = np.linalg.norm(np.roll(belt, -1, 0) - belt, axis=1) * 1e3
            tip = body_q[sim._finger_tip_body, :3]
            k = int(np.argmin(np.linalg.norm(belt - tip, axis=1)))
            near = link[[(k - 1) % len(link), k]]
            if i == 0:
                prev = belt
            speed = np.linalg.norm(belt - prev, axis=1).max() / sim.frame_dt
            prev = belt
            rows.append([sim.sim_time, *np.clip(force, -drive.effort_limit, drive.effort_limit),
                         g.franka_width_mm, link.max(), near.max(), speed,
                         float(np.isfinite(body_q).all())])
        r = np.asarray(rows)
        last = r[r[:, 0] >= r[-1, 0] - 1.0]  # the last second
        width = last[:, 3]
        return {"grip_n": grip, "ke": drive.ke, "kd": drive.kd, "effort_limit": drive.effort_limit,
                "finite": bool(r[:, 7].all()), "steps": len(r),
                "force_n_last1s": [float(np.abs(last[:, 1]).mean()), float(np.abs(last[:, 2]).mean())],
                "squeeze_n_last1s": float(np.abs(last[:, 1:3]).sum(1).mean()),
                "gap_mm_last1s": float(width.mean()), "gap_mm_range_last1s": [float(width.min()),
                                                                              float(width.max())],
                "pad_sink_mm_last1s": float((2e3 * BELT_RADIUS - width.mean()) / 2.0),
                "belt_diameter_mm": 2e3 * BELT_RADIUS,
                "max_link_stretch_pct": float(100 * (r[:, 4].max() / REST_LINK_MM - 1)),
                "grip_link_stretch_pct_last1s": float(100 * (last[:, 5].max() / REST_LINK_MM - 1)),
                "max_belt_speed_mps": float(r[1:, 6].max()),
                "series": {"t": r[:, 0].tolist(), "gap_mm": r[:, 3].tolist(),
                           "force_n": np.abs(r[:, 1:3]).mean(1).tolist()}}
    finally:
        sim.close("check_franka_grip")


def f2(grips: list[float], hold_s: float, out: Path | None) -> str:
    res = [hold(None, hold_s)] + [hold(g, hold_s) for g in grips]
    if out is not None:
        out.write_text(json.dumps(res, indent=1))
    d = res[0]
    lines = []
    for r in res:
        _require(r["finite"], f"grip {r['grip_n']}: non-finite state")
        lines.append(f"{r['grip_n'] or 'default'}: force/finger {r['force_n_last1s'][0]:.2f}/"
                     f"{r['force_n_last1s'][1]:.2f} N (limit {r['effort_limit']:g}), gap "
                     f"{r['gap_mm_last1s']:.2f} mm, pad sink {r['pad_sink_mm_last1s']:.2f} mm, "
                     f"link stretch max {r['max_link_stretch_pct']:.1f} % (grip "
                     f"{r['grip_link_stretch_pct_last1s']:.1f} %), belt speed max "
                     f"{r['max_belt_speed_mps']:.3f} m/s")
        if r["grip_n"] == 30.0:
            _require(all(abs(f - 15.0) <= 0.3 for f in r["force_n_last1s"]),
                     f"30 N not on its limit: {r['force_n_last1s']}")
            _require(r["gap_mm_last1s"] < d["gap_mm_last1s"], "30 N gap not below default")
    return "; ".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pre-dir", type=Path, default=None)
    ap.add_argument("--sim", action="store_true", help="F2 static hold (GPU)")
    ap.add_argument("--grip", type=float, nargs="+", default=[30.0])
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=None, help="F2 results json")
    a = ap.parse_args()
    ok = True
    checks = [("F1", lambda: f1(a.pre_dir))]
    if a.sim:
        checks.append(("F2", lambda: f2(a.grip, a.hold_s, a.out)))
    for name, fn in checks:
        try:
            print(f"[PASS] {name} {fn()}", flush=True)
        except AssertionError as e:
            ok = False
            print(f"[FAIL] {name} {e}", flush=True)
    print("ALL FRANKA-GRIP CHECKS PASSED" if ok else "FRANKA-GRIP CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
