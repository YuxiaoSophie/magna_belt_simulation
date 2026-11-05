#!/usr/bin/env python3
"""Stage-2 start set (V12-91): stage-1 end states -> UR release + clean exit -> snapshot.

For every engaged end state of the stage-1 suites (``phase2_probe.seated_episodes``): restore it in
the stage-2 scene, open the UR, lift it straight up and then to place_9 while the Franka holds still
(the direct place_9 retreat slides the open jaws along the belt and lifts it off the large pulley),
settle, and keep the state when the Franka still holds the belt and the large pulley stays engaged.
The stage-2 MPC starts from these snapshots (the UR parked at place_9, the Franka at its release pose).

Run (stage-2 scene, one sim on the GPU):
    ROUND_BELT_SCENE=assets/round_belt_task/round_belt_scene_stage2.yaml \\
      flock data/lcs/realfirst/cmp/v12_79/sim.lock uv run --frozen python scripts/lcs/stage2_starts.py \\
      --suite data/lcs/realfirst/cmp/v12_90/suite --out data/lcs/start_states/stage2 \\
      --lcm-url 'udpm://239.255.76.191:7793?ttl=0'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase2_probe as pp

STAGE2_SCENE = "round_belt_scene_stage2.yaml"


def ur_exit(sim, ctx, byte: int, lift_mm: float, speed: float) -> tuple[str | None, list]:
    """UR straight up ``lift_mm``, then to place_9; the Franka holds its measured pose."""
    ee0 = sim.franka_measured_pose7()
    u0 = pp.pose7_mat(pp.UrTracking.fk(sim.state_0.joint_q.numpy()[sim.arm_coords()[1]]))
    ur9, samples = ctx["ur9"], []
    try:
        for target in (np.r_[u0[:3] + [0.0, 0.0, lift_mm * 1e-3], u0[3:]], ur9):
            ctx["ur9"], ctx["targets"] = target, [("hold", ee0)]
            row, s = pp.drive(sim, ctx, byte, argparse.Namespace(speed_mps=speed, hold_s=0.3))
            samples += s
            if row["abort"]:
                return f"{row['abort_kind']}: {row['abort']}", samples
    finally:
        ctx["ur9"] = ur9
    return None, samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, help="stage-1 hard-suite dir(s), comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lcm-url", required=True)
    ap.add_argument("--ur-byte", type=int, default=191, help="UR jaw byte at release")
    ap.add_argument("--settle-s", type=float, default=2.0, help="after the UR opens")
    ap.add_argument("--lift-mm", type=float, default=40.0)
    ap.add_argument("--ur-speed-mps", type=float, default=0.05)
    ap.add_argument("--final-settle-s", type=float, default=1.0)
    # The stiffer stage-2 belt opens the sim hand to ~10 mm while still held (stage 1 uses 10).
    ap.add_argument("--franka-width-max", type=float, default=20.0)
    ap.add_argument("--allow-default-scene", action="store_true")
    a = ap.parse_args()

    import collect_lcs_dataset as cli
    import eval_learned_mpc as harness
    from loguru import logger

    import round_belt_task.offline_simulation as osim
    from round_belt_task.constants import SCENE_DIRECTIVES
    from round_belt_task.osc_simulation import RoundBeltOscSimulation
    from task_common import belt_cloud as bc
    from task_common import sim_snapshot

    if Path(SCENE_DIRECTIVES).name != STAGE2_SCENE and not a.allow_default_scene:
        raise SystemExit(f"scene {SCENE_DIRECTIVES}: set ROUND_BELT_SCENE to the {STAGE2_SCENE} variant")
    osim.FRANKA_WIDTH_MM = (osim.FRANKA_WIDTH_MM[0], a.franka_width_max)
    eps = [e for s in a.suite.split(",") for e in pp.seated_episodes((REPO_ROOT / s).resolve())]
    if not eps:
        raise SystemExit(f"{a.suite}: no engaged episodes with an end_state")
    out = (REPO_ROOT / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sim = RoundBeltOscSimulation.build(lcm_url=a.lcm_url, osc_timeout_s=5.0,
                                       bridge_cls=harness.ControllerBridge)
    variants, rejected = [], []
    try:
        binary, root = harness.default_osc_binary()
        sim.start_osc(out / "osc.log", binary=binary, cwd=root)
        geo = {a.ur_byte: sim.clearance_geometry(jaw_bytes=(a.ur_byte,))}
        plate = geo[a.ur_byte][0].plate
        ctx = {"geo": geo, "large7": bc.known_large_pulley_pose7(),
               "plate_top": float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half),
               "origins": cli.sim_camera_origins(sim)}
        _, ctx["ur9"] = pp.franka_targets("scene")
        for ep in eps:
            sid = f"{ep['arm']}_{ep['start']}_{ep['repeat']}"
            w0 = time.perf_counter()
            pp.release(sim, ctx, sim_snapshot.load(REPO_ROOT / ep["end_state"]), a.ur_byte, a.settle_s)
            why, _ = ur_exit(sim, ctx, a.ur_byte, a.lift_mm, a.ur_speed_mps)
            if why is None:
                sim.commander_hook = sim.make_hold_hook()
                for _ in range(round(a.final_settle_s / sim.frame_dt)):
                    sim.control_step()
            body_q = sim.state_0.body_q.numpy()
            g = sim.grasp_state(body_q)
            m = pp.pulley_metrics(body_q, sim)
            measured = {**m, "franka_width_mm": float(g.franka_width_mm),
                        "franka_tip_gap_mm": float(g.franka_tip_gap_mm),
                        "ee_franka": sim.franka_measured_pose7().tolist(),
                        "ee_ur": pp.pose7_mat(pp.UrTracking.fk(
                            sim.state_0.joint_q.numpy()[sim.arm_coords()[1]])).tolist()}
            held = g.held_for("franka")
            row = {"id": sid, "source_end_state": ep["end_state"], "arm": ep["arm"],
                   "start": ep["start"], "repeat": ep["repeat"], "held": bool(held),
                   "measured": measured}
            # sim GT (large-pulley label) curates the set; the controller never sees it
            if why or not held or m["large"]["label"] != "engaged":
                rejected.append({**row, "reason": why or ("grasp" if not held else
                                                          f"large {m['large']['label']}")})
                logger.warning(f"[stage2] {sid} rejected: {rejected[-1]['reason']}")
                continue
            snap = sim_snapshot.capture(sim, sid, notes=f"stage-2 start from {ep['end_state']}")
            sim_snapshot.save(snap, out / f"{sid}_osc.npz")
            variants.append({**row, "file": f"{sid}_osc.npz"})
            logger.info(f"[stage2] {sid}: large {m['large']['wrap_deg']:.0f} deg, hand "
                        f"{g.franka_width_mm:.1f} mm, {time.perf_counter() - w0:.0f} s")
    finally:
        sim.close("finished")
        index = {"set": "stage2", "kind": "stage2", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "scene": str(SCENE_DIRECTIVES), "suite": a.suite, "args": vars(a),
                 "method": "stage-1 end state -> UR open, straight up, place_9 (Franka holds) -> settle",
                 "variants": variants, "rejected": rejected}
        (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"stage-2 starts: {len(variants)} kept, {len(rejected)} rejected -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
