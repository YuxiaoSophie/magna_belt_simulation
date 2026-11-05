"""GPU check of the bike-chain sim (V13-04): MuJoCo model, 3 s rest, round-belt identity, RTF.

Run (GPU, under the sim lock):
    flock data/lcs/realfirst/cmp/v12_79/sim.lock \
        uv run --frozen python scripts/checks/check_bike_chain_sim.py [--skip-roundbelt] [--neq N] \
        [--pad-condim N]
(--neq / --pad-condim / --finger-effort override the expected closure CONNECT count / pad condim; the negative
flips use 3.)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import newton
import newton.examples
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bike_chain_task import simulation as S
from bike_chain_task.outcome import SPROCKETS, classify_sprocket, rollers_from_inner_bodies
from task_common import defaults

SPIKE_MD = ROOT / "data/bike_chain/v13_02b/spike.md"
ACCEPTED_RUN = "i30_ls10_imp1_tc0.02_r2.5_s10"
SIM_SECONDS = 3.0
MAX_GAP_MAX_MM = 0.5  # incl. the closing pin
OK = True


def rep(name: str, ok: bool, msg: str = "") -> None:
    global OK
    OK &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)


def spike_rtf() -> float:
    """The accepted run's S6 RTF (full scene) from spike.md."""
    for line in SPIKE_MD.read_text().splitlines():
        if line.startswith(f"| {ACCEPTED_RUN} | S6 |"):
            return float(re.search(r"RTF (\d\.\d+)", line).group(1))
    raise RuntimeError(f"no S6 row for {ACCEPTED_RUN} in {SPIKE_MD}")


def build_sim() -> S.BikeChainTaskSimulation:
    parser = S.BikeChainTaskSimulation.create_parser()
    parser.set_defaults(viewer="null", cameras=False)
    sys.argv = sys.argv[:1]  # init() parses argv
    viewer, args = newton.examples.init(parser)
    return S.BikeChainTaskSimulation(viewer, args)


def chain_pairs(sim, n_contacts: int) -> int:
    """Active contacts whose two geoms both sit on chain bodies."""
    geom = sim.solver.mjw_data.contact.geom.numpy()[:n_contacts]
    shape = sim.solver.mjc_geom_to_newton_shape.numpy()[0][geom]
    chain = np.isin(sim.model.shape_body.numpy()[shape], sim.info.belt_bodies)
    return int((chain[:, 0] & chain[:, 1]).sum())


def g1_model(sim, want_connect: int, want_condim: int, want_finger_effort: float) -> None:
    mj, model, info = sim.solver.mj_model, sim.model, sim.info
    to_body = sim.solver.mjc_body_to_newton.numpy()[0]
    chain = list(info.belt_bodies)
    rows = [k for k in range(mj.neq) if mj.eq_type[k] == mujoco.mjtEq.mjEQ_CONNECT
            and {int(to_body[mj.eq_obj1id[k]]), int(to_body[mj.eq_obj2id[k]])}
            == {chain[0], chain[-1]}]
    rep("G1 solver", type(sim.solver).__name__ == "SolverMuJoCo",
        f"{type(sim.solver).__name__}, total neq {mj.neq}")
    rep("G1 closure", len(rows) == want_connect,
        f"{len(rows)} CONNECT rows between the last and first link (want {want_connect})")
    qd = model.joint_qd_start.numpy()
    dof = sim.solver.mjc_dof_to_newton_dof.numpy()[0]
    owner = np.searchsorted(qd, dof, side="right") - 1  # newton joint of each MuJoCo dof
    n_chain = int(np.isin(owner, info.belt_joints).sum())
    n_spr = int(np.isin(owner, info.pulley_joints).sum())
    rep("G1 nv", (n_chain, n_spr) == (69, 3), f"nv {mj.nv} = {n_chain} chain (6 + 63) + {n_spr} "
        f"sprocket + {mj.nv - n_chain - n_spr} robot")
    shape = sim.solver.mjc_geom_to_newton_shape.numpy()[0]
    on_chain = np.isin(model.shape_body.numpy()[shape], chain)
    contype = np.asarray(mj.geom_contype)[on_chain]
    rep("G1 contype", on_chain.sum() > 0 and (contype != 0).all(),
        f"{int(on_chain.sum())} chain geoms, min contype {contype.min()}")
    sim.step()
    n = int(sim.solver.mjw_data.nacon.numpy()[0])
    rep("G1 no chain-chain contact", chain_pairs(sim, n) == 0, f"{n} contacts after one step")
    damp = [float(mj.dof_damping[int(np.flatnonzero(dof == qd[j])[0])]) for j in info.pulley_joints]
    rep("G1 sprocket damping", np.allclose(damp, 0.01), f"dof_damping {damp}")
    fric = sim.solver.mjw_model.geom_friction.numpy().reshape(-1, mj.ngeom, 3)[0, :, 0]
    on_spr = np.isin(model.shape_body.numpy()[shape], info.pulley_bodies)
    on_pad = np.isin(shape, info.contact_override_shapes)
    want = {"chain": S.CHAIN_CONTACT_MU, "sprocket": S.CHAIN_CONTACT_MU,
            "pad": defaults.GRIPPER_CONTACT_MU}
    mus = {k: fric[m] for k, m in (("chain", on_chain), ("sprocket", on_spr), ("pad", on_pad))}
    rep("G1 friction", all(v.size and np.allclose(v, want[k]) for k, v in mus.items()),
        f"geom mu {({k: np.unique(v.round(4)).tolist() for k, v in mus.items()})}")
    # Pad condim / torsional / rolling reach MuJoCo as configured (opt-in, V13-05).
    tors = sim.solver.mjw_model.geom_friction.numpy().reshape(-1, mj.ngeom, 3)[0, on_pad]
    cond = np.asarray(sim.solver.mjw_model.geom_condim.numpy()).reshape(-1)[-mj.ngeom:][on_pad]
    cls = S._BikeChainTask
    rep("G1 pad contact", on_pad.sum() == 4 and (cond == want_condim).all()
        and np.allclose(tors[:, 1], cls.pad_mu_torsional)
        and np.allclose(tors[:, 2], cls.pad_mu_rolling),
        f"{int(on_pad.sum())} pad geoms, condim {np.unique(cond).tolist()} (want {want_condim}), "
        f"torsional {np.unique(tors[:, 1].round(5)).tolist()} m, rolling "
        f"{np.unique(tors[:, 2].round(6)).tolist()} m")
    # Real grasp forces (V13-05): force-limited fingers and the 2F-85 driver torque limit.
    labels, qd = list(model.joint_label), model.joint_qd_start.numpy()
    fingers = [int(qd[labels.index(n)]) for n in S.PANDA_FINGER_LABELS]
    drivers = S.find_gripper_driver_indices(model, info.gripper_joints[0],
                                            info.gripper_joints[-1] + 1)[1]
    eff, ke = model.joint_effort_limit.numpy(), model.joint_target_ke.numpy()
    jdof = np.asarray(mj.jnt_dofadr)
    to_mj = {int(sim.solver.mjc_dof_to_newton_dof.numpy()[0][j]): k for k, j in enumerate(jdof)}
    frc = {d: float(mj.jnt_actfrcrange[to_mj[d]][1]) for d in fingers + list(drivers) if d in to_mj}
    rep("G1 grip force", np.allclose(eff[fingers], want_finger_effort)
        and np.allclose(eff[list(drivers)], cls.gripper_effort_limit)
        and np.allclose(ke[cfg_idx(sim, "finger_target_indices")], cls.finger_drive_ke)
        and all(np.isclose(v, eff[d]) for d, v in frc.items()) and len(frc) == 4,
        f"finger effort {np.unique(eff[fingers]).tolist()} N (want {want_finger_effort}), ke "
        f"{np.unique(ke[cfg_idx(sim, 'finger_target_indices')]).tolist()}, 2F-85 driver effort "
        f"{np.unique(eff[list(drivers)]).tolist()} N m, MuJoCo actfrcrange {sorted(set(frc.values()))}")


def cfg_idx(sim, key: str) -> list[int]:
    return list(sim.info.joint_config[key])


def g2_rest(sim) -> float:
    for _ in range(round(SIM_SECONDS / sim.frame_dt) - 1):
        sim.step()
    m = sim.chain_rest_metrics()
    rep("G2 rest", not m["nan"] and m["gap_p95_mm"] < S.MAX_GAP_P95_MM
        and m["gap_max_mm"] < MAX_GAP_MAX_MM
        and m["min_z_above_tier_mm"] > S.MIN_Z_ABOVE_TIER_MM
        and m["arm_drift_rad"] < S.MAX_ARM_DRIFT_RAD,
        f"gap p95 {m['gap_p95_mm']:.4f} max {m['gap_max_mm']:.4f} mm (closure "
        f"{m['closure_gap_mm']:.4f}), min z above tier {m['min_z_above_tier_mm']:+.3f} mm, "
        f"arm drift {m['arm_drift_rad']:.1e} rad, nan {m['nan']}")
    body_q = sim.state_0.body_q.numpy()
    rollers = rollers_from_inner_bodies(body_q[sim.info.belt_bodies[0::2]])
    labels = [classify_sprocket(rollers, body_q[b], w)[0]
              for b, w in zip(sim.info.pulley_bodies, SPROCKETS, strict=True)]
    rep("G2 outcome", labels == ["outside"] * 3, f"small/large/tensioner {labels}")
    return sim.rtf()


def g3_roundbelt() -> None:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/checks/check_belt_pin.py")],
                       check=False, capture_output=True, text=True)
    rep("G3 round-belt check_belt_pin", r.returncode == 0 and r.stdout.count("[PASS]") == 4,
        f"exit {r.returncode}, {r.stdout.count('[PASS]')} PASS, {time.time() - t0:.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-roundbelt", action="store_true")
    ap.add_argument("--neq", type=int, default=2)
    ap.add_argument("--pad-condim", type=int, default=None)
    ap.add_argument("--finger-effort", type=float, default=None)
    args = ap.parse_args()
    t0 = time.time()
    sim = build_sim()
    g1_model(sim, args.neq, args.pad_condim or sim.pad_condim,
             args.finger_effort or sim.finger_effort_limit)
    rtf = g2_rest(sim)
    bound = 0.8 * spike_rtf()
    rep("G4 RTF", rtf >= bound, f"physics RTF {rtf:.2f} (bound {bound:.2f} = 0.8 x spike S6)")
    if not args.skip_roundbelt:
        g3_roundbelt()
    print(f"{'ALL PASS' if OK else 'FAILED'} ({time.time() - t0:.0f} s)")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
