#!/usr/bin/env python3
"""V12-90 Stage 0 probe: UR release from phase-1 seated end states, scripted Franka phase 2.

``run``: per seated episode (suite ``index.json`` rows with ``outcome == "engaged"`` and an
``end_state``) and release byte: restore the end snapshot (OSC settle), command the 2F-85 to the
byte, step ``--settle-s`` (Franka on the hold hook). From the ``--drive-byte`` release (and byte 0
for the first ``--drive-byte0-first`` episodes) the UR retreats by min-jerk to magna's
``place_9`` UR pose while the Franka goes ``place_9 -> place_10 -> target_lcs_positions`` 1..3
(EE part, board frame -> world), then holds ``--hold-s``; segment = max(1 s, length / speed).
Aborts as ``scripted_reset`` (Franka tip floor, 2F-85 board clearance, UR IK miss) with the grasp
test ``held_for(held, "franka")``. Both pulleys' ``frame_metrics`` every ``SAMPLE_STEPS``; a
camera belt-seg cloud (noise off) at the released state and after the hold. Sim GT only.
``report``: ``report.md`` (numbers only) + ``report.html`` from ``probes.json`` and the suite.

    uv run --frozen python scripts/lcs/phase2_probe.py --self-check
    setsid nohup flock data/lcs/realfirst/cmp/v12_79/sim.lock uv run --frozen python \\
        scripts/lcs/phase2_probe.py run --suite data/lcs/realfirst/cmp/v12_90/suite \\
        --lcm-url 'udpm://239.255.76.191:7791?ttl=0' --out data/lcs/realfirst/cmp/v12_90/probe \\
        > data/lcs/realfirst/cmp/v12_90/probe.log 2>&1 &
    uv run --frozen python scripts/lcs/phase2_probe.py report \\
        --probe data/lcs/realfirst/cmp/v12_90/probe --suite data/lcs/realfirst/cmp/v12_90/suite \\
        --out data/lcs/realfirst/cmp/v12_90/report.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from round_belt_task.arm_kinematics import UrTracking, ik, mat4, rpy_to_mat3
from round_belt_task.episode_io import pose7_mat, write_json
from round_belt_task.outcome import LARGE_SEAT_MM, SMALL_SEAT_MM, classify, frame_metrics
from round_belt_task.waypoints import (
    MAGNA_PARAMS_SIM_YAML,
    X_W_BOARD_PLANNING,
    board_pose,
    load_params,
    load_pre_mpc_segment,
)
from task_common import belt_cloud as bc

SEG_NEAR_MM = 35.0  # small: seg points within this of the small axis (in plane)
# shortcut: large uses the same margin past its seat (35 mm of its axis is inside the pulley)
SEG_NEAR_LARGE_MM = LARGE_SEAT_MM + SEG_NEAR_MM - SMALL_SEAT_MM
KEEP_DEG = 60.0


def wxyz(q_xyzw) -> np.ndarray:
    return np.roll(np.asarray(q_xyzw, dtype=np.float64), 1)


def franka_targets(board: str) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    """World ``(label, xyz+wxyz)``: place_9, place_10, target_1..3; and place_9's UR pose7.
    ``board`` for the targets: ``scene`` = this scene's board weld (the compiled waypoints'),
    ``live`` = the yaml's ``task_board_position`` (27 mm off the scene's board)."""
    wps = load_pre_mpc_segment(MAGNA_PARAMS_SIM_YAML, first="place_9", last="place_10")
    out = [(w.label, np.concatenate([w.franka_pos, wxyz(w.franka_quat_xyzw)])) for w in wps]
    params = load_params()
    X_WB = X_W_BOARD_PLANNING if board == "scene" else board_pose(params=params)
    X_WB = X_WB if params.get("target_poses_in_board_frame") else np.eye(4)
    for i, row in enumerate(params["target_lcs_positions"]):
        # [ee_xyz, ee_rpy, keypoint_xyz]; the keypoint is on the large pulley, not a target
        out.append((f"target_{i + 1}", pose7_mat(X_WB @ mat4(rpy_to_mat3(row[3:6]), row[:3]))))
    return out, np.concatenate([wps[0].ur_pos, wxyz(wps[0].ur_quat_xyzw)])


def self_check(board: str) -> dict:
    targets, ur9 = franka_targets(board)
    small = bc.known_small_pulley_pose7()
    d = {lab: float(np.linalg.norm(bc.to_pulley_frame(p[:3], small)[0, :2]) * 1e3)
         for lab, p in targets}
    for lab, _ in targets[2:]:
        assert 15.0 <= d[lab] <= 40.0, f"{lab} {d[lab]:.1f} mm from the small pulley axis"
    assert targets[0][1][2] > targets[1][1][2], "place_9 not above place_10"
    assert len(targets) == 5 and ur9[2] > targets[0][1][2]
    print(f"self-check ok ({board} board): in-plane distance to the small pulley axis [mm] "
          + ", ".join(f"{k} {v:.1f}" for k, v in d.items()))
    return d


# --- sim ----------------------------------------------------------------------------------------

def pulley_metrics(body_q, sim) -> dict:
    belt = body_q[sim.info.belt_bodies, :3].astype(np.float64)
    out = {}
    for name, i, seat in (("small", 0, SMALL_SEAT_MM), ("large", 1, LARGE_SEAT_MM)):
        fm = frame_metrics(belt, body_q[int(sim.info.pulley_bodies[i])].astype(np.float64), seat)
        out[name] = {**dataclasses.asdict(fm), "label": classify(fm)}
    return out


def sample(sim, ctx, t0: float, clear: float | None) -> dict:
    body_q = sim.state_0.body_q.numpy()
    joint_q = sim.state_0.joint_q.numpy()
    ee_f = sim.franka_measured_pose7(joint_q)
    ee_u = pose7_mat(UrTracking.fk(joint_q[sim.arm_coords()[1]]))
    g = sim.grasp_state(body_q)
    return {"t": sim.osc_time_s(sim.step_index) - t0, "ee_f": ee_f.tolist(),
            "ee_u": ee_u.tolist(), **pulley_metrics(body_q, sim),
            "held": [bool(v) for v in g.held()], "ur_tip_gap_mm": g.ur_tip_gap_mm,
            "tip_mm": float(ee_f[2] - ctx["plate_top"]) * 1e3,
            "clearance_mm": None if clear is None else clear * 1e3}


def seg_counts(sim, ctx) -> tuple[dict, np.ndarray]:
    joint_q = sim.state_0.joint_q.numpy()
    ee_f = sim.franka_measured_pose7(joint_q)
    ee_u = pose7_mat(UrTracking.fk(joint_q[sim.arm_coords()[1]]))
    xyz, rgb = sim.point_cloud()
    seg = bc.belt_seg_frame(xyz, rgb, bc.CLOUD_NOISE_PRESETS["off"], bc.DEFAULT_SEG,
                            np.random.default_rng(0), ctx["origins"],
                            bc.jaw_centres_from_poses(ee_f, ee_u), ctx["large7"])["pcd_belt_seg"]
    r_small = np.linalg.norm(seg[:, :2] - ctx["small_in_large"][:2], axis=1) * 1e3
    r_large = np.linalg.norm(seg[:, :2], axis=1) * 1e3
    return {"n_seg_total": len(seg), "n_seg_small": int((r_small <= SEG_NEAR_MM).sum()),
            "n_seg_large": int((r_large <= SEG_NEAR_LARGE_MM).sum())}, seg


def release(sim, ctx, snap, byte: int, settle_s: float) -> tuple[dict, list]:
    import eval_learned_mpc as harness

    if snap is not None:  # None: release from the live state (harness --stage2-chain)
        sim.restore(snap)
    t0 = sim.osc_time_s(sim.step_index)
    m0 = pulley_metrics(sim.state_0.body_q.numpy(), sim)
    sim.set_grippers(None, byte)
    board, gripper = ctx["geo"][byte]
    samples = []
    for k in range(round(settle_s / sim.frame_dt) + 1):
        if k:
            sim.control_step()
        if k % harness.SAMPLE_STEPS == 0:
            clear = float(gripper.measure(board, sim.state_0.body_q.numpy().astype(np.float64)))
            samples.append(sample(sim, ctx, t0, clear))
    end = samples[-1]
    wraps = [s["large"]["wrap_deg"] for s in samples]
    row = {"byte": byte, "wrap_restored": m0["large"]["wrap_deg"],
           "h_restored_mm": m0["large"]["h_median_mm"], "label_restored": m0["large"]["label"],
           "wrap_release": wraps[0], "wrap_min": min(wraps), "wrap_end": wraps[-1],
           "h_end_mm": end["large"]["h_median_mm"], "keeps_60": wraps[-1] >= KEEP_DEG,
           "label_end": end["large"]["label"], "small_label_end": end["small"]["label"],
           "held_end": end["held"], "ur_tip_gap_end_mm": end["ur_tip_gap_mm"]}
    return row, samples


def drive(sim, ctx, byte: int, a) -> tuple[dict, list]:
    """``scripted_reset``'s loop on the phase-2 paths; samples every ``SAMPLE_STEPS``."""
    import collect_lcs_dataset as cli
    import eval_learned_mpc as harness
    from scripted_reset import MinJerkPath

    from round_belt_task.commander import CommanderParams, pose_mat, saved_traj_message
    from round_belt_task.offline_simulation import held_for

    dt = sim.frame_dt
    knot_t = CommanderParams().dt * np.arange(harness.N_KNOTS)
    u_coords = sim.arm_coords()[1]
    joint_q = sim.state_0.joint_q.numpy()
    f0 = sim.franka_measured_pose7(joint_q)
    u0 = pose7_mat(UrTracking.fk(joint_q[u_coords]))
    path_f, path_u = MinJerkPath(f0), MinJerkPath(u0)
    prev, seg_t = f0, []
    for i, (label, p) in enumerate(ctx["targets"]):
        length = np.linalg.norm(p[:3] - prev[:3])
        if i == 0:  # the UR retreats during the place_9 segment, then holds
            length = max(length, np.linalg.norm(ctx["ur9"][:3] - u0[:3]))
        dur = max(1.0, float(length) / a.speed_mps)
        path_f.add(dur, p[:3], p[3:])
        if i == 0:
            path_u.add(dur, ctx["ur9"][:3], ctx["ur9"][3:])
        seg_t.append((label, path_f.end))
        prev = p
    path_f.add(a.hold_s, prev[:3], prev[3:])
    seg_t.append(("hold", path_f.end))
    n_steps = round(path_f.end / dt)
    board, gripper = ctx["geo"][byte]
    t0 = sim.osc_time_s(sim.step_index)

    def hook(step, t, joint_q, body_q):
        knots = [path_f(t - t0 + o) for o in knot_t]
        return saved_traj_message(round(t * 1e6), np.stack([k[0] for k in knots]),
                                  np.stack([k[1] for k in knots]), t + knot_t)

    q_ur = sim.arm_targets()[1]
    lead_gain = sim.arm_kd / sim.arm_ke / dt
    tip_min = clear_min = math.inf
    last_clear = None
    board_neg, k = 0, 0
    abort = kind = None
    samples = [sample(sim, ctx, t0, None)]
    w0 = time.perf_counter()
    sim.commander_hook = hook
    try:
        while k < n_steps:
            p, q = path_u(sim.osc_time_s(sim.step_index + 1) - t0)
            q_new, err_p, err_r, iters = ik(UrTracking, pose_mat(p, q), q_ur,
                                            pos_tol=cli.UR_IK_POS_TOL, rot_tol=cli.UR_IK_ROT_TOL)
            if err_p > cli.UR_IK_POS_TOL or err_r > cli.UR_IK_ROT_TOL:
                abort, kind = (f"UR IK missed at step {k} ({err_p * 1e3:.3f} mm, "
                               f"{np.degrees(err_r):.3f} deg after {iters} iterations)"), "ur_ik"
                break
            lead = cli.UR_VELOCITY_LEAD * lead_gain if k > 0 else 0.0
            sim.set_ur_target(q_new + lead * (q_new - q_ur))
            q_ur = q_new
            sim.control_step()
            k += 1
            tip = float(sim.franka_measured_pose7()[2] - ctx["plate_top"])
            tip_min = min(tip_min, tip)
            if k % cli.CLEARANCE_EVERY == 0:
                last_clear = float(gripper.measure(
                    board, sim.state_0.body_q.numpy().astype(np.float64)))
                clear_min = min(clear_min, last_clear)
            board_neg = board_neg + 1 if last_clear is not None and last_clear < 0.0 else 0
            if tip * 1e3 < harness.TIP_FLOOR_MM:
                abort, kind = (f"franka finger_tip {tip * 1e3:.2f} mm above the plate at step "
                               f"{k}"), "tip_floor"
            elif board_neg > harness.BOARD_NEGATIVE_STEPS:
                abort, kind = (f"2F-85 board clearance < 0 for {board_neg} steps at step {k}",
                               "board_clearance")
            elif k % harness.SAMPLE_STEPS == 0:
                samples.append(sample(sim, ctx, t0, last_clear))
                if not held_for(samples[-1]["held"], "franka"):
                    abort, kind = (f"grasp lost at step {k} ({sim.grasp_state().describe()})",
                                   "grasp_lost")
            if abort is not None:
                break
    finally:
        sim.commander_hook = sim.make_hold_hook()
        if abort is not None:
            sim.set_ur_target(sim.state_0.joint_q.numpy()[u_coords])
    if samples[-1]["t"] < (k - 0.5) * dt:
        samples.append(sample(sim, ctx, t0, last_clear))
    end = samples[-1]
    row = {"byte": byte, "abort": abort, "abort_kind": kind, "sim_s": k * dt,
           "planned_s": path_f.end, "wall_s": time.perf_counter() - w0,
           "segments": [(lab, round(t, 3)) for lab, t in seg_t],
           "tip_min_mm": tip_min * 1e3, "clearance_min_mm": clear_min * 1e3,
           "small_wrap_end": end["small"]["wrap_deg"], "small_h_end_mm": end["small"]["h_median_mm"],
           "small_label_end": end["small"]["label"],
           "large_wrap_end": end["large"]["wrap_deg"], "large_label_end": end["large"]["label"],
           "both_engaged": end["small"]["label"] == "engaged"
           and end["large"]["label"] == "engaged",
           "held_end": end["held"]}
    return row, samples


def seated_episodes(suite: Path) -> list[dict]:
    """Engaged rows with an end_state, in the suite's start-major run order."""
    import run_hard_suite as hs

    out = []
    for d in sorted(p.parent for p in suite.glob("*/index.json")):
        key = next((k for k in hs.KEYS if d.name.endswith("_" + k)), None)
        if key is None:
            continue
        for ep in json.loads((d / "index.json").read_text())["episodes"]:
            if ep.get("outcome") == "engaged" and ep.get("end_state"):
                out.append({"arm": d.name[:-len(key) - 1], "start": key,
                            "repeat": ep.get("repeat"), "start_id": ep["start_state"]["id"],
                            "end_state": str((d / ep["end_state"]).relative_to(REPO_ROOT))})
    order = {k: i for i, k in enumerate(hs.KEYS)}
    return sorted(out, key=lambda r: (order[r["start"]], r["arm"], r["repeat"]))


def cmd_run(a) -> int:
    import collect_lcs_dataset as cli
    import eval_learned_mpc as harness
    from loguru import logger

    from round_belt_task.osc_bridge import OscTimeout
    from round_belt_task.osc_simulation import RoundBeltOscSimulation
    from task_common import sim_snapshot

    out = (REPO_ROOT / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    eps = seated_episodes((REPO_ROOT / a.suite).resolve())
    if not eps:
        raise SystemExit(f"{a.suite}: no engaged episodes with an end_state")
    bytes_ = [int(b) for b in a.release_bytes.split(",")]
    if a.drive_byte not in bytes_ or (a.drive_byte0_first and 0 not in bytes_):
        raise SystemExit("--drive-byte (and byte 0 when --drive-byte0-first) need a release")
    targets, ur9 = franka_targets(a.target_board)
    meta = {"self_check_mm": self_check(a.target_board), "episodes": len(eps), "args": vars(a)}
    sim = RoundBeltOscSimulation.build(lcm_url=a.lcm_url, osc_timeout_s=5.0,
                                       bridge_cls=harness.ControllerBridge)
    rows = []
    try:
        binary, root = harness.default_osc_binary()
        sim.start_osc(out / "osc.log", binary=binary, cwd=root)
        sim.restore(sim_snapshot.load(REPO_ROOT / eps[0]["end_state"]))
        live = sim.state_0.body_q.numpy()[int(sim.info.pulley_bodies[0])][:3].astype(float)
        err = float(np.linalg.norm(live - bc.known_small_pulley_pose7()[:3]) * 1e3)
        logger.info(f"[PROBE] small pulley live vs known_small_pulley_pose7: {err:.3f} mm")
        assert err <= 2.0, f"small pulley {err:.2f} mm from known_small_pulley_pose7"
        meta["small_pulley_err_mm"] = err
        geo = {b: sim.clearance_geometry(jaw_bytes=(b,)) for b in bytes_}
        large7, plate = bc.known_large_pulley_pose7(), geo[bytes_[0]][0].plate
        ctx = {"geo": geo, "targets": targets, "ur9": ur9, "large7": large7,
               "plate_top": float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half),
               "origins": cli.sim_camera_origins(sim),
               "small_in_large": bc.to_pulley_frame(bc.known_small_pulley_pose7()[:3],
                                                    large7)[0]}
        write_json(out / "meta.json", meta)
        for i, ep in enumerate(eps):
            snap = sim_snapshot.load(REPO_ROOT / ep["end_state"])
            for byte in bytes_:
                stem = f"{ep['arm']}_{ep['start']}_{ep['repeat']}_b{byte}"
                row = {**ep, "byte": byte, "file": f"{stem}.json", "release": None,
                       "drive": None, "error": None}
                rec, segs = {}, {}
                w0 = time.perf_counter()
                try:
                    row["release"], rec["release"] = release(sim, ctx, snap, byte, a.settle_s)
                    if byte == a.drive_byte or (byte == 0 and i < a.drive_byte0_first):
                        c0 = {}
                        if not a.no_pcd:
                            c0, segs["released"] = seg_counts(sim, ctx)
                        row["drive"], rec["drive"] = drive(sim, ctx, byte, a)
                        row["drive"]["seg_released"] = c0
                        if not a.no_pcd:
                            row["drive"]["seg_end"], segs["end"] = seg_counts(sim, ctx)
                except OscTimeout as exc:
                    if not exc.process_alive:
                        raise
                    sim.bridge.forget_pending()
                    row["error"] = f"osc timeout: {exc}"
                row["wall_s"] = time.perf_counter() - w0
                write_json(out / row["file"], {**row, "samples": rec})
                if segs:
                    np.savez_compressed(out / f"{stem}_seg.npz", **segs)
                rows.append(row)
                write_json(out / "probes.json", {"meta": meta, "rows": rows})
                r, d = row["release"] or {}, row["drive"] or {}
                logger.info(f"[PROBE] {i + 1}/{len(eps)} {stem}: release wrap "
                            f"{r.get('wrap_release', float('nan')):.0f} -> "
                            f"{r.get('wrap_end', float('nan')):.0f} {r.get('label_end')}"
                            + (f"; drive small {d['small_wrap_end']:.0f} {d['small_label_end']}"
                               f" large {d['large_wrap_end']:.0f} {d['large_label_end']}"
                               f" abort {d['abort']}" if d else "")
                            + (f"; error {row['error']}" if row["error"] else ""))
    finally:
        sim.close("finished")
    return 0


# --- report -------------------------------------------------------------------------------------

def _f(x, s=".0f") -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, s)


def _q(v, p) -> float:
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.percentile(v, p)) if v else float("nan")


def _counts(v) -> str:
    return ", ".join(f"{k} {n}" for k, n in sorted(Counter(v).items())) or "-"


def report_md(probe: Path, suite: Path) -> tuple[str, list[dict]]:
    data = json.loads((probe / "probes.json").read_text())
    rows, meta = data["rows"], data["meta"]
    hs = json.loads((suite / "hard_suite.json").read_text())["rows"]
    md = ["# V12-90 Stage 0 probe (sim GT, numbers only)", "",
          f"Probe `{probe.relative_to(REPO_ROOT)}`, suite `{suite.relative_to(REPO_ROOT)}`. "
          f"Release settle {meta['args']['settle_s']:g} s, drive speed "
          f"{meta['args']['speed_mps']:g} m/s (segment = max(1 s, length / speed)), hold "
          f"{meta['args']['hold_s']:g} s, targets on the {meta['args']['target_board']} board. "
          "Small pulley live vs known pose: "
          f"{meta.get('small_pulley_err_mm', float('nan')):.3f} mm. Franka targets' in-plane "
          "distance to the small axis [mm]: "
          + ", ".join(f"{k} {v:.1f}" for k, v in meta["self_check_mm"].items()) + ".", "",
          "## (1) Suite", "", "| arm | engaged | outcomes | note |", "|---|---|---|---|"]
    for arm in dict.fromkeys(r["arm"] for r in hs):
        rr = [r for r in hs if r["arm"] == arm]
        n = sum(r["engaged"] for r in rr)
        md.append(f"| {arm} | {n}/{len(rr)} | {_counts(r['outcome'] for r in rr)} | "
                  + ("below 18/20 (reference suites 20/20)" if n < 18 else "") + " |")
    seated = list(dict.fromkeys((r["arm"], r["start"], r["repeat"]) for r in rows))
    md += ["", f"Seated set used: {len(seated)} episodes ("
           + _counts(a for a, _, _ in seated) + ").", "",
           "## (2) Release (large pulley, after the settle)", "",
           ("| byte | N | keeps >= 60 deg | wrap at release min / med | wrap end min / med | "
            "min-of-min wrap | labels end | Franka held end | errors |"), "|" + "---|" * 9]
    bytes_ = sorted({r["byte"] for r in rows}, reverse=True)
    for b in bytes_:
        rr = [r["release"] for r in rows if r["byte"] == b and r["release"]]
        md.append(f"| {b} | {len(rr)} | {sum(r['keeps_60'] for r in rr)}/{len(rr)} | "
                  f"{_f(_q([r['wrap_release'] for r in rr], 0))} / "
                  f"{_f(_q([r['wrap_release'] for r in rr], 50))} | "
                  f"{_f(_q([r['wrap_end'] for r in rr], 0))} / "
                  f"{_f(_q([r['wrap_end'] for r in rr], 50))} | "
                  f"{_f(_q([r['wrap_min'] for r in rr], 0))} | "
                  f"{_counts(r['label_end'] for r in rr)} | "
                  f"{sum(r['held_end'][0] for r in rr)}/{len(rr)} | "
                  f"{sum(1 for r in rows if r['byte'] == b and r['error'])} |")
    starts = list(dict.fromkeys(r["start"] for r in rows))
    md += ["", "Keeps >= 60 deg per start (n/N):", "",
           "| start | " + " | ".join(f"b{b}" for b in bytes_) + " |",
           "|---|" + "---|" * len(bytes_)]
    for s in starts:
        cells = []
        for b in bytes_:
            rr = [r["release"] for r in rows if r["start"] == s and r["byte"] == b and r["release"]]
            cells.append(f"{sum(r['keeps_60'] for r in rr)}/{len(rr)}")
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    drives = [r for r in rows if r["drive"]]
    md += ["", "## (3) Drive (place_9 -> place_10 -> targets 1-3 -> hold)", "",
           ("| arm | start | ep | byte | small wrap | small h | small label | large wrap | "
            "large label | both engaged | abort | tip min mm | clearance min mm | wall s | sim s |"),
           "|" + "---|" * 15]
    for r in drives:
        d = r["drive"]
        md.append(f"| {r['arm']} | {r['start']} | {r['repeat']} | {r['byte']} | "
                  f"{_f(d['small_wrap_end'])} | {_f(d['small_h_end_mm'], '.1f')} | "
                  f"{d['small_label_end']} | {_f(d['large_wrap_end'])} | "
                  f"{d['large_label_end']} | {'yes' if d['both_engaged'] else 'no'} | "
                  f"{d['abort_kind'] or '-'} | {_f(d['tip_min_mm'], '.1f')} | "
                  f"{_f(d['clearance_min_mm'], '.1f')} | {_f(d['wall_s'], '.1f')} | "
                  f"{_f(d['sim_s'], '.2f')} |")
    md += ["", (f"Probe wall time {sum(r['wall_s'] for r in rows) / 60:.1f} min "
                f"({len(rows)} probes, {len(drives)} drives)."),
           "", ("| byte | both engaged | small wrap med / p10 | large wrap after med / p10 | "
                "small labels | aborts |"), "|---|---|---|---|---|---|"]
    for b in bytes_:
        dd = [r["drive"] for r in drives if r["byte"] == b]
        if not dd:
            continue
        sw, lw = [d["small_wrap_end"] for d in dd], [d["large_wrap_end"] for d in dd]
        md.append(f"| {b} | {sum(d['both_engaged'] for d in dd)}/{len(dd)} | "
                  f"{_f(_q(sw, 50))} / {_f(_q(sw, 10))} | {_f(_q(lw, 50))} / {_f(_q(lw, 10))} | "
                  f"{_counts(d['small_label_end'] for d in dd)} | "
                  f"{_counts(d['abort_kind'] or 'none' for d in dd)} |")
    md += ["", "## (4) Camera belt-seg points (noise off)", "",
           (f"small: within {SEG_NEAR_MM:g} mm of the small axis in plane; large: within "
            f"{SEG_NEAR_LARGE_MM:.2f} mm of the large axis (the same "
            f"{SEG_NEAR_MM - SMALL_SEAT_MM:.2f} mm past its seat); raw points in "
            "`<stem>_seg.npz` (large-pulley frame)."), "",
           "drive end = after the hold, or at the abort.", "",
           "| when | byte | N | small med / min | large med / min | total med / min |",
           "|---|---|---|---|---|---|"]
    for when in ("seg_released", "seg_end"):
        for b in bytes_:
            cc = [r["drive"][when] for r in drives if r["byte"] == b and r["drive"].get(when)]
            if not cc:
                continue
            cell = " | ".join(f"{_f(_q([c[k] for c in cc], 50))} / {_f(_q([c[k] for c in cc], 0))}"
                              for k in ("n_seg_small", "n_seg_large", "n_seg_total"))
            md.append(f"| {'released' if when == 'seg_released' else 'drive end'} | {b} | "
                      f"{len(cc)} | {cell} |")
    aborts = [r["drive"]["abort"] for r in drives if r["drive"]["abort"]]
    errors = [r["error"] for r in rows if r["error"]]
    md += ["", "## (5) Failure notes", "",
           f"{len(aborts)} drive aborts, {len(errors)} probe errors. First aborts:", ""]
    md += [f"- `{x}`" for x in aborts[:3]] or ["- none"]
    md += [f"- error: `{x}`" for x in errors[:3]]
    md += ["", "Wrap vs time per drive: `report.html`."]
    return "\n".join(md) + "\n", drives


def report_html(probe: Path, drives: list[dict]) -> str:
    W, H, P = 640, 260, 30
    t_max = max([r["drive"]["planned_s"] for r in drives] or [1.0])
    svgs = []
    for pulley in ("small", "large"):
        lines = []
        for r in drives:
            s = json.loads((probe / r["file"]).read_text())["samples"]["drive"]
            pts = " ".join(f"{P + x['t'] / t_max * (W - 2 * P):.1f},"
                           f"{H - P - x[pulley]['wrap_deg'] / 360 * (H - 2 * P):.1f}" for x in s)
            col = "#c33" if r["drive"]["abort"] else "#36c" if r["byte"] else "#3a3"
            name = f"{r['arm']} {r['start']} {r['repeat']} b{r['byte']}"
            lines.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.2"'
                         f' opacity="0.7"><title>{name}</title></polyline>')
        y60 = H - P - 60 / 360 * (H - 2 * P)
        svgs.append(f'<h3>{pulley} pulley wrap [deg, 0-360] vs drive time [s, 0-{t_max:.1f}]</h3>'
                    f'<svg width="{W}" height="{H}" style="border:1px solid #ccc">'
                    f'<line x1="{P}" x2="{W - P}" y1="{y60:.1f}" y2="{y60:.1f}" stroke="#999" '
                    f'stroke-dasharray="4"/>{"".join(lines)}</svg>')
    return ("<!doctype html><meta charset='utf-8'><title>V12-90 probe</title><body "
            "style='font-family:sans-serif'><p>One line per drive (hover: episode). Blue byte "
            "191, green byte 0, red aborted. Dashed: 60 deg.</p>" + "".join(svgs) + "</body>\n")


def cmd_report(a) -> int:
    probe, suite = (REPO_ROOT / a.probe).resolve(), (REPO_ROOT / a.suite).resolve()
    md, drives = report_md(probe, suite)
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    out.with_suffix(".html").write_text(report_html(probe, drives))
    print(md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-check", action="store_true", help="CPU target-geometry asserts")
    ap.add_argument("--target-board", choices=("scene", "live"), default="scene",
                    help="board pose for target_lcs_positions (see franka_targets)")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--lcm-url", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--release-bytes", default="191,0")
    r.add_argument("--drive-byte", type=int, default=191)
    r.add_argument("--drive-byte0-first", type=int, default=5)
    r.add_argument("--settle-s", type=float, default=2.0)
    r.add_argument("--hold-s", type=float, default=2.0)
    r.add_argument("--speed-mps", type=float, default=0.08)
    r.add_argument("--no-pcd", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--probe", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.self_check:
        self_check(a.target_board)
        return 0
    if a.cmd is None:
        ap.error("run | report | --self-check")
    return cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
