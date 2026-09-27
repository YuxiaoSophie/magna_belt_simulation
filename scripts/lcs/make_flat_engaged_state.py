#!/usr/bin/env python3
"""Build a held, groove-engaged, tensioned belt state (an MPC target) by driving the arms.

Phases (``--phases``, default all; each reads the previous phase's file):

* ``demo``: replay the flat-hold demo (collector ``--scenario nominal --hold-ur-gripper
  --nominal-ur-dz-mm 0 --pre-hold-s 0``) and snapshot its final state ->
  ``<target>/demo_final_osc.npz``.
* ``tension``: restore it, move both grasps away from the pulley along their free spans in
  ``--step-mm`` steps (Franka: OSC Cartesian min-jerk; UR: per-step IK line), measure belt
  stretch / wrap / h_median / tilt / grasp / clearances after each, pick the step, hold it
  ``--verify-s`` under the OSC hold-at-measured hook (backing off if it drifts out) and save
  ``<out>/flat_engaged_osc.npz`` + ``index.json``.
* ``observe``: restore that file as the eval does (0.5 s hold), save the observation (cropped
  camera cloud, 150 material belt points, 40-dim state, EE poses, belt bodies, metrics), encode
  it with ``--deploy`` (``z_target`` + whitened distances) and record ``--record-s`` of hold.

Run:
    uv run python scripts/lcs/make_flat_engaged_state.py --lcm-url 'udpm://239.255.76.125:7725?ttl=0'
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import collect_lcs_dataset as col
import make_ee_offset_starts as eo
import make_grasp_variants as gv
from loguru import logger

from round_belt_task import perturbation as pert
from round_belt_task.arm_kinematics import UrTracking, ik, quat_xyzw_to_mat3
from round_belt_task.episode_io import pose7_mat
from round_belt_task.outcome import (
    DEFAULT_THRESHOLDS,
    LARGE_SEAT_MM,
    belt_in_pulley_frame,
    classify,
    frame_metrics,
    slant_metrics,
)
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, load_pre_mpc_segment
from task_common import lcs_dataset as lcs
from task_common import sim_snapshot
from task_common.latent_encoder import LatentEncoder, LearnedLcs

DEFAULT_OUT = sim_snapshot.DEFAULT_START_STATE_DIR / "flat_engaged"
DEFAULT_TARGET = REPO_ROOT / "data" / "lcs" / "synthetic_targets" / "flat_engaged"
DEFAULT_DEPLOY = Path("/home/hienbui/git/lcs_learning/outputs/sim_belt_v2_20260925/"
                      "deploy_v2_flat_pp2/deploy.npz")
DEFAULT_SPLIT = DEFAULT_DEPLOY.parents[1] / "split.json"
FLAT_DEMO = REPO_ROOT / "data" / "lcs" / "demo_flat" / "demo_episode.npz"
STATE_ID = "flat_engaged"


# --- measurements ---------------------------------------------------------------------------------


def _xform_points(pose7: np.ndarray, local: np.ndarray) -> np.ndarray:
    """``(n, 3)`` world points of ``local`` offsets in bodies ``pose7`` (xyz + xyzw)."""
    R = np.stack([quat_xyzw_to_mat3(q) for q in pose7[:, 3:7]])
    return pose7[:, :3] + np.einsum("nij,nj->ni", R, local)


class RodGauge:
    """Belt stretch from the rod joints: capsules are rigid, so all stretch is joint separation."""

    def __init__(self, sim) -> None:
        m = sim.model
        joints = np.asarray(sim.info.belt_joints, dtype=np.int64)
        parent, child = m.joint_parent.numpy()[joints], m.joint_child.numpy()[joints]
        keep = parent >= 0  # the rod's free root joint has no parent
        self.joints, self.parent, self.child = joints[keep], parent[keep], child[keep]
        self.X_p = m.joint_X_p.numpy()[self.joints].astype(np.float64)
        self.X_c = m.joint_X_c.numpy()[self.joints].astype(np.float64)
        ke = m.joint_target_ke.numpy()
        self.ke = ke[m.joint_qd_start.numpy()[self.joints]].astype(np.float64)
        # Rigid length of each capsule = its two joint anchors (start = X_c, end = X_p).
        start = {int(c): self.X_c[k, :3] for k, c in enumerate(self.child)}
        end = {int(p): self.X_p[k, :3] for k, p in enumerate(self.parent)}
        bodies = [b for b in sim.info.belt_bodies if b in start and b in end]
        self.rigid_len = float(sum(np.linalg.norm(end[b] - start[b]) for b in bodies))
        self.n_bodies = len(sim.info.belt_bodies)
        self.n_closed = len(bodies)

    def measure(self, body_q: np.ndarray) -> dict:
        body_q = np.asarray(body_q, dtype=np.float64)
        pa = _xform_points(body_q[self.parent], self.X_p[:, :3])
        ca = _xform_points(body_q[self.child], self.X_c[:, :3])
        R = np.stack([quat_xyzw_to_mat3(q) for q in body_q[self.parent, 3:7]])
        axis = R[:, :, 2]  # capsule axis = body +z
        gap = ca - pa
        axial = np.einsum("ij,ij->i", gap, axis)
        force = self.ke * axial
        return {"axial_gap_mm": axial * 1e3, "gap_norm_mm": np.linalg.norm(gap, axis=1) * 1e3,
                "stretch_pct": float(axial.sum() / self.rigid_len * 100.0),
                "gap_len_pct": float(np.linalg.norm(gap, axis=1).sum()
                                     / self.rigid_len * 100.0),
                "tension_n": force}


def loop_len_pct(belt: np.ndarray) -> float:
    """Body-centre loop length vs the rest ellipse's, in %."""
    def length(p):
        return float(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1).sum())
    return (length(belt) / length(lcs.rest_belt_bodies(len(belt))) - 1.0) * 100.0


def path_len_pct(belt: np.ndarray, path: list[int]) -> float:
    """Body-centre length along ``path`` (body indices) vs the same bodies at rest, in %."""
    rest = lcs.rest_belt_bodies(len(belt))
    idx = np.asarray(path)
    return float(np.linalg.norm(np.diff(belt[idx], axis=0), axis=1).sum()
                 / np.linalg.norm(np.diff(rest[idx], axis=0), axis=1).sum() - 1.0) * 100.0


def pulley_path(n: int, a: int, b: int, seated: np.ndarray) -> list[int]:
    """Body indices from ``a`` to ``b`` along the side of the loop that holds the seated bodies."""
    fwd = [(a + k) % n for k in range((b - a) % n + 1)]
    bwd = [(a - k) % n for k in range((a - b) % n + 1)]
    return fwd if seated[fwd].sum() >= seated[bwd].sum() else bwd


@dataclasses.dataclass
class Ctx:
    sim: object
    gauge: RodGauge
    board: object
    gripper: object
    plate_top_z: float
    tangent: np.ndarray
    ref_meas: dict | None = None
    base: dict | None = None
    jaw: bool = False


def pulley_pose(sim, body_q) -> np.ndarray:
    return np.asarray(body_q[int(sim.info.pulley_bodies[1])], dtype=np.float64)


def seated_mask(belt, pose) -> np.ndarray:
    h, r, _ = belt_in_pulley_frame(belt, pose)
    th = DEFAULT_THRESHOLDS
    return (np.abs(h) <= th.in_groove_axial_mm) & (np.abs(r - LARGE_SEAT_MM)
                                                    <= th.in_groove_radial_mm)


def span_segment(ctx: Ctx, belt, grasp_body: int, seated) -> list[int]:
    """Bodies from ``grasp_body`` walking the short way to the first seated body."""
    n = len(belt)
    best = None
    for step in (1, -1):
        path = [grasp_body]
        i = grasp_body
        for _ in range(n):
            i = (i + step) % n
            path.append(i)
            if seated[i]:
                break
        if seated[path[-1]] and (best is None or len(path) < len(best)):
            best = path
    return best or [grasp_body]


def snapshot_metrics(ctx: Ctx) -> dict:
    sim = ctx.sim
    body_q = sim.state_0.body_q.numpy().astype(np.float64)
    belt = body_q[sim.info.belt_bodies, :3]
    pose = pulley_pose(sim, body_q)
    fm = frame_metrics(belt, pose)
    sm = slant_metrics(belt, pose, ctx.tangent)
    rod = ctx.gauge.measure(body_q)
    X_f, X_u = sim.ee_poses(body_q)
    grasp = sim.grasp_state(body_q)
    seated = seated_mask(belt, pose)
    # Tension of the grasp-to-grasp segment over the pulley vs the free loop.
    gi = {k: int(np.argmin(np.linalg.norm(belt - X[:3, 3], axis=1)))
          for k, X in (("franka", X_f), ("ur", X_u))}
    seg = pulley_path(len(belt), gi["franka"], gi["ur"], seated)
    out = {
        "seg_stretch_pct": path_len_pct(belt, seg), "seg_bodies": [seg[0], seg[-1], len(seg)],
        "outcome": classify(fm), "wrap_deg": fm.wrap_deg, "h_median_mm": fm.h_median_mm,
        "h_min_mm": fm.h_min_mm, "h_max_mm": fm.h_max_mm, "r_median_mm": fm.r_median_mm,
        "seated_bodies": fm.seated_bodies, "tilt_deg": sm.slant_deg, "tilt_dir": sm.slant_dir,
        "stretch_pct": rod["stretch_pct"], "gap_len_pct": rod["gap_len_pct"],
        "tension_n_max": float(rod["tension_n"].max()),
        "tension_n_median": float(np.median(rod["tension_n"])),
        "tension_n": rod["tension_n"].tolist(),
        "centre_loop_len_pct": loop_len_pct(belt),
        "held": list(grasp.held()), "grasp": grasp.describe(),
        "ur_board_clearance_mm": float(ctx.gripper.measure(ctx.board, body_q)) * 1e3,
        "franka_tip_clearance_mm": float(X_f[2, 3] - ctx.plate_top_z) * 1e3,
        "ee_franka": pose7_mat(X_f).tolist(), "ee_ur": pose7_mat(X_u).tolist(),
        "grasp_bodies": gi, "belt_z_mm": [float(belt[:, 2].min() * 1e3),
                                          float(belt[:, 2].max() * 1e3)],
    }
    # Per-joint tension near the pulley (seated bodies) = the belt tension on the groove.
    seated_idx = np.flatnonzero(seated)
    joint_of_parent = {int(p): k for k, p in enumerate(ctx.gauge.parent)}
    belt_ids = list(sim.info.belt_bodies)
    ks = [joint_of_parent[belt_ids[i]] for i in seated_idx if belt_ids[i] in joint_of_parent]
    out["tension_n_seated_mean"] = float(np.mean(rod["tension_n"][ks])) if ks else float("nan")
    if ctx.ref_meas is not None:
        off = gv.offsets(gv.measure(sim, body_q), ctx.ref_meas)
        out["grasp_slide_mm"] = {k: off[k]["slide_mm"] for k in ("franka", "ur")}
    if ctx.jaw:
        out["ur_jaw"] = gv.ur_jaw_depth(sim, body_q)
        out["fingertip_z_mm"] = out["ur_jaw"]["fingertip_xyz"][2] * 1e3
        out["groove_z_mm"] = float(pose[2]) * 1e3  # groove seat plane h = 0 = pulley origin
        out["tip_minus_groove_mm"] = out["fingertip_z_mm"] - out["groove_z_mm"]
    if ctx.base is not None:
        for k in ("seg_stretch_pct", "stretch_pct", "centre_loop_len_pct"):
            out["d_" + k] = out[k] - ctx.base[k]
    return out


def short(m: dict) -> str:
    slide = m.get("grasp_slide_mm")
    s = "" if slide is None else f" slide F {slide['franka']:+.1f} U {slide['ur']:+.1f} mm"
    d = "" if "d_seg_stretch_pct" not in m else f" (d {m['d_seg_stretch_pct']:+.3f})"
    jaw = "" if "ur_jaw" not in m else (f" UR jaw depth {m['ur_jaw']['depth_mm']:.1f} mm "
                                        f"tip-groove {m['tip_minus_groove_mm']:+.1f} mm")
    return (f"{m['outcome']} wrap {m['wrap_deg']:.0f} h {m['h_median_mm']:+.2f} mm tilt "
            f"{m['tilt_deg']:.2f} ({m['tilt_dir']}) seg {m['seg_stretch_pct']:+.3f} %{d} loop "
            f"{m['centre_loop_len_pct']:+.3f} % rod {m['stretch_pct']:+.3f} % "
            f"T seated {m['tension_n_seated_mean']:.2f} N max {m['tension_n_max']:.2f} N held "
            f"{m['held']} clr UR {m['ur_board_clearance_mm']:.2f} F tip "
            f"{m['franka_tip_clearance_mm']:.1f} mm{s}{jaw}")


def ok_engaged(m: dict, args, min_wrap: float | None = None) -> list[str]:
    bad = []
    min_wrap = args.min_wrap_deg if min_wrap is None else min_wrap
    if not all(m["held"]):
        bad.append(f"held {m['held']}")
    if not m["wrap_deg"] >= min_wrap:
        bad.append(f"wrap {m['wrap_deg']:.0f} < {min_wrap:g}")
    if not abs(m["h_median_mm"]) <= args.max_h_mm:
        bad.append(f"|h| {m['h_median_mm']:.2f} > {args.max_h_mm:g}")
    if not (m["ur_board_clearance_mm"] > 0.0 and m["franka_tip_clearance_mm"] > 0.0):
        bad.append("clearance <= 0")
    return bad


def build_ctx(sim, params: Path) -> Ctx:
    nominal = load_pre_mpc_segment(params, first=col.FIRST, last=col.LAST)
    board, gripper = sim.clearance_geometry()
    plate = board.plate
    return Ctx(sim=sim, gauge=RodGauge(sim), board=board, gripper=gripper,
               plate_top_z=float(plate.pos[2] + np.abs(plate.rot[2]) @ plate.half),
               tangent=pert.belt_tangent(nominal))


# --- phase: demo ----------------------------------------------------------------------------------


def demo_context(sim, args, out: Path, ur_dz: str, franka_off: str | None):
    """Collector ``(OscContext, args, start)`` for UR-held nominal replays with these offsets."""
    cargs = col.create_parser().parse_args([
        "--backend", "osc", "--scenario", "nominal", "--hold-ur-gripper",
        f"--nominal-ur-dz-mm={ur_dz}", "--pre-hold-s", "0", "--episodes", "1", "--no-pcd",
        "--lcm-url", sim.osc_url, "--out", str(out), "--params", str(args.params)]
        + (["--start-state", str(args.start_state)] if args.start_state else [])
        + ([f"--nominal-franka-offset-mm={franka_off}"] if franka_off else []))
    out.mkdir(parents=True, exist_ok=True)
    nominal = load_pre_mpc_segment(cargs.params, first=col.FIRST, last=col.LAST)
    start = args.start_state or col.DEFAULT_START_STATES["osc"]
    snap = sim_snapshot.load(start)
    sim.restore(snap)  # as collect(): the guard's geometry is read at the start pose
    jaws = sorted({w.ur_gripper_byte for w in nominal if w.ur_gripper_byte is not None})
    board, gripper = sim.clearance_geometry(jaw_bytes=tuple(jaws))
    cctx = col.OscContext(
        args=cargs, out=out, n=col.sample_steps(cargs.sample_period),
        thresholds=DEFAULT_THRESHOLDS, tangent=pert.belt_tangent(nominal), snap=snap,
        nominal=nominal, board=board, gripper=gripper,
        excite=col.excitation_config(cargs, "nominal"), scenario="nominal",
        osc=sim.osc.describe(), start_state=start, opts=col.hold_options(cargs, "nominal"))
    return cctx, cargs, start


def waypoint_offsets(ur_dz: str, franka_off: str | None) -> dict:
    return {"ur_dz_mm": col.parse_ur_dz(ur_dz),
            "franka_offset_mm": col.parse_franka_offset(franka_off)}


def phase_demo(sim, ctx: Ctx, args) -> Path:
    cctx, cargs, start = demo_context(sim, args, args.target / "demo_replay", args.ur_dz_mm,
                                      args.franka_offset_mm)
    tries, best = [], None
    for i in range(args.demo_tries):
        row = col.run_osc_episode(sim, cctx, i, pert.Perturbation(intent="nominal"))
        if row["status"] != "ok":
            tries.append({"episode": i, "status": row["status"], "reason": row.get("reason")})
            continue
        m = snapshot_metrics(ctx)
        snap_i = sim_snapshot.capture(sim, "demo_final", notes=(
            f"flat-hold demo replay episode {i}: nominal, UR held, dz {args.ur_dz_mm}, pre-hold 0; "
            f"final settled state; {m['grasp']}"))
        t = {"episode": i, "status": "ok", "outcome": row["outcome"],
             "final_wrap_deg": row["final_wrap_deg"], "final_h_median_mm":
             row["final_h_median_mm"], "final_slant_deg": row["final_slant_deg"],
             "grasp_ok_final": row["grasp_ok_final"], "now": short(m)}
        if ctx.jaw:
            t["ur_jaw"] = m["ur_jaw"]
        tries.append(t)
        logger.info(f"[DEMO] replay {i}: {row['outcome']} wrap {row['final_wrap_deg']:.0f} "
                    f"slant {row['final_slant_deg']:.2f} | {short(m)}")
        score = abs(row["final_wrap_deg"] - 126.0) + 10.0 * abs(row["final_h_median_mm"])
        if row["outcome"] == "engaged" and all(row["grasp_ok_final"]) and (
                best is None or score < best[0]):
            best = (score, snap_i, i)
        if best is not None and best[0] < 5.0:
            break
    if best is None:
        raise RuntimeError(f"no engaged + held demo replay in {tries}")
    path = sim_snapshot.save(best[1], args.target / "demo_final_osc.npz")
    extra = ({} if args.ur_dz_mm == "0" and not args.franka_offset_mm
             else {"waypoint_offsets": waypoint_offsets(args.ur_dz_mm, args.franka_offset_mm)})
    (args.target / "demo_replay.json").write_text(json.dumps(
        {"picked_episode": best[2], "tries": tries, "start_state": str(start),
         "collector_args": {k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(cargs).items()}, **extra}, indent=2) + "\n")
    logger.success(f"[DEMO] picked replay {best[2]} -> {path}")
    return path


# --- phase: grid (opt-in) -------------------------------------------------------------------------


def parse_grid(text: str) -> list[tuple[str, str | None]]:
    """``"UR_DZ[|FX,FY,FZ];..."`` -> ``[(ur_dz, franka_offset or None)]``."""
    out = []
    for item in text.split(";"):
        if item.strip():
            ur, _, f = item.strip().partition("|")
            out.append((ur.strip(), f.strip() or None))
    return out


def grid_score(r: dict) -> tuple:
    """Higher is better: passes, wrap, then lower tilt (all after restore + settle)."""
    m = r["restored"]
    return (not r["bad"], m["wrap_deg"], -m["tilt_deg"])


def phase_grid(sim, ctx: Ctx, args) -> Path:
    """Replay each waypoint-offset setting ``--demo-tries`` times; judge the state as the tension
    phase starts it (restore + ``--settle-s``); save the best replay as the demo final."""
    dt = sim.frame_dt
    rows, snaps = [], {}
    for ur_dz, f_off in parse_grid(args.grid):
        name = f"ur{ur_dz}_f{f_off or '0'}".replace(",", "_").replace(":", "-")
        cctx, _, _ = demo_context(sim, args, args.target / "grid" / name, ur_dz, f_off)
        for i in range(args.demo_tries):
            row = col.run_osc_episode(sim, cctx, i, pert.Perturbation(intent="nominal"))
            r = {"setting": name, "ur_dz": ur_dz, "franka_offset": f_off,
                 "offsets": waypoint_offsets(ur_dz, f_off), "episode": i,
                 "status": row["status"], "reason": row.get("reason")}
            if row["status"] != "ok":
                r["bad"] = [f"status {row['status']}"]
                rows.append(r)
                continue
            end = snapshot_metrics(ctx)
            snap_i = sim_snapshot.capture(sim, "demo_final", notes=(
                f"flat-hold demo replay {name} episode {i}: nominal, UR held, pre-hold 0, "
                f"offsets {json.dumps(r['offsets'])}; {end['grasp']}"))
            sim.restore(snap_i, settle_steps=round(args.settle_s / dt))
            m = snapshot_metrics(ctx)
            bad = ok_engaged(m, args) + ([] if row["outcome"] == "engaged" else
                                         [f"collector {row['outcome']}"])
            tol = args.tip_groove_tol_mm
            if tol is not None and not abs(m.get("tip_minus_groove_mm", np.inf)) <= tol:
                bad.append(f"tip-groove {m.get('tip_minus_groove_mm', np.nan):+.1f} mm")
            keep = ("outcome", "wrap_deg", "h_median_mm", "tilt_deg", "tilt_dir", "held",
                    "ur_board_clearance_mm", "franka_tip_clearance_mm", "seg_stretch_pct",
                    "ur_jaw", "ee_franka", "ee_ur", "fingertip_z_mm", "groove_z_mm",
                    "tip_minus_groove_mm")
            r.update({"collector_outcome": row["outcome"], "bad": bad,
                      "end": {k: end.get(k) for k in keep},
                      "restored": {k: m.get(k) for k in keep}})
            rows.append(r)
            snaps[(name, i)] = snap_i
            logger.info(f"[GRID] {name} #{i}: end {short(end)} || restored {short(m)}"
                        + (f" | BAD {bad}" if bad else ""))
    ok = [r for r in rows if "restored" in r]
    if not ok or all(r["bad"] for r in ok):
        (args.target / "grid.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        raise RuntimeError("no grid setting passed")
    by = {}
    for r in ok:
        by.setdefault(r["setting"], []).append(r)
    # setting: most passes, then median passing wrap, then lower median passing tilt
    def setting_key(name):
        good = [r for r in by[name] if not r["bad"]]
        if not good:
            return (0, 0.0, 0.0)
        return (len(good), float(np.median([r["restored"]["wrap_deg"] for r in good])),
                -float(np.median([r["restored"]["tilt_deg"] for r in good])))
    best_setting = max(by, key=setting_key)
    best = max(by[best_setting], key=grid_score)
    path = sim_snapshot.save(snaps[(best["setting"], best["episode"])],
                             args.target / "demo_final_osc.npz")
    payload = {"picked_setting": best_setting, "picked_episode": best["episode"],
               "waypoint_offsets": best["offsets"], "tries_per_setting": args.demo_tries,
               "start_state": str(args.start_state or col.DEFAULT_START_STATES["osc"]),
               "judged_after_restore_settle_s": args.settle_s, "rows": rows}
    (args.target / "grid.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.target / "demo_replay.json").write_text(json.dumps(
        {"picked_episode": best["episode"], "picked_setting": best_setting,
         "waypoint_offsets": best["offsets"], "grid": "grid.json",
         "start_state": payload["start_state"]}, indent=2) + "\n")
    logger.success(f"[GRID] picked {best_setting} #{best['episode']} -> {path}")
    return path


# --- phase: tension -------------------------------------------------------------------------------


def span_dirs(ctx: Ctx) -> dict:
    """Per arm: unit horizontal direction from the span's pulley departure body to the grasp."""
    sim = ctx.sim
    body_q = sim.state_0.body_q.numpy().astype(np.float64)
    belt = body_q[sim.info.belt_bodies, :3]
    seated = seated_mask(belt, pulley_pose(sim, body_q))
    out = {}
    for name, X in zip(("franka", "ur"), sim.ee_poses(body_q), strict=True):
        g = int(np.argmin(np.linalg.norm(belt - X[:3, 3], axis=1)))
        seg = span_segment(ctx, belt, g, seated)
        d = X[:3, 3] - belt[seg[-1]]
        d[2] = 0.0
        out[name] = {"dir": (d / np.linalg.norm(d)).tolist(), "grasp_body": g,
                     "departure_body": int(seg[-1]), "span_bodies": len(seg) - 1,
                     "span_len_mm": float(np.linalg.norm(X[:3, 3] - belt[seg[-1]]) * 1e3)}
    return out


def move_both(sim, Xf0, Xf1, Xu0, Xu1, move_s: float, hold_s: float, q_prev: list):
    """Franka min-jerk OSC target + UR per-step IK line, simultaneously; then hold the targets."""
    dt = sim.frame_dt
    steps = round(move_s / dt)
    sim.commander_hook = eo.franka_move_hook(sim, Xf0, Xf1, move_s)

    def on_step(k):
        s = eo.min_jerk((k + 1) / steps)
        X = Xu0.copy()
        X[:3, 3] = Xu0[:3, 3] + s * (Xu1[:3, 3] - Xu0[:3, 3])
        q, ep, er, _ = ik(UrTracking, X, q_prev[0], pos_tol=1e-6, rot_tol=1e-5)
        if ep > 1e-5 or er > 1e-4:
            raise RuntimeError(f"UR IK err {ep:.2e} m {er:.2e} rad")
        q_prev[0] = q
        sim.set_ur_target(q)

    ok, why, _ = eo.run_steps(sim, steps, on_step)
    if ok:
        ok, why, _ = eo.run_steps(sim, round(hold_s / dt))
    return ok, why


def hold_verify(ctx: Ctx, args, label: str) -> tuple[bool, list[dict]]:
    """``--verify-s`` under the hold-at-measured hook; metrics every 0.5 s."""
    sim = ctx.sim
    sim.commander_hook = sim.make_hold_hook()
    trace, good = [], True
    per = round(0.5 / sim.frame_dt)
    for k in range(max(1, round(args.verify_s / 0.5))):
        ok, why, _ = eo.run_steps(sim, per)
        m = snapshot_metrics(ctx)
        bad = ok_engaged(m, args) + ([] if ok else [why])
        trace.append({"t_s": 0.5 * (k + 1), **{key: m[key] for key in (
            "outcome", "wrap_deg", "h_median_mm", "tilt_deg", "seg_stretch_pct",
            "d_seg_stretch_pct", "centre_loop_len_pct", "stretch_pct",
            "tension_n_seated_mean", "held", "ur_board_clearance_mm",
            "franka_tip_clearance_mm")}, "bad": bad})
        logger.info(f"[VERIFY {label}] t {0.5 * (k + 1):.1f} s: {short(m)}"
                    + (f" | BAD {bad}" if bad else ""))
        good = good and not bad
    return good and trace[-1]["d_seg_stretch_pct"] >= args.min_stretch_pct, trace


def demo_offsets(args) -> dict | None:
    """Place-waypoint offsets the demo final was replayed with (None = nominal)."""
    for name in ("demo_replay.json", "raise.json"):
        f = args.target / name
        if f.exists():
            return json.loads(f.read_text()).get("waypoint_offsets")
    return None


def start_and_base(args) -> tuple[Path, Path | None]:
    """Tension start state and, for ``--raise-from``, the separate stretch baseline."""
    if args.raise_from is not None:
        return args.target / "raise_final_osc.npz", args.raise_base
    return args.target / "demo_final_osc.npz", None


def phase_tension(sim, ctx: Ctx, args) -> Path:
    start_path, base_path = start_and_base(args)
    if base_path is None:
        return tension_from(sim, ctx, args, start_path, None)
    # raise mode: back off one raise step at a time until tension + hold verify pass
    rfile = args.target / "raise.json"
    r = json.loads(rfile.read_text())
    attempts = []
    for k in range(r["picked_step"], -1, -1):
        row = next(x for x in r["rows"] if x["step"] == k)
        info = {"raise_from": r["raise_from"], "used_step": k,
                "ur_raise_mm": k * r["step_mm"], "reached_at_step": r["picked_step"],
                "reached": r["reached"], "tip_minus_groove_mm_at_step": row["tip_minus_groove_mm"],
                "rows": r["rows"], "attempts": attempts}
        try:
            path = tension_from(sim, ctx, args,
                                args.target / "raise_steps" / f"step_{k:02d}_osc.npz",
                                base_path, info)
        except RuntimeError as e:
            attempts.append({"step": k, "error": str(e)[:300]})
            logger.warning(f"[TENSION] raise step {k} failed: {str(e)[:200]}")
            continue
        attempts.append({"step": k, "passed": True})
        r.update({"tension_raise_step": k, "tension_attempts": attempts})
        rfile.write_text(json.dumps(r, indent=2) + "\n")
        return path
    r["tension_attempts"] = attempts
    rfile.write_text(json.dumps(r, indent=2) + "\n")
    raise RuntimeError("no raise step passed tension + verify")


def tension_from(sim, ctx: Ctx, args, start_path: Path, base_path: Path | None,
                 raise_info: dict | None = None) -> Path:
    dt = sim.frame_dt
    if base_path is not None:
        sim.restore(sim_snapshot.load(base_path), settle_steps=round(args.settle_s / dt))
        ctx.ref_meas, ctx.base = None, None
        ctx.base = snapshot_metrics(ctx)
    snap0 = sim_snapshot.load(start_path)
    sim.restore(snap0, settle_steps=round(args.settle_s / dt))
    ctx.ref_meas = gv.measure(sim, sim.state_0.body_q.numpy())
    base = snapshot_metrics(ctx)
    if base_path is None:
        ctx.base = base
    base = snapshot_metrics(ctx)
    logger.info(f"[TENSION] start: {short(base)}")
    dirs = span_dirs(ctx)
    logger.info(f"[TENSION] span dirs: {json.dumps(dirs)}")
    Xf0 = sim.ee_poses()[0]
    q_u0 = sim.arm_targets()[1]
    Xu0 = UrTracking.fk(q_u0)
    q_prev = [q_u0]
    df, du = np.asarray(dirs["franka"]["dir"]), np.asarray(dirs["ur"]["dir"])
    rows = [{"step": 0, "per_arm_mm": 0.0, **base}]
    snaps = {0: sim_snapshot.capture(sim, args.state_id)}
    Xf_prev, Xu_prev = Xf0, Xu0
    k = 0
    while k < args.max_steps and rows[-1]["d_seg_stretch_pct"] < args.target_stretch_pct:
        k += 1
        d = k * args.step_mm * 1e-3
        Xf1, Xu1 = Xf0.copy(), Xu0.copy()
        Xf1[:3, 3] += args.franka_share * 2.0 * d * df
        Xu1[:3, 3] += (1.0 - args.franka_share) * 2.0 * d * du
        ok, why = move_both(sim, Xf_prev, Xf1, Xu_prev, Xu1, args.move_s, args.hold_s, q_prev)
        Xf_prev, Xu_prev = Xf1, Xu1
        m = snapshot_metrics(ctx)
        X_f, X_u = sim.ee_poses()
        bad = ok_engaged(m, args, args.step_min_wrap_deg) + ([] if ok else [why])
        row = {"step": k, "per_arm_mm": k * args.step_mm,
               "franka_cmd_mm": ((Xf1[:3, 3] - Xf0[:3, 3]) * 1e3).tolist(),
               "ur_cmd_mm": ((Xu1[:3, 3] - Xu0[:3, 3]) * 1e3).tolist(),
               "franka_meas_mm": ((X_f[:3, 3] - Xf0[:3, 3]) * 1e3).tolist(),
               "ur_meas_mm": ((X_u[:3, 3] - Xu0[:3, 3]) * 1e3).tolist(), **m, "bad": bad}
        rows.append(row)
        logger.info(f"[TENSION] step {k} (+{2 * k * args.step_mm:g} mm total): {short(m)}"
                    + (f" | BAD {bad}" if bad else ""))
        if bad:
            break
        snaps[k] = sim_snapshot.capture(sim, args.state_id)
        if m["d_seg_stretch_pct"] >= args.target_stretch_pct:
            break
    good_steps = [r["step"] for r in rows if r["step"] in snaps
                  and (r["step"] > 0 or base_path is not None)
                  and r["d_seg_stretch_pct"] >= args.min_stretch_pct]
    verify, chosen = [], None
    for s in sorted(good_steps, reverse=True):
        sim.restore(snaps[s], settle_steps=0)
        passed, trace = hold_verify(ctx, args, f"step {s}")
        verify.append({"step": s, "passed": passed, "trace": trace})
        if passed:
            chosen = s
            break
    if chosen is None:
        raise RuntimeError(f"no tension step verified (rows {[(r['step'], r.get('bad')) for r in rows]})")
    final = snapshot_metrics(ctx)
    origin = ("flat-hold demo final state" if base_path is None
              else f"raised-UR state {start_path.name} (from {args.raise_from})")
    notes = (f"{args.state_id}: {origin}, both grasps moved "
             f"{chosen * args.step_mm:g} mm each (Franka share {args.franka_share:g}) away from "
             f"the large pulley along their free spans, then {args.verify_s:g} s under magna OSC "
             f"hold; {short(final)}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = sim_snapshot.save(sim_snapshot.capture(sim, args.state_id, notes=notes),
                             args.out / f"{args.state_id}_osc.npz")
    grasp = sim.restore(sim_snapshot.load(path), settle_steps=round(0.5 / dt))
    recheck = snapshot_metrics(ctx)
    logger.info(f"[TENSION] recheck (restore + 0.5 s): {short(recheck)}")
    bad = [e for e in gv.OSC_LOG_ERRORS if e in sim.osc.log_text()]
    if bad:
        raise RuntimeError(f"OSC log has {bad}")
    ref = start_path
    index = {
        "set": args.out.name, "kind": "flat_engaged",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "nominal_start": str(ref), "nominal_start_sha256": gv.sha256(ref),
        "method": "driven: demo final -> both grasps stepped apart along the free spans "
                  "(Franka OSC min-jerk Cartesian target, UR per-step IK line), hold-at-"
                  "measured verify",
        "stretch_metric": {
            "seg_stretch_pct": "body-centre length of the Franka grasp -> pulley -> UR grasp "
                               "segment vs the same bodies at rest (criteria use its increase "
                               "d_seg over the demo final)",
            "centre_loop_len_pct": "48-body centre loop length vs the rest ellipse",
            "stretch_pct": "sum of rod-joint axial separations / sum of rigid capsule "
                           "lengths; tension_n = joint target_ke x axial separation"},
        "criteria": {"min_wrap_deg": args.min_wrap_deg, "max_h_mm": args.max_h_mm,
                     "min_stretch_pct": args.min_stretch_pct,
                     "target_stretch_pct": args.target_stretch_pct,
                     **({} if args.step_min_wrap_deg is None
                        else {"step_min_wrap_deg": args.step_min_wrap_deg})},
        "step_mm": args.step_mm, "franka_share": args.franka_share, "move_s": args.move_s,
        "hold_s": args.hold_s, "verify_s": args.verify_s, "span_dirs": dirs,
        "rod": {"rigid_len_mm": ctx.gauge.rigid_len * 1e3, "joints": len(ctx.gauge.joints),
                "stretch_ke_n_per_m": float(np.median(ctx.gauge.ke))},
        "steps": [{k2: v for k2, v in r.items() if k2 != "tension_n"} for r in rows],
        "verify": verify,
        "variants": [{"id": args.state_id, "file": path.name, "held": all(recheck["held"]),
                      "commanded": {"per_arm_mm": chosen * args.step_mm,
                                    "franka_share": args.franka_share},
                      "measured": {k2: v for k2, v in final.items() if k2 != "tension_n"},
                      "recheck_0p5s": {k2: v for k2, v in recheck.items() if k2 != "tension_n"},
                      "reason": None, "notes": notes}],
    }
    offsets = demo_offsets(args)
    if offsets is not None:
        index["waypoint_offsets"] = offsets
        index["variants"][0]["commanded"]["waypoint_offsets"] = offsets
    if raise_info is not None:
        index["stretch_baseline"] = str(base_path)
        index["raise"] = raise_info
        index["variants"][0]["commanded"]["ur_raise_mm"] = raise_info["ur_raise_mm"]
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    if not all(grasp.held()):
        raise RuntimeError(f"recheck lost the grasp: {grasp.describe()}")
    logger.success(f"[TENSION] step {chosen} -> {path}")
    return path


# --- phase: raise (opt-in) ------------------------------------------------------------------------


def phase_raise(sim, ctx: Ctx, args) -> bool:
    """From a seated state, step the UR straight up (IK line, Franka held) until the fingertip
    reaches the groove z; keep the last step that stays seated without jaw slip."""
    if not ctx.jaw:
        raise ValueError("--phases raise needs --ur-jaw")
    dt = sim.frame_dt
    sim.restore(sim_snapshot.load(args.raise_base), settle_steps=round(args.settle_s / dt))
    ctx.ref_meas, ctx.base = None, None
    ctx.base = snapshot_metrics(ctx)
    sim.restore(sim_snapshot.load(args.raise_from), settle_steps=round(args.settle_s / dt))
    ctx.ref_meas = gv.measure(sim, sim.state_0.body_q.numpy())
    m0 = snapshot_metrics(ctx)
    jaw0 = m0["ur_jaw"]["depth_mm"]
    logger.info(f"[RAISE] start: {short(m0)}")
    Xf = sim.ee_poses()[0]
    q_u0 = sim.arm_targets()[1]
    Xu0 = UrTracking.fk(q_u0)
    X_u0 = sim.ee_poses()[1]
    q_prev = [q_u0]
    keep = ("outcome", "wrap_deg", "h_median_mm", "tilt_deg", "tilt_dir", "held",
            "ur_board_clearance_mm", "franka_tip_clearance_mm", "seg_stretch_pct",
            "d_seg_stretch_pct", "fingertip_z_mm", "groove_z_mm", "tip_minus_groove_mm",
            "grasp_slide_mm", "ee_franka", "ee_ur")

    def row_of(k, m, bad):
        X_f, X_u = sim.ee_poses()
        return {"step": k, "ur_cmd_dz_mm": k * args.raise_step_mm,
                "ur_meas_mm": ((X_u[:3, 3] - X_u0[:3, 3]) * 1e3).tolist(),
                "franka_meas_mm": ((X_f[:3, 3] - Xf[:3, 3]) * 1e3).tolist(),
                "jaw_depth_mm": m["ur_jaw"]["depth_mm"],
                "jaw_slip_mm": m["ur_jaw"]["depth_mm"] - jaw0,
                **{k2: m.get(k2) for k2 in keep}, "bad": bad}

    rows = [row_of(0, m0, ok_engaged(m0, args))]
    if rows[0]["bad"]:
        raise RuntimeError(f"raise start not seated: {rows[0]['bad']}")
    snaps, good = {0: sim_snapshot.capture(sim, args.state_id)}, 0
    reached = abs(m0["tip_minus_groove_mm"]) <= args.tip_groove_tol
    Xu_prev = Xu0
    for k in range(1, args.raise_max_steps + 1):
        if reached:
            break
        Xu1 = Xu0.copy()
        Xu1[2, 3] += k * args.raise_step_mm * 1e-3
        ok, why = move_both(sim, Xf, Xf, Xu_prev, Xu1, args.move_s, args.hold_s, q_prev)
        Xu_prev = Xu1
        m = snapshot_metrics(ctx)
        slip = max(abs(m["ur_jaw"]["depth_mm"] - jaw0), abs(m["grasp_slide_mm"]["ur"]))
        bad = ok_engaged(m, args) + ([] if ok else [why]) + (
            [f"jaw slip {slip:.2f} mm"] if slip > args.max_slip_mm else [])
        rows.append(row_of(k, m, bad))
        logger.info(f"[RAISE] step {k} (+{k * args.raise_step_mm:g} mm): {short(m)}"
                    + (f" | BAD {bad}" if bad else ""))
        if bad:
            break
        snaps[k], good = sim_snapshot.capture(sim, args.state_id), k
        reached = abs(m["tip_minus_groove_mm"]) <= args.tip_groove_tol
    sim.restore(snaps[good], settle_steps=0)
    args.target.mkdir(parents=True, exist_ok=True)
    sim_snapshot.save(snaps[good], args.target / "raise_final_osc.npz")
    for k, snap in snaps.items():
        sim_snapshot.save(snap, args.target / "raise_steps" / f"step_{k:02d}_osc.npz")
    offs = None
    src = args.raise_from.parents[0] / "index.json"
    if src.exists():
        offs = json.loads(src.read_text()).get("waypoint_offsets")
    tip = next(r for r in rows if r["step"] == good)["tip_minus_groove_mm"]
    payload = {"raise_from": str(args.raise_from), "raise_from_sha256": gv.sha256(args.raise_from),
               "stretch_baseline": str(args.raise_base), "waypoint_offsets": offs,
               "ur_raise_mm": good * args.raise_step_mm, "reached": bool(reached),
               "picked_step": good, "tip_minus_groove_mm": tip,
               "criteria": {"min_wrap_deg": args.min_wrap_deg, "max_h_mm": args.max_h_mm,
                            "max_slip_mm": args.max_slip_mm,
                            "tip_groove_tol_mm": args.tip_groove_tol},
               "step_mm": args.raise_step_mm, "move_s": args.move_s, "hold_s": args.hold_s,
               "franka": "held at its measured start pose (no correction)", "rows": rows}
    (args.target / "raise.json").write_text(json.dumps(payload, indent=2) + "\n")
    (logger.success if reached else logger.warning)(
        f"[RAISE] step {good} (+{good * args.raise_step_mm:g} mm), tip - groove {tip:+.2f} mm, "
        f"reached {reached}")
    return bool(reached)


# --- phase: observe -------------------------------------------------------------------------------


def observation(ctx: Ctx) -> dict:
    """The collector's frame: cropped camera cloud, material belt points, measured 40-dim state."""
    sim = ctx.sim
    joint_q = sim.state_0.joint_q.numpy()
    q_f, q_u = sim.arm_positions()
    v_f, v_u = sim.arm_velocities()
    ee_f = sim.franka_measured_pose7(joint_q)
    ee_u = pose7_mat(UrTracking.fk(joint_q[sim.arm_coords()[1]]))
    body_q = sim.state_0.body_q.numpy()
    belt = body_q[sim.info.belt_bodies, :3].astype(np.float64)
    xyz, rgb = sim.point_cloud()
    return {"pcd": lcs.camera_points(xyz), "pcd_rgb": np.asarray(rgb),
            "pcd_belt": lcs.belt_points_ordered(belt),
            "state": lcs.state_vector(q_f, q_u, v_f, v_u, ee_f, ee_u),
            "ee_franka": ee_f, "ee_ur": ee_u, "belt_xyz": belt.astype(np.float32),
            "pulley_large_pose": lcs.pose_from_xyz_xyzw(pulley_pose(sim, body_q)),
            "body_q": body_q.astype(np.float32)}


def train_latents(enc: LatentEncoder, split: Path) -> tuple[np.ndarray, list]:
    files = json.loads(split.read_text())["train"]
    zs, ids = [], []
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            z = enc.encode_batch(list(d["pcd"]), d["state"], d["pcd_belt"])
            outcome = str(d["sim_outcome"]) if "sim_outcome" in d.files else "?"
        zs.append(z)
        ids += [(f, t, outcome) for t in range(len(z))]
    return np.concatenate(zs), ids


COMPARE_KEYS = ("outcome", "wrap_deg", "h_median_mm", "tilt_deg", "tilt_dir", "seg_stretch_pct",
                "centre_loop_len_pct", "stretch_pct", "tension_n_seated_mean", "held",
                "ur_board_clearance_mm", "franka_tip_clearance_mm", "ee_franka", "ee_ur")


def compare_to(sim, target: Path, obs: dict, m: dict, z: np.ndarray, model) -> dict:
    """This observation vs another target dir's (latent, belt, metrics, UR jaw depth)."""
    with np.load(target / "observation.npz", allow_pickle=True) as d:
        z_o, belt_o = d["z_target"], d["pcd_belt"].astype(np.float64)
        m_o, body_q_o = json.loads(str(d["metrics_json"])), d["body_q"]
    pb = obs["pcd_belt"].astype(np.float64)
    return {"target": str(target), "dist_whitened": model.whitened_dist(z, z_o),
            "belt_rmse_mm": float(np.sqrt(((pb - belt_o) ** 2).sum(1).mean()) * 1e3),
            "ur_jaw": {"this": gv.ur_jaw_depth(sim, obs["body_q"]),
                       "other": gv.ur_jaw_depth(sim, body_q_o)},
            "metrics": {k: {"this": m.get(k), "other": m_o.get(k)} for k in COMPARE_KEYS}}


def phase_observe(sim, ctx: Ctx, args) -> Path:
    dt = sim.frame_dt
    path = args.out / f"{args.state_id}_osc.npz"
    _, base_path = start_and_base(args)
    sim.restore(sim_snapshot.load(base_path or args.target / "demo_final_osc.npz"),
                settle_steps=round(args.settle_s / dt))
    ctx.ref_meas = gv.measure(sim, sim.state_0.body_q.numpy())
    ctx.base = None
    ctx.base = snapshot_metrics(ctx)
    sim.restore(sim_snapshot.load(path), settle_steps=round(0.5 / dt))
    m_eval = snapshot_metrics(ctx)  # what the eval sees after its 0.5 s settle
    logger.info(f"[OBSERVE] restore + 0.5 s: {short(m_eval)}")
    # The restore transient relaxes over ~3 s; observe the settled state.
    sim.settle(round(max(0.0, args.observe_settle_s - 0.5) / dt))
    obs = observation(ctx)
    m = snapshot_metrics(ctx)
    logger.info(f"[OBSERVE] {short(m)}; pcd {obs['pcd'].shape}")
    enc = LatentEncoder.load(args.deploy)
    model = LearnedLcs.load(args.deploy)
    z = enc.encode(obs["pcd"], obs["state"], obs["pcd_belt"])
    with np.load(FLAT_DEMO, allow_pickle=True) as d:
        t_last = len(d["state"]) - 1
        z_demo = enc.encode(d["pcd"][t_last], d["state"][t_last], d["pcd_belt"][t_last])
        demo_belt = d["pcd_belt"][t_last].astype(np.float64)
    dist_demo = model.whitened_dist(z, z_demo)
    dist_stage = [model.whitened_dist(z, g) for g in model.stage_goals]
    t0 = time.perf_counter()
    Z, ids = train_latents(enc, args.split)
    w = np.linalg.norm((Z - z) / model.z_std, axis=1)
    order = np.argsort(w)[:args.k_nearest]
    w_demo = np.linalg.norm((Z - z_demo) / model.z_std, axis=1)
    nearest = [{"file": ids[i][0], "frame": ids[i][1], "outcome": ids[i][2],
                "dist": float(w[i])} for i in order]
    logger.info(f"[OBSERVE] encoded {len(Z)} train frames in {time.perf_counter() - t0:.0f} s")
    pb = obs["pcd_belt"].astype(np.float64)
    report = {
        "state_file": str(path), "state_sha256": gv.sha256(path), "deploy": str(args.deploy),
        "deploy_sha256": gv.sha256(args.deploy), "observe_settle_s": args.observe_settle_s,
        "metrics": m, "metrics_restore_0p5s": m_eval, "demo_final_metrics": ctx.base,
        "z_target": z.tolist(),
        "dist_whitened": {
            "to_flat_demo_final": dist_demo,
            "to_deploy_stage_goals": dist_stage,
            "goal_tol_whitened": model.goal_tol,
            "nearest_train": nearest,
            "nearest_train_p50_of_k": float(np.median(w[order])),
            "flat_demo_final_nearest_train": float(w_demo.min()),
            "n_train_frames": len(Z), "split": str(args.split),
            "ood_note": "compare nearest_train[0].dist with flat_demo_final_nearest_train "
                        "(the demo ending is itself held-UR, out of the training set)",
        },
        "belt_rmse_vs_flat_demo_final_mm": float(np.sqrt(((pb - demo_belt) ** 2).sum(1).mean())
                                                 * 1e3),
    }
    if demo_offsets(args) is not None:
        report["waypoint_offsets"] = demo_offsets(args)
    if base_path is not None:
        r = json.loads((args.target / "raise.json").read_text())
        report["raise"] = {k: r.get(k) for k in ("raise_from", "picked_step", "reached",
                                                  "tension_raise_step", "tension_attempts",
                                                  "step_mm")}
    if args.compare_to:
        cmp = [compare_to(sim, t, obs, m, z, model) for t in args.compare_to]
        report["compare"] = cmp[0] if len(cmp) == 1 else cmp
    args.target.mkdir(parents=True, exist_ok=True)
    out = args.target / "observation.npz"
    np.savez(out, pcd=obs["pcd"], pcd_rgb=obs["pcd_rgb"], pcd_belt=obs["pcd_belt"],
             state=obs["state"], ee_franka=obs["ee_franka"], ee_ur=obs["ee_ur"],
             belt_xyz=obs["belt_xyz"], pulley_large_pose=obs["pulley_large_pose"],
             body_q=obs["body_q"], z_target=z, z_flat_demo_final=z_demo,
             z_std=model.z_std, metrics_json=np.asarray(json.dumps(m)),
             report_json=np.asarray(json.dumps(report)),
             deploy=np.asarray(str(args.deploy)), state_file=np.asarray(str(path)))
    (args.target / "observation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    cmps = report.get("compare", [])
    for c in [cmps] if isinstance(cmps, dict) else cmps:
        logger.info(f"[OBSERVE] vs {c['target']}: z dist {c['dist_whitened']:.3f}, belt rmse "
                    f"{c['belt_rmse_mm']:.1f} mm, jaw depth {c['ur_jaw']['this']['depth_mm']:.1f}"
                    f" vs {c['ur_jaw']['other']['depth_mm']:.1f} mm")
    logger.info(f"[OBSERVE] z dist: flat demo final {dist_demo:.3f}, stage goals "
                f"{np.round(dist_stage, 3).tolist()}, nearest train {nearest[0]['dist']:.3f} "
                f"(demo final's nearest {w_demo.min():.3f}) -> {out}")
    if args.record_s > 0:
        sim.restore(sim_snapshot.load(path), settle_steps=0)
        name = f"{args.state_id}-hold"
        sim.start_recording(args.target / "recordings", name,
                            extra_meta={"synthetic_target": str(path)})
        eo.run_steps(sim, round(args.record_s / dt))
        rec = col._finish_recording(sim, args.target, name, "hold done")
        logger.info(f"[OBSERVE] recording {rec}; after hold: {short(snapshot_metrics(ctx))}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lcm-url", required=True, help="private LCM URL for the OSC")
    parser.add_argument("--phases", default="demo,tension,observe")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--params", type=Path, default=MAGNA_PARAMS_SIM_YAML)
    parser.add_argument("--deploy", type=Path, default=DEFAULT_DEPLOY)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--demo-tries", type=int, default=3)
    parser.add_argument("--step-mm", type=float, default=1.0, help="per arm, per step")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--franka-share", type=float, default=0.5,
                        help="fraction of the separation moved by the Franka")
    parser.add_argument("--move-s", type=float, default=0.6)
    parser.add_argument("--hold-s", type=float, default=0.6)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--verify-s", type=float, default=3.0)
    parser.add_argument("--record-s", type=float, default=3.0)
    parser.add_argument("--observe-settle-s", type=float, default=3.0,
                        help="hold after the restore before the observation")
    parser.add_argument("--min-wrap-deg", type=float, default=90.0)
    parser.add_argument("--max-h-mm", type=float, default=1.0)
    parser.add_argument("--min-stretch-pct", type=float, default=0.3,
                        help="min segment stretch increase over the demo final")
    parser.add_argument("--target-stretch-pct", type=float, default=1.0,
                        help="stop stepping at this segment stretch increase")
    parser.add_argument("--k-nearest", type=int, default=5)
    parser.add_argument("--start-state", type=Path, default=None,
                        help="demo replay start (default: the collector's osc pre_place_1)")
    parser.add_argument("--state-id", default=STATE_ID, help="saved state name / label")
    parser.add_argument("--step-min-wrap-deg", type=float, default=None,
                        help="wrap floor while stepping (default --min-wrap-deg; verify keeps it)")
    parser.add_argument("--ur-dz-mm", default="0", help="demo replay --nominal-ur-dz-mm")
    parser.add_argument("--ur-jaw", action="store_true",
                        help="add the UR jaw depth (belt vs 2F-85 fingertip) to the metrics")
    parser.add_argument("--compare-to", type=Path, nargs="+", default=None,
                        help="observe: compare with other target dirs' observation.npz")
    parser.add_argument("--franka-offset-mm", default=None, metavar="X,Y,Z[:X,Y,Z]",
                        help="demo replay --nominal-franka-offset-mm")
    parser.add_argument("--raise-from", type=Path, default=None,
                        help="raise phase: seated start state (enables raise-mode tension)")
    parser.add_argument("--raise-base", type=Path, default=None,
                        help="raise phase: untensioned state the stretch gain is measured from")
    parser.add_argument("--raise-step-mm", type=float, default=1.0)
    parser.add_argument("--raise-max-steps", type=int, default=30)
    parser.add_argument("--max-slip-mm", type=float, default=0.5,
                        help="raise phase: max UR jaw depth / slide change from the start")
    parser.add_argument("--tip-groove-tol", type=float, default=1.0,
                        help="raise phase: stop when |fingertip z - groove z| <= this (mm)")
    parser.add_argument("--raise-partial", action="store_true",
                        help="continue to tension/observe even if the groove was not reached")
    parser.add_argument("--tip-groove-tol-mm", type=float, default=None,
                        help="grid: also require |UR fingertip z - groove z| <= this (--ur-jaw)")
    parser.add_argument("--grid", default=None, metavar="UR_DZ[|FX,FY,FZ];...",
                        help="grid phase: waypoint-offset settings, --demo-tries replays each")
    args = parser.parse_args()
    gv.configure_logging()
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    from round_belt_task.osc_simulation import RoundBeltOscSimulation

    log_path = Path(tempfile.mkdtemp(prefix="make_flat_engaged_")) / "osc.log"
    sim = RoundBeltOscSimulation.build(lcm_url=args.lcm_url)
    sim.args.record_state_every = col.RECORD_STATE_EVERY
    reason = "error"
    try:
        warm = sim.start_osc(log_path)
        logger.info(f"OSC warm-up {warm:.2f} s, log {log_path}")
        ctx = build_ctx(sim, args.params)
        ctx.jaw = args.ur_jaw
        logger.info(f"rod: {len(ctx.gauge.joints)} joints, rigid length "
                    f"{ctx.gauge.rigid_len * 1e3:.2f} mm, ke {np.median(ctx.gauge.ke):g} N/m")
        if "raise" in phases and not phase_raise(sim, ctx, args) and not args.raise_partial:
            phases = [p for p in phases if p not in ("tension", "observe")]
            logger.warning("[RAISE] groove not reached: stopping before tension/observe")
        if "grid" in phases:
            phase_grid(sim, ctx, args)
        if "demo" in phases:
            phase_demo(sim, ctx, args)
        if "tension" in phases:
            phase_tension(sim, ctx, args)
        if "observe" in phases:
            phase_observe(sim, ctx, args)
        reason = "finished"
    finally:
        sim.close(reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
