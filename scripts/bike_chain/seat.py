"""Scripted bimanual seat of the bike chain on the small sprocket (V13-05).

Replays magna's recorded Franka + UR10 demo (``pre_mpc_motion`` of
``assets/bike_chain_task/bike_chain_controller_params.yaml``, WORLD frame) through
:class:`BikeChainOscSimulation` (Franka on magna's OSC, UR position-driven via IK), labels the
chain with ``bike_chain_task.outcome`` and writes per-episode rows + ``summary.md`` + ``seat.html``.

Re-basing: the demo is world frame and the sim's board weld is the hardware calibration the demo
was recorded against (yaml ``task_board_position`` == weld), so the rigid re-base is identity. The
aligned board moved the small sprocket +3.00 mm along its axis: the UR walk targets get that shift.

flock data/lcs/realfirst/cmp/v12_79/sim.lock uv run --frozen python scripts/bike_chain/seat.py \
    --starts nominal --record --out data/bike_chain/v13_05/smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from loguru import logger

from bike_chain_task import outcome
from bike_chain_task.constants import _CHAIN, BIKE_CHAIN_ASSETS_DIR, X_W_BOARD, X_W_UR10
from bike_chain_task.directives import _CHAIN_SPEC
from bike_chain_task.osc_simulation import BikeChainOscSimulation
from round_belt_task.arm_kinematics import (
    X_WRIST3_TRACKING,
    UrdfChain,
    ik,
    model_pose_ur_tracking,
    rot_axis_angle,
    wp_transform_to_mat4,
)
from round_belt_task.commander import (
    CommanderParams,
    FrankaWaypointCommander,
    UrLineCommander,
    angular_distance,
    mat3_to_quat,
    quat_to_mat3,
    saved_traj_message,
    targets_from_waypoints,
)
from round_belt_task.constants import UR10_URDF
from round_belt_task.osc_bridge import OscTimeout
from round_belt_task.waypoints import load_params, load_pre_mpc_segment
from task_common import sim_snapshot

PARAMS = BIKE_CHAIN_ASSETS_DIR / "bike_chain_controller_params.yaml"
FIRST, LAST = "pre-pick-franka", "mar9_1555pm_ts_6051"  # facts file: pick 0-4, carry 5-17, walk 18-26
PHASES = (("pick", 0), ("carry", 5), ("walk", 18))
PICK = range(2, 4)  # the two at-holder waypoints
HOLD_BOTH = range(4, 14)  # from the lift until the Franka opens at 14
LIFT = 4
STACK_BOTTOM_M, STACK_MID_M = -0.00125, 0.00375  # inner-link frame: outer plate bottom / middle
SMALL_SHIFT_M = 0.003  # aligned board: small sprocket +3.00 mm along its axis
UR_IK_POS_TOL, UR_IK_ROT_TOL = 1e-4, 7.5e-4
UR_VELOCITY_LEAD = 1.0
FRANKA_HELD_MM = (6.0, 14.0)
LOST_DEBOUNCE_S = 0.5
SETTLE_S, HOLD_S, SAMPLE_S, RENDER_S = 1.0, 2.0, 0.1, 0.1
LAST_N = 15
SUITE_XY_MM, SUITE_YAW_DEG = 5.0, 5.0

UR_TRACKING = UrdfChain(UR10_URDF, "base_link", "wrist_3_link",
                        X_pre=wp_transform_to_mat4(X_W_UR10), X_post=X_WRIST3_TRACKING)


@dataclasses.dataclass(frozen=True)
class Start:
    name: str
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    yaw_deg: float = 0.0


def suite_starts(seed: int, kind: str) -> list[Start]:
    """Nominal, plus 8 draws: x or y shift of +-[1, SUITE_XY] mm and yaw of +-[1, SUITE_YAW] deg."""
    out = [Start("nominal")]
    if kind == "nominal":
        return out
    rng = np.random.default_rng(seed)
    for k in range(8):
        mag, yaw = rng.uniform(1.0, SUITE_XY_MM), rng.uniform(1.0, SUITE_YAW_DEG)
        sx, sy = (1 if k & 1 else -1), (1 if k & 2 else -1)  # every sign pair twice
        on_x = k < 4
        out.append(Start(f"s{k + 1}", sx * mag if on_x else 0.0, 0.0 if on_x else sx * mag,
                         sy * yaw))
    return out


def load_targets(args, labels_all: list[str]):
    """Commander targets ``first..last`` (world), adjusted by the named offsets; global indices."""
    waypoints = load_pre_mpc_segment(args.params, args.first, args.last)
    base = labels_all.index(args.first)
    small_axis = np.asarray(wp.quat_rotate(X_W_BOARD.q, wp.vec3(0.0, 0.0, 1.0)))  # board z
    for k, w in enumerate(waypoints):
        g = base + k
        dz = args.z_offset_mm
        w.franka_pos = w.franka_pos + [0.0, 0.0, dz * 1e-3]
        if w.ur_pos is not None:
            w.ur_pos = w.ur_pos + [0.0, 0.0, dz * 1e-3]
            if g >= PHASES[2][1]:
                w.ur_pos = w.ur_pos + small_axis * SMALL_SHIFT_M + [0, 0, args.ur_z_offset_mm * 1e-3]
        if args.hard_grip:
            if w.franka_gripper_mm is not None and w.franka_gripper_mm < FRANKA_HELD_MM[0]:
                w.franka_gripper_mm = 0.0
            if w.ur_gripper_byte is not None and w.ur_gripper_byte >= 230:
                w.ur_gripper_byte = 255
    f_targets, u_targets = targets_from_waypoints(waypoints)
    return f_targets, u_targets, [base + k for k in range(len(waypoints))]


def tool_geometry(sim) -> dict:
    """Pinch axis of each gripper in its tool frame (Franka finger_tip, UR tracking)."""
    body_q = sim.state_0.body_q.numpy().astype(np.float64)
    X_f, X_u = sim.ee_poses(body_q)
    fb = body_q[sim.info.franka_finger_bodies[:2], :3]
    pb = body_q[sim.info.gripper_pad_bodies[:2], :3]
    return {"franka_axis": X_f[:3, :3].T @ (fb[1] - fb[0]) / np.linalg.norm(fb[1] - fb[0]),
            "ur_axis": X_u[:3, :3].T @ (pb[1] - pb[0]) / np.linalg.norm(pb[1] - pb[0])}


def chain_pick(sim, ctx, f_targets, u_targets, gidx, pick_dz_m: float):
    """Pick waypoints (0..LIFT) moved onto the inner link nearest each arm's demo pick point: xy
    on the link midpoint, yaw turned so the pinch axis is normal to the link, the Franka tip at the
    stack bottom and the UR tracking point at the stack middle (+ ``pick_dz_m``)."""
    if PICK[0] not in gidx:
        return f_targets, u_targets, {}
    k2 = gidx.index(PICK[0])
    inner = sim.state_0.body_q.numpy().astype(np.float64)[np.asarray(sim.info.belt_bodies)[0::2]]
    rot = [quat_to_mat3([q[3], q[0], q[1], q[2]]) for q in inner[:, 3:7]]  # xyzw -> wxyz
    mid = np.array([p[:3] + r @ [0.0, outcome.PITCH_MM * 5e-4, 0.0] for p, r in zip(inner, rot)])
    geo = ctx["tool"]
    f_out, u_out, info = list(f_targets), list(u_targets), {}
    for arm, targets in (("franka", f_out), ("ur", u_out)):
        demo = targets[k2]
        i = int(np.argmin(np.linalg.norm(mid[:, :2] - demo.pos[:2], axis=1)))
        normal = np.cross([0.0, 0.0, 1.0], rot[i][:, 1])
        axis = quat_to_mat3(demo.quat_wxyz) @ geo[f"{arm}_axis"]
        yaw = math.atan2(axis[0] * normal[1] - axis[1] * normal[0], axis[:2] @ normal[:2])
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2  # a pinch axis has no sign
        Rz = rot_axis_angle([0.0, 0.0, 1.0], yaw)
        quat = mat3_to_quat(Rz @ quat_to_mat3(demo.quat_wxyz))
        # The demo's UR tracking point at the stack middle gave a 20-contact 2F-85 grip.
        z0 = STACK_BOTTOM_M if arm == "franka" else STACK_MID_M
        dz = pick_dz_m + (ctx.get("franka_pick_dz_m", 0.0) if arm == "franka" else 0.0)
        grip = np.array([*mid[i, :2], inner[i, 2] + z0 + dz])
        info[arm] = {"link": 2 * i, "dxy_mm": ((grip[:2] - demo.pos[:2]) * 1e3).round(2).tolist(),
                     "dyaw_deg": round(math.degrees(yaw), 2), "pick_z": round(float(grip[2]), 5)}
        for k, g in enumerate(gidx):
            if g > LIFT or targets[k] is None:
                continue
            z = grip[2] if g in PICK else targets[k].pos[2]
            targets[k] = dataclasses.replace(targets[k], pos=np.array([*grip[:2], z]),
                                             quat_wxyz=quat)
    return f_out, u_out, info


def phase_of(g: int) -> str:
    return next(name for name, i0 in reversed(PHASES) if g >= i0)


class Contacts:
    """Chain contacts with the Franka fingers / the 2F-85, from MuJoCo's contact buffer."""

    def __init__(self, sim) -> None:
        info = sim.info
        self.sim = sim
        self.shape_body = sim.model.shape_body.numpy()
        self.g2s = sim.solver.mjc_geom_to_newton_shape.numpy()[0]
        self.chain = np.asarray(info.belt_bodies)
        self.fingers = np.asarray(info.franka_finger_bodies)
        self.gripper = np.asarray(info.gripper_bodies)

    def __call__(self) -> dict:
        d = self.sim.solver.mjw_data
        n = int(d.nacon.numpy()[0])
        out = {"franka_n": 0, "franka_bodies": 0, "ur_n": 0, "ur_bodies": 0}
        if not n:
            return out
        shapes = self.g2s[d.contact.geom.numpy()[:n]]
        bodies = np.where(shapes >= 0, self.shape_body[np.maximum(shapes, 0)], -1)
        chain = np.isin(bodies, self.chain)
        for key, group in (("franka", self.fingers), ("ur", self.gripper)):
            robot = np.isin(bodies, group)
            hit = (chain[:, 0] & robot[:, 1]) | (chain[:, 1] & robot[:, 0])
            other = np.where(chain[:, 0], bodies[:, 1], bodies[:, 0])[hit]
            out[f"{key}_n"], out[f"{key}_bodies"] = int(hit.sum()), len(set(other.tolist()))
        return out


def chain_state(sim) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(rollers (64, 3), sprocket pose7 small/large/tensioner)``."""
    body_q = sim.state_0.body_q.numpy().astype(np.float64)
    inner = body_q[np.asarray(sim.info.belt_bodies)[0::2]]
    return (outcome.rollers_from_inner_bodies(inner),
            [body_q[b] for b in sim.info.pulley_bodies])


def pin_tilt(link_q: np.ndarray) -> tuple[float, float]:
    """(mean pin axis, max per-link pin axis) angle to world z [deg].  Replaces the SVD plane of
    the rollers, which reads a large tilt once the lifted ring folds in-plane toward a line."""
    z = Rotation.from_quat(link_q[:, 3:7]).apply([0.0, 0.0, 1.0])
    m = z.mean(axis=0)
    return (float(np.degrees(np.arccos(min(1.0, abs(m[2]) / np.linalg.norm(m))))),
            float(np.degrees(np.arccos(np.clip(np.abs(z[:, 2]), 0.0, 1.0))).max()))


def min_jerk(tau: float) -> float:
    """0 -> 1 with zero velocity and acceleration at both ends."""
    tau = min(max(tau, 0.0), 1.0)
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)


class SmoothFranka:
    """``--smooth-lift``: min-jerk position knots per waypoint from the lift on, each segment
    starting where (and when) the previous one ended, so every stop has zero velocity and
    acceleration; peak speed = the commander's linear speed."""

    def __init__(self, lin_speed: float, min_s: float = 0.5) -> None:
        self.lin_speed, self.min_s = lin_speed, min_s
        self.segs: list[tuple] = []  # (g, p0, p1, t0, T)

    def knots(self, g: int, target: np.ndarray, t: float, meas: np.ndarray,
              times: np.ndarray) -> np.ndarray:
        if not self.segs or self.segs[-1][0] != g:
            p0, t0 = (meas, t) if not self.segs else (
                self.segs[-1][2], max(t, self.segs[-1][3] + self.segs[-1][4]))
            T = max(1.875 * float(np.linalg.norm(target - p0)) / self.lin_speed, self.min_s)
            self.segs = [*self.segs[-1:], (g, np.asarray(p0, float), np.asarray(target, float),
                                           t0, T)]
        out = []
        for tk in times:
            seg = next((sg for sg in reversed(self.segs) if tk >= sg[3]), self.segs[0])
            out.append(seg[1] + (seg[2] - seg[1]) * min_jerk((tk - seg[3]) / seg[4]))
        return np.array(out)


def warp_line(line, t2: float) -> float:
    """Min-jerk time warp of a UR line: same endpoints and duration, zero end velocities."""
    span = line.t1 - line.t0
    return t2 if span <= 0.0 else line.t0 + span * min_jerk((t2 - line.t0) / span)


def labels_now(rollers, poses) -> dict:
    out = {}
    for name, pose in zip(("small", "large"), poses[:2]):
        label, meshed = outcome.classify_sprocket(rollers, pose, name)
        fm = outcome.sprocket_metrics(rollers, pose, name)
        out[name] = {"label": label, "meshed": None if math.isnan(meshed) else meshed,
                     "wrap_deg": float(fm.wrap_deg)}
    return out


class Recorder:
    """Optional viser: live view and/or one ``.viser`` per episode (``newton.viewer.ViewerViser``)."""

    def __init__(self, live_port: int | None, record: bool) -> None:
        self.viewer, self.t, self.on = None, 0.0, False
        if live_port is None and not record:
            return
        import newton.viewer

        from utils.viewer_patches import patch_viewer_shape_names, patch_viser_texture_material

        with socket.socket() as sock:
            if live_port is not None:  # viser silently moves to the next free port
                if sock.connect_ex(("127.0.0.1", live_port)) == 0:
                    raise SystemExit(f"--viewer-port {live_port} is in use")
                port = live_port
            else:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        patch_viewer_shape_names()
        patch_viser_texture_material()
        self.viewer = newton.viewer.ViewerViser(port=port, label="bike chain seat")
        self.record = record

    def begin(self, path: Path) -> None:
        if self.viewer is None or not self.record:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # shortcut: re-arms ViewerViser's private serializer per episode, until it has an API
        self.viewer._record_to_viser = str(path)
        self.viewer._serializer = self.viewer._server.get_scene_serializer()
        self.on = True

    def frame(self, sim) -> None:
        if self.viewer is None:
            return
        self.t += RENDER_S  # monotonic across episodes (a restore rewinds sim_time)
        self.viewer.begin_frame(self.t)
        self.viewer.log_state(sim.state_0)
        self.viewer.end_frame()

    def end(self) -> None:
        if self.on:
            self.viewer.save_recording()
            self.on = False


def run_episode(sim, ctx, start: Start, rep: int, ep_dir: Path, rec: Recorder) -> dict:
    t_wall = time.perf_counter()
    sim.restore(ctx["snap"], settle_steps=0)
    sim.reset_chain_pose((start.dx_mm * 1e-3, start.dy_mm * 1e-3), math.radians(start.yaw_deg))
    rec.begin(ep_dir / "recordings" / "seat.viser")
    every_render = round(RENDER_S / sim.frame_dt)
    every_sample = round(SAMPLE_S / sim.frame_dt)
    for k in range(round(SETTLE_S / sim.frame_dt)):
        sim.control_step()
        if k % every_render == 0:
            rec.frame(sim)
    rest = sim.chain_rest_metrics()
    f_targets, u_targets, gidx = ctx["targets"]
    pick_info = {}
    if ctx["chain_pick"]:
        f_targets, u_targets, pick_info = chain_pick(sim, ctx, f_targets, u_targets, gidx,
                                                     ctx["pick_dz_m"])
        logger.info(f"[SEAT] pick from chain: {pick_info}")
    hand0, byte0 = sim.gripper_commands()
    params = ctx["params"]
    franka = FrankaWaypointCommander(f_targets, params, hand_mm=hand0)
    ur = UrLineCommander(u_targets, params, byte=byte0)
    u_coords = sim.arm_coords()[1]
    state: dict = {"tick": None}
    smooth = SmoothFranka(params.lin_speed) if ctx.get("smooth_lift") else None
    grip_ref: dict = {}  # arm -> (link index, link-in-tool rotation) at the lift

    def hook(step, t, joint_q, body_q):
        ee = sim.franka_measured_pose7(joint_q)
        X_u = UR_TRACKING.fk(joint_q[u_coords])
        u_t = u_targets[franka.index]
        e_pos = 0.0 if u_t is None else float(np.linalg.norm(X_u[:3, 3] - u_t.pos))
        e_ori = 0.0 if u_t is None else angular_distance(mat3_to_quat(X_u[:3, :3]), u_t.quat_wxyz)
        cmd = franka.tick(t, ee[:3], ee[3:], e_pos, e_ori)
        if smooth is not None and gidx[cmd.target_index] >= LIFT:
            cmd = dataclasses.replace(cmd, knots_pos=smooth.knots(
                gidx[cmd.target_index], f_targets[cmd.target_index].pos, t, ee[:3], cmd.times))
        ur.on_latch(cmd.latched_index)
        state["tick"] = (cmd, ur.tick(t, X_u, u_targets[cmd.target_index]), ee, X_u, e_pos)
        return saved_traj_message(round(t * 1e6), cmd.knots_pos, cmd.knots_quat, cmd.times)

    samples, aborted, reason, fail_g = [], False, "", None
    q_ur = sim.arm_targets()[1]
    lead_gain = sim.arm_kd / sim.arm_ke / sim.frame_dt
    lead_line, done_at, lost_since, k = None, None, None, 0
    sync0 = None  # (Franka tip z, UR tracking z) at lift start, for --sync-lift
    max_steps = round(ctx["max_s"] / sim.frame_dt)
    hold_steps = round(ctx.get("hold_s", HOLD_S) / sim.frame_dt)
    sim.commander_hook = hook
    t_loop = time.perf_counter()
    try:
        while True:
            tick = state["tick"]
            if tick is not None:
                t_osc = sim.osc_time_s(sim.step_index + 1)
                if smooth is not None and gidx[tick[0].target_index] >= LIFT:
                    t_osc = warp_line(tick[1].line, t_osc)
                X = tick[1].pose_at(t_osc)
                if (ctx.get("sync_lift") or smooth is not None) and gidx[tick[0].target_index] == LIFT:
                    # UR z follows the Franka's measured rise: same start, end and height change.
                    sync0 = sync0 or (float(tick[2][2]), float(tick[3][2, 3]))
                    X = X.copy()
                    X[2, 3] = sync0[1] + float(tick[2][2]) - sync0[0]
                q_new, err_p, err_r, _ = ik(UR_TRACKING, X, q_ur, pos_tol=UR_IK_POS_TOL,
                                            rot_tol=UR_IK_ROT_TOL)
                if err_p > UR_IK_POS_TOL or err_r > UR_IK_ROT_TOL:
                    raise RuntimeError(f"ur IK missed ({err_p * 1e3:.2f} mm)")
                lead = UR_VELOCITY_LEAD * lead_gain if tick[1].line is lead_line else 0.0
                sim.set_ur_target(q_new + lead * (q_new - q_ur))
                q_ur, lead_line = q_new, tick[1].line
                sim.set_grippers(tick[0].hand_mm, tick[1].byte)
            sim.control_step()
            k += 1
            cmd, ur_cmd, ee, X_u, _ = state["tick"]
            g = gidx[cmd.target_index]
            if k % every_render == 0:
                rec.frame(sim)
            if k % every_sample == 0:
                rollers, poses = chain_state(sim)
                grasp = sim.grasp_state()
                c = ctx["contacts"]()
                lab = labels_now(rollers, poses)
                franka_held = (FRANKA_HELD_MM[0] <= grasp.franka_width_mm <= FRANKA_HELD_MM[1]
                               and c["franka_bodies"] >= 2)
                ur_held = c["ur_bodies"] >= 2
                body_q = sim.state_0.body_q.numpy().astype(np.float64)
                links = body_q[np.asarray(sim.info.belt_bodies)]
                tilt_mean, tilt_max = pin_tilt(links)
                in_grip = {}
                for arm, X in zip(("franka", "ur"), sim.ee_poses(body_q)):
                    i = grip_ref.get(arm, (int(np.argmin(np.linalg.norm(
                        links[:, :3] - X[:3, 3], axis=1))), None))[0]
                    rel = Rotation.from_matrix(X[:3, :3]).inv() * Rotation.from_quat(
                        links[i, 3:7])
                    if g >= LIFT and arm not in grip_ref:
                        grip_ref[arm] = (i, rel)
                    in_grip[arm] = (float(np.degrees((grip_ref[arm][1].inv() * rel).magnitude()))
                                    if arm in grip_ref else 0.0)
                samples.append({
                    "t": round(k * sim.frame_dt, 3), "wp": g, "phase": cmd.phase,
                    "wrap_small": lab["small"]["wrap_deg"], "meshed_small": lab["small"]["meshed"],
                    "label_small": lab["small"]["label"], "wrap_large": lab["large"]["wrap_deg"],
                    "label_large": lab["large"]["label"], "franka_z": float(ee[2]),
                    "ur_z": float(X_u[2, 3]), "franka_width_mm": grasp.franka_width_mm,
                    "ur_byte": ur_cmd.byte, "franka_contacts": c["franka_n"],
                    "ur_contacts": c["ur_n"], "franka_held": franka_held, "ur_held": ur_held,
                    "chain_min_z": float(rollers[:, 2].min()), "chain_tilt_deg": tilt_mean,
                    "pin_tilt_max_deg": tilt_max, "franka_link_rot_deg": in_grip["franka"],
                    "ur_link_rot_deg": in_grip["ur"],
                    "chain_com": rollers.mean(axis=0).round(4).tolist(),
                    "_rollers": rollers, "_poses": poses})
                if g in HOLD_BOTH and not (franka_held and ur_held):
                    lost_since = k if lost_since is None else lost_since
                    if (k - lost_since) * sim.frame_dt >= LOST_DEBOUNCE_S:
                        who = "+".join(n for n, h in (("franka", franka_held), ("ur", ur_held))
                                       if not h)
                        raise RuntimeError(f"grasp lost ({who}) at {cmd.phase}")
                else:
                    lost_since = None
            if done_at is None and cmd.phase == "done":
                done_at = k
            if done_at is not None and k >= done_at + hold_steps:
                break
            if done_at is None and k >= max_steps:
                raise RuntimeError(f"timeout at {cmd.phase} after {ctx['max_s']:g} s")
    except (RuntimeError, OscTimeout) as exc:
        aborted, reason = True, str(exc)
        fail_g = gidx[state["tick"][0].target_index] if state["tick"] else 0
        logger.warning(f"[SEAT] {start.name}_{rep}: abort: {reason}")
        if isinstance(exc, OscTimeout) or (sim.osc is not None and not sim.osc.alive()):
            ctx["restart_osc"] = True
    finally:
        sim.commander_hook = sim.make_hold_hook()
        rec.end()
    loop_wall = time.perf_counter() - t_loop
    tail = samples[-LAST_N:]
    if tail:
        rollers_T = [s["_rollers"] for s in tail]
        small = outcome.classify_sprocket_episode(rollers_T, [s["_poses"][0] for s in tail],
                                                  "small", last_n=LAST_N)
        large = outcome.classify_sprocket_episode(rollers_T, [s["_poses"][1] for s in tail],
                                                  "large", last_n=LAST_N)
    else:
        small = large = "none"
    last = samples[-1] if samples else {}
    ok_small = small == "engaged"
    if not aborted and not ok_small:
        reason, fail_g = f"small {small}", gidx[-1]
    row = {
        "start": start.name, "rep": rep, "dx_mm": round(start.dx_mm, 2),
        "dy_mm": round(start.dy_mm, 2), "yaw_deg": round(start.yaw_deg, 2),
        "label_small": small, "label_large": large, "success_small": ok_small,
        "success_both": ok_small and large == "engaged",
        "meshed_small": last.get("meshed_small"), "wrap_small_deg": last.get("wrap_small"),
        "aborted": aborted, "reason": reason,
        "fail_phase": None if fail_g is None else phase_of(fail_g),
        "fail_waypoint": None if fail_g is None else ctx["labels"][fail_g],
        "at_fail": {k2: last.get(k2) for k2 in ("franka_contacts", "ur_contacts",
                                                 "franka_width_mm", "chain_min_z", "chain_com",
                                                 "wrap_small", "wrap_large")},
        "rest_after_settle": rest, "pick": pick_info, "sim_s": round(k * sim.frame_dt, 2),
        "wall_s": round(time.perf_counter() - t_wall, 1),
        "rtf": round(k * sim.frame_dt / max(loop_wall, 1e-9), 3),
    }
    ep_dir.mkdir(parents=True, exist_ok=True)
    series = [{k2: v for k2, v in s.items() if not k2.startswith("_")} for s in samples]
    (ep_dir / "episode.json").write_text(json.dumps({**row, "samples": series}))
    logger.info(f"[SEAT] {start.name}_{rep}: small {small} large {large} "
                f"{'ABORT ' + reason if aborted else ''} sim {row['sim_s']} s wall "
                f"{row['wall_s']} s rtf {row['rtf']}")
    return row


def write_summary(out: Path, rows: list[dict], meta: dict) -> None:
    n = len(rows)
    ks, kb = sum(r["success_small"] for r in rows), sum(r["success_both"] for r in rows)
    lines = [f"# V13-05 scripted seat: {out.name}", "",
             f"success_small **{ks}/{n}**, success_both **{kb}/{n}**", "",
             f"flags: `{json.dumps(meta['flags'])}`", "",
             ("`chain_tilt_deg` = angle of the mean pin axis (link z) to world z (V13-05; was "
              "the SVD plane of the rollers, which misreads an in-plane-folded ring)."), "",
             ("| start | dx mm | dy mm | yaw deg | small | large | meshed | wrap deg | aborted "
              "| fail phase | waypoint | reason | F/U contacts peak (at end) | sim s | wall s | rtf |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    series = {f"{r['start']}_{r['rep']}": json.loads(
        (out / f"{r['start']}_{r['rep']}" / "episode.json").read_text())["samples"] for r in rows}
    for r in rows:
        a, ser = r["at_fail"], series[f"{r['start']}_{r['rep']}"]
        peak = "/".join(str(max((p[k] for p in ser), default=0))
                        for k in ("franka_contacts", "ur_contacts"))
        m = r["meshed_small"]
        w = r["wrap_small_deg"]
        lines.append(
            f"| {r['start']}_{r['rep']} | {r['dx_mm']} | {r['dy_mm']} | {r['yaw_deg']} | "
            f"{r['label_small']} | {r['label_large']} | {'' if m is None else f'{m:.2f}'} | "
            f"{'' if w is None else f'{w:.0f}'} | {r['aborted']} | {r['fail_phase'] or ''} | "
            f"{r['fail_waypoint'] or ''} | {r['reason']} | {peak} ({a.get('franka_contacts')}/"
            f"{a.get('ur_contacts')}) | {r['sim_s']} | {r['wall_s']} | {r['rtf']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "seat.html").write_text(HTML.replace("__TITLE__", f"V13-05 seat {out.name}").replace(
        "__ROWS__", json.dumps([{k: v for k, v in r.items() if k != "rest_after_settle"}
                                for r in rows])).replace("__DATA__", json.dumps(series)))


HTML = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<body style="font-family:sans-serif;max-width:1200px"><h2>__TITLE__</h2>
<table id="tab" border=1 cellspacing=0 cellpadding=3 style="font-size:12px"></table>
<p>Metric <select id="m"></select> episodes: <span id="eps"></span></p>
<svg id="plot" width="1150" height="380" style="border:1px solid #ccc"></svg>
<p style="font-size:12px">Hover a line for its episode; dotted verticals = waypoint changes of the
first ticked episode (hover for the label index).</p>
<script>
const R=__ROWS__,D=__DATA__,keys=Object.keys(D),cols=["#36c","#c33","#3a3","#c90","#93c","#099",
"#666","#e6a","#8a5","#05a"],on={};
const cs=["start","dx_mm","dy_mm","yaw_deg","label_small","label_large","success_small",
"success_both","meshed_small","wrap_small_deg","fail_phase","fail_waypoint","reason","sim_s","rtf"];
document.getElementById("tab").innerHTML="<tr>"+cs.map(c=>`<th>${c}</th>`).join("")+"</tr>"+
 R.map(r=>"<tr style='background:"+(r.success_small?"#dfd":"#fdd")+"'>"+cs.map(c=>{let v=r[c];
 if(typeof v=="number")v=+v.toFixed(3);return `<td>${v??""}</td>`}).join("")+"</tr>").join("");
const num=k=>keys.some(e=>D[e].some(p=>typeof p[k]=="number"||typeof p[k]=="boolean"));
const ms=[...new Set(keys.flatMap(k=>D[k].length?Object.keys(D[k][0]):[]))].filter(k=>k!="t"&&num(k));
const sel=document.getElementById("m");ms.forEach(m=>sel.add(new Option(m,m)));
sel.value=ms.includes("wrap_small")?"wrap_small":ms[0];
const es=document.getElementById("eps");keys.forEach((k,i)=>{on[k]=true;
 const l=document.createElement("label");l.style.color=cols[i%cols.length];
 l.innerHTML=`<input type=checkbox checked> ${k} `;l.firstChild.onchange=e=>{on[k]=e.target.checked;
 draw()};es.appendChild(l)});
function draw(){const m=sel.value,W=1150,H=380,P=45,s=document.getElementById("plot");
 const v=p=>typeof p[m]=="boolean"?+p[m]:p[m];let tx=0,lo=Infinity,hi=-Infinity;
 keys.filter(k=>on[k]).forEach(k=>D[k].forEach(p=>{tx=Math.max(tx,p.t);const y=v(p);
  if(y!==null&&isFinite(y)){lo=Math.min(lo,y);hi=Math.max(hi,y)}}));
 if(hi==lo){hi+=1;lo-=1}const X=t=>P+t/tx*(W-2*P),Y=y=>H-P-(y-lo)/(hi-lo)*(H-2*P);
 let g=`<text x=2 y=15 font-size=12>${hi.toPrecision(4)}</text><text x=2 y=${H-P} font-size=12>`+
  `${lo.toPrecision(4)}</text><text x=${W-P-20} y=${H-5} font-size=12>${tx.toFixed(1)} s</text>`;
 const f=keys.find(k=>on[k]);if(f){let w=null;D[f].forEach(p=>{if(p.wp!==w){w=p.wp;
  g+=`<line x1=${X(p.t)} x2=${X(p.t)} y1=${P/2} y2=${H-P} stroke=#bbb stroke-dasharray=2>`+
  `<title>waypoint ${w}</title></line>`}})}
 keys.forEach((k,i)=>{if(!on[k])return;const pts=D[k].filter(p=>v(p)!==null&&isFinite(v(p)))
  .map(p=>X(p.t).toFixed(1)+","+Y(v(p)).toFixed(1)).join(" ");
  g+=`<polyline points="${pts}" fill=none stroke="${cols[i%cols.length]}" stroke-width=1.5>`+
   `<title>${k}</title></polyline>`});s.innerHTML=g}
sel.onchange=draw;draw();
</script></body>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lcm-url", default="udpm://239.255.78.63:7863?ttl=0")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/bike_chain/v13_05/smoke")
    ap.add_argument("--params", type=Path, default=PARAMS)
    ap.add_argument("--first", default=FIRST)
    ap.add_argument("--last", default=LAST)
    ap.add_argument("--starts", choices=("nominal", "suite"), default="nominal")
    ap.add_argument("--only", nargs="*", default=None, help="run only these start names")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-s", type=float, default=300.0, help="sim-time cap per episode")
    ap.add_argument("--hold-s", type=float, default=HOLD_S, help="hold after the last waypoint")
    ap.add_argument("--sync-lift", action="store_true",
                    help="lift: UR z follows the Franka's measured rise (same timing and height)")
    ap.add_argument("--smooth-lift", action="store_true",
                    help="min-jerk profile per waypoint from the lift on, both arms (+ sync-lift)")
    ap.add_argument("--record", action="store_true", help="one .viser per episode")
    ap.add_argument("--viewer", choices=("viser",), default=None)
    ap.add_argument("--viewer-port", type=int, default=8085)
    ap.add_argument("--z-offset-mm", type=float, default=0.0, help="all waypoints, both arms")
    ap.add_argument("--ur-z-offset-mm", type=float, default=0.0, help="UR walk targets (18-26)")
    ap.add_argument("--pick-z-offset-mm", type=float, default=0.0, help="holder pick (2-3)")
    ap.add_argument("--franka-pick-z-offset-mm", type=float, default=0.0,
                    help="holder pick (2-3), Franka only (< 0: deeper)")
    ap.add_argument("--chain-pick", action=argparse.BooleanOptionalAction, default=True,
                    help="pick waypoints from the settled chain's links (else demo verbatim)")
    ap.add_argument("--hard-grip", action="store_true",
                    help="closed commands -> Franka 0 mm, Robotiq 255 (more squeeze)")
    ap.add_argument("--diag-impratio", type=float, default=None,
                    help="DIAGNOSTIC ONLY: override the solver impratio (not a tuning knob)")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="commander speeds")
    args = ap.parse_args()
    if "7667" in args.lcm_url:
        raise SystemExit("never 7667")
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<level>{level: <7}</level> | {message}")

    labels_all = [r["label"] for r in load_params(args.params)["pre_mpc_motion"]]
    y = load_params(args.params)
    params = CommanderParams(
        pos_tol=y["predefined_motion_position_tolerance"],
        ori_tol=y["predefined_motion_orientation_tolerance"],
        lin_speed=y["predefined_motion_linear_speed"] * args.speed_scale,
        ang_speed=y["predefined_motion_angular_speed"] * args.speed_scale,
        ur_lin_speed=y["ur_predefined_motion_linear_speed"] * args.speed_scale,
        ur_ang_speed=y["ur_predefined_motion_angular_speed"] * args.speed_scale)
    starts = suite_starts(args.seed, args.starts)
    if args.only:
        starts = [s for s in starts if s.name in args.only]
    rec = Recorder(args.viewer_port if args.viewer else None, args.record)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.diag_impratio is not None:
        BikeChainOscSimulation.solver_kwargs = {**BikeChainOscSimulation.solver_kwargs,
                                                "impratio": args.diag_impratio}
    sim = BikeChainOscSimulation.build(lcm_url=args.lcm_url, viewer=rec.viewer)
    rows = []
    try:
        warm = sim.start_osc(out / "osc.log")
        X_meas = model_pose_ur_tracking(sim.state_0.body_q.numpy(), sim._ur_wrist_body)
        err = np.linalg.norm(UR_TRACKING.fk(sim.arm_positions()[1])[:3, 3] - X_meas[:3, 3])
        assert err < 1e-4, f"UR tracking FK {err * 1e3:.3f} mm off the model"
        sim.set_grippers(None, 0)
        sim.settle(100)
        ctx = {"snap": sim_snapshot.capture(sim, "v13_05-start"), "params": params,
               "targets": load_targets(args, labels_all), "labels": labels_all,
               "contacts": Contacts(sim), "max_s": args.max_s, "hold_s": args.hold_s, "tool": tool_geometry(sim),
               "chain_pick": args.chain_pick, "pick_dz_m": args.pick_z_offset_mm * 1e-3,
               "franka_pick_dz_m": args.franka_pick_z_offset_mm * 1e-3, "sync_lift": args.sync_lift,
               "smooth_lift": args.smooth_lift}
        logger.info(f"[SEAT] tool geometry {ctx['tool']}")
        flags = {k: v for k, v in vars(args).items() if k in (
            "first", "last", "starts", "seed", "repeats", "max_s", "z_offset_mm", "ur_z_offset_mm",
            "pick_z_offset_mm", "franka_pick_z_offset_mm", "sync_lift", "smooth_lift", "speed_scale", "hard_grip", "chain_pick", "diag_impratio")}
        meta = {"flags": flags, "osc_warm_up_s": warm, "lcm_url": args.lcm_url,
                "rebase": "identity (board weld == demo calibration) + small +3.00 mm on UR walk",
                "pin": {k: _CHAIN.get(k, _CHAIN_SPEC[k]) for k in ("damping", "frictionloss")}}
        (out / "run.json").write_text(json.dumps(meta, indent=1))
        rows_path = out / "episodes.jsonl"
        rows_path.write_text("")
        for start in starts:
            for rep in range(args.repeats):
                if ctx.pop("restart_osc", False):
                    sim.stop_osc()
                    sim.start_osc(out / f"osc_{start.name}_{rep}.log")
                row = run_episode(sim, ctx, start, rep, out / f"{start.name}_{rep}", rec)
                rows.append(row)
                with rows_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                write_summary(out, rows, meta)
    finally:
        sim.close("seat done")
    print((out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
