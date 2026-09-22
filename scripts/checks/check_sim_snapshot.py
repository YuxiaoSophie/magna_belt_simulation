#!/usr/bin/env python3
"""Headless check that ``task_common.sim_snapshot`` restores a ``RoundBeltLcmSimulation`` state.

Builds the sim (private LCM group, null viewer, no cameras, not realtime), measures the sim's
own restore+rerun error on an unperturbed scene (S1), then requires a restored mid-motion state
(hand opening, Robotiq closing) to reproduce the original continuation in the same sim (S2) and
in a second, freshly built sim loaded from disk (S3); S4 checks the refusals, S5 the CUDA graph
and step/time bookkeeping. Checks share the sim's state and run in order.

Run:
    uv run python scripts/checks/check_sim_snapshot.py
    uv run python scripts/checks/check_sim_snapshot.py --keep
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# The task packages live under src/; make them importable regardless of CWD.
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import lcm
import newton.examples

import task_common  # noqa: F401  (puts lcmtypes/ on sys.path)
from robotiq import lcmt_robotiq_command
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from task_common import sim_snapshot

sys.path.insert(0, str(Path(__file__).parent))
from lcm_peer_utils import StatePeer, hand_command_msg

# A private multicast group so this check never disturbs a running magna stack.
PRIVATE_LCM_URL = "udpm://239.255.76.81:7681?ttl=0"
SETTLE_STEPS = 210
NOISE_STEPS = 200
IDLE_STEPS = 80
STROKE_STEPS = 20
CONTINUE_STEPS = 150
HAND_WIDTH = 0.040
ROBOT_TOL_MM = 0.5
ARM_TOL_RAD = 1e-3
BELT_TOL_MM = 2.0
BELT_NOISE_FACTOR = 3.0
HAND_WIDTH_TOL_MM = 0.2
RUNTIME_BUDGET_S = 90.0


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _run(ctx: SimpleNamespace, sim: RoundBeltLcmSimulation, n: int,
         hand_width: float | None = None) -> SimpleNamespace:
    """``n`` control steps (republishing the hand goal each step, as magna does); the result."""
    widths = np.empty(n)
    for i in range(n):
        if hand_width is not None:
            utime = round(sim.step_index * 1e6 * sim.frame_dt)
            ctx.peer.publish(sim.channels.franka_hand_input_channel,
                             hand_command_msg(hand_width, utime))
            ctx.peer.wait_delivered()
        sim.control_step()
        q = sim.state_0.joint_q.numpy()[ctx.hand_coords]
        widths[i] = float(-q[0] + q[1]) * 1e3
    return SimpleNamespace(step=sim.step_index, body_q=sim.state_0.body_q.numpy().copy(),
                           joint_q=sim.state_0.joint_q.numpy().copy(), widths=widths)


def _errors(ctx: SimpleNamespace, ref: SimpleNamespace, got: SimpleNamespace) -> dict:
    def max_mm(bodies: np.ndarray) -> float:
        delta = got.body_q[bodies, :3].astype(np.float64) - ref.body_q[bodies, :3]
        return float(np.linalg.norm(delta, axis=1).max()) * 1e3

    arm = np.abs(got.joint_q[ctx.arm_coords].astype(np.float64) - ref.joint_q[ctx.arm_coords])
    return {"robot_mm": max_mm(ctx.robot_bodies), "belt_mm": max_mm(ctx.belt_bodies),
            "arm_rad": float(arm.max()), "hand_mm": float(np.abs(got.widths - ref.widths).max())}


def _fmt(err: dict) -> str:
    return (f"robot {err['robot_mm']:.4f} mm, belt {err['belt_mm']:.4f} mm, arm "
            f"{err['arm_rad']:.2e} rad, hand width {err['hand_mm']:.4f} mm")


def _require_fidelity(ctx: SimpleNamespace, err: dict, what: str) -> None:
    belt_tol = max(BELT_TOL_MM, BELT_NOISE_FACTOR * ctx.noise_belt)
    _require(err["robot_mm"] <= ROBOT_TOL_MM,
             f"{what}: robot bodies {err['robot_mm']:.4f} mm > {ROBOT_TOL_MM} mm")
    _require(err["arm_rad"] <= ARM_TOL_RAD,
             f"{what}: arm joints {err['arm_rad']:.3g} rad > {ARM_TOL_RAD} rad")
    _require(err["belt_mm"] <= belt_tol,
             f"{what}: belt bodies {err['belt_mm']:.4f} mm > {belt_tol:.4f} mm")
    _require(err["hand_mm"] <= HAND_WIDTH_TOL_MM,
             f"{what}: hand width {err['hand_mm']:.4f} mm > {HAND_WIDTH_TOL_MM} mm at some step")


def _publish_robotiq(ctx: SimpleNamespace, sim: RoundBeltLcmSimulation, position: int) -> None:
    cmd = lcmt_robotiq_command()
    cmd.utime = round(sim.step_index * 1e6 * sim.frame_dt)
    cmd.position, cmd.speed, cmd.force = position, 255, 0
    ctx.robotiq_lc.publish(sim.channels.robotiq_command_channel, cmd.encode())
    time.sleep(0.020)  # no echo channel for Robotiq: a fixed delay stands in for wait_delivered


@check("S0 build + settle")
def check_s0(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    _run(ctx, sim, SETTLE_STEPS)
    _require(sim.step_index == SETTLE_STEPS, f"step_index {sim.step_index} != {SETTLE_STEPS}")
    _require(not sim.use_cuda_graph or sim.physics_graph is not None,
             "CUDA graph not captured after the settle")
    return (f"{SETTLE_STEPS} steps, cuda graph {'captured' if sim.physics_graph else 'off'}, "
            f"{len(ctx.robot_bodies)} robot / {len(ctx.belt_bodies)} belt bodies")


@check("S1 baseline noise")
def check_s1(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    ctx.a0 = sim_snapshot.capture(sim, "a0", notes="check_sim_snapshot S1")
    a1 = _run(ctx, sim, NOISE_STEPS)
    sim_snapshot.restore(sim, ctx.a0)
    a1_again = _run(ctx, sim, NOISE_STEPS)
    _require(a1_again.step == a1.step, f"rerun ended at step {a1_again.step} != {a1.step}")
    err = _errors(ctx, a1, a1_again)
    ctx.noise_belt = err["belt_mm"]
    return (f"noise_robot {err['robot_mm']:.4f} mm, noise_belt {err['belt_mm']:.4f} mm, "
            f"arm joint_q {err['arm_rad']:.2e} rad over {NOISE_STEPS} steps")


@check("S2 mid-motion fidelity")
def check_s2(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    sim_snapshot.restore(sim, ctx.a0)
    _publish_robotiq(ctx, sim, 255)
    _run(ctx, sim, IDLE_STEPS)
    _require(sim.robotiq_position_byte == 255, "Robotiq command 255 was not received")
    b0_run = _run(ctx, sim, STROKE_STEPS, HAND_WIDTH)
    ctx.b0 = sim_snapshot.capture(sim, "b0", notes="check_sim_snapshot S2 mid-stroke")
    _require(not np.array_equal(ctx.b0.hand_ramp_q, ctx.b0.hand_goal_q),
             f"hand ramp finished before the capture ({ctx.b0.hand_ramp_q.tolist()})")
    ctx.b1 = _run(ctx, sim, CONTINUE_STEPS, HAND_WIDTH)
    sim_snapshot.restore(sim, ctx.b0)
    b1_again = _run(ctx, sim, CONTINUE_STEPS, HAND_WIDTH)
    err = _errors(ctx, ctx.b1, b1_again)
    _require_fidelity(ctx, err, "S2")
    stroke = ctx.b1.widths[-1] - b0_run.widths[-1]
    return (f"B0 at step {ctx.b0.step_index}, hand width {b0_run.widths[-1]:.2f} mm (+"
            f"{stroke:.2f} mm to go); {_fmt(err)}")


@check("S3 second sim from disk")
def check_s3(ctx: SimpleNamespace) -> str:
    path = sim_snapshot.save(ctx.b0, ctx.tmp_root / "b0.npz")
    ctx.b0_path = path
    sim2 = RoundBeltLcmSimulation(ctx.viewer, ctx.args)
    ctx.sim2 = sim2
    sim_snapshot.restore(sim2, sim_snapshot.load(path))
    b1_other = _run(ctx, sim2, CONTINUE_STEPS, HAND_WIDTH)
    err = _errors(ctx, ctx.b1, b1_other)
    _require_fidelity(ctx, err, "S3")
    return f"{path.name} {path.stat().st_size / 1e3:.0f} kB; {_fmt(err)}"


def _refusal(sim: RoundBeltLcmSimulation, snap: sim_snapshot.SimSnapshot) -> str | None:
    """The ``ValueError`` message of ``restore``, None if it applied."""
    try:
        sim_snapshot.restore(sim, snap)
    except ValueError as exc:
        return str(exc)
    return None


@check("S4 refusal")
def check_s4(ctx: SimpleNamespace) -> str:
    sim = ctx.sim
    index = len(ctx.b0.meta["body_labels"]) // 2
    edited = sim_snapshot.load(ctx.b0_path)
    edited.meta["body_labels"][index] += "_edited"
    edited = sim_snapshot.load(sim_snapshot.save(edited, ctx.tmp_root / "b0_edited.npz"))
    truncated = sim_snapshot.load(ctx.b0_path)
    truncated.joint_q = truncated.joint_q[:-1]

    messages = []
    for what, snap, needle in (("edited body_labels", edited, f"body_labels[{index}]"),
                               ("truncated joint_q", truncated, "joint_q")):
        body_q = sim.state_0.body_q.numpy().copy()
        target_q = sim.control.joint_target_q.numpy().copy()
        step, graph = sim.step_index, sim.physics_graph
        message = _refusal(sim, snap)
        _require(message is not None and needle in message,
                 f"{what}: ValueError naming {needle!r} expected, got {message!r}")
        _require(np.array_equal(sim.state_0.body_q.numpy(), body_q),
                 f"{what}: body_q changed by the refused restore")
        _require(np.array_equal(sim.control.joint_target_q.numpy(), target_q),
                 f"{what}: joint_target_q changed by the refused restore")
        _require(sim.step_index == step and sim.physics_graph is graph,
                 f"{what}: step_index / physics_graph changed by the refused restore")
        messages.append(message)
    return "; ".join(messages)


@check("S5 graph + bookkeeping")
def check_s5(ctx: SimpleNamespace) -> str:
    sim, b0 = ctx.sim, ctx.b0
    sim_snapshot.restore(sim, b0)
    _require(sim.physics_graph is None, "physics_graph not dropped by restore")
    _require(sim.step_index == b0.step_index and sim.sim_time == b0.sim_time,
             f"step {sim.step_index} / t {sim.sim_time} != snapshot {b0.step_index} / "
             f"{b0.sim_time}")
    sim.control_step()
    _require(not sim.use_cuda_graph or sim.physics_graph is not None,
             "physics_graph not re-captured by the first step after restore")
    want_step = b0.step_index + 1
    _require(sim.step_index == want_step and sim.frame_id == want_step,
             f"step_index {sim.step_index} / frame_id {sim.frame_id} != {want_step}")
    want_time = want_step * sim.frame_dt
    _require(abs(sim.sim_time - want_time) < 1e-9, f"sim_time {sim.sim_time} != {want_time}")
    return (f"graph dropped, {'re-captured' if sim.physics_graph else 'off (no CUDA)'}; "
            f"step {b0.step_index} -> {sim.step_index}, t {sim.sim_time:.3f} s")


def main() -> int:
    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_sim_snapshot_"))

    parser = RoundBeltLcmSimulation.create_parser()
    parser.add_argument("--keep", action="store_true",
                        help="keep the temp snapshot dir instead of deleting it")
    parser.set_defaults(viewer="null", realtime=False, cameras=False, lcm_url=PRIVATE_LCM_URL)
    exit_code = 0
    ctx = SimpleNamespace(tmp_root=tmp_root)
    viewer = args = None
    try:
        viewer, args = newton.examples.init(parser)
        sim = RoundBeltLcmSimulation(viewer, args)
        io = sim.recorder_meta()["robot_io"]
        ctx.viewer, ctx.args, ctx.sim = viewer, args, sim
        ctx.hand_coords = np.asarray(io["franka_hand"]["coords"], dtype=np.int64)
        ctx.arm_coords = np.asarray(io["franka"]["coords"] + io["ur10"]["coords"],
                                    dtype=np.int64)
        ctx.robot_bodies = np.asarray(sim.info.robot_bodies, dtype=np.int64)
        ctx.belt_bodies = np.asarray(sim.info.belt_bodies, dtype=np.int64)
        ctx.peer = StatePeer(PRIVATE_LCM_URL, sim.channels)
        ctx.robotiq_lc = lcm.LCM(PRIVATE_LCM_URL)
        for name, fn in CHECKS:
            try:
                detail = fn(ctx)
            except AssertionError as exc:
                print(f"[FAIL] {name}: {exc}", file=sys.stderr)
                exit_code = 1
                break
            except Exception as exc:  # noqa: BLE001 - report, then still clean up
                print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
                traceback.print_exc()
                exit_code = 1
                break
            print(f"[PASS] {name}: {detail}")
    finally:
        if viewer is not None:
            viewer.close()
        if getattr(args, "keep", False):
            print(f"[INFO] kept {tmp_root}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        if runtime > RUNTIME_BUDGET_S:
            print(f"[WARN] runtime {runtime:.1f} s > {RUNTIME_BUDGET_S:g} s budget")
        print(f"ALL SNAPSHOT CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
