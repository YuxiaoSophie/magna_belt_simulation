#!/usr/bin/env python3
"""Live check of the LATENT_STATE encoder node and the LCM sim's opt-in perception publishers.

L0 the three vendored perception lcmtypes; L1 the sim's ``POINT_CLOUD_CROPPED`` /
``RoundBeltState`` publishers (in-process sim, bit-exact vs the collector's cloud, belt back to
world; the default ``--test`` sim publishes neither); L2 ``scripts/lcs/latent_encoder_node.py``
as a subprocess against the same sim (one latent per cloud, bit-exact ``z``, FK/proprio, rate,
latency); L3 the hardware mappings on synthetic messages; L4 hygiene. All on a private URL.

Run:
    uv run python scripts/checks/check_latent_encoder_node.py
    uv run python scripts/checks/check_latent_encoder_node.py --keep
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "checks", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import check_lcmtypes
import latent_encoder_node as node_mod
import lcm
import newton
from loguru import logger

from dairlib import lcmt_robot_output, lcmt_timestamped_saved_traj
from drake import lcmt_point_cloud, lcmt_point_cloud_field
from magna import lcmt_round_belt_state
from round_belt_task.arm_kinematics import FrankaTip, UrTracking
from round_belt_task.commander import mat3_to_quat
from round_belt_task.lcm_simulation import RoundBeltLcmSimulation
from round_belt_task.waypoints import MAGNA_PARAMS_SIM_YAML, board_pose
from task_common import lcs_dataset as lcs
from task_common.latent_encoder import (
    LATENT_STATE_CHANNEL,
    LatentEncoder,
    parse_latent_state_message,
    resize_points_ordered,
)
from task_common.perception_lcm import (
    from_frame,
    point_cloud_msg,
    point_cloud_xyz,
    round_belt_points,
    round_belt_state_msg,
    to_frame,
)

PRIVATE_LCM_URL = "udpm://239.255.76.97:7697?ttl=0"
PRIVATE_PORT, SHARED_PORT = "7697", "7667"
CLOUD_CH, BELT_CH = "POINT_CLOUD_CROPPED", "RoundBeltState"
FRANKA_CH, UR_CH = "FRANKA_STATE", "UR_STATE_SIM"
EVERY = 15
L1_STEPS = 300
L2_STEPS = 630
MSG_WAIT_S = 1.0
NODE_START_TIMEOUT_S = 60.0
TEST_TIMEOUT_S = 240.0
LATENCY_MEAN_MS = 30.0
RUNTIME_BUDGET_S = 360.0
NODE = REPO_ROOT / "scripts" / "lcs" / "latent_encoder_node.py"
SIM_SCRIPT = REPO_ROOT / "scripts" / "round_belt_lcm_simulation.py"
MAIN_MAGNA = Path("/home/hienbui/git/magna")


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


class Skip(Exception):
    pass


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _gpu_apps() -> set[int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {int(v) for v in out.split() if v.strip().isdigit()}


class Listener:
    """This process's peer handle: every message on the watched channels, with receive wall time."""

    DECODERS: ClassVar[dict] = {
        CLOUD_CH: lcmt_point_cloud.decode, BELT_CH: lcmt_round_belt_state.decode,
        FRANKA_CH: lcmt_robot_output.decode, UR_CH: lcmt_robot_output.decode,
        LATENT_STATE_CHANNEL: lcmt_timestamped_saved_traj.decode,
    }

    def __init__(self) -> None:
        self.lc = lcm.LCM(PRIVATE_LCM_URL)
        self.msgs: dict[str, list] = {ch: [] for ch in self.DECODERS}
        for ch, dec in self.DECODERS.items():
            self.lc.subscribe(ch, self._handler(ch, dec))

    def _handler(self, ch, dec):
        def on_message(_ch: str, data: bytes) -> None:
            self.msgs[ch].append((dec(data), time.perf_counter()))

        return on_message

    def clear(self) -> None:
        for v in self.msgs.values():
            v.clear()

    def drain(self, timeout_ms: int = 0) -> None:
        while self.lc.handle_timeout(timeout_ms) > 0:
            timeout_ms = 0

    def wait(self, cond, timeout_s: float = MSG_WAIT_S) -> bool:
        deadline = time.monotonic() + timeout_s
        while not cond():
            if time.monotonic() > deadline:
                return False
            self.lc.handle_timeout(10)
        return True

    def by_utime(self, ch: str) -> dict[int, tuple]:
        return {int(m.utime): (m, w) for m, w in self.msgs[ch]}


def _arm_io(sim):
    return sim._io_by_spec_name("franka"), sim._io_by_spec_name("ur10")


def _sim_fk_path(sim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The harness's ``(ee_f, ee_u, prop)`` from the sim's own state."""
    fr, ur = _arm_io(sim)
    q = sim.state_0.joint_q.numpy()
    qd = sim.state_0.joint_qd.numpy()
    q_f, q_u = q[fr.coords].astype(np.float64), q[ur.coords].astype(np.float64)
    v_f, v_u = qd[fr.dofs].astype(np.float64), qd[ur.dofs].astype(np.float64)
    X = FrankaTip.fk(q[fr.coords])
    ee_f = np.concatenate([X[:3, 3], mat3_to_quat(X[:3, :3])])
    Xu = UrTracking.fk(q_u)
    ee_u = np.concatenate([Xu[:3, 3], mat3_to_quat(Xu[:3, :3])])
    return ee_f, ee_u, lcs.state_vector(q_f, q_u, v_f, v_u, ee_f, ee_u)


# --- L0 ------------------------------------------------------------------------------------------

@check("L0 lcmtypes")
def check_l0(ctx) -> str:
    for pkg, name, cls in (("drake", "lcmt_point_cloud", lcmt_point_cloud),
                           ("drake", "lcmt_point_cloud_field", lcmt_point_cloud_field),
                           ("magna", "lcmt_round_belt_state", lcmt_round_belt_state)):
        check_lcmtypes._check_fingerprint(pkg, name, cls)
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-1, 1, (2000, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (2000, 3), dtype=np.uint8)
    msg = point_cloud_msg(75000, xyz, rgb)
    dec = lcmt_point_cloud.decode(msg.encode())
    _require(dec.width == 2000 and dec.height == 1 and dec.point_step == 16,
             f"cloud header {dec.width}x{dec.height} step {dec.point_step}")
    _require(dec.data == msg.data and np.array_equal(point_cloud_xyz(dec), xyz), "cloud data")
    _require((len(msg.encode()) - msg.data_size) % 16 == 0, "data not 16-byte aligned")
    bpts = rng.uniform(-0.3, 0.3, (48, 3))
    bmsg = lcmt_round_belt_state.decode(round_belt_state_msg(75000, bpts).encode())
    _require(bmsg.num_points == 48 and bmsg.num_control_points == 0, "belt counts")
    _require(np.array_equal(round_belt_points(bmsg), bpts.astype(np.float32).astype(np.float64)),
             "belt points")
    fp = "fingerprints vs magna ok" if check_lcmtypes.MAGNA_AVAILABLE else "fingerprints SKIP"
    return f"2000-point cloud ({len(msg.encode())} B) + 48-point belt round trip, {fp}"


# --- L1 ------------------------------------------------------------------------------------------

@check("L1 sim publishers")
def check_l1(ctx) -> str:
    lis = ctx.listener
    # Defaults: the --test sim (no perception flags) publishes neither channel; also L4's exit.
    lis.clear()
    log = ctx.tmp / "sim_test.log"
    with log.open("w") as out:
        proc = subprocess.Popen([sys.executable, str(SIM_SCRIPT), "--test", "--lcm-url",
                                 PRIVATE_LCM_URL], cwd=REPO_ROOT, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
    ctx.pids.append(proc.pid)
    deadline = time.monotonic() + TEST_TIMEOUT_S
    while proc.poll() is None:
        if time.monotonic() > deadline:
            os.killpg(proc.pid, signal.SIGKILL)
            raise AssertionError("--test sim timed out")
        lis.lc.handle_timeout(50)
    lis.drain(200)
    ctx.test_exit = proc.returncode
    n_franka = len(lis.msgs[FRANKA_CH])
    _require(proc.returncode == 0, f"--test sim exit {proc.returncode} (log {log})")
    _require(n_franka > 0, "--test sim published no FRANKA_STATE here")
    _require(not lis.msgs[CLOUD_CH] and not lis.msgs[BELT_CH],
             f"defaults published {len(lis.msgs[CLOUD_CH])} clouds, "
             f"{len(lis.msgs[BELT_CH])} belts")

    stats_lines: list[str] = []
    logger.add(lambda m: stats_lines.append(str(m)), level="INFO", format="{message}",
               filter=lambda r: "[STATS]" in r["message"] or "[PERCEPTION]" in r["message"])
    args = RoundBeltLcmSimulation.create_parser().parse_args(
        ["--viewer", "null", "--no-realtime", "--cameras", "--publish-point-cloud",
         "--publish-belt-state", "--lcm-url", PRIVATE_LCM_URL])
    t0 = time.perf_counter()
    ctx.sim = sim = RoundBeltLcmSimulation(newton.viewer.ViewerNull(num_frames=100), args)
    build_s = time.perf_counter() - t0
    X_WB = board_pose(MAGNA_PARAMS_SIM_YAML)
    cloud = sim.point_clouds["cropped_point_cloud"]
    lis.clear()
    n_points, belt_err, samples = [], 0.0, 0
    for _ in range(L1_STEPS):
        sim.control_step()
        k, utime = sim.step_index, sim.step_index * 5000
        if k % EVERY:
            continue
        ok = lis.wait(lambda u=utime: u in lis.by_utime(CLOUD_CH) and u in lis.by_utime(BELT_CH))
        _require(ok, f"step {k}: no cloud/belt with utime {utime}")
        franka = lis.by_utime(FRANKA_CH)
        _require(utime in franka, f"step {k}: no FRANKA_STATE with utime {utime}")
        sim.cameras.update(sim.state_0)
        ref = cloud.compute()[0]
        got = point_cloud_xyz(lis.by_utime(CLOUD_CH)[utime][0])
        _require(got.shape == ref.shape and np.array_equal(got, ref),
                 f"step {k}: cloud {got.shape} != collector pcd {ref.shape}")
        _require(len(got) >= 500, f"step {k}: {len(got)} points < 500")
        bmsg = lis.by_utime(BELT_CH)[utime][0]
        _require(bmsg.frame_name == "taskboard" and bmsg.num_points == 48,
                 f"belt {bmsg.frame_name!r} {bmsg.num_points}")
        body = sim.state_0.body_q.numpy()[sim.info.belt_bodies, :3].astype(np.float64)
        belt_err = max(belt_err, float(np.abs(from_frame(X_WB, round_belt_points(bmsg))
                                              - body).max()))
        n_points.append(len(got))
        samples += 1
    lis.drain(100)
    utimes = sorted(lis.by_utime(CLOUD_CH))
    _require(utimes == [s * 5000 for s in range(EVERY, L1_STEPS + 1, EVERY)],
             f"cloud utimes {utimes[:4]}... not every {EVERY} steps")
    _require(sorted(lis.by_utime(BELT_CH)) == utimes, "belt utimes != cloud utimes")
    _require(belt_err <= 1e-6, f"belt back to world err {belt_err:.2e} m")
    sim._log_stats(time.perf_counter())
    stats = next((s for s in stats_lines if "[STATS]" in s), "")
    m = re.search(r"perception \d+ x \(render\+cloud\+publish mean ([\d.]+) ms max ([\d.]+) ms",
                  stats)
    _require(m is not None, f"no perception cost in [STATS]: {stats!r}")
    ctx.render_ms = (float(m.group(1)), float(m.group(2)))
    return (f"--test exit 0 ({n_franka} FRANKA_STATE, 0 perception msgs); build {build_s:.1f} s; "
            f"{samples} cloud+belt pairs at step % {EVERY} == 0, utime == FRANKA_STATE, clouds "
            f"bit-exact ({min(n_points)}-{max(n_points)} pts), belt err {belt_err:.1e} m; "
            f"per-render cost mean {m.group(1)} / max {m.group(2)} ms")


# --- L2 ------------------------------------------------------------------------------------------

def _start_node(ctx) -> subprocess.Popen:
    log = ctx.tmp / "node.log"
    out = log.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(NODE), "--lcm-url", PRIVATE_LCM_URL, "--deploy",
         str(ctx.deploy), "--ur-state-channel", UR_CH, "--state-match", "exact",
         "--stats-every", "2"],
        cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    ctx.node, ctx.node_log = proc, log
    ctx.pids.append(proc.pid)
    deadline = time.monotonic() + NODE_START_TIMEOUT_S
    while "[NODE] udpm" not in log.read_text():
        _require(proc.poll() is None, f"node exited {proc.returncode}: {log.read_text()[-800:]}")
        _require(time.monotonic() < deadline, "node did not start")
        time.sleep(0.1)
    return proc


@check("L2 node end-to-end")
def check_l2(ctx) -> str:
    if not ctx.deploy.is_file():
        raise Skip(f"{ctx.deploy} not found")
    sim, lis = ctx.sim, ctx.listener
    enc = LatentEncoder.load(ctx.deploy)
    X_WB = board_pose(MAGNA_PARAMS_SIM_YAML)
    _start_node(ctx)
    lis.clear()
    fk_ref: dict[int, tuple] = {}
    step0 = sim.step_index
    t0 = time.perf_counter()
    for _ in range(L2_STEPS):
        sim.control_step()
        if sim.step_index % EVERY == 0:
            utime = sim.step_index * 5000
            fk_ref[utime] = _sim_fk_path(sim)
            # Wait for the answer so the receive time is the latency, not our next step.
            lis.wait(lambda u=utime: any(int(m.utime) == u
                                         for m, _ in lis.msgs[LATENT_STATE_CHANNEL]))
        lis.drain()
    sim_wall = time.perf_counter() - t0
    n_clouds = len(fk_ref)
    lis.wait(lambda: len(lis.msgs[LATENT_STATE_CHANNEL]) >= n_clouds, 5.0)
    lis.drain(200)

    clouds, belts = lis.by_utime(CLOUD_CH), lis.by_utime(BELT_CH)
    franka, ur = lis.by_utime(FRANKA_CH), lis.by_utime(UR_CH)
    latents = lis.msgs[LATENT_STATE_CHANNEL]
    utimes = [int(m.utime) for m, _ in latents]
    _require(len(latents) == n_clouds, f"{len(latents)} latents for {n_clouds} clouds")
    _require(all(b > a for a, b in pairwise(utimes)), "latent utimes not increasing")
    _require(utimes == sorted(fk_ref), "latent utimes != cloud utimes")
    _require(all(b - a == EVERY * 5000 for a, b in pairwise(utimes)),
             f"rate != 1 per {EVERY} steps")
    z_err = fk_err = 0.0
    lat_ms = []
    for msg, wall in latents:
        u, t, z, ee_f, ee_u, prop = parse_latent_state_message(msg)
        _require(t == u * 1e-6, f"t {t} != utime {u}")
        cmsg, cwall = clouds[u]
        pc = lcs.camera_points(point_cloud_xyz(cmsg))
        belt = lcs.belt_points_ordered(from_frame(X_WB, round_belt_points(belts[u][0])))
        fr = node_mod.arm_state(franka[u][0], node_mod.FRANKA_POSITION_NAMES)
        urs = node_mod.arm_state(ur[u][0], node_mod.UR_POSITION_NAMES)
        ref_f, ref_u, ref_prop = fk_ref[u]
        Xf, Xu = FrankaTip.fk(fr.q), UrTracking.fk(urs.q)
        prop_msgs = lcs.state_vector(
            fr.q, urs.q, fr.v, urs.v, np.concatenate([Xf[:3, 3], mat3_to_quat(Xf[:3, :3])]),
            np.concatenate([Xu[:3, 3], mat3_to_quat(Xu[:3, :3])]))
        z_ref = enc.encode(pc, prop_msgs, belt)
        z_err = max(z_err, float(np.abs(z - z_ref).max()))
        diffs = (ee_f - ref_f, ee_u - ref_u, prop - ref_prop, prop_msgs - ref_prop)
        fk_err = max(fk_err, *(float(np.abs(d).max()) for d in diffs))
        lat_ms.append((wall - cwall) * 1e3)
    _require(z_err == 0.0, f"z differs from the rebuilt encode by {z_err:.2e}")
    _require(fk_err <= 1e-9, f"ee/proprio differ from the FK path by {fk_err:.2e}")
    time.sleep(2.5)  # one more node stats line
    text = ctx.node_log.read_text()
    misses = [int(v) for v in re.findall(r"state-match misses (\d+)", text)]
    errors = [int(v) for v in re.findall(r"errors (\d+)", text)]
    _require(misses and max(misses) == 0, f"node state-match misses {misses}")
    _require(not errors or max(errors) == 0, f"node errors {errors}")
    lat = np.asarray(lat_ms)
    under_load = "" if lat.mean() < LATENCY_MEAN_MS else " (> 30 ms: measured under load)"
    ctx.l2 = (lat.mean(), lat.max(), len(latents), sim_wall)
    return (f"{len(latents)} latents == {n_clouds} clouds (steps {step0 + EVERY}..{sim.step_index}"
            f", 1 per {EVERY} steps, strictly increasing), z bit-exact, ee/proprio err "
            f"{fk_err:.1e}, 0 misses; cloud->latent wall mean {lat.mean():.1f} max "
            f"{lat.max():.1f} ms{under_load}; sim {L2_STEPS} steps in {sim_wall:.1f} s "
            f"({len(latents) / sim_wall:.2f} latents/s wall)")


# --- L3 ------------------------------------------------------------------------------------------

@check("L3 hardware mapping")
def check_l3(ctx) -> str:
    rng = np.random.default_rng(1)
    X_WB = board_pose(MAGNA_PARAMS_SIM_YAML)
    world = rng.uniform(-0.2, 0.2, (48, 3)) + np.array([0.45, 0.0, 0.05])
    msg = lcmt_round_belt_state.decode(
        round_belt_state_msg(5000, to_frame(X_WB, world), "taskboard").encode())
    back = node_mod.belt_world(msg, X_WB)
    err = float(np.abs(back - world).max())
    _require(err <= 1e-6, f"taskboard -> world err {err:.2e}")
    ident = node_mod.belt_world(
        lcmt_round_belt_state.decode(round_belt_state_msg(5000, world, "world").encode()), None)
    _require(np.abs(ident - world).max() <= 1e-6, "world frame not passed through")
    try:
        node_mod.belt_world(msg, None)
        raise AssertionError("taskboard frame without a board pose accepted")
    except ValueError:
        pass
    verts = rng.uniform(-0.2, 0.2, (150, 3))
    msg150 = lcmt_round_belt_state.decode(
        round_belt_state_msg(5000, to_frame(X_WB, verts), "taskboard").encode())
    b150 = node_mod.belt_input(node_mod.belt_world(msg150, X_WB), "points150", lcs.BELT_POINTS)
    _require(b150.shape == (150, 3) and b150.dtype == np.float32, f"points150 {b150.shape}")
    ref = resize_points_ordered(from_frame(X_WB, round_belt_points(msg150)), lcs.BELT_POINTS)
    _require(np.array_equal(b150, ref), "points150 != loader resize")
    b48 = node_mod.belt_input(world, "bodies48")
    _require(np.array_equal(b48, lcs.belt_points_ordered(world)), "bodies48 != material points")
    # PointCloudToLcm layout with rgb, and a permuted layout: decode must follow byte offsets.
    xyz = rng.uniform(0, 1, (300, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, (300, 3), dtype=np.uint8)
    cmsg = lcmt_point_cloud.decode(point_cloud_msg(5000, xyz, rgb).encode())
    names = [(f.name, f.byte_offset, f.datatype) for f in cmsg.fields]
    _require(names == [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("rgb", 12, 6)], f"{names}")
    _require(np.array_equal(point_cloud_xyz(cmsg), xyz), "rgb layout decode")
    rec = np.zeros(300, dtype=[("rgb", "u1", (4,)), ("z", "<f4"), ("pad", "u1", (4,)),
                               ("x", "<f4"), ("y", "<f4")])
    rec["x"], rec["y"], rec["z"] = xyz.T
    perm = point_cloud_msg(5000, xyz[:0])
    perm.fields = []
    for name, off, dt in (("rgb", 0, 6), ("z", 4, 7), ("x", 12, 7), ("y", 16, 7)):
        f = lcmt_point_cloud_field()
        f.name, f.byte_offset, f.datatype, f.count = name, off, dt, 1
        perm.fields.append(f)
    perm.num_fields, perm.point_step, perm.width = 4, rec.dtype.itemsize, 300
    perm.data = rec.tobytes()
    perm.data_size = perm.row_step = len(perm.data)
    got = point_cloud_xyz(lcmt_point_cloud.decode(perm.encode()))
    _require(np.array_equal(got, xyz), "permuted layout decode")
    return (f"taskboard -> world err {err:.1e} m, points150 -> (150, 3) via loader resize, "
            f"PointCloudToLcm x/y/z/rgb (step 16) and a permuted 20-byte layout -> xyz only")


# --- L4 ------------------------------------------------------------------------------------------

@check("L4 hygiene")
def check_l4(ctx) -> str:
    node = getattr(ctx, "node", None)
    if node is not None and node.poll() is None:
        os.kill(node.pid, signal.SIGINT)
        try:
            node.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.kill(node.pid, signal.SIGKILL)
            node.wait()
            raise AssertionError("node ignored SIGINT") from None
        _require("[NODE] stopped" in ctx.node_log.read_text(), "node did not exit cleanly")
    for pid in ctx.pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f"pid {pid} still alive")
    left = subprocess.run(["pgrep", "-f", PRIVATE_PORT], capture_output=True, text=True,
                          check=False).stdout.split()
    left = [p for p in left if int(p) != os.getpid()]
    _require(not left, f"processes on {PRIVATE_PORT}: {left}")
    gpu = _gpu_apps()
    gpu_note = "nvidia-smi unavailable"
    if gpu is not None and ctx.gpu_baseline is not None:
        extra = gpu - ctx.gpu_baseline - {os.getpid()}
        _require(not (extra & set(ctx.pids)), f"our pids on the GPU: {extra & set(ctx.pids)}")
        gpu_note = "GPU apps at baseline" + (f" (foreign: {sorted(extra)})" if extra else "")
    _require(getattr(ctx, "test_exit", None) == 0, f"--test exit {getattr(ctx, 'test_exit', None)}")
    code = "SIGINT -> clean exit" if node is not None else "no node"
    return (f"node {code}, pids {ctx.pids} gone, pgrep -f {PRIVATE_PORT} empty, {gpu_note}, "
            f"--test on {PRIVATE_PORT} exit 0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the sim/node logs")
    parser.add_argument("--deploy", type=Path, default=node_mod.DEFAULT_DEPLOY)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stdout, level="WARNING", format="{level: <7} | {message}")
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="check_latent_encoder_node_"))
    magna_before = subprocess.run(["git", "-C", str(MAIN_MAGNA), "status", "--porcelain"],
                                  capture_output=True, text=True, check=False).stdout
    ctx = SimpleNamespace(tmp=tmp, deploy=args.deploy, pids=[], gpu_baseline=_gpu_apps(),
                          listener=Listener())
    exit_code = 0
    try:
        for name, fn in CHECKS:
            try:
                detail = fn(ctx)
            except Skip as exc:
                print(f"[SKIP] {name}: {exc}", flush=True)
                continue
            except AssertionError as exc:
                print(f"[FAIL] {name}: {exc}", file=sys.stderr)
                exit_code = 1
            except Exception as exc:  # noqa: BLE001 - report, then still clean up
                print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
                traceback.print_exc()
                exit_code = 1
            if exit_code:
                if name != "L4 hygiene":
                    try:
                        check_l4(ctx)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[FAIL] cleanup: {exc!r}", file=sys.stderr)
                break
            print(f"[PASS] {name}: {detail}", flush=True)
    finally:
        node = getattr(ctx, "node", None)
        if node is not None and node.poll() is None:
            os.kill(node.pid, signal.SIGKILL)
        if args.keep or exit_code:
            print(f"[INFO] kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    magna_after = subprocess.run(["git", "-C", str(MAIN_MAGNA), "status", "--porcelain"],
                                 capture_output=True, text=True, check=False).stdout
    if magna_after != magna_before:
        print("[FAIL] magna checkout status changed during the check", file=sys.stderr)
        exit_code = 1
    runtime = time.perf_counter() - t0
    if exit_code == 0:
        if runtime > RUNTIME_BUDGET_S:
            print(f"[WARN] runtime {runtime:.1f} s > {RUNTIME_BUDGET_S:g} s budget")
        print(f"ALL LATENT ENCODER NODE CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
