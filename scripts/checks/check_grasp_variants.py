#!/usr/bin/env python3
"""End-to-end check of ``scripts/lcs/make_grasp_variants.py`` (grasp-varied start states).

G0 a temp 3-variant set (``--count 2``, seed 7, half the ``set1`` box) builds and its nominal
``gv_00`` matches ``pre_place_1_osc.npz``; G1 every saved state restores into an OSC sim and
holds for 200 steps; G2 the ``set1`` probe's +-10 mm slides measure with the commanded sign and
magnitude; G3 no process on the private port, GPU apps at baseline, ``pre_place_1*.npz``
unchanged, temp set removed.

Run:
    uv run python scripts/checks/check_grasp_variants.py
    uv run python scripts/checks/check_grasp_variants.py --keep
    uv run python scripts/checks/check_grasp_variants.py \\
        --lcm-url 'udpm://239.255.76.100:7700?ttl=0'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from loguru import logger
from private_url import pgrep_others, url_port

from round_belt_task import arm_kinematics as ak
from task_common import sim_snapshot

PRIVATE_LCM_URL = "udpm://239.255.76.93:7693?ttl=0"
PRIVATE_PORT = "7693"
SCRIPT = REPO_ROOT / "scripts" / "lcs" / "make_grasp_variants.py"
SET1 = sim_snapshot.DEFAULT_START_STATE_DIR / "grasp_variants" / "set1"
REFERENCE = sim_snapshot.DEFAULT_START_STATE_DIR / "pre_place_1_osc.npz"
PROTECTED = sorted(sim_snapshot.DEFAULT_START_STATE_DIR.glob("pre_place_1*.npz"))
KNOBS = ("franka_slide_mm", "franka_roll_deg", "ur_slide_mm", "ur_roll_deg")
FALLBACK_HALF_BOX = "5 5 5 5"  # only if set1 has neither index.json nor probe.json
MINI_COUNT, MINI_SEED = 2, 7
BUILD_TIMEOUT_S = 420.0
G0_MM, G0_DEG, G0_BELT_RMS_MM = 1.5, 1.0, 8.0
G0_SLIDE_MM, G0_ROLL_DEG = 5.0, 3.0
G1_STEPS, G1_DRIFT_MM = 200, 5.0
G2_SLIDE_MM, G2_REL_TOL = 10, 0.5
OSC_LOG_ERRORS = ("resetting", "Exception caught")
RUNTIME_BUDGET_S = 480.0


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


CHECKS: list[tuple[str, callable]] = []


def check(title: str):
    def register(fn):
        CHECKS.append((title, fn))
        return fn

    return register


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_apps() -> set[int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {int(v) for v in out.split() if v.strip().isdigit()}


def _pose_err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 1e3, math.degrees(ak.rot_angle(a, b))


def _half_box() -> tuple[str, str]:
    """``--box`` text at half the set1 box, and where it came from."""
    for name in ("index.json", "probe.json"):
        path = SET1 / name
        if path.exists():
            box = json.loads(path.read_text())["box"]
            text = " ".join(f"{box[k][0] / 2:g}:{box[k][1] / 2:g}" for k in KNOBS)
            return text, str(path)
    return FALLBACK_HALF_BOX, "fallback (no set1 index/probe)"


@check("G0 nominal variant")
def check_g0(ctx: SimpleNamespace) -> str:
    box, source = _half_box()
    cmd = [sys.executable, str(SCRIPT), "--root", str(ctx.tmp_root), "--set", "mini",
           "--count", str(MINI_COUNT), "--seed", str(MINI_SEED), "--box", box,
           "--lcm-url", PRIVATE_LCM_URL]
    log = ctx.tmp_root / "build.log"
    t0 = time.perf_counter()
    with log.open("w") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                              timeout=BUILD_TIMEOUT_S, check=False)
    ctx.build_s = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-15:])
        raise AssertionError(f"build exit {proc.returncode} (log {log}):\n{tail}")
    ctx.set_dir = ctx.tmp_root / "mini"
    ctx.index = json.loads((ctx.set_dir / "index.json").read_text())
    nominal = ctx.index["variants"][0]
    _require(nominal["id"] == "gv_00" and nominal["held"], f"gv_00 row {nominal}")

    from round_belt_task.osc_simulation import RoundBeltOscSimulation
    ctx.sim = sim = RoundBeltOscSimulation.build(lcm_url=PRIVATE_LCM_URL)
    got = sim_snapshot.load(ctx.set_dir / nominal["file"]).body_q.astype(np.float64)
    ref = sim_snapshot.load(REFERENCE).body_q.astype(np.float64)
    f_mm, f_deg = _pose_err(ak.model_pose_franka_tip(got, sim._finger_tip_body),
                            ak.model_pose_franka_tip(ref, sim._finger_tip_body))
    u_mm, u_deg = _pose_err(ak.model_pose_ur_tracking(got, sim._ur_wrist_body),
                            ak.model_pose_ur_tracking(ref, sim._ur_wrist_body))
    belts = [b[sim.info.belt_bodies, :3] for b in (got, ref)]
    rms = float(np.sqrt(np.mean(np.sum((belts[0] - belts[1]) ** 2, axis=1)))) * 1e3
    _require(f_mm < G0_MM and f_deg < G0_DEG, f"finger_tip {f_mm:.2f} mm {f_deg:.2f} deg")
    _require(u_mm < G0_MM and u_deg < G0_DEG, f"UR tracking {u_mm:.2f} mm {u_deg:.2f} deg")
    _require(rms < G0_BELT_RMS_MM, f"belt rms {rms:.2f} mm")
    off = nominal["measured"]["offsets"]
    for name in ("franka", "ur"):
        o = off[name]
        _require(abs(o["slide_mm"]) < G0_SLIDE_MM and abs(o["roll_deg"]) < G0_ROLL_DEG,
                 f"{name} offsets {o}")
    return (f"mini set built in {ctx.build_s:.0f} s (box '{box}' from {source}); gv_00 vs "
            f"{REFERENCE.name}: finger_tip {f_mm:.2f} mm {f_deg:.2f} deg, UR {u_mm:.2f} mm "
            f"{u_deg:.2f} deg, belt rms {rms:.2f} mm; offsets franka "
            f"{off['franka']['slide_mm']:+.2f} mm {off['franka']['roll_deg']:+.2f} deg, ur "
            f"{off['ur']['slide_mm']:+.2f} mm {off['ur']['roll_deg']:+.2f} deg")


@check("G1 mini set restores + holds")
def check_g1(ctx: SimpleNamespace) -> str:
    index, sim = ctx.index, ctx.sim
    for key in ("set", "seed", "box", "params_sha256", "nominal_start_sha256", "variants"):
        _require(key in index, f"index.json lacks {key!r}")
    rows = index["variants"]
    _require(len(rows) == MINI_COUNT + 1 and index["seed"] == MINI_SEED,
             f"{len(rows)} variants, seed {index['seed']}")
    for r in rows:
        for key in ("id", "commanded", "measured", "held", "file", "notes"):
            _require(key in r, f"{r.get('id')} lacks {key!r}")
        _require(r["held"] and r["file"], f"{r['id']} not held: {r['reason']}")
    sim.start_osc(ctx.tmp_root / "osc.log")
    ctx.osc_pid = sim.osc.pid
    parts = []
    for r in rows:
        snap = sim_snapshot.load(ctx.set_dir / r["file"])
        _require(str(r["id"]) in snap.meta["notes"] and "utime offset" in snap.meta["notes"],
                 f"{r['file']} notes: {snap.meta['notes'][:80]}")
        sim.restore(snap, settle_steps=0)
        before = sim.belt_positions()
        grasp = sim.settle(G1_STEPS)
        drift = float(np.linalg.norm(sim.belt_positions() - before, axis=1).max()) * 1e3
        _require(grasp.held() == (True, True), f"{r['id']} held {grasp.held()} "
                 f"({grasp.describe()})")
        _require(drift < G1_DRIFT_MM, f"{r['id']} belt drift {drift:.2f} mm")
        o = r["measured"]["offsets"]
        parts.append(f"{r['id']} drift {drift:.2f} mm (franka {o['franka']['slide_mm']:+.1f} mm "
                     f"{o['franka']['roll_deg']:+.1f} deg, ur {o['ur']['slide_mm']:+.1f} mm "
                     f"{o['ur']['roll_deg']:+.1f} deg)")
    bad = [e for e in OSC_LOG_ERRORS if e in sim.osc.log_text()]
    _require(not bad, f"OSC log has {bad}")
    return f"{len(rows)} states held {G1_STEPS} steps: " + "; ".join(parts)


@check("G2 slide sign consistency")
def check_g2(ctx: SimpleNamespace) -> str:
    path = SET1 / "probe.json"
    if not path.exists():
        ctx.skipped = f"no {path}"
        return f"[SKIP] no {path}"
    rows = {r["id"]: r for r in json.loads(path.read_text())["rows"]}
    parts, skipped = [], []
    for name in ("franka", "ur"):
        for sign in (1, -1):
            rid = f"{name}_slide_mm{sign * G2_SLIDE_MM:+d}"
            r = rows.get(rid)
            if r is None or not r["held"]:
                skipped.append(f"{rid} ({'absent' if r is None else r['reason']})")
                continue
            got = r["measured"]["offsets"][name]["slide_mm"]
            want = sign * G2_SLIDE_MM
            _require(got * want > 0 and abs(got - want) <= G2_REL_TOL * abs(want),
                     f"{rid}: measured {got:+.2f} mm")
            parts.append(f"{rid} -> {got:+.2f} mm")
    if skipped:
        print(f"[SKIP] G2: {', '.join(skipped)} did not hold", flush=True)
    return "; ".join(parts) + (f"; skipped {len(skipped)}" if skipped else "")


@check("G3 hygiene")
def check_g3(ctx: SimpleNamespace) -> str:
    if ctx.sim is not None:
        ctx.sim.close("finished")
    if ctx.osc_pid is not None:
        try:
            os.kill(ctx.osc_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError(f"OSC pid {ctx.osc_pid} still alive")
    left = pgrep_others(PRIVATE_PORT)
    _require(not left, f"processes on {PRIVATE_PORT}: {left}")
    changed = [p.name for p in PROTECTED if _sha256(p) != ctx.hashes[p]]
    _require(not changed, f"changed: {changed}")
    gpu = _gpu_apps()
    gpu_note = "nvidia-smi unavailable"
    if gpu is not None and ctx.gpu_baseline is not None:
        extra = gpu - ctx.gpu_baseline - {os.getpid()}
        _require(ctx.osc_pid not in extra, f"OSC pid on the GPU: {extra}")
        gpu_note = "GPU apps at baseline" + (f" (foreign: {sorted(extra)})" if extra else "")
    if ctx.keep:
        kept = f"kept {ctx.tmp_root}"
    else:
        shutil.rmtree(ctx.tmp_root, ignore_errors=True)
        _require(not ctx.tmp_root.exists(), f"{ctx.tmp_root} not removed")
        kept = "temp set removed"
    return (f"pgrep -f {PRIVATE_PORT} empty, {gpu_note}, {len(PROTECTED)} pre_place_1*.npz "
            f"sha256 unchanged, {kept}")


def main() -> int:
    global PRIVATE_LCM_URL, PRIVATE_PORT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", action="store_true", help="keep the temp set and logs")
    parser.add_argument("--lcm-url", default=PRIVATE_LCM_URL)
    args = parser.parse_args()
    PRIVATE_LCM_URL, PRIVATE_PORT = args.lcm_url, url_port(args.lcm_url)
    logger.remove()
    logger.add(sys.stdout, level="WARNING", format="{level: <7} | {message}")

    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="check_grasp_variants_"))
    ctx = SimpleNamespace(tmp_root=tmp_root, keep=args.keep, gpu_baseline=_gpu_apps(),
                          hashes={p: _sha256(p) for p in PROTECTED}, sim=None, osc_pid=None)
    exit_code = 0
    try:
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
            print(f"[PASS] {name}: {detail}", flush=True)
    finally:
        if ctx.sim is not None:
            ctx.sim.close("finished" if exit_code == 0 else "failed")
        if exit_code and tmp_root.exists():
            print(f"[INFO] kept {tmp_root}")

    runtime = time.perf_counter() - t0
    if exit_code == 0:
        if runtime > RUNTIME_BUDGET_S:
            print(f"[WARN] runtime {runtime:.1f} s > {RUNTIME_BUDGET_S:g} s budget")
        print(f"ALL GRASP VARIANT CHECKS PASSED ({runtime:.1f} s)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
