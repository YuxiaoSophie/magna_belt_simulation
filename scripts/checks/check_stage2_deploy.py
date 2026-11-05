#!/usr/bin/env python3
"""CPU checks of the stage-2 deploy prep (V12-94 P4, ``prep_v49.py``) per model dir: the deploy
loads (n_u 12), the UR input bounds are zero (deploy +-1e-9, MPPI exactly 0), the MPPI yaml is
the stage-2 one (H 7, u_franka_only, no prior, envelope / floor / outer box from
``v12_94/prep/envelope.json``), goal / waypoint latents have the right shapes and are reached by
the successful walks within their tols, and the envelope / floor / support box cover the S2a
data. No sim, no GPU, no LCM.

Run:
    uv run --frozen python scripts/checks/check_stage2_deploy.py [--dir <model dir> ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from latent_stage_tols import N_LAST, STREAK, streak_fire
from latent_waypoints import tol_frames

from task_common.latent_encoder import (
    LatentEncoder,
    LearnedLcs,
    encode_episode_frames,
    episode_input_keys,
)
from task_common.mppi_lcs import UR_CHANNELS, MppiParams

SIM = REPO_ROOT / "data/lcs/realfirst/cmp/encoder_study/sim"
PREP = REPO_ROOT / "data/lcs/realfirst/cmp/v12_94/prep"
S2A = REPO_ROOT / "data/lcs/realfirst/cmp/v12_94/tiers/S2a/buffer.json"
TILT = REPO_ROOT / "data/lcs/realfirst/cmp/v12_96/tilt_limits.json"
FR = [0, 1, 2, 6, 7, 8]
SCALE = np.r_[[1e3] * 6, [math.degrees(1.0)] * 6]
UR_HALF = 1e-9
REACH_MIN = 0.9  # p90 tols: >= 90 % of the walks reach every waypoint in its window
GOAL_MATCH = 1e-3  # whitened: torch export goal vs numpy-twin waypoint latent


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rot_deg(q0, q) -> np.ndarray:
    d = np.abs(np.sum(np.asarray(q0)[None, :] * np.asarray(q), axis=1))
    return np.degrees(2 * np.arccos(np.clip(d, -1.0, 1.0)))


class Model:
    def __init__(self, d: Path, env: dict, walks: dict):
        self.d, self.env, self.walks = d, env, walks
        self.dep = json.loads((d / "deploy.json").read_text())
        self.deploy = Path(self.dep["deploy"])
        self.yaml = yaml.safe_load((d / "mppi_hard_suite.yaml").read_text())
        self.params = MppiParams.from_yaml(d / "mppi_hard_suite.yaml")
        self.goal = json.loads((d / "goal.json").read_text())
        self.stt = json.loads((d / "stage_tols.json").read_text())
        with np.load(d / "latent_waypoints.npz", allow_pickle=False) as f:
            self.wp = {k: f[k] for k in f.files}
        self._z: dict[str, np.ndarray] = {}

    def latents(self, enc, path: str) -> np.ndarray:
        if path not in self._z:
            with np.load(path, allow_pickle=True) as f:
                n = len(f["actions"])
                z, empty = encode_episode_frames(enc, f, range(n))
            _require(not empty.any(), f"{path}: {int(empty.sum())} empty belt clouds")
            self._z[path] = z
        return self._z[path]


def c_files(m: Model) -> str:
    need = ["model.json", "box_0.json", "trust_region.json", "goal.json", "stage_tols.json",
            "latent_waypoints.npz", "support_box.json", "mppi_hard_suite.yaml", "loop.yaml"]
    miss = [n for n in need if not (m.d / n).is_file()]
    miss += [str(p) for p in (m.deploy, m.deploy.parent / "decoder.npz") if not p.is_file()]
    _require(not miss, f"missing {miss}")
    _require(m.dep["deploy_sha256"] == _sha(m.deploy), "deploy.json sha != deploy.npz")
    _require(m.dep["check_latent_encoder"] == "PASS", "check_latent_encoder did not pass")
    dec = m.d / "decoder_decfit" / "decoder.npz"
    return f"deploy {m.deploy.parent.name}, encoder check PASS, decfit decoder " + (
        "present" if dec.is_file() else "ABSENT (chain export pending)")


def c_deploy(m: Model) -> str:
    lcs, enc = LearnedLcs.load(m.deploy), LatentEncoder.load(m.deploy)
    _require(lcs.n_u == 12, f"n_u {lcs.n_u} != 12")
    with np.load(m.deploy, allow_pickle=True) as d:
        g1, g2 = d["z_goal_stage1"], d["z_goal_stage2"]
        lb, ub = d["u_lb"], d["u_ub"]
    _require(g1.shape == g2.shape == (lcs.n_x,), f"goal shapes {g1.shape} {g2.shape}")
    ur = list(UR_CHANNELS)
    _require(np.all(np.abs(lb[ur]) <= UR_HALF) and np.all(np.abs(ub[ur]) <= UR_HALF),
             f"deploy UR bounds {lb[ur]} {ub[ur]}")
    _require(np.all(lb[FR] < 0) and np.all(ub[FR] > 0), "a Franka channel cannot reverse")
    bx = json.loads((m.d / "box_0.json").read_text())
    _require(np.allclose(lb, bx["u_lb"]) and np.allclose(ub, bx["u_ub"]), "deploy box != box_0")
    return f"n_x {lcs.n_x}, n_u 12, encoder {type(enc).__name__}, UR +-{UR_HALF:g}"


def c_bounds(m: Model) -> str:
    with np.load(m.deploy, allow_pickle=True) as d:
        ns = SimpleNamespace(u_lb_np=np.asarray(d["u_lb"]), u_ub_np=np.asarray(d["u_ub"]),
                             n_u=int(d["n_u"]))
    lb, ub = m.params.bounds(ns)
    ur = list(UR_CHANNELS)
    _require(np.all(lb[ur] == 0.0) and np.all(ub[ur] == 0.0), f"MPPI UR bounds {lb[ur]} {ub[ur]}")
    ob = m.env["outer_box"]
    if "rot_deg_xyz" in (m.params.u_box_outer or {}):  # V12-96 per-axis box (prep_v50)
        xyz = json.loads(TILT.read_text())["u_box_outer"]["rot_deg_xyz"]
        ob = {**ob, "rot_deg_xyz": xyz}
    _require(m.params.u_box_outer == ob, f"u_box_outer {m.params.u_box_outer} != {ob}")
    half = np.r_[[ob["trans_m"]] * 3, [math.radians(ob["rot_deg"])] * 3]
    _require(np.all(-half - 1e-12 <= lb[FR]) and np.all(ub[FR] <= half + 1e-12), "outside outer")
    p1, p99 = np.asarray(m.env["franka_p1"]), np.asarray(m.env["franka_p99"])
    blo, bhi = lb[FR] * SCALE[FR], ub[FR] * SCALE[FR]
    _require(np.all(blo <= p1 + 1e-3) and np.all(bhi >= p99 - 1e-3),
             f"box {np.round(blo, 2)}..{np.round(bhi, 2)} misses the data p1..p99")
    U = []
    for r in json.loads(S2A.read_text())["rollouts"]:
        with np.load(r["episode"], allow_pickle=True) as f:
            U.append(np.asarray(f["actions"], np.float64))
    U = np.concatenate(U)
    inside = np.all((U >= lb - 1e-12) & (U <= ub + 1e-12), axis=1).mean()
    m.box_disp = (np.round(blo, 2).tolist(), np.round(bhi, 2).tolist())
    return (f"Franka box (mm, deg) lo {m.box_disp[0]} hi {m.box_disp[1]}; outer +-"
            f"{ob['trans_m'] * 1e3:g} mm / +-{ob['rot_deg']:g} deg; {inside:.1%} of {len(U)} S2a "
            "frames inside")


def c_yaml(m: Model) -> str:
    p, y, env = m.params, m.yaml, m.env
    _require(p.H == 7 and p.u_franka_only and p.action_noise_frac == 0.0, "H / franka / noise")
    _require(p.prior_alpha == 0.0 and p.prior_weight == 0.0 and p.prior_sync is None
             and p.bc_prior is None, "a demo action prior is on")
    _require(p.stage_start == "progress" and p.stage_reach_streak == 3, "stage settings")
    _require(p.rot_envelope_deg == env["rot_envelope_deg"], f"envelope {p.rot_envelope_deg}")
    _require(p.z_min_franka == env["z_min_franka"], "z_min_franka != envelope.json")
    sb = json.loads((m.d / "support_box.json").read_text())
    _require(y["support_box"] == {k: sb[k] for k in ("lo", "hi", "weight", "hard_margin_m")},
             "yaml support_box != support_box.json")
    _require(y["stage_goal_tols"] == [m.goal["tol_1"], -1.0], "stage_goal_tols")
    tilt = ""
    if "tilt_max_deg" in y:  # V12-96 tilt limit (prep_v50)
        want = json.loads(TILT.read_text())["tilt_max_deg"]
        _require(y["tilt_max_deg"] == want, f"tilt_max_deg {y['tilt_max_deg']} != {want}")
        tilt = f", tilt {y['tilt_max_deg']}, rot_deg_xyz {y['u_box_outer'].get('rot_deg_xyz')}"
    return (f"H 7, u_franka_only, no prior, envelope {p.rot_envelope_deg}, z_min_franka "
            f"{p.z_min_franka:.4f}{tilt}")


def c_goal(m: Model) -> str:
    w, lcs = m.wp, LearnedLcs.load(m.deploy)
    K = len(w["frames"])
    _require(w["z_wp"].shape == (K, lcs.n_x) and w["tol"].shape == (K,), "waypoint shapes")
    _require(w["tol"][-1] == -1.0 and np.all(w["tol"][:-1] >= float(w["floor"])), "tols")
    fr = w["frames"].tolist()
    _require(fr == sorted(set(fr)) and int(w["latch"]) in fr and int(w["first_hold"]) in fr,
             f"frames {fr}")
    _require(str(w["latch_after"]) == "transit" and str(w["first_hold_label"]) == "end",
             "phase marks")
    _require(str(w["deploy_sha256"]) == m.dep["deploy_sha256"], "waypoints on another deploy")
    goal_ep = str(Path(m.walks["goal"]).resolve())
    _require(str(w["demo_episode"]) == goal_ep == str(Path(m.goal["episode"]).resolve()),
             "goal demo differs between goal.json / waypoints / walks.json")
    with np.load(m.deploy, allow_pickle=True) as d:
        g1, g2, gfr = d["z_goal_stage1"], d["z_goal_stage2"], d["goal_frames"].tolist()
    _require(gfr == [int(w["latch"]), len(w["z_traj"]) - 1], f"deploy goal frames {gfr}")
    d1 = lcs.whitened_dist(g1, w["z_wp"][fr.index(int(w["latch"]))])
    d2 = lcs.whitened_dist(g2, w["z_wp"][-1])
    _require(d1 < GOAL_MATCH and d2 < GOAL_MATCH, f"export vs numpy goal {d1:.2e} {d2:.2e}")
    return (f"{K} waypoints, frames {fr}, latch {int(w['latch'])}, first hold "
            f"{int(w['first_hold'])}; goal frames {gfr}")


def c_reach(m: Model) -> str:
    w, lcs, enc = m.wp, LearnedLcs.load(m.deploy), LatentEncoder.load(m.deploy)
    _require(episode_input_keys(enc)[0] == "pcd_belt_seg", "not a belt-cloud encoder")
    per = json.loads(str(w["per_demo"]))
    K, fr, win = len(w["frames"]), w["frames"].tolist(), int(w["window"])
    hit = np.zeros((len(per), K), bool)
    in_order = 0
    for j, r in enumerate(per):
        z = m.latents(enc, r["episode"])
        for k in range(K - 1):
            ts = tol_frames(r["aligned_frames"][k], r["latch"], win, len(z),
                            fr[k] == int(w["latch"]))
            hit[j, k] = min(lcs.whitened_dist(z[t], w["z_wp"][k]) for t in ts) < w["tol"][k]
        t0 = 0
        for k in range(K - 1):  # node rule: STREAK latents inside, stages in order
            dist = [lcs.whitened_dist(x, w["z_wp"][k]) for x in z]
            f = streak_fire(dist, float(w["tol"][k]), STREAK, t0)
            if f is None:
                break
            t0 = f + 1
        else:
            in_order += 1
    frac = hit[:, :-1].mean(0)
    _require(frac.min() >= REACH_MIN, f"waypoint reach fractions {np.round(frac, 2).tolist()}")
    g2 = m.stt["tol"][1]
    with np.load(m.deploy, allow_pickle=True) as d:
        zg = d["z_goal_stage2"]
    last = [np.median([lcs.whitened_dist(x, zg) for x in m.latents(enc, r["episode"])[-N_LAST:]])
            for r in per]
    n_last = int(np.sum(np.asarray(last) < g2))
    st = m.stt
    m.reach = {"frac_min": float(frac.min()), "in_order": in_order, "n": len(per),
               "final_in_tol2": n_last, "latch_in_tol1": st["n_latch_inside_tol1"],
               "streak_in_window": st["n_streak_in_window"]}
    return (f"{len(per)} walks: per-waypoint reach min {frac.min():.2f} (>= {REACH_MIN}); "
            f"all stages in order (streak {STREAK}, no timeouts) {in_order}/{len(per)}; end-hold "
            f"median within tol_2 {g2:.3f}: {n_last}/{len(per)}; stage tols: latch inside tol_1 "
            f"{st['n_latch_inside_tol1']}/{st['n_demos']}, streak {st['n_streak_in_window']}/"
            f"{st['n_demos']}")


def c_cover(m: Model) -> str:
    p = m.params
    sb = json.loads((m.d / "support_box.json").read_text())
    lo, hi = np.asarray(sb["lo"]), np.asarray(sb["hi"])
    rf, ru, zf, rise, out = [], [], [], [], 0
    for r in json.loads(S2A.read_text())["rollouts"]:
        with np.load(r["episode"], allow_pickle=True) as f:
            ef, eu = np.asarray(f["sim_ee_franka"], np.float64), np.asarray(f["sim_ee_ur"],
                                                                              np.float64)
        rf.append(_rot_deg(ef[0, 3:7], ef[:, 3:7]).max())
        ru.append(_rot_deg(eu[0, 3:7], eu[:, 3:7]).max())
        zf.append(ef[:, 2].min())
        rise.append((ef[:, 2] - ef[0, 2]).max())
        P = np.concatenate([ef[:, :3], eu[:, :3]], axis=1)
        out += int(np.sum(np.any((P < lo) | (P > hi), axis=1)))
    env = p.rot_envelope_deg
    _require(max(rf) < env["franka"] and max(ru) < env["ur"],
             f"rotation max Franka {max(rf):.1f} / UR {max(ru):.3f} vs envelope {env}")
    _require(min(zf) >= p.z_min_franka, f"Franka z {min(zf):.4f} < z_min_franka")
    _require(max(rise) <= p.z_ceiling_above_start_m, f"Franka rise {max(rise):.4f} > ceiling")
    _require(out == 0, f"{out} S2a frames outside the support box")
    return (f"{len(rf)} episodes: Franka rotation max {max(rf):.1f} < {env['franka']:g} deg, UR "
            f"{max(ru):.3f} < {env['ur']:g}; Franka z min {min(zf) * 1e3:.1f} mm >= "
            f"{p.z_min_franka * 1e3:.1f}; rise {max(rise) * 1e3:.1f} <= "
            f"{p.z_ceiling_above_start_m * 1e3:g} mm; support box holds every frame")


CHECKS = [("D0 artefacts", c_files), ("D1 deploy loads, n_u 12, UR bounds", c_deploy),
          ("D2 MPPI input box", c_bounds), ("D3 MPPI yaml", c_yaml),
          ("D4 goal / waypoint latents", c_goal), ("D5 walks reach the waypoints", c_reach),
          ("D6 envelope / floor / support box cover S2a", c_cover)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", type=Path, nargs="+",
                    default=[SIM / n for n in ("V49a", "V49b", "V49c")])
    a = ap.parse_args()
    env = json.loads((PREP / "envelope.json").read_text())
    walks = json.loads((PREP / "walks.json").read_text())
    t0, n_fail = time.perf_counter(), 0
    for d in a.dir:
        if not (d / "deploy.json").is_file():
            print(f"[FAIL] {d}: no deploy.json (prep_v49.py not run)", file=sys.stderr)
            n_fail += 1
            continue
        m = Model(d, env, walks)
        for name, fn in CHECKS:
            try:
                detail = fn(m)
            except AssertionError as exc:
                print(f"[FAIL] {d.name} {name}: {exc}", file=sys.stderr)
                n_fail += 1
                continue
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                print(f"[FAIL] {d.name} {name}: {exc!r}", file=sys.stderr)
                n_fail += 1
                continue
            print(f"[PASS] {d.name} {name}: {detail}")
    if n_fail:
        print(f"{n_fail} STAGE-2 DEPLOY CHECK(S) FAILED", file=sys.stderr)
        return 1
    print(f"ALL STAGE-2 DEPLOY CHECKS PASSED ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
