#!/usr/bin/env python3
"""CPU checks of the V12-110 whole-run prep (``prep_v52_s1.py`` / ``prep_v52_s2.py`` /
``v12_110/suite_configs.py``) per picked model: W1 the stage-1 and stage-2 deploys are the same
model (every deploy.npz array bitwise equal except the exporter's data / goal / box keys), both
encoder checks PASS; W2 stage-2 UR inputs pinned, stage-1 not; W3 the suite yamls differ from
their bases only in stage_goal_tols / support_box (= goal.json tol_1 / support_box.json); W4 the
stage-2 dir passes check_stage2_deploy.py's D0-D6. No sim, no GPU, no LCM.

Run:
    uv run --frozen python scripts/checks/check_wholerun_deploy.py [--pick P] [--configs D] [NAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_stage2_deploy as c2

from task_common.mppi_lcs import UR_CHANNELS

CMP = c2.REPO_ROOT / "data/lcs/realfirst/cmp"
BASE = {1: CMP / "encoder_study/sim87_V45c_ur30/configs/V45c_ur30_H7.yaml",
        2: CMP / "encoder_study/sim104_V50b/configs/V50b_ffg29_ws1_du.yaml"}
# exporter keys computed from --data-files / --goal-* / --u-bounds-json (export_learned_lcs_deploy.py)
PER_DEPLOY = {"z_goal", "z_std", "z_mean", "u_lb", "u_ub", "goal_tol_whitened", "z_start_example",
              "goal_source", "z_goal_mean_engaged", "z_goal_stage1", "z_goal_stage2",
              "goal_frames", "exported_at", "data_dirs", "reference_z_f32_err"}
YAML_KEYS = {"stage_goal_tols", "support_box"}
PIN = 2e-9


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _deploy(d: Path) -> tuple[dict, dict]:
    dep = json.loads((d / "deploy.json").read_text())
    with np.load(dep["deploy"], allow_pickle=True) as f:
        return dep, {k: f[k] for k in f.files}


def w1(name: str, a) -> str:
    (j1, n1), (j2, n2) = _deploy(a.sim / name), _deploy(a.sim / f"{name}_s2")
    _require(j1["check_latent_encoder"] == j2["check_latent_encoder"] == "PASS",
             "check_latent_encoder not PASS on both deploys")
    _require(set(n1) == set(n2), f"key sets differ: {sorted(set(n1) ^ set(n2))}")
    bad = sorted(k for k in set(n1) - PER_DEPLOY
                 if n1[k].dtype != n2[k].dtype or n1[k].shape != n2[k].shape
                 or n1[k].tobytes() != n2[k].tobytes())
    _require(not bad, f"model arrays differ: {bad}")
    diff = sorted(k for k in PER_DEPLOY & set(n1) if n1[k].tobytes() != n2[k].tobytes())
    return (f"{len(n1) - len(PER_DEPLOY & set(n1))} model arrays bitwise equal "
            f"(checkpoint_sha256 {str(n1['checkpoint_sha256'])[:12]}); differ: {diff}")


def w2(name: str, a) -> str:
    (_, n1), (_, n2) = _deploy(a.sim / name), _deploy(a.sim / f"{name}_s2")
    ur = list(UR_CHANNELS)
    w1_, w2_ = (n1["u_ub"] - n1["u_lb"])[ur], (n2["u_ub"] - n2["u_lb"])[ur]
    _require(np.all(w2_ <= PIN), f"stage-2 UR widths {w2_}")
    _require(np.all(w1_ > PIN), f"stage-1 UR widths {w1_}")
    return f"UR dims {ur}: stage-2 width max {w2_.max():.1e}, stage-1 width min {w1_.min():.2e}"


def w3(name: str, a) -> str:
    out = []
    for st, d in ((1, a.sim / name), (2, a.sim / f"{name}_s2")):
        cfg = a.configs / f"{name}_stage{st}.yaml"
        b, y = yaml.safe_load(BASE[st].read_text()), yaml.safe_load(cfg.read_text())
        diff = sorted(k for k in set(b) | set(y) if b.get(k) != y.get(k))
        _require(set(diff) <= YAML_KEYS, f"{cfg.name} vs {BASE[st].name}: {diff}")
        goal = json.loads((d / "goal.json").read_text())
        sb = json.loads((d / "support_box.json").read_text())
        _require(y["stage_goal_tols"] == [goal["tol_1"], *b["stage_goal_tols"][1:]],
                 f"{cfg.name} stage_goal_tols {y['stage_goal_tols']} != tol_1 {goal['tol_1']}")
        _require(y["support_box"] == {k: sb[k] for k in ("lo", "hi", "weight", "hard_margin_m")},
                 f"{cfg.name} support_box != {d.name}/support_box.json")
        out.append(f"stage{st} {diff}")
    return "; ".join(out)


def w4(name: str, a) -> str:
    env = json.loads((c2.PREP / "envelope.json").read_text())
    walks = json.loads((c2.PREP / "walks.json").read_text())
    m = c2.Model(a.sim / f"{name}_s2", env, walks)
    for label, fn in c2.CHECKS:
        try:
            fn(m)
        except AssertionError as exc:
            raise AssertionError(f"{label}: {exc}") from exc
    return f"{name}_s2 D0-D6 PASS"


CHECKS = [("W1 same model in both deploys", w1), ("W2 UR pinned in stage 2 only", w2),
          ("W3 suite yamls = base + 2 keys", w3), ("W4 check_stage2_deploy", w4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help="default: pick.json's prep list")
    ap.add_argument("--pick", type=Path, default=CMP / "v12_107/pick.json")
    ap.add_argument("--sim", type=Path, default=c2.SIM, help="parent of <name>, <name>_s2")
    ap.add_argument("--configs", type=Path, default=CMP / "v12_110/configs")
    ap.add_argument("--prep", type=Path, default=c2.PREP,
                    help="W4: dir with the stage-2 walks.json / envelope.json")
    ap.add_argument("--s2a", type=Path, default=c2.S2A, help="W4: the stage-2 prep buffer")
    a = ap.parse_args()
    c2.PREP, c2.S2A = a.prep, a.s2a
    names = a.names or json.loads(a.pick.read_text())["prep"]
    t0, n_fail = time.perf_counter(), 0
    for name in names:
        for label, fn in CHECKS:
            try:
                detail = fn(name, a)
            except AssertionError as exc:
                print(f"[FAIL] {name} {label}: {exc}", file=sys.stderr)
                n_fail += 1
                continue
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                print(f"[FAIL] {name} {label}: {exc!r}", file=sys.stderr)
                n_fail += 1
                continue
            print(f"[PASS] {name} {label}: {detail}")
    if n_fail:
        print(f"{n_fail} WHOLE-RUN DEPLOY CHECK(S) FAILED", file=sys.stderr)
        return 1
    print(f"ALL WHOLE-RUN DEPLOY CHECKS PASSED ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
