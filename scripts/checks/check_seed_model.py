#!/usr/bin/env python3
"""Checks of the V12-46 loop keys ``seed_model`` (iteration 0 = an external checkpoint) and
``response_gate`` (report-only V12-44 statistic). No sim; CPU only (CUDA_VISIBLE_DEVICES="").

S1 config: ``check_cfg`` refuses bad ``seed_model`` / ``response_gate`` blocks; v12_36.yaml,
   v12_46.yaml accepted.
S2 dry-run identity pre vs tree (stdout bytes): v12_36.yaml iterations 0-9, C1_transport.yaml 0/1.
S3 v12_46 dry-run on a fresh root, iterations 0-2: rc 0; seed step adopts seed_model (no
   refit_from_buffer in iteration 0); refits 1 / 2 --init-checkpoint = the seed_model checkpoint,
   iteration 1 --gate-ref = its report; no --mppi-demo-actions.
S4 iteration 0 on a scratch root (seed, trust, export on CPU) + the collect / eval yamls: merged
   seed buffer bytes == V12-35's (= T1 rows), model.json / gates.json / refit.json adopt the
   checkpoint, deploy == the V17d sim arm's (float noise), collect yaml == sim45 V17d_prog.yaml
   minus loop-generated keys (seed, tol_1 at 1e-6, the explicit prior zeros = defaults); a
   checkpoint that is not the report's is refused; ``st_refit`` (stubbed refit) at iteration 1
   inits from the seed_model checkpoint.
S5 response_gate.compute == sim44/diag/gate_out.md (V17a / V17b / V17c on its 1719 states).

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_seed_model.py \\
        --pre-dir <pre-change tree (repo layout)> [--only S1,..] [--out <json>]
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CMP = REPO / "data/lcs/realfirst/cmp"
CFG = CMP / "configs"
ES = CMP / "encoder_study"
V17D = ES / "sim/V17d"


def req(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def load_loop(d: Path | None, tag: str):
    path = REPO / "scripts/lcs/realfirst_loop.py" if d is None else d / "scripts/lcs/realfirst_loop.py"
    mod = types.ModuleType(f"realfirst_loop_{tag}")
    mod.__file__ = str(REPO / "scripts/lcs/realfirst_loop.py")
    sys.modules[mod.__name__] = mod
    exec(compile(path.read_text(), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def dry_main(rl, argv: list[str]) -> str:
    buf, old = io.StringIO(), sys.argv
    sys.argv = ["realfirst_loop.py", *argv]
    try:
        with contextlib.redirect_stdout(buf):
            rl.main()
    finally:
        sys.argv = old
    return buf.getvalue()


def ns(root: Path, i: int, cfg: Path, dry: bool):
    return argparse.Namespace(smoke=False, root=str(root), iter=i, config=str(cfg), dry_run=dry,
                              collect_port=None, eval_port=None, force=None, ignore_stop=False)


def cfg46() -> dict:
    return yaml.safe_load((CFG / "v12_46.yaml").read_text())


def s1(ctx) -> str:
    rl = ctx.tree
    base = rl.merge(rl.DEFAULTS, yaml.safe_load((CFG / "v12_36.yaml").read_text()))
    rl.check_cfg(base)
    rl.check_cfg(rl.merge(rl.DEFAULTS, cfg46()))
    bad = [{"seed_model": "x.pt"}, {"seed_model": {"checkpoint": "a.pt"}},
           {"seed_model": {"checkpoint": "a.pt", "refit_report": "r.json", "tag": 1}},
           {"seed_model": {"checkpoint": "", "refit_report": "r.json"}},
           {"seed_model": {"checkpoint": "a.pt", "refit_report": "r.json", "source": 3}},
           {"response_gate": True}, {"response_gate": {"stride": 0}},
           {"response_gate": {"stride": 2.0}}, {"response_gate": {"max_cos": "x"}},
           {"response_gate": {"k": 1}}, {"response_gate": {"stride": True}}]
    for b in bad:
        try:
            rl.check_cfg({**copy.deepcopy(base), **b})
        except SystemExit:
            continue
        raise AssertionError(f"accepted {b}")
    rl.check_cfg({**copy.deepcopy(base), "response_gate": {}})
    return f"{len(bad)} bad blocks refused; v12_36, v12_46, response_gate {{}} accepted"


def s2(ctx) -> str:
    if ctx.pre_dir is None:
        return "skipped (no --pre-dir)"
    cases = [("v12_36", CMP / "v12_36", range(10)), ("C1_transport", CMP / "C1_transport", (0, 1))]
    n = 0
    for name, root, iters in cases:
        for i in iters:
            argv = ["--root", str(root), "--config", str(CFG / f"{name}.yaml"), "--iter", str(i),
                    "--step", "all", "--dry-run"]
            a = dry_main(load_loop(ctx.pre_dir, f"p_{name}{i}"), argv)
            b = dry_main(load_loop(None, f"t_{name}{i}"), argv)
            req(a == b and len(a) > 200, f"{name} iter {i}: dry-run differs")
            n += 1
    return f"{n}/{n} dry-runs identical (v12_36 0-9, C1_transport 0/1)"


def s3(ctx) -> str:
    sm = cfg46()["seed_model"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "v12_46"
        outs = []
        for i in (0, 1, 2):
            outs.append(dry_main(load_loop(None, f"s3_{i}"), [
                "--root", str(root), "--config", str(CFG / "v12_46.yaml"), "--iter", str(i),
                "--step", "all", "--dry-run"]))
        req(not root.exists(), "dry-run wrote the root")
    req("seed_model: theta_0 = " + sm["checkpoint"] in outs[0]
        and "refit_from_buffer.py" not in outs[0], "iteration 0 seed")
    for i, o in ((1, outs[1]), (2, outs[2])):
        t = next(x for x in o.splitlines() if "refit_from_buffer.py" in x).split()
        req(t[t.index("--init-checkpoint") + 1] == sm["checkpoint"], f"i{i} init")
        if i == 1:
            req(t[t.index("--gate-ref") + 1] == sm["refit_report"], "i1 gate-ref")
    req(not any("--mppi-demo-actions" in o or "prior_" in o for o in outs), "prior in dry-run")
    req(all("udpm://239.255.76.178:7778" in o and "7779" in o for o in outs), "ports")
    return "iterations 0-2 resolve; init = seed_model checkpoint (i1, i2), i1 gate-ref its report"


def strip_hdr(p: Path) -> dict:
    return yaml.safe_load("\n".join(x for x in p.read_text().splitlines()
                                    if not x.startswith("#")))


def s4(ctx) -> str:
    rl = load_loop(None, "s4")
    _e = rl.export_cmds

    def cpu(*a, **k):
        c, e, d, ch = _e(*a, **k)
        return c + ["--device", "cpu"], e, d, ch  # the cache encoder defaults to cuda

    rl.export_cmds = cpu
    sm = cfg46()["seed_model"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = cfg46()
        cfg.update(lcs_out=str(td / "lcs"), seed_buffer_merged=str(td / "root/seed_buffer_merged.json"))
        (td / "cfg.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        c = rl.Ctx(ns(td / "root", 0, td / "cfg.yaml", False))
        with contextlib.redirect_stdout(io.StringIO()):
            for st in ("seed", "trust", "export"):
                rl.do_step(c, st, False)
            col = rl.mppi_yaml(c, c.it() / "mppi_collect.yaml", cfg["action_noise_frac"],
                               cfg["seed"])
            ev = rl.mppi_yaml(c, c.it() / "eval/mppi_eval_hs_set2.yaml", 0.0, cfg["seed"])
        it = c.it()
        req((td / "root/seed_buffer_merged.json").read_bytes()
            == (CMP / "v12_35/seed_buffer_merged.json").read_bytes(), "merged seed buffer bytes")
        m, g, r = (rl.jload(it / f) for f in ("model.json", "gates.json", "refit.json"))
        req(m["checkpoint"] == sm["checkpoint"] and m["refit_report"] == sm["refit_report"]
            and m["id"] == "V18a" and m["decision"] == "seed" and m["seed_model_same_buffer_rows"]
            and g["seed_model"]["same_buffer_rows"] and r["report"] == sm["refit_report"]
            and not (td / "lcs/iter0/refit").exists(), f"iteration 0 model {m}")
        a = np.load(rl.jload(it / "deploy.json")["deploy"], allow_pickle=True)
        b = np.load(rl.jload(V17D / "deploy.json")["deploy"], allow_pickle=True)
        req(set(a.files) == set(b.files), "deploy keys")
        dmax = max(float(np.max(np.abs(a[k] - b[k]))) for k in a.files
                   if a[k].dtype.kind == "f" and k != "exported_at")
        req(dmax < 1e-5 and all(np.array_equal(a[k], b[k]) for k in a.files
                                if a[k].dtype.kind not in "fU" and k != "exported_at"),
            f"deploy vs V17d sim arm {dmax}")
        ref = strip_hdr(ES / "sim45/configs/V17d_prog.yaml")
        got = strip_hdr(col)
        diff = sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))
        req(diff == ["prior_alpha", "prior_weight", "seed", "stage_goal_tols"]
            and ref["prior_alpha"] == ref["prior_weight"] == 0.0
            and "prior_alpha" not in got and got["stage_goal_tols"][1] == -1.0
            and abs(got["stage_goal_tols"][0] - ref["stage_goal_tols"][0]) < 1e-6
            and got["H"] == 24 and got["stage_start"] == "progress", f"collect yaml {diff}")
        e = strip_hdr(ev)
        req(sorted(k for k in e if e[k] != got[k]) == ["action_noise_frac"]
            and e["action_noise_frac"] == 0.0, "eval yaml")
        bad = {**sm, "checkpoint": str(it / "box_0.json")}  # an existing file, not the report's
        cfg["seed_model"] = bad
        (td / "bad.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rl.st_seed(rl.Ctx(ns(td / "root_bad", 0, td / "bad.yaml", False)), False)
            raise AssertionError("mismatched seed_model accepted")
        except SystemExit:
            pass
        seen = {}
        rep = rl.jload(sm["refit_report"])

        def fake_refit(cx, cmd, tag, step):
            seen["cmd"] = [str(x) for x in cmd]
            return {**rep, "decision": "accepted", "checkpoint": str(td / "cand.pt"),
                    "gates": None, "timings_s": {"total": 0.0}}

        rl.run_refit = fake_refit
        rl.behaviour_check = lambda cx, ck: {"accept": True, "line": "ACCEPT (stub)"}
        (td / "root/iter0/buffer.json").write_bytes((td / "root/seed_buffer_merged.json")
                                                    .read_bytes())
        with contextlib.redirect_stdout(io.StringIO()):
            rl.st_refit(rl.Ctx(ns(td / "root", 1, td / "cfg.yaml", False)), False)
        cmd = seen["cmd"]
        m1 = rl.jload(td / "root/iter1/model.json")
        req(cmd[cmd.index("--init-checkpoint") + 1] == sm["checkpoint"]
            and cmd[cmd.index("--gate-ref") + 1] == sm["refit_report"]
            and m1["from_seed"] == "V18a" and m1["id"] == "V18b", f"st_refit {m1}")
    return (f"seed_model adopted (V18a, merged buffer == V12-35's bytes), deploy == V17d arm "
            f"(max {dmax:.1e}), collect yaml == V17d_prog minus {diff}, eval = collect + noise 0; "
            "mismatch refused; i1 refit init / gate-ref = V17d")


def s5(ctx) -> str:
    import response_gate as rgm

    t = os.path.getmtime(ES / "sim44/diag/gate_out.md")
    eps = []
    for f in sorted(glob.glob(str(ES / "sim*/V17*_*/index.json"))):
        if os.path.getmtime(f) > t:
            continue
        idx = json.loads(Path(f).read_text())
        if "summary" in idx:
            eps += [str(Path(f).parent / e["file"]) for e in idx["episodes"] if e.get("file")]
    want = {"V17a": (0.10, 0.01, True), "V17b": (0.70, 0.52, False), "V17c": (0.47, -0.12, False)}
    for m, (fx, fy, ok) in want.items():
        r = rgm.compute(json.loads((ES / "sim" / m / "deploy.json").read_text())["deploy"], eps)
        ch = r["channels"]
        req(r["n_states"] == 1719 and round(ch["F+x"]["cos_median"], 2) == fx
            and round(ch["F+y"]["cos_median"], 2) == fy and r["pass"] is ok
            and abs(ch["F+x"]["cos_median"] + ch["F-x"]["cos_median"]) < 1e-9, f"{m} {r}")
    return "V17a / V17b / V17c == gate_out.md (1719 states)"


CHECKS = [("S1 config", s1), ("S2 dry-run identity", s2), ("S3 v12_46 dry-run", s3),
          ("S4 iteration 0 + refit init", s4), ("S5 response gate", s5)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="pre-change tree with scripts/lcs/realfirst_loop.py (repo layout)")
    ap.add_argument("--only", default=None, help="comma list of S1..S5")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit('run with CUDA_VISIBLE_DEVICES=""')
    ctx = types.SimpleNamespace(tree=load_loop(None, "tree"), pre_dir=a.pre_dir)
    t0, ok, res = time.perf_counter(), 0, {}
    sel = [c for c in CHECKS if not a.only or c[0].split()[0] in a.only.split(",")]
    for name, fn in sel:
        t1 = time.perf_counter()
        try:
            detail = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - report every check
            traceback.print_exc()
            print(f"[FAIL] {name}: {exc!r}", flush=True)
            res[name] = {"pass": False, "detail": repr(exc)}
            continue
        ok += 1
        res[name] = {"pass": True, "detail": detail, "s": round(time.perf_counter() - t1, 1)}
        print(f"[PASS] {name}: {detail} ({time.perf_counter() - t1:.0f} s)", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{ok}/{len(sel)} {'PASS' if ok == len(sel) else 'FAIL'} ({time.perf_counter() - t0:.0f} s)")
    return 0 if ok == len(sel) else 1


if __name__ == "__main__":
    sys.exit(main())
