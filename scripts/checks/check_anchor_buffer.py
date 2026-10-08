#!/usr/bin/env python3
"""Checks of the MILES anchor buffer (``buffer_policy`` ``anchor_kinds`` / ``anchor_weights`` /
``anchor_frac``, ``realfirst_loop.py`` ``anchor`` + ``collection_schedule``). No sim, no GPU.

A1 identity: ``apply`` without anchors == ``anchor_kinds=None`` == the pre-change module (rows,
   evicted, stats) on the v1 buffer and 300 seeded synthetic buffers over caps.
A2 eviction exemption: with anchors every anchor row is kept for any cap (also short-aborted /
   failed anchors), and the pool's kept / evicted rows == ``apply`` of the pool alone.
A3 cap on the pool only: kept pool rows <= cap unless the min_failures floor note; anchors may
   exceed cap; stats ``pool`` / ``anchors`` counts.
A4 frame share: ``anchor_weights`` on synthetic buffers (pool weights 1 / 0.25, random frames):
   sum_anchor(w f) / sum(w f) == frac, pool weights unchanged, anchor weight ratios kept, all > 0;
   a seeded per-tuple weighted draw (refit_from_buffer's sampler) lands at frac; degenerate cases
   unchanged.
A5 decay: ``anchor_frac`` frac at refit 1, decay_to at decay_by, linear between, held after.
A6 loop: ``st_buffer`` replay of V12-18 iterations 0-3 (recorded attempts / labels, the merged
   seed) on a scratch root: anchor off -> buffer.json / allowlist.json bytes pre == tree; anchor
   on (demo, frac 0.4, decay 0.2) -> no demo evicted, pool <= cap, the anchor frame share of
   every buffer == anchor_frac(i + 1) recomputed from the episodes' frame counts.
A7 config / schedule: ``check_cfg`` refusals (anchor without buffer_policy, bad frac / decay /
   kinds; bad schedule keys, schedule + pool rule); ``draw_start`` with ``collection_schedule``
   = entry i mod n for i 0..12, eval starts refused (sha), --dry-run collect line names the entry.

    CUDA_VISIBLE_DEVICES="" uv run --frozen python scripts/checks/check_anchor_buffer.py \\
        --pre-dir <dir with the pre-change src/task_common/buffer_policy.py and
                   scripts/lcs/realfirst_loop.py (repo layout)>
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import math
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

from task_common import buffer_policy as bp

V1 = REPO / "data/lcs/realfirst/v1"
V18 = REPO / "data/lcs/realfirst/cmp/v12_18"
CFG18 = REPO / "data/lcs/realfirst/cmp/configs/v12_18.yaml"
INF = math.inf
KINDS = ("demo", "miles")


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_loop(d: Path | None, tag: str):
    path = REPO / "scripts/lcs/realfirst_loop.py" if d is None else d / "scripts/lcs/realfirst_loop.py"
    mod = types.ModuleType(f"realfirst_loop_{tag}")
    mod.__file__ = str(REPO / "scripts/lcs/realfirst_loop.py")
    sys.modules[mod.__name__] = mod
    exec(compile(path.read_text(), mod.__file__, "exec"), mod.__dict__)  # noqa: S102
    return mod


def ids(rows) -> list:
    return [r.get("id") for r in rows]


def v1_rows() -> list[dict]:
    raw = json.loads((V1 / "buffer.json").read_text())["rollouts"]
    att = {}
    for f in sorted(V1.glob("iter*/attempts.json")):
        for a in json.loads(f.read_text())["attempts"]:
            if a.get("episode"):
                att[str(Path(a["episode"]).resolve())] = a
    out = []
    for r in raw:
        a = att.get(str(Path(r["episode"]).resolve()))
        out.append({**r, **({k: a[k] for k in ("status", "reason")} if a else {"status": "ok"})})
    return out


def synthetic(rng, n: int, kinds=("demo", "miles", "excitation", "mpc")) -> list[dict]:
    rows = []
    for i in range(n):
        k = kinds[int(rng.integers(len(kinds)))]
        rows.append({"id": f"s{i}", "episode": f"/ep/{i}.npz", "kind": k,
                     "label_human": bool(rng.random() < (0.6 if k == "demo" else 0.25)),
                     "status": "aborted" if rng.random() < 0.35 else "ok",
                     "n_mpc_frames": int(rng.integers(20, 470))})
    return rows


def a1(ctx) -> str:
    n = 0
    cases = [(ctx.v1, cap, sam, mf, ex) for cap in (0, 10, 25, 50) for sam in (0, 5, INF)
             for mf in (0, 10) for ex in (None, 11)]
    rng = np.random.default_rng(20261006)
    for _ in range(300):
        cases.append((synthetic(rng, int(rng.integers(0, 90))), int(rng.integers(0, 60)),
                      [0, 2, 5, INF][int(rng.integers(4))], int(rng.integers(0, 12)), None))
    for rows, cap, sam, mf, ex in cases:
        ref = (ctx.pre_bp or bp).apply(rows, cap, sam, mf, ex)
        for got in (bp.apply(rows, cap, sam, mf, ex), bp.apply(rows, cap, sam, mf, ex,
                                                                 anchor_kinds=None)):
            _require(json.dumps(got, default=str) == json.dumps(ref, default=str),
                     f"cap {cap} sam {sam} mf {mf}: differs from the pre-change apply")
        n += 1
    return (f"{n} cases (v1 buffer x caps / short-abort max / min failures / exempt + 300 "
            f"synthetic): kept, evicted, stats identical to "
            f"{'the pre-change module' if ctx.pre_bp else 'apply() without the argument'}")


def a2(ctx) -> str:
    rng = np.random.default_rng(1)
    n_anchor = 0
    for _ in range(400):
        rows = synthetic(rng, int(rng.integers(1, 120)))
        cap, sam, mf = int(rng.integers(0, 60)), [0, 3, INF][int(rng.integers(3))], \
            int(rng.integers(0, 12))
        kept, ev, _ = bp.apply(rows, cap, sam, mf, anchor_kinds=KINDS)
        anc = [r["id"] for r in rows if r["kind"] in KINDS]
        _require(set(anc) <= set(ids(kept)) and not set(anc) & set(ids(ev)), "anchor evicted")
        pool = [r for r in rows if r["kind"] not in KINDS]
        pk, pe, _ = bp.apply(pool, cap, sam, mf)
        _require(ids([r for r in kept if r["kind"] not in KINDS]) == ids(pk)
                 and ids(ev) == ids(pe), "pool decision != apply(pool)")
        _require(ids(kept) == [i for i in ids(rows) if i not in set(ids(ev))], "order changed")
        n_anchor += len(anc)
    return (f"400 synthetic buffers ({n_anchor} anchor rows incl. aborted / failed): no anchor "
            "evicted; pool kept / evicted == apply(pool alone); order kept")


def a3(ctx) -> str:
    rng = np.random.default_rng(2)
    over = 0
    for _ in range(300):
        rows = synthetic(rng, int(rng.integers(1, 150)))
        cap, mf = int(rng.integers(0, 60)), int(rng.integers(0, 12))
        kept, _, st = bp.apply(rows, cap, 5, mf, anchor_kinds=KINDS)
        pool = [r for r in kept if r["kind"] not in KINDS]
        _require(len(pool) <= cap or st["notes"], f"pool {len(pool)} > cap {cap} without a note")
        _require(st["over_cap"] == (len(pool) > cap), "over_cap on the pool")
        _require(st["pool"]["n"] == len(pool) and st["anchors"]["n"] == len(kept) - len(pool)
                 and st["anchor_kinds"] == list(KINDS), "pool / anchor stats")
        over += len(kept) > cap
    rows = [{"id": f"d{i}", "kind": "demo", "label_human": False, "status": "ok"}
            for i in range(80)] + [{"id": f"m{i}", "kind": "mpc", "label_human": False,
                                    "status": "ok", "n_mpc_frames": 400} for i in range(40)]
    kept, ev, _ = bp.apply(rows, cap=50, short_abort_max=5, min_failures=10, anchor_kinds=KINDS)
    _require(len(kept) == 120 and not ev, "80 anchors + 40 pool under cap 50 evicted something")
    kept, ev, _ = bp.apply(rows + [{"id": f"x{i}", "kind": "mpc", "label_human": False,
                                    "status": "ok", "n_mpc_frames": 400} for i in range(20)],
                           cap=50, short_abort_max=5, min_failures=10, anchor_kinds=KINDS)
    _require(ids(ev) == [f"m{i}" for i in range(10)], f"fifo on the pool: {ids(ev)}")
    return (f"pool <= cap (or the floor note) in 300 buffers ({over} with anchors past cap); 80 "
            "failed demos + 40 MPC under cap 50: nothing evicted; +20 MPC: the 10 oldest MPC go")


def a4(ctx) -> str:
    rng = np.random.default_rng(3)
    worst, worst_mc = 0.0, 0.0
    for _ in range(300):
        rows = synthetic(rng, int(rng.integers(2, 80)))
        kept, _, _ = bp.apply(rows, 50, 5, 10, anchor_kinds=KINDS)
        frames = {r["episode"]: int(rng.integers(40, 600)) for r in kept}
        frac = float(rng.uniform(0.05, 0.95))
        out, st = bp.anchor_weights(kept, KINDS, frac, frames)
        anc = [r["kind"] in KINDS for r in out]
        if not any(anc) or all(anc):
            _require([r["weight"] for r in out] == [r["weight"] for r in kept]
                     and st["scale"] is None, "degenerate buffer reweighted")
            continue
        w = np.array([r["weight"] for r in out])
        f = np.array([frames[r["episode"]] for r in out])
        a = np.array(anc)
        share = (w * f)[a].sum() / (w * f).sum()
        worst = max(worst, abs(share - frac))
        _require(abs(share - frac) < 1e-12 and abs(st["share_after"] - frac) < 1e-12,
                 f"share {share} != {frac}")
        _require(all(r1["weight"] == r0["weight"] for r1, r0, x in zip(out, kept, anc) if not x),
                 "pool weight changed")
        w0 = np.array([r["weight"] for r in kept])
        _require(np.allclose(w[a] / w0[a], st["scale"], rtol=1e-12), "anchor ratios changed")
        _require(bool((w > 0).all()), "weight <= 0")
        _require(all("weight" in r for r in kept) and kept[0] is not out[0], "rows not copied")
        # refit_from_buffer: each tuple drawn with its episode's weight
        tw = np.repeat(w, f)
        draw = rng.choice(len(tw), 200_000, p=tw / tw.sum())
        mc = np.repeat(a, f)[draw].mean()
        worst_mc = max(worst_mc, abs(mc - frac))
    _require(worst_mc < 0.01, f"sampled share off by {worst_mc}")
    d = [{"id": "d", "episode": "d", "kind": "demo", "weight": 1.0}]
    _require(bp.anchor_weights(d, KINDS, 0.4, {"d": 100})[1]["scale"] is None, "anchors only")
    return (f"300 buffers: |share - frac| <= {worst:.1e}, pool weights (1 / 0.25) unchanged, "
            f"anchor ratios kept; 200k-tuple weighted draws within {worst_mc:.4f} of frac; "
            "anchors only / pool only unchanged")


def a5(ctx) -> str:
    f = bp.anchor_frac
    _require(f(0.4, None, 1) == 0.4 == f(0.4, None, 9), "no decay")
    seq = [f(0.5, 0.2, i) for i in range(1, 12)]
    want = [0.5 - 0.3 * min(i - 1, 8) / 8 for i in range(1, 12)]
    _require(np.allclose(seq, want, atol=1e-15), f"decay {seq}")
    _require(abs(f(0.5, 0.2, 5, decay_by=5) - 0.2) < 1e-15, "decay_by")
    return "frac at refit 1, linear to decay_to at decay_by (9), held after: " + \
        ", ".join(f"{x:.3f}" for x in seq[:9])


@contextlib.contextmanager
def buffer_policy_module(mod):
    old = sys.modules.get("task_common.buffer_policy")
    import task_common
    old_attr = getattr(task_common, "buffer_policy", None)
    sys.modules["task_common.buffer_policy"] = mod
    task_common.buffer_policy = mod
    try:
        yield
    finally:
        sys.modules["task_common.buffer_policy"] = old
        task_common.buffer_policy = old_attr


def fake_run(cmd, log, cwd=None, ctx=None, step=None) -> int:
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"\n# fake {' '.join(map(str, cmd))}\nPASS\n")
    return 0


def replay(rl, root: Path, over: dict, n_it: int, bp_mod) -> list[dict]:
    """st_buffer for iterations 0..n_it-1 on ``root`` with V12-18's attempts / labels."""
    cfg = yaml.safe_load(CFG18.read_text())
    cfg.update(seed_buffer_merged=str(V18 / "seed_buffer_merged.json"), **over)
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    rl.run = fake_run
    out = []
    with buffer_policy_module(bp_mod):
        for i in range(n_it):
            it = root / f"iter{i}"
            it.mkdir(parents=True, exist_ok=True)
            for f in ("attempts.json", "labels.json"):
                shutil.copy(V18 / f"iter{i}" / f, it / f)
            a = argparse.Namespace(smoke=False, root=str(root), iter=i, config=str(root / "cfg.yaml"),
                                   dry_run=True, collect_port=None, eval_port=None, force=None,
                                   ignore_stop=False)
            ctx = rl.Ctx(a)
            with contextlib.redirect_stdout(io.StringIO()):
                rl.st_buffer(ctx, False)
            out.append({"buffer": (it / "buffer.json").read_bytes(),
                        "allow": (it / "allowlist.json").read_bytes()})
    return out


def frames_of(ep: str, cache: dict) -> int:
    if ep not in cache:
        with np.load(ep, allow_pickle=True) as d:
            cache[ep] = len(d["actions"])
    return cache[ep]


def a6(ctx) -> str:
    n_it = len([i for i in range(10) if (V18 / f"iter{i}" / "labels.json").is_file()])
    n_it = min(n_it, 4)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tree = replay(ctx.tree_loop, td / "off", {}, n_it, bp)
        if ctx.pre_loop is not None:  # same root: buffer.json names its source path
            shutil.rmtree(td / "off")
            pre = replay(ctx.pre_loop, td / "off", {}, n_it, ctx.pre_bp)
            _require(all(p == t for p, t in zip(pre, tree)), "anchor off: bytes differ pre vs tree")
        rec = sum(json.loads(t["buffer"])["rollouts"] == json.loads(
            (V18 / f"iter{i}" / "buffer.json").read_text())["rollouts"] for i, t in enumerate(tree))
        anc = {"kinds": ["demo"], "frac": 0.4, "decay_to": 0.2}
        on = replay(ctx.tree_loop, td / "tree_on", {"anchor": anc}, n_it, bp)
        cache, lines = {}, []
        n_demo = sum(r["kind"] == "demo" for r in json.loads(
            (V18 / "seed_buffer_merged.json").read_text())["rollouts"])
        for i, x in enumerate(on):
            b = json.loads(x["buffer"])
            rows = b["rollouts"]
            demo = [r for r in rows if r["kind"] == "demo"]
            pool = [r for r in rows if r["kind"] != "demo"]
            _require(len(demo) == n_demo, f"iter {i}: {len(demo)} demos kept of {n_demo}")
            _require(len(pool) <= b["max"], f"iter {i}: pool {len(pool)} > cap")
            wf = [(r["weight"] * frames_of(r["episode"], cache), r["kind"] == "demo") for r in rows]
            share = sum(w for w, d in wf if d) / sum(w for w, _ in wf)
            want = bp.anchor_frac(0.4, 0.2, i + 1)
            _require(abs(share - want) < 1e-9 and b["anchor"]["refit_iter"] == i + 1,
                     f"iter {i}: share {share} != {want}")
            _require(all(r["weight"] == (bp.SHORT_ABORT_WEIGHT if r["short_abort"] else 1.0)
                         for r in pool), f"iter {i}: pool weight changed")
            off = json.loads(tree[i]["buffer"])["rollouts"]
            lines.append(f"i{i}: {len(demo)} demos (off {sum(r['kind'] == 'demo' for r in off)}),"
                         f" pool {len(pool)}, demo share {share:.3f} (x{b['anchor']['scale']:.2f})")
    return (f"anchor off: {n_it} iterations {'pre == tree bytes' if ctx.pre_loop else 'tree only'}"
            f" ({rec}/{n_it} rows == the recorded V12-18 buffers); on: " + "; ".join(lines))


def a7(ctx) -> str:
    rl = ctx.tree_loop
    base = yaml.safe_load(CFG18.read_text())
    bad = [{"anchor": {"frac": 0.4}, "buffer_policy": None}, {"anchor": {"frac": 1.0}},
           {"anchor": {"frac": 0.0}}, {"anchor": {"frac": None, "decay_to": 0.2}},
           {"anchor": {"decay_to": 1.2}}, {"anchor": {"kinds": []}},
           {"anchor": {"kinds": "demo"}}, {"anchor": {"decay_by": 1}},
           {"collection_schedule": []}, {"collection_schedule": ["pre_place_1_osc"]},
           {"collection_schedule": "ur_slant/ee_ur_z-30"},
           {"collection_schedule": ["ur_slant/ee_ur_z-30"],
            "collection_pool_rule": {"while_frac_gt": 0.0, "pool": ["x"], "then_pool": ["y"]}}]
    for b in bad:
        try:
            rl.check_cfg(rl.merge(rl.merge(rl.DEFAULTS, base), b))
        except SystemExit:
            continue
        raise AssertionError(f"config {b} accepted")
    rl.check_cfg(rl.merge(rl.merge(rl.DEFAULTS, base), {"anchor": {"frac": None}}))
    sched = ["grasp_variants/set2/gv_00", "ur_slant/ee_ur_z-30", "ur_pivot/up-20"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "cfg.yaml").write_text(yaml.safe_dump({**base, "collection_schedule": sched}))
        got = []
        for i in range(13):
            a = argparse.Namespace(smoke=False, root=str(root), iter=i,
                                   config=str(root / "cfg.yaml"), dry_run=True, collect_port=None,
                                   eval_port=None, force=None, ignore_stop=False)
            ctx_i = rl.Ctx(a)
            d = rl.draw_start(ctx_i)
            got.append(d["start"]["key"])
            _require(d["schedule"]["index"] == i % 3 and d["seed"] == base["seed"] + i, "entry")
        _require(got == [sched[i % 3] for i in range(13)], f"schedule {got}")
        a.iter = 0
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rl.st_collect(rl.Ctx(a), True)
        _require("start grasp_variants/set2/gv_00 (pool 3, unused 3; schedule entry 0/3)"
                 in buf.getvalue(), f"dry collect line: {buf.getvalue()}")
        (root / "cfg.yaml").write_text(yaml.safe_dump(
            {**base, "collection_schedule": ["grasp_variants/set1/gv_00"]}))
        _require(rl.draw_start(rl.Ctx(a))["start"]["key"] == "grasp_variants/set1/gv_00",
                 "a non-eval start of a set holding eval starts")
        (root / "cfg.yaml").write_text(yaml.safe_dump(
            {**base, "collection_schedule": ["grasp_variants/set1/gv_01"]}))
        try:
            rl.draw_start(rl.Ctx(a))
        except SystemExit as e:
            _require("eval start" in str(e), f"refusal text {e}")
        else:
            raise AssertionError("an eval start was accepted in collection_schedule")
    return (f"{len(bad)} bad anchor / schedule configs refused; schedule entries i mod 3 for i "
            "0..12 (seed + i); eval start set1/gv_01 refused, set1/gv_00 accepted; dry collect "
            "names the entry")


CHECKS = [("A1 identity without anchors", a1), ("A2 eviction exemption", a2),
          ("A3 cap on the pool only", a3), ("A4 frame share", a4), ("A5 decay", a5),
          ("A6 loop buffer step", a6), ("A7 config and schedule", a7)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-dir", type=Path, default=None,
                    help="pre-change files in the repo layout (A1 / A6 compare against them)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    logging.getLogger(bp.__name__).setLevel(logging.ERROR)
    ctx = types.SimpleNamespace(v1=v1_rows(), pre_bp=None, pre_loop=None,
                                tree_loop=load_loop(None, "tree"))
    if a.pre_dir:
        ctx.pre_bp = load_module(a.pre_dir / "src/task_common/buffer_policy.py", "bp_pre")
        logging.getLogger(ctx.pre_bp.__name__).setLevel(logging.ERROR)
        ctx.pre_loop = load_loop(a.pre_dir, "pre")
    t0, ok, res = time.perf_counter(), 0, {}
    sel = [c for c in CHECKS if not a.only or c[0].split()[0] in a.only.split(",")]
    for name, fn in sel:
        try:
            detail = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - report every check
            traceback.print_exc()
            print(f"[FAIL] {name}: {exc!r}")
            res[name] = {"pass": False, "detail": repr(exc)}
            continue
        ok += 1
        res[name] = {"pass": True, "detail": detail}
        print(f"[PASS] {name}: {detail}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{ok}/{len(sel)} {'PASS' if ok == len(sel) else 'FAIL'} ({time.perf_counter() - t0:.1f} s)")
    return 0 if ok == len(sel) else 1


if __name__ == "__main__":
    sys.exit(main())
