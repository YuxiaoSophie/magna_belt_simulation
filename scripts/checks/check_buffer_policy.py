#!/usr/bin/env python3
"""Check of ``task_common.buffer_policy`` on the v1 loop buffer (35 rows: 20 seed + 15 MPC).

No sim. The v1 buffer rows predate the ``status`` field: MPC rows are joined with e2's
``iter<i>/attempts.json`` (``status``, ``reason``, ``duration_s``) by episode path.
B0 no success evicted under any cap; B1 short-abort cap 5 evicts the oldest short aborts;
B2 failures leave in FIFO order, never below ``min_failures``; B3 weights; B4 identity with e2's
``realfirst_loop.fifo`` (successes exempt until N) on the v1 buffer and seeded synthetic buffers.
``--out-dir`` writes ``check.json`` and the iteration replay (policy applied after every v1
iteration, the loop's order).

Run:
    uv run --frozen python scripts/checks/check_buffer_policy.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "lcs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from realfirst_loop import fifo

from task_common import buffer_policy as bp

V1 = REPO_ROOT / "data" / "lcs" / "realfirst" / "v1"
INF = math.inf


def load_v1() -> tuple[list[dict], list[dict]]:
    """(raw rows, rows joined with the attempts' status / reason / duration_s)."""
    raw = json.loads((V1 / "buffer.json").read_text())["rollouts"]
    att = {}
    for f in sorted(V1.glob("iter*/attempts.json")):
        for a in json.loads(f.read_text())["attempts"]:
            if a.get("episode"):
                att[str(Path(a["episode"]).resolve())] = a
    joined = []
    for r in raw:
        a = att.get(str(Path(r["episode"]).resolve()))
        if r["kind"] == "mpc" and a is None:
            raise AssertionError(f"{r['id']}: no attempts row")
        extra = {k: a[k] for k in ("status", "reason", "duration_s")} if a else {"status": "ok"}
        joined.append({**r, **extra})
    return raw, joined


def ids(rows: list[dict]) -> list[str]:
    return [r["id"] for r in rows]


def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def check_b0(ctx) -> str:
    rows, n = ctx.rows, 0
    succ = {r["id"] for r in rows if bp.is_success(r)}
    for cap in range(len(rows) + 2):
        for sam in (0, 1, 5, INF):
            for mf in (0, 10):
                kept, ev, _ = bp.apply(rows, cap, sam, mf)
                _require(succ <= set(ids(kept)), f"cap {cap} sam {sam} mf {mf}: success evicted")
                _require(not succ & set(ids(ev)), "success in evicted")
                _require(len(kept) + len(ev) == len(rows), "rows lost")
                n += 1
    return f"{len(succ)} successes kept in all {n} (cap, short_abort_max, min_failures) runs"


def check_b1(ctx) -> str:
    rows = ctx.rows
    short = [r for r in rows if bp.is_short_abort(r)]
    mpc = [r for r in rows if r["kind"] == "mpc"]
    raw_short = sum(bp.is_short_abort(r) for r in ctx.raw)
    kept, ev, st = bp.apply(rows, cap=50, short_abort_max=5, min_failures=10)
    _require(ids(ev) == ids(short[:len(short) - 5]), f"evicted {ids(ev)}")
    _require(st["evicted_short_abort_ids"] == ids(ev) and not st["evicted_fifo_ids"], "fifo ran")
    _require(ids([r for r in kept if r["short_abort"]]) == ids(short[-5:]), "newest 5 not kept")
    _require(raw_short == 0, "raw v1 rows carry neither status nor reason")
    long_ab = [r["id"] for r in mpc if bp.is_aborted(r) and not bp.is_short_abort(r)]
    ctx.out["b1"] = {"short_abort_ids": ids(short), "evicted": ids(ev),
                     "long_aborts_kept": long_ab, "raw_rows_short": raw_short}
    return (f"{len(short)} of {len(mpc)} MPC rows are short aborts (< {bp.MIN_MPC_FRAMES} MPC "
            f"frames: {[r['n_mpc_frames'] for r in short]}); cap 5 evicts the oldest "
            f"{len(ev)} ({ids(ev)[0]}..{ids(ev)[-1]}); long abort kept {long_ab}; raw v1 rows "
            f"without the attempts join: {raw_short} qualify")


def check_b2(ctx) -> str:
    rows, lines = ctx.rows, []
    fails = [r for r in rows if not bp.is_success(r)]
    for cap, mf in ((30, 10), (25, 10), (20, 0), (5, 10)):
        kept, ev, st = bp.apply(rows, cap=cap, short_abort_max=INF, min_failures=mf)
        n_ev = min(max(0, len(rows) - cap), max(0, len(fails) - mf))
        _require(ids(ev) == ids(fails[:n_ev]), f"cap {cap}: evicted {ids(ev)}")
        _require(ids(kept) == [i for i in ids(rows) if i not in set(ids(ev))], "order changed")
        kf = sum(not bp.is_success(r) for r in kept)
        _require(kf >= min(mf, len(fails)), f"cap {cap}: {kf} failures < {mf}")
        _require(st["over_cap"] == (len(kept) > cap) and (not st["over_cap"]) == (not st["notes"]),
                 "over-cap note")
        lines.append(f"cap {cap}/min_f {mf}: -{len(ev)} -> {len(kept)} ({kf} failures"
                     + (", over cap" if st["over_cap"] else "") + ")")
    return "; ".join(lines)


def check_b3(ctx) -> str:
    kept, ev, _ = bp.apply(ctx.rows, cap=50, short_abort_max=5, min_failures=10)
    for r in kept + ev:
        want = bp.SHORT_ABORT_WEIGHT if bp.is_short_abort(r) else 1.0
        _require(r["weight"] == want == bp.weight_of(r) and r["short_abort"] == (want != 1.0),
                 f"{r['id']}: weight {r['weight']}")
    _require(all("weight" not in r for r in ctx.rows), "input rows mutated")
    w = [r["weight"] for r in kept]
    return (f"kept {len(kept)}: weight 0.25 x {w.count(0.25)}, 1.0 x {w.count(1.0)}, "
            f"sum {sum(w):.2f}; input rows not mutated")


def _synthetic(rng: np.random.Generator, n: int) -> list[dict]:
    return [{"id": f"s{i}", "kind": "mpc", "label_human": bool(rng.random() < 0.3),
             "status": "aborted" if rng.random() < 0.4 else "ok",
             "n_mpc_frames": int(rng.integers(30, 470))} for i in range(n)]


def check_b4(ctx) -> str:
    n = 0
    cases = [(ctx.rows, cap, ex) for cap in range(len(ctx.rows) + 2) for ex in (0, 5, 11, 20)]
    rng = np.random.default_rng(20261005)
    for _ in range(300):
        rows = _synthetic(rng, int(rng.integers(0, 90)))
        cases.append((rows, int(rng.integers(0, 60)), int(rng.integers(0, 25))))
    for rows, cap, ex in cases:
        k0, e0 = fifo(rows, cap, ex)
        k1, e1, _ = bp.apply(rows, cap=cap, short_abort_max=INF, min_failures=0,
                             success_exempt_until=ex)
        _require(ids(k0) == ids(k1) and ids(e0) == ids(e1), f"cap {cap} exempt {ex} differs")
        n += 1
    k0, e0 = fifo(ctx.rows, 50, 20)
    return (f"{n} cases (v1 buffer x caps 0..{len(ctx.rows) + 1} x exempt 0/5/11/20 + 300 "
            f"synthetic) identical to fifo(); fifo(v1, 50, 20) keeps {len(k0)}, evicts "
            f"{len(e0)}")


CHECKS = [("B0 successes never evicted", check_b0), ("B1 short-abort cap", check_b1),
          ("B2 failures FIFO", check_b2), ("B3 weights", check_b3),
          ("B4 identity with e2 fifo", check_b4)]


def replay(joined: list[dict]) -> list[dict]:
    """The policy (defaults) after every v1 iteration: B_{i+1} = apply(B_i + MPC_i)."""
    steps = [("seed", [r for r in joined if r["iter"] == -1])]
    for i in sorted({r["iter"] for r in joined if r["iter"] >= 0}):
        steps.append((f"after iter {i}", [r for r in joined if r["iter"] == i]))
    buf, out = [], []
    for name, new in steps:
        kept, _, st = bp.apply(buf + new)
        buf = [{k: v for k, v in r.items() if k not in ("weight", "short_abort")} for r in kept]
        out.append({"step": name, "added": ids(new), "n_in": st["in"]["n"],
                    "n_kept": st["kept"]["n"], "kept_success": st["kept"]["success"],
                    "kept_failure": st["kept"]["failure"],
                    "kept_mpc": st["kept"]["by_kind"].get("mpc", 0),
                    "kept_short_abort": st["kept"]["short_abort"],
                    "weight_sum": st["kept"]["weight_sum"], "evicted": st["evicted_ids"],
                    "weighted_0_25": [r["id"] for r in kept if r["weight"] != 1.0],
                    "notes": st["notes"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    # the sweeps hit the over-cap note on purpose; it is asserted via stats['notes']
    logging.getLogger(bp.__name__).setLevel(logging.ERROR)
    raw, joined = load_v1()
    ctx = type("Ctx", (), {})()
    ctx.raw, ctx.rows, ctx.out = raw, joined, {}
    code, results = 0, {}
    for name, fn in CHECKS:
        try:
            detail = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - report every check
            print(f"[FAIL] {name}: {exc!r}", file=sys.stderr)
            traceback.print_exc()
            results[name] = {"pass": False, "detail": repr(exc)}
            code = 1
            continue
        print(f"[PASS] {name}: {detail}")
        results[name] = {"pass": True, "detail": detail}
    rp = replay(joined)
    print("\nreplay (cap 50, short-abort max 5 / weight 0.25, min failures 10):")
    for s in rp:
        print(f"  {s['step']:<13} in {s['n_in']:2d} -> kept {s['n_kept']:2d} (succ "
              f"{s['kept_success']}, fail {s['kept_failure']}, mpc {s['kept_mpc']}, short "
              f"{s['kept_short_abort']}, weight sum {s['weight_sum']:.2f}); evicted "
              f"{s['evicted'] or '-'}; weighted {s['weighted_0_25'] or '-'}")
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "check.json").write_text(json.dumps(
            {"buffer": str(V1 / "buffer.json"), "checks": results, **ctx.out, "replay": rp},
            indent=1) + "\n")
    if code == 0:
        print(f"ALL BUFFER POLICY CHECKS PASSED ({time.perf_counter() - t0:.1f} s)")
    return code


if __name__ == "__main__":
    sys.exit(main())
