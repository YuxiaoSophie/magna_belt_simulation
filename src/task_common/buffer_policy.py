"""Buffer policy of the real-first loop v1.1 (PLAN-20261005-realfirst-v1_1 §1 change 5).

Rows are buffer rollouts in age order (oldest first, as the loop appends them).

    successes   ``label_human`` True: never evicted
    short abort harness ``status`` "aborted" with ``n_mpc_frames`` < 134 (10 s at 75 ms); rows
                without ``status`` (v1 buffer rows) count as aborted when ``reason`` contains
                "abort"; at most ``short_abort_max`` kept (the oldest go first), refit weight 0.25
    failures    FIFO (oldest first) until ``len(kept) <= cap``, never below ``min_failures``
                failures: the buffer may then exceed ``cap`` (noted in the stats)

Frames before an abort are KEPT: the transitions are real; the policy caps and down-weights the
volume of short aborted rollouts, not their frames. ``success_exempt_until`` (default None =
never evicted) reproduces e2's ``realfirst_loop.fifo`` successes-exempt-until-N rule (check B4).

Anchor buffer (PLAN-20261006-miles, opt-in): ``anchor_kinds`` (default None = unchanged) rows
(e.g. demo, miles) are never evicted and do not count toward ``cap``; the short-abort cap, FIFO
and ``min_failures`` act on the other rows (the on-policy pool). ``anchor_weights`` then scales the
anchors' ``weight`` so their frames make ``frac`` of the frame-weighted total (sum of weight x
frames; ``refit_from_buffer.py`` draws each episode's tuples in proportion to its weight); pool
rows keep their weights (1.0, short aborts 0.25). ``anchor_frac``: linear decay over refits.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

MIN_MPC_FRAMES = 134
SHORT_ABORT_WEIGHT = 0.25

log = logging.getLogger(__name__)


def is_success(row: dict) -> bool:
    return row.get("label_human") is True


def is_aborted(row: dict) -> bool:
    if row.get("status") is not None:
        return row["status"] == "aborted"
    return "abort" in str(row.get("reason") or "").lower()


def is_short_abort(row: dict, min_mpc_frames: int = MIN_MPC_FRAMES) -> bool:
    return is_aborted(row) and int(row.get("n_mpc_frames") or 0) < min_mpc_frames


def weight_of(row: dict, min_mpc_frames: int = MIN_MPC_FRAMES) -> float:
    return SHORT_ABORT_WEIGHT if is_short_abort(row, min_mpc_frames) else 1.0


def counts(rows: list[dict], min_mpc_frames: int = MIN_MPC_FRAMES) -> dict:
    return {"n": len(rows), "by_kind": dict(Counter(str(r.get("kind")) for r in rows)),
            "success": sum(is_success(r) for r in rows),
            "failure": sum(not is_success(r) for r in rows),
            "short_abort": sum(is_short_abort(r, min_mpc_frames) for r in rows),
            "weight_sum": sum(weight_of(r, min_mpc_frames) for r in rows)}


def _num(x: float):
    return "inf" if isinstance(x, float) and math.isinf(x) else x


def is_anchor(row: dict, anchor_kinds) -> bool:
    return anchor_kinds is not None and row.get("kind") in anchor_kinds


def apply(rows: list[dict], cap: int = 50, short_abort_max: float = 5, min_failures: int = 10,
          success_exempt_until: int | None = None, min_mpc_frames: int = MIN_MPC_FRAMES,
          anchor_kinds=None) -> tuple[list[dict], list[dict], dict]:
    """``(kept, evicted, stats)``; rows are copied, kept rows gain ``weight`` and
    ``short_abort``, evicted rows also ``evicted_by`` ("short_abort_cap" | "fifo").
    ``anchor_kinds``: those rows are kept and outside ``cap`` (the pool = the other rows)."""
    kept = [dict(r) for r in rows]
    for r in kept:
        r["short_abort"] = is_short_abort(r, min_mpc_frames)
        r["weight"] = SHORT_ABORT_WEIGHT if r["short_abort"] else 1.0
    evicted: list[dict] = []

    def out(r: dict, why: str) -> None:
        kept.remove(r)
        evicted.append({**r, "evicted_by": why})

    def pool() -> list[dict]:
        return [r for r in kept if not is_anchor(r, anchor_kinds)]

    short = [r for r in pool() if r["short_abort"] and not is_success(r)]
    n_over = len(short) - short_abort_max
    for r in short[:int(n_over)] if n_over > 0 else []:
        out(r, "short_abort_cap")
    notes = []
    while len(pool()) > cap:
        rest = pool()
        fails = [r for r in rest if not is_success(r)]
        succ = [r for r in rest if is_success(r)]
        exempt = success_exempt_until is None or len(succ) <= success_exempt_until
        pick = (fails[0] if len(fails) > min_failures else None) if exempt else rest[0]
        if pick is not None and not is_success(pick) and len(fails) <= min_failures:
            pick = None
        if pick is None and success_exempt_until is not None and succ:
            pick = succ[0]
        if pick is None:
            notes.append(f"{len(rest)} rows > cap {cap}: {len(succ)} successes are never "
                         f"evicted and failures are at the floor ({len(fails)} <= "
                         f"{min_failures})")
            log.warning("buffer_policy: " + notes[-1])
            break
        out(pick, "fifo")
    stats = {"cap": cap, "short_abort_max": _num(short_abort_max), "min_failures": min_failures,
             "success_exempt_until": success_exempt_until, "min_mpc_frames": min_mpc_frames,
             "in": counts(rows, min_mpc_frames), "kept": counts(kept, min_mpc_frames),
             "evicted": counts(evicted, min_mpc_frames),
             "evicted_ids": [r.get("id") for r in evicted],
             "evicted_short_abort_ids": [r.get("id") for r in evicted
                                         if r["evicted_by"] == "short_abort_cap"],
             "evicted_fifo_ids": [r.get("id") for r in evicted if r["evicted_by"] == "fifo"],
             "over_cap": len(pool()) > cap, "notes": notes}
    if anchor_kinds is not None:  # absent by default: stats (buffer.json "policy") unchanged
        stats.update(anchor_kinds=list(anchor_kinds), pool=counts(pool(), min_mpc_frames),
                     anchors=counts([r for r in kept if is_anchor(r, anchor_kinds)],
                                    min_mpc_frames))
    return kept, evicted, stats


def anchor_frac(frac: float, decay_to: float | None, refit_iter: int, decay_by: int = 9) -> float:
    """Anchor frame share of the refit at ``refit_iter`` (>= 1): ``frac`` at the first refit,
    linearly to ``decay_to`` at iteration ``decay_by``, then held."""
    if decay_to is None:
        return float(frac)
    t = min(max((refit_iter - 1) / (decay_by - 1), 0.0), 1.0)
    return float(frac) + (float(decay_to) - float(frac)) * t


def anchor_weights(rows: list[dict], anchor_kinds, frac: float, frames: dict
                   ) -> tuple[list[dict], dict]:
    """``(rows, stats)``: copies whose anchor ``weight`` is scaled by s so that
    sum_anchor(w f) / sum_all(w f) == ``frac`` (``frames``: episode -> frame count); pool rows
    unchanged. No anchor or no pool frames: weights unchanged (``stats["scale"]`` None)."""
    out = [dict(r) for r in rows]
    anc = [r for r in out if is_anchor(r, anchor_kinds)]
    oth = [r for r in out if not is_anchor(r, anchor_kinds)]
    wa = sum(float(r["weight"]) * frames[r["episode"]] for r in anc)
    wo = sum(float(r["weight"]) * frames[r["episode"]] for r in oth)
    st = {"kinds": list(anchor_kinds), "frac": float(frac), "n_anchor": len(anc),
          "n_pool": len(oth), "frames_anchor": sum(frames[r["episode"]] for r in anc),
          "frames_pool": sum(frames[r["episode"]] for r in oth),
          "share_before": wa / (wa + wo) if wa + wo > 0 else None, "scale": None}
    if wa <= 0 or wo <= 0:
        st["note"] = "no anchor or no pool frames: weights unchanged"
        return out, st
    s = frac / (1.0 - frac) * wo / wa
    for r in anc:
        r["weight"] = float(r["weight"]) * s
    st.update(scale=s, share_after=sum(r["weight"] * frames[r["episode"]] for r in anc)
              / sum(r["weight"] * frames[r["episode"]] for r in out))
    return out, st
