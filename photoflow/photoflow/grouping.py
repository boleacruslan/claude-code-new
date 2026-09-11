"""Burst / near-duplicate grouping.

A wedding set is full of near-identical frames: three shots of the same kiss,
eight of the bouquet toss. Grouping them lets the culler surface one best
frame per moment instead of ranking every frame in isolation.
"""

from __future__ import annotations

from typing import Sequence

from .quality import hamming


def group_bursts(records: Sequence[dict], max_gap_seconds: float = 3.0,
                 max_hash_distance: int = 12) -> None:
    """Assign a ``group`` index to each record, in place.

    Records must already be ordered by capture time. Two neighbouring frames
    join the same group when they were shot close together *and* look alike;
    the comparison chains to the previous frame so a slowly drifting burst
    stays one group.
    """
    group = -1
    prev = None
    for rec in records:
        start_new = True
        if prev is not None:
            gap = _seconds_between(prev, rec)
            close_in_time = gap is not None and gap <= max_gap_seconds
            similar = hamming(prev["_hash"], rec["_hash"]) <= max_hash_distance
            start_new = not (close_in_time and similar)
        if start_new:
            group += 1
        rec["group"] = group
        prev = rec


def _seconds_between(a: dict, b: dict) -> float | None:
    ta, tb = a.get("capture_time"), b.get("capture_time")
    if ta is None or tb is None:
        return None
    return abs((tb - ta).total_seconds())


def mark_group_picks(records: Sequence[dict]) -> None:
    """Flag the highest-scoring frame in each group as the pick."""
    best: dict[int, dict] = {}
    for rec in records:
        rec["is_pick"] = False
        gid = rec["group"]
        current = best.get(gid)
        if current is None or rec["score"] > current["score"]:
            best[gid] = rec
    for rec in best.values():
        rec["is_pick"] = True
    sizes: dict[int, int] = {}
    for rec in records:
        sizes[rec["group"]] = sizes.get(rec["group"], 0) + 1
    for rec in records:
        rec["group_size"] = sizes[rec["group"]]
