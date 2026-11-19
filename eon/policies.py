"""Heuristic RMSA policies.

Each policy receives the simulator and the list of candidate paths (already
filtered by modulation reach, each with the free blocks that fit) and returns
``(candidate, start_slot)`` or ``None`` to block.
"""
from __future__ import annotations

import numpy as np


def sp_ff(sim, cands):
    """Shortest path, first fit: only the shortest feasible-by-reach path."""
    c = cands[0]
    if c.path_idx != 0 or c.starts.size == 0:
        return None
    return c, int(c.starts[0])


def ksp_ff(sim, cands):
    """k shortest paths in order, first fit on the first path that has room."""
    for c in cands:
        if c.starts.size:
            return c, int(c.starts[0])
    return None


def ksp_bf(sim, cands):
    """First path with room, best fit: the smallest free block that fits."""
    for c in cands:
        if c.starts.size:
            j = int(np.argmin(c.lens))
            return c, int(c.starts[j])
    return None


def least_resources(sim, cands):
    """Path that uses the fewest slot-links (slots x hops), first fit."""
    best, cost = None, None
    for c in cands:
        if c.starts.size:
            v = c.need * c.path.hops
            if cost is None or v < cost:
                best, cost = (c, int(c.starts[0])), v
    return best


def placements(c):
    """Candidate start slots: both edges of every block that fits."""
    lo = c.starts
    hi = c.starts + c.lens - c.need
    return np.unique(np.concatenate([lo, hi]))


def make_fragmentation_aware(w=1.0):
    """Minimise slot-links used + w x (new fragments cut on the path's links).

    Placing a block flush against occupied slots on every link of the path
    leaves no small gaps behind; the cut count measures how many links get a
    new free sliver next to the block.
    """
    def fa(sim, cands):
        best, cost = None, None
        for c in cands:
            if c.starts.size == 0:
                continue
            res = c.need * c.path.hops
            for st in placements(c):
                v = res + w * sim.spec.cuts(c.path.link_ids, int(st), c.need)
                if cost is None or v < cost - 1e-9:
                    best, cost = (c, int(st)), v
        return best
    fa.__name__ = f"fa_w{w:g}"
    return fa


ksp_fa = make_fragmentation_aware(1.0)

POLICIES = {
    "SP-FF": sp_ff,
    "KSP-FF": ksp_ff,
    "KSP-BF": ksp_bf,
    "Least-Resources": least_resources,
    "Frag-Aware": ksp_fa,
}
