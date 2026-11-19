"""Event-driven simulator of a flex-grid elastic optical network.

Spectrum: every link has S frequency slots of 12.5 GHz (S = 320 fills the
4 THz C-band). A lightpath needs the same contiguous block of slots on every
link of its path (continuity + contiguity) plus one guard slot.

Traffic: Poisson arrivals, exponential holding time with mean 1, so the
offered load in Erlang equals the arrival rate. Source and destination are
uniform over node pairs; the bit rate is 100, 200 or 400 Gb/s with equal
probability.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .physics import LinkParams, ModulationPicker, slots_needed

RATES = (100, 200, 400)


def free_blocks(free):
    """Start indices and lengths of the runs of True in a boolean vector."""
    f = np.concatenate(([0], free.view(np.int8), [0]))
    d = np.diff(f)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


@dataclass
class Candidate:
    path_idx: int
    path: object
    fmt: dict
    need: int
    starts: np.ndarray     # free blocks on the path that fit the demand
    lens: np.ndarray


class Spectrum:
    def __init__(self, n_links, S):
        self.S = S
        self.occ = np.zeros((n_links, S), dtype=bool)

    def path_free(self, link_ids):
        return ~self.occ[link_ids].any(axis=0)

    def allocate(self, link_ids, start, n):
        assert not self.occ[link_ids, start:start + n].any()
        self.occ[link_ids, start:start + n] = True

    def release(self, link_ids, start, n):
        self.occ[link_ids, start:start + n] = False

    def cuts(self, link_ids, start, n):
        """Free slots left directly next to the block on each link (new fragments)."""
        c = 0
        if start > 0:
            c += int((~self.occ[link_ids, start - 1]).sum())
        if start + n < self.S:
            c += int((~self.occ[link_ids, start + n]).sum())
        return c

    def utilisation(self):
        return self.occ.mean()

    def fragmentation(self):
        """Mean external fragmentation over links: 1 - largest free block / free slots."""
        vals = []
        for row in self.occ:
            free = ~row
            nf = free.sum()
            if nf == 0:
                continue
            _, lens = free_blocks(free)
            vals.append(1 - lens.max() / nf)
        return float(np.mean(vals)) if vals else 0.0


class Simulator:
    def __init__(self, net, load, S=320, seed=0, params=LinkParams(), fixed_format=None,
                 rates=RATES):
        self.net = net
        self.load = load
        self.S = S
        self.rng = np.random.default_rng(seed)
        self.spec = Spectrum(net.n_links, S)
        self.picker = ModulationPicker(params, fixed=fixed_format)
        self.rates = rates
        self.nodes = list(net.graph.nodes)
        self.t = 0.0
        self.events = []        # (t_end, id, link_ids, start, n)
        self._id = 0
        self._cand_cache = {}
        self.stats = dict(arrived=0, blocked=0, bw_arrived=0.0, bw_blocked=0.0,
                          by_rate={r: [0, 0] for r in rates}, by_hops={},
                          formats={})

    # traffic -----------------------------------------------------------------
    def next_request(self):
        self.t += self.rng.exponential(1.0 / self.load)
        while self.events and self.events[0][0] <= self.t:
            _, _, ids, st, n = heapq.heappop(self.events)
            self.spec.release(ids, st, n)
        i, j = self.rng.choice(len(self.nodes), 2, replace=False)
        rate = self.rates[self.rng.integers(len(self.rates))]
        hold = self.rng.exponential(1.0)
        return self.nodes[i], self.nodes[j], rate, hold

    # candidates ----------------------------------------------------------------
    def _static(self, s, d, rate):
        key = (s, d, rate)
        if key not in self._cand_cache:
            out = []
            for k, p in enumerate(self.net.paths[(s, d)]):
                fmt = self.picker.pick(p.spans)
                if fmt is None:
                    continue
                out.append((k, p, fmt, slots_needed(rate, fmt["bits"])))
            self._cand_cache[key] = out
        return self._cand_cache[key]

    def candidates(self, s, d, rate):
        """Every path with its format, slot count and the free blocks that fit."""
        out = []
        for k, p, fmt, need in self._static(s, d, rate):
            st, ln = free_blocks(self.spec.path_free(p.link_ids))
            ok = ln >= need
            out.append(Candidate(k, p, fmt, need, st[ok], ln[ok]))
        return out

    # decisions -----------------------------------------------------------------
    def commit(self, cand, start, rate, hold):
        ids = cand.path.link_ids
        self.spec.allocate(ids, start, cand.need)
        self._id += 1
        heapq.heappush(self.events, (self.t + hold, self._id, ids, start, cand.need))
        self.stats["formats"][cand.fmt["name"]] = self.stats["formats"].get(cand.fmt["name"], 0) + 1

    def record(self, rate, blocked, hops=None):
        st = self.stats
        st["arrived"] += 1
        st["bw_arrived"] += rate
        st["by_rate"][rate][0] += 1
        if blocked:
            st["blocked"] += 1
            st["bw_blocked"] += rate
            st["by_rate"][rate][1] += 1
        if hops is not None:
            h = st["by_hops"].setdefault(hops, [0, 0])
            h[0] += 1
            h[1] += int(blocked)

    def reset_stats(self):
        self.stats = dict(arrived=0, blocked=0, bw_arrived=0.0, bw_blocked=0.0,
                          by_rate={r: [0, 0] for r in self.rates}, by_hops={}, formats={})


def run(net, policy, load, n_requests=30000, warmup=5000, seed=0, S=320,
        fixed_format=None, trace_every=0, rates=RATES):
    """Simulate n_requests after warm-up with a heuristic policy.

    policy(sim, candidates) -> (Candidate, start) or None.
    Returns a dict with request/bandwidth blocking, utilisation and
    fragmentation (sampled every 100 requests) and an optional trace.
    """
    sim = Simulator(net, load, S=S, seed=seed, fixed_format=fixed_format, rates=rates)
    util, frag, trace = [], [], []
    for i in range(warmup + n_requests):
        if i == warmup:
            sim.reset_stats()
        s, d, rate, hold = sim.next_request()
        cands = sim.candidates(s, d, rate)
        choice = policy(sim, cands) if cands else None
        shortest_hops = net.paths[(s, d)][0].hops
        if choice is None:
            sim.record(rate, True, shortest_hops)
        else:
            sim.commit(choice[0], choice[1], rate, hold)
            sim.record(rate, False, shortest_hops)
        if i >= warmup and i % 100 == 0:
            util.append(sim.spec.utilisation())
            frag.append(sim.spec.fragmentation())
        if trace_every and i % trace_every == 0:
            trace.append((i, sim.spec.occ.copy()))
    st = sim.stats
    return dict(bp=st["blocked"] / st["arrived"], bbp=st["bw_blocked"] / st["bw_arrived"],
                blocked=st["blocked"], arrived=st["arrived"],
                util=float(np.mean(util)), frag=float(np.mean(frag)),
                by_rate={r: v[1] / max(v[0], 1) for r, v in st["by_rate"].items()},
                by_hops={h: v[1] / max(v[0], 1) for h, v in sorted(st["by_hops"].items())},
                formats=st["formats"], trace=trace)
