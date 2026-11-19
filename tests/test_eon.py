import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eon import physics  # noqa: E402
from eon.policies import POLICIES  # noqa: E402
from eon.sim import Simulator, free_blocks, run  # noqa: E402
from eon.topology import load  # noqa: E402


def test_free_blocks():
    st, ln = free_blocks(np.array([1, 1, 0, 1, 0, 0, 1, 1, 1], bool))
    assert st.tolist() == [0, 3, 6] and ln.tolist() == [2, 1, 3]


def test_reach_is_monotonic():
    t = physics.reach_table()
    reach = [r["reach_km"] for r in t]
    assert reach == sorted(reach, reverse=True)
    assert t[0]["name"] == "BPSK" and t[0]["reach_km"] > 5000


def test_gsnr_has_an_optimum():
    p = physics.LinkParams()
    opt = physics.optimum_launch_dbm(p)
    g = physics.gsnr_db(10, p, np.array([opt - 3, opt, opt + 3]))
    assert g[1] > g[0] and g[1] > g[2]


def test_topologies():
    n = load("nsfnet")
    assert n.n_nodes == 14 and n.n_links == 22
    assert len(n.paths[(1, 14)]) == 5
    g = load("germany50", k=2)
    assert g.n_nodes == 50 and g.n_links == 88


def test_allocation_respects_continuity_and_release():
    net = load("nsfnet")
    sim = Simulator(net, load=100, seed=1)
    c = sim.candidates(1, 14, 400)[0]
    st = int(c.starts[0])
    sim.commit(c, st, 400, hold=0.5)
    assert sim.spec.occ[c.path.link_ids, st:st + c.need].all()
    sim.t = 10.0
    sim.events and sim.next_request()
    assert not sim.spec.occ.any()


def test_policies_order_at_high_load():
    net = load("nsfnet")
    bp = {k: run(net, POLICIES[k], 200, n_requests=3000, warmup=1000, seed=3)["bp"]
          for k in ("SP-FF", "KSP-FF", "Frag-Aware")}
    assert bp["SP-FF"] > bp["KSP-FF"] > bp["Frag-Aware"]
