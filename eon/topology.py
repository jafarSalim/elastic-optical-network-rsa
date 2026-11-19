"""Network topologies and k-shortest paths.

Two reference networks are included in ``topologies/``:

* NSFNET as used in DeepRMSA (Chen et al., JLT 2019): 14 nodes, 22 links,
  link lengths in km from the paper.
* Germany50 from SNDlib: 50 nodes, 88 links, geographic coordinates. Link
  length is the great-circle distance times a routing factor of 1.25
  (fibre does not follow straight lines).
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from itertools import islice

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOPO_DIR = os.path.join(HERE, "..", "topologies")

# NSFNET node positions (longitude, latitude) for drawing only.
NSFNET_SITES = {
    1: ("Seattle", -122.3, 47.6), 2: ("Palo Alto", -122.1, 37.4),
    3: ("San Diego", -117.2, 32.7), 4: ("Salt Lake City", -111.9, 40.8),
    5: ("Boulder", -105.3, 40.0), 6: ("Houston", -95.4, 29.8),
    7: ("Lincoln", -96.7, 40.8), 8: ("Champaign", -88.2, 40.1),
    9: ("Pittsburgh", -80.0, 40.4), 10: ("Atlanta", -84.4, 33.7),
    11: ("Ann Arbor", -83.7, 42.3), 12: ("Ithaca", -76.5, 42.4),
    13: ("Princeton", -74.7, 40.4), 14: ("College Park", -76.9, 39.0),
}


@dataclass
class Network:
    name: str
    graph: nx.Graph
    pos: dict                      # node -> (x, y) for plots
    labels: dict                   # node -> name
    links: list = field(default_factory=list)   # [(u, v)] index = link id
    link_km: np.ndarray = None
    k: int = 5
    paths: dict = field(default_factory=dict)   # (s, d) -> list of Path

    @property
    def n_nodes(self):
        return self.graph.number_of_nodes()

    @property
    def n_links(self):
        return len(self.links)


@dataclass
class Path:
    nodes: tuple
    link_ids: np.ndarray
    km: float
    spans: int          # 80 km amplifier spans along the path
    hops: int


def _haversine(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_nsfnet():
    g = nx.Graph()
    with open(os.path.join(TOPO_DIR, "nsfnet_chen.txt")) as f:
        rows = [ln.split() for ln in f if ln.strip() and not ln.startswith("#")]
    n_nodes, n_links = int(rows[0][0]), int(rows[1][0])
    for i in range(1, n_nodes + 1):
        g.add_node(i)
    for u, v, km in rows[2:2 + n_links]:
        g.add_edge(int(u), int(v), km=float(km))
    pos = {i: (lon, lat) for i, (_, lon, lat) in NSFNET_SITES.items()}
    labels = {i: name for i, (name, _, _) in NSFNET_SITES.items()}
    return Network("NSFNET", g, pos, labels)


def load_germany50(routing_factor=1.25):
    ns = {"s": "http://sndlib.zib.de/network"}
    root = ET.parse(os.path.join(TOPO_DIR, "germany50.xml")).getroot()
    g = nx.Graph()
    pos = {}
    for nd in root.findall(".//s:node", ns):
        x = float(nd.find("s:coordinates/s:x", ns).text)
        y = float(nd.find("s:coordinates/s:y", ns).text)
        pos[nd.get("id")] = (x, y)
        g.add_node(nd.get("id"))
    for lk in root.findall(".//s:link", ns):
        u, v = lk.find("s:source", ns).text, lk.find("s:target", ns).text
        km = routing_factor * _haversine(*pos[u], *pos[v])
        g.add_edge(u, v, km=round(km, 1))
    labels = {n: n for n in g.nodes}
    return Network("Germany50", g, pos, labels)


def prepare(net: Network, k=5, span_km=80.0):
    """Index the links and compute the k shortest (by km) paths of every pair."""
    net.k = k
    net.links = [tuple(e) for e in net.graph.edges]
    net.link_km = np.array([net.graph.edges[e]["km"] for e in net.links])
    lid = {}
    for i, (u, v) in enumerate(net.links):
        lid[(u, v)] = lid[(v, u)] = i
    spans_link = np.ceil(net.link_km / span_km).astype(int)
    nodes = list(net.graph.nodes)
    for s in nodes:
        for d in nodes:
            if s == d:
                continue
            ps = []
            for p in islice(nx.shortest_simple_paths(net.graph, s, d, weight="km"), k):
                ids = np.array([lid[(p[i], p[i + 1])] for i in range(len(p) - 1)])
                ps.append(Path(tuple(p), ids, float(net.link_km[ids].sum()),
                               int(spans_link[ids].sum()), len(ids)))
            net.paths[(s, d)] = ps
    return net


def load(name, k=5):
    net = {"nsfnet": load_nsfnet, "germany50": load_germany50}[name.lower()]()
    return prepare(net, k=k)
