"""assets/spectrum.gif: NSFNET at 160 Erlang filling up under the fragmentation-aware policy.

Left: the network, each link coloured by the share of its 320 slots in use.
Right: the spectrum of every link (rows) over the 320 slots (columns).
Frames come from the same simulation as figure 4 (one snapshot every 250 requests).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402
from eon.policies import POLICIES  # noqa: E402
from eon.sim import run  # noqa: E402
from eon.topology import load  # noqa: E402


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    net = load("nsfnet")
    r = run(net, POLICIES["Frag-Aware"], 160, n_requests=8000, warmup=0, seed=11, trace_every=200)
    style.apply()
    cm = plt.get_cmap("turbo")
    S = r["trace"][0][1].shape[1]
    colors = np.array(cm(np.linspace(0.05, 0.95, S)))[:, :3]
    bg = np.array(matplotlib.colors.to_rgb(style.BG))
    for f, (i, occ) in enumerate(r["trace"][:40]):
        fig = plt.figure(figsize=(11, 4.6))
        ax = fig.add_axes([0.01, 0.06, 0.42, 0.82])
        u = occ.mean(axis=1)
        for (a, b), x in zip(net.links, u):
            (x1, y1), (x2, y2) = net.pos[a], net.pos[b]
            ax.plot([x1, x2], [y1, y2], color=plt.get_cmap("plasma")(0.15 + 0.85 * x), lw=1.5 + 7 * x,
                    solid_capstyle="round")
        xy = np.array([net.pos[n] for n in net.graph.nodes])
        ax.scatter(xy[:, 0], xy[:, 1], s=60, c="#22d3ee", edgecolors="white", zorder=3)
        ax.set_xlim(-126, -70)
        ax.set_axis_off()
        ax.set_title(f"NSFNET, request {i:,}: {100 * occ.mean():.0f} % of the spectrum in use", fontsize=12)
        ax2 = fig.add_axes([0.47, 0.13, 0.5, 0.75])
        rgb = np.where(occ[:, :, None], colors[None, :, :], bg[None, None, :])
        ax2.imshow(rgb, aspect="auto", interpolation="nearest")
        ax2.set(xlabel="frequency slot (12.5 GHz)", ylabel="link", title="spectrum of every link")
        ax2.grid(False)
        fig.savefig(os.path.join(out_dir, f"f{f:03d}.png"), dpi=90, facecolor=style.BG)
        plt.close(fig)
    print("frames:", min(40, len(r["trace"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/eon_frames")
