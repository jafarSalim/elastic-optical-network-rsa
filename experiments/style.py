"""Figure style: "prism" — spectral colours on deep navy."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BG = "#0a0f2c"
PANEL = "#111842"
FG = "#e6e9ff"
GRID = "#2a3170"
MUTED = "#8b93c9"

COLORS = {
    "SP-FF": "#8b93c9",
    "KSP-FF": "#22d3ee",
    "KSP-BF": "#fbbf24",
    "Least-Resources": "#a78bfa",
    "Frag-Aware": "#ff4d9d",
    "DRL (PPO)": "#a3e635",
}
MARKERS = {"SP-FF": "v", "KSP-FF": "o", "KSP-BF": "s", "Least-Resources": "D",
           "Frag-Aware": "*", "DRL (PPO)": "P"}
FORMAT_COLORS = {"BPSK": "#ef4444", "QPSK": "#f97316", "8QAM": "#facc15",
                 "16QAM": "#22c55e", "32QAM": "#3b82f6", "64QAM": "#a855f7"}

# occupancy colormap: empty navy -> spectral
SPECTRUM = LinearSegmentedColormap.from_list(
    "prism", ["#0a0f2c", "#3b0764", "#be185d", "#f97316", "#facc15", "#a3e635", "#22d3ee"])


def apply():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": FG, "axes.titlecolor": FG,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": FG,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID, "legend.framealpha": 0.9,
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12.5,
        "axes.titleweight": "bold", "axes.labelsize": 11, "lines.linewidth": 2.2,
        "figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight",
    })


def title(fig, text, sub=None):
    fig.suptitle(text, x=0.01, ha="left", fontsize=15, fontweight="bold", color=FG, y=1.02)
    if sub:
        fig.text(0.01, 0.975, sub, ha="left", va="top", fontsize=10.5, color=MUTED)
