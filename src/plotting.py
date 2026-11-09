"""Shared figure style so every chart in the project reads as one system."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Categorical slots 1-3: a CVD-validated set (safe for all-pairs use, e.g. scatter/maps)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
# Diverging: orange (below) <- neutral gray -> blue (above)
DIVERGING = LinearSegmentedColormap.from_list("below_above", [C2, "#e4e3df", C1])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#e8f0fb", "#7fb0ea", C1, "#123f75"])

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110,
}


def use_style():
    plt.rcParams.update(STYLE)


def event(ax, date, label, y=0.98):
    """Dashed vertical marker for a dated event, labeled at fractional height y."""
    ax.axvline(pd.Timestamp(date), color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(pd.Timestamp(date), y, " " + label, transform=ax.get_xaxis_transform(),
            color=INK2, fontsize=8.5, va="top")


def year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def dollars(ax, axis="y"):
    fmt = plt.FuncFormatter(lambda v, _: f"${v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def source_note(fig, text):
    fig.text(0.01, -0.02, text, color=INK2, fontsize=8)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
