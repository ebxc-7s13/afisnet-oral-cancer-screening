"""Supplementary Fig. S1 (v10): identical to the v9 figure except that the reference bar is labelled as the
generator partition (431 images = 366 fitting + 65 checkpoint selection) instead of 'Real training'.
Usage: python make_figS1_v10.py <revision11 results dir> <out dir>"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.8, "ytick.major.size": 2.8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "pdf.fonttype": 42,
    "axes.titlelocation": "left", "axes.titleweight": "normal", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333", "text.color": "#222222",
})
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
ORDER = ["normal", "smoke", "cancer"]
LABEL = {"normal": "Normal", "smoke": "Suspicious", "cancer": "Cancer"}
COL = {"normal": "#3b6ea8", "smoke": "#c9851a", "cancer": "#b8282f"}      # validated categorical set (CVD-safe)
def panel(ax, letter, title):
    ax.set_title(title, loc="left", fontsize=8, pad=6)
    ax.text(-0.24, 1.10, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)


# ================================================================ supplementary: generator fidelity vs unseen images
G = pd.read_csv(SRC / "gan_unseen" / "gan_metrics_vs_unseen.csv") if (SRC / "gan_unseen").exists() or True else None
if G is not None:
    sets = ["Real training partition (region-disjoint)", "NTP all (unfiltered)", "NTP QC-clean (hygiene filter)", "NTP QC-selected (215)"]
    short = ["Real\n(431)", "All\ngenerated", "QC-\nclean", "QC-\nselected"]
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.35), gridspec_kw={"wspace": 0.42})
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.22)
    cls_col = {"normal": COL["normal"], "cancer": COL["cancer"]}
    for ax, (metric, lab, letter, ttl) in zip(axes, (("KID_mean", "KID", "A", "Kernel Inception Distance"),
                                                     ("precision_matched", "Precision", "B", "Precision (realism)"),
                                                     ("recall_matched", "Recall", "C", "Recall (diversity)"))):
        panel(ax, letter, ttl)
        for k, c in enumerate(("normal", "cancer")):
            g = G[G.cls == c].set_index("set").reindex(sets)
            x = np.arange(len(sets)) + (k - 0.5) * 0.36
            yerr = g.KID_sd.values if metric == "KID_mean" else None
            ax.bar(x, g[metric].values, width=0.34, color=cls_col[c], alpha=0.9 if k else 0.85, edgecolor="white", lw=0.6,
                   yerr=yerr, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK), zorder=3, label=c.capitalize())
        ax.set_xticks(range(len(sets))); ax.set_xticklabels(short, fontsize=6.3, linespacing=1.0)
        ax.set_ylabel(lab); ygrid(ax)
        if metric != "KID_mean":
            ax.set_ylim(0, 1.05)
        else:
            ax.axhline(0, color=INK, lw=0.6)
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.3)
    fig.savefig(OUT / "figS1_generator_vs_unseen.png", dpi=600); fig.savefig(OUT / "figS1_generator_vs_unseen.pdf")
    plt.close(fig)
print("written", OUT)
