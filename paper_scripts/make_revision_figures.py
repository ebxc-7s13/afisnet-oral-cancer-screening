"""Fig. 10 (screening of the suspicious group), Supplementary Fig. S1 (generator against the 177 unseen images) and
Supplementary Fig. S4 (format-matched control), plus the region-bootstrap forest plot (not in the paper), from the
output of notebooks/AFiSNet_Revision11_screening_and_checks.ipynb.
Usage: python make_revision_figures.py <afis_results/revision11> <out dir>"""
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
RNG = np.random.default_rng(7)


def panel(ax, letter, title):
    ax.set_title(title, loc="left", fontsize=8, pad=6)
    ax.text(-0.24, 1.10, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)


def fmt_p(p):
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        return f"$p$ < 10$^{{{e + 1}}}$"
    return f"$p$ = {p:.2g}" if p >= 1e-3 else f"$p$ = {p:.1e}"


def boot_region(d, stat, n=2000, seed=11):
    """95% CI of stat(df) from resampling imaging regions."""
    regs = d.region.unique(); idx = {r: np.where(d.region.values == r)[0] for r in regs}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        pick = np.concatenate([idx[r] for r in rng.choice(regs, len(regs), replace=True)])
        vals.append(stat(d.iloc[pick]))
    return np.percentile(vals, [2.5, 97.5])


def half_violin(ax, x, v, color, width=0.36, log=False, bw=None):
    from scipy.stats import gaussian_kde
    vv = np.log10(v) if log else v
    lo, hi = np.percentile(vv, [0.5, 99.5])
    grid = np.linspace(vv.min(), vv.max(), 300)
    k = gaussian_kde(vv, bw_method=bw)(grid)
    k = k / k.max() * width
    g = 10 ** grid if log else grid
    ax.fill_betweenx(g, x, x + k, color=color, alpha=0.28, lw=0, zorder=2)
    ax.plot(x + k, g, color=color, lw=0.8, zorder=3)


def raincloud(ax, data, col, log=False, bw=None, ylim=None):
    for i, g in enumerate(ORDER):
        v = data[data.group == g][col].values.astype(float)
        half_violin(ax, i + 0.04, v, COL[g], log=log, bw=bw)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(plt.Rectangle((i - 0.035, q1), 0.07, q3 - q1, fc="white", ec=INK, lw=0.7, zorder=5))
        ax.plot([i - 0.035, i + 0.035], [med, med], color=INK, lw=1.1, zorder=6)
        jx = i - 0.12 - RNG.uniform(0, 0.17, len(v))
        ax.scatter(jx, v, s=2.2, color=COL[g], alpha=0.55, lw=0, zorder=4, rasterized=True)
        m = v.mean()
        ax.scatter([i], [m], marker="D", s=14, color="white", edgecolor=INK, lw=0.7, zorder=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{LABEL[g]}\n$n$ = {(data.group == g).sum()}" for g in ORDER])
    ax.set_xlim(-0.45, 2.5)
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)


def bracket(ax, x1, x2, y, text, h=0.02, log=False):
    if log:
        y2 = y * (1 + h * 6)
        ax.plot([x1, x1, x2, x2], [y, y2, y2, y], color=INK, lw=0.6, clip_on=False)
        ax.text((x1 + x2) / 2, y2 * 1.04, text, ha="center", va="bottom", fontsize=6.2, clip_on=False)
    else:
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=INK, lw=0.6, clip_on=False)
        ax.text((x1 + x2) / 2, y + h + 0.008, text, ha="center", va="bottom", fontsize=6.2, clip_on=False)


# ---------------------------------------------------------------- data
d = pd.read_csv(SRC / "screening_all" / "per_image_primary.csv")
S = json.loads((SRC / "screening_all" / "screening_stats.json").read_text())
ens = pd.read_csv(SRC / "screening_all" / "smokers_5model_mean.csv")
every = pd.read_csv(SRC / "screening_all" / "smokers_scores_by_every_fold_model.csv")
jr = pd.read_csv(SRC / "screening_all" / "development_images_jpeg_rescored.csv")
q95 = S["design"]["novelty_reference_q95_608"]
thr_med = float(d.threshold.median())
pp = S["primary_probability"]["image"]["pairwise"]
pn = S["primary_novelty"]["image"]["pairwise"]

# ================================================================ main figure
fig = plt.figure(figsize=(7.09, 4.9))
gs = fig.add_gridspec(2, 3, left=0.085, right=0.985, top=0.91, bottom=0.1, wspace=0.5, hspace=0.62,
                      width_ratios=[1.25, 0.85, 1.2])

# A: cancer probability
ax = fig.add_subplot(gs[0, 0]); panel(ax, "A", "Predicted cancer probability")
raincloud(ax, d, "p_cancer", bw=0.12, ylim=(-0.03, 1.36))
ax.axhline(thr_med, xmax=0.985, ls=(0, (3, 2)), color=MUTED, lw=0.7, zorder=1)
ax.set_yticks(np.linspace(0, 1, 6)); ax.spines["left"].set_bounds(0, 1)
ax.set_ylabel("Cancer probability"); ygrid(ax)
[l.set_visible(False) for l in ax.get_ygridlines()[6:]]
bracket(ax, 0, 0.97, 1.05, fmt_p(pp["normal vs smoke"]["p_holm"]))
bracket(ax, 1.03, 2, 1.05, fmt_p(pp["smoke vs cancer"]["p_holm"]))
bracket(ax, 0, 2, 1.2, f"trend {fmt_p(S['primary_probability']['image']['jonckheere_normal<smoke<cancer']['p_one_sided'])}")

# B: share called cancer-like
ax = fig.add_subplot(gs[0, 1]); panel(ax, "B", "Cells called cancer-like")
for i, g in enumerate(ORDER):
    dd = d[d.group == g]
    r = dd.positive.mean(); lo, hi = boot_region(dd, lambda x: x.positive.mean())
    ax.bar(i, 100 * r, width=0.62, color=COL[g], alpha=0.9, zorder=3, edgecolor="white", lw=0.8)
    ax.errorbar(i, 100 * r, yerr=[[100 * (r - lo)], [100 * (hi - r)]], color=INK, lw=0.7, capsize=2, zorder=4)
    ax.text(i, 100 * hi + 3, f"{100 * r:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold", color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(["Normal", "Suspi-\ncious", "Cancer"], fontsize=6.8)
ax.set_ylim(0, 108); ax.set_ylabel("Cells above threshold (%)"); ygrid(ax)

# C: novelty
ax = fig.add_subplot(gs[0, 2]); panel(ax, "C", "Distance from the training data")
raincloud(ax, d, "novelty", log=True, bw=0.25)
ax.axhline(q95, ls=(0, (3, 2)), color=MUTED, lw=0.7, zorder=1)
ax.set_ylabel("Novelty, Mahalanobis distance"); ygrid(ax)
ytop = d.novelty.max()
bracket(ax, 0, 1, ytop * 1.6, fmt_p(pn["normal vs smoke"]["p_holm"]), log=True)
bracket(ax, 1, 2, ytop * 1.6, fmt_p(pn["smoke vs cancer"]["p_holm"]), log=True)
ax.set_ylim(d.novelty.min() * 0.6, ytop * 9)

# D: agreement of the five fold models on suspicious cells
ax = fig.add_subplot(gs[1, 0]); panel(ax, "D", "Agreement of the five fold models")
cnt = ens.n_models_positive.value_counts().reindex(range(6), fill_value=0)
shade = ["#d9dde3", "#f1dcb5", "#e8c48a", "#dcaa5d", "#cf9132", "#c9851a"]
ax.bar(range(6), 100 * cnt / cnt.sum(), color=[COL["normal"]] + ["#ecd3a6", "#e0b673", "#d49c45", "#c9851a"] + [COL["cancer"]],
       width=0.7, edgecolor="white", lw=0.8, zorder=3)
for k, v in cnt.items():
    extra = {0: "\nall\nnormal-like", 5: "\nall\ncancer-like"}.get(k, "")
    ax.text(k, 100 * v / cnt.sum() + 1.5, f"{100 * v / cnt.sum():.0f}%{extra}", ha="center", va="bottom",
            fontsize=6.0, linespacing=1.05)
ax.set_xticks(range(6)); ax.set_xlabel("Fold models calling the cell cancer-like")
ax.set_ylabel("Suspicious-group cells (%)"); ax.set_ylim(0, 66); ygrid(ax)

# E: each fold model separately
ax = fig.add_subplot(gs[1, 1:]); panel(ax, "E", "Every fold model, scored separately")
dev = d[d.group != "smoke"]
pos = np.arange(5)
off = {"normal": -0.22, "smoke": 0.0, "cancer": 0.22}
for g in ORDER:
    for f in range(5):
        if g == "smoke":
            dd = every[every.fold_model == f]
        else:
            dd = dev[(dev.group == g) & (dev.fold_model == f)]
        m = dd.p_cancer.mean(); lo, hi = boot_region(dd, lambda x: x.p_cancer.mean(), n=1000)
        ax.errorbar(f + off[g], m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4.2, color=COL[g], mec="white", mew=0.5,
                    elinewidth=0.9, capsize=0, zorder=3)
ax.set_xticks(pos); ax.set_xticklabels([f"Model {f + 1}" for f in pos])
ax.set_ylim(0, 1.0); ax.set_ylabel("Mean cancer probability"); ygrid(ax)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=COL[g], ms=4.5, label=LABEL[g]) for g in ORDER],
          frameon=False, ncol=3, loc="center", bbox_to_anchor=(0.55, 0.69), handletextpad=0.2, columnspacing=1.4)
fig.savefig(OUT / "fig_screening_suspicious.png", dpi=600)
fig.savefig(OUT / "fig_screening_suspicious.pdf")
plt.close(fig)

# ================================================================ supplementary: format-matched control
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), gridspec_kw={"wspace": 0.55, "width_ratios": [1.1, 1]})
fig.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.2)
ax = axes[0]; panel(ax, "A", "Cancer probability, same file format")
dj = jr.copy()
for i, g in enumerate(ORDER):
    if g == "smoke":
        dd = d[d.group == "smoke"]; m = dd.p_cancer.mean(); lo, hi = boot_region(dd, lambda x: x.p_cancer.mean())
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="s", ms=5, color=COL[g], mec="white", mew=0.5, elinewidth=0.9, zorder=3)
        continue
    dd = dj[dj.group == g]
    for k, (col, mk, fc) in enumerate((("p_cancer", "o", COL[g]), ("p_cancer_jpeg", "s", COL[g]))):
        m = dd[col].mean(); lo, hi = boot_region(dd, lambda x, c=col: x[c].mean())
        ax.errorbar(i - 0.13 + 0.26 * k, m, yerr=[[m - lo], [hi - m]], fmt=mk, ms=5 if mk == "s" else 4.5,
                    mfc=fc if k == 1 else "white", color=COL[g], mew=0.9, elinewidth=0.9, zorder=3)
    ax.plot([i - 0.13, i + 0.13], [dd.p_cancer.mean(), dd.p_cancer_jpeg.mean()], color=COL[g], lw=0.7, zorder=2)
ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[g] for g in ORDER]); ax.set_ylim(0, 1); ax.set_ylabel("Mean cancer probability")
ygrid(ax)
ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="original (lossless)"),
                   Line2D([], [], marker="s", ls="", color=INK, label="JPEG, quality 95")],
          frameon=False, loc="upper left", fontsize=6)
ax = axes[1]; panel(ax, "B", "Novelty, same file format")
for i, g in enumerate(ORDER):
    if g == "smoke":
        v = d[d.group == "smoke"].novelty; ax.scatter(i, v.median(), marker="s", s=26, color=COL[g], zorder=3)
        ax.plot([i, i], v.quantile([.25, .75]), color=COL[g], lw=0.9); continue
    dd = dj[dj.group == g]
    for k, col in enumerate(("novelty", "novelty_jpeg")):
        v = dd[col]; x = i - 0.13 + 0.26 * k
        ax.plot([x, x], v.quantile([.25, .75]), color=COL[g], lw=0.9)
        ax.scatter(x, v.median(), marker="o" if k == 0 else "s", s=22, facecolor="white" if k == 0 else COL[g],
                   edgecolor=COL[g], lw=0.9, zorder=3)
ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels([LABEL[g] for g in ORDER])
ax.set_ylabel("Novelty (median, IQR)"); ygrid(ax)
ax.set_yticks([5, 10, 20, 50]); ax.set_yticklabels(["5", "10", "20", "50"]); ax.minorticks_off()
fig.savefig(OUT / "figS_format_matched_control.png", dpi=600); fig.savefig(OUT / "figS_format_matched_control.pdf")
plt.close(fig)

# ================================================================ supplementary: region-bootstrap forest plot
B = pd.read_csv(SRC / "cluster_bootstrap" / "delta_auc_cluster_bootstrap.csv")
NICE = {"final: +3,000 all": "+ all 3,000 generated", "final: real only": "real only", "final: +QC-clean": "+ QC-clean",
        "final: +215 QC-selected": "+ 215 QC-selected", "final: +random, QC-clean count": "+ random (same n)",
        "XA: +3,000 all": "+ all 3,000 generated", "XA: real only": "real only", "XA: +215 QC-selected": "+ 215 QC-selected",
        "XA: +215 random": "+ 215 random", "NextViT-S: real only": "NextViT-S", "CoAtNet-0: real only": "CoAtNet-0",
        "ResNet-50: real only": "ResNet-50"}
rows = [("AFiS-Net-XA", "XA: +3,000 all", "XA: real only"), ("AFiS-Net-XA", "XA: +215 QC-selected", "XA: real only"),
        ("AFiS-Net-XA", "XA: +215 QC-selected", "XA: +215 random"), ("AFiS-Net-XA", "XA: +3,000 all", "XA: +215 QC-selected"),
        ("AFiS-Net", "final: +215 QC-selected", "final: real only"), ("AFiS-Net", "final: +3,000 all", "final: real only"),
        ("AFiS-Net", "final: +QC-clean", "final: real only"), ("AFiS-Net", "final: +QC-clean", "final: +random, QC-clean count"),
        ("AFiS-Net vs reference", "final: real only", "NextViT-S: real only"), ("AFiS-Net vs reference", "final: real only", "CoAtNet-0: real only"),
        ("AFiS-Net vs reference", "final: real only", "ResNet-50: real only")]
fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
fig.subplots_adjust(left=0.36, right=0.98, top=0.88, bottom=0.14)
ylab = []
for j, (sub, ttl) in enumerate((("all", "All 608 images"), ("gan_unseen", "177 images the generator never saw"))):
    ax = axes[j]; ax.set_title(ttl, loc="left", fontsize=8)
    for k, (grp, a, b) in enumerate(rows):
        r = B[(B.subset == sub) & (B.A == a) & (B.B == b)]
        y = len(rows) - 1 - k
        if j == 0:
            ylab.append((y, f"{NICE[a]} vs {NICE[b]}" if grp != "AFiS-Net vs reference" else f"AFiS-Net vs {NICE[b]}"))
        if r.empty:
            continue
        r = r.iloc[0]; sig = r.ci_low > 0 or r.ci_high < 0
        c = INK if sig else "#9a9a9a"
        ax.plot([r.ci_low, r.ci_high], [y, y], color=c, lw=1.1, solid_capstyle="round")
        ax.scatter(r.d_auc, y, s=18, color=c if sig else "white", edgecolor=c, lw=0.9, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    for ysep in (len(rows) - 4.5, len(rows) - 8.5):
        ax.axhline(ysep, color=GRID, lw=0.8)
    ax.set_xlabel("$\\Delta$AUC (95% CI)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_yticks([y for y, _ in ylab]); axes[0].set_yticklabels([t for _, t in ylab], fontsize=6.5)
axes[0].tick_params(axis="y", length=0)
for yc, txt in ((len(rows) - 1, "AFiS-Net-XA"), (len(rows) - 5, "AFiS-Net"), (len(rows) - 9, "Reference models")):
    fig.text(0.008, axes[0].transData.transform((0, yc))[1] / fig.dpi / fig.get_figheight(), txt,
             fontsize=6.8, fontweight="bold", va="center", ha="left", rotation=0)
fig.savefig(OUT / "figS_region_bootstrap_auc.png", dpi=600); fig.savefig(OUT / "figS_region_bootstrap_auc.pdf")
plt.close(fig)

# ================================================================ supplementary: generator fidelity vs unseen images
G = pd.read_csv(SRC / "gan_unseen" / "gan_metrics_vs_unseen.csv") if (SRC / "gan_unseen" / "gan_metrics_vs_unseen.csv").exists() else None
if G is not None:
    sets = ["Real training partition (region-disjoint)", "NTP all (unfiltered)", "NTP QC-clean (hygiene filter)", "NTP QC-selected (215)"]
    short = ["Real\ntraining", "All\ngenerated", "QC-\nclean", "QC-\nselected"]
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
    fig.savefig(OUT / "figS_generator_vs_unseen.png", dpi=600); fig.savefig(OUT / "figS_generator_vs_unseen.pdf")
    plt.close(fig)
print("figures written to", OUT)
