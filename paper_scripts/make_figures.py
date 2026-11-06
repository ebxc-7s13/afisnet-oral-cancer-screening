"""Result figures of the AFiS-Net manuscript (Figs 6-8, Supplementary Fig. S2), rebuilt from the Kaggle result files.
Usage: python make_figures.py RESULTS_DIR OUT_DIR"""
import sys, os, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

R = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 600})
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#6b46c1"
M = pd.read_csv(R / "stats_v2" / "per_run_metrics_v2.csv")


def arm(f, m, e, seeds=None):
    d = M[(M.family == f) & (M.model == m) & (M.experiment == e)]
    return d if seeds is None else d[d.seed.isin(seeds)]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)


def preds(f, m, e, s):
    out = []
    for p in sorted(glob.glob(str(R / "runs" / f / m / e / f"seed_{s}" / "fold_*" / "predictions.csv"))):
        d = pd.read_csv(p); d["y"] = (d.GroundTruth == "cancer").astype(int); d["yhat"] = (d.Prediction == "cancer").astype(int)
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ---- Fig. 6: pooled out-of-fold ROC and confusion matrices, AFiS-Net, 3 seeds
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.2, 1, 1], "wspace": 0.45})
grid = np.linspace(0, 1, 201)
for exp, col, lab in (("Raw", GREY, "Real only"), ("Raw_SelectedSynthetic", BLUE, "Real + 215 QC-selected")):
    tprs, aucs = [], []
    for f in sorted(glob.glob(str(R / "runs" / "E1_kfold" / "AFiSNet" / exp / "seed_*" / "fold_*" / "predictions.csv"))):
        d = pd.read_csv(f); y = (d.GroundTruth == "cancer").astype(int)
        fpr, tpr, _ = roc_curve(y, d.Probability_Cancer); tprs.append(np.interp(grid, fpr, tpr)); aucs.append(roc_auc_score(y, d.Probability_Cancer))
    mt, sd = np.mean(tprs, 0), np.std(tprs, 0); mt[0] = 0
    axes[0].plot(grid, mt, color=col, lw=1.6, label=f"{lab}\nAUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    axes[0].fill_between(grid, np.clip(mt - sd, 0, 1), np.clip(mt + sd, 0, 1), color=col, alpha=0.15, lw=0)
axes[0].plot([0, 1], [0, 1], ls=":", c="k", lw=0.7)
axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate"); axes[0].legend(loc="lower right", frameon=False, fontsize=6.3, handlelength=1.2)
axes[0].set_title("A  ROC, mean of 15 test folds", loc="left")
for ax, exp, lab in ((axes[1], "Raw", "B  Real only"), (axes[2], "Raw_SelectedSynthetic", "C  Real + QC-selected")):
    cms = [confusion_matrix(preds("E1_kfold", "AFiSNet", exp, s).y, preds("E1_kfold", "AFiSNet", exp, s).yhat, labels=[0, 1]) for s in (42, 100, 123)]
    cm = np.mean(cms, 0)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(1).max())
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Normal", "Cancer"]); ax.set_yticklabels(["Normal", "Cancer"])
    ax.set_xlabel("Predicted"); ax.set_title(lab, loc="left")
    if exp == "Raw":
        ax.set_ylabel("True")
    for sp in ax.spines.values():
        sp.set_visible(False)
save(fig, "fig_cv_roc_confusion")

# ---- Fig. 7: quantity vs curation (seed 42, same five folds)
pts = [(0, arm("E1_kfold", "AFiSNet", "Raw", [42])), (215, arm("E3_synthetic_source", "AFiSNet", "Raw_RandomUnfiltered", [42]))] + \
      [(n, arm("E3_dose", "AFiSNet", f"Raw_Random{n:04d}")) for n in (500, 1000, 2000)] + \
      [(3000, arm("E3_synthetic_source", "AFiSNet", "Raw_AllUnfiltered", [42]))]
qc = arm("E1_kfold", "AFiSNet", "Raw_SelectedSynthetic", [42]); qa = arm("E3_qc_scale", "AFiSNet", "Raw_QCfilteredAll")
rm = arm("E3_qc_scale", "AFiSNet", "Raw_RandomMatchedQCAll")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, met, lab, tag in zip(axes, ("auc_all", "f1_05_all"), ("AUC", "Macro F1 (threshold 0.5)"), ("A", "B")):
    xs = np.array([float(d.n_synthetic.median()) for _, d in pts]); ms = np.array([d[met].mean() for _, d in pts])
    se = np.array([d[met].std(ddof=1) / np.sqrt(len(d)) for _, d in pts])
    ax.errorbar(xs, ms, yerr=se, color=ORANGE, marker="o", ms=4, lw=1.4, capsize=2.5, label="Random NTP images")
    for d, col, mk, l, dx in ((qc, BLUE, "s", "215 QC-selected (original pipeline)", -45),
                               (qa, GREEN, "D", "QC top-80% of all images", -45), (rm, PURPLE, "^", "Random, same number", 45)):
        ax.errorbar([float(d.n_synthetic.median()) + dx], [d[met].mean()], yerr=[d[met].std(ddof=1) / np.sqrt(len(d))],
                    color=col, marker=mk, ms=5, ls="none", capsize=2.5, label=l)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000]); ax.set_xlim(-120, 3150)
    ax.set_xlabel("Synthetic images added to training"); ax.set_ylabel(lab)
    ax.set_title(f"{tag}  {lab}", loc="left")
axes[1].legend(frameon=False, loc="lower right")
save(fig, "fig_quantity_vs_curation")

# ---- Fig. 8: data scaling (3 seeds)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
for ax, met, lab, tag in zip(axes, ("auc_all", "f1_05_all"), ("AUC", "Macro F1 (threshold 0.5)"), ("A", "B")):
    for base, col, l, dx in (("Raw", GREY, "Real only", -1.2), ("Raw_SelectedSynthetic", BLUE, "Real + 215 QC-selected", 1.2)):
        xs, ms, se = [], [], []
        for frac in (0.25, 0.5, 0.75, 1.0):
            d = arm("E1_kfold", "AFiSNet", base) if frac == 1.0 else arm("E5_scaling", "AFiSNet", f"{base}_frac{int(frac * 100):03d}")
            xs.append(frac * 100 + dx); ms.append(d[met].mean()); se.append(d[met].std(ddof=1) / np.sqrt(len(d)))
        ax.errorbar(xs, ms, yerr=se, color=col, marker="o", ms=4, lw=1.4, capsize=2.5, label=l)
    ax.set_xticks([25, 50, 75, 100]); ax.set_xlabel("Real training regions used (%)"); ax.set_ylabel(lab)
    ax.set_title(f"{tag}  {lab}", loc="left")
axes[0].legend(frameon=False, loc="lower right")
save(fig, "fig_data_scaling")

# ---- Fig. 10 (screening of the suspicious group) and Supplementary Figs S1, S4: see make_revision_figures.py

# ---- Supplementary Fig. S2: FID at matched sample size
g = pd.read_csv(R / "gan_metrics_v2" / "gan_metrics_matched_n.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
lab = {"NTP QC-selected (paper)": "QC-selected\n(215)", "NTP all (unfiltered)": "All\n(3,000)", "NTP QC top-80% of all": "QC top-80%\n(2,124)"}
for ax, c, tag in zip(axes, ("normal", "cancer"), ("A  Normal", "B  Cancer")):
    d = g[g.cls == c].reset_index(drop=True)
    ax.bar(range(len(d)), d.FID_matched_mean, yerr=[d.FID_matched_mean - d.FID_matched_low, d.FID_matched_high - d.FID_matched_mean],
           color=[BLUE, ORANGE, GREEN][:len(d)], alpha=0.8, capsize=3)
    ax.set_xticks(range(len(d))); ax.set_xticklabels([lab[s] for s in d.set], fontsize=7.5)
    ax.set_ylabel(f"FID at n = {int(d.matched_n.iloc[0])} (lower is better)"); ax.set_title(tag, loc="left")
save(fig, "figS_fid_matched")
print("figures written to", OUT)
