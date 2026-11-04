"""Figures 6 and 7 (ROC/confusion matrices; synthetic quantity vs curation) for the final AFiS-Net configuration (concatenation + average pooling).
Usage: python fig_final.py RES9 OUT"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
R, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 600})
BLUE, ORANGE, GREEN, GREY, PURPLE, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#6b46c1", "#c53030"
FM = "AFiSNet_ConcatAvg"
ARMS = {"Raw": ("E9_final", "Raw"), "QC215": ("E9_select", "Raw_SelectedSynthetic"), "All": ("E9_final", "Raw_AllUnfiltered"),
        "Clean": ("E9_final", "Raw_QCclean"), "RandClean": ("E9_final", "Raw_RandomMatchQCclean")}

def pred_files(key, seeds=None):
    fam, exp = ARMS[key]
    fs = sorted(glob.glob(str(R / "runs" / fam / FM / exp / "seed_*" / "fold_*" / "predictions.csv")))
    return [f for f in fs if seeds is None or int(f.split("seed_")[1].split("/")[0]) in seeds]

def load(f):
    d = pd.read_csv(f); d["y"] = (d.GroundTruth == "cancer").astype(int); d["yhat"] = (d.Prediction == "cancer").astype(int); return d

def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)

# ---- Fig. 5
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.45})
grid = np.linspace(0, 1, 201)
for key, seeds, col, lab in (("Raw", None, GREY, "Real only (15 folds)"), ("QC215", None, BLUE, "+215 QC-selected (15)"),
                             ("All", [42], ORANGE, "+3,000 all (5, seed 42)"), ("Clean", [42], GREEN, "+2,764 QC-clean (5, seed 42)")):
    tprs, aucs = [], []
    for f in pred_files(key, seeds):
        d = load(f); fpr, tpr, _ = roc_curve(d.y, d.Probability_Cancer)
        tprs.append(np.interp(grid, fpr, tpr)); aucs.append(roc_auc_score(d.y, d.Probability_Cancer))
    mt = np.mean(tprs, 0); mt[0] = 0
    axes[0].plot(grid, mt, color=col, lw=1.4, label=lab)
axes[0].plot([0, 1], [0, 1], ls=":", c="k", lw=0.7)
axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate")
axes[0].legend(loc="lower right", bbox_to_anchor=(1.02, 0.02), frameon=False, fontsize=5.8, handlelength=1.2)
axes[0].set_title("A  ROC, mean over test folds", loc="left")
for ax, key, seeds, lab in ((axes[1], "Raw", [42, 100, 123], "B  Real only"), (axes[2], "Clean", [42], "C  + QC-clean (seed 42)")):
    cms = []
    for s in seeds:
        d = pd.concat([load(f) for f in pred_files(key, [s])]); cms.append(confusion_matrix(d.y, d.yhat, labels=[0, 1]))
    cm = np.mean(cms, 0)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(1).max())
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:.1f}" if len(seeds) > 1 else f"{v:.0f}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Normal", "Cancer"]); ax.set_yticklabels(["Normal", "Cancer"])
    ax.set_xlabel("Predicted"); ax.set_title(lab, loc="left")
    if key == "Raw": ax.set_ylabel("True")
    for sp in ax.spines.values(): sp.set_visible(False)
save(fig, "fig_cv_roc_confusion")

# ---- Fig. 6: quantity vs curation, AFiS-Net-XA dose curve + final-model points (seed 42)
M = pd.read_csv(R / "stats_v2" / "per_run_metrics_v2.csv")
def arm(f, m, e, seeds=None):
    d = M[(M.family == f) & (M.model == m) & (M.experiment == e)]
    return d if seeds is None else d[d.seed.isin(seeds)]
pts = [(0, arm("E1_kfold", "AFiSNet", "Raw", [42])), (215, arm("E3_synthetic_source", "AFiSNet", "Raw_RandomUnfiltered", [42]))] + \
      [(n, arm("E3_dose", "AFiSNet", f"Raw_Random{n:04d}")) for n in (500, 1000, 2000)] + \
      [(3000, arm("E3_synthetic_source", "AFiSNet", "Raw_AllUnfiltered", [42]))]
qc = arm("E1_kfold", "AFiSNet", "Raw_SelectedSynthetic", [42]); qa = arm("E3_qc_scale", "AFiSNet", "Raw_QCfilteredAll")
rm = arm("E3_qc_scale", "AFiSNet", "Raw_RandomMatchedQCAll")
fin = [(arm("E9_final", FM, "Raw", [42]), "o", 25, None), (arm("E9_select", FM, "Raw_SelectedSynthetic", [42]), "s", 45, "AFiS-Net: 215 QC-selected"),
       (arm("E9_final", FM, "Raw_QCclean", [42]), "P", -60, "AFiS-Net: QC-clean (hygiene filter)"),
       (arm("E9_final", FM, "Raw_RandomMatchQCclean", [42]), "^", 60, "AFiS-Net: random, same number"),
       (arm("E9_final", FM, "Raw_AllUnfiltered", [42]), "o", 25, "AFiS-Net: real only / all 3,000")]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
for ax, met, lab, tag in zip(axes, ("auc_all", "f1_05_all"), ("AUC", "Macro F1 (threshold 0.5)"), ("A", "B")):
    xs = np.array([float(d.n_synthetic.median()) for _, d in pts]); ms = np.array([d[met].mean() for _, d in pts])
    se = np.array([d[met].std(ddof=1) / np.sqrt(len(d)) for _, d in pts])
    ax.errorbar(xs, ms, yerr=se, color=ORANGE, marker="o", ms=4, lw=1.4, capsize=2.5, label="AFiS-Net-XA: random images")
    for d, col, mk, l, dx in ((qc, BLUE, "s", "AFiS-Net-XA: 215 QC-selected", -45),
                               (qa, GREEN, "D", "AFiS-Net-XA: QC top-80%", -45), (rm, PURPLE, "^", "AFiS-Net-XA: random, same number", 45)):
        ax.errorbar([float(d.n_synthetic.median()) + dx], [d[met].mean()], yerr=[d[met].std(ddof=1) / np.sqrt(len(d))],
                    color=col, marker=mk, ms=5, ls="none", capsize=2.5, label=l)
    for d, mk, dx, l in fin:
        ax.errorbar([float(d.n_synthetic.median()) + dx], [d[met].mean()], yerr=[d[met].std(ddof=1) / np.sqrt(len(d))],
                    color=RED, marker=mk, ms=5, mfc="white", mew=1.2, ls="none", capsize=2.5, label=l)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000]); ax.set_xlim(-120, 3200)
    ax.set_xlabel("Synthetic images added to training"); ax.set_ylabel(lab); ax.set_title(f"{tag}  {lab}", loc="left")
axes[1].legend(frameon=False, loc="lower right", fontsize=5.6, ncol=1)
save(fig, "fig_quantity_vs_curation")
for d, mk, _, _ in fin: print(mk, d.n_synthetic.median(), round(d.auc_all.mean(), 4), round(d.f1_05_all.mean(), 4), len(d))
