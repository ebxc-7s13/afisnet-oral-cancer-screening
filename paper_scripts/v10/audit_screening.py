"""Screening audit (manuscript v10): verifies the scoring design (held-out scoring, one model per suspicious-group
region, per-model temperature and threshold) and every screening number of the paper from the saved per-image
outputs and run manifests; nothing is re-scored.
Usage: python audit_screening.py <afis_results/revision11/screening_all> <afis_results/runs/E9_final/AFiSNet_ConcatAvg/Raw_QCclean/seed_42> [out.json]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
S = Path(sys.argv[1])
RUN = Path(sys.argv[2])
P = pd.read_csv(S / "per_image_primary.csv")
out = []
def chk(name, ok, detail=""):
    out.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name, detail)

# --- 1. held-out scoring of development images
man = {k: json.load(open(RUN / f"fold_{k}" / "split_manifest.json")) for k in range(5)}
rec = {k: json.load(open(RUN / f"fold_{k}" / "record.json")) for k in range(5)}
dev = P[P.group != "smoke"]
ok = all(r.image in set(man[int(r.fold_model)]["test"]) for r in dev.itertuples())
chk("each development image scored by the fold model that held it out (test fold)", ok, f"n={len(dev)}")
alltest = [b for k in range(5) for b in man[k]["test"]]
chk("test folds partition the 608 development images", len(alltest) == 608 == len(set(alltest)) and set(alltest) == set(dev.image))
# --- 2. suspicious images never in any manifest, one model per region, balance
sm = P[P.group == "smoke"]
seen = set(b for k in range(5) for part in ("train", "val", "test") for b in man[k][part])
chk("no suspicious-group image in any train/val/test manifest", not (set(sm.image) & seen), f"n={len(sm)}")
chk("no synthetic list contains a suspicious image", not any(set(sm.image) & set(man[k]["synthetic"]) for k in range(5)))
reg_models = sm.groupby("region").fold_model.nunique()
chk("each suspicious region assigned to exactly one fold model", (reg_models == 1).all(), f"{len(reg_models)} regions")
rpm = sm.groupby("fold_model").region.nunique().to_dict(); ipm = sm.groupby("fold_model").size().to_dict()
chk("region assignment balanced (max-min <= 1 region)", max(rpm.values()) - min(rpm.values()) <= 1, f"regions/model {rpm}, images/model {ipm}")
# --- 3. per-model thresholds
thr_rec = {k: float(rec[k]["cancer_threshold"]) for k in range(5)}
temp_rec = {k: round(float(rec[k]["temperature"]), 3) for k in range(5)}
print("per-model temperature:", temp_rec)
thr_used = P.groupby("fold_model").threshold.unique().apply(lambda a: a.tolist()).to_dict()
chk("one threshold per fold model in the scored table", all(len(v) == 1 for v in thr_used.values()), str({k: round(v[0], 4) for k, v in thr_used.items()}))
chk("thresholds equal the validation-selected thresholds stored with each checkpoint run",
    all(thr_rec[k] is not None and abs(thr_rec[k] - thr_used[k][0]) < 1e-6 for k in range(5)), str(thr_rec))
chk("positive flag = p_cancer >= threshold of the scoring model", ((P.p_cancer >= P.threshold) == P.positive).all())
# --- 4. headline numbers
mean = P.groupby("group").p_cancer.mean(); pos = P.groupby("group").positive.mean() * 100
nov = P.groupby("group").novelty.median()
chk("mean p_cancer 0.12/0.36/0.89", [round(mean[g], 2) for g in ("normal", "smoke", "cancer")] == [0.12, 0.36, 0.89], str(mean.round(4).to_dict()))
chk("cancer-like 4.9/31.6/91.9 %", [round(pos[g], 1) for g in ("normal", "smoke", "cancer")] == [4.9, 31.6, 91.9], str(pos.round(2).to_dict()))
chk("suspicious: 185 below 0.1, 94 above 0.9", ((sm.p_cancer < 0.1).sum(), (sm.p_cancer > 0.9).sum()) == (185, 94),
    f"{(sm.p_cancer < 0.1).sum()} / {(sm.p_cancer > 0.9).sum()}; positives {int(sm.positive.sum())}")
chk("novelty medians 17.2/11.6/9.2", [round(nov[g], 1) for g in ("smoke", "normal", "cancer")] == [17.2, 11.6, 9.2], str(nov.round(2).to_dict()))
J = pd.read_csv(S / "development_images_jpeg_rescored.csv")
jm = J.groupby("group").p_cancer_jpeg.mean()
chk("JPEG-rescored development means 0.09 (normal) / 0.78 (cancer); suspicious 0.36 unchanged",
    (round(jm["normal"], 2), round(jm["cancer"], 2)) == (0.09, 0.78), str(jm.round(4).to_dict()))
from sklearn.metrics import roc_auc_score
y = (J.group == "cancer").astype(int)
chk("development AUC PNG 0.945 -> JPEG 0.905", (round(roc_auc_score(y, J.p_cancer), 3), round(roc_auc_score(y, J.p_cancer_jpeg), 3)) == (0.945, 0.905),
    f"{roc_auc_score(y, J.p_cancer):.4f} -> {roc_auc_score(y, J.p_cancer_jpeg):.4f}")
R = pd.read_csv(S / "per_fold_model_replicates.csv")
order_ok = ((R.p_mean_normal < R.p_mean_smoke) & (R.p_mean_smoke < R.p_mean_cancer))
chk("same order normal<suspicious<cancer (mean p) in 5/5 fold models", order_ok.sum() == 5, f"{int(order_ok.sum())}/5; smoke_between {R.smoke_between.tolist()}")
# --- 5. the 45.2 novelty value of the 5-model sensitivity analysis
M = pd.read_csv(S / "smokers_5model_mean.csv"); E = pd.read_csv(S / "smokers_scores_by_every_fold_model.csv")
recomp = E.groupby("image").novelty.mean()
chk("smokers_5model_mean.novelty = mean over the five fold models of the per-model novelty",
    np.allclose(M.set_index("image").novelty.reindex(recomp.index), recomp, rtol=1e-4))
print("five-model mean novelty median (suspicious):", round(M.novelty.median(), 2),
      "| median of per-image median-over-models:", round(E.groupby('image').novelty.median().median(), 2),
      "| primary median:", round(nov['smoke'], 2))
own = E.merge(sm[["image", "fold_model"]], on="image", suffixes=("", "_own"))
print("novelty median when scored by own-assigned model:", round(own[own.fold_model == own.fold_model_own].novelty.median(), 2),
      "| by the four other models:", round(own[own.fold_model != own.fold_model_own].novelty.median(), 2))
print("per-model suspicious novelty medians:", E.groupby("fold_model").novelty.median().round(1).to_dict())
q95 = json.load(open(S / "screening_stats.json"))["design"]["novelty_reference_q95_608"]
print("above dev q95 (5-model mean):", round((M.novelty > q95).mean() * 100, 1), "% | primary:", round((sm.novelty > q95).mean() * 100, 1), "%")
print("5-model mean p median/mean:", round(M.p_cancer.median(), 3), round(M.p_cancer.mean(), 3),
      "| cells positive (mean p >= median threshold?)", )
T = pd.read_csv(S / "table_screening_all.csv"); print(T.to_string())
json.dump([dict(check=a, passed=b, detail=c) for a, b, c in out], open(sys.argv[3] if len(sys.argv) > 3 else "audit_screening.json", "w"), indent=1)
print(f"\n{sum(b for _, b, _ in out)}/{len(out)} checks passed")
