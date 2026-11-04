"""Seed-42 and three-seed summaries for the final AFiS-Net configuration (concatenation + average pooling)."""
import json, glob, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, "nb")
R9, R7 = "res9", "res7/afis_results"

def recs(root, fam, model, exp):
    rows = []
    for f in sorted(glob.glob(f"{root}/runs/{fam}/{model}/{exp}/seed_*/fold_*/record.json")):
        r = json.load(open(f)); s = int(f.split("seed_")[1].split("/")[0]); k = int(f.split("fold_")[1].split("/")[0])
        rows.append(dict(seed=s, fold=k, auc=r["roc_auc_macro"], f1=r["f1_macro"], mcc=r["mcc"],
                         sen=r["Cancer_recall_sensitivity"], ece=r["ece"], brier=r["brier_score"], path=f.rsplit("/", 1)[0]))
    return pd.DataFrame(rows)

def preds(df, seed):
    d = df[df.seed == seed]
    if len(d) < 5: return None
    P = pd.concat([pd.read_csv(p + "/predictions.csv") for p in d.path])
    return P

arms = {
 "final Raw": recs(R9, "E9_final", "AFiSNet_ConcatAvg", "Raw"),
 "final +215 QC": recs(R9, "E9_select", "AFiSNet_ConcatAvg", "Raw_SelectedSynthetic"),
 "final +3000": recs(R9, "E9_final", "AFiSNet_ConcatAvg", "Raw_AllUnfiltered"),
 "final +QC-clean": recs(R9, "E9_final", "AFiSNet_ConcatAvg", "Raw_QCclean"),
 "final +random(QC-clean n)": recs(R9, "E9_final", "AFiSNet_ConcatAvg", "Raw_RandomMatchQCclean"),
 "orig Raw": recs(R7, "E1_kfold", "AFiSNet", "Raw"),
 "orig +215 QC": recs(R7, "E1_kfold", "AFiSNet", "Raw_SelectedSynthetic"),
 "orig +3000": recs(R7, "E3_synthetic_source", "AFiSNet", "Raw_AllUnfiltered"),
}
print(arms["final Raw"].head(2))
pd.set_option("display.width", 220)
S = []
for k, d in arms.items():
    for lab, dd in (("3 seeds", d), ("seed 42", d[d.seed == 42])):
        if len(dd): S.append(dict(arm=k, set=lab, n=len(dd), **{m: f"{dd[m].mean():.3f}±{dd[m].std():.3f}" for m in ("auc", "f1", "mcc", "sen")},
                                  ece=round(dd.ece.mean(), 3), brier=round(dd.brier.mean(), 3)))
print(pd.DataFrame(S).to_string(index=False))

def delong_cols(P1, P2):
    from afis_analysis import _delong
    lab = [c for c in P1.columns if c.lower() in ("y", "true", "label", "true_label", "y_true")]
    return lab

P = preds(arms["final Raw"], 42); print(P.columns.tolist()[:12])

from afis_analysis import _delong
def pooled(df, seed=42):
    P = preds(df, seed)
    P = P.drop_duplicates("Filename")
    return P.assign(y=(P.GroundTruth.astype(str).str.lower() == "cancer").astype(int))[["Filename", "y", "Probability_Cancer"]]

def test(a, b, seed=42):
    A_, B_ = arms[a], arms[b]
    j = A_[A_.seed == seed].merge(B_[B_.seed == seed], on=["seed", "fold"], suffixes=("_a", "_b"))
    pa, pb = pooled(A_, seed), pooled(B_, seed)
    m = pa.merge(pb, on="Filename")
    auc1, auc2, p = _delong(m.y_x.values, m.Probability_Cancer_x.values, m.Probability_Cancer_y.values)[:3]
    out = dict(A=a, B=b, n=len(j), dAUC_runs=j.auc_a.mean() - j.auc_b.mean(), p_wil=wilcoxon(j.auc_a - j.auc_b).pvalue,
               dF1=j.f1_a.mean() - j.f1_b.mean(), dMCC=j.mcc_a.mean() - j.mcc_b.mean(), dECE=j.ece_a.mean() - j.ece_b.mean(),
               pooledAUC_a=auc1, pooledAUC_b=auc2, p_delong=p, n_img=len(m))
    return out
T = [test("final +215 QC", "final Raw"), test("final +3000", "final Raw"), test("final +QC-clean", "final Raw"),
     test("final +random(QC-clean n)", "final Raw"), test("final +QC-clean", "final +3000"),
     test("final +QC-clean", "final +random(QC-clean n)"), test("final +3000", "final +215 QC"), test("final +QC-clean", "final +215 QC"),
     test("final +3000", "orig +3000"), test("final +QC-clean", "orig +3000"), test("final Raw", "orig Raw")]
print(pd.DataFrame(T).round(4).to_string(index=False))
