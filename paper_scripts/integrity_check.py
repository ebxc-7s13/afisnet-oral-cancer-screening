"""Integrity check for the manuscript. (1) Regenerates every table from the result files and confirms the inlined
tables in main.tex are identical. (2) Recomputes the key numbers quoted in the text and confirms each appears.
(3) Confirms banned strings are absent. Prints a Markdown report; exit code 1 on any problem.
Usage: python integrity_check.py FINAL_DIR RES7 RES9 RECORD_TESTS"""
import json
import pathlib
import subprocess
import sys
import tempfile

D, R7, R9, RB = sys.argv[1:5]
D = pathlib.Path(D)
tex = (D / "main.tex").read_text()
flat = " ".join(tex.split())
rows, bad = [], 0


def check(item, expected, where="text"):
    global bad
    ok = " ".join(expected.split()) in flat
    bad += not ok
    rows.append((item, expected, "OK" if ok else "MISSING", where))


# (1) tables regenerated from data
tmp = pathlib.Path(tempfile.mkdtemp())
subprocess.run([sys.executable, str(pathlib.Path(__file__).parent / "make_tables.py"), R7, R9, RB, str(tmp)], check=True,
               capture_output=True)
for name in ("classifiers", "ablation", "synthetic", "gensets", "attribution"):
    body = (tmp / f"tab_{name}.tex").read_text().strip()
    for line in body.splitlines():
        check(f"Table {name}", line, f"table {name}")
N = json.load(open(tmp / "numbers.json"))

# (2) key text numbers
f = N["T4_final"]
check("Final real-only AUC", f"AUC ($0.955\\pm0.031$; macro F1 $0.876$, MCC $0.761$, cancer sensitivity $0.901$")
assert (f["auc_raw"], f["sd_raw"], f["f1_raw"], f["mcc_raw"], f["sen_raw"]) == (0.955, 0.031, 0.876, 0.761, 0.901), f
fc = N["final_cmp"]
d = [round(fc[f"final vs baselines (real only)|final AFiSNet_ConcatAvg|{b}"]["d"], 3) for b in ("NextViT_Small", "CoAtNet0", "ResNet50")]
check("Final vs baselines", f"by ${d[0]:+.3f}$, ${d[1]:+.3f}$, and ${d[2]:+.3f}$ AUC (Holm-adjusted $p\\le0.001$)")
assert max(fc[f"final vs baselines (real only)|final AFiSNet_ConcatAvg|{b}"]["p_holm"] for b in ("NextViT_Small", "CoAtNet0", "ResNet50")) <= 0.001
sb = N["syn_baselines"]
check("Baseline synthetic gains", f"improved NextViT-S (${sb['NextViT_Small']['d']:+.3f}$ AUC) and ResNet-50 (${sb['ResNet50']['d']:+.3f}$)")
check("Ablation p range", f"Holm-adjusted $p\\ge{N['abl_p_range'][0]:.2f}$")
check("CNN-only ECE", f"calibration error {N['abl_AFiSNet_CNNonly']['ece']:.3f} versus {N['abl_AFiSNet']['ece']:.3f}")
fx = [fc[f"final vs original AFiS-Net|AFiSNet_ConcatAvg {k}|AFiSNet {k}"] for k in ("real only", "+215 QC-selected")]
check("Final vs XA (3 seeds)", f"with real images (${fx[0]['d']:+.3f}$) or QC-selected images (${fx[1]['d']:+.3f}$; three seeds, Holm-adjusted $p={min(x['p_holm'] for x in fx):.2f}$)")
import glob as _g
from scipy.stats import wilcoxon as _w
import pandas as _pd
_r = _pd.read_csv(f"{R7}/stats/all_runs.csv")
_xa = _r[(_r.model == "AFiSNet") & (_r.experiment == "Raw_AllUnfiltered") & (_r.seed == 42)].sort_values("fold").roc_auc_macro.values
_fi = _pd.Series({int(f.split("fold_")[1].split("/")[0]): json.load(open(f))["roc_auc_macro"] for f in _g.glob(f"{R9}/runs/E9_final/AFiSNet_ConcatAvg/Raw_AllUnfiltered/seed_42/fold_*/record.json")}).sort_index().values
assert len(_xa) == len(_fi) == 5
check("Final vs XA all-3000 seed 42", f"slightly lower with all 3,000 images on a single seed (${_fi.mean() - _xa.mean():+.3f}$; unadjusted $p={_w(_fi, _xa).pvalue:.2f}$, the smallest attainable with five pairs)")
q = N["xa_qc_vs_rand"]
check("XA random vs QC", f"(AUC ${N['T6_+ random, same size']['xa']:.3f}$ versus ${N['T6_+ QC-selected (ranked)']['xa']:.3f}$, Holm-adjusted $p={q['p']:.2f}$;")
a = N["xa_all_vs_real"]
check("XA all vs real", f"raised the AUC from ${N['T6_Real only']['xa']:.3f}$ to ${N['T6_+ all generated']['xa']:.3f}$ (Holm-adjusted $p={a['p']:.3f}$")
check("XA all F1/ECE", f"raised the macro F1-score by ${a['df1']:.3f}$, and reduced the calibration error by {100*(1-a['ece_all']/a['ece_raw']):.0f}\\% (ECE ${a['ece_raw']:.3f}$ to ${a['ece_all']:.3f}$)".replace("$0.05$", "$0.050$"))
check("Fold-aware", f"($\\Delta$AUC ${N['E6']['d']:+.3f}$, $p={N['E6']['p']:.2f}$)")
check("QC at scale", f"(AUC ${N['qcscale']['qc80']:.3f}$ versus ${N['qcscale']['rm']:.3f}$)")
fin = {k: N[k]["fin"] for k in N if k.startswith("T6_") and N[k]["fin"]}
check("Final QC-clean F1/MCC", f"macro F1-score (${fin['T6_+ QC-clean (hygiene filter)']['f1']:.3f}$) and MCC (${fin['T6_+ QC-clean (hygiene filter)']['mcc']:.3f}$)")
x = fc["final: QC-clean vs all|+QC-clean|+3,000 all"]
assert x["ci"] == "[-0.0004, +0.0124]"
check("QC-clean vs all", f"($\\Delta$AUC ${x['d']:+.3f}$, 95\\% CI $-0.0004$ to $+0.012$)")
check("DeLong QC-clean", f"(DeLong $p={fc['final: synthetic effect|+QC-clean (hygiene filter)|real only']['delong']:.2f}$ for QC-clean;")
e5 = N["E5_abs"]
check("Scaling 25%", f"from ${e5['raw25']:.3f}$ to ${e5['syn25']:.3f}$ (Holm-adjusted $p={N['E5']['025']['p']:.3f}$) and lowered the ECE from ${e5['ece_raw25']:.3f}$ to ${e5['ece_syn25']:.3f}$")
check("Scaling 50%", f"with 50\\% the AUC gain was ${N['E5']['050']['d']:+.3f}$ (Holm-adjusted $p={N['E5']['050']['p']:.2f}$)")
qc = N["qc_clean"]
check("QC-clean kept", f"retained {qc['kept_n'] + qc['kept_c']:,} of the 3,000 images ({qc['kept_n']:,} normal, {qc['kept_c']:,} cancer)")
r = qc["reasons"]
check("QC-clean reasons", f"removing {r['technical defect']} technical defects, {r['unrealistic (unlike any real cell)']} unrealistic images, and {r['class-ambiguous']} extremely class-ambiguous images")
t = {(x["cls"], x["set"]): x for x in N["texture"]}
mae = lambda c, k: f"{t[(c, k)]['spectral_MAE_log10']:.3f}"
check("Spectral MAE", f"Spectral MAE was {mae('normal', 'QC-selected')} and {mae('cancer', 'QC-selected')} (normal, cancer) for the QC-selected images, {mae('normal', 'All generated')} and {mae('cancer', 'All generated')} for the complete pool, and {mae('normal', 'QC-clean (hygiene)')} and {mae('cancer', 'QC-clean (hygiene)')} for the QC-clean set")
cor = [x["LBP_correlation"] for x in N["texture"]]; it = [x["LBP_intersection"] for x in N["texture"]]
check("LBP ranges", f"(correlation {min(cor):.3f}--{max(cor):.3f}, intersection {min(it):.3f}--{max(it):.3f})")
s = N["screen_final"]
check("Screening means", f"on average {s['normal']:.2f} for normal cells, {s['smoke']:.2f} for the susceptible group, and {s['cancer']:.2f} for cancer cells")
check("Screening all-3000", "({normal:.2f}, {smoke:.2f}, {cancer:.2f})".format(**N["screen_all"]))
check("Screening XA", "({normal:.2f}, {smoke:.2f}, {cancer:.2f}; both".format(**N["screen_xa"]))
st = N["screen_stats"]
assert st["image"]["jonckheere_normal<smoke<cancer"]["p_one_sided"] < 5.1e-5 and st["region"]["jonckheere_normal<smoke<cancer"]["p_one_sided"] < 5.1e-5
assert max(v["p_holm"] for v in st["image"]["pairwise_mannwhitney"].values()) < 1e-4
nv = N["novelty"]
check("Novelty comparison medians", f"(median {nv['normal']['median']:.1f} and {nv['cancer']['median']:.1f} versus {nv['reference']['median']:.1f}; none above the reference 95th percentile)")
check("Novelty smokers", f"(median {nv['smoke']['median']:.1f}, interquartile range {nv['smoke']['iqr'][0]:.0f}--{nv['smoke']['iqr'][1]:.0f}; Kruskal--Wallis $p={nv['kruskal_p']:.1e}$".replace("e-04", "\\times10^{-4}"))
assert max(nv["smoke>normal_p"], nv["smoke>cancer_p"]) * 3 < 1e-3 and nv["smoke"]["p_greater_than_reference"] < 1e-3
check("Screening KW", f"Kruskal--Wallis $p={st['image']['kruskal_p']:.0e}$".replace("e-08", "\\times10^{-8}"))
check("Regions", "within " + ", ".join(str(st['region']['n'][g]) for g in ("normal", "smoke")) + f", and {st['region']['n']['cancer']} imaging regions")
assert round(nv["smoke"]["frac_above_ref_q95"] * 18) == 7
jc = N["jpeg"]
check("JPEG control", f"(median ratio {jc['novelty_ratio_median']:.2f}, $p={jc['wilcoxon_p_novelty']:.2f}$")
# screening disagreement, JPEG effects, unseen AUC
_s = _pd.read_csv(f"{R9}/screening_final/per_image_screening_v8.csv"); _sm = _s[_s.group == "smoke"]
check("Smoker disagreement", f"(median between-model s.d.\\ {_sm.p_cancer_sd.median():.2f}; above 0.3 for {(_sm.p_cancer_sd > 0.3).sum()} of 18)")
check("Threshold median", f"validation-selected threshold (median {_s.threshold.median():.2f})")
from sklearn.metrics import roc_auc_score as _auc
_j = _pd.read_csv(f"{R9}/screening_final/jpeg_control_per_image.csv"); _y = (_j.cls == "cancer").astype(int)
_fl = ((_j.p_cancer_png >= .5) != (_j.p_cancer_jpeg >= .5)); _cn = ((_j.p_cancer_png >= .5) & (_j.p_cancer_jpeg < .5))
_q = N["novelty"]["reference"]["q95"]
check("JPEG effects", f"({_fl.sum()} of 608 predictions changed at 0.5, {_cn.sum()} of them from cancer to normal; out-of-fold AUC {_auc(_y, _j.p_cancer_png):.3f} to {_auc(_y, _j.p_cancer_jpeg):.3f})")
check("JPEG q95 counts", f"{(_j.novelty_jpeg > _q).sum()} versus {(_j.novelty_png > _q).sum()} images above the 95th percentile")
_v = _pd.read_csv(str(D / "supp/data_tables/paired_tests_v2_auc_f1at05_unseen_delong.csv"))
_x = _v[(_v.A == "+all 3,000 unfiltered") & (_v.B == "real only")].iloc[0]
check("Unseen AUC/F1", f"(AUC ${_x.d_auc_unseen:+.3f}$, $p={_x.p_auc_unseen:.2f}$; macro F1 at threshold 0.5 ${_x.d_f1_05_unseen:+.3f}$, $p={_x.p_f1_05_unseen:.2f}$)")
_x = _v[(_v.A == "+215 QC-selected") & (_v.B == "+215 random")].iloc[0]
assert _x.d_auc_all < 0
check("QC vs random recomputed", f"random selection was slightly better, Holm-adjusted $p={_x.p_auc_all_holm:.2f}$")
for sx in ("608 single-cell images", "125 participants", "3,000 images (1,500 per class)", "real training partition (431 images)",
           "215 images (123 normal, 92 cancer)", "18 susceptible images", "single seed"):
    check("Count", sx)
for sx in ("616", "353 normal", "RTX 4000", "this revision", "initial analysis", "Kaggle", "notebook", "seed-42 folds, whereas"):
    ok = sx not in tex
    bad += not ok
    rows.append(("Must be absent", sx, "OK" if ok else "PRESENT", "whole text"))

out = ["| Check | Expected text (recomputed from result files) | Status | Where |", "|---|---|---|---|"]
out += [f"| {a_} | `{b_}` | {c_} | {d_} |" for a_, b_, c_, d_ in rows]
print("\n".join(out))
print(f"\n{len(rows)} checks, {bad} problems")
sys.exit(1 if bad else 0)
