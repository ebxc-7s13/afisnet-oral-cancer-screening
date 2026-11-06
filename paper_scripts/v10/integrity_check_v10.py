"""Integrity check for the manuscript (v10; v9 checks plus the v10 corrections). (1) Regenerates every table from the result files and confirms the inlined
tables in main.tex are identical. (2) Recomputes the key numbers quoted in the text and confirms each appears.
(3) Confirms banned strings are absent. Prints a Markdown report; exit code 1 on any problem.
The screening numbers (suspicious group, session 11) are checked by integrity_check_screening.py.
Usage: python integrity_check_v10.py FINAL_DIR RES7 RES9 RECORD_TESTS COMMIT12   (FINAL_DIR holds main.tex and supp/data_tables)"""
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
subprocess.run([sys.executable, str(next(q for q in (pathlib.Path(__file__).parent / "make_tables.py", pathlib.Path(__file__).parent.parent / "make_tables.py") if q.exists())), R7, R9, RB, str(tmp)], check=True,
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
check("Spectral MAE", f"The spectral MAE (normal, cancer) was {mae('normal', 'QC-selected')} and {mae('cancer', 'QC-selected')} for the QC-selected images, {mae('normal', 'All generated')} and {mae('cancer', 'All generated')} for the complete pool, and {mae('normal', 'QC-clean (hygiene)')} and {mae('cancer', 'QC-clean (hygiene)')} for the QC-clean set")
cor = [x["LBP_correlation"] for x in N["texture"]]; it = [x["LBP_intersection"] for x in N["texture"]]
check("LBP ranges", f"(correlation {min(cor):.3f}--{max(cor):.3f}, intersection {min(it):.3f}--{max(it):.3f})")
_v = _pd.read_csv(str(D / "supp/data_tables/paired_tests_v2_auc_f1at05_unseen_delong.csv"))
_x = _v[(_v.A == "+all 3,000 unfiltered") & (_v.B == "real only")].iloc[0]
check("Unseen AUC/F1", f"(AUC ${_x.d_auc_unseen:+.3f}$, $p={_x.p_auc_unseen:.2f}$; macro F1 at threshold 0.5 ${_x.d_f1_05_unseen:+.3f}$, $p={_x.p_f1_05_unseen:.2f}$)")
_x = _v[(_v.A == "+215 QC-selected") & (_v.B == "+215 random")].iloc[0]
assert _x.d_auc_all < 0
check("QC vs random recomputed", f"random selection was slightly better, Holm-adjusted $p={_x.p_auc_all_holm:.2f}$")
for sx in ("608 single-cell images", "125 participants", "3,000 images (1,500 per class)", "generator partition (431 images)",
           "215 images (123 normal, 92 cancer)", "294 images of the suspicious group", "single seed",
           # v10: generator data roles, QC equation, acquisition, cohort wording (all verified against code/outputs)
           "37 normal and 28 cancer images", "366 images (210 normal, 156 cancer)", "2,000 images (1,000 per class)",
           "at least 5 cigarettes per day for the preceding seven years", "387.5\\times387.5", "141.5",
           "$\\approx$7.0 kimg", "$\\approx$10.6 kimg", "1,200 highest-scoring normal images", "all 924 cancer images",
           "0.94 (normal) and 0.92 (cancer) among the QC-selected images and 0.94 and 0.93 in the complete pool",
           "from 0.89 to 0.78 for cancer cells and from 0.12 to 0.09 for normal cells", "commit \\texttt{" + sys.argv[5] + "}", "RTX 4000 Ada GPU (20\\,GB)", "oral squamous cell carcinoma (OSCC)", "same selected generator checkpoint", "kept constant throughout to avoid saturation. Only the three fluorescence channels were used"):
    check("Count", sx)
for sx in ("616", "353 normal", "RTX 6000", "dysplasia", "histopathology-positive", "two generations of the trained generator", "according to the protocol", "this revision", "initial analysis", "Kaggle", "notebook", "seed-42 folds, whereas", "susceptible", "18 suspicious",
           # v10: statements found inaccurate in the audit must not return
           "5--6 cigarettes", "malignancy axis", "malignancy score", "8.6 kimg", "12.9 kimg", "pass over the 431",
           "sequential excitation", "top 40\\% of the remainder", "nearest-neighbour distance (0.07)", "real training partition",
           "so file format does not explain these differences", "guards against synthetic images",
           "did not merely reproduce", "upstream screening and triage"):
    ok = sx not in tex
    bad += not ok
    rows.append(("Must be absent", sx, "OK" if ok else "PRESENT", "whole text"))

out = ["| Check | Expected text (recomputed from result files) | Status | Where |", "|---|---|---|---|"]
out += [f"| {a_} | `{b_}` | {c_} | {d_} |" for a_, b_, c_, d_ in rows]
print("\n".join(out))
print(f"\n{len(rows)} checks, {bad} problems")
sys.exit(1 if bad else 0)
