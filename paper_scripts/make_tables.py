"""Builds every results table of the manuscript directly from the Kaggle result files, so no number is retyped.
Writes LaTeX tabular bodies to OUT/tab_*.tex and the key text numbers to OUT/numbers.json.
Usage: python make_tables.py RES7 RES9 RECORD_TESTS OUT"""
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd

R7, R9, RB, OUT = sys.argv[1:5]
OUT = pathlib.Path(OUT)
OUT.mkdir(parents=True, exist_ok=True)
runs = pd.read_csv(f"{R7}/stats/all_runs.csv")
rb = pd.read_csv(RB)
N = {}


def arm(model, exp, seeds=None):
    d = runs[(runs.model == model) & (runs.experiment == exp)]
    d = d[d.seed.isin(seeds)] if seeds else d
    return pd.DataFrame(dict(auc=d.roc_auc_macro, f1=d.f1_macro, mcc=d.mcc, sen=d.Cancer_recall_sensitivity,
                             ece=d.ece, brier=d.brier_score, seed=d.seed))


def fin(fam, exp, seeds=None):
    rows = []
    for f in sorted(glob.glob(f"{R9}/runs/{fam}/AFiSNet_ConcatAvg/{exp}/seed_*/fold_*/record.json")):
        r = json.load(open(f))
        sd = int(f.split("seed_")[1].split("/")[0])
        if seeds is None or sd in seeds:
            rows.append(dict(seed=sd, auc=r["roc_auc_macro"], f1=r["f1_macro"], mcc=r["mcc"],
                             sen=r["Cancer_recall_sensitivity"], ece=r["ece"], brier=r["brier_score"]))
    return pd.DataFrame(rows)


def ms(d, k):
    return f"${d[k].mean():.3f}\\pm{d[k].std():.3f}$"


def m(d, k):
    return f"{d[k].mean():.3f}"


def rbp(A, B):
    x = rb[(rb.A == A) & (rb.B == B)]
    assert len(x) == 1, (A, B)
    return x.iloc[0]


sig = lambda p: "$^{\\dagger}$" if p < 0.05 else ""

# ---------------- Table: classifier comparison (real only vs +215 QC-selected), three seeds
models = [("final", "AFiS-Net (final)", None), ("AFiSNet", "AFiS-Net-XA", "AFiSNet"),
          ("NextViT_Small", "NextViT-S (transformer)", "NextViT_Small"), ("CoAtNet0", "CoAtNet-0 (hybrid)", "CoAtNet0"),
          ("ResNet50", "ResNet-50 (CNN)", "ResNet50")]
lines, supp = [], []
for key, lab, rbk in models:
    if key == "final":
        a, b = fin("E9_final", "Raw"), fin("E9_select", "Raw_SelectedSynthetic")
        p = json.load(open(f"{R9}/final_model.json"))  # existence check
        cf = pd.read_csv(f"{R9}/stats_final/comparisons_final_AFiSNet_ConcatAvg.csv")
        psyn = cf[(cf.group == "final: synthetic effect") & (cf.A == "+215 QC-selected")].iloc[0].p_auc_holm
    else:
        a, b = arm(key, "Raw"), arm(key, "Raw_SelectedSynthetic")
        psyn = rbp(f"{rbk} +QC", f"{rbk} real").p_auc_holm
    assert len(a) == 15 and len(b) == 15, (key, len(a), len(b))
    bold = key == "final"
    L = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
    lines.append(f"{L(lab)} & {ms(a, 'auc')} & {ms(b, 'auc')}{sig(psyn)} & {m(a, 'f1')} & {m(b, 'f1')} & {m(a, 'ece')} & {m(b, 'ece')} \\\\")
    for tr, d in (("Real only", a), ("+215 QC-selected", b)):
        supp.append(f"{lab} & {tr} & {ms(d, 'auc')} & {ms(d, 'f1')} & {ms(d, 'mcc')} & {ms(d, 'sen')} & {m(d, 'ece')} & {m(d, 'brier')} \\\\")
    N[f"T4_{key}"] = dict(auc_raw=round(a.auc.mean(), 3), sd_raw=round(a.auc.std(), 3), auc_syn=round(b.auc.mean(), 3),
                          f1_raw=round(a.f1.mean(), 3), mcc_raw=round(a.mcc.mean(), 3), sen_raw=round(a.sen.mean(), 3),
                          ece_raw=round(a.ece.mean(), 3), ece_syn=round(b.ece.mean(), 3), p_syn=float(psyn))
(OUT / "tab_classifiers.tex").write_text("\n".join(lines))
(OUT / "tabS_classifiers_full.tex").write_text("\n".join(supp))

# ---------------- Table: ablation (QC-selected augmentation, three seeds); delta vs AFiS-Net-XA
full = arm("AFiSNet", "Raw_SelectedSynthetic")
abl = [("AFiSNet", "AFiS-Net-XA (cross-attention + GeM)", "reference"),
       ("AFiSNet_Concat", "Concatenation + GeM", "fusion"),
       ("AFiSNet_AvgPool", "Cross-attention + average pooling", "pooling"),
       ("final", "Concatenation + avg. pooling (AFiS-Net)", "fusion and pooling"),
       ("AFiSNet_SwinOnly", "Transformer branch only", "CNN branch removed"),
       ("AFiSNet_CNNonly", "Convolutional branch only", "transformer branch removed"),
       ("AFiSNet_noSupCon", "Without contrastive loss", "loss term removed")]
lines, pmax, pmin = [], 0, 1
for key, lab, chg in abl:
    d = fin("E9_select", "Raw_SelectedSynthetic") if key == "final" else arm(key, "Raw_SelectedSynthetic")
    assert len(d) == 15, key
    if key == "AFiSNet":
        delta = "--"
    else:
        delta = f"${d.auc.mean() - full.auc.mean():+.3f}$"
        if key != "final":
            p = rbp("full", key).p_auc_holm
            pmin, pmax = min(pmin, p), max(pmax, p)
    bold = key == "final"
    L = (lambda s: f"\\textbf{{{s}}}") if bold else (lambda s: s)
    lines.append(f"{L(lab)} & {ms(d, 'auc')} & {m(d, 'f1')} & {m(d, 'ece')} & {delta} \\\\")
    N[f"abl_{key}"] = dict(auc=round(d.auc.mean(), 3), f1=round(d.f1.mean(), 3), ece=round(d.ece.mean(), 3),
                           d=round(d.auc.mean() - full.auc.mean(), 3))
cf = pd.read_csv(f"{R9}/stats_final/comparisons_final_AFiSNet_ConcatAvg.csv")
pfin = cf[(cf.group == "final vs original AFiS-Net") & (cf.A == "AFiSNet_ConcatAvg +215 QC-selected")].iloc[0].p_auc_holm
N["abl_p_range"] = [round(pmin, 2), round(max(pmax, pfin), 2)]
(OUT / "tab_ablation.tex").write_text("\n".join(lines))

# ---------------- Table: synthetic-data composition
xa = {e: arm("AFiSNet", e) for e in ("Raw", "Raw_SelectedSynthetic", "Raw_RandomUnfiltered", "Raw_AllUnfiltered")}
for e, d in xa.items():
    assert len(d) == 15, e
f42 = {e: fin(fam, e, [42]) for e, fam in (("Raw", "E9_final"), ("Raw_SelectedSynthetic", "E9_select"),
                                          ("Raw_AllUnfiltered", "E9_final"), ("Raw_QCclean", "E9_final"),
                                          ("Raw_RandomMatchQCclean", "E9_final"))}
for e, d in f42.items():
    assert len(d) == 5, e
pxa = {"Raw_SelectedSynthetic": rbp("AFiSNet +QC", "AFiSNet real").p_auc_holm,
       "Raw_RandomUnfiltered": rbp("rand215", "real").p_auc_holm,
       "Raw_AllUnfiltered": rbp("all", "real").p_auc_holm}
rows = [("Real only", "0", "Raw", "Raw"), ("+ QC-selected (ranked)", "215", "Raw_SelectedSynthetic", "Raw_SelectedSynthetic"),
        ("+ random, same size", "215", "Raw_RandomUnfiltered", None), ("+ all generated", "3,000", "Raw_AllUnfiltered", "Raw_AllUnfiltered"),
        ("+ QC-clean (hygiene filter)", "2,764", None, "Raw_QCclean"), ("+ random, QC-clean size", "2,764", None, "Raw_RandomMatchQCclean")]
lines = []
for lab, n, ex, ef in rows:
    c1 = (ms(xa[ex], "auc") + (sig(pxa[ex]) if ex in pxa else "")) if ex else "--"
    if ef:
        d = f42[ef]
        c2, c3, c4 = ms(d, "auc"), m(d, "f1"), m(d, "ece")
    else:
        c2 = c3 = c4 = "--"
    lines.append(f"{lab} & {n} & {c1} & {c2} & {c3} & {c4} \\\\")
    N[f"T6_{lab}"] = dict(xa=None if not ex else round(xa[ex].auc.mean(), 3),
                          fin=None if not ef else dict(auc=round(f42[ef].auc.mean(), 3), f1=round(f42[ef].f1.mean(), 3),
                                                       mcc=round(f42[ef].mcc.mean(), 3), ece=round(f42[ef].ece.mean(), 3)))
(OUT / "tab_synthetic.tex").write_text("\n".join(lines))
N["xa_all_vs_real"] = dict(d=round(rbp("all", "real").d_roc_auc_macro, 3), p=round(rbp("all", "real").p_auc_holm, 3),
                           df1=round(rbp("all", "real").d_f1_macro, 3), pf1=round(rbp("all", "real").p_f1_holm, 3),
                           ece_raw=round(xa["Raw"].ece.mean(), 3), ece_all=round(xa["Raw_AllUnfiltered"].ece.mean(), 3),
                           p_ece=round(rbp("all", "real").p_ece, 3))
N["xa_all_vs_qc"] = dict(d=round(rbp("all", "QC").d_roc_auc_macro, 3), p=round(rbp("all", "QC").p_auc_holm, 3))
N["xa_qc_vs_rand"] = dict(d=round(rbp("QC", "rand215").d_roc_auc_macro, 3), p=round(rbp("QC", "rand215").p_auc_holm, 2))
N["qcscale"] = dict(qc80=round(arm("AFiSNet", "Raw_QCfilteredAll", [42]).auc.mean(), 3),
                    rm=round(arm("AFiSNet", "Raw_RandomMatchedQCAll", [42]).auc.mean(), 3), p=round(rbp("qc80", "rm").p_auc_holm, 2))
N["dose"] = {k: round(arm("AFiSNet", e, [42]).auc.mean(), 3) for k, e in
             (("0", "Raw"), ("500", "Raw_Random0500"), ("1000", "Raw_Random1000"), ("2000", "Raw_Random2000"), ("3000", "Raw_AllUnfiltered"))}
N["E5"] = {k: dict(d=round(rbp(k, "real").d_roc_auc_macro, 3), p=round(rbp(k, "real").p_auc_holm, 3),
                   dece=round(rbp(k, "real").d_ece, 3), pece=round(rbp(k, "real").p_ece, 3)) for k in ("025", "050", "075")}
N["E5_abs"] = dict(raw25=round(arm("AFiSNet", "Raw_frac025").auc.mean(), 3), sd_raw25=round(arm("AFiSNet", "Raw_frac025").auc.std(), 3),
                   syn25=round(arm("AFiSNet", "Raw_SelectedSynthetic_frac025").auc.mean(), 3),
                   sd_syn25=round(arm("AFiSNet", "Raw_SelectedSynthetic_frac025").auc.std(), 3),
                   ece_raw25=round(arm("AFiSNet", "Raw_frac025").ece.mean(), 3), ece_syn25=round(arm("AFiSNet", "Raw_SelectedSynthetic_frac025").ece.mean(), 3))
N["E6"] = dict(d=round(rbp("fa", "QC").d_roc_auc_macro, 3), p=round(rbp("fa", "QC").p_auc_holm, 2))
N["final_cmp"] = {f"{r.group}|{r.A}|{r.B}": dict(d=float(r.d_auc), p_holm=float(r.p_auc_holm),
                                                   delong=None if pd.isna(r.p_delong_median) else round(r.p_delong_median, 3),
                                                   ci=r.ci95_auc) for r in cf.itertuples()}
N["base_cmp_xa"] = {b: dict(d=round(rbp("AFiS", b).d_roc_auc_macro, 3), p=round(rbp("AFiS", b).p_auc_holm, 4)) for b in ("NextViT_Small", "CoAtNet0", "ResNet50")}
N["syn_baselines"] = {k: dict(d=round(rbp(f"{k} +QC", f"{k} real").d_roc_auc_macro, 3), p=round(rbp(f"{k} +QC", f"{k} real").p_auc_holm, 4),
                              dece=round(rbp(f"{k} +QC", f"{k} real").d_ece, 3)) for k in ("NextViT_Small", "ResNet50", "CoAtNet0")}

# ---------------- Table: generated sets at matched n
g = pd.read_csv(f"{R9}/gan_metrics_v2/gan_metrics_matched_n.csv")
names = [("NTP QC-selected (paper)", "QC-selected (ranked)"), ("NTP QC-clean (hygiene filter)", "QC-clean (hygiene filter)"),
         ("NTP all (unfiltered)", "All generated"), ("NTP QC top-80% of all", "QC top-ranked")]
lines, supp = [], []
for key, lab in names:
    cells, n = [], []
    for c in ("normal", "cancer"):
        r = g[(g.cls == c) & (g.set == key)].iloc[0]
        n.append(f"{int(r.n_gen_total):,}")
        cells += [f"{r.FID_matched_mean:.1f}", f"{r.precision_matched:.2f}", f"{r.recall_matched:.2f}", f"{r.coverage_matched:.2f}"]
        supp.append(f"{c.capitalize()} & {lab} ({int(r.n_gen_total):,}) & {r.FID_matched_mean:.1f} [{r.FID_matched_low:.1f}, {r.FID_matched_high:.1f}] & "
                    f"${r.KID_mean:.3f}\\pm{r.KID_sd:.3f}$ & {r.precision_matched:.2f} & {r.recall_matched:.2f} & {r.density_matched:.2f} & {r.coverage_matched:.2f} \\\\")
    if "top-80" not in key:
        lines.append(f"{lab} & {n[0]} / {n[1]} & " + " & ".join(cells) + " \\\\")
(OUT / "tab_gensets.tex").write_text("\n".join(lines))
supp.sort(key=lambda s: s.startswith("Cancer"))
(OUT / "tabS_gensets_full.tex").write_text("\n".join(supp))
N["matched_n"] = {c: int(g[g.cls == c].matched_n.iloc[0]) for c in ("normal", "cancer")}

# ---------------- Table: attribution evaluation (final model, QC-clean)
e = pd.read_csv(f"{R9}/explainability_final/summary_explainability_v2.csv")
e = e[e.arm == "Raw+Synthetic"]
lines = []
for c in ("cancer", "normal"):
    for meth, lab in (("Occlusion", "Occlusion"), ("Multi-level Grad-CAM++", "Grad-CAM++")):
        r = e[(e.cls == c) & (e.method == meth)].iloc[0]
        f = lambda s: s.split(" ±")[0]
        lines.append(f"{c.capitalize()} & {lab} & {r.n_valid}/{r.n} & {f(r.attention_ratio)} & {r.cell_area_fraction:.2f} & {r.pointing_game:.2f} & "
                     f"{f(r.deletion_auc_cam)} & {f(r.deletion_auc_random)} & {f(r.insertion_auc_cam)} & {f(r.insertion_auc_random)} \\\\")
(OUT / "tab_attribution.tex").write_text("\n".join(lines))

# ---------------- text numbers
q = json.load(open(f"{R9}/qc_clean/summary.json"))
rj = q["rejected_by_reason"]
N["qc_clean"] = dict(kept_n=q["n_kept"]["normal"], kept_c=q["n_kept"]["cancer"],
                     reasons={k: sum(rj[c].get(k, 0) for c in rj) for k in set(k for c in rj for k in rj[c])})
t = pd.read_csv(f"{R9}/texture_v2/texture_metrics.csv")
N["texture"] = t.round(3).to_dict(orient="records")
for nm, d in (("screen_final", "screening_final"), ("screen_all", "screening_final_all3000"), ("screen_xa", "screening_v8")):
    gs = pd.read_csv(f"{R9}/{d}/group_summary_leakfree.csv").set_index("group")
    N[nm] = {k: round(gs.loc[k, "p_cancer_mean"], 2) for k in ("normal", "smoke", "cancer")}
N["screen_stats"] = json.load(open(f"{R9}/screening_final/stats_leakfree.json"))
N["novelty"] = json.load(open(f"{R9}/screening_final/novelty_stats.json"))
N["jpeg"] = json.load(open(f"{R9}/screening_final/jpeg_control_summary.json"))
(OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=str))
print("tables written to", OUT)
