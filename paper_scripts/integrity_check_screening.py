"""Checks the screening of the suspicious group and the other session-11 numbers (generator against the 177 unseen
images, hold-out counts, crop statistics, Supplementary Table S5) against the revision-11 result files, plus
structural checks of the manuscript.
Usage: python integrity_check_screening.py MANUSCRIPT_DIR SUPPLEMENT_DIR REVISION11_DIR"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

M, SUP, R = (Path(a) for a in sys.argv[1:4])
tex = (M / "main.tex").read_text()
sup = (SUP / "supplementary.tex").read_text()
ok, bad = 0, []


def check(name, cond):
    global ok
    if cond:
        ok += 1
    else:
        bad.append(name)


def has(name, s, where=tex):
    check(name + f" [{s[:60]}]", s in where)


p = pd.read_csv(R / "screening_all" / "per_image_primary.csv")
S = json.loads((R / "screening_all" / "screening_stats.json").read_text())
ens = pd.read_csv(R / "screening_all" / "smokers_5model_mean.csv")
reps = pd.read_csv(R / "screening_all" / "per_fold_model_replicates.csv")
jr = pd.read_csv(R / "screening_all" / "development_images_jpeg_rescored.csv")
G = pd.read_csv(R / "gan_unseen" / "gan_metrics_vs_unseen.csv")
F = json.loads((R / "facts" / "facts.json").read_text())
grp = {g: p[p.group == g] for g in ("normal", "smoke", "cancer")}

# ---- counts
has("n suspicious", "294 images of the suspicious group (127 imaging regions)")
check("n suspicious data", len(grp["smoke"]) == 294 and grp["smoke"].region.nunique() == 127)
check("regions normal/cancer", grp["normal"].region.nunique() == 117 and grp["cancer"].region.nunique() == 156)
has("regions text", "averaging within 117, 127, and 156 imaging regions")
# ---- means and CIs
for g, s in (("normal", "0.12 for normal cells"), ("smoke", "0.36 for the suspicious group"), ("cancer", "0.89 for cancer cells")):
    has("mean " + g, s); check("mean value " + g, f"{grp[g].p_cancer.mean():.2f}" == s.split()[0])


def boot(d, f, n=2000, seed=11):
    regs = d.region.unique(); idx = {r: np.where(d.region.values == r)[0] for r in regs}
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        v.append(f(d.iloc[np.concatenate([idx[r] for r in rng.choice(regs, len(regs))])]))
    return np.percentile(v, [2.5, 97.5])


ci = {g: boot(grp[g], lambda x: x.p_cancer.mean()) for g in grp}
has("CI normal", f"(95\\% CI {ci['normal'][0]:.2f}--{ci['normal'][1]:.2f})")
has("CI susp", f"({ci['smoke'][0]:.2f}--{ci['smoke'][1]:.2f})")
has("CI cancer", f"({ci['cancer'][0]:.2f}--{ci['cancer'][1]:.2f})")
rc = {g: boot(grp[g], lambda x: x.positive.mean()) for g in grp}
has("rate CI susp", f"(95\\% CI {100 * rc['smoke'][0]:.0f}--{100 * rc['smoke'][1]:.0f}\\%)")
has("rate CI normal", f"5\\% of normal cells ({100 * rc['normal'][0]:.0f}--{100 * rc['normal'][1]:.0f}\\%)")
has("rate CI cancer", f"92\\% of cancer cells ({100 * rc['cancer'][0]:.0f}--{100 * rc['cancer'][1]:.0f}\\%)")
check("rates", [round(100 * grp[g].positive.mean()) for g in grp] == [5, 32, 92])
# ---- p-values
pi = S["primary_probability"]
check("JT image < 1e-80", pi["image"]["jonckheere_normal<smoke<cancer"]["p_one_sided"] < 1e-80)
check("JT region < 1e-50", pi["region"]["jonckheere_normal<smoke<cancer"]["p_one_sided"] < 1e-50)
check("pairwise < 1e-10", max(v["p_holm"] for v in pi["image"]["pairwise"].values()) < 1e-10)
# ---- bimodality
s = grp["smoke"].p_cancer
check("bimodal 185/94", (s < 0.1).sum() == 185 and (s > 0.9).sum() == 94)
has("bimodal text", "185 of its 294 cells (63\\%) scored below 0.1")
# ---- agreement
c = ens.n_models_positive.value_counts()
check("agreement 52/14/34", (round(100 * c.get(0, 0) / 294), round(100 * c.get(5, 0) / 294)) == (52, 14)
      and round(100 * (294 - c.get(0, 0) - c.get(5, 0)) / 294) == 34)
check("median sd 0.02", f"{ens.p_cancer_sd.median():.2f}" == "0.02")
has("per-model ranges", f"mean probability {reps.p_mean_normal.min():.2f}--{reps.p_mean_normal.max():.2f} for normal, "
    f"{reps.p_mean_smoke.min():.2f}--{reps.p_mean_smoke.max():.2f} for suspicious, and "
    f"{reps.p_mean_cancer.min():.2f}--{reps.p_mean_cancer.max():.2f} for cancer cells")
check("per-model JT < 1e-20", reps.p_jt.max() < 1e-20 and reps.smoke_between.all())
check("5-model mean same", f"{ens.p_cancer.mean():.2f}" == f"{s.mean():.2f}")
# ---- novelty
n = {g: grp[g].novelty.median() for g in grp}
has("novelty medians", f"median novelty ({n['smoke']:.1f}) was modestly higher than that of normal ({n['normal']:.1f}) and cancer cells ({n['cancer']:.1f}")
pn = S["primary_novelty"]["image"]["pairwise"]
check("novelty p", pn["normal vs smoke"]["p_holm"] < 1e-7 and pn["smoke vs cancer"]["p_holm"] < 1e-11)
has("cliff", f"Cliff's $\\delta={pn['normal vs smoke']['cliffs_delta_second_minus_first']:.2f}$")
q95 = S["design"]["novelty_reference_q95_608"]
fr = {g: round(100 * (grp[g].novelty > q95).mean()) for g in grp}
has("q95 shares", f"10\\% of them exceeded the 95th percentile of the development images, against {fr['normal']}\\% of normal and {fr['cancer']}\\% of cancer cells")
check("q95 susp 10", fr["smoke"] == 10)
dev = p[p.group != "smoke"]
has("novelty median 608", f"median {dev.novelty.median():.1f} for the 608 development images")
# ---- JPEG
mj = jr.groupby("group").p_cancer_jpeg.mean()
has("jpeg means", f"mean probability {mj['normal']:.2f}, 0.36, and {mj['cancer']:.2f}")
check("jpeg JT", S["sensitivity_jpeg_matched_probability"]["image"]["jonckheere_normal<smoke<cancer"]["p_one_sided"] < 1e-60)
pj = S["sensitivity_jpeg_matched_novelty"]["image"]["pairwise"]
check("jpeg novelty", pj["normal vs smoke"]["p_holm"] < 1e-4 and pj["smoke vs cancer"]["p_holm"] < 1e-4)
# ---- generator vs unseen
g = G.set_index(["set", "cls"])
a = "NTP all (unfiltered)"
has("KID unseen", f"KID of {g.loc[(a, 'normal'), 'KID_mean']:.3f} (normal) and {g.loc[(a, 'cancer'), 'KID_mean']:.3f} (cancer)")
has("prec/rec unseen", f"precision {g.loc[(a, 'normal'), 'precision_matched']:.2f} and {g.loc[(a, 'cancer'), 'precision_matched']:.2f} and recall "
    f"{g.loc[(a, 'normal'), 'recall_matched']:.2f} and {g.loc[(a, 'cancer'), 'recall_matched']:.2f}")
check("real KID ~0", abs(g.loc[("Real training partition (region-disjoint)", "normal"), "KID_mean"]) < 0.002)
# ---- facts
h = F["generator_holdout"]
has("holdout", f"({h['normal']['held_out_for_checkpoint_selection']} normal and {h['cancer']['held_out_for_checkpoint_selection']} cancer images)")
check("fit 366", h["normal"]["used_for_gradient_updates"] + h["cancer"]["used_for_gradient_updates"] == 366)
cr = F["foreground_crop"]["development images"]
check("crop 607", cr["cropped"] == 607)
has("aspect", f"width-to-height ratio {cr['aspect_ratio_p5_p95'][0]:.2f}--{cr['aspect_ratio_p5_p95'][1]:.2f}")
# ---- supplement Table S5
for row in ("348 (117) & 0.12", "294 (127) & 0.36", "260 (156) & 0.89", "& 4.9\\% & 11.6 & 6.0\\%", "& 31.6\\% & 17.2 & 9.9\\%",
            "& 91.9\\% & 9.2 & 3.8\\%", "& 45.2 & 18.0\\%", "0.09 & -- & 11.6 & 3.4\\%", "0.78 & -- & 11.2 & 6.2\\%"):
    has("Table S5", row, sup)
# ---- structure
body = tex[tex.index("\\section{Introduction}"):]
methods = tex[tex.index("\\subsection{Participants"):tex.index("\\subsection{Dataset and cross-validation}")]
other = tex.replace(methods, "")
check("no 'susceptible'", not re.search(r"(?i)susceptib", tex + sup))
check("'smoker' only in Methods 2.1", not re.search(r"(?i)smoker", other) and not re.search(r"(?i)smok", sup))
ab = tex[tex.index("begin{abstract}") + 15:tex.index("end{abstract}")]
check("abstract <= 250 words", len(re.sub(r"\$[^$]*\$", "X", ab).split()) <= 250)
hl = M.parent / "02_submission_items" / "Highlights.txt"
for line in (hl.read_text().splitlines() if hl.exists() else []):
    check("highlight <= 85 chars", len(line) <= 85)
if (M / "main.log").exists():
    check("no undefined refs", "undefined" not in (M / "main.log").read_text(errors="ignore").lower())
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
    check("figure exists " + f, (M / f).exists())
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", sup):
    check("supp figure exists " + f, (SUP / f).exists())
print(f"{ok} checks passed, {len(bad)} problems")
for b in bad:
    print("  PROBLEM:", b)
