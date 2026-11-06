"""Independent checks of the numbers added or changed in manuscript v10 that the v9 integrity checks do not cover.
Usage: python audit_v10_extra.py <main.tex> <revision11 dir> <data_tables dir> <gan_training.py> <out.json>"""
import json, math, re, sys
from pathlib import Path
import pandas as pd

tex, R, DT, GT, OUT = sys.argv[1:6]
s = " ".join(open(tex).read().split()); R = Path(R); DT = Path(DT)
res = []
def chk(name, cond, detail=""):
    res.append(dict(check=name, passed=bool(cond), detail=str(detail))); print("PASS" if cond else "FAIL", name, detail)

f = json.load(open(R / "facts" / "facts.json"))["generator_holdout"]
chk("Table 1 fitting/selection/partition (normal) = facts.json", (f["normal"]["used_for_gradient_updates"], f["normal"]["held_out_for_checkpoint_selection"], f["normal"]["train_partition"]) == (210, 37, 247), f["normal"])
chk("Table 1 fitting/selection/partition (cancer) = facts.json", (f["cancer"]["used_for_gradient_updates"], f["cancer"]["held_out_for_checkpoint_selection"], f["cancer"]["train_partition"]) == (156, 28, 184), f["cancer"])
chk("Table 1 row strings present", "Normal & 210 & 37 & 247 & 49 & 52 & 348" in s and "Cancer & 156 & 28 & 184 & 36 & 40 & 260" in s)
# hold-out rule from code: n_val = max(2, round(0.15 n))
code = open(GT).read()
chk("code: n_val = max(2, round(val_fraction * n)), val_fraction = 0.15", "max(2, int(round(cfg.val_fraction * len(ordered))))" in code and "val_fraction: float = 0.15" in code)
chk("hold-out arithmetic 247->37, 184->28", (max(2, round(0.15 * 247)), max(2, round(0.15 * 184))) == (37, 28))
P = pd.read_csv(R / "screening_all" / "per_image_primary.csv"); d = P[P.group != "smoke"]
ct = pd.crosstab(d.group, d.gan_partition)
chk("val/test per class (49/52, 36/40) from per-image partition labels", (ct.loc["normal", "val"], ct.loc["normal", "test"], ct.loc["cancer", "val"], ct.loc["cancer", "test"]) == (49, 52, 36, 40), ct.to_dict())
# epochs: steps_per_epoch = len(train_loader) = floor(366/16) with drop_last
spe = 366 // 16
chk("NTP warm-up 20 epochs ~7.0 kimg and ramp 30 epochs ~10.6 kimg", (round(20 * spe * 16 / 1000, 1), round(30 * spe * 16 / 1000, 1)) == (7.0, 10.6), f"{spe} iterations/epoch")
chk("code: ntp_warmup_epochs = 20, ntp_ramp_epochs = 30, ntp_interval = 4, ntp_sub_batch = 8",
    all(x in code for x in ("ntp_warmup_epochs: int = 20", "ntp_ramp_epochs: int = 30", "ntp_interval: int = 4", "ntp_sub_batch: int = 8")))
chk("code: KID every 25 kimg, 256 generated/class, subsets 100 x 24, patience 600 kimg, min delta 5e-4",
    all(x in code for x in ("snapshot_kimg: float = 25.0", "metric_gen_per_class: int = 256", "kid_subsets: int = 100", "kid_subset_class: int = 24", "patience_kimg: float = 600.0", "kid_min_delta: float = 5e-4")))
chk("code: NTP bank updated only inside train_step from the training batch", "self.ntp_bank.update(real01[:cfg.ntp_sub_batch], labels[:cfg.ntp_sub_batch])" in code and code.count("ntp_bank.update(") == 1)
M = pd.read_csv(DT / "memorisation_summary.csv")
g = {(r.cls, r.set): r for r in M.itertuples()}
vals = [round(g[("normal", "QC-selected synthetic")].nn_sim_max, 2), round(g[("cancer", "QC-selected synthetic")].nn_sim_max, 2),
        round(g[("normal", "all NTP synthetic")].nn_sim_max, 2), round(g[("cancer", "all NTP synthetic")].nn_sim_max, 2)]
chk("memorisation maxima 0.94/0.92 (QC-selected) and 0.94/0.93 (pool)", vals == [0.94, 0.92, 0.94, 0.93], vals)
med = [round(g[("normal", "QC-selected synthetic")].nn_sim_median, 2), round(g[("cancer", "QC-selected synthetic")].nn_sim_median, 2),
       round(g[("normal", "all NTP synthetic")].nn_sim_median, 2), round(g[("cancer", "all NTP synthetic")].nn_sim_median, 2)]
chk("memorisation medians 0.90/0.88 and 0.87/0.83", med == [0.90, 0.88, 0.87, 0.83], med)
chk("no image at or above the 0.995 near-copy threshold", (M.frac_ge_0995_copy_threshold == 0).all())
Q = json.load(open(DT / "qc_clean_summary.json"))
chk("QC-clean kept 1,389 + 1,375 = 2,764", (Q["n_kept"]["normal"], Q["n_kept"]["cancer"]) == (1389, 1375))
U = pd.read_csv(DT / "generator_vs_unseen_177.csv").set_index(["cls", "set"])
a = U.loc[[("normal", "NTP all (unfiltered)"), ("cancer", "NTP all (unfiltered)")]]
chk("unseen KID 0.016/0.012, precision 0.74/0.87, recall 0.84/0.73",
    [round(x, 3) for x in a.KID_mean] == [0.016, 0.012] and [round(x, 2) for x in a.precision_matched] == [0.74, 0.87] and [round(x, 2) for x in a.recall_matched] == [0.84, 0.73])
G = pd.read_csv(DT / "gan_metrics_matched_n.csv").set_index(["cls", "set"])
chk("KID against the 431-image partition 0.013/0.011", [round(G.loc[(c, "NTP all (unfiltered)")].KID_mean, 3) for c in ("normal", "cancer")] == [0.013, 0.011])
chk("generator partition reference n = 247/184 in matched-n metrics", [int(G.loc[(c, "NTP all (unfiltered)")].n_real) for c in ("normal", "cancer")] == [247, 184])
sc = pd.read_csv(DT / "qc_ranked_complete_pool_scores.csv")
chk("2,124 top-ranked = 1,200 normal + 924 cancer", sc.groupby("Class").Selected.sum().to_dict() == {"cancer": 924, "normal": 1200})
x = re.sub(r"\$[^$]*\$", "X", s[s.index(r"\begin{abstract}") + 16:s.index(r"\end{abstract}")])
chk("no RTX 6000 / dysplasia left", "RTX 6000" not in s and "dysplasia" not in s)
chk("abstract <= 250 words", len(x.split()) <= 250, len(x.split()))
json.dump(res, open(OUT, "w"), indent=1)
print(f"{sum(r['passed'] for r in res)}/{len(res)} passed")
