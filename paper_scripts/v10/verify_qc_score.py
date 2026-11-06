"""Recovers the QC ranking score of gan_quality_selection.py from its saved per-image output (scores.csv) and
checks the selection rule. The script is read-only: it does not change or re-run the QC pipeline.

What it tests
1. Recomputes Final_Score from the raw per-image metrics with the weights that the script actually uses
   (nominal weights rescaled by 1/0.75 for the disabled discriminator and IQA terms, per-class min-max
   normalisation, LPIPS and local-density distances inverted, binary isolation-forest flag x (1 - 0.25 o)),
   once WITH the nearest-neighbour-distance term and once WITHOUT it. In compute_final_scores() the weight key
   'nn_dist' looks for the column 'n_nn_dist', but the normalised column is called 'n_nn', so the term never
   enters the score. The variant without the term reproduces the saved scores.
2. Checks that the selected images are the highest-ranked eligible images (class margin > 0, max DINOv2
   similarity < 0.995, diversity filter) up to min(floor(top_percent * N_gen), floor(ratio * N_real), N_eligible).

Usage: python verify_qc_score.py <scores.csv> [top_percent=0.8] [real_ratio=1e9] [n_real_normal=247] [n_real_cancer=184]
(The complete-pool run documented in the paper used top_percent = 0.8 and no ratio cap; the 215-image set
used the defaults 0.4 and 0.5, but its per-image scores were not saved.)"""
import json
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
top = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
n_real = {"normal": int(sys.argv[4]) if len(sys.argv) > 4 else 247, "cancer": int(sys.argv[5]) if len(sys.argv) > 5 else 184}

DESIRED = dict(dino=.20, clip=.12, disc=.10, lpips=.08, nn_dist=.07, local_density=.03, class_margin=.25,
               musiq=.05, maniqa=.05, brisque=.03, niqe=.02)
ACTIVE = ["dino", "clip", "lpips", "nn_dist", "local_density", "class_margin"]      # disc and IQA disabled
tot = sum(DESIRED[k] for k in ACTIVE)
W = {k: DESIRED[k] / tot for k in ACTIVE}                                            # = redistribute_weights()
COL = dict(dino=("DINO_Similarity", False), clip=("CLIP_Similarity", False), lpips=("LPIPS", True),
           nn_dist=("Nearest_Neighbor_Distance", True), local_density=("Local_Density", True),
           class_margin=("Class_Margin", False))


def mm(a, invert):
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi == lo:
        return np.full(len(a), 0.5)
    n = (a - lo) / (hi - lo)
    return 1 - n if invert else n


out = {"weights_after_redistribution": {k: round(v, 6) for k, v in W.items()}}
for name, keys in (("with_nn_term", ACTIVE), ("without_nn_term_as_executed", [k for k in ACTIVE if k != "nn_dist"])):
    err = 0.0
    for c, g in df.groupby("Class"):
        s = sum(W[k] * mm(g[COL[k][0]].to_numpy(float), COL[k][1]) for k in keys)
        s = s * (1 - 0.25 * g["Outlier_Score"].to_numpy(float))
        err = max(err, float(np.abs(s - g["Final_Score"].to_numpy(float)).max()))
    out[f"max_abs_error_{name}"] = err
out["sum_of_applied_weights"] = round(sum(W[k] for k in ACTIVE if k != "nn_dist"), 6)
out["outlier_values"] = sorted(map(float, df["Outlier_Score"].unique()))
sel = {}
for c, g in df.groupby("Class"):
    g = g.sort_values("Final_Score", ascending=False)
    elig = g[g["eligible"]]
    n_gen = len(g)
    k = min(max(1, int(n_gen * top)), max(1, int(n_real[c] * ratio)), len(elig))
    ok = set(elig["Image"].iloc[:k]) == set(g.loc[g["Selected"], "Image"])
    sel[c] = dict(n_generated=n_gen, class_ambiguous=int((g["Class_Margin"] <= 0).sum()),
                  near_copies=int((g["DINO_Max_Similarity"] >= 0.995).sum()), eligible=len(elig), selected=int(g["Selected"].sum()),
                  selected_equals_top_ranked_eligible=bool(ok), max_DINO_similarity_to_real=round(float(g["DINO_Max_Similarity"].max()), 4))
out["selection"] = sel
out["total_selected"] = int(df["Selected"].sum())
print(json.dumps(out, indent=1))
