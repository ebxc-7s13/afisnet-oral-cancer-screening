# =============================================================================
# afis_revision.py  --  revision analyses (session 11)
# -----------------------------------------------------------------------------
# R1  facts for Methods: generator hold-out counts, foreground-crop statistics, smoker census
# R2  generator fidelity against the 177 real images the generator never saw
# R3  screening with ALL suspicious-group (smoker) images against ALL 608 normal/cancer images, every image scored by
#     exactly one model that never saw it (symmetric), plus three sensitivity analyses
#     (5-model mean, per-fold-model replicates, JPEG-matched comparison groups)
# R4  region-clustered bootstrap CIs for pooled out-of-fold AUC (all 608 and the 177 GAN-unseen images)
# R5  optional: does the NTP loss matter downstream? final AFiS-Net + all baseline-generator images
#     vs + all NTP images (seed 42, 5 folds)
# Every function is idempotent and writes under RESULTS/revision11/.
# =============================================================================
import os
import io
import re
import json
import time
import zlib
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import afis_core as A
import afis_analysis as AN

OUT = A.RESULTS / "revision11"
CLASSES = ("normal", "cancer")
FINAL_MODEL = "AFiSNet_ConcatAvg"
SCREEN_FAMILY, SCREEN_EXP = "E9_final", "Raw_QCclean"      # final AFiS-Net trained with the QC-clean set
RETRAIN_FAMILY = "R11_screen"                               # used only if the seed-42 checkpoints are missing
NTP_FAMILY, NTP_EXP, NTP_MATCH_EXP = "R11_ntp", "Raw_AllBaseline", "Raw_NTPmatchBaseline"
SEED = 42
ORDER = ["normal", "smoke", "cancer"]
NAMES = {"normal": "Normal", "smoke": "Suspicious", "cancer": "Cancer"}   # "smoke" = suspicious group (tobacco smokers)
COLS = {"normal": "#4a5568", "smoke": "#dd6b20", "cancer": "#c53030"}
SMOKE_DIR_NAMES = {"smoke", "smoker", "smokers", "smoke_all", "smokers_all", "all_smokers", "susceptible"}
_ROI = re.compile(r"roi_(\d+)_(\d{8})_(\d{6})", re.IGNORECASE)


def _out(sub=""):
    d = OUT / sub if sub else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def _json(obj, path):
    def conv(v):
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    Path(path).write_text(json.dumps(conv(obj), indent=1))


def region_of(name):
    """Same key as kfold_classification.region_group: (roi id, acquisition date); otherwise the file name."""
    m = _ROI.search(os.path.basename(str(name)))
    return f"roi{m.group(1)}_{m.group(2)}" if m else os.path.basename(str(name))


def _images_recursive(d):
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(AN.IMG_EXTS)]
    return sorted(out)


def _md5(p):
    return AN._md5(p)


def _pub():
    AN._pub_rc()


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# =============================================================================
# Statistics helpers
# =============================================================================
def _holm(ps):
    return AN._holm(ps)


def _jt(groups):
    """Jonckheere-Terpstra test for an increasing trend (normal approximation, large samples)."""
    from scipy.stats import mannwhitneyu, norm
    J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            J += float(mannwhitneyu(groups[j], groups[i]).statistic)    # pairs with group j > group i (+0.5 ties)
    n = np.array([len(g) for g in groups], float); N = n.sum()
    mean = (N ** 2 - (n ** 2).sum()) / 4.0
    var = (N ** 2 * (2 * N + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72.0
    z = (J - mean) / np.sqrt(var)
    return float(J), float(z), float(norm.sf(z))


def _cliff(a, b):
    """Cliff's delta = P(a > b) - P(a < b)."""
    from scipy.stats import mannwhitneyu
    if not len(a) or not len(b):
        return float("nan")
    return float(2 * mannwhitneyu(a, b).statistic / (len(a) * len(b)) - 1)


def group_stats(df, col, order=ORDER, region_col="region"):
    """Image- and region-level comparison of one score between the groups."""
    from scipy.stats import kruskal, mannwhitneyu
    res = {}
    levels = [("image", df)]
    if region_col in df:
        levels.append(("region", df.groupby(["group", region_col], as_index=False)[col].mean()))
    for level, d in levels:
        vals = {g: d[d.group == g][col].dropna().values for g in order}
        vals = {g: v for g, v in vals.items() if len(v)}
        r = {"n": {g: len(v) for g, v in vals.items()},
             "median": {g: float(np.median(v)) for g, v in vals.items()},
             "iqr": {g: [float(np.quantile(v, .25)), float(np.quantile(v, .75))] for g, v in vals.items()},
             "mean": {g: float(np.mean(v)) for g, v in vals.items()}}
        if len(vals) >= 2 and all(len(v) > 1 for v in vals.values()):
            r["kruskal_p"] = float(kruskal(*vals.values()).pvalue)
        if all(g in vals for g in ("normal", "smoke", "cancer")):
            J, z, p = _jt([vals["normal"], vals["smoke"], vals["cancer"]])
            r["jonckheere_normal<smoke<cancer"] = {"J": J, "z": z, "p_one_sided": p}
        pw = []
        keys = list(vals)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = vals[keys[i]], vals[keys[j]]
                pw.append((f"{keys[i]} vs {keys[j]}", float(mannwhitneyu(a, b).pvalue), _cliff(b, a)))
        if pw:
            adj = _holm([p for _, p, _ in pw])
            r["pairwise"] = {k: {"p": p, "p_holm": float(h), "cliffs_delta_second_minus_first": cd}
                             for (k, p, cd), h in zip(pw, adj)}
        res[level] = r
    return res


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


# =============================================================================
# R1. Facts for Methods
# =============================================================================
def _crop_info(path):
    """Exactly the logic of classification.ForegroundCrop, returning what it did."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8); signal = arr.max(axis=2); h, w = signal.shape
    info = dict(width=w, height=h, coverage=np.nan, cropped=False, reason="", crop_w=w, crop_h=h)
    if signal.size == 0 or signal.max() < 8:
        info["reason"] = "empty image"; return info
    hist = np.bincount(signal.ravel(), minlength=256).astype(np.float64)
    weight = np.cumsum(hist); mean = np.cumsum(hist * np.arange(256)); total = weight[-1]
    between = np.zeros(256); valid = (weight > 0) & (weight < total)
    between[valid] = (mean[valid] * total - mean[-1] * weight[valid]) ** 2 / (weight[valid] * (total - weight[valid]))
    mask = signal > max(6, int(np.argmax(between)))
    cov = float(mask.mean()); info["coverage"] = cov
    if cov < 0.01 or cov > 0.92:
        info["reason"] = "coverage < 1%" if cov < 0.01 else "coverage > 92%"; return info
    ys, xs = np.where(mask)
    pad = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 0.08))
    l, r = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
    t, b = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
    if r - l < 8 or b - t < 8:
        info["reason"] = "box < 8 px"; return info
    info.update(cropped=True, crop_w=r - l, crop_h=b - t)
    return info


def facts(paths, smoker_dir=None):
    out = _out("facts")
    C, K = A.import_pipeline(paths)
    res = {}
    # ---- generator hold-out (gan_training._deterministic_holdout, val_fraction 0.15, per class)
    hold = {}
    rows = []
    for c in CLASSES:
        files = AN._list_images(os.path.join(paths["train"], c))
        ordered = sorted(files, key=lambda p: int(hashlib.sha256(os.path.basename(p).encode()).hexdigest()[:8], 16))
        n_val = max(2, int(round(0.15 * len(ordered))))
        hold[c] = {"train_partition": len(files), "used_for_gradient_updates": len(files) - n_val,
                   "held_out_for_checkpoint_selection": n_val}
        rows += [dict(image=os.path.basename(p), cls=c, role="checkpoint hold-out" if i < n_val else "generator fit")
                 for i, p in enumerate(ordered)]
    pd.DataFrame(rows).to_csv(out / "generator_holdout_per_image.csv", index=False)
    res["generator_holdout"] = hold
    # ---- foreground crop on the 608 development images (and smokers)
    pool = AN.real_pool(paths)
    groups = {"development images": [v[0] for v in pool.values()]}
    if smoker_dir:
        groups["smokers"] = _images_recursive(smoker_dir)
    crop_tf = C.ForegroundCrop()
    crop_rows = []
    for g, ps in groups.items():
        for p in ps:
            info = _crop_info(p)
            if len(crop_rows) < 20:                       # consistency check against the official class
                assert crop_tf(Image.open(p).convert("RGB")).size == (info["crop_w"], info["crop_h"]), p
            crop_rows.append(dict(group=g, image=os.path.basename(p), **info))
    cr = pd.DataFrame(crop_rows); cr["aspect_w_over_h"] = cr.crop_w / cr.crop_h
    cr.to_csv(out / "foreground_crop_per_image.csv", index=False)
    cs = {}
    for g, d in cr.groupby("group"):
        dc = d[d.cropped]
        cs[g] = {"n": len(d), "cropped": int(d.cropped.sum()),
                 "not_cropped": d[~d.cropped].reason.value_counts().to_dict(),
                 "input_size_px": sorted({f"{w}x{h}" for w, h in zip(d.width, d.height)})[:5],
                 "crop_side_px_median": [float(dc.crop_w.median()), float(dc.crop_h.median())] if len(dc) else None,
                 "aspect_ratio_median": float(dc.aspect_w_over_h.median()) if len(dc) else None,
                 "aspect_ratio_p5_p95": [float(dc.aspect_w_over_h.quantile(.05)), float(dc.aspect_w_over_h.quantile(.95))] if len(dc) else None,
                 "coverage_median": float(d.coverage.median())}
    res["foreground_crop"] = cs
    _json(res, out / "facts.json")
    print(json.dumps(res, indent=1, default=str))
    return res


# =============================================================================
# R2. Generator fidelity against the 177 real images the generator never saw
# =============================================================================
def gan_unseen(paths, n_rep=100, force=False):
    out = _out("gan_unseen")
    if (out / "gan_metrics_vs_unseen.csv").exists() and not force:
        print(pd.read_csv(out / "gan_metrics_vs_unseen.csv").round(4).to_string(index=False)); return
    if A.SMOKE:
        n_rep = 2
    dev = AN._dev()
    unseen = {c: AN._list_images(os.path.join(paths["val"], c)) + AN._list_images(os.path.join(paths["test"], c))
              for c in CLASSES}
    sets = {"Real training partition (region-disjoint)": {c: AN._list_images(os.path.join(paths["train"], c)) for c in CLASSES},
            "NTP all (unfiltered)": AN.class_sets(paths, "synthetic_all_ntp"),
            "NTP QC-clean (hygiene filter)": ({c: AN._list_images(str(A.qc_clean_dir() / c)) for c in CLASSES}
                                              if A.qc_clean_ready() else {c: [] for c in CLASSES}),
            "NTP QC-selected (215)": AN.class_sets(paths, "synthetic_selected"),
            "Baseline all (no NTP)": AN.class_sets(paths, "synthetic_all_baseline")}
    sets = {k: v for k, v in sets.items() if all(len(v[c]) >= 10 for c in CLASSES)}
    print({k: {c: len(v[c]) for c in CLASSES} for k, v in sets.items()}, "| reference:", {c: len(unseen[c]) for c in CLASSES})
    rows, drows = [], []
    for c in CLASSES:
        fr = AN._inception_features(unseen[c], dev)
        fg_all = {name: AN._inception_features(s[c], dev) for name, s in sets.items()}
        m = max(10, int(0.8 * min(len(v) for v in fg_all.values())))
        k_kid = min(50, len(fr), m)
        rng = np.random.default_rng(2026)
        kid_r = [rng.choice(len(fr), k_kid, replace=False) for _ in range(100)]
        kids = {}
        for name, fg in fg_all.items():
            r2 = np.random.default_rng(zlib.crc32(f"{c}|{name}".encode()))
            idx = [r2.choice(len(fg), m, replace=False) for _ in range(n_rep)]
            fids = np.array([AN._fid(fr, fg[i]) for i in idx])
            kid, _ = AN._kid_subsets(fr, fg, n_sub=100, size=k_kid, rng=np.random.default_rng(7), idx_r=kid_r)
            pr = pd.DataFrame([AN._prdc(fr, fg[i], k=5) for i in idx[:20]]).mean().to_dict()
            kids[name] = kid
            rows.append(dict(cls=c, set=name, n_reference_unseen=len(fr), n_set=len(fg), matched_n=m,
                             FID_matched_mean=fids.mean(), FID_matched_low=np.percentile(fids, 2.5),
                             FID_matched_high=np.percentile(fids, 97.5), KID_subset_size=k_kid,
                             KID_mean=kid.mean(), KID_sd=kid.std(), **{f"{k}_matched": v for k, v in pr.items()}))
        if "NTP all (unfiltered)" in kids and "Baseline all (no NTP)" in kids:
            kn, kb = kids["NTP all (unfiltered)"], kids["Baseline all (no NTP)"]
            drows.append(dict(cls=c, comparison="NTP all vs Baseline all, reference = 177 unseen real",
                              KID_ntp=kn.mean(), KID_baseline=kb.mean(),
                              relative_KID_reduction=1 - kn.mean() / kb.mean(),
                              frac_subsets_ntp_lower=float((kn < kb).mean())))
    df = pd.DataFrame(rows)
    ref = df[df.set.str.startswith("Real training")].set_index("cls")
    df["KID_relative_to_real_training"] = df.apply(lambda r: r.KID_mean / ref.loc[r.cls, "KID_mean"]
                                                   if r.cls in ref.index else np.nan, axis=1)
    df.to_csv(out / "gan_metrics_vs_unseen.csv", index=False)
    if drows:
        pd.DataFrame(drows).to_csv(out / "ntp_vs_baseline_vs_unseen.csv", index=False)
        print(pd.DataFrame(drows).round(4).to_string(index=False))
    show = df[["cls", "set", "n_set", "matched_n", "FID_matched_mean", "KID_mean", "KID_sd",
               "KID_relative_to_real_training", "precision_matched", "recall_matched", "coverage_matched"]]
    AN._df_to_latex(show.round(4), out / "table_gan_vs_unseen.tex",
                    "Generator fidelity against the 177 real development images that were not used to fit the generator "
                    "(Inception-v3 features). FID at matched sample size (mean over random subsets), KID (mean $\\pm$ s.d. "
                    "over 100 subsets), precision/recall/coverage ($k=5$). The real training partition, which is "
                    "region-disjoint from these images, gives the reference level expected for real cells.",
                    "tab:gan_unseen")
    print(show.round(4).to_string(index=False))


# =============================================================================
# R3. Screening: all smokers vs all 608 normal / cancer images
# =============================================================================
def find_smoker_dir(paths, input_root="/kaggle/input", override=None):
    if override:
        return override
    cands = []
    for d, _ in A._walk_dirs(input_root):
        if os.path.basename(d).lower() in SMOKE_DIR_NAMES:
            n = len(_images_recursive(d))
            if n:
                cands.append((n, d))
    if not cands:
        return None
    cands.sort(key=lambda t: (-t[0], -t[1].count(os.sep)))     # most images; on a tie the deepest folder
    print("Smoker folders found:", *[f"  {n:5d} images  {d}" for n, d in cands], sep="\n")
    print("Using the largest:", cands[0][1], "(set SMOKER_DIR to choose another)")
    return cands[0][1]


def screening_checkpoints():
    for fam in (SCREEN_FAMILY, RETRAIN_FAMILY):
        ck = AN._ckpts(fam, SCREEN_EXP, FINAL_MODEL)
        if len(ck) == A.N_FOLDS and all((p.parent / "split_manifest.json").exists() for p in ck.values()):
            return fam, ck
    return None, {}


def retrain_jobs():
    """Seed-42 final-model QC-clean runs with checkpoints kept (only needed if the old checkpoints are not attached)."""
    return [A.make_job(RETRAIN_FAMILY, FINAL_MODEL, SCREEN_EXP, SEED, f, "qc_clean", keep_ckpt=True, session=11)
            for f in range(A.N_FOLDS)]


def ntp_jobs(paths):
    """Final AFiS-Net + all baseline-generator (no NTP) images, seed 42. If the baseline set is not the same size as the
    NTP set (1,500 per class), an NTP arm with the same per-class counts is queued too, so the comparison is size-matched."""
    broot = _baseline_root(paths)
    if not broot or "synthetic_all_ntp" not in paths:
        return []
    nb = {c: len(AN._list_images(os.path.join(broot, c))) for c in CLASSES}
    nn = {c: len(AN._list_images(os.path.join(paths["synthetic_all_ntp"], c))) for c in CLASSES}
    print("Baseline-generator images per class:", nb, "| NTP images per class:", nn)
    if min(nb.values()) < (10 if A.SMOKE else 50):
        print("Too few baseline-generator images -> R5 skipped."); return []
    J = [A.make_job(NTP_FAMILY, FINAL_MODEL, NTP_EXP, SEED, f, "baseline_all_unfiltered", session=11) for f in range(A.N_FOLDS)]
    if any(nb[c] != nn[c] for c in CLASSES):
        J += [A.make_job(NTP_FAMILY, FINAL_MODEL, NTP_MATCH_EXP, SEED, f, "ntp_match_baseline", session=11) for f in range(A.N_FOLDS)]
    return J


def _baseline_root(paths):
    r = paths.get("synthetic_all_baseline")
    if r and os.path.isdir(r):
        return r
    g = A.RESULTS / "baseline_qc" / "generated" / "final_generated"
    return str(g) if g.is_dir() else None


def _scorer(C, members, dev, bs=32):
    """Batched version of afis_analysis._score_member (same probability, novelty and threshold)."""
    import torch
    _, eval_tf = C.build_transforms(224, C.Config())

    def run(fold, pils):
        model, ckd = members[fold]
        names = [str(n) for n in ckd.get("class_names", ["normal", "cancer"])]
        cidx = next((i for i, n in enumerate(names) if n.lower() == "cancer"), len(names) - 1)
        T = float(ckd.get("decision", {}).get("temperature", 1.0)) or 1.0
        ref = ckd["embedding_reference"]["classes"]
        cen = [np.asarray(ref[n]["centroid"].cpu() if hasattr(ref[n]["centroid"], "cpu") else ref[n]["centroid"]) for n in names]
        var = [np.asarray(ref[n]["variance"].cpu() if hasattr(ref[n]["variance"], "cpu") else ref[n]["variance"]) for n in names]
        P, N = [], []
        for i in range(0, len(pils), bs):
            x = torch.stack([eval_tf(im) for im in pils[i:i + bs]]).to(dev)
            with torch.no_grad():
                f = model.extract_features(x).float()
                P.append(torch.softmax(model.classifier(f).float() / max(T, 1e-6), 1)[:, cidx].cpu().numpy())
            fe = f.cpu().numpy()
            N.append(np.min(np.stack([((fe - c_) ** 2 / (v_ + 1e-8)).sum(1) for c_, v_ in zip(cen, var)], 1), 1))
        return np.concatenate(P), np.concatenate(N), float(ckd["decision"]["cancer_threshold"])
    return run, eval_tf


def _tiny(p):
    g = np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR), float).max(2).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def _smoker_inventory(smoker_dir, pool):
    """All smoker images, minus exact duplicates, near-duplicates and copies of development images."""
    files = _images_recursive(smoker_dir)
    pool_hash = {_md5(v[0]): b for b, v in pool.items()}
    rows, seen = [], {}
    for p in files:
        rel = os.path.relpath(p, smoker_dir)
        sub = os.path.dirname(rel).split(os.sep)[0] if os.path.dirname(rel) else ""
        b = os.path.basename(p); h = _md5(p)
        status = "kept"
        if b in pool or h in pool_hash:
            status = "excluded: copy of a development image"
        elif h in seen:
            status = f"excluded: exact duplicate of {seen[h]}"
        else:
            seen[h] = rel
        has_roi = bool(_ROI.search(b))
        rows.append(dict(image=rel, path=p, subfolder=sub, file_format=os.path.splitext(b)[1].lower().lstrip("."),
                         region=region_of(b) if has_roi else (sub or b), region_from="filename" if has_roi else
                         ("subfolder" if sub else "file name (singleton)"), md5=h, status=status))
    inv = pd.DataFrame(rows)
    keep = inv[inv.status == "kept"].index.tolist()
    if len(keep) > 1:                                   # near-duplicates (e.g. PNG and JPEG of the same cell)
        V = np.stack([_tiny(inv.loc[i, "path"]) for i in keep]) / 1024.0
        S = V @ V.T
        for a in range(len(keep)):
            if inv.loc[keep[a], "status"] != "kept":
                continue
            for b_ in np.where(S[a, a + 1:] > 0.995)[0] + a + 1:
                if inv.loc[keep[b_], "status"] == "kept":
                    inv.loc[keep[b_], "status"] = f"excluded: near-duplicate of {inv.loc[keep[a], 'image']}"
    return inv


def _assign_folds(regions, n_folds, seed=2026):
    """Every region of the smokers goes to exactly one fold model (balanced, reproducible)."""
    u = sorted(set(regions))
    perm = np.random.default_rng(seed).permutation(len(u))
    return {u[k]: int(i % n_folds) for i, k in enumerate(perm)}


def screening_full(paths, smoker_dir, force=False):
    from scipy.stats import mannwhitneyu
    out = _out("screening_all")
    fam, ck = screening_checkpoints()
    if not ck:
        print("Seed-42 checkpoints of the final AFiS-Net (QC-clean) are missing -> run the retraining cell first.")
        return
    if not smoker_dir:
        print("No smoker folder found -> screening skipped."); return
    C, K = A.import_pipeline(paths)
    pool = AN.real_pool(paths)
    inv = _smoker_inventory(smoker_dir, pool)
    inv.drop(columns=["path"]).to_csv(out / "smoker_inventory.csv", index=False)
    sm = inv[inv.status == "kept"].reset_index(drop=True)
    sig = hashlib.md5(("|".join(sorted(sm.md5)) + fam).encode()).hexdigest()
    flag = out / "_signature.txt"
    if flag.exists() and flag.read_text().strip() == sig and not force:
        print("Screening already done for this exact smoker set and checkpoints.")
        print((out / "summary.txt").read_text() if (out / "summary.txt").exists() else "")
        return
    print(f"Smokers: {len(inv)} files, {len(sm)} kept, {sm.region.nunique()} regions "
          f"(regions from {sm.region_from.value_counts().to_dict()}); excluded: "
          f"{inv[inv.status != 'kept'].status.str.replace(r' of (?!a development).*', '', regex=True).value_counts().to_dict() or 'none'}")
    dev = AN._dev()
    folds = sorted(ck)
    members = {f: AN.load_ckpt(C, ck[f], dev) for f in folds}
    run, eval_tf = _scorer(C, members, dev)
    test_fold = {}
    for f, pth in ck.items():
        for b in json.loads((pth.parent / "split_manifest.json").read_text())["test"]:
            test_fold[b] = f
    missing = [b for b in pool if b not in test_fold]
    assert not missing, f"{len(missing)} development images are in no test fold, e.g. {missing[:3]}"
    # ---- sanity: batched scorer == afis_analysis._score_member
    b0 = sorted(pool)[0]; f0 = test_fold[b0]
    pc1, ood1, _ = AN._score_member(members[f0][0], members[f0][1],
                                    eval_tf(Image.open(pool[b0][0]).convert("RGB")).unsqueeze(0).to(dev))
    pc2, ood2, _ = run(f0, [Image.open(pool[b0][0]).convert("RGB")])
    assert abs(pc1 - pc2[0]) < 1e-3 and abs(ood1 - ood2[0]) / max(1.0, ood1) < 1e-3, (pc1, pc2, ood1, ood2)
    # ---- 608 development images: each by the fold model that held it out
    dev_rows = []
    for f in folds:
        bs = sorted(b for b in pool if test_fold[b] == f)
        pils = [Image.open(pool[b][0]).convert("RGB") for b in bs]
        P, N, thr = run(f, pils)
        dev_rows += [dict(group=pool[b][1], image=b, region=region_of(b), fold_model=f, p_cancer=float(p), novelty=float(n),
                          threshold=thr, file_format=os.path.splitext(b)[1].lower().lstrip("."), gan_partition=pool[b][2])
                     for b, p, n in zip(bs, P, N)]
    devdf = pd.DataFrame(dev_rows)
    # ---- smokers: every fold model scores every smoker image (per-model scores kept)
    pils = [Image.open(p).convert("RGB") for p in sm.path]
    Psm, Nsm, thr = {}, {}, {}
    for f in folds:
        Psm[f], Nsm[f], thr[f] = run(f, pils)
    long = pd.concat([pd.DataFrame(dict(image=sm.image, region=sm.region, fold_model=f, p_cancer=Psm[f], novelty=Nsm[f],
                                        threshold=thr[f])) for f in folds], ignore_index=True)
    long.to_csv(out / "smokers_scores_by_every_fold_model.csv", index=False)
    assign = _assign_folds(sm.region.tolist(), len(folds))
    sm["fold_model"] = sm.region.map(lambda r: folds[assign[r]])
    pick = {(r.image, r.fold_model) for r in sm.itertuples()}
    prim = long[[(i, f) in pick for i, f in zip(long.image, long.fold_model)]].copy()
    prim["group"] = "smoke"; prim["file_format"] = prim.image.map(dict(zip(sm.image, sm.file_format)))
    df = pd.concat([devdf, prim], ignore_index=True)
    df["positive"] = df.p_cancer >= df.threshold; df["positive_at_0_5"] = df.p_cancer >= 0.5
    df.to_csv(out / "per_image_primary.csv", index=False)
    # ---- smoker ensemble (sensitivity A) and model disagreement
    P = np.stack([Psm[f] for f in folds], 1); NV = np.stack([Nsm[f] for f in folds], 1)
    TH = np.array([thr[f] for f in folds])
    ens = pd.DataFrame(dict(group="smoke", image=sm.image, region=sm.region, p_cancer=P.mean(1), p_cancer_sd=P.std(1),
                            novelty=NV.mean(1), threshold=float(np.median(TH)),
                            n_models_positive=(P >= TH[None, :]).sum(1)))
    ens.to_csv(out / "smokers_5model_mean.csv", index=False)
    df_ens = pd.concat([devdf, ens[["group", "image", "region", "p_cancer", "novelty", "threshold"]]], ignore_index=True)
    # ---- per-fold replicates (sensitivity B): model f scores its own held-out normal/cancer images and all smokers
    rep_rows = []
    for f in folds:
        d = pd.concat([devdf[devdf.fold_model == f],
                       pd.DataFrame(dict(group="smoke", image=sm.image, region=sm.region, p_cancer=Psm[f], novelty=Nsm[f]))])
        st = group_stats(d, "p_cancer")["image"]; sn = group_stats(d, "novelty")["image"]
        rep_rows.append(dict(fold_model=f, threshold=thr[f],
                             **{f"n_{g}": st["n"].get(g, 0) for g in ORDER},
                             **{f"p_median_{g}": st["median"].get(g) for g in ORDER},
                             **{f"p_mean_{g}": st["mean"].get(g) for g in ORDER},
                             **{f"novelty_median_{g}": sn["median"].get(g) for g in ORDER},
                             p_kruskal=st.get("kruskal_p"),
                             p_jt=st.get("jonckheere_normal<smoke<cancer", {}).get("p_one_sided"),
                             smoke_between=bool(st["median"].get("normal", 1) < st["median"].get("smoke", np.nan) < st["median"].get("cancer", 0))))
    reps = pd.DataFrame(rep_rows); reps.to_csv(out / "per_fold_model_replicates.csv", index=False)
    # ---- JPEG-matched comparison groups (sensitivity C)
    jpeg_res, df_jpeg = None, None
    sm_jpeg = sm.file_format.isin(["jpg", "jpeg"]).mean()
    if sm_jpeg > 0.5:
        qs = [q for q in (AN._jpeg_quality(p) for p in sm.path[sm.file_format.isin(["jpg", "jpeg"])]) if q]
        q = int(np.median(qs)) if qs else 95
        jr = []
        for f in folds:
            bs = sorted(b for b in pool if test_fold[b] == f)
            Pj, Nj, _ = run(f, [AN._jpeg_roundtrip(Image.open(pool[b][0]).convert("RGB"), q) for b in bs])
            jr += [dict(image=b, p_cancer_jpeg=float(p), novelty_jpeg=float(n)) for b, p, n in zip(bs, Pj, Nj)]
        jr = pd.DataFrame(jr)
        df_jpeg = devdf.merge(jr, on="image")
        df_jpeg.to_csv(out / "development_images_jpeg_rescored.csv", index=False)
        dj = df_jpeg.assign(p_cancer=df_jpeg.p_cancer_jpeg, novelty=df_jpeg.novelty_jpeg)[["group", "image", "region", "p_cancer", "novelty", "threshold"]]
        df_jpeg = pd.concat([dj, prim[["group", "image", "region", "p_cancer", "novelty", "threshold"]]], ignore_index=True)
        from sklearn.metrics import roc_auc_score
        y = (devdf.set_index("image").loc[jr.image].group == "cancer").astype(int).values
        jpeg_res = {"jpeg_quality": q, "share_of_smoker_files_jpeg": float(sm_jpeg),
                    "auc_608_lossless": float(roc_auc_score(y, devdf.set_index("image").loc[jr.image].p_cancer)),
                    "auc_608_jpeg": float(roc_auc_score(y, jr.p_cancer_jpeg))}
    # ---- statistics
    ref = devdf.novelty.values; q95 = float(np.quantile(ref, 0.95))
    S = {"design": {"checkpoints": f"{fam}/{FINAL_MODEL}/{SCREEN_EXP}/seed_{SEED}",
                    "normal_cancer": "all 608 development images, each scored by the fold model that held it out",
                    "smokers": "each smoker region assigned to one fold model (balanced); no model saw any smoker",
                    "smoker_images": int(len(sm)), "smoker_regions": int(sm.region.nunique()),
                    "smoker_files_excluded": inv[inv.status != "kept"].status.str.replace(r" of (?!a development).*", "", regex=True).value_counts().to_dict(),
                    "novelty_reference_q95_608": q95},
         "primary_probability": group_stats(df, "p_cancer"),
         "primary_novelty": group_stats(df, "novelty"),
         "sensitivity_5model_mean_probability": group_stats(df_ens, "p_cancer"),
         "sensitivity_5model_mean_novelty": group_stats(df_ens, "novelty"),
         "per_fold_model_replicates": {"smoke_between_normal_and_cancer_in_n_models": int(reps.smoke_between.sum()),
                                       "n_models": int(len(reps)), "max_jt_p": float(reps.p_jt.max()) if reps.p_jt.notna().any() else None},
         "smoker_model_disagreement": {"median_sd_between_models": float(ens.p_cancer_sd.median()),
                                       "share_unanimous_negative": float((ens.n_models_positive == 0).mean()),
                                       "share_unanimous_positive": float((ens.n_models_positive == len(folds)).mean()),
                                       "share_split": float(((ens.n_models_positive > 0) & (ens.n_models_positive < len(folds))).mean())}}
    for g in ORDER:
        d = df[df.group == g]
        S.setdefault("rates", {})[g] = {"n": int(len(d)), "positive_at_validation_threshold": float(d.positive.mean()),
                                        "positive_at_0_5": float(d.positive_at_0_5.mean()),
                                        "novelty_above_q95": float((d.novelty > q95).mean())}
    if "smoke" in set(df.group):
        for g in ("normal", "cancer"):
            S.setdefault("novelty_smoke_greater_than", {})[g] = float(mannwhitneyu(df[df.group == "smoke"].novelty,
                                                                                   df[df.group == g].novelty, alternative="greater").pvalue)
    if df_jpeg is not None:
        S["sensitivity_jpeg_matched_probability"] = group_stats(df_jpeg, "p_cancer")
        S["sensitivity_jpeg_matched_novelty"] = group_stats(df_jpeg, "novelty")
        S["jpeg"] = jpeg_res
    _json(S, out / "screening_stats.json")
    # ---- summary table
    trows = []
    for lab, d in (("Primary (one model per image)", df), ("Suspicious: mean of 5 models", df_ens)) + \
                  ((("JPEG-matched normal/cancer", df_jpeg),) if df_jpeg is not None else ()):
        st = group_stats(d, "p_cancer")
        for g in ORDER:
            dd = d[d.group == g]
            if not len(dd):
                continue
            trows.append(dict(analysis=lab, group=NAMES[g], images=len(dd), regions=dd.region.nunique(),
                              p_median=dd.p_cancer.median(), p_q1=dd.p_cancer.quantile(.25), p_q3=dd.p_cancer.quantile(.75),
                              p_mean=dd.p_cancer.mean(), novelty_median=dd.novelty.median(),
                              novelty_above_q95=(dd.novelty > q95).mean(),
                              jt_p_image=st["image"].get("jonckheere_normal<smoke<cancer", {}).get("p_one_sided"),
                              jt_p_region=st.get("region", {}).get("jonckheere_normal<smoke<cancer", {}).get("p_one_sided")))
    T = pd.DataFrame(trows); T.to_csv(out / "table_screening_all.csv", index=False)
    AN._df_to_latex(T.round(3), out / "table_screening_all.tex",
                    "Screening of all suspicious-group images against all 608 development images. Every image is scored by one "
                    "final AFiS-Net fold model (seed 42) that was not trained on it; suspicious-group regions are assigned to one "
                    "fold model each. JT: one-sided Jonckheere--Terpstra test for normal $<$ suspicious $<$ cancer.",
                    "tab:screening_all")
    lines = [f"Checkpoints: {S['design']['checkpoints']}",
             f"Suspicious group: {len(sm)} images / {sm.region.nunique()} regions; normal {int((devdf.group == 'normal').sum())}, "
             f"cancer {int((devdf.group == 'cancer').sum())}"]
    for r in T.itertuples():
        lines.append(f"{r.analysis:32s} {r.group:8s} n={r.images:4d} regions={r.regions:4d} "
                     f"P median {r.p_median:.3f} [{r.p_q1:.3f}-{r.p_q3:.3f}] mean {r.p_mean:.3f} | novelty median "
                     f"{r.novelty_median:.1f}, >q95 {100 * r.novelty_above_q95:.0f}% | JT p image {_fmt_p(r.jt_p_image)}, "
                     f"region {_fmt_p(r.jt_p_region)}")
    lines.append(f"Per-fold replicates: suspicious group between normal and cancer in {int(reps.smoke_between.sum())}/{len(reps)} models")
    if jpeg_res:
        lines.append(f"JPEG control (q={jpeg_res['jpeg_quality']}): AUC on 608 lossless {jpeg_res['auc_608_lossless']:.3f} "
                     f"vs JPEG {jpeg_res['auc_608_jpeg']:.3f}")
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    _fig_screening_all(df, reps, q95, out)
    del members
    flag.write_text(sig)


def _fig_screening_all(df, reps, q95, out):
    _pub()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.1], "wspace": 0.5})
    rng = np.random.default_rng(1)
    order = [g for g in ORDER if (df.group == g).any()]
    for ax, (col, ttl, tf, ylab) in zip(axes[:2], (("p_cancer", "A  Cancer probability", lambda v: v, "Predicted cancer probability"),
                                                   ("novelty", "B  Novelty", np.log10, "log$_{10}$ Mahalanobis distance"))):
        for i, g in enumerate(order):
            v = tf(df[df.group == g][col].values.astype(float))
            bp = ax.boxplot([v], positions=[i], widths=0.55, showfliers=False, patch_artist=True)
            for bb in bp["boxes"]:
                bb.set(facecolor=COLS[g], alpha=0.2, edgecolor=COLS[g])
            for k in ("medians", "whiskers", "caps"):
                for e in bp[k]:
                    e.set(color=COLS[g])
            a = 0.85 if len(v) < 80 else 0.35
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=5 if len(v) > 80 else 9, color=COLS[g], alpha=a, lw=0, zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{NAMES[g]}\n({(df.group == g).sum()})" for g in order], fontsize=7)
        ax.set_title(ttl, loc="left", fontsize=9); ax.set_ylabel(ylab); _despine(ax)
    axes[0].axhline(float(df.threshold.median()), ls="--", c="k", lw=0.7)
    axes[1].axhline(np.log10(q95), ls=":", c="k", lw=0.7)
    ax = axes[2]
    for g in order:
        ax.plot(reps.fold_model + 1, reps[f"p_mean_{g}"], "o-", color=COLS[g], ms=4, lw=1, label=NAMES[g])
    ax.set_xticks(reps.fold_model + 1); ax.set_xlabel("Fold model"); ax.set_ylabel("Mean cancer probability")
    ax.set_ylim(-0.02, 1.02); ax.set_title("C  Each fold model", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="best"); _despine(ax)
    AN._savefig(fig, out / "fig_screening_all", dpi=600)


# =============================================================================
# R4. Region-clustered bootstrap of pooled out-of-fold AUC
# =============================================================================
ARMS = {
    "final: real only": ("E9_final", FINAL_MODEL, "Raw"),
    "final: +215 QC-selected": ("E9_select", FINAL_MODEL, "Raw_SelectedSynthetic"),
    "final: +215 random": ("E9_final", FINAL_MODEL, "Raw_RandomUnfiltered"),
    "final: +3,000 all": ("E9_final", FINAL_MODEL, "Raw_AllUnfiltered"),
    "final: +QC-clean": ("E9_final", FINAL_MODEL, "Raw_QCclean"),
    "final: +random, QC-clean count": ("E9_final", FINAL_MODEL, "Raw_RandomMatchQCclean"),
    "final: +all baseline-generator images": (NTP_FAMILY, FINAL_MODEL, NTP_EXP),
    "final: +NTP images, baseline count": (NTP_FAMILY, FINAL_MODEL, NTP_MATCH_EXP),
    "XA: real only": ("E1_kfold", "AFiSNet", "Raw"),
    "XA: +215 QC-selected": ("E1_kfold", "AFiSNet", "Raw_SelectedSynthetic"),
    "XA: +215 random": ("E3_synthetic_source", "AFiSNet", "Raw_RandomUnfiltered"),
    "XA: +3,000 all": ("E3_synthetic_source", "AFiSNet", "Raw_AllUnfiltered"),
    "NextViT-S: real only": ("E1_kfold", "NextViT_Small", "Raw"),
    "CoAtNet-0: real only": ("E1_kfold", "CoAtNet0", "Raw"),
    "ResNet-50: real only": ("E1_kfold", "ResNet50", "Raw"),
}
COMPARISONS = [
    ("final: +3,000 all", "final: real only"),
    ("final: +QC-clean", "final: real only"),
    ("final: +215 QC-selected", "final: real only"),
    ("final: +QC-clean", "final: +3,000 all"),
    ("final: +QC-clean", "final: +random, QC-clean count"),
    ("final: +215 QC-selected", "final: +215 random"),
    ("final: +3,000 all", "final: +all baseline-generator images"),
    ("final: +NTP images, baseline count", "final: +all baseline-generator images"),
    ("XA: +3,000 all", "XA: real only"),
    ("XA: +215 QC-selected", "XA: real only"),
    ("XA: +215 QC-selected", "XA: +215 random"),
    ("XA: +3,000 all", "XA: +215 QC-selected"),
    ("final: real only", "NextViT-S: real only"),
    ("final: real only", "CoAtNet-0: real only"),
    ("final: real only", "ResNet-50: real only"),
]


def _arm_preds(arm):
    """{seed: DataFrame(Filename, y, p)} for every seed with all five folds."""
    fam, model, exp = ARMS[arm]
    base = A.RUNS / fam / model / exp
    res = {}
    for sd in sorted(base.glob("seed_*")) if base.exists() else []:
        fr = [sd / f"fold_{f}" / "predictions.csv" for f in range(A.N_FOLDS)]
        if all(p.exists() for p in fr):
            d = pd.concat([pd.read_csv(p) for p in fr], ignore_index=True)
            d = d.drop_duplicates("Filename")
            res[int(sd.name.split("_")[1])] = pd.DataFrame(dict(Filename=d.Filename, y=(d.GroundTruth == "cancer").astype(int),
                                                                p=d.Probability_Cancer.astype(float)))
    return res


class _AUC:
    """Weighted AUC for a fixed score vector; weights are bootstrap multiplicities (ties count one half)."""
    def __init__(self, y, s):
        o = np.argsort(s, kind="mergesort")
        self.o, self.y = o, np.asarray(y)[o].astype(float)
        _, self.inv = np.unique(np.asarray(s)[o], return_inverse=True)
        self.nb = self.inv.max() + 1

    def __call__(self, w=None):
        w = np.ones(len(self.o)) if w is None else np.asarray(w, float)[self.o]
        bw = np.bincount(self.inv, weights=w, minlength=self.nb)
        end = np.cumsum(bw); rank = (end - bw + end + 1) / 2.0          # average (weighted) rank of each tie block
        P = float((w * self.y).sum()); N = float(w.sum() - P)
        if P == 0 or N == 0:
            return np.nan
        return float(((w * self.y * rank[self.inv]).sum() - P * (P + 1) / 2.0) / (P * N))


def cluster_bootstrap(orig_split=None, n_boot=2000, seed=2026):
    """ΔAUC between two training sets: AUC of the pooled out-of-fold predictions per seed (608 images), averaged over the
    seeds both arms have; 95% CI from a bootstrap that resamples imaging regions (stratified by class), applied
    identically to both arms and all seeds. Repeated for the 177 images not used to fit the generator."""
    out = _out("cluster_bootstrap")
    preds = {a: _arm_preds(a) for a in ARMS}
    preds = {a: v for a, v in preds.items() if v}
    print("Arms with complete seeds:", {a: sorted(v) for a, v in preds.items()})
    if not preds:
        print("No predictions found."); return
    files = sorted(set.union(*[set(d.Filename) for v in preds.values() for d in v.values()]))
    lab = {}
    for v in preds.values():
        for d in v.values():
            lab.update(dict(zip(d.Filename, d.y)))
    files = [f for f in files if f in lab]
    idx = {f: i for i, f in enumerate(files)}
    y_all = np.array([lab[f] for f in files])
    reg = np.array([region_of(f) for f in files])
    regions = sorted(set(reg)); reg_i = np.array([regions.index(r) for r in reg])
    reg_cls = pd.Series(y_all).groupby(reg_i).first()            # regions hold one class
    by_cls = {c: reg_cls[reg_cls == c].index.values for c in (0, 1)}
    unseen = np.array([(orig_split or {}).get(f) in ("val", "test") for f in files]) if orig_split else None
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(regions)))
    for b in range(n_boot):
        for c, rs in by_cls.items():
            W[b] += np.bincount(rng.choice(rs, len(rs), replace=True), minlength=len(regions))
    Wimg = W[:, reg_i]                                             # image weights = multiplicity of its region

    def arm_matrix(arm):
        M = {}
        for s, d in preds[arm].items():
            v = np.full(len(files), np.nan); v[[idx[f] for f in d.Filename if f in idx]] = d.p.values[[f in idx for f in d.Filename]]
            M[s] = v
        return M
    mats = {a: arm_matrix(a) for a in preds}
    subsets = [("all", np.ones(len(files), bool))] + ([("gan_unseen", unseen)] if unseen is not None and unseen.any() else [])
    rows, crow = [], []
    for sub, m in subsets:
        for a in mats:
            seeds = sorted(mats[a]); est, boots = [], np.zeros((n_boot, len(seeds)))
            for k, s in enumerate(seeds):
                ok = m & np.isfinite(mats[a][s])
                f_ = _AUC(y_all[ok], mats[a][s][ok]); Wk = Wimg[:, ok]
                est.append(f_())
                boots[:, k] = [f_(Wk[b]) for b in range(n_boot)]
            bm = boots.mean(1)
            rows.append(dict(subset=sub, arm=a, seeds=",".join(map(str, seeds)), n_images=int(m.sum()),
                             n_regions=int(len(set(reg[m]))), auc_pooled=float(np.mean(est)),
                             ci_low=float(np.percentile(bm, 2.5)), ci_high=float(np.percentile(bm, 97.5))))
        for a, b_ in COMPARISONS:
            if a not in mats or b_ not in mats:
                continue
            seeds = sorted(set(mats[a]) & set(mats[b_]))
            if not seeds:
                continue
            d_est, d_boot = [], np.zeros((n_boot, len(seeds)))
            for k, s in enumerate(seeds):
                ok = m & np.isfinite(mats[a][s]) & np.isfinite(mats[b_][s])
                fa, fb = _AUC(y_all[ok], mats[a][s][ok]), _AUC(y_all[ok], mats[b_][s][ok]); Wk = Wimg[:, ok]
                d_est.append(fa() - fb())
                d_boot[:, k] = [fa(Wk[bb]) - fb(Wk[bb]) for bb in range(n_boot)]
            db = d_boot.mean(1)
            p = float(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean())))
            crow.append(dict(subset=sub, A=a, B=b_, seeds=",".join(map(str, seeds)), n_images=int(m.sum()),
                             d_auc=float(np.mean(d_est)), ci_low=float(np.percentile(db, 2.5)),
                             ci_high=float(np.percentile(db, 97.5)), p_bootstrap=max(p, 1.0 / n_boot)))
    R = pd.DataFrame(rows); Cc = pd.DataFrame(crow)
    if not Cc.empty:
        Cc["p_holm_within_subset"] = np.nan
        for sub, ii in Cc.groupby("subset").groups.items():
            Cc.loc[ii, "p_holm_within_subset"] = _holm(Cc.loc[ii, "p_bootstrap"].values)
    R.to_csv(out / "auc_by_arm_cluster_ci.csv", index=False)
    Cc.to_csv(out / "delta_auc_cluster_bootstrap.csv", index=False)
    if not Cc.empty:
        show = Cc.copy()
        show["$\\Delta$AUC [95\\% CI]"] = show.apply(lambda r: f"{r.d_auc:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
        AN._df_to_latex(show[["subset", "A", "B", "seeds", "n_images", "$\\Delta$AUC [95\\% CI]", "p_bootstrap", "p_holm_within_subset"]].round(4),
                        out / "table_delta_auc_cluster.tex",
                        "Difference in pooled out-of-fold AUC with 95\\% confidence intervals from a bootstrap over imaging "
                        f"regions (stratified by class, {n_boot} resamples), averaged over the seeds both training sets share. "
                        "gan\\_unseen: the 177 development images not used to fit the generator.", "tab:cluster_boot")
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False)); print(Cc.round(4).to_string(index=False))
    return R, Cc


# =============================================================================
# R5. NTP downstream comparison (statistics; the runs are queued in the notebook)
# =============================================================================
def ntp_downstream_stats():
    from scipy.stats import wilcoxon
    out = _out("ntp_downstream")
    rec = AN.load_records()
    if rec.empty:
        return
    def arm(fam, exp):
        d = rec[(rec.family == fam) & (rec.model == FINAL_MODEL) & (rec.experiment == exp) & (rec.seed == SEED)]
        return d.set_index("fold").roc_auc_macro
    base, raw = arm(NTP_FAMILY, NTP_EXP), arm("E9_final", "Raw")
    ntp = arm(NTP_FAMILY, NTP_MATCH_EXP)
    ntp_lab = "+NTP images (baseline count)"
    if ntp.empty:
        ntp, ntp_lab = arm("E9_final", "Raw_AllUnfiltered"), "+all NTP images"
    rows = []
    for la, a, lb, b in ((ntp_lab, ntp, "+all baseline images", base), ("+all baseline images", base, "real only", raw),
                         (ntp_lab, ntp, "real only", raw)):
        j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        if len(j) >= 2:
            d = j.a - j.b
            rows.append(dict(A=la, B=lb, folds=len(j), auc_A=j.a.mean(), auc_B=j.b.mean(), d_auc=d.mean(),
                             folds_A_better=int((d > 0).sum()), p_wilcoxon=float(wilcoxon(d).pvalue) if (d != 0).any() else np.nan))
    R = pd.DataFrame(rows); R.to_csv(out / "ntp_vs_baseline_downstream.csv", index=False)
    print(R.round(4).to_string(index=False) if len(R) else "NTP downstream runs not complete yet.")
