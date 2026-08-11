#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
screening_analysis.py
=====================
Stage 5 of the pipeline accompanying:

    "Quality Controlled Synthetic Augmentation for AI-Enabled Label-free
     Digital Cytology of Oral Cancer Screening"

FEATURE-SPACE SCREENING OF AN INDEPENDENT COHORT with a trained binary
(normal-vs-cancer) encoder, applied to three groups ordered by the clinical
progression hypothesis:  normal < smoke < cancer.

SCIENTIFIC FRAMING (important)
------------------------------
The encoder (e.g. AFiSNet / AFiS-Net) is trained ONLY on normal and
cancer cells. The smoker ("suspected") group is an UNSEEN cohort that was
excluded from generator training, classifier training, model selection, and
threshold selection. This script does NOT claim that the binary model
diagnoses that group. It tests whether the learned malignancy-related
feature space places the unseen cohort at an intermediate position between
the two trained classes. For every image it reports:

  * P(cancer) from the classifier head          (temperature-calibrated)
  * P(cancer) from prototype/Mahalanobis route  (independent, feature-space)
  * confidence and predictive entropy           (per-sample certainty)
  * epistemic uncertainty                       (std across ensemble members)
  * Mahalanobis novelty / OOD score             (distance from the two
                                                 trained class distributions;
                                                 expected to be higher for a
                                                 genuinely unseen group)

Results are aggregated per group and per ROI, with progression statistics
(Spearman monotonic trend, Kruskal-Wallis, one-sided pairwise Mann-Whitney U,
and a normal-vs-cancer AUC internal validation), and written as CSVs, a text
report, and publication-style figures.

By default a k-fold ENSEMBLE is used: point --checkpoint at a directory and
every best_model.pt beneath it (e.g. the five cross-validation folds from
kfold_classification.py) is averaged; the spread across members provides the
epistemic uncertainty. A single checkpoint file also works.

USAGE
  python screening_analysis.py \
      --checkpoint results/kfold/AFiSNet/Raw_SelectedSynthetic \
      --data-root  data/screening \
      --groups normal,smoke,cancer \
      --output results/screening

where --data-root contains one subfolder per group (common folder-name
aliases such as "smoker" or "healthy" are recognised), each full of images.
"""

from __future__ import annotations

import os
import re
import csv
import glob
import json
import argparse
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from PIL import Image

# Reuse the trained, unmodified pipeline: model construction + weight loading,
# the exact evaluation preprocessing, and the Mahalanobis prototype scoring.
import classification as C

try:
    from scipy import stats as _sp_stats
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Ordinal malignancy rank encodes the clinical progression hypothesis. Aliases
# map common folder spellings to a canonical group name.
GROUP_ORDER = ["normal", "smoke", "cancer"]
GROUP_RANK = {"normal": 0, "smoke": 1, "cancer": 2}
GROUP_ALIASES = {
    "normal": "normal", "healthy": "normal",
    "smoke": "smoke", "smoker": "smoke", "smoking": "smoke",
    "cancer": "cancer", "malignant": "cancer", "oscc": "cancer", "tumor": "cancer",
}
_ROI_RE = re.compile(r"roi_(\d+)_(\d{8})_(\d{6})", re.IGNORECASE)


# =============================================================================
# Model loading (k-fold ensemble or single checkpoint)
# =============================================================================
def find_checkpoints(path: str) -> List[str]:
    """Return one or more checkpoint paths. A directory is searched recursively
    for best_model.pt (the k CV folds -> ensemble); a file is used directly."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "**", "best_model.pt"), recursive=True))
        if found:
            return found
    raise FileNotFoundError(
        f"No checkpoint found at '{path}'. Pass a best_model.pt or a folder "
        f"containing fold_*/best_model.pt.")


def _load_deployment_model(checkpoint_path: str, device: torch.device):
    """Load a trusted deployment checkpoint (built by classification.run_one).

    Mirrors classification.load_deployment_model but forces ``weights_only=False``.
    PyTorch >= 2.6 defaults torch.load to ``weights_only=True``, which cannot
    unpickle the numpy scalars our full checkpoints legitimately contain (saved
    config, metrics, embedding reference). These files are produced by this same
    pipeline, so full unpickling is safe here.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", {})
    backbone = architecture.get("backbone", checkpoint.get("model_name"))
    class_names = checkpoint.get("class_names")
    state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if backbone not in C.MODEL_SPECS or not class_names or state is None:
        raise ValueError("Unsupported checkpoint: expected a deployment checkpoint "
                         "produced by classification.run_one().")
    model = C.build_model(backbone, len(class_names), device, allow_random_init=True,
                          pretrained=False,
                          embedding_dim=architecture.get("feature_dim", 256),
                          embedding_dropout=architecture.get("embedding_dropout", 0.10))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def load_ensemble(path: str, device: torch.device):
    """Load every checkpoint into (model, checkpoint) pairs, all in eval mode."""
    ckpts = find_checkpoints(path)
    members = []
    for cp in ckpts:
        try:
            model, checkpoint = _load_deployment_model(cp, device)
            members.append((model, checkpoint, cp))
        except Exception as exc:
            print(f"[WARN] could not load {cp}: {exc}")
    if not members:
        raise RuntimeError("No usable checkpoints could be loaded.")
    backbone = members[0][1].get("architecture", {}).get("backbone", "unknown")
    print(f"[MODEL] Loaded {len(members)} checkpoint(s) for ensemble | backbone={backbone}")
    return members


# =============================================================================
# Data discovery
# =============================================================================
def canonical_group(name: str) -> Optional[str]:
    key = name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    for alias, canon in GROUP_ALIASES.items():
        if key == alias.replace("-", "").replace("_", ""):
            return canon
    return None


def discover_groups(data_root: str, requested: List[str]) -> "OrderedDict[str, List[str]]":
    """Map each requested group to its image files (folder matched by alias)."""
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"--data-root not found: {data_root}")
    subdirs = {d.lower(): os.path.join(data_root, d)
               for d in os.listdir(data_root)
               if os.path.isdir(os.path.join(data_root, d))}
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for req in requested:
        canon = canonical_group(req) or req.strip().lower()
        folder = None
        for dname, dpath in subdirs.items():
            if canonical_group(dname) == canon or dname == req.strip().lower():
                folder = dpath
                break
        if folder is None:
            print(f"[WARN] no folder found for group '{req}' under {data_root}; skipping.")
            continue
        imgs = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if imgs:
            groups[canon] = imgs
            print(f"[DATA] group '{canon}': {len(imgs)} images  ({folder})")
        else:
            print(f"[WARN] group '{canon}' folder has no images: {folder}")
    if not groups:
        raise RuntimeError("No image groups discovered.")
    return groups


def roi_of(path: str) -> str:
    m = _ROI_RE.search(os.path.basename(path))
    return f"roi{m.group(1)}_{m.group(2)}" if m else os.path.basename(path)


# =============================================================================
# Inference
# =============================================================================
def _cancer_index(class_names: List[str]) -> int:
    for i, c in enumerate(class_names):
        if str(c).lower() == "cancer":
            return i
    return len(class_names) - 1  # fall back to last


@torch.no_grad()
def predict_batch(members, images: torch.Tensor, device: torch.device
                  ) -> Dict[str, np.ndarray]:
    """Run every ensemble member on a batch; return per-sample screening signals.

    Returns arrays indexed by sample: p_cancer_clf, p_cancer_proto, confidence,
    entropy, epistemic_std, ood_score, and the primary-member embedding.
    """
    images = images.to(device)
    per_fold_pcancer = []       # classifier P(cancer) per member
    per_fold_pproto = []        # prototype P(cancer) per member
    per_fold_ood = []           # min Mahalanobis distance per member
    primary_embedding = None

    for idx, (model, checkpoint, _) in enumerate(members):
        class_names = checkpoint.get("class_names", ["normal", "cancer"])
        cidx = _cancer_index(class_names)
        temp = float(checkpoint.get("decision", {}).get("temperature", 1.0)) or 1.0
        feats = model.extract_features(images).float()           # [B, 256] L2-normed
        logits = model.classifier(feats).float() / max(temp, 1e-6)
        probs = torch.softmax(logits, dim=1)
        per_fold_pcancer.append(probs[:, cidx].cpu().numpy())
        feats_np = feats.cpu().numpy()
        if idx == 0:
            primary_embedding = feats_np
        # prototype (feature-space) route + OOD, using this member's reference
        ref = checkpoint.get("embedding_reference")
        if ref and "classes" in ref:
            names = list(class_names)
            cen = np.stack([ref["classes"][n]["centroid"].cpu().numpy()
                            if hasattr(ref["classes"][n]["centroid"], "cpu")
                            else np.asarray(ref["classes"][n]["centroid"]) for n in names])
            var = np.stack([ref["classes"][n]["variance"].cpu().numpy()
                            if hasattr(ref["classes"][n]["variance"], "cpu")
                            else np.asarray(ref["classes"][n]["variance"]) for n in names])
            d = ((feats_np[:, None, :] - cen[None, :, :]) ** 2 / (var[None, :, :] + 1e-8)).sum(2)
            per_fold_ood.append(d.min(axis=1))
            p_proto = torch.softmax(torch.tensor(-d), dim=1).numpy()[:, cidx]
            per_fold_pproto.append(p_proto)

    pcancer = np.stack(per_fold_pcancer, axis=0)                 # [folds, B]
    mean_pcancer = pcancer.mean(axis=0)
    epistemic = pcancer.std(axis=0) if pcancer.shape[0] > 1 else np.zeros_like(mean_pcancer)
    p2 = np.stack([mean_pcancer, 1 - mean_pcancer], axis=1)
    confidence = p2.max(axis=1)
    entropy = -(p2 * np.log(np.clip(p2, 1e-8, 1.0))).sum(axis=1)
    out = {
        "p_cancer_clf": mean_pcancer,
        "p_cancer_proto": (np.stack(per_fold_pproto).mean(0)
                           if per_fold_pproto else np.full_like(mean_pcancer, np.nan)),
        "confidence": confidence,
        "entropy": entropy,
        "epistemic_std": epistemic,
        "ood_score": (np.stack(per_fold_ood).mean(0)
                      if per_fold_ood else np.full_like(mean_pcancer, np.nan)),
        "embedding": primary_embedding,
    }
    return out


def load_images(paths: List[str], transform, size: int) -> torch.Tensor:
    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (size, size))
        tensors.append(transform(img))
    return torch.stack(tensors)


def run_inference(members, groups, cfg, device, batch_size=32) -> List[dict]:
    """Per-sample screening records across all groups."""
    primary_ckpt = members[0][1]
    img_size = primary_ckpt.get("preprocessing", {}).get("image_size") \
        or C.MODEL_SPECS.get(primary_ckpt.get("architecture", {}).get("backbone", ""), {}).get("img_size", 224)
    _, eval_tf = C.build_transforms(img_size, cfg)

    records = []
    for group, paths in groups.items():
        embs = []
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            batch = load_images(chunk, eval_tf, img_size)
            res = predict_batch(members, batch, device)
            embs.append(res["embedding"])
            for j, path in enumerate(chunk):
                records.append({
                    "group": group,
                    "group_rank": GROUP_RANK.get(group, np.nan),
                    "filename": os.path.basename(path),
                    "roi": roi_of(path),
                    "p_cancer_clf": float(res["p_cancer_clf"][j]),
                    "p_cancer_proto": float(res["p_cancer_proto"][j]),
                    "confidence": float(res["confidence"][j]),
                    "entropy": float(res["entropy"][j]),
                    "epistemic_std": float(res["epistemic_std"][j]),
                    "ood_score": float(res["ood_score"][j]),
                })
        # stash embeddings on the group's records for the projection plot
        if embs:
            E = np.concatenate(embs, axis=0)
            gi = [r for r in records if r["group"] == group]
            for k, r in enumerate(gi[-len(E):]):
                r["_emb"] = E[k]
    return records


# =============================================================================
# Aggregation + statistics
# =============================================================================
def _ci95(x):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    m, s = x.mean(), x.std(ddof=1)
    h = 1.96 * s / np.sqrt(len(x))
    return (m - h, m + h)


def aggregate_groups(records: List[dict], threshold: float = 0.5) -> List[dict]:
    by = defaultdict(list)
    for r in records:
        by[r["group"]].append(r)
    rows = []
    for group in sorted(by, key=lambda g: GROUP_RANK.get(g, 99)):
        rs = by[group]
        pc = np.array([r["p_cancer_clf"] for r in rs])
        lo, hi = _ci95(pc)
        rows.append({
            "group": group, "group_rank": GROUP_RANK.get(group, np.nan), "n": len(rs),
            "mean_p_cancer": float(pc.mean()), "std_p_cancer": float(pc.std()),
            "ci95_low": lo, "ci95_high": hi, "median_p_cancer": float(np.median(pc)),
            "mean_p_cancer_proto": float(np.nanmean([r["p_cancer_proto"] for r in rs])),
            "mean_confidence": float(np.mean([r["confidence"] for r in rs])),
            "mean_entropy": float(np.mean([r["entropy"] for r in rs])),
            "mean_epistemic_std": float(np.mean([r["epistemic_std"] for r in rs])),
            "mean_ood_novelty": float(np.nanmean([r["ood_score"] for r in rs])),
            "percent_predicted_cancer": float(100.0 * (pc >= threshold).mean()),
        })
    return rows


def aggregate_rois(records: List[dict], threshold: float = 0.5) -> List[dict]:
    by = defaultdict(list)
    for r in records:
        by[(r["group"], r["roi"])].append(r)
    rows = []
    for (group, roi), rs in sorted(by.items(), key=lambda kv: (GROUP_RANK.get(kv[0][0], 99), kv[0][1])):
        pc = np.array([r["p_cancer_clf"] for r in rs])
        rows.append({
            "group": group, "roi": roi, "n_frames": len(rs),
            "mean_p_cancer": float(pc.mean()), "std_p_cancer": float(pc.std()),
            "mean_confidence": float(np.mean([r["confidence"] for r in rs])),
            "mean_ood_novelty": float(np.nanmean([r["ood_score"] for r in rs])),
            "roi_prediction": "cancer" if pc.mean() >= threshold else "normal",
        })
    return rows


def progression_statistics(records: List[dict]) -> dict:
    """Monotonic-trend, group-difference and Normal-vs-Cancer AUC tests."""
    out = {}
    ranks = np.array([r["group_rank"] for r in records], float)
    pc = np.array([r["p_cancer_clf"] for r in records], float)
    valid = ~np.isnan(ranks)
    ranks, pc = ranks[valid], pc[valid]

    # Spearman monotonic trend (ordinal group -> cancer probability)
    if _HAS_SCIPY and len(np.unique(ranks)) > 1:
        rho, p = _sp_stats.spearmanr(ranks, pc)
        out["spearman_trend_rho"] = float(rho)
        out["spearman_trend_p"] = float(p)
        # Kruskal-Wallis across groups
        groups_vals = [pc[ranks == g] for g in np.unique(ranks)]
        try:
            h, pk = _sp_stats.kruskal(*groups_vals)
            out["kruskal_H"] = float(h); out["kruskal_p"] = float(pk)
        except Exception:
            pass
        # pairwise Mann-Whitney U (adjacent groups) + Normal vs Cancer AUC
        out["pairwise"] = {}
        uniq = sorted(np.unique(ranks))
        inv = {v: k for k, v in GROUP_RANK.items()}
        for a, b in zip(uniq[:-1], uniq[1:]):
            va, vb = pc[ranks == a], pc[ranks == b]
            if len(va) and len(vb):
                try:
                    u, pu = _sp_stats.mannwhitneyu(va, vb, alternative="less")
                    out["pairwise"][f"{inv.get(a,a)}<{inv.get(b,b)}"] = float(pu)
                except Exception:
                    pass
    # Normal vs Cancer AUC (sanity: does the loaded model separate the KNOWN classes?)
    n_mask = ranks == GROUP_RANK["normal"]; c_mask = ranks == GROUP_RANK["cancer"]
    if n_mask.any() and c_mask.any():
        y = np.concatenate([np.zeros(n_mask.sum()), np.ones(c_mask.sum())])
        s = np.concatenate([pc[n_mask], pc[c_mask]])
        try:
            from sklearn.metrics import roc_auc_score
            out["normal_vs_cancer_auc"] = float(roc_auc_score(y, s))
        except Exception:
            pass
    return out


# =============================================================================
# Figures (publication style)
# =============================================================================
def _order_present(groups_present):
    return [g for g in GROUP_ORDER if g in groups_present]


def figures(records, group_rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 15,
                         "figure.autolayout": True})
    present = _order_present({r["group"] for r in records})
    data = {g: np.array([r["p_cancer_clf"] for r in records if r["group"] == g]) for g in present}
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.9, len(present)))

    # (1) headline: violin/box of P(cancer) by group, ordered by malignancy
    fig, ax = plt.subplots(figsize=(9, 6))
    parts = ax.violinplot([data[g] for g in present], showmeans=True, showextrema=False)
    for pc_, c in zip(parts["bodies"], colors):
        pc_.set_facecolor(c); pc_.set_alpha(0.6)
    ax.set_xticks(range(1, len(present) + 1)); ax.set_xticklabels(present, fontweight="bold")
    ax.set_ylabel("Predicted cancer probability", fontweight="bold")
    ax.set_title("Feature-space malignancy continuum", fontweight="bold")
    ax.axhline(0.5, ls="--", c="grey", alpha=0.7); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"cancer_probability_by_group.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (2) mean +/- CI progression line
    fig, ax = plt.subplots(figsize=(8, 6))
    gr = {r["group"]: r for r in group_rows}
    xs = list(range(len(present)))
    means = [gr[g]["mean_p_cancer"] for g in present]
    err = [[gr[g]["mean_p_cancer"] - gr[g]["ci95_low"] for g in present],
           [gr[g]["ci95_high"] - gr[g]["mean_p_cancer"] for g in present]]
    ax.errorbar(xs, means, yerr=err, marker="o", ms=10, lw=3, capsize=6, color="#b2182b")
    ax.set_xticks(xs); ax.set_xticklabels(present, fontweight="bold")
    ax.set_ylabel("Mean cancer probability (95% CI)", fontweight="bold")
    ax.set_title("Malignancy progression", fontweight="bold"); ax.grid(alpha=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"progression_mean_ci.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # (3) novelty / OOD by group (shows the unseen Smoke group is unfamiliar)
    ood = {g: np.array([r["ood_score"] for r in records if r["group"] == g and r["ood_score"] == r["ood_score"]]) for g in present}
    if any(len(v) for v in ood.values()):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.boxplot([ood[g] for g in present], patch_artist=True)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, fontweight="bold")
        ax.set_ylabel("Mahalanobis novelty (OOD)", fontweight="bold")
        ax.set_title("Distance from the binary model's known classes", fontweight="bold"); ax.grid(alpha=0.25)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"novelty_ood_by_group.{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # (4) 2D PCA embedding coloured by group (primary member's space)
    embs = [(r["group"], r["_emb"]) for r in records if "_emb" in r]
    if embs:
        try:
            from sklearn.decomposition import PCA
            X = np.stack([e for _, e in embs]); g = [gg for gg, _ in embs]
            xy = PCA(n_components=2).fit_transform(X)
            fig, ax = plt.subplots(figsize=(8, 7))
            for gi, c in zip(present, colors):
                m = [i for i, gg in enumerate(g) if gg == gi]
                ax.scatter(xy[m, 0], xy[m, 1], s=22, color=c, alpha=0.7, label=gi, edgecolors="none")
            ax.set_title("Embedding space (PCA) by group", fontweight="bold")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); ax.grid(alpha=0.2)
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(out_dir, f"embedding_pca_by_group.{ext}"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"[WARN] PCA plot skipped: {exc}")


# =============================================================================
# Reporting
# =============================================================================
def _f(v, n=3):
    try:
        return f"{float(v):.{n}f}" if float(v) == float(v) else "n/a"
    except Exception:
        return "n/a"


def write_outputs(records, group_rows, roi_rows, stats, out_dir, ckpt_src):
    os.makedirs(out_dir, exist_ok=True)
    # per-sample CSV
    fields = ["group", "group_rank", "roi", "filename", "p_cancer_clf", "p_cancer_proto",
              "confidence", "entropy", "epistemic_std", "ood_score"]
    with open(os.path.join(out_dir, "per_sample_predictions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fields})
    # per-group CSV
    with open(os.path.join(out_dir, "per_group_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(group_rows[0].keys()))
        w.writeheader(); w.writerows(group_rows)
    # per-ROI CSV
    if roi_rows:
        with open(os.path.join(out_dir, "per_roi_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(roi_rows[0].keys()))
            w.writeheader(); w.writerows(roi_rows)
    json.dump(stats, open(os.path.join(out_dir, "progression_statistics.json"), "w"), indent=2)

    # report
    L = []
    L.append("=" * 88)
    L.append("INDEPENDENT SCREENING — FEATURE-SPACE ANALYSIS (Normal / Smoke / Cancer)")
    L.append("=" * 88)
    L.append(f"Encoder checkpoints: {ckpt_src}")
    L.append("Note: the encoder is BINARY (Normal vs Cancer). Smoke is unseen; its score is a")
    L.append("position on the learned Normal<->Cancer axis, not a diagnosis.")
    L.append("")
    L.append("-" * 88)
    L.append(f"{'group':12s} {'n':>5} {'meanP(cancer)':>15} {'95% CI':>17} {'conf':>6} "
             f"{'entropy':>8} {'OOD':>8} {'%pred.cancer':>12}")
    L.append("-" * 88)
    for r in group_rows:
        L.append(f"{r['group']:12s} {r['n']:>5} {_f(r['mean_p_cancer']):>15} "
                 f"[{_f(r['ci95_low'])},{_f(r['ci95_high'])}] {_f(r['mean_confidence']):>6} "
                 f"{_f(r['mean_entropy']):>8} {_f(r['mean_ood_novelty'],1):>8} "
                 f"{_f(r['percent_predicted_cancer'],1):>12}")
    L.append("")
    L.append("PROGRESSION ANALYSIS (does malignancy score rise Normal->Smoke->Cancer?)")
    if "spearman_trend_rho" in stats:
        L.append(f"  Spearman monotonic trend : rho={_f(stats['spearman_trend_rho'])}, "
                 f"p={_f(stats.get('spearman_trend_p'),5)}")
    if "kruskal_H" in stats:
        L.append(f"  Kruskal-Wallis (groups)  : H={_f(stats['kruskal_H'],2)}, p={_f(stats.get('kruskal_p'),5)}")
    for k, v in stats.get("pairwise", {}).items():
        L.append(f"  Mann-Whitney {k:22s}: p={_f(v,5)} (one-sided)")
    if "normal_vs_cancer_auc" in stats:
        L.append(f"  Normal-vs-Cancer AUC     : {_f(stats['normal_vs_cancer_auc'])}  "
                 f"(sanity check on the two known classes)")
    L.append("")
    # verdict
    rho = stats.get("spearman_trend_rho", float("nan"))
    if rho == rho:
        if rho > 0.3 and stats.get("spearman_trend_p", 1) < 0.05:
            L.append("VERDICT: the feature space shows a significant monotonic malignancy continuum;")
            L.append("         the binary encoder generalises to grade unseen intermediate states.")
        else:
            L.append("VERDICT: no strong monotonic continuum detected — the binary embedding does not")
            L.append("         cleanly order the intermediate states; a multi-class Stage-3 model is advised.")
    report = "\n".join(L)
    open(os.path.join(out_dir, "screening_report.txt"), "w", encoding="utf-8").write(report)
    print("\n" + report)


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Independent feature-space screening across "
                                            "Normal/Smoke/Cancer groups.")
    p.add_argument("--checkpoint", required=True,
                   help="A best_model.pt OR a folder with fold_*/best_model.pt (ensembled).")
    p.add_argument("--data-root", required=True,
                   help="Folder containing one subfolder per group.")
    p.add_argument("--groups", default="normal,smoke,cancer",
                   help="Comma-separated group names (folder aliases accepted).")
    p.add_argument("--output", default=os.path.join("results", "screening"),
                   help="Output directory (default: results/screening).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Cancer decision threshold for %%-predicted-cancer (default 0.5).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("=" * 72)
    print("INDEPENDENT SCREENING — feature-space malignancy continuum")
    print("=" * 72)
    print(f"[device] {device}")

    cfg = C.Config()
    cfg.use_foreground_crop = True  # match the training-time eval preprocessing

    members = load_ensemble(args.checkpoint, device)
    groups = discover_groups(args.data_root, [g for g in args.groups.split(",") if g.strip()])

    records = run_inference(members, groups, cfg, device, batch_size=args.batch_size)
    group_rows = aggregate_groups(records, threshold=args.threshold)
    roi_rows = aggregate_rois(records, threshold=args.threshold)
    stats = progression_statistics(records)

    os.makedirs(args.output, exist_ok=True)
    figures(records, group_rows, os.path.join(args.output, "Figures"))
    write_outputs(records, group_rows, roi_rows, stats, args.output, str(args.checkpoint))
    print(f"\n[SCREENING] wrote per-sample / per-group / per-ROI CSVs, figures and report to: {args.output}")


if __name__ == "__main__":
    main()
