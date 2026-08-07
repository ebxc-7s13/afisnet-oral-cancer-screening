#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kfold_classification.py
=======================
Stage 4 of the pipeline accompanying:

    "Quality Controlled Synthetic Augmentation for AI-Enabled Label-free
     Digital Cytology of Oral Cancer Screening"

LEAKAGE-SAFE, REGION-GROUPED K-FOLD CROSS-VALIDATION for the leading models,
reusing the full classification.py pipeline (model registry, training loop,
feature banks, Grad-CAM, deployment checkpoints, benchmarking) unchanged.

WHY THIS SCRIPT EXISTS
----------------------
Multiple single-cell frames can originate from the same physical imaging
region (same ROI, acquired seconds apart). A conventional frame-level split
can therefore place near-identical frames in both the training and the test
set, producing optimistic performance estimates that do not reflect true
generalisation. This script provides the leakage-safe protocol reported in
the paper:

  (a) ALL real images (train + val + test folders) are pooled, and
  (b) StratifiedGroupKFold assigns entire region-groups -- keyed by
      (roi_id, acquisition_date) parsed from the filename -- to a single
      fold, so no imaging region is ever shared between training,
      validation, and test partitions.

Quality-selected synthetic images are added to the TRAINING folds only; the
validation and test folds always contain real images exclusively. Identical
folds are used for every model so comparisons are paired.

MODELS CARRIED FORWARD (manuscript names in parentheses)
--------------------------------------------------------
  CAFNet_Hybrid  (AFiS-Net)      -- proposed dual-branch ConvNeXtV2 + SwinV2
                                    encoder with cross-attention fusion.
  NextViT_Small  (Transformer)   -- best transformer from the 22-model
                                    benchmark; largest augmentation gain.
  CoAtNet0       (Conv-attention)-- best conv-attention hybrid.
  ResNet50       (CNN baseline)  -- standard, well-understood reference.

OUTPUT (same artifact layout as classification.py, one subfolder per model)
  results/kfold/<model>/<experiment>/seed_<seed>/fold_<k>/   per-fold artifacts
  results/kfold/<model>/<experiment>/cv_summary.csv          mean +/- std
  results/kfold/KFold_Comparison.csv                         all models
  results/kfold/KFold_Raw_vs_Synthetic.txt                   paired comparison
  results/kfold/Benchmark/                                   ranking + figures

USAGE
  python kfold_classification.py                       # 4 models, 5 folds, seed 42
  python kfold_classification.py --folds 5 --models CAFNet_Hybrid,NextViT_Small
  python kfold_classification.py --no-synthetic        # Raw experiment only
"""

from __future__ import annotations

import os
import re
import csv
import glob
import argparse
import statistics as st
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Reuse the entire classification framework unchanged (this registers every
# model, including the advanced timm backbones and the CAFNet_Hybrid /
# AFiS-Net encoder).
import classification as C
from sklearn.model_selection import StratifiedGroupKFold


# The four models carried forward (see rationale in the header).
DEFAULT_KFOLD_MODELS = ["CAFNet_Hybrid", "NextViT_Small", "CoAtNet0", "ResNet50"]

# Leakage-safe region identifier from the filename: roi_<id>_<YYYYMMDD>_<HHMMSS>.
_SESSION_RE = re.compile(r"roi_(\d+)_(\d{8})_(\d{6})", re.IGNORECASE)


# =============================================================================
# Data pooling + leakage-safe grouping
# =============================================================================
def region_group(path: str) -> str:
    """Group key that keeps all frames of one physical region together.

    Uses (roi_id, acquisition_date). The ROI index alone is NOT sufficient
    because the same index can be reused on different dates for physically
    different regions (and different classes); adding the acquisition date
    disambiguates them. Any filename that does not match the expected pattern
    falls back to being its own singleton group.
    """
    fn = os.path.basename(path)
    m = _SESSION_RE.search(fn)
    if m:
        return f"roi{m.group(1)}_{m.group(2)}"
    return fn


def pool_real_data(class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    """Pool every real image (train_raw + val + test folders) into one set.

    Deduplicates by basename so an image present in more than one legacy folder
    is counted once. Only the two clinical classes are kept.
    """
    roots = [C.RAW_ROOT, C.VAL_ROOT, C.TEST_ROOT]
    seen: Dict[str, Tuple[str, int]] = {}
    for root in roots:
        try:
            rows, _ = C.scan_flat(root)
        except Exception as exc:
            print(f"[KFOLD] note: could not read {root}: {exc}")
            continue
        for path, label in rows:
            if label not in class_to_idx:
                continue
            base = os.path.basename(path)
            if base not in seen:
                seen[base] = (path, class_to_idx[label])
    samples = sorted(seen.values(), key=lambda t: t[0])
    if not samples:
        raise RuntimeError("No real images found to pool for cross-validation.")
    return samples


def make_grouped_folds(samples: List[Tuple[str, int]], n_folds: int, seed: int
                       ) -> List[Dict[str, list]]:
    """Return a list of per-fold {train, val, test} splits (all group-disjoint).

    For fold i: test = fold i, val = fold (i+1) % k, train = the remaining folds.
    Because StratifiedGroupKFold assigns each region-group to exactly one test
    fold, train / val / test never share a region -> no ROI leakage anywhere.
    """
    paths = [p for p, _ in samples]
    labels = np.array([y for _, y in samples])
    groups = np.array([region_group(p) for p in paths])
    n_groups_per_class = {
        int(c): len(set(groups[labels == c])) for c in np.unique(labels)}
    min_groups = min(n_groups_per_class.values())
    if min_groups < n_folds:
        raise ValueError(
            f"Only {min_groups} region-groups in the smallest class but "
            f"{n_folds} folds requested. Reduce --folds to <= {min_groups}.")

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    test_folds = [np.array(te) for _, te in skf.split(paths, labels, groups)]

    folds: List[Dict[str, list]] = []
    n = len(samples)
    for i in range(n_folds):
        test_idx = set(test_folds[i].tolist())
        val_idx = set(test_folds[(i + 1) % n_folds].tolist())
        train_idx = [j for j in range(n) if j not in test_idx and j not in val_idx]
        pick = lambda ix: [(paths[j], int(labels[j])) for j in ix]
        split = {
            "train": pick(sorted(train_idx)),
            "val": pick(sorted(val_idx)),
            "test": pick(sorted(test_idx)),
        }
        # Assertion: guarantee zero region overlap across the three sets.
        g_tr = {region_group(p) for p, _ in split["train"]}
        g_va = {region_group(p) for p, _ in split["val"]}
        g_te = {region_group(p) for p, _ in split["test"]}
        assert not (g_tr & g_te) and not (g_va & g_te) and not (g_tr & g_va), \
            f"Region leakage detected in fold {i} — this must never happen."
        folds.append(split)
    return folds


# =============================================================================
# Cross-validation aggregation + raw-vs-synthetic report
# =============================================================================
_SUMMARY_METRICS = [
    "accuracy", "balanced_accuracy", "f1_macro", "mcc", "cohen_kappa",
    "roc_auc_macro", "Cancer_recall_sensitivity", "Cancer_precision",
    "specificity_macro", "ece", "brier_score",
]


def _mean_std(values):
    vals = [float(v) for v in values if v is not None and float(v) == float(v)]
    if not vals:
        return float("nan"), float("nan")
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def write_cv_summaries(records: List[dict], cfg) -> "list":
    """Per-model cv_summary.csv + a combined comparison + raw-vs-synthetic txt."""
    by_me = defaultdict(list)
    for r in records:
        by_me[(r["model"], r["experiment"])].append(r)

    combined_rows = []
    for (model, exp), recs in sorted(by_me.items()):
        row = {"model": model, "experiment": exp, "n_folds": len(recs)}
        for m in _SUMMARY_METRICS:
            mu, sd = _mean_std([r.get(m) for r in recs])
            row[f"{m}_mean"] = mu
            row[f"{m}_std"] = sd
        combined_rows.append(row)
        # per-model/experiment summary file
        out_dir = os.path.join(cfg.results_root, model, exp)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "cv_summary.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "mean", "std", "n_folds"])
            for m in _SUMMARY_METRICS:
                mu, sd = _mean_std([r.get(m) for r in recs])
                w.writerow([m, f"{mu:.6f}", f"{sd:.6f}", len(recs)])

    # combined comparison CSV
    if combined_rows:
        keys = list(combined_rows[0].keys())
        with open(os.path.join(cfg.results_root, "KFold_Comparison.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(combined_rows)

    # raw vs raw+synthetic text report
    _write_raw_vs_syn(by_me, cfg)
    return combined_rows


def _write_raw_vs_syn(by_me, cfg):
    models = sorted({m for (m, _e) in by_me})
    L = []
    L.append("=" * 92)
    L.append("K-FOLD (ROI-GROUPED, LEAKAGE-SAFE) — RAW vs RAW+SYNTHETIC")
    L.append("Oral Autofluorescence Microscopy | Normal vs Cancer feature learning")
    L.append("=" * 92)
    L.append("")
    L.append("Splitting: StratifiedGroupKFold on (roi_id, date) — no region shared across folds.")
    L.append("Values are mean +/- std over folds (leakage-safe generalization estimates).")
    L.append("")
    hdr = (f"{'model':16s} {'exp':10s} {'F1':>13} {'MCC':>13} {'AUC':>13} "
           f"{'CancerSens':>11} {'ECE':>7}")
    L.append(hdr); L.append("-" * len(hdr))
    for model in models:
        for exp in ("Raw", "Raw_SelectedSynthetic"):
            if (model, exp) not in by_me:
                continue
            recs = by_me[(model, exp)]
            f1 = _mean_std([r.get("f1_macro") for r in recs])
            mcc = _mean_std([r.get("mcc") for r in recs])
            auc = _mean_std([r.get("roc_auc_macro") for r in recs])
            cs = _mean_std([r.get("Cancer_recall_sensitivity") for r in recs])
            ece = _mean_std([r.get("ece") for r in recs])
            L.append(f"{model:16s} {exp[:10]:10s} "
                     f"{f1[0]:.3f}+/-{f1[1]:.3f} {mcc[0]:.3f}+/-{mcc[1]:.3f} "
                     f"{auc[0]:.3f}+/-{auc[1]:.3f} {cs[0]:11.3f} {ece[0]:7.3f}")
        # delta line
        if (model, "Raw") in by_me and (model, "Raw_SelectedSynthetic") in by_me:
            rF1 = _mean_std([r.get("f1_macro") for r in by_me[(model, "Raw")]])[0]
            sF1 = _mean_std([r.get("f1_macro") for r in by_me[(model, "Raw_SelectedSynthetic")]])[0]
            d = sF1 - rF1
            verd = "synthetic HELPED" if d > 0.005 else ("synthetic HURT" if d < -0.005 else "no change")
            L.append(f"{'':16s} {'-> delta':10s} F1 {rF1:.3f} -> {sF1:.3f}  ({d:+.3f})  {verd}")
        L.append("")
    path = os.path.join(cfg.results_root, "KFold_Raw_vs_Synthetic.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


# =============================================================================
# Main
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Leakage-safe, region-grouped k-fold cross-validation for the "
                    "top oral-cancer models (reuses classification.py).")
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated model names. Default: "
                        + ",".join(DEFAULT_KFOLD_MODELS))
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds (default 5).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument("--results-root", type=str, default=None,
                   help="Output directory (default: results/kfold beside this script).")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    p.add_argument("--no-synthetic", action="store_true",
                   help="Run only the Raw experiment (skip Raw+Synthetic).")
    p.add_argument("--list-models", action="store_true", help="List models and exit.")
    return p


def main():
    args = build_argparser().parse_args()
    if args.list_models:
        print("Available models:", ", ".join(C.MODEL_SPECS.keys()))
        return

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else list(DEFAULT_KFOLD_MODELS))
    unknown = [m for m in models if m not in C.MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Use --list-models.")

    cfg = C.Config()
    cfg.results_root = args.results_root or str(C.PROJECT_ROOT / "results" / "kfold")
    cfg.models = tuple(models)
    cfg.n_folds = int(args.folds)
    cfg.patient_level_split = True
    if args.epochs:
        cfg.epochs = args.epochs
    os.makedirs(cfg.results_root, exist_ok=True)

    class_names = list(C.CLASSES)                 # ["normal", "cancer"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    print("=" * 72)
    print("ROI-GROUPED K-FOLD CROSS-VALIDATION (leakage-safe)")
    print("=" * 72)
    print(f"Device={cfg.device} | Models={models} | Folds={args.folds} | Seed={args.seed}")
    print(f"RESULTS_ROOT={cfg.results_root}")

    # Pool all real data and build leakage-safe folds ONCE (identical folds for
    # every model so comparisons are paired).
    real_samples = pool_real_data(class_to_idx)
    counts = np.bincount([y for _, y in real_samples], minlength=len(class_names))
    print(f"Pooled real images: {len(real_samples)}  (normal={counts[0]}, cancer={counts[1]})")
    folds = make_grouped_folds(real_samples, cfg.n_folds, args.seed)
    print(f"Built {len(folds)} group-disjoint folds. "
          f"Fold test sizes: {[len(f['test']) for f in folds]}")

    # Experiments: Raw, and (unless disabled) Raw + selected synthetic in TRAIN.
    experiments = {"Raw": []}
    if not args.no_synthetic and os.path.isdir(C.SELECTED_SYN_ROOT):
        syn = C.get_synthetic(real_samples, C.SELECTED_SYN_ROOT, class_to_idx,
                              use_hash=cfg.use_hash_dedup)
        if syn:
            experiments["Raw_SelectedSynthetic"] = syn
            print(f"Selected synthetic images (added to TRAIN only): {len(syn)}")
    elif not args.no_synthetic:
        print(f"[KFOLD] no synthetic folder at {C.SELECTED_SYN_ROOT}; running Raw only.")

    all_records = []
    y_trues_list = defaultdict(lambda: defaultdict(list))
    y_probs_list = defaultdict(lambda: defaultdict(list))

    for exp_name, syn_train in experiments.items():
        for model_name in models:
            for k, split in enumerate(folds):
                tag = f"{model_name}/{exp_name}/fold{k}"
                # Resilient: one failing (model, fold) is logged and skipped so
                # the rest of the cross-validation still completes.
                try:
                    record, cm, macro_tpr, y_true, y_pred, y_prob = C.run_one(
                        model_name, exp_name, args.seed, split, syn_train,
                        class_names, cfg, fold=k)
                except Exception as exc:
                    import traceback
                    print(f"[SKIP] {tag} failed: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    if cfg.device == "cuda":
                        C.torch.cuda.empty_cache()
                    continue
                record["fold"] = k
                all_records.append(record)
                y_trues_list[exp_name][model_name].append(y_true)
                y_probs_list[exp_name][model_name].append(y_prob)
                print(f"[OK] {tag}: acc={record.get('accuracy', float('nan')):.3f} "
                      f"f1={record.get('f1_macro', float('nan')):.3f} "
                      f"mcc={record.get('mcc', float('nan')):.3f}")

    if not all_records:
        print("[KFOLD] no successful runs; nothing to summarise.")
        return

    # Cross-validation summaries (mean +/- std over folds) + raw-vs-synthetic.
    write_cv_summaries(all_records, cfg)

    # Full benchmarking (ranking, figures, publication-style report) over the
    # seed_/fold_ folder layout produced above.
    if getattr(cfg, "run_benchmark", True):
        try:
            C.run_full_benchmark(all_records, cfg, class_names,
                                 y_trues_list, y_probs_list)
        except Exception as exc:
            import traceback
            print(f"[BENCHMARK] skipped: {exc}")
            traceback.print_exc()

    print(f"\n[KFOLD] complete. Results in: {cfg.results_root}")


if __name__ == "__main__":
    main()
