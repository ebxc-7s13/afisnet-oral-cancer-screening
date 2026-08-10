#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
classification.py
=================
Stage 3 of the pipeline accompanying:

    "Quality Controlled Synthetic Augmentation for AI-Enabled Label-free
     Digital Cytology of Oral Cancer Screening"

Deep-learning classification and synthetic-augmentation experiments on
single-cell confocal autofluorescence images (normal vs cancer).

WHAT THIS SCRIPT DOES
---------------------
* Registers 22 ImageNet-pretrained backbones: 6 torchvision baselines plus
  16 timm-based encoders (CNN, transformer, and conv-attention hybrids),
  including the dual-branch convolution-transformer architecture referred to
  as AFiS-Net in the manuscript (registry name: ``AFiSNet``).
* Fine-tunes every selected model under two experimental conditions:
  ``Raw`` (real images only) and ``Raw_SelectedSynthetic`` (real images plus
  the quality-selected synthetic images produced by gan_quality_selection.py,
  added to the TRAINING set only).
* Repeats each experiment over three random seeds (42, 100, 123) and reports
  mean +/- standard deviation.
* Evaluates accuracy, balanced accuracy, per-class sensitivity/specificity,
  macro F1, MCC, Cohen's kappa, macro/micro ROC-AUC, Brier score, and
  calibration (ECE/MCE) with validation-only temperature scaling and a
  validation-only Youden-J cancer decision threshold.
* Produces Grad-CAM++ and cell-focused attribution maps, hard-example
  analyses, 256-D deployment embedding banks with diagonal-Mahalanobis
  OOD references (used later by screening_analysis.py), paired statistical
  tests (t-test / Wilcoxon) for Raw vs Raw+Synthetic, McNemar tests between
  models, a soft-voting ensemble, and an automatic benchmarking report.

This module is intentionally a single self-contained file; it is imported by
kfold_classification.py (leakage-safe cross-validation) and by
screening_analysis.py (independent screening cohort analysis), which reuse
its model registry, preprocessing, training loop, and checkpoint format.

DATA LAYOUT (defaults; override via CLI flags or environment variables)
----------------------------------------------------------------------
    data/train/{normal,cancer}/*.png     real training images
    data/val/{normal,cancer}/*.png       real validation images
    data/test/{normal,cancer}/*.png      real held-out test images
    outputs/selected_synthetic/{normal,cancer}/*.png
                                         quality-selected synthetic images
Results are written to ``results/classification/<model>/<experiment>/...``.

USAGE
-----
    python classification.py                       # 6 torchvision baselines
    python classification.py --models all          # all 22 registered models
    python classification.py --models AFiSNet,NextViT_Small
    python classification.py --list-models
"""

import os
import re
import json
import random
import shutil
import time
import warnings
import hashlib
import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import update_bn

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.models import (
    resnet18, ResNet18_Weights, resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b1, EfficientNet_B1_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    densenet121, DenseNet121_Weights,
)

try:
    from torch.hub import load_state_dict_from_url
except Exception:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix, roc_auc_score,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
# Dataset roots. Each may be overridden by an environment variable or by the
# corresponding command-line flag (see main()); defaults are relative to the
# repository so the pipeline runs without machine-specific paths.
RAW_ROOT = os.environ.get("AFIS_TRAIN_ROOT", str(PROJECT_ROOT / "data" / "train"))
VAL_ROOT = os.environ.get("AFIS_VAL_ROOT", str(PROJECT_ROOT / "data" / "val"))
TEST_ROOT = os.environ.get("AFIS_TEST_ROOT", str(PROJECT_ROOT / "data" / "test"))
# Produced by gan_quality_selection.py: contains ONLY the quality-selected,
# class-consistent synthetic subset (never the full generated pool).
SELECTED_SYN_ROOT = os.environ.get(
    "AFIS_SELECTED_SYN_ROOT", str(PROJECT_ROOT / "outputs" / "selected_synthetic"))
# The full, unfiltered generated pool (optional baseline only; not the default
# training input because it contains unfiltered images).
ALL_SYN_ROOT = os.environ.get(
    "AFIS_ALL_SYN_ROOT", str(PROJECT_ROOT / "outputs" / "gan" / "final_generated"))

SEEDS = (42, 100, 123)
# Classifier training deliberately uses only the two clinical anchor classes.
# The smoker (suspected) cohort is held out entirely and analysed later, as an
# unseen group, by screening_analysis.py.
CLASSES = ("normal", "cancer")
CANON = {"normal": "Normal", "cancer": "Cancer", "smoke": "Smoke", "premalignant": "Premalignant"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PRETRAINED_URLS = {
    "ResNet18":       "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "ResNet50":       "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "EfficientNetB0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
    "EfficientNetB1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
    "EfficientNetB2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
    "DenseNet121":    "https://download.pytorch.org/models/densenet121-a639ec97.pth",
}

# img_size added per model for native resolution
MODEL_SPECS = {
    "ResNet18":       dict(ctor=resnet18,       weights=ResNet18_Weights.IMAGENET1K_V1,       head="fc",       in_feat=512,  cam="resnet",   img_size=224),
    "ResNet50":       dict(ctor=resnet50,       weights=ResNet50_Weights.IMAGENET1K_V1,       head="fc",       in_feat=2048, cam="resnet",   img_size=224),
    "EfficientNetB0": dict(ctor=efficientnet_b0, weights=EfficientNet_B0_Weights.IMAGENET1K_V1, head="effnet",  in_feat=1280, cam="features", img_size=224),
    "EfficientNetB1": dict(ctor=efficientnet_b1, weights=EfficientNet_B1_Weights.IMAGENET1K_V1, head="effnet",  in_feat=1280, cam="features", img_size=240),
    "EfficientNetB2": dict(ctor=efficientnet_b2, weights=EfficientNet_B2_Weights.IMAGENET1K_V1, head="effnet",  in_feat=1408, cam="features", img_size=260),
    "DenseNet121":    dict(ctor=densenet121,    weights=DenseNet121_Weights.IMAGENET1K_V1,    head="densenet", in_feat=1024, cam="features", img_size=224),
}
DEFAULT_MODELS = list(MODEL_SPECS.keys())
EXPERIMENTS = ("Raw", "Raw_SelectedSynthetic")
REAL_RE = re.compile(r"roi_(\d+)_(\d{8})_(\d{6})", re.IGNORECASE)

# =============================================================================
# ADVANCED ARCHITECTURES (merged, formerly advanced_architectures.py)
# -----------------------------------------------------------------------------
# ADDITIVE state-of-the-art backbone library + flagship hybrid encoder. This is
# the fully self-contained, inlined version of the former companion module: no
# external import is required. If `timm` is unavailable the registration is a
# no-op and the six torchvision baselines behave exactly as before (zero
# regression). Every model here satisfies the exact backbone contract this
# module expects: `backbone(image) -> [B, native_feature_dim]` (a flat pooled vector),
# with native_feature_dim fixed (ADVANCED_OUT_DIM) so FeatureClassifier's strict
# check and the run_one dimension guard hold deterministically. Feature widths
# are read from timm feature_info at construction (nothing hardcoded/guessed).
# =============================================================================
from typing import Dict, List, Optional

try:
    import timm
    _HAS_TIMM = True
except Exception:  # pragma: no cover
    timm = None
    _HAS_TIMM = False


# ---- Pooling ----------------------------------------------------------------
class GeMPool2d(nn.Module):
    """Generalized-Mean pooling: a learnable interpolation between average- and
    max-pooling. Well suited to fluorescence microscopy where the diagnostic
    signal is spatially sparse (bright nuclei/membranes on dark background):
    p -> 1 behaves like average pooling, p -> inf like max pooling.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True) -> None:
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * float(p))
        else:
            self.register_buffer("p", torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,H,W] -> [B,C]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.flatten(1)


class AttentionPool2d(nn.Module):
    """Single-query spatial attention pooling (a learnable alternative to GAP)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,H,W] -> [B,C]
        weights = torch.softmax(self.score(x).flatten(2), dim=-1)  # [B,1,HW]
        values = x.flatten(2)                                      # [B,C,HW]
        return (values * weights).sum(-1)                          # [B,C]


# ---- Multi-scale Feature Pyramid fusion -------------------------------------
class FPNFusion(nn.Module):
    """Lightweight top-down Feature Pyramid over selected backbone stages.

    Lateral 1x1 convs bring every stage to a common width; a top-down pathway
    injects deep semantics into the finer maps; 3x3 smoothing removes aliasing.
    The finest fused map is exposed as the Grad-CAM target (best localisation).
    """

    def __init__(self, channels: List[int], fpn_dim: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in channels])
        self.smooth = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1)
                                     for _ in channels])
        self.fpn_dim = fpn_dim

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # feats: shallow -> deep (largest -> smallest spatial)
        lat = [conv(f) for conv, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 2, -1, -1):
            lat[i] = lat[i] + F.interpolate(
                lat[i + 1], size=lat[i].shape[-2:], mode="nearest")
        return [conv(l) for conv, l in zip(self.smooth, lat)]

    @property
    def cam_conv(self) -> nn.Module:
        # Finest-resolution fused level -> sharpest, most interpretable CAM.
        return self.smooth[0]


# ---- Generic multi-scale timm backbone --------------------------------------
class MultiScaleTimmBackbone(nn.Module):
    """timm `features_only` encoder + FPN fusion + GeM pooling -> fixed vector.

    Returns a flat ``[B, out_dim]`` descriptor so it drops straight into
    FeatureClassifier (which then projects out_dim -> 256 -> L2-norm).
    """

    def __init__(self, timm_name: str, pretrained: bool, out_dim: int = 1024,
                 n_levels: int = 3, fpn_dim: int = 256) -> None:
        super().__init__()
        if not _HAS_TIMM:
            raise RuntimeError("timm is required for MultiScaleTimmBackbone")
        self.body = timm.create_model(timm_name, pretrained=pretrained,
                                      features_only=True)
        channels = self.body.feature_info.channels()
        self.n_levels = max(1, min(n_levels, len(channels)))
        sel = channels[-self.n_levels:]
        self.fpn = FPNFusion(sel, fpn_dim)
        self.pools = nn.ModuleList([GeMPool2d() for _ in sel])
        self.head = nn.Sequential(
            nn.Linear(fpn_dim * self.n_levels, out_dim), nn.GELU())
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)[-self.n_levels:]
        fused = self.fpn(feats)
        pooled = torch.cat([p(f) for p, f in zip(self.pools, fused)], dim=1)
        return self.head(pooled)

    def gradcam_target(self) -> nn.Module:
        return self.fpn.cam_conv


# ---- Flagship hybrid encoder ------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """The CNN multi-scale descriptor (query) attends to the transformer's
    spatial context tokens (key/value). This conditions local fluorescence
    texture on global tissue context rather than merely concatenating the two.
    """

    def __init__(self, desc_dim: int, ctx_channels: int, attn_dim: int = 384,
                 heads: int = 4) -> None:
        super().__init__()
        self.ctx_proj = nn.Conv2d(ctx_channels, attn_dim, 1)
        self.q_proj = nn.Linear(desc_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)
        self.attn_dim = attn_dim

    def forward(self, desc: torch.Tensor, ctx_map: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx_proj(ctx_map).flatten(2).transpose(1, 2)  # [B, HW, d]
        q = self.q_proj(desc).unsqueeze(1)                       # [B, 1, d]
        out, _ = self.attn(q, ctx, ctx)                          # [B, 1, d]
        return self.norm(out.squeeze(1))                         # [B, d]


class FlagshipHybrid(nn.Module):
    """AFiSNet: ConvNeXt V2 (local texture / morphology) + Swin V2 (global
    context) fused by cross-attention, pooled with GeM, into a fixed embedding.

    Rationale: ConvNeXt V2's FCMAE self-supervised pretraining yields strong,
    transferable local texture features (nuclei, cytoplasm, membranes); Swin V2
    supplies hierarchical global context; cross-attention lets the local
    descriptor query that context. This targets biological feature separation
    while staying practical (~convnextv2_nano + swinv2_cr_tiny, well-pretrained).
    """

    def __init__(self, pretrained: bool, out_dim: int = 1024,
                 cnn_name: str = "convnextv2_nano",
                 vit_name: str = "swinv2_cr_tiny_ns_224",
                 n_levels: int = 3, fpn_dim: int = 256, attn_dim: int = 384) -> None:
        super().__init__()
        if not _HAS_TIMM:
            raise RuntimeError("timm is required for FlagshipHybrid")
        # Local-texture branch (multi-scale).
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained,
                                     features_only=True)
        cnn_ch = self.cnn.feature_info.channels()
        self.n_levels = max(1, min(n_levels, len(cnn_ch)))
        self.fpn = FPNFusion(cnn_ch[-self.n_levels:], fpn_dim)
        self.pools = nn.ModuleList([GeMPool2d() for _ in range(self.n_levels)])
        desc_dim = fpn_dim * self.n_levels
        # Global-context branch (deepest stage only).
        self.vit = timm.create_model(vit_name, pretrained=pretrained,
                                     features_only=True)
        ctx_ch = self.vit.feature_info.channels()[-1]
        self.fusion = CrossAttentionFusion(desc_dim, ctx_ch, attn_dim)
        # Joint projection to the fixed encoder dimension.
        self.head = nn.Sequential(
            nn.LayerNorm(desc_dim + attn_dim),
            nn.Linear(desc_dim + attn_dim, out_dim), nn.GELU())
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)[-self.n_levels:]
        fused = self.fpn(feats)
        desc = torch.cat([p(f) for p, f in zip(self.pools, fused)], dim=1)
        ctx_map = self.vit(x)[-1]
        glob = self.fusion(desc, ctx_map)
        return self.head(torch.cat([desc, glob], dim=1))

    def gradcam_target(self) -> nn.Module:
        # Grad-CAM targets the CNN texture branch — the interpretable pathway.
        return self.fpn.cam_conv


# ---- Registry ---------------------------------------------------------------
# Curated for a small biomedical microscopy regime: parameter-efficient variants
# with strong transfer. All entries verified constructible in timm 1.0.28.
_MULTISCALE: Dict[str, dict] = {
    # CNN — local fluorescence texture / cellular morphology
    "ConvNeXtV2_Nano":     dict(timm="convnextv2_nano",         img=224),
    "ConvNeXtV2_Tiny":     dict(timm="convnextv2_tiny",         img=224),
    "EfficientNetV2_S":    dict(timm="tf_efficientnetv2_s",     img=224),
    "InceptionNeXt_Tiny":  dict(timm="inception_next_tiny",     img=224),
    "Res2Net50":           dict(timm="res2net50_26w_4s",        img=224),
    "ResNeSt50":           dict(timm="resnest50d",              img=224),
    # Transformer — global context
    "SwinV2_Tiny":         dict(timm="swinv2_cr_tiny_ns_224",      img=224),
    "DeiT3_Small":         dict(timm="deit3_small_patch16_224", img=224),
    "NextViT_Small":       dict(timm="nextvit_small",           img=224),
    # Hybrid conv+attention — strong balance for limited data
    "CoAtNet0":            dict(timm="coatnet_0_rw_224",        img=224),
    "MaxViT_Nano":         dict(timm="maxvit_rmlp_nano_rw_256", img=256),
    "EdgeNeXt_Small":      dict(timm="edgenext_small",          img=224),
    "EfficientViT_B0":     dict(timm="efficientvit_b0",         img=224),
    "EfficientFormerV2_S1":dict(timm="efficientformerv2_s1",    img=224),
    "MobileViT_XS":        dict(timm="mobilevit_xs",            img=256),
}
_FLAGSHIP: Dict[str, dict] = {
    "AFiSNet": dict(cnn="convnextv2_nano", vit="swinv2_cr_tiny_ns_224", img=224),
}

# Standard encoder output dimension (native_feature_dim) for every advanced
# model. Fixed so the static in_feat check below is deterministic.
ADVANCED_OUT_DIM = 1024


def _make_multiscale_ctor(timm_name: str, out_dim: int):
    def ctor(weights=None):
        pretrained = weights is not None
        return MultiScaleTimmBackbone(timm_name, pretrained=pretrained, out_dim=out_dim)
    return ctor


def _make_flagship_ctor(cnn: str, vit: str, out_dim: int):
    def ctor(weights=None):
        pretrained = weights is not None
        return FlagshipHybrid(pretrained=pretrained, out_dim=out_dim,
                              cnn_name=cnn, vit_name=vit)
    return ctor


def register_advanced_models(model_specs: dict) -> List[str]:
    """Add the advanced backbones + flagship to MODEL_SPECS in place.

    No-op (returns []) if timm is unavailable, so the base pipeline is never
    affected. Existing entries are never overwritten. Returns the list of names
    that were newly registered. These are NOT added to DEFAULT_MODELS, so the
    default training run is byte-for-byte unchanged; select them explicitly via
    Config.models, e.g. models=("AFiSNet", "ConvNeXtV2_Nano").
    """
    if not _HAS_TIMM:
        return []
    added: List[str] = []
    for name, meta in _MULTISCALE.items():
        if name in model_specs:
            continue
        model_specs[name] = dict(
            ctor=_make_multiscale_ctor(meta["timm"], ADVANCED_OUT_DIM),
            weights=True,           # truthy sentinel -> ctor(pretrained=True)
            head="advanced",        # backbone already returns [B, in_feat]
            in_feat=ADVANCED_OUT_DIM,
            cam="advanced",
            img_size=meta["img"],
        )
        added.append(name)
    for name, meta in _FLAGSHIP.items():
        if name in model_specs:
            continue
        model_specs[name] = dict(
            ctor=_make_flagship_ctor(meta["cnn"], meta["vit"], ADVANCED_OUT_DIM),
            weights=True,
            head="advanced",
            in_feat=ADVANCED_OUT_DIM,
            cam="advanced",
            img_size=meta["img"],
        )
        added.append(name)
    return added


def advanced_cam_target(backbone: nn.Module) -> Optional[nn.Module]:
    """Return the Grad-CAM target for an advanced backbone, else None.

    cam_target_layer calls this first; any module exposing ``gradcam_target()``
    (our wrappers) returns a spatial conv suitable for Grad-CAM++.
    """
    if hasattr(backbone, "gradcam_target"):
        try:
            return backbone.gradcam_target()
        except Exception:
            return None
    return None


def verify_advanced_models(names: Optional[List[str]] = None,
                           device: str = "cpu", verbose: bool = True) -> Dict[str, str]:
    """Build each advanced model with pretrained=False and check the contract:
    forward runs, output dim == ADVANCED_OUT_DIM, and a Grad-CAM target exists.

    Returns {name: "OK" | "FAIL: <reason>"}. One-off smoke test; no downloads.
    """
    results: Dict[str, str] = {}
    if not _HAS_TIMM:
        return {"_timm": "FAIL: timm not installed"}
    specs: dict = {}
    register_advanced_models(specs)
    todo = names or list(specs.keys())
    for name in todo:
        try:
            meta = specs[name]
            model = meta["ctor"](weights=None).to(device).eval()
            sz = meta["img_size"]
            with torch.no_grad():
                out = model(torch.zeros(2, 3, sz, sz, device=device))
            assert out.shape == (2, ADVANCED_OUT_DIM), f"bad shape {tuple(out.shape)}"
            assert advanced_cam_target(model) is not None, "no gradcam target"
            results[name] = "OK"
        except Exception as exc:  # pragma: no cover
            results[name] = f"FAIL: {type(exc).__name__}: {exc}"
        if verbose:
            print(f"  [{ 'OK ' if results[name]=='OK' else 'XX ' }] {name:22s} {results[name]}")
    return results


# Register the advanced models AFTER DEFAULT_MODELS is frozen above, so the
# default training run is byte-for-byte unchanged (new encoders are opt-in via
# Config.models). advanced_cam_target is always defined regardless of timm.
ADVANCED_MODELS = []
try:
    ADVANCED_MODELS = register_advanced_models(MODEL_SPECS)
    if ADVANCED_MODELS:
        print(f"[ARCH] Registered {len(ADVANCED_MODELS)} advanced backbones "
              f"(opt-in via Config.models): {', '.join(ADVANCED_MODELS)}")
    else:
        print("[ARCH] timm not available; using the six torchvision baselines only.")
except Exception as _adv_exc:  # pragma: no cover
    print(f"[ARCH] Advanced backbone registration skipped ({_adv_exc}); "
          f"using the six torchvision baselines only.")

# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    img_size: int = 224               # fallback; overridden per-model via MODEL_SPECS
    epochs: int = 50
    batch_size: int = 32
    accum_steps: int = 1            # gradient accumulation (1 = disabled)
    lr: float = 1e-4
    lr_backbone_factor: float = 0.1   # backbone LR multiplier when discriminative LR enabled
    use_discriminative_lr: bool = True
    # Explicit L1/L2 penalties are added to the training loss below.  Keep
    # AdamW's decoupled weight decay off to avoid applying L2 twice.
    weight_decay: float = 0.0
    l1_lambda: float = 1e-8
    l2_lambda: float = 1e-6
    regularize_classifier_only: bool = False
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    freeze_backbone: bool = False
    freeze_backbone_epochs: int = 2  # warm-start head, then fine-tune encoder
    early_stop_patience: int = 10
    num_workers: int = 4
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15
    min_class_samples: int = 10       # guard: abort if any class has fewer images
    # Repeated crops of the same imaging region (ROI) must never cross folds.
    # Keep every ROI in one fold by default; set False only for an explicitly
    # labelled exploratory per-image split.
    patient_level_split: bool = True
    n_gradcam: int = 12             # increased from 6 for richer explainability
    cell_thresh: float = 0.08       # None = adaptive percentile-based
    use_adaptive_cell_thresh: bool = False
    tta_n: int = 0                  # test-time augmentation views (0 = disabled)
    use_swa: bool = False
    swa_start: int = 25
    use_hash_dedup: bool = True     # content-hash synthetic deduplication
    use_foreground_crop: bool = True
    include_all_synthetic_baseline: bool = False
    # Cross-entropy remains primary.  The optional supervised-contrastive term
    # directly shapes the saved 256-D deployment embedding.
    supervised_contrastive_weight: float = 0.10
    contrastive_temperature: float = 0.10
    deployment_embedding_dim: int = 256
    embedding_dropout: float = 0.10
    save_feature_banks: bool = True
    save_embedding_visualizations: bool = True
    use_umap_if_available: bool = True
    ood_validation_quantile: float = 0.95
    calibrate_temperature: bool = True
    calibration_max_iter: int = 50
    n_folds: int = 0                # 0: repeated holdout; >=2: grouped CV
    allow_random_init: bool = False # medical experiments must opt in explicitly
    results_root: str = str(PROJECT_ROOT / "results" / "classification")
    models: tuple = field(default_factory=lambda: tuple(DEFAULT_MODELS))
    # ---- Automatic benchmarking (additive; runs after all models finish) ----
    run_benchmark: bool = True          # generate the comparison report at the end
    benchmark_measure_speed: bool = True  # time inference / FPS / peak GPU memory
    benchmark_infer_batch: int = 16     # batch size used for the speed measurement
    benchmark_infer_iters: int = 20     # timed iterations (after warm-up)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def _worker_init(_):
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws); random.seed(ws)


# =============================================================================
# DATA
# =============================================================================
def scan_flat(root):
    """Return (list[(path, class_name)], class_names) from root/<class>/*.img."""
    root = str(root).strip().strip('"').strip("'")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root does not exist:\n  {root}")
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    rows = []
    for c in classes:
        cdir = os.path.join(root, c)
        for fn in sorted(os.listdir(cdir)):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                rows.append((os.path.join(cdir, fn), c.lower()))
    if not rows:
        raise RuntimeError(f"No images found under flat class folders in:\n  {root}")
    return rows, [c.lower() for c in classes]


def _file_hash(path):
    """MD5 hash of file contents for robust deduplication."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def get_synthetic(raw_samples, syn_root, class_to_idx, use_hash=False):
    """GAN images = files in syn_root not present in raw (by class+basename or MD5)."""
    if use_hash:
        raw_keys = set((c, _file_hash(p)) for p, c in raw_samples)
        key_fn = _file_hash
    else:
        raw_keys = set((c, os.path.basename(p)) for p, c in raw_samples)
        key_fn = os.path.basename
    syn_rows, _ = scan_flat(syn_root)
    result = [(p, class_to_idx[c]) for p, c in syn_rows
              if c in class_to_idx and (class_to_idx[c], key_fn(p)) not in raw_keys]
    n_filtered = len(syn_rows) - len(result)
    if n_filtered:
        print(f"[INFO] Synthetic deduplication filtered {n_filtered} images (hash={use_hash})")
    return result


def extract_patient_id(path):
    """Extract the ROI identifier encoded in the filename.

    The supplied names do not contain a verified patient ID.  This identifier
    is therefore used to stop repeated crops from one ROI being split across
    folds; it must not be described as a patient-level validation split.
    """
    fn = os.path.basename(path)
    m = REAL_RE.match(fn)
    if m:
        return m.group(1)
    return fn

def stratified_split(samples, class_to_idx, cfg, seed):
    """Reproducible stratified train/val/test split (per-image, class-balanced)."""
    paths = [p for p, _ in samples]
    labels = np.array([c if isinstance(c, int) else class_to_idx[c] for _, c in samples])
    counts = np.bincount(labels, minlength=len(class_to_idx))
    if counts.min() < cfg.min_class_samples:
        raise ValueError(
            f"Class '{list(class_to_idx.keys())[int(counts.argmin())]}' has only "
            f"{counts.min()} samples (minimum required: {cfg.min_class_samples}). "
            f"Aborting to prevent degenerate splits."
        )
    idx = np.arange(len(samples))
    tr, te = train_test_split(idx, test_size=cfg.split_test, stratify=labels, random_state=seed)
    val_rel = cfg.split_val / (cfg.split_train + cfg.split_val)
    tr, va = train_test_split(tr, test_size=val_rel, stratify=labels[tr], random_state=seed)
    pick = lambda ix: [(paths[i], int(labels[i])) for i in ix]
    return pick(tr), pick(va), pick(te)


def stratified_split_patient(samples, class_to_idx, cfg, seed):
    """ROI-grouped split: prevents the same labelled ROI crossing folds."""
    # ROI numbers are reused by different class folders, so include the class
    # in the group key instead of incorrectly merging unrelated labels.
    patient_map = {}
    for p, c in samples:
        pid = f"{c}:{extract_patient_id(p)}"
        patient_map.setdefault(pid, []).append((p, c))
    patients = list(patient_map.keys())
    # Patient-level label = majority class
    patient_labels = []
    for pid in patients:
        cls_list = [c if isinstance(c, int) else class_to_idx[c] for _, c in patient_map[pid]]
        patient_labels.append(max(set(cls_list), key=cls_list.count))
    patient_labels = np.array(patient_labels)
    group_counts = np.bincount(patient_labels, minlength=len(class_to_idx))
    if group_counts.min() < 3:
        raise ValueError(
            "At least three independent ROI groups per class are required for "
            "a grouped train/validation/test split. Add verified group metadata "
            "or use an explicitly exploratory per-image split."
        )
    idx = np.arange(len(patients))
    tr, te = train_test_split(idx, test_size=cfg.split_test, stratify=patient_labels, random_state=seed)
    val_rel = cfg.split_val / (cfg.split_train + cfg.split_val)
    tr, va = train_test_split(tr, test_size=val_rel, stratify=patient_labels[tr], random_state=seed)
    def pick(ix):
        result = []
        for i in ix:
            pid = patients[i]
            # Ensure proper mapping for result list
            result.extend([(p, c if isinstance(c, int) else class_to_idx[c]) for p, c in patient_map[pid]])
        return result
    return pick(tr), pick(va), pick(te)


def grouped_cross_validation_splits(samples, class_to_idx, cfg, seed):
    """Return grouped train/validation/test folds without ROI leakage."""
    if cfg.n_folds < 2:
        raise ValueError("n_folds must be at least 2 for grouped cross-validation.")
    groups = np.array([f"{c}:{extract_patient_id(p)}" for p, c in samples])
    labels = np.array([c if isinstance(c, int) else class_to_idx[c] for _, c in samples])
    unique_groups, first = np.unique(groups, return_index=True)
    group_labels = labels[first]
    if np.bincount(group_labels, minlength=len(class_to_idx)).min() < cfg.n_folds + 2:
        raise ValueError(
            "Each class needs at least n_folds + 2 independent ROI groups so every fold "
            "also retains stratified train and validation groups."
        )
    splitter = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_val, test) in enumerate(splitter.split(np.zeros(len(samples)), labels, groups), start=1):
        train_val_groups = np.unique(groups[train_val])
        train_val_labels = np.array([labels[np.flatnonzero(groups == g)[0]] for g in train_val_groups])
        train_groups, val_groups = train_test_split(
            train_val_groups, test_size=cfg.split_val / (cfg.split_train + cfg.split_val),
            stratify=train_val_labels, random_state=seed + fold,
        )
        train_set, val_set = set(train_groups), set(val_groups)
        pick = lambda indices: [(samples[i][0], int(labels[i])) for i in indices]
        folds.append({"fold": fold,
                      "train": pick([i for i in train_val if groups[i] in train_set]),
                      "val": pick([i for i in train_val if groups[i] in val_set]),
                      "test": pick(test)})
    return folds

class ImageListDataset(Dataset):
    def __init__(self, samples, transform, default_size=224):
        self.samples = list(samples)
        self.transform = transform
        self.default_size = default_size
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception as exc:
            # Corruption-safe fallback: return blank tensor so training does not crash
            print(f"[WARN] Corrupted image {path}: {exc}; returning blank tensor.")
            img = torch.zeros(3, self.default_size, self.default_size)
        return img, y


class GaussianNoise:
    """Additive Gaussian noise for photon-shot-noise simulation in microscopy."""
    def __init__(self, sigma=0.005):
        self.sigma = sigma
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.sigma


class ForegroundCrop:
    """Crop the fluorescent foreground while preserving a small context band.

    The source images have a large, near-black surround.  Resizing the full
    square makes that surround a high-capacity shortcut and also exaggerates
    the 128 px versus 256 px domain difference between GAN and real images.
    This conservative crop is applied identically during train, validation,
    test, and Grad-CAM generation.
    """
    def __init__(self, padding: float = 0.08, min_coverage: float = 0.01):
        self.padding = padding
        self.min_coverage = min_coverage

    def __call__(self, img):
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        signal = arr.max(axis=2)
        if signal.size == 0 or signal.max() < 8:
            return img

        # Otsu on the maximum fluorescence channel separates dark background
        # from tissue without using any class-specific information.
        hist = np.bincount(signal.ravel(), minlength=256).astype(np.float64)
        weight = np.cumsum(hist)
        mean = np.cumsum(hist * np.arange(256))
        total = weight[-1]
        between = np.zeros(256, dtype=np.float64)
        valid = (weight > 0) & (weight < total)
        between[valid] = (
            (mean[valid] * total - mean[-1] * weight[valid]) ** 2
            / (weight[valid] * (total - weight[valid]))
        )
        threshold = int(np.argmax(between))
        mask = signal > max(6, threshold)
        coverage = float(mask.mean())
        if coverage < self.min_coverage or coverage > 0.92:
            return img

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return img
        h, w = signal.shape
        pad = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * self.padding))
        left, right = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
        top, bottom = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
        if right - left < 8 or bottom - top < 8:
            return img
        return img.crop((left, top, right, bottom))


def build_transforms(img_size, cfg=None):
    """Build train/eval transforms with consistent foreground handling."""
    crop = ForegroundCrop() if cfg is None or cfg.use_foreground_crop else None
    train_list = []
    if crop is not None:
        train_list.append(crop)
    train_list.extend([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.RandomRotation(15),                     # increased from 10
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), interpolation=InterpolationMode.BICUBIC),
        # NO photometric (colour/brightness/contrast) jitter. Absolute channel
        # intensities carry the NAD(P)H / FAD balance that provides the
        # diagnostic contrast between normal and cancer cells, so perturbing
        # them would corrupt the label. This matches the identical exclusion
        # applied to the GAN augmentation pipeline in gan_training.py.
    ])
    # Elastic deformation if available (torchvision >= 0.13)
    try:
        train_list.append(T.ElasticTransform(alpha=15.0, sigma=3.0))
    except Exception:
        pass
    train_list.extend([
        T.ToTensor(),
        GaussianNoise(sigma=0.005),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_tf = T.Compose(train_list)
    eval_list = []
    if crop is not None:
        eval_list.append(crop)
    eval_list.extend([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose(eval_list)
    return train_tf, eval_tf

def make_loader(ds, cfg, shuffle, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
                      worker_init_fn=_worker_init, generator=g,
                      persistent_workers=(cfg.num_workers > 0), drop_last=False)


def class_weights_from(samples, n, device):
    counts = np.bincount([y for _, y in samples], minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)

# =============================================================================
# MODEL FACTORY
# =============================================================================
class FeatureClassifier(nn.Module):
    """ImageNet encoder with a compact, L2-normalized deployment embedding."""
    def __init__(self, backbone, native_feature_dim, embedding_dim, num_classes, model_name, cam_kind,
                 dropout=0.10):
        super().__init__()
        self.backbone = backbone
        self.native_feature_dim = int(native_feature_dim)
        self.feature_dim = int(embedding_dim)
        # Do not expand a ResNet-18's native 512-D feature simply to conform
        # to another backbone. Larger encoders use 1024 -> 512 -> 256;
        # ResNet-18 uses its natural 512 -> 256 path.
        width_1 = min(1024, self.native_feature_dim)
        width_2 = min(512, width_1)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.native_feature_dim),
            nn.Linear(self.native_feature_dim, width_1), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(width_1), nn.Linear(width_1, width_2), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(width_2), nn.Linear(width_2, self.feature_dim),
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.model_name = model_name
        self._cam = cam_kind

    def extract_native_features(self, image):
        features = self.backbone(image)
        if features.ndim != 2 or features.shape[1] != self.native_feature_dim:
            raise RuntimeError(
                f"{self.model_name} returned {tuple(features.shape)}; expected "
                f"[batch, {self.native_feature_dim}] native backbone features."
            )
        return features

    def extract_features(self, image):
        """Return the 256-D, unit-norm deployment embedding."""
        return F.normalize(self.projector(self.extract_native_features(image)), dim=1, eps=1e-8)

    def forward_features(self, image):
        return self.extract_features(image)

    def forward(self, image, return_features=False):
        features = self.extract_features(image)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def build_model(name, num_classes, device, allow_random_init=False, pretrained=True, embedding_dim=256,
                embedding_dropout=0.10):
    spec = MODEL_SPECS[name]
    pretrained_ok = bool(pretrained)
    if not pretrained:
        model = spec["ctor"](weights=None)
    else:
        try:
            model = spec["ctor"](weights=spec["weights"])
        except Exception as e1:
            print(f"[WARN] {name}: weights API failed ({e1}); trying explicit URL.")
            try:
                model = spec["ctor"](weights=None)
                sd = load_state_dict_from_url(PRETRAINED_URLS[name], progress=True)
                model.load_state_dict(sd)
            except Exception as e2:
                if not allow_random_init:
                    raise RuntimeError(
                        f"{name} ImageNet weights are unavailable. Refusing random initialization "
                        "for this medical feature-learning experiment; cache/download weights or "
                        "set Config.allow_random_init=True explicitly."
                    ) from e2
                print(f"[WARN] {name}: URL download failed ({e2}); explicit random-init fallback.")
                model = spec["ctor"](weights=None); pretrained_ok = False
    if spec["head"] == "fc":
        model.fc = nn.Identity()
    elif spec["head"] == "effnet":
        model.classifier = nn.Identity()
    elif spec["head"] == "densenet":
        model.classifier = nn.Identity()
    elif spec["head"] == "advanced":
        # The advanced_architectures wrapper already returns a flat
        # [B, in_feat] descriptor (multi-scale FPN + GeM pooling), so there is
        # no torchvision-style classification head to strip.
        pass
    model = FeatureClassifier(model, spec["in_feat"], embedding_dim, num_classes, name, spec["cam"],
                              dropout=embedding_dropout)
    if spec["head"] == "advanced":
        # Fail-fast contract check: a dummy forward verifies that the
        # backbone truly emits [B, in_feat] before training/optimizer setup,
        # turning any dimension mismatch into a clear error here instead of a
        # crash mid-epoch. Runs on CPU prior to the .to(device) move below.
        was_training = model.training
        model.eval()
        with torch.no_grad():
            _probe_sz = int(spec.get("img_size", 224))
            _ = model.extract_native_features(torch.zeros(2, 3, _probe_sz, _probe_sz))
        model.train(was_training)
    model._pretrained_ok = pretrained_ok
    return model.to(device)


def load_deployment_model(checkpoint_path, device=None):
    """Load a deployment checkpoint (written by run_one) without ImageNet downloads.

    Returns ``(model, checkpoint)``.  The returned model exposes
    ``model.extract_features(batch)`` and is already in evaluation mode.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint.get("architecture", {})
    backbone = architecture.get("backbone", checkpoint.get("model_name"))
    class_names = checkpoint.get("class_names")
    state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if backbone not in MODEL_SPECS or not class_names or state is None:
        raise ValueError("Unsupported checkpoint: expected a deployment checkpoint "
                         "produced by classification.run_one().")
    model = build_model(backbone, len(class_names), device, allow_random_init=True, pretrained=False,
                        embedding_dim=architecture.get("feature_dim", 256),
                        embedding_dropout=architecture.get("embedding_dropout", 0.10))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def get_classifier_module(model, cam_kind):
    return model.classifier if hasattr(model, "classifier") else None


def freeze_backbone(model, cam_kind):
    """Train only the final layer (feature-extraction mode)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in list(model.projector.parameters()) + list(model.classifier.parameters()):
        p.requires_grad_(True)


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def build_optimizer(model, cfg, spec):
    """Build optimizer with optional discriminative learning rates."""
    if cfg.use_discriminative_lr and not cfg.freeze_backbone:
        head_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("classifier.", "projector.")):
                head_params.append(param)
            else:
                backbone_params.append(param)
        param_groups = [
            {"params": backbone_params, "lr": cfg.lr * cfg.lr_backbone_factor, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        ]
        print(f"[OPT] Discriminative LR: backbone={cfg.lr * cfg.lr_backbone_factor:.2e}, head={cfg.lr:.2e}")
        return torch.optim.AdamW(param_groups)
    else:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)


# =============================================================================
# TRAIN / EVAL
# =============================================================================
def _autocast(cfg):
    return torch.amp.autocast("cuda", enabled=(cfg.amp and cfg.device == "cuda"))


def explicit_l1_l2_penalty(model, cfg):
    """Return coupled L1 and L2 penalties for trainable weight tensors.

    Biases and BatchNorm scale/shift parameters are intentionally excluded.
    The penalties are explicit parts of the loss, rather than relying on
    AdamW's decoupled weight decay, and are logged each epoch for auditability.
    """
    l1 = torch.zeros((), device=cfg.device)
    l2 = torch.zeros((), device=cfg.device)
    for name, param in model.named_parameters():
        if not param.requires_grad or param.ndim < 2:
            continue
        if cfg.regularize_classifier_only and not (name.startswith("classifier.") or name.startswith("projector.") or name.endswith(".fc.weight")):
            continue
        l1 = l1 + param.abs().sum()
        l2 = l2 + param.square().sum()
    l1_term = cfg.l1_lambda * l1
    l2_term = cfg.l2_lambda * l2
    return l1_term + l2_term, l1_term.detach(), l2_term.detach()


def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup = max(1, cfg.warmup_epochs * steps_per_epoch)
    total = max(warmup + 1, cfg.epochs * steps_per_epoch)
    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        cos = 0.5 * (1 + np.cos(np.pi * prog))
        mn = cfg.min_lr / max(cfg.lr, 1e-12)
        return mn + (1 - mn) * cos
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def supervised_contrastive_loss(features, targets, temperature):
    """Stable SupCon loss on native embeddings; returns zero when no pair exists."""
    if features.shape[0] < 2:
        return features.new_zeros(())
    z = F.normalize(features.float(), dim=1)
    logits = z @ z.T / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    positives = targets[:, None].eq(targets[None, :]) & ~eye
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_zeros(())
    log_prob = logits - torch.log((torch.exp(logits) * ~eye).sum(dim=1, keepdim=True).clamp_min(1e-12))
    return -(log_prob * positives).sum(dim=1)[valid].div(positive_count[valid]).mean()


def set_backbone_trainable(model, trainable):
    for p in model.backbone.parameters():
        p.requires_grad_(trainable)

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg):
    model.train(); tot_loss = tot_data_loss = tot_l1 = tot_l2 = tot_contrastive = 0.0; correct = n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        x = x.to(cfg.device, non_blocking=True); y = y.to(cfg.device, non_blocking=True)
        with _autocast(cfg):
            out, features = model(x, return_features=True)
            data_loss = criterion(out, y)
            contrastive = supervised_contrastive_loss(features, y, cfg.contrastive_temperature)
        reg_loss, l1_term, l2_term = explicit_l1_l2_penalty(model, cfg)
        loss = (data_loss + cfg.supervised_contrastive_weight * contrastive + reg_loss) / cfg.accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        bs = y.size(0)
        tot_loss += loss.item() * cfg.accum_steps * bs
        tot_data_loss += data_loss.item() * bs
        tot_l1 += l1_term.item() * bs
        tot_l2 += l2_term.item() * bs
        tot_contrastive += contrastive.item() * bs
        correct += (out.argmax(1) == y).sum().item(); n += bs
    denom = max(1, n)
    return (tot_loss / denom, correct / denom, tot_data_loss / denom,
            tot_l1 / denom, tot_l2 / denom, tot_contrastive / denom)


@torch.no_grad()
def evaluate(model, loader, criterion, cfg, collect=False, tta_n=0, temperature=1.0, cancer_threshold=None):
    model.eval(); tot_loss, correct, n = 0.0, 0, 0
    ys, ps, prob = [], [], []
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True); y = y.to(cfg.device, non_blocking=True)
        with _autocast(cfg):
            out = model(x)
            # Test-Time Augmentation: average over deterministic flips
            if tta_n > 0:
                probs = [torch.softmax(out.float() / max(float(temperature), 1e-6), 1)]
                if tta_n >= 2:
                    probs.append(torch.softmax(model(torch.flip(x, [3])).float() / max(float(temperature), 1e-6), 1))
                if tta_n >= 3:
                    probs.append(torch.softmax(model(torch.flip(x, [2])).float() / max(float(temperature), 1e-6), 1))
                if tta_n >= 4:
                    probs.append(torch.softmax(model(torch.flip(torch.flip(x, [2]), [3])).float() / max(float(temperature), 1e-6), 1))
                p = torch.stack(probs[:tta_n]).mean(0)
            else:
                p = torch.softmax((out.float() / max(float(temperature), 1e-6)), 1)
            loss = criterion(out, y)
        bs = y.size(0); tot_loss += loss.item() * bs
        pred = p.argmax(1)
        if cancer_threshold is not None and p.shape[1] == 2:
            pred = (p[:, 1] >= cancer_threshold).long()
        correct += (pred == y).sum().item(); n += bs
        if collect:
            ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy()); prob.append(p.cpu().numpy())
    vl, va = tot_loss / max(1, n), correct / max(1, n)
    if collect:
        return vl, va, np.concatenate(ys), np.concatenate(ps), np.concatenate(prob)
    return vl, va


@torch.no_grad()
def collect_logits(model, loader, cfg):
    """Collect uncalibrated validation logits for post-hoc temperature scaling."""
    model.eval(); logits, labels = [], []
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        with _autocast(cfg):
            logits.append(model(x).float().cpu())
        labels.append(y.cpu())
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(validation_logits, validation_labels, cfg):
    """Fit a scalar temperature on validation data only; never touch test data."""
    if not cfg.calibrate_temperature or len(validation_labels) < 2:
        return 1.0
    log_temperature = torch.zeros(1, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=cfg.calibration_max_iter)
    def closure():
        optimizer.zero_grad()
        loss = criterion(validation_logits / log_temperature.exp().clamp(0.05, 10.0), validation_labels)
        loss.backward()
        return loss
    try:
        optimizer.step(closure)
        return float(log_temperature.detach().exp().clamp(0.05, 10.0).item())
    except Exception as exc:
        print(f"[WARN] Temperature calibration failed ({exc}); using T=1.")
        return 1.0


# =============================================================================
# METRICS & CALIBRATION
# =============================================================================
def specificity_per_class(cm):
    cm = np.asarray(cm, float); n = cm.shape[0]; tot = cm.sum(); spec = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = tot - tp - fn - fp
        spec[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return spec


def compute_calibration(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    confidences = y_prob.max(1)
    predictions = y_prob.argmax(1)
    accuracies = (predictions == y_true).astype(float)
    ece = 0.0; mce = 0.0
    for i in range(n_bins):
        in_bin = (confidences > i / n_bins) & (confidences <= (i + 1) / n_bins)
        prop = in_bin.mean()
        if prop > 0:
            acc = accuracies[in_bin].mean()
            conf = confidences[in_bin].mean()
            ece += prop * abs(acc - conf)
            mce = max(mce, abs(acc - conf))
    return {"ece": float(ece), "mce": float(mce)}


def compute_optimal_thresholds(y_true, y_prob, class_names):
    """Per-class Youden's J optimal thresholds from validation set."""
    n = len(class_names); labels = list(range(n))
    ytb = label_binarize(y_true, classes=labels)
    if n == 2:
        ytb = np.hstack([1 - ytb, ytb])
    thresholds = np.full(n, 0.5)
    for i in range(n):
        if len(np.unique(ytb[:, i])) < 2:
            continue
        fpr, tpr, thresh = roc_curve(ytb[:, i], y_prob[:, i])
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        thresholds[i] = thresh[best_idx]
    return thresholds


def compute_cancer_threshold(y_true, y_prob):
    """Validation-only Youden-J threshold for the Cancer probability."""
    if y_prob.ndim != 2 or y_prob.shape[1] != 2 or len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    valid = np.isfinite(thresholds)
    if not valid.any():
        return 0.5
    best = np.argmax(np.where(valid, tpr - fpr, -np.inf))
    return float(np.clip(thresholds[best], 0.0, 1.0))


def compute_metrics(y_true, y_pred, y_prob, class_names):
    n = len(class_names); labels = list(range(n))
    p_c, r_c, f_c, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    spec_c = specificity_per_class(cm)
    ytb = label_binarize(y_true, classes=labels)
    if n == 2:
        ytb = np.hstack([1 - ytb, ytb])
    auc_c = [np.nan] * n; auc_macro = auc_micro = np.nan
    try:
        auc_macro = roc_auc_score(ytb, y_prob, average="macro", multi_class="ovr")
        # Micro-AUC: ravel all one-vs-rest predictions; note this is a global metric,
        # less clinically interpretable than macro-AUC but included for completeness.
        auc_micro = roc_auc_score(ytb.ravel(), y_prob.ravel())
        for i in range(n):
            if len(np.unique(ytb[:, i])) > 1:
                auc_c[i] = roc_auc_score(ytb[:, i], y_prob[:, i])
    except Exception as e:
        print(f"[WARN] AUC: {e}")
    cal = compute_calibration(y_true, y_prob)
    brier = float(np.mean((y_prob[:, 1] - y_true) ** 2)) if n == 2 else np.nan
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m, "f1_weighted": f_w,
        "sensitivity_macro": r_m, "specificity_macro": float(np.mean(spec_c)),
        "mcc": matthews_corrcoef(y_true, y_pred), "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "roc_auc_macro": float(auc_macro), "roc_auc_micro": float(auc_micro),
        "ece": cal["ece"], "mce": cal["mce"], "brier_score": brier,
    }
    for i, c in enumerate(class_names):
        cc = CANON.get(c, c)
        m[f"{cc}_precision"] = float(p_c[i]); m[f"{cc}_recall_sensitivity"] = float(r_c[i])
        m[f"{cc}_specificity"] = float(spec_c[i]); m[f"{cc}_f1"] = float(f_c[i])
        m[f"{cc}_roc_auc"] = float(auc_c[i]); m[f"{cc}_support"] = int(sup[i])
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.items()}, cm, ytb

def save_calibration_artifacts(y_true, y_prob, out_dir, n_bins=10):
    """Write reliability data and a publication-quality binary reliability plot."""
    if y_prob.shape[1] != 2:
        return
    cancer_prob = y_prob[:, 1]
    rows = []
    for lo, hi in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(1 / n_bins, 1, n_bins)):
        mask = (cancer_prob >= lo) & ((cancer_prob < hi) if hi < 1 else (cancer_prob <= hi))
        if mask.any():
            rows.append({"bin_lower": lo, "bin_upper": hi, "count": int(mask.sum()),
                         "mean_predicted_probability": float(cancer_prob[mask].mean()),
                         "observed_cancer_frequency": float(y_true[mask].mean())})
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(out_dir, "calibration_bins.csv"), index=False)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(frame["mean_predicted_probability"], frame["observed_cancer_frequency"], "o-", lw=3,
            color="#0072B2", label="Cancer probability")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted cancer probability",
           ylabel="Observed cancer frequency", title="Reliability Diagram")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(out_dir, "reliability_diagram.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# PLOTS  (large publication fonts, 300 DPI)
# =============================================================================
plt.rcParams.update({"font.size": 16, "axes.titlesize": 22, "axes.labelsize": 18,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
                     "figure.autolayout": True})


def plot_curve(history, keys, labels, ylab, title, path):
    e = history["epoch"]; fig, ax = plt.subplots(figsize=(10, 7))
    markers = ("o", "s", "^", "D", "v")
    for i, (k, l) in enumerate(zip(keys, labels)):
        mk = markers[i % len(markers)]
        ax.plot(e, history[k], lw=3, marker=mk, ms=4, label=l)
    ax.set_xlabel("Epoch", fontsize=18, fontweight="bold"); ax.set_ylabel(ylab, fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.tick_params(labelsize=16); ax.legend(fontsize=16); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_confusion(cm, class_names, title, path, normalize=False):
    cm = np.asarray(cm, float)
    if normalize:
        rs = cm.sum(1, keepdims=True); rs[rs == 0] = 1; disp = cm / rs; fmt = ".2f"
    else:
        disp = cm; fmt = ".0f"
    names = [CANON.get(c, c) for c in class_names]
    fig, ax = plt.subplots(figsize=(9, 8)); im = ax.imshow(disp, interpolation="nearest", cmap="Blues")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=14)
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names)
    ax.set_xlabel("Predicted label", fontsize=22, fontweight="bold")
    ax.set_ylabel("True label", fontsize=22, fontweight="bold")
    ax.set_title(title, fontsize=24, fontweight="bold", pad=16)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    thr = disp.max() / 2 if disp.max() > 0 else 0.5
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            ax.text(j, i, format(disp[i, j], fmt), ha="center", va="center",
                    color="white" if disp[i, j] > thr else "black", fontsize=20, fontweight="bold")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_roc(ytb, y_prob, class_names, title, path):
    fig, ax = plt.subplots(figsize=(10, 8)); colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(class_names))))
    grid = np.linspace(0, 1, 200); mean_tpr = np.zeros_like(grid); valid = 0
    for i, c in enumerate(class_names):
        if len(np.unique(ytb[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(ytb[:, i], y_prob[:, i]); a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=3, label=f"{CANON.get(c, c)} (AUC={a:.3f})")
        mean_tpr += np.interp(grid, fpr, tpr); valid += 1
    if valid:
        mean_tpr /= valid; mean_tpr[0] = 0.0
        ax.plot(grid, mean_tpr, color="navy", lw=4, ls="--", label=f"Macro (AUC={auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=2, ls=":")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=22, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=22, fontweight="bold")
    ax.set_title(title, fontsize=24, fontweight="bold", pad=16)
    ax.legend(loc="lower right", fontsize=15); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def plot_mean_roc(roc_store, class_names, title, path):
    """Mean +/- std macro ROC across seeds."""
    grid = np.linspace(0, 1, 200); fig, ax = plt.subplots(figsize=(10, 8))
    tprs = []
    for macro_tpr in roc_store:
        tprs.append(macro_tpr)
    if tprs:
        tprs = np.array(tprs); mean_tpr = tprs.mean(0); std_tpr = tprs.std(0)
        ax.plot(grid, mean_tpr, color="navy", lw=4, label=f"Mean macro (AUC={auc(grid, mean_tpr):.3f})")
        ax.fill_between(grid, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
                        color="navy", alpha=0.2, label="+/- 1 SD")
    ax.plot([0, 1], [0, 1], color="grey", lw=2, ls=":")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=22, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=22, fontweight="bold")
    ax.set_title(title, fontsize=24, fontweight="bold", pad=16)
    ax.legend(loc="lower right", fontsize=15); ax.grid(alpha=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def macro_tpr_on_grid(ytb, y_prob, class_names):
    grid = np.linspace(0, 1, 200); acc = np.zeros_like(grid); v = 0
    for i in range(len(class_names)):
        if len(np.unique(ytb[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(ytb[:, i], y_prob[:, i]); acc += np.interp(grid, fpr, tpr); v += 1
    if v:
        acc /= v; acc[0] = 0.0
    return acc

# =============================================================================
# EXPLAINABILITY  --  Grad-CAM++  +  Cell-Focused Activation Map + Cell-Attn Ratio
# =============================================================================
def cam_target_layer(model, cam_kind):
    # FeatureClassifier keeps the ImageNet module under ``backbone`` while
    # exposing a stable feature API.  CAM must still target its final spatial
    # convolution, never the linear binary head.
    backbone = getattr(model, "backbone", model)
    if cam_kind == "resnet":
        return backbone.layer4[-1]
    if cam_kind == "features":
        return backbone.features[-1]
    if cam_kind == "advanced":
        # advanced_architectures wrappers expose their interpretable spatial
        # conv (the finest FPN level) via gradcam_target(); fall through to the
        # generic last-Conv2d search if that helper is unavailable.
        target = advanced_cam_target(backbone)
        if target is not None:
            return target
    for m in reversed(list(backbone.modules())):
        if isinstance(m, nn.Conv2d):
            return m
    return None


def cam_target_name(model, cam_kind):
    """Qualified module path of the layer Grad-CAM++ actually hooks.

    Resolved by identity lookup against ``cam_target_layer`` -- the same
    function ``run_explainability`` uses -- so the provenance recorded in a
    deployment checkpoint can never drift from the module that was really
    used.  This matters for AFiSNet and the other advanced encoders, whose
    CAM target is the finest fused FPN level (``backbone.fpn.smooth.0``),
    not a torchvision-style ``features``/``layer4`` stage.

    Returns None when the model exposes no spatial convolution to hook.
    """
    target = cam_target_layer(model, cam_kind)
    if target is None:
        return None
    for name, module in model.named_modules():
        if module is target:
            return name
    return type(target).__name__


class GradCAMpp:
    def __init__(self, model, target):
        self.model = model; self.a = None; self.g = None
        self.h1 = target.register_forward_hook(self._f)
        self.h2 = target.register_full_backward_hook(self._b)
    def _f(self, m, i, o): 
        # Clone to explicitly break off from the computation graph view, 
        # avoiding PyTorch in-place view modification errors.
        self.a = o.clone().detach() if isinstance(o, torch.Tensor) else o
    def _b(self, m, gi, go): 
        self.g = go[0].clone().detach() if isinstance(go[0], torch.Tensor) else go[0]
    def remove(self): 
        self.h1.remove(); self.h2.remove()
    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()
        g = self.g; a = self.a
        g2, g3 = g ** 2, g ** 3
        denom = 2 * g2 + a.sum(dim=(2, 3), keepdim=True) * g3
        alpha = g2 / (denom + 1e-8)
        w = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1))
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy()[0], class_idx


def tissue_mask(rgb01, thresh=None, close=11, median=5):
    """Segment fluorescent tissue: luminance threshold + morphological cleanup.
    If thresh is None, uses percentile-based adaptive threshold."""
    lum = rgb01.max(2)
    if thresh is None:
        # Adaptive: background is typically the darkest 15-25% of pixels
        thresh = np.percentile(lum, 20)
    m = Image.fromarray(((lum > thresh) * 255).astype(np.uint8), "L")
    if median >= 3:
        m = m.filter(ImageFilter.MedianFilter(median))
    if close >= 3 and close % 2 == 1:
        m = m.filter(ImageFilter.MaxFilter(close)); m = m.filter(ImageFilter.MinFilter(close))
    return (np.asarray(m).astype(np.float32) / 255.0) > 0.5


def _upsample_cam(cam, H, W):
    t = torch.tensor(cam)[None, None]
    return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()


def save_gradcampp(cam, base, out, title):
    up = _upsample_cam(cam, base.shape[0], base.shape[1])
    fig, ax = plt.subplots(figsize=(7, 7)); ax.imshow(base)
    # Colorblind-friendly viridis instead of jet
    ax.imshow(up, cmap="viridis", alpha=0.45)
    ax.set_title(title, fontsize=18, fontweight="bold"); ax.axis("off")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)


def save_cellfocused(cam, base, mask, out, title):
    """Grad-CAM++ restricted to segmented tissue; black background rejected.
    Returns the Cell-Attention Ratio = fraction of CAM energy inside the tissue."""
    H, W = base.shape[:2]
    up = _upsample_cam(cam, H, W)
    car = float((up * mask).sum() / (up.sum() + 1e-8))
    gray = np.dot(base[..., :3], [0.299, 0.587, 0.114])
    heat = np.ma.masked_where(~mask, up)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    ax.imshow(heat, cmap="viridis", alpha=0.55, vmin=0, vmax=1)  # viridis for accessibility
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.5)
    ax.set_title(title + f"\nCell-Attention Ratio = {car:.2f}", fontsize=16, fontweight="bold")
    ax.axis("off")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return car


def run_explainability(model, samples, class_names, cfg, img_size, gdir, cdir,
                       temperature=1.0, cancer_threshold=0.5):
    os.makedirs(gdir, exist_ok=True); os.makedirs(cdir, exist_ok=True)
    model.eval()
    target = cam_target_layer(model, getattr(model, "_cam", None))
    if target is None:
        return np.nan
    cammer = GradCAMpp(model, target)
    _, eval_tf = build_transforms(img_size, cfg)
    cars = []
    try:
        for k, (path, true_idx) in enumerate(samples, start=1):
            try:
                pil = Image.open(path).convert("RGB")
                x = eval_tf(pil).unsqueeze(0).to(cfg.device)
                # Show exactly the foreground-cropped image seen by the
                # network, not an uncropped 224 px surrogate.
                display_pil = ForegroundCrop()(pil) if cfg.use_foreground_crop else pil
                base = np.asarray(display_pil.resize((img_size, img_size))).astype(np.float32) / 255.0
                with torch.no_grad():
                    probabilities = torch.softmax(model(x).float() / max(float(temperature), 1e-6), 1)
                    pred = int((probabilities[0, 1] >= cancer_threshold).item())
                    conf = float(probabilities[0, pred].item())
                with torch.enable_grad():
                    cam, _ = cammer(x, class_idx=pred)
                ttl = (f"GT:{CANON.get(class_names[true_idx], class_names[true_idx])}  "
                       f"Pred:{CANON.get(class_names[pred], class_names[pred])}  Conf:{conf:.2f}")
                save_gradcampp(cam, base, os.path.join(gdir, f"{k:02d}_gradcampp.png"), "Grad-CAM++\n" + ttl)
                thresh = None if cfg.use_adaptive_cell_thresh else cfg.cell_thresh
                mask = tissue_mask(base, thresh)
                car = save_cellfocused(cam, base, mask, os.path.join(cdir, f"{k:02d}_cellfocused.png"),
                                       "Cell-Focused Grad-CAM++\n" + ttl)
                cars.append(car)
            except Exception as e:
                print(f"[WARN] CAM failed for {path}: {e}")
    finally:
        cammer.remove()
    return float(np.mean(cars)) if cars else np.nan

# =============================================================================
# HARD-EXAMPLE ANALYSIS
# =============================================================================
def save_hard_examples(y_true, y_pred, y_prob, test_paths, class_names, out_dir, n=10):
    """Save the n hardest examples: highest confidence on wrong class or lowest on correct."""
    if len(y_true) == 0:
        return
    correct_prob = y_prob[np.arange(len(y_true)), y_true]
    pred_prob = y_prob[np.arange(len(y_true)), y_pred]
    hardness = pred_prob - correct_prob  # high = very wrong / very confident wrong
    hardest = np.argsort(hardness)[-n:][::-1]
    rows = []
    for idx in hardest:
        rows.append({
            "filename": os.path.basename(test_paths[idx]),
            "true_label": class_names[y_true[idx]],
            "pred_label": class_names[y_pred[idx]],
            "confidence_correct": float(correct_prob[idx]),
            "confidence_predicted": float(pred_prob[idx]),
            "hardness_score": float(hardness[idx]),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "hard_examples.csv"), index=False)


@torch.no_grad()
def collect_embedding_bank(model, loader, cfg, temperature=1.0, cancer_threshold=0.5):
    """Collect exactly the normalized deployment embeddings seen at evaluation."""
    model.eval(); vectors, labels, probabilities, predictions = [], [], [], []
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        features = model.extract_features(x).float()
        probs = torch.softmax(model.classifier(features).float() / max(float(temperature), 1e-6), dim=1)
        pred = (probs[:, 1] >= cancer_threshold).long() if probs.shape[1] == 2 else probs.argmax(1)
        vectors.append(features.cpu())
        labels.append(y.cpu())
        probabilities.append(probs.cpu())
        predictions.append(pred.cpu())
    return {
        "embeddings": torch.cat(vectors).numpy().astype(np.float32),
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "probabilities": torch.cat(probabilities).numpy().astype(np.float32),
        "predictions": torch.cat(predictions).numpy().astype(np.int64),
    }


def build_embedding_reference(features, targets, class_names):
    """Stable diagonal-Gaussian prototypes from real training embeddings only."""
    result = {"normalization": "L2-normalized 256-D deployment embedding", "embedding_dim": int(features.shape[1]),
              "classes": {}}
    for index, name in enumerate(class_names):
        values = features[targets == index].astype(np.float32, copy=False)
        if len(values) == 0:
            raise RuntimeError(f"Cannot create a prototype for {name}: no real training embeddings.")
        result["classes"][name] = {
            "count": int(len(values)),
            "centroid": torch.from_numpy(values.mean(axis=0)),
            # Diagonal covariance is deliberately used as the primary OOD
            # model: a dense 256x256 covariance is rank-deficient/noisy for
            # this few-shot cohort.
            "variance": torch.from_numpy(np.maximum(values.var(axis=0), 1e-8)),
        }
    return result


def mahalanobis_ood_scores(features, reference, class_names):
    """Nearest class diagonal-Mahalanobis distance; larger means less familiar."""
    centroids = np.stack([reference["classes"][name]["centroid"].cpu().numpy() for name in class_names])
    variances = np.stack([reference["classes"][name]["variance"].cpu().numpy() for name in class_names])
    distances = ((features[:, None, :] - centroids[None, :, :]) ** 2 / variances[None, :, :]).sum(axis=2)
    return distances.min(axis=1).astype(np.float32), distances.argmin(axis=1).astype(np.int64)


def save_embedding_bank(bank_name, bank, paths, class_names, out_dir, ood_scores=None, ood_threshold=None):
    """Persist embeddings, labels, filenames, probabilities and OOD scores without inference reruns."""
    bank_dir = os.path.join(out_dir, "FeatureBank", bank_name)
    os.makedirs(bank_dir, exist_ok=True)
    np.save(os.path.join(bank_dir, "embeddings.npy"), bank["embeddings"])
    np.save(os.path.join(bank_dir, "labels.npy"), bank["labels"])
    np.save(os.path.join(bank_dir, "probabilities.npy"), bank["probabilities"])
    table = pd.DataFrame({
        "filename": [os.path.basename(path) for path in paths],
        "path": list(paths),
        "true_label": [class_names[i] for i in bank["labels"]],
        "prediction": [class_names[i] for i in bank["predictions"]],
        "probability_normal": bank["probabilities"][:, 0],
        "probability_cancer": bank["probabilities"][:, 1],
        "predictive_entropy": -(bank["probabilities"] * np.log(np.clip(bank["probabilities"], 1e-8, 1.0))).sum(axis=1),
    })
    if ood_scores is not None:
        table["mahalanobis_ood_score"] = ood_scores
        table["is_ood"] = ood_scores > float(ood_threshold)
    table.to_csv(os.path.join(bank_dir, "filenames.csv"), index=False)


def save_reference_artifacts(reference, train_bank, train_paths, class_names, out_dir, ood_threshold):
    """Write prototypes/covariances and a confirmed-case retrieval bank."""
    reference_dir = os.path.join(out_dir, "EmbeddingReference")
    os.makedirs(reference_dir, exist_ok=True)
    metadata = {"embedding_dim": reference["embedding_dim"], "normalization": reference["normalization"],
                "ood_method": "nearest diagonal Mahalanobis", "ood_threshold": float(ood_threshold), "classes": {}}
    for name in class_names:
        cls = reference["classes"][name]
        centroid, variance = cls["centroid"].cpu().numpy(), cls["variance"].cpu().numpy()
        np.save(os.path.join(reference_dir, f"{name}_centroid.npy"), centroid)
        np.save(os.path.join(reference_dir, f"{name}_variance.npy"), variance)
        # Explicit covariance matrices are exported for downstream tools; the
        # diagonal form is the statistically stable model used by this code.
        np.save(os.path.join(reference_dir, f"{name}_covariance.npy"), np.diag(variance))
        metadata["classes"][name] = {"count": cls["count"], "centroid_file": f"{name}_centroid.npy",
                                      "variance_file": f"{name}_variance.npy", "covariance_file": f"{name}_covariance.npy"}
    with open(os.path.join(reference_dir, "reference_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    save_embedding_bank("train_retrieval", train_bank, train_paths, class_names, out_dir)


def query_nearest_neighbors(query_embedding, bank_embeddings, filenames, k=5):
    """Exact cosine retrieval with no FAISS dependency; inputs must be L2-normalized."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    bank = np.asarray(bank_embeddings, dtype=np.float32)
    if bank.ndim != 2 or query.shape[0] != bank.shape[1]:
        raise ValueError("Query and retrieval bank embedding dimensions do not match.")
    k = max(1, min(int(k), len(bank)))
    query = query / max(np.linalg.norm(query), 1e-8)
    scores = bank @ query
    order = np.argsort(-scores)[:k]
    return [{"index": int(i), "filename": str(filenames[i]), "cosine_similarity": float(scores[i])} for i in order]


def _plot_embedding(coords, labels, class_names, title, output_path):
    colors = ["#0072B2", "#D55E00"]
    fig, ax = plt.subplots(figsize=(9, 8))
    for index, name in enumerate(class_names):
        mask = labels == index
        ax.scatter(coords[mask, 0], coords[mask, 1], s=45, alpha=0.80, color=colors[index % len(colors)], label=CANON.get(name, name))
    ax.set(title=title, xlabel="Component 1", ylabel="Component 2")
    ax.legend(); ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close(fig)


def save_embedding_visualizations(features, labels, class_names, out_dir, cfg):
    """PCA, t-SNE and optional UMAP plots/coordinates for publication analysis."""
    if not cfg.save_embedding_visualizations or len(features) < 4:
        return
    visual_dir = os.path.join(out_dir, "EmbeddingVisualizations")
    os.makedirs(visual_dir, exist_ok=True)
    reducers = {"PCA": PCA(n_components=2, random_state=0)}
    perplexity = min(30, max(2, (len(features) - 1) // 3))
    reducers["tSNE"] = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=0)
    if cfg.use_umap_if_available:
        try:
            import umap
            reducers["UMAP"] = umap.UMAP(n_components=2, n_neighbors=min(15, len(features) - 1), min_dist=0.15, random_state=0)
        except ImportError:
            print("[INFO] umap-learn is unavailable; PCA and t-SNE were still generated.")
    for name, reducer in reducers.items():
        try:
            coords = reducer.fit_transform(features)
            pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": [class_names[i] for i in labels]}).to_csv(
                os.path.join(visual_dir, f"{name.lower()}_coordinates.csv"), index=False)
            _plot_embedding(coords, labels, class_names, f"{name} of 256-D deployment embeddings",
                            os.path.join(visual_dir, f"{name.lower()}.png"))
        except Exception as exc:
            print(f"[WARN] {name} embedding visualization failed: {exc}")

# =============================================================================
# ONE RUN  (model x experiment x seed)
# =============================================================================
def run_one(model_name, exp_name, seed, split, syn_train, class_names, cfg, fold=None):
    set_seed(seed)
    run_parts = [f"seed_{seed}"] + ([] if fold is None else [f"fold_{fold}"])
    out_dir = os.path.join(cfg.results_root, model_name, exp_name, *run_parts)
    os.makedirs(out_dir, exist_ok=True)

    # Use per-model native resolution
    img_size = MODEL_SPECS[model_name].get("img_size", cfg.img_size)
    train_tf, eval_tf = build_transforms(img_size, cfg)

    raw_train_samples = list(split["train"])
    train_samples = list(raw_train_samples)
    if syn_train:
        train_samples = train_samples + list(syn_train)
    val_samples, test_samples = list(split["val"]), list(split["test"])
    test_paths = [p for p, _ in test_samples]

    train_loader = make_loader(ImageListDataset(train_samples, train_tf, default_size=img_size), cfg, True, seed)
    train_feature_loader = make_loader(ImageListDataset(train_samples, eval_tf, default_size=img_size), cfg, False, seed)
    raw_train_feature_loader = make_loader(ImageListDataset(raw_train_samples, eval_tf, default_size=img_size), cfg, False, seed)
    val_loader = make_loader(ImageListDataset(val_samples, eval_tf, default_size=img_size), cfg, False, seed)
    test_loader = make_loader(ImageListDataset(test_samples, eval_tf, default_size=img_size), cfg, False, seed)

    model = build_model(model_name, len(class_names), cfg.device, cfg.allow_random_init,
                        embedding_dim=cfg.deployment_embedding_dim, embedding_dropout=cfg.embedding_dropout)
    if model.native_feature_dim != MODEL_SPECS[model_name]["in_feat"] or model.feature_dim != cfg.deployment_embedding_dim:
        raise RuntimeError("Feature-dimension audit failed: model specification and deployment encoder disagree.")
    if cfg.freeze_backbone or cfg.freeze_backbone_epochs > 0:
        freeze_backbone(model, getattr(model, "_cam", None))
    cw = class_weights_from(train_samples, len(class_names), cfg.device) if cfg.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg, MODEL_SPECS[model_name])
    scheduler = build_scheduler(optimizer, cfg, max(1, math.ceil(len(train_loader) / cfg.accum_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and cfg.device == "cuda"))

    history = {k: [] for k in [
        "epoch", "train_loss", "train_data_loss", "train_l1_penalty",
        "train_l2_penalty", "train_contrastive_loss", "train_acc", "val_loss", "val_acc",
        "val_f1_macro", "lr",
    ]}
    # Macro-F1 is more appropriate than raw accuracy for an imbalanced binary
    # medical task; it prevents selection based only on the larger class.
    best_val, best_state, best_epoch, no_imp = -float("inf"), None, 0, 0
    best_vl = float("inf")

    # Stochastic Weight Averaging (manual)
    swa_state = None
    swa_count = 0

    t0 = time.time()
    for epoch in range(cfg.epochs):
        if not cfg.freeze_backbone and cfg.freeze_backbone_epochs and epoch == cfg.freeze_backbone_epochs:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, cfg, MODEL_SPECS[model_name])
            # Preserve cosine/warmup progress when rebuilding optimizer after
            # the explicit gradual-unfreezing boundary.
            scheduler = build_scheduler(optimizer, cfg, max(1, math.ceil(len(train_loader) / cfg.accum_steps)))
            scheduler.last_epoch = epoch * math.ceil(len(train_loader) / cfg.accum_steps) - 1
            print(f"    [UNFREEZE] backbone enabled at epoch {epoch + 1}")
        tl, ta, train_data_loss, train_l1, train_l2, train_contrastive = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg
        )
        vl, va, y_val_epoch, y_pred_epoch, _ = evaluate(
            model, val_loader, criterion, cfg, collect=True
        )
        _, _, val_f1, _ = precision_recall_fscore_support(
            y_val_epoch, y_pred_epoch, average="macro", zero_division=0
        )
        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1); history["train_loss"].append(tl)
        history["train_data_loss"].append(train_data_loss)
        history["train_l1_penalty"].append(train_l1)
        history["train_l2_penalty"].append(train_l2)
        history["train_contrastive_loss"].append(train_contrastive)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va); history["val_f1_macro"].append(val_f1); history["lr"].append(lr_now)

        # SWA update
        if cfg.use_swa and epoch >= cfg.swa_start:
            if swa_state is None:
                swa_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    swa_state[k] = (swa_state[k] * swa_count + v.detach().cpu()) / (swa_count + 1)
            swa_count += 1

        improved = (val_f1 > best_val) or (val_f1 == best_val and vl < best_vl)
        if improved:
            best_val, best_vl, best_epoch = val_f1, vl, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        print(f"    [{model_name}/{exp_name}/s{seed}] ep {epoch+1:2d}/{cfg.epochs} "
              f"tr {tl:.3f}/{ta:.3f} val {vl:.3f}/{va:.3f} f1 {val_f1:.3f} "
              f"L1 {train_l1:.4f} L2 {train_l2:.4f} best_f1 {best_val:.3f}@{best_epoch}")
        if no_imp >= cfg.early_stop_patience:
            print(f"    early stop @ {epoch+1}"); break
    train_seconds = time.time() - t0

    # Load best state (or SWA state if enabled and available)
    if cfg.use_swa and swa_state is not None:
        model.load_state_dict(swa_state)
        # Averaged BatchNorm statistics are invalid until refreshed on the
        # non-augmented training distribution.
        update_bn(train_feature_loader, model, device=cfg.device)
        print(f"    [SWA] Loaded averaged weights from epochs {cfg.swa_start}+")
    elif best_state is not None:
        model.load_state_dict(best_state)

    total_params, trainable_params = count_params(model)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    plot_curve(history, ["train_acc", "val_acc", "val_f1_macro"], ["Training Accuracy", "Validation Accuracy", "Validation Macro-F1"],
               "Accuracy", f"{model_name} ({exp_name}) seed {seed} — Accuracy",
               os.path.join(out_dir, "accuracy_curve.png"))
    plot_curve(history, ["train_data_loss", "val_loss"], ["Training Data Loss", "Validation Loss"],
               "Loss", f"{model_name} ({exp_name}) seed {seed} — Loss",
               os.path.join(out_dir, "loss_curve.png"))
    plot_curve(history, ["train_l1_penalty", "train_l2_penalty", "train_contrastive_loss"], ["L1 Penalty", "L2 Penalty", "SupCon Loss"],
               "Penalty", f"{model_name} ({exp_name}) seed {seed} - Explicit Regularization",
               os.path.join(out_dir, "regularization_curve.png"))

    # ---- compute optimal thresholds on validation set ----
    val_logits, val_labels = collect_logits(model, val_loader, cfg)
    temperature = fit_temperature(val_logits, val_labels, cfg)
    _, _, y_val_true, y_val_pred, y_val_prob = evaluate(
        model, val_loader, criterion, cfg, collect=True, temperature=temperature
    )
    cancer_threshold = compute_cancer_threshold(y_val_true, y_val_prob)
    optimal_thresholds = np.array([1.0 - cancer_threshold, cancer_threshold], dtype=np.float32)
    np.save(os.path.join(out_dir, "optimal_thresholds.npy"), optimal_thresholds)

    # ---- test evaluation + metrics ----
    _, _, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, cfg, collect=True, tta_n=cfg.tta_n,
        temperature=temperature, cancer_threshold=cancer_threshold
    )
    metrics, cm, ytb = compute_metrics(y_true, y_pred, y_prob, class_names)

    # The reference distribution and retrieval bank use real images only.
    # Synthetic GAN samples are deliberately excluded from these clinical
    # anchors even though they remain valid train-time augmentation.
    train_bank = collect_embedding_bank(model, raw_train_feature_loader, cfg, temperature, cancer_threshold)
    reference = build_embedding_reference(train_bank["embeddings"], train_bank["labels"], class_names)
    val_bank = collect_embedding_bank(model, val_loader, cfg, temperature, cancer_threshold)
    test_bank = collect_embedding_bank(model, test_loader, cfg, temperature, cancer_threshold)
    val_ood_scores, _ = mahalanobis_ood_scores(val_bank["embeddings"], reference, class_names)
    ood_threshold = float(np.quantile(val_ood_scores, cfg.ood_validation_quantile))
    test_ood_scores, test_prototype = mahalanobis_ood_scores(test_bank["embeddings"], reference, class_names)
    metrics["ood_threshold"] = ood_threshold
    metrics["ood_rate_test"] = float(np.mean(test_ood_scores > ood_threshold))
    metrics["prototype_agreement_test"] = float(np.mean(test_prototype == y_pred))
    if cfg.save_feature_banks:
        save_embedding_bank("validation", val_bank, [p for p, _ in val_samples], class_names, out_dir,
                            val_ood_scores, ood_threshold)
        save_embedding_bank("test", test_bank, test_paths, class_names, out_dir, test_ood_scores, ood_threshold)
        save_reference_artifacts(reference, train_bank, [p for p, _ in raw_train_samples], class_names, out_dir, ood_threshold)
    save_embedding_visualizations(np.vstack([val_bank["embeddings"], test_bank["embeddings"]]),
                                  np.concatenate([val_bank["labels"], test_bank["labels"]]), class_names, out_dir, cfg)
    pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_csv(
        os.path.join(out_dir, "metrics.csv"), index=False)
    pd.DataFrame({"Filename": [os.path.basename(p) for p in test_paths],
                  "GroundTruth": [class_names[i] for i in y_true],
                  "Prediction": [class_names[i] for i in y_pred],
                  "Confidence": y_prob.max(1),
                  "Probability_Normal": y_prob[:, 0], "Probability_Cancer": y_prob[:, 1],
                  "Temperature": temperature, "CancerThreshold": cancer_threshold}).to_csv(
                      os.path.join(out_dir, "predictions.csv"), index=False)
    save_calibration_artifacts(y_true, y_prob, out_dir)
    plot_confusion(cm, class_names, f"{model_name} ({exp_name}) s{seed} — Confusion Matrix",
                   os.path.join(out_dir, "confusion_matrix.png"), False)
    plot_confusion(cm, class_names, f"{model_name} ({exp_name}) s{seed} — Normalized CM",
                   os.path.join(out_dir, "normalized_confusion_matrix.png"), True)
    plot_roc(ytb, y_prob, class_names, f"{model_name} ({exp_name}) s{seed} — ROC",
             os.path.join(out_dir, "roc_curve.png"))
    macro_tpr = macro_tpr_on_grid(ytb, y_prob, class_names)

    # Hard-example analysis
    save_hard_examples(y_true, y_pred, y_prob, test_paths, class_names, out_dir, n=10)

    # ---- explainability: include both correct and misclassified samples ----
    sel = []
    # Correctly classified per class
    for c in range(len(class_names)):
        ci = list(np.where((y_true == c) & (y_true == y_pred))[0])[:max(1, cfg.n_gradcam // (2 * len(class_names)))]
        sel += ci
    # Misclassified per class
    for c in range(len(class_names)):
        ci = list(np.where((y_true == c) & (y_true != y_pred))[0])[:max(1, cfg.n_gradcam // (2 * len(class_names)))]
        sel += ci
    if len(sel) < cfg.n_gradcam:
        sel += [i for i in range(len(test_paths)) if i not in sel][:cfg.n_gradcam - len(sel)]
    samples = [(test_paths[i], int(y_true[i])) for i in sel][:cfg.n_gradcam]
    car = run_explainability(model, samples, class_names, cfg, img_size,
                              os.path.join(out_dir, "GradCAM"), os.path.join(out_dir, "CellFocusedCAM"),
                              temperature=temperature, cancer_threshold=cancer_threshold)
    metrics["cell_attention_ratio"] = float(car) if car == car else np.nan
    metrics["temperature"] = temperature
    metrics["cancer_threshold"] = cancer_threshold

    # Save the exact feature distributions and preprocessing contract needed
    # to deploy this encoder without retraining in the later screening phase.
    checkpoint = {
        "schema_version": 3,
        "model_state_dict": model.state_dict(),
        "architecture": {"backbone": model_name, "classifier": "Linear", "feature_dim": model.feature_dim,
                         "native_feature_dim": model.native_feature_dim, "embedding_dropout": cfg.embedding_dropout,
                         "num_classes": len(class_names), "pretrained_imagenet": getattr(model, "_pretrained_ok", True)},
        "class_names": class_names,
        "positive_class": "cancer",
        "preprocessing": {"image_size": img_size, "foreground_crop": cfg.use_foreground_crop,
                            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                            "interpolation": "bicubic", "eval_augmentation": False},
        "decision": {"temperature": temperature, "cancer_threshold": cancer_threshold,
                     "thresholds": optimal_thresholds.tolist(), "selection_metric": "validation_binary_f1"},
        "gradcam": {"method": "Grad-CAM++",
                    "target_layer": cam_target_name(model, getattr(model, "_cam", None)),
                    "cell_focused": True},
        "embedding_reference": {**reference, "ood_threshold": ood_threshold,
                                "ood_validation_quantile": cfg.ood_validation_quantile,
                                "retrieval_bank": "FeatureBank/train_retrieval"},
        "training": {"best_epoch": best_epoch, "best_validation_f1": best_val, "seed": seed,
                     "experiment": exp_name, "optimizer": type(optimizer).__name__,
                     "optimizer_groups": [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups],
                     "scheduler": type(scheduler).__name__, "config": asdict(cfg)},
        "validation_metrics": compute_metrics(y_val_true, y_val_pred, y_val_prob, class_names)[0],
        "test_metrics": metrics, "num_parameters": total_params, "trainable_parameters": trainable_params,
    }
    torch.save(checkpoint, os.path.join(out_dir, "best_model.pt"))
    # Rewrite after explainability/calibration so the tabular artifact matches
    # the deployment checkpoint rather than an earlier partial metric set.
    pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_csv(
        os.path.join(out_dir, "metrics.csv"), index=False)

    record = dict(model=model_name, experiment=exp_name, seed=seed, fold=fold,
                  n_train=len(train_samples), n_val=len(val_samples), n_test=len(test_samples),
                  best_epoch=best_epoch, train_seconds=train_seconds,
                  num_parameters=total_params, trainable_parameters=trainable_params,
                  pretrained_loaded=getattr(model, "_pretrained_ok", True),
                  optimal_thresholds=optimal_thresholds.tolist(),
                  **metrics)
    del model
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return record, cm, macro_tpr, y_true, y_pred, y_prob

# =============================================================================
# AGGREGATION  (mean +/- std across seeds) + pooled CM + reports
# =============================================================================
PER_CLASS_KEYS = ["precision", "recall_sensitivity", "specificity", "f1", "roc_auc"]
OVERALL_KEYS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted", "specificity_macro", "mcc", "cohen_kappa",
                "roc_auc_macro", "roc_auc_micro", "ece", "mce", "brier_score", "cell_attention_ratio"]


def _ms(vals):
    vals = [v for v in vals if v is not None and (isinstance(v, str) is False) and v == v]
    if not vals:
        return (np.nan, np.nan)
    return float(np.mean(vals)), float(np.std(vals))


def aggregate_model_exp(records, cms, tprs, y_trues_list, y_preds_list, y_probs_list,
                        model_name, exp_name, class_names, cfg):
    out_dir = os.path.join(cfg.results_root, model_name, exp_name)
    adir = os.path.join(out_dir, "averaged"); os.makedirs(adir, exist_ok=True)
    keys = OVERALL_KEYS + [f"{CANON.get(c, c)}_{k}" for c in class_names for k in PER_CLASS_KEYS]
    rows = []
    for k in keys:
        mean, std = _ms([r.get(k, np.nan) for r in records])
        rows.append({"metric": k, "mean": mean, "std": std, "mean_std": f"{mean:.4f} +/- {std:.4f}"})
    pd.DataFrame(rows).to_csv(os.path.join(adir, "metrics_mean_std.csv"), index=False)

    # Mean CM (averaged elementwise across seeds)
    mean_cm = np.mean(np.stack(cms, 0), axis=0)
    plot_confusion(mean_cm, class_names, f"{model_name} ({exp_name}) — Mean Confusion Matrix",
                   os.path.join(adir, "confusion_matrix_mean.png"), False)
    plot_confusion(mean_cm, class_names, f"{model_name} ({exp_name}) — Mean Normalized CM",
                   os.path.join(adir, "normalized_confusion_matrix_mean.png"), True)

    # Pooled CM (concatenate all predictions across seeds)
    if y_trues_list:
        yt_pool = np.concatenate(y_trues_list)
        yp_pool = np.concatenate(y_preds_list)
        pool_cm = confusion_matrix(yt_pool, yp_pool, labels=list(range(len(class_names))))
        plot_confusion(pool_cm, class_names, f"{model_name} ({exp_name}) — Pooled Confusion Matrix",
                       os.path.join(adir, "confusion_matrix_pooled.png"), False)
        plot_confusion(pool_cm, class_names, f"{model_name} ({exp_name}) — Pooled Normalized CM",
                       os.path.join(adir, "normalized_confusion_matrix_pooled.png"), True)
        # Save pooled metrics
        if y_probs_list:
            y_pool_prob = np.concatenate(y_probs_list)
            pool_metrics, _, _ = compute_metrics(yt_pool, yp_pool, y_pool_prob, class_names)
            pd.DataFrame([{"metric": k, "value": v} for k, v in pool_metrics.items()]).to_csv(
                os.path.join(adir, "metrics_pooled.csv"), index=False)

    plot_mean_roc(tprs, class_names, f"{model_name} ({exp_name}) — Mean ROC (3 seeds)",
                  os.path.join(adir, "roc_curve_mean.png"))
    write_model_report(records, mean_cm, model_name, exp_name, class_names,
                       os.path.join(out_dir, "classification_report.txt"))
    return {"model": model_name, "experiment": exp_name,
            **{k: _ms([r.get(k, np.nan) for r in records])[0] for k in OVERALL_KEYS},
            **{k + "_std": _ms([r.get(k, np.nan) for r in records])[1] for k in OVERALL_KEYS}}


def write_model_report(records, mean_cm, model_name, exp_name, class_names, path):
    L = ["=" * 78, f"{model_name}  |  {exp_name}  |  mean +/- std over seeds {list(SEEDS)}", "=" * 78, ""]
    L.append("PER-CLASS METRICS")
    L.append(f"{'Class':10s} {'Precision':>16s} {'Recall/Sens':>16s} {'Specificity':>16s} {'F1':>16s} {'ROC-AUC':>16s}")
    for c in class_names:
        cc = CANON.get(c, c); cells = []
        for k in ["precision", "recall_sensitivity", "specificity", "f1", "roc_auc"]:
            m, s = _ms([r.get(f"{cc}_{k}", np.nan) for r in records]); cells.append(f"{m:.3f}+/-{s:.3f}")
        L.append(f"{cc:10s} " + " ".join(f"{x:>16s}" for x in cells))
    L.append("")
    L.append("OVERALL METRICS")
    for k in OVERALL_KEYS:
        m, s = _ms([r.get(k, np.nan) for r in records])
        L.append(f"  {k:22s}: {m:.4f} +/- {s:.4f}")
    L.append("")
    L.append("MEAN CONFUSION MATRIX (rows=true, cols=pred; averaged over seeds)")
    names = [CANON.get(c, c) for c in class_names]
    L.append("            " + " ".join(f"{n:>10s}" for n in names))
    for i, n in enumerate(names):
        L.append(f"    {n:8s} " + " ".join(f"{mean_cm[i, j]:>10.2f}" for j in range(len(names))))
    L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return "\n".join(L)

def write_final_report(all_records, model_reports_text, comp_df, cfg):
    root = cfg.results_root
    pd.DataFrame(all_records).to_csv(os.path.join(root, "master_metrics.csv"), index=False)
    comp_df.to_csv(os.path.join(root, "model_comparison.csv"), index=False)
    rank = comp_df.sort_values(["f1_macro", "roc_auc_macro", "accuracy"], ascending=False).reset_index(drop=True)
    rank.insert(0, "rank", np.arange(1, len(rank) + 1))
    rank.to_csv(os.path.join(root, "model_ranking.csv"), index=False)
    if cfg.n_folds >= 2:
        split_protocol = f"{cfg.n_folds}-fold ROI-grouped cross-validation"
    else:
        split_protocol = "ROI-grouped stratified" if cfg.patient_level_split else "per-image stratified (exploratory)"
    L = ["#" * 80,
         "ORAL AUTOFLUORESCENCE MICROSCOPY — BINARY FEATURE LEARNING (normal/cancer)",
         "ImageNet-pretrained feature encoder with a binary linear training head.",
         f"Seeds {list(SEEDS)}; values are MEAN +/- STD. FIXED TEST/VAL DATASET FOLDERS;",
         "Validation/test are real only; selected GAN images are added to training only.",
         "ROI grouping is not a substitute for a true patient- or session-held-out test set.",
         "Reporting aligned with medical-AI standards (TRIPOD-AI / CLAIM).",
         "#" * 80, ""]
    L.append("SUMMARY TABLE (test-set accuracy / macro-F1 / macro ROC-AUC, mean +/- std):")
    L.append(f"{'Model':16s} {'Experiment':14s} {'Accuracy':>18s} {'Macro-F1':>18s} {'Macro-AUC':>18s}")
    for _, r in comp_df.iterrows():
        L.append(f"{r['model']:16s} {r['experiment']:14s} "
                 f"{r['accuracy']:.4f}+/-{r['accuracy_std']:.4f}   "
                 f"{r['f1_macro']:.4f}+/-{r['f1_macro_std']:.4f}   "
                 f"{r['roc_auc_macro']:.4f}+/-{r['roc_auc_macro_std']:.4f}")
    L.append(""); L.append("=" * 80); L.append("PER-MODEL DETAILED REPORTS"); L.append("=" * 80); L.append("")
    for t in model_reports_text:
        L.append(t); L.append("")
    with open(os.path.join(root, "ClassificationReport.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[final] wrote {root}/ClassificationReport.txt, master_metrics.csv, "
          "model_comparison.csv, model_ranking.csv")


# =============================================================================
# STATISTICAL TESTS
# =============================================================================
def run_statistical_tests(all_records, cfg):
    """Paired tests for Raw versus the selected-synthetic experiment."""
    if not HAS_SCIPY:
        print("[WARN] scipy not installed; skipping statistical tests.")
        return
    results = []
    for model_name in cfg.models:
        for metric in ["accuracy", "f1_macro", "roc_auc_macro"]:
            raw_vals = [r[metric] for r in all_records if r["model"] == model_name and r["experiment"] == "Raw"]
            syn_vals = [r[metric] for r in all_records if r["model"] == model_name and r["experiment"] == "Raw_SelectedSynthetic"]
            if len(raw_vals) == len(syn_vals) and len(raw_vals) >= 2:
                t_stat, p_ttest = stats.ttest_rel(syn_vals, raw_vals)
                try:
                    w_stat, p_wilcoxon = stats.wilcoxon(syn_vals, raw_vals)
                except Exception:
                    w_stat, p_wilcoxon = np.nan, np.nan
                results.append({
                    "model": model_name, "metric": metric,
                    "n_seeds": len(raw_vals),
                    "raw_mean": float(np.mean(raw_vals)), "syn_mean": float(np.mean(syn_vals)),
                    "difference": float(np.mean(syn_vals) - np.mean(raw_vals)),
                    "t_statistic": float(t_stat), "p_value_ttest": float(p_ttest),
                    "w_statistic": float(w_stat), "p_value_wilcoxon": float(p_wilcoxon),
                })
    if results:
        pd.DataFrame(results).to_csv(os.path.join(cfg.results_root, "statistical_tests.csv"), index=False)
        print(f"[stats] Wrote statistical_tests.csv ({len(results)} comparisons)")


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's chi-squared test with continuity correction."""
    b = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
    c = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    d = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    e = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))
    if c + d == 0:
        return 1.0, 0.0
    stat = (abs(c - d) - 1) ** 2 / (c + d)
    if HAS_SCIPY:
        p_value = 1 - stats.chi2.cdf(stat, 1)
    else:
        p_value = np.nan
    return float(p_value), float(stat)


def run_mcnemar_tests(all_records, cfg):
    """McNemar tests between all model pairs within each (experiment, seed)."""
    if not HAS_SCIPY:
        return
    # We need predictions per run. Since we didn't store them in all_records,
    # we load from saved predictions.csv files.
    results = []
    for exp_name in sorted({r["experiment"] for r in all_records}):
        for seed in SEEDS:
            preds, reference = {}, None
            for model_name in cfg.models:
                pcsv = os.path.join(cfg.results_root, model_name, exp_name, f"seed_{seed}", "predictions.csv")
                if os.path.exists(pcsv):
                    df = pd.read_csv(pcsv)
                    required = {"Filename", "GroundTruth", "Prediction"}
                    if not required.issubset(df.columns):
                        continue
                    df = df.sort_values("Filename").reset_index(drop=True)
                    frame = df[["Filename", "GroundTruth"]]
                    if reference is None:
                        reference = frame
                    elif not reference.equals(frame):
                        print(f"[McNemar] Skipping non-aligned {model_name}/{exp_name}/s{seed} predictions.")
                        continue
                    preds[model_name] = np.array([CLASSES.index(str(v).lower()) for v in df["Prediction"]])
            models = list(preds.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    if len(preds[models[i]]) == len(preds[models[j]]):
                        gt = np.array([CLASSES.index(str(v).lower()) for v in reference["GroundTruth"]])
                        p_val, stat = mcnemar_test(
                            gt, preds[models[i]], preds[models[j]],
                        )
                        results.append({
                            "experiment": exp_name, "seed": seed,
                            "model_a": models[i], "model_b": models[j],
                            "mcnemar_statistic": stat, "p_value": p_val,
                        })
    if results:
        pd.DataFrame(results).to_csv(os.path.join(cfg.results_root, "mcnemar_tests.csv"), index=False)
        print(f"[stats] Wrote mcnemar_tests.csv ({len(results)} comparisons)")


# =============================================================================
# ENSEMBLE (soft-voting across models per experiment/seed)
# =============================================================================
def build_ensemble_predictions(cfg, class_names):
    """For each (experiment, seed), ensemble all available models by soft-voting."""
    ensemble_records = []
    for exp_name in EXPERIMENTS:
        for seed in SEEDS:
            all_probs, all_thresholds, reference = [], [], None
            for model_name in cfg.models:
                pcsv = os.path.join(cfg.results_root, model_name, exp_name, f"seed_{seed}", "predictions.csv")
                if os.path.exists(pcsv):
                    df = pd.read_csv(pcsv)
                    required = {"Filename", "GroundTruth", "Probability_Normal", "Probability_Cancer", "CancerThreshold"}
                    if not required.issubset(df.columns):
                        print(f"[ENSEMBLE] Skipping outdated predictions: {pcsv}")
                        continue
                    df = df.sort_values("Filename").reset_index(drop=True)
                    if reference is None:
                        reference = df[["Filename", "GroundTruth"]]
                    elif not reference.equals(df[["Filename", "GroundTruth"]]):
                        print(f"[ENSEMBLE] Skipping {model_name}: its test rows do not align with the reference split.")
                        continue
                    all_probs.append(df[["Probability_Normal", "Probability_Cancer"]].to_numpy(float))
                    all_thresholds.append(float(df["CancerThreshold"].iloc[0]))
            if len(all_probs) < 2 or reference is None:
                continue
            y_true = np.array([class_names.index(str(x).lower()) for x in reference["GroundTruth"]])
            y_prob = np.mean(np.stack(all_probs), axis=0)
            threshold = float(np.mean(all_thresholds))
            y_pred = (y_prob[:, 1] >= threshold).astype(int)
            metrics, cm, ytb = compute_metrics(y_true, y_pred, y_prob, class_names)
            metrics["cancer_threshold"] = threshold
            out_dir = os.path.join(cfg.results_root, "SoftVotingEnsemble", exp_name, f"seed_{seed}")
            os.makedirs(out_dir, exist_ok=True)
            reference.assign(Prediction=[class_names[i] for i in y_pred],
                             Probability_Normal=y_prob[:, 0], Probability_Cancer=y_prob[:, 1],
                             CancerThreshold=threshold).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
            pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_csv(
                os.path.join(out_dir, "metrics.csv"), index=False)
            plot_confusion(cm, class_names, f"Soft-voting ensemble ({exp_name}) s{seed}",
                           os.path.join(out_dir, "confusion_matrix.png"))
            plot_roc(ytb, y_prob, class_names, f"Soft-voting ensemble ({exp_name}) s{seed} ROC",
                     os.path.join(out_dir, "roc_curve.png"))
            save_calibration_artifacts(y_true, y_prob, out_dir)
            ensemble_records.append(dict(model="SoftVotingEnsemble", experiment=exp_name, seed=seed,
                                         n_models=len(all_probs), **metrics))
    if ensemble_records:
        print(f"[ENSEMBLE] Wrote {len(ensemble_records)} soft-voting evaluations.")
    return ensemble_records

# =============================================================================
# MAIN
# =============================================================================
# =============================================================================
# AUTOMATIC MODEL BENCHMARKING, RANKING & PUBLICATION REPORTING
# -----------------------------------------------------------------------------
# Runs automatically at the end of main() after every selected model has
# trained. It is purely additive: it READS the artifacts the pipeline already
# produced (per-seed records in memory; saved test feature banks; deployment
# checkpoints) and WRITES a new self-contained ``Benchmark/`` folder. It never
# modifies any existing output, and it is wrapped so a benchmarking error can
# never fail an otherwise-successful training run.
#
# Scientific ranking (the objective is a strong feature ENCODER for future
# screening, not accuracy alone): the overall score is a weighted blend of
# classification performance, feature-space quality, calibration, robustness and
# computational efficiency (weights in BENCHMARK_WEIGHTS below).
# =============================================================================
BENCHMARK_CLASSIFICATION_KEYS = [
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
    "f1_macro", "f1_weighted", "sensitivity_macro", "specificity_macro",
    "mcc", "cohen_kappa", "roc_auc_macro", "roc_auc_micro",
]
BENCHMARK_CALIBRATION_KEYS = ["ece", "mce", "brier_score"]
BENCHMARK_FEATURE_KEYS = [
    "silhouette", "davies_bouldin", "calinski_harabasz", "intra_class_distance",
    "inter_class_distance", "fisher_ratio", "embedding_compactness",
    "embedding_separation",
]
# Weighted scientific score (sums to 1.0). Feature quality is weighted highly
# because the learned embedding is the deliverable for future screening.
BENCHMARK_WEIGHTS = {
    "classification": 0.35,
    "feature_quality": 0.25,
    "calibration": 0.15,
    "robustness": 0.15,
    "efficiency": 0.10,
}


def _safe_mean(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def _safe_std(values):
    arr = np.array([v for v in values if v is not None], dtype=float)
    return float(np.nanstd(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def _minmax_norm(series, higher_better=True):
    """Direction-aware min-max normalisation to [0, 1]; NaN -> neutral 0.5."""
    v = pd.to_numeric(series, errors="coerce").astype(float)
    finite = v[np.isfinite(v)]
    if finite.empty or finite.max() <= finite.min():
        return pd.Series(0.5, index=series.index)
    norm = (v - finite.min()) / (finite.max() - finite.min())
    norm = norm if higher_better else (1.0 - norm)
    return norm.fillna(0.5)


def benchmark_pr_auc(model, exp, y_trues_list, y_probs_list):
    """Mean binary PR-AUC (average precision) across seeds, if probs available."""
    if not y_trues_list or not y_probs_list:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score
        yt = y_trues_list.get(exp, {}).get(model, [])
        yp = y_probs_list.get(exp, {}).get(model, [])
        scores = []
        for t, p in zip(yt, yp):
            t = np.asarray(t); p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] == 2 and len(np.unique(t)) > 1:
                scores.append(average_precision_score(t, p[:, 1]))
        return float(np.mean(scores)) if scores else float("nan")
    except Exception:
        return float("nan")


def _feature_quality_one(X, y):
    """Feature-space separation metrics for ONE embedding bank (one model space)."""
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score)
    classes = np.unique(y)
    centroids = {int(c): X[y == c].mean(axis=0) for c in classes}
    global_mean = X.mean(axis=0)
    intra = float(np.mean([np.linalg.norm(X[i] - centroids[int(y[i])])
                           for i in range(len(X))]))
    cvecs = list(centroids.values())
    inter_pairs = [np.linalg.norm(cvecs[a] - cvecs[b])
                   for a in range(len(cvecs)) for b in range(a + 1, len(cvecs))]
    inter = float(np.mean(inter_pairs)) if inter_pairs else float("nan")
    within = float(np.mean([((X[y == c] - centroids[int(c)]) ** 2).sum(axis=1).mean()
                            for c in classes]))
    between = float(np.mean([((centroids[int(c)] - global_mean) ** 2).sum()
                             for c in classes]))
    return {
        "silhouette": float(silhouette_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
        "intra_class_distance": intra,
        "inter_class_distance": inter,
        "fisher_ratio": float(between / (within + 1e-12)),
        "embedding_compactness": intra,   # lower is better
        "embedding_separation": inter,    # higher is better
    }


def benchmark_feature_quality(model, exp, cfg, class_names):
    """Feature-space separation metrics from the saved TEST embedding banks.

    run_one saves banks under <model>/<exp>/seed_<seed>[/fold_<fold>]/FeatureBank/
    test/. Each seed/fold is a DIFFERENT trained encoder with its own embedding
    space, so metrics are computed per bank and then AVERAGED (never pooled — the
    spaces are not mutually comparable). This fixes an earlier bug where a fixed
    <model>/<exp>/FeatureBank/test path found nothing and every feature-quality
    score fell back to a neutral 0.5.
    """
    import glob as _glob
    keys = BENCHMARK_FEATURE_KEYS
    out = {k: float("nan") for k in keys}
    emb_files = sorted(_glob.glob(os.path.join(
        cfg.results_root, model, exp, "**", "FeatureBank", "test", "embeddings.npy"),
        recursive=True))
    if not emb_files:
        return out
    per_bank = defaultdict(list)
    for emb_p in emb_files:
        lab_p = emb_p.replace("embeddings.npy", "labels.npy")
        if not os.path.exists(lab_p):
            continue
        try:
            X = np.load(emb_p).astype(np.float64)
            y = np.load(lab_p).astype(np.int64)
            if len(np.unique(y)) < 2 or len(y) < 3:
                continue
            for k, v in _feature_quality_one(X, y).items():
                if v == v:
                    per_bank[k].append(v)
        except Exception as exc:
            print(f"[BENCHMARK] feature quality failed for {emb_p}: {exc}")
    for k in keys:
        if per_bank.get(k):
            out[k] = float(np.mean(per_bank[k]))
    return out


def benchmark_deployment(model, exp, cfg):
    """Model size, validation accuracy, and (optionally) inference speed / mem."""
    out = {"model_size_mb": float("nan"), "val_accuracy": float("nan"),
           "inference_ms_per_image": float("nan"), "fps": float("nan"),
           "peak_gpu_mb": float("nan")}
    # run_one saves checkpoints under <model>/<exp>/seed_<seed>[/fold_<k>]/;
    # use the first checkpoint found (all share the same architecture).
    import glob as _glob
    found = sorted(_glob.glob(os.path.join(
        cfg.results_root, model, exp, "**", "best_model.pt"), recursive=True))
    if not found:
        return out
    ckpt = found[0]
    out["model_size_mb"] = float(os.path.getsize(ckpt) / 1e6)
    try:
        meta = torch.load(ckpt, map_location="cpu")
        out["val_accuracy"] = float(meta.get("validation_metrics", {}).get("accuracy", float("nan")))
    except Exception:
        pass
    if not cfg.benchmark_measure_speed:
        return out
    try:
        net, _ = load_deployment_model(ckpt, cfg.device)
        net.eval()
        sz = MODEL_SPECS[model].get("img_size", cfg.img_size)
        bs = int(cfg.benchmark_infer_batch)
        x = torch.zeros(bs, 3, sz, sz, device=cfg.device)
        with torch.no_grad():
            for _ in range(3):
                net(x)
            if cfg.device == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(int(cfg.benchmark_infer_iters)):
                net(x)
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / max(1, int(cfg.benchmark_infer_iters))
        out["inference_ms_per_image"] = float(dt * 1000.0 / bs)
        out["fps"] = float(bs / dt) if dt > 0 else float("nan")
        if cfg.device == "cuda":
            out["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / 1e6)
        del net
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        print(f"[BENCHMARK] speed measurement failed for {model}/{exp}: {exc}")
    return out


def benchmark_collect(all_records, cfg, class_names, y_trues_list, y_probs_list):
    """Build one benchmark row per (model, experiment)."""
    groups = {}
    for r in all_records:
        groups.setdefault((r["model"], r["experiment"]), []).append(r)
    rows = []
    for (model, exp), recs in sorted(groups.items()):
        row = {"model": model, "experiment": exp, "n_seeds": len(recs)}
        for key in BENCHMARK_CLASSIFICATION_KEYS + BENCHMARK_CALIBRATION_KEYS:
            vals = [r.get(key) for r in recs]
            row[key] = _safe_mean(vals)
            row[key + "_std"] = _safe_std(vals)
        row["pr_auc"] = benchmark_pr_auc(model, exp, y_trues_list, y_probs_list)
        row["num_parameters"] = int(_safe_mean([r.get("num_parameters") for r in recs]) or 0)
        row["trainable_parameters"] = int(_safe_mean([r.get("trainable_parameters") for r in recs]) or 0)
        row["train_seconds"] = _safe_mean([r.get("train_seconds") for r in recs])
        row["best_epoch"] = _safe_mean([r.get("best_epoch") for r in recs])
        row["epoch_seconds"] = (row["train_seconds"] / row["best_epoch"]
                                if row["best_epoch"] and np.isfinite(row["best_epoch"])
                                and row["best_epoch"] > 0 else float("nan"))
        row.update(benchmark_feature_quality(model, exp, cfg, class_names))
        row.update(benchmark_deployment(model, exp, cfg))
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_score_and_rank(df):
    """Add normalised sub-scores + weighted overall score + rank (higher=better)."""
    if df.empty:
        return df
    df = df.copy()
    # Classification sub-score: key accuracy/agreement/discrimination metrics.
    cls = np.mean([
        _minmax_norm(df["accuracy"], True), _minmax_norm(df["balanced_accuracy"], True),
        _minmax_norm(df["f1_macro"], True), _minmax_norm(df["mcc"], True),
        _minmax_norm(df["roc_auc_macro"], True), _minmax_norm(df.get("pr_auc", pd.Series(np.nan, index=df.index)), True),
    ], axis=0)
    # Feature-quality sub-score.
    fq = np.mean([
        _minmax_norm(df["silhouette"], True),
        _minmax_norm(df["calinski_harabasz"], True),
        _minmax_norm(df["fisher_ratio"], True),
        _minmax_norm(df["embedding_separation"], True),
        _minmax_norm(df["davies_bouldin"], False),      # lower better
        _minmax_norm(df["embedding_compactness"], False),  # lower better
    ], axis=0)
    # Calibration sub-score (all lower-better).
    cal = np.mean([
        _minmax_norm(df["ece"], False), _minmax_norm(df["mce"], False),
        _minmax_norm(df["brier_score"], False),
    ], axis=0)
    # Robustness: high balanced accuracy + low seed variance of F1/MCC.
    rob = np.mean([
        _minmax_norm(df["balanced_accuracy"], True),
        _minmax_norm(df.get("f1_macro_std", pd.Series(np.nan, index=df.index)), False),
        _minmax_norm(df.get("mcc_std", pd.Series(np.nan, index=df.index)), False),
    ], axis=0)
    # Efficiency: fewer params, smaller, faster.
    eff = np.mean([
        _minmax_norm(df["num_parameters"], False),
        _minmax_norm(df["model_size_mb"], False),
        _minmax_norm(df["inference_ms_per_image"], False),
    ], axis=0)
    df["score_classification"] = cls
    df["score_feature_quality"] = fq
    df["score_calibration"] = cal
    df["score_robustness"] = rob
    df["score_efficiency"] = eff
    w = BENCHMARK_WEIGHTS
    df["overall_score"] = (w["classification"] * cls + w["feature_quality"] * fq
                           + w["calibration"] * cal + w["robustness"] * rob
                           + w["efficiency"] * eff)
    df = df.sort_values("overall_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _benchmark_bar(df, column, title, ylabel, path, higher_better=True):
    """One publication-quality horizontal bar chart comparing models."""
    if column not in df.columns:
        return
    data = df[["model", "experiment", column]].copy()
    data["label"] = data["model"] + "\n(" + data["experiment"] + ")"
    data = data.dropna(subset=[column])
    if data.empty:
        return
    data = data.sort_values(column, ascending=higher_better)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(data) + 2)))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(data)))
    ax.barh(data["label"], data[column], color=colors, edgecolor="black", linewidth=1.2)
    ax.set_xlabel(ylabel, fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(data[column]):
        ax.text(v, i, f" {v:.3g}", va="center", fontsize=13, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def benchmark_figures(df, fig_dir):
    """Generate the full set of comparison figures (PNG 300 DPI + vector PDF)."""
    os.makedirs(fig_dir, exist_ok=True)
    charts = [
        ("accuracy", "Test Accuracy Comparison", "Accuracy", True),
        ("balanced_accuracy", "Balanced Accuracy Comparison", "Balanced Accuracy", True),
        ("roc_auc_macro", "ROC-AUC Comparison", "ROC-AUC (macro)", True),
        ("pr_auc", "PR-AUC Comparison", "PR-AUC", True),
        ("f1_macro", "F1-score Comparison", "F1 (macro)", True),
        ("mcc", "MCC Comparison", "Matthews Corr. Coef.", True),
        ("ece", "Calibration (ECE) Comparison", "Expected Calibration Error", False),
        ("silhouette", "Feature Quality (Silhouette) Comparison", "Silhouette Score", True),
        ("fisher_ratio", "Feature Separability (Fisher Ratio)", "Fisher Discriminant Ratio", True),
        ("embedding_separation", "Embedding Separation Comparison", "Inter-class Distance", True),
        ("num_parameters", "Parameter Count Comparison", "Parameters", False),
        ("model_size_mb", "Model Size Comparison", "Checkpoint Size (MB)", False),
        ("train_seconds", "Training Time Comparison", "Training Time (s)", False),
        ("inference_ms_per_image", "Inference Time Comparison", "ms / image", False),
        ("fps", "Throughput Comparison", "Frames per Second", True),
        ("peak_gpu_mb", "Peak GPU Memory Comparison", "Peak GPU Memory (MB)", False),
        ("overall_score", "Overall Weighted Score", "Composite Score", True),
    ]
    for col, title, ylab, hb in charts:
        _benchmark_bar(df, col, title, ylab,
                       os.path.join(fig_dir, f"compare_{col}"), higher_better=hb)


def benchmark_write_tables(df, bench_dir):
    """Write the five required CSV summaries."""
    df.to_csv(os.path.join(bench_dir, "Benchmark_Report.csv"), index=False)
    id_cols = ["rank", "model", "experiment", "n_seeds"]
    cls_cols = [c for c in df.columns
                if c in BENCHMARK_CLASSIFICATION_KEYS + BENCHMARK_CALIBRATION_KEYS + ["pr_auc"]]
    df[id_cols + cls_cols].to_csv(
        os.path.join(bench_dir, "Model_Comparison.csv"), index=False)
    feat_cols = [c for c in BENCHMARK_FEATURE_KEYS if c in df.columns]
    df[id_cols + feat_cols].to_csv(
        os.path.join(bench_dir, "Feature_Comparison.csv"), index=False)
    dep_cols = [c for c in ["num_parameters", "trainable_parameters", "model_size_mb",
                            "inference_ms_per_image", "fps", "peak_gpu_mb",
                            "train_seconds", "epoch_seconds"] if c in df.columns]
    df[id_cols + dep_cols].to_csv(
        os.path.join(bench_dir, "Deployment_Comparison.csv"), index=False)
    rank_cols = ["rank", "model", "experiment", "overall_score", "score_classification",
                 "score_feature_quality", "score_calibration", "score_robustness",
                 "score_efficiency"]
    df[[c for c in rank_cols if c in df.columns]].to_csv(
        os.path.join(bench_dir, "Model_Ranking.csv"), index=False)


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}" if np.isfinite(float(v)) else "n/a"
    except Exception:
        return "n/a"


def benchmark_publication_report(df, bench_dir, cfg, class_names):
    """Human-readable, journal-style report incl. the recommended best model."""
    if df.empty:
        return
    best = df.iloc[0]
    lines = []
    lines.append("=" * 78)
    lines.append("AUTOMATIC MODEL BENCHMARK - PUBLICATION SUMMARY")
    lines.append("Oral Autofluorescence Microscopy | Binary Feature Learning (Normal vs Cancer)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Models benchmarked : {len(df)} (model x experiment configurations)")
    lines.append(f"Classes            : {', '.join(class_names)}")
    lines.append(f"Scoring weights    : " + ", ".join(f"{k}={v}" for k, v in BENCHMARK_WEIGHTS.items()))
    lines.append("")
    lines.append("-" * 78)
    lines.append("OVERALL RANKING (weighted scientific score; higher is better)")
    lines.append("-" * 78)
    lines.append(f"{'#':>2}  {'Model':22s} {'Exp':20s} {'Score':>7} {'Acc':>6} "
                 f"{'AUC':>6} {'MCC':>6} {'Silh':>6} {'ECE':>6} {'Params(M)':>9}")
    for _, r in df.iterrows():
        lines.append(
            f"{int(r['rank']):>2}  {str(r['model'])[:22]:22s} {str(r['experiment'])[:20]:20s} "
            f"{_fmt(r['overall_score'],3):>7} {_fmt(r['accuracy'],3):>6} "
            f"{_fmt(r['roc_auc_macro'],3):>6} {_fmt(r['mcc'],3):>6} "
            f"{_fmt(r.get('silhouette'),3):>6} {_fmt(r.get('ece'),3):>6} "
            f"{_fmt(r['num_parameters']/1e6,1):>9}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("RECOMMENDED BEST OVERALL MODEL")
    lines.append("=" * 78)
    lines.append(f"Best Overall Model : {best['model']}  (experiment: {best['experiment']})")
    lines.append("")
    lines.append("Reason:")
    lines.append(f"  Highest weighted score ({_fmt(best['overall_score'],3)}) across classification "
                 f"({_fmt(best['score_classification'],3)}), feature quality "
                 f"({_fmt(best['score_feature_quality'],3)}), calibration "
                 f"({_fmt(best['score_calibration'],3)}), robustness "
                 f"({_fmt(best['score_robustness'],3)}) and efficiency "
                 f"({_fmt(best['score_efficiency'],3)}). Because the deliverable is a transferable "
                 f"encoder, feature-space separability is weighted heavily, not accuracy alone.")
    lines.append("")
    lines.append("Advantages:")
    lines.append(f"  - Test accuracy {_fmt(best['accuracy'],4)}, balanced accuracy "
                 f"{_fmt(best['balanced_accuracy'],4)}, ROC-AUC {_fmt(best['roc_auc_macro'],4)}, "
                 f"MCC {_fmt(best['mcc'],4)}.")
    lines.append(f"  - Feature separability: silhouette {_fmt(best.get('silhouette'),4)}, "
                 f"Fisher ratio {_fmt(best.get('fisher_ratio'),3)}, "
                 f"Calinski-Harabasz {_fmt(best.get('calinski_harabasz'),1)}.")
    lines.append(f"  - Calibration: ECE {_fmt(best.get('ece'),4)}, Brier {_fmt(best.get('brier_score'),4)}.")
    lines.append("")
    lines.append("Disadvantages / limitations:")
    dis = []
    if np.isfinite(best.get("num_parameters", np.nan)) and best["num_parameters"] > 30e6:
        dis.append(f"relatively large ({best['num_parameters']/1e6:.1f}M params) - watch overfitting "
                   f"on the small cohort; rely on GAN augmentation + freeze-warmup.")
    if np.isfinite(best.get("ece", np.nan)) and best["ece"] > 0.1:
        dis.append("calibration (ECE) leaves room for improvement — keep temperature scaling on.")
    if np.isfinite(best.get("inference_ms_per_image", np.nan)) and best["inference_ms_per_image"] > 20:
        dis.append("higher inference latency - consider a lighter encoder for edge screening.")
    if not dis:
        dis.append("no major weakness identified within this cohort; validate on external data.")
    for d in dis:
        lines.append(f"  - {d}")
    lines.append("")
    lines.append("Expected Screening Performance:")
    lines.append(f"  On this held-out test set the encoder separates Normal vs Cancer with "
                 f"ROC-AUC {_fmt(best['roc_auc_macro'],3)} and sensitivity "
                 f"{_fmt(best.get('sensitivity_macro'),3)} / specificity "
                 f"{_fmt(best.get('specificity_macro'),3)}. The L2-normalised 256-D embedding is "
                 f"ready for Mahalanobis OOD scoring, centroid/prototype retrieval and similarity "
                 f"search, so it can extend to Normal-Smoker-Premalignant-Cancer screening without "
                 f"architectural redesign. External, patient-level validation is required before "
                 f"clinical use.")
    lines.append("")
    lines.append("Recommended for Deployment:")
    lines.append(f"  YES - {best['model']} ({best['experiment']}). Deploy the saved checkpoint at "
                 f"{os.path.join(cfg.results_root, str(best['model']), str(best['experiment']), 'best_model.pt')} "
                 f"using load_deployment_model(); the embedding contract is unchanged.")
    lines.append("")
    lines.append("-" * 78)
    lines.append("PER-MODEL SUMMARY")
    lines.append("-" * 78)
    for _, r in df.iterrows():
        lines.append(f"* {r['model']} ({r['experiment']}) - rank {int(r['rank'])}")
        lines.append(f"    params={r['num_parameters']/1e6:.1f}M size={_fmt(r.get('model_size_mb'),1)}MB "
                     f"train={_fmt(r.get('train_seconds'),0)}s infer={_fmt(r.get('inference_ms_per_image'),2)}ms/img "
                     f"fps={_fmt(r.get('fps'),1)}")
        lines.append(f"    acc={_fmt(r['accuracy'],4)} bal_acc={_fmt(r['balanced_accuracy'],4)} "
                     f"auc={_fmt(r['roc_auc_macro'],4)} pr_auc={_fmt(r.get('pr_auc'),4)} "
                     f"mcc={_fmt(r['mcc'],4)} kappa={_fmt(r['cohen_kappa'],4)}")
        lines.append(f"    feature: silhouette={_fmt(r.get('silhouette'),4)} DB={_fmt(r.get('davies_bouldin'),3)} "
                     f"CH={_fmt(r.get('calinski_harabasz'),1)} fisher={_fmt(r.get('fisher_ratio'),3)}")
        lines.append(f"    calibration: ece={_fmt(r.get('ece'),4)} mce={_fmt(r.get('mce'),4)} "
                     f"brier={_fmt(r.get('brier_score'),4)}")
    report = "\n".join(lines)
    with open(os.path.join(bench_dir, "Benchmark_Publication_Report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(bench_dir, "Benchmark_Publication_Report.md"), "w", encoding="utf-8") as fh:
        fh.write("```\n" + report + "\n```\n")
    print("\n" + report)


def run_full_benchmark(all_records, cfg, class_names, y_trues_list=None, y_probs_list=None):
    """Top-level entry: collect -> score/rank -> tables -> figures -> report."""
    bench_dir = os.path.join(cfg.results_root, "Benchmark")
    os.makedirs(bench_dir, exist_ok=True)
    print("\n" + "=" * 72)
    print("AUTOMATIC MODEL BENCHMARKING & COMPARISON")
    print("=" * 72)
    df = benchmark_collect(all_records, cfg, class_names, y_trues_list, y_probs_list)
    if df.empty:
        print("[BENCHMARK] no records to benchmark.")
        return
    df = benchmark_score_and_rank(df)
    benchmark_write_tables(df, bench_dir)
    benchmark_figures(df, os.path.join(bench_dir, "Figures"))
    benchmark_publication_report(df, bench_dir, cfg, class_names)
    print(f"\n[BENCHMARK] wrote report, 5 CSVs and comparison figures to: {bench_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Oral autofluorescence classification & feature-learning framework")
    parser.add_argument("--models", type=str, default=None,
                        help="'all' to run every registered model (6 baselines + advanced), "
                             "or a comma-separated list of names (see --list-models). "
                             "Default: the six torchvision baselines.")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Output directory. Each model writes results_root/<model>/<experiment>/. "
                             "Default: results/classification beside this script.")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds, e.g. 42,100,123. Default: 42,100,123.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="Skip the automatic benchmarking report at the end.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print every available model name and exit.")
    parser.add_argument("--train-root", type=str, default=None,
                        help="Real training images (class folders). Default: data/train")
    parser.add_argument("--val-root", type=str, default=None,
                        help="Real validation images (class folders). Default: data/val")
    parser.add_argument("--test-root", type=str, default=None,
                        help="Real test images (class folders). Default: data/test")
    parser.add_argument("--synthetic-root", type=str, default=None,
                        help="Quality-selected synthetic images (class folders). "
                             "Default: outputs/selected_synthetic")
    args, _ = parser.parse_known_args()

    if args.list_models:
        print(f"Available models ({len(MODEL_SPECS)}):")
        base = [m for m in MODEL_SPECS if m in DEFAULT_MODELS]
        adv = [m for m in MODEL_SPECS if m not in DEFAULT_MODELS]
        print("  torchvision baselines :", ", ".join(base))
        print("  advanced (timm)       :", ", ".join(adv) if adv else "(timm not installed)")
        return

    global RAW_ROOT, VAL_ROOT, TEST_ROOT, SELECTED_SYN_ROOT
    if args.train_root:
        RAW_ROOT = args.train_root
    if args.val_root:
        VAL_ROOT = args.val_root
    if args.test_root:
        TEST_ROOT = args.test_root
    if args.synthetic_root:
        SELECTED_SYN_ROOT = args.synthetic_root

    cfg = Config()
    if args.models:
        if args.models.strip().lower() == "all":
            cfg.models = tuple(MODEL_SPECS.keys())          # every registered model
        else:
            requested = [m.strip() for m in args.models.split(",") if m.strip()]
            unknown = [m for m in requested if m not in MODEL_SPECS]
            if unknown:
                raise SystemExit(
                    f"Unknown model(s): {unknown}\nRun with --list-models to see valid names.")
            cfg.models = tuple(requested)
    if args.results_root:
        cfg.results_root = args.results_root
    if args.epochs:
        cfg.epochs = args.epochs
    if args.no_benchmark:
        cfg.run_benchmark = False
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS

    os.makedirs(cfg.results_root, exist_ok=True)
    print("=" * 72)
    print("ORAL AUTOFLUORESCENCE BINARY FEATURE LEARNING (Normal vs Cancer)")
    print("=" * 72)
    print(f"Device={cfg.device} | Models({len(cfg.models)})={list(cfg.models)} | Seeds={list(seeds)}")
    print(f"RESULTS_ROOT={cfg.results_root}  (per-model subfolders: <model>/<experiment>/)")
    print(f"RAW_TRAIN={RAW_ROOT}\nSELECTED_SYN={SELECTED_SYN_ROOT}")
    print(f"VAL={VAL_ROOT}\nTEST={TEST_ROOT}")

    class_names = list(CLASSES)  # fixed order: normal=0, cancer=1
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # LOAD TRAINING DATA
    discovered_train, _ = scan_flat(RAW_ROOT)
    raw_train_samples = [(path, class_to_idx[label]) for path, label in discovered_train if label in class_names]

    # LOAD VALIDATION DATA
    discovered_val, _ = scan_flat(VAL_ROOT)
    val_samples = [(path, class_to_idx[label]) for path, label in discovered_val if label in class_names]

    # LOAD TEST DATA
    discovered_test, _ = scan_flat(TEST_ROOT)
    test_samples = [(path, class_to_idx[label]) for path, label in discovered_test if label in class_names]

    missing = [name for name in class_names if not any(label == class_to_idx[name] for _, label in raw_train_samples)]
    if missing:
        raise RuntimeError(f"Training requires normal and cancer class folders; missing in train: {missing}")

    synthetic_by_experiment = {"Raw": []}
    
    if os.path.isdir(SELECTED_SYN_ROOT):
        selected_syn = get_synthetic(
            raw_train_samples, SELECTED_SYN_ROOT, class_to_idx, use_hash=cfg.use_hash_dedup
        )
        if selected_syn:
            synthetic_by_experiment["Raw_SelectedSynthetic"] = selected_syn

    all_records = []
    cms, tprs, y_trues_list, y_preds_list, y_probs_list = {}, {}, {}, {}, {}

    for seed in seeds:
        # Construct the static split directly from the folders
        split = {
            "train": raw_train_samples,
            "val": val_samples,
            "test": test_samples
        }
        
        for exp_name, syn_train in synthetic_by_experiment.items():
            for model_name in cfg.models:
                cms.setdefault(exp_name, {}).setdefault(model_name, [])
                tprs.setdefault(exp_name, {}).setdefault(model_name, [])
                y_trues_list.setdefault(exp_name, {}).setdefault(model_name, [])
                y_preds_list.setdefault(exp_name, {}).setdefault(model_name, [])
                y_probs_list.setdefault(exp_name, {}).setdefault(model_name, [])

                # Resilient multi-model runs: a single model failing (e.g. a
                # missing pretrained download or an out-of-memory backbone) is
                # logged and skipped so the remaining models still train and the
                # benchmark still runs. Successful models are unaffected.
                try:
                    record, cm, macro_tpr, y_true, y_pred, y_prob = run_one(
                        model_name, exp_name, seed, split, syn_train, class_names, cfg
                    )
                except Exception as _run_exc:
                    import traceback as _tb
                    print(f"[SKIP] {model_name}/{exp_name}/s{seed} failed: "
                          f"{type(_run_exc).__name__}: {_run_exc}")
                    _tb.print_exc()
                    if cfg.device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                all_records.append(record)
                cms[exp_name][model_name].append(cm)
                tprs[exp_name][model_name].append(macro_tpr)
                y_trues_list[exp_name][model_name].append(y_true)
                y_preds_list[exp_name][model_name].append(y_pred)
                y_probs_list[exp_name][model_name].append(y_prob)

    # Aggregation
    reports_text = []
    model_comps = []
    for exp_name in synthetic_by_experiment.keys():
        for model_name in cfg.models:
            recs = [r for r in all_records if r["model"] == model_name and r["experiment"] == exp_name]
            if not recs:
                continue
            agg = aggregate_model_exp(
                recs, cms[exp_name][model_name], tprs[exp_name][model_name],
                y_trues_list[exp_name][model_name], y_preds_list[exp_name][model_name],
                y_probs_list[exp_name][model_name],
                model_name, exp_name, class_names, cfg
            )
            model_comps.append(agg)
            reports_text.append(write_model_report(
                recs, np.mean(np.stack(cms[exp_name][model_name], 0), axis=0),
                model_name, exp_name, class_names,
                os.path.join(cfg.results_root, model_name, exp_name, "classification_report.txt")
            ))

    if model_comps:
        comp_df = pd.DataFrame(model_comps)
        write_final_report(all_records, reports_text, comp_df, cfg)
        run_statistical_tests(all_records, cfg)
        run_mcnemar_tests(all_records, cfg)
        build_ensemble_predictions(cfg, class_names)

        # Automatic benchmarking / ranking / publication report. Wrapped so a
        # benchmarking error can never fail an otherwise-successful training run
        # (all training artifacts are already saved by this point).
        if getattr(cfg, "run_benchmark", True):
            try:
                run_full_benchmark(all_records, cfg, class_names,
                                   y_trues_list, y_probs_list)
            except Exception as _bench_exc:
                import traceback as _tb
                print(f"[BENCHMARK] skipped due to error: {_bench_exc}")
                _tb.print_exc()

if __name__ == "__main__":
    main()
