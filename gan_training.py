#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gan_training.py
===============
Stage 1 of the pipeline accompanying:

    "Quality Controlled Synthetic Augmentation for AI-Enabled Label-free
     Digital Cytology of Oral Cancer Screening"

Class-conditional StyleGAN2-ADA generator with the proposed Neural Texture
Preservation (NTP) loss, for synthesising single-cell confocal
autofluorescence images of oral exfoliated cells (two classes: cancer,
normal).

WHAT THIS SCRIPT IMPLEMENTS
---------------------------
* A conditional StyleGAN2-ADA recipe for the low-data regime: modulated /
  demodulated convolutions, FIR-smoothed resampling, minibatch standard
  deviation, projection-discriminator class conditioning, generator EMA,
  lazy R1 regularisation, LeCam regularisation, adaptive discriminator
  augmentation (ADA), style mixing, and lazy path-length regularisation.
* The NEURAL TEXTURE PRESERVATION (NTP) loss described in the manuscript:
  five unpaired, distribution-level statistics-matching penalties that pull
  the texture of the generated set toward class-wise EMA targets computed
  from real images --
      (1) VGG-16 Gram-matrix texture matching,
      (2) VGG-16 feature mean/std matching,
      (3) Haar-wavelet high-frequency band-energy matching,
      (4) a one-sided (hinge) total-variation anti-speckle term, and
      (5) a soft-masked fluorescence-photometry statistic
          (coverage / foreground mean / foreground std).
  All descriptors are matched with a scale-invariant relative MSE, and the
  overall NTP coefficient is warm-started and then linearly ramped
  (warm-up 20 epochs, ramp 30 epochs). Setting --no-ntp recovers the pure
  StyleGAN2-ADA baseline used for the ablation in the paper.
* A deterministic per-class hold-out (val_fraction of the training folder,
  by filename hash) used exclusively for KID/FID checkpoint selection, so
  checkpoints are never selected on their own training images. The best
  checkpoint is the one with the lowest held-out KID.
* Native-resolution data handling: the 256x256 source ROIs are RandomCrop'd
  to 224x224 for training (no resampling blur; free translation
  augmentation) and CenterCrop'd for metric computation, with
  label-preserving horizontal/vertical flips and 90-degree rotations.
  Photometric jitter is deliberately excluded so that channel intensities
  retain their diagnostic meaning.
* Milestone monitoring: per-class KID/FID, improved precision/recall, and a
  nearest-neighbour memorisation check.
* Final generation: after training (or via --mode generate), the EMA
  generator synthesises N images per class (default 1000) into class-named
  folders ready for gan_quality_selection.py, plus a zip archive.

DATA LAYOUT (defaults; override via CLI flags)
----------------------------------------------
    data/train/cancer/*.png      real training images, class folders
    data/train/normal/*.png
Outputs: outputs/gan/ (previews, comparisons, final_generated/),
checkpoints/gan/ (latest.pth, generator_best.pth, periodic snapshots),
logs/gan/ (TensorBoard, if available).

USAGE
-----
    python gan_training.py                          # train + generate
    python gan_training.py --no-ntp                 # StyleGAN2-ADA ablation
    python gan_training.py --mode generate --weights checkpoints/gan/generator_best.pth
    python gan_training.py --data data/train --per-class 1000

NOTE: the defaults below are the exact configuration used for the
experiments in the paper (single NVIDIA RTX 4000 Ada, 20 GB); results are
sensitive to these settings, so change them only deliberately.
"""

from __future__ import annotations

import os
# Must be set before the first CUDA context. Expandable segments dramatically
# reduce fragmentation and the reserved high-water mark that pushes Windows/WDDM
# into slow "Shared GPU memory".
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import math
import time
import random
import hashlib
import zipfile
import warnings
import argparse
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autograd, optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

import torchvision.utils as vutils
from PIL import Image

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    _TORCHMETRICS = True
except Exception as _e:  # pragma: no cover
    _TORCHMETRICS = False
    warnings.warn(f"[metrics] torchmetrics unavailable — KID/FID disabled: {_e}")

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD = True
except Exception:  # pragma: no cover
    _TENSORBOARD = False

try:
    import torchvision.models as tvm
    _TORCHVISION_MODELS = True
except Exception:  # pragma: no cover
    _TORCHVISION_MODELS = False

try:
    import pynvml  # type: ignore
    _PYNVML = True
except Exception:  # pragma: no cover
    _PYNVML = False


ACT_GAIN: float = math.sqrt(2.0)
LEAKY_SLOPE: float = 0.2


# =============================================================================
# Module 1 — Configuration
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Immutable single source of truth for every hyperparameter and path."""

    # ---- Data / IO ----------------------------------------------------------
    # The GAN trains on the raw class folders directly. A deterministic per-class
    # hold-out is carved internally (see val_fraction) for KID/FID selection, so
    # no external, potentially-leaky val/test folder is required.
    # All paths are relative to the working directory; override via CLI flags.
    train_data_path: str = os.path.join("data", "train")
    output_path: str = os.path.join("outputs", "gan")
    checkpoint_path: str = os.path.join("checkpoints", "gan")
    log_path: str = os.path.join("logs", "gan")
    val_fraction: float = 0.15          # fraction of each class held out for KID

    # ---- Geometry -----------------------------------------------------------
    # Native pyramid 7 -> 14 -> 28 -> 56 -> 112 -> 224 (224 = 7 * 2^5). The source
    # crops are 256x256; RandomCrop(224) keeps native pixel scale (no resample).
    image_size: int = 224
    base_resolution: int = 7
    source_size: int = 256
    channels: int = 3
    # Fixed label order everywhere (datasets, conditioning, previews, metrics,
    # checkpoints, generation): 0 = cancer, 1 = normal.
    num_classes: int = 2

    # ---- Latent / mapping ---------------------------------------------------
    z_dim: int = 128
    embed_dim_g: int = 128
    w_dim: int = 512
    map_depth: int = 2
    map_lr_mul: float = 0.01

    # ---- Capacity -----------------------------------------------------------
    channel_base: int = 16384
    channel_max: int = 512
    mbstd_group: int = 4

    # ---- Optimisation (RTX 4000 Ada, 20 GB) --------------------------------
    batch_size: int = 16
    lr: float = 0.002
    beta1: float = 0.0
    beta2: float = 0.99
    adam_eps: float = 1e-8
    weight_decay: float = 0.0

    # ---- Regularisers -------------------------------------------------------
    r1_gamma: float = 0.8
    r1_interval: int = 16
    lecam_lambda: float = 0.05
    lecam_decay: float = 0.99
    output_tanh: bool = False           # unbounded StyleGAN2 RGB; clamp only for IO
    style_mixing_prob: float = 0.90     # NOW functional (per-block W+)
    path_length_weight: float = 2.0     # NOW functional (lazy PLR)
    path_length_interval: int = 8
    path_length_decay: float = 0.01
    path_batch_shrink: int = 2
    use_relativistic: bool = False      # optional relativistic (RaGAN) loss; OFF by default

    # ---- Neural Texture Preservation (NTP) ---------------------------------
    # Unpaired, distribution-level texture statistics matching (see header).
    # Overall coefficient is warm-started then linearly ramped (see ntp_weight).
    # Set ntp_lambda = 0 (or pass --no-ntp) to recover the pure
    # StyleGAN2-ADA baseline used in the ablation.
    ntp_lambda: float = 1.0
    ntp_interval: int = 4               # lazy: apply every N generator steps
    ntp_sub_batch: int = 8              # images per NTP call (caps VGG memory)
    ntp_warmup_epochs: int = 20
    ntp_ramp_epochs: int = 30
    ntp_target_beta: float = 0.99       # EMA smoothing of the real-statistics bank
    ntp_eps: float = 1e-6
    # sub-weights inside NTP (relative importance of each texture descriptor)
    ntp_w_gram: float = 1.0             # VGG Gram texture matching
    ntp_w_featstat: float = 1.0         # VGG feature mean/std matching
    ntp_w_wavelet: float = 0.5          # Haar high-frequency band energy matching
    ntp_w_tv: float = 0.10              # matched TV anti-speckle (excess only)
    ntp_w_fluor: float = 0.50           # fluorescence photometry statistics
    # fluorescence soft-mask parameters
    fluor_threshold_k: float = 0.5
    fluor_temperature: float = 0.08
    # VGG layers (torchvision vgg16.features indices at ReLU outputs)
    vgg_gram_layers: Tuple[int, ...] = (3, 8, 15)        # relu1_2, relu2_2, relu3_3
    vgg_stat_layers: Tuple[int, ...] = (15, 22)          # relu3_3, relu4_3

    # ---- ADA ----------------------------------------------------------------
    ada_target: float = 0.6
    ada_interval: int = 4
    ada_kimg: float = 500.0
    ada_p_max: float = 0.80

    # ---- EMA / truncation ---------------------------------------------------
    ema_kimg: float = 20.0
    w_avg_beta: float = 0.995
    truncation_psi: float = 0.90        # augmentation diversity > heavy truncation

    # ---- Schedule -----------------------------------------------------------
    total_kimg: float = 3000.0
    snapshot_kimg: float = 25.0
    milestone_kimg: float = 250.0
    log_interval_steps: int = 10
    patience_kimg: float = 600.0
    kid_min_delta: float = 5e-4

    # ---- Metrics / validation efficiency -----------------------------------
    metric_gen_per_class: int = 256
    metric_chunk: int = 64
    kid_subset_class: int = 24
    kid_subset_overall: int = 48
    kid_subsets: int = 100

    # ---- Checkpoint IO throttle --------------------------------------------
    periodic_kimg: float = 250.0
    ckpt_keep: int = 3

    # ---- Runtime ------------------------------------------------------------
    seed: int = 42
    num_workers: int = min(8, os.cpu_count() or 8)
    prefetch_factor: int = 4
    use_amp: bool = True
    channels_last: bool = True
    cudnn_benchmark: bool = True
    deterministic: bool = False
    nan_patience: int = 50
    show_gpu_util: bool = True

    # ---- Dataset augmentation (label-preserving for microscopy fields) ------
    aug_random_crop: bool = True        # 256 -> 224 random crop (native scale)
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_rot90: bool = True

    final_images_per_class: int = 1000

    # ---------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def total_steps(self) -> int:
        return int(self.total_kimg * 1000 // self.batch_size)

    @property
    def snapshot_steps(self) -> int:
        return max(1, int(self.snapshot_kimg * 1000 // self.batch_size))

    @property
    def milestone_steps(self) -> int:
        return max(1, int(self.milestone_kimg * 1000 // self.batch_size))

    @property
    def periodic_steps(self) -> int:
        return max(1, int(self.periodic_kimg * 1000 // self.batch_size))

    @property
    def patience_steps(self) -> int:
        return max(1, int(self.patience_kimg * 1000 // self.batch_size))

    @property
    def ema_beta(self) -> float:
        return 0.5 ** (self.batch_size / (self.ema_kimg * 1000.0))

    def nf(self, res: int) -> int:
        return min(self.channel_max, self.channel_base // res)

    def resolutions(self) -> List[int]:
        values, res = [], self.base_resolution
        while res <= self.image_size:
            values.append(res)
            res *= 2
        return values

    def num_ws(self) -> int:
        return len(self.resolutions())

    def ntp_weight(self, epoch_index: int) -> float:
        """Scheduled overall NTP coefficient for a zero-based epoch.

        Warm-up (ntp_warmup_epochs) keeps the fragile early StyleGAN2 dynamics
        pure, then a linear ramp over ntp_ramp_epochs brings in the texture term.
        """
        if self.ntp_lambda == 0.0 or epoch_index < self.ntp_warmup_epochs:
            return 0.0
        if self.ntp_ramp_epochs == 0:
            return self.ntp_lambda
        progress = (epoch_index - self.ntp_warmup_epochs) / self.ntp_ramp_epochs
        return self.ntp_lambda * min(1.0, max(0.0, progress))

    def validate_cfg(self) -> None:
        assert self.batch_size % self.mbstd_group == 0, \
            "batch_size must be divisible by mbstd_group"
        assert self.image_size == 224, "This medical configuration is fixed at 224 px"
        assert self.source_size >= self.image_size, \
            "source_size must be >= image_size for cropping"
        ratio = self.image_size // self.base_resolution
        assert self.base_resolution > 0 and self.image_size % self.base_resolution == 0
        assert ratio > 0 and (ratio & (ratio - 1)) == 0, \
            "image_size/base_resolution must be a power of two"
        assert self.num_classes == 2, "This configuration supports exactly two classes"
        assert 0.0 <= self.style_mixing_prob <= 1.0
        assert self.path_length_interval > 0 and self.path_batch_shrink >= 1
        assert 0.0 < self.val_fraction < 0.5
        assert self.ntp_lambda >= 0.0 and self.ntp_interval > 0
        assert self.ntp_sub_batch >= 2

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class TrainState:
    step: int = 0
    ada_p: float = 0.0
    best_kid: float = float("inf")
    best_fid: float = float("inf")
    no_improve_steps: int = 0
    nan_skips: int = 0
    pl_mean: float = 0.0
    last_ntp: float = 0.0
    ntp_history: List[Dict[str, float]] = field(default_factory=list)


# =============================================================================
# Seed / backends / numerical helpers
# =============================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_backends(cfg: Config) -> None:
    if cfg.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)


def grads_finite(module: nn.Module) -> bool:
    """Single-sync finiteness check (one stacked reduction, one bool() sync)."""
    flags = [torch.isfinite(p.grad).all()
             for p in module.parameters() if p.grad is not None]
    if not flags:
        return True
    return bool(torch.stack(flags).all())


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


# =============================================================================
# Equalized-learning-rate primitives
# =============================================================================
class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 lr_mul: float = 1.0, bias_init: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = (nn.Parameter(torch.full((out_features,), float(bias_init)))
                     if bias else None)
        self.scale = (1.0 / math.sqrt(in_features)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight * self.scale
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, w, b)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 padding: int = 0, bias: bool = True, lr_mul: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel, kernel) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        fan_in = in_ch * kernel * kernel
        self.scale = (1.0 / math.sqrt(fan_in)) * lr_mul
        self.lr_mul = lr_mul
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight * self.scale
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding)


def leaky_act(x: Tensor) -> Tensor:
    return F.leaky_relu(x, LEAKY_SLOPE) * ACT_GAIN


# =============================================================================
# FIR-smoothed 2x up-sampling
# =============================================================================
class Upsample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = k[:, None] * k[None, :]
        k = k / k.sum()
        self.register_buffer("kernel", k[None, None])

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        c = x.size(1)
        w = self.kernel.expand(c, 1, 3, 3).to(x.dtype)
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(x, w, groups=c)


# =============================================================================
# Modulated convolution + noise injection
# =============================================================================
class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, w_dim: int,
                 demodulate: bool = True) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.demodulate = demodulate
        self.padding = kernel // 2
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.scale = 1.0 / math.sqrt(in_ch * kernel * kernel)
        self.affine = EqualizedLinear(w_dim, in_ch, bias=True, bias_init=1.0)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        b, c, h, wd = x.shape
        style = self.affine(w)
        weight = self.scale * self.weight
        weight = weight.unsqueeze(0) * style.view(b, 1, c, 1, 1)
        if self.demodulate:
            with torch.autocast(device_type=weight.device.type, enabled=False):
                w32 = weight.float()
                demod = torch.rsqrt(w32.pow(2).sum(dim=[2, 3, 4]) + 1e-8)
            weight = weight * demod.to(weight.dtype).view(b, self.out_ch, 1, 1, 1)
        weight = weight.view(b * self.out_ch, c, self.kernel, self.kernel)
        x = x.reshape(1, b * c, h, wd)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        return out.view(b, self.out_ch, h, wd)


class NoiseInjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.strength = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3),
                                device=x.device, dtype=x.dtype)
        return x + self.strength * noise


# =============================================================================
# Generator — mapping + 7x7-base synthesis with per-block W+
# =============================================================================
class MappingNetwork(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.num_classes, cfg.embed_dim_g)
        nn.init.normal_(self.embed.weight)
        layers: List[nn.Module] = []
        in_dim = cfg.z_dim + cfg.embed_dim_g
        for i in range(cfg.map_depth):
            layers.append(EqualizedLinear(
                in_dim if i == 0 else cfg.w_dim, cfg.w_dim, lr_mul=cfg.map_lr_mul))
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _pixel_norm(v: Tensor) -> Tensor:
        return v * torch.rsqrt(v.pow(2).mean(dim=1, keepdim=True) + 1e-8)

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        z = self._pixel_norm(z)
        e = self._pixel_norm(self.embed(y))
        x = torch.cat([z, e], dim=1)
        for layer in self.layers:
            x = leaky_act(layer(x))
        return x


class SynthesisLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, kernel, w_dim, demodulate=True)
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x = self.conv(x, w)
        x = self.noise(x)
        x = x + self.bias.view(1, -1, 1, 1)
        return leaky_act(x)


class ToRGB(nn.Module):
    def __init__(self, in_ch: int, w_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, 1, w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        return self.conv(x, w) + self.bias.view(1, 3, 1, 1)


class SynthesisBlock(nn.Module):
    """One resolution stage. All internal layers share this block's style vector.

    Style mixing therefore happens at block granularity (blocks before the
    crossover use w1, blocks after use w2) — valid, simpler than per-conv mixing,
    and sufficient for both the style-mixing regulariser and path-length reg.
    """

    def __init__(self, in_ch: int, out_ch: int, w_dim: int,
                 is_first: bool = False) -> None:
        super().__init__()
        self.is_first = is_first
        if is_first:
            self.up = None
            self.conv0 = SynthesisLayer(in_ch, out_ch, w_dim)
            self.conv1 = None
        else:
            self.up = Upsample()
            self.conv0 = SynthesisLayer(in_ch, out_ch, w_dim)
            self.conv1 = SynthesisLayer(out_ch, out_ch, w_dim)
        self.torgb = ToRGB(out_ch, w_dim)

    def forward(self, x: Tensor, rgb: Optional[Tensor], w: Tensor
                ) -> Tuple[Tensor, Tensor]:
        if not self.is_first:
            x = self.up(x)
        x = self.conv0(x, w)
        if self.conv1 is not None:
            x = self.conv1(x, w)
        y = self.torgb(x, w)
        rgb = y if rgb is None else self.up(rgb) + y
        return x, rgb


class SynthesisNetwork(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        res_list = cfg.resolutions()                 # [7, 14, 28, 56, 112, 224]
        self.output_tanh = cfg.output_tanh
        base = cfg.base_resolution
        self.const = nn.Parameter(torch.randn(1, cfg.nf(base), base, base))
        self.blocks = nn.ModuleList()
        for i, res in enumerate(res_list):
            if i == 0:
                self.blocks.append(
                    SynthesisBlock(cfg.nf(base), cfg.nf(base), cfg.w_dim,
                                   is_first=True))
            else:
                self.blocks.append(
                    SynthesisBlock(cfg.nf(res // 2), cfg.nf(res), cfg.w_dim))

    def forward(self, ws: Tensor) -> Tensor:
        # ws: [B, num_ws, w_dim] — one style vector per block.
        x = self.const.expand(ws.size(0), -1, -1, -1)
        rgb: Optional[Tensor] = None
        for i, block in enumerate(self.blocks):
            x, rgb = block(x, rgb, ws[:, i])
        return torch.tanh(rgb) if self.output_tanh else rgb


class Generator(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_ws = cfg.num_ws()
        self.mapping = MappingNetwork(cfg)
        self.synthesis = SynthesisNetwork(cfg)
        self.register_buffer("w_avg", torch.zeros(cfg.w_dim))

    def _truncate(self, w: Tensor, psi: float) -> Tensor:
        if psi == 1.0:
            return w
        return self.w_avg.unsqueeze(0) + psi * (w - self.w_avg.unsqueeze(0))

    def get_ws(self, z: Tensor, y: Tensor, truncation_psi: float = 1.0,
               style_mixing_prob: float = 0.0) -> Tuple[Tensor, Tensor]:
        """Return (ws [B, num_ws, w_dim], w_primary [B, w_dim]).

        w_primary is the first-latent mapping output; it feeds w_avg tracking and
        path-length regularisation. Style mixing splices a second latent at a
        random crossover block during training only.
        """
        w1 = self.mapping(z, y)
        w1t = self._truncate(w1, truncation_psi)
        ws = w1t.unsqueeze(1).repeat(1, self.n_ws, 1)
        if style_mixing_prob > 0.0 and random.random() < style_mixing_prob:
            z2 = torch.randn_like(z)
            w2 = self._truncate(self.mapping(z2, y), truncation_psi)
            crossover = random.randint(1, self.n_ws - 1)
            ws[:, crossover:] = w2.unsqueeze(1).repeat(1, self.n_ws - crossover, 1)
        return ws, w1

    def synthesis_from_ws(self, ws: Tensor) -> Tensor:
        return self.synthesis(ws)

    def forward(self, z: Tensor, y: Tensor, truncation_psi: float = 1.0,
                style_mixing_prob: float = 0.0) -> Tuple[Tensor, Tensor]:
        ws, w_primary = self.get_ws(z, y, truncation_psi, style_mixing_prob)
        img = self.synthesis(ws)
        return img, w_primary


# =============================================================================
# Discriminator — 7x7 base, minibatch-stddev, projection conditioning
# =============================================================================
class MinibatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4, num_new: int = 1) -> None:
        super().__init__()
        self.group_size = group_size
        self.num_new = num_new

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        g = self.group_size if (b % self.group_size == 0) else b
        f = self.num_new
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = x.float().reshape(g, b // g, f, c // f, h, w)
            y = y - y.mean(dim=0, keepdim=True)
            y = y.square().mean(dim=0)
            y = (y + 1e-8).sqrt()
            y = y.mean(dim=[2, 3, 4])
            y = y.reshape(-1, f, 1, 1)
            y = y.repeat(g, 1, h, w)
        return torch.cat([x, y.to(x.dtype)], dim=1)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv0 = EqualizedConv2d(in_ch, out_ch, 3, padding=1)
        self.conv1 = EqualizedConv2d(out_ch, out_ch, 3, padding=1)
        self.skip = EqualizedConv2d(in_ch, out_ch, 1, bias=True)
        self.down = nn.AvgPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        s = self.down(self.skip(x))
        h = leaky_act(self.conv0(x))
        h = leaky_act(self.conv1(h))
        h = self.down(h)
        return (h + s) * (1.0 / math.sqrt(2.0))


class Discriminator(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        base = cfg.base_resolution                   # 7
        img_res = cfg.image_size                     # 224
        self.from_rgb = EqualizedConv2d(3, cfg.nf(img_res), 1)
        blocks: List[nn.Module] = []
        res = img_res
        in_ch = cfg.nf(img_res)
        while res > base:                            # 224 -> 112 -> 56 -> 28 -> 14 -> 7
            out_ch = cfg.nf(res // 2)
            blocks.append(DiscriminatorBlock(in_ch, out_ch))
            in_ch = out_ch
            res //= 2
        self.blocks = nn.ModuleList(blocks)
        self.mbstd = MinibatchStdDev(cfg.mbstd_group, 1)
        self.conv_out = EqualizedConv2d(cfg.nf(base) + 1, cfg.nf(base), 3, padding=1)
        self.fc = EqualizedLinear(cfg.nf(base) * base * base, cfg.nf(base))
        self.logit = EqualizedLinear(cfg.nf(base), 1)
        self.embed = EqualizedLinear(cfg.num_classes, cfg.nf(base), bias=False)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = leaky_act(self.from_rgb(x))
        for block in self.blocks:
            h = block(h)
        h = self.mbstd(h)
        h = leaky_act(self.conv_out(h))
        h = h.flatten(1)
        feat = leaky_act(self.fc(h))
        out = self.logit(feat)
        onehot = F.one_hot(y, self.num_classes).to(feat.dtype)
        proj = (self.embed(onehot) * feat).sum(dim=1, keepdim=True)
        return out + proj


# =============================================================================
# ADA — adaptive discriminator augmentation (geometric pipeline)
# =============================================================================
def _rotation_matrix(angle: Tensor, device: torch.device) -> Tensor:
    b = angle.shape[0]
    c, s = torch.cos(angle), torch.sin(angle)
    m = torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)
    m[:, 0, 0], m[:, 0, 1] = c, -s
    m[:, 1, 0], m[:, 1, 1] = s, c
    return m


def augment_geometric(x: Tensor, p: float) -> Tensor:
    if p <= 0.0:
        return x
    b, _, h, w = x.shape
    dev = x.device
    with torch.autocast(device_type=dev.type, enabled=False):
        xf = x.float()
        eye = torch.eye(3, device=dev).unsqueeze(0).repeat(b, 1, 1)
        m = eye.clone()

        def bern(prob: float) -> Tensor:
            return (torch.rand(b, device=dev) < prob).float()

        sel = bern(p)
        sign = torch.where(torch.rand(b, device=dev) < 0.5,
                           torch.full((b,), -1.0, device=dev),
                           torch.ones(b, device=dev))
        sign = 1.0 + sel * (sign - 1.0)
        mf = eye.clone(); mf[:, 0, 0] = sign
        m = mf @ m

        sel = bern(p)
        k = torch.randint(0, 4, (b,), device=dev).float()
        m = _rotation_matrix(sel * k * (math.pi / 2.0), dev) @ m

        sel = bern(p)
        ang = sel * (torch.rand(b, device=dev) * 2.0 - 1.0) * math.pi
        m = _rotation_matrix(ang, dev) @ m

        sel = bern(p)
        s = torch.exp((torch.randn(b, device=dev) * 0.2) * sel)
        ms = eye.clone(); ms[:, 0, 0] = s; ms[:, 1, 1] = s
        m = ms @ m

        sel = bern(p)
        rx = max(1, int(0.125 * w))
        ry = max(1, int(0.125 * h))
        tx = torch.randint(-rx, rx + 1, (b,), device=dev).float() / (w / 2.0) * sel
        ty = torch.randint(-ry, ry + 1, (b,), device=dev).float() / (h / 2.0) * sel
        mt = eye.clone(); mt[:, 0, 2] = tx; mt[:, 1, 2] = ty
        m = mt @ m

        sel = bern(p)
        tx = (torch.rand(b, device=dev) * 2.0 - 1.0) * 0.125 * 2.0 * sel
        ty = (torch.rand(b, device=dev) * 2.0 - 1.0) * 0.125 * 2.0 * sel
        mt2 = eye.clone(); mt2[:, 0, 2] = tx; mt2[:, 1, 2] = ty
        m = mt2 @ m

        theta = torch.linalg.inv(m)[:, :2, :]
        grid = F.affine_grid(theta, xf.shape, align_corners=False)
        out = F.grid_sample(xf, grid, mode="bilinear",
                            padding_mode="reflection", align_corners=False)
    return out.to(x.dtype)


class AdaptiveAugment:
    """ADA controller with GPU-resident accumulators (no per-step .item())."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.p = 0.0
        self.target = cfg.ada_target
        self.interval = cfg.ada_interval
        self.kimg = cfg.ada_kimg
        self.p_max = cfg.ada_p_max
        self.batch = cfg.batch_size
        self._sign_sum = torch.zeros((), device=device)
        self._count = 0

    @torch.no_grad()
    def accumulate(self, real_logits: Tensor) -> None:
        self._sign_sum += torch.sign(real_logits).sum()
        self._count += real_logits.numel()

    def update(self) -> float:
        if self._count == 0:
            return float("nan")
        r_t = (self._sign_sum / self._count).item()
        step = (self.batch * self.interval) / (self.kimg * 1000.0)
        adjust = step if r_t > self.target else -step
        self.p = float(min(self.p_max, max(0.0, self.p + adjust)))
        self._sign_sum.zero_()
        self._count = 0
        return r_t

    def __call__(self, x: Tensor) -> Tensor:
        return augment_geometric(x, self.p)


# =============================================================================
# Adversarial losses (logistic default; optional relativistic) + R1 + LeCam + PLR
# =============================================================================
def d_logistic_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def g_logistic_loss(fake_logits: Tensor) -> Tensor:
    return F.softplus(-fake_logits).mean()


def d_relativistic_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    """Relativistic-average (RaGAN) discriminator loss. Optional; OFF by default."""
    r_mean = real_logits.mean()
    f_mean = fake_logits.mean()
    return (F.softplus(-(real_logits - f_mean)).mean()
            + F.softplus(fake_logits - r_mean).mean())


def g_relativistic_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    r_mean = real_logits.mean()
    f_mean = fake_logits.mean()
    return (F.softplus(real_logits - f_mean).mean()
            + F.softplus(-(fake_logits - r_mean)).mean())


def r1_penalty(real_logits: Tensor, real_img: Tensor) -> Tensor:
    grad, = autograd.grad(outputs=real_logits.sum(), inputs=real_img,
                          create_graph=True)
    return grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()


class LeCamRegularizer:
    """LeCam regularisation with GPU-resident EMAs (canonical ReLU-gated form)."""

    def __init__(self, decay: float = 0.99) -> None:
        self.decay = decay
        self.ema_real: Optional[Tensor] = None
        self.ema_fake: Optional[Tensor] = None

    def penalty(self, real_logits: Tensor, fake_logits: Tensor) -> Tensor:
        if self.ema_real is None:
            return real_logits.new_zeros(())
        return (F.relu(real_logits - self.ema_fake).pow(2).mean()
                + F.relu(self.ema_real - fake_logits).pow(2).mean())

    @torch.no_grad()
    def update(self, real_logits: Tensor, fake_logits: Tensor) -> None:
        rm = real_logits.mean().detach()
        fm = fake_logits.mean().detach()
        if self.ema_real is None:
            self.ema_real = rm.clone().float()
            self.ema_fake = fm.clone().float()
        else:
            self.ema_real.mul_(self.decay).add_(rm.float(), alpha=1 - self.decay)
            self.ema_fake.mul_(self.decay).add_(fm.float(), alpha=1 - self.decay)

    def state_dict(self) -> Dict[str, Optional[float]]:
        return {"ema_real": None if self.ema_real is None else float(self.ema_real.item()),
                "ema_fake": None if self.ema_fake is None else float(self.ema_fake.item())}

    def load_state_dict(self, state: Dict, device: torch.device) -> None:
        er, ef = state.get("ema_real"), state.get("ema_fake")
        self.ema_real = None if er is None else torch.tensor(float(er), device=device)
        self.ema_fake = None if ef is None else torch.tensor(float(ef), device=device)


def path_length_penalty(fake_img: Tensor, ws: Tensor, pl_mean: float,
                        decay: float) -> Tuple[Tensor, float]:
    """Lazy StyleGAN2 path-length regularisation.

    Encourages a fixed-magnitude, smoothly varying mapping from W to image space,
    which visibly reduces spatial/texture artifacts. Returns (penalty, new_mean).
    """
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=ws,
                          create_graph=True)
    path_lengths = grad.square().sum(dim=2).mean(dim=1).sqrt()  # [B]
    mean = pl_mean + decay * (path_lengths.mean().item() - pl_mean)
    penalty = (path_lengths - mean).pow(2).mean()
    return penalty, mean


# =============================================================================
# Neural Texture Preservation (NTP) loss
# =============================================================================
_RGB_TO_GRAY = (0.299, 0.587, 0.114)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _grayscale(image01: Tensor) -> Tensor:
    coeff = image01.new_tensor(_RGB_TO_GRAY).view(1, 3, 1, 1)
    return (image01 * coeff).sum(dim=1, keepdim=True)


class VGGFeatures(nn.Module):
    """Frozen VGG16 feature taps for texture (Gram) and feature-statistic losses.

    Inputs are expected in [0, 1]; ImageNet normalisation is applied internally.
    The network is frozen and never contributes trainable parameters. Runs in
    FP32 for numerical stability even when the surrounding step uses AMP.
    """

    def __init__(self, gram_layers: Tuple[int, ...], stat_layers: Tuple[int, ...],
                 device: torch.device) -> None:
        super().__init__()
        if not _TORCHVISION_MODELS:
            raise RuntimeError("torchvision models unavailable for VGG texture loss")
        # IMAGENET1K_V1: standard classification weights, trained with the same
        # ImageNet mean/std normalisation this module applies below. (The
        # alternative FEATURES/amdegroot weights used a different input scaling
        # and would mismatch our normalisation.)
        weights = tvm.VGG16_Weights.IMAGENET1K_V1
        vgg = tvm.vgg16(weights=weights).features.eval()
        self.max_layer = max(max(gram_layers), max(stat_layers))
        self.features = vgg[: self.max_layer + 1].to(device)
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.gram_layers = tuple(gram_layers)
        self.stat_layers = tuple(stat_layers)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x01: Tensor) -> Dict[int, Tensor]:
        x = (x01 - self.mean.to(x01.device)) / self.std.to(x01.device)
        taps: Dict[int, Tensor] = {}
        wanted = set(self.gram_layers) | set(self.stat_layers)
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in wanted:
                taps[i] = h
            if i >= self.max_layer:
                break
        return taps


def gram_matrix(feat: Tensor) -> Tensor:
    """Batch-mean Gram matrix [C, C] — the canonical texture descriptor."""
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2)) / (c * h * w)
    return g.mean(dim=0)


def rel_mse(pred: Tensor, target: Tensor, eps: float) -> Tensor:
    """Scale-invariant matching error: MSE normalised by the target's own energy.

    Gram magnitudes vary by orders of magnitude across VGG layers, and wavelet /
    fluorescence statistics live on different scales again. Dividing by the
    target's mean square makes every descriptor's contribution O(1) at large
    deviation and ~0 at a match, so a single ntp_lambda is portable and no
    per-layer hand-tuning is needed.
    """
    return F.mse_loss(pred, target) / (target.detach().pow(2).mean() + eps)


def haar_high_freq_energy(image01: Tensor) -> Tensor:
    """Mean |LH|, |HL|, |HH| Haar sub-band energy on grayscale -> vector [3].

    Summarises how much high-frequency (edge/texture) content an image set
    carries, without needing a per-image target. Matching it against the real
    class statistics discourages the over-smoothed, low-frequency images that
    adversarial training alone tends to favour.
    """
    gray = _grayscale(image01)
    k = gray.new_tensor([
        [[1.0, 1.0], [-1.0, -1.0]],   # LH (horizontal edges)
        [[1.0, -1.0], [1.0, -1.0]],   # HL (vertical edges)
        [[1.0, -1.0], [-1.0, 1.0]],   # HH (diagonal detail)
    ]).unsqueeze(1) * 0.5             # [3,1,2,2]
    bands = F.conv2d(gray, k, stride=2)          # [B,3,H/2,W/2]
    return bands.abs().mean(dim=(0, 2, 3))       # [3]


def total_variation(image01: Tensor) -> Tensor:
    dh = (image01[:, :, 1:, :] - image01[:, :, :-1, :]).abs().mean()
    dw = (image01[:, :, :, 1:] - image01[:, :, :, :-1]).abs().mean()
    return dh + dw


def fluorescence_stats(image01: Tensor, threshold_k: float, temperature: float,
                       eps: float) -> Tensor:
    """Distribution-level fluorescence-photometry descriptor.

    Returns [coverage, foreground_mean, foreground_std] averaged over the batch.
    A differentiable soft mask concentrates the statistic on bright fluorescent
    structures (nuclei, cytoplasm, membranes) rather than the dark background.
    It is matched only as a summary *statistic*, never as a per-pixel target,
    so it cannot induce averaging/blur.
    """
    gray = _grayscale(image01)
    mean = gray.mean(dim=(2, 3), keepdim=True)
    std = gray.std(dim=(2, 3), keepdim=True, unbiased=False)
    thr = mean + threshold_k * std
    mask = torch.sigmoid((gray - thr) / temperature)
    coverage = mask.mean(dim=(1, 2, 3))
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(eps)
    fg_mean = (mask * gray).sum(dim=(1, 2, 3)) / denom
    fg_var = (mask * (gray - fg_mean.view(-1, 1, 1, 1)) ** 2).sum(dim=(1, 2, 3)) / denom
    fg_std = fg_var.clamp_min(0.0).sqrt()
    return torch.stack([coverage.mean(), fg_mean.mean(), fg_std.mean()])


class TextureTargetBank:
    """Per-class EMA of real-image texture statistics.

    Maintaining an EMA target (rather than comparing to whatever reals happen to
    share a minibatch) both stabilises the objective and removes the need to run
    VGG on reals with gradients. Only the generated sub-batch needs a grad-enabled
    forward, keeping the extra cost to roughly one VGG pass per NTP step.
    """

    def __init__(self, cfg: Config, vgg: VGGFeatures, device: torch.device) -> None:
        self.cfg = cfg
        self.vgg = vgg
        self.device = device
        self.beta = cfg.ntp_target_beta
        self.gram: Dict[int, Dict[int, Tensor]] = {c: {} for c in range(cfg.num_classes)}
        self.fmean: Dict[int, Dict[int, Tensor]] = {c: {} for c in range(cfg.num_classes)}
        self.fstd: Dict[int, Dict[int, Tensor]] = {c: {} for c in range(cfg.num_classes)}
        self.wave: Dict[int, Optional[Tensor]] = {c: None for c in range(cfg.num_classes)}
        self.tv: Dict[int, Optional[Tensor]] = {c: None for c in range(cfg.num_classes)}
        self.fluor: Dict[int, Optional[Tensor]] = {c: None for c in range(cfg.num_classes)}
        self.seen: Dict[int, int] = {c: 0 for c in range(cfg.num_classes)}

    @staticmethod
    def _ema(old: Optional[Tensor], new: Tensor, beta: float) -> Tensor:
        return new.detach() if old is None else old.mul(beta).add(new.detach(), alpha=1 - beta)

    @torch.no_grad()
    def update(self, real01: Tensor, labels: Tensor) -> None:
        cfg = self.cfg
        with torch.autocast(device_type=self.device.type, enabled=False):
            real01 = real01.float()
            for c in range(cfg.num_classes):
                m = labels == c
                if m.sum() < 2:
                    continue
                xc = real01[m]
                taps = self.vgg(xc)
                for li in cfg.vgg_gram_layers:
                    self.gram[c][li] = self._ema(self.gram[c].get(li),
                                                 gram_matrix(taps[li]), self.beta)
                for li in cfg.vgg_stat_layers:
                    self.fmean[c][li] = self._ema(self.fmean[c].get(li),
                                                  taps[li].mean(dim=(0, 2, 3)), self.beta)
                    self.fstd[c][li] = self._ema(self.fstd[c].get(li),
                                                 taps[li].std(dim=(0, 2, 3)), self.beta)
                self.wave[c] = self._ema(self.wave[c], haar_high_freq_energy(xc), self.beta)
                self.tv[c] = self._ema(self.tv[c], total_variation(xc).reshape(1), self.beta)
                self.fluor[c] = self._ema(
                    self.fluor[c],
                    fluorescence_stats(xc, cfg.fluor_threshold_k,
                                       cfg.fluor_temperature, cfg.ntp_eps), self.beta)
                self.seen[c] += 1

    def ready(self, c: int) -> bool:
        return self.seen[c] > 0

    def loss_fake(self, fake01: Tensor, labels: Tensor) -> Dict[str, Tensor]:
        """Generator-side texture loss for the fake sub-batch (grad flows to G)."""
        cfg = self.cfg
        zero = fake01.new_zeros(())
        acc = {k: zero.clone() for k in ("gram", "featstat", "wavelet", "tv", "fluor")}
        n_used = 0
        with torch.autocast(device_type=self.device.type, enabled=False):
            fake01 = fake01.float()
            for c in range(cfg.num_classes):
                m = labels == c
                if m.sum() < 2 or not self.ready(c):
                    continue
                xc = fake01[m]
                eps = cfg.ntp_eps
                taps = self.vgg(xc)
                n_gram = max(1, len(cfg.vgg_gram_layers))
                for li in cfg.vgg_gram_layers:
                    if li in self.gram[c]:
                        acc["gram"] = acc["gram"] + rel_mse(
                            gram_matrix(taps[li]), self.gram[c][li], eps) / n_gram
                n_stat = max(1, len(cfg.vgg_stat_layers))
                for li in cfg.vgg_stat_layers:
                    if li in self.fmean[c]:
                        acc["featstat"] = acc["featstat"] + (
                            rel_mse(taps[li].mean(dim=(0, 2, 3)), self.fmean[c][li], eps)
                            + rel_mse(taps[li].std(dim=(0, 2, 3)), self.fstd[c][li], eps)
                        ) / (2 * n_stat)
                if self.wave[c] is not None:
                    acc["wavelet"] = acc["wavelet"] + rel_mse(
                        haar_high_freq_energy(xc), self.wave[c], eps)
                if self.tv[c] is not None:
                    # matched TV: penalise only *excess* variation (speckle), never
                    # legitimate texture the reals also carry -> no oversmoothing.
                    tv_real = self.tv[c].reshape(())
                    acc["tv"] = acc["tv"] + F.relu(
                        total_variation(xc) - tv_real).pow(2) / (tv_real.pow(2) + eps)
                if self.fluor[c] is not None:
                    acc["fluor"] = acc["fluor"] + rel_mse(
                        fluorescence_stats(xc, cfg.fluor_threshold_k,
                                           cfg.fluor_temperature, cfg.ntp_eps),
                        self.fluor[c], eps)
                n_used += 1
        if n_used > 0:
            for k in acc:
                acc[k] = acc[k] / n_used
        return acc


def combine_ntp(parts: Dict[str, Tensor], cfg: Config) -> Tensor:
    return (cfg.ntp_w_gram * parts["gram"]
            + cfg.ntp_w_featstat * parts["featstat"]
            + cfg.ntp_w_wavelet * parts["wavelet"]
            + cfg.ntp_w_tv * parts["tv"]
            + cfg.ntp_w_fluor * parts["fluor"])


# =============================================================================
# EMA
# =============================================================================
@torch.no_grad()
def update_ema(g_ema: nn.Module, g: nn.Module, beta: float) -> None:
    for p_ema, p in zip(g_ema.parameters(), g.parameters()):
        p_ema.lerp_(p.detach(), 1.0 - beta)
    for b_ema, b in zip(g_ema.buffers(), g.buffers()):
        b_ema.copy_(b)


# =============================================================================
# Dataset / transforms / loaders  (native-scale crops, leak-safe hold-out)
# =============================================================================
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
CLASS_TO_IDX: Dict[str, int] = {"cancer": 0, "normal": 1}
CLASS_NAMES: List[str] = ["cancer", "normal"]


def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXTENSIONS)


def recursive_image_files(folder: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(folder):
        for fn in filenames:
            if is_image_file(fn):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


class ClassFolderDataset(Dataset):
    def __init__(self, root: str, transform) -> None:
        self.root = os.path.abspath(root)
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Data path not found: {self.root}")
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        for entry in sorted(os.scandir(self.root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            name = entry.name.lower()
            if name not in CLASS_TO_IDX:
                raise ValueError(
                    f"Unknown class folder '{entry.name}' under {self.root}. "
                    f"Expected exactly: {list(CLASS_TO_IDX)}.")
            idx = CLASS_TO_IDX[name]
            for path in recursive_image_files(entry.path):
                self.samples.append((path, idx))
        if not self.samples:
            raise ValueError(f"No images found under {self.root}")
        self.samples.sort(key=lambda s: s[0])
        self.classes = CLASS_NAMES

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for _, lbl in self.samples:
            counts[lbl] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transforms(cfg: Config, train: bool):
    import torchvision.transforms as T
    from torchvision.transforms import InterpolationMode

    ops: List = []
    # Native-scale crop: sources are 256x256; cropping to 224 avoids resampling
    # blur and gives free translation augmentation. If a source is not >= 224 the
    # Resize guarantees a valid crop.
    ops.append(T.Resize(cfg.source_size, interpolation=InterpolationMode.BICUBIC,
                        antialias=True))
    if train and cfg.aug_random_crop:
        ops.append(T.RandomCrop(cfg.image_size))
    else:
        ops.append(T.CenterCrop(cfg.image_size))
    if train:
        if cfg.aug_hflip:
            ops.append(T.RandomHorizontalFlip(0.5))
        if cfg.aug_vflip:
            ops.append(T.RandomVerticalFlip(0.5))
        if cfg.aug_rot90:
            ops.append(T.RandomChoice([
                T.RandomRotation((0, 0)),
                T.RandomRotation((90, 90)),
                T.RandomRotation((180, 180)),
                T.RandomRotation((270, 270)),
            ]))
    ops.append(T.ToTensor())
    ops.append(T.Normalize([0.5] * 3, [0.5] * 3))
    return T.Compose(ops)


def _deterministic_holdout(samples: List[Tuple[str, int]], cfg: Config
                           ) -> Tuple[List[int], List[int]]:
    """Per-class deterministic hold-out by filename hash.

    A hash split is stable across runs (unlike a re-seeded random draw) so the
    checkpoint-selection set never silently leaks into training between runs.
    NOTE: this is a file-level split. If patient/session identifiers exist they
    should be used instead to prevent ROI-level leakage across the split.
    """
    by_class: Dict[int, List[int]] = {c: [] for c in range(cfg.num_classes)}
    for i, (path, lbl) in enumerate(samples):
        by_class[lbl].append(i)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for c, idxs in by_class.items():
        def keyf(i: int) -> float:
            h = hashlib.sha256(os.path.basename(samples[i][0]).encode()).hexdigest()
            return int(h[:8], 16) / 0xFFFFFFFF
        ordered = sorted(idxs, key=keyf)
        n_val = max(2, int(round(cfg.val_fraction * len(ordered))))
        val_idx.extend(ordered[:n_val])
        train_idx.extend(ordered[n_val:])
    return sorted(train_idx), sorted(val_idx)


def get_dataloaders(cfg: Config):
    train_tf = build_transforms(cfg, train=True)
    val_tf = build_transforms(cfg, train=False)

    base_train = ClassFolderDataset(cfg.train_data_path, train_tf)
    base_val = ClassFolderDataset(cfg.train_data_path, val_tf)

    train_idx, val_idx = _deterministic_holdout(base_train.samples, cfg)
    train_ds = Subset(base_train, train_idx)
    val_ds = Subset(base_val, val_idx)

    train_labels = np.array([base_train.samples[i][1] for i in train_idx])
    counts = np.bincount(train_labels, minlength=cfg.num_classes)
    val_counts = np.bincount(
        [base_val.samples[i][1] for i in val_idx], minlength=cfg.num_classes)
    print(f"[DATA] Classes        : {CLASS_NAMES}")
    print(f"[DATA] Train counts   : {counts.tolist()}  (total {len(train_idx)})")
    print(f"[DATA] Val(KID) counts: {val_counts.tolist()}  (total {len(val_idx)})")

    weights = 1.0 / np.maximum(counts, 1)[train_labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_idx), replacement=True)

    common = dict(num_workers=cfg.num_workers, pin_memory=True,
                  worker_init_fn=_worker_init_fn)
    pf = cfg.prefetch_factor if cfg.num_workers > 0 else None
    pw = cfg.num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True,
        persistent_workers=pw, prefetch_factor=pf, **common)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
        persistent_workers=pw, prefetch_factor=pf, **common)
    return train_loader, val_loader, CLASS_NAMES


# =============================================================================
# Metrics — KID (selection) + FID (secondary); persistent, chunked, freed
# =============================================================================
class InceptionFeatures(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        if not _TORCHVISION_MODELS:
            raise RuntimeError("torchvision models unavailable for feature metrics")
        weights = tvm.Inception_V3_Weights.DEFAULT
        net = tvm.inception_v3(weights=weights, aux_logits=True,
                               transform_input=False)
        net.fc = nn.Identity()
        net.eval()
        self.net = net.to(device)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x01: Tensor) -> Tensor:
        x = F.interpolate(x01, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.net(x.float())


class MetricEvaluator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.available = _TORCHMETRICS
        self.kid_class = None
        self.kid_overall = None
        self.fid = None

    def _build(self, device: torch.device) -> None:
        if self.kid_class is not None:
            return
        cfg = self.cfg
        self.kid_class = KernelInceptionDistance(
            feature=2048, subset_size=cfg.kid_subset_class,
            subsets=cfg.kid_subsets, normalize=True).to(device)
        self.kid_overall = KernelInceptionDistance(
            feature=2048, subset_size=cfg.kid_subset_overall,
            subsets=cfg.kid_subsets, normalize=True).to(device)
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def _update_real(self, real_cpu: Tensor, device: torch.device) -> None:
        ch = self.cfg.metric_chunk
        for i in range(0, real_cpu.size(0), ch):
            x = (real_cpu[i:i + ch] * 0.5 + 0.5).clamp(0, 1).to(device)
            self.kid_class.update(x, real=True)
            self.kid_overall.update(x, real=True)
            self.fid.update(x, real=True)
            del x

    @torch.no_grad()
    def _gen_update(self, g_ema: Generator, cls: int, n: int,
                    device: torch.device) -> None:
        cfg = self.cfg
        ch = cfg.metric_chunk
        remaining = n
        while remaining > 0:
            bs = min(ch, remaining)
            z = torch.randn(bs, cfg.z_dim, device=device)
            y = torch.full((bs,), cls, dtype=torch.long, device=device)
            img, _ = g_ema(z, y, truncation_psi=cfg.truncation_psi)
            remaining -= bs
            if not torch.isfinite(img).all():
                del z, img
                continue
            x = (img * 0.5 + 0.5).clamp(0, 1)
            self.kid_class.update(x, real=False)
            self.kid_overall.update(x, real=False)
            self.fid.update(x, real=False)
            del z, img, x

    @torch.no_grad()
    def compute(self, g_ema: Generator, val_loader: DataLoader,
                device: torch.device) -> Dict[str, float]:
        if not self.available:
            return {"kid_mean": float("inf"), "kid_overall": float("nan"),
                    "fid": float("nan")}
        cfg = self.cfg
        self._build(device)
        self.kid_overall.reset()
        self.fid.reset()
        g_ema.eval()

        reals: Dict[int, List[Tensor]] = {c: [] for c in range(cfg.num_classes)}
        for imgs, lbls in val_loader:
            for c in range(cfg.num_classes):
                m = (lbls == c)
                if m.any():
                    reals[c].append(imgs[m])

        kid_per_class: Dict[int, float] = {}
        for c in range(cfg.num_classes):
            self.kid_class.reset()
            if not reals[c]:
                kid_per_class[c] = float("nan")
                continue
            self._update_real(torch.cat(reals[c]), device)
            self._gen_update(g_ema, c, cfg.metric_gen_per_class, device)
            try:
                kid_per_class[c] = float(self.kid_class.compute()[0].item())
            except Exception:
                kid_per_class[c] = float("nan")

        try:
            kid_overall = float(self.kid_overall.compute()[0].item())
        except Exception:
            kid_overall = float("nan")
        try:
            fid = float(self.fid.compute().item())
        except Exception:
            fid = float("nan")

        valid = [v for v in kid_per_class.values() if not math.isnan(v)]
        kid_mean = float(np.mean(valid)) if valid else float("inf")

        g_ema.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        out = {"kid_mean": kid_mean, "kid_overall": kid_overall, "fid": fid}
        for c in range(cfg.num_classes):
            out[f"kid_{CLASS_NAMES[c]}"] = kid_per_class.get(c, float("nan"))
        return out

    @torch.no_grad()
    def sample_for_pr(self, g_ema: Generator, cls: int, n: int,
                      device: torch.device) -> Tensor:
        cfg = self.cfg
        out: List[Tensor] = []
        remaining = n
        while remaining > 0:
            bs = min(cfg.metric_chunk, remaining)
            z = torch.randn(bs, cfg.z_dim, device=device)
            y = torch.full((bs,), cls, dtype=torch.long, device=device)
            img, _ = g_ema(z, y, truncation_psi=cfg.truncation_psi)
            remaining -= bs
            if not torch.isfinite(img).all():
                continue
            out.append((img * 0.5 + 0.5).clamp(0, 1))
        return torch.cat(out) if out else torch.zeros(
            0, 3, cfg.image_size, cfg.image_size, device=device)


@torch.no_grad()
def improved_precision_recall(real_feat: Tensor, fake_feat: Tensor, k: int = 3
                              ) -> Tuple[float, float]:
    def knn_radii(feat: Tensor) -> Tensor:
        kk = max(1, min(k, feat.size(0) - 1))
        d = torch.cdist(feat, feat)
        d.fill_diagonal_(float("inf"))
        return d.topk(kk, largest=False).values[:, -1]
    real_r = knn_radii(real_feat)
    fake_r = knn_radii(fake_feat)
    d_rf = torch.cdist(fake_feat, real_feat)
    precision = (d_rf <= real_r.unsqueeze(0)).any(dim=1).float().mean().item()
    d_fr = torch.cdist(real_feat, fake_feat)
    recall = (d_fr <= fake_r.unsqueeze(0)).any(dim=1).float().mean().item()
    return precision, recall


@torch.no_grad()
def nn_memorization(real_feat: Tensor, fake_feat: Tensor) -> Dict[str, float]:
    d_fr = torch.cdist(fake_feat, real_feat).min(dim=1).values
    d_rr = torch.cdist(real_feat, real_feat)
    d_rr.fill_diagonal_(float("inf"))
    d_rr = d_rr.min(dim=1).values
    return {"memo_fake_to_real_median": float(d_fr.median().item()),
            "memo_real_to_real_median": float(d_rr.median().item())}


# =============================================================================
# Checkpoint manager
# =============================================================================
def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path: str, cfg: Config, state: TrainState,
                    G: Generator, D: Discriminator, G_ema: Generator,
                    opt_G: optim.Optimizer, opt_D: optim.Optimizer,
                    lecam: LeCamRegularizer, ada: AdaptiveAugment,
                    class_names: List[str], z_fixed: Optional[Tensor]) -> None:
    ckpt = {
        "step": state.step,
        "best_kid": state.best_kid,
        "best_fid": state.best_fid,
        "no_improve_steps": state.no_improve_steps,
        "nan_skips": state.nan_skips,
        "pl_mean": state.pl_mean,
        "ntp": {"last_value": state.last_ntp, "history": state.ntp_history},
        "ada_p": ada.p,
        "lecam": lecam.state_dict(),
        "z_fixed": None if z_fixed is None else z_fixed.cpu(),
        "G": G.state_dict(),
        "D": D.state_dict(),
        "G_ema": G_ema.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "config": asdict(cfg),
        "class_names": class_names,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    _atomic_save(ckpt, path)


def load_checkpoint(path: str, G: Generator, D: Discriminator, G_ema: Generator,
                    opt_G: Optional[optim.Optimizer], opt_D: Optional[optim.Optimizer],
                    lecam: Optional[LeCamRegularizer], ada: Optional[AdaptiveAugment],
                    state: Optional[TrainState], device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G"], strict=True)
    D.load_state_dict(ckpt["D"], strict=True)
    G_ema.load_state_dict(ckpt["G_ema"], strict=True)
    if opt_G is not None:
        opt_G.load_state_dict(ckpt["opt_G"])
    if opt_D is not None:
        opt_D.load_state_dict(ckpt["opt_D"])
    if lecam is not None and "lecam" in ckpt:
        lecam.load_state_dict(ckpt["lecam"], device)
    if ada is not None:
        ada.p = ckpt.get("ada_p", 0.0)
    if state is not None:
        state.step = ckpt.get("step", 0)
        state.best_kid = ckpt.get("best_kid", float("inf"))
        state.best_fid = ckpt.get("best_fid", float("inf"))
        state.no_improve_steps = ckpt.get("no_improve_steps", 0)
        state.nan_skips = ckpt.get("nan_skips", 0)
        state.pl_mean = float(ckpt.get("pl_mean", 0.0))
        ntp = ckpt.get("ntp", {})
        state.last_ntp = float(ntp.get("last_value", 0.0))
        history = ntp.get("history", [])
        state.ntp_history = [
            {str(k): float(v) for k, v in row.items()}
            for row in history if isinstance(row, dict)]
    rng = ckpt.get("rng")
    if rng is not None:
        try:
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[ckpt] could not restore RNG state: {e}")
    return ckpt


def load_ema_from_checkpoint(ckpt: Dict, g_ema: Generator) -> List[str]:
    g_ema.load_state_dict(ckpt["G_ema"], strict=True)
    return list(CLASS_NAMES)


# =============================================================================
# Sampling & visualisation
# =============================================================================
@torch.no_grad()
def save_preview(g_ema: Generator, cfg: Config, class_names: List[str],
                 kimg: float, z_fixed: Tensor, device: torch.device) -> None:
    g_ema.eval()
    all_imgs: List[Tensor] = []
    for c in range(cfg.num_classes):
        y = torch.full((z_fixed.size(0),), c, dtype=torch.long, device=device)
        img, _ = g_ema(z_fixed.to(device), y, truncation_psi=1.0)
        if not torch.isfinite(img).all():
            g_ema.train()
            return
        all_imgs.append(img.cpu())
    grid = torch.cat(all_imgs, dim=0)
    path = os.path.join(cfg.output_path, f"preview_{int(kimg):05d}kimg.png")
    vutils.save_image(grid, path, normalize=True, nrow=z_fixed.size(0),
                      value_range=(-1, 1))
    g_ema.train()


@torch.no_grad()
def save_compare_grids(g_ema: Generator, val_loader: DataLoader, cfg: Config,
                       class_names: List[str], kimg: float,
                       device: torch.device) -> None:
    g_ema.eval()
    collected: Dict[int, List[Tensor]] = {c: [] for c in range(cfg.num_classes)}
    for imgs, lbls in val_loader:
        for c in range(cfg.num_classes):
            m = (lbls == c)
            if m.any():
                collected[c].append(imgs[m])
        if all(sum(t.size(0) for t in collected[c]) >= 8
               for c in range(cfg.num_classes)):
            break

    fig, axes = plt.subplots(cfg.num_classes, 2, figsize=(14, 4 * cfg.num_classes))
    if cfg.num_classes == 1:
        axes = [axes]
    for c in range(cfg.num_classes):
        cname = class_names[c]
        if not collected[c]:
            continue
        real = torch.cat(collected[c])[:8].to(device)
        z = torch.randn(8, cfg.z_dim, device=device)
        y = torch.full((8,), c, dtype=torch.long, device=device)
        fake, _ = g_ema(z, y, truncation_psi=cfg.truncation_psi)
        if not torch.isfinite(fake).all():
            continue
        real_np = vutils.make_grid(real.cpu(), nrow=4, normalize=True,
                                   value_range=(-1, 1)).permute(1, 2, 0).numpy()
        fake_np = vutils.make_grid(fake.cpu(), nrow=4, normalize=True,
                                   value_range=(-1, 1)).permute(1, 2, 0).numpy()
        axes[c][0].imshow(real_np); axes[c][0].set_title(f"REAL — {cname}", fontsize=11)
        axes[c][0].axis("off")
        axes[c][1].imshow(fake_np)
        axes[c][1].set_title(f"GENERATED — {cname}  ({int(kimg)} kimg)", fontsize=11)
        axes[c][1].axis("off")
    plt.suptitle(f"Real vs Generated  |  {int(kimg)} kimg", fontsize=13, y=1.01)
    plt.tight_layout()
    cmp_path = os.path.join(cfg.output_path, f"compare_{int(kimg):05d}kimg.png")
    plt.savefig(cmp_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    g_ema.train()


# =============================================================================
# Final generation
# =============================================================================
@torch.no_grad()
def generate_final(g_ema: Generator, cfg: Config, class_names: List[str],
                   device: torch.device, per_class: int) -> str:
    g_ema.eval()
    gen_root = os.path.join(cfg.output_path, "final_generated")
    os.makedirs(gen_root, exist_ok=True)
    for c_idx in range(cfg.num_classes):
        c_name = class_names[c_idx]
        # Plain class-named folders so the output drops directly into the
        # classifier's synthetic-root expectation (cancer/ and normal/).
        c_dir = os.path.join(gen_root, c_name)
        os.makedirs(c_dir, exist_ok=True)
        count, retries = 0, 0
        while count < per_class:
            bs = min(50, per_class - count)
            z = torch.randn(bs, cfg.z_dim, device=device)
            y = torch.full((bs,), c_idx, dtype=torch.long, device=device)
            img, _ = g_ema(z, y, truncation_psi=cfg.truncation_psi)
            if not torch.isfinite(img).all():
                retries += 1
                if retries > 20:
                    break
                continue
            img = (img * 0.5 + 0.5).clamp(0, 1)
            for i in range(bs):
                vutils.save_image(
                    img[i], os.path.join(c_dir, f"{c_name}_{count:04d}.png"))
                count += 1
        print(f"[GEN] {c_name}: {count}/{per_class}")

    zip_path = os.path.join(cfg.output_path, "generated_images.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(gen_root):
            for fn in files:
                full = os.path.join(root, fn)
                zf.write(full, os.path.relpath(full, gen_root))

    _atomic_save({"G_ema": g_ema.state_dict(), "class_names": class_names,
                  "config": asdict(cfg)},
                 os.path.join(cfg.output_path, "generator_final_ema.pth"))
    return zip_path


# =============================================================================
# NTP monitoring plot + logger
# =============================================================================
def save_ntp_plot(history: List[Dict[str, float]], cfg: Config) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    raw = [row["ntp"] for row in history]
    weighted = [row["weighted_ntp"] for row in history]
    weights = [row["lambda"] for row in history]

    fig, ax_loss = plt.subplots(figsize=(8.5, 4.4))
    ax_loss.plot(epochs, raw, color="#3565a8", linewidth=1.8, label="NTP (raw)")
    ax_loss.plot(epochs, weighted, color="#d87928", linewidth=1.8,
                 label="lambda x NTP")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.grid(alpha=0.25)
    ax_lambda = ax_loss.twinx()
    ax_lambda.plot(epochs, weights, color="#4f8a5b", linestyle="--",
                   linewidth=1.5, label="NTP lambda")
    ax_lambda.set_ylabel("NTP coefficient")
    ax_lambda.set_ylim(bottom=0.0,
                       top=max(cfg.ntp_lambda * 1.1, max(weights) * 1.1, 1e-6))
    la, lba = ax_loss.get_legend_handles_labels()
    lb, lbb = ax_lambda.get_legend_handles_labels()
    ax_loss.legend(la + lb, lba + lbb, loc="upper right")
    ax_loss.set_title("Neural Texture Preservation loss")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.output_path, "ntp_history.png"), dpi=140)
    plt.close(fig)


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class StatusLogger:
    def __init__(self, cfg: Config, total_steps: int, steps_per_epoch: int) -> None:
        self.cfg = cfg
        self.total_steps = total_steps
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.total_epochs = max(1, math.ceil(total_steps / self.steps_per_epoch))
        self.t0 = time.time()
        self._nvml_handle = None
        if cfg.show_gpu_util and _PYNVML and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                    torch.cuda.current_device())
            except Exception:
                self._nvml_handle = None

    def _gpu(self) -> Tuple[float, int]:
        mem = (torch.cuda.memory_reserved() / 1e9
               if torch.cuda.is_available() else 0.0)
        util = -1
        if self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
            except Exception:
                util = -1
        return mem, util

    def live(self, step: int, stats: Dict[str, float], ada_p: float, lr: float,
             state: TrainState) -> None:
        epoch = (step - 1) // self.steps_per_epoch + 1
        batch_i = (step - 1) % self.steps_per_epoch + 1
        elapsed = time.time() - self.t0
        eta = elapsed / max(1, step) * (self.total_steps - step)
        mem, util = self._gpu()
        util_s = f"{util:3d}%" if util >= 0 else " -- "
        imgs = step * self.cfg.batch_size
        best_kid = "inf" if not math.isfinite(state.best_kid) else f"{state.best_kid:.4f}"
        line = (
            f"\rEp {epoch}/{self.total_epochs} "
            f"B {batch_i}/{self.steps_per_epoch} "
            f"| {_fmt_time(elapsed)}<{_fmt_time(eta)} "
            f"| D {stats['loss_d']:.3f} G {stats['loss_g']:.3f} "
            f"NTP {stats['ntp']:.3f}(x{stats['ntp_lambda']:.2f}) "
            f"PL {stats.get('pl', 0):.3f} R1 {stats['r1']:.2f} "
            f"| Dr {stats.get('d_real', 0):+.2f} Df {stats.get('d_fake', 0):+.2f} "
            f"Gs {stats.get('g_std', 0):.3f} "
            f"| ADA {ada_p:.3f} lr {lr:.1e} "
            f"| bKID {best_kid} "
            f"| {mem:.1f}G {util_s} | {imgs/1000:.0f}k"
        )
        sys.stdout.write(line + "   ")
        sys.stdout.flush()

    def event(self, msg: str) -> None:
        sys.stdout.write("\n" + msg + "\n")
        sys.stdout.flush()


# =============================================================================
# Trainer
# =============================================================================
class Trainer:
    def __init__(self, cfg: Config) -> None:
        cfg.validate_cfg()
        self.cfg = cfg
        self.device = cfg.device

        for d in (cfg.output_path, cfg.checkpoint_path, cfg.log_path):
            os.makedirs(d, exist_ok=True)
        cfg.save_json(os.path.join(cfg.checkpoint_path, "config.json"))

        seed_everything(cfg.seed)
        setup_backends(cfg)

        (self.train_loader, self.val_loader, self.class_names) = get_dataloaders(cfg)

        self.G = Generator(cfg).to(self.device)
        self.D = Discriminator(cfg).to(self.device)
        self.G_ema = deepcopy(self.G).eval()
        set_requires_grad(self.G_ema, False)

        if cfg.channels_last and self.device.type == "cuda":
            self.G = self.G.to(memory_format=torch.channels_last)
            self.D = self.D.to(memory_format=torch.channels_last)
            self.G_ema = self.G_ema.to(memory_format=torch.channels_last)

        self.opt_G = optim.Adam(self.G.parameters(), lr=cfg.lr,
                                betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps,
                                weight_decay=cfg.weight_decay)
        self.opt_D = optim.Adam(self.D.parameters(), lr=cfg.lr,
                                betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps,
                                weight_decay=cfg.weight_decay)

        self.ada = AdaptiveAugment(cfg, self.device)
        self.lecam = LeCamRegularizer(cfg.lecam_decay)
        self.metrics = MetricEvaluator(cfg)
        self.state = TrainState()

        # Neural Texture Preservation module (frozen VGG + per-class EMA bank).
        self.ntp_bank: Optional[TextureTargetBank] = None
        if cfg.ntp_lambda > 0.0 and _TORCHVISION_MODELS:
            try:
                vgg = VGGFeatures(cfg.vgg_gram_layers, cfg.vgg_stat_layers, self.device)
                self.ntp_bank = TextureTargetBank(cfg, vgg, self.device)
                print("[NTP] VGG texture module ready "
                      f"(gram={cfg.vgg_gram_layers}, stat={cfg.vgg_stat_layers})")
            except Exception as e:  # pragma: no cover
                warnings.warn(f"[NTP] disabled — {e}")
                self.ntp_bank = None
        elif cfg.ntp_lambda > 0.0:
            warnings.warn("[NTP] torchvision unavailable — texture loss disabled")

        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        self.writer = (SummaryWriter(cfg.log_path)
                       if (_TENSORBOARD and self.device.type == "cuda") else None)
        self.z_fixed = torch.randn(8, cfg.z_dim)
        self._steps_per_epoch = 1
        self._current_epoch = 0

        self._acc = {k: torch.zeros((), device=self.device)
                     for k in ("loss_d", "loss_g", "ntp", "ntp_lambda",
                               "d_real", "d_fake", "g_std")}
        self._acc_r1 = torch.zeros((), device=self.device)
        self._acc_pl = torch.zeros((), device=self.device)
        self._acc_n = 0
        self._acc_r1_n = 0
        self._acc_pl_n = 0
        self._epoch_ntp = torch.zeros((), device=self.device)
        self._epoch_ntp_weighted = torch.zeros((), device=self.device)
        self._epoch_ntp_n = 0

        n_g = sum(p.numel() for p in self.G.parameters())
        n_d = sum(p.numel() for p in self.D.parameters())
        print(f"[MODEL] Generator     : {n_g / 1e6:.2f}M params")
        print(f"[MODEL] Discriminator : {n_d / 1e6:.2f}M params")
        print(f"[MODEL] Resolutions   : {cfg.resolutions()}  (num_ws={cfg.num_ws()})")
        print(f"[MODEL] Device        : {self.device} | AMP(bf16): {self.use_amp}")

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype,
                              enabled=self.use_amp)

    # ---- one optimisation step ---------------------------------------------
    def train_step(self, real_img: Tensor, labels: Tensor) -> None:
        cfg = self.cfg
        real_img = real_img.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        if cfg.channels_last and self.device.type == "cuda":
            real_img = real_img.to(memory_format=torch.channels_last)
        b = real_img.size(0)

        # ---------------- Discriminator update ------------------------------
        z_d = torch.randn(b, cfg.z_dim, device=self.device)
        self.opt_D.zero_grad(set_to_none=True)
        with self._autocast():
            with torch.no_grad():
                fake_img, _ = self.G(z_d, labels,
                                     style_mixing_prob=cfg.style_mixing_prob)
            real_aug = self.ada(real_img)
            fake_aug = self.ada(fake_img.detach())
            real_logits = self.D(real_aug, labels)
            fake_logits = self.D(fake_aug, labels)
            if cfg.use_relativistic:
                loss_d = d_relativistic_loss(real_logits, fake_logits)
            else:
                loss_d = d_logistic_loss(real_logits, fake_logits)
            loss_lecam = cfg.lecam_lambda * self.lecam.penalty(real_logits, fake_logits)
            loss_d_total = loss_d + loss_lecam

        self.lecam.update(real_logits.detach(), fake_logits.detach())
        self.ada.accumulate(real_logits.detach())
        self._acc["d_real"] += real_logits.detach().float().mean()
        self._acc["d_fake"] += fake_logits.detach().float().mean()

        loss_d_total.backward()

        if self.state.step % cfg.r1_interval == 0:
            with torch.autocast(device_type=self.device.type, enabled=False):
                real_tmp = real_img.detach().float().requires_grad_(True)
                real_logits_r1 = self.D(real_tmp, labels)
                r1 = r1_penalty(real_logits_r1, real_tmp)
                r1_loss = (cfg.r1_gamma / 2.0) * r1 * cfg.r1_interval
            r1_loss.backward()
            self._acc_r1 += r1.detach()
            self._acc_r1_n += 1

        if grads_finite(self.D):
            self.opt_D.step()
        else:
            self.state.nan_skips += 1

        # ---------------- Update the real-statistics bank (no grad) ----------
        # Kept in step with the lazy NTP schedule so its cost is amortised.
        if self.ntp_bank is not None and self.state.step % cfg.ntp_interval == 0:
            real01 = (real_img.detach().clamp(-1, 1) * 0.5 + 0.5)
            self.ntp_bank.update(real01[:cfg.ntp_sub_batch], labels[:cfg.ntp_sub_batch])

        # ---------------- Generator update ----------------------------------
        z_g = torch.randn(b, cfg.z_dim, device=self.device)
        self.opt_G.zero_grad(set_to_none=True)
        with self._autocast():
            fake_img, w_primary = self.G(z_g, labels,
                                         style_mixing_prob=cfg.style_mixing_prob)
            fake_aug = self.ada(fake_img)
            g_logits = self.D(fake_aug, labels)
            if cfg.use_relativistic:
                with torch.no_grad():
                    real_logits_g = self.D(self.ada(real_img), labels)
                loss_g_adv = g_relativistic_loss(real_logits_g, g_logits)
            else:
                loss_g_adv = g_logistic_loss(g_logits)

        # Neural Texture Preservation (lazy, warm-started, ramped).
        ntp_weight = cfg.ntp_weight(self._current_epoch)
        loss_ntp = fake_img.new_zeros((), dtype=torch.float32)
        if (self.ntp_bank is not None and ntp_weight > 0.0
                and self.state.step % cfg.ntp_interval == 0):
            fake01 = (fake_img.clamp(-1, 1) * 0.5 + 0.5)[:cfg.ntp_sub_batch]
            lab_sb = labels[:cfg.ntp_sub_batch]
            parts = self.ntp_bank.loss_fake(fake01, lab_sb)
            loss_ntp = combine_ntp(parts, cfg)
        weighted_ntp = loss_ntp * ntp_weight
        loss_g_total = loss_g_adv + weighted_ntp
        loss_g_total.backward()

        if grads_finite(self.G):
            self.opt_G.step()
        else:
            self.state.nan_skips += 1

        # ---------------- Lazy path-length regularisation -------------------
        if (cfg.path_length_weight > 0.0
                and self.state.step % cfg.path_length_interval == 0):
            self.opt_G.zero_grad(set_to_none=True)
            pl_bs = max(1, b // cfg.path_batch_shrink)
            z_pl = torch.randn(pl_bs, cfg.z_dim, device=self.device)
            y_pl = labels[:pl_bs]
            with torch.autocast(device_type=self.device.type, enabled=False):
                ws_pl, _ = self.G.get_ws(z_pl, y_pl)
                ws_pl = ws_pl.requires_grad_(True)
                img_pl = self.G.synthesis_from_ws(ws_pl).float()
                pl_pen, new_mean = path_length_penalty(
                    img_pl, ws_pl, self.state.pl_mean, cfg.path_length_decay)
                pl_loss = (cfg.path_length_weight * pl_pen * cfg.path_length_interval)
            pl_loss.backward()
            self.state.pl_mean = new_mean
            if grads_finite(self.G):
                self.opt_G.step()
            self._acc_pl += pl_pen.detach()
            self._acc_pl_n += 1

        # ---------------- w_avg + EMA ---------------------------------------
        with torch.no_grad():
            mean_w = w_primary.detach().mean(dim=0)
            self.G.w_avg.mul_(cfg.w_avg_beta).add_(mean_w, alpha=1 - cfg.w_avg_beta)
        update_ema(self.G_ema, self.G, cfg.ema_beta)

        if self.state.step % cfg.ada_interval == 0:
            self.ada.update()

        # ---------------- accumulate scalars (synced at log interval) --------
        self._acc["loss_d"] += loss_d.detach()
        self._acc["loss_g"] += loss_g_total.detach()
        self._acc["ntp"] += loss_ntp.detach()
        self._acc["ntp_lambda"] += ntp_weight
        self._acc["g_std"] += fake_img.detach().float().std()
        self._acc_n += 1
        self._epoch_ntp += loss_ntp.detach()
        self._epoch_ntp_weighted += weighted_ntp.detach()
        self._epoch_ntp_n += 1

    def _drain_acc(self) -> Dict[str, float]:
        n = max(1, self._acc_n)
        stats = {k: float((v / n).item()) for k, v in self._acc.items()}
        stats["r1"] = float((self._acc_r1 / max(1, self._acc_r1_n)).item()
                            ) if self._acc_r1_n else 0.0
        stats["pl"] = float((self._acc_pl / max(1, self._acc_pl_n)).item()
                            ) if self._acc_pl_n else 0.0
        for v in self._acc.values():
            v.zero_()
        self._acc_r1.zero_(); self._acc_pl.zero_()
        self._acc_n = 0; self._acc_r1_n = 0; self._acc_pl_n = 0
        return stats

    def _finalize_epoch_ntp(self, epoch: int, logger: StatusLogger) -> None:
        n = max(1, self._epoch_ntp_n)
        raw = float((self._epoch_ntp / n).item())
        weighted = float((self._epoch_ntp_weighted / n).item())
        weight = self.cfg.ntp_weight(epoch - 1)
        record = {"epoch": float(epoch), "ntp": raw,
                  "weighted_ntp": weighted, "lambda": weight}
        self.state.last_ntp = raw
        self.state.ntp_history.append(record)
        save_ntp_plot(self.state.ntp_history, self.cfg)
        logger.event(f"[NTP] epoch {epoch:03d} | raw={raw:.6f} | "
                     f"lambda={weight:.4f} | weighted={weighted:.6f}")
        if self.writer is not None:
            self.writer.add_scalar("epoch/ntp", raw, epoch)
            self.writer.add_scalar("epoch/ntp_weighted", weighted, epoch)
            self.writer.add_scalar("epoch/ntp_lambda", weight, epoch)
        self._epoch_ntp.zero_(); self._epoch_ntp_weighted.zero_()
        self._epoch_ntp_n = 0

    def validate(self, kimg: float, logger: StatusLogger) -> Tuple[float, float]:
        logger.event(f"[VAL] {int(kimg)} kimg — computing KID/FID …")
        metrics = self.metrics.compute(self.G_ema, self.val_loader, self.device)
        kid_mean = metrics.get("kid_mean", float("inf"))
        fid = metrics.get("fid", float("nan"))
        lines = [f"  {'metric':<16}: value"]
        for k, v in metrics.items():
            lines.append(f"  {k:<16}: {v:.5f}")
        logger.event("\n".join(lines))
        if self.writer is not None:
            for k, v in metrics.items():
                if math.isfinite(v):
                    self.writer.add_scalar(f"val/{k}", v, self.state.step)
        return kid_mean, fid

    @torch.no_grad()
    def milestone(self, kimg: float, logger: StatusLogger) -> None:
        if not _TORCHVISION_MODELS:
            return
        cfg = self.cfg
        try:
            feat = InceptionFeatures(self.device)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[milestone] feature extractor unavailable: {e}")
            return
        reals: Dict[int, List[Tensor]] = {c: [] for c in range(cfg.num_classes)}
        for imgs, lbls in self.val_loader:
            for c in range(cfg.num_classes):
                m = (lbls == c)
                if m.any():
                    reals[c].append((imgs[m] * 0.5 + 0.5).clamp(0, 1))
        for c in range(cfg.num_classes):
            name = self.class_names[c]
            if not reals[c]:
                continue
            real = torch.cat(reals[c]).to(self.device)
            fake = self.metrics.sample_for_pr(
                self.G_ema, c, min(512, 4 * real.size(0)), self.device)
            if fake.size(0) == 0:
                logger.event(f"[MILESTONE {int(kimg)}k] {name:<7} "
                             f"skipped (no finite samples)")
                continue
            rf = torch.cat([feat(real[i:i + cfg.metric_chunk])
                            for i in range(0, real.size(0), cfg.metric_chunk)])
            ff = torch.cat([feat(fake[i:i + cfg.metric_chunk])
                            for i in range(0, fake.size(0), cfg.metric_chunk)])
            prec, rec = improved_precision_recall(rf, ff, k=3)
            memo = nn_memorization(rf, ff)
            logger.event(
                f"[MILESTONE {int(kimg)}k] {name:<7} P={prec:.3f} R={rec:.3f} "
                f"memo(f->r)={memo['memo_fake_to_real_median']:.2f} "
                f"memo(r->r)={memo['memo_real_to_real_median']:.2f}")
        del feat
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def train(self, resume: Optional[str] = None) -> None:
        cfg = self.cfg
        if resume is not None and os.path.exists(resume):
            ckpt = load_checkpoint(resume, self.G, self.D, self.G_ema, self.opt_G,
                                   self.opt_D, self.lecam, self.ada, self.state,
                                   self.device)
            zf = ckpt.get("z_fixed")
            if zf is not None:
                self.z_fixed = zf
            print(f"[RESUME] from {resume} at step {self.state.step}")

        steps_per_epoch = max(1, len(self.train_loader))
        self._steps_per_epoch = steps_per_epoch
        logger = StatusLogger(cfg, cfg.total_steps, steps_per_epoch)
        loader = self._infinite(self.train_loader)
        lr = self.opt_G.param_groups[0]["lr"]

        while self.state.step < cfg.total_steps:
            real_img, labels = next(loader)
            self._current_epoch = self.state.step // steps_per_epoch
            self.train_step(real_img, labels)
            self.state.step += 1
            step = self.state.step
            kimg = step * cfg.batch_size / 1000.0

            if step % steps_per_epoch == 0 or step == cfg.total_steps:
                completed_epoch = (step - 1) // steps_per_epoch + 1
                self._finalize_epoch_ntp(completed_epoch, logger)

            if step % cfg.log_interval_steps == 0:
                stats = self._drain_acc()
                logger.live(step, stats, self.ada.p, lr, self.state)
                if self.writer is not None:
                    for k, v in stats.items():
                        self.writer.add_scalar(f"train/{k}", v, step)
                    self.writer.add_scalar("train/ada_p", self.ada.p, step)

            if self.state.nan_skips > cfg.nan_patience:
                raise RuntimeError(
                    f"[FATAL] {self.state.nan_skips} non-finite-grad skips — "
                    f"investigate (this is a bug, not a transient).")

            if step % cfg.snapshot_steps == 0:
                save_preview(self.G_ema, cfg, self.class_names, kimg,
                             self.z_fixed, self.device)
                save_compare_grids(self.G_ema, self.val_loader, cfg,
                                   self.class_names, kimg, self.device)
                kid_mean, fid = self.validate(kimg, logger)

                if math.isfinite(fid) and fid < self.state.best_fid:
                    self.state.best_fid = fid
                if step % cfg.milestone_steps == 0:
                    self.milestone(kimg, logger)

                self._save(kimg, is_best=False)
                if self.metrics.available and kid_mean < self.state.best_kid - cfg.kid_min_delta:
                    self.state.best_kid = kid_mean
                    self.state.no_improve_steps = 0
                    self._save(kimg, is_best=True)
                    logger.event(f"[BEST] new best KID={kid_mean:.5f} @ {int(kimg)} kimg "
                                 f"-> generator_best.pth")
                elif self.metrics.available:
                    self.state.no_improve_steps += cfg.snapshot_steps
                logger.event(f"[CKPT] saved latest.pth @ {int(kimg)} kimg")

                if self.metrics.available and self.state.no_improve_steps >= cfg.patience_steps:
                    logger.event(f"[EARLY STOP] no KID improvement for "
                                 f"{cfg.patience_kimg} kimg. "
                                 f"Best KID={self.state.best_kid:.5f}")
                    break

        total_time = time.time() - logger.t0
        logger.event(f"[DONE] {self.state.step} steps "
                     f"({self.state.step * cfg.batch_size / 1000.0:.0f} kimg) in "
                     f"{_fmt_time(total_time)} | best KID={self.state.best_kid:.5f} "
                     f"| best FID={self.state.best_fid:.2f}")

    def _infinite(self, loader: DataLoader):
        while True:
            for batch in loader:
                yield batch

    def _prune_periodic(self) -> None:
        cfg = self.cfg
        try:
            files = [f for f in os.listdir(cfg.checkpoint_path)
                     if f.startswith("generator_") and f.endswith("kimg.pth")]
            files.sort()
            for f in files[:-cfg.ckpt_keep]:
                try:
                    os.remove(os.path.join(cfg.checkpoint_path, f))
                except OSError:
                    pass
        except OSError:
            pass

    def _save(self, kimg: float, is_best: bool) -> None:
        cfg = self.cfg
        name = "generator_best.pth" if is_best else "latest.pth"
        save_checkpoint(os.path.join(cfg.checkpoint_path, name), cfg, self.state,
                        self.G, self.D, self.G_ema, self.opt_G, self.opt_D,
                        self.lecam, self.ada, self.class_names, self.z_fixed)
        if not is_best and self.state.step % cfg.periodic_steps == 0:
            periodic = os.path.join(cfg.checkpoint_path,
                                    f"generator_{int(kimg):05d}kimg.pth")
            save_checkpoint(periodic, cfg, self.state, self.G, self.D, self.G_ema,
                            self.opt_G, self.opt_D, self.lecam, self.ada,
                            self.class_names, self.z_fixed)
            self._prune_periodic()


# =============================================================================
# Entry point
# =============================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Class-conditional StyleGAN2-ADA + Neural Texture Preservation "
                    "for autofluorescence cell-image synthesis")
    p.add_argument("--mode", choices=["train", "generate"], default="train")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--per-class", type=int, default=None)
    p.add_argument("--data", type=str, default=None,
                   help="override train_data_path (class-folder root; default data/train)")
    p.add_argument("--output-path", type=str, default=None,
                   help="override output_path (default outputs/gan)")
    p.add_argument("--checkpoint-path", type=str, default=None,
                   help="override checkpoint_path (default checkpoints/gan)")
    p.add_argument("--log-path", type=str, default=None,
                   help="override log_path (default logs/gan)")
    p.add_argument("--no-ntp", action="store_true",
                   help="ablation: disable Neural Texture Preservation "
                        "(pure StyleGAN2-ADA baseline)")
    p.add_argument("--relativistic", action="store_true",
                   help="use optional RaGAN adversarial loss (experimental)")
    return p


def _build_config(args: argparse.Namespace) -> Config:
    overrides: Dict = {}
    if args.data is not None:
        overrides["train_data_path"] = args.data
    if args.output_path is not None:
        overrides["output_path"] = args.output_path
    if args.checkpoint_path is not None:
        overrides["checkpoint_path"] = args.checkpoint_path
    if args.log_path is not None:
        overrides["log_path"] = args.log_path
    if args.no_ntp:
        overrides["ntp_lambda"] = 0.0
    if args.relativistic:
        overrides["use_relativistic"] = True
    return Config(**overrides) if overrides else Config()


def main() -> None:
    args = build_argparser().parse_args()
    cfg = _build_config(args)
    per_class = args.per_class if args.per_class is not None else cfg.final_images_per_class

    if args.mode == "train":
        trainer = Trainer(cfg)
        trainer.train(resume=args.resume)
        # Final generation uses the BEST-KID checkpoint if one was saved, else EMA.
        best_path = os.path.join(cfg.checkpoint_path, "generator_best.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=trainer.device, weights_only=False)
            load_ema_from_checkpoint(ckpt, trainer.G_ema)
            print("[GEN] using generator_best.pth (best validation KID) for generation")
        zip_path = generate_final(trainer.G_ema, cfg, trainer.class_names,
                                  trainer.device, per_class=per_class)
        print(f"[GENERATE] wrote {zip_path}")
    else:
        device = cfg.device
        g_ema = Generator(cfg).to(device).eval()
        weights = args.weights or os.path.join(cfg.checkpoint_path, "generator_best.pth")
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        class_names = load_ema_from_checkpoint(ckpt, g_ema)
        zip_path = generate_final(g_ema, cfg, class_names, device, per_class=per_class)
        print(f"[GENERATE] wrote {zip_path}")


if __name__ == "__main__":
    main()
