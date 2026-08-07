#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gan_quality_selection.py
========================
Stage 2 of the pipeline accompanying:

    "Quality Controlled Synthetic Augmentation for AI-Enabled Label-free
     Digital Cytology of Oral Cancer Screening"

Automatic quality control of the GAN-generated autofluorescence images.
NOT every generated image is used for classifier training: every candidate
is scored for quality, diversity, and class consistency, and only the
selected subset is copied forward for synthetic augmentation. The whole
selection process is deterministic and fully automated (no manual
inspection or cherry-picking); all thresholds are defined below.

Per-image metrics:
    - DINOv2 similarity (mean of top-10 nearest real images; model size
      auto-selected from available VRAM)
    - CLIP similarity (ViT-L/14, mean of top-10 nearest real images)
    - LPIPS perceptual distance (AlexNet, vs the nearest DINO neighbour)
    - Nearest-neighbour distance (DINO embedding, Euclidean)
    - Local density (mean distance to the top-K real neighbours)
    - Class margin (DINO similarity to own class minus similarity to the
      other class) -- the class-consistency criterion
    - Outlier score (Isolation Forest / LOF, applied as a penalty)
    - Optional discriminator realism score and optional no-reference IQA
      metrics (MUSIQ, MANIQA, BRISQUE, NIQE); disabled by default

Selection safeguards:
    - Exact raw-image copies and exact GAN duplicates are excluded upfront
      by content hash.
    - Near-copies of real training images (DINO max similarity >=
      COPY_SIMILARITY_THRESHOLD) are rejected -- the memorisation safeguard.
    - Cross-class-ambiguous images (class margin <= MIN_CLASS_MARGIN) are
      rejected.
    - A greedy diversity filter removes near-duplicate synthetic images
      (pairwise DINO similarity > DIVERSITY_THRESHOLD).
    - At most TOP_PERCENT of the candidates, capped at
      MAX_SELECTED_TO_REAL_RATIO x the number of real images, are selected
      per class by the weighted composite quality score.

Post-selection distribution metrics (selected set vs real set, DINO space):
    - FID, KID, precision, recall, density, coverage

Outputs (under --output-folder):
    - <class>/                    : the selected synthetic images
    - scores.csv                  : per-image metrics + selection decision
    - summary.csv                 : per-class selection summary
    - dataset_metrics.csv         : FID/KID/P/R/D/C of the selected set
    - results/<class>/            : histograms, UMAP/t-SNE projections,
                                    heatmaps, pairplots, contact sheets
    - gan_quality_selection.log   : execution log with timing & GPU info

USAGE
-----
    python gan_quality_selection.py
    python gan_quality_selection.py --real-folder data/train \
        --gan-folder outputs/gan/final_generated \
        --output-folder outputs/selected_synthetic
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import logging
import warnings
import json
import hashlib
import time
import gc
from pathlib import Path
from typing import (
    List, Dict, Tuple, Optional, Any, Callable, Union, Set
)
from dataclasses import dataclass
from functools import wraps
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel

try:
    import lpips
    HAS_LPIPS: bool = True
except ImportError:
    HAS_LPIPS = False

try:
    import faiss
    HAS_FAISS: bool = True
    try:
        faiss.StandardGpuResources()
        HAS_FAISS_GPU: bool = True
    except Exception:
        HAS_FAISS_GPU = False
except ImportError:
    HAS_FAISS = False
    HAS_FAISS_GPU = False

try:
    import umap
    HAS_UMAP: bool = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.manifold import TSNE
    from scipy.linalg import sqrtm
    HAS_SKLEARN: bool = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pyiqa
    HAS_PYIQA: bool = True
except ImportError:
    HAS_PYIQA = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# CONFIGURATION
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
# Defaults are relative to the repository; override via the CLI flags below.
REAL_FOLDER: str = str(PROJECT_ROOT / "data" / "train")
GAN_FOLDER: str = str(PROJECT_ROOT / "outputs" / "gan" / "final_generated")
OUTPUT_FOLDER: str = str(PROJECT_ROOT / "outputs" / "selected_synthetic")

TOP_PERCENT: float = 0.40
MAX_SELECTED_TO_REAL_RATIO: float = 0.5
DIVERSITY_THRESHOLD: float = 0.98
COPY_SIMILARITY_THRESHOLD: float = 0.995
MIN_CLASS_MARGIN: float = 0.0

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE: int = 32
NUM_WORKERS: int = 0
PREFETCH_FACTOR: int = 2

ENABLE_FP16: bool = torch.cuda.is_available()
ENABLE_PIN_MEMORY: bool = torch.cuda.is_available()
ENABLE_CACHE: bool = True
ENABLE_FAISS: bool = HAS_FAISS
ENABLE_UMAP: bool = HAS_UMAP
ENABLE_TSNE: bool = HAS_SKLEARN
ENABLE_DIVERSITY: bool = True
ENABLE_LOCAL_DENSITY: bool = True
ENABLE_OUTLIERS: bool = True
OUTLIER_METHOD: str = "isolation_forest"
OUTLIER_PENALTY: float = 0.25
ENABLE_DATASET_METRICS: bool = True

ENABLE_MUSIQ: bool = False
ENABLE_MANIQA: bool = False
ENABLE_BRISQUE: bool = False
ENABLE_NIQE: bool = False

USE_DISCRIMINATOR: bool = False
DISCRIMINATOR_PATH: str = "discriminator.pth"

DINO_AUTO_SELECT: bool = True
DINO_MODEL_FALLBACK: str = "facebook/dinov2-base"
DINO_MODEL_PREFERRED: str = "facebook/dinov2-large"
DINO_VRAM_THRESHOLD_GB: float = 10.0
LOCAL_DENSITY_K: int = 10
SEED: int = 42

_DESIRED_WEIGHTS: Dict[str, float] = {
    "dino": 0.20,
    "clip": 0.12,
    "disc": 0.10,
    "lpips": 0.08,
    "nn_dist": 0.07,
    "local_density": 0.03,
    "class_margin": 0.25,
    "musiq": 0.05,
    "maniqa": 0.05,
    "brisque": 0.03,
    "niqe": 0.02,
}

# =====================================================================
# LOGGING & TIMING
# =====================================================================
_LOG_FMT = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logging() -> logging.Logger:
    """Console logger, created at import time. The file handler is attached in
    main() once OUTPUT_FOLDER is final (it may be overridden on the CLI)."""
    logger = logging.getLogger("GAN_Quality")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(_LOG_FMT))
        logger.addHandler(ch)
    return logger


def attach_log_file(log_file: str = "gan_quality_selection.log") -> None:
    """Attach the run log file inside OUTPUT_FOLDER (idempotent)."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    fh = logging.FileHandler(
        os.path.join(OUTPUT_FOLDER, log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    logger.addHandler(fh)

logger = setup_logging()

def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper

@contextmanager
def gpu_memory_tracker(label: str = ""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1024**3
        yield
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"[{label}] GPU memory: {start_mem:.2f} -> {end_mem:.2f} GB "
                    f"(delta {end_mem - start_mem:+.2f} GB)")
    else:
        yield

# =====================================================================
# GPU & MEMORY UTILITIES
# =====================================================================
def get_gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    if not torch.cuda.is_available():
        return info

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    free = total - allocated

    info.update({
        "available": True,
        "name": props.name,
        "total_gb": total,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free,
        "multi_processor_count": props.multi_processor_count,
    })
    return info

def select_dino_model() -> str:
    if not DINO_AUTO_SELECT or not torch.cuda.is_available():
        return DINO_MODEL_FALLBACK
    gpu = get_gpu_info()
    if gpu.get("free_gb", 0) >= DINO_VRAM_THRESHOLD_GB:
        logger.info(f"GPU free VRAM {gpu['free_gb']:.1f} GB >= threshold {DINO_VRAM_THRESHOLD_GB} GB → using {DINO_MODEL_PREFERRED}")
        return DINO_MODEL_PREFERRED
    else:
        logger.info(f"GPU free VRAM {gpu['free_gb']:.1f} GB < threshold {DINO_VRAM_THRESHOLD_GB} GB → using {DINO_MODEL_FALLBACK}")
        return DINO_MODEL_FALLBACK

# =====================================================================
# UTILITIES
# =====================================================================
def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _compute_content_hash(paths: List[str]) -> str:
    hasher = hashlib.sha256()
    for p in sorted(paths):
        hasher.update(p.encode("utf-8"))
        try:
            st = os.stat(p)
            hasher.update(f"{st.st_size}:{st.st_mtime:.6f}".encode())
        except OSError:
            hasher.update(b"missing")
    return hasher.hexdigest()[:24]

def redistribute_weights(desired: Dict[str, float], availability: Dict[str, bool]) -> Dict[str, float]:
    available = {k: v for k, v in desired.items() if availability.get(k, False) and v > 0}
    unavailable_total = sum(v for k, v in desired.items() if not availability.get(k, False) or v <= 0)
    if not available:
        return {}
    total_available = sum(available.values())
    if total_available == 0:
        return {k: 1.0 / len(available) for k in available}
    for k in available:
        available[k] += unavailable_total * (available[k] / total_available)
    total = sum(available.values())
    return {k: round(v / total, 6) for k, v in available.items()}

def min_max_norm(arr: np.ndarray, invert: bool = False) -> np.ndarray:
    if arr.size == 0:
        return arr
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return np.ones_like(arr) * 0.5
    normed = (arr - vmin) / (vmax - vmin)
    return 1.0 - normed if invert else normed

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default

# =====================================================================
# IMAGE LOADING 
# =====================================================================
_VALID_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")

def load_image_safe(path: str, preserve_icc: bool = False) -> Image.Image:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            icc = img.info.get("icc_profile") if preserve_icc else None
            img = img.convert("RGB")
            if preserve_icc and icc:
                img.info["icc_profile"] = icc
        return img
    except Exception as exc:
        raise RuntimeError(f"Failed to load image {path}: {exc}") from exc

def image_content_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

class ImageFolderSafe(Dataset):
    def __init__(self, folder_path: str, transform: Optional[Callable] = None) -> None:
        self.folder_path = Path(folder_path)
        self.transform = transform
        self.image_paths: List[str] = []
        self.corrupted: List[str] = []
        
        logger.info(f"Scanning {folder_path} ...")
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[-1].lower()
                if ext in _VALID_EXTS:
                    path = os.path.join(root, file)
                    if self._is_valid_image(path):
                        self.image_paths.append(path)
                    else:
                        self.corrupted.append(path)
                        logger.warning(f"Skipping corrupted image: {path}")
        self.image_paths.sort()
        if self.corrupted:
            logger.warning(f"Total corrupted images skipped: {len(self.corrupted)}")

    @staticmethod
    def _is_valid_image(path: str) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        try:
            img = load_image_safe(path)
            if self.transform:
                img = self.transform(img)
            return img, path
        except Exception as exc:
            logger.error(f"Error loading {path}: {exc}")
            return torch.zeros(3, 224, 224), path

def list_valid_image_paths(folder: str) -> List[str]:
    return ImageFolderSafe(folder).image_paths

def gan_only_paths(real_paths: List[str], combined_paths: List[str]) -> Tuple[List[str], Dict[str, int]]:
    real_hashes = {image_content_hash(path) for path in real_paths}
    candidates: List[str] = []
    seen_candidate_hashes: Set[str] = set()
    raw_copies = 0
    duplicate_gan = 0
    for path in combined_paths:
        digest = image_content_hash(path)
        if digest in real_hashes:
            raw_copies += 1
        elif digest in seen_candidate_hashes:
            duplicate_gan += 1
        else:
            seen_candidate_hashes.add(digest)
            candidates.append(path)
    return candidates, {
        "raw_copies_excluded": raw_copies,
        "exact_gan_duplicates_excluded": duplicate_gan,
    }

class PathListDataset(Dataset):
    def __init__(self, paths: List[str], transform: Optional[Callable] = None) -> None:
        self.paths = paths
        self.transform = transform
    def __len__(self) -> int:
        return len(self.paths)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = load_image_safe(path)
        if self.transform:
            img = self.transform(img)
        return img, path

def make_inference_loader(dataset: Dataset) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "shuffle": False,
        "pin_memory": ENABLE_PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)

class LetterboxResize:
    def __init__(self, target_size: int = 256, fill: Tuple[int, int, int] = (128, 128, 128)) -> None:
        self.target_size = target_size
        self.fill = fill
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        result = Image.new("RGB", (self.target_size, self.target_size), self.fill)
        pad_w = (self.target_size - new_w) // 2
        pad_h = (self.target_size - new_h) // 2
        result.paste(img, (pad_w, pad_h))
        return result

class LPIPSPairDataset(Dataset):
    def __init__(self, gen_paths: List[str], real_paths: List[str], transform: Optional[Callable] = None) -> None:
        self.gen_paths = gen_paths
        self.real_paths = real_paths
        self.transform = transform or transforms.Compose([
            LetterboxResize(target_size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    def __len__(self) -> int:
        return len(self.gen_paths)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        gen_img = load_image_safe(self.gen_paths[idx])
        real_img = load_image_safe(self.real_paths[idx])
        return (self.transform(gen_img), self.transform(real_img), self.gen_paths[idx])

# =====================================================================
# SMART CACHE SYSTEM
# =====================================================================
class SmartCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.hits: int = 0
        self.misses: int = 0

    def _meta_path(self, cache_name: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_name}.meta.json")
    def _data_path(self, cache_name: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_name}.pt")

    def load(self, cache_name: str, paths: List[str], model_name: str, embed_dim: int, preprocess_cfg: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, List[str]]]:
        data_path = self._data_path(cache_name)
        meta_path = self._meta_path(cache_name)
        if not (os.path.exists(data_path) and os.path.exists(meta_path)):
            self.misses += 1
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.misses += 1
            return None

        current_meta = {
            "content_hash": _compute_content_hash(paths),
            "image_count": len(paths),
            "model_name": model_name,
            "embed_dim": embed_dim,
            "preprocess_cfg": preprocess_cfg,
        }
        for key, val in current_meta.items():
            if meta.get(key) != val:
                logger.info(f"Cache invalidation [{cache_name}]: '{key}' changed")
                self.misses += 1
                return None
        try:
            data = torch.load(data_path, map_location="cpu")
            self.hits += 1
            logger.info(f"Cache hit [{cache_name}]: {len(paths)} embeddings loaded")
            return data["embeddings"], data["paths"]
        except Exception as exc:
            logger.warning(f"Cache load failed [{cache_name}]: {exc}")
            self.misses += 1
            return None

    def save(self, cache_name: str, embeddings: torch.Tensor, paths: List[str], model_name: str, embed_dim: int, preprocess_cfg: Dict[str, Any]) -> None:
        data_path = self._data_path(cache_name)
        meta_path = self._meta_path(cache_name)
        meta = {
            "content_hash": _compute_content_hash(paths),
            "image_count": len(paths),
            "model_name": model_name,
            "embed_dim": embed_dim,
            "preprocess_cfg": preprocess_cfg,
        }
        torch.save({"embeddings": embeddings, "paths": paths}, data_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def get_stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

# =====================================================================
# EMBEDDING EXTRACTORS
# =====================================================================
class HFProcessorTransform:
    def __init__(self, model_name: str, model_type: str, processor: Any = None):
        self.model_name = model_name
        self.model_type = model_type
        self.processor = processor
    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.processor is None:
            if self.model_type == "dino":
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            else:
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
        out = self.processor(images=img, return_tensors="pt")
        return out["pixel_values"].squeeze(0)

class ModelRegistry:
    _models: Dict[str, Any] = {}
    _processors: Dict[str, Any] = {}
    @classmethod
    def get(cls, key: str):
        return cls._models.get(key), cls._processors.get(key)
    @classmethod
    def set(cls, key: str, model: Any, processor: Any) -> None:
        cls._models[key] = model
        cls._processors[key] = processor

class EmbeddingExtractor:
    def __init__(self, model_type: str, device: str, use_fp16: bool, cache: SmartCache) -> None:
        self.model_type = model_type
        self.device = device
        self.use_fp16 = use_fp16
        self.cache = cache
        self.model: nn.Module
        self.processor: Any
        self.model_name: str
        self.embed_dim: int
        self._load_model()

    def _load_model(self) -> None:
        cached_model, cached_processor = ModelRegistry.get(self.model_type)
        if cached_model is not None:
            logger.info(f"Reusing cached {self.model_type.upper()} model")
            self.model = cached_model
            self.processor = cached_processor
            self.model_name = getattr(self, "_model_name", "cached")
            self.embed_dim = getattr(self, "_embed_dim", 768)
            return

        logger.info(f"Loading {self.model_type.upper()} model ...")
        if self.model_type == "dino":
            self.model_name = select_dino_model()
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        else:
            self.model_name = "openai/clip-vit-large-patch14"
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name)
            
        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            if self.model_type == "dino":
                out = self.model(pixel_values=dummy)
                self.embed_dim = out.last_hidden_state.shape[-1]
            else:
                out = self.model.get_image_features(pixel_values=dummy)
                if isinstance(out, torch.Tensor): self.embed_dim = out.shape[-1]
                elif hasattr(out, "image_embeds"): self.embed_dim = out.image_embeds.shape[-1]
                elif hasattr(out, "pooler_output"): self.embed_dim = out.pooler_output.shape[-1]
                else: self.embed_dim = out[0].shape[-1]

        ModelRegistry.set(self.model_type, self.model, self.processor)
        logger.info(f"{self.model_type.upper()} loaded: {self.model_name}, embed_dim={self.embed_dim}")

    def _make_transform(self) -> Callable:
        processor = self.processor if NUM_WORKERS == 0 else None
        return HFProcessorTransform(self.model_name, self.model_type, processor)

    def _preprocess_config(self) -> Dict[str, Any]:
        cfg = self.processor.to_dict() if hasattr(self.processor, "to_dict") else {"model": self.model_name}
        # CRITICAL FIX: Round-trip via JSON to normalize tuple to list and avoid false cache misses
        return json.loads(json.dumps(cfg))

    @timed
    def extract(self, data_source: Union[str, List[str]], cache_name: str) -> Tuple[torch.Tensor, List[str]]:
        if isinstance(data_source, str):
            ds = ImageFolderSafe(data_source)
            paths = ds.image_paths
        else:
            paths = list(data_source)

        if ENABLE_CACHE:
            cached = self.cache.load(cache_name, paths, self.model_name, self.embed_dim, self._preprocess_config())
            if cached is not None:
                return cached

        logger.info(f"Extracting {self.model_type} embeddings ({len(paths)} images) ...")
        transform = self._make_transform()
        dataset = ImageFolderSafe(data_source, transform=transform) if isinstance(data_source, str) else PathListDataset(paths, transform=transform)
        loader = make_inference_loader(dataset)

        all_embs: List[torch.Tensor] = []
        all_paths: List[str] = []

        with torch.no_grad():
            for imgs, batch_paths in tqdm(loader, desc=f"Ext. {self.model_type.upper()}"):
                imgs = imgs.to(self.device, non_blocking=ENABLE_PIN_MEMORY)
                with torch.autocast(device_type="cuda", enabled=self.use_fp16 and self.device == "cuda"):
                    if self.model_type == "dino":
                        out = self.model(pixel_values=imgs)
                        embs = out.last_hidden_state[:, 0, :]
                    else:
                        out = self.model.get_image_features(pixel_values=imgs)
                        if isinstance(out, torch.Tensor): embs = out
                        elif hasattr(out, "image_embeds"): embs = out.image_embeds
                        elif hasattr(out, "pooler_output"): embs = out.pooler_output
                        else: embs = out[0]
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs.cpu())
                all_paths.extend(batch_paths)

        if not all_embs:
            return torch.empty(0), []
        final_embs = torch.cat(all_embs, dim=0)

        if ENABLE_CACHE:
            self.cache.save(cache_name, final_embs, all_paths, self.model_name, self.embed_dim, self._preprocess_config())
        return final_embs, all_paths

# =====================================================================
# NEAREST-NEIGHBOUR SEARCH
# =====================================================================
class ReusableNearestNeighborSearch:
    def __init__(self, use_faiss: bool = ENABLE_FAISS) -> None:
        self.use_faiss = use_faiss and HAS_FAISS
        self.use_faiss_gpu = self.use_faiss and HAS_FAISS_GPU
        self._index_cache: Dict[str, Any] = {}
        self._backend = self._detect_backend()
        logger.info(f"NN backend selected: {self._backend}")

    def _detect_backend(self) -> str:
        if self.use_faiss_gpu: return "faiss_gpu"
        elif self.use_faiss: return "faiss_cpu"
        elif HAS_SKLEARN: return "sklearn"
        else: return "torch"

    def _cache_key(self, db: torch.Tensor) -> str:
        h = hashlib.sha256()
        h.update(f"shape={tuple(db.shape)};dtype={db.dtype}".encode("ascii"))
        if db.numel() > 0:
            h.update(db[0, :8].cpu().numpy().tobytes())
            h.update(db[-1, :8].cpu().numpy().tobytes())
            mid = db.shape[0] // 2
            h.update(db[mid, :8].cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def _build_index(self, db: torch.Tensor) -> Any:
        key = self._cache_key(db)
        if key in self._index_cache:
            return self._index_cache[key]
        dim = db.shape[1]
        db_np = db.cpu().numpy().astype("float32")

        if self._backend == "faiss_gpu":
            try:
                res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatIP(dim)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                gpu_index.add(db_np)
                self._index_cache[key] = gpu_index
                return gpu_index
            except Exception as exc:
                logger.warning(f"FAISS GPU failed ({exc}), falling back to CPU")
                self._backend = "faiss_cpu"

        if self._backend == "faiss_cpu":
            index = faiss.IndexFlatIP(dim)
            index.add(db_np)
            self._index_cache[key] = index
            return index

        if self._backend == "sklearn":
            nn = NearestNeighbors(n_neighbors=min(10, db.shape[0]), metric="cosine", n_jobs=-1)
            nn.fit(db_np)
            self._index_cache[key] = nn
            return nn

        self._index_cache[key] = db
        return db

    def search(self, query: torch.Tensor, db: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        k = min(k, db.shape[0])
        index = self._build_index(db)
        if self._backend.startswith("faiss"):
            q_np = query.cpu().numpy().astype("float32")
            distances, indices = index.search(q_np, k)
            return torch.from_numpy(distances), torch.from_numpy(indices)
        elif self._backend == "sklearn":
            q_np = query.cpu().numpy().astype("float32")
            distances, indices = index.kneighbors(q_np, n_neighbors=k)
            similarities = 1.0 - distances
            return torch.from_numpy(similarities), torch.from_numpy(indices)
        else:
            query_dev = query.to(DEVICE)
            db_dev = db.to(DEVICE)
            sims = torch.mm(query_dev, db_dev.t())
            topk_sims, topk_idx = torch.topk(sims, k=k, dim=-1, largest=True)
            return topk_sims.cpu(), topk_idx.cpu()

# =====================================================================
# OPTIONAL METRICS
# =====================================================================
class IQAMetrics:
    def __init__(self, device: str) -> None:
        self.device = device
        self._models: Dict[str, Any] = {}
        self.availability: Dict[str, bool] = {"musiq": False, "maniqa": False, "brisque": False, "niqe": False}
        self._load_models()

    def _load_models(self) -> None:
        if not HAS_PYIQA: return
        metric_map = {
            "musiq": (ENABLE_MUSIQ, "musiq"),
            "maniqa": (ENABLE_MANIQA, "maniqa"),
            "brisque": (ENABLE_BRISQUE, "brisque"),
            "niqe": (ENABLE_NIQE, "niqe"),
        }
        for key, (enabled, metric_name) in metric_map.items():
            if not enabled: continue
            try:
                model = pyiqa.create_metric(metric_name, device=self.device)
                self._models[key] = model
                self.availability[key] = True
            except Exception as exc:
                logger.warning(f"Failed to load IQA metric {metric_name}: {exc}")

    def compute(self, paths: List[str]) -> Dict[str, np.ndarray]:
        results: Dict[str, np.ndarray] = {}
        if not self._models: return results
        images, valid_paths = [], []
        transform = transforms.Compose([LetterboxResize(target_size=224), transforms.ToTensor()])
        for p in paths:
            try:
                images.append(transform(load_image_safe(p)))
                valid_paths.append(p)
            except Exception: pass
        if not images: return results
        batch = torch.stack(images).to(self.device, non_blocking=ENABLE_PIN_MEMORY)
        for key, model in self._models.items():
            try:
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", enabled=ENABLE_FP16 and self.device == "cuda"):
                        scores = model(batch)
                results[key] = scores.view(-1).cpu().numpy()
            except Exception:
                results[key] = np.zeros(len(valid_paths))
        return results

class DistributionMetrics:
    @staticmethod
    def compute_all(real_embs: np.ndarray, gen_embs: np.ndarray, k: int = 3) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if len(real_embs) < 2 or len(gen_embs) < 2:
            return metrics
        metrics["fid"] = DistributionMetrics._compute_fid(real_embs, gen_embs)
        metrics["kid"] = DistributionMetrics._compute_kid(real_embs, gen_embs)
        metrics.update(DistributionMetrics._compute_prdc(real_embs, gen_embs, k=k))
        return metrics

    @staticmethod
    def _compute_fid(real_embs: np.ndarray, gen_embs: np.ndarray) -> float:
        mu_r, sigma_r = real_embs.mean(axis=0), np.cov(real_embs, rowvar=False)
        mu_g, sigma_g = gen_embs.mean(axis=0), np.cov(gen_embs, rowvar=False)
        diff = mu_r - mu_g
        covmean = sqrtm(sigma_r.dot(sigma_g))
        if np.iscomplexobj(covmean): covmean = covmean.real
        fid = float(diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean))
        return float(max(fid, 0.0))

    @staticmethod
    def _compute_kid(real_embs: np.ndarray, gen_embs: np.ndarray, max_subset_size: int = 1000) -> float:
        n = min(len(real_embs), len(gen_embs), max_subset_size)
        if n < 10: return 0.0
        if len(real_embs) > n: real_embs = real_embs[np.random.choice(len(real_embs), n, replace=False)]
        if len(gen_embs) > n: gen_embs = gen_embs[np.random.choice(len(gen_embs), n, replace=False)]
        d = real_embs.shape[1]
        rr = real_embs @ real_embs.T / d + 1.0
        gg = gen_embs @ gen_embs.T / d + 1.0
        rg = real_embs @ gen_embs.T / d + 1.0
        mmd = (rr ** 3).mean() + (gg ** 3).mean() - 2 * (rg ** 3).mean()
        return float(max(mmd, 0.0))

    @staticmethod
    def _compute_prdc(real_embs: np.ndarray, gen_embs: np.ndarray, k: int = 3) -> Dict[str, float]:
        k_real = min(k, len(real_embs) - 1)
        k_gen = min(k, len(gen_embs) - 1)
        if k_real < 1 or k_gen < 1:
            return {"precision": np.nan, "recall": np.nan, "density": np.nan, "coverage": np.nan}
        
        real_t = torch.from_numpy(real_embs)
        gen_t = torch.from_numpy(gen_embs)
        
        # CRITICAL FIX: Memory-efficient distance computation using pure PyTorch cdist
        # Avoids OOM risk from standard NxMxD tensor broadcasts.
        rr = torch.cdist(real_t, real_t).numpy()
        gg = torch.cdist(gen_t, gen_t).numpy()
        gr = torch.cdist(gen_t, real_t).numpy()
        
        np.fill_diagonal(rr, np.inf)
        np.fill_diagonal(gg, np.inf)
        
        real_radii = np.partition(rr, k_real - 1, axis=1)[:, k_real - 1]
        gen_radii = np.partition(gg, k_gen - 1, axis=1)[:, k_gen - 1]

        precision = (gr <= real_radii[None, :]).any(axis=1).mean()
        recall = (gr <= gen_radii[:, None]).any(axis=0).mean()
        density = (gr <= real_radii[None, :]).sum(axis=1).mean() / k_real
        coverage = (gr.min(axis=0) <= real_radii).mean()

        return {"precision": float(precision), "recall": float(recall), "density": float(density), "coverage": float(coverage)}

class OutlierDetector:
    def __init__(self, method: str = OUTLIER_METHOD) -> None:
        self.method = method
        self.available = HAS_SKLEARN
    def detect(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.available or not ENABLE_OUTLIERS or len(embeddings) < 10:
            return np.zeros(len(embeddings))
        try:
            if self.method == "isolation_forest":
                clf = IsolationForest(contamination="auto", random_state=SEED, n_jobs=-1)
                preds = clf.fit_predict(embeddings)
                scores = (preds == -1).astype(float)
            else:
                clf = LocalOutlierFactor(n_neighbors=min(20, len(embeddings) - 1), n_jobs=-1)
                preds = clf.fit_predict(embeddings)
                scores = (preds == -1).astype(float)
            return scores
        except Exception as exc:
            logger.warning(f"Outlier detection failed: {exc}")
            return np.zeros(len(embeddings))

class DiscriminatorLoader:
    @staticmethod
    def load(path: str, device: str) -> Optional[nn.Module]:
        if not os.path.exists(path): return None
        try:
            checkpoint = torch.load(path, map_location=device)
        except Exception: return None
        if isinstance(checkpoint, nn.Module): return checkpoint
        if isinstance(checkpoint, dict):
            state_dict = DiscriminatorLoader._extract_state_dict(checkpoint)
            if state_dict is None: return None
            arch = DiscriminatorLoader._detect_architecture(state_dict)
            model = DiscriminatorLoader._build_model(arch, state_dict)
            if model is None: return None
            try:
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                return model
            except Exception: return None
        return None

    @staticmethod
    def _extract_state_dict(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key in ("state_dict", "model", "discriminator", "netD"):
            if key in checkpoint and isinstance(checkpoint[key], dict): return checkpoint[key]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()): return checkpoint
        return None

    @staticmethod
    def _detect_architecture(state_dict: Dict[str, Any]) -> str:
        keys = set(state_dict.keys())
        if any(p in " ".join(keys) for p in ["fromrgb", "b", "conv0", "conv1"]): return "stylegan2"
        if any(p in k for k in keys for p in ["main.0", "main.1", "features"]): return "dcgan"
        if len(keys) > 10 and all("model" in k or "conv" in k for k in list(keys)[:10]): return "patchgan"
        return "generic"

    @staticmethod
    def _build_model(arch: str, state_dict: Dict[str, Any]) -> Optional[nn.Module]:
        return DiscriminatorLoader._build_generic_discriminator()

    @staticmethod
    def _build_generic_discriminator(nc: int = 3, ndf: int = 64, img_size: int = 256) -> nn.Module:
        class _GenericDiscriminator(nn.Module):
            def __init__(self, nc: int, ndf: int, img_size: int) -> None:
                super().__init__()
                layers: List[nn.Module] = []
                in_ch, out_ch, curr_size = nc, ndf, img_size
                while curr_size > 4:
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.LeakyReLU(0.2, inplace=True),
                    ))
                    in_ch, out_ch, curr_size = out_ch, min(out_ch * 2, 512), curr_size // 2
                layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False))
                self.main = nn.Sequential(*layers)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.main(x)
                return out.view(x.size(0), -1)
        return _GenericDiscriminator(nc, ndf, img_size)

class LPIPSComputer:
    def __init__(self, device: str, use_fp16: bool) -> None:
        self.device = device
        self.use_fp16 = use_fp16
        self.model = lpips.LPIPS(net="alex") if HAS_LPIPS else None
        if self.model is not None:
            self.model.eval()
            self.model.to(device)
    def is_available(self) -> bool:
        return self.model is not None
    def compute(self, gen_paths: List[str], real_paths: List[str]) -> np.ndarray:
        if not self.is_available(): return np.zeros(len(gen_paths))
        dataset = LPIPSPairDataset(gen_paths, real_paths)
        loader = make_inference_loader(dataset)
        scores: List[float] = []
        with torch.no_grad():
            for gen_img, real_img, _ in tqdm(loader, desc="LPIPS"):
                gen_img, real_img = gen_img.to(self.device, non_blocking=ENABLE_PIN_MEMORY), real_img.to(self.device, non_blocking=ENABLE_PIN_MEMORY)
                with torch.autocast(device_type="cuda", enabled=self.use_fp16 and self.device == "cuda"):
                    dist = self.model(gen_img, real_img)
                scores.extend(dist.view(-1).cpu().numpy().tolist())
        return np.array(scores)

# =====================================================================
# METRIC COMPUTER
# =====================================================================
class MetricComputer:
    def __init__(self, dino_extractor: EmbeddingExtractor, clip_extractor: EmbeddingExtractor, nn_search: ReusableNearestNeighborSearch, lpips_computer: LPIPSComputer, disc_model: Optional[nn.Module], iqa_metrics: IQAMetrics, outlier_detector: OutlierDetector, device: str, use_fp16: bool) -> None:
        self.dino_ext = dino_extractor
        self.clip_ext = clip_extractor
        self.nn_search = nn_search
        self.lpips = lpips_computer
        self.disc_model = disc_model
        self.iqa = iqa_metrics
        self.outlier = outlier_detector
        self.device = device
        self.use_fp16 = use_fp16

    @timed
    def compute_all(self, real_paths_input: List[str], gan_paths_input: List[str], cls: str, other_real_paths: Optional[List[str]] = None) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, List[str], List[str]]:
        logger.info(f"[{cls}] Extracting DINO embeddings ...")
        dino_real, real_paths = self.dino_ext.extract(real_paths_input, f"dino_real_{cls}")
        dino_gan, gan_paths = self.dino_ext.extract(gan_paths_input, f"dino_gan_{cls}")
        logger.info(f"[{cls}] Extracting CLIP embeddings ...")
        clip_real, _ = self.clip_ext.extract(real_paths_input, f"clip_real_{cls}")
        clip_gan, _ = self.clip_ext.extract(gan_paths_input, f"clip_gan_{cls}")
        
        n_gen, n_real = len(gan_paths), len(real_paths)
        if n_gen == 0 or n_real == 0:
            raise ValueError(f"Not enough images in class {cls} (real={n_real}, gen={n_gen})")

        dino_mean, dino_max, nn_dist = self._compute_topk_similarity(dino_gan, dino_real, k=10)
        clip_mean, clip_max, _ = self._compute_topk_similarity(clip_gan, clip_real, k=10)

        other_mean, other_max, class_margin = np.zeros(n_gen), np.zeros(n_gen), np.zeros(n_gen)
        if other_real_paths:
            dino_other, _ = self.dino_ext.extract(other_real_paths, f"dino_real_other_{cls}")
            if len(dino_other):
                other_mean, other_max, _ = self._compute_topk_similarity(dino_gan, dino_other, k=10)
                class_margin = dino_mean - other_mean

        local_density = self._compute_local_density(dino_gan, dino_real, k=LOCAL_DENSITY_K) if ENABLE_LOCAL_DENSITY else np.zeros(n_gen)
        lpips_scores = self.lpips.compute(gan_paths, self._get_nearest_real_paths(dino_gan, dino_real, real_paths)) if self.lpips.is_available() else np.zeros(n_gen)
        disc_scores = self._compute_discriminator(gan_paths) if self.disc_model is not None else np.zeros(n_gen)
        iqa_results = self.iqa.compute(gan_paths) if self.iqa._models else {}
        outlier_scores = self.outlier.detect(dino_gan.numpy()) if ENABLE_OUTLIERS else np.zeros(n_gen)

        df = pd.DataFrame({
            "Image": [Path(p).name for p in gan_paths],
            "Path": gan_paths,
            "Class": cls,
            "DINO_Similarity": dino_mean,
            "DINO_Max_Similarity": dino_max,
            "Other_Class_DINO_Similarity": other_mean,
            "Other_Class_DINO_Max_Similarity": other_max,
            "Class_Margin": class_margin,
            "CLIP_Similarity": clip_mean,
            "CLIP_Max_Similarity": clip_max,
            "LPIPS": lpips_scores,
            "Discriminator_Score": disc_scores,
            "Nearest_Neighbor_Distance": nn_dist,
            "Local_Density": local_density,
            "Outlier_Score": outlier_scores,
        })
        for key, scores in iqa_results.items(): df[key.upper()] = scores
        df["original_idx"] = np.arange(n_gen)
        return df, dino_real, dino_gan, real_paths, gan_paths

    def _compute_topk_similarity(self, query: torch.Tensor, db: torch.Tensor, k: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        topk_sims, _ = self.nn_search.search(query, db, k=min(k, db.shape[0]))
        return topk_sims.mean(dim=-1).numpy(), topk_sims[:, 0].numpy(), np.sqrt(np.clip(2.0 - 2.0 * topk_sims[:, 0].numpy(), 0.0, None))

    def _compute_local_density(self, query: torch.Tensor, db: torch.Tensor, k: int = 10) -> np.ndarray:
        topk_sims, _ = self.nn_search.search(query, db, k=min(k, db.shape[0]))
        return np.sqrt(np.clip(2.0 - 2.0 * topk_sims.numpy(), 0.0, None)).mean(axis=1)

    def _get_nearest_real_paths(self, query: torch.Tensor, db: torch.Tensor, db_paths: List[str]) -> List[str]:
        _, indices = self.nn_search.search(query, db, k=1)
        return [db_paths[idx] for idx in indices.squeeze(-1).tolist()]

    def _compute_discriminator(self, paths: List[str]) -> np.ndarray:
        transform = transforms.Compose([LetterboxResize(target_size=256), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
        loader = make_inference_loader(PathListDataset(paths, transform=transform))
        self.disc_model.eval()
        self.disc_model.to(self.device)
        scores: List[float] = []
        with torch.no_grad():
            for img, _ in tqdm(loader, desc="Disc"):
                with torch.autocast(device_type="cuda", enabled=self.use_fp16 and self.device == "cuda"):
                    out = self.disc_model(img.to(self.device, non_blocking=ENABLE_PIN_MEMORY))
                    if out.dim() > 1:
                        out = out.view(out.size(0), -1)
                        out = F.softmax(out, dim=1)[:, 0] if out.shape[1] > 1 else out.squeeze(1)
                    if out.max() > 1.0 or out.min() < 0.0: out = torch.sigmoid(out)
                scores.extend(out.clamp(0.0, 1.0).view(-1).cpu().numpy().tolist())
        return np.array(scores)

def filter_diversity(df_sorted: pd.DataFrame, embs: torch.Tensor, nn_search: ReusableNearestNeighborSearch, threshold: float = DIVERSITY_THRESHOLD) -> pd.DataFrame:
    if len(df_sorted) == 0 or not ENABLE_DIVERSITY: return df_sorted
    logger.info("Applying diversity filter ...")
    ordered_indices = df_sorted["original_idx"].values
    n = len(ordered_indices)
    kept_mask = np.zeros(n, dtype=bool)
    kept_embs_list: List[torch.Tensor] = []
    embs_device = embs.to(DEVICE)
    for i in tqdm(range(n), desc="Diversity"):
        curr_emb = embs_device[ordered_indices[i]].unsqueeze(0)
        if not kept_embs_list:
            kept_mask[i] = True
            kept_embs_list.append(curr_emb)
            continue
        max_sim = torch.mm(curr_emb, torch.cat(kept_embs_list, dim=0).t()).max().item()
        if max_sim <= threshold:
            kept_mask[i] = True
            kept_embs_list.append(curr_emb)
    df_filtered = df_sorted.iloc[kept_mask].copy()
    logger.info(f"Retained {len(df_filtered)} / {len(df_sorted)} images after diversity filter")
    return df_filtered

# =====================================================================
# VISUALISATIONS
# =====================================================================
def create_histograms(df: pd.DataFrame, out_dir: str) -> None:
    logger.info("Generating histograms ...")
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    metrics = [
        ("DINO_Similarity", "dino_histogram.png"),
        ("CLIP_Similarity", "clip_histogram.png"),
        ("LPIPS", "lpips_histogram.png"),
        ("Discriminator_Score", "discriminator_histogram.png"),
        ("Final_Score", "final_score_histogram.png"),
        ("Local_Density", "local_density_histogram.png"),
    ]
    for col, fname in metrics:
        try:
            if col not in df.columns or df[col].isnull().all():
                continue
            plot_df = df[np.isfinite(df[col].to_numpy(dtype=float))].copy()
            if plot_df.empty or plot_df[col].nunique(dropna=True) < 2:
                logger.info(f"Skipping {col} histogram: metric is constant or empty")
                continue
            
            # Construct dictionary safely to avoid Seaborn hue mapping crash
            palette = {}
            if True in plot_df["Selected"].values: palette[True] = "green"
            if False in plot_df["Selected"].values: palette[False] = "red"

            # CRITICAL FIX: Safe KDE check to avoid Scipy LinAlgError
            kde_safe = True
            for sel_val in plot_df["Selected"].unique():
                sub = plot_df[plot_df["Selected"] == sel_val][col]
                if len(sub) < 2 or sub.var() < 1e-5 or sub.nunique() < 2:
                    kde_safe = False
                    break
            
            try:
                plt.figure(figsize=(8, 5))
                sns.histplot(
                    data=plot_df, x=col, hue="Selected", kde=kde_safe,
                    bins=min(30, max(10, plot_df[col].nunique())),
                    palette=palette
                )
                plt.title(f"{col} Distribution")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, fname), dpi=150)
            except Exception as plot_err:
                logger.warning(f"Failed {col} with KDE={kde_safe}: {plot_err}. Trying fallback...")
                plt.close('all')
                plt.figure(figsize=(8, 5))
                sns.histplot(data=plot_df, x=col, hue="Selected", kde=False, palette=palette)
                plt.title(f"{col} Distribution (Fallback)")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, fname), dpi=150)
        except Exception as exc:
            logger.warning(f"Failed completely on histogram {col}: {exc}")
        finally:
            plt.close('all')

def create_correlation_heatmap(df: pd.DataFrame, out_path: str) -> None:
    try:
        cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"original_idx", "Selected", "cache_used"}]
        if len(cols) < 2: return
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
        plt.title("Metric Correlation Matrix")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
    except Exception as exc: logger.warning(f"Heatmap failed: {exc}")
    finally: plt.close('all')

def create_pairplot(df: pd.DataFrame, out_path: str) -> None:
    try:
        available = [c for c in ["DINO_Similarity", "CLIP_Similarity", "LPIPS", "Final_Score"] if c in df.columns and not df[c].isnull().all()]
        if len(available) < 2: return
        
        plot_df = df.copy()
        palette = {}
        if True in plot_df["Selected"].values: palette[True] = "green"
        if False in plot_df["Selected"].values: palette[False] = "red"

        # Apply slight jitter to essentially zero-variance columns to protect Seaborn
        for col in available:
            if plot_df[col].nunique() < 2 or plot_df[col].var() < 1e-5:
                plot_df[col] = plot_df[col] + np.random.normal(0, 1e-5, size=len(plot_df))

        g = sns.pairplot(
            plot_df, hue="Selected", vars=available, palette=palette,
            plot_kws={"alpha": 0.6, "s": 20}, diag_kind="hist", corner=True,
        )
        g.fig.suptitle("Metric Pairplot", y=1.02)
        g.savefig(out_path, dpi=150)
    except Exception as exc: logger.warning(f"Pairplot failed: {exc}")
    finally: plt.close('all')

def create_violin_plots(df: pd.DataFrame, out_path: str) -> None:
    try:
        if "Class" not in df.columns or "Final_Score" not in df.columns or df["Final_Score"].nunique() < 2: return
        palette = {}
        if True in df["Selected"].values: palette[True] = "green"
        if False in df["Selected"].values: palette[False] = "red"

        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=df, x="Class", y="Final_Score", hue="Selected",
            split=(df["Selected"].nunique() == 2), palette=palette, inner="quartile"
        )
        plt.title("Final Score Distribution by Class")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
    except Exception as exc: logger.warning(f"Violin plot failed: {exc}")
    finally: plt.close('all')

def create_projection_plot(real_embs: torch.Tensor, gen_embs: torch.Tensor, df_gen: pd.DataFrame, out_path: str, method: str = "umap") -> None:
    try:
        if method == "umap" and not HAS_UMAP: return
        if method == "tsne" and not HAS_SKLEARN: return
        
        real_np = real_embs.numpy()
        if len(real_np) > 3000: real_np = real_np[np.random.choice(len(real_np), 3000, replace=False)]

        gen_np = gen_embs.numpy()
        gen_sel = np.zeros(len(gen_np), dtype=bool)
        if df_gen["Selected"].any():
            selected_indices = df_gen.loc[df_gen["Selected"], "original_idx"].to_numpy(dtype=int)
            gen_sel[selected_indices] = True

        if len(gen_np) > 3000:
            idx = np.random.choice(len(gen_np), 3000, replace=False)
            gen_np = gen_np[idx]
            gen_sel = gen_sel[idx]

        all_pts = np.vstack([real_np, gen_np])
        if len(all_pts) < 3: return
        labels = ["Real"] * len(real_np) + ["Selected" if s else "Rejected" for s in gen_sel]

        if method == "umap":
            embedding = umap.UMAP(n_components=2, random_state=SEED).fit_transform(all_pts)
        else:
            perplexity = min(30, max(1, len(all_pts) - 2))
            embedding = TSNE(n_components=2, random_state=SEED, perplexity=perplexity).fit_transform(all_pts)

        plot_df = pd.DataFrame({"X": embedding[:, 0], "Y": embedding[:, 1], "Type": labels})
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=plot_df, x="X", y="Y", hue="Type", palette={"Real": "blue", "Selected": "green", "Rejected": "red"}, s=15, alpha=0.7)
        plt.title(f"{method.upper()}: Real vs Generated (DINO Embeddings)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
    except Exception as exc: logger.warning(f"{method.upper()} plot failed: {exc}")
    finally: plt.close('all')

def create_contact_sheet(df_subset: pd.DataFrame, out_path: str, title: str, thumb_size: Tuple[int, int] = (128, 128), nrow: int = 10) -> None:
    try:
        if df_subset.empty: return
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 7)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 6)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        images: List[Image.Image] = []
        for _, row in df_subset.head(50).iterrows():
            try:
                img = load_image_safe(row["Path"]).convert("RGB").resize(thumb_size, Image.BILINEAR)
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                draw.rectangle([0, thumb_size[1] - 34, thumb_size[0], thumb_size[1]], fill=(0, 0, 0, 180))
                
                draw.text((2, thumb_size[1] - 33), Path(row["Path"]).name[:18], fill=(255, 255, 0), font=font)
                draw.text((2, thumb_size[1] - 22), f"F:{row.get('Final_Score',0):.3f} D:{row.get('DINO_Similarity',0):.3f}", fill=(255, 255, 255), font=font_small)
                draw.text((2, thumb_size[1] - 12), f"L:{row.get('LPIPS',0):.3f}", fill=(200, 200, 255), font=font_small)
                
                img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                images.append(transforms.ToTensor()(img))
            except Exception: pass

        if not images: return
        grid_img = transforms.ToPILImage()(make_grid(images, nrow=nrow, padding=2, normalize=False))
        final = Image.new("RGB", (grid_img.width, grid_img.height + 30), color="white")
        final.paste(grid_img, (0, 30))
        ImageDraw.Draw(final).text((10, 8), title, fill=(0, 0, 0))
        final.save(out_path)
    except Exception as exc: logger.warning(f"Contact sheet {title} failed: {exc}")

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def discover_classes(real_folder: str, gan_folder: str) -> List[str]:
    real_items = {p.name for p in Path(real_folder).iterdir() if p.is_dir()}
    gan_items = {p.name for p in Path(gan_folder).iterdir() if p.is_dir()}
    return [c for c in sorted(real_items & gan_items) if not c.startswith(".")]

def compute_final_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    norm_map = {
        "n_dino": ("DINO_Similarity", False), "n_clip": ("CLIP_Similarity", False),
        "n_lpips": ("LPIPS", True), "n_nn": ("Nearest_Neighbor_Distance", True),
        "n_disc": ("Discriminator_Score", False), "n_local_density": ("Local_Density", True),
        "n_class_margin": ("Class_Margin", False), "n_musiq": ("MUSIQ", False),
        "n_maniqa": ("MANIQA", False), "n_brisque": ("BRISQUE", False), "n_niqe": ("NIQE", False),
    }
    for norm_col, (src_col, invert) in norm_map.items():
        df[norm_col] = min_max_norm(df[src_col].values, invert=invert) if src_col in df.columns else 0.5
    
    score_terms = [weight * df[f"n_{key}"].values for key, weight in weights.items() if f"n_{key}" in df.columns]
    final = np.sum(score_terms, axis=0) if score_terms else np.ones(len(df)) * 0.5

    if "Outlier_Score" in df.columns and OUTLIER_PENALTY > 0:
        final = final * (1.0 - (OUTLIER_PENALTY * df["Outlier_Score"].values))
    df["Final_Score"] = final
    return df

def main() -> None:
    total_start = time.perf_counter()
    set_seed(SEED)
    attach_log_file()

    logger.info("=" * 70)
    logger.info("Starting GAN Quality Selection Pipeline")
    logger.info("=" * 70)

    gpu_info = get_gpu_info()
    if gpu_info["available"]: logger.info(f"GPU: {gpu_info['name']} | Total VRAM: {gpu_info['total_gb']:.1f} GB | Free: {gpu_info['free_gb']:.1f} GB")
    else: logger.info("Running on CPU")

    classes = discover_classes(REAL_FOLDER, GAN_FOLDER)
    if not classes:
        logger.error("No matching classes found.")
        sys.exit(1)
    
    real_paths_by_class, gan_paths_by_class, candidate_audit = {}, {}, {}
    for cls in classes:
        real_paths = list_valid_image_paths(os.path.join(REAL_FOLDER, cls))
        gan_paths, audit = gan_only_paths(real_paths, list_valid_image_paths(os.path.join(GAN_FOLDER, cls)))
        if not real_paths or not gan_paths: continue
        real_paths_by_class[cls], gan_paths_by_class[cls], candidate_audit[cls] = real_paths, gan_paths, audit
        logger.info(f"[{cls}] real={len(real_paths)}, GAN candidates={len(gan_paths)}, raw copies={audit['raw_copies_excluded']}")

    iqa = IQAMetrics(DEVICE)
    disc_available = USE_DISCRIMINATOR and os.path.exists(DISCRIMINATOR_PATH)
    availability = {
        "dino": True, "clip": True, "disc": disc_available, "lpips": HAS_LPIPS,
        "nn_dist": True, "local_density": ENABLE_LOCAL_DENSITY, "class_margin": True,
        "musiq": iqa.availability.get("musiq", False), "maniqa": iqa.availability.get("maniqa", False),
        "brisque": iqa.availability.get("brisque", False), "niqe": iqa.availability.get("niqe", False),
    }
    active_weights = redistribute_weights(_DESIRED_WEIGHTS, availability)
    
    cache = SmartCache(os.path.join(OUTPUT_FOLDER, ".cache"))
    dino_extractor = EmbeddingExtractor("dino", DEVICE, ENABLE_FP16, cache)
    clip_extractor = EmbeddingExtractor("clip", DEVICE, ENABLE_FP16, cache)
    lpips_computer = LPIPSComputer(DEVICE, ENABLE_FP16)
    
    disc_model = None
    if disc_available:
        disc_model = DiscriminatorLoader.load(DISCRIMINATOR_PATH, DEVICE)
        if disc_model is None:
            availability["disc"] = False
            active_weights = redistribute_weights(_DESIRED_WEIGHTS, availability)
            
    nn_search = ReusableNearestNeighborSearch(use_faiss=ENABLE_FAISS)
    computer = MetricComputer(dino_extractor, clip_extractor, nn_search, lpips_computer, disc_model, iqa, OutlierDetector(method=OUTLIER_METHOD), DEVICE, ENABLE_FP16)

    all_results, summary_rows, dataset_metric_rows, per_class_times = [], [], [], {}

    for cls in classes:
        cls_start = time.perf_counter()
        if cls not in real_paths_by_class or cls not in gan_paths_by_class: continue
        out_cls_dir = os.path.join(OUTPUT_FOLDER, cls)
        os.makedirs(out_cls_dir, exist_ok=True)

        try:
            with gpu_memory_tracker(f"class_{cls}"):
                df, dino_real, dino_gan, real_paths, gan_paths = computer.compute_all(
                    real_paths_by_class[cls], gan_paths_by_class[cls], cls,
                    [p for c, p_list in real_paths_by_class.items() if c != cls for p in p_list]
                )
        except ValueError as exc:
            logger.warning(str(exc))
            continue

        df = compute_final_scores(df, active_weights).sort_values(by="Final_Score", ascending=False).reset_index(drop=True)
        near_real_copy, class_ambiguous = df["DINO_Max_Similarity"] >= COPY_SIMILARITY_THRESHOLD, df["Class_Margin"] <= MIN_CLASS_MARGIN
        df["eligible"] = ~(near_real_copy | class_ambiguous)
        
        df_filtered = filter_diversity(df[df["eligible"]].copy(), dino_gan, nn_search, DIVERSITY_THRESHOLD)
        df_filtered["Selected"] = False
        n_select = min(max(1, int(len(gan_paths) * TOP_PERCENT)), max(1, int(len(real_paths) * MAX_SELECTED_TO_REAL_RATIO)), len(df_filtered))
        if n_select > 0: df_filtered.iloc[:n_select, df_filtered.columns.get_loc("Selected")] = True
        
        selected_paths = set(df_filtered[df_filtered["Selected"]]["Path"])
        df["Selected"] = df["Path"].isin(selected_paths)
        df["selection_reason"] = "below_quality_rank"
        df.loc[near_real_copy, "selection_reason"] = "near_real_copy"
        df.loc[class_ambiguous, "selection_reason"] = "cross_class_ambiguous"
        df.loc[df["Path"].isin(selected_paths), "selection_reason"] = "top_quality"
        df["duplicate_removed"] = df["eligible"] & ~df["Path"].isin(set(df_filtered["Path"]))
        df.loc[df["duplicate_removed"], "selection_reason"] = "near_duplicate"

        logger.info(f"[{cls}] Copying selected images ...")
        for p in df[df["Selected"]]["Path"]: shutil.copy2(p, os.path.join(out_cls_dir, Path(p).name))
        all_results.append(df)

        if ENABLE_DATASET_METRICS and df["Selected"].any():
            selected_embs = dino_gan[df.loc[df["Selected"], "original_idx"].to_numpy(dtype=int)].numpy()
            dataset_metric_rows.append({"Class": cls, **DistributionMetrics.compute_all(dino_real.numpy(), selected_embs)})

        summary_rows.append({
            "Class": cls, "Number_Real": len(real_paths), "Number_Generated": len(gan_paths),
            "Raw_Copies_Excluded": candidate_audit[cls]["raw_copies_excluded"], "Exact_GAN_Duplicates_Excluded": candidate_audit[cls]["exact_gan_duplicates_excluded"],
            "Number_Selected": int(df["Selected"].sum()), "Number_Duplicates_Removed": int(df["duplicate_removed"].sum()),
            "Number_Near_Real_Copies_Rejected": int(near_real_copy.sum()), "Number_Cross_Class_Ambiguous_Rejected": int(class_ambiguous.sum()),
            "Number_Outliers": int(df["Outlier_Score"].sum()) if "Outlier_Score" in df.columns else 0,
            "Mean_DINO": df["DINO_Similarity"].mean(), "Mean_Class_Margin": df["Class_Margin"].mean(),
            "Mean_CLIP": df["CLIP_Similarity"].mean(), "Mean_LPIPS": df["LPIPS"].mean(), "Mean_Final_Score": df["Final_Score"].mean(),
        })

        per_class_times[cls] = time.perf_counter() - cls_start
        logger.info(f"[{cls}] Completed in {per_class_times[cls]:.2f}s")

        res_dir = os.path.join(OUTPUT_FOLDER, "results", cls)
        logger.info(f"[{cls}] Generating visualisations ...")
        create_histograms(df, res_dir)
        create_correlation_heatmap(df, os.path.join(res_dir, "correlation_heatmap.png"))
        create_pairplot(df, os.path.join(res_dir, "pairplot.png"))
        create_violin_plots(df, os.path.join(res_dir, "violin_plots.png"))
        if ENABLE_UMAP: create_projection_plot(dino_real, dino_gan, df, os.path.join(res_dir, "UMAP_real_vs_generated.png"), method="umap")
        if ENABLE_TSNE: create_projection_plot(dino_real, dino_gan, df, os.path.join(res_dir, "tSNE_real_vs_generated.png"), method="tsne")
        create_contact_sheet(df.head(50), os.path.join(res_dir, "Best50.png"), f"Best 50 GAN Images - {cls}")
        create_contact_sheet(df.tail(50), os.path.join(res_dir, "Worst50.png"), f"Worst 50 GAN Images - {cls}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.drop(columns=[c for c in ["Path", "original_idx"] + [f"n_{k}" for k in _DESIRED_WEIGHTS] if c in final_df.columns], inplace=True)
        final_df.to_csv(os.path.join(OUTPUT_FOLDER, "scores.csv"), index=False)
        pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_FOLDER, "summary.csv"), index=False)
        if dataset_metric_rows: pd.DataFrame(dataset_metric_rows).to_csv(os.path.join(OUTPUT_FOLDER, "dataset_metrics.csv"), index=False)

    cache_stats = cache.get_stats()
    logger.info("=" * 70 + "\nEXECUTION SUMMARY\n" + "=" * 70)
    logger.info(f"Total runtime: {time.perf_counter() - total_start:.2f}s")
    for cls, t in per_class_times.items(): logger.info(f"  {cls}: {t:.2f}s")
    logger.info(f"Cache hits: {cache_stats['hits']}, misses: {cache_stats['misses']}")
    logger.info("Finished. Top-quality synthetic images successfully selected.")

# =====================================================================
# VALIDATION
# =====================================================================
def _validate_normalization() -> bool:
    test_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(min_max_norm(test_arr), [0.0, 0.25, 0.5, 0.75, 1.0]), "Normalisation failed"
    assert np.allclose(min_max_norm(test_arr, invert=True), [1.0, 0.75, 0.5, 0.25, 0.0]), "Inverted normalisation failed"
    return True

def _validate_weight_redistribution() -> bool:
    desired, avail = {"a": 0.3, "b": 0.3, "c": 0.2, "d": 0.2}, {"a": True, "b": True, "c": False, "d": True}
    assert abs(sum(redistribute_weights(desired, avail).values()) - 1.0) < 1e-6, "Weights redistribution failed"
    return True

def _validate_pipeline() -> None:
    logger.info("Running internal validation ...")
    for name, check_fn in [("normalisation", _validate_normalization), ("weight_redistribution", _validate_weight_redistribution)]:
        try:
            check_fn()
            logger.info(f"  ✓ {name}")
        except Exception as exc:
            logger.error(f"  ✗ {name}: {exc}")
    logger.info("Validation complete")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quality-controlled selection of GAN-generated "
                    "autofluorescence images for synthetic augmentation.")
    p.add_argument("--real-folder", type=str, default=None,
                   help="Real training images (class folders). Default: data/train")
    p.add_argument("--gan-folder", type=str, default=None,
                   help="Generated images (class folders). "
                        "Default: outputs/gan/final_generated")
    p.add_argument("--output-folder", type=str, default=None,
                   help="Where the selected subset and reports are written. "
                        "Default: outputs/selected_synthetic")
    p.add_argument("--discriminator", type=str, default=None,
                   help="Optional discriminator checkpoint for the realism score.")
    return p


if __name__ == "__main__":
    _args = build_argparser().parse_args()
    if _args.real_folder:
        REAL_FOLDER = _args.real_folder
    if _args.gan_folder:
        GAN_FOLDER = _args.gan_folder
    if _args.output_folder:
        OUTPUT_FOLDER = _args.output_folder
    if _args.discriminator:
        DISCRIMINATOR_PATH = _args.discriminator
        USE_DISCRIMINATOR = True
    _validate_pipeline()
    main()
    main()
