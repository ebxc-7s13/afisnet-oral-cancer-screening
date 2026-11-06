# =============================================================================
# afis_core.py  --  shared core for the AFiS-Net Kaggle experiment notebook
# -----------------------------------------------------------------------------
# * clones the official repository (pinned commit) and imports it UNCHANGED
#   (classification.py / kfold_classification.py are the reference code)
# * discovers the uploaded data, writes paths.json for the GPU workers
# * defines every experiment as a "job" = one call of classification.run_one()
#   on one region-grouped fold, exactly as kfold_classification.main() does
# * registers the AFiS-Net ablation variants (E4) next to the repo models
# * runs a resumable, budget-aware job queue on 1-2 GPUs
# =============================================================================
import os
import sys
import json
import time
import glob
import shutil
import random
import subprocess
import traceback
from pathlib import Path

import numpy as np

REPO_URL = "https://github.com/ebxc-7s13/afisnet-oral-cancer-screening.git"
REPO_COMMIT = "d55dbb7"          # commit audited for this notebook
SMOKE = os.environ.get("AFIS_SMOKE", "0") == "1"   # tiny CPU self-test only

WORK = Path(os.environ.get("AFIS_WORK", "/kaggle/working"))
RESULTS = WORK / "afis_results"
REPO_DIR = WORK / "afisnet_repo"
RUNS = RESULTS / "runs"
QUEUE = RESULTS / "queue"
PATHS_JSON = RESULTS / "paths.json"
MARKER = "AFIS_RESULTS_MARKER.json"

SEEDS = [42, 100, 123]           # seed 42 = the seed of the submitted paper
N_FOLDS = 5
KFOLD_MODELS = ["AFiSNet", "NextViT_Small", "CoAtNet0", "ResNet50"]
SYN_EXP = "Raw_SelectedSynthetic"
ABLATION_MODELS = ["AFiSNet_CNNonly", "AFiSNet_SwinOnly", "AFiSNet_Concat",
                   "AFiSNet_AvgPool", "AFiSNet_noSupCon"]
SCALING_FRACTIONS = [0.25, 0.50, 0.75]
RANDOM_SUBSET_SEED = 2026
DOSE_SIZES = [500, 1000, 2000]   # session 6/7: random NTP subsets (half per class); 3000 = Raw_AllUnfiltered
QC_ALL_TOP_PERCENT = 0.80        # session 6: official QC script re-run on ALL NTP images, keep the top 80 %
# session 9/10: choice of the final AFiS-Net configuration (pre-declared rule, see choose_final_model)
FINAL_CANDIDATES = [            # (family holding the 15 QC-augmented runs, model, simplicity rank: lower = simpler)
    ("E9_select", "AFiSNet_ConcatAvg", 0),     # concatenation fusion + average pooling (both simplifications)
    ("E4_ablation", "AFiSNet_Concat", 1),      # concatenation fusion, GeM pooling
    ("E4_ablation", "AFiSNet_AvgPool", 1),     # cross-attention fusion, average pooling
    ("E1_kfold", "AFiSNet", 2),                # original design: cross-attention + GeM
]
FINAL_TIE_MARGIN = 0.005         # AUC within this of the best = equivalent -> simplest wins
OPTIONAL_FINAL_ARMS = False      # data-scaling curve with the final model (adds ~4 h; the paper keeps the AFiS-Net-XA curve)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# 1. Repository
# =============================================================================
def ensure_repo():
    """Clone the official repository at the audited commit (idempotent)."""
    if (REPO_DIR / "classification.py").exists():
        return REPO_DIR
    src = os.environ.get("AFIS_REPO_SRC")            # local copy (self-test only)
    if src:
        shutil.copytree(src, REPO_DIR, ignore=shutil.ignore_patterns(".git"))
        return REPO_DIR
    subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "-q", REPO_COMMIT], check=True)
    return REPO_DIR


# =============================================================================
# 2. Data discovery
# =============================================================================
def ci_dir(d):
    """Return d, or its case-insensitive match (e.g. 'Cancer' for 'cancer'); None if absent."""
    d = str(d)
    if os.path.isdir(d):
        return d
    parent, name = os.path.split(d.rstrip(os.sep))
    if os.path.isdir(parent):
        for e in sorted(os.listdir(parent)):
            if e.lower() == name.lower() and os.path.isdir(os.path.join(parent, e)):
                return os.path.join(parent, e)
    return None


def _has_classes(d, names=("normal", "cancer")):
    return all(ci_dir(os.path.join(d, n)) for n in names)


_SKIP_DIRS = {"afis_results", "afisnet_repo", "afis_code", ".git", "__pycache__"}   # previous-session outputs


def _walk_dirs(root, max_depth=8):
    root = str(root)
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if dirpath.count(os.sep) - base >= max_depth:
            dirnames[:] = []
        yield dirpath, dirnames


def discover_inputs(input_root="/kaggle/input", overrides=None):
    """Locate the uploaded folders by their documented names.

    Expected names (anywhere below input_root):
      real/train_raw (or train), real/val, real/test   (each with normal/ cancer/, any case)
      synthetic_selected/                    (normal/ cancer/)   required
      synthetic_all_ntp/                     (normal/ cancer/)   optional, E3/E8
      synthetic_all_baseline/                (normal/ cancer/)   optional, E3/E8
      synthetic_selected_baseline/           (normal/ cancer/)   optional, E3
      screening/                             (normal/ smoke|smoker/ cancer/)
      baseline_gan*.pth                      optional (generation only, no training)
    """
    overrides = overrides or {}
    found = {}
    for d, subdirs in _walk_dirs(input_root):
        name = os.path.basename(d).lower()
        # the repository calls the training folder train_raw; 'train' is accepted as well
        if name in ("train_raw", "train") and _has_classes(d) and "train" not in found:
            parent = os.path.dirname(d)
            val, test = ci_dir(os.path.join(parent, "val")), ci_dir(os.path.join(parent, "test"))
            if val and test and _has_classes(val) and _has_classes(test):
                found["train"] = d
                found["val"] = val
                found["test"] = test
        for key in ("synthetic_selected_baseline", "synthetic_selected",
                    "synthetic_all_ntp", "synthetic_all_baseline"):
            if name == key and _has_classes(d) and key not in found:
                found[key] = d
        if name == "screening" and "screening" not in found:
            subs = {s.lower() for s in subdirs}
            if "normal" in subs and "cancer" in subs and ({"smoke", "smoker"} & subs):
                found["screening"] = d
    for p in sorted(glob.glob(os.path.join(str(input_root), "**", "baseline_gan*.pth"), recursive=True)):
        if not (set(Path(p).parts) & _SKIP_DIRS):
            found.setdefault("baseline_ckpt", p)
    found.update({k: v for k, v in overrides.items() if v})
    for k in ("train", "synthetic_selected"):
        if k not in found:
            seen = [d for d, _ in _walk_dirs(input_root, max_depth=4)][:60]
            raise FileNotFoundError(
                f"Could not find '{k}' under {input_root}. See the data-layout cell at the top.\n"
                "Folders that were found:\n  " + "\n  ".join(seen))
    return found


def save_paths(paths):
    RESULTS.mkdir(parents=True, exist_ok=True)
    PATHS_JSON.write_text(json.dumps(paths, indent=2))


def load_paths():
    return json.loads(PATHS_JSON.read_text())


# =============================================================================
# 3. Pipeline import (env vars MUST be set before classification is imported)
# =============================================================================
_PIPE = {}


def import_pipeline(paths=None):
    """Import the repository modules exactly as kfold_classification.py does."""
    if "C" in _PIPE:
        return _PIPE["C"], _PIPE["K"]
    paths = paths or load_paths()
    os.environ["AFIS_TRAIN_ROOT"] = paths["train"]
    os.environ["AFIS_VAL_ROOT"] = paths["val"]
    os.environ["AFIS_TEST_ROOT"] = paths["test"]
    os.environ["AFIS_SELECTED_SYN_ROOT"] = paths["synthetic_selected"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    import classification as C          # noqa: E402  (official code)
    import kfold_classification as K    # noqa: E402  (official code)
    assert C.RAW_ROOT == paths["train"], "classification.py did not pick up AFIS_TRAIN_ROOT"
    register_variants(C)
    if SMOKE:
        # Self-test only: no internet in the test sandbox -> random init.
        for name, spec in C.MODEL_SPECS.items():
            spec["weights"] = None
    _PIPE["C"], _PIPE["K"] = C, K
    return C, K


# =============================================================================
# 4. AFiS-Net ablation variants (E4)
# =============================================================================
def register_variants(C):
    """Register ablated AFiS-Net encoders beside the official 'AFiSNet'.

    The class reproduces C.FlagshipHybrid (same modules, same creation order,
    same FPN / GeM / cross-attention / head) and switches single components
    off, so each variant differs from AFiS-Net in exactly one design choice.
    """
    if "AFiSNet_CNNonly" in C.MODEL_SPECS:
        return
    import torch
    import torch.nn as nn
    import timm

    class _AvgPool(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3))

    class AblationHybrid(nn.Module):
        def __init__(self, pretrained, out_dim=1024, use_cnn=True, use_vit=True,
                     fusion="xattn", pool="gem", cnn_name="convnextv2_nano",
                     vit_name="swinv2_cr_tiny_ns_224", n_levels=3, fpn_dim=256, attn_dim=384):
            super().__init__()
            assert use_cnn or use_vit
            self.use_cnn, self.use_vit, self.fusion_mode = use_cnn, use_vit, fusion
            width = 0
            if use_cnn:
                self.cnn = timm.create_model(cnn_name, pretrained=pretrained, features_only=True)
                cnn_ch = self.cnn.feature_info.channels()
                self.n_levels = max(1, min(n_levels, len(cnn_ch)))
                self.fpn = C.FPNFusion(cnn_ch[-self.n_levels:], fpn_dim)
                self.pools = nn.ModuleList([C.GeMPool2d() if pool == "gem" else _AvgPool()
                                            for _ in range(self.n_levels)])
                desc_dim = fpn_dim * self.n_levels
                width += desc_dim
            if use_vit:
                self.vit = timm.create_model(vit_name, pretrained=pretrained, features_only=True)
                ctx_ch = self.vit.feature_info.channels()[-1]
                if use_cnn and fusion == "xattn":
                    self.fusion = C.CrossAttentionFusion(desc_dim, ctx_ch, attn_dim)
                else:
                    # Context without attention: 1x1 projection + global average
                    # pooling + LayerNorm (mirrors the projection/normalisation
                    # of CrossAttentionFusion, minus the attention itself).
                    self.ctx_proj = nn.Conv2d(ctx_ch, attn_dim, 1)
                    self.ctx_norm = nn.LayerNorm(attn_dim)
                width += attn_dim
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_dim), nn.GELU())
            self.out_dim = int(out_dim)

        def forward(self, x):
            parts = []
            if self.use_cnn:
                feats = self.cnn(x)[-self.n_levels:]
                fused = self.fpn(feats)
                desc = torch.cat([p(f) for p, f in zip(self.pools, fused)], dim=1)
                parts.append(desc)
            if self.use_vit:
                ctx_map = self.vit(x)[-1]
                if self.use_cnn and self.fusion_mode == "xattn":
                    parts.append(self.fusion(desc, ctx_map))
                else:
                    parts.append(self.ctx_norm(self.ctx_proj(ctx_map).mean(dim=(2, 3))))
            return self.head(torch.cat(parts, dim=1))

        def gradcam_target(self):
            return self.fpn.cam_conv if self.use_cnn else self.ctx_proj

    def make(**kw):
        def ctor(weights=None):
            return AblationHybrid(pretrained=weights is not None, out_dim=C.ADVANCED_OUT_DIM, **kw)
        return ctor

    base = dict(C.MODEL_SPECS["AFiSNet"])
    variants = {
        "AFiSNet_CNNonly": dict(use_vit=False),
        "AFiSNet_SwinOnly": dict(use_cnn=False),
        "AFiSNet_Concat": dict(fusion="concat"),
        "AFiSNet_AvgPool": dict(pool="avg"),
        "AFiSNet_ConcatAvg": dict(fusion="concat", pool="avg"),
    }
    for name, kw in variants.items():
        spec = dict(base)
        spec["ctor"] = make(**kw)
        C.MODEL_SPECS[name] = spec
    # identical architecture; the supervised-contrastive weight is set to 0 per job
    C.MODEL_SPECS["AFiSNet_noSupCon"] = dict(base)
    C.AblationHybrid = AblationHybrid


# =============================================================================
# 5. Jobs
# =============================================================================
def make_job(family, model, exp, seed, fold, syn="none", frac=1.0, overrides=None,
             keep_ckpt=False, n_gradcam=0, session=1, nsyn=0):
    return dict(family=family, model=model, exp=exp, seed=int(seed), fold=int(fold),
                syn=syn, frac=float(frac), overrides=overrides or {}, keep_ckpt=bool(keep_ckpt),
                n_gradcam=int(n_gradcam), session=int(session), nsyn=int(nsyn))


def qc_all_dir():
    return RESULTS / "qc_all" / "selected"


def qc_clean_dir():
    return RESULTS / "qc_clean" / "selected"


def qc_clean_ready():
    d = qc_clean_dir()
    return (RESULTS / "qc_clean" / "summary.json").exists() and (d / "normal").is_dir() and (d / "cancer").is_dir()


def qc_all_ready():
    d = qc_all_dir()
    return (d / "summary.csv").exists() and (d / "normal").is_dir() and (d / "cancer").is_dir()


def job_id(j):
    return f"{j['family']}__{j['model']}__{j['exp']}__s{j['seed']}__f{j['fold']}"


def run_dir(j):
    return RUNS / j["family"] / j["model"] / j["exp"] / f"seed_{j['seed']}" / f"fold_{j['fold']}"


def is_done(j):
    return (run_dir(j) / "record.json").exists()


def _record_aucs(family, model, exp):
    d = RUNS / family / model / exp
    vals = []
    for f in d.glob("seed_*/fold_*/record.json") if d.exists() else []:
        r = json.loads(f.read_text())
        vals.append((float(r.get("roc_auc_macro", float("nan"))), float(r.get("f1_macro", float("nan"))),
                     float(r.get("ece", float("nan")))))
    return vals


def final_model_file():
    return RESULTS / "final_model.json"


def choose_final_model():
    """Pre-declared rule: among the four candidates (original design and the operator-simplified variants), each
    trained with QC-selected augmentation over the same 15 fold x seed runs, take the highest mean AUC; any
    candidate within FINAL_TIE_MARGIN of it counts as equivalent and the simplest of those is chosen (ties in
    simplicity -> higher AUC). The decision is written once and never revised, so later results cannot change it."""
    f = final_model_file()
    if f.exists():
        return json.loads(f.read_text())["final_model"]
    rows = []
    for fam, model, simp in FINAL_CANDIDATES:
        v = _record_aucs(fam, model, SYN_EXP)
        n_need = len(SEEDS) * N_FOLDS if not SMOKE else 1
        if len(v) < n_need:
            log(f"Final-model choice waits for {fam}/{model}: {len(v)} of {n_need} runs.")
            return None
        a = np.array(v)
        rows.append(dict(family=fam, model=model, simplicity=simp, runs=len(v), auc_mean=float(np.nanmean(a[:, 0])),
                         f1_mean=float(np.nanmean(a[:, 1])), ece_mean=float(np.nanmean(a[:, 2]))))
    best = max(r["auc_mean"] for r in rows)
    eq = [r for r in rows if r["auc_mean"] >= best - FINAL_TIE_MARGIN]
    eq.sort(key=lambda r: (r["simplicity"], -r["auc_mean"]))
    choice = eq[0]["model"]
    f.write_text(json.dumps({"final_model": choice, "rule": choose_final_model.__doc__.strip(),
                             "tie_margin": FINAL_TIE_MARGIN, "candidates": rows,
                             "decided_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    log(f"FINAL MODEL = {choice}  (" + ", ".join(f"{r['model']} {r['auc_mean']:.4f}" for r in rows) + ")")
    return choice


def final_arm(model, exp):
    """(family, model) that holds the runs of the final model for a training set."""
    if model == "AFiSNet":
        return {"Raw": "E1_kfold", SYN_EXP: "E1_kfold", "Raw_AllUnfiltered": "E3_synthetic_source",
                "Raw_RandomUnfiltered": "E3_synthetic_source"}.get(exp, "E5_scaling" if "frac" in exp else "E9_final"), model
    if exp == SYN_EXP:
        return ("E9_select" if model == "AFiSNet_ConcatAvg" else "E4_ablation"), model
    return "E9_final", model


def build_final_jobs(paths, model, seeds=None, folds=None):
    """Session 9/10 (stage B): the main experiments with the final model, plus the QC hygiene-filter test.
    Priority order: seed 42 first for every essential arm (real only, +all 3,000, +QC-clean, random matched to
    QC-clean), then seeds 100/123, then +215 random, then (optional) the data-scaling curve. If GPU time runs out,
    complete seed-42 comparisons exist before anything else."""
    seeds = list(seeds or SEEDS)
    folds = list(folds or range(N_FOLDS))
    has_all = "synthetic_all_ntp" in paths
    J = []
    for s in seeds:
        arms = []
        if model != "AFiSNet":
            arms.append(("Raw", "none"))
        if model != "AFiSNet" and has_all:
            arms.append(("Raw_AllUnfiltered", "all_unfiltered"))
        if qc_clean_ready():
            arms.append(("Raw_QCclean", "qc_clean"))
        for exp, syn in arms:
            for f in folds:
                J.append(make_job("E9_final", model, exp, s, f, syn, keep_ckpt=(s == seeds[0]), session=9))
        if s == seeds[0] and qc_clean_ready() and has_all:
            for f in folds:
                J.append(make_job("E9_final", model, "Raw_RandomMatchQCclean", s, f, "random_match_qcclean", session=9))
    if model != "AFiSNet" and has_all:
        for s in seeds:
            for f in folds:
                J.append(make_job("E9_final", model, "Raw_RandomUnfiltered", s, f, "random_unfiltered", session=9))
    if OPTIONAL_FINAL_ARMS and model != "AFiSNet":
        for s in seeds:
            for frac in SCALING_FRACTIONS:
                tag = f"frac{int(round(frac * 100)):03d}"
                for exp, syn in ((f"Raw_{tag}", "none"), (f"{SYN_EXP}_{tag}", "selected")):
                    for f in folds:
                        J.append(make_job("E9_final", model, exp, s, f, syn, frac=frac, session=9))
    if SMOKE:
        J = [j for j in J if j["seed"] == seeds[0] and ("frac" not in j["exp"] or j["frac"] == 0.5)]
    return J


def build_jobs(paths, seeds=None, folds=None):
    """Every job of every session, in priority order."""
    seeds = list(seeds or SEEDS)
    folds = list(folds or range(N_FOLDS))
    has_all_ntp = "synthetic_all_ntp" in paths
    has_base_sel = "synthetic_selected_baseline" in paths or \
        (RESULTS / "baseline_qc" / "selected" / "normal").is_dir()
    J = []
    # ---- Session 1: E1 AFiS-Net (seed 42 first; its checkpoints are kept) + NextViT
    for model in ["AFiSNet", "NextViT_Small"]:
        for s in seeds:
            for exp, syn in [("Raw", "none"), (SYN_EXP, "selected")]:
                for f in folds:
                    keep = model == "AFiSNet" and s == seeds[0]
                    J.append(make_job("E1_kfold", model, exp, s, f, syn, keep_ckpt=keep,
                                      n_gradcam=12 if model == "AFiSNet" else 0, session=1))
    # ---- Session 2: E1 CoAtNet0 + ResNet50
    for model in ["CoAtNet0", "ResNet50"]:
        for s in seeds:
            for exp, syn in [("Raw", "none"), (SYN_EXP, "selected")]:
                for f in folds:
                    J.append(make_job("E1_kfold", model, exp, s, f, syn, session=2))
    # ---- Session 3: E3 synthetic-source arms + E6 fold-aware selection (AFiS-Net)
    arms = []
    if has_all_ntp:
        arms += [("Raw_RandomUnfiltered", "random_unfiltered"), ("Raw_AllUnfiltered", "all_unfiltered")]
    if has_base_sel:
        arms += [("Raw_BaselineSelected", "baseline_selected")]
    for s in seeds:
        for exp, syn in arms:
            for f in folds:
                J.append(make_job("E3_synthetic_source", "AFiSNet", exp, s, f, syn, session=3))
        for f in folds:
            J.append(make_job("E6_foldaware", "AFiSNet", "Raw_FoldAwareSelected", s, f,
                              "foldaware", session=3))
    # ---- Session 4: E4 AFiS-Net ablations (Raw + selected synthetic, as the headline arm)
    for s in seeds:
        for model in ABLATION_MODELS:
            ov = {"supervised_contrastive_weight": 0.0} if model == "AFiSNet_noSupCon" else {}
            for f in folds:
                J.append(make_job("E4_ablation", model, SYN_EXP, s, f, "selected", overrides=ov,
                                  session=4))
    # ---- Session 5: E5 data-scaling curve (seed 42), Raw vs Raw+Synthetic
    for frac in SCALING_FRACTIONS:
        tag = f"frac{int(round(frac * 100)):03d}"
        for exp, syn in [(f"Raw_{tag}", "none"), (f"{SYN_EXP}_{tag}", "selected")]:
            for f in folds:
                J.append(make_job("E5_scaling", "AFiSNet", exp, seeds[0], f, syn, frac=frac,
                                  session=5))
    # ---- Session 6: (a) QC at scale, (b) E5 with seeds 100/123, (c) dose-response 500/1000
    if has_all_ntp and qc_all_ready():
        for f in folds:
            J.append(make_job("E3_qc_scale", "AFiSNet", "Raw_QCfilteredAll", seeds[0], f, "qc_all", session=6))
            J.append(make_job("E3_qc_scale", "AFiSNet", "Raw_RandomMatchedQCAll", seeds[0], f,
                              "random_match_qcall", session=6))
    for s in seeds[1:]:
        for frac in SCALING_FRACTIONS:
            tag = f"frac{int(round(frac * 100)):03d}"
            for exp, syn in [(f"Raw_{tag}", "none"), (f"{SYN_EXP}_{tag}", "selected")]:
                for f in folds:
                    J.append(make_job("E5_scaling", "AFiSNet", exp, s, f, syn, frac=frac, session=6))
    if has_all_ntp:
        for n in DOSE_SIZES:
            for f in folds:
                J.append(make_job("E3_dose", "AFiSNet", f"Raw_Random{n:04d}", seeds[0], f, "random_n",
                                  nsyn=n, session=6 if n < 2000 else 7))
    # ---- Session 7: does the "all NTP images" gain carry over to other backbones?
    if has_all_ntp:
        for model in ["NextViT_Small", "ResNet50"]:
            for f in folds:
                J.append(make_job("E3_generality", model, "Raw_AllUnfiltered", seeds[0], f, "all_unfiltered",
                                  session=7))
    # ---- Session 8: AFiS-Net + all 3,000 NTP images (seed 42) re-trained with checkpoints kept, so that the
    #      screening cohort can also be scored by the best-performing configuration (E3 runs keep no weights)
    if has_all_ntp:
        for f in folds:
            J.append(make_job("E8_screening", "AFiSNet", "Raw_AllUnfiltered", seeds[0], f, "all_unfiltered",
                              keep_ckpt=True, session=8))
    # ---- Session 9 (stage A): the one untested candidate for the final model, trained like the ablation
    #      variants (QC-selected augmentation, 3 seeds), so that all four candidates can be compared
    for s in seeds:
        for f in folds:
            J.append(make_job("E9_select", "AFiSNet_ConcatAvg", SYN_EXP, s, f, "selected",
                              keep_ckpt=(s == seeds[0]), session=9))
    if SMOKE:   # self-test only: a reduced grid that still touches every code path
        def _keep(j):
            if j["family"] == "E1_kfold":
                return j["model"] in ("AFiSNet", "ResNet50") and j["seed"] in seeds[:2]
            if j["family"] == "E5_scaling":
                return j["frac"] == 0.5 and j["seed"] in seeds[:2]
            if j["family"] == "E3_dose":
                return j["nsyn"] == 500
            if j["family"] == "E3_generality":
                return j["model"] == "ResNet50"
            return j["seed"] == seeds[0]
        J = [j for j in J if _keep(j)]
    return J


# =============================================================================
# 6. Synthetic sources and data-scaling subsets
# =============================================================================
def _syn_list(C, real, class_to_idx, root, use_hash):
    return C.get_synthetic(real, root, class_to_idx, use_hash=use_hash)


def get_synthetic_for_job(job, C, K, real, class_to_idx, cfg, split, paths):
    syn = job["syn"]
    if syn == "none":
        return [], {}
    if syn == "selected":                         # identical to kfold_classification.main
        return _syn_list(C, real, class_to_idx, paths["synthetic_selected"], cfg.use_hash_dedup), {}
    if syn in ("all_unfiltered", "random_unfiltered"):
        pool = _syn_list(C, real, class_to_idx, paths["synthetic_all_ntp"], cfg.use_hash_dedup)
        if syn == "all_unfiltered":
            return pool, {}
        sel = _syn_list(C, real, class_to_idx, paths["synthetic_selected"], cfg.use_hash_dedup)
        rng = np.random.default_rng(RANDOM_SUBSET_SEED)
        out = []
        for c in sorted(set(y for _, y in sel)):
            n_c = sum(1 for _, y in sel if y == c)
            cand = sorted(p for p, y in pool if y == c)
            pick = rng.choice(len(cand), size=min(n_c, len(cand)), replace=False)
            out += [(cand[i], c) for i in sorted(pick)]
        return out, {"random_subset_seed": RANDOM_SUBSET_SEED}
    if syn == "random_n":                         # dose-response: n/2 random NTP images per class
        pool = _syn_list(C, real, class_to_idx, paths["synthetic_all_ntp"], cfg.use_hash_dedup)
        n = int(job.get("nsyn", 0))
        rng = np.random.default_rng(RANDOM_SUBSET_SEED + n)
        out = []
        for c in sorted(set(y for _, y in pool)):
            cand = sorted(p for p, y in pool if y == c)
            pick = rng.choice(len(cand), size=min(n // 2, len(cand)), replace=False)
            out += [(cand[i], c) for i in sorted(pick)]
        return out, {"random_subset_seed": RANDOM_SUBSET_SEED + n, "requested": n}
    if syn in ("qc_all", "random_match_qcall"):
        if not qc_all_ready():
            raise FileNotFoundError("QC-at-scale selection missing (session-6 preparation step).")
        sel = _syn_list(C, real, class_to_idx, str(qc_all_dir()), cfg.use_hash_dedup)
        if syn == "qc_all":
            return sel, {"qc_top_percent": QC_ALL_TOP_PERCENT}
        pool = _syn_list(C, real, class_to_idx, paths["synthetic_all_ntp"], cfg.use_hash_dedup)
        rng = np.random.default_rng(RANDOM_SUBSET_SEED + 1)
        out = []
        for c in sorted(set(y for _, y in sel)):
            n_c = sum(1 for _, y in sel if y == c)
            cand = sorted(p for p, y in pool if y == c)
            pick = rng.choice(len(cand), size=min(n_c, len(cand)), replace=False)
            out += [(cand[i], c) for i in sorted(pick)]
        return out, {"random_subset_seed": RANDOM_SUBSET_SEED + 1, "matched_to": "qc_all"}
    if syn in ("qc_clean", "random_match_qcclean"):
        if not qc_clean_ready():
            raise FileNotFoundError("QC-clean selection missing (session-9 preparation step).")
        sel = _syn_list(C, real, class_to_idx, str(qc_clean_dir()), cfg.use_hash_dedup)
        if syn == "qc_clean":
            return sel, {"qc": "hygiene filter (qc_clean)"}
        pool = _syn_list(C, real, class_to_idx, paths["synthetic_all_ntp"], cfg.use_hash_dedup)
        rng = np.random.default_rng(RANDOM_SUBSET_SEED + 2)
        out = []
        for c in sorted(set(y for _, y in sel)):
            n_c = sum(1 for _, y in sel if y == c)
            cand = sorted(p for p, y in pool if y == c)
            pick = rng.choice(len(cand), size=min(n_c, len(cand)), replace=False)
            out += [(cand[i], c) for i in sorted(pick)]
        return out, {"random_subset_seed": RANDOM_SUBSET_SEED + 2, "matched_to": "qc_clean"}
    if syn in ("baseline_all_unfiltered", "ntp_match_baseline"):   # session 11: does the NTP loss matter downstream?
        broot = paths.get("synthetic_all_baseline")
        if not broot or not os.path.isdir(broot):
            g = RESULTS / "baseline_qc" / "generated" / "final_generated"
            broot = str(g) if g.is_dir() else None
        if not broot:
            raise FileNotFoundError("No baseline-generator images (synthetic_all_baseline/ or baseline_qc/generated).")
        base = _syn_list(C, real, class_to_idx, broot, cfg.use_hash_dedup)
        if syn == "baseline_all_unfiltered":
            return base, {"source": broot}
        pool = _syn_list(C, real, class_to_idx, paths["synthetic_all_ntp"], cfg.use_hash_dedup)
        rng = np.random.default_rng(RANDOM_SUBSET_SEED + 3)
        out = []
        for c in sorted(set(y for _, y in base)):
            n_c = sum(1 for _, y in base if y == c)
            cand = sorted(p for p, y in pool if y == c)
            pick = rng.choice(len(cand), size=min(n_c, len(cand)), replace=False)
            out += [(cand[i], c) for i in sorted(pick)]
        return out, {"random_subset_seed": RANDOM_SUBSET_SEED + 3, "matched_to": "baseline_all"}
    if syn == "baseline_selected":
        root = paths.get("synthetic_selected_baseline") or str(RESULTS / "baseline_qc" / "selected")
        if not os.path.isdir(root):
            raise FileNotFoundError("Baseline-selected synthetic set not available "
                                    "(upload synthetic_selected_baseline/ or a baseline GAN).")
        return _syn_list(C, real, class_to_idx, root, cfg.use_hash_dedup), {}
    if syn == "foldaware":
        sel = _syn_list(C, real, class_to_idx, paths["synthetic_selected"], cfg.use_hash_dedup)
        emb_file = RESULTS / "embeddings" / "dino_embeddings.npz"
        if not emb_file.exists():
            raise FileNotFoundError("DINOv2 embeddings missing; run the embedding step (session 1).")
        E = np.load(emb_file, allow_pickle=True)
        real_names = list(E["real_names"]); R = E["real_emb"]
        syn_names = list(E["sel_names"]); S = E["sel_emb"]
        held = {K.region_group(p) for p, _ in split["val"] + split["test"]}
        real_region = np.array([K.region_group(n) for n in real_names])
        held_mask = np.isin(real_region, list(held))
        idx = {n: i for i, n in enumerate(syn_names)}
        keep, dropped = [], []
        for p, y in sel:
            i = idx.get(os.path.basename(p))
            if i is None:
                keep.append((p, y)); continue
            nn = int(np.argmax(R @ S[i]))
            (dropped if held_mask[nn] else keep).append((p, y))
        info = {"n_selected": len(sel), "n_dropped_near_heldout": len(dropped),
                "dropped": [os.path.basename(p) for p, _ in dropped]}
        return keep, info
    raise ValueError(f"unknown synthetic source {syn}")


def subsample_regions(split, frac, seed, fold, K):
    """Keep a class-stratified fraction of TRAINING regions (val/test untouched).
    The same regions are drawn for the Raw and Raw+Synthetic arms (paired)."""
    if frac >= 0.999:
        return split
    rng = np.random.default_rng(10007 * seed + 101 * fold + int(round(frac * 1000)))
    by_class = {}
    for p, y in split["train"]:
        by_class.setdefault(y, set()).add(K.region_group(p))
    keep = set()
    for y in sorted(by_class):
        groups = sorted(by_class[y])
        n = max(1, int(round(frac * len(groups))))
        keep |= set(rng.choice(groups, size=n, replace=False).tolist())
    new = dict(split)
    new["train"] = [(p, y) for p, y in split["train"] if K.region_group(p) in keep]
    return new


# =============================================================================
# 7. Running one job (inside a GPU worker process)
# =============================================================================
def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v if isinstance(v, (str, bool)) or v is None else str(v)


def run_job(job):
    paths = load_paths()
    C, K = import_pipeline(paths)
    cfg = C.Config()
    cfg.results_root = str(RUNS / job["family"])
    cfg.n_folds = N_FOLDS
    cfg.patient_level_split = True
    cfg.run_benchmark = False
    cfg.n_gradcam = job["n_gradcam"]
    cfg.num_workers = int(os.environ.get("AFIS_NUM_WORKERS", "2"))
    for k, v in job["overrides"].items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Config has no field {k}")
        setattr(cfg, k, v)
    if SMOKE:
        cfg.epochs = 1; cfg.batch_size = 8; cfg.num_workers = 0; cfg.freeze_backbone_epochs = 0
        cfg.allow_random_init = True; cfg.save_embedding_visualizations = False
    class_names = list(C.CLASSES)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    real = K.pool_real_data(class_to_idx)
    folds = K.make_grouped_folds(real, N_FOLDS, job["seed"])
    split = subsample_regions(folds[job["fold"]], job["frac"], job["seed"], job["fold"], K)
    syn, syn_info = get_synthetic_for_job(job, C, K, real, class_to_idx, cfg, split, paths)
    t0 = time.time()
    record, cm, macro_tpr, y_true, y_pred, y_prob = C.run_one(
        job["model"], job["exp"], job["seed"], split, syn, class_names, cfg, fold=job["fold"])
    out = run_dir(job)
    assert out.is_dir(), f"run_one wrote to an unexpected folder (expected {out})"
    manifest = {k: [os.path.basename(p) for p, _ in split[k]] for k in ("train", "val", "test")}
    manifest["synthetic"] = [os.path.basename(p) for p, _ in syn]
    manifest["synthetic_info"] = syn_info
    (out / "split_manifest.json").write_text(json.dumps(manifest))
    if not job["keep_ckpt"]:
        ck = out / "best_model.pt"
        if ck.exists():
            ck.unlink()
    record = dict(record)
    record.update(family=job["family"], fold=job["fold"], seed=job["seed"], syn_source=job["syn"],
                  train_fraction=job["frac"], n_synthetic=len(syn), wall_seconds=time.time() - t0,
                  job_id=job_id(job), confusion_matrix=np.asarray(cm).tolist())
    tmp = out / "record.json.tmp"
    tmp.write_text(json.dumps(_jsonable(record), indent=1))
    os.replace(tmp, out / "record.json")
    return record


# =============================================================================
# 8. Budget-aware, resumable multi-GPU queue
# =============================================================================
def worker_main(jobs_file, deadline, worker_name):
    jobs = json.loads(Path(jobs_file).read_text())
    locks = QUEUE / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    durations = []
    by_key = {}
    default_est = 1.0 if SMOKE else 25 * 60.0

    def _estimate(job):
        key = (job["family"], job["model"], job["syn"], int(job.get("nsyn", 0)), float(job["frac"]))
        if key in by_key:
            return float(np.median(by_key[key])), key
        if SMOKE:
            return default_est, key
        if job["syn"] in ("all_unfiltered", "qc_all", "random_match_qcall", "qc_clean", "random_match_qcclean",
                          "baseline_all_unfiltered", "ntp_match_baseline"):
            return 65 * 60.0, key                       # ~3,000 synthetic images: ~1 h per fold on a T4
        if job["syn"] == "random_n":
            return (10 + 55 * int(job.get("nsyn", 0)) / 3000.0) * 60.0, key
        return (float(np.median(durations)) if durations else default_est), key
    for job in jobs:
        if (QUEUE / "STOP").exists():
            log(f"{worker_name}: STOP file found"); break
        if is_done(job):
            continue
        jid = job_id(job)
        lock = locks / f"{jid}.lock"
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, worker_name.encode()); os.close(fd)
        except FileExistsError:
            continue
        est, key = _estimate(job)
        if time.time() + 1.2 * est > deadline:
            lock.unlink(missing_ok=True)
            log(f"{worker_name}: {jid} (~{est / 60:.0f} min) does not fit the remaining budget -> next session")
            continue
        log(f"{worker_name}: START {jid}")
        t0 = time.time()
        try:
            run_job(job)
            durations.append(time.time() - t0)
            by_key.setdefault(key, []).append(time.time() - t0)
            log(f"{worker_name}: DONE  {jid}  ({(time.time() - t0) / 60:.1f} min)")
        except Exception as exc:
            err = run_dir(job); err.mkdir(parents=True, exist_ok=True)
            (err / "error.txt").write_text(traceback.format_exc())
            log(f"{worker_name}: FAIL  {jid}: {type(exc).__name__}: {exc}")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    log(f"{worker_name}: exit")


def run_queue(jobs, hours, n_gpus=None, poll=60):
    """Run jobs on all visible GPUs in parallel until done or out of time."""
    import torch
    QUEUE.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(QUEUE / "locks", ignore_errors=True)
    (QUEUE / "STOP").unlink(missing_ok=True)
    todo = [j for j in jobs if not is_done(j)]
    if not todo:
        log("Nothing to run: every job of this session is already complete.")
        return
    jobs_file = QUEUE / f"jobs_{int(time.time())}.json"
    jobs_file.write_text(json.dumps(todo))
    if n_gpus is None:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_workers = max(1, n_gpus)
    deadline = time.time() + hours * 3600.0
    log(f"Launching {n_workers} worker(s) for {len(todo)} pending job(s); budget {hours:.2f} h")
    procs = []
    for w in range(n_workers):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(w) if n_gpus > 0 else ""
        env["AFIS_WORK"] = str(WORK)
        logf = open(QUEUE / f"worker{w}.log", "a")
        code = ("import sys; sys.path.insert(0, %r); import afis_core as A; "
                "A.worker_main(%r, %r, %r)" % (str(Path(__file__).resolve().parent), str(jobs_file),
                                                deadline, f"gpu{w}"))
        procs.append((subprocess.Popen([sys.executable, "-c", code], env=env, stdout=logf,
                                       stderr=subprocess.STDOUT, cwd=str(WORK)), logf))
        time.sleep(5)
    while any(p.poll() is None for p, _ in procs):
        time.sleep(poll)
        done = sum(is_done(j) for j in todo)
        failed = sum((run_dir(j) / "error.txt").exists() and not is_done(j) for j in todo)
        left = (deadline - time.time()) / 3600
        log(f"progress: {done}/{len(todo)} done, {failed} failed, {left:.2f} h of budget left")
    for p, f in procs:
        f.close()
    done = sum(is_done(j) for j in todo)
    log(f"Queue finished: {done}/{len(todo)} jobs completed this session.")
    for w in range(n_workers):
        lines = (QUEUE / f"worker{w}.log").read_text().splitlines()
        fails = [l for l in lines if " FAIL " in l]
        if fails:
            log(f"worker{w} failures:\n  " + "\n  ".join(fails[-10:]))


def status_table(jobs):
    import pandas as pd
    rows = []
    for j in jobs:
        st = "done" if is_done(j) else ("failed" if (run_dir(j) / "error.txt").exists() else "pending")
        rows.append(dict(session=j["session"], family=j["family"], model=j["model"], exp=j["exp"], status=st))
    df = pd.DataFrame(rows)
    return df.groupby(["session", "family", "model", "exp", "status"]).size().unstack(fill_value=0)


# =============================================================================
# 9. Carry results between Kaggle sessions
# =============================================================================
def restore_previous(input_root="/kaggle/input"):
    """Copy results of earlier sessions (attached as notebook output) into WORK."""
    hits = glob.glob(os.path.join(str(input_root), "**", MARKER), recursive=True)
    if not hits:
        log("No previous-session results attached (fine for session 1).")
        return None
    # newest marker wins
    hits.sort(key=lambda p: json.loads(Path(p).read_text()).get("saved_at", 0))
    src = Path(hits[-1]).parent
    if src.resolve() == RESULTS.resolve():
        return src
    log(f"Restoring previous results from {src}")
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        dst = RESULTS / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst)
    # a fresh session never inherits stale locks / failures
    shutil.rmtree(QUEUE / "locks", ignore_errors=True)
    for e in RUNS.rglob("error.txt") if RUNS.exists() else []:
        e.unlink()
    return src


def write_marker(session):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / MARKER).write_text(json.dumps({"saved_at": time.time(), "session": session}))
