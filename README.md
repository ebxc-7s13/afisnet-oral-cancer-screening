# Texture-Preserving Generative Augmentation for Representation Learning in Limited-Data Autofluorescence Cytology

Code for the paper:

> **Texture-Preserving Generative Augmentation for Representation Learning in Limited-Data Autofluorescence Cytology**
> Siluveru Raja Viveka Vardhan, Sk Sher Md, Mousumi Pal, Ananya Barui
> Centre for Healthcare Science and Technology, Indian Institute of Engineering Science and Technology Shibpur; Guru Nanak Institute of Dental Sciences and Research
> Submitted to *Biomedical Signal Processing and Control*.

Oral cancer is often diagnosed late, and diagnosis still relies on invasive biopsy. Label-free confocal autofluorescence imaging (AFI) of exfoliated oral epithelial cells is a label-free way to study the autofluorescence of these cells, but clinical AFI datasets are small. The workflow here is investigational and has not been validated for participant-level diagnosis.

This repository contains every script used in the study:
- a class-conditional StyleGAN2-ADA with a **Neural Texture Preservation (NTP) loss** that matches class-level texture statistics and global grey-scale fluorescence statistics of autofluorescent cells;
- an automated **quality-control (QC)** pipeline for synthetic images;
- the **AFiS-Net** classifier (Autofluorescence Intelligent Screening Network) and three reference architectures;
- **region-grouped five-fold cross-validation**, repeated over three seeds;
- the controlled experiments on synthetic-data quantity and curation;
- quantitative attribution analysis;
- **screening of an unseen suspicious group** (an exposure-defined group of active cigarette smokers without a diagnosed oral lesion), scored by models that never saw it.

The cancer class comprises histopathologically confirmed oral epithelial dysplasia or oral squamous cell carcinoma; the classifier output is the probability of this binary training class, not a validated probability of carcinoma.

## Main results reported in the paper

| Result | Value |
|---|---|
| AFiS-Net, real images only (15 runs) | AUC 0.955 ± 0.031 |
| AFiS-Net vs NextViT-S / CoAtNet-0 / ResNet-50 | +0.050 / +0.030 / +0.033 AUC (Holm-adjusted p ≤ 0.001) |
| AFiS-Net-XA, real only → + all 3,000 generated images | AUC 0.954 → 0.973 (Holm-adjusted p = 0.001) |
| QC-selected (215) vs random (215) synthetic images | no difference; ranking by QC score did not help |
| AFiS-Net + QC-clean set (2,764 images, hygiene filter, single seed) | AUC 0.969, macro F1 0.934 (random set of equal size: AUC 0.970) |
| Generator vs the 177 real images it never used (neither fitting nor checkpoint selection) | KID 0.016 (normal) / 0.012 (cancer), similar to 0.013 / 0.011 against its 431-image partition |
| Unseen suspicious group (294 cells, 127 regions) vs all 348 normal / 260 cancer cells | 32% called cancer-like vs 5% / 92%; mean cancer probability 0.36 vs 0.12 / 0.89; same order in all five fold models |

## Repository structure

```
.
├── gan_training.py            # Stage 1: class-conditional StyleGAN2-ADA + NTP loss; image generation
├── gan_quality_selection.py   # Stage 2: QC scoring and ranked selection of synthetic images
├── classification.py          # Stage 3: classifier benchmark (22 architectures), training, calibration, embeddings
├── kfold_classification.py    # Stage 4: region-grouped five-fold cross-validation
├── screening_analysis.py      # Stage 5: stand-alone screening of the held-out cohort
├── experiments/
│   ├── afis_core.py           # experiments of the paper: job queue, folds, synthetic sources, AFiS-Net variants,
│   │                          #   pre-declared model-selection rule
│   ├── afis_analysis.py       # QC hygiene filter, matched-n generator metrics, texture analysis, attribution
│   │                          #   metrics, paired statistics
│   └── afis_revision.py       # screening of the suspicious group, generator vs the 177 unseen images,
│                              #   hold-out and crop facts, region-clustered bootstrap
├── notebooks/
│   └── AFiSNet_Revision11_screening_and_checks.ipynb   # Kaggle notebook that ran afis_revision.py
├── paper_scripts/
│   ├── make_tables.py         # builds every results table of the paper from the result files
│   ├── integrity_check.py     # recomputes the numbers quoted in the paper and checks them against the manuscript
│   ├── integrity_check_screening.py   # same for the screening and the other session-11 numbers
│   ├── fig_final.py           # Figs 6 and 7 (ROC/confusion; quantity vs curation)
│   ├── make_figures.py        # Figs 6-8 and Supplementary Fig. S2
│   ├── make_revision_figures.py   # Fig. 10 and Supplementary Figs S1 and S4
│   ├── fig_attribution.py     # Fig. 9 (occlusion and Grad-CAM++ maps)
│   ├── v10/                   # manuscript v10 verification scripts (see below)
│   ├── final_model_analysis.py
│   ├── fix_texture_fig.py     # re-letters the panels of the texture figure
│   └── afisnet_architecture.tex   # Fig. 3, architecture diagram (TikZ)
├── requirements.txt
├── LICENSE
└── README.md
```

The five stage scripts are standalone. `kfold_classification.py`, `screening_analysis.py` and the modules in `experiments/` import `classification.py` and reuse its model registry, preprocessing, training loop and checkpoint format unchanged.

## Pipeline and settings

### 1. Synthetic image generation: `gan_training.py`

- **Model.** Class-conditional StyleGAN2-ADA with:
  - a projection discriminator;
  - adaptive discriminator augmentation (target 0.6);
  - lazy R1 regularisation (γ = 0.8, every 16 iterations);
  - LeCam regularisation (λ = 0.05);
  - style mixing (0.9) and path-length regularisation (weight 2);
  - a generator EMA with a 20-kimg half-life.
- **Training.** Adam (β₁ = 0, β₂ = 0.99, learning rate 2 × 10⁻³), batch 16, up to 3,000 kimg (early stop after 600 kimg without a KID improvement of 5 × 10⁻⁴), at 224 × 224 resolution. Training images are random 224 crops of the 256 × 256 single-cell images, with flips and 90° rotations.
- **NTP loss.** Five distribution-level statistics, each matched to class-wise EMA targets (decay 0.99) with a scale-invariant relative MSE (ε = 10⁻⁶):

  | Statistic | Weight |
  |---|---|
  | VGG-16 Gram matrices (relu1_2, relu2_2, relu3_3) | 1.0 |
  | VGG-16 feature mean/std (relu3_3, relu4_3) | 1.0 |
  | Haar wavelet band energy | 0.5 |
  | One-sided total-variation hinge | 0.1 |
  | Soft-masked fluorescence photometry (κ = 0.5, τ = 0.08) | 0.5 |

  The overall NTP weight is 0 for 20 epochs, then ramps linearly to 1 over the next 30 (one epoch = 22 iterations of 16 images from the fitting subset, so ≈ 7.0 and ≈ 10.6 kimg). The loss and its EMA targets are updated every 4th iteration on sub-batches of 8 images. The targets are computed only from real images of the training mini-batches (the fitting subset); the checkpoint-selection images never enter them. The fluorescence term uses grey-scale statistics and does not constrain the ratios between colour channels.
- **Data roles (`_deterministic_holdout`, `get_dataloaders`).** The training folder (the 431-image generator partition in the paper: 247 normal, 184 cancer) is split by a SHA-256 hash of the file name: `max(2, round(0.15 n))` images per class (37 normal, 28 cancer) form the checkpoint-selection subset and the remaining 366 (210 normal, 156 cancer) the fitting subset. Only the fitting subset feeds the discriminator and generator updates, R1, ADA and the NTP targets.
- **Checkpoint selection.** Every 25 kimg, KID is computed per class between 256 generated images and the checkpoint-selection images of that class (100 subsets of 24); the checkpoint with the lowest mean KID is saved as `generator_best.pth` and used for generation. FID is logged but not used. The 177 validation and test images of the paper are never seen by the generator.
- **Baseline.** `--no-ntp` gives the StyleGAN2-ADA baseline used in the paper.
- **Generation.** The EMA generator samples with truncation ψ = 0.9. The paper used two generations: 1,000 images per class (the default), from which the 215 QC-selected images were ranked, and a separate generation of 1,500 per class (the 3,000-image complete pool used for all other arms).

### 2. Quality control: `gan_quality_selection.py` and `experiments/afis_analysis.py`

**Ranked selection (QC-selected)** is done by `gan_quality_selection.py`. Every generated image is compared with the real training images of its class (DINOv2, CLIP, LPIPS) and scored as executed by `compute_final_scores()`:

S = [ w_m·m̃ + w_d·d̃ + w_c·c̃ + w_l·(1 − l̃) + w_ρ·(1 − ρ̃) ] · (1 − 0.25·o)

| Symbol | Term | Nominal weight | Weight used |
|---|---|---|---|
| m | DINOv2 class margin (mean top-10 similarity to own class − to other class) | 0.25 | 0.3333 |
| d | DINOv2 mean top-10 similarity to own class | 0.20 | 0.2667 |
| c | CLIP mean top-10 similarity to own class | 0.12 | 0.1600 |
| l | LPIPS distance to the DINOv2-nearest real image (inverted) | 0.08 | 0.1067 |
| ρ | mean DINOv2 distance √(2 − 2cos) to the 10 nearest real images (inverted, i.e. local density) | 0.03 | 0.0400 |
| o | isolation-forest outlier flag ∈ {0, 1}, fitted on the generated images of the class | – | ×(1 − 0.25 o) |

A tilde is min–max normalisation over the generated images of the class. The weights of the disabled terms (discriminator, IQA) are redistributed by `redistribute_weights()` (factor 1/0.75), and the sum is not renormalised; neither affects the ranking. Images are sorted by decreasing S.

> **Known issue, kept as run.** A nearest-neighbour-distance term (nominal weight 0.07) is defined, but the weight key `nn_dist` looks for the column `n_nn_dist` while the normalised column is `n_nn`, so the term never enters the score. The code is left unchanged because this is how every reported selection was produced; `paper_scripts/v10/verify_qc_score.py` reproduces the saved per-image scores only without this term (max. error 6 × 10⁻⁷).

The script then rejects class-ambiguous images (margin ≤ 0) and near-copies (max DINOv2 cosine ≥ 0.995 to a real image of the class), removes near-duplicates of already kept images (cosine > 0.98) in rank order, and keeps the highest-ranked images up to min(⌊0.4 N_gen⌋, ⌊0.5 N_real⌋) per class. With 1,000 generated images per class the cap of half the real images is binding, which gave the paper's 215 images (123 normal, 92 cancer). The QC-at-scale arm (`afis_analysis.qc_all_ntp`) re-runs the same script on the 3,000-image pool with `TOP_PERCENT = 0.8` and no cap (2,124 images).

**Hygiene filter (QC-clean)** is `afis_analysis.qc_clean_ntp`. It applies no ranking. An image is removed only if it fails a check that real cells of its class almost never fail:
- a technical defect;
- a near-copy of a real image (≥ 0.995);
- an unrealistic image (below the real 1st percentile of nearest-real similarity);
- extreme class ambiguity (below the real 5th percentile of class margin);
- a duplicate (≥ 0.98).

Reference distributions are computed leave-one-region-out, and no criterion removes more than the extreme 2.5–5% tail of the pool. In the paper this kept 2,764 of 3,000 images.

The paper's central result is that adding *more* synthetic images helps, while ranking them by QC score does not. The ranked selection is therefore an experimental arm, not a required step.

### 3. Classification: `classification.py`

The script fine-tunes ImageNet-pretrained models:
- 6 torchvision and 16 timm models;
- including AFiS-Net, registered as `AFiSNet`, which is the cross-attention + GeM design called **AFiS-Net-XA** in the paper.

Preprocessing:
- deterministic foreground crop (Otsu threshold on the maximum-intensity projection, 8% padding, skipped below 1% or above 92% foreground);
- resize to 224 × 224;
- training augmentation: flips, ±15° rotation, 5% translation, elastic deformation (α = 15, σ = 3) and Gaussian noise (σ = 0.005);
- no photometric augmentation.

Training:
- AdamW, learning rate 10⁻⁴ for the head and 10⁻⁵ for the backbones;
- backbones frozen for 2 warm-up epochs, then cosine decay;
- batch 32, at most 50 epochs, early stopping with patience 10;
- gradient clipping at 1.0;
- loss: class-weighted cross-entropy with label smoothing 0.05, plus a supervised contrastive term (weight 0.1, temperature 0.1) on the L2-normalised 256-D embedding;
- explicit L1 (10⁻⁸) and L2 (10⁻⁶) penalties, with the decoupled weight decay set to 0.

Model selection and calibration:
- the checkpoint with the best validation macro F1 is kept;
- probabilities are calibrated by temperature scaling on the validation set;
- the decision threshold is set there by Youden's J.

The script also writes per-run metrics, Grad-CAM++ maps, embedding banks and diagonal-Mahalanobis references.

### 4. Region-grouped cross-validation: `kfold_classification.py`

Several single-cell images can come from the same imaging region, so the 608 real images are split with `StratifiedGroupKFold` on (ROI id, acquisition date) parsed from the filename. Fold *i* is the test set, fold *i*+1 the validation set, and the remaining three folds the training set. The split is repeated with seeds 42, 100 and 123.

Synthetic images are added only to the training folds, and every model sees identical folds, so all comparisons are paired.

### 5. Experiments in the paper: `experiments/`

`afis_core.py` defines every cross-validation job of the paper:
- reference architectures (NextViT-S, CoAtNet-0, ResNet-50);
- synthetic sources: QC-selected, random 215, all 3,000, 500/1,000/2,000 random, top-ranked 2,124, QC-clean 2,764, and random sets matched to these;
- the reduced-real-data series (25/50/75%);
- the fold-aware leakage control;
- the AFiS-Net ablation (`AblationHybrid`: CNN only, transformer only, concatenation, average pooling, both, no contrastive loss);
- the pre-declared rule for the final configuration, `choose_final_model`: candidates within 0.005 of the best mean AUC are equivalent, and the simplest wins.

The final AFiS-Net is **`AFiSNet_ConcatAvg`**: average-pooled FPN levels of ConvNeXt V2-Nano (768-D), concatenated with the average-pooled, layer-normalised deepest Swin V2-Tiny stage (384-D). Jobs are queued and resumable across GPU sessions. Paths default to a Kaggle layout and can be changed with the `AFIS_WORK` environment variable and the `input_root`/`overrides` arguments.

`afis_analysis.py` contains the analyses of the paper:
- the QC hygiene filter;
- generator metrics at matched sample size (FID, KID, precision/recall/density/coverage);
- Fourier and LBP texture analysis;
- occlusion and Grad-CAM++ evaluation (share of attribution inside the cell, pointing game, deletion/insertion);
- paired statistics: Wilcoxon with Holm adjustment, DeLong, Jonckheere–Terpstra, Kruskal–Wallis and Mann–Whitney.

It also holds the earlier 18-image screening (`screening_v8`), which the paper no longer uses.

`afis_revision.py` contains the analyses added for the submitted version, run by the notebook in `notebooks/`:
- **screening of the suspicious group** (`screening_full`): all 608 development images are scored by the fold model that held them out, and each imaging region of the suspicious group is assigned to one of the five final AFiS-Net models, so every image is scored by exactly one model that never saw it. Group comparison with image- and region-level Jonckheere–Terpstra, Kruskal–Wallis and Holm-adjusted Mann–Whitney tests, Cliff's δ and region-bootstrap confidence intervals; sensitivity analyses with the mean of the five models, each fold model separately, and normal/cancer images re-encoded as JPEG like the suspicious-group files; diagonal-Mahalanobis novelty. Exact duplicates, near-duplicates and copies of development images are removed from the suspicious group automatically;
- generator metrics against the 177 real images not used to fit the generator (`gan_unseen`);
- generator hold-out counts and foreground-crop statistics (`facts`);
- region-clustered bootstrap of pooled out-of-fold AUC (`cluster_bootstrap`; not reported in the paper);
- an optional downstream comparison of the NTP and baseline generators (`ntp_jobs`; not run for the paper).

Internally the suspicious group keeps the folder and key name `smoke`.

### 6. Stand-alone screening: `screening_analysis.py`

This script applies a trained k-fold ensemble to a held-out cohort. It reports per image:
- calibrated probability and prototype probability;
- entropy and epistemic uncertainty;
- Mahalanobis novelty.

It also gives group and ROI summaries. The screening numbers in the paper come from `afis_revision.screening_full`.

## Manuscript version 10: verification scripts

`paper_scripts/v10/` checks the manuscript against the code and the saved outputs; none of these scripts trains or generates anything.

| Script | Checks |
|---|---|
| `verify_qc_score.py` | recovers the QC ranking equation from a saved `scores.csv` and the selection rule (2,124-image run) |
| `check_qc_selected_provenance.py` | the 215 QC-selected images are not part of the 3,000-image pool (separate generation) |
| `audit_screening.py` | held-out scoring, one fold model per suspicious-group region, per-model thresholds, and all screening numbers (17 checks) |
| `make_figS1.py` | Supplementary Fig. S1 with the generator-partition label |

## How to run

```bash
pip install -r requirements.txt

python gan_training.py --data data/train                 # train the generator (add --no-ntp for the baseline)
python gan_training.py --mode generate --weights checkpoints/gan/generator_best.pth --per-class 1500
python gan_quality_selection.py                          # QC-selected set
python classification.py --models all                    # benchmark on the fixed split
python kfold_classification.py                           # region-grouped cross-validation
python screening_analysis.py --checkpoint results/kfold/AFiSNet/Raw_SelectedSynthetic --data-root data/screening
```

The screening of the suspicious group and the other session-11 analyses run in `notebooks/AFiSNet_Revision11_screening_and_checks.ipynb` (Kaggle, 2×T4). It restores the earlier results, reuses the seed-42 checkpoints of the final AFiS-Net, and writes `afis_results/revision11/`.

To build the paper's tables and figures and check the manuscript numbers from the experiment results:

```bash
python paper_scripts/make_tables.py RES7 RES9 record_based_tests.csv out/
python paper_scripts/make_revision_figures.py RES/revision11 out/
python paper_scripts/fig_attribution.py RES out/fig_explainability
python paper_scripts/integrity_check.py MANUSCRIPT_DIR RES7 RES9 record_based_tests.csv
python paper_scripts/integrity_check_screening.py MANUSCRIPT_DIR SUPPLEMENT_DIR RES/revision11
```

Every script accepts `--help`. Dataset locations can be set with CLI flags or the `AFIS_*` environment variables documented in `classification.py`.

## Data

The scripts expect class-named folders of single-cell images:

```
data/
├── train/{normal,cancer}/
├── val/{normal,cancer}/
├── test/{normal,cancer}/
└── screening/smoke/                      # suspicious group (tobacco smokers), held out from all training
```

Filenames follow `roi_<id>_<YYYYMMDD>_<HHMMSS>*.png`. The ROI id and acquisition date define the region groups.

**Only the code is distributed here.** Patient privacy and institutional ethics requirements prevent publishing the confocal autofluorescence images; they are available from the corresponding author on reasonable request. The per-run metrics, paired tests, per-image QC decisions, attribution metrics, generator metrics and per-image screening results are provided as CSV files in the supplementary material of the paper.

## Hardware

In the paper, the generator was trained on one NVIDIA RTX 6000. The classifier experiments ran on NVIDIA T4 GPUs with mixed precision.

## Naming

| Code identifier | Name in the paper |
|---|---|
| `AFiSNet_ConcatAvg` | AFiS-Net (final configuration) |
| `AFiSNet` | AFiS-Net-XA (cross-attention + GeM) |
| `NextViT_Small` | NextViT-S |
| `CoAtNet0` | CoAtNet-0 |
| `ResNet50` | ResNet-50 |

## Funding

This work was supported by the IIT Kharagpur AI4ICPS I-Hub Foundation [grant number DRC/IITKGP-AI4ICPS I HF/CHST/AB/019/23-24].

## License

MIT License, see [LICENSE](LICENSE).
