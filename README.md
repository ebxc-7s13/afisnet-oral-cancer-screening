# Quality-Controlled Synthetic Augmentation for AI-Enabled Label-Free Digital Cytology of Oral Cancer Screening

Official code repository accompanying the research paper:

> **Quality Controlled Synthetic Augmentation for AI-Enabled Label-free Digital Cytology of Oral Cancer Screening**
> Siluveru Raja Viveka Vardhan, Sk Sher Md, Mousumi Pal, Ananya Barui
> Centre for Healthcare Science and Technology, Indian Institute of Engineering Science and Technology Shibpur; Guru Nanak Institute of Dental Sciences and Research

Oral cancer is frequently diagnosed late, and diagnosis still relies on invasive biopsy. Label-free confocal autofluorescence imaging (AFI) of exfoliated oral epithelial cells detects the metabolic changes of malignant transformation non-invasively, but the limited size of clinical AFI datasets makes reliable deep-learning training difficult.

This repository implements the complete study workflow:

**GAN synthesis → Quality-controlled selection → Synthetic augmentation → Classification → Leakage-safe validation → Independent screening**

A class-conditional StyleGAN2-ADA generator, extended with a Neural Texture Preservation (NTP) loss that protects the sparse, high-frequency nuclear and perinuclear detail of AFI, synthesises realistic single-cell images. Crucially, **not every generated image is used for training**: every candidate passes through an automated quality-control stage that scores quality, diversity, and class consistency, and only the selected subset is added to the real training set. Classifiers — including the proposed dual-branch convolution–transformer **AFiS-Net** — are then benchmarked, re-validated under leakage-safe region-grouped cross-validation, and finally applied to an independent, entirely unseen cohort of tobacco smokers to test whether the learned malignancy-related feature space places them between the normal and cancer classes.

## Overview

- **Task**: binary classification (normal vs cancer) of 256×256 single-cell pseudo-RGB confocal autofluorescence images, followed by feature-space screening of an unseen intermediate-risk cohort.
- **Data scarcity solution**: texture-preserving class-conditional GAN synthesis with automated, conservative quality control before augmentation.
- **Evaluation discipline**: fixed region-level splits, three random seeds, temperature calibration, validation-only threshold selection, and region-grouped five-fold cross-validation so that frames from the same physical imaging region never cross partitions.
- **Interpretability**: Grad-CAM++ and cell-focused attribution maps, embedding visualisations, and Mahalanobis out-of-distribution (novelty) scoring.

## Pipeline

```
Real Autofluorescence Images
        ↓
Class-Conditional GAN (StyleGAN2-ADA + NTP loss)        gan_training.py
        ↓
Synthetic Images (1,000 per class)
        ↓
Quality-Controlled Synthetic Image Selection            gan_quality_selection.py
        ↓
Raw + Selected Synthetic Training Dataset
        ↓
Deep Learning Classification (22 architectures)         classification.py
        ↓
Leakage-Safe Region-Grouped Cross-Validation            kfold_classification.py
        ↓
Independent Screening Analysis (unseen cohort)          screening_analysis.py
```

## Repository Structure

```
.
├── gan_training.py            # Stage 1 — GAN training & image generation
├── gan_quality_selection.py   # Stage 2 — synthetic-image quality control
├── classification.py          # Stage 3 — classifier benchmarking (Raw vs Raw+Synthetic)
├── kfold_classification.py    # Stage 4 — leakage-safe region-grouped k-fold CV
├── screening_analysis.py      # Stage 5 — independent screening cohort analysis
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The stages are deliberately kept as five standalone, executable scripts (no package structure). `kfold_classification.py` and `screening_analysis.py` import `classification.py` to reuse its model registry, preprocessing, training loop, and checkpoint format unchanged.

## Workflow

### 1. GAN Training and Image Generation — `gan_training.py`

Trains the class-conditional StyleGAN2-ADA generator (projection discriminator, ADA, R1 and LeCam regularisation, style mixing, path-length regularisation, generator EMA) with the proposed **Neural Texture Preservation loss**: five unpaired, distribution-level statistics-matching terms (VGG-16 Gram matrices, VGG-16 feature mean/std, Haar-wavelet band energies, a one-sided total-variation hinge, and soft-masked fluorescence-photometry statistics), each matched to class-wise EMA targets computed from real images with a scale-invariant relative MSE. The NTP coefficient is warm-started for 20 epochs and linearly ramped over the next 30. Generator checkpoints are selected by Kernel Inception Distance on an internal, deterministically held-out validation subset (never on the generator's own training images). After training, the EMA generator writes 1,000 images per class into class-named folders. `--no-ntp` reproduces the StyleGAN2-ADA ablation baseline reported in the paper.

### 2. Synthetic Image Quality Selection — `gan_quality_selection.py`

The quality-control stage central to the paper's contribution. Every generated image is scored with DINOv2 and CLIP similarity to its nearest real neighbours, LPIPS perceptual distance, nearest-neighbour distance, local density, a DINOv2 **class-margin** criterion (class consistency), and an outlier penalty (Isolation Forest). Safeguards reject exact duplicates, near-copies of real training images (the memorisation safeguard), cross-class-ambiguous images, and near-duplicate synthetic images (greedy diversity filter). At most 40 % of candidates — capped at 0.5 × the number of real training images per class — survive. **Only this selected subset proceeds to classifier augmentation.** The selected set is additionally evaluated against the real distribution with FID, KID, and precision/recall/density/coverage. All thresholds are defined at the top of the script; the process is fully automated with no manual inspection.

### 3. Classification — `classification.py`

Fine-tunes 22 ImageNet-pretrained architectures (6 torchvision baselines + 16 timm encoders spanning CNNs, vision transformers, and conv–attention hybrids, including the proposed **AFiS-Net**, registered as `CAFNet_Hybrid`) under two conditions — `Raw` (real images only) and `Raw_SelectedSynthetic` (real + quality-selected synthetic images added to training only) — across three seeds. It reports accuracy, balanced accuracy, per-class sensitivity/specificity, macro F1, MCC, Cohen's kappa, ROC-AUC, Brier score, and calibration (ECE/MCE) with validation-only temperature scaling and Youden-J thresholding; produces Grad-CAM++ and cell-focused attribution maps, hard-example analyses, embedding banks with Mahalanobis OOD references (consumed later by the screening stage), paired significance tests for Raw vs Raw+Synthetic, McNemar tests, and an automatic benchmarking/ranking report.

### 4. Leakage-Safe Region-Grouped Cross-Validation — `kfold_classification.py`

This stage exists because multiple single-cell frames can originate from the **same physical imaging region**, acquired seconds apart. A conventional frame-level split can place near-identical frames in both training and test sets, producing optimistic estimates that are not valid generalisation performance. This script therefore pools all real images and applies **Leakage-Safe Region-Grouped K-Fold Cross-Validation**: `StratifiedGroupKFold` on a group key of (ROI id, acquisition date) parsed from the filenames, so that every frame of a region stays inside a single fold and no region is ever shared between the training, validation, and test partitions (asserted at runtime). Quality-selected synthetic images are added to the training folds only, and identical folds are used for every model so comparisons are paired. The four models carried forward are `CAFNet_Hybrid` (AFiS-Net), `NextViT_Small` (transformer), `CoAtNet0` (conv–attention hybrid), and `ResNet50` (CNN baseline). The paper's headline results are the cross-validated estimates from this stage.

### 5. Independent Screening Analysis — `screening_analysis.py`

Evaluates the trained binary encoder on the independent screening cohort (non-smoker / smoker / cancer). The smoker ("suspected") group was **unseen during every stage of training, model selection, and threshold selection**; this script does *not* claim the binary classifier diagnoses it. Instead, it investigates whether the group's samples occupy an intermediate position within the learned malignancy-related feature space. Using a k-fold ensemble of deployment checkpoints, it reports per image the calibrated classifier cancer probability, an independent prototype/Mahalanobis-based probability, confidence, predictive entropy, epistemic uncertainty (spread across ensemble members), and a Mahalanobis novelty (OOD) score; aggregates per group and per ROI; and computes the progression statistics (Spearman monotonic trend, Kruskal–Wallis, one-sided pairwise Mann–Whitney U, and a normal-vs-cancer AUC internal validation), together with figures and a text report.

## How to Run

Run the stages in order (defaults reproduce the configuration used in the paper):

```bash
python gan_training.py
python gan_quality_selection.py
python classification.py --models all
python kfold_classification.py
python screening_analysis.py --checkpoint results/kfold/CAFNet_Hybrid/Raw_SelectedSynthetic --data-root data/screening
```

Useful variants:

```bash
python gan_training.py --no-ntp                        # StyleGAN2-ADA ablation baseline
python gan_training.py --mode generate --weights checkpoints/gan/generator_best.pth
python classification.py --list-models                 # show all 22 registered models
python kfold_classification.py --no-synthetic          # Raw-only cross-validation
```

Every script accepts `--help`. Dataset locations can be overridden with CLI flags (`--data`, `--real-folder`/`--gan-folder`/`--output-folder`, `--train-root`/`--val-root`/`--test-root`/`--synthetic-root`, `--data-root`) or the `AFIS_*` environment variables documented in `classification.py`.

## Dataset Structure

The scripts expect class-named folders of single-cell images (PNG/JPG/TIFF/BMP):

```
data/
├── train/                 # real training images (development set)
│   ├── normal/
│   └── cancer/
├── val/                   # real validation images
│   ├── normal/
│   └── cancer/
├── test/                  # real held-out test images
│   ├── normal/
│   └── cancer/
└── screening/             # independent screening cohort (held out entirely)
    ├── normal/
    ├── smoke/             # aliases such as "smoker" are recognised
    └── cancer/
```

Real filenames follow the pattern `roi_<id>_<YYYYMMDD>_<HHMMSS>*.png`; the ROI id and acquisition date are parsed from the filename to build the leakage-safe region groups used in Stages 4–5. Generated and selected synthetic images are written to `outputs/gan/final_generated/` and `outputs/selected_synthetic/` respectively, matching the defaults of the downstream stages.

**This repository provides the implementation only.** The patient-derived confocal autofluorescence images cannot be made publicly available because of institutional patient-privacy and ethical requirements; data are available from the corresponding author upon reasonable request.

## Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended (experiments in the paper used a single NVIDIA RTX 4000 Ada, 20 GB). Optional accelerations and extras (`faiss-cpu`, `umap-learn`, `pyiqa`, `tensorboard`, `pynvml`) are listed, commented out, in `requirements.txt`; every script degrades gracefully without them.

## Notes on Naming

The model registry key `CAFNet_Hybrid` in the code corresponds to the architecture named **AFiS-Net** in the manuscript (the registry key is preserved so that trained deployment checkpoints, which store the backbone name, remain loadable). Likewise, `NextViT_Small` and `CoAtNet0` are the "Transformer" and "Conv–attention" models of the paper's cross-validation table.

## Citation

If you use this code, please cite:

```bibtex
@article{vardhan_quality_controlled_synthetic,
  title   = {Quality Controlled Synthetic Augmentation for AI-Enabled Label-free Digital Cytology of Oral Cancer Screening},
  author  = {Siluveru Raja Viveka Vardhan and Sk Sher Md and Mousumi Pal and Ananya Barui},
  journal = {...},
  year    = {...},
  doi     = {...}
}
```

(Journal, year, and DOI will be added upon publication.)

## Acknowledgements

This work was supported by the IIT Kharagpur AI4ICPS I-Hub Foundation under Grant No. DRC/IITKGP-AI4ICPS I HF/CHST/AB/019/23-24.

## License

This repository is released under the [MIT License](LICENSE).
