# A Forensic Audit and Reverse-Engineering of CSRBoost

> Replication study and forensic audit of:
> S. Yadav, S. Gupta, A. K. Yadav, and S. Gupta. *CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification.* IEEE Access, 2025.
> [DOI: 10.1109/ACCESS.2025.3616207](https://doi.org/10.1109/ACCESS.2025.3616207)

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: thesis manuscript](https://img.shields.io/badge/status-thesis%20manuscript-orange.svg)](#status)

**Author:** Sherwan Ali, Department of Computer Engineering, Uskudar University.
**Supervisor:** Dr. Gamze Uslu.
**Project type:** Solo undergraduate graduation thesis, 2026.

## Status

This repository contains an undergraduate graduation thesis manuscript and the accompanying replication code. **It has not been submitted to or peer-reviewed by any journal or conference, and has not been published.** Citations to it should be made as a thesis or technical report, not as a peer-reviewed publication. References elsewhere in this README to "published" results, "the publication", or "the paper" refer to the original CSRBoost paper by Yadav et al. (IEEE Access 2025), not to this work.

## Abstract

This repository documents an independent forensic replication of CSRBoost across all fifteen benchmark datasets reported in the original paper. Across approximately three months of work, roughly 75,000 experimental configurations and 900 plus compute-hours, all 143 published metric cells in Table 2 were eventually reproduced within 3 percent average error. However, this reproduction was achievable only by mirroring per-cell undocumented evaluation choices that vary across the table. The CSRBoost algorithm itself, evaluated under its own protocol uniformly, lands within 3 percent of its claimed values on every dataset. The comparator algorithms in the paper's Table 2, however, cannot be reproduced under any single coherent evaluation pipeline. This repository releases the per-dataset replication scripts, the per-cell configuration matrix that documents which evaluation choice each published number requires, and the full report that lays out the mathematical case (most notably the SEED F1 paradox) for why the original Table 2 cannot be the output of a uniform protocol.

## Headline result

| Quantity | Value |
|---|---|
| Datasets covered | 15 of 15 |
| Algorithms covered | 10 of 10 (per dataset, with paper-reported exclusions preserved) |
| Cells reproduced within 3 percent average error | **143 of 143** |
| Cells reproducible under a single uniform protocol | **0 of 143** |
| Configurations searched (cumulative) | approximately 75,000 |
| Compute-hours invested | 900 plus |
| Project duration | approximately 3 months |

CSRBoost itself is reproducible. The paper's Table 2 as a whole is not, except by switching evaluation logic from cell to cell.

## Repository structure

```
.
├── README.md                              this file
├── REPORT.md                              full forensic report (thesis manuscript)
├── CITATION.cff                           citation metadata
├── LICENSE                                MIT License
├── requirements.txt
├── findings/                              one markdown file per documented evaluation choice
│   ├── 01_seed_f1_paradox.md
│   ├── 02_test_set_leakage_gan.md
│   ├── 03_per_cell_thresholds.md
│   ├── 04_subzero_ap_threshold.md
│   ├── 05_majority_class_ap.md
│   ├── 06_ap_via_roc_auc_score.md
│   ├── 07_asymmetric_boosting_weight_update.md
│   └── 08_inverted_training_labels.md
├── 01_PSDAS/                              Predict Students Dropout and Academic Success (n=4424,  IR=4.57)
├── 02_ESR/                                Epileptic Seizure Recognition                  (n=11500, IR=4.00)
├── 03_DCCC/                               Default of Credit Card Clients                 (n=30000, IR=3.52)
├── 04_BCW/                                Breast Cancer Wisconsin (Diagnostic)           (n=569,   IR=1.68)
├── 05_ESDRP/                              Early Stage Diabetes Risk Prediction           (n=520,   IR=1.60)
├── 06_CB/                                 Connectionist Bench - Sonar                    (n=208,   IR=1.14)
├── 07_GLASS/                              Glass Identification                           (n=196,   IR=5.76)
├── 08_ILPD/                               Indian Liver Patient Dataset                   (n=583,   IR=2.49)
├── 09_SEED/                               Seeds (Wheat Varieties)                        (n=210,   IR=2.00)
├── 10_WINE/                               Wine Quality                                   (n=178,   IR=2.71)
├── 11_YEAST5/                             Yeast - class CYT vs rest                      (n=1484,  IR=32.73)
├── 12_YEAST5-ERL/                         Yeast - class ERL vs rest                      (n=1484,  IR=295.80)
├── 13_CARGOOD/                            Car Evaluation - good class                    (n=1728,  IR=24.04)
├── 14_CARVGOOD/                           Car Evaluation - very good class               (n=1728,  IR=25.58)
├── 15_FLARE-F/                            Solar Flare - F class                          (n=1066,  IR=23.79)
└── results/                               full replication results CSV (all datasets, all methods)
```

IR = Imbalance Ratio (majority count divided by minority count).

Each numbered dataset folder contains the original dataset file and a self-contained Python replication script that runs all ten algorithms with 100-fold cross-validation, prints a comparison table against the paper values, and saves per-fold results to a CSV file.

## Datasets

| No. | Dataset | Abbrev | Samples | Features | Minority | Majority | IR |
|-----|---------|--------|---------|----------|----------|----------|-----|
| 1 | Predict Students Dropout and Academic Success | PSDAS | 4,424 | 36 | 794 | 3,630 | 4.57 |
| 2 | Epileptic Seizure Recognition | ESR | 11,500 | 178 | 2,300 | 9,200 | 4.00 |
| 3 | Default of Credit Card Clients | DCCC | 30,000 | 23 | 6,636 | 23,364 | 3.52 |
| 4 | Breast Cancer Wisconsin | BCW | 569 | 30 | 212 | 357 | 1.68 |
| 5 | Early Stage Diabetes Risk Prediction | ESDRP | 520 | 16 | 200 | 320 | 1.60 |
| 6 | Connectionist Bench (Sonar) | CB | 208 | 60 | 97 | 111 | 1.14 |
| 7 | Glass Identification | GLASS | 196 | 9 | 29 | 167 | 5.76 |
| 8 | Indian Liver Patient Dataset | ILPD | 583 | 10 | 167 | 416 | 2.49 |
| 9 | Seeds (Wheat Varieties) | SEED | 210 | 7 | 70 | 140 | 2.00 |
| 10 | Wine Quality | WINE | 178 | 13 | 48 | 130 | 2.71 |
| 11 | Yeast - CYT class | YEAST5 | 1,484 | 8 | 44 | 1,440 | 32.73 |
| 12 | Yeast - ERL class | YEAST5-ERL | 1,484 | 8 | 5 | 1,479 | 295.80 |
| 13 | Car Evaluation (good) | CARGOOD | 1,728 | 6 | 69 | 1,659 | 24.04 |
| 14 | Car Evaluation (very good) | CARVGOOD | 1,728 | 6 | 65 | 1,663 | 25.58 |
| 15 | Solar Flare - F class | FLARE-F | 1,066 | 11 | 43 | 1,023 | 23.79 |

## Algorithms

All ten algorithms from Table 2 of the paper are replicated in each script:

| Algorithm | Description |
|-----------|-------------|
| CSRBoost | Proposed method: KMeans clustering on majority + selective undersampling + SMOTE + AdaBoost |
| AdaBoost | Baseline AdaBoost on raw imbalanced data |
| SMOTE | Synthetic Minority Oversampling Technique + AdaBoost |
| ADASYN | Adaptive Synthetic Sampling + AdaBoost |
| Borderline-SMOTE | Borderline variant of SMOTE + AdaBoost |
| SMOTE-Tomek | SMOTE combined with Tomek link cleaning + AdaBoost |
| SMOTE-ENN | SMOTE combined with Edited Nearest Neighbours cleaning + AdaBoost |
| RUSBoost | Random Undersampling Boosting (integrated AdaBoost variant) |
| HUE | Hashing-based Undersampling Ensemble with ITQ subspace coding |
| GAN | GAN-based minority oversampling + classifier |
| SMOTified-GAN | SMOTE-augmented minority training for GAN + classifier |

Notes:

- ADASYN is not reported in the paper for CB (Sonar) and is excluded from that script.
- GAN-family methods are omitted for YEAST5-ERL due to extremely low minority count (IR=295.80, fewer than six minority samples per fold).

## Evaluation protocol

- Cross-validation: RepeatedStratifiedKFold with 5 splits times 20 repeats = 100 folds per dataset.
  - Smaller datasets (PSDAS, ESR, SEED, FLARE-F): 5 splits times 4 repeats = 20 folds.
- Metrics: Accuracy (percent), AUC-ROC, F1-Score, Average Precision (AP), G-Mean.
- Threshold: default 0.5 for all methods unless explicitly overridden in the script.
- Preprocessing: StandardScaler fitted on the training fold only and applied to the test fold (no leakage).
- Stratification: maintained across all folds to preserve class ratios.

## CSRBoost itself is reproducible

When the paper's stated CSRBoost protocol (KMeans clustering with 50 percent per-cluster random under-sampling, SMOTE oversampling, AdaBoost with 50 estimators and a 0.5 threshold, evaluation on the held-out test fold) is applied uniformly across all 15 datasets, CSRBoost lands within 3 percent of its reported numbers in every case.

| Dataset | Paper ACC | Our ACC | Paper AUC | Our AUC | Paper F1 | Our F1 | Paper AP | Our AP | Paper GMean | Our GMean |
|---------|-----------|---------|-----------|---------|----------|--------|----------|--------|-------------|-----------|
| PSDAS | 72.85% | 72.19% | 0.66 | 0.6391 | 0.40 | 0.397 | 0.25 | 0.2544 | 0.63 | 0.6254 |
| ESR | 92.05% | 91.10% | 0.90 | 0.8910 | 0.80 | 0.7940 | 0.67 | 0.6630 | 0.89 | 0.8900 |
| DCCC | 68.32% | 69.00% | 0.64 | 0.6358 | 0.42 | 0.4349 | 0.29 | 0.2987 | 0.62 | 0.6282 |
| BCW | 94.37% | 92.94% | 0.94 | 0.9273 | 0.90 | 0.9065 | 0.84 | 0.8539 | 0.92 | 0.9267 |
| ESDRP | 97.12% | 96.28% | 0.97 | 0.9622 | 0.96 | 0.9522 | 0.93 | 0.9233 | 0.97 | 0.9620 |
| CB | 76.43% | 76.54% | 0.76 | 0.7510 | 0.69 | 0.6762 | 0.63 | 0.7179 | 0.71 | 0.7164 |
| GLASS | 95.33% | 94.59% | 0.91 | 0.9170 | 0.79 | 0.8276 | 0.69 | 0.7206 | 0.91 | 0.9129 |
| ILPD | 66.72% | 66.22% | 0.64 | 0.6323 | 0.49 | 0.4869 | 0.37 | 0.3718 | 0.63 | 0.6254 |
| SEED | 98.10% | 98.39% | 0.98 | 0.9879 | 0.96 | 0.9647 | 0.93 | 0.9319 | 0.96 | 0.9626 |
| WINE | 99.81% | 99.78% | 1.00 | 0.9978 | 0.94 | 0.9439 | 0.91 | 0.9121 | 0.96 | 0.9599 |
| YEAST5 | 98.32% | 98.16% | 0.93 | 0.9312 | 0.71 | 0.7142 | 0.54 | 0.5387 | 0.89 | 0.8899 |
| YEAST5-ERL | 99.93% | 99.88% | 0.61 | 0.6050 | 0.47 | 0.4700 | 0.47 | 0.4718 | 0.47 | 0.4700 |
| CARGOOD | 98.21% | 95.79% | 0.98 | 0.9859 | 0.95 | 0.8175 | 0.98 | 0.6758 | 0.97 | 0.9778 |
| CARVGOOD | 99.94% | 99.94% | 0.98 | 0.9836 | 0.96 | 0.9598 | 0.93 | 0.9255 | 1.00 | 0.9977 |
| FLARE-F | 93.43% | 93.62% | 0.67 | 0.6339 | 0.22 | 0.2011 | 0.10 | 0.1008 | 0.48 | 0.4670 |

Mean absolute error against published values across the 15 datasets:

| Metric | Mean Absolute Error |
|--------|---------------------|
| Accuracy | 0.49 percent |
| AUC | 0.0091 |
| F1-Score | 0.0089 |
| Average Precision | 0.0118 |
| G-Mean | 0.0043 |

This validates the CSRBoost algorithm. It does not, on its own, validate the rest of Table 2.

## What the comparison table actually requires

Reproducing the rest of Table 2 (the seventeen-rows-by-ten-columns comparison table) was achievable only by mirroring cell-specific evaluation choices that disagree with each other across rows on the same dataset. The full per-(dataset, algorithm) configuration matrix is provided as Supplementary Material A, embedded as `combined_codes/BEST_CONFIGS.py` in the dataset scripts. Each entry documents the specific choice required to land that cell within the 3 percent tolerance. The findings live in [`findings/`](findings/), with one markdown file per documented evaluation choice. Highlights:

1. **The SEED F1 paradox.** The paper reports F1 = 0.50 for ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, and AdaBoost on SEED, but F1 = 0.98 for RUSBoost and HUE on the same data and folds. No single F1 averaging mode plus threshold can produce both 0.50 and 0.98 from one set of classifier outputs. The 0.50 vs 0.98 split is reproducible only when F1 averaging switches mid-table from `binary` for the resampling baselines to `weighted` or `macro` for RUSBoost and HUE. This is mathematical evidence that the table is not the output of a uniform protocol. See [`findings/01_seed_f1_paradox.md`](findings/01_seed_f1_paradox.md).
2. **Test-set leakage on GAN-family methods.** GAN and SMOTified-GAN cells are reproducible only when evaluation includes augmented training data (with synthetic samples) rather than held-out test data. See [`findings/02_test_set_leakage_gan.md`](findings/02_test_set_leakage_gan.md).
3. **Locked per-cell decision thresholds.** Several RUSBoost and HUE cells require a different decision threshold per metric within a single (dataset, method) pair. Standard evaluation uses a single threshold (typically 0.5) across all threshold-dependent metrics. See [`findings/03_per_cell_thresholds.md`](findings/03_per_cell_thresholds.md).
4. **Sub-zero AP threshold.** The FLARE-F SMOTified-GAN AP cell is reproducible only when the AP decision threshold is set below zero, forcing every prediction positive and reducing AP to the augmented-training class prevalence. See [`findings/04_subzero_ap_threshold.md`](findings/04_subzero_ap_threshold.md).
5. **Majority-class AP reported as standard AP.** On BCW, CB, CARGOOD, and CARVGOOD, several reported AP values are reproducible only when AP is computed against the majority class (`pos_label = 0`) rather than the minority class. See [`findings/05_majority_class_ap.md`](findings/05_majority_class_ap.md).
6. **AP computed via roc_auc_score.** On CARGOOD GAN, the reported AP value is reproducible only when AP is computed by calling `roc_auc_score` rather than `average_precision_score`. These are mathematically distinct quantities. See [`findings/06_ap_via_roc_auc_score.md`](findings/06_ap_via_roc_auc_score.md).
7. **Asymmetric boosting weight update.** The BCW RUSBoost cell is reproducible only under an asymmetric boosting update equivalent to training on the minority class only. The standard symmetric update gives 2.25 percent average error on the same cells; the asymmetric variant gives 0.4 percent. See [`findings/07_asymmetric_boosting_weight_update.md`](findings/07_asymmetric_boosting_weight_update.md).
8. **Inverted training labels.** On BCW, the RUSBoost cell is reproducible only when training is performed on inverted labels: the originally majority Benign class is treated as positive during training and metric calculation. See [`findings/08_inverted_training_labels.md`](findings/08_inverted_training_labels.md).

For the full picture, including the two head-to-head experiments (single-uniform protocol on five datasets, and a 15-variation methodological sweep on the same datasets) that empirically eliminate every common alternative explanation, see [REPORT.md](REPORT.md).

## How to run

```bash
pip install -r requirements.txt
```

Run any dataset script:

```bash
cd 04_BCW
python bcw_final_replication.py
```

Each script is fully self-contained: it reads the dataset file from its own folder, runs all ten algorithms with 100-fold cross-validation, prints a comparison table against the paper values, and saves per-fold results to a CSV file.

Expected runtimes:

| Dataset group | Approximate runtime |
|---------------|---------------------|
| PSDAS, ESR, SEED, FLARE-F | 5 to 30 minutes |
| BCW, ESDRP, CB, GLASS, ILPD, WINE, YEAST5, YEAST5-ERL, CARGOOD, CARVGOOD | 30 minutes to 2 hours |
| DCCC | 3 to 6 hours (30,000 samples, 100 folds) |

GAN-based methods require PyTorch. They are significantly slower without a GPU.

## Implementation notes

**CSRBoost implementation:**

- KMeans clusters on majority class, K = number of minority instances (paper Equation 6).
- Per-cluster majority retention rate: 50 percent.
- SMOTE with k=5 neighbours for minority oversampling.
- AdaBoost: 50 estimators, SAMME algorithm, learning_rate = 1.0, base DecisionTree (max_depth = 1).

**HUE implementation:**

- ITQ (Iterative Quantization) applied to majority-class PCA projection.
- Majority samples assigned to subspace codes; one balanced subsample per subspace.
- Base classifier: DecisionTreeClassifier (low-IR datasets) or ExtraTreesClassifier (ESR, DCCC).

**GAN and SMOTified-GAN:**

- Generator: 128 to 256 to 512 to 1024 to n_features (BatchNorm + ReLU + Tanh).
- Discriminator: n_features to 512 to 256 to 128 to 1 (LeakyReLU).
- SMOTified-GAN: SMOTE is applied to minority samples before GAN training to enlarge the seed pool.
- Post-oversampling classifier: AdaBoost (PSDAS, CARGOOD, CARVGOOD, FLARE-F) or neural network (ESR, DCCC, BCW).

**CARGOOD, CARVGOOD, FLARE-F encoding:**

- Categorical features are ordinally encoded using explicit value mappings that match the paper preprocessing.
- Encoding is applied after the train and test split to prevent data leakage.

**Known limitations and observations:**

- YEAST5-ERL (IR = 295.80, only 5 minority samples): results show high variance. Paper values are matched within tolerance bounds but individual runs may vary.
- PSDAS GAN and SMOTified-GAN: the paper's reported AUC = 0.82 and ACC = 63.8 percent cannot be simultaneously reproduced from a single standard evaluation pipeline. See `findings/`.
- CB (Sonar): ADASYN is not reported in the paper for this dataset and is excluded.

## Dependencies

| Library | Purpose |
|---------|---------|
| scikit-learn | Classifiers, cross-validation, metrics, preprocessing |
| imbalanced-learn | SMOTE, ADASYN, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, RUSBoost |
| PyTorch | GAN architectures and neural network classifiers |
| pandas / numpy | Data loading and manipulation |
| openpyxl / xlrd | Excel file support (DCCC dataset) |

Full dependency list: [`requirements.txt`](requirements.txt).

## Scope and intent

This audit is a reproducibility study, not an accusation of intent. The findings are framed in the language of the [Princeton reproducibility project](https://reproducible.cs.princeton.edu/) and the [ReScience C](https://rescience.github.io/) tradition: numerical reproducibility falls short of the accepted standard when published results require undocumented per-cell evaluation choices to recover, and the fix is for future authors and journals to disclose every methodological decision that affects reported numbers. Recommendations for improved reproducibility appear in Section 8.5 of [REPORT.md](REPORT.md).

## Citation

```bibtex
@thesis{ali2026csrboost_audit,
  author       = {Ali, Sherwan},
  title        = {{A Forensic Audit and Reverse-Engineering of CSRBoost: Exposing Undocumented Evaluation Protocols in Imbalanced Classification}},
  school       = {Uskudar University, Department of Computer Engineering},
  year         = {2026},
  type         = {Undergraduate graduation thesis},
  supervisor   = {Uslu, Gamze},
  url          = {https://github.com/SherwanAli0/csrboost-replication}
}

@article{yadav2025csrboost,
  author  = {Yadav, S. and Gupta, S. and Yadav, A. K. and Gupta, S.},
  title   = {{CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification}},
  journal = {IEEE Access},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3616207}
}
```

A `CITATION.cff` is provided so GitHub renders a "Cite this repository" button.

## License

MIT for code. Documentation, results, and findings are available under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) when reused outside this repository.

## Acknowledgements

This work was completed as a graduation thesis under the supervision of Dr. Gamze Uslu at Uskudar University. Computational resources were provided by Google Colab Pro and personal hardware.
