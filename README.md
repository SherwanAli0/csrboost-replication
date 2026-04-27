# CSRBoost Replication Study

Replication of the paper:

**CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification**
Yadav et al., IEEE Access, 2025.

Graduation project .
Student: Sherwan Ali
Supervisor: Dr. Gamze Uslu

---

## What This Repository Contains

A full replication of Table 2 from the CSRBoost paper across all 15 benchmark datasets.
Each dataset has its own folder containing the dataset file and the replication script.

The replication achieved **143 out of 143 metric-algorithm combinations within 3 percent average error** compared to the published paper values.

---

## Repository Structure

```
.
├── 01_PSDAS/              Predict Students Dropout and Academic Success
├── 02_ESR/                Epileptic Seizure Recognition
├── 03_DCCC/               Default of Credit Card Clients
├── 04_BCW/                Breast Cancer Wisconsin
├── 05_ESDRP/              Early Stage Diabetes Risk Prediction
├── 06_CB/                 Connectionist Bench (Sonar)
├── 07_GLASS/              Glass Identification
├── 08_ILPD/               Indian Liver Patient Dataset
├── 09_SEED/               Seeds Dataset
├── 10_WINE/               Wine Dataset
├── 11_YEAST5/             Yeast Dataset (class CYT vs rest)
├── 12_YEAST5-ERL/         Yeast Dataset (class ERL vs rest, IR=295)
├── 13_CARGOOD/            Car Evaluation (good class)
├── 14_CARVGOOD/           Car Evaluation (very good class)
├── 15_FLARE-F/            Solar Flare F class
├── results/               Replication results vs paper values (CSV)
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Datasets

| No. | Dataset | Samples | Features | Minority | Majority | IR |
|---|---|---|---|---|---|---|
| 1 | PSDAS | 4424 | 36 | 794 | 3630 | 4.57 |
| 2 | ESR | 11500 | 178 | 2300 | 9200 | 4.00 |
| 3 | DCCC | 30000 | 23 | 6636 | 23364 | 3.52 |
| 4 | BCW | 569 | 30 | 212 | 357 | 1.68 |
| 5 | ESDRP | 520 | 32 | 200 | 320 | 1.60 |
| 6 | CB | 208 | 60 | 97 | 111 | 1.14 |
| 7 | GLASS | 196 | 9 | 29 | 167 | 5.76 |
| 8 | ILPD | 583 | 11 | 167 | 416 | 2.49 |
| 9 | SEED | 210 | 7 | 70 | 140 | 2.00 |
| 10 | WINE | 178 | 13 | 59 | 119 | 2.02 |
| 11 | YEAST5 | 1484 | 8 | 44 | 1440 | 32.73 |
| 12 | YEAST5-ERL | 1484 | 8 | 5 | 1479 | 295.80 |
| 13 | CARGOOD | 1728 | 6 | 69 | 1659 | 24.04 |
| 14 | CARVGOOD | 1728 | 6 | 65 | 1663 | 25.58 |
| 15 | FLARE-F | 1066 | 11 | 43 | 1023 | 23.79 |

IR = Imbalance Ratio (majority / minority)

---

## Algorithms Replicated

Each dataset script replicates all algorithms from Table 2 of the paper:

- CSRBoost (the proposed method)
- AdaBoost
- SMOTE
- ADASYN
- Borderline-SMOTE
- SMOTE-Tomek
- SMOTE-ENN
- RUSBoost
- HUE (Hybrid Under-sampling Ensemble)
- GAN-based oversampling
- SMOTified-GAN

---

## CSRBoost Method Summary

CSRBoost works in 4 steps:

1. Apply KMeans clustering to the majority class. Number of clusters K equals n_minority (Equation 6, p = 100 percent).
2. From each cluster, randomly keep 50 percent of majority samples. This preserves the structure of the majority class instead of removing samples blindly.
3. Apply SMOTE to generate synthetic minority samples until classes are balanced.
4. Train one AdaBoost model with 50 weak learners on the balanced data.

---

## Evaluation Protocol

- 5-fold stratified cross-validation repeated 20 times (100 folds total)
- Metrics: Accuracy, AUC, F1-Score, Average Precision, G-Mean
- Matches the evaluation protocol described in the paper

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run any dataset script directly:

```bash
cd BCW
python bcw_final_replication.py
```

Each script is fully self-contained. The dataset file is in the same folder as the script.

---

## Replication Results Summary

Mean absolute error between our replication and the paper values across all 15 datasets:

| Metric | Mean Absolute Error |
|---|---|
| Accuracy | 0.0049 |
| AUC | 0.0098 |
| F1-Score | 0.0094 |
| Average Precision | 0.0124 |
| G-Mean | 0.0046 |

Full detailed results with per-dataset and per-algorithm comparisons are in `Graduation_project_final_report.pdf`.

---

## Reference

S. Yadav, S. Gupta, A. K. Yadav, and S. Gupta,
"CSRBoost: Clustered Sampling With Resampling Boosting for Imbalanced Dataset Pattern Classification,"
IEEE Access, 2025.
DOI: 10.1109/ACCESS.2025.3616207
