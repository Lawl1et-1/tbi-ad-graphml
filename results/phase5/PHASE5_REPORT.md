# Phase 5 report: AD vs control classifiers

Date: 2026-11-03. Scripts: `src/phase5_models.py`, `src/gat.py`, `src/phase5_dev.py`, `scripts/run_phase5_6.sh`.
Tables: `results/phase5/cv_metrics.csv`, `cv_auroc_by_dataset.csv`, `lodo.csv`, `oof_<model>.csv`, `splits.csv`.

## Setup
- Data: harmonized expression, 4,000 genes, **575 training samples from 336 donors** (GSE132903, GSE118553, GSE48350).
- CV: `StratifiedGroupKFold(5)` on **donors** (no donor in two folds, checked by assertion), stratified by
  label × dataset, repeated with seeds 42/43/44. Gene z-scoring fit on the training fold only.
- Metrics: AUROC, AUPRC, balanced accuracy (threshold 0.5). Mean over seeds ± SD, and a 95% CI from a **donor-level
  bootstrap** (1,000 resamples) of the seed-averaged out-of-fold predictions.
- Baselines: logistic regression L2 / L1 (C chosen by inner 3-fold donor-grouped CV), random forest (500 trees),
  MLP (256-64, early stopping).
- GAT: one graph per sample on the shared STRING topology (18,864 edges + self-loops). Node input = expression +
  4-d learned gene embedding → linear → GATConv(8, 2 heads) → GATConv(8, 1 head) → **per-gene readout**
  (each gene's final state → scalar, weighted by a learned per-gene weight, summed). 16,953 parameters.
  AdamW lr 3e-3, batch 16, ≤ 40 epochs, early stopping on a 15% donor-held-out split of the training fold
  (minimum 10 epochs, patience 8). ~3 min per fold on a laptop CPU.

## GAT design history (kept for transparency)
- **v1 (mean + max pooling) failed:** in 3 of 5 folds early stopping picked epoch 1–2 and AUROC was 0.43–0.66.
  Pooling discards which gene a signal comes from. Results archived in `results/phase5/v1_pooling/`, not used.
- **v2 (per-gene readout)** was chosen in `src/phase5_dev.py` using **only an inner validation split of fold 0's
  training data** (validation AUROC 0.90 vs 0.52 for pooling with the same training fixes). Test folds were never
  used to choose the design.

## Results: cross-validation (pooled over the 3 training cohorts)

| Model | Seeds | AUROC | 95% CI | AUPRC | Balanced acc. |
|---|---|---|---|---|---|
| **Logistic regression L2** | 3 | **0.946 ± 0.003** | 0.926–0.968 | 0.943 | 0.873 |
| GAT, no edges (self-loops only) | 1 | 0.931 | 0.905–0.954 | 0.930 | 0.868 |
| Logistic regression L1 | 3 | 0.929 ± 0.006 | 0.919–0.962 | 0.925 | 0.859 |
| GAT, degree-preserving random graph | 1 | 0.906 | 0.873–0.933 | 0.918 | 0.826 |
| **GAT, STRING graph (main)** | 2 | 0.904 ± 0.006 | 0.885–0.943 | 0.914 | 0.822 |
| MLP | 3 | 0.900 ± 0.009 | 0.878–0.940 | 0.908 | 0.820 |
| GAT, co-expression graph | 1 | 0.875 | 0.839–0.909 | 0.888 | 0.793 |
| Random forest | 3 | 0.848 ± 0.001 | 0.812–0.888 | 0.865 | 0.773 |

**Honest reading:**
1. **A linear model beats the GAT.** Logistic regression L2 is the best model (0.946 vs 0.904; CIs overlap only slightly).
2. **The STRING topology does not help.** The GAT on a degree-preserving random graph does as well (0.906), and
   removing all edges does better (0.931). Message passing over PPI edges mixes each gene's signal with its neighbours'
   and blurs gene-specific information that a linear model uses directly.
3. The co-expression graph is worse still (0.875), even though it was built from training data (slightly optimistic).
4. With ~575 samples and a dominant, mostly linear AD axis (Phase 2–3), there is little room for graph structure to add value.

## Results: per-cohort AUROC inside CV

| Model | GSE118553 | GSE132903 | GSE48350 |
|---|---|---|---|
| LR L2 | 0.989 | 0.981 | 0.931 |
| LR L1 | 0.968 | 0.955 | 0.952 |
| GAT STRING | 0.978 | 0.948 | 0.787 |
| GAT no edges | 0.976 | 0.961 | 0.855 |
| MLP | 0.968 | 0.952 | 0.753 |
| RF | 0.940 | 0.892 | 0.683 |

Per-cohort AUROCs are higher than the pooled ones because scores are offset between cohorts (calibration).
GSE48350 (relative values, inferred donor IDs) is the hardest cohort.

## Dataset-identity check: leave-one-dataset-out (train on 2 cohorts, test on the 3rd)

| Held-out cohort | LR L2 | LR L1 | RF | MLP | GAT STRING |
|---|---|---|---|---|---|
| GSE118553 | 0.70 | 0.67 | **0.78** | 0.75 | 0.75 |
| GSE132903 | **0.84** | 0.77 | 0.79 | 0.82 | **0.84** |
| GSE48350 | **0.72** | 0.70 | 0.64 | 0.66 | 0.70 |

Performance drops from ~0.95 to **0.64–0.84** on an unseen cohort. So part of what the models learn is
cohort-specific (platform, brain region, cohort-specific pathology). These cross-cohort numbers, not the pooled CV,
are the realistic expectation for transfer to the Allen cohort. On this test the GAT is on par with LR L2 (mean 0.76 vs
0.75), so it generalizes no worse, just no better.

Other identity checks: after ComBat, dataset is predictable only at chance (Phase 2); external test on GSE33000
(Phase 6): AUROC 0.89–0.93 with age/sex adjustment, so the signature is not just the training cohorts' identity.

## Limitations
- GAT hyperparameters were fixed (one design selection on validation data), not tuned per fold; the baselines' C was
  tuned in inner CV. A full GAT hyperparameter search could close part of the gap, but not the random-graph result.
- Ablations ran with 1 seed (CPU time); the STRING GAT with 2 seeds. Seed SD for the GAT was 0.006.
- ComBat and variance-based gene selection used all samples without labels (transductive, unsupervised). This cannot
  leak labels, but test samples did influence the batch-correction parameters.
- The co-expression graph used all training samples, so its CV is slightly optimistic.
