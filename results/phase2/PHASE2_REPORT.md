# Phase 2 report: preprocessing and harmonization

Date: 2026-11-03. Seed: 42. Scripts: `src/phase1_metadata.py` (roles), `src/phase2_preprocess.py`,
`src/phase2_harmonize.py`, `src/combat.py`, test `tests/test_combat.py`.
Outputs: `data/processed/harmonized_expr.parquet` (1,418 samples × 4,000 genes, float32),
`harmonized_samples.csv`, `selected_genes.txt`, `allen_counts.parquet` (raw counts for Phase 3).

## Decisions taken at the start of Phase 2 (user said "choose the best option")
- GSE5281 dropped (laser-captured neurons, no donor IDs). **GSE118553 added**.
- GSE33000 = external test only (never trained on), because of age/sex confounding.

## Sample roles

| Role | Dataset | Samples | Donors | AD / control (samples) |
|---|---|---|---|---|
| train | GSE132903 (MTG) | 195 | 195 | 97 / 98 |
| train | GSE118553 (temporal, entorhinal, frontal) | 207 | 78 | 129 / 78 |
| train | GSE48350 (HIP, EC, PCG, SFG; age ≥ 60) | 173 | 63 | 80 / 93 |
| test_ext | GSE33000 (DLPFC) | 467 | 467 | 310 / 157 |
| transfer | GSE104687 Allen (TCx, PCx, HIP, FWM) | 376 | 106 | 152 / 196 (+28 other dementia) |

Training total: **575 samples, 336 donors**. Excluded: GSE118553 cerebellum and "asymptomatic AD",
GSE48350 controls < 60 y, GSE33000 Huntington's.

## Step 1: IDs → gene symbols, log2 scale

| Dataset | Input values | Transformation | One probe per gene chosen by | Genes |
|---|---|---|---|---|
| GSE132903, GSE118553 | log2 (Illumina) | none | highest mean | 19,236 / 20,750 |
| GSE48350 | **already divided by each gene's median** | log2 | highest mean in GSE5281 (same chip, absolute values) | 20,848 |
| GSE33000 | log10 ratio vs pooled reference | × log2(10) → log2 ratio | highest variance | 19,410 |
| Allen | RSEM expected counts | DESeq2 median-of-ratios, log2(x+1); CPM ≥ 1 in ≥ 25% of samples | Entrez → symbol | 22,724 expressed of 50,281 |

Probes that map to several genes were dropped. **11,027 genes** are present in all five datasets and expressed in Allen brains.

## Step 2: Allen sequencing batches
ComBat on the 11 sequencing batches, run separately inside each region (smallest batch-region cell: 5 samples).
Safe to do: TBI status is balanced across batches (p = 0.27, Phase 1).

## Step 3: gene set (4,000 genes)
1. Of the 11,027 shared genes, **9,198** have at least one STRING v12 edge with score ≥ 700 to another shared gene.
2. Ranked by variance inside each training dataset (average rank across the 3), top 4,000 kept.
   **No labels were used**, so selection cannot leak diagnosis information.
3. Marker check: kept APOE, TREM2, GFAP, C1QA, C3, AIF1, CD163, SST, VGF, BDNF, SNAP25, SYT1, SPP1, HLA-DRA, SERPINA3, CHI3L1.
   Not kept: APP, MAPT, CLU (in the shared genes but low variance), AQP4 and CD68 (missing from at least one platform).
   I did **not** force markers in: it would bias the Phase 6 explanations toward known genes.

## Step 4: cross-dataset ComBat
- Batch = dataset × region group (13 batches), so neither dataset nor region identity remains as a simple shift.
- **No diagnosis covariate.** Protecting a label in ComBat inflates group differences when AD/control ratios
  differ between batches (Nygaard et al., *Biostatistics* 2016), and it would require the AD labels of the held-out
  Allen and GSE33000 samples. Without it, the correction is conservative: some real AD signal may be removed in
  batches that are AD-heavy (GSE118553: 62% AD).
- **Bug found and fixed:** the `combat` PyPI package (0.3.3) returned data whose gene variances were shrunk ~50×
  and unequal across batches (GSE118553 kept ~10× more variance than the others, producing a spray of outliers in PCA).
  Replaced by our own implementation of Johnson et al. 2007 (`src/combat.py`), tested on simulated data with known
  batch shifts/scales and known true signal (`tests/test_combat.py`, passes).

## Results

| Metric | Before | After |
|---|---|---|
| Variance explained by PC1 | 94% (= dataset) | 27% |
| Dataset predictable from top 20 PCs (balanced accuracy, 5-fold; chance = 0.20) | 0.998 | **0.188** |
| Silhouette by dataset | 0.456 | −0.050 |
| Silhouette by AD vs control (training samples) | 0.016 | 0.050 |

Figures: `pca_before_combat.png`, `pca_after_combat.png` (colored by dataset, diagnosis, region).

**Unsupervised sanity check.** PCA never sees labels, yet after correction PC1 separates AD from control within
every dataset (`pc1_ad_auroc_by_dataset.csv`):

| Dataset | Role | AUROC of PC1 for AD | PC1 vs age in controls (Spearman ρ) |
|---|---|---|---|
| GSE132903 | train | 0.78 | −0.09 (p = 0.40) |
| GSE118553 | train | 0.75 | n/a |
| GSE48350 | train | 0.59 | −0.13 (p = 0.21) |
| GSE33000 | external test | 0.90 | **−0.50 (p < 0.001)** |
| Allen | transfer | 0.65 (clinical AD) | 0.09 (p = 0.23) |

The GSE33000 value is inflated by age: its controls are ~20 years younger, and PC1 tracks age in those controls.
This supports keeping it as a test set with age-adjusted reporting.

## Limitations added in Phase 2
- GSE48350 and GSE33000 carry only relative values (ratio to gene median / to a pool), so absolute expression
  levels are lost for them; harmonization relies on per-gene location/scale alignment.
- The main AD axis in bulk tissue likely reflects cell composition (neuron loss, glial increase) as much as
  per-cell gene regulation. Phase 6 explanations must be read with this in mind (cell-type deconvolution would help).
- Probe-to-gene collapsing (one probe per gene) can pick different transcripts on different platforms.
- ComBat without a diagnosis covariate may slightly weaken the AD signal in AD-heavy batches.
- Allen white matter (FWM) has no training counterpart; its transferred scores should be read with caution.
