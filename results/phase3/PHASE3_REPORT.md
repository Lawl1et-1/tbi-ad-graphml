# Phase 3 report: classical analysis (baseline biology)

Date: 2026-11-03. Seed: 42. Scripts: `src/phase3_de_ad.py`, `src/phase3_de_tbi.py`, `src/phase3_overlap.py`,
`src/phase3_enrich.py`, `src/phase3_figures.py`. All tables in `results/phase3/`.

## 1. AD vs control (training cohorts, replication in GSE33000)

Per dataset: gene-wise linear model `expr ~ AD + age + sex (+ region)` on the dataset's own log2 values,
donor-clustered robust SEs. Meta-analysis: weighted Stouffer (weights √donors). 11,027 genes tested.

| Dataset | Samples (donors) | AD / control | Genes FDR < 0.05 |
|---|---|---|---|
| GSE132903 | 195 (195) | 97 / 98 | 7,247 |
| GSE118553 | 207 (78) | 129 / 78 | 3,916 |
| GSE48350 | 173 (63) | 80 / 93 | 541 |
| **Meta (3 cohorts)** | 575 (336) | | 6,625; **3,914 with the same sign in all 3** (1,736 up, 2,178 down) |
| GSE33000 (replication) | 467 (467) | 310 / 157 | 8,100 |

- **Replication:** 86% of the 3,914 signature genes change in the same direction in GSE33000; gene-wide Spearman
  between meta z and GSE33000 z = 0.64. (GSE33000 is age-confounded, so its own gene counts are inflated.)
- Between training cohorts, gene-wide z correlation is 0.27–0.36: the signature is shared but noisy per cohort.
- Top genes: NRN1, NAP1L5, TUBB2A, HSPB3 (down); HBP1, TEAD2, SLC15A3, PLXNB1, CHST6 (up).

**Enrichment (over-representation, background = 11,027 genes):**
- Down in AD: **neuron and interneuron markers** (PanglaoDB, p_adj ~ 10⁻³⁹), Neuronal System, chemical synaptic
  transmission, **oxidative phosphorylation** / electron transport, KEGG Parkinson/Huntington/prion pathways.
- Up in AD: vascular smooth muscle / mesangial / stellate (stromal-vascular) markers, **TGF-β signaling**,
  **NOD-like receptor** (innate immune) signaling, transcription regulation.
This is the textbook bulk-tissue AD picture (neuronal and mitochondrial loss, vascular/glial/immune increase), and
much of it reflects **cell composition**.

Figure: `fig_ad_volcano.png`. Tables: `de_ad_<dataset>.csv`, `de_ad_meta.csv`, `enrich_ad_ora.csv`.

## 2. TBI vs no TBI (Allen, DESeq2 on raw counts, one model per region)

Main model `~ age + RIN + sex + batch + dementia + TBI`; sensitivity model without dementia.

| Region | Samples | Genes tested | TBI FDR < 0.05 | TBI p < 0.01 (expected by chance) | Dementia FDR < 0.05 |
|---|---|---|---|---|---|
| Temporal cortex | 98 | 22,670 | 0 | 367 (227) | 0 (14 at FDR 0.1) |
| Parietal cortex | 91 | 22,787 | 0 | 94 (228) | 0 |
| Hippocampus | 94 | 22,361 | 0 | 193 (224) | 2 |
| White matter | 93 | 22,562 | 0 | 196 (226) | 2 |

**No single gene is significantly associated with TBI in any region**, with or without dementia adjustment.
This agrees with the original Allen study (Miller et al., 2017), which also found weak TBI expression effects.
Only temporal cortex shows an excess of nominal hits. Even dementia, a much stronger phenotype, gives almost no
FDR-significant genes at n ≈ 95 per region: this cohort is underpowered for single-gene tests.

## 3. Does the TBI signature resemble the AD signature?

| Region | Overlap with AD signature (hypergeometric) | Same direction | Spearman ρ(AD z, TBI stat) | **Matched-pair permutation p** |
|---|---|---|---|---|
| Temporal cortex | 82 genes, 1.03×, p = 0.39 | 60% | 0.05 | 0.87 |
| Parietal cortex | 13, 0.68×, p = 0.97 | 23% | −0.17 | 0.34 |
| Hippocampus | 64, **1.58×, p = 6×10⁻⁶** | **94%** | **0.39** | **0.11** |
| White matter | 17, 0.63×, p = 0.996 | 82% | 0.44 | 0.10 |
| **All regions (mean ρ, post-hoc global test)** | | | 0.17 | **0.24** |

Without dementia adjustment, results barely change (hippocampus p = 0.08, global p = 0.21).

**Key methodological point.** The naive tests (hypergeometric, gene-level Spearman, gene-permutation GSEA) say
the hippocampal TBI signature is strongly AD-like. But when TBI labels are swapped at random inside the matched
donor pairs (1,000 times), correlations of ±0.5 with the AD axis appear routinely (`fig_tbi_vs_ad.png`, bottom row).
The reason: genes are strongly correlated (neuronal module, glial module), so any split of donors that differs a bit
in cell composition lines up with the AD axis. **Correct inference must permute donors, not genes.** Figure:
`fig_naive_vs_permutation.png`.

**Conclusion of Phase 3:** at the gene-expression level, there is **no statistically reliable evidence** that TBI
brains carry the AD signature in this cohort. There is a non-significant trend in the AD direction in hippocampus and
white matter, and none in cortex.

## 4. TBI pathway / cell-type enrichment (preranked GSEA on DESeq2 statistic; gene-permutation, descriptive)

| Gene set | TCx | PCx | HIP | FWM |
|---|---|---|---|---|
| KEGG Ribosome | −3.56* | −3.03* | −3.25* | −1.01 |
| KEGG Proteasome | −2.70* | −1.95* | −2.60* | −2.34* |
| KEGG Spliceosome | −2.68* | −2.56* | −2.17* | −0.84 |
| KEGG Oxidative phosphorylation | −1.88* | −0.81 | −2.88* | −2.57* |
| PanglaoDB Microglia | 0.92 | −1.23 | **2.41*** | 1.46 |
| PanglaoDB Macrophages | 0.80 | −1.21 | **2.08*** | 1.35 |
| PanglaoDB Neurons | 1.54* | 2.79* | −1.61* | −2.80* |

NES values; * = FDR < 0.05 (gene-permutation, optimistic). Negative = lower in TBI.

- **Hippocampal microglia/macrophage increase in TBI** matches published reports of chronic microglial activation
  and myeloid inflammation years after a single TBI. This is the most biologically coherent TBI signal.
- **Lower ribosome / proteasome / spliceosome expression in TBI** is consistent across regions. Ribosomal and
  housekeeping modules are sensitive to RNA quality and post-mortem factors, so even with RIN in the model, this
  could be technical. **Hypothesis only.**
- Neuronal markers go in opposite directions in cortex (up) and hippocampus/white matter (down), which again points
  to composition differences between groups rather than a uniform TBI program.

Tables: `gsea_tbi_by_region.csv`, `gsea_tbi_vs_ad_sets.csv`, `overlap_summary.csv`, `overlap_global_test.json`.

## Limitations added in Phase 3
- Per-region DESeq2 ignores that regions share donors; a mixed model on counts (e.g. glmmTMB / dream in R) would be better.
- The permutation null uses OLS on log-CPM (DESeq2 is too slow to rerun 1,000×); observed statistics were computed both ways and agree (Spearman ρ within 0.01).
- GSEA p-values permute genes, not donors, so they are optimistic; I report them as descriptive.
- The global test across regions was added after seeing per-region results (post-hoc).
- Bulk tissue: shifts in cell composition and per-cell expression changes cannot be separated without deconvolution.
