# Phase 6 report: transfer to the TBI cohort and interpretation

Date: 2026-11-03. Scripts: `src/phase6_transfer.py`, `src/phase6_robustness.py`, `src/phase6_explain.py`,
`src/phase6_explain_lr.py`. Tables in `results/phase6/`. Figures copied to `results/figures/`.

## Final models and score
For each model type, 5 models, each trained on a random 80% of the 336 training donors (seeds 101–105); the other 20%
was used for GAT early stopping and as a quick check (held-out AUROC: LR L2 0.93–0.98, GAT 0.88–0.97).
**AD-likeness score** = mean predicted log-odds over the 5 models, standardized across Allen samples (SD units).
Neither GSE33000 nor Allen was used for fitting or z-scoring.

## 1. External test (GSE33000, prefrontal cortex, never trained on)

| Model | AUROC (all) | AUROC, age ≥ 65 (292 AD / 68 ctrl) | OR per SD, adjusted for age + sex |
|---|---|---|---|
| LR L2 | 0.893 | 0.882 | 15.6 (7.6–32.0) |
| LR L1 | 0.922 | 0.901 | 16.9 |
| RF | 0.930 | 0.900 | 10.3 |
| MLP | 0.929 | 0.912 | 15.9 |
| GAT | 0.926 | 0.915 | 12.5 (6.9–22.4) |

The signature transfers to a new platform and brain region, and survives age/sex adjustment.

## 2. Allen sanity check (before looking at TBI)

The score should be higher in Allen donors with clinical AD and track neuropathology:

| Model | AUROC clinical AD vs control (all regions) | ρ Braak | ρ CERAD | ρ NIA-Reagan |
|---|---|---|---|---|
| LR L2 | 0.71 | 0.22 | 0.30 | 0.25 |
| GAT | 0.71 | 0.18 | 0.26 | 0.21 |
| MLP | 0.71 | 0.18 | 0.24 | 0.18 |
| LR L1 | 0.71 | 0.17 | 0.26 | 0.21 |
| RF | 0.66 | 0.16 | 0.24 | 0.18 |

Weaker than in the training cohorts (as expected from the leave-one-cohort-out results, and because Allen's AD label
is clinical in a very old cohort with mixed pathologies), but clearly present.

## 3. Main question: do TBI donors score higher?

Mixed model `score ~ TBI + age + sex + dementia + RIN + region + (1 | donor)` on all 376 samples, plus matched pairs
(48 complete TBI / non-TBI pairs from the study design, donor-mean score).

| Model | TBI effect (SD) [95% CI] | p | Pair diff (SD) [95% CI] | **Exact sign-flip p** | Dose (per TBI) p | Random-signature percentile |
|---|---|---|---|---|---|---|
| **LR L2** | **+0.38 [0.12, 0.63]** | 0.003 | +0.36 [0.07, 0.65] | **0.024** | 0.009 | 99.95 |
| LR L1 | +0.35 [0.09, 0.61] | 0.007 | +0.34 [0.06, 0.63] | 0.025 | 0.023 | 99.9 |
| MLP | +0.37 [0.10, 0.64] | 0.006 | +0.36 [0.06, 0.66] | 0.027 | 0.008 | 99.95 |
| **GAT** | +0.30 [0.04, 0.57] | 0.026 | +0.31 [0.01, 0.62] | 0.054 | 0.045 | 99.5 |
| RF | +0.14 [−0.11, 0.40] | 0.26 | +0.15 [−0.14, 0.45] | 0.32 | 0.17 | 87.6 |

For scale: dementia on the same score is about +0.47 SD (LR L2, p < 0.001). Per-region unadjusted Cohen's d (LR L2):
temporal 0.35, parietal 0.40, hippocampus 0.49, white matter 0.48.

**Robustness (LR L2 unless noted):**
- **AD-specific, not generic:** the TBI shift exceeds that of 99.95% of 2,000 random gene signatures drawn with the
  training data's covariance structure (p = 0.001; 99.2% for independent random weights). So TBI brains are not just
  "different"; they are shifted along the AD direction in particular.
- **Not a broad expression axis:** removing the top 1, 3 or 5 principal components of the Allen data leaves the effect
  unchanged (+0.35 to +0.38 SD, p ≈ 0.02); removing 10 PCs weakens it (+0.26, p = 0.07).
- **No single region drives it:** leaving out any one region keeps it (β 0.35–0.38, p ≤ 0.012). Strongest in
  hippocampus (β 0.45, p = 0.013), consistent with the Phase 3 trend.
- **Adjusting or not for dementia barely changes it** (+0.41 unadjusted vs +0.38 adjusted), and the TBI × dementia
  interaction is not significant (p = 0.51).
- **But:** in non-demented donors alone (56 donors) the effect is smaller and not significant (LR +0.25 SD, p = 0.19;
  GAT +0.10, p = 0.60). The data cannot tell whether the shift is equally present without dementia (low power) or partly
  carried by demented TBI donors.
- **Multiplicity:** with Bonferroni over the 5 models, the best sign-flip p becomes 0.12. If the GAT is taken as the
  pre-specified primary model (as in the project plan), its exact p is 0.054.

**Reconciling with Phase 3** (gene-level correlation, global permutation p = 0.24): Phase 3 asked whether *all* 11,000
genes' TBI effects line up with AD, which mixes the AD axis with many unrelated, correlated modules. The score is a
single targeted projection learned on independent cohorts, so it has more power for a small, coherent shift.
Both say there is no large effect; the score detects a modest one.

**Verdict:** modest, AD-direction-specific evidence that brains with a history of TBI with loss of consciousness look
slightly more AD-like (~0.3–0.4 SD, about three-quarters of the dementia effect on the same score), with a dose trend.
Nominally significant for the linear and MLP models, borderline for the GAT, not significant after correction across
models. **Association only.**

## 4. Explanations

### GAT (Integrated Gradients, attention) — not reliable
- IG importance **disagrees between the 5 GAT models**: mean top-200 Jaccard 0.13, mean Spearman 0.33; only 23 genes
  are in the top 200 of ≥ 4 models.
- **Attention tracks node degree** (ρ = 0.96 between a gene's incoming attention and its STRING degree) and is unrelated to IG
  (ρ = −0.03). Attention weights here are not an explanation.
- GAT importance is unrelated to AD differential expression (ρ = 0.00) and weakly related to LR coefficients (ρ = 0.20).
- Exploratory TBI-vs-pair attribution difference: no gene at FDR < 0.05.

### Logistic regression L2 — stable, used for interpretation
- Coefficients agree across the 5 models (mean Spearman **0.85**, top-200 Jaccard 0.38); **102 stable genes**
  (`lr_stable_genes.csv`); 75% push in the direction of their AD differential expression sign
  (multivariate coefficients can flip sign for correlated genes).
- Top stable genes: TAC1, BTK, TSPAN1, PIF1, LSP1, AQP9, CACNA1H, RPH3A, GABRG3, TEAD2, CARTPT, GFAP, ANGPTL1,
  ALDH1A1, NRP1, NQO1, TFRC.
  Includes glial/immune genes (GFAP 99.5th percentile, BTK, LSP1, AQP9, CX3CR1 90th, SPP1 92nd, C1QC 95th) and
  neuronal/neuropeptide genes (TAC1, CARTPT, VGF 83rd, NRN1 86th, GABRG3).
- Enrichment of stable genes (background 4,000): little survives FDR (translation initiation / TLR-MyD88 members among
  genes pushing away from AD; BH3-only apoptosis among genes pushing toward AD); the gene set is small.

### What drives the higher TBI score? (LR contributions, matched pairs)
Score = Σ coefficient × expression, so each gene's and gene set's share of the TBI–pair difference is exact.
- **No single gene** (0 of 4,000) and **no gene set** (0 of 925 KEGG / Reactome / brain cell-type sets) reaches FDR < 0.05.
- The largest nominal contributors (p ≈ 0.001–0.01, FDR ≈ 0.73): Reactome developmental biology, PI metabolism,
  non-integrin ECM interactions, G-protein signaling, FOXO-mediated transcription, carbohydrate metabolism.
- Cell-type marker sets each carry only 1–7% of the shift (oligodendrocyte / OPC / interneuron / pyramidal slightly
  positive, microglia slightly negative), none significant.
- **Conclusion:** the AD-like shift in TBI brains is **diffuse**, spread thinly over many genes, not a single pathway
  or cell type that can be pinned down with 48 pairs.

## 5. Comparison with Phase 3 and published findings

| Expected from literature | What we see |
|---|---|
| AD bulk signature = neuronal / synaptic / mitochondrial loss, glial and immune gain | **Matches** (Phase 3 enrichment, strongly) |
| TBI → chronic microglial / myeloid inflammation | **Partly**: hippocampal microglia/macrophage GSEA up in TBI (Phase 3, gene-permutation FDR < 0.001), but microglial genes do *not* carry the TBI shift in the AD score |
| TBI → modestly increased AD-type pathology / dementia risk | **Consistent in direction**: small AD-like expression shift with dose trend; no excess dementia in this matched cohort (Phase 1) |
| Original Allen study: weak expression effects of TBI | **Matches**: 0 FDR genes per region |

**New observations, labelled as hypotheses:**
1. The AD-like shift in TBI is strongest in **hippocampus and white matter**.
2. Ribosome / proteasome / spliceosome genes are lower in TBI in most regions (Phase 3 GSEA); possibly technical.
3. Candidate genes with large stable weights and immune/glial roles (BTK, LSP1, AQP9, GFAP) could be checked
   in independent TBI cohorts, e.g. with cell-type deconvolution or single-nucleus data.

## Limitations
- 48 complete matched pairs; effects of 0.3 SD are near the detection limit and confidence intervals are wide.
- TBI = self-reported, with loss of consciousness, often decades earlier; no severity or timing analysis beyond count.
- The AD score partly reflects cell composition (neuronal loss / gliosis); a TBI shift on it may also be compositional.
- Several models, regions and analyses were examined; only the matched-pair and mixed-model tests were planned up front.
  The random-signature control, PC removal and leave-one-region-out were added after the main result (post-hoc robustness).
- Integrated Gradients used a zero (= mean training sample) baseline and 16 steps; GAT explanations were unstable anyway.
- Association only, post-mortem tissue, no causal claims.
