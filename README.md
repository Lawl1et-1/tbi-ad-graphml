# Do brains with a history of TBI look like Alzheimer's brains?

A reproducible, laptop-scale study on public post-mortem brain transcriptomes. A model learns the molecular signature
of Alzheimer's disease (AD vs control) in three cohorts, then scores brains from the **Allen Aging, Dementia and TBI
study** it never saw, to ask whether donors with a past traumatic brain injury (TBI) look more "AD-like", and which genes
drive that.

> **All results are computational associations in post-mortem tissue. Nothing here shows causation or supports any
> clinical use.**

**Project page with an interactive score explorer: https://lawl1et-1.github.io/tbi-ad-graphml/**
(static site in `docs/`, data exported by `src/phase7_export_site.py`).

## Main findings

1. **A robust AD signature.** 3,914 genes change consistently in AD across three training cohorts (meta-analysis),
   86% replicate in direction in a fourth cohort (GSE33000). Down in AD: neurons, synapses, oxidative phosphorylation.
   Up: vascular/stromal, TGF-β and innate-immune signaling. Much of this reflects cell composition.
2. **No single TBI gene.** In 106 Allen donors (4 regions), no gene is significantly associated with TBI (FDR < 0.05)
   in any region. Gene-level overlap tests with AD look impressive in hippocampus (p = 6×10⁻⁶) but **do not survive a
   matched-donor permutation test** (p = 0.11; all regions p = 0.24): genes move together, so naive gene-level p-values
   overstate the evidence.
3. **A modest, AD-specific shift at the sample level.** Scored as one AD-likeness number per sample, TBI donors score
   **+0.36 SD** higher than their matched non-TBI partners (logistic regression; exact sign-flip p = 0.024; mixed model
   adjusted for age, sex, dementia, RIN and region: +0.38 SD, 95% CI 0.12–0.63, p = 0.003), with a dose trend
   (p = 0.009). The shift is larger than for **99.95%** of random gene signatures, so it is specific to the AD direction.
   It is strongest in hippocampus. Caveats: it does not survive Bonferroni correction across 5 models (best 0.12), and it
   weakens in non-demented donors alone (+0.25 SD, p = 0.19).
4. **The graph neural network did not beat simple baselines.** Logistic regression (AUROC 0.946) beat the GAT on the
   STRING network (0.904). The GAT did as well on a degree-preserving *random* graph (0.906) and better with *no edges*
   (0.931): the protein-interaction topology did not add predictive value here. On held-out cohorts all models drop to
   ~0.70–0.84 AUROC.
5. **What drives it: no single gene or pathway.** The logistic-regression score is exactly decomposable; its gene
   weights are stable across models (ρ = 0.85, 102 stable genes, e.g. TAC1, BTK, LSP1, AQP9, GFAP, CARTPT). But
   no gene (0/4,000) and no pathway or cell-type set (0/925) carries the TBI shift at FDR < 0.05: the AD-like shift is
   diffuse. The GAT's explanations were **not** usable: Integrated Gradients disagreed across the 5 GAT models
   (top-200 Jaccard 0.13) and attention simply followed node degree (ρ = 0.96).
6. **Hypotheses (not findings):** hippocampal microglia/macrophage markers are higher in TBI brains (GSEA), in line with
   reports of chronic microglial activation after TBI; ribosome/proteasome genes are lower in TBI in most regions
   (possibly technical); immune/glial genes with large stable weights (BTK, LSP1, AQP9, GFAP) are candidates to check
   in independent cohorts.

## Data (all public)

| Role | Dataset | Tissue | Platform | Samples (donors) |
|---|---|---|---|---|
| Train | GSE132903 | middle temporal gyrus | Illumina HT-12 v4 | 195 (195) |
| Train | GSE118553 | temporal, entorhinal, frontal cortex | Illumina HT-12 v4 | 207 (78) |
| Train | GSE48350 | hippocampus, entorhinal, postcentral, superior frontal (age ≥ 60) | Affymetrix U133 Plus 2 | 173 (63) |
| External test | GSE33000 | dorsolateral prefrontal cortex | Rosetta/Agilent 44K | 467 (467) |
| TBI cohort | GSE104687 / Allen ADT study | temporal & parietal cortex, hippocampus, white matter | RNA-seq (raw RSEM counts) | 376 (106) |

GSE5281 was considered and **dropped** (laser-captured neurons, no donor IDs). GSE33000 is used only for testing because
its controls are ~20 years younger than its AD cases. Probe annotations from GEO; protein network: STRING v12
(combined score ≥ 700). Gene sets: Enrichr libraries (GO BP 2023, KEGG 2021, Reactome 2022, PanglaoDB 2021).

## Method

1. **Metadata and roles** (`src/phase1_*.py`): clean sample tables; donor IDs; confounder checks.
2. **Harmonization** (`src/phase2_*.py`): probes → genes; log2 scales; Allen counts → DESeq2-normalized log2;
   ComBat (own implementation, `src/combat.py`, tested in `tests/`) on Allen sequencing batch, then on
   dataset × region, **without labels**. 4,000 genes: STRING-connected and most variable in training data.
   After correction, dataset identity is predictable only at chance (0.19 vs 0.20).
3. **Classical biology** (`src/phase3_*.py`): AD differential expression (donor-clustered OLS + Stouffer meta),
   TBI DESeq2 per region, overlap tests with a **matched-pair permutation null**, GSEA / over-representation.
4. **Graphs** (`src/phase4_graph.py`): STRING and k-NN co-expression graphs on the 4,000 genes.
5. **Models** (`src/phase5_models.py`, `src/gat.py`): donor-level 5-fold CV × 3 seeds; LR L1/L2, RF, MLP; GAT with a
   per-gene readout (design chosen on validation data only); graph ablations; leave-one-cohort-out.
6. **Transfer and interpretation** (`src/phase6_*.py`): 5 final models per type (80% donor subsamples) → AD-likeness
   score; external test; Allen sanity checks (clinical AD, Braak, CERAD); TBI tests (mixed model, matched pairs,
   dose, interaction); robustness (sign-flip, random signatures, PC removal, leave-one-region-out); Integrated
   Gradients, attention, gene stability across models, enrichment.

Seeds are fixed everywhere (42 unless noted). Every number in this README comes from a file in `results/`.

## Results in figures (`results/figures/`)

| Figure | Shows |
|---|---|
| `pca_before_combat.png`, `pca_after_combat.png` | batch correction: datasets mix, AD/control axis appears |
| `fig_ad_volcano.png` | AD meta-analysis signature |
| `fig_tbi_vs_ad.png`, `fig_naive_vs_permutation.png` | gene-level TBI–AD overlap and why naive tests overstate it |
| `fig_model_performance.png` | GAT vs baselines, CV and leave-one-cohort-out |
| `fig_tbi_scores_lr_l2.png`, `fig_tbi_scores_gat.png` | AD-likeness scores, TBI vs no TBI, matched pairs |
| `fig_random_signature_null.png` | the TBI shift is specific to the AD direction |
| `fig_importance_network.png` | stable AD-score genes (logistic regression) on the STRING network |

Detailed reports: `results/phase1/PHASE1_REPORT.md` … `results/phase6/PHASE6_REPORT.md`.

## Limitations

- **Small TBI cohort**: 106 donors, 48 complete matched pairs; effects of ~0.3 SD are near the detection limit.
- **TBI definition**: self-reported TBI with loss of consciousness, often decades before death; no severity data.
- **Post-mortem bulk tissue**: expression changes mix per-cell regulation with cell-composition shifts (neuron loss,
  gliosis); no deconvolution here. No post-mortem interval available for adjustment.
- **Batch effects**: four platforms; two cohorts carry only relative values. ComBat was run without labels
  (conservative but can remove signal) and used all samples (transductive, no labels).
- **Generalization**: performance drops from ~0.95 (pooled CV) to ~0.75 on an unseen cohort.
- **Multiplicity**: several models and regions were tested; the main TBI effect does not survive Bonferroni over models.
- **Association only**: no causal or clinical claims.

## Reproduce

Windows / Linux, Python 3.12, CPU only. Downloads ~2 GB of raw data.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt      # exact versions: requirements-lock.txt
```

Download the raw data as described in `results/phase1/PHASE1_REPORT.md` (GEO series matrices, GPL annotations,
Allen portal files and per-sample RSEM files, STRING v12, Enrichr GMTs) into `data/raw/`. Then run in order:

```bash
.venv/Scripts/python src/phase1_metadata.py && .venv/Scripts/python src/phase1_explore.py
.venv/Scripts/python src/phase2_preprocess.py && .venv/Scripts/python src/phase2_harmonize.py
.venv/Scripts/python src/phase3_de_ad.py && .venv/Scripts/python src/phase3_de_tbi.py
.venv/Scripts/python src/phase3_overlap.py && .venv/Scripts/python src/phase3_enrich.py && .venv/Scripts/python src/phase3_figures.py
.venv/Scripts/python src/phase4_graph.py
.venv/Scripts/python src/phase5_models.py baselines && .venv/Scripts/python src/phase5_models.py lodo --models lr_l2 lr_l1 rf mlp
sh scripts/run_phase5_6.sh          # GAT CV + ablations + LODO + Phase 6 fit/score (~2 h on CPU)
.venv/Scripts/python src/phase6_robustness.py && .venv/Scripts/python src/phase6_explain.py   # IG ~1 h on CPU
.venv/Scripts/python src/phase6_explain_lr.py
.venv/Scripts/python src/phase7_figures.py
.venv/Scripts/python tests/test_combat.py
```

Explore the results interactively:

```bash
.venv/Scripts/streamlit run app/app.py
```

## Project layout

```
data/raw/         downloaded inputs (not in git)      src/        pipeline scripts, one per step
data/processed/   sample tables, harmonized data      tests/      ComBat tests
results/phaseN/   reports, tables, figures            app/        Streamlit explorer
results/figures/  slide-ready figures                 scripts/    long-running batch script
```
