# Project brief: Shared molecular signatures between TBI and Alzheimer's Disease (graph ML)

> How to resume: open a new session in `D:\rayanportfolio\tbi-ad-graphml` and say
> "Read PROJECT_BRIEF.md and continue from the current status."

## Status

- [x] Phase 0: Setup — done 2026-11-03 (venv `.venv`, Python 3.12; inmoose would not build, PyPI `combat` was buggy → own ComBat in `src/combat.py`)
- [x] Phase 1: Data acquisition and exploration — done 2026-11-03, see `results/phase1/PHASE1_REPORT.md`
- [x] Phase 2: Preprocessing and harmonization — done 2026-11-03, see `results/phase2/PHASE2_REPORT.md`
- [x] Phase 3: Classical analysis — done 2026-11-03, see `results/phase3/PHASE3_REPORT.md`
- [x] Phase 4: Gene graph construction — done 2026-11-03, see `results/phase4/PHASE4_REPORT.md`
- [x] Phase 5: Models — done 2026-11-03, see `results/phase5/PHASE5_REPORT.md`
- [x] Phase 6: Transfer to TBI and interpretation — done 2026-11-03, see `results/phase6/PHASE6_REPORT.md`
- [x] Phase 7: Deliverables — done 2026-11-03: `README.md`, `results/figures/`, Streamlit app `app/app.py` (tested in browser)

**ALL PHASES DONE.** Live site: https://lawl1et-1.github.io/tbi-ad-graphml/ (GitHub Pages from main /docs; rebuild data with `src/phase7_export_site.py`; bump `?v=` in docs/index.html after changing app.js/style.css). Repo: https://github.com/Lawl1et-1/tbi-ad-graphml (public). Possible next steps (not started): git repo + push; slide deck from `results/figures/`;
cell-type deconvolution of Allen data; per-fold co-expression graph; GAT hyperparameter search.

**Working mode (user, 2026-11-03): "keep working without asking me".** Go through phases without waiting for a
go-ahead; still write each phase's report and update this status after each phase.

Decisions taken (user: "choose the best option"):
- GSE5281 dropped; GSE118553 added (temporal, entorhinal, frontal; no cerebellum, no AsymAD).
- GSE33000 = external test only. Training = GSE132903 + GSE118553 + GSE48350 (age ≥ 60): 575 samples, 336 donors.
- Sample roles (train / test_ext / transfer / excluded) are in the `role` column of `data/processed/samples_all.csv`.
- ComBat batch = dataset × region group, no diagnosis covariate (leak-free, conservative).

**Final results (2026-11-03):** GAT STRING CV AUROC 0.904 < LR-L2 0.946; GAT on random graph 0.906, no edges 0.931
(topology adds nothing). External GSE33000 AUROC 0.89–0.93. TBI donors score +0.38 SD higher on the AD score (LR-L2 mixed
model p = 0.003; matched-pair sign-flip p = 0.024; GAT p = 0.054; not significant after Bonferroni over 5 models);
AD-specific vs random signatures (99.95th percentile); weaker in non-demented donors only. GAT explanations unstable;
LR explanations stable but no gene/pathway carries the TBI shift at FDR < 0.05 (diffuse).

**History (2026-11-03):** Phases 2–4 done. Phase 2: 4,000 genes, dataset predictability after ComBat 0.19
(chance 0.20). Phase 3: AD signature 3,914 genes (replicates in GSE33000, 86% same sign); TBI has 0 FDR genes in all
regions; AD–TBI correlation not significant under matched-pair permutation (global p = 0.24; HIP p = 0.11, FWM p = 0.10).
Phase 4: STRING graph 18,864 edges, 359 isolated genes.
Phase 5 so far: CV AUROC LR-L2 0.946, LR-L1 0.929, MLP 0.900, RF 0.848; leave-one-dataset-out drops to 0.64–0.84.
GAT v1 (mean+max pooling) failed (unstable, archived in `results/phase5/v1_pooling/`); v2 uses a per-gene readout,
chosen on validation data only (`src/phase5_dev.py`: 0.90 vs 0.52). `scripts/run_phase5_6.sh` runs GAT CV (string, random,
none, coexpr), GAT LODO, evaluation, then Phase 6 fit + score. Every runner skips finished work, so just rerun it.
After it: `src/phase6_explain.py` (IG, attention, stability), then write PHASE5/PHASE6 reports, then Phase 7.
No git repo yet (not initialized).

Allen raw counts: `data/raw/allen/rsem/<rnaseq_profile_id>.tsv` (377 files, all verified complete; drop 496100315, not in GEO).

Run order: `.venv/Scripts/python src/phase1_metadata.py`, `src/phase1_explore.py`, `src/phase2_preprocess.py`,
`src/phase2_harmonize.py`, `src/phase3_de_ad.py`, `src/phase3_de_tbi.py` (~5 min), `src/phase3_overlap.py`,
`src/phase3_enrich.py` (~30 min), `src/phase3_figures.py`, `src/phase4_graph.py`, `src/phase5_models.py baselines`,
`scripts/run_phase5_6.sh` (~2 h), `src/phase6_robustness.py`, `src/phase6_explain.py` (~1 h), `src/phase6_explain_lr.py`,
`src/phase7_figures.py`. App: `.venv/Scripts/streamlit run app/app.py`. Tests: `.venv/Scripts/python tests/test_combat.py`.

First instruction: **start with Phase 0 and Phase 1 only, then stop and report.**

(Update this section at the end of each phase with what was done and key numbers.)

---

## Context

I am a final-year CS/AI student. I want a research-grade portfolio project, presentable to a jury, built only on public data and runnable on a laptop CPU or free Colab. I have experience with PyTorch Geometric and GAT models. I have no wet lab, so every claim must stay at the level of computational evidence (association, never causation).

## Research question

Do human brains with a history of TBI show gene expression patterns that resemble Alzheimer's disease, and which genes and pathways drive that resemblance?

## Core idea

1. Train a model to recognize the AD molecular signature (AD vs control) from brain gene expression, where each sample is a gene graph (nodes = genes, edges = known interactions, node features = that sample's expression).
2. Apply the trained model to TBI and non-TBI samples it never saw, to produce an "AD-likeness score" per sample.
3. Test whether TBI samples score higher than matched controls.
4. Use model explanations to identify which genes and pathways drive the score, and compare them to published findings.

## Working rules

- Work in phases. At the end of each phase, stop, summarize what you did, show key numbers and plots, and wait for my go-ahead before the next phase.
- Before using any dataset ID, verify it exists and check its metadata (tissue, platform, sample counts, available labels). Do not assume. If a dataset listed below is wrong or unsuitable, say so and propose alternatives.
- Fix random seeds everywhere. Log every result to a `results/` folder.
- Split data by donor, never by sample, to avoid leakage when a donor has several brain regions.
- Always compare the GAT to simple baselines. If a baseline matches or beats the GAT, report it honestly.
- Never overstate results. Write limitations as you go.

## Phase 0: Setup

- Python project with `uv` or `venv`, `requirements.txt`, clear folder structure (`data/raw`, `data/processed`, `src/`, `notebooks/`, `results/`, `app/`).
- Libraries to consider: GEOparse, pandas, scanpy or anndata if useful, pydeseq2, inmoose (pycombat) for batch correction, gseapy, networkx, torch, torch_geometric, captum, scikit-learn, matplotlib/seaborn, streamlit.

## Phase 1: Data acquisition and exploration

Candidate datasets (verify each one):

- Allen Institute "Aging, Dementia and TBI" study (aging.brain-map.org, also on GEO, likely GSE104687): RNA-seq from several brain regions of older donors with and without TBI history and dementia diagnosis. Most valuable one, because TBI and dementia labels come from the same cohort.
- AD vs control brain cohorts for training the AD signature, for example GSE33000 (prefrontal cortex, large), GSE5281 (multiple regions), GSE132903 or GSE48350. Pick based on region overlap with the Allen data.

Tasks: download, inspect metadata, build a clean sample table (donor ID, region, age, sex, PMI if available, TBI status, AD/dementia status, platform). Report class balance per dataset and region. Flag confounders (age, sex, region).

## Phase 2: Preprocessing and harmonization

- Map all probes/IDs to gene symbols. Keep genes present in all datasets.
- Normalize appropriately per platform (VST or log-CPM for RNA-seq; check microarray is already normalized).
- Batch correction across datasets with ComBat, preserving biological labels. Show PCA/UMAP before and after, colored by dataset and by diagnosis.
- Reduce to a tractable gene set (about 2000 to 5000 genes): highly variable genes plus genes in the PPI network.

## Phase 3: Classical analysis (baseline biology)

- Differential expression: AD vs control, and TBI vs non-TBI (within the Allen cohort, adjusting for age, sex, region, and dementia status).
- Overlap of the two DE gene lists, with a statistical test (hypergeometric) and enrichment analysis (GO, KEGG, Reactome via gseapy).
- This phase must stand on its own: it is the sanity check for everything after.

## Phase 4: Gene graph construction

- Download the STRING human PPI network, keep high-confidence edges (combined score >= 700), restrict to the selected genes.
- Optionally add a co-expression layer and compare both graphs.
- Report graph statistics (nodes, edges, connected components, isolated genes).

## Phase 5: Models

- Baselines on the same gene set: logistic regression with L1/L2, random forest, MLP.
- GAT graph-level classifier: one graph per sample, shared topology, node features = expression (plus optional gene embeddings). Global pooling, then classifier. Keep it small enough for CPU.
- Train on AD vs control with donor-level cross-validation. Report AUROC, AUPRC, balanced accuracy, with confidence intervals.
- Check that the model does not learn dataset or region identity (test performance across held-out datasets if possible).

## Phase 6: Transfer to TBI and interpretation

- Apply the trained models to Allen samples, stratified by TBI status and dementia status.
- Statistical comparison of AD-likeness scores (TBI vs no TBI, controlling for age, sex, region, dementia). Effect sizes, not only p-values.
- Explanations: Integrated Gradients (captum) and/or GNNExplainer, plus attention weights. Aggregate to gene-level and pathway-level importance.
- Robustness: repeat with several seeds and folds, keep only genes stable across runs.
- Compare top genes and pathways with the Phase 3 overlap and with published TBI/AD findings (for example myeloid/macrophage and glial inflammation signatures). Report what matches and what is new, and label new findings as hypotheses.

## Phase 7: Deliverables

- Clean README: question, data, method, results, limitations (small TBI sample size, post-mortem tissue, batch effects, association only), how to reproduce.
- Figures ready for slides: PCA before/after correction, DE overlap, model performance vs baselines, score distributions TBI vs non-TBI, gene/pathway importance network.
- A small Streamlit app: pick a sample or region, see its AD-likeness score, and explore the top genes and their network neighborhood.
