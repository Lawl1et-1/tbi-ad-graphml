# Phase 0 + Phase 1 report

Date: 2026-11-03. Seed: 42. Scripts: `src/phase1_metadata.py`, `src/phase1_explore.py`.
Tables: `data/processed/samples_*.csv`. Numbers: `results/phase1/*.csv`, log in `explore_log.txt`.

## Phase 0: setup

- Python 3.12 venv in `.venv` (no `uv` on this machine). `requirements.txt` + exact versions in `requirements-lock.txt`.
- torch 2.15 (CPU), torch_geometric 2.8.1, pandas 3.1, pydeseq2, gseapy, captum, scanpy, streamlit.
- **Change from plan:** `inmoose` needs a C compiler on Windows and failed to build. Replaced by `combat`
  (pure-Python ComBat, same empirical Bayes method). `six` added (GEOparse needs it but does not declare it).
- Folders: `data/raw`, `data/processed`, `src`, `notebooks`, `results`, `app`.

## Phase 1: datasets verified

| Dataset | Exists | Tissue / regions | Platform | Samples | Donors | Labels | Verdict |
|---|---|---|---|---|---|---|---|
| GSE104687 (Allen ADT) | yes | TCx, PCx, HIP, FWM (bulk, macrodissected) | Illumina HiSeq 2500 RNA-seq | 376 (377 on Allen portal) | 106 | TBI w/ LOC, dementia, NINCDS AD dx, Braak, CERAD, NIA-Reagan, APOE4, RIN, batch, matched-pair ID | **Use (TBI cohort)** |
| GSE132903 | yes | middle temporal gyrus (bulk) | Illumina HT-12 v4 | 195 | 195 | AD 97 / ND 98, age, sex | **Use (best region match to TCx)** |
| GSE48350 | yes | HIP, EC, postcentral gyrus, SFG (bulk) | Affymetrix U133 Plus 2 | 253 | ~85 | AD / control, age, sex; Braak+APOE+MMSE for AD only | **Use with care** |
| GSE33000 | yes | DLPFC (bulk) | Rosetta/Agilent 44K, two-colour vs pool | 624 | 624 | AD 310 / ND 157 / Huntington 157, age, sex | **Use with care, or as external test only** |
| GSE5281 | yes | 6 regions, **laser-captured neurons** | Affymetrix U133 Plus 2 | 161 | unknown | AD / control, age, sex | **Drop** |

### Raw data downloaded
- GEO series matrices for all five (`data/raw/GSE*/`), probe annotations for GPL570, GPL10558, GPL4372 (`data/raw/platforms/`; each maps ~19.5–21.8k unique genes).
- Allen portal: donor table (107 donors), file manifest, pathology/protein table, FPKM matrices (normalized + unnormalized),
  and the per-sample RSEM files with **raw `expected_count`** (`data/raw/allen/rsem/`), needed for DESeq2 in Phase 3.
  GEO only carries FPKM for this study. Verified: 377 files, each with the same 50,283 genes and no missing counts;
  all 376 GEO samples have a count file. The extra file (profile 496100315) is the sample the authors excluded; we drop it too.

### Class balance (samples)

| Dataset | Region | AD | Control | Other |
|---|---|---|---|---|
| GSE104687 | TCx / PCx / HIP / FWM | 41 / 37 / 36 / 38 | 50 / 47 / 51 / 48 | 7 per region (non-AD dementia) |
| GSE132903 | MTG | 97 | 98 | |
| GSE48350 | HIP / EC / PCG / SFG | 19 / 15 / 25 / 21 | 43 / 39 / 43 / 48 | controls include ages 20–59 |
| GSE33000 | DLPFC | 310 | 157 | 157 Huntington's (drop) |

Allen donors: **56 TBI, 50 no TBI**. By clinical diagnosis: TBI = 22 AD / 28 control / 6 other dementia;
no TBI = 21 AD / 28 control / 1 other dementia. 52 matched pairs (`control_set`), 2 unpaired donors.

### Confounders (donor level)

**AD-training datasets, AD vs control**

| Dataset | Age AD vs control (median) | p | Sex | p |
|---|---|---|---|---|
| GSE33000 | 82 vs **62** | 3e-45 | AD 56% F, control **22% F** | 2e-12 |
| GSE132903 | 86 vs 85 | 0.89 | balanced | 1.0 |
| GSE48350 (≥60 y) | 85.5 vs 83 | 0.31 | balanced | 0.80 |
| GSE5281 | 79 vs 79 | 0.86 | 43% vs 28% F | 0.09 |

GSE33000 is badly confounded: a model trained on it could learn age and sex instead of AD.
Restricting controls to ≥65 y leaves 68 controls, still 11 years younger (72 vs 83).

**Allen, TBI vs no TBI**: no significant difference in age, sex, dementia, Braak, CERAD, NIA-Reagan,
APOE4, education, race, RIN, region, or sequencing batch (all p > 0.19). The cohort was designed as matched
pairs, and it shows. Braak 5–6 is somewhat more frequent in TBI donors (20 vs 12) but not significant (p = 0.33).

Figures: `confounders_overview.png`, `class_balance.png`.

## Problems found and proposals

1. **GSE5281 is unsuitable.** It profiles laser-captured neurons, not bulk tissue, so its expression reflects one
   cell type and won't transfer to Allen bulk RNA-seq. Donor IDs are also missing (the 161 samples come from
   far fewer donors, so a donor split is impossible). **Proposal: drop it and add GSE118553** (verified:
   Illumina HT-12, same platform as GSE132903; bulk temporal cortex 115, entorhinal 98, frontal 96, cerebellum 92;
   AD 167 / asymptomatic AD 134 / control 100). Its temporal and entorhinal samples match Allen TCx and HIP.
2. **GSE33000 age/sex confounding.** Proposal: keep it out of training and use it as an *external held-out test*
   (does the signature still separate AD from control in another region and platform?), and report results
   with age adjustment.
3. **GSE48350 AD donors have no ID.** I inferred donors from (sex, age, Braak, APOE, MMSE): 28 AD donors.
   This is approximate. Young controls (20–59 y, 33 donors) must be dropped.
4. **Value scales differ.** GSE33000 = log10 ratio vs pool (clipped ±2); GSE132903 = log2; GSE48350 and GSE5281 =
   linear MAS5-like values. Phase 2 will log-transform, and GSE48350 may be re-normalized from CEL files (RMA).
5. **No PMI** per donor in any series matrix (GSE33000 gives only a cohort mean, 17.8 ± 8.3 h). Limitation.
6. **Allen has its own AD labels.** 43 donors with clinical AD. This gives a built-in sanity check for Phase 6:
   the AD-likeness score should be higher in Allen AD donors than Allen controls before we look at TBI.

## Limitations so far
- Allen TBI = self-reported TBI with loss of consciousness, often decades before death; no severity grading beyond LOC duration.
- Allen AD label is clinical (NINCDS-ADRDA possible/probable), not purely neuropathological.
- Only 106 Allen donors → limited power for TBI comparisons; effects must be reported with CIs.
- All tissue is post-mortem; no PMI available for adjustment.
