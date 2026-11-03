# Phase 4 report: gene graphs

Date: 2026-11-03. Seed: 42. Script: `src/phase4_graph.py`.
Outputs: `data/processed/graph_string.csv`, `data/processed/graph_coexpr.csv`, `results/phase4/graph_stats.json`,
`results/phase4/degree_distributions.png`.

Nodes = the 4,000 genes selected in Phase 2 (STRING-connected, highest variance in training data, no labels used).

## Graphs

| | STRING v12, score ≥ 700 | Co-expression (k = 10 nearest by \|Spearman ρ\|) |
|---|---|---|
| Built from | curated + experimental protein interactions | training samples only (harmonized), no labels |
| Edges | 18,864 | 36,265 |
| Mean / median / max degree | 9.4 / 5 / 187 | 18.1 / 13 / 155 |
| Isolated genes | **359** | 0 |
| Connected components | 396 | 1 |
| Largest component | 3,559 genes (89%) | 4,000 |
| Average clustering | 0.30 | 0.23 |
| Mean shortest path (giant, sampled) | 4.2 | 3.8 |
| Top hubs | EGFR, SRC, IL6, FN1, CD44, CDC42, EP300, STAT1, IL1B, ITGB1, MYC, JUN | NELL1, HPRT1, ENC1, MOAP1, ATP6V1B2, PNMA2, NAPB, GOT1, UCHL1, GABRA1 |

**Overlap:** 655 shared edges, 7.7× more than expected by chance (86), but Jaccard only 0.012.
The two graphs carry mostly different information: STRING encodes known biology (hubs are signaling and
inflammation genes), co-expression encodes tissue structure (hubs are neuronal genes, i.e. the neuron module that
AD tissue loses).

## Notes for Phase 5
- The 359 isolated STRING genes had STRING partners only outside the selected 4,000. The GAT adds self-loops, so they
  still contribute their own expression, but they receive no neighbour messages.
- STRING hub genes (degree up to 187) will dominate message passing; attention can down-weight neighbours, and
  degree should be checked as a confounder when interpreting attention / importance (hubs may look important just
  because they are hubs).
- Main model uses STRING (prior knowledge, independent of the expression data). Co-expression is the comparison graph.
  Because the co-expression graph is built from training expression, using it in cross-validation is slightly
  optimistic (it saw all training folds); for a strict comparison it should be rebuilt per fold.

## Limitations
- STRING is biased toward well-studied genes (more edges for genes with more publications).
- PPI edges are not brain-specific; many interactions may not happen in adult brain tissue.
