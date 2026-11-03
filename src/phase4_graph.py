"""Phase 4: gene graphs over the 4,000 selected genes.

1. STRING v12, combined score >= 700 (high confidence), undirected, both genes selected.
2. Co-expression: Spearman correlation on the *training* samples only (harmonized data), each gene
   linked to its k = 10 most correlated genes by |rho| (symmetrized). Built without labels.
Graph statistics for both, plus their edge overlap.

Outputs: data/processed/graph_string.csv, graph_coexpr.csv (gene_a, gene_b, weight),
         results/phase4/graph_stats.json, degree_distributions.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

SEED = 42
K = 10
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)


def string_edges(genes: list[str]) -> pd.DataFrame:
    info = pd.read_csv(RAW / "string" / "9606.protein.info.v12.0.txt.gz", sep="\t")
    id2sym = info.set_index("#string_protein_id")["preferred_name"]
    links = pd.read_csv(RAW / "string" / "9606.protein.links.v12.0.txt.gz", sep=" ")
    links = links[links.combined_score >= 700]
    e = pd.DataFrame({"a": links.protein1.map(id2sym), "b": links.protein2.map(id2sym),
                      "weight": links.combined_score / 1000})
    gs = set(genes)
    e = e[e.a.isin(gs) & e.b.isin(gs) & (e.a != e.b)]
    e[["a", "b"]] = np.sort(e[["a", "b"]].values, axis=1)  # undirected
    return e.groupby(["a", "b"], as_index=False).weight.max()


def coexpr_edges(genes: list[str]) -> pd.DataFrame:
    X = pd.read_parquet(PROC / "harmonized_expr.parquet")[genes]
    meta = pd.read_csv(PROC / "harmonized_samples.csv").set_index("sample_id").loc[X.index]
    R = X[meta.role == "train"].rank().values
    R = (R - R.mean(0)) / R.std(0)
    C = (R.T @ R) / R.shape[0]  # Spearman
    np.fill_diagonal(C, 0)
    nn = np.argsort(-np.abs(C), axis=1)[:, :K]
    rows = [(genes[i], genes[j], C[i, j]) for i in range(len(genes)) for j in nn[i]]
    e = pd.DataFrame(rows, columns=["a", "b", "weight"])
    e[["a", "b"]] = np.sort(e[["a", "b"]].values, axis=1)
    return e.drop_duplicates(["a", "b"])


def stats_for(e: pd.DataFrame, genes: list[str]) -> dict:
    G = nx.Graph()
    G.add_nodes_from(genes)
    G.add_edges_from(e[["a", "b"]].itertuples(index=False))
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    deg = np.array([d for _, d in G.degree()])
    giant = G.subgraph(max(nx.connected_components(G), key=len))
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "isolated_genes": int((deg == 0).sum()), "components": len(comps),
            "largest_component": comps[0], "second_component": comps[1] if len(comps) > 1 else 0,
            "mean_degree": float(deg.mean()), "median_degree": float(np.median(deg)), "max_degree": int(deg.max()),
            "density": float(nx.density(G)), "avg_clustering": float(nx.average_clustering(G)),
            "top_hubs": [g for g, _ in sorted(G.degree, key=lambda x: -x[1])[:15]],
            "giant_avg_shortest_path_sample": float(np.mean([
                nx.shortest_path_length(giant, *np.random.default_rng(SEED + i).choice(list(giant.nodes), 2, replace=False))
                for i in range(300)]))}, deg


def main() -> None:
    genes = (PROC / "selected_genes.txt").read_text().split("\n")
    s, c = string_edges(genes), coexpr_edges(genes)
    s.to_csv(PROC / "graph_string.csv", index=False)
    c.to_csv(PROC / "graph_coexpr.csv", index=False)
    st, ds = stats_for(s, genes)
    sc, dc = stats_for(c, genes)
    se, ce = set(map(tuple, s[["a", "b"]].values)), set(map(tuple, c[["a", "b"]].values))
    n_pairs = len(genes) * (len(genes) - 1) / 2
    shared = len(se & ce)
    out = {"string_ge700": st, "coexpression_knn10": sc,
           "edge_overlap": {"shared_edges": shared, "jaccard": shared / len(se | ce),
                            "expected_shared_if_random": len(se) * len(ce) / n_pairs,
                            "fold_over_random": shared / (len(se) * len(ce) / n_pairs)}}
    (OUT / "graph_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, d, t in [(ax[0], ds, "STRING (score ≥ 700)"), (ax[1], dc, f"Co-expression (kNN, k={K})")]:
        a.hist(d, bins=60, color="#34495e")
        a.set_yscale("log"); a.set_xlabel("degree"); a.set_ylabel("genes (log)"); a.set_title(t)
    fig.tight_layout()
    fig.savefig(OUT / "degree_distributions.png", dpi=140)


if __name__ == "__main__":
    main()
