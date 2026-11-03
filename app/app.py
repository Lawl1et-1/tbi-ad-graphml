"""AD-likeness explorer for the Allen Aging, Dementia & TBI cohort.

Run:  .venv/Scripts/streamlit run app/app.py
Reads only files produced by the pipeline (results/phase6, data/processed). Research use only:
scores are model outputs on post-mortem tissue and show association, not causation or diagnosis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
P6, PROC = ROOT / "results" / "phase6", ROOT / "data" / "processed"
REGION_NAMES = {"TCx": "Temporal cortex", "PCx": "Parietal cortex", "HIP": "Hippocampus", "FWM": "White matter"}
MODEL_NAMES = {"lr_l2": "Logistic regression L2 (best)", "gat": "GAT (STRING graph)", "lr_l1": "Logistic regression L1",
               "mlp": "MLP", "rf": "Random forest"}

st.set_page_config(page_title="TBI and AD explorer", layout="wide")


@st.cache_data
def load():
    scores = pd.read_csv(P6 / "scores.csv", index_col=0)
    lr = pd.read_csv(P6 / "lr_gene_importance.csv", index_col=0)
    stable = pd.read_csv(P6 / "lr_stable_genes.csv", index_col=0)
    edges = pd.read_csv(PROC / "graph_string.csv")
    X = pd.read_parquet(PROC / "harmonized_expr.parquet").loc[scores.index[scores.role == "transfer"]]
    mu, sd = np.load(P6 / "models" / "zscore_mu.npy"), np.load(P6 / "models" / "zscore_sd.npy")
    contrib = pd.DataFrame((X.values - mu) / sd * lr.coef_mean.reindex(X.columns).values,
                           index=X.index, columns=X.columns)
    ig = None
    ig_files = sorted(P6.glob("ig_gat_s*.npy"))
    if ig_files and (P6 / "ig_samples.csv").exists():
        ids = pd.read_csv(P6 / "ig_samples.csv").iloc[:, 0].values
        genes = (PROC / "selected_genes.txt").read_text().split("\n")
        ig = pd.DataFrame(np.mean([np.load(f) for f in ig_files], axis=0), index=ids, columns=genes)
    return scores, lr, stable, edges, contrib, ig


scores, lr, stable, edges, contrib, ig = load()
allen = scores[scores.role == "transfer"].copy()
models = [m for m in MODEL_NAMES if m in scores.columns]

st.title("Does a TBI brain look like an Alzheimer's brain?")
st.caption("Models trained on AD vs control brain tissue (3 public cohorts) and applied to the Allen Aging, Dementia & "
           "TBI study. Higher score = more AD-like expression. **Computational association only, not a diagnosis.**")

with st.sidebar:
    model = st.selectbox("Model", models, format_func=MODEL_NAMES.get)
    region = st.selectbox("Brain region", list(REGION_NAMES), format_func=REGION_NAMES.get)
    sub = allen[allen.region == region].copy()
    sub["label"] = (sub.donor_id + " | " + np.where(sub.tbi == 1, "TBI", "no TBI") + " | "
                    + np.where(sub.dementia == 1, "dementia", "no dementia"))
    sample = st.selectbox("Sample", list(sub.index), format_func=lambda s: sub.loc[s, "label"])

# Standardize the score inside Allen so units are comparable across models.
allen["z"] = (allen[model] - allen[model].mean()) / allen[model].std()
sub = allen[allen.region == region]
row = allen.loc[sample]

c1, c2, c3, c4 = st.columns(4)
c1.metric("AD-likeness (SD units)", f"{row.z:+.2f}")
c2.metric("Percentile in this region", f"{(sub.z < row.z).mean() * 100:.0f}%")
c3.metric("TBI with loss of consciousness", "yes" if row.tbi == 1 else "no")
c4.metric("Clinical diagnosis", str(row.diagnosis))

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Score distribution in this region")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(allen.z.min(), allen.z.max(), 30)
    ax.hist(sub.z[sub.tbi == 0], bins=bins, alpha=0.6, label="no TBI", color="#7f8c8d")
    ax.hist(sub.z[sub.tbi == 1], bins=bins, alpha=0.6, label="TBI", color="#c0392b")
    ax.axvline(row.z, color="k", lw=2, label="selected sample")
    ax.set_xlabel("AD-likeness (SD)"); ax.set_ylabel("samples"); ax.legend(fontsize=8)
    st.pyplot(fig)
    g = sub.groupby(["tbi", "dementia"]).z.agg(["mean", "count"]).round(2)
    g.index = [f"{'TBI' if t else 'no TBI'}, {'dementia' if d else 'no dementia'}" for t, d in g.index]
    st.dataframe(g, width="stretch")

with right:
    st.subheader("Genes driving this sample's score")
    if model == "gat" and ig is not None and sample in ig.index:
        top, xlabel = ig.loc[sample].sort_values(key=np.abs, ascending=False).head(20), "Integrated Gradients"
        st.caption("Note: GAT attributions were not stable across the 5 GAT models (see the Phase 6 report).")
    else:
        top, xlabel = contrib.loc[sample].sort_values(key=np.abs, ascending=False).head(20), "LR L2 contribution (coef x z)"
        if model != "lr_l2":
            st.caption("Showing logistic-regression contributions (exact and stable) for this sample.")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.barh(top.index[::-1], top.values[::-1], color=np.where(top.values[::-1] > 0, "#c0392b", "#2980b9"))
    ax.set_xlabel(f"{xlabel} (red = pushes toward AD)")
    st.pyplot(fig)

st.subheader("Gene neighborhood (STRING, score >= 700)")
gene = st.selectbox("Gene", list(stable.index))
G = nx.Graph()
G.add_weighted_edges_from(edges[["a", "b", "weight"]].itertuples(index=False))
if gene in G:
    nbrs = sorted(G.neighbors(gene), key=lambda n: -lr.abs_coef.get(n, 0))
    H = G.subgraph([gene] + nbrs[:25])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pos = nx.spring_layout(H, seed=42)
    col = [lr.ad_meta_z.get(n, 0) for n in H.nodes]
    size = [300 + 3000 * lr.abs_coef.get(n, 0) / lr.abs_coef.max() for n in H.nodes]
    nx.draw_networkx_edges(H, pos, alpha=0.3, ax=ax)
    nodes = nx.draw_networkx_nodes(H, pos, node_color=col, cmap="coolwarm", vmin=-8, vmax=8, node_size=size, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=8, ax=ax)
    plt.colorbar(nodes, ax=ax, label="AD meta z (red = up in AD)")
    ax.set_axis_off()
    st.pyplot(fig)
    st.caption(f"{gene}: {G.degree[gene]} STRING partners among the 4,000 model genes; showing up to 25 with the largest "
               "LR weight. Node size = |LR coefficient|.")
else:
    st.warning(f"{gene} has no high-confidence STRING edge among the model genes.")

st.subheader("Stable AD-score genes (logistic regression, top 200 in at least 4 of 5 models)")
st.dataframe(stable[["coef_mean", "coef_sd", "ad_meta_z", "gat_absIG", "pushes"]].round(4), width="stretch")

st.caption("Data: GEO GSE132903, GSE118553, GSE48350 (training), GSE33000 (external test), GSE104687 / Allen Institute "
           "Aging, Dementia and TBI Study (transfer). Limitations: small TBI sample (106 donors), post-mortem tissue, "
           "batch effects, bulk tissue composition, association only.")
