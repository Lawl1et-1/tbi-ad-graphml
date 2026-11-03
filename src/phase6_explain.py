"""Phase 6b: explanations for the AD-likeness score.

- Integrated Gradients (captum) for each of the 5 final GAT models, baseline = average training
  sample (all-zero after z-scoring). Samples: all Allen samples + 150 random training samples.
- GAT layer-1 attention per edge (mean over heads and samples).
- Comparison with LR-L2 coefficients and the Phase 3 AD meta z-scores.
- Stability: gene kept if it is in the top 200 by mean |IG| in >= 4 of 5 models.
- Enrichment of stable genes (background = the 4,000 model genes); published marker ranks.
- Exploratory: genes whose attributions differ between TBI and matched non-TBI donors.

Outputs: results/phase6/ig_<model>.parquet, gene_importance.csv, stable_genes.csv,
         attention_edges.csv, enrich_stable_genes.csv, tbi_attribution_diff.csv, explain_summary.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import gseapy as gp
import joblib
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat import build_edge_index, batch_graph  # noqa: E402
from phase3_enrich import LIBS, read_gmt  # noqa: E402
from phase6_transfer import MOD, OUT, SEEDS, load_all, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TOP = 200
torch.set_num_threads(8)
MARKERS = {
    "microglia/macrophage": ["AIF1", "CD163", "C1QA", "C1QB", "C1QC", "TREM2", "TYROBP", "CSF1R", "CD14", "HLA-DRA",
                             "SPP1", "CD74", "FCGR3A", "ITGAM", "P2RY12", "CX3CR1"],
    "reactive astrocyte": ["GFAP", "SERPINA3", "CHI3L1", "C3", "VIM", "CD44", "OSMR", "S100B"],
    "neuronal/synaptic": ["SST", "VGF", "BDNF", "SNAP25", "SYT1", "NRN1", "GAP43", "CALB1", "PVALB", "NPY"],
    "complement": ["C1QA", "C1QB", "C3", "C4A", "C4B", "CFH", "C1S", "C1R"],
}


def integrated_gradients(Z: np.ndarray, idx: np.ndarray, genes, ei) -> dict[int, np.ndarray]:
    out = {}
    for s in SEEDS:
        path = OUT / f"ig_gat_s{s}.npy"
        if path.exists():
            out[s] = np.load(path); continue
        m = load_model("gat", s, len(genes), ei).model
        m.eval()
        cache = {}

        def fwd(x):
            b = x.shape[0]
            if b not in cache:
                cache[b] = batch_graph(ei, len(genes), b)
            e, bt = cache[b]
            return m(x, e, bt)

        ig = IntegratedGradients(fwd)
        X = torch.tensor(Z[idx], dtype=torch.float32)
        attrs = []
        for i in range(0, len(X), 4):
            attrs.append(ig.attribute(X[i:i + 4], baselines=torch.zeros_like(X[i:i + 4]), n_steps=16,
                                      internal_batch_size=16).detach().numpy())
        out[s] = np.vstack(attrs)
        np.save(path, out[s])
        print("IG done seed", s, flush=True)
    return out


def attention(Z: np.ndarray, idx: np.ndarray, genes, ei) -> pd.DataFrame:
    acc = None
    for s in SEEDS:
        m = load_model("gat", s, len(genes), ei).model
        m.eval()
        with torch.no_grad():
            for i in range(0, len(idx), 16):
                x = torch.tensor(Z[idx[i:i + 16]], dtype=torch.float32)
                e, _ = batch_graph(ei, len(genes), len(x))
                _, (att1, _) = m.node_embed(x, e, return_attention=True)
                a = att1[1].mean(1).reshape(len(x), -1).mean(0).numpy()  # per original edge
                acc = a if acc is None else acc + a
    acc /= len(SEEDS) * int(np.ceil(len(idx) / 16))
    src, dst = ei.numpy()
    df = pd.DataFrame({"source": np.array(genes)[src], "target": np.array(genes)[dst], "attention": acc})
    return df[df.source != df.target].sort_values("attention", ascending=False)


def main() -> None:
    X, meta = load_all()
    genes = list(X.columns)
    mu, sd = np.load(MOD / "zscore_mu.npy"), np.load(MOD / "zscore_sd.npy")
    Z = (X.values.astype(np.float32) - mu) / sd
    ei = build_edge_index(pd.read_csv(PROC / "graph_string.csv"), genes)
    rng = np.random.default_rng(42)
    allen_idx = np.where(meta.role == "transfer")[0]
    train_idx = rng.choice(np.where(meta.role == "train")[0], 150, replace=False)
    idx = np.r_[allen_idx, train_idx]
    pd.Series(X.index[idx]).to_csv(OUT / "ig_samples.csv", index=False)

    ig = integrated_gradients(Z, idx, genes, ei)
    summary = {}
    imp = pd.DataFrame({f"absIG_s{s}": np.abs(a).mean(0) for s, a in ig.items()}, index=genes)
    imp["absIG_mean"] = imp.mean(axis=1)
    imp["signedIG_mean"] = np.mean([a.mean(0) for a in ig.values()], axis=0)
    ranks = imp[[f"absIG_s{s}" for s in SEEDS]].rank(ascending=False)
    imp["n_models_top200"] = (ranks <= TOP).sum(axis=1)
    tops = {s: set(ranks.index[ranks[f"absIG_s{s}"] <= TOP]) for s in SEEDS}
    summary["top200_pairwise_jaccard"] = float(np.mean([len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                                                       for a, b in combinations(SEEDS, 2)]))
    summary["spearman_absIG_between_models_mean"] = float(np.mean(
        [stats.spearmanr(imp[f"absIG_s{a}"], imp[f"absIG_s{b}"]).statistic for a, b in combinations(SEEDS, 2)]))

    # Comparisons: LR-L2 coefficients, AD meta z, node degree.
    coef = np.mean([joblib.load(MOD / f"lr_l2_s{s}.joblib").coef_[0] for s in SEEDS], axis=0)
    imp["lr_l2_coef"] = coef
    ad = pd.read_csv(ROOT / "results" / "phase3" / "de_ad_meta.csv", index_col=0)
    imp["ad_meta_z"] = ad.z_meta.reindex(genes).values
    deg = pd.Series(np.bincount(ei[0].numpy(), minlength=len(genes)) - 1, index=genes)
    imp["string_degree"] = deg.values
    summary["spearman_absIG_vs_abs_lr_coef"] = float(stats.spearmanr(imp.absIG_mean, imp.lr_l2_coef.abs()).statistic)
    summary["spearman_absIG_vs_abs_ad_z"] = float(stats.spearmanr(imp.absIG_mean, imp.ad_meta_z.abs()).statistic)
    summary["spearman_absIG_vs_degree"] = float(stats.spearmanr(imp.absIG_mean, imp.string_degree).statistic)
    imp = imp.sort_values("absIG_mean", ascending=False)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp.to_csv(OUT / "gene_importance.csv", index_label="gene")

    stable = imp[imp.n_models_top200 >= 4].copy()
    stable["direction"] = np.where(stable.ad_meta_z > 0, "up_in_AD", "down_in_AD")
    stable.to_csv(OUT / "stable_genes.csv", index_label="gene")
    summary["n_stable_genes"] = int(len(stable))
    summary["stable_genes_in_phase3_AD_signature"] = int(((ad.reindex(stable.index).fdr_meta < 0.05) &
                                                          (ad.reindex(stable.index).n_same_sign == 3)).sum())
    summary["stable_top30"] = stable.index[:30].tolist()

    # Phase 3 overlap genes (AD signature & TBI p<0.01, same direction) among stable genes.
    p3 = set()
    for reg in ["TCx", "PCx", "HIP", "FWM"]:
        r = pd.read_csv(ROOT / "results" / "phase3" / f"de_tbi_adjusted_{reg}.csv", index_col=0)
        hit = r.index[(r.p < 0.01)]
        sig = ad[(ad.fdr_meta < 0.05) & (ad.n_same_sign == 3)]
        same = [g for g in hit if g in sig.index and np.sign(r.loc[g, "stat"]) == np.sign(sig.loc[g, "z_meta"])]
        p3 |= set(same)
    summary["phase3_overlap_genes_total"] = len(p3)
    summary["phase3_overlap_genes_in_stable"] = sorted(p3 & set(stable.index))

    # Enrichment of stable genes against the 4,000-gene background.
    rows = []
    for name in LIBS:
        for direction, g in [("all", stable.index), ("up_in_AD", stable.index[stable.direction == "up_in_AD"]),
                             ("down_in_AD", stable.index[stable.direction == "down_in_AD"])]:
            if len(g) < 5:
                continue
            r = gp.enrich(gene_list=list(g), gene_sets=read_gmt(name), background=genes, no_plot=True).results
            r.insert(0, "library", name); r.insert(0, "genes", direction)
            rows.append(r)
    enr = pd.concat(rows)
    enr.to_csv(OUT / "enrich_stable_genes.csv", index=False)
    summary["enrichment_top"] = (enr[enr["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
                                 .groupby(["genes", "library"]).head(3)[["genes", "library", "Term", "Overlap", "Adjusted P-value"]]
                                 .to_dict("records"))

    # Published markers: percentile rank in the IG ranking (100 = most important).
    summary["marker_percentiles"] = {k: {g: round(100 * (1 - (imp.loc[g, "rank"] - 1) / len(imp)), 1)
                                         for g in v if g in imp.index} for k, v in MARKERS.items()}

    # Attention.
    att = attention(Z, idx, genes, ei)
    att.head(2000).to_csv(OUT / "attention_edges.csv", index=False)
    gene_att = att.groupby("target").attention.sum().reindex(genes).fillna(0)
    summary["spearman_incoming_attention_vs_degree"] = float(stats.spearmanr(gene_att, deg).statistic)
    summary["spearman_incoming_attention_vs_absIG"] = float(stats.spearmanr(gene_att, imp.absIG_mean.reindex(genes)).statistic)
    summary["top_attention_edges"] = att.head(15)[["source", "target", "attention"]].round(4).to_dict("records")

    # Exploratory TBI attribution difference (matched pairs, donor-mean IG, ensemble of models).
    ig_mean = pd.DataFrame(np.mean(list(ig.values()), axis=0), index=X.index[idx], columns=genes)
    a = meta.loc[X.index[allen_idx]]
    dm = ig_mean.loc[a.index].groupby(a.donor_id).mean()
    dinfo = a.groupby("donor_id").agg(tbi=("tbi", "first"), cs=("control_set", "first"))
    diffs = []
    for _, g in dinfo.groupby("cs"):
        if len(g) == 2 and g.tbi.nunique() == 2:
            diffs.append(dm.loc[g.index[g.tbi == 1][0]] - dm.loc[g.index[g.tbi == 0][0]])
    D = pd.DataFrame(diffs)
    test = pd.DataFrame({"mean_diff": D.mean(), "p": [stats.wilcoxon(D[g]).pvalue if D[g].abs().sum() > 0 else 1.0 for g in genes]},
                        index=genes)
    test["fdr"] = multipletests(test.p, method="fdr_bh")[1]
    test["stable"] = test.index.isin(stable.index)
    test = test.sort_values("p")
    test.to_csv(OUT / "tbi_attribution_diff.csv", index_label="gene")
    summary["tbi_attr_pairs"] = len(D)
    summary["tbi_attr_fdr05"] = int((test.fdr < 0.05).sum())
    st = test[test.stable]
    st_fdr = multipletests(st.p, method="fdr_bh")[1]
    summary["tbi_attr_fdr05_within_stable"] = int((st_fdr < 0.05).sum())
    summary["tbi_attr_top10_stable"] = st.head(10)[["mean_diff", "p"]].round(5).reset_index().to_dict("records")
    (OUT / "explain_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "enrichment_top"}, indent=2, default=float))
    print(pd.DataFrame(summary["enrichment_top"]).to_string(index=False))


if __name__ == "__main__":
    main()
