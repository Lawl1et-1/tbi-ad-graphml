"""Phase 6d: interpretation with the logistic-regression (L2) AD score.

Why: LR-L2 is the best model and is exactly decomposable: score = sum_g coef_g * z_g + b.
The GAT explanations were unstable across models (results/phase6/explain_summary.json).

1. Gene stability: top 200 by |coef| in >= 4 of the 5 final LR models.
2. Gene contributions c_g = coef_g * z_g per Allen sample; donor means; matched pair difference
   (TBI minus non-TBI) per gene and per gene set (sum over member genes), Wilcoxon + BH-FDR.
   Gene sets: PanglaoDB brain cell types, KEGG, Reactome (members restricted to the 4,000 model genes).
3. Enrichment of stable genes, published-marker ranks, comparison with Phase 3 / GAT.
Outputs: results/phase6/lr_gene_importance.csv, lr_stable_genes.csv, lr_pathway_tbi_contrib.csv, lr_explain_summary.json
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
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_enrich import read_gmt  # noqa: E402
from phase6_explain import MARKERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P6, PROC = ROOT / "results" / "phase6", ROOT / "data" / "processed"
SEEDS = [101, 102, 103, 104, 105]
TOP = 200
BRAIN_CELLS = ["Microglia", "Macrophages", "Astrocytes", "Oligodendrocytes", "Oligodendrocyte Progenitor Cells",
               "Neurons", "Interneurons", "Pyramidal Cells", "Endothelial Cells", "Pericytes", "Smooth Muscle Cells",
               "Ependymal Cells", "T Cells", "B Cells", "Fibroblasts"]


def pair_matrix(C: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    dm = C.groupby(a.donor_id).mean()
    info = a.groupby("donor_id").agg(tbi=("tbi", "first"), cs=("control_set", "first"))
    rows = []
    for _, g in info.groupby("cs"):
        if len(g) == 2 and g.tbi.nunique() == 2:
            rows.append(dm.loc[g.index[g.tbi == 1][0]] - dm.loc[g.index[g.tbi == 0][0]])
    return pd.DataFrame(rows)


def test_cols(D: pd.DataFrame) -> pd.DataFrame:
    p = [stats.wilcoxon(D[c]).pvalue if D[c].abs().sum() > 0 else 1.0 for c in D]
    r = pd.DataFrame({"mean_diff": D.mean(), "frac_pairs_positive": (D > 0).mean(), "p": p}, index=D.columns)
    r["fdr"] = multipletests(r.p, method="fdr_bh")[1]
    return r.sort_values("p")


def main() -> None:
    X = pd.read_parquet(PROC / "harmonized_expr.parquet")
    sc = pd.read_csv(P6 / "scores.csv", index_col=0)
    genes = list(X.columns)
    mu, sd = np.load(P6 / "models" / "zscore_mu.npy"), np.load(P6 / "models" / "zscore_sd.npy")
    coefs = pd.DataFrame({s: joblib.load(P6 / "models" / f"lr_l2_s{s}.joblib").coef_[0] for s in SEEDS}, index=genes)
    ad = pd.read_csv(ROOT / "results" / "phase3" / "de_ad_meta.csv", index_col=0)
    gat = pd.read_csv(P6 / "gene_importance.csv", index_col=0)
    out = {}

    ranks = coefs.abs().rank(ascending=False)
    tops = {s: set(ranks.index[ranks[s] <= TOP]) for s in SEEDS}
    out["lr_top200_pairwise_jaccard"] = float(np.mean([len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                                                      for a, b in combinations(SEEDS, 2)]))
    out["lr_coef_spearman_between_models"] = float(np.mean([stats.spearmanr(coefs[a], coefs[b]).statistic
                                                           for a, b in combinations(SEEDS, 2)]))
    imp = pd.DataFrame({"coef_mean": coefs.mean(1), "coef_sd": coefs.std(1), "n_models_top200": (ranks <= TOP).sum(1),
                        "ad_meta_z": ad.z_meta.reindex(genes), "gat_absIG": gat.absIG_mean.reindex(genes)})
    imp["abs_coef"] = imp.coef_mean.abs()
    out["spearman_coef_vs_ad_z"] = float(stats.spearmanr(imp.coef_mean, imp.ad_meta_z, nan_policy="omit").statistic)
    out["spearman_abscoef_vs_gat_absIG"] = float(stats.spearmanr(imp.abs_coef, imp.gat_absIG).statistic)
    imp = imp.sort_values("abs_coef", ascending=False)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp.to_csv(P6 / "lr_gene_importance.csv", index_label="gene")
    stable = imp[imp.n_models_top200 >= 4].copy()
    stable["pushes"] = np.where(stable.coef_mean > 0, "toward_AD", "away_from_AD")
    stable.to_csv(P6 / "lr_stable_genes.csv", index_label="gene")
    out["lr_n_stable"] = int(len(stable))
    out["lr_stable_top40"] = stable.index[:40].tolist()
    out["lr_stable_sign_agrees_with_ad_z"] = float((np.sign(stable.coef_mean) == np.sign(stable.ad_meta_z)).mean())
    out["marker_percentiles_lr"] = {k: {g: round(100 * (1 - (imp.loc[g, "rank"] - 1) / len(imp)), 1)
                                        for g in v if g in imp.index} for k, v in MARKERS.items()}

    rows = []
    for name in ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022", "PanglaoDB_Augmented_2021"]:
        for lbl, g in [("toward_AD", stable.index[stable.pushes == "toward_AD"]),
                       ("away_from_AD", stable.index[stable.pushes == "away_from_AD"])]:
            r = gp.enrich(gene_list=list(g), gene_sets=read_gmt(name), background=genes, no_plot=True).results
            r.insert(0, "library", name); r.insert(0, "genes", lbl)
            rows.append(r)
    enr = pd.concat(rows)
    enr.to_csv(P6 / "lr_enrich_stable_genes.csv", index=False)
    out["lr_enrichment_top"] = (enr[enr["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
                                .groupby(["genes", "library"]).head(4)[["genes", "library", "Term", "Overlap", "Adjusted P-value"]]
                                .to_dict("records"))

    # Contributions in Allen samples.
    a = sc[sc.role == "transfer"]
    Z = (X.loc[a.index].values - mu) / sd
    C = pd.DataFrame(Z * imp.coef_mean.reindex(genes).values, index=a.index, columns=genes)
    D = pair_matrix(C, a)
    out["pairs"] = len(D)
    out["check_sum_of_gene_diffs_SDraw"] = float(D.sum(1).mean())
    gt = test_cols(D)
    gt.to_csv(P6 / "lr_gene_tbi_contrib.csv", index_label="gene")
    out["gene_level_tbi_contrib_fdr05"] = int((gt.fdr < 0.05).sum())
    out["gene_level_top10"] = gt.head(10).round(5).reset_index().to_dict("records")

    sets = {}
    for lib in ["PanglaoDB_Augmented_2021", "KEGG_2021_Human", "Reactome_2022"]:
        for term, mem in read_gmt(lib).items():
            if lib.startswith("Panglao") and term not in BRAIN_CELLS:
                continue
            m = [g for g in mem if g in C.columns]
            if 10 <= len(m) <= 400:
                sets[f"{lib.split('_')[0]}: {term}"] = m
    P = pd.DataFrame({k: D[v].sum(1) for k, v in sets.items()})
    pt = test_cols(P)
    pt["n_genes"] = [len(sets[k]) for k in pt.index]
    # Express each set's mean pair difference as a share of the total score difference.
    pt["share_of_total_diff"] = pt.mean_diff / D.sum(1).mean()
    pt.to_csv(P6 / "lr_pathway_tbi_contrib.csv", index_label="gene_set")
    out["pathway_sets_tested"] = len(pt)
    out["pathway_fdr05"] = int((pt.fdr < 0.05).sum())
    out["pathway_top15"] = pt.head(15).round(4).reset_index().to_dict("records")
    cells = pt[pt.index.str.startswith("PanglaoDB")]
    out["cell_types"] = cells.round(4).reset_index().to_dict("records")
    (P6 / "lr_explain_summary.json").write_text(json.dumps(out, indent=2, default=float))
    for k in ["lr_top200_pairwise_jaccard", "lr_coef_spearman_between_models", "spearman_coef_vs_ad_z",
              "spearman_abscoef_vs_gat_absIG", "lr_n_stable", "lr_stable_sign_agrees_with_ad_z", "lr_stable_top40",
              "pairs", "gene_level_tbi_contrib_fdr05", "pathway_sets_tested", "pathway_fdr05"]:
        print(k, out[k])
    print(pd.DataFrame(out["lr_enrichment_top"]).to_string(index=False))
    print(pt.head(15)[["n_genes", "mean_diff", "share_of_total_diff", "frac_pairs_positive", "p", "fdr"]].round(4).to_string())
    print(cells[["n_genes", "mean_diff", "share_of_total_diff", "p", "fdr"]].round(4).to_string())
    print(out["marker_percentiles_lr"])


if __name__ == "__main__":
    main()
