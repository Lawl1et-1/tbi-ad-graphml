"""Phase 3a: AD vs control differential expression on the array datasets.

Per dataset: gene-wise OLS  expr ~ AD + age + sex (+ region), on that dataset's own log2 data
(not the harmonized table), with donor-clustered robust standard errors (several regions per donor
in GSE118553 / GSE48350). Training datasets are combined by weighted Stouffer meta-analysis
(weights = sqrt(n donors)); GSE33000 is kept apart as replication.

Outputs: results/phase3/de_ad_<dataset>.csv, results/phase3/de_ad_meta.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN = ["GSE132903", "GSE118553", "GSE48350"]
REPL = "GSE33000"


def clustered_ols(Y: np.ndarray, X: np.ndarray, k: int, clusters: np.ndarray):
    """Coefficient k for every column of Y, with cluster-robust (CR1) SE. Returns beta, se, t, df."""
    n, p = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ X.T @ Y
    E = Y - X @ B
    a = XtX_inv[k]                       # row of (X'X)^-1 for coefficient k
    w = X @ a                            # per-sample weight
    codes, uniq = pd.factorize(clusters)
    G = len(uniq)
    U = np.zeros((G, Y.shape[1]))
    np.add.at(U, codes, w[:, None] * E)  # sum over samples inside each cluster
    corr = G / (G - 1) * (n - 1) / (n - p)
    se = np.sqrt((U ** 2).sum(axis=0) * corr)
    beta = B[k]
    return beta, se, beta / se, G - 1


def run_dataset(ds: str, genes: list[str]) -> pd.DataFrame:
    x = pd.read_parquet(PROC / f"expr_{ds}.parquet")[genes]
    x = x.fillna(x.mean())
    m = pd.read_csv(PROC / f"samples_{ds}.csv").set_index("sample_id").loc[x.index]
    keep = (m.role != "excluded") & m.diagnosis.isin(["AD", "control"])
    x, m = x[keep], m[keep]
    age = m.age.fillna(m.age.median())
    cols = {"intercept": 1.0, "AD": (m.diagnosis == "AD").astype(float),
            "age": (age - age.mean()) / age.std(), "male": (m.sex == "M").astype(float)}
    D = pd.DataFrame(cols, index=m.index)
    if m.region.nunique() > 1:
        D = D.join(pd.get_dummies(m.region, prefix="reg", drop_first=True).astype(float))
    beta, se, t, df = clustered_ols(x.values, D.values, 1, m.donor_id.values)
    p = 2 * stats.t.sf(np.abs(t), df)
    res = pd.DataFrame({"logFC": beta, "se": se, "t": t, "p": p}, index=genes)
    res["fdr"] = multipletests(res.p, method="fdr_bh")[1]
    res["z"] = np.sign(res.t) * stats.norm.isf(res.p / 2)
    res.attrs = {"n_samples": int(len(m)), "n_donors": int(m.donor_id.nunique()),
                 "n_AD": int((m.diagnosis == "AD").sum()), "n_ctrl": int((m.diagnosis == "control").sum())}
    res.to_csv(OUT / f"de_ad_{ds}.csv", index_label="gene")
    return res


def main() -> None:
    genes = (PROC / "phase2_gene_universe.txt").read_text().split("\n")
    per = {ds: run_dataset(ds, genes) for ds in TRAIN + [REPL]}
    summary = {ds: {**r.attrs, "fdr05": int((r.fdr < 0.05).sum()),
                    "fdr05_up": int(((r.fdr < 0.05) & (r.t > 0)).sum())} for ds, r in per.items()}

    w = {ds: np.sqrt(per[ds].attrs["n_donors"]) for ds in TRAIN}
    Z = pd.DataFrame({ds: per[ds].z for ds in TRAIN})
    meta = pd.DataFrame(index=genes)
    meta["z_meta"] = sum(Z[ds] * w[ds] for ds in TRAIN) / np.sqrt(sum(v ** 2 for v in w.values()))
    meta["p_meta"] = 2 * stats.norm.sf(meta.z_meta.abs())
    meta["fdr_meta"] = multipletests(meta.p_meta, method="fdr_bh")[1]
    meta["logFC_mean"] = pd.DataFrame({ds: per[ds].logFC for ds in TRAIN}).mean(axis=1)
    meta["n_same_sign"] = (np.sign(Z).eq(np.sign(meta.z_meta), axis=0)).sum(axis=1)
    for ds in TRAIN + [REPL]:
        meta[f"z_{ds}"] = per[ds].z
        meta[f"fdr_{ds}"] = per[ds].fdr
    meta = meta.sort_values("p_meta")
    meta.to_csv(OUT / "de_ad_meta.csv", index_label="gene")

    sig = meta[(meta.fdr_meta < 0.05) & (meta.n_same_sign == 3)]
    summary["meta"] = {"genes_tested": len(genes), "fdr05": int((meta.fdr_meta < 0.05).sum()),
                       "fdr05_consistent_3of3": int(len(sig)), "up": int((sig.z_meta > 0).sum()),
                       "down": int((sig.z_meta < 0).sum())}
    # Replication: do meta-significant genes move the same way in GSE33000?
    r = meta.loc[sig.index]
    summary["replication_GSE33000"] = {
        "sign_agreement": float((np.sign(r.z_meta) == np.sign(r[f"z_{REPL}"])).mean()),
        "spearman_z_all_genes": float(stats.spearmanr(meta.z_meta, meta[f"z_{REPL}"]).statistic)}
    summary["pairwise_spearman_z"] = {f"{a}~{b}": float(stats.spearmanr(per[a].z, per[b].z).statistic)
                                      for i, a in enumerate(TRAIN) for b in TRAIN[i + 1:]}
    (OUT / "de_ad_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nTop 15 meta genes:\n", meta.head(15)[["z_meta", "fdr_meta", "logFC_mean", "n_same_sign", f"z_{REPL}"]].round(3))


if __name__ == "__main__":
    main()
