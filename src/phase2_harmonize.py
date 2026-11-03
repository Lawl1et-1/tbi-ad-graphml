"""Phase 2b: gene set selection + ComBat batch correction + before/after diagnostics.

Design choices (see results/phase2/PHASE2_REPORT.md):
- Allen: ComBat on sequencing batch, run separately inside each region (TBI is balanced across batches).
- Gene set: genes in all datasets AND with >= 1 STRING edge (score >= 700) to another such gene,
  then the top N by variance in the training datasets. Labels are never used for selection.
- Cross-dataset ComBat: batch = dataset x region group. No diagnosis covariate, because
  (a) label-aware ComBat inflates group differences when designs are unbalanced, and
  (b) it would need AD labels for the Allen and GSE33000 samples, which are held out.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from combat import combat  # our implementation (src/combat.py)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

SEED = 42
N_GENES = 4000
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
RES = ROOT / "results" / "phase2"
RES.mkdir(parents=True, exist_ok=True)
DATASETS = ["GSE132903", "GSE118553", "GSE48350", "GSE33000", "GSE104687"]
TRAIN = ["GSE132903", "GSE118553", "GSE48350"]
LOG: dict = {"seed": SEED, "n_genes_target": N_GENES}

# Known markers, only used to report whether they survive gene selection.
MARKERS = ["APP", "MAPT", "APOE", "TREM2", "GFAP", "AQP4", "C1QA", "C3", "CD68", "AIF1", "CD163",
           "SST", "VGF", "BDNF", "SNAP25", "SYT1", "CLU", "SPP1", "HLA-DRA", "SERPINA3", "CHI3L1"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = (PROC / "phase2_gene_universe.txt").read_text().split("\n")
    meta = pd.read_csv(PROC / "samples_all.csv")
    xs = []
    for ds in DATASETS:
        x = pd.read_parquet(PROC / f"expr_{ds}.parquet")[universe]
        x = x.fillna(x.mean())  # GSE33000 has 0.003% missing values
        xs.append(x)
    X = pd.concat(xs)
    meta = meta.set_index("sample_id").loc[X.index]
    allen = pd.read_csv(PROC / "samples_GSE104687.csv").set_index("sample_id")
    meta["seq_batch"] = allen["batch"].reindex(meta.index)
    meta["batch"] = meta["dataset"] + "|" + meta["region_group"]
    return X, meta


def allen_batch_correct(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    m = meta[meta.dataset == "GSE104687"]
    sizes = []
    for reg, idx in m.groupby("region").groups.items():
        b = m.loc[idx, "seq_batch"]
        sizes.append(int(b.value_counts().min()))
        X.loc[idx] = combat(X.loc[idx], b.values).values
    LOG["allen_seq_batch_min_samples_per_region_batch"] = min(sizes)
    return X


def select_genes(X: pd.DataFrame, meta: pd.DataFrame) -> list[str]:
    info = pd.read_csv(RAW / "string" / "9606.protein.info.v12.0.txt.gz", sep="\t")
    id2sym = info.set_index("#string_protein_id")["preferred_name"]
    links = pd.read_csv(RAW / "string" / "9606.protein.links.v12.0.txt.gz", sep=" ")
    links = links[links.combined_score >= 700]
    a, b = links.protein1.map(id2sym), links.protein2.map(id2sym)
    genes = set(X.columns)
    ok = a.isin(genes) & b.isin(genes) & (a != b)
    connected = set(a[ok]) | set(b[ok])
    LOG["string_edges_ge700_total"] = int(len(links) // 2)
    LOG["universe_genes_with_string_edge"] = len(connected)

    # Variance rank inside each training dataset, averaged. Unsupervised: no labels.
    ranks = []
    for ds in TRAIN:
        v = X.loc[meta.dataset == ds, sorted(connected)].var()
        ranks.append(v.rank(ascending=False))
    mean_rank = pd.concat(ranks, axis=1).mean(axis=1).sort_values()
    chosen = mean_rank.index[:N_GENES].tolist()
    LOG["genes_selected"] = len(chosen)
    LOG["markers_kept"] = [g for g in MARKERS if g in chosen]
    LOG["markers_missing"] = [g for g in MARKERS if g not in chosen]
    LOG["markers_not_in_universe"] = [g for g in MARKERS if g not in genes]
    return chosen


def diagnostics(X: pd.DataFrame, meta: pd.DataFrame, tag: str) -> dict:
    """PCA + how well dataset / region / diagnosis can be read off the expression."""
    Z = (X - X.mean()) / X.std().replace(0, 1)
    pcs = PCA(n_components=50, random_state=SEED).fit(Z)
    P = pcs.transform(Z)
    out = {"pc1_var": float(pcs.explained_variance_ratio_[0]), "pc2_var": float(pcs.explained_variance_ratio_[1])}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=2000)
    out["dataset_predictable_bal_acc"] = float(cross_val_score(clf, P[:, :20], meta.dataset, cv=cv, scoring="balanced_accuracy").mean())
    out["silhouette_dataset"] = float(silhouette_score(P[:, :20], meta.dataset, random_state=SEED))
    lab = meta.diagnosis.isin(["AD", "control"]) & (meta.role == "train")
    out["silhouette_diagnosis_train"] = float(silhouette_score(P[lab.values, :20], meta.diagnosis[lab], random_state=SEED))
    df = pd.DataFrame(P[:, :2], columns=["PC1", "PC2"], index=X.index).join(meta)
    df["diagnosis"] = df["diagnosis"].where(df["diagnosis"].isin(["AD", "control"]), "other")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, col in zip(ax, ["dataset", "diagnosis", "region_group"]):
        sns.scatterplot(data=df, x="PC1", y="PC2", hue=col, s=10, alpha=0.7, ax=a, linewidth=0)
        a.set_title(f"{tag}: colored by {col}")
        a.legend(fontsize=7, markerscale=1.5)
    for a in ax:
        a.set_xlabel(f"PC1 ({out['pc1_var']:.0%})"); a.set_ylabel(f"PC2 ({out['pc2_var']:.0%})")
    fig.tight_layout()
    fig.savefig(RES / f"pca_{tag}.png", dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    X, meta = load()
    LOG["samples"] = meta.groupby(["role", "dataset"]).size().rename(lambda t: "|".join(t)).to_dict() \
        if False else {f"{r}|{d}": int(n) for (r, d), n in meta.groupby(["role", "dataset"]).size().items()}
    X = allen_batch_correct(X, meta)
    genes = select_genes(X, meta)
    X = X[genes]
    LOG["before"] = diagnostics(X, meta, "before_combat")
    Xc = combat(X, meta.batch.values)
    LOG["after"] = diagnostics(Xc, meta, "after_combat")
    LOG["n_batches"] = int(meta.batch.nunique())
    Xc.astype("float32").to_parquet(PROC / "harmonized_expr.parquet")
    meta.reset_index().to_csv(PROC / "harmonized_samples.csv", index=False)
    (PROC / "selected_genes.txt").write_text("\n".join(genes))
    (RES / "harmonize_log.json").write_text(json.dumps(LOG, indent=2))
    print(json.dumps({k: LOG[k] for k in ["before", "after", "genes_selected", "markers_missing",
                                         "markers_not_in_universe", "universe_genes_with_string_edge"]}, indent=2))


if __name__ == "__main__":
    main()
