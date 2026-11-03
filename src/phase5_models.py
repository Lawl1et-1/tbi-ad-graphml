"""Phase 5: AD vs control classifiers on the harmonized training data.

Same donor-level folds for every model: StratifiedGroupKFold(5) on donors, stratified by
label x dataset, 3 repeats (seeds 42, 43, 44). Genes are z-scored with training-fold statistics.

Usage:
  python src/phase5_models.py baselines
  python src/phase5_models.py gat --graph string --seeds 42 43
  python src/phase5_models.py gat --graph random|none|coexpr --seeds 42
  python src/phase5_models.py lodo          # leave-one-dataset-out, all models
  python src/phase5_models.py evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat import GATClassifier, build_edge_index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase5"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44]
# Chosen in src/phase5_dev.py on validation data only (gene readout: 0.90 vs pooling: 0.52).
GAT_CFG = dict(hidden=8, heads=2, emb_dim=4, batch_size=16, epochs=40, patience=8, lr=3e-3, weight_decay=1e-4,
               readout="gene", min_epochs=10)
torch.set_num_threads(8)


# ------------------------------------------------------------------ data
def load():
    X = pd.read_parquet(PROC / "harmonized_expr.parquet")
    meta = pd.read_csv(PROC / "harmonized_samples.csv").set_index("sample_id").loc[X.index]
    keep = meta.role == "train"
    X, meta = X[keep], meta[keep]
    y = (meta.diagnosis == "AD").astype(int).values
    return X, meta, y


def splits(meta: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    path = OUT / "splits.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    strat = pd.Series(y, index=meta.index).astype(str) + "_" + meta.dataset
    cols = {}
    for s in SEEDS:
        f = np.zeros(len(meta), int)
        for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(meta, strat, meta.donor_id)):
            f[te] = k
        cols[f"fold_s{s}"] = f
    df = pd.DataFrame(cols, index=meta.index)
    # Sanity: no donor in two folds.
    for c in df:
        assert (df.groupby(meta.donor_id)[c].nunique() == 1).all()
    df.to_csv(path)
    return df


def zscore(Xtr: np.ndarray, *others: np.ndarray):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others]


# ------------------------------------------------------------- baselines
def tuned_lr(penalty: str, Xtr, ytr, groups, seed):
    """Pick C by inner 3-fold donor-grouped CV (AUROC)."""
    best, best_c = -1, None
    Cs = [0.001, 0.01, 0.1, 1.0] if penalty == "l2" else [0.01, 0.05, 0.1, 0.5]
    for C in Cs:
        aucs = []
        for tr, va in GroupKFold(3).split(Xtr, ytr, groups):
            m = make_lr(penalty, C, seed).fit(Xtr[tr], ytr[tr])
            aucs.append(roc_auc_score(ytr[va], m.predict_proba(Xtr[va])[:, 1]))
        if np.mean(aucs) > best:
            best, best_c = np.mean(aucs), C
    return make_lr(penalty, best_c, seed).fit(Xtr, ytr), best_c


def make_lr(penalty, C, seed):
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=5000, class_weight="balanced",
                              random_state=seed)


def make_baseline(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced",
                                      n_jobs=8, random_state=seed)
    if name == "mlp":
        return MLPClassifier((256, 64), alpha=1e-2, early_stopping=True, validation_fraction=0.15,
                             max_iter=300, random_state=seed)
    raise ValueError(name)


def fit_predict(name: str, Xtr, ytr, gtr, Xte, seed, edge_index=None):
    info = {}
    if name in ("lr_l1", "lr_l2"):
        m, c = tuned_lr(name[-2:], Xtr, ytr, gtr, seed)
        info["C"] = c
        if name == "lr_l1":
            info["nonzero"] = int((m.coef_ != 0).sum())
    elif name.startswith("gat"):
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(Xtr, ytr, gtr))
        m = GATClassifier(edge_index, Xtr.shape[1], seed=seed, **GAT_CFG).fit(Xtr[tr], ytr[tr], Xtr[va], ytr[va])
        info["best_epoch"] = m.best_epoch
    else:
        m = make_baseline(name, seed).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], info


# ----------------------------------------------------------------- graphs
def graph_edges(kind: str, genes: list[str]) -> torch.Tensor:
    if kind == "none":
        return build_edge_index(pd.DataFrame(columns=["a", "b"]), genes)
    if kind == "coexpr":
        return build_edge_index(pd.read_csv(PROC / "graph_coexpr.csv"), genes)
    e = pd.read_csv(PROC / "graph_string.csv")
    if kind == "random":
        # Degree-preserving rewiring of the STRING graph: same degrees, random partners.
        G = nx.Graph(); G.add_nodes_from(genes); G.add_edges_from(e[["a", "b"]].itertuples(index=False))
        nx.double_edge_swap(G, nswap=10 * G.number_of_edges(), max_tries=100 * G.number_of_edges(), seed=42)
        e = pd.DataFrame(list(G.edges()), columns=["a", "b"])
    return build_edge_index(e, genes)


# ---------------------------------------------------------------- runners
def run_cv(names: list[str], seeds: list[int], edge_index=None, tag: str | None = None) -> None:
    X, meta, y = load()
    sp = splits(meta, y)
    Xv, groups = X.values.astype(np.float32), meta.donor_id.values
    for name in names:
        label = tag or name
        path = OUT / f"oof_{label}.csv"
        prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
        rows = [prev] if len(prev) else []
        for seed in seeds:
            if len(prev) and prev[prev.seed == seed].fold.nunique() == 5:
                print(f"skip {label} seed {seed} (done)"); continue
            if len(prev):  # drop a partially finished seed and redo it
                rows = [prev[prev.seed != seed]]
            for k in range(5):
                t0 = time.time()
                te = (sp[f"fold_s{seed}"] == k).values
                Xtr, Xte = zscore(Xv[~te], Xv[te])
                p, info = fit_predict(name, Xtr, y[~te], groups[~te], Xte, seed, edge_index)
                rows.append(pd.DataFrame({"sample_id": X.index[te], "seed": seed, "fold": k, "prob": p,
                                          "y": y[te], "info": json.dumps(info)}))
                print(f"{label} seed {seed} fold {k}: AUROC {roc_auc_score(y[te], p):.3f} "
                      f"{info} {time.time() - t0:.0f}s", flush=True)
                pd.concat(rows).to_csv(path, index=False)  # checkpoint after each fold


def run_lodo(names: list[str], edge_index) -> None:
    """Train on two training datasets, test on the third (dataset-identity check)."""
    X, meta, y = load()
    Xv, groups = X.values.astype(np.float32), meta.donor_id.values
    path = OUT / "lodo.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["held_out"], r["model"]) for r in rows}
    for ds in sorted(meta.dataset.unique()):
        te = (meta.dataset == ds).values
        Xtr, Xte = zscore(Xv[~te], Xv[te])
        for name in names:
            if (ds, name) in done:
                continue
            t0 = time.time()
            p, info = fit_predict(name, Xtr, y[~te], groups[~te], Xte, 42, edge_index)
            r = {"held_out": ds, "model": name, "n_test": int(te.sum()), "auroc": roc_auc_score(y[te], p),
                 "auprc": average_precision_score(y[te], p),
                 "bal_acc": balanced_accuracy_score(y[te], p > 0.5), **info}
            rows.append(r)
            print(r, f"{time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "lodo.csv", index=False)


# -------------------------------------------------------------- evaluation
def donor_bootstrap(df: pd.DataFrame, donors: pd.Series, n=1000, seed=42) -> dict:
    """95% CI by resampling donors (all of a donor's samples together)."""
    rng = np.random.default_rng(seed)
    d = df.assign(donor=donors.loc[df.index].values)
    by = {k: v for k, v in d.groupby("donor")}
    keys = np.array(list(by))
    stats_ = {"auroc": [], "auprc": [], "bal_acc": []}
    for _ in range(n):
        s = pd.concat([by[k] for k in rng.choice(keys, len(keys))])
        if s.y.nunique() < 2:
            continue
        stats_["auroc"].append(roc_auc_score(s.y, s.prob))
        stats_["auprc"].append(average_precision_score(s.y, s.prob))
        stats_["bal_acc"].append(balanced_accuracy_score(s.y, s.prob > 0.5))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats_.items()}


def evaluate() -> None:
    _, meta, _ = load()
    rows, per_ds = [], []
    for f in sorted(OUT.glob("oof_*.csv")):
        model = f.stem[4:]
        o = pd.read_csv(f)
        per_seed = o.groupby("seed").apply(lambda g: pd.Series({
            "auroc": roc_auc_score(g.y, g.prob), "auprc": average_precision_score(g.y, g.prob),
            "bal_acc": balanced_accuracy_score(g.y, g.prob > 0.5)}))
        ens = o.groupby("sample_id").agg(prob=("prob", "mean"), y=("y", "first"))
        ci = donor_bootstrap(ens, meta.donor_id)
        row = {"model": model, "n_seeds": int(o.seed.nunique())}
        for k in ["auroc", "auprc", "bal_acc"]:
            row[f"{k}_mean"] = per_seed[k].mean()
            row[f"{k}_sd_over_seeds"] = per_seed[k].std() if len(per_seed) > 1 else np.nan
            row[f"{k}_ens_ci95"] = f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
        rows.append(row)
        e = ens.join(meta.dataset)
        for ds, g in e.groupby("dataset"):
            per_ds.append({"model": model, "dataset": ds, "auroc": roc_auc_score(g.y, g.prob)})
    res = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False)
    res.to_csv(OUT / "cv_metrics.csv", index=False)
    pd.DataFrame(per_ds).pivot(index="model", columns="dataset", values="auroc").to_csv(OUT / "cv_auroc_by_dataset.csv")
    print(res.round(3).to_string(index=False))
    print(pd.DataFrame(per_ds).pivot(index="model", columns="dataset", values="auroc").round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baselines", "gat", "lodo", "evaluate"])
    ap.add_argument("--graph", default="string", choices=["string", "random", "none", "coexpr"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--models", nargs="+", default=["lr_l2", "lr_l1", "rf", "mlp", "gat"])
    a = ap.parse_args()
    genes = (PROC / "selected_genes.txt").read_text().split("\n")
    if a.cmd == "baselines":
        run_cv(["lr_l2", "lr_l1", "rf", "mlp"], a.seeds)
    elif a.cmd == "gat":
        run_cv(["gat"], a.seeds, graph_edges(a.graph, genes), tag=f"gat_{a.graph}")
    elif a.cmd == "lodo":
        run_lodo(a.models, graph_edges("string", genes) if "gat" in a.models else None)
    else:
        evaluate()


if __name__ == "__main__":
    main()
