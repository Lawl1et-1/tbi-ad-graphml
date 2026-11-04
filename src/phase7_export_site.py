"""Phase 7: export compact JSON + figures for the static GitHub Pages site (docs/).

Outputs:
  docs/data/samples.json  Allen samples: metadata, AD-likeness score per model (SD units),
                          top 12 LR-L2 gene contributions per sample
  docs/data/genes.json    stable LR genes with coefficient, AD z, and up to 20 STRING neighbours
  docs/data/summary.json  headline numbers, read from the results files
  docs/figures/*.png      slide figures
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, PROC = ROOT / "results", ROOT / "data" / "processed"
P6 = R / "phase6"
DOCS = ROOT / "docs"
(DOCS / "data").mkdir(parents=True, exist_ok=True)
(DOCS / "figures").mkdir(parents=True, exist_ok=True)
MODELS = ["lr_l2", "gat", "lr_l1", "mlp", "rf"]
TOP_GENES = 12


def r(x, n=3):
    return None if pd.isna(x) else round(float(x), n)


def samples() -> None:
    sc = pd.read_csv(P6 / "scores.csv", index_col=0)
    a = sc[sc.role == "transfer"].copy()
    lr = pd.read_csv(P6 / "lr_gene_importance.csv", index_col=0)
    X = pd.read_parquet(PROC / "harmonized_expr.parquet").loc[a.index]
    mu, sd = np.load(P6 / "models" / "zscore_mu.npy"), np.load(P6 / "models" / "zscore_sd.npy")
    C = pd.DataFrame((X.values - mu) / sd * lr.coef_mean.reindex(X.columns).values, index=X.index, columns=X.columns)
    z = {m: (a[m] - a[m].mean()) / a[m].std() for m in MODELS}
    out = []
    for s, row in a.iterrows():
        top = C.loc[s].sort_values(key=np.abs, ascending=False).head(TOP_GENES)
        out.append({"id": s, "donor": row.donor_id, "region": row.region, "tbi": int(row.tbi),
                    "dementia": int(row.dementia), "dx": row.diagnosis, "braak": r(row.braak, 0),
                    "age": r(row.age, 0), "sex": row.sex,
                    "s": {m: r(z[m][s]) for m in MODELS},
                    "g": [[g, r(v)] for g, v in top.items()]})
    (DOCS / "data" / "samples.json").write_text(json.dumps(out, separators=(",", ":")))
    print("samples", len(out))


def genes() -> None:
    lr = pd.read_csv(P6 / "lr_gene_importance.csv", index_col=0)
    st = pd.read_csv(P6 / "lr_stable_genes.csv", index_col=0)
    e = pd.read_csv(PROC / "graph_string.csv")
    nb: dict[str, list[str]] = {}
    for a_, b_ in e[["a", "b"]].itertuples(index=False):
        nb.setdefault(a_, []).append(b_)
        nb.setdefault(b_, []).append(a_)
    out = {}
    for g, row in st.iterrows():
        ns = sorted(nb.get(g, []), key=lambda n: -lr.abs_coef.get(n, 0))[:20]
        out[g] = {"coef": r(row.coef_mean, 4), "adz": r(row.ad_meta_z, 2), "deg": len(nb.get(g, [])),
                  "n": [[n, r(lr.coef_mean.get(n), 4), r(lr.ad_meta_z.get(n), 2)] for n in ns]}
    (DOCS / "data" / "genes.json").write_text(json.dumps(out, separators=(",", ":")))
    print("genes", len(out))


def summary() -> None:
    cv = pd.read_csv(R / "phase5" / "cv_metrics.csv").set_index("model")
    lodo = pd.read_csv(R / "phase5" / "lodo.csv").groupby("model").auroc.mean()
    tbi = json.loads((P6 / "tbi_tests.json").read_text())
    rob = json.loads((P6 / "robustness.json").read_text())
    ext = json.loads((P6 / "external_test.json").read_text())
    ad = json.loads((R / "phase3" / "de_ad_summary.json").read_text())
    glob = json.loads((R / "phase3" / "overlap_global_test.json").read_text())
    out = {
        "cv": {m: {"auroc": r(cv.loc[m, "auroc_mean"]), "ci": cv.loc[m, "auroc_ens_ci95"]} for m in cv.index},
        "lodo": {m: r(v) for m, v in lodo.items()},
        "tbi": {m: {"beta": r(tbi[m]["mixed_tbi_beta_SD"], 2), "ci": [r(x, 2) for x in tbi[m]["mixed_tbi_CI95"]],
                    "p": r(tbi[m]["mixed_tbi_p"], 3), "pair": r(rob[m]["pair_mean_diff_SD"], 2),
                    "signflip_p": r(rob[m]["signflip_p"], 3), "dose_p": r(tbi[m]["dose_p"], 3),
                    "random_pct": r(rob[m]["random_cov_percentile"], 2)} for m in MODELS},
        "external": {m: {"auroc": r(ext[m]["auroc_all"]), "auroc65": r(ext[m]["auroc_age_ge65"])} for m in MODELS},
        "ad_signature": ad["meta"], "replication_sign_agreement": r(ad["replication_GSE33000"]["sign_agreement"]),
        "gene_level_global_p": r(glob["adjusted"]["perm_p_two_sided"], 2),
    }
    (DOCS / "data" / "summary.json").write_text(json.dumps(out, indent=1))


def figures() -> None:
    for f in (R / "figures").glob("*.png"):
        shutil.copy(f, DOCS / "figures" / f.name)
    print("figures", len(list((DOCS / "figures").glob("*.png"))))


if __name__ == "__main__":
    samples(); genes(); summary(); figures()
