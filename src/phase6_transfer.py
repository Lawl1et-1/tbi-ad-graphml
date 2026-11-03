"""Phase 6a: final models -> AD-likeness scores on GSE33000 (external) and Allen (TBI cohort).

Final models: for each model type, 5 models, each fit on a random 80% of training donors (seed s),
the other 20% used for GAT early stopping. Score = mean predicted log-odds over the 5 models.
Nothing from Allen or GSE33000 is used for fitting or z-scoring (training-set statistics only).

Usage: python src/phase6_transfer.py fit [--models ...] ; python src/phase6_transfer.py score
Outputs: results/phase6/models/, scores.csv, external_test.json, allen_sanity.csv, tbi_tests.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat import GATClassifier, build_edge_index  # noqa: E402
from phase5_models import GAT_CFG, make_baseline, tuned_lr, zscore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase6"
MOD = OUT / "models"
MOD.mkdir(parents=True, exist_ok=True)
SEEDS = [101, 102, 103, 104, 105]
MODELS = ["lr_l2", "lr_l1", "rf", "mlp", "gat"]
torch.set_num_threads(8)


def load_all():
    X = pd.read_parquet(PROC / "harmonized_expr.parquet")
    meta = pd.read_csv(PROC / "harmonized_samples.csv").set_index("sample_id").loc[X.index]
    allen = pd.read_csv(PROC / "samples_GSE104687.csv").set_index("sample_id")
    for c in ["tbi", "dementia", "braak", "cerad", "nia_reagan", "rin", "num_tbi_w_loc", "control_set", "apoe4"]:
        meta[c] = allen[c].reindex(meta.index).combine_first(meta[c]) if c in meta else allen[c].reindex(meta.index)
    return X, meta


def fit(models: list[str]) -> None:
    X, meta = load_all()
    tr = (meta.role == "train").values
    Xtr_all = X.values[tr].astype(np.float32)
    y = (meta.diagnosis[tr] == "AD").astype(int).values
    groups = meta.donor_id.values[tr]
    mu, sd = Xtr_all.mean(0), Xtr_all.std(0)
    sd[sd == 0] = 1
    np.save(MOD / "zscore_mu.npy", mu); np.save(MOD / "zscore_sd.npy", sd)
    Z = (Xtr_all - mu) / sd
    genes = list(X.columns)
    ei = build_edge_index(pd.read_csv(PROC / "graph_string.csv"), genes)
    for name in models:
        for s in SEEDS:
            path = MOD / f"{name}_s{s}.{'pt' if name == 'gat' else 'joblib'}"
            if path.exists():
                print("exists", path.name); continue
            t0 = time.time()
            a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(Z, y, groups))
            if name == "gat":
                m = GATClassifier(ei, len(genes), seed=s, **GAT_CFG).fit(Z[a], y[a], Z[b], y[b])
                torch.save({"state": m.model.state_dict(), "cfg": GAT_CFG, "best_epoch": m.best_epoch}, path)
                info = {"best_epoch": m.best_epoch}
            elif name.startswith("lr"):
                m, c = tuned_lr(name[-2:], Z[a], y[a], groups[a], s)
                joblib.dump(m, path); info = {"C": c}
            else:
                m = make_baseline(name, s).fit(Z[a], y[a]); joblib.dump(m, path); info = {}
            val_auc = roc_auc_score(y[b], m.predict_proba(Z[b])[:, 1])
            print(f"{name} s{s}: held-out 20% AUROC {val_auc:.3f} {info} {time.time() - t0:.0f}s", flush=True)


def load_model(name: str, s: int, n_genes: int, ei):
    if name == "gat":
        ck = torch.load(MOD / f"gat_s{s}.pt")
        m = GATClassifier(ei, n_genes, seed=s, **ck["cfg"])
        from gat import GeneGAT
        kw = {k: ck["cfg"][k] for k in ["hidden", "heads", "emb_dim", "readout"] if k in ck["cfg"]}
        m.model = GeneGAT(n_genes, **kw); m.model.load_state_dict(ck["state"]); m.model.eval()
        return m
    return joblib.load(MOD / f"{name}_s{s}.joblib")


def score() -> None:
    X, meta = load_all()
    mu, sd = np.load(MOD / "zscore_mu.npy"), np.load(MOD / "zscore_sd.npy")
    Z = (X.values.astype(np.float32) - mu) / sd
    genes = list(X.columns)
    ei = build_edge_index(pd.read_csv(PROC / "graph_string.csv"), genes)
    sc = pd.DataFrame(index=X.index)
    for name in MODELS:
        logits = []
        for s in SEEDS:
            if not (MOD / f"{name}_s{s}.{'pt' if name == 'gat' else 'joblib'}").exists():
                continue
            p = load_model(name, s, len(genes), ei).predict_proba(Z)[:, 1].clip(1e-6, 1 - 1e-6)
            logits.append(np.log(p / (1 - p)))
        if logits:
            sc[name] = np.mean(logits, axis=0)
            sc[f"{name}_n_models"] = len(logits)
    sc.join(meta).to_csv(OUT / "scores.csv", index_label="sample_id")
    models = [m for m in MODELS if m in sc]
    external(sc, meta, models)
    allen_tests(sc, meta, models)


def external(sc, meta, models) -> None:
    g = meta[meta.role == "test_ext"].join(sc)
    g["y"] = (g.diagnosis == "AD").astype(int)
    old = g[g.age >= 65]
    out = {}
    for m in models:
        g["z"] = (g[m] - g[m].mean()) / g[m].std()
        fit_ = smf.logit("y ~ z + age + C(sex)", data=g).fit(disp=0)
        out[m] = {"auroc_all": roc_auc_score(g.y, g[m]), "auroc_age_ge65": roc_auc_score(old.y, old[m]),
                  "n_age_ge65": [int(old.y.sum()), int((1 - old.y).sum())],
                  "OR_per_SD_adjusted_age_sex": float(np.exp(fit_.params["z"])),
                  "OR_CI95": np.exp(fit_.conf_int().loc["z"]).tolist(), "p_adjusted": float(fit_.pvalues["z"])}
    (OUT / "external_test.json").write_text(json.dumps(out, indent=2))
    print("EXTERNAL GSE33000\n", pd.DataFrame(out).T.round(4).to_string())


def allen_tests(sc, meta, models) -> None:
    a = meta[meta.role == "transfer"].join(sc)
    rows, tbi = [], {}
    for m in models:
        a["s"] = (a[m] - a[m].mean()) / a[m].std()  # SD units across Allen samples
        # 1) Sanity: clinical AD vs control, and neuropathology correlations.
        for reg, g in list(a.groupby("region")) + [("all", a)]:
            k = g.diagnosis.isin(["AD", "control"])
            rows.append({"model": m, "region": reg,
                         "auroc_clinical_AD": roc_auc_score(g.diagnosis[k] == "AD", g.s[k]),
                         "rho_braak": stats.spearmanr(g.s, g.braak).statistic,
                         "rho_cerad": stats.spearmanr(g.s, g.cerad).statistic,
                         "rho_nia_reagan": stats.spearmanr(g.s, g.nia_reagan).statistic})
        # 2) TBI: mixed model with donor random intercept.
        d = a.dropna(subset=["age", "rin"]).copy()
        mm = smf.mixedlm("s ~ tbi + age + C(sex) + dementia + rin + C(region)", d, groups=d.donor_id).fit()
        mm_int = smf.mixedlm("s ~ tbi * dementia + age + C(sex) + rin + C(region)", d, groups=d.donor_id).fit()
        mm_noadj = smf.mixedlm("s ~ tbi + age + C(sex) + rin + C(region)", d, groups=d.donor_id).fit()
        d["tbi_dose"] = d.num_tbi_w_loc.clip(upper=2)
        mm_dose = smf.mixedlm("s ~ tbi_dose + age + C(sex) + dementia + rin + C(region)", d, groups=d.donor_id).fit()
        res = {"mixed_tbi_beta_SD": mm.params["tbi"], "mixed_tbi_CI95": mm.conf_int().loc["tbi"].tolist(),
               "mixed_tbi_p": mm.pvalues["tbi"],
               "mixed_dementia_beta_SD": mm.params["dementia"], "mixed_dementia_p": mm.pvalues["dementia"],
               "unadjusted_for_dementia_tbi_beta": mm_noadj.params["tbi"], "unadjusted_tbi_p": mm_noadj.pvalues["tbi"],
               "interaction_tbi_x_dementia_beta": mm_int.params["tbi:dementia"],
               "interaction_p": mm_int.pvalues["tbi:dementia"],
               "dose_beta_per_tbi": mm_dose.params["tbi_dose"], "dose_p": mm_dose.pvalues["tbi_dose"]}
        # 3) Matched pairs: donor-mean score, TBI minus matched non-TBI.
        dm = a.groupby("donor_id").agg(s=("s", "mean"), tbi=("tbi", "first"), cs=("control_set", "first"),
                                       dementia=("dementia", "first"))
        diffs = []
        for _, g in dm.groupby("cs"):
            if len(g) == 2 and g.tbi.nunique() == 2:
                diffs.append(g.loc[g.tbi == 1, "s"].iloc[0] - g.loc[g.tbi == 0, "s"].iloc[0])
        diffs = np.array(diffs)
        rng = np.random.default_rng(42)
        boot = [rng.choice(diffs, len(diffs)).mean() for _ in range(5000)]
        res.update({"pairs": len(diffs), "pair_mean_diff_SD": diffs.mean(),
                    "pair_diff_CI95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    "pair_wilcoxon_p": stats.wilcoxon(diffs).pvalue,
                    "frac_pairs_TBI_higher": float((diffs > 0).mean())})
        # 4) Per region, donor-level Cohen's d (TBI - no TBI), unadjusted, for the plot.
        for reg, g in a.groupby("region"):
            x1, x0 = g.s[g.tbi == 1], g.s[g.tbi == 0]
            res[f"d_{reg}"] = (x1.mean() - x0.mean()) / np.sqrt((x1.var() + x0.var()) / 2)
            res[f"p_mw_{reg}"] = stats.mannwhitneyu(x1, x0).pvalue
        tbi[m] = {k: (float(v) if np.isscalar(v) else v) for k, v in res.items()}
    pd.DataFrame(rows).to_csv(OUT / "allen_sanity.csv", index=False)
    (OUT / "tbi_tests.json").write_text(json.dumps(tbi, indent=2))
    print("ALLEN SANITY\n", pd.DataFrame(rows).round(3).to_string(index=False))
    keys = ["mixed_tbi_beta_SD", "mixed_tbi_CI95", "mixed_tbi_p", "mixed_dementia_beta_SD", "unadjusted_tbi_p",
            "interaction_p", "dose_p", "pair_mean_diff_SD", "pair_diff_CI95", "pair_wilcoxon_p", "frac_pairs_TBI_higher"]
    print("TBI TESTS\n", pd.DataFrame(tbi).T[keys].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "score"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    a = ap.parse_args()
    fit(a.models) if a.cmd == "fit" else score()
