"""Phase 6c: is the higher AD-likeness score in TBI donors robust and AD-specific?

1. Exact matched-pair sign-flip permutation of the donor-mean score difference (no model assumptions).
2. Random-signature control: 2,000 random gene-weight vectors scored on Allen. If random scores also
   separate TBI donors, the effect reflects a generic expression shift, not AD resemblance.
   Variant 2b: random directions drawn with the same covariance structure as real expression
   (random combinations of training samples' principal axes), which is a harder control.
3. Same AD-score test after removing the first principal components of the Allen data
   (a rough guard against broad composition / quality axes).
4. Per region and leave-one-region-out.
5. Multiplicity: Bonferroni over the 5 models.
Outputs: results/phase6/robustness.json, fig_random_signature_null.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
P6 = ROOT / "results" / "phase6"
PROC = ROOT / "data" / "processed"
MODELS = ["lr_l2", "lr_l1", "rf", "mlp", "gat"]
N_RAND = 2000
rng = np.random.default_rng(42)


def pair_diff(score: pd.Series, a: pd.DataFrame) -> np.ndarray:
    d = pd.DataFrame({"s": score, "donor": a.donor_id, "tbi": a.tbi, "cs": a.control_set}).groupby("donor").agg(
        s=("s", "mean"), tbi=("tbi", "first"), cs=("cs", "first"))
    out = []
    for _, g in d.groupby("cs"):
        if len(g) == 2 and g.tbi.nunique() == 2:
            out.append(g.loc[g.tbi == 1, "s"].iloc[0] - g.loc[g.tbi == 0, "s"].iloc[0])
    return np.array(out)


def signflip_p(diffs: np.ndarray, n=20000) -> float:
    obs = diffs.mean()
    flips = rng.choice([-1, 1], size=(n, len(diffs)))
    null = (flips * diffs).mean(axis=1)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def std(x):
    return (x - x.mean()) / x.std()


def main() -> None:
    sc = pd.read_csv(P6 / "scores.csv", index_col=0)
    a = sc[sc.role == "transfer"].copy()
    X = pd.read_parquet(PROC / "harmonized_expr.parquet")
    mu, sd = np.load(P6 / "models" / "zscore_mu.npy"), np.load(P6 / "models" / "zscore_sd.npy")
    Za = (X.loc[a.index].values - mu) / sd
    Ztr = (X.loc[sc.index[sc.role == "train"]].values - mu) / sd
    out = {}

    # 1. Exact sign-flip test per model; Bonferroni over 5 models.
    for m in MODELS:
        d = pair_diff(std(a[m]), a)
        p = signflip_p(d)
        out[m] = {"pairs": len(d), "pair_mean_diff_SD": float(d.mean()), "signflip_p": p,
                  "signflip_p_bonferroni5": min(1.0, 5 * p)}

    # 2. Random-signature null on the pair-difference statistic.
    W = rng.normal(size=(Za.shape[1], N_RAND))
    R = Za @ W
    null_iid = np.array([pair_diff(pd.Series(std(R[:, i]), index=a.index), a).mean() for i in range(N_RAND)])
    # 2b. Directions inside the training expression space (covariance-matched).
    U, S, Vt = np.linalg.svd(Ztr - Ztr.mean(0), full_matrices=False)
    k = 50
    Wc = Vt[:k].T @ (rng.normal(size=(k, N_RAND)) * S[:k, None])
    Rc = Za @ Wc
    null_cov = np.array([pair_diff(pd.Series(std(Rc[:, i]), index=a.index), a).mean() for i in range(N_RAND)])
    for m in MODELS:
        obs = out[m]["pair_mean_diff_SD"]
        out[m]["random_iid_percentile"] = float((null_iid < obs).mean() * 100)
        out[m]["random_iid_p_one_sided"] = float((np.sum(null_iid >= obs) + 1) / (N_RAND + 1))
        out[m]["random_cov_percentile"] = float((null_cov < obs).mean() * 100)
        out[m]["random_cov_p_one_sided"] = float((np.sum(null_cov >= obs) + 1) / (N_RAND + 1))
    out["random_null_summary"] = {"iid_mean": float(null_iid.mean()), "iid_sd": float(null_iid.std()),
                                  "cov_mean": float(null_cov.mean()), "cov_sd": float(null_cov.std()),
                                  "cov_frac_positive": float((null_cov > 0).mean())}

    # 3. Remove the top Allen principal components from the expression, then rescore with LR-L2 weights.
    import joblib
    coef = np.mean([joblib.load(P6 / "models" / f"lr_l2_s{s}.joblib").coef_[0] for s in [101, 102, 103, 104, 105]], axis=0)
    Zc = Za - Za.mean(0)
    Ua, Sa, Vta = np.linalg.svd(Zc, full_matrices=False)
    for k in [0, 1, 3, 5, 10]:
        Zr = Zc - (Ua[:, :k] * Sa[:k]) @ Vta[:k] if k else Zc
        s = pd.Series(std(Zr @ coef), index=a.index)
        d = pair_diff(s, a)
        out.setdefault("lr_l2_after_removing_allen_PCs", {})[k] = {
            "pair_mean_diff_SD": float(d.mean()), "signflip_p": signflip_p(d)}
    pc = pd.DataFrame(Ua[:, :5] * Sa[:5], index=a.index, columns=[f"PC{i+1}" for i in range(5)])
    out["allen_PCs_vs_tbi_pairdiff"] = {c: float(pair_diff(std(pc[c]), a).mean()) for c in pc}
    out["allen_PCs_corr_with_lr_l2_score"] = {c: float(np.corrcoef(pc[c], a.lr_l2)[0, 1]) for c in pc}

    # 4. Per region, and leave one region out (mixed model, LR-L2 and GAT).
    for m in ["lr_l2", "gat"]:
        a["s"] = std(a[m])
        reg = {}
        for r in ["TCx", "PCx", "HIP", "FWM"]:
            g = a[a.region == r]
            f = smf.ols("s ~ tbi + age + C(sex) + dementia + rin", g).fit()
            reg[r] = {"beta": float(f.params["tbi"]), "p": float(f.pvalues["tbi"])}
            h = a[a.region != r]
            f2 = smf.mixedlm("s ~ tbi + age + C(sex) + dementia + rin + C(region)", h, groups=h.donor_id).fit()
            reg[f"without_{r}"] = {"beta": float(f2.params["tbi"]), "p": float(f2.pvalues["tbi"])}
        # Only non-demented donors (TBI effect not via dementia).
        nd = a[a.dementia == 0]
        f3 = smf.mixedlm("s ~ tbi + age + C(sex) + rin + C(region)", nd, groups=nd.donor_id).fit()
        reg["non_demented_only"] = {"beta": float(f3.params["tbi"]), "p": float(f3.pvalues["tbi"]),
                                    "donors": int(nd.donor_id.nunique())}
        out[m]["by_region"] = reg

    (P6 / "robustness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_cov, bins=60, color="#95a5a6", alpha=0.8, label="random signatures (covariance-matched)")
    ax.hist(null_iid, bins=60, color="#bdc3c7", alpha=0.6, label="random signatures (independent weights)")
    for m, c in [("lr_l2", "#c0392b"), ("gat", "#8e44ad")]:
        ax.axvline(out[m]["pair_mean_diff_SD"], color=c, lw=2, label=f"{m} AD score")
    ax.set_xlabel("TBI minus matched non-TBI donor (score SD)"); ax.set_ylabel("random signatures")
    ax.set_title("Is the TBI shift specific to the AD signature?")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(P6 / "fig_random_signature_null.png", dpi=150)


if __name__ == "__main__":
    main()
