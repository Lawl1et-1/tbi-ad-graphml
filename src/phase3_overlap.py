"""Phase 3c: does the TBI expression signature overlap the AD signature?

For each Allen region and TBI model (adjusted / unadjusted for dementia):
1. Hypergeometric test: AD signature genes vs TBI nominal p < 0.01 genes (TBI has no FDR hits),
   plus direction agreement inside the overlap.
2. Spearman correlation between the AD meta z-score and the TBI Wald statistic over shared genes.
3. Permutation null for (2): TBI labels swapped inside the matched donor pairs (`control_set`),
   1,000 times, with a fast OLS on log-CPM using the same covariates. This respects the matching and
   the gene-gene correlation that make naive gene-level p-values too small.
4. Positive control: same tests for the within-Allen dementia signature.

Outputs: results/phase3/overlap_summary.csv, overlap_perm_null_<region>.npy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_PERM = 1000
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase3"
REGIONS = ["TCx", "PCx", "HIP", "FWM"]


def ad_signature() -> tuple[pd.DataFrame, set, set]:
    meta = pd.read_csv(OUT / "de_ad_meta.csv", index_col=0)
    sig = meta[(meta.fdr_meta < 0.05) & (meta.n_same_sign == 3)]
    return meta, set(sig.index[sig.z_meta > 0]), set(sig.index[sig.z_meta < 0])


def ols_t(Y: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ X.T @ Y
    E = Y - X @ B
    s2 = (E ** 2).sum(axis=0) / (X.shape[0] - X.shape[1])
    return B[k] / np.sqrt(s2 * XtX_inv[k, k])


def region_design(m: pd.DataFrame, tbi: np.ndarray, with_dementia: bool) -> np.ndarray:
    cols = [np.ones(len(m)), tbi,
            ((m.age - m.age.mean()) / m.age.std()).values,
            ((m.rin - m.rin.mean()) / m.rin.std()).values,
            (m.sex == "M").astype(float).values]
    if with_dementia:
        cols.append(m.dementia.astype(float).values)
    D = np.column_stack(cols)
    return np.hstack([D, pd.get_dummies(m.batch, drop_first=True).astype(float).values])


def pair_swaps(m: pd.DataFrame, rng: np.random.Generator, n: int) -> np.ndarray:
    """n permuted TBI vectors: inside each complete matched pair, swap labels with prob 0.5."""
    tbi = m.tbi.values.astype(float)
    pos = pd.Series(range(len(m)), index=m.index)
    pairs = [g.index for _, g in m.groupby("control_set") if len(g) == 2 and g.tbi.nunique() == 2]
    perms = np.tile(tbi, (n, 1))
    for idx in pairs:
        i, j = pos[idx[0]], pos[idx[1]]
        flip = rng.random(n) < 0.5
        perms[flip, i], perms[flip, j] = tbi[j], tbi[i]
    return perms, len(pairs)


def hypergeom(a: set, b: set, universe: set) -> tuple[int, float, float]:
    a, b = a & universe, b & universe
    k = len(a & b)
    p = stats.hypergeom.sf(k - 1, len(universe), len(a), len(b))
    expected = len(a) * len(b) / len(universe)
    return k, k / expected if expected else np.nan, p


def main() -> None:
    rng = np.random.default_rng(SEED)
    meta, ad_up, ad_dn = ad_signature()
    ad_all = ad_up | ad_dn
    counts = pd.read_parquet(PROC / "allen_counts.parquet")
    samples = pd.read_csv(PROC / "samples_GSE104687.csv").set_index("sample_id")
    rows = []
    for reg in REGIONS:
        m = samples[samples.region == reg]
        c = counts[m.index]
        cpm = c / c.sum() * 1e6
        c = c[(cpm >= 1).mean(axis=1) >= 0.25]
        shared = sorted(set(c.index) & set(meta.index))
        logcpm = np.log2(cpm.loc[shared] + 1).T.values
        z_ad = meta.loc[shared, "z_meta"].values
        perms, n_pairs = pair_swaps(m, rng, N_PERM)

        for kind, fname, with_dem in [("tbi_adjusted", f"de_tbi_adjusted_{reg}.csv", True),
                                      ("tbi_unadjusted", f"de_tbi_unadjusted_{reg}.csv", False),
                                      ("dementia", f"de_dementia_{reg}.csv", None)]:
            r = pd.read_csv(OUT / fname, index_col=0).reindex(shared).dropna(subset=["stat", "p"])
            universe = set(r.index)
            hits = set(r.index[r.p < 0.01])
            k, fold, p_h = hypergeom(ad_all, hits, universe)
            ov = sorted(ad_all & hits)
            same = int((np.sign(r.loc[ov, "stat"]) == np.sign(meta.loc[ov, "z_meta"])).sum())
            rho = stats.spearmanr(meta.loc[r.index, "z_meta"], r.stat).statistic
            row = {"region": reg, "signature": kind, "n_samples": len(m), "genes_shared": len(universe),
                   "hits_p01": len(hits), "ad_sig_in_universe": len(ad_all & universe),
                   "overlap": k, "fold_enrichment": round(fold, 2), "p_hypergeom": p_h,
                   "overlap_same_direction": same, "frac_same_direction": round(same / k, 3) if k else np.nan,
                   "spearman_AD_vs_stat": round(rho, 4)}
            if kind.startswith("tbi"):
                # Permutation null on the OLS t-statistic (DESeq2 is too slow to rerun 1,000 times).
                X0 = region_design(m, m.tbi.values.astype(float), with_dem)
                t_obs = ols_t(logcpm, X0, 1)
                rho_ols = stats.spearmanr(z_ad, t_obs).statistic
                null = np.array([stats.spearmanr(z_ad, ols_t(logcpm, region_design(m, pv, with_dem), 1)).statistic
                                 for pv in perms])
                np.save(OUT / f"overlap_perm_null_{kind}_{reg}.npy", null)
                row.update({"spearman_AD_vs_OLS_t": round(rho_ols, 4), "matched_pairs_swapped": n_pairs,
                            "perm_p_two_sided": (np.sum(np.abs(null) >= abs(rho_ols)) + 1) / (N_PERM + 1),
                            "perm_null_sd": round(null.std(), 4),
                            "rho_over_null_sd": round(rho_ols / null.std(), 2)})
            rows.append(row)
            print({k2: row[k2] for k2 in ["region", "signature", "overlap", "fold_enrichment", "p_hypergeom",
                                          "frac_same_direction", "spearman_AD_vs_stat"]
                   + (["perm_p_two_sided", "rho_over_null_sd"] if "perm_p_two_sided" in row else [])}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "overlap_summary.csv", index=False)
    global_test(meta, counts, samples, rng)


def global_test(meta, counts, samples, rng) -> None:
    """Post-hoc global test: mean Spearman(AD z, TBI t) over the 4 regions, with the SAME donor-level
    pair swaps applied in every region (regions share donors, so they are not independent)."""
    donors = samples.drop_duplicates("donor_id").set_index("donor_id")
    pairs = [g.index.tolist() for _, g in donors.groupby("control_set") if len(g) == 2 and g.tbi.nunique() == 2]
    flips = rng.random((N_PERM, len(pairs))) < 0.5
    prepared = []
    for reg in REGIONS:
        m = samples[samples.region == reg]
        c = counts[m.index]
        cpm = c / c.sum() * 1e6
        keep = (cpm >= 1).mean(axis=1) >= 0.25
        shared = sorted(set(c.index[keep]) & set(meta.index))
        prepared.append((m, np.log2(cpm.loc[shared] + 1).T.values, meta.loc[shared, "z_meta"].values))
    out = {}
    for with_dem in [True, False]:
        def stat(flip_row):
            swap = {}
            for (d1, d2), f in zip(pairs, flip_row):
                if f:
                    swap[d1], swap[d2] = d2, d1
            rhos = []
            for m, Y, z in prepared:
                tbi_by_donor = donors.tbi.rename(index=lambda d: d)
                lab = m.donor_id.map(lambda d: tbi_by_donor[swap.get(d, d)]).values.astype(float)
                rhos.append(stats.spearmanr(z, ols_t(Y, region_design(m, lab, with_dem), 1)).statistic)
            return np.mean(rhos), rhos
        obs, obs_r = stat(np.zeros(len(pairs), bool))
        null = np.array([stat(f)[0] for f in flips])
        key = "adjusted" if with_dem else "unadjusted"
        out[key] = {"mean_rho_obs": round(obs, 4), "per_region": dict(zip(REGIONS, np.round(obs_r, 4))),
                    "perm_p_two_sided": (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1),
                    "perm_p_one_sided_positive": (np.sum(null >= obs) + 1) / (N_PERM + 1),
                    "null_sd": round(null.std(), 4), "pairs": len(pairs)}
        np.save(OUT / f"overlap_global_null_{key}.npy", null)
        print("GLOBAL", key, out[key], flush=True)
    import json
    (OUT / "overlap_global_test.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
