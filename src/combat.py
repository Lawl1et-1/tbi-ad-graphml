"""ComBat batch correction (Johnson, Li & Rabinovic 2007, parametric empirical Bayes).

Written here because the `combat` PyPI package (0.3.3) returned data with gene variances
shrunk ~50x and unequal across batches (see results/phase2/PHASE2_REPORT.md).
Tested in tests/test_combat.py against simulated batch effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _it_sol(s_data, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4, max_iter=500):
    n = (~np.isnan(s_data)).sum(axis=1)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((s_data - g_new[:, None]) ** 2).sum(axis=1)
        d_new = _postvar(sum2, n, a, b)
        change = max(np.max(np.abs(g_new - g_old) / np.abs(g_old).clip(1e-12)),
                     np.max(np.abs(d_new - d_old) / d_old.clip(1e-12)))
        g_old, d_old = g_new, d_new
        if change < conv:
            break
    return g_new, d_new


def combat(data: pd.DataFrame, batch, covariates: pd.DataFrame | None = None) -> pd.DataFrame:
    """data: samples x genes. batch: one label per sample. covariates: optional design to protect.

    Returns batch-corrected data on the original scale (samples x genes).
    """
    X = data.values.T.astype(float)  # genes x samples
    batch = pd.Series(np.asarray(batch), index=data.index)
    levels = batch.unique()
    B = pd.get_dummies(batch).reindex(columns=levels).values.astype(float)  # samples x batches
    n_batch = B.sum(axis=0)
    if (n_batch < 2).any():
        raise ValueError("every batch needs >= 2 samples")
    design = B if covariates is None else np.hstack([B, covariates.values.astype(float)])
    n_samples = X.shape[1]

    # Standardize: fit gene-wise linear model, pooled variance.
    beta = np.linalg.lstsq(design, X.T, rcond=None)[0]  # p x genes
    grand_mean = (n_batch / n_samples) @ beta[: len(levels)]
    var_pooled = ((X - (design @ beta).T) ** 2).mean(axis=1)
    var_pooled[var_pooled == 0] = np.median(var_pooled)
    stand_mean = np.outer(grand_mean, np.ones(n_samples))
    if covariates is not None:
        stand_mean += (covariates.values.astype(float) @ beta[len(levels):]).T
    s = (X - stand_mean) / np.sqrt(var_pooled)[:, None]

    # Batch effect estimates and empirical Bayes priors.
    gamma_hat = np.linalg.lstsq(B, s.T, rcond=None)[0]  # batches x genes
    delta_hat = np.vstack([s[:, B[:, i] == 1].var(axis=1, ddof=1) for i in range(len(levels))])
    gamma_bar, t2 = gamma_hat.mean(axis=1), gamma_hat.var(axis=1, ddof=1)
    m, v = delta_hat.mean(axis=1), delta_hat.var(axis=1, ddof=1)
    a_prior, b_prior = (2 * v + m ** 2) / v, (m * v + m ** 3) / v

    gamma_star, delta_star = np.zeros_like(gamma_hat), np.zeros_like(delta_hat)
    for i in range(len(levels)):
        cols = B[:, i] == 1
        gamma_star[i], delta_star[i] = _it_sol(s[:, cols], gamma_hat[i], delta_hat[i],
                                              gamma_bar[i], t2[i], a_prior[i], b_prior[i])

    adj = s.copy()
    for i in range(len(levels)):
        cols = B[:, i] == 1
        adj[:, cols] = (s[:, cols] - gamma_star[i][:, None]) / np.sqrt(delta_star[i])[:, None]
    out = adj * np.sqrt(var_pooled)[:, None] + stand_mean
    return pd.DataFrame(out.T, index=data.index, columns=data.columns)
