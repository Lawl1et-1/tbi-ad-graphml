"""Phase 3 figures (slide-ready)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase3"
REGIONS = ["TCx", "PCx", "HIP", "FWM"]
NAMES = {"TCx": "Temporal cortex", "PCx": "Parietal cortex", "HIP": "Hippocampus", "FWM": "White matter"}


def volcano() -> None:
    m = pd.read_csv(RES / "de_ad_meta.csv", index_col=0)
    sig = (m.fdr_meta < 0.05) & (m.n_same_sign == 3)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(m.logFC_mean[~sig], -np.log10(m.p_meta[~sig]), s=3, c="#bdc3c7", label="not significant")
    up, dn = sig & (m.z_meta > 0), sig & (m.z_meta < 0)
    ax.scatter(m.logFC_mean[up], -np.log10(m.p_meta[up]), s=3, c="#c0392b", label=f"up in AD ({up.sum()})")
    ax.scatter(m.logFC_mean[dn], -np.log10(m.p_meta[dn]), s=3, c="#2980b9", label=f"down in AD ({dn.sum()})")
    for g in m.index[:12]:
        ax.annotate(g, (m.logFC_mean[g], -np.log10(m.p_meta[g])), fontsize=7)
    ax.set_xlabel("mean log2 fold change (AD vs control)"); ax.set_ylabel("-log10 meta p")
    ax.set_title("AD signature: meta-analysis of 3 training cohorts\n(FDR < 0.05 and same sign in all 3)")
    ax.legend(fontsize=8, markerscale=3)
    fig.tight_layout(); fig.savefig(RES / "fig_ad_volcano.png", dpi=150); plt.close(fig)


def tbi_vs_ad() -> None:
    m = pd.read_csv(RES / "de_ad_meta.csv", index_col=0)
    ov = pd.read_csv(RES / "overlap_summary.csv")
    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    for i, reg in enumerate(REGIONS):
        r = pd.read_csv(RES / f"de_tbi_adjusted_{reg}.csv", index_col=0).dropna(subset=["stat"])
        sh = r.index.intersection(m.index)
        a = ax[0, i]
        a.hexbin(m.loc[sh, "z_meta"], r.loc[sh, "stat"], gridsize=60, bins="log", cmap="viridis", mincnt=1)
        row = ov[(ov.region == reg) & (ov.signature == "tbi_adjusted")].iloc[0]
        a.set_title(f"{NAMES[reg]}\nSpearman ρ = {row.spearman_AD_vs_stat:.2f}, permutation p = {row.perm_p_two_sided:.2f}", fontsize=10)
        a.set_xlabel("AD meta z"); a.set_ylabel("TBI Wald statistic (DESeq2)")
        a.axhline(0, c="w", lw=0.5); a.axvline(0, c="w", lw=0.5)
        null = np.load(RES / f"overlap_perm_null_tbi_adjusted_{reg}.npy")
        b = ax[1, i]
        b.hist(null, bins=40, color="#95a5a6")
        b.axvline(row.spearman_AD_vs_OLS_t, c="#c0392b", lw=2, label="observed")
        b.set_xlabel("ρ(AD z, TBI t) under matched-pair label swaps"); b.set_ylabel("permutations")
        b.legend(fontsize=8)
    fig.suptitle("Does the TBI signature resemble the AD signature? (adjusted for age, sex, RIN, batch, dementia)")
    fig.tight_layout(); fig.savefig(RES / "fig_tbi_vs_ad.png", dpi=150); plt.close(fig)


def naive_vs_perm() -> None:
    ov = pd.read_csv(RES / "overlap_summary.csv")
    t = ov[ov.signature == "tbi_adjusted"].set_index("region").loc[REGIONS]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(REGIONS))
    ax.bar(x - 0.2, -np.log10(t.p_hypergeom), 0.4, label="naive hypergeometric test", color="#e67e22")
    ax.bar(x + 0.2, -np.log10(t.perm_p_two_sided), 0.4, label="matched-pair permutation", color="#2c3e50")
    ax.axhline(-np.log10(0.05), ls="--", c="k", lw=1, label="p = 0.05")
    ax.set_xticks(x, [NAMES[r] for r in REGIONS]); ax.set_ylabel("-log10 p")
    ax.set_title("AD–TBI overlap: gene-level tests overstate the evidence")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "fig_naive_vs_permutation.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    volcano(); tbi_vs_ad(); naive_vs_perm()
    print("figures written")
