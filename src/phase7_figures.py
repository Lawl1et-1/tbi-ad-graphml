"""Phase 7: slide-ready figures in results/figures/ (copies earlier key figures too)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = R / "figures"
FIG.mkdir(exist_ok=True)
PROC = ROOT / "data" / "processed"
NAMES = {"lr_l2": "LR L2", "lr_l1": "LR L1", "rf": "Random forest", "mlp": "MLP", "gat_string": "GAT (STRING)",
         "gat_none": "GAT (no edges)", "gat_random": "GAT (random graph)", "gat_coexpr": "GAT (co-expression)", "gat": "GAT (STRING)"}
REG = {"TCx": "Temporal cortex", "PCx": "Parietal cortex", "HIP": "Hippocampus", "FWM": "White matter"}
plt.rcParams.update({"font.size": 10})


def performance() -> None:
    cv = pd.read_csv(R / "phase5" / "cv_metrics.csv")
    lo = pd.read_csv(R / "phase5" / "lodo.csv").groupby("model").auroc.mean()
    cv["lo"] = cv.auroc_ens_ci95.str.strip("[]").str.split(",").str[0].astype(float)
    cv["hi"] = cv.auroc_ens_ci95.str.strip("[]").str.split(",").str[1].astype(float)
    cv = cv.sort_values("auroc_mean")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    col = ["#8e44ad" if m.startswith("gat") else "#2c3e50" for m in cv.model]
    ax[0].barh([NAMES[m] for m in cv.model], cv.auroc_mean, xerr=[cv.auroc_mean - cv.lo, cv.hi - cv.auroc_mean],
               color=col, capsize=3)
    ax[0].set_xlim(0.75, 1); ax[0].set_xlabel("AUROC (donor-level 5-fold CV, 95% CI)")
    ax[0].set_title("AD vs control, pooled training cohorts")
    lo = lo.reindex(["lr_l2", "lr_l1", "rf", "mlp", "gat"])
    ax[1].bar([NAMES[m] for m in lo.index], lo.values, color=["#2c3e50"] * 4 + ["#8e44ad"])
    ax[1].set_ylim(0.5, 1); ax[1].set_ylabel("mean AUROC on held-out cohort")
    ax[1].set_title("Leave-one-cohort-out"); ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(FIG / "fig_model_performance.png", dpi=170); plt.close(fig)


def score_distributions() -> None:
    sc = pd.read_csv(R / "phase6" / "scores.csv", index_col=0)
    a = sc[sc.role == "transfer"].copy()
    tests = json.loads((R / "phase6" / "tbi_tests.json").read_text())
    rob = json.loads((R / "phase6" / "robustness.json").read_text())
    for m in ["lr_l2", "gat"]:
        a["score"] = (a[m] - a[m].mean()) / a[m].std()
        a["TBI"] = np.where(a.tbi == 1, "TBI", "no TBI")
        a["Region"] = a.region.map(REG)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2, 1]})
        sns.violinplot(data=a, x="Region", y="score", hue="TBI", split=True, inner="quart", ax=ax[0],
                       palette={"TBI": "#c0392b", "no TBI": "#95a5a6"}, order=list(REG.values()))
        ax[0].set_ylabel("AD-likeness score (SD)"); ax[0].set_xlabel("")
        t = tests[m]
        ax[0].set_title(f"{NAMES[m]}: TBI effect {t['mixed_tbi_beta_SD']:+.2f} SD "
                        f"[{t['mixed_tbi_CI95'][0]:.2f}, {t['mixed_tbi_CI95'][1]:.2f}], mixed model p = {t['mixed_tbi_p']:.3f}")
        # Matched pairs, donor-mean score.
        d = a.groupby("donor_id").agg(s=("score", "mean"), tbi=("tbi", "first"), cs=("control_set", "first"))
        pairs = [(g.loc[g.tbi == 0, "s"].iloc[0], g.loc[g.tbi == 1, "s"].iloc[0])
                 for _, g in d.groupby("cs") if len(g) == 2 and g.tbi.nunique() == 2]
        for x0, x1 in pairs:
            ax[1].plot([0, 1], [x0, x1], color="#c0392b" if x1 > x0 else "#7f8c8d", alpha=0.5, lw=1)
        ax[1].set_xticks([0, 1], ["matched no TBI", "TBI"]); ax[1].set_xlim(-0.3, 1.3)
        ax[1].set_ylabel("donor mean score (SD)")
        ax[1].set_title(f"{len(pairs)} matched pairs: mean diff {rob[m]['pair_mean_diff_SD']:+.2f} SD\n"
                        f"sign-flip p = {rob[m]['signflip_p']:.3f}; {t['frac_pairs_TBI_higher']:.0%} pairs higher in TBI")
        fig.tight_layout(); fig.savefig(FIG / f"fig_tbi_scores_{m}.png", dpi=170); plt.close(fig)


def importance_network() -> None:
    """Stable LR-L2 genes (top 200 |coef| in >= 4/5 models) on the STRING graph.
    LR is used because the GAT explanations were not stable across models (Phase 6 report)."""
    st = pd.read_csv(R / "phase6" / "lr_stable_genes.csv", index_col=0)
    e = pd.read_csv(PROC / "graph_string.csv")
    G = nx.Graph(); G.add_edges_from(e[["a", "b"]].itertuples(index=False))
    # Stable genes plus STRING partners that link at least two of them (to show modules).
    core = set(g for g in st.index if g in G)
    linkers = {n for n in G if n not in core and sum(1 for m in G[n] if m in core) >= 2}
    H = G.subgraph(core | linkers).copy()
    H.remove_nodes_from([n for n in list(H) if H.degree[n] == 0])
    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(H, seed=42, k=0.4)
    w = st.coef_mean.abs() / st.coef_mean.abs().max()
    nodes_core = [n for n in H if n in core]
    nodes_link = [n for n in H if n not in core]
    nx.draw_networkx_edges(H, pos, alpha=0.25, ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=nodes_link, node_size=40, node_color="#dddddd", ax=ax)
    sc = nx.draw_networkx_nodes(H, pos, nodelist=nodes_core, node_size=[120 + 900 * w[n] for n in nodes_core],
                                node_color=[st.coef_mean[n] for n in nodes_core], cmap="coolwarm",
                                vmin=-st.coef_mean.abs().max(), vmax=st.coef_mean.abs().max(), ax=ax)
    nx.draw_networkx_labels(H, pos, {n: n for n in nodes_core}, font_size=7, ax=ax)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="LR coefficient (red = pushes toward AD)")
    ax.set_title(f"Stable AD-score genes (LR L2, top 200 in >=4/5 models): {len(nodes_core)} of {len(st)} in STRING,\n"
                 f"grey = STRING partners linking >= 2 of them; node size = |coefficient|")
    ax.set_axis_off()
    fig.tight_layout(); fig.savefig(FIG / "fig_importance_network.png", dpi=170); plt.close(fig)


def copy_existing() -> None:
    for src in [R / "phase2" / "pca_before_combat.png", R / "phase2" / "pca_after_combat.png",
                R / "phase3" / "fig_ad_volcano.png", R / "phase3" / "fig_tbi_vs_ad.png",
                R / "phase3" / "fig_naive_vs_permutation.png", R / "phase6" / "fig_random_signature_null.png",
                R / "phase1" / "confounders_overview.png"]:
        if src.exists():
            shutil.copy(src, FIG / src.name)


if __name__ == "__main__":
    performance(); score_distributions(); copy_existing()
    if (R / "phase6" / "lr_stable_genes.csv").exists():
        importance_network()
    print(sorted(p.name for p in FIG.glob("*.png")))
