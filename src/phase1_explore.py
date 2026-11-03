"""Phase 1: class balance and confounder checks. Writes to results/phase1/."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)


def two_group_tests(df: pd.DataFrame, group: str, a, b, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Mann-Whitney for numeric, chi-square for categorical. One row per variable."""
    rows = []
    ga, gb = df[df[group] == a], df[df[group] == b]
    for v in numeric:
        x, y = ga[v].dropna(), gb[v].dropna()
        if len(x) > 1 and len(y) > 1:
            p = stats.mannwhitneyu(x, y).pvalue
            rows.append({"variable": v, "test": "Mann-Whitney", f"{a}": f"{x.median():.1f} (n={len(x)})",
                         f"{b}": f"{y.median():.1f} (n={len(y)})", "p": p})
    for v in categorical:
        tab = pd.crosstab(df[group], df[v])
        if tab.shape[1] > 1 and tab.shape[0] == 2:
            p = stats.chi2_contingency(tab)[1]
            fmt = lambda g: ", ".join(f"{k}:{n}" for k, n in tab.loc[g].items())
            rows.append({"variable": v, "test": "chi-square", f"{a}": fmt(a), f"{b}": fmt(b), "p": p})
    return pd.DataFrame(rows)


def main() -> None:
    allt = pd.read_csv(PROC / "samples_all.csv")
    summary: dict = {}

    # 1. Class balance per dataset and region (samples and donors).
    bal = (allt.groupby(["dataset", "region", "diagnosis"])
           .agg(samples=("sample_id", "size"), donors=("donor_id", "nunique")).reset_index())
    bal.to_csv(OUT / "class_balance_by_region.csv", index=False)
    print(bal.pivot_table(index=["dataset", "region"], columns="diagnosis", values="samples", fill_value=0))

    # 2. AD-training datasets: AD vs control confounders, one row per donor.
    conf = []
    for ds in ["GSE33000", "GSE132903", "GSE48350", "GSE5281"]:
        t = allt[(allt.dataset == ds) & allt.diagnosis.isin(["AD", "control"])]
        if ds == "GSE48350":
            t = t[t.age >= 60]  # drop young controls (20-59 y) that have no AD counterpart
        donors = t.drop_duplicates("donor_id")
        r = two_group_tests(donors, "diagnosis", "AD", "control", ["age"], ["sex"])
        r.insert(0, "dataset", ds)
        conf.append(r)
    conf = pd.concat(conf)
    conf.to_csv(OUT / "confounders_AD_vs_control.csv", index=False)
    print("\nAD vs control confounders (donor level):\n", conf.to_string(index=False))

    # 3. Allen cohort: TBI vs no TBI, donor level.
    al = pd.read_csv(PROC / "samples_GSE104687.csv")
    don = al.drop_duplicates("donor_id").copy()
    don["tbi_lbl"] = don.tbi.map({1: "TBI", 0: "noTBI"})
    tbi = two_group_tests(don, "tbi_lbl", "TBI", "noTBI",
                          ["age", "braak", "cerad", "nia_reagan", "education_years"],
                          ["sex", "dementia", "apoe4", "diagnosis", "race"])
    tbi.to_csv(OUT / "allen_confounders_TBI_vs_noTBI.csv", index=False)
    print("\nAllen TBI vs no TBI (donor level):\n", tbi.to_string(index=False))
    # Sample-level: RIN differs by region and is a known RNA-seq confounder.
    rin = two_group_tests(al.assign(tbi_lbl=al.tbi.map({1: "TBI", 0: "noTBI"})), "tbi_lbl", "TBI", "noTBI",
                          ["rin"], ["region", "batch"])
    rin.to_csv(OUT / "allen_sample_confounders.csv", index=False)
    print("\nAllen sample-level:\n", rin.to_string(index=False))

    xt = pd.crosstab([don.tbi_lbl], [don.diagnosis], margins=True)
    xt.to_csv(OUT / "allen_tbi_x_diagnosis_donors.csv")
    print("\nAllen donors, TBI x diagnosis:\n", xt)
    reg = pd.crosstab([al.region], [al.tbi.map({1: "TBI", 0: "noTBI"}), al.dementia.map({1: "Dem", 0: "NoDem"})])
    reg.to_csv(OUT / "allen_region_x_tbi_x_dementia_samples.csv")
    print("\nAllen samples per region:\n", reg)
    print("\nAllen matched pairs: control_set values per donor:", don.control_set.value_counts().value_counts().to_dict())

    # 4. Figures.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    train = allt[allt.diagnosis.isin(["AD", "control"]) & (allt.dataset != "GSE104687")]
    train = train[~((train.dataset == "GSE48350") & (train.age < 60))].drop_duplicates("donor_id")
    sns.boxplot(data=train, x="dataset", y="age", hue="diagnosis", ax=axes[0])
    axes[0].set_title("Age at death, AD vs control (donors)")
    sns.boxplot(data=don, x="tbi_lbl", y="age", hue="dementia", ax=axes[1])
    axes[1].set_title("Allen: age by TBI and dementia (donors)")
    sns.countplot(data=don, x="braak", hue="tbi_lbl", ax=axes[2])
    axes[2].set_title("Allen: Braak stage by TBI (donors)")
    fig.tight_layout()
    fig.savefig(OUT / "confounders_overview.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cnt = allt[allt.diagnosis.isin(["AD", "control"])].groupby(["dataset", "region_group", "diagnosis"]).size().unstack(fill_value=0)
    cnt.plot.barh(stacked=True, ax=ax, color=["#c0392b", "#7f8c8d"])
    ax.set_xlabel("samples"); ax.set_title("AD and control samples by dataset and region")
    fig.tight_layout()
    fig.savefig(OUT / "class_balance.png", dpi=150)

    summary["n_samples"] = allt.groupby("dataset").size().to_dict()
    summary["n_donors"] = allt.groupby("dataset").donor_id.nunique().to_dict()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
