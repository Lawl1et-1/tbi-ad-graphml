"""Phase 3d: pathway and cell-type enrichment.

- AD signature (up / down separately): over-representation (gseapy.enrich, offline GMT files),
  background = the 11,027 genes tested.
- TBI (adjusted model, each region): preranked GSEA on the DESeq2 Wald statistic.
- TBI x AD: GSEA of the TBI ranking against the AD-up / AD-down gene sets themselves.
Libraries: GO BP 2023, KEGG 2021, Reactome 2022, PanglaoDB cell-type markers 2021.
Gene-permutation GSEA p-values ignore gene-gene correlation; treat them as descriptive.
"""
from __future__ import annotations

from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
GS_DIR = ROOT / "data" / "raw" / "genesets"
OUT = ROOT / "results" / "phase3"
LIBS = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022", "PanglaoDB_Augmented_2021"]
REGIONS = ["TCx", "PCx", "HIP", "FWM"]


def read_gmt(name: str) -> dict[str, list[str]]:
    sets = {}
    for line in (GS_DIR / f"{name}.gmt").read_text(encoding="utf-8").splitlines():
        parts = [p for p in line.split("\t") if p]
        sets[parts[0]] = [g.split(",")[0] for g in parts[1:]]
    return sets


def main() -> None:
    libs = {n: read_gmt(n) for n in LIBS}
    meta = pd.read_csv(OUT / "de_ad_meta.csv", index_col=0)
    sig = meta[(meta.fdr_meta < 0.05) & (meta.n_same_sign == 3)]
    background = meta.index.tolist()

    ora = []
    for direction, genes in [("AD_up", sig.index[sig.z_meta > 0]), ("AD_down", sig.index[sig.z_meta < 0])]:
        for name, gs in libs.items():
            res = gp.enrich(gene_list=list(genes), gene_sets=gs, background=background, no_plot=True).results
            res.insert(0, "library", name)
            res.insert(0, "list", direction)
            ora.append(res)
    ora = pd.concat(ora)
    ora.to_csv(OUT / "enrich_ad_ora.csv", index=False)
    top = ora[ora["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value").groupby(["list", "library"]).head(5)
    print(top[["list", "library", "Term", "Overlap", "Adjusted P-value"]].to_string(index=False))

    # AD-up / AD-down as gene sets, capped at the 300 strongest genes each.
    ad_sets = {"AD_up_top300": sig[sig.z_meta > 0].index[:300].tolist(),
               "AD_down_top300": sig[sig.z_meta < 0].index[:300].tolist()}
    gsea_rows, ad_rows = [], []
    for reg in REGIONS:
        r = pd.read_csv(OUT / f"de_tbi_adjusted_{reg}.csv", index_col=0).dropna(subset=["stat"])
        rnk = r["stat"].sort_values(ascending=False)
        for name, gs in {**libs, "AD_signature": ad_sets}.items():
            pre = gp.prerank(rnk=rnk, gene_sets=gs, min_size=15, max_size=500, permutation_num=1000,
                             seed=SEED, threads=6, no_plot=True, outdir=None, verbose=False)
            res = pre.res2d.copy()
            res.insert(0, "library", name)
            res.insert(0, "region", reg)
            (ad_rows if name == "AD_signature" else gsea_rows).append(res)
        print("GSEA done", reg, flush=True)
    gsea = pd.concat(gsea_rows)
    gsea.to_csv(OUT / "gsea_tbi_by_region.csv", index=False)
    ad = pd.concat(ad_rows)
    ad.to_csv(OUT / "gsea_tbi_vs_ad_sets.csv", index=False)
    print(ad[["region", "Term", "NES", "NOM p-val", "FDR q-val"]].to_string(index=False))
    for c in ["FDR q-val", "NES"]:
        gsea[c] = pd.to_numeric(gsea[c])
    best = gsea[gsea["FDR q-val"] < 0.05].sort_values("FDR q-val").groupby(["region", "library"]).head(4)
    print(best[["region", "library", "Term", "NES", "FDR q-val"]].to_string(index=False))


if __name__ == "__main__":
    main()
