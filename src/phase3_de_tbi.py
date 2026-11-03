"""Phase 3b: TBI vs no TBI (and dementia vs none) in the Allen cohort with DESeq2.

One DESeq2 model per region, because the same donors give several regions and DESeq2 has no
random effects. Raw RSEM expected counts. Genes: CPM >= 1 in >= 25% of that region's samples.

Models:
  adjusted   ~ age + rin + sex + batch + dementia + tbi   (main, as specified in the brief)
  unadjusted ~ age + rin + sex + batch + tbi              (sensitivity: dementia could be a mediator)
The dementia coefficient of the adjusted model gives a within-Allen dementia signature.

Outputs: results/phase3/de_tbi_<model>_<region>.csv, de_dementia_<region>.csv, de_tbi_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {
    "adjusted": "~ age + rin + sex + batch + dementia + tbi",
    "unadjusted": "~ age + rin + sex + batch + tbi",
}


def run_region(counts: pd.DataFrame, meta: pd.DataFrame, region: str, summary: dict) -> None:
    m = meta[meta.region == region].copy()
    c = counts[m.index]
    cpm = c / c.sum() * 1e6
    c = c[(cpm >= 1).mean(axis=1) >= 0.25]
    md = pd.DataFrame({
        "age": (m.age - m.age.mean()) / m.age.std(),
        "rin": (m.rin - m.rin.mean()) / m.rin.std(),
        "sex": m.sex.astype(str),
        "batch": m.batch.astype(str),
        "dementia": np.where(m.dementia == 1, "Dem", "NoDem"),
        "tbi": np.where(m.tbi == 1, "Y", "N"),
    }, index=m.index)
    for name, design in MODELS.items():
        dds = DeseqDataSet(counts=c.T, metadata=md, design=design, quiet=True, n_cpus=6)
        dds.deseq2()
        targets = [("tbi", "Y", "N", f"de_tbi_{name}_{region}.csv")]
        if name == "adjusted":
            targets.append(("dementia", "Dem", "NoDem", f"de_dementia_{region}.csv"))
        for var, lvl, ref, fname in targets:
            st = DeseqStats(dds, contrast=[var, lvl, ref], quiet=True, n_cpus=6)
            st.summary()
            r = st.results_df.rename(columns={"log2FoldChange": "logFC", "pvalue": "p", "padj": "fdr"})
            r.to_csv(OUT / fname, index_label="gene")
            key = fname[:-4]
            summary[key] = {"genes": int(len(r)), "n": int(len(m)),
                            "fdr05": int((r.fdr < 0.05).sum()), "fdr10": int((r.fdr < 0.10).sum()),
                            "p001": int((r.p < 0.001).sum()), "p01": int((r.p < 0.01).sum()),
                            "expected_p01_by_chance": round(0.01 * r.p.notna().sum(), 1)}
            print(key, summary[key], flush=True)


def main() -> None:
    counts = pd.read_parquet(PROC / "allen_counts.parquet")
    meta = pd.read_csv(PROC / "samples_GSE104687.csv").set_index("sample_id")
    regions = sys.argv[1:] or ["TCx", "PCx", "HIP", "FWM"]
    path = OUT / "de_tbi_summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for reg in regions:
        run_region(counts, meta, reg, summary)
        path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
