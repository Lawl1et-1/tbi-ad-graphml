"""Phase 2a: map every dataset to gene symbols and put it on a log2 scale.

Outputs (data/processed/):
  expr_<dataset>.parquet   samples x genes, log2 scale, only samples with role != excluded
  allen_counts.parquet     genes x samples raw RSEM expected counts (for DESeq2 in Phase 3)
  phase2_gene_universe.txt genes present and expressed in all datasets
"""
from __future__ import annotations

import glob
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results" / "phase2"
RES.mkdir(parents=True, exist_ok=True)
LOG: dict = {}


def load_series_matrix(gse: str) -> pd.DataFrame:
    """Probe x sample matrix from a GEO series matrix file."""
    path = glob.glob(str(RAW / gse / "*series_matrix.txt.gz"))[0]
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        body = "".join(l for l in fh if not l.startswith("!"))
    df = pd.read_csv(io.StringIO(body), sep="\t", index_col=0, dtype={"ID_REF": str})
    df.index = df.index.astype(str)  # GPL4372 IDs are numeric; annotation IDs are text
    return df.apply(pd.to_numeric, errors="coerce")


def probe_to_symbol(gpl: str) -> pd.Series:
    """Probe ID -> gene symbol. Probes that hit several genes ('///') are dropped."""
    path = RAW / "platforms" / f"{gpl}.annot.gz"
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        lines = fh.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("ID\t"))
    ann = pd.read_csv(io.StringIO("".join(l for l in lines[start:] if not l.startswith("!"))),
                      sep="\t", usecols=["ID", "Gene symbol"], dtype=str)
    ann = ann.dropna()
    ann = ann[~ann["Gene symbol"].str.contains("///")]
    return ann.set_index("ID")["Gene symbol"]


def collapse(expr: pd.DataFrame, sym: pd.Series, rank: pd.Series) -> pd.DataFrame:
    """Keep one probe per gene: the one with the highest `rank` value (mean or variance)."""
    expr = expr.loc[expr.index.intersection(sym.index)]
    df = pd.DataFrame({"sym": sym.reindex(expr.index), "rank": rank.reindex(expr.index)})
    best = df.sort_values("rank", ascending=False).groupby("sym").head(1)
    out = expr.loc[best.index]
    out.index = best["sym"].values
    return out


def save(ds: str, genes_x_samples: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(PROC / f"samples_{ds}.csv")
    keep = s.loc[s.role != "excluded", "sample_id"]
    x = genes_x_samples[keep].T  # samples x genes
    x.columns = x.columns.astype(str)
    x.to_parquet(PROC / f"expr_{ds}.parquet")
    LOG[ds] = {"samples": int(x.shape[0]), "genes": int(x.shape[1]),
               "min": float(np.nanmin(x.values)), "median": float(np.nanmedian(x.values)),
               "max": float(np.nanmax(x.values)), "nan_frac": float(np.isnan(x.values).mean())}
    print(ds, LOG[ds])
    return x


def arrays() -> dict[str, pd.DataFrame]:
    out = {}
    # Illumina HT-12 sets: already log2 normalized. Pick probe with highest mean.
    ilmn = probe_to_symbol("GPL10558")
    for ds in ["GSE132903", "GSE118553"]:
        e = load_series_matrix(ds)
        out[ds] = save(ds, collapse(e, ilmn, e.mean(axis=1)))

    # GSE48350: values were divided by each gene's median (relative, not absolute).
    # Rank probes by absolute mean in GSE5281 (same GPL570 chip, MAS5 values).
    affy = probe_to_symbol("GPL570")
    ref_mean = np.log2(load_series_matrix("GSE5281").clip(lower=1)).mean(axis=1)
    e = np.log2(load_series_matrix("GSE48350").clip(lower=0.01))
    out["GSE48350"] = save("GSE48350", collapse(e, affy, ref_mean))

    # GSE33000: log10 ratio vs a pooled reference -> log2 ratio. Rank probes by variance.
    e = load_series_matrix("GSE33000") * np.log2(10)
    out["GSE33000"] = save("GSE33000", collapse(e, probe_to_symbol("GPL4372"), e.var(axis=1)))
    return out


def median_of_ratios(counts: pd.DataFrame) -> pd.Series:
    """DESeq2 size factors (genes x samples input)."""
    log = np.log(counts.replace(0, np.nan))
    ok = log.notna().all(axis=1)
    ref = log[ok].mean(axis=1)
    return np.exp((log[ok].sub(ref, axis=0)).median(axis=0))


def allen() -> pd.DataFrame:
    s = pd.read_csv(PROC / "samples_GSE104687.csv")
    genes = pd.read_csv(RAW / "allen" / "expr" / "rows-genes.csv")
    entrez2sym = genes.set_index("gene_entrez_id")["gene_symbol"]
    cols = {}
    for pid in s.rnaseq_profile_id.astype(int):
        d = pd.read_csv(RAW / "allen" / "rsem" / f"{pid}.tsv", sep="\t", usecols=["gene_id", "expected_count"])
        cols[pid] = d.set_index("gene_id")["expected_count"]
    counts = pd.DataFrame(cols).round().astype(int)
    counts.index = entrez2sym.reindex(counts.index).values
    counts = counts[counts.index.notna()]
    counts = counts.groupby(level=0).sum()
    counts.columns = s.set_index("rnaseq_profile_id").loc[counts.columns, "sample_id"].values
    counts.to_parquet(PROC / "allen_counts.parquet")

    # Expressed genes: CPM >= 1 in at least 25% of samples.
    cpm = counts / counts.sum() * 1e6
    expressed = (cpm >= 1).mean(axis=1) >= 0.25
    LOG["allen_genes_total"] = int(len(counts))
    LOG["allen_genes_expressed"] = int(expressed.sum())
    sf = median_of_ratios(counts[expressed])
    norm = np.log2(counts[expressed] / sf + 1)
    LOG["allen_size_factor_range"] = [float(sf.min()), float(sf.max())]
    return save("GSE104687", norm)


def main() -> None:
    arr = arrays()
    al = allen()
    sets = [set(x.columns) for x in [*arr.values(), al]]
    universe = sorted(set.intersection(*sets))
    LOG["gene_universe"] = len(universe)
    (PROC / "phase2_gene_universe.txt").write_text("\n".join(universe))
    (RES / "preprocess_log.json").write_text(json.dumps(LOG, indent=2))
    print("genes in all datasets (Allen-expressed):", len(universe))


if __name__ == "__main__":
    main()
