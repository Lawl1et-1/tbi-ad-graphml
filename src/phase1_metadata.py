"""Phase 1: build clean sample tables from GEO series matrices + Allen donor files.

Output: data/processed/samples_<dataset>.csv and data/processed/samples_all.csv
Columns: sample_id, dataset, donor_id, region, region_group, age, sex, pmi,
         diagnosis (AD / control / other), dementia, tbi, platform, + dataset extras.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Coarse region groups so datasets can be compared to the Allen regions
# (Allen: TCx = temporal cortex, PCx = parietal cortex, HIP = hippocampus, FWM = white matter).
REGION_GROUP = {
    "TCx": "temporal_cortex", "MTG": "temporal_cortex", "TC": "temporal_cortex",
    "FC": "frontal_cortex", "CB": "cerebellum",
    "PCx": "parietal_cortex", "PCG": "parietal_cortex",  # postcentral gyrus is parietal
    "HIP": "hippocampus", "EC": "entorhinal_cortex",
    "FWM": "white_matter",
    "DLPFC": "frontal_cortex", "SFG": "frontal_cortex",
    "PC": "cingulate_cortex", "VCX": "occipital_cortex",
}


def read_series_header(gse: str) -> pd.DataFrame:
    """Return one row per sample with every '!Sample_*' header field.

    Repeated characteristics lines are split into 'key: value' columns so that
    rows where GEO shifted the field order are still parsed correctly.
    """
    path = next((RAW / gse).glob("*series_matrix.txt.gz"))
    fields: dict[str, list[str]] = {}
    chars: list[list[str]] = []
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            key, vals = parts[0][len("!Sample_"):], parts[1:]
            if key.startswith("characteristics"):
                chars.append(vals)
            elif key not in fields:
                fields[key] = vals
    df = pd.DataFrame(fields)
    for vals in chars:
        for i, v in enumerate(vals):
            if ":" in v:
                k, val = v.split(":", 1)
                # GSE5281 values end in a stray non-ASCII byte; keep printable ASCII only.
                clean = "".join(c for c in val if 32 <= ord(c) < 127).strip()
                df.loc[i, "ch:" + k.strip().lower()] = clean
    return df


def age_num(x) -> float:
    """'90-94' -> 92, '100+' -> 100, '72 yrs' -> 72, '90+' -> 90."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)", s)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    m = re.search(r"\d+", s)
    return float(m.group()) if m else np.nan


def sex_norm(x) -> str:
    s = str(x).strip().lower()
    return {"m": "M", "male": "M", "f": "F", "female": "F"}.get(s, "NA")


# ---------------------------------------------------------------- Allen (TBI)
def allen() -> pd.DataFrame:
    h = read_series_header("GSE104687")
    donors = pd.read_csv(RAW / "allen" / "DonorInformation.csv")
    files = pd.read_csv(RAW / "allen" / "tbi_data_files.csv")
    df = pd.DataFrame({
        "sample_id": h["geo_accession"],
        "title": h["title"],
        "dataset": "GSE104687",
        "donor_id": h["ch:donor_name"],
        "region": h["ch:brain region"],
        "age_raw": h["ch:age_at_death"],
        "sex": h["ch:sex"].map(sex_norm),
        "pmi": np.nan,  # not released per donor
        "rin": pd.to_numeric(h["ch:rin"], errors="coerce"),
        "batch": h["ch:batch"],
        "hemisphere": h["ch:hemisphere"],
        "dementia": (h["ch:act_demented"] == "Dementia").astype(int),
        "tbi": (h["ch:ever_tbi_w_loc"] == "Y").astype(int),
        "num_tbi_w_loc": pd.to_numeric(h["ch:num_tbi_w_loc"], errors="coerce"),
        "braak": pd.to_numeric(h["ch:braak"], errors="coerce"),
        "cerad": pd.to_numeric(h["ch:cerad"], errors="coerce"),
        "nia_reagan": pd.to_numeric(h["ch:nia_reagan"], errors="coerce"),
        "apoe4": h["ch:apo_e4_allele"],
        "platform": "GPL16791 (Illumina HiSeq 2500, RNA-seq)",
    })
    df["age"] = df["age_raw"].map(age_num)
    # Add donor-level fields that GEO does not carry.
    d = donors.rename(columns={"donor_id": "donor_id_allen", "name": "donor_id"})[
        ["donor_id", "donor_id_allen", "dsm_iv_clinical_diagnosis", "nincds_arda_diagnosis",
         "control_set", "education_years", "race"]]
    df = df.merge(d, on="donor_id", how="left")
    # Link GEO sample -> Allen rnaseq_profile_id (needed to pull expression columns).
    files["region"] = files["structure_acronym"]
    link = files[["donor_name", "region", "rnaseq_profile_id", "rnaseq_total_reads",
                  "rnaseq_percent_reads_aligned_to_mrna"]].rename(columns={"donor_name": "donor_id"})
    df = df.merge(link, on=["donor_id", "region"], how="left")
    # AD label for Allen: clinical AD diagnosis (NINCDS-ADRDA probable/possible AD).
    nincds = df["nincds_arda_diagnosis"].fillna("").str.lower()
    df["diagnosis"] = np.where(nincds.str.contains("alzheimer"), "AD",
                               np.where(df["dementia"] == 0, "control", "other_dementia"))
    return df


# ---------------------------------------------------------- AD training sets
def gse33000() -> pd.DataFrame:
    h = read_series_header("GSE33000")
    status = h["ch:disease status"]
    return pd.DataFrame({
        "sample_id": h["geo_accession"], "title": h["title"], "dataset": "GSE33000",
        "donor_id": "GSE33000_" + h["source_name_ch2"],  # one sample per donor
        "region": "DLPFC", "age": h["ch:age"].map(age_num),
        "sex": h["ch:gender"].map(sex_norm), "pmi": np.nan,
        "diagnosis": status.map({"Alzheimer's disease": "AD", "non-demented": "control"}).fillna("HD"),
        "platform": "GPL4372 (Rosetta/Merck Agilent 44K, log10 ratio vs pool)",
    })


def gse132903() -> pd.DataFrame:
    h = read_series_header("GSE132903")
    donor = h["title"].str.extract(r"_(\d{2}-\d+)$")[0]
    return pd.DataFrame({
        "sample_id": h["geo_accession"], "title": h["title"], "dataset": "GSE132903",
        "donor_id": "GSE132903_" + donor.fillna(h["geo_accession"]),
        "region": "MTG", "age": h["ch:expired_age (years)"].map(age_num),
        "sex": h["ch:sex"].map(sex_norm), "pmi": np.nan,
        "diagnosis": h["ch:diagnosis"].map({"AD": "AD", "ND": "control"}),
        "platform": "GPL10558 (Illumina HumanHT-12 v4)",
    })


def gse5281() -> pd.DataFrame:
    h = read_series_header("GSE5281")
    reg = h["ch:organ region"].str.lower()
    reg_map = {"entorhinal cortex": "EC", "hippocampus": "HIP", "medial temporal gyrus": "MTG",
               "superior frontal gyrus": "SFG", "primary visual cortex": "VCX",
               "posterior cingulate": "PC",
               "posterior singulate": "PC"}
    region = reg.map(lambda r: next((v for k, v in reg_map.items() if k in r), r))
    return pd.DataFrame({
        "sample_id": h["geo_accession"], "title": h["title"], "dataset": "GSE5281",
        # Donor IDs are not in the series matrix; see limitations.
        "donor_id": "GSE5281_unknown_" + h["geo_accession"],
        "region": region, "age": h["ch:age"].map(age_num),
        "sex": h["ch:sex"].map(sex_norm), "pmi": np.nan,
        "diagnosis": np.where(h["ch:disease state"].str.contains("Alzheimer"), "AD", "control"),
        "cell_type": h["ch:cell type"],
        "platform": "GPL570 (Affymetrix U133 Plus 2.0), laser-captured neurons",
    })


def gse48350() -> pd.DataFrame:
    h = read_series_header("GSE48350")
    reg_map = {"postcentral gyrus": "PCG", "post-central gyrus": "PCG", "superior frontal gyrus": "SFG",
               "hippocampus": "HIP", "entorhinal cortex": "EC"}
    is_ad = h["title"].str.contains("_AD", regex=False)
    df = pd.DataFrame({
        "sample_id": h["geo_accession"], "title": h["title"], "dataset": "GSE48350",
        "region": h["ch:brain region"].str.lower().map(reg_map),
        "age": h["ch:age (yrs)"].map(age_num), "sex": h["ch:gender"].map(sex_norm), "pmi": np.nan,
        "diagnosis": np.where(is_ad, "AD", "control"),
        "braak_roman": h.get("ch:braak stage"), "apoe": h.get("ch:apoe genotype"),
        "mmse": h.get("ch:mmse"),
        "platform": "GPL570 (Affymetrix U133 Plus 2.0)",
    })
    ind = h["ch:individual"].str.split(",").str[0].str.strip()
    # AD donors have no ID: infer one from (sex, age, braak, apoe, mmse). Approximate.
    key = df[["sex", "age", "braak_roman", "apoe", "mmse"]].fillna("NA").astype(str).agg("|".join, axis=1)
    inferred = "GSE48350_ADinf_" + pd.Series(pd.factorize(key)[0], index=df.index).astype(str)
    df["donor_id"] = np.where(ind.notna(), "GSE48350_C" + ind.fillna(""), inferred)
    df["donor_id_inferred"] = ind.isna()
    return df


def gse118553() -> pd.DataFrame:
    """Added after Phase 1 to replace GSE5281 (bulk tissue, same platform as GSE132903)."""
    h = read_series_header("GSE118553")
    reg_map = {"Temporal_Cortex": "TC", "Entorhinal_Cortex": "EC", "Frontal_Cortex": "FC",
               "Cerebellum": "CB"}
    donor = h["ch:individual"].str.replace(r"\s+", "_", regex=True)
    return pd.DataFrame({
        "sample_id": h["geo_accession"], "title": h["title"], "dataset": "GSE118553",
        "donor_id": "GSE118553_" + donor,
        "region": h["ch:tissue"].map(reg_map), "age": h["ch:age"].map(age_num),
        "sex": h["ch:gender"].map(sex_norm), "pmi": np.nan,
        "diagnosis": h["ch:disease state"].map({"AD": "AD", "control": "control", "AsymAD": "AsymAD"}),
        "platform": "GPL10558 (Illumina HumanHT-12 v4)",
    })


def assign_role(t: pd.DataFrame) -> pd.Series:
    """Which part of the pipeline each sample belongs to (decided after Phase 1).

    train    : AD-signature training (GSE132903, GSE118553, GSE48350 aged >= 60)
    test_ext : external AD test only, never trained on (GSE33000, confounded by age/sex)
    transfer : Allen TBI cohort, never trained on
    excluded : not used (GSE5281, Huntington's, AsymAD, cerebellum, young controls)
    """
    ds = t["dataset"].iloc[0]
    if ds == "GSE104687":
        return pd.Series("transfer", index=t.index)
    if ds == "GSE5281":
        return pd.Series("excluded", index=t.index)
    ok = t["diagnosis"].isin(["AD", "control"])
    if ds == "GSE33000":
        return pd.Series(np.where(ok, "test_ext", "excluded"), index=t.index)
    if ds == "GSE118553":
        ok &= t["region"] != "CB"
    if ds == "GSE48350":
        ok &= t["age"] >= 60
    return pd.Series(np.where(ok, "train", "excluded"), index=t.index)


def main() -> None:
    tables = {
        "GSE104687": allen(), "GSE33000": gse33000(), "GSE132903": gse132903(),
        "GSE5281": gse5281(), "GSE48350": gse48350(), "GSE118553": gse118553(),
    }
    for name, t in tables.items():
        t["region_group"] = t["region"].map(REGION_GROUP).fillna("other")
        for col in ("dementia", "tbi"):
            if col not in t:
                t[col] = np.nan
        t["role"] = assign_role(t)
        t.to_csv(OUT / f"samples_{name}.csv", index=False)
    common = ["sample_id", "dataset", "donor_id", "region", "region_group", "age", "sex",
              "pmi", "diagnosis", "dementia", "tbi", "platform", "role"]
    pd.concat([t[common] for t in tables.values()]).to_csv(OUT / "samples_all.csv", index=False)
    print("wrote", ", ".join(f"{k}={len(v)}" for k, v in tables.items()))


if __name__ == "__main__":
    main()
