"""Adim 8: MILE (GSE13159) ile GSE14062 ve Golub arasinda hasta ortusmesi.

GSE14062 ve MILE ayni cipte (GPL570) olculdu; ayni hastanin sinif ortalamasi cikarilmis
profili bu nedenle cok yuksek korele kalir. Bir ornegin en iyi eslesmesi disindaki tum
korelasyonlar, ayni hasta olmayan ciftler icin referans dagilim olarak kullanilir.
Eslesme, seçimde kullanilmayan MLL durumuyla bagimsiz olarak denetlenir.

Cikti results/tables/gse14062_in_mile.csv: MILE B4 ve B1 icin kullanilirken bu ornekler
cikarilmalidir. 886 MB indirir; `run_all.py --skip-mile` ile atlanabilir.
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR, TAB_DIR, update_summary  # noqa: E402
from data_io import collapse_to_genes, load_annotation, load_golub, load_gse14062, log2_floor, probe_to_symbol  # noqa: E402
from step1_download import download  # noqa: E402
from step7_overlap import cross_correlation, mutual_best_pairs  # noqa: E402

MILE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13159/matrix/GSE13159_series_matrix.txt.gz"
N_PROBES = 2000


def load_mile() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ALL ve AML orneklerinin ifadesini okur (float32, yaklasik 290 MB bellek); meta tum ornekleri icerir."""
    path = RAW_DIR / "mile/GSE13159_series_matrix.txt.gz"
    download(MILE_URL, path)
    ids, classes, tissue, skip = None, None, None, 0
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for n, line in enumerate(f):
            if line.startswith("!Sample_geo_accession"):
                ids = [x.strip().strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip().strip('"') for x in line.split("\t")[1:]]
                if vals[0].startswith("leukemia class"):
                    classes = [v.split(": ", 1)[1] for v in vals]
                elif vals[0].startswith("sample type"):
                    tissue = [v.split(": ", 1)[1] for v in vals]
            elif line.startswith("!series_matrix_table_begin"):
                skip = n + 1
                break
    meta = pd.DataFrame({"mile_class": classes, "tissue": tissue}, index=ids)
    meta["label"] = ["AML" if c.startswith("AML") else ("ALL" if "ALL" in c else "diger") for c in meta["mile_class"]]
    keep = list(meta.index[meta["label"] != "diger"])
    expr = pd.read_csv(path, sep="\t", skiprows=skip, usecols=["ID_REF", *keep], index_col=0,
                       comment="!", dtype={k: "float32" for k in keep})
    return expr[keep], meta


def pair_gse_mile(gse, expr, meta, lineage: str) -> pd.DataFrame:
    probes = gse.expr.index.intersection(expr.index)
    g_cols = list(gse.meta.index[gse.meta["label"] == lineage])
    m_cols = list(meta.index[meta["label"] == lineage])
    table = cross_correlation(gse.expr.loc[probes], expr.loc[probes], g_cols, m_cols, N_PROBES)
    values = table.values.copy()
    # Referans: her GSE14062 orneginin en iyi eslesmesi disarida birakilinca kalan korelasyonlar
    best_idx = values.argmax(axis=1)
    values[np.arange(len(g_cols)), best_idx] = np.nan
    reference = np.nanquantile(values, 0.9999)
    pairs = mutual_best_pairs(table.T).rename(columns={"other": "gse14062", "golub": "mile"})
    pairs["second_best_r"] = np.nanmax(values, axis=1)[[g_cols.index(g) for g in pairs["gse14062"]]]
    pairs["lineage"] = lineage
    pairs["reference_r"] = float(reference)
    pairs["matched"] = pairs["r"] > reference
    return pairs


def main() -> None:
    gse = load_gse14062()
    expr, meta = load_mile()

    pairs = pd.concat([pair_gse_mile(gse, expr, meta, "ALL"), pair_gse_mile(gse, expr, meta, "AML")], ignore_index=True)
    pairs["gse_mll"] = gse.meta.loc[pairs["gse14062"], "mll"].values
    pairs["mile_class"] = meta.loc[pairs["mile"], "mile_class"].values
    pairs["mile_mll"] = pairs["mile_class"].str.contains("MLL")
    pairs["mile_tissue"] = meta.loc[pairs["mile"], "tissue"].values
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    matched = pairs[pairs["matched"]]
    matched.to_csv(TAB_DIR / "gse14062_in_mile.csv", index=False)

    # Eslesmede kullanilmayan MLL durumu iki veri setinde uyusuyor mu?
    mll_table = pd.crosstab(matched["gse_mll"], matched["mile_mll"])
    agree = int((matched["gse_mll"] == matched["mile_mll"]).sum())

    # Golub - MILE: farkli cipler; GSE14062 ile ayni olcutle yalnizca en yuksek degerler raporlanir.
    golub = load_golub()
    sym_g = probe_to_symbol(golub.expr.index, load_annotation("GPL80"))
    g_genes = collapse_to_genes(log2_floor(golub.expr), sym_g)
    m_genes = collapse_to_genes(expr, probe_to_symbol(expr.index, load_annotation("GPL570")))
    shared = g_genes.index.intersection(m_genes.index)
    g_meta = golub.meta
    golub_tops = {}
    for name, g_cols, m_cols in [
        ("AML", list(g_meta.index[g_meta["label"] == "AML"]), list(meta.index[meta["label"] == "AML"])),
        ("ALL", list(g_meta.index[g_meta["label"] == "ALL"]), list(meta.index[meta["label"] == "ALL"])),
    ]:
        best = mutual_best_pairs(cross_correlation(g_genes.loc[shared], m_genes.loc[shared], g_cols, m_cols, 1000))
        golub_tops[name] = {"top_r": float(best["r"].max()), "n_mutual_above_0_7": int((best["r"] > 0.7).sum())}

    n_all, n_aml = int((meta["label"] == "ALL").sum()), int((meta["label"] == "AML").sum())
    m_all, m_aml = int((matched["lineage"] == "ALL").sum()), int((matched["lineage"] == "AML").sum())
    update_summary("mile_overlap", {
        "n_probes": N_PROBES,
        "mile_n_samples": int(len(meta)),
        "mile_other_total": int((meta["label"] == "diger").sum()),
        "mile_tissue_all": {k: int(v) for k, v in meta["tissue"].value_counts().items()},
        "gse14062_samples": int(len(gse.meta)),
        "matched_total": int(len(matched)),
        "matched_all": m_all,
        "matched_aml": m_aml,
        "reference_r_all": float(pairs.loc[pairs["lineage"] == "ALL", "reference_r"].iloc[0]),
        "reference_r_aml": float(pairs.loc[pairs["lineage"] == "AML", "reference_r"].iloc[0]),
        "matched_r_min": float(matched["r"].min()),
        "matched_r_median": float(matched["r"].median()),
        "matched_r_max": float(matched["r"].max()),
        "second_best_r_median": float(matched["second_best_r"].median()),
        "mll_status_agreement": f"{agree}/{len(matched)}",
        "mll_table": {str(k): {str(a): int(b) for a, b in v.items()} for k, v in mll_table.to_dict().items()},
        "mile_tissue_of_matches": matched["mile_tissue"].value_counts().to_dict(),
        "mile_all_total": n_all,
        "mile_aml_total": n_aml,
        "mile_all_after_exclusion": n_all - m_all,
        "mile_aml_after_exclusion": n_aml - m_aml,
        "golub_vs_mile": golub_tops,
    })
    print("MILE ortusme adimi tamam.")


if __name__ == "__main__":
    main()
