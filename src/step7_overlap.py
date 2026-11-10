"""Adim 7: Veri setleri arasinda hasta ortusmesi.

Ayni ornek iki farkli cipte olculduyse, sinif ortalamasi cikarildiktan sonra kalan hastaya
ozgu profil yine de yuksek korele kalmalidir. Ancak soy icindeki alt tipler de (ayni hasta
olmadan) korelasyon uretir; farkli soylardan cift kullanan bir esik bunu yakalayamaz.
Bu yuzden referans, hastalari ayni olamayacak bir veri seti ciftidir:

  - Golub (ABD, 1999) - GSE14062 (Italya, AIEOP): soy ici en yuksek korelasyon referanstir.
  - Golub - Armstrong 2002: elenen aday. Makale, AML orneklerinin Golub 1999'da
    tanimlandigini belirtir; bu referansla sinanir.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TAB_DIR, update_summary  # noqa: E402
from data_io import (  # noqa: E402
    collapse_to_genes,
    load_annotation,
    load_armstrong,
    load_golub,
    load_gse14062,
    log2_floor,
    log_expression,
    probe_to_symbol,
)

N_GENES = 1000


def within_class_z(frame: pd.DataFrame, cols: list) -> pd.DataFrame:
    x = frame[cols]
    sd = x.std(axis=1, ddof=1).replace(0, np.nan)
    return x.sub(x.mean(axis=1), axis=0).div(sd, axis=0)


def cross_correlation(g: pd.DataFrame, a: pd.DataFrame, g_cols: list, a_cols: list, n_genes: int) -> pd.DataFrame:
    """Her iki grupta da degisken genlerde, sinif ici z-skorlu ornek-ornek korelasyonu."""
    rank = g[g_cols].var(axis=1).rank() + a[a_cols].var(axis=1).rank()
    genes = rank.nlargest(n_genes).index
    zg = within_class_z(g, g_cols).loc[genes].fillna(0)
    za = within_class_z(a, a_cols).loc[genes].fillna(0)
    r = np.corrcoef(zg.values.T, za.values.T)[: len(g_cols), len(g_cols):]
    return pd.DataFrame(r, index=g_cols, columns=a_cols)


def mutual_best_pairs(table: pd.DataFrame) -> pd.DataFrame:
    best_g, best_a = table.idxmax(axis=1), table.idxmax(axis=0)
    rows = [{"other": a, "golub": g, "r": float(table.loc[g, a])}
            for a, g in best_a.items() if best_g[g] == a]
    return pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)


def lineage_pairs(g_s, o_s, g_meta, o_labels, n_genes=N_GENES) -> dict:
    """Her soy icin karsilikli en iyi ciftler."""
    g_cols = {"AML": list(g_meta.index[g_meta["label"] == "AML"]),
              "ALL": list(g_meta.index[g_meta["lineage"] == "B-ALL"])}
    return {lin: mutual_best_pairs(cross_correlation(g_s, o_s, g_cols[lin], list(o_labels.index[o_labels == lin]), n_genes))
            for lin in ("AML", "ALL")}


def main() -> None:
    golub = load_golub()
    meta = golub.meta
    g_genes = collapse_to_genes(log2_floor(golub.expr), probe_to_symbol(golub.expr.index, load_annotation("GPL80")))

    gse = load_gse14062()
    r_genes = collapse_to_genes(log_expression(gse), probe_to_symbol(gse.expr.index, load_annotation("GPL570")))
    arm = load_armstrong()
    a_genes = collapse_to_genes(log2_floor(arm.expr), probe_to_symbol(arm.expr.index, load_annotation("GPL91")))
    shared_r = g_genes.index.intersection(r_genes.index)
    shared_a = g_genes.index.intersection(a_genes.index)

    results, sensitivity = {}, {}
    for n in (500, N_GENES, 2000):
        ref = lineage_pairs(g_genes.loc[shared_r], r_genes.loc[shared_r], meta, gse.meta["label"], n)
        arm_pairs = lineage_pairs(g_genes.loc[shared_a], a_genes.loc[shared_a], meta, arm.meta["label"], n)
        reference = {lin: float(ref[lin]["r"].max()) for lin in ref}
        sensitivity[str(n)] = {
            "reference_aml": reference["AML"],
            "reference_all": reference["ALL"],
            "armstrong_aml_above": int((arm_pairs["AML"]["r"] > reference["AML"]).sum()),
            "armstrong_all_above": int((arm_pairs["ALL"]["r"] > reference["ALL"]).sum()),
        }
        if n == N_GENES:
            aml = arm_pairs["AML"].copy()
            aml["golub_set"] = meta.loc[aml["golub"], "set"].values
            aml["above_reference"] = aml["r"] > reference["AML"]
            TAB_DIR.mkdir(parents=True, exist_ok=True)
            aml.to_csv(TAB_DIR / "overlap_armstrong_aml.csv", index=False)
            ref["AML"].to_csv(TAB_DIR / "overlap_gse14062_golub_aml.csv", index=False)
            ref["ALL"].to_csv(TAB_DIR / "overlap_gse14062_golub_all.csv", index=False)
            flagged = aml[aml["above_reference"]]
            results = {
                "n_genes": N_GENES,
                "armstrong_shape": {"n_probes": int(arm.expr.shape[0]), "n_samples": int(arm.expr.shape[1])},
                "armstrong_class_counts": {k: int(v) for k, v in arm.meta["label"].value_counts().items()},
                "reference_gse14062_golub": {
                    "aml_top_r": reference["AML"],
                    "all_top_r": reference["ALL"],
                },
                "armstrong_golub": {
                    "n_aml": int((arm.meta["label"] == "AML").sum()),
                    "aml_top_r": float(aml["r"].max()),
                    "aml_above_reference": int(len(flagged)),
                    "aml_flagged_in_golub_train": int((flagged["golub_set"] == "train").sum()),
                    "aml_flagged_in_golub_test": int((flagged["golub_set"] == "test").sum()),
                    "aml_flagged_pairs": [f"{a}={g}" for a, g in zip(flagged["other"], flagged["golub"])],
                    "all_top_r": float(arm_pairs["ALL"]["r"].max()),
                    "all_above_reference": int((arm_pairs["ALL"]["r"] > reference["ALL"]).sum()),
                },
            }
    results["gene_count_sensitivity"] = sensitivity
    update_summary("overlap", results)
    print("Ortusme adimi tamam.")


if __name__ == "__main__":
    main()
