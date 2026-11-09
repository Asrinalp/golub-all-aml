"""Adim 2: Veri kalitesi ve etiket-sutun eslesmesinin dogrulanmasi (iki veri seti)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR, TAB_DIR, update_summary  # noqa: E402
from data_io import (  # noqa: E402
    _read_golub_file,
    dudoit_filter,
    filtered_log,
    load_golub,
    load_gse14062,
    log_expression,
)

# Zangrando ve ark. (2009) Tablo 1 ve Yontemler bolumu
PAPER_COUNTS = {("ALL", False): 90, ("AML", False): 26, ("ALL", True): 16, ("AML", True): 8}
PAPER_FILTERED_PROBES = 5130


def value_profile(expr: pd.DataFrame) -> dict:
    v = expr.values.ravel()
    return {
        "n_probes": int(expr.shape[0]),
        "n_samples": int(expr.shape[1]),
        "n_missing": int(np.isnan(v).sum()),
        "n_inf": int(np.isinf(v).sum()),
        "min": float(np.nanmin(v)),
        "max": float(np.nanmax(v)),
        "median": float(np.nanmedian(v)),
        "pct_negative": float(100 * np.mean(v < 0)),
        "pct_nonpositive": float(100 * np.mean(v <= 0)),
        "pct_below_100": float(100 * np.mean(v < 100)),
        "pct_above_16000": float(100 * np.mean(v > 16000)),
    }


def call_profile(expr: pd.DataFrame, calls: pd.DataFrame) -> dict:
    flat_calls = calls.values.ravel()
    flat_vals = expr.values.ravel()
    out = {}
    for c in sorted(set(flat_calls)):
        mask = flat_calls == c
        out[c] = {
            "pct": float(100 * mask.mean()),
            "median_value": float(np.median(flat_vals[mask])),
            "pct_negative": float(100 * np.mean(flat_vals[mask] < 0)),
        }
    neg = flat_vals < 0
    out["pct_negatives_called_absent"] = float(100 * np.mean(flat_calls[neg] == "A"))
    return out


def array_qc(ds) -> pd.DataFrame:
    """Dizi (ornek) duzeyinde kalite: ortanca, ortalama diziler arasi korelasyon, yuzde P."""
    log = log_expression(ds)
    usable = log.loc[log.std(axis=1) > 0]
    corr = np.corrcoef(usable.values.T)
    np.fill_diagonal(corr, np.nan)
    qc = pd.DataFrame(index=ds.expr.columns)
    qc["label"] = ds.meta["label"]
    for column in ("set", "source", "tissue", "mll"):
        if column in ds.meta and not (ds.name == "GSE14062" and column == "source"):
            qc[column] = ds.meta[column]
    qc["median_raw"] = ds.expr.median(axis=0)
    qc["mean_corr"] = np.nanmean(corr, axis=0)
    if ds.calls is not None:
        qc["pct_present"] = (ds.calls == "P").mean(axis=0) * 100
        qc["pct_nonpositive"] = (ds.expr <= 0).mean(axis=0) * 100
    # Robust z: |z| > 3.5 olan diziler aykiri aday olarak isaretlenir (silinmez).
    rz_cols = []
    for col in ("pct_present", "mean_corr"):
        if col in qc:
            med = qc[col].median()
            qc[f"rz_{col}"] = (qc[col] - med) / (1.4826 * (qc[col] - med).abs().median())
            rz_cols.append(f"rz_{col}")
    qc["outlier"] = (qc[rz_cols].abs() > 3.5).any(axis=1)
    return qc


def verify_golub_labels(ds) -> dict:
    """Uc bagimsiz denetim: actual.csv ile Broad tablosu, sayimlar ve sutun sirasi tuzagi."""
    agree = int((ds.meta["label"] == ds.meta["label_table"]).sum())
    labels_in_id_order = ds.meta["label"].sort_index()
    # Tuzak: dosya sutun sirasi hasta numarasi sirasi sanilirsa kac ornek yanlis etiketlenir?
    naive = pd.Series(labels_in_id_order.values, index=ds.file_order)
    wrong = int((naive != ds.meta["label"].reindex(ds.file_order)).sum())
    train = ds.meta[ds.meta["set"] == "train"]
    test = ds.meta[ds.meta["set"] == "test"]
    return {
        "labels_agree_actual_vs_broad_table": f"{agree}/{len(ds.meta)}",
        "train_counts": train["label"].value_counts().to_dict(),
        "test_counts": test["label"].value_counts().to_dict(),
        "total_counts": ds.meta["label"].value_counts().to_dict(),
        "train_ids_are_1_38": sorted(train.index) == list(range(1, 39)),
        "test_ids_are_39_72": sorted(test.index) == list(range(39, 73)),
        "file_order_is_numeric": ds.file_order == sorted(ds.file_order),
        "file_order_head_test": ds.file_order[38:50],
        "mislabelled_if_column_order_trusted": wrong,
        "lineage_counts": ds.meta["lineage"].value_counts().to_dict(),
        "tissue_by_set": pd.crosstab(ds.meta["set"], ds.meta["tissue"]).to_dict(),
        "source_by_label": pd.crosstab(ds.meta["source"], ds.meta["label"]).to_dict(),
    }


def verify_gse14062(ds) -> dict:
    """GEO basliklarini makalenin sayimlariyla ve yayimlanan filtreyle karsilastirir."""
    counts = ds.meta.groupby(["label", "mll"]).size().to_dict()
    return {
        "counts_by_label_mll": {f"{k[0]}_{'MLL' if k[1] else 'noMLL'}": int(v) for k, v in counts.items()},
        "counts_match_paper": {k: int(counts.get(k, -1)) for k in PAPER_COUNTS} == PAPER_COUNTS,
        "total_counts": ds.meta["label"].value_counts().to_dict(),
        "characteristics": ds.meta["characteristics"].value_counts().to_dict(),
        "platform": ds.meta["platform"].unique().tolist(),
        "samples_in_order_of_title": ds.meta["number"].tolist() == list(range(1, len(ds.meta) + 1)),
        "columns_match_sample_table": list(ds.expr.columns) == list(ds.meta.index),
        "n_filtered_iqr": int(len(filtered_log(ds))),
        "paper_filtered_probes": PAPER_FILTERED_PROBES,
        "n_affx": int(ds.expr.index.str.startswith("AFFX").sum()),
    }


def verify_kaggle_vs_broad() -> dict:
    """Kaggle CSV ile Broad TXT dosyalarindaki degerler bire bir ayni mi?"""
    results = {}
    for name, kag, broad in [
        ("train", "data_set_ALL_AML_train.csv", "data_set_ALL_AML_train.txt"),
        ("test", "data_set_ALL_AML_independent.csv", "data_set_ALL_AML_independent.txt"),
    ]:
        k_expr, _, _ = _read_golub_file(RAW_DIR / "golub_kaggle" / kag)
        # Broad satirlari fazladan sekmeyle bitiyor; index_col=False kaymayi onler.
        raw = pd.read_csv(RAW_DIR / "golub_broad" / broad, sep="\t", dtype=str, index_col=False)
        cols = [c for c in raw.columns[2:] if not c.startswith("call") and not c.startswith("Unnamed")]
        b_expr = raw[cols].apply(pd.to_numeric, errors="coerce")
        b_expr.columns = [int(c) for c in cols]
        same_shape = k_expr.shape == b_expr.shape
        same_probes = same_shape and bool((raw["Gene Accession Number"].values == k_expr.index.values).all())
        same_columns = list(b_expr.columns) == list(k_expr.columns)
        diff = int((k_expr.values != b_expr.values).sum()) if same_shape and same_columns else None
        results[name] = {
            "broad_shape": list(b_expr.shape),
            "kaggle_shape": list(k_expr.shape),
            "same_probe_order": same_probes,
            "same_column_order": same_columns,
            "n_different_values": diff,
        }
    return results


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    golub, gse = load_golub(), load_gse14062()
    g_train = golub.expr.loc[:, golub.meta["set"] == "train"]
    summary = {
        "golub_values": value_profile(golub.expr),
        "gse14062_values": value_profile(gse.expr),
        "golub_calls": call_profile(golub.expr, golub.calls),
        "golub_labels": verify_golub_labels(golub),
        "golub_kaggle_vs_broad": verify_kaggle_vs_broad(),
        "gse14062_checks": verify_gse14062(gse),
        "golub_unique_probes": bool(golub.expr.index.is_unique),
        "gse14062_unique_probes": bool(gse.expr.index.is_unique),
        "golub_affx_probes": int(golub.expr.index.str.startswith("AFFX").sum()),
        "dudoit_filter": {
            "golub_all_72": int(len(dudoit_filter(golub.expr))),
            "golub_train_38": int(len(dudoit_filter(g_train))),
        },
    }
    for ds, tag in [(golub, "golub"), (gse, "gse14062")]:
        qc = array_qc(ds)
        qc.to_csv(TAB_DIR / f"array_qc_{tag}.csv")
        part = {
            "mean_corr_min": float(qc["mean_corr"].min()),
            "mean_corr_max": float(qc["mean_corr"].max()),
            "outliers": [str(s) for s in qc.index[qc["outlier"]]],
            "outlier_details": {str(s): {"mean_corr": float(qc.loc[s, "mean_corr"]), "robust_z": float(qc.loc[s, "rz_mean_corr"]), "label": qc.loc[s, "label"]}
                                for s in qc.index[qc["outlier"]]},
        }
        if "pct_present" in qc:
            part["pct_present_min"] = float(qc["pct_present"].min())
            part["pct_present_max"] = float(qc["pct_present"].max())
        summary[f"{tag}_array_qc"] = part
    update_summary("quality", summary)
    print("Kalite adimi tamam.")


if __name__ == "__main__":
    main()
