"""Ham dosyalari okur; etiketleri ORNEK KIMLIGI uzerinden esler."""
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR  # noqa: E402

FLOOR, CEIL = 100, 16000  # Dudoit ve ark. (2002) taban/tavan degerleri
IQR_CUTOFF = 1.15  # Zangrando ve ark. (2009) GSE14062 icin kullandigi filtre


@dataclass
class Dataset:
    name: str
    expr: pd.DataFrame      # prob x ornek; ham deger (GSE14062'de yayimlanan RMA log2)
    calls: pd.DataFrame | None  # prob x ornek; P/A/M (GSE14062'de yok)
    meta: pd.DataFrame      # ornek basina etiket ve ortak degiskenler
    file_order: list        # sutunlarin dosyadaki sirasi


def _read_golub_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    raw = pd.read_csv(path, dtype=str)
    columns = list(raw.columns)
    genes = raw["Gene Accession Number"].values
    # Duzen: Description, Accession, <ornek>, call, <ornek>, call.1, ...
    value_cols = [c for c in columns[2:] if not c.startswith("call")]
    call_cols = [c for c in columns[2:] if c.startswith("call")]
    expr = raw[value_cols].apply(pd.to_numeric, errors="coerce").set_axis(genes)
    expr.columns = expr.columns.astype(int)
    calls = raw[call_cols].set_axis(genes)
    calls.columns = expr.columns
    return expr, calls, expr.columns.tolist()


def load_sample_table() -> pd.DataFrame:
    """Broad'un table_ALL_AML_samples.txt dosyasini ayristirir."""
    rows = []
    text = (RAW_DIR / "golub_broad/table_ALL_AML_samples.txt").read_text(encoding="latin-1")
    for line in text.splitlines():
        m = re.match(r"^(\d+)\s+(ALL|AML)\s+(BM|PB)\b(.*)$", line)
        if not m:
            continue
        rest = m.group(4)
        cell = re.search(r"\b([TB])-cell", rest)
        fab = re.search(r"\bM(\d)\b", rest)
        tail = rest.split()
        gender = re.search(r"-\s*([MF])\b", rest)
        rows.append({
            "sample": int(m.group(1)),
            "label_table": m.group(2),
            "tissue": m.group(3),
            "lineage": (cell.group(1) + "-ALL") if cell else ("AML" if m.group(2) == "AML" else "bilinmiyor"),
            "fab": ("M" + fab.group(1)) if fab else "",
            "gender": gender.group(1) if gender else "",
            "ps": float(tail[-2]),
            "source": tail[-1],
        })
    return pd.DataFrame(rows).set_index("sample").sort_index()


def load_golub() -> Dataset:
    base = RAW_DIR / "golub_kaggle"
    train, train_calls, train_order = _read_golub_file(base / "data_set_ALL_AML_train.csv")
    test, test_calls, test_order = _read_golub_file(base / "data_set_ALL_AML_independent.csv")
    if not train.index.equals(test.index):
        raise ValueError("Egitim ve test dosyalarinda prob sirasi farkli")
    labels = pd.read_csv(base / "actual.csv").set_index("patient")["cancer"]
    samples = sorted(train_order + test_order)
    expr = pd.concat([train, test], axis=1)[samples]
    calls = pd.concat([train_calls, test_calls], axis=1)[samples]
    meta = load_sample_table().loc[samples].copy()
    meta["label"] = labels.loc[samples]
    meta["set"] = ["train" if s in train_order else "test" for s in samples]
    return Dataset("Golub", expr, calls, meta, train_order + test_order)


def load_gse14062() -> Dataset:
    """Etiketler GSM basligindan okunur: 'ALL without MLL - sample 1' gibi."""
    base = RAW_DIR / "gse14062"
    expr = pd.read_csv(base / "expression.tsv.gz", sep="\t", index_col=0)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    samples = pd.read_csv(base / "samples.tsv", sep="\t").set_index("gsm")
    parsed = samples["title"].str.extract(r"^(ALL|AML) (with|without) MLL - sample (\d+)$")
    if parsed.isna().any().any():
        raise ValueError("Beklenmeyen GSE14062 ornek basligi")
    meta = samples.copy()
    meta["label"] = parsed[0]
    meta["mll"] = parsed[1] == "with"
    meta["number"] = parsed[2].astype(int)
    file_order = list(expr.columns)
    return Dataset("GSE14062", expr, None, meta.loc[file_order], file_order)


def load_armstrong() -> Dataset:
    """Elenen aday; yalnizca hasta ortusmesi denetimi icin."""
    plain = pd.read_csv(RAW_DIR / "armstrong/expression_data.txt", sep="\t", dtype=str)
    expr = plain.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").set_axis(plain["Name"].values)
    expr.columns = [re.sub(r"_+", "_", c) for c in expr.columns]
    meta = pd.DataFrame(index=expr.columns)
    meta["label"] = [c.split("_")[0] for c in expr.columns]
    return Dataset("Armstrong", expr, None, meta, expr.columns.tolist())


def log2_floor(expr: pd.DataFrame, floor: float = FLOOR, ceil: float = CEIL) -> pd.DataFrame:
    return np.log2(expr.clip(lower=floor, upper=ceil))


def dudoit_filter(expr: pd.DataFrame, floor: float = FLOOR, ceil: float = CEIL) -> pd.Index:
    """max/min > 5 ve max - min > 500 olan problari dondurur (Dudoit ve ark. 2002)."""
    clipped = expr.clip(lower=floor, upper=ceil)
    hi, lo = clipped.max(axis=1), clipped.min(axis=1)
    return clipped.index[(hi / lo > 5) & (hi - lo > 500)]


def iqr_filter(log_expr: pd.DataFrame, cutoff: float = IQR_CUTOFF) -> pd.Index:
    """Ceyrekler acikligi esigi gecen problari dondurur (Zangrando ve ark. 2009)."""
    iqr = log_expr.quantile(0.75, axis=1) - log_expr.quantile(0.25, axis=1)
    return log_expr.index[iqr > cutoff]


def log_expression(ds: Dataset) -> pd.DataFrame:
    """Tum problar icin log2 olcek: Golub taban/tavanli, GSE14062 yayimlanan RMA."""
    return ds.expr if ds.name == "GSE14062" else log2_floor(ds.expr)


def filtered_log(ds: Dataset) -> pd.DataFrame:
    """Veri setinin yayimlandigi calismadaki filtreyle log2 matris."""
    if ds.name == "GSE14062":
        return ds.expr.loc[iqr_filter(ds.expr)]
    return log2_floor(ds.expr).loc[dudoit_filter(ds.expr)]


def load_annotation(gpl: str) -> pd.DataFrame:
    path = RAW_DIR / f"annot/{gpl}.annot"
    lines = path.read_text(encoding="latin-1").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ID\t"))
    end = next((i for i, line in enumerate(lines) if line.startswith("!platform_table_end")), len(lines))
    rows = [line.split("\t") for line in lines[start + 1:end]]
    header = lines[start].split("\t")
    df = pd.DataFrame([r[:len(header)] for r in rows], columns=header)
    return df[["ID", "Gene symbol"]].rename(columns={"Gene symbol": "symbol"}).set_index("ID")


def probe_to_symbol(index: pd.Index, annot: pd.DataFrame, split_multi: bool = False) -> pd.Series:
    """Prob -> gen sembolu eslemesi.

    split_multi=False (kural B): yalnizca tek sembollu problar kalir.
    split_multi=True  (kural A): 'A///B' problari her sembole ayri sayilir.
    AFFX kontrol problari ve eslenmeyen problar her iki kuralda da atilir.
    """
    symbols = annot["symbol"].reindex(index).fillna("")
    symbols = symbols[(symbols != "") & ~symbols.index.str.startswith("AFFX")]
    if not split_multi:
        return symbols[~symbols.str.contains("///")]
    return symbols.str.split("///").explode().str.strip()


def collapse_to_genes(log_expr: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    """Ayni gene dusen problardan ortalamasi en yuksek olani secer (etiket kullanmaz)."""
    means = log_expr.loc[mapping.index].mean(axis=1)
    best = means.groupby(mapping.values).idxmax()
    out = log_expr.loc[best.values]
    out.index = best.index
    return out.sort_index()
