"""Adim 1: Ham verileri indirir ve SHA-256 ozetlerini kaydeder.

Ham veri depoya konmaz; bu betik her seferinde ayni dosyalari
yeniden indirir ve ozetleri data/raw/MANIFEST.json dosyasina yazar.

"""
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW_DIR  # noqa: E402

BROAD_GOLUB = "https://pubs.broadinstitute.org/mpr/projects/Leukemia"
BROAD_ARMSTRONG = "https://pubs.broadinstitute.org/mpr/projects/MLL"
GEO_ANNOT = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/{gpl}/annot/{gpl}.annot.gz"
GEO_QUERY = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
RETRY_WAITS = [30, 60, 120, 0]
HEADERS = {"User-Agent": "golub-all-aml-eda/1.0 (academic course project)"}

SOURCES = {
    "golub_kaggle/gene-expression.zip": "https://www.kaggle.com/api/v1/datasets/download/crawford/gene-expression",
    "golub_broad/data_set_ALL_AML_train.txt": f"{BROAD_GOLUB}/data_set_ALL_AML_train.txt",
    "golub_broad/data_set_ALL_AML_independent.txt": f"{BROAD_GOLUB}/data_set_ALL_AML_independent.txt",
    "golub_broad/table_ALL_AML_samples.txt": f"{BROAD_GOLUB}/table_ALL_AML_samples.txt",
    "golub_broad/Files_descriptions.txt": f"{BROAD_GOLUB}/Files_descriptions.txt",
    # Elenen aday: yalnizca hasta ortusmesi denetimi icin (step7_overlap.py)
    "armstrong/expression_data.txt": f"{BROAD_ARMSTRONG}/expression_data.txt",
    "annot/GPL80.annot.gz": GEO_ANNOT.format(gpl="GPL80"),
    "annot/GPL91.annot.gz": GEO_ANNOT.format(gpl="GPL91"),
    "annot/GPL570.annot.gz": GEO_ANNOT.format(gpl="GPL570"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"  mevcut: {target.relative_to(RAW_DIR)}")
        return
    print(f"  indiriliyor: {target.relative_to(RAW_DIR)}")
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt, wait in enumerate(RETRY_WAITS):
        try:
            with requests.get(url, stream=True, timeout=300, headers=HEADERS) as r:
                if r.status_code == 429 and attempt < len(RETRY_WAITS) - 1:
                    print(f"    hiz siniri (429); {wait} sn bekleniyor")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                # iter_content, sunucunun gzip kodlamasini cozer (r.raw cozmez).
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            break
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            if attempt == len(RETRY_WAITS) - 1:
                raise
            print(f"    baglanti kesildi ({type(e).__name__}); {wait} sn bekleniyor")
            time.sleep(wait)
    tmp.replace(target)


def safe_unzip(zip_path: Path, out_dir: Path) -> None:
    """ZIP icindeki yollarin hedef klasorun disina cikmadigini denetleyerek acar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            dest = (out_dir / member).resolve()
            if root not in dest.parents and dest != root:
                raise ValueError(f"Guvensiz ZIP yolu: {member}")
        z.extractall(out_dir)


def gunzip(path: Path) -> Path:
    out = path.with_suffix("")
    if not out.exists():
        with gzip.open(path, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


def geo_text(params: dict, tries: int = 5) -> str:
    for attempt in range(tries):
        try:
            r = requests.get(GEO_QUERY, params={"form": "text", **params}, headers=HEADERS, timeout=180)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            print(f"    yeniden deneniyor ({type(e).__name__})")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("erisilemiyor")


def download_gse14062() -> None:
    """GSE14062 ifade matrisini ve ornek bilgilerini GSM kayitlarindan olusturur."""
    out = RAW_DIR / "gse14062"
    matrix, samples = out / "expression.tsv.gz", out / "samples.tsv"
    if matrix.exists() and samples.exists():
        print("  mevcut: gse14062")
        return
    out.mkdir(parents=True, exist_ok=True)
    rows = gse14062_sample_rows()
    print(f"  GSE14062: {len(rows)} ornek bilgisi alindi; degerler indiriliyor")
    probes, columns = None, {}
    for k, row in enumerate(rows, 1):
        text = geo_text({"acc": row["gsm"], "targ": "self", "view": "data"})
        body = text.split("!sample_table_begin", 1)[1].split("!sample_table_end", 1)[0].strip().splitlines()
        if not body[0].startswith("ID_REF"):
            raise ValueError(f"{row['gsm']}: beklenmeyen tablo basligi {body[0]}")
        pairs = [line.split("\t")[:2] for line in body[1:]]
        ids = [p[0] for p in pairs]
        if probes is None:
            probes = ids
        elif ids != probes:
            raise ValueError(f"{row['gsm']}: prob sirasi farkli")
        columns[row["gsm"]] = [p[1] for p in pairs]
        if k % 20 == 0:
            print(f"    {k}/{len(rows)}", flush=True)
        time.sleep(0.34)
    tmp = matrix.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
        f.write("ID_REF\t" + "\t".join(columns) + "\n")
        for i, probe in enumerate(probes):
            f.write(probe + "\t" + "\t".join(columns[g][i] for g in columns) + "\n")
    tmp.replace(matrix)
    write_sample_table(rows, samples)


def gse14062_sample_rows() -> list:
    """GEO metni CRLF satir sonu kullanir; alanlar bu yuzden temizlenerek okunur."""
    brief = geo_text({"acc": "GSE14062", "targ": "gsm", "view": "brief"}).replace("\r", "")
    rows = []
    for rec in re.split(r"(?m)^\^SAMPLE = ", brief)[1:]:
        field = lambda key: [v.strip() for v in re.findall(rf"(?m)^!Sample_{key} = (.+)$", rec)]
        rows.append({
            "gsm": rec.splitlines()[0].strip(),
            "title": field("title")[0],
            "source": field("source_name_ch1")[0],
            "characteristics": "; ".join(field("characteristics_ch1")),
            "platform": field("platform_id")[0],
        })
    return rows


def write_sample_table(rows: list, path: Path) -> None:
    columns = ["gsm", "title", "source", "characteristics", "platform"]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(row[c] for c in columns) + "\n")


def main() -> None:
    for rel, url in SOURCES.items():
        download(url, RAW_DIR / rel)
    safe_unzip(RAW_DIR / "golub_kaggle/gene-expression.zip", RAW_DIR / "golub_kaggle")
    for gz in (RAW_DIR / "annot").glob("*.gz"):
        gunzip(gz)
    download_gse14062()
    manifest = {rel: {"url": url, "sha256": sha256(RAW_DIR / rel)} for rel, url in SOURCES.items()}
    manifest["gse14062/expression.tsv.gz"] = {"url": f"{GEO_QUERY}?acc=GSE14062", "sha256": sha256(RAW_DIR / "gse14062/expression.tsv.gz")}
    (RAW_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Tamam: {len(manifest)} dosya; ozetler data/raw/MANIFEST.json dosyasinda.")


if __name__ == "__main__":
    main()
