"""OpenAlex uzerinden IEEE yayinlarinda Golub ALL/AML taramasi (Ek B icin)."""
import json
from pathlib import Path
import re
import sys

import time

import requests

sys.stdout.reconfigure(encoding="utf-8")
H = {"User-Agent": "golub-all-aml-coursework"}
B = "https://api.openalex.org/works"
IEEE = "P4310319808"
QUERIES = {
    "IEEE-1": '(leukemia OR leukaemia) AND ("gene expression" OR microarray) AND ("feature selection" OR "gene selection" OR classification)',
    "IEEE-2": '(leukemia OR leukaemia) AND ("deep learning" OR "neural network" OR "convolutional" OR transformer) AND ("gene expression" OR microarray)',
    "IEEE-3": '(Golub OR "ALL/AML" OR "ALL-AML" OR "AML/ALL") AND (leukemia OR leukaemia)',
}
GOLUB = re.compile(r"golub|7129|7,129|72 (samples|patients)|all[/ -]aml|aml[/ -]all|47 all|25 aml", re.I)


def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((i, word) for word, idx in inv.items() for i in idx)
    return " ".join(word for _, word in pos)


out = {}
for key, q in QUERIES.items():
    for years in ("1999-2020", "2021-2026"):
        flt = f"title_and_abstract.search:{q},publication_year:{years},primary_location.source.publisher_lineage:{IEEE}"
        rows, cursor = [], "*"
        while cursor:
            for attempt in range(6):
                r = requests.get(B, params={"filter": flt, "per-page": 200, "cursor": cursor,
                                            "select": "id,doi,title,publication_year,type,primary_location,abstract_inverted_index,open_access"},
                                 headers=H, timeout=60).json()
                if "results" in r:
                    break
                time.sleep(3 * (attempt + 1))
            rows += r["results"]
            cursor = r["meta"].get("next_cursor") if r["results"] else None
        total = r["meta"]["count"]
        has_abs = sum(1 for w in rows if w.get("abstract_inverted_index"))
        golub = []
        for w in rows:
            text = (w.get("title") or "") + " " + abstract(w)
            if GOLUB.search(text):
                golub.append({"year": w["publication_year"], "doi": w.get("doi"), "title": w.get("title"),
                              "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                              "oa": (w.get("open_access") or {}).get("is_oa"),
                              "hit": GOLUB.search(text).group(0)})
        out[f"{key} {years}"] = {"query": q, "years": years, "count": total, "with_abstract": has_abs, "golub_in_abstract": golub}
        print(f"{key} {years}: {total} kayit ({has_abs} ozetli), baslik/ozette Golub izi {len(golub)}")
        time.sleep(1)
json.dump(out, open(Path(__file__).with_name("openalex_ieee.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in out.items():
    for g in v["golub_in_abstract"]:
        print(k, g["year"], g["oa"], g["hit"], "|", (g["venue"] or "")[:40], "|", (g["title"] or "")[:90], g["doi"])
