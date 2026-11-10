"""Tum ara rapor EDA ciktilarini tek komutla yeniden uretir.

Kullanim:
    python run_all.py                 # veriyi indirir, tum tablo ve sekilleri uretir
    python run_all.py --skip-download # data/raw zaten varsa indirmeyi atlar
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Veri indirme", "step1_download.py"),
    ("Veri kalitesi ve etiket dogrulama", "step2_quality.py"),
    ("Betimsel istatistik ve dagilimlar", "step3_distributions.py"),
    ("PCA ve t-SNE", "step4_dimred.py"),
    ("Ayirt edici genler ve isi haritalari", "step5_genes.py"),
    ("Egitim-test ve veri setleri arasi kayma", "step6_shift.py"),
    ("Veri setleri arasi hasta ortusmesi", "step7_overlap.py"),
    ("MILE ortusmesi (886 MB indirir)", "step8_mile_overlap.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="data/raw klasorunu yeniden indirme")
    parser.add_argument("--skip-mile", action="store_true", help="MILE indirme ve ortusme adimini atla")
    args = parser.parse_args()
    for title, script in STEPS:
        if args.skip_download and script == "step1_download.py":
            continue
        if args.skip_mile and script == "step8_mile_overlap.py":
            continue
        print(f"\n=== {title} ({script})", flush=True)
        start = time.time()
        subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "src" / script)], check=True, cwd=ROOT)
        print(f"    {time.time() - start:.1f} sn", flush=True)
    print("\nBitti. Ciktilar: results/figures, results/tables, results/eda_summary.json")


if __name__ == "__main__":
    main()
