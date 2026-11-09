"""Tum adimlarin ortak yollari, sabit tohumu ve sekil stili."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"
SUMMARY = RESULTS / "eda_summary.json"

SEED = 42
SEEDS = [42, 7, 123, 2024, 31337]  # t-SNE kararlilik kontrolu icin

# dataviz referans paleti: ilk uc kategorik slot tum ciftlerde dogrulandi.
ALL_COLOR = "#2a78d6"
AML_COLOR = "#eb6834"
MLL_COLOR = "#1baf7a"
CLASS_COLOR = {"ALL": ALL_COLOR, "AML": AML_COLOR, "MLL": MLL_COLOR}
CLASS_MARKER = {"ALL": "o", "AML": "s", "MLL": "^"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "svg.fonttype": "path",
        "svg.hashsalt": "golub-all-aml",  # ayni kod ayni dosyayi uretsin
        "image.composite_image": True,
    })


def save_fig(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Tarih metaverisi yazilmaz: ayni kod her calistirmada bayt bayt ayni dosyayi uretir
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.svg", metadata={"Date": None})  # rapora vektor olarak gomulur
    fig.savefig(FIG_DIR / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def update_summary(section: str, values: dict) -> None:
    """Rapordaki sayilarin tek kaynagi: results/eda_summary.json."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    data[section] = values
    SUMMARY.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
