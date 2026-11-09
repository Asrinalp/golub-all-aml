"""Adim 3: Betimsel istatistikler, log donusumu oncesi/sonrasi dagilimlar, sinif dagilimi."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_COLOR, INK, MUTED, TAB_DIR, save_fig, set_style, update_summary  # noqa: E402
from data_io import FLOOR, CEIL, dudoit_filter, filtered_log, load_golub, load_gse14062, log2_floor  # noqa: E402

NEUTRAL = "#7a7974"


def describe(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)),
        "skewness": float(stats.skew(values)),
        "excess_kurtosis": float(stats.kurtosis(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q99": float(np.quantile(values, 0.99)),
    }


def profile_golub(ds) -> dict:
    log = log2_floor(ds.expr)
    gene_sd = log.std(axis=1)
    return {
        "linear": describe(ds.expr.values.ravel()),
        "log2": describe(log.values.ravel()),
        "log2_filtered": describe(log.loc[dudoit_filter(ds.expr)].values.ravel()),
        "pct_at_floor": float(100 * np.mean(ds.expr.values <= FLOOR)),
        "pct_at_ceiling": float(100 * np.mean(ds.expr.values >= CEIL)),
        "probes_never_above_floor": int((ds.expr <= FLOOR).all(axis=1).sum()),
        "n_after_filter": int(len(dudoit_filter(ds.expr))),
        "gene_sd_median": float(gene_sd.median()),
        "gene_sd_q95": float(gene_sd.quantile(0.95)),
    }


def profile_gse(ds) -> dict:
    """GSE14062 RMA log2 yayimlanmistir; dogrusal olcek 2^x ile geri hesaplanir."""
    log = ds.expr
    gene_sd = log.std(axis=1)
    return {
        "linear": describe(np.power(2.0, log.values.ravel())),
        "log2": describe(log.values.ravel()),
        "log2_filtered": describe(filtered_log(ds).values.ravel()),
        "n_after_filter": int(len(filtered_log(ds))),
        "gene_sd_median": float(gene_sd.median()),
        "gene_sd_q95": float(gene_sd.quantile(0.95)),
        "array_median_min": float(log.median(axis=0).min()),
        "array_median_max": float(log.median(axis=0).max()),
    }


def plot_distributions(golub, gse) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.45), constrained_layout=True)
    panels = [
        (axes[0, 0], golub.expr.values.ravel(), "Golub: ham", "Ham ifade", (-1000, 4000)),
        (axes[0, 1], np.power(2.0, gse.expr.values.ravel()), "GSE14062: 2^x", "Doğrusal ifade", (0, 2000)),
    ]
    for ax, raw, title, xlabel, (lo, hi) in panels:
        inside = (raw >= lo) & (raw <= hi)
        ax.hist(raw[inside], bins=np.linspace(lo, hi, 61), color=NEUTRAL, weights=np.full(inside.sum(), 100 / raw.size))
        if lo < 0:
            ax.axvline(0, color=INK, lw=0.7, ls=":")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.text(0.97, 0.92, f"çarpıklık {stats.skew(raw):.1f}\ngösterilen: %{100 * inside.mean():.0f}".replace(".", ","),
                transform=ax.transAxes, ha="right", va="top", fontsize=6, color=MUTED)
    g_log = log2_floor(golub.expr).values.ravel()
    axes[1, 0].hist(g_log, bins=50, color=NEUTRAL, weights=np.full(g_log.size, 100 / g_log.size))
    axes[1, 0].set_title("Golub: log2")
    axes[1, 0].text(0.97, 0.92, f"çarpıklık {stats.skew(g_log):.1f}\ntabanda: %{100 * np.mean(golub.expr.values <= FLOOR):.0f}".replace(".", ","),
                    transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=6, color=MUTED)
    r_log = gse.expr.values.ravel()
    axes[1, 1].hist(r_log, bins=50, color=NEUTRAL, weights=np.full(r_log.size, 100 / r_log.size))
    axes[1, 1].set_title("GSE14062: RMA log2")
    axes[1, 1].text(0.97, 0.92, f"çarpıklık {stats.skew(r_log):.1f}".replace(".", ","),
                    transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=6, color=MUTED)
    for ax in axes[1, :]:
        ax.set_xlabel("log2(ifade)")
    for ax in axes.ravel():
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Değerlerin yüzdesi")
    save_fig(fig, "fig1_distributions")


def plot_class_distribution(golub, gse) -> None:
    rows = [
        ("Golub eğitim", golub.meta.query("set == 'train'")["label"].value_counts()),
        ("Golub test", golub.meta.query("set == 'test'")["label"].value_counts()),
        ("GSE14062", gse.meta["label"].value_counts()),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 1.15), constrained_layout=True)
    order = ["ALL", "AML"]
    for y, (_, counts) in enumerate(rows):
        total = counts.sum()
        left = 0.0
        for cls in order:
            share = 100 * counts[cls] / total
            ax.barh(y, share - 0.5, left=left + 0.25, color=CLASS_COLOR[cls], height=0.62)
            ax.text(left + share / 2, y, f"{int(counts[cls])}", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            left += share
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Örneklerin yüzdesi (çubuk içinde örnek sayısı)")
    ax.grid(axis="y", visible=False)
    handles = [Patch(color=CLASS_COLOR[c], label=c) for c in order]
    ax.legend(handles=handles, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.2)
    save_fig(fig, "fig2_class_distribution")


def main() -> None:
    set_style()
    golub, gse = load_golub(), load_gse14062()
    plot_distributions(golub, gse)
    plot_class_distribution(golub, gse)
    summary = {"golub": profile_golub(golub), "gse14062": profile_gse(gse)}

    rows = []
    for tag, part in summary.items():
        for scale in ["linear", "log2"]:
            rows.append({"veri_seti": tag, "olcek": scale, **part[scale]})
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TAB_DIR / "descriptive_stats.csv", index=False)
    update_summary("distributions", summary)
    print("Dagilim adimi tamam.")


if __name__ == "__main__":
    main()
