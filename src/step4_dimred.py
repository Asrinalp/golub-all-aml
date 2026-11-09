"""Adim 4: PCA (aciklanan varyans dahil) ve t-SNE; PC-ortak degisken iliskisi.
   (Keşif Amaçlı)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_samples, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_COLOR, CLASS_MARKER, MUTED, SEED, SEEDS, TAB_DIR, save_fig, set_style, update_summary  # noqa: E402
from data_io import filtered_log, load_golub, load_gse14062  # noqa: E402

PERPLEXITY = {"Golub": 15, "GSE14062": 30}


def zscore_genes(log_expr: pd.DataFrame) -> np.ndarray:
    x = log_expr.values
    return ((x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, ddof=1, keepdims=True)).T


def eta_squared(values: np.ndarray, groups: pd.Series) -> float:
    groups = pd.Series(np.asarray(groups))
    grand = values.mean()
    between = sum(len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2 for g in groups.unique())
    return float(between / ((values - grand) ** 2).sum())


def tsne(x: np.ndarray, perplexity: int, seed: int, init: str = "pca") -> np.ndarray:
    return TSNE(n_components=2, perplexity=perplexity, init=init, learning_rate="auto",
                random_state=seed, max_iter=2000).fit_transform(x)


def analyse(ds) -> dict:
    log = filtered_log(ds)
    x = zscore_genes(log)
    pca = PCA(random_state=SEED).fit(x)
    scores = pca.transform(x)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    labels = ds.meta["label"].values
    perplexity = PERPLEXITY[ds.name]

    top_pcs = scores[:, :10]
    # Sekil PCA baslatmasiyla uretilir; o deterministik oldugu icin kararlilik
    # olcumu rastgele baslatma ve bes farkli tohumla yapilir.
    embedding = tsne(scores[:, :20], perplexity, SEED)
    sil_by_seed = {s: float(silhouette_score(tsne(scores[:, :20], perplexity, s, init="random"), labels)) for s in SEEDS}
    qc = pd.read_csv(TAB_DIR / f"array_qc_{ds.name.lower()}.csv", index_col=0)
    qc.index = qc.index.astype(str)
    qc = qc.loc[ds.expr.columns.astype(str)]

    out = {
        "n_genes_used": int(log.shape[0]),
        "perplexity": perplexity,
        "evr_pc1": float(evr[0]),
        "evr_pc2": float(evr[1]),
        "evr_pc3": float(evr[2]),
        "pcs_for_50pct": int(np.searchsorted(cum, 0.5) + 1),
        "pcs_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
        "pcs_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
        "silhouette_pc2": float(silhouette_score(scores[:, :2], labels)),
        "silhouette_pc10": float(silhouette_score(top_pcs, labels)),
        "silhouette_tsne_pca_init": float(silhouette_score(embedding, labels)),
        "silhouette_tsne_random_init_mean": float(np.mean(list(sil_by_seed.values()))),
        "silhouette_tsne_random_init_sd": float(np.std(list(sil_by_seed.values()), ddof=1)),
        "tsne_trustworthiness": float(trustworthiness(scores[:, :20], embedding, n_neighbors=10)),
        "eta2_class_pc1": eta_squared(scores[:, 0], ds.meta["label"]),
        "eta2_class_pc2": eta_squared(scores[:, 1], ds.meta["label"]),
        "eta2_class_pc3": eta_squared(scores[:, 2], ds.meta["label"]),
        "corr_pc1_median_raw": float(np.corrcoef(scores[:, 0], qc["median_raw"])[0, 1]),
    }
    if "mll" in ds.meta:
        out["eta2_mll_pc1"] = eta_squared(scores[:, 0], ds.meta["mll"])
        out["eta2_mll_pc2"] = eta_squared(scores[:, 1], ds.meta["mll"])
    sil = pd.Series(silhouette_samples(top_pcs, labels), index=ds.expr.columns)
    out["negative_silhouette_samples"] = [str(s) for s in sil.index[sil < 0]]
    if "mll" in ds.meta:
        groups = ds.meta["label"] + ds.meta["mll"].map({True: "_MLL", False: ""})
        out["negative_silhouette_by_group"] = {k: f"{int((sil[groups == k] < 0).sum())}/{int((groups == k).sum())}" for k in groups.unique()}
    return {"summary": out, "scores": scores, "cum": cum, "evr": evr, "tsne": embedding}


def scatter(ax, xy, ds, xlabel, ylabel, title):
    meta = ds.meta
    # Dolu/bos ayrimi: Golub'da egitim/test, GSE14062'de MLL yok/var.
    filled_mask = (meta["set"] == "train").values if "set" in meta else (~meta["mll"]).values
    for cls in ["ALL", "AML"]:
        mask = (meta["label"] == cls).values
        for filled, group in [(True, mask & filled_mask), (False, mask & ~filled_mask)]:
            ax.scatter(xy[group, 0], xy[group, 1], s=20 if ds.name == "Golub" else 14, marker=CLASS_MARKER[cls],
                       facecolors=CLASS_COLOR[cls] if filled else "none",
                       edgecolors="white" if filled else CLASS_COLOR[cls],
                       linewidths=0.5 if filled else 1.0, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def pc_covariates(golub, scores) -> pd.DataFrame:
    meta = golub.meta
    rows = {}
    for name, column in [("Sınıf (ALL/AML)", "label"), ("Soy (B/T/AML)", "lineage"), ("Küme (eğitim/test)", "set"),
                         ("Doku (BM/PB)", "tissue"), ("Kaynak kurum", "source")]:
        rows[name] = {f"PC{i + 1}": eta_squared(scores[:, i], meta[column]) for i in range(5)}
    return pd.DataFrame(rows).T


def main() -> None:
    set_style()
    golub, gse = load_golub(), load_gse14062()
    g, r = analyse(golub), analyse(gse)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.65), constrained_layout=True)
    for row, (ds, res) in enumerate([(golub, g), (gse, r)]):
        evr = res["evr"]
        scatter(axes[row, 0], res["scores"], ds, f"PC1 (%{100 * evr[0]:.1f})", f"PC2 (%{100 * evr[1]:.1f})", f"{ds.name}: PCA")
        scatter(axes[row, 1], res["tsne"], ds, "t-SNE 1", "t-SNE 2", f"{ds.name}: t-SNE (perplexity {PERPLEXITY[ds.name]})")
        axes[row, 1].tick_params(labelbottom=False, labelleft=False)
        ax = axes[row, 2]
        x = np.arange(1, len(res["cum"]) + 1)
        ax.plot(x, 100 * res["cum"], color=MUTED, lw=2)
        for level, key in [(50, "pcs_for_50pct"), (80, "pcs_for_80pct")]:
            n = res["summary"][key]
            ax.scatter([n], [100 * res["cum"][n - 1]], s=24, color=MUTED, zorder=3)
            ax.annotate(f"%{level}: {n} PC", (n, 100 * res["cum"][n - 1]), xytext=(6, -8),
                        textcoords="offset points", fontsize=6.5, color=MUTED)
        ax.set_xlabel("Temel bileşen sayısı")
        ax.set_ylabel("Birikimli varyans (%)")
        ax.set_title(f"{ds.name}: açıklanan varyans")
        ax.set_ylim(0, 102)
    handles = [
        Line2D([], [], marker=CLASS_MARKER["ALL"], ls="", color=CLASS_COLOR["ALL"], label="ALL"),
        Line2D([], [], marker=CLASS_MARKER["AML"], ls="", color=CLASS_COLOR["AML"], label="AML"),
        Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, label="dolu: Golub eğitim / GSE14062 MLL'siz"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=MUTED, label="boş: Golub test / GSE14062 MLL'li"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=4, handletextpad=0.2, columnspacing=1.4)
    save_fig(fig, "fig3_pca_tsne")

    covariates = pc_covariates(golub, g["scores"])
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    covariates.to_csv(TAB_DIR / "golub_pc_covariates.csv")
    update_summary("dimred", {
        "golub": g["summary"],
        "gse14062": r["summary"],
        "golub_pc_eta_squared": covariates.round(4).to_dict(orient="index"),
    })
    print("Boyut indirgeme adimi tamam.")


if __name__ == "__main__":
    main()
