"""Adim 5: Ayirt edici genler, isi haritasi ve hiyerarsik kumeleme.

Golub icin gen siralamasi YALNIZCA 38 egitim orneginden yapilir; 34 test ornegi bu adimda
kullanilmaz. Disaridan sinama icin GSE14062 kullanilir: Golub egitim genleri, GSE14062
etiketlerine hic bakilmadan bu veri setine tasinir (imza skoru ve kumeleme).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_COLOR, CLASS_MARKER, MUTED, SEED, TAB_DIR, save_fig, set_style, update_summary  # noqa: E402
from data_io import (  # noqa: E402
    collapse_to_genes,
    dudoit_filter,
    filtered_log,
    load_annotation,
    load_golub,
    load_gse14062,
    log2_floor,
    probe_to_symbol,
)

N_PERMUTATIONS = 200
HEAT_CMAP = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])


def group_test(log: pd.DataFrame, group_a: np.ndarray, group_b: np.ndarray) -> pd.DataFrame:
    """Her prob icin Welch t ve Mann-Whitney U; Benjamini-Hochberg duzeltmesi."""
    a, b = log.values[:, group_a], log.values[:, group_b]
    t, p = stats.ttest_ind(a, b, axis=1, equal_var=False)
    t, p = np.nan_to_num(t), np.nan_to_num(p, nan=1.0)  # iki grupta da sabit problar
    _, p_u = stats.mannwhitneyu(a, b, axis=1, alternative="two-sided")
    out = pd.DataFrame({
        "log2_fc": a.mean(axis=1) - b.mean(axis=1),
        "t": t,
        "p_t": p,
        "q_t": multipletests(p, method="fdr_bh")[1],
        "p_u": p_u,
        "q_u": multipletests(p_u, method="fdr_bh")[1],
    }, index=log.index)
    return out.sort_values("t", key=np.abs, ascending=False)


def permutation_null(log: pd.DataFrame, is_a: np.ndarray, rng: np.random.Generator) -> dict:
    counts = []
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(is_a)
        _, p = stats.ttest_ind(log.values[:, perm], log.values[:, ~perm], axis=1, equal_var=False)
        counts.append(int((multipletests(np.nan_to_num(p, nan=1.0), method="fdr_bh")[1] < 0.05).sum()))
    return {
        "n_permutations": N_PERMUTATIONS,
        "median_sig_genes_q05": float(np.median(counts)),
        "q95_sig_genes_q05": float(np.quantile(counts, 0.95)),
        "max_sig_genes_q05": int(np.max(counts)),
    }


def summarise_tests(res: pd.DataFrame) -> dict:
    return {
        "n_tested": int(len(res)),
        "q05_t": int((res["q_t"] < 0.05).sum()),
        "q01_t": int((res["q_t"] < 0.01).sum()),
        "q05_u": int((res["q_u"] < 0.05).sum()),
        "q05_t_and_fc1": int(((res["q_t"] < 0.05) & (res["log2_fc"].abs() > 1)).sum()),
    }


def named(res: pd.DataFrame, n: int) -> list:
    """Ilk n probu gen sembolleriyle; sembolu olmayanlar prob kimligiyle."""
    return [s if isinstance(s, str) else p for p, s in res["symbol"].head(n).items()]


def floor_sensitivity(train_raw: pd.DataFrame, is_all: np.ndarray) -> dict:
    """Rehberdeki 20 tabani ile Dudoit'in 100 tabani benzer gen listesi veriyor mu?"""
    out, tops = {}, {}
    for floor in (20, 100):
        log = log2_floor(train_raw, floor=floor).loc[dudoit_filter(train_raw, floor=floor)]
        res = group_test(log, is_all, ~is_all)
        out[str(floor)] = {"n_after_filter": int(len(log)), "q05_t": int((res["q_t"] < 0.05).sum())}
        tops[floor] = set(res.index[:50])
    out["top50_overlap"] = len(tops[20] & tops[100])
    return out


def balanced_top(res: pd.DataFrame, n: int) -> list:
    """Her yonden esit sayida gen: ALL'de yuksek ve AML'de yuksek."""
    up = res[res["t"] > 0].index[: n // 2]
    down = res[res["t"] < 0].index[: n // 2]
    return list(up) + list(down)


def two_clusters(z: np.ndarray) -> np.ndarray:
    return fcluster(linkage(z.T, method="average", metric="correlation"), t=2, criterion="maxclust")


def cluster_ari(z: np.ndarray, labels: np.ndarray) -> float:
    return float(adjusted_rand_score(labels, two_clusters(z)))


def aml_cluster_counts(z: np.ndarray, is_all: np.ndarray) -> dict:
    """AML'lerin cogunlukta oldugu kumeye giren ALL ve bu kumenin disinda kalan AML sayisi."""
    cut = two_clusters(z)
    aml_cluster = pd.Series(cut[is_all == 0]).mode().iloc[0]
    return {
        "all_in_aml_cluster": int(((cut == aml_cluster) & (is_all == 1)).sum()),
        "aml_outside_aml_cluster": int(((cut != aml_cluster) & (is_all == 0)).sum()),
    }


def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1, ddof=1), axis=0)


def heatmap(fig, cell, frame: pd.DataFrame, classes: pd.Series, gene_names: list, title: str, n_up: int):
    """Satirlar z-skor; sutunlar ortalama bagli/korelasyon uzakligiyla kumelenir."""
    z = zscore_rows(frame)
    link = linkage(z.values.T, method="average", metric="correlation")
    grid = GridSpecFromSubplotSpec(3, 1, subplot_spec=cell, height_ratios=[0.5, 0.13, 4.0], hspace=0.03)
    ax_top = fig.add_subplot(grid[0])
    den = dendrogram(link, ax=ax_top, no_labels=True, color_threshold=0, above_threshold_color=MUTED)
    for coll in ax_top.collections:
        coll.set_linewidth(0.5)
    ax_top.axis("off")
    ax_top.set_title(title, pad=2)
    order = den["leaves"]

    ax_strip = fig.add_subplot(grid[1])
    colours = [CLASS_COLOR[classes.iloc[i]] for i in order]
    ax_strip.bar(range(len(order)), 1, width=1.0, color=colours, align="edge")
    ax_strip.set_xlim(0, len(order))
    ax_strip.axis("off")

    ax_heat = fig.add_subplot(grid[2])
    im = ax_heat.pcolormesh(z.values[:, order], cmap=HEAT_CMAP, vmin=-2.5, vmax=2.5,
                            edgecolors="none", antialiased=False, rasterized=False)
    ax_heat.invert_yaxis()
    ax_heat.grid(False)
    ax_heat.set_yticks(np.arange(len(gene_names)) + 0.5, gene_names, fontsize=6.3)
    ax_heat.set_xticks([])
    ax_heat.tick_params(length=0, pad=1.5)
    ax_heat.axhline(n_up, color="white", lw=1.6)
    ax_heat.set_xlabel(f"{frame.shape[1]} örnek (dendrogram sırasıyla)", fontsize=6.8)
    for side in ax_heat.spines.values():
        side.set_visible(False)
    return im


def signature_panel(ax, score: pd.Series, labels: pd.Series, mll: pd.Series, auc: float, title: str):
    """Her GSE14062 ornegi bir nokta: Golub imzasinin skoru, sinifa gore; bos isaret MLL'li."""
    jitter = np.random.default_rng(SEED)  # yalnizca nokta yerlesimi; analizdeki rng'yi etkilemez
    for x, cls in enumerate(["ALL", "AML"]):
        for filled in (True, False):
            sel = ((labels == cls) & (mll != filled)).values
            ax.scatter(x + jitter.uniform(-0.22, 0.22, sel.sum()), score.values[sel], s=11, marker=CLASS_MARKER[cls],
                       facecolors=CLASS_COLOR[cls] if filled else "none",
                       edgecolors="white" if filled else CLASS_COLOR[cls],
                       linewidths=0.4 if filled else 0.9, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.6, ls=":")
    counts = labels.value_counts()
    ax.set_xticks([0, 1], [f"ALL\n({counts['ALL']})", f"AML\n({counts['AML']})"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("İmza skoru (z)")
    ax.set_title(title, pad=3)
    ax.text(0.97, 0.97, f"ROC-AUC\n{auc:.3f}".replace(".", ","), transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.grid(axis="x", visible=False)


def main() -> None:
    set_style()
    rng = np.random.default_rng(SEED)
    golub, gse = load_golub(), load_gse14062()
    annot_g, annot_r = load_annotation("GPL80"), load_annotation("GPL570")

    # --- Golub: yalnizca egitim ornekleri ---
    train_cols = golub.meta.index[golub.meta["set"] == "train"]
    g_train_raw = golub.expr[train_cols]
    g_train = log2_floor(g_train_raw).loc[dudoit_filter(g_train_raw)]
    g_is_all = (golub.meta.loc[train_cols, "label"] == "ALL").values
    g_res = group_test(g_train, g_is_all, ~g_is_all)

    # --- GSE14062: tum ornekler ve MLL'siz ornekler ---
    r_log = filtered_log(gse)
    labels = gse.meta["label"]
    r_is_all = (labels == "ALL").values
    r_res = group_test(r_log, r_is_all, ~r_is_all)
    no_mll = (~gse.meta["mll"]).values
    r_res_nomll = group_test(r_log.loc[:, no_mll], (labels == "ALL").values[no_mll], (labels == "AML").values[no_mll])

    sym_g = probe_to_symbol(golub.expr.index, annot_g)
    sym_r = probe_to_symbol(gse.expr.index, annot_r)
    g_res.insert(0, "symbol", sym_g.reindex(g_res.index))
    r_res.insert(0, "symbol", sym_r.reindex(r_res.index))
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    g_res.head(50).to_csv(TAB_DIR / "top50_golub_train.csv")
    r_res.head(50).to_csv(TAB_DIR / "top50_gse14062_all_vs_aml.csv")

    # --- Golub egitim genleri -> GSE14062 gen duzeyi (etiket gormeden) ---
    r_genes = collapse_to_genes(gse.expr, sym_r)
    shared_res = g_res.dropna(subset=["symbol"]).drop_duplicates("symbol")
    shared_res = shared_res[shared_res["symbol"].isin(r_genes.index)].set_index("symbol")
    binary = r_is_all.astype(int)
    transport, scores = {}, {}
    for k in (20, 30, 50, 100):
        genes = balanced_top(shared_res, k)
        z = zscore_rows(r_genes.loc[genes]).values
        z_nomll = zscore_rows(r_genes.loc[genes, no_mll]).values
        # Imza skoru: Golub'daki yonlere gore (ALL'de yuksek - AML'de yuksek) ortalama z-skoru.
        # Model egitimi degildir; gen listesi ve yonler yalnizca Golub egitiminden gelir.
        up = genes[: k // 2]
        down = genes[k // 2:]
        score = zscore_rows(r_genes.loc[up]).mean(axis=0) - zscore_rows(r_genes.loc[down]).mean(axis=0)
        scores[k] = score
        transport[str(k)] = {
            "ari_all_samples": cluster_ari(z, binary),
            "ari_without_mll": cluster_ari(z_nomll, binary[no_mll]),
            "signature_auc_all": float(roc_auc_score(binary, score)),
            "signature_auc_without_mll": float(roc_auc_score(binary[no_mll], score[no_mll])),
            **aml_cluster_counts(z, binary),
        }

    # --- Etiketsiz kumeleme ---
    g_all_log = filtered_log(golub)
    hv_g = g_all_log.var(axis=1).nlargest(500).index
    hv_r = r_log.var(axis=1).nlargest(500).index
    unsup = {
        "golub_all72_ari": cluster_ari(zscore_rows(g_all_log.loc[hv_g]).values, (golub.meta["label"] == "ALL").values.astype(int)),
        "gse14062_ari": cluster_ari(zscore_rows(r_log.loc[hv_r]).values, binary),
        "gse14062_ari_without_mll": cluster_ari(zscore_rows(r_log.loc[hv_r, no_mll]).values, binary[no_mll]),
    }

    # --- Sekil: (a)-(b) isi haritasi ve kumeleme, (c) Golub imzasinin GSE14062'deki skoru ---
    n_show = 20  # rehber: en iyi 20-50 gen
    name = lambda res, p: res.loc[p, "symbol"] if isinstance(res.loc[p, "symbol"], str) else p  # noqa: E731
    fig = plt.figure(figsize=(7.1, 3.55))
    outer = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.035, 0.1, 0.78], wspace=0.62,
                             left=0.075, right=0.985, top=0.885, bottom=0.12)
    g_top = balanced_top(g_res, n_show)
    heatmap(fig, outer[0, 0], g_train.loc[g_top], golub.meta.loc[train_cols, "label"],
            [name(g_res, p) for p in g_top], f"(a) Golub eğitim: en iyi {n_show} gen", n_show // 2)
    r_top = balanced_top(r_res, n_show)
    im = heatmap(fig, outer[0, 1], r_log.loc[r_top], labels,
                 [name(r_res, p) for p in r_top], f"(b) GSE14062: en iyi {n_show} prob", n_show // 2)
    cell = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, 2], height_ratios=[0.6, 2.4, 0.6])
    cbar = fig.colorbar(im, cax=fig.add_subplot(cell[1]))
    cbar.set_label("Gen bazında z-skoru", fontsize=6.8)
    cbar.ax.tick_params(labelsize=6)
    signature_panel(fig.add_subplot(outer[0, 4]), scores[n_show], labels, gse.meta["mll"],
                    transport[str(n_show)]["signature_auc_all"], "(c) Golub imzası → GSE14062")
    handles = [Line2D([], [], marker="s", ls="", color=CLASS_COLOR[c], label=c) for c in ["ALL", "AML"]]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=MUTED, label="(c) boş: MLL'li"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncols=3, handletextpad=0.2, columnspacing=1.0)
    save_fig(fig, "fig4_heatmaps")

    update_summary("genes", {
        "golub_train": summarise_tests(g_res),
        "golub_train_permutation": permutation_null(g_train, g_is_all, rng),
        "gse14062_all_vs_aml": summarise_tests(r_res),
        "gse14062_without_mll": summarise_tests(r_res_nomll),
        "gse14062_permutation": permutation_null(r_log, r_is_all, rng),
        "golub_train_top10": named(g_res, 10),
        "gse14062_top10": named(r_res, 10),
        "golub_top_up_in_all": named(g_res[g_res["t"] > 0], 8),
        "golub_top_up_in_aml": named(g_res[g_res["t"] < 0], 8),
        "gse14062_top_up_in_all": named(r_res[r_res["t"] > 0], 8),
        "gse14062_top_up_in_aml": named(r_res[r_res["t"] < 0], 8),
        "floor_sensitivity_golub_train": floor_sensitivity(g_train_raw, g_is_all),
        "transport_golub_genes_to_gse14062": transport,
        "n_golub_train_genes_shared_with_gse14062": int(len(shared_res)),
        "unsupervised_top500_variance": unsup,
    })
    print("Gen adimi tamam.")


if __name__ == "__main__":
    main()
