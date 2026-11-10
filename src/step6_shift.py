"""Adim 6: Golub egitim-test kaymasi (batch) ve Golub-GSE14062 dagilim farklari.

Test etiketleri burada yalnizca betimsel olarak kullanilir: egitim-test farkini sinif
bilesiminden ayirmak icin. Hicbir gen listesi, olcekleme parametresi veya model karari
test setinden turetilmez.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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
from step4_dimred import eta_squared  # noqa: E402

NEUTRAL = "#7a7974"


def set_effect(log: pd.DataFrame, is_test: np.ndarray, covariates: np.ndarray) -> pd.DataFrame:
    """Her gen icin y ~ ortak degiskenler + kume modelinde kume katsayisinin t testi."""
    design = np.column_stack([np.ones(len(is_test)), covariates, is_test.astype(float)])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ log.values.T
    resid = log.values.T - design @ beta
    dof = design.shape[0] - design.shape[1]
    sigma2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(sigma2 * xtx_inv[-1, -1])
    t = np.nan_to_num(beta[-1] / se)
    p = 2 * stats.t.sf(np.abs(t), dof)
    return pd.DataFrame({"beta": beta[-1], "t": t, "p": p, "q": multipletests(p, method="fdr_bh")[1]}, index=log.index)


def batch_summary(golub) -> tuple[dict, dict]:
    meta = golub.meta
    log = filtered_log(golub)
    is_test = (meta["set"] == "test").values
    is_aml = (meta["label"] == "AML").values.astype(float)

    adjusted = set_effect(log, is_test, is_aml)
    unadjusted = set_effect(log, is_test, np.empty((len(is_test), 0)))
    bm = (meta["tissue"] == "BM").values
    adjusted_bm = set_effect(log.loc[:, bm], is_test[bm], is_aml[bm])
    all_only = (meta["label"] == "ALL").values
    all_b_bm = all_only & bm & (meta["lineage"] == "B-ALL").values
    within_all = set_effect(log.loc[:, all_b_bm], is_test[all_b_bm], np.empty((all_b_bm.sum(), 0)))
    aml_only = (meta["label"] == "AML").values & bm
    within_aml = set_effect(log.loc[:, aml_only], is_test[aml_only], np.empty((aml_only.sum(), 0)))

    qc = pd.read_csv(TAB_DIR / "array_qc_golub.csv", index_col=0)
    qc_tests = {}
    for col in ["pct_present", "median_raw", "pct_nonpositive", "mean_corr"]:
        p = stats.mannwhitneyu(qc.loc[qc["set"] == "train", col], qc.loc[qc["set"] == "test", col]).pvalue
        qc_tests[col] = {
            "train_median": float(qc.loc[qc["set"] == "train", col].median()),
            "test_median": float(qc.loc[qc["set"] == "test", col].median()),
            "p_mannwhitney": float(p),
        }

    # Egitimde uydurulan PCA, testi yalnizca izdusum olarak alir.
    train_cols = meta.index[~is_test]
    train_raw = golub.expr[train_cols]
    kept = dudoit_filter(train_raw)
    train_log = log2_floor(train_raw).loc[kept]
    mu, sd = train_log.mean(axis=1), train_log.std(axis=1, ddof=1)
    x_train = train_log.sub(mu, axis=0).div(sd, axis=0).values.T
    pca = PCA(n_components=10, random_state=SEED).fit(x_train)
    test_log = log2_floor(golub.expr[meta.index[is_test]]).loc[kept]
    x_test = test_log.sub(mu, axis=0).div(sd, axis=0).values.T
    s_train, s_test = pca.transform(x_train), pca.transform(x_test)

    def residual_share(x, model=pca):
        recon = model.inverse_transform(model.transform(x))
        return float(((x - recon) ** 2).sum(axis=1).mean() / (x ** 2).sum(axis=1).mean())

    # Adil referans: gorulmemis ornekler egitimde de yuksek artik verir. Egitim icinde
    # birini-disarida-birak (LOO) artigi, test artigiyla karsilastirilacak taban cizgisidir.
    loo_shares = []
    for i in range(train_log.shape[1]):
        rest = train_log.drop(columns=train_log.columns[i])
        m, s = rest.mean(axis=1), rest.std(axis=1, ddof=1)
        fit = PCA(n_components=10, random_state=SEED).fit(rest.sub(m, axis=0).div(s, axis=0).values.T)
        held = train_log.iloc[:, [i]].sub(m, axis=0).div(s, axis=0).values.T
        loo_shares.append(residual_share(held, fit))

    # Dizi basina ortanca merkezlemesi kuresel kaymayi ne kadar giderir?
    centred = log - log.median(axis=0)
    adjusted_centred = set_effect(centred, is_test, is_aml)

    src = meta.loc[meta.index[is_test], "source"]
    test_scores = pd.DataFrame(s_test[:, :2], index=src.index, columns=["PC1", "PC2"])
    summary = {
        "genes_q05_set_unadjusted": int((unadjusted["q"] < 0.05).sum()),
        "genes_q05_set_adjusted_for_class": int((adjusted["q"] < 0.05).sum()),
        "genes_q05_set_adjusted_for_class_bm_only": int((adjusted_bm["q"] < 0.05).sum()),
        "genes_q05_set_within_bcell_all_bm": int((within_all["q"] < 0.05).sum()),
        "genes_q05_set_within_aml_bm": int((within_aml["q"] < 0.05).sum()),
        "n_genes_tested": int(log.shape[0]),
        "pct_genes_q05_adjusted": float(100 * (adjusted["q"] < 0.05).mean()),
        "n_bcell_all_bm_by_set": pd.Series(is_test[all_b_bm]).map({False: "train", True: "test"}).value_counts().to_dict(),
        "n_aml_bm_by_set": pd.Series(is_test[aml_only]).map({False: "train", True: "test"}).value_counts().to_dict(),
        "array_qc_train_vs_test": qc_tests,
        "median_t_set_effect_adjusted": float(np.median(adjusted["t"])),
        "pct_genes_lower_in_test": float(100 * (adjusted["t"] < 0).mean()),
        "mean_beta_log2_set_effect": float(adjusted["beta"].mean()),
        "genes_q05_set_adjusted_after_median_centring": int((adjusted_centred["q"] < 0.05).sum()),
        "median_t_after_median_centring": float(np.median(adjusted_centred["t"])),
        "residual_share_train_in_sample": residual_share(x_train),
        "residual_share_train_loo_mean": float(np.mean(loo_shares)),
        "residual_share_test_projection": residual_share(x_test),
        "test_pc1_by_source": test_scores.groupby(src)["PC1"].median().round(2).to_dict(),
        "train_pc1_by_source": pd.Series(s_train[:, 0], index=train_cols).groupby(meta.loc[train_cols, "source"]).median().round(2).to_dict(),
        "silhouette_set_top10pc": float(silhouette_score(pca.transform(np.vstack([x_train, x_test])), np.r_[np.zeros(len(x_train)), np.ones(len(x_test))])),
    }
    plots = {
        "s_train": s_train, "s_test": s_test, "evr": pca.explained_variance_ratio_,
        "labels_train": meta.loc[train_cols, "label"].values,
        "labels_test": meta.loc[meta.index[is_test], "label"].values,
        "adjusted_t": adjusted["t"].values,
        "dof": len(is_test) - 3,
    }
    return summary, plots


def mapping_counts(golub, gse, annot_g, annot_r) -> dict:
    out = {}
    for rule, split in [("A_cok_sembol_bolunur", True), ("B_yalniz_tek_sembol", False)]:
        g = probe_to_symbol(golub.expr.index, annot_g, split_multi=split)
        r = probe_to_symbol(gse.expr.index, annot_r, split_multi=split)
        out[rule] = {
            "golub_probes_mapped": int(g.index.nunique()),
            "gse14062_probes_mapped": int(r.index.nunique()),
            "golub_unique_symbols": int(pd.unique(g.values).size),
            "gse14062_unique_symbols": int(pd.unique(r.values).size),
            "shared_symbols": int(len(set(g.values) & set(r.values))),
        }
    out["shared_probe_ids"] = int(len(golub.expr.index.intersection(gse.expr.index)))
    return out


def cross_dataset(golub, gse, annot_g, annot_r) -> tuple[dict, dict]:
    sym_g = probe_to_symbol(golub.expr.index, annot_g)
    sym_r = probe_to_symbol(gse.expr.index, annot_r)
    g_genes = collapse_to_genes(log2_floor(golub.expr), sym_g)
    r_genes = collapse_to_genes(gse.expr, sym_r)
    shared = g_genes.index.intersection(r_genes.index)
    g_s, r_s = g_genes.loc[shared], r_genes.loc[shared]

    train = golub.meta["set"] == "train"
    g_all, g_aml = train & (golub.meta["label"] == "ALL"), train & (golub.meta["label"] == "AML")
    labels = gse.meta["label"]
    g_fc = g_s.loc[:, g_all].mean(axis=1) - g_s.loc[:, g_aml].mean(axis=1)
    r_fc = r_s.loc[:, labels == "ALL"].mean(axis=1) - r_s.loc[:, labels == "AML"].mean(axis=1)

    t_g = stats.ttest_ind(g_s.loc[:, g_all], g_s.loc[:, g_aml], axis=1, equal_var=False).statistic
    t_r = stats.ttest_ind(r_s.loc[:, labels == "ALL"], r_s.loc[:, labels == "AML"], axis=1, equal_var=False).statistic
    t_g, t_r = pd.Series(t_g, index=shared).fillna(0), pd.Series(t_r, index=shared).fillna(0)
    top = {}
    for k in (50, 100, 200):
        g_top, r_top = set(t_g.abs().nlargest(k).index), set(t_r.abs().nlargest(k).index)
        overlap = len(g_top & r_top)
        top[str(k)] = {
            "overlap": overlap,
            "expected_by_chance": float(k * k / len(shared)),
            "hypergeom_p": float(stats.hypergeom.sf(overlap - 1, len(shared), k, k)),
            "sign_concordance_in_overlap": float(np.mean(np.sign(t_g[list(g_top & r_top)]) == np.sign(t_r[list(g_top & r_top)]))) if overlap else None,
            "sign_concordance_golub_top": float(np.mean(np.sign(t_g[list(g_top)]) == np.sign(t_r[list(g_top)]))),
            "shared_genes": sorted(g_top & r_top),
        }

    # Birlesik PCA: duzeltmesiz ve veri seti ici gen bazinda z-skorundan sonra.
    g_m, r_m = g_s.values, r_s.values
    dataset = np.r_[np.zeros(g_m.shape[1]), np.ones(r_m.shape[1])]
    classes = np.r_[golub.meta["label"].values, labels.values]

    def embed(mat):
        x = (mat - mat.mean(axis=1, keepdims=True)).T
        pca = PCA(n_components=10, random_state=SEED).fit(x)
        return pca.transform(x), pca.explained_variance_ratio_

    raw_scores, raw_evr = embed(np.hstack([g_m, r_m]))

    def z(m):
        sd = m.std(axis=1, ddof=1, keepdims=True)
        sd[sd == 0] = 1
        return (m - m.mean(axis=1, keepdims=True)) / sd

    z_scores, z_evr = embed(np.hstack([z(g_m), z(r_m)]))

    def assoc(scores):
        return {
            "eta2_pc1_dataset": eta_squared(scores[:, 0], pd.Series(dataset)),
            "eta2_pc1_class": eta_squared(scores[:, 0], pd.Series(classes)),
            "eta2_pc2_dataset": eta_squared(scores[:, 1], pd.Series(dataset)),
            "eta2_pc2_class": eta_squared(scores[:, 1], pd.Series(classes)),
            "silhouette_dataset_top10": float(silhouette_score(scores, dataset)),
            "silhouette_class_top10": float(silhouette_score(scores, classes)),
        }

    ks_all = stats.ks_2samp(g_m.ravel()[::7], r_m.ravel()[::7])
    summary = {
        "n_shared_genes_rule_b": int(len(shared)),
        "golub_median_log2_shared": float(np.median(g_m)),
        "gse14062_median_log2_shared": float(np.median(r_m)),
        "golub_iqr_log2_shared": float(np.subtract(*np.percentile(g_m, [75, 25]))),
        "gse14062_iqr_log2_shared": float(np.subtract(*np.percentile(r_m, [75, 25]))),
        "pct_at_floor_golub": float(100 * np.mean(g_m <= np.log2(100) + 1e-9)),
        "ks_statistic_log2": float(ks_all.statistic),
        "spearman_gene_means": float(stats.spearmanr(g_m.mean(axis=1), r_m.mean(axis=1)).statistic),
        "pearson_gene_means": float(np.corrcoef(g_m.mean(axis=1), r_m.mean(axis=1))[0, 1]),
        "mean_gene_shift_gse14062_minus_golub": float((r_m.mean(axis=1) - g_m.mean(axis=1)).mean()),
        "spearman_log2fc_all_vs_aml": float(stats.spearmanr(g_fc, r_fc).statistic),
        "spearman_t_all_vs_aml": float(stats.spearmanr(t_g, t_r).statistic),
        "top_gene_concordance": top,
        "joint_pca_no_correction": {**assoc(raw_scores), "evr_pc1": float(raw_evr[0]), "evr_pc2": float(raw_evr[1])},
        "joint_pca_per_dataset_z": {**assoc(z_scores), "evr_pc1": float(z_evr[0]), "evr_pc2": float(z_evr[1])},
    }
    plots = {
        "raw_scores": raw_scores, "raw_evr": raw_evr, "z_scores": z_scores, "z_evr": z_evr,
        "dataset": dataset, "classes": classes, "g_fc": g_fc.values, "r_fc": r_fc.values,
    }
    return summary, plots


def scatter_class(ax, xy, classes, filled_mask=None, size=16):
    for cls in ["ALL", "AML"]:
        m = classes == cls
        if filled_mask is None:
            ax.scatter(xy[m, 0], xy[m, 1], s=size, marker=CLASS_MARKER[cls], facecolors=CLASS_COLOR[cls],
                       edgecolors="white", linewidths=0.4, zorder=3)
        else:
            f = m & filled_mask
            h = m & ~filled_mask
            ax.scatter(xy[h, 0], xy[h, 1], s=size, marker=CLASS_MARKER[cls], facecolors="none",
                       edgecolors=CLASS_COLOR[cls], linewidths=0.8, zorder=3)
            ax.scatter(xy[f, 0], xy[f, 1], s=size, marker=CLASS_MARKER[cls], facecolors=CLASS_COLOR[cls],
                       edgecolors="white", linewidths=0.4, zorder=4)


def plot_batch(p):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6), constrained_layout=True)
    xy = np.vstack([p["s_train"][:, :2], p["s_test"][:, :2]])
    classes = np.r_[p["labels_train"], p["labels_test"]]
    filled = np.r_[np.ones(len(p["s_train"]), bool), np.zeros(len(p["s_test"]), bool)]
    scatter_class(axes[0], xy, classes, filled)
    axes[0].set_xlabel(f"Eğitim PC1 (%{100 * p['evr'][0]:.1f})")
    axes[0].set_ylabel(f"Eğitim PC2 (%{100 * p['evr'][1]:.1f})")
    axes[0].set_title("(a) Testin izdüşümü")
    t = p["adjusted_t"]
    bins = np.linspace(-8, 8, 49)
    axes[1].hist(t[np.abs(t) <= 8], bins=bins, density=True, color=NEUTRAL)
    grid = np.linspace(-8, 8, 200)
    axes[1].plot(grid, stats.t.pdf(grid, p["dof"]), color="#0b0b0b", lw=1.2, ls="--")
    axes[1].set_xlabel("Küme etkisi (t)")
    axes[1].set_ylabel("Yoğunluk")
    axes[1].set_title("(b) Gen başına kayma")
    axes[1].text(0.03, 0.97, "kesik: etki\nyoksa beklenen", transform=axes[1].transAxes, ha="left", va="top", fontsize=6, color=MUTED)
    axes[1].grid(axis="x", visible=False)
    axes[1].set_ylim(0, 0.58)
    handles = [
        Line2D([], [], marker="o", ls="", color=CLASS_COLOR["ALL"], label="ALL"),
        Line2D([], [], marker="s", ls="", color=CLASS_COLOR["AML"], label="AML"),
        Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, label="eğitim"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=MUTED, label="test"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=4, handletextpad=0.1, columnspacing=1.0)
    save_fig(fig, "fig5_train_test_shift")


def plot_cross(p, summary):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.95), constrained_layout=True)
    is_golub = p["dataset"] == 0
    scatter_class(axes[0], p["raw_scores"], p["classes"], is_golub)
    axes[0].set_title("(a) Düzeltmesiz birleşik PCA")
    axes[0].set_xlabel(f"PC1 (%{100 * p['raw_evr'][0]:.1f})")
    axes[0].set_ylabel(f"PC2 (%{100 * p['raw_evr'][1]:.1f})")
    scatter_class(axes[1], p["z_scores"], p["classes"], is_golub)
    axes[1].set_title("(b) Veri seti içi z-skoru sonrası")
    axes[1].set_xlabel(f"PC1 (%{100 * p['z_evr'][0]:.1f})")
    axes[1].set_ylabel(f"PC2 (%{100 * p['z_evr'][1]:.1f})")
    ax = axes[2]
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.scatter(p["g_fc"], p["r_fc"], s=2.5, color=NEUTRAL, alpha=0.35, linewidths=0)
    ax.set_xlabel("Golub eğitim: log2FC")
    ax.set_ylabel("GSE14062: log2FC")
    ax.set_title("(c) ALL–AML farkının uyumu")
    ax.text(0.03, 0.95, f"Spearman ρ = {summary['spearman_log2fc_all_vs_aml']:.2f}".replace(".", ","),
            transform=ax.transAxes, va="top", fontsize=7, color=MUTED)
    handles = [
        Line2D([], [], marker="o", ls="", color=CLASS_COLOR["ALL"], label="ALL"),
        Line2D([], [], marker="s", ls="", color=CLASS_COLOR["AML"], label="AML"),
        Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, label="Golub (dolu)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=MUTED, label="GSE14062 (boş)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=4, handletextpad=0.2, columnspacing=1.4)
    save_fig(fig, "fig6_cross_dataset")


def main() -> None:
    set_style()
    golub, gse = load_golub(), load_gse14062()
    annot_g, annot_r = load_annotation("GPL80"), load_annotation("GPL570")
    batch, batch_plots = batch_summary(golub)
    plot_batch(batch_plots)
    cross, cross_plots = cross_dataset(golub, gse, annot_g, annot_r)
    plot_cross(cross_plots, cross)
    update_summary("shift", {
        "golub_train_vs_test": batch,
        "cross_dataset": cross,
        "symbol_mapping": mapping_counts(golub, gse, annot_g, annot_r),
    })
    print("Kayma adimi tamam.")


if __name__ == "__main__":
    main()
