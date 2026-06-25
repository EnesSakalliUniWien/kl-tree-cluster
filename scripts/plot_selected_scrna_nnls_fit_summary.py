"""One-page NNLS branch-length fit summary for the selected scRNA TBS run."""
# ruff: noqa: I001

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from tree_break_selection.tree.optimized_branch_lengths import (
    _sample_leaf_pairs,
    _squared_standardized_targets,
)

from scripts.plot_selected_scrna_nnls_report import (
    BASE,
    DATASETS,
    METHOD_TITLE,
    SELECTED,
    build_full_tree_info,
    make_cluster_palette,
    plot_full_radial_tree,
    plot_umap,
)

OUT_FIG = BASE / "scrna_selected_adaptive_diffusion_nnls_fit_summary.png"
OUT_PDF = BASE / "scrna_selected_adaptive_diffusion_nnls_fit_summary_one_plot_per_page.pdf"
OUT_CSV = BASE / "scrna_selected_adaptive_diffusion_nnls_fit_summary.csv"


def _root_depths(edges: pd.DataFrame) -> tuple[dict[str, str], dict[str, float], str]:
    parent_of = dict(zip(edges["child"], edges["parent"]))
    branch_of = {
        (row.parent, row.child): 0.0 if pd.isna(row.branch_length) else float(row.branch_length)
        for row in edges.itertuples(index=False)
    }
    children: dict[str, list[str]] = {}
    for row in edges.itertuples(index=False):
        children.setdefault(row.parent, []).append(row.child)
    roots = sorted(set(edges["parent"]) - set(edges["child"]))
    if len(roots) != 1:
        raise ValueError(f"Expected one root, found {roots!r}")
    root = roots[0]
    depth = {root: 0.0}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            depth[child] = depth[node] + max(branch_of.get((node, child), 0.0), 0.0)
            stack.append(child)
    return parent_of, depth, root


def _ancestors(node: str, parent_of: dict[str, str]) -> list[str]:
    path = [node]
    while node in parent_of:
        node = parent_of[node]
        path.append(node)
    return path


def _tree_pair_distances(
    *,
    edges: pd.DataFrame,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    parent_of, depth, _root = _root_depths(edges)
    ancestor_paths = {}
    ancestor_sets = {}
    max_leaf = int(max(left.max(), right.max()))
    for index in range(max_leaf + 1):
        leaf = f"L{index}"
        path = _ancestors(leaf, parent_of)
        ancestor_paths[index] = path
        ancestor_sets[index] = set(path)

    distances = np.empty(len(left), dtype=float)
    for row, (left_index, right_index) in enumerate(zip(left, right, strict=True)):
        left_set = ancestor_sets[int(left_index)]
        lca = None
        for candidate in ancestor_paths[int(right_index)]:
            if candidate in left_set:
                lca = candidate
                break
        if lca is None:
            raise ValueError("Could not find a common ancestor.")
        distances[row] = (
            depth[f"L{int(left_index)}"] + depth[f"L{int(right_index)}"] - 2.0 * depth[lca]
        )
    return distances


def _fit_diagnostic(cfg: dict[str, object], pair_sample_size: int = 50_000) -> dict[str, object]:
    output_dir = Path(cfg["dir"])
    pca = pd.read_csv(output_dir / "benchmark_subset_pca.csv", index_col=0)
    assignments = pd.read_csv(output_dir / "method_assignments.csv")
    if list(pca.index.astype(str)) != list(assignments["cell_id"].astype(str)):
        raise ValueError(f"PCA rows do not align with assignments for {cfg['key']}")

    left, right, total_pairs = _sample_leaf_pairs(
        len(pca),
        pair_sample_size=pair_sample_size,
        random_state=0,
    )
    target = _squared_standardized_targets(pca, left, right)
    edges = pd.read_csv(output_dir / f"{SELECTED}_tree_edges.csv")
    fitted = _tree_pair_distances(edges=edges, left=left, right=right)
    residual = fitted - target
    sst = float(np.sum((target - target.mean()) ** 2))
    sse = float(np.sum(residual**2))
    correlation = float(np.corrcoef(target, fitted)[0, 1])
    lengths = pd.to_numeric(edges["branch_length"], errors="coerce").dropna()
    return {
        "dataset": str(cfg["key"]),
        "n_cells": int(len(pca)),
        "n_clusters": int(assignments[SELECTED].nunique()),
        "n_pairs_total": int(total_pairs),
        "n_pairs_used": int(len(left)),
        "target": target,
        "fitted": fitted,
        "residual": residual,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "correlation": correlation,
        "target_median": float(np.median(target)),
        "fitted_median": float(np.median(fitted)),
        "branch_length_median": float(lengths.median()),
        "branch_length_max": float(lengths.max()),
        "zero_branch_fraction": float((lengths <= 1e-12).mean()),
    }


def _plot_fit(ax, diagnostic: dict[str, object]) -> None:
    target = np.asarray(diagnostic["target"], dtype=float)
    fitted = np.asarray(diagnostic["fitted"], dtype=float)
    max_value = float(np.nanmax(np.concatenate([target, fitted]))) * 1.03
    max_value = max(max_value, 1e-9)
    hb = ax.hexbin(
        target,
        fitted,
        gridsize=54,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0,
    )
    ax.plot([0.0, max_value], [0.0, max_value], color="#111827", lw=1.0, ls="--")
    ax.set_xlim(0.0, max_value)
    ax.set_ylim(0.0, max_value)
    ax.set_xlabel("target pair distance", fontsize=9)
    ax.set_ylabel("NNLS tree path distance", fontsize=9)
    ax.set_title(
        f"{diagnostic['dataset']}: branch-length fit ({diagnostic['n_pairs_used']:,} sampled pairs)",
        fontsize=11,
        weight="bold",
    )
    ax.tick_params(labelsize=8)
    ax.grid(color="#e5e7eb", lw=0.45)
    cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("pair count, log scale", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _plot_metrics(ax, diagnostic: dict[str, object]) -> None:
    ax.axis("off")
    text = (
        f"{diagnostic['dataset']}\n"
        f"cells / clusters: {diagnostic['n_cells']:,} / {diagnostic['n_clusters']}\n"
        f"sampled pairs: {diagnostic['n_pairs_used']:,} of {diagnostic['n_pairs_total']:,}\n"
        f"R2: {diagnostic['r2']:.3f}\n"
        f"correlation: {diagnostic['correlation']:.3f}\n"
        f"RMSE: {diagnostic['rmse']:.3f}\n"
        f"MAE: {diagnostic['mae']:.3f}\n"
        f"median target / fitted: {diagnostic['target_median']:.3f} / {diagnostic['fitted_median']:.3f}\n"
        f"zero-length branches: {diagnostic['zero_branch_fraction']:.1%}\n"
        f"max branch length: {diagnostic['branch_length_max']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f9fafb", "edgecolor": "#d1d5db"},
    )


def _add_generated_at(fig: plt.Figure, generated_at: str) -> None:
    fig.text(
        0.99,
        0.01,
        f"Generated at: {generated_at}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#6b7280",
    )


def _save_single_plot_pages(diagnostics: list[dict[str, object]], generated_at: str) -> None:
    with PdfPages(
        OUT_PDF,
        metadata={
            "Title": "Selected scRNA NNLS-TBS fitting results",
            "Subject": f"Generated at: {generated_at}",
        },
    ) as pdf:
        for cfg, diagnostic in zip(DATASETS, diagnostics, strict=True):
            assignments = pd.read_csv(Path(cfg["dir"]) / "method_assignments.csv")
            palette = make_cluster_palette(assignments[SELECTED].unique())

            fig, ax = plt.subplots(figsize=(9, 8.2), facecolor="white")
            plot_umap(
                ax,
                assignments,
                SELECTED,
                palette,
                f"{cfg['title']}\nselected NNLS-TBS clusters",
                annotate_clusters=True,
                point_size=6,
            )
            _add_generated_at(fig, generated_at)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 8.2), facecolor="white")
            full_info = build_full_tree_info(cfg, assignments)
            plot_full_radial_tree(ax, full_info, palette, f"{cfg['title']}\nreal full radial tree")
            for text in list(ax.texts):
                text.remove()
            fig.text(
                0.5,
                0.035,
                "All leaves are plotted; leaf colors match the selected-cluster UMAP.",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#374151",
            )
            _add_generated_at(fig, generated_at)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.8, 8.2), facecolor="white")
            _plot_fit(ax, diagnostic)
            text = (
                f"cells / clusters: {diagnostic['n_cells']:,} / {diagnostic['n_clusters']}\n"
                f"sampled pairs: {diagnostic['n_pairs_used']:,}\n"
                f"R2: {diagnostic['r2']:.3f}\n"
                f"correlation: {diagnostic['correlation']:.3f}\n"
                f"RMSE: {diagnostic['rmse']:.3f}; MAE: {diagnostic['mae']:.3f}"
            )
            ax.text(
                0.03,
                0.97,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "#d1d5db",
                    "alpha": 0.92,
                },
            )
            _add_generated_at(fig, generated_at)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    diagnostics = [_fit_diagnostic(cfg) for cfg in DATASETS]
    summary = pd.DataFrame(
        [
            {key: value for key, value in diagnostic.items() if key not in {"target", "fitted", "residual"}}
            for diagnostic in diagnostics
        ]
    )
    summary.insert(0, "generated_at", generated_at)
    summary.to_csv(OUT_CSV, index=False)

    fig = plt.figure(figsize=(18, 13), facecolor="white")
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.0, 1.22, 1.0, 0.72],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.22,
    )
    for row, (cfg, diagnostic) in enumerate(zip(DATASETS, diagnostics, strict=True)):
        assignments = pd.read_csv(Path(cfg["dir"]) / "method_assignments.csv")
        palette = make_cluster_palette(assignments[SELECTED].unique())
        full_info = build_full_tree_info(cfg, assignments)
        ax_umap = fig.add_subplot(gs[row, 0])
        ax_tree = fig.add_subplot(gs[row, 1])
        ax_fit = fig.add_subplot(gs[row, 2])
        ax_metrics = fig.add_subplot(gs[row, 3])
        plot_umap(
            ax_umap,
            assignments,
            SELECTED,
            palette,
            f"{cfg['title']}\nselected clusters",
            annotate_clusters=True,
            point_size=5,
        )
        plot_full_radial_tree(ax_tree, full_info, palette, "real full tree")
        for text in list(ax_tree.texts):
            text.remove()
        _plot_fit(ax_fit, diagnostic)
        _plot_metrics(ax_metrics, diagnostic)

    fig.suptitle(
        f"Selected scRNA NNLS-TBS fitting results\n{METHOD_TITLE}",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.012,
        0.012,
        "Fit check uses the saved selected NNLS tree edges and the saved benchmark PCA matrix. "
        "Target is squared standardized Euclidean pair distance; fitted value is summed branch time along the real tree path. "
        f"Generated at: {generated_at}",
        fontsize=9,
        color="#374151",
    )
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    _save_single_plot_pages(diagnostics, generated_at)
    print(OUT_FIG)
    print(OUT_PDF)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
