#!/usr/bin/env python3
"""Run current TBS on adaptive-diffusion cosine subspace trees.

The output layout is intentionally subspace-first:

    <output-dir>/
      rankings/
      plots/
      subspaces/<weighting>/<block_name>/

Each subspace folder contains the cluster assignments, linkage tree, GO-IC
scores, cluster coherence, TF-IDF quality, subspace coordinates, diffusion
metadata, and per-axis GO-term loading tables/plots.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import textwrap
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmarks.shared.runners.tbs_runner import run_tbs_on_distance
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.plot.image_panel import draw_image_panel
from tree_break_selection.space_separation import (
    adaptive_spectral_blocks,
    block_adaptive_diffusion_geometry,
    coordinates_for_block,
    cosine_eigendecomposition,
    weight_feature_matrix,
)

from applications.endotypes._shared import load_binary_feature_matrix, matrix_slug, safe_name
from applications.endotypes.pipelines.spectral_records import spectral_block_record
from applications.endotypes.pipelines.tree_analysis_args import add_tree_analysis_arguments
from applications.endotypes.plots.go_ic_tree_summary_plots import (
    cluster_coherence,
    cluster_size_metrics,
    go_information_criterion,
    quality_tier,
    tfidf_within_cosine_quality,
)

RESULT_PREFIX = "current_adaptive_diffusion_subspace_tree"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_allGO_new.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    add_tree_analysis_arguments(
        parser,
        edge_alpha_default=DEFAULT_EDGE_ALPHA,
        sibling_alpha_default=DEFAULT_SIBLING_ALPHA,
    )
    parser.add_argument("--top-terms-per-axis", type=int, default=20)
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Optional label used in experiment directory and reader-facing artifact names.",
    )
    return parser.parse_args()


def default_output_dir(input_path: Path, dataset_label: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("results/analyses")
        / f"{matrix_slug(input_path, dataset_label)}_{RESULT_PREFIX}_{stamp}"
    )


def parse_go_term(column: str) -> tuple[str, str]:
    import re

    match = re.search(r"(GO:\d{7})", column)
    go_id = match.group(1) if match else ""
    term = column.replace(f"({go_id})", "").strip() if go_id else column
    return term, go_id


def binary_entropy_bits(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    out = np.zeros_like(p)
    valid = (p > 0.0) & (p < 1.0)
    q = p[valid]
    out[valid] = -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))
    return out


def load_binary_matrix(path: Path) -> pd.DataFrame:
    return load_binary_feature_matrix(
        path,
        drop_zero_columns=True,
        require_nonzero_rows=True,
        non_binary_message="contains values outside {0,1}.",
    )


def component_feature_loadings(
    values: np.ndarray, eigvals: np.ndarray, eigvecs: np.ndarray
) -> np.ndarray:
    """Return feature loadings for the sample-cosine eigenvectors.

    The cosine operator is X_norm X_norm^T. For positive eigenvalue lambda and
    sample eigenvector u, the corresponding feature loading is
    X_norm^T u / sqrt(lambda).
    """

    row_normed = normalize(values, norm="l2", axis=1)
    loadings = np.zeros((values.shape[1], len(eigvals)), dtype=float)
    for idx, eigval in enumerate(eigvals):
        if eigval <= 1e-12:
            continue
        loadings[:, idx] = row_normed.T @ eigvecs[:, idx] / math.sqrt(float(eigval))
    return np.nan_to_num(loadings)


def build_term_metadata(data: pd.DataFrame) -> pd.DataFrame:
    supports = data.sum(axis=0).to_numpy(dtype=int)
    prevalence = supports / max(len(data), 1)
    parsed = [parse_go_term(col) for col in data.columns]
    return pd.DataFrame(
        {
            "go_term": [term for term, _go_id in parsed],
            "go_id": [go_id for _term, go_id in parsed],
            "column": data.columns,
            "support": supports,
            "prevalence": prevalence,
            "entropy_bits": binary_entropy_bits(prevalence),
        }
    )


def write_axis_loadings(
    *,
    output_dir: Path,
    subspace_id: str,
    term_metadata: pd.DataFrame,
    loadings: np.ndarray,
    block_start: int,
    block_end: int,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    top_rows = []
    for component in range(block_start, block_end + 1):
        component_values = loadings[:, component - 1]
        frame = term_metadata.copy()
        frame.insert(0, "axis", int(component))
        frame.insert(1, "axis_name", f"mode_{component:02d}")
        frame["loading"] = component_values
        frame["abs_loading"] = np.abs(component_values)
        frame["loading_sign"] = np.where(frame["loading"] >= 0.0, "positive", "negative")
        frame["abs_rank"] = frame["abs_loading"].rank(method="first", ascending=False).astype(int)
        rows.append(frame)

        pos = frame.sort_values("loading", ascending=False).head(top_n).copy()
        pos.insert(2, "selection", "top_positive")
        neg = frame.sort_values("loading", ascending=True).head(top_n).copy()
        neg.insert(2, "selection", "top_negative")
        abs_top = frame.sort_values("abs_loading", ascending=False).head(top_n).copy()
        abs_top.insert(2, "selection", "top_absolute")
        top_rows.extend([pos, neg, abs_top])

    all_loadings = pd.concat(rows, ignore_index=True)
    top_loadings = pd.concat(top_rows, ignore_index=True)
    all_loadings.to_csv(output_dir / f"{subspace_id}__axis_term_loadings_all.csv", index=False)
    top_loadings.to_csv(output_dir / f"{subspace_id}__axis_top_terms.csv", index=False)
    return all_loadings, top_loadings


def wrap_labels(values: pd.Series, width: int = 38) -> list[str]:
    return ["\n".join(textwrap.wrap(str(value), width=width)) for value in values]


def plot_axis_terms(axis_frame: pd.DataFrame, output_path: Path, *, top_n: int) -> None:
    for axis, frame in axis_frame.groupby("axis", sort=True):
        top = frame.sort_values("abs_loading", ascending=False).head(top_n).copy()
        top = top.sort_values("loading")
        height = max(5.5, min(10.5, 0.34 * len(top) + 2.0))
        fig, ax = plt.subplots(figsize=(11.5, height))
        colors = np.where(top["loading"] >= 0, "#4c78a8", "#e45756")
        ax.barh(wrap_labels(top["go_term"], width=42), top["loading"], color=colors)
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(f"Mode {int(axis):02d}: strongest GO-term loadings")
        ax.set_xlabel("feature loading")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.18)
        fig.tight_layout()
        fig.savefig(output_path.parent / f"{output_path.stem}__mode_{int(axis):02d}.png", dpi=180)
        plt.close(fig)


def plot_combined_axis_terms(
    axis_frame: pd.DataFrame,
    output_path: Path,
    *,
    terms_per_axis: int = 8,
    max_terms: int = 60,
) -> None:
    """Plot one signed-loading heatmap across all axes in a subspace."""

    pivot = _selected_axis_term_heatmap(
        axis_frame,
        terms_per_axis=terms_per_axis,
        max_terms=max_terms,
    )
    if pivot.empty:
        return
    height = max(9.0, min(28.0, 0.32 * len(pivot) + 3.0))
    width = max(12.0, min(28.0, 0.42 * len(pivot.columns) + 8.5))
    fig, ax = plt.subplots(figsize=(width, height))
    limit = float(np.nanmax(np.abs(pivot.to_numpy())))
    if not math.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{int(axis):02d}" for axis in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), wrap_labels(pd.Series(pivot.index), width=58))
    ax.set_xlabel("cosine mode")
    ax.set_title("Combined GO-term loadings across subspace axes")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.colorbar(image, ax=ax, label="feature loading", fraction=0.028, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _selected_axis_term_heatmap(
    axis_frame: pd.DataFrame, *, terms_per_axis: int, max_terms: int
) -> pd.DataFrame:
    selected_frames = []
    for _axis, frame in axis_frame.groupby("axis", sort=True):
        selected_frames.append(
            frame.sort_values("abs_loading", ascending=False).head(terms_per_axis)
        )
    if not selected_frames:
        return pd.DataFrame()
    selected = pd.concat(selected_frames, ignore_index=True)
    term_order = (
        selected.groupby(["go_term", "go_id"], as_index=False)
        .agg(max_abs_loading=("abs_loading", "max"))
        .sort_values("max_abs_loading", ascending=False)
        .head(max_terms)
    )
    selected_terms = set(zip(term_order["go_term"], term_order["go_id"], strict=False))
    frame = axis_frame[
        axis_frame.apply(lambda row: (row["go_term"], row["go_id"]) in selected_terms, axis=1)
    ].copy()
    frame["term_label"] = frame.apply(
        lambda row: f"{row['go_term']} ({row['go_id']})" if row["go_id"] else str(row["go_term"]),
        axis=1,
    )
    order = [
        f"{row.go_term} ({row.go_id})" if row.go_id else str(row.go_term)
        for row in term_order.itertuples(index=False)
    ]
    return (
        frame.pivot_table(index="term_label", columns="axis", values="loading", aggfunc="first")
        .reindex(order)
        .fillna(0.0)
    )


def draw_axis_terms_heatmap(
    fig: plt.Figure,
    axis_frame: pd.DataFrame,
    title: str,
    *,
    terms_per_axis: int = 6,
    max_terms: int = 36,
) -> None:
    pivot = _selected_axis_term_heatmap(
        axis_frame,
        terms_per_axis=terms_per_axis,
        max_terms=max_terms,
    )
    ax = fig.add_subplot(111)
    if pivot.empty:
        ax.axis("off")
        ax.text(
            0.5, 0.5, "No GO-term loading data available", ha="center", va="center", fontsize=13
        )
        return
    limit = float(np.nanmax(np.abs(pivot.to_numpy())))
    if not math.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{int(axis):02d}" for axis in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), wrap_labels(pd.Series(pivot.index), width=62))
    ax.set_xlabel("cosine mode", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=8.8)
    cbar = fig.colorbar(image, ax=ax, label="feature loading", fraction=0.028, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.37, right=0.91, top=0.9, bottom=0.08)


def axis_term_summary_lines(top_terms: pd.DataFrame) -> list[str]:
    """Summarize the strongest term loadings for compact embedding annotations."""

    absolute = top_terms[top_terms["selection"].eq("top_absolute")].copy()
    if absolute.empty:
        absolute = top_terms.copy()
    n_axes = int(absolute["axis"].nunique())
    terms_per_axis = 2 if n_axes <= 6 else 1
    lines: list[str] = []
    for axis, frame in absolute.groupby("axis", sort=True):
        top = frame.sort_values("abs_loading", ascending=False).head(terms_per_axis)
        labels = []
        for row in top.itertuples(index=False):
            sign = "+" if float(row.loading) >= 0 else "-"
            labels.append(f"{row.go_term} ({sign}{abs(float(row.loading)):.3f})")
        lines.append(f"mode {int(axis):02d}: " + "; ".join(labels))
    return lines


def _wrap_annotation_lines(lines: list[str], *, width: int = 58) -> str:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent="  "))
        wrapped.append("")
    return "\n".join(wrapped).rstrip()


def _embedding_figure_with_optional_axis_terms(
    axis_term_lines: list[str] | None,
) -> tuple[plt.Figure, plt.Axes]:
    if not axis_term_lines:
        return plt.subplots(figsize=(8, 6.5))

    fig, (ax, text_ax) = plt.subplots(
        1,
        2,
        figsize=(16, 8.5),
        gridspec_kw={"width_ratios": [1.2, 0.95]},
    )
    text_ax.axis("off")
    text_ax.set_title("Highest GO-term loadings by eigenmode", loc="left", fontsize=11)
    text_ax.text(
        0.0,
        0.98,
        _wrap_annotation_lines(axis_term_lines),
        va="top",
        ha="left",
        fontsize=8.0,
        linespacing=1.15,
        transform=text_ax.transAxes,
    )
    return fig, ax


def plot_subspace_embedding(
    coords: np.ndarray,
    labels: np.ndarray | None,
    output_path: Path,
    title: str,
    *,
    axis_term_lines: list[str] | None = None,
) -> None:
    if coords.shape[1] == 1:
        embedding = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
        method = "axis"
    else:
        try:
            import umap

            embedding = umap.UMAP(
                n_components=2,
                n_neighbors=min(18, max(2, coords.shape[0] - 1)),
                min_dist=0.05,
                metric="euclidean",
                random_state=1729,
            ).fit_transform(coords)
            method = "UMAP"
        except Exception:
            embedding = PCA(n_components=2, random_state=1729).fit_transform(coords)
            method = "PCA"
    fig, ax = _embedding_figure_with_optional_axis_terms(axis_term_lines)
    if labels is None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], color="#4c78a8", s=18, alpha=0.72, linewidths=0
        )
        ax.set_title(f"{title}\nsubspace embedding; no cluster assignments ({method})")
    else:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="turbo", s=18, alpha=0.88, linewidths=0
        )
        ax.set_title(f"{title}\nsubspace embedding by cluster ({method})")
    ax.set_xlabel("axis 1")
    ax.set_ylabel("axis 2")
    if labels is not None:
        fig.colorbar(scatter, ax=ax, label="cluster id", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_diffusion_embedding(
    distances: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    *,
    axis_term_lines: list[str] | None = None,
) -> None:
    matrix = squareform(distances)
    embedding = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=1729,
        normalized_stress="auto",
    ).fit_transform(matrix)
    fig, ax = _embedding_figure_with_optional_axis_terms(axis_term_lines)
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="turbo", s=18, alpha=0.88, linewidths=0
    )
    ax.set_title(f"{title}\nadaptive diffusion distance embedding by cluster")
    ax.set_xlabel("MDS axis 1")
    ax.set_ylabel("MDS axis 2")
    fig.colorbar(scatter, ax=ax, label="cluster id", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dendrogram(linkage_matrix: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        linkage_matrix,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#333333",
        link_color_func=lambda _node_id: "#333333",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("adaptive diffusion distance")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_tree_subtree_clusters(
    fig: plt.Figure,
    linkage_matrix: np.ndarray,
    assignments: pd.DataFrame,
    title: str,
) -> None:
    labels = assignments["cluster_id"].to_numpy(dtype=int)
    cluster_sizes = assignments["cluster_id"].value_counts().sort_values(ascending=False)
    n_leaves = len(labels)
    n_clusters = int(assignments["cluster_id"].nunique())

    grid = fig.add_gridspec(3, 1, height_ratios=[7.4, 0.72, 1.25], hspace=0.09)
    ax_tree = fig.add_subplot(grid[0])
    ax_strip = fig.add_subplot(grid[1], sharex=ax_tree)
    ax_text = fig.add_subplot(grid[2])

    dendro = dendrogram(
        linkage_matrix,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#333333",
        link_color_func=lambda _node_id: "#333333",
        ax=ax_tree,
    )
    leaf_order = np.asarray(dendro["leaves"], dtype=int)
    ordered_labels = labels[leaf_order]
    ax_strip.imshow(
        ordered_labels[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap="turbo",
        extent=(0, 10 * n_leaves, 0, 1),
        vmin=float(np.min(labels)),
        vmax=float(np.max(labels)),
    )
    ax_tree.set_title(title, fontsize=15, pad=10)
    ax_tree.set_ylabel("adaptive diffusion distance", fontsize=12)
    ax_tree.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_tree.tick_params(axis="y", labelsize=10)
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("cluster", rotation=0, ha="right", va="center", labelpad=30, fontsize=12)
    ax_strip.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_strip.spines.values():
        spine.set_linewidth(0.6)

    ax_text.axis("off")
    top_sizes = ", ".join(
        f"C{int(cid)}={int(size)}" for cid, size in cluster_sizes.head(16).items()
    )
    ax_text.text(
        0.0,
        0.98,
        "Tree cluster assignments: the colored strip is the final current TBS cluster id in dendrogram leaf order.\n"
        f"Clusters: {n_clusters}; genes: {n_leaves}; largest cluster sizes: {top_sizes}",
        va="top",
        fontsize=12,
        linespacing=1.25,
        transform=ax_text.transAxes,
    )


def plot_tree_subtree_clusters(
    linkage_matrix: np.ndarray,
    assignments: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(18, 10.5))
    _draw_tree_subtree_clusters(fig, linkage_matrix, assignments, title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def assignments_from_labels(labels: np.ndarray, index: pd.Index) -> pd.DataFrame:
    labels = labels.astype(int)
    sizes = pd.Series(labels).value_counts().sort_index()
    out = pd.DataFrame({"gene": index.astype(str), "cluster_id": labels})
    out["cluster_size"] = out["cluster_id"].map(sizes).astype(int)
    return out


def run_current_kl(
    data: pd.DataFrame,
    distances: np.ndarray,
    *,
    edge_alpha: float,
    sibling_alpha: float,
) -> np.ndarray:
    result = run_tbs_on_distance(
        data,
        distances,
        sibling_alpha,
        tree_linkage_method="average",
        edge_alpha=edge_alpha,
    )
    if result.labels is None:
        raise RuntimeError(f"Current TBS returned no labels: {result.skip_reason or result.status}")
    return np.asarray(result.labels, dtype=int)


def attach_artifact_paths(ranking: pd.DataFrame) -> pd.DataFrame:
    out = ranking.copy()
    rows: list[dict[str, object]] = []
    for row in out.itertuples(index=False):
        subspace_id = f"{row.weighting}__{row.block_name}"
        subspace_safe = safe_name(subspace_id)
        subspace_dir = Path(row.subspace_dir)
        rows.append(
            {
                "cluster_assignments": getattr(row, "assignments_path", ""),
                "failure_status": str(subspace_dir / f"{subspace_safe}__failure_status.csv")
                if getattr(row, "status", "") != "ok"
                else "",
                "linkage_matrix": getattr(row, "linkage_path", ""),
                "tree_dendrogram_png": str(subspace_dir / f"{subspace_safe}__tree_dendrogram.png"),
                "tree_subtree_clusters_png": str(
                    subspace_dir / f"{subspace_safe}__tree_subtree_clusters.png"
                ),
                "subspace_embedding_png": str(
                    subspace_dir / f"{subspace_safe}__subspace_embedding_clusters.png"
                ),
                "adaptive_diffusion_embedding_png": str(
                    subspace_dir / f"{subspace_safe}__adaptive_diffusion_embedding_clusters.png"
                ),
                "axis_terms_combined_png": str(
                    subspace_dir / f"{subspace_safe}__axis_terms_combined.png"
                ),
                "axis_top_terms": str(subspace_dir / f"{subspace_safe}__axis_top_terms.csv"),
                "axis_term_loadings_all": str(
                    subspace_dir / f"{subspace_safe}__axis_term_loadings_all.csv"
                ),
                "go_ic_quality_summary": str(
                    subspace_dir / f"{subspace_safe}__go_ic_quality_summary.csv"
                ),
                "cluster_coherence": str(subspace_dir / f"{subspace_safe}__cluster_coherence.csv"),
                "tfidf_cluster_quality": str(
                    subspace_dir / f"{subspace_safe}__tfidf_cluster_quality.csv"
                ),
                "subspace_embedding_terms_only_png": str(
                    subspace_dir / f"{subspace_safe}__subspace_embedding_terms_only.png"
                ),
                "subspace_embedding_annotated_terms_png": str(
                    subspace_dir
                    / f"{subspace_safe}__subspace_embedding_clusters_annotated_terms.png"
                ),
                "adaptive_diffusion_embedding_annotated_terms_png": str(
                    subspace_dir
                    / f"{subspace_safe}__adaptive_diffusion_embedding_clusters_annotated_terms.png"
                ),
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_specificity_aware_rank(
    ranking: pd.DataFrame, coherence_long: pd.DataFrame | None
) -> pd.DataFrame:
    out = ranking.copy()
    out["old_display_rank"] = out.get("display_rank", pd.Series(index=out.index, dtype=float))
    out["specific_cluster_count"] = 0
    out["specific_cluster_fraction"] = 0.0
    out["weighted_specificity_delta"] = 0.0
    out["median_specificity_delta"] = 0.0
    out["mean_specificity_delta"] = 0.0
    out["specificity_score"] = 0.0
    if coherence_long is not None and not coherence_long.empty:
        metrics: dict[str, dict[str, float]] = {}
        for run_id, frame in coherence_long.groupby("run_id"):
            deltas = pd.to_numeric(frame["top_term_prevalence_delta"], errors="coerce").fillna(0.0)
            sizes = pd.to_numeric(frame["cluster_size"], errors="coerce").fillna(0.0)
            specific = frame["coherent_by_rule"].astype(bool) & deltas.gt(0.0)
            specific_deltas = deltas[specific]
            weight_denom = float(sizes[specific].sum())
            weighted = (
                float((deltas[specific] * sizes[specific]).sum() / weight_denom)
                if weight_denom > 0
                else 0.0
            )
            fraction = float(specific.sum() / max(len(frame), 1))
            metrics[str(run_id)] = {
                "specific_cluster_count": float(specific.sum()),
                "specific_cluster_fraction": fraction,
                "weighted_specificity_delta": weighted,
                "median_specificity_delta": float(specific_deltas.median())
                if len(specific_deltas)
                else 0.0,
                "mean_specificity_delta": float(specific_deltas.mean())
                if len(specific_deltas)
                else 0.0,
                "specificity_score": 0.5 * fraction + 0.5 * weighted,
            }
        for column in [
            "specific_cluster_count",
            "specific_cluster_fraction",
            "weighted_specificity_delta",
            "median_specificity_delta",
            "mean_specificity_delta",
            "specificity_score",
        ]:
            out[column] = out["run_id"].map(
                lambda run_id: metrics.get(str(run_id), {}).get(column, 0.0)
            )
        out["specific_cluster_count"] = out["specific_cluster_count"].astype(int)

    ok_mask = out["status"].eq("ok")
    if ok_mask.any():
        order = out.loc[ok_mask].sort_values(
            [
                "quality_tier",
                "specificity_score",
                "specific_cluster_fraction",
                "weighted_specificity_delta",
                "go_bic_active_per_gene",
            ],
            ascending=[True, False, False, False, True],
        )
        ranks = dict(zip(order["run_id"], np.arange(1, len(order) + 1), strict=False))
        out.loc[ok_mask, "specificity_aware_rank"] = (
            out.loc[ok_mask, "run_id"].map(ranks).astype(int)
        )
        out.loc[ok_mask, "display_rank"] = out.loc[ok_mask, "specificity_aware_rank"].astype(int)
    return out.sort_values(["status", "display_rank"], ascending=[False, True]).reset_index(
        drop=True
    )


def write_combined_axis_terms_pdf(ranking: pd.DataFrame, output_dir: Path) -> Path | None:
    ok = ranking[ranking["status"].eq("ok")].sort_values("display_rank")
    if ok.empty:
        return None
    pdf_path = output_dir / f"{RESULT_PREFIX}_axis_terms_combined_by_subspace.pdf"
    with PdfPages(pdf_path) as pdf:
        for row in ok.itertuples(index=False):
            fig, ax = plt.subplots(figsize=(16, 10))
            draw_image_panel(
                ax,
                getattr(row, "axis_terms_combined_png", ""),
                f"Rank {int(row.display_rank)}: {row.weighting} / {row.block_name}\n"
                "Combined signed GO-term feature loadings across subspace eigenmodes",
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def write_workflow_pdfs(
    output_dir: Path, ranking: pd.DataFrame, artifact_prefix: str
) -> dict[str, str]:
    ok = ranking[ranking["status"].eq("ok")].sort_values("display_rank")
    plots_dir = output_dir / f"{artifact_prefix}_quality_aware_go_ic_plots"
    method_dir = (
        output_dir
        / f"{artifact_prefix}_quality_aware_go_ic_by_method"
        / "current__adaptive_diffusion_cosine_subspace"
    )
    pages_dir = method_dir / "tree_pages"
    plots_dir.mkdir(parents=True, exist_ok=True)
    method_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    method_prefix = (
        f"{artifact_prefix}_quality_aware_go_ic_current__adaptive_diffusion_cosine_subspace"
    )
    ranking.to_csv(method_dir / f"{method_prefix}_tree_ranking.csv", index=False)

    pdf_path = method_dir / f"{method_prefix}_tree_pages.pdf"
    with PdfPages(pdf_path) as pdf:
        for row in ok.itertuples(index=False):
            rank = int(row.display_rank)
            title = (
                f"Rank {rank}: current adaptive diffusion cosine subspace, "
                f"{row.weighting} / {row.block_name}"
            )
            metrics = (
                "Method: current TBS gate on an average-linkage tree built from adaptive diffusion "
                "distances inside this cosine eigenspace subspace.\n"
                "Values: GO-BIC/gene is lower-is-better within quality tier; specificity score "
                "combines the fraction and strength of clusters with specific enriched GO terms.\n"
                f"Clusters: {int(row.n_clusters)}; specific clusters: {int(row.specific_cluster_count)}; "
                f"specificity score: {float(row.specificity_score):.3f}; "
                f"GO-BIC/gene: {float(row.go_bic_active_per_gene):.3f}."
            )

            linkage_path = Path(str(getattr(row, "linkage_matrix", "")))
            assignments_path = Path(str(getattr(row, "cluster_assignments", "")))
            fig = plt.figure(figsize=(18, 10.5))
            if linkage_path.exists() and assignments_path.exists():
                _draw_tree_subtree_clusters(
                    fig,
                    pd.read_csv(linkage_path).to_numpy(),
                    pd.read_csv(assignments_path),
                    f"{title}: tree with cluster assignments",
                )
            else:
                ax = fig.add_subplot(111)
                draw_image_panel(
                    ax,
                    getattr(row, "tree_subtree_clusters_png", "")
                    or getattr(row, "tree_dendrogram_png", ""),
                    f"{title}: tree with cluster assignments",
                )
            fig.text(
                0.02,
                0.02,
                textwrap.fill(metrics.replace("\n", " "), width=180),
                ha="left",
                va="bottom",
                fontsize=8.5,
            )
            fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
            page_path = pages_dir / (
                f"{rank:02d}_current__adaptive_diffusion_cosine_subspace__"
                f"{safe_name(row.weighting)}__{safe_name(row.block_name)}__tree_page.png"
            )
            fig.savefig(page_path, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(18, 9))
            draw_image_panel(
                axes[0],
                getattr(row, "subspace_embedding_annotated_terms_png", ""),
                "Subspace embedding",
            )
            draw_image_panel(
                axes[1],
                getattr(row, "adaptive_diffusion_embedding_annotated_terms_png", ""),
                "Adaptive diffusion embedding",
            )
            fig.suptitle(title, fontsize=14)
            fig.tight_layout()
            page_path = pages_dir / (
                f"{rank:02d}_current__adaptive_diffusion_cosine_subspace__"
                f"{safe_name(row.weighting)}__{safe_name(row.block_name)}__embedding_page.png"
            )
            fig.savefig(page_path, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)

            fig = plt.figure(figsize=(18, 10.5))
            loadings_path = Path(str(getattr(row, "axis_term_loadings_all", "")))
            if loadings_path.exists():
                draw_axis_terms_heatmap(
                    fig,
                    pd.read_csv(loadings_path),
                    f"{title}: strongest GO-term feature loadings",
                )
            else:
                ax = fig.add_subplot(111)
                draw_image_panel(
                    ax, getattr(row, "axis_terms_combined_png", ""), f"{title}: GO-term loadings"
                )
                fig.tight_layout()
            page_path = pages_dir / (
                f"{rank:02d}_current__adaptive_diffusion_cosine_subspace__"
                f"{safe_name(row.weighting)}__{safe_name(row.block_name)}__terms_page.png"
            )
            fig.savefig(page_path, dpi=180)
            pdf.savefig(fig)
            plt.close(fig)

    shutil.copyfile(
        pdf_path, plots_dir / f"{artifact_prefix}_quality_aware_go_ic_all_tree_pages.pdf"
    )
    ranking.to_csv(
        plots_dir / f"{artifact_prefix}_quality_aware_go_ic_tree_ranking.csv", index=False
    )

    (method_dir / "README.md").write_text(
        "\n".join(
            [
                "# Current Adaptive Diffusion Cosine-Subspace Results",
                "",
                f"- Main tree/subspace PDF: `{pdf_path.name}`",
                "- `tree_pages/`: two PNG pages per ranked subspace.",
                "- Ranking is specificity-aware within quality tier.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (plots_dir / "README.md").write_text(
        "\n".join(
            [
                "# Quality-Aware GO-IC Plots",
                "",
                f"- All tree/subspace pages PDF: `{artifact_prefix}_quality_aware_go_ic_all_tree_pages.pdf`",
                "- Method-separated results are under `../"
                f"{artifact_prefix}_quality_aware_go_ic_by_method/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "method_dir": str(method_dir),
        "method_tree_pages_pdf": str(pdf_path),
        "all_tree_pages_pdf": str(
            plots_dir / f"{artifact_prefix}_quality_aware_go_ic_all_tree_pages.pdf"
        ),
    }


def write_artifact_index(
    output_dir: Path,
    ranking: pd.DataFrame,
    artifact_prefix: str,
    workflow_outputs: dict[str, str],
) -> None:
    ranking.to_csv(output_dir / "artifact_index.csv", index=False)
    plot_rows = []
    for row in ranking.itertuples(index=False):
        plot_rows.append(
            {
                "display_rank": getattr(row, "display_rank", ""),
                "weighting": row.weighting,
                "block_name": row.block_name,
                "status": row.status,
                "tree_dendrogram_png": getattr(row, "tree_dendrogram_png", ""),
                "subspace_embedding_png": getattr(row, "subspace_embedding_png", ""),
                "adaptive_diffusion_embedding_png": getattr(
                    row, "adaptive_diffusion_embedding_png", ""
                ),
                "axis_terms_combined_png": getattr(row, "axis_terms_combined_png", ""),
            }
        )
    pd.DataFrame(plot_rows).to_csv(output_dir / "subspace_plot_index.csv", index=False)
    manifest = {
        "artifact_prefix": artifact_prefix,
        "result_prefix": RESULT_PREFIX,
        "ranking_note": (
            "specificity_aware_rank is the preferred reading order: quality tier, "
            "specificity score, specificity fraction, weighted specificity delta, "
            "then GO-BIC/gene. old_display_rank preserves the earlier quality-tiered GO-IC order."
        ),
        "ranking_csv": str(output_dir / "rankings" / f"{RESULT_PREFIX}_ranking.csv"),
        "specificity_ranking_csv": str(
            output_dir / "rankings" / f"{RESULT_PREFIX}_specificity_aware_ranking.csv"
        ),
        "workflow_outputs": workflow_outputs,
        "subspaces": ranking.to_dict(orient="records"),
    }
    (output_dir / "connected_results_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Artifact Index",
        "",
        "This experiment uses the full current adaptive-diffusion cosine-subspace tree pipeline.",
        "Every input matrix should produce this same directory structure.",
        "",
        "## Main Files",
        "",
        f"- [ranking CSV](rankings/{RESULT_PREFIX}_ranking.csv)",
        f"- [specificity-aware ranking CSV](rankings/{RESULT_PREFIX}_specificity_aware_ranking.csv)",
        "- [connected manifest](connected_results_manifest.json)",
        "- [artifact table](artifact_index.csv)",
        "- [subspace plot table](subspace_plot_index.csv)",
        "",
        "## PDFs",
        "",
        f"- [method tree-pages PDF]({Path(workflow_outputs['method_tree_pages_pdf']).relative_to(output_dir)})",
        f"- [all tree-pages PDF]({Path(workflow_outputs['all_tree_pages_pdf']).relative_to(output_dir)})",
        "",
        "## Subspaces",
        "",
        "- `subspaces/<weighting>/<block_name>/` contains coordinates, linkage tree, cluster assignments or failure status, GO-IC quality summary, coherence tables, TF-IDF quality, and axis term-loading files.",
    ]
    (output_dir / "ARTIFACT_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_prefix = matrix_slug(args.input, args.dataset_label)
    output_dir = args.output_dir or default_output_dir(args.input, args.dataset_label)
    rankings_dir = output_dir / "rankings"
    plots_dir = output_dir / "plots"
    subspaces_dir = output_dir / "subspaces"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    subspaces_dir.mkdir(parents=True, exist_ok=True)

    data = load_binary_matrix(args.input)
    term_metadata = build_term_metadata(data)
    config = {
        "input": str(args.input),
        "output_dir": str(output_dir),
        "artifact_prefix": artifact_prefix,
        "dataset_label": args.dataset_label or artifact_prefix,
        "method_version": "current",
        "tree_geometry": "adaptive_diffusion_cosine_subspace",
        "edge_alpha": float(args.edge_alpha),
        "sibling_alpha": float(args.sibling_alpha),
        "max_rank": int(args.max_rank),
        "min_segment_length": int(args.min_segment_length),
        "max_segments": int(args.max_segments),
        "diffusion_k_neighbors": int(args.diffusion_k_neighbors),
        "diffusion_time": int(args.diffusion_time),
        "diffusion_components": int(args.diffusion_components),
        "adaptive_bandwidth_type": args.adaptive_bandwidth_type,
        "adaptive_epsilon": args.adaptive_epsilon,
        "adaptive_metric": args.adaptive_metric,
        "weightings": list(args.weightings),
        "block_names": list(args.block_names or []),
        "top_terms_per_axis": int(args.top_terms_per_axis),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    summary_rows: list[dict[str, object]] = []
    cluster_coherence_frames: list[pd.DataFrame] = []
    tfidf_quality_frames: list[pd.DataFrame] = []
    block_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    allow_blocks = set(args.block_names or [])

    for weighting in args.weightings:
        print(f"[{weighting}] cosine eigendecomposition", flush=True)
        values = weight_feature_matrix(data, weighting)
        eigvals, eigvecs = cosine_eigendecomposition(values, args.max_rank)
        feature_loadings = component_feature_loadings(values, eigvals, eigvecs)
        total_energy = float(np.sum(eigvals))
        blocks, diagnostics = adaptive_spectral_blocks(
            eigvals,
            min_segment_length=args.min_segment_length,
            max_segments=args.max_segments,
        )
        for component, eigval in enumerate(eigvals, start=1):
            spectrum_rows.append(
                {
                    "weighting": weighting,
                    "component": int(component),
                    "eigenvalue": float(eigval),
                    "fraction_of_kept_operator_energy": float(eigval / total_energy)
                    if total_energy > 0
                    else math.nan,
                }
            )

        for block in blocks:
            if allow_blocks and block.block_name not in allow_blocks:
                continue
            subspace_id = f"{weighting}__{block.block_name}"
            subspace_safe = safe_name(subspace_id)
            print(f"[{subspace_id}] adaptive diffusion current TBS", flush=True)
            subspace_dir = subspaces_dir / safe_name(weighting) / safe_name(block.block_name)
            subspace_dir.mkdir(parents=True, exist_ok=True)
            coords = coordinates_for_block(eigvals, eigvecs, block)
            block_energy = (
                float(np.sum(eigvals[block.block_start - 1 : block.block_end]) / total_energy)
                if total_energy > 0
                else math.nan
            )
            block_record = {
                "run_id": f"current__adaptive_diffusion_cosine_subspace__{subspace_id}",
                "method_version": "current",
                "tree_geometry": "adaptive_diffusion_cosine_subspace",
                "weighting": weighting,
                **spectral_block_record(
                    block,
                    block_energy=block_energy,
                    diagnostics=diagnostics,
                ),
                "subspace_dir": str(subspace_dir),
            }
            block_rows.append(block_record)
            (subspace_dir / f"{subspace_safe}__metadata.json").write_text(
                json.dumps({**block_record, "spectral_block": asdict(block)}, indent=2),
                encoding="utf-8",
            )
            pd.DataFrame(
                coords,
                index=data.index.astype(str),
                columns=[
                    f"mode_{idx:02d}" for idx in range(block.block_start, block.block_end + 1)
                ],
            ).rename_axis("gene").reset_index().to_csv(
                subspace_dir / f"{subspace_safe}__subspace_coordinates.csv",
                index=False,
            )
            all_loadings, top_loadings = write_axis_loadings(
                output_dir=subspace_dir,
                subspace_id=subspace_safe,
                term_metadata=term_metadata,
                loadings=feature_loadings,
                block_start=block.block_start,
                block_end=block.block_end,
                top_n=args.top_terms_per_axis,
            )
            plot_axis_terms(
                all_loadings,
                subspace_dir / f"{subspace_safe}__axis_terms.png",
                top_n=min(args.top_terms_per_axis, 20),
            )
            plot_combined_axis_terms(
                all_loadings,
                subspace_dir / f"{subspace_safe}__axis_terms_combined.png",
                terms_per_axis=min(8, args.top_terms_per_axis),
                max_terms=60,
            )
            axis_term_lines = axis_term_summary_lines(top_loadings)
            plot_subspace_embedding(
                coords,
                None,
                subspace_dir / f"{subspace_safe}__subspace_embedding_terms_only.png",
                subspace_id,
                axis_term_lines=axis_term_lines,
            )

            distances: np.ndarray | None = None
            linkage_path = ""
            try:
                geometry = block_adaptive_diffusion_geometry(
                    coords,
                    k_neighbors=args.diffusion_k_neighbors,
                    diffusion_time=args.diffusion_time,
                    n_components=args.diffusion_components,
                    metric=args.adaptive_metric,
                    bandwidth_type=args.adaptive_bandwidth_type,
                    epsilon=args.adaptive_epsilon,
                )
                distances = geometry.distance_condensed
                diffusion_metadata = geometry.metadata
                z = linkage(distances, method="average")
                linkage_path = str(subspace_dir / f"{subspace_safe}__linkage_matrix.csv")
                pd.DataFrame(z, columns=["left", "right", "distance", "count"]).to_csv(
                    linkage_path,
                    index=False,
                )
                plot_dendrogram(
                    z,
                    subspace_dir / f"{subspace_safe}__tree_dendrogram.png",
                    f"{subspace_id}: adaptive diffusion average-linkage tree",
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic output records failures.
                diffusion_metadata = {}
                status = "failed_diffusion"
                error = repr(exc)
                labels = None
            else:
                try:
                    labels = run_current_kl(
                        data,
                        distances,
                        edge_alpha=args.edge_alpha,
                        sibling_alpha=args.sibling_alpha,
                    )
                    status = "ok"
                    error = ""
                except Exception as exc:  # noqa: BLE001 - diagnostic output records failures.
                    status = "failed_gate"
                    error = repr(exc)
                    labels = None

            (subspace_dir / f"{subspace_safe}__diffusion_metadata.json").write_text(
                json.dumps(diffusion_metadata, indent=2, default=str),
                encoding="utf-8",
            )

            if status == "ok":
                assert labels is not None
                assert distances is not None
                assignments = assignments_from_labels(labels, data.index)
                assignments_path = subspace_dir / f"{subspace_safe}__cluster_assignments.csv"
                assignments.to_csv(assignments_path, index=False)
                assignments["cluster_id"].value_counts().sort_index().rename_axis(
                    "cluster_id"
                ).reset_index(name="cluster_size").to_csv(
                    subspace_dir / f"{subspace_safe}__cluster_sizes.csv",
                    index=False,
                )
                plot_tree_subtree_clusters(
                    z,
                    assignments,
                    subspace_dir / f"{subspace_safe}__tree_subtree_clusters.png",
                    f"{subspace_id}: adaptive diffusion tree with current TBS cluster assignments",
                )
                plot_subspace_embedding(
                    coords,
                    labels,
                    subspace_dir / f"{subspace_safe}__subspace_embedding_clusters.png",
                    subspace_id,
                )
                plot_subspace_embedding(
                    coords,
                    labels,
                    subspace_dir
                    / f"{subspace_safe}__subspace_embedding_clusters_annotated_terms.png",
                    subspace_id,
                    axis_term_lines=axis_term_lines,
                )
                plot_diffusion_embedding(
                    distances,
                    labels,
                    subspace_dir / f"{subspace_safe}__adaptive_diffusion_embedding_clusters.png",
                    subspace_id,
                )
                plot_diffusion_embedding(
                    distances,
                    labels,
                    subspace_dir
                    / f"{subspace_safe}__adaptive_diffusion_embedding_clusters_annotated_terms.png",
                    subspace_id,
                    axis_term_lines=axis_term_lines,
                )
                info = go_information_criterion(data, labels)
                coherence = cluster_coherence(data, labels)
                tfidf_quality = tfidf_within_cosine_quality(data, labels)
                sizes = cluster_size_metrics(labels)
                coherence_frame = coherence["coherence_table"].copy()
                coherence_frame.insert(0, "run_id", block_record["run_id"])
                coherence_frame.to_csv(
                    subspace_dir / f"{subspace_safe}__cluster_coherence.csv", index=False
                )
                cluster_coherence_frames.append(coherence_frame)
                tfidf_frame = tfidf_quality["tfidf_quality_table"].copy()
                tfidf_frame.insert(0, "run_id", block_record["run_id"])
                tfidf_frame.to_csv(
                    subspace_dir / f"{subspace_safe}__tfidf_cluster_quality.csv", index=False
                )
                tfidf_quality_frames.append(tfidf_frame)
                rank_fields = {
                    **info,
                    **sizes,
                    "coherent_cluster_count": coherence["coherent_cluster_count"],
                    "coherent_cluster_fraction": coherence["coherent_cluster_fraction"],
                    "median_significant_terms_q05": coherence["median_significant_terms_q05"],
                    "min_cluster_q_value": coherence["min_cluster_q_value"],
                    "median_within_tfidf_cosine": tfidf_quality["median_within_tfidf_cosine"],
                    "weighted_mean_within_tfidf_cosine": tfidf_quality[
                        "weighted_mean_within_tfidf_cosine"
                    ],
                }
            else:
                assignments_path = None
                pd.DataFrame(
                    [
                        {
                            "status": status,
                            "error": error,
                            "note": "No cluster assignments were produced because the current TBS gate did not complete for this subspace.",
                        }
                    ]
                ).to_csv(subspace_dir / f"{subspace_safe}__failure_status.csv", index=False)
                rank_fields = {
                    "go_log_likelihood": math.nan,
                    "go_bic_active": math.nan,
                    "go_aic_active": math.nan,
                    "go_bic_full": math.nan,
                    "go_active_parameters": pd.NA,
                    "go_full_parameters": pd.NA,
                    "go_bic_active_per_gene": math.nan,
                    "n_clusters": pd.NA,
                    "largest_cluster_size": pd.NA,
                    "largest_cluster_fraction": math.nan,
                    "singleton_clusters": pd.NA,
                    "singleton_fraction": math.nan,
                    "singleton_gene_fraction": math.nan,
                    "median_cluster_size": math.nan,
                    "coherent_cluster_count": pd.NA,
                    "coherent_cluster_fraction": math.nan,
                    "median_significant_terms_q05": math.nan,
                    "min_cluster_q_value": math.nan,
                    "median_within_tfidf_cosine": math.nan,
                    "weighted_mean_within_tfidf_cosine": math.nan,
                }

            pd.DataFrame([rank_fields]).to_csv(
                subspace_dir / f"{subspace_safe}__go_ic_quality_summary.csv", index=False
            )
            top_loadings.to_csv(subspace_dir / f"{subspace_safe}__axis_top_terms.csv", index=False)
            summary_rows.append(
                {
                    **block_record,
                    "status": status,
                    "error": error,
                    "assignments_path": str(assignments_path)
                    if assignments_path is not None
                    else "",
                    "linkage_path": linkage_path,
                    "axis_top_terms_path": str(
                        subspace_dir / f"{subspace_safe}__axis_top_terms.csv"
                    ),
                    **rank_fields,
                }
            )

    ranking = pd.DataFrame.from_records(summary_rows)
    ok_mask = ranking["status"].eq("ok")
    if ok_mask.any():
        raw_order = ranking.loc[ok_mask].sort_values(
            [
                "go_bic_active",
                "coherent_cluster_fraction",
                "weighted_mean_within_tfidf_cosine",
                "singleton_gene_fraction",
            ],
            ascending=[True, False, False, True],
        )
        raw_rank = dict(zip(raw_order["run_id"], np.arange(1, len(raw_order) + 1), strict=False))
        tiers = ranking.loc[ok_mask].apply(quality_tier, axis=1, result_type="expand")
        ranking.loc[ok_mask, "quality_tier"] = tiers[0].astype(int).to_numpy()
        ranking.loc[ok_mask, "quality_tier_label"] = tiers[1].astype(str).to_numpy()
        ranking.loc[ok_mask, "raw_go_ic_rank"] = (
            ranking.loc[ok_mask, "run_id"].map(raw_rank).astype(int)
        )
        ok_sorted = ranking.loc[ok_mask].sort_values(
            [
                "quality_tier",
                "go_bic_active",
                "coherent_cluster_fraction",
                "weighted_mean_within_tfidf_cosine",
                "singleton_gene_fraction",
            ],
            ascending=[True, True, False, False, True],
        )
        display_rank = dict(
            zip(ok_sorted["run_id"], np.arange(1, len(ok_sorted) + 1), strict=False)
        )
        ranking.loc[ok_mask, "display_rank"] = (
            ranking.loc[ok_mask, "run_id"].map(display_rank).astype(int)
        )
        ranking = ranking.sort_values(
            ["status", "display_rank"], ascending=[False, True]
        ).reset_index(drop=True)
    coherence_long = (
        pd.concat(cluster_coherence_frames, ignore_index=True) if cluster_coherence_frames else None
    )
    ranking = attach_artifact_paths(ranking)
    ranking = add_specificity_aware_rank(ranking, coherence_long)
    ok_mask = ranking["status"].eq("ok")
    ranking_path = rankings_dir / f"{RESULT_PREFIX}_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    ranking.to_csv(rankings_dir / f"{RESULT_PREFIX}_specificity_aware_ranking.csv", index=False)
    pd.DataFrame(block_rows).to_csv(
        rankings_dir / f"{RESULT_PREFIX}_subspace_blocks.csv", index=False
    )
    pd.DataFrame(spectrum_rows).to_csv(rankings_dir / f"{RESULT_PREFIX}_spectrum.csv", index=False)
    if coherence_long is not None:
        coherence_long.to_csv(
            rankings_dir / f"{RESULT_PREFIX}_cluster_coherence_long.csv",
            index=False,
        )
    if tfidf_quality_frames:
        pd.concat(tfidf_quality_frames, ignore_index=True).to_csv(
            rankings_dir / f"{RESULT_PREFIX}_tfidf_quality_long.csv",
            index=False,
        )

    if ok_mask.any():
        top = ranking[ranking["status"].eq("ok")].sort_values("display_rank").head(20)
        fig, ax = plt.subplots(figsize=(12, max(5, 0.38 * len(top) + 2)))
        labels = [
            f"{int(row.display_rank):02d} {row.weighting} {row.block_name}"
            for row in top.itertuples(index=False)
        ]
        ax.barh(labels[::-1], top["go_bic_active_per_gene"].to_numpy()[::-1], color="#4c78a8")
        ax.set_title("Current adaptive diffusion subspace trees: specificity-aware rank")
        ax.set_xlabel("GO-BIC active per gene (lower is better within comparable quality tier)")
        ax.grid(axis="x", alpha=0.18)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{RESULT_PREFIX}_top_ranked_subspaces.png", dpi=180)
        plt.close(fig)

        pdf_path = output_dir / f"{RESULT_PREFIX}_axis_terms_by_subspace.pdf"
        with PdfPages(pdf_path) as pdf:
            for row in (
                ranking[ranking["status"].eq("ok")]
                .sort_values("display_rank")
                .itertuples(index=False)
            ):
                subspace_id = f"{row.weighting}__{row.block_name}"
                subspace_safe = safe_name(subspace_id)
                axis_path = Path(row.subspace_dir) / f"{subspace_safe}__axis_top_terms.csv"
                top_terms = pd.read_csv(axis_path)
                for axis, frame in top_terms[top_terms["selection"].eq("top_absolute")].groupby(
                    "axis", sort=True
                ):
                    plot_data = (
                        frame.sort_values("abs_loading", ascending=False)
                        .head(15)
                        .sort_values("loading")
                    )
                    fig, ax = plt.subplots(figsize=(11, 7))
                    colors = np.where(plot_data["loading"] >= 0, "#4c78a8", "#e45756")
                    ax.barh(
                        wrap_labels(plot_data["go_term"], width=42),
                        plot_data["loading"],
                        color=colors,
                    )
                    ax.axvline(0.0, color="#333333", linewidth=0.8)
                    ax.set_title(
                        f"Rank {int(row.display_rank)}: {subspace_id}, mode {int(axis):02d}\n"
                        "Top absolute GO-term loadings for this subspace axis"
                    )
                    ax.set_xlabel("feature loading")
                    ax.tick_params(axis="y", labelsize=7)
                    ax.grid(axis="x", alpha=0.18)
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
        write_combined_axis_terms_pdf(ranking, output_dir)

    workflow_outputs = write_workflow_pdfs(output_dir, ranking, artifact_prefix)
    write_artifact_index(output_dir, ranking, artifact_prefix, workflow_outputs)

    readme_lines = [
        "# Current Adaptive Diffusion Subspace Tree Experiment",
        "",
        f"Input: `{args.input}`",
        "Method version: `current`",
        "Tree geometry: `adaptive_diffusion_cosine_subspace`",
        f"Experiment directory: `{output_dir}`",
        "",
        "Directory layout:",
        "- `rankings/`: experiment-level ranking, spectrum, block metadata, and long quality tables.",
        "- `plots/`: experiment-level summary plots.",
        "- `subspaces/<weighting>/<block_name>/`: one folder per subspace with assignments, tree, quality CSVs, coordinates, and axis term-loading plots.",
        f"- `{artifact_prefix}_quality_aware_go_ic_by_method/current__adaptive_diffusion_cosine_subspace/`: method-separated PDF, PNG pages, and copied ranking CSV.",
        f"- `{artifact_prefix}_quality_aware_go_ic_plots/`: all-tree/ordered PDFs for the experiment.",
        "- `ARTIFACT_INDEX.md`, `artifact_index.csv`, `subspace_plot_index.csv`, and `connected_results_manifest.json`: connected artifact tables.",
        "",
        "Ranking:",
        "- `display_rank` equals `specificity_aware_rank` for completed rows.",
        "- `specificity_aware_rank` sorts by quality tier, cluster specificity score, specific-cluster fraction, weighted specificity delta, then lower GO-BIC active/gene.",
        "- `old_display_rank` preserves the older quality-tier then GO-BIC order.",
        "- `raw_go_ic_rank` preserves the raw GO-IC order for audit.",
        "- `go_bic_active_per_gene` is lower-is-better only within comparable quality tiers.",
        "",
        "Axis term loadings:",
        "- `axis_term_loadings_all.csv` stores every GO term loading for every cosine mode in the subspace.",
        "- `axis_top_terms.csv` stores the top positive, negative, and absolute GO-term loadings per axis.",
        "- Positive and negative signs are orientation-dependent; the absolute loading is the stable strength score.",
        "",
        f"Ranking CSV: `{ranking_path}`",
        f"Artifact index: `{output_dir / 'ARTIFACT_INDEX.md'}`",
    ]
    if ok_mask.any():
        best = ranking[ranking["status"].eq("ok")].sort_values("display_rank").iloc[0]
        readme_lines.extend(
            [
                "",
                "Top ranked subspace:",
                f"- `{best['weighting']} / {best['block_name']}`",
                f"- clusters: `{int(best['n_clusters'])}`",
                f"- quality tier: `{best['quality_tier_label']}`",
                f"- GO-BIC active/gene: `{float(best['go_bic_active_per_gene']):.6f}`",
                f"- coherent clusters: `{int(best['coherent_cluster_count'])}/{int(best['n_clusters'])}`",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"Wrote current adaptive diffusion subspace tree experiment: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
