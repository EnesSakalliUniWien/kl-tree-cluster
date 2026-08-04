"""Render all-method pancreas UMAP panels with every cluster colored."""
# ruff: noqa: I001

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METADATA_COLUMNS = {"cell_id", "celltype", "umap1", "umap2"}
METHOD_LABELS = {
    "tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001": "TBS topology",
    "tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001": (
        "TBS recomputed NNLS branch-time"
    ),
    "tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001": (
        "TBS raw-linkage diagnostic"
    ),
    "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001": (
        "TBS adaptive diffusion topology"
    ),
    "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001": (
        "TBS adaptive diffusion recomputed NNLS"
    ),
    "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001": (
        "TBS adaptive diffusion raw-linkage diagnostic"
    ),
    "leiden": "Leiden",
    "louvain": "Louvain",
    "k_means_true_k": "K-means true K",
    "spectral_true_k": "Spectral true K",
    "hdbscan": "HDBSCAN",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def assignment_key(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def padded_limits(values: pd.Series, fraction: float = 0.025) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    padding = max((high - low) * fraction, 1e-6)
    return low - padding, high + padding


def categorical_palette(ids: list[str]) -> dict[str, tuple[float, float, float]]:
    ordered_ids = sorted(ids, key=lambda value: (value.startswith("-"), value))
    palette: dict[str, tuple[float, float, float]] = {}
    for index, item_id in enumerate(ordered_ids):
        hue = ((index * 0.38196601125) + 0.03) % 1.0
        saturation = [0.92, 0.78, 0.86, 0.70][index % 4]
        value = [0.72, 0.86, 0.62, 0.78][index % 4]
        palette[item_id] = tuple(mcolors.hsv_to_rgb((hue, saturation, value)))
    return palette


def cluster_label(cluster_id: str) -> str:
    return "noise" if cluster_id == "-1" else f"C{cluster_id}"


def draw_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    title: str,
    subtitle: str,
    min_label_size: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    cluster_id_labels: bool,
) -> dict[str, object]:
    value_strings = values.astype(str)
    counts = value_strings.value_counts()
    palette = categorical_palette(counts.index.to_list())

    for cluster_id in counts.sort_index().index:
        mask = value_strings == cluster_id
        ax.scatter(
            frame.loc[mask, "umap1"],
            frame.loc[mask, "umap2"],
            s=9,
            c=[palette[str(cluster_id)]],
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )

    labeled_ids = counts[counts >= min_label_size].index
    for cluster_id in labeled_ids:
        mask = value_strings == cluster_id
        x = float(frame.loc[mask, "umap1"].median())
        y = float(frame.loc[mask, "umap2"].median())
        ax.text(
            x,
            y,
            (
                f"{cluster_label(str(cluster_id))}\n{int(counts.loc[cluster_id])}"
                if cluster_id_labels
                else f"{cluster_id}\n{int(counts.loc[cluster_id])}"
            ),
            ha="center",
            va="center",
            fontsize=6.2,
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "#111827",
                "linewidth": 0.25,
                "alpha": 0.76,
            },
        )

    ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, fontweight="bold")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(False)
    return {
        "panel": title,
        "clusters": int(counts.size),
        "clusters_colored": int(counts.size),
        "clusters_labeled_ge_50_cells": int(len(labeled_ids)),
        "cells_in_labeled_clusters_ge_50": int(counts.loc[labeled_ids].sum())
        if len(labeled_ids)
        else 0,
        "cells_in_unlabeled_clusters": int(len(frame) - counts.loc[labeled_ids].sum())
        if len(labeled_ids)
        else int(len(frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root()
            / "raw"
            / "assets"
            / "benchmark-results"
            / "pancreas_scrna_cluster_benchmark_20260623"
        ),
    )
    parser.add_argument("--title", default="Pancreas scRNA clustering UMAPs")
    args = parser.parse_args()

    output_dir = args.output_dir
    assignments = pd.read_csv(output_dir / "method_assignments.csv")
    metrics = pd.read_csv(output_dir / "method_metrics.csv")
    metric_by_key = {
        assignment_key(str(row.method)): row for row in metrics.itertuples(index=False)
    }

    method_columns = [
        column
        for column in assignments.columns
        if column not in METADATA_COLUMNS and column in METHOD_LABELS
    ]
    panel_count = len(method_columns) + 1
    ncols = 3
    nrows = int(np.ceil(panel_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 5.8 * nrows))
    flat_axes = np.asarray(axes).reshape(-1)
    x_limits = padded_limits(assignments["umap1"])
    y_limits = padded_limits(assignments["umap2"])
    min_label_size = 50

    summary_rows: list[dict[str, object]] = []
    celltype_values = assignments["celltype"].astype(str)
    summary_rows.append(
        draw_panel(
            flat_axes[0],
            assignments,
            celltype_values,
            title="Curated cell type",
            subtitle=f"{celltype_values.nunique()} labels; all labels colored",
            min_label_size=min_label_size,
            x_limits=x_limits,
            y_limits=y_limits,
            cluster_id_labels=False,
        )
    )

    for ax, column in zip(flat_axes[1:], method_columns, strict=False):
        values = pd.to_numeric(assignments[column], errors="coerce").fillna(-9999).astype(int)
        metric = metric_by_key.get(column)
        if metric is None:
            subtitle = f"{values.nunique()} clusters; all clusters colored"
        else:
            subtitle = (
                f"{int(metric.n_clusters)} clusters; purity {metric.weighted_cluster_purity:.3f}; "
                f"recall {metric.weighted_label_dominant_cluster_recall:.3f}; V {metric.v_measure:.3f}"
            )
        summary = draw_panel(
            ax,
            assignments,
            values,
            title=METHOD_LABELS[column],
            subtitle=subtitle,
            min_label_size=min_label_size,
            x_limits=x_limits,
            y_limits=y_limits,
            cluster_id_labels=True,
        )
        summary["cluster_column"] = column
        summary_rows.append(summary)

    for ax in flat_axes[panel_count:]:
        ax.axis("off")

    fig.suptitle(
        f"{args.title}\nall assigned clusters colored; labels only for clusters >= 50 cells",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    png_path = output_dir / "all_methods_umap_clusters_all_colored.png"
    pdf_path = output_dir / "all_methods_umap_clusters_all_colored.pdf"
    fig.savefig(png_path, dpi=260)
    fig.savefig(pdf_path)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "all_methods_umap_clusters_all_colored_summary.csv",
        index=False,
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
