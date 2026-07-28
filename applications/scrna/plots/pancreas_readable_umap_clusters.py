"""Readable UMAP panels for pancreas cluster assignments.

Every final TBS cluster is colored. Labels are restricted to larger clusters so
the plot remains readable, but small clusters are not greyed out or hidden.
"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


METHODS = [
    (
        "tbs_topology_projected_adaptive_k90_alpha0p01_edge0p001",
        "Topology-only TBS",
    ),
    (
        "tbs_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001",
        "Recomputed NNLS branch-time TBS",
    ),
    (
        "tbs_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001",
        "Raw-linkage branch-time diagnostic TBS",
    ),
    (
        "tbs_adaptive_diffusion_topology_projected_adaptive_k90_alpha0p01_edge0p001",
        "Adaptive diffusion topology TBS",
    ),
    (
        "tbs_adaptive_diffusion_branch_time_recomputed_nnls_projected_adaptive_k90_alpha0p01_edge0p001",
        "Adaptive diffusion recomputed NNLS branch-time TBS",
    ),
    (
        "tbs_adaptive_diffusion_raw_linkage_branch_time_diagnostic_projected_adaptive_k90_alpha0p01_edge0p001",
        "Adaptive diffusion raw-linkage branch-time diagnostic TBS",
    ),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def label_positions(frame: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    rows = []
    for cluster_id, group in frame.groupby(cluster_column):
        rows.append(
            {
                "cluster_id": cluster_id,
                "x": float(group["umap1"].median()),
                "y": float(group["umap2"].median()),
                "n_cells": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def cluster_palette(cluster_ids: pd.Index | list[int]) -> dict[int, tuple[float, float, float]]:
    ids = sorted(int(cluster_id) for cluster_id in cluster_ids)
    palette: dict[int, tuple[float, float, float]] = {}
    for index, cluster_id in enumerate(ids):
        hue = ((index * 0.38196601125) + 0.03) % 1.0
        saturation = [0.92, 0.78, 0.86, 0.70][index % 4]
        value = [0.72, 0.86, 0.62, 0.78][index % 4]
        palette[cluster_id] = tuple(mcolors.hsv_to_rgb((hue, saturation, value)))
    return palette


def padded_limits(values: pd.Series, fraction: float = 0.025) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    padding = max((high - low) * fraction, 1e-6)
    return low - padding, high + padding


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
    args = parser.parse_args()

    output_dir = args.output_dir
    assignments = pd.read_csv(output_dir / "method_assignments.csv")

    min_label_size = 50
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(8.5, 21), sharex=True, sharey=True)
    x_limits = padded_limits(assignments["umap1"])
    y_limits = padded_limits(assignments["umap2"])

    summary_rows: list[dict[str, object]] = []
    for ax, (column, title) in zip(axes, METHODS, strict=True):
        values = assignments[column].astype(int)
        sizes = values.value_counts()
        all_clusters = sizes.sort_index().index.to_list()
        labeled_clusters = sizes[sizes >= min_label_size].index.to_list()
        palette = cluster_palette(all_clusters)
        frame = assignments.copy()
        frame["_cluster"] = values

        for cluster_id in all_clusters:
            cluster_frame = frame[frame["_cluster"] == cluster_id]
            ax.scatter(
                cluster_frame["umap1"],
                cluster_frame["umap2"],
                s=9,
                c=[palette[int(cluster_id)]],
                alpha=0.9,
                linewidths=0,
            )

        labels = label_positions(frame[frame["_cluster"].isin(labeled_clusters)], "_cluster")
        for _, row in labels.iterrows():
            ax.text(
                row["x"],
                row["y"],
                f"C{int(row['cluster_id'])}\n{int(row['n_cells'])}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "#111827",
                    "linewidth": 0.25,
                    "alpha": 0.72,
                },
            )

        summary_rows.append(
            {
                "method": title,
                "cluster_column": column,
                "total_clusters": int(sizes.size),
                "clusters_colored": int(sizes.size),
                "clusters_labeled_ge_50_cells": int(len(labeled_clusters)),
                "cells_in_labeled_clusters_ge_50": int(sizes.loc[labeled_clusters].sum()),
                "cells_in_unlabeled_clusters": int(len(frame) - sizes.loc[labeled_clusters].sum()),
            }
        )

        ax.set_title(
            (
                f"{title}: {sizes.size} clusters, "
                f"{len(labeled_clusters)} clusters >= {min_label_size} cells"
            ),
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("UMAP2")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(False)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("UMAP1")
    fig.suptitle(
        "Readable TBS clusters on pancreas UMAP\nall final clusters colored; large clusters labeled",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = output_dir / "tbs_readable_umap_clusters_all_colored.png"
    pdf_path = output_dir / "tbs_readable_umap_clusters_all_colored.pdf"
    fig.savefig(png_path, dpi=260)
    fig.savefig(pdf_path)
    plt.close(fig)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "tbs_readable_umap_clusters_all_colored_summary.csv", index=False)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
