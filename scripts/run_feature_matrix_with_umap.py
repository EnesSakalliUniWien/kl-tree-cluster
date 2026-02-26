#!/usr/bin/env python3
"""Run KL-tree decomposition on a TSV feature matrix and export UMAP results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from kl_clustering_analysis.plot.cluster_tree_visualization import plot_tree_with_clusters
from kl_clustering_analysis.tree.io import tree_from_linkage
from kl_clustering_analysis.tree.poset_tree import PosetTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KL tree decomposition on a TSV matrix and generate UMAP outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("feature_matrix.tsv"),
        help="Path to TSV file where rows are samples and columns are binary features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to benchmarks/results/feature_matrix_<timestamp>/",
    )
    parser.add_argument(
        "--alpha-local",
        type=float,
        default=float(config.ALPHA_LOCAL),
        help="Gate 2 significance alpha (child-parent).",
    )
    parser.add_argument(
        "--sibling-alpha",
        type=float,
        default=float(config.SIBLING_ALPHA),
        help="Gate 3 significance alpha (sibling).",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=42,
        help="UMAP random_state.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _default_output_dir() -> Path:
    return Path("benchmarks/results") / f"feature_matrix_{_timestamp()}"


def _load_binary_matrix(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", index_col=0)
    if data.empty:
        raise ValueError(f"Input matrix is empty: {path}")

    numeric = data.apply(pd.to_numeric, errors="raise")
    unique_values = np.unique(numeric.values)
    invalid_values = [v for v in unique_values if v not in (0, 1)]
    if invalid_values:
        raise ValueError(
            "Input contains non-binary values. "
            f"Found values outside {{0,1}}: {invalid_values[:10]}"
        )
    return numeric.astype(int)


def _run_decomposition(
    data_df: pd.DataFrame,
    alpha_local: float,
    sibling_alpha: float,
) -> tuple[PosetTree, dict[str, object], pd.DataFrame]:
    distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    linkage_matrix = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)

    tree = tree_from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    decomposition = tree.decompose(
        leaf_data=data_df,
        alpha_local=alpha_local,
        sibling_alpha=sibling_alpha,
    )

    assignments = build_sample_cluster_assignments(decomposition)
    if assignments.empty:
        raise ValueError("No cluster assignments were produced by decomposition.")

    if set(assignments.index) != set(data_df.index):
        missing = sorted(set(data_df.index) - set(assignments.index))
        extra = sorted(set(assignments.index) - set(data_df.index))
        raise ValueError(
            "Assignment sample IDs do not match input matrix rows. "
            f"Missing: {missing[:10]}, Extra: {extra[:10]}"
        )

    assignments = assignments.loc[data_df.index]
    return tree, decomposition, assignments


def _save_umap_plot(
    data_df: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> None:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP package is not installed. Install with: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(data_df.values)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab20",
        s=35,
        alpha=0.9,
    )
    ax.set_title("UMAP projection colored by KL clusters")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("cluster_id")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_df = _load_binary_matrix(input_path)
    tree, decomposition, assignments = _run_decomposition(
        data_df=data_df,
        alpha_local=args.alpha_local,
        sibling_alpha=args.sibling_alpha,
    )

    labels = assignments["cluster_id"].to_numpy()
    cluster_sizes = (
        assignments.groupby("cluster_id", as_index=False)
        .size()
        .rename(columns={"size": "n_samples"})
        .sort_values("cluster_id")
    )

    assignments_path = output_dir / "cluster_assignments.csv"
    cluster_sizes_path = output_dir / "cluster_sizes.csv"
    annotated_data_path = output_dir / "data_with_clusters.tsv"
    summary_path = output_dir / "summary.json"
    umap_png_path = output_dir / "umap_clusters.png"
    tree_png_path = output_dir / "tree_clusters.png"

    assignments.to_csv(assignments_path)
    cluster_sizes.to_csv(cluster_sizes_path, index=False)

    # Join cluster assignments back to original data table
    annotated = data_df.copy()
    annotated.insert(0, "cluster_id", assignments["cluster_id"])
    annotated = annotated.sort_values("cluster_id")
    annotated.to_csv(annotated_data_path, sep="\t")

    summary = {
        "input_path": str(input_path),
        "n_samples": int(data_df.shape[0]),
        "n_features": int(data_df.shape[1]),
        "num_clusters": int(decomposition.get("num_clusters", -1)),
        "alpha_local": float(args.alpha_local),
        "sibling_alpha": float(args.sibling_alpha),
        "distance_metric": config.TREE_DISTANCE_METRIC,
        "linkage_method": config.TREE_LINKAGE_METHOD,
        "output_files": {
            "cluster_assignments_csv": str(assignments_path),
            "data_with_clusters_tsv": str(annotated_data_path),
            "cluster_sizes_csv": str(cluster_sizes_path),
            "umap_png": str(umap_png_path),
            "tree_png": str(tree_png_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _save_umap_plot(
        data_df=data_df,
        labels=labels,
        output_path=umap_png_path,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.umap_random_state,
    )

    # Tree visualization
    n_clusters = decomposition.get("num_clusters", 0)
    n_leaves = data_df.shape[0]
    tree_h = max(10, n_leaves * 0.06)  # scale height with leaf count
    fig_tree, ax_tree = plt.subplots(figsize=(16, tree_h))
    plot_tree_with_clusters(
        tree,
        decomposition,
        results_df=tree.stats_df,
        layout="rectangular",
        title=f"KL Tree — {n_clusters} clusters (α={args.sibling_alpha})",
        ax=ax_tree,
        node_size=12,
        font_size=9,
    )
    fig_tree.tight_layout()
    fig_tree.savefig(tree_png_path, dpi=200)
    # Also save PDF for vector quality
    tree_pdf_path = output_dir / "tree_clusters.pdf"
    fig_tree.savefig(tree_pdf_path)
    plt.close(fig_tree)

    print("Run complete")
    print(f"  input:        {input_path}")
    print(f"  output_dir:   {output_dir}")
    print(f"  n_samples:    {data_df.shape[0]}")
    print(f"  n_features:   {data_df.shape[1]}")
    print(f"  num_clusters: {decomposition.get('num_clusters', 'NA')}")
    print(f"  assignments:  {assignments_path}")
    print(f"  data+clust:   {annotated_data_path}")
    print(f"  sizes:        {cluster_sizes_path}")
    print(f"  summary:      {summary_path}")
    print(f"  umap:         {umap_png_path}")
    print(f"  tree_png:     {tree_png_path}")
    print(f"  tree_pdf:     {tree_pdf_path}")


if __name__ == "__main__":
    main()
