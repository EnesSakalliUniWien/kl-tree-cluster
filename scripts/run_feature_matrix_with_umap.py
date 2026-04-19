#!/usr/bin/env python3
"""Run KL-tree decomposition on a TSV feature matrix and export UMAP results."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_RUNTIME_CACHE_ROOT = Path(tempfile.gettempdir()) / "kl_te_runtime_cache"
for _cache_dir in (
    _RUNTIME_CACHE_ROOT / "numba",
    _RUNTIME_CACHE_ROOT / "matplotlib",
    _RUNTIME_CACHE_ROOT / "xdg",
):
    _cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("NUMBA_CACHE_DIR", str(_RUNTIME_CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_CACHE_ROOT / "xdg"))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import TruncatedSVD

from benchmarks.shared.runners.kl_diffusion_runner import (
    _build_adaptive_diffusion_distance,
    _build_diffusion_distance,
)
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from kl_clustering_analysis.plot.cluster_color_mapping import (
    build_cluster_color_spec,
    present_cluster_ids,
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
        default=float(config.EDGE_ALPHA),
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
    parser.add_argument(
        "--embedding-method",
        choices=("auto", "umap", "svd"),
        default="auto",
        help="Embedding for the cluster plot. 'auto' tries UMAP and falls back to SVD.",
    )
    parser.add_argument(
        "--tree-method",
        choices=("kl", "kl_diffusion", "kl_diffusion_adaptive"),
        default="kl",
        help="Tree construction method: standard KL tree, fixed-k diffusion tree, or adaptive diffusion tree.",
    )
    parser.add_argument(
        "--diffusion-k-neighbors",
        type=int,
        default=15,
        help="k-NN neighborhood size for the diffusion tree path.",
    )
    parser.add_argument(
        "--diffusion-time",
        type=int,
        default=3,
        help="Diffusion time exponent for the diffusion tree path.",
    )
    parser.add_argument(
        "--diffusion-components",
        type=int,
        default=30,
        help="Number of diffusion eigencomponents to keep.",
    )
    parser.add_argument(
        "--adaptive-neighbor-k",
        type=int,
        default=None,
        help="Optional sparse neighbor support for adaptive diffusion. Omit to choose automatically.",
    )
    parser.add_argument(
        "--adaptive-bandwidth-type",
        default="-1/(d+2)",
        help="pydiffmap bandwidth_type for adaptive diffusion, e.g. '-1/(d+2)'.",
    )
    parser.add_argument(
        "--adaptive-epsilon",
        default="median",
        help="Adaptive diffusion epsilon. Use a float or one of: median, mean, q75, bgh.",
    )
    parser.add_argument(
        "--adaptive-metric",
        choices=("hamming", "euclidean"),
        default="hamming",
        help="Distance metric for the adaptive diffusion kernel.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _default_output_dir() -> Path:
    return Path("benchmarks/results") / f"feature_matrix_{_timestamp()}"


def _configure_runtime_cache_dirs() -> Path:
    """Route third-party caches to a writable temp location.

    This avoids sandbox/home-directory cache failures from Numba, UMAP,
    Matplotlib, and fontconfig-backed libraries.
    """
    return _RUNTIME_CACHE_ROOT


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
    tree_method: str,
    diffusion_k_neighbors: int,
    diffusion_time: int,
    diffusion_components: int,
    adaptive_neighbor_k: int | None,
    adaptive_bandwidth_type: str,
    adaptive_epsilon: str,
    adaptive_metric: str,
) -> tuple[PosetTree, dict[str, object], pd.DataFrame]:
    adaptive_metadata: dict[str, object] | None = None
    if tree_method == "kl_diffusion":
        distance_condensed = _build_diffusion_distance(
            data_df,
            k_neighbors=diffusion_k_neighbors,
            diffusion_time=diffusion_time,
            n_components=diffusion_components,
        )
        linkage_method = "average"
        distance_metric = "diffusion"
    elif tree_method == "kl_diffusion_adaptive":
        distance_condensed, adaptive_metadata = _build_adaptive_diffusion_distance(
            data_df,
            k_neighbors=adaptive_neighbor_k,
            diffusion_time=diffusion_time,
            n_components=diffusion_components,
            metric=adaptive_metric,
            bandwidth_type=adaptive_bandwidth_type,
            epsilon=adaptive_epsilon,
            return_metadata=True,
        )
        linkage_method = "average"
        distance_metric = "adaptive_diffusion"
    else:
        distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
        linkage_method = config.TREE_LINKAGE_METHOD
        distance_metric = config.TREE_DISTANCE_METRIC

    linkage_matrix = linkage(distance_condensed, method=linkage_method)

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
    decomposition.setdefault("_tree_method_metadata", {})
    decomposition["_tree_method_metadata"] = {
        "tree_method": tree_method,
        "distance_metric": distance_metric,
        "linkage_method": linkage_method,
        "diffusion_k_neighbors": int(diffusion_k_neighbors),
        "diffusion_time": int(diffusion_time),
        "diffusion_components": int(diffusion_components),
        "adaptive_diffusion": adaptive_metadata,
    }
    return tree, decomposition, assignments


def _compute_embedding(
    data_df: pd.DataFrame,
    method: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[np.ndarray, str, tuple[str, str], str | None]:
    fallback_reason: str | None = None

    if method in {"auto", "umap"}:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            embedding = reducer.fit_transform(data_df.values)
            return embedding, "umap", ("UMAP-1", "UMAP-2"), None
        except Exception as exc:
            if method == "umap":
                raise RuntimeError(f"UMAP embedding failed: {exc}") from exc
            fallback_reason = f"UMAP unavailable; fell back to TruncatedSVD ({exc})"

    reducer = TruncatedSVD(n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(data_df.values)
    return embedding, "svd", ("SVD-1", "SVD-2"), fallback_reason


def _save_embedding_plot(
    data_df: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
    coordinates_path: Path,
    method: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[str, str | None]:
    import matplotlib.pyplot as plt

    embedding, method_used, axis_labels, embedding_note = _compute_embedding(
        data_df=data_df,
        method=method,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    cluster_ids = present_cluster_ids(labels)
    n_clusters = (max(cluster_ids) + 1) if cluster_ids else 0
    color_spec = build_cluster_color_spec(n_clusters)
    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    coordinates = pd.DataFrame(
        embedding,
        index=data_df.index,
        columns=[axis_labels[0], axis_labels[1]],
    )
    coordinates.insert(0, "cluster_id", labels)
    coordinates.to_csv(coordinates_path, sep="\t")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap=color_spec.cmap,
        norm=color_spec.norm,
        s=35,
        alpha=0.9,
    )
    title_prefix = "UMAP" if method_used == "umap" else "Truncated SVD"
    ax.set_title(f"{title_prefix} projection colored by KL clusters")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if cluster_ids:
        if len(cluster_ids) > 20:
            top_cluster_ids = list(
                cluster_sizes.sort_values(ascending=False).head(20).index.astype(int)
            )
            legend_title = "Largest KL clusters"
        else:
            top_cluster_ids = cluster_ids
            legend_title = "KL clusters"

        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=color_spec.id_to_color[cluster_id],
                markeredgecolor="none",
                label=f"Cluster {cluster_id} (n={int(cluster_sizes.get(cluster_id, 0))})",
            )
            for cluster_id in top_cluster_ids
        ]
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return method_used, embedding_note


def main() -> None:
    args = parse_args()
    cache_root = _configure_runtime_cache_dirs()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = args.output_dir if args.output_dir is not None else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_df = _load_binary_matrix(input_path)
    t0 = time.perf_counter()
    print(
        f"Loaded input: {data_df.shape[0]} samples x {data_df.shape[1]} features "
        f"from {input_path}"
    )
    print(f"Tree method: {args.tree_method}")
    if args.tree_method == "kl_diffusion":
        print(
            "Diffusion params: "
            f"k_neighbors={args.diffusion_k_neighbors}, "
            f"diffusion_time={args.diffusion_time}, "
            f"n_components={args.diffusion_components}"
        )
    if args.tree_method == "kl_diffusion_adaptive":
        print(
            "Adaptive diffusion params: "
            f"neighbor_k={args.adaptive_neighbor_k}, "
            f"bandwidth_type={args.adaptive_bandwidth_type}, "
            f"epsilon={args.adaptive_epsilon}, "
            f"metric={args.adaptive_metric}, "
            f"diffusion_time={args.diffusion_time}, "
            f"n_components={args.diffusion_components}"
        )
    print("Running KL decomposition...")
    tree, decomposition, assignments = _run_decomposition(
        data_df=data_df,
        alpha_local=args.alpha_local,
        sibling_alpha=args.sibling_alpha,
        tree_method=args.tree_method,
        diffusion_k_neighbors=args.diffusion_k_neighbors,
        diffusion_time=args.diffusion_time,
        diffusion_components=args.diffusion_components,
        adaptive_neighbor_k=args.adaptive_neighbor_k,
        adaptive_bandwidth_type=args.adaptive_bandwidth_type,
        adaptive_epsilon=args.adaptive_epsilon,
        adaptive_metric=args.adaptive_metric,
    )
    decomp_elapsed = time.perf_counter() - t0
    print(f"Decomposition complete in {decomp_elapsed:.2f}s")

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
    embedding_png_path = output_dir / "embedding_clusters.png"
    embedding_coordinates_path = output_dir / "embedding_coordinates.tsv"
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
        "tree_method": args.tree_method,
        "distance_metric": decomposition["_tree_method_metadata"]["distance_metric"],
        "linkage_method": decomposition["_tree_method_metadata"]["linkage_method"],
        "runtime_cache_root": str(cache_root),
        "diffusion_params": {
            "k_neighbors": int(args.diffusion_k_neighbors),
            "diffusion_time": int(args.diffusion_time),
            "n_components": int(args.diffusion_components),
        },
        "adaptive_diffusion": decomposition["_tree_method_metadata"].get("adaptive_diffusion"),
        "output_files": {
            "cluster_assignments_csv": str(assignments_path),
            "data_with_clusters_tsv": str(annotated_data_path),
            "cluster_sizes_csv": str(cluster_sizes_path),
            "embedding_png": str(embedding_png_path),
            "embedding_coordinates_tsv": str(embedding_coordinates_path),
            "tree_png": str(tree_png_path),
        },
    }
    print("Rendering embedding plot...")
    embedding_start = time.perf_counter()
    embedding_method_used, embedding_note = _save_embedding_plot(
        data_df=data_df,
        labels=labels,
        output_path=embedding_png_path,
        coordinates_path=embedding_coordinates_path,
        method=args.embedding_method,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.umap_random_state,
    )
    embedding_elapsed = time.perf_counter() - embedding_start
    print(f"Embedding export complete in {embedding_elapsed:.2f}s")
    summary["embedding_method"] = embedding_method_used
    if embedding_note is not None:
        summary["embedding_note"] = embedding_note
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Tree visualization
    n_clusters = decomposition.get("num_clusters", 0)
    n_leaves = data_df.shape[0]
    tree_h = max(10, n_leaves * 0.06)  # scale height with leaf count
    print("Rendering tree plot...")
    tree_plot_start = time.perf_counter()
    import matplotlib.pyplot as plt

    fig_tree, ax_tree = plt.subplots(figsize=(16, tree_h))
    plot_tree_with_clusters(
        tree,
        decomposition,
        annotations_df=tree.annotations_df,
        layout="rectangular",
        title=f"KL Tree — {n_clusters} clusters (α={args.sibling_alpha})",
        ax=ax_tree,
        node_size=12,
        font_size=9,
    )
    if n_clusters > 20:
        legend = ax_tree.get_legend()
        if legend is not None:
            legend.remove()
        ax_tree.text(
            1.01,
            0.98,
            f"{n_clusters} clusters\nSee cluster_sizes.csv for full sizes.",
            transform=ax_tree.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )
    fig_tree.tight_layout()
    fig_tree.savefig(tree_png_path, dpi=200, bbox_inches="tight")
    # Also save PDF for vector quality
    tree_pdf_path = output_dir / "tree_clusters.pdf"
    fig_tree.savefig(tree_pdf_path, bbox_inches="tight")
    plt.close(fig_tree)
    tree_elapsed = time.perf_counter() - tree_plot_start
    print(f"Tree export complete in {tree_elapsed:.2f}s")

    print("Run complete")
    print(f"  input:        {input_path}")
    print(f"  output_dir:   {output_dir}")
    print(f"  n_samples:    {data_df.shape[0]}")
    print(f"  n_features:   {data_df.shape[1]}")
    print(f"  tree_method:  {args.tree_method}")
    print(f"  num_clusters: {decomposition.get('num_clusters', 'NA')}")
    print(f"  assignments:  {assignments_path}")
    print(f"  data+clust:   {annotated_data_path}")
    print(f"  sizes:        {cluster_sizes_path}")
    print(f"  summary:      {summary_path}")
    print(f"  embedding:    {embedding_png_path}")
    print(f"  embed_tsv:    {embedding_coordinates_path}")
    print(f"  embed_method: {embedding_method_used}")
    if embedding_note is not None:
        print(f"  embed_note:   {embedding_note}")
    print(f"  tree_png:     {tree_png_path}")
    print(f"  tree_pdf:     {tree_pdf_path}")


if __name__ == "__main__":
    main()
