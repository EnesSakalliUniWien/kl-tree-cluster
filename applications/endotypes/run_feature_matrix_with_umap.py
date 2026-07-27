#!/usr/bin/env python3
"""Run TBS-tree decomposition on a TSV feature matrix and export UMAP results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_RUNTIME_CACHE_ROOT = Path(tempfile.gettempdir()) / "tbs_runtime_cache"
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
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tree_break_selection import config
from tree_break_selection.hierarchy_analysis.cluster_assignments import (
    build_sample_cluster_assignments,
)
from tree_break_selection.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from tree_break_selection.plot.cluster_color_mapping import (
    build_cluster_color_spec,
    present_cluster_ids,
)
from tree_break_selection.plot.cluster_tree_visualization import plot_tree_with_clusters
from tree_break_selection.space_separation import (
    adaptive_diffusion_distance,
    hamming_knn_diffusion_distance,
)
from tree_break_selection.tree.io import tree_from_linkage
from tree_break_selection.tree.poset_tree import PosetTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TBS tree decomposition on a TSV matrix and generate UMAP outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix.tsv"),
        help="Path to TSV file where rows are samples and columns are binary features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "benchmarks/results/03_gene_go_feature_matrix_runs/feature_matrix_<timestamp>/"
        ),
    )
    parser.add_argument(
        "--edge-alpha",
        type=float,
        default=float(DEFAULT_EDGE_ALPHA),
        help="Edge-divergence significance alpha.",
    )
    parser.add_argument(
        "--sibling-alpha",
        type=float,
        default=float(DEFAULT_SIBLING_ALPHA),
        help="Sibling-divergence significance alpha.",
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
        choices=("umap", "svd"),
        default="umap",
        help="Embedding for the cluster plot.",
    )
    parser.add_argument(
        "--tree-method",
        choices=("tbs", "tbs_diffusion", "tbs_diffusion_adaptive", "paper_cosine_complete"),
        default="tbs",
        help="Tree construction method: standard TBS tree, fixed-k diffusion tree, or adaptive diffusion tree.",
    )
    parser.add_argument(
        "--tree-distance-metric",
        choices=("hamming", "rogerstanimoto", "jaccard", "dice", "euclidean", "cosine"),
        default=config.TREE_DISTANCE_METRIC,
        help="Pairwise distance metric for the TBS tree path.",
    )
    parser.add_argument(
        "--tree-linkage-method",
        choices=("average", "complete", "single", "ward"),
        default=config.TREE_LINKAGE_METHOD,
        help="Hierarchical linkage method for the TBS tree path.",
    )
    parser.add_argument(
        "--reference-endotypes",
        type=Path,
        default=None,
        help="Optional supplementary endotype file for congruence analysis against paper labels.",
    )
    parser.add_argument(
        "--paper-cluster-count",
        type=int,
        default=None,
        help="Cluster count for paper_cosine_complete mode. Defaults to the number of matched reference endotypes when available.",
    )
    parser.add_argument(
        "--flat-cluster-count",
        type=int,
        default=None,
        help="Optional exact flat cut K for non-paper tree methods. Uses the chosen tree geometry but skips TBS gating.",
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
        default=15,
        help="Sparse neighbor support for adaptive diffusion.",
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
    return (
        Path("benchmarks/results")
        / "03_gene_go_feature_matrix_runs"
        / f"feature_matrix_{_timestamp()}"
    )


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
    edge_alpha: float,
    sibling_alpha: float,
    tree_method: str,
    tree_distance_metric: str,
    tree_linkage_method: str,
    diffusion_k_neighbors: int,
    diffusion_time: int,
    diffusion_components: int,
    adaptive_neighbor_k: int,
    adaptive_bandwidth_type: str,
    adaptive_epsilon: str,
    adaptive_metric: str,
) -> tuple[PosetTree, dict[str, object], pd.DataFrame]:
    (
        linkage_matrix,
        distance_metric,
        linkage_method,
        adaptive_metadata,
    ) = _build_linkage_tree(
        data_df=data_df,
        tree_method=tree_method,
        tree_distance_metric=tree_distance_metric,
        tree_linkage_method=tree_linkage_method,
        diffusion_k_neighbors=diffusion_k_neighbors,
        diffusion_time=diffusion_time,
        diffusion_components=diffusion_components,
        adaptive_neighbor_k=adaptive_neighbor_k,
        adaptive_bandwidth_type=adaptive_bandwidth_type,
        adaptive_epsilon=adaptive_epsilon,
        adaptive_metric=adaptive_metric,
    )

    tree = tree_from_linkage(linkage_matrix, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df)
    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        leaf_data=data_df,
    )
    decomposition = tree.decompose(
        gate_annotation_bundle=gate_bundle,
        leaf_data=data_df,
        edge_alpha=edge_alpha,
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


def _build_linkage_tree(
    *,
    data_df: pd.DataFrame,
    tree_method: str,
    tree_distance_metric: str,
    tree_linkage_method: str,
    diffusion_k_neighbors: int,
    diffusion_time: int,
    diffusion_components: int,
    adaptive_neighbor_k: int,
    adaptive_bandwidth_type: str,
    adaptive_epsilon: str,
    adaptive_metric: str,
) -> tuple[np.ndarray, str, str, dict[str, object] | None]:
    adaptive_metadata: dict[str, object] | None = None
    if tree_method == "tbs_diffusion":
        distance_condensed = hamming_knn_diffusion_distance(
            data_df,
            k_neighbors=diffusion_k_neighbors,
            diffusion_time=diffusion_time,
            n_components=diffusion_components,
        )
        linkage_method = "average"
        distance_metric = "diffusion"
    elif tree_method == "tbs_diffusion_adaptive":
        distance_condensed, adaptive_metadata = adaptive_diffusion_distance(
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
        if tree_linkage_method == "ward" and tree_distance_metric != "euclidean":
            raise ValueError(
                "Ward linkage requires euclidean distance. "
                f"Got tree_distance_metric={tree_distance_metric!r}."
            )
        distance_condensed = pdist(data_df.values, metric=tree_distance_metric)
        linkage_method = tree_linkage_method
        distance_metric = tree_distance_metric

    linkage_matrix = linkage(distance_condensed, method=linkage_method)
    return linkage_matrix, distance_metric, linkage_method, adaptive_metadata


def _assignments_from_flat_cut(
    data_df: pd.DataFrame,
    *,
    linkage_matrix: np.ndarray,
    cluster_count: int,
) -> pd.DataFrame:
    if cluster_count < 2:
        raise ValueError(f"flat cluster count must be at least 2, got {cluster_count}")

    raw_labels = cut_tree(linkage_matrix, n_clusters=[int(cluster_count)]).reshape(-1)
    cluster_ids, _ = pd.factorize(raw_labels, sort=True)
    assignments = pd.DataFrame(
        {
            "cluster_id": cluster_ids.astype(int),
            "cluster_root": pd.NA,
        },
        index=data_df.index,
    )
    cluster_sizes = assignments["cluster_id"].value_counts()
    assignments["cluster_size"] = assignments["cluster_id"].map(cluster_sizes).astype(int)
    return assignments


def _run_paper_cosine_clustering(
    data_df: pd.DataFrame,
    *,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], pd.DataFrame]:
    if cluster_count < 2:
        raise ValueError(f"paper cluster count must be at least 2, got {cluster_count}")

    distance_condensed = pdist(data_df.values, metric="cosine")
    linkage_matrix = linkage(distance_condensed, method="complete")
    raw_labels = cut_tree(linkage_matrix, n_clusters=[int(cluster_count)]).reshape(-1)
    cluster_ids, _ = pd.factorize(raw_labels, sort=True)
    labels = cluster_ids.astype(int)

    assignments = pd.DataFrame(
        {
            "cluster_id": labels,
            "cluster_root": pd.NA,
        },
        index=data_df.index,
    )
    cluster_sizes = assignments["cluster_id"].value_counts()
    assignments["cluster_size"] = assignments["cluster_id"].map(cluster_sizes).astype(int)

    metadata = {
        "tree_method": "paper_cosine_complete",
        "distance_metric": "cosine",
        "linkage_method": "complete",
        "target_clusters": int(cluster_count),
    }
    return linkage_matrix, labels, metadata, assignments


def _parse_reference_endotypes(path: Path) -> dict[str, dict[str, object]]:
    entrez_to_endotype: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            first_cell = row[0].strip() if row else ""
            if first_cell.startswith("#") or first_cell in {"SUM", "AVERAGE"}:
                continue
            if len(row) < 10:
                continue
            cluster_id_text = row[4].strip()
            if not cluster_id_text.isdigit():
                continue
            endotype = {
                "cluster_id": int(cluster_id_text),
                "color": row[5].strip() or "#808080",
                "rank": row[6].strip(),
                "name": row[7].strip() or f"Cluster {cluster_id_text}",
                "type": row[8].strip() or "cluster",
            }
            for gene_id in (cell.strip() for cell in row[9:] if cell.strip()):
                entrez_to_endotype[str(gene_id)] = endotype
    return entrez_to_endotype


def _map_symbols_to_entrez(symbols: list[str], batch_size: int = 200) -> dict[str, str | None]:
    mapped: dict[str, str | None] = {}
    endpoint = "https://mygene.info/v3/query"
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        body = urlencode(
            {
                "q": ",".join(batch),
                "scopes": "symbol",
                "fields": "symbol,entrezgene,taxid",
                "species": "human",
                "size": 1,
            }
        ).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                "Failed to query mygene.info for gene symbol -> Entrez mapping."
            ) from exc

        if isinstance(payload, dict):
            payload = [payload]
        for record in payload:
            query = record.get("query")
            if not query:
                continue
            entrez = record.get("entrezgene") or record.get("_id")
            mapped[str(query)] = None if entrez is None else str(entrez)
        for symbol in batch:
            mapped.setdefault(symbol, None)
    return mapped


def _build_reference_comparison(
    assignments: pd.DataFrame,
    reference_endotypes_path: Path,
    output_dir: Path,
) -> tuple[dict[str, object], int | None]:
    symbol_to_entrez = _map_symbols_to_entrez(assignments.index.tolist())
    entrez_to_endotype = _parse_reference_endotypes(reference_endotypes_path)

    comparison = assignments.copy()
    comparison.index.name = "gene_symbol"
    comparison["entrez_id"] = comparison.index.map(symbol_to_entrez)

    endotype_meta = comparison["entrez_id"].map(entrez_to_endotype)
    comparison["reference_cluster_id"] = endotype_meta.map(
        lambda x: x["cluster_id"] if isinstance(x, dict) else pd.NA
    )
    comparison["reference_cluster_name"] = endotype_meta.map(
        lambda x: x["name"] if isinstance(x, dict) else "Unassigned"
    )
    comparison["reference_cluster_type"] = endotype_meta.map(
        lambda x: x["type"] if isinstance(x, dict) else "unassigned"
    )
    comparison["reference_cluster_color"] = endotype_meta.map(
        lambda x: x["color"] if isinstance(x, dict) else "#bdbdbd"
    )
    comparison["matched_reference_label"] = comparison["reference_cluster_id"].notna()

    alignment_path = output_dir / "reference_endotype_alignment.csv"
    comparison.to_csv(alignment_path)

    matched = comparison[comparison["matched_reference_label"]].copy()
    if matched.empty:
        metrics = {
            "reference_endotypes_path": str(reference_endotypes_path),
            "matched_genes": 0,
            "unmatched_genes": int(len(comparison)),
            "n_reference_clusters_observed": 0,
            "ari": None,
            "nmi": None,
            "alignment_csv": str(alignment_path),
            "confusion_matrix_csv": None,
        }
        return metrics, None

    matched["reference_cluster_id"] = matched["reference_cluster_id"].astype(int)
    reference_cluster_count = int(matched["reference_cluster_id"].nunique())
    confusion = pd.crosstab(
        matched["cluster_id"].astype(int),
        matched["reference_cluster_name"].astype(str),
        rownames=["our_cluster"],
        colnames=["reference_endotype"],
    )
    confusion_path = output_dir / "reference_endotype_confusion_matrix.csv"
    confusion.to_csv(confusion_path)

    ari_value = float(
        adjusted_rand_score(
            matched["reference_cluster_id"].to_numpy(),
            matched["cluster_id"].astype(int).to_numpy(),
        )
    )
    nmi_value = float(
        normalized_mutual_info_score(
            matched["reference_cluster_id"].to_numpy(),
            matched["cluster_id"].astype(int).to_numpy(),
        )
    )
    metrics = {
        "reference_endotypes_path": str(reference_endotypes_path),
        "matched_genes": int(len(matched)),
        "unmatched_genes": int(len(comparison) - len(matched)),
        "n_reference_clusters_observed": reference_cluster_count,
        "ari": ari_value,
        "nmi": nmi_value,
        "alignment_csv": str(alignment_path),
        "confusion_matrix_csv": str(confusion_path),
    }
    return metrics, reference_cluster_count


def _compute_embedding(
    data_df: pd.DataFrame,
    method: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> tuple[np.ndarray, str, tuple[str, str], str | None]:
    if method == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(data_df.values)
        return embedding, "umap", ("UMAP-1", "UMAP-2"), None

    from sklearn.decomposition import TruncatedSVD

    reducer = TruncatedSVD(n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(data_df.values)
    return embedding, "svd", ("SVD-1", "SVD-2"), None


def _save_embedding_plot(
    data_df: pd.DataFrame,
    labels: np.ndarray,
    output_path: Path,
    coordinates_path: Path,
    method: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    label_title: str,
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
    ax.set_title(f"{title_prefix} projection colored by {label_title}")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    if cluster_ids:
        if len(cluster_ids) > 20:
            top_cluster_ids = list(
                cluster_sizes.sort_values(ascending=False).head(20).index.astype(int)
            )
            legend_title = "Largest TBS clusters"
        else:
            top_cluster_ids = cluster_ids
            legend_title = "TBS clusters"

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


def _save_linkage_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=None, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Genes")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


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
    reference_metrics: dict[str, object] | None = None
    inferred_reference_cluster_count: int | None = None
    if args.reference_endotypes is not None:
        if not args.reference_endotypes.exists():
            raise FileNotFoundError(
                f"Reference endotypes file not found: {args.reference_endotypes}"
            )
        print(f"Reference endotypes: {args.reference_endotypes}")
    if args.tree_method == "tbs_diffusion":
        print(
            "Diffusion params: "
            f"k_neighbors={args.diffusion_k_neighbors}, "
            f"diffusion_time={args.diffusion_time}, "
            f"n_components={args.diffusion_components}"
        )
    if args.tree_method == "tbs":
        print(
            "TBS tree params: "
            f"distance_metric={args.tree_distance_metric}, "
            f"linkage_method={args.tree_linkage_method}"
        )
    if args.tree_method == "tbs_diffusion_adaptive":
        print(
            "Adaptive diffusion params: "
            f"neighbor_k={args.adaptive_neighbor_k}, "
            f"bandwidth_type={args.adaptive_bandwidth_type}, "
            f"epsilon={args.adaptive_epsilon}, "
            f"metric={args.adaptive_metric}, "
            f"diffusion_time={args.diffusion_time}, "
            f"n_components={args.diffusion_components}"
        )
    if args.tree_method == "paper_cosine_complete" and args.reference_endotypes is not None:
        reference_metrics, inferred_reference_cluster_count = _build_reference_comparison(
            pd.DataFrame(
                {
                    "cluster_id": np.zeros(len(data_df), dtype=int),
                    "cluster_root": pd.NA,
                    "cluster_size": np.ones(len(data_df), dtype=int),
                },
                index=data_df.index,
            ),
            args.reference_endotypes,
            output_dir,
        )

    if args.tree_method == "paper_cosine_complete":
        target_clusters = args.paper_cluster_count
        if target_clusters is None:
            target_clusters = inferred_reference_cluster_count
        if target_clusters is None:
            raise ValueError(
                "paper_cosine_complete requires either --paper-cluster-count "
                "or --reference-endotypes with at least one matched reference cluster."
            )
        print(
            "Running paper-faithful clustering: "
            f"cosine distance + complete linkage + maxclust={int(target_clusters)}"
        )
        linkage_matrix, labels, tree_metadata, assignments = _run_paper_cosine_clustering(
            data_df,
            cluster_count=int(target_clusters),
        )
        decomposition = {
            "num_clusters": int(assignments["cluster_id"].nunique()),
            "_tree_method_metadata": tree_metadata,
        }
        tree = None
        decomp_elapsed = time.perf_counter() - t0
        print(f"Paper-faithful clustering complete in {decomp_elapsed:.2f}s")
        if args.reference_endotypes is not None:
            reference_metrics, _ = _build_reference_comparison(
                assignments,
                args.reference_endotypes,
                output_dir,
            )
    elif args.flat_cluster_count is not None:
        print(
            "Running flat-cut clustering on selected tree geometry: "
            f"tree_method={args.tree_method}, K={int(args.flat_cluster_count)}"
        )
        (
            linkage_matrix,
            distance_metric,
            linkage_method,
            adaptive_metadata,
        ) = _build_linkage_tree(
            data_df=data_df,
            tree_method=args.tree_method,
            tree_distance_metric=args.tree_distance_metric,
            tree_linkage_method=args.tree_linkage_method,
            diffusion_k_neighbors=args.diffusion_k_neighbors,
            diffusion_time=args.diffusion_time,
            diffusion_components=args.diffusion_components,
            adaptive_neighbor_k=args.adaptive_neighbor_k,
            adaptive_bandwidth_type=args.adaptive_bandwidth_type,
            adaptive_epsilon=args.adaptive_epsilon,
            adaptive_metric=args.adaptive_metric,
        )
        assignments = _assignments_from_flat_cut(
            data_df,
            linkage_matrix=linkage_matrix,
            cluster_count=int(args.flat_cluster_count),
        )
        labels = assignments["cluster_id"].to_numpy()
        decomposition = {
            "num_clusters": int(assignments["cluster_id"].nunique()),
            "_tree_method_metadata": {
                "tree_method": f"{args.tree_method}_flat_cut",
                "distance_metric": distance_metric,
                "linkage_method": linkage_method,
                "diffusion_k_neighbors": int(args.diffusion_k_neighbors),
                "diffusion_time": int(args.diffusion_time),
                "diffusion_components": int(args.diffusion_components),
                "adaptive_diffusion": adaptive_metadata,
                "flat_cluster_count": int(args.flat_cluster_count),
            },
        }
        tree = None
        decomp_elapsed = time.perf_counter() - t0
        print(f"Flat-cut clustering complete in {decomp_elapsed:.2f}s")
        if args.reference_endotypes is not None:
            reference_metrics, _ = _build_reference_comparison(
                assignments,
                args.reference_endotypes,
                output_dir,
            )
    else:
        print("Running TBS decomposition...")
        tree, decomposition, assignments = _run_decomposition(
            data_df=data_df,
            edge_alpha=args.edge_alpha,
            sibling_alpha=args.sibling_alpha,
            tree_method=args.tree_method,
            tree_distance_metric=args.tree_distance_metric,
            tree_linkage_method=args.tree_linkage_method,
            diffusion_k_neighbors=args.diffusion_k_neighbors,
            diffusion_time=args.diffusion_time,
            diffusion_components=args.diffusion_components,
            adaptive_neighbor_k=args.adaptive_neighbor_k,
            adaptive_bandwidth_type=args.adaptive_bandwidth_type,
            adaptive_epsilon=args.adaptive_epsilon,
            adaptive_metric=args.adaptive_metric,
        )
        linkage_matrix = None
        decomp_elapsed = time.perf_counter() - t0
        print(f"Decomposition complete in {decomp_elapsed:.2f}s")
        labels = assignments["cluster_id"].to_numpy()
        if args.reference_endotypes is not None:
            reference_metrics, _ = _build_reference_comparison(
                assignments,
                args.reference_endotypes,
                output_dir,
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
        "edge_alpha": (
            float(args.edge_alpha)
            if args.tree_method != "paper_cosine_complete" and args.flat_cluster_count is None
            else None
        ),
        "sibling_alpha": (
            float(args.sibling_alpha)
            if args.tree_method != "paper_cosine_complete" and args.flat_cluster_count is None
            else None
        ),
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
    if args.flat_cluster_count is not None:
        summary["flat_cluster_count"] = int(args.flat_cluster_count)
    if reference_metrics is not None:
        summary["reference_comparison"] = reference_metrics
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
        label_title=(
            "paper-faithful clusters"
            if args.tree_method == "paper_cosine_complete"
            else "TBS clusters"
        ),
    )
    embedding_elapsed = time.perf_counter() - embedding_start
    print(f"Embedding export complete in {embedding_elapsed:.2f}s")
    summary["embedding_method"] = embedding_method_used
    if embedding_note is not None:
        summary["embedding_note"] = embedding_note
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tree_pdf_path: Path | None = None
    if args.tree_method == "paper_cosine_complete":
        print("Rendering dendrogram...")
        tree_plot_start = time.perf_counter()
        _save_linkage_dendrogram(
            linkage_matrix,
            tree_png_path,
            title=(
                "Paper-faithful hierarchical clustering "
                f"(cosine + complete, K={decomposition.get('num_clusters', 'NA')})"
            ),
        )
        tree_elapsed = time.perf_counter() - tree_plot_start
        print(f"Dendrogram export complete in {tree_elapsed:.2f}s")
    elif args.flat_cluster_count is not None:
        print("Rendering dendrogram...")
        tree_plot_start = time.perf_counter()
        _save_linkage_dendrogram(
            linkage_matrix,
            tree_png_path,
            title=(
                f"Flat-cut tree — {args.tree_method}, "
                f"K={decomposition.get('num_clusters', 'NA')}"
            ),
        )
        tree_elapsed = time.perf_counter() - tree_plot_start
        print(f"Dendrogram export complete in {tree_elapsed:.2f}s")
    else:
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
            title=f"TBS Tree — {n_clusters} clusters (α={args.sibling_alpha})",
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
    if tree_pdf_path is not None:
        print(f"  tree_pdf:     {tree_pdf_path}")
    if reference_metrics is not None:
        print(f"  ref_match_n:  {reference_metrics['matched_genes']}")
        print(f"  ref_ARI:      {reference_metrics['ari']}")
        print(f"  ref_NMI:      {reference_metrics['nmi']}")
        print(f"  ref_align:    {reference_metrics['alignment_csv']}")
        if reference_metrics["confusion_matrix_csv"] is not None:
            print(f"  ref_confuse:  {reference_metrics['confusion_matrix_csv']}")


if __name__ == "__main__":
    main()
