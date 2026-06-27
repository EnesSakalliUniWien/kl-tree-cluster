"""Cluster features inside selected raw KAK/cosine eigenvariant subspaces.

This is the feature-side counterpart to the existing gene-side KAK lens
diagnostics. For a row-normalized weighted feature matrix ``Z`` and
``K = Z Z.T = U Lambda U.T``, the raw feature axes are

    Q = Z.T U Lambda^{-1/2}.

For each selected spectral block ``B``, this diagnostic clusters feature rows
in the block embedding ``Q_B Lambda_B^{1/2}``. The leaves are features, not
genes. Outputs are diagnostic-only and do not promote selected-subspace
calibration to production.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
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
from tree_break_selection.tree.feature_space import continuous_feature_space_from_columns
from tree_break_selection.tree.poset_tree import PosetTree

from benchmarks.diagnostics.spectral.adaptive_cosine_kak_benchmark_probe import (
    SpectralBlock,
    adaptive_spectral_blocks,
    cosine_eigendecomposition,
    sibling_method_counts,
    weighted_matrix,
)
from benchmarks.diagnostics.spectral.adaptive_cosine_kak_matrix_probe import load_matrix
from benchmarks.diagnostics.spectral.kak_lens_feature_axis_clustering import (
    feature_axes_from_cosine_eigenvectors,
    row_normalized_weighted_matrix,
)

SCHEMA_VERSION = "kak_feature_subspace_clustering/v1"


@dataclass(frozen=True)
class FeatureSubspaceResult:
    """Summary for one feature-side eigenvariant clustering run."""

    schema_version: str
    weighting: str
    block_name: str
    block_start: int
    block_end: int
    subspace_dimensions: int
    active_features: int
    total_features: int
    active_feature_energy_fraction: float
    tree_linkage_method: str
    edge_alpha: float
    sibling_alpha: float
    status: str
    n_feature_clusters: int | None
    singleton_feature_fraction: float
    largest_feature_cluster_fraction: float
    runtime_sec: float
    sibling_test_method_counts: str
    feature_coordinates_path: str
    feature_assignments_path: str
    feature_cluster_summary_path: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster features inside raw KAK/cosine eigenvariant subspaces."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--weightings", nargs="+", default=["binary"], choices=["binary", "tfidf"])
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument(
        "--max-active-features",
        type=int,
        default=2000,
        help="Top feature rows by block energy to cluster. Use 0 to include all nonzero rows.",
    )
    parser.add_argument("--tree-linkage-method", default="average", choices=["average", "complete", "ward"])
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    return parser.parse_args()


def default_output_dir(input_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return (
        Path("benchmarks/results/diagnostics")
        / f"kak_feature_subspace_clustering_{input_path.stem}_{stamp}"
    )


def feature_coordinates_for_block(
    *,
    features: pd.Index,
    eigvals: np.ndarray,
    feature_axes: np.ndarray,
    block: SpectralBlock,
) -> pd.DataFrame:
    """Return feature coordinates in ``Q_B Lambda_B^{1/2}``."""

    start = block.block_start - 1
    end = block.block_end
    block_eigvals = np.asarray(eigvals[start:end], dtype=float)
    if np.any(block_eigvals <= 0.0):
        raise ValueError(f"Block {block.block_name!r} contains non-positive eigenvalues.")
    coords = feature_axes[:, start:end] * np.sqrt(block_eigvals)[np.newaxis, :]
    columns = [f"mode_{component:02d}" for component in range(block.block_start, block.block_end + 1)]
    return pd.DataFrame(coords, index=features.astype(str), columns=columns)


def select_active_feature_coordinates(
    coordinates: pd.DataFrame,
    *,
    max_active_features: int,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Select feature rows with nonzero block energy, capped by descending energy."""

    if max_active_features < 0:
        raise ValueError("max_active_features must be non-negative.")
    energy = pd.Series(
        np.sum(np.square(coordinates.to_numpy(dtype=float)), axis=1),
        index=coordinates.index,
        name="block_feature_energy",
    ).sort_values(ascending=False)
    nonzero = energy[energy > 1e-15]
    if nonzero.empty:
        raise ValueError("No features have positive energy in this subspace.")
    selected = nonzero if max_active_features == 0 else nonzero.head(max_active_features)
    active_fraction = float(selected.sum() / energy.sum()) if float(energy.sum()) > 0.0 else math.nan
    return coordinates.loc[selected.index], selected, active_fraction


def build_feature_tree(
    feature_coordinates: pd.DataFrame,
    *,
    tree_linkage_method: str,
) -> PosetTree:
    values = feature_coordinates.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Feature subspace tree requires at least three active features.")
    if not np.isfinite(values).all():
        raise ValueError("Feature subspace coordinates contain non-finite values.")

    if tree_linkage_method in {"average", "complete"}:
        distances = pdist(values, metric="euclidean")
        if not np.isfinite(distances).all() or np.allclose(distances, 0.0):
            raise ValueError("Degenerate feature subspace distances.")
        linkage_matrix = linkage(distances, method=tree_linkage_method)
    elif tree_linkage_method == "ward":
        if np.allclose(values, values[0]):
            raise ValueError("Degenerate feature subspace coordinates for Ward linkage.")
        linkage_matrix = linkage(values, method="ward", metric="euclidean")
    else:
        raise ValueError(f"Unsupported tree_linkage_method={tree_linkage_method!r}.")
    return PosetTree.from_linkage(linkage_matrix, leaf_names=feature_coordinates.index.tolist())


def write_feature_cluster_summary(
    *,
    assignments: pd.DataFrame,
    feature_energy: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frame = assignments.copy()
    frame["block_feature_energy"] = feature_energy.reindex(frame.index).astype(float)
    for cluster_id, group in frame.groupby("cluster_id", sort=True):
        rows.append(
            {
                "feature_cluster_id": int(cluster_id),
                "cluster_size": int(group.shape[0]),
                "cluster_energy": float(group["block_feature_energy"].sum()),
                "top_features": ";".join(
                    group.sort_values("block_feature_energy", ascending=False).head(12).index.astype(str)
                ),
            }
        )
    summary = pd.DataFrame.from_records(rows).sort_values(
        ["cluster_energy", "cluster_size"],
        ascending=False,
    )
    summary.to_csv(output_path, index=False)
    return summary


def run_feature_block(
    *,
    data: pd.DataFrame,
    weighting: str,
    block: SpectralBlock,
    eigvals: np.ndarray,
    feature_axes: np.ndarray,
    output_dir: Path,
    max_active_features: int,
    tree_linkage_method: str,
    edge_alpha: float,
    sibling_alpha: float,
) -> FeatureSubspaceResult:
    start_sec = time.perf_counter()
    safe_run_id = f"{weighting}__{block.block_name}".replace("/", "_")
    block_dir = output_dir / safe_run_id
    block_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = block_dir / "feature_subspace_coordinates.csv"
    assignments_path = block_dir / "feature_cluster_assignments.csv"
    cluster_summary_path = block_dir / "feature_cluster_summary.csv"

    feature_coordinates = feature_coordinates_for_block(
        features=data.columns,
        eigvals=eigvals,
        feature_axes=feature_axes,
        block=block,
    )
    active_coordinates, active_energy, active_energy_fraction = select_active_feature_coordinates(
        feature_coordinates,
        max_active_features=max_active_features,
    )
    active_coordinates.assign(block_feature_energy=active_energy).to_csv(coordinates_path)

    try:
        tree = build_feature_tree(
            active_coordinates,
            tree_linkage_method=tree_linkage_method,
        )
        feature_space = continuous_feature_space_from_columns(active_coordinates.columns)
        tree.populate_node_divergences(active_coordinates, feature_space=feature_space)
        gate_bundle = run_gate_annotation_pipeline(
            tree,
            tree.annotations_df.copy(),
            edge_alpha=edge_alpha,
            sibling_alpha=sibling_alpha,
            leaf_data=active_coordinates,
            feature_space=feature_space,
        )
        decomposition = tree.decompose(
            gate_annotation_bundle=gate_bundle,
            leaf_data=active_coordinates,
            feature_space=feature_space,
            edge_alpha=edge_alpha,
            sibling_alpha=sibling_alpha,
        )
        assignments = build_sample_cluster_assignments(decomposition).loc[active_coordinates.index]
        assignments.to_csv(assignments_path)
        write_feature_cluster_summary(
            assignments=assignments,
            feature_energy=active_energy,
            output_path=cluster_summary_path,
        )
        cluster_sizes = assignments["cluster_id"].value_counts()
        return FeatureSubspaceResult(
            schema_version=SCHEMA_VERSION,
            weighting=weighting,
            block_name=block.block_name,
            block_start=int(block.block_start),
            block_end=int(block.block_end),
            subspace_dimensions=int(block.block_end - block.block_start + 1),
            active_features=int(active_coordinates.shape[0]),
            total_features=int(data.shape[1]),
            active_feature_energy_fraction=active_energy_fraction,
            tree_linkage_method=tree_linkage_method,
            edge_alpha=float(edge_alpha),
            sibling_alpha=float(sibling_alpha),
            status="ok",
            n_feature_clusters=int(cluster_sizes.shape[0]),
            singleton_feature_fraction=float((cluster_sizes == 1).sum() / max(cluster_sizes.shape[0], 1)),
            largest_feature_cluster_fraction=float(cluster_sizes.max() / active_coordinates.shape[0]),
            runtime_sec=float(time.perf_counter() - start_sec),
            sibling_test_method_counts=sibling_method_counts(tree.annotations_df),
            feature_coordinates_path=str(coordinates_path),
            feature_assignments_path=str(assignments_path),
            feature_cluster_summary_path=str(cluster_summary_path),
            error="",
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic rows preserve failures.
        pd.DataFrame().to_csv(assignments_path)
        pd.DataFrame().to_csv(cluster_summary_path)
        return FeatureSubspaceResult(
            schema_version=SCHEMA_VERSION,
            weighting=weighting,
            block_name=block.block_name,
            block_start=int(block.block_start),
            block_end=int(block.block_end),
            subspace_dimensions=int(block.block_end - block.block_start + 1),
            active_features=int(active_coordinates.shape[0]),
            total_features=int(data.shape[1]),
            active_feature_energy_fraction=active_energy_fraction,
            tree_linkage_method=tree_linkage_method,
            edge_alpha=float(edge_alpha),
            sibling_alpha=float(sibling_alpha),
            status="failed_gate",
            n_feature_clusters=None,
            singleton_feature_fraction=math.nan,
            largest_feature_cluster_fraction=math.nan,
            runtime_sec=float(time.perf_counter() - start_sec),
            sibling_test_method_counts="",
            feature_coordinates_path=str(coordinates_path),
            feature_assignments_path=str(assignments_path),
            feature_cluster_summary_path=str(cluster_summary_path),
            error=repr(exc),
        )


def run_feature_subspace_clustering(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = args.output_dir or default_output_dir(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_matrix(args.input)
    rows: list[dict[str, object]] = []

    for weighting in args.weightings:
        print(f"[feature-subspace] {weighting}", flush=True)
        values = weighted_matrix(data, weighting)
        eigvals, eigvecs = cosine_eigendecomposition(values, args.max_rank)
        blocks, _diagnostics = adaptive_spectral_blocks(
            eigvals,
            min_segment_length=args.min_segment_length,
            max_segments=args.max_segments,
        )
        z = row_normalized_weighted_matrix(data, weighting)
        feature_axes = feature_axes_from_cosine_eigenvectors(z, eigvals, eigvecs)

        for block in blocks:
            print(f"[feature-subspace] {weighting} / {block.block_name}", flush=True)
            result = run_feature_block(
                data=data,
                weighting=weighting,
                block=block,
                eigvals=eigvals,
                feature_axes=feature_axes,
                output_dir=output_dir,
                max_active_features=args.max_active_features,
                tree_linkage_method=args.tree_linkage_method,
                edge_alpha=args.edge_alpha,
                sibling_alpha=args.sibling_alpha,
            )
            rows.append(asdict(result))
            pd.DataFrame.from_records(rows).to_csv(
                output_dir / "kak_feature_subspace_clustering_summary.csv",
                index=False,
            )

    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(output_dir / "kak_feature_subspace_clustering_summary.csv", index=False)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# KAK Feature Subspace Clustering",
                "",
                f"Schema: `{SCHEMA_VERSION}`",
                f"Input: `{args.input}`",
                f"Tree linkage: `{args.tree_linkage_method}`",
                f"Max active features: `{args.max_active_features}`",
                "",
                "Feature leaves are clustered in `Q_B sqrt(Lambda_B)` coordinates.",
                "This is diagnostic-only and does not provide selected-subspace calibration.",
                "",
                "## Status Counts",
                "",
                summary["status"].value_counts(dropna=False).to_string() if not summary.empty else "No rows.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote KAK feature subspace clustering: {output_dir}", flush=True)
    return summary


def main() -> None:
    run_feature_subspace_clustering(parse_args())


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA_VERSION",
    "FeatureSubspaceResult",
    "build_feature_tree",
    "feature_coordinates_for_block",
    "run_feature_block",
    "run_feature_subspace_clustering",
    "select_active_feature_coordinates",
]
