"""Map raw KAK/cosine lens axes back to features, then cluster genes.

For the raw cosine operator, let ``Z`` be the row-normalized weighted feature
matrix and ``K = Z Z.T``. If ``K u_j = lambda_j u_j``, the corresponding
feature-axis loading vector is

    q_j = Z.T u_j / sqrt(lambda_j)

and the gene coordinate is ``Z q_j = sqrt(lambda_j) u_j``. This diagnostic uses
that exact dual relation for raw KAK/cosine lenses, writes feature loadings for
the common axis and selected variant lens axes, then runs the normal Tree-Break Selection gate
pipeline on the lens tree.

Separated diffusion lenses are intentionally excluded: their coordinates are a
nonlinear overlay on top of KAK coordinates and need a different attribution
method.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
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
from tree_break_selection.hierarchy_analysis.tree_decomposition import TreeDecomposition
from tree_break_selection.space_separation import (
    adaptive_spectral_blocks,
    coordinates_for_block,
    cosine_eigendecomposition,
    weight_feature_matrix,
)

from benchmarks.diagnostics.spectral.adaptive_cosine.adaptive_cosine_kak_benchmark_probe import (
    sibling_method_counts,
)
from benchmarks.diagnostics.spectral.adaptive_cosine.adaptive_cosine_kak_matrix_probe import (
    load_matrix,
)
from benchmarks.diagnostics.spectral.adaptive_cosine.kak_lens_alpha_sweep import (
    LensSpec,
    build_lens_tree,
    find_block,
    parse_lens,
)

SCHEMA_VERSION = "kak_lens_feature_axis_clustering/v1"
DEFAULT_RAW_LENSES = (
    LensSpec("raw_kak", "binary", "adaptive_modes_10_15"),
    LensSpec("raw_kak", "tfidf", "adaptive_modes_14_30"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map raw KAK/cosine lens axes to feature loadings and cluster genes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/feature_matrices/feature_matrix_julia_GOCC_GOBP_GOMF_combined.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--lens",
        action="append",
        type=parse_lens,
        default=None,
        help="Lens spec family:weighting:block_name. Only raw_kak lenses are exact.",
    )
    parser.add_argument(
        "--tree-linkage-method", default="average", choices=["average", "complete", "ward"]
    )
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--min-segment-length", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--top-features", type=int, default=30)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    return parser.parse_args()


def default_output_dir(input_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return (
        Path("benchmarks/results/diagnostics")
        / f"kak_lens_feature_axis_clustering_{input_path.stem}_{stamp}"
    )


def row_normalized_weighted_matrix(data: pd.DataFrame, weighting: str) -> np.ndarray:
    values = weight_feature_matrix(data, weighting)
    row_norms = np.linalg.norm(values, axis=1)
    if np.any(row_norms <= 1e-12):
        raise ValueError("Rows with zero norm cannot enter the raw KAK/cosine operator.")
    return normalize(values, norm="l2", axis=1)


def feature_axes_from_cosine_eigenvectors(
    row_normalized_values: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
) -> np.ndarray:
    eigvals = np.asarray(eigvals, dtype=float)
    eigvecs = np.asarray(eigvecs, dtype=float)
    positive = eigvals > 1e-12
    if not np.all(positive):
        raise ValueError("Feature-axis recovery requires positive eigenvalues.")
    axes = row_normalized_values.T @ eigvecs
    axes = axes / np.sqrt(eigvals)[np.newaxis, :]
    norms = np.linalg.norm(axes, axis=0)
    if np.any(norms <= 1e-8):
        raise ValueError("Recovered feature axis has near-zero norm.")
    return axes / norms[np.newaxis, :]


def feature_axis_debug_metrics(
    row_normalized_values: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    feature_axes: np.ndarray,
) -> dict[str, float]:
    expected = eigvecs * np.sqrt(eigvals)[np.newaxis, :]
    reconstructed = row_normalized_values @ feature_axes
    gram = feature_axes.T @ feature_axes
    identity = np.eye(gram.shape[0])
    return {
        "feature_axis_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - expected))
        ),
        "feature_axis_reconstruction_rmse": float(
            np.sqrt(np.mean((reconstructed - expected) ** 2))
        ),
        "feature_axis_orthogonality_max_abs_error": float(np.max(np.abs(gram - identity))),
    }


def feature_axis_scores(
    *,
    features: pd.Index,
    eigvals: np.ndarray,
    feature_axes: np.ndarray,
    block_start: int,
    block_end: int,
) -> pd.DataFrame:
    common = feature_axes[:, 0]
    start = block_start - 1
    end = block_end
    block_axes = feature_axes[:, start:end]
    block_eigvals = eigvals[start:end]
    block_energy = float(np.sum(block_eigvals))
    if block_energy <= 0:
        raise ValueError("Selected lens block has no positive energy.")
    variant_energy = np.sum((block_axes * block_axes) * block_eigvals[np.newaxis, :], axis=1)
    variant_energy_fraction = variant_energy / block_energy
    connection = np.abs(common) * np.sqrt(np.maximum(variant_energy_fraction, 0.0))
    table = pd.DataFrame(
        {
            "feature": features.astype(str),
            "common_axis_loading": common,
            "abs_common_axis_loading": np.abs(common),
            "variant_loading_energy": variant_energy,
            "variant_loading_energy_fraction": variant_energy_fraction,
            "axis_connection_score": connection,
        }
    )
    table["common_axis_rank"] = (
        table["abs_common_axis_loading"].rank(method="first", ascending=False).astype(int)
    )
    table["variant_energy_rank"] = (
        table["variant_loading_energy_fraction"].rank(method="first", ascending=False).astype(int)
    )
    table["axis_connection_rank"] = (
        table["axis_connection_score"].rank(method="first", ascending=False).astype(int)
    )
    return table.sort_values("axis_connection_rank")


def component_feature_loadings(
    *,
    features: pd.Index,
    eigvals: np.ndarray,
    feature_axes: np.ndarray,
    block_start: int,
    block_end: int,
    top_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    common_axis = feature_axes[:, 0]
    for component in range(block_start, block_end + 1):
        axis = feature_axes[:, component - 1]
        dot_common = float(np.dot(axis, common_axis))
        order = np.argsort(np.abs(axis))[::-1][:top_features]
        for rank, feature_index in enumerate(order, start=1):
            rows.append(
                {
                    "component": int(component),
                    "eigenvalue": float(eigvals[component - 1]),
                    "feature": str(features[feature_index]),
                    "loading": float(axis[feature_index]),
                    "abs_loading": float(abs(axis[feature_index])),
                    "loading_rank": int(rank),
                    "component_feature_axis_dot_common_axis": dot_common,
                }
            )
    return pd.DataFrame.from_records(rows)


def cluster_axis_summary(
    *,
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    coordinates: np.ndarray,
    common_axis_score: np.ndarray,
    feature_scores: pd.DataFrame,
    top_features: int,
    min_cluster_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = assignments["cluster_id"].astype(int).to_numpy()
    variant_radius = np.linalg.norm(coordinates, axis=1)
    summary_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    score_lookup = feature_scores.set_index("feature")
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        size = int(mask.sum())
        if size < min_cluster_size:
            continue
        rest = ~mask
        cluster_prevalence = data.loc[mask].mean(axis=0)
        rest_prevalence = (
            data.loc[rest].mean(axis=0) if int(rest.sum()) else cluster_prevalence * 0.0
        )
        delta = (cluster_prevalence - rest_prevalence).sort_values(ascending=False)
        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": size,
                "mean_common_axis_score": float(np.mean(common_axis_score[mask])),
                "mean_variant_radius": float(np.mean(variant_radius[mask])),
                "median_variant_radius": float(np.median(variant_radius[mask])),
                "top_prevalence_delta": float(delta.iloc[0]),
                "top_prevalence_delta_feature": str(delta.index[0]),
            }
        )
        for rank, (feature, value) in enumerate(delta.head(top_features).items(), start=1):
            axis_row = (
                score_lookup.loc[str(feature)] if str(feature) in score_lookup.index else None
            )
            top_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": size,
                    "feature": str(feature),
                    "prevalence_delta_rank": int(rank),
                    "cluster_prevalence": float(cluster_prevalence[feature]),
                    "rest_prevalence": float(rest_prevalence[feature]),
                    "prevalence_delta": float(value),
                    "common_axis_loading": (
                        float(axis_row["common_axis_loading"]) if axis_row is not None else math.nan
                    ),
                    "variant_loading_energy_fraction": (
                        float(axis_row["variant_loading_energy_fraction"])
                        if axis_row is not None
                        else math.nan
                    ),
                    "axis_connection_score": (
                        float(axis_row["axis_connection_score"])
                        if axis_row is not None
                        else math.nan
                    ),
                    "axis_connection_rank": (
                        int(axis_row["axis_connection_rank"]) if axis_row is not None else pd.NA
                    ),
                }
            )
    return pd.DataFrame.from_records(summary_rows), pd.DataFrame.from_records(top_rows)


def write_lens_plots(
    *,
    output_dir: Path,
    lens_id: str,
    assignments: pd.DataFrame,
    coordinates: np.ndarray,
    common_axis_score: np.ndarray,
    feature_scores: pd.DataFrame,
    top_features: int,
) -> None:
    safe_lens_id = lens_id.replace(":", "_").replace("/", "_")
    labels = assignments["cluster_id"].astype(int).to_numpy()
    variant_radius = np.linalg.norm(coordinates, axis=1)
    sizes = pd.Series(labels).value_counts()
    large_clusters = set(sizes.head(20).index.tolist())
    color_values = np.array([label if label in large_clusters else -1 for label in labels])

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    scatter = ax.scatter(
        common_axis_score,
        variant_radius,
        c=color_values,
        s=22,
        cmap="tab20",
        alpha=0.78,
        linewidths=0.0,
    )
    ax.set_xlabel("Common cosine-axis score")
    ax.set_ylabel("Selected lens variant radius")
    ax.set_title(f"{lens_id}: common axis vs variant lens radius")
    fig.colorbar(scatter, ax=ax, label="cluster id; -1 = smaller clusters")
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_lens_id}_axis_cluster_scatter.png", dpi=180)
    plt.close(fig)

    top = feature_scores.sort_values("axis_connection_rank").head(top_features).iloc[::-1]
    fig_height = max(5.0, 0.28 * len(top) + 1.2)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    ax.barh(top["feature"], top["axis_connection_score"], color="#4c78a8", alpha=0.86)
    ax.set_xlabel("|common loading| * sqrt(variant energy fraction)")
    ax.set_title(f"{lens_id}: top feature connections between common and variant axes")
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_lens_id}_feature_connection_top.png", dpi=180)
    plt.close(fig)


def run_lens(
    *,
    data: pd.DataFrame,
    spec: LensSpec,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    blocks: list[object],
    row_normalized_values: np.ndarray,
    feature_axes: np.ndarray,
    axis_debug_metrics: dict[str, float],
    output_dir: Path,
    tree_linkage_method: str,
    edge_alpha: float,
    sibling_alpha: float,
    top_features: int,
    min_cluster_size: int,
) -> dict[str, object]:
    if spec.family != "raw_kak":
        raise ValueError(f"{spec.lens_id} is not exact-feature-mappable; use raw_kak lenses only.")
    block = find_block(blocks, spec.block_name)
    coordinates = coordinates_for_block(eigvals, eigvecs, block)
    common_axis_score = eigvecs[:, 0] * math.sqrt(float(eigvals[0]))
    tree, tree_metadata = build_lens_tree(
        data=data,
        lens_coordinates=coordinates,
        spec=spec,
        tree_linkage_method=tree_linkage_method,
        metadata={"tree_metric": "raw_kak_block_euclidean"},
    )
    gate_bundle = run_gate_annotation_pipeline(
        tree,
        tree.annotations_df.copy(),
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        leaf_data=data,
    )
    decomposition = TreeDecomposition(
        tree=tree,
        gate_annotation_bundle=gate_bundle,
    ).decompose_tree()
    assignments = build_sample_cluster_assignments(decomposition).loc[data.index]
    reconstructed_block = (
        row_normalized_values @ feature_axes[:, block.block_start - 1 : block.block_end]
    )
    block_reconstruction_max_abs_error = float(np.max(np.abs(reconstructed_block - coordinates)))

    feature_scores = feature_axis_scores(
        features=data.columns,
        eigvals=eigvals,
        feature_axes=feature_axes,
        block_start=block.block_start,
        block_end=block.block_end,
    )
    component_loadings = component_feature_loadings(
        features=data.columns,
        eigvals=eigvals,
        feature_axes=feature_axes,
        block_start=block.block_start,
        block_end=block.block_end,
        top_features=top_features,
    )
    cluster_summary, cluster_top_features = cluster_axis_summary(
        data=data,
        assignments=assignments,
        coordinates=coordinates,
        common_axis_score=common_axis_score,
        feature_scores=feature_scores,
        top_features=top_features,
        min_cluster_size=min_cluster_size,
    )

    lens_dir = output_dir / spec.lens_id
    lens_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(lens_dir / "cluster_assignments.csv")
    pd.DataFrame(
        {
            "gene": data.index.astype(str),
            "cluster_id": assignments["cluster_id"].astype(int).to_numpy(),
            "common_axis_score": common_axis_score,
            "variant_radius": np.linalg.norm(coordinates, axis=1),
            **{f"lens_coord_{i + 1}": coordinates[:, i] for i in range(coordinates.shape[1])},
        }
    ).to_csv(lens_dir / "gene_lens_coordinates.csv", index=False)
    feature_scores.to_csv(lens_dir / "lens_feature_axis_scores.csv", index=False)
    component_loadings.to_csv(lens_dir / "component_feature_loadings_top.csv", index=False)
    cluster_summary.to_csv(lens_dir / "cluster_axis_summary.csv", index=False)
    cluster_top_features.to_csv(lens_dir / "cluster_top_features_with_axis_scores.csv", index=False)
    write_lens_plots(
        output_dir=lens_dir,
        lens_id=spec.lens_id,
        assignments=assignments,
        coordinates=coordinates,
        common_axis_score=common_axis_score,
        feature_scores=feature_scores,
        top_features=top_features,
    )

    labels = assignments["cluster_id"].astype(int)
    sizes = labels.value_counts()
    return {
        "schema_version": SCHEMA_VERSION,
        "lens_id": spec.lens_id,
        "weighting": spec.weighting,
        "block_name": spec.block_name,
        "tree_linkage_method": tree_linkage_method,
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "block_start": int(block.block_start),
        "block_end": int(block.block_end),
        "subspace_dimensions": int(block.block_end - block.block_start + 1),
        "n_clusters": int(labels.nunique()),
        "singleton_gene_fraction": float((sizes == 1).sum() / len(labels)),
        "largest_cluster_size": int(sizes.max()),
        "largest_cluster_fraction": float(sizes.max() / len(labels)),
        "top_axis_connection_feature": str(feature_scores.iloc[0]["feature"]),
        "top_axis_connection_score": float(feature_scores.iloc[0]["axis_connection_score"]),
        "block_coordinate_reconstruction_max_abs_error": (block_reconstruction_max_abs_error),
        **axis_debug_metrics,
        "sibling_test_method_counts": sibling_method_counts(tree.annotations_df),
        "tree_metadata": repr(tree_metadata),
        "lens_output_dir": str(lens_dir),
    }


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# KAK Lens Feature-Axis Clustering",
        "",
        "This diagnostic maps raw KAK/cosine lens axes back to original feature loadings using the exact dual SVD relation.",
        "It then clusters genes through the normal Tree-Break Selection gate path on the selected lens tree.",
        "",
        "Separated diffusion lenses are excluded because their feature attribution is nonlinear and not exact under this map.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "kak_lens_feature_axis_clustering_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_matrix(args.input)
    lens_specs = tuple(args.lens) if args.lens else DEFAULT_RAW_LENSES

    eigensystems: dict[
        str,
        tuple[np.ndarray, np.ndarray, list[object], np.ndarray, np.ndarray, dict[str, float]],
    ] = {}
    rows: list[dict[str, object]] = []
    for spec in lens_specs:
        if spec.family != "raw_kak":
            raise ValueError(f"Only raw_kak lenses have an exact feature-axis map: {spec.lens_id}")
        if spec.weighting not in eigensystems:
            values = weight_feature_matrix(data, spec.weighting)
            eigvals, eigvecs = cosine_eigendecomposition(values, args.max_rank)
            blocks, _ = adaptive_spectral_blocks(
                eigvals,
                min_segment_length=args.min_segment_length,
                max_segments=args.max_segments,
            )
            z = row_normalized_weighted_matrix(data, spec.weighting)
            feature_axes = feature_axes_from_cosine_eigenvectors(z, eigvals, eigvecs)
            axis_debug_metrics = feature_axis_debug_metrics(
                z,
                eigvals,
                eigvecs,
                feature_axes,
            )
            eigensystems[spec.weighting] = (
                eigvals,
                eigvecs,
                blocks,
                z,
                feature_axes,
                axis_debug_metrics,
            )
        eigvals, eigvecs, blocks, z, feature_axes, axis_debug_metrics = eigensystems[spec.weighting]
        rows.append(
            run_lens(
                data=data,
                spec=spec,
                eigvals=eigvals,
                eigvecs=eigvecs,
                blocks=blocks,
                row_normalized_values=z,
                feature_axes=feature_axes,
                axis_debug_metrics=axis_debug_metrics,
                output_dir=output_dir,
                tree_linkage_method=args.tree_linkage_method,
                edge_alpha=args.edge_alpha,
                sibling_alpha=args.sibling_alpha,
                top_features=args.top_features,
                min_cluster_size=args.min_cluster_size,
            )
        )

    summary = pd.DataFrame.from_records(rows)
    summary.to_csv(output_dir / "kak_lens_feature_axis_clustering_summary.csv", index=False)
    write_report(summary, output_dir)
    print(f"Wrote KAK lens feature-axis clustering: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
