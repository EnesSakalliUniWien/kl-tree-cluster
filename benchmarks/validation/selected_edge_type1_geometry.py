#!/usr/bin/env python3
"""Selected-edge Type-I and geometry diagnostic.

This module generates validation evidence only. It does not change production
alpha defaults and does not add fallback calibration.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation import (
    annotate_child_parent_divergence_with_context,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.contrast_covariance import (
    build_contrast_covariance,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.types.sibling_pair_record import (
    SiblingPairRecord,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from kl_clustering_analysis.tree.feature_space import (
    FeatureSpace,
    bernoulli_feature_space_from_columns,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.diagnostics.calibration.covariance_laplacian_panel import (
    analyze_covariance_laplacian,
)
from benchmarks.diagnostics.calibration.statistic_distribution_shape_panel import (
    infer_satterthwaite_reference_from_eigenvalues,
)
from benchmarks.shared.cases import get_default_test_cases, get_test_cases_by_suite
from benchmarks.shared.generators.case_data_contracts import one_hot_encode_categorical

SCHEMA_VERSION = "selected_edge_type1_geometry/v1"
GENERATED_BY = "benchmarks.validation.selected_edge_type1_geometry"
DEFAULT_EDGE_ALPHA_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01)
DEFAULT_MODES = ("fixed_tree", "selected_tree", "selected_tree_traversal")
EDGE_COLUMNS = (
    "Child_Parent_Divergence_Test_Statistic",
    "Child_Parent_Divergence_df",
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_P_Value_BH",
    "Child_Parent_Divergence_Tested",
    "Child_Parent_Divergence_Significant",
    "leaf_count",
)
SIBLING_COLUMNS = (
    "Sibling_Test_Statistic",
    "Sibling_Degrees_of_Freedom",
    "Sibling_Divergence_P_Value",
    "Sibling_Divergence_P_Value_Corrected",
    "Sibling_BH_Different",
    "Sibling_Projection_Dimension",
)
COVARIANCE_LAPLACIAN_MAX_FEATURES = 512


@dataclass(frozen=True)
class SelectedEdgeGeometryConfig:
    """Runtime contract for selected-edge geometry evidence."""

    output_dir: Path
    suite: str
    case_names: tuple[str, ...]
    modes: tuple[str, ...]
    edge_alphas: tuple[float, ...]
    sibling_alpha: float
    replicates: int
    base_seed: int

    @property
    def edge_rows_path(self) -> Path:
        return self.output_dir / "selected_edge_geometry_edges.csv"

    @property
    def sibling_rows_path(self) -> Path:
        return self.output_dir / "selected_edge_geometry_siblings.csv"

    @property
    def final_rows_path(self) -> Path:
        return self.output_dir / "selected_edge_geometry_final.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "selected_edge_geometry_manifest.json"


def parse_alpha_grid(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated alpha grid."""
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("Alpha grid must contain at least one value.")
    invalid = [value for value in values if not 0.0 < value < 1.0]
    if invalid:
        raise ValueError(f"Alpha values must lie in (0, 1): {invalid!r}")
    return values


def parse_names(raw: str | None) -> tuple[str, ...]:
    """Parse optional comma-separated names."""
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def validate_modes(modes: Sequence[str]) -> tuple[str, ...]:
    """Validate selected-edge simulation modes."""
    allowed = set(DEFAULT_MODES)
    result = tuple(str(mode) for mode in modes)
    invalid = sorted(set(result) - allowed)
    if invalid:
        raise ValueError(f"Unknown selected-edge mode(s): {invalid!r}; allowed={sorted(allowed)!r}")
    if not result:
        raise ValueError("At least one selected-edge mode is required.")
    return result


def build_run_id(config: SelectedEdgeGeometryConfig) -> str:
    """Return a timestamped diagnostic run id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_id = "-".join(config.modes)
    return f"selected_edge_type1_geometry__{config.suite}__{mode_id}__{stamp}"


def regenerate_null_case(
    *,
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None = None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate a global-null matrix for one selected-edge replicate."""
    rng = np.random.default_rng(int(seed))
    if source_family == "binary_template" and feature_representation == "binary":
        matrix = rng.binomial(1, 0.5, size=(int(n_samples), int(n_features))).astype(int)
        data = pd.DataFrame(
            matrix,
            index=[f"S{i}" for i in range(int(n_samples))],
            columns=[f"F{j}" for j in range(int(n_features))],
        )
        feature_space = bernoulli_feature_space_from_columns(tuple(data.columns))
        n_features_original = int(n_features)
        categories = None
    elif source_family == "categorical_multinomial" and feature_representation == "categorical_one_hot":
        if n_categories is None:
            raise ValueError(
                "Categorical selected-edge null regeneration requires n_categories."
            )
        if int(n_categories) < 2:
            raise ValueError(f"n_categories must be at least 2; got {n_categories!r}.")
        matrix = rng.integers(
            low=0,
            high=int(n_categories),
            size=(int(n_samples), int(n_features)),
            endpoint=False,
        )
        data, _raw_dimension, feature_space = one_hot_encode_categorical(
            matrix,
            int(n_categories),
            [f"S{i}" for i in range(int(n_samples))],
        )
        n_features_original = int(n_features)
        categories = int(n_categories)
    else:
        raise ValueError(
            "Selected-edge Type-I diagnostic supports only binary_template/binary "
            "and categorical_multinomial/categorical_one_hot null regeneration; "
            f"got {source_family!r}/{feature_representation!r} for {case_id!r}."
        )
    metadata = {
        "case_id": str(case_id),
        "source_family": str(source_family),
        "feature_representation": str(feature_representation),
        "n_samples": int(n_samples),
        "n_features": int(data.shape[1]),
        "n_features_original": n_features_original,
        "n_categories": categories,
        "true_null": True,
        "seed": int(seed),
        "feature_space": feature_space,
    }
    return data, metadata


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required annotation columns: {missing!r}.")


def _node_depth_map(tree: nx.DiGraph) -> dict[object, int]:
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root; got {roots!r}.")
    return dict(nx.single_source_shortest_path_length(tree, roots[0]))


def _finite_or_nan(value: object) -> float:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("nan")


def _positive_p_or_nan(value: object) -> float:
    numeric = _finite_or_nan(value)
    if not np.isfinite(numeric):
        return float("nan")
    if numeric <= 0.0:
        return float("nan")
    return numeric


def negative_log10_action(p_value: float) -> float:
    """Return the edge-selection action coordinate."""
    p_value = float(p_value)
    if not np.isfinite(p_value) or not 0.0 < p_value <= 1.0:
        return float("nan")
    return float(-np.log10(p_value))


def edge_statistic_margin(statistic: float, degrees_of_freedom: float, edge_alpha: float) -> float:
    """Return sqrt-statistic margin over the chi-square alpha boundary."""
    statistic = float(statistic)
    degrees_of_freedom = float(degrees_of_freedom)
    if not (np.isfinite(statistic) and np.isfinite(degrees_of_freedom)):
        return float("nan")
    if statistic < 0.0 or degrees_of_freedom <= 0.0:
        return float("nan")
    return float(np.sqrt(statistic) - np.sqrt(chi2.isf(float(edge_alpha), degrees_of_freedom)))


def _spectral_entropy(eigenvalues: np.ndarray) -> float:
    finite = np.asarray(eigenvalues, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        return float("nan")
    weights = finite / np.sum(finite)
    return float(-np.sum(weights * np.log(weights)))


def _effective_rank(eigenvalues: np.ndarray) -> float:
    entropy = _spectral_entropy(eigenvalues)
    if not np.isfinite(entropy):
        return float("nan")
    return float(np.exp(entropy))


def _tree_balance(tree: nx.DiGraph, parent: object) -> float:
    children = list(tree.successors(parent))
    if len(children) != 2:
        return float("nan")
    counts = [int(tree.nodes[child]["leaf_count"]) for child in children]
    total = sum(counts)
    if total <= 0:
        return float("nan")
    return float(min(counts) / total)


def _build_tree_from_data(data: pd.DataFrame) -> PosetTree:
    distances = pdist(data.to_numpy(dtype=float), metric="hamming")
    linkage_matrix = linkage(distances, method="average")
    return PosetTree.from_linkage(linkage_matrix, leaf_names=data.index.tolist())


def _prepare_tree_for_mode(
    *,
    mode: str,
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None,
    tree_seed: int,
    data_seed: int,
) -> tuple[PosetTree, pd.DataFrame, FeatureSpace, bool, bool]:
    if mode in {"selected_tree", "selected_tree_traversal"}:
        data, metadata = regenerate_null_case(
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            seed=data_seed,
        )
        return _build_tree_from_data(data), data, metadata["feature_space"], True, False
    if mode == "fixed_tree":
        tree_data, _tree_metadata = regenerate_null_case(
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            seed=tree_seed,
        )
        test_data, test_metadata = regenerate_null_case(
            case_id=case_id,
            source_family=source_family,
            feature_representation=feature_representation,
            n_samples=n_samples,
            n_features=n_features,
            n_categories=n_categories,
            seed=data_seed,
        )
        return _build_tree_from_data(tree_data), test_data, test_metadata["feature_space"], False, True
    raise ValueError(f"Unknown selected-edge mode: {mode!r}.")


def _annotate_edges(
    tree: PosetTree,
    data: pd.DataFrame,
    *,
    edge_alpha: float,
    feature_space: FeatureSpace,
) -> tuple[pd.DataFrame, object]:
    tree.populate_node_divergences(data, feature_space=feature_space)
    if tree.annotations_df is None:
        raise ValueError("Tree distribution population did not produce annotations_df.")
    annotations, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        tree.annotations_df,
        significance_level_alpha=float(edge_alpha),
        leaf_data=data,
        feature_space=feature_space,
    )
    tree.annotations_df = annotations
    return annotations, spectral_context


def extract_edge_geometry_rows(
    *,
    tree: PosetTree,
    annotations_df: pd.DataFrame,
    spectral_context: object,
    run_id: str,
    mode: str,
    source_family: str,
    feature_representation: str,
    case_id: str,
    replicate: int,
    tree_seed: int,
    data_seed: int,
    edge_alpha: float,
    selected_tree: bool,
    fixed_tree: bool,
) -> list[dict[str, object]]:
    """Extract strict edge-level diagnostic rows."""
    _require_columns(annotations_df, EDGE_COLUMNS, context="selected-edge annotations")
    depths = _node_depth_map(tree)
    rows: list[dict[str, object]] = []
    for parent, child in tree.edges():
        if child not in annotations_df.index or parent not in annotations_df.index:
            raise ValueError(f"Annotations must contain edge endpoints {parent!r}->{child!r}.")
        stat = _finite_or_nan(annotations_df.at[child, "Child_Parent_Divergence_Test_Statistic"])
        df_value = _finite_or_nan(annotations_df.at[child, "Child_Parent_Divergence_df"])
        raw_p = _positive_p_or_nan(annotations_df.at[child, "Child_Parent_Divergence_P_Value"])
        bh_p = _positive_p_or_nan(annotations_df.at[child, "Child_Parent_Divergence_P_Value_BH"])
        n_child = int(annotations_df.at[child, "leaf_count"])
        n_parent = int(annotations_df.at[parent, "leaf_count"])
        eigenvalues = np.asarray(
            spectral_context.principal_component_eigenvalues_by_node.get(str(parent), ()),
            dtype=float,
        )
        leading = float(eigenvalues[0]) if eigenvalues.size else float("nan")
        raw_mp_signal_count = int(
            spectral_context.raw_mp_signal_counts_by_node.get(str(parent), 0)
        )
        effective_rows = int(
            spectral_context.effective_independent_rows_by_node.get(str(parent), 0)
        )
        projection_dimension = int(
            spectral_context.test_projection_dimensions_by_node.get(str(parent), 0)
        )
        active_dimension = int(
            np.asarray(tree.nodes[parent]["distribution"], dtype=float).shape[0]
        )
        mp_upper_edge = (
            float((1.0 + np.sqrt(active_dimension / effective_rows)) ** 2)
            if effective_rows > 0
            else float("nan")
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "mode": mode,
                "source_family": source_family,
                "feature_representation": feature_representation,
                "case_id": case_id,
                "replicate": int(replicate),
                "tree_seed": int(tree_seed),
                "data_seed": int(data_seed),
                "node_id": str(child),
                "parent_id": str(parent),
                "child_id": str(child),
                "node_depth": int(depths[child]),
                "n_parent": n_parent,
                "n_child": n_child,
                "sample_ratio": float(n_child / n_parent),
                "edge_alpha": float(edge_alpha),
                "edge_raw_stat": stat,
                "edge_df": df_value,
                "edge_raw_p": raw_p,
                "edge_bh_p": bh_p,
                "edge_tested": bool(annotations_df.at[child, "Child_Parent_Divergence_Tested"]),
                "edge_rejected": bool(
                    annotations_df.at[child, "Child_Parent_Divergence_Significant"]
                ),
                "ancestor_reached": not bool(
                    annotations_df.at[child, "Child_Parent_Divergence_Ancestor_Blocked"]
                ),
                "selected_tree": bool(selected_tree),
                "fixed_tree": bool(fixed_tree),
                "projection_dimension": projection_dimension,
                "mp_upper_edge": mp_upper_edge,
                "raw_mp_signal_count": raw_mp_signal_count,
                "effective_independent_rows": effective_rows,
                "leading_eigenvalue": leading,
                "selected_eigenvalue_over_mp": (
                    float(leading / mp_upper_edge)
                    if np.isfinite(leading) and np.isfinite(mp_upper_edge) and mp_upper_edge > 0.0
                    else float("nan")
                ),
                "effective_rank": _effective_rank(eigenvalues),
                "spectral_entropy": _spectral_entropy(eigenvalues),
                "edge_z_norm": float("nan"),
                "edge_projected_norm": float(np.sqrt(stat)) if np.isfinite(stat) else float("nan"),
                "edge_projection_energy_ratio": float("nan"),
                "edge_statistic_margin": edge_statistic_margin(stat, df_value, edge_alpha),
                "edge_bh_action": negative_log10_action(bh_p),
                "merge_margin": float("nan"),
                "merge_margin_rank": float("nan"),
                "merge_persistence": float("nan"),
                "first_order_signed_distance": float("nan"),
                "null_whitened_signed_distance": float("nan"),
                "tie_cell_status": "not_materialized_in_selected_edge_type1_runner",
                "tree_balance": _tree_balance(tree, parent),
                "subtree_leaf_count": n_child,
                "descendant_leaf_count": n_child,
                "path_length_from_root": int(depths[child]),
            }
        )
    return rows


def _calibration_support_status(row: pd.Series) -> str:
    if bool(row.get("Sibling_Divergence_Skipped", False)):
        return "not_tested"
    corrected = row.get("Sibling_Divergence_P_Value_Corrected", np.nan)
    raw = row.get("Sibling_Divergence_P_Value", np.nan)
    if np.isfinite(float(corrected)):
        return "fdr_corrected"
    if np.isfinite(float(raw)):
        return "raw_only_not_reachable"
    return "not_tested"


def _projection_inputs(tree: PosetTree, spectral_context: object) -> tuple[
    dict[object, int],
    dict[object, np.ndarray],
    dict[object, np.ndarray],
]:
    projection_dimensions = derive_sibling_projection_dimensions_from_child_edge_comparisons(
        tree,
        spectral_context=spectral_context,
    )
    parent_projections, parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            projection_dimensions,
            spectral_context=spectral_context,
        )
    )
    return projection_dimensions, parent_projections, parent_eigenvalues


def _collect_raw_sibling_records(
    *,
    tree: PosetTree,
    annotations_df: pd.DataFrame,
    spectral_context: object,
    feature_space: FeatureSpace,
) -> dict[str, SiblingPairRecord]:
    projection_dimensions, parent_projections, parent_eigenvalues = _projection_inputs(
        tree,
        spectral_context,
    )
    records, _non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        sibling_projection_dimensions_from_edge_comparisons=projection_dimensions,
        parent_principal_component_projections=parent_projections,
        parent_principal_component_eigenvalues=parent_eigenvalues,
        feature_space=feature_space,
    )
    return {str(record.parent): record for record in records}


def _covariance_laplacian_empty_fields(prefix: str, status: str) -> dict[str, object]:
    return {
        f"{prefix}_laplacian_status": status,
        f"{prefix}_covariance_effective_rank": float("nan"),
        f"{prefix}_off_diagonal_abs_mass_fraction": float("nan"),
        f"{prefix}_n_connected_components": float("nan"),
        f"{prefix}_largest_component_fraction": float("nan"),
        f"{prefix}_normalized_laplacian_lambda2": float("nan"),
    }


def _select_laplacian_fields(prefix: str, row: Mapping[str, object]) -> dict[str, object]:
    return {
        f"{prefix}_laplacian_status": row["laplacian_status"],
        f"{prefix}_covariance_effective_rank": row["covariance_effective_rank"],
        f"{prefix}_off_diagonal_abs_mass_fraction": row[
            "off_diagonal_abs_mass_fraction"
        ],
        f"{prefix}_n_connected_components": row["n_connected_components"],
        f"{prefix}_largest_component_fraction": row["largest_component_fraction"],
        f"{prefix}_normalized_laplacian_lambda2": row[
            "normalized_laplacian_lambda2"
        ],
    }


def _block_diagonal_covariance(
    covariance_blocks: Sequence[np.ndarray],
    *,
    max_features: int = COVARIANCE_LAPLACIAN_MAX_FEATURES,
) -> np.ndarray | None:
    width = int(sum(np.asarray(block).shape[0] for block in covariance_blocks))
    if width <= 0 or width > int(max_features):
        return None
    matrix = np.zeros((width, width), dtype=float)
    offset = 0
    for block in covariance_blocks:
        covariance_block = np.asarray(block, dtype=float)
        next_offset = offset + int(covariance_block.shape[0])
        matrix[offset:next_offset, offset:next_offset] = covariance_block
        offset = next_offset
    return matrix


def _sibling_contrast_laplacian_fields(
    *,
    tree: PosetTree,
    parent: object,
    left: object,
    right: object,
    feature_space: FeatureSpace,
) -> dict[str, object]:
    try:
        contrast_covariance = build_contrast_covariance(
            np.asarray(tree.nodes[left]["distribution"], dtype=float),
            np.asarray(tree.nodes[right]["distribution"], dtype=float),
            float(tree.nodes[left]["leaf_count"]),
            float(tree.nodes[right]["leaf_count"]),
            comparison="sibling",
            feature_space=feature_space,
        )
        matrix = _block_diagonal_covariance(contrast_covariance.covariance_blocks)
        if matrix is None:
            return _covariance_laplacian_empty_fields(
                "sibling_contrast",
                "laplacian_skipped_high_dimension",
            )
        row = analyze_covariance_laplacian(
            matrix,
            matrix_id=f"sibling_contrast:{parent}",
            matrix_context_role="sibling_contrast_covariance",
            feature_family=feature_space.family_label,
            correlation_threshold=0.05,
        )
        return _select_laplacian_fields("sibling_contrast", row)
    except (ValueError, np.linalg.LinAlgError):
        return _covariance_laplacian_empty_fields(
            "sibling_contrast",
            "laplacian_unavailable",
        )


def _parent_spectral_laplacian_fields(
    *,
    spectral_context: object,
    parent: object,
) -> dict[str, object]:
    key = str(parent)
    projections = getattr(spectral_context, "principal_component_projections_by_node", {})
    eigenvalues_by_node = getattr(
        spectral_context,
        "principal_component_eigenvalues_by_node",
        {},
    )
    if key not in projections or key not in eigenvalues_by_node:
        return _covariance_laplacian_empty_fields(
            "parent_spectral",
            "laplacian_missing_spectral_context",
        )
    projection = np.asarray(projections[key], dtype=float)
    eigenvalues = np.asarray(eigenvalues_by_node[key], dtype=float)
    if projection.ndim != 2 or eigenvalues.ndim != 1 or projection.shape[0] != eigenvalues.shape[0]:
        return _covariance_laplacian_empty_fields(
            "parent_spectral",
            "laplacian_invalid_spectral_context",
        )
    if projection.shape[1] > COVARIANCE_LAPLACIAN_MAX_FEATURES:
        return _covariance_laplacian_empty_fields(
            "parent_spectral",
            "laplacian_skipped_high_dimension",
        )
    try:
        covariance = (projection.T * eigenvalues) @ projection
        row = analyze_covariance_laplacian(
            covariance,
            matrix_id=f"parent_spectral:{parent}",
            matrix_context_role="parent_spectral_covariance",
            correlation_threshold=0.05,
        )
        return _select_laplacian_fields("parent_spectral", row)
    except (ValueError, np.linalg.LinAlgError):
        return _covariance_laplacian_empty_fields(
            "parent_spectral",
            "laplacian_unavailable",
        )


def _covariance_inferred_reference_fields(
    *,
    spectral_context: object,
    parent: object,
    projection_dimension: float,
) -> dict[str, float]:
    key = str(parent)
    eigenvalues_by_node = getattr(
        spectral_context,
        "principal_component_eigenvalues_by_node",
        {},
    )
    if key not in eigenvalues_by_node:
        return {
            "covariance_inferred_df": float("nan"),
            "covariance_inferred_reference_scale": float("nan"),
            "covariance_inferred_df_ratio_to_current": float("nan"),
        }
    eigenvalues = np.asarray(eigenvalues_by_node[key], dtype=float)
    k = int(projection_dimension) if np.isfinite(projection_dimension) else 0
    if k <= 0 or eigenvalues.size < k:
        return {
            "covariance_inferred_df": float("nan"),
            "covariance_inferred_reference_scale": float("nan"),
            "covariance_inferred_df_ratio_to_current": float("nan"),
        }
    try:
        scale, inferred_df = infer_satterthwaite_reference_from_eigenvalues(
            eigenvalues[:k],
        )
    except ValueError:
        scale = float("nan")
        inferred_df = float("nan")
    return {
        "covariance_inferred_df": inferred_df,
        "covariance_inferred_reference_scale": scale,
        "covariance_inferred_df_ratio_to_current": (
            float(inferred_df / projection_dimension)
            if np.isfinite(inferred_df) and projection_dimension > 0.0
            else float("nan")
        ),
    }


def extract_sibling_geometry_rows(
    *,
    tree: PosetTree,
    annotations_df: pd.DataFrame,
    spectral_context: object,
    feature_space: FeatureSpace,
    raw_sibling_records_by_parent: Mapping[str, SiblingPairRecord],
    run_id: str,
    mode: str,
    source_family: str,
    feature_representation: str,
    case_id: str,
    replicate: int,
    tree_seed: int,
    data_seed: int,
    edge_alpha: float,
    sibling_alpha: float,
) -> list[dict[str, object]]:
    """Extract sibling-level rows from available annotation columns."""
    has_sibling_annotations = set(SIBLING_COLUMNS).issubset(annotations_df.columns)
    if not has_sibling_annotations and not raw_sibling_records_by_parent:
        return []
    depths = _node_depth_map(tree)
    rows: list[dict[str, object]] = []
    for parent in tree.nodes:
        children = list(tree.successors(parent))
        if len(children) != 2:
            continue
        left, right = children
        n_left = int(tree.nodes[left]["leaf_count"])
        n_right = int(tree.nodes[right]["leaf_count"])
        n_parent = int(tree.nodes[parent]["leaf_count"])
        left_edge_raw = _positive_p_or_nan(
            annotations_df.at[left, "Child_Parent_Divergence_P_Value"]
        )
        right_edge_raw = _positive_p_or_nan(
            annotations_df.at[right, "Child_Parent_Divergence_P_Value"]
        )
        left_edge_bh = _positive_p_or_nan(
            annotations_df.at[left, "Child_Parent_Divergence_P_Value_BH"]
        )
        right_edge_bh = _positive_p_or_nan(
            annotations_df.at[right, "Child_Parent_Divergence_P_Value_BH"]
        )
        raw_record = raw_sibling_records_by_parent.get(str(parent))
        raw_sibling_stat = (
            float(raw_record.stat)
            if raw_record is not None and np.isfinite(raw_record.stat)
            else float("nan")
        )
        sibling_df = (
            float(raw_record.degrees_of_freedom)
            if raw_record is not None and np.isfinite(raw_record.degrees_of_freedom)
            else (
                _finite_or_nan(annotations_df.at[parent, "Sibling_Degrees_of_Freedom"])
                if has_sibling_annotations
                else float("nan")
            )
        )
        raw_sibling_p = (
            float(raw_record.p_value)
            if raw_record is not None and np.isfinite(raw_record.p_value)
            else (
                float(chi2.sf(raw_sibling_stat, df=sibling_df))
                if np.isfinite(raw_sibling_stat) and sibling_df > 0.0
                else float("nan")
            )
        )
        adjusted_stat = (
            _finite_or_nan(annotations_df.at[parent, "Sibling_Test_Statistic"])
            if has_sibling_annotations
            else float("nan")
        )
        adjusted_p = (
            _positive_p_or_nan(annotations_df.at[parent, "Sibling_Divergence_P_Value"])
            if has_sibling_annotations
            else float("nan")
        )
        sibling_bh_p = (
            _positive_p_or_nan(
                annotations_df.at[parent, "Sibling_Divergence_P_Value_Corrected"]
            )
            if has_sibling_annotations
            else float("nan")
        )
        sibling_projection_dimension = (
            float(raw_record.sibling_projection_dimension)
            if raw_record is not None
            and np.isfinite(raw_record.sibling_projection_dimension)
            else (
                _finite_or_nan(annotations_df.at[parent, "Sibling_Projection_Dimension"])
                if has_sibling_annotations
                else float("nan")
            )
        )
        selected_ratio = (
            float(raw_sibling_stat / sibling_df)
            if np.isfinite(raw_sibling_stat) and np.isfinite(sibling_df) and sibling_df > 0.0
            else float("nan")
        )
        adjusted_selected_ratio = (
            float(adjusted_stat / sibling_df)
            if np.isfinite(adjusted_stat) and np.isfinite(sibling_df) and sibling_df > 0.0
            else float("nan")
        )
        covariance_fields = _covariance_inferred_reference_fields(
            spectral_context=spectral_context,
            parent=parent,
            projection_dimension=sibling_projection_dimension,
        )
        sibling_laplacian_fields = _sibling_contrast_laplacian_fields(
            tree=tree,
            parent=parent,
            left=left,
            right=right,
            feature_space=feature_space,
        )
        parent_laplacian_fields = _parent_spectral_laplacian_fields(
            spectral_context=spectral_context,
            parent=parent,
        )
        inflation_factor = (
            float(raw_sibling_stat / adjusted_stat)
            if np.isfinite(raw_sibling_stat)
            and np.isfinite(adjusted_stat)
            and adjusted_stat > 0.0
            else float("nan")
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "mode": mode,
                "source_family": source_family,
                "feature_representation": feature_representation,
                "case_id": case_id,
                "replicate": int(replicate),
                "tree_seed": int(tree_seed),
                "data_seed": int(data_seed),
                "parent_id": str(parent),
                "node_depth": int(depths[parent]),
                "n_parent": n_parent,
                "n_left": n_left,
                "n_right": n_right,
                "child_balance": float(min(n_left, n_right) / max(n_left + n_right, 1)),
                "sibling_alpha": float(sibling_alpha),
                "edge_alpha": float(edge_alpha),
                "left_edge_raw_p": left_edge_raw,
                "right_edge_raw_p": right_edge_raw,
                "left_edge_bh_p": left_edge_bh,
                "right_edge_bh_p": right_edge_bh,
                "edge_path_open": bool(
                    annotations_df.at[left, "Child_Parent_Divergence_Significant"]
                    or annotations_df.at[right, "Child_Parent_Divergence_Significant"]
                ),
                "sibling_raw_stat": raw_sibling_stat,
                "sibling_adjusted_stat": adjusted_stat,
                "sibling_df": sibling_df,
                "sibling_raw_p": raw_sibling_p,
                "inflation_factor": inflation_factor,
                "sibling_adjusted_p": adjusted_p,
                "sibling_bh_p": sibling_bh_p,
                "sibling_rejected": bool(
                    annotations_df.at[parent, "Sibling_BH_Different"]
                )
                if has_sibling_annotations
                else False,
                "calibration_support_status": _calibration_support_status(
                    annotations_df.loc[parent]
                )
                if has_sibling_annotations
                else "raw_only_pre_calibration",
                "sibling_projection_dimension": sibling_projection_dimension,
                "edge_to_sibling_cosine": float("nan"),
                "edge_to_sibling_cosine_squared": float("nan"),
                "sibling_subspace_capture": float("nan"),
                "principal_angle_min": float("nan"),
                "principal_angle_max": float("nan"),
                "selected_ratio": selected_ratio,
                "adjusted_selected_ratio": adjusted_selected_ratio,
                "selected_ratio_log": (
                    float(np.log(selected_ratio)) if selected_ratio > 0.0 else float("nan")
                ),
                **covariance_fields,
                **sibling_laplacian_fields,
                **parent_laplacian_fields,
            }
        )
    return rows


def _decompose_for_final(
    *,
    tree: PosetTree,
    data: pd.DataFrame,
    edge_alpha: float,
    sibling_alpha: float,
    feature_space: FeatureSpace,
) -> tuple[str, int, str]:
    try:
        result = tree.decompose(
            annotations_df=tree.annotations_df,
            leaf_data=data,
            feature_space=feature_space,
            edge_alpha=float(edge_alpha),
            sibling_alpha=float(sibling_alpha),
        )
    except ValueError as exc:
        return "decomposition_error", 0, str(exc)
    return "ok", int(result["num_clusters"]), ""


def run_selected_edge_replicate(
    *,
    case_id: str,
    source_family: str,
    feature_representation: str,
    n_samples: int,
    n_features: int,
    n_categories: int | None = None,
    replicate: int,
    data_seed: int,
    tree_seed: int,
    mode: str,
    edge_alpha: float,
    sibling_alpha: float,
    run_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run one selected-edge Type-I replicate."""
    tree, data, feature_space, selected_tree, fixed_tree = _prepare_tree_for_mode(
        mode=mode,
        case_id=case_id,
        source_family=source_family,
        feature_representation=feature_representation,
        n_samples=n_samples,
        n_features=n_features,
        n_categories=n_categories,
        tree_seed=tree_seed,
        data_seed=data_seed,
    )
    edge_annotations, spectral_context = _annotate_edges(
        tree,
        data,
        edge_alpha=edge_alpha,
        feature_space=feature_space,
    )
    edge_rows = extract_edge_geometry_rows(
        tree=tree,
        annotations_df=edge_annotations,
        spectral_context=spectral_context,
        run_id=run_id,
        mode=mode,
        source_family=source_family,
        feature_representation=feature_representation,
        case_id=case_id,
        replicate=replicate,
        tree_seed=tree_seed,
        data_seed=data_seed,
        edge_alpha=edge_alpha,
        selected_tree=selected_tree,
        fixed_tree=fixed_tree,
    )
    raw_sibling_records_by_parent = _collect_raw_sibling_records(
        tree=tree,
        annotations_df=edge_annotations,
        spectral_context=spectral_context,
        feature_space=feature_space,
    )

    status, found_clusters, failure_reason = _decompose_for_final(
        tree=tree,
        data=data,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
        feature_space=feature_space,
    )
    annotations_after_decomposition = tree.annotations_df
    sibling_rows = (
        extract_sibling_geometry_rows(
            tree=tree,
            annotations_df=annotations_after_decomposition,
            spectral_context=spectral_context,
            feature_space=feature_space,
            raw_sibling_records_by_parent=raw_sibling_records_by_parent,
            run_id=run_id,
            mode=mode,
            source_family=source_family,
            feature_representation=feature_representation,
            case_id=case_id,
            replicate=replicate,
            tree_seed=tree_seed,
            data_seed=data_seed,
            edge_alpha=edge_alpha,
            sibling_alpha=sibling_alpha,
        )
        if annotations_after_decomposition is not None
        else []
    )
    final_rows = [
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "mode": mode,
            "source_family": source_family,
            "feature_representation": feature_representation,
            "case_id": case_id,
            "replicate": int(replicate),
            "tree_seed": int(tree_seed),
            "data_seed": int(data_seed),
            "edge_alpha": float(edge_alpha),
            "sibling_alpha": float(sibling_alpha),
            "n_samples": int(n_samples),
            "n_features": int(data.shape[1]),
            "true_null": True,
            "status": status,
            "failure_reason": failure_reason,
            "found_clusters": int(found_clusters),
            "false_split": bool(found_clusters > 1),
            "max_depth_opened": int(
                max((row["node_depth"] for row in edge_rows if row["edge_rejected"]), default=0)
            ),
            "n_edge_tested": int(sum(bool(row["edge_tested"]) for row in edge_rows)),
            "n_edge_rejected": int(sum(bool(row["edge_rejected"]) for row in edge_rows)),
            "n_sibling_tested": int(
                sum(np.isfinite(float(row["sibling_adjusted_p"])) for row in sibling_rows)
            ),
            "n_sibling_rejected": int(sum(bool(row["sibling_rejected"]) for row in sibling_rows)),
            "n_calibration_support_failures": int(status == "decomposition_error"),
        }
    ]
    return edge_rows, sibling_rows, final_rows


def _select_cases(*, suite: str, case_names: Sequence[str]) -> list[dict[str, object]]:
    cases = get_default_test_cases() if suite == "default" else get_test_cases_by_suite(suite)
    if not case_names:
        return [dict(case) for case in cases]
    by_name = {str(case["name"]): case for case in cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown case names for suite {suite!r}: {missing!r}")
    return [dict(by_name[name]) for name in case_names]


def _case_contract(case: dict[str, object]) -> tuple[str, str, str, int, int, int | None]:
    name = str(case["name"])
    generator = str(case.get("generator", ""))
    if generator in {"binary", "planted_hierarchy_deep_signal"}:
        return (
            name,
            "binary_template",
            "binary",
            int(case["n_samples"]),
            int(case["n_features"]),
            None,
        )
    if generator == "categorical":
        return (
            name,
            "categorical_multinomial",
            "categorical_one_hot",
            int(case["n_samples"]),
            int(case["n_features"]),
            int(case["n_categories"]),
        )
    if generator in {
        "blobs_continuous",
        "dimensional_gaussian_continuous",
        "gaussian_outliers_continuous",
    }:
        raise ValueError(
            "Selected-edge Type-I runner does not implement continuous null "
            f"regeneration yet; case {name!r} has generator={generator!r}."
        )
    if generator in {"blobs_quantile", "phylogenetic", "temporal_evolution"}:
        raise ValueError(
            "Selected-edge Type-I runner supports direct categorical generator "
            "cases only. Quantile, phylogenetic, and temporal sequence nulls "
            f"need their own selected-null generator; case {name!r} has "
            f"generator={generator!r}."
        )
    else:
        raise ValueError(
            "Selected-edge Type-I runner supports binary and direct categorical "
            "generator cases only; "
            f"case {name!r} has generator={generator!r}."
        )


def run_selected_edge_geometry(config: SelectedEdgeGeometryConfig) -> dict[str, object]:
    """Run selected-edge diagnostics and write CSV outputs."""
    return run_selected_edge_geometry_for_replicates(
        config,
        replicate_indices=tuple(range(config.replicates)),
    )


def run_selected_edge_geometry_for_replicates(
    config: SelectedEdgeGeometryConfig,
    *,
    replicate_indices: Sequence[int],
) -> dict[str, object]:
    """Run selected-edge diagnostics for explicit replicate indices."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(config)
    cases = _select_cases(suite=config.suite, case_names=config.case_names)
    replicate_tuple = tuple(int(index) for index in replicate_indices)
    if not replicate_tuple:
        raise ValueError("At least one replicate index is required.")
    invalid_replicates = [
        index for index in replicate_tuple if index < 0 or index >= config.replicates
    ]
    if invalid_replicates:
        raise ValueError(
            "Replicate indices must satisfy 0 <= index < replicates; "
            f"invalid={invalid_replicates!r}, replicates={config.replicates!r}."
        )
    if len(set(replicate_tuple)) != len(replicate_tuple):
        raise ValueError(f"Replicate indices must be unique; got {replicate_tuple!r}.")

    edge_rows: list[dict[str, object]] = []
    sibling_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    for case in cases:
        (
            case_id,
            source_family,
            feature_representation,
            n_samples,
            n_features,
            n_categories,
        ) = _case_contract(case)
        for replicate in replicate_tuple:
            for edge_alpha in config.edge_alphas:
                for mode in config.modes:
                    data_seed = config.base_seed + replicate * 1009 + int(edge_alpha * 1_000_000)
                    tree_seed = config.base_seed + replicate * 917 + 17
                    replicate_edges, replicate_siblings, replicate_final = (
                        run_selected_edge_replicate(
                            case_id=case_id,
                            source_family=source_family,
                            feature_representation=feature_representation,
                            n_samples=n_samples,
                            n_features=n_features,
                            n_categories=n_categories,
                            replicate=replicate,
                            data_seed=data_seed,
                            tree_seed=tree_seed,
                            mode=mode,
                            edge_alpha=edge_alpha,
                            sibling_alpha=config.sibling_alpha,
                            run_id=run_id,
                        )
                    )
                    edge_rows.extend(replicate_edges)
                    sibling_rows.extend(replicate_siblings)
                    final_rows.extend(replicate_final)

    pd.DataFrame.from_records(edge_rows).to_csv(config.edge_rows_path, index=False)
    pd.DataFrame.from_records(sibling_rows).to_csv(config.sibling_rows_path, index=False)
    pd.DataFrame.from_records(final_rows).to_csv(config.final_rows_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "suite": config.suite,
        "case_names": [str(case["name"]) for case in cases],
        "modes": list(config.modes),
        "edge_alphas": list(config.edge_alphas),
        "sibling_alpha": float(config.sibling_alpha),
        "replicates": int(config.replicates),
        "replicate_indices": list(replicate_tuple),
        "base_seed": int(config.base_seed),
        "n_edge_rows": int(len(edge_rows)),
        "n_sibling_rows": int(len(sibling_rows)),
        "n_final_rows": int(len(final_rows)),
        "outputs": {
            "edges": str(config.edge_rows_path),
            "siblings": str(config.sibling_rows_path),
            "final": str(config.final_rows_path),
        },
        "note": (
            "Diagnostic selected-edge Type-I and geometry evidence only. "
            "Does not change production defaults or add a fallback calibration law."
        ),
    }
    config.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", default="binary")
    parser.add_argument("--case-names")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument(
        "--edge-alphas",
        default=",".join(str(value) for value in DEFAULT_EDGE_ALPHA_GRID),
    )
    parser.add_argument("--sibling-alpha", type=float, default=0.01)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260604)
    args = parser.parse_args(argv)

    if args.replicates <= 0:
        raise ValueError("replicates must be positive.")
    if not 0.0 < float(args.sibling_alpha) < 1.0:
        raise ValueError("sibling_alpha must lie in (0, 1).")
    config = SelectedEdgeGeometryConfig(
        output_dir=args.output_dir,
        suite=str(args.suite),
        case_names=parse_names(args.case_names),
        modes=validate_modes(parse_names(args.modes)),
        edge_alphas=parse_alpha_grid(str(args.edge_alphas)),
        sibling_alpha=float(args.sibling_alpha),
        replicates=int(args.replicates),
        base_seed=int(args.base_seed),
    )
    manifest = run_selected_edge_geometry(config)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
