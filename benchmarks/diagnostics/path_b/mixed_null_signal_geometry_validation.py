"""Mixed null/signal validation for recursive p-value geometry.

This diagnostic labels sibling contexts from benchmark ground truth, joins
those labels to recursive edge/sibling/eigenspace geometry, and evaluates
predeclared covariate sets. It is diagnostic only: it does not change runtime
calibration, sibling gates, edge gates, or traversal.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks.diagnostics.path_b.recursive_pvalue_geometry import (
    build_recursive_pvalue_geometry_panels,
)
from benchmarks.shared.cases import (
    BINARY_BENCHMARK_GENERATORS,
    CATEGORICAL_BENCHMARK_GENERATORS,
    CONTINUOUS_BENCHMARK_GENERATORS,
    DISCRETIZED_GAUSSIAN_BENCHMARK_GENERATORS,
    GRAPH_BENCHMARK_GENERATORS,
    get_test_cases_by_suite,
)
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.time import format_timestamp_utc
from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.contrast_covariance import (
    build_null_whitened_tangent_matrix,
)
from kl_clustering_analysis.tree.distributions import (
    require_node_continuous_covariance_by_block,
)
from kl_clustering_analysis.tree.feature_space import FeatureSpace

STUDY_ROLE = "diagnostic_mixed_null_signal_geometry_not_calibration"


@dataclass(frozen=True)
class MixedGeometryOutputs:
    labeled_nodes_csv: Path
    edge_panel_csv: Path
    model_validation_csv: Path
    model_summary_csv: Path
    calibration_summary_csv: Path
    case_status_csv: Path
    report_md: Path
    manifest_json: Path


@dataclass(frozen=True)
class GeometryModelSpec:
    model_id: str
    description: str
    predictors: tuple[str, ...]
    estimator: str = "linear"


def _feature_family(case: dict[str, object], metadata: dict[str, object]) -> str:
    feature_space = metadata.get("feature_space")
    if feature_space is not None and hasattr(feature_space, "family_label"):
        return str(feature_space.family_label)
    generator = str(case.get("generator", "unknown"))
    if generator in BINARY_BENCHMARK_GENERATORS:
        return "bernoulli"
    if generator in CATEGORICAL_BENCHMARK_GENERATORS:
        return "categorical"
    if generator in CONTINUOUS_BENCHMARK_GENERATORS:
        return "continuous"
    if generator in DISCRETIZED_GAUSSIAN_BENCHMARK_GENERATORS:
        return "discretized_gaussian"
    if generator in GRAPH_BENCHMARK_GENERATORS:
        return "graph"
    return generator


def _depth_bin(depth: object) -> str:
    depth_value = pd.to_numeric(pd.Series([depth]), errors="coerce").iloc[0]
    if not np.isfinite(depth_value):
        return "depth_unknown"
    depth_int = int(depth_value)
    if depth_int <= 1:
        return "depth_0_1"
    if depth_int <= 3:
        return "depth_2_3"
    if depth_int <= 7:
        return "depth_4_7"
    if depth_int <= 15:
        return "depth_8_15"
    return "depth_ge16"


def _parent_size_bin(parent_fraction: float) -> str:
    if not np.isfinite(parent_fraction) or parent_fraction <= 0.0:
        return "parent_unknown"
    if parent_fraction <= 0.05:
        return "parent_0_0.05"
    if parent_fraction <= 0.10:
        return "parent_0.05_0.10"
    if parent_fraction <= 0.25:
        return "parent_0.10_0.25"
    if parent_fraction <= 0.50:
        return "parent_0.25_0.50"
    return "parent_0.50_1.00"


def _root(tree: Any) -> object:
    if hasattr(tree, "root"):
        return tree.root()
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, got {roots!r}.")
    return roots[0]


def _vector_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=float)))


def _unoriented_angle_to_leading_axis_deg(vector: np.ndarray) -> float:
    values = np.asarray(vector, dtype=float)
    radius = _vector_norm(values)
    if values.size == 0 or radius <= 1e-12:
        return math.nan
    cosine = abs(float(values[0]) / radius)
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def _independent_radius_fraction(vector: np.ndarray) -> float:
    values = np.asarray(vector, dtype=float)
    radius = _vector_norm(values)
    if values.size <= 1 or radius <= 1e-12:
        return 0.0
    return float(_vector_norm(values[1:]) / radius)


def _safe_median_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    norms = np.linalg.norm(np.asarray(values, dtype=float), axis=1)
    finite = norms[np.isfinite(norms)]
    return float(np.median(finite)) if finite.size else math.nan


def _truth_counts(labels: Sequence[object]) -> dict[object, int]:
    counts: dict[object, int] = {}
    for label in labels:
        if pd.isna(label):
            continue
        key = label.item() if hasattr(label, "item") else label
        counts[key] = counts.get(key, 0) + 1
    return counts


def _dominant_label_and_purity(counts: dict[object, int]) -> tuple[object | None, float]:
    total = int(sum(counts.values()))
    if total <= 0:
        return None, math.nan
    label, count = max(counts.items(), key=lambda item: item[1])
    return label, float(count / total)


def _same_label_pair_rate(left_counts: dict[object, int], right_counts: dict[object, int]) -> float:
    left_total = int(sum(left_counts.values()))
    right_total = int(sum(right_counts.values()))
    if left_total <= 0 or right_total <= 0:
        return math.nan
    numerator = sum(
        int(left_counts.get(label, 0)) * int(right_counts.get(label, 0))
        for label in set(left_counts).union(right_counts)
    )
    return float(numerator / (left_total * right_total))


def _entropy(counts: dict[object, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return math.nan
    probabilities = np.asarray([count / total for count in counts.values() if count > 0], dtype=float)
    return float(-np.sum(probabilities * np.log(probabilities)))


def classify_sibling_truth_context(
    left_labels: Sequence[object],
    right_labels: Sequence[object],
    *,
    purity_threshold: float = 0.80,
    null_same_label_rate_threshold: float = 0.80,
    signal_same_label_rate_threshold: float = 0.20,
) -> dict[str, object]:
    """Classify a binary sibling context using benchmark truth labels."""
    left_counts = _truth_counts(left_labels)
    right_counts = _truth_counts(right_labels)
    parent_counts: dict[object, int] = {}
    for label, count in left_counts.items():
        parent_counts[label] = parent_counts.get(label, 0) + count
    for label, count in right_counts.items():
        parent_counts[label] = parent_counts.get(label, 0) + count

    left_label, left_purity = _dominant_label_and_purity(left_counts)
    right_label, right_purity = _dominant_label_and_purity(right_counts)
    min_child_purity = float(np.nanmin([left_purity, right_purity]))
    same_label_rate = _same_label_pair_rate(left_counts, right_counts)
    truth_separation = 1.0 - same_label_rate if math.isfinite(same_label_rate) else math.nan
    parent_truth_count = len(parent_counts)

    if parent_truth_count <= 0:
        context = "unknown_context"
    elif parent_truth_count == 1:
        context = "null_context"
    elif (
        left_label == right_label
        and math.isfinite(same_label_rate)
        and same_label_rate >= null_same_label_rate_threshold
    ):
        context = "null_context"
    elif (
        left_label != right_label
        and math.isfinite(same_label_rate)
        and same_label_rate <= signal_same_label_rate_threshold
        and min_child_purity >= purity_threshold
    ):
        context = "signal_context"
    else:
        context = "mixed_context"

    return {
        "truth_context_label": context,
        "is_null_context": context == "null_context",
        "is_signal_context": context == "signal_context",
        "is_mixed_context": context == "mixed_context",
        "parent_truth_cluster_count": int(parent_truth_count),
        "left_truth_cluster_count": int(len(left_counts)),
        "right_truth_cluster_count": int(len(right_counts)),
        "left_dominant_truth_label": "" if left_label is None else str(left_label),
        "right_dominant_truth_label": "" if right_label is None else str(right_label),
        "left_dominant_purity": left_purity,
        "right_dominant_purity": right_purity,
        "min_child_truth_purity": min_child_purity,
        "cross_child_same_truth_pair_rate": same_label_rate,
        "truth_sibling_separation_score": truth_separation,
        "parent_truth_entropy": _entropy(parent_counts),
    }


def label_tree_sibling_contexts(
    *,
    case: dict[str, object],
    tree: Any,
    true_labels: np.ndarray,
    sample_index: pd.Index,
    feature_family: str,
) -> pd.DataFrame:
    """Return one truth-label row per binary internal tree node."""
    truth_by_leaf: dict[object, object] = {}
    for sample_id, label in zip(sample_index, true_labels, strict=True):
        truth_by_leaf[sample_id] = label
        truth_by_leaf[str(sample_id)] = label

    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    n_samples = int(len(sample_index))
    rows: list[dict[str, object]] = []
    for node in tree.nodes:
        children = list(tree.successors(node))
        if len(children) != 2:
            continue
        left, right = children
        left_leaves = tuple(descendant_sets[left])
        right_leaves = tuple(descendant_sets[right])
        left_truth = [truth_by_leaf.get(leaf, truth_by_leaf.get(str(leaf), np.nan)) for leaf in left_leaves]
        right_truth = [truth_by_leaf.get(leaf, truth_by_leaf.get(str(leaf), np.nan)) for leaf in right_leaves]
        left_size = len(left_truth)
        right_size = len(right_truth)
        parent_size = left_size + right_size
        if parent_size <= 0:
            continue
        beta_left = left_size / parent_size
        beta_right = right_size / parent_size
        balance = min(beta_left, beta_right)
        context = classify_sibling_truth_context(left_truth, right_truth)
        rows.append(
            {
                "case_id": str(case["name"]),
                "node_id": str(node),
                "left_child_id": str(left),
                "right_child_id": str(right),
                "generator": str(case.get("generator", "")),
                "case_category": str(case.get("category", "")),
                "feature_family": feature_family,
                "left_child_leaf_count_truth": int(left_size),
                "right_child_leaf_count_truth": int(right_size),
                "parent_leaf_count_truth": int(parent_size),
                "parent_size_fraction": float(parent_size / max(n_samples, 1)),
                "parent_size_bin": _parent_size_bin(parent_size / max(n_samples, 1)),
                "left_barycentric_weight": float(beta_left),
                "right_barycentric_weight": float(beta_right),
                "barycentric_balance": float(balance),
                "log_barycentric_leverage": (
                    float(math.log(max(beta_left, beta_right) / balance))
                    if balance > 0.0
                    else math.nan
                ),
                "log_sampling_variance_scale": float(math.log((1.0 / left_size) + (1.0 / right_size))),
                **context,
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def build_row_aligned_kak_geometry_panel(
    *,
    case_id: str,
    tree: Any,
    leaf_data: pd.DataFrame,
    feature_space: FeatureSpace | None,
    spectral_context: Any,
) -> pd.DataFrame:
    """Compute KAK-style radius/angle/action terms for KL sibling nodes.

    The frame is the root selected PCA basis in the same null-whitened tangent
    coordinates used by the edge gate. This makes KAK-style geometry row-aligned
    to the labeled sibling panel without relying on cached KAK block pages.
    """
    root = _root(tree)
    root_key = str(root)
    projection = spectral_context.principal_component_projections_by_node.get(root_key)
    if projection is None:
        return pd.DataFrame()
    projection_matrix = np.asarray(projection, dtype=float)
    if projection_matrix.ndim != 2 or projection_matrix.shape[0] == 0:
        return pd.DataFrame()

    continuous_covariance_by_block = None
    if feature_space is not None:
        continuous_covariance_by_block = require_node_continuous_covariance_by_block(
            tree,
            root,
            feature_space,
        )
    tangent = build_null_whitened_tangent_matrix(
        leaf_data.to_numpy(dtype=float, copy=False),
        np.asarray(tree.nodes[root]["distribution"], dtype=float),
        feature_space=feature_space,
        continuous_covariance_by_block=continuous_covariance_by_block,
        ridge=1e-12,
    )
    coords = tangent @ projection_matrix.T
    if coords.ndim != 2 or coords.shape[1] == 0:
        return pd.DataFrame()

    descendant_sets = tree.compute_descendant_sets(use_labels=True)
    leaf_to_index: dict[object, int] = {}
    for index, label in enumerate(leaf_data.index):
        leaf_to_index[label] = index
        leaf_to_index[str(label)] = index

    leading_axis_score = coords[:, 0]
    rows: list[dict[str, object]] = []
    for node in tree.nodes:
        children = list(tree.successors(node))
        if len(children) != 2:
            continue
        left, right = children
        try:
            left_indices = np.asarray(
                [leaf_to_index[leaf] for leaf in descendant_sets[left]],
                dtype=int,
            )
            right_indices = np.asarray(
                [leaf_to_index[leaf] for leaf in descendant_sets[right]],
                dtype=int,
            )
        except KeyError:
            continue
        if left_indices.size == 0 or right_indices.size == 0:
            continue

        parent_indices = np.concatenate([left_indices, right_indices])
        parent_coords = coords[parent_indices]
        left_coords = coords[left_indices]
        right_coords = coords[right_indices]
        parent_centroid = parent_coords.mean(axis=0)
        left_centroid = left_coords.mean(axis=0)
        right_centroid = right_coords.mean(axis=0)
        parent_centered = parent_coords - parent_centroid
        left_centered = left_coords - left_centroid
        right_centered = right_coords - right_centroid

        parent_radius_q50 = _safe_median_norm(parent_centered)
        left_radius_q50 = _safe_median_norm(left_centered)
        right_radius_q50 = _safe_median_norm(right_centered)
        parent_radius_scale = (
            max(parent_radius_q50, 1e-12)
            if math.isfinite(parent_radius_q50)
            else math.nan
        )
        child_radius_scale = (
            max(left_radius_q50 + right_radius_q50, 1e-12)
            if math.isfinite(left_radius_q50) and math.isfinite(right_radius_q50)
            else math.nan
        )

        n_left = int(left_indices.size)
        n_right = int(right_indices.size)
        n_parent = n_left + n_right
        balance = float(min(n_left, n_right) / n_parent)
        sibling_vector = left_centroid - right_centroid
        sibling_distance = _vector_norm(sibling_vector)
        parent_ratio = (
            float(sibling_distance / parent_radius_scale)
            if math.isfinite(parent_radius_scale)
            else math.nan
        )
        child_ratio = (
            float(sibling_distance / child_radius_scale)
            if math.isfinite(child_radius_scale)
            else math.nan
        )
        action_proxy = (
            balance * (1.0 - balance) * parent_ratio**2
            if math.isfinite(parent_ratio)
            else math.nan
        )
        action_capped = min(action_proxy, 1.0) if math.isfinite(action_proxy) else math.nan
        angle = _unoriented_angle_to_leading_axis_deg(parent_centroid)
        independent = _independent_radius_fraction(parent_centroid)
        rows.append(
            {
                "case_id": str(case_id),
                "node_id": str(node),
                "row_aligned_kak_frame": "root_selected_pca_whitened_tangent",
                "row_aligned_kak_projection_dimension": int(coords.shape[1]),
                "geometry_parent_radius": _vector_norm(parent_centroid),
                "geometry_angle_to_leading_axis_deg": angle,
                "geometry_independent_radius_fraction": independent,
                "geometry_sibling_separation_parent_ratio": parent_ratio,
                "geometry_sibling_separation_child_ratio": child_ratio,
                "geometry_abs_common_axis_gap": float(
                    abs(
                        np.mean(leading_axis_score[left_indices])
                        - np.mean(leading_axis_score[right_indices])
                    )
                ),
                "action_budget_proxy": action_proxy,
                "action_budget_proxy_capped": action_capped,
                "angular_shell_risk_score": (
                    action_capped * np.clip(angle / 90.0, 0.0, 1.0) * independent
                    if math.isfinite(action_capped) and math.isfinite(angle)
                    else math.nan
                ),
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows)


def _finite_numeric(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(np.nan, index=table.index, dtype=float)
    return pd.to_numeric(table[column], errors="coerce").astype(float)


def _finite_median(table: pd.DataFrame, column: str) -> float:
    values = _finite_numeric(table, column).replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else math.nan


def _edge_parent_aggregates(edge_panel: pd.DataFrame) -> pd.DataFrame:
    if edge_panel.empty:
        return pd.DataFrame()
    table = edge_panel.copy()
    group_cols = ["case_id", "parent_id"]
    rows: list[dict[str, object]] = []
    for (case_id, parent_id), group in table.groupby(group_cols, sort=False):
        raw_margin = _finite_numeric(group, "edge_raw_alpha_margin")
        bh_margin = _finite_numeric(group, "edge_alpha_margin")
        rows.append(
            {
                "case_id": str(case_id),
                "node_id": str(parent_id),
                "min_child_edge_raw_neglog10_p": float(
                    _finite_numeric(group, "edge_raw_neglog10_p").min()
                ),
                "min_child_edge_bh_neglog10_p": float(
                    _finite_numeric(group, "edge_neglog10_bh_p").min()
                ),
                "max_child_edge_raw_neglog10_p": float(
                    _finite_numeric(group, "edge_raw_neglog10_p").max()
                ),
                "edge_raw_action_count": int((raw_margin > 0.0).sum()),
                "edge_bh_action_count": int((bh_margin > 0.0).sum()),
                "mean_child_subspace_chordal_distance": float(
                    _finite_numeric(group, "subspace_chordal_distance_normalized").mean()
                ),
                "max_child_subspace_chordal_distance": float(
                    _finite_numeric(group, "subspace_chordal_distance_normalized").max()
                ),
                "max_edge_raw_minus_parent_sibling_raw_neglog10": float(
                    _finite_numeric(
                        group,
                        "edge_raw_minus_parent_sibling_raw_neglog10",
                    ).max()
                ),
                "max_child_raw_minus_parent_sibling_raw_neglog10": float(
                    _finite_numeric(
                        group,
                        "child_raw_minus_parent_sibling_raw_neglog10",
                    ).max()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def build_mixed_null_signal_geometry_panel(
    *,
    node_panel: pd.DataFrame,
    edge_panel: pd.DataFrame,
    truth_labels: pd.DataFrame,
    sibling_alpha: float,
    kak_geometry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join recursive geometry to ground-truth sibling-context labels."""
    if node_panel.empty or truth_labels.empty:
        return pd.DataFrame()
    edge_agg = _edge_parent_aggregates(edge_panel)
    table = node_panel.merge(
        truth_labels.drop(columns=["study_role"], errors="ignore"),
        on=["case_id", "node_id"],
        how="inner",
    )
    if not edge_agg.empty:
        table = table.merge(edge_agg, on=["case_id", "node_id"], how="left")
    if kak_geometry is not None and not kak_geometry.empty:
        table = table.merge(
            kak_geometry.drop(columns=["study_role"], errors="ignore"),
            on=["case_id", "node_id"],
            how="left",
        )

    table["depth_bin"] = [_depth_bin(value) for value in table["depth"]]
    table["split_rejected_at_sibling_alpha"] = table["sibling_bh_different"].astype(bool)
    table["raw_sibling_rejected_at_alpha"] = (
        _finite_numeric(table, "sibling_raw_p_value") <= float(sibling_alpha)
    )
    table["selected_nonnull_leakage_risk"] = (
        table["is_signal_context"].astype(bool) | table["is_mixed_context"].astype(bool)
    )
    table["edge_action"] = _finite_numeric(table, "min_child_edge_raw_neglog10_p")
    table["edge_bh_action"] = _finite_numeric(table, "min_child_edge_bh_neglog10_p")
    table["log_parent_leaf_count"] = np.log(
        np.maximum(_finite_numeric(table, "parent_leaf_count_truth"), 1.0)
    )
    table["log_parent_size_fraction"] = np.log(
        np.maximum(_finite_numeric(table, "parent_size_fraction"), 1e-12)
    )
    table["log_selected_eigenvalue_gap_ratio"] = np.log(
        np.maximum(
            _finite_numeric(table, "selected_eigenvalue_gap_ratio").replace(np.inf, np.nan),
            1e-12,
        )
    )
    table["sibling_tail_sensitivity_log10"] = np.log10(
        np.maximum(_finite_numeric(table, "sibling_chi_square_tail_sensitivity"), 1e-300)
    )
    table["edge_sibling_alpha_margin_gap"] = (
        _finite_numeric(table, "edge_raw_alpha_margin")
        if "edge_raw_alpha_margin" in table.columns
        else np.nan
    )
    table["study_role"] = STUDY_ROLE
    return table


def geometry_model_specs(table: pd.DataFrame | None = None) -> tuple[GeometryModelSpec, ...]:
    """Return predeclared covariate sets for mixed null/signal validation."""
    base_context = (
        "log_parent_leaf_count",
        "log_parent_size_fraction",
        "depth",
        "test_projection_dimension",
    )
    chi_square = (
        "sibling_raw_neglog10_p",
        "sibling_degrees_of_freedom",
        "sibling_chi_square_tail_sensitivity",
    )
    selected_tail = (
        "edge_action",
        "edge_raw_action_count",
        "log_parent_leaf_count",
        "depth",
        "test_projection_dimension",
    )
    barycentric_spectral = (
        *selected_tail,
        "barycentric_balance",
        "log_barycentric_leverage",
        "log_sampling_variance_scale",
        "raw_mp_signal_count",
        "log_selected_eigenvalue_gap_ratio",
        "top_selected_eigenvalue_mass_fraction",
        "selected_eigenvalue_effective_rank",
    )
    recursive_geometry = (
        *barycentric_spectral,
        "sibling_raw_alpha_margin",
        "sibling_alpha_margin",
        "recursive_sibling_neglog10_gradient",
        "edge_sibling_connectivity_score",
        "max_edge_raw_minus_parent_sibling_raw_neglog10",
        "max_child_raw_minus_parent_sibling_raw_neglog10",
        "mean_child_subspace_chordal_distance",
        "max_child_subspace_chordal_distance",
    )
    kak_core_terms = (
        "geometry_parent_radius",
        "geometry_angle_to_leading_axis_deg",
        "geometry_independent_radius_fraction",
        "action_budget_proxy",
        "action_budget_proxy_capped",
    )
    kak_surface_terms = (
        *kak_core_terms,
        "geometry_sibling_separation_parent_ratio",
        "geometry_sibling_separation_child_ratio",
        "geometry_abs_common_axis_gap",
        "angular_shell_risk_score",
    )
    continuous_surface = (
        *chi_square,
        *recursive_geometry,
        *kak_surface_terms,
    )
    return (
        GeometryModelSpec(
            "context_only",
            "truth_context ~ parent size + depth + projection dimension",
            base_context,
        ),
        GeometryModelSpec(
            "chi_square_only",
            "truth_context ~ sibling chi-square p-value coordinates",
            chi_square,
        ),
        GeometryModelSpec(
            "selected_tail_baseline",
            "truth_context ~ edge action + parent size + depth + projection dimension",
            selected_tail,
        ),
        GeometryModelSpec(
            "barycentric_edge_spectral",
            "truth_context ~ edge action + barycentric balance + selected spectrum",
            barycentric_spectral,
        ),
        GeometryModelSpec(
            "recursive_pvalue_geometry",
            "truth_context ~ barycentric/spectral + recursive edge/sibling geometry",
            recursive_geometry,
        ),
        GeometryModelSpec(
            "kak_radius_angle_action",
            "truth_context ~ KAK radius/angle/action terms when row-aligned columns exist",
            (*recursive_geometry, *kak_core_terms),
        ),
        GeometryModelSpec(
            "sklearn_logistic_edge_sibling_kak_surface",
            "truth_context ~ regularized scikit-learn continuous edge/sibling/KAK surface",
            continuous_surface,
            estimator="sklearn_logistic",
        ),
        GeometryModelSpec(
            "sklearn_hist_gradient_edge_sibling_kak_surface",
            "truth_context ~ scikit-learn nonlinear edge/sibling/KAK surface",
            continuous_surface,
            estimator="sklearn_hist_gradient",
        ),
    )


def _binary_auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same length.")
    finite = np.isfinite(scores) & np.isfinite(labels)
    scores = scores[finite]
    labels = labels[finite].astype(bool)
    n_positive = int(np.sum(labels))
    n_negative = int(labels.shape[0] - n_positive)
    if n_positive == 0 or n_negative == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[labels]))
    return float(
        (positive_rank_sum - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def _dummy_columns(table: pd.DataFrame, column: str) -> tuple[str, ...]:
    if column not in table.columns:
        return ()
    values = tuple(sorted(str(value) for value in table[column].dropna().unique()))
    if len(values) <= 1:
        return ()
    created: list[str] = []
    for value in values[1:]:
        column_name = f"{column}__{value}"
        table[column_name] = table[column].astype(str).eq(value).astype(float)
        created.append(column_name)
    return tuple(created)


def prepare_model_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Return finite null/signal rows with numeric predictors and dummies."""
    if panel.empty:
        return pd.DataFrame()
    table = panel[panel["truth_context_label"].isin(["null_context", "signal_context"])].copy()
    if table.empty:
        return table
    table["signal_label"] = table["is_signal_context"].astype(bool).astype(float)
    table["null_label"] = table["is_null_context"].astype(bool).astype(float)
    for column in (
        "feature_family",
        "depth_bin",
        "parent_size_bin",
        "case_category",
    ):
        _dummy_columns(table, column)
    return table.replace([np.inf, -np.inf], np.nan)


def _fold_ids(table: pd.DataFrame, split_strategy: str) -> pd.Series:
    if split_strategy == "leave_one_case_out":
        return table["case_id"].astype(str)
    if split_strategy == "leave_one_feature_family_out":
        return table["feature_family"].astype(str)
    if split_strategy == "leave_one_depth_bin_out":
        return table["depth_bin"].astype(str)
    if split_strategy == "case_hash_modulo_5":
        return table["case_id"].astype(str).map(lambda value: zlib.crc32(value.encode("utf-8")) % 5)
    raise ValueError(f"Unknown split strategy {split_strategy!r}.")


def _active_predictors(
    table: pd.DataFrame,
    predictors: Sequence[str],
) -> tuple[str, ...]:
    available = [predictor for predictor in predictors if predictor in table.columns]
    available.extend(
        column
        for prefix in ("feature_family__", "depth_bin__", "parent_size_bin__", "case_category__")
        for column in table.columns
        if column.startswith(prefix)
    )
    unique_available = tuple(dict.fromkeys(available))
    return tuple(
        predictor
        for predictor in unique_available
        if table[predictor].replace([np.inf, -np.inf], np.nan).dropna().nunique() > 1
    )


def _fit_linear_score(
    train: pd.DataFrame,
    *,
    predictors: Sequence[str],
    min_train_rows_per_predictor: int,
) -> dict[str, object]:
    active = _active_predictors(train, predictors)
    minimum_rows = 0
    finite = pd.DataFrame()
    while active:
        columns = list(active) + ["signal_label"]
        candidate = train[columns].replace([np.inf, -np.inf], np.nan).dropna()
        minimum_rows = max(
            len(active) + 2,
            int(min_train_rows_per_predictor) * max(len(active), 1),
        )
        if candidate.shape[0] >= minimum_rows:
            finite = candidate
            break
        missing_counts = train[list(active)].replace([np.inf, -np.inf], np.nan).isna().sum()
        drop_column = str(missing_counts.sort_values(ascending=False).index[0])
        active = tuple(predictor for predictor in active if predictor != drop_column)
    if not active:
        return {
            "status": "insufficient_train_rows",
            "active_predictors": active,
            "n_train": 0,
            "minimum_rows": int(minimum_rows),
        }
    if finite["signal_label"].nunique() < 2:
        return {"status": "single_class_train", "active_predictors": active, "n_train": int(finite.shape[0])}
    x_raw = finite[list(active)].to_numpy(dtype=float)
    means = x_raw.mean(axis=0)
    stds = x_raw.std(axis=0, ddof=0)
    positive_std = stds > 0.0
    if not bool(np.all(positive_std)):
        active = tuple(np.asarray(active, dtype=object)[positive_std].tolist())
        x_raw = x_raw[:, positive_std]
        means = means[positive_std]
        stds = stds[positive_std]
    if not active:
        return {"status": "no_nonconstant_predictors", "active_predictors": active, "n_train": int(finite.shape[0])}
    x = (x_raw - means) / stds
    design = np.column_stack([np.ones(x.shape[0]), x])
    y = finite["signal_label"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "status": "ok",
        "active_predictors": active,
        "n_train": int(finite.shape[0]),
        "means": means,
        "standard_deviations": stds,
        "coefficients": coefficients,
        "train_signal_rate": float(np.mean(y)),
    }


def _predict_linear_score(model: dict[str, object], test: pd.DataFrame) -> pd.Series:
    active = tuple(str(value) for value in model["active_predictors"])
    finite = test[list(active)].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(dtype=float)
    x_raw = finite[list(active)].to_numpy(dtype=float)
    means = np.asarray(model["means"], dtype=float)
    stds = np.asarray(model["standard_deviations"], dtype=float)
    x = (x_raw - means) / stds
    design = np.column_stack([np.ones(x.shape[0]), x])
    scores = design @ np.asarray(model["coefficients"], dtype=float)
    return pd.Series(scores, index=finite.index, dtype=float)


def _fit_sklearn_score(
    train: pd.DataFrame,
    *,
    predictors: Sequence[str],
    estimator: str,
    min_train_rows_per_predictor: int,
) -> dict[str, object]:
    active = _active_predictors(train, predictors)
    minimum_rows = 0
    finite = pd.DataFrame()
    while active:
        columns = list(active) + ["signal_label"]
        candidate = train[columns].replace([np.inf, -np.inf], np.nan).dropna()
        minimum_rows = max(
            len(active) + 2,
            int(min_train_rows_per_predictor) * max(len(active), 1),
        )
        if candidate.shape[0] >= minimum_rows:
            finite = candidate
            break
        missing_counts = train[list(active)].replace([np.inf, -np.inf], np.nan).isna().sum()
        drop_column = str(missing_counts.sort_values(ascending=False).index[0])
        active = tuple(predictor for predictor in active if predictor != drop_column)
    if not active:
        return {
            "status": "insufficient_train_rows",
            "active_predictors": active,
            "n_train": 0,
            "minimum_rows": int(minimum_rows),
        }
    if finite["signal_label"].nunique() < 2:
        return {
            "status": "single_class_train",
            "active_predictors": active,
            "n_train": int(finite.shape[0]),
        }

    if estimator == "sklearn_logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=20260606,
            ),
        )
    elif estimator == "sklearn_hist_gradient":
        model = HistGradientBoostingClassifier(
            l2_regularization=1e-3,
            max_leaf_nodes=15,
            random_state=20260606,
        )
    else:
        raise ValueError(f"Unknown sklearn estimator {estimator!r}.")

    x = finite[list(active)].to_numpy(dtype=float)
    y = finite["signal_label"].to_numpy(dtype=int)
    model.fit(x, y)
    return {
        "status": "ok",
        "active_predictors": active,
        "n_train": int(finite.shape[0]),
        "estimator": estimator,
        "model": model,
        "train_signal_rate": float(np.mean(y)),
    }


def _predict_sklearn_score(model: dict[str, object], test: pd.DataFrame) -> pd.Series:
    active = tuple(str(value) for value in model["active_predictors"])
    finite = test[list(active)].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(dtype=float)
    x = finite[list(active)].to_numpy(dtype=float)
    scores = model["model"].predict_proba(x)[:, 1]
    return pd.Series(scores, index=finite.index, dtype=float)


def _missing_row_aligned_kak_columns(table: pd.DataFrame, spec: GeometryModelSpec) -> list[str]:
    required = (
        "geometry_parent_radius",
        "geometry_angle_to_leading_axis_deg",
        "geometry_independent_radius_fraction",
        "action_budget_proxy",
        "action_budget_proxy_capped",
    )
    if not any(column in spec.predictors for column in required):
        return []
    return [column for column in required if column not in table.columns]


def evaluate_geometry_models(
    panel: pd.DataFrame,
    *,
    min_train_rows_per_predictor: int = 5,
    min_test_rows: int = 3,
) -> pd.DataFrame:
    """Evaluate predeclared geometry models on held-out null/signal rows."""
    table = prepare_model_table(panel)
    if table.empty:
        return pd.DataFrame()
    split_strategies = (
        "case_hash_modulo_5",
        "leave_one_feature_family_out",
        "leave_one_depth_bin_out",
        "leave_one_case_out",
    )
    rows: list[dict[str, object]] = []
    for split_strategy in split_strategies:
        folds = _fold_ids(table, split_strategy)
        for spec in geometry_model_specs(table):
            missing_kak = _missing_row_aligned_kak_columns(table, spec)
            if missing_kak:
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "model_description": spec.description,
                        "model_estimator": spec.estimator,
                        "split_strategy": split_strategy,
                        "predictors": ",".join(spec.predictors),
                        "n_train_rows": 0,
                        "n_test_rows": int(table.shape[0]),
                        "n_scored_rows": 0,
                        "n_null_scored_rows": 0,
                        "n_signal_scored_rows": 0,
                        "holdout_signal_auc": math.nan,
                        "score_signal_minus_null_mean": math.nan,
                        "null_false_split_rate_at_sibling_alpha": math.nan,
                        "signal_retention_rate_at_sibling_alpha": math.nan,
                        "null_raw_p_rejection_rate_at_alpha": math.nan,
                        "signal_raw_p_rejection_rate_at_alpha": math.nan,
                        "model_status": f"missing_row_aligned_kak_geometry_{split_strategy}",
                        "failure_reasons": "missing_columns:" + ",".join(missing_kak),
                        "study_role": STUDY_ROLE,
                    }
                )
                continue
            predictions: list[pd.Series] = []
            failures: list[str] = []
            n_train_total = 0
            n_test_total = 0
            for fold in tuple(sorted(folds.dropna().unique())):
                train = table.loc[folds.ne(fold)]
                test = table.loc[folds.eq(fold)]
                n_train_total += int(train.shape[0])
                n_test_total += int(test.shape[0])
                if test.shape[0] < min_test_rows:
                    failures.append(f"fold_{fold}:insufficient_test_rows")
                    continue
                if test["signal_label"].nunique() < 2:
                    failures.append(f"fold_{fold}:single_class_test")
                    continue
                if spec.estimator == "linear":
                    model = _fit_linear_score(
                        train,
                        predictors=spec.predictors,
                        min_train_rows_per_predictor=min_train_rows_per_predictor,
                    )
                else:
                    model = _fit_sklearn_score(
                        train,
                        predictors=spec.predictors,
                        estimator=spec.estimator,
                        min_train_rows_per_predictor=min_train_rows_per_predictor,
                    )
                if model["status"] != "ok":
                    failures.append(f"fold_{fold}:{model['status']}")
                    continue
                score = (
                    _predict_linear_score(model, test)
                    if spec.estimator == "linear"
                    else _predict_sklearn_score(model, test)
                )
                if score.empty:
                    failures.append(f"fold_{fold}:no_finite_test_predictors")
                    continue
                predictions.append(score)

            if not predictions:
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "model_description": spec.description,
                        "model_estimator": spec.estimator,
                        "split_strategy": split_strategy,
                        "predictors": ",".join(spec.predictors),
                        "n_train_rows": int(n_train_total),
                        "n_test_rows": int(n_test_total),
                        "n_scored_rows": 0,
                        "n_null_scored_rows": 0,
                        "n_signal_scored_rows": 0,
                        "holdout_signal_auc": math.nan,
                        "score_signal_minus_null_mean": math.nan,
                        "null_false_split_rate_at_sibling_alpha": math.nan,
                        "signal_retention_rate_at_sibling_alpha": math.nan,
                        "null_raw_p_rejection_rate_at_alpha": math.nan,
                        "signal_raw_p_rejection_rate_at_alpha": math.nan,
                        "model_status": f"no_valid_holdout_folds_{split_strategy}",
                        "failure_reasons": ";".join(failures),
                        "study_role": STUDY_ROLE,
                    }
                )
                continue

            scores = pd.concat(predictions).sort_index()
            scored = table.loc[scores.index].copy()
            labels = scored["signal_label"].to_numpy(dtype=float)
            score_values = scores.to_numpy(dtype=float)
            null_mask = scored["is_null_context"].astype(bool)
            signal_mask = scored["is_signal_context"].astype(bool)
            split_rejected = scored["split_rejected_at_sibling_alpha"].astype(bool)
            raw_rejected = scored["raw_sibling_rejected_at_alpha"].astype(bool)
            rows.append(
                {
                    "model_id": spec.model_id,
                    "model_description": spec.description,
                    "model_estimator": spec.estimator,
                    "split_strategy": split_strategy,
                    "predictors": ",".join(spec.predictors),
                    "n_train_rows": int(n_train_total),
                    "n_test_rows": int(n_test_total),
                    "n_scored_rows": int(scored.shape[0]),
                    "n_null_scored_rows": int(null_mask.sum()),
                    "n_signal_scored_rows": int(signal_mask.sum()),
                    "holdout_signal_auc": _binary_auc_score(score_values, labels),
                    "score_signal_minus_null_mean": float(
                        np.mean(score_values[signal_mask.to_numpy(dtype=bool)])
                        - np.mean(score_values[null_mask.to_numpy(dtype=bool)])
                    )
                    if bool(null_mask.any()) and bool(signal_mask.any())
                    else math.nan,
                    "null_false_split_rate_at_sibling_alpha": float(
                        split_rejected[null_mask].mean()
                    )
                    if bool(null_mask.any())
                    else math.nan,
                    "signal_retention_rate_at_sibling_alpha": float(
                        split_rejected[signal_mask].mean()
                    )
                    if bool(signal_mask.any())
                    else math.nan,
                    "null_raw_p_rejection_rate_at_alpha": float(raw_rejected[null_mask].mean())
                    if bool(null_mask.any())
                    else math.nan,
                    "signal_raw_p_rejection_rate_at_alpha": float(raw_rejected[signal_mask].mean())
                    if bool(signal_mask.any())
                    else math.nan,
                    "model_status": f"diagnostic_holdout_{split_strategy}",
                    "failure_reasons": ";".join(failures),
                    "study_role": STUDY_ROLE,
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_geometry_models(validation: pd.DataFrame) -> pd.DataFrame:
    """Return compact model summaries and gains versus chi-square-only."""
    if validation.empty:
        return pd.DataFrame()
    valid = validation[
        validation["model_status"].astype(str).str.startswith("diagnostic_holdout")
    ].copy()
    rows: list[dict[str, object]] = []
    baseline = valid[valid["model_id"].eq("chi_square_only")]
    baseline_auc = (
        float(baseline["holdout_signal_auc"].median())
        if not baseline.empty
        else math.nan
    )
    for model_id, group in validation.groupby("model_id", sort=False):
        model_valid = group[
            group["model_status"].astype(str).str.startswith("diagnostic_holdout")
        ]
        median_auc = (
            float(model_valid["holdout_signal_auc"].median())
            if not model_valid.empty
            else math.nan
        )
        rows.append(
            {
                "model_id": model_id,
                "n_valid_splits": int(model_valid.shape[0]),
                "median_signal_auc": median_auc,
                "best_signal_auc": float(model_valid["holdout_signal_auc"].max())
                if not model_valid.empty
                else math.nan,
                "median_auc_gain_vs_chi_square_only": (
                    median_auc - baseline_auc
                    if math.isfinite(median_auc) and math.isfinite(baseline_auc)
                    else math.nan
                ),
                "median_score_signal_minus_null_mean": float(
                    model_valid["score_signal_minus_null_mean"].median()
                )
                if not model_valid.empty
                else math.nan,
                "median_null_false_split_rate": float(
                    model_valid["null_false_split_rate_at_sibling_alpha"].median()
                )
                if not model_valid.empty
                else math.nan,
                "median_signal_retention_rate": float(
                    model_valid["signal_retention_rate_at_sibling_alpha"].median()
                )
                if not model_valid.empty
                else math.nan,
                "study_role": STUDY_ROLE,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["median_signal_auc", "median_auc_gain_vs_chi_square_only"],
        ascending=[False, False],
    )


def summarize_labeled_calibration(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed split/rejection behavior by truth label and geometry bins."""
    if panel.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    groupings: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("truth_context", ("truth_context_label",)),
        ("feature_family_truth_context", ("feature_family", "truth_context_label")),
        ("depth_truth_context", ("depth_bin", "truth_context_label")),
        ("parent_size_truth_context", ("parent_size_bin", "truth_context_label")),
    )
    for grouping_id, columns in groupings:
        for keys, group in panel.groupby(list(columns), dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = {
                "grouping_id": grouping_id,
                "n_contexts": int(group.shape[0]),
                "split_rejection_rate_at_sibling_alpha": float(
                    group["split_rejected_at_sibling_alpha"].astype(bool).mean()
                ),
                "raw_p_rejection_rate_at_alpha": float(
                    group["raw_sibling_rejected_at_alpha"].astype(bool).mean()
                ),
                "median_sibling_raw_neglog10_p": _finite_median(
                    group,
                    "sibling_raw_neglog10_p",
                ),
                "median_edge_action": _finite_median(group, "edge_action"),
                "median_barycentric_balance": _finite_median(
                    group,
                    "barycentric_balance",
                ),
                "study_role": STUDY_ROLE,
            }
            for column, key in zip(columns, keys, strict=True):
                record[column] = key
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def _run_case(
    *,
    case: dict[str, object],
    edge_alpha: float,
    sibling_alpha: float,
    spectral_minimum_dimension: int,
    passthrough: bool,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = prepare_case_inputs(case, ["kl"])
    params = dict(METHOD_SPECS["kl"].param_grid[0])
    params["spectral_minimum_dimension"] = int(spectral_minimum_dimension)
    params["passthrough"] = bool(passthrough)
    distance_condensed = (
        inputs.distance_condensed
        if bool(inputs.metadata.get("requires_precomputed_kl_distance"))
        else None
    )
    result = run_clustering_result(
        data_df=inputs.data,
        method_id="kl",
        params=params,
        seed=case["seed"],
        significance_level=sibling_alpha,
        edge_alpha=edge_alpha,
        distance_matrix=inputs.distance_matrix,
        distance_condensed=distance_condensed,
        feature_space=inputs.metadata.get("feature_space"),
    )
    feature_family = _feature_family(case, inputs.metadata)
    status = {
        "case_id": str(case["name"]),
        "feature_family": feature_family,
        "status": result.status,
        "skip_reason": result.skip_reason or "",
        "found_clusters": int(result.found_clusters),
        "study_role": STUDY_ROLE,
    }
    if result.status != "ok" or result.extra is None:
        return status, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    gate_bundle = result.extra.get("gate_bundle")
    if gate_bundle is None:
        status["status"] = "skip"
        status["skip_reason"] = "missing_gate_bundle"
        return status, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    edge_panel, node_panel = build_recursive_pvalue_geometry_panels(
        case_id=str(case["name"]),
        tree=result.extra["tree"],
        annotations=result.extra["annotations"],
        spectral_context=gate_bundle.edge_gate_result.spectral_context,
        edge_alpha=edge_alpha,
        sibling_alpha=sibling_alpha,
    )
    truth_labels = label_tree_sibling_contexts(
        case=case,
        tree=result.extra["tree"],
        true_labels=inputs.labels,
        sample_index=inputs.data.index,
        feature_family=feature_family,
    )
    kak_geometry = build_row_aligned_kak_geometry_panel(
        case_id=str(case["name"]),
        tree=result.extra["tree"],
        leaf_data=inputs.data,
        feature_space=inputs.metadata.get("feature_space"),
        spectral_context=gate_bundle.edge_gate_result.spectral_context,
    )
    return status, edge_panel, node_panel, truth_labels, kak_geometry


def _write_report(
    *,
    output_dir: Path,
    panel: pd.DataFrame,
    model_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    case_status: pd.DataFrame,
) -> Path:
    report_path = output_dir / "mixed_null_signal_geometry_report.md"
    ok_cases = int(case_status["status"].eq("ok").sum()) if not case_status.empty else 0
    lines = [
        "# Mixed Null/Signal Geometry Validation",
        "",
        f"- study_role: `{STUDY_ROLE}`",
        f"- ok_cases: `{ok_cases}`",
        f"- labeled_contexts: `{len(panel)}`",
        "",
    ]
    if not panel.empty:
        counts = panel["truth_context_label"].value_counts().sort_index()
        lines.append("## Truth Contexts")
        lines.append("")
        for label, count in counts.items():
            lines.append(f"- `{label}`: `{int(count)}`")
        lines.append("")
    if not model_summary.empty:
        best = model_summary.iloc[0]
        lines.extend(
            [
                "## Model Summary",
                "",
                f"- best_model: `{best.model_id}`",
                f"- median_signal_auc: `{best.median_signal_auc:.6f}`",
                f"- median_auc_gain_vs_chi_square_only: `{best.median_auc_gain_vs_chi_square_only:.6f}`",
                "",
            ]
        )
        for row in model_summary.itertuples(index=False):
            auc = row.median_signal_auc
            gain = row.median_auc_gain_vs_chi_square_only
            auc_text = f"{auc:.6f}" if isinstance(auc, float) and math.isfinite(auc) else str(auc)
            gain_text = f"{gain:.6f}" if isinstance(gain, float) and math.isfinite(gain) else str(gain)
            lines.append(f"- `{row.model_id}`: median AUC `{auc_text}`, gain `{gain_text}`")
        lines.append("")
    truth_rows = (
        calibration_summary[calibration_summary["grouping_id"].eq("truth_context")]
        if not calibration_summary.empty
        else pd.DataFrame()
    )
    if not truth_rows.empty:
        lines.extend(["## Calibration Readout", ""])
        for row in truth_rows.itertuples(index=False):
            label = getattr(row, "truth_context_label")
            lines.append(
                f"- `{label}`: split rejection `{row.split_rejection_rate_at_sibling_alpha:.6f}`, "
                f"raw-p rejection `{row.raw_p_rejection_rate_at_alpha:.6f}`, "
                f"n `{int(row.n_contexts)}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            (
                "The panel validates coordinates against benchmark truth labels. "
                "It is not a production selected-tail calibration because the rows "
                "are selected-tree contexts and do not provide a complete external "
                "null law."
            ),
            (
                "KAK radius/angle/action terms are row-aligned to KL sibling nodes "
                "using the root selected PCA frame in null-whitened tangent "
                "coordinates. These terms are still diagnostic covariates, not a "
                "production traversal or calibration rule."
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_mixed_null_signal_geometry_validation(
    *,
    output_dir: Path,
    case_suite: str = "method_proof",
    max_cases: int | None = None,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    spectral_minimum_dimension: int = 1,
    passthrough: bool = config.PASSTHROUGH,
    min_train_rows_per_predictor: int = 5,
    min_test_rows: int = 3,
) -> MixedGeometryOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = get_test_cases_by_suite(case_suite)
    if max_cases is not None:
        cases = cases[: int(max_cases)]

    edge_tables: list[pd.DataFrame] = []
    node_tables: list[pd.DataFrame] = []
    truth_tables: list[pd.DataFrame] = []
    kak_tables: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_copy = dict(case)
        case_copy["test_case_num"] = index
        status, edge_panel, node_panel, truth_labels, kak_geometry = _run_case(
            case=case_copy,
            edge_alpha=edge_alpha,
            sibling_alpha=sibling_alpha,
            spectral_minimum_dimension=spectral_minimum_dimension,
            passthrough=passthrough,
        )
        status_rows.append(status)
        if not edge_panel.empty:
            edge_tables.append(edge_panel)
        if not node_panel.empty:
            node_tables.append(node_panel)
        if not truth_labels.empty:
            truth_tables.append(truth_labels)
        if not kak_geometry.empty:
            kak_tables.append(kak_geometry)

    edge_panel = pd.concat(edge_tables, ignore_index=True) if edge_tables else pd.DataFrame()
    node_panel = pd.concat(node_tables, ignore_index=True) if node_tables else pd.DataFrame()
    truth_labels = pd.concat(truth_tables, ignore_index=True) if truth_tables else pd.DataFrame()
    kak_geometry = pd.concat(kak_tables, ignore_index=True) if kak_tables else pd.DataFrame()
    case_status = pd.DataFrame.from_records(status_rows)
    labeled_panel = build_mixed_null_signal_geometry_panel(
        node_panel=node_panel,
        edge_panel=edge_panel,
        truth_labels=truth_labels,
        kak_geometry=kak_geometry,
        sibling_alpha=sibling_alpha,
    )
    model_validation = evaluate_geometry_models(
        labeled_panel,
        min_train_rows_per_predictor=min_train_rows_per_predictor,
        min_test_rows=min_test_rows,
    )
    model_summary = summarize_geometry_models(model_validation)
    calibration_summary = summarize_labeled_calibration(labeled_panel)

    labeled_nodes_csv = output_dir / "mixed_null_signal_labeled_nodes.csv"
    edge_panel_csv = output_dir / "mixed_null_signal_edges.csv"
    model_validation_csv = output_dir / "mixed_null_signal_model_validation.csv"
    model_summary_csv = output_dir / "mixed_null_signal_model_summary.csv"
    calibration_summary_csv = output_dir / "mixed_null_signal_calibration_summary.csv"
    case_status_csv = output_dir / "mixed_null_signal_case_status.csv"
    labeled_panel.to_csv(labeled_nodes_csv, index=False)
    edge_panel.to_csv(edge_panel_csv, index=False)
    model_validation.to_csv(model_validation_csv, index=False)
    model_summary.to_csv(model_summary_csv, index=False)
    calibration_summary.to_csv(calibration_summary_csv, index=False)
    case_status.to_csv(case_status_csv, index=False)
    report_md = _write_report(
        output_dir=output_dir,
        panel=labeled_panel,
        model_summary=model_summary,
        calibration_summary=calibration_summary,
        case_status=case_status,
    )
    manifest_json = output_dir / "manifest.json"
    manifest = {
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "case_suite": case_suite,
        "max_cases": max_cases,
        "edge_alpha": float(edge_alpha),
        "sibling_alpha": float(sibling_alpha),
        "spectral_minimum_dimension": int(spectral_minimum_dimension),
        "passthrough": bool(passthrough),
        "min_train_rows_per_predictor": int(min_train_rows_per_predictor),
        "min_test_rows": int(min_test_rows),
        "outputs": {
            "labeled_nodes": str(labeled_nodes_csv),
            "edges": str(edge_panel_csv),
            "model_validation": str(model_validation_csv),
            "model_summary": str(model_summary_csv),
            "calibration_summary": str(calibration_summary_csv),
            "case_status": str(case_status_csv),
            "report": str(report_md),
        },
        "interpretation": (
            "Diagnostic mixed null/signal validation of recursive geometry. "
            "No production calibration or traversal rule is promoted."
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return MixedGeometryOutputs(
        labeled_nodes_csv=labeled_nodes_csv,
        edge_panel_csv=edge_panel_csv,
        model_validation_csv=model_validation_csv,
        model_summary_csv=model_summary_csv,
        calibration_summary_csv=calibration_summary_csv,
        case_status_csv=case_status_csv,
        report_md=report_md,
        manifest_json=manifest_json,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-suite", default="method_proof")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--edge-alpha", type=float, default=DEFAULT_EDGE_ALPHA)
    parser.add_argument("--sibling-alpha", type=float, default=DEFAULT_SIBLING_ALPHA)
    parser.add_argument("--spectral-minimum-dimension", type=int, default=1)
    parser.add_argument("--passthrough", action=argparse.BooleanOptionalAction, default=config.PASSTHROUGH)
    parser.add_argument("--min-train-rows-per-predictor", type=int, default=5)
    parser.add_argument("--min-test-rows", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_mixed_null_signal_geometry_validation(
        output_dir=args.output_dir,
        case_suite=str(args.case_suite),
        max_cases=args.max_cases,
        edge_alpha=float(args.edge_alpha),
        sibling_alpha=float(args.sibling_alpha),
        spectral_minimum_dimension=int(args.spectral_minimum_dimension),
        passthrough=bool(args.passthrough),
        min_train_rows_per_predictor=int(args.min_train_rows_per_predictor),
        min_test_rows=int(args.min_test_rows),
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "MixedGeometryOutputs",
    "build_mixed_null_signal_geometry_panel",
    "classify_sibling_truth_context",
    "build_row_aligned_kak_geometry_panel",
    "evaluate_geometry_models",
    "geometry_model_specs",
    "label_tree_sibling_contexts",
    "prepare_model_table",
    "run_mixed_null_signal_geometry_validation",
    "summarize_geometry_models",
    "summarize_labeled_calibration",
]
