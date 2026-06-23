"""Recursive p-value and spectral-geometry diagnostic.

This diagnostic checks how edge p-values, sibling p-values, selected PCA
subspaces, eigenvalue gaps, and chi-square tail sensitivity vary along the
tree. It is descriptive only: it does not change edge, sibling, traversal, or
calibration behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2

from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.runners.dispatch import run_clustering_result
from benchmarks.shared.runners.method_registry import METHOD_SPECS
from benchmarks.shared.util.case_inputs import prepare_case_inputs
from benchmarks.shared.util.time import format_timestamp_utc
from tree_break_selection import config
from tree_break_selection.hierarchy_analysis.statistics.alpha_contract import (
    DEFAULT_EDGE_ALPHA,
    DEFAULT_SIBLING_ALPHA,
)

STUDY_ROLE = "diagnostic_recursive_pvalue_geometry_not_calibration"
P_VALUE_FLOOR = 1e-300


@dataclass(frozen=True)
class RecursivePValueGeometryOutputs:
    edge_panel_csv: Path
    node_panel_csv: Path
    summary_csv: Path
    case_status_csv: Path
    report_md: Path
    manifest_json: Path


def neg_log10_pvalue(value: object) -> float:
    p_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(p_value):
        return math.nan
    return float(-math.log10(max(float(p_value), P_VALUE_FLOOR)))


def alpha_margin(value: object, alpha: float) -> float:
    transformed = neg_log10_pvalue(value)
    if not math.isfinite(transformed):
        return math.nan
    return float(transformed - (-math.log10(float(alpha))))


def chi_square_tail_sensitivity(statistic: object, df: object) -> float:
    statistic_value = pd.to_numeric(pd.Series([statistic]), errors="coerce").iloc[0]
    df_value = pd.to_numeric(pd.Series([df]), errors="coerce").iloc[0]
    if not np.isfinite(statistic_value) or not np.isfinite(df_value) or df_value <= 0:
        return math.nan
    survival = float(chi2.sf(float(statistic_value), df=float(df_value)))
    density = float(chi2.pdf(float(statistic_value), df=float(df_value)))
    if survival <= 0.0:
        return math.inf
    return float(density / survival / math.log(10.0))


def chi_square_log10_residual(
    statistic: object,
    df: object,
    p_value: object,
) -> float:
    statistic_value = pd.to_numeric(pd.Series([statistic]), errors="coerce").iloc[0]
    df_value = pd.to_numeric(pd.Series([df]), errors="coerce").iloc[0]
    observed = pd.to_numeric(pd.Series([p_value]), errors="coerce").iloc[0]
    if not np.isfinite(statistic_value) or not np.isfinite(df_value) or not np.isfinite(observed):
        return math.nan
    if df_value == 0 and statistic_value == 0:
        expected = 1.0
    elif df_value <= 0:
        return math.nan
    else:
        expected = float(chi2.sf(float(statistic_value), df=float(df_value)))
    return float(
        abs(
            -math.log10(max(float(observed), P_VALUE_FLOOR))
            + math.log10(max(expected, P_VALUE_FLOOR))
        )
    )


def eigenvalue_geometry(eigenvalues: np.ndarray | None) -> dict[str, float]:
    if eigenvalues is None:
        return {
            "selected_eigenvalue_count": 0.0,
            "top_selected_eigenvalue": math.nan,
            "second_selected_eigenvalue": math.nan,
            "selected_eigenvalue_gap_ratio": math.nan,
            "top_selected_eigenvalue_mass_fraction": math.nan,
            "selected_eigenvalue_effective_rank": math.nan,
        }
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return {
            "selected_eigenvalue_count": 0.0,
            "top_selected_eigenvalue": math.nan,
            "second_selected_eigenvalue": math.nan,
            "selected_eigenvalue_gap_ratio": math.nan,
            "top_selected_eigenvalue_mass_fraction": math.nan,
            "selected_eigenvalue_effective_rank": math.nan,
        }
    total = float(values.sum())
    weights = values / total
    entropy = float(-np.sum(weights * np.log(weights)))
    top = float(values[0])
    second = float(values[1]) if values.size > 1 else math.nan
    return {
        "selected_eigenvalue_count": float(values.size),
        "top_selected_eigenvalue": top,
        "second_selected_eigenvalue": second,
        "selected_eigenvalue_gap_ratio": float(top / second)
        if math.isfinite(second) and second > 0.0
        else math.inf,
        "top_selected_eigenvalue_mass_fraction": float(weights[0]),
        "selected_eigenvalue_effective_rank": float(math.exp(entropy)),
    }


def principal_subspace_alignment(
    parent_projection: np.ndarray | None,
    child_projection: np.ndarray | None,
) -> dict[str, float]:
    """Return sign-invariant parent-child PCA subspace alignment metrics."""
    if parent_projection is None or child_projection is None:
        return {
            "subspace_common_dimension": 0.0,
            "subspace_largest_cosine": math.nan,
            "subspace_mean_squared_cosine": math.nan,
            "subspace_chordal_distance_normalized": math.nan,
        }
    parent = np.asarray(parent_projection, dtype=float)
    child = np.asarray(child_projection, dtype=float)
    if parent.ndim != 2 or child.ndim != 2 or parent.size == 0 or child.size == 0:
        return {
            "subspace_common_dimension": 0.0,
            "subspace_largest_cosine": math.nan,
            "subspace_mean_squared_cosine": math.nan,
            "subspace_chordal_distance_normalized": math.nan,
        }
    common_dimension = min(parent.shape[0], child.shape[0])
    if common_dimension <= 0:
        return {
            "subspace_common_dimension": 0.0,
            "subspace_largest_cosine": math.nan,
            "subspace_mean_squared_cosine": math.nan,
            "subspace_chordal_distance_normalized": math.nan,
        }
    singular_values = np.linalg.svd(parent[:common_dimension] @ child[:common_dimension].T, compute_uv=False)
    clipped = np.clip(singular_values[:common_dimension], 0.0, 1.0)
    sum_sq = float(np.sum(clipped**2))
    return {
        "subspace_common_dimension": float(common_dimension),
        "subspace_largest_cosine": float(clipped[0]) if clipped.size else math.nan,
        "subspace_mean_squared_cosine": float(sum_sq / common_dimension),
        "subspace_chordal_distance_normalized": float(
            math.sqrt(max(common_dimension - sum_sq, 0.0) / common_dimension)
        ),
    }


def _root(tree: nx.DiGraph) -> object:
    if hasattr(tree, "root"):
        return tree.root()
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, got {roots!r}.")
    return roots[0]


def _depths(tree: nx.DiGraph) -> dict[object, int]:
    return {node: int(depth) for node, depth in nx.single_source_shortest_path_length(tree, _root(tree)).items()}


def _lookup(annotations: pd.DataFrame, node: object, column: str) -> object:
    if column not in annotations.columns or node not in annotations.index:
        return np.nan
    return annotations.loc[node, column]


def _finite_corr(table: pd.DataFrame, left: str, right: str) -> float:
    if left not in table.columns or right not in table.columns or table.empty:
        return math.nan
    pair = table[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if pair.shape[0] < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return math.nan
    return float(pair[left].corr(pair[right], method="spearman"))


def _finite_pair_count(table: pd.DataFrame, left: str, right: str) -> int:
    if left not in table.columns or right not in table.columns or table.empty:
        return 0
    return int(table[[left, right]].replace([np.inf, -np.inf], np.nan).dropna().shape[0])


def build_recursive_pvalue_geometry_panels(
    *,
    case_id: str,
    tree: nx.DiGraph,
    annotations: pd.DataFrame,
    spectral_context: Any,
    edge_alpha: float,
    sibling_alpha: float,
    ari: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    depths = _depths(tree)
    projections = spectral_context.principal_component_projections_by_node
    eigenvalues_by_node = spectral_context.principal_component_eigenvalues_by_node
    dimensions = spectral_context.test_projection_dimensions_by_node
    raw_mp_counts = spectral_context.raw_mp_signal_counts_by_node

    edge_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    for node in tree.nodes:
        children = list(tree.successors(node))
        parent_list = list(tree.predecessors(node))
        parent = parent_list[0] if parent_list else None
        sibling_raw_p = _lookup(annotations, node, "Sibling_Divergence_P_Value")
        sibling_bh_p = _lookup(annotations, node, "Sibling_Divergence_P_Value_Corrected")
        sibling_stat = _lookup(annotations, node, "Sibling_Test_Statistic")
        sibling_df = _lookup(annotations, node, "Sibling_Degrees_of_Freedom")
        child_edge_bh_values = [
            _lookup(annotations, child, "Child_Parent_Divergence_P_Value_BH")
            for child in children
        ]
        child_edge_bh_neglog = [neg_log10_pvalue(value) for value in child_edge_bh_values]
        finite_child_edge_bh = [
            value for value in child_edge_bh_neglog if math.isfinite(value)
        ]
        child_sibling_neglog = [
            neg_log10_pvalue(_lookup(annotations, child, "Sibling_Divergence_P_Value_Corrected"))
            for child in children
        ]
        finite_child_sibling = [
            value for value in child_sibling_neglog if math.isfinite(value)
        ]
        eig = eigenvalue_geometry(eigenvalues_by_node.get(str(node)))
        node_record = {
            "case_id": case_id,
            "node_id": str(node),
            "parent_id": "" if parent is None else str(parent),
            "depth": int(depths.get(node, -1)),
            "leaf_count": _lookup(annotations, node, "leaf_count"),
            "n_children": int(len(children)),
            "ari": np.nan if ari is None else float(ari),
            "test_projection_dimension": int(dimensions.get(str(node), 0)),
            "raw_mp_signal_count": int(raw_mp_counts.get(str(node), 0)),
            **eig,
            "sibling_raw_p_value": sibling_raw_p,
            "sibling_raw_neglog10_p": neg_log10_pvalue(sibling_raw_p),
            "sibling_raw_alpha_margin": alpha_margin(sibling_raw_p, sibling_alpha),
            "sibling_bh_p_value": sibling_bh_p,
            "sibling_neglog10_p": neg_log10_pvalue(sibling_bh_p),
            "sibling_alpha_margin": alpha_margin(sibling_bh_p, sibling_alpha),
            "sibling_test_statistic": sibling_stat,
            "sibling_degrees_of_freedom": sibling_df,
            "sibling_chi_square_log10_residual": chi_square_log10_residual(
                sibling_stat,
                sibling_df,
                sibling_raw_p,
            ),
            "sibling_chi_square_tail_sensitivity": chi_square_tail_sensitivity(
                sibling_stat,
                sibling_df,
            ),
            "min_child_edge_neglog10_bh_p": min(finite_child_edge_bh)
            if finite_child_edge_bh
            else math.nan,
            "max_child_edge_neglog10_bh_p": max(finite_child_edge_bh)
            if finite_child_edge_bh
            else math.nan,
            "max_child_sibling_neglog10_bh_p": max(finite_child_sibling)
            if finite_child_sibling
            else math.nan,
            "recursive_sibling_neglog10_gradient": (
                max(finite_child_sibling) - neg_log10_pvalue(sibling_bh_p)
                if finite_child_sibling and math.isfinite(neg_log10_pvalue(sibling_bh_p))
                else math.nan
            ),
            "edge_sibling_connectivity_score": (
                min(finite_child_edge_bh) + alpha_margin(sibling_bh_p, sibling_alpha)
                if finite_child_edge_bh and math.isfinite(alpha_margin(sibling_bh_p, sibling_alpha))
                else math.nan
            ),
            "sibling_bh_different": bool(_lookup(annotations, node, "Sibling_BH_Different"))
            if not pd.isna(_lookup(annotations, node, "Sibling_BH_Different"))
            else False,
            "sibling_divergence_skipped": bool(_lookup(annotations, node, "Sibling_Divergence_Skipped"))
            if not pd.isna(_lookup(annotations, node, "Sibling_Divergence_Skipped"))
            else False,
            "study_role": STUDY_ROLE,
        }
        node_rows.append(node_record)

        for child in children:
            child_sibling_raw_p = _lookup(annotations, child, "Sibling_Divergence_P_Value")
            child_sibling_p = _lookup(annotations, child, "Sibling_Divergence_P_Value_Corrected")
            edge_p = _lookup(annotations, child, "Child_Parent_Divergence_P_Value")
            edge_bh_p = _lookup(annotations, child, "Child_Parent_Divergence_P_Value_BH")
            edge_stat = _lookup(annotations, child, "Child_Parent_Divergence_Test_Statistic")
            edge_df = _lookup(annotations, child, "Child_Parent_Divergence_df")
            alignment = principal_subspace_alignment(
                projections.get(str(node)),
                projections.get(str(child)),
            )
            edge_rows.append(
                {
                    "case_id": case_id,
                    "parent_id": str(node),
                    "child_id": str(child),
                    "parent_depth": int(depths.get(node, -1)),
                    "child_depth": int(depths.get(child, -1)),
                    "parent_leaf_count": _lookup(annotations, node, "leaf_count"),
                    "child_leaf_count": _lookup(annotations, child, "leaf_count"),
                    "ari": np.nan if ari is None else float(ari),
                    "parent_sibling_raw_p_value": sibling_raw_p,
                    "parent_sibling_raw_neglog10_p": neg_log10_pvalue(sibling_raw_p),
                    "parent_sibling_raw_alpha_margin": alpha_margin(sibling_raw_p, sibling_alpha),
                    "parent_sibling_bh_p_value": sibling_bh_p,
                    "parent_sibling_neglog10_p": neg_log10_pvalue(sibling_bh_p),
                    "parent_sibling_alpha_margin": alpha_margin(sibling_bh_p, sibling_alpha),
                    "child_sibling_raw_p_value": child_sibling_raw_p,
                    "child_sibling_raw_neglog10_p": neg_log10_pvalue(child_sibling_raw_p),
                    "child_sibling_raw_alpha_margin": alpha_margin(child_sibling_raw_p, sibling_alpha),
                    "child_sibling_bh_p_value": child_sibling_p,
                    "child_sibling_neglog10_p": neg_log10_pvalue(child_sibling_p),
                    "edge_raw_p_value": edge_p,
                    "edge_raw_neglog10_p": neg_log10_pvalue(edge_p),
                    "edge_raw_alpha_margin": alpha_margin(edge_p, edge_alpha),
                    "edge_bh_p_value": edge_bh_p,
                    "edge_neglog10_bh_p": neg_log10_pvalue(edge_bh_p),
                    "edge_alpha_margin": alpha_margin(edge_bh_p, edge_alpha),
                    "edge_test_statistic": edge_stat,
                    "edge_degrees_of_freedom": edge_df,
                    "edge_chi_square_log10_residual": chi_square_log10_residual(
                        edge_stat,
                        edge_df,
                        edge_p,
                    ),
                    "edge_chi_square_tail_sensitivity": chi_square_tail_sensitivity(
                        edge_stat,
                        edge_df,
                    ),
                    "edge_raw_minus_parent_sibling_raw_neglog10": (
                        neg_log10_pvalue(edge_p) - neg_log10_pvalue(sibling_raw_p)
                        if math.isfinite(neg_log10_pvalue(edge_p))
                        and math.isfinite(neg_log10_pvalue(sibling_raw_p))
                        else math.nan
                    ),
                    "child_raw_minus_parent_sibling_raw_neglog10": (
                        neg_log10_pvalue(child_sibling_raw_p) - neg_log10_pvalue(sibling_raw_p)
                        if math.isfinite(neg_log10_pvalue(child_sibling_raw_p))
                        and math.isfinite(neg_log10_pvalue(sibling_raw_p))
                        else math.nan
                    ),
                    "edge_minus_parent_sibling_neglog10": (
                        neg_log10_pvalue(edge_bh_p) - neg_log10_pvalue(sibling_bh_p)
                        if math.isfinite(neg_log10_pvalue(edge_bh_p))
                        and math.isfinite(neg_log10_pvalue(sibling_bh_p))
                        else math.nan
                    ),
                    "child_minus_parent_sibling_neglog10": (
                        neg_log10_pvalue(child_sibling_p) - neg_log10_pvalue(sibling_bh_p)
                        if math.isfinite(neg_log10_pvalue(child_sibling_p))
                        and math.isfinite(neg_log10_pvalue(sibling_bh_p))
                        else math.nan
                    ),
                    "parent_test_projection_dimension": int(dimensions.get(str(node), 0)),
                    "child_test_projection_dimension": int(dimensions.get(str(child), 0)),
                    "parent_raw_mp_signal_count": int(raw_mp_counts.get(str(node), 0)),
                    "child_raw_mp_signal_count": int(raw_mp_counts.get(str(child), 0)),
                    **alignment,
                    "child_parent_edge_significant": bool(
                        _lookup(annotations, child, "Child_Parent_Divergence_Significant")
                    )
                    if not pd.isna(
                        _lookup(annotations, child, "Child_Parent_Divergence_Significant")
                    )
                    else False,
                    "study_role": STUDY_ROLE,
                }
            )

    return pd.DataFrame.from_records(edge_rows), pd.DataFrame.from_records(node_rows)


def summarize_recursive_pvalue_geometry(
    edge_panel: pd.DataFrame,
    node_panel: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "summary_id": "raw_edge_parent_sibling_pvalue_coupling",
            "scope": "global",
            "case_id": "",
            "metric": "spearman",
            "value": _finite_corr(
                edge_panel,
                "edge_raw_neglog10_p",
                "parent_sibling_raw_neglog10_p",
            ),
            "n_rows": _finite_pair_count(
                edge_panel,
                "edge_raw_neglog10_p",
                "parent_sibling_raw_neglog10_p",
            ),
            "interpretation": "Coupling before Tree-BH/BH sparsity between raw edge and raw parent sibling evidence.",
        },
        {
            "summary_id": "edge_parent_sibling_pvalue_coupling",
            "scope": "global",
            "case_id": "",
            "metric": "spearman",
            "value": _finite_corr(edge_panel, "edge_neglog10_bh_p", "parent_sibling_neglog10_p"),
            "n_rows": _finite_pair_count(edge_panel, "edge_neglog10_bh_p", "parent_sibling_neglog10_p"),
            "interpretation": "Coupling between edge evidence and the sibling test at the parent.",
        },
        {
            "summary_id": "raw_recursive_sibling_pvalue_continuity",
            "scope": "global",
            "case_id": "",
            "metric": "spearman",
            "value": _finite_corr(
                edge_panel,
                "parent_sibling_raw_neglog10_p",
                "child_sibling_raw_neglog10_p",
            ),
            "n_rows": _finite_pair_count(
                edge_panel,
                "parent_sibling_raw_neglog10_p",
                "child_sibling_raw_neglog10_p",
            ),
            "interpretation": "Parent-child sibling evidence continuity before traversal-aligned BH correction.",
        },
        {
            "summary_id": "recursive_sibling_pvalue_continuity",
            "scope": "global",
            "case_id": "",
            "metric": "spearman",
            "value": _finite_corr(edge_panel, "parent_sibling_neglog10_p", "child_sibling_neglog10_p"),
            "n_rows": _finite_pair_count(edge_panel, "parent_sibling_neglog10_p", "child_sibling_neglog10_p"),
            "interpretation": "Continuity of sibling evidence from a parent to child neighborhoods.",
        },
        {
            "summary_id": "raw_subspace_rotation_vs_edge_parent_sibling_gap",
            "scope": "global",
            "case_id": "",
            "metric": "spearman",
            "value": _finite_corr(
                edge_panel,
                "subspace_chordal_distance_normalized",
                "edge_raw_minus_parent_sibling_raw_neglog10",
            ),
            "n_rows": _finite_pair_count(
                edge_panel,
                "subspace_chordal_distance_normalized",
                "edge_raw_minus_parent_sibling_raw_neglog10",
            ),
            "interpretation": "Whether selected-subspace rotation tracks raw edge-vs-parent-sibling p-value gaps.",
        },
        {
            "summary_id": "subspace_rotation_vs_edge_parent_sibling_gap",
            "scope": "global",
            "case_id": "",
            "metric": "spearman",
            "value": _finite_corr(
                edge_panel,
                "subspace_chordal_distance_normalized",
                "edge_minus_parent_sibling_neglog10",
            ),
            "n_rows": _finite_pair_count(
                edge_panel,
                "subspace_chordal_distance_normalized",
                "edge_minus_parent_sibling_neglog10",
            ),
            "interpretation": "Whether PCA subspace rotation tracks edge-vs-parent-sibling p-value gaps.",
        },
        {
            "summary_id": "eigen_gap_vs_sibling_tail_sensitivity",
            "scope": "global",
            "case_id": "",
            "metric": "spearman",
            "value": _finite_corr(
                node_panel,
                "selected_eigenvalue_gap_ratio",
                "sibling_chi_square_tail_sensitivity",
            ),
            "n_rows": _finite_pair_count(
                node_panel,
                "selected_eigenvalue_gap_ratio",
                "sibling_chi_square_tail_sensitivity",
            ),
            "interpretation": "Whether sharp selected spectra coincide with more chi-square tail sensitivity.",
        },
        {
            "summary_id": "edge_chi_square_mechanical_residual_median",
            "scope": "global",
            "case_id": "",
            "metric": "median_abs_log10_p_residual",
            "value": float(edge_panel["edge_chi_square_log10_residual"].median())
            if "edge_chi_square_log10_residual" in edge_panel.columns
            else math.nan,
            "n_rows": int(edge_panel["edge_chi_square_log10_residual"].dropna().shape[0])
            if "edge_chi_square_log10_residual" in edge_panel.columns
            else 0,
            "interpretation": "Mechanical consistency of stored edge p-values with chi-square survival.",
        },
        {
            "summary_id": "sibling_chi_square_mechanical_residual_median",
            "scope": "global",
            "case_id": "",
            "metric": "median_abs_log10_p_residual",
            "value": float(node_panel["sibling_chi_square_log10_residual"].median())
            if "sibling_chi_square_log10_residual" in node_panel.columns
            else math.nan,
            "n_rows": int(node_panel["sibling_chi_square_log10_residual"].dropna().shape[0])
            if "sibling_chi_square_log10_residual" in node_panel.columns
            else 0,
            "interpretation": "Mechanical consistency of stored sibling raw p-values with chi-square survival.",
        },
    ]
    edge_groups = (
        edge_panel.groupby("case_id", sort=True)
        if "case_id" in edge_panel.columns
        else []
    )
    for case_id, case_edges in edge_groups:
        rows.extend(
            [
                {
                    "summary_id": "raw_edge_parent_sibling_pvalue_coupling",
                    "scope": "case",
                    "case_id": str(case_id),
                    "metric": "spearman",
                    "value": _finite_corr(
                        case_edges,
                        "edge_raw_neglog10_p",
                        "parent_sibling_raw_neglog10_p",
                    ),
                    "n_rows": _finite_pair_count(
                        case_edges,
                        "edge_raw_neglog10_p",
                        "parent_sibling_raw_neglog10_p",
                    ),
                    "interpretation": "Coupling before Tree-BH/BH sparsity between raw edge and raw parent sibling evidence.",
                },
                {
                    "summary_id": "edge_parent_sibling_pvalue_coupling",
                    "scope": "case",
                    "case_id": str(case_id),
                    "metric": "spearman",
                    "value": _finite_corr(
                        case_edges,
                        "edge_neglog10_bh_p",
                        "parent_sibling_neglog10_p",
                    ),
                    "n_rows": _finite_pair_count(
                        case_edges,
                        "edge_neglog10_bh_p",
                        "parent_sibling_neglog10_p",
                    ),
                    "interpretation": "Coupling between edge evidence and the sibling test at the parent.",
                },
                {
                    "summary_id": "raw_recursive_sibling_pvalue_continuity",
                    "scope": "case",
                    "case_id": str(case_id),
                    "metric": "spearman",
                    "value": _finite_corr(
                        case_edges,
                        "parent_sibling_raw_neglog10_p",
                        "child_sibling_raw_neglog10_p",
                    ),
                    "n_rows": _finite_pair_count(
                        case_edges,
                        "parent_sibling_raw_neglog10_p",
                        "child_sibling_raw_neglog10_p",
                    ),
                    "interpretation": "Parent-child sibling evidence continuity before traversal-aligned BH correction.",
                },
                {
                    "summary_id": "recursive_sibling_pvalue_continuity",
                    "scope": "case",
                    "case_id": str(case_id),
                    "metric": "spearman",
                    "value": _finite_corr(
                        case_edges,
                        "parent_sibling_neglog10_p",
                        "child_sibling_neglog10_p",
                    ),
                    "n_rows": _finite_pair_count(
                        case_edges,
                        "parent_sibling_neglog10_p",
                        "child_sibling_neglog10_p",
                    ),
                    "interpretation": "Continuity of sibling evidence from a parent to child neighborhoods.",
                },
                {
                    "summary_id": "raw_subspace_rotation_vs_edge_parent_sibling_gap",
                    "scope": "case",
                    "case_id": str(case_id),
                    "metric": "spearman",
                    "value": _finite_corr(
                        case_edges,
                        "subspace_chordal_distance_normalized",
                        "edge_raw_minus_parent_sibling_raw_neglog10",
                    ),
                    "n_rows": _finite_pair_count(
                        case_edges,
                        "subspace_chordal_distance_normalized",
                        "edge_raw_minus_parent_sibling_raw_neglog10",
                    ),
                    "interpretation": "Whether selected-subspace rotation tracks raw edge-vs-parent-sibling p-value gaps.",
                },
                {
                    "summary_id": "subspace_rotation_vs_edge_parent_sibling_gap",
                    "scope": "case",
                    "case_id": str(case_id),
                    "metric": "spearman",
                    "value": _finite_corr(
                        case_edges,
                        "subspace_chordal_distance_normalized",
                        "edge_minus_parent_sibling_neglog10",
                    ),
                    "n_rows": _finite_pair_count(
                        case_edges,
                        "subspace_chordal_distance_normalized",
                        "edge_minus_parent_sibling_neglog10",
                    ),
                    "interpretation": "Whether PCA subspace rotation tracks edge-vs-parent-sibling p-value gaps.",
                },
            ]
        )
    node_groups = (
        node_panel.groupby("case_id", sort=True)
        if "case_id" in node_panel.columns
        else []
    )
    for case_id, case_nodes in node_groups:
        rows.append(
            {
                "summary_id": "eigen_gap_vs_sibling_tail_sensitivity",
                "scope": "case",
                "case_id": str(case_id),
                "metric": "spearman",
                "value": _finite_corr(
                    case_nodes,
                    "selected_eigenvalue_gap_ratio",
                    "sibling_chi_square_tail_sensitivity",
                ),
                "n_rows": _finite_pair_count(
                    case_nodes,
                    "selected_eigenvalue_gap_ratio",
                    "sibling_chi_square_tail_sensitivity",
                ),
                "interpretation": "Whether sharp selected spectra coincide with more chi-square tail sensitivity.",
            }
        )
    table = pd.DataFrame.from_records(rows)
    table["study_role"] = STUDY_ROLE
    return table


def _run_case(
    *,
    case: dict[str, object],
    edge_alpha: float,
    sibling_alpha: float,
    spectral_minimum_dimension: int,
    passthrough: bool,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    inputs = prepare_case_inputs(case, ["tbs"])
    params = dict(METHOD_SPECS["tbs"].param_grid[0])
    params["spectral_minimum_dimension"] = int(spectral_minimum_dimension)
    params["passthrough"] = bool(passthrough)
    distance_condensed = (
        inputs.distance_condensed
        if bool(inputs.metadata.get("requires_precomputed_tbs_distance"))
        else None
    )
    result = run_clustering_result(
        data_df=inputs.data,
        method_id="tbs",
        params=params,
        seed=case["seed"],
        significance_level=sibling_alpha,
        edge_alpha=edge_alpha,
        distance_matrix=inputs.distance_matrix,
        distance_condensed=distance_condensed,
        feature_space=inputs.metadata.get("feature_space"),
    )
    status = {
        "case_id": str(case["name"]),
        "status": result.status,
        "skip_reason": result.skip_reason or "",
        "found_clusters": int(result.found_clusters),
        "study_role": STUDY_ROLE,
    }
    if result.status != "ok" or result.extra is None:
        return status, pd.DataFrame(), pd.DataFrame()
    gate_bundle = result.extra.get("gate_bundle")
    if gate_bundle is None:
        status["status"] = "skip"
        status["skip_reason"] = "missing_gate_bundle"
        return status, pd.DataFrame(), pd.DataFrame()
    return (
        status,
        *build_recursive_pvalue_geometry_panels(
            case_id=str(case["name"]),
            tree=result.extra["tree"],
            annotations=result.extra["annotations"],
            spectral_context=gate_bundle.edge_gate_result.spectral_context,
            edge_alpha=edge_alpha,
            sibling_alpha=sibling_alpha,
        ),
    )


def _write_report(
    output_dir: Path,
    *,
    summary: pd.DataFrame,
    case_status: pd.DataFrame,
    edge_panel: pd.DataFrame,
    node_panel: pd.DataFrame,
) -> Path:
    report_path = output_dir / "recursive_pvalue_geometry_report.md"
    ok_cases = int(case_status["status"].eq("ok").sum()) if not case_status.empty else 0
    lines = [
        "# Recursive P-Value Geometry Diagnostic",
        "",
        f"- study_role: `{STUDY_ROLE}`",
        f"- ok_cases: `{ok_cases}`",
        f"- edge_rows: `{len(edge_panel)}`",
        f"- node_rows: `{len(node_panel)}`",
        "",
        "## Summary",
        "",
    ]
    global_summary = summary[summary["scope"].eq("global")] if "scope" in summary else summary
    for row in global_summary.itertuples(index=False):
        value = row.value
        rendered = f"{value:.6f}" if isinstance(value, float) and math.isfinite(value) else str(value)
        lines.append(f"- `{row.summary_id}`: `{rendered}` ({row.interpretation})")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "These rows test coordinate relationships after the tree, edge path, "
                "projection dimension, and PCA basis have already been selected. "
                "Large correlations are diagnostic geometry, not production "
                "chi-square calibration."
            ),
            (
                "Subspace alignment uses principal angles and is therefore invariant "
                "to eigenvector sign flips. Individual selected eigenvectors are not "
                "stable objects when eigenvalue gaps are small; the selected subspace "
                "is the safer coordinate object."
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_recursive_pvalue_geometry_diagnostic(
    *,
    output_dir: Path,
    case_suite: str = "method_proof",
    max_cases: int | None = None,
    edge_alpha: float = DEFAULT_EDGE_ALPHA,
    sibling_alpha: float = DEFAULT_SIBLING_ALPHA,
    spectral_minimum_dimension: int = 1,
    passthrough: bool = config.PASSTHROUGH,
) -> RecursivePValueGeometryOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = get_test_cases_by_suite(case_suite)
    if max_cases is not None:
        cases = cases[: int(max_cases)]

    edge_tables: list[pd.DataFrame] = []
    node_tables: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_copy = dict(case)
        case_copy["test_case_num"] = index
        status, edge_panel, node_panel = _run_case(
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

    edge_panel = pd.concat(edge_tables, ignore_index=True) if edge_tables else pd.DataFrame()
    node_panel = pd.concat(node_tables, ignore_index=True) if node_tables else pd.DataFrame()
    case_status = pd.DataFrame.from_records(status_rows)
    summary = summarize_recursive_pvalue_geometry(edge_panel, node_panel)

    edge_panel_csv = output_dir / "recursive_pvalue_geometry_edges.csv"
    node_panel_csv = output_dir / "recursive_pvalue_geometry_nodes.csv"
    summary_csv = output_dir / "recursive_pvalue_geometry_summary.csv"
    case_status_csv = output_dir / "recursive_pvalue_geometry_case_status.csv"
    edge_panel.to_csv(edge_panel_csv, index=False)
    node_panel.to_csv(node_panel_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    case_status.to_csv(case_status_csv, index=False)
    report_md = _write_report(
        output_dir,
        summary=summary,
        case_status=case_status,
        edge_panel=edge_panel,
        node_panel=node_panel,
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
        "outputs": {
            "edge_panel": str(edge_panel_csv),
            "node_panel": str(node_panel_csv),
            "summary": str(summary_csv),
            "case_status": str(case_status_csv),
            "report": str(report_md),
        },
        "interpretation": (
            "Diagnostic-only recursive p-value geometry. These outputs do not "
            "promote a production calibration or traversal rule."
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return RecursivePValueGeometryOutputs(
        edge_panel_csv=edge_panel_csv,
        node_panel_csv=node_panel_csv,
        summary_csv=summary_csv,
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_recursive_pvalue_geometry_diagnostic(
        output_dir=args.output_dir,
        case_suite=str(args.case_suite),
        max_cases=args.max_cases,
        edge_alpha=float(args.edge_alpha),
        sibling_alpha=float(args.sibling_alpha),
        spectral_minimum_dimension=int(args.spectral_minimum_dimension),
        passthrough=bool(args.passthrough),
    )
    print(
        json.dumps(
            {key: str(value) for key, value in outputs.__dict__.items()},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RecursivePValueGeometryOutputs",
    "alpha_margin",
    "build_recursive_pvalue_geometry_panels",
    "chi_square_log10_residual",
    "chi_square_tail_sensitivity",
    "eigenvalue_geometry",
    "neg_log10_pvalue",
    "principal_subspace_alignment",
    "run_recursive_pvalue_geometry_diagnostic",
    "summarize_recursive_pvalue_geometry",
]
