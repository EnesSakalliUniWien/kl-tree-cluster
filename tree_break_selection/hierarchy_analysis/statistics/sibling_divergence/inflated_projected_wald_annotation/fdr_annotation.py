"""Traversal-aligned FDR annotation for sibling divergence results."""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from tree_break_selection.core_utils.data_utils import initialize_sibling_divergence_columns

from ...multiple_testing import benjamini_hochberg_correction


def _validate_sibling_results(
    parents: list[object],
    results: list[tuple[float, float, float]],
    method_labels: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(parents) != len(results):
        raise ValueError(
            "Sibling FDR parents and results must align one-to-one: "
            f"{len(parents)} parent(s), {len(results)} result(s)."
        )
    if len(set(parents)) != len(parents):
        raise ValueError("Sibling FDR parents must be unique.")
    if method_labels is not None and len(method_labels) != len(parents):
        raise ValueError(
            "Sibling FDR method labels must align one-to-one with parents: "
            f"{len(method_labels)} label(s), {len(parents)} parent(s)."
        )

    test_statistics = np.array([result[0] for result in results], dtype=float)
    degrees_of_freedom_values = np.array([result[1] for result in results], dtype=float)
    p_values = np.array([result[2] for result in results], dtype=float)

    if not np.isfinite(test_statistics).all():
        raise ValueError("Sibling test statistics must be finite before FDR correction.")
    if not np.isfinite(degrees_of_freedom_values).all() or np.any(degrees_of_freedom_values < 0):
        raise ValueError(
            "Sibling test degrees of freedom must be finite and non-negative before FDR correction."
        )
    if not np.isfinite(p_values).all() or np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("Sibling p-values must be finite values in [0, 1] before FDR correction.")

    return test_statistics, degrees_of_freedom_values, p_values


def _root_nodes(tree: nx.DiGraph) -> list[object]:
    roots = [node for node in tree.nodes if tree.in_degree(node) == 0]
    if not roots:
        raise ValueError("Sibling FDR requires at least one root node.")
    return roots


def _sibling_gate_calibration_label(method_label: str) -> str:
    supported_labels = {
        "fixed_global_chi_square": "fixed_subspace_chi_square",
        "fixed_coordinate_bh": "fixed_subspace_bh",
        "fixed_coordinate_by": "fixed_subspace_by",
        "fixed_coordinate_holm": "fixed_subspace_holm",
        "fixed_coordinate_bonferroni": "fixed_subspace_bonferroni",
        "fixed_block_bh": "fixed_subspace_bh",
        "fixed_block_simes_bh": "fixed_subspace_simes_bh",
        "global_chi_square": "data_independent_chi_square",
        "coordinate_bonferroni": "data_independent_coordinate_bonferroni",
        "coordinate_bh": "data_independent_coordinate_bh",
        "block_bonferroni": "data_independent_block_bonferroni",
        "block_bh": "data_independent_block_bh",
    }
    if method_label in supported_labels:
        return supported_labels[method_label]
    if method_label == "zero_dimensional_sibling_record":
        return "degenerate_zero_dimensional"
    if method_label.startswith("external_selected_tail"):
        return "selected_tail_calibrated"
    if "empirical_null_inflation" in method_label:
        return "empirical_null_inflation"
    raise ValueError(f"Unsupported sibling test method label: {method_label!r}.")


def _parent_passes_traversal_prerequisites(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    parent: object,
) -> bool:
    children = list(tree.successors(parent))
    if len(children) != 2:
        return False

    if "Child_Parent_Divergence_Significant" not in annotations_df.columns:
        raise ValueError(
            "Traversal-aligned sibling FDR requires 'Child_Parent_Divergence_Significant'."
        )

    missing_children = [child for child in children if child not in annotations_df.index]
    if missing_children:
        raise ValueError(
            "Traversal-aligned sibling FDR requires child-parent annotations for "
            f"children {missing_children!r}."
        )

    child_flags = annotations_df.loc[children, "Child_Parent_Divergence_Significant"]
    if child_flags.isna().any():
        raise ValueError(
            "Traversal-aligned sibling FDR requires non-null child-parent "
            f"significance values for children of {parent!r}."
        )
    return bool(child_flags.astype(bool).any())


def apply_traversal_aligned_sibling_bh_results(
    tree: nx.DiGraph,
    annotations_df: pd.DataFrame,
    parents: list[object],
    results: list[tuple[float, float, float]],
    alpha: float,
    *,
    method_labels: list[str] | None = None,
    skipped_parents: Sequence[object] | None = None,
) -> pd.DataFrame:
    """Apply traversal-aligned sibling BH over the edge-reachable frontier.

    The clustering traversal can only use a sibling decision at nodes whose
    structural prerequisite and edge-divergence gate are open. Therefore the sibling FDR
    family is built top-down by tree depth over that reachable frontier instead
    of over every internal node in the tree.
    """
    if skipped_parents:
        annotations_df.loc[skipped_parents, "Sibling_Divergence_Skipped"] = True

    if not results:
        return annotations_df

    test_statistics, degrees_of_freedom_values, p_values = _validate_sibling_results(
        parents,
        results,
        method_labels,
    )
    invalid_test_flags = np.zeros(len(results), dtype=bool)

    annotations_df.loc[parents, "Sibling_Test_Statistic"] = test_statistics
    annotations_df.loc[parents, "Sibling_Degrees_of_Freedom"] = degrees_of_freedom_values
    annotations_df.loc[parents, "Sibling_Divergence_P_Value"] = p_values
    annotations_df.loc[parents, "Sibling_Divergence_Invalid"] = invalid_test_flags

    if method_labels is not None:
        annotations_df.loc[parents, "Sibling_Test_Method"] = method_labels
        annotations_df.loc[parents, "Sibling_Gate_P_Value_Calibration"] = [
            _sibling_gate_calibration_label(label) for label in method_labels
        ]
        annotations_df.loc[parents, "Sibling_Gate_P_Value_Role"] = "active_traversal_sibling_gate"

    parent_index = {parent: index for index, parent in enumerate(parents)}
    frontier = _root_nodes(tree)

    while frontier:
        candidate_parents = [
            parent
            for parent in frontier
            if parent in parent_index
            and _parent_passes_traversal_prerequisites(tree, annotations_df, parent)
        ]
        if candidate_parents:
            candidate_indices = np.array(
                [parent_index[parent] for parent in candidate_parents],
                dtype=int,
            )
            reject, corrected_p_values, _ = benjamini_hochberg_correction(
                p_values[candidate_indices],
                alpha=alpha,
            )

            annotations_df.loc[candidate_parents, "Sibling_Divergence_P_Value_Corrected"] = (
                corrected_p_values
            )
            annotations_df.loc[candidate_parents, "Sibling_BH_Different"] = reject
            annotations_df.loc[candidate_parents, "Sibling_BH_Same"] = ~reject

        next_frontier: list[object] = []
        for parent in candidate_parents:
            next_frontier.extend(tree.successors(parent))
        frontier = next_frontier

    return annotations_df


def init_sibling_annotation_df(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and initialize the standard sibling-annotation columns."""
    if len(annotations_df) == 0:
        raise ValueError("Empty dataframe")
    return initialize_sibling_divergence_columns(annotations_df.copy())


def mark_non_binary_as_skipped(
    annotations_df: pd.DataFrame,
    non_binary_nodes: Sequence[object],
) -> None:
    """Mark non-binary or leaf nodes as skipped sibling tests."""
    if not non_binary_nodes:
        return
    annotations_df.loc[list(non_binary_nodes), "Sibling_Divergence_Skipped"] = True


__all__ = [
    "apply_traversal_aligned_sibling_bh_results",
    "init_sibling_annotation_df",
    "mark_non_binary_as_skipped",
]
