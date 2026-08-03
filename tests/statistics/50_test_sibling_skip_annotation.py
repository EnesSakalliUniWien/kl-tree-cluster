"""Contracts for sibling-divergence skipped-node annotations."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from tree_break_selection.hierarchy_analysis.statistics.sibling_divergence.inflated_projected_wald_annotation.pipeline import (
    annotate_sibling_divergence,
)


def _binary_tree() -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_edge("N4", "N2", branch_length=0.3)
    tree.add_edge("N4", "N3", branch_length=0.3)
    tree.add_edge("N2", "L0", branch_length=0.1)
    tree.add_edge("N2", "L1", branch_length=0.1)
    tree.add_edge("N3", "L2", branch_length=0.1)
    tree.add_edge("N3", "L3", branch_length=0.1)

    rng = np.random.default_rng(42)
    feature_count = 20
    for node in ["L0", "L1", "L2", "L3"]:
        tree.nodes[node]["distribution"] = rng.random(feature_count) * 0.5
        tree.nodes[node]["leaf_count"] = 1
        tree.nodes[node]["label"] = node
    for node in ["N2", "N3", "N4"]:
        tree.nodes[node]["distribution"] = rng.random(feature_count) * 0.5
        tree.nodes[node]["leaf_count"] = 4 if node == "N4" else 2
        tree.nodes[node]["label"] = node
    return tree


def _edge_annotations(tree: nx.DiGraph) -> pd.DataFrame:
    annotations = pd.DataFrame(index=list(tree.nodes))
    annotations["Child_Parent_Divergence_Significant"] = True
    annotations["Child_Parent_Divergence_P_Value_BH"] = 0.01
    annotations["Child_Parent_Divergence_P_Value"] = 0.01
    annotations["Child_Parent_Divergence_Test_Statistic"] = 5.0
    annotations["Child_Parent_Divergence_df"] = 1.0
    annotations["Child_Parent_Divergence_Invalid"] = False
    annotations["Child_Parent_Divergence_Tested"] = True
    annotations["Child_Parent_Divergence_Ancestor_Blocked"] = False
    annotations.loc[["L2", "L3"], "Child_Parent_Divergence_Significant"] = False
    annotations.loc[["L2", "L3"], "Child_Parent_Divergence_P_Value_BH"] = 1.0
    annotations.loc[["L2", "L3"], "Child_Parent_Divergence_P_Value"] = 1.0
    return annotations


def test_inflated_projected_wald_marks_leaves_as_skipped() -> None:
    tree = _binary_tree()
    annotations = _edge_annotations(tree)
    sibling_parent_ids = ["N4", "N2", "N3"]
    parent_pca = {parent: np.eye(20, dtype=float)[:1] for parent in sibling_parent_ids}

    result = annotate_sibling_divergence(
        tree,
        annotations,
        sibling_projection_dimensions_from_edge_comparisons={
            parent: 1 for parent in sibling_parent_ids
        },
        parent_principal_component_projections=parent_pca,
        parent_principal_component_eigenvalues={
            parent: np.ones(1, dtype=float) for parent in sibling_parent_ids
        },
    )

    for leaf in ["L0", "L1", "L2", "L3"]:
        assert bool(result.loc[leaf, "Sibling_Divergence_Skipped"])


def _annotate(tree: nx.DiGraph, annotations: pd.DataFrame) -> pd.DataFrame:
    sibling_parent_ids = ["N4", "N2", "N3"]
    return annotate_sibling_divergence(
        tree,
        annotations,
        sibling_projection_dimensions_from_edge_comparisons={
            parent: 1 for parent in sibling_parent_ids
        },
        parent_principal_component_projections={
            parent: np.eye(20, dtype=float)[:1] for parent in sibling_parent_ids
        },
        parent_principal_component_eigenvalues={
            parent: np.ones(1, dtype=float) for parent in sibling_parent_ids
        },
    )


def test_skipped_null_like_parent_retains_calibration_degrees_of_freedom() -> None:
    tree = _binary_tree()
    result = _annotate(tree, _edge_annotations(tree))

    assert bool(result.loc["N3", "Sibling_Divergence_Skipped"])
    assert float(result.loc["N3", "Sibling_Degrees_of_Freedom"]) == 1.0


def test_annotations_expose_role_support_for_every_sibling_record() -> None:
    tree = _binary_tree()
    result = _annotate(tree, _edge_annotations(tree))

    # N3's children L2/L3 are both non-significant, so N3 is null-like and is
    # admissible empirical-null calibration support. N2's children are both
    # significant, tested and unblocked, so it is focal and not support.
    assert bool(result.loc["N3", "Sibling_Role_Supported"])
    assert not bool(result.loc["N2", "Sibling_Role_Supported"])


def test_annotations_expose_sibling_null_weight_for_every_record() -> None:
    tree = _binary_tree()
    result = _annotate(tree, _edge_annotations(tree))

    # Calibration support needs all three of dof > 0, weight > 0 and role
    # support, so the weight must survive for skipped parents too.
    weight = float(result.loc["N3", "Sibling_Null_Weight"])
    assert 0.0 <= weight <= 1.0


def test_annotations_expose_parent_positive_eigenvalue_count() -> None:
    tree = _binary_tree()
    result = _annotate(tree, _edge_annotations(tree))

    # dof == 0 is reachable only when the parent spectrum has no positive
    # eigenvalues, so the available rank must be auditable next to the dof.
    # The fixture supplies a single unit eigenvalue per parent.
    assert float(result.loc["N3", "Sibling_Parent_Positive_Eigenvalue_Count"]) == 1.0
