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
