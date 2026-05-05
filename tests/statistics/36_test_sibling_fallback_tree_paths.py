from __future__ import annotations

import numpy as np
import pandas as pd
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _build_cherry_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    tree.add_node(
        "root",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="root",
        leaf_count=200,
    )
    tree.add_node(
        "A",
        is_leaf=True,
        distribution=np.array([0.12, 0.12, 0.12, 0.88, 0.88, 0.88], dtype=float),
        label="A",
        leaf_count=100,
    )
    tree.add_node(
        "B",
        is_leaf=True,
        distribution=np.array([0.88, 0.88, 0.88, 0.12, 0.12, 0.12], dtype=float),
        label="B",
        leaf_count=100,
    )
    tree.add_edge("root", "A", branch_length=0.25)
    tree.add_edge("root", "B", branch_length=0.20)

    annotations_df = pd.DataFrame(
        {
            "leaf_count": {
                "root": 200,
                "A": 100,
                "B": 100,
            }
        }
    )
    leaf_data = pd.DataFrame(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ],
        index=["A", "B"],
        dtype=float,
    )
    return tree, annotations_df, leaf_data


def _build_mixed_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    tree.add_node(
        "root",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="root",
        leaf_count=300,
    )
    tree.add_node(
        "I",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="I",
        leaf_count=200,
    )
    tree.add_node(
        "L1",
        is_leaf=True,
        distribution=np.array([0.10, 0.20, 0.20, 0.80, 0.90, 0.80], dtype=float),
        label="L1",
        leaf_count=100,
    )
    tree.add_node(
        "L2",
        is_leaf=True,
        distribution=np.array([0.90, 0.80, 0.80, 0.20, 0.10, 0.20], dtype=float),
        label="L2",
        leaf_count=100,
    )
    tree.add_node(
        "L3",
        is_leaf=True,
        distribution=np.array([0.80, 0.80, 0.80, 0.20, 0.20, 0.20], dtype=float),
        label="L3",
        leaf_count=100,
    )
    tree.add_edge("root", "I")
    tree.add_edge("root", "L3")
    tree.add_edge("I", "L1")
    tree.add_edge("I", "L2")

    annotations_df = pd.DataFrame(
        {
            "leaf_count": {
                "root": 300,
                "I": 200,
                "L1": 100,
                "L2": 100,
                "L3": 100,
            }
        }
    )
    leaf_data = pd.DataFrame(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        index=["L1", "L2", "L3"],
        dtype=float,
    )
    return tree, annotations_df, leaf_data


def test_cherry_with_leaf_data_omits_leaf_pair_parent_from_edge_derived_sibling_projection_dimensions() -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()

    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)
    out = bundle.annotated_df
    assert bundle.gate_two_result is not None
    spectral_projection_dimensions_by_node = (
        bundle.gate_two_result.spectral_context.spectral_projection_dimensions_by_node
    )
    assert spectral_projection_dimensions_by_node is not None
    assert spectral_projection_dimensions_by_node["A"] == 0
    assert spectral_projection_dimensions_by_node["B"] == 0
    assert spectral_projection_dimensions_by_node["root"] > 0

    sibling_projection_dimensions_from_edge_comparisons = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=bundle.gate_two_result.spectral_context,
        )
    )
    assert sibling_projection_dimensions_from_edge_comparisons is None

    assert np.isfinite(out.loc["root", "Sibling_Degrees_of_Freedom"])
    assert np.isfinite(out.loc["root", "Sibling_Divergence_P_Value"])


def test_mixed_parent_with_leaf_data_keeps_internal_parent_in_edge_derived_sibling_projection_dimensions() -> None:
    tree, annotations_df, leaf_data = _build_mixed_tree()

    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)

    assert bundle.gate_two_result is not None
    spectral_projection_dimensions_by_node = (
        bundle.gate_two_result.spectral_context.spectral_projection_dimensions_by_node
    )
    assert spectral_projection_dimensions_by_node is not None
    assert spectral_projection_dimensions_by_node["L1"] == 0
    assert spectral_projection_dimensions_by_node["L2"] == 0
    assert spectral_projection_dimensions_by_node["L3"] == 0
    assert spectral_projection_dimensions_by_node["I"] > 0
    assert spectral_projection_dimensions_by_node["root"] > 0

    sibling_projection_dimensions_from_edge_comparisons = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=bundle.gate_two_result.spectral_context,
        )
    )
    assert sibling_projection_dimensions_from_edge_comparisons is not None
    assert set(sibling_projection_dimensions_from_edge_comparisons) == {"root"}
    assert sibling_projection_dimensions_from_edge_comparisons["root"] > 0
    assert "I" not in sibling_projection_dimensions_from_edge_comparisons

    legacy_projection_dimensions = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            bundle.annotated_df,
        )
    )
    assert legacy_projection_dimensions == sibling_projection_dimensions_from_edge_comparisons

    legacy_parent_projections, legacy_parent_eigenvalues = (
        collect_parent_principal_component_inputs_for_sibling_tests(
            bundle.annotated_df,
            legacy_projection_dimensions,
        )
    )
    assert legacy_parent_projections is not None
    assert set(legacy_parent_projections) == {"root"}
    assert legacy_parent_eigenvalues is not None


def test_decompose_without_leaf_data_disables_spectral_metadata_and_merges() -> None:
    tree, annotations_df, _leaf_data = _build_cherry_tree()

    result = tree.decompose(annotations_df=annotations_df.copy(), leaf_data=None)

    assert result["num_clusters"] == 1
    assert tree.annotations_df is not None
    assert "_spectral_dims" not in tree.annotations_df.attrs
    assert np.isnan(tree.annotations_df.loc["root", "Sibling_Degrees_of_Freedom"])
    assert np.isnan(tree.annotations_df.loc["root", "Sibling_Divergence_P_Value"])
