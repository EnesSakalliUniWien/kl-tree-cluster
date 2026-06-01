from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence_with_context,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context import (
    SpectralContext,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _build_cherry_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    tree.add_node(
        "top",
        is_leaf=False,
        distribution=np.array([0.20, 0.20, 0.20, 0.80, 0.80, 0.80], dtype=float),
        label="top",
        leaf_count=400,
    )
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
    tree.add_node(
        "cal",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="cal",
        leaf_count=200,
    )
    tree.add_node(
        "C",
        is_leaf=True,
        distribution=np.array([0.45, 0.45, 0.45, 0.55, 0.55, 0.55], dtype=float),
        label="C",
        leaf_count=100,
    )
    tree.add_node(
        "D",
        is_leaf=True,
        distribution=np.array([0.55, 0.55, 0.55, 0.45, 0.45, 0.45], dtype=float),
        label="D",
        leaf_count=100,
    )
    tree.add_edge("top", "root", branch_length=0.10)
    tree.add_edge("top", "cal", branch_length=0.10)
    tree.add_edge("root", "A", branch_length=0.25)
    tree.add_edge("root", "B", branch_length=0.20)
    tree.add_edge("cal", "C", branch_length=0.10)
    tree.add_edge("cal", "D", branch_length=0.10)
    tree.graph["root"] = "top"

    annotations_df = pd.DataFrame(
        {
            "leaf_count": {
                "top": 400,
                "root": 200,
                "A": 100,
                "B": 100,
                "cal": 200,
                "C": 100,
                "D": 100,
            }
        }
    )
    leaf_data = pd.DataFrame(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [0.45, 0.45, 0.45, 0.55, 0.55, 0.55],
            [0.55, 0.55, 0.55, 0.45, 0.45, 0.45],
        ],
        index=["A", "B", "C", "D"],
        dtype=float,
    )
    return tree, annotations_df, leaf_data


def _build_mixed_tree() -> tuple[PosetTree, pd.DataFrame, pd.DataFrame]:
    tree = PosetTree()
    tree.add_node(
        "top",
        is_leaf=False,
        distribution=np.array([0.20, 0.20, 0.20, 0.80, 0.80, 0.80], dtype=float),
        label="top",
        leaf_count=500,
    )
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
    tree.add_node(
        "cal",
        is_leaf=False,
        distribution=np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=float),
        label="cal",
        leaf_count=200,
    )
    tree.add_node(
        "C",
        is_leaf=True,
        distribution=np.array([0.45, 0.45, 0.45, 0.55, 0.55, 0.55], dtype=float),
        label="C",
        leaf_count=100,
    )
    tree.add_node(
        "D",
        is_leaf=True,
        distribution=np.array([0.55, 0.55, 0.55, 0.45, 0.45, 0.45], dtype=float),
        label="D",
        leaf_count=100,
    )
    tree.add_edge("top", "root")
    tree.add_edge("top", "cal")
    tree.add_edge("root", "I")
    tree.add_edge("root", "L3")
    tree.add_edge("I", "L1")
    tree.add_edge("I", "L2")
    tree.add_edge("cal", "C")
    tree.add_edge("cal", "D")
    tree.graph["root"] = "top"

    annotations_df = pd.DataFrame(
        {
            "leaf_count": {
                "top": 500,
                "root": 300,
                "I": 200,
                "L1": 100,
                "L2": 100,
                "L3": 100,
                "cal": 200,
                "C": 100,
                "D": 100,
            }
        }
    )
    leaf_data = pd.DataFrame(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0.45, 0.45, 0.45, 0.55, 0.55, 0.55],
            [0.55, 0.55, 0.55, 0.45, 0.45, 0.45],
        ],
        index=["L1", "L2", "L3", "C", "D"],
        dtype=float,
    )
    return tree, annotations_df, leaf_data


def test_cherry_with_leaf_data_uses_parent_dimension_for_leaf_pair_parent() -> None:
    tree, annotations_df, leaf_data = _build_cherry_tree()

    _edge_df, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        annotations_df.copy(),
        leaf_data=leaf_data,
    )
    test_projection_dimensions_by_node = (
        spectral_context.test_projection_dimensions_by_node
    )
    assert test_projection_dimensions_by_node["A"] == 0
    assert test_projection_dimensions_by_node["B"] == 0
    assert test_projection_dimensions_by_node["C"] == 0
    assert test_projection_dimensions_by_node["D"] == 0
    assert test_projection_dimensions_by_node["root"] > 0
    assert test_projection_dimensions_by_node["cal"] > 0
    assert test_projection_dimensions_by_node["top"] > 0

    sibling_projection_dimensions_from_edge_comparisons = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=spectral_context,
        )
    )
    assert sibling_projection_dimensions_from_edge_comparisons == {
        "root": test_projection_dimensions_by_node["root"],
        "cal": test_projection_dimensions_by_node["cal"],
        "top": test_projection_dimensions_by_node["top"],
    }

    with pytest.raises(ValueError, match="selected non-null"):
        run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)


def test_mixed_parent_with_leaf_data_keeps_internal_parent_in_edge_derived_sibling_projection_dimensions() -> None:
    tree, annotations_df, leaf_data = _build_mixed_tree()

    bundle = run_gate_annotation_pipeline(tree, annotations_df.copy(), leaf_data=leaf_data)

    test_projection_dimensions_by_node = (
        bundle.edge_gate_result.spectral_context.test_projection_dimensions_by_node
    )
    assert test_projection_dimensions_by_node["L1"] == 0
    assert test_projection_dimensions_by_node["L2"] == 0
    assert test_projection_dimensions_by_node["L3"] == 0
    assert test_projection_dimensions_by_node["C"] == 0
    assert test_projection_dimensions_by_node["D"] == 0
    assert test_projection_dimensions_by_node["I"] > 0
    assert test_projection_dimensions_by_node["root"] > 0
    assert test_projection_dimensions_by_node["cal"] > 0
    assert test_projection_dimensions_by_node["top"] > 0

    sibling_projection_dimensions_from_edge_comparisons = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=bundle.edge_gate_result.spectral_context,
        )
    )
    assert set(sibling_projection_dimensions_from_edge_comparisons) == {
        "root",
        "I",
        "cal",
        "top",
    }
    assert sibling_projection_dimensions_from_edge_comparisons["root"] > 0
    assert sibling_projection_dimensions_from_edge_comparisons["I"] == (
        test_projection_dimensions_by_node["I"]
    )
    assert sibling_projection_dimensions_from_edge_comparisons["cal"] == (
        test_projection_dimensions_by_node["cal"]
    )
    assert 0 < sibling_projection_dimensions_from_edge_comparisons["top"] <= (
        test_projection_dimensions_by_node["top"]
    )


def test_decompose_without_leaf_data_raises_for_missing_spectral_contract() -> None:
    tree, annotations_df, _leaf_data = _build_cherry_tree()

    with pytest.raises(ValueError, match="require leaf_data"):
        tree.decompose(annotations_df=annotations_df.copy(), leaf_data=None)


def test_edge_derived_sibling_dimensions_require_all_child_spectral_dimensions() -> None:
    tree, _annotations_df, _leaf_data = _build_cherry_tree()
    spectral_context = SpectralContext(
        test_projection_dimensions_by_node={"A": 1},
        raw_mp_signal_counts_by_node={"A": 0},
        effective_independent_rows_by_node={"A": 1},
        mp_threshold_rows_by_node={"A": 1},
        principal_component_projections_by_node={},
        principal_component_eigenvalues_by_node={},
    )

    with pytest.raises(ValueError, match="missing 'root'"):
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=spectral_context,
        )
