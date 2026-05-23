from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pandas.testing as pdt
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import (
    Gate2Result,
    GateAnnotationBundle,
    SpectralContext,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.column_contracts import (
    validate_edge_gate_columns,
    validate_sibling_gate_columns,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence_with_context,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.adjusted_wald_annotation.pipeline import (
    annotate_sibling_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.parent_principal_component_inputs import (
    collect_parent_principal_component_inputs_for_sibling_tests,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.gate_inputs.projection_dimensions import (
    derive_sibling_projection_dimensions_from_child_edge_comparisons,
)


def _build_small_binary_tree() -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    tree = nx.DiGraph()
    tree.add_edge("root", "A", branch_length=0.25)
    tree.add_edge("root", "B", branch_length=0.20)
    tree.add_edge("cal", "C", branch_length=0.10)
    tree.add_edge("cal", "D", branch_length=0.10)

    root_dist = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64)
    a_dist = np.array([0.12, 0.12, 0.12, 0.88, 0.88, 0.88], dtype=np.float64)
    b_dist = np.array([0.88, 0.88, 0.88, 0.12, 0.12, 0.12], dtype=np.float64)
    cal_dist = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64)
    c_dist = np.array([0.45, 0.45, 0.45, 0.55, 0.55, 0.55], dtype=np.float64)
    d_dist = np.array([0.55, 0.55, 0.55, 0.45, 0.45, 0.45], dtype=np.float64)

    for node, dist, leaf_count, is_leaf in (
        ("root", root_dist, 200, False),
        ("A", a_dist, 100, True),
        ("B", b_dist, 100, True),
        ("cal", cal_dist, 200, False),
        ("C", c_dist, 100, True),
        ("D", d_dist, 100, True),
    ):
        tree.nodes[node]["distribution"] = dist
        tree.nodes[node]["leaf_count"] = leaf_count
        tree.nodes[node]["is_leaf"] = is_leaf
        tree.nodes[node]["label"] = node

    base_df = pd.DataFrame(
        {
            "leaf_count": {
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
        [a_dist, b_dist, c_dist, d_dist],
        index=["A", "B", "C", "D"],
        columns=[f"feature_{feature_index}" for feature_index in range(len(a_dist))],
    )
    return tree, base_df, leaf_data


def test_gate_annotation_pipeline_matches_sequential_gate_annotations(monkeypatch) -> None:
    tree, base_df, leaf_data = _build_small_binary_tree()

    edge_df, spectral_context = annotate_child_parent_divergence_with_context(
        tree,
        base_df.copy(),
        significance_level_alpha=0.01,
        leaf_data=leaf_data,
    )
    sibling_projection_dimensions = (
        derive_sibling_projection_dimensions_from_child_edge_comparisons(
            tree,
            spectral_context=spectral_context,
        )
    )
    (
        parent_principal_component_projections,
        parent_principal_component_eigenvalues,
    ) = collect_parent_principal_component_inputs_for_sibling_tests(
        sibling_projection_dimensions,
        spectral_context=spectral_context,
    )
    sequential_df = annotate_sibling_divergence(
        tree,
        edge_df,
        significance_level_alpha=0.01,
        sibling_projection_dimensions_from_edge_comparisons=sibling_projection_dimensions,
        parent_principal_component_projections=parent_principal_component_projections,
        parent_principal_component_eigenvalues=parent_principal_component_eigenvalues,
    )

    bundle = run_gate_annotation_pipeline(
        tree,
        base_df.copy(),
        alpha_local=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data,
    )

    assert isinstance(bundle, GateAnnotationBundle)
    assert isinstance(bundle.gate_two_result, Gate2Result)
    assert isinstance(bundle.gate_two_result.spectral_context, SpectralContext)
    pipeline_df = bundle.annotated_df

    sequential_gate_cols = [
        col
        for col in sequential_df.columns
        if col.startswith("Child_Parent_") or col.startswith("Sibling_")
    ]
    pipeline_gate_cols = [
        col
        for col in pipeline_df.columns
        if col.startswith("Child_Parent_") or col.startswith("Sibling_")
    ]

    assert pipeline_gate_cols == sequential_gate_cols
    pdt.assert_frame_equal(
        pipeline_df[sequential_gate_cols],
        sequential_df[sequential_gate_cols],
        check_dtype=False,
    )

    # Verify all expected gate columns are present in the annotated DataFrame
    expected_edge_cols = tuple(
        col for col in sequential_gate_cols if col.startswith("Child_Parent_")
    )
    expected_sibling_cols = tuple(col for col in sequential_gate_cols if col.startswith("Sibling_"))
    actual_edge_cols = tuple(col for col in pipeline_df.columns if col.startswith("Child_Parent_"))
    actual_sibling_cols = tuple(col for col in pipeline_df.columns if col.startswith("Sibling_"))
    assert actual_edge_cols == expected_edge_cols
    assert actual_sibling_cols == expected_sibling_cols
    assert validate_edge_gate_columns(pipeline_df) == expected_edge_cols
    assert validate_sibling_gate_columns(pipeline_df) == expected_sibling_cols
    assert "pipeline" in bundle.metadata
    assert "edge" in bundle.metadata
    assert "sibling" in bundle.metadata
