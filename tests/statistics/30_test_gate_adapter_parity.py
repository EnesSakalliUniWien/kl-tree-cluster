from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pandas.testing as pdt

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import (
    GateAnnotationBundle,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    annotate_sibling_divergence,
)


def _build_small_binary_tree() -> tuple[nx.DiGraph, pd.DataFrame]:
    tree = nx.DiGraph()
    tree.add_edge("root", "A", branch_length=0.25)
    tree.add_edge("root", "B", branch_length=0.20)

    root_dist = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64)
    a_dist = np.array([0.12, 0.12, 0.12, 0.88, 0.88, 0.88], dtype=np.float64)
    b_dist = np.array([0.88, 0.88, 0.88, 0.12, 0.12, 0.12], dtype=np.float64)

    for node, dist, leaf_count, is_leaf in (
        ("root", root_dist, 200, False),
        ("A", a_dist, 100, True),
        ("B", b_dist, 100, True),
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
            }
        }
    )
    return tree, base_df


def test_gate_adapter_pipeline_matches_sequential_gate_annotations(monkeypatch) -> None:
    tree, base_df = _build_small_binary_tree()

    monkeypatch.setattr(config, "SIBLING_TEST_METHOD", "cousin_adjusted_wald")

    edge_df = annotate_child_parent_divergence(
        tree,
        base_df.copy(),
        significance_level_alpha=0.01,
    )
    sequential_df = annotate_sibling_divergence(
        tree,
        edge_df,
        significance_level_alpha=0.01,
    )

    bundle = run_gate_annotation_pipeline(
        tree,
        base_df.copy(),
        alpha_local=0.01,
        sibling_alpha=0.01,
        sibling_method=config.SIBLING_TEST_METHOD,
    )

    assert isinstance(bundle, GateAnnotationBundle)
    adapter_df = bundle.annotated_df

    sequential_gate_cols = [
        col
        for col in sequential_df.columns
        if col.startswith("Child_Parent_") or col.startswith("Sibling_")
    ]
    adapter_gate_cols = [
        col
        for col in adapter_df.columns
        if col.startswith("Child_Parent_") or col.startswith("Sibling_")
    ]

    assert adapter_gate_cols == sequential_gate_cols
    pdt.assert_frame_equal(
        adapter_df[sequential_gate_cols],
        sequential_df[sequential_gate_cols],
        check_dtype=False,
    )

    # Verify all expected gate columns are present in the annotated DataFrame
    expected_edge_cols = tuple(
        col for col in sequential_gate_cols if col.startswith("Child_Parent_")
    )
    expected_sibling_cols = tuple(col for col in sequential_gate_cols if col.startswith("Sibling_"))
    actual_edge_cols = tuple(col for col in adapter_df.columns if col.startswith("Child_Parent_"))
    actual_sibling_cols = tuple(col for col in adapter_df.columns if col.startswith("Sibling_"))
    assert actual_edge_cols == expected_edge_cols
    assert actual_sibling_cols == expected_sibling_cols
    assert bundle.local_gate_columns == expected_edge_cols
    assert bundle.sibling_gate_columns == expected_sibling_cols
    assert "pipeline" in bundle.metadata
    assert "edge" in bundle.metadata
    assert "sibling" in bundle.metadata
