from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import (
    EDGE_GATE_COLUMNS,
    SIBLING_GATE_COLUMNS,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)


def _build_small_tree_with_leaf_data() -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
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
        dtype=np.float64,
    )
    return tree, annotations_df, leaf_data


@pytest.mark.parametrize(
    "sibling_method",
    ["cousin_adjusted_wald"],
)
@pytest.mark.parametrize("use_leaf_data", [False, True])
def test_pipeline_supports_current_gate_annotation_modes(
    sibling_method: str, use_leaf_data: bool
) -> None:
    tree, annotations_df, leaf_data = _build_small_tree_with_leaf_data()

    bundle = run_gate_annotation_pipeline(
        tree,
        annotations_df.copy(),
        alpha_local=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data if use_leaf_data else None,
        sibling_method=sibling_method,
    )
    out = bundle.annotated_df
    for col in EDGE_GATE_COLUMNS:
        assert col in out.columns
    for col in SIBLING_GATE_COLUMNS:
        assert col in out.columns
    assert bundle.metadata["pipeline"] == "gate_annotation"
    assert "edge" in bundle.metadata
    assert "sibling" in bundle.metadata


def test_pipeline_rejects_unknown_sibling_method() -> None:
    tree, annotations_df, _leaf_data = _build_small_tree_with_leaf_data()
    with pytest.raises(ValueError):
        run_gate_annotation_pipeline(
            tree,
            annotations_df.copy(),
            sibling_method="not_a_real_method",
        )
