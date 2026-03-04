from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import (
    LEGACY_EDGE_COLUMNS,
    LEGACY_SIBLING_COLUMNS,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.errors import DecompositionMethodError
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

    stats_df = pd.DataFrame(
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
    return tree, stats_df, leaf_data


@pytest.mark.parametrize(
    "sibling_method",
    [
        "wald",
        "cousin_ftest",
        "cousin_adjusted_wald",
        "cousin_tree_guided",
        "cousin_weighted_wald",
    ],
)
@pytest.mark.parametrize(
    "spectral_method",
    [None, "none", "effective_rank", "marchenko_pastur", "active_features"],
)
def test_pipeline_supports_current_config_method_combinations(
    monkeypatch, sibling_method: str, spectral_method: str | None
) -> None:
    tree, stats_df, leaf_data = _build_small_tree_with_leaf_data()
    monkeypatch.setattr(config, "EDGE_CALIBRATION", False)
    monkeypatch.setattr(config, "PROJECTION_RANDOM_SEED", 123)

    needs_leaf_data = spectral_method not in (None, "none")
    bundle = run_gate_annotation_pipeline(
        tree,
        stats_df.copy(),
        alpha_local=0.01,
        sibling_alpha=0.01,
        leaf_data=leaf_data if needs_leaf_data else None,
        spectral_method=spectral_method,
        min_k=4,
        sibling_method=sibling_method,
        edge_calibration=False,
    )
    out = bundle.annotated_df
    for col in LEGACY_EDGE_COLUMNS:
        assert col in out.columns
    for col in LEGACY_SIBLING_COLUMNS:
        assert col in out.columns
    assert bundle.metadata["resolved_methods"]["sibling_method"] == sibling_method
    if spectral_method == "none":
        assert bundle.metadata["resolved_methods"]["spectral_method"] is None
    else:
        assert bundle.metadata["resolved_methods"]["spectral_method"] == spectral_method


def test_pipeline_rejects_unknown_sibling_method() -> None:
    tree, stats_df, _leaf_data = _build_small_tree_with_leaf_data()
    with pytest.raises(DecompositionMethodError, match="Unknown sibling calibration method"):
        run_gate_annotation_pipeline(
            tree,
            stats_df.copy(),
            sibling_method="not_a_real_method",
            edge_calibration=False,
        )


def test_pipeline_rejects_unknown_spectral_method() -> None:
    tree, stats_df, leaf_data = _build_small_tree_with_leaf_data()
    with pytest.raises(DecompositionMethodError, match="Unknown spectral k estimator"):
        run_gate_annotation_pipeline(
            tree,
            stats_df.copy(),
            leaf_data=leaf_data,
            spectral_method="not_a_real_method",
            edge_calibration=False,
        )
