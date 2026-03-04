from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.decomposition.core.contracts import GateAnnotationBundle
from kl_clustering_analysis.hierarchy_analysis.decomposition.gates.orchestrator import (
    run_gate_annotation_pipeline,
)
from kl_clustering_analysis.hierarchy_analysis.gate_annotations import compute_gate_annotations


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


@pytest.mark.parametrize("sibling_method", ["wald", "cousin_weighted_wald"])
def test_gate_adapter_pipeline_matches_legacy_annotations(monkeypatch, sibling_method: str) -> None:
    tree, base_df = _build_small_binary_tree()

    monkeypatch.setattr(config, "SIBLING_TEST_METHOD", sibling_method)
    monkeypatch.setattr(config, "EDGE_CALIBRATION", False)
    monkeypatch.setattr(config, "PROJECTION_RANDOM_SEED", 123)

    legacy_df = compute_gate_annotations(
        tree,
        base_df.copy(),
        alpha_local=0.01,
        sibling_alpha=0.01,
        spectral_method=None,
        min_k=4,
    )
    bundle = run_gate_annotation_pipeline(
        tree,
        base_df.copy(),
        alpha_local=0.01,
        sibling_alpha=0.01,
        spectral_method=None,
        min_k=4,
        sibling_method=sibling_method,
        edge_calibration=False,
    )

    assert isinstance(bundle, GateAnnotationBundle)
    adapter_df = bundle.annotated_df

    legacy_gate_cols = [
        col
        for col in legacy_df.columns
        if col.startswith("Child_Parent_") or col.startswith("Sibling_")
    ]
    adapter_gate_cols = [
        col
        for col in adapter_df.columns
        if col.startswith("Child_Parent_") or col.startswith("Sibling_")
    ]

    assert adapter_gate_cols == legacy_gate_cols
    pdt.assert_frame_equal(adapter_df[legacy_gate_cols], legacy_df[legacy_gate_cols], check_dtype=False)

    assert bundle.local_gate_column == "Child_Parent_Divergence_Significant"
    assert bundle.sibling_gate_column == "Sibling_BH_Different"
    assert bundle.local_gate_columns == tuple(
        col for col in legacy_gate_cols if col.startswith("Child_Parent_")
    )
    assert bundle.sibling_gate_columns == tuple(col for col in legacy_gate_cols if col.startswith("Sibling_"))
    assert bundle.metadata["column_names"]["edge"] == list(bundle.local_gate_columns)
    assert bundle.metadata["column_names"]["sibling"] == list(bundle.sibling_gate_columns)


def test_compute_gate_annotations_delegates_to_orchestrator(monkeypatch) -> None:
    tree, base_df = _build_small_binary_tree()
    sentinel_df = base_df.copy()
    sentinel_df["sentinel"] = 1

    captured: dict[str, object] = {}

    def _fake_pipeline(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return GateAnnotationBundle(annotated_df=sentinel_df)

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.gate_annotations.run_gate_annotation_pipeline",
        _fake_pipeline,
    )
    monkeypatch.setattr(config, "SIBLING_TEST_METHOD", "cousin_tree_guided")

    out = compute_gate_annotations(
        tree,
        base_df,
        alpha_local=0.02,
        sibling_alpha=0.03,
        leaf_data=None,
        spectral_method="effective_rank",
        min_k=7,
    )

    assert out is sentinel_df
    assert captured["args"] == (tree, base_df)
    assert captured["kwargs"] == {
        "alpha_local": 0.02,
        "sibling_alpha": 0.03,
        "leaf_data": None,
        "spectral_method": "effective_rank",
        "min_k": 7,
        "sibling_method": "cousin_tree_guided",
        "fdr_method": "tree_bh",
        "sibling_spectral_dims": None,
        "sibling_pca_projections": None,
        "sibling_pca_eigenvalues": None,
        "edge_calibration": None,
    }
