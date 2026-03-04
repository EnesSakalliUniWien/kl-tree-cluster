from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis import (
    annotate_edge_gate as top_annotate_edge_gate,
    annotate_sibling_gate as top_annotate_sibling_gate,
    compute_gate_annotations as top_compute_gate_annotations,
)
from kl_clustering_analysis.hierarchy_analysis.decomposition.methods import sibling_calibration
from kl_clustering_analysis.hierarchy_analysis.decomposition.methods.sibling_calibration import (
    apply_sibling_calibration,
)
from kl_clustering_analysis.hierarchy_analysis.gate_annotations import (
    annotate_edge_gate,
    annotate_sibling_gate,
    compute_gate_annotations,
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


def test_compute_gate_annotations_legacy_shim_warns_and_runs(monkeypatch) -> None:
    import kl_clustering_analysis.hierarchy_analysis.gate_annotations as gate_annotations_module

    gate_annotations_module._WARNED_LEGACY_ENTRYPOINTS.clear()
    monkeypatch.setattr(config, "SIBLING_TEST_METHOD", "wald")
    monkeypatch.setattr(config, "EDGE_CALIBRATION", False)
    monkeypatch.setattr(config, "PROJECTION_RANDOM_SEED", 123)
    tree, base_df = _build_small_binary_tree()

    with pytest.deprecated_call(match="compute_gate_annotations is deprecated"):
        out = compute_gate_annotations(
            tree,
            base_df.copy(),
            alpha_local=0.01,
            sibling_alpha=0.01,
            spectral_method=None,
            minimum_projection_dimension=4,
        )

    assert "Child_Parent_Divergence_Significant" in out.columns
    assert "Sibling_BH_Different" in out.columns


def test_legacy_gate_shim_wrappers_warn_and_return_dataframes(monkeypatch) -> None:
    import kl_clustering_analysis.hierarchy_analysis.gate_annotations as gate_annotations_module

    gate_annotations_module._WARNED_LEGACY_ENTRYPOINTS.clear()
    monkeypatch.setattr(config, "SIBLING_TEST_METHOD", "wald")
    monkeypatch.setattr(config, "EDGE_CALIBRATION", False)
    monkeypatch.setattr(config, "PROJECTION_RANDOM_SEED", 123)
    tree, base_df = _build_small_binary_tree()

    with pytest.deprecated_call(match="annotate_edge_gate is deprecated"):
        edge_df = annotate_edge_gate(
            tree,
            base_df.copy(),
            significance_level_alpha=0.01,
            spectral_method=None,
            minimum_projection_dimension=4,
        )
    assert isinstance(edge_df, pd.DataFrame)
    assert "Child_Parent_Divergence_Significant" in edge_df.columns

    with pytest.deprecated_call(match="annotate_sibling_gate is deprecated"):
        sibling_df = annotate_sibling_gate(
            tree,
            edge_df.copy(),
            significance_level_alpha=0.01,
            sibling_method="wald",
        )
    assert isinstance(sibling_df, pd.DataFrame)
    assert "Sibling_BH_Different" in sibling_df.columns


def test_apply_sibling_calibration_warns_and_remains_non_breaking(monkeypatch) -> None:
    sibling_calibration._APPLY_CALIBRATION_WARNING_EMITTED = False
    monkeypatch.setattr(config, "PROJECTION_RANDOM_SEED", 123)

    tree, _base_df = _build_small_binary_tree()
    df = pd.DataFrame(index=["root", "A", "B"])
    df["Child_Parent_Divergence_Significant"] = [False, True, True]

    with pytest.deprecated_call(match="apply_sibling_calibration is deprecated"):
        out = apply_sibling_calibration("wald", tree, df, significance_level_alpha=0.01)
    assert "Sibling_BH_Different" in out.columns


def test_top_level_re_exports_keep_legacy_import_paths_alive() -> None:
    assert callable(top_compute_gate_annotations)
    assert callable(top_annotate_edge_gate)
    assert callable(top_annotate_sibling_gate)
