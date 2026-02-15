from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_projected_test,
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    annotate_sibling_divergence,
    sibling_divergence_test,
)


def _make_two_edge_tree() -> tuple[nx.DiGraph, pd.DataFrame]:
    tree = nx.DiGraph()
    tree.add_edge("root", "A")
    tree.add_edge("root", "B")

    tree.nodes["root"]["distribution"] = np.array([0.5, 0.5], dtype=float)
    tree.nodes["A"]["distribution"] = np.array([0.4, 0.6], dtype=float)
    tree.nodes["B"]["distribution"] = np.array([0.6, 0.4], dtype=float)

    nodes_df = pd.DataFrame(
        {
            "leaf_count": {
                "root": 10,
                "A": 5,
                "B": 5,
            }
        }
    )
    return tree, nodes_df


def _make_sibling_tree() -> tuple[nx.DiGraph, pd.DataFrame]:
    tree = nx.DiGraph()
    tree.add_edge("root", "L")
    tree.add_edge("root", "R")

    tree.nodes["root"]["distribution"] = np.array([0.5, 0.5], dtype=float)
    tree.nodes["L"]["distribution"] = np.array([0.4, 0.6], dtype=float)
    tree.nodes["R"]["distribution"] = np.array([0.6, 0.4], dtype=float)

    tree.nodes["root"]["leaf_count"] = 10
    tree.nodes["L"]["leaf_count"] = 5
    tree.nodes["R"]["leaf_count"] = 5

    nodes_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": {
                "root": False,
                "L": True,
                "R": False,
            }
        }
    )
    return tree, nodes_df


def test_child_parent_nonfinite_keeps_nan_and_uses_conservative_correction(
    monkeypatch,
) -> None:
    tree, nodes_df = _make_two_edge_tree()

    def _fake_compute_p_values_via_projection(
        tree: nx.DiGraph,
        child_ids: list[str],
        parent_ids: list[str],
        child_leaf_counts: np.ndarray,
        parent_leaf_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([np.nan, 3.0], dtype=float),  # stats
            np.array([np.nan, 1.0], dtype=float),  # dfs
            np.array([np.nan, 0.01], dtype=float),  # pvals
            np.array([True, False], dtype=bool),  # invalid mask
        )

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance._compute_p_values_via_projection",
        _fake_compute_p_values_via_projection,
    )

    out = annotate_child_parent_divergence(
        tree=tree,
        nodes_statistics_dataframe=nodes_df,
        significance_level_alpha=0.05,
        fdr_method="flat",
    )

    assert np.isnan(out.loc["A", "Child_Parent_Divergence_P_Value"])
    assert out.loc["A", "Child_Parent_Divergence_P_Value_BH"] == 1.0
    assert bool(out.loc["A", "Child_Parent_Divergence_Significant"]) is False
    assert bool(out.loc["A", "Child_Parent_Divergence_Invalid"]) is True

    assert np.isfinite(out.loc["B", "Child_Parent_Divergence_P_Value_BH"])
    assert bool(out.loc["B", "Child_Parent_Divergence_Invalid"]) is False

    audit = out.attrs.get("child_parent_divergence_audit", {})
    assert audit.get("total_tests") == 2
    assert audit.get("invalid_tests") == 1
    assert audit.get("conservative_path_tests") == 1


def test_edge_projected_test_nonfinite_z_returns_invalid(monkeypatch) -> None:
    def _fake_standardized_z(
        child_dist: np.ndarray,
        parent_dist: np.ndarray,
        n_child: int,
        n_parent: int,
        branch_length: float | None = None,
        mean_branch_length: float | None = None,
    ) -> np.ndarray:
        return np.array([np.nan, 0.0], dtype=float)

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance._compute_standardized_z",
        _fake_standardized_z,
    )

    stat, df, pval, invalid = _compute_projected_test(
        child_dist=np.array([0.4, 0.6], dtype=float),
        parent_dist=np.array([0.5, 0.5], dtype=float),
        n_child=5,
        n_parent=10,
        seed=123,
    )

    assert bool(invalid) is True
    assert np.isnan(stat)
    assert np.isnan(df)
    assert np.isnan(pval)


def test_sibling_nonfinite_keeps_nan_and_uses_conservative_correction(
    monkeypatch,
) -> None:
    tree, nodes_df = _make_sibling_tree()

    def _fake_sibling_test(
        left_dist: np.ndarray,
        right_dist: np.ndarray,
        n_left: float,
        n_right: float,
        branch_length_left: float | None = None,
        branch_length_right: float | None = None,
        mean_branch_length: float | None = None,
        *,
        test_id: str | None = None,
    ) -> tuple[float, float, float]:
        return np.nan, np.nan, np.nan

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test.sibling_divergence_test",
        _fake_sibling_test,
    )

    out = annotate_sibling_divergence(
        tree=tree,
        nodes_statistics_dataframe=nodes_df,
        significance_level_alpha=0.05,
    )

    assert np.isnan(out.loc["root", "Sibling_Divergence_P_Value"])
    assert out.loc["root", "Sibling_Divergence_P_Value_Corrected"] == 1.0
    assert bool(out.loc["root", "Sibling_BH_Different"]) is False
    assert bool(out.loc["root", "Sibling_Divergence_Invalid"]) is True

    audit = out.attrs.get("sibling_divergence_audit", {})
    assert audit.get("total_tests") == 1
    assert audit.get("invalid_tests") == 1
    assert audit.get("conservative_path_tests") == 1


def test_sibling_divergence_nonfinite_z_returns_nan(monkeypatch) -> None:
    def _fake_standardize_proportion_difference(
        theta_1: np.ndarray,
        theta_2: np.ndarray,
        n_1: float,
        n_2: float,
        eps: float = 1e-10,
        branch_length_sum: float | None = None,
        mean_branch_length: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array([np.nan, 0.0], dtype=float), np.array([1.0, 1.0], dtype=float)

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test.standardize_proportion_difference",
        _fake_standardize_proportion_difference,
    )

    stat, df, pval = sibling_divergence_test(
        left_dist=np.array([0.4, 0.6], dtype=float),
        right_dist=np.array([0.5, 0.5], dtype=float),
        n_left=10.0,
        n_right=10.0,
    )

    assert np.isnan(stat)
    assert np.isnan(df)
    assert np.isnan(pval)
