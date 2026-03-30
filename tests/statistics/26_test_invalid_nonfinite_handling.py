from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    annotate_sibling_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing import (
    sibling_divergence_test,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection import (
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic import (
    _resolve_sibling_projection_dimension,
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
            },
            "Child_Parent_Divergence_P_Value_BH": {
                "root": 1.0,
                "L": 0.01,
                "R": 1.0,
            },
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
        spectral_dims=None,
        pca_projections=None,
        pca_eigenvalues=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([np.nan, 3.0], dtype=float),  # stats
            np.array([np.nan, 1.0], dtype=float),  # dfs
            np.array([np.nan, 0.01], dtype=float),  # pvals
            np.array([True, False], dtype=bool),  # invalid mask
        )

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence.run_child_parent_tests_across_tree",
        _fake_compute_p_values_via_projection,
    )

    out = annotate_child_parent_divergence(
        tree=tree,
        annotations_df=nodes_df,
        significance_level_alpha=0.05,
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


def test_sibling_nonfinite_keeps_nan_and_uses_conservative_correction(
    monkeypatch,
) -> None:
    tree, nodes_df = _make_sibling_tree()

    def _fake_sibling_test(
        left_distribution: np.ndarray,
        right_distribution: np.ndarray,
        left_sample_size: float,
        right_sample_size: float,
        branch_length_left: float | None = None,
        branch_length_right: float | None = None,
        mean_branch_length: float | None = None,
        *,
        spectral_k: int | None = None,
        pca_projection: np.ndarray | None = None,
        pca_eigenvalues: np.ndarray | None = None,
        whitening: str = "per_component",
        **kwargs,
    ) -> tuple[float, float, float]:
        return np.nan, np.nan, np.nan

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection.sibling_divergence_test",
        _fake_sibling_test,
    )

    out = annotate_sibling_divergence(
        tree=tree,
        annotations_df=nodes_df,
        significance_level_alpha=0.05,
    )

    assert np.isnan(out.loc["root", "Sibling_Divergence_P_Value"])
    assert out.loc["root", "Sibling_Divergence_P_Value_Corrected"] == 1.0
    assert bool(out.loc["root", "Sibling_BH_Different"]) is False
    assert bool(out.loc["root", "Sibling_Divergence_Invalid"]) is True

    audit = out.attrs.get("sibling_divergence_audit", {})
    assert audit.get("total_pairs") == 1
    assert audit.get("calibration_method") == "weighted_mean"
    assert audit.get("calibration_n") == 0
    assert audit.get("test_method") == "cousin_adjusted_wald"


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
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.standardize_proportion_difference",
        _fake_standardize_proportion_difference,
    )

    stat, df, pval = sibling_divergence_test(
        left_distribution=np.array([0.4, 0.6], dtype=float),
        right_distribution=np.array([0.5, 0.5], dtype=float),
        left_sample_size=10.0,
        right_sample_size=10.0,
    )

    assert np.isnan(stat)
    assert np.isnan(df)
    assert np.isnan(pval)


def test_resolve_sibling_projection_dimension_uses_johnson_lindenstrauss_fallback_for_missing_dimension(
    monkeypatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def _fake_compute_projection_dimension(
        total_sample_size: int,
        n_features: int,
    ) -> int:
        calls.append((total_sample_size, n_features))
        return 7

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.compute_projection_dimension",
        _fake_compute_projection_dimension,
    )

    resolved_k, source = _resolve_sibling_projection_dimension(
        projection_dimension_from_edge_comparisons=None,
        left_sample_size=3.0,
        right_sample_size=4.0,
        n_features=5,
    )

    assert resolved_k == 7
    assert source == "johnson_lindenstrauss_fallback"
    assert calls == [(7, 5)]


def test_collect_sibling_pair_records_ignores_parent_principal_component_basis_and_uses_johnson_lindenstrauss_fallback_when_edge_derived_dimension_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = nx.DiGraph()
    tree.add_edge("root", "L")
    tree.add_edge("root", "R")

    tree.nodes["root"]["distribution"] = np.array([0.5, 0.5], dtype=float)
    tree.nodes["L"]["distribution"] = np.array([0.4, 0.6], dtype=float)
    tree.nodes["R"]["distribution"] = np.array([0.6, 0.4], dtype=float)
    tree.nodes["root"]["leaf_count"] = 10
    tree.nodes["L"]["leaf_count"] = 5
    tree.nodes["R"]["leaf_count"] = 5

    annotations_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_Significant": {
                "root": False,
                "L": False,
                "R": False,
            },
            "Child_Parent_Divergence_P_Value_BH": {
                "root": 1.0,
                "L": 1.0,
                "R": 1.0,
            },
        }
    )
    annotations_df.attrs["_pca_projections"] = {
        "root": np.array(
            [
                [3.0, 4.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
    }
    annotations_df.attrs["_pca_eigenvalues"] = {"root": np.array([2.5, 0.5], dtype=float)}

    captured: dict[str, object] = {}

    def _fake_sibling_test(
        left_distribution: np.ndarray,
        right_distribution: np.ndarray,
        left_sample_size: float,
        right_sample_size: float,
        branch_length_left: float | None = None,
        branch_length_right: float | None = None,
        mean_branch_length: float | None = None,
        *,
        projection_dimension_from_edge_comparisons: int | None = None,
        parent_principal_component_projection: np.ndarray | None = None,
        parent_principal_component_eigenvalues: np.ndarray | None = None,
        whitening: str = "per_component",
        **kwargs,
    ) -> tuple[float, float, float]:
        captured["projection_dimension_from_edge_comparisons"] = (
            projection_dimension_from_edge_comparisons
        )
        captured["parent_principal_component_projection"] = (
            parent_principal_component_projection
        )
        captured["parent_principal_component_eigenvalues"] = (
            parent_principal_component_eigenvalues
        )
        return 1.0, 1.0, 0.5

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.sibling_pair_collection.sibling_divergence_test",
        _fake_sibling_test,
    )

    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_branch_length=None,
        sibling_projection_dimensions_from_edge_comparisons=None,
        parent_principal_component_projections=None,
        parent_principal_component_eigenvalues=None,
    )

    assert "root" not in non_binary
    assert len(records) == 1
    assert captured["projection_dimension_from_edge_comparisons"] is None
    assert captured["parent_principal_component_projection"] is None
    assert captured["parent_principal_component_eigenvalues"] is None
    assert (
        records[0].projection_dimension_source == "johnson_lindenstrauss_fallback"
    )
    assert np.isnan(records[0].resolved_projection_dimension)
    assert records[0].used_parent_principal_component_basis is False


def test_annotate_sibling_divergence_persists_jl_projection_diagnostics() -> None:
    tree, nodes_df = _make_sibling_tree()

    out = annotate_sibling_divergence(
        tree=tree,
        annotations_df=nodes_df,
        significance_level_alpha=0.05,
    )

    assert (
        out.loc["root", "Sibling_Projection_Dimension_Source"]
        == "johnson_lindenstrauss_fallback"
    )
    assert float(out.loc["root", "Sibling_Resolved_Projection_Dimension"]) > 0.0
    assert bool(out.loc["root", "Sibling_Used_Parent_Principal_Component_Basis"]) is False

    audit = out.attrs.get("sibling_divergence_audit", {})
    assert audit.get("calibration_projection_dimension_source_counts") == {
        "johnson_lindenstrauss_fallback": 1
    }
    assert audit.get("excluded_from_calibration_projection_dimension_source_counts") == {}
    assert audit.get("projection_dimension_source_counts") == {
        "johnson_lindenstrauss_fallback": 1
    }
    assert audit.get("tested_projection_dimension_source_counts") == {
        "johnson_lindenstrauss_fallback": 1
    }


def test_resolve_sibling_projection_dimension_rejects_nonpositive_dimension(
    monkeypatch,
) -> None:
    def _fail_compute_projection_dimension(
        total_sample_size: int,
        n_features: int,
    ) -> int:
        raise AssertionError("JL fallback should not run for invalid spectral_k")

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.compute_projection_dimension",
        _fail_compute_projection_dimension,
    )

    with pytest.raises(ValueError, match="Invalid projection_dimension_from_edge_comparisons=0"):
        _resolve_sibling_projection_dimension(
            projection_dimension_from_edge_comparisons=0,
            left_sample_size=2.0,
            right_sample_size=3.0,
            n_features=4,
        )


def test_resolve_sibling_projection_dimension_uses_supplied_edge_derived_dimension_without_johnson_lindenstrauss_fallback(
    monkeypatch,
) -> None:
    def _fail_compute_projection_dimension(
        total_sample_size: int,
        n_features: int,
    ) -> int:
        raise AssertionError("JL fallback should not run for positive spectral_k")

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.compute_projection_dimension",
        _fail_compute_projection_dimension,
    )

    resolved_k, source = _resolve_sibling_projection_dimension(
        projection_dimension_from_edge_comparisons=3,
        left_sample_size=2.0,
        right_sample_size=3.0,
        n_features=4,
    )

    assert resolved_k == 3
    assert source == "derived_from_edge_comparisons"
