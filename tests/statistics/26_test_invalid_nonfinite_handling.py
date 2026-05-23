from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
    annotate_child_parent_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.adjusted_wald_annotation.pipeline import (
    annotate_sibling_divergence,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection import (
    collect_sibling_pair_records,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.sibling_divergence_test import (
    sibling_divergence_test,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.projection.pair_testing.projection_dimension import (
    resolve_sibling_projection_dimension,
)


def _make_two_edge_tree() -> tuple[nx.DiGraph, pd.DataFrame]:
    tree = nx.DiGraph()
    tree.add_edge("root", "A")
    tree.add_edge("root", "B")

    tree.nodes["root"]["distribution"] = np.array([0.5, 0.5], dtype=float)
    tree.nodes["A"]["distribution"] = np.array([0.4, 0.6], dtype=float)
    tree.nodes["B"]["distribution"] = np.array([0.6, 0.4], dtype=float)
    tree.nodes["A"]["label"] = "A"
    tree.nodes["B"]["label"] = "B"

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
    tree.add_edge("cal", "CL")
    tree.add_edge("cal", "CR")

    tree.nodes["root"]["distribution"] = np.array([0.5, 0.5], dtype=float)
    tree.nodes["L"]["distribution"] = np.array([0.4, 0.6], dtype=float)
    tree.nodes["R"]["distribution"] = np.array([0.6, 0.4], dtype=float)
    tree.nodes["cal"]["distribution"] = np.array([0.5, 0.5], dtype=float)
    tree.nodes["CL"]["distribution"] = np.array([0.5, 0.5], dtype=float)
    tree.nodes["CR"]["distribution"] = np.array([0.5, 0.5], dtype=float)

    tree.nodes["root"]["leaf_count"] = 10
    tree.nodes["L"]["leaf_count"] = 5
    tree.nodes["R"]["leaf_count"] = 5
    tree.nodes["cal"]["leaf_count"] = 10
    tree.nodes["CL"]["leaf_count"] = 5
    tree.nodes["CR"]["leaf_count"] = 5

    nodes_df = pd.DataFrame(
        {
            "Child_Parent_Divergence_P_Value": {
                "root": np.nan,
                "L": 0.01,
                "R": 1.0,
                "cal": np.nan,
                "CL": 1.0,
                "CR": 1.0,
            },
            "Child_Parent_Divergence_P_Value_BH": {
                "root": np.nan,
                "L": 0.01,
                "R": 1.0,
                "cal": np.nan,
                "CL": 1.0,
                "CR": 1.0,
            },
            "Child_Parent_Divergence_Significant": {
                "root": False,
                "L": True,
                "R": False,
                "cal": False,
                "CL": False,
                "CR": False,
            },
            "Child_Parent_Divergence_df": {
                "root": np.nan,
                "L": 1.0,
                "R": 1.0,
                "cal": np.nan,
                "CL": 1.0,
                "CR": 1.0,
            },
            "Child_Parent_Divergence_Invalid": {
                "root": False,
                "L": False,
                "R": False,
                "cal": False,
                "CL": False,
                "CR": False,
            },
            "Child_Parent_Divergence_Tested": {
                "root": False,
                "L": True,
                "R": True,
                "cal": False,
                "CL": True,
                "CR": True,
            },
            "Child_Parent_Divergence_Ancestor_Blocked": {
                "root": False,
                "L": False,
                "R": False,
                "cal": False,
                "CL": False,
                "CR": False,
            },
        }
    )
    return tree, nodes_df


def test_child_parent_nonfinite_results_raise_before_correction(
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
        "kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.child_parent_divergence_annotation.run_child_parent_tests_across_tree",
        _fake_compute_p_values_via_projection,
    )

    with pytest.raises(ValueError, match="invalid result flags"):
        annotate_child_parent_divergence(
            tree=tree,
            annotations_df=nodes_df,
            significance_level_alpha=0.05,
            leaf_data=pd.DataFrame(
                [[0.0, 1.0], [1.0, 0.0]],
                index=["A", "B"],
                dtype=float,
            ),
        )


def test_sibling_nonfinite_results_raise_before_correction(
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
        projection_dimension_from_edge_comparisons: int | None = None,
        parent_principal_component_projection: np.ndarray | None = None,
        parent_principal_component_eigenvalues: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        return np.nan, np.nan, np.nan

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection.sibling_divergence_test",
        _fake_sibling_test,
    )

    with pytest.raises(ValueError, match="finite test statistic"):
        annotate_sibling_divergence(
            tree=tree,
            annotations_df=nodes_df,
            significance_level_alpha=0.05,
            sibling_projection_dimensions_from_edge_comparisons={"root": 2},
            parent_principal_component_projections={"root": np.eye(2, dtype=float)},
            parent_principal_component_eigenvalues={"root": np.ones(2, dtype=float)},
        )


def test_sibling_divergence_nonfinite_z_raises(monkeypatch) -> None:
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
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.wald_statistic.sibling_z_scores.standardize_proportion_difference",
        _fake_standardize_proportion_difference,
    )

    with pytest.raises(ValueError, match="z-scores must be finite"):
        sibling_divergence_test(
            left_distribution=np.array([0.4, 0.6], dtype=float),
            right_distribution=np.array([0.5, 0.5], dtype=float),
            left_sample_size=10.0,
            right_sample_size=10.0,
            projection_dimension_from_edge_comparisons=2,
            parent_principal_component_projection=np.eye(2, dtype=float),
            parent_principal_component_eigenvalues=np.ones(2, dtype=float),
        )


def test_resolve_sibling_projection_dimension_rejects_missing_gate2_dimension() -> None:
    with pytest.raises(ValueError, match="must be supplied by Gate 2 context"):
        resolve_sibling_projection_dimension(
            projection_dimension_from_edge_comparisons=None,
            left_sample_size=3.0,
            right_sample_size=4.0,
            n_features=5,
        )


def test_collect_sibling_pair_records_requires_edge_derived_dimension_and_parent_pca(
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
            "Child_Parent_Divergence_P_Value": {
                "root": np.nan,
                "L": 1.0,
                "R": 1.0,
            },
            "Child_Parent_Divergence_P_Value_BH": {
                "root": np.nan,
                "L": 1.0,
                "R": 1.0,
            },
            "Child_Parent_Divergence_Significant": {
                "root": False,
                "L": False,
                "R": False,
            },
            "Child_Parent_Divergence_df": {
                "root": np.nan,
                "L": 1.0,
                "R": 1.0,
            },
            "Child_Parent_Divergence_Invalid": {
                "root": False,
                "L": False,
                "R": False,
            },
            "Child_Parent_Divergence_Tested": {
                "root": False,
                "L": True,
                "R": True,
            },
            "Child_Parent_Divergence_Ancestor_Blocked": {
                "root": False,
                "L": False,
                "R": False,
            },
        }
    )
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
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.pair_testing.collection.record_collection.sibling_divergence_test",
        _fake_sibling_test,
    )

    records, non_binary = collect_sibling_pair_records(
        tree,
        annotations_df,
        mean_branch_length=None,
        sibling_projection_dimensions_from_edge_comparisons={"root": 2},
        parent_principal_component_projections={"root": np.eye(2, dtype=float)},
        parent_principal_component_eigenvalues={"root": np.ones(2, dtype=float)},
    )

    assert "root" not in non_binary
    assert len(records) == 1
    assert captured["projection_dimension_from_edge_comparisons"] == 2
    np.testing.assert_array_equal(
        captured["parent_principal_component_projection"],
        np.eye(2, dtype=float),
    )
    np.testing.assert_array_equal(
        captured["parent_principal_component_eigenvalues"],
        np.ones(2, dtype=float),
    )
    assert records[0].projection_dimension_source == "derived_from_edge_comparisons"
    assert records[0].resolved_projection_dimension == 2.0
    assert records[0].used_parent_principal_component_basis is True


def test_annotate_sibling_divergence_persists_derived_projection_metadata() -> None:
    tree, nodes_df = _make_sibling_tree()

    out = annotate_sibling_divergence(
        tree=tree,
        annotations_df=nodes_df,
        significance_level_alpha=0.05,
        sibling_projection_dimensions_from_edge_comparisons={"root": 2, "cal": 2},
        parent_principal_component_projections={
            "root": np.eye(2, dtype=float),
            "cal": np.eye(2, dtype=float),
        },
        parent_principal_component_eigenvalues={
            "root": np.ones(2, dtype=float),
            "cal": np.ones(2, dtype=float),
        },
    )

    assert (
        out.loc["root", "Sibling_Projection_Dimension_Source"]
        == "derived_from_edge_comparisons"
    )
    assert float(out.loc["root", "Sibling_Resolved_Projection_Dimension"]) > 0.0
    assert bool(out.loc["root", "Sibling_Used_Parent_Principal_Component_Basis"]) is True

    audit = out.attrs.get("sibling_divergence_audit", {})
    assert audit.get("projection_dimension_source_counts") == {
        "derived_from_edge_comparisons": 2
    }
    assert audit.get("tested_projection_dimension_source_counts") == {
        "derived_from_edge_comparisons": 1
    }


def test_resolve_sibling_projection_dimension_rejects_negative_dimension() -> None:
    with pytest.raises(ValueError, match="Invalid projection_dimension_from_edge_comparisons=-1"):
        resolve_sibling_projection_dimension(
            projection_dimension_from_edge_comparisons=-1,
            left_sample_size=2.0,
            right_sample_size=3.0,
            n_features=4,
        )


def test_resolve_sibling_projection_dimension_accepts_zero_dimension() -> None:
    resolved_k, source = resolve_sibling_projection_dimension(
        projection_dimension_from_edge_comparisons=0,
        left_sample_size=2.0,
        right_sample_size=3.0,
        n_features=4,
    )

    assert resolved_k == 0
    assert source == "derived_from_edge_comparisons"


def test_resolve_sibling_projection_dimension_uses_supplied_edge_derived_dimension() -> None:
    resolved_k, source = resolve_sibling_projection_dimension(
        projection_dimension_from_edge_comparisons=3,
        left_sample_size=2.0,
        right_sample_size=3.0,
        n_features=4,
    )

    assert resolved_k == 3
    assert source == "derived_from_edge_comparisons"
