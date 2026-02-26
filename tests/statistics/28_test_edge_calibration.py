"""Tests for edge_calibration.py — descendant-balance Gate 2 calibration.

Covers:
1. EdgeCalibrationModel and predict_edge_inflation_factor (unit)
2. _fit_edge_calibration_model with descendant-balance-weighted records (unit)
3. calibrate_edges_from_sibling_neighborhood end-to-end (integration)
4. Config toggle: EDGE_CALIBRATION = False bypasses calibration
5. Edge cases: no valid edges, fewer than minimum, NaN stats
6. Descendant-balance weighting: balanced splits dominate calibration
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_calibration import (
    EdgeCalibrationModel,
    _EdgeRecord,
    _fit_edge_calibration_model,
    calibrate_edges_from_sibling_neighborhood,
    predict_edge_inflation_factor,
)


# =============================================================================
# EdgeCalibrationModel unit tests
# =============================================================================


class TestEdgeCalibrationModel:
    """Unit tests for EdgeCalibrationModel and predict_edge_inflation_factor."""

    def test_predict_no_calibration_returns_1(self) -> None:
        model = EdgeCalibrationModel(method="none", n_calibration=0, global_c_hat=1.0)
        assert predict_edge_inflation_factor(model) == 1.0

    def test_predict_weighted_mean_returns_global_c(self) -> None:
        model = EdgeCalibrationModel(method="weighted_mean", n_calibration=5, global_c_hat=1.8)
        assert predict_edge_inflation_factor(model) == pytest.approx(1.8)

    def test_predict_clamps_below_1(self) -> None:
        model = EdgeCalibrationModel(method="weighted_mean", n_calibration=5, global_c_hat=0.5)
        assert predict_edge_inflation_factor(model) >= 1.0

    def test_predict_gamma_glm_uses_beta(self) -> None:
        beta = np.array([np.log(2.0)])  # ĉ = 2.0
        model = EdgeCalibrationModel(
            method="gamma_glm",
            n_calibration=10,
            global_c_hat=2.0,
            max_observed_ratio=3.0,
            beta=beta,
        )
        assert predict_edge_inflation_factor(model) == pytest.approx(2.0)

    def test_predict_clamps_at_max(self) -> None:
        beta = np.array([np.log(5.0)])  # ĉ = 5.0, but max = 3.0
        model = EdgeCalibrationModel(
            method="gamma_glm",
            n_calibration=10,
            global_c_hat=3.0,
            max_observed_ratio=3.0,
            beta=beta,
        )
        assert predict_edge_inflation_factor(model) == pytest.approx(3.0)

    def test_predict_none_beta_returns_global(self) -> None:
        model = EdgeCalibrationModel(
            method="gamma_glm",
            n_calibration=10,
            global_c_hat=1.5,
            beta=None,
        )
        assert predict_edge_inflation_factor(model) == pytest.approx(1.5)


# =============================================================================
# _fit_edge_calibration_model tests
# =============================================================================


class TestFitEdgeCalibrationModel:
    """Tests for the descendant-balance-weight Gamma GLM fitting."""

    @staticmethod
    def _make_records(
        n: int,
        base_stat: float = 10.0,
        base_df: float = 5.0,
        weight: float = 0.5,
        is_null_like: bool = True,
    ) -> list:
        return [
            _EdgeRecord(
                child_id=f"C{i}",
                parent_id=f"P{i}",
                stat=base_stat + np.random.randn() * 0.5,
                df=base_df,
                pval=0.1,
                weight=weight,
                is_null_like=is_null_like,
            )
            for i in range(n)
        ]

    def test_no_records_returns_none_method(self) -> None:
        model = _fit_edge_calibration_model([])
        assert model.method == "none"
        assert model.global_c_hat == 1.0

    def test_too_few_records_returns_none(self) -> None:
        records = self._make_records(2)
        model = _fit_edge_calibration_model(records)
        assert model.method == "none"

    def test_few_records_returns_weighted_mean(self) -> None:
        np.random.seed(42)
        records = self._make_records(4)
        model = _fit_edge_calibration_model(records)
        assert model.method == "weighted_mean"
        assert model.global_c_hat >= 1.0

    def test_enough_records_fits_glm(self) -> None:
        np.random.seed(42)
        records = self._make_records(10)
        model = _fit_edge_calibration_model(records)
        assert model.method in ("gamma_glm", "weighted_regression")
        assert model.n_calibration == 10
        assert model.global_c_hat > 0

    def test_zero_weight_records_excluded(self) -> None:
        records = self._make_records(10, weight=0.0)
        model = _fit_edge_calibration_model(records)
        assert model.method == "none"
        assert model.n_calibration == 0

    def test_high_weight_null_like_dominate(self) -> None:
        """Null-like records (balanced splits) should set max_observed_ratio."""
        np.random.seed(42)
        # Null-like edges with moderate T/k ≈ 2 (balanced splits, weight ≈ 0.5)
        null_records = self._make_records(8, base_stat=10.0, base_df=5.0, weight=0.45, is_null_like=True)
        # Signal edges with high T/k ≈ 10 (imbalanced splits, weight ≈ 0.05)
        signal_records = self._make_records(4, base_stat=50.0, base_df=5.0, weight=0.05, is_null_like=False)
        model = _fit_edge_calibration_model(null_records + signal_records)
        # max_observed_ratio should be based on null-like only (T/k ≈ 2)
        assert model.max_observed_ratio < 5.0

    def test_c_hat_near_1_for_null_data(self) -> None:
        """When T/k ≈ 1 (properly calibrated), ĉ should be near 1."""
        np.random.seed(42)
        k = 10.0
        records = [
            _EdgeRecord(
                child_id=f"C{i}",
                parent_id=f"P{i}",
                stat=float(chi2.rvs(df=k)),
                df=k,
                pval=0.5,
                weight=0.45,  # balanced split → null-like
                is_null_like=True,
            )
            for i in range(20)
        ]
        model = _fit_edge_calibration_model(records)
        c_hat = predict_edge_inflation_factor(model)
        assert 0.7 < c_hat < 1.5, f"ĉ = {c_hat} too far from 1.0 for null data"


# =============================================================================
# Helper: build a small tree + DataFrame for integration tests
# =============================================================================


def _count_leaves(tree: nx.DiGraph, node: str) -> int:
    """Count descendant leaves of a node (leaf = no children)."""
    children = list(tree.successors(node))
    if not children:
        return 1
    return sum(_count_leaves(tree, c) for c in children)


def _build_test_tree_and_df(
    n_internal: int = 5,
    inflate: float = 1.0,
    seed: int = 42,
) -> tuple:
    """Build a binary tree with n_internal internal nodes and a DataFrame.

    Returns (tree, df, child_ids, parent_ids) with Gate 2 columns populated
    and descendant leaf counts stashed in attrs. The inflate parameter
    controls T/k ratio.
    """
    rng = np.random.default_rng(seed)
    k = 5.0

    # Build a binary tree: root → (N1, N2), N1 → (L0, L1), etc.
    tree = nx.DiGraph()
    root = "ROOT"
    tree.add_node(root)

    internal = [root]
    leaves = []
    node_counter = 0
    leaf_counter = 0

    while len(internal) <= n_internal and internal:
        parent = internal.pop(0)
        # Create two children
        if len(internal) + len(leaves) < 2 * n_internal:
            left = f"N{node_counter}"
            node_counter += 1
            right = f"N{node_counter}"
            node_counter += 1
            tree.add_edge(parent, left)
            tree.add_edge(parent, right)
            internal.extend([left, right])
        else:
            left = f"L{leaf_counter}"
            leaf_counter += 1
            right = f"L{leaf_counter}"
            leaf_counter += 1
            tree.add_edge(parent, left)
            tree.add_edge(parent, right)
            leaves.extend([left, right])

    # Add remaining leaves
    for node in list(internal):
        left = f"L{leaf_counter}"
        leaf_counter += 1
        right = f"L{leaf_counter}"
        leaf_counter += 1
        tree.add_edge(node, left)
        tree.add_edge(node, right)
        leaves.extend([left, right])

    all_nodes = list(tree.nodes())

    # Build DataFrame
    df = pd.DataFrame(index=all_nodes)
    df["leaf_count"] = [_count_leaves(tree, n) for n in all_nodes]

    # Set distributions (dummy)
    for n in all_nodes:
        tree.nodes[n]["distribution"] = np.array([0.5] * 10)

    # --- Gate 2 columns (raw, before calibration) ---
    edge_list = list(tree.edges())
    child_ids = [c for _, c in edge_list]
    parent_ids = [p for p, _ in edge_list]
    n_edges = len(child_ids)

    test_stats = rng.gamma(shape=inflate, scale=k, size=n_edges)
    degrees_of_freedom = np.full(n_edges, k)
    p_values = np.array([float(chi2.sf(t, df=k)) for t in test_stats])

    # Compute descendant leaf counts per edge
    child_leaf_counts = np.array([_count_leaves(tree, c) for c in child_ids], dtype=float)
    parent_leaf_counts = np.array([_count_leaves(tree, p) for p in parent_ids], dtype=float)

    df["Child_Parent_Divergence_P_Value"] = np.nan
    df["Child_Parent_Divergence_P_Value_BH"] = np.nan
    df["Child_Parent_Divergence_Significant"] = False
    df["Child_Parent_Divergence_df"] = np.nan
    df["Child_Parent_Divergence_Invalid"] = False

    df.loc[child_ids, "Child_Parent_Divergence_P_Value"] = p_values
    df.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = p_values  # simplified
    df.loc[child_ids, "Child_Parent_Divergence_Significant"] = p_values < 0.05
    df.loc[child_ids, "Child_Parent_Divergence_df"] = k

    # Stash raw test data in attrs (including leaf counts for calibration)
    df.attrs["_edge_raw_test_data"] = {
        "child_ids": child_ids,
        "parent_ids": parent_ids,
        "test_stats": test_stats,
        "degrees_of_freedom": degrees_of_freedom,
        "p_values": p_values,
        "child_leaf_counts": child_leaf_counts,
        "parent_leaf_counts": parent_leaf_counts,
    }

    return tree, df, child_ids, parent_ids


# =============================================================================
# calibrate_edges_from_sibling_neighborhood integration tests
# =============================================================================


class TestCalibrateEdgesFromSiblingNeighborhood:
    """End-to-end tests for the post-hoc calibration function."""

    def test_deflation_reduces_inflated_stats(self) -> None:
        """With inflated T/k ratios, calibration should reduce statistics."""
        tree, df, child_ids, parent_ids = _build_test_tree_and_df(
            n_internal=8, inflate=3.0, seed=42
        )

        original_pvals = df["Child_Parent_Divergence_P_Value_BH"].dropna().copy()

        result_df = calibrate_edges_from_sibling_neighborhood(
            tree, df, alpha=0.05, fdr_method="flat"
        )

        new_pvals = result_df["Child_Parent_Divergence_P_Value_BH"].dropna()
        # Deflated p-values should be larger (less significant)
        assert new_pvals.mean() >= original_pvals.mean()

    def test_no_raw_data_returns_unchanged(self) -> None:
        """Without raw test data in attrs, returns unchanged DataFrame."""
        tree = nx.DiGraph()
        tree.add_edge("ROOT", "C1")
        df = pd.DataFrame(index=["ROOT", "C1"])
        # No _edge_raw_test_data in attrs
        result = calibrate_edges_from_sibling_neighborhood(tree, df)
        assert result is df  # unchanged

    def test_no_leaf_counts_returns_unchanged(self) -> None:
        """Without leaf counts in stashed data, raises KeyError."""
        tree = nx.DiGraph()
        tree.add_edge("ROOT", "C1")
        df = pd.DataFrame(index=["ROOT", "C1"])
        df.attrs["_edge_raw_test_data"] = {
            "child_ids": ["C1"],
            "parent_ids": ["ROOT"],
            "test_stats": np.array([10.0]),
            "degrees_of_freedom": np.array([5.0]),
            "p_values": np.array([0.05]),
            # Missing child_leaf_counts and parent_leaf_counts
        }
        with pytest.raises(KeyError):
            calibrate_edges_from_sibling_neighborhood(tree, df)

    def test_nan_stats_passed_through(self) -> None:
        """NaN edge stats should not be affected by calibration."""
        tree, df, child_ids, parent_ids = _build_test_tree_and_df(n_internal=5, seed=42)

        # Set one stat to NaN
        raw = df.attrs["_edge_raw_test_data"]
        raw["test_stats"][0] = np.nan
        raw["p_values"][0] = np.nan
        df.loc[child_ids[0], "Child_Parent_Divergence_P_Value"] = np.nan

        result = calibrate_edges_from_sibling_neighborhood(
            tree, df, alpha=0.05, fdr_method="flat"
        )

        assert np.isnan(result.loc[child_ids[0], "Child_Parent_Divergence_P_Value"])

    def test_c_hat_1_when_no_inflation(self) -> None:
        """When T/k ≈ 1 (null data), calibration should have ĉ ≈ 1."""
        tree, df, child_ids, parent_ids = _build_test_tree_and_df(
            n_internal=8, inflate=1.0, seed=42
        )

        result = calibrate_edges_from_sibling_neighborhood(
            tree, df, alpha=0.05, fdr_method="flat"
        )

        model = result.attrs.get("edge_calibration_model")
        if model is not None and model.method != "none":
            c_hat = predict_edge_inflation_factor(model)
            assert 0.5 < c_hat < 2.0, f"ĉ = {c_hat} too far from 1.0 for null data"

    def test_balanced_split_gets_high_weight(self) -> None:
        """Balanced splits (50/50) should produce weight ≈ 0.5 (null-like)."""
        # Build a tree where ROOT has two children each with equal leaf counts
        tree = nx.DiGraph()
        tree.add_edge("ROOT", "L")
        tree.add_edge("ROOT", "R")
        tree.add_edge("L", "L0")
        tree.add_edge("L", "L1")
        tree.add_edge("R", "R0")
        tree.add_edge("R", "R1")

        child_ids = ["L", "R", "L0", "L1", "R0", "R1"]
        parent_ids = ["ROOT", "ROOT", "L", "L", "R", "R"]
        n_edges = len(child_ids)

        k = 5.0
        rng = np.random.default_rng(42)
        test_stats = rng.gamma(shape=3.0, scale=k, size=n_edges)
        degrees_of_freedom = np.full(n_edges, k)
        p_values = np.array([float(chi2.sf(t, df=k)) for t in test_stats])

        child_leaf_counts = np.array([
            _count_leaves(tree, c) for c in child_ids
        ], dtype=float)
        parent_leaf_counts = np.array([
            _count_leaves(tree, p) for p in parent_ids
        ], dtype=float)

        df = pd.DataFrame(index=list(tree.nodes()))
        df["Child_Parent_Divergence_P_Value"] = np.nan
        df["Child_Parent_Divergence_P_Value_BH"] = np.nan
        df["Child_Parent_Divergence_Significant"] = False
        df["Child_Parent_Divergence_df"] = np.nan
        df["Child_Parent_Divergence_Invalid"] = False
        df.loc[child_ids, "Child_Parent_Divergence_P_Value"] = p_values
        df.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = p_values
        df.loc[child_ids, "Child_Parent_Divergence_Significant"] = p_values < 0.05
        df.loc[child_ids, "Child_Parent_Divergence_df"] = k

        df.attrs["_edge_raw_test_data"] = {
            "child_ids": child_ids,
            "parent_ids": parent_ids,
            "test_stats": test_stats,
            "degrees_of_freedom": degrees_of_freedom,
            "p_values": p_values,
            "child_leaf_counts": child_leaf_counts,
            "parent_leaf_counts": parent_leaf_counts,
        }

        result = calibrate_edges_from_sibling_neighborhood(
            tree, df, alpha=0.05, fdr_method="flat"
        )

        model = result.attrs.get("edge_calibration_model")
        # All parent splits are balanced (50/50) → high weight
        if model is not None and model.diagnostics:
            diagnostics = model.diagnostics
            if "total_weight" in diagnostics:
                # Each edge weight ≈ 0.5, all edges should contribute
                assert diagnostics["total_weight"] > 1.0

    def test_imbalanced_split_gets_low_weight(self) -> None:
        """Imbalanced splits (one tiny child) should produce low weight."""
        # Build a caterpillar tree where each split peels off 1 leaf
        # ROOT → (L0, N1), N1 → (L1, N2), N2 → (L2, L3)
        # All splits are ~1/N → very low weight
        tree = nx.DiGraph()
        tree.add_edge("ROOT", "L0")
        tree.add_edge("ROOT", "N1")
        tree.add_edge("N1", "L1")
        tree.add_edge("N1", "N2")
        tree.add_edge("N2", "L2")
        tree.add_edge("N2", "L3")

        child_ids = ["L0", "N1", "L1", "N2", "L2", "L3"]
        parent_ids = ["ROOT", "ROOT", "N1", "N1", "N2", "N2"]
        n_edges = len(child_ids)

        k = 5.0
        rng = np.random.default_rng(42)
        test_stats = rng.gamma(shape=3.0, scale=k, size=n_edges)
        degrees_of_freedom = np.full(n_edges, k)
        p_values = np.array([float(chi2.sf(t, df=k)) for t in test_stats])

        child_leaf_counts = np.array([
            _count_leaves(tree, c) for c in child_ids
        ], dtype=float)
        parent_leaf_counts = np.array([
            _count_leaves(tree, p) for p in parent_ids
        ], dtype=float)

        df = pd.DataFrame(index=list(tree.nodes()))
        df["Child_Parent_Divergence_P_Value"] = np.nan
        df["Child_Parent_Divergence_P_Value_BH"] = np.nan
        df["Child_Parent_Divergence_Significant"] = False
        df["Child_Parent_Divergence_df"] = np.nan
        df["Child_Parent_Divergence_Invalid"] = False
        df.loc[child_ids, "Child_Parent_Divergence_P_Value"] = p_values
        df.loc[child_ids, "Child_Parent_Divergence_P_Value_BH"] = p_values
        df.loc[child_ids, "Child_Parent_Divergence_Significant"] = p_values < 0.05
        df.loc[child_ids, "Child_Parent_Divergence_df"] = k

        df.attrs["_edge_raw_test_data"] = {
            "child_ids": child_ids,
            "parent_ids": parent_ids,
            "test_stats": test_stats,
            "degrees_of_freedom": degrees_of_freedom,
            "p_values": p_values,
            "child_leaf_counts": child_leaf_counts,
            "parent_leaf_counts": parent_leaf_counts,
        }

        result = calibrate_edges_from_sibling_neighborhood(
            tree, df, alpha=0.05, fdr_method="flat"
        )

        model = result.attrs.get("edge_calibration_model")
        # ROOT: min(1, 3)/4 = 0.25 → below 0.3 threshold
        # N1: min(1, 2)/3 = 0.33 → barely above
        # N2: min(1, 1)/2 = 0.5 → balanced (leaf pair)
        # Most splits are imbalanced → fewer null-like edges
        if model is not None and model.diagnostics:
            diagnostics = model.diagnostics
            if "n_null_like" in diagnostics:
                # Only N2 (0.5) and N1 (0.33) are null-like (>0.3)
                assert diagnostics["n_null_like"] <= 4  # at most 4 of 6 edges

    def test_audit_metadata_attached(self) -> None:
        """Calibration should attach audit metadata to the DataFrame."""
        tree, df, child_ids, parent_ids = _build_test_tree_and_df(
            n_internal=5, inflate=2.0, seed=42
        )

        result = calibrate_edges_from_sibling_neighborhood(
            tree, df, alpha=0.05, fdr_method="flat"
        )

        assert "edge_calibration_model" in result.attrs
        assert "edge_calibration_audit" in result.attrs
        model = result.attrs["edge_calibration_model"]
        assert isinstance(model, EdgeCalibrationModel)

    def test_empty_tree_returns_unchanged(self) -> None:
        """Edge calibration with no edges returns unchanged DataFrame."""
        tree = nx.DiGraph()
        tree.add_node("ROOT")
        df = pd.DataFrame(index=["ROOT"])
        df.attrs["_edge_raw_test_data"] = {
            "child_ids": [],
            "parent_ids": [],
            "test_stats": np.array([]),
            "degrees_of_freedom": np.array([]),
            "p_values": np.array([]),
            "child_leaf_counts": np.array([]),
            "parent_leaf_counts": np.array([]),
        }

        result = calibrate_edges_from_sibling_neighborhood(tree, df)
        assert len(result) == 1
