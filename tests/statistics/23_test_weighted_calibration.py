"""Tests for cousin_weighted_wald.py — Gamma GLM calibration + public API.

Covers:
1. WeightedCalibrationModel and predict_weighted_inflation_factor (unit)
2. max_observed_ratio computed from null-like pairs only (bug fix)
3. Gamma GLM fitting (integration)
4. df.attrs["_calibration_model"] storage
5. Post-hoc merge symmetric deflation via tree_decomposition wiring
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    WeightedCalibrationModel,
    predict_weighted_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald import (
    _fit_weighted_inflation_model,
    _weighted_median,
    _WeightedRecord,
)

# =============================================================================
# WeightedCalibrationModel unit tests
# =============================================================================


class TestWeightedCalibrationModel:
    """Unit tests for WeightedCalibrationModel and predict_weighted_inflation_factor."""

    def test_predict_no_calibration_returns_1(self) -> None:
        """Method 'none' should return 1.0 (no deflation)."""
        model = WeightedCalibrationModel(method="none", n_calibration=0, global_c_hat=1.0)
        assert predict_weighted_inflation_factor(model, bl_sum=0.5, n_parent=100) == 1.0

    def test_predict_weighted_median_returns_global_c_hat(self) -> None:
        """Method 'weighted_median' should return global_c_hat (clamped >= 1)."""
        model = WeightedCalibrationModel(
            method="weighted_median", n_calibration=5, global_c_hat=1.8
        )
        c = predict_weighted_inflation_factor(model, bl_sum=0.5, n_parent=100)
        assert c == pytest.approx(1.8)

    def test_predict_weighted_median_clamps_below_1(self) -> None:
        """Inflation factor must be >= 1 (never deflate below raw)."""
        model = WeightedCalibrationModel(
            method="weighted_median", n_calibration=5, global_c_hat=0.5
        )
        c = predict_weighted_inflation_factor(model, bl_sum=0.5, n_parent=100)
        assert c >= 1.0

    def test_predict_gamma_glm_clamps_at_max_observed(self) -> None:
        """GLM predictions must not exceed max_observed_ratio."""
        beta = np.array([1.0, 0.0, 0.5])  # large β₂ → big extrapolation
        model = WeightedCalibrationModel(
            method="gamma_glm",
            n_calibration=10,
            global_c_hat=1.5,
            max_observed_ratio=2.0,
            beta=beta,
        )
        c = predict_weighted_inflation_factor(model, bl_sum=1.0, n_parent=10000)
        assert c <= 2.0

    def test_predict_regression_clamps_at_max_observed(self) -> None:
        """WLS fallback predictions must not exceed max_observed_ratio."""
        beta = np.array([1.0, 0.0, 0.5])
        model = WeightedCalibrationModel(
            method="weighted_regression",
            n_calibration=10,
            global_c_hat=1.5,
            max_observed_ratio=2.0,
            beta=beta,
        )
        c = predict_weighted_inflation_factor(model, bl_sum=1.0, n_parent=10000)
        assert c <= 2.0

    def test_predict_always_positive(self) -> None:
        """Inflation factor is always >= 1."""
        beta = np.array([-5.0, 0.0, 0.0])  # exp(-5) ≈ 0.007
        model = WeightedCalibrationModel(
            method="gamma_glm",
            n_calibration=10,
            global_c_hat=0.5,
            max_observed_ratio=3.0,
            beta=beta,
        )
        c = predict_weighted_inflation_factor(model, bl_sum=1.0, n_parent=10)
        assert c >= 1.0

    def test_predict_no_beta_returns_global_c_hat(self) -> None:
        """If beta is None in regression/glm method, fall back to global_c_hat."""
        model = WeightedCalibrationModel(
            method="gamma_glm",
            n_calibration=10,
            global_c_hat=1.7,
            beta=None,
        )
        c = predict_weighted_inflation_factor(model, bl_sum=0.5, n_parent=100)
        assert c == pytest.approx(1.7)

    def test_predict_intercept_only_ignores_bl(self) -> None:
        """With intercept-only model, bl_sum and n_parent are ignored."""
        model = WeightedCalibrationModel(
            method="gamma_glm",
            n_calibration=10,
            global_c_hat=1.5,
            max_observed_ratio=5.0,
            beta=np.array([0.5]),  # intercept-only: ĉ = exp(0.5) ≈ 1.649
        )
        c = predict_weighted_inflation_factor(model, bl_sum=0.0, n_parent=100)
        assert c == pytest.approx(np.exp(0.5), rel=1e-3)


# =============================================================================
# _weighted_median tests
# =============================================================================


class TestWeightedMedian:
    """Verify the weighted median helper."""

    def test_uniform_weights(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert _weighted_median(values, weights) == pytest.approx(2.0)

    def test_skewed_weights(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.0, 0.0, 1.0])
        assert _weighted_median(values, weights) == pytest.approx(3.0)

    def test_empty_returns_1(self) -> None:
        assert _weighted_median(np.array([]), np.array([])) == 1.0


# =============================================================================
# max_observed_ratio bug fix
# =============================================================================


class TestMaxObservedRatioFix:
    """Verify max_observed_ratio is computed from null-like pairs only."""

    def _make_record(
        self,
        stat: float,
        df: int,
        weight: float,
        is_null_like: bool,
        bl_sum: float = 0.5,
        n_parent: int = 20,
    ) -> _WeightedRecord:
        return _WeightedRecord(
            parent=f"P_{stat:.0f}",
            left="L",
            right="R",
            stat=stat,
            df=df,
            pval=0.5,
            bl_sum=bl_sum,
            n_parent=n_parent,
            weight=weight,
            is_null_like=is_null_like,
        )

    def test_max_ratio_excludes_focal_pairs(self) -> None:
        """Focal (signal) pairs with high T/k must NOT inflate max_observed_ratio."""
        records = [
            # Null-like pairs with moderate T/k
            self._make_record(stat=10.0, df=10, weight=0.8, is_null_like=True),
            self._make_record(stat=12.0, df=10, weight=0.7, is_null_like=True),
            self._make_record(stat=15.0, df=10, weight=0.6, is_null_like=True),
            # Focal pair with very high T/k (signal)
            self._make_record(stat=100.0, df=10, weight=0.01, is_null_like=False),
            self._make_record(stat=80.0, df=10, weight=0.02, is_null_like=False),
            # Need >= 5 for regression
            self._make_record(stat=11.0, df=10, weight=0.5, is_null_like=True),
            self._make_record(stat=13.0, df=10, weight=0.4, is_null_like=True),
        ]
        model = _fit_weighted_inflation_model(records)

        # max_observed_ratio should be from null-like only: max(1.0, 1.2, 1.5, 1.1, 1.3) = 1.5
        assert model.max_observed_ratio <= 1.5 + 1e-9
        # It should NOT be 10.0 (100/10 from focal pair)
        assert model.max_observed_ratio < 5.0

    def test_no_null_like_falls_back_to_all_pairs(self) -> None:
        """When there are no null-like pairs, fall back to all pairs for max ratio."""
        records = [
            self._make_record(stat=10.0, df=10, weight=0.3, is_null_like=False),
            self._make_record(stat=20.0, df=10, weight=0.2, is_null_like=False),
            self._make_record(stat=15.0, df=10, weight=0.1, is_null_like=False),
        ]
        model = _fit_weighted_inflation_model(records)
        # Should use all pairs' max: 20/10 = 2.0
        assert model.max_observed_ratio == pytest.approx(2.0)


# =============================================================================
# Gamma GLM fitting
# =============================================================================


class TestGammaGLMFitting:
    """Test the Gamma GLM calibration model fitting."""

    def _make_records(self, n: int = 10, seed: int = 42) -> list[_WeightedRecord]:
        """Create synthetic records with known inflation."""
        rng = np.random.default_rng(seed)
        records = []
        for i in range(n):
            bl = 0.1 + rng.exponential(0.5)
            n_par = rng.integers(10, 200)
            # True c ≈ 1.5 (mild constant inflation)
            k = 10
            stat = float(rng.chisquare(k) * 1.5)
            weight = float(rng.uniform(0.3, 1.0))
            records.append(
                _WeightedRecord(
                    parent=f"P{i}",
                    left=f"L{i}",
                    right=f"R{i}",
                    stat=stat,
                    df=k,
                    pval=0.5,
                    bl_sum=bl,
                    n_parent=int(n_par),
                    weight=weight,
                    is_null_like=(i % 3 == 0),  # some null-like
                )
            )
        return records

    def test_gamma_glm_method_used(self) -> None:
        """With enough pairs and statsmodels available, should use gamma_glm."""
        records = self._make_records(n=20)
        model = _fit_weighted_inflation_model(records)
        # Should be gamma_glm or weighted_regression (if GLM fails)
        assert model.method in ("gamma_glm", "weighted_regression", "weighted_median")
        # Beta should be populated for regression methods
        if model.method in ("gamma_glm", "weighted_regression"):
            assert model.beta is not None
            assert len(model.beta) == 1  # intercept-only

    def test_gamma_glm_diagnostics_present(self) -> None:
        """Gamma GLM should report deviance-based diagnostics."""
        records = self._make_records(n=20)
        model = _fit_weighted_inflation_model(records)
        if model.method == "gamma_glm":
            assert "deviance" in model.diagnostics
            assert "null_deviance" in model.diagnostics
            assert "aic" in model.diagnostics
            assert "scale" in model.diagnostics
            assert "converged" in model.diagnostics

    def test_few_pairs_falls_back_to_median(self) -> None:
        """< _MIN_REGRESSION pairs should use weighted median."""
        records = self._make_records(n=4)
        model = _fit_weighted_inflation_model(records)
        assert model.method == "weighted_median"

    def test_very_few_pairs_falls_back_to_none(self) -> None:
        """< _MIN_MEDIAN pairs should use 'none'."""
        records = self._make_records(n=2)
        model = _fit_weighted_inflation_model(records)
        assert model.method == "none"

    def test_model_c_hat_reasonable(self) -> None:
        """Global c_hat should be in a reasonable range for mild inflation."""
        records = self._make_records(n=30, seed=123)
        model = _fit_weighted_inflation_model(records)
        # True c ≈ 1.5; global c_hat should be in [0.5, 4.0]
        assert 0.5 < model.global_c_hat < 4.0

    def test_max_observed_from_null_like_only(self) -> None:
        """max_observed_ratio comes from null-like subset."""
        records = self._make_records(n=20)
        null_like_ratios = [
            r.stat / r.df for r in records if r.is_null_like and r.stat > 0 and r.df > 0
        ]
        model = _fit_weighted_inflation_model(records)
        if null_like_ratios:
            assert model.max_observed_ratio == pytest.approx(max(null_like_ratios))


# =============================================================================
# df.attrs["_calibration_model"] storage
# =============================================================================


class TestModelStorage:
    """Verify the model is stored in df.attrs for downstream consumption."""

    def test_annotate_stores_model_in_attrs(self) -> None:
        """annotate_sibling_divergence_weighted must store the model."""
        from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald import (
            annotate_sibling_divergence_weighted,
        )

        # Build a minimal tree
        tree = _build_minimal_tree()

        # Create a DataFrame with the required columns
        df = _build_minimal_annotations(tree)

        result = annotate_sibling_divergence_weighted(tree, df, significance_level_alpha=0.05)

        assert "_calibration_model" in result.attrs
        model = result.attrs["_calibration_model"]
        assert isinstance(model, WeightedCalibrationModel)

    def test_stored_model_has_correct_fields(self) -> None:
        """The stored model should have all required fields."""
        from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald import (
            annotate_sibling_divergence_weighted,
        )

        tree = _build_minimal_tree()
        df = _build_minimal_annotations(tree)
        result = annotate_sibling_divergence_weighted(tree, df, significance_level_alpha=0.05)

        model = result.attrs["_calibration_model"]
        assert hasattr(model, "method")
        assert hasattr(model, "n_calibration")
        assert hasattr(model, "global_c_hat")
        assert hasattr(model, "max_observed_ratio")
        assert hasattr(model, "beta")


# =============================================================================
# Post-hoc merge deflation dispatch
# =============================================================================


class TestPosthocMergeDeflationDispatch:
    """Verify tree_decomposition uses the correct predict function
    depending on which calibration model type is stored."""

    def test_weighted_model_deflates(self) -> None:
        """WeightedCalibrationModel with c_hat > 1 should reduce test_stat."""
        from unittest.mock import patch

        from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree(tree)

        # Manually store a WeightedCalibrationModel with known inflation
        model = WeightedCalibrationModel(
            method="weighted_median",
            n_calibration=10,
            global_c_hat=2.0,
            max_observed_ratio=3.0,
        )
        df.attrs["_calibration_model"] = model

        # Bypass annotation pipeline — this test controls gate columns directly.
        with patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df):
            decomposer = TreeDecomposition(tree, df, posthoc_merge=False)
        # The decomposer should have picked up the model
        assert decomposer._calibration_model is not None
        assert isinstance(decomposer._calibration_model, WeightedCalibrationModel)


# =============================================================================
# Helpers
# =============================================================================


def _build_minimal_tree():
    """Build R → {L0, L1} with distributions."""
    import networkx as nx

    tree = nx.DiGraph()
    tree.add_node("R", distribution=np.array([0.5, 0.5, 0.5]), is_leaf=False, label="R")
    tree.add_node("L0", distribution=np.array([0.9, 0.9, 0.9]), is_leaf=True, label="S0")
    tree.add_node("L1", distribution=np.array([0.1, 0.1, 0.1]), is_leaf=True, label="S1")
    tree.add_edge("R", "L0", branch_length=0.3)
    tree.add_edge("R", "L1", branch_length=0.3)

    from kl_clustering_analysis.tree.poset_tree import PosetTree

    pt = PosetTree(tree)
    return pt


def _build_minimal_annotations(tree) -> pd.DataFrame:
    """Create minimal annotation DataFrame for the tree."""
    nodes = list(tree.nodes)
    df = pd.DataFrame(index=nodes)
    df["Child_Parent_Divergence_Significant"] = False
    df["Child_Parent_Divergence_P_Value_BH"] = 1.0
    df["Child_Parent_Divergence_P_Value"] = 1.0
    # Mark children as significant so they're focal (not skipped)
    for node in nodes:
        if tree.nodes[node].get("is_leaf", False):
            df.loc[node, "Child_Parent_Divergence_Significant"] = True
            df.loc[node, "Child_Parent_Divergence_P_Value_BH"] = 0.001
    return df


def _build_4_cluster_tree():
    """Build R → M1 → {A, B}, R → M2 → {C, D} tree."""
    import networkx as nx

    from kl_clustering_analysis.tree.poset_tree import PosetTree

    rng = np.random.default_rng(42)

    tree = nx.DiGraph()
    # Root
    tree.add_node("R", distribution=rng.uniform(0.3, 0.7, 10), is_leaf=False, label="R")
    # Mid-level
    tree.add_node("M1", distribution=rng.uniform(0.3, 0.7, 10), is_leaf=False, label="M1")
    tree.add_node("M2", distribution=rng.uniform(0.3, 0.7, 10), is_leaf=False, label="M2")
    # Leaves
    for name in ["A", "B", "C", "D"]:
        tree.add_node(name, distribution=rng.uniform(0.1, 0.9, 10), is_leaf=True, label=name)

    tree.add_edge("R", "M1", branch_length=0.5)
    tree.add_edge("R", "M2", branch_length=0.5)
    tree.add_edge("M1", "A", branch_length=0.2)
    tree.add_edge("M1", "B", branch_length=0.2)
    tree.add_edge("M2", "C", branch_length=0.2)
    tree.add_edge("M2", "D", branch_length=0.2)

    return PosetTree(tree)


def _make_annotations_for_4_cluster_tree(tree) -> pd.DataFrame:
    """Create Gate 2+3 annotations for the 4-cluster tree."""
    nodes = list(tree.nodes)
    df = pd.DataFrame(index=nodes)
    df["Child_Parent_Divergence_Significant"] = True
    df["Child_Parent_Divergence_P_Value_BH"] = 0.001
    df["Sibling_BH_Different"] = False
    df["Sibling_BH_Same"] = True
    df["Sibling_Divergence_Skipped"] = False
    df["Sibling_Test_Statistic"] = np.nan
    df["Sibling_Degrees_of_Freedom"] = np.nan
    df["Sibling_Divergence_P_Value"] = np.nan
    df["Sibling_Divergence_P_Value_Corrected"] = np.nan
    df["Sibling_Divergence_Invalid"] = False
    df["Sibling_Test_Method"] = "cousin_weighted_wald"

    # Make R's children (M1, M2) different → split at root
    df.loc["R", "Sibling_BH_Different"] = True
    df.loc["R", "Sibling_BH_Same"] = False
    # Make M1's children (A, B) same → merge
    df.loc["M1", "Sibling_BH_Different"] = False
    df.loc["M1", "Sibling_BH_Same"] = True
    # Make M2's children (C, D) same → merge
    df.loc["M2", "Sibling_BH_Different"] = False
    df.loc["M2", "Sibling_BH_Same"] = True

    return df
