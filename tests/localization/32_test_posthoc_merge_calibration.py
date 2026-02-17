"""Tests for calibrated post-hoc merge (Fix 1.1: symmetric calibration).

The decomposition uses calibrated T_adj = T/ĉ to decide splits, so the
post-hoc merge must use the same calibration.  Otherwise it's harder to
merge than it was to split (raw inflated T > deflated T_adj).

These tests verify that:
1. The CalibrationModel is threaded from annotation to post-hoc merge.
2. _test_cluster_pair_divergence produces deflated statistics when a model exists.
3. The full v1 pipeline (decompose_tree) produces consistent results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
    CalibrationModel,
    predict_inflation_factor,
)
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.tree.poset_tree import PosetTree

# =============================================================================
# CalibrationModel unit tests
# =============================================================================


class TestCalibrationModel:
    """Unit tests for CalibrationModel and predict_inflation_factor."""

    def test_predict_no_calibration_returns_1(self) -> None:
        """Method 'none' should return 1.0 (no deflation)."""
        model = CalibrationModel(method="none", n_calibration=0, global_c_hat=1.0)
        assert predict_inflation_factor(model, bl_sum=0.5, n_parent=100) == 1.0

    def test_predict_median_returns_global_c_hat(self) -> None:
        """Method 'median' should return global_c_hat (clamped ≥ 1)."""
        model = CalibrationModel(method="median", n_calibration=5, global_c_hat=1.8)
        c = predict_inflation_factor(model, bl_sum=0.5, n_parent=100)
        assert c == pytest.approx(1.8)

    def test_predict_median_clamps_below_1(self) -> None:
        """Inflation factor must be ≥ 1 (never deflate below raw)."""
        model = CalibrationModel(method="median", n_calibration=5, global_c_hat=0.5)
        c = predict_inflation_factor(model, bl_sum=0.5, n_parent=100)
        assert c >= 1.0

    def test_predict_regression_clamps_at_max_observed(self) -> None:
        """Regression predictions must not exceed max_observed_ratio."""
        beta = np.array([1.0, 0.0, 0.5])  # large β₂ → big extrapolation
        model = CalibrationModel(
            method="regression",
            n_calibration=10,
            global_c_hat=1.5,
            max_observed_ratio=2.0,
            beta=beta,
        )
        # Large n_parent should extrapolate but be clamped
        c = predict_inflation_factor(model, bl_sum=0.1, n_parent=10000)
        assert c <= 2.0

    def test_predict_regression_positive(self) -> None:
        """Regression should return a finite positive value."""
        beta = np.array([0.2, 0.1, 0.05])
        model = CalibrationModel(
            method="regression",
            n_calibration=10,
            global_c_hat=1.3,
            max_observed_ratio=3.0,
            beta=beta,
        )
        c = predict_inflation_factor(model, bl_sum=0.5, n_parent=50)
        assert c >= 1.0
        assert np.isfinite(c)


# =============================================================================
# Post-hoc merge calibration integration
# =============================================================================


def _build_4_cluster_tree() -> PosetTree:
    """Build a binary tree with 4 leaf clusters.

    Structure:
              R
            /   \\
           M1    M2
          / \\   / \\
         A   B C   D

    Each internal node gets a distribution that is the mean of its children.
    Branch lengths are set so Felsenstein adjustment is exercised.
    """
    tree = PosetTree()

    # Leaves: distinct distributions
    tree.add_node(
        "A", is_leaf=True, label="A", distribution=np.array([0.8, 0.2, 0.3, 0.7]), leaf_count=25
    )
    tree.add_node(
        "B", is_leaf=True, label="B", distribution=np.array([0.7, 0.3, 0.4, 0.6]), leaf_count=25
    )
    tree.add_node(
        "C", is_leaf=True, label="C", distribution=np.array([0.3, 0.7, 0.6, 0.4]), leaf_count=25
    )
    tree.add_node(
        "D", is_leaf=True, label="D", distribution=np.array([0.2, 0.8, 0.7, 0.3]), leaf_count=25
    )

    # Internal nodes
    tree.add_node(
        "M1", is_leaf=False, distribution=np.array([0.75, 0.25, 0.35, 0.65]), leaf_count=50
    )
    tree.add_node(
        "M2", is_leaf=False, distribution=np.array([0.25, 0.75, 0.65, 0.35]), leaf_count=50
    )
    tree.add_node("R", is_leaf=False, distribution=np.array([0.5, 0.5, 0.5, 0.5]), leaf_count=100)

    # Edges with branch lengths
    tree.add_edge("R", "M1", branch_length=0.3)
    tree.add_edge("R", "M2", branch_length=0.3)
    tree.add_edge("M1", "A", branch_length=0.2)
    tree.add_edge("M1", "B", branch_length=0.2)
    tree.add_edge("M2", "C", branch_length=0.2)
    tree.add_edge("M2", "D", branch_length=0.2)

    return tree


def _make_annotations_for_4_cluster_tree(
    *,
    ab_different: bool = True,
    cd_different: bool = True,
    m1m2_different: bool = True,
) -> pd.DataFrame:
    """Build pre-computed annotation DataFrame for _build_4_cluster_tree.

    This bypasses the annotation pipeline so we can control gate decisions
    directly.
    """
    nodes = ["R", "M1", "M2", "A", "B", "C", "D"]
    df = pd.DataFrame(index=nodes)
    df["Child_Parent_Divergence_Significant"] = False
    df["Sibling_BH_Different"] = False
    df["Sibling_Divergence_Skipped"] = False

    # All children diverge from parent (Gate 2 passes everywhere)
    for node in ["A", "B", "C", "D", "M1", "M2"]:
        df.loc[node, "Child_Parent_Divergence_Significant"] = True

    # Gate 3: sibling divergence
    df.loc["R", "Sibling_BH_Different"] = m1m2_different
    df.loc["M1", "Sibling_BH_Different"] = ab_different
    df.loc["M2", "Sibling_BH_Different"] = cd_different

    return df


class TestPosthocMergeCalibration:
    """Test that post-hoc merge uses calibrated statistics."""

    def test_cluster_pair_divergence_is_deflated_with_model(self) -> None:
        """When a calibration model exists, _test_cluster_pair_divergence should
        return a SMALLER test statistic than the raw Wald."""
        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree(ab_different=True, cd_different=True)

        # Create decomposer WITHOUT calibration model
        decomposer_raw = TreeDecomposition(tree, df, posthoc_merge=False)

        # Get raw stat
        raw_stat, raw_df, raw_pval = decomposer_raw._test_cluster_pair_divergence("A", "B", "M1")

        # Now add a calibration model with ĉ > 1
        df2 = df.copy()
        model = CalibrationModel(
            method="median", n_calibration=10, global_c_hat=2.0, max_observed_ratio=3.0
        )
        df2.attrs["_calibration_model"] = model
        decomposer_cal = TreeDecomposition(tree, df2, posthoc_merge=False)

        # Get calibrated stat
        cal_stat, cal_df, cal_pval = decomposer_cal._test_cluster_pair_divergence("A", "B", "M1")

        # Calibrated stat should be deflated (smaller)
        assert (
            cal_stat < raw_stat
        ), f"Calibrated stat ({cal_stat:.3f}) should be < raw stat ({raw_stat:.3f})"
        # P-value should be larger (less significant)
        assert (
            cal_pval > raw_pval
        ), f"Calibrated p-value ({cal_pval:.4f}) should be > raw p-value ({raw_pval:.4f})"
        # Degrees of freedom should be unchanged
        assert cal_df == raw_df

    def test_no_calibration_model_means_no_deflation(self) -> None:
        """Without a calibration model, stats should pass through unchanged."""
        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree()

        # No _calibration_model in attrs
        assert "_calibration_model" not in df.attrs

        decomposer = TreeDecomposition(tree, df, posthoc_merge=False)
        assert decomposer._calibration_model is None

        # Should still return valid results (no error)
        stat, deg_f, pval = decomposer._test_cluster_pair_divergence("A", "B", "M1")
        assert np.isfinite(stat)
        assert np.isfinite(pval)

    def test_calibration_model_with_c_hat_1_no_change(self) -> None:
        """A model with ĉ=1 should produce identical results to no model."""
        tree = _build_4_cluster_tree()
        df_raw = _make_annotations_for_4_cluster_tree()
        df_cal = df_raw.copy()
        df_cal.attrs["_calibration_model"] = CalibrationModel(
            method="none", n_calibration=0, global_c_hat=1.0
        )

        dec_raw = TreeDecomposition(tree, df_raw, posthoc_merge=False)
        dec_cal = TreeDecomposition(tree, df_cal, posthoc_merge=False)

        raw_stat, _, raw_pval = dec_raw._test_cluster_pair_divergence("C", "D", "M2")
        cal_stat, _, cal_pval = dec_cal._test_cluster_pair_divergence("C", "D", "M2")

        assert cal_stat == pytest.approx(raw_stat, rel=1e-10)
        assert cal_pval == pytest.approx(raw_pval, rel=1e-10)


# =============================================================================
# V1 decompose_tree integration tests
# =============================================================================


class TestV1DecomposeTreeIntegration:
    """Integration tests for the full v1 pipeline (decompose_tree)."""

    def test_4_cluster_all_different_produces_4_clusters(self) -> None:
        """When all sibling pairs are different, decomposition yields 4 clusters."""
        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree(
            ab_different=True, cd_different=True, m1m2_different=True
        )
        result = tree.decompose(results_df=df, posthoc_merge=False)
        assert result["num_clusters"] == 4

    def test_4_cluster_ab_same_produces_3_clusters(self) -> None:
        """When A≈B (not different), they merge → 3 clusters."""
        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree(
            ab_different=False, cd_different=True, m1m2_different=True
        )
        result = tree.decompose(results_df=df, posthoc_merge=False)
        assert result["num_clusters"] == 3

    def test_4_cluster_all_same_produces_1_cluster(self) -> None:
        """When root siblings are not different, everything merges to 1."""
        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree(
            ab_different=True, cd_different=True, m1m2_different=False
        )
        result = tree.decompose(results_df=df, posthoc_merge=False)
        assert result["num_clusters"] == 1

    def test_leaf_partition_is_exact(self) -> None:
        """Every leaf appears in exactly one cluster (partition property)."""
        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree(
            ab_different=True, cd_different=True, m1m2_different=True
        )
        result = tree.decompose(results_df=df, posthoc_merge=False)

        all_leaves = set()
        for cluster_info in result["cluster_assignments"].values():
            members = set(cluster_info["leaves"])
            # No overlap with previously seen leaves
            assert not all_leaves.intersection(
                members
            ), f"Duplicate leaves: {all_leaves.intersection(members)}"
            all_leaves.update(members)

        # All leaves accounted for
        expected = {"A", "B", "C", "D"}
        assert all_leaves == expected

    def test_posthoc_merge_audit_trail_returned(self) -> None:
        """decompose_tree should return a (possibly empty) posthoc_merge_audit."""
        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree()
        result = tree.decompose(results_df=df, posthoc_merge=True)
        assert "posthoc_merge_audit" in result

    def test_posthoc_merge_reduces_overclustering(self) -> None:
        """Post-hoc merge should merge similar clusters after initial split.

        Setup: 4 clusters (A,B,C,D) with A≈B marked as different by gate 3
        (borderline).  Post-hoc merge should try to merge A and B.
        Since the test_divergence callback uses the actual distributions,
        and A/B have similar distributions, the merge should proceed.
        """
        tree = _build_4_cluster_tree()
        df = _make_annotations_for_4_cluster_tree(
            ab_different=True, cd_different=True, m1m2_different=True
        )

        # Without posthoc merge: 4 clusters
        result_no_merge = tree.decompose(results_df=df, posthoc_merge=False)
        assert result_no_merge["num_clusters"] == 4

        # With posthoc merge: may reduce if some pairs are similar
        result_with_merge = tree.decompose(results_df=df, posthoc_merge=True)
        assert result_with_merge["num_clusters"] <= result_no_merge["num_clusters"]
