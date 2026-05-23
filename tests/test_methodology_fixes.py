"""Tests for the three methodology fixes:

1. Fix 1: spectral_k floor raised from 1 to 4
2. Fix 2: Non-binary and leaf nodes marked as Sibling_Divergence_Skipped=True
3. Fix 3: Shared Satterthwaite helper (compute_projected_pvalue)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.projected_wald.projected_wald_reference_distribution import (
    compute_projected_pvalue,
)
from scipy.stats import chi2

# =============================================================================
# Fix 3: Shared Satterthwaite helper tests
# =============================================================================


class TestComputeProjectedPvalue:
    """Tests for the shared compute_projected_pvalue helper."""

    def test_rejects_missing_eigenvalues(self):
        """Projected Wald calibration requires the PCA spectrum."""
        rng = np.random.default_rng(42)
        projected = rng.standard_normal(10)

        with pytest.raises(ValueError, match="requires PCA eigenvalues"):
            compute_projected_pvalue(projected, eigenvalues=None)

    def test_rejects_empty_eigenvalues(self):
        """An empty spectrum is malformed Gate 2 PCA context."""
        rng = np.random.default_rng(42)
        projected = rng.standard_normal(5)

        with pytest.raises(ValueError, match="same length"):
            compute_projected_pvalue(projected, eigenvalues=np.array([]))

    def test_satterthwaite_calibration_with_eigenvalues(self):
        """Eigenvalue-aware projected tests use Satterthwaite calibration."""
        projected = np.array([1.0, 2.0, 3.0])
        eigenvalues = np.array([2.0, 1.0, 0.5])
        reference = compute_projected_pvalue(projected, eigenvalues=eigenvalues)
        expected_stat = float(np.sum(projected**2))
        expected_df = float(np.sum(eigenvalues) ** 2) / float(np.sum(eigenvalues**2))
        expected_scale = float(np.sum(eigenvalues**2)) / float(np.sum(eigenvalues))
        assert abs(reference.statistic - expected_stat) < 1e-10
        assert abs(reference.reference_scale - expected_scale) < 1e-10
        assert abs(reference.degrees_of_freedom - expected_df) < 1e-10
        assert abs(
            reference.p_value - float(chi2.sf(expected_stat / expected_scale, df=expected_df))
        ) < 1e-10


# =============================================================================
# Fix 1: spectral_k floor test
# =============================================================================


class TestSpectralKFloor:
    """Verify the Gate 2 spectral path uses its fixed small floor."""

    def test_gate2_spectral_minimum_projection_dimension_is_fixed(self, monkeypatch):
        """Gate 2 should pass the fixed spectral floor into the spectral estimator."""
        import kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context as spectral_module
        import networkx as nx

        tree = nx.DiGraph()
        tree.add_edge("root", "L0")
        tree.add_edge("root", "L1")
        tree.nodes["L0"]["label"] = "L0"
        tree.nodes["L1"]["label"] = "L1"
        tree.nodes["L0"]["is_leaf"] = True
        tree.nodes["L1"]["is_leaf"] = True
        leaf_data = pd.DataFrame([[0.0], [1.0]], index=["L0", "L1"], columns=["F0"])

        captured: list[int] = []

        def _fake_compute_spectral_decomposition(*args, **kwargs):
            captured.append(kwargs["minimum_projection_dimension"])
            return {}, {}, {}

        monkeypatch.setattr(
            spectral_module,
            "compute_spectral_decomposition",
            _fake_compute_spectral_decomposition,
        )

        spectral_module.compute_child_parent_spectral_context(tree, leaf_data)

        assert captured == [2]

    def test_gate2_spectral_context_requires_paired_projection_eigenvalue_keys(
        self, monkeypatch
    ):
        """Gate 2 PCA projections and eigenvalues must be keyed identically."""
        import kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context as spectral_module
        import networkx as nx

        tree = nx.DiGraph()
        tree.add_edges_from([("root", "L0"), ("root", "L1")])
        for leaf in ["L0", "L1"]:
            tree.nodes[leaf]["label"] = leaf
            tree.nodes[leaf]["is_leaf"] = True
        leaf_data = pd.DataFrame([[0.0], [1.0]], index=["L0", "L1"], columns=["F0"])

        def _fake_compute_spectral_decomposition(*args, **kwargs):
            return {"root": 1}, {"root": np.array([[1.0]])}, {}

        monkeypatch.setattr(
            spectral_module,
            "compute_spectral_decomposition",
            _fake_compute_spectral_decomposition,
        )

        with pytest.raises(ValueError, match="matching PCA projection/eigenvalue node keys"):
            spectral_module.compute_child_parent_spectral_context(tree, leaf_data)

    def test_gate2_spectral_context_requires_projection_rows_to_match_eigenvalues(
        self, monkeypatch
    ):
        """Gate 2 PCA projection row count must match the whitening eigenvalues."""
        import kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context as spectral_module
        import networkx as nx

        tree = nx.DiGraph()
        tree.add_edges_from([("root", "L0"), ("root", "L1")])
        for leaf in ["L0", "L1"]:
            tree.nodes[leaf]["label"] = leaf
            tree.nodes[leaf]["is_leaf"] = True
        leaf_data = pd.DataFrame([[0.0], [1.0]], index=["L0", "L1"], columns=["F0"])

        def _fake_compute_spectral_decomposition(*args, **kwargs):
            return (
                {"root": 2},
                {"root": np.array([[1.0]])},
                {"root": np.array([1.0, 0.5])},
            )

        monkeypatch.setattr(
            spectral_module,
            "compute_spectral_decomposition",
            _fake_compute_spectral_decomposition,
        )

        with pytest.raises(ValueError, match="projection/eigenvalue row count mismatch"):
            spectral_module.compute_child_parent_spectral_context(tree, leaf_data)

    def test_single_active_feature_spectral_path_returns_coordinate_projection(self):
        """A one-active-feature node is already a valid 1D spectral problem."""
        import networkx as nx
        from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
            compute_spectral_decomposition,
        )

        tree = nx.DiGraph()
        tree.add_edges_from([("root", "L0"), ("root", "L1"), ("root", "L2")])
        for leaf in ["L0", "L1", "L2"]:
            tree.nodes[leaf]["label"] = leaf
            tree.nodes[leaf]["is_leaf"] = True
        tree.nodes["root"]["is_leaf"] = False
        tree.nodes["root"]["distribution"] = np.array([1.0 / 3.0, 0.0])

        leaf_data = pd.DataFrame(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            index=["L0", "L1", "L2"],
            columns=["F0", "F1"],
        )

        spectral_dimensions, pca_projections, pca_eigenvalues = (
            compute_spectral_decomposition(
                tree,
                leaf_data,
                minimum_projection_dimension=2,
                include_internal=False,
            )
        )

        assert spectral_dimensions["root"] == 1
        np.testing.assert_array_equal(pca_projections["root"], np.array([[1.0, 0.0]]))
        np.testing.assert_array_equal(pca_eigenvalues["root"], np.array([1.0]))

    def test_spectral_decomposition_requires_leaf_data_for_every_leaf_label(self):
        """Missing leaf rows must fail instead of silently shrinking a subtree."""
        import networkx as nx
        from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
            compute_spectral_decomposition,
        )

        tree = nx.DiGraph()
        tree.add_edges_from([("root", "L0"), ("root", "L1")])
        for leaf in ["L0", "L1"]:
            tree.nodes[leaf]["label"] = leaf
            tree.nodes[leaf]["is_leaf"] = True

        leaf_data = pd.DataFrame([[0.0]], index=["L0"], columns=["F0"])

        with pytest.raises(ValueError, match="missing from leaf_data"):
            compute_spectral_decomposition(tree, leaf_data)

    def test_invalid_spectral_job_env_does_not_fall_back_to_auto(self, monkeypatch):
        """Invalid KL_TE_N_JOBS should fail instead of silently using auto workers."""
        from kl_clustering_analysis.hierarchy_analysis.statistics.projection.spectral.marchenko_pastur import (
            _get_n_jobs,
        )

        monkeypatch.setenv("KL_TE_N_JOBS", "not-an-int")

        with pytest.raises(ValueError):
            _get_n_jobs(16)


# =============================================================================
# Fix 2: Non-binary skipped flag tests
# =============================================================================


class TestNonBinarySkippedFlag:
    """Verify non-binary and leaf nodes are marked as Sibling_Divergence_Skipped."""

    def _build_simple_tree(self):
        """Build a small tree with binary and non-binary structure.

        Tree structure:
            root (N4)
           /         \\
         N2           N3
        /  \\         /  \\
       L0   L1      L2   L3

        All nodes are binary, all leaves are L0-L3.
        Leaves should be marked as skipped.
        """
        import networkx as nx

        tree = nx.DiGraph()
        # Build tree
        tree.add_edge("N4", "N2", branch_length=0.3)
        tree.add_edge("N4", "N3", branch_length=0.3)
        tree.add_edge("N2", "L0", branch_length=0.1)
        tree.add_edge("N2", "L1", branch_length=0.1)
        tree.add_edge("N3", "L2", branch_length=0.1)
        tree.add_edge("N3", "L3", branch_length=0.1)

        rng = np.random.default_rng(42)
        d = 20
        for node in ["L0", "L1", "L2", "L3"]:
            tree.nodes[node]["distribution"] = rng.random(d) * 0.5
            tree.nodes[node]["leaf_count"] = 1
            tree.nodes[node]["label"] = node
        for node in ["N2", "N3", "N4"]:
            tree.nodes[node]["distribution"] = rng.random(d) * 0.5
            tree.nodes[node]["leaf_count"] = 4 if node == "N4" else 2
            tree.nodes[node]["label"] = node

        return tree

    def _make_base_df(self, tree):
        """Create a base dataframe with edge test columns filled in."""
        nodes = list(tree.nodes)
        df = pd.DataFrame(index=nodes)
        # Simulate: all children diverge from parent (edge-significant)
        df["Child_Parent_Divergence_Significant"] = True
        df["Child_Parent_Divergence_P_Value_BH"] = 0.01
        df["Child_Parent_Divergence_P_Value"] = 0.01
        df["Child_Parent_Divergence_df"] = 1.0
        df["Child_Parent_Divergence_Invalid"] = False
        df["Child_Parent_Divergence_Tested"] = True
        df["Child_Parent_Divergence_Ancestor_Blocked"] = False
        df.loc[["L2", "L3"], "Child_Parent_Divergence_Significant"] = False
        df.loc[["L2", "L3"], "Child_Parent_Divergence_P_Value_BH"] = 1.0
        df.loc[["L2", "L3"], "Child_Parent_Divergence_P_Value"] = 1.0
        return df

    def test_adjusted_wald_marks_leaves_as_skipped(self):
        """Adjusted Wald annotator should mark leaves as Sibling_Divergence_Skipped."""
        from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.adjusted_wald_annotation.pipeline import (
            annotate_sibling_divergence,
        )

        tree = self._build_simple_tree()
        df = self._make_base_df(tree)
        sibling_parent_ids = ["N4", "N2", "N3"]
        parent_pca = {
            parent: np.eye(20, dtype=float)[:1]
            for parent in sibling_parent_ids
        }
        result = annotate_sibling_divergence(
            tree,
            df,
            sibling_projection_dimensions_from_edge_comparisons={
                parent: 1 for parent in sibling_parent_ids
            },
            parent_principal_component_projections=parent_pca,
            parent_principal_component_eigenvalues={
                parent: np.ones(1, dtype=float)
                for parent in sibling_parent_ids
            },
        )

        for leaf in ["L0", "L1", "L2", "L3"]:
            assert bool(
                result.loc[leaf, "Sibling_Divergence_Skipped"]
            ), f"Leaf {leaf} should be marked as Sibling_Divergence_Skipped"
