"""Regression contracts for edge-gate spectral context construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestSpectralContextRegressions:
    """Verify edge-gate spectral context invariants."""

    def test_edge_gate_spectral_minimum_projection_dimension_is_fixed(self, monkeypatch):
        """The edge gate should pass the fixed spectral floor into the spectral estimator."""
        import networkx as nx
        import tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context as spectral_module

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
            from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.spectral_decomposition_result import (
                SpectralDecompositionResult,
            )

            captured.append(kwargs["minimum_projection_dimension"])
            return SpectralDecompositionResult(
                test_projection_dimensions_by_node={},
                raw_mp_signal_counts_by_node={},
                effective_independent_rows_by_node={},
                mp_threshold_rows_by_node={},
                principal_component_projections_by_node={},
                principal_component_eigenvalues_by_node={},
            )

        monkeypatch.setattr(
            spectral_module,
            "compute_spectral_decomposition",
            _fake_compute_spectral_decomposition,
        )

        spectral_module.compute_child_parent_spectral_context(tree, leaf_data)

        assert captured == [2]

    def test_edge_gate_spectral_context_requires_paired_projection_eigenvalue_keys(
        self, monkeypatch
    ):
        """Edge-gate PCA projections and eigenvalues must be keyed identically."""
        import networkx as nx
        import tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context as spectral_module

        tree = nx.DiGraph()
        tree.add_edges_from([("root", "L0"), ("root", "L1")])
        for leaf in ["L0", "L1"]:
            tree.nodes[leaf]["label"] = leaf
            tree.nodes[leaf]["is_leaf"] = True
        leaf_data = pd.DataFrame([[0.0], [1.0]], index=["L0", "L1"], columns=["F0"])

        def _fake_compute_spectral_decomposition(*args, **kwargs):
            from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.spectral_decomposition_result import (
                SpectralDecompositionResult,
            )

            return SpectralDecompositionResult(
                test_projection_dimensions_by_node={"root": 1},
                raw_mp_signal_counts_by_node={"root": 1},
                effective_independent_rows_by_node={"root": 2},
                mp_threshold_rows_by_node={"root": 2},
                principal_component_projections_by_node={"root": np.array([[1.0]])},
                principal_component_eigenvalues_by_node={},
            )

        monkeypatch.setattr(
            spectral_module,
            "compute_spectral_decomposition",
            _fake_compute_spectral_decomposition,
        )

        with pytest.raises(ValueError, match="matching PCA projection/eigenvalue node keys"):
            spectral_module.compute_child_parent_spectral_context(tree, leaf_data)

    def test_edge_gate_spectral_context_requires_projection_rows_to_match_eigenvalues(
        self, monkeypatch
    ):
        """Edge-gate PCA projection row count must match the whitening eigenvalues."""
        import networkx as nx
        import tree_break_selection.hierarchy_analysis.statistics.child_parent_divergence.child_parent_divergence_annotation.spectral_context as spectral_module

        tree = nx.DiGraph()
        tree.add_edges_from([("root", "L0"), ("root", "L1")])
        for leaf in ["L0", "L1"]:
            tree.nodes[leaf]["label"] = leaf
            tree.nodes[leaf]["is_leaf"] = True
        leaf_data = pd.DataFrame([[0.0], [1.0]], index=["L0", "L1"], columns=["F0"])

        def _fake_compute_spectral_decomposition(*args, **kwargs):
            from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.spectral_decomposition_result import (
                SpectralDecompositionResult,
            )

            return SpectralDecompositionResult(
                test_projection_dimensions_by_node={"root": 2},
                raw_mp_signal_counts_by_node={"root": 2},
                effective_independent_rows_by_node={"root": 2},
                mp_threshold_rows_by_node={"root": 2},
                principal_component_projections_by_node={"root": np.array([[1.0]])},
                principal_component_eigenvalues_by_node={"root": np.array([1.0, 0.5])},
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
        from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
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

        spectral_decomposition = compute_spectral_decomposition(
            tree,
            leaf_data,
            minimum_projection_dimension=2,
        )

        assert spectral_decomposition.test_projection_dimensions_by_node["root"] == 1
        assert spectral_decomposition.raw_mp_signal_counts_by_node["root"] == 0
        assert spectral_decomposition.effective_independent_rows_by_node["root"] == 3
        assert spectral_decomposition.mp_threshold_rows_by_node["root"] == 3
        np.testing.assert_array_equal(
            spectral_decomposition.principal_component_projections_by_node["root"],
            np.array([[1.0, 0.0]]),
        )
        np.testing.assert_allclose(
            spectral_decomposition.principal_component_eigenvalues_by_node["root"],
            np.array([1.0]),
            atol=1e-10,
        )

    def test_branch_length_internal_state_changes_internal_spectral_rows(self):
        """Branch-length state rows are a separate diagnostic from empirical barycenters."""
        import networkx as nx
        from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
            INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE,
            compute_spectral_decomposition,
        )

        tree = nx.DiGraph()
        tree.add_edge("root", "I0", branch_length=1.0)
        tree.add_edge("root", "L2", branch_length=1.0)
        tree.add_edge("I0", "L0", branch_length=0.01)
        tree.add_edge("I0", "L1", branch_length=10.0)
        for leaf, distribution in {
            "L0": np.array([0.0, 0.0]),
            "L1": np.array([1.0, 0.0]),
            "L2": np.array([0.0, 1.0]),
        }.items():
            tree.nodes[leaf]["label"] = leaf
            tree.nodes[leaf]["is_leaf"] = True
            tree.nodes[leaf]["distribution"] = distribution
        tree.nodes["I0"]["is_leaf"] = False
        tree.nodes["root"]["is_leaf"] = False
        tree.nodes["I0"]["distribution"] = np.array([0.5, 0.0])
        tree.nodes["root"]["distribution"] = np.array([1.0 / 3.0, 1.0 / 3.0])

        leaf_data = pd.DataFrame(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            index=["L0", "L1", "L2"],
            columns=["F0", "F1"],
        )

        empirical = compute_spectral_decomposition(
            tree,
            leaf_data,
            minimum_projection_dimension=1,
            include_internal_barycenters=True,
        )
        branch_length = compute_spectral_decomposition(
            tree,
            leaf_data,
            minimum_projection_dimension=1,
            include_internal_barycenters=True,
            internal_distribution_mode=INTERNAL_DISTRIBUTION_BRANCH_LENGTH_STATE,
        )

        assert empirical.effective_independent_rows_by_node["root"] == 3
        assert branch_length.effective_independent_rows_by_node["root"] == 3
        assert empirical.mp_threshold_rows_by_node["root"] == 3
        assert branch_length.mp_threshold_rows_by_node["root"] == 3
        assert empirical.descendant_leaf_row_counts_by_node["root"] == 3
        assert empirical.internal_distribution_row_counts_by_node["root"] == 1
        assert empirical.spectral_matrix_row_counts_by_node["root"] == 4
        assert empirical.descendant_leaf_row_counts_by_node["I0"] == 2
        assert empirical.internal_distribution_row_counts_by_node["I0"] == 0
        assert empirical.spectral_matrix_row_counts_by_node["I0"] == 2
        assert branch_length.descendant_leaf_row_counts_by_node["root"] == 3
        assert branch_length.internal_distribution_row_counts_by_node["root"] == 1
        assert branch_length.spectral_matrix_row_counts_by_node["root"] == 4
        assert not np.allclose(
            empirical.full_component_eigenvalues_by_node["root"],
            branch_length.full_component_eigenvalues_by_node["root"],
        )

    def test_spectral_decomposition_requires_leaf_data_for_every_leaf_label(self):
        """Missing leaf rows must fail instead of silently shrinking a subtree."""
        import networkx as nx
        from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.tree_estimator import (
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
        """Invalid TBS_N_JOBS should fail instead of silently using auto workers."""
        from tree_break_selection.hierarchy_analysis.statistics.projection.spectral.marchenko_pastur import (
            _get_n_jobs,
        )

        monkeypatch.setenv("TBS_N_JOBS", "not-an-int")

        with pytest.raises(ValueError):
            _get_n_jobs(16)
