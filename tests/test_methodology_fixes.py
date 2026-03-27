"""Tests for the three methodology fixes:

1. Fix 1: spectral_k floor raised from 1 to 4
2. Fix 2: Non-binary and leaf nodes marked as Sibling_Divergence_Skipped=True
3. Fix 3: Shared Satterthwaite helper (compute_projected_pvalue)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis.hierarchy_analysis.statistics.projection.chi2_pvalue import (
    compute_projected_pvalue,
)

# =============================================================================
# Fix 3: Shared Satterthwaite helper tests
# =============================================================================


class TestComputeProjectedPvalue:
    """Tests for the shared compute_projected_pvalue helper."""

    def test_plain_chi2_no_eigenvalues(self):
        """Without eigenvalues, should return plain chi-square(k)."""
        rng = np.random.default_rng(42)
        projected = rng.standard_normal(10)
        stat, df, pval = compute_projected_pvalue(projected, 10, eigenvalues=None)
        assert df == 10.0
        expected_stat = float(np.sum(projected**2))
        assert abs(stat - expected_stat) < 1e-10
        expected_pval = float(chi2.sf(expected_stat, df=10))
        assert abs(pval - expected_pval) < 1e-10

    def test_plain_chi2_empty_eigenvalues(self):
        """Empty eigenvalue array should behave like None."""
        rng = np.random.default_rng(42)
        projected = rng.standard_normal(5)
        stat1, df1, pval1 = compute_projected_pvalue(projected, 5, eigenvalues=None)
        stat2, df2, pval2 = compute_projected_pvalue(projected, 5, eigenvalues=np.array([]))
        assert stat1 == stat2
        assert df1 == df2
        assert pval1 == pval2

    def test_whitened_mode(self):
        """Eigenvalue whitening: T = Σ w²/λ ~ χ²(k)."""
        projected = np.array([1.0, 2.0, 3.0])
        eigenvalues = np.array([2.0, 1.0, 0.5])
        stat, df, pval = compute_projected_pvalue(projected, 3, eigenvalues=eigenvalues)
        expected_stat = 1.0 / 2.0 + 4.0 / 1.0 + 9.0 / 0.5
        assert abs(stat - expected_stat) < 1e-10
        assert df == 3.0
        assert abs(pval - float(chi2.sf(expected_stat, df=3))) < 1e-10


# =============================================================================
# Fix 1: spectral_k floor test
# =============================================================================


class TestSpectralKFloor:
    """Verify the spectral path uses its own small floor (SPECTRAL_MINIMUM_DIMENSION)."""

    def test_spectral_minimum_projection_dimension_decoupled_from_global(self, monkeypatch):
        """The spectral path should pass config.SPECTRAL_MINIMUM_DIMENSION into the spectral estimator."""
        import networkx as nx

        import kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence.child_parent_spectral_decomposition as spectral_module
        from kl_clustering_analysis import config
        from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
            compute_child_parent_spectral_context,
        )

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

        monkeypatch.setattr(config, "SPECTRAL_MINIMUM_DIMENSION", 3)
        monkeypatch.setattr(
            spectral_module,
            "compute_spectral_decomposition",
            _fake_compute_spectral_decomposition,
        )

        compute_child_parent_spectral_context(tree, leaf_data)

        assert captured == [3]

    def test_spectral_minimum_projection_dimension_config_exists(self):
        """config.SPECTRAL_MINIMUM_DIMENSION must exist and be a small integer."""
        from kl_clustering_analysis import config

        assert hasattr(config, "SPECTRAL_MINIMUM_DIMENSION")
        assert isinstance(config.SPECTRAL_MINIMUM_DIMENSION, int)
        assert config.SPECTRAL_MINIMUM_DIMENSION >= 1
        assert config.SPECTRAL_MINIMUM_DIMENSION <= 4  # must be small — the whole point

    def test_single_feature_subtree_audit_blocks_low_information_subtrees_when_tree_is_dangerous(
        self, monkeypatch
    ):
        """Low-leverage one-active nodes should be blocked when they dominate the tree."""
        import networkx as nx

        from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence._single_feature_subtree_policy import (
            _build_single_feature_subtree_audit,
        )

        tree = nx.DiGraph()
        tree.add_edges_from(
            [
                ("root", "A"),
                ("root", "B"),
                ("A", "A1"),
                ("A", "L2"),
                ("A1", "L0"),
                ("A1", "L1"),
                ("B", "L3"),
                ("B", "L4"),
            ]
        )

        for leaf in ["L0", "L1", "L2", "L3", "L4"]:
            tree.nodes[leaf]["label"] = leaf
            tree.nodes[leaf]["is_leaf"] = True

        tree.nodes["A1"]["distribution"] = np.array([0.5, 0.0])
        tree.nodes["A"]["distribution"] = np.array([0.3, 0.0])
        tree.nodes["B"]["distribution"] = np.array([0.0, 1.0])
        tree.nodes["root"]["distribution"] = np.array([0.3, 0.4])

        leaf_data = pd.DataFrame(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 2.0],
            ],
            index=["L0", "L1", "L2", "L3", "L4"],
            columns=["F0", "F1"],
        )

        monkeypatch.setattr(
            "kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence._single_feature_subtree_policy._find_low_variance_ratio_threshold",
            lambda ratios: (True, 0.5),
        )

        audit = _build_single_feature_subtree_audit(
            tree,
            leaf_data,
            node_pca_projections={},
            node_pca_eigenvalues={
                "root": np.array([1.0]),
                "A": np.array([1.0]),
            },
        )

        assert audit["candidate_nodes"] == 3
        assert audit["dangerous_tree"] is True
        assert audit["blocked_node_ids"] == ["A", "A1"]
        assert audit["allowed_node_ids"] == ["B"]

        node_audit = {node["node_id"]: node for node in audit["nodes"]}
        assert node_audit["A"]["is_low_leverage"] is True
        assert node_audit["A1"]["is_low_leverage"] is True
        assert node_audit["B"]["is_low_leverage"] is False
        assert node_audit["A"]["allowed_one_active_1d"] is False
        assert node_audit["A1"]["allowed_one_active_1d"] is False
        assert node_audit["B"]["allowed_one_active_1d"] is True


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
            tree.nodes[node]["n_descendant_leaves"] = 5
            tree.nodes[node]["label"] = node
        for node in ["N2", "N3", "N4"]:
            tree.nodes[node]["distribution"] = rng.random(d) * 0.5
            tree.nodes[node]["n_descendant_leaves"] = 10
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
        return df

    def test_adjusted_wald_marks_leaves_as_skipped(self):
        """Adjusted Wald annotator should mark leaves as Sibling_Divergence_Skipped."""
        from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence import (
            annotate_sibling_divergence_adjusted,
        )

        tree = self._build_simple_tree()
        df = self._make_base_df(tree)
        result = annotate_sibling_divergence_adjusted(tree, df)

        for leaf in ["L0", "L1", "L2", "L3"]:
            assert bool(
                result.loc[leaf, "Sibling_Divergence_Skipped"]
            ), f"Leaf {leaf} should be marked as Sibling_Divergence_Skipped"
