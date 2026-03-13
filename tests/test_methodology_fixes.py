"""Tests for the three methodology fixes:

1. Fix 1: spectral_k floor raised from 1 to 4
2. Fix 2: Non-binary and leaf nodes marked as Sibling_Divergence_Skipped=True
3. Fix 3: Shared Satterthwaite helper (compute_projected_pvalue)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from kl_clustering_analysis.hierarchy_analysis.statistics.projection.satterthwaite import (
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

    def test_spectral_minimum_projection_dimension_decoupled_from_global(self):
        """The spectral path should use config.SPECTRAL_MINIMUM_DIMENSION, not the global JL minimum_projection_dimension."""
        import inspect

        from kl_clustering_analysis.hierarchy_analysis.statistics.child_parent_divergence import (
            compute_child_parent_spectral_context,
        )

        source = inspect.getsource(compute_child_parent_spectral_context)
        # The spectral path reads SPECTRAL_MINIMUM_DIMENSION from config instead of
        # forwarding the global JL-derived minimum_projection_dimension.
        assert "SPECTRAL_MINIMUM_DIMENSION" in source
        # The old pattern that forwarded the global minimum_projection_dimension should be gone.
        assert "minimum_projection_dimension if isinstance(minimum_projection_dimension, int) else 4" not in source
        assert "minimum_projection_dimension if isinstance(minimum_projection_dimension, int) else 1" not in source

    def test_spectral_minimum_projection_dimension_config_exists(self):
        """config.SPECTRAL_MINIMUM_DIMENSION must exist and be a small integer."""
        from kl_clustering_analysis import config

        assert hasattr(config, "SPECTRAL_MINIMUM_DIMENSION")
        assert isinstance(config.SPECTRAL_MINIMUM_DIMENSION, int)
        assert config.SPECTRAL_MINIMUM_DIMENSION >= 1
        assert config.SPECTRAL_MINIMUM_DIMENSION <= 4  # must be small — the whole point


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

    def test_base_wald_marks_leaves_as_skipped(self):
        """Base Wald annotator should mark leaves as Sibling_Divergence_Skipped."""
        from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
            annotate_sibling_divergence,
        )

        tree = self._build_simple_tree()
        df = self._make_base_df(tree)
        result = annotate_sibling_divergence(tree, df)

        # Leaves should be skipped (they have no children to test)
        for leaf in ["L0", "L1", "L2", "L3"]:
            assert (
                result.loc[leaf, "Sibling_Divergence_Skipped"] is True
                or result.loc[leaf, "Sibling_Divergence_Skipped"] == True
            ), f"Leaf {leaf} should be marked as Sibling_Divergence_Skipped"

    def test_adjusted_wald_marks_leaves_as_skipped(self):
        """Adjusted Wald annotator should mark leaves as Sibling_Divergence_Skipped."""
        from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_adjusted_wald import (
            annotate_sibling_divergence_adjusted,
        )

        tree = self._build_simple_tree()
        df = self._make_base_df(tree)
        result = annotate_sibling_divergence_adjusted(tree, df)

        for leaf in ["L0", "L1", "L2", "L3"]:
            assert (
                result.loc[leaf, "Sibling_Divergence_Skipped"] is True
                or result.loc[leaf, "Sibling_Divergence_Skipped"] == True
            ), f"Leaf {leaf} should be marked as Sibling_Divergence_Skipped"

    def test_collect_test_arguments_returns_non_binary(self):
        """_collect_test_arguments should return non-binary nodes separately."""
        from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
            _collect_test_arguments,
        )

        tree = self._build_simple_tree()
        df = self._make_base_df(tree)

        parents, args, skipped, non_binary = _collect_test_arguments(tree, df)

        # Leaves (L0-L3) have 0 children -> non-binary
        assert set(non_binary) >= {
            "L0",
            "L1",
            "L2",
            "L3",
        }, f"Leaves should be in non_binary list, got {non_binary}"
