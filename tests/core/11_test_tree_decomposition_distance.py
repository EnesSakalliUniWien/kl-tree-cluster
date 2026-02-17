"""Tests for tree decomposition distance calculations."""

import pytest
import networkx as nx
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import (
    TreeDecomposition,
)


class TestTreeDecompositionDistance:
    @pytest.fixture
    def weighted_tree(self):
        """Create a tree with branch lengths for testing.

           root
           /  \\ (len=10)
          A    B (len=5)
         /      \\ (len=2)
        A1       B1

        Dist(A, B) = Dist(A, root) + Dist(B, root) = 1 + 10 = 11?
        Wait, let's define explicitly.

        root -> A (1.0)
        root -> B (5.0)
        A -> A1 (2.0)
        B -> B1 (3.0)
        """
        tree = nx.DiGraph()
        # Mock distribution/samples for initialization
        defaults = {
            "distribution": np.array([0.5, 0.5]),
            "is_leaf": False,
            "sample_size": 10,
        }

        nodes = ["root", "A", "B", "A1", "B1"]
        for n in nodes:
            tree.add_node(n, **defaults)

        tree.nodes["A1"]["is_leaf"] = True
        tree.nodes["B1"]["is_leaf"] = True

        # Edges with weights
        tree.add_edge("root", "A", branch_length=1.0)
        tree.add_edge("root", "B", branch_length=5.0)
        tree.add_edge("A", "A1", branch_length=2.0)
        tree.add_edge("B", "B1", branch_length=3.0)

        # Add required PosetTree methods
        tree.compute_descendant_sets = MagicMock(
            return_value={
                "root": {"A1", "B1"},
                "A": {"A1"},
                "B": {"B1"},
                "A1": {"A1"},
                "B1": {"B1"},
            }
        )
        tree.find_lca_for_set = MagicMock(return_value="root")

        # Mock find_lca for the specific pair tests
        # This is strictly for the test logic usage, actual find_lca logic is in PosetTree
        def lca_side_effect(u, v):
            if u == v:
                return u
            if u in ("A1", "A") and v in ("B1", "B"):
                return "root"
            if u in ("B1", "B") and v in ("A1", "A"):
                return "root"
            if u == "A1" and v == "A":
                return "A"
            if u == "root":
                return "root"
            return "root"  # Default fallback

        tree.find_lca = MagicMock(side_effect=lca_side_effect)

        return tree

    def test_divergence_uses_path_distances(self, weighted_tree):
        """Test that _test_node_pair_divergence passes correct path lengths."""

        # Initialize decomposer with dummy results_df
        nodes = ["root", "A", "B", "A1", "B1"]
        dummy_df = pd.DataFrame(index=nodes)
        dummy_df["Child_Parent_Divergence_Significant"] = True
        dummy_df["Sibling_BH_Different"] = True
        dummy_df["Sibling_Divergence_Skipped"] = False

        decomposer = TreeDecomposition(weighted_tree, results_df=dummy_df)

        # Mock the low-level test function
        with patch(
            "kl_clustering_analysis.hierarchy_analysis.tree_decomposition.sibling_divergence_test"
        ) as mock_test:
            mock_test.return_value = (1.0, 1.0, 0.05)

            # Test 1: Siblings A vs B
            # LCA = root
            # Dist(A, root) = 1.0
            # Dist(B, root) = 5.0
            decomposer._test_node_pair_divergence("A", "B")

            args, kwargs = mock_test.call_args
            # Check positional args for distributions/samples (not focus here)
            # Check expected kwarg branch lengths (or if passed as positional, check index)
            # Signature: (left_dist, right_dist, n_left, n_right, branch_length_left, branch_length_right)

            # Since implementation might pass as kwargs or positional logic in decomposition,
            # we need to align with how we implement it.
            # Plan is to pass as named arguments to sibling_divergence_test wrapper?
            # Actually, sibling_divergence_test signature is:
            # (left_dist, right_dist, n_left, n_right, branch_length_left=None, branch_length_right=None)

            # Let's inspect call args.
            assert kwargs.get("branch_length_left") == 1.0
            assert kwargs.get("branch_length_right") == 5.0

            # Test 2: Non-siblings A1 vs B1
            # LCA = root
            # Dist(A1, root) = Dist(A1, A) + Dist(A, root) = 2.0 + 1.0 = 3.0
            # Dist(B1, root) = Dist(B1, B) + Dist(B, root) = 3.0 + 5.0 = 8.0
            decomposer._test_node_pair_divergence("A1", "B1")

            args, kwargs = mock_test.call_args
            assert kwargs.get("branch_length_left") == 3.0
            assert kwargs.get("branch_length_right") == 8.0
