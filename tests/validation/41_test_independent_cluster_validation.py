"""Standalone sanity checks for TreeDecomposition behavior.

These tests avoid the heavier fixtures in ``test_cluster_validation.py`` and only
use NetworkX + pandas to build a minimal hierarchy so the module can be verified
in isolation.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _make_binary_tree() -> tuple[PosetTree, pd.DataFrame]:
    """Construct a tiny binary tree plus stats DataFrame for testing."""

    tree = PosetTree()
    tree.add_node("root", is_leaf=False, distribution=np.array([0.5, 0.5]))
    tree.add_node("L", is_leaf=True, label="L", distribution=np.array([0.7, 0.3]))
    tree.add_node("R", is_leaf=True, label="R", distribution=np.array([0.3, 0.7]))
    tree.add_edge("root", "L")
    tree.add_edge("root", "R")

    stats = pd.DataFrame.from_dict(
        {
            "root": {
                "distribution": np.array([0.5, 0.5]),
                "is_leaf": False,
                "Child_Parent_Divergence_Significant": False,
                "Sibling_BH_Different": True,
                "Sibling_Divergence_Skipped": False,
                "Sibling_Divergence_P_Value_Corrected": 0.01,
            },
            "L": {
                "distribution": np.array([0.7, 0.3]),
                "is_leaf": True,
                "Child_Parent_Divergence_Significant": True,
                "Sibling_BH_Different": False,
                "Sibling_Divergence_Skipped": False,
            },
            "R": {
                "distribution": np.array([0.3, 0.7]),
                "is_leaf": True,
                "Child_Parent_Divergence_Significant": True,
                "Sibling_BH_Different": False,
                "Sibling_Divergence_Skipped": False,
            },
        },
        orient="index",
    )
    return tree, stats


class TestIndependentClusterValidation(unittest.TestCase):
    """Minimal, dependency-light validation of TreeDecomposition."""

    def test_split_occurs_when_all_gates_pass(self) -> None:
        tree, stats = _make_binary_tree()
        # Bypass annotation pipeline — this test controls gate columns directly.
        with patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df):
            # Disable post-hoc merge to test base decomposition behavior
            results = tree.decompose(
                results_df=stats, posthoc_merge=False, use_signal_localization=False
            )

        self.assertEqual(results["num_clusters"], 2)
        leaves = sorted(
            tuple(sorted(cluster["leaves"])) for cluster in results["cluster_assignments"].values()
        )
        self.assertEqual(leaves, [("L",), ("R",)])
        print("Split test clusters:", leaves)

    def test_merge_when_sibling_divergence_fails(self) -> None:
        tree, stats = _make_binary_tree()
        stats.loc["root", "Sibling_BH_Different"] = False
        # Bypass annotation pipeline — this test controls gate columns directly.
        with patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df):
            results = tree.decompose(results_df=stats, use_signal_localization=False)

        self.assertEqual(results["num_clusters"], 1)
        cluster = next(iter(results["cluster_assignments"].values()))
        self.assertCountEqual(cluster["leaves"], ["L", "R"])
        print("Merge test cluster:", cluster["leaves"])


if __name__ == "__main__":
    unittest.main()
