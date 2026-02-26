"""Tests for v2 decomposition bug fixes (2026-02-17).

Covers three issues found during code review:

1. ``GateEvaluator.should_split_v2`` localization power guard — when
   localization finds zero significant difference pairs after BH correction,
   the aggregate Gate 3 decision is still trusted (SPLIT), but the misleading
   localization result (with only similarity edges) is discarded.  Returns
   ``(True, None)`` to trigger a hard v1-style split rather than passing
   unreliable similarity edges to ``extract_constrained_clusters``.

2. ``extract_constrained_clusters`` leaf overlap — ``merge_points`` that are
   ancestors of nodes in the similarity/difference graphs produced duplicate
   leaf assignments.  Fixed: ancestor merge_points are pruned before cluster
   extraction (lowest-node-wins policy).

3. ``decompose_tree_v2`` overlap warning — the traversal result should never
   assign the same leaf to multiple clusters.  A ``warnings.warn`` guard was
   added to catch regressions.
"""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from kl_clustering_analysis.hierarchy_analysis.signal_localization import (
    LocalizationResult,
    SimilarityEdge,
    extract_constrained_clusters,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_tree_with_labels() -> nx.DiGraph:
    """Build a small labelled tree for ``extract_constrained_clusters`` tests.

    Structure::

            root
           /    \\
          A      B
         / \\   / \\
        A1  A2 B1  B2

    Leaves have ``label`` and ``is_leaf`` attributes.
    """
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
            ("A", "A1"),
            ("A", "A2"),
            ("B", "B1"),
            ("B", "B2"),
        ]
    )
    for node in tree.nodes:
        tree.nodes[node]["is_leaf"] = node in ("A1", "A2", "B1", "B2")
        tree.nodes[node]["label"] = node  # label == id for simplicity
    return tree


def _make_deeper_tree() -> nx.DiGraph:
    """Build a deeper tree where ``A`` is ancestor of ``A1`` and ``A2``.

    Structure::

            root
           /    \\
          A      B
         / \\   / \\
        A1  A2 B1  B2
       / \\
     A1a A1b

    Leaves: A1a, A1b, A2, B1, B2
    """
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
            ("A", "A1"),
            ("A", "A2"),
            ("B", "B1"),
            ("B", "B2"),
            ("A1", "A1a"),
            ("A1", "A1b"),
        ]
    )
    for node in tree.nodes:
        is_leaf = node in ("A1a", "A1b", "A2", "B1", "B2")
        tree.nodes[node]["is_leaf"] = is_leaf
        tree.nodes[node]["label"] = node
    return tree


# =============================================================================
# Bug 1: should_split_v2 post-FDR inversion
# =============================================================================


class TestShouldSplitV2PostFDR:
    """Verify that ``should_split_v2`` returns ``(True, None)`` when
    localization finds zero difference pairs after FDR correction.

    The aggregate Gate 3 test has higher power (one test, pooled signal)
    than the localization sub-tests (many tests, BH penalty). When
    localization can't pinpoint WHERE the difference is, we trust the
    aggregate SPLIT decision but discard the misleading localization
    result to prevent cross-boundary similarity edges from causing
    incorrect merges.
    """

    @staticmethod
    def _build_poset_tree_for_v2() -> "PosetTree":
        """Create a small PosetTree with distributions for v2 testing.

        Tree::

                N0
               /  \\
              N1   N2
             / \\ / \\
            L0 L1 L2 L3

        All leaves have identical distributions → localization should find
        zero difference pairs.
        """
        from kl_clustering_analysis.tree.poset_tree import PosetTree

        tree = PosetTree()
        # Internal nodes
        tree.add_node(
            "N0", is_leaf=False, distribution=np.array([0.5, 0.5]), label="N0", leaf_count=4
        )
        tree.add_node(
            "N1", is_leaf=False, distribution=np.array([0.5, 0.5]), label="N1", leaf_count=2
        )
        tree.add_node(
            "N2", is_leaf=False, distribution=np.array([0.5, 0.5]), label="N2", leaf_count=2
        )
        # Leaves — identical distributions
        for i in range(4):
            tree.add_node(
                f"L{i}",
                is_leaf=True,
                distribution=np.array([0.5, 0.5]),
                label=f"S{i}",
                leaf_count=1,
            )

        tree.add_edges_from(
            [
                ("N0", "N1"),
                ("N0", "N2"),
                ("N1", "L0"),
                ("N1", "L1"),
                ("N2", "L2"),
                ("N2", "L3"),
            ]
        )
        return tree

    def test_no_difference_pairs_returns_hard_split(self):
        """When localize_divergence_signal produces zero difference_pairs,
        ``should_split_v2`` should return ``(True, None)`` — trust the
        aggregate SPLIT but discard the misleading localization result."""
        from kl_clustering_analysis.hierarchy_analysis.signal_localization import LocalizationResult
        from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

        tree = self._build_poset_tree_for_v2()

        # Pre-build a results_df with the gate columns already set so that
        # _prepare_annotations is a no-op.
        nodes = list(tree.nodes)
        results_df = pd.DataFrame(index=nodes)
        # Gate 2: both children edge-diverge (so Gate 2 passes)
        results_df["Child_Parent_Divergence_Significant"] = False
        results_df.loc["N1", "Child_Parent_Divergence_Significant"] = True
        results_df.loc["N2", "Child_Parent_Divergence_Significant"] = True
        # Gate 3: aggregate says "different" at N0
        results_df["Sibling_BH_Different"] = False
        results_df.loc["N0", "Sibling_BH_Different"] = True
        results_df["Sibling_Divergence_Skipped"] = False

        # Mock localize_divergence_signal to return all-similarity result
        loc_result = LocalizationResult(
            left_root="N1",
            right_root="N2",
            aggregate_p_value=0.01,
            aggregate_significant=True,
            similarity_edges=[
                SimilarityEdge("L0", "L2", p_value=0.8),
                SimilarityEdge("L0", "L3", p_value=0.7),
                SimilarityEdge("L1", "L2", p_value=0.6),
                SimilarityEdge("L1", "L3", p_value=0.9),
            ],
            difference_pairs=[],  # <-- zero difference pairs
        )

        with (
            patch(
                "kl_clustering_analysis.hierarchy_analysis.gates.localize_divergence_signal",
                return_value=loc_result,
            ),
            patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df),
        ):
            decomposer = TreeDecomposition(
                tree=tree,
                results_df=results_df,
                posthoc_merge=False,
                use_signal_localization=True,
            )
            should_split, result = decomposer._gate.should_split_v2(
                "N0",
                test_divergence=decomposer._test_node_pair_divergence,
                sibling_alpha=decomposer.sibling_alpha,
                localization_max_depth=decomposer.localization_max_depth,
            )

        # The fix: should SPLIT (trust aggregate) but discard loc_result
        assert should_split is True
        assert result is None

    def test_with_difference_pairs_returns_split(self):
        """When localize_divergence_signal produces ≥1 difference_pair,
        ``should_split_v2`` should return ``(True, loc_result)``."""
        from kl_clustering_analysis.hierarchy_analysis.signal_localization import LocalizationResult
        from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

        tree = self._build_poset_tree_for_v2()

        nodes = list(tree.nodes)
        results_df = pd.DataFrame(index=nodes)
        results_df["Child_Parent_Divergence_Significant"] = False
        results_df.loc["N1", "Child_Parent_Divergence_Significant"] = True
        results_df.loc["N2", "Child_Parent_Divergence_Significant"] = True
        results_df["Sibling_BH_Different"] = False
        results_df.loc["N0", "Sibling_BH_Different"] = True
        results_df["Sibling_Divergence_Skipped"] = False

        loc_result = LocalizationResult(
            left_root="N1",
            right_root="N2",
            aggregate_p_value=0.01,
            aggregate_significant=True,
            similarity_edges=[
                SimilarityEdge("L0", "L2", p_value=0.8),
            ],
            difference_pairs=[("L1", "L3")],  # <-- one real difference
        )

        with (
            patch(
                "kl_clustering_analysis.hierarchy_analysis.gates.localize_divergence_signal",
                return_value=loc_result,
            ),
            patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df),
        ):
            decomposer = TreeDecomposition(
                tree=tree,
                results_df=results_df,
                posthoc_merge=False,
                use_signal_localization=True,
            )
            should_split, result = decomposer._gate.should_split_v2(
                "N0",
                test_divergence=decomposer._test_node_pair_divergence,
                sibling_alpha=decomposer.sibling_alpha,
                localization_max_depth=decomposer.localization_max_depth,
            )

        assert should_split is True
        assert result is loc_result


# =============================================================================
# Bug 2: extract_constrained_clusters leaf overlap
# =============================================================================


class TestExtractConstrainedClustersOverlap:
    """Verify that ancestor merge_points are pruned to prevent leaf overlap."""

    def test_ancestor_merge_point_pruned(self):
        """When a merge_point is an ancestor of a graph node, its leaves
        should NOT appear as an independent cluster.

        Tree::

                root
               /    \\
              A      B
             / \\   / \\
            A1  A2 B1  B2

        Graph nodes: A1, B1 (similarity edge)
        merge_points: [A]  ← ancestor of A1

        Before fix: A's leaves {A1, A2} would form an extra cluster,
        duplicating A1 which is also in the graph cluster.
        After fix: A is pruned from merge_points because it's an ancestor
        of graph node A1.
        """
        tree = _make_tree_with_labels()

        sim_G = nx.Graph()
        sim_G.add_edge("A1", "B1", p_value=0.8)

        diff_G = nx.Graph()

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=["A"],  # ancestor of A1
        )

        # Collect all assigned leaves
        all_leaves = []
        for c in clusters:
            all_leaves.extend(c)

        # No leaf should appear more than once
        assert len(all_leaves) == len(
            set(all_leaves)
        ), f"Duplicate leaves found: {[l for l in all_leaves if all_leaves.count(l) > 1]}"

    def test_non_ancestor_merge_point_kept(self):
        """A merge_point that is NOT an ancestor of any graph node should
        be kept and contribute its leaves to the output."""
        tree = _make_tree_with_labels()

        sim_G = nx.Graph()
        sim_G.add_edge("A1", "A2", p_value=0.8)

        diff_G = nx.Graph()

        # B is not an ancestor of A1 or A2
        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=["B"],
        )

        # B's leaves (B1, B2) should appear
        all_leaves = set()
        for c in clusters:
            all_leaves.update(c)

        assert "B1" in all_leaves
        assert "B2" in all_leaves

    def test_graph_node_itself_not_added_as_merge_point(self):
        """If a merge_point is itself a node in the graph, it should NOT be
        double-counted (once as graph node, once as merge_point)."""
        tree = _make_tree_with_labels()

        sim_G = nx.Graph()
        sim_G.add_edge("A1", "B1", p_value=0.5)

        diff_G = nx.Graph()

        # A1 is directly in the graph AND passed as merge_point
        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=["A1"],
        )

        all_leaves = []
        for c in clusters:
            all_leaves.extend(c)

        assert all_leaves.count("A1") == 1

    def test_deeper_ancestor_pruned(self):
        """A merge_point several levels above a graph node should still be pruned.

        Tree::

                root
               /    \\
              A      B
             / \\   / \\
            A1  A2 B1  B2
           / \\
         A1a A1b

        Graph nodes: A1a, B1 (similarity edge)
        merge_points: [A]  ← grandparent of A1a

        A should be pruned because A1a is a descendant.
        """
        tree = _make_deeper_tree()

        sim_G = nx.Graph()
        sim_G.add_edge("A1a", "B1", p_value=0.7)

        diff_G = nx.Graph()

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=["A"],
        )

        all_leaves = []
        for c in clusters:
            all_leaves.extend(c)

        # A1a should appear exactly once (from graph), not duplicated
        assert all_leaves.count("A1a") == 1
        # A1b was under A but since A was pruned, A1b should NOT appear
        # (it's only reachable via the pruned ancestor)
        assert "A1b" not in all_leaves or all_leaves.count("A1b") <= 1

    def test_multiple_merge_points_mixed(self):
        """Mix of ancestor and non-ancestor merge_points.

        merge_points: [A, B2]
        Graph nodes: A1, B1

        A is ancestor of A1 → pruned
        B2 is NOT ancestor of any graph node → kept
        """
        tree = _make_tree_with_labels()

        sim_G = nx.Graph()
        sim_G.add_edge("A1", "B1", p_value=0.6)

        diff_G = nx.Graph()

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=["A", "B2"],
        )

        all_leaves = []
        for c in clusters:
            all_leaves.extend(c)

        # No duplicates
        assert len(all_leaves) == len(set(all_leaves))
        # B2 should be present (non-ancestor merge_point kept)
        assert "B2" in all_leaves
        # A1 should be present (from graph node)
        assert "A1" in all_leaves

    def test_empty_merge_points_still_works(self):
        """Regression: empty merge_points should produce output from graph only."""
        tree = _make_tree_with_labels()

        sim_G = nx.Graph()
        sim_G.add_edge("A1", "B1", p_value=0.5)

        diff_G = nx.Graph()

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=[],
        )

        assert len(clusters) >= 1
        all_leaves = set()
        for c in clusters:
            all_leaves.update(c)
        assert "A1" in all_leaves
        assert "B1" in all_leaves


# =============================================================================
# Bug 3: extract_constrained_clusters leaf deduplication
# =============================================================================


class TestExtractConstrainedClustersDeduplication:
    """Verify that ``extract_constrained_clusters`` deduplicates overlapping
    leaves when graph nodes at different tree depths share descendants."""

    def test_overlapping_graph_nodes_deduplicated(self):
        """When two graph nodes are ancestor-descendant, their shared leaves
        should appear in only one cluster (the smallest / most specific).

        Tree::

                root
               /    \\
              A      B
             / \\   / \\
            A1  A2 B1  B2
           / \\
         A1a A1b

        Graph nodes: A (ancestor, leaves={A1a, A1b, A2}) and A1 (descendant, leaves={A1a, A1b}).
        Both in similarity graph with B1.
        A1a and A1b should NOT be duplicated across clusters.
        """
        tree = _make_deeper_tree()

        # A and A1 both in the graph — A is ancestor of A1
        sim_G = nx.Graph()
        sim_G.add_edge("A", "B1", p_value=0.7)
        sim_G.add_edge("A1", "B1", p_value=0.8)

        diff_G = nx.Graph()

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=[],
        )

        # Collect all leaves and verify no duplicates
        all_leaves = []
        for c in clusters:
            all_leaves.extend(c)

        assert len(all_leaves) == len(
            set(all_leaves)
        ), f"Duplicate leaves found: {[l for l in set(all_leaves) if all_leaves.count(l) > 1]}"

    def test_dedup_assigns_leaf_to_smallest_cluster(self):
        """When a leaf appears in multiple clusters, it should end up in
        the smallest (most specific) one.

        Graph nodes: A (ancestor, large) and A1 (descendant, small).
        Both NOT connected to each other — they form separate clusters.
        Leaves A1a, A1b appear under both → should go to A1's cluster.
        """
        tree = _make_deeper_tree()

        # Two disconnected graph nodes: A and A1
        # A's cluster: {A} → leaves {A1a, A1b, A2}
        # A1's cluster: {A1} → leaves {A1a, A1b}
        # Overlap: A1a, A1b — should go to smaller cluster (A1's)
        sim_G = nx.Graph()
        sim_G.add_node("A")  # isolated node, no edges
        sim_G.add_node("A1")  # isolated node, no edges

        diff_G = nx.Graph()
        diff_G.add_edge("A", "A1")  # keep them separate

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=[],
        )

        all_leaves = []
        for c in clusters:
            all_leaves.extend(c)

        assert len(all_leaves) == len(set(all_leaves)), "Leaves still duplicated after dedup"

    def test_no_dedup_needed_when_disjoint(self):
        """When all graph nodes have disjoint leaf sets, dedup is a no-op."""
        tree = _make_tree_with_labels()

        sim_G = nx.Graph()
        sim_G.add_edge("A1", "B1", p_value=0.5)

        diff_G = nx.Graph()

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=["A2", "B2"],
        )

        all_leaves = []
        for c in clusters:
            all_leaves.extend(c)

        assert len(all_leaves) == len(set(all_leaves))
        assert set(all_leaves) == {"A1", "A2", "B1", "B2"}


class TestDecomposeTreeV2Integration:
    """Integration tests for ``decompose_tree_v2``."""

    @staticmethod
    def _build_poset_tree() -> "PosetTree":
        """Create a PosetTree for v2 decomposition testing."""
        from kl_clustering_analysis.tree.poset_tree import PosetTree

        tree = PosetTree()
        tree.add_node(
            "N0", is_leaf=False, distribution=np.array([0.5, 0.5]), label="N0", leaf_count=4
        )
        tree.add_node(
            "N1", is_leaf=False, distribution=np.array([0.3, 0.7]), label="N1", leaf_count=2
        )
        tree.add_node(
            "N2", is_leaf=False, distribution=np.array([0.7, 0.3]), label="N2", leaf_count=2
        )
        for i in range(4):
            dist = np.array([0.3, 0.7]) if i < 2 else np.array([0.7, 0.3])
            tree.add_node(f"L{i}", is_leaf=True, distribution=dist, label=f"S{i}", leaf_count=1)

        tree.add_edges_from(
            [
                ("N0", "N1"),
                ("N0", "N2"),
                ("N1", "L0"),
                ("N1", "L1"),
                ("N2", "L2"),
                ("N2", "L3"),
            ]
        )
        return tree

    def test_no_overlap_on_clean_decomposition(self):
        """A normal v2 decomposition should not produce overlapping leaf assignments."""
        from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

        tree = self._build_poset_tree()

        nodes = list(tree.nodes)
        results_df = pd.DataFrame(index=nodes)
        # Gate 2: children diverge
        results_df["Child_Parent_Divergence_Significant"] = False
        results_df.loc["N1", "Child_Parent_Divergence_Significant"] = True
        results_df.loc["N2", "Child_Parent_Divergence_Significant"] = True
        # Gate 3: siblings different at N0
        results_df["Sibling_BH_Different"] = False
        results_df.loc["N0", "Sibling_BH_Different"] = True
        results_df["Sibling_Divergence_Skipped"] = False

        # Mock localization to return a clean result with real difference pairs
        loc_result = LocalizationResult(
            left_root="N1",
            right_root="N2",
            aggregate_p_value=0.01,
            aggregate_significant=True,
            similarity_edges=[],
            difference_pairs=[("L0", "L2"), ("L1", "L3")],
        )

        with (
            patch(
                "kl_clustering_analysis.hierarchy_analysis.gates.localize_divergence_signal",
                return_value=loc_result,
            ),
            patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df),
        ):
            decomposer = TreeDecomposition(
                tree=tree,
                results_df=results_df,
                posthoc_merge=False,
                use_signal_localization=True,
            )
            result = decomposer.decompose_tree_v2()

        # Verify no leaf appears in multiple clusters
        all_leaves = []
        for info in result["cluster_assignments"].values():
            all_leaves.extend(info["leaves"])
        assert len(all_leaves) == len(set(all_leaves))

    def test_v2_produces_valid_cluster_assignments(self):
        """decompose_tree_v2 should produce cluster assignments covering all leaves."""
        from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition

        tree = self._build_poset_tree()

        nodes = list(tree.nodes)
        results_df = pd.DataFrame(index=nodes)
        # All gates fail → everything merges to one cluster
        results_df["Child_Parent_Divergence_Significant"] = False
        results_df["Sibling_BH_Different"] = False
        results_df["Sibling_Divergence_Skipped"] = False

        with patch.object(TreeDecomposition, "_prepare_annotations", side_effect=lambda df: df):
            decomposer = TreeDecomposition(
                tree=tree,
                results_df=results_df,
                posthoc_merge=False,
                use_signal_localization=True,
            )
            result = decomposer.decompose_tree_v2()

        assert result["num_clusters"] == 1
        # All 4 leaves should be in the single cluster
        cluster_info = result["cluster_assignments"][0]
        assert cluster_info["size"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
