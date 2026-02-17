"""Tests for signal localization module.

Tests the localize_divergence_signal function and related utilities
for finding WHERE divergence originates in a tree.
"""

import pytest
import networkx as nx
import numpy as np

from kl_clustering_analysis.hierarchy_analysis.signal_localization import (
    SimilarityEdge,
    LocalizationResult,
    localize_divergence_signal,
    merge_difference_graphs,
    extract_constrained_clusters,
    build_cross_boundary_similarity,
    merge_similarity_graphs,
)


# =============================================================================
# Fixtures
# =============================================================================


def _make_test_tree() -> nx.DiGraph:
    """Create a simple test tree with distributions.
    
    Structure::
    
        root
        /  \\
       A    B
      / \\  / \\
     A1 A2 B1 B2
     
    Where A1 ≈ B1 (similar) but A2 ≠ B2 (different)
    """
    tree = nx.DiGraph()

    # Add edges
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

    # A1 and B1 have similar distributions (both favor feature 0)
    # A2 and B2 have different distributions (A2 favors 1, B2 favors 2)
    distributions = {
        "root": np.array([0.25, 0.25, 0.25, 0.25]),
        "A": np.array([0.3, 0.3, 0.2, 0.2]),
        "B": np.array([0.3, 0.2, 0.3, 0.2]),
        "A1": np.array([0.5, 0.2, 0.15, 0.15]),  # Similar to B1
        "A2": np.array([0.1, 0.7, 0.1, 0.1]),  # Different from B2
        "B1": np.array([0.5, 0.2, 0.15, 0.15]),  # Similar to A1
        "B2": np.array([0.1, 0.1, 0.7, 0.1]),  # Different from A2
    }

    sample_sizes = {
        "root": 400,
        "A": 200,
        "B": 200,
        "A1": 100,
        "A2": 100,
        "B1": 100,
        "B2": 100,
    }

    for node in tree.nodes:
        tree.nodes[node]["distribution"] = distributions[node]
        tree.nodes[node]["sample_size"] = sample_sizes[node]
        tree.nodes[node]["is_leaf"] = node in ["A1", "A2", "B1", "B2"]

    return tree


def _make_mock_test_func(p_values: dict):
    """Create a mock test function with predetermined p-values."""

    def test_func(node_a: str, node_b: str):
        key = frozenset([node_a, node_b])
        p = p_values.get(key, 0.5)
        return (1.0, 1.0, p)  # (stat, df, p_value)

    return test_func


# =============================================================================
# Tests for SimilarityEdge
# =============================================================================


class TestSimilarityEdge:
    def test_creation(self):
        edge = SimilarityEdge("A", "B", 0.5)
        assert edge.node_a == "A"
        assert edge.node_b == "B"
        assert edge.p_value == 0.5

    def test_equality_is_order_independent(self):
        edge1 = SimilarityEdge("A", "B", 0.5)
        edge2 = SimilarityEdge("B", "A", 0.5)
        assert edge1 == edge2

    def test_hash_is_order_independent(self):
        edge1 = SimilarityEdge("A", "B", 0.5)
        edge2 = SimilarityEdge("B", "A", 0.5)
        assert hash(edge1) == hash(edge2)

    def test_different_edges_not_equal(self):
        edge1 = SimilarityEdge("A", "B", 0.5)
        edge2 = SimilarityEdge("A", "C", 0.5)
        assert edge1 != edge2


# =============================================================================
# Tests for LocalizationResult
# =============================================================================


class TestLocalizationResult:
    def test_has_soft_boundaries_when_edges_exist(self):
        result = LocalizationResult(
            left_root="A",
            right_root="B",
            aggregate_p_value=0.01,
            aggregate_significant=True,
            similarity_edges=[SimilarityEdge("A1", "B1", 0.5)],
        )
        assert result.has_soft_boundaries is True

    def test_no_soft_boundaries_when_empty(self):
        result = LocalizationResult(
            left_root="A",
            right_root="B",
            aggregate_p_value=0.01,
            aggregate_significant=True,
            similarity_edges=[],
        )
        assert result.has_soft_boundaries is False

    def test_all_different_when_significant_no_edges(self):
        result = LocalizationResult(
            left_root="A",
            right_root="B",
            aggregate_p_value=0.01,
            aggregate_significant=True,
            similarity_edges=[],
        )
        assert result.all_different is True

    def test_get_similarity_graph(self):
        result = LocalizationResult(
            left_root="A",
            right_root="B",
            aggregate_p_value=0.01,
            aggregate_significant=True,
            similarity_edges=[
                SimilarityEdge("A1", "B1", 0.5),
                SimilarityEdge("A1", "B2", 0.6),
            ],
        )
        G = result.get_similarity_graph()
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
        assert G.has_edge("A1", "B1")
        assert G.has_edge("A1", "B2")


# =============================================================================
# Tests for localize_divergence_signal
# =============================================================================


class TestLocalizeDivergenceSignal:
    def test_returns_similarity_edge_when_not_significant(self):
        """When aggregate comparison is not significant, should return similarity edge."""
        tree = _make_test_tree()

        # All tests return high p-value (not significant)
        test_func = _make_mock_test_func({})  # Default p=0.5

        result = localize_divergence_signal(
            tree=tree,
            left_root="A",
            right_root="B",
            test_divergence=test_func,
            alpha=0.05,
        )

        assert result.aggregate_significant is False
        assert len(result.similarity_edges) == 1
        assert result.similarity_edges[0].node_a == "A"
        assert result.similarity_edges[0].node_b == "B"

    def test_finds_partial_similarity(self):
        """A1≈B1 but A2≠B2 should find the A1-B1 similarity edge."""
        tree = _make_test_tree()

        p_values = {
            frozenset(["A", "B"]): 0.001,  # Aggregate: significant
            frozenset(["A1", "B1"]): 0.6,  # Similar!
            frozenset(["A1", "B2"]): 0.001,  # Different
            frozenset(["A2", "B1"]): 0.001,  # Different
            frozenset(["A2", "B2"]): 0.001,  # Different
        }
        test_func = _make_mock_test_func(p_values)

        result = localize_divergence_signal(
            tree=tree,
            left_root="A",
            right_root="B",
            test_divergence=test_func,
            alpha=0.05,
            apply_fdr=False,  # Skip FDR for predictable results
        )

        assert result.aggregate_significant is True
        assert result.has_soft_boundaries is True

        # Should find exactly A1-B1 as similar
        similar_pairs = {
            frozenset([e.node_a, e.node_b]) for e in result.similarity_edges
        }
        assert frozenset(["A1", "B1"]) in similar_pairs

    def test_all_different_returns_no_similarity_edges(self):
        """When all pairs are different, should have no similarity edges."""
        tree = _make_test_tree()

        p_values = {
            frozenset(["A", "B"]): 0.001,
            frozenset(["A1", "B1"]): 0.001,
            frozenset(["A1", "B2"]): 0.001,
            frozenset(["A2", "B1"]): 0.001,
            frozenset(["A2", "B2"]): 0.001,
        }
        test_func = _make_mock_test_func(p_values)

        result = localize_divergence_signal(
            tree=tree,
            left_root="A",
            right_root="B",
            test_divergence=test_func,
            alpha=0.05,
            apply_fdr=False,
        )

        assert result.aggregate_significant is True
        assert result.has_soft_boundaries is False
        assert result.all_different is True

    def test_respects_max_depth(self):
        """Should not recurse beyond max_depth."""
        tree = _make_test_tree()

        # All pairs significant → would normally recurse
        p_values = {frozenset(["A", "B"]): 0.001}
        test_func = _make_mock_test_func(p_values)

        result = localize_divergence_signal(
            tree=tree,
            left_root="A",
            right_root="B",
            test_divergence=test_func,
            alpha=0.05,
            max_depth=0,  # No recursion
        )

        assert result.depth_reached == 0

    def test_tracks_nodes_tested(self):
        """Should track number of pairwise tests performed."""
        tree = _make_test_tree()
        test_func = _make_mock_test_func({})

        result = localize_divergence_signal(
            tree=tree,
            left_root="A",
            right_root="B",
            test_divergence=test_func,
            alpha=0.05,
        )

        assert result.nodes_tested >= 1


# =============================================================================
# Tests for merge_similarity_graphs
# =============================================================================


class TestMergeSimilarityGraphs:
    def test_empty_input(self):
        G = merge_similarity_graphs({})
        assert G.number_of_edges() == 0

    def test_merges_edges_from_multiple_results(self):
        results = {
            "parent1": LocalizationResult(
                left_root="A",
                right_root="B",
                aggregate_p_value=0.01,
                aggregate_significant=True,
                similarity_edges=[SimilarityEdge("A1", "B1", 0.5)],
            ),
            "parent2": LocalizationResult(
                left_root="C",
                right_root="D",
                aggregate_p_value=0.01,
                aggregate_significant=True,
                similarity_edges=[SimilarityEdge("C1", "D1", 0.6)],
            ),
        }

        G = merge_similarity_graphs(results)
        assert G.number_of_edges() == 2
        assert G.has_edge("A1", "B1")
        assert G.has_edge("C1", "D1")

    def test_keeps_lower_p_value_for_duplicate_edges(self):
        results = {
            "parent1": LocalizationResult(
                left_root="A",
                right_root="B",
                aggregate_p_value=0.01,
                aggregate_significant=True,
                similarity_edges=[SimilarityEdge("A1", "B1", 0.5)],
            ),
            "parent2": LocalizationResult(
                left_root="A",
                right_root="B",
                aggregate_p_value=0.01,
                aggregate_significant=True,
                similarity_edges=[SimilarityEdge("A1", "B1", 0.3)],  # Lower p-value
            ),
        }

        G = merge_similarity_graphs(results)
        assert G.number_of_edges() == 1
        assert G.edges["A1", "B1"]["p_value"] == 0.3


# =============================================================================
# Tests for merge_difference_graphs
# =============================================================================


class TestMergeDifferenceGraphs:
    def test_merges_difference_pairs(self):
        results = {
            "parent1": LocalizationResult(
                left_root="A",
                right_root="B",
                aggregate_p_value=0.001,
                aggregate_significant=True,
                difference_pairs=[("A1", "B1"), ("A2", "B2")],
            )
        }
        G = merge_difference_graphs(results)
        assert G.number_of_edges() == 2
        assert G.has_edge("A1", "B1")
        assert G.has_edge("A2", "B2")


# =============================================================================
# Tests for extract_constrained_clusters
# =============================================================================


class TestExtractConstrainedClusters:
    def test_merges_when_no_conflict(self):
        """Should merge A and B when similar and no difference exists."""
        tree = _make_test_tree()

        # Similarity graph: A1 ~ B1
        sim_G = nx.Graph()
        sim_G.add_edge("A1", "B1", p_value=0.5)

        # Difference graph: Empty
        diff_G = nx.Graph()

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=[],
        )

        # Find cluster containing A1
        a1_cluster = next(c for c in clusters if "A1" in c)
        assert "B1" in a1_cluster

    def test_prevents_merge_when_conflict_exists(self):
        """Should NOT merge A and B when similar IF a difference exists."""
        tree = _make_test_tree()

        # Similarity graph: A1 ~ B1 (Strong similarity)
        sim_G = nx.Graph()
        sim_G.add_edge("A1", "B1", p_value=0.8)

        # Difference graph: A1 != B1 (Explicit difference found elsewhere or overridden)
        # OR more commonly: A1 != B1_sibling and we are checking cluster-level constraints.
        # Let's test direct conflict first.
        diff_G = nx.Graph()
        diff_G.add_edge("A1", "B1")

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=[],
        )

        # A1 and B1 should be separate
        # Note: Depending on merge_points, they might be singletons or part of larger pre-existing groups.
        # Here they are just nodes in the graphs.

        a1_cluster = next(c for c in clusters if "A1" in c)
        assert "B1" not in a1_cluster

    def test_transitive_merge_blocked_by_conflict(self):
        """Test blocking A ~ B ~ C if A != C."""
        tree = _make_test_tree()

        # Similarity: A1 ~ B1, B1 ~ C1
        sim_G = nx.Graph()
        sim_G.add_edge("A1", "B1", p_value=0.9)
        sim_G.add_edge("B1", "C1", p_value=0.5)  # Weaker similarity

        # Difference: A1 != C1 (Cannot-Link)
        diff_G = nx.Graph()
        diff_G.add_edge("A1", "C1")

        # We need C1 in the tree for _get_all_leaves to work
        tree.add_node("C1", is_leaf=True)

        clusters = extract_constrained_clusters(
            similarity_graph=sim_G,
            difference_graph=diff_G,
            tree=tree,
            merge_points=[],
        )

        # Logic:
        # 1. Sort edges: (A1, B1, 0.9), (B1, C1, 0.5)
        # 2. Process (A1, B1): No conflict. Merge {A1, B1}.
        # 3. Process (B1, C1): Cluster is {A1, B1}. Target is {C1}.
        #    Check conflict {A1, B1} vs {C1}.
        #    A1 is in {A1, B1}, C1 is in {C1}. Edge (A1, C1) exists in diff_G.
        #    Conflict! Do not merge.

        a1_cluster = next(c for c in clusters if "A1" in c)
        assert "B1" in a1_cluster  # First merge succeeded
        assert "C1" not in a1_cluster  # Second merge blocked


# =============================================================================
# Tests for build_cross_boundary_similarity
# =============================================================================


class TestBuildCrossBoundarySimilarity:
    def test_builds_results_for_all_split_points(self):
        tree = _make_test_tree()
        test_func = _make_mock_test_func({})

        split_points = [("root", "A", "B")]

        results = build_cross_boundary_similarity(
            tree=tree,
            split_points=split_points,
            test_divergence=test_func,
            alpha=0.05,
        )

        assert "root" in results
        assert isinstance(results["root"], LocalizationResult)


# =============================================================================
# Integration test
# =============================================================================


class TestIntegration:
    def test_recurses_into_unbalanced_tree(self):
        """Should recurse seamlessly into deep subtrees even if one side is a leaf.

        Scenario:
        L (leaf) vs R (root of subtree R->R1->R2)
        - L vs R: Different
        - L vs R1: Different
        - L vs R2: Similar

        The algorithm must drill down to find the L-R2 similarity.
        """
        tree = _make_test_tree()

        # Add a deep chain specifically for this test
        # R (root) -> R1 -> R2 (leaf)
        tree.add_node("L", is_leaf=True, sample_size=100)
        tree.add_node("R", is_leaf=False, sample_size=100)
        tree.add_node("R1", is_leaf=False, sample_size=100)
        tree.add_node("R2", is_leaf=True, sample_size=100)

        tree.add_edge("R", "R1")
        tree.add_edge("R1", "R2")

        # Mock p-values
        # L vs R -> Different
        # L vs R1 -> Different
        # L vs R2 -> Similar
        p_values = {
            frozenset(["L", "R"]): 0.001,
            frozenset(["L", "R1"]): 0.001,
            frozenset(["L", "R2"]): 0.6,
        }
        test_func = _make_mock_test_func(p_values)

        result = localize_divergence_signal(
            tree=tree,
            left_root="L",
            right_root="R",
            test_divergence=test_func,
            alpha=0.05,
            apply_fdr=False,
            max_depth=5,
        )

        assert result.aggregate_significant is True

        # Should find similarity between L and R2
        similar_pairs = {
            frozenset([e.node_a, e.node_b]) for e in result.similarity_edges
        }
        assert frozenset(["L", "R2"]) in similar_pairs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
