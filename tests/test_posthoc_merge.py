import networkx as nx

from kl_clustering_analysis.hierarchy_analysis.posthoc_merge import apply_posthoc_merge


def test_posthoc_merge_respects_significant_pairs_at_lca() -> None:
    """Non-significant pair across a boundary should not override significant evidence."""

    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "left"),
            ("root", "right"),
            ("left", "A"),
            ("left", "B"),
            ("right", "C"),
            ("right", "D"),
        ]
    )
    children = {n: list(tree.successors(n)) for n in tree.nodes}

    cluster_roots = {"A", "B", "C", "D"}

    # Only B vs C is non-significant; other pairs across the same LCA (root)
    # are significant and should block the merge.
    p_values = {
        frozenset({"A", "B"}): 0.001,
        frozenset({"C", "D"}): 0.002,
        frozenset({"A", "C"}): 0.001,
        frozenset({"A", "D"}): 0.0015,
        frozenset({"B", "C"}): 0.2,  # non-significant
        frozenset({"B", "D"}): 0.0012,
    }

    def fake_test(a: str, b: str, lca: str) -> tuple[float, float, float]:
        # return (test_stat, df, p_value)
        return 0.0, 1.0, p_values[frozenset({a, b})]

    merged, _audit = apply_posthoc_merge(
        cluster_roots=cluster_roots,
        alpha=0.05,
        tree=tree,
        children=children,
        root="root",
        test_divergence=fake_test,
    )

    assert merged == cluster_roots


def test_posthoc_merge_merges_when_pairs_are_all_similar() -> None:
    """Baseline sanity check: merge proceeds when no pair is significant."""

    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("root", "A"),
            ("root", "B"),
        ]
    )
    children = {n: list(tree.successors(n)) for n in tree.nodes}

    cluster_roots = {"A", "B"}

    def fake_test(a: str, b: str, lca: str) -> tuple[float, float, float]:
        return 0.0, 1.0, 0.2

    merged, _audit = apply_posthoc_merge(
        cluster_roots=cluster_roots,
        alpha=0.05,
        tree=tree,
        children=children,
        root="root",
        test_divergence=fake_test,
    )

    assert merged == {"root"}


def test_posthoc_merge_does_not_reintroduce_descendants_after_ancestor_merge() -> None:
    """Regression: merging at an ancestor must not allow later descendant merges (antichain/partition)."""

    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("R", "A"),
            ("R", "B"),
            ("A", "C"),
            ("A", "D"),
            ("C", "A1"),
            ("D", "A2"),
            ("D", "A3"),
        ]
    )
    children = {n: list(tree.successors(n)) for n in tree.nodes}

    # Initial cluster roots (an antichain).
    cluster_roots = {"A1", "A2", "A3", "B"}

    # Ensure the greedy order merges at the ancestor boundary R first, then would
    # attempt the descendant boundary D if stale pairs were not filtered out.
    p_values = {
        frozenset({"A1", "B"}): 0.99,
        frozenset({"A2", "B"}): 0.90,
        frozenset({"A3", "B"}): 0.91,
        frozenset({"A2", "A3"}): 0.98,
        frozenset({"A1", "A2"}): 0.80,
        frozenset({"A1", "A3"}): 0.81,
    }

    def fake_test(a: str, b: str, lca: str) -> tuple[float, float, float]:
        return 0.0, 1.0, p_values[frozenset({a, b})]

    merged, _audit = apply_posthoc_merge(
        cluster_roots=cluster_roots,
        alpha=0.05,
        tree=tree,
        children=children,
        root="R",
        test_divergence=fake_test,
    )

    # Correct behavior: once we merge at R, no descendant root can coexist with R.
    assert merged == {"R"}
