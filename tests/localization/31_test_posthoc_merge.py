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
    """Targeted removal + antichain guard: only safe merges proceed.

    Tree structure:
         R
        / \\
       A    B
      / \\
     C    D
    /    / \\
   A1  A2  A3

    All pairs are similar (p >> 0.05).
    Greedy order (highest p first): A1↔B, A2↔A3, A3↔B, ...

    With the antichain guard:
    - A1↔B (LCA=R): blocked because D-descendants are also under R
    - A2↔A3 (LCA=D): proceeds — no other roots under D
    After A2↔A3 merge: cluster_roots = {A1, D, B}
    - Remaining pairs involving A2/A3 are stale (no longer in cluster_roots)

    Final: {A1, D, B} — not {R} as with the old blanket removal.
    """

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

    # A2↔A3 merge at D; A1↔B blocked by antichain guard (D under R).
    assert merged == {"A1", "D", "B"}


def test_posthoc_merge_targeted_removal_preserves_third_cluster() -> None:
    """Fix 1.2: Merging A↔B must not absorb unrelated C under a different subtree.

    Tree structure:
              R
            /   \\
          M1     M2
         / \\     |
        A   B    C

    A and B are similar → merge at M1.
    C is under M2, a sibling of M1 under R.
    C must NOT be absorbed by the A↔B merge.
    """
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("R", "M1"),
            ("R", "M2"),
            ("M1", "A"),
            ("M1", "B"),
            ("M2", "C"),
        ]
    )
    children = {n: list(tree.successors(n)) for n in tree.nodes}

    cluster_roots = {"A", "B", "C"}

    p_values = {
        frozenset({"A", "C"}): 0.001,  # A≠C (significant)
        frozenset({"B", "C"}): 0.001,  # B≠C (significant)
        frozenset({"A", "B"}): 0.8,  # A≈B (not significant → merge)
    }

    def fake_test(a: str, b: str, lca: str) -> tuple[float, float, float]:
        return 0.0, 1.0, p_values[frozenset({a, b})]

    merged, audit = apply_posthoc_merge(
        cluster_roots=cluster_roots,
        alpha=0.05,
        tree=tree,
        children=children,
        root="R",
        test_divergence=fake_test,
    )

    # A and B should merge at M1; C should remain standalone
    assert "M1" in merged, f"Expected M1 (A+B merge) in result, got {merged}"
    assert "C" in merged, f"Expected C to survive, got {merged}"
    assert len(merged) == 2, f"Expected 2 cluster roots, got {len(merged)}: {merged}"


def test_posthoc_merge_independent_merges_at_different_boundaries() -> None:
    """Fix 1.2/1.3: Independent similar pairs at different boundaries both merge.

    Tree structure:
              R
            /   \\
          M1     M2
         / \\    / \\
        A   B  C   D

    A≈B and C≈D are both not significant.
    Cross-boundary pairs (A↔C, A↔D, B↔C, B↔D) are all significant.
    Both within-subtree merges should proceed independently.
    """
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("R", "M1"),
            ("R", "M2"),
            ("M1", "A"),
            ("M1", "B"),
            ("M2", "C"),
            ("M2", "D"),
        ]
    )
    children = {n: list(tree.successors(n)) for n in tree.nodes}

    cluster_roots = {"A", "B", "C", "D"}

    p_values = {
        frozenset({"A", "B"}): 0.8,  # A≈B (at M1 boundary)
        frozenset({"C", "D"}): 0.9,  # C≈D (at M2 boundary)
        frozenset({"A", "C"}): 0.001,  # cross-boundary, significant
        frozenset({"A", "D"}): 0.001,  # cross-boundary, significant
        frozenset({"B", "C"}): 0.001,  # cross-boundary, significant
        frozenset({"B", "D"}): 0.001,  # cross-boundary, significant
    }

    def fake_test(a: str, b: str, lca: str) -> tuple[float, float, float]:
        return 0.0, 1.0, p_values[frozenset({a, b})]

    merged, audit = apply_posthoc_merge(
        cluster_roots=cluster_roots,
        alpha=0.05,
        tree=tree,
        children=children,
        root="R",
        test_divergence=fake_test,
    )

    # Both A↔B and C↔D should merge independently.
    assert "M1" in merged, f"Expected M1 (A+B merge), got {merged}"
    assert "M2" in merged, f"Expected M2 (C+D merge), got {merged}"
    assert len(merged) == 2, f"Expected 2 clusters, got {len(merged)}: {merged}"


def test_posthoc_merge_audit_trail_records_all_pairs() -> None:
    """Audit trail should record every candidate pair with its outcome."""
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("R", "A"),
            ("R", "B"),
        ]
    )
    children = {n: list(tree.successors(n)) for n in tree.nodes}

    def fake_test(a: str, b: str, lca: str) -> tuple[float, float, float]:
        return 5.0, 2.0, 0.3

    _, audit = apply_posthoc_merge(
        cluster_roots={"A", "B"},
        alpha=0.05,
        tree=tree,
        children=children,
        root="R",
        test_divergence=fake_test,
    )

    assert len(audit) == 1
    pair = audit[0]
    assert "left_cluster" in pair
    assert "right_cluster" in pair
    assert "p_value" in pair
    assert "is_significant" in pair
    assert "was_merged" in pair
    assert pair["test_stat"] == 5.0
    assert pair["df"] == 2.0
