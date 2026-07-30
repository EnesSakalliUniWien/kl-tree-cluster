"""Regression tests for ASV benchmark measurement boundaries."""

from __future__ import annotations

import asv_benchmarks.bench_tree_construction as tree_benchmarks


def test_populate_divergence_asv_benchmark_does_not_rebuild_tree(monkeypatch) -> None:
    """The populate benchmark must not include a second tree-build measurement."""
    build_calls = 0
    original_build_tree = tree_benchmarks.build_tree

    def counted_build_tree(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build_tree(*args, **kwargs)

    monkeypatch.setattr(tree_benchmarks, "build_tree", counted_build_tree)

    suite = tree_benchmarks.TreeConstructionSuite()
    suite.setup("gaussian_120x48")
    assert build_calls == 1

    suite.time_average_linkage_tree_build("gaussian_120x48")
    assert build_calls == 2

    suite.time_populate_node_divergences("gaussian_120x48")
    assert build_calls == 2
