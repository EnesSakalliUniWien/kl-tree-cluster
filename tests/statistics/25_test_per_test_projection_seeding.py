from __future__ import annotations

import networkx as nx
import numpy as np

from kl_clustering_analysis import config
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_p_values_via_projection,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.random_projection import (
    derive_projection_seed,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    _run_tests,
)


def test_derive_projection_seed_is_deterministic_and_unique() -> None:
    seed_a_1 = derive_projection_seed(42, "edge:root->A")
    seed_a_2 = derive_projection_seed(42, "edge:root->A")
    seed_b = derive_projection_seed(42, "edge:root->B")

    assert seed_a_1 == seed_a_2
    assert seed_a_1 != seed_b


def test_edge_tests_use_distinct_per_edge_seeds(monkeypatch) -> None:
    tree = nx.DiGraph()
    tree.add_edge("root", "A")
    tree.add_edge("root", "B")
    tree.nodes["root"]["distribution"] = np.array([0.4, 0.6], dtype=float)
    tree.nodes["A"]["distribution"] = np.array([0.3, 0.7], dtype=float)
    tree.nodes["B"]["distribution"] = np.array([0.5, 0.5], dtype=float)

    captured_seeds: list[int] = []

    def _fake_projected_test(
        child_dist: np.ndarray,
        parent_dist: np.ndarray,
        n_child: int,
        n_parent: int,
        seed: int,
        branch_length: float | None = None,
        mean_branch_length: float | None = None,
        spectral_k: int | None = None,
        pca_projection: np.ndarray | None = None,
        pca_eigenvalues: np.ndarray | None = None,
    ) -> tuple[float, float, float, bool]:
        captured_seeds.append(seed)
        return 0.0, 1.0, 1.0, False

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance._compute_projected_test",
        _fake_projected_test,
    )

    child_ids = ["A", "B"]
    parent_ids = ["root", "root"]
    child_leaf_counts = np.array([5, 5], dtype=float)
    parent_leaf_counts = np.array([10, 10], dtype=float)

    _compute_p_values_via_projection(
        tree,
        child_ids,
        parent_ids,
        child_leaf_counts,
        parent_leaf_counts,
    )

    assert len(captured_seeds) == 2
    assert captured_seeds[0] != captured_seeds[1]
    assert captured_seeds[0] == derive_projection_seed(
        config.PROJECTION_RANDOM_SEED, "edge:root->A"
    )
    assert captured_seeds[1] == derive_projection_seed(
        config.PROJECTION_RANDOM_SEED, "edge:root->B"
    )


def test_sibling_run_tests_passes_unique_test_ids(monkeypatch) -> None:
    parents = ["P1", "P2"]
    args = [
        (np.array([0.3, 0.7]), np.array([0.4, 0.6]), 10, 10, None, None),
        (np.array([0.2, 0.8]), np.array([0.5, 0.5]), 12, 12, None, None),
    ]
    captured_ids: list[str | None] = []

    def _fake_sibling_test(
        left_dist: np.ndarray,
        right_dist: np.ndarray,
        n_left: float,
        n_right: float,
        branch_length_left: float | None = None,
        branch_length_right: float | None = None,
        mean_branch_length: float | None = None,
        *,
        test_id: str | None = None,
        spectral_k: int | None = None,
        pca_projection: np.ndarray | None = None,
        pca_eigenvalues: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        captured_ids.append(test_id)
        return 0.0, 1.0, 1.0

    monkeypatch.setattr(
        "kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test.sibling_divergence_test",
        _fake_sibling_test,
    )

    _run_tests(parents, args, mean_branch_length=None)

    assert captured_ids == ["sibling:P1", "sibling:P2"]
