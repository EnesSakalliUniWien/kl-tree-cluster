from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.shared.oracle_tree_recoverability import (
    FAILURE_CLASS_GATE_OVER_SPLIT,
    FAILURE_CLASS_GATE_UNDER_SPLIT,
    FAILURE_CLASS_SOLVED,
    FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE,
    FAILURE_CLASS_TREE_UNRECOVERABLE,
    classify_tree_recoverability_failure,
    oracle_subtree_cut,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def _tree_from_children(children: np.ndarray) -> PosetTree:
    linkage_matrix = np.column_stack(
        [
            children,
            np.arange(1, len(children) + 1, dtype=float),
            np.arange(2, len(children) + 2, dtype=float),
        ]
    )
    return PosetTree.from_linkage(
        linkage_matrix,
        leaf_names=["A", "B", "C", "D"],
    )


def test_oracle_subtree_cut_recovers_aligned_clades() -> None:
    tree = _tree_from_children(
        np.asarray(
            [
                [0, 1],
                [2, 3],
                [4, 5],
            ]
        )
    )
    result = oracle_subtree_cut(
        tree,
        sample_index=pd.Index(["A", "B", "C", "D"]),
        true_labels=np.asarray([0, 0, 1, 1]),
        exact_k=2,
    )

    assert result.ari == 1.0
    assert result.found_clusters == 2


def test_oracle_subtree_cut_exposes_unrecoverable_crossed_clades() -> None:
    tree = _tree_from_children(
        np.asarray(
            [
                [0, 2],
                [1, 3],
                [4, 5],
            ]
        )
    )
    result = oracle_subtree_cut(
        tree,
        sample_index=pd.Index(["A", "B", "C", "D"]),
        true_labels=np.asarray([0, 0, 1, 1]),
        exact_k=2,
    )

    assert result.ari < 1.0
    assert result.found_clusters == 2


def test_oracle_subtree_cut_handles_single_class_truth() -> None:
    tree = _tree_from_children(
        np.asarray(
            [
                [0, 1],
                [2, 3],
                [4, 5],
            ]
        )
    )
    result = oracle_subtree_cut(
        tree,
        sample_index=pd.Index(["A", "B", "C", "D"]),
        true_labels=np.asarray([0, 0, 0, 0]),
    )

    assert result.ari == 1.0
    assert result.found_clusters == 1


@pytest.mark.parametrize(
    (
        "kl_ari",
        "kl_found_clusters",
        "true_clusters",
        "oracle_true_k_subtree_ari",
        "expected",
    ),
    [
        (0.99, 5, 4, 1.0, FAILURE_CLASS_SOLVED),
        (0.10, 4, 4, 0.40, FAILURE_CLASS_TREE_UNRECOVERABLE),
        (0.10, 2, 4, 0.90, FAILURE_CLASS_GATE_UNDER_SPLIT),
        (0.10, 9, 4, 0.90, FAILURE_CLASS_GATE_OVER_SPLIT),
        (0.10, 4, 4, 0.90, FAILURE_CLASS_TREE_RECOVERABLE_STATISTICAL_FAILURE),
    ],
)
def test_classify_tree_recoverability_failure(
    kl_ari: float,
    kl_found_clusters: int,
    true_clusters: int,
    oracle_true_k_subtree_ari: float,
    expected: str,
) -> None:
    assert (
        classify_tree_recoverability_failure(
            kl_ari=kl_ari,
            kl_found_clusters=kl_found_clusters,
            true_clusters=true_clusters,
            oracle_true_k_subtree_ari=oracle_true_k_subtree_ari,
        )
        == expected
    )
