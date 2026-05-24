from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.shared.oracle_tree_recoverability import oracle_subtree_cut
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
