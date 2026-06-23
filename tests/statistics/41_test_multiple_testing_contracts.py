from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from tree_break_selection.hierarchy_analysis.statistics.multiple_testing import (
    benjamini_hochberg_correction,
)
from tree_break_selection.hierarchy_analysis.statistics.multiple_testing.tree_bh import (
    apply_tree_bh_correction,
)


def _two_child_tree() -> nx.DiGraph:
    tree = nx.DiGraph()
    tree.add_edge("root", "A")
    tree.add_edge("root", "B")
    return tree


def test_bh_correction_rejects_nonfinite_p_values() -> None:
    with pytest.raises(ValueError, match="finite before correction"):
        benjamini_hochberg_correction(np.array([0.01, np.nan]), alpha=0.05)


def test_bh_correction_rejects_out_of_range_p_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg_correction(np.array([0.01, 1.2]), alpha=0.05)


def test_tree_bh_requires_p_values_to_align_with_child_ids() -> None:
    with pytest.raises(ValueError, match="align one-to-one with child_ids"):
        apply_tree_bh_correction(
            _two_child_tree(),
            np.array([0.01]),
            ["A", "B"],
            alpha=0.05,
        )


def test_tree_bh_rejects_nonfinite_p_values() -> None:
    with pytest.raises(ValueError, match="finite before correction"):
        apply_tree_bh_correction(
            _two_child_tree(),
            np.array([0.01, np.nan]),
            ["A", "B"],
            alpha=0.05,
        )


def test_tree_bh_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        apply_tree_bh_correction(
            _two_child_tree(),
            np.array([0.01, 0.02]),
            ["A", "B"],
            alpha=0.0,
        )


def test_tree_bh_requires_child_ids_to_be_non_root_edges() -> None:
    with pytest.raises(ValueError, match="exactly one parent"):
        apply_tree_bh_correction(
            _two_child_tree(),
            np.array([0.01]),
            ["root"],
            alpha=0.05,
        )


def test_tree_bh_requires_child_ids_to_exist_in_tree() -> None:
    with pytest.raises(ValueError, match="present in the tree"):
        apply_tree_bh_correction(
            _two_child_tree(),
            np.array([0.01]),
            ["missing"],
            alpha=0.05,
        )
