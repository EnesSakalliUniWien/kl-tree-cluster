"""Unit tests for local KL helper utilities.

These checks ensure that cached values, recomputation paths, and Series extraction
behave predictably for downstream statistical routines.
"""

import numpy as np
import pandas as pd
import networkx as nx

from hierarchy_analysis.decomposition_utils import binary_kl
from hierarchy_analysis.local_kl_utils import get_local_kl_value, get_local_kl_series


def _make_simple_tree():
    """Build a minimal parent→child graph with preset Bernoulli distributions."""
    G = nx.DiGraph()
    parent_dist = np.array([0.6, 0.4], dtype=float)
    child_dist = np.array([0.9, 0.1], dtype=float)
    G.add_node("P", distribution=parent_dist, is_leaf=False)
    G.add_node("C", distribution=child_dist, is_leaf=True)
    G.add_edge("P", "C")
    return G, child_dist, parent_dist


def test_get_local_kl_value_returns_stored_attribute():
    tree, *_ = _make_simple_tree()
    tree.nodes["C"]["kl_divergence_local"] = 0.123

    value = get_local_kl_value(tree, "C", "P")

    assert value == 0.123


def test_get_local_kl_value_recomputes_when_missing():
    tree, child_dist, parent_dist = _make_simple_tree()
    tree.nodes["C"].pop("kl_divergence_local", None)

    expected = binary_kl(child_dist, parent_dist)
    value = get_local_kl_value(tree, "C", "P")

    assert np.isclose(value, expected)


def test_get_local_kl_value_uses_default_without_parent():
    tree, *_ = _make_simple_tree()
    tree.nodes["C"].pop("kl_divergence_local", None)

    value = get_local_kl_value(tree, "C", parent_id=None, default=-1.0)

    assert value == -1.0


def test_get_local_kl_series_handles_missing_column():
    df = pd.DataFrame({"leaf_count": [3, 4]}, index=["A", "B"])

    series = get_local_kl_series(df)

    assert list(series.index) == ["A", "B"]
    assert series.isna().all()
    assert series.dtype == float


def test_get_local_kl_series_returns_existing_series():
    df = pd.DataFrame({"kl_divergence_local": [0.1, 0.2]}, index=["A", "B"])

    series = get_local_kl_series(df)

    assert all(np.isclose(series.values, [0.1, 0.2]))
    assert list(series.index) == ["A", "B"]
