import networkx as nx
import numpy as np
import pandas as pd
import pytest
from tree_break_selection.tree.branch_lengths import compute_ultrametric_branch_lengths
from tree_break_selection.tree.construction import tree_from_linkage
from tree_break_selection.tree.poset_tree import PosetTree


def _leaf_labels_under(G: nx.DiGraph, node):
    if G.out_degree(node) == 0:
        return {G.nodes[node].get("label", node)}
    out = set()
    for child in G.successors(node):
        out |= _leaf_labels_under(G, child)
    return out


def _clusters_by_node(G: nx.DiGraph):
    return {n: frozenset(_leaf_labels_under(G, n)) for n in G.nodes}


def _assert_laminar_and_inner_nodes_consistent(G: nx.DiGraph):
    clusters = _clusters_by_node(G)
    nodes = list(G.nodes)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = clusters[nodes[i]], clusters[nodes[j]]
            assert a.issubset(b) or b.issubset(a) or a.isdisjoint(b)

    inner_nodes_poset = {n for n, s in clusters.items() if len(s) > 1}
    inner_nodes_graph = {n for n in G.nodes if G.out_degree(n) > 0}
    assert inner_nodes_poset == inner_nodes_graph


def test_find_lca_for_set_rejects_empty_node_set():
    G = PosetTree()
    G.add_node("root", is_leaf=False, label="root")
    G.add_node("left", is_leaf=True, label="left")
    G.add_node("right", is_leaf=True, label="right")
    G.add_edge("root", "left", branch_length=1.0)
    G.add_edge("root", "right", branch_length=1.0)
    G.graph["root"] = "root"

    with pytest.raises(ValueError, match="empty node set"):
        G.find_lca_for_set([])


def test_populate_node_divergences_requires_leaf_labels():
    G = PosetTree()
    G.add_node("root", is_leaf=False)
    G.add_node("left", is_leaf=True)
    G.add_node("right", is_leaf=True, label="right")
    G.add_edge("root", "left")
    G.add_edge("root", "right")
    leaf_data = pd.DataFrame([[0.0], [1.0]], index=["left", "right"])

    with pytest.raises(KeyError, match="label"):
        G.populate_node_divergences(leaf_data)


def test_populate_node_divergences_requires_explicit_leaf_flags():
    G = PosetTree()
    G.add_node("root", is_leaf=False)
    G.add_node("left", label="left")
    G.add_node("right", is_leaf=True, label="right")
    G.add_edge("root", "left")
    G.add_edge("root", "right")
    leaf_data = pd.DataFrame([[0.0], [1.0]], index=["left", "right"])

    with pytest.raises(KeyError, match="is_leaf"):
        G.populate_node_divergences(leaf_data)


def test_compute_ultrametric_branch_lengths_requires_merge_distances():
    children = np.array([[0, 1]], dtype=int)

    with pytest.raises(ValueError, match="merge distances"):
        compute_ultrametric_branch_lengths(2, children, distances=None)


def test_compute_ultrametric_branch_lengths_normalizes_to_root_time():
    children = np.array([[0, 1], [2, 3], [4, 5]], dtype=int)
    distances = np.array([2.0, 6.0, 10.0], dtype=float)

    lengths = compute_ultrametric_branch_lengths(4, children, distances)

    assert lengths[("N4", "L0")] == pytest.approx(0.2)
    assert lengths[("N4", "L1")] == pytest.approx(0.2)
    assert lengths[("N5", "L2")] == pytest.approx(0.6)
    assert lengths[("N5", "L3")] == pytest.approx(0.6)
    assert lengths[("N6", "N4")] == pytest.approx(0.8)
    assert lengths[("N6", "N5")] == pytest.approx(0.4)
    assert lengths[("N6", "N4")] + lengths[("N4", "L0")] == pytest.approx(1.0)
    assert lengths[("N6", "N5")] + lengths[("N5", "L2")] == pytest.approx(1.0)


def test_compute_ultrametric_branch_lengths_rejects_nonmonotone_heights():
    children = np.array([[0, 1], [2, 3], [4, 5]], dtype=int)
    distances = np.array([2.0, 6.0, 5.0], dtype=float)

    with pytest.raises(ValueError, match="nondecreasing"):
        compute_ultrametric_branch_lengths(4, children, distances)


def test_from_scipy_linkage_binary_data():
    pytest.importorskip("scipy")
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    X = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=int,
    )
    leaf_names = [f"s{i}" for i in range(len(X))]
    D = pdist(X, metric="hamming")
    Z = linkage(D, method="average")

    G = tree_from_linkage(Z, leaf_names=leaf_names)

    assert G.number_of_nodes() == 2 * len(X) - 1
    assert nx.is_directed_acyclic_graph(G)
    assert nx.is_tree(G.to_undirected())

    _assert_laminar_and_inner_nodes_consistent(G)
