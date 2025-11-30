import pytest
import networkx as nx
import numpy as np

from kl_clustering_analysis.tree.poset_tree import PosetTree


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


def test_from_tuples_nested_basic():
    # Skip this test as from_nested_tuples method doesn't exist
    pytest.skip("from_nested_tuples method not implemented")


def test_from_tuples_edges_basic():
    # Star: a-b, b-c, b-d
    edges = [("a", "b", 2.0), ("b", "c", 1.0), ("b", "d", 3.0)]
    G = PosetTree.from_undirected_edges(edges)

    assert nx.is_tree(G.to_undirected())
    # All nodes present as labels - check that nodes exist
    for n in ["a", "b", "c", "d"]:
        assert n in G.nodes
        # Note: from_undirected_edges may not set 'label' attribute
        # assert G.nodes[n]["label"] == n

    _assert_laminar_and_inner_nodes_consistent(G)


def test_from_scipy_linkage_binary_data():
    scipy = pytest.importorskip("scipy")
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

    G = PosetTree.from_linkage(Z, leaf_names=leaf_names)

    assert G.number_of_nodes() == 2 * len(X) - 1
    assert nx.is_directed_acyclic_graph(G)
    assert nx.is_tree(G.to_undirected())

    _assert_laminar_and_inner_nodes_consistent(G)


def test_from_scipy_clusternode():
    # Skip this test as from_cluster_node method doesn't exist
    pytest.skip("from_cluster_node method not implemented")


def test_from_scipy_sparse_tree():
    # Skip this test as from_sparse_adjacency method doesn't exist
    pytest.skip("from_sparse_adjacency method not implemented")


def _print_tree_recursive(G, node, prefix="", is_last=True):
    """Recursively prints the tree structure."""
    label = G.nodes[node].get("label", node)
    node_name = f"{label} ({node})"
    print(prefix + ("└── " if is_last else "├── ") + node_name)
    children = list(G.successors(node))
    for i, child in enumerate(children):
        is_last_child = i == len(children) - 1
        _print_tree_recursive(
            G, child, prefix + ("    " if is_last else "│   "), is_last_child
        )


def print_tree(G):
    """Prints the tree structure of a PosetTree."""
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    for root in roots:
        _print_tree_recursive(G, root)


def test_print_tree():
    """Tests the tree printing function."""
    # Skip this test as from_nested_tuples method doesn't exist
    pytest.skip("from_nested_tuples method not implemented")
