import networkx as nx

from kl_clustering_analysis.plot.cluster_tree_visualization import _rectangular_tree_layout


def test_rectangular_tree_layout_monotone_depth_and_deterministic():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("root", "a"),
            ("root", "b"),
            ("a", "c"),
            ("a", "d"),
        ]
    )

    pos1 = _rectangular_tree_layout(G, x_gap=1.0, y_gap=1.0)
    pos2 = _rectangular_tree_layout(G, x_gap=1.0, y_gap=1.0)
    assert pos1 == pos2

    # Parent is above child (y decreases with depth).
    assert pos1["root"][1] > pos1["a"][1]
    assert pos1["root"][1] > pos1["b"][1]
    assert pos1["a"][1] > pos1["c"][1]
    assert pos1["a"][1] > pos1["d"][1]

    # Deterministic horizontal ordering induced by sorted children.
    assert pos1["a"][0] < pos1["b"][0]
    assert pos1["c"][0] < pos1["d"][0]
    assert pos1["c"][0] <= pos1["a"][0] <= pos1["d"][0]
