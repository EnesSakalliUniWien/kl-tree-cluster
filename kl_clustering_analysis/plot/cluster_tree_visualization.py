"""
Cluster tree visualization (NetworkX + matplotlib).

Design goals:
- Simple, systematic plotting
- Radial layout via Graphviz ``twopi`` when available
- Leaf node colors match cluster IDs (consistent with UMAP plots)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .cluster_color_mapping import build_cluster_color_spec, present_cluster_ids


def _map_nodes_to_clusters(
    cluster_assignments: Dict,
    label_to_node: Dict,
    tree,
) -> Dict:
    """Map tree nodes to their cluster IDs."""
    node_to_cluster: Dict = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            if leaf in label_to_node:
                node_id = label_to_node[leaf]
            elif leaf in tree.nodes():
                node_id = leaf
            else:
                continue
            node_to_cluster[node_id] = cluster_id

        root = info["root_node"]
        if root not in node_to_cluster:
            node_to_cluster[root] = cluster_id
    return node_to_cluster


def _graphviz_twopi_layout(G: nx.Graph, args: str = "") -> Dict:
    """Compute a Graphviz ``twopi`` layout with graceful fallbacks."""
    try:
        return nx.nx_agraph.graphviz_layout(G, prog="twopi", args=args)
    except Exception:
        pass

    try:
        return nx.nx_pydot.graphviz_layout(G, prog="twopi")
    except Exception:
        pass

    return nx.spring_layout(G, seed=0)


def plot_tree_with_clusters(
    tree,
    decomposition_results: Dict,
    results_df=None,
    use_labels: bool = True,
    width: int = 900,
    height: int = 600,
    node_size: int = 20,
    font_size: int = 10,
    show_cluster_boundaries: bool = True,
    colormap: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
):
    """
    Plot hierarchical tree with cluster assignments.

    Notes
    -----
    - This is intentionally a lightweight plot. ``show_cluster_boundaries`` is a
      placeholder (not implemented).
    - Leaf nodes are colored by their cluster ID; internal nodes are gray.
    """
    _ = results_df
    _ = show_cluster_boundaries
    _ = use_labels

    cluster_assignments = decomposition_results["cluster_assignments"]
    num_clusters = decomposition_results["num_clusters"]

    spec = build_cluster_color_spec(
        num_clusters, base_cmap=colormap, unassigned_color="#CCCCCC"
    )
    cluster_id_to_color = spec.id_to_color
    unassigned_color = spec.unassigned_color

    label_to_node: Dict = {}
    for node in tree.nodes():
        node_attrs = tree.nodes[node] if hasattr(tree, "nodes") else {}
        label = node_attrs.get("label", node) if isinstance(node_attrs, dict) else node
        label_to_node[label] = node

    node_to_cluster = _map_nodes_to_clusters(cluster_assignments, label_to_node, tree)

    G = nx.DiGraph()
    G.add_nodes_from(tree.nodes())
    G.add_edges_from(tree.edges())

    leaves = {n for n in G.nodes() if G.out_degree(n) == 0}
    node_colors = []
    for node in G.nodes():
        if node in leaves and node in node_to_cluster:
            try:
                cluster_id = int(node_to_cluster[node])
            except Exception:
                cluster_id = None
            node_colors.append(cluster_id_to_color.get(cluster_id, unassigned_color))
        else:
            node_colors.append(unassigned_color)

    pos = _graphviz_twopi_layout(G, args="")

    if title is None:
        title = (
            f"Hierarchical Tree with Cluster Decomposition: "
            f"{num_clusters} Independent Clusters Identified"
        )

    if figsize is None:
        figsize = (max(width, 1) / 100.0, max(height, 1) / 100.0)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(
        G,
        pos,
        node_size=node_size,
        alpha=0.5,
        node_color=node_colors,
        edge_color="gray",
        with_labels=False,
        arrows=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=font_size)
    ax.axis("equal")
    ax.set_axis_off()

    leaf_cluster_ids = [
        int(node_to_cluster[n])
        for n in leaves
        if n in node_to_cluster and str(node_to_cluster[n]).lstrip("-").isdigit()
    ]
    present_ids = present_cluster_ids(leaf_cluster_ids)
    if present_ids:
        handles = [
            Patch(facecolor=cluster_id_to_color[cid], edgecolor="none", label=f"{cid}")
            for cid in present_ids
            if cid in cluster_id_to_color
        ]
        if handles:
            ax.legend(
                handles=handles,
                title="Cluster",
                loc="best",
                frameon=False,
                fontsize=max(font_size - 1, 6),
                title_fontsize=max(font_size - 1, 6),
            )

    if show:
        plt.show()

    return fig, ax
