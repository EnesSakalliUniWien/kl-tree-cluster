"""
Visualization functions for cluster decomposition results on hierarchical trees.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import networkx as nx
from typing import Dict, Optional, Tuple


def plot_tree_with_clusters(
    tree,
    decomposition_results: Dict,
    results_df=None,
    use_labels: bool = True,
    figsize: Tuple[int, int] = (20, 14),
    node_size: int = 3000,
    font_size: int = 10,
    show_cluster_boundaries: bool = True,
    colormap: str = "Set3",
    title: Optional[str] = None,
    layout: str = "hierarchical",
):
    """
    Visualize hierarchical tree with cluster assignments.

    Args:
        tree: PosetTree object
        decomposition_results: Results from ClusterDecomposer.decompose_tree()
        results_df: Optional DataFrame with statistical test results
        use_labels: Whether to use node labels instead of IDs
        figsize: Figure size (width, height)
        node_size: Size of nodes in the plot
        font_size: Font size for node labels
        show_cluster_boundaries: Whether to draw boxes around clusters
        colormap: Matplotlib colormap name for cluster colors
        title: Optional custom title

    Returns:
        fig, ax, cluster_colors: Matplotlib figure, axes, and color mapping
    """
    # Extract cluster information
    cluster_assignments = decomposition_results["cluster_assignments"]
    num_clusters = decomposition_results["num_clusters"]

    # Create color mapping for clusters
    cmap = plt.cm.get_cmap(colormap)
    cluster_colors_list = [
        cmap(i / max(num_clusters - 1, 1)) for i in range(num_clusters)
    ]

    # Create mapping from labels to node IDs
    label_to_node = {}
    for node in tree.nodes():
        label = tree.nodes[node].get("label", node)
        label_to_node[label] = node

    # Map nodes to clusters
    node_to_cluster = {}
    for cluster_id, info in cluster_assignments.items():
        # Map all leaves in this cluster
        for leaf in info["leaves"]:
            # Convert label to node ID if necessary
            if leaf in label_to_node:
                node_id = label_to_node[leaf]
            elif leaf in tree.nodes():
                # Already a node ID
                node_id = leaf
            else:
                # Skip if not found
                continue
            node_to_cluster[node_id] = cluster_id

        # Map the root node of this cluster
        root = info["root_node"]
        if root not in node_to_cluster:
            node_to_cluster[root] = cluster_id

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create layout
    pos = _compute_layout(tree, layout=layout)

    # Prepare node colors
    node_colors = []
    node_labels = {}

    for node in tree.nodes():
        node_data = tree.nodes[node]

        # Get label
        if use_labels:
            label = node_data.get("label", node)
        else:
            label = node
        node_labels[node] = label

        # Determine color
        if node in node_to_cluster:
            cluster_id = node_to_cluster[node]
            node_colors.append(cluster_colors_list[cluster_id])
        else:
            # Unassigned nodes (internal nodes not part of cluster roots)
            node_colors.append("#CCCCCC")  # Gray

    # Draw edges
    nx.draw_networkx_edges(
        tree, pos, ax=ax, edge_color="gray", width=1.5, alpha=0.6, arrows=False
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        tree,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=2,
    )

    # Draw labels
    nx.draw_networkx_labels(
        tree, pos, labels=node_labels, ax=ax, font_size=font_size, font_weight="bold"
    )

    # Add cluster boundaries
    if show_cluster_boundaries:
        _add_cluster_boundaries(ax, pos, node_to_cluster, cluster_colors_list)

    # Create legend
    legend_elements = []
    for cluster_id in sorted(cluster_assignments.keys()):
        info = cluster_assignments[cluster_id]
        color = cluster_colors_list[cluster_id]
        label = f"Cluster {cluster_id} ({info['size']} samples)"
        legend_elements.append(mpatches.Patch(color=color, label=label))

    # Add unassigned to legend if there are any
    if len(node_to_cluster) < len(tree.nodes()):
        legend_elements.append(
            mpatches.Patch(color="#CCCCCC", label="Unassigned (internal)")
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=font_size,
        frameon=True,
        shadow=True,
    )

    # Set title
    if title is None:
        title = (
            f"Hierarchical Tree with Cluster Decomposition\n"
            f"{num_clusters} Independent Clusters Identified"
        )
    ax.set_title(title, fontsize=16, weight="bold", pad=20)

    ax.axis("off")
    plt.tight_layout()

    return fig, ax, cluster_colors_list


def _compute_layout(tree, layout: str = "hierarchical"):
    """
    Compute node positions for the tree.

    Supported layouts:
    - 'hierarchical': Graphviz 'dot' if available, else leveled spring
    - 'radial': Graphviz 'twopi' (circular tree) if available, else circular
    - 'circular': NetworkX circular layout
    - 'spring': Force-directed spring layout
    """
    layout = (layout or "hierarchical").lower()

    # Helper: try graphviz via pygraphviz
    def _graphviz(prog: str):
        try:
            roots = [n for n, d in tree.in_degree() if d == 0]
            root = roots[0] if roots else None
            if root is not None:
                return nx.nx_agraph.graphviz_layout(tree, prog=prog, root=root)
            return nx.nx_agraph.graphviz_layout(tree, prog=prog)
        except Exception:
            return None

    if layout in ("hierarchical", "dot"):
        pos = _graphviz("dot")
        if pos is not None:
            return pos
        # Fallback to leveled spring
        pos = nx.spring_layout(tree, k=2, iterations=50, seed=42)
        # Level nodes by distance to leaves for vertical stratification
        levels = {}
        for node in tree.nodes():
            if tree.nodes[node].get("is_leaf", False):
                levels[node] = 0
            else:
                children = list(tree.successors(node))
                levels[node] = (
                    max(levels.get(c, 0) for c in children) + 1 if children else 0
                )
        max_level = max(levels.values()) if levels else 1
        for n in pos:
            x, y = pos[n]
            lvl = levels.get(n, 0)
            pos[n] = (x, 1 - (lvl / max(max_level, 1)))
        return pos

    if layout in ("radial", "circular_tree", "twopi"):
        pos = _graphviz("twopi")
        if pos is not None:
            return pos
        # Fallback: circular layout
        return nx.circular_layout(tree)

    if layout == "circular":
        return nx.circular_layout(tree)

    if layout == "spring":
        return nx.spring_layout(tree, k=2, iterations=50, seed=42)

    # Default fallback
    return nx.spring_layout(tree, k=2, iterations=50, seed=42)


def _add_cluster_boundaries(ax, pos, node_to_cluster, cluster_colors):
    """
    Add rectangular boundaries around clusters.

    Args:
        ax: Matplotlib axes
        pos: Node positions
        node_to_cluster: Mapping of nodes to cluster IDs
        cluster_colors: List of colors for each cluster
    """
    # Group nodes by cluster
    cluster_nodes = {}
    for node, cluster_id in node_to_cluster.items():
        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(node)

    # Draw boundary for each cluster
    for cluster_id, nodes in cluster_nodes.items():
        if len(nodes) < 2:
            continue

        # Get bounding box
        xs = [pos[node][0] for node in nodes]
        ys = [pos[node][1] for node in nodes]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add padding
        padding = 0.05
        width = x_max - x_min + 2 * padding
        height = y_max - y_min + 2 * padding

        # Draw rectangle
        rect = Rectangle(
            (x_min - padding, y_min - padding),
            width,
            height,
            linewidth=2,
            edgecolor=cluster_colors[cluster_id],
            facecolor="none",
            linestyle="--",
            alpha=0.6,
        )
        ax.add_patch(rect)


def plot_cluster_summary(
    decomposition_results: Dict, figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot summary statistics of cluster decomposition.

    Args:
        decomposition_results: Results from ClusterDecomposer.decompose_tree()
        figsize: Figure size (width, height)

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    cluster_assignments = decomposition_results["cluster_assignments"]
    independence_analysis = decomposition_results["independence_analysis"]

    # Extract data
    cluster_sizes = [info["size"] for info in cluster_assignments.values()]
    cluster_ids = list(cluster_assignments.keys())

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Cluster sizes
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_ids)))
    bars = ax1.bar(
        cluster_ids, cluster_sizes, color=colors, edgecolor="black", linewidth=1.5
    )
    ax1.set_xlabel("Cluster ID", fontsize=12, weight="bold")
    ax1.set_ylabel("Number of Samples", fontsize=12, weight="bold")
    ax1.set_title("Cluster Size Distribution", fontsize=14, weight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    # Plot 2: Independence analysis
    # Handle different independence_analysis structures
    independent_count = 0
    dependent_count = 0
    alpha_local = 0.05
    decision_mode = "unknown"

    if isinstance(independence_analysis, dict):
        # Case A: New summary-only structure from ClusterDecomposer
        if ("alpha_local" in independence_analysis) or (
            "decision_mode" in independence_analysis
        ):
            alpha_local = independence_analysis.get("alpha_local", 0.05)
            decision_mode = independence_analysis.get("decision_mode", "unknown")
            # We don't track per-parent decisions here. Show conservative pie.
            independent_count = 0
            dependent_count = len(cluster_ids)
        else:
            # Case B: Older per-parent structure: {parent_node: {avg_correlation, are_independent, ...}}
            vals = list(independence_analysis.values())
            if vals and isinstance(vals[0], dict):
                for v in vals:
                    indep = bool(v.get("are_independent", False))
                    if indep:
                        independent_count += 1
                    else:
                        dependent_count += 1
                decision_mode = "per_parent"
            else:
                # Unknown dict content
                dependent_count = len(cluster_ids)
    else:
        # Unknown/None
        dependent_count = len(cluster_ids)

    pie_data = [independent_count, dependent_count]
    pie_labels = [
        f"Independent\n({independent_count})",
        f"Dependent\n({dependent_count})",
    ]
    pie_colors = ["#90EE90", "#FF6B6B"]

    ax2.pie(
        pie_data,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11, "weight": "bold"},
    )
    ax2.set_title(
        f"Sibling Independence\n(α={alpha_local}, mode={decision_mode})",
        fontsize=14,
        weight="bold",
    )
    plt.tight_layout()
    return fig, (ax1, ax2)
