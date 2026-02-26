"""
Cluster tree visualization (NetworkX + matplotlib).

Design goals:
- Simple, systematic plotting
- Radial layout via Graphviz ``twopi`` when available
- Leaf node colors match cluster IDs (consistent with UMAP plots)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .cluster_color_mapping import build_cluster_color_spec, present_cluster_ids
from .config import (
    EDGE_DRAW_ORDER,
    EDGE_STYLES,
    HALO_SIZE_MULTIPLIER,
    HALO_SIZE_OFFSET,
    HALO_STYLES,
    INTERNAL_NODE_STYLE,
    LEAF_NODE_STYLE,
    LEGEND_NODE_MARKER_SIZE,
    UNASSIGNED_NODE_COLOR,
)

CHILD_PARENT_SIGNIFICANT_COL = "Child_Parent_Divergence_Significant"
SIBLING_DIFFERENT_COL = "Sibling_BH_Different"
SIBLING_SKIPPED_COL = "Sibling_Divergence_Skipped"


def _normalize_optional_bool(value: object) -> Optional[bool]:
    """Convert heterogeneous truthy/falsy values to bool or None."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return None
        if value in (0.0, 1.0):
            return bool(int(value))
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "t", "yes", "y", "1"}:
            return True
        if token in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _lookup_results_value(results_df, node_id: object, column_name: str) -> object | None:
    """Lookup a per-node value in ``results_df`` with robust id matching."""
    if results_df is None:
        return None

    columns = getattr(results_df, "columns", None)
    index = getattr(results_df, "index", None)
    if columns is None or index is None or column_name not in columns:
        return None

    for candidate in (node_id, str(node_id)):
        try:
            if candidate in index:
                return results_df.at[candidate, column_name]
        except Exception:
            continue
    return None


def _group_internal_nodes_for_halo(
    G: nx.DiGraph,
    leaves: set,
    results_df,
) -> tuple[list[object], list[object]]:
    """Split internal nodes into (significant, tested-not-significant)."""
    significant: list[object] = []
    tested_not_significant: list[object] = []

    for node in G.nodes():
        if node in leaves:
            continue
        # Root has no parent; child-parent test is undefined there.
        if G.in_degree(node) == 0:
            continue

        flag = _normalize_optional_bool(
            _lookup_results_value(results_df, node, CHILD_PARENT_SIGNIFICANT_COL)
        )
        if flag is True:
            significant.append(node)
        elif flag is False:
            tested_not_significant.append(node)

    return significant, tested_not_significant


def _group_edges_for_sibling_style(
    G: nx.DiGraph, results_df
) -> dict[str, list[tuple[object, object]]]:
    """Group edges by sibling-test status of the parent node."""
    groups: dict[str, list[tuple[object, object]]] = {
        "different": [],
        "not_different": [],
        "missing": [],
    }

    for parent, child in G.edges():
        skipped = _normalize_optional_bool(
            _lookup_results_value(results_df, parent, SIBLING_SKIPPED_COL)
        )
        if skipped is True:
            groups["missing"].append((parent, child))
            continue

        different = _normalize_optional_bool(
            _lookup_results_value(results_df, parent, SIBLING_DIFFERENT_COL)
        )
        if different is True:
            groups["different"].append((parent, child))
        elif different is False:
            groups["not_different"].append((parent, child))
        else:
            groups["missing"].append((parent, child))

    return groups


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


def _sorted_children(G: nx.DiGraph, node: object) -> list[object]:
    children = list(G.successors(node))
    try:
        return sorted(children)
    except TypeError:
        return sorted(children, key=lambda x: str(x))


def _rectangular_tree_layout(
    G: nx.DiGraph,
    *,
    x_gap: float = 1.0,
    y_gap: float = 1.0,
) -> Dict[object, Tuple[float, float]]:
    """Deterministic rectangular layout for a rooted directed tree (or forest).

    Invariant
    ---------
    - y(node) = -depth(node) * y_gap
    - leaves get increasing x; internal nodes are centered over children

    Notes
    -----
    - Works without Graphviz and is deterministic by construction.
    - If ``G`` is a forest, each root is laid out left-to-right with spacing.
    """
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        roots = [sorted(G.nodes, key=str)[0]] if G.nodes else []
    else:
        roots = sorted(roots, key=str)

    pos: Dict[object, Tuple[float, float]] = {}
    x_offset = 0.0

    for root in roots:
        component_nodes = {root} | nx.descendants(G, root)

        depth: Dict[object, int] = {root: 0}
        queue = deque([root])
        while queue:
            node = queue.popleft()
            d0 = depth[node]
            for child in _sorted_children(G, node):
                if child in component_nodes and child not in depth:
                    depth[child] = d0 + 1
                    queue.append(child)

        postorder: list[object] = []
        stack: list[tuple[object, bool]] = [(root, False)]
        seen: set[object] = set()
        while stack:
            node, expanded = stack.pop()
            if node not in component_nodes:
                continue
            if expanded:
                postorder.append(node)
                continue
            if node in seen:
                continue
            seen.add(node)
            stack.append((node, True))
            for child in reversed(_sorted_children(G, node)):
                stack.append((child, False))

        x_by_node: Dict[object, float] = {}
        leaf_cursor = x_offset
        for node in postorder:
            children = [c for c in _sorted_children(G, node) if c in component_nodes]
            if not children:
                x_by_node[node] = leaf_cursor
                leaf_cursor += x_gap
            else:
                x_by_node[node] = float(np.mean([x_by_node[c] for c in children]))

        for node in component_nodes:
            pos[node] = (
                x_by_node.get(node, x_offset),
                -float(depth.get(node, 0)) * y_gap,
            )

        x_offset = leaf_cursor + x_gap

    return pos


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

    # Deterministic fallback that still looks like a tree.
    return _rectangular_tree_layout(nx.DiGraph(G))


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
    layout: str = "rectangular",
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional["plt.Axes"] = None,
    show: bool = False,
):
    """
    Plot hierarchical tree with cluster assignments.

    Notes
    -----
    - This is intentionally a lightweight plot. ``show_cluster_boundaries`` is a
      placeholder (not implemented).
    - Leaf nodes are colored by their cluster ID; internal nodes are gray.
    """
    _ = show_cluster_boundaries
    _ = use_labels

    cluster_assignments = decomposition_results["cluster_assignments"]
    num_clusters = decomposition_results["num_clusters"]

    spec = build_cluster_color_spec(
        num_clusters, base_cmap=colormap, unassigned_color=UNASSIGNED_NODE_COLOR
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
    leaf_nodes = [n for n in G.nodes() if n in leaves]
    internal_nodes = [n for n in G.nodes() if n not in leaves]
    leaf_node_colors: list[str] = []
    for node in leaf_nodes:
        if node in node_to_cluster:
            try:
                cluster_id = int(node_to_cluster[node])
            except Exception:
                cluster_id = None
            leaf_node_colors.append(cluster_id_to_color.get(cluster_id, unassigned_color))
        else:
            leaf_node_colors.append(unassigned_color)

    if layout == "rectangular":
        pos = _rectangular_tree_layout(G)
    elif layout == "radial":
        pos = _graphviz_twopi_layout(G, args="")
    else:
        raise ValueError(f"Unknown layout={layout!r}; expected 'rectangular' or 'radial'.")

    if title is None:
        title = (
            f"Hierarchical Tree with Cluster Decomposition: "
            f"{num_clusters} Independent Clusters Identified"
        )

    if ax is None:
        if figsize is None:
            figsize = (max(width, 1) / 100.0, max(height, 1) / 100.0)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    edge_groups = _group_edges_for_sibling_style(G, results_df)
    for edge_group in EDGE_DRAW_ORDER:
        edgelist = edge_groups[edge_group]
        if not edgelist:
            continue
        edge_style = EDGE_STYLES[edge_group]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            edge_color=edge_style["edge_color"],
            width=edge_style["width"],
            style=edge_style["style"],
            arrows=False,
            ax=ax,
            alpha=edge_style["alpha"],
        )

    if internal_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=internal_nodes,
            node_size=node_size,
            alpha=INTERNAL_NODE_STYLE["alpha"],
            node_color=INTERNAL_NODE_STYLE["node_color"],
            linewidths=INTERNAL_NODE_STYLE["linewidths"],
            ax=ax,
        )

    if leaf_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=leaf_nodes,
            node_size=node_size,
            alpha=LEAF_NODE_STYLE["alpha"],
            node_color=leaf_node_colors,
            linewidths=LEAF_NODE_STYLE["linewidths"],
            edgecolors=LEAF_NODE_STYLE["edgecolors"],
            ax=ax,
        )

    halo_significant, halo_tested_not_significant = _group_internal_nodes_for_halo(
        G, leaves, results_df
    )
    halo_size = max(node_size * HALO_SIZE_MULTIPLIER, node_size + HALO_SIZE_OFFSET)
    if halo_tested_not_significant:
        halo_non_sig_style = HALO_STYLES["tested_not_significant"]
        halo_non_sig = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=halo_tested_not_significant,
            node_size=halo_size,
            node_color="none",
            edgecolors=halo_non_sig_style["edgecolors"],
            linewidths=halo_non_sig_style["linewidths"],
            ax=ax,
        )
        halo_non_sig.set_linestyle(halo_non_sig_style["linestyle"])

    if halo_significant:
        halo_sig_style = HALO_STYLES["significant"]
        halo_sig = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=halo_significant,
            node_size=halo_size,
            node_color="none",
            edgecolors=halo_sig_style["edgecolors"],
            linewidths=halo_sig_style["linewidths"],
            ax=ax,
        )
        halo_sig.set_linestyle(halo_sig_style["linestyle"])
    ax.set_title(title, fontsize=font_size)
    # Do not force equal aspect: trees with many leaves collapse vertically.
    ax.set_aspect("auto")
    ax.set_axis_off()

    leaf_cluster_ids = [
        int(node_to_cluster[n])
        for n in leaves
        if n in node_to_cluster and str(node_to_cluster[n]).lstrip("-").isdigit()
    ]
    present_ids = present_cluster_ids(leaf_cluster_ids)
    halo_sig_style = HALO_STYLES["significant"]
    halo_non_sig_style = HALO_STYLES["tested_not_significant"]
    edge_diff_style = EDGE_STYLES["different"]
    edge_not_diff_style = EDGE_STYLES["not_different"]
    edge_missing_style = EDGE_STYLES["missing"]
    style_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=halo_sig_style["edgecolors"],
            markerfacecolor="none",
            markeredgecolor=halo_sig_style["edgecolors"],
            linestyle=halo_sig_style["linestyle"],
            linewidth=halo_sig_style["linewidths"],
            markersize=LEGEND_NODE_MARKER_SIZE,
            label=halo_sig_style["legend_label"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=halo_non_sig_style["edgecolors"],
            markerfacecolor="none",
            markeredgecolor=halo_non_sig_style["edgecolors"],
            linestyle=halo_non_sig_style["linestyle"],
            linewidth=halo_non_sig_style["linewidths"],
            markersize=LEGEND_NODE_MARKER_SIZE,
            label=halo_non_sig_style["legend_label"],
        ),
        Line2D(
            [0],
            [0],
            color=edge_diff_style["edge_color"],
            linewidth=edge_diff_style["width"],
            linestyle=edge_diff_style["style"],
            label=edge_diff_style["legend_label"],
        ),
        Line2D(
            [0],
            [0],
            color=edge_not_diff_style["edge_color"],
            linewidth=edge_not_diff_style["width"],
            linestyle=edge_not_diff_style["style"],
            label=edge_not_diff_style["legend_label"],
        ),
        Line2D(
            [0],
            [0],
            color=edge_missing_style["edge_color"],
            linewidth=edge_missing_style["width"],
            linestyle=edge_missing_style["style"],
            label=edge_missing_style["legend_label"],
        ),
    ]

    if present_ids:
        handles = [
            Patch(facecolor=cluster_id_to_color[cid], edgecolor="none", label=f"{cid}")
            for cid in present_ids
            if cid in cluster_id_to_color
        ]
    else:
        handles = []

    handles.extend(style_handles)
    if handles:
        ax.legend(
            handles=handles,
            title="Legend",
            loc="best",
            frameon=False,
            fontsize=max(font_size - 1, 6),
            title_fontsize=max(font_size - 1, 6),
        )

    if show:
        import warnings

        warnings.warn(
            (
                "plot_tree_with_clusters: 'show' is deprecated and will be removed; "
                "save figures externally instead."
            ),
            DeprecationWarning,
        )

    return fig, ax
