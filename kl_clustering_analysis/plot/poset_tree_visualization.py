"""
Visualization functions for PosetTree with poset encoding.

This module provides functions to visualize hierarchical trees with their
poset (partially ordered set) encoding, showing each node's descendant leaf set.
Includes independence-based coloring for statistical analysis visualization.
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import Optional, Tuple, Dict, Union, List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def _get_tree_layout(tree, layout: str = "hierarchical") -> Dict:
    """
    Compute node positions for tree visualization.

    Args:
        tree: NetworkX graph (PosetTree)
        layout: 'hierarchical' (uses Graphviz dot) or 'spring'

    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    layout = (layout or "hierarchical").lower()

    if layout == "hierarchical":
        try:
            return nx.nx_agraph.graphviz_layout(tree, prog="dot")
        except Exception:
            return nx.spring_layout(tree, k=2, iterations=50)

    if layout in ("radial", "circular_tree", "twopi"):
        try:
            roots = [n for n, d in tree.in_degree() if d == 0]
            root = roots[0] if roots else None
            if root is not None:
                return nx.nx_agraph.graphviz_layout(tree, prog="twopi", root=root)
            return nx.nx_agraph.graphviz_layout(tree, prog="twopi")
        except Exception:
            return nx.circular_layout(tree)

    if layout == "circular":
        return nx.circular_layout(tree)

    if layout == "spring":
        return nx.spring_layout(tree, k=2, iterations=50)

    raise ValueError(
        f"Unknown layout: {layout}. Use 'hierarchical', 'radial', 'circular', or 'spring'."
    )


def _draw_tree_structure(
    tree,
    pos: Dict,
    ax: Axes,
    node_colors: List[str],
    labels: Dict[str, str],
    node_size: int = 3000,
    font_size: int = 9,
    edge_color: str = "gray",
) -> None:
    """
    Draw the basic tree structure (edges, nodes, labels).

    Args:
        tree: NetworkX graph (PosetTree)
        pos: Node positions dictionary
        ax: Matplotlib axes
        node_colors: List of colors for each node
        labels: Dictionary mapping node IDs to label strings
        node_size: Size of nodes
        font_size: Font size for labels
        edge_color: Color for edges
    """
    # Draw edges
    nx.draw_networkx_edges(
        tree,
        pos,
        edge_color=edge_color,
        arrows=True,
        arrowsize=15,
        arrowstyle="->",
        width=2,
        ax=ax,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        tree,
        pos,
        node_color=node_colors,
        node_size=node_size,
        node_shape="o",
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(
        tree,
        pos,
        labels,
        font_size=font_size,
        font_weight="bold",
        font_family="monospace",
        ax=ax,
    )

    ax.axis("off")


def _format_set_for_label(desc_set: frozenset, max_elements: int = 4) -> str:
    """
    Format a frozenset for display in node labels.

    Args:
        desc_set: Frozenset of elements to format
        max_elements: Maximum number of elements to show before truncating

    Returns:
        Formatted string representation of the set

    Example:
        >>> _format_set_for_label(frozenset({'A', 'B', 'C'}), max_elements=4)
        '{A, B, C}'
        >>> _format_set_for_label(frozenset({'A', 'B', 'C', 'D', 'E'}), max_elements=3)
        '{A, B, C, ...+2}'
    """
    if len(desc_set) <= max_elements:
        return "{" + ", ".join(sorted(desc_set)) + "}"
    else:
        elements = sorted(desc_set)[: max_elements - 1]
        remaining = len(desc_set) - (max_elements - 1)
        return "{" + ", ".join(elements) + f", ...+{remaining}" + "}"


def plot_tree_with_poset_encoding(
    tree,
    use_labels: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    layout: str = "hierarchical",
    node_size: int = 3000,
    font_size: int = 9,
    leaf_color: str = "#90EE90",
    internal_color: str = "#87CEEB",
    edge_color: str = "gray",
    max_elements_display: int = 4,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes, Dict]:
    """
    Visualize a PosetTree with poset encoding showing descendant leaf sets.

    Args:
        tree: PosetTree instance to visualize
        use_labels: If True, use leaf labels in sets. If False, use node IDs.
        figsize: Figure size as (width, height) tuple
        layout: Layout algorithm - 'hierarchical' (uses Graphviz dot) or 'spring'
        node_size: Size of nodes in the visualization
        font_size: Font size for node labels
        leaf_color: Color for leaf nodes (hex or named color)
        internal_color: Color for internal nodes (hex or named color)
        edge_color: Color for edges
        max_elements_display: Maximum number of set elements to show before truncating
        ax: Optional matplotlib Axes to plot on. If None, creates new figure.

    Returns:
        Tuple of (figure, axes, poset_dict) where poset_dict maps node IDs to
        their descendant leaf sets.

    Example:
        >>> from tree.poset_tree import PosetTree
        >>> tree = PosetTree.from_linkage(Z, leaf_names=['A', 'B', 'C', 'D'])
        >>> fig, ax, poset_dict = plot_tree_with_poset_encoding(tree)
        >>> plt.show()
    """
    # Compute the poset encoding (descendant sets)
    poset_dict = tree.compute_descendant_sets(use_labels=use_labels)

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Determine layout
    pos = _get_tree_layout(tree, layout)

    # Color nodes by type (leaf vs internal)
    node_colors = [
        leaf_color if tree.nodes[node].get("is_leaf", False) else internal_color
        for node in tree.nodes()
    ]

    # Create labels with poset encoding
    labels = {}
    for node in tree.nodes():
        desc_set = poset_dict[node]
        set_str = _format_set_for_label(desc_set, max_elements_display)
        labels[node] = f"{node}\n{set_str}"

    # Draw tree structure
    _draw_tree_structure(
        tree, pos, ax, node_colors, labels, node_size, font_size, edge_color
    )

    # Set title
    ax.set_title(
        "Hierarchical Tree with Poset Encoding\n"
        "(Each node shows its descendant leaf set)",
        fontsize=16,
        weight="bold",
        pad=20,
    )

    return fig, ax, poset_dict


def plot_tree_with_independence_coloring(
    tree,
    test_results: Union[pd.DataFrame, Dict],
    dependence_column: str = "Are_Features_Dependent",
    test_columns: Optional[Dict[str, str]] = None,
    use_labels: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    layout: str = "hierarchical",
    node_size: int = 3000,
    font_size: int = 9,
    leaf_color: str = "#90EE90",
    dependent_color: str = "#FF6B6B",
    independent_color: str = "#87CEEB",
    edge_color: str = "gray",
    show_poset_labels: bool = True,
    max_elements_display: int = 4,
    show_legend: bool = True,
    show_test_details: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes, Dict]:
    """
    Visualize PosetTree with nodes colored by feature independence test results.

    This function colors internal nodes based on independence test results:
    - Dependent nodes (red): Features are dependent - node has meaningful structure
    - Independent nodes (blue): Features are independent - node lacks structure
    - Leaf nodes (green): Individual samples (not tested)

    Args:
        tree: PosetTree instance to visualize
        test_results: DataFrame or dict with test results
            If DataFrame: must have columns matching nodes (index or 'Node' column)
                and a column specified by dependence_column
            If dict: {node_id: True/False} indicating if features are dependent
        dependence_column: Column name for overall dependence indicator
            (default: 'Are_Features_Dependent'). Values should be boolean or
            'Features Dependent'/'Features Independent' strings
        test_columns: Optional dict mapping test names to column names for detailed results
            Example: {
                'Conservative': 'Independence_Conservative_Result',
                'Liberal': 'Independence_Liberal_Result'
            }
            If provided and show_test_details=True, will show which tests found dependence
        use_labels: If True, show poset encoding in labels
        figsize: Figure size as (width, height) tuple
        layout: 'hierarchical' (Graphviz dot) or 'spring'
        node_size: Size of nodes
        font_size: Font size for labels
        leaf_color: Color for leaf nodes (hex or named)
        dependent_color: Color for dependent nodes (default: red)
        independent_color: Color for independent nodes (default: blue)
        edge_color: Color for edges
        show_poset_labels: If True, show descendant sets in labels
        max_elements_display: Max set elements before truncation
        show_legend: If True, show legend explaining colors
        show_test_details: If True and test_columns provided, show which tests found dependence
        ax: Optional matplotlib Axes. If None, creates new figure.

    Returns:
        Tuple of (figure, axes, dependence_dict) where dependence_dict
        maps node IDs to their feature dependence status.

    Example:
        >>> # After running statistical tests
        >>> results_df = annotate_root_node_significance(
        ...     node_stats_df,
        ...     total_number_of_features=10
        ... )
        >>> fig, ax, dep_dict = plot_tree_with_independence_coloring(
        ...     tree,
        ...     results_df,
        ...     dependence_column='Are_Features_Dependent',
        ...     test_columns={
        ...         'Conservative': 'Independence_Conservative_Result',
        ...         'Liberal': 'Independence_Liberal_Result'
        ...     }
        ... )
        >>> plt.show()
    """
    # Parse test results
    if isinstance(test_results, pd.DataFrame):
        # Try to find node column
        if "Node" in test_results.columns:
            dep_dict = dict(
                zip(
                    test_results["Node"],
                    test_results[dependence_column],
                )
            )
            # Get test details if requested
            test_details = {}
            if test_columns and show_test_details:
                for test_name, col_name in test_columns.items():
                    if col_name in test_results.columns:
                        test_details[test_name] = dict(
                            zip(test_results["Node"], test_results[col_name])
                        )
        else:
            # Use index as node names
            dep_dict = test_results[dependence_column].to_dict()
            # Get test details if requested
            test_details = {}
            if test_columns and show_test_details:
                for test_name, col_name in test_columns.items():
                    if col_name in test_results.columns:
                        test_details[test_name] = test_results[col_name].to_dict()
    else:
        dep_dict = test_results
        test_details = {}

    # Normalize dependence values to boolean
    normalized_dep = {}
    for node, value in dep_dict.items():
        if isinstance(value, bool):
            normalized_dep[node] = value
        elif isinstance(value, str):
            normalized_dep[node] = value.lower() in [
                "features dependent",
                "dependent",
                "true",
            ]
        else:
            normalized_dep[node] = bool(value)

    # Compute poset encoding if needed for labels
    if show_poset_labels:
        poset_dict = tree.compute_descendant_sets(use_labels=use_labels)
    else:
        poset_dict = {}

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Determine layout
    pos = _get_tree_layout(tree, layout)

    # Color nodes based on dependence
    node_colors = []
    for node in tree.nodes():
        if tree.nodes[node].get("is_leaf", False):
            node_colors.append(leaf_color)
        else:
            # Internal node - check if features are dependent
            is_dependent = normalized_dep.get(node, False)
            if is_dependent:
                node_colors.append(dependent_color)
            else:
                node_colors.append(independent_color)

    # Create labels
    labels = {}
    for node in tree.nodes():
        label_parts = [node]

        # Add poset encoding if requested
        if show_poset_labels and node in poset_dict:
            desc_set = poset_dict[node]
            set_str = _format_set_for_label(desc_set, max_elements_display)
            label_parts.append(set_str)

        # Add test details if requested and node is internal
        if (
            show_test_details
            and test_details
            and not tree.nodes[node].get("is_leaf", False)
            and node in normalized_dep
        ):
            # Show which tests found dependence
            dependent_tests = []
            for test_name, results in test_details.items():
                if node in results:
                    result_str = str(results[node])
                    if "dependent" in result_str.lower():
                        dependent_tests.append(test_name)

            if dependent_tests:
                label_parts.append(f"({', '.join(dependent_tests)})")

        labels[node] = "\n".join(label_parts)

    # Draw tree structure
    _draw_tree_structure(
        tree, pos, ax, node_colors, labels, node_size, font_size, edge_color
    )

    # Add title
    ax.set_title(
        "Hierarchical Tree with Feature Independence Test Results\n"
        "(Red: Features Dependent, Blue: Features Independent, Green: Leaves)",
        fontsize=16,
        weight="bold",
        pad=20,
    )

    # Add legend
    if show_legend:
        legend_elements = [
            Patch(facecolor=leaf_color, edgecolor="black", label="Leaf Nodes"),
            Patch(
                facecolor=dependent_color,
                edgecolor="black",
                label="Features Dependent (Has Structure)",
            ),
            Patch(
                facecolor=independent_color,
                edgecolor="black",
                label="Features Independent (Random)",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    return fig, ax, normalized_dep


def print_poset_encoding(poset_dict: Dict, sort_keys: bool = True) -> None:
    """
    Print the poset encoding (set-of-sets representation) to console.

    Args:
        poset_dict: Dictionary mapping node IDs to frozensets of descendant leaves
        sort_keys: If True, sort the keys before printing

    Example:
        >>> poset_dict = tree.compute_descendant_sets(use_labels=True)
        >>> print_poset_encoding(poset_dict)
          L0: frozenset({'Sample_0'})
          L1: frozenset({'Sample_1'})
          N16: frozenset({'Sample_0', 'Sample_1', ...})
    """
    keys = sorted(poset_dict.keys()) if sort_keys else poset_dict.keys()
    for node_id in keys:
        desc_set = poset_dict[node_id]
        print(f"  {node_id}: {desc_set}")


def print_poset_properties(tree, poset_dict: Dict) -> None:
    """
    Print statistical properties and relationships of the poset.

    Args:
        tree: PosetTree instance
        poset_dict: Dictionary mapping node IDs to frozensets

    Example:
        >>> print_poset_properties(tree, poset_dict)
          Total nodes: 15
          Leaf nodes: 8
          Internal nodes: 7

          Subset relations (examples):
            N16 ⊆ N22: frozenset({'Sample_0', 'Sample_1'}) ⊆ frozenset({...})
    """
    print(f"  Total nodes: {len(poset_dict)}")
    print(
        f"  Leaf nodes: {sum(1 for n in tree.nodes() if tree.nodes[n].get('is_leaf', False))}"
    )
    print(
        f"  Internal nodes: {sum(1 for n in tree.nodes() if not tree.nodes[n].get('is_leaf', False))}"
    )
    print(f"\n  Subset relations (examples):")

    # Show some subset relationships
    internal_nodes = [
        n for n in tree.nodes() if not tree.nodes[n].get("is_leaf", False)
    ]
    if len(internal_nodes) >= 2:
        for i, n1 in enumerate(internal_nodes[:3]):
            for n2 in internal_nodes[i + 1 : i + 3]:
                if poset_dict[n1] <= poset_dict[n2]:
                    print(f"    {n1} ⊆ {n2}: {poset_dict[n1]} ⊆ {poset_dict[n2]}")
                elif poset_dict[n2] <= poset_dict[n1]:
                    print(f"    {n2} ⊆ {n1}: {poset_dict[n2]} ⊆ {poset_dict[n1]}")

