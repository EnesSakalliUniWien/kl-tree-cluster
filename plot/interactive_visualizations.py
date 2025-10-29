"""
Interactive visualization functions for PosetTree using Plotly.

This module provides clean, interactive visualizations that are easier
to read and study than static matplotlib plots.
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Dict, Union
import networkx as nx


def create_interactive_sunburst(
    tree,
    test_results: Optional[Union[pd.DataFrame, Dict]] = None,
    dependence_column: str = "Are_Features_Dependent",
    use_labels: bool = True,
    show_values: bool = True,
    color_scheme: str = "RdYlBu_r",
    title: str = "Interactive Hierarchical Tree",
) -> go.Figure:
    """
    Create an interactive sunburst chart for hierarchical tree visualization.

    Sunburst charts are MUCH cleaner than traditional dendrograms for studying
    hierarchical structures:
    - Click to zoom into subtrees
    - Hover to see detailed information
    - Color-coded by statistical significance
    - Automatic label placement (no overlap!)

    Args:
        tree: PosetTree instance
        test_results: DataFrame or dict with test results
            If DataFrame: should have dependence_column
            If dict: {node_id: is_dependent (bool)}
        dependence_column: Column name for dependence indicator
        use_labels: If True, use leaf labels; if False, use node IDs
        show_values: If True, show values (leaf counts) in hover
        color_scheme: Plotly color scheme
        title: Chart title

    Returns:
        Plotly Figure object (can be shown with fig.show() or saved)

    Example:
        >>> fig = create_interactive_sunburst(tree, results_df)
        >>> fig.show()  # Opens in browser
        >>> fig.write_html("tree_visualization.html")  # Save to file
    """
    # Parse test results
    dep_dict = {}
    if test_results is not None:
        if isinstance(test_results, pd.DataFrame):
            if "Node" in test_results.columns:
                dep_dict = dict(
                    zip(test_results["Node"], test_results[dependence_column])
                )
            else:
                dep_dict = test_results[dependence_column].to_dict()
        else:
            dep_dict = test_results

        # Normalize to boolean
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
        dep_dict = normalized_dep

    # Compute descendant sets - use indices for cleaner display
    poset_dict_indices = tree.compute_descendant_sets(use_labels=False)

    # Build hierarchy for sunburst
    ids = []
    labels = []
    parents = []
    values = []
    colors = []
    hover_texts = []

    # Find root node
    root_candidates = [n for n in tree.nodes() if tree.in_degree(n) == 0]
    root = root_candidates[0] if root_candidates else list(tree.nodes())[0]

    # Traverse tree in BFS order
    from collections import deque

    queue = deque([root])
    visited = set()

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        # Get node info
        node_data = tree.nodes[node]
        is_leaf = node_data.get("is_leaf", False)
        descendants_indices = poset_dict_indices[node]
        value = len(descendants_indices)

        # Determine color
        if is_leaf:
            color_val = 0.0  # Green
            status = "Leaf"
        else:
            is_dependent = dep_dict.get(node, False)
            color_val = 1.0 if is_dependent else -1.0  # Red=dependent, Blue=independent
            status = "Features Dependent" if is_dependent else "Features Independent"

        # Create label (use original labels for node display)
        label = node_data.get("label", node) if use_labels else node

        # Format descendant set for display - use INDICES for compact display
        if len(descendants_indices) <= 8:
            desc_str = (
                "{" + ", ".join(str(x) for x in sorted(descendants_indices)) + "}"
            )
        else:
            desc_list = sorted(descendants_indices)[:6]
            desc_str = (
                "{"
                + ", ".join(str(x) for x in desc_list)
                + f", ... +{len(descendants_indices) - 6} more"
                + "}"
            )

        # Create hover text with poset encoding
        hover = f"<b>{node}</b><br>"
        hover += f"Type: {status}<br>"
        hover += f"Poset Encoding: {desc_str}<br>"
        hover += f"Leaf Count: {len(descendants_indices)}<br>"
        if not is_leaf and test_results is not None and node in dep_dict:
            hover += "<br>Statistical Test:<br>"
            hover += f"  Result: {status}"

        # Add to lists
        ids.append(node)
        labels.append(label)
        values.append(value)
        colors.append(color_val)
        hover_texts.append(hover)

        # Find parent
        predecessors = list(tree.predecessors(node))
        if predecessors:
            parents.append(predecessors[0])
        else:
            parents.append("")  # Root has no parent

        # Add children to queue
        for child in tree.successors(node):
            queue.append(child)

    # Create sunburst
    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors,
                colorscale=[
                    [0.0, "#87CEEB"],  # Blue - independent
                    [0.5, "#90EE90"],  # Green - leaves
                    [1.0, "#FF6B6B"],  # Red - dependent
                ],
                cmin=-1,
                cmax=1,
                showscale=False,
                line=dict(color="black", width=2),
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            branchvalues="total",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, family="Arial Black")),
        margin=dict(t=60, l=0, r=0, b=0),
        height=800,
    )

    return fig


def create_interactive_tree_network(
    tree,
    test_results: Optional[Union[pd.DataFrame, Dict]] = None,
    dependence_column: str = "Are_Features_Dependent",
    use_labels: bool = True,
    layout: str = "tree",
) -> go.Figure:
    """
    Create an interactive network visualization using Plotly.

    This provides a cleaner alternative to matplotlib networkx visualizations:
    - Hover to see node details
    - Zoom and pan
    - Clean edge routing
    - No label overlap

    Args:
        tree: PosetTree instance
        test_results: DataFrame or dict with test results
        dependence_column: Column name for dependence indicator
        use_labels: If True, use leaf labels; if False, use node IDs
        layout: Layout algorithm ('tree', 'radial', or 'force')

    Returns:
        Plotly Figure object
    """
    # Parse test results
    dep_dict = {}
    if test_results is not None:
        if isinstance(test_results, pd.DataFrame):
            if "Node" in test_results.columns:
                dep_dict = dict(
                    zip(test_results["Node"], test_results[dependence_column])
                )
            else:
                dep_dict = test_results[dependence_column].to_dict()
        else:
            dep_dict = test_results

        # Normalize to boolean
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
        dep_dict = normalized_dep

    # Get positions
    if layout == "tree":
        try:
            pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")
        except Exception:
            pos = nx.spring_layout(tree, k=2, iterations=50)
    elif layout == "radial":
        pos = nx.shell_layout(tree)
    else:  # force
        pos = nx.spring_layout(tree, k=2, iterations=50)

    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in tree.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Compute poset encoding (descendant leaf sets) - use indices for cleaner display
    poset_dict = tree.compute_descendant_sets(use_labels=False)

    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_hover_text = []
    node_sizes = []

    for node in tree.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Get node info
        node_data = tree.nodes[node]
        is_leaf = node_data.get("is_leaf", False)

        # Get poset encoding (descendant set) - format as indices
        desc_set = poset_dict[node]
        if len(desc_set) <= 8:
            # Show all indices if 8 or fewer
            desc_str = "{" + ", ".join(str(x) for x in sorted(desc_set)) + "}"
        else:
            # Show first few and count
            desc_list = sorted(desc_set)[:6]
            desc_str = (
                "{"
                + ", ".join(str(x) for x in desc_list)
                + f", ... +{len(desc_set) - 6} more"
                + "}"
            )

        # Determine color and text
        if is_leaf:
            node_colors.append("#90EE90")
            node_text.append(f"{node}")
            node_hover_text.append(f"{node}<br>Leaf<br>Poset: {desc_str}")
            node_sizes.append(15)
        else:
            is_dependent = dep_dict.get(node, False)
            if is_dependent:
                node_colors.append("#FF6B6B")
                node_text.append(f"{node}")
                node_hover_text.append(
                    f"{node}<br>Features Dependent<br>Poset: {desc_str}"
                )
            else:
                node_colors.append("#87CEEB")
                node_text.append(f"{node}")
                node_hover_text.append(
                    f"{node}<br>Features Independent<br>Poset: {desc_str}"
                )
            node_sizes.append(25)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        hovertext=node_hover_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color="black"),
        ),
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Interactive Tree Network", font=dict(size=20)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            plot_bgcolor="white",
        ),
    )

    return fig


def save_interactive_visualization(
    fig: go.Figure,
    filename: str = "tree_visualization.html",
    auto_open: bool = True,
) -> str:
    """
    Save Plotly figure to HTML file.

    Args:
        fig: Plotly Figure
        filename: Output filename
        auto_open: If True, automatically open in browser

    Returns:
        Path to saved file
    """
    fig.write_html(filename, auto_open=auto_open)
    return filename
