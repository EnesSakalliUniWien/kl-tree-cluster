"""
Significance visualization for hierarchical clustering.

Functions for creating enhanced dendrograms with statistical significance
annotations and visual indicators for hypothesis testing results.
"""

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_enhanced_dendrogram_with_significance(Z, leaf_names, node_summary_df):
    """
    Create enhanced dendrogram with significance annotations and visual indicators.

    Parameters:
    -----------
    Z : ndarray
        Linkage matrix from hierarchical clustering
    leaf_names : list
        Names of leaf nodes in dendrogram order
    node_summary_df : DataFrame
        Summary of node significance results with columns: 'Node', 'P_Value', 'Bonferroni_Result'
    """
    # Create a mapping from node names to their statistical data
    node_info = {}
    for _, row in node_summary_df.iterrows():
        node_info[row["Node"]] = {
            "p_value": row["P_Value"],
            "bonferroni_significant": row["Bonferroni_Result"] == "Significant",
        }

    # Create the enhanced dendrogram with larger figure for better visibility
    plt.figure(figsize=(14, 8))
    dendro_plot = dendrogram(Z, labels=leaf_names, leaf_rotation=90, color_threshold=0)

    # Extract coordinate information from dendrogram for annotation placement
    icoord = dendro_plot["icoord"]
    dcoord = dendro_plot["dcoord"]

    # Use linkage matrix to map dendrogram order to node names
    num_leaves = len(leaf_names)
    num_merges = Z.shape[0]

    # Build mapping from dendrogram position to linkage matrix index
    dendro_to_linkage = {}

    for dendro_idx in range(len(dcoord)):
        y_height = dcoord[dendro_idx][1]
        x_center = (icoord[dendro_idx][1] + icoord[dendro_idx][2]) / 2

        # Find all linkage matrix rows with this height
        height_matches = []
        for link_idx in range(num_merges):
            if abs(Z[link_idx, 2] - y_height) < 1e-10:
                height_matches.append(link_idx)

        if len(height_matches) == 1:
            dendro_to_linkage[dendro_idx] = height_matches[0]
        elif len(height_matches) > 1:
            unassigned = [
                idx for idx in height_matches if idx not in dendro_to_linkage.values()
            ]

            if len(unassigned) == 1:
                dendro_to_linkage[dendro_idx] = unassigned[0]
            elif len(unassigned) > 1:
                best_match = None
                best_distance = float("inf")

                for candidate_idx in unassigned:
                    left_idx = int(Z[candidate_idx, 0])
                    right_idx = int(Z[candidate_idx, 1])

                    if left_idx < num_leaves:
                        left_x = dendro_plot["ivl"].index(leaf_names[left_idx]) * 10
                    else:
                        left_dendro_idx = [
                            k
                            for k, v in dendro_to_linkage.items()
                            if v == left_idx - num_leaves
                        ]
                        if left_dendro_idx:
                            left_x = (
                                icoord[left_dendro_idx[0]][1]
                                + icoord[left_dendro_idx[0]][2]
                            ) / 2
                        else:
                            left_x = x_center

                    if right_idx < num_leaves:
                        right_x = dendro_plot["ivl"].index(leaf_names[right_idx]) * 10
                    else:
                        right_dendro_idx = [
                            k
                            for k, v in dendro_to_linkage.items()
                            if v == right_idx - num_leaves
                        ]
                        if right_dendro_idx:
                            right_x = (
                                icoord[right_dendro_idx[0]][1]
                                + icoord[right_dendro_idx[0]][2]
                            ) / 2
                        else:
                            right_x = x_center

                    expected_x = (left_x + right_x) / 2
                    distance = abs(expected_x - x_center)

                    if distance < best_distance:
                        best_distance = distance
                        best_match = candidate_idx

                dendro_to_linkage[dendro_idx] = (
                    best_match if best_match is not None else unassigned[0]
                )

    # Add annotations for each internal node
    for i in range(len(icoord)):
        x_pos = (icoord[i][1] + icoord[i][2]) / 2
        y_pos = dcoord[i][1]

        # Get the correct linkage matrix index
        if i in dendro_to_linkage:
            link_idx = dendro_to_linkage[i]
            if link_idx == num_merges - 1:
                node_name = "R"
            else:
                node_name = f"N{num_merges - link_idx}"
        else:
            if i == num_merges - 1:
                node_name = "R"
            else:
                node_name = f"N{num_merges - i}"

        # Get node information if available
        if node_name in node_info:
            p_val = node_info[node_name]["p_value"]
            is_significant = node_info[node_name]["bonferroni_significant"]

            # Choose visual style based on significance
            color = "red" if is_significant else "blue"
            marker = "o" if is_significant else "s"

            # Plot the node marker
            plt.plot(
                x_pos,
                y_pos,
                marker,
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )

            # Add text annotation with node name and p-value
            plt.annotate(
                f"{node_name}\np={p_val:.2e}",
                xy=(x_pos, y_pos),
                xytext=(5, 10),
                textcoords="offset points",
                fontsize=9,
                color=color,
                weight="bold" if is_significant else "normal",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.8,
                ),
            )

    # Add title and labels
    plt.title(
        "Enhanced Dendrogram with Node Identifiers and P-Values\n"
        + "Red circles = Bonferroni significant, Blue squares = Not significant"
    )
    plt.xlabel("Leaf Nodes")
    plt.ylabel("Distance (Hamming)")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Significant (Bonferroni)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            markersize=8,
            label="Not Significant",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()

    return dendro_plot
