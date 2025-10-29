"""
Similarity analysis for hierarchical clustering.

Functions for analyzing similarity patterns and improvements across
hierarchical tree structures based on similarity metrics.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist
from typing import List


def _calculate_node_similarity(
    node_leaves: List[str], sample_features: pd.DataFrame
) -> float:
    """
    Calculate mean similarity within a node's subtree.

    Similarity measures how alike the leaf samples are to each other.
    For binary data: similarity = fraction of matching positions

    Example with 2 samples:
        Sample A: [0, 1, 0, 1]
        Sample B: [0, 1, 1, 1]
        Matches:   T  T  F  T  → 3 out of 4 positions match
        Similarity = 0.75 (75% similar)

    For multiple samples: average similarity across all pairs

    Args:
        node_leaves: List of leaf labels in the subtree
        sample_features: DataFrame where rows are leaves (indexed by leaf labels)
            and columns are features

    Returns:
        Mean similarity (1 - Hamming distance) within the subtree.
        Returns 1.0 for single-leaf nodes (perfectly similar to itself).
    """
    # If only one leaf, it's perfectly similar to itself
    if len(node_leaves) < 2:
        return 1.0

    # Get the data rows for these specific leaves from the full DataFrame
    node_data = sample_features.loc[node_leaves].values

    # Special case: exactly 2 leaves
    if len(node_leaves) == 2:
        # Compare element-by-element: True where different, False where same
        differences = node_data[0] != node_data[1]
        # Count fraction of positions that are DIFFERENT
        fraction_different = np.mean(differences)
        # Similarity = 1 - fraction_different (fraction that MATCH)
        similarity = 1 - fraction_different
    else:
        # General case: 3 or more leaves
        # Calculate Hamming distance for all pairs (how many positions differ)
        intra_distances = pdist(node_data, metric="hamming")
        # Convert distances to similarities (1 - distance = fraction matching)
        intra_similarities = 1 - intra_distances
        # Average similarity across all pairs
        similarity = np.mean(intra_similarities)

    return similarity


def _create_node_analysis_record(
    node_name: str,
    node_leaves: List[str],
    node_similarity: float,
    similarity_improvement: float,
    node_significance: dict,
    parent_name: str,
    parent_significant: bool,
) -> dict:
    """
    Create a dictionary record for a node's similarity analysis results.

    Args:
        node_name: Node identifier (e.g., 'N5')
        node_leaves: List of leaf labels under this node
        node_similarity: Average similarity within this node's cluster
        similarity_improvement: Improvement over baseline similarity
        node_significance: Dictionary mapping node names to significance (True/False)
        parent_name: Parent node identifier (None if root)
        parent_significant: Whether parent is significant

    Returns:
        Dictionary with all analysis results for this node:
            - 'Node': Node identifier
            - 'Is_Significant': Whether node passes significance test
            - 'Num_Leaves': Number of leaf descendants
            - 'Similarity': Mean similarity within node's subtree
            - 'Similarity_Improvement': Improvement over baseline
            - 'Parent': Parent node identifier (None for root)
            - 'Parent_Significant': Whether parent is significant
            - 'Leaves': Comma-separated list of leaf labels
    """
    return {
        "Node": node_name,  # Node identifier
        "Is_Significant": node_significance[
            node_name
        ],  # Whether node passes significance test
        "Num_Leaves": len(node_leaves),  # Number of leaf descendants
        "Similarity": node_similarity,  # Mean similarity within node's subtree
        "Similarity_Improvement": similarity_improvement,  # Improvement over baseline
        "Parent": parent_name,  # Parent node identifier (None for root)
        "Parent_Significant": parent_significant,  # Whether parent is significant
        "Leaves": ", ".join(sorted(node_leaves)),  # Comma-separated list of leaf labels
    }


def analyze_hierarchy_similarity_patterns(
    tree: "nx.DiGraph",
    significance_results: pd.DataFrame,
    sample_features: pd.DataFrame,
    significance_column: str = "Bonferroni_Result",
) -> pd.DataFrame:
    """
    Analyze similarity patterns and improvements across the hierarchical tree.

    This function evaluates how well each node's subtree clusters together
    by measuring how similar the leaves are within each node and comparing
    to the overall baseline similarity.

    Simple explanation:
        - Baseline similarity: average similarity across ALL leaf pairs in dataset
        - Node similarity: average similarity within a specific cluster/node
        - Improvement: how much better the node's cluster is vs random pairing

    Example:
        If baseline similarity is 0.60 (60% matching on average) and a node
        has similarity 0.85, then improvement is +0.25 (25% better clustering).

    Args:
        tree: PosetTree instance (NetworkX DiGraph) representing the hierarchy.
            Nodes should have 'is_leaf' and 'label' attributes.
    significance_results: DataFrame with significance test results for each node.
            Required columns: significance_column; index must be node ids
        sample_features: DataFrame where each row is a leaf sample (indexed by leaf labels)
            and columns are features. Index must match leaf labels in the tree.
        significance_column: Column name in significance_results indicating significance
            (default: 'Bonferroni_Result')

    Returns:
        DataFrame with similarity analysis for each internal node:
            - 'Node': Node identifier
            - 'Is_Significant': Whether node passes significance test
            - 'Num_Leaves': Number of leaf descendants
            - 'Similarity': Mean similarity within node's subtree
            - 'Similarity_Improvement': Improvement over baseline
            - 'Parent': Parent node identifier (None for root)
            - 'Parent_Significant': Whether parent is significant
            - 'Leaves': Comma-separated list of leaf labels
    """
    # Step 1: Calculate baseline similarity across ALL leaves
    # This is our reference point - how similar are random pairs?
    data_matrix = sample_features.values  # Convert DataFrame to numpy array
    overall_distances = pdist(
        data_matrix, metric="hamming"
    )  # All pairwise Hamming distances
    overall_similarities = 1 - overall_distances  # Convert distances to similarities
    baseline_similarity = np.mean(
        overall_similarities
    )  # Average similarity across all pairs

    # Step 2: Extract significance information from summary DataFrame
    # Build a dictionary: node_name -> is_significant (True/False)
    node_significance = {}
    for node_id, row in significance_results.iterrows():  # Loop through each row
        node_significance[str(node_id)] = row[significance_column] == "Significant"

    # Step 3: Analyze each internal node
    analysis_results = []
    for node_name in node_significance.keys():  # Loop through each node we're testing
        # Get all leaf labels under this node
        node_leaves = tree.get_leaves(node=node_name, return_labels=True)
        # Calculate average similarity within this node's cluster
        node_similarity = _calculate_node_similarity(node_leaves, sample_features)
        # How much better is this cluster than random? (positive = good clustering)
        similarity_improvement = node_similarity - baseline_similarity

        # Get this node's parent using NetworkX (None if root)
        # In a tree, each node has exactly one parent (or zero if root)
        parent_name = next(tree.predecessors(node_name), None)
        # Is the parent also significant? (False if no parent or parent not significant)
        parent_significant = (
            node_significance.get(parent_name, False) if parent_name else False
        )

        # Create summary statistics for this node and add to results
        node_summary = _create_node_analysis_record(
            node_name=node_name,
            node_leaves=node_leaves,
            node_similarity=node_similarity,
            similarity_improvement=similarity_improvement,
            node_significance=node_significance,
            parent_name=parent_name,
            parent_significant=parent_significant,
        )
        analysis_results.append(node_summary)

    # Convert list of dictionaries to DataFrame and sort by cluster size
    return pd.DataFrame(analysis_results).sort_values("Num_Leaves")


def _extract_clusters_from_tree(
    tree: "nx.DiGraph",
    significance_results: pd.DataFrame,
    sample_features: pd.DataFrame,
    significance_column: str,
    similarity_threshold: float,
) -> tuple:
    """
    Extract clusters from tree using significance and similarity improvement.

    Args:
        tree: PosetTree instance (NetworkX DiGraph) representing the hierarchy
        significance_results: DataFrame with significance test results for each node
            Required columns: significance_column; index must be node ids
        sample_features: DataFrame where rows are leaves (indexed by leaf labels)
            and columns are features
        significance_column: Column name in significance_results indicating significance
        similarity_threshold: Threshold for similarity advantage over parent

    Returns:
        Tuple of (cluster_assignments, boundary_nodes, boundary_reasons):
            - cluster_assignments: Dict mapping leaf labels to cluster IDs
            - boundary_nodes: Set of node names that are cluster boundaries
            - boundary_reasons: Dict mapping boundary node names to reason strings
    """
    # Extract significance information from DataFrame
    node_significance = {}
    for node_id, row in significance_results.iterrows():
        node_significance[str(node_id)] = row[significance_column] == "Significant"

    # Calculate baseline similarity across all leaves
    data_matrix = sample_features.values
    overall_distances = pdist(data_matrix, metric="hamming")
    overall_similarities = 1 - overall_distances
    baseline_similarity = np.mean(overall_similarities)

    cluster_root_nodes = []
    boundary_reasons = {}
    similarity_analysis = {}

    for node_name in node_significance.keys():
        is_significant = node_significance[node_name]
        # Get parent using NetworkX (in a tree, each node has exactly one parent)
        parent_name = next(tree.predecessors(node_name), None)

        if is_significant:
            # Get all leaf labels under this node using PosetTree method
            node_leaves = tree.get_leaves(node=node_name, return_labels=True)
            # Calculate similarity within this node's cluster
            node_similarity = _calculate_node_similarity(node_leaves, sample_features)
            similarity_improvement = node_similarity - baseline_similarity

            similarity_analysis[node_name] = {
                "node_similarity": node_similarity,
                "similarity_improvement": similarity_improvement,
                "num_leaves": len(node_leaves),
            }

            if parent_name is None:
                parent_significant = False
                parent_similarity_improvement = 0
            else:
                parent_significant = node_significance.get(parent_name, False)
                if parent_significant and parent_name in similarity_analysis:
                    parent_similarity_improvement = similarity_analysis[parent_name][
                        "similarity_improvement"
                    ]
                else:
                    parent_similarity_improvement = 0

            similarity_advantage = (
                similarity_improvement - parent_similarity_improvement
            )

            is_cluster_boundary = False
            reason = ""

            if not parent_significant:
                is_cluster_boundary = True
                reason = "significant with non-significant parent"
            elif similarity_advantage > similarity_threshold:
                is_cluster_boundary = True
                reason = f"similarity advantage over parent ({similarity_advantage:.4f} > {similarity_threshold})"

            if is_cluster_boundary:
                cluster_root_nodes.append(node_name)
                boundary_reasons[node_name] = reason

    # Assign leaves to clusters based on boundary nodes
    cluster_assignments = {}
    cluster_id = 0

    for root_node in cluster_root_nodes:
        # Use PosetTree's get_leaves() method instead of manual recursion
        leaves_in_cluster = tree.get_leaves(node=root_node, return_labels=True)
        for leaf in leaves_in_cluster:
            cluster_assignments[leaf] = cluster_id
        cluster_id += 1

    # Get all leaves in the tree to assign any unassigned leaves to singleton clusters
    all_leaves = tree.get_leaves(return_labels=True)
    for leaf in all_leaves:
        if leaf not in cluster_assignments:
            cluster_assignments[leaf] = cluster_id
            cluster_id += 1

    return cluster_assignments, set(cluster_root_nodes), boundary_reasons
