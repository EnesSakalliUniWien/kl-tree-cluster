"""Complete trace of the decomposition."""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import networkx as nx

from kl_clustering_analysis.benchmarking.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

# Generate data
data_dict, cluster_dict = generate_random_feature_matrix(
    n_rows=50,
    n_cols=30,
    n_clusters=3,
    entropy_param=0.05,
    balanced_clusters=True,
    random_seed=42,
)
data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)

# Build tree
Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
decomposition = tree.decompose(leaf_data=data_df)

stats_df = tree.stats_df


def get_leaves(node):
    """Get all leaves under a node."""
    return [n for n in nx.descendants(tree, node) if tree.nodes[n].get("is_leaf")] or [
        node
    ]


def should_split(node):
    """Manual implementation of _should_split logic."""
    children = list(tree.successors(node))

    # Gate 1: Binary
    if len(children) != 2:
        return False, f"Gate 1 CLOSED: {len(children)} children"

    left, right = children

    # Gate 2: Local KL
    left_sig = (
        stats_df.loc[left].get("Child_Parent_Divergence_Significant")
        if left in stats_df.index
        else None
    )
    right_sig = (
        stats_df.loc[right].get("Child_Parent_Divergence_Significant")
        if right in stats_df.index
        else None
    )

    if left_sig is None or right_sig is None:
        pass  # Skip gate if missing
    elif not (left_sig or right_sig):
        return False, f"Gate 2 CLOSED: neither diverges"

    # Gate 3: Sibling
    sib_diff = (
        stats_df.loc[node].get("Sibling_BH_Different")
        if node in stats_df.index
        else None
    )
    sib_skipped = (
        stats_df.loc[node].get("Sibling_Divergence_Skipped")
        if node in stats_df.index
        else None
    )

    if sib_diff is None:
        if sib_skipped:
            return False, "Gate 3 CLOSED: skipped"
        return False, "Gate 3 ERROR: missing"

    if not sib_diff:
        p_val = stats_df.loc[node].get("Sibling_Divergence_P_Value", np.nan)
        return False, f"Gate 3 CLOSED: p={p_val:.4f}"

    return True, "SPLIT"


print("=" * 70)
print("FULL DECOMPOSITION TRACE")
print("=" * 70)

root = tree.root()
stack = [(root, 0)]  # (node, depth)
processed = set()
clusters = []

while stack:
    node, depth = stack.pop()
    if node in processed:
        continue
    processed.add(node)

    indent = "  " * depth
    leaf_count = (
        stats_df.loc[node].get("leaf_count", len(get_leaves(node)))
        if node in stats_df.index
        else len(get_leaves(node))
    )

    should, reason = should_split(node)

    print(f"{indent}{node} (n={leaf_count}): {reason}")

    if should:
        children = list(tree.successors(node))
        for child in reversed(children):  # Reverse to process left first
            stack.append((child, depth + 1))
    else:
        leaves = get_leaves(node)
        clusters.append((node, leaves))

print(f"\n{'=' * 70}")
print(f"CLUSTERS FOUND: {len(clusters)}")
print("=" * 70)

for i, (root_node, leaves) in enumerate(clusters):
    leaf_labels = [tree.nodes[l].get("label", l) for l in leaves]
    true_clusters = [cluster_dict.get(lbl, -1) for lbl in leaf_labels]
    print(f"\nCluster {i} (root={root_node}, size={len(leaves)}):")
    print(f"  True cluster breakdown: {dict(pd.Series(true_clusters).value_counts())}")
