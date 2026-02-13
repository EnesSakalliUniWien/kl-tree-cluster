"""Trace through the entire decomposition tree."""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.generators import generate_random_feature_matrix
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


def should_split(node):
    """Manual implementation of _should_split logic."""
    children = list(tree.successors(node))

    # Gate 1: Binary
    if len(children) != 2:
        return False, "Gate 1 CLOSED: not 2 children"

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
        return (
            False,
            f"Gate 2 CLOSED: neither child diverges (left={left_sig}, right={right_sig})",
        )

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
            return False, "Gate 3 CLOSED: sibling test skipped"
        return False, "Gate 3 ERROR: missing sibling annotations"

    if not sib_diff:
        p_val = stats_df.loc[node].get("Sibling_Divergence_P_Value", np.nan)
        return False, f"Gate 3 CLOSED: siblings not different (p={p_val:.4f})"

    return True, "ALL GATES OPEN"


print("=" * 70)
print("COMPLETE DECOMPOSITION TRACE")
print("=" * 70)

root = tree.root()
stack = [root]
processed = set()
clusters = []
level = 0
max_levels = 10

while stack and level < max_levels:
    level += 1
    print(f"\n{'=' * 50}")
    print(f"LEVEL {level}: Stack = {stack}")
    print(f"{'=' * 50}")

    node = stack.pop()
    if node in processed:
        continue
    processed.add(node)

    leaf_count = (
        stats_df.loc[node].get("leaf_count", "?") if node in stats_df.index else "?"
    )
    print(f"\nProcessing: {node} (leaves={leaf_count})")

    children = list(tree.successors(node))
    print(f"  Children: {children}")

    should, reason = should_split(node)
    print(f"  Should split? {should} - {reason}")

    if should:
        # Add children to stack
        left, right = children
        stack.append(right)
        stack.append(left)
        print(f"  -> SPLIT: Adding children to stack")
    else:
        # Collect leaves as cluster
        leaves = [
            n
            for n in tree.nodes()
            if tree.nodes[n].get("is_leaf") and tree.has_path(node, n)
        ]
        clusters.append(set(leaves))
        print(f"  -> MERGE: Collecting {len(leaves)} leaves as cluster")

print(f"\n\n{'=' * 70}")
print(f"RESULT: {len(clusters)} clusters found")
print(f"{'=' * 70}")

for i, cl in enumerate(clusters):
    print(f"  Cluster {i}: {len(cl)} members")
