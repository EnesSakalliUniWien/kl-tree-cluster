"""Deep debug of decomposition to find why 0 clusters are returned."""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.benchmarking.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
from kl_clustering_analysis import config

print("=" * 70)
print("DEEP DEBUG: Tracing decomposition step by step")
print("=" * 70)

# Generate data with good separation
data_dict, cluster_dict = generate_random_feature_matrix(
    n_rows=50,
    n_cols=30,
    n_clusters=3,
    entropy_param=0.05,
    balanced_clusters=True,
    random_seed=42,
)

data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
true_labels = np.array([cluster_dict[name] for name in data_df.index])

# Build and decompose tree
Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
decomposition = tree.decompose(leaf_data=data_df)

# Access the TreeDecomposition that was used
stats_df = tree.stats_df

print(f"\nData: {data_df.shape}, True clusters: 3")
print(f"\nConfig:")
print(f"  SIBLING_ALPHA: {config.SIBLING_ALPHA}")
print(f"  ALPHA_LOCAL: {config.ALPHA_LOCAL}")

# Manually trace through decomposition
print("\n" + "=" * 70)
print("MANUAL DECOMPOSITION TRACE")
print("=" * 70)

root = tree.root()
print(f"\nStarting at root: {root}")

# Check root
children = list(tree.successors(root))
print(f"Root children: {children}")
print(f"Number of children: {len(children)}")

if len(children) != 2:
    print("  -> Gate 1 CLOSED: Not exactly 2 children")
else:
    left, right = children

    # Check Local KL gate
    local_sig_left = (
        stats_df.loc[left].get("Child_Parent_Divergence_Significant")
        if left in stats_df.index
        else None
    )
    local_sig_right = (
        stats_df.loc[right].get("Child_Parent_Divergence_Significant")
        if right in stats_df.index
        else None
    )

    print(f"\nGate 2 (Local KL):")
    print(f"  Left  ({left}): Child_Parent_Divergence_Significant = {local_sig_left}")
    print(f"  Right ({right}): Child_Parent_Divergence_Significant = {local_sig_right}")

    if local_sig_left or local_sig_right:
        print("  -> Gate 2 OPEN")
    else:
        print("  -> Gate 2 CLOSED: Neither child diverges from parent")

    # Check Sibling Divergence gate
    sib_different = (
        stats_df.loc[root].get("Sibling_BH_Different")
        if root in stats_df.index
        else None
    )
    sib_pval = (
        stats_df.loc[root].get("Sibling_Divergence_P_Value")
        if root in stats_df.index
        else None
    )

    print(f"\nGate 3 (Sibling Divergence):")
    print(f"  Parent ({root}): Sibling_BH_Different = {sib_different}")
    print(f"  Parent ({root}): Sibling_P_Value = {sib_pval}")

    if sib_different:
        print("  -> Gate 3 OPEN")
    else:
        print("  -> Gate 3 CLOSED: Siblings not significantly different")

# Now check what decompose actually did
print("\n" + "=" * 70)
print("DECOMPOSITION RESULT")
print("=" * 70)

cluster_report = decomposition.get("cluster_report", {})
print(f"\nNumber of clusters found: {len(cluster_report)}")

# Let me print all siblings tests from stats_df
print("\n" + "=" * 70)
print("ALL SIBLING TESTS")
print("=" * 70)

sibling_cols = [c for c in stats_df.columns if "Sibling" in c]
print(f"\nSibling-related columns: {sibling_cols}")

# Print info for all internal nodes
internal_nodes = [n for n in tree.nodes() if list(tree.successors(n))]
print(f"\nInternal nodes: {len(internal_nodes)}")

for node in sorted(internal_nodes)[:10]:  # First 10
    row = stats_df.loc[node]
    sib_diff = row.get("Sibling_BH_Different")
    sib_pval = row.get("Sibling_Divergence_P_Value")
    leaf_count = row.get("leaf_count", "?")
    print(
        f"  {node:6s} (n={leaf_count:>2}): Sibling_BH_Different={sib_diff}, p={sib_pval:.4f}"
        if sib_pval
        else f"  {node:6s}: no data"
    )

# Let's trace what tree.decompose() actually does
print("\n" + "=" * 70)
print("CHECKING POSET TREE DECOMPOSE METHOD")
print("=" * 70)

# Read the PosetTree class to see what decompose does
import inspect

print(f"\nPosetTree.decompose signature:")
print(inspect.signature(tree.decompose))
