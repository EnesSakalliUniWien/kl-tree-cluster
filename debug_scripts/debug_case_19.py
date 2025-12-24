"""Debug script for test case 19: binary_balanced_low_noise with d=1200, n=72.

This case is severely under-splitting (k=1 instead of k=4).
Let's investigate why the statistical tests are failing to detect cluster structure.
"""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

# Reproduce the problematic case
test_config = {
    "name": "binary_balanced_low_noise",
    "generator": "binary",
    "n_rows": 72,
    "n_cols": 1200,
    "n_clusters": 4,
    "entropy_param": 0.25,  # 25% noise
    "balanced_clusters": True,
    "seed": 314,
}

print("=" * 80)
print("DEBUG: Test Case 19 - binary_balanced_low_noise (72x1200, k=4)")
print("=" * 80)

# Generate the data
data_dict, cluster_dict = generate_random_feature_matrix(
    n_rows=test_config["n_rows"],
    n_cols=test_config["n_cols"],
    n_clusters=test_config["n_clusters"],
    entropy_param=test_config["entropy_param"],
    balanced_clusters=test_config["balanced_clusters"],
    random_seed=test_config["seed"],
)

# Convert to dataframe
sample_names = list(data_dict.keys())
data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
true_labels = np.array([cluster_dict[name] for name in sample_names])

print(f"\nData shape: {data_df.shape}")
print(f"True clusters: {len(np.unique(true_labels))}")
print(f"Cluster sizes: {np.bincount(true_labels)}")

# Check data properties
print("\n" + "-" * 40)
print("DATA STATISTICS")
print("-" * 40)
feature_means = data_df.mean(axis=0)
print(
    f"Feature means: min={feature_means.min():.3f}, max={feature_means.max():.3f}, "
    f"median={feature_means.median():.3f}"
)

# Check per-cluster feature means
print("\nPer-cluster mean feature values:")
for c in range(4):
    cluster_mask = true_labels == c
    cluster_names = [sample_names[i] for i, m in enumerate(cluster_mask) if m]
    cluster_means = data_df.loc[cluster_names].mean(axis=0)
    print(
        f"  Cluster {c}: mean={cluster_means.mean():.3f}, std={cluster_means.std():.3f}"
    )

# Build tree
print("\n" + "-" * 40)
print("BUILDING TREE")
print("-" * 40)
Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
print(f"Tree nodes: {tree.number_of_nodes()}")
print(f"Tree edges: {tree.number_of_edges()}")

# Check current config
print("\n" + "-" * 40)
print("CONFIG")
print("-" * 40)
print(f"USE_RANDOM_PROJECTION: {config.USE_RANDOM_PROJECTION}")
print(
    "PROJECTION_DECISION: JL-based (project if compute_projection_dimension(n_eff, n_features) < n_features and n_features > n_eff)"
)
print(f"SIBLING_ALPHA: {config.SIBLING_ALPHA}")
print(f"ALPHA_LOCAL: {config.ALPHA_LOCAL}")

# Run decomposition (this runs annotations internally)
print("\n" + "-" * 40)
print("DECOMPOSITION")
print("-" * 40)

decomposition = tree.decompose(leaf_data=data_df)
cluster_report = decomposition.get("cluster_report", {})

n_clusters_found = len(cluster_report)
print(f"Clusters found: {n_clusters_found} (expected: 4)")

# Look at the stats_df for diagnostics
print("\n" + "-" * 40)
print("STATISTICAL ANNOTATIONS (first few levels)")
print("-" * 40)

stats_df = tree.stats_df
if stats_df is not None:
    print(f"Stats DF shape: {stats_df.shape}")
    print(f"Stats DF columns: {list(stats_df.columns)}")
    print(f"Stats DF index: {stats_df.index.name}")

    # Find root
    root = tree.root()
    print(f"\nRoot: {root}")

    # Look at root level statistics
    if root in stats_df.index:
        row = stats_df.loc[root]
        print(f"\nRoot node statistics:")
        print(f"  Sibling_BH_Different: {row.get('Sibling_BH_Different', 'N/A')}")
        print(
            f"  Sibling_Divergence_P_Value: {row.get('Sibling_Divergence_P_Value', 'N/A')}"
        )
        print(f"  Sibling_Test_Statistic: {row.get('Sibling_Test_Statistic', 'N/A')}")
        print(
            f"  Sibling_Degrees_of_Freedom: {row.get('Sibling_Degrees_of_Freedom', 'N/A')}"
        )

        # Look at children statistics
        children = list(tree.successors(root))
        print(f"\nRoot children: {children}")
        for child in children:
            if child in stats_df.index:
                row = stats_df.loc[child]
                n_leaves = row.get("n_leaves", row.get("leaf_count", "?"))
                print(f"\n  {child} (n_leaves={n_leaves}):")
                print(f"    KL_BH_Significant: {row.get('KL_BH_Significant', 'N/A')}")
                print(f"    KL_P_Value: {row.get('KL_P_Value', 'N/A')}")
                print(
                    f"    Sibling_BH_Different: {row.get('Sibling_BH_Different', 'N/A')}"
                )
                print(
                    f"    Sibling_P_Value: {row.get('Sibling_Divergence_P_Value', 'N/A')}"
                )
else:
    print("No stats_df available!")

# What's the issue?
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
if n_clusters_found == 1:
    print("The algorithm is NOT splitting at all!")
    print("\nPossible causes:")
    print("  1. Local KL gate: No child diverges from parent")
    print("  2. Sibling gate: Siblings not detected as different")
    print("  3. Chi-square test too conservative in high dimensions (d=1200, n=72)")
    print("\nThe key ratio is d/n = 1200/72 = 16.7")
    print(
        "Random projection activates when d > 2*n (i.e., d > 144), so projection IS being used"
    )
    print("Target projection dimension k = 8*log(72) = 34")
elif n_clusters_found < 4:
    print(f"The algorithm is under-splitting: found {n_clusters_found}, expected 4")
else:
    print(f"Decomposition working correctly: found {n_clusters_found} clusters")
