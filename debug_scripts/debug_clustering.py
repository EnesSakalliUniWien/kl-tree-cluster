"""Debug why the clustering algorithm fails on well-separated data."""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, adjusted_rand_score

from kl_clustering_analysis.benchmarking.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

print("=" * 70)
print("DEBUG: Why does clustering fail on well-separated data?")
print("=" * 70)

# Generate data with good separation
data_dict, cluster_dict = generate_random_feature_matrix(
    n_rows=50,
    n_cols=30,
    n_clusters=3,
    entropy_param=0.05,  # Low noise = good separation
    balanced_clusters=True,
    random_seed=42,
)

data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
true_labels = np.array([cluster_dict[name] for name in data_df.index])

print(f"\nData shape: {data_df.shape}")
print(f"True clusters: {len(np.unique(true_labels))}")
print(
    f"Silhouette score: {silhouette_score(data_df.values, true_labels, metric='hamming'):.3f}"
)

# Build tree
print("\n" + "-" * 50)
print("BUILDING TREE")
print("-" * 50)

Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
print(f"Tree nodes: {tree.number_of_nodes()}")

# Run decomposition
print("\n" + "-" * 50)
print("RUNNING DECOMPOSITION")
print("-" * 50)

print(f"Config SIBLING_ALPHA: {config.SIBLING_ALPHA}")
print(f"Config ALPHA_LOCAL: {config.ALPHA_LOCAL}")
print(f"Config USE_RANDOM_PROJECTION: {config.USE_RANDOM_PROJECTION}")

decomposition = tree.decompose(leaf_data=data_df)
cluster_report = decomposition.get("cluster_report", {})

print(f"\nClusters found: {len(cluster_report)}")

# Check stats_df
print("\n" + "-" * 50)
print("STATISTICAL ANNOTATIONS")
print("-" * 50)

stats_df = tree.stats_df
root = tree.root()

print(f"Root: {root}")
if root in stats_df.index:
    row = stats_df.loc[root]
    print(f"\nRoot node statistics:")
    print(f"  Sibling_BH_Different: {row.get('Sibling_BH_Different')}")
    print(f"  Sibling_Divergence_P_Value: {row.get('Sibling_Divergence_P_Value'):.6f}")
    print(f"  Sibling_Test_Statistic: {row.get('Sibling_Test_Statistic'):.2f}")
    print(f"  Sibling_Degrees_of_Freedom: {row.get('Sibling_Degrees_of_Freedom'):.1f}")
    print(f"  Sibling_JSD: {row.get('Sibling_JSD')}")

# Check root's children
children = list(tree.successors(root))
print(f"\nRoot children: {children}")

for child in children:
    if child in stats_df.index:
        row = stats_df.loc[child]
        n_leaves = row.get("leaf_count", "?")
        print(f"\n  {child} (n_leaves={n_leaves}):")
        print(f"    Child_Parent_Divergence_Significant: {row.get('Child_Parent_Divergence_Significant')}")
        print(f"    Child_Parent_Divergence_P_Value_BH: {row.get('Child_Parent_Divergence_P_Value_BH')}")
        print(f"    Sibling_BH_Different: {row.get('Sibling_BH_Different')}")
        print(f"    Sibling_P_Value: {row.get('Sibling_Divergence_P_Value')}")

# What would it take to be significant?
print("\n" + "-" * 50)
print("SIGNIFICANCE ANALYSIS")
print("-" * 50)

from scipy.stats import chi2

if root in stats_df.index:
    row = stats_df.loc[root]
    test_stat = row.get("Sibling_Test_Statistic")
    df = row.get("Sibling_Degrees_of_Freedom")

    # What critical value is needed?
    for alpha in [0.05, 0.01, 0.001]:
        critical = chi2.ppf(1 - alpha, df=df)
        print(
            f"  α={alpha}: critical value = {critical:.2f}, test_stat={test_stat:.2f}, significant={test_stat > critical}"
        )

# Compute ARI
print("\n" + "-" * 50)
print("FINAL RESULT")
print("-" * 50)

label_map = {}
for cl_id, info in cluster_report.items():
    for leaf in info["members"]:
        label_map[leaf] = cl_id

pred_labels = [label_map.get(name, 0) for name in data_df.index]
pred_labels = np.array(pred_labels)

ari = adjusted_rand_score(true_labels, pred_labels)
print(f"True clusters: {len(np.unique(true_labels))}")
print(f"Found clusters: {len(cluster_report) if cluster_report else 1}")
print(f"ARI: {ari:.4f}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print("""
The sibling divergence test p-value is > alpha, so it fails to reject H0.
This means the test says "siblings are NOT significantly different" 
even though the data has clear cluster structure.

The test is TOO CONSERVATIVE.
""")
