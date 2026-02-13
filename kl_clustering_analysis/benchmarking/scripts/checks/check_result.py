"""Check what decompose_tree returns."""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.tree_decomposition import TreeDecomposition
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

# Get decomposition
decomposition = tree.decompose(leaf_data=data_df)

print("=" * 70)
print("DECOMPOSITION RESULT STRUCTURE")
print("=" * 70)

print(f"\nTop-level keys: {list(decomposition.keys())}")

for key, value in decomposition.items():
    if isinstance(value, dict):
        print(f"\n{key}: dict with {len(value)} items")
        if len(value) < 10:
            for k, v in value.items():
                if isinstance(v, dict):
                    print(f"  {k}: {list(v.keys())}")
                else:
                    print(f"  {k}: {type(v).__name__}")
    else:
        print(f"\n{key}: {value}")

# Check cluster_assignments specifically
cluster_assignments = decomposition.get("cluster_assignments", {})
print(f"\n\ncluster_assignments has {len(cluster_assignments)} clusters")

for cl_id, info in cluster_assignments.items():
    print(f"\nCluster {cl_id}:")
    print(f"  root_node: {info.get('root_node')}")
    print(f"  size: {info.get('size')}")
    print(
        f"  leaves: {info.get('leaves')[:5]}..."
        if len(info.get("leaves", [])) > 5
        else f"  leaves: {info.get('leaves')}"
    )

# Now check what cluster_report is
print("\n\n" + "=" * 70)
print("CLUSTER_REPORT KEY")
print("=" * 70)

if "cluster_report" in decomposition:
    print("cluster_report exists!")
    print(decomposition["cluster_report"])
else:
    print("NO cluster_report key in decomposition!")
    print("Available keys:", list(decomposition.keys()))
