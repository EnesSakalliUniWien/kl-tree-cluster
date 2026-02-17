#!/usr/bin/env python3
"""Trace KL decomposition on gaussian_clear_1 to understand ARI≈0 despite K=3."""
import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from kl_clustering_analysis.tree.poset_tree import PosetTree

# Reproduce gaussian_clear_1
X, y_true = make_blobs(n_samples=30, n_features=30, centers=3, cluster_std=0.5, random_state=100)
X_bin = (X > np.median(X, axis=0)).astype(int)
data_df = pd.DataFrame(
    X_bin, index=[f"S{j}" for j in range(30)], columns=[f"F{j}" for j in range(30)]
)

dist = pdist(data_df.values, metric="hamming")
Z = linkage(dist, method="average")

# Reference: what does a simple 3-cut give?
y_cut3 = fcluster(Z, t=3, criterion="maxclust")
print(f"Simple 3-cut ARI: {adjusted_rand_score(y_true, y_cut3):.3f}")

# KL decomposition
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
decomp = tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)

assignments = decomp.get("cluster_assignments", {})
K = len(assignments)
print(f"\nKL found K={K}")

# Show cluster assignments
for cid, info in sorted(assignments.items()):
    leaves = sorted(info["leaves"], key=lambda x: int(x[1:]))
    true_labels = [y_true[int(l[1:])] for l in leaves]
    label_counts = {}
    for tl in true_labels:
        label_counts[tl] = label_counts.get(tl, 0) + 1
    print(
        f"  Cluster {cid} (n={info['size']}): true labels={label_counts}  leaves={leaves[:10]}{'...' if len(leaves) > 10 else ''}"
    )

# Compute ARI
pred = {s: -1 for s in data_df.index}
for cid, info in assignments.items():
    for leaf in info["leaves"]:
        pred[leaf] = cid
y_pred_str = [pred[f"S{i}"] for i in range(30)]
name_to_int = {n: i for i, n in enumerate(sorted(set(y_pred_str)))}
y_pred = [name_to_int[n] for n in y_pred_str]

ari = adjusted_rand_score(y_true, y_pred)
print(f"\nKL ARI: {ari:.4f}")

# Show the cluster roots in the tree
cluster_roots = decomp.get("cluster_roots", set())
print(f"\nCluster roots: {sorted(cluster_roots)}")

# Check tree.stats_df for gate decisions at cluster root parents
stats = tree.stats_df
if stats is not None:
    print("\nGate decisions at internal nodes near the root:")
    cols = [
        c for c in stats.columns if any(k in c for k in ["Sibling", "Child_Parent_Divergence_Sig"])
    ]
    # Show only non-leaf nodes
    internal = stats[stats.index.str.startswith("N")]
    # Show the last few (near root)
    root_area = internal.tail(10)
    for node_id in root_area.index:
        edge_sig = root_area.loc[node_id].get("Child_Parent_Divergence_Significant", "?")
        sib_diff = root_area.loc[node_id].get("Sibling_BH_Different", "?")
        sib_same = root_area.loc[node_id].get("Sibling_BH_Same", "?")
        sib_skip = root_area.loc[node_id].get("Sibling_Divergence_Skipped", "?")
        print(
            f"  {node_id}: edge_sig={edge_sig}  sib_diff={sib_diff}  sib_same={sib_same}  sib_skip={sib_skip}"
        )
        print(
            f"  {node_id}: edge_sig={edge_sig}  sib_diff={sib_diff}  sib_same={sib_same}  sib_skip={sib_skip}"
        )
