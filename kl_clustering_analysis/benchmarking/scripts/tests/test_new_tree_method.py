"""Quick test to verify the new tree inference settings work."""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

print("=" * 70)
print("TESTING NEW TREE INFERENCE CONFIGURATION")
print("=" * 70)
print(f"\nConfig settings:")
print(f"  TREE_DISTANCE_METRIC: {config.TREE_DISTANCE_METRIC}")
print(f"  TREE_LINKAGE_METHOD: {config.TREE_LINKAGE_METHOD}")

# Run comparison
results = {"old": [], "new": []}

for seed in range(5):
    for noise in [0.05, 0.10, 0.15]:
        # Generate data
        data_dict, cluster_dict = generate_random_feature_matrix(
            n_rows=100,
            n_cols=50,
            n_clusters=3,
            entropy_param=noise,
            balanced_clusters=True,
            random_seed=42 + seed,
        )
        data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)
        true_labels = np.array([cluster_dict[name] for name in data_df.index])

        # OLD method: hamming + complete
        Z_old = linkage(pdist(data_df.values, metric="hamming"), method="complete")
        tree_old = PosetTree.from_linkage(Z_old, leaf_names=data_df.index.tolist())
        decomp_old = tree_old.decompose(leaf_data=data_df)

        clusters_old = decomp_old.get("cluster_assignments", {})
        label_map_old = {}
        for cl_id, info in clusters_old.items():
            for leaf in info["leaves"]:
                label_map_old[leaf] = cl_id
        pred_old = np.array([label_map_old.get(n, 0) for n in data_df.index])
        ari_old = adjusted_rand_score(true_labels, pred_old)

        # NEW method: rogerstanimoto + average
        Z_new = linkage(
            pdist(data_df.values, metric="rogerstanimoto"), method="average"
        )
        tree_new = PosetTree.from_linkage(Z_new, leaf_names=data_df.index.tolist())
        decomp_new = tree_new.decompose(leaf_data=data_df)

        clusters_new = decomp_new.get("cluster_assignments", {})
        label_map_new = {}
        for cl_id, info in clusters_new.items():
            for leaf in info["leaves"]:
                label_map_new[leaf] = cl_id
        pred_new = np.array([label_map_new.get(n, 0) for n in data_df.index])
        ari_new = adjusted_rand_score(true_labels, pred_new)

        results["old"].append(ari_old)
        results["new"].append(ari_new)

print(f"\n{'=' * 50}")
print("COMPARISON RESULTS")
print("=" * 50)

old_mean = np.mean(results["old"])
new_mean = np.mean(results["new"])
improvement = (new_mean - old_mean) / old_mean * 100

print(f"\nOLD (hamming + complete):     Mean ARI = {old_mean:.3f}")
print(f"NEW (rogerstanimoto + average): Mean ARI = {new_mean:.3f}")
print(f"\nImprovement: {improvement:+.1f}%")

if new_mean > old_mean:
    print("\n✅ New method is BETTER!")
else:
    print("\n⚠️ New method is not better in this test")
