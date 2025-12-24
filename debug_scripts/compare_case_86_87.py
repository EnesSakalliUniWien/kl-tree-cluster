"""
Compare Case 86 vs Case 87 - different sample/feature ratios.

Case 86: 300 samples, 400 features (more features than samples)
Case 87: 500 samples, 50 features (more samples than features)
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config


def run_case(case_config, case_name):
    print("=" * 70)
    print(f"{case_name}")
    print("=" * 70)
    print(
        f"Config: n_rows={case_config['n_rows']}, n_cols={case_config['n_cols']}, n_clusters={case_config['n_clusters']}"
    )
    print(f"  entropy_param={case_config['entropy_param']}, seed={case_config['seed']}")
    print(
        f"  Ratio n_rows/n_cols = {case_config['n_rows'] / case_config['n_cols']:.2f}"
    )

    np.random.seed(case_config["seed"])
    leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
        n_rows=case_config["n_rows"],
        n_cols=case_config["n_cols"],
        n_clusters=case_config["n_clusters"],
        entropy_param=case_config["entropy_param"],
        balanced_clusters=case_config.get("balanced_clusters", True),
        random_seed=case_config["seed"],
    )

    data_df = pd.DataFrame.from_dict(leaf_matrix_dict, orient="index")
    distance_condensed = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(distance_condensed, method=config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

    decomp = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=config.SIBLING_ALPHA,
    )

    n_found = decomp.get("num_clusters", 0)
    cluster_assignments_found = decomp.get("cluster_assignments", {})

    # Convert to sample->cluster mapping
    sample_to_cluster = {}
    for cluster_id, info in cluster_assignments_found.items():
        for leaf in info["leaves"]:
            sample_to_cluster[leaf] = cluster_id

    common_keys = sorted(
        set(cluster_assignments.keys()) & set(sample_to_cluster.keys())
    )
    ari = adjusted_rand_score(
        [cluster_assignments[k] for k in common_keys],
        [sample_to_cluster[k] for k in common_keys],
    )
    nmi = normalized_mutual_info_score(
        [cluster_assignments[k] for k in common_keys],
        [sample_to_cluster[k] for k in common_keys],
    )

    print(f"\nResults: Found {n_found} clusters (expected {case_config['n_clusters']})")
    print(f"  ARI: {ari:.4f}, NMI: {nmi:.4f}")

    # Show cluster sizes
    sizes = [info["size"] for info in cluster_assignments_found.values()]
    print(f"  Cluster sizes: {sorted(sizes, reverse=True)}")
    print()


if __name__ == "__main__":
    # Case 86: n_rows=300, n_cols=400 (more features than samples)
    case_86 = {
        "name": "binary_many_clusters",
        "generator": "binary",
        "n_rows": 300,
        "n_cols": 400,
        "n_clusters": 15,
        "entropy_param": 0.1,
        "balanced_clusters": True,
        "seed": 7003,
    }

    # Case 87: n_rows=500, n_cols=50 (more samples than features)
    case_87 = {
        "name": "overlap_heavy_4c_small_feat",
        "generator": "binary",
        "n_rows": 500,
        "n_cols": 50,
        "n_clusters": 4,
        "entropy_param": 0.4,
        "balanced_clusters": True,
        "seed": 8000,
    }

    run_case(case_86, "CASE 86: binary_many_clusters (300 samples, 400 features)")
    run_case(case_87, "CASE 87: overlap_heavy_4c_small_feat (500 samples, 50 features)")
