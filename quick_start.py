import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# Import the necessary functions from your library
from tree.poset_tree import PosetTree


def main():
    """
    A small, self-contained example of the full analysis pipeline.
    """
    print("--- Starting Analysis Pipeline ---")

    # 1. --- Data Generation ---
    X, y_true = make_blobs(
        n_samples=80,
        n_features=30,
        centers=4,
        cluster_std=1.2,
        random_state=42,
    )
    X_binary = (X > np.median(X, axis=0)).astype(int)
    data = pd.DataFrame(
        X_binary,
        index=[f"Sample_{j}" for j in range(X.shape[0])],
    )
    print(
        f"\nStep 1: Generated a binary dataset with {data.shape[0]} samples and {data.shape[1]} features."
    )
    print(f"Ground truth contains {len(np.unique(y_true))} clusters.")

    # --- Execute the Core Pipeline ---
    # 2. linkage()
    Z = linkage(pdist(data.values, metric="hamming"), method="complete")
    print("\nStep 2: Created hierarchy with SciPy linkage.")

    # 3. PosetTree.from_linkage()
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    print("Step 3: Converted hierarchy to PosetTree structure.")

    # 4. PosetTree.decompose()
    # The decompose method now handles divergence calculation and annotation internally
    significance_level = 0.05
    decomposition_results = tree.decompose(
        leaf_data=data,
        alpha_local=significance_level,
        sibling_alpha=significance_level,
    )
    print(
        "Step 4: Decomposed the tree to extract significant clusters (metrics computed internally)."
    )

    # --- Display Results ---
    print("\n--- Analysis Complete ---")
    num_found = decomposition_results.get("num_clusters", 0)
    print(f"\nAlgorithm found {num_found} clusters.")

    cluster_assignments = decomposition_results.get("cluster_assignments", {})

    predicted_labels = {}
    if cluster_assignments:
        for cluster_id, info in cluster_assignments.items():
            print(
                f"  - Cluster {cluster_id} (root: {info['root_node']}): {info['size']} samples"
            )
            for leaf in info["leaves"]:
                predicted_labels[leaf] = cluster_id

    # --- Validation Check ---
    ordered_true_labels = []
    ordered_pred_labels = []
    for sample_name in data.index:
        sample_index = int(sample_name.split("_")[1])
        ordered_true_labels.append(y_true[sample_index])
        ordered_pred_labels.append(predicted_labels.get(sample_name, -1))

    if ordered_pred_labels:
        ari_score = adjusted_rand_score(ordered_true_labels, ordered_pred_labels)
        print(f"\nValidation: Adjusted Rand Index (ARI) = {ari_score:.4f}")
        print("(1.0 is a perfect match, 0.0 is random assignment)")


if __name__ == "__main__":
    main()
