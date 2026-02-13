"""Check if the tree structure itself is wrong for noisy data.

Compare ground truth cluster assignments to tree structure.
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)


def main():
    np.random.seed(42)

    print("Testing if hierarchical clustering itself is the problem")
    print("=" * 70)
    print("\nCutting dendrogram at true K clusters vs ground truth:")
    print("-" * 70)
    print("Entropy | K | ARI (cut at K) | Description")
    print("-" * 70)

    for entropy in [0.1, 0.2, 0.3]:
        for true_k in [2, 3, 4]:
            data_dict, labels = generate_random_feature_matrix(
                n_rows=200,
                n_cols=50,
                n_clusters=true_k,
                entropy_param=entropy,
                random_seed=42,
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index")

            # Build linkage
            Z = linkage(pdist(df.values, metric="hamming"), method="average")

            # Cut at exactly true_k clusters
            pred_labels = fcluster(Z, t=true_k, criterion="maxclust")

            # Map to same order as df
            true_labels = [labels[name] for name in df.index]

            ari = adjusted_rand_score(true_labels, pred_labels)

            if ari > 0.9:
                desc = "✓ Tree structure good"
            elif ari > 0.5:
                desc = "~ Tree partially correct"
            else:
                desc = "✗ Tree structure broken"

            print(f"{entropy:.1f}     | {true_k} |     {ari:.3f}      | {desc}")


if __name__ == "__main__":
    main()
