"""
Purpose: Analyze branch length correlations with statistical tests.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_correlation_by_linkage__branch_length__analysis.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_blobs
import warnings

warnings.filterwarnings("ignore")


def generate_clustered_data(
    n_clusters=5, n_samples=300, n_features=20, cluster_std=1.0, random_state=42
):
    """Generate clustered data."""
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X, y_true


def generate_binary_data(
    n_clusters=5, n_samples=300, n_features=50, sparsity=0.3, random_state=42
):
    """Generate binary clustered data (like MNIST binarized)."""
    rng = np.random.default_rng(random_state)

    # Create cluster centers
    centers = rng.random((n_clusters, n_features)) > (1 - sparsity)

    # Assign samples to clusters
    labels = rng.integers(0, n_clusters, n_samples)

    # Generate samples with noise around centers
    X = np.zeros((n_samples, n_features), dtype=int)
    for i in range(n_samples):
        center = centers[labels[i]]
        # Flip bits with some probability
        noise = rng.random(n_features) < 0.1
        X[i] = np.logical_xor(center, noise).astype(int)

    return X, labels


def get_leaves(Z, idx, n_leaves):
    """Get all leaves under a node."""
    if idx < n_leaves:
        return {idx}
    merge_idx = idx - n_leaves
    left_idx = int(Z[merge_idx, 0])
    right_idx = int(Z[merge_idx, 1])
    return get_leaves(Z, left_idx, n_leaves) | get_leaves(Z, right_idx, n_leaves)


def compute_branch_metrics_and_truth(Z, y_true, n_leaves):
    """
    Compute branch length metrics and ground truth for both tests.
    """
    n_merges = len(Z)

    # Find parent of each node
    parent_of = {}
    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        node_id = n_leaves + i
        parent_of[left_idx] = node_id
        parent_of[right_idx] = node_id

    # Data for sibling test (per merge)
    sibling_data = []

    # Data for child-parent test (per child edge)
    child_parent_data = []

    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        merge_height = Z[i, 2]
        node_id = n_leaves + i

        # Child heights
        left_height = 0 if left_idx < n_leaves else Z[left_idx - n_leaves, 2]
        right_height = 0 if right_idx < n_leaves else Z[right_idx - n_leaves, 2]

        # Branch lengths
        left_branch = merge_height - left_height
        right_branch = merge_height - right_height
        sum_branch = left_branch + right_branch

        # Branch to parent
        if node_id in parent_of:
            parent_id = parent_of[node_id]
            parent_height = Z[parent_id - n_leaves, 2]
            branch_to_parent = parent_height - merge_height
        else:
            branch_to_parent = 0

        # Ground truth for SIBLING test: do children have disjoint clusters?
        left_leaves = get_leaves(Z, left_idx, n_leaves)
        right_leaves = get_leaves(Z, right_idx, n_leaves)
        left_labels = set(y_true[list(left_leaves)])
        right_labels = set(y_true[list(right_leaves)])
        sibling_boundary = len(left_labels & right_labels) == 0

        sibling_data.append(
            {
                "sum_branch": sum_branch,
                "left_branch": left_branch,
                "right_branch": right_branch,
                "max_branch": max(left_branch, right_branch),
                "geom_mean": np.sqrt(left_branch * right_branch)
                if left_branch > 0 and right_branch > 0
                else 0,
                "height": merge_height,
                "branch_to_parent": branch_to_parent,
                "is_boundary": sibling_boundary,
            }
        )

        # Ground truth for CHILD-PARENT test: does child differ from parent?
        parent_leaves = get_leaves(Z, node_id, n_leaves)
        parent_labels = set(y_true[list(parent_leaves)])

        # Left child
        left_differs = left_labels != parent_labels
        child_parent_data.append(
            {
                "branch": left_branch,
                "differs": left_differs,
            }
        )

        # Right child
        right_differs = right_labels != parent_labels
        child_parent_data.append(
            {
                "branch": right_branch,
                "differs": right_differs,
            }
        )

    return sibling_data, child_parent_data


def evaluate_correlations(sibling_data, child_parent_data):
    """Evaluate branch length correlations for both tests."""

    results = {}

    # SIBLING TEST
    sib_df = pd.DataFrame(sibling_data)
    y_sib = sib_df["is_boundary"].values

    if y_sib.sum() > 0 and y_sib.sum() < len(y_sib):
        for col in [
            "sum_branch",
            "left_branch",
            "right_branch",
            "max_branch",
            "geom_mean",
            "height",
            "branch_to_parent",
        ]:
            try:
                x = sib_df[col].values
                if np.std(x) > 1e-10:
                    auc = roc_auc_score(y_sib, x)
                    corr = np.corrcoef(y_sib, x)[0, 1]
                    results[f"sibling_{col}"] = {"auc": auc, "corr": corr}
            except:
                pass

    # CHILD-PARENT TEST
    cp_df = pd.DataFrame(child_parent_data)
    y_cp = cp_df["differs"].values

    if y_cp.sum() > 0 and y_cp.sum() < len(y_cp):
        try:
            x = cp_df["branch"].values
            if np.std(x) > 1e-10:
                auc = roc_auc_score(y_cp, x)
                corr = np.corrcoef(y_cp, x)[0, 1]
                results["child_parent_branch"] = {"auc": auc, "corr": corr}
        except:
            pass

    return results


def run_analysis():
    """Run analysis across different inference methods."""

    # Configurations to test (matching MNIST benchmark)
    linkage_methods = ["single", "complete", "average", "weighted", "ward"]

    # For continuous data
    continuous_distances = ["euclidean", "cityblock", "cosine"]

    # For binary data
    binary_distances = ["jaccard", "dice", "rogerstanimoto", "hamming"]

    all_results = []

    print("=" * 100)
    print("BRANCH LENGTH CORRELATIONS ACROSS INFERENCE METHODS")
    print("=" * 100)

    # Test with continuous data
    print("\n" + "-" * 100)
    print("CONTINUOUS DATA (Gaussian blobs)")
    print("-" * 100)

    for seed in range(10):
        X, y_true = generate_clustered_data(
            n_clusters=5, n_samples=300, cluster_std=1.5, random_state=seed
        )
        n_leaves = len(X)

        for linkage_method in linkage_methods:
            for distance in continuous_distances:
                # Skip incompatible combinations
                if (
                    linkage_method in ["centroid", "median", "ward"]
                    and distance != "euclidean"
                ):
                    continue

                try:
                    dist_matrix = pdist(X, metric=distance)
                    Z = linkage(dist_matrix, method=linkage_method)

                    sibling_data, child_parent_data = compute_branch_metrics_and_truth(
                        Z, y_true, n_leaves
                    )

                    results = evaluate_correlations(sibling_data, child_parent_data)

                    for metric, vals in results.items():
                        all_results.append(
                            {
                                "data_type": "continuous",
                                "distance": distance,
                                "linkage": linkage_method,
                                "metric": metric,
                                "auc": vals["auc"],
                                "corr": vals["corr"],
                                "seed": seed,
                            }
                        )
                except Exception as e:
                    pass

    # Test with binary data
    print("\n" + "-" * 100)
    print("BINARY DATA (like binarized MNIST)")
    print("-" * 100)

    for seed in range(10):
        X, y_true = generate_binary_data(
            n_clusters=5, n_samples=300, n_features=50, random_state=seed
        )
        n_leaves = len(X)

        for linkage_method in linkage_methods:
            for distance in binary_distances:
                # Ward requires euclidean
                if linkage_method == "ward" and distance != "euclidean":
                    # Use euclidean on binary data for ward
                    dist = "euclidean"
                else:
                    dist = distance

                try:
                    dist_matrix = pdist(X, metric=dist)
                    Z = linkage(dist_matrix, method=linkage_method)

                    sibling_data, child_parent_data = compute_branch_metrics_and_truth(
                        Z, y_true, n_leaves
                    )

                    results = evaluate_correlations(sibling_data, child_parent_data)

                    for metric, vals in results.items():
                        all_results.append(
                            {
                                "data_type": "binary",
                                "distance": dist,
                                "linkage": linkage_method,
                                "metric": metric,
                                "auc": vals["auc"],
                                "corr": vals["corr"],
                                "seed": seed,
                            }
                        )
                except Exception as e:
                    pass

    return pd.DataFrame(all_results)


def print_summary(df):
    """Print summary analysis."""

    print("\n" + "=" * 100)
    print("SUMMARY: BRANCH LENGTHS BY LINKAGE METHOD")
    print("=" * 100)

    # Focus on key metrics
    key_metrics = ["sibling_sum_branch", "child_parent_branch"]

    for metric in key_metrics:
        print(f"\n{'=' * 80}")
        print(f"METRIC: {metric}")
        print(f"{'=' * 80}")

        metric_df = df[df["metric"] == metric]

        if len(metric_df) == 0:
            print("  No data")
            continue

        # Group by linkage method
        summary = (
            metric_df.groupby(["data_type", "linkage", "distance"])
            .agg(
                {
                    "auc": ["mean", "std"],
                    "corr": ["mean", "std"],
                }
            )
            .round(3)
        )

        summary.columns = ["auc_mean", "auc_std", "corr_mean", "corr_std"]
        summary = summary.sort_values("auc_mean", ascending=False)

        print(summary.to_string())

    # Overall by linkage method
    print("\n" + "=" * 100)
    print("OVERALL: MEAN AUC BY LINKAGE METHOD")
    print("=" * 100)

    for metric in key_metrics:
        print(f"\n{metric}:")
        metric_df = df[df["metric"] == metric]

        by_linkage = (
            metric_df.groupby("linkage")
            .agg(
                {
                    "auc": "mean",
                    "corr": "mean",
                }
            )
            .round(3)
            .sort_values("auc", ascending=False)
        )

        for linkage, row in by_linkage.iterrows():
            auc = row["auc"]
            corr = row["corr"]
            # Indicator for positive vs negative correlation
            sign = "✅" if corr > 0 else "❌"
            print(f"  {linkage:<12} AUC={auc:.3f}, corr={corr:+.3f} {sign}")

    # Check for methods where correlation is inverted
    print("\n" + "=" * 100)
    print("ANALYSIS: WHERE DOES BRANCH LENGTH FAIL?")
    print("=" * 100)

    for metric in key_metrics:
        metric_df = df[df["metric"] == metric]

        # Find cases with negative correlation
        neg_cases = metric_df[metric_df["corr"] < 0]

        if len(neg_cases) > 0:
            print(f"\n{metric} - NEGATIVE CORRELATION CASES:")
            neg_summary = (
                neg_cases.groupby(["data_type", "linkage", "distance"])
                .agg(
                    {
                        "auc": "mean",
                        "corr": "mean",
                    }
                )
                .round(3)
            )
            print(neg_summary.to_string())
        else:
            print(f"\n{metric}: All correlations positive ✅")


if __name__ == "__main__":
    df = run_analysis()
    print_summary(df)

    # Save detailed results
    df.to_csv(
        "/Users/berksakalli/Projects/kl-te-cluster/results/branch_length_by_linkage.csv",
        index=False,
    )
    print("\n\nDetailed results saved to results/branch_length_by_linkage.csv")
