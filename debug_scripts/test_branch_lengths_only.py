"""
Test ONLY branch length metrics (actual distances, not heights).

Branch length = distance from parent to child = parent_height - child_height

We want to understand:
1. How do branch lengths relate to cluster boundaries?
2. Can branch lengths predict true splits?
3. What transformations make them useful?
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.datasets import make_blobs
import warnings

warnings.filterwarnings("ignore")


def generate_data(n_clusters=5, n_samples=200, cluster_std=1.0, random_state=42):
    """Generate clustered data."""
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=10,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X, y_true


def compute_branch_lengths(Z, n_leaves):
    """
    Compute branch lengths for each internal node.

    For each internal node (merge), compute:
    - left_branch: distance from this node to left child
    - right_branch: distance from this node to right child
    - parent_branch: distance from this node to its parent
    - sum_branch: left + right
    - max_branch: max(left, right)
    - min_branch: min(left, right)
    - ratio_branch: min/max (asymmetry)
    """
    n_merges = len(Z)

    # First pass: find parent of each node
    parent_of = {}
    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        node_id = n_leaves + i
        parent_of[left_idx] = node_id
        parent_of[right_idx] = node_id

    branch_data = []

    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        merge_height = Z[i, 2]
        merge_count = int(Z[i, 3])
        node_id = n_leaves + i

        # Get heights of children
        if left_idx < n_leaves:
            left_height = 0.0  # Leaf
            left_count = 1
        else:
            left_merge_idx = left_idx - n_leaves
            left_height = Z[left_merge_idx, 2]
            left_count = int(Z[left_merge_idx, 3])

        if right_idx < n_leaves:
            right_height = 0.0  # Leaf
            right_count = 1
        else:
            right_merge_idx = right_idx - n_leaves
            right_height = Z[right_merge_idx, 2]
            right_count = int(Z[right_merge_idx, 3])

        # Branch lengths = distance from parent to child
        left_branch = merge_height - left_height
        right_branch = merge_height - right_height

        # Distance to parent (0 if root)
        if node_id in parent_of:
            parent_id = parent_of[node_id]
            parent_height = Z[parent_id - n_leaves, 2]
            parent_branch = parent_height - merge_height
        else:
            parent_branch = 0.0  # Root has no parent

        # Various branch length metrics
        sum_branch = left_branch + right_branch
        max_branch = max(left_branch, right_branch)
        min_branch = min(left_branch, right_branch)
        ratio_branch = min_branch / max_branch if max_branch > 0 else 1.0
        diff_branch = abs(left_branch - right_branch)

        # Weighted by cluster sizes
        weighted_sum = left_branch * left_count + right_branch * right_count
        weighted_avg = weighted_sum / merge_count

        # Product of branches (geometric mean related)
        prod_branch = left_branch * right_branch
        geom_mean = np.sqrt(prod_branch) if prod_branch > 0 else 0

        # Harmonic mean
        if left_branch > 0 and right_branch > 0:
            harmonic_mean = (
                2 * left_branch * right_branch / (left_branch + right_branch)
            )
        else:
            harmonic_mean = 0

        # Combined child-parent metrics
        total_branch = sum_branch + parent_branch  # All three edges
        child_parent_ratio = (
            sum_branch / parent_branch if parent_branch > 0 else float("inf")
        )
        child_parent_diff = sum_branch - parent_branch

        # Ratio of going down vs going up
        down_up_ratio = (
            sum_branch / (sum_branch + parent_branch)
            if (sum_branch + parent_branch) > 0
            else 0.5
        )

        branch_data.append(
            {
                "merge_idx": i,
                "height": merge_height,
                "left_branch": left_branch,
                "right_branch": right_branch,
                "parent_branch": parent_branch,
                "sum_branch": sum_branch,
                "max_branch": max_branch,
                "min_branch": min_branch,
                "ratio_branch": ratio_branch,
                "diff_branch": diff_branch,
                "weighted_sum": weighted_sum,
                "weighted_avg": weighted_avg,
                "prod_branch": prod_branch,
                "geom_mean": geom_mean,
                "harmonic_mean": harmonic_mean,
                "total_branch": total_branch,
                "child_parent_ratio": child_parent_ratio,
                "child_parent_diff": child_parent_diff,
                "down_up_ratio": down_up_ratio,
                "left_count": left_count,
                "right_count": right_count,
                "merge_count": merge_count,
            }
        )

    return branch_data


def compute_path_lengths(Z, n_leaves):
    """
    Compute path lengths from each internal node down to leaves.

    This is the total "distance traveled" to reach all descendant leaves.
    """
    n_merges = len(Z)

    # For each node, store total path length to all descendant leaves
    path_to_leaves = {}

    # Initialize leaves with 0 path
    for i in range(n_leaves):
        path_to_leaves[i] = 0

    # Process merges bottom-up
    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        merge_height = Z[i, 2]
        node_id = n_leaves + i

        # Get child heights
        if left_idx < n_leaves:
            left_height = 0
            left_count = 1
        else:
            left_height = Z[left_idx - n_leaves, 2]
            left_count = int(Z[left_idx - n_leaves, 3])

        if right_idx < n_leaves:
            right_height = 0
            right_count = 1
        else:
            right_height = Z[right_idx - n_leaves, 2]
            right_count = int(Z[right_idx - n_leaves, 3])

        # Branch lengths
        left_branch = merge_height - left_height
        right_branch = merge_height - right_height

        # Total path = paths from children + new branches * count
        left_path = path_to_leaves.get(left_idx, 0) + left_branch * left_count
        right_path = path_to_leaves.get(right_idx, 0) + right_branch * right_count

        path_to_leaves[node_id] = left_path + right_path

    return path_to_leaves


def compute_ancestor_distances(Z, n_leaves):
    """
    Compute distance metrics along the ancestry path.

    For each node, compute:
    - Total path from root to this node
    - Number of ancestors
    - Average branch length along ancestry
    """
    n_merges = len(Z)
    root_id = n_leaves + n_merges - 1

    # Find parent of each node
    parent = {}
    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        node_id = n_leaves + i
        parent[left_idx] = node_id
        parent[right_idx] = node_id

    # Compute path from root to each internal node
    path_from_root = {}
    ancestor_count = {}

    for i in range(n_merges):
        node_id = n_leaves + i
        merge_height = Z[i, 2]

        # Trace up to root
        total_path = 0
        n_ancestors = 0
        current = node_id

        while current in parent:
            parent_id = parent[current]
            parent_height = Z[parent_id - n_leaves, 2]
            current_height = Z[current - n_leaves, 2] if current >= n_leaves else 0
            branch_up = parent_height - merge_height if current == node_id else 0

            # Actually compute the branch to parent
            if current >= n_leaves:
                current_h = Z[current - n_leaves, 2]
            else:
                current_h = 0
            parent_h = Z[parent_id - n_leaves, 2]
            branch = parent_h - current_h

            total_path += branch
            n_ancestors += 1
            current = parent_id

        path_from_root[node_id] = total_path
        ancestor_count[node_id] = n_ancestors

    return path_from_root, ancestor_count


def identify_true_splits(Z, y_true, n_leaves):
    """
    Identify which merges cross true cluster boundaries.
    """
    n_merges = len(Z)

    # Get cluster assignments for each leaf
    def get_leaves(idx):
        if idx < n_leaves:
            return {idx}
        merge_idx = idx - n_leaves
        left_idx = int(Z[merge_idx, 0])
        right_idx = int(Z[merge_idx, 1])
        return get_leaves(left_idx) | get_leaves(right_idx)

    true_splits = []
    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])

        left_leaves = get_leaves(left_idx)
        right_leaves = get_leaves(right_idx)

        left_labels = set(y_true[list(left_leaves)])
        right_labels = set(y_true[list(right_leaves)])

        # True split if merging different true clusters
        is_split = (
            len(left_labels & right_labels) == 0
            and len(left_labels) > 0
            and len(right_labels) > 0
        )
        true_splits.append(is_split)

    return np.array(true_splits)


def evaluate_metrics(
    branch_data, true_splits, path_to_leaves, path_from_root, ancestor_count, n_leaves
):
    """Evaluate each branch length metric as predictor of true splits."""

    # Branch length metrics (ONLY these, no height)
    metrics = {
        "left_branch": [d["left_branch"] for d in branch_data],
        "right_branch": [d["right_branch"] for d in branch_data],
        "parent_branch": [d["parent_branch"] for d in branch_data],
        "sum_branch": [d["sum_branch"] for d in branch_data],
        "max_branch": [d["max_branch"] for d in branch_data],
        "min_branch": [d["min_branch"] for d in branch_data],
        "ratio_branch": [d["ratio_branch"] for d in branch_data],
        "diff_branch": [d["diff_branch"] for d in branch_data],
        "weighted_sum": [d["weighted_sum"] for d in branch_data],
        "weighted_avg": [d["weighted_avg"] for d in branch_data],
        "prod_branch": [d["prod_branch"] for d in branch_data],
        "geom_mean": [d["geom_mean"] for d in branch_data],
        "harmonic_mean": [d["harmonic_mean"] for d in branch_data],
        "total_branch": [d["total_branch"] for d in branch_data],
        "child_parent_ratio": [d["child_parent_ratio"] for d in branch_data],
        "child_parent_diff": [d["child_parent_diff"] for d in branch_data],
        "down_up_ratio": [d["down_up_ratio"] for d in branch_data],
        "path_to_leaves": [
            path_to_leaves.get(n_leaves + d["merge_idx"], 0) for d in branch_data
        ],
        "path_from_root": [
            path_from_root.get(n_leaves + d["merge_idx"], 0) for d in branch_data
        ],
        "ancestor_count": [
            ancestor_count.get(n_leaves + d["merge_idx"], 0) for d in branch_data
        ],
    }

    # For comparison only
    metrics["height"] = [d["height"] for d in branch_data]

    results = {}
    for name, values in metrics.items():
        values = np.array(values)

        # Skip if no variance
        if np.std(values) < 1e-10:
            continue

        # Compute AUC
        try:
            auc = roc_auc_score(true_splits, values)
            corr = np.corrcoef(true_splits, values)[0, 1]
            results[name] = {"auc": auc, "corr": corr}
        except:
            pass

    return results


def run_analysis(n_runs=20):
    """Run analysis across multiple random seeds."""

    all_results = []

    for seed in range(n_runs):
        for n_clusters in [3, 5, 8]:
            for cluster_std in [0.5, 1.0, 2.0, 3.0]:  # Test different separation levels
                X, y_true = generate_data(
                    n_clusters=n_clusters, cluster_std=cluster_std, random_state=seed
                )

            # Build hierarchy
            dist_matrix = pdist(X)
            Z = linkage(dist_matrix, method="ward")
            n_leaves = len(X)

            # Compute all metrics
            branch_data = compute_branch_lengths(Z, n_leaves)
            path_to_leaves = compute_path_lengths(Z, n_leaves)
            path_from_root, ancestor_count = compute_ancestor_distances(Z, n_leaves)

            # Identify true splits
            true_splits = identify_true_splits(Z, y_true, n_leaves)

            if true_splits.sum() == 0 or true_splits.sum() == len(true_splits):
                continue

            # Evaluate
            results = evaluate_metrics(
                branch_data,
                true_splits,
                path_to_leaves,
                path_from_root,
                ancestor_count,
                n_leaves,
            )

            for metric, vals in results.items():
                all_results.append(
                    {
                        "seed": seed,
                        "n_clusters": n_clusters,
                        "metric": metric,
                        "auc": vals["auc"],
                        "corr": vals["corr"],
                    }
                )

    return all_results


def print_summary(all_results):
    """Print summary of results."""
    import pandas as pd

    df = pd.DataFrame(all_results)

    print("=" * 80)
    print("BRANCH LENGTH METRICS ANALYSIS (NO HEIGHT)")
    print("=" * 80)

    # Group by metric
    summary = (
        df.groupby("metric")
        .agg({"auc": ["mean", "std"], "corr": ["mean", "std"]})
        .round(3)
    )

    summary.columns = ["auc_mean", "auc_std", "corr_mean", "corr_std"]
    summary = summary.sort_values("auc_mean", ascending=False)

    print("\nMean AUC and Correlation across all runs:")
    print("-" * 60)
    print(summary.to_string())

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Separate into positively and negatively correlated
    pos_corr = summary[summary["corr_mean"] > 0].sort_values(
        "auc_mean", ascending=False
    )
    neg_corr = summary[summary["corr_mean"] < 0].sort_values("auc_mean", ascending=True)

    print("\n✅ POSITIVELY CORRELATED (higher value → more likely true split):")
    print("-" * 60)
    if len(pos_corr) > 0:
        print(pos_corr.to_string())
    else:
        print("  None found!")

    print("\n❌ NEGATIVELY CORRELATED (higher value → LESS likely true split):")
    print("-" * 60)
    if len(neg_corr) > 0:
        print(neg_corr.to_string())
    else:
        print("  None found!")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR BRANCH LENGTH USAGE")
    print("=" * 80)

    best_pos = pos_corr.head(3) if len(pos_corr) > 0 else None
    best_neg = neg_corr.tail(3) if len(neg_corr) > 0 else None

    if best_pos is not None and len(best_pos) > 0:
        print("\nBest branch length metrics (positive correlation):")
        for metric in best_pos.index:
            auc = best_pos.loc[metric, "auc_mean"]
            corr = best_pos.loc[metric, "corr_mean"]
            print(f"  {metric}: AUC={auc:.3f}, corr={corr:+.3f}")

    if best_neg is not None and len(best_neg) > 0:
        print("\nMetrics that could work INVERTED:")
        for metric in best_neg.index:
            auc = best_neg.loc[metric, "auc_mean"]
            corr = best_neg.loc[metric, "corr_mean"]
            inv_auc = 1 - auc
            print(
                f"  1/{metric} or -{metric}: AUC={inv_auc:.3f} (original corr={corr:+.3f})"
            )


if __name__ == "__main__":
    print("Running branch length analysis...")
    results = run_analysis(n_runs=30)
    print_summary(results)
