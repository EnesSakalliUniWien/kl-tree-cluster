"""
Purpose: Test branch lengths for EACH of the two statistical tests:.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_per_test_metrics__branch_length__validation.py
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
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


def get_leaves(Z, idx, n_leaves):
    """Get all leaves under a node."""
    if idx < n_leaves:
        return {idx}
    merge_idx = idx - n_leaves
    left_idx = int(Z[merge_idx, 0])
    right_idx = int(Z[merge_idx, 1])
    return get_leaves(Z, left_idx, n_leaves) | get_leaves(Z, right_idx, n_leaves)


def compute_all_metrics(Z, n_leaves):
    """
    Compute branch length metrics for each node.

    Returns data organized by:
    - Each internal node (for child-parent test perspective)
    - Each parent node with two children (for sibling test perspective)
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

    # Store data for each merge
    merge_data = []

    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        merge_height = Z[i, 2]
        node_id = n_leaves + i

        # Child heights
        left_height = 0 if left_idx < n_leaves else Z[left_idx - n_leaves, 2]
        right_height = 0 if right_idx < n_leaves else Z[right_idx - n_leaves, 2]

        # Branch lengths TO children
        left_branch = merge_height - left_height
        right_branch = merge_height - right_height
        sum_branch = left_branch + right_branch

        # Branch length TO parent (from this node)
        if node_id in parent_of:
            parent_id = parent_of[node_id]
            parent_height = Z[parent_id - n_leaves, 2]
            branch_to_parent = parent_height - merge_height
        else:
            branch_to_parent = 0

        # For child-parent test: branch from child to this parent
        # Left child's branch to parent = left_branch
        # Right child's branch to parent = right_branch

        merge_data.append(
            {
                "merge_idx": i,
                "node_id": node_id,
                "height": merge_height,
                "left_idx": left_idx,
                "right_idx": right_idx,
                "left_branch": left_branch,  # Distance from this node to left child
                "right_branch": right_branch,  # Distance from this node to right child
                "sum_branch": sum_branch,  # Total distance to both children
                "branch_to_parent": branch_to_parent,  # Distance up to parent
                "geom_mean_children": np.sqrt(left_branch * right_branch)
                if left_branch > 0 and right_branch > 0
                else 0,
            }
        )

    return merge_data


def identify_ground_truth(Z, y_true, n_leaves):
    """
    Identify ground truth for both test types:

    1. Child-Parent test (per child node):
       - Is child's cluster composition different from parent's?

    2. Sibling test (per parent node):
       - Do the two children belong to different true clusters?
    """
    n_merges = len(Z)

    # For sibling test: does this merge cross cluster boundaries?
    sibling_truth = []
    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])

        left_leaves = get_leaves(Z, left_idx, n_leaves)
        right_leaves = get_leaves(Z, right_idx, n_leaves)

        left_labels = set(y_true[list(left_leaves)])
        right_labels = set(y_true[list(right_leaves)])

        # True if siblings have NO overlapping clusters
        is_boundary = len(left_labels & right_labels) == 0
        sibling_truth.append(is_boundary)

    # For child-parent test: does child have different composition than parent?
    child_parent_truth = []
    child_branches = []  # The actual branch length for this child

    for i in range(n_merges):
        left_idx = int(Z[i, 0])
        right_idx = int(Z[i, 1])
        node_id = n_leaves + i
        merge_height = Z[i, 2]

        parent_leaves = get_leaves(Z, node_id, n_leaves)
        parent_labels = set(y_true[list(parent_leaves)])

        # Left child
        left_leaves = get_leaves(Z, left_idx, n_leaves)
        left_labels = set(y_true[list(left_leaves)])
        left_height = 0 if left_idx < n_leaves else Z[left_idx - n_leaves, 2]
        left_branch = merge_height - left_height

        # Does left child have different composition than parent?
        left_differs = left_labels != parent_labels
        child_parent_truth.append(left_differs)
        child_branches.append(left_branch)

        # Right child
        right_leaves = get_leaves(Z, right_idx, n_leaves)
        right_labels = set(y_true[list(right_leaves)])
        right_height = 0 if right_idx < n_leaves else Z[right_idx - n_leaves, 2]
        right_branch = merge_height - right_height

        right_differs = right_labels != parent_labels
        child_parent_truth.append(right_differs)
        child_branches.append(right_branch)

    return {
        "sibling_truth": np.array(sibling_truth),
        "child_parent_truth": np.array(child_parent_truth),
        "child_branches": np.array(child_branches),
    }


def evaluate_for_sibling_test(merge_data, sibling_truth):
    """
    Evaluate branch length metrics for the SIBLING test.

    Question: Which branch length metric predicts whether
    siblings belong to different clusters?
    """
    metrics = {
        "sum_branch": [d["sum_branch"] for d in merge_data],
        "left_branch": [d["left_branch"] for d in merge_data],
        "right_branch": [d["right_branch"] for d in merge_data],
        "max_branch": [max(d["left_branch"], d["right_branch"]) for d in merge_data],
        "min_branch": [min(d["left_branch"], d["right_branch"]) for d in merge_data],
        "geom_mean": [d["geom_mean_children"] for d in merge_data],
        "branch_to_parent": [d["branch_to_parent"] for d in merge_data],
        "height": [d["height"] for d in merge_data],
    }

    results = {}
    for name, values in metrics.items():
        values = np.array(values)
        if np.std(values) < 1e-10:
            continue
        try:
            auc = roc_auc_score(sibling_truth, values)
            corr = np.corrcoef(sibling_truth, values)[0, 1]
            results[name] = {"auc": auc, "corr": corr}
        except:
            pass

    return results


def evaluate_for_child_parent_test(child_branches, child_parent_truth):
    """
    Evaluate branch length for the CHILD-PARENT test.

    Question: Does the branch length from child to parent
    predict whether the child has different cluster composition?
    """
    if len(child_branches) == 0 or child_parent_truth.sum() == 0:
        return {}

    try:
        auc = roc_auc_score(child_parent_truth, child_branches)
        corr = np.corrcoef(child_parent_truth, child_branches)[0, 1]
        return {"branch_to_parent": {"auc": auc, "corr": corr}}
    except:
        return {}


def run_analysis(n_runs=20):
    """Run analysis for both test types."""

    sibling_results = []
    child_parent_results = []

    for seed in range(n_runs):
        for n_clusters in [3, 5, 8]:
            for cluster_std in [0.5, 1.0, 2.0, 3.0]:
                X, y_true = generate_data(
                    n_clusters=n_clusters, cluster_std=cluster_std, random_state=seed
                )

                dist_matrix = pdist(X)
                Z = linkage(dist_matrix, method="ward")
                n_leaves = len(X)

                merge_data = compute_all_metrics(Z, n_leaves)
                truth = identify_ground_truth(Z, y_true, n_leaves)

                # Skip if no variance in truth
                if truth["sibling_truth"].sum() == 0 or truth[
                    "sibling_truth"
                ].sum() == len(truth["sibling_truth"]):
                    continue

                # Evaluate for sibling test
                sib_eval = evaluate_for_sibling_test(merge_data, truth["sibling_truth"])
                for metric, vals in sib_eval.items():
                    sibling_results.append(
                        {
                            "seed": seed,
                            "n_clusters": n_clusters,
                            "cluster_std": cluster_std,
                            "metric": metric,
                            "auc": vals["auc"],
                            "corr": vals["corr"],
                        }
                    )

                # Evaluate for child-parent test
                if truth["child_parent_truth"].sum() > 0 and truth[
                    "child_parent_truth"
                ].sum() < len(truth["child_parent_truth"]):
                    cp_eval = evaluate_for_child_parent_test(
                        truth["child_branches"], truth["child_parent_truth"]
                    )
                    for metric, vals in cp_eval.items():
                        child_parent_results.append(
                            {
                                "seed": seed,
                                "n_clusters": n_clusters,
                                "cluster_std": cluster_std,
                                "metric": metric,
                                "auc": vals["auc"],
                                "corr": vals["corr"],
                            }
                        )

    return sibling_results, child_parent_results


def print_results(sibling_results, child_parent_results):
    """Print summary for both test types."""
    import pandas as pd

    print("=" * 80)
    print("BRANCH LENGTHS FOR EACH STATISTICAL TEST")
    print("=" * 80)

    # SIBLING TEST
    print("\n" + "-" * 80)
    print("1. SIBLING DIVERGENCE TEST")
    print("   Question: Do the two children belong to different true clusters?")
    print("-" * 80)

    sib_df = pd.DataFrame(sibling_results)
    sib_summary = (
        sib_df.groupby("metric")
        .agg({"auc": ["mean", "std"], "corr": ["mean", "std"]})
        .round(3)
    )
    sib_summary.columns = ["auc_mean", "auc_std", "corr_mean", "corr_std"]
    sib_summary = sib_summary.sort_values("auc_mean", ascending=False)

    print("\nBest metrics for SIBLING test:")
    print(sib_summary.to_string())

    # CHILD-PARENT TEST
    print("\n" + "-" * 80)
    print("2. CHILD-PARENT DIVERGENCE TEST")
    print("   Question: Does child have different cluster composition than parent?")
    print("-" * 80)

    cp_df = pd.DataFrame(child_parent_results)
    if len(cp_df) > 0:
        cp_summary = (
            cp_df.groupby("metric")
            .agg({"auc": ["mean", "std"], "corr": ["mean", "std"]})
            .round(3)
        )
        cp_summary.columns = ["auc_mean", "auc_std", "corr_mean", "corr_std"]

        print("\nBranch length (child → parent) for CHILD-PARENT test:")
        print(cp_summary.to_string())
    else:
        print("\nInsufficient data for child-parent analysis")

    # RECOMMENDATIONS
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_sibling = sib_summary.iloc[0]
    print(f"\nFor SIBLING test: Use '{sib_summary.index[0]}'")
    print(
        f"  AUC = {best_sibling['auc_mean']:.3f}, Correlation = {best_sibling['corr_mean']:+.3f}"
    )

    if len(cp_df) > 0:
        print(f"\nFor CHILD-PARENT test: Use 'branch_to_parent' (the edge length)")
        print(
            f"  AUC = {cp_summary.loc['branch_to_parent', 'auc_mean']:.3f}, "
            f"Correlation = {cp_summary.loc['branch_to_parent', 'corr_mean']:+.3f}"
        )


if __name__ == "__main__":
    print("Analyzing branch lengths for each statistical test...")
    sibling_results, child_parent_results = run_analysis(n_runs=30)
    print_results(sibling_results, child_parent_results)
