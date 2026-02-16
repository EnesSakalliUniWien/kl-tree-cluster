"""
Purpose: Debug script to compare variance formulas and their effect on clustering.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_variance_formula_comparison__branch_length__diagnostic.py
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from sklearn.metrics import adjusted_rand_score

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def analyze_single_case(true_k=2, entropy=0.1, n_rows=200, n_cols=50, seed=42):
    """Analyze a single test case in detail."""
    print("=" * 70)
    print(f"CASE: K={true_k}, entropy={entropy}, n={n_rows}, d={n_cols}")
    print("=" * 70)

    # Generate data
    data_dict, labels = generate_random_feature_matrix(
        n_rows=n_rows,
        n_cols=n_cols,
        n_clusters=true_k,
        entropy_param=entropy,
        random_seed=seed,
    )
    df = pd.DataFrame.from_dict(data_dict, orient="index")

    # Build tree
    Z = linkage(pdist(df.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())

    print(f"\nTree: {len(tree.nodes)} nodes, {len(tree.edges)} edges")

    # Get branch lengths
    branch_lengths = [d.get("branch_length", 0) for _, _, d in tree.edges(data=True)]
    print(
        f"Branch lengths: min={min(branch_lengths):.4f}, max={max(branch_lengths):.4f}, mean={np.mean(branch_lengths):.4f}"
    )

    # Populate distributions
    tree.populate_node_divergences(df)

    # Analyze key edges (those with larger branch lengths = potential cluster boundaries)
    print("\n" + "-" * 70)
    print("TOP 10 EDGES BY BRANCH LENGTH (likely cluster boundaries)")
    print("-" * 70)

    edges_with_bl = [
        (p, c, tree.edges[p, c].get("branch_length", 0)) for p, c in tree.edges()
    ]
    edges_with_bl.sort(key=lambda x: -x[2])

    print(
        f"\n{'Edge':<20} {'n_child':>8} {'n_parent':>8} {'BL':>8} {'|diff|':>10} {'z_naive':>10} {'z_nested':>10} {'p_naive':>12} {'p_nested':>12}"
    )
    print("-" * 120)

    for parent_id, child_id, bl in edges_with_bl[:15]:
        child_dist = np.array(tree.nodes[child_id]["distribution"])
        parent_dist = np.array(tree.nodes[parent_id]["distribution"])
        n_child = tree.nodes[child_id]["leaf_count"]
        n_parent = tree.nodes[parent_id]["leaf_count"]

        diff = child_dist - parent_dist
        mean_abs_diff = np.abs(diff).mean()

        # Naive variance
        var_naive = parent_dist * (1 - parent_dist) / n_child
        var_naive = np.maximum(var_naive, 1e-10)
        z_naive = diff / np.sqrt(var_naive)
        stat_naive = np.sum(z_naive**2)

        # Nested variance
        nested_factor = 1.0 / n_child - 1.0 / n_parent
        if nested_factor > 0:
            var_nested = parent_dist * (1 - parent_dist) * nested_factor
            var_nested = np.maximum(var_nested, 1e-10)
            z_nested = diff / np.sqrt(var_nested)
            stat_nested = np.sum(z_nested**2)
        else:
            stat_nested = 0

        d = len(diff)
        p_naive = chi2.sf(stat_naive, d)
        p_nested = chi2.sf(stat_nested, d) if nested_factor > 0 else 1.0

        # Format edge name
        edge_name = f"{parent_id}->{child_id}"
        if len(edge_name) > 20:
            edge_name = edge_name[:17] + "..."

        print(
            f"{edge_name:<20} {n_child:>8} {n_parent:>8} {bl:>8.4f} {mean_abs_diff:>10.4f} {stat_naive:>10.1f} {stat_nested:>10.1f} {p_naive:>12.2e} {p_nested:>12.2e}"
        )

    # Run decomposition
    print("\n" + "-" * 70)
    print("DECOMPOSITION RESULTS")
    print("-" * 70)

    decomp = tree.decompose(leaf_data=df, alpha_local=0.05, sibling_alpha=0.05)
    stats_df = tree.stats_df

    # Show test results
    if "Child_Parent_Divergence_P_Value" in stats_df.columns:
        pvals = stats_df["Child_Parent_Divergence_P_Value"].dropna()
        sig = stats_df.get("Child_Parent_Divergence_Significant", pd.Series()).sum()
        print(f"\nChild-Parent Tests:")
        print(f"  Significant: {sig}/{len(pvals)}")
        print(
            f"  P-values: min={pvals.min():.2e}, median={pvals.median():.2e}, max={pvals.max():.2e}"
        )

    if "Sibling_Divergence_P_Value" in stats_df.columns:
        sib_pvals = stats_df["Sibling_Divergence_P_Value"].dropna()
        sib_diff = stats_df.get("Sibling_BH_Different", pd.Series()).sum()
        print(f"\nSibling Tests:")
        print(f"  Different: {sib_diff}/{len(sib_pvals)}")
        if len(sib_pvals) > 0:
            print(
                f"  P-values: min={sib_pvals.min():.2e}, median={sib_pvals.median():.2e}, max={sib_pvals.max():.2e}"
            )

    # Extract clusters
    cluster_assignments = decomp.get("cluster_assignments", {})
    leaf_to_cluster = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            leaf_to_cluster[leaf] = cluster_id

    pred_labels = [leaf_to_cluster.get(name, 0) for name in df.index]
    true_labels = [labels[name] for name in df.index]

    pred_k = len(set(pred_labels))
    ari = adjusted_rand_score(true_labels, pred_labels)

    print(f"\nClustering Result:")
    print(f"  True K: {true_k}")
    print(f"  Pred K: {pred_k}")
    print(f"  ARI: {ari:.4f}")

    diff_k = pred_k - true_k
    if diff_k == 0:
        print(f"  Status: ✓ EXACT")
    elif diff_k > 0:
        print(f"  Status: OVER-SPLIT by {diff_k}")
    else:
        print(f"  Status: UNDER-SPLIT by {-diff_k}")

    return {"true_k": true_k, "pred_k": pred_k, "ari": ari, "entropy": entropy}


def main():
    print("VARIANCE FORMULA COMPARISON DEBUG")
    print("=" * 70)
    print()

    results = []

    # Test key cases
    for true_k in [2, 3, 4]:
        for entropy in [0.1, 0.2]:
            result = analyze_single_case(true_k=true_k, entropy=entropy)
            results.append(result)
            print()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'K':>3} | {'entropy':>7} | {'Pred K':>6} | {'ARI':>6} | {'Status':<15}")
    print("-" * 50)
    for r in results:
        diff = r["pred_k"] - r["true_k"]
        status = (
            "✓" if diff == 0 else (f"+{diff} OVER" if diff > 0 else f"{diff} UNDER")
        )
        print(
            f"{r['true_k']:>3} | {r['entropy']:>7.1f} | {r['pred_k']:>6} | {r['ari']:>6.3f} | {status:<15}"
        )


if __name__ == "__main__":
    main()
