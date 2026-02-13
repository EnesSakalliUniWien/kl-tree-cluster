"""Debug script to analyze branch length scale and variance components.

Investigates why the nested variance correction leads to over-splitting.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2
import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from kl_clustering_analysis.tree.poset_tree import PosetTree
from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)


def analyze_branch_lengths(tree):
    """Analyze branch length distribution in tree."""
    print("=" * 70)
    print("BRANCH LENGTH DISTRIBUTION")
    print("=" * 70)

    branch_lengths = []
    for parent, child in tree.edges():
        bl = tree.edges[parent, child].get("branch_length", 0)
        branch_lengths.append(bl)

    bl = np.array(branch_lengths)
    print(f"  Count: {len(bl)}")
    print(f"  Min:   {bl.min():.6f}")
    print(f"  Max:   {bl.max():.6f}")
    print(f"  Mean:  {bl.mean():.6f}")
    print(f"  Median:{np.median(bl):.6f}")
    print(f"  Std:   {bl.std():.6f}")
    print()

    # Percentiles
    for p in [10, 25, 50, 75, 90]:
        print(f"  {p}th percentile: {np.percentile(bl, p):.6f}")

    return bl


def analyze_variance_components(tree, n_edges=10):
    """Compare variance formulas for sample edges."""
    print("\n" + "=" * 70)
    print("VARIANCE COMPONENT ANALYSIS")
    print("=" * 70)

    # Sort edges by branch length to get interesting ones
    edges = [(p, c, tree.edges[p, c].get("branch_length", 0)) for p, c in tree.edges()]
    edges = sorted(edges, key=lambda x: -x[2])[:n_edges]  # highest BL first

    results = []

    for parent_id, child_id, _ in edges:
        child_dist = np.array(tree.nodes[child_id]["distribution"])
        parent_dist = np.array(tree.nodes[parent_id]["distribution"])
        n_child = tree.nodes[child_id]["leaf_count"]
        n_parent = tree.nodes[parent_id]["leaf_count"]
        bl = tree.edges[parent_id, child_id].get("branch_length", 1.0)

        # Bernoulli variance component
        bernoulli_var = parent_dist * (1 - parent_dist)

        # Old variance (naive): Var = θ(1-θ)/n_child
        var_naive = bernoulli_var / n_child

        # New variance (nested): Var = θ(1-θ) × (1/n_child - 1/n_parent)
        var_nested = bernoulli_var * (1.0 / n_child - 1.0 / n_parent)

        # With branch length adjustment
        var_with_bl = var_nested * bl

        results.append(
            {
                "edge": f"{parent_id}->{child_id}",
                "n_child": n_child,
                "n_parent": n_parent,
                "n_ratio": n_child / n_parent,
                "branch_length": bl,
                "var_naive": var_naive.mean(),
                "var_nested": var_nested.mean(),
                "var_with_bl": var_with_bl.mean(),
                "nested_to_naive": var_nested.mean() / var_naive.mean()
                if var_naive.mean() > 0
                else 0,
                "bl_to_naive": var_with_bl.mean() / var_naive.mean()
                if var_naive.mean() > 0
                else 0,
            }
        )

    print(
        f"\n{'Edge':<15} {'n_child':>8} {'n_parent':>8} {'n_ratio':>8} {'BL':>8} {'var_ratio':>10} {'BL_ratio':>10}"
    )
    print("-" * 77)

    for r in results:
        print(
            f"{r['edge']:<15} {r['n_child']:>8} {r['n_parent']:>8} {r['n_ratio']:>8.2f} {r['branch_length']:>8.4f} {r['nested_to_naive']:>10.4f} {r['bl_to_naive']:>10.4f}"
        )

    return results


def analyze_z_scores_and_pvalues(tree, n_edges=5):
    """Compare z-scores and p-values for different variance formulas."""
    print("\n" + "=" * 70)
    print("Z-SCORE AND P-VALUE COMPARISON")
    print("=" * 70)

    # Sort edges by branch length to get interesting ones
    edges = [(p, c, tree.edges[p, c].get("branch_length", 0)) for p, c in tree.edges()]
    edges = sorted(edges, key=lambda x: -x[2])[:n_edges]  # highest BL first

    for parent_id, child_id, _ in edges:
        child_dist = np.array(tree.nodes[child_id]["distribution"])
        parent_dist = np.array(tree.nodes[parent_id]["distribution"])
        n_child = tree.nodes[child_id]["leaf_count"]
        n_parent = tree.nodes[parent_id]["leaf_count"]
        bl = tree.edges[parent_id, child_id].get("branch_length", 1.0)

        diff = child_dist - parent_dist
        d = len(diff)  # dimensionality

        # Naive variance
        var_naive = parent_dist * (1 - parent_dist) / n_child
        var_naive = np.maximum(var_naive, 1e-10)
        z_naive = diff / np.sqrt(var_naive)

        # Nested variance
        var_nested = parent_dist * (1 - parent_dist) * (1.0 / n_child - 1.0 / n_parent)
        var_nested = np.maximum(var_nested, 1e-10)
        z_nested = diff / np.sqrt(var_nested)

        # With branch length
        var_bl = var_nested * bl
        var_bl = np.maximum(var_bl, 1e-10)
        z_bl = diff / np.sqrt(var_bl)

        # Chi-square statistics (sum of squared z-scores)
        stat_naive = np.sum(z_naive**2)
        stat_nested = np.sum(z_nested**2)
        stat_bl = np.sum(z_bl**2)

        # P-values (using d as df for unprojected test)
        p_naive = chi2.sf(stat_naive, d)
        p_nested = chi2.sf(stat_nested, d)
        p_bl = chi2.sf(stat_bl, d)

        print(f"\nEdge {parent_id} -> {child_id}:")
        print(f"  n_child={n_child}, n_parent={n_parent}, BL={bl:.4f}, d={d}")
        print(f"  Mean |diff|: {np.abs(diff).mean():.6f}")
        print()
        print(f"  Naive:   ||z||² = {stat_naive:10.1f}, p = {p_naive:.2e}")
        print(f"  Nested:  ||z||² = {stat_nested:10.1f}, p = {p_nested:.2e}")
        print(f"  With BL: ||z||² = {stat_bl:10.1f}, p = {p_bl:.2e}")
        print()
        print(f"  Nested/Naive ratio: {stat_nested / stat_naive:.2f}x")
        print(f"  BL/Naive ratio:     {stat_bl / stat_naive:.2f}x")


def analyze_what_bl_should_be(tree, n_edges=10):
    """Calculate what branch length would be needed to match naive variance."""
    print("\n" + "=" * 70)
    print("BRANCH LENGTH NEEDED TO MATCH NAIVE VARIANCE")
    print("=" * 70)

    # Sort edges by branch length to get interesting ones
    all_edges = [
        (p, c, tree.edges[p, c].get("branch_length", 0)) for p, c in tree.edges()
    ]
    edges = sorted(all_edges, key=lambda x: -x[2])[:n_edges]  # highest BL first

    print(
        f"\n{'Edge':<15} {'n_child':>8} {'n_ratio':>8} {'actual_BL':>10} {'needed_BL':>10} {'BL_mult':>10}"
    )
    print("-" * 71)

    for parent_id, child_id, _ in edges:
        n_child = tree.nodes[child_id]["leaf_count"]
        n_parent = tree.nodes[parent_id]["leaf_count"]
        bl = tree.edges[parent_id, child_id].get("branch_length", 1.0)

        # Nested variance factor: (1/n_child - 1/n_parent)
        # Naive variance factor: 1/n_child
        #
        # For them to be equal with BL:
        # (1/n_child - 1/n_parent) * BL = 1/n_child
        # BL = (1/n_child) / (1/n_child - 1/n_parent)
        # BL = n_parent / (n_parent - n_child)

        nested_factor = 1.0 / n_child - 1.0 / n_parent
        naive_factor = 1.0 / n_child

        if nested_factor > 0:
            needed_bl = naive_factor / nested_factor
            multiplier = needed_bl / bl if bl > 0 else float("inf")
        else:
            needed_bl = float("inf")
            multiplier = float("inf")

        print(
            f"{parent_id}->{child_id:<8} {n_child:>8} {n_child / n_parent:>8.2f} {bl:>10.4f} {needed_bl:>10.4f} {multiplier:>10.2f}x"
        )


def main():
    print("=" * 70)
    print("BRANCH LENGTH SCALE INVESTIGATION")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    data_dict, labels = generate_random_feature_matrix(
        n_rows=80, n_cols=8, n_clusters=2, entropy_param=0.1, random_seed=42
    )
    # Convert dict to DataFrame
    data_df = pd.DataFrame.from_dict(data_dict, orient="index")
    X = data_df.values

    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features, 2 clusters")

    # Build tree
    Z = linkage(pdist(X, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.populate_node_divergences(data_df)

    print(f"Tree: {tree.number_of_nodes()} nodes, {tree.number_of_edges()} edges")

    # Analyses
    analyze_branch_lengths(tree)
    analyze_variance_components(tree)
    analyze_z_scores_and_pvalues(tree)
    analyze_what_bl_should_be(tree)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The nested variance correction:
    Var_nested = θ(1-θ) × (1/n_child - 1/n_parent)

is SMALLER than the naive variance:
    Var_naive = θ(1-θ) / n_child

by a factor of (n_parent - n_child) / n_parent.

When n_child is 50% of n_parent, nested variance is 50% of naive.
This makes z-scores √2 ≈ 1.41× larger → p-values much smaller.

The branch length should compensate, but typical BLs are ~0.5-1.0,
which is NOT enough to counteract the 2× reduction in variance.

POSSIBLE FIXES:
1. Normalize branch lengths by tree height
2. Use a different BL scaling (e.g., BL² or exp(BL))
3. Don't use nested correction (revert to naive)
4. Increase significance threshold (lower α)
""")


if __name__ == "__main__":
    main()
