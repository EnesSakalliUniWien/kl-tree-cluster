"""
Purpose: Debug script to analyze branch length integration options.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_branch_length_integration_diagnostic__branch_length__analysis.py
"""

import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2

from benchmarks.shared.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree


def get_tree_height(tree):
    """Compute tree height (max root-to-leaf path length)."""
    root = tree.root()

    def height_from(node):
        children = list(tree.successors(node))
        if not children:
            return 0
        return max(
            tree.edges[node, c].get("branch_length", 0) + height_from(c)
            for c in children
        )

    return height_from(root)


def analyze_branch_lengths(tree):
    """Analyze branch length distribution."""
    bls = [tree.edges[p, c].get("branch_length", 0) for p, c in tree.edges()]
    bls = np.array(bls)

    height = get_tree_height(tree)

    print("=" * 70)
    print("BRANCH LENGTH ANALYSIS")
    print("=" * 70)
    print(f"\nRaw Branch Lengths (from linkage):")
    print(f"  Count: {len(bls)}")
    print(f"  Min:   {bls.min():.4f}")
    print(f"  Max:   {bls.max():.4f}")
    print(f"  Mean:  {bls.mean():.4f}")
    print(f"  Median:{np.median(bls):.4f}")
    print(f"  Std:   {bls.std():.4f}")
    print(f"  Tree height: {height:.4f}")

    return bls, height


def test_normalization_strategies(bls, height):
    """Test different branch length normalization strategies."""
    print("\n" + "=" * 70)
    print("NORMALIZATION STRATEGIES")
    print("=" * 70)

    strategies = {
        "raw": bls,
        "by_height": bls / height if height > 0 else bls,
        "by_mean": bls / bls.mean() if bls.mean() > 0 else bls,
        "1 + bl": 1 + bls,
        "1 + bl/mean": 1 + bls / bls.mean() if bls.mean() > 0 else 1 + bls,
        "exp(bl)": np.exp(bls),
    }

    print(
        f"\n{'Strategy':<15} {'Min':>8} {'Max':>8} {'Mean':>8} {'Effect on Variance':>20}"
    )
    print("-" * 60)

    for name, normalized in strategies.items():
        # For Felsenstein: Var_adjusted = Var_base * BL
        # If mean(BL) < 1: variance shrinks, z grows, more significant
        # If mean(BL) > 1: variance grows, z shrinks, less significant
        effect = (
            "SHRINKS (oversplit)"
            if normalized.mean() < 1
            else "GROWS (undersplit)"
            if normalized.mean() > 1
            else "NEUTRAL"
        )
        print(
            f"{name:<15} {normalized.min():>8.4f} {normalized.max():>8.4f} {normalized.mean():>8.4f} {effect:>20}"
        )

    return strategies


def simulate_test_with_normalization(tree, df, strategies):
    """Simulate how different normalizations affect test results."""
    print("\n" + "=" * 70)
    print("SIMULATED TEST RESULTS WITH DIFFERENT NORMALIZATIONS")
    print("=" * 70)

    # Populate distributions
    tree.populate_node_divergences(df)

    # Find edges with significant branch length (potential cluster boundaries)
    edges_with_bl = [
        (p, c, tree.edges[p, c].get("branch_length", 0)) for p, c in tree.edges()
    ]
    edges_with_bl.sort(key=lambda x: -x[2])

    # Take top 5 edges by branch length
    test_edges = edges_with_bl[:5]

    print(f"\nTesting top 5 edges by branch length:")

    for parent_id, child_id, bl in test_edges:
        child_dist = np.array(tree.nodes[child_id]["distribution"])
        parent_dist = np.array(tree.nodes[parent_id]["distribution"])
        n_child = tree.nodes[child_id]["leaf_count"]
        n_parent = tree.nodes[parent_id]["leaf_count"]

        diff = child_dist - parent_dist
        d = len(diff)

        # Base variance (naive formula)
        var_base = parent_dist * (1 - parent_dist) / n_child
        var_base = np.maximum(var_base, 1e-10)

        z_base = diff / np.sqrt(var_base)
        stat_base = np.sum(z_base**2)
        p_base = chi2.sf(stat_base, d)

        print(f"\n  Edge {parent_id} -> {child_id} (n_child={n_child}, BL={bl:.4f}):")
        print(f"    Base (no BL):  χ² = {stat_base:8.1f}, p = {p_base:.2e}")

        height = get_tree_height(tree)
        bl_mean = np.mean(
            [tree.edges[p, c].get("branch_length", 0) for p, c in tree.edges()]
        )

        # Test each normalization
        normalizations = {
            "raw BL": bl,
            "BL/height": bl / height if height > 0 else bl,
            "BL/mean": bl / bl_mean if bl_mean > 0 else bl,
            "1 + BL": 1 + bl,
            "1 + BL/mean": 1 + bl / bl_mean if bl_mean > 0 else 1 + bl,
        }

        for name, bl_norm in normalizations.items():
            if bl_norm > 0:
                var_adj = var_base * bl_norm
                z_adj = diff / np.sqrt(var_adj)
                stat_adj = np.sum(z_adj**2)
                p_adj = chi2.sf(stat_adj, d)
                change = "↓ less sig" if p_adj > p_base else "↑ more sig"
                print(
                    f"    {name:<12}: χ² = {stat_adj:8.1f}, p = {p_adj:.2e} ({change})"
                )


def recommend_normalization():
    """Print recommendation based on analysis."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
For Felsenstein's PIC to work correctly, we need:
    Var_adjusted = Var_base × BL_normalized

Where BL_normalized should:
1. Be ≥ 1 on average (to not shrink variance)
2. Scale proportionally to expected divergence
3. Be larger for deep splits, smaller for shallow splits

RECOMMENDED APPROACH: Use "1 + BL/mean(BL)"
- This ensures BL_normalized ≥ 1 (no variance shrinkage)
- Mean(BL_normalized) = 2 (moderate variance inflation)
- Preserves relative ordering of branch lengths
- Edges with above-average BL get more variance → harder to reject H₀
- Edges with below-average BL get less extra variance → easier to reject H₀

This makes intuitive sense:
- Long branches (cluster boundaries) → need stronger evidence
- Short branches (noise) → easier to dismiss as noise

FOR CHILD-PARENT TEST:
- Use nested variance: Var = θ(1-θ) × (1/n_child - 1/n_parent)
- Multiply by normalized BL: (1 + BL/mean_BL)
- This accounts for both the statistical correlation AND the tree topology

FOR SIBLING TEST:
- Use pooled variance: Var = θ(1-θ) × (1/n₁ + 1/n₂)
- Multiply by normalized BL_sum: (1 + (BL_L + BL_R)/mean_BL)
- Both siblings accumulate divergence from common ancestor
""")


def main():
    print("BRANCH LENGTH INTEGRATION ANALYSIS")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    data_dict, labels = generate_random_feature_matrix(
        n_rows=100, n_cols=20, n_clusters=3, entropy_param=0.1, random_seed=42
    )
    df = pd.DataFrame.from_dict(data_dict, orient="index")

    print(f"\nTest data: {df.shape[0]} samples, {df.shape[1]} features, 3 clusters")

    # Build tree
    Z = linkage(pdist(df.values, metric="hamming"), method="average")
    tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())

    print(f"Tree: {len(tree.nodes)} nodes, {len(tree.edges)} edges")

    # Analyze branch lengths
    bls, height = analyze_branch_lengths(tree)

    # Test normalizations
    strategies = test_normalization_strategies(bls, height)

    # Simulate tests
    simulate_test_with_normalization(tree, df, strategies)

    # Recommendation
    recommend_normalization()


if __name__ == "__main__":
    main()
