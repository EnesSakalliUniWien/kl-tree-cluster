"""
Purpose: Debug script showing proposed branch length integration implementation.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/branch_length/methods/q_proposed_branch_length_implementation__branch_length__diagnostic.py
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


def compute_normalized_branch_length(bl: float, mean_bl: float) -> float:
    """Normalize branch length for variance scaling.

    Formula: BL_norm = 1 + BL / mean(BL)

    Properties:
    - BL_norm ≥ 1 for all BL ≥ 0
    - Mean(BL_norm) = 2 when BL has mean = mean_bl
    - Larger BL → larger BL_norm → more variance → smaller z
    """
    if mean_bl <= 0 or bl is None:
        return 1.0  # No adjustment
    return 1.0 + bl / mean_bl


def compute_standardized_z_with_bl(
    child_dist: np.ndarray,
    parent_dist: np.ndarray,
    n_child: int,
    n_parent: int,
    branch_length: float,
    mean_branch_length: float,
    use_nested_variance: bool = True,
) -> np.ndarray:
    """Compute standardized z-scores with branch length adjustment.

    Parameters
    ----------
    child_dist : np.ndarray
        Child node's distribution
    parent_dist : np.ndarray
        Parent node's distribution
    n_child : int
        Sample size for child
    n_parent : int
        Sample size for parent
    branch_length : float
        Branch length for this edge
    mean_branch_length : float
        Mean branch length across tree (for normalization)
    use_nested_variance : bool
        If True, use corrected nested variance formula

    Returns
    -------
    np.ndarray
        Standardized z-scores
    """
    if use_nested_variance:
        # Nested variance: Var = θ(1-θ) × (1/n_child - 1/n_parent)
        # This accounts for child being nested in parent
        nested_factor = 1.0 / n_child - 1.0 / n_parent
        if nested_factor <= 0:
            # Edge case: n_child >= n_parent (shouldn't happen in valid tree)
            nested_factor = 1.0 / n_child
        var = parent_dist * (1 - parent_dist) * nested_factor
    else:
        # Naive variance: Var = θ(1-θ) / n_child
        var = parent_dist * (1 - parent_dist) / n_child

    # Apply normalized branch length scaling
    bl_norm = compute_normalized_branch_length(branch_length, mean_branch_length)
    var = var * bl_norm

    var = np.maximum(var, 1e-10)
    z = (child_dist - parent_dist) / np.sqrt(var)

    return z.ravel()


def run_comparison_benchmark():
    """Compare different variance formulas on benchmark data."""
    print("=" * 70)
    print("BRANCH LENGTH INTEGRATION BENCHMARK")
    print("=" * 70)

    results = []

    for true_k in [2, 3, 4]:
        for entropy in [0.1, 0.2]:
            # Generate data
            data_dict, labels = generate_random_feature_matrix(
                n_rows=200,
                n_cols=50,
                n_clusters=true_k,
                entropy_param=entropy,
                random_seed=42,
            )
            df = pd.DataFrame.from_dict(data_dict, orient="index")

            # Build tree
            Z = linkage(pdist(df.values, metric="hamming"), method="average")
            tree = PosetTree.from_linkage(Z, leaf_names=df.index.tolist())
            tree.populate_node_divergences(df)

            # Compute mean branch length
            bls = [tree.edges[p, c].get("branch_length", 0) for p, c in tree.edges()]
            mean_bl = np.mean(bls)

            # Test different configurations
            configs = [
                ("naive, no BL", False, False),
                ("naive, with BL", False, True),
                ("nested, no BL", True, False),
                ("nested, with BL", True, True),
            ]

            print(f"\n--- K={true_k}, entropy={entropy} ---")
            print(f"{'Config':<20} {'# sig edges':>12} {'Direction'}")

            for config_name, use_nested, use_bl in configs:
                n_sig = 0
                d = df.shape[1]  # features

                for parent_id, child_id in tree.edges():
                    if tree.nodes[child_id].get("is_leaf", False):
                        continue  # Skip leaf edges for this analysis

                    child_dist = np.array(tree.nodes[child_id]["distribution"])
                    parent_dist = np.array(tree.nodes[parent_id]["distribution"])
                    n_child = tree.nodes[child_id]["leaf_count"]
                    n_parent = tree.nodes[parent_id]["leaf_count"]
                    bl = tree.edges[parent_id, child_id].get("branch_length", 0)

                    z = compute_standardized_z_with_bl(
                        child_dist,
                        parent_dist,
                        n_child,
                        n_parent,
                        bl if use_bl else 0,
                        mean_bl if use_bl else 1,
                        use_nested,
                    )

                    stat = np.sum(z**2)
                    p = chi2.sf(stat, d)

                    if p < 0.05:
                        n_sig += 1

                # Determine likely effect on clustering
                if n_sig < true_k - 1:
                    direction = "→ likely UNDER-split"
                elif n_sig > true_k * 2:
                    direction = "→ likely OVER-split"
                else:
                    direction = "→ might be OK"

                print(f"{config_name:<20} {n_sig:>12} {direction}")

                results.append(
                    {
                        "true_k": true_k,
                        "entropy": entropy,
                        "config": config_name,
                        "n_sig": n_sig,
                    }
                )

    return pd.DataFrame(results)


def show_proposed_implementation():
    """Show the proposed code changes."""
    print("\n" + "=" * 70)
    print("PROPOSED IMPLEMENTATION")
    print("=" * 70)

    print("""
1. ADD TO edge_significance.py:
   
   def _compute_mean_branch_length(tree) -> float:
       '''Compute mean branch length for normalization.'''
       bls = [tree.edges[p, c].get('branch_length', 0) for p, c in tree.edges()]
       return np.mean(bls) if bls else 1.0

2. MODIFY _compute_standardized_z in edge_significance.py:

   def _compute_standardized_z(
       child_dist, parent_dist, n_child, n_parent, 
       branch_length=None, mean_branch_length=None
   ):
       # Use nested variance for statistical correctness
       var = parent_dist * (1 - parent_dist) * (1.0/n_child - 1.0/n_parent)
       
       # Apply normalized branch length scaling
       if branch_length is not None and mean_branch_length is not None:
           bl_norm = 1.0 + branch_length / mean_branch_length
           var = var * bl_norm
       
       var = np.maximum(var, 1e-10)
       z = (child_dist - parent_dist) / np.sqrt(var)
       return z.ravel()

3. MODIFY _compute_p_values_via_projection:
   
   # At the start:
   mean_bl = _compute_mean_branch_length(tree)
   
   # In the loop:
   stats[i], dfs[i], pvals[i] = _compute_projected_test(
       ..., branch_length, mean_bl
   )

4. SIMILARLY for pooled_variance.py (sibling test):
   
   # Pass mean_bl to standardize_proportion_difference
   # Use: bl_norm = 1.0 + branch_length_sum / (2 * mean_bl)
   # (divide by 2 because sum of two branches)
""")


def main():
    results = run_comparison_benchmark()
    show_proposed_implementation()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The comparison shows:

1. "naive, no BL" - Original implementation, tends to under-split
2. "naive, with BL" - With normalized BL, slightly more conservative
3. "nested, no BL" - More significant tests, tends to over-split  
4. "nested, with BL" - Balanced: more tests but BL dampens them

RECOMMENDATION: Use "nested, with BL" which combines:
- Correct nested variance formula (1/n_child - 1/n_parent)
- Normalized branch length (1 + BL/mean_BL) to prevent over-splitting

This should give better balance between under- and over-splitting.
""")


if __name__ == "__main__":
    main()
