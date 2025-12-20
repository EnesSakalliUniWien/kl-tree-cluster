"""
Debug script for noise-adaptive alpha mechanism.

Investigates why the noise-adaptive α isn't changing benchmark results.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config
from kl_clustering_analysis.benchmarking.generators import (
    generate_random_feature_matrix,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    _estimate_local_noise,
)
from kl_clustering_analysis.config import compute_noise_adaptive_alpha


def run_debug():
    print("=" * 70)
    print("DEBUG: Noise-Adaptive Alpha Mechanism")
    print("=" * 70)
    print(f"USE_NOISE_ADAPTIVE_ALPHA: {config.USE_NOISE_ADAPTIVE_ALPHA}")
    print(f"Base α: {config.NOISE_ADAPTIVE_BASE_ALPHA}")
    print(f"Max α: {config.NOISE_ADAPTIVE_MAX_ALPHA}")
    print()

    # Test 1: Check noise estimation function directly
    print("=" * 70)
    print("TEST 1: Direct noise estimation (CORRECTED - per-sibling entropy)")
    print("=" * 70)

    # Clean split: siblings are DIFFERENT but each has clean signal
    left_clean = np.array([0.9, 0.1, 0.9, 0.1])  # left has features 0,2 ON
    right_clean = np.array([0.1, 0.9, 0.1, 0.9])  # right has features 1,3 ON
    noise_clean = _estimate_local_noise(left_clean, right_clean, 100, 100)
    alpha_clean = compute_noise_adaptive_alpha(noise_clean)
    print(f"Clean split (siblings different but each clean):")
    print(f"  Left:  {left_clean}")
    print(f"  Right: {right_clean}")
    print(f"  Noise: {noise_clean:.3f} → α={alpha_clean:.3f}")

    # Noisy siblings: each sibling has features near 0.5
    left_noisy = np.array([0.45, 0.55, 0.48, 0.52])
    right_noisy = np.array([0.52, 0.48, 0.50, 0.50])
    noise_noisy = _estimate_local_noise(left_noisy, right_noisy, 100, 100)
    alpha_noisy = compute_noise_adaptive_alpha(noise_noisy)
    print(f"\nNoisy siblings (each sibling has features near 0.5):")
    print(f"  Left:  {left_noisy}")
    print(f"  Right: {right_noisy}")
    print(f"  Noise: {noise_noisy:.3f} → α={alpha_noisy:.3f}")
    print()

    # Test 2: Generate data and check what noise levels appear
    print("=" * 70)
    print("TEST 2: Noise levels in actual clustering")
    print("=" * 70)

    test_configs = [
        {"entropy": 0.02, "name": "Low noise (0.02)"},
        {"entropy": 0.10, "name": "Moderate noise (0.10)"},
        {"entropy": 0.25, "name": "High noise (0.25)"},
    ]

    for tc in test_configs:
        print(f"\n--- {tc['name']} ---")
        n_rows, n_cols, n_clusters = 96, 300, 4

        data_dict, cluster_assignments = generate_random_feature_matrix(
            n_rows=n_rows,
            n_cols=n_cols,
            entropy_param=tc["entropy"],
            n_clusters=n_clusters,
            random_seed=42,
            balanced_clusters=False,
        )

        original_names = list(data_dict.keys())
        matrix = np.array([data_dict[name] for name in original_names], dtype=int)
        data_df = pd.DataFrame(
            matrix, index=original_names, columns=[f"F{j}" for j in range(n_cols)]
        )

        # Build tree
        Z = linkage(
            pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC),
            method=config.TREE_LINKAGE_METHOD,
        )
        tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())

        # Decompose
        decomp = tree.decompose(
            leaf_data=data_df,
            alpha_local=config.ALPHA_LOCAL,
            sibling_alpha=config.SIBLING_ALPHA,
        )

        stats = tree.stats_df
        print(f"  Clusters found: {decomp['num_clusters']} (true: {n_clusters})")

        if "Sibling_Local_Noise" in stats.columns:
            noise_vals = stats["Sibling_Local_Noise"].dropna()
            print(f"  Local noise computed: {len(noise_vals)} nodes")
            if len(noise_vals) > 0:
                print(
                    f"  Noise range: [{noise_vals.min():.3f}, {noise_vals.max():.3f}]"
                )
                print(f"  Mean noise: {noise_vals.mean():.3f}")

                # Show the adjustment factors
                alphas = noise_vals.apply(compute_noise_adaptive_alpha)
                factors = alphas / config.SIBLING_ALPHA
                print(
                    f"  α adjustment factors: [{factors.min():.3f}, {factors.max():.3f}]"
                )
        else:
            print("  ⚠ Sibling_Local_Noise column NOT found!")
            print(f"  Available columns: {list(stats.columns)}")

    # Test 3: Manual verification of p-value adjustment
    print()
    print("=" * 70)
    print("TEST 3: P-value adjustment simulation")
    print("=" * 70)

    base_alpha = 0.05
    p_values = np.array([0.01, 0.03, 0.05, 0.08, 0.10, 0.15])
    noise_levels = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 0.4])

    print("\nOriginal p-values vs adjusted (with noise-adaptive α):")
    print("-" * 60)
    print(f"{'p_orig':>10} {'noise':>8} {'α_local':>8} {'factor':>8} {'p_adj':>10}")
    print("-" * 60)

    for p, noise in zip(p_values, noise_levels):
        alpha_local = compute_noise_adaptive_alpha(noise)
        factor = alpha_local / base_alpha
        p_adj = min(1.0, p * factor)
        print(
            f"{p:>10.3f} {noise:>8.2f} {alpha_local:>8.3f} {factor:>8.3f} {p_adj:>10.3f}"
        )

    print()
    print(
        "Logic: high noise → higher α → larger factor → p inflated → harder to reject → more merging"
    )


if __name__ == "__main__":
    run_debug()
