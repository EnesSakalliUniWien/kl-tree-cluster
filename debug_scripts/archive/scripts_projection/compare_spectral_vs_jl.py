#!/usr/bin/env python3
"""Compare spectral vs JL projection on integration test cases.

Tests both the legacy JL-based dimension and the new per-node spectral
dimension on the two integration test cases, reporting K, ARI, and
assigned fraction for each.

Usage:
    python scripts/compare_spectral_vs_jl.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

import kl_clustering_analysis.config as config
from benchmarks.shared.generators import generate_random_feature_matrix
from benchmarks.shared.util.decomposition import _labels_from_decomposition
from kl_clustering_analysis.tree.io import tree_from_linkage


def run_pipeline(data_df, leaf_data, spectral_method=None, sibling_alpha=0.05):
    """Run the full pipeline with a given spectral method."""
    config.SPECTRAL_METHOD = spectral_method

    Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
    tree = tree_from_linkage(Z, leaf_names=data_df.index.tolist())
    decomposition = tree.decompose(
        leaf_data=data_df,
        alpha_local=config.ALPHA_LOCAL,
        sibling_alpha=sibling_alpha,
    )
    return decomposition


def evaluate(decomposition, data_df, true_clusters):
    """Compute ARI, assigned fraction, and K."""
    predicted = _labels_from_decomposition(decomposition, data_df.index.tolist())
    true_labels = [true_clusters[name] for name in data_df.index]

    assigned_mask = np.array(predicted) != -1
    assigned_frac = float(np.mean(assigned_mask))
    k = decomposition["num_clusters"]

    ari = 0.0
    if assigned_mask.any():
        ari = adjusted_rand_score(
            np.array(true_labels)[assigned_mask],
            np.array(predicted)[assigned_mask],
        )
    return k, ari, assigned_frac


# =====================================================================
# Test cases
# =====================================================================

CASES = [
    {
        "name": "balanced_4c (72×40, entropy=0.1)",
        "true_k": 4,
        "params": dict(
            n_rows=72,
            n_cols=40,
            entropy_param=0.1,
            n_clusters=4,
            random_seed=314,
            balanced_clusters=True,
        ),
    },
    {
        "name": "unbalanced_4c (96×36, entropy=0.25)",
        "true_k": 4,
        "params": dict(
            n_rows=96,
            n_cols=36,
            entropy_param=0.25,
            n_clusters=4,
            random_seed=2024,
            balanced_clusters=False,
        ),
    },
]

METHODS = [
    ("JL (legacy)", None),
    ("effective_rank", "effective_rank"),
    ("marchenko_pastur", "marchenko_pastur"),
    ("active_features", "active_features"),
]


def main():
    # =====================================================================
    # Part 1: Real data — feature_matrix.tsv
    # =====================================================================
    from pathlib import Path

    fpath = Path("feature_matrix.tsv")
    if fpath.exists():
        print("=" * 80)
        print("REAL DATA: feature_matrix.tsv")
        print("=" * 80)
        df = pd.read_csv(fpath, sep="\t", index_col=0)
        print(f"Shape: {df.shape[0]} x {df.shape[1]}, sparsity={1 - df.values.mean():.3f}\n")

        print(f"{'Method':<22s}  {'K':>4s}  {'Cluster sizes'}")
        print("-" * 70)

        for method_name, spectral_method in METHODS:
            config.SPECTRAL_METHOD = spectral_method
            Z = linkage(
                pdist(df.values, metric=config.TREE_DISTANCE_METRIC),
                method=config.TREE_LINKAGE_METHOD,
            )
            tree = tree_from_linkage(Z, leaf_names=df.index.tolist())
            dec = tree.decompose(
                leaf_data=df, alpha_local=config.ALPHA_LOCAL, sibling_alpha=config.SIBLING_ALPHA
            )
            k = dec["num_clusters"]
            ca = dec["cluster_assignments"]
            sizes = sorted([info["size"] for info in ca.values()], reverse=True)
            print(f"{method_name:<22s}  {k:>4d}  {sizes}")

        print()

    # =====================================================================
    # Part 2: Synthetic integration test cases
    # =====================================================================
    print("=" * 80)
    print("SYNTHETIC CASES")
    print("=" * 80)

    print(
        f"\n{'Case':<40s}  {'Method':<20s}  {'True K':>6s}  {'K':>4s}  "
        f"{'ARI':>6s}  {'Assigned':>8s}"
    )
    print("-" * 95)

    for case in CASES:
        data_dict, true_clusters = generate_random_feature_matrix(**case["params"])
        data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)

        for method_name, spectral_method in METHODS:
            decomposition = run_pipeline(data_df, data_df, spectral_method=spectral_method)
            k, ari, assigned_frac = evaluate(decomposition, data_df, true_clusters)

            print(
                f"{case['name']:<40s}  {method_name:<20s}  {case['true_k']:>6d}  "
                f"{k:>4d}  {ari:>6.3f}  {assigned_frac:>8.3f}"
            )

        print()

    # Reset config
    config.SPECTRAL_METHOD = "effective_rank"


if __name__ == "__main__":
    main()
