#!/usr/bin/env python
"""Debug why gaussian cases return K=1.

Tests a few cases with different spectral_method settings to isolate
whether the effective_rank spectral method is causing the K=1 collapse.
"""
import warnings

warnings.filterwarnings("ignore")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree


def run_case(case_dict, spectral_method):
    """Run decomposition and return K, edge-sig count, sibling-diff count."""
    binary, labels, _, _ = generate_case_data(case_dict)

    Z = linkage(
        pdist(binary.values, metric=config.TREE_DISTANCE_METRIC),
        method=config.TREE_LINKAGE_METHOD,
    )
    tree = PosetTree.from_linkage(Z, leaf_names=binary.index.tolist())

    # Override spectral method
    old = config.SPECTRAL_METHOD
    config.SPECTRAL_METHOD = spectral_method
    try:
        res = tree.decompose(leaf_data=binary)
    finally:
        config.SPECTRAL_METHOD = old

    df = tree.stats_df
    n_edge_sig = int(df["Child_Parent_Divergence_Significant"].sum())
    n_sib_diff = int(df["Sibling_BH_Different"].sum())
    K = res.get("num_clusters", -1)
    true_K = case_dict.get("expected_k", "?")

    return K, true_K, n_edge_sig, n_sib_diff, binary.shape


def main():
    all_cases = {c["name"]: c for c in get_default_test_cases()}

    # Test a subset of cases that should be easy
    test_names = [
        "binary_perfect_2c",
        "binary_perfect_4c",
        "gaussian_clear_1",
        "gaussian_clear_2",
        "trivial_2c",
        "block_diag_4c",
    ]

    spectral_methods = ["effective_rank", None]

    print(
        f"{'Case':<25} {'Method':<16} {'Shape':<12} {'K':>3} {'TrueK':>5} {'EdgeSig':>7} {'SibDiff':>7}"
    )
    print("-" * 85)

    for name in test_names:
        if name not in all_cases:
            print(f"{name:<25} NOT FOUND")
            continue
        for sm in spectral_methods:
            label = str(sm) if sm else "JL-only"
            try:
                K, true_K, n_edge, n_sib, shape = run_case(all_cases[name], sm)
                print(
                    f"{name:<25} {label:<16} {str(shape):<12} {K:>3} {str(true_K):>5} {n_edge:>7} {n_sib:>7}"
                )
            except Exception as e:
                print(f"{name:<25} {label:<16} ERROR: {e}")


if __name__ == "__main__":
    main()
    main()
