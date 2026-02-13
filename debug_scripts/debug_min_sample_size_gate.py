"""
Debug script: Demonstrate impact of min sample size gate (n >= 10) on Case 18.

Case 18 (overlapping_binary_unbalanced_overlap_unbal_4c_small) produces 69 clusters
instead of the expected 4. The root cause: the algorithm splits down to individual
leaves where no statistical test has power.

This script:
1. Reads case_18_power_analysis.csv
2. Shows sample size and power distributions
3. Simulates the effect of a min_samples_per_child=10 gate
4. Re-runs Case 18 with the gate applied and compares ARI
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")


def analyze_csv():
    """Read and analyze the case 18 power analysis CSV."""
    df = pd.read_csv(
        "/Users/berksakalli/Projects/kl-te-cluster/debug_scripts/case_18_power_analysis.csv"
    )

    print("=" * 70)
    print("CASE 18 POWER ANALYSIS — SAMPLE SIZE GATE IMPACT")
    print("=" * 70)
    print()

    # --- 1. Basic stats ---
    print(f"Total split decisions: {len(df)}")
    print(f"CP significant nodes: {(df['cp_significant'] == True).sum()}")
    print(f"SB different nodes: {(df['sb_different'] == True).sum()}")
    print()

    # --- 2. Sample size distributions ---
    min_child = df[["n_left", "n_right"]].min(axis=1)
    print("=== min(n_left, n_right) Distribution ===")
    print(f"  Mean:   {min_child.mean():.1f}")
    print(f"  Median: {min_child.median():.0f}")
    print(f"  25th:   {min_child.quantile(0.25):.0f}")
    print(f"  75th:   {min_child.quantile(0.75):.0f}")
    print(f"  Max:    {min_child.max():.0f}")
    print()

    print("=== Value counts of min(n_left, n_right) ===")
    vc = min_child.value_counts().sort_index()
    for val, count in vc.items():
        pct = 100 * count / len(df)
        print(f"  n={int(val):3d}: {count:3d} splits ({pct:5.1f}%)")
    print()

    # --- 3. Power by sample size ---
    print("=== Power by min(n_left, n_right) Bucket ===")
    df["min_child"] = min_child
    buckets = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 999)]
    for lo, hi in buckets:
        mask = (df["min_child"] >= lo) & (df["min_child"] <= hi)
        subset = df[mask]
        if len(subset) == 0:
            continue
        mean_power = subset["power"].mean()
        sufficient = (subset["power"] >= 0.8).sum()
        print(
            f"  n=[{lo:3d},{hi:3d}]: {len(subset):3d} splits, "
            f"mean power={mean_power:.3f}, sufficient (≥80%)={sufficient}"
        )
    print()

    # --- 4. Simulate min sample size gate ---
    print("=" * 70)
    print("SIMULATED IMPACT OF MIN SAMPLE SIZE GATE")
    print("=" * 70)
    print()

    for threshold in [5, 10, 15, 20]:
        # A split is blocked if min(n_left, n_right) < threshold
        blocked = min_child < threshold
        allowed = ~blocked

        # Among blocked splits, how many were declared "different"?
        blocked_but_different = (blocked & (df["sb_different"] == True)).sum()

        # Among allowed splits, how many had sufficient power?
        allowed_sufficient = (allowed & (df["power"] >= 0.8)).sum()
        allowed_different = (allowed & (df["sb_different"] == True)).sum()

        # Underpowered "different" declarations that would be blocked
        underpowered_blocked = (
            blocked & (df["sb_different"] == True) & (df["power"] < 0.5)
        ).sum()

        print(f"  Threshold: min(n_left, n_right) >= {threshold}")
        print(f"    Splits blocked: {blocked.sum()} / {len(df)}")
        print(f"    Splits allowed: {allowed.sum()} / {len(df)}")
        print(f"    Blocked 'different' declarations: {blocked_but_different}")
        print(f"    Underpowered 'different' blocked: {underpowered_blocked}")
        print(f"    Allowed with power >= 80%: {allowed_sufficient}")
        print(f"    Allowed 'different': {allowed_different}")
        print()

    return df


def run_case_18_with_min_samples():
    """Re-run Case 18 with min_samples_per_child=10 and compare to baseline."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    from benchmarks.shared.cases import get_test_cases_by_category
    from benchmarks.shared.generators import generate_case_data
    from kl_clustering_analysis import config as kl_config
    from kl_clustering_analysis.tree.poset_tree import PosetTree

    # --- Generate case 18 data ---
    cases = get_test_cases_by_category("overlapping_binary_unbalanced")
    case_18 = [c for c in cases if "unbal_4c_small" in c.get("name", "")][0]
    df_data, true_labels, features, metadata = generate_case_data(case_18)

    print("=" * 70)
    print("CASE 18 RE-RUN WITH MIN SAMPLE SIZE GATE")
    print("=" * 70)
    print()
    print(f"Data shape: {df_data.shape}")
    print(f"True clusters: {len(set(true_labels))}")
    print()

    dist = pdist(df_data.values, metric=kl_config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=kl_config.TREE_LINKAGE_METHOD)

    # --- Baseline: no min sample size ---
    tree_baseline = PosetTree.from_linkage(Z, leaf_names=df_data.index.tolist())
    result_baseline = tree_baseline.decompose(
        leaf_data=df_data, alpha_local=0.05, sibling_alpha=0.05
    )

    labels_baseline = _extract_labels(result_baseline, df_data.index.tolist())
    ari_baseline = adjusted_rand_score(true_labels, labels_baseline)
    nmi_baseline = normalized_mutual_info_score(true_labels, labels_baseline)
    k_baseline = len(set(labels_baseline))

    print(
        f"Baseline (no gate):  K={k_baseline:3d}, ARI={ari_baseline:.4f}, NMI={nmi_baseline:.4f}"
    )

    # --- With min sample size gate ---
    for threshold in [5, 10, 15, 20]:
        tree_ms = PosetTree.from_linkage(Z, leaf_names=df_data.index.tolist())
        result_ms = tree_ms.decompose(
            leaf_data=df_data,
            alpha_local=0.05,
            sibling_alpha=0.05,
            min_samples_per_child=threshold,
        )

        labels_ms = _extract_labels(result_ms, df_data.index.tolist())
        ari_ms = adjusted_rand_score(true_labels, labels_ms)
        nmi_ms = normalized_mutual_info_score(true_labels, labels_ms)
        k_ms = len(set(labels_ms))

        improvement = "✓" if ari_ms > ari_baseline else "✗"
        print(
            f"min_samples={threshold:2d}:      K={k_ms:3d}, "
            f"ARI={ari_ms:.4f}, NMI={nmi_ms:.4f} {improvement}"
        )

    print()


def _extract_labels(result, sample_names):
    """Extract flat label array from decomposition result."""
    cluster_assignments = result.get("cluster_assignments", result)
    label_map = {}
    for cluster_id, info in cluster_assignments.items():
        for leaf in info["leaves"]:
            label_map[leaf] = cluster_id
    return [label_map.get(name, -1) for name in sample_names]


if __name__ == "__main__":
    # Part 1: Analyze CSV
    df_analysis = analyze_csv()

    print()

    # Part 2: Re-run with gate (requires the gate to be implemented in tree_decomposition.py)
    try:
        run_case_18_with_min_samples()
    except TypeError as e:
        if "min_samples_per_child" in str(e):
            print("=" * 70)
            print(
                "NOTE: min_samples_per_child not yet implemented in TreeDecomposition."
            )
            print("Implement it in config.py and tree_decomposition.py first.")
            print(f"Error: {e}")
            print("=" * 70)
        else:
            raise
