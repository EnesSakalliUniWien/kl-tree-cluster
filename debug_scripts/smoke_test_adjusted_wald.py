"""Smoke test: cousin-adjusted Wald vs original Wald on a few benchmark cases.

Compares cluster count and ARI for both methods.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

# ── Test cases ──────────────────────────────────────────────────────────────


def make_sparse_binary(n_per=50, p=200, k=4, seed=42):
    """Strong block-diagonal signal — each cluster owns exclusive features."""
    rng = np.random.default_rng(seed)
    block_size = p // k
    rows = []
    labels = []
    for c in range(k):
        X_cluster = rng.binomial(1, 0.1, size=(n_per, p))
        # Each cluster lights up its own block at high probability
        X_cluster[:, c * block_size : (c + 1) * block_size] = rng.binomial(
            1, 0.9, size=(n_per, block_size)
        )
        rows.append(X_cluster)
        labels.extend([c] * n_per)
    X = np.vstack(rows)
    df = pd.DataFrame(
        X, index=[f"S{i}" for i in range(len(labels))], columns=[f"F{j}" for j in range(p)]
    )
    return df, np.array(labels)


def make_gaussian(n_per=80, p=50, k=3, sep=4.0, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    block = p // k
    for c in range(k):
        center = np.zeros(p)
        center[c * block : (c + 1) * block] = sep
        rows.append(rng.normal(center, 1.0, size=(n_per, p)))
        labels.extend([c] * n_per)
    X = np.vstack(rows)
    X_bin = (X > np.median(X, axis=0)).astype(int)
    df = pd.DataFrame(
        X_bin, index=[f"S{i}" for i in range(len(labels))], columns=[f"F{j}" for j in range(p)]
    )
    return df, np.array(labels)


def make_null(n=100, p=50, seed=42):
    """Single-cluster null data — should NOT be split."""
    rng = np.random.default_rng(seed)
    X = rng.binomial(1, 0.5, size=(n, p))
    df = pd.DataFrame(X, index=[f"S{i}" for i in range(n)], columns=[f"F{j}" for j in range(p)])
    return df, np.zeros(n, dtype=int)


# Use the benchmark generators for proper cases
try:
    sys.path.insert(0, ".")
    from benchmarks.shared.cases import get_default_test_cases
    from benchmarks.shared.generators import generate_case_data

    USE_BENCHMARK = True
except ImportError:
    USE_BENCHMARK = False

CASES_MANUAL = [
    ("null_k1", make_null, {}),
    ("sparse_k2", make_sparse_binary, {"k": 2}),
    ("sparse_k4", make_sparse_binary, {"k": 4}),
    ("sparse_k8", make_sparse_binary, {"k": 8, "p": 400}),
    ("gaussian_k3", make_gaussian, {"k": 3}),
    ("gaussian_k5", make_gaussian, {"k": 5, "p": 50}),
]


# ── Runner ──────────────────────────────────────────────────────────────────


def run_case(name, data, true_labels, method):
    config.SIBLING_TEST_METHOD = method
    Z = linkage(
        pdist(data.values, metric=config.TREE_DISTANCE_METRIC), method=config.TREE_LINKAGE_METHOD
    )
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())
    result = tree.decompose(leaf_data=data)
    clusters = result["cluster_assignments"]
    pred = np.array([clusters.get(s, -1) for s in data.index])
    k_found = len(set(pred))
    ari = adjusted_rand_score(true_labels, pred) if len(set(true_labels)) > 1 else 0.0

    # Extract calibration info from tree.stats_df
    audit = {}
    if tree.stats_df is not None and hasattr(tree.stats_df, "attrs"):
        audit = tree.stats_df.attrs.get("sibling_divergence_audit", {})

    return k_found, ari, audit


# ── Main ────────────────────────────────────────────────────────────────────


def _print_row(name, true_k, k_w, ari_w, k_a, ari_a, audit):
    cal_method = audit.get("calibration_method", "?")
    c_hat = audit.get("global_c_hat", float("nan"))
    n_cal = audit.get("calibration_n", 0)

    ari_w_s = f"{ari_w:.3f}"
    ari_a_s = f"{ari_a:.3f}"
    c_hat_s = f"{c_hat:.2f}" if np.isfinite(c_hat) else "N/A"

    flag = ""
    if ari_a > ari_w + 0.01:
        flag = " ◀ adj wins"
    elif ari_w > ari_a + 0.01:
        flag = " ◀ wald wins"

    print(
        f"{name:<30} {true_k:>6}  |  {k_w:>6} {ari_w_s:>7}  |  "
        f"{k_a:>5} {ari_a_s:>7}  |  {cal_method:>12} {c_hat_s:>6} {n_cal:>5}{flag}"
    )


def main():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    header = f"{'Case':<30} {'True K':>6}  |  {'Wald K':>6} {'ARI':>7}  |  {'Adj K':>5} {'ARI':>7}  |  {'Cal Method':>12} {'ĉ':>6} {'N cal':>5}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # 1. Manual cases with strong signal
    for name, gen_fn, kwargs in CASES_MANUAL:
        data, labels = gen_fn(**kwargs)
        true_k = len(set(labels))

        k_w, ari_w, _ = run_case(name, data, labels, "wald")
        k_a, ari_a, audit = run_case(name, data, labels, "cousin_adjusted_wald")

        _print_row(name, true_k, k_w, ari_w, k_a, ari_a, audit)

    # 2. Benchmark cases (if available)
    if USE_BENCHMARK:
        print("-" * len(header))
        all_cases = get_default_test_cases()
        # Pick a subset of interesting cases
        pick_names = [
            "sparse_features_72x72",
            "sparse_features_100x500",
            "binary_perfect_2c",
            "binary_perfect_4c",
            "binary_perfect_8c",
            "binary_low_noise_4c",
            "binary_moderate_4c",
            "gauss_clear_small",
            "gauss_clear_medium",
            "gauss_moderate_3c",
            "gauss_noisy_3c",
        ]
        for tc in all_cases:
            if tc["name"] not in pick_names:
                continue
            try:
                data_df, true_labels, _, _ = generate_case_data(tc)
                true_k = tc.get("n_clusters", len(set(true_labels)))

                k_w, ari_w, _ = run_case(tc["name"], data_df, true_labels, "wald")
                k_a, ari_a, audit = run_case(
                    tc["name"], data_df, true_labels, "cousin_adjusted_wald"
                )

                _print_row(tc["name"], true_k, k_w, ari_w, k_a, ari_a, audit)
            except Exception as e:
                print(f"{tc['name']:<30} ERROR: {e}")

    print("=" * len(header))


if __name__ == "__main__":
    main()
