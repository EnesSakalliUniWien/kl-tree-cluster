"""Check signal quality of improved test cases."""

import sys

sys.path.insert(0, ".")
from tests.improved_test_cases import BINARY_TEST_CASES
from kl_clustering_analysis.benchmarking.generators.generate_random_feature_matrix import (
    generate_random_feature_matrix,
)
import numpy as np
from scipy.linalg import qr


def measure_signal(case):
    leaf_matrix_dict, cluster_assignments = generate_random_feature_matrix(
        n_rows=case["n_rows"],
        n_cols=case["n_cols"],
        n_clusters=case["n_clusters"],
        entropy_param=case["entropy_param"],
        feature_sparsity=case.get("feature_sparsity"),
        balanced_clusters=case.get("balanced_clusters", True),
        random_seed=case["seed"],
    )
    sample_names = list(leaf_matrix_dict.keys())
    X = np.array([leaf_matrix_dict[n] for n in sample_names])
    y = np.array([cluster_assignments[n] for n in sample_names])

    uc = np.unique(y)
    if len(uc) < 2:
        return 0
    X0, X1 = X[y == uc[0]], X[y == uc[1]]
    X_all = np.vstack([X0, X1])
    mean, std = X_all.mean(0), X_all.std(0)
    std[std < 1e-10] = 1.0
    X0_std, X1_std = (X0 - mean) / std, (X1 - mean) / std

    n_cols = X.shape[1]
    k = min(max(10, int(4 * np.log(len(X)))), n_cols)
    np.random.seed(42)
    G = np.random.standard_normal((k, n_cols))
    Q, _ = qr(G.T, mode="economic")
    R = Q.T
    z0, z1 = R @ X0_std.mean(0), R @ X1_std.mean(0)
    n_eff = len(X0) * len(X1) / (len(X0) + len(X1))
    T = n_eff * np.sum((z0 - z1) ** 2)
    return T / k


weak = 0
for c in BINARY_TEST_CASES:
    r = measure_signal(c)
    if r < 1.5:
        weak += 1
        print(f"{c['name']}: ratio={r:.2f}")

print(f"Weak cases: {weak}/{len(BINARY_TEST_CASES)}")
if weak == 0:
    print("✓ All cases have good signal!")
