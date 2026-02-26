#!/usr/bin/env python3
"""Verify benchmark suite changes: case counts, generator features, and data integrity.

Run from the project root:
    python scripts/verify_benchmark_changes.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    return condition


# --------------------------------------------------------------------------
# 1. Case configuration integrity
# --------------------------------------------------------------------------
def verify_case_configs() -> list[bool]:
    header("1. Case Configuration Integrity")
    results: list[bool] = []

    from benchmarks.shared.cases import get_default_test_cases
    from benchmarks.shared.cases.binary import BINARY_CASES
    from benchmarks.shared.cases.gaussian import GAUSSIAN_CASES
    from benchmarks.shared.cases.overlapping import OVERLAPPING_CASES

    # --- Binary ---
    binary_count = sum(len(v) for v in BINARY_CASES.values())
    results.append(check("Binary case count", binary_count == 28, f"got {binary_count}"))

    binary_categories = set(BINARY_CASES.keys())
    expected_binary_cats = {
        "binary_balanced_low_noise",
        "binary_sparse_features",
        "binary_null",
        "binary_multiscale",
        "binary_noise_features",
    }
    results.append(
        check(
            "Binary subcategories (new)",
            expected_binary_cats.issubset(binary_categories),
            (
                f"missing: {expected_binary_cats - binary_categories}"
                if expected_binary_cats - binary_categories
                else f"all present ({len(binary_categories)} total)"
            ),
        )
    )

    # All binary cases have feature_sparsity
    all_have_sparsity = all(
        "feature_sparsity" in case for cases in BINARY_CASES.values() for case in cases
    )
    results.append(check("All binary cases have feature_sparsity", all_have_sparsity))

    # Noise-feature cases have noise_features > 0
    noise_feat_cases = [
        c for c in BINARY_CASES.get("binary_noise_features", []) if c.get("noise_features", 0) > 0
    ]
    results.append(
        check(
            "Noise-feature cases present",
            len(noise_feat_cases) == 3,
            f"got {len(noise_feat_cases)}",
        )
    )

    # Null cases have K=1
    null_cases = BINARY_CASES.get("binary_null", [])
    all_null_k1 = all(c["n_clusters"] == 1 for c in null_cases)
    results.append(check("Binary null cases K=1", all_null_k1 and len(null_cases) == 2))

    # --- Gaussian ---
    gauss_count = sum(len(v) for v in GAUSSIAN_CASES.values())
    results.append(check("Gaussian case count", gauss_count == 12, f"got {gauss_count}"))

    gauss_categories = set(GAUSSIAN_CASES.keys())
    results.append(
        check(
            "gaussian_clear/gaussian_mixed removed",
            "gaussian_clear" not in gauss_categories and "gaussian_mixed" not in gauss_categories,
        )
    )
    results.append(check("gaussian_null present", "gaussian_null" in gauss_categories))

    gauss_null = GAUSSIAN_CASES.get("gaussian_null", [])
    all_gauss_null_k1 = all(c["n_clusters"] == 1 for c in gauss_null)
    results.append(check("Gaussian null K=1", all_gauss_null_k1 and len(gauss_null) == 2))

    # --- Overlapping ---
    overlap_count = sum(len(v) for v in OVERLAPPING_CASES.values())
    results.append(check("Overlapping case count", overlap_count == 29, f"got {overlap_count}"))

    # All binary overlap cases have feature_sparsity
    binary_overlap_cats = [k for k in OVERLAPPING_CASES if k.startswith("overlapping_binary")]
    binary_overlap_cases = [c for cat in binary_overlap_cats for c in OVERLAPPING_CASES[cat]]
    all_overlap_sparsity = all("feature_sparsity" in c for c in binary_overlap_cases)
    results.append(
        check(
            "All binary overlap cases have feature_sparsity",
            all_overlap_sparsity,
            f"{len(binary_overlap_cases)} binary overlap cases checked",
        )
    )

    # --- Totals ---
    all_cases = get_default_test_cases()
    total = len(all_cases)
    results.append(check("Total case count", total >= 95, f"got {total}"))

    # Unique names
    names = [c["name"] for c in all_cases]
    unique_names = len(set(names))
    results.append(
        check("All case names unique", unique_names == total, f"{unique_names}/{total} unique")
    )

    return results


# --------------------------------------------------------------------------
# 2. Generator RNG isolation
# --------------------------------------------------------------------------
def verify_rng_isolation() -> list[bool]:
    header("2. Generator RNG Isolation (no global state)")
    results: list[bool] = []
    import numpy as np

    from benchmarks.shared.generators.generate_random_feature_matrix import (
        generate_random_feature_matrix,
    )

    # Two calls with the same seed must produce identical output
    d1, c1 = generate_random_feature_matrix(
        n_rows=30,
        n_cols=20,
        entropy_param=0.2,
        n_clusters=3,
        random_seed=999,
        feature_sparsity=0.05,
    )
    d2, c2 = generate_random_feature_matrix(
        n_rows=30,
        n_cols=20,
        entropy_param=0.2,
        n_clusters=3,
        random_seed=999,
        feature_sparsity=0.05,
    )
    same_output = d1 == d2 and c1 == c2
    results.append(check("Same seed → identical output", same_output))

    # A call should NOT pollute global np.random state
    np.random.seed(42)
    before = np.random.random()
    np.random.seed(42)
    _ = generate_random_feature_matrix(
        n_rows=20,
        n_cols=20,
        entropy_param=0.1,
        n_clusters=2,
        random_seed=123,
        feature_sparsity=0.05,
    )
    after = np.random.random()
    results.append(check("Generator does not pollute global RNG", before == after))

    return results


# --------------------------------------------------------------------------
# 3. Noise-feature support
# --------------------------------------------------------------------------
def verify_noise_features() -> list[bool]:
    header("3. Noise-Feature Support")
    results: list[bool] = []

    from benchmarks.shared.generators.generate_random_feature_matrix import (
        generate_random_feature_matrix,
    )

    n_cols, noise = 40, 60
    d, c = generate_random_feature_matrix(
        n_rows=30,
        n_cols=n_cols,
        entropy_param=0.2,
        n_clusters=3,
        random_seed=42,
        feature_sparsity=0.05,
        noise_features=noise,
    )
    actual_width = len(next(iter(d.values())))
    results.append(
        check(
            "Matrix width = n_cols + noise_features",
            actual_width == n_cols + noise,
            f"expected {n_cols + noise}, got {actual_width}",
        )
    )

    # Noise columns (indices n_cols .. n_cols+noise) should be ~Bernoulli(0.5)
    import numpy as np

    mat = np.array(list(d.values()))
    noise_means = mat[:, n_cols:].mean(axis=0)
    mean_of_means = float(noise_means.mean())
    results.append(
        check(
            "Noise columns ≈ Bernoulli(0.5)",
            0.3 < mean_of_means < 0.7,
            f"mean={mean_of_means:.3f}",
        )
    )

    return results


# --------------------------------------------------------------------------
# 4. End-to-end case generation smoke test
# --------------------------------------------------------------------------
def verify_case_generation() -> list[bool]:
    header("4. End-to-End Case Generation (smoke test)")
    results: list[bool] = []

    from benchmarks.shared.cases import get_default_test_cases
    from benchmarks.shared.generators.generate_case_data import generate_case_data

    test_names = [
        "binary_balanced_low_noise",  # standard binary
        "binary_null_small",  # K=1 null
        "binary_noise_feat_50i_200n",  # noise features
        "gauss_clear_small",  # Gaussian improved
        "gauss_null_small",  # Gaussian null
    ]

    all_cases = {c["name"]: c for c in get_default_test_cases()}

    for name in test_names:
        case = all_cases.get(name)
        if case is None:
            results.append(check(f"Generate '{name}'", False, "case not found"))
            continue
        try:
            data_df, true_labels, _, metadata = generate_case_data(case)
            ok = (
                data_df.shape[0] > 0
                and len(true_labels) == data_df.shape[0]
                and metadata.get("n_clusters") is not None
            )
            detail = f"shape={data_df.shape}, K={metadata.get('n_clusters')}"
            results.append(check(f"Generate '{name}'", ok, detail))
        except Exception as exc:
            results.append(check(f"Generate '{name}'", False, f"ERROR: {exc}"))
            traceback.print_exc()

    return results


# --------------------------------------------------------------------------
# 5. Sparse template quality check
# --------------------------------------------------------------------------
def verify_sparse_templates() -> list[bool]:
    header("5. Sparse Template Quality")
    results: list[bool] = []
    import numpy as np

    from benchmarks.shared.generators.generate_random_feature_matrix import (
        generate_random_feature_matrix,
    )

    # K=8 with sparse templates — each cluster should own ~n_cols/K features
    n_cols, K = 160, 8
    d, c = generate_random_feature_matrix(
        n_rows=80,
        n_cols=n_cols,
        entropy_param=0.05,
        n_clusters=K,
        random_seed=42,
        feature_sparsity=0.05,
    )

    mat = np.array(list(d.values()))
    labels = np.array(list(c.values()))

    # Compute per-cluster mean for each feature
    cluster_means = np.zeros((K, n_cols))
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            cluster_means[k] = mat[mask].mean(axis=0)

    # For each feature the "owning" cluster should have highest mean
    owner = cluster_means.argmax(axis=0)
    # Each cluster should own roughly n_cols/K features (±50%)
    expected = n_cols // K
    counts = [(owner == k).sum() for k in range(K)]
    all_reasonable = all(expected * 0.3 <= cnt <= expected * 2.0 for cnt in counts)
    results.append(
        check(
            f"K={K}: each cluster owns features",
            all_reasonable,
            f"counts={counts}, expected≈{expected}",
        )
    )

    # Inter-cluster distance: all pairs should be reasonably separated
    from scipy.spatial.distance import pdist

    dists = pdist(cluster_means, metric="hamming")
    min_dist = float(dists.min())
    results.append(
        check(
            "Min pairwise Hamming > 0.05",
            min_dist > 0.05,
            f"min={min_dist:.4f}",
        )
    )

    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    print("Benchmark Suite Verification Script")
    print(f"Project root: {ROOT}")

    all_results: list[bool] = []

    all_results.extend(verify_case_configs())
    all_results.extend(verify_rng_isolation())
    all_results.extend(verify_noise_features())
    all_results.extend(verify_case_generation())
    all_results.extend(verify_sparse_templates())

    passed = sum(all_results)
    total = len(all_results)
    failed = total - passed

    header("Summary")
    print(f"  {passed}/{total} checks passed")
    if failed:
        print(f"  {failed} FAILED")
        return 1
    else:
        print("  All checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
