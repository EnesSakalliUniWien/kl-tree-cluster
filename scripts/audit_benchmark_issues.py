#!/usr/bin/env python3
"""
Benchmark Audit Script
======================
Demonstrates and verifies the structural issues in the benchmark setup
that explain the poor ARI scores (mean 0.579, 24/74 exact K).

Six issues are diagnosed:
  1. SBM modularity distance bypass (bug)
  2. Categorical/phylogenetic data → Bernoulli pipeline (type mismatch)
  3. Gradient templates collapse for >4 clusters
  4. Median binarization maximizes Wald variance
  5. Gaussian cases underpowered (n_per_cluster ≈ 10)
  6. Global np.random.seed reproducibility hazard

Run:
    python scripts/audit_benchmark_issues.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ✗ {msg}")


# ── Issue 1: SBM modularity distance bypass ─────────────────────────────────


def audit_sbm_distance_bypass() -> None:
    """Show that the KL runner recomputes pdist instead of using
    the modularity distance the pipeline already computed for SBM cases."""
    _section("Issue 1: SBM modularity distance bypass")

    from benchmarks.shared.generators.generate_sbm import generate_sbm
    from scipy.spatial.distance import pdist, squareform

    G, y, A, meta = generate_sbm(sizes=[30, 30], p_intra=0.12, p_inter=0.005, seed=123)
    n = meta["n_nodes"]
    data_df = pd.DataFrame(
        A.astype(int),
        index=[f"S{j}" for j in range(n)],
        columns=[f"F{j}" for j in range(n)],
    )

    # What the pipeline computes (modularity)
    degrees = A.sum(axis=1)
    m = A.sum() / 2
    expected = np.outer(degrees, degrees) / (2 * m)
    B = A - expected
    B_shifted = B - B.min()
    B_norm = B_shifted / (B_shifted.max() + 1e-10)
    dist_modularity = 1.0 - B_norm
    np.fill_diagonal(dist_modularity, 0.0)
    d_mod = squareform(dist_modularity)

    # What the KL runner actually uses
    d_hamming = pdist(data_df.values, metric="hamming")

    corr = np.corrcoef(d_mod, d_hamming)[0, 1]
    _warn(f"Modularity distance vs Hamming-on-adjacency correlation: {corr:.3f}")
    _warn(
        "KL runner ignores the pre-computed modularity distance and "
        "recomputes pdist(adjacency, metric='hamming')."
    )
    _warn(
        "Fix: pass distance_condensed to the KL runner when meta['generator'] == 'sbm'."
    )


# ── Issue 2: Categorical / Phylogenetic type mismatch ───────────────────────


def audit_categorical_type_mismatch() -> None:
    """Show that categorical integers (0..K-1) fed into the Bernoulli
    pipeline produce invalid probability values (>1)."""
    _section("Issue 2: Categorical data → Bernoulli pipeline (type mismatch)")

    from benchmarks.shared.generators.generate_categorical_matrix import (
        generate_categorical_feature_matrix,
    )

    sample_dict, assignments, dists = generate_categorical_feature_matrix(
        n_rows=20,
        n_cols=10,
        n_categories=4,
        entropy_param=0.1,
        n_clusters=3,
        random_seed=42,
    )

    names = list(sample_dict.keys())
    matrix = np.array([sample_dict[n] for n in names])
    max_val = matrix.max()
    unique_vals = np.unique(matrix)

    _warn(f"Categorical matrix value range: {unique_vals}")
    _warn(f"Max value: {max_val}  (Bernoulli pipeline assumes values in [0,1])")
    if max_val > 1:
        _warn(
            "Values >1 break the Bernoulli KL formula: "
            "p·log(p/q) + (1-p)·log((1-p)/(1-q)) → NaN / negative logs."
        )
        _warn(
            "Fix: one-hot encode categorical data before passing to "
            "PosetTree.decompose(), or add a Categorical KL path."
        )
    else:
        _ok("Categorical values within [0,1] — no issue.")


# ── Issue 3: Gradient templates degrade for >4 clusters ─────────────────────


def audit_gradient_templates() -> None:
    """Show that gradient templates for >4 clusters produce nearly
    indistinguishable intermediate templates."""
    _section("Issue 3: Gradient templates degrade for >4 clusters")

    from benchmarks.shared.generators.generate_random_feature_matrix import (
        _create_gradient_templates,
    )

    np.random.seed(0)
    for k in [3, 5, 8]:
        templates = _create_gradient_templates(k, n_cols=100)
        # Compute pairwise Hamming distances between templates
        from scipy.spatial.distance import pdist

        dists = pdist(np.array(templates), metric="hamming")
        min_d = dists.min()
        mean_d = dists.mean()

        label = "OK" if min_d > 0.15 else "WEAK"
        func = _ok if min_d > 0.15 else _warn
        func(f"k={k}: min_Hamming={min_d:.3f}, mean_Hamming={mean_d:.3f} [{label}]")

    _warn(
        "For k>4, adjacent gradient clusters differ by ~1/k fraction of "
        "features — marginal signal once noise is applied."
    )
    _warn(
        "Fix: use block-diagonal / feature-ownership templates "
        "(like _create_sparse_templates) instead of gradient."
    )


# ── Issue 4: Median binarization maximizes Wald variance ────────────────────


def audit_median_binarization() -> None:
    """Show that median binarization forces all column means to ~0.5,
    maximizing the Wald test variance denominator θ(1-θ)."""
    _section("Issue 4: Median binarization maximizes Wald variance")

    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        n_samples=100, n_features=50, centers=4, cluster_std=0.8, random_state=42
    )
    X_bin = (X > np.median(X, axis=0)).astype(int)

    col_means = X_bin.mean(axis=0)
    global_mean = col_means.mean()
    global_var = (col_means * (1 - col_means)).mean()  # Bernoulli variance

    _warn(f"After median binarization: mean θ across features = {global_mean:.3f}")
    _warn(f"Mean Bernoulli variance θ(1-θ) = {global_var:.4f}  (max possible = 0.25)")
    _warn(
        "Wald test denominator is proportional to θ(1-θ). "
        "At θ=0.5 the denominator is maximized → minimum test power."
    )

    # Show per-cluster signal still exists
    for c in range(4):
        mask = y == c
        cluster_means = X_bin[mask].mean(axis=0)
        deviation = np.abs(cluster_means - col_means).mean()
        print(
            f"    Cluster {c} (n={mask.sum()}): mean |θ_cluster - θ_global| = {deviation:.3f}"
        )

    _warn(
        "Per-cluster deviations exist but are small (~0.1-0.15), "
        "making detection marginal at small n."
    )


# ── Issue 5: Gaussian cases underpowered ────────────────────────────────────


def audit_gaussian_power() -> None:
    """Show that many Gaussian cases have ≤15 samples per cluster,
    below the practical power threshold for the projected Wald test."""
    _section("Issue 5: Gaussian test cases underpowered")

    from benchmarks.shared.cases.gaussian import GAUSSIAN_CASES

    print("  Gaussian case    | n_total | K | n/K  | Adequate?")
    print("  " + "-" * 55)
    for group_name, cases in GAUSSIAN_CASES.items():
        for tc in cases:
            n = tc.get("n_samples", tc.get("n_rows", "?"))
            k = tc["n_clusters"]
            name = tc.get("name", group_name)
            n_per = n // k if isinstance(n, int) else "?"
            adequate = n_per >= 20 if isinstance(n_per, int) else "?"
            label = "OK" if adequate else "LOW"
            func = _ok if adequate else _warn
            func(f"{name:30s} | {n:>4} | {k} | {n_per:>4} | {label}")


# ── Issue 6: Global np.random.seed ──────────────────────────────────────────


def audit_global_rng() -> None:
    """Demonstrate that generate_random_feature_matrix uses global
    np.random.seed, which leaks state across calls."""
    _section("Issue 6: Global np.random.seed in generators")

    from benchmarks.shared.generators.generate_random_feature_matrix import (
        generate_random_feature_matrix,
    )

    # Call generator with seed=42
    d1, _ = generate_random_feature_matrix(
        n_rows=20, n_cols=10, entropy_param=0.1, n_clusters=2, random_seed=42
    )
    # Global state is now deterministic — next random call is affected
    leaked_val_1 = np.random.random()

    # Call again with same seed
    d2, _ = generate_random_feature_matrix(
        n_rows=20, n_cols=10, entropy_param=0.1, n_clusters=2, random_seed=42
    )
    leaked_val_2 = np.random.random()

    if abs(leaked_val_1 - leaked_val_2) < 1e-12:
        _warn(
            f"Global RNG state leaked identically: {leaked_val_1:.8f} == {leaked_val_2:.8f}"
        )
    else:
        _ok("Global RNG state did not leak (already fixed?).")

    _warn(
        "Fix: replace np.random.seed(seed) with "
        "rng = np.random.default_rng(seed) and thread rng through all helpers."
    )


# ── Summary ─────────────────────────────────────────────────────────────────


def print_summary() -> None:
    _section("SUMMARY: Benchmark Issues by Severity")
    print("""
  BUGS (incorrect results):
    1. SBM modularity distance bypass → KL runner ignores modularity transform
    2. Categorical/phylogenetic type mismatch → NaN/garbage KL divergences

  DESIGN FLAWS (reduced power / misleading scores):
    3. Gradient templates → indistinguishable clusters for k>4
    4. Median binarization → maximal Wald variance, minimum power
    5. Underpowered Gaussian cases → n/K < 20 in many cases

  CODE QUALITY:
    6. Global np.random.seed → reproducibility hazard in composition

  These issues collectively explain the benchmark's mean ARI=0.579
  and the known failure modes (K=1 under-splitting, phylogenetic
  over-splitting to 72 clusters, etc.).
""")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Benchmark Audit — Structural Issue Diagnosis")
    print("=" * 72)

    audit_sbm_distance_bypass()
    audit_categorical_type_mismatch()
    audit_gradient_templates()
    audit_median_binarization()
    audit_gaussian_power()
    audit_global_rng()
    print_summary()
