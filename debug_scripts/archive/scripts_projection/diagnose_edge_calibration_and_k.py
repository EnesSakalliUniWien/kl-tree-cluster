#!/usr/bin/env python3
"""Diagnose edge significance calibration and projection dimension selection.

Three key questions:
1. CALIBRATION — Is the edge test χ²(k) well-calibrated? What is the empirical
   inflation factor at leaf edges (null-like)? Can we calibrate Gate 2 the
   same way cousin_adjusted_wald calibrates Gate 3?
2. PROJECTION DIMENSION — The JL lemma uses n_child to pick k. But that's
   wrong: we're testing ONE z-vector, not preserving pairwise distances of
   n points. What function of (n_descendants, d) gives the right k?
3. EPSILON — Currently eps=0.3 is hardcoded. Can we infer it from data?

Candidate k-selection strategies:
  (a) JL(n_child, eps=0.3)        — current baseline, misapplied
  (b) k = d                       — full dimension (no projection)
  (c) Effective rank of data      — exp(Shannon entropy of eigenvalue spectrum)
  (d) Marchenko-Pastur threshold  — eigenvalues above MP upper bound
  (e) Parallel analysis           — eigenvalues above random reference
  (f) Signal-adaptive             — number of z_j exceeding Bonferroni threshold

Usage:
    python scripts/diagnose_edge_calibration_and_k.py [--input feature_matrix.tsv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from kl_clustering_analysis import config
from kl_clustering_analysis.core_utils.data_utils import extract_node_sample_size
from kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils import (
    compute_mean_branch_length,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance import (
    _compute_standardized_z,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.projection.random_projection import (
    generate_projection_matrix,
)
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.tree.io import tree_from_linkage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, default=Path("feature_matrix.tsv"))
    return p.parse_args()


# =====================================================================
# K-selection strategies
# =====================================================================


def k_jl(n_child: int, d: int, eps: float = 0.3) -> int:
    """Current JL-based dimension (baseline)."""
    k = johnson_lindenstrauss_min_dim(n_samples=max(n_child, 1), eps=eps)
    return int(min(max(k, config.PROJECTION_MIN_K), d))


def k_full(n_child: int, d: int) -> int:
    """Full dimension — no projection."""
    return d


def effective_rank(cov_eigenvalues: np.ndarray) -> int:
    """Effective rank via Shannon entropy of the eigenvalue spectrum.

    erank(Σ) = exp( -Σ p_i log p_i ) where p_i = λ_i / Σλ_j

    This gives a continuous dimensionality measure (Roy & Vetterli, 2007).
    If all eigenvalues are equal, erank = d. If one dominates, erank ≈ 1.
    """
    eigenvalues = np.maximum(cov_eigenvalues, 0)  # numerical guard
    total = np.sum(eigenvalues)
    if total <= 0:
        return 1
    p = eigenvalues / total
    p = p[p > 0]  # remove zeros for log
    entropy = -np.sum(p * np.log(p))
    return max(1, int(np.round(np.exp(entropy))))


def marchenko_pastur_k(eigenvalues: np.ndarray, n: int, d: int) -> int:
    """Count eigenvalues above the Marchenko-Pastur upper bound.

    For an n×d random matrix with i.i.d. entries ~ N(0, σ²/n), the sample
    covariance eigenvalues are supported on [(1-√γ)², (1+√γ)²] · σ²
    where γ = d/n.

    Eigenvalues exceeding the upper bound are signal; count them.
    """
    gamma = d / n
    # Estimate σ² from the bulk (median eigenvalue is robust)
    sigma2 = float(np.median(eigenvalues))
    upper_bound = sigma2 * (1 + np.sqrt(gamma)) ** 2
    n_signal = int(np.sum(eigenvalues > upper_bound))
    return max(1, n_signal)


def parallel_analysis_k(data: np.ndarray, n_permutations: int = 20) -> int:
    """Parallel analysis (Horn 1965).

    Generate random data of same shape, compute eigenvalues, keep components
    whose eigenvalues exceed the 95th percentile of random reference.
    """
    n, d = data.shape
    # Real eigenvalues
    cov = np.cov(data.T)
    real_eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]

    # Random reference eigenvalues
    rng = np.random.default_rng(42)
    random_eigs_all = np.zeros((n_permutations, d))
    for i in range(n_permutations):
        random_data = rng.standard_normal((n, d))
        random_cov = np.cov(random_data.T)
        random_eigs_all[i] = np.sort(np.linalg.eigvalsh(random_cov))[::-1]

    # 95th percentile of random eigenvalues at each position
    threshold = np.percentile(random_eigs_all, 95, axis=0)

    # Count how many real eigenvalues exceed the threshold
    n_signal = int(np.sum(real_eigs > threshold))
    return max(1, n_signal)


# =====================================================================
# Edge calibration analysis
# =====================================================================


def compute_edge_data(
    tree: PosetTree,
    mean_bl: float | None,
) -> list[dict]:
    """Collect T/k ratios and metadata for all edges."""
    records = []
    for parent, child in tree.edges():
        n_child = extract_node_sample_size(tree, child)
        n_parent = extract_node_sample_size(tree, parent)
        child_dist = tree.nodes[child].get("distribution")
        parent_dist = tree.nodes[parent].get("distribution")
        if child_dist is None or parent_dist is None:
            continue
        child_dist = np.asarray(child_dist, dtype=np.float64)
        parent_dist = np.asarray(parent_dist, dtype=np.float64)
        if n_child >= n_parent:
            continue

        bl = tree.edges[parent, child].get("branch_length")

        # Compute z WITHOUT Felsenstein (to see raw calibration)
        z_raw = _compute_standardized_z(
            child_dist,
            parent_dist,
            n_child,
            n_parent,
            branch_length=None,
            mean_branch_length=None,
        )
        T_raw = float(np.sum(z_raw**2))
        d = len(z_raw)

        # With Felsenstein
        z_f = _compute_standardized_z(
            child_dist,
            parent_dist,
            n_child,
            n_parent,
            branch_length=bl,
            mean_branch_length=mean_bl,
        )
        T_f = float(np.sum(z_f**2))

        records.append(
            {
                "parent": parent,
                "child": child,
                "n_child": n_child,
                "n_parent": n_parent,
                "bl": bl if bl is not None else 0.0,
                "d": d,
                "z_raw": z_raw,
                "T_raw": T_raw,
                "ratio_raw": T_raw / d,
                "T_felsen": T_f,
                "ratio_felsen": T_f / d,
            }
        )
    return records


def print_calibration_analysis(records: list[dict]) -> None:
    """Analyze leaf vs internal edges for calibration viability."""
    leaf_edges = [r for r in records if r["n_child"] == 1]
    internal_edges = [r for r in records if r["n_child"] > 1]

    print("=" * 80)
    print("SECTION 1: EDGE CALIBRATION ANALYSIS")
    print("(Can we calibrate Gate 2 like cousin_adjusted_wald calibrates Gate 3?)")
    print("=" * 80)

    d = records[0]["d"]
    print(f"\nTotal edges: {len(records)} ({len(leaf_edges)} leaf, {len(internal_edges)} internal)")
    print(f"Feature dimension d = {d}")

    # --- Leaf edges: null-like calibration candidates ---
    lr = np.array([r["ratio_raw"] for r in leaf_edges])
    lf = np.array([r["ratio_felsen"] for r in leaf_edges])

    print("\n--- LEAF EDGES (null-like candidates, n_child=1) ---")
    print(
        f"  Raw T/k:     mean={np.mean(lr):.4f}  median={np.median(lr):.4f}  "
        f"std={np.std(lr):.4f}  [min={np.min(lr):.4f}, max={np.max(lr):.4f}]"
    )
    print(
        f"  Felsen T/k:  mean={np.mean(lf):.4f}  median={np.median(lf):.4f}  "
        f"std={np.std(lf):.4f}"
    )
    print("  Expected under H0: T/k → 1.0 (as χ²(k)/k → 1)")
    print(f"  Post-selection inflation (raw median): {np.median(lr):.4f}")

    # Percentile comparison with χ²(d)/d
    print(f"\n  Percentile comparison (raw T/k vs theoretical χ²({d})/{d}):")
    print(f"  {'Pct':>5s}  {'Empirical':>10s}  {'Theoretical':>11s}  {'Ratio':>7s}")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        emp = np.percentile(lr, pct)
        theo = chi2.ppf(pct / 100, df=d) / d
        print(f"  {pct:>5d}  {emp:>10.4f}  {theo:>11.4f}  {emp / theo:>7.3f}")

    # Type I error rate
    p_raw = np.array([float(chi2.sf(r["T_raw"], df=r["d"])) for r in leaf_edges])
    n_sig = int(np.sum(p_raw < 0.05))
    print("\n  Leaf edges significant at α=0.05 (raw, no correction):")
    print(f"    {n_sig}/{len(leaf_edges)} = {100 * n_sig / len(leaf_edges):.1f}%")
    print("    Expected under H0: 5.0%")
    inflation = (n_sig / len(leaf_edges)) / 0.05 if n_sig > 0 else 0
    print(f"    → Empirical Type I inflation: {inflation:.2f}x")

    # --- Internal edges ---
    ir = np.array([r["ratio_raw"] for r in internal_edges])
    print("\n--- INTERNAL EDGES (n_child > 1) ---")
    print(
        f"  Raw T/k:     mean={np.mean(ir):.4f}  median={np.median(ir):.4f}  "
        f"std={np.std(ir):.4f}"
    )

    # --- Calibration model feasibility ---
    print("\n--- COUSIN-STYLE EDGE CALIBRATION FEASIBILITY ---")
    print("  Sibling calibration uses 'null-like' pairs = siblings where")
    print("  neither child is edge-significant. For EDGE calibration,")
    print("  leaf edges are the natural null-like set (single observation")
    print("  vs parent, minimal expected divergence).")
    print("")
    print(f"  Available null-like edges (leaves): {len(leaf_edges)}")
    print(f"  Ratio leaf/total: {len(leaf_edges) / len(records):.1%}")

    # Regression features: bl, n_parent
    bls = np.array([r["bl"] for r in leaf_edges])
    nps = np.array([r["n_parent"] for r in leaf_edges])
    print(f"  BL range: [{np.min(bls):.6f}, {np.max(bls):.6f}]")
    print(f"  n_parent range: [{np.min(nps)}, {np.max(nps)}]")

    # Try fitting log-linear regression like cousin_adjusted_wald
    valid = (bls > 0) & (nps > 0) & (lr > 0)
    if np.sum(valid) >= 5:
        log_r = np.log(lr[valid])
        X = np.column_stack(
            [
                np.ones(np.sum(valid)),
                np.log(bls[valid]),
                np.log(nps[valid].astype(float)),
            ]
        )
        beta, residuals, rank, sv = np.linalg.lstsq(X, log_r, rcond=None)
        fitted = X @ beta
        ss_res = float(np.sum((log_r - fitted) ** 2))
        ss_tot = float(np.sum((log_r - np.mean(log_r)) ** 2))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print("\n  Log-linear regression: log(T/k) = β₀ + β₁·log(BL) + β₂·log(n_parent)")
        print(f"    β₀ = {beta[0]:.4f}, β₁ = {beta[1]:.4f}, β₂ = {beta[2]:.4f}")
        print(f"    R² = {r_sq:.4f}")
        print(f"    median ĉ = e^(median fitted) = {np.exp(np.median(fitted)):.4f}")

        if r_sq < 0.05:
            print("    → R² too low: inflation is NOT explained by BL/n_parent")
            print(f"    → A global median ĉ = {np.median(lr[valid]):.4f} is more appropriate")
    else:
        print(f"\n  Too few valid leaf edges ({np.sum(valid)}) for regression")


def print_k_selection_analysis(records: list[dict], data: np.ndarray) -> None:
    """Compare projection dimension selection strategies."""
    print("\n" + "=" * 80)
    print("SECTION 2: PROJECTION DIMENSION (k) SELECTION")
    print("(What function of n_descendants and d gives the right k?)")
    print("=" * 80)

    n, d = data.shape

    # Compute data covariance eigenvalues
    print(f"\nData dimensions: {n} samples × {d} features")
    cov = np.cov(data.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]

    # Strategy results
    k_erank = effective_rank(eigenvalues)
    k_mp = marchenko_pastur_k(eigenvalues, n, d)
    k_pa = parallel_analysis_k(data, n_permutations=20)

    print("\n--- DATA-DRIVEN k ESTIMATES (independent of n_child) ---")
    print(f"  Effective rank (Shannon entropy):     k = {k_erank}")
    print(f"  Marchenko-Pastur signal eigenvalues:  k = {k_mp}")
    print(f"  Parallel analysis (Horn 1965):        k = {k_pa}")
    print(f"  Full dimension:                       k = {d}")

    # Show eigenvalue spectrum
    print("\n--- EIGENVALUE SPECTRUM ---")
    print(f"  Top 20 eigenvalues (of {d} total):")
    for i in range(min(20, d)):
        cumvar = np.sum(eigenvalues[: i + 1]) / np.sum(eigenvalues) * 100
        print(f"    λ_{i + 1:>3d} = {eigenvalues[i]:>10.6f}  (cum. var: {cumvar:>6.2f}%)")

    # Variance explained thresholds
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    for frac in [0.5, 0.75, 0.9, 0.95, 0.99]:
        idx = int(np.searchsorted(cumvar, frac))
        print(f"  {frac * 100:.0f}% variance explained by k = {idx + 1} components")

    # JL dimension for different n_child values
    print("\n--- JL DIMENSION vs n_child (current approach) ---")
    print(f"  {'n_child':>8s}  {'k_JL':>6s}  {'k/d':>6s}  |  Note")
    for nc in [1, 2, 3, 5, 10, 15, 20, 50, 100, 200, 500]:
        k = k_jl(nc, d)
        note = ""
        if k == d:
            note = "saturated (k=d)"
        elif k == config.PROJECTION_MIN_K:
            note = "floored at min_k"
        print(f"  {nc:>8d}  {k:>6d}  {k / d:>6.2f}  |  {note}")

    # --- Epsilon sensitivity ---
    print("\n--- EPSILON SENSITIVITY (how eps affects k at various n) ---")
    print(f"  {'eps':>6s}", end="")
    ns = [2, 5, 10, 50, 200, 626]
    for nc in ns:
        print(f"  {'n=' + str(nc):>8s}", end="")
    print()
    for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.99]:
        print(f"  {eps:>6.2f}", end="")
        for nc in ns:
            k = k_jl(nc, d, eps=eps)
            print(f"  {k:>8d}", end="")
        print()

    # --- Key insight about JL misapplication ---
    print("\n--- WHY JL IS MISAPPLIED ---")
    print("  JL lemma: For n points in ℝ^d, random projection to k dimensions")
    print("  preserves ALL pairwise distances within 1±ε, if k ≥ 8 ln(n)/ε².")
    print("")
    print("  Edge test projects ONE z-vector to k dims and tests ||Rz||² ~ χ²(k).")
    print("  There are no 'n points' whose pairwise distances need preserving.")
    print("  n_child enters the JL formula but has no statistical meaning here.")
    print("")
    print("  The correct question: what k maximizes power for detecting ||μ||² > 0")
    print("  when z ~ N(μ, I_d) and T_k = ||R_k z||² ~ χ²(k, λ_k)?")
    print("")
    print("  For uninformed (random) R: λ_k = (k/d)·λ")
    print("    Power(k) ≈ Φ(λ√k / (d√(2k)) - z_α) = Φ(λ/(d√2)·√k - z_α)")
    print("    → Monotonically INCREASING in k → k=d always optimal")
    print("")
    print("  For informed (signal-subspace) R: λ_k = λ (full signal preserved)")
    print("    Power(k) ≈ Φ(λ/√(2k) - z_α)")
    print("    → DECREASING in k → optimal k = signal dimensionality s")
    print("")
    print("  ⇒ For random projection, k=d is always optimal (projection hurts).")
    print("  ⇒ For informed projection, k=s (signal dim) is optimal.")
    print("  ⇒ Data-driven k (e.g. effective rank, MP) matters only if we use")
    print("     informed projection (PCA-based or signal-adaptive).")


def print_power_at_different_k(records: list[dict], data: np.ndarray) -> None:
    """Show how T and p-value change as a function of k for representative edges."""
    print("\n" + "=" * 80)
    print("SECTION 3: POWER vs PROJECTION DIMENSION (k)")
    print("(Empirical: how does p-value change with k on real edges?)")
    print("=" * 80)

    n, d = data.shape

    # Pick a few representative edges
    internal = [r for r in records if r["n_child"] > 1]
    internal.sort(key=lambda r: -r["n_child"])

    # Pick edges at different scales
    if len(internal) >= 3:
        candidates = [
            internal[0],  # largest
            internal[len(internal) // 2],  # medium
            internal[-1],  # smallest
        ]
    else:
        candidates = internal[:3]

    # Also add a leaf edge as control
    leaf = [r for r in records if r["n_child"] == 1]
    if leaf:
        # Pick the leaf with median T/k
        leaf.sort(key=lambda r: r["ratio_raw"])
        candidates.append(leaf[len(leaf) // 2])

    k_values = [10, 20, 50, 100, 150, 200, 300, 400, d]

    for rec in candidates:
        z = rec["z_raw"]
        edge_d = len(z)
        print(
            f"\n--- Edge: {rec['parent']} → {rec['child']} "
            f"(n_child={rec['n_child']}, d={edge_d}, "
            f"T_full={rec['T_raw']:.1f}, T/d={rec['ratio_raw']:.3f}) ---"
        )
        print(f"  k_JL(n_child)={k_jl(rec['n_child'], edge_d)}")

        print(
            f"  {'k':>5s}  {'T_k':>8s}  {'E[T_k|H0]':>10s}  {'T_k/k':>7s}  "
            f"{'p_value':>10s}  {'significant':>11s}"
        )
        for k in k_values:
            kk = min(k, edge_d)
            # Generate projection and compute T_k
            seed = hash(f"{rec['parent']}-{rec['child']}-{kk}") % (2**31)
            R = generate_projection_matrix(edge_d, kk, seed, use_cache=False)
            projected = R @ z
            T_k = float(np.sum(projected**2))
            p_k = float(chi2.sf(T_k, df=kk))
            sig = "YES" if p_k < 0.05 else "no"
            print(
                f"  {kk:>5d}  {T_k:>8.1f}  {kk:>10.1f}  {T_k / kk:>7.3f}  "
                f"{p_k:>10.2e}  {sig:>11s}"
            )


def print_data_driven_epsilon(data: np.ndarray) -> None:
    """Investigate data-driven epsilon inference."""
    print("\n" + "=" * 80)
    print("SECTION 4: DATA-DRIVEN EPSILON INFERENCE")
    print("(Can we infer ε from the data instead of hardcoding 0.3?)")
    print("=" * 80)

    n, d = data.shape

    # Compute sample covariance eigenvalues
    cov = np.cov(data.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total_var = np.sum(eigenvalues)

    print("\n  The JL ε controls distortion: ||Rx||² = (1±ε)||x||².")
    print("  For a test, ε controls how much signal we lose in projection.")
    print("  Larger ε → smaller k → more signal loss → less power.")
    print("")
    print("  If the data lives on a low-dimensional manifold (effective rank << d),")
    print("  most of z will align with a few principal directions. An informed")
    print("  projection can use k ≈ effective_rank without losing signal,")
    print("  equivalent to a very small 'effective ε'.")
    print("")

    # Compute effective epsilon: if we project to k dims and capture
    # fraction f of variance, the effective distortion is ε_eff = 1 - f
    print("  Variance-to-dimension tradeoff:")
    print(f"  {'k':>6s}  {'var_captured':>12s}  {'ε_eff':>7s}  " f"{'k_from_JL(n=626)':>16s}")
    cumvar = np.cumsum(eigenvalues) / total_var
    for k in [10, 20, 50, 100, 150, 200, 300, 400, d]:
        kk = min(k, d)
        vc = cumvar[kk - 1]
        eps_eff = 1 - vc
        # What JL epsilon would give this k?
        # k ≈ 8 ln(n) / eps² → eps ≈ sqrt(8 ln(n) / k)
        eps_jl = np.sqrt(8 * np.log(n) / kk) if kk > 0 else float("inf")
        print(f"  {kk:>6d}  {vc:>12.4f}  {eps_eff:>7.4f}  {eps_jl:>16.3f}")

    # Suggest data-driven epsilon
    k_erank = effective_rank(eigenvalues)
    eps_suggested = np.sqrt(8 * np.log(n) / k_erank) if k_erank > 0 else 0.3
    print(f"\n  Effective rank = {k_erank} → back-inferred ε ≈ {eps_suggested:.3f}")
    print("  (This means: if you want k ≈ effective_rank from JL,")
    print(f"   you'd need ε = {eps_suggested:.3f} instead of 0.3)")

    # But the real insight:
    print("\n  HOWEVER: ε is a distortion guarantee, not a power parameter.")
    print("  For uninformed random projection, k=d is always most powerful.")
    print("  ε only matters if you're preserving pairwise distances (JL setting).")
    print("  For a single-vector test, ε has no direct statistical meaning.")
    print("")
    print("  The correct data-driven approach: use INFORMED projection")
    print("  (PCA-based) with k = signal dimension, not JL with tuned ε.")


def print_recommendations(records: list[dict], data: np.ndarray) -> None:
    """Synthesize recommendations."""
    n, d = data.shape
    cov = np.cov(data.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]

    k_erank = effective_rank(eigenvalues)
    k_mp = marchenko_pastur_k(eigenvalues, n, d)
    k_pa = parallel_analysis_k(data, n_permutations=20)

    leaf_edges = [r for r in records if r["n_child"] == 1]
    lr = np.array([r["ratio_raw"] for r in leaf_edges])

    print("\n" + "=" * 80)
    print("SECTION 5: RECOMMENDATIONS")
    print("=" * 80)

    print("\n  1. PROJECTION DIMENSION:")
    print(f"     Current: k = JL(n_child, ε=0.3) → varies from {k_jl(1, d)} to {d}")
    print("     Problem: JL is misapplied (no pairwise distances to preserve)")
    print("")
    print("     Data-driven alternatives:")
    print(f"       effective_rank = {k_erank}")
    print(f"       MP threshold  = {k_mp}")
    print(f"       parallel analysis = {k_pa}")
    print("")
    print("     For RANDOM projection: k=d always optimal (no projection)")
    print("     For INFORMED projection: k = MP or parallel analysis")
    print("")

    print("  2. EPSILON:")
    print("     Current: hardcoded 0.3")
    print("     If JL is replaced with data-driven k, ε is irrelevant.")
    print(f"     If JL is kept, back-inferred ε from effective_rank = {k_erank}:")
    eps_back = np.sqrt(8 * np.log(n) / k_erank) if k_erank > 0 else 0.3
    print(f"       ε ≈ {eps_back:.3f}")
    print("")

    print("  3. EDGE CALIBRATION (Gate 2):")
    print(f"     Leaf-edge inflation (median T/k): {np.median(lr):.4f}")
    if np.median(lr) > 1.1:
        print(f"     → Post-selection inflation detected ({np.median(lr):.2f}x)")
        print("     → Same cousin-adjusted approach as Gate 3 is viable:")
        print("        (a) Use leaf edges as null-like calibration set")
        print("        (b) Fit log(T/k) ~ β₀ + β₁·log(BL) + β₂·log(n_parent)")
        print("        (c) Deflate internal edges by predicted ĉ")
    elif np.median(lr) < 0.9:
        print("     → T/k < 1 at leaves: test is CONSERVATIVE (under-powered)")
        print("     → This explains K=1 collapse on sparse data")
        print("     → Need to FIX the test (projection, Felsenstein) not calibrate")
    else:
        print("     → T/k ≈ 1 at leaves: test is well-calibrated under null")
        print("     → Calibration is not needed for Gate 2 itself")
        print("     → Power issue is in the test statistic, not calibration")


def main() -> None:
    args = parse_args()

    # Load data and build tree
    data_df = pd.read_csv(args.input, sep="\t", index_col=0).astype(int)
    data = data_df.values
    n, d = data.shape
    print(f"Data: {n} samples × {d} features")
    print(f"Sparsity: {100 * (data == 0).mean():.1f}% zeros")

    dist = pdist(data, metric=config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
    tree = tree_from_linkage(Z, leaf_names=data_df.index.tolist())
    tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)
    mean_bl = compute_mean_branch_length(tree)

    print(f"Tree: {tree.number_of_nodes()} nodes, {tree.number_of_edges()} edges")
    print(f"Mean branch length: {mean_bl:.6f}" if mean_bl else "Mean BL: None")

    # Collect edge data
    records = compute_edge_data(tree, mean_bl)
    print(f"Analyzable edges: {len(records)}")

    # Run analyses
    print_calibration_analysis(records)
    print_k_selection_analysis(records, data)
    print_power_at_different_k(records, data)
    print_data_driven_epsilon(data)
    print_recommendations(records, data)


if __name__ == "__main__":
    main()
