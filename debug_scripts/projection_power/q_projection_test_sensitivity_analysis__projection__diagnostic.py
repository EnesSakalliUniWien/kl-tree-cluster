"""
Purpose: Analysis: Improving projection test sensitivity with scipy.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/projection_power/q_projection_test_sensitivity_analysis__projection__diagnostic.py
"""

import numpy as np
from scipy import stats
from scipy.linalg import qr
from scipy.stats import chi2, f as f_dist


def current_chi_square_test(z_proj: np.ndarray, k: int) -> float:
    """Current method: ||Rz||² ~ χ²(k)."""
    test_stat = np.sum(z_proj**2)
    return float(chi2.sf(test_stat, df=k))


def hotelling_t2_test(
    left_means: np.ndarray,
    right_means: np.ndarray,
    n_left: int,
    n_right: int,
    pooled_var: np.ndarray,
) -> tuple[float, float, float]:
    """Hotelling's T² test using F-distribution.

    More appropriate for comparing multivariate means with estimated variance.

    T² = n_eff * Δμ'·S⁻¹·Δμ

    Under H₀: T² * (n1+n2-p-1) / (p*(n1+n2-2)) ~ F(p, n1+n2-p-1)

    This has better power than χ² for small samples because it accounts
    for variance estimation uncertainty.
    """
    p = len(left_means)
    n_total = n_left + n_right
    n_eff = (2.0 * n_left * n_right) / (n_left + n_right)

    diff = left_means - right_means

    # Compute T² (using diagonal covariance for independent features)
    # For Bernoulli: variance = θ(1-θ)(1/n₁ + 1/n₂)
    var_diag = np.maximum(pooled_var, 1e-10)

    # T² = n_eff * Σ (diff_i² / var_i)
    t2 = n_eff * np.sum(diff**2 / var_diag)

    # Convert to F-statistic
    df1 = p
    df2 = n_total - p - 1

    if df2 <= 0:
        # Not enough samples for F-test, fall back to chi-square
        return t2, float(p), float(chi2.sf(t2, df=p))

    f_stat = t2 * df2 / (p * (n_total - 2))
    p_value = float(f_dist.sf(f_stat, df1, df2))

    return f_stat, float(df1), p_value


def projected_hotelling_test(
    left_dist: np.ndarray,
    right_dist: np.ndarray,
    n_left: int,
    n_right: int,
    k: int,
    n_trials: int = 5,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Hotelling T² test after random projection.

    Uses F-distribution which is more powerful for small samples.
    """
    d = len(left_dist)
    n_total = n_left + n_right

    eps = 1e-10
    pooled = 0.5 * (left_dist + right_dist)
    pooled = np.clip(pooled, eps, 1.0 - eps)
    inv_n = 1.0 / n_left + 1.0 / n_right
    var_diff = pooled * (1.0 - pooled) * inv_n
    var_diff = np.maximum(var_diff, eps)

    # Standardize difference
    diff = left_dist - right_dist
    z = diff / np.sqrt(var_diff)

    # Project and compute Hotelling T²
    rng = np.random.default_rng(seed)

    f_stats = []
    for trial in range(n_trials):
        # Generate orthonormal projection
        G = rng.standard_normal((k, d))
        Q, _ = qr(G.T, mode="economic")
        R = Q.T  # (k, d) with orthonormal rows

        z_proj = R @ z

        # For projected data, use χ²(k) approximation
        # But with F-correction for sample size
        df1 = k
        df2 = n_total - k - 1

        if df2 > 0:
            t2 = np.sum(z_proj**2)
            f_stat = t2 * df2 / (k * max(n_total - 2, 1))
            f_stats.append(f_stat)
        else:
            # Fall back to chi-square
            f_stats.append(np.sum(z_proj**2))

    avg_f = np.mean(f_stats)

    # Use F-distribution if enough samples
    df2 = n_total - k - 1
    if df2 > 0:
        p_value = float(f_dist.sf(avg_f, k, df2))
    else:
        p_value = float(chi2.sf(avg_f, df=k))

    return avg_f, float(k), p_value


def compare_tests():
    """Compare sensitivity of different tests."""
    np.random.seed(42)

    # Simulate data: two groups with small difference
    d = 3000
    k = 14
    n_left, n_right = 30, 30

    # True effect: 5% of features differ by 0.2
    theta_base = np.random.uniform(0.3, 0.7, d)
    effect_size = 0.15
    n_diff = int(0.05 * d)  # 5% of features
    diff_idx = np.random.choice(d, n_diff, replace=False)

    # Simulate H₁ (different)
    theta_left = theta_base.copy()
    theta_right = theta_base.copy()
    theta_right[diff_idx] += effect_size
    theta_right = np.clip(theta_right, 0.05, 0.95)

    # Variance
    pooled = 0.5 * (theta_left + theta_right)
    inv_n = 1 / n_left + 1 / n_right
    var_diff = pooled * (1 - pooled) * inv_n

    # Standardize
    diff = theta_left - theta_right
    z = diff / np.sqrt(var_diff)

    # Generate projection
    rng = np.random.default_rng(42)
    G = rng.standard_normal((k, d))
    Q, _ = qr(G.T, mode="economic")
    R = Q.T
    z_proj = R @ z

    print("=" * 70)
    print("TEST COMPARISON: Detecting small differences in high-d data")
    print("=" * 70)
    print(f"d={d}, k={k}, n_left={n_left}, n_right={n_right}")
    print(f"Effect: {n_diff} features differ by {effect_size}")
    print()

    # Test 1: Current χ² test
    chi2_stat = np.sum(z_proj**2)
    p_chi2 = chi2.sf(chi2_stat, df=k)
    print(f"1. Chi-square test (current):")
    print(f"   T = {chi2_stat:.2f}, df = {k}, p = {p_chi2:.4f}")
    print(f"   Detect at α=0.05? {'YES' if p_chi2 < 0.05 else 'NO'}")

    # Test 2: Hotelling T² with F-distribution
    n_total = n_left + n_right
    df2 = n_total - k - 1
    if df2 > 0:
        f_stat = chi2_stat * df2 / (k * (n_total - 2))
        p_f = f_dist.sf(f_stat, k, df2)
        print(f"\n2. Hotelling T² with F-distribution:")
        print(f"   F = {f_stat:.2f}, df1 = {k}, df2 = {df2}, p = {p_f:.4f}")
        print(f"   Detect at α=0.05? {'YES' if p_f < 0.05 else 'NO'}")

    # Test 3: Increase projection dimension
    k_larger = 50
    G2 = rng.standard_normal((k_larger, d))
    Q2, _ = qr(G2.T, mode="economic")
    R2 = Q2.T
    z_proj2 = R2 @ z
    chi2_stat2 = np.sum(z_proj2**2)
    p_chi2_k50 = chi2.sf(chi2_stat2, df=k_larger)
    print(f"\n3. Chi-square with larger k={k_larger}:")
    print(f"   T = {chi2_stat2:.2f}, df = {k_larger}, p = {p_chi2_k50:.4f}")
    print(f"   Detect at α=0.05? {'YES' if p_chi2_k50 < 0.05 else 'NO'}")

    # Test 4: Direct chi-square on high-d (no projection)
    chi2_full = np.sum(z**2)
    p_full = chi2.sf(chi2_full, df=d)
    print(f"\n4. Full-dimensional chi-square (no projection):")
    print(f"   T = {chi2_full:.2f}, df = {d}, p = {p_full:.4f}")
    print(f"   Detect at α=0.05? {'YES' if p_full < 0.05 else 'NO'}")

    print()
    print("=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("1. Increase PROJECTION_K_MULTIPLIER from 4.0 to 8.0 or higher")
    print("2. Use F-distribution instead of χ² for sample-size correction")
    print("3. Consider permutation test for exact p-values")
    print("4. Add effect-size threshold (Cohen's d) alongside p-value")


def monte_carlo_power_analysis():
    """Compare power of different tests under H₁."""
    np.random.seed(123)

    d = 3000
    n_left, n_right = 30, 30
    n_simulations = 500
    alpha = 0.05

    # Effect sizes to test
    effect_sizes = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    k_values = [14, 30, 50, 100]

    print("\n" + "=" * 70)
    print("MONTE CARLO POWER ANALYSIS")
    print("=" * 70)
    print(f"d={d}, n_left={n_left}, n_right={n_right}, α={alpha}")
    print(f"Simulations per condition: {n_simulations}")
    print()

    for effect in effect_sizes:
        print(f"\nEffect size (5% features differ by {effect}):")
        print(f"  {'k':>6} | {'Power (χ²)':>12} | {'Power (F)':>12}")
        print(f"  {'-' * 6}+{'-' * 14}+{'-' * 14}")

        for k in k_values:
            reject_chi2 = 0
            reject_f = 0

            for _ in range(n_simulations):
                # Generate data
                theta_base = np.random.uniform(0.3, 0.7, d)
                n_diff = int(0.05 * d)
                diff_idx = np.random.choice(d, n_diff, replace=False)

                theta_left = theta_base.copy()
                theta_right = theta_base.copy()
                theta_right[diff_idx] += effect
                theta_right = np.clip(theta_right, 0.05, 0.95)

                # Standardize
                pooled = 0.5 * (theta_left + theta_right)
                inv_n = 1 / n_left + 1 / n_right
                var_diff = pooled * (1 - pooled) * inv_n
                var_diff = np.maximum(var_diff, 1e-10)
                diff = theta_left - theta_right
                z = diff / np.sqrt(var_diff)

                # Project
                rng = np.random.default_rng()
                G = rng.standard_normal((k, d))
                Q, _ = qr(G.T, mode="economic")
                R = Q.T
                z_proj = R @ z

                # Chi-square test
                stat = np.sum(z_proj**2)
                p_chi2 = chi2.sf(stat, df=k)
                if p_chi2 < alpha:
                    reject_chi2 += 1

                # F-test
                n_total = n_left + n_right
                df2 = n_total - k - 1
                if df2 > 0:
                    f_stat = stat * df2 / (k * (n_total - 2))
                    p_f = f_dist.sf(f_stat, k, df2)
                    if p_f < alpha:
                        reject_f += 1
                else:
                    if p_chi2 < alpha:
                        reject_f += 1

            power_chi2 = reject_chi2 / n_simulations
            power_f = reject_f / n_simulations
            print(f"  {k:>6} | {power_chi2:>12.1%} | {power_f:>12.1%}")


if __name__ == "__main__":
    compare_tests()
    monte_carlo_power_analysis()
