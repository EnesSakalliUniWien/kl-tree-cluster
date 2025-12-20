"""Validate that the fixed sibling divergence test has correct Type I error.

This script tests the new standardized Euclidean test statistic:
    T = Σ (θ_left - θ_right)² / Var[Δθ] ~ χ²(p)

where Var[Δθ] = θ_pooled(1-θ_pooled)(1/n_left + 1/n_right).
"""

import numpy as np
from scipy import stats

np.random.seed(42)


def simulate_null_standardized_euclidean(
    n_left: int = 100,
    n_right: int = 100,
    p: int = 50,
    theta_range: tuple = (0.1, 0.9),
    n_simulations: int = 2000,
) -> dict:
    """Simulate the standardized Euclidean test under H₀."""

    # True distribution (same for both siblings under H₀)
    theta_true = np.random.uniform(theta_range[0], theta_range[1], size=p)

    test_statistics = []
    p_values = []

    for _ in range(n_simulations):
        # Generate samples from Bernoulli(θ_j) for each feature
        X_left = np.random.binomial(1, theta_true, size=(n_left, p))
        X_right = np.random.binomial(1, theta_true, size=(n_right, p))

        # Empirical distributions
        theta_left = X_left.mean(axis=0)
        theta_right = X_right.mean(axis=0)

        # Pooled estimate
        eps = 1e-10
        pooled = 0.5 * (theta_left + theta_right)
        pooled = np.clip(pooled, eps, 1.0 - eps)

        # Variance of the difference
        inverse_n_sum = 1.0 / n_left + 1.0 / n_right
        var_diff = pooled * (1.0 - pooled) * inverse_n_sum
        var_diff = np.maximum(var_diff, eps)

        # Test statistic
        diff = theta_left - theta_right
        T = np.sum(diff**2 / var_diff)
        test_statistics.append(T)

        # p-value
        pval = stats.chi2.sf(T, df=p)
        p_values.append(pval)

    test_statistics = np.array(test_statistics)
    p_values = np.array(p_values)

    # Compare to χ²(p)
    theoretical_mean = p
    theoretical_var = 2 * p

    empirical_mean = np.mean(test_statistics)
    empirical_var = np.var(test_statistics)

    # Type I error at different levels
    type_i_005 = np.mean(p_values < 0.05)
    type_i_010 = np.mean(p_values < 0.10)
    type_i_001 = np.mean(p_values < 0.01)

    return {
        "n_left": n_left,
        "n_right": n_right,
        "p": p,
        "theoretical_mean": theoretical_mean,
        "empirical_mean": empirical_mean,
        "mean_ratio": empirical_mean / theoretical_mean,
        "theoretical_var": theoretical_var,
        "empirical_var": empirical_var,
        "type_i_001": type_i_001,
        "type_i_005": type_i_005,
        "type_i_010": type_i_010,
    }


def main():
    print("=" * 70)
    print("VALIDATING STANDARDIZED EUCLIDEAN TEST STATISTIC")
    print("=" * 70)
    print()
    print("Testing: T = Σ (θ_left - θ_right)² / Var[Δθ] ~ χ²(p)")
    print("where Var[Δθ] = θ_pooled(1-θ_pooled)(1/n_left + 1/n_right)")
    print()

    # Test different configurations
    configs = [
        {"n_left": 50, "n_right": 50, "p": 20},
        {"n_left": 100, "n_right": 100, "p": 50},
        {"n_left": 200, "n_right": 200, "p": 100},
        {"n_left": 50, "n_right": 150, "p": 50},  # Unbalanced
    ]

    print("-" * 70)
    print(f"{'Config':<25} {'Mean Ratio':<12} {'Type I (5%)':<15} {'Status':<10}")
    print("-" * 70)

    all_pass = True
    for cfg in configs:
        result = simulate_null_standardized_euclidean(**cfg, n_simulations=2000)

        # Check if Type I error is within acceptable range (3-7% for 5% nominal)
        is_valid = 0.03 <= result["type_i_005"] <= 0.07
        status = "✓ PASS" if is_valid else "✗ FAIL"
        if not is_valid:
            all_pass = False

        config_str = f"n=({cfg['n_left']},{cfg['n_right']}), p={cfg['p']}"
        print(
            f"{config_str:<25} "
            f"{result['mean_ratio']:.3f}       "
            f"{result['type_i_005']:.1%}           "
            f"{status}"
        )

    print("-" * 70)
    print()

    if all_pass:
        print("✓ ALL CONFIGURATIONS PASS - The fix is correct!")
    else:
        print("✗ SOME CONFIGURATIONS FAILED - Review needed")

    # Detailed output for one configuration
    print()
    print("=" * 70)
    print("DETAILED RESULTS FOR n=(100,100), p=50")
    print("=" * 70)
    result = simulate_null_standardized_euclidean(
        n_left=100, n_right=100, p=50, n_simulations=5000
    )

    print(f"\nTest Statistic Distribution:")
    print(f"  Theoretical χ²({result['p']}) mean: {result['theoretical_mean']:.1f}")
    print(f"  Empirical mean:                    {result['empirical_mean']:.2f}")
    print(f"  Ratio (should be ~1.0):            {result['mean_ratio']:.3f}")

    print(f"\nType I Error Rates:")
    print(f"  α = 0.01: {result['type_i_001']:.1%} (nominal: 1%)")
    print(f"  α = 0.05: {result['type_i_005']:.1%} (nominal: 5%)")
    print(f"  α = 0.10: {result['type_i_010']:.1%} (nominal: 10%)")


if __name__ == "__main__":
    main()
