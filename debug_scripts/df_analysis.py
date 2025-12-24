"""
Simulation study: Variance-weighted df vs Theoretical df

This script analyzes whether the variance-weighted degrees of freedom formula
4·θ·(1-θ) is mathematically correct for the chi-square approximation of
2*n*KL divergence under the null hypothesis.

Key questions:
1. Does 2*n*KL follow χ²(p) or χ²(df_weighted)?
2. How does the choice of df affect Type I error rates?
3. Which df formula gives better calibrated p-values?
"""

import numpy as np
from scipy.stats import chi2, kstest
import matplotlib.pyplot as plt


def kl_bernoulli(theta_hat: np.ndarray, theta: np.ndarray, eps: float = 1e-10) -> float:
    """KL divergence for Bernoulli: sum over features."""
    theta_hat = np.clip(theta_hat, eps, 1 - eps)
    theta = np.clip(theta, eps, 1 - eps)
    kl = theta_hat * np.log(theta_hat / theta) + (1 - theta_hat) * np.log(
        (1 - theta_hat) / (1 - theta)
    )
    return float(np.sum(kl))


def jsd_bernoulli(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence for Bernoulli distributions."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    m = 0.5 * (p + q)
    kl_pm = kl_bernoulli(p, m, eps)
    kl_qm = kl_bernoulli(q, m, eps)
    return 0.5 * kl_pm + 0.5 * kl_qm


def simulate_kl_null_distribution(
    n_samples: int,
    n_features: int,
    theta_parent: np.ndarray,
    n_simulations: int = 5000,
) -> np.ndarray:
    """Simulate 2*n*KL under H0 (child = parent)."""
    test_stats = []

    for _ in range(n_simulations):
        # Generate child samples from parent distribution (H0 true)
        samples = np.random.binomial(1, theta_parent, size=(n_samples, n_features))
        theta_child = np.mean(samples, axis=0)

        kl = kl_bernoulli(theta_child, theta_parent)
        test_stat = 2 * n_samples * kl
        test_stats.append(test_stat)

    return np.array(test_stats)


def simulate_jsd_null_distribution(
    n_left: int,
    n_right: int,
    n_features: int,
    theta_parent: np.ndarray,
    n_simulations: int = 5000,
) -> np.ndarray:
    """Simulate 2*n_eff*JSD under H0 (siblings have same distribution)."""
    n_eff = (2.0 * n_left * n_right) / (n_left + n_right)
    test_stats = []

    for _ in range(n_simulations):
        # Generate both siblings from same parent distribution (H0 true)
        samples_left = np.random.binomial(1, theta_parent, size=(n_left, n_features))
        samples_right = np.random.binomial(1, theta_parent, size=(n_right, n_features))

        theta_left = np.mean(samples_left, axis=0)
        theta_right = np.mean(samples_right, axis=0)

        jsd = jsd_bernoulli(theta_left, theta_right)
        test_stat = 2 * n_eff * jsd
        test_stats.append(test_stat)

    return np.array(test_stats)


def variance_weighted_df(theta: np.ndarray) -> float:
    """Calculate variance-weighted effective df."""
    return float(np.sum(4 * theta * (1 - theta)))


def analyze_df_calibration(test_stats: np.ndarray, df_options: dict) -> dict:
    """Analyze how well different df values calibrate the test statistics."""
    results = {}

    for name, df in df_options.items():
        # KS test for goodness of fit
        ks_stat, ks_pvalue = kstest(test_stats, chi2(df=df).cdf)

        # Empirical vs theoretical moments
        empirical_mean = np.mean(test_stats)
        empirical_var = np.var(test_stats)
        theoretical_mean = df
        theoretical_var = 2 * df

        # Type I error at α = 0.05
        critical_value = chi2.ppf(0.95, df=df)
        type1_error = np.mean(test_stats > critical_value)

        results[name] = {
            "df": df,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue,
            "empirical_mean": empirical_mean,
            "theoretical_mean": theoretical_mean,
            "mean_ratio": empirical_mean / theoretical_mean,
            "empirical_var": empirical_var,
            "theoretical_var": theoretical_var,
            "var_ratio": empirical_var / theoretical_var,
            "type1_error": type1_error,
        }

    return results


def print_analysis_results(results: dict, case_name: str):
    """Pretty print analysis results."""
    print(f"\n{'=' * 70}")
    print(f"  {case_name}")
    print(f"{'=' * 70}")
    print(
        f"{'Method':<20} {'df':>8} {'KS p-val':>10} {'Mean Ratio':>12} {'Var Ratio':>12} {'Type I':>10}"
    )
    print("-" * 70)

    for name, r in results.items():
        ks_str = f"{r['ks_pvalue']:.4f}" if r["ks_pvalue"] > 0.001 else "<0.001"
        print(
            f"{name:<20} {r['df']:>8.1f} {ks_str:>10} {r['mean_ratio']:>12.3f} {r['var_ratio']:>12.3f} {r['type1_error']:>10.3f}"
        )

    print("-" * 70)
    print("  KS p-value > 0.05 → Good fit to χ² distribution")
    print("  Mean/Var Ratio ≈ 1.0 → Correct df")
    print("  Type I ≈ 0.05 → Correct calibration")


def run_comprehensive_analysis():
    """Run comprehensive df analysis across different scenarios."""
    np.random.seed(42)

    n_sims = 5000
    n_samples = 100
    n_features = 20

    print("\n" + "=" * 70)
    print("  SIMULATION STUDY: Variance-Weighted df vs Theoretical df")
    print("  Testing: 2*n*KL ~ χ²(df) under H0")
    print("=" * 70)
    print(f"  n_samples = {n_samples}, n_features = {n_features}, n_sims = {n_sims}")

    # =========================================================================
    # PART 1: KL Divergence (Child vs Parent)
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("  PART 1: KL DIVERGENCE (Child vs Parent)")
    print("#" * 70)

    # Case 1: Uniform θ = 0.5
    theta_uniform = np.full(n_features, 0.5)
    stats = simulate_kl_null_distribution(n_samples, n_features, theta_uniform, n_sims)
    df_theoretical = n_features
    df_weighted = variance_weighted_df(theta_uniform)

    results = analyze_df_calibration(
        stats, {"Theoretical (p)": df_theoretical, "Variance-weighted": df_weighted}
    )
    print_analysis_results(results, "Case 1: θ = 0.5 (uniform, max variance)")

    # Case 2: Extreme θ near boundaries
    theta_extreme = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
    stats = simulate_kl_null_distribution(n_samples, n_features, theta_extreme, n_sims)
    df_weighted = variance_weighted_df(theta_extreme)

    results = analyze_df_calibration(
        stats, {"Theoretical (p)": df_theoretical, "Variance-weighted": df_weighted}
    )
    print_analysis_results(results, "Case 2: θ = 0.1 or 0.9 (extreme)")

    # Case 3: Mixed θ
    theta_mixed = np.linspace(0.1, 0.9, n_features)
    stats = simulate_kl_null_distribution(n_samples, n_features, theta_mixed, n_sims)
    df_weighted = variance_weighted_df(theta_mixed)

    results = analyze_df_calibration(
        stats, {"Theoretical (p)": df_theoretical, "Variance-weighted": df_weighted}
    )
    print_analysis_results(results, "Case 3: θ = linspace(0.1, 0.9) (mixed)")

    # Case 4: Very extreme θ
    theta_very_extreme = np.concatenate([np.full(10, 0.01), np.full(10, 0.99)])
    stats = simulate_kl_null_distribution(
        n_samples, n_features, theta_very_extreme, n_sims
    )
    df_weighted = variance_weighted_df(theta_very_extreme)

    results = analyze_df_calibration(
        stats, {"Theoretical (p)": df_theoretical, "Variance-weighted": df_weighted}
    )
    print_analysis_results(results, "Case 4: θ = 0.01 or 0.99 (very extreme)")

    # =========================================================================
    # PART 2: JSD (Sibling vs Sibling)
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("  PART 2: JENSEN-SHANNON DIVERGENCE (Sibling vs Sibling)")
    print("#" * 70)

    n_left = 50
    n_right = 50

    # Case 1: Uniform θ = 0.5
    theta_uniform = np.full(n_features, 0.5)
    stats = simulate_jsd_null_distribution(
        n_left, n_right, n_features, theta_uniform, n_sims
    )
    df_theoretical = n_features
    df_weighted = variance_weighted_df(theta_uniform)

    results = analyze_df_calibration(
        stats, {"Theoretical (p)": df_theoretical, "Variance-weighted": df_weighted}
    )
    print_analysis_results(results, "Case 1: θ = 0.5 (uniform)")

    # Case 2: Extreme θ
    theta_extreme = np.concatenate([np.full(10, 0.1), np.full(10, 0.9)])
    stats = simulate_jsd_null_distribution(
        n_left, n_right, n_features, theta_extreme, n_sims
    )
    df_weighted = variance_weighted_df(theta_extreme)

    results = analyze_df_calibration(
        stats, {"Theoretical (p)": df_theoretical, "Variance-weighted": df_weighted}
    )
    print_analysis_results(results, "Case 2: θ = 0.1 or 0.9 (extreme)")

    # Case 3: Asymmetric sample sizes
    n_left_asym = 80
    n_right_asym = 20

    theta_mixed = np.linspace(0.1, 0.9, n_features)
    stats = simulate_jsd_null_distribution(
        n_left_asym, n_right_asym, n_features, theta_mixed, n_sims
    )
    df_weighted = variance_weighted_df(theta_mixed)

    results = analyze_df_calibration(
        stats, {"Theoretical (p)": df_theoretical, "Variance-weighted": df_weighted}
    )
    print_analysis_results(
        results, f"Case 3: Asymmetric n_left={n_left_asym}, n_right={n_right_asym}"
    )

    # =========================================================================
    # PART 3: Summary and Recommendation
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  SUMMARY AND RECOMMENDATION")
    print("=" * 70)
    print(
        """
Key Findings:
-------------
1. For θ ≈ 0.5 (max variance): Both df formulas give identical results
   since 4·0.5·0.5 = 1.0 for each feature.

2. For extreme θ (near 0 or 1): 
   - Theoretical df (p) may OVERESTIMATE df → CONSERVATIVE (lower power)
   - Variance-weighted df may UNDERESTIMATE df → LIBERAL (higher Type I error)

3. The chi-square approximation 2*n*KL ~ χ²(df) assumes:
   - Large sample sizes (n >> p)
   - θ bounded away from 0 and 1

Mathematical Derivation:
------------------------
For Bernoulli feature i with true θᵢ:
- MLE: θ̂ᵢ ~ N(θᵢ, θᵢ(1-θᵢ)/n)
- Fisher Information: I(θᵢ) = 1/(θᵢ(1-θᵢ))
- Score statistic: (θ̂ᵢ - θᵢ)² · I(θᵢ) · n ~ χ²(1)

The 2*n*KL statistic equals the score statistic asymptotically,
so each feature contributes exactly 1 df, regardless of θᵢ.

The variance-weighted formula 4·θ·(1-θ) appears to conflate:
- Variance of θ̂ (which is θ(1-θ)/n)  
- Contribution to df (which is always 1 per feature)

Recommendation:
---------------
Use df = p (number of features) as the theoretically correct value.
The variance-weighted formula is NOT justified by asymptotic theory.

However, in practice with small samples or extreme θ, the chi-square
approximation itself may be poor, regardless of df choice.
"""
    )


if __name__ == "__main__":
    run_comprehensive_analysis()
