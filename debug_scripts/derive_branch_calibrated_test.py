#!/usr/bin/env python3
"""
Option B: Branch-Length Calibrated Divergence Test

Mathematical Framework
======================

CURRENT APPROACH:
-----------------
Test: H₀: θ_L = θ_R  (siblings have identical distributions)
Stat: T = (θ̂_L - θ̂_R)ᵀ Σ⁻¹ (θ̂_L - θ̂_R) ~ χ²_d

Problem: This ignores branch length information entirely.

PROPOSED APPROACH:
------------------
Key insight: Under an evolutionary model, we EXPECT some divergence
between siblings proportional to their branch lengths, even if they
came from the same ancestral population.

New test: H₀: Observed divergence = Expected divergence for branch lengths
          H₁: Observed divergence > Expected (true cluster separation)

The question: What is the expected divergence between two distributions
that evolved from a common ancestor along branches b_L and b_R?

This script derives and validates the mathematical relationship.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from kl_clustering_analysis.tree.poset_tree import PosetTree


# =============================================================================
# PART 1: THEORETICAL EXPECTED DIVERGENCE UNDER JUKES-CANTOR
# =============================================================================


def jc_mutation_probability(branch_length: float, K: int = 4) -> float:
    """Probability that a site mutates along a branch under Jukes-Cantor.

    P(mutation) = (K-1)/K * (1 - exp(-K*b/(K-1)))

    For K=4: P(mutation) = 0.75 * (1 - exp(-4b/3))
    """
    return ((K - 1) / K) * (1 - np.exp(-K * branch_length / (K - 1)))


def expected_site_difference(b_L: float, b_R: float, K: int = 4) -> float:
    """Expected probability that a site differs between L and R.

    After evolving from common ancestor along branches b_L and b_R:
    P(L ≠ R) = P(at least one mutated to different state)

    This is complex to derive exactly, but under independence assumption:
    P(L ≠ R) ≈ P(L mutated) + P(R mutated) - 2*P(both mutated to same new state)

    For symmetric mutation model:
    P(differ) = 1 - [(1-p_L)(1-p_R) + p_L*p_R/(K-1)]

    where p_L = P(L mutated), p_R = P(R mutated)
    """
    p_L = jc_mutation_probability(b_L, K)
    p_R = jc_mutation_probability(b_R, K)

    # P(same) = P(neither mutated) + P(both mutated to same state)
    # P(neither) = (1-p_L)(1-p_R)
    # P(both to same) = p_L * p_R * (1/K)  [simplified]
    p_same = (1 - p_L) * (1 - p_R) + p_L * p_R / K

    return 1 - p_same


def expected_hamming_distance(b_L: float, b_R: float, K: int = 4) -> float:
    """Expected Hamming distance between L and R after evolution.

    E[Hamming] = d * P(site differs)

    For normalized Hamming (per site): just P(site differs)
    """
    return expected_site_difference(b_L, b_R, K)


def expected_squared_difference_binary(
    b_L: float, b_R: float, n_L: int, n_R: int
) -> float:
    """Expected squared difference in proportions for binary case.

    For binary data (K=2), θ = P(X=1).

    After evolution with branches b_L, b_R from ancestor with θ_A:
    θ_L = θ_A(1-2p_L) + p_L
    θ_R = θ_A(1-2p_R) + p_R

    where p = mutation probability = 0.5(1 - exp(-2b))

    E[(θ_L - θ_R)²] = E[θ_A²](1-2p_L - 1+2p_R)² + ...

    This is complex. Let's derive it empirically.
    """
    # For binary: K=2
    p_L = 0.5 * (1 - np.exp(-2 * b_L))
    p_R = 0.5 * (1 - np.exp(-2 * b_R))

    # Under uniform prior on θ_A ~ Uniform(0,1):
    # E[θ_A] = 0.5, E[θ_A²] = 1/3, Var(θ_A) = 1/12

    # E[(θ_L - θ_R)²] when θ_L, θ_R are evolved versions of same θ_A
    # = (2p_L - 2p_R)² * Var(θ_A)  [if deterministic mutation]

    # But mutation is stochastic. For now, use empirical calibration.
    return None  # Need empirical calibration


# =============================================================================
# PART 2: EMPIRICAL CALIBRATION OF EXPECTED DIVERGENCE
# =============================================================================


def generate_evolved_siblings(
    n_samples: int,
    n_features: int,
    b_L: float,
    b_R: float,
    K: int = 4,
    rng: np.random.RandomState = None,
):
    """Generate two sibling populations that evolved from a common ancestor.

    Returns empirical distributions θ̂_L and θ̂_R.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Generate common ancestor (random categories)
    ancestor = rng.randint(0, K, size=n_features)

    # Evolve ancestor to get "true" sibling ancestral sequences
    def evolve(seq, b):
        P = np.full((K, K), ((K - 1) / K) * (1 - np.exp(-K * b / (K - 1))) / (K - 1))
        np.fill_diagonal(P, 1 / K + ((K - 1) / K) * np.exp(-K * b / (K - 1)))
        return np.array([rng.choice(K, p=P[s]) for s in seq])

    sib_L_ancestor = evolve(ancestor, b_L)
    sib_R_ancestor = evolve(ancestor, b_R)

    # Sample from each sibling (add terminal branch noise)
    terminal = 0.01  # Small within-cluster variation

    samples_L = np.array([evolve(sib_L_ancestor, terminal) for _ in range(n_samples)])
    samples_R = np.array([evolve(sib_R_ancestor, terminal) for _ in range(n_samples)])

    # Compute empirical distributions (frequency of each category per feature)
    theta_L = np.zeros((n_features, K))
    theta_R = np.zeros((n_features, K))

    for j in range(n_features):
        for c in range(K):
            theta_L[j, c] = (samples_L[:, j] == c).mean()
            theta_R[j, c] = (samples_R[:, j] == c).mean()

    return theta_L, theta_R, sib_L_ancestor, sib_R_ancestor


def compute_divergence_metrics(theta_L: np.ndarray, theta_R: np.ndarray) -> dict:
    """Compute various divergence metrics between two distributions."""
    eps = 1e-10
    d, K = theta_L.shape

    # L2 squared difference (sum over all features and categories)
    l2_sq = np.sum((theta_L - theta_R) ** 2)
    l2_sq_per_feature = np.sum((theta_L - theta_R) ** 2, axis=1)

    # JS divergence per feature
    m = 0.5 * (theta_L + theta_R)
    m = np.clip(m, eps, 1)
    theta_L_c = np.clip(theta_L, eps, 1)
    theta_R_c = np.clip(theta_R, eps, 1)

    kl_Lm = np.sum(theta_L_c * np.log(theta_L_c / m), axis=1)
    kl_Rm = np.sum(theta_R_c * np.log(theta_R_c / m), axis=1)
    js_per_feature = 0.5 * (kl_Lm + kl_Rm)
    js_total = js_per_feature.sum()

    return {
        "l2_squared": l2_sq,
        "l2_squared_mean": l2_sq / d,
        "js_total": js_total,
        "js_mean": js_total / d,
        "l2_sq_per_feature": l2_sq_per_feature,
        "js_per_feature": js_per_feature,
    }


def calibrate_expected_divergence(
    n_trials: int = 50,
    n_samples: int = 50,
    n_features: int = 100,
    K: int = 4,
):
    """Empirically calibrate expected divergence as function of branch lengths.

    For various (b_L, b_R) pairs, compute the average observed divergence
    to establish E[D | b_L, b_R].
    """
    branch_lengths = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    results = []

    for b in branch_lengths:
        # Symmetric case: b_L = b_R = b
        l2_values = []
        js_values = []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial * 1000 + int(b * 100))
            theta_L, theta_R, _, _ = generate_evolved_siblings(
                n_samples, n_features, b, b, K, rng
            )
            metrics = compute_divergence_metrics(theta_L, theta_R)
            l2_values.append(metrics["l2_squared_mean"])
            js_values.append(metrics["js_mean"])

        results.append(
            {
                "branch_length": b,
                "total_branch": 2 * b,  # Total evolutionary distance
                "expected_l2_mean": np.mean(l2_values),
                "std_l2_mean": np.std(l2_values),
                "expected_js_mean": np.mean(js_values),
                "std_js_mean": np.std(js_values),
                "expected_hamming": expected_hamming_distance(b, b, K),
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# PART 3: PROPOSED TEST STATISTIC
# =============================================================================


def branch_calibrated_test_statistic(
    theta_L: np.ndarray,
    theta_R: np.ndarray,
    n_L: int,
    n_R: int,
    b_L: float,
    b_R: float,
    expected_div_fn,
) -> tuple:
    """Compute branch-length calibrated test statistic.

    T = (D_obs - E[D | b_L, b_R]) / SE(D)

    Under H₀: T ~ N(0,1) approximately
    Under H₁: T >> 0 (observed divergence exceeds expected)

    Returns: (T_statistic, p_value_one_sided)
    """
    d, K = theta_L.shape

    # Observed divergence
    metrics = compute_divergence_metrics(theta_L, theta_R)
    D_obs = metrics["l2_squared_mean"]

    # Expected divergence from calibration
    D_expected = expected_div_fn(b_L, b_R)

    # Sampling variance of D_obs
    # For L2 difference: Var(θ̂_L - θ̂_R) ≈ Var(θ̂_L) + Var(θ̂_R)
    # Var(θ̂) = θ(1-θ)/n for each category

    # Pooled variance estimate
    theta_pool = 0.5 * (theta_L + theta_R)
    var_per_category = theta_pool * (1 - theta_pool) * (1 / n_L + 1 / n_R)

    # Variance of squared difference is complex, use simple approximation
    # SE ≈ sqrt(2 * mean(var)) for sum of squared differences
    SE = np.sqrt(2 * var_per_category.mean() / d)

    if SE < 1e-10:
        SE = 1e-10

    T = (D_obs - D_expected) / SE
    p_value = 1 - stats.norm.cdf(T)  # One-sided: excess divergence

    return T, p_value, D_obs, D_expected, SE


# =============================================================================
# PART 4: COMPARISON WITH CURRENT TEST
# =============================================================================


def current_wald_test(
    theta_L: np.ndarray, theta_R: np.ndarray, n_L: int, n_R: int
) -> tuple:
    """Current Wald chi-square test (ignores branch length).

    H₀: θ_L = θ_R
    T = Σ_j (θ̂_L^j - θ̂_R^j)² / Var(θ̂_L^j - θ̂_R^j) ~ χ²_d
    """
    d, K = theta_L.shape

    # Flatten to 1D for test
    diff = (theta_L - theta_R).flatten()

    # Pooled variance
    theta_pool = 0.5 * (theta_L + theta_R)
    var = (theta_pool * (1 - theta_pool) * (1 / n_L + 1 / n_R)).flatten()
    var = np.maximum(var, 1e-10)

    # Chi-square statistic
    chi2 = np.sum(diff**2 / var)
    df = len(diff)
    p_value = 1 - stats.chi2.cdf(chi2, df)

    return chi2, df, p_value


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def main():
    print("=" * 80)
    print("BRANCH-LENGTH CALIBRATED DIVERGENCE TEST: MATHEMATICAL DERIVATION")
    print("=" * 80)

    # Step 1: Calibrate expected divergence
    print("\n" + "=" * 80)
    print("STEP 1: Empirical Calibration of Expected Divergence")
    print("=" * 80)

    calibration_df = calibrate_expected_divergence(
        n_trials=30, n_samples=50, n_features=100, K=4
    )

    print("\nCalibration Table:")
    print(calibration_df.to_string(index=False))

    # Fit model: E[D] = f(total_branch)
    # Try: E[D] = a * (1 - exp(-c * b)) + sampling_variance

    from scipy.optimize import curve_fit

    def div_model(b, a, c, d):
        """Expected divergence as function of total branch length."""
        return a * (1 - np.exp(-c * b)) + d

    popt, pcov = curve_fit(
        div_model,
        calibration_df["total_branch"],
        calibration_df["expected_l2_mean"],
        p0=[0.1, 1.0, 0.01],
        maxfev=5000,
    )
    a, c, d = popt

    print(f"\n--- Fitted Model ---")
    print(f"E[L2_mean | b] = {a:.4f} * (1 - exp(-{c:.4f} * b)) + {d:.4f}")

    # Predict and check fit
    calibration_df["predicted"] = div_model(calibration_df["total_branch"], *popt)
    calibration_df["residual"] = (
        calibration_df["expected_l2_mean"] - calibration_df["predicted"]
    )
    r_squared = (
        1
        - (calibration_df["residual"] ** 2).sum()
        / (
            (
                calibration_df["expected_l2_mean"]
                - calibration_df["expected_l2_mean"].mean()
            )
            ** 2
        ).sum()
    )
    print(f"R² = {r_squared:.4f}")

    # Create expected divergence function
    def expected_div(b_L, b_R):
        total_b = b_L + b_R
        return div_model(total_b, *popt)

    # Step 2: Compare tests on simulated data
    print("\n" + "=" * 80)
    print("STEP 2: Comparison of Tests on Simulated Data")
    print("=" * 80)

    print("\nScenario A: Siblings from SAME ancestor (null should NOT reject)")
    print("-" * 70)

    scenarios_null = []
    for b in [0.1, 0.3, 0.5, 1.0]:
        for trial in range(10):
            rng = np.random.RandomState(42 + trial)
            theta_L, theta_R, _, _ = generate_evolved_siblings(50, 100, b, b, 4, rng)

            # Current test
            chi2, df, p_current = current_wald_test(theta_L, theta_R, 50, 50)

            # Proposed test
            T, p_proposed, D_obs, D_exp, SE = branch_calibrated_test_statistic(
                theta_L, theta_R, 50, 50, b, b, expected_div
            )

            scenarios_null.append(
                {
                    "branch": b,
                    "trial": trial,
                    "current_p": p_current,
                    "proposed_p": p_proposed,
                    "D_obs": D_obs,
                    "D_exp": D_exp,
                }
            )

    df_null = pd.DataFrame(scenarios_null)

    # False positive rate at alpha=0.05
    for b in [0.1, 0.3, 0.5, 1.0]:
        subset = df_null[df_null["branch"] == b]
        fp_current = (subset["current_p"] < 0.05).mean()
        fp_proposed = (subset["proposed_p"] < 0.05).mean()
        print(
            f"Branch={b:.1f}: Current FPR={fp_current:.0%}, Proposed FPR={fp_proposed:.0%}"
        )

    print("\nScenario B: Siblings from DIFFERENT ancestors (should reject)")
    print("-" * 70)

    scenarios_alt = []
    for b in [0.1, 0.3, 0.5, 1.0]:
        for trial in range(10):
            rng = np.random.RandomState(42 + trial)

            # Generate from DIFFERENT ancestors (add extra divergence)
            ancestor1 = rng.randint(0, 4, size=100)
            ancestor2 = rng.randint(0, 4, size=100)  # Different ancestor!

            def evolve(seq, branch, rng):
                K = 4
                P = np.full(
                    (K, K),
                    ((K - 1) / K) * (1 - np.exp(-K * branch / (K - 1))) / (K - 1),
                )
                np.fill_diagonal(
                    P, 1 / K + ((K - 1) / K) * np.exp(-K * branch / (K - 1))
                )
                return np.array([rng.choice(K, p=P[s]) for s in seq])

            sib_L = evolve(ancestor1, b, rng)
            sib_R = evolve(ancestor2, b, rng)

            terminal = 0.01
            samples_L = np.array([evolve(sib_L, terminal, rng) for _ in range(50)])
            samples_R = np.array([evolve(sib_R, terminal, rng) for _ in range(50)])

            theta_L = np.zeros((100, 4))
            theta_R = np.zeros((100, 4))
            for j in range(100):
                for c in range(4):
                    theta_L[j, c] = (samples_L[:, j] == c).mean()
                    theta_R[j, c] = (samples_R[:, j] == c).mean()

            # Current test
            chi2, df, p_current = current_wald_test(theta_L, theta_R, 50, 50)

            # Proposed test
            T, p_proposed, D_obs, D_exp, SE = branch_calibrated_test_statistic(
                theta_L, theta_R, 50, 50, b, b, expected_div
            )

            scenarios_alt.append(
                {
                    "branch": b,
                    "trial": trial,
                    "current_p": p_current,
                    "proposed_p": p_proposed,
                    "D_obs": D_obs,
                    "D_exp": D_exp,
                }
            )

    df_alt = pd.DataFrame(scenarios_alt)

    # True positive rate at alpha=0.05
    for b in [0.1, 0.3, 0.5, 1.0]:
        subset = df_alt[df_alt["branch"] == b]
        tp_current = (subset["current_p"] < 0.05).mean()
        tp_proposed = (subset["proposed_p"] < 0.05).mean()
        print(
            f"Branch={b:.1f}: Current TPR={tp_current:.0%}, Proposed TPR={tp_proposed:.0%}"
        )

    print("\n" + "=" * 80)
    print("STEP 3: Mathematical Formulation")
    print("=" * 80)
    print("""
CURRENT TEST (ignores branch length):
-------------------------------------
H₀: θ_L = θ_R
T = Σⱼ (θ̂_L^j - θ̂_R^j)² / Var(θ̂_L^j - θ̂_R^j)
Under H₀: T ~ χ²_d

PROBLEM: At high branch lengths, siblings WILL differ due to
evolutionary drift even if they came from the same ancestor.
This leads to high false positive rates.


PROPOSED TEST (calibrated by branch length):
--------------------------------------------
H₀: E[D_obs | b_L, b_R] = expected evolutionary divergence
H₁: D_obs > E[D | b_L, b_R] (true cluster separation)

T = (D_obs - E[D | b_L, b_R]) / SE(D_obs)
Under H₀: T ~ N(0,1) approximately

The expected divergence function is calibrated empirically:
E[L2_mean | b] ≈ a * (1 - exp(-c * b)) + baseline

Where b = b_L + b_R (total branch length from common ancestor)


KEY INSIGHT:
------------
Branch length tells us how much divergence to EXPECT.
We only flag as significant when observed >> expected.

This should:
1. Reduce false positives at high branch lengths
2. Maintain power to detect true cluster separation
3. Properly account for evolutionary structure
""")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.errorbar(
        calibration_df["total_branch"],
        calibration_df["expected_l2_mean"],
        yerr=calibration_df["std_l2_mean"],
        fmt="bo-",
        capsize=3,
        label="Empirical",
    )
    b_range = np.linspace(0, 4, 100)
    ax.plot(b_range, div_model(b_range, *popt), "r-", linewidth=2, label="Fitted model")
    ax.set_xlabel("Total Branch Length (b_L + b_R)")
    ax.set_ylabel("E[L2_mean]")
    ax.set_title("Expected Divergence Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        calibration_df["total_branch"],
        calibration_df["expected_hamming"],
        "g-o",
        label="Theory (Hamming)",
    )
    ax.plot(
        calibration_df["total_branch"],
        calibration_df["expected_l2_mean"] * 100,
        "b-o",
        label="Empirical L2 (×100)",
    )
    ax.set_xlabel("Total Branch Length")
    ax.set_ylabel("Expected Distance")
    ax.set_title("Theoretical vs Empirical")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    fp_current = df_null.groupby("branch")["current_p"].apply(
        lambda x: (x < 0.05).mean()
    )
    fp_proposed = df_null.groupby("branch")["proposed_p"].apply(
        lambda x: (x < 0.05).mean()
    )
    x = fp_current.index
    ax.bar(
        x - 0.1, fp_current.values, 0.2, label="Current Test", color="red", alpha=0.7
    )
    ax.bar(
        x + 0.1, fp_proposed.values, 0.2, label="Proposed Test", color="blue", alpha=0.7
    )
    ax.axhline(0.05, color="black", linestyle="--", label="Nominal α=0.05")
    ax.set_xlabel("Branch Length")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR Under Null (Same Ancestor)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    tp_current = df_alt.groupby("branch")["current_p"].apply(
        lambda x: (x < 0.05).mean()
    )
    tp_proposed = df_alt.groupby("branch")["proposed_p"].apply(
        lambda x: (x < 0.05).mean()
    )
    ax.bar(
        x - 0.1, tp_current.values, 0.2, label="Current Test", color="red", alpha=0.7
    )
    ax.bar(
        x + 0.1, tp_proposed.values, 0.2, label="Proposed Test", color="blue", alpha=0.7
    )
    ax.set_xlabel("Branch Length")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("TPR Under Alternative (Different Ancestors)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(repo_root / "results" / "branch_calibrated_test.png", dpi=150)
    print(f"\nPlot saved to: results/branch_calibrated_test.png")


if __name__ == "__main__":
    main()
