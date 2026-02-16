"""
Purpose: Derive and probe a refined branch-calibrated divergence test formulation.
Inputs: Synthetic evolutionary simulations and calibration settings.
Outputs: Console/plot diagnostics for calibrated test behavior.
Expected runtime: ~30-180 seconds.
How to run: python debug_scripts/branch_length/methods/q_branch_calibrated_test_derivation__branch_length__refined.py
"""

#!/usr/bin/env python3
"""
Option B: Refined Branch-Length Calibrated Test

The previous attempt had inflated FPR because the SE was underestimated.
Here we derive the correct variance of D_obs under the null.

Key insight: Under the null, D_obs has BOTH:
1. Variance from finite sampling (θ̂ ≠ θ)
2. Variance from evolutionary stochasticity (θ_L ≠ θ_R even from same ancestor)

The correct null distribution should account for BOTH sources.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))


def generate_evolved_siblings(
    n_samples: int,
    n_features: int,
    b_L: float,
    b_R: float,
    K: int = 4,
    rng: np.random.RandomState = None,
):
    """Generate two sibling populations that evolved from a common ancestor."""
    if rng is None:
        rng = np.random.RandomState(42)

    ancestor = rng.randint(0, K, size=n_features)

    def evolve(seq, b):
        P = np.full((K, K), ((K - 1) / K) * (1 - np.exp(-K * b / (K - 1))) / (K - 1))
        np.fill_diagonal(P, 1 / K + ((K - 1) / K) * np.exp(-K * b / (K - 1)))
        return np.array([rng.choice(K, p=P[s]) for s in seq])

    sib_L_ancestor = evolve(ancestor, b_L)
    sib_R_ancestor = evolve(ancestor, b_R)

    terminal = 0.01
    samples_L = np.array([evolve(sib_L_ancestor, terminal) for _ in range(n_samples)])
    samples_R = np.array([evolve(sib_R_ancestor, terminal) for _ in range(n_samples)])

    theta_L = np.zeros((n_features, K))
    theta_R = np.zeros((n_features, K))

    for j in range(n_features):
        for c in range(K):
            theta_L[j, c] = (samples_L[:, j] == c).mean()
            theta_R[j, c] = (samples_R[:, j] == c).mean()

    return theta_L, theta_R


def compute_l2_mean(theta_L: np.ndarray, theta_R: np.ndarray) -> float:
    """Mean L2 squared divergence per feature."""
    return np.sum((theta_L - theta_R) ** 2) / theta_L.shape[0]


# =============================================================================
# REFINED APPROACH: Use full null distribution calibration
# =============================================================================


def calibrate_null_distribution(
    b: float,
    n_samples: int = 50,
    n_features: int = 100,
    n_trials: int = 200,
    K: int = 4,
) -> dict:
    """Calibrate the FULL null distribution at a given branch length.

    Returns mean and std of D_obs under the null (same ancestor).
    """
    d_values = []

    for trial in range(n_trials):
        rng = np.random.RandomState(trial)
        theta_L, theta_R = generate_evolved_siblings(
            n_samples, n_features, b, b, K, rng
        )
        d_values.append(compute_l2_mean(theta_L, theta_R))

    return {
        "branch": b,
        "mean": np.mean(d_values),
        "std": np.std(d_values),
        "p05": np.percentile(d_values, 5),
        "p95": np.percentile(d_values, 95),
        "p99": np.percentile(d_values, 99),
    }


def build_calibration_table():
    """Build calibration table for null distribution at various branch lengths."""
    branches = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]

    results = []
    print("Calibrating null distributions...")
    for b in branches:
        print(f"  b = {b:.2f}")
        results.append(calibrate_null_distribution(b, n_trials=300))

    return pd.DataFrame(results)


def fit_null_distribution_model(calib_df: pd.DataFrame) -> dict:
    """Fit parametric models for E[D] and Std[D] as functions of branch length."""

    # Fit mean: E[D | b] = a * (1 - exp(-c * b)) + d
    def mean_model(b, a, c, d):
        return a * (1 - np.exp(-c * b)) + d

    popt_mean, _ = curve_fit(
        mean_model,
        2 * calib_df["branch"],  # total branch
        calib_df["mean"],
        p0=[1.5, 1.0, 0.01],
        maxfev=5000,
    )

    # Fit std: Std[D | b] = e * b^f + g  (power law)
    def std_model(b, e, f, g):
        return e * np.power(b + 0.01, f) + g

    popt_std, _ = curve_fit(
        std_model,
        2 * calib_df["branch"],
        calib_df["std"],
        p0=[0.1, 0.5, 0.02],
        maxfev=5000,
    )

    return {
        "mean_params": popt_mean,
        "std_params": popt_std,
        "mean_model": mean_model,
        "std_model": std_model,
    }


def calibrated_z_test(
    theta_L: np.ndarray,
    theta_R: np.ndarray,
    b_L: float,
    b_R: float,
    model_params: dict,
) -> tuple:
    """Refined test using fully calibrated null distribution.

    Z = (D_obs - E[D | b]) / Std[D | b]

    Under H₀: Z ~ N(0,1)
    """
    D_obs = compute_l2_mean(theta_L, theta_R)

    total_b = b_L + b_R
    E_D = model_params["mean_model"](total_b, *model_params["mean_params"])
    Std_D = model_params["std_model"](total_b, *model_params["std_params"])

    Z = (D_obs - E_D) / Std_D
    p_value = 1 - stats.norm.cdf(Z)  # one-sided

    return Z, p_value, D_obs, E_D, Std_D


def test_with_true_different_clusters(
    b: float,
    divergence: float,
    n_samples: int = 50,
    n_features: int = 100,
    K: int = 4,
    rng: np.random.RandomState = None,
):
    """Generate siblings from DIFFERENT clusters (alternative hypothesis).

    divergence: fraction of sites that differ between ancestral sequences
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Two distinct ancestors
    ancestor_L = rng.randint(0, K, size=n_features)
    # Create ancestor_R by mutating divergence fraction of sites
    n_mut = int(divergence * n_features)
    ancestor_R = ancestor_L.copy()
    mut_sites = rng.choice(n_features, n_mut, replace=False)
    for s in mut_sites:
        options = [c for c in range(K) if c != ancestor_L[s]]
        ancestor_R[s] = rng.choice(options)

    def evolve(seq, branch, rng):
        P = np.full(
            (K, K), ((K - 1) / K) * (1 - np.exp(-K * branch / (K - 1))) / (K - 1)
        )
        np.fill_diagonal(P, 1 / K + ((K - 1) / K) * np.exp(-K * branch / (K - 1)))
        return np.array([rng.choice(K, p=P[s]) for s in seq])

    sib_L = evolve(ancestor_L, b, rng)
    sib_R = evolve(ancestor_R, b, rng)

    terminal = 0.01
    samples_L = np.array([evolve(sib_L, terminal, rng) for _ in range(n_samples)])
    samples_R = np.array([evolve(sib_R, terminal, rng) for _ in range(n_samples)])

    theta_L = np.zeros((n_features, K))
    theta_R = np.zeros((n_features, K))
    for j in range(n_features):
        for c in range(K):
            theta_L[j, c] = (samples_L[:, j] == c).mean()
            theta_R[j, c] = (samples_R[:, j] == c).mean()

    return theta_L, theta_R


def main():
    print("=" * 80)
    print("REFINED BRANCH-LENGTH CALIBRATED TEST")
    print("=" * 80)

    # Step 1: Full calibration
    print("\n--- Step 1: Calibrating Full Null Distribution ---")
    calib_df = build_calibration_table()
    print("\nCalibration Table:")
    print(calib_df.to_string(index=False))

    model = fit_null_distribution_model(calib_df)
    print(
        f"\nMean model: E[D|b] = {model['mean_params'][0]:.4f} * (1 - exp(-{model['mean_params'][1]:.4f} * b)) + {model['mean_params'][2]:.4f}"
    )
    print(
        f"Std model:  Std[D|b] = {model['std_params'][0]:.4f} * (b+0.01)^{model['std_params'][1]:.4f} + {model['std_params'][2]:.4f}"
    )

    # Step 2: Test under null
    print("\n" + "=" * 80)
    print("Step 2: Test Under NULL (same ancestor)")
    print("=" * 80)

    null_results = []
    for b in [0.1, 0.3, 0.5, 1.0]:
        for trial in range(100):
            rng = np.random.RandomState(5000 + trial)
            theta_L, theta_R = generate_evolved_siblings(50, 100, b, b, 4, rng)
            Z, p, D_obs, E_D, Std_D = calibrated_z_test(theta_L, theta_R, b, b, model)
            null_results.append({"branch": b, "Z": Z, "p": p, "D_obs": D_obs})

    null_df = pd.DataFrame(null_results)

    print("\nFalse Positive Rate at α=0.05:")
    for b in [0.1, 0.3, 0.5, 1.0]:
        subset = null_df[null_df["branch"] == b]
        fpr = (subset["p"] < 0.05).mean()
        print(f"  b={b:.1f}: FPR = {fpr:.1%} (expected ~5%)")

    # Step 3: Test under alternative
    print("\n" + "=" * 80)
    print("Step 3: Test Under ALTERNATIVE (different ancestors)")
    print("=" * 80)

    for true_divergence in [0.1, 0.3, 0.5]:
        print(f"\n--- True Divergence = {true_divergence:.0%} ---")
        alt_results = []
        for b in [0.1, 0.3, 0.5, 1.0]:
            for trial in range(50):
                rng = np.random.RandomState(10000 + trial)
                theta_L, theta_R = test_with_true_different_clusters(
                    b, true_divergence, 50, 100, 4, rng
                )
                Z, p, D_obs, E_D, Std_D = calibrated_z_test(
                    theta_L, theta_R, b, b, model
                )
                alt_results.append(
                    {"branch": b, "divergence": true_divergence, "Z": Z, "p": p}
                )

        alt_df = pd.DataFrame(alt_results)
        for b in [0.1, 0.3, 0.5, 1.0]:
            subset = alt_df[
                (alt_df["branch"] == b) & (alt_df["divergence"] == true_divergence)
            ]
            tpr = (subset["p"] < 0.05).mean()
            print(f"  b={b:.1f}: Power = {tpr:.0%}")

    # Step 4: Key insights
    print("\n" + "=" * 80)
    print("KEY MATHEMATICAL FORMULATION")
    print("=" * 80)
    print("""
THE REFINED TEST:
-----------------

1. CALIBRATION PHASE (done once, offline):
   - For each branch length b, simulate M pairs of siblings 
     that evolved from the SAME ancestor
   - Record mean and std of L2 divergence: μ(b), σ(b)
   - Fit parametric models: μ(b) = a(1 - e^(-cb)) + d
                            σ(b) = e·b^f + g

2. TESTING PHASE (for each split decision):
   - Observe θ̂_L, θ̂_R (empirical distributions of siblings)
   - Extract branch lengths b_L, b_R from tree
   - Compute D_obs = ||θ̂_L - θ̂_R||² / d
   - Compute expected: μ = μ(b_L + b_R)
   - Compute expected std: σ = σ(b_L + b_R)
   - Z-score: Z = (D_obs - μ) / σ
   - p-value: p = 1 - Φ(Z)  [one-sided, testing excess divergence]

3. DECISION:
   - Reject H₀ (split is real) if p < α
   - This tests whether observed divergence EXCEEDS what we'd 
     expect from evolutionary drift alone

INTERPRETATION:
---------------
- D_obs ≤ μ(b): Divergence is explained by branch length alone
  → siblings evolved from same ancestor → DON'T SPLIT
  
- D_obs >> μ(b): Divergence exceeds evolutionary expectation
  → siblings represent truly different populations → SPLIT

This is more principled than the current approach because:
1. It uses branch length to set the null expectation
2. It accounts for evolutionary stochasticity
3. The null hypothesis is "same evolutionary origin"
   rather than "identical distributions"
""")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.errorbar(
        calib_df["branch"] * 2,
        calib_df["mean"],
        yerr=calib_df["std"],
        fmt="bo-",
        capsize=3,
        label="Mean ± Std",
    )
    b_range = np.linspace(0, 4, 100)
    ax.plot(
        b_range,
        model["mean_model"](b_range, *model["mean_params"]),
        "r-",
        linewidth=2,
        label="Fitted μ(b)",
    )
    ax.fill_between(
        b_range,
        model["mean_model"](b_range, *model["mean_params"])
        - model["std_model"](b_range, *model["std_params"]),
        model["mean_model"](b_range, *model["mean_params"])
        + model["std_model"](b_range, *model["std_params"]),
        alpha=0.2,
        color="red",
    )
    ax.set_xlabel("Total Branch Length")
    ax.set_ylabel("E[D_obs | H₀]")
    ax.set_title("Null Distribution: Mean ± Std vs Branch Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(calib_df["branch"] * 2, calib_df["std"], "go-", label="Empirical Std")
    ax.plot(
        b_range,
        model["std_model"](b_range, *model["std_params"]),
        "r-",
        linewidth=2,
        label="Fitted σ(b)",
    )
    ax.set_xlabel("Total Branch Length")
    ax.set_ylabel("Std[D_obs | H₀]")
    ax.set_title("Null Distribution: Standard Deviation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for b in [0.1, 0.3, 0.5, 1.0]:
        subset = null_df[null_df["branch"] == b]
        ax.hist(subset["Z"], bins=20, alpha=0.5, label=f"b={b}", density=True)
    z_range = np.linspace(-4, 4, 100)
    ax.plot(z_range, stats.norm.pdf(z_range), "k-", linewidth=2, label="N(0,1)")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.set_title("Z-scores Under Null (should follow N(0,1))")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    fp_rates = null_df.groupby("branch")["p"].apply(lambda x: (x < 0.05).mean())
    ax.bar(fp_rates.index, fp_rates.values, width=0.15, color="blue", alpha=0.7)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=2, label="Nominal α=0.05")
    ax.set_xlabel("Branch Length")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR at α=0.05 (should be ~5%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.15)

    plt.tight_layout()
    plt.savefig(repo_root / "results" / "refined_branch_test.png", dpi=150)
    print(f"\nPlot saved to: results/refined_branch_test.png")


if __name__ == "__main__":
    main()
