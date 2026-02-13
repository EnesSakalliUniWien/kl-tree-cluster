"""
Demonstrates the mathematical issues with variance calculations.

This script shows:
1. How the nested variance formula works correctly for valid trees
2. How degenerate trees (n_child ≈ n_parent) create numerical instability
3. The mathematical derivation and why the fallback was wrong
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")


def compute_nested_variance(theta, n_child, n_parent):
    """Compute the nested variance for child-parent comparison."""
    nested_factor = 1.0 / n_child - 1.0 / n_parent
    return theta * (1 - theta) * nested_factor


def compute_fallback_variance(theta, n_child):
    """Compute the WRONG fallback variance (used before fix)."""
    return theta * (1 - theta) / n_child


def plot_variance_comparison():
    """Plot how variance behaves for different tree structures."""
    theta = 0.5  # Maximum variance case
    n_parent = 100

    # Range of child sizes from 1 to 100
    n_child_values = np.arange(1, n_parent + 1)

    nested_variances = []
    fallback_variances = []

    for n_child in n_child_values:
        if n_child < n_parent:
            nested_var = compute_nested_variance(theta, n_child, n_parent)
        else:
            nested_var = np.nan  # Invalid

        fallback_var = compute_fallback_variance(theta, n_child)

        nested_variances.append(nested_var)
        fallback_variances.append(fallback_var)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Variance comparison
    ax = axes[0, 0]
    valid_mask = np.array(n_child_values) < n_parent
    ax.semilogy(
        n_child_values[valid_mask],
        np.array(nested_variances)[valid_mask],
        "b-",
        linewidth=2,
        label="Correct nested variance",
    )
    ax.semilogy(
        n_child_values,
        fallback_variances,
        "r--",
        linewidth=2,
        label="Wrong fallback variance",
    )
    ax.axvline(
        x=n_parent, color="k", linestyle=":", alpha=0.5, label=f"n_parent={n_parent}"
    )
    ax.set_xlabel("Child sample size (n_child)", fontsize=12)
    ax.set_ylabel("Variance", fontsize=12)
    ax.set_title("Variance Comparison: Correct vs Fallback", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Variance inflation factor
    ax = axes[0, 1]
    inflation_factors = []
    for n_child in n_child_values[:-1]:  # Exclude n_child == n_parent
        if n_child < n_parent:
            nested_var = compute_nested_variance(theta, n_child, n_parent)
            fallback_var = compute_fallback_variance(theta, n_child)
            inflation = fallback_var / nested_var if nested_var > 0 else np.inf
            inflation_factors.append(inflation)

    ax.semilogy(n_child_values[:-1], inflation_factors, "g-", linewidth=2)
    ax.set_xlabel("Child sample size (n_child)", fontsize=12)
    ax.set_ylabel("Variance inflation factor", fontsize=12)
    ax.set_title("Fallback Variance is X times more conservative", fontsize=14)
    ax.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="No inflation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Z-score comparison
    ax = axes[1, 0]
    delta_theta = 0.1  # Difference between child and parent

    z_correct = []
    z_fallback = []
    for n_child in n_child_values[:-1]:
        nested_var = compute_nested_variance(theta, n_child, n_parent)
        fallback_var = compute_fallback_variance(theta, n_child)

        z_corr = delta_theta / np.sqrt(nested_var)
        z_fall = delta_theta / np.sqrt(fallback_var)

        z_correct.append(z_corr)
        z_fallback.append(z_fall)

    ax.plot(n_child_values[:-1], z_correct, "b-", linewidth=2, label="Correct z-score")
    ax.plot(
        n_child_values[:-1], z_fallback, "r--", linewidth=2, label="Fallback z-score"
    )
    ax.set_xlabel("Child sample size (n_child)", fontsize=12)
    ax.set_ylabel("Z-score", fontsize=12)
    ax.set_title(f"Z-score for θ_diff = {delta_theta}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistical power loss
    ax = axes[1, 1]
    alpha = 0.05
    z_critical = 1.96  # Two-tailed test

    power_correct = []
    power_fallback = []

    for n_child in n_child_values[:-1]:
        nested_var = compute_nested_variance(theta, n_child, n_parent)
        fallback_var = compute_fallback_variance(theta, n_child)

        z_corr = delta_theta / np.sqrt(nested_var)
        z_fall = delta_theta / np.sqrt(fallback_var)

        # Power = P(reject H0 | H1 true) ≈ P(Z > z_critical - z_true)
        power_corr = 1 - 0.5 * (1 + np.sign(z_critical - z_corr))  # Simplified
        power_fall = 1 - 0.5 * (1 + np.sign(z_critical - z_fall))

        power_correct.append(power_corr if z_corr > z_critical else 0.05)
        power_fallback.append(power_fall if z_fall > z_critical else 0.05)

    ax.plot(
        n_child_values[:-1], power_correct, "b-", linewidth=2, label="Correct power"
    )
    ax.plot(
        n_child_values[:-1], power_fallback, "r--", linewidth=2, label="Fallback power"
    )
    ax.axhline(y=0.05, color="k", linestyle=":", alpha=0.5, label="α = 0.05")
    ax.set_xlabel("Child sample size (n_child)", fontsize=12)
    ax.set_ylabel("Statistical power", fontsize=12)
    ax.set_title("Power Loss Due to Fallback", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/Users/berksakalli/Projects/kl-te-cluster/debug_scripts/variance_comparison.png",
        dpi=150,
    )
    print("Plot saved to debug_scripts/variance_comparison.png")


def demonstrate_covariance_structure():
    """Demonstrate the covariance structure mathematically."""
    print("=" * 70)
    print("MATHEMATICAL DERIVATION")
    print("=" * 70)
    print()

    print("Setup:")
    print("  - Parent P with n_p samples")
    print("  - Child C with n_c samples, where C ⊂ P")
    print("  - True parameter θ for all samples")
    print()

    print("Variance of parent estimate:")
    print("  Var(θ̂_p) = θ(1-θ) / n_p")
    print()

    print("Variance of child estimate:")
    print("  Var(θ̂_c) = θ(1-θ) / n_c")
    print()

    print("Covariance (since C ⊂ P, estimates share samples):")
    print("  Cov(θ̂_c, θ̂_p) = (n_c/n_p) × Var(θ̂_c)")
    print("                = (n_c/n_p) × θ(1-θ)/n_c")
    print("                = θ(1-θ) / n_p")
    print()

    print("Variance of difference:")
    print("  Var(θ̂_c - θ̂_p) = Var(θ̂_c) + Var(θ̂_p) - 2×Cov(θ̂_c, θ̂_p)")
    print("                 = θ(1-θ)/n_c + θ(1-θ)/n_p - 2×θ(1-θ)/n_p")
    print("                 = θ(1-θ) × (1/n_c + 1/n_p - 2/n_p)")
    print("                 = θ(1-θ) × (1/n_c - 1/n_p)")
    print()

    print("=" * 70)
    print("WHEN DOES THIS BREAK?")
    print("=" * 70)
    print()

    print("The formula requires: 1/n_c - 1/n_p > 0")
    print("Which means: n_c < n_p")
    print()

    print("When n_c = n_p (child IS the parent):")
    print("  Variance = θ(1-θ) × (1/n_p - 1/n_p) = 0")
    print("  → Division by zero in z-score!")
    print()

    print("When n_c > n_p (impossible in valid tree):")
    print("  Variance becomes NEGATIVE!")
    print("  → This indicates a BUG in tree construction")
    print()

    print("=" * 70)
    print("THE WRONG FALLBACK")
    print("=" * 70)
    print()

    print("Old code did:")
    print("  if nested_factor <= 0:")
    print("      nested_factor = 1.0 / n_child  # WRONG!")
    print()

    print("This changes the test from:")
    print("  H_0: θ_c = θ_p  (child same as parent)")
    print("To:")
    print("  H_0: θ_c = 0.5  (child at chance level)")
    print()

    print("These are DIFFERENT hypotheses with DIFFERENT Type I error rates!")
    print()


def numerical_examples():
    """Show numerical examples of the issue."""
    print("=" * 70)
    print("NUMERICAL EXAMPLES")
    print("=" * 70)
    print()

    theta = 0.5
    delta = 0.1  # θ_c = 0.6, θ_p = 0.5

    examples = [
        (10, 100, "Normal case"),
        (50, 100, "Large child"),
        (90, 100, "Degenerate case 1"),
        (99, 100, "Degenerate case 2"),
    ]

    for n_c, n_p, desc in examples:
        var_correct = compute_nested_variance(theta, n_c, n_p)
        var_fallback = compute_fallback_variance(theta, n_c)

        z_correct = delta / np.sqrt(var_correct)
        z_fallback = delta / np.sqrt(var_fallback)

        inflation = var_fallback / var_correct

        print(f"{desc}: n_c={n_c}, n_p={n_p}")
        print(f"  Correct variance:   {var_correct:.6f}")
        print(f"  Fallback variance:  {var_fallback:.6f}")
        print(f"  Inflation factor:   {inflation:.2f}x")
        print(f"  Correct z-score:    {z_correct:.2f}")
        print(f"  Fallback z-score:   {z_fallback:.2f}")
        print(
            f"  Power loss:         {((z_correct - z_fallback) / z_correct * 100):.1f}%"
        )
        print()


if __name__ == "__main__":
    demonstrate_covariance_structure()
    numerical_examples()
    plot_variance_comparison()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("The fix ensures that:")
    print("1. Invalid tree structures are detected immediately")
    print("2. Silent fallback to wrong variance formula is prevented")
    print("3. Test validity is maintained (correct Type I error rate)")
    print()
    print("Degenerate trees (n_c ≈ n_p) indicate:")
    print("- Poor clustering algorithm choice")
    print("- Data quality issues")
    print("- Incorrect tree construction")
    print()
    print("These should be FIXED at the source, not masked by fallbacks.")
