"""
Purpose: Summary: The real issue is STATISTICAL POWER, not asymmetric comparison.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/diagnostics/q_statistical_power_issue_summary__power__diagnostic.py
"""

import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import jensenshannon


def demonstrate_power_issue():
    """Show how power increases with sample size."""
    print("=" * 70)
    print("Statistical Power vs Sample Size")
    print("=" * 70)

    dist_A = np.array([0.20, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05])
    dist_B = np.array([0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20])

    print("\nDistributions:")
    print(f"  A: {dist_A}")
    print(f"  B: {dist_B}")
    print(f"  Difference: {np.abs(dist_A - dist_B).mean():.3f} average per feature")

    print("\nPower analysis (Wald test):")
    print(f"{'n':<6} {'statistic':<12} {'p-value':<12} {'significant?'}")
    print("-" * 45)

    for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        pooled = 0.5 * (dist_A + dist_B)  # Equal weights
        var = pooled * (1 - pooled) * (2.0 / n)  # (1/n + 1/n)
        z = (dist_A - dist_B) / np.sqrt(var)
        stat = np.sum(z**2)
        df = len(z)
        p_val = chi2.sf(stat, df=df)
        sig = "✓ YES" if p_val < 0.05 else "✗ NO"
        print(f"{n:<6} {stat:<12.2f} {p_val:<12.6f} {sig}")

    print("\n" + "-" * 70)
    print("CONCLUSION: Need n ≥ 32 samples per group to detect this difference!")
    print("-" * 70)


def compare_to_distance_based():
    """Compare p-value test to distance-based approach."""
    print("\n" + "=" * 70)
    print("Alternative: Distance-Based Criterion")
    print("=" * 70)

    dist_A = np.array([0.20, 0.20, 0.20, 0.20, 0.05, 0.05, 0.05, 0.05])
    dist_B = np.array([0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20])
    dist_C = dist_A + np.random.normal(0, 0.01, 8)  # Very similar to A
    dist_C = np.clip(dist_C, 0.01, 0.99)

    # Normalize
    dist_A = dist_A / dist_A.sum()
    dist_B = dist_B / dist_B.sum()
    dist_C = dist_C / dist_C.sum()

    print("\nDistributions (normalized):")
    print(f"  A: {dist_A}")
    print(f"  B: {dist_B}")
    print(f"  C: {dist_C} (similar to A)")

    # Jensen-Shannon divergence
    js_AB = jensenshannon(dist_A, dist_B)
    js_AC = jensenshannon(dist_A, dist_C)
    js_BC = jensenshannon(dist_B, dist_C)

    print("\nJensen-Shannon distances:")
    print(f"  A vs B: {js_AB:.4f} (different distributions)")
    print(f"  A vs C: {js_AC:.4f} (similar distributions)")
    print(f"  B vs C: {js_BC:.4f}")

    print("\nWith distance threshold = 0.3:")
    threshold = 0.3
    print(f"  A vs B: {js_AB:.4f} > {threshold} → DIFFERENT ✓")
    print(f"  A vs C: {js_AC:.4f} < {threshold} → SAME ✓")

    print("""
    ADVANTAGE: Distance-based approach works regardless of sample size!
    It measures the actual distance between distributions, not whether
    we have enough data to prove they're different.
    """)


if __name__ == "__main__":
    demonstrate_power_issue()
    compare_to_distance_based()
