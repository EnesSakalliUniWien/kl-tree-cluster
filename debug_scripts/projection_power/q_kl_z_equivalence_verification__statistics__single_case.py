"""
Purpose: Verify numerical equivalence/consistency between KL and z-score formulations.
Inputs: Synthetic or controlled inputs defined in-script.
Outputs: Console verification metrics and discrepancies.
Expected runtime: ~5-30 seconds.
How to run: python debug_scripts/projection_power/q_kl_z_equivalence_verification__statistics__single_case.py
"""

import numpy as np
from kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance import standardize_proportion_difference
from kl_clustering_analysis.information_metrics.kl_divergence.divergence_metrics import _kl_categorical_general

def run_comparison():
    print("=== Verification: Wald Statistic vs Scaled KL Divergence ===")
    print("Theory: Sum(Z^2) ≈ 2 * N * KL(P || Q)\n")

    # Case 1: Binary Feature
    # Parent (Q) is 50/50. Child (P) splits slightly to 55/45.
    # N_child = 100
    
    N = 100
    Q_prob = 0.5
    P_prob = 0.55
    
    print(f"Scenario 1: Binary Feature (N={N})")
    print(f"Parent (Q) = [{Q_prob}, {1-Q_prob}]")
    print(f"Child  (P) = [{P_prob}, {1-P_prob}]")

    # 1. Calculate KL Divergence
    # _kl_categorical_general expects array inputs. 
    # For binary 1D input, it treats them as Bernoulli parameters (p).
    kl_div = _kl_categorical_general(np.array([P_prob]), np.array([Q_prob]), eps=1e-10)
    scaled_kl = 2 * N * kl_div
    
    # 2. Calculate Z-scores (Wald Test)
    # We use standardize_proportion_difference
    # Important: In the Child-Parent test, we assume H0 is the Parent.
    # So the variance is calculated based on the PARENT's distribution Q used for both.
    # Wait, let's check assumptions in edge_significance.py vs standardize_proportion_difference.
    
    # In edge_significance.py:
    # var = parent_dist * (1 - parent_dist) / n_child  <-- Uses Parent variance
    # z = (child - parent) / sqrt(var)
    
    # Let's manually calculate Z first based on the formula in manuscript
    var = Q_prob * (1 - Q_prob) / N
    z_manual = (P_prob - Q_prob) / np.sqrt(var)
    sum_z_sq_manual = z_manual**2
    
    print(f"\n[Calculations]")
    print(f"KL Divergence: {kl_div[0]:.6f}")
    print(f"2 * N * KL:    {scaled_kl[0]:.6f}")
    print(f"Z-score:       {z_manual:.6f}")
    print(f"Sum(Z^2):      {sum_z_sq_manual:.6f}")
    
    diff = abs(scaled_kl[0] - sum_z_sq_manual)
    print(f"\nDifference:    {diff:.6f}")
    print(f"Ratio:         {sum_z_sq_manual / scaled_kl[0]:.6f}")
    
    print("-" * 50)
    
    # Case 2: Multi-class (Categorical)
    # 3 categories: A, B, C
    print(f"Scenario 2: Categorical Feature (3 classes, N={N})")
    Q_cat = np.array([[0.33, 0.33, 0.34]]) # Parent
    P_cat = np.array([[0.35, 0.31, 0.34]]) # Child
    
    print(f"Parent (Q) = {Q_cat}")
    print(f"Child  (P) = {P_cat}")
    
    # KL
    kl_cat = _kl_categorical_general(P_cat, Q_cat, eps=1e-10)
    scaled_kl_cat = 2 * N * kl_cat
    
    # Z-scores (Manual Calculation for transparency)
    # Z_i = (P_i - Q_i) / sqrt(Q_i(1-Q_i)/N)
    
    # Note: simple term-wise Z-score assumes independence, but categories sum to 1.
    # Our code calculates Z for *each* category and sums them.
    # Let's see what the code does.
    
    # The code in edge_significance.py:
    # var = parent_dist * (1 - parent_dist) / n_child
    # z = (child_dist - parent_dist) / np.sqrt(var)
    # return z.ravel()
    
    var_cat = Q_cat * (1 - Q_cat) / N
    z_cat = (P_cat - Q_cat) / np.sqrt(var_cat)
    sum_z_sq_cat = np.sum(z_cat**2)
    
    print(f"\n[Calculations]")
    print(f"KL Divergence: {kl_cat[0]:.6f}")
    print(f"2 * N * KL:    {scaled_kl_cat[0]:.6f}")
    print(f"Sum(Z^2):      {sum_z_sq_cat:.6f}") # Sum of squared Zs over all categories
    
    print(f"Ratio:         {sum_z_sq_cat / scaled_kl_cat[0]:.6f}")
    
    # EXPLANATION OF RATIO FOR CATEGORICAL:
    # The sum of Z^2 typically overestimates the Chi2 statistic for categorical 
    # because of the correlation between categories (they sum to 1).
    # The true Pearson Chi-square statistic is sum((O-E)^2 / E).
    # O = N*P, E = N*Q.
    # (NP - NQ)^2 / NQ = N^2(P-Q)^2 / NQ = N * (P-Q)^2 / Q
    #
    # Our Z score is: (P-Q) / sqrt(Q(1-Q)/N)
    # Z^2 = (P-Q)^2 / (Q(1-Q)/N) = N * (P-Q)^2 / (Q(1-Q))
    #
    # Notice the denominator!
    # Pearson/KL uses Q.
    # Our Z-score uses Q(1-Q).
    # Since (1-Q) < 1, our Z-score denominator is smaller, so Z^2 is LARGER.
    # This suggests our "Naive Z-score" approach is essentially a conservative upper bound?
    # Or implies we treat features as independent binary variables (one-vs-rest).
    
    print("\nNote: For categorical, our Z-score formula treats each category as a binary One-vs-Rest problem.")
    print("This produces a slightly different statistic than the pure Pearson Chi-Square, but is monotonically related.")

if __name__ == "__main__":
    run_comparison()
