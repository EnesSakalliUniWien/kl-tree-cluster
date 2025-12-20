"""Debug script for random projection chi-square formula."""

import numpy as np
from scipy.stats import chi2
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    generate_projection_matrix,
    projected_euclidean_distance_squared,
)

np.random.seed(42)

# Simulate comparing 2 samples vs 2 samples in 1200 dimensions
# Under H₀, both groups should have same θ
d = 1200
n_left, n_right = 2, 2
n_eff = (2 * n_left * n_right) / (n_left + n_right)
k = 10

print(f"Scenario: n_left={n_left}, n_right={n_right}, d={d}, k={k}")
print()

# Generate true theta (same for both groups under H₀)
theta_true = np.random.uniform(0.3, 0.7, d)

# Estimate theta from samples (Binomial sampling variance)
left_samples = np.random.binomial(1, theta_true, size=(n_left, d))
right_samples = np.random.binomial(1, theta_true, size=(n_right, d))

theta_left = left_samples.mean(axis=0)
theta_right = right_samples.mean(axis=0)

print(f"True θ mean: {theta_true.mean():.3f}")
print(f"Left θ̂ mean: {theta_left.mean():.3f}")
print(f"Right θ̂ mean: {theta_right.mean():.3f}")

# Original squared distance
orig_dist_sq = np.sum((theta_left - theta_right) ** 2)
print(f"||Δθ̂||² = {orig_dist_sq:.4f}")

# Expected value under H₀:
# E[||Δθ̂||²] = Σ Var[θ̂_left - θ̂_right]
#            = Σ θ(1-θ) * (1/n_left + 1/n_right)
expected_dist_sq = np.sum(theta_true * (1 - theta_true)) * (1 / n_left + 1 / n_right)
print(f"E[||Δθ̂||²] under H₀ = {expected_dist_sq:.4f}")

# Project
R = generate_projection_matrix(d, k, random_state=42)
proj_dist_sq = projected_euclidean_distance_squared(theta_left, theta_right, R)
print(f"||R·Δθ̂||² = {proj_dist_sq:.4f}")

# Variance
pooled = 0.5 * (theta_left + theta_right)
avg_var = np.mean(pooled * (1 - pooled))
print(f"avg_var = {avg_var:.4f}")

# Current chi-square statistic
chi_sq_current = n_eff * proj_dist_sq / (k * avg_var)
print(f"Current χ² = n_eff * proj_dist² / (k * avg_var) = {chi_sq_current:.2f}")

# The CORRECT formula should account for sampling variance
# Under H₀, the difference Δθ̂ has variance:
#   Var[Δθ̂_j] = θ_j(1-θ_j) * (1/n_left + 1/n_right)
# So the standardized statistic should be:
#   χ² = ||Δθ̂||² / Var[Δθ̂] per dimension
#      = ||Δθ̂||² / (avg_var * (1/n_left + 1/n_right))
#
# With projection:
#   χ² = ||R·Δθ̂||² / (k * avg_var * (1/n_left + 1/n_right))

inverse_n_sum = 1 / n_left + 1 / n_right
chi_sq_correct = proj_dist_sq / (k * avg_var * inverse_n_sum)
print(
    f"Correct χ² = proj_dist² / (k * avg_var * (1/n_left + 1/n_right)) = {chi_sq_correct:.2f}"
)

print()
print("=" * 60)
print("DIAGNOSIS:")
print("=" * 60)
print(f"Current formula uses n_eff = {n_eff}")
print(f"But n_eff = 2*n_left*n_right/(n_left+n_right) = harmonic mean")
print(f"And (1/n_left + 1/n_right) = {inverse_n_sum}")
print(f"Note: n_eff * (1/n_left + 1/n_right) = 2 always!")
print()
print(f"So current χ² = n_eff * proj_dist² / (k * avg_var)")
print(f"            = {n_eff} * {proj_dist_sq:.4f} / ({k} * {avg_var:.4f})")
print(f"            = {chi_sq_current:.2f}")
print()
print(f"Correct χ² = proj_dist² / (k * avg_var * (1/n_left + 1/n_right))")
print(f"          = {proj_dist_sq:.4f} / ({k} * {avg_var:.4f} * {inverse_n_sum})")
print(f"          = {chi_sq_correct:.2f}")
print()
print("The current formula is MISSING the (1/n_left + 1/n_right) factor!")
print("This is causing massively inflated test statistics.")

print()
print("=" * 60)
print("FURTHER ANALYSIS - Why still inflated?")
print("=" * 60)

# The issue is that ||R·Δθ̂||² ≠ ||Δθ̂||²
# By Johnson-Lindenstrauss, E[||R·x||²] ≈ ||x||² (not equal, approximately)
# But actually for sparse Rademacher: E[||R·x||²] = ||x||² exactly if R is scaled correctly

# Let's check: what should the projection preserve?
# If Δθ̂ has variance Var[Δθ̂_j] = θ_j(1-θ_j)*(1/n_left + 1/n_right)
# Then E[||Δθ̂||²] = Σ Var[Δθ̂_j] = d * avg_var * (1/n_left + 1/n_right)

print(f"E[||Δθ̂||²] = d * avg_var * (1/n_left + 1/n_right)")
print(
    f"           = {d} * {avg_var:.4f} * {inverse_n_sum} = {d * avg_var * inverse_n_sum:.2f}"
)

# After projection, by JL lemma:
# E[||R·Δθ̂||²] ≈ ||Δθ̂||² (for fixed Δθ̂)
# But we want E over the sampling distribution of Δθ̂!

# The key insight: we need to normalize by the expected squared norm
# χ² = ||R·Δθ̂||² / E[||R·Δθ̂||²]    (under H₀)
#    ≈ ||R·Δθ̂||² / E[||Δθ̂||²]       (by JL)
#    = ||R·Δθ̂||² / (d * avg_var * (1/n_left + 1/n_right))

chi_sq_v2 = proj_dist_sq / (d * avg_var * inverse_n_sum)
print(f"χ² = proj_dist² / (d * avg_var * (1/n_left + 1/n_right))")
print(f"   = {proj_dist_sq:.4f} / ({d} * {avg_var:.4f} * {inverse_n_sum})")
print(f"   = {chi_sq_v2:.4f}")
print()
print("This should be ~1 under H₀, not χ²(k)!")

print()
print("=" * 60)
print("THE ACTUAL FIX - Projection for chi-square")
print("=" * 60)
# The projection DOES NOT give a chi-square distribution directly.
#
# What chi-square approximation works for JSD?
# 2*N*JSD ~ χ²(df) where df = d-1 for multinomial
#
# For binary features with different variances:
# Sum of (Δθ̂_j)² / Var[Δθ̂_j] ~ χ²(d)
# = Sum of (Δθ̂_j)² / (θ_j(1-θ_j) * (1/n_left + 1/n_right)) ~ χ²(d)
#
# But we want a LOWER dimensional test.
#
# The correct approach:
# 1. Standardize each dimension: z_j = Δθ̂_j / sqrt(θ_j(1-θ_j) * (1/n_left + 1/n_right))
# 2. Project: R·z where R is k×d
# 3. Then ||R·z||² ~ χ²(k) if z ~ N(0, I)

z = (theta_left - theta_right) / np.sqrt(pooled * (1 - pooled) * inverse_n_sum + 1e-10)
z_dist_sq = np.sum(z**2)
print(f"||z||² (standardized, full) = {z_dist_sq:.2f}")
print(f"Expected ||z||² under H₀ = d = {d}")

R_z = R @ z
proj_z_dist_sq = np.sum(R_z**2)
print(f"||R·z||² (standardized, projected) = {proj_z_dist_sq:.2f}")
print(f"Expected ||R·z||² under H₀ = k = {k}")

print()
print("SO THE FIX IS:")
print("1. Standardize Δθ̂ by dividing by sqrt(var per dimension)")
print("2. Project the standardized vector")
print("3. ||R·z||² ~ χ²(k)")

print()
print("=" * 60)
print("DEBUG: Projection Matrix Properties")
print("=" * 60)

# Our projection matrix R is k×d
# For sparse Rademacher: R_ij ∈ {-sqrt(3), 0, +sqrt(3)} with P(0)=2/3, P(±sqrt(3))=1/6
# This is scaled so that E[R_ij²] = 1
#
# For any vector x:
# E[||Rx||²] = E[Σ_i (Σ_j R_ij x_j)²]
#            = Σ_i E[(Σ_j R_ij x_j)²]
#            = Σ_i Σ_j E[R_ij²] x_j²   (cross terms vanish due to independence)
#            = k * Σ_j x_j² * E[R_ij²]
#            = k * ||x||²   if E[R_ij²] = 1

print(f"R shape: {R.shape}")
print(f"Mean of R: {R.mean():.4f}")
print(f"E[R_ij²] = {np.mean(R**2):.4f} (should be ~1)")
print(f"Nonzero fraction: {(R != 0).mean():.4f} (should be ~1/3)")

# So E[||Rx||²] = k * ||x||²
# Therefore for standardized z:
# E[||R·z||²] = k * ||z||²
# But ||z||² ~ χ²(d) under H₀
# So E[||R·z||²] = k * d under H₀ !!

print()
print(f"||z||² = {z_dist_sq:.2f}")
print(f"||R·z||² = {proj_z_dist_sq:.2f}")
print(f"Ratio: ||R·z||² / ||z||² = {proj_z_dist_sq / z_dist_sq:.4f}")
print(f"Expected ratio: k = {k}")

print()
print("AH-HA! The projection MULTIPLIES the norm by ~k, not preserves it!")
print(f"E[||R·z||²] = k * ||z||² ≈ k * d = {k * d}")
print()
print("So the correct chi-square is:")
chi_sq_final = proj_z_dist_sq / d
print(f"χ² = ||R·z||² / d = {proj_z_dist_sq:.2f} / {d} = {chi_sq_final:.2f}")
print(f"This should be ~k = {k} under H₀")
print()

# Better yet, let's derive from first principles
chi_sq_final2 = proj_dist_sq / (d * avg_var * inverse_n_sum)
print("=" * 60)
print("FINAL FORMULA (not using standardization):")
print("=" * 60)
print(f"χ² = ||R·Δθ̂||² / (d * avg_var * (1/n_left + 1/n_right))")
print(f"   = {proj_dist_sq:.4f} / ({d} * {avg_var:.4f} * {inverse_n_sum})")
print(f"   = {chi_sq_final2:.4f}")
print()
print("Wait, this is ~0.74, not ~k=10...")
print()
print("Let me re-derive with correct E[||Rx||²] = k * ||x||²:")
chi_sq_final3 = proj_dist_sq / (k * d * avg_var * inverse_n_sum / d)
print(f"If E[||R·Δθ̂||²] = k * E[||Δθ̂||²] = k * d * avg_var * (1/n_left + 1/n_right)")
print(f"Then χ² = ||R·Δθ̂||² / (k * avg_var * (1/n_left + 1/n_right)) with df=d")
print(f"      = {proj_dist_sq:.4f} / ({k} * {avg_var:.4f} * {inverse_n_sum})")
print(f"      = {proj_dist_sq / (k * avg_var * inverse_n_sum):.4f}")
print()
print("Hmm, still not matching. Let's just Monte Carlo it:")
print()

# Monte Carlo
np.random.seed(123)
n_sim = 1000
chi_sq_samples = []
for _ in range(n_sim):
    # Sample under H₀
    left_s = np.random.binomial(1, theta_true, size=(n_left, d))
    right_s = np.random.binomial(1, theta_true, size=(n_right, d))
    theta_l = left_s.mean(axis=0)
    theta_r = right_s.mean(axis=0)
    delta = theta_l - theta_r

    proj_delta = R @ delta
    proj_d_sq = np.sum(proj_delta**2)

    # Various normalizations
    pooled_s = 0.5 * (theta_l + theta_r)
    avg_var_s = np.mean(pooled_s * (1 - pooled_s))

    # Try different formulas
    chi_sq_samples.append(
        {
            "v1_current": n_eff * proj_d_sq / (k * avg_var_s),
            "v2_with_inv_n": proj_d_sq / (k * avg_var_s * inverse_n_sum),
            "v3_div_d": proj_d_sq / (d * avg_var_s * inverse_n_sum),
            "v4_k_times_d": proj_d_sq / (k * d * avg_var_s * inverse_n_sum / d),
        }
    )

print(f"Monte Carlo with {n_sim} simulations under H₀:")
print(
    f"v1 (current): mean={np.mean([s['v1_current'] for s in chi_sq_samples]):.2f}, expected df={k}"
)
print(
    f"v2 (with 1/n factor): mean={np.mean([s['v2_with_inv_n'] for s in chi_sq_samples]):.2f}"
)
print(f"v3 (div d): mean={np.mean([s['v3_div_d'] for s in chi_sq_samples]):.4f}")
print()
print(f"For χ²(k={k}), expected mean = {k}")

print()
print("=" * 60)
print("UNDERSTANDING THE PROJECTION SCALING")
print("=" * 60)

# The projection matrix has R_ij ∈ {-1, 0, +1} * sqrt(3/k)
# So R_ij ∈ {-sqrt(3/k), 0, +sqrt(3/k)} with probs {1/6, 2/3, 1/6}
# E[R_ij²] = (1/6 + 1/6) * (3/k) = (1/3) * (3/k) = 1/k

print(f"After scaling by sqrt(3/k):")
print(f"E[R_ij²] = (1/3) * (3/k) = 1/k = {1 / k:.4f}")
print(f"Observed E[R_ij²] = {np.mean(R**2):.4f}")

# For a vector x of dimension d:
# E[||Rx||²] = E[Σ_i (Σ_j R_ij x_j)²]
# Each row of Rx is independent, so:
# E[||Rx||²] = k * E[(Σ_j R_1j x_j)²]
#            = k * [Var[Σ_j R_1j x_j] + (E[Σ_j R_1j x_j])²]
#            = k * [Σ_j Var[R_1j] x_j² + 0]  (cross-terms vanish, E[R_ij]=0)
#            = k * Σ_j (1/k) x_j²
#            = ||x||²

print(f"E[||Rx||²] = k * Σ_j E[R_1j²] x_j² = k * (1/k) * ||x||² = ||x||²")
print()

# Let's verify with Monte Carlo on fixed x
test_x = theta_left - theta_right
true_dist_sq = np.sum(test_x**2)
print(f"||x||² = {true_dist_sq:.4f}")

# Multiple projections
proj_dists = []
for seed in range(100):
    R_test = generate_projection_matrix(d, k, random_state=seed)
    proj_x = R_test @ test_x
    proj_dists.append(np.sum(proj_x**2))

print(f"Mean ||Rx||² over 100 random R: {np.mean(proj_dists):.4f}")
print(f"This should equal ||x||² = {true_dist_sq:.4f}")
print()

# Now the key insight: under H₀, x = Δθ̂ is a random vector with:
# E[Δθ̂_j] = 0
# Var[Δθ̂_j] = θ_j(1-θ_j) * (1/n_left + 1/n_right)
#
# So E[||Δθ̂||²] = Σ Var[Δθ̂_j] ≈ d * avg_var * (1/n_left + 1/n_right)
#
# And E[||R·Δθ̂||²] = E[||Δθ̂||²] = d * avg_var * (1/n_left + 1/n_right)
# (because JL preserves norm in expectation)

expected_norm_sq = d * avg_var * inverse_n_sum
print(f"E[||Δθ̂||²] = d * avg_var * (1/n_left + 1/n_right)")
print(f"          = {d} * {avg_var:.4f} * {inverse_n_sum:.4f}")
print(f"          = {expected_norm_sq:.2f}")
print()

# The chi-square statistic should be:
# χ² = ||R·Δθ̂||² / E[||R·Δθ̂||²] * df
# But we want df degrees of freedom, so:
# χ² = df * ||R·Δθ̂||² / E[||R·Δθ̂||²]
#    = df * ||R·Δθ̂||² / (d * avg_var * (1/n_left + 1/n_right))

# What should df be? Under H₀, the projected vector R·Δθ̂ has k components
# But they're not independent - they're linear combinations of d dependent Δθ̂_j

# Actually, the correct approach is:
# If Δθ̂ ~ N(0, σ² I) approximately, then ||Δθ̂||² / σ² ~ χ²(d)
# And ||R·Δθ̂||² / σ² ~ χ²(k) for orthonormal R
# But our R is not orthonormal!

# Let's try: the test statistic should have E[χ²] = k under H₀
# So: χ² = k * ||R·Δθ̂||² / E[||R·Δθ̂||²]
#        = k * ||R·Δθ̂||² / (d * avg_var * (1/n_left + 1/n_right))

print("=" * 60)
print("FINAL CORRECT FORMULA")
print("=" * 60)
chi_sq_corrected = k * proj_dist_sq / expected_norm_sq
print(f"χ² = k * ||R·Δθ̂||² / (d * avg_var * (1/n_left + 1/n_right))")
print(f"   = {k} * {proj_dist_sq:.4f} / {expected_norm_sq:.2f}")
print(f"   = {chi_sq_corrected:.2f}")
print(f"Expected under H₀: ~k = {k}")
print()

# Verify with Monte Carlo
print("Monte Carlo verification (1000 samples under H₀):")
chi_sq_corrected_samples = []
for _ in range(1000):
    left_s = np.random.binomial(1, theta_true, size=(n_left, d))
    right_s = np.random.binomial(1, theta_true, size=(n_right, d))
    theta_l = left_s.mean(axis=0)
    theta_r = right_s.mean(axis=0)
    delta = theta_l - theta_r

    proj_delta = R @ delta
    proj_d_sq = np.sum(proj_delta**2)

    pooled_s = 0.5 * (theta_l + theta_r)
    avg_var_s = np.mean(pooled_s * (1 - pooled_s))
    expected_s = d * avg_var_s * inverse_n_sum

    chi_sq_corrected_samples.append(k * proj_d_sq / expected_s)

print(f"Mean χ²: {np.mean(chi_sq_corrected_samples):.2f} (expected: {k})")
print(
    f"Std χ²: {np.std(chi_sq_corrected_samples):.2f} (expected for χ²(k): {np.sqrt(2 * k):.2f})"
)
print()

# Chi-square(k) has mean=k and variance=2k
# Let's also check the p-value distribution
from scipy.stats import chi2

p_values = [1 - chi2.cdf(x, df=k) for x in chi_sq_corrected_samples]
print(f"Fraction p < 0.05: {np.mean(np.array(p_values) < 0.05):.3f} (expected: 0.05)")
print(f"Fraction p < 0.01: {np.mean(np.array(p_values) < 0.01):.3f} (expected: 0.01)")

print()
print("=" * 60)
print("EFFECT OF SAMPLE SIZE")
print("=" * 60)

for n_test in [2, 5, 10, 20, 50]:
    n_left_t, n_right_t = n_test, n_test
    n_eff_t = (2 * n_left_t * n_right_t) / (n_left_t + n_right_t)
    inverse_n_sum_t = 1 / n_left_t + 1 / n_right_t

    chi_samples = []
    for _ in range(500):
        left_s = np.random.binomial(1, theta_true, size=(n_left_t, d))
        right_s = np.random.binomial(1, theta_true, size=(n_right_t, d))
        theta_l = left_s.mean(axis=0)
        theta_r = right_s.mean(axis=0)
        delta = theta_l - theta_r

        proj_delta = R @ delta
        proj_d_sq = np.sum(proj_delta**2)

        pooled_s = 0.5 * (theta_l + theta_r)
        avg_var_s = np.mean(pooled_s * (1 - pooled_s))
        expected_s = d * avg_var_s * inverse_n_sum_t

        chi_samples.append(k * proj_d_sq / expected_s)

    p_vals = [1 - chi2.cdf(x, df=k) for x in chi_samples]
    print(
        f"n={n_test:3d}: mean χ²={np.mean(chi_samples):6.2f}, p<0.05 rate={np.mean(np.array(p_vals) < 0.05):.3f}"
    )

print()
print("=" * 60)
print("ALTERNATIVE: Use multiple projection trials")
print("=" * 60)

# Average over multiple random projections to reduce variance
n_trials = 10
chi_samples_avg = []
for _ in range(500):
    left_s = np.random.binomial(1, theta_true, size=(n_left, d))
    right_s = np.random.binomial(1, theta_true, size=(n_right, d))
    theta_l = left_s.mean(axis=0)
    theta_r = right_s.mean(axis=0)
    delta = theta_l - theta_r

    pooled_s = 0.5 * (theta_l + theta_r)
    avg_var_s = np.mean(pooled_s * (1 - pooled_s))
    expected_s = d * avg_var_s * inverse_n_sum

    # Average projection distance over n_trials
    proj_dists = []
    for trial in range(n_trials):
        R_t = generate_projection_matrix(d, k, random_state=trial)
        proj_delta = R_t @ delta
        proj_dists.append(np.sum(proj_delta**2))
    avg_proj_d_sq = np.mean(proj_dists)

    chi_samples_avg.append(k * avg_proj_d_sq / expected_s)

p_vals_avg = [1 - chi2.cdf(x, df=k) for x in chi_samples_avg]
print(f"With {n_trials} projection trials averaged:")
print(f"Mean χ²: {np.mean(chi_samples_avg):.2f} (expected: {k})")
print(f"p<0.05 rate: {np.mean(np.array(p_vals_avg) < 0.05):.3f} (expected: 0.05)")

print()
print("=" * 60)
print("ALTERNATIVE: Higher k dimension")
print("=" * 60)

for k_test in [10, 20, 50, 100]:
    R_k = generate_projection_matrix(d, k_test, random_state=42)
    chi_samples_k = []
    for _ in range(500):
        left_s = np.random.binomial(1, theta_true, size=(n_left, d))
        right_s = np.random.binomial(1, theta_true, size=(n_right, d))
        theta_l = left_s.mean(axis=0)
        theta_r = right_s.mean(axis=0)
        delta = theta_l - theta_r

        proj_delta = R_k @ delta
        proj_d_sq = np.sum(proj_delta**2)

        pooled_s = 0.5 * (theta_l + theta_r)
        avg_var_s = np.mean(pooled_s * (1 - pooled_s))
        expected_s = d * avg_var_s * inverse_n_sum

        chi_samples_k.append(k_test * proj_d_sq / expected_s)

    p_vals_k = [1 - chi2.cdf(x, df=k_test) for x in chi_samples_k]
    print(
        f"k={k_test:3d}: mean χ²={np.mean(chi_samples_k):6.2f}, p<0.05 rate={np.mean(np.array(p_vals_k) < 0.05):.3f}"
    )
