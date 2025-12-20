"""Debug: Check the projection scaling issue.

The test statistic is:
  T = ||R @ z||²

where z is standardized (mean 0, var 1 per component under H₀).

For T ~ χ²(k), we need the rows of R to be:
1. Orthonormal (so ||R @ z||² = Σ (rᵢᵀz)² where rᵢᵀz ~ N(0,1))
2. OR properly scaled so E[||R @ z||²] = k

Let's check what sklearn's SparseRandomProjection does.
"""

import numpy as np
from sklearn.random_projection import SparseRandomProjection

# Simulate H₀
d = 3000  # features
k = 14  # projection dimension
n_trials = 1000

# Create projection matrix
projector = SparseRandomProjection(n_components=k, density="auto", random_state=42)
dummy = np.zeros((1, d))
projector.fit(dummy)
R = projector.components_.toarray()

print(f"Projection matrix shape: {R.shape}")
print(f"density = 1/sqrt({d}) = {1 / np.sqrt(d):.4f}")

# Check row norms
row_norms = np.linalg.norm(R, axis=1)
print(f"\nRow norms:")
print(f"  Mean: {row_norms.mean():.4f}")
print(f"  Std:  {row_norms.std():.4f}")
print(f"  Min:  {row_norms.min():.4f}")
print(f"  Max:  {row_norms.max():.4f}")

# What's the expected row norm?
# SparseRandomProjection uses entries = {-sqrt(3/dens), 0, +sqrt(3/dens)}
# with prob = {dens/2, 1-dens, dens/2}
density = 1.0 / np.sqrt(d)
n_nonzero_expected = d * density
print(f"\nExpected nonzeros per row: {n_nonzero_expected:.1f}")
print(f"Entry variance: {3.0:.1f}")  # ±sqrt(3/dens) with prob dens, so var = 3
print(f"Expected row norm²: {n_nonzero_expected * 3.0:.1f}")
print(f"Expected row norm: {np.sqrt(n_nonzero_expected * 3.0):.2f}")

# Check actual
actual_nonzero = np.count_nonzero(R, axis=1).mean()
print(f"Actual nonzeros per row: {actual_nonzero:.1f}")

# Now test: what is E[||R @ z||²] when z ~ N(0, I)?
test_stats = []
for _ in range(n_trials):
    z = np.random.randn(d)
    proj = R @ z
    test_stats.append(np.sum(proj**2))

mean_stat = np.mean(test_stats)
std_stat = np.std(test_stats)

print(f"\n{'=' * 50}")
print(f"E[||R @ z||²] when z ~ N(0, I):")
print(f"  Observed mean: {mean_stat:.1f}")
print(f"  Expected for χ²({k}): {k}")
print(f"  Ratio (observed/expected): {mean_stat / k:.1f}")
print(f"  Observed std: {std_stat:.1f}")
print(f"  Expected for χ²({k}): {np.sqrt(2 * k):.1f}")
print(f"{'=' * 50}")

# What scaling is needed?
# We need E[||R @ z||²] = k
# Currently E[||R @ z||²] = Σᵢ E[(rᵢᵀz)²] = Σᵢ ||rᵢ||² = k * mean(||rᵢ||²)
mean_row_norm_sq = np.mean(row_norms**2)
print(f"\nMean ||rᵢ||²: {mean_row_norm_sq:.1f}")
print(f"To get χ²({k}), we need ||rᵢ||² = 1 for each row")
print(f"Current scaling factor: {mean_row_norm_sq:.1f}")
print(f"Fix: divide R by sqrt(mean_row_norm_sq) = {np.sqrt(mean_row_norm_sq):.2f}")

# Test with normalized R
R_normalized = R / row_norms[:, np.newaxis]

test_stats_normalized = []
for _ in range(n_trials):
    z = np.random.randn(d)
    proj = R_normalized @ z
    test_stats_normalized.append(np.sum(proj**2))

mean_normalized = np.mean(test_stats_normalized)
print(f"\nWith row-normalized R:")
print(f"  E[||R_norm @ z||²] = {mean_normalized:.2f} (expected: {k})")

# But wait - sklearn is designed for DISTANCE PRESERVATION, not NORM PRESERVATION
# The JL lemma says: ||R(x-y)||² ≈ (d/k) * ||x-y||² with high probability
# So there's a scaling factor of (d/k) built in!

print(f"\n{'=' * 50}")
print(f"EXPLANATION:")
print(f"sklearn's SparseRandomProjection is designed for DISTANCE preservation")
print(f"The JL lemma: ||Rx - Ry||² ≈ ||x - y||² (approximately)")
print(f"")
print(f"For our chi-square test, we need: ||Rz||² ~ χ²(k) when z ~ N(0,I)")
print(f"This requires R to have orthonormal rows (||rᵢ|| = 1)")
print(f"")
print(f"Current mean(||rᵢ||²) = {mean_row_norm_sq:.1f}")
print(f"So T = ||Rz||² ~ {mean_row_norm_sq:.0f} × χ²({k})")
print(f"                ~ χ²({int(mean_row_norm_sq * k)})")
print(f"")
print(f"This explains the 200-700× inflation in test statistics!")
print(f"{'=' * 50}")
