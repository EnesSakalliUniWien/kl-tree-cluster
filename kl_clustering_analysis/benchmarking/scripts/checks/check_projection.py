"""Check which test cases use random projection."""

from kl_clustering_analysis.benchmarking.test_cases import get_default_test_cases
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    compute_projection_dimension,
)

test_cases = get_default_test_cases()
print("Case_Name                         n       d    k(JL)  k<d?")
print("-" * 75)
for tc in test_cases:
    # Handle both naming conventions
    n = tc.get("n_samples", tc.get("n_rows", 100))
    d = tc.get("n_features", tc.get("n_cols", 50))
    name = tc.get("name", "Unknown")[:30]
    k = int(compute_projection_dimension(n, d))
    print(f"{name:30s} {n:5d}   {d:5d}   {k:5d}  {k < d}")
