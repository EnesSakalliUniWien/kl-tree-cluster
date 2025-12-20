"""Check which test cases use random projection."""

from tests.test_cases_config import get_default_test_cases
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    should_use_projection,
)

test_cases = get_default_test_cases()
print("Case_Name                         n       d    d>2n?  Use Projection?")
print("-" * 75)
for tc in test_cases:
    # Handle both naming conventions
    n = tc.get("n_samples", tc.get("n_rows", 100))
    d = tc.get("n_features", tc.get("n_cols", 50))
    name = tc.get("name", "Unknown")[:30]
    use_proj = should_use_projection(d, n)
    exceeds = d > 2 * n
    print(f"{name:30s} {n:5d}   {d:5d}   {str(exceeds):5s}  {use_proj}")
