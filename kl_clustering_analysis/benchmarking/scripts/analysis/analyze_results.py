"""Analyze results by projection usage."""

from kl_clustering_analysis.benchmarking.pipeline import benchmark_cluster_algorithm
from tests.test_cases_config import get_default_test_cases
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    compute_projection_dimension,
)

test_cases = get_default_test_cases()
df_results, _ = benchmark_cluster_algorithm(test_cases=test_cases, verbose=False)

print("=" * 90)
print("Results by Projection Usage")
print("=" * 90)

# Add projection info to results
for i, tc in enumerate(test_cases):
    n = tc.get("n_samples", tc.get("n_rows", 100))
    d = tc.get("n_features", tc.get("n_cols", 50))
    df_results.loc[i, "Uses_Projection"] = compute_projection_dimension(n, d) < d
    df_results.loc[i, "n_rows"] = n
    df_results.loc[i, "n_cols"] = d

# Summary
proj_results = df_results[df_results["Uses_Projection"] == True]
no_proj_results = df_results[df_results["Uses_Projection"] == False]

print(f"\nWith Projection (d > 2n): {len(proj_results)} cases")
print(f"  Mean ARI: {proj_results['ARI'].mean():.3f}")
print(f"  Perfect (ARI=1.0): {(proj_results['ARI'] == 1.0).sum()}/{len(proj_results)}")
print(
    f"  Exact k match: {(proj_results['True'] == proj_results['Found']).sum()}/{len(proj_results)}"
)

print(f"\nWithout Projection (d <= 2n): {len(no_proj_results)} cases")
print(f"  Mean ARI: {no_proj_results['ARI'].mean():.3f}")
print(
    f"  Perfect (ARI=1.0): {(no_proj_results['ARI'] == 1.0).sum()}/{len(no_proj_results)}"
)
print(
    f"  Exact k match: {(no_proj_results['True'] == no_proj_results['Found']).sum()}/{len(no_proj_results)}"
)

print("\n" + "=" * 90)
print("Detailed Projection Results")
print("=" * 90)
print(
    proj_results[["Case_Name", "n_rows", "n_cols", "True", "Found", "ARI"]].to_string()
)

print("\n" + "=" * 90)
print("Detailed Non-Projection Results")
print("=" * 90)
print(
    no_proj_results[
        ["Case_Name", "n_rows", "n_cols", "True", "Found", "ARI"]
    ].to_string()
)
