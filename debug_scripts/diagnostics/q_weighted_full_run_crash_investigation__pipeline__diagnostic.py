"""
Purpose: Investigate why 17 benchmark cases crashed during the full weighted run.
Inputs: Synthetic/benchmark data and configuration defined in-script.
Outputs: Console diagnostics and optional generated artifacts (plots/tables/files).
Expected runtime: ~10-180 seconds depending on dataset and settings.
How to run: python debug_scripts/diagnostics/q_weighted_full_run_crash_investigation__pipeline__diagnostic.py
"""

import warnings

import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators.generate_case_data import generate_case_data
from kl_clustering_analysis import config
from kl_clustering_analysis.tree.poset_tree import PosetTree

warnings.filterwarnings("ignore")

# Load completed cases from the first (longer) run
run_csv = "benchmarks/results/run_20260216_092739Z/full_benchmark_comparison.csv"
df_done = pd.read_csv(run_csv)
done_cases = set(df_done["case_id"].unique()) if "case_id" in df_done.columns else set()
if not done_cases:
    # Try alternative column name
    for col in df_done.columns:
        if "case" in col.lower() or "name" in col.lower():
            done_cases = set(df_done[col].unique())
            print(f"Using column '{col}' for case names, found {len(done_cases)} completed")
            break

all_cases = get_default_test_cases()
all_names = [c.get("name", f"case_{i}") for i, c in enumerate(all_cases)]

missing_names = [n for n in all_names if n not in done_cases]
print(f"Total cases: {len(all_cases)}")
print(f"Completed: {len(done_cases)}")
print(f"Missing: {len(missing_names)}")
print(f"\nMissing cases: {missing_names}")

# Generate data for missing cases and check sizes
print(f"\n{'Case':<40s} {'n':>6s} {'p':>6s} {'MB_est':>8s} │ {'KL_ok':>6s} {'Error'}")
print("─" * 100)

config.SIBLING_TEST_METHOD = "cousin_weighted_wald"

for case_name in missing_names:
    case = next(c for c in all_cases if c.get("name") == case_name)
    try:
        data_df, true_labels, dist, meta = generate_case_data(case)
        n, p = data_df.shape
        mb_est = (n * p * 8) / (1024 * 1024)  # float64

        # Try clustering WITHOUT plotting
        try:
            if dist is None:
                dist = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
            Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
            tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
            result = tree.decompose(
                leaf_data=data_df,
                alpha_local=config.ALPHA_LOCAL,
                sibling_alpha=config.SIBLING_ALPHA,
            )
            k_found = result.get("num_clusters", "?")
            kl_status = f"K={k_found}"
        except Exception as e:
            kl_status = "FAIL"
            print(f"{case_name:<40s} {n:>6d} {p:>6d} {mb_est:>7.1f}M │ {kl_status:>6s} {e}")
            continue

        print(f"{case_name:<40s} {n:>6d} {p:>6d} {mb_est:>7.1f}M │ {kl_status:>6s}")

    except Exception as e:
        print(f"{case_name:<40s} {'?':>6s} {'?':>6s} {'?':>8s} │ {'ERR':>6s} datagen: {e}")

print("\nConclusion: If all KL_ok show K=N, the crash is from UMAP/plotting, not clustering.")
