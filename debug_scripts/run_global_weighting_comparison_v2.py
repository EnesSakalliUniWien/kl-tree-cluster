#!/usr/bin/env python3
"""
Run two benchmark comparisons with global weighting OFF and ON.
Uses subprocess to ensure completely fresh Python processes for each run.
"""

import subprocess
import sys
from datetime import datetime
import pandas as pd


def run_benchmark_with_config(use_weighting: bool) -> str:
    """Run benchmark in a fresh subprocess with specified config."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "ON" if use_weighting else "OFF"
    output_file = f"results/benchmark_global_{suffix}_{timestamp}.csv"

    print(f"\n{'=' * 80}")
    print(f"RUNNING BENCHMARK: Global Weighting {suffix}")
    print(f"{'=' * 80}\n")

    # Create a temporary Python script that sets config and runs benchmark
    script = f"""
import sys
# Modify config BEFORE any imports
import kl_clustering_analysis.config as config
config.USE_GLOBAL_DIVERGENCE_WEIGHTING = {use_weighting}

print(f"Config: USE_GLOBAL_DIVERGENCE_WEIGHTING = {{config.USE_GLOBAL_DIVERGENCE_WEIGHTING}}")
print(f"Config: GLOBAL_WEIGHT_METHOD = {{config.GLOBAL_WEIGHT_METHOD}}")

# Now import and run benchmark
from kl_clustering_analysis.benchmarking import benchmark_cluster_algorithm
df_results, fig = benchmark_cluster_algorithm(
    significance_level=0.01,
    plot_umap=False,
    plot_manifold=False
)
df_results.to_csv("{output_file}", index=False)
print(f"Saved results to {output_file}")
"""

    # Write temporary script
    temp_script = f"_temp_benchmark_{suffix}.py"
    with open(temp_script, "w") as f:
        f.write(script)

    # Run in subprocess
    try:
        result = subprocess.run(
            [sys.executable, temp_script],
            check=True,
            capture_output=False,  # Show output in real-time
        )
    finally:
        # Clean up temp script
        import os

        if os.path.exists(temp_script):
            os.remove(temp_script)

    print(f"\n✅ Run complete: {output_file}\n")
    return output_file


def compare_results(off_file: str, on_file: str):
    """Compare benchmark results between OFF and ON configurations."""
    print(f"\n{'=' * 80}")
    print("COMPARISON ANALYSIS")
    print(f"{'=' * 80}\n")

    # Load results
    df_off = pd.read_csv(off_file)
    df_on = pd.read_csv(on_file)

    # Filter to KL Divergence method
    df_off_kl = df_off[df_off["Method"] == "KL Divergence"].copy()
    df_on_kl = df_on[df_on["Method"] == "KL Divergence"].copy()

    print(f"Total KL Divergence test cases: {len(df_off_kl)}\n")

    # Merge on case name
    merged = df_off_kl.merge(
        df_on_kl, on=["Test", "Case_Name", "True"], suffixes=("_off", "_on")
    )

    # Compute differences
    merged["cluster_diff"] = merged["Found_on"] - merged["Found_off"]
    merged["ari_diff"] = merged["ARI_on"] - merged["ARI_off"]
    merged["nmi_diff"] = merged["NMI_on"] - merged["NMI_off"]

    # Summary statistics
    print("Cluster Count Changes (ON - OFF):")
    print(f"  Mean: {merged['cluster_diff'].mean():.2f}")
    print(f"  Median: {merged['cluster_diff'].median():.2f}")
    print(f"  Min: {merged['cluster_diff'].min():.0f}")
    print(f"  Max: {merged['cluster_diff'].max():.0f}")
    print(f"  Fewer clusters (ON): {(merged['cluster_diff'] < 0).sum()}")
    print(f"  Same clusters: {(merged['cluster_diff'] == 0).sum()}")
    print(f"  More clusters (ON): {(merged['cluster_diff'] > 0).sum()}")
    print()

    print("ARI Changes (ON - OFF):")
    print(f"  Mean: {merged['ari_diff'].mean():.4f}")
    print(f"  Median: {merged['ari_diff'].median():.4f}")
    print(f"  Improved (>0.01): {(merged['ari_diff'] > 0.01).sum()}")
    print(f"  Unchanged (±0.01): {(abs(merged['ari_diff']) <= 0.01).sum()}")
    print(f"  Degraded (<-0.01): {(merged['ari_diff'] < -0.01).sum()}")
    print()

    print("NMI Changes (ON - OFF):")
    print(f"  Mean: {merged['nmi_diff'].mean():.4f}")
    print(f"  Median: {merged['nmi_diff'].median():.4f}")
    print()

    # Top improvements
    print("=" * 80)
    print("TOP 5 CASES WITH LARGEST CLUSTER REDUCTION (ON < OFF)")
    print("=" * 80)
    cluster_reductions = merged[merged["cluster_diff"] < 0].nsmallest(5, "cluster_diff")
    if len(cluster_reductions) == 0:
        cluster_reductions = merged.nsmallest(5, "cluster_diff")

    display_cols = [
        "Case_Name",
        "True",
        "Found_off",
        "Found_on",
        "cluster_diff",
        "ARI_off",
        "ARI_on",
        "ari_diff",
    ]
    print(cluster_reductions[display_cols].to_string(index=False))
    print()

    print("=" * 80)
    print("TOP 5 CASES WITH LARGEST ARI IMPROVEMENT (ON > OFF)")
    print("=" * 80)
    ari_improvements = merged[merged["ari_diff"] > 0].nlargest(5, "ari_diff")
    if len(ari_improvements) == 0:
        ari_improvements = merged.nlargest(5, "ari_diff")

    display_cols2 = [
        "Case_Name",
        "True",
        "Found_off",
        "Found_on",
        "ARI_off",
        "ARI_on",
        "ari_diff",
    ]
    print(ari_improvements[display_cols2].to_string(index=False))
    print()

    # Save comparison
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    comparison_file = f"results/comparison_OFF_vs_ON_{timestamp}.csv"
    merged.to_csv(comparison_file, index=False)

    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)
    print(f"  OFF: {off_file}")
    print(f"  ON:  {on_file}")
    print(f"  Comparison: {comparison_file}")
    print("=" * 80)


if __name__ == "__main__":
    # Run both benchmarks
    off_file = run_benchmark_with_config(use_weighting=False)
    on_file = run_benchmark_with_config(use_weighting=True)

    # Compare results
    compare_results(off_file, on_file)
