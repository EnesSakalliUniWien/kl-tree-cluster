#!/usr/bin/env python3
"""
Debug script to verify global divergence weighting configuration and benchmark status.
"""

import pandas as pd
from pathlib import Path
import sys

# Import the actual config module
import kl_clustering_analysis.config as config

print("=" * 80)
print("GLOBAL DIVERGENCE WEIGHTING - CONFIGURATION DEBUG")
print("=" * 80)
print()

# Check current config values
print("Current Configuration (from config.py):")
print(f"  USE_GLOBAL_DIVERGENCE_WEIGHTING: {config.USE_GLOBAL_DIVERGENCE_WEIGHTING}")
print(f"  GLOBAL_WEIGHT_METHOD: {config.GLOBAL_WEIGHT_METHOD}")
print(f"  GLOBAL_WEIGHT_STRENGTH: {config.GLOBAL_WEIGHT_STRENGTH}")
print(f"  GLOBAL_WEIGHT_PERCENTILE: {config.GLOBAL_WEIGHT_PERCENTILE}")
print()

# Check if benchmark files exist
baseline_file = Path("results/validation_results_baseline_2025-12-20_19-44-33.csv")
weighted_file = Path(
    "results/validation_results_global_weighting_2025-12-20_19-38-18.csv"
)

print("=" * 80)
print("BENCHMARK FILES STATUS")
print("=" * 80)
print()

if baseline_file.exists():
    df_baseline = pd.read_csv(baseline_file)
    df_baseline_kl = df_baseline[df_baseline["Method"] == "KL Divergence"]
    print(f"✅ Baseline file found: {baseline_file.name}")
    print(f"   - Total rows: {len(df_baseline)}")
    print(f"   - KL Divergence cases: {len(df_baseline_kl)}")
else:
    print(f"❌ Baseline file NOT found: {baseline_file}")

print()

if weighted_file.exists():
    df_weighted = pd.read_csv(weighted_file)
    df_weighted_kl = df_weighted[df_weighted["Method"] == "KL Divergence"]
    print(f"✅ Weighted file found: {weighted_file.name}")
    print(f"   - Total rows: {len(df_weighted)}")
    print(f"   - KL Divergence cases: {len(df_weighted_kl)}")
else:
    print(f"❌ Weighted file NOT found: {weighted_file}")

print()
print("=" * 80)
print("IMPORTANT FINDINGS")
print("=" * 80)
print()

if baseline_file.exists() and weighted_file.exists():
    # Compare the results
    df_comparison = df_baseline_kl.merge(
        df_weighted_kl, on="Test", suffixes=("_base", "_weight")
    )

    cluster_diff = df_comparison["Found_weight"] - df_comparison["Found_base"]
    ari_diff = df_comparison["ARI_weight"] - df_comparison["ARI_base"]

    print("Comparison between benchmark runs:")
    print(
        f"  Cluster count differences: min={cluster_diff.min()}, max={cluster_diff.max()}, mean={cluster_diff.mean():.4f}"
    )
    print(
        f"  ARI differences: min={ari_diff.min():.4f}, max={ari_diff.max():.4f}, mean={ari_diff.mean():.4f}"
    )
    print()

    if cluster_diff.abs().sum() == 0 and ari_diff.abs().sum() == 0:
        print("⚠️  BOTH BENCHMARKS PRODUCED IDENTICAL RESULTS")
        print()
        print("This happened because:")
        print("  1. Both benchmark scripts modified config in-memory only")
        print("  2. The imported config module retained its original value (False)")
        print("  3. Therefore, both runs actually had weighting DISABLED")
        print()
        print("Solution:")
        print(
            f"  - config.py now has USE_GLOBAL_DIVERGENCE_WEIGHTING = {config.USE_GLOBAL_DIVERGENCE_WEIGHTING}"
        )
        print("  - Run a NEW benchmark to see the actual impact of global weighting")
        print("  - Compare the new run with the baseline file")
    else:
        print("✅ BENCHMARKS SHOW DIFFERENCES")
        print()
        print(f"  Cases with fewer clusters (weighted): {(cluster_diff < 0).sum()}")
        print(f"  Cases with more clusters (weighted): {(cluster_diff > 0).sum()}")
        print(f"  Cases with improved ARI: {(ari_diff > 0.01).sum()}")
        print(f"  Cases with degraded ARI: {(ari_diff < -0.01).sum()}")

print()
print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()

if config.USE_GLOBAL_DIVERGENCE_WEIGHTING:
    print("✅ Global weighting is ENABLED in config.py")
    print()
    print("To test the feature:")
    print("  1. Run benchmark with current config (weighting enabled)")
    print("  2. Compare with baseline file (weighting disabled)")
    print("  3. Analyze differences in cluster counts and accuracy")
    print()
    print("Example command:")
    print(
        '  python -c "from kl_clustering_analysis.benchmarking import benchmark_cluster_algorithm; '
    )
    print('            df, fig = benchmark_cluster_algorithm(significance_level=0.01)"')
else:
    print("⚠️  Global weighting is DISABLED in config.py")
    print()
    print("To enable:")
    print("  1. Set USE_GLOBAL_DIVERGENCE_WEIGHTING = True in config.py")
    print("  2. Run benchmark to see the impact")

print()
print("=" * 80)
