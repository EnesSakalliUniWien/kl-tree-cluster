#!/usr/bin/env python3
"""
Diagnostic script to verify global weighting is being applied correctly.
This directly tests the edge_significance function with and without weighting.
"""

import numpy as np
import pandas as pd
import sys

# Check config first
from kl_clustering_analysis import config

print("=" * 80)
print("CONFIGURATION CHECK")
print("=" * 80)
print(f"USE_GLOBAL_DIVERGENCE_WEIGHTING: {config.USE_GLOBAL_DIVERGENCE_WEIGHTING}")
print(f"GLOBAL_WEIGHT_METHOD: {config.GLOBAL_WEIGHT_METHOD}")
print(f"GLOBAL_WEIGHT_STRENGTH: {config.GLOBAL_WEIGHT_STRENGTH}")
print(f"GLOBAL_WEIGHT_PERCENTILE: {config.GLOBAL_WEIGHT_PERCENTILE}")
print()

# Generate test data using sklearn
from sklearn.datasets import make_blobs
from kl_clustering_analysis.tree.poset_tree import PosetTree

print("=" * 80)
print("GENERATING TEST DATA")
print("=" * 80)
X, true_labels = make_blobs(n_samples=300, n_features=10, centers=4, random_state=42)
print(f"Data shape: {X.shape}")
print(f"True clusters: {len(np.unique(true_labels))}")
print()

# Build tree using from_agglomerative
print("=" * 80)
print("BUILDING CLUSTER TREE")
print("=" * 80)
tree = PosetTree.from_agglomerative(X)
print(f"Tree built with {tree.number_of_nodes()} nodes")
print()

# Create leaf data (probability distributions for each sample)
# Normalize each row to sum to 1
X_positive = X - X.min(axis=0)  # Make all values positive
X_normalized = X_positive / X_positive.sum(axis=1, keepdims=True)
leaf_names = [f"leaf_{i}" for i in range(X.shape[0])]
leaf_data = pd.DataFrame(X_normalized, index=leaf_names)

# Populate node divergences
print("=" * 80)
print("POPULATING NODE DIVERGENCES")
print("=" * 80)
tree.populate_node_divergences(leaf_data)
print("Node divergences populated")
print()

# Get the stats dataframe
nodes_df = tree.stats_df
print("Stats DataFrame columns:")
print(nodes_df.columns.tolist())
print()

# Check if kl_divergence_global exists (the actual column name)
if "kl_divergence_global" in nodes_df.columns:
    print("✅ 'kl_divergence_global' column EXISTS")
    kl_to_root = nodes_df["kl_divergence_global"].dropna()
    print(f"   Non-null values: {len(kl_to_root)}")
    print(f"   Range: [{kl_to_root.min():.4f}, {kl_to_root.max():.4f}]")
    print(f"   Mean: {kl_to_root.mean():.4f}")
else:
    print("❌ 'kl_divergence_global' column MISSING")
    print("   Global weighting cannot work without this column!")
print()

# Now run decomposition to trigger statistical annotations
print("=" * 80)
print("RUNNING DECOMPOSITION (triggers statistical annotations)")
print("=" * 80)
decomp_results = tree.decompose()
print(
    f"Decomposition complete: {len(decomp_results.get('clusters', {}))} clusters found"
)
print()

# Get updated stats dataframe
nodes_df = tree.stats_df
print("Updated Stats DataFrame columns:")
print([c for c in nodes_df.columns if "Global" in c or "global" in c or "Weight" in c])
print()

# Check if Child_Parent_Divergence_Global_Weight exists
if "Child_Parent_Divergence_Global_Weight" in nodes_df.columns:
    print("✅ 'Child_Parent_Divergence_Global_Weight' column EXISTS")
    weights = nodes_df["Child_Parent_Divergence_Global_Weight"].dropna()
    print(f"   Non-null values: {len(weights)}")
    print(f"   Range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"   Mean: {weights.mean():.4f}")
    print(f"   Std:  {weights.std():.4f}")

    # Check if weights vary (are not all 1.0)
    if weights.std() > 0.001:
        print(
            f"\n   ✅ Weights VARY (std={weights.std():.4f}) - Global weighting IS working!"
        )

        # Show distribution of bonuses vs penalties
        bonus_count = (weights < 1.0).sum()
        neutral_count = ((weights >= 0.99) & (weights <= 1.01)).sum()
        penalty_count = (weights > 1.0).sum()
        total = len(weights)

        print(f"\n   Distribution (data-driven symmetric weighting):")
        print(
            f"     Bonus (weight < 1.0):   {bonus_count:4d} edges ({100 * bonus_count / total:.1f}%)"
        )
        print(
            f"     Neutral (≈ 1.0):        {neutral_count:4d} edges ({100 * neutral_count / total:.1f}%)"
        )
        print(
            f"     Penalty (weight > 1.0): {penalty_count:4d} edges ({100 * penalty_count / total:.1f}%)"
        )

        # Show bucket distribution
        print(f"\n   Bucket distribution:")
        buckets = [
            (0.5, 0.75),
            (0.75, 0.9),
            (0.9, 1.0),
            (1.0, 1.1),
            (1.1, 1.3),
            (1.3, 1.5),
            (1.5, 2.0),
        ]
        for low, high in buckets:
            count = ((weights >= low) & (weights < high)).sum()
            bar = "█" * (count // 10) if count > 0 else ""
            print(f"     [{low:.2f}, {high:.2f}): {count:4d} {bar}")
        count = (weights >= 2.0).sum()
        if count > 0:
            bar = "█" * (count // 10)
            print(f"     [2.00, 2.00]: {count:4d} {bar}")
    else:
        all_ones = (weights == 1.0).all()
        if all_ones:
            print(f"\n   ⚠️ Weights are ALL 1.0 - Global weighting may NOT be applied")
        else:
            print(f"\n   ⚠️ Weights are constant but not 1.0 - Check implementation")

    # Show some example weights
    print("\n   Sample weights (first 10 non-null):")
    cols_to_show = [
        "Child_Parent_Divergence_Global_Weight",
    ]
    available_cols = [c for c in cols_to_show if c in nodes_df.columns]
    sample = nodes_df[available_cols].dropna().head(10)
    print(sample.to_string(index=True))
else:
    print("❌ 'Child_Parent_Divergence_Global_Weight' column MISSING")
    print("   The edge_significance function may not be outputting this column")
print()

# Check significance results
print("=" * 80)
print("SIGNIFICANCE TEST RESULTS")
print("=" * 80)
if "Child_Parent_Divergence_Significant" in nodes_df.columns:
    sig = nodes_df["Child_Parent_Divergence_Significant"]
    print(f"Significant edges: {sig.sum()} / {sig.notna().sum()}")

if "Child_Parent_Divergence_P_Value" in nodes_df.columns:
    pvals = nodes_df["Child_Parent_Divergence_P_Value"].dropna()
    print(f"P-values range: [{pvals.min():.2e}, {pvals.max():.2e}]")
print()

# Final verdict
print("=" * 80)
print("VERDICT")
print("=" * 80)
if config.USE_GLOBAL_DIVERGENCE_WEIGHTING:
    if "Child_Parent_Divergence_Global_Weight" in nodes_df.columns:
        weights = nodes_df["Child_Parent_Divergence_Global_Weight"].dropna()
        if weights.std() > 0.001:
            print("✅ Global weighting is ENABLED and WORKING")
            print(f"   Weights range from {weights.min():.3f} to {weights.max():.3f}")
        elif (weights == 1.0).all():
            print("⚠️ Global weighting is ENABLED but all weights = 1.0")
            print("   This suggests the weighting logic is not being triggered")
            print("   Check if kl_divergence_to_root values are valid")
        else:
            print("⚠️ Global weighting is ENABLED but weights are constant")
            print("   This may indicate an issue with the data or implementation")
    else:
        print("❌ Global weighting is ENABLED but weight column not found")
        print(
            "   The annotate_child_parent_divergence function may not be outputting weights"
        )
else:
    print("ℹ️ Global weighting is DISABLED in config")
    print("   Set USE_GLOBAL_DIVERGENCE_WEIGHTING = True to enable")
