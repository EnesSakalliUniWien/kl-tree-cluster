"""
Extract case 18 data for external power analysis.

This script generates the data and tree structure for case 18,
then saves all intermediate statistics for power analysis.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import sys
import json

sys.path.insert(0, "/Users/berksakalli/Projects/kl-te-cluster")

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.tree.distributions import populate_distributions
from kl_clustering_analysis.hierarchy_analysis.statistics import (
    annotate_child_parent_divergence,
    annotate_sibling_divergence,
)
from benchmarks.shared.cases import get_test_cases_by_category


def generate_case_18_data():
    """Generate the exact data for case 18."""
    from benchmarks.shared.generators import generate_case_data

    # Get case 18 config
    cases = get_test_cases_by_category("overlapping_binary_unbalanced")
    case_18 = [c for c in cases if "unbal_4c_small" in c.get("name", "")][0]

    print(f"Case 18 config: {case_18}")

    # Generate data (returns 4 values: df, labels, features, metadata)
    df, true_labels, features, metadata = generate_case_data(case_18)

    return df, true_labels, case_18

    # Populate distributions
    populate_distributions(tree, data)

    # Annotate statistics
    tree.stats_df = annotate_child_parent_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )
    tree.stats_df = annotate_sibling_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )

    return tree


def extract_split_decisions(tree):
    """Extract all split decisions with power-relevant statistics."""
    decisions = []

    for node_id in tree.nodes():
        if tree.nodes[node_id].get("is_leaf", False):
            continue

        children = list(tree.successors(node_id))
        if len(children) != 2:
            continue

        left, right = children

        # Get statistics
        n_parent = tree.nodes[node_id].get("leaf_count", 0)
        n_left = tree.nodes[left].get("leaf_count", 0)
        n_right = tree.nodes[right].get("leaf_count", 0)

        # Get distributions
        dist_parent = tree.nodes[node_id].get("distribution", None)
        dist_left = tree.nodes[left].get("distribution", None)
        dist_right = tree.nodes[right].get("distribution", None)

        # Get test results from stats_df if available
        cp_pval = None
        cp_significant = None
        sb_pval = None
        sb_different = None

        if hasattr(tree, "stats_df") and tree.stats_df is not None:
            if node_id in tree.stats_df.index:
                row = tree.stats_df.loc[node_id]
                cp_pval = row.get("Child_Parent_Divergence_P_Value", None)
                cp_significant = row.get("Child_Parent_Divergence_Significant", None)
                sb_pval = row.get("Sibling_Divergence_P_Value", None)
                sb_different = row.get("Sibling_BH_Different", None)

        decisions.append(
            {
                "node_id": node_id,
                "n_parent": n_parent,
                "n_left": n_left,
                "n_right": n_right,
                "depth": None,  # Will compute
                "cp_pval": cp_pval,
                "cp_significant": cp_significant,
                "sb_pval": sb_pval,
                "sb_different": sb_different,
                "dist_parent_mean": float(np.mean(dist_parent))
                if dist_parent is not None
                else None,
                "dist_left_mean": float(np.mean(dist_left))
                if dist_left is not None
                else None,
                "dist_right_mean": float(np.mean(dist_right))
                if dist_right is not None
                else None,
            }
        )

    return pd.DataFrame(decisions)


def compute_power_for_all_splits(df_decisions, min_effect_size=0.1):
    """Compute power for each split decision."""
    from kl_clustering_analysis.hierarchy_analysis.statistics.power_analysis import (
        power_wald_two_sample,
        cohens_h,
    )

    results = []

    for _, row in df_decisions.iterrows():
        n_left = row["n_left"]
        n_right = row["n_right"]

        # Skip if sample sizes are invalid
        if pd.isna(n_left) or pd.isna(n_right) or n_left < 2 or n_right < 2:
            results.append(
                {
                    "power": 0.0,
                    "effect_size": 0.0,
                    "n_required": np.inf,
                    "is_sufficient": False,
                }
            )
            continue

        # Estimate effect size from observed proportions
        p_left = row["dist_left_mean"] if not pd.isna(row["dist_left_mean"]) else 0.5
        p_right = row["dist_right_mean"] if not pd.isna(row["dist_right_mean"]) else 0.5

        # Use minimum effect size if observed is smaller
        effect_size = max(abs(cohens_h(p_left, p_right)), min_effect_size)

        # Compute power for sibling test
        power = power_wald_two_sample(
            n1=n_left,
            n2=n_right,
            p1=p_left,
            p2=p_right,
            alpha=0.05,
        )

        # Compute required sample size for 80% power
        # Approximate: n_required ≈ (Z_α/2 + Z_β)² × 2 × σ² / δ²
        z_alpha = stats.norm.ppf(0.975)  # Two-sided
        z_beta = stats.norm.ppf(0.80)
        pooled_var = (p_left * (1 - p_left) + p_right * (1 - p_right)) / 2
        delta = (
            abs(p_left - p_right) if abs(p_left - p_right) > 0.01 else min_effect_size
        )
        n_required = (z_alpha + z_beta) ** 2 * 2 * pooled_var / (delta**2)

        results.append(
            {
                "power": power,
                "effect_size": effect_size,
                "n_required": n_required,
                "is_sufficient": power >= 0.8,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 70)
    print("EXTRACTING CASE 18 DATA FOR POWER ANALYSIS")
    print("=" * 70)
    print()

    # Step 1: Generate data
    print("Step 1: Generating case 18 data...")
    data, true_assignments, config = generate_case_18_data()
    print(f"  Data shape: {data.shape}")
    print(f"  True clusters: {len(set(true_assignments.values()))}")
    print()

    # Step 2: Build tree
    print("Step 2: Building and annotating tree...")
    tree = build_and_annotate_tree(data)
    print(f"  Total nodes: {len(tree.nodes())}")
    print(
        f"  Internal nodes: {len([n for n in tree.nodes() if not tree.nodes[n].get('is_leaf', False)])}"
    )
    print()

    # Step 3: Extract split decisions
    print("Step 3: Extracting split decisions...")
    df_decisions = extract_split_decisions(tree)
    print(f"  Total split decisions: {len(df_decisions)}")
    print()

    # Step 4: Compute power
    print("Step 4: Computing power for each split...")
    from scipy import stats  # Import here for power calculation

    df_power = compute_power_for_all_splits(df_decisions)
    print(f"  Splits with sufficient power (≥80%): {df_power['is_sufficient'].sum()}")
    print(f"  Splits with insufficient power: {(~df_power['is_sufficient']).sum()}")
    print()

    # Step 5: Combine and save
    print("Step 5: Saving results...")
    df_combined = pd.concat([df_decisions, df_power], axis=1)

    output_path = "/Users/berksakalli/Projects/kl-te-cluster/debug_scripts/case_18_power_analysis.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    # Step 6: Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print("Sample size distribution:")
    print(df_combined[["n_left", "n_right"]].describe())
    print()

    print("Power distribution:")
    print(df_combined["power"].describe())
    print()

    print("Split outcomes vs power:")
    crosstab = pd.crosstab(
        df_combined["sb_different"].fillna("Unknown"),
        df_combined["is_sufficient"],
        margins=True,
    )
    print(crosstab)
    print()

    print("Low-power splits that were declared 'different':")
    low_power_different = df_combined[
        (~df_combined["is_sufficient"]) & (df_combined["sb_different"] == True)
    ]
    print(f"  Count: {len(low_power_different)}")
    if len(low_power_different) > 0:
        print(
            low_power_different[
                ["node_id", "n_left", "n_right", "power", "sb_pval"]
            ].head(10)
        )
