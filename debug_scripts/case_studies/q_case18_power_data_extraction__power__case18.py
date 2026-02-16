"""
Purpose: Extract case 18 split/power diagnostics for external statistical analysis.
Inputs: Benchmark case 18 generation and decomposition pipeline outputs.
Outputs: Extracted console/data artifacts for power analysis.
Expected runtime: ~20-120 seconds.
How to run: python debug_scripts/case_studies/q_case18_power_data_extraction__power__case18.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy import stats
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.tree.distributions import populate_distributions
from kl_clustering_analysis.hierarchy_analysis.statistics import (
    annotate_child_parent_divergence,
    annotate_sibling_divergence,
)
from benchmarks.shared.cases import get_test_cases_by_category
from kl_clustering_analysis.hierarchy_analysis.statistics.power_analysis import (
    power_wald_two_sample,
    cohens_h,
)
from kl_clustering_analysis import config as kl_config


REQUIRED_STATS_COLUMNS = [
    "Child_Parent_Divergence_P_Value",
    "Child_Parent_Divergence_Significant",
    "Sibling_Divergence_P_Value",
    "Sibling_BH_Different",
]

OUTPUT_PATH = Path(__file__).resolve().parent / "case_18_power_analysis.csv"


def _require_node_attr(tree, node_id: str, attr_name: str):
    """Get a required node attribute or raise an explicit error."""
    if attr_name not in tree.nodes[node_id]:
        raise ValueError(
            f"Missing required node attribute {attr_name!r} for node {node_id!r}"
        )
    return tree.nodes[node_id][attr_name]


def _validate_distribution(name: str, dist: np.ndarray) -> np.ndarray:
    """Validate that a distribution vector is finite and bounded in [0, 1]."""
    arr = np.asarray(dist, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} distribution is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} distribution contains non-finite values")
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError(
            f"{name} distribution must be in [0, 1]; got min={arr.min()}, max={arr.max()}"
        )
    return arr


def _validate_probability(name: str, p: float) -> float:
    """Validate scalar probability in [0, 1]."""
    if not np.isfinite(p):
        raise ValueError(f"{name} is not finite: {p}")
    if p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")
    return p


def generate_case_18_data():
    """Generate the exact data for case 18."""
    from benchmarks.shared.generators import generate_case_data

    cases = get_test_cases_by_category("overlapping_binary_unbalanced")
    case_18_matches = [c for c in cases if "unbal_4c_small" in c["name"]]
    if not case_18_matches:
        raise ValueError("Could not find case matching 'unbal_4c_small'")
    case_18 = case_18_matches[0]

    print(f"Case 18 config: {case_18}")

    # Generate data
    df, true_labels, features, metadata = generate_case_data(case_18)

    return df, true_labels, case_18


def build_and_annotate_tree(data):
    """Build tree and annotate with statistics."""
    dist = pdist(data.values, metric=kl_config.TREE_DISTANCE_METRIC)
    Z = linkage(dist, method=kl_config.TREE_LINKAGE_METHOD)
    tree = PosetTree.from_linkage(Z, leaf_names=data.index.tolist())

    populate_distributions(tree, data)

    # Create empty DataFrame with node_id as index
    tree.stats_df = pd.DataFrame(index=list(tree.nodes()))

    # Add leaf_count column
    tree.stats_df["leaf_count"] = [
        int(_require_node_attr(tree, n, "leaf_count")) for n in tree.nodes()
    ]

    tree.stats_df = annotate_child_parent_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )
    tree.stats_df = annotate_sibling_divergence(
        tree, tree.stats_df, significance_level_alpha=0.05
    )
    missing_cols = [c for c in REQUIRED_STATS_COLUMNS if c not in tree.stats_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required stats columns after annotation: {missing_cols}"
        )

    return tree


def extract_split_decisions(tree):
    """Extract all split decisions with power-relevant statistics."""
    decisions = []

    for node_id in tree.nodes():
        if tree.out_degree(node_id) == 0:
            continue

        children = list(tree.successors(node_id))
        if len(children) != 2:
            continue

        left, right = children

        n_parent = int(_require_node_attr(tree, node_id, "leaf_count"))
        n_left = int(_require_node_attr(tree, left, "leaf_count"))
        n_right = int(_require_node_attr(tree, right, "leaf_count"))

        dist_parent = _validate_distribution(
            f"node {node_id}", _require_node_attr(tree, node_id, "distribution")
        )
        dist_left = _validate_distribution(
            f"left child {left}", _require_node_attr(tree, left, "distribution")
        )
        dist_right = _validate_distribution(
            f"right child {right}", _require_node_attr(tree, right, "distribution")
        )

        if not hasattr(tree, "stats_df") or tree.stats_df is None:
            raise ValueError(
                "Tree is missing stats_df; annotate tree before extraction"
            )
        if node_id not in tree.stats_df.index:
            raise ValueError(f"Node {node_id!r} missing from stats_df index")
        row = tree.stats_df.loc[node_id]
        cp_pval = row["Child_Parent_Divergence_P_Value"]
        cp_significant = row["Child_Parent_Divergence_Significant"]
        sb_pval = row["Sibling_Divergence_P_Value"]
        sb_different = row["Sibling_BH_Different"]

        decisions.append(
            {
                "node_id": node_id,
                "n_parent": n_parent,
                "n_left": n_left,
                "n_right": n_right,
                "cp_pval": cp_pval,
                "cp_significant": cp_significant,
                "sb_pval": sb_pval,
                "sb_different": sb_different,
                "dist_parent_mean": float(np.mean(dist_parent)),
                "dist_left_mean": float(np.mean(dist_left)),
                "dist_right_mean": float(np.mean(dist_right)),
            }
        )

    return pd.DataFrame(decisions)


def compute_power_for_all_splits(df_decisions, min_effect_size=0.2):
    """Compute power for each split decision."""
    results = []

    for _, row in df_decisions.iterrows():
        n_left = row["n_left"]
        n_right = row["n_right"]

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

        p_left_raw = row["dist_left_mean"]
        p_right_raw = row["dist_right_mean"]
        if pd.isna(p_left_raw) or pd.isna(p_right_raw):
            raise ValueError(
                f"Missing required distribution means for node {row['node_id']!r}"
            )
        p_left = _validate_probability("p_left", float(p_left_raw))
        p_right = _validate_probability("p_right", float(p_right_raw))

        observed_effect = abs(cohens_h(p_left, p_right))
        effect_size = max(observed_effect, min_effect_size)

        power = power_wald_two_sample(
            n1=n_left,
            n2=n_right,
            p1=p_left,
            p2=p_right,
            alpha=0.05,
        )

        z_alpha = stats.norm.ppf(0.975)
        z_beta = stats.norm.ppf(0.80)
        pooled_var = (p_left * (1 - p_left) + p_right * (1 - p_right)) / 2
        target_delta = max(abs(p_left - p_right), min_effect_size)
        n_required = (z_alpha + z_beta) ** 2 * 2 * pooled_var / (target_delta**2)

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

    print("Step 1: Generating case 18 data...")
    data, true_labels, case_config = generate_case_18_data()
    print(f"  Data shape: {data.shape}")
    print(f"  True clusters: {len(set(true_labels))}")
    print()

    print("Step 2: Building and annotating tree...")
    tree = build_and_annotate_tree(data)
    print(f"  Total nodes: {len(tree.nodes())}")
    print()

    print("Step 3: Extracting split decisions...")
    df_decisions = extract_split_decisions(tree)
    print(f"  Total split decisions: {len(df_decisions)}")
    print()

    print("Step 4: Computing power...")
    df_power = compute_power_for_all_splits(df_decisions)
    print(f"  Splits with sufficient power (>=80%): {df_power['is_sufficient'].sum()}")
    print(f"  Splits with insufficient power: {(~df_power['is_sufficient']).sum()}")
    print()

    print("Step 5: Saving results...")
    df_combined = pd.concat([df_decisions, df_power], axis=1)
    output_path = OUTPUT_PATH
    df_combined.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path.resolve()}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Sample sizes:")
    print(df_combined[["n_left", "n_right"]].describe())
    print()
    print("Power distribution:")
    print(df_combined["power"].describe())
    print()
    print("Splits with power < 20%:", (df_combined["power"] < 0.2).sum())
    print(
        "Splits declared 'different' but power < 50%:",
        ((df_combined["sb_different"] == True) & (df_combined["power"] < 0.5)).sum(),
    )
