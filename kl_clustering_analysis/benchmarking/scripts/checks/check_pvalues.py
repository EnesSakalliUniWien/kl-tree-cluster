"""Check raw vs corrected p-values for sibling tests."""

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from benchmarks.shared.generators import generate_random_feature_matrix
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis import config

# Generate data
data_dict, cluster_dict = generate_random_feature_matrix(
    n_rows=50,
    n_cols=30,
    n_clusters=3,
    entropy_param=0.05,
    balanced_clusters=True,
    random_seed=42,
)
data_df = pd.DataFrame.from_dict(data_dict, orient="index").astype(int)

# Build tree
Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
decomposition = tree.decompose(leaf_data=data_df)

stats_df = tree.stats_df

print("=" * 70)
print("RAW VS CORRECTED P-VALUES FOR KEY NODES")
print("=" * 70)

print(f"\nConfig: SIBLING_ALPHA = {config.SIBLING_ALPHA}")

# Get key nodes
root = tree.root()
children = list(tree.successors(root))

key_nodes = [root] + children

# Also get their children
for child in children:
    grandchildren = list(tree.successors(child))
    key_nodes.extend(grandchildren)

print(f"\nKey nodes to analyze: {key_nodes}")

print("\n" + "-" * 70)
print(
    f"{'Node':<8} {'leaves':>6} {'Raw P':>12} {'Corrected P':>12} {'BH_Diff':>8} {'Test Stat':>10} {'df':>6}"
)
print("-" * 70)

for node in key_nodes:
    if node not in stats_df.index:
        continue
    row = stats_df.loc[node]
    leaf_count = row.get("leaf_count", 0)
    raw_p = row.get("Sibling_Divergence_P_Value", np.nan)
    corrected_p = row.get("Sibling_Divergence_P_Value_Corrected", np.nan)
    bh_diff = row.get("Sibling_BH_Different", None)
    test_stat = row.get("Sibling_Test_Statistic", np.nan)
    df = row.get("Sibling_Degrees_of_Freedom", np.nan)

    raw_str = f"{raw_p:.2e}" if not np.isnan(raw_p) else "NaN"
    corr_str = f"{corrected_p:.2e}" if not np.isnan(corrected_p) else "NaN"

    print(
        f"{node:<8} {leaf_count:>6} {raw_str:>12} {corr_str:>12} {str(bh_diff):>8} {test_stat:>10.2f} {df:>6.1f}"
    )

# Check Local KL as well
print("\n" + "-" * 70)
print("LOCAL KL (CHILD VS PARENT) FOR KEY NODES")
print("-" * 70)
print(f"{'Node':<8} {'leaves':>6} {'Div_Sig':>10} {'Div P (BH)':>12}")
print("-" * 70)

for node in key_nodes:
    if node not in stats_df.index:
        continue
    row = stats_df.loc[node]
    leaf_count = row.get("leaf_count", 0)
    local_sig = row.get("Child_Parent_Divergence_Significant", None)
    local_p = row.get("Child_Parent_Divergence_P_Value_Corrected", np.nan)

    p_str = f"{local_p:.2e}" if local_p is not None and not np.isnan(local_p) else "NaN"
    print(f"{node:<8} {leaf_count:>6} {str(local_sig):>10} {p_str:>12}")

# Show what clusters SHOULD be found
print("\n" + "-" * 70)
print("TRUE CLUSTER STRUCTURE IN DATA")
print("-" * 70)

for cl_id in sorted(set(cluster_dict.values())):
    members = [k for k, v in cluster_dict.items() if v == cl_id]
    print(f"True cluster {cl_id}: {len(members)} members")

print("\n" + "-" * 70)
print("DIAGNOSIS")
print("-" * 70)
print(f"""
The problem: N95 has corrected p-value = 0.0367 > alpha = {config.SIBLING_ALPHA}

Even though the raw p-value might be significant, BH correction inflated it.

Options:
1. Increase SIBLING_ALPHA from 0.01 to 0.05
2. Use raw p-value instead of corrected for gating
3. The test is actually correct - N95's children are genuinely similar
""")
