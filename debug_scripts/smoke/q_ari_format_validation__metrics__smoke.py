"""
Purpose: Validate case-data shape assumptions and ARI computation consistency.
Inputs: Local benchmark case data loaded inside the script.
Outputs: Console diagnostics about data format and ARI values.
Expected runtime: ~5-20 seconds.
How to run: python debug_scripts/smoke/q_ari_format_validation__metrics__smoke.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.shared.cases import get_default_test_cases
from benchmarks.shared.generators import generate_case_data
from kl_clustering_analysis import config
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from kl_clustering_analysis.tree.poset_tree import PosetTree
import numpy as np

cases = get_default_test_cases()
c = cases[0]
print(f"Case: {c.get('name')}, Generator: {c.get('generator','blobs')}")
data_df, y, x_orig, meta = generate_case_data(c)
print(f"data_df shape: {data_df.shape}")
print(f"y shape: {y.shape}, unique y: {sorted(set(y))}")
print(f"data_df index[:5]: {list(data_df.index[:5])}")
print(f"meta keys: {list(meta.keys())}")

# Now run decomposition
config.SIBLING_TEST_METHOD = "wald"
dist = pdist(data_df.values, metric=config.TREE_DISTANCE_METRIC)
Z = linkage(dist, method=config.TREE_LINKAGE_METHOD)
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
decomp = tree.decompose(leaf_data=data_df, alpha_local=0.05, sibling_alpha=0.05)

assignments = decomp.get("cluster_assignments", {})
print(f"\nnum_clusters: {decomp.get('num_clusters')}")
print(f"assignments keys: {list(assignments.keys())[:5]}")
if assignments:
    first_key = list(assignments.keys())[0]
    first_val = assignments[first_key]
    print(f"First assignment key: {first_key}")
    print(f"First assignment value type: {type(first_val)}")
    print(f"First assignment value[:5]: {first_val[:5] if isinstance(first_val, list) else first_val}")

# Build labels
label_map = {}
for cluster_id, (root, members) in enumerate(assignments.items()):
    for m in members:
        label_map[m] = cluster_id

pred = [label_map.get(name, -1) for name in data_df.index]
print(f"\npred[:10]: {pred[:10]}")
print(f"true[:10]: {list(y[:10])}")

ari = adjusted_rand_score(y, pred)
print(f"ARI: {ari:.4f}")
