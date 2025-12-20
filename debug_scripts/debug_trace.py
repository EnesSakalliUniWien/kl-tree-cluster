"""Debug trace on sparse_features_extreme case."""

import numpy as np
from tests.test_cases_config import get_default_test_cases
from kl_clustering_analysis.benchmarking.pipeline import _generate_case_data
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from kl_clustering_analysis.tree.poset_tree import PosetTree
from kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test import (
    _sibling_divergence_chi_square_test,
)
from kl_clustering_analysis.hierarchy_analysis.statistics.random_projection import (
    should_use_projection,
)

# Get sparse_features_extreme test case
cases = get_default_test_cases()
tc = [c for c in cases if c.get("name") == "sparse_features_extreme"][0]

data_df, true_labels, X_orig, meta = _generate_case_data(tc)
print(f"Test case: {tc['name']}")
print(
    f"n_samples={meta['n_samples']}, n_features={meta['n_features']}, n_clusters={meta['n_clusters']}"
)
print()

# Build tree
Z = linkage(pdist(data_df.values, metric="hamming"), method="complete")
tree = PosetTree.from_linkage(Z, leaf_names=data_df.index.tolist())
tree.populate_node_divergences(leaf_data=data_df)

# Check a few nodes at different levels
print("Sample of sibling tests at different tree levels:")
print("-" * 80)

checked = 0
for node in tree.internal_nodes:
    children = list(tree.children(node))
    if len(children) == 2:
        left, right = children
        left_dist = tree.nodes[left].get("distribution")
        right_dist = tree.nodes[right].get("distribution")
        parent_dist = tree.nodes[node].get("distribution")
        left_n = tree.nodes[left].get("n_leaves", 1)
        right_n = tree.nodes[right].get("n_leaves", 1)

        if left_dist is not None and right_dist is not None:
            d = len(left_dist)
            n_eff = (2 * left_n * right_n) / (left_n + right_n)
            use_proj = should_use_projection(d, int(n_eff))

            # Call the test
            jsd, chi_sq, df, p = _sibling_divergence_chi_square_test(
                left_dist, right_dist, left_n, right_n, parent_dist
            )
            print(
                f"Node {node}: n_left={left_n}, n_right={right_n}, n_eff={n_eff:.1f}, d={d}"
            )
            print(
                f"  projected={use_proj}, chi_sq={chi_sq:.2f}, df={df:.0f}, p={p:.6f}"
            )
            print()

            checked += 1
            if checked >= 10:
                break
