# tree/

Core tree data structure and distribution population.

## poset_tree.py — `PosetTree`

NetworkX `DiGraph` subclass. Central data structure for the entire pipeline.

| Method                                      | What it does                                                                                   |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `from_linkage(Z, leaf_names)`               | Build tree from SciPy linkage matrix. Computes branch lengths as merge distance deltas normalized by the root merge height, so root-to-leaf time sums to 1. |
| `from_agglomerative(X, ...)`                | Build tree from sklearn `AgglomerativeClustering` fit.                                         |
| `from_undirected_edges(edges)`              | Orient a non-empty undirected weighted tree into a directed `PosetTree`.                       |
| `root()`                                    | Return the root node (in-degree 0), cached after first call.                                   |
| `get_leaves(node, return_labels)`           | Collect leaf labels globally or under a subtree.                                               |
| `compute_descendant_sets()`                 | Map every node → frozenset of its descendant leaf labels.                                      |
| `find_lca(a, b)`                            | Lowest common ancestor of two nodes using depth-based walk. O(depth).                          |
| `find_lca_for_set(nodes)`                   | LCA for a non-empty collection of nodes.                                                       |
| `populate_node_divergences(leaf_data, feature_space=None)` | Populate flat raw-coordinate distributions and leaf counts. Categorical and continuous data require an explicit feature-space contract. Stores result in `annotations_df`. |
| `decompose(annotations_df, leaf_data, **kw)` | Thin facade: builds `TreeDecomposition` and runs `decompose_tree()`.                           |
| `build_sample_cluster_assignments(results)` | Per-sample cluster table from decomposition output.                                            |

## topology.py

Strict rooted-tree topology helpers used by `PosetTree`.

| Function                             | What it does                                                                 |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| `is_leaf(tree, node)`                | Read the explicit `is_leaf` node attribute.                                   |
| `get_leaf_values(tree, ...)`         | Collect leaf labels or ids without label/id fallback coercion.                |
| `compute_descendant_leaf_sets(tree)` | Bottom-up descendant leaf-set aggregation.                                    |
| `lowest_common_ancestor(tree, a, b)` | Depth-based LCA for nodes in the same rooted tree.                            |
| `lowest_common_ancestor_for_set(...)` | LCA for a non-empty node collection.                                          |

## distributions.py

Bottom-up distribution population (called by `populate_node_divergences`).

| Function                                             | What it does                                                                                                                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `populate_distributions(tree, leaf_data, feature_space=None)` | Postorder traversal: leaves get validated flat raw-coordinate vectors under the active `FeatureSpace`, and internal nodes get empirical subtree barycenters (leaf-count-weighted means). Sets `distribution` and `leaf_count` on each node; each continuous block gets an empirical descendant covariance under `continuous_covariance_by_block`: full dense covariance when feasible, or a diagonal variance vector in high-dimensional regimes. |
| `_calculate_leaf_distribution(tree, node, data)`     | Set distribution for a single leaf from `leaf_data`.                                                                                                  |
| `_calculate_hierarchy_node_distribution(tree, node)` | Compute an internal node's distribution as the empirical subtree barycenter of its children's distributions; also updates the parent's total descendant leaf count. |

## continuous_distance.py

Continuous tree-distance helpers for Gaussian/Brownian data.

| Function | What it does |
| -------- | ------------ |
| `continuous_time_distance_condensed(data, feature_space, ...)` | Returns condensed pairwise Brownian-time dissimilarities \((x_a-x_b)^T\widehat\Sigma^{-1}(x_a-x_b)/d\) for pure continuous feature spaces. |
| `estimate_continuous_covariance_by_block(data, feature_space, ...)` | Estimates shrinkage empirical covariance blocks with an identity target and scale-relative ridge for stable Mahalanobis-time distances. |
