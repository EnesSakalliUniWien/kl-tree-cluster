# tree/

Core tree data structure and distribution population.

## poset_tree.py — `PosetTree`

NetworkX `DiGraph` subclass. Central data structure for the entire pipeline.

| Method                                      | What it does                                                                                   |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `from_linkage(Z, leaf_names)`               | Build tree from SciPy linkage matrix. Computes branch lengths as merge distance deltas.        |
| `from_agglomerative(X, ...)`                | Build tree from sklearn `AgglomerativeClustering` fit.                                         |
| `from_undirected_edges(edges)`              | Orient an undirected weighted tree into a directed `PosetTree`.                                |
| `root()`                                    | Return the root node (in-degree 0), cached after first call.                                   |
| `get_leaves(node, return_labels)`           | Collect leaf labels globally or under a subtree.                                               |
| `compute_descendant_sets()`                 | Map every node → frozenset of its descendant leaf labels.                                      |
| `find_lca(a, b)`                            | Lowest common ancestor of two nodes using depth-based walk. O(depth).                          |
| `find_lca_for_set(nodes)`                   | LCA for a collection of nodes (iterative pairwise reduction).                                  |
| `populate_node_divergences(leaf_data)`      | Populate distributions, leaf counts, global/local KL divergences. Stores result in `stats_df`. |
| `decompose(results_df, leaf_data, **kw)`    | Thin facade: builds `TreeDecomposition`, runs `decompose_tree()` or `decompose_tree_v2()`.     |
| `build_sample_cluster_assignments(results)` | Per-sample cluster table from decomposition output.                                            |

## distributions.py

Bottom-up distribution population (called by `populate_node_divergences`).

| Function                                             | What it does                                                                                                                                          |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `populate_distributions(tree, leaf_data)`            | Postorder traversal: leaves get raw feature vectors, internal nodes get leaf-count-weighted means. Sets `distribution` and `leaf_count` on each node. |
| `_calculate_leaf_distribution(tree, node, data)`     | Set distribution for a single leaf from `leaf_data`.                                                                                                  |
| `_calculate_hierarchy_node_distribution(tree, node)` | Weighted average of children's distributions (weight = leaf count).                                                                                   |
