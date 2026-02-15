# hierarchy_analysis/

Tree decomposition engine and supporting modules.

## tree_decomposition.py — `TreeDecomposition`

The statistical engine. Walks the tree top-down applying three gates to carve clusters.

| Method                                                    | What it does                                                                                                                                                                |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__(tree, results_df, ...)`                         | Configure thresholds, pre-cache node metadata, run `_prepare_annotations()`, build fast lookup dicts for gate decisions.                                                    |
| `_prepare_annotations(df)`                                | If gate columns are missing, runs Gate 2 (`annotate_child_parent_divergence`) then Gate 3 (sibling test selected by `config.SIBLING_TEST_METHOD`). Idempotent.              |
| `_cache_node_metadata()`                                  | Extract distributions, leaf flags, labels into dicts for O(1) access.                                                                                                       |
| `_should_split(parent)`                                   | **Core gate logic (v1)**. Gate 1: binary children? Gate 2: at least one child diverges from parent? Gate 3: siblings significantly different? Returns bool.                 |
| `_should_split_v2(parent)`                                | **Enhanced gate logic**. Same gates, but when Gate 3 passes, runs `localize_divergence_signal()` to find WHERE divergence originates. Returns `(bool, LocalizationResult)`. |
| `decompose_tree()`                                        | Iterative depth-first traversal using `_should_split()`. Collects leaf sets, builds cluster assignments, optionally applies post-hoc merge.                                 |
| `decompose_tree_v2()`                                     | Same traversal with `_should_split_v2()`. Builds similarity/difference graphs, extracts constrained clusters, applies post-hoc merge.                                       |
| `_process_node_for_decomposition(node, stack, leaf_sets)` | Split → push children; merge → collect leaves.                                                                                                                              |
| `_test_node_pair_divergence(a, b)`                        | Wald χ² test between two arbitrary tree nodes (used by signal localization and post-hoc merge).                                                                             |
| `_test_cluster_pair_divergence(a, b, ancestor)`           | Same test but for cluster roots (used by post-hoc merge).                                                                                                                   |
| `_compute_cluster_distribution(root)`                     | Return `(distribution, sample_size)` for a cluster root.                                                                                                                    |
| `_maybe_apply_posthoc_merge_with_audit(assignments)`      | Optionally run `apply_posthoc_merge()`, return `(assignments, audit_trail)`.                                                                                                |

## cluster_assignments.py

Pure functions for building cluster metadata.

| Function                                          | What it does                                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `build_cluster_assignments(leaf_sets, find_root)` | Convert list of leaf sets → `{cluster_id: {root_node, leaves, size}}`.                                 |
| `build_sample_cluster_assignments(results)`       | Convert decomposition output → per-sample DataFrame with `cluster_id`, `cluster_root`, `cluster_size`. |

## posthoc_merge.py

Bottom-up merge pass to reduce over-splitting.

| Function                                               | What it does                                                                                                                                                                     |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `apply_posthoc_merge(cluster_roots, alpha, tree, ...)` | Collects all sibling-boundary cluster pairs, runs Wald tests, BH-corrects, greedily merges non-significant pairs (highest p-value first). Returns `(merged_roots, audit_trail)`. |
| `_get_leaf_clusters_under_node(node, roots, tree)`     | Find which cluster roots are descendants of a node.                                                                                                                              |

## signal_localization.py

Recursive cross-boundary testing to find WHERE divergence originates.

| Class/Function                                                                   | What it does                                                                                                                                                                                    |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SimilarityEdge`                                                                 | Dataclass: two nodes that are NOT significantly different (potential merge).                                                                                                                    |
| `LocalizationResult`                                                             | Dataclass: aggregate result of localization — similarity edges, difference pairs, depth reached.                                                                                                |
| `localize_divergence_signal(tree, left, right, test_fn, alpha, ...)`             | Recursively drills down cross-boundary pairs. Keeps only deepest test results (no double-counting). BH-corrects all leaf-level p-values. Categorizes into similarity edges vs difference pairs. |
| `extract_constrained_clusters(sim_graph, diff_graph, tree, merge_points)`        | Union-Find merge of similar pairs, respecting Cannot-Link constraints from difference graph.                                                                                                    |
| `merge_similarity_graphs(results)`                                               | Combine similarity edges from all split points into one graph.                                                                                                                                  |
| `merge_difference_graphs(results)`                                               | Combine difference pairs from all split points into one graph.                                                                                                                                  |
| `build_cross_boundary_similarity(tree, split_points, test_fn, alpha, max_depth)` | Run localization for all split points. Returns `{parent: LocalizationResult}`.                                                                                                                  |
