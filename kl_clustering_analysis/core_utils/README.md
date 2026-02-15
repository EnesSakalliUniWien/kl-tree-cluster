# core_utils/

Shared utility functions used across the pipeline.

## data_utils.py

DataFrame helpers for extracting and writing node-level annotations.

| Function                                               | What it does                                                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `extract_leaf_counts(df, node_ids)`                    | Pull `leaf_count` column for specified nodes. Raises if missing.                                                    |
| `extract_node_distribution(tree, node_id)`             | Get `distribution` attribute from a tree node as float64 array.                                                     |
| `extract_node_sample_size(tree, node_id)`              | Get leaf count from node attributes. Fallback chain: `leaf_count` → `sample_size` → `n_leaves` → count descendants. |
| `assign_divergence_results(df, child_ids, pvals, ...)` | Write Gate 2 result columns (`Child_Parent_Divergence_*`) to DataFrame.                                             |
| `initialize_sibling_divergence_columns(df)`            | Initialize all Gate 3 output columns with defaults (False / NaN).                                                   |
| `extract_bool_column_dict(df, column)`                 | Convert a boolean DataFrame column to `{node_id: bool}` dict for O(1) lookups.                                      |

## pipeline_helpers.py

Convenience functions for quick prototyping.

| Function                                     | What it does                                                                   |
| -------------------------------------------- | ------------------------------------------------------------------------------ |
| `create_test_case_data(n, p, k, ...)`        | Generate synthetic binary data with k clusters.                                |
| `build_hierarchical_tree(X, method, metric)` | `pdist` → `linkage` → `PosetTree.from_linkage()`. One-liner tree construction. |
| `run_statistical_analysis(tree, X)`          | Populate divergences + decompose in one call.                                  |

## tree_utils.py

| Function                    | What it does                                                               |
| --------------------------- | -------------------------------------------------------------------------- |
| `compute_node_depths(tree)` | BFS from root → `{node_id: depth}` dict. Used by tree-aware BH correction. |
