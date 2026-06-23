# core_utils/

Shared utility functions used across the pipeline.

## data_utils.py

DataFrame helpers for extracting and writing node-level annotations.

| Function                                               | What it does                                                                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `extract_leaf_counts(df, node_ids)`                    | Pull `leaf_count` column for specified nodes. Raises if missing.                                                    |
| `extract_node_distribution(tree, node_id)`             | Get `distribution` attribute from a tree node as float64 array.                                                     |
| `extract_node_sample_size(tree, node_id)`              | Get required `leaf_count` from node attributes. Raises when the annotated-tree contract is missing.                  |
| `assign_divergence_results(df, child_ids, stats, pvals, ...)` | Write edge projected-Wald statistic, p-values, degrees of freedom, and Tree-BH state to DataFrame.                    |
| `initialize_sibling_divergence_columns(df)`            | Initialize all sibling-divergence output columns with defaults (False / NaN).                                        |
| `extract_row_column_maps(df)`                          | Materialize a DataFrame as both `{node_id: {column: value}}` and `{column: {node_id: value}}` for O(1) lookups.    |
| `extract_bool_column_dict(df, column)`                 | Convert a boolean DataFrame column to `{node_id: bool}` dict for O(1) lookups.                                      |

## tree_utils.py

| Function                    | What it does                                                               |
| --------------------------- | -------------------------------------------------------------------------- |
| `compute_node_depths(tree)` | BFS from root → `{node_id: depth}` dict. Used by tree-aware BH correction. |
