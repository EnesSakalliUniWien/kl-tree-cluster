# Function Map and Development History

_Generated on 2026-02-16 09:19 UTC_

## Scope

- Package scanned: `kl_clustering_analysis`
- Benchmark/debug entrypoints scanned: `benchmarks/full/run.py`, `benchmarks/shared/debug_trace.py`, `comparison_output.log`, `debug_scripts/*.py`
- Function map is AST-based (classes, methods, functions) plus docstring first line for purpose.
- Per-function history uses `git blame` on the definition line when available; local/untracked code is marked explicitly.

## Current Working Tree Notes

- Detected local workspace changes (not modified by this report generation):
  - ` M kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/__init__.py`
  - ` M kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py`
  - `?? benchmarks/compare_sibling_methods.py`
  - `?? debug_ca.py`
  - `?? kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/cousin_weighted_wald.py`

## Debug Snapshot (from `comparison_output.log`)

- Log lines: 2094
- `No eligible parent nodes ...` warnings: 14
- `divide by zero encountered in divide` warnings (OPTICS): 8
- `Traceback` occurrences: 0
- `FAILED` token occurrences: 0

## Debug Entry Flow

- `benchmarks/full/run.py::run_benchmarks` orchestrates full-case execution, optional calibration, PDF merge, and optional failure diagnosis.
- `benchmarks/shared/debug_trace.py::diagnose_benchmark_failures` scans low-ARI cases and classifies likely under-split/over-split patterns.
- `benchmarks/shared/debug_trace.py::analyze_single_case` performs root split checks and sibling significance heuristics per audit CSV.

## Operational Structure (Core Pipeline)

1. Build hierarchy
- `kl_clustering_analysis/tree/poset_tree.py::PosetTree.from_linkage` and related constructors create the directed tree.
- `kl_clustering_analysis/tree/distributions.py::populate_distributions` computes per-node distributions and leaf counts.
- `kl_clustering_analysis/information_metrics/kl_divergence/divergence_metrics.py::_populate_local_kl` and `_populate_global_kl` attach divergence metrics.
2. Annotate statistical evidence
- `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py::annotate_child_parent_divergence` marks child-parent signal edges.
- `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/*.py::annotate_sibling_divergence*` runs sibling-level separation tests.
- `kl_clustering_analysis/hierarchy_analysis/statistics/multiple_testing/*.py` applies BH/TreeBH style multiplicity control.
3. Decide splits and emit clusters
- `kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py::TreeDecomposition._should_split` is the main gate sequence (binary structure -> child-parent signal -> sibling difference).
- `kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py::TreeDecomposition.decompose_tree` performs traversal and cluster extraction.
- `kl_clustering_analysis/hierarchy_analysis/posthoc_merge.py::apply_posthoc_merge` optionally merges statistically similar adjacent clusters.
4. Produce reporting/debug artifacts
- `kl_clustering_analysis/hierarchy_analysis/cluster_assignments.py` builds cluster/sample assignment tables.
- `benchmarks/full/run.py::run_benchmarks` writes CSV/PDF benchmark outputs.
- `benchmarks/shared/debug_trace.py` converts low-ARI outcomes into failure-mode diagnostics.

## Internal Module Dependency Map

- `kl_clustering_analysis.__init__` -> `kl_clustering_analysis`
- `kl_clustering_analysis.config` -> (none)
- `kl_clustering_analysis.core_utils.data_utils` -> (none)
- `kl_clustering_analysis.core_utils.pipeline_helpers` -> `kl_clustering_analysis`, `kl_clustering_analysis.hierarchy_analysis.statistics`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test`, `kl_clustering_analysis.tree.poset_tree`
- `kl_clustering_analysis.core_utils.tree_utils` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.__init__` -> `kl_clustering_analysis.hierarchy_analysis.statistics`, `kl_clustering_analysis.hierarchy_analysis.tree_decomposition`, `kl_clustering_analysis.information_metrics`
- `kl_clustering_analysis.hierarchy_analysis.cluster_assignments` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.posthoc_merge` -> `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`
- `kl_clustering_analysis.hierarchy_analysis.signal_localization` -> `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`
- `kl_clustering_analysis.hierarchy_analysis.statistics.__init__` -> `kl_clustering_analysis.hierarchy_analysis.statistics.clt_validity`, `kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence`
- `kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.statistics.categorical_mahalanobis` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.statistics.clt_validity` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.__init__` -> `kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.chi_square_test`, `kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance`, `kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.utils`
- `kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.chi_square_test` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.edge_significance` -> `kl_clustering_analysis`, `kl_clustering_analysis.core_utils.data_utils`, `kl_clustering_analysis.core_utils.tree_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`, `kl_clustering_analysis.hierarchy_analysis.statistics.random_projection`
- `kl_clustering_analysis.hierarchy_analysis.statistics.kl_tests.utils` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.__init__` -> `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.dispatcher`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.flat_correction`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.level_wise_correction`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.tree_bh_correction`
- `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.dispatcher` -> `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.flat_correction`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.level_wise_correction`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.tree_bh_correction`
- `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.flat_correction` -> `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base`
- `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.level_wise_correction` -> `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.base`
- `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing.tree_bh_correction` -> `kl_clustering_analysis.core_utils.tree_utils`
- `kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.statistics.power_analysis` -> (none)
- `kl_clustering_analysis.hierarchy_analysis.statistics.random_projection` -> `kl_clustering_analysis`
- `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.__init__` -> `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_adjusted_wald`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_calibrated_test`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_tree_guided`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test`
- `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_adjusted_wald` -> `kl_clustering_analysis`, `kl_clustering_analysis.core_utils.data_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test`
- `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_calibrated_test` -> `kl_clustering_analysis`, `kl_clustering_analysis.core_utils.data_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test`
- `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_tree_guided` -> `kl_clustering_analysis`, `kl_clustering_analysis.core_utils.data_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test`
- `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.cousin_weighted_wald` -> `kl_clustering_analysis`, `kl_clustering_analysis.core_utils.data_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test`
- `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test` -> `kl_clustering_analysis`, `kl_clustering_analysis.core_utils.data_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.categorical_mahalanobis`, `kl_clustering_analysis.hierarchy_analysis.statistics.multiple_testing`, `kl_clustering_analysis.hierarchy_analysis.statistics.pooled_variance`, `kl_clustering_analysis.hierarchy_analysis.statistics.random_projection`
- `kl_clustering_analysis.hierarchy_analysis.tree_decomposition` -> `kl_clustering_analysis`, `kl_clustering_analysis.core_utils.data_utils`, `kl_clustering_analysis.hierarchy_analysis.cluster_assignments`, `kl_clustering_analysis.hierarchy_analysis.posthoc_merge`, `kl_clustering_analysis.hierarchy_analysis.signal_localization`, `kl_clustering_analysis.hierarchy_analysis.statistics`, `kl_clustering_analysis.hierarchy_analysis.statistics.branch_length_utils`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence`, `kl_clustering_analysis.hierarchy_analysis.statistics.sibling_divergence.sibling_divergence_test`, `kl_clustering_analysis.tree.poset_tree`
- `kl_clustering_analysis.information_metrics.__init__` -> `kl_clustering_analysis.information_metrics.kl_divergence`
- `kl_clustering_analysis.information_metrics.kl_divergence.__init__` -> `kl_clustering_analysis.information_metrics.kl_divergence.divergence_metrics`
- `kl_clustering_analysis.information_metrics.kl_divergence.divergence_metrics` -> `kl_clustering_analysis`
- `kl_clustering_analysis.plot.cluster_color_mapping` -> (none)
- `kl_clustering_analysis.plot.cluster_tree_visualization` -> `kl_clustering_analysis.plot.cluster_color_mapping`
- `kl_clustering_analysis.tree.__init__` -> (none)
- `kl_clustering_analysis.tree.distributions` -> (none)
- `kl_clustering_analysis.tree.poset_tree` -> `kl_clustering_analysis`, `kl_clustering_analysis.core_utils.tree_utils`, `kl_clustering_analysis.hierarchy_analysis.cluster_assignments`, `kl_clustering_analysis.hierarchy_analysis.tree_decomposition`, `kl_clustering_analysis.information_metrics.kl_divergence.divergence_metrics`, `kl_clustering_analysis.tree.distributions`
- `kl_clustering_analysis.tree.util.__init__` -> (none)
- `kl_clustering_analysis.tree.util.tree_construct_helpers` -> (none)

## Development Timeline (Git)

- 2025-11-05 `c348d47` Packaged correctly, and fixed uv installation.
- 2025-11-05 `884ad97` Addapted, tests
- 2025-11-30 `97c6d9b` Removed and clean up code environment
- 2025-12-17 `8111d55` Updated and Restructured Projects.
- 2025-12-18 `d1b0bb5` Update usage of Jensen-Shannon divergence with a chi-square approximation for statistical significance, consistent with the KL divergence chi-square test.
- 2025-12-20 `66d3f2a` feat: Include plots for all clustering methods in benchmark visualization
- 2025-12-20 `375f08a` feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline
- 2025-12-24 `3cbdaf9` Add global divergence weighting and refactor multiple testing
- 2025-12-25 `e3d229d` Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge
- 2025-12-25 `48ec633` benchmark/plots: stream PDFs, timestamp outputs, and clean up MI + tree primitives
- 2026-01-22 `e69adcc` Small Updates, for the parent check and sibling check.
- 2026-02-07 `ea21bff` Branch Length Test
- 2026-02-13 `c87a6f1` Fixes for branchlengths, and benchmark dedup
- 2026-02-15 `ac77484` refactor: dead code removal + extract cluster_assignments module
- 2026-02-15 `bf7c1e7` feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs

## Function Map by File

### `kl_clustering_analysis/core_utils/data_utils.py`

- File history: 9 commits, first `2025-11-02 9f94bc27` (SMall Updates, better ReadMe. Modularisation of the statistical ,methods.), last `2026-02-15 ac774849` (refactor: dead code removal + extract cluster_assignments module).
- Line 8: `function extract_leaf_counts` -> Extract leaf counts for specified nodes.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 44: `function extract_node_distribution` -> Extract distribution for a single node, converted to float64.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 76: `function extract_node_sample_size` -> Extract sample size (leaf count) for a node.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 122: `function assign_divergence_results` -> Assign child-parent divergence test results to the nodes dataframe.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 196: `function initialize_sibling_divergence_columns` -> Initialize sibling divergence output columns in the dataframe.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 220: `function extract_bool_column_dict` -> Extract a boolean column from DataFrame as a dictionary.. `def-history: 2025-12-24 e3d229d1 Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge`

### `kl_clustering_analysis/core_utils/pipeline_helpers.py`

- File history: 4 commits, first `2025-11-30 97c6d9ba` (Removed and clean up code environment), last `2025-12-25 e3d229d1` (Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge).
- Line 26: `function create_test_case_data` -> Create synthetic test data using the same method as the validation suite.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 63: `function build_hierarchical_tree` -> Build hierarchical tree from feature matrix using scipy linkage.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 84: `function run_statistical_analysis` -> Run statistical tests on the hierarchical tree.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`

### `kl_clustering_analysis/core_utils/tree_utils.py`

- File history: 2 commits, first `2025-12-25 e3d229d1` (Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge), last `2026-02-15 ac774849` (refactor: dead code removal + extract cluster_assignments module).
- Line 14: `function compute_node_depths` -> Compute depth of each node from the root via BFS.. `def-history: 2025-12-24 e3d229d1 Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge`

### `kl_clustering_analysis/hierarchy_analysis/cluster_assignments.py`

- File history: 2 commits, first `2026-02-15 ac774849` (refactor: dead code removal + extract cluster_assignments module), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 16: `function build_cluster_assignments` -> Build a cluster assignment dictionary from collected leaf sets.. `def-history: 2026-02-15 ac774849 refactor: dead code removal + extract cluster_assignments module`
- Line 48: `function build_sample_cluster_assignments` -> Build per-sample cluster assignments from decomposition output.. `def-history: 2026-02-15 ac774849 refactor: dead code removal + extract cluster_assignments module`

### `kl_clustering_analysis/hierarchy_analysis/posthoc_merge.py`
- File history: 5 commits, first `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 16: `function _get_leaf_clusters_under_node` -> Get all cluster root nodes that are descendants of the given node.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 44: `function apply_posthoc_merge` -> Apply tree-respecting post-hoc merging to reduce over-splitting.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`

### `kl_clustering_analysis/hierarchy_analysis/signal_localization.py`
- File history: 4 commits, first `2026-02-07 ea21bff4` (Branch Length Test), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 41: `class SimilarityEdge` -> An edge indicating two nodes are NOT significantly different.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 50: `method SimilarityEdge.__hash__` -> (no docstring; infer from name/context). `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 53: `method SimilarityEdge.__eq__` -> (no docstring; infer from name/context). `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 60: `class LocalizationResult` -> Result of signal localization between two subtrees.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 94: `method LocalizationResult.has_soft_boundaries` -> True if there are cross-boundary similarities (potential merges).. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 99: `method LocalizationResult.all_different` -> True if all tested pairs are significantly different.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 103: `method LocalizationResult.get_similarity_graph` -> Build a graph where edges indicate non-significant differences.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 121: `function _get_children` -> Get immediate children of a node.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 126: `function _is_leaf` -> Check if a node is a leaf (no children).. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 136: `function localize_divergence_signal` -> Recursively test cross-boundary pairs to localize divergence.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 287: `function merge_difference_graphs` -> Merge all difference pairs into a single graph.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 311: `function extract_constrained_clusters` -> Extract clusters using similarity edges constrained by difference pairs.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 446: `function _get_all_leaves` -> Get all leaf labels under a given node.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 466: `function build_cross_boundary_similarity` -> Build localization results for all split points in the tree.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 516: `function merge_similarity_graphs` -> Merge all similarity edges into a single graph.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`

### `kl_clustering_analysis/hierarchy_analysis/statistics/branch_length_utils.py`
- File history: 1 commits, first `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 14: `function sanitize_positive_branch_length` -> Return a finite positive branch length, else ``None``.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 27: `function compute_mean_branch_length` -> Compute mean branch length across valid edges.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`

### `kl_clustering_analysis/hierarchy_analysis/statistics/categorical_mahalanobis.py`
- File history: 1 commits, first `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 13: `function categorical_whitened_vector` -> Build a covariance-whitened categorical difference vector.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 79: `function mahalanobis_wald_categorical` -> Compute multinomial Wald statistic with drop-last parametrization.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`

### `kl_clustering_analysis/hierarchy_analysis/statistics/clt_validity.py`
- File history: 2 commits, first `2026-02-13 c87a6f13` (Fixes for branchlengths, and benchmark dedup), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 40: `class CLTValidityResult` -> Result of CLT validity check for a node.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 67: `function compute_third_absolute_moment` -> Compute third absolute central moment for Bernoulli distribution.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 90: `function compute_third_absolute_moment_categorical` -> Compute third absolute central moment for Categorical distribution.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 121: `function compute_variance_bernoulli` -> Compute variance for Bernoulli distribution.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 141: `function berry_esseen_bound` -> Compute Berry-Esseen upper bound on normal approximation error.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 185: `function check_clt_validity_bernoulli` -> Check if CLT approximation is valid for Bernoulli features.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 292: `function compute_minimum_n_berry_esseen` -> Compute minimum sample size required for CLT validity.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 334: `function check_split_clt_validity` -> Check CLT validity for both children of a split.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`

### `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/chi_square_test.py`
- File history: 4 commits, first `2025-11-30 97c6d9ba` (Removed and clean up code environment), last `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing).
- Line 15: `function kl_divergence_chi_square_test` -> Test KL divergence significance using chi-square approximation.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 73: `function kl_divergence_chi_square_test_batch` -> Vectorized chi-square test for multiple KL divergence values.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`

### `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/edge_significance.py`
- File history: 9 commits, first `2025-11-30 97c6d9ba` (Removed and clean up code environment), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 55: `function _compute_standardized_z` -> Compute standardized z-scores for child vs parent.. `def-history: 2025-12-24 e3d229d1 Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge`
- Line 138: `function _compute_projected_test` -> Compute projected Wald test for one edge.. `def-history: 2025-12-24 e3d229d1 Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge`
- Line 227: `function _compute_p_values_via_projection` -> Compute p-values for all edges via random projection.. `def-history: 2025-12-24 e3d229d1 Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge`
- Line 308: `function annotate_child_parent_divergence` -> Test child-parent divergence using projected Wald test.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`

### `kl_clustering_analysis/hierarchy_analysis/statistics/kl_tests/utils.py`
- File history: 2 commits, first `2025-11-30 97c6d9ba` (Removed and clean up code environment), last `2025-12-20 375f08a9` (feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline).
- Line 8: `function get_local_kl_series` -> Extract the local KL divergence column as a float Series.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`

### `kl_clustering_analysis/hierarchy_analysis/statistics/multiple_testing/base.py`
- File history: 2 commits, first `2025-11-30 97c6d9ba` (Removed and clean up code environment), last `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing).
- Line 21: `function benjamini_hochberg_correction` -> Apply Benjamini-Hochberg FDR correction to p-values.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`

### `kl_clustering_analysis/hierarchy_analysis/statistics/multiple_testing/dispatcher.py`
- File history: 1 commits, first `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing), last `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing).
- Line 19: `function apply_multiple_testing_correction` -> Apply BH correction with optional hierarchical structure awareness.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`

### `kl_clustering_analysis/hierarchy_analysis/statistics/multiple_testing/flat_correction.py`
- File history: 1 commits, first `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing), last `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing).
- Line 16: `function flat_bh_correction` -> Apply standard flat BH correction across all p-values.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`

### `kl_clustering_analysis/hierarchy_analysis/statistics/multiple_testing/level_wise_correction.py`
- File history: 1 commits, first `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing), last `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing).
- Line 17: `function level_wise_bh_correction` -> Apply BH correction separately at each tree level.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`

### `kl_clustering_analysis/hierarchy_analysis/statistics/multiple_testing/tree_bh_correction.py`
- File history: 3 commits, first `2025-12-20 375f08a9` (feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline), last `2025-12-25 e3d229d1` (Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge).
- Line 31: `class TreeBHResult` -> Results from TreeBH correction.. `def-history: 2025-12-20 375f08a9 feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline`
- Line 52: `function _get_root_nodes` -> Find root nodes (nodes with no parents).. `def-history: 2025-12-20 375f08a9 feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline`
- Line 57: `function _get_families_by_parent` -> Group child indices by their parent node.. `def-history: 2025-12-20 375f08a9 feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline`
- Line 77: `function _compute_family_threshold` -> Compute the adjusted threshold for a family.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 130: `function tree_bh_correction` -> Apply TreeBH hierarchical FDR correction.. `def-history: 2025-12-20 375f08a9 feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline`

### `kl_clustering_analysis/hierarchy_analysis/statistics/pooled_variance.py`
- File history: 5 commits, first `2025-12-24 3cbdaf91` (Add global divergence weighting and refactor multiple testing), last `2026-02-15 ac774849` (refactor: dead code removal + extract cluster_assignments module).
- Line 32: `function _is_categorical` -> Check if array represents categorical distributions (2D).. `def-history: 2026-01-22 e69adcc3 Small Updates, for the parent check and sibling check.`
- Line 37: `function _flatten_categorical` -> Flatten categorical distribution to 1D for Wald test.. `def-history: 2026-01-22 e69adcc3 Small Updates, for the parent check and sibling check.`
- Line 48: `function compute_pooled_proportion` -> Compute the pooled proportion estimate under H₀: θ₁ = θ₂.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 83: `function compute_pooled_variance` -> Compute the variance of the difference between two proportions.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 125: `function standardize_proportion_difference` -> Compute standardized difference (z-scores) between two proportions.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`

### `kl_clustering_analysis/hierarchy_analysis/statistics/power_analysis.py`
- File history: 2 commits, first `2026-02-13 c87a6f13` (Fixes for branchlengths, and benchmark dedup), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 26: `class PowerResult` -> Result of a power analysis calculation.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 47: `function cohens_h` -> Compute Cohen's h effect size for two proportions.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 75: `function power_wald_two_sample` -> Compute power for a two-sample Wald test of proportions.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 141: `function power_wald_nested` -> Compute power for nested child-parent divergence test.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 193: `function compute_child_parent_power` -> Compute power for child-parent divergence test.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 264: `function compute_sibling_power` -> Compute power for sibling divergence test.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 322: `function check_power_sufficient` -> Quick check if power is sufficient for a test.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`

### `kl_clustering_analysis/hierarchy_analysis/statistics/random_projection.py`
- File history: 8 commits, first `2025-12-20 375f08a9` (feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline), last `2026-02-15 ac774849` (refactor: dead code removal + extract cluster_assignments module).
- Line 33: `function _generate_structured_orthonormal_rows` -> Generate a sparse orthonormal projection with signed coordinate rows.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 53: `function compute_projection_dimension` -> Compute target dimension for random projection using Johnson-Lindenstrauss lemma.. `def-history: 2025-12-20 375f08a9 feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline`
- Line 104: `function _generate_orthonormal_projection` -> Generate an orthonormal projection matrix via QR decomposition.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 184: `function generate_projection_matrix` -> Generate an orthonormal random projection matrix.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 215: `function derive_projection_seed` -> Derive a deterministic per-test projection seed.. `def-history: 2026-02-13 c87a6f13 Fixes for branchlengths, and benchmark dedup`
- Line 239: `function _maybe_audit_projection` -> Optionally audit projection matrices to TensorBoard.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`

### `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/cousin_adjusted_wald.py`
- File history: 1 commits, first `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 69: `class _SiblingRecord` -> Per–sibling-pair record for calibration pipeline.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 84: `class _CalibrationModel` -> Result of fitting the inflation model.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 100: `function _fit_inflation_model` -> Estimate the post-selection inflation factor from null-like pairs.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 231: `function _predict_c` -> Predict inflation factor ĉ for a focal pair.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 275: `function _collect_all_pairs` -> Collect ALL binary-child parent nodes and compute raw Wald stats.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 344: `function _deflate_and_test` -> Deflate focal pairs and compute adjusted p-values.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 380: `function _apply_results_adjusted` -> Apply deflated results with BH correction to DataFrame.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 431: `function annotate_sibling_divergence_adjusted` -> Test sibling divergence using cousin-adjusted Wald.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`

### `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/cousin_calibrated_test.py`
- File history: 1 commits, first `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 67: `function _compute_sibling_stat` -> Compute projected Wald χ² statistic for a sibling pair.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 99: `function _get_uncle` -> Find the uncle node (parent's sibling) and grandparent.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 117: `function _get_cousin_reference` -> Compute the cousin-level reference statistic T_{UL,UR}.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 159: `function cousin_ftest` -> Cousin-calibrated F-test for sibling divergence.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 241: `function _collect_test_arguments_cousin` -> Collect sibling pairs eligible for testing.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 278: `function _run_cousin_tests` -> Execute cousin F-tests for all collected pairs.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 305: `function _apply_results_cousin` -> Apply test results with BH correction to dataframe.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 354: `function annotate_sibling_divergence_cousin` -> Test sibling divergence using cousin-calibrated F-test.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`

### `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/cousin_tree_guided.py`

- File history: 1 commits, first `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 86: `class _SiblingRecord` -> Per–sibling-pair record for calibration pipeline.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 105: `function _build_null_index` -> Index null-like pairs by their parent node for O(1) lookup.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 120: `function _collect_null_ratios_in_subtree` -> Collect T/k ratios from null-like pairs within a subtree.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 146: `function _find_nearest_null_cousin` -> Walk the tree bidirectionally from the focal parent to find nearest null-like relatives.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 233: `function _compute_local_c_hat` -> Compute the inflation factor ĉ for a focal pair using tree-guided search.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 283: `function _collect_all_pairs` -> Collect ALL binary-child parent nodes and compute raw Wald stats.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 348: `function _deflate_and_test` -> Deflate focal pairs using tree-guided cousin search.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 394: `function _apply_results` -> Apply deflated results with BH correction to DataFrame.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 445: `function annotate_sibling_divergence_tree_guided` -> Test sibling divergence using tree-guided cousin calibration.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`

### `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/cousin_weighted_wald.py`

- File history: local/untracked in current workspace (not present in HEAD).
- Line 96: `class _WeightedRecord` -> Per-sibling-pair record with continuous weight.. `def-history: local/uncommitted or unavailable`
- Line 112: `class _WeightedCalibrationModel` -> Result of fitting the weighted inflation model.. `def-history: local/uncommitted or unavailable`
- Line 128: `function _get_edge_pvalue` -> Get the BH-corrected edge p-value for a node.. `def-history: local/uncommitted or unavailable`
- Line 148: `function _weighted_median` -> Compute the weighted median of values with given weights.. `def-history: local/uncommitted or unavailable`
- Line 161: `function _fit_weighted_inflation_model` -> Estimate post-selection inflation using weighted regression on ALL pairs.. `def-history: local/uncommitted or unavailable`
- Line 316: `function _predict_c` -> Predict inflation factor ĉ for a pair.. `def-history: local/uncommitted or unavailable`
- Line 349: `function _collect_weighted_pairs` -> Collect ALL sibling pairs with continuous weights from edge p-values.. `def-history: local/uncommitted or unavailable`
- Line 430: `function _deflate_and_test` -> Deflate focal pairs and compute adjusted p-values.. `def-history: local/uncommitted or unavailable`
- Line 460: `function _apply_results` -> Apply deflated results with BH correction to DataFrame.. `def-history: local/uncommitted or unavailable`
- Line 511: `function annotate_sibling_divergence_weighted` -> Test sibling divergence using weighted cousin-adjusted Wald.. `def-history: local/uncommitted or unavailable`

### `kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/sibling_divergence_test.py`

- File history: 8 commits, first `2025-12-18 d1b0bb59` (Update usage of Jensen-Shannon divergence with a chi-square approximation for statistical significance, consistent with the KL divergence chi-square test.), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 46: `function _compute_chi_square_pvalue` -> Compute χ²(k) test statistic and p-value from projected z-scores.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 55: `function sibling_divergence_test` -> Two-sample Wald test for sibling divergence.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 181: `function _get_binary_children` -> Return (left, right) children if parent has exactly 2, else None.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 189: `function _either_child_significant` -> Check if at least one child has significant child-parent divergence.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 198: `function _get_sibling_data` -> Extract distributions, sample sizes, and branch lengths for sibling pair.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 239: `function _collect_test_arguments` -> Collect sibling pairs eligible for testing.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 285: `function _run_tests` -> Execute sibling divergence tests for all collected pairs.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 313: `function _apply_results` -> Apply test results with BH correction to dataframe.. `def-history: 2025-12-24 3cbdaf91 Add global divergence weighting and refactor multiple testing`
- Line 357: `function annotate_sibling_divergence` -> Test sibling divergence and annotate results in dataframe.. `def-history: 2025-12-18 d1b0bb59 Update usage of Jensen-Shannon divergence with a chi-square approximation`

### `kl_clustering_analysis/hierarchy_analysis/tree_decomposition.py`

- File history: 17 commits, first `2025-10-29 fc3a4b0c` (Kl-Te-Cluster), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 41: `class TreeDecomposition` -> Annotate a hierarchy with significance tests and carve it into clusters.. `def-history: 2025-12-20 375f08a9 feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline`
- Line 62: `method TreeDecomposition.__init__` -> Configure decomposition thresholds and pre-compute reusable metadata.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 170: `method TreeDecomposition._prepare_annotations` -> Ensure statistical annotation columns are present on *results_df*.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 247: `method TreeDecomposition._cache_node_metadata` -> Cache node attributes for fast repeated access during decomposition.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 274: `method TreeDecomposition._get_all_leaves` -> Return the leaf partition beneath a node using precomputed cache.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 291: `method TreeDecomposition._find_cluster_root` -> Identify the lowest common ancestor for a collection of leaf labels.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 318: `method TreeDecomposition._leaf_count` -> (no docstring; infer from name/context). `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 321: `method TreeDecomposition._child_diverges_from_parent` -> Determine whether the local divergence test flags ``child`` as divergent.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 339: `method TreeDecomposition._process_node_for_decomposition` -> Apply split-or-merge decision for one node during traversal.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 366: `method TreeDecomposition._iterate_nodes_to_visit` -> Yield nodes from a mutable last in, first out list exactly once.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 386: `method TreeDecomposition._build_cluster_assignments` -> Build cluster assignment dictionary from collected leaf sets.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 395: `method TreeDecomposition._should_split` -> Evaluate statistical gates and return ``True`` when parent should split.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 463: `method TreeDecomposition._test_node_pair_divergence` -> Test divergence between two arbitrary tree nodes.. `def-history: 2026-02-15 ac774849 refactor: dead code removal + extract cluster_assignments module`
- Line 516: `method TreeDecomposition._check_edge_significance` -> Check if a node is significantly different from its parent.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 526: `method TreeDecomposition._should_split_v2` -> Enhanced split decision with signal localization.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 617: `method TreeDecomposition._compute_cluster_distribution` -> Compute the distribution for a cluster.. `def-history: 2026-02-15 ac774849 refactor: dead code removal + extract cluster_assignments module`
- Line 636: `method TreeDecomposition._test_cluster_pair_divergence` -> Test if two clusters are significantly different.. `def-history: 2025-12-20 375f08a9 feat: implement TreeBH hierarchical FDR correction and refactor statistics pipeline`
- Line 706: `method TreeDecomposition.decompose_tree` -> Return cluster assignments by iteratively traversing the hierarchy.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 741: `method TreeDecomposition.decompose_tree_v2` -> Return cluster assignments using signal localization for soft boundaries.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 838: `method TreeDecomposition._maybe_apply_posthoc_merge_with_audit` -> Optionally apply tree-respecting post-hoc merge and return audit trail.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`

### `kl_clustering_analysis/information_metrics/kl_divergence/divergence_metrics.py`
- File history: 6 commits, first `2025-11-30 97c6d9ba` (Removed and clean up code environment), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 15: `function _kl_categorical_general` -> KL divergence for Categorical distributions (including Bernoulli).. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 49: `function _kl_poisson` -> KL divergence for Poisson distributions.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 68: `function calculate_kl_divergence_vector` -> Element-wise KL divergence: D_KL(Q||P).. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 96: `function calculate_kl_divergence_per_feature` -> Per-feature KL divergence for specified distribution.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 131: `function _populate_local_kl` -> Compute LOCAL KL divergence for each edge: KL(child||parent).. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 145: `function _populate_global_kl` -> Compute GLOBAL KL divergence for all nodes: KL(node||root).. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 174: `function _extract_hierarchy_statistics` -> Collect distributions and KL metrics into a DataFrame indexed by node_id.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`

### `kl_clustering_analysis/plot/cluster_color_mapping.py`
- File history: 1 commits, first `2025-12-17 8111d55f` (Updated and Restructured Projects.), last `2025-12-17 8111d55f` (Updated and Restructured Projects.).
- Line 23: `class ClusterColorSpec` -> Color configuration for integer cluster IDs 0..n-1 plus optional -1.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 34: `function _golden_ratio_palette` -> Generate n visually distinct colors with deterministic hue spacing.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 44: `function _discrete_colors_from_matplotlib_cmap` -> (no docstring; infer from name/context). `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 57: `function build_cluster_color_spec` -> Build a discrete colormap + normalizer for cluster labels.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 124: `function present_cluster_ids` -> Sorted unique cluster IDs from label sequence (excludes -1).. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`

### `kl_clustering_analysis/plot/cluster_tree_visualization.py`
- File history: 7 commits, first `2025-10-29 fc3a4b0c` (Kl-Te-Cluster), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 28: `function _normalize_optional_bool` -> Convert heterogeneous truthy/falsy values to bool or None.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 53: `function _lookup_results_value` -> Lookup a per-node value in ``results_df`` with robust id matching.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 72: `function _group_internal_nodes_for_halo` -> Split internal nodes into (significant, tested-not-significant).. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 99: `function _group_edges_for_sibling_style` -> Group edges by sibling-test status of the parent node.. `def-history: 2026-02-15 bf7c1e7d feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs`
- Line 130: `function _map_nodes_to_clusters` -> Map tree nodes to their cluster IDs.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 153: `function _sorted_children` -> (no docstring; infer from name/context). `def-history: 2025-12-25 48ec633a benchmark/plots: stream PDFs, timestamp outputs, and clean up MI + tree primitives`
- Line 161: `function _rectangular_tree_layout` -> Deterministic rectangular layout for a rooted directed tree (or forest).. `def-history: 2025-12-25 48ec633a benchmark/plots: stream PDFs, timestamp outputs, and clean up MI + tree primitives`
- Line 242: `function _graphviz_twopi_layout` -> Compute a Graphviz ``twopi`` layout with graceful fallbacks.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`
- Line 258: `function plot_tree_with_clusters` -> Plot hierarchical tree with cluster assignments.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`

### `kl_clustering_analysis/tree/distributions.py`
- File history: 1 commits, first `2026-02-07 ea21bff4` (Branch Length Test), last `2026-02-07 ea21bff4` (Branch Length Test).
- Line 16: `function _calculate_leaf_distribution` -> Set distribution and leaf count for a leaf node.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 30: `function _calculate_hierarchy_node_distribution` -> Weighted mean of children distributions.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`
- Line 66: `function populate_distributions` -> Populate 'distribution' and 'leaf_count' for all nodes bottom-up.. `def-history: 2026-02-07 ea21bff4 Branch Length Test`

### `kl_clustering_analysis/tree/poset_tree.py`
- File history: 14 commits, first `2025-10-29 fc3a4b0c` (Kl-Te-Cluster), last `2026-02-15 bf7c1e7d` (feat: add tree-guided cousin calibration, null-pipeline FPR benchmark, module READMEs).
- Line 31: `class PosetTree` -> Directed tree wrapper that exposes hierarchy operations.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 52: `method PosetTree.__init__` -> Initialize PosetTree with stats_df property.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 59: `method PosetTree.from_agglomerative` -> Construct a tree from an :class:`sklearn.cluster.AgglomerativeClustering` fit.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 142: `method PosetTree.from_undirected_edges` -> Orient an undirected tree and promote it to :class:`PosetTree`.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 183: `method PosetTree.from_linkage` -> Builds a tree from a SciPy linkage matrix.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 233: `method PosetTree.root` -> Return the cached root node, discovering it if necessary.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 244: `method PosetTree.get_leaves` -> Collect leaf nodes globally or within a subtree.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 279: `method PosetTree._is_leaf` -> Check if a node is a leaf.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 286: `method PosetTree.compute_descendant_sets` -> Map each node to the set of leaf labels under it.. `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 312: `method PosetTree._get_depths` -> Computes and caches node depths from the root.. `def-history: 2025-12-24 e3d229d1 Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge`
- Line 318: `method PosetTree.find_lca` -> Find the lowest common ancestor (LCA) of two nodes.. `def-history: 2025-12-24 e3d229d1 Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge`
- Line 372: `method PosetTree.find_lca_for_set` -> Find the lowest common ancestor for a collection of nodes.. `def-history: 2025-12-24 e3d229d1 Tree decomposition: centralize post-hoc merge application, make merged cluster IDs deterministic, and fix LCA handling in post-hoc merge`
- Line 403: `method PosetTree.populate_node_divergences` -> Populate tree nodes with distributions and KL divergences.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 434: `method PosetTree.decompose` -> Run ``TreeDecomposition`` directly from the tree.. `def-history: 2025-11-30 97c6d9ba Removed and clean up code environment`
- Line 488: `method PosetTree.build_sample_cluster_assignments` -> Build a per-sample cluster assignment table from decomposition output.. `def-history: 2025-12-17 8111d55f Updated and Restructured Projects.`

### `kl_clustering_analysis/tree/util/tree_construct_helpers.py`
- File history: 3 commits, first `2025-10-29 fc3a4b0c` (Kl-Te-Cluster), last `2026-02-07 ea21bff4` (Branch Length Test).
- Line 6: `function count_cluster_leaves` -> (no docstring; infer from name/context). `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 14: `function add_cluster_node_recursive` -> (no docstring; infer from name/context). `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`
- Line 32: `function add_nested_tuple_recursive` -> (no docstring; infer from name/context). `def-history: 2025-10-29 fc3a4b0c Kl-Te-Cluster`

## Debug Scripts Index

- Total scripts: 84
- analyze_*: 10
- benchmark_*: 7
- debug_*: 22
- derive_*: 2
- diagnose_*: 4
- explore_*: 5
- other: 16
- quick_*: 2
- smoke_*: 1
- test_*: 12
- verify_*: 3

- `_check_ari_format.py` [other] -> Quick check of case data format and ARI computation.
- `_smoke_test_cousin.py` [other] -> Smoke test for cousin F-test integration.
- `analyze_branch_by_linkage.py` [analyze_*] -> Analyze branch length correlations with statistical tests
- `analyze_branch_divergence_math.py` [analyze_*] -> Analyze the mathematical relationship between branch length and divergence.
- `analyze_case_17.py` [analyze_*] -> Analyze Case 17: binary_balanced_low_noise with 72x160, 4 clusters, entropy=0.25.
- `analyze_case_24.py` [analyze_*] -> Analyze Case 24: sparse_features_moderate
- `analyze_case_55_signal.py` [analyze_*] -> Analyze actual signal strength in Case 55 data.
- `analyze_distribution_diffs.py` [analyze_*] -> Analyze distribution differences between parent and child nodes.
- `analyze_noise_detection.py` [analyze_*] -> Analyze approaches to detect clusters in noisy binary data.
- `analyze_projection_sensitivity.py` [analyze_*] -> Analysis: Improving projection test sensitivity with scipy.
- `analyze_projection_signal.py` [analyze_*] -> Analyze how projection affects signal for realistic cluster differences.
- `analyze_sbm_issue.py` [analyze_*] -> Debug script to understand why SBM test cases get ARI = 0.
- `benchmark_advanced_methods.py` [benchmark_*] -> Benchmark advanced tree construction methods.
- `benchmark_branch_length_predictive.py` [benchmark_*] -> Branch Length Predictive Power Analysis
- `benchmark_branch_length_weighting.py` [benchmark_*] -> Benchmark: Compare clustering with and without branch length weighting.
- `benchmark_cousin_vs_wald.py` [benchmark_*] -> A/B benchmark: Wald (old) vs Cousin F-test (new) on the full benchmark suite.
- `benchmark_current_clustering.py` [benchmark_*] -> Benchmark the current clustering pipeline to verify it finds clusters correctly.
- `benchmark_three_sibling_methods.py` [benchmark_*] -> Full benchmark comparing all three sibling test methods.
- `benchmark_tree_construction.py` [benchmark_*] -> Benchmark different tree construction methods for robustness to noise.
- `check_tree_quality.py` [other] -> Check if the tree structure itself is wrong for noisy data.
- `compare_fix_a_vs_b.py` [other] -> (no module docstring)
- `compare_integration_models.py` [other] -> Mathematical Models for Branch Length Integration
- `compare_projection_methods.py` [other] -> Compare sparse vs orthonormal projection methods on clustering accuracy.
- `compare_tree_metrics.py` [other] -> Compare hamming vs rogerstanimoto distance metrics for tree inference.
- `compare_v1_v2_algorithms.py` [other] -> Compare decompose_tree (v1) vs decompose_tree_v2 (signal localization).
- `debug_asymmetric_a_vs_children_b.py` [debug_*] -> Debug: Compare A with children of B (asymmetric sibling comparison).
- `debug_asymmetric_clean.py` [debug_*] -> Clean test: Asymmetric comparison with proper probability data.
- `debug_asymmetric_siblings.py` [debug_*] -> Debug script to analyze asymmetric sibling scenarios.
- `debug_bh_correction.py` [debug_*] -> Diagnostic: Verify BH correction behavior under null hypothesis.
- `debug_bl_ablation.py` [debug_*] -> Diagnostic: Isolate branch-length vs nested-variance vs projection effects on edge inflation.
- `debug_branch_length_integration.py` [debug_*] -> Debug script to analyze branch length integration options.
- `debug_branch_length_math.py` [debug_*] -> Debug script to investigate mathematical inconsistency in branch length usage.
- `debug_branch_length_scale.py` [debug_*] -> Debug script to analyze branch length scale and variance components.
- `debug_branch_length_sibling_test.py` [debug_*] -> Debug script to verify branch-length adjustment in sibling divergence test.
- `debug_clustering_pipeline.py` [debug_*] -> Debug script for clustering pipeline failures.
- `debug_leaf_weighted_bl.py` [debug_*] -> Diagnostic: Test whether leaf-count-weighted branch lengths improve sibling calibration.
- `debug_min_sample_size_gate.py` [debug_*] -> Debug script: Demonstrate impact of min sample size gate (n >= 10) on Case 18.
- `debug_power_issue_summary.py` [debug_*] -> Summary: The real issue is STATISTICAL POWER, not asymmetric comparison.
- `debug_proposed_bl_implementation.py` [debug_*] -> Debug script showing proposed branch length integration implementation.
- `debug_sbm_statistical_tests.py` [debug_*] -> Debug file: Understanding why SBM test cases fail with our statistical tests.
- `debug_sbm_tree.py` [debug_*] -> Debug why SBM hierarchical clustering finds only 1 cluster.
- `debug_sibling_iteration_logic.py` [debug_*] -> Debug script to analyze sibling testing, iteration, and cluster merging logic.
- `debug_sibling_test_sensitivity.py` [debug_*] -> Debug: Why is the sibling test not finding differences?
- `debug_sibling_test_trace.py` [debug_*] -> Debug: Why is sibling test not detecting ANY differences?
- `debug_uncle_informed.py` [debug_*] -> (no module docstring)
- `debug_unconditional_sibling.py` [debug_*] -> Diagnostic: Sibling test calibration when run unconditionally at ALL binary parents.
- `debug_variance_comparison.py` [debug_*] -> Debug script to compare variance formulas and their effect on clustering.
- `demonstrate_branch_length_usage.py` [other] -> How to Use Branch Lengths with Our KL-Divergence Method
- `demonstrate_variance_issues.py` [other] -> Demonstrates the mathematical issues with variance calculations.
- `derive_branch_calibrated_test.py` [derive_*] -> Option B: Branch-Length Calibrated Divergence Test
- `derive_branch_calibrated_test_v2.py` [derive_*] -> Option B: Refined Branch-Length Calibrated Test
- `diagnose_k1_compact.py` [diagnose_*] -> Compact diagnostic: show regression coefficients and root ĉ for K=1 cases.
- `diagnose_k1_overdeflation.py` [diagnose_*] -> Diagnose the K=1 over-deflation problem in cousin_adjusted_wald.
- `diagnose_pipeline.py` [diagnose_*] -> Diagnose the full pipeline: edge test → sibling test → _should_split gates.
- `diagnose_sibling_test_liberality.py` [diagnose_*] -> Diagnose why sibling test is too liberal (fails to detect differences).
- `explore_branch_length_integration.py` [explore_*] -> Exploring How to Leverage Branch Length Information
- `explore_branch_length_relationship.py` [explore_*] -> Explore the relationship between branch lengths and distribution weighting.
- `explore_branch_vs_tests.py` [explore_*] -> Explore relationship between branch lengths and statistical test results.
- `explore_noisy_behavior.py` [explore_*] -> Explore why noisy data has so many significant edges.
- `explore_tree_heights.py` [explore_*] -> Explore tree height values in benchmark data.
- `extract_case_18_for_analysis.py` [other] -> Extract case 18 data for external power analysis.
- `extract_case_18_for_analysis_v2.py` [other] -> Extract case 18 data for external power analysis.
- `investigate_branch_length_usage.py` [other] -> Investigate How to Use Branch Lengths in Our Method
- `investigate_case_19.py` [other] -> Investigate why test case 19 (d=1200, n=72, k=4) is under-splitting.
- `investigate_edge_weights.py` [other] -> Explore Edge Weights and MRCA Distances
- `quick_benchmark.py` [quick_*] -> Quick benchmark to test current clustering configuration.
- `quick_projection_benchmark.py` [quick_*] -> Quick benchmark comparing sparse vs orthonormal projection methods.
- `smoke_test_adjusted_wald.py` [smoke_*] -> Smoke test: cousin-adjusted Wald vs original Wald on a few benchmark cases.
- `test_branch_length_distributions.py` [test_*] -> Debug script to test branch length integration in distribution calculation.
- `test_branch_lengths_only.py` [test_*] -> Test ONLY branch length metrics (actual distances, not heights).
- `test_branch_lengths_per_test.py` [test_*] -> Test branch lengths for EACH of the two statistical tests:
- `test_inconsistency_coefficient.py` [test_*] -> Test Inconsistency Coefficient as Predictive Feature
- `test_neighbor_joining.py` [test_*] -> Test Neighbor-Joining tree construction vs UPGMA (average linkage).
- `test_pca_ward.py` [test_*] -> Test the best tree construction method: PCA + Ward.
- `test_real_tree_validation.py` [test_*] -> Test real tree structures for validity.
- `test_sbm_modularity.py` [test_*] -> Test modularity matrix transformation on all SBM test cases.
- `test_sbm_pipeline.py` [test_*] -> Quick benchmark to test SBM cases with the updated pipeline.
- `test_sbm_transformations.py` [test_*] -> Test different data transformations for SBM graphs to make them work with hierarchical clustering.
- `test_spectral_tree.py` [test_*] -> Test Spectral Embedding + Hierarchical Clustering.
- `test_variance_formula_errors.py` [test_*] -> Debug script to test the error-raising behavior for invalid tree structures.
- `trace_harmonic_effect.py` [other] -> Trace why harmonic weighting decreases boundary detection AUC.
- `verify_felsenstein_logic.py` [verify_*] -> Verify Felsenstein's PIC logic and our implementation.
- `verify_kl_z_equivalence.py` [verify_*] -> (no module docstring)
- `verify_projection_math.py` [verify_*] -> (no module docstring)
