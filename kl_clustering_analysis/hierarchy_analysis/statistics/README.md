# statistics/

Projected-Wald test implementations, projection helpers, and multiple-testing utilities used
by the decomposition pipeline.

## Current Layout

| Path | Purpose |
| ---- | ------- |
| `child_parent_divergence/child_parent_divergence.py` | Public Gate 2 annotation entrypoint. |
| `child_parent_divergence/child_parent_tree_testing.py` | Tree-wide Gate 2 execution over parent-child edges. |
| `child_parent_divergence/child_parent_projected_wald.py` | Child-parent z-score construction and the single-edge projected-Wald kernel. |
| `child_parent_divergence/child_parent_spectral_decomposition.py` | Spectral context for per-node projection dimensions and PCA-based projections. |
| `sibling_divergence/adjusted_wald_annotation.py` | Cousin-adjusted sibling Wald path with post-selection inflation correction. |
| `sibling_divergence/__init__.py` | Public Gate 3 entrypoint. |
| `sibling_divergence/pair_testing/` | Sibling-pair collection and the core Wald statistic kernel. |
| `sibling_divergence/inflation_correction/` | Inflation-model fitting and prediction for the adjusted sibling path. |
| `sibling_divergence/bh_annotation.py` | Shared DataFrame writing and BH result bookkeeping for Gate 3. |
| `projection/` | Projection bases, projected-Wald p-values, chi-square helpers, and spectral `k` estimators. |
| `multiple_testing/` | Flat BH, level-wise BH, TreeBH, and dispatch logic. |
| `branch_length_utils.py` | Shared branch-length sanitization and tree-wide summaries. |
| `power_analysis.py` | Power calculations for edge and sibling tests. |
| `categorical_mahalanobis.py` | Drop-last-basis whitening for categorical distributions. |

## Runtime Flow

1. Gate 2 annotates child-parent divergence across the tree.
2. Optional spectral context from Gate 2 can be forwarded into Gate 3 projection settings.
3. Gate 3 runs the production cousin-adjusted Wald sibling test.
4. Multiple-testing correction is applied through the dispatcher selected by the decomposition layer.

## Stable Entrypoints

- `annotate_child_parent_divergence(...)` is the public Gate 2 entrypoint.
- `annotate_sibling_divergence(...)` is the public Gate 3 entrypoint.
- `apply_multiple_testing_correction(...)` is the shared correction dispatcher.
- `run_projected_wald_kernel(...)` is the common projected-Wald kernel used underneath the test adapters.
