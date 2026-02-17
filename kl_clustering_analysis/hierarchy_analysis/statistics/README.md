# statistics/

Statistical tests, variance estimation, projections, and corrections.

## kl_tests/edge_significance.py — Gate 2

Child-parent divergence testing (projected Wald χ²).

| Function                                                              | What it does                                                                                                                                                                                                                        |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `annotate_child_parent_divergence(tree, df, alpha, fdr_method)`       | **Public API.** Tests every edge. Extracts distributions and branch lengths, computes projected Wald T = ‖R·z‖² ~ χ²(k), applies FDR correction (tree_bh / flat / level_wise). Writes `Child_Parent_Divergence_Significant` column. |
| `_compute_standardized_z(child, parent, n_c, n_p, bl, mean_bl)`       | Nested variance z-scores: `z = (θ_child − θ_parent) / √(Var)` with `Var = θ(1−θ)(1/n_c − 1/n_p)` and Felsenstein scaling `Var *= 1 + BL/mean_BL`.                                                                                   |
| `_compute_projected_test(child, parent, n_c, n_p, seed, bl, mean_bl)` | Project z to k dims via orthonormal R, compute T = ‖R·z‖², p = χ²_sf(T, k). Returns `(stat, df, pval, invalid)`.                                                                                                                    |
| `_compute_p_values_via_projection(tree, children, parents, ...)`      | Batch wrapper: loops all edges, extracts branch lengths, calls `_compute_projected_test`.                                                                                                                                           |

## sibling_divergence/ — Gate 3

Three interchangeable sibling test implementations. Selected by `config.SIBLING_TEST_METHOD`.

### sibling_divergence_test.py — `"wald"` (raw Wald)

| Function                                                              | What it does                                                                                                                                                |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sibling_divergence_test(left, right, n_l, n_r, bl_l, bl_r, mean_bl)` | **Core test.** Standardize proportions → random project → T = ‖R·z‖² ~ χ²(k). Handles both Bernoulli (1D) and categorical (2D). Returns `(stat, df, pval)`. |
| `annotate_sibling_divergence(tree, df, alpha)`                        | **Public API.** Collect eligible pairs → run tests → BH correct → write `Sibling_BH_Different` column.                                                      |
| `_get_binary_children(tree, parent)`                                  | Return `(left, right)` if parent has exactly 2 children, else `None`.                                                                                       |
| `_either_child_significant(left, right, sig_map)`                     | True if at least one child passed Gate 2.                                                                                                                   |
| `_get_sibling_data(tree, parent, left, right)`                        | Extract distributions, sample sizes, branch lengths for a sibling pair.                                                                                     |
| `_collect_test_arguments(tree, df)`                                   | Iterate tree, collect eligible sibling pairs (skip if neither child passed Gate 2).                                                                         |
| `_run_tests(...)`                                                     | Execute `sibling_divergence_test` for all collected pairs.                                                                                                  |
| `_apply_results(df, parents, results, alpha)`                         | BH-correct p-values, write result columns.                                                                                                                  |

### cousin_adjusted_wald.py — `"cousin_adjusted_wald"` (production default)

Deflates raw Wald T by estimated post-selection inflation factor ĉ.

| Class/Function                                          | What it does                                                                                                                                                                                        |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_SiblingRecord`                                        | Dataclass: `(parent, left, right, stat, df, pval, bl_sum, n_parent, is_null_like)`.                                                                                                                 |
| `CalibrationModel`                                      | Dataclass: fitted inflation model — method, coefficients, max_observed_ratio. (Public API.)                                                                                                         |
| `_fit_inflation_model(records)`                         | From null-like pairs (neither child passed Gate 2), compute `r_i = T_i/k_i`. ≥5 pairs → OLS regression `log(r) = β₀ + β₁·log(BL_sum) + β₂·log(n_parent)`. 3–4 → global median. <3 → no calibration. |
| `predict_inflation_factor(model, bl_sum, n_parent)`     | Predict ĉ for a focal pair. Clamped to `[1.0, max_observed_ratio]`. (Public API.)                                                                                                                   |
| `_collect_all_pairs(tree, df, mean_bl)`                 | Compute raw Wald stats for ALL binary-child parents. Label each null-like or focal.                                                                                                                 |
| `_deflate_and_test(records, model)`                     | For each focal pair: `T_adj = T / ĉ`, `p = χ²_sf(T_adj, k)`.                                                                                                                                        |
| `_apply_results_adjusted(df, ...)`                      | BH-correct deflated p-values, write columns + `Sibling_Test_Method`.                                                                                                                                |
| `annotate_sibling_divergence_adjusted(tree, df, alpha)` | **Public API.** Orchestrates: collect → fit → deflate → correct → audit.                                                                                                                            |

### cousin_calibrated_test.py — `"cousin_ftest"` (alternative)

F-test using the uncle's sibling stat as denominator (inflation cancels).

| Function                                                    | What it does                                                                                                                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cousin_ftest(tree, parent, left, right, mean_bl)`          | Find uncle (parent's sibling under grandparent). Compute `F = (T_LR/k_LR) / (T_UL_UR/k_UU) ~ F(k_LR, k_UU)`. Falls back to raw Wald when uncle unavailable. Returns `(stat, df, pval, used_ftest)`. |
| `_get_uncle(tree, parent)`                                  | Walk up to grandparent → return grandparent's other child.                                                                                                                                          |
| `_get_cousin_reference(tree, uncle, mean_bl)`               | Compute Wald T for uncle's two children (the denominator).                                                                                                                                          |
| `_compute_sibling_stat(tree, parent, left, right, mean_bl)` | Compute raw Wald T for a sibling pair (helper shared with F-test).                                                                                                                                  |
| `annotate_sibling_divergence_cousin(tree, df, alpha)`       | **Public API.** Collect → run cousin F-tests → BH correct → write columns.                                                                                                                          |

## pooled_variance.py

Variance estimation for the Wald test.

| Function                                                             | What it does                                                                                              |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `compute_pooled_proportion(θ₁, θ₂, n₁, n₂)`                          | Pooled estimate under H₀: `(n₁θ₁ + n₂θ₂) / (n₁+n₂)`.                                                      |
| `compute_pooled_variance(θ₁, θ₂, n₁, n₂)`                            | `Var = p(1−p)(1/n₁ + 1/n₂)` using pooled proportion.                                                      |
| `standardize_proportion_difference(θ₁, θ₂, n₁, n₂, bl_sum, mean_bl)` | `z = (θ₁ − θ₂) / √Var` with Felsenstein scaling `Var *= 1 + BL_sum/(2·mean_BL)`. Returns `(z, variance)`. |

## random_projection.py

JL-lemma random projection for dimension reduction.

| Function                                     | What it does                                                                                                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `compute_projection_dimension(n, d, eps)`    | Target dimension k from JL lemma: `k ≈ 8·ln(n)/ε²`, floored at `min_k`, capped at `d`.                                                |
| `generate_projection_matrix(d, k, seed)`     | Orthonormal R via QR decomposition of Gaussian matrix. Guarantees `R·Rᵀ = I_k` so `‖R·z‖² ~ χ²(k)` exactly. Cached by `(d, k, seed)`. |
| `derive_projection_seed(base_seed, test_id)` | Deterministic per-test seed via BLAKE2b hash of `"{base_seed}                                                                         | {test_id}"`. |

## branch_length_utils.py

Shared branch-length sanitization.

| Function                                 | What it does                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `sanitize_positive_branch_length(value)` | Return finite positive float, else `None`.                                            |
| `compute_mean_branch_length(tree)`       | Mean across edges that carry `branch_length` attribute. Returns `None` if none exist. |

## multiple_testing/

FDR correction implementations.

| Function                                                | What it does                                                      |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| `benjamini_hochberg_correction(pvals, alpha)`           | Standard BH procedure. Returns `(reject, pvals_adj, n_rejected)`. |
| `tree_bh_correction(pvals, tree, children, alpha)`      | Hierarchical BH per Bogomolov et al. (2021).                      |
| `flat_bh_correction(pvals, alpha)`                      | Flat BH (ignores tree structure).                                 |
| `level_wise_bh_correction(pvals, depths, alpha)`        | BH per tree depth level.                                          |
| `apply_multiple_testing_correction(pvals, method, ...)` | Dispatcher: routes to `tree_bh`, `flat`, or `level_wise`.         |

## Other modules

| Module                       | What it does                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `clt_validity.py`            | Berry-Esseen bounds, CLT validity checks, minimum sample size computation.    |
| `power_analysis.py`          | Power calculations for Wald tests (Cohen's h, two-sample, nested).            |
| `categorical_mahalanobis.py` | Mahalanobis-whitened vectors for categorical distributions (drop-last basis). |
