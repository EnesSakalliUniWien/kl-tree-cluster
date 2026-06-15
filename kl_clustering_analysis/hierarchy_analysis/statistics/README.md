# statistics/

Statistical kernels and multiple-testing logic used by the decomposition gate
pipeline.

## Public Entrypoints

| Entrypoint | Purpose |
| ---------- | ------- |
| `annotate_child_parent_divergence` | Annotate child-parent edge evidence for tree traversal. |
| `annotate_sibling_divergence` | Annotate sibling split evidence with projected-Wald testing, inflation, and FDR. |
| `annotate_fixed_subspace_sibling_divergence` | Opt-in sibling gate using fixed covariance-whitened coordinate or feature-block BH aggregation without parent PCA adaptation. |
| `benjamini_hochberg_correction` | Flat BH correction helper. |
| `apply_tree_bh_correction` | Tree-aware BH correction for hierarchical edge testing. |

## Directory Map

| Path | Purpose |
| ---- | ------- |
| `contrast_covariance.py` | Canonical contrast-covariance object for projected-Wald statistics. |
| `projection/projected_wald/` | Shared projected-Wald kernel, projection basis, and reference distribution. |
| `projection/spectral/` | Node spectral tasks, Marchenko-Pastur thresholding, and tree-level spectral context. |
| `projection/projection_dimension_estimation/` | Projection dimension estimators. |
| `child_parent_divergence/` | Child-parent projected-Wald tests and tree-level edge annotation. |
| `sibling_divergence/` | Sibling pair collection, projected-Wald testing, fixed-subspace gates, empirical-null inflation, and sibling FDR. |
| `multiple_testing/` | Flat BH, tree-BH, and stopping-edge recovery helpers. |
| `branch_length_utils.py` | Optional branch-length scaling utilities. |

## Active Method Flow

1. Build node distributions on the `PosetTree`.
2. Build spectral/projection context for each node.
3. Run child-parent projected-Wald tests.
4. Run the configured sibling gate. The default path collects sibling pair
   records, estimates empirical-null inflation from valid calibration support,
   and applies sibling FDR. The opt-in fixed-subspace path skips parent PCA and
   edge-derived sibling dimensions, then applies either the full fixed
   chi-square statistic or coordinate-wise/feature-block BH p-value
   aggregation before traversal-aligned sibling FDR. A positive
   `sibling_gate_alpha_penalty` divides the sibling FDR alpha before
   annotation, matching the selected-topology penalty used in diagnostics. An
   optional root-stability guard can then fail closed on an unstable open root
   split without changing non-root sibling decisions. Fixed-subspace profiles
   can also attach a selected-permutation guard with explicit scope: `root`,
   broad `open_internal`, narrower `passthrough_descendant` contexts, or the
   conservative `global_sibling_min_passthrough_descendant` selected-family
   correction. The refined global scope reruns selected-family Monte Carlo
   floor pass-through cases at higher resolution before accepting the split.
   Pure Bernoulli and pure categorical fixed-coordinate BH p-values use the
   exact vectorized `compute_whitened_wald_contrast` path before the existing
   coordinate-wise BH aggregation.
5. Write sibling gate columns.
6. `TreeDecomposition` consumes the annotation bundle during top-down traversal.

## Contracts

- Feature-space covariance and contrast dimensions come from the active
  `FeatureSpace`; tests should not infer alternate schema names.
- Unsupported calibration support is an explicit method status. It is not
  replaced with a neutral inflation value.
- Projection quantities and raw feature coordinates are separate objects.
- Fixed-subspace sibling gates are opt-in method choices, not automatic
  production promotion.
- Selected-permutation guard scopes are diagnostic method choices. The
  `passthrough_descendant` scope targets descendant splits reached through an
  ordinary closed sibling ancestor and avoids compounding below explicitly
  guard-blocked roots. The
  `global_sibling_min_passthrough_descendant` scope evaluates those
  pass-through descendants against the minimum fixed-subspace sibling p-value
  over every binary parent in a fully reselected permutation null tree.
- DataFrame annotations are a transport format for gate columns; statistical
  kernels should use typed records where available.
