# statistics/

Statistical kernels and multiple-testing logic used by the decomposition gate
pipeline.

## Public Entrypoints

| Entrypoint | Purpose |
| ---------- | ------- |
| `annotate_child_parent_divergence` | Annotate child-parent edge evidence for tree traversal. |
| `annotate_sibling_divergence` | Annotate sibling split evidence with projected-Wald testing, inflation, and FDR. |
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
| `sibling_divergence/` | Sibling pair collection, projected-Wald testing, empirical-null inflation, and sibling FDR. |
| `multiple_testing/` | Flat BH, tree-BH, and stopping-edge recovery helpers. |
| `branch_length_utils.py` | Optional branch-length scaling utilities. |

## Active Method Flow

1. Build node distributions on the `PosetTree`.
2. Build spectral/projection context for each node.
3. Run child-parent projected-Wald tests.
4. Collect sibling pair records where both child-parent edges support a split.
5. Estimate empirical-null inflation from valid calibration support.
6. Apply sibling FDR and write sibling gate columns.
7. `TreeDecomposition` consumes the annotation bundle during top-down traversal.

## Contracts

- Feature-space covariance and contrast dimensions come from the active
  `FeatureSpace`; tests should not infer alternate schema names.
- Unsupported calibration support is an explicit method status. It is not
  replaced with a neutral inflation value.
- Projection quantities and raw feature coordinates are separate objects.
- DataFrame annotations are a transport format for gate columns; statistical
  kernels should use typed records where available.
