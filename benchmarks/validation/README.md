# Validation Evidence Scaffolds

This directory contains strict scaffolds for tracking and generating validation
evidence that is not yet complete in the manuscript.

The method-constant manifest is deliberately strict:

- it enumerates every method constant currently requiring validation;
- it defines required output fields for future validation runs;
- it can record existing benchmark output paths as candidate sources;
- it does not read, summarize, or infer metrics from those paths;
- it marks evidence as `missing` until an explicit validation artifact supplies
  the required fields.

## Method Constants Covered

- `edge_alpha`
- `sibling_alpha`
- `mp_upper_edge_threshold`
- `min_spectral_dimension`
- `sibling_projection_dimension_rule`
- `empirical_null_weight_rule`
- `context_bandwidth_rule`
- `pass_through_traversal`

## Feature-Covariance Targets

Feature covariance is tracked separately from method constants because it is a
model-validation question, not a scalar tuning-constant question.

- `categorical_multinomial_drop_last_covariance`
- `continuous_empirical_gaussian_covariance`

The feature-covariance runner generates local sibling-null simulations in the
full tangent space. These results validate only the local covariance model under
a fixed contrast. They do not validate PCA-selected projections, MP dimension
selection, sibling FDR, traversal, tree construction, or empirical-null
inflation.

## Selected-PCA Projected-Wald Target

Selected PCA is tracked separately from feature covariance because it asks
whether the fixed-subspace chi-square reference remains calibrated after the
projection rows and MP dimension are selected from the same local data used by
the contrast.

- `selected_pca_projected_wald_reference`

The selected-PCA runner generates fixed-membership Gaussian sibling-null
simulations. It uses the production null-whitened tangent coordinates, MP
dimension selection, PCA projection recovery, and projected-Wald kernel. These
results validate only the local data-selected projection effect. They do not
validate hierarchy construction, tree-selected sibling pairs, sibling FDR,
traversal, empirical-null inflation, categorical blocks, or real-data model
misspecification.

## Usage

Run an alpha-grid benchmark diagnostic over the active KL path:

```bash
uv run python -m benchmarks.validation.alpha_grid_search \
  --suite full \
  --edge-alphas 0.0001,0.0003,0.001,0.003,0.01 \
  --sibling-alphas 0.001,0.003,0.01,0.03,0.1 \
  --output-dir benchmarks/results/alpha_grid_full
```

This diagnostic compares benchmark behavior across edge and sibling alpha
constants. It is not a proof of selected-tree Type-I error control.

Run a selected-edge Type-I geometry smoke over binary null cases:

```bash
uv run python -m benchmarks.validation.selected_edge_type1_geometry run \
  --suite binary \
  --case-names binary_2clusters \
  --modes fixed_tree,selected_tree \
  --edge-alphas 0.0001,0.001 \
  --sibling-alpha 0.01 \
  --replicates 5 \
  --base-seed 20260604 \
  --output-dir benchmarks/results/selected_edge_type1_geometry_smoke
```

This diagnostic separates fixed-tree edge calibration from same-data
selected-tree edge behavior. It is evidence for the selected-tree Type-I
problem and does not change production alpha defaults or add a calibration
fallback.

Rank descriptive selected-edge geometry variables:

```bash
uv run python -m benchmarks.diagnostics.analysis.selected_edge_geometry_analysis \
  --edge-rows benchmarks/results/selected_edge_type1_geometry_smoke/selected_edge_geometry_edges.csv \
  --output benchmarks/results/selected_edge_type1_geometry_smoke/selected_edge_geometry_models.csv
```

Run a traversal-aligned sibling-FDR smoke:

```bash
uv run python -m benchmarks.validation.traversal_sibling_fdr_null \
  --layers synthetic_valid_p \
  --case-names synthetic_balanced_binary_tree \
  --replicates 200 \
  --alpha 0.01 \
  --base-seed 20260604 \
  --output-dir benchmarks/results/traversal_sibling_fdr_smoke/synthetic
```

Run the binary fixed-tree, selected-tree, and inflated layers:

```bash
uv run python -m benchmarks.validation.traversal_sibling_fdr_null \
  --layers fixed_tree_wald,selected_tree_wald,selected_tree_inflated \
  --suite binary \
  --case-names binary_2clusters \
  --replicates 20 \
  --alpha 0.01 \
  --edge-alpha 0.001 \
  --base-seed 20260604 \
  --output-dir benchmarks/results/traversal_sibling_fdr_smoke/binary
```

This diagnostic separates algorithmic FDR behavior from fixed-tree
projected-Wald calibration, same-data selected-tree effects, and
empirical-inflation support failures. It does not add a production fallback or
change alpha defaults.

Create a manifest skeleton:

```bash
uv run python -m benchmarks.validation.method_constants_manifest create \
  benchmarks/results/04_generic_benchmark_runs/run_20260325_134017Z \
  --output benchmarks/validation/manifests/method_constant_validation_manifest.json
```

Validate a manifest:

```bash
uv run python -m benchmarks.validation.method_constants_manifest validate \
  benchmarks/validation/manifests/method_constant_validation_manifest.json
```

List constants:

```bash
uv run python -m benchmarks.validation.method_constants_manifest constants
```

Create a missing-evidence manifest for feature covariance:

```bash
uv run python -m benchmarks.validation.feature_covariance_calibration manifest \
  --output benchmarks/validation/manifests/feature_covariance_validation_manifest.json
```

Run a feature-covariance calibration simulation:

```bash
uv run python -m benchmarks.validation.feature_covariance_calibration run \
  --replicates 1000 \
  --seed 20260601 \
  --output benchmarks/results/validation/feature_covariance_calibration.json \
  --csv-output benchmarks/results/validation/feature_covariance_calibration.csv
```

Validate a feature-covariance manifest or report:

```bash
uv run python -m benchmarks.validation.feature_covariance_calibration validate \
  benchmarks/results/validation/feature_covariance_calibration.json
```

Create a missing-evidence manifest for selected-PCA projected-Wald calibration:

```bash
uv run python -m benchmarks.validation.selected_pca_projected_wald_calibration manifest \
  --output benchmarks/validation/manifests/selected_pca_projected_wald_validation_manifest.json
```

Run a selected-PCA projected-Wald calibration simulation:

```bash
uv run python -m benchmarks.validation.selected_pca_projected_wald_calibration run \
  --replicates 1000 \
  --seed 20260601 \
  --output benchmarks/results/validation/selected_pca_projected_wald_calibration.json \
  --csv-output benchmarks/results/validation/selected_pca_projected_wald_calibration.csv
```

Validate a selected-PCA manifest or report:

```bash
uv run python -m benchmarks.validation.selected_pca_projected_wald_calibration validate \
  benchmarks/results/validation/selected_pca_projected_wald_calibration.json
```

## Evidence Policy

Existing benchmark outputs are recorded under `source_paths` only. They are not
used as evidence by this scaffold, because historical files may not contain the
locked seeds, configuration manifest, ablation grid, confidence intervals, and
constant-specific endpoints required for manuscript validation.

For each constant, the generated manifest contains:

- `evidence_status: "missing"`;
- `evidence.metrics: {}`;
- `evidence.source_path: null`;
- `evidence.missing_required_fields` listing every required field.

Future validation jobs should either extend this manifest with complete evidence
records or generate a separate results artifact that can be validated against the
same required field contract.

Feature-covariance and selected-PCA calibration reports follow the same evidence
policy: a smoke run proves the runner works, but it is not manuscript evidence
unless the report records a locked design, seed, commit, dirty-worktree status,
command, grid, confidence intervals, endpoints, and limitations.
