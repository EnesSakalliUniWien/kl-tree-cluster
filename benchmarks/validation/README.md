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
- `include_internal_spectral_rows`
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

## Usage

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

Feature-covariance calibration reports follow the same evidence policy: a smoke
run proves the runner works, but it is not manuscript evidence unless the report
records a locked design, seed, commit, dirty-worktree status, command, grid,
confidence intervals, endpoints, and limitations.
