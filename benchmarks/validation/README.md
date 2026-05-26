# Method-Constant Validation Manifest

This directory contains a small stdlib-only scaffold for tracking evidence for
method constants that are not yet validated in the manuscript.

The scaffold is deliberately strict:

- it enumerates every method constant currently requiring validation;
- it defines required output fields for future validation runs;
- it can record existing benchmark output paths as candidate sources;
- it does not read, summarize, or infer metrics from those paths;
- it marks evidence as `missing` until an explicit validation artifact supplies
  the required fields.

## Constants Covered

- `edge_alpha`
- `sibling_alpha`
- `mp_upper_edge_threshold`
- `min_spectral_dimension`
- `include_internal_spectral_rows`
- `sibling_projection_dimension_rule`
- `empirical_null_weight_rule`
- `context_bandwidth_rule`
- `pass_through_traversal`

## Usage

Create a manifest skeleton:

```bash
python -m benchmarks.validation.method_constants_manifest create \
  benchmarks/results/04_generic_benchmark_runs/run_20260325_134017Z \
  --output benchmarks/validation/manifests/method_constant_validation_manifest.json
```

Validate a manifest:

```bash
python -m benchmarks.validation.method_constants_manifest validate \
  benchmarks/validation/manifests/method_constant_validation_manifest.json
```

List constants:

```bash
python -m benchmarks.validation.method_constants_manifest constants
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
