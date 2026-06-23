---
title: Legacy c2ef9a69 Method Comparison Panel 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/legacy_c2ef9a69_method_comparison_panel.py
  - tests/validation/152_test_legacy_c2ef9a69_method_comparison_panel.py
  - benchmarks/shared/runners/legacy_commit_runner.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_method_comparison_panel/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_method_comparison_panel/legacy_c2ef9a69_method_comparison_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_method_comparison_panel/legacy_c2ef9a69_method_comparison_pairwise.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/legacy_c2ef9a69_method_comparison_panel/legacy_c2ef9a69_method_comparison_summary.csv
tags:
  - source
  - diagnostics
  - legacy
  - benchmark
---

# Legacy c2ef9a69 Method Comparison Panel 2026-06-16

## Summary

`legacy_c2ef9a69_method_comparison_panel.py` compares the current standard
`tbs` method against the full old package snapshot exposed as
`tbs_legacy_c2ef9a69`. The one-replicate panel covers six compact binary cases:
`binary_perfect_2c`, `binary_low_noise_2c`, `binary_null_small`,
`overlap_part_4c_small`, `overlap_mod_4c_small`, and
`overlap_heavy_4c_small_feat`, each in selected-null and signal roles.

The result is mixed. The full legacy method avoids some strict current-method
skips and fixes one current over-split, but it also creates a selected-null
false split and under-splits the hardest overlap signal case. This supports
using the old method as a diagnostic comparison, not as a direct replacement.

## Key Points

- The run wrote `24` method rows and `12` paired comparisons.
- In selected-null rows, five pairs completed in both variants. The legacy
  method had one ARI regression: `overlap_mod_4c_small` selected-null changed
  from one current cluster with ARI `1.0` to three legacy clusters with ARI
  `0.0`.
- In signal rows, four pairs completed in both variants. The legacy method
  improved `binary_low_noise_2c` from current `8` clusters and ARI `0.606061`
  to the correct two clusters and ARI `1.0`.
- The legacy method completed `binary_perfect_2c` signal and
  `overlap_heavy_4c_small_feat` signal where the current method skipped under
  strict sibling-inflation support. These rows are not counted as paired ARI
  deltas because the current row has no labels.
- The legacy method under-split `overlap_heavy_4c_small_feat` signal to one
  cluster with ARI `0.0`.
- Summary deltas: selected-null mean legacy-minus-current ARI is `-0.2` and
  signal mean legacy-minus-current ARI is `0.098485` over rows with paired
  ARI values. Selected-null legacy false splits increase from `0` to `1`;
  signal current false splits decrease from `1` to `0`, while legacy
  under-splits increase from `0` to `1`.

## Evidence

- `legacy_c2ef9a69_method_comparison_panel.py` runs both variants through
  standard benchmark dispatch and writes rows, pairwise, summary, and manifest
  outputs.
- `152_test_legacy_c2ef9a69_method_comparison_panel.py` verifies the pairwise
  delta and output-writing contract.
- The manifest records suite `binary`, one replicate, base seed `20260613`,
  edge alpha `0.001`, sibling alpha `0.01`, and the six case names above.

## Links

- [[legacy-c2ef9a69-method-package-20260616]]
- [[legacy-internal-spectral-comparison-panel-20260616]]
- [[old-vs-current-method-stack-comparison-20260615]]
