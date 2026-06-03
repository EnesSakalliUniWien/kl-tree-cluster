---
title: Hierarchy Gate Separation 2026-06-03
type: source
status: reviewed
updated: 2026-06-03
sources:
  - benchmarks/results/run_20260601_173845Z_full/full_benchmark_comparison.csv
  - raw/assets/benchmark-results/hierarchy_gate_separation_20260603/manifest.json
  - raw/assets/benchmark-results/hierarchy_gate_separation_20260603/case_failure_separation.csv
  - raw/assets/benchmark-results/hierarchy_gate_separation_20260603/separation_summary.csv
  - raw/assets/benchmark-results/hierarchy_gate_separation_20260603/separation_by_category.csv
  - raw/assets/benchmark-results/hierarchy_gate_separation_20260603/oracle_ok_cases/oracle_tree_recoverability.csv
  - raw/assets/benchmark-results/hierarchy_gate_separation_20260603/gate_path_trace_ok_cases/gate_path_trace_summary.csv
tags:
  - source
  - benchmark
  - oracle
  - gates
  - hierarchy
---

# Hierarchy Gate Separation 2026-06-03

## Summary

This rerun separates current full-suite KL failures into hierarchy/metric,
gate, calibration-support, covariance-boundary, oracle-matched, and solved
buckets before any statistical change is considered. It uses the latest
strict KL-only full benchmark, runs oracle tree recoverability only for the 85
runnable KL rows, and keeps the 25 skipped rows as explicit contract outcomes
rather than imputing a partition.

## Key Points

- The latest strict KL-only full benchmark has `110` cases: `85` `ok` rows and
  `25` explicit skips.
- Among the `85` runnable rows, oracle recoverability classification gives:
  `62` solved, `14` tree/metric unrecoverable, `4` oracle-matched below solved,
  `4` gate over-splits, and `1` gate under-split.
- The `25` skips are separate from both tree and gate classifications:
  `24` are `calibration_support_undefined`, and `1` is a
  `continuous_covariance_boundary`.
- The current all-case separation is:

```text
solved                            62
calibration_support_undefined     24
tree_or_metric_unrecoverable      14
gate_over_split                    4
oracle_matched_below_solved        4
continuous_covariance_boundary     1
gate_under_split                   1
```

- The actionable gate set is only five runnable cases:
  `binary_low_noise_2c`, `cat_mod_4cat_6c`,
  `phylo_dna_8taxa_low_mut`, `phylo_protein_4taxa`, and
  `phylo_protein_8taxa`.
- The under-split case is `cat_mod_4cat_6c`: the trace stops above two oracle
  boundaries after open child-parent edge evidence because the sibling gate is
  closed.
- The over-split cases divide into direct sibling splits and pass-through
  fragmentation inside oracle boundaries. The phylogenetic protein/DNA rows
  are dominated by direct sibling false splits inside true clades; the binary
  low-noise row mixes direct splits and pass-through fragmentation.
- The `14` tree/metric unrecoverable runnable rows must not drive gate or
  alpha changes. They require hierarchy, metric, representation, or
  benchmark-construction analysis before changing statistics.
- The `4` oracle-matched-below-solved rows also should not drive statistical
  changes: KL already matches the best available exact-\(K\) tree cut within
  tolerance, but that tree cut is below the solved threshold.
- Calibration-support undefined skips are a separate selected-hierarchy null
  problem. They require an admissible selected calibration support contract or
  external selected-tail law, not local gate tuning.

## Evidence

- `benchmarks/results/run_20260601_173845Z_full/full_benchmark_comparison.csv`
  is the source benchmark with `85` `ok` rows and `25` skips.
- `raw/assets/benchmark-results/hierarchy_gate_separation_20260603/oracle_ok_cases/oracle_tree_recoverability.csv`
  records oracle recoverability for the runnable KL rows.
- `raw/assets/benchmark-results/hierarchy_gate_separation_20260603/gate_path_trace_ok_cases/gate_path_trace_summary.csv`
  records the five gate-failure traces.
- `raw/assets/benchmark-results/hierarchy_gate_separation_20260603/case_failure_separation.csv`
  records one separation bucket and required next action for each of the 110
  cases.
- `raw/assets/benchmark-results/hierarchy_gate_separation_20260603/manifest.json`
  records the separation rules and source paths.

## Links

- [[oracle-gate-path-diagnostic]]
- [[open-mathematical-questions]]
- [[benchmark-pipeline-contract]]
