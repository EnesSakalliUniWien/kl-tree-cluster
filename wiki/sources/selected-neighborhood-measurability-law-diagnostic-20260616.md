---
title: Selected Neighborhood Measurability Law Diagnostic 2026-06-16
type: source
status: reviewed
updated: 2026-06-16
sources:
  - benchmarks/diagnostics/calibration/selected_neighborhood_measurability_law.py
  - tests/validation/144_test_selected_neighborhood_measurability_law.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_measurability_law_overlap_expanded/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_measurability_law_overlap_expanded/selected_neighborhood_measurability_law_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_measurability_law_overlap_expanded/selected_neighborhood_measurability_law_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_measurability_law_overlap_expanded_candidates/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_measurability_law_overlap_expanded_candidates/selected_neighborhood_measurability_law_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_measurability_law_overlap_expanded_candidates/selected_neighborhood_measurability_law_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_refined_candidate_audit_smoke/measurability_law/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_refined_candidate_audit_smoke/measurability_law/selected_neighborhood_measurability_law_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_refined_candidate_audit_smoke/measurability_law/selected_neighborhood_measurability_law_summary.csv
tags:
  - source
  - diagnostics
  - measurability
  - traversal
  - neighborhood
---

# Selected Neighborhood Measurability Law Diagnostic 2026-06-16

## Summary

`selected_neighborhood_measurability_law.py` implements the selected-
neighborhood measurability law as a diagnostic evaluator. It applies direct
selected-family sibling p-values first, permits diagnostic rescue only when
direct measurement is blocked and supported interpolated priors plus topology
coherence are present, and otherwise fails closed with an explicit bottleneck
label.

## Key Points

- The implementation includes an unclipped
  `compute_child_interpolated_null_prior` helper. Probability-domain violations
  raise or mark the row invalid instead of being made valid by clipping.
- The row evaluator produces `split`, `diagnostic_rescue`, or `fail_closed`
  actions and reports bottlenecks such as
  `direct_measurable_not_significant`, `interpolated_prior_unavailable`,
  `support_bottleneck`, `selection_bottleneck_selected_nonnull_only`,
  `bandwidth_bottleneck`, and topology coherence failures.
- The focused tests cover direct-test precedence, direct-test fail-closed
  behavior, supported interpolation rescue, weak interpolated priors, invalid
  probability-domain values, selected-nonnull-only support, topology failures,
  and output writing.
- Running the diagnostic on the expanded overlap selected-neighborhood rows
  produced `139,860` row annotations: `3,050` direct splits and `136,810`
  fail-closed rows. No `diagnostic_rescue` rows fired because existing rows do
  not yet contain interpolated child or pair priors.
- The regenerated candidate-only expanded overlap audit joins the hold-out
  interpolation comparison and produces `69,860` candidate rows with `83`
  columns: `3,050` direct splits and `66,810` fail-closed rows. All rows have
  interpolation support available under the diagnostic contract, but no
  diagnostic rescue fires. The `104` non-direct rows now have selected-tree
  structural topology audited: `41` non-root rows have a structural balance
  product below the coherence floor, and `63` root rows have outgoing balance
  but require a separate root-selected topology law because they have no
  incoming branch. Thus the blocker is topology coherence, not unavailable
  interpolated p-like support.
- The refined candidate-audit implementation now joins optional hold-out
  interpolation rows and optional spectral-flow edge rows. Its output exposes
  traversal state, effective interpolation support, support weight, bandwidth
  scales, stable/signal neighborhood distances, topology coherence variables,
  selected-tree structural fallback fields, interpolation behavior labels, and
  spectral bottleneck status in one candidate-level table.
- The full candidate audit also localizes spectral availability:
  `67,272/69,860` rows are `spectral_not_joined`, `1,931` are
  `spectral_floor_only`, `605` show
  `spectral_subspace_rotation_bottleneck`, and `52` have observed diagnostic
  spectral flow.
- A real-row smoke on `overlap_unbal_4c_small` signal replicate `0` wrote
  `399` candidate rows and `83` columns under
  `selected_neighborhood_refined_candidate_audit_smoke/measurability_law`.
  The rows contain `19` direct splits, `378` direct-measurable closed rows,
  one low structural-balance bottleneck, and one
  `root_selected_topology_requires_root_law` bottleneck. The optional spectral
  join did not overlap this subset, so all rows are explicitly labeled
  `spectral_not_joined`.

## Evidence

- `benchmarks/diagnostics/calibration/selected_neighborhood_measurability_law.py`
  implements the evaluator, summary, CLI, and output manifest.
- `tests/validation/144_test_selected_neighborhood_measurability_law.py`
  validates the mathematical and row-level contracts.
- The two main manifests record full-row and candidate-only expanded overlap
  runs; the candidate manifest now records the joined interpolation and
  spectral-flow inputs.
- The refined smoke manifest records the joined candidate-audit table with
  interpolation and spectral join paths.

## Links

- [[selected-neighborhood-measurability-law]]
- [[selected-neighborhood-bottleneck-law]]
- [[overlap-signal-suppression-localizer-20260616]]
