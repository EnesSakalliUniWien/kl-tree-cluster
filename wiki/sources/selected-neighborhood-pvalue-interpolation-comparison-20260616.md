---
title: Selected Neighborhood Pvalue Interpolation Comparison 2026-06-16
type: source
status: reviewed
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/selected_neighborhood_pvalue_interpolation_comparison.py
  - tests/validation/145_test_selected_neighborhood_pvalue_interpolation_comparison.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_case_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_tau_s_sensitivity.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_tau_s_range.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_region_bandwidths.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_candidates/selected_neighborhood_pvalue_interpolation_region_tau_s_range.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_selected_pass_through_expanded_distribution_with_topology_bandwidths/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_joined_topology_bandwidths/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_joined_topology_bandwidths/selected_neighborhood_pvalue_interpolation_region_bandwidths.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_joined_topology_bandwidths/selected_neighborhood_pvalue_interpolation_region_tau_s_range.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_joined_topology_bandwidths_branch_length_metric/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_joined_topology_bandwidths_branch_length_metric/selected_neighborhood_pvalue_interpolation_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_joined_topology_bandwidths_branch_length_metric/selected_neighborhood_pvalue_interpolation_tau_s_range.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_pvalue_interpolation_comparison_overlap_expanded_joined_topology_bandwidths_branch_length_metric/selected_neighborhood_pvalue_interpolation_case_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_refined_candidate_audit_smoke/pvalue_interpolation/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_refined_candidate_audit_smoke/pvalue_interpolation/selected_neighborhood_pvalue_interpolation_rows.csv
tags:
  - source
  - diagnostics
  - pvalues
  - interpolation
  - bandwidth
---

# Selected Neighborhood Pvalue Interpolation Comparison 2026-06-16

## Summary

`selected_neighborhood_pvalue_interpolation_comparison.py` compares direct
selected-neighborhood sibling p-values against hold-out interpolated p-like
values on the same candidate rows. The diagnostic reconstructs stable-support
and signal-anchor neighborhoods from measured sibling p-values, excludes the
target row from its own interpolation, and reports whether interpolation lowers
or raises the direct p-value. The interpolated value is explicitly diagnostic,
not a calibrated production p-value.

## Key Points

- The expanded overlap candidate run produced `69,860` comparison rows from
  `139,860` input rows. Interpolation computed for every candidate row.
- With default `tau_s = 1`, no interpolated p-like value crossed `alpha =
  0.01`. The diagnostic therefore behaves conservatively rather than as a
  signal-rescue rule.
- Interpolation suppressed all direct significant selected-null rows:
  `1,432` selected-null direct positives became nonsignificant under the
  interpolated p-like value. There were no selected-null false-open rows under
  the default bandwidth.
- Interpolation also missed all direct significant signal rows: `1,722` signal
  direct positives became nonsignificant under the interpolated p-like value.
- The stable weighted mean p-like support is high: median interpolated values
  are about `0.36` to `0.38` by role and method, far above `0.01`.
- The regenerated full candidate run now includes effective interpolation
  support. Median \(n_{\mathrm{eff}}\) is `3.664014` for selected-null rows
  and `3.498292` for signal rows in both compared method profiles, showing
  that many nominal support anchors reduce to a small effective local support
  after tree-distance and scale weighting.
- The comparison rows now carry the old bandwidth coordinates needed for a
  region-level audit: `tau_b`, `tau_t`, `tau_s`, `h_k`, and
  `distance_to_stopping_edge`. The runner writes method-level `tau_s` ranges,
  role-specific topology-region bandwidth summaries, and region-level
  `tau_s` admissibility intervals.
- The added best-case bandwidth calculation shows that direct significant
  signal rows need median optimistic `tau_s` about `21.1` to cross `0.01`,
  while suppressed selected-null direct positives need median optimistic
  `tau_s` about `14.8`. Thus widening `tau_s` alone would reopen selected-null
  mistakes before reliably rescuing signal.
- The method-level `tau_s` interval check is unfavorable. On the expanded
  candidate run, all `24` method/fraction rows are
  `tau_s_range_empty_selected_null_reopens_first`; for the median recovery
  target, signal needs about `21.13`, while the selected-null leak upper bound
  is about `10.95` at a `10%` leak budget.
- The tau-sensitivity summary confirms the unfavorable tradeoff. At
  `tau_s = 20`, the best-case calculation recovers about `45%` of direct signal
  positives but reopens about `81%` of selected-null direct positives for each
  compared method profile.
- The original expanded distribution artifact was generated with
  `topology_rows = null` and `conditional_law_rows = null`, so all old
  topology-neighborhood bandwidth columns were empty. A regenerated joined
  distribution using `context_negative_topology_conditioning` and
  `conditional_law_weight1_min1_diagnostic` produced only `48/139,860`
  old-and-current topology-neighborhood rows.
- On the joined topology-bandwidth comparison, `116/140` role-regions still
  have only a partial bandwidth vector without finite `tau_b`; `24/140` have
  sparse finite `tau_b` support. No region has full-row `tau_b` coverage.
  Region-level `tau_s` intervals are mostly blocked: `602` rows report
  `tau_s_range_empty_selected_null_reopens_first`, `120` have no finite signal
  lower bound, `96` have no finite selected-null upper bound, and only `22`
  local interval rows are admissible.
- The tree-distance cache now has both topology-hop and branch-length runs.
  The branch-length artifact reports
  `cached_all_pairs_branch_length_tree_distances` for all `69,860` rows,
  compresses the median nearest-support distance from `1.0` hop to `0.03`
  branch-length units, and compresses the median best-case required `tau_s` from
  `150.826030` to `2.838978`.
- Branch lengths improve localization but do not validate interpolation as a
  rescue rule. At the method level, the `25%` signal-recovery versus `5%`
  selected-null-leak interval is still empty: branch length gives signal lower
  bound `0.048578` and selected-null upper bound `0.003706`, while hop distance
  gives `15.480732` versus `10.617175`. In both metrics the selected null
  reopens first.
- On `overlap_extreme_4c`, the branch-length metric catches signal
  interpolated positives (`426`), but it also creates more selected-null
  interpolated positives (`939`) than the hop metric (`599`). The diagnostic
  status remains `interpolation_false_open_risk` for selected null.
- The conditional-topology and refined global pass-through method profiles
  have identical p-value comparison counts in this row set, so the p-value
  interpolation diagnostic localizes a shared bandwidth/calibration issue
  rather than a method-profile difference.
- The implementation now reports effective interpolation support,
  \(n_{\mathrm{eff}}=(\sum_v w_v)^2/\sum_v w_v^2\), so downstream
  measurability tables can distinguish many weak anchors from a few dominant
  anchors. A real-row smoke on `overlap_unbal_4c_small` signal replicate `0`
  wrote `399` candidate rows with the new `effective_support` column.
- The full `69,860`-row candidate output was regenerated with the optimized
  vectorized interpolation path after the initial row-wise implementation was
  too slow for the full expanded panel.

## Evidence

- `benchmarks/diagnostics/calibration/selected_neighborhood_pvalue_interpolation_comparison.py`
  implements hold-out interpolation, row summaries, case summaries, and
  tau-sensitivity summaries.
- `tests/validation/145_test_selected_neighborhood_pvalue_interpolation_comparison.py`
  validates signal lowering, target hold-out, selected-null conservatism,
  selected-nonnull exclusion, bandwidth sensitivity, and output writing.
- The expanded overlap output directory records the manifest, row-level
  comparison CSV, summary CSV, case summary CSV, and tau-sensitivity CSV.
- The joined topology-bandwidth output directory records the sparse coverage
  of old bandwidth coordinates and the region-level interval table.
- The refined smoke output records the effective-support schema on a compact
  real selected-neighborhood subset.

## Links

- [[selected-neighborhood-measurability-law]]
- [[selected-neighborhood-bottleneck-law]]
- [[sibling-null-prior-interpolation-audit-20260604]]
- [[overlap-method-clustering-comparison-20260616]]
