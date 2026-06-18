---
title: Selected Neighborhood Conditional Support Panel 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/selected_neighborhood_conditional_support_panel.py
  - tests/validation/187_test_selected_neighborhood_conditional_support_panel.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_conditional_support_overlap_expanded_candidates/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_conditional_support_overlap_expanded_candidates/selected_neighborhood_conditional_support_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_conditional_support_overlap_expanded_candidates/selected_neighborhood_conditional_support_case_summary.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_conditional_support_overlap_expanded_candidates/selected_neighborhood_conditional_support_summary.csv
tags:
  - source
  - diagnostics
  - neighborhood
  - support
  - root
---

# Selected Neighborhood Conditional Support Panel 2026-06-17

## Summary

`selected_neighborhood_conditional_support_panel.py` joins selected-
neighborhood measurability rows with selected-root validity and selected-root
tail evidence. It asks whether a candidate has admissible local neighborhood
support after the root context is known. Direct measurable sibling splits are
kept visible, but they are not counted as neighborhood support or neighborhood
leakage.

The first expanded-overlap run is conservative: it reports `0` conditional
neighborhood support passes, `0` selected-null neighborhood leaks, and `0`
hard-negative leaks. `overlap_extreme_4c` remains a hard negative.

## Key Points

- The panel evaluates `69,860` candidate rows across seven overlap cases, both
  `signal` and `selected_null` roles, and two method profiles.
- The run writes `28` case/method/role summaries and one overall summary under
  `selected_neighborhood_conditional_support_overlap_expanded_candidates`.
- The hard-negative control is supported: both `overlap_extreme_4c` signal
  method profiles are labeled `hard_negative_control_supported`, with
  `5,990` signal rows blocked by
  `root_validity_failed_hard_negative_control` and `0` hard-negative leaks.
- Direct measurable sibling splits are separated from neighborhood evidence:
  the run records `201` direct splits as
  `direct_split_already_measured_not_neighborhood_rescue`, but they do not
  increment conditional support, promotion, selected-null leak, or hard-negative
  leak counts.
- No candidate is promotable under the strict joined rule:
  `conditional_support_pass_count = 0` and
  `promotion_eligible_count = 0`.
- Local support is not the main missing piece in the expanded artifact. Most
  rows have enough interpolation support by count and effective support, but
  topology and root context block the decision before bandwidth can act.
- The remaining non-root non-direct bottlenecks are narrow: `19` rows are
  blocked by `effective_support_below_floor`, and `12` rows are blocked by
  `topology_balance_product_below_floor`. No non-root row passes the combined
  local-support, topology, and strict spectral-flow requirements.
- Root rows are deliberately not rescued by the neighborhood layer. They remain
  under the selected-root validity and selected-root tail laws.

## Evidence

- `benchmarks/diagnostics/calibration/selected_neighborhood_conditional_support_panel.py`
  implements the root-context join, row classifier, case summary, overall
  summary, and CLI.
- `tests/validation/187_test_selected_neighborhood_conditional_support_panel.py`
  covers hard-negative blocking, valid-root/missing-tail diagnostic support,
  selected-null leak detection, direct split separation, local support
  bottlenecks, root non-direct blocking, and output writing.
- The output manifest records the input measurability rows, root-validity rows,
  root-tail rows, support thresholds, strict spectral-flow requirement, and the
  hard-negative case list.
- The overall summary reports `row_count = 69860`,
  `hard_negative_leak_count = 0`, `selected_null_leak_count = 0`,
  `conditional_support_pass_count = 0`, and
  `summary_status = conditional_support_fail_closed_or_not_promotable`.

## Links

- [[selected-neighborhood-bottleneck-law]]
- [[selected-neighborhood-measurability-law]]
- [[selected-neighborhood-measurability-law-diagnostic-20260616]]
- [[selected-neighborhood-topology-frontier-diagnostic-20260616]]
- [[root-selected-validity-replay-panel-20260617]]
