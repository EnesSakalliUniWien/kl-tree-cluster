---
title: Selected Neighborhood Internal Spectral Flow Conditional Energy 2026-06-17
type: source
status: reviewed
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/selected_neighborhood_internal_spectral_flow_conditional_energy.py
  - tests/validation/189_test_selected_neighborhood_internal_spectral_flow_conditional_energy.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_internal_spectral_flow_overlap_seven_case/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_internal_spectral_flow_overlap_seven_case/selected_neighborhood_internal_spectral_flow_neighborhood_energy.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_internal_spectral_flow_conditional_energy_overlap_seven_case/manifest.json
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_internal_spectral_flow_conditional_energy_overlap_seven_case/selected_neighborhood_internal_spectral_flow_conditional_energy_rows.csv
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/selected_neighborhood_internal_spectral_flow_conditional_energy_overlap_seven_case/selected_neighborhood_internal_spectral_flow_conditional_energy_summary.csv
tags:
  - source
  - diagnostics
  - spectral
  - internal-barycenters
  - conditional-inference
  - root-validity
---

# Selected Neighborhood Internal Spectral Flow Conditional Energy 2026-06-17

## Summary

`selected_neighborhood_internal_spectral_flow_conditional_energy.py`
postprocesses internal-barycenter neighborhood energy under selected-root
context. It closes the gap between "internal smoothing was observed" and
"internal smoothing is valid conditional rescue evidence." A signal row becomes
a diagnostic rescue candidate only if the selected root is valid, the selected
root tail is supported, strict shared MP transport improves, and the paired
selected-null row does not show internal-only support.

The seven-case overlap run reports `0/7` signal rescue candidates. This means
the internal-barycenter channel reacts, but not in a conditionally valid way
for rescue.

## Key Points

- The larger internal spectral-flow run covers seven overlap cases:
  `overlap_extreme_4c`, `overlap_heavy_4c_small_feat`,
  `overlap_mod_4c_small`, `overlap_mod_6c_med`,
  `overlap_part_4c_small`, `overlap_unbal_4c_small`, and
  `overlap_unbal_6c_med`, each with `selected_null` and `signal` roles.
- The larger run writes `27,972` node rows, `27,944` edge rows, `13,986`
  node-pairwise rows, `13,972` edge-pairwise rows, and `14`
  neighborhood-energy rows.
- Internal MP support increases from `2,448` to `3,322` edges on selected null
  and from `1,893` to `3,151` edges on signal.
- Internal-only MP support remains mirrored: `874` selected-null edges and
  `1,260` signal edges.
- Strict shared energy improves in only two signal cases:
  `overlap_extreme_4c` and `overlap_heavy_4c_small_feat`. Both have failed
  root validity, and `overlap_extreme_4c` is the hard negative.
- The only row with both root validity and calibrated root-tail support is
  `overlap_unbal_6c_med` signal, but its strict shared energy degrades, so it
  fails closed.
- All seven selected-null rows have internal-only energy warnings. These are
  paired controls showing that internal-only support is not valid rescue
  evidence.
- The summary reports `rescue_candidate_count = 0`,
  `selected_null_warning_count = 7`,
  `paired_selected_null_warning_count = 7`, and
  `summary_status = conditional_internal_energy_rescue_fail_closed`.

## Evidence

- The conditional postprocess classifies signal rows by root validity, root-tail
  support, strict shared MP energy improvement, hard-negative status, and paired
  selected-null internal-only support.
- The tests cover paired selected-null blocking, a positive diagnostic candidate
  when controls are clean, root-validity blocking, strict-energy blocking, and
  output writing.
- The seven-case run confirms that internal-barycenter energy should stay a
  bottleneck/coherence diagnostic. It is not valid conditional rescue evidence
  on this benchmark surface.

## Links

- [[selected-neighborhood-internal-spectral-flow-panel-20260617]]
- [[selected-neighborhood-bottleneck-law]]
- [[root-tree-geometry-hard-negative-replay-20260617]]
- [[root-conditional-kernel-spectral-law]]
