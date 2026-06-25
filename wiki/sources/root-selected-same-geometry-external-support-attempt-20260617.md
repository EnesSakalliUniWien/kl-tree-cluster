---
title: Root Selected Same Geometry External Support Attempt 2026-06-17
type: source
status: draft
updated: 2026-06-17
sources:
  - benchmarks/diagnostics/calibration/root_selected_same_geometry_external_support_attempt.py
  - tests/validation/182_test_root_selected_same_geometry_external_support_attempt.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_selected_same_geometry_external_support_attempt_five_target_tiny_smoke
tags:
  - source
  - diagnostics
  - root
  - spectral
  - selected-tail
---

# Root Selected Same Geometry External Support Attempt 2026-06-17

## Summary

`root_selected_same_geometry_external_support_attempt.py` makes the missing
selected-root support loop executable. It generates likelihood-ratio
external-null roots, replays those generated matrices through the
selected-neighborhood topology stack, joins measured bandwidth/topology \(B\),
computes deformed-MP \(H_u\), and reruns the support-aware selected-root tail
panel.

The panel remains diagnostic-only. It does not rescue root splits; it only
reports whether same-\((C,T,A,E,B,H_u)\) support exists and whether that support
has nonzero \(S_{H_u}\).

## Key Points

- The five-target tiny smoke generated `5` target-conditioned external-null
  candidates and all `5` completed TBS replay.
- None of the `5` candidates hit the target pre-topology stratum
  \((C,T,A,E)\).
- After topology and \(H_u\) joining, `1/5` target rows had same-tail-stratum
  support, but that support had `S_Hu = 0`.
- No generated row supplied nonzero same-stratum \(S_{H_u}\), and no generated
  row exceeded an observed positive target \(S_{H_u}\).
- The selected-root tail panel reports `3/7` roots with conservative empirical
  p-values and `4/7` fail-closed, but the new support for
  `overlap_mod_4c_small` is underpowered zero-excess support with p-value
  `0.5`, not a solved positive spectral-tail law.
- The operational bottleneck is now sharper: the proposal/replay stack works,
  but the generator still misses selected tie-rank/action-edge root geometry
  before \(B,H_u\) can produce useful nonzero support.

## Evidence

- `root_selected_same_geometry_external_support_attempt_summary.csv` reports
  `target_count = 5`, `generated_candidate_count = 5`,
  `replay_completed_candidate_count = 5`,
  `pre_topology_supported_target_count = 0`,
  `new_nonzero_s_h_u_supported_target_count = 0`, and
  `summary_status = same_geometry_nonzero_s_h_u_support_missing_fail_closed`.
- `root_selected_same_geometry_external_support_attempt_rows.csv` records that
  `overlap_mod_4c_small` has `new_same_tail_stratum_support_count = 1` but
  `new_same_tail_positive_s_h_u_support_count = 0`.
- `target_conditioned_importance_target_rows.csv` records zero pre-topology
  hits for all five external-law targets.
- `root_selected_spectral_tail_law_summary.csv` reports
  `calibrated_tail_count = 3` and `fail_closed_missing_support_count = 4` after
  adding the generated rows.
- `182_test_root_selected_same_geometry_external_support_attempt.py` verifies
  target selection, exclusion of \(H_u\)-missing external rows from deformed
  tail support, and row-level counting of new same-geometry nonzero support.

## Links

- [[root-selected-external-law-equation-20260617]]
- [[root-selected-conditional-tilt-feasibility-20260617]]
- [[root-tie-rank-target-conditioned-importance-frontier-20260617]]
- [[root-selected-deformed-tail-support-gap-20260617]]
