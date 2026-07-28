---
title: Root Tie Rank Calibration Feasibility 2026-06-16
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/root/tie_rank/root_tie_rank_calibration_feasibility.py
  - tests/validation/calibration/root/tie_rank/156_test_root_tie_rank_calibration_feasibility.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/root_tie_rank_calibration_feasibility_overlap_case_family
tags:
  - source
  - diagnostics
  - root
  - calibration
  - selected-region
---

# Root Tie Rank Calibration Feasibility 2026-06-16

## Summary

`root_tie_rank_calibration_feasibility.py` converts the mixed root selected
region
\[
\mathcal E_{\mathrm{root}}
=
\mathcal E_{\mathrm{margin}}
\cap
\mathcal E_{\mathrm{tie}}
\cap
\mathcal E_{\mathrm{rank}}
\]
into explicit calibration strata. The strata use the mixed-region component,
selected tie-rank band, root edge-margin band, root spectral-ratio band, and
bandwidth-reopen status. The panel reports whether each stratum has enough
selected-null support to estimate a conditional root tail law.

## Key Points

- The seven-case overlap run writes `7` row records, `7` stratum records, and
  one summary row under `root_tie_rank_calibration_feasibility_overlap_case_family`.
- All seven input rows are unlabeled diagnostic rows, not external selected-null
  calibration support, so all seven strata have
  `external_null_support_missing`.
- With target alpha `0.01`, plus-one empirical p-value resolution requires at
  least `99` selected-null rows per stratum.
- With target alpha `0.01` and relative tail standard error target `0.25`, the
  tail-precision requirement is `1584` selected-null rows per stratum.
- Because the overlap artifact currently has seven strata and zero admissible
  null support, the missing support is `693` selected-null root simulations for
  alpha-resolution only and `11088` for the stated tail-precision target.
- The result makes the next calibration run precise: simulate selected-null
  roots into the same tie-rank, edge, spectral, and bandwidth strata before
  attempting any root rescue p-value.

## Evidence

- The implementation bins selected tie-rank fraction, log edge-margin,
  spectral ratio over MP, and bandwidth-reopen status into a conditioning key.
- The test covers the plus-one alpha-resolution count, the binomial tail
  precision count, selected-null-only support accounting, conservative
  empirical tail p-values when support exists, missing support statuses, and
  output writing.
- The generated overlap summary records `row_count = 7`, `stratum_count = 7`,
  `calibration_null_support_count = 0`,
  `missing_alpha_resolution_null_count_total = 693`, and
  `missing_tail_precision_null_count_total = 11088`.

## Links

- [[root-selected-mixed-region-law-20260616]]
- [[root-selected-tie-cell-burden-20260616]]
- [[selected-neighborhood-measurability-law]]
