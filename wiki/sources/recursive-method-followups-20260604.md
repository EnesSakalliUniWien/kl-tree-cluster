---
title: Recursive Method Followups 2026-06-04
type: source
status: reviewed
updated: 2026-06-04
sources:
  - raw/inbox/recursive-method-program-20260604.txt
  - kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/types/inflation_model.py
  - kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/empirical_null_inflation_estimation.py
  - tests/statistics/35_test_empirical_null_inflation_estimation.py
  - benchmarks/diagnostics/calibration/sibling_null_weight_rule_validation.py
  - tests/validation/69_test_sibling_null_weight_rule_validation.py
  - raw/assets/benchmark-results/sibling_null_weight_rule_validation_20260604/manifest.json
  - raw/assets/benchmark-results/sibling_null_weight_rule_validation_20260604/sibling_null_weight_rule_summary.csv
  - benchmarks/diagnostics/spectral/sibling_projection_dimension_rule_grid.py
  - tests/validation/70_test_sibling_projection_dimension_rule_grid.py
  - raw/assets/benchmark-results/sibling_projection_dimension_rule_grid_20260604/manifest.json
  - raw/assets/benchmark-results/sibling_projection_dimension_rule_grid_20260604/sibling_projection_dimension_rule_grid.csv
  - raw/assets/benchmark-results/mp_kmin_q14_q15_smoke_20260604/mp_kmin_contract_smoke.csv
tags:
  - source
  - calibration
  - spectral
  - diagnostics
---

# Recursive Method Followups 2026-06-04

## Summary

This follow-up ingests the recursive method program and implements or records
the next actionable items after Q5. It covers Q9--Q11 internal support and
weight-rule diagnostics, Q14--Q15 MP/minimum-dimension smoke evidence, and Q17
sibling projection-dimension rule-grid diagnostics. The remaining larger items
are recorded as ideas and validation tracks rather than production changes.

## Key Points

- Q9/Q11 now have explicit support-threshold reporting in code. Focal
  calibration decisions report positive-weight counts, selected-nonnull counts,
  strict/stopped support, family/local effective sample size, max weight share,
  leave-one-record sensitivity, threshold values, and failure reasons.
- Q9/Q11 threshold enforcement is opt-in through the calibration decision and
  scalar prediction helpers. With enforcement enabled, sparse support returns
  `undefined_sparse_context`; this protects validation runs without silently
  changing every existing production caller before thresholds are manuscript
  validated.
- Q10 now has a diagnostic weight-rule grid for `current_product_bh_p`,
  `min_edge_bh_p`, `geometric_mean_bh_p`, `fisher_combined_null_p`, and
  `hard_null_indicator_0_05`.
- Running the Q10 grid on the selected-geometry records shows that this input
  lacks internal-support labels, so it cannot validate a replacement rule. It
  still exposes rule behavior: current product has effective sample size
  `72.63`, max weight share `0.0250`, and weighted selected-ratio mean `7.37`,
  while the unweighted selected-ratio mean is `159.87`.
- Q14/Q15 targeted MP/k-min smoke over `gauss_null_large`, `binary_2clusters`,
  and `cat_highcard_20cat_4c` shows that `leaf_only_floor0` errors on all three
  cases, `leaf_only_floor1` runs all three with median ARI `1.0`, and
  `leaf_only_floor2` plus finite-null floor-2 both have two strict
  calibration-support failures. This does not validate `k_min=1`; it confirms
  that `k_min=2` is still entangled with support skips.
- Q17 now has a projection-dimension rule-grid diagnostic on selected-geometry
  rows. The current edge-derived rule uses `k=1` for about `75.0%` of selected
  records and `k=2` for about `25.0%`. Raw MP parent signal count would set
  `k=0` for about `47.1%` and differs from the current rule on about `58.7%`.
  A raw-MP floor-1 rule differs on about `27.8%`; parent dimension and raw-MP
  floor-2 differ on about `33.2%`.
- Remaining items Q19--Q44 require larger simulation or new model work:
  high-cardinality categorical attribution, continuous shrinkage/low-rank
  covariance, traversal/FDR proof or simulation, pass-through replacement,
  phylogenetic null generators, manuscript result manifests, and full
  selected-region law.

## Evidence

- `raw/inbox/recursive-method-program-20260604.txt` is the ingested method
  program and execution order.
- `empirical_null_inflation_estimation.py` and `inflation_model.py` implement
  the Q9/Q11 support-threshold reporting and opt-in enforcement.
- `tests/statistics/35_test_empirical_null_inflation_estimation.py` verifies
  support metadata, sparse-context enforcement, and custom threshold policies.
- `sibling_null_weight_rule_validation.py` and
  `69_test_sibling_null_weight_rule_validation.py` implement and test the Q10
  diagnostic weight-rule grid.
- `sibling_projection_dimension_rule_grid.py` and
  `70_test_sibling_projection_dimension_rule_grid.py` implement and test the
  Q17 dimension-rule distribution grid.
- `sibling_null_weight_rule_summary.csv`,
  `sibling_projection_dimension_rule_grid.csv`, and
  `mp_kmin_contract_smoke.csv` record the current diagnostic outputs.

## Links

- [[open-mathematical-questions]]
- [[selected-tail-law-q5-validation-20260604]]
- [[local-marchenko-pastur-rule]]
- [[selected-hierarchy-null-support-contract]]
