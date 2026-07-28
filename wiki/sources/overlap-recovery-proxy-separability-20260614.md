---
title: Overlap Recovery Proxy Separability 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_recovery_proxy_separability.py
  - tests/validation/calibration/overlap/111_test_overlap_recovery_proxy_separability.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/recovery_proxy_separability/overlap_recovery_proxy_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/recovery_proxy_separability/overlap_recovery_proxy_metric_separability.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/recovery_proxy_separability/overlap_recovery_proxy_threshold_scan.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/recovery_proxy_separability/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - truth-geometry
---

# Overlap Recovery Proxy Separability 2026-06-14

## Summary

`overlap_recovery_proxy_separability.py` evaluates whether non-oracle
structural proxies can approximate the oracle distinction from
[[overlap-weak-truth-geometry-20260614]]: balanced or partial truth recovery
versus one-sided fragments, diffuse mismatches, and wrong-granularity splits.
The panel uses oracle labels only for evaluation; the scanned metrics are
observable structural quantities.

## Key Points

- The runner writes `overlap_recovery_proxy_rows.csv`,
  `overlap_recovery_proxy_metric_separability.csv`,
  `overlap_recovery_proxy_threshold_scan.csv`, and `manifest.json`.
- Scanned non-oracle metrics include child pairwise-Jaccard min/max/gap,
  homogeneity-gain min/max/gap, child-size balance, edge-norm balance,
  subspace consensus, a balanced-recovery proxy score, and a fragment-risk
  proxy score.
- Against one-sided fragment-like rows, the proxies are strong. The best metric
  is `fragment_risk_proxy_score` with `AUC = 0.955556`; a zero-fragment-leakage
  threshold retains `3/5` truth-recovery rows. Size balance, barycentric
  balance, edge-norm balance, and homogeneity gain each have `AUC = 0.911111`
  versus fragment-like rows and also retain `3/5` at zero fragment leakage.
- Against all non-recovery rows, the proxies are not sufficient. The best
  metric is `homogeneity_gain_min` with `AUC = 0.84`, but its zero-negative
  threshold retains only `1/5` truth-recovery rows.
- The composite balanced-recovery proxy improves fragment detection but still
  cannot solve the full selected-family mixture: `AUC = 0.866667` versus
  fragment-like rows but only `0.69` versus all non-recovery rows.
- Method implication: a non-oracle fragment-risk guard is plausible for
  avoiding one-sided pure-fragment traversal errors, but it is not a complete
  replacement for a selected-family null law or recovery law. Diffuse mismatch
  and wrong-granularity modes remain separate blockers.

## Evidence

- `tests/validation/calibration/overlap/111_test_overlap_recovery_proxy_separability.py` verifies
  proxy metric derivation, recovery versus fragment-like metric direction, and
  output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/111_test_overlap_recovery_proxy_separability.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_recovery_proxy_separability.py tests/validation/calibration/overlap/111_test_overlap_recovery_proxy_separability.py`.

## Links

- [[overlap-weak-truth-geometry-20260614]]
- [[overlap-weak-family-thresholds-20260614]]
- [[overlap-structural-decision-zones-20260614]]
- [[open-mathematical-questions]]
