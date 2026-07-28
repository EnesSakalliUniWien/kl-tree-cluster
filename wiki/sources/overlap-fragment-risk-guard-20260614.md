---
title: Overlap Fragment Risk Guard 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_fragment_risk_guard.py
  - tests/validation/calibration/overlap/112_test_overlap_fragment_risk_guard.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/fragment_risk_guard/overlap_fragment_risk_guard_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/fragment_risk_guard/overlap_fragment_risk_guard_scan.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/fragment_risk_guard/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - guard
---

# Overlap Fragment Risk Guard 2026-06-14

## Summary

`overlap_fragment_risk_guard.py` scans non-oracle fragment-risk guard
thresholds over the whole `unstable_weak_homogeneity_zone`, including
selected-null rows. Oracle truth-geometry labels are used only for evaluation.
This is diagnostic-only, but it is the first concrete threshold family that
looks like a plausible runtime guard for one-sided fragment failures.

## Key Points

- The runner writes `overlap_fragment_risk_guard_rows.csv`,
  `overlap_fragment_risk_guard_scan.csv`, and `manifest.json`.
- The guard rows contain `23` selected-null rows, `5` truth-recovery rows,
  `9` fragment-like rows, and `11` diffuse-or-wrong rows.
- The best candidate in the focused run is
  `fragment_risk_proxy_score >= 1.252729`. It blocks `8/9`
  fragment-like rows, retains `5/5` truth-recovery rows, blocks `6/23`
  selected-null rows, and blocks `1/11` diffuse-or-wrong rows.
- Simpler symmetry guards are close: `size_balance <= 0.331667`,
  `barycentric_balance <= 0.331667`, or `edge_norm_balance <= 0.496259`
  each block `7/9` fragment-like rows while retaining `5/5` truth-recovery
  rows.
- Child pairwise-Jaccard gap and homogeneity-gain gap thresholds around
  `0.045512` block `8/9` fragment-like rows, retain `5/5` truth-recovery rows,
  and block fewer selected-null rows (`2/23`).
- The guard is not a full selected-family law. It does not solve
  diffuse-or-wrong or balanced-wrong-granularity weak rows, and all reported
  thresholds are fitted on the focused overlap diagnostic panel.
- Method implication: a fragment-risk guard is a credible next runtime
  diagnostic for one-sided split failures, but production promotion still
  needs broader null/signal validation and a separate selected-family recovery
  rule for non-fragment weak failures.

## Evidence

- `tests/validation/calibration/overlap/112_test_overlap_fragment_risk_guard.py` verifies
  weak-zone guard row construction with selected-null and signal rows,
  candidate guard threshold scanning, and output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/112_test_overlap_fragment_risk_guard.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_fragment_risk_guard.py tests/validation/calibration/overlap/112_test_overlap_fragment_risk_guard.py`.

## Links

- [[overlap-recovery-proxy-separability-20260614]]
- [[overlap-weak-truth-geometry-20260614]]
- [[overlap-structural-decision-zones-20260614]]
- [[open-mathematical-questions]]
