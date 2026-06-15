---
title: Overlap Weak Truth Geometry 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - benchmarks/diagnostics/calibration/overlap_weak_truth_geometry.py
  - tests/validation/110_test_overlap_weak_truth_geometry.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_truth_geometry/overlap_weak_truth_geometry_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_truth_geometry/overlap_weak_truth_geometry_families.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_truth_geometry/overlap_weak_truth_geometry_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/weak_truth_geometry/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - truth-geometry
---

# Overlap Weak Truth Geometry 2026-06-14

## Summary

`overlap_weak_truth_geometry.py` is an oracle-only diagnostic for weak accepted
signal splits in binary overlap cases. It uses truth labels and child truth
purities to explain why selected-family p-values and scalar structural
thresholds fail: many statistically strong weak splits are not balanced
recovery of the target structure, but one-sided fragments, diffuse mismatches,
or wrong-granularity splits.

## Key Points

- The runner writes `overlap_weak_truth_geometry_rows.csv`,
  `overlap_weak_truth_geometry_families.csv`,
  `overlap_weak_truth_geometry_summary.csv`, and `manifest.json`.
- The diagnostic classifies weak accepted signal rows into
  `balanced_truth_recovery`, `partial_truth_recovery`,
  `one_sided_pure_fragment`, `one_sided_mixed_remainder`,
  `balanced_but_wrong_granularity`, and `diffuse_truth_mismatch`.
- In the three-replicate overlap run, weak accepted signal rows split into:
  `3` balanced truth recoveries, `2` partial truth recoveries, `8`
  one-sided pure fragments, `1` one-sided mixed remainder, `3` balanced but
  wrong-granularity rows, and `8` diffuse truth mismatches.
- Family modes are similarly mixed: `5` weak signal families contain truth
  recovery, while `2` are diffuse mismatch, `2` are one-sided pure fragment,
  and `1` is balanced wrong granularity.
- One-sided pure fragments have high maximum child purity but low minimum child
  purity: median max child purity is `0.984311`, median min child purity is
  `0.331126`, and median child-purity gap is `0.653227`.
- Balanced truth recovery rows have both children substantially purer than the
  parent: median parent truth purity is `0.475962`, median min child purity is
  `0.778846`, and median max child purity is `0.878505`.
- This explains why selected-family p-values alone are misleading. A split can
  be highly selected because it isolates a pure child, while the sibling side is
  still a mixed remainder and the split is truth-misaligned.
- Method implication: the selected-family traversal null law needs a structural
  recovery target, not only an extremeness target. Until then, weak families
  should be surfaced as unstable multi-scale zones with oracle-diagnostic
  labels in benchmark reports, not as production accepts.

## Evidence

- `tests/validation/110_test_overlap_weak_truth_geometry.py` verifies truth
  geometry mode classification, signal-only weak row filtering, family
  aggregation, and output writing.
- Verification passed:
  `pytest tests/validation/110_test_overlap_weak_truth_geometry.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_weak_truth_geometry.py tests/validation/110_test_overlap_weak_truth_geometry.py`.

## Links

- [[overlap-weak-family-thresholds-20260614]]
- [[overlap-weak-zone-separability-20260614]]
- [[overlap-structural-decision-zones-20260614]]
- [[open-mathematical-questions]]
