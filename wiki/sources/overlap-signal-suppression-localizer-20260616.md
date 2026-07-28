---
title: Overlap Signal Suppression Localizer 2026-06-16
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_signal_suppression_localizer.py
  - tests/validation/calibration/overlap/143_test_overlap_signal_suppression_localizer.py
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_signal_suppression_localization
  - raw/assets/benchmark-results/specific_small_method_benchmark_20260615/overlap_signal_suppression_localization_binary_suite
tags:
  - source
  - diagnostics
  - clustering
  - traversal
  - overlap
---

# Overlap Signal Suppression Localizer 2026-06-16

## Summary

`overlap_signal_suppression_localizer.py` joins paired signal clustering
comparisons with selected-neighborhood candidate contrast rows. It localizes
where the refined guarded traversal profile suppresses conditional-profile
movement and classifies the run-level effect as useful, harmful/overfragmented,
neutral, or not a suppressed signal candidate.

## Key Points

- The expanded overlap localization covers `35` signal pairs and `169` signal
  candidate rows. It finds `3` useful suppressed-movement runs, `5`
  harmful/overfragmented runs, `1` suppressed-candidate run without a final
  cluster-count change, and `26` runs with no suppressed signal candidate.
- The expanded useful rows are localized to `overlap_heavy_4c_small_feat`
  replicates `2` and `1`, plus `overlap_extreme_4c` replicate `4`. Their
  left-minus-right run-row ARI deltas are small but positive:
  `0.001587`, `0.001176`, and `0.000212`.
- The broader binary-suite overlap localization covers `42` signal pairs and
  `902` signal candidate rows. It finds `4` useful suppressed-movement runs,
  `2` harmful/overfragmented runs, `1` suppressed-candidate run without a final
  cluster-count change, and `35` runs with no suppressed signal candidate.
- The binary-suite useful rows are `overlap_heavy_4c_med_feat` replicate `1`
  with ARI delta `0.071754`, `overlap_heavy_8c_large_feat` replicate `0` with
  ARI delta `0.047422`, and the two repeated
  `overlap_heavy_4c_small_feat` replicates `2` and `1`.
- The useful rows are explained by a small number of suppressed candidate
  nodes. Across the binary-suite useful rows, there are `11` suppressed
  candidate nodes: `6` `left_split|right_guard_blocked` nodes and `5`
  `left_pass_through` nodes. All `11` are traversal-only pairs, so old/current
  topology-neighborhood evidence still does not cover these positives.
- Harmful extra movement is concentrated in `overlap_unbal_4c_small` and
  `overlap_part_4c_small`. In the expanded run, these harmful rows account for
  `21` suppressed candidates, mostly pass-through suppressions. This supports
  keeping the refined guard fail-closed outside the localized heavy-overlap
  signal cases.
- The result sharpens the merged method direction: do not globally relax the
  selected-family guard. Instead, target a heavy-overlap signal rescue law for
  the suppressed split/pass-through nodes and require topology/bandwidth
  support before promotion.

## Evidence

- `benchmarks/diagnostics/calibration/overlap/overlap_signal_suppression_localizer.py`
  implements the join and classification logic.
- `tests/validation/calibration/overlap/143_test_overlap_signal_suppression_localizer.py` validates
  useful, harmful, and no-suppression status classification and verifies output
  creation.
- The manifests record the paired clustering CSVs, candidate contrast CSVs,
  signal pair counts, signal candidate row counts, and output paths.

## Links

- [[overlap-method-clustering-comparison-20260616]]
- [[selected-neighborhood-bottleneck-law]]
- [[selected-neighborhood-distribution-panel-20260615]]
