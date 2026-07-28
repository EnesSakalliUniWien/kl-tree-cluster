---
title: Overlap Structural Continuous Rule 2026-06-14
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_structural_continuous_rule.py
  - tests/validation/calibration/overlap/106_test_overlap_structural_continuous_rule.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/continuous_rules/overlap_structural_continuous_rule_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/continuous_rules/manifest.json
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/continuous_rules_fine/overlap_structural_continuous_rule_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/continuous_rules_fine/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - thresholds
---

# Overlap Structural Continuous Rule 2026-06-14

## Summary

`overlap_structural_continuous_rule.py` turns the binned overlap-context
diagnostic into a smooth traversal-context threshold surface. It is
diagnostic-only. The surface is meant to test whether depth, parent size, and
barycentric balance can replace one global homogeneity-gain cutoff with a
continuous structural sibling rule.

## Key Points

- The implemented threshold is
  \[
  \tau(u)=b
  +\lambda_d \exp(-d_u/s)
  +\lambda_n\frac{\log(1+n_u)}{\log(1+n_0)}
  +\lambda_\beta\max(0,\beta_0-\beta_u),
  \]
  where \(d_u\) is node depth, \(n_u\) is parent size, and \(\beta_u\) is
  barycentric balance.
- A row is a continuous structural accept only when the original traversal
  accepted the internal split, the sibling p-value is below the rule threshold,
  `homogeneity_gain_min >= tau`, and `subspace_consensus_jaccard_topk` is above
  the rule threshold.
- The initial grid evaluated `432` rules over the three-replicate overlap
  rows. It found no `continuous_rule_candidate`: `96` rules were
  `continuous_rule_null_uncontrolled`, and `336` were
  `continuous_rule_signal_loss`.
- The best initial rule blocked all null and truth-misaligned accepts, but
  retained only `16/17` truth-aligned signal accepts and `7/8` truth-aligned
  signal case-replicates. The lost case-replicate was a shallow, balanced
  `overlap_unbal_4c_small` split with homogeneity gain about `0.006985`.
- A finer grid evaluated `1200` rules with smaller shallow penalties and
  stronger balance penalties. It still found no candidate: `1104` rules were
  `continuous_rule_signal_loss`, `56` were
  `continuous_rule_null_uncontrolled`, and `40` kept truth-misaligned signal.
- Direct row inspection explains why a hard accept rule is not yet available.
  All null accepts and all truth-misaligned signal accepts are
  `weak_homogeneity_gain`; only `12/17` truth-aligned signal accepts are
  `structural_same_subspace_supported`. The remaining `5/17` truth-aligned
  signal accepts live in the weak structural region, numerically overlapping
  null or truth-misaligned rows.
- The method implication is a continuous traversal structural rule with an
  explicit ambiguous zone. Deep/internal structural same-subspace accepts can
  be treated as stronger diagnostic evidence, while shallow, root, large-parent,
  or weak-homogeneity rows should become guard-blocked or multi-scale unstable
  regions unless a selected-family null law supplies production support.

## Evidence

- `tests/validation/calibration/overlap/106_test_overlap_structural_continuous_rule.py` verifies
  that the threshold surface is stricter for root/large contexts than for
  deep/smaller contexts, that a synthetic deep signal can be separated from a
  synthetic root null under fixed parameters, and that the runner writes its
  outputs.
- Verification passed:
  `pytest tests/validation/calibration/overlap/106_test_overlap_structural_continuous_rule.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_structural_continuous_rule.py tests/validation/calibration/overlap/106_test_overlap_structural_continuous_rule.py`.

## Links

- [[overlap-structural-sibling-panel-20260614]]
- [[overlap-structural-threshold-sensitivity-20260614]]
- [[overlap-structural-context-thresholds-20260614]]
- [[open-mathematical-questions]]
