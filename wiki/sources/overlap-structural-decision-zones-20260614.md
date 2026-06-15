---
title: Overlap Structural Decision Zones 2026-06-14
type: source
status: reviewed
updated: 2026-06-14
sources:
  - benchmarks/diagnostics/calibration/overlap_structural_decision_zones.py
  - tests/validation/107_test_overlap_structural_decision_zones.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/decision_zones/overlap_structural_decision_zone_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/decision_zones/overlap_structural_decision_zone_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/decision_zones/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - multiscale
---

# Overlap Structural Decision Zones 2026-06-14

## Summary

`overlap_structural_decision_zones.py` converts overlap structural sibling rows
into diagnostic traversal zones. It uses the continuous context threshold as a
guarding surface but does not force weak-homogeneity rows into production
accepts or rejects. The output is a three-way interpretation layer:
stable structural accept, unstable weak-homogeneity zone, and nonaccepted or
blocked rows.

## Key Points

- The runner writes `overlap_structural_decision_zone_rows.csv`,
  `overlap_structural_decision_zone_summary.csv`, and `manifest.json`.
- The diagnostic surface used for the run was the conservative continuous
  context rule with base `0.005`, shallow penalty `0.015`, imbalance penalty
  `0.005`, subspace threshold `0.15`, and sibling p threshold `0.001`.
- The rule is now exposed as signed margins, not only as a boolean. For node
  depth \(d\), parent size \(n_u\), and barycentric balance \(b_u\), the
  context homogeneity threshold is
  \[
  \tau(u)=\tau_0+\lambda_d e^{-d/s_d}
  +\lambda_n\frac{\log(1+n_u)}{\log(1+n_{\mathrm{ref}})}
  +\lambda_b(0.5-b_u)_+.
  \]
  The row reports \(h_u-\tau(u)\), \(c_u-c_0\), and
  \(\log(\alpha/p_u)\). A stable structural accept requires all three signed
  margins to be nonnegative and the split status to be
  `structural_same_subspace_supported`.
- On the three-replicate overlap rows, the stable structural accept zone has
  `12` accepted rows, all truth-aligned signal and no selected-null or
  truth-misaligned signal rows.
- The unstable weak-homogeneity zone has `48` accepted rows: `23` selected-null
  rows, `20` truth-misaligned signal rows, and `5` truth-aligned signal rows.
  This is the empirical reason a binary accept threshold is unsafe.
- After regenerating the output with explicit margins, the median continuous
  context minimum margin is `0.017777` in `stable_structural_accept` and
  `-0.009088` in `unstable_weak_homogeneity_zone`. This supports the rule
  shape: weak-zone rows should be exposed as unstable/multi-scale unless a
  selected-family law later promotes them.
- The nonaccepted or leaf zone has `112` rows, including stopped rows and leaf
  fragments; it is not a structural accept set.
- The accepted-row crosstab is therefore exact in this diagnostic run:
  stable structural accept equals clean truth-aligned evidence, while every
  null and truth-misaligned accepted split is in the unstable weak zone.
- Method implication: the traversal output should expose stable internal
  regions and unstable weak-homogeneity pass-through zones separately. The
  weak zone can still contain real signal, but it cannot be promoted by the
  current statistics without a selected-family null law or stronger structural
  evidence.

## Evidence

- `tests/validation/107_test_overlap_structural_decision_zones.py` verifies
  stable, weak, context-blocked, and nonaccepted zone assignment; signed
  continuous margins; summary truth role counts; and output writing.
- Verification passed:
  `pytest tests/validation/107_test_overlap_structural_decision_zones.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_structural_decision_zones.py tests/validation/107_test_overlap_structural_decision_zones.py`.

## Links

- [[overlap-structural-sibling-panel-20260614]]
- [[overlap-structural-continuous-rule-20260614]]
- [[overlap-structural-context-thresholds-20260614]]
- [[open-mathematical-questions]]
