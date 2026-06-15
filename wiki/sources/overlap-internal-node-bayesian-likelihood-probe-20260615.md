---
title: Overlap Internal-Node Bayesian Likelihood Probe 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/overlap_internal_node_bayesian_likelihood_probe.py
  - tests/validation/122_test_overlap_internal_node_bayesian_likelihood_probe.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_bayesian_likelihood_probe/overlap_internal_node_likelihood_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_bayesian_likelihood_probe/overlap_internal_node_likelihood_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_bayesian_likelihood_probe/overlap_internal_node_likelihood_families.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/internal_node_bayesian_likelihood_probe/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - bayesian
---

# Overlap Internal-Node Bayesian Likelihood Probe 2026-06-15

## Summary

`overlap_internal_node_bayesian_likelihood_probe.py` tests a row-level,
overlap-aware structural likelihood for residual weak traversal rows. It
separates coherent internal-node candidates from mixed selected-family
aggregates. It is diagnostic-only and does not promote production thresholds.

## Key Points

- The runner writes `overlap_internal_node_likelihood_rows.csv`,
  `overlap_internal_node_likelihood_summary.csv`,
  `overlap_internal_node_likelihood_families.csv`, and `manifest.json`.
- The row-level screen keeps selected-family p-value Bayes evidence separate
  from structural neighborhood evidence. It requires nonnegative context
  margin, soft subspace support at `0.15`, size balance at `0.33`, edge-norm
  balance at `0.49`, and fragment-risk proxy at most `1.25`.
- In the focused residual run, the probe selects `4/5` truth-recovery internal
  nodes while selecting `0/17` selected-null, `0/10` diffuse/wrong, and `0/1`
  fragment-like rows.
- The selected-null, diffuse/wrong, and fragment-like rows are all blocked by
  negative context margin. This confirms that context margin is the current
  useful selected-null and wrong-granularity blocker.
- The row-level probe recovers partial-truth rows in `overlap_mod_8c_large`
  that the family-level Bayesian aggregate rejected because each selected
  family also contained a diffuse/wrong neighboring row.
- Family-level reporting is still necessary: a residual selected family may
  contain one coherent internal node and one blocked neighbor. The right output
  is therefore multi-scale internal-node candidates inside an unstable family,
  not a flat family promotion.
- Method implication: the next conditional Bayesian traversal law should be
  row-level first and family-aware second. It should keep the context-margin
  block, use soft overlap subspace support, and report mixed families rather
  than forcing a whole selected family to open or close.

## Evidence

- `tests/validation/122_test_overlap_internal_node_bayesian_likelihood_probe.py`
  verifies row-level acceptance versus family status, context blocking, and
  output writing.
- Verification passed:
  `pytest tests/validation/122_test_overlap_internal_node_bayesian_likelihood_probe.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_internal_node_bayesian_likelihood_probe.py tests/validation/122_test_overlap_internal_node_bayesian_likelihood_probe.py`.

## Links

- [[overlap-conditional-bayesian-traversal-law-20260615]]
- [[overlap-bayesian-neighborhood-component-audit-20260615]]
- [[overlap-selected-family-law-requirements-20260614]]
- [[open-mathematical-questions]]
