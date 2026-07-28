---
title: Overlap Context-Negative Bayesian Topology Law 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_context_negative_bayesian_topology_law.py
  - tests/validation/calibration/overlap/132_test_overlap_context_negative_bayesian_topology_law.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_law/overlap_context_negative_bayesian_topology_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_law/overlap_context_negative_bayesian_topology_component_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_law/overlap_context_negative_bayesian_topology_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_bayesian_topology_law/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - topology
  - bayesian
---

# Overlap Context-Negative Bayesian Topology Law 2026-06-15

## Summary

`overlap_context_negative_bayesian_topology_law.py` turns the
single-positive topology hint into a continuous diagnostic law. It does not
learn a max-negative threshold. Instead, it applies fixed interpretable
component likelihood ratios to selected-neighborhood topology coordinates:
incoming branch balance, outgoing sibling balance, outgoing edge-norm balance,
anti-fragment evidence, selected-family evidence, and a soft penalty for
negative context.

## Key Points

- The runner writes row-level component scores, a component summary, a one-row
  summary, and `manifest.json`.
- The component priors favor coherent selected topology with beta
  signal/background densities on unit-scaled balance variables:
  signal `Beta(6, 2)` versus background `Beta(2, 5)`.
- The focused context-negative emergent slice has `26` rows: `1`
  truth-recovery row and `25` negatives.
- The posterior-style topology score ranks the truth row first, with
  truth posterior log odds `34.101999`, maximum negative `29.147130`, and
  margin `4.954869`.
- The useful components are not raw edge p-values. Outgoing balance and
  outgoing edge-norm balance each rank the truth row first. Incoming balance
  ranks the truth row fourth, anti-fragment evidence second, selected-family
  evidence third, and the soft context penalty sixth.
- This refines the missing conditioning variable to a vector-valued
  income/outcome selected-neighborhood topology term:
  coherent outgoing topology conditioned on selected-family evidence and
  incoming branch context.
- The follow-up sensitivity audit confirms this is not selected-family/context
  confounding: topology-only and outgoing-topology-only profiles separate at
  every tested context penalty weight, while selected-family plus context never
  separates.
- The result remains diagnostic-only. The prior weights are fixed and
  interpretable, not calibrated; the focused slice still has only one
  context-negative truth row, so broader support or a pooled Bayesian model is
  required before any traversal law can be promoted.

## Evidence

- `tests/validation/calibration/overlap/132_test_overlap_context_negative_bayesian_topology_law.py`
  verifies that high topology values increase likelihood, a balanced
  income/outcome truth row ranks first in a synthetic case, missing components
  contribute neutral evidence, and the runner writes all outputs.
- Verification passed:
  `pytest tests/validation/calibration/overlap/132_test_overlap_context_negative_bayesian_topology_law.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_context_negative_bayesian_topology_law.py tests/validation/calibration/overlap/132_test_overlap_context_negative_bayesian_topology_law.py`.

## Links

- [[overlap-context-negative-topology-transfer-20260615]]
- [[overlap-context-negative-bayesian-topology-sensitivity-20260615]]
- [[overlap-context-negative-topology-conditioning-20260615]]
- [[overlap-context-negative-edge-conditioning-20260615]]
- [[overlap-bayesian-incidence-mode-law-20260615]]
- [[open-mathematical-questions]]
