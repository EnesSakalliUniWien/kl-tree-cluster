---
title: Overlap Context-Negative Topology Conditioning 2026-06-15
type: source
status: reviewed
updated: 2026-07-28
sources:
  - benchmarks/diagnostics/calibration/overlap/overlap_context_negative_topology_conditioning.py
  - tests/validation/calibration/overlap/130_test_overlap_context_negative_topology_conditioning.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_conditioning/overlap_context_negative_topology_conditioning_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_conditioning/overlap_context_negative_topology_conditioning_metric_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_conditioning/overlap_context_negative_topology_conditioning_category_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_conditioning/overlap_context_negative_topology_conditioning_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_conditioning/overlap_context_negative_topology_conditioning_threshold_scan.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_topology_conditioning/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - topology
---

# Overlap Context-Negative Topology Conditioning 2026-06-15

## Summary

`overlap_context_negative_topology_conditioning.py` scans selected
neighborhood and topology variables inside the same
`context_negative_emergent` slice where raw edge-test evidence failed. It joins
incidence-mode rows with branch-incidence topology, row-level transfer-gap
structural evidence, and income/outcome junction context.

## Key Points

- The runner writes topology-conditioning rows, per-metric summaries,
  per-category summaries, threshold scans, a one-row summary, and `manifest.json`.
- The row contract now carries `parent_id`, `topology_support_role`, and
  `topology_signal_role` so downstream conditional topology-law diagnostics can
  use selected-neighborhood bandwidth evidence without inferring support from
  truth labels.
- In the regenerated focused rows, `17` rows are marked `strict_null` support,
  `8` signal-side negatives are marked `selected_nonnull`, and the single
  `truth_recovery` row is marked with `topology_signal_role = signal`.
- `sibling_projection_dimension` is optional on the branch-incidence input. If
  the upstream artifact predates that column, `neighborhood_scale_source` is
  `missing_sibling_projection_dimension` rather than aborting the topology-role
  producer.
- In the focused overlap run, the ambiguous subset again contains `26` rows:
  `1` truth-recovery row and `25` negative rows.
- Seven numeric topology/neighborhood metrics separate the single truth row
  from all negatives in this focused slice. No categorical decision/status
  variable does.
- The best separator is the higher-order topology relation
  `balance_product = incoming_branch_balance * outgoing_balance`. The truth
  row has value `0.232891`; the largest negative value is `0.222500`, so the
  max-negative margin is `0.010391`.
- Pure outgoing split balance also separates, but narrowly: the truth row has
  `outgoing_balance = 0.492891`, while the largest negative value is
  `0.491304`, margin `0.001587`.
- Outgoing edge-norm balance is also a separator in this focused slice
  (`0.971963` truth versus `0.965812` max negative), but it should be read as
  neighborhood symmetry, not raw edge-test significance.
- The diagnostic status is
  `topology_conditioning_single_truth_separator_candidate`, not production
  readiness. There is only one truth-recovery row in the ambiguous slice, and
  the useful thresholds are max-negative cutpoints that require transfer
  validation.
- The follow-up transfer audit preserves `balance_product` as the most
  plausible topology direction, but does not validate a threshold: the only
  positive row cannot appear in held-out truth while also leaving training
  truth support.

## Evidence

- `tests/validation/calibration/overlap/130_test_overlap_context_negative_topology_conditioning.py`
  verifies single-truth separator detection, explicit support/signal-role
  production, selected-nonnull exclusion marking, missing projection-dimension
  tolerance, overlapping metric failure, and output writing.
- Verification passed:
  `pytest tests/validation/calibration/overlap/130_test_overlap_context_negative_topology_conditioning.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap/overlap_context_negative_topology_conditioning.py tests/validation/calibration/overlap/130_test_overlap_context_negative_topology_conditioning.py`.

## Links

- [[overlap-context-negative-edge-conditioning-20260615]]
- [[overlap-context-negative-topology-transfer-20260615]]
- [[overlap-bayesian-incidence-mode-law-20260615]]
- [[overlap-branch-incidence-junction-panel-20260615]]
- [[open-mathematical-questions]]
