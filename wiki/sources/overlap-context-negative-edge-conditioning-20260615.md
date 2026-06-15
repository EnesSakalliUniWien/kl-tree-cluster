---
title: Overlap Context-Negative Edge Conditioning 2026-06-15
type: source
status: reviewed
updated: 2026-06-15
sources:
  - benchmarks/diagnostics/calibration/overlap_context_negative_edge_conditioning.py
  - tests/validation/129_test_overlap_context_negative_edge_conditioning.py
  - benchmarks/diagnostics/calibration/overlap_branch_incidence_junction_panel.py
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_edge_conditioning/overlap_context_negative_edge_conditioning_rows.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_edge_conditioning/overlap_context_negative_edge_conditioning_metric_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_edge_conditioning/overlap_context_negative_edge_conditioning_summary.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_edge_conditioning/overlap_context_negative_edge_conditioning_threshold_scan.csv
  - raw/assets/benchmark-results/overlap_structural_sibling_replicates_20260614/context_negative_edge_conditioning/manifest.json
tags:
  - source
  - diagnostics
  - overlap
  - traversal
  - edge
---

# Overlap Context-Negative Edge Conditioning 2026-06-15

## Summary

`overlap_context_negative_edge_conditioning.py` answers the immediate
follow-up question for the `context_negative_emergent` mode: whether existing
child-parent edge tests provide an additional conditioning variable that
separates the one missed truth-recovery row from the negative ambiguous rows.
The diagnostic is fail-closed and post-run only.

## Key Points

- The branch-incidence panel now exports edge-test evidence from the existing
  annotation contract: raw and BH child-parent p-values, tested/rejected flags,
  and outgoing edge summaries for the incoming node, incoming sibling, and both
  outgoing children.
- The edge-conditioning runner writes rows, metric summaries, threshold scans,
  a one-row summary, and `manifest.json`.
- In the focused overlap run, the ambiguous context-negative emergent subset
  contains `26` rows: `1` truth-recovery row and `25` negative rows.
- No scanned edge variable separates the truth row with zero negative leakage:
  the summary status is `edge_conditioning_no_zero_negative_separator`.
- The best edge-strength metrics are still informative as ranks but not as
  admissible gates. Outgoing edge negative-log BH evidence has AUC `0.916667`,
  but the truth row has value `177.772205` while negative rows reach the
  p-value floor at `300.0`.
- Boolean edge rejection variables are saturated in this slice: incoming and
  outgoing edge rejection counts are `2` for both the truth row and negatives,
  so the edge tests exist but do not add a usable binary condition here.
- The follow-up selected-neighborhood topology scan finds a more appropriate
  candidate family: incoming/outgoing balance relations, not raw edge-test
  significance.

## Evidence

- `tests/validation/129_test_overlap_context_negative_edge_conditioning.py`
  verifies zero-negative separator detection, overlapping truth/negative
  failure, and output writing.
- Verification passed:
  `pytest tests/validation/129_test_overlap_context_negative_edge_conditioning.py -q`
  and
  `ruff check benchmarks/diagnostics/calibration/overlap_context_negative_edge_conditioning.py tests/validation/129_test_overlap_context_negative_edge_conditioning.py`.

## Links

- [[overlap-bayesian-incidence-mode-law-20260615]]
- [[overlap-branch-incidence-junction-panel-20260615]]
- [[overlap-context-negative-topology-conditioning-20260615]]
- [[open-mathematical-questions]]
